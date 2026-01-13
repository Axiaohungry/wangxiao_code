# tools/segment_semantic.py
# Output semantic_map with class IDs:
#   0 = vegetation
#   1 = building
#   2 = road
#
# Priority: SegFormer local checkpoint -> fallback CV 3-class.
# Designed for GTX1660 6GB: tiled inference + auto fallback.

import os
import json
import time
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


CLASS_NAMES = {0: "vegetation", 1: "building", 2: "road"}


def ensure_parent(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def read_rgb(path: str) -> np.ndarray:
    if cv2 is None:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        return np.array(im, dtype=np.uint8)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_id_png(path: str, ids: np.ndarray):
    ensure_parent(path)
    ids_u8 = ids.astype(np.uint8, copy=False)
    if cv2 is None:
        from PIL import Image
        Image.fromarray(ids_u8, mode="L").save(path)
    else:
        cv2.imwrite(path, ids_u8)


def write_vis_png(path: str, ids: np.ndarray):
    # Fixed palette for easy inspection (not used by training)
    # veg=green, building=orange-ish, road=blue-ish
    ensure_parent(path)
    h, w = ids.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[ids == 0] = (34, 139, 34)    # vegetation
    vis[ids == 1] = (255, 165, 0)    # building
    vis[ids == 2] = (70, 130, 180)   # road
    if cv2 is None:
        from PIL import Image
        Image.fromarray(vis, mode="RGB").save(path)
    else:
        cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def semantic_cv_fallback(rgb: np.ndarray, seed: int = 0, sample_n: int = 200000):
    """
    Fast CV 3-class segmentation:
    - KMeans in HSV on sampled pixels
    - Assign clusters -> veg/building/road by heuristics
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not available; cannot run CV fallback.")

    H, W = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    flat = hsv.reshape(-1, 3).astype(np.float32)
    n = flat.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_n, n), replace=False)
    sample = flat[idx]

    # KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    K = 3
    _compact, labels_s, centers = cv2.kmeans(
        sample, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.float32)  # HSV centers

    # Assign every pixel to nearest center (vectorized)
    # distance in HSV space
    d0 = np.sum((flat - centers[0]) ** 2, axis=1)
    d1 = np.sum((flat - centers[1]) ** 2, axis=1)
    d2 = np.sum((flat - centers[2]) ** 2, axis=1)
    lab = np.argmin(np.stack([d0, d1, d2], axis=1), axis=1).astype(np.int32)
    lab = lab.reshape(H, W)

    # Heuristic labeling of clusters:
    # - vegetation: center whose HSV converts to the "greenest" BGR (highest ExG)
    # - road: low saturation cluster among remaining (and relatively bright)
    # - building: the remaining cluster
    centers_u8 = np.clip(centers, 0, 255).astype(np.uint8).reshape(1, K, 3)
    bgr_centers = cv2.cvtColor(centers_u8, cv2.COLOR_HSV2BGR).reshape(K, 3).astype(np.float32)
    # ExG = 2G - R - B  (note BGR)
    exg = 2 * bgr_centers[:, 1] - bgr_centers[:, 2] - bgr_centers[:, 0]
    veg_k = int(np.argmax(exg))

    remaining = [k for k in range(K) if k != veg_k]
    sat = centers[:, 1]
    val = centers[:, 2]
    # road tends to have low saturation; if tie, higher value
    rem_sat = [(k, float(sat[k]), float(val[k])) for k in remaining]
    rem_sat.sort(key=lambda t: (t[1], -t[2]))
    road_k = int(rem_sat[0][0])
    building_k = int([k for k in range(K) if k not in (veg_k, road_k)][0])

    mapping = {veg_k: 0, building_k: 1, road_k: 2}
    out = np.vectorize(mapping.get)(lab).astype(np.uint8)

    return out, {
        "clusters_hsv": centers.tolist(),
        "cluster_assignment": {"veg_cluster": veg_k, "building_cluster": building_k, "road_cluster": road_k},
        "method_detail": "kmeans_hsv_3c"
    }


def build_3class_logits(logits, id2label):
    """
    If model has >3 classes, try map to 3 groups by label keywords.
    Return 3-class logits tensor.
    """
    import torch

    num_labels = logits.shape[1]
    if num_labels == 3:
        return logits, {"mapping": "native_3"}

    # keyword sets (best-effort)
    veg_kw = ["tree", "grass", "plant", "vegetation", "forest", "bush"]
    bld_kw = ["building", "house", "skyscraper", "wall", "roof", "tower"]
    road_kw = ["road", "street", "highway", "sidewalk", "path", "runway"]

    def pick_ids(kws):
        ids = []
        for i in range(num_labels):
            name = str(id2label.get(i, "")).lower()
            if any(k in name for k in kws):
                ids.append(i)
        return ids

    veg_ids = pick_ids(veg_kw)
    bld_ids = pick_ids(bld_kw)
    road_ids = pick_ids(road_kw)

    # fallback: if mapping too weak, use top-3 most frequent classes not possible here; return as-is
    if len(veg_ids) == 0 or len(bld_ids) == 0 or len(road_ids) == 0:
        # Cannot reliably map; treat as failure and force fallback
        return None, {"mapping": "failed_keywords", "veg_ids": veg_ids, "bld_ids": bld_ids, "road_ids": road_ids}

    def max_pool(ids):
        x = logits[:, ids, :, :]
        return torch.amax(x, dim=1, keepdim=True)

    l3 = torch.cat([max_pool(veg_ids), max_pool(bld_ids), max_pool(road_ids)], dim=1)
    return l3, {"mapping": "keyword_group", "veg_ids": veg_ids, "bld_ids": bld_ids, "road_ids": road_ids}


def semantic_segformer(rgb: np.ndarray, ckpt_dir: str, tile: int, device: str):
    """
    Tiled SegFormer inference.
    Returns (semantic_ids_u8, extra_info_dict)
    """
    import torch

    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    except Exception as e:
        raise RuntimeError(f"transformers not available: {e}")

    ckpt = Path(ckpt_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"segformer_ckpt not found: {ckpt_dir}")

    # Device
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        device = "cpu"

    processor = SegformerImageProcessor.from_pretrained(str(ckpt), local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(str(ckpt), local_files_only=True)
    model.eval()

    torch_device = torch.device(device)
    model.to(torch_device)

    id2label = getattr(model.config, "id2label", {}) or {}

    H, W = rgb.shape[:2]
    t = int(tile)
    overlap = max(32, t // 8)
    stride = t - overlap

    # Accumulate 3-class probs for smooth stitching
    acc = np.zeros((3, H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    def run_tile(tile_rgb: np.ndarray):
        inputs = processor(images=tile_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(torch_device)

        with torch.no_grad():
            out = model(pixel_values=pixel_values)
            logits = out.logits  # [1, C, h, w] (often downsampled)
            # upsample to tile size
            logits = torch.nn.functional.interpolate(
                logits, size=tile_rgb.shape[:2], mode="bilinear", align_corners=False
            )

            l3, map_info = build_3class_logits(logits, id2label)
            if l3 is None:
                raise RuntimeError(f"Cannot map model labels to 3 classes: {map_info}")

            probs = torch.softmax(l3, dim=1)[0].detach().float().cpu().numpy()  # [3, th, tw]
            return probs, map_info

    # Iterate tiles
    map_info_final = None
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1 = min(y0 + t, H)
            x1 = min(x0 + t, W)
            tile_rgb = rgb[y0:y1, x0:x1]

            # Optional pad to tile size (processor is fine without padding, but keep consistent)
            try:
                probs, map_info = run_tile(tile_rgb)
                map_info_final = map_info
            except RuntimeError as e:
                # Auto-handle CUDA OOM by shrinking tile or switching CPU
                msg = str(e).lower()
                if "out of memory" in msg or "cuda" in msg:
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    raise
                raise

            acc[:, y0:y1, x0:x1] += probs[:, : (y1 - y0), : (x1 - x0)]
            cnt[y0:y1, x0:x1] += 1.0

    cnt = np.maximum(cnt, 1e-6)
    probs_full = acc / cnt[None, :, :]
    ids = np.argmax(probs_full, axis=0).astype(np.uint8)

    return ids, {
        "used_device": str(torch_device),
        "tile": t,
        "overlap": overlap,
        "stride": stride,
        "label_mapping": map_info_final if map_info_final is not None else {}
    }


def main():
    ap = ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--method", default="segformer", choices=["segformer", "cv", "auto"])
    ap.add_argument("--segformer_ckpt", default="weights\\segformer_b0_local")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fallback_enable", type=int, default=1)
    ap.add_argument("--out_stats_json", default="", help="default: artifacts/semantic_stats.json")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    t0 = time.time()
    rgb = read_rgb(args.rgb)
    H, W = rgb.shape[:2]

    used_method = None
    extra = {}
    ids = None

    # Decide path
    def try_segformer(tile):
        return semantic_segformer(rgb, args.segformer_ckpt, tile=tile, device=args.device)

    if args.method in ("segformer", "auto"):
        try:
            # Try requested tile; if OOM, reduce then retry; if still fails, fallback
            tile_try = int(args.tile)
            for tile_try in [tile_try, 384, 256]:
                try:
                    ids, extra = try_segformer(tile_try)
                    used_method = "segformer"
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if ("out of memory" in msg) or ("cuda" in msg):
                        continue
                    # non-OOM error: break to fallback if enabled
                    extra = {"segformer_error": str(e)}
                    break

            if used_method != "segformer":
                raise RuntimeError(extra.get("segformer_error", "segformer failed"))
        except Exception as e:
            if int(args.fallback_enable) != 1 and args.method == "segformer":
                raise
            # Fallback CV
            ids, extra2 = semantic_cv_fallback(rgb, seed=int(args.seed))
            used_method = "cv_fallback"
            extra.update({"fallback_reason": str(e), **extra2})

    if args.method == "cv":
        ids, extra = semantic_cv_fallback(rgb, seed=int(args.seed))
        used_method = "cv_fallback"

    ids = ids.astype(np.uint8, copy=False)

    # Write outputs
    ensure_parent(args.out_png)
    ensure_parent(args.out_npy)
    write_id_png(args.out_png, ids)
    np.save(args.out_npy, ids)

    # Always write a colored preview for inspection
    vis_path = str(Path(args.out_png).with_name("semantic_map_vis.png"))
    write_vis_png(vis_path, ids)

    # Stats
    hist = {CLASS_NAMES[i]: int((ids == i).sum()) for i in (0, 1, 2)}
    total = int(ids.size)
    ratio = {k: float(v) / float(total) for k, v in hist.items()}

    stats = {
        "time_sec": round(time.time() - t0, 3),
        "rgb": os.path.abspath(args.rgb),
        "H": int(H),
        "W": int(W),
        "used_method": used_method,
        "class_id_map": {"0": "vegetation", "1": "building", "2": "road"},
        "hist": hist,
        "ratio": ratio,
        "extra": extra,
        "notes": [
            "semantic_map.png stores class IDs (0/1/2). Use semantic_map_vis.png for human inspection.",
            "If segformer weights are missing or incompatible, cv_fallback is used when fallback_enable=1."
        ],
    }

    if args.out_stats_json.strip():
        stats_path = Path(args.out_stats_json)
    else:
        stats_path = Path(args.out_png).parent / "semantic_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Wrote:")
    print(" ", args.out_png)
    print(" ", args.out_npy)
    print(" ", vis_path)
    print(" ", str(stats_path))
    print("[SEMANTIC] used_method=", used_method, " ratio=", ratio)


if __name__ == "__main__":
    main()
