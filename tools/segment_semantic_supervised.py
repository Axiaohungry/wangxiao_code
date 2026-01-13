import os, json, time
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

def ensure_parent(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def read_rgb(path: str) -> np.ndarray:
    if cv2 is None:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
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
    # veg=green, building=orange, road=blue
    ensure_parent(path)
    h, w = ids.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[ids == 0] = (34, 139, 34)
    vis[ids == 1] = (255, 165, 0)
    vis[ids == 2] = (70, 130, 180)
    if cv2 is None:
        from PIL import Image
        Image.fromarray(vis, mode="RGB").save(path)
    else:
        cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo = np.nanpercentile(x, 1)
    hi = np.nanpercentile(x, 99)
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def build_features(rgb: np.ndarray, dsm: np.ndarray | None, hs: np.ndarray | None) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32) / 255.0
    g = rgb[..., 1].astype(np.float32) / 255.0
    b = rgb[..., 2].astype(np.float32) / 255.0
    feats = [r, g, b]

    if dsm is not None:
        feats.append(norm01(dsm))
    if hs is not None:
        feats.append(hs.astype(np.float32))

    return np.stack(feats, axis=-1)  # H W C

def fit_classifier_sklearn(X, y):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            max_iter=300,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=None
        ))
    ])
    clf.fit(X, y)
    return clf

def predict_tiled(clf, feats: np.ndarray, tile: int = 512) -> np.ndarray:
    H, W, C = feats.shape
    out = np.zeros((H, W), dtype=np.uint8)
    for y0 in range(0, H, tile):
        y1 = min(H, y0 + tile)
        for x0 in range(0, W, tile):
            x1 = min(W, x0 + tile)
            f = feats[y0:y1, x0:x1].reshape(-1, C)
            p = clf.predict(f).astype(np.uint8)
            out[y0:y1, x0:x1] = p.reshape(y1 - y0, x1 - x0)
    return out

def main():
    ap = ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--label_mask", required=True, help="uint8 png: 0/1/2 labeled, 255 unlabeled")
    ap.add_argument("--dsm_npy", default="", help="optional float32 HxW")
    ap.add_argument("--hillshade_npy", default="", help="optional float32 HxW in [0,1]")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--out_stats_json", default="")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--max_train_samples", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    rgb = read_rgb(args.rgb)
    H, W = rgb.shape[:2]

    if cv2 is None:
        from PIL import Image
        mask = np.array(Image.open(args.label_mask).convert("L"), dtype=np.uint8)
    else:
        mask = cv2.imread(args.label_mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(args.label_mask)

    if mask.shape != (H, W):
        raise ValueError(f"label_mask shape {mask.shape} != rgb shape {(H, W)}")

    dsm = None
    if args.dsm_npy and Path(args.dsm_npy).exists():
        dsm = np.load(args.dsm_npy).astype(np.float32)
        if dsm.shape != (H, W):
            raise ValueError(f"dsm shape {dsm.shape} != {(H,W)}")

    hs = None
    if args.hillshade_npy and Path(args.hillshade_npy).exists():
        hs = np.load(args.hillshade_npy).astype(np.float32)
        if hs.shape != (H, W):
            raise ValueError(f"hillshade shape {hs.shape} != {(H,W)}")

    feats = build_features(rgb, dsm, hs)  # H W C
    C = feats.shape[-1]

    labeled = (mask != 255)
    n_lab = int(labeled.sum())
    if n_lab <= 0:
        raise RuntimeError("No labeled pixels found in label_mask (all 255).")

    X = feats[labeled].reshape(-1, C)
    y = mask[labeled].reshape(-1).astype(np.int64)

    # subsample for speed
    rng = np.random.default_rng(args.seed)
    if X.shape[0] > args.max_train_samples:
        idx = rng.choice(X.shape[0], size=args.max_train_samples, replace=False)
        X, y = X[idx], y[idx]

    # train
    used_method = "supervised_logreg"
    try:
        clf = fit_classifier_sklearn(X, y)
        yhat = clf.predict(X)
        train_acc = float((yhat == y).mean())
    except Exception as e:
        # minimal fallback: nearest centroid
        used_method = "supervised_centroid"
        train_acc = None
        cent = {}
        for cls in [0, 1, 2]:
            m = (y == cls)
            if m.sum() == 0:
                cent[cls] = None
            else:
                cent[cls] = X[m].mean(axis=0)
        def predict_centroid(feat_flat):
            out = np.zeros((feat_flat.shape[0],), dtype=np.uint8)
            for i in range(feat_flat.shape[0]):
                best_c, best_d = 0, 1e18
                for cls in [0, 1, 2]:
                    if cent[cls] is None:
                        continue
                    d = np.sum((feat_flat[i] - cent[cls]) ** 2)
                    if d < best_d:
                        best_d = d
                        best_c = cls
                out[i] = best_c
            return out
        class Dummy:
            def predict(self, feat_flat):
                return predict_centroid(feat_flat)
        clf = Dummy()

    pred = predict_tiled(clf, feats, tile=int(args.tile))

    # hard-keep labeled pixels
    pred[labeled] = mask[labeled].astype(np.uint8)

    # optional light smoothing (remove salt noise) if cv2 available
    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for cls in [0, 1, 2]:
            m = (pred == cls).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
            pred[m == 255] = cls

    ensure_parent(args.out_png)
    ensure_parent(args.out_npy)
    write_id_png(args.out_png, pred)
    np.save(args.out_npy, pred)

    vis_path = str(Path(args.out_png).with_name("semantic_map_vis.png"))
    write_vis_png(vis_path, pred)

    # stats
    hist = { "vegetation": int((pred == 0).sum()),
             "building":   int((pred == 1).sum()),
             "road":       int((pred == 2).sum()) }
    total = int(pred.size)
    ratio = {k: float(v) / float(total) for k, v in hist.items()}

    stats = {
        "time_sec": round(time.time() - t0, 3),
        "used_method": used_method,
        "rgb": os.path.abspath(args.rgb),
        "label_mask": os.path.abspath(args.label_mask),
        "dsm_npy": os.path.abspath(args.dsm_npy) if args.dsm_npy else "",
        "hillshade_npy": os.path.abspath(args.hillshade_npy) if args.hillshade_npy else "",
        "H": int(H), "W": int(W),
        "feature_dim": int(C),
        "labeled_pixels_total": int(n_lab),
        "train_acc_on_labeled": train_acc,
        "ratio": ratio,
        "hist": hist,
        "notes": [
            "label_mask uses 0/1/2 for classes and 255 for unlabeled.",
            "Predictions keep labeled pixels fixed.",
            "Using DSM/Hillshade features helps separate roof vs road when RGB is similar."
        ]
    }

    if args.out_stats_json.strip():
        stats_path = Path(args.out_stats_json)
    else:
        stats_path = Path(args.out_png).parent / "semantic_stats.json"  # keep same name for check_manifest
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Wrote:")
    print(" ", args.out_png)
    print(" ", args.out_npy)
    print(" ", vis_path)
    print(" ", str(stats_path))
    print("[SEMANTIC_SUP] used_method=", used_method, " train_acc=", train_acc, " ratio=", ratio)

if __name__ == "__main__":
    main()
