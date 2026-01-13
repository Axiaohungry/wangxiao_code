# tools/fuse_priors_v2.py
# Phase4.3: fuse priors + (optional) shadow_map + semantic_map -> priors_v2
# Supports two modes:
#   A) priors_v1 (Nx5) + shadow_npy (HxW) + semantic_map (HxW) -> priors_v2 (Nx7)
#   B) priors_in (NxD, already has shadow) + semantic_map (HxW) -> overwrite semantic col, output priors_out (NxD)
#
# Key fix: explicit NDC->pixel Y convention switch (ndc_y_mode), plus semantic/shadow map transforms.

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import cv2

from scene.gaussian_model import GaussianModel


# -------------------------
# Utils
# -------------------------
def safe_load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_tensor_from_pt(path: Path) -> torch.Tensor:
    obj = safe_torch_load(str(path), map_location="cpu")
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for k in ["priors", "priors_pt", "data", "features", "tensor"]:
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
        for v in obj.values():
            if torch.is_tensor(v):
                return v
    raise ValueError(f"Cannot find tensor in {path}")


def apply_2d_transform(arr: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "none":
        return arr
    if mode == "flipud":
        return np.flipud(arr)
    if mode == "fliplr":
        return np.fliplr(arr)
    if mode == "rot180":
        return np.flipud(np.fliplr(arr))
    raise ValueError(f"Unknown 2D transform mode: {mode}")


def write_gray_debug(path: Path, img_f: np.ndarray, p_lo=1.0, p_hi=99.5, gamma=0.6) -> None:
    """
    Robust contrast stretch for debugging. img_f can be float or uint8.
    """
    x = img_f.astype(np.float32)
    m = np.isfinite(x)
    if m.any():
        v = x[m]
        lo = np.percentile(v, p_lo)
        hi = np.percentile(v, p_hi)
        x = (x - lo) / max(hi - lo, 1e-6)
        x = np.clip(x, 0, 1)
        x = np.power(x, gamma)
    x_u8 = (x * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), x_u8)


def write_semantic_debug(path: Path, sem_u8: np.ndarray) -> None:
    """
    sem_u8: HxW, values 0/1/2. Writes visible grayscale (0/127/255).
    """
    vis = (sem_u8.astype(np.uint8) * 127).clip(0, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


# -------------------------
# Camera math (matches your render/train scripts)
# -------------------------
def look_at_topdown(cam_pos: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    z_axis = target - cam_pos
    dist = torch.norm(z_axis)
    if dist < 1e-6:
        return torch.eye(4, dtype=torch.float32)

    z_axis = z_axis / dist
    x_axis = torch.cross(up, z_axis, dim=0)
    if torch.norm(x_axis) < 1e-6:
        tmp = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        x_axis = torch.cross(tmp, z_axis, dim=0)
    x_axis = x_axis / (torch.norm(x_axis) + 1e-8)

    y_axis = torch.cross(z_axis, x_axis, dim=0)
    y_axis = y_axis / (torch.norm(y_axis) + 1e-8)

    R = torch.stack([x_axis, y_axis, z_axis], dim=0)  # 3x3
    T = -torch.matmul(R, cam_pos)

    w2v = torch.eye(4, dtype=torch.float32)
    w2v[:3, :3] = R
    w2v[:3, 3] = T
    return w2v.transpose(0, 1).contiguous()


def get_projection_matrix(znear: float, zfar: float, fovX: float, fovY: float) -> torch.Tensor:
    tanHalfFovY = np.tan(fovY / 2.0)
    tanHalfFovX = np.tan(fovX / 2.0)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, dtype=torch.float32)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def try_read_dsm_stats(artifacts_dir: Path) -> Optional[Dict[str, Any]]:
    p = artifacts_dir / "dsm_float_stats.json"
    if p.exists():
        return safe_load_json(p)
    return None


def compute_topdown_fullT(
    xyz: torch.Tensor,
    cams_data: List[Dict[str, Any]],
    W: int,
    H: int,
    zoom: float,
    shift_x: float,
    shift_y: float,
    angle_deg: float,
    multiplier: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    center = xyz.mean(dim=0)
    min_xyz, _ = torch.min(xyz, dim=0)
    max_xyz, _ = torch.max(xyz, dim=0)
    span_x = (max_xyz[0] - min_xyz[0]).item()
    span_y = (max_xyz[1] - min_xyz[1]).item()
    max_span = max(span_x, span_y)

    ref = cams_data[0]
    fov_y = 2.0 * np.arctan(ref["height"] / (2.0 * ref["fy"]))
    fov_x_mod = 2.0 * np.arctan(np.tan(fov_y / 2.0) * multiplier)

    # PCA (CPU)
    xyz_centered = xyz - xyz.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    _, eigvecs = torch.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    axis1 = eigvecs[:, 1]

    cam_centers = [np.array(c["position"], dtype=np.float32) for c in cams_data[:10]]
    mean_cam_pos = torch.tensor(np.mean(cam_centers, axis=0), dtype=torch.float32)
    if torch.dot((mean_cam_pos - center), normal) < 0:
        normal = -normal

    up_axis = axis1 - torch.dot(axis1, normal) * normal
    up_axis = up_axis / (torch.norm(up_axis) + 1e-8)
    right_axis = torch.cross(up_axis, normal, dim=0)

    if abs(angle_deg) > 1e-6:
        rad = np.deg2rad(angle_deg)
        cos_a = float(np.cos(rad))
        sin_a = float(np.sin(rad))
        new_up = up_axis * cos_a + right_axis * sin_a
        up_axis = new_up / (torch.norm(up_axis) + 1e-8)
        right_axis = torch.cross(up_axis, normal, dim=0)

    base_height = (max_span / 2.0) / np.tan(fov_y / 2.0)
    target_height = base_height / zoom

    shift_vec = (right_axis * shift_x) + (up_axis * shift_y)
    target_center = center + shift_vec
    cam_pos = target_center + normal * target_height

    w2v = look_at_topdown(cam_pos, target_center, up_axis)
    proj = get_projection_matrix(0.01, 100.0, fov_x_mod, fov_y).transpose(0, 1).contiguous()
    full_T = w2v @ proj

    info = {
        "fov_y": float(fov_y),
        "fov_x_mod": float(fov_x_mod),
        "span_x": float(span_x),
        "span_y": float(span_y),
        "max_span": float(max_span),
        "zoom": float(zoom),
        "shift_x": float(shift_x),
        "shift_y": float(shift_y),
        "angle_deg": float(angle_deg),
        "multiplier": float(multiplier),
        "W": float(W),
        "H": float(H),
    }
    return full_T, info


# -------------------------
# Load semantic map
# -------------------------
def load_semantic_id_map(path: Path) -> np.ndarray:
    """
    Accepts:
      - .npy (recommended): HxW uint8 ids {0,1,2}
      - .png (single channel): HxW
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
        if arr.ndim != 2:
            raise ValueError(f"semantic npy must be HxW, got {arr.shape}")
        return arr

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(str(path))
    if img.ndim == 3:
        # if user gave a visualization RGB, refuse loudly
        raise ValueError(
            f"semantic_png must be a single-channel ID map or semantic_map.npy. Got 3-channel image: {path}"
        )
    return img


# -------------------------
# Project & sample
# -------------------------
def project_and_sample(
    xyz_np: np.ndarray,
    full_T_np: np.ndarray,
    H: int,
    W: int,
    ndc_y_mode: str,
    shadow_map: Optional[np.ndarray],
    semantic_map: np.ndarray,
    chunk: int = 400_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    ndc_y_mode:
      - y_down: y = (0.5 - ndc_y*0.5)*(H-1)   (top-left origin)
      - y_up:   y = (ndc_y*0.5 + 0.5)*(H-1)  (bottom-left origin)
    """
    ndc_y_mode = ndc_y_mode.lower()
    if ndc_y_mode not in ["y_down", "y_up"]:
        raise ValueError(f"ndc_y_mode must be y_down or y_up, got {ndc_y_mode}")

    N = xyz_np.shape[0]
    shadow_vals = np.empty((N,), dtype=np.float32)
    sem_ids = np.empty((N,), dtype=np.uint8)
    xs_all = np.empty((N,), dtype=np.int32)
    ys_all = np.empty((N,), dtype=np.int32)

    in_view_count = 0

    full_T_np = full_T_np.astype(np.float32)
    sem_map_u = semantic_map.astype(np.uint8)

    has_shadow = shadow_map is not None
    if has_shadow:
        sh_map_f = shadow_map.astype(np.float32)

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        pts = xyz_np[s:e].astype(np.float32)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        hom = np.concatenate([pts, ones], axis=1)

        clip = hom @ full_T_np
        w = clip[:, 3:4]
        w = np.where(np.abs(w) < 1e-8, 1e-8, w)
        ndc = clip[:, 0:3] / w

        x = (ndc[:, 0] * 0.5 + 0.5) * (W - 1)

        if ndc_y_mode == "y_down":
            y = (0.5 - ndc[:, 1] * 0.5) * (H - 1)
        else:
            y = (ndc[:, 1] * 0.5 + 0.5) * (H - 1)

        in_view = (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
        in_view_count += int(in_view.sum())

        xi = np.clip(np.round(x).astype(np.int32), 0, W - 1)
        yi = np.clip(np.round(y).astype(np.int32), 0, H - 1)

        xs_all[s:e] = xi
        ys_all[s:e] = yi

        se = sem_map_u[yi, xi]
        se = np.where(in_view, se, 0).astype(np.uint8)
        sem_ids[s:e] = se

        if has_shadow:
            sh = sh_map_f[yi, xi]
            sh = np.where(in_view, sh, 0.5).astype(np.float32)
            shadow_vals[s:e] = sh
        else:
            shadow_vals[s:e] = 1.0  # neutral

    in_view_ratio = float(in_view_count) / float(N)
    return shadow_vals, sem_ids, xs_all, ys_all, in_view_ratio


def make_topdown_vis(
    H: int,
    W: int,
    xs: np.ndarray,
    ys: np.ndarray,
    sem: np.ndarray,
    intensity: np.ndarray,
    out_png: Path,
    max_points: int = 1_200_000,
) -> None:
    """
    Sparse splat: color by semantic, intensity by provided scalar (shadow or 1).
    """
    palette = {
        0: np.array([0, 200, 0], dtype=np.float32),      # vegetation
        1: np.array([0, 0, 220], dtype=np.float32),      # building
        2: np.array([220, 220, 220], dtype=np.float32)   # road
    }
    vis = np.zeros((H, W, 3), dtype=np.float32)

    N = xs.shape[0]
    keep = min(N, int(max_points))
    if keep < N:
        idx = np.linspace(0, N - 1, keep, dtype=np.int64)
        xs2, ys2, sem2, it2 = xs[idx], ys[idx], sem[idx], intensity[idx]
    else:
        xs2, ys2, sem2, it2 = xs, ys, sem, intensity

    colors = np.zeros((xs2.shape[0], 3), dtype=np.float32)
    for k, c in palette.items():
        m = (sem2 == k)
        if np.any(m):
            colors[m] = c

    it2 = it2.astype(np.float32).clip(0.0, 1.0)
    colors = colors * it2[:, None]
    vis[ys2, xs2, :] = colors

    vis_u8 = np.clip(vis, 0, 255).astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), vis_u8)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True)

    # Mode A: build v2 from priors_v1
    ap.add_argument("--priors_v1", default="", help="Nx5 tensor. If provided and priors_in empty -> Mode A.")
    ap.add_argument("--shadow_npy", default="", help="HxW float32 shadow map. Required in Mode A.")
    ap.add_argument("--semantic_map", required=True, help="semantic_map.npy or single-channel semantic_map.png (HxW ids)")

    # Mode B: overwrite semantic in existing priors_in (recommended for your case)
    ap.add_argument("--priors_in", default="", help="Existing priors tensor NxD. If provided -> Mode B.")
    ap.add_argument("--semantic_col", type=int, default=-1, help="Which column in priors_in to overwrite with semantic scalar (default last).")
    ap.add_argument("--shadow_col_for_vis", type=int, default=5, help="Which column in priors_in to use as intensity for visualization when shadow_npy not provided.")

    # Output
    ap.add_argument("--out_priors_v2", required=True)
    ap.add_argument("--out_vis_png", required=True)
    ap.add_argument("--out_dir_debug", default="", help="If set, write debug images here (semantic/shadow).")

    # Camera params
    ap.add_argument("--zoom", type=float, default=5.4)
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--shift_y", type=float, default=-1.2)
    ap.add_argument("--angle", type=float, default=-31.0)
    ap.add_argument("--multiplier", type=float, default=0.85)

    # Fix knobs
    ap.add_argument("--ndc_y_mode", default="y_down", choices=["y_down", "y_up"],
                    help="Switch to y_up if your result is vertically flipped vs topdown_final.")
    ap.add_argument("--semantic_transform", default="none", choices=["none", "flipud", "fliplr", "rot180"],
                    help="Transform semantic_map before sampling (use flipud if ONLY semantic looks upside-down).")
    ap.add_argument("--shadow_transform", default="none", choices=["none", "flipud", "fliplr", "rot180"],
                    help="Transform shadow_map before sampling (rare; only if shadow_map has opposite orientation).")

    ap.add_argument("--chunk", type=int, default=400000)
    args = ap.parse_args()

    t0 = time.time()

    ply_path = Path(args.ply)
    cam_path = Path(args.cameras)
    sem_path = Path(args.semantic_map)

    out_pt = Path(args.out_priors_v2)
    out_vis = Path(args.out_vis_png)

    for p in [ply_path, cam_path, sem_path]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    priors_in_path = Path(args.priors_in) if args.priors_in else None
    priors_v1_path = Path(args.priors_v1) if args.priors_v1 else None
    shadow_path = Path(args.shadow_npy) if args.shadow_npy else None

    mode_b = priors_in_path is not None and str(priors_in_path) != ""
    mode_a = (not mode_b) and (priors_v1_path is not None and str(priors_v1_path) != "")

    if not (mode_a or mode_b):
        raise ValueError("You must provide either --priors_in (Mode B) or --priors_v1 + --shadow_npy (Mode A).")

    if mode_a and (shadow_path is None or str(shadow_path) == "" or not shadow_path.exists()):
        raise ValueError("Mode A requires --shadow_npy pointing to an existing .npy file.")

    cams_data = safe_load_json(cam_path)
    if not isinstance(cams_data, list) or len(cams_data) < 1:
        raise ValueError("cameras.json is not a non-empty list")

    # Load semantic map
    semantic_map = load_semantic_id_map(sem_path).astype(np.uint8)
    semantic_map = apply_2d_transform(semantic_map, args.semantic_transform)
    H, W = semantic_map.shape

    # Optional: load shadow map (for Mode A sampling, or just for debug)
    shadow_map = None
    if mode_a:
        shadow_map = np.load(str(shadow_path)).astype(np.float32)
        if shadow_map.ndim != 2:
            raise ValueError(f"shadow_map must be HxW, got {shadow_map.shape}")
        shadow_map = apply_2d_transform(shadow_map, args.shadow_transform)
        if shadow_map.shape != (H, W):
            raise ValueError(f"shadow_map shape {shadow_map.shape} != semantic_map shape {(H, W)}")

    # Debug dumps
    if args.out_dir_debug:
        dbg = Path(args.out_dir_debug)
        write_semantic_debug(dbg / "semantic_map_debug.png", semantic_map)
        if shadow_map is not None:
            write_gray_debug(dbg / "shadow_map_debug.png", shadow_map)

    # Load Gaussians xyz (alignment source of truth)
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(str(ply_path))
    xyz = gaussians.get_xyz.detach().cpu().float()
    xyz_np = xyz.numpy()

    # Determine camera params: prefer DSM stats if shadow_npy exists (compatible with your pipeline)
    zoom = args.zoom
    shift_x = args.shift_x
    shift_y = args.shift_y
    angle = args.angle
    multiplier = args.multiplier

    if shadow_path is not None and shadow_path.exists():
        dsm_stats = try_read_dsm_stats(shadow_path.parent)
        if dsm_stats and "camera" in dsm_stats:
            cam = dsm_stats["camera"]
            zoom = float(cam.get("zoom", zoom))
            shift_x = float(cam.get("shift_x", shift_x))
            shift_y = float(cam.get("shift_y", shift_y))
            angle = float(cam.get("angle", angle))
            multiplier = float(cam.get("multiplier", multiplier))

    full_T, cam_info = compute_topdown_fullT(
        xyz=xyz,
        cams_data=cams_data,
        W=W,
        H=H,
        zoom=zoom,
        shift_x=shift_x,
        shift_y=shift_y,
        angle_deg=angle,
        multiplier=multiplier,
    )
    full_T_np = full_T.numpy()

    # Project & sample maps -> per-point
    sampled_shadow_vals, sem_ids, xs, ys, in_view_ratio = project_and_sample(
        xyz_np=xyz_np,
        full_T_np=full_T_np,
        H=H,
        W=W,
        ndc_y_mode=args.ndc_y_mode,
        shadow_map=shadow_map,
        semantic_map=semantic_map,
        chunk=int(args.chunk),
    )

    # semantic scaling: id/2 -> [0,1]
    sem_scaled = (sem_ids.astype(np.float32) / 2.0).astype(np.float32)

    # Build output priors
    if mode_a:
        pri_v1 = load_tensor_from_pt(priors_v1_path).float()
        if pri_v1.ndim != 2 or pri_v1.shape[1] != 5:
            raise ValueError(f"priors_v1 must be Nx5, got {tuple(pri_v1.shape)}")
        if pri_v1.shape[0] != xyz.shape[0]:
            raise ValueError(f"xyz count mismatch: xyz={xyz.shape[0]} vs priors_v1={pri_v1.shape[0]}")

        shadow_t = torch.from_numpy(sampled_shadow_vals).unsqueeze(1)
        sem_t = torch.from_numpy(sem_scaled).unsqueeze(1)
        pri_v2 = torch.cat([pri_v1.float(), shadow_t, sem_t], dim=1).contiguous()

        intensity_for_vis = sampled_shadow_vals  # sampled from map

    else:
        pri_in = load_tensor_from_pt(priors_in_path).float()
        if pri_in.ndim != 2:
            raise ValueError(f"priors_in must be 2D tensor, got {tuple(pri_in.shape)}")
        if pri_in.shape[0] != xyz.shape[0]:
            raise ValueError(f"xyz count mismatch: xyz={xyz.shape[0]} vs priors_in={pri_in.shape[0]}")

        N, D = pri_in.shape
        si = args.semantic_col if args.semantic_col >= 0 else (D + args.semantic_col)
        if not (0 <= si < D):
            raise ValueError(f"semantic_col out of range: semantic_col={args.semantic_col} resolved={si} D={D}")

        pri_v2 = pri_in.clone()
        pri_v2[:, si] = torch.from_numpy(sem_scaled)

        # visualization intensity: prefer shadow_npy if provided; else use a column in priors_in
        if shadow_map is not None:
            intensity_for_vis = sampled_shadow_vals
        else:
            scol = int(args.shadow_col_for_vis)
            if not (0 <= scol < D):
                scol = min(5, D - 1)
            intensity_for_vis = pri_v2[:, scol].detach().cpu().numpy().astype(np.float32)
            intensity_for_vis = np.clip(intensity_for_vis, 0.0, 1.0)

    # Save priors
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pri_v2, str(out_pt))

    # Vis
    make_topdown_vis(H, W, xs, ys, sem_ids, intensity_for_vis, out_vis)

    # Stats
    n_samp = min(int(pri_v2.shape[0]), 200000)
    sem_hist = np.bincount(sem_ids[:n_samp].astype(np.int64), minlength=3).tolist()

    finite_ratio = float(torch.isfinite(pri_v2).all(dim=1).float().mean().item())
    stats = {
        "time_sec": round(time.time() - t0, 3),
        "mode": "B(priors_in overwrite semantic)" if mode_b else "A(priors_v1 fuse)",
        "N": int(pri_v2.shape[0]),
        "D_out": int(pri_v2.shape[1]),
        "H": int(H),
        "W": int(W),
        "ply": str(ply_path),
        "cameras": str(cam_path),
        "priors_in": str(priors_in_path) if mode_b else "",
        "priors_v1": str(priors_v1_path) if mode_a else "",
        "shadow_npy": str(shadow_path) if mode_a else str(shadow_path) if (shadow_path and shadow_path.exists()) else "",
        "semantic_map": str(sem_path),
        "out_priors_v2": str(out_pt),
        "out_vis_png": str(out_vis),
        "finite_ratio": finite_ratio,
        "in_view_ratio": float(in_view_ratio),
        "ndc_y_mode": args.ndc_y_mode,
        "semantic_transform": args.semantic_transform,
        "shadow_transform": args.shadow_transform,
        "semantic_hist_sample200k": sem_hist,
        "camera_info": cam_info,
        "notes": [
            "semantic stored as scalar in [0,1] via id/2; decode with round(x*2).",
            "If vertically flipped vs topdown_final, switch --ndc_y_mode y_up.",
            "If ONLY semantic is upside-down, keep ndc_y_mode and set --semantic_transform flipud.",
        ],
    }
    stats_path = out_pt.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Wrote:")
    print(f"  {out_pt}")
    print(f"  {out_vis}")
    print(f"  {stats_path}")
    print(f"[STATS] finite_ratio={finite_ratio:.4f} in_view_ratio={in_view_ratio:.4f} sem_hist(sample)={sem_hist}")
    print(f"[FIX] ndc_y_mode={args.ndc_y_mode} semantic_transform={args.semantic_transform}")


if __name__ == "__main__":
    main()
