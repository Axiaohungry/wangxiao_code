import argparse, json, math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../wangxiao_code
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import cv2

from scene.gaussian_model import GaussianModel


def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_tensor_from_pt(path: str) -> torch.Tensor:
    obj = safe_torch_load(path, map_location="cpu")
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


def load_hw_from(path: str) -> Tuple[int, int]:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
        if arr.ndim != 2:
            raise ValueError(f"ref_hw_from npy must be HxW, got {arr.shape}")
        H, W = arr.shape
        return int(H), int(W)
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(str(p))
    H, W = img.shape[:2]
    return int(H), int(W)


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

    R = torch.stack([x_axis, y_axis, z_axis], dim=0)
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


def compute_topdown_fullT(
    xyz: torch.Tensor,
    cams_data: List[Dict[str, Any]],
    zoom: float,
    shift_x: float,
    shift_y: float,
    angle_deg: float,
    multiplier: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    center = xyz.mean(dim=0)
    min_xyz, _ = torch.min(xyz, dim=0)
    max_xyz, _ = torch.max(xyz, dim=0)
    span_x = (max_xyz[0] - min_xyz[0]).item()
    span_y = (max_xyz[1] - min_xyz[1]).item()
    max_span = max(span_x, span_y)

    ref = cams_data[0]
    fov_y = 2.0 * np.arctan(ref["height"] / (2.0 * ref["fy"]))
    fov_x_mod = 2.0 * np.arctan(np.tan(fov_y / 2.0) * multiplier)

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
    full_T = (w2v @ proj).numpy().astype(np.float32)

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
    }
    return full_T, info


def percentile_stretch_gray(values_u8: np.ndarray, mask: np.ndarray, p_lo=1.0, p_hi=99.5, gamma=0.6) -> np.ndarray:
    x = values_u8.astype(np.float32) / 255.0
    m = mask.astype(bool)
    if m.any():
        v = x[m]
        lo = np.percentile(v, p_lo)
        hi = np.percentile(v, p_hi)
        x = (x - lo) / max(hi - lo, 1e-6)
        x = np.clip(x, 0, 1)
        x = np.power(x, gamma)
    return (x * 255.0).astype(np.uint8)


def dilate_u8(img: np.ndarray, k: int = 3, it: int = 1) -> np.ndarray:
    ker = np.ones((k, k), np.uint8)
    return cv2.dilate(img, ker, iterations=it)


def save_gray_and_enh(out_base: Path, img_u8: np.ndarray) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_base), img_u8)

    mask = img_u8 > 0
    enh = percentile_stretch_gray(img_u8, mask=mask, gamma=0.6)
    enh = dilate_u8(enh, k=3, it=1)
    cv2.imwrite(str(out_base.with_name(out_base.stem + "_enh.png")), enh)


def save_jet_and_enh(out_base: Path, img_u8: np.ndarray) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    jet = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_base), jet)

    g = cv2.cvtColor(jet, cv2.COLOR_BGR2GRAY)
    mask = g > 0
    # enhance per-channel via same scalar stretch derived from gray
    enh_gray = percentile_stretch_gray(g, mask=mask, gamma=0.6)
    scale = (enh_gray.astype(np.float32) + 1e-6) / (g.astype(np.float32) + 1e-6)
    enh = np.clip(jet.astype(np.float32) * scale[..., None], 0, 255).astype(np.uint8)
    enh = dilate_u8(enh, k=3, it=1)
    cv2.imwrite(str(out_base.with_name(out_base.stem + "_enh.png")), enh)


def make_sem_color(sem_id: np.ndarray) -> np.ndarray:
    # BGR
    palette = {
        0: np.array([0, 200, 0], np.uint8),      # veg
        1: np.array([0, 0, 220], np.uint8),      # bldg
        2: np.array([220, 220, 220], np.uint8),  # road
    }
    H, W = sem_id.shape
    out = np.zeros((H, W, 3), np.uint8)
    for k, c in palette.items():
        m = (sem_id == k)
        out[m] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--ref_hw_from", required=True, help="Use semantic_map.npy (or any HxW image) to set output resolution.")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--zoom", type=float, default=5.4)
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--shift_y", type=float, default=-1.2)
    ap.add_argument("--angle", type=float, default=-31.0)
    ap.add_argument("--multiplier", type=float, default=0.85)
    ap.add_argument("--ndc_y_mode", default="y_up", choices=["y_up", "y_down"])
    ap.add_argument("--max_points", type=int, default=1200000)
    ap.add_argument("--sample_stats", type=int, default=200000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load priors
    pri = load_tensor_from_pt(args.priors).float()
    if pri.ndim != 2:
        raise ValueError(f"priors must be 2D, got {pri.shape}")
    N, D = pri.shape
    if D != 11:
        print(f"[Warn] expected D=11(v3), got D={D}. Will still try best-effort visualization.")
    finite_ratio = float(torch.isfinite(pri).all(dim=1).float().mean().item())

    # load xyz
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(args.ply)
    xyz = gaussians.get_xyz.detach().cpu().float()
    if xyz.shape[0] != N:
        raise ValueError(f"xyz N mismatch: xyz={xyz.shape[0]} priors={N}")
    xyz_np = xyz.numpy().astype(np.float32)

    # camera
    cams_data = json.loads(Path(args.cameras).read_text(encoding="utf-8"))
    full_T, cam_info = compute_topdown_fullT(
        xyz=xyz,
        cams_data=cams_data,
        zoom=args.zoom,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        angle_deg=args.angle,
        multiplier=args.multiplier,
    )

    H, W = load_hw_from(args.ref_hw_from)

    # indices for splat
    keep = min(int(args.max_points), int(N))
    if keep < N:
        idx = np.linspace(0, N - 1, keep, dtype=np.int64)
    else:
        idx = np.arange(N, dtype=np.int64)

    pts = xyz_np[idx]
    ones = np.ones((pts.shape[0], 1), np.float32)
    hom = np.concatenate([pts, ones], axis=1)  # Mx4
    clip = hom @ full_T
    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    ndc = clip[:, 0:3] / w

    x = (ndc[:, 0] * 0.5 + 0.5) * (W - 1)
    if args.ndc_y_mode == "y_down":
        y = (0.5 - ndc[:, 1] * 0.5) * (H - 1)
    else:
        y = (ndc[:, 1] * 0.5 + 0.5) * (H - 1)

    in_view = (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
    xi = np.clip(np.round(x).astype(np.int32), 0, W - 1)
    yi = np.clip(np.round(y).astype(np.int32), 0, H - 1)

    # feature getters (best-effort)
    def col(j: int) -> np.ndarray:
        if j < 0 or j >= D:
            return None
        return pri[idx, j].detach().cpu().numpy().astype(np.float32)

    # prepare scalar maps (splat)
    def splat_scalar(val01: np.ndarray) -> np.ndarray:
        img = np.zeros((H, W), np.float32)
        v = val01.copy()
        v[~in_view] = 0.0
        v = np.clip(v, 0.0, 1.0)
        img[yi, xi] = v.astype(np.float32)
        return img

    # semantic id
    sem_id_img = None
    if D >= 10:
        sem = pri[idx, 7:10].detach().cpu().numpy().astype(np.float32)
        sem_id = np.argmax(sem, axis=1).astype(np.uint8)
        sem_id[~in_view] = 0
        sem_id_img = np.zeros((H, W), np.uint8)
        sem_id_img[yi, xi] = sem_id

    # stats
    audit: Dict[str, Any] = {
        "priors_path": args.priors,
        "shape": [int(N), int(D)],
        "finite_ratio": finite_ratio,
        "camera": cam_info,
        "ndc_y_mode": args.ndc_y_mode,
        "out_hw": [int(H), int(W)],
    }

    # per-column stats (fast full scan)
    cols = {}
    for j in range(D):
        t = pri[:, j]
        cols[str(j)] = {
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
            "std": float(t.std().item()),
        }
    audit["col_stats"] = cols

    # normals
    if D >= 3:
        n = pri[:, 0:3]
        nn = torch.norm(n, dim=1)
        audit["normal_norm_mean"] = float(nn.mean().item())
        audit["normal_norm_std"] = float(nn.std().item())

    # semOH + sun
    n_s = min(int(args.sample_stats), int(N))
    if D == 11:
        sem_s = pri[:n_s, 7:10]
        ssum = sem_s.sum(dim=1)
        audit["semOH_sum_min"] = float(ssum.min().item())
        audit["semOH_sum_max"] = float(ssum.max().item())
        audit["semOH_sum_mean"] = float(ssum.mean().item())
        sem_id_s = torch.argmax(sem_s, dim=1)
        audit["semantic_hist_sample200k"] = torch.bincount(sem_id_s, minlength=3).cpu().tolist()

        sun = pri[:n_s, 10].clamp(0, 1)
        audit["sun_min"] = float(sun.min().item())
        audit["sun_max"] = float(sun.max().item())
        audit["sun_mean"] = float(sun.mean().item())
        audit["sun_std"] = float(sun.std().item())
        audit["sun_sparse_lt_0p05"] = float((sun < 0.05).float().mean().item())

    (out_dir / "priors_v3_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] wrote audit:", str(out_dir / "priors_v3_audit.json"))

    # --- Make maps ---
    # semantic id
    if sem_id_img is not None:
        sem_vis = make_sem_color(sem_id_img)
        cv2.imwrite(str(out_dir / "sem_id_topdown.png"), sem_vis)
        sem_enh = dilate_u8(sem_vis, k=3, it=1)
        cv2.imwrite(str(out_dir / "sem_id_topdown_enh.png"), sem_enh)

    # scalar channels: height(3), slope(4), shadowA(5), shadowB(6), sun(10)
    def save_scalar(name: str, val: np.ndarray):
        img = splat_scalar(val)
        u8 = (img * 255.0).astype(np.uint8)
        save_gray_and_enh(out_dir / f"{name}_topdown.png", u8)
        save_jet_and_enh(out_dir / f"{name}_topdown_jet.png", u8)

    h = col(3)
    if h is not None:
        # normalize height for visualization by percentiles on in-view points
        v = h[in_view]
        if v.size > 0:
            lo, hi = np.percentile(v, 1.0), np.percentile(v, 99.5)
            h01 = np.clip((h - lo) / max(hi - lo, 1e-6), 0, 1)
            save_scalar("height", h01)

    sl = col(4)
    if sl is not None:
        v = sl[in_view]
        if v.size > 0:
            lo, hi = np.percentile(v, 1.0), np.percentile(v, 99.5)
            sl01 = np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)
            save_scalar("slope", sl01)

    shA = col(5)
    if shA is not None:
        save_scalar("shadowA", np.clip(shA, 0, 1))

    shB = col(6)
    if shB is not None:
        save_scalar("shadowB", np.clip(shB, 0, 1))

    sun = col(10)
    if sun is not None:
        save_scalar("sun", np.clip(sun, 0, 1))

    # overlay sem * sun
    if sem_id_img is not None and sun is not None:
        sun_img = splat_scalar(np.clip(sun, 0, 1))
        sun_u8 = (sun_img * 255.0).astype(np.uint8)
        sem_rgb = make_sem_color(sem_id_img).astype(np.float32)
        factor = (0.25 + 0.75 * (sun_u8.astype(np.float32) / 255.0))[..., None]
        overlay = np.clip(sem_rgb * factor, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / "overlay_sem_sun.png"), overlay)
        overlay_enh = dilate_u8(overlay, k=3, it=1)
        cv2.imwrite(str(out_dir / "overlay_sem_sun_enh.png"), overlay_enh)

    print("[DONE] wrote images to:", str(out_dir))


if __name__ == "__main__":
    main()
