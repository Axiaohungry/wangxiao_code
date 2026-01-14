# tools/vis_shadow_mean.py
# Render shadowA/shadowB as per-pixel aggregated heatmaps (mean/max), avoiding "splat speckle" illusions.

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import cv2

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scene.gaussian_model import GaussianModel

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

def compute_fullT(xyz: torch.Tensor, cams_data, W, H, zoom, shift_x, shift_y, angle_deg, multiplier):
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
        up_axis = new_up / (torch.norm(new_up) + 1e-8)
        right_axis = torch.cross(up_axis, normal, dim=0)

    base_height = (max_span / 2.0) / np.tan(fov_y / 2.0)
    target_height = base_height / zoom
    shift_vec = (right_axis * shift_x) + (up_axis * shift_y)
    target_center = center + shift_vec
    cam_pos = target_center + normal * target_height

    w2v = look_at_topdown(cam_pos, target_center, up_axis)
    proj = get_projection_matrix(0.01, 100.0, fov_x_mod, fov_y).transpose(0, 1).contiguous()
    full_T = (w2v @ proj).numpy().astype(np.float32)
    return full_T

def project_xy(xyz_np, full_T_np, W, H, ndc_y_mode):
    hom = np.concatenate([xyz_np.astype(np.float32), np.ones((xyz_np.shape[0], 1), np.float32)], axis=1)
    clip = hom @ full_T_np
    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    ndc = clip[:, 0:3] / w
    x = (ndc[:, 0] * 0.5 + 0.5) * (W - 1)

    # IMPORTANT: choose one convention and stick to it
    if ndc_y_mode == "y_up":
        # ndc y is up, image y is down
        y = (0.5 - ndc[:, 1] * 0.5) * (H - 1)
    else:
        # ndc y is already down-like (rare). Keep as is.
        y = (ndc[:, 1] * 0.5 + 0.5) * (H - 1)

    in_view = (ndc[:, 0] >= -1) & (ndc[:, 0] <= 1) & (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1)
    xi = np.clip(np.round(x).astype(np.int32), 0, W - 1)
    yi = np.clip(np.round(y).astype(np.int32), 0, H - 1)
    return xi, yi, in_view

def agg_to_image(xi, yi, val, H, W, mode="mean"):
    idx = yi.astype(np.int64) * W + xi.astype(np.int64)
    if mode == "max":
        # max via sorting (memory-friendly enough for debug with 4.4M)
        order = np.argsort(idx)
        idx_s = idx[order]
        val_s = val[order]
        out = np.full((H * W,), np.nan, np.float32)
        # group max
        start = 0
        while start < idx_s.size:
            end = start + 1
            while end < idx_s.size and idx_s[end] == idx_s[start]:
                end += 1
            out[idx_s[start]] = np.max(val_s[start:end])
            start = end
        out = np.nan_to_num(out, nan=0.0)
        return out.reshape(H, W)
    else:
        acc = np.bincount(idx, weights=val, minlength=H * W).astype(np.float32)
        cnt = np.bincount(idx, minlength=H * W).astype(np.float32)
        img = np.divide(acc, np.maximum(cnt, 1.0), dtype=np.float32)
        img[cnt == 0] = 0.0
        return img.reshape(H, W)

def to_u8(img, p_lo=1.0, p_hi=99.5, gamma=0.6):
    m = img > 0
    if np.any(m):
        v = img[m]
        lo = np.percentile(v, p_lo)
        hi = np.percentile(v, p_hi)
        x = (img - lo) / max(hi - lo, 1e-6)
        x = np.clip(x, 0, 1)
        x = np.power(x, gamma)
        return (x * 255).astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--ref_hw_from", required=True)  # .npy
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ndc_y_mode", default="y_up", choices=["y_up", "y_down"])
    ap.add_argument("--zoom", type=float, default=5.4)
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--shift_y", type=float, default=-1.2)
    ap.add_argument("--angle", type=float, default=-31.0)
    ap.add_argument("--multiplier", type=float, default=0.85)
    ap.add_argument("--mode", default="mean", choices=["mean", "max"])
    ap.add_argument("--blur", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = np.load(args.ref_hw_from)
    H, W = ref.shape[:2]

    pri = torch.load(args.priors, map_location="cpu")
    if not torch.is_tensor(pri):
        raise ValueError("priors must be a torch Tensor")
    pri = pri.float().numpy()

    gauss = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gauss.load_ply(args.ply)
    xyz = gauss.get_xyz.detach().cpu().float().numpy()
    if xyz.shape[0] != pri.shape[0]:
        raise ValueError(f"xyz N mismatch: {xyz.shape[0]} vs priors {pri.shape[0]}")

    cams_data = json.loads(Path(args.cameras).read_text(encoding="utf-8"))
    fullT = compute_fullT(torch.from_numpy(xyz).float(), cams_data, W, H, args.zoom, args.shift_x, args.shift_y, args.angle, args.multiplier)

    xi, yi, in_view = project_xy(xyz, fullT, W, H, args.ndc_y_mode)
    # optionally drop out-of-view to avoid polluting map
    xi = xi[in_view]; yi = yi[in_view]

    shadowA = pri[in_view, 5].astype(np.float32)
    shadowB = pri[in_view, 6].astype(np.float32)

    imgA = agg_to_image(xi, yi, shadowA, H, W, mode=args.mode)
    imgB = agg_to_image(xi, yi, shadowB, H, W, mode=args.mode)

    if args.blur and args.blur > 0:
        k = int(max(3, round(args.blur * 6) | 1))
        imgA = cv2.GaussianBlur(imgA, (k, k), args.blur)
        imgB = cv2.GaussianBlur(imgB, (k, k), args.blur)

    uA = to_u8(imgA)
    uB = to_u8(imgB)

    cv2.imwrite(str(out_dir / f"shadowA_{args.mode}.png"), uA)
    cv2.imwrite(str(out_dir / f"shadowB_{args.mode}.png"), uB)
    cv2.imwrite(str(out_dir / f"shadowA_{args.mode}_jet.png"), cv2.applyColorMap(uA, cv2.COLORMAP_JET))
    cv2.imwrite(str(out_dir / f"shadowB_{args.mode}_jet.png"), cv2.applyColorMap(uB, cv2.COLORMAP_JET))

    audit = {
        "H": int(H), "W": int(W),
        "ndc_y_mode": args.ndc_y_mode,
        "mode": args.mode,
        "blur": float(args.blur),
        "in_view_ratio": float(np.mean(in_view)),
        "shadowA_stats": [float(shadowA.min()), float(shadowA.max()), float(shadowA.mean()), float(shadowA.std())],
        "shadowB_stats": [float(shadowB.min()), float(shadowB.max()), float(shadowB.mean()), float(shadowB.std())],
    }
    (out_dir / "shadow_mean_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("[OK] wrote:", out_dir)

if __name__ == "__main__":
    main()
