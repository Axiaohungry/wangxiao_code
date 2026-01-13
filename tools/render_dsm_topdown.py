# tools/render_dsm_topdown.py
# Phase4.1.1: render DSM/Depth (float) + topdown repro RGB
# Camera construction is aligned with render_top_down.py (PCA axes, right/up definition, max_span, defaults).

import os
import sys
import json
import math
import time
from argparse import ArgumentParser

import numpy as np
import cv2
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

try:
    from scene.gaussian_model import GaussianModel
except Exception:
    from scene import GaussianModel

from gaussian_renderer import render


# --- MiniCam (same structure as render_top_down.py) ---
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = int(width)
        self.image_height = int(height)
        self.FoVy = float(fovy)
        self.FoVx = float(fovx)
        self.znear = float(znear)
        self.zfar = float(zfar)
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform

        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)

    def get_calib_matrix_nerf(self):
        focal_y = self.image_height / (2.0 * self.tanfovy)
        focal_x = self.image_width / (2.0 * self.tanfovx)
        intrinsic = torch.tensor(
            [[focal_x, 0, self.image_width / 2.0],
             [0, focal_y, self.image_height / 2.0],
             [0, 0, 1]],
            dtype=torch.float32, device=self.world_view_transform.device
        )
        extrinsic = self.world_view_transform.transpose(0, 1)
        return intrinsic, extrinsic


def get_projection_matrix(znear, zfar, fovX, fovY, device="cuda"):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def look_at_topdown(cam_pos, target, up):
    z_axis = target - cam_pos
    dist = torch.norm(z_axis)
    if dist < 1e-5:
        return torch.eye(4, device=cam_pos.device)
    z_axis = z_axis / dist

    x_axis = torch.cross(up, z_axis, dim=0)
    if torch.norm(x_axis) < 1e-5:
        tmp = torch.tensor([1.0, 0.0, 0.0], device=cam_pos.device)
        x_axis = torch.cross(tmp, z_axis, dim=0)
    x_axis = x_axis / (torch.norm(x_axis) + 1e-8)

    y_axis = torch.cross(z_axis, x_axis, dim=0)
    y_axis = y_axis / (torch.norm(y_axis) + 1e-8)

    R = torch.stack([x_axis, y_axis, z_axis], dim=0)
    T = -torch.matmul(R, cam_pos)

    w2v = torch.eye(4, device=cam_pos.device)
    w2v[:3, :3] = R
    w2v[:3, 3] = T
    return w2v.transpose(0, 1).contiguous()


def _first_key(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def infer_fov_y(cams_data):
    # Try common fov keys
    frames = cams_data["frames"] if isinstance(cams_data, dict) and "frames" in cams_data else cams_data
    if not frames:
        return math.radians(60.0)

    c0 = frames[0]
    for k in ["FoVy", "FovY", "fovy", "fov_y", "camera_angle_y"]:
        if k in c0:
            return float(c0[k])

    # Infer from intrinsics if present
    h = c0.get("height", c0.get("h", c0.get("H", None)))
    fy = c0.get("fy", c0.get("fl_y", c0.get("focal_y", None)))
    if (h is not None) and (fy is not None) and float(fy) > 1e-6:
        return 2.0 * math.atan(float(h) / (2.0 * float(fy)))

    return math.radians(60.0)


def infer_cam_positions(cams_data, max_n=256):
    frames = cams_data["frames"] if isinstance(cams_data, dict) and "frames" in cams_data else cams_data
    pos = []
    for c in frames[:max_n]:
        p = None
        if "position" in c:
            p = c["position"]
        elif "camera_center" in c:
            p = c["camera_center"]
        elif "cam_pos" in c:
            p = c["cam_pos"]
        else:
            tm = c.get("transform_matrix", None)
            if tm is not None and isinstance(tm, (list, tuple)) and len(tm) == 4:
                p = [tm[0][3], tm[1][3], tm[2][3]]

        if p is not None and len(p) == 3:
            pos.append([float(p[0]), float(p[1]), float(p[2])])

    if not pos:
        return None
    return torch.tensor(pos, dtype=torch.float32)


def normalize_to_u8(x: np.ndarray):
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x, dtype=np.uint8), (0.0, 1.0)
    vmin = float(np.nanmin(x))
    vmax = float(np.nanmax(x))
    if vmax - vmin < 1e-8:
        return np.zeros_like(x, dtype=np.uint8), (vmin, vmax)
    y = (x - vmin) / (vmax - vmin)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8), (vmin, vmax)


def main():
    ap = ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True)

    ap.add_argument("--resolution", type=int, default=2048)
    ap.add_argument("--downscale", type=int, default=1)

    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_rgb_repro", required=True)

    # Defaults aligned with render_top_down.py + your topdown_final command
    ap.add_argument("--zoom", type=float, default=5.4)
    ap.add_argument("--shift_x", type=float, default=0.0)     # IMPORTANT: your topdown_final used shift_x=0
    ap.add_argument("--shift_y", type=float, default=-1.2)
    ap.add_argument("--angle", type=float, default=-31.0)
    ap.add_argument("--multiplier", type=float, default=0.85)

    ap.add_argument("--znear", type=float, default=0.01)
    ap.add_argument("--zfar", type=float, default=100.0)
    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()
    t0 = time.time()

    os.makedirs(os.path.dirname(args.out_npy), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_rgb_repro), exist_ok=True)

    device = args.device
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        device = "cpu"

    if not os.path.exists(args.ply):
        raise FileNotFoundError(args.ply)
    if not os.path.exists(args.cameras):
        raise FileNotFoundError(args.cameras)

    # Load gaussians (geometry only)
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(args.ply)

    xyz = gaussians.get_xyz.detach().to(device)
    center = xyz.mean(dim=0)

    # max_span definition aligned with render_top_down.py (use world x/y spans)
    min_xyz = torch.min(xyz, dim=0).values
    max_xyz = torch.max(xyz, dim=0).values
    span_x = float((max_xyz[0] - min_xyz[0]).item())
    span_y = float((max_xyz[1] - min_xyz[1]).item())
    max_span = max(span_x, span_y)

    # PCA on CPU aligned with render_top_down.py
    xyz_cpu = xyz.detach().float().cpu()
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = (xyz_centered.t() @ xyz_centered) / max(1, xyz_centered.shape[0])
    eigvals, eigvecs = torch.linalg.eigh(cov)
    normal = eigvecs[:, 0].to(device)  # smallest variance axis
    axis1 = eigvecs[:, 1].to(device)

    # Determine normal sign using mean camera position (from cameras.json)
    with open(args.cameras, "r", encoding="utf-8") as f:
        cams_data = json.load(f)
    cam_pos_list = infer_cam_positions(cams_data)
    if cam_pos_list is not None:
        mean_cam_pos = cam_pos_list.mean(dim=0).to(device)
        if torch.dot((mean_cam_pos - center), normal) < 0:
            normal = -normal

    # up/right aligned with render_top_down.py
    up_axis = axis1 - torch.dot(axis1, normal) * normal
    up_axis = up_axis / (torch.norm(up_axis) + 1e-8)
    right_axis = torch.cross(up_axis, normal, dim=0)

    # optional in-plane rotation (same as render_top_down.py)
    if float(args.angle) != 0.0:
        rad = math.radians(float(args.angle))
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        new_up = up_axis * cos_a + right_axis * sin_a
        up_axis = new_up / (torch.norm(new_up) + 1e-8)
        right_axis = torch.cross(up_axis, normal, dim=0)

    # FOV aligned: use FoVy from cameras.json when available (proxy for ref_cam.FoVy)
    fov_y = float(infer_fov_y(cams_data))
    fov_x_mod = 2.0 * math.atan(math.tan(fov_y / 2.0) * float(args.multiplier))

    # Render resolution
    W = int(args.resolution // max(1, args.downscale))
    H = int(args.resolution // max(1, args.downscale))

    # Camera placement aligned with render_top_down.py
    base_height = (max_span / 2.0) / math.tan(fov_y / 2.0)
    target_height = base_height / float(args.zoom)
    shift_vec = (right_axis * float(args.shift_x)) + (up_axis * float(args.shift_y))
    target_center = center + shift_vec
    cam_pos = target_center + normal * target_height

    w2v_T = look_at_topdown(cam_pos, target_center, up_axis)
    P = get_projection_matrix(float(args.znear), float(args.zfar), fov_x_mod, fov_y, device=device).transpose(0, 1)
    full_T = w2v_T @ P

    custom_cam = MiniCam(
        width=W,
        height=H,
        fovy=fov_y,
        fovx=fov_x_mod,
        znear=float(args.znear),
        zfar=float(args.zfar),
        world_view_transform=w2v_T,
        full_proj_transform=full_T
    )

    # Minimal pipe args (render() requires these attrs in many 3DGS forks)
    pipe = type("Pipe", (object,), {
        "compute_cov3D_python": False,
        "convert_SHs_python": False,
        "brdf": False,
        "brdf_mode": "pbbr",
    })()

    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    with torch.no_grad():
        out = render(custom_cam, gaussians, pipe, bg, scaling_modifier=1.0)

    # RGB repro
    rgb = out["render"]
    if rgb.ndim == 4:
        rgb = rgb[0]
    rgb = rgb.detach().float().clamp(0, 1).cpu().numpy()
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(args.out_rgb_repro, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))

    # Depth (prefer renderer-provided depth)
    depth_key = _first_key(out, ["depth", "depths", "depth_map", "D", "median_depth", "render_depth"])
    if depth_key is None:
        raise KeyError(f"render() output has no depth-like key. keys={list(out.keys())}")

    d = out[depth_key]
    if isinstance(d, torch.Tensor):
        if d.ndim == 4:
            d = d[0, 0]
        elif d.ndim == 3 and d.shape[0] in [1, 3]:
            d = d[0]
        depth = d.detach().float().cpu().numpy().astype(np.float32)
    else:
        depth = np.array(d, dtype=np.float32)

    np.save(args.out_npy, depth)

    vis_u8, (vmin, vmax) = normalize_to_u8(depth)
    cv2.imwrite(args.out_png, vis_u8)

    finite = np.isfinite(depth)
    stats = {
        "time_sec": round(time.time() - t0, 3),
        "ply": os.path.abspath(args.ply),
        "cameras": os.path.abspath(args.cameras),
        "H": int(depth.shape[0]),
        "W": int(depth.shape[1]),
        "finite_ratio": float(finite.mean()),
        "min": float(np.nanmin(depth)),
        "max": float(np.nanmax(depth)),
        "mean": float(np.nanmean(depth)),
        "std": float(np.nanstd(depth)),
        "vis_min": float(vmin),
        "vis_max": float(vmax),
        "depth_source": f"render_out[{depth_key}]",
        "camera": {
            "zoom": float(args.zoom),
            "shift_x": float(args.shift_x),
            "shift_y": float(args.shift_y),
            "angle": float(args.angle),
            "multiplier": float(args.multiplier),
            "fov_y": float(fov_y),
            "fov_x_mod": float(fov_x_mod),
            "znear": float(args.znear),
            "zfar": float(args.zfar),
            "max_span": float(max_span),
            "span_x": float(span_x),
            "span_y": float(span_y),
        }
    }
    stats_path = os.path.splitext(args.out_npy)[0] + "_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[OK] Wrote:")
    print(" ", args.out_npy)
    print(" ", args.out_png)
    print(" ", args.out_rgb_repro)
    print(" ", stats_path)
    print(f"[DSM] finite_ratio={stats['finite_ratio']:.4f} min={stats['min']:.6f} max={stats['max']:.6f} src={stats['depth_source']}")
    print(f"    [MEMO] Params: --multiplier {args.multiplier} --shift_x {args.shift_x} --shift_y {args.shift_y} --zoom {args.zoom} --angle {args.angle}")


if __name__ == "__main__":
    main()
