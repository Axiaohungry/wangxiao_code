# tools/render_thermal_topdown_pred.py
# Render a single topdown thermal prediction (raw gray PNG + raw NPY) from a trained thermal net.
# Minimal, 6GB-safe (single frame, supports render_scale).

import os
import sys
import json
import math
import cv2
import torch
import numpy as np
from argparse import ArgumentParser

# Ensure project root import works
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.thermal_network import ThermalAttrNet


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)

    def get_calib_matrix_nerf(self):
        focal_y = self.image_height / (2.0 * self.tanfovy)
        focal_x = self.image_width / (2.0 * self.tanfovx)
        intrinsic = torch.tensor([
            [focal_x, 0, self.image_width / 2.0],
            [0, focal_y, self.image_height / 2.0],
            [0, 0, 1]
        ], dtype=torch.float32, device="cuda")
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


@torch.no_grad()
def force_apply_texture_no_grad(gaussians: GaussianModel, rgb01: torch.Tensor):
    C0 = 0.28209479177387814
    sh_dc = (rgb01 - 0.5) / C0  # [N,3]
    dc = gaussians._features_dc
    if isinstance(dc, torch.nn.Parameter):
        dc.data[:, 0, :].copy_(sh_dc)
    else:
        gaussians._features_dc = sh_dc.unsqueeze(1)
    if getattr(gaussians, "_features_rest", None) is not None and gaussians._features_rest.numel() > 0:
        if isinstance(gaussians._features_rest, torch.nn.Parameter):
            gaussians._features_rest.data.zero_()
        else:
            gaussians._features_rest.zero_()
    gaussians.active_sh_degree = 0


def safe_load_tensor(path: str) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    ap = ArgumentParser()
    ap.add_argument("--model_path", "-m", default="output/debug_run")
    ap.add_argument("--priors_path", required=True)
    ap.add_argument("--thermal_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)

    # camera params (must match your topdown/LST alignment)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--render_scale", type=float, default=0.5)  # 0.5->1024
    ap.add_argument("--zoom", type=float, default=5.4)
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--shift_y", type=float, default=-1.2)
    ap.add_argument("--angle", type=float, default=-31.0)
    ap.add_argument("--multiplier", type=float, default=0.85)
    ap.add_argument("--tag", default="pred")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ply_path = os.path.join(args.model_path, "point_cloud/iteration_7000/point_cloud.ply")
    cameras_json_path = os.path.join(args.model_path, "cameras.json")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(ply_path)
    if not os.path.exists(cameras_json_path):
        raise FileNotFoundError(cameras_json_path)
    if not os.path.exists(args.priors_path):
        raise FileNotFoundError(args.priors_path)
    if not os.path.exists(args.thermal_ckpt):
        raise FileNotFoundError(args.thermal_ckpt)

    device = "cuda"
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    pipeline_args = type("Pipe", (object,), {
        "compute_cov3D_python": False,
        "convert_SHs_python": False,
        "brdf": False,
        "brdf_mode": "pbbr",
    })()

    # Load gaussians
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)
    gaussians._xyz.requires_grad = False
    for name in ["_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
        if hasattr(gaussians, name) and getattr(gaussians, name) is not None:
            getattr(gaussians, name).requires_grad = False

    xyz = gaussians.get_xyz.detach().to(device)
    center = xyz.mean(dim=0)
    min_xyz, _ = torch.min(xyz, dim=0)
    max_xyz, _ = torch.max(xyz, dim=0)
    span_x = (max_xyz[0] - min_xyz[0]).item()
    span_y = (max_xyz[1] - min_xyz[1]).item()
    max_span = max(span_x, span_y)

    # Load priors (NxD)
    pri = safe_load_tensor(args.priors_path)
    if not torch.is_tensor(pri):
        raise ValueError("priors_path must be a torch.Tensor saved by torch.save(tensor, ...)")
    pri = pri.to(device).float()
    if pri.dim() != 2 or pri.shape[0] != xyz.shape[0]:
        raise ValueError(f"priors shape mismatch: {tuple(pri.shape)} vs xyz {tuple(xyz.shape)}")

    # Build camera (match train_thermal_robust)
    with open(cameras_json_path, "r") as f:
        cams_data = json.load(f)
    ref_cam = cams_data[0]
    fov_y = 2 * math.atan(ref_cam["height"] / (2 * ref_cam["fy"]))
    fov_x_mod = 2 * math.atan(math.tan(fov_y / 2) * args.multiplier)

    # PCA
    xyz_cpu = xyz.detach().cpu()
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    _, eigvecs = torch.linalg.eigh(cov)
    normal = eigvecs[:, 0].to(device)
    axis1 = eigvecs[:, 1].to(device)

    cam_centers = [np.array(c["position"]) for c in cams_data[:10]]
    mean_cam_pos = torch.tensor(np.mean(cam_centers, axis=0)).float().to(device)
    if torch.dot((mean_cam_pos - center), normal) < 0:
        normal = -normal

    up_axis = axis1 - torch.dot(axis1, normal) * normal
    up_axis = up_axis / (torch.norm(up_axis) + 1e-8)
    right_axis = torch.cross(up_axis, normal, dim=0)

    if abs(args.angle) > 1e-6:
        rad = math.radians(args.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        new_up = up_axis * cos_a + right_axis * sin_a
        up_axis = new_up / (torch.norm(new_up) + 1e-8)
        right_axis = torch.cross(up_axis, normal, dim=0)

    W = int(args.width * args.render_scale)
    H = int(args.height * args.render_scale)

    base_height = (max_span / 2.0) / np.tan(fov_y / 2.0)
    target_height = base_height / args.zoom
    shift_vec = (right_axis * args.shift_x) + (up_axis * args.shift_y)
    target_center = center + shift_vec
    cam_pos = target_center + normal * target_height

    w2v = look_at_topdown(cam_pos, target_center, up_axis)
    proj = get_projection_matrix(0.01, 100.0, fov_x_mod, fov_y).transpose(0, 1)
    full_T = w2v @ proj
    cam = MiniCam(W, H, fov_y, fov_x_mod, 0.01, 100.0, w2v, full_T)

    # Load net (dynamic input_ch)
    in_ch = 3 + pri.shape[1]
    net = ThermalAttrNet(input_ch=in_ch, W=16).to(device)
    sd = safe_load_tensor(args.thermal_ckpt)
    net.load_state_dict(sd, strict=True)
    net.eval()

    # Forward
    xyz_norm = (xyz - center) / (max_span + 1e-6)
    full_input = torch.cat([xyz_norm, pri], dim=1).contiguous()
    thermal_val = net(full_input)          # [N,1]
    rgb = thermal_val.expand(-1, 3)        # [N,3] in 0..1

    force_apply_texture_no_grad(gaussians, rgb)
    out = render(cam, gaussians, pipeline_args, bg)
    pred = out["render"][0].detach().clamp(0, 1)       # [H,W]

    pred_np = pred.cpu().numpy().astype(np.float32)
    np.save(os.path.join(args.out_dir, f"{args.tag}_raw.npy"), pred_np)

    pred_u8 = (pred_np * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, f"{args.tag}_gray.png"), pred_u8)

    pred_color = cv2.applyColorMap(pred_u8, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(args.out_dir, f"{args.tag}_jet.png"), pred_color)

    stats = {
        "tag": args.tag,
        "H": int(H), "W": int(W),
        "priors_shape": [int(pri.shape[0]), int(pri.shape[1])],
        "input_ch": int(in_ch),
        "pred_min": float(pred_np.min()),
        "pred_max": float(pred_np.max()),
        "pred_mean": float(pred_np.mean()),
        "pred_std": float(pred_np.std()),
    }
    with open(os.path.join(args.out_dir, f"{args.tag}_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[OK] Wrote:")
    print(" ", os.path.join(args.out_dir, f"{args.tag}_raw.npy"))
    print(" ", os.path.join(args.out_dir, f"{args.tag}_gray.png"))
    print(" ", os.path.join(args.out_dir, f"{args.tag}_jet.png"))
    print(" ", os.path.join(args.out_dir, f"{args.tag}_stats.json"))


if __name__ == "__main__":
    main()
