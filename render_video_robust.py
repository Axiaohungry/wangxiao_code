# D:\PycharmProjects\wangxiao_code\render_video_robust.py
import os
import sys
import json
import math
import cv2
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

# -----------------------------------------------------------------------------
# Ensure project import works no matter where you run
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.thermal_network import ThermalAttrNet


# ==============================================================================
# MiniCam (same convention as project; fixed 3x3 intrinsic)
# ==============================================================================
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
        ], dtype=torch.float32, device=self.world_view_transform.device)
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


# IMPORTANT: exactly the same look-at as render_video_v1_compare.py
def get_look_at(cam_pos, target, up):
    z_axis = target - cam_pos
    dist = torch.norm(z_axis)
    if dist < 1e-5:
        return torch.eye(4, device=cam_pos.device)

    z_axis = z_axis / dist
    x_axis = torch.cross(up, z_axis, dim=0)

    if torch.norm(x_axis) < 1e-5:
        temp = torch.tensor([1.0, 0.0, 0.0], device=cam_pos.device)
        if torch.abs(torch.dot(temp, z_axis)) > 0.9:
            temp = torch.tensor([0.0, 1.0, 0.0], device=cam_pos.device)
        x_axis = torch.cross(temp, z_axis, dim=0)

    x_axis = x_axis / (torch.norm(x_axis) + 1e-8)
    y_axis = torch.cross(z_axis, x_axis, dim=0)
    y_axis = y_axis / (torch.norm(y_axis) + 1e-8)

    R = torch.stack([x_axis, y_axis, z_axis], dim=0)
    T = -torch.matmul(R, cam_pos)

    w2v = torch.eye(4, device=cam_pos.device)
    w2v[:3, :3] = R
    w2v[:3, 3] = T
    return w2v.transpose(0, 1).contiguous()


def parse_vec3(s: str) -> torch.Tensor:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected vec3 like 'x,y,z', got: {s}")
    v = torch.tensor([float(parts[0]), float(parts[1]), float(parts[2])], dtype=torch.float32, device="cuda")
    v = v / (torch.norm(v) + 1e-8)
    return v


# ==============================================================================
# Force-apply SH DC (no grad needed for video)
# ==============================================================================
@torch.no_grad()
def force_apply_texture_no_grad(gaussians: GaussianModel, rgb01: torch.Tensor):
    """
    rgb01: [N, 3] in 0..1
    Writes to gaussians SH (degree=0), so render() uses the new colors.
    """
    C0 = 0.28209479177387814
    sh_dc = (rgb01 - 0.5) / C0  # [N,3]

    # Write into existing storage if possible (avoid realloc / keep parameters)
    dc = gaussians._features_dc
    if isinstance(dc, torch.nn.Parameter):
        dc.data[:, 0, :].copy_(sh_dc)
    else:
        gaussians._features_dc = sh_dc.unsqueeze(1)

    # Clear higher-order SH to avoid mixing old colors
    if getattr(gaussians, "_features_rest", None) is not None:
        if gaussians._features_rest.numel() > 0:
            if isinstance(gaussians._features_rest, torch.nn.Parameter):
                gaussians._features_rest.data.zero_()
            else:
                gaussians._features_rest.zero_()

    gaussians.active_sh_degree = 0


def main():
    ap = ArgumentParser()
    ap.add_argument("--model_path", "-m", default="output/debug_run")
    ap.add_argument("--priors_path", required=True)
    ap.add_argument("--thermal_ckpt", required=True)
    ap.add_argument("--output_video", default="output/thermal_robust/video_robust.mp4")

    ap.add_argument("--render_res", type=int, default=1024)
    ap.add_argument("--n_frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)

    # Orbit params (match your v1 physics defaults)
    ap.add_argument("--elevation", type=float, default=45.0)
    ap.add_argument("--orbit_mul", type=float, default=1.5)
    ap.add_argument("--start_angle_deg", type=float, default=0.0)

    # Optional: expected_up from bake_priors_physics.py (will be auto-aligned to -pca_normal)
    ap.add_argument("--expected_up", default="", help='optional vec3 "x,y,z" from bake_priors_physics')

    # Chunking to protect 6GB VRAM
    ap.add_argument("--chunk", type=int, default=300000)

    args = ap.parse_args()
    device = "cuda"

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

    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)

    # 1) Load Gaussians
    print("Loading Geometry...")
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)

    gaussians._xyz.requires_grad = False
    for name in ["_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
        getattr(gaussians, name).requires_grad = False

    xyz = gaussians.get_xyz.detach()
    center = xyz.mean(dim=0)

    # 2) Load Priors (safe load)
    print(f"Loading Priors: {args.priors_path}")
    priors = torch.load(args.priors_path, map_location="cuda", weights_only=True)
    if priors.shape[0] != xyz.shape[0] or priors.shape[1] != 5:
        raise ValueError(f"priors shape mismatch: {tuple(priors.shape)} vs xyz {tuple(xyz.shape)}")

    # 3) Load Robust Net (input=8, xyz_norm+priors)
    print(f"Loading Thermal Net: {args.thermal_ckpt}")
    net = ThermalAttrNet(input_ch=8, W=16).cuda()
    net.load_state_dict(torch.load(args.thermal_ckpt, map_location="cuda", weights_only=True))
    net.eval()

    # 4) FOV from cameras.json
    with open(cameras_json_path, "r") as f:
        cams_data = json.load(f)
    ref_cam = cams_data[0]
    fov_y = 2 * math.atan(ref_cam["height"] / (2 * ref_cam["fy"]))
    fov_x = fov_y

    # 5) Build orbit basis (STRICTLY match render_video_v1_compare: up_vec = -pca_normal)
    xyz_cpu = xyz.cpu()
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    _, eigvecs = torch.linalg.eigh(cov)
    pca_normal = eigvecs[:, 0].to(device)
    up_ref = (-pca_normal) / (torch.norm(pca_normal) + 1e-8)  # this is the "good view" convention

    if args.expected_up.strip():
        exp = parse_vec3(args.expected_up)
        # Align expected_up to the same hemisphere as up_ref to avoid flipped view
        if torch.dot(exp, up_ref) < 0:
            exp = -exp
        up_vec = exp
        print(f"[Up] using expected_up aligned. dot(expected_up, -pca)= {torch.dot(up_vec, up_ref).item():.4f}")
    else:
        up_vec = up_ref
        print("[Up] using -PCA normal (same as render_video_v1_compare.py)")

    temp = torch.tensor([1.0, 0.0, 0.0], device=device)
    if torch.abs(torch.dot(temp, up_vec)) > 0.9:
        temp = torch.tensor([0.0, 1.0, 0.0], device=device)
    right_vec = torch.cross(up_vec, temp, dim=0)
    right_vec = right_vec / (torch.norm(right_vec) + 1e-8)
    fwd_vec = torch.cross(up_vec, right_vec, dim=0)
    fwd_vec = fwd_vec / (torch.norm(fwd_vec) + 1e-8)

    # Orbit radius
    max_dist = torch.max(torch.norm(xyz - center, dim=1)).item()
    orbit_radius = max_dist * args.orbit_mul

    elevation_rad = math.radians(args.elevation)
    h_up = orbit_radius * math.sin(elevation_rad)
    r_plane = orbit_radius * math.cos(elevation_rad)

    # 6) Build xyz_norm exactly like train_thermal_robust.py
    min_xyz, _ = torch.min(xyz, dim=0)
    max_xyz, _ = torch.max(xyz, dim=0)
    span_x = (max_xyz[0] - min_xyz[0]).item()
    span_y = (max_xyz[1] - min_xyz[1]).item()
    max_span = max(span_x, span_y)
    xyz_norm = (xyz - center) / (max_span + 1e-6)

    # 7) Predict colors in chunks, then FORCE APPLY into gaussians SH
    print("Pre-calculating robust thermal colors (chunked) + force_apply ...")
    N = xyz.shape[0]
    C0 = 0.28209479177387814

    # We write directly into gaussians._features_dc to avoid extra huge tensors
    dc = gaussians._features_dc
    if isinstance(dc, torch.nn.Parameter):
        dc_data = dc.data
    else:
        # if not parameter, make sure it exists with correct shape
        gaussians._features_dc = torch.zeros((N, 1, 3), device=device, dtype=torch.float32)
        dc_data = gaussians._features_dc

    with torch.no_grad():
        for s in tqdm(range(0, N, args.chunk), desc="infer+apply"):
            e = min(N, s + args.chunk)
            inp = torch.cat([xyz_norm[s:e], priors[s:e]], dim=1).contiguous()  # [B,8]
            t = net(inp)  # [B,1] in 0..1
            rgb = t.repeat(1, 3)  # [B,3]
            sh = (rgb - 0.5) / C0  # [B,3]
            dc_data[s:e, 0, :].copy_(sh)

        # clear rest & set degree
        if getattr(gaussians, "_features_rest", None) is not None and gaussians._features_rest.numel() > 0:
            if isinstance(gaussians._features_rest, torch.nn.Parameter):
                gaussians._features_rest.data.zero_()
            else:
                gaussians._features_rest.zero_()
        gaussians.active_sh_degree = 0

    # 8) Render orbit WITHOUT override_color (colors already written)
    pipe = type("Pipe", (object,), {"compute_cov3D_python": False, "convert_SHs_python": False, "brdf": False, "brdf_mode": "pbbr"})()
    bg = torch.tensor([0.0, 0.0, 0.0], device=device)

    frames = []
    print("Rendering robust orbit...")
    start_angle = math.radians(args.start_angle_deg)

    for i in tqdm(range(args.n_frames)):
        angle = start_angle + 2 * math.pi * (i / args.n_frames)

        # Exactly the same sign convention as render_video_v1_compare.py
        # (You already verified this view is correct.)
        offset = (right_vec * math.cos(angle) * r_plane) + \
                 (fwd_vec * math.sin(angle) * r_plane) - \
                 (up_vec * h_up)

        cam_pos = center + offset
        w2v = get_look_at(cam_pos, center, up_vec)
        proj = get_projection_matrix(0.1, 1000.0, fov_x, fov_y, device=device).transpose(0, 1)
        full_proj = w2v @ proj
        cam = MiniCam(args.render_res, args.render_res, fov_y, fov_x, 0.1, 1000.0, w2v, full_proj)

        out = render(cam, gaussians, pipe, bg)["render"]  # [3,H,W]
        pred = out[0, :, :]  # grayscale since RGB identical
        img = pred.detach().cpu().numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(cv2.applyColorMap(img, cv2.COLORMAP_JET))

    print(f"Saving to: {args.output_video}")
    h, w, _ = frames[0].shape
    out_vid = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for f in frames:
        out_vid.write(f)
    out_vid.release()
    print("Done.")


if __name__ == "__main__":
    main()


# render_video_robust.py：对任意 robust ckpt 渲染环绕视频（可指定 --expected_up，也可自动用 PCA+相机纠正得到 real_up）