import torch
import os
import cv2
import json
import numpy as np
import math
from tqdm import tqdm
from argparse import ArgumentParser
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.thermal_network import ThermalAttrNet
import sys


# ==============================================================================
# 1. 核心相机类 (MiniCam) - 已修复内参维度
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

        # --- [关键修复] ---
        # 必须是 3x3 矩阵，否则 graphics_utils.py 里的矩阵乘法会崩
        intrinsic = torch.tensor([
            [focal_x, 0, self.image_width / 2.0],
            [0, focal_y, self.image_height / 2.0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.world_view_transform.device)

        # 外参保持 4x4
        extrinsic = self.world_view_transform.transpose(0, 1)
        return intrinsic, extrinsic


# ==============================================================================
# Helper Functions
# ==============================================================================
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


def get_look_at(cam_pos, target, up):
    z_axis = target - cam_pos
    dist = torch.norm(z_axis)
    if dist < 1e-5: return torch.eye(4, device=cam_pos.device)
    z_axis = z_axis / dist

    x_axis = torch.cross(up, z_axis, dim=0)
    # 鲁棒性检查
    if torch.norm(x_axis) < 1e-5:
        temp = torch.tensor([1.0, 0.0, 0.0], device=cam_pos.device)
        if torch.abs(torch.dot(temp, z_axis)) > 0.9: temp = torch.tensor([0.0, 1.0, 0.0], device=cam_pos.device)
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default="output/debug_run")
    parser.add_argument("--thermal_ckpt", type=str, default="output/thermal_v1/thermal_net_v1.pth")
    parser.add_argument("--priors_path", type=str, default="output/debug_run/priors.pt")
    parser.add_argument("--output_video", type=str, default="output/thermal_v1/video_v1_physics.mp4")
    parser.add_argument("--render_res", type=int, default=1024)
    parser.add_argument("--elevation", type=float, default=45.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)

    # 1. Load Geometry
    print("Loading Geometry...")
    ply_path = os.path.join(args.model_path, "point_cloud/iteration_7000/point_cloud.ply")
    cameras_json_path = os.path.join(args.model_path, "cameras.json")

    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)

    gaussians._xyz.requires_grad = False
    for name in ["_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
        getattr(gaussians, name).requires_grad = False

    # 2. Load Priors
    print(f"Loading Priors: {args.priors_path}")
    priors = torch.load(args.priors_path).cuda()

    # 3. Load Network (v1, Input=8)
    print(f"Loading Thermal Net v1...")
    thermal_net = ThermalAttrNet(input_ch=8, W=16).cuda()
    thermal_net.load_state_dict(torch.load(args.thermal_ckpt))
    thermal_net.eval()

    # 4. Orientation (Video 1 Logic: Inverted PCA)
    device = "cuda"
    xyz = gaussians.get_xyz.detach()
    center = xyz.mean(dim=0)

    xyz_cpu = xyz.cpu();
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)
    pca_normal = eigvecs[:, 0].to(device)

    # 强制使用 -PCA 作为 Up 向量 (你的正确方向)
    up_vec = -pca_normal
    up_vec = up_vec / torch.norm(up_vec)

    # 构造旋转平面
    temp = torch.tensor([1.0, 0.0, 0.0], device=device)
    if torch.abs(torch.dot(temp, up_vec)) > 0.9: temp = torch.tensor([0.0, 1.0, 0.0], device=device)
    right_vec = torch.cross(up_vec, temp);
    right_vec = right_vec / torch.norm(right_vec)
    fwd_vec = torch.cross(up_vec, right_vec);
    fwd_vec = fwd_vec / torch.norm(fwd_vec)

    # Orbit Params
    max_dist = torch.max(torch.norm(xyz - center, dim=1)).item()
    orbit_radius = max_dist * 1.5

    with open(cameras_json_path, 'r') as f:
        cams_data = json.load(f)
    ref_cam = cams_data[0]
    fov_y = 2 * math.atan(ref_cam['height'] / (2 * ref_cam['fy']))
    fov_x = fov_y

    # 5. Render Loop
    frames = []
    n_frames = 120
    elevation_rad = math.radians(args.elevation)
    h_up = orbit_radius * math.sin(elevation_rad)
    r_plane = orbit_radius * math.cos(elevation_rad)

    pipe = type('Pipe', (object,),
                {"compute_cov3D_python": False, "convert_SHs_python": False, "brdf": False, "brdf_mode": "pbbr"})()
    bg = torch.tensor([0.0, 0.0, 0.0]).cuda()

    print("Pre-calculating Thermal Colors...")
    with torch.no_grad():
        full_input = torch.cat([gaussians.get_xyz, priors], dim=1)
        t_vals = thermal_net(full_input)
        colors = t_vals.repeat(1, 3)

    print("Rendering Orbit...")
    for i in tqdm(range(n_frames)):
        angle = 2 * math.pi * (i / n_frames)

        # Corrected Orbit Logic: - (up_vec * h_up) to fly above
        offset = (right_vec * math.cos(angle) * r_plane) + \
                 (fwd_vec * math.sin(angle) * r_plane) - \
                 (up_vec * h_up)

        cam_pos = center + offset
        w2v = get_look_at(cam_pos, center, up_vec)
        proj = get_projection_matrix(0.1, 1000.0, fov_x, fov_y).transpose(0, 1)
        full_proj = w2v @ proj
        cam = MiniCam(args.render_res, args.render_res, fov_y, fov_x, 0.1, 1000.0, w2v, full_proj)

        out = render(cam, gaussians, pipe, bg, override_color=colors)["render"]
        img = out.detach().permute(1, 2, 0).cpu().numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(cv2.applyColorMap(img, cv2.COLORMAP_JET))

    print(f"Saving video to {args.output_video}")
    h, w, _ = frames[0].shape
    out_vid = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    for f in frames: out_vid.write(f)
    out_vid.release()
    print("Done!")