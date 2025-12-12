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
# MiniCam
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


def get_look_at(cam_pos, target, up):
    z_axis = target - cam_pos
    dist = torch.norm(z_axis)
    if dist < 1e-5: return torch.eye(4, device=cam_pos.device)
    z_axis = z_axis / dist

    x_axis = torch.cross(up, z_axis, dim=0)
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
    parser.add_argument("--thermal_ckpt", type=str, default="output/thermal_v0/thermal_net_v0.pth")
    parser.add_argument("--output_video", type=str, default="output/thermal_v0/video_orbit_final.mp4")
    parser.add_argument("--render_res", type=int, default=1024)
    # 俯视 45 度
    parser.add_argument("--elevation", type=float, default=45.0)
    args = parser.parse_args()

    # 1. Load Geometry
    print("Loading Geometry...")
    ply_path = os.path.join(args.model_path, "point_cloud/iteration_7000/point_cloud.ply")
    cameras_json_path = os.path.join(args.model_path, "cameras.json")

    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)

    gaussians._xyz.requires_grad = False
    for name in ["_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
        getattr(gaussians, name).requires_grad = False

    # 2. Load Thermal Net
    print(f"Loading Thermal Net...")
    thermal_net = ThermalAttrNet(W=16).cuda()
    thermal_net.load_state_dict(torch.load(args.thermal_ckpt))
    thermal_net.eval()

    # 3. Calculate Orientation (Based on Video 1 logic)
    print("Calculating Orientation...")
    device = "cuda"
    xyz = gaussians.get_xyz.detach()
    center = xyz.mean(dim=0)

    xyz_cpu = xyz.cpu()
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)

    pca_normal = eigvecs[:, 0].to(device)

    # --- [关键逻辑修改] ---
    # 你确认 "Video 1" (Auto_PCA_Inverted) 的方向是对的
    # 所以我们锁定使用 -pca_normal 作为 Up 向量
    up_vec = -pca_normal
    up_vec = up_vec / torch.norm(up_vec)

    # 构建旋转平面
    temp = torch.tensor([1.0, 0.0, 0.0], device=device)
    if torch.abs(torch.dot(temp, up_vec)) > 0.9: temp = torch.tensor([0.0, 1.0, 0.0], device=device)
    right_vec = torch.cross(up_vec, temp)
    right_vec = right_vec / torch.norm(right_vec)
    fwd_vec = torch.cross(up_vec, right_vec)  # Forward vector on the plane
    fwd_vec = fwd_vec / torch.norm(fwd_vec)

    # 4. Orbit Settings
    max_dist = torch.max(torch.norm(xyz - center, dim=1)).item()
    orbit_radius = max_dist * 1.5

    with open(cameras_json_path, 'r') as f:
        cams_data = json.load(f)
    ref_cam = cams_data[0]
    fov_y = 2 * math.atan(ref_cam['height'] / (2 * ref_cam['fy']))
    fov_x = fov_y

    pipe = type('Pipe', (object,),
                {"compute_cov3D_python": False, "convert_SHs_python": False, "brdf": False, "brdf_mode": "pbbr"})()
    bg = torch.tensor([0.0, 0.0, 0.0]).cuda()

    print("Pre-calculating Thermal Colors...")
    with torch.no_grad():
        t_vals = thermal_net(gaussians.get_xyz)
        colors = t_vals.repeat(1, 3)

    print(f"Rendering Corrected Orbit (Up looking Skyward)...")

    frames = []
    n_frames = 120
    elevation_rad = math.radians(args.elevation)

    for i in tqdm(range(n_frames)):
        angle = 2 * math.pi * (i / n_frames)

        # --- [关键修正] ---
        # 之前是 + (up_vec * h_up)，导致钻地
        # 现在改成 - (up_vec * h_up)，强制往反方向（天上）飞

        h_up = orbit_radius * math.sin(elevation_rad)
        r_plane = orbit_radius * math.cos(elevation_rad)

        # 相机位置 = 中心 + 水平旋转偏移 - 垂直偏移(反向升空)
        offset = (right_vec * math.cos(angle) * r_plane) + \
                 (fwd_vec * math.sin(angle) * r_plane) - \
                 (up_vec * h_up)

        cam_pos = center + offset

        # LookAt: 保持 up_vec 不变，这样画面里的房子才是正的
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