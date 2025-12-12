# D:\PycharmProjects\wangxiao_code\train_thermal_v0.py
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
# 1. 核心相机类
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


def look_at_topdown(cam_pos, target, up):
    z_axis = target - cam_pos
    dist = torch.norm(z_axis)
    if dist < 1e-5: return torch.eye(4, device=cam_pos.device)
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


# ==============================================================================
# 2. 训练主逻辑
# ==============================================================================
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model_path", "-m", type=str, default="output/debug_run",
                        help="Trained 3DGS model path")
    parser.add_argument("--gt_path", type=str, default="output/debug_run/lst_gt.png",
                        help="Path to aligned LST GT png")
    parser.add_argument("--output_path", type=str, default="output/thermal_v0")
    parser.add_argument("--iterations", type=int, default=5000)

    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--zoom", type=float, default=5.4)
    parser.add_argument("--shift_x", type=float, default=0.0)
    parser.add_argument("--shift_y", type=float, default=-1.2)
    parser.add_argument("--angle", type=float, default=-31.0)
    parser.add_argument("--multiplier", type=float, default=0.85)

    parser.add_argument("--train_scale", type=float, default=0.5)

    args = parser.parse_args()

    # 路径检查
    ply_path = os.path.join(args.model_path, "point_cloud/iteration_7000/point_cloud.ply")
    cameras_json_path = os.path.join(args.model_path, "cameras.json")

    if not os.path.exists(ply_path):
        print(f"[Error] PLY not found: {ply_path}")
        sys.exit(1)
    if not os.path.exists(cameras_json_path):
        print(f"[Error] cameras.json not found: {cameras_json_path}")
        sys.exit(1)
    if not os.path.exists(args.gt_path):
        print(f"[Error] GT Image not found: {args.gt_path}")
        sys.exit(1)

    print("-------------------------------------------------")
    print("LIGHTWEIGHT MODE: Loading only Geometry & Metadata")
    print("-------------------------------------------------")

    # 1. 加载几何
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)

    gaussians._xyz.requires_grad = False
    gaussians._features_dc.requires_grad = False
    gaussians._features_rest.requires_grad = False
    gaussians._opacity.requires_grad = False
    gaussians._scaling.requires_grad = False
    gaussians._rotation.requires_grad = False

    print(f"Loaded {gaussians.get_xyz.shape[0]} points.")

    # 2. 读取元数据
    with open(cameras_json_path, 'r') as f:
        cams_data = json.load(f)

    ref_cam_data = cams_data[0]
    h_ref = ref_cam_data['height']
    fy_ref = ref_cam_data['fy']
    fov_y = 2 * math.atan(h_ref / (2 * fy_ref))

    # 计算平均相机位置
    cam_centers = []
    for c in cams_data[:10]:
        pos = np.array(c['position'])
        cam_centers.append(pos)
    mean_cam_pos = torch.tensor(np.mean(cam_centers, axis=0)).float().cuda()

    # 3. 几何计算
    device = "cuda"
    xyz = gaussians.get_xyz.detach()
    center = xyz.mean(dim=0)
    min_xyz, _ = torch.min(xyz, dim=0)
    max_xyz, _ = torch.max(xyz, dim=0)
    span_x = (max_xyz[0] - min_xyz[0]).item()
    span_y = (max_xyz[1] - min_xyz[1]).item()
    max_span = max(span_x, span_y)

    xyz_cpu = xyz.detach().cpu()
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)
    normal = eigvecs[:, 0].to(device)
    axis1 = eigvecs[:, 1].to(device)

    if torch.dot((mean_cam_pos - center), normal) < 0:
        normal = -normal

    up_axis = axis1 - torch.dot(axis1, normal) * normal
    up_axis = up_axis / (torch.norm(up_axis) + 1e-8)
    right_axis = torch.cross(up_axis, normal)

    if args.angle != 0.0:
        rad = math.radians(args.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        new_up = up_axis * cos_a + right_axis * sin_a
        up_axis = new_up / torch.norm(new_up)
        right_axis = torch.cross(up_axis, normal)

    # 4. 构造相机
    train_w = int(args.width * args.train_scale)
    train_h = int(args.height * args.train_scale)
    print(f"Target Resolution: {train_w}x{train_h} (Scale: {args.train_scale})")

    fov_x_mod = 2 * math.atan(math.tan(fov_y / 2) * args.multiplier)

    base_height = (max_span / 2.0) / np.tan(fov_y / 2.0)
    target_height = base_height / args.zoom
    shift_vec = (right_axis * args.shift_x) + (up_axis * args.shift_y)
    target_center = center + shift_vec
    cam_pos = target_center + normal * target_height

    w2v_T = look_at_topdown(cam_pos, target_center, up_axis)
    znear, zfar = 0.01, 100.0
    P = get_projection_matrix(znear, zfar, fov_x_mod, fov_y, device=device).transpose(0, 1)
    full_T = w2v_T @ P

    view_cam = MiniCam(train_w, train_h, fov_y, fov_x_mod, znear, zfar, w2v_T, full_T)

    # 5. 读取 GT
    print(f"Reading GT: {args.gt_path}")
    gt_img = cv2.imread(args.gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        raise ValueError(f"Failed to read GT image from {args.gt_path}")

    gt_img_resized = cv2.resize(gt_img, (train_w, train_h), interpolation=cv2.INTER_AREA)
    gt_tensor = torch.from_numpy(gt_img_resized).float().cuda() / 255.0

    # 6. 训练准备
    thermal_net = ThermalAttrNet().cuda()
    optimizer = torch.optim.Adam(thermal_net.parameters(), lr=0.005)

    bg = torch.tensor([0.0, 0.0, 0.0]).cuda()

    # --- [修复] 构造符合 NTR-Gaussian 要求的伪造 Pipeline ---
    # 必须包含 brdf, compute_cov3D_python 等属性
    pipeline_args = type('Pipe', (object,), {
        "compute_cov3D_python": False,
        "convert_SHs_python": False,
        "brdf": False,  # 关键修复：关闭 BRDF
        "brdf_mode": "pbbr"  # 占位
    })()

    print("Starting Thermal Training...")
    os.makedirs(args.output_path, exist_ok=True)

    pbar = tqdm(range(1, args.iterations + 1))
    for i in pbar:
        xyz_in = gaussians.get_xyz
        thermal_val = thermal_net(xyz_in)
        override_color = thermal_val.repeat(1, 3)

        # 传入修复后的 pipeline_args
        render_pkg = render(view_cam, gaussians, pipeline_args, bg, override_color=override_color)
        pred_img = render_pkg["render"]

        pred_thermal = pred_img[0, :, :]
        loss = torch.abs(pred_thermal - gt_tensor).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0:
            pbar.set_description(f"Loss: {loss.item():.5f}")

        if i % 500 == 0 or i == args.iterations:
            vis = pred_thermal.detach().cpu().numpy()
            vis = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
            vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.output_path, f"train_{i:04d}.png"), vis_color)

    torch.save(thermal_net.state_dict(), os.path.join(args.output_path, "thermal_net_v0.pth"))
    print(f"Training Done! Results saved to {args.output_path}")