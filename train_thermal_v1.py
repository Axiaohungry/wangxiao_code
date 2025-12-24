# D:\PycharmProjects\wangxiao_code\train_thermal_v1.py
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


# --- MiniCam 类 (保持不变) ---
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
        intrinsic = torch.tensor([[focal_x, 0, self.image_width / 2], [0, focal_y, self.image_height / 2], [0, 0, 1]],
                                 dtype=torch.float32, device="cuda")
        extrinsic = self.world_view_transform.transpose(0, 1)
        return intrinsic, extrinsic


# --- Helper Functions (保持不变) ---
def get_projection_matrix(znear, zfar, fovX, fovY, device="cuda"):
    tanHalfFovY = math.tan((fovY / 2));
    tanHalfFovX = math.tan((fovX / 2))
    top = tanHalfFovY * znear;
    bottom = -top;
    right = tanHalfFovX * znear;
    left = -right
    P = torch.zeros(4, 4, device=device);
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left);
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left);
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign;
    P[2, 2] = z_sign * zfar / (zfar - znear);
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def look_at_topdown(cam_pos, target, up):
    z_axis = target - cam_pos;
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
    R = torch.stack([x_axis, y_axis, z_axis], dim=0);
    T = -torch.matmul(R, cam_pos)
    w2v = torch.eye(4, device=cam_pos.device);
    w2v[:3, :3] = R;
    w2v[:3, 3] = T
    return w2v.transpose(0, 1).contiguous()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default="output/debug_run")
    parser.add_argument("--gt_path", type=str, default="output/debug_run/lst_gt.png")
    parser.add_argument("--priors_path", type=str, default="output/debug_run/priors.pt")
    parser.add_argument("--output_path", type=str, default="output/thermal_v1")
    parser.add_argument("--iterations", type=int, default=5000)

    # Cam params
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--zoom", type=float, default=5.4)
    parser.add_argument("--shift_x", type=float, default=0.0)
    parser.add_argument("--shift_y", type=float, default=-1.2)
    parser.add_argument("--angle", type=float, default=-31.0)
    parser.add_argument("--multiplier", type=float, default=0.85)
    parser.add_argument("--train_scale", type=float, default=0.5)  # 512x512

    args = parser.parse_args()

    # Check paths
    ply_path = os.path.join(args.model_path, "point_cloud/iteration_7000/point_cloud.ply")
    cameras_json_path = os.path.join(args.model_path, "cameras.json")
    if not os.path.exists(ply_path) or not os.path.exists(args.priors_path):
        print("Error: Missing geometry or priors file.")
        sys.exit(1)

    print("=== Training Thermal Field v1 (Physics Informed) ===")

    # 1. Load Geometry
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)
    # Freeze
    gaussians._xyz.requires_grad = False
    for name in ["_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
        getattr(gaussians, name).requires_grad = False

    # 2. Load Priors
    print(f"Loading Priors from {args.priors_path}...")
    # priors: [N, 5] (Normal*3, Height, Slope)
    priors = torch.load(args.priors_path).cuda()
    print(f"Priors loaded: {priors.shape}")

    # 3. Setup Camera
    with open(cameras_json_path, 'r') as f:
        cams_data = json.load(f)
    ref_cam = cams_data[0]
    fov_y = 2 * math.atan(ref_cam['height'] / (2 * ref_cam['fy']))

    # Cam Geometry
    xyz = gaussians.get_xyz.detach()
    center = xyz.mean(dim=0)

    # Re-calculate axes for top-down view
    xyz_cpu = xyz.cpu();
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)
    normal = eigvecs[:, 0].cuda();
    axis1 = eigvecs[:, 1].cuda()

    cam_centers = [np.array(c['position']) for c in cams_data[:10]]
    mean_cam_pos = torch.tensor(np.mean(cam_centers, axis=0)).float().cuda()
    if torch.dot((mean_cam_pos - center), normal) < 0: normal = -normal

    up_axis = axis1 - torch.dot(axis1, normal) * normal
    up_axis = up_axis / (torch.norm(up_axis) + 1e-8)
    right_axis = torch.cross(up_axis, normal)

    if args.angle != 0.0:
        rad = math.radians(args.angle);
        cos_a = math.cos(rad);
        sin_a = math.sin(rad)
        new_up = up_axis * cos_a + right_axis * sin_a
        up_axis = new_up / torch.norm(new_up)
        right_axis = torch.cross(up_axis, normal)

    # Build Cam
    train_w = int(args.width * args.train_scale)
    train_h = int(args.height * args.train_scale)
    fov_x_mod = 2 * math.atan(math.tan(fov_y / 2) * args.multiplier)

    xyz_min, _ = torch.min(xyz, dim=0);
    xyz_max, _ = torch.max(xyz, dim=0)
    span_x = (xyz_max[0] - xyz_min[0]).item();
    span_y = (xyz_max[1] - xyz_min[1]).item()
    max_span = max(span_x, span_y)

    base_height = (max_span / 2.0) / np.tan(fov_y / 2.0)
    target_height = base_height / args.zoom
    shift_vec = (right_axis * args.shift_x) + (up_axis * args.shift_y)
    target_center = center + shift_vec
    cam_pos = target_center + normal * target_height

    w2v = look_at_topdown(cam_pos, target_center, up_axis)
    proj = get_projection_matrix(0.01, 100.0, fov_x_mod, fov_y).transpose(0, 1)
    full_T = w2v @ proj
    view_cam = MiniCam(train_w, train_h, fov_y, fov_x_mod, 0.01, 100.0, w2v, full_T)

    # 4. Load GT
    gt_img = cv2.imread(args.gt_path, cv2.IMREAD_GRAYSCALE)
    gt_resized = cv2.resize(gt_img, (train_w, train_h), interpolation=cv2.INTER_AREA)
    gt_tensor = torch.from_numpy(gt_resized).float().cuda() / 255.0

    # 5. Network (v1)
    # Input channel = 3 (XYZ) + 5 (Priors) = 8
    thermal_net = ThermalAttrNet(input_ch=8, W=16).cuda()
    optimizer = torch.optim.Adam(thermal_net.parameters(), lr=0.005)
    bg = torch.tensor([0.0, 0.0, 0.0]).cuda()
    pipeline_args = type('Pipe', (object,), {"compute_cov3D_python": False, "convert_SHs_python": False, "brdf": False,
                                             "brdf_mode": "pbbr"})()

    # 6. Training Loop
    os.makedirs(args.output_path, exist_ok=True)
    pbar = tqdm(range(1, args.iterations + 1))

    # 预先拼好 Input Tensor (Frozen)
    # [N, 8]
    full_input = torch.cat([gaussians.get_xyz, priors], dim=1)

    for i in pbar:
        # Forward (with Priors)
        thermal_val = thermal_net(full_input)
        override_color = thermal_val.repeat(1, 3)

        out = render(view_cam, gaussians, pipeline_args, bg, override_color=override_color)["render"]
        pred = out[0, :, :]

        loss = torch.abs(pred - gt_tensor).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0: pbar.set_description(f"Loss: {loss.item():.5f}")

        if i % 500 == 0 or i == args.iterations:
            vis = pred.detach().cpu().numpy()
            vis = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
            vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.output_path, f"train_{i:04d}.png"), vis_color)

    torch.save(thermal_net.state_dict(), os.path.join(args.output_path, "thermal_net_v1.pth"))
    print("Training v1 Done!")