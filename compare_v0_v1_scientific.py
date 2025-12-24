import torch
import os
import cv2
import numpy as np
import json
import math
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.thermal_network import ThermalAttrNet
from argparse import ArgumentParser
import sys


# --- Helper: Torch Pearson Correlation ---
def torch_corr(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return cost.item()


# --- MiniCam ---
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width;
        self.image_height = height
        self.FoVy = fovy;
        self.FoVx = fovx;
        self.znear = znear;
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.tanfovx = math.tan(self.FoVx * 0.5);
        self.tanfovy = math.tan(self.FoVy * 0.5)

    def get_calib_matrix_nerf(self):
        focal_y = self.image_height / (2.0 * self.tanfovy)
        focal_x = self.image_width / (2.0 * self.tanfovx)
        intrinsic = torch.tensor([[focal_x, 0, self.image_width / 2], [0, focal_y, self.image_height / 2], [0, 0, 1]],
                                 dtype=torch.float32, device="cuda")
        extrinsic = self.world_view_transform.transpose(0, 1)
        return intrinsic, extrinsic


def get_look_at(cam_pos, target, up):
    z_axis = target - cam_pos;
    dist = torch.norm(z_axis)
    if dist < 1e-5: return torch.eye(4, device="cuda")
    z_axis = z_axis / dist
    x_axis = torch.cross(up, z_axis, dim=0)
    if torch.norm(x_axis) < 1e-5:
        temp = torch.tensor([1.0, 0.0, 0.0], device="cuda")
        if torch.abs(torch.dot(temp, z_axis)) > 0.9: temp = torch.tensor([0.0, 1.0, 0.0], device="cuda")
        x_axis = torch.cross(temp, z_axis, dim=0)
    x_axis = x_axis / torch.norm(x_axis)
    y_axis = torch.cross(z_axis, x_axis, dim=0);
    y_axis = y_axis / torch.norm(y_axis)
    R = torch.stack([x_axis, y_axis, z_axis], dim=0);
    T = -torch.matmul(R, cam_pos)
    w2v = torch.eye(4, device="cuda");
    w2v[:3, :3] = R;
    w2v[:3, 3] = T
    return w2v.transpose(0, 1).contiguous()


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


# --- [核心] 暴力替换纹理函数 ---
def force_apply_texture(gaussians, color_tensor):
    """
    暴力修改高斯内部数据，将 Thermal Color 转换为 0阶 SH 系数并覆盖。
    """
    # 1. 颜色转 SH 系数 (逆运算)
    # SH = (RGB - 0.5) / C0
    C0 = 0.28209479177387814
    sh_dc = (color_tensor - 0.5) / C0

    # 2. 暴力覆盖 DC (直流分量)
    # 形状: [N, 1, 3]
    gaussians._features_dc = sh_dc.unsqueeze(1).contiguous()

    # 3. [修复点] 清空高阶 SH
    # 不能设为 None，必须设为一个“维度为0”的空 Tensor，否则 torch.cat 会崩
    # 形状: [N, 0, 3]
    gaussians._features_rest = torch.zeros((sh_dc.shape[0], 0, 3), device=sh_dc.device)

    # 4. 标记为 active_sh_degree = 0
    gaussians.active_sh_degree = 0


def main():
    # 你的默认对齐参数
    SHIFT_X = 0.0
    SHIFT_Y = -1.2
    ZOOM = 5.4
    ANGLE = -31.0

    base_dir = "output/debug_run"
    ckpt_v0 = "output/thermal_v0/thermal_net_v0.pth"
    ckpt_v1 = "output/thermal_v1/thermal_net_v1.pth"
    priors_path = os.path.join(base_dir, "priors.pt")

    print("=== 科学验证 (暴力替换版): v0 vs v1 ===")

    # 1. 加载几何
    print("1. 加载几何与先验...")
    ply_path = os.path.join(base_dir, "point_cloud/iteration_7000/point_cloud.ply")
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)

    # Freeze
    gaussians._xyz.requires_grad = False
    for name in ["_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
        getattr(gaussians, name).requires_grad = False

    xyz = gaussians.get_xyz.cuda()
    priors = torch.load(priors_path).cuda()
    height_prior = priors[:, 3]

    # 2. 计算 ROI 中心
    device = "cuda"
    center = xyz.mean(dim=0)

    xyz_cpu = xyz.cpu();
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)
    pca_normal = eigvecs[:, 0].to(device)
    axis1 = eigvecs[:, 1].to(device)

    cameras_json_path = os.path.join(base_dir, "cameras.json")
    with open(cameras_json_path, 'r') as f:
        cams_data = json.load(f)
    cam_centers = [np.array(c['position']) for c in cams_data[:10]]
    mean_cam_pos = torch.tensor(np.mean(cam_centers, axis=0)).float().cuda()
    if torch.dot((mean_cam_pos - center), pca_normal) < 0: pca_normal = -pca_normal

    up_axis_world = -pca_normal
    if torch.norm(up_axis_world) == 0: up_axis_world = torch.tensor([0, 0, 1.0]).cuda()
    up_axis_world = up_axis_world / torch.norm(up_axis_world)

    right_axis = torch.cross(up_axis_world, axis1)
    temp_fwd = torch.cross(up_axis_world, right_axis)

    if ANGLE != 0.0:
        rad = math.radians(ANGLE)
        cos_a = math.cos(rad);
        sin_a = math.sin(rad)
        new_right = right_axis * cos_a + temp_fwd * sin_a
        new_fwd = -right_axis * sin_a + temp_fwd * cos_a
        right_axis = new_right
        temp_fwd = new_fwd

    shift_vec = (right_axis * SHIFT_X) + (temp_fwd * SHIFT_Y)
    target_center = center + shift_vec

    # 3. 加载网络
    print("3. 加载网络...")
    net_v0 = ThermalAttrNet(input_ch=3, W=16).cuda()
    net_v0.load_state_dict(torch.load(ckpt_v0))
    net_v1 = ThermalAttrNet(input_ch=8, W=16).cuda()
    net_v1.load_state_dict(torch.load(ckpt_v1))

    # 4. 预测温度 (Tensor Level)
    print("   计算全图温度 Tensor...")
    with torch.no_grad():
        temp_v0 = net_v0(xyz).squeeze()
        full_input = torch.cat([xyz, priors], dim=1)
        temp_v1 = net_v1(full_input).squeeze()

    # 验证 Tensor 是否不同
    diff_tensor = torch.abs(temp_v0 - temp_v1).mean()
    print(f"   [Debug] Tensor L1 Diff: {diff_tensor:.6f} (如果不为0，说明网络确实不同)")

    # 5. 渲染准备
    xyz_min, _ = torch.min(xyz, dim=0);
    xyz_max, _ = torch.max(xyz, dim=0)
    scene_radius = torch.norm(xyz_max - xyz_min) * 0.5
    dist = scene_radius * 0.5

    cam_offset = (up_axis_world * dist) + (temp_fwd * dist)
    cam_pos = target_center + cam_offset

    w2v = get_look_at(cam_pos, target_center, up_axis_world)
    proj = get_projection_matrix(0.1, 1000.0, 0.9, 0.9).transpose(0, 1)
    full_T = w2v @ proj
    cam = MiniCam(1024, 1024, 0.9, 0.9, 0.1, 1000.0, w2v, full_T)

    pipe = type('Pipe', (object,),
                {"compute_cov3D_python": False, "convert_SHs_python": False, "brdf": False, "brdf_mode": "pbbr"})()
    bg = torch.tensor([0.0, 0.0, 0.0]).cuda()

    # --- [A] 渲染 v0 (暴力替换) ---
    print("   渲染 v0...")
    color_v0 = temp_v0.unsqueeze(1).repeat(1, 3)
    # 暴力修改高斯内部数据
    force_apply_texture(gaussians, color_v0)
    # 此时不需要传 override_color，因为我们已经改了模型本身
    img_v0 = render(cam, gaussians, pipe, bg)["render"]
    img_v0 = img_v0.detach().permute(1, 2, 0).cpu().numpy()

    # --- [B] 渲染 v1 (暴力替换) ---
    print("   渲染 v1...")
    color_v1 = temp_v1.unsqueeze(1).repeat(1, 3)
    # 暴力修改高斯内部数据
    force_apply_texture(gaussians, color_v1)
    img_v1 = render(cam, gaussians, pipe, bg)["render"]
    img_v1 = img_v1.detach().permute(1, 2, 0).cpu().numpy()

    # Debug Info
    print(f"   [Debug] v0 Stats: Min={img_v0.min():.4f}, Max={img_v0.max():.4f}, Mean={img_v0.mean():.4f}")
    print(f"   [Debug] v1 Stats: Min={img_v1.min():.4f}, Max={img_v1.max():.4f}, Mean={img_v1.mean():.4f}")

    # 差异图
    diff = np.abs(img_v1 - img_v0)
    diff_mean = np.mean(diff, axis=2)

    # Auto Contrast
    if diff_mean.max() > diff_mean.min():
        diff_norm = (diff_mean - diff_mean.min()) / (diff_mean.max() - diff_mean.min())
        print(f"   [Diff Range] {diff_mean.min():.6f} - {diff_mean.max():.6f}")
    else:
        diff_norm = diff_mean
        print("   [Warning] 依然完全一样...见鬼了")

    out_dir = "output/comparison_final"
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(f"{out_dir}/roi_side_v0.png",
                cv2.cvtColor((np.clip(img_v0, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{out_dir}/roi_side_v1.png",
                cv2.cvtColor((np.clip(img_v1, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    diff_color = cv2.applyColorMap((diff_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(f"{out_dir}/roi_diff.png", diff_color)

    print(f"\n对比完成: {out_dir}")


if __name__ == "__main__":
    main()