# D:\PycharmProjects\wangxiao_code\render_top_down.py
import os
import torch
import numpy as np
import math
import cv2
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args


# --- 1. 标准化的 MiniCam (将被训练脚本复用) ---
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


@torch.no_grad()
def main(dataset, pipe, args):
    print(f"[1] Loading Scene... (Res: {args.resolution})")
    dataset.sh_degree = 0
    gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, pipe.brdf_mode, dataset.brdf_envmap_res,
                              dataset.feature_time)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    train_cams = scene.getTrainCameras()
    ref_cam = train_cams[0]
    device = ref_cam.world_view_transform.device

    # --- 几何计算 ---
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

    mean_cam_pos = torch.stack([c.camera_center for c in train_cams]).mean(dim=0).to(device)
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

    # --- 构造最终相机 ---
    print(f"[2] Rendering Single View | Multiplier: {args.multiplier} | Shift X: {args.shift_x}")

    fov_y = ref_cam.FoVy
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

    custom_cam = MiniCam(
        width=args.width,
        height=args.height,
        fovy=fov_y,
        fovx=fov_x_mod,
        znear=znear,
        zfar=zfar,
        world_view_transform=w2v_T,
        full_proj_transform=full_T
    )

    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    out = render(custom_cam, gaussians, pipe, bg, scaling_modifier=1.0)
    img = out["render"]

    os.makedirs(args.model_path, exist_ok=True)
    img_np = img.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 文件名固定，方便后续查找
    filename = f"topdown_final.png"
    save_path = os.path.join(args.model_path, filename)
    cv2.imwrite(save_path, img_np)
    print(f"    -> Saved Final Render: {save_path}")
    print(
        f"    [MEMO] Use these params for training: --multiplier {args.multiplier} --shift_x {args.shift_x} --shift_y {args.shift_y}")


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=7000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)

    # 你的默认参数
    parser.add_argument("--zoom", type=float, default=5.4)
    parser.add_argument("--shift_x", type=float, default=1.0)
    parser.add_argument("--shift_y", type=float, default=-1.2)
    parser.add_argument("--angle", type=float, default=-31.0)

    # 新增：比例修正系数
    parser.add_argument("--multiplier", type=float, default=0.85, help="Aspect ratio corrector")

    args = get_combined_args(parser)
    dataset = model.extract(args)
    if not hasattr(dataset, "brdf_dim"): dataset.brdf_dim = -1
    if not hasattr(dataset, "sh_degree"): dataset.sh_degree = 0
    dataset.feature_time = False
    pipe = pipeline.extract(args)
    main(dataset, pipe, args)