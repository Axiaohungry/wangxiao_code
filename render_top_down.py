# D:\PycharmProjects\wangxiao_code\render_top_down.py
import os
import copy
import cv2
import torch
import numpy as np
import math
from argparse import ArgumentParser

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args


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
    print(
        f"[1] Loading Scene | Zoom: {args.zoom}x | Shift: ({args.shift_x}m, {args.shift_y}m) | Angle: {args.angle} deg")
    dataset.sh_degree = 0
    gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, pipe.brdf_mode, dataset.brdf_envmap_res,
                              dataset.feature_time)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    train_cams = scene.getTrainCameras()
    ref_cam = train_cams[0]
    device = ref_cam.world_view_transform.device

    # --- 2. 几何计算 (PCA) ---
    xyz = gaussians.get_xyz.detach()
    center = xyz.mean(dim=0)
    min_xyz, _ = torch.min(xyz, dim=0)
    max_xyz, _ = torch.max(xyz, dim=0)

    span_x = (max_xyz[0] - min_xyz[0]).item()
    span_y = (max_xyz[1] - min_xyz[1]).item()
    max_span = max(span_x, span_y)

    # PCA 法线
    xyz_cpu = xyz.detach().cpu()
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)

    normal = eigvecs[:, 0].to(device)
    axis1 = eigvecs[:, 1].to(device)

    # 方向矫正
    mean_cam_pos = torch.stack([c.camera_center for c in train_cams]).mean(dim=0).to(device)
    if torch.dot((mean_cam_pos - center), normal) < 0:
        normal = -normal

    # 构建初始正交基
    up_axis = axis1 - torch.dot(axis1, normal) * normal
    up_axis = up_axis / (torch.norm(up_axis) + 1e-8)
    right_axis = torch.cross(up_axis, normal)

    # --- [新增] 应用旋转 ---
    if args.angle != 0.0:
        rad = math.radians(args.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        # 在切平面上旋转 Up 向量
        # new_up = up * cos + right * sin
        new_up = up_axis * cos_a + right_axis * sin_a
        up_axis = new_up / torch.norm(new_up)
        # 更新 right 轴
        right_axis = torch.cross(up_axis, normal)

    # --- 3. 聚焦计算 ---
    fov_y = ref_cam.FoVy
    base_height = (max_span / 2.0) / np.tan(fov_y / 2.0)

    target_height = base_height / args.zoom

    # 应用平移 (基于旋转后的坐标轴移动)
    shift_vec = (right_axis * args.shift_x) + (up_axis * args.shift_y)
    target_center = center + shift_vec
    cam_pos = target_center + normal * target_height

    # --- 4. 渲染 ---
    print(f"[4] Rendering {args.width}x{args.height} ...")

    w2v_T = look_at_topdown(cam_pos, target_center, up_axis)

    W_ref = ref_cam.world_view_transform
    F_ref = ref_cam.full_proj_transform
    P_ref = torch.inverse(W_ref) @ F_ref

    full_T = w2v_T @ P_ref

    top_cam = copy.deepcopy(ref_cam)
    top_cam.world_view_transform = w2v_T
    top_cam.full_proj_transform = full_T
    top_cam.camera_center = torch.inverse(w2v_T)[3, :3]
    top_cam.image_width = args.width
    top_cam.image_height = args.height

    bg = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device)

    out = render(top_cam, gaussians, pipe, bg, scaling_modifier=1.0)
    img = out["render"]

    os.makedirs(args.model_path, exist_ok=True)
    img_np = img.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 文件名记录所有参数，方便后续训练时查阅
    filename = f"align_Z{args.zoom}_X{int(args.shift_x)}_Y{int(args.shift_y)}_A{int(args.angle)}.png"
    save_path = os.path.join(args.model_path, filename)
    cv2.imwrite(save_path, img_np)
    print(f"    -> Saved: {save_path}")
    print(
        f"    [IMPORTANT] Please remember these params for training: Zoom={args.zoom}, X={args.shift_x}, Y={args.shift_y}, Angle={args.angle}")


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=7000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # 核心参数
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--zoom", type=float, default=5.4)
    parser.add_argument("--shift_x", type=float, default=1.0)
    parser.add_argument("--shift_y", type=float, default=-1.2)
    parser.add_argument("--angle", type=float, default=-31.0, help="旋转角度(度)，正数为逆时针")

    args = get_combined_args(parser)
    dataset = model.extract(args)
    if not hasattr(dataset, "brdf_dim"): dataset.brdf_dim = -1
    if not hasattr(dataset, "sh_degree"): dataset.sh_degree = 0
    dataset.feature_time = False

    pipe = pipeline.extract(args)
    main(dataset, pipe, args)
