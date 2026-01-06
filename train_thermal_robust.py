# D:\PycharmProjects\wangxiao_code\train_thermal_robust.py
import os
import sys
import json
import math
import cv2
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.thermal_network import ThermalAttrNet


# -------------------------
# 1) MiniCam + projection (保持与 render_top_down.py 一致)
# -------------------------
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


# -------------------------
# 2) Fix Risk D: 强制覆盖 SH DC（保梯度）
# -------------------------
def apply_sh0_from_rgb(gaussians: GaussianModel, rgb01: torch.Tensor, empty_rest: torch.Tensor):
    """
    rgb01: [N, 3], range [0,1], requires_grad True
    通过覆盖 _features_dc/_features_rest 保证渲染一定使用网络预测颜色。
    """
    C0 = 0.28209479177387814
    sh_dc = (rgb01 - 0.5) / C0  # [N,3]
    gaussians._features_dc = sh_dc.unsqueeze(1).contiguous()   # [N,1,3], 保留梯度链
    gaussians._features_rest = empty_rest                      # [N,0,3], 无条件清空
    gaussians.active_sh_degree = 0


def safe_get_alpha(render_out: dict):
    """
    尝试从渲染结果里拿 alpha/accumulation（不同分支命名可能不一样）。
    返回 shape [H,W] 的 tensor 或 None。
    """
    for k in ["alpha", "accumulation", "accum", "A"]:
        if k in render_out:
            a = render_out[k]
            if isinstance(a, torch.Tensor):
                # 常见 shape: [1,H,W] 或 [H,W]
                if a.dim() == 3:
                    return a[0]
                if a.dim() == 2:
                    return a
    return None


# -------------------------
# 3) 主流程
# -------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default="output/debug_run")
    parser.add_argument("--gt_path", type=str, default="output/debug_run/lst_gt.png")
    parser.add_argument("--priors_path", type=str, default="output/debug_run/priors.pt")
    parser.add_argument("--output_path", type=str, default="output/thermal_robust")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-3)

    # cam params（务必与你生成 topdown_final + manual_align 时一致）
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--zoom", type=float, default=5.4)
    parser.add_argument("--shift_x", type=float, default=0.0)
    parser.add_argument("--shift_y", type=float, default=-1.2)
    parser.add_argument("--angle", type=float, default=-31.0)
    parser.add_argument("--multiplier", type=float, default=0.85)
    parser.add_argument("--train_scale", type=float, default=0.5)  # 2048*0.5=1024

    # mask options
    parser.add_argument("--mask_mode", type=str, default="gt+alpha",
                        help="gt | alpha | gt+alpha | none")
    parser.add_argument("--gt_eps", type=float, default=1.0/255.0,
                        help="GT threshold when mask_mode contains gt")
    parser.add_argument("--alpha_eps", type=float, default=1e-3,
                        help="Alpha threshold when mask_mode contains alpha")

    # misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--grad_every", type=int, default=100)
    parser.add_argument("--fail_grad_eps", type=float, default=1e-10)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ply_path = os.path.join(args.model_path, "point_cloud/iteration_7000/point_cloud.ply")
    cameras_json_path = os.path.join(args.model_path, "cameras.json")
    if not os.path.exists(ply_path):
        print(f"[Error] PLY not found: {ply_path}")
        sys.exit(1)
    if not os.path.exists(args.priors_path):
        print(f"[Error] priors not found: {args.priors_path} (run bake_priors_physics.py first)")
        sys.exit(1)
    if not os.path.exists(args.gt_path):
        print(f"[Error] GT not found: {args.gt_path}")
        sys.exit(1)

    print("=== Robust Thermal Training (Fix Risk D & C) ===")

    # --- Load Geometry ---
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)

    # freeze geometry parameters
    gaussians._xyz.requires_grad = False
    for name in ["_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
        if hasattr(gaussians, name) and getattr(gaussians, name) is not None:
            getattr(gaussians, name).requires_grad = False
    gaussians.active_sh_degree = 0

    xyz = gaussians.get_xyz.detach()  # [N,3]
    center = xyz.mean(dim=0)
    min_xyz, _ = torch.min(xyz, dim=0)
    max_xyz, _ = torch.max(xyz, dim=0)
    span_x = (max_xyz[0] - min_xyz[0]).item()
    span_y = (max_xyz[1] - min_xyz[1]).item()
    max_span = max(span_x, span_y)

    # --- Load Priors ---
    priors = torch.load(args.priors_path).cuda()
    if priors.dim() != 2 or priors.shape[0] != xyz.shape[0]:
        print(f"[Error] priors shape mismatch: priors={priors.shape}, xyz={xyz.shape}")
        sys.exit(1)
    print(f"Priors loaded: {priors.shape}")

    # --- Camera setup (match render_top_down style) ---
    with open(cameras_json_path, "r") as f:
        cams_data = json.load(f)
    ref_cam = cams_data[0]
    # 与你旧脚本一致：从 fy/height 推 fovy
    fov_y = 2 * math.atan(ref_cam["height"] / (2 * ref_cam["fy"]))
    fov_x_mod = 2 * math.atan(math.tan(fov_y / 2) * args.multiplier)

    # PCA (CPU) - 与旧脚本保持一致；这里不改你的逻辑
    xyz_cpu = xyz.detach().cpu()
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)
    normal = eigvecs[:, 0].cuda()
    axis1 = eigvecs[:, 1].cuda()

    cam_centers = [np.array(c["position"]) for c in cams_data[:10]]
    mean_cam_pos = torch.tensor(np.mean(cam_centers, axis=0)).float().cuda()
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
        up_axis = new_up / (torch.norm(new_up) + 1e-8)
        right_axis = torch.cross(up_axis, normal)

    # build cam (关键：base_height 用 max_span，匹配 render_top_down)
    train_w = int(args.width * args.train_scale)
    train_h = int(args.height * args.train_scale)

    base_height = (max_span / 2.0) / np.tan(fov_y / 2.0)
    target_height = base_height / args.zoom
    shift_vec = (right_axis * args.shift_x) + (up_axis * args.shift_y)
    target_center = center + shift_vec
    cam_pos = target_center + normal * target_height

    w2v = look_at_topdown(cam_pos, target_center, up_axis)
    proj = get_projection_matrix(0.01, 100.0, fov_x_mod, fov_y).transpose(0, 1)
    full_T = w2v @ proj
    view_cam = MiniCam(train_w, train_h, fov_y, fov_x_mod, 0.01, 100.0, w2v, full_T)

    # --- Load GT ---
    gt_img = cv2.imread(args.gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        raise ValueError("GT not readable.")
    gt_resized = cv2.resize(gt_img, (train_w, train_h), interpolation=cv2.INTER_AREA)
    gt_tensor = torch.from_numpy(gt_resized).float().cuda() / 255.0  # [H,W]

    # --- Prepare mask (Fix Risk C) ---
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    pipeline_args = type("Pipe", (object,), {
        "compute_cov3D_python": False,
        "convert_SHs_python": False,
        "brdf": False,
        "brdf_mode": "pbbr",
    })()

    valid_mask = torch.ones_like(gt_tensor)

    if args.mask_mode.lower() != "none":
        if "gt" in args.mask_mode.lower():
            gt_mask = (gt_tensor > args.gt_eps).float()
            valid_mask = valid_mask * gt_mask

        if "alpha" in args.mask_mode.lower():
            with torch.no_grad():
                # 用当前 gaussians 原始颜色渲染一次拿 alpha（颜色不影响 alpha）
                tmp = render(view_cam, gaussians, pipeline_args, bg)
                a = safe_get_alpha(tmp)
                if a is None:
                    print("[Warn] render_out has no alpha/accumulation key. alpha-mask skipped.")
                else:
                    alpha_mask = (a > args.alpha_eps).float()
                    valid_mask = valid_mask * alpha_mask

    coverage = (valid_mask.sum() / valid_mask.numel()).item() * 100.0
    print(f"Valid Mask Coverage: {coverage:.2f}%")
    if coverage < 1.0:
        print("[FATAL] Mask coverage too small. Check gt/mask_mode/camera params.")
        sys.exit(1)

    # 保存 mask 方便你肉眼检查
    os.makedirs(args.output_path, exist_ok=True)
    mask_vis = (valid_mask.detach().cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_path, "valid_mask.png"), mask_vis)

    # --- Network ---
    thermal_net = ThermalAttrNet(input_ch=8, W=16).cuda()
    thermal_net.train()
    optimizer = torch.optim.Adam(thermal_net.parameters(), lr=args.lr)

    # 输入：XYZ + priors
    # 建议做一个轻量归一化（只影响优化稳定性，不改变渲染几何）
    xyz_norm = (gaussians.get_xyz.detach() - center) / (max_span + 1e-6)
    full_input = torch.cat([xyz_norm, priors], dim=1).contiguous()

    # 空 rest（一次性创建，避免循环反复分配）
    empty_rest = torch.empty((xyz.shape[0], 0, 3), device="cuda", dtype=torch.float32)

    # --- Sanity check: 常数颜色渲染差异，验证强制改色真的生效 ---
    with torch.no_grad():
        c1 = torch.full((xyz.shape[0], 3), 0.1, device="cuda")
        apply_sh0_from_rgb(gaussians, c1, empty_rest)
        r1 = render(view_cam, gaussians, pipeline_args, bg)["render"][0].mean().item()

        c2 = torch.full((xyz.shape[0], 3), 0.9, device="cuda")
        apply_sh0_from_rgb(gaussians, c2, empty_rest)
        r2 = render(view_cam, gaussians, pipeline_args, bg)["render"][0].mean().item()

    print(f"[Sanity] ConstColor mean: 0.1 -> {r1:.4f}, 0.9 -> {r2:.4f}, diff={abs(r2-r1):.4f}")
    if abs(r2 - r1) < 0.05:
        print("[FATAL] Force-apply texture seems ineffective. Renderer may not be using _features_dc.")
        sys.exit(1)

    # --- Train ---
    pbar = tqdm(range(1, args.iterations + 1))
    for it in pbar:
        thermal_val = thermal_net(full_input)          # [N,1]
        rgb = thermal_val.expand(-1, 3)                # [N,3]，不额外分配大内存

        apply_sh0_from_rgb(gaussians, rgb, empty_rest)

        out = render(view_cam, gaussians, pipeline_args, bg)
        pred = out["render"][0]                        # [H,W]

        diff = (pred - gt_tensor).abs() * valid_mask
        loss = diff.sum() / (valid_mask.sum() + 1e-6)

        if not torch.isfinite(loss):
            print("[FATAL] Loss is NaN/Inf. Stop.")
            sys.exit(1)

        loss.backward()

        # 梯度监测（更鲁棒）
        if it % args.grad_every == 0:
            g1 = thermal_net.layers[0].weight.grad
            g2 = thermal_net.out.weight.grad
            if (g1 is None) or (g2 is None):
                print("\n[FATAL] Gradient is None (graph disconnected).")
                sys.exit(1)
            g1m = g1.abs().mean().item()
            g2m = g2.abs().mean().item()
            if (g1m < args.fail_grad_eps) and (g2m < args.fail_grad_eps):
                print(f"\n[FATAL] Gradient too small: g1={g1m:.3e}, g2={g2m:.3e}")
                sys.exit(1)
            print(f"\n[Grad] it={it} g1={g1m:.3e} g2={g2m:.3e}")

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if it % 10 == 0:
            pbar.set_description(f"Loss: {loss.item():.6f}")

        if it % args.save_every == 0 or it == args.iterations:
            vis = pred.detach().clamp(0, 1).cpu().numpy()
            vis_u8 = (vis * 255).astype(np.uint8)
            vis_color = cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.output_path, f"train_{it:04d}.png"), vis_color)

    torch.save(thermal_net.state_dict(), os.path.join(args.output_path, "thermal_net_robust.pth"))
    print("Training Done!")


if __name__ == "__main__":
    main()
# 相机几何严格对齐 render_top_down.py（max_span 计算、base_height 公式一致）。
#
# GT resize 用 INTER_AREA；mask 支持 gt/alpha/gt+alpha/none。
#
# 强制改 SH：无条件清空 _features_rest，并加入常数色差 sanity test。
#
# 梯度检查更鲁棒：同时检查第一层和输出层，区分 None 与“很小”。
#
# 输入对 xyz 做轻量归一化，提升稳定性（不影响渲染几何）。