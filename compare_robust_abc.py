# D:\PycharmProjects\wangxiao_code\compare_robust_abc.py
import os
import sys
import json
import math
import hashlib
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

# 允许从任意工作目录运行
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.thermal_network import ThermalAttrNet


# -----------------------------
# Camera utilities (match your robust training / render_top_down logic)
# -----------------------------
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
        intrinsic = torch.tensor(
            [[focal_x, 0, self.image_width / 2.0],
             [0, focal_y, self.image_height / 2.0],
             [0, 0, 1]],
            dtype=torch.float32, device=self.world_view_transform.device
        )
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


def look_at(cam_pos, target, up):
    # robust look-at (dim specified to avoid warnings)
    z_axis = target - cam_pos
    dist = torch.norm(z_axis)
    if dist < 1e-6:
        return torch.eye(4, device=cam_pos.device)
    z_axis = z_axis / (dist + 1e-8)

    x_axis = torch.cross(up, z_axis, dim=0)
    if torch.norm(x_axis) < 1e-6:
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


def look_at_topdown(cam_pos, target, up):
    # keep the same function name as your training scripts
    return look_at(cam_pos, target, up)


# -----------------------------
# Force-apply texture (same idea as compare_v0_v1_scientific.py)
# -----------------------------
def force_apply_texture(gaussians: GaussianModel, rgb01: torch.Tensor):
    """
    rgb01: [N, 3] in [0,1]
    Convert to SH0 and overwrite gaussians internal buffers.
    """
    C0 = 0.28209479177387814
    sh_dc = (rgb01 - 0.5) / C0                      # [N,3]
    gaussians._features_dc = sh_dc.unsqueeze(1).contiguous()  # [N,1,3]
    gaussians._features_rest = torch.zeros((sh_dc.shape[0], 0, 3), device=sh_dc.device)
    gaussians.active_sh_degree = 0


# -----------------------------
# Helpers
# -----------------------------
def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def torch_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    # Pearson correlation
    x = x.float()
    y = y.float()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (vx.pow(2).sum().sqrt() * vy.pow(2).sum().sqrt() + 1e-12)
    return float((vx * vy).sum() / denom)


def load_gt_and_mask(gt_path: str, render_res: int, gt_eps: float):
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"GT not found: {gt_path}")
    gt = cv2.resize(gt, (render_res, render_res), interpolation=cv2.INTER_NEAREST)
    gt_t = torch.from_numpy(gt).float().cuda() / 255.0
    mask = (gt_t > gt_eps).float()
    return gt_t, mask


@torch.no_grad()
def main():
    parser = ArgumentParser()

    # core inputs
    parser.add_argument("--model_path", type=str, default="output/debug_run")
    parser.add_argument("--render_res", type=int, default=1024)
    parser.add_argument("--gt_path", type=str, default="output/debug_run/lst_gt.png")
    parser.add_argument("--gt_eps", type=float, default=0.01)
    parser.add_argument("--out_dir", type=str, default="output/robust_compare")

    # camera params (match train_thermal_robust / render_top_down)
    parser.add_argument("--zoom", type=float, default=5.4)
    parser.add_argument("--shift_x", type=float, default=0.0)
    parser.add_argument("--shift_y", type=float, default=-1.2)
    parser.add_argument("--angle_deg", type=float, default=-31.0)
    parser.add_argument("--multiplier", type=float, default=0.85)

    # side-view params (make it closer by default)
    parser.add_argument("--side_dist_mul", type=float, default=0.75, help="smaller => closer")
    parser.add_argument("--side_elev_deg", type=float, default=20.0)

    # PCA speed
    parser.add_argument("--pca_sample", type=int, default=300000)

    # expected up for logging only
    parser.add_argument("--expected_up", type=str, default="")

    # ABC models
    parser.add_argument("--ckpt_old", type=str, required=True)
    parser.add_argument("--priors_old", type=str, required=True)
    parser.add_argument("--ckpt_new", type=str, required=True)
    parser.add_argument("--priors_new", type=str, required=True)
    parser.add_argument("--ckpt_shuf", type=str, required=True)
    parser.add_argument("--priors_shuf", type=str, required=True)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda"
    bg = torch.tensor([0.0, 0.0, 0.0], device=device)
    pipe = type("Pipe", (object,), {"compute_cov3D_python": False, "convert_SHs_python": False,
                                    "brdf": False, "brdf_mode": "pbbr"})()

    # -----------------------------
    # Load geometry
    # -----------------------------
    ply_path = os.path.join(args.model_path, "point_cloud/iteration_7000/point_cloud.ply")
    cameras_json_path = os.path.join(args.model_path, "cameras.json")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(ply_path)
    if not os.path.exists(cameras_json_path):
        raise FileNotFoundError(cameras_json_path)

    print("=== Compare Robust A/B/C (force_apply_texture + closer camera) ===")
    print(f"[Geometry] {ply_path}")
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)

    # freeze (safe)
    gaussians._xyz.requires_grad = False
    for name in ["_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
        getattr(gaussians, name).requires_grad = False

    xyz = gaussians.get_xyz.detach()  # [N,3] on GPU
    center = xyz.mean(dim=0)

    # span
    min_xyz, _ = torch.min(xyz, dim=0)
    max_xyz, _ = torch.max(xyz, dim=0)
    span_x = float((max_xyz[0] - min_xyz[0]).item())
    span_y = float((max_xyz[1] - min_xyz[1]).item())
    max_span = max(span_x, span_y)

    # xyz_norm MUST match train_thermal_robust.py
    xyz_norm = (xyz - center) / (max_span + 1e-8)

    # reference FoV from cameras.json (match your robust training)
    with open(cameras_json_path, "r") as f:
        cams = json.load(f)
    ref = cams[0]
    fov_y = 2 * math.atan(ref["height"] / (2 * ref["fy"]))
    fov_x_mod = 2 * math.atan(math.tan(fov_y / 2) * args.multiplier)

    # -----------------------------
    # PCA axis (sample for speed)
    # -----------------------------
    N = xyz.shape[0]
    n_samp = min(args.pca_sample, N)
    # deterministic sampling
    torch.manual_seed(0)
    idx = torch.randperm(N, device=xyz.device)[:n_samp]
    xyz_s = xyz[idx].detach().cpu()
    xyz_c = xyz_s - xyz_s.mean(dim=0, keepdim=True)
    cov = (xyz_c.T @ xyz_c) / xyz_c.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)

    normal = eigvecs[:, 0].to(device)   # candidate "up"
    axis1 = eigvecs[:, 1].to(device)    # in-plane axis

    # camera correction (match render_top_down / robust training)
    cam_centers = [np.array(c["position"]) for c in cams[:10]]
    mean_cam_pos = torch.tensor(np.mean(cam_centers, axis=0)).float().to(device)
    if torch.dot((mean_cam_pos - center), normal) < 0:
        normal = -normal

    # make in-plane up_axis + right_axis (match robust training)
    up_axis = axis1 - torch.dot(axis1, normal) * normal
    up_axis = up_axis / (torch.norm(up_axis) + 1e-8)
    right_axis = torch.cross(up_axis, normal, dim=0)

    # apply roll angle around normal (same as robust training)
    if abs(args.angle_deg) > 1e-6:
        rad = math.radians(args.angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        new_up = up_axis * cos_a + right_axis * sin_a
        up_axis = new_up / (torch.norm(new_up) + 1e-8)
        right_axis = torch.cross(up_axis, normal, dim=0)

    if args.expected_up.strip():
        try:
            exp = np.array([float(x) for x in args.expected_up.split(",")], dtype=np.float32)
            exp = exp / (np.linalg.norm(exp) + 1e-8)
            got = normal.detach().cpu().numpy()
            got = got / (np.linalg.norm(got) + 1e-8)
            cos_sim = float(np.dot(exp, got))
            print(f"[UpCheck] cos_sim(expected, normal)= {cos_sim:.6f}  (1.0 is best)")
        except Exception as e:
            print(f"[UpCheck] parse failed: {e}")

    # -----------------------------
    # Build TOPDOWN cam (match train_thermal_robust / render_top_down)
    # -----------------------------
    base_height = (max_span / 2.0) / math.tan(fov_y / 2.0)
    target_height = base_height / args.zoom

    shift_vec = (right_axis * args.shift_x) + (up_axis * args.shift_y)
    target_center = center + shift_vec
    cam_pos_top = target_center + normal * target_height

    w2v_top = look_at_topdown(cam_pos_top, target_center, up_axis)
    proj_top = get_projection_matrix(0.01, 100.0, fov_x_mod, fov_y, device=device).transpose(0, 1)
    full_top = w2v_top @ proj_top
    cam_top = MiniCam(args.render_res, args.render_res, fov_y, fov_x_mod, 0.01, 100.0, w2v_top, full_top)

    # -----------------------------
    # Build SIDE cam (closer by default)
    # -----------------------------
    side_dist = max_span * args.side_dist_mul
    side_elev = math.radians(args.side_elev_deg)
    side_h = side_dist * math.tan(side_elev)
    # put camera on +right, lift by +normal
    cam_pos_side = target_center + right_axis * side_dist + normal * side_h

    w2v_side = look_at(cam_pos_side, target_center, normal)  # use "normal" as world up
    proj_side = get_projection_matrix(0.01, 200.0, fov_x_mod, fov_y, device=device).transpose(0, 1)
    full_side = w2v_side @ proj_side
    cam_side = MiniCam(args.render_res, args.render_res, fov_y, fov_x_mod, 0.01, 200.0, w2v_side, full_side)

    # -----------------------------
    # Load GT/mask (for optional quantitative checks)
    # -----------------------------
    gt_t, mask = load_gt_and_mask(args.gt_path, args.render_res, args.gt_eps)
    cov_rate = float(mask.sum().item() / mask.numel() * 100.0)
    print(f"[Mask] coverage={cov_rate:.2f}%  (white means valid pixels)")

    # -----------------------------
    # Sanity: constant colors must change render
    # -----------------------------
    def sanity_constant(cam, tag):
        c1 = torch.full((xyz.shape[0], 3), 0.1, device=device)
        c2 = torch.full((xyz.shape[0], 3), 0.9, device=device)
        force_apply_texture(gaussians, c1)
        im1 = render(cam, gaussians, pipe, bg)["render"][0]
        force_apply_texture(gaussians, c2)
        im2 = render(cam, gaussians, pipe, bg)["render"][0]
        d = float(torch.abs(im2 - im1).mean().item())
        print(f"[Sanity-{tag}] mean(|0.9-0.1|)={d:.6f}  (must be > 0)")
        return d

    dtop = sanity_constant(cam_top, "TOP")
    dside = sanity_constant(cam_side, "SIDE")
    if dtop < 1e-6 or dside < 1e-6:
        print("[FATAL] Render did not respond to force_apply_texture. This indicates renderer override/caching mismatch.")
        print("        Stop here and check gaussian_renderer/render outputs & gaussians internal buffers.")
        return

    # -----------------------------
    # Run one model
    # -----------------------------
    def run_one(name, ckpt_path, priors_path):
        print(f"\n--- [{name}] ---")
        print(f"  ckpt:   {ckpt_path}")
        print(f"  priors: {priors_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        if not os.path.exists(priors_path):
            raise FileNotFoundError(priors_path)

        print(f"  ckpt_sha1={sha1_of_file(ckpt_path)[:10]}  priors_sha1={sha1_of_file(priors_path)[:10]}")

        pri = torch.load(priors_path, map_location="cuda").float()
        if pri.ndim != 2 or pri.shape[1] != 5:
            raise ValueError(f"priors shape must be [N,5], got {tuple(pri.shape)}")

        x = torch.cat([xyz_norm, pri], dim=1)  # [N,8] MUST match training
        net = ThermalAttrNet(input_ch=8, W=16).cuda()
        sd = torch.load(ckpt_path, map_location="cuda")
        net.load_state_dict(sd)
        net.eval()

        t = net(x).squeeze()  # [N]
        t = torch.clamp(t, 0.0, 1.0)

        # stats
        tmin = float(t.min().item()); tmax = float(t.max().item())
        tmean = float(t.mean().item()); tstd = float(t.std().item())
        sat_lo = float((t < 1e-3).float().mean().item() * 100.0)
        sat_hi = float((t > 1 - 1e-3).float().mean().item() * 100.0)
        print(f"  t: min={tmin:.6f} max={tmax:.6f} mean={tmean:.6f} std={tstd:.6f}")
        print(f"  saturation: <1e-3 {sat_lo:.3f}%  >1-1e-3 {sat_hi:.3f}%")

        # render TOP
        rgb = t.unsqueeze(1).expand(-1, 3).contiguous()
        force_apply_texture(gaussians, rgb)
        out_top = render(cam_top, gaussians, pipe, bg)["render"][0].detach()  # [H,W]
        out_side = render(cam_side, gaussians, pipe, bg)["render"][0].detach()

        # masked L1 vs GT (top view only)
        l1 = (torch.abs(out_top - gt_t) * mask).sum() / (mask.sum() + 1e-6)
        print(f"  top_masked_L1={float(l1.item()):.6f}")

        return t.detach(), out_top, out_side, pri

    t_old, im_top_old, im_side_old, pri_old = run_one("OLD", args.ckpt_old, args.priors_old)
    t_new, im_top_new, im_side_new, pri_new = run_one("NEW", args.ckpt_new, args.priors_new)
    t_shu, im_top_shu, im_side_shu, pri_shu = run_one("SHUFFLED", args.ckpt_shuf, args.priors_shuf)

    # -----------------------------
    # Numeric checks: are models really different?
    # -----------------------------
    def sample_diff(a, b, tag):
        # sample on points (tensor diff)
        torch.manual_seed(0)
        idx = torch.randint(0, a.numel(), (200000,), device=a.device)
        d = torch.mean(torch.abs(a[idx] - b[idx])).item()
        print(f"[TensorDiff] mean|{tag}| on 200k pts = {d:.6f}")
        return d

    sample_diff(t_old, t_new, "OLD-NEW")
    sample_diff(t_new, t_shu, "NEW-SHU")
    sample_diff(t_old, t_shu, "OLD-SHU")

    # correlation vs height channel (priors[:,3]) — easy physics sanity
    # note: this is "height feature the model sees", not necessarily true physical height.
    print("\n[Corr] t vs priors_height (channel 3)")
    print(f"  OLD     corr={torch_corr(t_old, pri_old[:, 3]):.4f}")
    print(f"  NEW     corr={torch_corr(t_new, pri_new[:, 3]):.4f}")
    print(f"  SHUFFLE corr={torch_corr(t_shu, pri_shu[:, 3]):.4f}")

    # -----------------------------
    # Save images (grayscale + colormap)
    # -----------------------------
    def save_gray_and_color(img_t: torch.Tensor, stem: str):
        # img_t: [H,W] float 0..1
        g = (torch.clamp(img_t, 0, 1) * 255).byte().cpu().numpy()
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_gray.png"), g)
        cm = cv2.applyColorMap(g, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_jet.png"), cm)

    save_gray_and_color(im_top_old, "top_old")
    save_gray_and_color(im_top_new, "top_new")
    save_gray_and_color(im_top_shu, "top_shuf")
    save_gray_and_color(im_side_old, "side_old")
    save_gray_and_color(im_side_new, "side_new")
    save_gray_and_color(im_side_shu, "side_shuf")

    # diffs
    def save_diff(a: torch.Tensor, b: torch.Tensor, stem: str):
        d = torch.abs(a - b)
        d = d / (d.max() + 1e-8)
        m = (d * 255).byte().cpu().numpy()
        cm = cv2.applyColorMap(m, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_inferno.png"), cm)
        print(f"[ImgDiff] {stem} mean={float(d.mean().item()):.6f} max={float(d.max().item()):.6f}")

    print("\n[ImageDiff] TOP")
    save_diff(im_top_old, im_top_new, "diff_top_old_new")
    save_diff(im_top_new, im_top_shu, "diff_top_new_shuf")
    save_diff(im_top_old, im_top_shu, "diff_top_old_shuf")

    print("\n[ImageDiff] SIDE")
    save_diff(im_side_old, im_side_new, "diff_side_old_new")
    save_diff(im_side_new, im_side_shu, "diff_side_new_shuf")
    save_diff(im_side_old, im_side_shu, "diff_side_old_shuf")

    # save mask preview
    mask_img = (mask * 255).byte().cpu().numpy()
    cv2.imwrite(os.path.join(args.out_dir, "valid_mask_resized.png"), mask_img)

    print(f"\n[DONE] outputs saved to: {args.out_dir}")
    print("Open *_jet.png to compare, and diff_*_inferno.png to see differences amplified.")


if __name__ == "__main__":
    main()

# 对 old / new / shuffled 三个结果做同视角 top-down + side-view 对比，输出对比图、差分图、以及一个 metrics.txt