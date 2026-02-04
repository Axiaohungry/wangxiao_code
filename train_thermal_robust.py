# D:\PycharmProjects\wangxiao_code\train_thermal_robust.py
import os
import sys
import json
import math
import time  # [PATCH] for elapsed time in loss.csv
import cv2
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.thermal_network import ThermalAttrNet


# -------------------------
# 0) Safe load helpers
# -------------------------
def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_tensor_from_pt(path: str, map_location="cpu") -> torch.Tensor:
    obj = safe_torch_load(path, map_location=map_location)
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for k in ["priors", "priors_pt", "data", "features", "tensor"]:
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
        for v in obj.values():
            if torch.is_tensor(v):
                return v
    raise ValueError(f"Cannot find tensor in {path}")


def dump_json(path: str, data: Dict[str, Any]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_semantic_map_2d(path: str) -> np.ndarray:
    """
    Returns HxW uint8 semantic IDs.
    Supports .npy or image file (png/jpg) single-channel.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".npy":
        a = np.load(str(p))
        if a.ndim != 2:
            raise ValueError(f"semantic_map npy must be HxW, got {a.shape}")
        return a.astype(np.uint8)

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read semantic_map image: {path}")
    if img.ndim == 3:
        img = img[..., 0]
    if img.ndim != 2:
        raise ValueError(f"semantic_map image must be single-channel, got {img.shape}")
    return img.astype(np.uint8)


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
# 2) Force SH0 (保梯度)
# -------------------------
def apply_sh0_from_rgb(gaussians: GaussianModel, rgb01: torch.Tensor, empty_rest: torch.Tensor):
    C0 = 0.28209479177387814
    rgb01 = torch.clamp(rgb01.float(), 0.0, 1.0)
    sh_dc = ((rgb01 - 0.5) / C0).float()
    gaussians._features_dc = sh_dc.unsqueeze(1).contiguous()
    gaussians._features_rest = empty_rest.float()
    gaussians.active_sh_degree = 0


def safe_get_alpha(render_out: dict):
    for k in ["alpha", "accumulation", "accum", "A"]:
        if k in render_out:
            a = render_out[k]
            if isinstance(a, torch.Tensor):
                if a.dim() == 3:
                    return a[0]
                if a.dim() == 2:
                    return a
    return None


def pred_activation(x: torch.Tensor, mode: str) -> torch.Tensor:
    mode = mode.lower()
    if mode == "sigmoid":
        return torch.sigmoid(x)
    if mode == "tanh01":
        return 0.5 * (torch.tanh(x) + 1.0)
    if mode == "clamp01":
        return torch.clamp(x, 0.0, 1.0)
    if mode == "none":
        return x
    raise ValueError(f"Unknown pred_act: {mode}")


def zscore_norm(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True)
    sd = torch.clamp(sd, min=eps)
    return (x - mu) / sd, mu.squeeze(0), sd.squeeze(0)


def audit_priors_v3_if_applicable(priors: torch.Tensor, sample_n: int = 200000) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if priors.dim() != 2:
        return out
    N, D = priors.shape
    out["shape"] = [int(N), int(D)]
    if D == 11:
        n = min(int(sample_n), int(N))
        sem = priors[:n, 7:10]
        s = sem.sum(dim=1)
        out["semOH_sum_min"] = float(s.min().item())
        out["semOH_sum_max"] = float(s.max().item())
        out["semOH_sum_mean"] = float(s.mean().item())
        out["semOH_unique_values_sample"] = [float(v) for v in torch.unique(sem).cpu().tolist()[:10]]

        sun = priors[:n, 10]
        out["sun_min"] = float(sun.min().item())
        out["sun_max"] = float(sun.max().item())
        out["sun_mean"] = float(sun.mean().item())
        out["sun_std"] = float(sun.std().item())
    return out


def compute_inview_mask_points_np(xyz_np: np.ndarray, full_T_np: np.ndarray, chunk: int = 400_000) -> np.ndarray:
    """
    full_T uses row-vector convention: [x y z 1] @ full_T -> clip
    Returns boolean mask (N,) whether point is inside NDC [-1,1] in x/y and has valid w.
    """
    N = xyz_np.shape[0]
    mask = np.zeros((N,), dtype=np.bool_)
    full_T_np = full_T_np.astype(np.float32)

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        pts = xyz_np[s:e].astype(np.float32)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        hom = np.concatenate([pts, ones], axis=1)  # Mx4
        clip = hom @ full_T_np  # Mx4
        w = clip[:, 3]
        w = np.where(np.abs(w) < 1e-8, 1e-8, w)
        ndc = clip[:, 0:3] / w[:, None]
        in_view = (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
        mask[s:e] = in_view
    return mask


# [PATCH] loss csv helpers (minimal, no other behavior change)
def _loss_csv_init(output_path: str) -> Path:
    p = Path(output_path) / "loss.csv"
    if not p.exists():
        p.write_text("iter,loss,lr,elapsed_sec,mask_coverage_pct,inview_points_pct\n", encoding="utf-8")
    return p


def _loss_csv_append(csv_path: Path, row: str):
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        f.write(row)


# -------------------------
# 3) Main
# -------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default="output/debug_run")
    parser.add_argument("--gt_path", type=str, default="output/debug_run/lst_gt.png")
    parser.add_argument("--priors_path", type=str, default="output/debug_run/priors.pt")
    parser.add_argument("--output_path", type=str, default="output/thermal_robust")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-3)

    # cam params
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--zoom", type=float, default=5.4)
    parser.add_argument("--shift_x", type=float, default=0.0)
    parser.add_argument("--shift_y", type=float, default=-1.2)
    parser.add_argument("--angle", type=float, default=-31.0)
    parser.add_argument("--multiplier", type=float, default=0.85)
    parser.add_argument("--train_scale", type=float, default=0.5)

    # mask options
    parser.add_argument("--mask_mode", type=str, default="gt+alpha", help="gt | alpha | gt+alpha | none")
    parser.add_argument("--gt_eps", type=float, default=1.0 / 255.0)
    parser.add_argument("--alpha_eps", type=float, default=1e-3)

    # stability / normalization
    parser.add_argument("--pred_act", type=str, default="sigmoid",
                        choices=["sigmoid", "tanh01", "clamp01", "none"])
    parser.add_argument("--priors_norm", type=str, default="none", choices=["none", "zscore"])
    parser.add_argument("--stats_sample", type=int, default=200000)

    # speed: predict only in-view points
    parser.add_argument("--predict_inview_only", action="store_true",
                        help="Only run thermal_net forward for points inside current topdown frustum.")
    parser.add_argument("--inview_chunk", type=int, default=400000)

    # optional semantic weighted loss (pixel-space)
    parser.add_argument("--semantic_map_path", type=str, default="",
                        help="Optional semantic_map.(png|npy) aligned with topdown. IDs {0,1,2}.")
    parser.add_argument("--w_veg", type=float, default=1.0)
    parser.add_argument("--w_bldg", type=float, default=3.0)
    parser.add_argument("--w_road", type=float, default=2.0)

    # AMP
    parser.add_argument("--amp", action="store_true", help="Enable torch.cuda.amp mixed precision.")

    # misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--grad_every", type=int, default=100)
    parser.add_argument("--fail_grad_eps", type=float, default=1e-10)

    # [PATCH] loss logging
    parser.add_argument("--log_every", type=int, default=10,
                        help="Append one row to output_path/loss.csv every N iters (N>=1).")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ply_path = os.path.join(args.model_path, "point_cloud/iteration_7000/point_cloud.ply")
    cameras_json_path = os.path.join(args.model_path, "cameras.json")
    if not os.path.exists(ply_path):
        print(f"[Error] PLY not found: {ply_path}")
        sys.exit(1)
    if not os.path.exists(args.priors_path):
        print(f"[Error] priors not found: {args.priors_path}")
        sys.exit(1)
    if not os.path.exists(args.gt_path):
        print(f"[Error] GT not found: {args.gt_path}")
        sys.exit(1)

    os.makedirs(args.output_path, exist_ok=True)
    dump_json(os.path.join(args.output_path, "run_args.json"), vars(args))

    # [PATCH] init loss.csv + timers (no other folder layout changes)
    loss_csv = _loss_csv_init(args.output_path)
    t0 = time.time()

    print("=== Robust Thermal Training (v3-ready, frustum-opt, semantic-weight) ===")
    print(f"[IO] priors_path={args.priors_path}")
    print(f"[IO] gt_path={args.gt_path}")
    print(f"[IO] output_path={args.output_path}")
    print(f"[LossCSV] {str(loss_csv)} (log_every={max(1,int(args.log_every))})")

    # --- Load Geometry ---
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)

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

    # --- Load Priors (robust) ---
    priors = load_tensor_from_pt(args.priors_path, map_location="cpu").float().cuda()
    if priors.dim() != 2 or priors.shape[0] != xyz.shape[0]:
        print(f"[Error] priors shape mismatch: priors={tuple(priors.shape)}, xyz={tuple(xyz.shape)}")
        sys.exit(1)
    print(f"[Priors] loaded shape={tuple(priors.shape)}")

    audit = audit_priors_v3_if_applicable(priors, sample_n=int(args.stats_sample))
    if audit:
        dump_json(os.path.join(args.output_path, "priors_audit.json"), audit)
        if priors.shape[1] == 11:
            print("[Priors-Audit] v3 detected (dim=11)")
            print(f"  semOH_sum min/max/mean = {audit.get('semOH_sum_min'):.6f}/{audit.get('semOH_sum_max'):.6f}/{audit.get('semOH_sum_mean'):.6f}")
            print(f"  sun_facing min/max/mean/std = {audit.get('sun_min'):.6f}/{audit.get('sun_max'):.6f}/{audit.get('sun_mean'):.6f}/{audit.get('sun_std'):.6f}")

            if not (abs(audit.get("semOH_sum_min", 0.0) - 1.0) < 1e-6 and abs(audit.get("semOH_sum_max", 0.0) - 1.0) < 1e-6):
                print("[FATAL] semOH sum is not 1.0 everywhere (sample). v3 invalid.")
                sys.exit(1)
            if not (0.0 <= audit.get("sun_min", -1.0) <= 1.0 and 0.0 <= audit.get("sun_max", 2.0) <= 1.0):
                print("[FATAL] sun_facing out of [0,1] (sample). v3 invalid.")
                sys.exit(1)

    if args.priors_norm.lower() == "zscore":
        priors, mu, sd = zscore_norm(priors)
        dump_json(os.path.join(args.output_path, "priors_norm_stats.json"), {
            "mode": "zscore",
            "mu": [float(v) for v in mu.detach().cpu().tolist()],
            "sd": [float(v) for v in sd.detach().cpu().tolist()],
        })
        print("[Priors] applied zscore normalization")

    # --- Camera setup (match render_top_down style) ---
    with open(cameras_json_path, "r", encoding="utf-8") as f:
        cams_data = json.load(f)
    ref_cam = cams_data[0]
    fov_y = 2 * math.atan(ref_cam["height"] / (2 * ref_cam["fy"]))
    fov_x_mod = 2 * math.atan(math.tan(fov_y / 2) * args.multiplier)

    # PCA (CPU)
    xyz_cpu = xyz.detach().cpu()
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    _, eigvecs = torch.linalg.eigh(cov)
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

    train_w = int(args.width * args.train_scale)
    train_h = int(args.height * args.train_scale)
    if train_w <= 0 or train_h <= 0:
        print("[FATAL] train_w/train_h invalid.")
        sys.exit(1)

    base_height = (max_span / 2.0) / np.tan(fov_y / 2.0)
    target_height = base_height / args.zoom
    shift_vec = (right_axis * args.shift_x) + (up_axis * args.shift_y)
    target_center = center + shift_vec
    cam_pos = target_center + normal * target_height

    w2v = look_at_topdown(cam_pos, target_center, up_axis)
    proj = get_projection_matrix(0.01, 100.0, fov_x_mod, fov_y).transpose(0, 1)
    full_T = (w2v @ proj).detach()
    view_cam = MiniCam(train_w, train_h, fov_y, fov_x_mod, 0.01, 100.0, w2v, full_T)

    # --- Optional frustum in-view mask (point-space) ---
    idx_inview = None
    inview_points_pct = 100.0  # [PATCH] for loss.csv
    if args.predict_inview_only:
        full_T_np = full_T.detach().cpu().numpy()
        xyz_np = xyz.detach().cpu().numpy()
        mask_np = compute_inview_mask_points_np(xyz_np, full_T_np, chunk=int(args.inview_chunk))
        inview_points_pct = float(mask_np.mean()) * 100.0  # [PATCH]
        ratio = inview_points_pct
        np.save(os.path.join(args.output_path, "point_inview_mask.npy"), mask_np.astype(np.uint8))
        print(f"[InView] points in frustum = {ratio:.2f}%  (saved point_inview_mask.npy)")

        idx = np.nonzero(mask_np)[0].astype(np.int64)
        if idx.size < 10000:
            print("[FATAL] Too few in-view points. Camera params likely mismatch.")
            sys.exit(1)
        idx_inview = torch.from_numpy(idx).to(device="cuda", dtype=torch.long)

    # --- Load GT ---
    gt_img = cv2.imread(args.gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        raise ValueError("GT not readable.")
    gt_resized = cv2.resize(gt_img, (train_w, train_h), interpolation=cv2.INTER_AREA)
    gt_tensor = torch.from_numpy(gt_resized).float().cuda() / 255.0
    print(f"[GT] resized to {train_w}x{train_h}, mean={gt_tensor.mean().item():.6f}")

    # --- Optional semantic weighted loss (pixel-space) ---
    weight_map = torch.ones_like(gt_tensor)
    if args.semantic_map_path.strip():
        sem2d = load_semantic_map_2d(args.semantic_map_path)
        sem2d_r = cv2.resize(sem2d, (train_w, train_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(args.output_path, "semantic_map_resized.png"), sem2d_r.astype(np.uint8))
        w = np.ones_like(sem2d_r, dtype=np.float32)
        w[sem2d_r == 0] = float(args.w_veg)
        w[sem2d_r == 1] = float(args.w_bldg)
        w[sem2d_r == 2] = float(args.w_road)
        weight_map = torch.from_numpy(w).float().cuda()
        print(f"[LossWeight] semantic_map enabled. w_veg={args.w_veg} w_bldg={args.w_bldg} w_road={args.w_road}")

    # --- Prepare mask ---
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
                tmp = render(view_cam, gaussians, pipeline_args, bg)
                a = safe_get_alpha(tmp)
                if a is None:
                    print("[Warn] render_out has no alpha/accumulation key. alpha-mask skipped.")
                else:
                    alpha_mask = (a > args.alpha_eps).float()
                    valid_mask = valid_mask * alpha_mask

    final_wmask = valid_mask * weight_map
    coverage = (final_wmask.gt(0).float().mean().item()) * 100.0
    print(f"[Mask] mode={args.mask_mode}, coverage={coverage:.2f}%")
    if coverage < 1.0:
        print("[FATAL] Mask coverage too small.")
        sys.exit(1)

    cv2.imwrite(os.path.join(args.output_path, "valid_mask.png"),
                (valid_mask.detach().cpu().numpy() * 255).astype(np.uint8))

    # --- Network ---
    in_ch = 3 + priors.shape[1]
    print(f"[Phase5] input_ch={in_ch} (XYZ=3 + priors={priors.shape[1]})")
    thermal_net = ThermalAttrNet(input_ch=in_ch, W=16).cuda()
    thermal_net.train()

    optimizer = torch.optim.Adam(thermal_net.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))

    # inputs
    xyz_norm = (gaussians.get_xyz.detach() - center) / (max_span + 1e-6)

    if idx_inview is not None:
        full_input = torch.cat([xyz_norm[idx_inview], priors[idx_inview]], dim=1).contiguous()
        N_in = int(full_input.shape[0])
        print(f"[Input] using in-view points only: {N_in} / {xyz.shape[0]}")
    else:
        full_input = torch.cat([xyz_norm, priors], dim=1).contiguous()

    empty_rest = torch.empty((xyz.shape[0], 0, 3), device="cuda", dtype=torch.float32)

    # --- Sanity: const color ---
    with torch.no_grad():
        c1 = torch.full((xyz.shape[0], 3), 0.1, device="cuda")
        apply_sh0_from_rgb(gaussians, c1, empty_rest)
        r1 = render(view_cam, gaussians, pipeline_args, bg)["render"][0].mean().item()

        c2 = torch.full((xyz.shape[0], 3), 0.9, device="cuda")
        apply_sh0_from_rgb(gaussians, c2, empty_rest)
        r2 = render(view_cam, gaussians, pipeline_args, bg)["render"][0].mean().item()

    print(f"[Sanity] ConstColor mean: 0.1 -> {r1:.4f}, 0.9 -> {r2:.4f}, diff={abs(r2-r1):.4f}")
    if abs(r2 - r1) < 0.05:
        print("[FATAL] Force-apply texture seems ineffective.")
        sys.exit(1)

    # --- Train ---
    log_every = int(max(1, args.log_every))  # [PATCH]
    pbar = tqdm(range(1, args.iterations + 1))
    for it in pbar:
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=bool(args.amp)):
            thermal_raw = thermal_net(full_input)  # [M,1] or [N,1]
            thermal_val = pred_activation(thermal_raw, args.pred_act)
            rgb_in = thermal_val.expand(-1, 3)

            # expand to full N for renderer
            rgb_full = torch.full((xyz.shape[0], 3), 0.5, device="cuda", dtype=rgb_in.dtype)
            if idx_inview is not None:
                rgb_full[idx_inview] = rgb_in
            else:
                rgb_full = rgb_in  # already Nx3

            apply_sh0_from_rgb(gaussians, rgb_full.float(), empty_rest)

            with torch.cuda.amp.autocast(enabled=False):
                out = render(view_cam, gaussians, pipeline_args, bg)
                pred = out["render"][0].float()

            diff = (pred - gt_tensor).abs()
            wmask = final_wmask
            loss = (diff * wmask).sum() / (wmask.sum() + 1e-6)

        if not torch.isfinite(loss):
            print("[FATAL] Loss is NaN/Inf.")
            sys.exit(1)

        scaler.scale(loss).backward()

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

        scaler.step(optimizer)
        scaler.update()

        if it % 10 == 0:
            pbar.set_description(f"Loss: {loss.item():.6f}")

        # [PATCH] write loss.csv
        if (it == 1) or (it % log_every == 0) or (it == args.iterations):
            lr_now = float(optimizer.param_groups[0]["lr"])
            elapsed = float(time.time() - t0)
            lval = float(loss.detach().item())
            row = f"{it},{lval:.10f},{lr_now:.8g},{elapsed:.3f},{coverage:.4f},{inview_points_pct:.4f}\n"
            _loss_csv_append(loss_csv, row)

        if it % args.save_every == 0 or it == args.iterations:
            vis = pred.detach().clamp(0, 1).cpu().numpy()
            vis_u8 = (vis * 255).astype(np.uint8)
            vis_color = cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.output_path, f"train_{it:04d}.png"), vis_color)
            cv2.imwrite(os.path.join(args.output_path, f"train_{it:04d}_gray.png"), vis_u8)

    torch.save(thermal_net.state_dict(), os.path.join(args.output_path, "thermal_net_robust.pth"))
    print("Training Done!")


if __name__ == "__main__":
    main()
