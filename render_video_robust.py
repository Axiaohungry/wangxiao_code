# D:\PycharmProjects\wangxiao_code\render_video_robust.py
import os
import sys
import json
import math
import cv2
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

# -----------------------------------------------------------------------------
# Ensure project import works no matter where you run
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.thermal_network import ThermalAttrNet


# ==============================================================================
# Verified Baseline Defaults (IMPORTANT)
# ------------------------------------------------------------------------------
DEFAULT_PRIORS_PATH = r"output\debug_run\priors.pt"
DEFAULT_THERMAL_CKPT = r"output\thermal_robust_new\thermal_net_robust.pth"
DEFAULT_OUTPUT_VIDEO = r"output\thermal_robust_new\video_robust_new_expectedup.mp4"
DEFAULT_EXPECTED_UP = "0.04849088,-0.74267894,-0.6678897"


# ==============================================================================
# MiniCam (same convention as project; fixed 3x3 intrinsic)
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


# IMPORTANT: exactly the same look-at as render_video_v1_compare.py
def get_look_at(cam_pos, target, up):
    z_axis = target - cam_pos
    dist = torch.norm(z_axis)
    if dist < 1e-5:
        return torch.eye(4, device=cam_pos.device)

    z_axis = z_axis / dist
    x_axis = torch.cross(up, z_axis, dim=0)

    if torch.norm(x_axis) < 1e-5:
        temp = torch.tensor([1.0, 0.0, 0.0], device=cam_pos.device)
        if torch.abs(torch.dot(temp, z_axis)) > 0.9:
            temp = torch.tensor([0.0, 1.0, 0.0], device=cam_pos.device)
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


def parse_vec3(s: str) -> torch.Tensor:
    raw = s.strip()
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if len(parts) != 3:
        raise ValueError(f'--expected_up expects format "x,y,z" (3 floats), got: {s!r}')
    try:
        vals = [float(parts[0]), float(parts[1]), float(parts[2])]
    except ValueError as e:
        raise ValueError(f'--expected_up must be 3 floats like "0.1,0.2,0.3", got: {s!r}') from e
    v = torch.tensor(vals, dtype=torch.float32, device="cuda")
    v = v / (torch.norm(v) + 1e-8)
    return v


# ==============================================================================
# v3-compatible helpers (minimal additions)
# ==============================================================================
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


def pred_activation(x: torch.Tensor, mode: str) -> torch.Tensor:
    mode = (mode or "sigmoid").lower()
    if mode == "sigmoid":
        return torch.sigmoid(x)
    if mode == "tanh01":
        return 0.5 * (torch.tanh(x) + 1.0)
    if mode == "clamp01":
        return torch.clamp(x, 0.0, 1.0)
    if mode == "none":
        return x
    raise ValueError(f"Unknown --pred_act: {mode}")


def infer_net_dims_from_state_dict(sd: dict) -> tuple:
    """
    Infer (input_ch, W) from the first linear layer weight.
    Expect a weight shaped [W, input_ch]. (In your runs W usually = 16.)
    """
    best = None  # (input_ch, W)
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if v.ndim == 2 and v.shape[0] >= 4 and v.shape[1] >= 4:
            # Prefer typical first-layer size: out=W (often 16), in=input_ch (often 8/14/...)
            # Avoid output layer: [1, W]
            if v.shape[0] > 1 and v.shape[1] > 1:
                # Heuristic: first layer often has out dim >= 8
                if best is None:
                    best = (int(v.shape[1]), int(v.shape[0]))
                else:
                    # Prefer larger input_ch (more informative) when multiple candidates exist
                    if int(v.shape[1]) > best[0]:
                        best = (int(v.shape[1]), int(v.shape[0]))
    if best is None:
        raise ValueError("Cannot infer input_ch/W from ckpt. Unexpected state_dict format.")
    return best  # (input_ch, W)


def apply_zscore_from_stats(priors: torch.Tensor, stats_json_path: str) -> torch.Tensor:
    """
    Apply zscore using saved mu/sd in priors_norm_stats.json (same as train_thermal_robust.py).
    """
    stats = json.loads(open(stats_json_path, "r", encoding="utf-8").read())
    if str(stats.get("mode", "")).lower() != "zscore":
        print(f"[Warn] priors_norm_stats mode is not zscore: {stats.get('mode')}. Skip.")
        return priors
    mu = torch.tensor(stats["mu"], dtype=priors.dtype, device=priors.device).view(1, -1)
    sd = torch.tensor(stats["sd"], dtype=priors.dtype, device=priors.device).view(1, -1)
    if mu.numel() != priors.shape[1] or sd.numel() != priors.shape[1]:
        raise ValueError(f"priors_norm_stats dim mismatch: mu/sd={mu.numel()} priors_dim={priors.shape[1]}")
    sd = torch.clamp(sd, min=1e-6)
    return (priors - mu) / sd


def main():
    ap = ArgumentParser()
    ap.add_argument("--no_align_expected_up", action="store_true",
                    help="Do NOT hemisphere-align expected_up to -PCA normal; use it as-is.")
    ap.add_argument("--model_path", "-m", default="output/debug_run")

    # Defaults = verified baseline (CLI overrides)
    ap.add_argument("--priors_path", default=DEFAULT_PRIORS_PATH)
    ap.add_argument("--thermal_ckpt", default=DEFAULT_THERMAL_CKPT)
    ap.add_argument("--output_video", default=DEFAULT_OUTPUT_VIDEO)

    ap.add_argument("--render_res", type=int, default=1024)
    ap.add_argument("--n_frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)

    # Orbit params
    ap.add_argument("--elevation", type=float, default=45.0)
    ap.add_argument("--orbit_mul", type=float, default=1.5)
    ap.add_argument("--start_angle_deg", type=float, default=0.0)

    # expected_up
    ap.add_argument(
        "--expected_up",
        default=DEFAULT_EXPECTED_UP,
        help='expected "up" direction vec3 as "x,y,z"',
    )

    # Chunking
    ap.add_argument("--chunk", type=int, default=300000)

    # v3 compat: normalization + activation (match training)
    ap.add_argument("--priors_norm_stats", default="", help="Path to priors_norm_stats.json (zscore).")
    ap.add_argument("--pred_act", default="sigmoid", choices=["sigmoid", "tanh01", "clamp01", "none"])

    args = ap.parse_args()
    device = "cuda"

    ply_path = os.path.join(args.model_path, "point_cloud/iteration_7000/point_cloud.ply")
    cameras_json_path = os.path.join(args.model_path, "cameras.json")

    if not os.path.exists(ply_path):
        raise FileNotFoundError(ply_path)
    if not os.path.exists(cameras_json_path):
        raise FileNotFoundError(cameras_json_path)
    if not os.path.exists(args.priors_path):
        raise FileNotFoundError(args.priors_path)
    if not os.path.exists(args.thermal_ckpt):
        raise FileNotFoundError(args.thermal_ckpt)
    if args.priors_norm_stats.strip() and (not os.path.exists(args.priors_norm_stats)):
        raise FileNotFoundError(args.priors_norm_stats)

    # Create output dir
    out_dir = os.path.dirname(args.output_video)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1) Load Gaussians
    print("Loading Geometry.")
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)

    gaussians._xyz.requires_grad = False
    for name in ["_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
        getattr(gaussians, name).requires_grad = False

    xyz = gaussians.get_xyz.detach()
    center = xyz.mean(dim=0)

    # 2) Load Priors (v3 compatible)
    print(f"Loading Priors: {args.priors_path}")
    priors = load_tensor_from_pt(args.priors_path, map_location="cuda").float()
    if priors.shape[0] != xyz.shape[0]:
        raise ValueError(f"priors N mismatch: {tuple(priors.shape)} vs xyz {tuple(xyz.shape)}")

    if args.priors_norm_stats.strip():
        print(f"[Priors] applying zscore from: {args.priors_norm_stats}")
        priors = apply_zscore_from_stats(priors, args.priors_norm_stats)

    pri_dim = int(priors.shape[1])
    print(f"[Priors] shape={tuple(priors.shape)} (dim={pri_dim})")

    # --- Load Robust Net (infer input_ch/W from ckpt correctly) ---
    print(f"Loading Thermal Net: {args.thermal_ckpt}")

    # 安全加载（也能避免 FutureWarning）
    try:
        sd = torch.load(args.thermal_ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(args.thermal_ckpt, map_location="cpu")

    # 关键：Linear.weight 形状 = (out_features, in_features)
    if "layers.0.weight" not in sd:
        # 兜底：找最像第一层的 weight
        k0 = [k for k in sd.keys() if k.endswith("layers.0.weight")]
        if not k0:
            raise KeyError("Cannot find layers.0.weight in ckpt.")
        w0 = sd[k0[0]]
    else:
        w0 = sd["layers.0.weight"]

    ckpt_W = int(w0.shape[0])  # out_features
    ckpt_in_ch = int(w0.shape[1])  # in_features

    print(f"[Net] inferred from ckpt: input_ch={ckpt_in_ch}, W={ckpt_W}")

    net = ThermalAttrNet(input_ch=ckpt_in_ch, W=ckpt_W).cuda()
    net.load_state_dict(sd if isinstance(sd, dict) else sd, strict=True)
    net.eval()

    # 4) FOV from cameras.json
    with open(cameras_json_path, "r") as f:
        cams_data = json.load(f)
    ref_cam = cams_data[0]
    fov_y = 2 * math.atan(ref_cam["height"] / (2 * ref_cam["fy"]))
    fov_x = fov_y

    # 5) Build orbit basis
    xyz_cpu = xyz.cpu()
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    _, eigvecs = torch.linalg.eigh(cov)
    pca_normal = eigvecs[:, 0].to(device)
    up_ref = (-pca_normal) / (torch.norm(pca_normal) + 1e-8)

    if args.expected_up.strip():
        exp = parse_vec3(args.expected_up)
        if (not args.no_align_expected_up) and torch.dot(exp, up_ref) < 0:
            exp = -exp
        up_vec = exp
        print(f"[Up] using expected_up. dot(expected_up, -pca)= {torch.dot(up_vec, up_ref).item():.4f}"
              + (" (no_align)" if args.no_align_expected_up else ""))
    else:
        up_vec = up_ref
        print("[Up] using -PCA normal (same as render_video_v1_compare.py)")

    temp = torch.tensor([1.0, 0.0, 0.0], device=device)
    if torch.abs(torch.dot(temp, up_vec)) > 0.9:
        temp = torch.tensor([0.0, 1.0, 0.0], device=device)
    right_vec = torch.cross(up_vec, temp, dim=0)
    right_vec = right_vec / (torch.norm(right_vec) + 1e-8)
    fwd_vec = torch.cross(up_vec, right_vec, dim=0)
    fwd_vec = fwd_vec / (torch.norm(fwd_vec) + 1e-8)

    # Orbit radius
    max_dist = torch.max(torch.norm(xyz - center, dim=1)).item()
    orbit_radius = max_dist * args.orbit_mul

    elevation_rad = math.radians(args.elevation)
    h_up = orbit_radius * math.sin(elevation_rad)
    r_plane = orbit_radius * math.cos(elevation_rad)

    # 6) Build xyz_norm exactly like train_thermal_robust.py
    min_xyz, _ = torch.min(xyz, dim=0)
    max_xyz, _ = torch.max(xyz, dim=0)
    span_x = (max_xyz[0] - min_xyz[0]).item()
    span_y = (max_xyz[1] - min_xyz[1]).item()
    max_span = max(span_x, span_y)
    xyz_norm = (xyz - center) / (max_span + 1e-6)

    # 7) Predict colors in chunks, then FORCE APPLY into gaussians SH
    print("Pre-calculating thermal colors (chunked) + force_apply.")
    N = xyz.shape[0]
    C0 = 0.28209479177387814

    dc = gaussians._features_dc
    if isinstance(dc, torch.nn.Parameter):
        dc_data = dc.data
    else:
        gaussians._features_dc = torch.zeros((N, 1, 3), device=device, dtype=torch.float32)
        dc_data = gaussians._features_dc

    with torch.no_grad():
        for s in tqdm(range(0, N, args.chunk), desc="infer+apply"):
            e = min(N, s + args.chunk)

            inp = torch.cat([xyz_norm[s:e], priors[s:e]], dim=1).contiguous()  # [B, 3+pri_dim]
            t_raw = net(inp)  # [B,1] raw
            t = pred_activation(t_raw, args.pred_act)  # match training
            rgb = t.repeat(1, 3)  # [B,3]
            sh = (rgb - 0.5) / C0  # [B,3]
            dc_data[s:e, 0, :].copy_(sh)

        if getattr(gaussians, "_features_rest", None) is not None and gaussians._features_rest.numel() > 0:
            if isinstance(gaussians._features_rest, torch.nn.Parameter):
                gaussians._features_rest.data.zero_()
            else:
                gaussians._features_rest.zero_()
        gaussians.active_sh_degree = 0

    # 8) Render orbit
    pipe = type("Pipe", (object,), {"compute_cov3D_python": False, "convert_SHs_python": False, "brdf": False, "brdf_mode": "pbbr"})()
    bg = torch.tensor([0.0, 0.0, 0.0], device=device)

    frames = []
    print("Rendering orbit.")
    start_angle = math.radians(args.start_angle_deg)

    for i in tqdm(range(args.n_frames)):
        angle = start_angle + 2 * math.pi * (i / args.n_frames)

        offset = (right_vec * math.cos(angle) * r_plane) + \
                 (fwd_vec * math.sin(angle) * r_plane) - \
                 (up_vec * h_up)

        cam_pos = center + offset
        w2v = get_look_at(cam_pos, center, up_vec)
        proj = get_projection_matrix(0.1, 1000.0, fov_x, fov_y, device=device).transpose(0, 1)
        full_proj = w2v @ proj
        cam = MiniCam(args.render_res, args.render_res, fov_y, fov_x, 0.1, 1000.0, w2v, full_proj)

        out = render(cam, gaussians, pipe, bg)["render"]  # [3,H,W]
        pred = out[0, :, :]  # grayscale since RGB identical
        img = pred.detach().cpu().numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(cv2.applyColorMap(img, cv2.COLORMAP_JET))

    print(f"Saving to: {args.output_video}")
    h, w, _ = frames[0].shape
    out_vid = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for f in frames:
        out_vid.write(f)
    out_vid.release()
    print("Done.")


if __name__ == "__main__":
    main()
