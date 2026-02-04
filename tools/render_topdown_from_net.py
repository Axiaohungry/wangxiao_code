# tools/render_topdown_from_net.py
# Render one topdown frame using trained ThermalAttrNet weights (no training), for pre/post crop verification.

import os, sys, json, math
from pathlib import Path
import numpy as np
import cv2
import torch
from argparse import ArgumentParser

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.thermal_network import ThermalAttrNet

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

def zscore_norm(x: torch.Tensor, eps: float = 1e-6):
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True).clamp(min=eps)
    return (x - mu) / sd, mu.squeeze(0), sd.squeeze(0)

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

def apply_sh0_from_rgb(gaussians: GaussianModel, rgb01: torch.Tensor, empty_rest: torch.Tensor):
    C0 = 0.28209479177387814
    rgb01 = torch.clamp(rgb01.float(), 0.0, 1.0)
    sh_dc = ((rgb01 - 0.5) / C0).float()
    gaussians._features_dc = sh_dc.unsqueeze(1).contiguous()
    gaussians._features_rest = empty_rest.float()
    gaussians.active_sh_degree = 0

def main():
    ap = ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--priors_path", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--train_scale", type=float, default=0.5)

    ap.add_argument("--zoom", type=float, default=5.4)
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--shift_y", type=float, default=-1.2)
    ap.add_argument("--angle", type=float, default=-31.0)
    ap.add_argument("--multiplier", type=float, default=0.85)

    ap.add_argument("--pred_act", type=str, default="sigmoid", choices=["sigmoid","tanh01","clamp01","none"])
    ap.add_argument("--priors_norm", type=str, default="none", choices=["none","zscore"])
    ap.add_argument("--amp", action="store_true")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_path = Path(args.model_path) / "point_cloud" / "iteration_7000" / "point_cloud.ply"
    cameras_json_path = Path(args.model_path) / "cameras.json"
    assert ply_path.exists(), ply_path
    assert cameras_json_path.exists(), cameras_json_path
    assert Path(args.priors_path).exists(), args.priors_path
    assert Path(args.ckpt_path).exists(), args.ckpt_path

    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(str(ply_path))

    xyz = gaussians.get_xyz.detach().cuda()
    center = xyz.mean(dim=0)
    mn, _ = torch.min(xyz, dim=0)
    mx, _ = torch.max(xyz, dim=0)
    max_span = max(float(mx[0]-mn[0]), float(mx[1]-mn[1]))

    priors = load_tensor_from_pt(args.priors_path, map_location="cpu").float().cuda()
    assert priors.dim() == 2 and priors.shape[0] == xyz.shape[0], (priors.shape, xyz.shape)

    if args.priors_norm == "zscore":
        priors, mu, sd = zscore_norm(priors)
        (out_dir / "priors_norm_stats.json").write_text(json.dumps({
            "mu": [float(v) for v in mu.detach().cpu().tolist()],
            "sd": [float(v) for v in sd.detach().cpu().tolist()],
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    cams_data = json.loads(cameras_json_path.read_text(encoding="utf-8"))
    ref_cam = cams_data[0]
    fov_y = 2 * math.atan(ref_cam["height"] / (2 * ref_cam["fy"]))
    fov_x_mod = 2 * math.atan(math.tan(fov_y / 2) * args.multiplier)

    # PCA
    xyz_cpu = xyz.detach().cpu()
    xc = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xc.T @ xc / xc.shape[0]
    _, eigvecs = torch.linalg.eigh(cov)
    normal = eigvecs[:, 0].cuda()
    axis1 = eigvecs[:, 1].cuda()

    cam_centers = [np.array(c["position"]) for c in cams_data[:10]]
    mean_cam = torch.tensor(np.mean(cam_centers, axis=0)).float().cuda()
    if torch.dot((mean_cam - center), normal) < 0:
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

    base_height = (max_span / 2.0) / np.tan(fov_y / 2.0)
    target_height = base_height / args.zoom
    shift_vec = right_axis * args.shift_x + up_axis * args.shift_y
    target_center = center + shift_vec
    cam_pos = target_center + normal * target_height

    w2v = look_at_topdown(cam_pos, target_center, up_axis)
    proj = get_projection_matrix(0.01, 100.0, fov_x_mod, fov_y).transpose(0, 1)
    full_T = (w2v @ proj).detach()
    view_cam = MiniCam(train_w, train_h, fov_y, fov_x_mod, 0.01, 100.0, w2v, full_T)

    # net
    in_ch = 3 + priors.shape[1]
    net = ThermalAttrNet(input_ch=in_ch, W=16).cuda().eval()
    sd = safe_torch_load(args.ckpt_path, map_location="cpu")
    net.load_state_dict(sd, strict=True)

    xyz_norm = (xyz - center) / (max_span + 1e-6)
    full_input = torch.cat([xyz_norm, priors], dim=1).contiguous()

    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    pipe = type("Pipe", (object,), {"compute_cov3D_python": False, "convert_SHs_python": False, "brdf": False, "brdf_mode": "pbbr"})()

    empty_rest = torch.empty((xyz.shape[0], 0, 3), device="cuda", dtype=torch.float32)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(args.amp)):
        t_raw = net(full_input)
        t_val = pred_activation(t_raw, args.pred_act)
        rgb = t_val.expand(-1, 3)
        apply_sh0_from_rgb(gaussians, rgb.float(), empty_rest)
        out = render(view_cam, gaussians, pipe, bg)
        pred = out["render"][0].float().clamp(0, 1)

    u8 = (pred.detach().cpu().numpy() * 255).astype(np.uint8)
    jet = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_dir / "pred_gray.png"), u8)
    cv2.imwrite(str(out_dir / "pred_jet.png"), jet)
    (out_dir / "render_audit.json").write_text(json.dumps({
        "train_w": train_w, "train_h": train_h,
        "mean": float(pred.mean().item()),
        "std": float(pred.std().item()),
        "min": float(pred.min().item()),
        "max": float(pred.max().item()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] wrote:", out_dir)

if __name__ == "__main__":
    main()
