# esttools/render_alpha_mask_topdown.py  (v2: ref_model_path + flip options)
import argparse, json, math
from pathlib import Path
import numpy as np
import cv2
import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render


def look_at_topdown(cam_pos, target, up, device="cuda"):
    z_axis = target - cam_pos
    dist = torch.norm(z_axis)
    if dist < 1e-6:
        return torch.eye(4, device=device)
    z_axis = z_axis / dist
    x_axis = torch.cross(up, z_axis, dim=0)
    if torch.norm(x_axis) < 1e-6:
        tmp = torch.tensor([1.0, 0.0, 0.0], device=device)
        x_axis = torch.cross(tmp, z_axis, dim=0)
    x_axis = x_axis / (torch.norm(x_axis) + 1e-8)
    y_axis = torch.cross(z_axis, x_axis, dim=0)
    y_axis = y_axis / (torch.norm(y_axis) + 1e-8)
    R = torch.stack([x_axis, y_axis, z_axis], dim=0)
    T = -torch.matmul(R, cam_pos)
    w2v = torch.eye(4, device=device)
    w2v[:3, :3] = R
    w2v[:3, 3] = T
    return w2v.transpose(0, 1).contiguous()


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


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = int(width)
        self.image_height = int(height)
        self.FoVy = float(fovy)
        self.FoVx = float(fovx)
        self.znear = float(znear)
        self.zfar = float(zfar)
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)

    def get_calib_matrix_nerf(self):
        device = self.world_view_transform.device
        focal_y = self.image_height / (2.0 * self.tanfovy)
        focal_x = self.image_width / (2.0 * self.tanfovx)
        intrinsic = torch.tensor([
            [focal_x, 0.0, self.image_width / 2.0],
            [0.0, focal_y, self.image_height / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device)
        extrinsic = self.world_view_transform.transpose(0, 1)
        return intrinsic, extrinsic


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


def load_xyz_and_cams(model_path: str):
    model_path = Path(model_path)
    ply = model_path / "point_cloud/iteration_7000/point_cloud.ply"
    cams = model_path / "cameras.json"
    if not ply.exists(): raise FileNotFoundError(str(ply))
    if not cams.exists(): raise FileNotFoundError(str(cams))
    cams_data = json.loads(cams.read_text(encoding="utf-8"))
    gauss = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gauss.load_ply(str(ply))
    xyz = gauss.get_xyz.detach()
    return xyz, cams_data, str(ply)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)         # render THIS model (ROI)
    ap.add_argument("--ref_model_path", default="")        # compute camera pose from THIS model (full/debug_run)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--render_res", type=int, default=1024)

    ap.add_argument("--zoom", type=float, default=5.4)
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--shift_y", type=float, default=-1.2)
    ap.add_argument("--angle", type=float, default=-31.0)
    ap.add_argument("--multiplier", type=float, default=0.85)

    ap.add_argument("--alpha_eps", type=float, default=0.02)
    ap.add_argument("--erode_px", type=int, default=0)
    ap.add_argument("--flip_ud", action="store_true")
    ap.add_argument("--flip_lr", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # render model (ROI)
    xyz_roi, cams_roi, _ = load_xyz_and_cams(args.model_path)
    gauss_roi = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gauss_roi.load_ply(str(Path(args.model_path)/"point_cloud/iteration_7000/point_cloud.ply"))

    # reference model for pose
    if args.ref_model_path.strip():
        xyz_ref, cams_ref, _ = load_xyz_and_cams(args.ref_model_path)
        xyz_pose = xyz_ref
        cams_data = cams_ref
        pose_source = args.ref_model_path
    else:
        xyz_pose = xyz_roi
        cams_data = cams_roi
        pose_source = args.model_path

    ref = cams_data[0]
    fov_y = 2 * math.atan(ref["height"] / (2 * ref["fy"]))
    fov_x_mod = 2 * math.atan(math.tan(fov_y / 2) * args.multiplier)

    # PCA on xyz_pose
    xyz_cpu = xyz_pose.detach().cpu()
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    _, eigvecs = torch.linalg.eigh(cov)
    normal = eigvecs[:, 0].to(device)
    axis1 = eigvecs[:, 1].to(device)

    center = xyz_pose.mean(dim=0).to(device)

    cam_centers = [np.array(c["position"], dtype=np.float32) for c in cams_data[:10]]
    mean_cam_pos = torch.tensor(np.mean(cam_centers, axis=0), dtype=torch.float32, device=device)
    if torch.dot((mean_cam_pos - center), normal) < 0:
        normal = -normal

    up_axis = axis1 - torch.dot(axis1, normal) * normal
    up_axis = up_axis / (torch.norm(up_axis) + 1e-8)
    right_axis = torch.cross(up_axis, normal, dim=0)

    if abs(args.angle) > 1e-6:
        rad = math.radians(args.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        up_axis = (up_axis * cos_a + right_axis * sin_a)
        up_axis = up_axis / (torch.norm(up_axis) + 1e-8)
        right_axis = torch.cross(up_axis, normal, dim=0)

    min_xyz, _ = torch.min(xyz_pose, dim=0)
    max_xyz, _ = torch.max(xyz_pose, dim=0)
    span_x = (max_xyz[0] - min_xyz[0]).item()
    span_y = (max_xyz[1] - min_xyz[1]).item()
    max_span = max(span_x, span_y)

    base_height = (max_span / 2.0) / math.tan(fov_y / 2.0)
    target_height = base_height / args.zoom
    shift_vec = (right_axis * args.shift_x) + (up_axis * args.shift_y)
    target_center = center + shift_vec
    cam_pos = target_center + normal * target_height

    w2v = look_at_topdown(cam_pos, target_center, up_axis, device=device)
    proj = get_projection_matrix(0.01, 100.0, fov_x_mod, fov_y, device=device).transpose(0, 1)
    full_T = w2v @ proj
    cam = MiniCam(args.render_res, args.render_res, fov_y, fov_x_mod, 0.01, 100.0, w2v, full_T)

    pipe = type("Pipe", (object,), {"compute_cov3D_python": False, "convert_SHs_python": False, "brdf": False, "brdf_mode": "pbbr"})()
    bg = torch.tensor([0.0, 0.0, 0.0], device=device)

    with torch.no_grad():
        out = render(cam, gauss_roi, pipe, bg)
        a = safe_get_alpha(out)
        if a is None:
            raise RuntimeError("render() output has no alpha/accum key")
        alpha = a.detach().float().clamp(0, 1).cpu().numpy()

    if args.flip_ud:
        alpha = np.flipud(alpha)
    if args.flip_lr:
        alpha = np.fliplr(alpha)

    alpha_u8 = (alpha * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "alpha_u8.png"), alpha_u8)

    mask = (alpha > float(args.alpha_eps)).astype(np.uint8) * 255
    if args.erode_px and args.erode_px > 0:
        k = 2 * int(args.erode_px) + 1
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    out_mask = out_dir / ("mask_alpha_a%02d.png" % int(round(args.alpha_eps * 100)))
    cv2.imwrite(str(out_mask), mask)

    valid_ratio = float((mask > 0).mean())
    (out_dir / "mask_meta.json").write_text(json.dumps({
        "render_model_path": args.model_path,
        "pose_source_model_path": pose_source,
        "render_res": int(args.render_res),
        "alpha_eps": float(args.alpha_eps),
        "erode_px": int(args.erode_px),
        "flip_ud": bool(args.flip_ud),
        "flip_lr": bool(args.flip_lr),
        "valid_ratio": valid_ratio
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] wrote:", out_mask, "valid_ratio=", valid_ratio, "pose_source=", pose_source)


if __name__ == "__main__":
    main()