# tools/vis_priors_topdown.py
# Visualize priors (v2/v3) by projecting per-point attributes back to topdown 2D.
# Outputs (raw):
#   - sem_id_topdown.png
#   - sun_topdown.png / sun_topdown_jet.png
#   - shadowA_topdown.png / shadowB_topdown.png
#   - overlay_sem_sun.png
#   - density_topdown.png
# And optionally (enhanced):
#   - *_enh.png
#
# Minimal deps: numpy/torch/opencv/json

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import cv2


# --- Minimal PLY reader for x,y,z (ascii or binary_little_endian) ---
PLY_TYPE_TO_DTYPE = {
    "char": np.int8, "int8": np.int8,
    "uchar": np.uint8, "uint8": np.uint8,
    "short": np.int16, "int16": np.int16,
    "ushort": np.uint16, "uint16": np.uint16,
    "int": np.int32, "int32": np.int32,
    "uint": np.uint32, "uint32": np.uint32,
    "float": np.float32, "float32": np.float32,
    "double": np.float64, "float64": np.float64,
}


def read_ply_xyz(ply_path: str) -> np.ndarray:
    with open(ply_path, "rb") as f:
        header_lines: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY header")
            s = line.decode("ascii", errors="ignore").strip()
            header_lines.append(s)
            if s == "end_header":
                break

        fmt = None
        vertex_count = None
        in_vertex = False
        props: List[Tuple[str, str]] = []

        for s in header_lines:
            if s.startswith("format "):
                fmt = s.split()[1]
            elif s.startswith("element vertex "):
                vertex_count = int(s.split()[2])
                in_vertex = True
                props = []
            elif s.startswith("element "):
                if in_vertex and not s.startswith("element vertex "):
                    in_vertex = False
            elif in_vertex and s.startswith("property "):
                toks = s.split()
                if toks[1] == "list":
                    continue
                ptype, pname = toks[1], toks[2]
                props.append((ptype, pname))

        if fmt is None or vertex_count is None:
            raise ValueError("PLY header missing format or vertex count")

        names = [p[1] for p in props]
        try:
            ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
        except ValueError:
            raise ValueError("PLY vertex properties must include x,y,z")

        if fmt == "ascii":
            xyz = np.zeros((vertex_count, 3), dtype=np.float32)
            for i in range(vertex_count):
                parts = f.readline().decode("ascii", errors="ignore").strip().split()
                xyz[i, 0] = float(parts[ix])
                xyz[i, 1] = float(parts[iy])
                xyz[i, 2] = float(parts[iz])
            return xyz

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {fmt} (expected ascii or binary_little_endian)")

        dtype_fields = []
        for ptype, pname in props:
            if ptype not in PLY_TYPE_TO_DTYPE:
                raise ValueError(f"Unsupported PLY property type: {ptype}")
            dtype_fields.append((pname, PLY_TYPE_TO_DTYPE[ptype]))
        row_dtype = np.dtype(dtype_fields)

        data = np.fromfile(f, dtype=row_dtype, count=vertex_count)
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
        return xyz


def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_tensor_from_pt(path: str) -> torch.Tensor:
    obj = safe_torch_load(path)
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


def look_at_topdown(cam_pos: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    z_axis = target - cam_pos
    dist = torch.norm(z_axis)
    if dist < 1e-6:
        return torch.eye(4, dtype=torch.float32)
    z_axis = z_axis / dist
    x_axis = torch.cross(up, z_axis, dim=0)
    if torch.norm(x_axis) < 1e-6:
        tmp = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        x_axis = torch.cross(tmp, z_axis, dim=0)
    x_axis = x_axis / (torch.norm(x_axis) + 1e-8)
    y_axis = torch.cross(z_axis, x_axis, dim=0)
    y_axis = y_axis / (torch.norm(y_axis) + 1e-8)
    R = torch.stack([x_axis, y_axis, z_axis], dim=0)
    T = -torch.matmul(R, cam_pos)
    w2v = torch.eye(4, dtype=torch.float32)
    w2v[:3, :3] = R
    w2v[:3, 3] = T
    return w2v.transpose(0, 1).contiguous()


def get_projection_matrix(znear: float, zfar: float, fovX: float, fovY: float) -> torch.Tensor:
    tanHalfFovY = np.tan(fovY / 2.0)
    tanHalfFovX = np.tan(fovX / 2.0)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4, dtype=torch.float32)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def compute_topdown_fullT(
    xyz: torch.Tensor,
    cams_data: List[Dict[str, Any]],
    W: int,
    H: int,
    zoom: float,
    shift_x: float,
    shift_y: float,
    angle_deg: float,
    multiplier: float,
) -> torch.Tensor:
    center = xyz.mean(dim=0)
    min_xyz, _ = torch.min(xyz, dim=0)
    max_xyz, _ = torch.max(xyz, dim=0)
    span_x = (max_xyz[0] - min_xyz[0]).item()
    span_y = (max_xyz[1] - min_xyz[1]).item()
    max_span = max(span_x, span_y)

    ref = cams_data[0]
    fov_y = 2.0 * np.arctan(ref["height"] / (2.0 * ref["fy"]))
    fov_x_mod = 2.0 * np.arctan(np.tan(fov_y / 2.0) * multiplier)

    # PCA (CPU)
    xyz_cpu = xyz.detach().cpu()
    xyz_centered = xyz_cpu - xyz_cpu.mean(dim=0, keepdim=True)
    cov = xyz_centered.T @ xyz_centered / xyz_centered.shape[0]
    _, eigvecs = torch.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    axis1 = eigvecs[:, 1]

    cam_centers = [np.array(c["position"], dtype=np.float32) for c in cams_data[:10]]
    mean_cam_pos = torch.tensor(np.mean(cam_centers, axis=0), dtype=torch.float32)
    if torch.dot((mean_cam_pos - center.cpu()), normal) < 0:
        normal = -normal

    up_axis = axis1 - torch.dot(axis1, normal) * normal
    up_axis = up_axis / (torch.norm(up_axis) + 1e-8)
    right_axis = torch.cross(up_axis, normal, dim=0)

    if abs(angle_deg) > 1e-6:
        rad = np.deg2rad(angle_deg)
        cos_a = float(np.cos(rad))
        sin_a = float(np.sin(rad))
        new_up = up_axis * cos_a + right_axis * sin_a
        up_axis = new_up / (torch.norm(up_axis) + 1e-8)
        right_axis = torch.cross(up_axis, normal, dim=0)

    base_height = (max_span / 2.0) / np.tan(fov_y / 2.0)
    target_height = base_height / zoom

    shift_vec = (right_axis * shift_x) + (up_axis * shift_y)
    target_center = center.cpu() + shift_vec
    cam_pos = target_center + normal * target_height

    w2v = look_at_topdown(cam_pos, target_center, up_axis)
    proj = get_projection_matrix(0.01, 100.0, fov_x_mod, fov_y).transpose(0, 1).contiguous()
    full_T = w2v @ proj
    return full_T


def splat_scalar(H: int, W: int, xs: np.ndarray, ys: np.ndarray, val: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    acc = np.zeros((H, W), dtype=np.float32)
    if mode == "max":
        np.maximum.at(acc, (ys, xs), val)
        return acc
    if mode == "mean":
        cnt = np.zeros((H, W), dtype=np.float32)
        np.add.at(acc, (ys, xs), val)
        np.add.at(cnt, (ys, xs), 1.0)
        return acc / np.maximum(cnt, 1.0)
    raise ValueError("splat_mode must be max or mean")


def enhance_gray_u8(img_u8: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    assert img_u8.dtype == np.uint8 and img_u8.ndim == 2
    m = img_u8 > 0
    if not np.any(m):
        return img_u8
    v = img_u8[m].astype(np.float32)
    lo = np.percentile(v, 1.0)
    hi = np.percentile(v, 99.5)
    x = (img_u8.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    x = np.clip(x, 0.0, 1.0)
    x = np.power(x, gamma)
    return (x * 255.0).astype(np.uint8)


def enhance_bgr_u8(img_u8: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    assert img_u8.dtype == np.uint8 and img_u8.ndim == 3
    g = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    m = g > 0
    if not np.any(m):
        return img_u8
    v = g[m].astype(np.float32)
    lo = np.percentile(v, 1.0)
    hi = np.percentile(v, 99.5)
    x = (img_u8.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    x = np.clip(x, 0.0, 1.0)
    x = np.power(x, gamma)
    return (x * 255.0).astype(np.uint8)


def maybe_dilate(img: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return img
    ker = np.ones((k, k), np.uint8)
    return cv2.dilate(img, ker, iterations=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors_pt", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--W", type=int, default=2048)
    ap.add_argument("--H", type=int, default=2048)

    ap.add_argument("--zoom", type=float, default=5.4)
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--shift_y", type=float, default=-1.2)
    ap.add_argument("--angle", type=float, default=-31.0)
    ap.add_argument("--multiplier", type=float, default=0.85)

    ap.add_argument("--sample_points", type=int, default=1200000)
    ap.add_argument("--splat_mode", type=str, default="max", choices=["max", "mean"])

    # orientation fix
    ap.add_argument("--flip_ud", action="store_true", help="flip vertically to match topdown_final convention")
    ap.add_argument("--flip_lr", action="store_true", help="flip horizontally (usually not needed)")

    # readability
    ap.add_argument("--dilate", type=int, default=2, help="dilation kernel size for readability (0 disables)")
    ap.add_argument("--write_enh", action="store_true", help="also write *_enh.png with percentile+gamma")
    ap.add_argument("--enh_gamma", type=float, default=0.5)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pri = load_tensor_from_pt(args.priors_pt).float()
    if pri.ndim != 2:
        raise ValueError(f"priors must be 2D, got {tuple(pri.shape)}")
    N, D = pri.shape
    print(f"[Priors] {args.priors_pt} shape={(N, D)}")

    xyz = read_ply_xyz(args.ply)
    if xyz.shape[0] != N:
        raise ValueError(f"PLY N={xyz.shape[0]} != priors N={N}")

    cams = json.loads(Path(args.cameras).read_text(encoding="utf-8"))

    xyz_t = torch.from_numpy(xyz).float()
    full_T = compute_topdown_fullT(
        xyz=xyz_t,
        cams_data=cams,
        W=args.W, H=args.H,
        zoom=args.zoom,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        angle_deg=args.angle,
        multiplier=args.multiplier,
    )
    full_T_np = full_T.numpy().astype(np.float32)

    keep = min(int(args.sample_points), int(N))
    if keep < N:
        idx = np.linspace(0, N - 1, keep, dtype=np.int64)
        xyz_s = xyz[idx]
        pri_s = pri[idx].cpu().numpy()
    else:
        xyz_s = xyz
        pri_s = pri.cpu().numpy()

    H, W = int(args.H), int(args.W)

    ones = np.ones((xyz_s.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([xyz_s.astype(np.float32), ones], axis=1)
    clip = hom @ full_T_np
    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    ndc = clip[:, 0:3] / w

    # NOTE: this mapping is the source of the vertical flip issue.
    x = (ndc[:, 0] * 0.5 + 0.5) * (W - 1)
    y = (0.5 - ndc[:, 1] * 0.5) * (H - 1)

    in_view = (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)

    xi = np.clip(np.round(x).astype(np.int32), 0, W - 1)
    yi = np.clip(np.round(y).astype(np.int32), 0, H - 1)

    # keep only in-view for clean vis
    xi = xi[in_view]
    yi = yi[in_view]
    pri_s = pri_s[in_view]

    # optional flips to match topdown_final
    if args.flip_lr:
        xi = (W - 1) - xi
    if args.flip_ud:
        yi = (H - 1) - yi

    in_view_ratio = float(xi.shape[0]) / float(max(1, keep))
    print(f"[Project] in_view={xi.shape[0]}/{keep} = {in_view_ratio:.4f} flip_ud={args.flip_ud} flip_lr={args.flip_lr}")

    # density
    density = np.zeros((H, W), dtype=np.uint16)
    np.add.at(density, (yi, xi), 1)
    density_u8 = np.clip(density.astype(np.float32) / max(1.0, density.max()) * 255.0, 0, 255).astype(np.uint8)
    density_u8 = maybe_dilate(density_u8, args.dilate)
    cv2.imwrite(str(out_dir / "density_topdown.png"), density_u8)

    # parse channels
    sem_id = None
    sun = None
    shadowA = None
    shadowB = None

    if D == 11:
        sem_oh = pri_s[:, 7:10]
        sem_id = np.argmax(sem_oh, axis=1).astype(np.uint8)
        sun = pri_s[:, 10].astype(np.float32)
        shadowA = pri_s[:, 5].astype(np.float32)
        shadowB = pri_s[:, 6].astype(np.float32)
    elif D == 8:
        sem_id = np.round(pri_s[:, 7] * 2.0).clip(0, 2).astype(np.uint8)
        shadowA = pri_s[:, 5].astype(np.float32)
        shadowB = pri_s[:, 6].astype(np.float32)
    elif D == 7:
        sem_id = np.round(pri_s[:, 6] * 2.0).clip(0, 2).astype(np.uint8)
        shadowA = pri_s[:, 5].astype(np.float32)
    else:
        print("[Warn] Unrecognized priors dim; only density will be written.")

    # semantic palette (BGR)
    palette = {
        0: np.array([0, 200, 0], dtype=np.uint8),      # veg
        1: np.array([0, 0, 220], dtype=np.uint8),      # building
        2: np.array([220, 220, 220], dtype=np.uint8),  # road
    }

    sem_img = None
    if sem_id is not None:
        sem_img = np.zeros((H, W, 3), dtype=np.uint8)
        col = np.zeros((sem_id.shape[0], 3), dtype=np.uint8)
        for k, c in palette.items():
            m = (sem_id == k)
            if np.any(m):
                col[m] = c
        sem_img[yi, xi, :] = col
        sem_img = maybe_dilate(sem_img, args.dilate)
        cv2.imwrite(str(out_dir / "sem_id_topdown.png"), sem_img)

    if sun is not None:
        sun = np.clip(sun, 0.0, 1.0)
        sun_map = splat_scalar(H, W, xi, yi, sun, mode=args.splat_mode)
        sun_u8 = (sun_map * 255.0).astype(np.uint8)
        sun_u8 = maybe_dilate(sun_u8, args.dilate)
        cv2.imwrite(str(out_dir / "sun_topdown.png"), sun_u8)
        cv2.imwrite(str(out_dir / "sun_topdown_jet.png"), cv2.applyColorMap(sun_u8, cv2.COLORMAP_JET))

        if sem_img is not None:
            overlay = (sem_img.astype(np.float32) * (sun_u8.astype(np.float32) / 255.0)[..., None]).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / "overlay_sem_sun.png"), overlay)

    if shadowA is not None:
        sh = np.clip(shadowA, 0.0, 1.0)
        sh_map = splat_scalar(H, W, xi, yi, sh, mode=args.splat_mode)
        sh_u8 = (sh_map * 255.0).astype(np.uint8)
        sh_u8 = maybe_dilate(sh_u8, args.dilate)
        cv2.imwrite(str(out_dir / "shadowA_topdown.png"), sh_u8)

    if shadowB is not None:
        sh = np.clip(shadowB, 0.0, 1.0)
        sh_map = splat_scalar(H, W, xi, yi, sh, mode=args.splat_mode)
        sh_u8 = (sh_map * 255.0).astype(np.uint8)
        sh_u8 = maybe_dilate(sh_u8, args.dilate)
        cv2.imwrite(str(out_dir / "shadowB_topdown.png"), sh_u8)

    # enhanced outputs
    if args.write_enh:
        for fn in ["sun_topdown.png", "shadowA_topdown.png", "shadowB_topdown.png", "density_topdown.png"]:
            p = out_dir / fn
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 2:
                    enh = enhance_gray_u8(img, gamma=args.enh_gamma)
                else:
                    enh = enhance_bgr_u8(img, gamma=args.enh_gamma)
                cv2.imwrite(str(out_dir / (p.stem + "_enh.png")), enh)

        for fn in ["sem_id_topdown.png", "overlay_sem_sun.png", "sun_topdown_jet.png"]:
            p = out_dir / fn
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                enh = enhance_bgr_u8(img, gamma=args.enh_gamma) if img.ndim == 3 else enhance_gray_u8(img, gamma=args.enh_gamma)
                cv2.imwrite(str(out_dir / (p.stem + "_enh.png")), enh)

    # stats
    pixel_coverage = float((density > 0).mean())
    stats = {
        "priors_path": str(args.priors_pt),
        "ply": str(args.ply),
        "cameras": str(args.cameras),
        "H": H, "W": W,
        "sample_points": int(keep),
        "in_view_points": int(xi.shape[0]),
        "in_view_ratio": in_view_ratio,
        "pixel_coverage": pixel_coverage,
        "dim": int(D),
        "flip_ud": bool(args.flip_ud),
        "flip_lr": bool(args.flip_lr),
        "splat_mode": args.splat_mode,
        "dilate": int(args.dilate),
    }
    (out_dir / "vis_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Wrote:", str(out_dir))
    print("  - sem_id_topdown.png / *_enh.png")
    print("  - sun_topdown*.png / *_enh.png")
    print("  - shadowA/B_topdown.png / *_enh.png")
    print("  - overlay_sem_sun.png / *_enh.png")
    print("  - density_topdown.png / *_enh.png")
    print("  - vis_stats.json")


if __name__ == "__main__":
    main()
