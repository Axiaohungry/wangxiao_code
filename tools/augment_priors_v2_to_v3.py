# tools/augment_priors_v2_to_v3.py
# Create priors_v3 by:
# - semantic scalar -> one-hot (C)
# - add sun_facing (1): clamp(dot(normal, light_dir), 0, 1)
#   where light_dir = -ray_dir
# - ray_dir is derived from point-cloud PCA topdown right_axis (horizontal) + ground normal (vertical),
#   with explicit sun elevation.
#
# Minimal deps: numpy/torch/json. Includes minimal PLY xyz reader.

import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


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


# --- Minimal PLY reader for x,y,z (supports ascii or binary_little_endian) ---
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


def compute_topdown_axes_from_xyz_and_cameras(
    xyz: np.ndarray,
    cameras_json: str,
    angle_deg: float
) -> Dict[str, np.ndarray]:
    """
    PCA on xyz to get ground normal (smallest eigenvector) + an in-plane axis.
    Orient normal using mean camera position.
    Then compute right/up axes and apply in-plane rotation.
    """
    x = xyz.astype(np.float64)
    center = x.mean(axis=0)
    xc = x - center[None, :]
    cov = (xc.T @ xc) / max(1, x.shape[0])
    w, v = np.linalg.eigh(cov)  # ascending
    normal = v[:, 0].astype(np.float32)   # smallest variance axis (ground normal)
    axis1  = v[:, 1].astype(np.float32)

    cams = json.load(open(cameras_json, "r", encoding="utf-8"))
    cam_pos = np.array([c["position"] for c in cams[:10]], dtype=np.float32).mean(axis=0)
    if float(np.dot((cam_pos - center.astype(np.float32)), normal)) < 0.0:
        normal = -normal

    up = axis1 - np.dot(axis1, normal) * normal
    up = up / (np.linalg.norm(up) + 1e-8)
    right = np.cross(up, normal).astype(np.float32)
    right = right / (np.linalg.norm(right) + 1e-8)

    if abs(angle_deg) > 1e-6:
        rad = np.deg2rad(angle_deg)
        cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
        new_up = up * cos_a + right * sin_a
        up = new_up / (np.linalg.norm(new_up) + 1e-8)
        right = np.cross(up, normal).astype(np.float32)
        right = right / (np.linalg.norm(right) + 1e-8)

    return {"center": center.astype(np.float32), "normal": normal, "up": up, "right": right}


def infer_semantic_mode(sem_col: torch.Tensor, num_classes: int) -> str:
    mx = float(sem_col.max().item())
    if num_classes > 2 and mx <= 1.0 + 1e-6:
        return "scaled_01"
    return "raw_id"


def decode_semantic_ids(sem_col: torch.Tensor, num_classes: int, mode: str) -> torch.Tensor:
    if mode == "scaled_01":
        return (sem_col * float(num_classes - 1)).round().clamp(0, num_classes - 1).long()
    if mode == "raw_id":
        return sem_col.round().clamp(0, num_classes - 1).long()
    raise ValueError(f"Unknown semantic_value_mode: {mode}")


def _np_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors_in", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--out_priors", required=True)
    ap.add_argument("--out_stats", required=True)

    ap.add_argument("--angle", type=float, default=-31.0, help="match your topdown angle")
    ap.add_argument("--sun_dir_mode", default="right_to_left", choices=["right_to_left", "left_to_right"],
                    help="Defines ray direction in topdown: right_to_left means rays travel from right -> left.")
    ap.add_argument("--sun_elev_deg", type=float, default=75.0,
                    help="Sun elevation above horizon in degrees (used to add vertical component).")
    ap.add_argument("--semantic_index", type=int, default=-1, help="which channel is semantic scalar (default last)")
    ap.add_argument("--semantic_num_classes", type=int, default=3)
    ap.add_argument("--semantic_value_mode", default="auto",
                    choices=["auto", "raw_id", "scaled_01"],
                    help="auto: infer from value range; raw_id: 0..C-1; scaled_01: 0..1 scaled by (C-1)")
    args = ap.parse_args()

    pri = load_tensor_from_pt(args.priors_in).float()
    if pri.dim() != 2 or pri.shape[1] < 7:
        raise ValueError(f"Expected priors [N,>=7], got {tuple(pri.shape)}")
    N = int(pri.shape[0])

    xyz = read_ply_xyz(args.ply)
    if xyz.shape[0] != N:
        raise ValueError(f"PLY vertex count {xyz.shape[0]} != priors N {N}")

    axes = compute_topdown_axes_from_xyz_and_cameras(xyz, args.cameras, args.angle)
    right = axes["right"]
    ground_n = axes["normal"]

    # --- Build ray_dir (sun -> scene), then light_dir = -ray_dir (scene -> sun) ---
    elev = float(args.sun_elev_deg)
    elev_rad = np.deg2rad(elev)
    cos_e = float(np.cos(elev_rad))
    sin_e = float(np.sin(elev_rad))

    # horizontal ray direction in ground plane
    # right_to_left: rays travel to left => along -right
    if args.sun_dir_mode == "right_to_left":
        ray_h = (-right).astype(np.float32)
    else:
        ray_h = (right).astype(np.float32)

    ray_h = _np_normalize(ray_h)
    ground_n = _np_normalize(ground_n)

    # add vertical component: rays go downward => -ground_n * sin(elev)
    ray_dir = _np_normalize(ray_h * cos_e - ground_n * sin_e)

    # enforce "downward": dot(ray_dir, ground_n) should be negative
    if float(np.dot(ray_dir, ground_n)) > 0.0:
        ray_dir = (-ray_dir).astype(np.float32)

    light_dir = (-ray_dir).astype(np.float32)  # direction from surface to sun

    # normals assumed in priors ch0..2 (world)
    n = pri[:, 0:3]
    n = n / (torch.norm(n, dim=1, keepdim=True) + 1e-8)
    light = torch.from_numpy(light_dir).view(1, 3).float()
    dot = (n * light).sum(dim=1)  # ideally [-1,1]
    sun_facing = torch.clamp(dot, 0.0, 1.0).unsqueeze(1)  # [N,1]

    # semantic scalar -> ids -> onehot
    si = args.semantic_index if args.semantic_index >= 0 else (pri.shape[1] + args.semantic_index)
    sem_col = pri[:, si]

    if args.semantic_value_mode == "auto":
        sem_mode = infer_semantic_mode(sem_col, int(args.semantic_num_classes))
    else:
        sem_mode = args.semantic_value_mode

    sem_id = decode_semantic_ids(sem_col, int(args.semantic_num_classes), sem_mode)
    sem_oh = torch.zeros((N, int(args.semantic_num_classes)), dtype=torch.float32)
    sem_oh.scatter_(1, sem_id.view(-1, 1), 1.0)

    # build v3: keep all channels except semantic scalar, then append onehot + sun_facing
    keep = torch.cat([pri[:, :si], pri[:, si + 1:]], dim=1)
    pri_v3 = torch.cat([keep, sem_oh, sun_facing], dim=1).contiguous()

    Path(args.out_priors).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pri_v3, args.out_priors)

    def tstats(t: torch.Tensor):
        return {
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
            "std": float(t.std().item()),
        }

    # semantic histogram (sample for speed)
    samp = sem_id[:200000]
    hist = torch.bincount(samp, minlength=int(args.semantic_num_classes)).cpu().tolist()

    # dot stats (sample)
    dot_s = dot[:200000]
    dot_stats = tstats(dot_s)

    stats = {
        "priors_in": args.priors_in,
        "out_priors": args.out_priors,
        "N": N,
        "dim_in": int(pri.shape[1]),
        "dim_out": int(pri_v3.shape[1]),
        "topdown_angle_deg": float(args.angle),

        "sun_dir_mode": args.sun_dir_mode,
        "sun_elev_deg": float(args.sun_elev_deg),
        "sun_ray_dir_world": [float(x) for x in ray_dir.tolist()],
        "sun_light_dir_world": [float(x) for x in light_dir.tolist()],
        "dot_stats_sample200k": dot_stats,
        "sun_facing_stats": tstats(sun_facing),

        "semantic_index_removed": int(si),
        "semantic_num_classes": int(args.semantic_num_classes),
        "semantic_value_mode": sem_mode,
        "semantic_hist_sample200k": hist,
        "channels_layout": "keep(v2_without_sem_scalar),SemanticOH(C),SunFacing(1)",
        "notes": [
            "sun_facing = clamp(dot(normal, light_dir), 0, 1)",
            "light_dir = -ray_dir",
            "ray_dir follows sun_dir_mode + sun_elev_deg and is enforced to be downward wrt ground normal",
        ],
    }
    Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Wrote:")
    print(" ", args.out_priors)
    print(" ", args.out_stats)
    print(f"[V3] priors shape: {tuple(pri_v3.shape)}  (semantic_mode={sem_mode})")


if __name__ == "__main__":
    main()
