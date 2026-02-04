# tools/crop_roi_pipeline.py
# Precheck -> crop -> postcheck for ROI (frustum in-view) hard cut.
# Minimal, no refactor of existing training code.

import os, sys, json, hashlib, shutil
from pathlib import Path
import numpy as np
import torch

try:
    from plyfile import PlyData, PlyElement
except Exception as e:
    PlyData = None
    PlyElement = None

def sha256_file(p: Path, chunk=1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

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

def dump_text(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def dump_json(p: Path, d: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

def get_projection_matrix(znear, zfar, fovX, fovY) -> np.ndarray:
    tanHalfFovY = np.tan(fovY / 2.0)
    tanHalfFovX = np.tan(fovX / 2.0)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = np.zeros((4, 4), dtype=np.float32)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def look_at_topdown(cam_pos, target, up) -> np.ndarray:
    cam_pos = cam_pos.astype(np.float32)
    target = target.astype(np.float32)
    up = up.astype(np.float32)

    z_axis = target - cam_pos
    dist = np.linalg.norm(z_axis)
    if dist < 1e-6:
        return np.eye(4, dtype=np.float32)
    z_axis = z_axis / dist

    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        x_axis = np.cross(tmp, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    R = np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float32)
    T = (-R @ cam_pos).astype(np.float32)

    w2v = np.eye(4, dtype=np.float32)
    w2v[:3, :3] = R
    w2v[:3, 3] = T
    return w2v.T.copy()  # keep consistent with your training code

def compute_fullT_and_inview_mask(xyz: np.ndarray, cams_data: list,
                                  zoom: float, shift_x: float, shift_y: float,
                                  angle_deg: float, multiplier: float) -> tuple[np.ndarray, np.ndarray]:
    xyz = xyz.astype(np.float32)
    center = xyz.mean(axis=0)
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    span_x = float(mx[0] - mn[0])
    span_y = float(mx[1] - mn[1])
    max_span = max(span_x, span_y)

    ref = cams_data[0]
    fov_y = 2.0 * np.arctan(ref["height"] / (2.0 * ref["fy"]))
    fov_x_mod = 2.0 * np.arctan(np.tan(fov_y / 2.0) * float(multiplier))

    # PCA
    xc = xyz - xyz.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / max(1, xc.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    eigvecs = eigvecs.astype(np.float32)
    normal = eigvecs[:, 0]
    axis1 = eigvecs[:, 1]

    cam_centers = [np.array(c["position"], dtype=np.float32) for c in cams_data[:10]]
    mean_cam = np.mean(np.stack(cam_centers, axis=0), axis=0).astype(np.float32)
    if float(np.dot((mean_cam - center), normal)) < 0:
        normal = -normal

    up_axis = axis1 - float(np.dot(axis1, normal)) * normal
    up_axis = up_axis / (np.linalg.norm(up_axis) + 1e-8)
    right_axis = np.cross(up_axis, normal).astype(np.float32)

    if abs(angle_deg) > 1e-6:
        rad = np.deg2rad(angle_deg)
        cos_a = float(np.cos(rad))
        sin_a = float(np.sin(rad))
        new_up = up_axis * cos_a + right_axis * sin_a
        up_axis = new_up / (np.linalg.norm(new_up) + 1e-8)
        right_axis = np.cross(up_axis, normal).astype(np.float32)

    base_height = (max_span / 2.0) / np.tan(fov_y / 2.0)
    target_height = base_height / float(zoom)
    shift_vec = right_axis * float(shift_x) + up_axis * float(shift_y)
    target_center = center + shift_vec
    cam_pos = target_center + normal * target_height

    w2v = look_at_topdown(cam_pos, target_center, up_axis)
    proj = get_projection_matrix(0.01, 100.0, fov_x_mod, fov_y).T.copy()
    full_T = (w2v @ proj).astype(np.float32)

    # in-view mask: hom @ full_T -> clip, ndc in [-1,1]
    hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1), np.float32)], axis=1)
    clip = hom @ full_T
    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    ndc = clip[:, 0:3] / w
    in_view = (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
    return full_T, in_view

def write_manifest(run_dir: Path, paths: list[Path]):
    lines = []
    for p in paths:
        if p.exists() and p.is_file():
            lines.append(f"{p.as_posix()}\t{p.stat().st_size}\t{sha256_file(p)}")
    dump_text(run_dir / "manifest_snapshot.txt", "\n".join(lines) + "\n")

def precheck(args, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    ply = Path(args.ply)
    cams = Path(args.cameras)
    pri = Path(args.priors_path)

    report = []
    report.append(f"[PRECHECK] run_base={args.run_base} run_crop={args.run_crop}")
    report.append(f"[PATH] ply={ply} exists={ply.exists()} size={ply.stat().st_size if ply.exists() else -1}")
    report.append(f"[PATH] cameras={cams} exists={cams.exists()} size={cams.stat().st_size if cams.exists() else -1}")
    report.append(f"[PATH] priors={pri} exists={pri.exists()} size={pri.stat().st_size if pri.exists() else -1}")

    if PlyData is None:
        report.append("[FATAL] plyfile not installed. Run: pip install plyfile")
        dump_text(run_dir / "check_report.txt", "\n".join(report) + "\n")
        return False

    # load ply
    pd = PlyData.read(str(ply))
    if "vertex" not in pd:
        report.append("[FATAL] PLY has no vertex element.")
        dump_text(run_dir / "check_report.txt", "\n".join(report) + "\n")
        return False
    v = pd["vertex"].data
    N_ply = int(v.shape[0])
    report.append(f"[PLY] vertices={N_ply} fields={list(v.dtype.names)}")

    # load priors
    pri_t = load_tensor_from_pt(str(pri), map_location="cpu").float()
    if pri_t.dim() != 2:
        report.append(f"[FATAL] priors dim !=2 : {tuple(pri_t.shape)}")
        dump_text(run_dir / "check_report.txt", "\n".join(report) + "\n")
        return False
    N_pri, D = int(pri_t.shape[0]), int(pri_t.shape[1])
    report.append(f"[PRIORS] shape=({N_pri},{D}) finite={bool(torch.isfinite(pri_t).all().item())}")

    if N_ply != N_pri:
        report.append(f"[FATAL] N mismatch: ply={N_ply} priors={N_pri}")
        dump_text(run_dir / "check_report.txt", "\n".join(report) + "\n")
        return False

    # v3 sanity (if 11-dim)
    if D == 11:
        sem = pri_t[:200000, 7:10]
        s = sem.sum(dim=1)
        report.append(f"[V3] semOH_sum min/max/mean = {float(s.min()):.6f}/{float(s.max()):.6f}/{float(s.mean()):.6f}")
        sun = pri_t[:200000, 10]
        report.append(f"[V3] sun min/max/mean/std = {float(sun.min()):.6f}/{float(sun.max()):.6f}/{float(sun.mean()):.6f}/{float(sun.std()):.6f}")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    cams_data = json.loads(Path(args.cameras).read_text(encoding="utf-8"))
    full_T, in_view = compute_fullT_and_inview_mask(
        xyz, cams_data, args.zoom, args.shift_x, args.shift_y, args.angle, args.multiplier
    )
    in_ratio = float(in_view.mean())
    np.save(str(artifacts / "point_inview_mask.npy"), in_view.astype(np.uint8))
    dump_json(artifacts / "point_inview_audit.json", {
        "in_view_ratio": in_ratio,
        "N": int(xyz.shape[0]),
        "camera_params": {"zoom": args.zoom, "shift_x": args.shift_x, "shift_y": args.shift_y,
                          "angle": args.angle, "multiplier": args.multiplier}
    })
    report.append(f"[INVIEW] ratio={in_ratio:.6f} mask_saved=artifacts/point_inview_mask.npy")

    # manifest
    write_manifest(run_dir, [ply, cams, pri, artifacts / "point_inview_mask.npy", artifacts / "point_inview_audit.json"])
    dump_json(run_dir / "run_meta.json", {
        "run_base": args.run_base,
        "run_crop": args.run_crop,
        "mode": "precheck",
        "source": {"ply": str(ply), "cameras": str(cams), "priors": str(pri)},
        "camera_params": {"zoom": args.zoom, "shift_x": args.shift_x, "shift_y": args.shift_y,
                          "angle": args.angle, "multiplier": args.multiplier},
        "in_view_ratio_before": in_ratio,
        "N_before": int(xyz.shape[0]),
    })
    dump_text(run_dir / "check_report.txt", "\n".join(report) + "\n")
    return True

def do_crop(args, run_dir: Path):
    artifacts = run_dir / "artifacts"
    mask_p =artifacts / "point_inview_mask.npy"
    if not mask_p.exists():
        raise FileNotFoundError(f"mask not found: {mask_p} (run precheck first)")

    in_view = np.load(str(mask_p)).astype(np.uint8) > 0
    ply_in = Path(args.ply)
    pri_in = Path(args.priors_path)
    cams_in = Path(args.cameras)

    if PlyData is None:
        raise RuntimeError("plyfile not installed. Run: pip install plyfile")

    pd = PlyData.read(str(ply_in))
    v = pd["vertex"].data
    N = int(v.shape[0])
    if in_view.shape[0] != N:
        raise ValueError(f"mask N mismatch: {in_view.shape[0]} vs ply {N}")

    pri_t = load_tensor_from_pt(str(pri_in), map_location="cpu").float()
    if int(pri_t.shape[0]) != N:
        raise ValueError(f"priors N mismatch: {int(pri_t.shape[0])} vs ply {N}")

    keep = in_view
    N_keep = int(keep.sum())

    # output dirs
    model_out = run_dir / "model_roi"
    ply_out = model_out / "point_cloud" / "iteration_7000" / "point_cloud.ply"
    model_out.mkdir(parents=True, exist_ok=True)
    ply_out.parent.mkdir(parents=True, exist_ok=True)

    pri_out_dir = run_dir / "priors"
    pri_out_dir.mkdir(parents=True, exist_ok=True)
    pri_out = pri_out_dir / "priors_v3_roi.pt"

    # crop vertex
    v2 = v[keep]
    ve = PlyElement.describe(v2, "vertex")
    PlyData([ve], text=pd.text).write(str(ply_out))

    # crop priors
    pri2 = pri_t[torch.from_numpy(keep)]
    torch.save(pri2, str(pri_out))

    # copy cameras.json
    shutil.copy2(str(cams_in), str(model_out / "cameras.json"))

    dump_json(run_dir / "run_meta.json", {
        **json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8")),
        "mode": "crop",
        "N_after": N_keep,
        "in_view_ratio_after_expected": 1.0,
        "outputs": {
            "model_roi": str(model_out),
            "ply_roi": str(ply_out),
            "priors_roi": str(pri_out),
            "mask": str(mask_p),
        }
    })

def postcheck(args, run_dir: Path):
    report = []
    report.append(f"[POSTCHECK] run_base={args.run_base} run_crop={args.run_crop}")

    model_out = run_dir / "model_roi"
    ply_out = model_out / "point_cloud" / "iteration_7000" / "point_cloud.ply"
    pri_out = run_dir / "priors" / "priors_v3_roi.pt"
    cams_out = model_out / "cameras.json"

    ok = True
    for p in [ply_out, pri_out, cams_out]:
        report.append(f"[PATH] {p} exists={p.exists()} size={p.stat().st_size if p.exists() else -1}")
        ok = ok and p.exists()

    if not ok:
        dump_text(run_dir / "check_report.txt", "\n".join(report) + "\n")
        return False

    if PlyData is None:
        report.append("[FATAL] plyfile not installed.")
        dump_text(run_dir / "check_report.txt", "\n".join(report) + "\n")
        return False

    pd = PlyData.read(str(ply_out))
    v = pd["vertex"].data
    N_ply = int(v.shape[0])
    pri_t = load_tensor_from_pt(str(pri_out), map_location="cpu").float()
    N_pri, D = int(pri_t.shape[0]), int(pri_t.shape[1])
    report.append(f"[CROP] ply_N={N_ply} priors_N={N_pri} D={D} finite={bool(torch.isfinite(pri_t).all().item())}")

    if N_ply != N_pri:
        report.append("[FATAL] N mismatch after crop.")
        dump_text(run_dir / "check_report.txt", "\n".join(report) + "\n")
        return False

    # recompute in-view ratio on cropped points (should be ~1)
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    cams_data = json.loads(Path(args.cameras).read_text(encoding="utf-8"))
    _, in_view = compute_fullT_and_inview_mask(
        xyz, cams_data, args.zoom, args.shift_x, args.shift_y, args.angle, args.multiplier
    )
    in_ratio = float(in_view.mean())
    report.append(f"[INVIEW_AFTER] ratio={in_ratio:.6f} (expect >0.99)")

    # manifest refresh
    write_manifest(run_dir, [ply_out, pri_out, cams_out, run_dir / "check_report.txt"])
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    meta["mode"] = "postcheck"
    meta["in_view_ratio_after"] = in_ratio
    dump_json(run_dir / "run_meta.json", meta)

    dump_text(run_dir / "check_report.txt", "\n".join(report) + "\n")
    return in_ratio > 0.99

def parse():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["precheck", "crop", "postcheck"])
    ap.add_argument("--run_base", required=True)
    ap.add_argument("--run_crop", required=True)

    ap.add_argument("--model_path", default="output/debug_run")
    ap.add_argument("--ply", default="output/debug_run/point_cloud/iteration_7000/point_cloud.ply")
    ap.add_argument("--cameras", default="output/debug_run/cameras.json")
    ap.add_argument("--priors_path", required=False, default="")

    ap.add_argument("--zoom", type=float, default=5.4)
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--shift_y", type=float, default=-1.2)
    ap.add_argument("--angle", type=float, default=-31.0)
    ap.add_argument("--multiplier", type=float, default=0.85)
    return ap.parse_args()

def main():
    args = parse()
    root = Path("output") / "runs" / args.run_crop
    root.mkdir(parents=True, exist_ok=True)

    # resolve priors_path: if not given, infer from run_base
    if not args.priors_path:
        args.priors_path = str(Path("output") / "runs" / args.run_base / "priors" / "priors_v3_2shadow_semOH_sunfacing.pt")

    if args.mode == "precheck":
        ok = precheck(args, root)
        sys.exit(0 if ok else 2)
    elif args.mode == "crop":
        do_crop(args, root)
        print("[OK] crop done:", root)
        sys.exit(0)
    else:
        ok = postcheck(args, root)
        sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()
