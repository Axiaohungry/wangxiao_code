# D:\PycharmProjects\wangxiao_code\bake_priors_physics.py
import os
import sys
import json
import math
import torch
import numpy as np
from argparse import ArgumentParser

from scene.gaussian_model import GaussianModel
from utils.geo_tools import compute_normals_from_rotation


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", default="output/debug_run", type=str)
    parser.add_argument("--sample_n", type=int, default=300000, help="PCA sampling size to speed up")
    parser.add_argument("--backup_old", action="store_true", help="backup existing priors.pt -> priors_old.pt")
    args = parser.parse_args()

    ply_path = os.path.join(args.model_path, "point_cloud/iteration_7000/point_cloud.ply")
    cameras_json_path = os.path.join(args.model_path, "cameras.json")
    if not os.path.exists(ply_path):
        print(f"[Error] PLY not found: {ply_path}")
        sys.exit(1)
    if not os.path.exists(cameras_json_path):
        print(f"[Error] cameras.json not found: {cameras_json_path}")
        sys.exit(1)

    print("=== Baking Physics-Correct Priors (Fix Risk A) ===")

    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)

    xyz = gaussians.get_xyz.detach()          # [N,3] cuda
    rotation = gaussians.get_rotation.detach() # [N,4] cuda
    center = xyz.mean(dim=0)

    # --- PCA on subset (避免全量 CPU 协方差带来的不必要开销) ---
    N = xyz.shape[0]
    sample_n = min(args.sample_n, N)
    with torch.no_grad():
        idx = torch.randint(0, N, (sample_n,), device="cuda")
        sample = xyz[idx].detach().cpu()  # [sample_n,3] CPU
        sample = sample - sample.mean(dim=0, keepdim=True)
        cov = sample.T @ sample / sample.shape[0]
        eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending

    # 最小特征值对应“最薄方向”（对扁平场景通常是重力轴）
    normal = eigvecs[:, 0].cuda()
    normal = normal / (torch.norm(normal) + 1e-8)

    # --- Camera correction: 让 up 指向“天空侧” ---
    with open(cameras_json_path, "r") as f:
        cams_data = json.load(f)
    cam_centers = [np.array(c["position"]) for c in cams_data[:10]]
    mean_cam_pos = torch.tensor(np.mean(cam_centers, axis=0)).float().cuda()

    if torch.dot((mean_cam_pos - center), normal) < 0:
        normal = -normal
        print(" -> Flipped up-vector to point skyward.")

    real_up = normal / (torch.norm(normal) + 1e-8)
    print(f" -> Real Up Vector: {real_up.detach().cpu().numpy()}")

    # --- Priors ---
    point_normals = compute_normals_from_rotation(rotation)  # [N,3]
    point_normals = point_normals / (torch.norm(point_normals, dim=1, keepdim=True) + 1e-8)

    heights = ((xyz - center) * real_up).sum(dim=1)  # [N]
    h_min = heights.min()
    h_max = heights.max()
    norm_height = ((heights - h_min) / (h_max - h_min + 1e-6)).clamp(0, 1).unsqueeze(1)

    slope = torch.abs((point_normals * real_up).sum(dim=1)).clamp(0, 1).unsqueeze(1)

    priors = torch.cat([point_normals, norm_height, slope], dim=1).contiguous()  # [N,5]

    # --- Save ---
    save_path = os.path.join(args.model_path, "priors.pt")
    if args.backup_old and os.path.exists(save_path):
        backup_path = os.path.join(args.model_path, "priors_old.pt")
        os.replace(save_path, backup_path)
        print(f" -> Backup old priors: {backup_path}")

    torch.save(priors, save_path)

    meta = {
        "real_up": real_up.detach().cpu().tolist(),
        "sample_n": int(sample_n),
        "eigvals": eigvals.detach().cpu().tolist(),
        "h_min": float(h_min.detach().cpu().item()),
        "h_max": float(h_max.detach().cpu().item()),
    }
    with open(os.path.join(args.model_path, "priors_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Done! priors saved: {save_path}")
    print(f"Shape: {priors.shape} (Normal[3] + Height[1] + Slope[1])")


if __name__ == "__main__":
    main()
