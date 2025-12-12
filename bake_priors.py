# D:\PycharmProjects\wangxiao_code\bake_priors.py 用于生成特征文件
import torch
import os
import numpy as np
from argparse import ArgumentParser
from scene.gaussian_model import GaussianModel
from utils.geo_tools import compute_normals_from_rotation
import sys

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", default="output/debug_run", type=str)
    args = parser.parse_args()

    ply_path = os.path.join(args.model_path, "point_cloud/iteration_7000/point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Error: {ply_path} not found.")
        sys.exit(1)

    print("Loading Geometry...")
    # 轻量加载
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)

    device = "cuda"

    # 1. 获取基础数据
    xyz = gaussians.get_xyz.detach()  # [N, 3]
    rotation = gaussians.get_rotation.detach()  # [N, 4]
    scaling = gaussians.get_scaling.detach()  # [N, 3]

    N = xyz.shape[0]
    print(f"Processing {N} points...")

    # 2. 计算法线 (Normal)
    # 高斯球旋转后的局部 Z 轴通常被视为法线（如果是扁平高斯）
    normals = compute_normals_from_rotation(rotation)  # [N, 3]

    # 确保法线指向上方 (简单的 Dot Product 检查)
    # 假设 Z 是向上，如果法线朝下 (z < 0)，则翻转
    # 你的场景里 Z 轴可能是倒的，这取决于之前的 Pose 修正
    # 我们这里先保留原始法线，后续在网络里让它自己学正负关系

    # 3. 计算相对高度 (Normalized Height)
    z_vals = xyz[:, 2]
    z_min, z_max = torch.min(z_vals), torch.max(z_vals)
    height_norm = (z_vals - z_min) / (z_max - z_min + 1e-6)  # 0~1
    height_norm = height_norm.unsqueeze(1)  # [N, 1]

    # 4. 计算倾斜度 (Slope)
    # 法线与垂直向量(0,0,1)的夹角余弦
    # abs(nz) 越大，说明越平；接近 0 说明是墙面
    vertical_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
    # 既然你之前的视频修正里发现 Z 轴是反的，这里要注意
    # 但 Slope 只看绝对值，所以没关系
    slope = torch.abs(normals[:, 2]).unsqueeze(1)  # [N, 1]

    # 5. 组合特征
    # [Nx3 Normal, Nx1 Height, Nx1 Slope] -> Nx5
    priors = torch.cat([normals, height_norm, slope], dim=1)

    # 6. 保存
    save_path = os.path.join(args.model_path, "priors.pt")
    torch.save(priors, save_path)

    print("---------------------------------------")
    print(f"Priors Baked Successfully!")
    print(f"Shape: {priors.shape}")
    print(f"Saved to: {save_path}")
    print("Features: [Normal_X, Normal_Y, Normal_Z, Norm_Height, Slope]")
    print("---------------------------------------")