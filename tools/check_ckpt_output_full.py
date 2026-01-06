import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import torch
from scene.gaussian_model import GaussianModel
from scene.thermal_network import ThermalAttrNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="output/debug_run")
    ap.add_argument("--priors", default="output/debug_run/priors.pt")
    ap.add_argument("--ckpt", default="output/thermal_robust/thermal_net_robust.pth")
    ap.add_argument("--n", type=int, default=300000)  # 抽样点数，避免一次性全量统计太慢
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ply_path = os.path.join(args.model_path, "point_cloud/iteration_7000/point_cloud.ply")
    if not os.path.exists(ply_path): raise FileNotFoundError(ply_path)
    if not os.path.exists(args.priors): raise FileNotFoundError(args.priors)
    if not os.path.exists(args.ckpt): raise FileNotFoundError(args.ckpt)

    torch.manual_seed(args.seed)
    device = "cuda"

    # 1) load xyz
    gaussians = GaussianModel(sh_degree=0, brdf_dim=-1, brdf_mode="pbbr", brdf_envmap_res=0, feature_time=False)
    gaussians.load_ply(ply_path)
    xyz = gaussians.get_xyz.detach()  # [N,3]

    # 2) load priors
    pri = torch.load(args.priors, map_location="cuda", weights_only=True)  # [N,5]
    assert pri.shape[0] == xyz.shape[0] and pri.shape[1] == 5

    N = xyz.shape[0]
    n = min(args.n, N)
    idx = torch.randperm(N, device=device)[:n]

    # 3) build input exactly like training (xyz_norm + priors)
    center = xyz.mean(dim=0)
    min_xyz, _ = torch.min(xyz, dim=0)
    max_xyz, _ = torch.max(xyz, dim=0)
    span_x = (max_xyz[0] - min_xyz[0]).item()
    span_y = (max_xyz[1] - min_xyz[1]).item()
    max_span = max(span_x, span_y)

    xyz_norm = (xyz - center) / (max_span + 1e-6)
    x = torch.cat([xyz_norm[idx], pri[idx]], dim=1).contiguous()  # [n,8]

    # 4) load net
    net = ThermalAttrNet(input_ch=8, W=16).to(device)
    state = torch.load(args.ckpt, map_location="cuda", weights_only=True)
    net.load_state_dict(state)
    net.eval()

    with torch.no_grad():
        y = net(x).squeeze()  # [n]

    print(f"[Input] n={n} x_shape={tuple(x.shape)}")
    print(f"[Output] min={y.min().item():.6f} max={y.max().item():.6f} mean={y.mean().item():.6f} std={y.std(unbiased=False).item():.6f}")
    sat0 = (y < 1e-3).float().mean().item()
    sat1 = (y > 1-1e-3).float().mean().item()
    print(f"[Saturation] <1e-3: {sat0*100:.3f}%  >1-1e-3: {sat1*100:.3f}%")

    if not torch.isfinite(y).all():
        raise RuntimeError("Found NaN/Inf in network output.")
    if y.std(unbiased=False).item() < 1e-4:
        raise RuntimeError("Output is near-constant (suspicious).")

    print("[PASS] ckpt output looks non-degenerate and numerically sane.")

if __name__ == "__main__":
    main()
# 用真实训练输入结构做 ckpt 输出核对