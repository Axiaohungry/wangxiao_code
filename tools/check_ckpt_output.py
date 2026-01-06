import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse, os
import torch
from scene.thermal_network import ThermalAttrNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output/thermal_robust/thermal_net_robust.pth")
    ap.add_argument("--priors", default="output/debug_run/priors.pt")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    if not os.path.exists(args.priors):
        raise FileNotFoundError(args.priors)

    torch.manual_seed(args.seed)
    device = "cuda"

    pri = torch.load(args.priors, map_location="cuda", weights_only=True)
    N = pri.shape[0]
    idx = torch.randperm(N, device=device)[:min(args.n, N)]

    # 这里我们不加载 xyz，先只检查网络对 priors 的敏感性会不会输出常数
    # 若你希望严格检查 “xyz+priors”，请用你训练脚本里 gaussians.get_xyz 拼起来（更严格）。
    net = ThermalAttrNet(input_ch=8, W=16).to(device)
    state = torch.load(args.ckpt, map_location="cuda", weights_only=True)
    net.load_state_dict(state)
    net.eval()

    # 构造一个“弱检查输入”：xyz 用 0 填充，只看 priors 部分是否导致输出全常数
    x = torch.zeros((idx.numel(), 8), device=device)
    x[:, 3:] = pri[idx]  # priors: 5维 -> 填到后5维

    with torch.no_grad():
        y = net(x).squeeze()

    print(f"[Output] shape={tuple(y.shape)}")
    print(f"[Output] min={y.min().item():.6f} max={y.max().item():.6f} mean={y.mean().item():.6f} std={y.std(unbiased=False).item():.6f}")
    sat0 = (y < 1e-3).float().mean().item()
    sat1 = (y > 1-1e-3).float().mean().item()
    print(f"[Saturation] <1e-3: {sat0*100:.3f}%  >1-1e-3: {sat1*100:.3f}%")

    if y.std(unbiased=False).item() < 1e-4:
        raise RuntimeError("Output is near-constant. Suspicious (check training / input wiring).")

    print("[PASS] ckpt loads and output is non-degenerate.")

if __name__ == "__main__":
    main()
# 最小“可执行验证”
#
# 目标：不用改训练逻辑，只验证“ckpt 可加载 + 输出范围合理 + 非常规退化（全黑/全常数）不存在”。