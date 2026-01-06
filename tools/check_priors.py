import os, json, math
import argparse
import torch
import numpy as np

def stats_1d(x: torch.Tensor):
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", default="output/debug_run/priors.pt")
    ap.add_argument("--priors_old", default="output/debug_run/priors_old.pt")
    ap.add_argument("--expected_shape0", type=int, default=4418110)
    ap.add_argument("--expected_up", type=str, default="")  # e.g. "0.04849,-0.74267,-0.66789"
    ap.add_argument("--max_report", type=int, default=10)
    args = ap.parse_args()

    def check_exists(p):
        print(f"[Check] {p}")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        print(f"  size(MB)={os.path.getsize(p)/1024/1024:.2f}")

    check_exists(args.priors)
    if os.path.exists(args.priors_old):
        check_exists(args.priors_old)
    else:
        print("[Warn] priors_old.pt not found (maybe you skipped backup).")

    pri = torch.load(args.priors, map_location="cpu", weights_only=True)
    if not isinstance(pri, torch.Tensor):
        raise TypeError(f"priors is not a Tensor: {type(pri)}")

    print(f"[Priors] shape={tuple(pri.shape)} dtype={pri.dtype}")
    if pri.ndim != 2 or pri.shape[1] != 5:
        raise ValueError("priors must be [N,5].")
    if pri.shape[0] != args.expected_shape0:
        print(f"[Warn] N mismatch: got {pri.shape[0]} expected {args.expected_shape0}")

    # NaN / Inf
    is_finite = torch.isfinite(pri).all(dim=1)
    bad = (~is_finite).sum().item()
    print(f"[Finite] bad_rows={bad} ({bad/pri.shape[0]*100:.6f}%)")
    if bad > 0:
        idx = (~is_finite).nonzero(as_tuple=False).squeeze(1)[:args.max_report].tolist()
        print(f"  example bad idx: {idx}")
        raise ValueError("Found NaN/Inf in priors.")

    n = pri[:, 0:3]
    h = pri[:, 3]
    s = pri[:, 4]

    # Normal norm ~ 1
    nn = torch.linalg.norm(n, dim=1)
    nn_stats = stats_1d(nn)
    print("[Normal Norm]", nn_stats)

    tol = 0.05
    good = (nn > (1 - tol)) & (nn < (1 + tol))
    bad_ratio = 1.0 - good.float().mean().item()
    print(f"[Normal Norm] |norm-1|>{tol}: {bad_ratio*100:.4f}%")
    if bad_ratio > 0.1:
        print("[Warn] Too many non-unit normals. Check rotation->normal conversion.")

    # Height range
    hs = stats_1d(h)
    print("[Height]", hs)
    oob_h = ((h < -1e-3) | (h > 1 + 1e-3)).float().mean().item()
    print(f"[Height] out_of_[0,1]: {oob_h*100:.4f}%")

    # Slope range
    ss = stats_1d(s)
    print("[Slope]", ss)
    oob_s = ((s < -1e-3) | (s > 1 + 1e-3)).float().mean().item()
    print(f"[Slope] out_of_[0,1]: {oob_s*100:.4f}%")

    # Optional: expected_up check (just vector sanity)
    if args.expected_up:
        up = torch.tensor([float(v) for v in args.expected_up.split(",")], dtype=torch.float32)
        upn = float(torch.linalg.norm(up).item())
        print(f"[Expected Up] {up.tolist()} norm={upn:.6f}")
        if abs(upn - 1.0) > 1e-2:
            print("[Warn] expected_up not unit. (Not fatal, but indicates you copied wrong numbers.)")

    print("\n[PASS] priors basic checks passed.")

if __name__ == "__main__":
    main()
# 抽样统计：min/max/mean/std、NaN/Inf、normal 单位长度、height/slope 合法范围