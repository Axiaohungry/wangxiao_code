# tools/check_priors_v3.py
import os
import argparse
import torch

def stats_1d(x: torch.Tensor):
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", required=True)
    ap.add_argument("--expected_shape0", type=int, default=-1)  # -1 means skip
    ap.add_argument("--max_report", type=int, default=10)
    args = ap.parse_args()

    p = args.priors
    print(f"[Check] {p}")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    print(f"  size(MB)={os.path.getsize(p)/1024/1024:.2f}")

    try:
        pri = torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        pri = torch.load(p, map_location="cpu")

    if not isinstance(pri, torch.Tensor):
        raise TypeError(f"priors is not a Tensor: {type(pri)}")

    print(f"[Priors] shape={tuple(pri.shape)} dtype={pri.dtype}")
    if pri.ndim != 2:
        raise ValueError("priors must be [N,D].")

    N, D = pri.shape
    if args.expected_shape0 > 0 and N != args.expected_shape0:
        print(f"[Warn] N mismatch: got {N} expected {args.expected_shape0}")

    # NaN / Inf
    is_finite_row = torch.isfinite(pri).all(dim=1)
    bad = int((~is_finite_row).sum().item())
    print(f"[Finite] bad_rows={bad} ({bad/N*100:.6f}%)")
    if bad > 0:
        idx = (~is_finite_row).nonzero(as_tuple=False).squeeze(1)[:args.max_report].tolist()
        print(f"  example bad idx: {idx}")
        raise ValueError("Found NaN/Inf in priors.")

    # Common channels (assume layout consistent with your project)
    # v1/v2 (D>=5): normal(0:3), height(3), slope(4)
    n = pri[:, 0:3]
    h = pri[:, 3]
    s = pri[:, 4]

    nn = torch.linalg.norm(n, dim=1)
    nn_stats = stats_1d(nn)
    print("[Normal Norm]", nn_stats)

    tol = 0.05
    bad_ratio = float(((nn < (1 - tol)) | (nn > (1 + tol))).float().mean().item())
    print(f"[Normal Norm] |norm-1|>{tol}: {bad_ratio*100:.4f}%")

    hs = stats_1d(h)
    print("[Height]", hs)
    oob_h = float(((h < -1e-3) | (h > 1 + 1e-3)).float().mean().item())
    print(f"[Height] out_of_[0,1]: {oob_h*100:.4f}%")

    ss = stats_1d(s)
    print("[Slope]", ss)
    oob_s = float(((s < -1e-3) | (s > 1 + 1e-3)).float().mean().item())
    print(f"[Slope] out_of_[0,1]: {oob_s*100:.4f}%")

    if D == 11:
        # shadowA, shadowB
        shA = pri[:, 5]
        shB = pri[:, 6]
        print("[ShadowA]", stats_1d(shA))
        print("[ShadowB]", stats_1d(shB))
        oob_a = float(((shA < -1e-3) | (shA > 1 + 1e-3)).float().mean().item())
        oob_b = float(((shB < -1e-3) | (shB > 1 + 1e-3)).float().mean().item())
        print(f"[ShadowA] out_of_[0,1]: {oob_a*100:.4f}%")
        print(f"[ShadowB] out_of_[0,1]: {oob_b*100:.4f}%")

        # semantic one-hot [7:10]
        sem = pri[:, 7:10]
        sem_sum = sem.sum(dim=1)
        print("[SemOH sum]", stats_1d(sem_sum))
        sem_min = float(sem_sum.min().item())
        sem_max = float(sem_sum.max().item())
        if not (abs(sem_min - 1.0) < 1e-6 and abs(sem_max - 1.0) < 1e-6):
            raise ValueError(f"[FAIL] semOH sum not 1: min={sem_min} max={sem_max}")

        # sun_facing [10]
        sun = pri[:, 10]
        print("[SunFacing]", stats_1d(sun))
        oob_sun = float(((sun < -1e-6) | (sun > 1 + 1e-6)).float().mean().item())
        print(f"[SunFacing] out_of_[0,1]: {oob_sun*100:.4f}%")
        if oob_sun > 0:
            raise ValueError("[FAIL] sun_facing out of [0,1].")

        print("\n[PASS] v3 priors (D=11) basic checks passed.")
    else:
        print(f"\n[PASS] priors basic checks passed for D={D} (v1/v2-like).")

if __name__ == "__main__":
    main()
