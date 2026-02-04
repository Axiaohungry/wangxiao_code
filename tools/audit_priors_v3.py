import argparse, json
from pathlib import Path
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    P = torch.load(args.priors, map_location="cpu")
    if not torch.is_tensor(P) or P.ndim != 2:
        raise ValueError(f"Expected Tensor [N,C], got {type(P)}")

    P = P.float()
    N, C = P.shape
    finite = torch.isfinite(P)
    finite_ratio = float(finite.all(dim=1).float().mean().item())

    def stat(col):
        x = P[:, col]
        return {
            "min": float(x.min().item()),
            "max": float(x.max().item()),
            "mean": float(x.mean().item()),
            "std": float(x.std().item()),
        }

    out = {
        "shape": [int(N), int(C)],
        "finite_row_ratio": finite_ratio,
        "cols": {}
    }

    # v3 expected columns
    if C >= 11:
        out["cols"]["height"]  = stat(3)
        out["cols"]["slope"]   = stat(4)
        out["cols"]["shadowA"] = stat(5)
        out["cols"]["shadowB"] = stat(6)
        out["cols"]["sun"]     = stat(10)

        sem = P[:, 7:10]
        sem_sum = sem.sum(dim=1)
        out["semantic"] = {
            "sum_min": float(sem_sum.min().item()),
            "sum_mean": float(sem_sum.mean().item()),
            "sum_max": float(sem_sum.max().item()),
            "sum_is_1_ratio(tol=1e-3)": float(((sem_sum - 1.0).abs() < 1e-3).float().mean().item()),
            "sem_min": float(sem.min().item()),
            "sem_max": float(sem.max().item()),
        }
        # count how many rows look like one-hot
        is_onehot = ((sem == 0.0) | (sem == 1.0)).all(dim=1) & ((sem_sum - 1.0).abs() < 1e-3)
        out["semantic"]["onehot_ratio"] = float(is_onehot.float().mean().item())

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] wrote:", args.out_json)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
