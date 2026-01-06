import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="output/debug_run/priors.pt")
    ap.add_argument("--out_path", default="output/debug_run/priors_shuffled.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    pri = torch.load(args.in_path, map_location="cpu", weights_only=True)
    idx = torch.randperm(pri.shape[0])
    pri2 = pri[idx].contiguous()
    torch.save(pri2, args.out_path)
    print(f"Saved shuffled priors: {args.out_path} shape={tuple(pri2.shape)}")

if __name__ == "__main__":
    main()
# 置乱检验（Permutation Test）：证明“提升来自先验对应关系”而非分布巧合