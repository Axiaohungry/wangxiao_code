# tools/shadow_stats.py
import argparse
import numpy as np
import torch as T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", required=True)
    ap.add_argument("--colA", type=int, default=5)
    ap.add_argument("--colB", type=int, default=6)
    args = ap.parse_args()

    p = T.load(args.priors, map_location="cpu").float()
    A = p[:, args.colA].numpy()
    B = p[:, args.colB].numpy()

    print("shape=", tuple(p.shape))
    print("A min/max/mean/std=", float(A.min()), float(A.max()), float(A.mean()), float(A.std()))
    print("B min/max/mean/std=", float(B.min()), float(B.max()), float(B.mean()), float(B.std()))
    print("corr(A,B)=", float(np.corrcoef(A, B)[0, 1]))
    print("A==0.5 ratio=", float(np.mean(np.isclose(A, 0.5))))
    print("B==0.5 ratio=", float(np.mean(np.isclose(B, 0.5))))

if __name__ == "__main__":
    main()
