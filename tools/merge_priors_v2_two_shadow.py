# tools/merge_priors_v2_two_shadow.py
import argparse
from pathlib import Path
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors_a", required=True, help="v2 priors A (e.g., shadow=visibility)")
    ap.add_argument("--priors_b", required=True, help="v2 priors B (e.g., shadow=hillshade or effective)")
    ap.add_argument("--out", required=True, help="output merged priors (v2 with 2 shadow channels)")
    ap.add_argument("--shadow_col", type=int, default=5, help="shadow column index in v2 (default 5)")
    ap.add_argument("--semantic_col", type=int, default=-1, help="semantic index col (default last)")
    ap.add_argument("--check_tol", type=float, default=1e-5, help="tolerance for checking non-shadow cols match")
    args = ap.parse_args()

    pa = Path(args.priors_a)
    pb = Path(args.priors_b)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    A = torch.load(str(pa), map_location="cpu")
    B = torch.load(str(pb), map_location="cpu")

    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)):
        raise TypeError("Both priors must be torch.Tensor")

    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A={tuple(A.shape)} B={tuple(B.shape)}")

    if A.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {A.ndim}D")

    N, D = A.shape
    sh = int(args.shadow_col)
    sem = int(args.semantic_col if args.semantic_col >= 0 else (D - 1))

    if not (0 <= sh < D):
        raise ValueError(f"shadow_col out of range: {sh} for D={D}")
    if not (0 <= sem < D):
        raise ValueError(f"semantic_col out of range: {sem} for D={D}")
    if sh == sem:
        raise ValueError("shadow_col and semantic_col cannot be the same")

    # check non-shadow cols are (almost) identical
    cols_check = [i for i in range(D) if i != sh]
    diff = (A[:, cols_check].float() - B[:, cols_check].float()).abs().max().item()
    if diff > float(args.check_tol):
        raise RuntimeError(
            f"Non-shadow columns differ too much (max_abs_diff={diff:.6g} > tol={args.check_tol}). "
            f"That indicates A/B are not aligned outputs from the same base priors/ply."
        )

    # Build merged:
    # Keep all columns except original shadow, but insert two shadow channels before semantic (and keep semantic as last)
    # Layout target:
    # [0..D-1 except shadow] but with: shadow_vis, shadow_b inserted, semantic kept as last
    # For your current v2 layout (D=7): [N(0-2),H(3),S(4),Shadow(5),Sem(6)]
    # merged becomes 8D: [N(0-2),H(3),S(4),ShadowVis,ShadowB,Sem]
    base_cols = [i for i in range(D) if i not in (sh, sem)]
    merged = torch.cat([
        A[:, base_cols].float(),
        A[:, sh:sh+1].float(),  # shadow from A (e.g., visibility)
        B[:, sh:sh+1].float(),  # shadow from B (e.g., hillshade or effective)
        A[:, sem:sem+1].float() # semantic index
    ], dim=1)

    torch.save(merged, str(outp))
    print("[OK] Wrote:", str(outp))
    print("[MERGE] A:", str(pa))
    print("[MERGE] B:", str(pb))
    print("[SHADOW] A_col=", sh, " B_col=", sh, " semantic_col=", sem)
    print("[OUT] shape=", tuple(merged.shape), "finite=", bool(torch.isfinite(merged).all().item()))
    print("[CHECK] non-shadow max_abs_diff=", diff)

if __name__ == "__main__":
    main()
