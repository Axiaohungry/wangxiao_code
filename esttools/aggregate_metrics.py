# tools/aggregate_metrics.py
import argparse, json, math
from pathlib import Path
import csv
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_tags", nargs="+", required=True)  # list of RUN_TAG
    ap.add_argument("--rel_json", default="metrics/eval_image.json")  # per-run metrics json
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    rows = []
    for tag in args.run_tags:
        p = Path("output")/"runs"/tag/args.rel_json
        if not p.exists():
            raise FileNotFoundError(str(p))
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"run_tag": tag,
                     "PSNR": float(d["PSNR"]),
                     "SSIM": float(d["SSIM"]),
                     "MAE_norm": float(d["MAE_norm"]),
                     "RMSE_norm": float(d["RMSE_norm"]),
                     "valid_ratio": float(d["valid_ratio"])})
    # aggregate
    def agg(key):
        x = np.array([r[key] for r in rows], np.float64)
        return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0

    out = {"n": len(rows), "per_run": rows, "mean_std": {}}
    for k in ["PSNR","SSIM","MAE_norm","RMSE_norm","valid_ratio"]:
        mu, sd = agg(k)
        out["mean_std"][k] = {"mean": mu, "std": sd}

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric","mean","std","n"])
        for k,v in out["mean_std"].items():
            w.writerow([k, v["mean"], v["std"], out["n"]])
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] wrote:", args.out_csv)
    print("[OK] wrote:", args.out_json)

if __name__ == "__main__":
    main()