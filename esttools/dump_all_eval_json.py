# esttools/dump_all_eval_json.py
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

RE_T = re.compile(r"(?:_T(\d+))|(?:T(\d+))", re.IGNORECASE)

def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def infer_threshold(j: Dict[str, Any], json_path: Path) -> str:
    # Priority: filename contains _T248/_T252; else mask path contains T250; else T250
    m = RE_T.search(json_path.name)
    if m:
        t = m.group(1) or m.group(2)
        if t:
            return f"T{int(t)}"
    mask = str(j.get("mask", ""))
    m2 = RE_T.search(mask)
    if m2:
        t = m2.group(1) or m2.group(2)
        if t:
            return f"T{int(t)}"
    return "T250"

def method_from_eval_dir(eval_dir: Path) -> str:
    # e.g. output/runs/2026-02-25_eval_full_zscore -> full_zscore
    name = eval_dir.name
    if "_eval_" in name:
        return name.split("_eval_", 1)[1]
    if name.startswith("2026-02-25_eval_"):
        return name.replace("2026-02-25_eval_", "")
    return name

def pct_improve(lower_is_better: bool, base: float, cur: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    if lower_is_better:
        return 100.0 * (base - cur) / base
    return 100.0 * (cur - base) / base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_sum", default="2026-02-25_eval_suite_roi", help="output run tag for summary")
    ap.add_argument("--eval_base", default=r"output\runs", help="base folder containing eval runs")
    ap.add_argument("--eval_run_prefix", default="2026-02-25_eval_", help="prefix of eval run dirs")
    ap.add_argument("--methods", nargs="+", default=["full", "full_zscore", "sem_only", "shadow_only", "shuffled"])
    ap.add_argument("--out_dir", default="", help="override output dir; default output/runs/<run_sum>/metrics")
    ap.add_argument("--print_stdout", action="store_true", help="print merged json to stdout")
    args = ap.parse_args()

    project_root = Path(".").resolve()
    eval_base = project_root / args.eval_base
    if not eval_base.exists():
        raise SystemExit(f"[FATAL] eval_base not found: {eval_base}")

    out_dir = Path(args.out_dir) if args.out_dir else (project_root / "output" / "runs" / args.run_sum / "metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    # merged structure: thresholds -> method -> json + src path
    merged: Dict[str, Dict[str, Any]] = {}
    flat_rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for m in args.methods:
        eval_dir = eval_base / f"{args.eval_run_prefix}{m}"
        metrics_dir = eval_dir / "metrics"
        if not metrics_dir.exists():
            missing.append(str(metrics_dir))
            continue

        json_files = sorted(metrics_dir.glob("eval_image_roi*.json"))
        if not json_files:
            missing.append(str(metrics_dir / "eval_image_roi*.json"))
            continue

        for jf in json_files:
            j = read_json(jf)
            th = infer_threshold(j, jf)
            method = method_from_eval_dir(eval_dir)

            merged.setdefault(th, {})
            merged[th][method] = {
                "data": j,
                "src_json": str(jf).replace("/", "\\"),
            }

            flat_rows.append({
                "threshold": th,
                "method": method,
                "valid_ratio": float(j.get("valid_ratio", 0.0)),
                "MAE_norm": float(j["MAE_norm"]),
                "RMSE_norm": float(j["RMSE_norm"]),
                "PSNR_dB": float(j["PSNR"]),
                "SSIM": float(j["SSIM"]),
                "gt": str(j.get("gt", "")),
                "pred": str(j.get("pred", "")),
                "mask": str(j.get("mask", "")),
                "src_json": str(jf).replace("/", "\\"),
            })

    # write all_eval.json
    out_json = out_dir / "all_eval.json"
    out_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] wrote:", out_json)

    # write all_eval.csv
    out_csv = out_dir / "all_eval.csv"
    fieldnames = ["threshold", "method", "valid_ratio", "MAE_norm", "RMSE_norm", "PSNR_dB", "SSIM", "gt", "pred", "mask", "src_json"]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(flat_rows, key=lambda x: (x["threshold"], x["method"])):
            w.writerow(r)
    print("[OK] wrote:", out_csv)

    # write delta_vs_full.csv (per threshold)
    out_delta = out_dir / "delta_vs_full.csv"
    d_fields = [
        "threshold", "method",
        "dMAE_abs", "MAE_improve_pct",
        "dRMSE_abs", "RMSE_improve_pct",
        "dPSNR_abs", "PSNR_improve_pct",
        "dSSIM_abs", "SSIM_improve_pct",
    ]
    with open(out_delta, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=d_fields)
        w.writeheader()

        # index by threshold+method
        idx: Dict[Tuple[str, str], Dict[str, Any]] = {(r["threshold"], r["method"]): r for r in flat_rows}

        for th in sorted(set(r["threshold"] for r in flat_rows)):
            base_key = (th, "full")
            if base_key not in idx:
                continue
            base = idx[base_key]
            for m in sorted(set(r["method"] for r in flat_rows if r["threshold"] == th)):
                cur = idx[(th, m)]
                row = {
                    "threshold": th,
                    "method": m,
                    "dMAE_abs": cur["MAE_norm"] - base["MAE_norm"],
                    "MAE_improve_pct": pct_improve(True, base["MAE_norm"], cur["MAE_norm"]),
                    "dRMSE_abs": cur["RMSE_norm"] - base["RMSE_norm"],
                    "RMSE_improve_pct": pct_improve(True, base["RMSE_norm"], cur["RMSE_norm"]),
                    "dPSNR_abs": cur["PSNR_dB"] - base["PSNR_dB"],
                    "PSNR_improve_pct": pct_improve(False, base["PSNR_dB"], cur["PSNR_dB"]),
                    "dSSIM_abs": cur["SSIM"] - base["SSIM"],
                    "SSIM_improve_pct": pct_improve(False, base["SSIM"], cur["SSIM"]),
                }
                w.writerow(row)
    print("[OK] wrote:", out_delta)

    if missing:
        warn = out_dir / "missing_files.txt"
        warn.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print("[WARN] some files/dirs missing, wrote:", warn)

    if args.print_stdout:
        print(json.dumps(merged, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()