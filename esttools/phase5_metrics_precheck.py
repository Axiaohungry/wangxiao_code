# tools/phase5_metrics_precheck.py
import argparse, hashlib, json, os
from pathlib import Path
import cv2
import numpy as np

def sha256_file(p: Path, chunk=1024*1024):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def img_info(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[...,0]
    a = img.astype(np.float32)
    return {
        "path": str(p),
        "shape": [int(a.shape[0]), int(a.shape[1])],
        "dtype": str(img.dtype),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_tag", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--mask_auto", default="")  # optional valid_mask.png
    ap.add_argument("--points_csv", default="") # optional ground points
    args = ap.parse_args()

    run_dir = Path("output")/"runs"/args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/"logs").mkdir(exist_ok=True)
    (run_dir/"artifacts").mkdir(exist_ok=True)
    (run_dir/"metrics").mkdir(exist_ok=True)

    report = []
    manifest = []

    def add_file(p):
        p = Path(p)
        ok = p.exists()
        report.append(f"[FILE] {p} exists={ok} size={p.stat().st_size if ok else -1}")
        if ok and p.is_file():
            manifest.append(f"{p.as_posix()}\t{p.stat().st_size}\t{sha256_file(p)}")
        return ok

    ok_gt = add_file(args.gt)
    ok_pred = add_file(args.pred)
    ok_mask = True
    if args.mask_auto:
        ok_mask = add_file(args.mask_auto)
    ok_pts = True
    if args.points_csv:
        ok_pts = add_file(args.points_csv)

    if ok_gt:
        gi = img_info(Path(args.gt)); report.append(f"[GT_IMG] {json.dumps(gi, ensure_ascii=False)}")
    if ok_pred:
        pi = img_info(Path(args.pred)); report.append(f"[PRED_IMG] {json.dumps(pi, ensure_ascii=False)}")
    if args.mask_auto and ok_mask:
        mi = img_info(Path(args.mask_auto)); report.append(f"[MASK_IMG] {json.dumps(mi, ensure_ascii=False)}")

    # resolution check
    if ok_gt and ok_pred:
        g = img_info(Path(args.gt)); p = img_info(Path(args.pred))
        if g and p and (g["shape"] != p["shape"]):
            report.append("[FATAL] GT and PRED resolution mismatch.")
        else:
            report.append("[OK] GT and PRED resolution match.")

    (run_dir/"check_report.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    (run_dir/"manifest_snapshot.txt").write_text("\n".join(manifest)+"\n", encoding="utf-8")
    (run_dir/"run_meta.json").write_text(json.dumps({
        "run_tag": args.run_tag,
        "gt": args.gt,
        "pred": args.pred,
        "mask_auto": args.mask_auto,
        "points_csv": args.points_csv
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] wrote:", run_dir/"check_report.txt")
    print("[OK] wrote:", run_dir/"manifest_snapshot.txt")

if __name__ == "__main__":
    main()