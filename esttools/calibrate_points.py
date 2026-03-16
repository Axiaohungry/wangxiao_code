# tools/calibrate_points.py
import argparse, csv, json, math
from pathlib import Path
import cv2
import numpy as np

def read_gray01(p: str) -> np.ndarray:
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(p)
    if img.ndim == 3:
        img = img[...,0]
    x = img.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)         # pred_gray.png / train_*_gray.png
    ap.add_argument("--points_csv", required=True)   # ground_points_60.csv
    ap.add_argument("--out_fit_json", required=True)
    ap.add_argument("--out_err_csv", required=True)
    ap.add_argument("--col_x", default="x")
    ap.add_argument("--col_y", default="y")
    ap.add_argument("--col_t", default="temp_c")
    args = ap.parse_args()

    pred = read_gray01(args.pred)
    H, W = pred.shape[:2]

    xs, ys, ts = [], [], []
    with open(args.points_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            x = int(float(row[args.col_x])); y = int(float(row[args.col_y])); t = float(row[args.col_t])
            x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
            xs.append(x); ys.append(y); ts.append(t)

    xs = np.array(xs, np.int32); ys = np.array(ys, np.int32); ts = np.array(ts, np.float32)
    pv = pred[ys, xs].astype(np.float32)

    # Fit: T = a * p + b
    A = np.stack([pv, np.ones_like(pv)], axis=1)  # Nx2
    sol, _, _, _ = np.linalg.lstsq(A, ts, rcond=None)
    a, b = float(sol[0]), float(sol[1])

    t_hat = a*pv + b
    err = t_hat - ts
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err*err)))

    fit = {
        "model": "T = a*p + b",
        "a": a, "b": b,
        "n": int(len(ts)),
        "MAE_C": mae,
        "RMSE_C": rmse,
        "pred_path": args.pred,
        "points_csv": args.points_csv
    }
    Path(args.out_fit_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_err_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_fit_json).write_text(json.dumps(fit, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(args.out_err_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","x","y","pred_norm","gt_C","pred_C","err_C"])
        for i in range(len(ts)):
            w.writerow([i, int(xs[i]), int(ys[i]), float(pv[i]), float(ts[i]), float(t_hat[i]), float(err[i])])
        w.writerow([])
        w.writerow(["MAE_C", mae])
        w.writerow(["RMSE_C", rmse])

    print("[OK] wrote:", args.out_fit_json)
    print("[OK] wrote:", args.out_err_csv)

if __name__ == "__main__":
    main()