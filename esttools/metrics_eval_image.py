# tools/metrics_eval_image.py
import argparse, json, math
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

def psnr(x, y, mask=None, L=1.0):
    d = (x - y)
    if mask is not None:
        d = d[mask]
    mse = float(np.mean(d*d)) if d.size else float("nan")
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((L*L) / mse)

def ssim_gaussian(x, y, mask=None, L=1.0):
    # Standard SSIM with Gaussian window (11, sigma=1.5)
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1*L)**2, (K2*L)**2
    # Use full image for local stats; apply mask at the end (mean over masked pixels)
    x = x.astype(np.float32); y = y.astype(np.float32)
    mu1 = cv2.GaussianBlur(x, (11,11), 1.5)
    mu2 = cv2.GaussianBlur(y, (11,11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = cv2.GaussianBlur(x*x, (11,11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(y*y, (11,11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(x*y, (11,11), 1.5) - mu1_mu2
    num = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-12)
    if mask is not None:
        v = ssim_map[mask]
    else:
        v = ssim_map.reshape(-1)
    return float(np.mean(v)) if v.size else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--mask", default="")
    ap.add_argument("--gt_eps", type=float, default=1.0/255.0)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    gt = read_gray01(args.gt)
    pr = read_gray01(args.pred)
    if gt.shape != pr.shape:
        raise ValueError(f"shape mismatch: gt={gt.shape} pred={pr.shape}")

    mask = None
    if args.mask:
        m = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(args.mask)
        if m.ndim == 3:
            m = m[...,0]
        mask = (m.astype(np.float32) > 0.5)
    else:
        mask = (gt > float(args.gt_eps))

    d = (pr - gt)
    v = d[mask]
    mae = float(np.mean(np.abs(v))) if v.size else float("nan")
    rmse = float(np.sqrt(np.mean(v*v))) if v.size else float("nan")

    out = {
        "gt": args.gt,
        "pred": args.pred,
        "mask": args.mask if args.mask else "auto(gt>eps)",
        "valid_ratio": float(mask.mean()),
        "MAE_norm": mae,
        "RMSE_norm": rmse,
        "PSNR": psnr(pr, gt, mask=mask, L=1.0),
        "SSIM": ssim_gaussian(pr, gt, mask=mask, L=1.0),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    header = "PSNR,SSIM,MAE_norm,RMSE_norm,valid_ratio\n"
    row = f"{out['PSNR']:.6f},{out['SSIM']:.6f},{out['MAE_norm']:.8f},{out['RMSE_norm']:.8f},{out['valid_ratio']:.6f}\n"
    if not Path(args.out_csv).exists():
        Path(args.out_csv).write_text(header, encoding="utf-8")
    with open(args.out_csv, "a", encoding="utf-8", newline="") as f:
        f.write(row)

    print("[OK] wrote:", args.out_json)
    print("[OK] wrote:", args.out_csv)

if __name__ == "__main__":
    main()