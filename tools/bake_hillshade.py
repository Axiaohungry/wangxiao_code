# tools/bake_hillshade.py
import os
import json
import time
import math
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

# Optional OpenCV (preferred for fast blur / PNG write)
try:
    import cv2
except Exception:
    cv2 = None


def _ensure_parent(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def _imwrite_gray_u8(path: str, img01: np.ndarray):
    """Save a single-channel [0,1] float image to PNG (uint8)."""
    img_u8 = (np.clip(img01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    _ensure_parent(path)

    if cv2 is not None:
        cv2.imwrite(path, img_u8)
        return

    # Fallback to PIL
    from PIL import Image
    Image.fromarray(img_u8, mode="L").save(path)


def _stretch_percentile_to_01(x: np.ndarray, p_lo=1.0, p_hi=99.0) -> np.ndarray:
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi - lo < 1e-8:
        return np.clip(x, 0.0, 1.0)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def hillshade_from_dsm(
    dsm: np.ndarray,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    cell_size: float = 1.0,
    z_scale: float = 1.0,
    blur_sigma: float = 0.8,
    invert_z: bool = False,
) -> np.ndarray:
    """
    DSM -> Hillshade (float32, [0,1]).
    Convention:
      - azimuth: degrees clockwise from North (0=N, 90=E)
      - image x: right (East), image y: down (South)
    """
    dsm = dsm.astype(np.float32, copy=False)

    finite = np.isfinite(dsm)
    if not finite.all():
        med = np.median(dsm[finite]) if finite.any() else 0.0
        dsm = dsm.copy()
        dsm[~finite] = med

    if invert_z:
        dsm = -dsm

    # Smooth to stabilize gradients
    if blur_sigma > 0 and cv2 is not None:
        k = int(max(3, (blur_sigma * 6) // 2 * 2 + 1))  # odd kernel
        dsm_blur = cv2.GaussianBlur(dsm, (k, k), sigmaX=blur_sigma, sigmaY=blur_sigma)
    else:
        dsm_blur = dsm

    dz_dy, dz_dx = np.gradient(dsm_blur)

    # Convert to "north-up" y for GIS formulas:
    # image y increases downward (south), so north-gradient is -dy
    p = (dz_dx / max(cell_size, 1e-6)) * z_scale  # east
    q = (-dz_dy / max(cell_size, 1e-6)) * z_scale  # north

    slope = np.arctan(np.sqrt(p * p + q * q) + 1e-12)

    # Aspect clockwise from North
    aspect = np.arctan2(p, q)  # [-pi, pi]
    aspect = np.where(aspect < 0, aspect + 2.0 * math.pi, aspect)

    az = math.radians(sun_azimuth_deg % 360.0)
    zen = math.radians(90.0 - float(sun_elevation_deg))  # zenith angle

    hs = (math.cos(zen) * np.cos(slope) +
          math.sin(zen) * np.sin(slope) * np.cos(az - aspect))

    hs = np.clip(hs, 0.0, 1.0).astype(np.float32)
    return hs


def main():
    ap = ArgumentParser()
    ap.add_argument("--dsm_npy", required=True, help="path to artifacts/dsm_float.npy (float32 HxW)")
    ap.add_argument("--mode", default="continuous", choices=["continuous", "binary"])
    ap.add_argument("--sun_azimuth_deg", type=float, default=135.0)
    ap.add_argument("--sun_elevation_deg", type=float, default=35.0)
    ap.add_argument("--binary_threshold", type=float, default=0.35)

    ap.add_argument("--cell_size", type=float, default=1.0, help="pixel size in XY units (relative)")
    ap.add_argument("--z_scale", type=float, default=1.0, help="vertical exaggeration for gradients")
    ap.add_argument("--blur_sigma", type=float, default=0.8, help="Gaussian blur sigma before gradient")
    ap.add_argument("--invert_z", type=int, default=0, help="1 to invert DSM sign if shading looks flipped")

    # IMPORTANT: this must be inside main(), after ap is created
    ap.add_argument("--png_stretch", default="raw", choices=["raw", "p01p99"],
                    help="Only affects PNG visualization; NPY keeps raw continuous values.")

    ap.add_argument("--out_hillshade_npy", required=True)
    ap.add_argument("--out_hillshade_png", required=True)
    ap.add_argument("--out_shadow_npy", required=True)
    ap.add_argument("--out_shadow_png", required=True)
    ap.add_argument("--out_stats_json", default="", help="optional; default: artifacts/hillshade_stats.json")

    args = ap.parse_args()

    t0 = time.time()

    dsm_path = Path(args.dsm_npy)
    if not dsm_path.exists():
        raise FileNotFoundError(str(dsm_path))

    dsm = np.load(str(dsm_path))
    if dsm.ndim != 2:
        raise ValueError(f"DSM must be HxW, got shape={dsm.shape}")

    hs = hillshade_from_dsm(
        dsm=dsm,
        sun_azimuth_deg=args.sun_azimuth_deg,
        sun_elevation_deg=args.sun_elevation_deg,
        cell_size=args.cell_size,
        z_scale=args.z_scale,
        blur_sigma=args.blur_sigma,
        invert_z=bool(args.invert_z),
    )

    if args.mode == "continuous":
        shadow = hs
    else:
        shadow = (hs > float(args.binary_threshold)).astype(np.float32)

    # Save NPY (training inputs): keep raw values
    _ensure_parent(args.out_hillshade_npy)
    np.save(args.out_hillshade_npy, hs.astype(np.float32))

    _ensure_parent(args.out_shadow_npy)
    np.save(args.out_shadow_npy, shadow.astype(np.float32))

    # PNG visualization: optionally stretch contrast
    hs_png = hs
    shadow_png = shadow
    if args.png_stretch == "p01p99":
        hs_png = _stretch_percentile_to_01(hs, 1.0, 99.0)
        shadow_png = _stretch_percentile_to_01(shadow, 1.0, 99.0)

    _imwrite_gray_u8(args.out_hillshade_png, hs_png)
    _imwrite_gray_u8(args.out_shadow_png, shadow_png)

    # Stats
    finite_ratio = float(np.isfinite(hs).mean())
    stats = {
        "time_sec": round(time.time() - t0, 4),
        "input_dsm": str(dsm_path),
        "H": int(hs.shape[0]),
        "W": int(hs.shape[1]),
        "finite_ratio": finite_ratio,
        "hillshade": {
            "min": float(hs.min()),
            "max": float(hs.max()),
            "mean": float(hs.mean()),
            "std": float(hs.std()),
            "p01": float(np.percentile(hs, 1)),
            "p50": float(np.percentile(hs, 50)),
            "p99": float(np.percentile(hs, 99)),
        },
        "shadow_map": {
            "mode": args.mode,
            "binary_threshold": float(args.binary_threshold),
            "min": float(shadow.min()),
            "max": float(shadow.max()),
            "mean": float(shadow.mean()),
            "std": float(shadow.std()),
            "pct_lt_005": float((shadow < 0.05).mean()),
            "pct_gt_095": float((shadow > 0.95).mean()),
        },
        "params": {
            "sun_azimuth_deg": float(args.sun_azimuth_deg),
            "sun_elevation_deg": float(args.sun_elevation_deg),
            "cell_size": float(args.cell_size),
            "z_scale": float(args.z_scale),
            "blur_sigma": float(args.blur_sigma),
            "invert_z": int(args.invert_z),
            "png_stretch": str(args.png_stretch),
        },
        "notes": [
            "azimuth is clockwise from North; image x=East, y=South.",
            "If shadow direction looks flipped, set --invert_z 1.",
            "Prefer mode=continuous for training stability (shadow in [0,1]).",
            "png_stretch only affects PNG visualization; NPY remains raw."
        ],
    }

    if args.out_stats_json.strip():
        stats_path = Path(args.out_stats_json)
    else:
        stats_path = Path(args.out_shadow_png).parent / "hillshade_stats.json"

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Wrote:")
    print(f"  {args.out_hillshade_npy}")
    print(f"  {args.out_hillshade_png}")
    print(f"  {args.out_shadow_npy}")
    print(f"  {args.out_shadow_png}")
    print(f"  {str(stats_path)}")
    print(f"[HILLSHADE] finite_ratio={finite_ratio:.4f} "
          f"min={hs.min():.6f} max={hs.max():.6f} mean={hs.mean():.6f} std={hs.std():.6f}")
    print(f"[SHADOW] mode={args.mode} "
          f"min={shadow.min():.6f} max={shadow.max():.6f} mean={shadow.mean():.6f} std={shadow.std():.6f}")


if __name__ == "__main__":
    main()
