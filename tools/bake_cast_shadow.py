# tools/bake_cast_shadow.py
# 2.5D cast-shadow/visibility from DSM in topdown image space (CPU, fast).
# Outputs:
# - cast_visibility.npy/.png  (1=lit, 0=occluded; or soft in [0,1])
# - shadow_effective.npy/.png = hillshade * cast_visibility  (optional if hillshade provided)
# - cast_shadow_stats.json

import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def ensure_parent(p: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def percentile_stretch(x: np.ndarray, lo=1.0, hi=99.0) -> np.ndarray:
    a = np.percentile(x, lo)
    b = np.percentile(x, hi)
    if b <= a + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - a) / (b - a)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def box_blur_2d(img: np.ndarray, rx: int, ry: int | None = None) -> np.ndarray:
    """
    Separable box blur with reflect padding.
    rx: blur radius along x (axis=1)
    ry: blur radius along y (axis=0); if None, uses rx
    r<=0 disables blur on that axis.
    Complexity: O(HW).
    """
    if ry is None:
        ry = rx

    rx = int(rx)
    ry = int(ry)

    if rx <= 0 and ry <= 0:
        return img.astype(np.float32, copy=False)

    x = img.astype(np.float32, copy=False)

    def blur1d(arr: np.ndarray, radius: int, axis: int) -> np.ndarray:
        if radius <= 0:
            return arr.astype(np.float32, copy=False)

        k = 2 * radius + 1
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        ap = np.pad(arr, pad, mode="reflect")

        # cumulative sum with leading zero for easy window sums
        cs = np.cumsum(ap, axis=axis, dtype=np.float64)
        zshape = list(cs.shape)
        zshape[axis] = 1
        zeros = np.zeros(zshape, dtype=np.float64)
        cs0 = np.concatenate([zeros, cs], axis=axis)  # len = L+1

        n = arr.shape[axis]
        # window sum: cs0[i+k] - cs0[i], for i in [0, n-1]
        sl_end = [slice(None), slice(None)]
        sl_sta = [slice(None), slice(None)]
        sl_end[axis] = slice(k, k + n)
        sl_sta[axis] = slice(0, n)
        out = (cs0[tuple(sl_end)] - cs0[tuple(sl_sta)]) / float(k)
        return out.astype(np.float32)

    # axis=1 (x) uses rx; axis=0 (y) uses ry
    y = blur1d(x, rx, axis=1)
    y = blur1d(y, ry, axis=0)
    return y


def normalize_dsm(dsm: np.ndarray, mode: str) -> np.ndarray:
    d = dsm.astype(np.float32)
    m = np.isfinite(d)
    if not m.any():
        return np.zeros_like(d, dtype=np.float32)
    med = np.median(d[m])
    d[~m] = med

    if mode == "none":
        return d
    if mode == "minmax":
        mn = float(d.min())
        mx = float(d.max())
        if mx <= mn + 1e-12:
            return np.zeros_like(d, dtype=np.float32)
        return ((d - mn) / (mx - mn)).astype(np.float32)
    if mode == "p01p99":
        return percentile_stretch(d, 1.0, 99.0)
    raise ValueError(f"Unknown normalize mode: {mode}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def cast_visibility_axis(
    dsm: np.ndarray,
    tan_elev: float,
    direction: str,
    xy_step: float,
    eps: float,
    height_eps: float,
    soft: bool,
    soft_k: float
) -> np.ndarray:
    """
    direction:
      - right_to_left: sun comes from +x (image right), rays travel left
      - left_to_right: sun comes from -x (image left), rays travel right
      - top_to_bottom: sun comes from -y (image top), rays travel down
      - bottom_to_top: sun comes from +y (image bottom), rays travel up
    visibility: 1=lit, 0=occluded (or soft in [0,1])
    """
    H, W = dsm.shape
    dx = float(xy_step)

    # tolerate tiny height differences to reduce roof striping
    tol = float(max(eps, height_eps))

    if direction == "right_to_left":
        x = (np.arange(W, dtype=np.float32) * dx)[None, :]
        g = dsm - tan_elev * x  # g(u)=h(u)-tan(a)*u
        g_rev = g[:, ::-1]
        max_inc = np.maximum.accumulate(g_rev, axis=1)
        max_prev = np.concatenate([np.full((H, 1), -np.inf, np.float32), max_inc[:, :-1]], axis=1)
        margin = g_rev - max_prev  # >=0 => lit

        if soft:
            vis_rev = sigmoid((margin + tol) * float(soft_k))
        else:
            vis_rev = (margin >= -tol).astype(np.float32)
        return vis_rev[:, ::-1]

    if direction == "left_to_right":
        x = (np.arange(W, dtype=np.float32) * dx)[None, :]
        g = dsm + tan_elev * x  # g(u)=h(u)+tan(a)*u
        max_inc = np.maximum.accumulate(g, axis=1)
        max_prev = np.concatenate([np.full((H, 1), -np.inf, np.float32), max_inc[:, :-1]], axis=1)
        margin = g - max_prev

        if soft:
            vis = sigmoid((margin + tol) * float(soft_k))
        else:
            vis = (margin >= -tol).astype(np.float32)
        return vis

    if direction == "top_to_bottom":
        y = (np.arange(H, dtype=np.float32) * dx)[:, None]
        g = dsm + tan_elev * y
        max_inc = np.maximum.accumulate(g, axis=0)
        max_prev = np.concatenate([np.full((1, W), -np.inf, np.float32), max_inc[:-1, :]], axis=0)
        margin = g - max_prev

        if soft:
            vis = sigmoid((margin + tol) * float(soft_k))
        else:
            vis = (margin >= -tol).astype(np.float32)
        return vis

    if direction == "bottom_to_top":
        y = (np.arange(H, dtype=np.float32) * dx)[:, None]
        g = dsm - tan_elev * y
        g_rev = g[::-1, :]
        max_inc = np.maximum.accumulate(g_rev, axis=0)
        max_prev = np.concatenate([np.full((1, W), -np.inf, np.float32), max_inc[:-1, :]], axis=0)
        margin = g_rev - max_prev

        if soft:
            vis_rev = sigmoid((margin + tol) * float(soft_k))
        else:
            vis_rev = (margin >= -tol).astype(np.float32)
        return vis_rev[::-1, :]

    raise ValueError(f"Unknown direction: {direction}")


def save_png_gray01(path: str, x01: np.ndarray, stretch: str) -> None:
    x = x01.astype(np.float32)
    if stretch == "raw":
        y = np.clip(x, 0.0, 1.0)
    elif stretch == "p01p99":
        y = percentile_stretch(x, 1.0, 99.0)
    else:
        raise ValueError(f"Unknown png_stretch: {stretch}")
    u8 = (y * 255.0 + 0.5).astype(np.uint8)
    ensure_parent(path)
    # keep current behavior; Pillow may warn about mode deprecation in future
    Image.fromarray(u8, mode="L").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsm_npy", required=True)
    ap.add_argument("--hillshade_npy", default=None,
                    help="optional; if set, will output shadow_effective = hillshade * visibility")
    ap.add_argument("--out_visibility_npy", required=True)
    ap.add_argument("--out_visibility_png", required=True)
    ap.add_argument("--out_stats_json", required=True)

    ap.add_argument("--out_effective_shadow_npy", default=None)
    ap.add_argument("--out_effective_shadow_png", default=None)

    ap.add_argument("--direction", default="right_to_left",
                    choices=["right_to_left", "left_to_right", "top_to_bottom", "bottom_to_top"])
    ap.add_argument("--sun_elev_deg", type=float, default=55.0,
                    help="sun elevation angle in degrees (tunable)")
    ap.add_argument("--xy_step", type=float, default=1.0,
                    help="pixel step size; keep 1 unless you have metric scale")
    ap.add_argument("--normalize", default="p01p99", choices=["none", "minmax", "p01p99"])
    ap.add_argument("--height_scale", type=float, default=1.0,
                    help="multiplier after normalization to tune shadow length")
    ap.add_argument("--height_eps", type=float, default=0.002,
                    help="ignore tiny height differences to reduce roof striping")

    ap.add_argument("--blur_radius", type=int, default=2,
                    help="box blur radius on DSM before normalize (0=off)")
    ap.add_argument("--blur_radius_x", type=int, default=-1,
                    help="override blur radius along x/scan direction (-1=use blur_radius)")
    ap.add_argument("--blur_radius_y", type=int, default=-1,
                    help="override blur radius along y direction (-1=use blur_radius)")

    ap.add_argument("--eps", type=float, default=1e-6)

    ap.add_argument("--soft", action="store_true",
                    help="output soft visibility in [0,1] via sigmoid")
    ap.add_argument("--soft_k", type=float, default=20.0,
                    help="sigmoid sharpness for soft visibility")
    ap.add_argument("--png_stretch", default="raw", choices=["raw", "p01p99"])
    args = ap.parse_args()

    dsm = np.load(args.dsm_npy).astype(np.float32)
    if dsm.ndim != 2:
        raise ValueError(f"dsm_npy must be HxW, got {dsm.shape}")

    # reduce roof striping: smooth DSM before normalize (supports anisotropic blur)
    rx = int(args.blur_radius_x) if int(args.blur_radius_x) >= 0 else int(args.blur_radius)
    ry = int(args.blur_radius_y) if int(args.blur_radius_y) >= 0 else int(args.blur_radius)
    dsm = box_blur_2d(dsm, rx, ry)

    dsm_n = normalize_dsm(dsm, args.normalize)
    dsm_n = (dsm_n * float(args.height_scale)).astype(np.float32)

    elev = float(args.sun_elev_deg)
    elev = np.clip(elev, 1.0, 89.0)
    tan_elev = float(np.tan(np.deg2rad(elev)))

    vis = cast_visibility_axis(
        dsm=dsm_n,
        tan_elev=tan_elev,
        direction=args.direction,
        xy_step=float(args.xy_step),
        eps=float(args.eps),
        height_eps=float(args.height_eps),
        soft=bool(args.soft),
        soft_k=float(args.soft_k),
    ).astype(np.float32)

    ensure_parent(args.out_visibility_npy)
    np.save(args.out_visibility_npy, vis)
    save_png_gray01(args.out_visibility_png, vis, args.png_stretch)

    stats = {
        "dsm_path": str(args.dsm_npy),
        "hillshade_path": str(args.hillshade_npy) if args.hillshade_npy else None,
        "H": int(vis.shape[0]),
        "W": int(vis.shape[1]),
        "direction": args.direction,
        "sun_elev_deg": float(elev),
        "tan_elev": float(tan_elev),
        "normalize": args.normalize,
        "height_scale": float(args.height_scale),
        "height_eps": float(args.height_eps),
        "blur_radius": int(args.blur_radius),
        "blur_radius_x": int(rx),
        "blur_radius_y": int(ry),
        "eps": float(args.eps),
        "soft": bool(args.soft),
        "soft_k": float(args.soft_k),
        "visibility_min": float(vis.min()),
        "visibility_max": float(vis.max()),
        "visibility_mean": float(vis.mean()),
        "visibility_std": float(vis.std()),
        "visibility_lit_ratio": float((vis > 0.5).mean()),
        "xy-step":float(args.xy_step),
    }

    # Optional: effective shadow = hillshade * visibility
    if args.hillshade_npy and args.out_effective_shadow_npy and args.out_effective_shadow_png:
        hill = np.load(args.hillshade_npy).astype(np.float32)
        if hill.shape != vis.shape:
            raise ValueError(f"hillshade shape {hill.shape} != visibility shape {vis.shape}")
        eff = np.clip(hill * vis, 0.0, 1.0).astype(np.float32)
        ensure_parent(args.out_effective_shadow_npy)
        np.save(args.out_effective_shadow_npy, eff)
        save_png_gray01(args.out_effective_shadow_png, eff, args.png_stretch)
        stats.update({
            "effective_shadow_min": float(eff.min()),
            "effective_shadow_max": float(eff.max()),
            "effective_shadow_mean": float(eff.mean()),
            "effective_shadow_std": float(eff.std()),
        })

    ensure_parent(args.out_stats_json)
    Path(args.out_stats_json).write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("[OK] Wrote:")
    print(" ", args.out_visibility_npy)
    print(" ", args.out_visibility_png)
    if args.hillshade_npy and args.out_effective_shadow_npy and args.out_effective_shadow_png:
        print(" ", args.out_effective_shadow_npy)
        print(" ", args.out_effective_shadow_png)
    print(" ", args.out_stats_json)


if __name__ == "__main__":
    main()
