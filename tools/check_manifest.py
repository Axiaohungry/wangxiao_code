# tools/check_manifest.py
import os, sys, json, hashlib, time
from pathlib import Path

def sha256(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def fmt_size(n):
    for unit in ["B","KB","MB","GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=r"D:\PycharmProjects\wangxiao_code")
    ap.add_argument("--debug_root", default=r"output\debug_run")
    ap.add_argument("--run_dir", required=True, help=r"output\runs\<RUN_TAG>")
    ap.add_argument("--priors_key_candidates", default="priors,priors_pt,data,features", help="comma keys to try when loading priors")
    args = ap.parse_args()

    proj = Path(args.project_root)
    dbg  = proj / args.debug_root
    run  = proj / args.run_dir
    run.mkdir(parents=True, exist_ok=True)

    report_path   = run / "check_report.txt"
    manifest_path = run / "manifest_snapshot.txt"

    must = {
        "ply": dbg / r"point_cloud\iteration_7000\point_cloud.ply",
        "cameras": dbg / "cameras.json",
        "topdown": dbg / "topdown_final.png",
        "lst_gt": dbg / "lst_gt.png",
        "priors": dbg / "priors.pt",
    }

    lines = []
    lines.append(f"[Time] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"[Project] {proj}")
    lines.append(f"[DebugRoot] {dbg}")
    lines.append(f"[RunDir] {run}")
    lines.append("")

    # Existence + basic stats
    lines.append("== File Existence & Stats ==")
    manifest_lines = []
    ok_all = True
    for k, p in must.items():
        exists = p.exists()
        ok_all &= exists
        if exists:
            size = p.stat().st_size
            h = sha256(p)
            lines.append(f"[OK] {k}: {p} | {fmt_size(size)} | sha256={h[:16]}...")
            manifest_lines.append(f"{k}\t{p}\t{size}\t{h}")
        else:
            lines.append(f"[MISSING] {k}: {p}")
    lines.append("")

    # Image resolution check
    lines.append("== Image Resolution Consistency ==")
    try:
        from PIL import Image
        td = Image.open(must["topdown"]).size if must["topdown"].exists() else None
        lg = Image.open(must["lst_gt"]).size if must["lst_gt"].exists() else None
        lines.append(f"topdown_final.png: {td}")
        lines.append(f"lst_gt.png:       {lg}")
        if td and lg and td == lg:
            lines.append("[PASS] topdown and lst_gt resolutions match.")
        elif td and lg:
            lines.append("[FAIL] topdown and lst_gt resolutions DO NOT match.")
            ok_all = False
    except Exception as e:
        lines.append(f"[WARN] Image check failed: {e}")
    lines.append("")

    # Priors numeric health (sample stats)
    lines.append("== Priors Numeric Health (sample) ==")
    if must["priors"].exists():
        try:
            import torch
            try:
                obj = torch.load(must["priors"], map_location="cpu", weights_only=True)
            except TypeError:
                obj = torch.load(must["priors"], map_location="cpu")
            # try common layouts
            pri = None
            if isinstance(obj, torch.Tensor):
                pri = obj
            elif isinstance(obj, dict):
                for key in args.priors_key_candidates.split(","):
                    key = key.strip()
                    if key in obj and isinstance(obj[key], torch.Tensor):
                        pri = obj[key]; break
                if pri is None:
                    # fallback: first tensor value
                    for v in obj.values():
                        if isinstance(v, torch.Tensor):
                            pri = v; break
            if pri is None:
                lines.append("[WARN] Could not find tensor in priors.pt (dict layout unknown).")
            else:
                # sample to avoid heavy memory ops
                n = pri.shape[0]
                idx = torch.randint(0, n, (min(200000, n),))
                samp = pri[idx].float()
                nan = torch.isnan(samp).any().item()
                inf = torch.isinf(samp).any().item()
                lines.append(f"priors tensor shape: {tuple(pri.shape)} dtype={pri.dtype}")
                lines.append(f"NaN: {nan} | Inf: {inf}")
                lines.append(f"mean={samp.mean().item():.6f} std={samp.std().item():.6f} "
                             f"min={samp.min().item():.6f} max={samp.max().item():.6f}")
                if (not nan) and (not inf):
                    lines.append("[PASS] priors sample looks numerically healthy.")
                else:
                    lines.append("[FAIL] priors contains NaN/Inf in sample.")
                    ok_all = False
        except Exception as e:
            lines.append(f"[WARN] Priors check failed: {e}")
    else:
        lines.append("[SKIP] priors.pt missing.")
    lines.append("")

    # -------------------------
    # MINIMAL ADDITION: Semantic stats (optional) from this RUN_DIR
    # -------------------------
    sem_stats = run / "artifacts" / "semantic_stats.json"
    if sem_stats.exists():
        lines.append("== Semantic (optional) ==")
        try:
            s = json.loads(sem_stats.read_text(encoding="utf-8"))
            lines.append("used_method: " + str(s.get("used_method")))
            lines.append("ratio: " + json.dumps(s.get("ratio", {}), ensure_ascii=False))
        except Exception as e:
            lines.append(f"[WARN] semantic_stats parse failed: {e}")
        lines.append("")

    lines.append("== Summary ==")
    lines.append("OVERALL: " + ("PASS" if ok_all else "FAIL"))

    lines.append("Next: If FAIL, fix missing/mismatch items before Phase4.1/4.2.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {manifest_path}")
    sys.exit(0 if ok_all else 2)

if __name__ == "__main__":
    main()
