import argparse
import json
import os
import re
import sys
import time
import hashlib
import glob
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

def sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def deep_get(d: Dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"Path not found: {dotted} (stuck at {part})")
    return cur

def deep_set(d: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur: Any = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def parse_value(v: str) -> Any:
    # Try JSON parsing for numbers/bools/null/arrays/objects; fallback to raw string.
    try:
        return json.loads(v)
    except Exception:
        return v

def expand_vars(s: str, ctx: Dict[str, Any]) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1).strip()
        try:
            val = deep_get(ctx, key)
        except Exception:
            raise KeyError(f"Unknown variable: ${{{key}}}")
        return str(val)
    return VAR_PATTERN.sub(repl, s)

def resolve(obj: Any, ctx: Dict[str, Any]) -> Any:
    if isinstance(obj, str):
        return expand_vars(obj, ctx) if "${" in obj else obj
    if isinstance(obj, list):
        return [resolve(x, ctx) for x in obj]
    if isinstance(obj, dict):
        return {k: resolve(v, ctx) for k, v in obj.items()}
    return obj

def args_dict_to_argv(args: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for k, v in args.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                argv.append(flag)
            continue
        if v is None:
            continue
        if isinstance(v, list):
            for item in v:
                argv.append(flag)
                argv.append(str(item))
            continue
        argv.append(flag)
        argv.append(str(v))
    return argv

def ensure_dirs(run_dir: Path) -> Dict[str, Path]:
    sub = {
        "run_dir": run_dir,
        "logs_dir": run_dir / "logs",
        "artifacts_dir": run_dir / "artifacts",
        "ckpt_dir": run_dir / "ckpt",
        "renders_dir": run_dir / "renders",
        "metrics_dir": run_dir / "metrics",
    }
    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    return sub

def check_expects(expects: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    files = expects.get("files", [])
    for f in files:
        if not Path(f).exists():
            errors.append(f"missing file: {f}")

    nonempty = expects.get("nonempty_files", [])
    for f in nonempty:
        p = Path(f)
        if not p.exists():
            errors.append(f"missing file: {f}")
        else:
            if p.stat().st_size <= 0:
                errors.append(f"empty file: {f}")

    globs = expects.get("files_glob", [])
    for g in globs:
        if len(glob.glob(g)) == 0:
            errors.append(f"glob matched nothing: {g}")

    contains = expects.get("contains_text", [])
    for item in contains:
        path = Path(item["path"])
        must = item["must_contain"]
        if not path.exists():
            errors.append(f"missing file for contains_text: {path}")
            continue
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
            if must not in txt:
                errors.append(f"file does not contain required text: {path} missing '{must}'")
        except Exception as e:
            errors.append(f"failed reading text file: {path} err={e}")

    return (len(errors) == 0), errors

def tee_run(cmd: List[str], cwd: Path, log_path: Path, tee: bool) -> int:
    with log_path.open("w", encoding="utf-8", errors="ignore") as lf:
        lf.write("[CMD] " + " ".join(cmd) + "\n\n")
        lf.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True
        )
        assert p.stdout is not None
        for line in p.stdout:
            lf.write(line)
            if tee:
                sys.stdout.write(line)
        return p.wait()

def get_git_commit(project_root: Path) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None

# -------------------------
# MINIMAL ADDITION: load step stats if produced
# -------------------------
def _try_load_step_stats(artifacts_dir: Path, step_name: str) -> Optional[Dict[str, Any]]:
    candidates: List[Path] = []

    # generic convention
    candidates.append(artifacts_dir / f"{step_name}_stats.json")

    # project conventions observed in this repo
    if step_name == "dsm":
        candidates.append(artifacts_dir / "dsm_float_stats.json")
    if step_name == "hillshade":
        candidates.append(artifacts_dir / "hillshade_stats.json")
    if step_name == "semantic":
        candidates.append(artifacts_dir / "semantic_stats.json")

    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                return {"parse_failed": str(e), "path": str(p)}
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_tag", default=None)
    ap.add_argument("--steps", default=None, help="comma-separated override pipeline")
    ap.add_argument("--from", dest="from_step", default=None)
    ap.add_argument("--to", dest="to_step", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="dotted.path=value overrides (can repeat)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg_raw = cfg_path.read_text(encoding="utf-8")
    cfg: Dict[str, Any] = json.loads(cfg_raw)

    if args.run_tag:
        cfg.setdefault("run", {})["run_tag"] = args.run_tag

    # apply --set overrides before resolution
    for s in args.sets:
        if "=" not in s:
            raise ValueError(f"--set expects dotted.path=value, got: {s}")
        k, v = s.split("=", 1)
        deep_set(cfg, k.strip(), parse_value(v.strip()))

    project_root = Path(cfg["paths"]["project_root"])
    runs_root = project_root / cfg["paths"]["runs_root"]
    run_tag = cfg["run"]["run_tag"]
    run_dir = runs_root / run_tag

    subdirs = ensure_dirs(run_dir)

    # Build context for ${...} resolution
    ctx = {
        "paths": {
            "project_root": str(project_root),
            "debug_root": cfg["paths"]["debug_root"],
            "runs_root": cfg["paths"]["runs_root"],
        },
        "run": {
            **cfg.get("run", {}),
            "run_dir": str(subdirs["run_dir"]),
            "logs_dir": str(subdirs["logs_dir"]),
            "artifacts_dir": str(subdirs["artifacts_dir"]),
            "ckpt_dir": str(subdirs["ckpt_dir"]),
            "renders_dir": str(subdirs["renders_dir"]),
            "metrics_dir": str(subdirs["metrics_dir"]),
        },
        "inputs": cfg.get("inputs", {}),
        "steps": cfg.get("steps", {}),
    }

    # Resolve whole cfg after ctx initialized
    cfg_resolved = resolve(cfg, ctx)

    # Persist resolved config into run dir (for reproducibility)
    resolved_path = subdirs["run_dir"] / "config_resolved.json"
    resolved_path.write_text(json.dumps(cfg_resolved, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "run_tag": run_tag,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "RUNNING",
        "config_path": str(cfg_path),
        "config_sha256": sha256_text(cfg_raw),
        "config_resolved_path": str(resolved_path),
        "git_commit": get_git_commit(project_root),
        "steps": {}
    }
    meta_path = subdirs["run_dir"] / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    pipeline = cfg_resolved.get("pipeline", [])
    if args.steps:
        pipeline = [x.strip() for x in args.steps.split(",") if x.strip()]

    if args.from_step or args.to_step:
        if args.from_step and args.from_step not in pipeline:
            raise ValueError(f"--from {args.from_step} not in pipeline")
        if args.to_step and args.to_step not in pipeline:
            raise ValueError(f"--to {args.to_step} not in pipeline")
        start = pipeline.index(args.from_step) if args.from_step else 0
        end = pipeline.index(args.to_step) if args.to_step else (len(pipeline) - 1)
        pipeline = pipeline[start:end+1]

    fail_fast = bool(cfg_resolved.get("run", {}).get("fail_fast", True))
    skip_done = bool(cfg_resolved.get("run", {}).get("skip_done", True))
    tee = bool(cfg_resolved.get("run", {}).get("log_tee", True))

    overall_ok = True

    for name in pipeline:
        step = cfg_resolved["steps"].get(name)
        if not step:
            raise KeyError(f"Step '{name}' not found in config.steps")
        if not step.get("enabled", True):
            meta["steps"][name] = {"skipped": True, "reason": "disabled"}
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            continue

        cwd = Path(step.get("cwd", str(project_root)))
        stype = step.get("type", "python")
        script = step["script"]
        script_path = project_root / script

        # Build argv
        argv: List[str] = []
        if stype == "python" or script_path.suffix.lower() == ".py":
            argv = [sys.executable, str(script_path)]
        else:
            argv = [str(script_path)]

        args_dict = step.get("args", {})
        if args_dict:
            argv.extend(args_dict_to_argv(args_dict))

        raw = step.get("args_raw", [])
        if raw:
            argv.extend([str(x) for x in raw])

        expects = step.get("expects", {})

        # skip_done check
        ok_pre, _ = check_expects(expects) if expects else (False, [])
        if skip_done and expects and ok_pre:
            meta["steps"][name] = {"skipped": True, "reason": "expects already satisfied"}
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            continue

        if args.dry_run:
            print(f"[DRY RUN] step={name}")
            print("  cmd:", " ".join(argv))
            meta["steps"][name] = {"dry_run": True, "command": argv}
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            continue

        log_path = subdirs["logs_dir"] / f"step_{name}.log"
        t0 = time.time()
        rc = tee_run(argv, cwd=cwd, log_path=log_path, tee=tee)
        dt = time.time() - t0

        ok, errs = check_expects(expects) if expects else (rc == 0, [] if rc == 0 else [f"return_code={rc}"])
        step_ok = (rc == 0) and ok
        overall_ok &= step_ok

        meta["steps"][name] = {
            "skipped": False,
            "command": argv,
            "cwd": str(cwd),
            "log": str(log_path),
            "return_code": rc,
            "duration_sec": round(dt, 3),
            "expects_passed": ok,
            "expects_errors": errs
        }

        # -------------------------
        # MINIMAL ADDITION: attach stats if present
        # -------------------------
        stats_obj = _try_load_step_stats(subdirs["artifacts_dir"], name)
        if stats_obj is not None:
            meta["steps"][name]["stats"] = stats_obj

        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        if not step_ok and fail_fast:
            break

    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["status"] = "PASS" if overall_ok else "FAIL"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    sys.exit(0 if overall_ok else 2)

if __name__ == "__main__":
    main()
