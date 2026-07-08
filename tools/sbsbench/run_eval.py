#!/usr/bin/env python3
"""
run_eval - one-command offline SBS benchmark: harness every committed clip through the real
pipeline, score it, and gate the result against the committed baselines.

This is the entry point for the eval->fix->eval loop (docs/sbs-benchmark-plan.md): it needs no
choreography knowledge (build dir, conf, per-clip commands, thresholds all resolved here), emits
a machine-readable results.json with provenance, and its EXIT CODE is the verdict:

  0  no regressions vs baseline (or no baselines yet)
  1  at least one metric regressed past its threshold  -> results.json lists them
  2  setup/run error (engines missing, harness failed, ...)

Typical use:
  python tools/sbsbench/run_eval.py                     # eval vs committed baselines
  python tools/sbsbench/run_eval.py --update-baselines  # after an INTENDED change: re-baseline
  python tools/sbsbench/run_eval.py --extra --divergence 0.027   # pass A/B levers to the harness

Results land in <build-dir>/sbs_eval/<label>/ (SBS+depth frames per clip + results.json).
Baselines/thresholds/conf are committed next to this script; changing bench.conf or the clip set
invalidates the baselines -- regenerate them in the same commit.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
import sbsbench  # noqa: E402  (metric implementations)

# Depth models the two modes load (matches the client mode->model binding).
MODE_MODEL = {"movie": "da3mono_large_fp16", "game": "depth_anything_v2_fp16"}


def sha1_dir(path):
    h = hashlib.sha1()
    for f in sorted(glob.glob(os.path.join(path, "*"))):
        with open(f, "rb") as fh:
            h.update(os.path.basename(f).encode())
            h.update(fh.read())
    return h.hexdigest()[:12]


def git(args):
    try:
        return subprocess.run(["git", "-C", REPO] + args, capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:
        return ""


def sunshine_running():
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq sunshine.exe", "/NH"],
                             capture_output=True, text=True, timeout=15).stdout
        return "sunshine.exe" in out
    except Exception:
        return False


def check_engines(build_dir, conf_path, mode):
    """Fail fast if a needed TRT engine isn't prebuilt (a first-use build stalls the loop for
    minutes and skews perf). Engines are named <stem>*.engine in the build assets dir."""
    assets = os.path.join(build_dir, "assets")
    stems = [MODE_MODEL[mode]]
    conf = open(conf_path, encoding="utf-8").read()
    for key in ("sbs_3d_warp_model", "sbs_3d_warp_model_movie"):
        m = re.search(rf"^{key}\s*=\s*([^\s#]+)", conf, re.M)
        if m:
            stems.append(m.group(1))
    missing = [s for s in stems if not glob.glob(os.path.join(assets, s + "*.engine"))]
    return missing


def worse_delta(base, new, spec):
    """Signed 'how much worse' (positive = worse) given the metric's better-direction."""
    return (base - new) if spec.get("better") == "higher" else (new - base)


def main():
    ap = argparse.ArgumentParser(description="Run the offline SBS benchmark over the committed clip set.")
    ap.add_argument("--build-dir", default=os.path.join(REPO, "cmake-build-relwithdebinfo"))
    ap.add_argument("--conf", default=os.path.join(SCRIPT_DIR, "bench.conf"))
    ap.add_argument("--clips", nargs="*", help="clip names (default: all in clips/)")
    ap.add_argument("--mode", choices=["movie", "game"], default="movie")
    ap.add_argument("--label", default=None, help="run label (default: timestamp)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra harness args, e.g. --extra --divergence 0.027")
    ap.add_argument("--update-baselines", action="store_true",
                    help="write this run as the new committed baselines (use after intended changes)")
    ap.add_argument("--allow-build", action="store_true", help="proceed even if engines are missing")
    args = ap.parse_args()

    exe = os.path.join(args.build_dir, "sunshine.exe")
    clips_dir = os.path.join(SCRIPT_DIR, "clips")
    base_dir = os.path.join(SCRIPT_DIR, "baselines")
    thresholds = json.load(open(os.path.join(SCRIPT_DIR, "thresholds.json")))
    clips = args.clips or sorted(os.path.basename(d) for d in glob.glob(os.path.join(clips_dir, "*"))
                                 if os.path.isdir(d))
    if not clips:
        sys.exit("run_eval: no clips in " + clips_dir)
    if not os.path.exists(exe):
        sys.exit(f"run_eval: {exe} not found -- build first (ninja -C cmake-build-relwithdebinfo sunshine)")

    missing = check_engines(args.build_dir, args.conf, args.mode)
    if missing and not args.allow_build:
        print(f"run_eval: TRT engine(s) missing in {args.build_dir}/assets: {missing}\n"
              f"Prebuild them (run Apollo once / sbs_3d_prebuild_models) or pass --allow-build.")
        sys.exit(2)

    contention = sunshine_running()
    if contention:
        print("run_eval: WARNING another sunshine.exe is running -- perf numbers will be noisy "
              "(tagged gpu_contention in results.json; perf gate skipped).")

    label = args.label or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.build_dir, "sbs_eval", label)
    os.makedirs(out_root, exist_ok=True)

    meta = {
        "git_sha": git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(git(["status", "--porcelain", "-uno"])),
        "clip_set_sha1": {c: sha1_dir(os.path.join(clips_dir, c)) for c in clips},
        "mode": args.mode, "extra_args": args.extra, "conf": os.path.relpath(args.conf, REPO),
        "gpu_contention": contention,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    results, regressions, issues = {}, [], []
    for clip in clips:
        clip_dir = os.path.join(clips_dir, clip)
        out_dir = os.path.join(out_root, clip)
        cmd = [exe, os.path.abspath(args.conf), "--sbs-bench",
               "--frames", clip_dir, "--out", out_dir]
        if args.mode == "movie":
            cmd.append("--movie")
        cmd += args.extra
        print(f"[{clip}] harness...", flush=True)
        try:
            r = subprocess.run(cmd, cwd=args.build_dir, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            sys.exit(f"run_eval: harness timed out on {clip}")
        stdout = r.stdout + r.stderr
        if r.returncode != 0 or not glob.glob(os.path.join(out_dir, "sbs_*.png")):
            print(stdout[-2000:])
            sys.exit(f"run_eval: harness failed on {clip} (exit {r.returncode})")
        m = re.search(r"model '([^']+)', warp '([^']+)'", stdout)
        clip_meta = {"model": m.group(1), "warp": m.group(2)} if m else {}

        print(f"[{clip}] scoring...", flush=True)
        rows, agg = sbsbench.measure_sequence(out_dir, clip_dir)
        perf = {}
        perf_p = os.path.join(out_dir, "sbs_perf.json")
        if os.path.exists(perf_p):
            stages = json.load(open(perf_p)).get("stages", {})
            perf = {k: v.get("p50_ms", 0) for k, v in stages.items()}

        # Worst frame per gated metric (per-frame key = aggregate key minus the _p50 suffix),
        # so a triggered/regressed metric comes with a place to look.
        worst = {}
        for k in thresholds["metrics"]:
            fk = k if any(k in r for r in rows) else (k[:-4] if k.endswith("_p50") else k)
            vals = [(r.get(fk), i) for i, r in enumerate(rows) if fk in r]
            if vals:
                v, i = max(vals)
                worst[k] = {"frame": i, "value": round(v, 3)}

        entry = {"aggregate": agg, "perf_ms": perf, "meta": clip_meta, "worst_frame": worst}
        results[clip] = entry

        # Issue triggers (absolute, baseline-independent).
        for k, spec in thresholds["metrics"].items():
            if "trigger" in spec and agg.get(k, 0) > spec["trigger"]:
                issues.append({"clip": clip, "metric": k, "value": round(agg[k], 3),
                               "trigger": spec["trigger"], **worst.get(k, {})})

        # Regression gate vs baseline. A baseline is only valid for the exact frames it was made
        # from: if the clip content changed, gating against it is meaningless -- skip it loudly
        # instead of silently comparing apples to oranges.
        bp = os.path.join(base_dir, clip + ".json")
        if os.path.exists(bp) and not args.update_baselines:
            base = json.load(open(bp))
            base_sha = base.get("meta", {}).get("clip_sha1")
            if base_sha and base_sha != meta["clip_set_sha1"][clip]:
                print(f"run_eval: WARNING {clip} frames changed since its baseline "
                      f"({base_sha} -> {meta['clip_set_sha1'][clip]}); skipping its gate. "
                      f"Re-baseline with --update-baselines.")
                entry["stale_baseline"] = True
                results[clip] = entry
                continue
            for k, spec in thresholds["metrics"].items():
                b, n = base["aggregate"].get(k), agg.get(k)
                if b is None or n is None:
                    continue
                wd = worse_delta(b, n, spec)
                if wd > max(spec["abs_floor"], abs(b) * spec["rel_tol"]):
                    regressions.append({"clip": clip, "metric": k, "baseline": round(b, 3),
                                        "value": round(n, 3), **worst.get(k, {})})
            if not contention:
                for k, spec in thresholds["perf_ms"].items():
                    b, n = base.get("perf_ms", {}).get(k), perf.get(k)
                    if b and n and (n - b) > max(spec["abs_floor"], b * spec["rel_tol"]):
                        regressions.append({"clip": clip, "metric": "perf:" + k,
                                            "baseline": round(b, 2), "value": round(n, 2)})

        if args.update_baselines:
            os.makedirs(base_dir, exist_ok=True)
            json.dump({"aggregate": agg, "perf_ms": perf,
                       "meta": {**meta, **clip_meta, "clip_sha1": meta["clip_set_sha1"][clip]}},
                      open(bp, "w"), indent=2)

    verdict = "regressions" if regressions else "pass"
    out = {"meta": meta, "verdict": verdict, "regressions": regressions, "issues": issues,
           "clips": results}
    res_path = os.path.join(out_root, "results.json")
    json.dump(out, open(res_path, "w"), indent=2)

    print(f"\n=== {verdict.upper()} ===  ({res_path})")
    for r in regressions:
        print(f"  REGRESSION {r['clip']}.{r['metric']}: {r['baseline']} -> {r['value']}"
              + (f"  (worst frame {r['frame']})" if "frame" in r else ""))
    for i in issues:
        print(f"  issue {i['clip']}.{i['metric']} = {i['value']} (> {i['trigger']},"
              f" worst frame {i.get('frame', '?')})")
    if args.update_baselines:
        print(f"  baselines updated in {base_dir} -- commit them with the change that justified it.")
    sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()
