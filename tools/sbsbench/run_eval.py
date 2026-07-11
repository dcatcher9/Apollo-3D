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
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
import sbsbench  # noqa: E402  (metric implementations)

# Depth models the two modes load (matches the client mode->model binding).
MODE_MODEL = {"movie": "da3mono_large_fp16", "game": "depth_anything_v2_fp16"}
EVAL_SCHEMA = 4  # schema 3 + native metric-depth and exact optical-flow reference sidecars


def suite_defaults(name):
    if name == "core":
        return os.path.join(SCRIPT_DIR, "clips"), os.path.join(SCRIPT_DIR, "baselines")
    manifest_path = os.path.join(SCRIPT_DIR, "datasets", "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        fail(f"cannot load extended-suite manifest {manifest_path}: {exc}")
    cache = os.environ.get("APOLLO_SBS_DATASETS") or manifest["default_cache"]
    clips = os.path.join(os.path.abspath(cache), "prepared", manifest["prepared_suite"])
    return clips, os.path.join(SCRIPT_DIR, "baselines_extended")


def fail(message):
    print("run_eval: " + message, file=sys.stderr)
    raise SystemExit(2)


def sha256_files(paths):
    h = hashlib.sha256()
    for path in paths:
        h.update(os.path.basename(path).encode())
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def sha1_dir(path):
    # Hash source pixels plus validation references. Human-readable names/descriptions remain
    # excluded, while semantic metadata that changes scoring is part of the contract.
    h = hashlib.sha1()
    files = (glob.glob(os.path.join(path, "frame_*"))
             + glob.glob(os.path.join(path, "gt_depth", "frame_*"))
             + glob.glob(os.path.join(path, "gt_flow", "frame_*")))
    for f in sorted(files):
        with open(f, "rb") as fh:
            h.update(os.path.relpath(f, path).replace("\\", "/").encode())
            h.update(fh.read())
    try:
        meta = json.load(open(os.path.join(path, "meta.json"), encoding="utf-8"))
        semantic = {k: meta[k] for k in ("expected_flat", "gt_depth_kind") if k in meta}
        h.update(json.dumps(semantic, sort_keys=True).encode())
    except (OSError, ValueError):
        pass
    return h.hexdigest()[:12]


def extra_value(args, name, default=None):
    """Return the last explicit value for a two-token harness override."""
    value = default
    for i, token in enumerate(args[:-1]):
        if token == name:
            value = args[i + 1]
    return value


def conf_value(path, name, default=None):
    """Read a simple Sunshine `name = value` setting without interpreting unrelated config."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if "=" not in line:
                    continue
                key, value = (part.strip() for part in line.split("=", 1))
                if key == name:
                    default = value
    except OSError:
        pass
    return default


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


def check_engines(build_dir, mode):
    """Fail fast if a needed TRT engine isn't prebuilt (a first-use build stalls the loop for
    minutes and skews perf). Engines are named <stem>*.engine in the build assets dir."""
    assets = os.path.join(build_dir, "assets")
    stems = [MODE_MODEL[mode]]
    missing = [s for s in stems if not glob.glob(os.path.join(assets, s + "*.engine"))]
    return missing


def main():
    ap = argparse.ArgumentParser(description="Run the offline SBS benchmark over a reproducible clip suite.")
    ap.add_argument("--build-dir", default=os.path.join(REPO, "cmake-build-relwithdebinfo"))
    ap.add_argument("--conf", default=os.path.join(SCRIPT_DIR, "bench.conf"))
    ap.add_argument("--clips", nargs="*", help="clip names (default: all in clips/)")
    ap.add_argument("--suite", choices=["core", "extended"], default="core",
                    help="quick committed suite or prepared public-data suite")
    ap.add_argument("--clips-root", help="override suite source directory")
    ap.add_argument("--baseline-dir", help="override suite baseline directory")
    ap.add_argument("--mode", choices=["movie", "game"], default="game")
    ap.add_argument("--label", default=None, help="run label (default: timestamp)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra harness args, e.g. --extra --divergence 0.027")
    ap.add_argument("--update-baselines", action="store_true",
                    help="write this run as the new committed baselines (use after intended changes)")
    ap.add_argument("--comparison-only", action="store_true",
                    help="measure without committed-baseline gating (for a fresh matched A/B pair)")
    ap.add_argument("--report-control",
                    help="control run directory; generate an A/B HTML report after this run")
    ap.add_argument("--report-out",
                    help="report path (default: <this run>/report.html; requires --report-control)")
    ap.add_argument("--report-allow-config-diff", action="store_true",
                    help="allow an explicit profile-vs-profile report with different config hashes; clips/model/metrics must still match")
    ap.add_argument("--allow-build", action="store_true", help="proceed even if engines are missing")
    args = ap.parse_args()
    if args.comparison_only and args.update_baselines:
        fail("--comparison-only and --update-baselines are mutually exclusive")

    exe = os.path.join(args.build_dir, "sunshine.exe")
    default_clips, default_baselines = suite_defaults(args.suite)
    clips_dir = os.path.abspath(args.clips_root or default_clips)
    base_dir = os.path.abspath(args.baseline_dir or default_baselines)
    thresholds = json.load(open(os.path.join(SCRIPT_DIR, "thresholds.json")))
    clips = args.clips or sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(clips_dir, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "frame_*.*")))
    if not clips:
        fail("no clips in " + clips_dir)
    if not args.update_baselines and not args.comparison_only:
        missing_baselines = [c for c in clips if not os.path.exists(os.path.join(base_dir, c + ".json"))]
        if missing_baselines:
            fail(f"missing committed baseline(s) in {base_dir}: {missing_baselines}. "
                 "Use --comparison-only for a matched A/B or --update-baselines after validation.")
    if not os.path.exists(exe):
        fail(f"{exe} not found -- build first (ninja -C cmake-build-relwithdebinfo sunshine)")

    missing = check_engines(args.build_dir, args.mode)
    if missing and not args.allow_build:
        print(f"run_eval: TRT engine(s) missing in {args.build_dir}/assets: {missing}\n"
              f"Prebuild them (run Apollo once / sbs_3d_prebuild_models) or pass --allow-build.")
        raise SystemExit(2)

    contention = sunshine_running()
    if contention:
        print("run_eval: WARNING another sunshine.exe is running -- perf numbers will be noisy "
              "(tagged gpu_contention in results.json; perf gate skipped).")

    label = args.label or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.build_dir, "sbs_eval", label)
    os.makedirs(out_root, exist_ok=True)

    conf_sha = sha256_files([os.path.abspath(args.conf)])
    metric_sha = sha256_files([os.path.join(SCRIPT_DIR, "sbsbench.py"),
                               os.path.join(SCRIPT_DIR, "thresholds.json")])
    expected_model = MODE_MODEL[args.mode]
    expected_warp = extra_value(
        args.extra, "--warp", conf_value(args.conf, "sbs_3d_warp", "apollo"))
    if expected_warp not in {"apollo", "vd3d"}:
        fail(f"invalid --warp override {expected_warp!r}")
    expected_shift_profile = extra_value(
        args.extra, "--shift-profile", conf_value(args.conf, "sbs_3d_shift_profile", "apollo"))
    if expected_shift_profile not in {"apollo", "bestv2"}:
        fail(f"invalid --shift-profile override {expected_shift_profile!r}")
    meta = {
        "git_sha": git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(git(["status", "--porcelain"])),
        "clip_set_sha1": {c: sha1_dir(os.path.join(clips_dir, c)) for c in clips},
        "mode": args.mode, "suite": args.suite, "clips_root": clips_dir,
        "extra_args": args.extra,
        "conf": os.path.relpath(args.conf, REPO),
        "model": expected_model, "warp": expected_warp, "shift_profile": expected_shift_profile,
        "eval_schema": EVAL_SCHEMA, "depth_step": "current-once",
        "conf_sha256": conf_sha, "metric_sha256": metric_sha,
        "gpu_contention": contention,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "run_name": label,
    }

    results, regressions, issues, hard_failures, baseline_updates = {}, [], [], [], {}
    for clip in clips:
        clip_dir = os.path.join(clips_dir, clip)
        out_dir = os.path.join(out_root, clip)
        shutil.rmtree(out_dir, ignore_errors=True)  # a reused label must not retain stale frame IDs
        cmd = [exe, os.path.abspath(args.conf), "--sbs-bench",
               "--frames", clip_dir, "--out", out_dir,
               "--model", expected_model]
        cmd += args.extra
        print(f"[{clip}] harness...", flush=True)
        try:
            r = subprocess.run(cmd, cwd=args.build_dir, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            fail(f"harness timed out on {clip}")
        stdout = r.stdout + r.stderr
        if r.returncode != 0 or not glob.glob(os.path.join(out_dir, "sbs_*.png")):
            print(stdout[-2000:])
            fail(f"harness failed on {clip} (exit {r.returncode})")
        m = re.search(r"model '([^']+)'", stdout)
        actual_model = m.group(1) if m else None
        if actual_model != expected_model:
            fail(f"{clip}: expected model {expected_model!r}, harness reported {actual_model!r}")
        if "depth_step current-once" not in stdout:
            fail(f"{clip}: harness did not confirm current-once depth stepping")
        warp_match = re.search(r"depth_step current-once, warp ([a-z0-9_-]+)", stdout)
        actual_warp = warp_match.group(1) if warp_match else None
        if actual_warp != expected_warp:
            fail(f"{clip}: expected warp {expected_warp!r}, harness reported {actual_warp!r}")
        profile_match = re.search(r"shift_profile ([a-z0-9_-]+)", stdout)
        actual_shift_profile = profile_match.group(1) if profile_match else None
        if actual_shift_profile != expected_shift_profile:
            fail(f"{clip}: expected shift profile {expected_shift_profile!r}, "
                 f"harness reported {actual_shift_profile!r}")
        clip_meta = {"model": actual_model, "warp": actual_warp,
                     "shift_profile": actual_shift_profile}

        # A valid harness result has one source, raw-model, warp-input depth, and SBS artifact for
        # every numeric frame identity. This catches dropped/renumbered outputs before metrics run.
        source_ids = set(sbsbench.indexed_files(os.path.join(clip_dir, "frame_*.*"), "frame_"))
        sbs_ids = set(sbsbench.indexed_files(os.path.join(out_dir, "sbs_*.png"), "sbs_"))
        depth_ids = set(sbsbench.indexed_files(os.path.join(out_dir, "depth_*.png"), "depth_"))
        raw_ids = set(sbsbench.indexed_files(os.path.join(out_dir, "raw_*.f32"), "raw_"))
        if not source_ids or source_ids != sbs_ids or source_ids != depth_ids or source_ids != raw_ids:
            fail(f"{clip}: artifact frame-id mismatch source={sorted(source_ids)} "
                 f"sbs={sorted(sbs_ids)} depth={sorted(depth_ids)} raw={sorted(raw_ids)}")
        if not os.path.exists(os.path.join(out_dir, "raw_shape.json")):
            fail(f"{clip}: raw_shape.json missing")
        # Carry the clip's own metadata (scene name/description) into results so the run dir is
        # self-describing and the report can label clips without the source clips dir.
        cmp_path = os.path.join(clip_dir, "meta.json")
        if os.path.exists(cmp_path):
            try:
                clip_meta.update({k: v for k, v in json.load(open(cmp_path)).items()
                                  if k in ("name", "description", "expected_flat", "gt_depth_kind",
                                           "dataset", "homepage", "citation", "license_note", "suite")})
            except Exception:
                pass

        print(f"[{clip}] scoring...", flush=True)
        try:
            rows, agg = sbsbench.measure_sequence(
                out_dir, clip_dir, expected_flat=bool(clip_meta.get("expected_flat")))
        except ValueError as exc:
            fail(f"{clip}: {exc}")
        perf = {}
        perf_p = os.path.join(out_dir, "sbs_perf.json")
        if os.path.exists(perf_p):
            stages = json.load(open(perf_p)).get("stages", {})
            perf = {k: v.get("p50_ms", 0) for k, v in stages.items()}

        # Worst frame per gated metric (per-frame key = aggregate key minus the _p50 suffix),
        # so a triggered/regressed metric comes with a place to look.
        worst = {}
        for k in thresholds["metrics"]:
            if clip_meta.get("expected_flat") and k == "pop_spread_px":
                continue  # expected-flat score rewards lower false stereo
            fk = k if any(k in r for r in rows) else (
                k[:-4] if k.endswith(("_p50", "_p95")) else k)
            vals = [(r.get(fk), r.get("_frame_id", i)) for i, r in enumerate(rows) if fk in r]
            if vals:
                # "Worst" follows the metric direction. For score/pop/depth spread, the minimum
                # is the bad frame; artifact metrics use the maximum.
                choose = min if thresholds["metrics"][k].get("better") == "higher" else max
                v, i = choose(vals)
                worst[k] = {"frame": i, "worst_value": round(v, 3)}

        entry = {"aggregate": agg, "perf_ms": perf, "meta": clip_meta, "worst_frame": worst}
        results[clip] = entry

        for k, spec in thresholds["metrics"].items():
            if spec.get("role") != "hard" or k not in agg:
                continue
            if sbsbench.metric_gate_failed(agg[k], agg[k], spec):
                hard_failures.append({"clip": clip, "metric": k,
                                      **worst.get(k, {}), "value": round(agg[k], 3),
                                      "hard_min": spec.get("hard_min"),
                                      "hard_max": spec.get("hard_max")})

        # Issue triggers (absolute, baseline-independent).
        for k, spec in thresholds["metrics"].items():
            if clip_meta.get("expected_flat") and k == "pop_spread_px":
                continue
            if "trigger" in spec and agg.get(k, 0) > spec["trigger"]:
                issues.append({"clip": clip, "metric": k, "trigger": spec["trigger"],
                               **worst.get(k, {}), "value": round(agg[k], 3)})
            if "trigger_min" in spec and k in agg and agg[k] < spec["trigger_min"]:
                issues.append({"clip": clip, "metric": k, "trigger_min": spec["trigger_min"],
                               **worst.get(k, {}), "value": round(agg[k], 3)})

        # Regression gate vs baseline. A baseline is only valid for the exact frames it was made
        # from: if the clip content changed, gating against it is meaningless -- skip it loudly
        # instead of silently comparing apples to oranges.
        bp = os.path.join(base_dir, clip + ".json")
        if os.path.exists(bp) and not args.update_baselines and not args.comparison_only:
            base = json.load(open(bp))
            base_meta = base.get("meta", {})
            required = {
                "clip_sha1": meta["clip_set_sha1"][clip],
                "mode": args.mode,
                "model": expected_model,
                "eval_schema": EVAL_SCHEMA,
                "depth_step": "current-once",
                "conf_sha256": conf_sha,
                "metric_sha256": metric_sha,
            }
            mismatches = {k: (base_meta.get(k), v) for k, v in required.items()
                          if base_meta.get(k) != v}
            if mismatches:
                fail(f"{clip}: baseline context is stale/incompatible: {mismatches}. "
                     "Re-run with --update-baselines only after verifying the new eval contract.")
            for k, spec in thresholds["metrics"].items():
                if clip_meta.get("expected_flat") and k == "pop_spread_px":
                    continue
                if spec.get("role") == "hard":
                    continue  # absolute hard constraints were evaluated above, independent of baseline
                b, n = base["aggregate"].get(k), agg.get(k)
                if b is None or n is None:
                    continue
                if sbsbench.metric_gate_failed(b, n, spec):
                    regressions.append({"clip": clip, "metric": k, "baseline": round(b, 3),
                                        **worst.get(k, {}), "value": round(n, 3)})
            if not contention:
                for k, spec in thresholds["perf_ms"].items():
                    b, n = base.get("perf_ms", {}).get(k), perf.get(k)
                    if b and n and (n - b) > max(spec["abs_floor"], b * spec["rel_tol"]):
                        regressions.append({"clip": clip, "metric": "perf:" + k,
                                            "baseline": round(b, 2), "value": round(n, 2)})

        if args.update_baselines:
            baseline_updates[bp] = {
                "aggregate": agg, "perf_ms": perf,
                "meta": {**meta, **clip_meta, "clip_sha1": meta["clip_set_sha1"][clip]}}

    verdict = ("hard_failures" if hard_failures else "comparison_only" if args.comparison_only
               else "regressions" if regressions else "pass")
    out = {"meta": meta, "verdict": verdict, "regressions": regressions,
           "hard_failures": hard_failures, "issues": issues, "clips": results}
    res_path = os.path.join(out_root, "results.json")
    json.dump(out, open(res_path, "w"), indent=2)

    if args.update_baselines:
        if hard_failures:
            fail(f"refusing baseline update: {len(hard_failures)} hard comfort/integrity "
                 "failure(s); results preserved at " + res_path)
        os.makedirs(base_dir, exist_ok=True)
        for path, payload in baseline_updates.items():
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, path)

    report_path = None
    if args.report_out and not args.report_control:
        fail("--report-out requires --report-control")
    if args.report_control:
        control_dir = os.path.abspath(args.report_control)
        if not os.path.exists(os.path.join(control_dir, "results.json")):
            fail(f"report control has no results.json: {control_dir}")
        report_path = os.path.abspath(args.report_out or os.path.join(out_root, "report.html"))
        report_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "build_report.py"),
                      control_dir, out_root, report_path]
        if args.report_allow_config_diff:
            report_cmd.append("--allow-config-diff")
        report_run = subprocess.run(report_cmd, capture_output=True, text=True)
        if report_run.returncode:
            fail("report generation failed: " + (report_run.stderr or report_run.stdout)[-2000:])

    print(f"\n=== {verdict.upper()} ===  ({res_path})")
    for r in regressions:
        print(f"  REGRESSION {r['clip']}.{r['metric']}: {r['baseline']} -> {r['value']}"
              + (f"  (worst frame {r['frame']}, frame value {r['worst_value']})"
                 if "frame" in r else ""))
    for r in hard_failures:
        bounds = ", ".join(f"{k}={v}" for k, v in
                           (("min", r.get("hard_min")), ("max", r.get("hard_max")))
                           if v is not None)
        print(f"  HARD FAIL {r['clip']}.{r['metric']}: {r['value']} ({bounds})")
    for i in issues:
        relation = (f"> {i['trigger']}" if "trigger" in i else f"< {i['trigger_min']}")
        print(f"  issue {i['clip']}.{i['metric']} = {i['value']} ({relation},"
              f" worst frame {i.get('frame', '?')}, frame value {i.get('worst_value', '?')})")
    if report_path:
        print(f"  report: {report_path}")
    if args.update_baselines:
        print(f"  baselines updated in {base_dir} -- commit them with the change that justified it.")
    sys.exit(1 if regressions or hard_failures else 0)


if __name__ == "__main__":
    main()
