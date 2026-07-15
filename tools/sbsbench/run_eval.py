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
  python tools/sbsbench/run_eval.py --extra --subject-lock 0.6  # pass supported A/B levers

Results land in <build-dir>/sbs_eval/<label>/ (SBS+depth frames per clip + results.json).
Baselines/thresholds/conf are committed next to this script; changing bench.conf or the clip set
invalidates the baselines -- regenerate them in the same commit.
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
import sbsbench  # noqa: E402  (metric implementations)

EVAL_SCHEMA = 29  # explicit policy authorization + output-eye exact gates; harness 24
HARNESS_SCHEMA = 24

BASELINE_CONTEXT_FIELDS = (
    "mode", "suite", "extra_args", "model", "profile",
    "adaptive_pop", "adaptive_pop_max", "pop_strength",
    "ema", "ema_edge_change", "ema_edge_gradient", "ema_edge_strength",
    "minmax_ema", "subject_lock", "subject_recenter", "subject_stretch",
    "depth_short_side", "depth_max_aspect", "zero_plane", "cuda_graph",
    "artistic_style", "artistic_policy", "artistic_scale_override",
    "artistic_policy_consumed", "artistic_policy_authorization",
    "model_onnx_sha256", "policy_metadata_sha256",
    "deployment_geometry_allowlist_sha256",
    "output_interval", "output_gt_right_only", "literal_bestv2",
    "depth_compensation", "depth_override_frames", "depth_step",
    "depth_reuse_interval", "eval_schema", "conf_sha256", "metric_sha256",
    "policy_warp_source_sha256", "clip_sha1",
    "harness_schema", "source_width", "source_height", "model_input_width",
    "model_input_height", "eye_width", "eye_height", "color_mode",
    "content_scale_x", "content_scale_y", "disparity_raster_width",
    "disparity_raster_height", "artifact_mode", "warp_disparity",
    "warp_unclamped_disparity", "artistic_disparity_contract",
    "artistic_full_clamp_abs", "warp_mask",
)

# These fields intentionally identify the learned treatment rather than compatibility with the
# committed Apollo control. A policy-candidate gate still requires every geometry, cadence,
# evaluator, warp and resolved profile field above to match its baseline.
POLICY_CANDIDATE_TREATMENT_FIELDS = {
    "extra_args", "model", "conf_sha256", "artistic_style", "artistic_policy",
    "artistic_scale_override", "artistic_policy_consumed",
    "artistic_policy_authorization", "model_onnx_sha256",
    "policy_metadata_sha256", "deployment_geometry_allowlist_sha256",
}


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
            data = fh.read()
        # The evaluation contract is source semantics, not Git checkout EOL policy. Without this,
        # committing on Windows can invalidate freshly written baselines even though the Git blob
        # and Python behavior are unchanged.
        if os.path.splitext(path)[1].lower() in {".py", ".json", ".conf", ".md", ".hlsl"}:
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        h.update(data)
    return h.hexdigest()[:16]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_contract_sha():
    """Hash automatic metric implementation and thresholds.

    Runner/gating semantics are versioned separately by EVAL_SCHEMA. Hashing this entire file made
    comments and diagnostic wording invalidate otherwise-identical committed baselines.
    """
    return sha256_files([os.path.join(SCRIPT_DIR, "sbsbench.py"),
                         os.path.join(SCRIPT_DIR, "thresholds.json"),
                         os.path.abspath(__file__)])


def sha1_dir(path):
    # Hash source pixels plus validation references. Human-readable names/descriptions remain
    # excluded, while semantic metadata that changes scoring is part of the contract.
    h = hashlib.sha1()
    files = (glob.glob(os.path.join(path, "frame_*"))
             + glob.glob(os.path.join(path, "gt_depth", "frame_*"))
             + glob.glob(os.path.join(path, "gt_flow", "frame_*"))
             + glob.glob(os.path.join(path, "gt_right", "frame_*")))
    for f in sorted(files):
        with open(f, "rb") as fh:
            h.update(os.path.relpath(f, path).replace("\\", "/").encode())
            h.update(fh.read())
    try:
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as meta_file:
            meta = json.load(meta_file)
        semantic = {k: meta[k] for k in ("expected_flat", "gt_depth_kind", "dataset",
                                         "required_gt_depth", "required_gt_flow",
                                         "required_gt_stereo") if k in meta}
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


def metric_exempt_for_clip(spec, clip_meta):
    """Expected-flat clips diagnose false stereo instead of gating the stereo-volume axis."""
    return bool(clip_meta.get("expected_flat")) and spec.get("axis") == "stereo"


def missing_required_metric_evidence(aggregate, thresholds, clip_meta):
    """List always-required primary evidence absent from an otherwise scored clip."""
    missing = []
    for metric, spec in thresholds["metrics"].items():
        if metric_exempt_for_clip(spec, clip_meta):
            continue
        if not sbsbench.metric_evidence_required(spec, aggregate):
            continue
        value = aggregate.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            missing.append(metric)
    return missing


def validate_depth_override_manifest(root, clips_dir, clips, depth_every, override_all=False):
    """Validate an offline depth treatment before the harness can consume any of it.

    The override is deliberately fail-closed: a partial or stale directory must never be
    indistinguishable from a valid treatment. Returns the expected applied-frame count per clip.
    """
    manifest_path = os.path.join(root, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        fail(f"invalid depth-override manifest {manifest_path}: {exc}")
    expected_header = {
        "schema": 3,
        "method": ("flow-aware-ema-oracle" if override_all else
                   "classical-tile-phase-flow"),
        "depth_every": depth_every,
        "frame_policy": "all" if override_all else "held",
    }
    header_mismatch = {key: (value, manifest.get(key)) for key, value in expected_header.items()
                       if manifest.get(key) != value}
    if header_mismatch:
        fail(f"incompatible depth-override manifest: {header_mismatch}")
    manifest_clips = manifest.get("clips")
    if not isinstance(manifest_clips, dict):
        fail("depth-override manifest clips must be an object")

    counts = {}
    for clip in clips:
        clip_info = manifest_clips.get(clip)
        if not isinstance(clip_info, dict):
            fail(f"depth-override manifest lacks clip {clip}")
        clip_dir = os.path.join(clips_dir, clip)
        clip_sha = sha1_dir(clip_dir)
        if clip_info.get("clip_sha1") != clip_sha:
            fail(f"{clip}: depth override source hash mismatch: "
                 f"{clip_info.get('clip_sha1')} != {clip_sha}")
        source_ids = sorted(sbsbench.indexed_files(
            os.path.join(clip_dir, "frame_*.*"), "frame_"))
        expected_ids = (source_ids if override_all else
                        [frame_id for position, frame_id in enumerate(source_ids)
                         if position % depth_every != 0])
        recorded_ids = clip_info.get("override_frame_ids")
        if recorded_ids != expected_ids:
            fail(f"{clip}: manifest override-frame identities differ: "
                 f"expected={expected_ids}, recorded={recorded_ids}")
        actual_ids = sorted(sbsbench.indexed_files(
            os.path.join(root, clip, "depth_*.png"), "depth_"))
        if actual_ids != expected_ids:
            fail(f"{clip}: depth-override frame identities differ: "
                 f"expected={expected_ids}, actual={actual_ids}")
        if clip_info.get("override_frames") != len(expected_ids):
            fail(f"{clip}: manifest override-frame count is inconsistent")
        counts[clip] = len(expected_ids)
    return counts


def score_clip_gates(rows, agg, thresholds, clip_meta):
    """Return worst-frame evidence, absolute issues, and hard failures for one clip."""
    worst, issues, hard_failures = {}, [], []
    for metric, spec in thresholds["metrics"].items():
        if metric_exempt_for_clip(spec, clip_meta):
            continue
        frame_key = metric if any(metric in row for row in rows) else (
            metric[:-4] if metric.endswith(("_p50", "_p95")) else metric)
        values = [(row.get(frame_key), row.get("_frame_id", i))
                  for i, row in enumerate(rows) if frame_key in row]
        if values:
            choose = min if spec.get("better") == "higher" else max
            value, frame = choose(values)
            worst[metric] = {"frame": frame, "worst_value": round(value, 3)}
        aggregate_value = agg.get(metric)
        finite_aggregate = (isinstance(aggregate_value, (int, float)) and
                            math.isfinite(aggregate_value))
        if "trigger" in spec and finite_aggregate and aggregate_value > spec["trigger"]:
            issues.append({"metric": metric, "trigger": spec["trigger"],
                           **worst.get(metric, {}), "value": round(aggregate_value, 3)})
        if ("trigger_min" in spec and finite_aggregate and
                aggregate_value < spec["trigger_min"]):
            issues.append({"metric": metric, "trigger_min": spec["trigger_min"],
                           **worst.get(metric, {}), "value": round(aggregate_value, 3)})
        if spec.get("role") == "hard":
            value = aggregate_value
            if not finite_aggregate:
                hard_failures.append({"metric": metric, **worst.get(metric, {}),
                                      "value": None, "reason": "missing",
                                      "hard_min": spec.get("hard_min"),
                                      "hard_max": spec.get("hard_max")})
            elif sbsbench.metric_gate_failed(value, value, spec):
                hard_failures.append({"metric": metric, **worst.get(metric, {}),
                                      "value": round(value, 3),
                                      "hard_min": spec.get("hard_min"),
                                      "hard_max": spec.get("hard_max")})
    return worst, issues, hard_failures


def normalize_cli_paths(args):
    """Resolve path arguments before constructing outputs or selecting a subprocess cwd."""
    args.build_dir = os.path.abspath(args.build_dir)
    args.conf = os.path.abspath(args.conf)
    if args.clips_root:
        args.clips_root = os.path.abspath(args.clips_root)
    if args.baseline_dir:
        args.baseline_dir = os.path.abspath(args.baseline_dir)
    if args.report_control:
        args.report_control = os.path.abspath(args.report_control)
    if args.report_out:
        args.report_out = os.path.abspath(args.report_out)
    return args


def require_current_build(build_dir):
    """Build the production target so evaluation cannot accidentally run a stale executable."""
    ninja = shutil.which("ninja")
    cache_path = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8", errors="replace") as fh:
            match = re.search(r"^CMAKE_MAKE_PROGRAM:FILEPATH=(.+)$", fh.read(), re.MULTILINE)
        if match:
            ninja = match.group(1).strip()
    if not ninja:
        fail("cannot verify the Sunshine build is current: Ninja was not found")
    try:
        probe = subprocess.run(
            [ninja, "-C", build_dir, "sunshine"],
            capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as exc:
        fail(f"cannot verify the Sunshine build is current: {exc}")
    output = (probe.stdout or "") + (probe.stderr or "")
    if probe.returncode:
        fail("cannot build the current Sunshine executable: " + output[-2000:])


def expected_profile(conf, extra):
    """Resolve the startup production profile; every profile uses Apollo geometry."""
    profile = conf_value(conf, "sbs_3d_profile", "apollo")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", profile):
        fail(f"invalid sbs_3d_profile {profile!r}")
    return profile


def expected_profile_number(conf, profile, key, default, extra, cli_key, cast=float):
    value = conf_value(conf, f"sbs_3d_profile_{profile}_{key}", default)
    value = conf_value(conf, f"sbs_3d_{key}", value)
    value = extra_value(extra, cli_key, value)
    try:
        return cast(value)
    except (TypeError, ValueError):
        fail(f"invalid numeric value for {key}: {value!r}")


def expected_profile_bool(conf, profile, key, default, extra, cli_key):
    value = conf_value(conf, f"sbs_3d_profile_{profile}_{key}", default)
    value = conf_value(conf, f"sbs_3d_{key}", value)
    value = extra_value(extra, cli_key, value)
    normalized = str(value).strip().lower()
    if normalized in ("true", "yes", "on", "1"):
        return True
    if normalized in ("false", "no", "off", "0"):
        return False
    fail(f"invalid boolean value for {key}: {value!r}")


def expected_profile_string(conf, profile, key, default, extra, cli_key):
    value = conf_value(conf, f"sbs_3d_profile_{profile}_{key}", default)
    value = conf_value(conf, f"sbs_3d_{key}", value)
    return str(extra_value(extra, cli_key, value)).strip()


def expected_adaptive_pop(conf, profile, extra):
    """Resolve the flag-style harness override after the production config layers."""
    value = expected_profile_bool(conf, profile, "adaptive_pop", True, [], "")
    enabled_at = max((i for i, item in enumerate(extra) if item == "--adaptive-pop"),
                     default=-1)
    disabled_at = max((i for i, item in enumerate(extra) if item == "--no-adaptive-pop"),
                      default=-1)
    if enabled_at >= 0 or disabled_at >= 0:
        value = enabled_at > disabled_at
    return value


def expected_subject_stretch(conf, profile, extra):
    """Resolve the stretch flag after profile and explicit harness overrides."""
    value = expected_profile_bool(conf, profile, "subject_stretch", True, [], "")
    enabled_at = max((i for i, item in enumerate(extra)
                      if item == "--subject-stretch"), default=-1)
    disabled_at = max((i for i, item in enumerate(extra)
                       if item == "--no-subject-stretch"), default=-1)
    if enabled_at >= 0 or disabled_at >= 0:
        value = enabled_at > disabled_at
    return value


def expected_depth_model(conf, profile, extra):
    """Resolve the model with the same profile-first, explicit-override order as production."""
    model = "depth_anything_v2_fp16"
    model = conf_value(conf, f"sbs_3d_profile_{profile}_depth_model", model)
    model = conf_value(conf, "sbs_3d_depth_model", model)
    return extra_value(extra, "--model", model)


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


def check_engines(build_dir, model):
    """Fail fast if a needed TRT engine isn't prebuilt (a first-use build stalls the loop for
    minutes and skews perf). Engines are named <stem>*.engine in the build assets dir."""
    assets = os.path.join(build_dir, "assets")
    stems = [model]
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
    ap.add_argument("--label", default=None, help="run label (default: timestamp)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra harness args, e.g. --extra --subject-lock 0.6")
    ap.add_argument("--update-baselines", action="store_true",
                    help="write this run as the new committed baselines (use after intended changes)")
    ap.add_argument("--comparison-only", action="store_true",
                    help="measure without committed-baseline gating (for a fresh matched A/B pair)")
    ap.add_argument("--report-control",
                    help="control run directory; generate an A/B HTML report after this run")
    ap.add_argument("--report-out",
                    help="report path (default: <this run>/report.html; requires --report-control)")
    ap.add_argument("--report-allow-config-diff", action="store_true",
                    help="allow an explicit profile-vs-profile report with different config hashes; "
                         "clips and metrics must still match")
    ap.add_argument("--report-allow-model-diff", action="store_true",
                    help="allow an explicit depth-model A/B report; clips and metrics must match")
    ap.add_argument("--report-allow-depth-step-diff", action="store_true",
                    help="allow an explicit current-depth versus reused-depth cadence report")
    ap.add_argument("--allow-build", action="store_true", help="proceed even if engines are missing")
    args = normalize_cli_paths(ap.parse_args())
    if args.comparison_only and args.update_baselines:
        fail("--comparison-only and --update-baselines are mutually exclusive")
    literal_bestv2 = "--literal-bestv2" in args.extra
    depth_override_root = ""
    if "--depth-override-root" in args.extra:
        index = len(args.extra) - 1 - args.extra[::-1].index("--depth-override-root")
        if index + 1 >= len(args.extra):
            fail("--depth-override-root needs a value")
        depth_override_root = args.extra[index + 1]
        depth_override_root = os.path.abspath(depth_override_root)
        # The harness runs with build_dir as cwd, so make this experimental artifact root
        # unambiguous before forwarding it.
        args.extra[index + 1] = depth_override_root
    depth_override_all = "--depth-override-all" in args.extra
    depth_compensation = ("external-treatment" if depth_override_all else
                          "external-reference" if depth_override_root else "none")
    try:
        depth_reuse_interval = int(extra_value(args.extra, "--depth-every", 1))
    except (TypeError, ValueError):
        fail("--depth-every must be an integer")
    if not 1 <= depth_reuse_interval <= 8:
        fail("--depth-every must be between 1 and 8")
    try:
        output_interval = int(extra_value(args.extra, "--output-every", 1))
    except (TypeError, ValueError):
        fail("--output-every must be an integer")
    if output_interval < 1:
        fail("--output-every must be at least 1")
    output_gt_right_only = "--output-gt-right-only" in args.extra
    depth_step = ("current-once" if depth_reuse_interval == 1 else
                  f"reuse-{depth_reuse_interval}")
    if literal_bestv2 and not args.comparison_only:
        fail("--literal-bestv2 is reference-only and requires --comparison-only")
    if depth_override_root and not args.comparison_only:
        fail("--depth-override-root is reference-only and requires --comparison-only")
    if depth_override_all and not depth_override_root:
        fail("--depth-override-all requires --depth-override-root")
    if depth_override_all and depth_reuse_interval != 1:
        fail("--depth-override-all requires --depth-every 1")
    if depth_override_root and depth_reuse_interval == 1 and not depth_override_all:
        fail("--depth-override-root requires --depth-every greater than 1")

    exe = os.path.join(args.build_dir, "sunshine.exe")
    default_clips, default_baselines = suite_defaults(args.suite)
    clips_dir = os.path.abspath(args.clips_root or default_clips)
    base_dir = os.path.abspath(args.baseline_dir or default_baselines)
    if args.update_baselines:
        if args.extra:
            fail("refusing to update committed baselines with experimental harness overrides; "
                 "put an intended production setting in bench.conf first")
        if clips_dir != os.path.abspath(default_clips) or \
                base_dir != os.path.abspath(default_baselines):
            fail("refusing to update committed baselines from an overridden clip/baseline root")
    thresholds = json.load(open(os.path.join(SCRIPT_DIR, "thresholds.json")))
    clips = args.clips or sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(clips_dir, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "frame_*.*")))
    if not clips:
        fail("no clips in " + clips_dir)
    depth_override_counts = (validate_depth_override_manifest(
        depth_override_root, clips_dir, clips, depth_reuse_interval, depth_override_all)
        if depth_override_root else {clip: 0 for clip in clips})
    if not args.update_baselines and not args.comparison_only:
        missing_baselines = [c for c in clips if not os.path.exists(os.path.join(base_dir, c + ".json"))]
        if missing_baselines:
            fail(f"missing committed baseline(s) in {base_dir}: {missing_baselines}. "
                 "Use --comparison-only for a matched A/B or --update-baselines after validation.")
    if not os.path.exists(exe):
        fail(f"{exe} not found -- build first (ninja -C cmake-build-relwithdebinfo sunshine)")
    require_current_build(args.build_dir)

    expected_config_profile = expected_profile(args.conf, args.extra)
    expected_ema = expected_profile_number(
        args.conf, expected_config_profile, "ema", 0.5, args.extra, "--ema")
    expected_ema_edge_change = expected_profile_number(
        args.conf, expected_config_profile, "ema_edge_change", 0.05, args.extra,
        "--ema-edge-change")
    expected_ema_edge_gradient = expected_profile_number(
        args.conf, expected_config_profile, "ema_edge_gradient", 0.02, args.extra,
        "--ema-edge-gradient")
    expected_ema_edge_strength = expected_profile_number(
        args.conf, expected_config_profile, "ema_edge_strength", 0.25, args.extra,
        "--ema-edge-strength")
    expected_minmax_ema = expected_profile_number(
        args.conf, expected_config_profile, "minmax_ema", 0.18, args.extra,
        "--minmax-ema")
    expected_subject_lock = expected_profile_number(
        args.conf, expected_config_profile, "subject_lock", 0.5, args.extra,
        "--subject-lock")
    expected_subject_recenter = expected_profile_number(
        args.conf, expected_config_profile, "subject_recenter", 0.35, args.extra,
        "--subject-recenter")
    expected_subject_stretch_value = expected_subject_stretch(
        args.conf, expected_config_profile, args.extra)
    expected_depth_short_side = expected_profile_number(
        args.conf, expected_config_profile, "depth_short_side", 432, args.extra,
        "--depth-short-side", int)
    expected_depth_max_aspect = expected_profile_number(
        args.conf, expected_config_profile, "depth_max_aspect", 4.0, [], "")
    expected_cuda_graph = expected_profile_bool(
        args.conf, expected_config_profile, "cuda_graph", True, args.extra,
        "--cuda-graph")
    expected_adaptive = expected_adaptive_pop(args.conf, expected_config_profile, args.extra)
    expected_adaptive_max = expected_profile_number(
        args.conf, expected_config_profile, "adaptive_pop_max", 1.30, args.extra,
        "--adaptive-pop-max")
    expected_pop = expected_profile_number(
        args.conf, expected_config_profile, "pop_strength", 1.25, args.extra,
        "--pop-strength")
    expected_adaptive_max = max(expected_adaptive_max, expected_pop)
    expected_zero_plane = expected_profile_string(
        args.conf, expected_config_profile, "zero_plane", "legacy", args.extra,
        "--zero-plane")
    expected_artistic_style = expected_profile_string(
        args.conf, expected_config_profile, "artistic_style", "immersive", [], "")
    expected_artistic_policy = True
    for flag in args.extra:
        if flag == "--artistic-policy":
            expected_artistic_policy = True
        elif flag == "--no-artistic-policy":
            expected_artistic_policy = False
    try:
        expected_artistic_scale_override = float(extra_value(
            args.extra, "--artistic-scale-override", 0.0
        ))
    except (TypeError, ValueError):
        fail("--artistic-scale-override must be a number")
    if (expected_artistic_scale_override != 0.0 and not
            0.5 <= expected_artistic_scale_override <= 1.5):
        fail("--artistic-scale-override must be between 0.5 and 1.5")
    if expected_zero_plane not in ("legacy", "subject", "median", "background"):
        fail(f"invalid zero_plane value: {expected_zero_plane!r}")
    if expected_artistic_style not in ("clean", "balanced", "immersive"):
        fail(f"invalid artistic_style value: {expected_artistic_style!r}")
    expected_model = expected_depth_model(args.conf, expected_config_profile, args.extra)
    missing = check_engines(args.build_dir, expected_model)
    if missing and not args.allow_build:
        print(f"run_eval: TRT engine(s) missing in {args.build_dir}/assets: {missing}\n"
              f"Build it by starting Apollo once, or pass --allow-build.")
        raise SystemExit(2)

    contention = sunshine_running()
    if contention:
        print("run_eval: WARNING another sunshine.exe is running -- perf numbers will be noisy "
              "(tagged gpu_contention in results.json; perf gate skipped).")
        if args.update_baselines:
            fail("refusing --update-baselines while another sunshine.exe is running; "
                 "close the live host so committed performance baselines are trustworthy")

    label = args.label or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.build_dir, "sbs_eval", label)
    os.makedirs(out_root, exist_ok=True)

    conf_sha = sha256_files([os.path.abspath(args.conf)])
    metric_sha = metric_contract_sha()
    meta = {
        "git_sha": git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(git(["status", "--porcelain"])),
        "run_kind": None,
        "baseline_identities": {},
        "clip_set_sha1": {c: sha1_dir(os.path.join(clips_dir, c)) for c in clips},
        "mode": "profile", "suite": args.suite, "clips_root": clips_dir,
        "extra_args": args.extra,
        "conf": os.path.relpath(args.conf, REPO),
        "model": expected_model, "profile": expected_config_profile,
        "adaptive_pop": expected_adaptive,
        "adaptive_pop_max": expected_adaptive_max,
        "pop_strength": expected_pop,
        "ema": expected_ema,
        "ema_edge_change": expected_ema_edge_change,
        "ema_edge_gradient": expected_ema_edge_gradient,
        "ema_edge_strength": expected_ema_edge_strength,
        "minmax_ema": expected_minmax_ema,
        "subject_lock": expected_subject_lock,
        "subject_recenter": expected_subject_recenter,
        "subject_stretch": expected_subject_stretch_value,
        "depth_short_side": expected_depth_short_side,
        "depth_max_aspect": expected_depth_max_aspect,
        "zero_plane": expected_zero_plane,
        "cuda_graph": expected_cuda_graph,
        "artistic_style": expected_artistic_style,
        "artistic_policy": expected_artistic_policy,
        "artistic_policy_consumed": None,
        "artistic_policy_authorization": None,
        "model_onnx_sha256": None,
        "policy_metadata_sha256": None,
        "deployment_geometry_allowlist_sha256": None,
        "artistic_scale_override": expected_artistic_scale_override,
        "output_interval": output_interval,
        "output_gt_right_only": output_gt_right_only,
        "literal_bestv2": literal_bestv2,
        "depth_compensation": depth_compensation,
        "eval_schema": EVAL_SCHEMA, "depth_step": depth_step,
        "depth_reuse_interval": depth_reuse_interval,
        "conf_sha256": conf_sha, "metric_sha256": metric_sha,
        "policy_warp_source_sha256": None,
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
        contract_path = os.path.join(out_dir, "contract.json")
        if not os.path.exists(contract_path):
            fail(f"{clip}: harness did not write contract.json")
        contract = json.load(open(contract_path, encoding="utf-8"))
        expected_contract = {
            "schema": HARNESS_SCHEMA,
            "model": expected_model,
            "profile": expected_config_profile,
            "depth_step": depth_step,
            "depth_reuse_interval": depth_reuse_interval,
            "depth_compensation": depth_compensation,
            "depth_override_frames": depth_override_counts[clip],
            "ema": expected_ema,
            "ema_edge_change": expected_ema_edge_change,
            "ema_edge_gradient": expected_ema_edge_gradient,
            "ema_edge_strength": expected_ema_edge_strength,
            "adaptive_pop": expected_adaptive,
            "adaptive_pop_max": expected_adaptive_max,
            "pop_strength": expected_pop,
            "minmax_ema": expected_minmax_ema,
            "subject_lock": expected_subject_lock,
            "subject_recenter": expected_subject_recenter,
            "subject_stretch": expected_subject_stretch_value,
            "depth_short_side": expected_depth_short_side,
            "depth_max_aspect": expected_depth_max_aspect,
            "zero_plane": expected_zero_plane,
            "artistic_style": expected_artistic_style,
            "artistic_policy": expected_artistic_policy,
            "artistic_scale_override": expected_artistic_scale_override,
            "output_interval": output_interval,
            "output_gt_right_only": output_gt_right_only,
            "literal_bestv2": literal_bestv2,
            "cuda_graph": expected_cuda_graph,
        }
        mismatched = {key: (expected, contract.get(key))
                      for key, expected in expected_contract.items()
                      if contract.get(key) != expected}
        if mismatched:
            fail(f"{clip}: harness contract mismatch: {mismatched}")
        policy_consumed = contract.get("artistic_policy_consumed")
        policy_authorization = contract.get("artistic_policy_authorization")
        model_onnx_sha256 = contract.get("model_onnx_sha256")
        policy_metadata_sha256 = contract.get("policy_metadata_sha256")
        geometry_allowlist_sha256 = contract.get(
            "deployment_geometry_allowlist_sha256"
        )
        if not isinstance(policy_consumed, bool):
            fail(f"{clip}: harness omitted artistic-policy consumption state")
        run_kind = ("comparison_only" if args.comparison_only else
                    "policy_candidate_gate" if policy_consumed else
                    "baseline_gate")
        if meta["run_kind"] is None:
            meta["run_kind"] = run_kind
        elif meta["run_kind"] != run_kind:
            fail(f"{clip}: run kind changed within one run")
        expected_authorization = "candidate-evaluation" if policy_consumed else "none"
        if policy_authorization != expected_authorization:
            fail(f"{clip}: artistic-policy authorization is invalid: "
                 f"{policy_authorization!r} != {expected_authorization!r}")
        for name, value in (("model_onnx_sha256", model_onnx_sha256),
                            ("policy_metadata_sha256", policy_metadata_sha256),
                            ("deployment_geometry_allowlist_sha256",
                             geometry_allowlist_sha256)):
            if not isinstance(value, str):
                fail(f"{clip}: harness has invalid {name}")
            if policy_consumed:
                if (len(value) != 64 or
                        any(char not in "0123456789abcdef" for char in value)):
                    fail(f"{clip}: consumed policy has invalid {name}")
            elif value:
                fail(f"{clip}: unconsumed policy unexpectedly records {name}")
        if policy_consumed and (not expected_artistic_policy or
                                expected_artistic_scale_override > 0.0):
            fail(f"{clip}: policy was consumed during an ablation/override run")
        for key, value in (
                ("artistic_policy_consumed", policy_consumed),
                ("artistic_policy_authorization", policy_authorization),
                ("model_onnx_sha256", model_onnx_sha256),
                ("policy_metadata_sha256", policy_metadata_sha256),
                ("deployment_geometry_allowlist_sha256",
                 geometry_allowlist_sha256)):
            if meta[key] is None:
                meta[key] = value
            elif meta[key] != value:
                fail(f"{clip}: {key} changed within one run")
        if contract.get("metric_sha256") != metric_sha:
            fail(f"{clip}: harness binary metric contract is stale: "
                 f"{contract.get('metric_sha256')} != {metric_sha}")
        warp_source_hash = contract.get("policy_warp_source_sha256")
        if (not isinstance(warp_source_hash, str) or len(warp_source_hash) != 64 or
                any(char not in "0123456789abcdef" for char in warp_source_hash)):
            fail(f"{clip}: invalid/missing policy warp source hash")
        if meta["policy_warp_source_sha256"] is None:
            meta["policy_warp_source_sha256"] = warp_source_hash
        elif meta["policy_warp_source_sha256"] != warp_source_hash:
            fail(f"{clip}: policy warp source hash changed within one run")
        geometry_fields = ("source_width", "source_height", "model_input_width",
                           "model_input_height", "eye_width", "eye_height",
                           "disparity_raster_width", "disparity_raster_height")
        if any(not isinstance(contract.get(key), int) or contract[key] <= 0
               for key in geometry_fields):
            fail(f"{clip}: invalid/missing harness raster geometry")
        for key in ("content_scale_x", "content_scale_y"):
            value = contract.get(key)
            if (not isinstance(value, (int, float)) or not math.isfinite(value) or
                    value <= 0.0 or value > 1.0):
                fail(f"{clip}: invalid {key}: {value}")
        if (contract["disparity_raster_width"] != contract["eye_width"] or
                contract["disparity_raster_height"] != contract["eye_height"]):
            fail(f"{clip}: exact disparity is not the complete output-eye raster")
        if contract.get("color_mode") not in {
                "sdr-srgb-8bit", "linear-sdr-fp16", "hdr-scrgb-fp16"}:
            fail(f"{clip}: invalid/missing input color mode")
        clip_meta = {"model": contract["model"], "profile": contract["profile"],
                     "metric_sha256": contract["metric_sha256"],
                     "depth_step": contract["depth_step"],
                     "depth_reuse_interval": contract["depth_reuse_interval"],
                     "depth_compensation": contract["depth_compensation"],
                     "literal_bestv2": contract["literal_bestv2"],
                     "cuda_graph": contract["cuda_graph"],
                     "adaptive_pop": contract["adaptive_pop"],
                     "adaptive_pop_max": contract["adaptive_pop_max"],
                     "pop_strength": contract["pop_strength"],
                     "ema": contract["ema"],
                     "ema_edge_change": contract["ema_edge_change"],
                     "ema_edge_gradient": contract["ema_edge_gradient"],
                     "ema_edge_strength": contract["ema_edge_strength"],
                     "minmax_ema": contract["minmax_ema"],
                     "subject_lock": contract["subject_lock"],
                     "subject_recenter": contract["subject_recenter"],
                     "subject_stretch": contract["subject_stretch"],
                     "depth_short_side": contract["depth_short_side"],
                     "depth_max_aspect": contract["depth_max_aspect"],
                     "zero_plane": contract["zero_plane"],
                     "artistic_style": contract["artistic_style"],
                     "artistic_policy": contract["artistic_policy"],
                     "artistic_policy_consumed": policy_consumed,
                     "artistic_policy_authorization": policy_authorization,
                     "model_onnx_sha256": model_onnx_sha256,
                     "policy_metadata_sha256": policy_metadata_sha256,
                     "deployment_geometry_allowlist_sha256":
                         geometry_allowlist_sha256,
                     "artistic_scale_override": contract["artistic_scale_override"],
                     "output_interval": contract["output_interval"],
                     "output_gt_right_only": contract["output_gt_right_only"],
                     "depth_override_frames": contract["depth_override_frames"],
                     "policy_warp_source_sha256": warp_source_hash,
                     "harness_schema": contract["schema"],
                     "source_width": contract["source_width"],
                     "source_height": contract["source_height"],
                     "model_input_width": contract["model_input_width"],
                     "model_input_height": contract["model_input_height"],
                     "eye_width": contract["eye_width"],
                     "eye_height": contract["eye_height"],
                     "color_mode": contract["color_mode"],
                     "content_scale_x": contract["content_scale_x"],
                     "content_scale_y": contract["content_scale_y"],
                     "disparity_raster_width": contract["disparity_raster_width"],
                     "disparity_raster_height": contract["disparity_raster_height"],
                     "artifact_mode": contract["artifact_mode"],
                     "warp_disparity": contract["warp_disparity"],
                     "warp_unclamped_disparity": contract["warp_unclamped_disparity"],
                     "artistic_disparity_contract": contract["artistic_disparity_contract"],
                     "artistic_full_clamp_abs": contract["artistic_full_clamp_abs"],
                     "warp_mask": contract["warp_mask"],
                     "cuda_graph_captured": contract.get("cuda_graph_captured", False)}

        # A valid harness result has one source, raw-model, warp-input depth, and SBS artifact for
        # every numeric frame identity. This catches dropped/renumbered outputs before metrics run.
        source_by_id = sbsbench.indexed_files(
            os.path.join(clip_dir, "frame_*.*"), "frame_")
        ordered_source_ids = sorted(source_by_id)
        expected_output_ids = set(ordered_source_ids[::output_interval])
        if output_gt_right_only:
            gt_right_ids = set(sbsbench.indexed_files(
                os.path.join(clip_dir, "gt_right", "frame_*.*"), "frame_"))
            expected_output_ids &= gt_right_ids
        sbs_by_id = sbsbench.indexed_files(os.path.join(out_dir, "sbs_*.png"), "sbs_")
        sbs_ids = set(sbs_by_id)
        depth_ids = set(sbsbench.indexed_files(os.path.join(out_dir, "depth_*.png"), "depth_"))
        raw_ids = set(sbsbench.indexed_files(os.path.join(out_dir, "raw_*.f32"), "raw_"))
        mask_ids = set(sbsbench.indexed_files(
            os.path.join(out_dir, "warp_mask_*.png"), "warp_mask_"))
        disparity_ids = set(sbsbench.indexed_files(
            os.path.join(out_dir, "warp_disparity_*.f32"), "warp_disparity_"))
        unclamped_disparity_ids = set(sbsbench.indexed_files(
            os.path.join(out_dir, "warp_unclamped_disparity_*.f32"),
            "warp_unclamped_disparity_"))
        ema_mask_ids = set(sbsbench.indexed_files(
            os.path.join(out_dir, "ema_mask_*.png"), "ema_mask_"))
        if (contract.get("warp_mask") != {
                "red": "forward_disocclusion_before_fill"}):
            fail(f"{clip}: missing/unknown warp-mask channel contract")
        if (contract.get("warp_disparity") !=
                "exact_clamped_full_binocular_normalized_at_output_eye_raster_zero_bars"):
            fail(f"{clip}: missing/unknown exact warp-disparity contract")
        if (contract.get("warp_unclamped_disparity") !=
                "unclamped_full_binocular_normalized_at_artistic_scale_1_"
                "output_eye_raster_zero_bars"):
            fail(f"{clip}: missing/unknown unclamped warp-disparity contract")
        if (contract.get("artistic_disparity_contract") !=
                "clamp(raw_baseline_times_scale_to_plus_or_minus_0.142_times_"
                "aspect_scale_times_content_scale_x)"):
            fail(f"{clip}: missing/unknown artistic disparity contract")
        if (not expected_output_ids or expected_output_ids != sbs_ids
                or expected_output_ids != depth_ids or expected_output_ids != raw_ids
                or expected_output_ids != mask_ids or expected_output_ids != disparity_ids
                or expected_output_ids != unclamped_disparity_ids):
            fail(f"{clip}: sampled artifact frame-id mismatch "
                 f"expected={sorted(expected_output_ids)} "
                 f"sbs={sorted(sbs_ids)} depth={sorted(depth_ids)} raw={sorted(raw_ids)} "
                 f"warp_mask={sorted(mask_ids)} warp_disparity={sorted(disparity_ids)} "
                 f"warp_unclamped_disparity={sorted(unclamped_disparity_ids)}")
        first_sbs = sbsbench.load_gray(sbs_by_id[min(sbs_ids)])
        if (first_sbs.shape[0] != contract["eye_height"] or
                first_sbs.shape[1] != 2 * contract["eye_width"]):
            fail(f"{clip}: contract eye geometry does not match rendered SBS artifacts")
        first_source = sbsbench.load_gray(source_by_id[ordered_source_ids[0]])
        if (first_source.shape[0] != contract["source_height"] or
                first_source.shape[1] != contract["source_width"]):
            fail(f"{clip}: contract source geometry does not match source artifacts")
        if expected_ema_edge_change > 0.0 and ema_mask_ids != expected_output_ids:
            fail(f"{clip}: incomplete EMA motion-mask artifacts: {sorted(ema_mask_ids)}")
        if expected_ema_edge_change <= 0.0 and ema_mask_ids:
            fail(f"{clip}: unexpected EMA motion-mask artifacts while feature is disabled")
        if not os.path.exists(os.path.join(out_dir, "raw_shape.json")):
            fail(f"{clip}: raw_shape.json missing")
        # Carry the clip's own metadata (scene name/description) into results so the run dir is
        # self-describing and the report can label clips without the source clips dir.
        cmp_path = os.path.join(clip_dir, "meta.json")
        if os.path.exists(cmp_path):
            try:
                clip_meta.update({k: v for k, v in json.load(open(cmp_path)).items()
                                  if k in ("name", "description", "expected_flat", "gt_depth_kind",
                                           "required_gt_depth", "required_gt_flow",
                                           "required_gt_stereo",
                                           "dataset", "homepage", "citation", "license_note", "suite",
                                           "content_type", "source_url", "source_window",
                                           "source_artifacts")})
            except Exception:
                pass

        print(f"[{clip}] scoring...", flush=True)
        try:
            rows, agg = sbsbench.measure_sequence(
                out_dir, clip_dir, expected_flat=bool(clip_meta.get("expected_flat")))
        except ValueError as exc:
            fail(f"{clip}: {exc}")
        missing_required = missing_required_metric_evidence(agg, thresholds, clip_meta)
        if missing_required:
            fail(f"{clip}: required primary metric evidence is missing: {missing_required}")
        perf = {}
        perf_p = os.path.join(out_dir, "sbs_perf.json")
        if os.path.exists(perf_p):
            stages = json.load(open(perf_p)).get("stages", {})
            perf = {k: v.get("p50_ms", 0) for k, v in stages.items()}

        worst, clip_issues, clip_hard_failures = score_clip_gates(
            rows, agg, thresholds, clip_meta)
        issues.extend({"clip": clip, **item} for item in clip_issues)
        hard_failures.extend({"clip": clip, **item} for item in clip_hard_failures)

        clip_meta["clip_sha1"] = meta["clip_set_sha1"][clip]
        entry = {"aggregate": agg, "perf_ms": perf, "meta": clip_meta, "worst_frame": worst}
        results[clip] = entry

        # Regression gate vs baseline. A baseline is only valid for the exact frames it was made
        # from: if the clip content changed, gating against it is meaningless -- skip it loudly
        # instead of silently comparing apples to oranges.
        bp = os.path.join(base_dir, clip + ".json")
        if os.path.exists(bp) and not args.update_baselines and not args.comparison_only:
            base = json.load(open(bp))
            base_meta = base.get("meta", {})
            current_context = {
                **meta, **clip_meta,
                "clip_sha1": meta["clip_set_sha1"][clip],
            }
            context_fields = BASELINE_CONTEXT_FIELDS
            if meta["run_kind"] == "policy_candidate_gate":
                context_fields = tuple(
                    key for key in context_fields
                    if key not in POLICY_CANDIDATE_TREATMENT_FIELDS
                )
                meta["baseline_identities"][clip] = sha256_file(bp)
            required = {key: current_context.get(key) for key in context_fields}
            mismatches = {k: (base_meta.get(k), v) for k, v in required.items()
                          if base_meta.get(k) != v}
            if mismatches:
                fail(f"{clip}: baseline context is stale/incompatible: {mismatches}. "
                     "Re-run with --update-baselines only after verifying the new eval contract.")
            for k, spec in thresholds["metrics"].items():
                if metric_exempt_for_clip(spec, clip_meta):
                    continue
                if spec.get("role") == "hard":
                    continue  # absolute hard constraints were evaluated above, independent of baseline
                b, n = base["aggregate"].get(k), agg.get(k)
                if b is None:
                    continue
                if n is None or not isinstance(n, (int, float)) or not math.isfinite(n):
                    regressions.append({"clip": clip, "metric": k,
                                        "baseline": round(b, 3), "value": None,
                                        "reason": "missing-treatment-evidence"})
                    continue
                if sbsbench.metric_gate_failed(b, n, spec):
                    regressions.append({"clip": clip, "metric": k, "baseline": round(b, 3),
                                        **worst.get(k, {}), "value": round(n, 3)})
            if not contention:
                for k, spec in thresholds["perf_ms"].items():
                    b, n = base.get("perf_ms", {}).get(k), perf.get(k)
                    if (isinstance(b, (int, float)) and math.isfinite(b) and b > 0.0 and
                            (not isinstance(n, (int, float)) or not math.isfinite(n) or
                             n <= 0.0)):
                        regressions.append({"clip": clip, "metric": "perf:" + k,
                                            "baseline": round(b, 2), "value": None,
                                            "reason": "missing-treatment-evidence"})
                    elif (isinstance(b, (int, float)) and math.isfinite(b) and b > 0.0 and
                          isinstance(n, (int, float)) and math.isfinite(n) and
                          (n - b) > max(spec["abs_floor"], b * spec["rel_tol"])):
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
        if args.report_allow_model_diff:
            report_cmd.append("--allow-model-diff")
        if args.report_allow_depth_step_diff:
            report_cmd.append("--allow-depth-step-diff")
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
