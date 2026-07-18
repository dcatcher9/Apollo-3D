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
import os
import re
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
import sbsbench  # noqa: E402  (metric implementations)

EVAL_SCHEMA = 26  # runtime-pipeline provenance + fail-closed evidence; harness contract 15


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


def metric_contract_sha():
    """Hash automatic metric implementation and thresholds.

    Runner/gating semantics are versioned separately by EVAL_SCHEMA. Hashing this entire file made
    comments and diagnostic wording invalidate otherwise-identical committed baselines.
    """
    return sha256_files([os.path.join(SCRIPT_DIR, "sbsbench.py"),
                         os.path.join(SCRIPT_DIR, "thresholds.json")])


def load_clip_metadata(path, suite=None, required=True):
    """Load and validate clip metadata before any GPU work starts.

    Extended-suite clips are authenticated public-data evidence, so they must declare their
    dataset and at least one required reference contract.  A missing or malformed declaration
    must never turn an extended baseline update into an unlabelled ordinary image test.
    """
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        if required:
            raise ValueError(f"missing clip metadata {meta_path}")
        return {}
    try:
        with open(meta_path, encoding="utf-8") as meta_file:
            meta = json.load(meta_file)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid clip metadata {meta_path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"invalid clip metadata {meta_path}: root must be an object")

    requirement_keys = ("required_gt_depth", "required_gt_flow", "required_gt_stereo")
    for key in requirement_keys:
        if key in meta and not isinstance(meta[key], bool):
            raise ValueError(f"invalid clip metadata {meta_path}: {key} must be boolean")
    if suite == "extended":
        if not isinstance(meta.get("dataset"), str) or not meta["dataset"].strip():
            raise ValueError(f"invalid extended clip metadata {meta_path}: dataset is required")
        if not any(meta.get(key) is True for key in requirement_keys):
            raise ValueError(
                f"invalid extended clip metadata {meta_path}: at least one authenticated "
                "GT evidence requirement must be true")
        if meta.get("required_gt_depth") and meta.get("gt_depth_kind") not in {
                "disparity", "metric", "depth"}:
            raise ValueError(
                f"invalid extended clip metadata {meta_path}: required GT depth needs "
                "gt_depth_kind=disparity, metric, or depth")
    reference_patterns = {
        "required_gt_depth": os.path.join(path, "gt_depth", "frame_*.*"),
        "required_gt_flow": os.path.join(path, "gt_flow", "frame_*.npz"),
        "required_gt_stereo": os.path.join(path, "gt_right", "frame_*.*"),
    }
    for key, pattern in reference_patterns.items():
        if meta.get(key) and not glob.glob(pattern):
            raise ValueError(
                f"invalid clip metadata {meta_path}: {key} is true but no matching "
                "reference sidecars exist")
    return meta


def published_clip_metadata(source_meta):
    """Select reportable source metadata without colliding with evaluator provenance."""
    keys = (
        "name", "description", "expected_flat", "gt_depth_kind", "required_gt_depth",
        "required_gt_flow", "required_gt_stereo", "dataset", "homepage", "citation",
        "license_note", "content_type", "source_url", "source_window", "source_artifacts",
    )
    published = {key: source_meta[key] for key in keys if key in source_meta}
    if "suite" in source_meta:
        published["source_suite"] = source_meta["suite"]
    return published


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
    meta = load_clip_metadata(path, required=False)
    semantic = {k: meta[k] for k in ("expected_flat", "gt_depth_kind", "dataset",
                                     "required_gt_depth", "required_gt_flow",
                                     "required_gt_stereo") if k in meta}
    h.update(json.dumps(semantic, sort_keys=True).encode())
    return h.hexdigest()[:12]


def preflight_baselines(base_dir, clips, required_common, clip_hashes):
    """Validate every committed baseline before starting a harness subprocess."""
    baselines = {}
    for clip in clips:
        path = os.path.join(base_dir, clip + ".json")
        try:
            with open(path, encoding="utf-8") as baseline_file:
                baseline = json.load(baseline_file)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load committed baseline {path}: {exc}") from exc
        if not isinstance(baseline, dict):
            raise ValueError(f"committed baseline {path} must be an object")
        meta = baseline.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"committed baseline {path} has no metadata object")
        if not isinstance(baseline.get("aggregate"), dict):
            raise ValueError(f"committed baseline {path} has no aggregate object")
        if not isinstance(baseline.get("perf_ms"), dict):
            raise ValueError(f"committed baseline {path} has no perf_ms object")
        if meta.get("extra_args") != []:
            raise ValueError(
                f"{clip}: committed baseline is non-canonical: "
                f"extra_args={meta.get('extra_args')!r}; regenerate without --extra")
        required = {**required_common, "clip_sha1": clip_hashes[clip]}
        mismatches = {key: (meta.get(key), value) for key, value in required.items()
                      if meta.get(key) != value}
        if mismatches:
            raise ValueError(
                f"{clip}: baseline context is stale/incompatible: {mismatches}. "
                "Re-run with --update-baselines only after verifying the new eval contract.")
        baselines[clip] = baseline
    return baselines


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
        if spec.get("role") == "hard" and not sbsbench.metric_value_valid(agg, metric):
            hard_failures.append({"metric": metric, "value": None, "missing": True,
                                  "hard_min": spec.get("hard_min"),
                                  "hard_max": spec.get("hard_max")})
            continue
        frame_key = metric if any(metric in row for row in rows) else (
            metric[:-4] if metric.endswith(("_p50", "_p95")) else metric)
        values = [(row.get(frame_key), row.get("_frame_id", i))
                  for i, row in enumerate(rows) if frame_key in row]
        if values:
            choose = min if spec.get("better") == "higher" else max
            value, frame = choose(values)
            worst[metric] = {"frame": frame, "worst_value": round(value, 3)}
        if "trigger" in spec and agg.get(metric, 0) > spec["trigger"]:
            issues.append({"metric": metric, "trigger": spec["trigger"],
                           **worst.get(metric, {}), "value": round(agg[metric], 3)})
        if "trigger_min" in spec and metric in agg and agg[metric] < spec["trigger_min"]:
            issues.append({"metric": metric, "trigger_min": spec["trigger_min"],
                           **worst.get(metric, {}), "value": round(agg[metric], 3)})
        if spec.get("role") == "hard" and sbsbench.metric_value_valid(agg, metric):
            if sbsbench.metric_gate_failed(agg[metric], agg[metric], spec):
                hard_failures.append({"metric": metric, **worst.get(metric, {}),
                                      "value": round(agg[metric], 3),
                                      "hard_min": spec.get("hard_min"),
                                      "hard_max": spec.get("hard_max")})
    return worst, issues, hard_failures


def primary_evidence_failures(current, thresholds, clip, clip_meta, baseline=None, worst=None):
    """Fail closed when an applicable primary metric is unavailable for a decision.

    `requires` in thresholds.json makes conditional evidence explicit.  A baseline comparison
    requires the metric on both sides; a fresh comparison/baseline update requires it in the
    current run.  This prevents a metric implementation failure from becoming a neutral result.
    """
    worst = worst or {}
    failures = []
    for metric, spec in thresholds["metrics"].items():
        if spec.get("role") != "primary" or metric_exempt_for_clip(spec, clip_meta):
            continue
        observed = sbsbench.combined_metric_evidence(baseline, current)
        if not sbsbench.metric_evidence_applicable(metric, spec, observed, clip_meta):
            continue
        missing = []
        if baseline is not None and not sbsbench.metric_value_valid(baseline, metric):
            missing.append("baseline")
        if not sbsbench.metric_value_valid(current, metric):
            missing.append("current")
        if not missing:
            continue
        item = {"clip": clip, "metric": metric, "missing": True,
                "source": "+".join(missing), **worst.get(metric, {})}
        if baseline is not None and sbsbench.metric_value_valid(baseline, metric):
            item["baseline"] = round(baseline[metric], 3)
        failures.append(item)
    return failures


def perf_evidence_failures(base, current, thresholds, clip):
    """Return missing/invalid performance evidence before comparing numeric regressions."""
    failures = []
    for metric in thresholds["perf_ms"]:
        baseline_value = base.get(metric) if base is not None else None
        current_value = current.get(metric)
        baseline_valid = base is not None and sbsbench.metric_value_valid(base, metric)
        current_valid = sbsbench.metric_value_valid(current, metric)
        if base is not None and not baseline_valid:
            failures.append({"clip": clip, "metric": "perf:" + metric,
                             "missing": True, "source": "baseline"})
        if not current_valid:
            item = {"clip": clip, "metric": "perf:" + metric,
                    "missing": True, "source": "current"}
            if baseline_valid:
                item["baseline"] = round(baseline_value, 3)
            failures.append(item)
    return failures


def load_perf_metrics(path):
    """Load harness medians without inventing zero-cost samples for missing evidence."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as perf_file:
            payload = json.load(perf_file)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load performance evidence {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("stages"), dict):
        raise ValueError(f"invalid performance evidence {path}: stages must be an object")
    return {name: stage.get("p50_ms") if isinstance(stage, dict) else None
            for name, stage in payload["stages"].items()}


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
    build_env = os.environ.copy()
    if os.name == "nt":
        # The configured Ninja belongs to the MSYS2 UCRT64 toolchain.  A plain PowerShell/Python
        # process does not necessarily have the compiler DLLs (or the web-ui Node runtime) on
        # PATH, so a no-op build succeeds while the same command fails as soon as one source file
        # is stale.  Recreate the documented non-interactive build environment here; the eval gate
        # must not depend on whichever shell happened to launch it.
        ninja_dir = os.path.dirname(os.path.abspath(ninja))
        msys_root = os.path.dirname(os.path.dirname(ninja_dir))
        path_entries = [ninja_dir, os.path.join(msys_root, "usr", "bin")]
        node_dir = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "nodejs")
        if os.path.isdir(node_dir):
            path_entries.insert(0, node_dir)
        path_entries.append(build_env.get("PATH", ""))
        build_env["PATH"] = os.pathsep.join(entry for entry in path_entries if entry)
        build_env["MSYSTEM"] = "UCRT64"
        build_env["MSYS2_PATH_TYPE"] = "inherit"
    try:
        probe = subprocess.run(
            [ninja, "-C", build_dir, "sunshine"],
            capture_output=True, text=True, timeout=900, env=build_env)
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


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_shader_sha256(build_dir):
    """Hash every runtime-compiled DirectX shader with stable relative paths and EOLs.

    Windows development builds expose the source shader tree through an assets junction. Those
    files can therefore change evaluated pixels without changing sunshine.exe; executable
    provenance alone is insufficient for a matched A/B decision.
    """
    shader_root = os.path.join(build_dir, "assets", "shaders", "directx")
    paths = sorted(glob.glob(os.path.join(shader_root, "**", "*.hlsl"), recursive=True))
    if not paths:
        raise ValueError(f"no runtime DirectX shaders found below {shader_root}")
    digest = hashlib.sha256()
    for path in paths:
        relative = os.path.relpath(path, shader_root).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        with open(path, "rb") as shader_file:
            data = shader_file.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(data)
    return digest.hexdigest()


def engine_provenance(build_dir, model):
    """Return hashes for the exact model artifacts validated by :func:`check_engines`."""
    assets = os.path.join(build_dir, "assets")
    manifest_path = os.path.join(assets, model + ".active-engine.json")
    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    engine_name = manifest["engine"]
    engine_path = os.path.join(assets, engine_name)
    onnx_path = os.path.join(assets, model + ".onnx")
    actual_onnx_sha = file_sha256(onnx_path)
    if manifest.get("onnx_sha256") != actual_onnx_sha:
        raise ValueError("active engine manifest changed after validation")
    return {
        "engine_name": engine_name,
        "engine_sha256": file_sha256(engine_path),
        "onnx_sha256": actual_onnx_sha,
    }


def check_engines(build_dir, model):
    """Validate the runtime-published identity of the exact compatible TensorRT artifact.

    TensorRT plans are specific to the builder recipe, runtime ABI, CUDA device and ONNX bytes.
    A glob for ``<model>*.engine`` can therefore bless an engine the current executable will never
    load.  The runtime publishes this manifest only after resolving that full identity.
    """
    assets = os.path.join(build_dir, "assets")
    exe = os.path.join(build_dir, "sunshine.exe")
    manifest_path = os.path.join(assets, model + ".active-engine.json")
    issues = []
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        return [f"{os.path.basename(manifest_path)}: missing/invalid manifest ({exc})"]
    if manifest.get("schema") != 1:
        issues.append(f"manifest schema {manifest.get('schema')!r} != 1")
    if manifest.get("model") != model:
        issues.append(f"manifest model {manifest.get('model')!r} != {model!r}")
    engine_name = manifest.get("engine")
    if (not isinstance(engine_name, str) or not engine_name or
            os.path.basename(engine_name) != engine_name):
        issues.append("manifest engine must be one filename")
        engine_name = None
    onnx_path = os.path.join(assets, model + ".onnx")
    recorded_sha = manifest.get("onnx_sha256")
    try:
        actual_sha = file_sha256(onnx_path)
    except OSError as exc:
        issues.append(f"ONNX source unavailable: {exc}")
    else:
        if recorded_sha != actual_sha:
            issues.append("manifest ONNX SHA-256 does not match current source")
    if engine_name:
        engine_path = os.path.join(assets, engine_name)
        try:
            engine_ready = os.path.isfile(engine_path) and os.path.getsize(engine_path) > 0
        except OSError:
            engine_ready = False
        if not engine_ready:
            issues.append(f"exact engine missing/empty: {engine_name}")
        else:
            try:
                if os.path.getmtime(engine_path) > os.path.getmtime(manifest_path):
                    issues.append("exact engine changed after manifest publication")
            except OSError as exc:
                issues.append(f"cannot compare engine/manifest timestamps: {exc}")
    try:
        if os.path.getmtime(manifest_path) < os.path.getmtime(exe):
            issues.append("manifest predates sunshine.exe")
    except OSError as exc:
        issues.append(f"cannot compare manifest/executable timestamps: {exc}")
    return issues


def run_engine_preflight(exe, conf, build_dir, frames_dir, model):
    """Build/resolve an engine outside measured clips, then require a fresh exact manifest."""
    with tempfile.TemporaryDirectory(prefix="sbs-engine-preflight-", dir=build_dir) as out_dir:
        cmd = [exe, os.path.abspath(conf), "--sbs-bench", "--frames", frames_dir,
               "--out", out_dir, "--model", model, "--limit", "1"]
        try:
            result = subprocess.run(
                cmd, cwd=build_dir, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            fail("untimed TensorRT engine preflight timed out")
        if result.returncode != 0:
            print((result.stdout + result.stderr)[-2000:])
            fail(f"untimed TensorRT engine preflight failed (exit {result.returncode})")
    issues = check_engines(build_dir, model)
    if issues:
        fail("runtime did not publish a valid exact-engine manifest after preflight: " +
             "; ".join(issues))


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
    ap.add_argument("--report-allow-executable-diff", action="store_true",
                    help="allow an explicit old-code versus new-code report (binary and HLSL)")
    ap.add_argument("--allow-build", action="store_true", help="proceed even if engines are missing")
    args = normalize_cli_paths(ap.parse_args())
    if args.comparison_only and args.update_baselines:
        fail("--comparison-only and --update-baselines are mutually exclusive")
    if args.update_baselines and args.extra:
        fail("--update-baselines requires the canonical profile/config with no --extra overrides; "
             "move an accepted setting into bench.conf or production defaults first")
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
    thresholds = json.load(open(os.path.join(SCRIPT_DIR, "thresholds.json")))
    clips = args.clips or sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(clips_dir, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "frame_*.*")))
    if not clips:
        fail("no clips in " + clips_dir)
    try:
        source_clip_meta = {
            clip: load_clip_metadata(os.path.join(clips_dir, clip), suite=args.suite)
            for clip in clips
        }
    except ValueError as exc:
        fail(str(exc))
    depth_override_counts = (validate_depth_override_manifest(
        depth_override_root, clips_dir, clips, depth_reuse_interval, depth_override_all)
        if depth_override_root else {clip: 0 for clip in clips})
    if not args.update_baselines and not args.comparison_only:
        missing_baselines = [c for c in clips if not os.path.exists(os.path.join(base_dir, c + ".json"))]
        if missing_baselines:
            fail(f"missing committed baseline(s) in {base_dir}: {missing_baselines}. "
                 "Use --comparison-only for a matched A/B or --update-baselines after validation.")
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
    if expected_zero_plane not in ("legacy", "subject", "median", "background"):
        fail(f"invalid zero_plane value: {expected_zero_plane!r}")
    expected_model = expected_depth_model(args.conf, expected_config_profile, args.extra)

    conf_sha = sha256_files([os.path.abspath(args.conf)])
    metric_sha = metric_contract_sha()
    try:
        clip_hashes = {clip: sha1_dir(os.path.join(clips_dir, clip)) for clip in clips}
    except ValueError as exc:
        fail(str(exc))
    baseline_manifests = {}
    if not args.update_baselines and not args.comparison_only:
        required_baseline_context = {
            "mode": "profile",
            "run_kind": "baseline-update",
            "suite": args.suite,
            "model": expected_model,
            "profile": expected_config_profile,
            "eval_schema": EVAL_SCHEMA,
            "depth_step": depth_step,
            "depth_compensation": depth_compensation,
            "conf_sha256": conf_sha,
            "metric_sha256": metric_sha,
        }
        try:
            baseline_manifests = preflight_baselines(
                base_dir, clips, required_baseline_context, clip_hashes)
        except ValueError as exc:
            fail(str(exc))

    # Baseline and source-contract failures are intentionally resolved before checking/building
    # engines: a stale evaluator must not spend GPU time merely to discover it cannot decide.
    if not os.path.exists(exe):
        fail(f"{exe} not found -- build first (ninja -C cmake-build-relwithdebinfo sunshine)")
    require_current_build(args.build_dir)
    engine_issues = check_engines(args.build_dir, expected_model)
    if engine_issues and not args.allow_build:
        print(f"run_eval: exact TRT engine is not ready in {args.build_dir}/assets:\n  " +
              "\n  ".join(engine_issues) +
              "\nBuild it by starting this Apollo binary once, or pass --allow-build for an "
              "untimed one-frame preflight.")
        raise SystemExit(2)
    if engine_issues:
        print("run_eval: exact TRT engine is not ready; running one untimed preflight...",
              flush=True)
        run_engine_preflight(
            exe, args.conf, args.build_dir, os.path.join(clips_dir, clips[0]), expected_model)
    try:
        shader_sha = runtime_shader_sha256(args.build_dir)
        model_artifacts = engine_provenance(args.build_dir, expected_model)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        fail(f"cannot record exact runtime-pipeline provenance: {exc}")

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

    meta = {
        "git_sha": git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(git(["status", "--porcelain"])),
        "clip_set_sha1": clip_hashes,
        "mode": "profile", "suite": args.suite, "clips_root": clips_dir,
        "extra_args": args.extra,
        "conf": os.path.relpath(args.conf, REPO),
        "model": expected_model, "profile": expected_config_profile,
        "adaptive_pop": expected_adaptive,
        "adaptive_pop_max": expected_adaptive_max,
        "zero_plane": expected_zero_plane,
        "literal_bestv2": literal_bestv2,
        "depth_compensation": depth_compensation,
        "eval_schema": EVAL_SCHEMA, "depth_step": depth_step,
        "depth_reuse_interval": depth_reuse_interval,
        "conf_sha256": conf_sha, "metric_sha256": metric_sha,
        "executable_sha256": file_sha256(exe),
        "runtime_shader_sha256": shader_sha,
        **model_artifacts,
        "gpu_contention": contention,
        "run_kind": ("baseline-update" if args.update_baselines else
                     "comparison-only" if args.comparison_only else "baseline-gated"),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "run_name": label,
    }

    results, regressions, issues, hard_failures = {}, [], [], []
    evidence_failures, baseline_updates = [], {}
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
            "schema": 15,
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
            "zero_plane": expected_zero_plane,
            "literal_bestv2": literal_bestv2,
            "cuda_graph": expected_cuda_graph,
        }
        mismatched = {key: (expected, contract.get(key))
                      for key, expected in expected_contract.items()
                      if contract.get(key) != expected}
        if mismatched:
            fail(f"{clip}: harness contract mismatch: {mismatched}")
        clip_meta = {"model": contract["model"], "profile": contract["profile"],
                     "depth_compensation": contract["depth_compensation"],
                     "literal_bestv2": contract["literal_bestv2"],
                     "cuda_graph": contract["cuda_graph"],
                     "adaptive_pop": contract["adaptive_pop"],
                     "adaptive_pop_max": contract["adaptive_pop_max"],
                     "zero_plane": contract["zero_plane"],
                     "cuda_graph_captured": contract.get("cuda_graph_captured", False)}

        # A valid harness result has one source, raw-model, warp-input depth, and SBS artifact for
        # every numeric frame identity. This catches dropped/renumbered outputs before metrics run.
        source_ids = set(sbsbench.indexed_files(os.path.join(clip_dir, "frame_*.*"), "frame_"))
        clip_meta["source_frame_count"] = len(source_ids)
        sbs_ids = set(sbsbench.indexed_files(os.path.join(out_dir, "sbs_*.png"), "sbs_"))
        depth_ids = set(sbsbench.indexed_files(os.path.join(out_dir, "depth_*.png"), "depth_"))
        raw_ids = set(sbsbench.indexed_files(os.path.join(out_dir, "raw_*.f32"), "raw_"))
        mask_ids = set(sbsbench.indexed_files(
            os.path.join(out_dir, "warp_mask_*.png"), "warp_mask_"))
        ema_mask_ids = set(sbsbench.indexed_files(
            os.path.join(out_dir, "ema_mask_*.png"), "ema_mask_"))
        if (contract.get("warp_mask") != {
                "red": "forward_disocclusion_before_fill"}):
            fail(f"{clip}: missing/unknown warp-mask channel contract")
        if (not source_ids or source_ids != sbs_ids or source_ids != depth_ids
                or source_ids != raw_ids or source_ids != mask_ids):
            fail(f"{clip}: artifact frame-id mismatch source={sorted(source_ids)} "
                 f"sbs={sorted(sbs_ids)} depth={sorted(depth_ids)} raw={sorted(raw_ids)} "
                 f"warp_mask={sorted(mask_ids)}")
        if expected_ema_edge_change > 0.0 and ema_mask_ids != source_ids:
            fail(f"{clip}: incomplete EMA motion-mask artifacts: {sorted(ema_mask_ids)}")
        if expected_ema_edge_change <= 0.0 and ema_mask_ids:
            fail(f"{clip}: unexpected EMA motion-mask artifacts while feature is disabled")
        if not os.path.exists(os.path.join(out_dir, "raw_shape.json")):
            fail(f"{clip}: raw_shape.json missing")
        # Carry the clip's own metadata (scene name/description) into results so the run dir is
        # self-describing and the report can label clips without the source clips dir.
        clip_meta.update(published_clip_metadata(source_clip_meta[clip]))
        # `suite` is evaluator provenance ("core" or "extended") and is used to select the
        # committed baseline namespace. Prepared-dataset manifests may also carry their own suite
        # revision (for example "extended-v3"); keep that useful label under a distinct key so it
        # cannot overwrite the run contract when baseline metadata is merged below.

        print(f"[{clip}] scoring...", flush=True)
        try:
            rows, agg = sbsbench.measure_sequence(
                out_dir, clip_dir, expected_flat=bool(clip_meta.get("expected_flat")))
        except ValueError as exc:
            fail(f"{clip}: {exc}")
        perf_p = os.path.join(out_dir, "sbs_perf.json")
        try:
            perf = load_perf_metrics(perf_p)
        except ValueError as exc:
            fail(f"{clip}: {exc}")
        if args.comparison_only:
            evidence_failures.extend(primary_evidence_failures(
                agg, thresholds, clip, clip_meta, worst=None))
            evidence_failures.extend(perf_evidence_failures(None, perf, thresholds, clip))

        worst, clip_issues, clip_hard_failures = score_clip_gates(
            rows, agg, thresholds, clip_meta)
        issues.extend({"clip": clip, **item} for item in clip_issues)
        hard_failures.extend({"clip": clip, **item} for item in clip_hard_failures)

        entry = {"aggregate": agg, "perf_ms": perf, "meta": clip_meta, "worst_frame": worst}
        results[clip] = entry

        # Regression gate vs baseline. A baseline is only valid for the exact frames it was made
        # from: if the clip content changed, gating against it is meaningless -- skip it loudly
        # instead of silently comparing apples to oranges.
        bp = os.path.join(base_dir, clip + ".json")
        if not args.update_baselines and not args.comparison_only:
            base = baseline_manifests[clip]
            evidence_failures.extend(primary_evidence_failures(
                agg, thresholds, clip, clip_meta, baseline=base.get("aggregate", {}),
                worst=worst))
            for k, spec in thresholds["metrics"].items():
                if metric_exempt_for_clip(spec, clip_meta):
                    continue
                if spec.get("role") == "hard":
                    continue  # absolute hard constraints were evaluated above, independent of baseline
                b, n = base["aggregate"].get(k), agg.get(k)
                if b is None or n is None:
                    continue
                if sbsbench.metric_gate_failed(b, n, spec):
                    regressions.append({"clip": clip, "metric": k, "baseline": round(b, 3),
                                        **worst.get(k, {}), "value": round(n, 3)})
            evidence_failures.extend(perf_evidence_failures(
                base.get("perf_ms", {}), perf, thresholds, clip))
            if not contention:
                for k, spec in thresholds["perf_ms"].items():
                    b, n = base.get("perf_ms", {}).get(k), perf.get(k)
                    if b is not None and n is not None and (
                            n - b) > max(spec["abs_floor"], b * spec["rel_tol"]):
                        regressions.append({"clip": clip, "metric": "perf:" + k,
                                            "baseline": round(b, 2), "value": round(n, 2)})

        if args.update_baselines:
            evidence_failures.extend(primary_evidence_failures(
                agg, thresholds, clip, clip_meta, worst=worst))
            evidence_failures.extend(perf_evidence_failures(None, perf, thresholds, clip))
            baseline_updates[bp] = {
                "aggregate": agg, "perf_ms": perf,
                "meta": {**meta, **clip_meta, "clip_sha1": meta["clip_set_sha1"][clip]}}

    verdict = ("hard_failures" if hard_failures else
               "evidence_failures" if evidence_failures else
               "comparison_only" if args.comparison_only
               else "regressions" if regressions else "pass")
    out = {"meta": meta, "verdict": verdict, "regressions": regressions,
           "hard_failures": hard_failures, "evidence_failures": evidence_failures,
           "issues": issues, "clips": results}
    res_path = os.path.join(out_root, "results.json")
    json.dump(out, open(res_path, "w"), indent=2)

    if args.update_baselines:
        if hard_failures or evidence_failures:
            fail(f"refusing baseline update: {len(hard_failures)} hard and "
                 f"{len(evidence_failures)} missing-evidence failure(s); results preserved at "
                 + res_path)
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
        if args.report_allow_executable_diff:
            report_cmd.append("--allow-executable-diff")
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
        value = "missing" if r.get("missing") else r["value"]
        print(f"  HARD FAIL {r['clip']}.{r['metric']}: {value} ({bounds})")
    for r in evidence_failures:
        detail = (f"missing from {r.get('source', 'current run')}" if r.get("missing") else
                  f"invalid {r.get('source', 'evidence')} provenance")
        print(f"  EVIDENCE FAIL {r['clip']}.{r['metric']}: {detail}")
    for i in issues:
        relation = (f"> {i['trigger']}" if "trigger" in i else f"< {i['trigger_min']}")
        print(f"  issue {i['clip']}.{i['metric']} = {i['value']} ({relation},"
              f" worst frame {i.get('frame', '?')}, frame value {i.get('worst_value', '?')})")
    if report_path:
        print(f"  report: {report_path}")
    if args.update_baselines:
        print(f"  baselines updated in {base_dir} -- commit them with the change that justified it.")
    sys.exit(1 if regressions or hard_failures or evidence_failures else 0)


if __name__ == "__main__":
    main()
