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
import copy
import datetime
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import weakref

# Pin numeric kernels before NumPy initializes. Frame-level process workers own parallelism. Keep
# the caller's values so the production harness does not inherit evaluator-only thread limits.
_NUMERIC_THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                       "NUMEXPR_NUM_THREADS")
_ORIGINAL_NUMERIC_THREAD_ENV = {key: os.environ.get(key) for key in _NUMERIC_THREAD_ENV}
for _thread_env in _NUMERIC_THREAD_ENV:
    os.environ[_thread_env] = "1"

import numpy as np  # noqa: E402  (thread limits must precede numeric-runtime import)
import PIL  # noqa: E402
from PIL import Image  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
import sbsbench  # noqa: E402  (metric implementations)

EVAL_SCHEMA = 32  # compact exact-map/topology/binocular-conflict contract; harness contract 16
BASELINE_SNAPSHOT_SCHEMA = 1
BASELINE_SNAPSHOT_FILE = "baseline_snapshot.json"
TRAINING_LABEL_STATUS = "qualified"
TRAINING_LABEL_ROLES = {"reward", "risk", "hard"}


def production_subprocess_env():
    """Environment for Sunshine/build children without evaluator-only numeric thread limits."""
    environment = os.environ.copy()
    for key, original in _ORIGINAL_NUMERIC_THREAD_ENV.items():
        if original is None:
            environment.pop(key, None)
        else:
            environment[key] = original
    return environment


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


def image_size_set(paths):
    """Return decoded (width, height) values while closing every image handle."""
    sizes = set()
    for path in paths:
        with Image.open(path) as image:
            sizes.add(image.size)
    return sizes


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


def sha256_json(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def scored_artifact_digests(directory):
    """Hash authenticated and numeric-only artifact sets in one file traversal."""
    scored_fixed = {
        "contract.json", "sbs_perf.json", "warp_map_shape.json", "hdr_output_stats.json",
    }
    numeric_fixed = {"warp_map_shape.json", "hdr_output_stats.json"}
    frame_pattern = re.compile(r"^(?:sbs|depth|warp_map|warp_mask)_\d+\.(?:png|f32)$")
    paths = sorted(
        path for path in glob.glob(os.path.join(directory, "*"))
        if os.path.isfile(path) and
        (os.path.basename(path) in scored_fixed or
         frame_pattern.fullmatch(os.path.basename(path))))
    if not paths:
        raise ValueError(f"no scored artifacts in {directory}")
    scored_digest = hashlib.sha256()
    numeric_digest = hashlib.sha256()
    numeric_count = 0
    for path in paths:
        basename = os.path.basename(path)
        relative = os.path.relpath(path, directory).replace("\\", "/")
        numeric = basename in numeric_fixed or frame_pattern.fullmatch(basename)
        relative_bytes = relative.encode("utf-8")
        scored_digest.update(relative_bytes)
        scored_digest.update(b"\0")
        if numeric:
            numeric_count += 1
            numeric_digest.update(relative_bytes)
            numeric_digest.update(b"\0")
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                scored_digest.update(chunk)
                if numeric:
                    numeric_digest.update(chunk)
    return scored_digest.hexdigest(), (numeric_digest.hexdigest() if numeric_count else None)


def scored_artifact_sha256(directory):
    """Hash canonical metric inputs, excluding later reports and optional-oracle debris."""
    return scored_artifact_digests(directory)[0]


_REMEASUREMENT_SESSION_TOKEN = object()
_REMEASUREMENT_SESSION_ENTRIES = weakref.WeakKeyDictionary()


class _RemeasurementSession:
    """Opaque process-local cache for authenticated pixel remeasurements."""

    __slots__ = ("_token", "__weakref__")

    def __init__(self, token):
        if token is not _REMEASUREMENT_SESSION_TOKEN:
            raise TypeError("remeasurement sessions must be created by run_eval")
        self._token = token


def new_remeasurement_session():
    """Return an empty process-local cache that callers cannot pre-populate."""
    session = _RemeasurementSession(_REMEASUREMENT_SESSION_TOKEN)
    _REMEASUREMENT_SESSION_ENTRIES[session] = {}
    return session


def _validated_remeasurement_entries(session):
    if session is None:
        return None
    if (not isinstance(session, _RemeasurementSession) or
            session._token is not _REMEASUREMENT_SESSION_TOKEN):
        raise TypeError("invalid remeasurement session")
    try:
        return _REMEASUREMENT_SESSION_ENTRIES[session]
    except KeyError as exc:
        raise TypeError("unregistered remeasurement session") from exc


def metric_contract_files():
    """Return every local source file that defines an emitted automatic metric.

    Keep this list explicit. ``sbsbench.py`` delegates exact-map registration and artifact
    scoring to separate modules; hashing only the dispatcher
    would let a semantic change in one of those modules reuse stale baselines and frame labels.
    Optional learned oracles are intentionally absent because they cannot vote on the canonical
    result and carry their own checkpoint/implementation provenance.
    """
    return [
        os.path.join(SCRIPT_DIR, "sbsbench.py"),
        os.path.join(SCRIPT_DIR, "sbs_interocular_metrics.py"),
        os.path.join(SCRIPT_DIR, "sbs_interocular_phase_chroma.py"),
        os.path.join(SCRIPT_DIR, "sbs_interocular_photometric_rivalry.py"),
        os.path.join(SCRIPT_DIR, "sbs_stereo_window_metrics.py"),
        os.path.join(SCRIPT_DIR, "sbs_warp_shear_metrics.py"),
        os.path.join(SCRIPT_DIR, "thresholds.json"),
    ]


def metric_contract_sha():
    """Hash automatic metric implementation and thresholds.

    Runner/gating semantics are versioned separately by EVAL_SCHEMA. Hashing this entire file made
    comments and diagnostic wording invalidate otherwise-identical committed baselines.
    """
    return sha256_files(metric_contract_files())


def metric_runtime_provenance():
    """Versions whose numeric/image kernels can change emitted metric values."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
    }


def label_contract_sha():
    """Hash every local semantic component that can change reusable frame labels.

    `metric_sha256` intentionally stays narrow for ordinary report/baseline ergonomics. Training
    labels need a stricter contract: aggregation, eligibility, support and gate changes in this
    runner must invalidate cached labels even when the underlying pixel metric did not change.
    Runtime shader/executable/model/config/source hashes are recorded separately in the context.
    """
    return sha256_files(metric_contract_files() + [
        os.path.join(SCRIPT_DIR, "run_eval.py"),
        os.path.join(SCRIPT_DIR, "rescore_run.py"),
    ])


def label_context_sha(meta):
    """Bind label semantics to the exact renderer/model/config/source/candidate context."""
    keys = (
        "eval_schema", "run_kind", "metric_sha256", "label_contract_sha256",
        "clip_set_sha1", "conf_sha256",
        "executable_sha256", "runtime_shader_sha256", "model", "engine_name",
        "engine_sha256", "onnx_sha256", "profile", "extra_args", "depth_step",
        "depth_compensation", "literal_bestv2", "adaptive_pop", "adaptive_pop_max",
        "zero_plane", "training_labels", "training_label_gate", "metric_runtime",
        "scored_artifact_sha256", "baseline_snapshot_sha256",
    )
    return sha256_json({key: meta.get(key) for key in keys})


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

    if "required_gt_stereo" in meta:
        if not isinstance(meta["required_gt_stereo"], bool):
            raise ValueError(
                f"invalid clip metadata {meta_path}: retired required_gt_stereo must be boolean")
        if ("reference_stereo_available" in meta and
                meta["reference_stereo_available"] != meta["required_gt_stereo"]):
            raise ValueError(
                f"invalid clip metadata {meta_path}: conflicting retired/current stereo "
                "reference declarations")
        # Existing external prepared caches are immutable inputs. Migrate the retired spelling in
        # memory, but never publish or interpret it as consumed ground-truth evidence.
        meta["reference_stereo_available"] = meta.pop("required_gt_stereo")
    requirement_keys = ("required_gt_depth", "required_gt_flow")
    for key in requirement_keys:
        if key in meta and not isinstance(meta[key], bool):
            raise ValueError(f"invalid clip metadata {meta_path}: {key} must be boolean")
    if ("reference_stereo_available" in meta and
            not isinstance(meta["reference_stereo_available"], bool)):
        raise ValueError(
            f"invalid clip metadata {meta_path}: reference_stereo_available must be boolean")
    if "evaluation_role" in meta and meta["evaluation_role"] not in {
            "ground-truth", "reference-only"}:
        raise ValueError(
            f"invalid clip metadata {meta_path}: evaluation_role must be ground-truth or "
            "reference-only")
    if suite == "extended":
        if not isinstance(meta.get("dataset"), str) or not meta["dataset"].strip():
            raise ValueError(f"invalid extended clip metadata {meta_path}: dataset is required")
        has_consumed_gt = any(meta.get(key) is True for key in requirement_keys)
        if (not has_consumed_gt and meta.get("reference_stereo_available") is True and
                "evaluation_role" not in meta):
            meta["evaluation_role"] = "reference-only"
        is_reference_only = meta.get("evaluation_role") == "reference-only"
        if not has_consumed_gt and not (
                is_reference_only and meta.get("reference_stereo_available") is True):
            raise ValueError(
                f"invalid extended clip metadata {meta_path}: declare consumed depth/flow GT, "
                "or explicitly mark a diagnostic stereo pair evaluation_role=reference-only")
        if is_reference_only and has_consumed_gt:
            raise ValueError(
                f"invalid extended clip metadata {meta_path}: reference-only clips cannot "
                "declare consumed depth/flow GT")
        if meta.get("required_gt_depth") and meta.get("gt_depth_kind") not in {
                "disparity", "metric", "depth"}:
            raise ValueError(
                f"invalid extended clip metadata {meta_path}: required GT depth needs "
                "gt_depth_kind=disparity, metric, or depth")
    reference_patterns = {
        "required_gt_depth": os.path.join(path, "gt_depth", "frame_*.*"),
        "required_gt_flow": os.path.join(path, "gt_flow", "frame_*.npz"),
        "reference_stereo_available": os.path.join(path, "gt_right", "frame_*.*"),
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
        "required_gt_flow", "reference_stereo_available", "dataset", "homepage", "citation",
        "license_note", "content_type", "evaluation_role", "source_url", "source_window",
        "source_artifacts",
    )
    published = {key: source_meta[key] for key in keys if key in source_meta}
    if "suite" in source_meta:
        published["source_suite"] = source_meta["suite"]
    return published


def source_evidence_digests(path):
    """Return legacy source SHA-1 and full cache-only SHA-256 in one traversal."""
    # Hash source pixels plus validation references. Human-readable names/descriptions remain
    # excluded, while semantic metadata that changes scoring is part of the contract.
    legacy = hashlib.sha1()
    full = hashlib.sha256()
    semantic_sidecars = (
        "gt_depth", "gt_depth_valid", "gt_depth_valid_all", "gt_depth_valid_nonocc",
        "gt_flow", "gt_right", "gt_occlusion", "gt_outofframe", "gt_right_disparity",
        "gt_detail", "gt_match", "gt_sky",
    )
    files = glob.glob(os.path.join(path, "frame_*"))
    for directory in semantic_sidecars:
        files.extend(glob.glob(os.path.join(path, directory, "frame_*")))
    for f in sorted(files):
        relative = os.path.relpath(f, path).replace("\\", "/").encode()
        legacy.update(relative)
        full.update(relative)
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                legacy.update(chunk)
                full.update(chunk)
    meta = load_clip_metadata(path, required=False)
    semantic = {k: meta[k] for k in ("expected_flat", "gt_depth_kind", "dataset",
                                     "required_gt_depth", "required_gt_flow",
                                     "reference_stereo_available", "evaluation_role") if k in meta}
    semantic_bytes = json.dumps(semantic, sort_keys=True).encode()
    legacy.update(semantic_bytes)
    full.update(semantic_bytes)
    return legacy.hexdigest()[:12], full.hexdigest()


def sha1_dir(path):
    """Legacy committed-baseline source identity."""
    return source_evidence_digests(path)[0]


def _validate_baseline_manifest(baseline, clip, source, required_common, clip_hashes):
    """Validate one baseline manifest against the exact candidate evaluation contract."""
    if not isinstance(baseline, dict):
        raise ValueError(f"committed baseline {source} must be an object")
    meta = baseline.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"committed baseline {source} has no metadata object")
    if not isinstance(baseline.get("aggregate"), dict):
        raise ValueError(f"committed baseline {source} has no aggregate object")
    if not isinstance(baseline.get("perf_ms"), dict):
        raise ValueError(f"committed baseline {source} has no perf_ms object")
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
        _validate_baseline_manifest(
            baseline, clip, path, required_common, clip_hashes)
        baselines[clip] = baseline
    return baselines


def build_baseline_snapshot(base_dir, baselines):
    """Freeze the exact preflighted baseline evidence used by a gated run."""
    entries = {}
    for clip in sorted(baselines):
        path = os.path.join(base_dir, clip + ".json")
        try:
            with open(path, encoding="utf-8") as baseline_file:
                source_manifest = json.load(baseline_file)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot snapshot committed baseline {path}: {exc}") from exc
        if source_manifest != baselines[clip]:
            raise ValueError(f"committed baseline changed during preflight: {path}")
        entries[clip] = {
            "source": os.path.abspath(path),
            "source_file_sha256": file_sha256(path),
            "manifest_sha256": sha256_json(baselines[clip]),
            "manifest": copy.deepcopy(baselines[clip]),
        }
    return {
        "schema": BASELINE_SNAPSHOT_SCHEMA,
        "source_directory": os.path.abspath(base_dir),
        "clips": entries,
    }


def validate_baseline_snapshot(snapshot, clips, required_common, clip_hashes):
    """Authenticate and validate a frozen baseline snapshot, returning its manifests."""
    if not isinstance(snapshot, dict) or snapshot.get("schema") != BASELINE_SNAPSHOT_SCHEMA:
        raise ValueError(
            f"baseline snapshot schema must be {BASELINE_SNAPSHOT_SCHEMA}")
    entries = snapshot.get("clips")
    if not isinstance(entries, dict) or set(entries) != set(clips):
        raise ValueError("baseline snapshot clips must exactly cover the scored clip set")
    baselines = {}
    for clip in clips:
        entry = entries[clip]
        if not isinstance(entry, dict):
            raise ValueError(f"baseline snapshot clips.{clip} must be an object")
        manifest = entry.get("manifest")
        digest = entry.get("manifest_sha256")
        if not isinstance(digest, str) or digest != sha256_json(manifest):
            raise ValueError(f"baseline snapshot clips.{clip} manifest digest mismatch")
        source_digest = entry.get("source_file_sha256")
        if (not isinstance(source_digest, str) or
                not re.fullmatch(r"[0-9a-f]{64}", source_digest)):
            raise ValueError(
                f"baseline snapshot clips.{clip} has invalid source-file provenance")
        _validate_baseline_manifest(
            manifest, clip, f"snapshot:{clip}", required_common, clip_hashes)
        baselines[clip] = manifest
    return baselines


def baseline_required_context(candidate_meta):
    """Project candidate metadata onto the contract required of its canonical baseline."""
    return {
        "mode": "profile",
        "run_kind": "baseline-update",
        "suite": candidate_meta.get("suite"),
        "model": candidate_meta.get("model"),
        "profile": candidate_meta.get("profile"),
        "eval_schema": candidate_meta.get("eval_schema"),
        "depth_step": candidate_meta.get("depth_step"),
        "depth_compensation": candidate_meta.get("depth_compensation"),
        "conf_sha256": candidate_meta.get("conf_sha256"),
        "metric_sha256": candidate_meta.get("metric_sha256"),
        "label_contract_sha256": candidate_meta.get("label_contract_sha256"),
        "metric_runtime": candidate_meta.get("metric_runtime"),
    }


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
        row_states = [(row, sbsbench.metric_evidence_state(
            metric, spec, row, clip_meta)) for row in rows]
        applicable_rows = [row for row, state in row_states if state == "applicable"]
        clip_state = sbsbench.metric_evidence_state(metric, spec, agg, clip_meta)
        if not applicable_rows and clip_state == "unsupported":
            continue
        if spec.get("role") == "hard":
            missing_rows = [row.get("_frame_id", index)
                            for index, (row, state) in enumerate(row_states)
                            if state == "missing" or
                            (state == "applicable" and
                             not sbsbench.metric_value_valid(row, metric))]
            if (missing_rows or clip_state == "missing" or
                    (clip_state == "applicable" and
                     not sbsbench.metric_value_valid(agg, metric))):
                hard_failures.append({
                    "metric": metric, "value": None, "missing": True,
                    "missing_frames": missing_rows,
                    "hard_min": spec.get("hard_min"), "hard_max": spec.get("hard_max")})
                continue
        frame_key = metric if any(metric in row for row in rows) else (
            metric[:-4] if metric.endswith(("_p50", "_p95")) else metric)
        values = [(row.get(frame_key), row.get("_frame_id", i))
                  for i, row in enumerate(applicable_rows)
                  if sbsbench.metric_value_valid(row, frame_key)]
        if values:
            choose = min if spec.get("better") == "higher" else max
            value, frame = choose(values)
            worst[metric] = {"frame": frame, "worst_value": round(value, 3)}
        if ("trigger" in spec and sbsbench.metric_value_valid(agg, metric)
                and agg[metric] > spec["trigger"]):
            issues.append({"metric": metric, "trigger": spec["trigger"],
                           **worst.get(metric, {}), "value": round(agg[metric], 3)})
        if ("trigger_min" in spec and sbsbench.metric_value_valid(agg, metric)
                and agg[metric] < spec["trigger_min"]):
            issues.append({"metric": metric, "trigger_min": spec["trigger_min"],
                           **worst.get(metric, {}), "value": round(agg[metric], 3)})
        if spec.get("role") == "hard" and sbsbench.metric_value_valid(agg, metric):
            if sbsbench.metric_gate_failed(agg[metric], agg[metric], spec):
                hard_failures.append({"metric": metric, **worst.get(metric, {}),
                                      "value": round(agg[metric], 3),
                                      "hard_min": spec.get("hard_min"),
                                      "hard_max": spec.get("hard_max")})
    return worst, issues, hard_failures


def training_label_manifest(thresholds):
    """Describe label candidates without allowing an omitted/typoed status to opt in.

    Evaluation roles are intentionally independent from model-label qualification. A metric may
    remain a hard/primary run gate while its label is experimental or while it has no label role
    at all. Only the exact literal status ``qualified`` crosses this boundary.
    """
    qualified, excluded = [], []
    for metric, spec in thresholds["metrics"].items():
        label_role = spec.get("label")
        if label_role is None:
            continue
        item = {
            "metric": metric,
            "role": label_role,
            "status": spec.get("label_status", "unqualified"),
        }
        if item["status"] == TRAINING_LABEL_STATUS:
            if spec.get("scope") != "perceptual":
                raise ValueError(
                    f"qualified training metric {metric!r} must have perceptual scope")
            if label_role not in TRAINING_LABEL_ROLES:
                raise ValueError(
                    f"qualified training metric {metric!r} has invalid label role {label_role!r}")
            if spec.get("role") not in {"hard", "primary", "diagnostic"}:
                raise ValueError(
                    f"qualified training metric {metric!r} has invalid evaluator role")
            # Validate the requirement name without allowing the metric's value to authenticate it.
            sbsbench.metric_evidence_state(metric, spec, {}, {})
            qualified.append(item)
        else:
            excluded.append(item)
    return {
        "required_status": TRAINING_LABEL_STATUS,
        "qualified_metrics": qualified,
        "excluded_metrics": excluded,
    }


def build_frame_records(rows, thresholds, clip_meta):
    """Serialize only policy metrics/evidence; export only explicitly qualified labels.

    Detector debug scalars (sub-components, legacy aliases, raw residuals) remain available to
    their standalone validators but must not silently become DA-V2 training features merely
    because they happened to be emitted by a measurement helper.
    """
    metric_specs = thresholds["metrics"]
    policy_names = set(metric_specs)
    evidence_names = {
        spec.get("requires") for spec in metric_specs.values()
        if isinstance(spec.get("requires"), str)
    }
    serialized_names = policy_names | evidence_names
    records = []
    for position, row in enumerate(rows):
        numeric = {key: float(value) for key, value in row.items()
                   if key in serialized_names and sbsbench.metric_value_valid(row, key)}
        labels = canonical_frame_labels(numeric, thresholds, clip_meta)
        records.append({
            "frame_id": row.get("_frame_id", position),
            "metrics": numeric,
            "labels": labels,
        })
    return records


def canonical_frame_labels(numeric, thresholds, clip_meta):
    """Rebuild the only reusable label object from serialized metric evidence.

    Keeping this derivation shared by the writer and the verifier prevents an edited/stale
    ``labels`` object from becoming training truth merely because its JSON shape looks valid.
    """
    manifest = training_label_manifest(thresholds)
    qualified_names = {item["metric"] for item in manifest["qualified_metrics"]}
    qualified_specs = {
        metric: spec for metric, spec in thresholds["metrics"].items()
        if metric in qualified_names
    }
    labels = sbsbench.frame_label_evidence(numeric, qualified_specs, clip_meta)
    valid_count = sum(item.get("state") == "valid"
                      for item in labels["metrics"].values())
    labels["required_status"] = TRAINING_LABEL_STATUS
    labels["qualified_metric_count"] = len(qualified_specs)
    labels["valid_metric_count"] = valid_count
    if not qualified_specs:
        labels["eligible"] = False
        labels["reason"] = "no_qualified_training_labels"
    elif valid_count == 0:
        labels["eligible"] = False
        labels["reason"] = "no_valid_qualified_training_labels"
    return labels


def summarize_frame_labels(frame_records, thresholds):
    """Return backward-compatible counts plus the run's qualification manifest."""
    manifest = training_label_manifest(thresholds)
    return {
        "eligible_frames": sum(record["labels"]["eligible"] for record in frame_records),
        "abstained_frames": sum(not record["labels"]["eligible"]
                                for record in frame_records),
        "total_frames": len(frame_records),
        **manifest,
    }


def training_label_evidence_gate(results, thresholds=None, *, require_context=True):
    """Validate reusable labels against current contracts and their numeric evidence.

    Perceptual-risk failures can be useful negative examples, but capture/renderer/GT failures
    make the image-label pair untrustworthy and therefore block the whole export.
    """
    meta = results.get("meta")
    clips = results.get("clips")
    blockers = []
    if not isinstance(meta, dict) or meta.get("run_kind") not in {
            "baseline-gated", "baseline-update", "comparison-only"}:
        blockers.append("meta.run_kind")
    if not isinstance(clips, dict) or not clips:
        blockers.append("clips")
        clips = {}
    evidence_failures = results.get("evidence_failures")
    if not isinstance(evidence_failures, list):
        blockers.append("evidence_failures")
    elif evidence_failures:
        blockers.append("missing_metric_or_performance_evidence")
    quality_lists = {}
    for key in ("hard_failures", "regressions"):
        values = results.get(key)
        if not isinstance(values, list):
            blockers.append(key)
        else:
            quality_lists[key] = values
    if not isinstance(meta, dict):
        meta = {}
    if meta.get("run_kind") == "baseline-gated":
        snapshot_digest = meta.get("baseline_snapshot_sha256")
        if (not isinstance(snapshot_digest, str) or
                not re.fullmatch(r"[0-9a-f]{16}", snapshot_digest)):
            blockers.append("meta.baseline_snapshot_sha256")
    if thresholds is not None:
        label_manifest = training_label_manifest(thresholds)
        if meta.get("training_labels") != label_manifest:
            blockers.append("meta.training_labels")
        if not label_manifest["qualified_metrics"]:
            blockers.append("no_qualified_training_labels")
        if meta.get("eval_schema") != EVAL_SCHEMA:
            blockers.append("meta.eval_schema")
        if meta.get("metric_sha256") != metric_contract_sha():
            blockers.append("meta.metric_sha256")
        if meta.get("label_contract_sha256") != label_contract_sha():
            blockers.append("meta.label_contract_sha256")
    artifact_hashes = meta.get("scored_artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(clips):
        blockers.append("meta.scored_artifact_sha256")
    elif any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
             for value in artifact_hashes.values()):
        blockers.append("meta.scored_artifact_sha256.values")
    runtime = meta.get("metric_runtime")
    if (not isinstance(runtime, dict) or
            any(not isinstance(runtime.get(key), str) or not runtime[key]
                for key in ("python", "numpy", "pillow"))):
        blockers.append("meta.metric_runtime")
    elif thresholds is not None and runtime != metric_runtime_provenance():
        blockers.append("meta.metric_runtime.current")
    if not (isinstance(meta.get("label_contract_sha256"), str) and
            re.fullmatch(r"[0-9a-f]{16}", meta["label_contract_sha256"])):
        blockers.append("meta.label_contract_sha256")
    clip_hashes = meta.get("clip_set_sha1")
    if (not isinstance(clip_hashes, dict) or set(clip_hashes) != set(clips) or
            any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{12}", value)
                for value in clip_hashes.values())):
        blockers.append("meta.clip_set_sha1")
    for clip, entry in clips.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("frames"), list) or not entry[
                "frames"]:
            blockers.append(f"clips.{clip}.frames")
            continue
        entry_meta = entry.get("meta")
        if not isinstance(entry_meta, dict):
            blockers.append(f"clips.{clip}.meta")
            entry_meta = {}
        frame_ids = []
        for record in entry["frames"]:
            labels = record.get("labels") if isinstance(record, dict) else None
            if (not isinstance(record, dict) or
                    isinstance(record.get("frame_id"), bool) or
                    not isinstance(record.get("frame_id"), int) or
                    not isinstance(record.get("metrics"), dict) or
                    not isinstance(labels, dict) or
                    not isinstance(labels.get("eligible"), bool) or
                    not isinstance(labels.get("metrics"), dict)):
                blockers.append(f"clips.{clip}.frame_records")
                continue
            frame_ids.append(record["frame_id"])
            if thresholds is not None:
                expected_labels = canonical_frame_labels(
                    record["metrics"], thresholds, entry_meta)
                if labels != expected_labels:
                    blockers.append(f"clips.{clip}.frame_labels")
        if len(frame_ids) != len(set(frame_ids)):
            blockers.append(f"clips.{clip}.frame_ids")
        source_count = entry_meta.get("source_frame_count")
        if (isinstance(source_count, bool) or not isinstance(source_count, int) or
                source_count != len(entry["frames"])):
            blockers.append(f"clips.{clip}.frame_count")
        clip_hash = entry_meta.get("scored_artifact_sha256")
        if not isinstance(artifact_hashes, dict) or clip_hash != artifact_hashes.get(clip):
            blockers.append(f"clips.{clip}.scored_artifact_sha256")
        if thresholds is not None:
            expected_summary = summarize_frame_labels(entry["frames"], thresholds)
            if entry.get("label_summary") != expected_summary:
                blockers.append(f"clips.{clip}.label_summary")

    # Only an explicitly qualified perceptual hard metric may survive as a negative example.
    # Renderer/conformance, GT, temporal, unknown, and provenance failures invalidate labels.
    if thresholds is not None and "hard_failures" in quality_lists:
        for failure in quality_lists["hard_failures"]:
            metric = failure.get("metric") if isinstance(failure, dict) else None
            spec = thresholds["metrics"].get(metric, {})
            reusable_negative = (
                spec.get("scope") == "perceptual" and
                spec.get("label_status") == TRAINING_LABEL_STATUS and
                spec.get("label") in {"risk", "hard"})
            if not reusable_negative:
                blockers.append("nonperceptual_or_unqualified_hard_failure")

    # Candidate-quality regressions may be useful negative examples. The verdict still has to
    # agree with its lists so a partially-written or manually corrupted run fails closed.
    if (isinstance(evidence_failures, list) and
            "hard_failures" in quality_lists and "regressions" in quality_lists):
        expected_verdict = (
            "hard_failures" if quality_lists["hard_failures"] else
            "evidence_failures" if evidence_failures else
            "comparison_only" if meta.get("run_kind") == "comparison-only" else
            "regressions" if quality_lists["regressions"] else "pass")
        if results.get("verdict") != expected_verdict:
            blockers.append("verdict")
    if require_context:
        context = meta.get("label_context_sha256")
        if (not isinstance(context, str) or not re.fullmatch(r"[0-9a-f]{16}", context) or
                context != label_context_sha(meta)):
            blockers.append("meta.label_context_sha256")
    return {"passed": not blockers, "blockers": sorted(set(blockers)),
            "run_kind": meta.get("run_kind"), "verdict": results.get("verdict")}


def bind_training_labels_to_evidence_gate(results, thresholds):
    """Bind label eligibility to evidence/provenance, not candidate quality acceptance."""
    meta = results.setdefault("meta", {})
    meta.pop("label_context_sha256", None)
    gate = training_label_evidence_gate(results, thresholds, require_context=False)
    compact_gate = {
        "passed": bool(gate["passed"]),
        "run_kind": gate.get("run_kind"),
        "verdict": gate.get("verdict"),
        "blockers": gate.get("blockers", []),
    }
    meta["training_label_gate"] = compact_gate
    if gate["passed"]:
        meta["label_context_sha256"] = label_context_sha(meta)
        verified = training_label_evidence_gate(results, thresholds, require_context=True)
        if verified["passed"]:
            return compact_gate
        gate = verified
        compact_gate = {
            "passed": False,
            "run_kind": gate.get("run_kind"),
            "verdict": gate.get("verdict"),
            "blockers": gate.get("blockers", []),
        }
        meta["training_label_gate"] = compact_gate
    for entry in results.get("clips", {}).values():
        records = entry.get("frames", [])
        for record in records:
            labels = record.get("labels", {})
            labels["eligible"] = False
            labels["reason"] = "training_label_evidence_gate_failed"
        entry["label_summary"] = summarize_frame_labels(records, thresholds)
    meta["label_context_sha256"] = label_context_sha(meta)
    return compact_gate


_TEMPORAL_AGGREGATE_FRAME_METRICS = {
    "static_jitter_p95": "static_jitter",
    "flow_temporal_p95": "flow_temporal",
    "depth_gt_lag_f1_p95": "depth_gt_lag_f1",
}


def primary_evidence_failures(current, thresholds, clip, clip_meta, baseline=None, worst=None,
                              rows=None):
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
        current_state = sbsbench.metric_evidence_state(metric, spec, current, clip_meta)
        baseline_state = (sbsbench.metric_evidence_state(
            metric, spec, baseline, clip_meta) if baseline is not None else None)
        if current_state == "unsupported" and (
                baseline is None or baseline_state == "unsupported"):
            continue
        missing = []
        if baseline is not None:
            if baseline_state != "applicable":
                missing.append(f"baseline_{baseline_state}")
            elif not sbsbench.metric_value_valid(baseline, metric):
                missing.append("baseline")
        if current_state != "applicable":
            missing.append(f"current_{current_state}")
        elif not sbsbench.metric_value_valid(current, metric):
            missing.append("current")
        missing_frames = []
        if rows:
            frame_metric = _TEMPORAL_AGGREGATE_FRAME_METRICS.get(metric, metric)
            # Temporal p95 metrics summarize transitions, not frames. Frame zero has no previous
            # image and is structurally inapplicable even though the clip itself is multi-frame.
            # Validate the underlying transition value on subsequent rows instead of looking for
            # an aggregate ``*_p95`` key that can never exist on an individual frame.
            frame_rows = rows[1:] if frame_metric != metric else rows
            missing_frames = [
                row.get("_frame_id", index + (1 if frame_metric != metric else 0))
                for index, row in enumerate(frame_rows)
                if (sbsbench.metric_evidence_state(metric, spec, row, clip_meta) == "missing"
                    or (sbsbench.metric_evidence_state(metric, spec, row, clip_meta)
                        == "applicable"
                        and not sbsbench.metric_value_valid(row, frame_metric)))]
            if missing_frames and "current" not in missing:
                missing.append("current_frames")
        if not missing:
            continue
        item = {"clip": clip, "metric": metric, "missing": True,
                "source": "+".join(missing), **worst.get(metric, {})}
        if missing_frames:
            item["missing_frames"] = missing_frames
        if baseline is not None and sbsbench.metric_value_valid(baseline, metric):
            item["baseline"] = round(baseline[metric], 3)
        failures.append(item)
    return failures


def perf_evidence_failures(base, current, thresholds, clip):
    """Return missing/invalid performance evidence before comparing numeric regressions."""
    failures = []
    for metric in thresholds["perf_ms"]:
        baseline_value = base.get(metric) if base is not None else None
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


def score_baseline_comparison(current, perf, rows, worst, clip_meta, baseline, thresholds,
                              clip, *, skip_perf_regressions=False):
    """Deterministically reconstruct one candidate-vs-baseline decision.

    The live runner and later artifact verifier must share this implementation.  Duplicating the
    loop would let a report authenticate pixels while still trusting stale cached regressions.
    """
    baseline_aggregate = baseline.get("aggregate", {})
    baseline_perf = baseline.get("perf_ms", {})
    evidence = primary_evidence_failures(
        current, thresholds, clip, clip_meta, baseline=baseline_aggregate,
        worst=worst, rows=rows)
    regressions = []
    for metric, spec in thresholds["metrics"].items():
        if metric_exempt_for_clip(spec, clip_meta) or spec.get("role") != "primary":
            continue
        if (sbsbench.metric_evidence_state(
                metric, spec, baseline_aggregate, clip_meta) != "applicable" or
                sbsbench.metric_evidence_state(
                    metric, spec, current, clip_meta) != "applicable"):
            continue
        baseline_value = baseline_aggregate.get(metric)
        current_value = current.get(metric)
        if baseline_value is None or current_value is None:
            continue
        if sbsbench.metric_gate_failed(baseline_value, current_value, spec):
            regressions.append({
                "clip": clip,
                "metric": metric,
                "baseline": round(baseline_value, 3),
                **worst.get(metric, {}),
                "value": round(current_value, 3),
            })
    evidence.extend(perf_evidence_failures(
        baseline_perf, perf, thresholds, clip))
    if not skip_perf_regressions:
        for metric, spec in thresholds["perf_ms"].items():
            baseline_value = baseline_perf.get(metric)
            current_value = perf.get(metric)
            if baseline_value is not None and current_value is not None and (
                    current_value - baseline_value) > max(
                        spec["abs_floor"], baseline_value * spec["rel_tol"]):
                regressions.append({
                    "clip": clip,
                    "metric": "perf:" + metric,
                    "baseline": round(baseline_value, 2),
                    "value": round(current_value, 2),
                })
    return evidence, regressions


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


def _first_result_mismatch(observed, expected, path="results"):
    """Return the first deterministic structural mismatch between JSON-compatible values."""
    if isinstance(observed, dict) and isinstance(expected, dict):
        observed_keys = set(observed)
        expected_keys = set(expected)
        if observed_keys != expected_keys:
            missing = sorted(expected_keys - observed_keys)
            extra = sorted(observed_keys - expected_keys)
            return f"{path} keys differ: missing={missing}, extra={extra}"
        for key in sorted(expected_keys):
            mismatch = _first_result_mismatch(
                observed[key], expected[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(observed, list) and isinstance(expected, list):
        if len(observed) != len(expected):
            return f"{path} length differs: {len(observed)} != {len(expected)}"
        for index, (actual_item, expected_item) in enumerate(zip(observed, expected)):
            mismatch = _first_result_mismatch(
                actual_item, expected_item, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    if observed != expected:
        return f"{path} differs: recorded={observed!r}, remeasured={expected!r}"
    return None


def _require_matching_result(observed, expected, path):
    mismatch = _first_result_mismatch(observed, expected, path)
    if mismatch:
        raise ValueError(mismatch)


def authoritative_remeasurement_clip_meta(
        results, clip, clips_root, run_dir, source_sha1=None, artifact_sha256=None,
        validate_images=True):
    """Rebuild scoring metadata from source, frame identities, and harness output.

    ``results.json`` is only a cache.  In particular, an edited ``expected_flat`` flag or a
    forged ``source_frame_count`` can change metric applicability and label completeness.  This
    function deliberately does not read the cached per-clip metadata while constructing the
    replacement.  The source ``meta.json``, the schema-16 harness contract, and the complete set
    of decoded metric image identities are the authorities.
    """
    run_meta = results.get("meta")
    clips = results.get("clips")
    if not isinstance(run_meta, dict) or not isinstance(clips, dict):
        raise ValueError("results must contain meta and clips objects")
    if clip not in clips:
        raise ValueError(f"clips.{clip} is missing")

    source_dir = os.path.join(clips_root, clip)
    source_meta = load_clip_metadata(
        source_dir, suite=run_meta.get("suite"), required=True)
    source_files = sbsbench.indexed_files(
        os.path.join(source_dir, "frame_*.*"), "frame_")
    source_files = {frame_id: path for frame_id, path in source_files.items()
                    if path.lower().endswith((".png", ".jpg", ".jpeg"))}
    if not source_files:
        raise ValueError(f"clips.{clip}: source has no decodable image frames")

    artifact_dir = os.path.join(run_dir, clip)
    scored_images = {
        "SBS": sbsbench.indexed_files(os.path.join(artifact_dir, "sbs_*.png"), "sbs_"),
        "depth": sbsbench.indexed_files(
            os.path.join(artifact_dir, "depth_*.png"), "depth_"),
        "warp-mask": sbsbench.indexed_files(
            os.path.join(artifact_dir, "warp_mask_*.png"), "warp_mask_"),
    }
    mapping_files = sbsbench.indexed_files(
        os.path.join(artifact_dir, "warp_map_*.f32"), "warp_map_")
    source_ids = set(source_files)
    for name, indexed in (*scored_images.items(), ("warp-map", mapping_files)):
        if set(indexed) != source_ids:
            missing = sorted(source_ids - set(indexed))
            extra = sorted(set(indexed) - source_ids)
            raise ValueError(
                f"clips.{clip}: {name}/source frame-id mismatch: "
                f"missing {name}={missing}, extra {name}={extra}")
    if validate_images:
        try:
            # Standalone metadata authentication validates decodability. The full verifier skips
            # this duplicate prepass because its fresh measurement decodes every image, while an
            # in-memory reuse is byte/source-hash-bound to an earlier fresh measurement.
            image_size_set(source_files.values())
            for indexed in scored_images.values():
                image_size_set(indexed.values())
        except OSError as exc:
            raise ValueError(f"clips.{clip}: cannot decode scored image set: {exc}") from exc

    recorded_clip_hashes = run_meta.get("clip_set_sha1")
    if not isinstance(recorded_clip_hashes, dict) or set(recorded_clip_hashes) != set(clips):
        raise ValueError("meta.clip_set_sha1 must exactly cover the scored clip set")
    actual_clip_hash = source_sha1 if source_sha1 is not None else sha1_dir(source_dir)
    _require_matching_result(
        recorded_clip_hashes.get(clip), actual_clip_hash, f"meta.clip_set_sha1.{clip}")

    contract_path = os.path.join(artifact_dir, "contract.json")
    try:
        with open(contract_path, encoding="utf-8") as contract_file:
            contract = json.load(contract_file)
    except (OSError, ValueError) as exc:
        raise ValueError(f"clips.{clip}: invalid harness contract {contract_path}: {exc}") from exc
    if not isinstance(contract, dict) or contract.get("schema") != 16:
        raise ValueError(
            f"clips.{clip}: harness contract schema must be 16, got "
            f"{contract.get('schema') if isinstance(contract, dict) else None!r}")
    contract_keys = (
        "model", "profile", "depth_compensation", "literal_bestv2", "cuda_graph",
        "adaptive_pop", "adaptive_pop_max", "zero_plane",
    )
    authoritative = {key: contract[key] for key in contract_keys if key in contract}
    for key in contract_keys:
        if key not in contract:
            raise ValueError(f"clips.{clip}: harness contract is missing {key}")
    for key in ("model", "profile", "depth_compensation", "literal_bestv2", "cuda_graph",
                "adaptive_pop", "adaptive_pop_max", "zero_plane", "depth_step",
                "depth_reuse_interval"):
        if key in run_meta:
            if key not in contract:
                raise ValueError(f"clips.{clip}: harness contract is missing {key}")
            _require_matching_result(
                run_meta[key], contract[key], f"meta.{key} vs clips.{clip}.contract")
    if "cuda_graph_captured" in contract:
        authoritative["cuda_graph_captured"] = contract["cuda_graph_captured"]
    authoritative["source_frame_count"] = len(source_files)
    authoritative.update(published_clip_metadata(source_meta))

    recorded_artifacts = run_meta.get("scored_artifact_sha256")
    if not isinstance(recorded_artifacts, dict) or set(recorded_artifacts) != set(clips):
        raise ValueError("meta.scored_artifact_sha256 must exactly cover the scored clip set")
    actual_artifact_hash = (
        artifact_sha256 if artifact_sha256 is not None else
        scored_artifact_sha256(artifact_dir))
    _require_matching_result(
        recorded_artifacts.get(clip), actual_artifact_hash,
        f"meta.scored_artifact_sha256.{clip}")
    authoritative["scored_artifact_sha256"] = actual_artifact_hash
    return authoritative


def _authenticated_remeasurement_clip_meta(
        results, clip, clips_root, run_dir, source_sha1=None, artifact_sha256=None,
        validate_images=True):
    """Authenticate cached metadata, then return its authoritative reconstruction."""
    clips = results.get("clips")
    entry = clips.get(clip) if isinstance(clips, dict) else None
    entry_meta = entry.get("meta") if isinstance(entry, dict) else None
    if not isinstance(entry_meta, dict):
        raise ValueError(f"clips.{clip}.meta must be an object")
    authoritative = authoritative_remeasurement_clip_meta(
        results, clip, clips_root, run_dir,
        source_sha1=source_sha1, artifact_sha256=artifact_sha256,
        validate_images=validate_images)
    for key, expected in authoritative.items():
        _require_matching_result(
            entry_meta.get(key), expected, f"clips.{clip}.meta.{key}")
    return authoritative


def verify_results_against_artifacts(results, run_dir, clips_root, thresholds,
                                     remeasurement_session=None):
    """Remeasure a completed run and reject any stale or forged JSON evidence.

    ``results.json`` is a cache and presentation index, never an authority. This verifier binds it
    back to authenticated source frames and scored renderer artifacts, then deterministically
    reconstructs every value used by reports or reusable frame labels. Offline correctness is the
    priority, so the full metric stack is deliberately rerun rather than trusting cached scalars.
    A process-local ``remeasurement_session`` may reuse rows from another run only when this
    function independently confirms complete numeric-input and source-evidence digests. Plain
    caller-supplied dictionaries are never accepted as measurement authority.
    """
    if not isinstance(results, dict):
        raise ValueError("results root must be an object")
    clips = results.get("clips")
    meta = results.get("meta")
    if not isinstance(clips, dict) or not clips:
        raise ValueError("results.clips must be a non-empty object")
    if not isinstance(meta, dict):
        raise ValueError("results.meta must be an object")
    if not isinstance(thresholds, dict) or not isinstance(thresholds.get("metrics"), dict):
        raise ValueError("thresholds must contain a metrics object")
    reusable_entries = _validated_remeasurement_entries(remeasurement_session)
    if meta.get("eval_schema") != EVAL_SCHEMA:
        raise ValueError(
            f"stale evaluator schema {meta.get('eval_schema')!r}; expected {EVAL_SCHEMA}")
    if meta.get("metric_sha256") != metric_contract_sha():
        raise ValueError("recorded metric contract differs from the current implementation")
    if meta.get("label_contract_sha256") != label_contract_sha():
        raise ValueError("recorded label contract differs from the current implementation")
    if meta.get("metric_runtime") != metric_runtime_provenance():
        raise ValueError("recorded numeric runtime differs from the current runtime")

    recorded_artifacts = meta.get("scored_artifact_sha256")
    if not isinstance(recorded_artifacts, dict) or set(recorded_artifacts) != set(clips):
        raise ValueError("meta.scored_artifact_sha256 must exactly cover the scored clip set")
    recorded_clip_hashes = meta.get("clip_set_sha1")
    if (not isinstance(recorded_clip_hashes, dict) or
            set(recorded_clip_hashes) != set(clips)):
        raise ValueError("meta.clip_set_sha1 must exactly cover the scored clip set")

    expected = copy.deepcopy(results)
    expected_issues = []
    expected_hard_failures = []
    expected_evidence_failures = []
    expected_regressions = []
    run_kind = meta.get("run_kind")
    if run_kind not in {"comparison-only", "baseline-update", "baseline-gated"}:
        raise ValueError(f"unsupported run kind {run_kind!r}")

    baseline_manifests = {}
    if run_kind == "baseline-gated":
        snapshot_path = os.path.join(run_dir, BASELINE_SNAPSHOT_FILE)
        try:
            with open(snapshot_path, encoding="utf-8") as snapshot_file:
                snapshot = json.load(snapshot_file)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"cannot load authenticated baseline snapshot {snapshot_path}: {exc}") from exc
        _require_matching_result(
            meta.get("baseline_snapshot_sha256"), sha256_json(snapshot),
            "meta.baseline_snapshot_sha256")
        baseline_manifests = validate_baseline_snapshot(
            snapshot, list(clips), baseline_required_context(meta),
            recorded_clip_hashes)

    for clip in clips:
        entry = clips[clip]
        if not isinstance(entry, dict):
            raise ValueError(f"clips.{clip} must be an object")
        artifact_dir = os.path.join(run_dir, clip)
        source_dir = os.path.join(clips_root, clip)
        try:
            actual_clip_hash, source_sha256 = source_evidence_digests(source_dir)
            actual_artifact_hash, numeric_digest = scored_artifact_digests(artifact_dir)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"clips.{clip}: cannot authenticate scored artifacts: {exc}") from exc
        if numeric_digest is None:
            raise ValueError(f"clips.{clip}: no numeric metric inputs")
        entry_meta = _authenticated_remeasurement_clip_meta(
            results, clip, clips_root, run_dir,
            source_sha1=actual_clip_hash, artifact_sha256=actual_artifact_hash,
            validate_images=False)
        _require_matching_result(
            recorded_artifacts.get(clip), actual_artifact_hash,
            f"meta.scored_artifact_sha256.{clip}")
        _require_matching_result(
            entry_meta.get("scored_artifact_sha256"), actual_artifact_hash,
            f"clips.{clip}.meta.scored_artifact_sha256")

        cache_key = (
            clip, numeric_digest, source_sha256, metric_contract_sha(),
            json.dumps(metric_runtime_provenance(), sort_keys=True, separators=(",", ":")),
        )
        measured = (copy.deepcopy(reusable_entries[cache_key])
                    if reusable_entries is not None and cache_key in reusable_entries else None)
        if measured is None:
            try:
                measured = sbsbench.measure_sequence(artifact_dir, source_dir)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"clips.{clip}: authoritative remeasurement failed: {exc}") from exc
        if not isinstance(measured, tuple) or len(measured) != 2:
            raise ValueError(f"clips.{clip}: authoritative remeasurement produced no sequence")
        if reusable_entries is not None and cache_key not in reusable_entries:
            reusable_entries[cache_key] = copy.deepcopy(measured)
        rows, aggregate = measured
        aggregate = sbsbench.filter_aggregate_by_evidence(
            rows, aggregate, thresholds["metrics"], entry_meta)
        try:
            perf = load_perf_metrics(os.path.join(artifact_dir, "sbs_perf.json"))
        except ValueError as exc:
            raise ValueError(f"clips.{clip}: {exc}") from exc
        worst, clip_issues, clip_hard_failures = score_clip_gates(
            rows, aggregate, thresholds, entry_meta)
        expected_issues.extend({"clip": clip, **item} for item in clip_issues)
        expected_hard_failures.extend(
            {"clip": clip, **item} for item in clip_hard_failures)
        frame_records = build_frame_records(rows, thresholds, entry_meta)
        expected_entry = expected["clips"][clip]
        expected_entry["aggregate"] = aggregate
        expected_entry["perf_ms"] = perf
        expected_entry["worst_frame"] = worst
        expected_entry["frames"] = frame_records
        expected_entry["label_summary"] = summarize_frame_labels(frame_records, thresholds)
        if run_kind in {"comparison-only", "baseline-update"}:
            expected_evidence_failures.extend(primary_evidence_failures(
                aggregate, thresholds, clip, entry_meta, worst=worst, rows=rows))
            expected_evidence_failures.extend(
                perf_evidence_failures(None, perf, thresholds, clip))
        else:
            clip_evidence, clip_regressions = score_baseline_comparison(
                aggregate, perf, rows, worst, entry_meta, baseline_manifests[clip],
                thresholds, clip, skip_perf_regressions=bool(meta.get("gpu_contention")))
            expected_evidence_failures.extend(clip_evidence)
            expected_regressions.extend(clip_regressions)

    expected["issues"] = expected_issues
    expected["hard_failures"] = expected_hard_failures
    expected["evidence_failures"] = expected_evidence_failures
    expected["regressions"] = expected_regressions
    expected["verdict"] = (
        "hard_failures" if expected_hard_failures else
        "evidence_failures" if expected_evidence_failures else
        "comparison_only" if run_kind == "comparison-only" else
        "regressions" if expected_regressions else "pass")
    bind_training_labels_to_evidence_gate(expected, thresholds)

    for clip in clips:
        for key in ("aggregate", "perf_ms", "worst_frame", "frames", "label_summary"):
            _require_matching_result(
                clips[clip].get(key), expected["clips"][clip].get(key),
                f"clips.{clip}.{key}")
    for key in ("issues", "hard_failures"):
        _require_matching_result(results.get(key), expected.get(key), key)
    for key in ("evidence_failures", "regressions", "verdict"):
        _require_matching_result(results.get(key), expected.get(key), key)
    for key in ("training_label_gate", "label_context_sha256"):
        _require_matching_result(
            meta.get(key), expected["meta"].get(key), f"meta.{key}")
    return {
        "passed": True,
        "clips": list(clips),
        "frame_count": sum(len(entry["frames"]) for entry in expected["clips"].values()),
    }


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
    build_env = production_subprocess_env()
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
                cmd, cwd=build_dir, capture_output=True, text=True, timeout=900,
                env=production_subprocess_env())
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
    label_manifest = training_label_manifest(thresholds)
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
    label_sha = label_contract_sha()
    metric_runtime = metric_runtime_provenance()
    try:
        clip_hashes = {clip: sha1_dir(os.path.join(clips_dir, clip)) for clip in clips}
    except ValueError as exc:
        fail(str(exc))
    baseline_manifests = {}
    baseline_snapshot = None
    if not args.update_baselines and not args.comparison_only:
        required_baseline_context = baseline_required_context({
            "mode": "profile",
            "suite": args.suite,
            "model": expected_model,
            "profile": expected_config_profile,
            "eval_schema": EVAL_SCHEMA,
            "depth_step": depth_step,
            "depth_compensation": depth_compensation,
            "conf_sha256": conf_sha,
            "metric_sha256": metric_sha,
            "label_contract_sha256": label_sha,
            "metric_runtime": metric_runtime,
        })
        try:
            baseline_manifests = preflight_baselines(
                base_dir, clips, required_baseline_context, clip_hashes)
            baseline_snapshot = build_baseline_snapshot(base_dir, baseline_manifests)
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
        "label_contract_sha256": label_sha,
        "metric_runtime": metric_runtime,
        "training_labels": label_manifest,
        "executable_sha256": file_sha256(exe),
        "runtime_shader_sha256": shader_sha,
        **model_artifacts,
        "gpu_contention": contention,
        "run_kind": ("baseline-update" if args.update_baselines else
                     "comparison-only" if args.comparison_only else "baseline-gated"),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "run_name": label,
    }
    if baseline_snapshot is not None:
        meta["baseline_snapshot_sha256"] = sha256_json(baseline_snapshot)
    results, regressions, issues, hard_failures = {}, [], [], []
    scored_artifact_hashes = {}
    evidence_failures, baseline_updates = [], {}
    sbsbench.enable_reusable_spatial_executor()
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
            r = subprocess.run(
                cmd, cwd=args.build_dir, capture_output=True, text=True, timeout=900,
                env=production_subprocess_env())
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
            "schema": 16,
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
        source_by_id = sbsbench.indexed_files(
            os.path.join(clip_dir, "frame_*.*"), "frame_")
        source_ids = set(source_by_id)
        clip_meta["source_frame_count"] = len(source_ids)
        sbs_by_id = sbsbench.indexed_files(os.path.join(out_dir, "sbs_*.png"), "sbs_")
        sbs_ids = set(sbs_by_id)
        depth_ids = set(sbsbench.indexed_files(os.path.join(out_dir, "depth_*.png"), "depth_"))
        raw_ids = set(sbsbench.indexed_files(os.path.join(out_dir, "raw_*.f32"), "raw_"))
        mask_by_id = sbsbench.indexed_files(
            os.path.join(out_dir, "warp_mask_*.png"), "warp_mask_")
        mask_ids = set(mask_by_id)
        ema_mask_ids = set(sbsbench.indexed_files(
            os.path.join(out_dir, "ema_mask_*.png"), "ema_mask_"))
        mapping_by_id = sbsbench.indexed_files(
            os.path.join(out_dir, "warp_map_*.f32"), "warp_map_")
        mapping_ids = set(mapping_by_id)
        if (contract.get("warp_mask") != {
                "red": "forward_disocclusion_before_fill"}):
            fail(f"{clip}: missing/unknown warp-mask channel contract")
        if (not source_ids or source_ids != sbs_ids or source_ids != depth_ids
                or source_ids != raw_ids or source_ids != mask_ids
                or source_ids != mapping_ids):
            fail(f"{clip}: artifact frame-id mismatch source={sorted(source_ids)} "
                 f"sbs={sorted(sbs_ids)} depth={sorted(depth_ids)} raw={sorted(raw_ids)} "
                 f"warp_mask={sorted(mask_ids)} warp_map={sorted(mapping_ids)}")
        if expected_ema_edge_change > 0.0 and ema_mask_ids != source_ids:
            fail(f"{clip}: incomplete EMA motion-mask artifacts: {sorted(ema_mask_ids)}")
        if expected_ema_edge_change <= 0.0 and ema_mask_ids:
            fail(f"{clip}: unexpected EMA motion-mask artifacts while feature is disabled")
        if not os.path.exists(os.path.join(out_dir, "raw_shape.json")):
            fail(f"{clip}: raw_shape.json missing")
        expected_mapping_contract = {
            "file_pattern": "warp_map_<frame-id>.f32",
            "shape_contract": "warp_map_shape.json",
            "dtype": "float32-le",
            "layout": "row-major",
            "channels": ["raw_reproject_source_u_normalized"],
            "live_sample_transform": (
                "clamp(raw_reproject_source_u_normalized, 0, 1)"),
            "validity_companion": ("warp_mask_<frame-id>.png:red="
                                   "forward_disocclusion_before_fill; content validity derives "
                                   "from warp_map_shape.json"),
        }
        if contract.get("warp_mapping") != expected_mapping_contract:
            fail(f"{clip}: missing/unknown exact warp-mapping contract")
        mapping_shape_path = os.path.join(out_dir, "warp_map_shape.json")
        try:
            with open(mapping_shape_path, encoding="utf-8") as fh:
                mapping_shape = json.load(fh)
        except (OSError, ValueError) as exc:
            fail(f"{clip}: invalid warp_map_shape.json: {exc}")
        expected_shape_fields = {
            "schema": 1,
            "dtype": expected_mapping_contract["dtype"],
            "layout": expected_mapping_contract["layout"],
            "channels": expected_mapping_contract["channels"],
            "validity": {
                "content": ("derive from content_scale_x/content_scale_y and packed output "
                            "coordinate"),
                "forward_coverage": "warp_mask_<frame-id>.png red == 0 inside content",
            },
            "live_sample_source_u_normalized": (
                "clamp(raw_reproject_source_u_normalized, 0, 1)"),
            "derived_inverse_displacement_output_eye_px": (
                "(raw_reproject_source_u_normalized - "
                "aspect_fitted_unwarped_source_u) * "
                "content_scale_x * eye_width"),
            "derived_signed_binocular_disparity_px": (
                "invert both eye maps at common source-U samples; x_right - x_left"),
        }
        shape_mismatch = {key: (expected, mapping_shape.get(key))
                          for key, expected in expected_shape_fields.items()
                          if mapping_shape.get(key) != expected}
        width = mapping_shape.get("width")
        height = mapping_shape.get("height")
        eye_width = mapping_shape.get("eye_width")
        eye_height = mapping_shape.get("eye_height")
        source_width = mapping_shape.get("source_width")
        source_height = mapping_shape.get("source_height")
        scale_x = mapping_shape.get("content_scale_x")
        scale_y = mapping_shape.get("content_scale_y")
        valid_geometry = (isinstance(width, int) and width > 0 and
                          isinstance(height, int) and height > 0 and
                          isinstance(eye_width, int) and eye_width * 2 == width and
                          eye_height == height and
                          isinstance(source_width, int) and source_width > 0 and
                          isinstance(source_height, int) and source_height > 0 and
                          isinstance(scale_x, (int, float)) and 0.0 < scale_x <= 1.0 and
                          isinstance(scale_y, (int, float)) and 0.0 < scale_y <= 1.0)
        if shape_mismatch or not valid_geometry:
            fail(f"{clip}: warp-map shape contract mismatch: fields={shape_mismatch} "
                 f"geometry={mapping_shape}")
        try:
            source_sizes = image_size_set(source_by_id.values())
            sbs_sizes = image_size_set(sbs_by_id.values())
            mask_sizes = image_size_set(mask_by_id.values())
        except OSError as exc:
            fail(f"{clip}: cannot decode artifact geometry for warp-map validation: {exc}")
        expected_source_size = (source_width, source_height)
        expected_output_size = (width, height)
        source_aspect = source_width / source_height
        eye_aspect = eye_width / eye_height
        expected_scale_x = source_aspect / eye_aspect if eye_aspect > source_aspect else 1.0
        expected_scale_y = eye_aspect / source_aspect if eye_aspect < source_aspect else 1.0
        dimensions_match = (source_sizes == {expected_source_size} and
                            sbs_sizes == {expected_output_size} and
                            mask_sizes == {expected_output_size})
        scales_match = (abs(scale_x - expected_scale_x) <= 1e-5 and
                        abs(scale_y - expected_scale_y) <= 1e-5)
        if not dimensions_match or not scales_match:
            fail(f"{clip}: warp-map geometry is not bound to decoded artifacts: "
                 f"source={sorted(source_sizes)} expected={expected_source_size}, "
                 f"sbs={sorted(sbs_sizes)} mask={sorted(mask_sizes)} "
                 f"expected={expected_output_size}, scales={(scale_x, scale_y)} "
                 f"expected_scales={(expected_scale_x, expected_scale_y)}")
        expected_mapping_bytes = width * height * 4
        for frame_id in sorted(mapping_ids):
            mapping_path = mapping_by_id[frame_id]
            try:
                mapping_bytes = os.path.getsize(mapping_path)
            except OSError as exc:
                fail(f"{clip}: cannot stat warp map {mapping_path}: {exc}")
            if mapping_bytes != expected_mapping_bytes:
                fail(f"{clip}: warp-map byte size mismatch for {frame_id}: "
                     f"{mapping_bytes} != {expected_mapping_bytes}")
        # Carry the clip's own metadata (scene name/description) into results so the run dir is
        # self-describing and the report can label clips without the source clips dir.
        clip_meta.update(published_clip_metadata(source_clip_meta[clip]))
        artifact_sha = scored_artifact_sha256(out_dir)
        scored_artifact_hashes[clip] = artifact_sha
        clip_meta["scored_artifact_sha256"] = artifact_sha
        # `suite` is evaluator provenance ("core" or "extended") and is used to select the
        # committed baseline namespace. Prepared-dataset manifests may also carry their own suite
        # revision (for example "extended-v3"); keep that useful label under a distinct key so it
        # cannot overwrite the run contract when baseline metadata is merged below.

        print(f"[{clip}] scoring...", flush=True)
        try:
            rows, agg = sbsbench.measure_sequence(out_dir, clip_dir)
        except ValueError as exc:
            fail(f"{clip}: {exc}")
        agg = sbsbench.filter_aggregate_by_evidence(
            rows, agg, thresholds["metrics"], clip_meta)
        perf_p = os.path.join(out_dir, "sbs_perf.json")
        try:
            perf = load_perf_metrics(perf_p)
        except ValueError as exc:
            fail(f"{clip}: {exc}")
        if args.comparison_only:
            evidence_failures.extend(primary_evidence_failures(
                agg, thresholds, clip, clip_meta, worst=None, rows=rows))
            evidence_failures.extend(perf_evidence_failures(None, perf, thresholds, clip))

        worst, clip_issues, clip_hard_failures = score_clip_gates(
            rows, agg, thresholds, clip_meta)
        issues.extend({"clip": clip, **item} for item in clip_issues)
        hard_failures.extend({"clip": clip, **item} for item in clip_hard_failures)

        frame_records = build_frame_records(rows, thresholds, clip_meta)
        entry = {
            "aggregate": agg, "perf_ms": perf, "meta": clip_meta,
            "worst_frame": worst, "frames": frame_records,
            "label_summary": summarize_frame_labels(frame_records, thresholds),
        }
        results[clip] = entry

        # Regression gate vs baseline. A baseline is only valid for the exact frames it was made
        # from: if the clip content changed, gating against it is meaningless -- skip it loudly
        # instead of silently comparing apples to oranges.
        bp = os.path.join(base_dir, clip + ".json")
        if not args.update_baselines and not args.comparison_only:
            base = baseline_manifests[clip]
            clip_evidence, clip_regressions = score_baseline_comparison(
                agg, perf, rows, worst, clip_meta, base, thresholds, clip,
                skip_perf_regressions=contention)
            evidence_failures.extend(clip_evidence)
            regressions.extend(clip_regressions)

        if args.update_baselines:
            evidence_failures.extend(primary_evidence_failures(
                agg, thresholds, clip, clip_meta, worst=worst, rows=rows))
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
    meta["scored_artifact_sha256"] = scored_artifact_hashes
    bind_training_labels_to_evidence_gate(out, thresholds)
    if baseline_snapshot is not None:
        with open(os.path.join(out_root, BASELINE_SNAPSHOT_FILE), "w",
                  encoding="utf-8") as snapshot_file:
            json.dump(baseline_snapshot, snapshot_file, indent=2)
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
        report_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "generate_report.py"),
                      control_dir, out_root, report_path]
        if args.report_allow_config_diff:
            report_cmd.append("--allow-config-diff")
        if args.report_allow_model_diff:
            report_cmd.append("--allow-model-diff")
        if args.report_allow_depth_step_diff:
            report_cmd.append("--allow-depth-step-diff")
        if args.report_allow_executable_diff:
            report_cmd.append("--allow-executable-diff")
        # Scoring is complete. Do not retain the evaluator's 24 idle workers while the report
        # child creates its own pixel-verification pool.
        sbsbench.disable_reusable_spatial_executor()
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
