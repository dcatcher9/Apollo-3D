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
from collections import deque
from concurrent.futures import ProcessPoolExecutor
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
DEPTH_MODELS_DIR = os.path.join(REPO, "tools", "depth_models")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, DEPTH_MODELS_DIR)
import build_clip_hash_manifest as clip_hashes  # noqa: E402
import artistic_geometry_contract  # noqa: E402
import native_hdr_capture  # noqa: E402
import runtime_scene_evidence  # noqa: E402
import multiscale_batch  # noqa: E402
import sbsbench  # noqa: E402  (metric implementations)
import sbs_harness_contract as harness_contract  # noqa: E402

EVAL_SCHEMA = 31  # target-only ordinal evidence and exact artifact identities; harness 28
HARNESS_SCHEMA = harness_contract.HARNESS_SCHEMA
FRAME_GATE_EVIDENCE_SCHEMA = 1
FRAME_GATE_EVIDENCE_CONTRACT = "apollo-full-frame-gate-evidence-v1"
SELECTED_FRAME_GATE_EVIDENCE_CONTRACT = "apollo-target-frame-gate-evidence-v2"
FULL_FRAME_GATE_OUTPUT_SELECTION_CONTRACT = "all-source-frames-consecutive-v1"
SELECTED_FRAME_GATE_OUTPUT_SELECTION_CONTRACT = (
    "authenticated-label-targets-only-v2"
)
FRAME_GATE_EVIDENCE_FILENAME = "frame_gate_evidence.jsonl"

SCORE_WORKER_THREAD_ENV = (
    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS", "OPENCV_FOR_THREADS_NUM",
)
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
WINDOWS_RESERVED_COMPONENTS = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}

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
    "metric_preview_encoding", "hdr_source_kind", "hdr_input_scale",
    "sdr_white_level_raw",
    "content_scale_x", "content_scale_y", "disparity_raster_width",
    "disparity_raster_height", "artifact_mode", "warp_disparity",
    "warp_unclamped_disparity", "artistic_disparity_contract",
    "artistic_full_clamp_abs", "warp_mask",
)
LABEL_SELECTION_CONTEXT_FIELDS = (
    "output_selection_mode", "label_frame_ids", "output_selected_frame_ids",
    "output_label_frames_sha256",
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
CLIP_META_FIELDS = frozenset({
    "name", "description", "expected_flat", "gt_depth_kind",
    "required_gt_depth", "required_gt_flow", "required_gt_stereo",
    "dataset", "homepage", "citation", "license_note", "suite",
    "content_type", "source_url", "source_window", "source_artifacts",
})


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


def configure_score_worker_threads():
    """Prevent each scoring process from creating its own nested CPU thread pool."""
    for name in SCORE_WORKER_THREAD_ENV:
        os.environ[name] = "1"


def _initialize_score_worker():
    configure_score_worker_threads()
    cv2 = sys.modules.get("cv2")
    if cv2 is not None:
        cv2.setNumThreads(1)


def capture_score_worker_environment():
    """Capture exactly the parent variables temporarily changed for worker startup."""
    return {name: os.environ.get(name) for name in SCORE_WORKER_THREAD_ENV}


def restore_score_worker_environment(environment):
    """Restore the caller environment even when harnessing or scoring aborts."""
    for name, value in environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def score_clip_artifacts(seq_dir, frames_dir, expected_flat,
                         common_artifact_dir=None):
    """Process-pool entry point: metric calculation only; gating remains in the parent."""
    return sbsbench.measure_sequence(
        seq_dir, frames_dir, expected_flat=expected_flat,
        common_artifact_dir=common_artifact_dir,
    )


class ScoreWorkerError(RuntimeError):
    def __init__(self, clip, error):
        super().__init__(
            f"{clip}: scoring worker failed ({type(error).__name__}): {error}"
        )
        self.clip = clip


class BoundedOrderedScoreQueue:
    """Bounded future queue that always exposes results in producer order."""

    def __init__(self, executor, max_outstanding):
        if max_outstanding < 1:
            raise ValueError("max_outstanding must be at least 1")
        self.executor = executor
        self.max_outstanding = max_outstanding
        self.pending = deque()

    @property
    def outstanding(self):
        return len(self.pending)

    def _collect_next(self):
        clip, context, future = self.pending.popleft()
        try:
            rows, aggregate = future.result()
        except Exception as exc:
            raise ScoreWorkerError(clip, exc) from exc
        return clip, context, rows, aggregate

    def submit(self, clip, context, seq_dir, frames_dir, expected_flat,
               common_artifact_dir=None):
        """Submit one clip, collecting the oldest first when the bound is full."""
        completed = []
        if self.outstanding >= self.max_outstanding:
            completed.append(self._collect_next())
        try:
            if common_artifact_dir is None:
                # Preserve the public worker call shape for ordinary runs and
                # existing out-of-process integrations.
                future = self.executor.submit(
                    score_clip_artifacts, seq_dir, frames_dir, expected_flat,
                )
            else:
                future = self.executor.submit(
                    score_clip_artifacts, seq_dir, frames_dir, expected_flat,
                    common_artifact_dir,
                )
        except Exception as exc:
            raise ScoreWorkerError(clip, exc) from exc
        self.pending.append((clip, context, future))
        return completed

    def drain(self):
        while self.pending:
            yield self._collect_next()

    def cancel_pending(self):
        """Cancel queued work and forget all futures during exceptional cleanup."""
        while self.pending:
            _clip, _context, future = self.pending.popleft()
            future.cancel()


def validate_path_component(value, kind):
    """Require a portable, non-special, single filesystem component."""
    if (not isinstance(value, str) or value.endswith(".") or
            not SAFE_COMPONENT_RE.fullmatch(value)):
        raise ValueError(f"invalid {kind}: {value!r}")
    stem = value.split(".", 1)[0].upper()
    if stem in WINDOWS_RESERVED_COMPONENTS:
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


def contained_component(root, value, kind):
    """Resolve a validated component and reject lexical or resolved-root escapes."""
    value = validate_path_component(value, kind)
    absolute_root = os.path.abspath(root)
    candidate = os.path.abspath(os.path.join(absolute_root, value))
    real_root = os.path.realpath(absolute_root)
    real_candidate = os.path.realpath(candidate)
    try:
        lexical_contained = os.path.commonpath([absolute_root, candidate]) == absolute_root
        resolved_contained = os.path.commonpath([real_root, real_candidate]) == real_root
    except ValueError:
        lexical_contained = resolved_contained = False
    if not lexical_contained or not resolved_contained:
        raise ValueError(f"{kind} escapes its configured root: {value!r}")
    return candidate


def validate_clip_selection(clips_dir, clips):
    """Validate exact unique clip names before a dict or filesystem join can hide them."""
    names = list(clips)
    if not names:
        raise ValueError("clip selection is empty")
    canonical = []
    for name in names:
        clip_dir = contained_component(clips_dir, name, "clip name")
        canonical.append(os.path.normcase(os.path.realpath(clip_dir)))
        if not os.path.isdir(clip_dir):
            raise ValueError(f"clip directory is missing: {name}")
    if len(canonical) != len(set(canonical)):
        raise ValueError("clip selection contains duplicates")
    return names


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
    """Retain the public legacy helper while sharing one manifest identity contract."""
    return clip_hashes.sha1_dir(path)


def resolve_clip_hashes(clips_dir, clips, verify_manifest=False):
    """Resolve clip identities from a valid frozen manifest or direct content reads."""

    try:
        clips = validate_clip_selection(clips_dir, clips)
    except ValueError as error:
        raise clip_hashes.ClipHashManifestError(str(error)) from error
    manifest_path = os.path.join(clips_dir, clip_hashes.MANIFEST_NAME)
    if os.path.isfile(manifest_path):
        manifest_sha256 = sha256_file(manifest_path)
        identities = clip_hashes.verify_selected_clips(
            manifest_path, clips_dir, clips, full=verify_manifest
        )
        if sha256_file(manifest_path) != manifest_sha256:
            raise clip_hashes.ClipHashManifestError(
                "clip hash manifest changed during verification"
            )
        return identities, {
            "clip_hash_source": "manifest",
            "clip_hash_verification": "full" if verify_manifest else "stat",
            "clip_hash_manifest": os.path.abspath(manifest_path),
            "clip_hash_manifest_sha256": manifest_sha256,
        }
    return {
        clip: sha1_dir(contained_component(clips_dir, clip, "clip name"))
        for clip in clips
    }, {
        "clip_hash_source": "direct",
        "clip_hash_verification": "direct-content",
        "clip_hash_manifest": None,
        "clip_hash_manifest_sha256": None,
    }


def revalidate_clip_hashes(clips_dir, clips, expected_identities,
                           expected_provenance, verify_manifest=False):
    """Fail if selected source semantics or their frozen manifest changed mid-run."""
    current_identities, current_provenance = resolve_clip_hashes(
        clips_dir, clips, verify_manifest
    )
    if current_identities != expected_identities:
        raise clip_hashes.ClipHashManifestError(
            "clip source identities changed during evaluation"
        )
    provenance_fields = (
        "clip_hash_source", "clip_hash_verification", "clip_hash_manifest",
        "clip_hash_manifest_sha256",
    )
    mismatches = {
        field: (expected_provenance.get(field), current_provenance.get(field))
        for field in provenance_fields
        if expected_provenance.get(field) != current_provenance.get(field)
    }
    if mismatches:
        raise clip_hashes.ClipHashManifestError(
            f"clip hash provenance changed during evaluation: {mismatches}"
        )


def expected_hdr_source_kind(extra):
    """Resolve the harness source family without exposing it to the policy model."""
    simulated = "--simulate-hdr" in extra
    native_pq = "--native-hdr-scrgb" in extra
    if simulated and native_pq:
        raise ValueError(
            "--simulate-hdr and --native-hdr-scrgb are mutually exclusive"
        )
    if native_pq and any(
            flag in extra for flag in ("--hdr-scale", "--sdr-white-level-raw")):
        raise ValueError(
            "--native-hdr-scrgb cannot carry an SDR white scale or white level"
        )
    if native_pq:
        return harness_contract.HDR_SOURCE_NATIVE_PQ
    if simulated:
        return harness_contract.HDR_SOURCE_SIMULATED
    return harness_contract.HDR_SOURCE_SDR


def validate_hdr_contract_provenance(contract, expected_kind, origin):
    """Validate color/source/preview provenance and return normalized HDR fields."""
    actual_kind = contract.get("hdr_source_kind")
    if actual_kind != expected_kind:
        raise RuntimeError(
            f"{origin}: harness HDR source kind {actual_kind!r} does not match "
            f"requested {expected_kind!r}"
        )
    color_mode = contract.get("color_mode")
    if color_mode not in {
            harness_contract.COLOR_MODE_SDR,
            harness_contract.COLOR_MODE_LINEAR_SDR,
            harness_contract.COLOR_MODE_HDR}:
        raise RuntimeError(f"{origin}: invalid/missing input color mode")
    metric_preview_encoding = harness_contract.validate_metric_preview_encoding(
        color_mode,
        contract.get("metric_preview_encoding"),
        origin,
        hdr_source_kind=actual_kind,
    )
    hdr_scale = contract.get("hdr_input_scale")
    white_raw = contract.get("sdr_white_level_raw")
    scale_is_number = (
        isinstance(hdr_scale, (int, float)) and
        not isinstance(hdr_scale, bool) and
        math.isfinite(float(hdr_scale))
    )
    white_is_int = isinstance(white_raw, int) and not isinstance(white_raw, bool)

    if expected_kind == harness_contract.HDR_SOURCE_SIMULATED:
        if (color_mode != harness_contract.COLOR_MODE_HDR or
                not scale_is_number or float(hdr_scale) <= 0.0 or
                not white_is_int or white_raw <= 0 or
                abs(float(hdr_scale) - white_raw / 1000.0) > 1e-9):
            raise RuntimeError(
                f"{origin}: invalid/mismatched simulated-HDR SDR-white provenance"
            )
    elif expected_kind == harness_contract.HDR_SOURCE_NATIVE_PQ:
        if (color_mode != harness_contract.COLOR_MODE_HDR or
                not scale_is_number or float(hdr_scale) != 0.0 or
                not white_is_int or white_raw != 0):
            raise RuntimeError(
                f"{origin}: native-PQ HDR must carry zero SDR-white provenance"
            )
    elif expected_kind == harness_contract.HDR_SOURCE_SDR:
        if (color_mode not in {
                harness_contract.COLOR_MODE_SDR,
                harness_contract.COLOR_MODE_LINEAR_SDR} or
                not scale_is_number or float(hdr_scale) != 0.0 or
                not white_is_int or white_raw != 0):
            raise RuntimeError(
                f"{origin}: native SDR carries invalid HDR provenance"
            )
    else:
        raise RuntimeError(f"{origin}: unsupported HDR source kind {expected_kind!r}")
    return metric_preview_encoding, float(hdr_scale), white_raw


def authenticate_native_hdr_clip(clip_dir, source_by_id, full=False):
    """Authenticate exact scRGB sidecars and bind them one-to-one to previews."""
    try:
        authentication = native_hdr_capture.validate_clip(clip_dir, full=full)
    except RuntimeError as error:
        raise RuntimeError(f"native-HDR source authentication failed: {error}") from error
    frames = authentication.get("frames")
    if not isinstance(frames, dict) or set(frames) != set(source_by_id):
        raise RuntimeError(
            "native-HDR frame_model_sources.json does not exactly cover source previews"
        )
    for frame_id, source_path in source_by_id.items():
        row = frames.get(frame_id)
        preview_path = row.get("preview_path") if isinstance(row, dict) else None
        if preview_path is None or os.path.realpath(os.fspath(preview_path)) != os.path.realpath(
                source_path):
            raise RuntimeError(
                f"native-HDR frame {frame_id} preview identity differs from evaluator source"
            )
    return {
        "manifest": authentication["manifest"],
        "manifest_sha256": authentication["manifest_sha256"],
        "content_sha256": authentication["content_sha256"],
        "width": authentication["width"],
        "height": authentication["height"],
        "frame_count": authentication["frame_count"],
    }


def load_label_frame_manifest(clip_dir, source_frame_ids):
    """Load and authenticate a clip's generic sparse-output selection."""
    path = os.path.join(clip_dir, "label_frames.json")
    frame_ids = sbsbench.load_label_frame_ids(clip_dir)
    if frame_ids is None:
        raise ValueError(f"missing label-frame manifest {path}")
    missing = sorted(set(frame_ids) - set(source_frame_ids))
    if missing:
        raise ValueError(f"label_frames.json references missing source frames: {missing}")
    return frame_ids, sha256_file(path)


def load_optional_clip_metadata(clip_dir):
    """Load evaluator metadata when present; malformed evidence contracts never fail open."""
    payload = sbsbench.load_clip_meta(clip_dir)
    return {key: value for key, value in payload.items() if key in CLIP_META_FIELDS}


def resolve_output_selection(clip_dir, ordered_source_ids, output_interval,
                             output_gt_right_only, output_label_frames):
    """Return the exact output IDs, mode and optional manifest identity for one clip."""
    if output_label_frames:
        if output_gt_right_only:
            raise ValueError(
                "--output-label-frames and --output-gt-right-only are mutually exclusive"
            )
        if output_interval != 1:
            raise ValueError("--output-label-frames requires --output-every 1")
        label_frame_ids, manifest_sha256 = load_label_frame_manifest(
            clip_dir, ordered_source_ids
        )
        return {
            "mode": "label-frames",
            "label_frame_ids": label_frame_ids,
            "output_frame_ids": list(label_frame_ids),
            "label_frames_sha256": manifest_sha256,
        }

    frame_ids = list(ordered_source_ids[::output_interval])
    mode = "interval"
    if output_gt_right_only:
        mode = "gt-right"
        gt_right_ids = set(sbsbench.indexed_files(
            os.path.join(clip_dir, "gt_right", "frame_*.*"), "frame_"
        ))
        frame_ids = [frame_id for frame_id in frame_ids if frame_id in gt_right_ids]
    if not frame_ids:
        raise ValueError("output selection contains no source frames")
    return {
        "mode": mode,
        "label_frame_ids": [],
        "output_frame_ids": frame_ids,
        "label_frames_sha256": "",
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
    return clip_meta.get("expected_flat") is True and spec.get("axis") == "stereo"


def invalidate_publication_file(path, description):
    """Remove an old published result without ever recursing into a directory."""
    if not os.path.lexists(path):
        return
    if os.path.isdir(path) and not os.path.islink(path):
        raise ValueError(f"{description} path is a directory: {path}")
    os.unlink(path)


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
    try:
        clips = validate_clip_selection(clips_dir, clips)
        manifest_path = contained_component(root, "manifest.json", "override manifest")
    except ValueError as exc:
        fail(str(exc))
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
        clip_dir = contained_component(clips_dir, clip, "clip name")
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
        override_clip_dir = contained_component(root, clip, "override clip name")
        actual_ids = sorted(sbsbench.indexed_files(
            os.path.join(override_clip_dir, "depth_*.png"), "depth_"))
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


def canonical_json_bytes(value):
    """Serialize one authenticated JSONL record without platform-dependent whitespace."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )


def validate_full_frame_gate_coverage(source_frame_ids, output_selection, rows=None):
    """Require the exact complete consecutive source sequence for ordinal-oracle evidence."""
    source_frame_ids = list(source_frame_ids)
    if (not source_frame_ids or any(
            not isinstance(frame_id, int) or isinstance(frame_id, bool)
            for frame_id in source_frame_ids)):
        raise ValueError("frame-gate evidence requires nonempty integer source-frame identities")
    if source_frame_ids != sorted(set(source_frame_ids)):
        raise ValueError("frame-gate evidence source-frame identities must be unique and ordered")
    expected = list(range(source_frame_ids[0], source_frame_ids[-1] + 1))
    if source_frame_ids != expected:
        raise ValueError(
            "frame-gate evidence requires consecutive source-frame identities; "
            f"source={source_frame_ids}"
        )
    if output_selection.get("mode") != "interval":
        raise ValueError(
            "frame-gate evidence rejects sparse/GT-only output selection; use --output-every 1"
        )
    if output_selection.get("output_frame_ids") != source_frame_ids:
        raise ValueError(
            "frame-gate evidence requires every source frame in exact order; "
            f"source={source_frame_ids}, output={output_selection.get('output_frame_ids')}"
        )
    if rows is not None:
        row_ids = [row.get("_frame_id") for row in rows]
        if row_ids != source_frame_ids:
            raise ValueError(
                "frame-gate evidence requires one ordered metric row per source frame; "
                f"source={source_frame_ids}, rows={row_ids}"
            )
    return source_frame_ids


def frame_id_sequence_sha256(frame_ids):
    """Authenticate an ordered frame-ID sequence without relying on filename formatting."""
    return hashlib.sha256(canonical_json_bytes(list(frame_ids))).hexdigest()


def validate_selected_frame_gate_coverage(source_frame_ids, output_selection, rows=None):
    """Require authenticated target-only label-frame evidence."""
    if not isinstance(source_frame_ids, (list, tuple)):
        raise ValueError("selected frame-gate full source identities must be a sequence")
    source_frame_ids = list(source_frame_ids)
    if (not source_frame_ids or any(
            not isinstance(frame_id, int) or isinstance(frame_id, bool)
            for frame_id in source_frame_ids)):
        raise ValueError(
            "selected frame-gate evidence requires nonempty integer source identities"
        )
    if source_frame_ids != sorted(set(source_frame_ids)):
        raise ValueError(
            "selected frame-gate evidence source-frame identities must be unique and ordered"
        )
    expected_source = list(range(source_frame_ids[0], source_frame_ids[-1] + 1))
    if source_frame_ids != expected_source:
        raise ValueError(
            "selected frame-gate evidence requires a consecutive full source sequence; "
            f"source={source_frame_ids}"
        )
    if output_selection.get("mode") != "label-frames":
        raise ValueError(
            "selected frame-gate evidence requires authenticated --output-label-frames"
        )

    label_frame_ids = output_selection.get("label_frame_ids")
    output_frame_ids = output_selection.get("output_frame_ids")
    if (not isinstance(label_frame_ids, list) or not label_frame_ids or any(
                not isinstance(frame_id, int) or isinstance(frame_id, bool)
                for frame_id in label_frame_ids) or
            label_frame_ids != sorted(set(label_frame_ids))):
        raise ValueError(
            "selected frame-gate label identities must be nonempty, unique and ordered"
        )
    if (not isinstance(output_frame_ids, list) or not output_frame_ids or any(
                not isinstance(frame_id, int) or isinstance(frame_id, bool)
                for frame_id in output_frame_ids) or
            output_frame_ids != sorted(set(output_frame_ids))):
        raise ValueError(
            "selected frame-gate output identities must be nonempty, unique and ordered"
        )
    manifest_sha256 = output_selection.get("label_frames_sha256")
    if (not isinstance(manifest_sha256, str) or
            re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None):
        raise ValueError("selected frame-gate evidence lacks an authenticated label manifest")

    source_ids = set(source_frame_ids)
    missing_targets = sorted(set(label_frame_ids) - source_ids)
    if missing_targets:
        raise ValueError(
            f"selected frame-gate label targets are absent from source: {missing_targets}"
        )
    if output_frame_ids != label_frame_ids:
        raise ValueError(
            "selected frame-gate output must exactly contain label targets only; "
            f"expected={label_frame_ids}, output={output_frame_ids}"
        )
    if rows is not None:
        row_ids = [row.get("_frame_id") for row in rows]
        if row_ids != output_frame_ids:
            raise ValueError(
                "selected frame-gate evidence requires one ordered metric row per selected "
                f"output; output={output_frame_ids}, rows={row_ids}"
            )
    return source_frame_ids, label_frame_ids, output_frame_ids


def target_only_gate_thresholds(thresholds):
    """Exclude metrics that require a neighboring frame from target-only gates."""
    metrics = thresholds.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("threshold contract lacks metrics")
    filtered = {
        name: spec for name, spec in metrics.items()
        if spec.get("temporal_evidence") is not True
    }
    if not filtered or len(filtered) == len(metrics):
        raise ValueError(
            "target-only gate contract lacks explicit temporal metric markers"
        )
    return {**thresholds, "metrics": filtered}


def frame_metric_key(metric, row):
    """Map aggregate percentile names to their same-frame metric without guessing other aliases."""
    if metric in row:
        return metric
    if metric.endswith(("_p50", "_p95")):
        candidate = metric[:-4]
        if candidate in row:
            return candidate
    return None


def frame_gate_metric_evidence(row, ordinal, thresholds):
    """Return every primary/hard frame value and its baseline-independent violations."""
    metric_values = {"primary": {}, "hard": {}}
    violations = []
    for metric, spec in thresholds["metrics"].items():
        role = spec.get("role")
        if role not in metric_values:
            continue
        key = frame_metric_key(metric, row)
        raw_value = row.get(key) if key is not None else None
        value = None
        if (isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool) and
                math.isfinite(float(raw_value))):
            value = float(raw_value)
        metric_values[role][metric] = value

        # Spatial required evidence must exist on every frame. Temporal metrics can be undefined
        # on the first frame or on a low-support pair; their aggregate requirement is already
        # enforced by missing_required_metric_evidence(), and null remains explicit here so a
        # downstream safety selector cannot mistake absent evidence for a pass.
        has_ordinal_hard_bound = (
            "ordinal_hard_min" in spec or "ordinal_hard_max" in spec
        )
        required_now = role == "hard" or has_ordinal_hard_bound or (
            spec.get("required_evidence") is True and "min_frames" not in spec
        )
        if value is None:
            if required_now:
                raise ValueError(
                    f"required per-frame {role} metric {metric!r} is missing/non-finite "
                    f"at ordinal {ordinal}"
                )
            continue

        if role == "hard":
            if "hard_min" in spec and value < spec["hard_min"]:
                violations.append({
                    "metric": metric, "kind": "hard_min", "bound": spec["hard_min"],
                    "value": value,
                })
            if "hard_max" in spec and value > spec["hard_max"]:
                violations.append({
                    "metric": metric, "kind": "hard_max", "bound": spec["hard_max"],
                    "value": value,
                })
        else:
            if "trigger_min" in spec and value < spec["trigger_min"]:
                violations.append({
                    "metric": metric, "kind": "trigger_min", "bound": spec["trigger_min"],
                    "value": value,
                })
            if "trigger" in spec and value > spec["trigger"]:
                violations.append({
                    "metric": metric, "kind": "trigger_max", "bound": spec["trigger"],
                    "value": value,
                })
        # Ordinal training has a separate, explicitly authorized absolute-safety
        # namespace. Existing report triggers remain diagnostic evidence and are
        # never silently promoted into safe-frontier limits.
        if "ordinal_hard_min" in spec and value < spec["ordinal_hard_min"]:
            violations.append({
                "metric": metric, "kind": "ordinal_hard_min",
                "bound": spec["ordinal_hard_min"], "value": value,
            })
        if "ordinal_hard_max" in spec and value > spec["ordinal_hard_max"]:
            violations.append({
                "metric": metric, "kind": "ordinal_hard_max",
                "bound": spec["ordinal_hard_max"], "value": value,
            })
    return metric_values, violations


def build_frame_gate_clip_records(clip, rows, thresholds, context):
    """Build one clip's authenticated full-frame or selected-frame safety records."""
    selection = context["output_selection"]
    if selection.get("mode") == "interval":
        source_frame_ids = validate_full_frame_gate_coverage(
            context["source_frame_ids"], selection, rows
        )
        evidence_frame_ids = source_frame_ids
        label_frame_ids = []
        selection_contract = FULL_FRAME_GATE_OUTPUT_SELECTION_CONTRACT
    elif selection.get("mode") == "label-frames":
        source_frame_ids, label_frame_ids, evidence_frame_ids = (
            validate_selected_frame_gate_coverage(
                context["source_frame_ids"], selection, rows
            )
        )
        selection_contract = SELECTED_FRAME_GATE_OUTPUT_SELECTION_CONTRACT
    else:
        raise ValueError(
            f"{clip}: unsupported frame-gate output selection {selection.get('mode')!r}"
        )
    artifact_paths = context["artifact_paths"]
    if set(artifact_paths) != set(evidence_frame_ids):
        raise ValueError(
            f"{clip}: frame-gate artifacts do not exactly match selected metric rows"
        )
    try:
        geometry = artistic_geometry_contract.canonical_geometry_tuple(context["geometry"])
    except RuntimeError as error:
        raise ValueError(f"{clip}: invalid frame-gate artistic geometry: {error}") from error
    try:
        scene_evidence = runtime_scene_evidence.validate(
            context["runtime_scene_evidence"]
        )
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"{clip}: invalid/missing runtime scene evidence: {error}"
        ) from error
    scene_rows = scene_evidence["frames"]
    scene_frame_ids = [item["source_frame_id"] for item in scene_rows]
    if (scene_evidence["depth_reuse_interval"] != 1 or
            scene_evidence["source_frame_ids"] != source_frame_ids or
            scene_frame_ids != source_frame_ids):
        raise ValueError(
            f"{clip}: runtime scene evidence does not cover every frame at current depth"
        )
    scene_path = context.get("runtime_scene_evidence_path")
    if not isinstance(scene_path, str) or not os.path.isfile(scene_path):
        raise ValueError(f"{clip}: runtime scene evidence artifact is missing")
    geometry_key = artistic_geometry_contract.tuple_key(geometry).encode("utf-8")
    clip_record = {
        "record": "clip",
        "clip": clip,
        "clip_sha1": context["clip_sha1"],
        "harness_contract_sha256": context["harness_contract_sha256"],
        "frame_count": len(evidence_frame_ids),
        "first_frame_id": evidence_frame_ids[0],
        "last_frame_id": evidence_frame_ids[-1],
        "output_selection_contract": selection_contract,
        "expected_flat": context["expected_flat"],
        "geometry_contract": artistic_geometry_contract.GEOMETRY_CONTRACT,
        "geometry_sha256": hashlib.sha256(geometry_key).hexdigest(),
        "geometry": geometry,
        "runtime_scene_contract": runtime_scene_evidence.CONTRACT,
        "runtime_scene_evidence_sha256": sha256_file(scene_path),
        "runtime_scene_count": 1 + max(
            item["runtime_scene_id"] for item in scene_rows
        ),
        "completion_sequence_contract": scene_evidence[
            "completion_sequence_contract"
        ],
        "color": context["color"],
        "pipeline": context["pipeline"],
    }
    if selection_contract == SELECTED_FRAME_GATE_OUTPUT_SELECTION_CONTRACT:
        clip_record.update({
            "full_source_frame_count": len(source_frame_ids),
            "full_source_frame_ids": source_frame_ids,
            "full_source_frame_ids_sha256": frame_id_sequence_sha256(source_frame_ids),
            "label_frame_ids": label_frame_ids,
            "output_selected_frame_ids": evidence_frame_ids,
            "output_label_frames_sha256": selection["label_frames_sha256"],
        })
    records = [clip_record]
    frame_digest = hashlib.sha256()
    source_ordinals = {
        frame_id: ordinal for ordinal, frame_id in enumerate(source_frame_ids)
    }
    for frame_id, row in zip(evidence_frame_ids, rows):
        source_ordinal = source_ordinals[frame_id]
        paths = artifact_paths[frame_id]
        required_artifacts = {"source", "sbs", "depth", "warp_mask", "warp_disparity"}
        missing_artifacts = sorted(required_artifacts - set(paths))
        if missing_artifacts:
            raise ValueError(
                f"{clip} frame {frame_id}: frame-gate artifacts are missing {missing_artifacts}"
            )
        artifact_sha256 = {}
        for name, path in sorted(paths.items()):
            if not os.path.isfile(path):
                raise ValueError(f"{clip} frame {frame_id}: missing {name} artifact {path}")
            artifact_sha256[name] = sha256_file(path)
        metrics, violations = frame_gate_metric_evidence(
            row, source_ordinal, thresholds
        )
        frame_record = {
            "record": "frame",
            "clip": clip,
            "frame_id": frame_id,
            "ordinal": source_ordinal,
            "artifact_sha256": artifact_sha256,
            "metrics": metrics,
            "violations": violations,
            "runtime_scene": scene_rows[source_ordinal],
        }
        frame_digest.update(canonical_json_bytes(frame_record))
        records.append(frame_record)
    records.append({
        "record": "clip_end",
        "clip": clip,
        "frame_count": len(evidence_frame_ids),
        "frame_records_sha256": frame_digest.hexdigest(),
    })
    return records


def write_frame_gate_evidence(path, run_meta, thresholds, clip_record_groups,
                              results_sha256,
                              evidence_contract=FRAME_GATE_EVIDENCE_CONTRACT):
    """Atomically publish a canonical JSONL stream with a digest trailer."""
    if evidence_contract not in {
            FRAME_GATE_EVIDENCE_CONTRACT,
            SELECTED_FRAME_GATE_EVIDENCE_CONTRACT}:
        raise ValueError(f"unknown frame-gate evidence contract: {evidence_contract!r}")
    expected_selection_contract = (
        FULL_FRAME_GATE_OUTPUT_SELECTION_CONTRACT
        if evidence_contract == FRAME_GATE_EVIDENCE_CONTRACT else
        SELECTED_FRAME_GATE_OUTPUT_SELECTION_CONTRACT
    )
    for group in clip_record_groups:
        clip_records = [record for record in group if record.get("record") == "clip"]
        if (len(clip_records) != 1 or
                clip_records[0].get("output_selection_contract") !=
                expected_selection_contract):
            raise ValueError(
                "frame-gate clip selection contract differs from publication header"
            )
    primary = sorted(
        metric for metric, spec in thresholds["metrics"].items()
        if spec.get("role") == "primary"
    )
    hard = sorted(
        metric for metric, spec in thresholds["metrics"].items()
        if spec.get("role") == "hard"
    )
    header = {
        "record": "header",
        "schema": FRAME_GATE_EVIDENCE_SCHEMA,
        "contract": evidence_contract,
        "eval_schema": EVAL_SCHEMA,
        "harness_schema": HARNESS_SCHEMA,
        "metric_sha256": run_meta["metric_sha256"],
        "thresholds_sha256": sha256_file(os.path.join(SCRIPT_DIR, "thresholds.json")),
        "conf_sha256": run_meta["conf_sha256"],
        "clip_hash_manifest_sha256": run_meta.get("clip_hash_manifest_sha256"),
        "clip_set_sha1": run_meta["clip_set_sha1"],
        "results_sha256": results_sha256,
        "run_name": run_meta["run_name"],
        "suite": run_meta["suite"],
        "hdr_source_kind": run_meta["hdr_source_kind"],
        "precomputed_multiscale": run_meta.get("precomputed_multiscale", False),
        "multiscale_batch_manifest_sha256":
            run_meta.get("multiscale_batch_manifest_sha256"),
        "primary_metrics": primary,
        "hard_metrics": hard,
    }
    payload_records = [header]
    for group in clip_record_groups:
        payload_records.extend(group)
    payload_digest = hashlib.sha256()
    payload_bytes = []
    for record in payload_records:
        encoded = canonical_json_bytes(record)
        payload_digest.update(encoded)
        payload_bytes.append(encoded)
    frame_count = sum(record.get("frame_count", 0) for record in payload_records
                      if record.get("record") == "clip")
    trailer = {
        "record": "trailer",
        "payload_record_count": len(payload_records),
        "clip_count": len(clip_record_groups),
        "frame_count": frame_count,
        "payload_sha256": payload_digest.hexdigest(),
    }
    temporary = path + ".tmp"
    try:
        with open(temporary, "wb") as stream:
            for encoded in payload_bytes:
                stream.write(encoded)
            stream.write(canonical_json_bytes(trailer))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256_file(path)


def validate_frame_gate_evidence(path):
    """Read back a frame-gate sidecar and reject noncanonical, incomplete or tampered data."""
    with open(path, "rb") as stream:
        raw_lines = stream.readlines()
    if len(raw_lines) < 3 or any(not line.strip() for line in raw_lines):
        raise ValueError("frame-gate evidence is empty or contains blank records")
    try:
        records = [json.loads(line) for line in raw_lines]
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid frame-gate JSONL: {error}") from error
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("frame-gate evidence records must be JSON objects")
    if any(canonical_json_bytes(record) != raw for record, raw in zip(records, raw_lines)):
        raise ValueError("frame-gate evidence is not canonical JSONL")
    header, trailer = records[0], records[-1]
    if (header.get("record") != "header" or
            header.get("schema") != FRAME_GATE_EVIDENCE_SCHEMA or
            header.get("contract") not in {
                FRAME_GATE_EVIDENCE_CONTRACT,
                SELECTED_FRAME_GATE_EVIDENCE_CONTRACT,
            }):
        raise ValueError("frame-gate evidence header contract is missing or stale")
    if trailer.get("record") != "trailer":
        raise ValueError("frame-gate evidence trailer is missing")
    payload = raw_lines[:-1]
    digest = hashlib.sha256(b"".join(payload)).hexdigest()
    if (trailer.get("payload_record_count") != len(payload) or
            trailer.get("payload_sha256") != digest):
        raise ValueError("frame-gate evidence payload digest/count mismatch")

    selected_publication = (
        header["contract"] == SELECTED_FRAME_GATE_EVIDENCE_CONTRACT
    )
    expected_selection_contract = (
        SELECTED_FRAME_GATE_OUTPUT_SELECTION_CONTRACT if selected_publication else
        FULL_FRAME_GATE_OUTPUT_SELECTION_CONTRACT
    )
    open_clip = None
    expected_frame_ids = []
    source_ordinals = {}
    clip_frames = []
    seen_clips = set()
    clip_count = frame_count = 0
    for record in records[1:-1]:
        kind = record.get("record")
        if kind == "clip":
            if open_clip is not None:
                raise ValueError("frame-gate clip records overlap")
            clip = record.get("clip")
            if not isinstance(clip, str) or not clip or clip in seen_clips:
                raise ValueError("frame-gate clip identity is missing or duplicated")
            if record.get("output_selection_contract") != expected_selection_contract:
                raise ValueError(
                    "frame-gate clip selection contract differs from header"
                )
            if (record.get("runtime_scene_contract") != runtime_scene_evidence.CONTRACT or
                    not isinstance(record.get("runtime_scene_evidence_sha256"), str) or
                    re.fullmatch(
                        r"[0-9a-f]{64}", record["runtime_scene_evidence_sha256"]
                    ) is None):
                raise ValueError("frame-gate runtime-scene identity is missing or stale")

            if selected_publication:
                source_ids = record.get("full_source_frame_ids")
                selection = {
                    "mode": "label-frames",
                    "label_frame_ids": record.get("label_frame_ids"),
                    "output_frame_ids": record.get("output_selected_frame_ids"),
                    "label_frames_sha256": record.get(
                        "output_label_frames_sha256"
                    ),
                }
                source_ids, _label_ids, expected_frame_ids = (
                    validate_selected_frame_gate_coverage(source_ids, selection)
                )
                if (record.get("full_source_frame_count") != len(source_ids) or
                        record.get("full_source_frame_ids_sha256") !=
                        frame_id_sequence_sha256(source_ids)):
                    raise ValueError(
                        "selected frame-gate full-source identity/count mismatch"
                    )
            else:
                first_frame_id = record.get("first_frame_id")
                last_frame_id = record.get("last_frame_id")
                if (not isinstance(first_frame_id, int) or isinstance(first_frame_id, bool) or
                        not isinstance(last_frame_id, int) or isinstance(last_frame_id, bool) or
                        last_frame_id < first_frame_id):
                    raise ValueError("frame-gate full-frame bounds are invalid")
                expected_frame_ids = list(range(first_frame_id, last_frame_id + 1))
                selection = {
                    "mode": "interval",
                    "label_frame_ids": [],
                    "output_frame_ids": expected_frame_ids,
                    "label_frames_sha256": "",
                }
                source_ids = validate_full_frame_gate_coverage(
                    expected_frame_ids, selection
                )
            declared_frame_count = record.get("frame_count")
            if (not isinstance(declared_frame_count, int) or
                    isinstance(declared_frame_count, bool) or
                    declared_frame_count != len(expected_frame_ids) or
                    record.get("first_frame_id") != expected_frame_ids[0] or
                    record.get("last_frame_id") != expected_frame_ids[-1]):
                raise ValueError("frame-gate clip declared frame coverage is inconsistent")
            open_clip = record
            source_ordinals = {
                frame_id: ordinal for ordinal, frame_id in enumerate(source_ids)
            }
            clip_frames = []
            seen_clips.add(clip)
            clip_count += 1
        elif kind == "frame":
            if open_clip is None or record.get("clip") != open_clip.get("clip"):
                raise ValueError("frame-gate frame appears outside its clip")
            frame_id = record.get("frame_id")
            if (not isinstance(frame_id, int) or isinstance(frame_id, bool) or
                    len(clip_frames) >= len(expected_frame_ids) or
                    frame_id != expected_frame_ids[len(clip_frames)]):
                raise ValueError("frame-gate frame identities differ from declared selection")
            source_ordinal = source_ordinals[frame_id]
            scene = record.get("runtime_scene")
            ordinal = record.get("ordinal")
            scene_frame_id = scene.get("source_frame_id") if isinstance(scene, dict) else None
            scene_ordinal = (
                scene.get("source_frame_ordinal") if isinstance(scene, dict) else None
            )
            if (not isinstance(ordinal, int) or isinstance(ordinal, bool) or
                    ordinal != source_ordinal or not isinstance(scene, dict) or
                    not isinstance(scene_frame_id, int) or isinstance(scene_frame_id, bool) or
                    not isinstance(scene_ordinal, int) or isinstance(scene_ordinal, bool) or
                    scene_frame_id != frame_id or scene_ordinal != source_ordinal):
                raise ValueError(
                    "frame-gate frame ordinal/runtime-scene identity differs from full source"
                )
            artifact_sha256 = record.get("artifact_sha256")
            required_artifacts = {"source", "sbs", "depth", "warp_mask", "warp_disparity"}
            if (not isinstance(artifact_sha256, dict) or
                    not required_artifacts.issubset(artifact_sha256) or any(
                        not isinstance(value, str) or
                        re.fullmatch(r"[0-9a-f]{64}", value) is None
                        for value in artifact_sha256.values()
                    )):
                raise ValueError("frame-gate frame artifact identities are incomplete")
            metrics = record.get("metrics")
            if (not isinstance(metrics, dict) or
                    not isinstance(metrics.get("primary"), dict) or
                    not isinstance(metrics.get("hard"), dict) or
                    not isinstance(record.get("violations"), list)):
                raise ValueError("frame-gate metric evidence is malformed")
            clip_frames.append(record)
            frame_count += 1
        elif kind == "clip_end":
            if open_clip is None or record.get("clip") != open_clip.get("clip"):
                raise ValueError("frame-gate clip trailer is unmatched")
            frame_ids = [item["frame_id"] for item in clip_frames]
            clip_end_count = record.get("frame_count")
            if (frame_ids != expected_frame_ids or
                    not isinstance(clip_end_count, int) or
                    isinstance(clip_end_count, bool) or
                    clip_end_count != len(clip_frames) or
                    open_clip.get("frame_count") != len(clip_frames)):
                raise ValueError("frame-gate clip coverage is incomplete or inconsistent")
            frame_lines = [canonical_json_bytes(item) for item in clip_frames]
            if record.get("frame_records_sha256") != hashlib.sha256(
                    b"".join(frame_lines)).hexdigest():
                raise ValueError("frame-gate clip frame digest mismatch")
            open_clip = None
            expected_frame_ids = []
            source_ordinals = {}
            clip_frames = []
        else:
            raise ValueError(f"unknown frame-gate record kind: {kind!r}")
    if open_clip is not None:
        raise ValueError("frame-gate clip is missing its trailer")
    trailer_clip_count = trailer.get("clip_count")
    trailer_frame_count = trailer.get("frame_count")
    if (not isinstance(trailer_clip_count, int) or
            isinstance(trailer_clip_count, bool) or
            not isinstance(trailer_frame_count, int) or
            isinstance(trailer_frame_count, bool) or
            trailer_clip_count != clip_count or trailer_frame_count != frame_count):
        raise ValueError("frame-gate trailer clip/frame count mismatch")
    return records


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
    if getattr(args, "precomputed_multiscale_root", None):
        args.precomputed_multiscale_root = os.path.abspath(
            args.precomputed_multiscale_root
        )
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


def _run_main(score_lifecycle):
    ap = argparse.ArgumentParser(description="Run the offline SBS benchmark over a reproducible clip suite.")
    ap.add_argument("--build-dir", default=os.path.join(REPO, "cmake-build-relwithdebinfo"))
    ap.add_argument("--conf", default=os.path.join(SCRIPT_DIR, "bench.conf"))
    ap.add_argument("--clips", nargs="*", help="clip names (default: all in clips/)")
    ap.add_argument("--suite", choices=["core", "extended"], default="core",
                    help="quick committed suite or prepared public-data suite")
    ap.add_argument("--clips-root", help="override suite source directory")
    ap.add_argument("--baseline-dir", help="override suite baseline directory")
    ap.add_argument("--label", default=None, help="run label (default: timestamp)")
    ap.add_argument("--score-workers", type=int, default=4,
                    help="CPU metric worker processes (default: 4; GPU harness stays sequential)")
    ap.add_argument(
        "--verify-clip-hashes", action="store_true",
        help="fully re-hash content referenced by clip_hash_manifest.json",
    )
    frame_gate_group = ap.add_mutually_exclusive_group()
    frame_gate_group.add_argument(
        "--publish-frame-gates", action="store_true",
        help="publish authenticated all-frame primary/hard evidence for ordinal-oracle work",
    )
    frame_gate_group.add_argument(
        "--publish-selected-frame-gates", action="store_true",
        help=("publish authenticated target-only label-frame gates while binding "
              "the complete source/runtime-scene sequence"),
    )
    ap.add_argument(
        "--precomputed-multiscale-root",
        help=("internal offline path: score one authenticated scale from a "
              "single-inference multiscale harness clip"),
    )
    ap.add_argument(
        "--prevalidated-current-build-sha256",
        help=("internal multiscale driver receipt; skips Ninja only when the "
              "current sunshine.exe has this exact SHA-256"),
    )
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
    publish_gate_evidence = (
        args.publish_frame_gates or args.publish_selected_frame_gates
    )
    if args.score_workers < 1:
        fail("--score-workers must be at least 1")
    if args.comparison_only and args.update_baselines:
        fail("--comparison-only and --update-baselines are mutually exclusive")
    if args.precomputed_multiscale_root and (
            not args.comparison_only or not publish_gate_evidence or
            args.update_baselines or args.report_control or args.report_out):
        fail(
            "--precomputed-multiscale-root is ordinal-oracle-only and requires "
            "--comparison-only with full or selected frame-gate publication and "
            "without report/baseline mutation"
        )
    if (args.prevalidated_current_build_sha256 and
            not args.precomputed_multiscale_root):
        fail(
            "--prevalidated-current-build-sha256 requires "
            "--precomputed-multiscale-root"
        )
    try:
        expected_hdr_kind = expected_hdr_source_kind(args.extra)
    except ValueError as error:
        fail(str(error))
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
    output_label_frames = "--output-label-frames" in args.extra
    if output_label_frames and output_gt_right_only:
        fail("--output-label-frames and --output-gt-right-only are mutually exclusive")
    if output_label_frames and output_interval != 1:
        fail("--output-label-frames requires --output-every 1")
    if args.publish_frame_gates and (
            output_interval != 1 or output_gt_right_only or output_label_frames):
        fail(
            "--publish-frame-gates requires full --output-every 1 interval output; "
            "sparse label/GT-only selection is forbidden"
        )
    if args.publish_selected_frame_gates and (
            output_interval != 1 or output_gt_right_only or not output_label_frames):
        fail(
            "--publish-selected-frame-gates requires --output-label-frames with "
            "--output-every 1 and rejects GT-only selection"
        )
    if publish_gate_evidence and (
            depth_reuse_interval != 1 or depth_override_root):
        fail(
            "frame-gate publication requires uncompensated current-frame depth; "
            "depth reuse/override evidence cannot define an ordinal safety frontier"
        )
    if (publish_gate_evidence and
            "--runtime-scene-evidence" not in args.extra):
        # Gate publications must use the shipping SubjectState cut signal. Sparse metric rows
        # still bind the complete scene sequence; RGB storage boundaries are not resets.
        args.extra.append("--runtime-scene-evidence")
    output_selection_mode = (
        "label-frames" if output_label_frames else
        "gt-right" if output_gt_right_only else "interval"
    )
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
    gate_thresholds = (
        target_only_gate_thresholds(thresholds)
        if args.publish_selected_frame_gates else thresholds
    )
    clips = args.clips or sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(clips_dir, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "frame_*.*")))
    try:
        clips = validate_clip_selection(clips_dir, clips)
    except ValueError as error:
        fail(str(error))
    try:
        clip_set_sha1, clip_hash_provenance = resolve_clip_hashes(
            clips_dir, clips, args.verify_clip_hashes
        )
    except clip_hashes.ClipHashManifestError as error:
        fail(str(error))
    if args.precomputed_multiscale_root and len(clips) != 1:
        fail("authenticated multiscale scoring requires exactly one selected clip")
    depth_override_counts = (validate_depth_override_manifest(
        depth_override_root, clips_dir, clips, depth_reuse_interval, depth_override_all)
        if depth_override_root else {clip: 0 for clip in clips})
    if not args.update_baselines and not args.comparison_only:
        missing_baselines = [
            clip for clip in clips
            if not os.path.exists(contained_component(
                base_dir, clip + ".json", "baseline filename"
            ))
        ]
        if missing_baselines:
            fail(f"missing committed baseline(s) in {base_dir}: {missing_baselines}. "
                 "Use --comparison-only for a matched A/B or --update-baselines after validation.")
    if not os.path.exists(exe):
        fail(f"{exe} not found -- build first (ninja -C cmake-build-relwithdebinfo sunshine)")
    if args.prevalidated_current_build_sha256:
        expected_executable_sha256 = args.prevalidated_current_build_sha256
        if (not re.fullmatch(r"[0-9a-f]{64}", expected_executable_sha256) or
                sha256_file(exe) != expected_executable_sha256):
            fail("prevalidated multiscale executable identity changed")
    else:
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

    # Reusing a label starts a replacement run. Invalidate the old publication before any
    # environment/engine failure can leave a previous PASS looking like the result of this attempt.
    label = args.label or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        eval_root = os.path.join(args.build_dir, "sbs_eval")
        out_root = contained_component(eval_root, label, "run label")
        os.makedirs(out_root, exist_ok=True)
        for filename in ("results.json", "results.rescored.json", "report.html",
                         FRAME_GATE_EVIDENCE_FILENAME):
            invalidate_publication_file(
                contained_component(out_root, filename, "run publication"),
                "run publication",
            )
        if args.report_out:
            invalidate_publication_file(
                os.path.abspath(args.report_out), "report output"
            )
    except (OSError, ValueError) as error:
        fail(f"cannot initialize replacement run publication: {error}")

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

    conf_sha = sha256_files([os.path.abspath(args.conf)])
    metric_sha = metric_contract_sha()
    meta = {
        "git_sha": git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(git(["status", "--porcelain"])),
        "run_kind": None,
        "baseline_identities": {},
        "clip_set_sha1": clip_set_sha1,
        **clip_hash_provenance,
        "mode": "profile", "suite": args.suite, "clips_root": clips_dir,
        "extra_args": args.extra,
        "conf": os.path.relpath(args.conf, REPO),
        "model": expected_model, "profile": expected_config_profile,
        "hdr_source_kind": expected_hdr_kind,
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
        "output_selection_mode": output_selection_mode,
        "literal_bestv2": literal_bestv2,
        "depth_compensation": depth_compensation,
        "eval_schema": EVAL_SCHEMA, "depth_step": depth_step,
        "depth_reuse_interval": depth_reuse_interval,
        "conf_sha256": conf_sha, "metric_sha256": metric_sha,
        "policy_warp_source_sha256": None,
        "precomputed_multiscale": bool(args.precomputed_multiscale_root),
        "multiscale_batch_manifest_sha256": None,
        "gpu_contention": contention,
        "score_workers": args.score_workers,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "run_name": label,
    }

    results, regressions, issues, hard_failures, baseline_updates = {}, [], [], [], {}
    frame_gate_record_groups = []
    native_hdr_identities = {}

    def finalize_scored_clip(clip, context, rows, agg):
        """Apply evidence, gates and baseline logic in the parent process, in clip order."""
        clip_meta = context["clip_meta"]
        perf = context["perf"]
        missing_required = missing_required_metric_evidence(
            agg, gate_thresholds, clip_meta
        )
        if missing_required:
            fail(f"{clip}: required primary metric evidence is missing: {missing_required}")

        worst, clip_issues, clip_hard_failures = score_clip_gates(
            rows, agg, gate_thresholds, clip_meta)
        issues.extend({"clip": clip, **item} for item in clip_issues)
        hard_failures.extend({"clip": clip, **item} for item in clip_hard_failures)

        clip_meta["clip_sha1"] = meta["clip_set_sha1"][clip]
        entry = {"aggregate": agg, "perf_ms": perf, "meta": clip_meta, "worst_frame": worst}
        results[clip] = entry
        if publish_gate_evidence:
            try:
                frame_gate_record_groups.append(build_frame_gate_clip_records(
                    clip, rows, gate_thresholds, context["frame_gate"]
                ))
            except (OSError, ValueError) as error:
                fail(f"{clip}: cannot publish frame-gate evidence: {error}")

        # Regression gate vs baseline. A baseline is only valid for the exact frames it was made
        # from: if the clip content changed, gating against it is meaningless -- skip it loudly
        # instead of silently comparing apples to oranges.
        bp = contained_component(base_dir, clip + ".json", "baseline filename")
        if os.path.exists(bp) and not args.update_baselines and not args.comparison_only:
            base = json.load(open(bp))
            base_meta = base.get("meta", {})
            current_context = {
                **meta, **clip_meta,
                "clip_sha1": meta["clip_set_sha1"][clip],
            }
            context_fields = BASELINE_CONTEXT_FIELDS
            if output_label_frames:
                context_fields += LABEL_SELECTION_CONTEXT_FIELDS
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
            for key, spec in thresholds["metrics"].items():
                if metric_exempt_for_clip(spec, clip_meta):
                    continue
                if spec.get("role") == "hard":
                    continue  # absolute hard constraints are independent of baseline
                baseline_value, new_value = base["aggregate"].get(key), agg.get(key)
                if baseline_value is None:
                    continue
                if (new_value is None or not isinstance(new_value, (int, float)) or
                        not math.isfinite(new_value)):
                    regressions.append({"clip": clip, "metric": key,
                                        "baseline": round(baseline_value, 3), "value": None,
                                        "reason": "missing-treatment-evidence"})
                    continue
                if sbsbench.metric_gate_failed(baseline_value, new_value, spec):
                    regressions.append({
                        "clip": clip, "metric": key,
                        "baseline": round(baseline_value, 3),
                        **worst.get(key, {}), "value": round(new_value, 3),
                    })
            if not contention:
                for key, spec in thresholds["perf_ms"].items():
                    baseline_value = base.get("perf_ms", {}).get(key)
                    new_value = perf.get(key)
                    if (isinstance(baseline_value, (int, float)) and
                            math.isfinite(baseline_value) and baseline_value > 0.0 and
                            (not isinstance(new_value, (int, float)) or
                             not math.isfinite(new_value) or new_value <= 0.0)):
                        regressions.append({"clip": clip, "metric": "perf:" + key,
                                            "baseline": round(baseline_value, 2), "value": None,
                                            "reason": "missing-treatment-evidence"})
                    elif (isinstance(baseline_value, (int, float)) and
                          math.isfinite(baseline_value) and baseline_value > 0.0 and
                          isinstance(new_value, (int, float)) and math.isfinite(new_value) and
                          (new_value - baseline_value) >
                          max(spec["abs_floor"], baseline_value * spec["rel_tol"])):
                        regressions.append({"clip": clip, "metric": "perf:" + key,
                                            "baseline": round(baseline_value, 2),
                                            "value": round(new_value, 2)})

        if args.update_baselines:
            baseline_updates[bp] = {
                "aggregate": agg, "perf_ms": perf,
                "meta": {**meta, **clip_meta, "clip_sha1": meta["clip_set_sha1"][clip]},
            }

    # Never overlap CPU metric workers with the measured GPU harness.  Doing so makes perf gates
    # depend on clip order and worker count, and lets a one-clip run compare an idle measurement
    # against a full-suite baseline measured under artificial CPU contention.  Preserve all
    # validated artifact jobs here, then score them in parallel only after every harness exits.
    harness_environment = os.environ.copy()
    score_jobs = []
    for clip in clips:
        try:
            clip_dir = contained_component(clips_dir, clip, "clip name")
            publication_clip_dir = contained_component(
                out_root, clip, "output clip name"
            )
        except ValueError as error:
            fail(str(error))
        try:
            source_clip_meta = load_optional_clip_metadata(clip_dir)
        except ValueError as error:
            fail(f"{clip}: {error}")
        # A reused result label must not retain stale publication files.  Precomputed render
        # artifacts live under a separately authenticated batch root and are never deleted here.
        shutil.rmtree(publication_clip_dir, ignore_errors=True)
        source_by_id = sbsbench.indexed_files(
            os.path.join(clip_dir, "frame_*.*"), "frame_"
        )
        if expected_hdr_kind == harness_contract.HDR_SOURCE_NATIVE_PQ:
            try:
                native_hdr_identities[clip] = authenticate_native_hdr_clip(
                    clip_dir, source_by_id, full=args.verify_clip_hashes
                )
            except RuntimeError as error:
                fail(f"{clip}: {error}")
        ordered_source_ids = sorted(source_by_id)
        try:
            output_selection = resolve_output_selection(
                clip_dir, ordered_source_ids, output_interval,
                output_gt_right_only, output_label_frames,
            )
            if args.publish_frame_gates:
                validate_full_frame_gate_coverage(ordered_source_ids, output_selection)
            elif args.publish_selected_frame_gates:
                validate_selected_frame_gate_coverage(
                    ordered_source_ids, output_selection
                )
        except ValueError as exc:
            fail(f"{clip}: {exc}")
        expected_output_ids = set(output_selection["output_frame_ids"])
        batch_identity = None
        if args.precomputed_multiscale_root:
            try:
                batch_clip_root = contained_component(
                    args.precomputed_multiscale_root, clip,
                    "precomputed multiscale clip name",
                )
                batch_identity = multiscale_batch.validate(
                    batch_clip_root,
                    clip=clip,
                    clip_sha1=clip_set_sha1[clip],
                    executable_sha256=sha256_file(exe),
                    conf_sha256=conf_sha,
                    metric_sha256=metric_sha,
                    scale=expected_artistic_scale_override,
                )
            except (OSError, ValueError) as error:
                fail(f"{clip}: invalid precomputed multiscale batch: {error}")
            out_dir = str(batch_identity["scale_root"])
            common_artifact_dir = str(batch_identity["common_root"])
            manifest_sha256 = batch_identity["manifest_sha256"]
            if meta["multiscale_batch_manifest_sha256"] is None:
                meta["multiscale_batch_manifest_sha256"] = manifest_sha256
            elif meta["multiscale_batch_manifest_sha256"] != manifest_sha256:
                fail("multiscale batch manifest changed within one run")
            print(f"[{clip}] authenticated multiscale scale artifacts...", flush=True)
        else:
            out_dir = publication_clip_dir
            common_artifact_dir = out_dir
            cmd = [exe, os.path.abspath(args.conf), "--sbs-bench",
                   "--frames", clip_dir, "--out", out_dir,
                   "--model", expected_model]
            cmd += args.extra
            print(f"[{clip}] harness...", flush=True)
            try:
                r = subprocess.run(
                    cmd, cwd=args.build_dir, capture_output=True, text=True, timeout=900,
                    env=harness_environment,
                )
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
            "output_selection_mode": output_selection["mode"],
            "label_frame_ids": output_selection["label_frame_ids"],
            "output_selected_frame_ids": output_selection["output_frame_ids"],
            "output_label_frames_sha256": output_selection["label_frames_sha256"],
            "literal_bestv2": literal_bestv2,
            "cuda_graph": expected_cuda_graph,
            "hdr_source_kind": expected_hdr_kind,
        }
        if batch_identity is not None:
            expected_contract.update({
                "multiscale_batch": True,
                "multiscale_batch_contract": multiscale_batch.HARNESS_CONTRACT,
                "multiscale_scale_index": batch_identity["scale_row"]["index"],
                "multiscale_scale_float32_bits":
                    multiscale_batch.scale_float32_bits(
                        expected_artistic_scale_override
                    ),
                "multiscale_common_artifact_directory": "../../common",
            })
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
        try:
            metric_preview_encoding, hdr_scale, white_raw = (
                validate_hdr_contract_provenance(
                    contract, expected_hdr_kind, clip
                )
            )
        except RuntimeError as error:
            fail(str(error))
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
                     "output_selection_mode": contract["output_selection_mode"],
                     "label_frame_ids": contract["label_frame_ids"],
                     "output_selected_frame_ids": contract["output_selected_frame_ids"],
                     "output_label_frames_sha256":
                         contract["output_label_frames_sha256"],
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
                     "metric_preview_encoding": metric_preview_encoding,
                     "hdr_source_kind": contract["hdr_source_kind"],
                     "hdr_input_scale": hdr_scale,
                     "sdr_white_level_raw": white_raw,
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
        if batch_identity is not None:
            clip_meta.update({
                "precomputed_multiscale": True,
                "multiscale_batch_manifest_sha256":
                    batch_identity["manifest_sha256"],
                "multiscale_batch_contract": multiscale_batch.CONTRACT,
                "multiscale_scale_float32_bits":
                    multiscale_batch.scale_float32_bits(
                        expected_artistic_scale_override
                    ),
            })

        # A valid harness result has one source, raw-model, warp-input depth, and SBS artifact for
        # every numeric frame identity. This catches dropped/renumbered outputs before metrics run.
        sbs_by_id = sbsbench.indexed_files(os.path.join(out_dir, "sbs_*.png"), "sbs_")
        sbs_ids = set(sbs_by_id)
        depth_by_id = sbsbench.indexed_files(
            os.path.join(common_artifact_dir, "depth_*.png"), "depth_"
        )
        depth_ids = set(depth_by_id)
        raw_by_id = sbsbench.indexed_files(
            os.path.join(common_artifact_dir, "raw_*.f32"), "raw_"
        )
        raw_ids = set(raw_by_id)
        mask_by_id = sbsbench.indexed_files(
            os.path.join(out_dir, "warp_mask_*.png"), "warp_mask_")
        mask_ids = set(mask_by_id)
        disparity_by_id = sbsbench.indexed_files(
            os.path.join(out_dir, "warp_disparity_*.f32"), "warp_disparity_")
        disparity_ids = set(disparity_by_id)
        unclamped_disparity_ids = set(sbsbench.indexed_files(
            os.path.join(out_dir, "warp_unclamped_disparity_*.f32"),
            "warp_unclamped_disparity_"))
        ema_mask_ids = set(sbsbench.indexed_files(
            os.path.join(common_artifact_dir, "ema_mask_*.png"), "ema_mask_"))
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
        if not os.path.exists(os.path.join(common_artifact_dir, "raw_shape.json")):
            fail(f"{clip}: raw_shape.json missing")
        # Carry the clip's own metadata (scene name/description) into results so the run dir is
        # self-describing and the report can label clips without the source clips dir.
        clip_meta.update(source_clip_meta)

        perf = {}
        perf_p = os.path.join(common_artifact_dir, "sbs_perf.json")
        if batch_identity is None and os.path.exists(perf_p):
            stages = json.load(open(perf_p)).get("stages", {})
            perf = {k: v.get("p50_ms", 0) for k, v in stages.items()}

        score_context = {"clip_meta": clip_meta, "perf": perf}
        if publish_gate_evidence:
            runtime_scene_path = os.path.join(
                common_artifact_dir, "runtime_scene_evidence.json"
            )
            try:
                runtime_scenes = runtime_scene_evidence.load(
                    runtime_scene_path
                )
            except (OSError, ValueError) as error:
                fail(
                    f"{clip}: missing/invalid authoritative runtime scene "
                    f"evidence: {error}"
                )
            if runtime_scenes["source_frame_ids"] != ordered_source_ids:
                fail(
                    f"{clip}: runtime scene source identities differ from "
                    "the evaluated full-frame sequence"
                )
            if batch_identity is not None:
                os.makedirs(publication_clip_dir, exist_ok=True)
                published_scene_path = os.path.join(
                    publication_clip_dir, "runtime_scene_evidence.json"
                )
                shutil.copyfile(runtime_scene_path, published_scene_path)
            optional_artifacts = {
                "gt_depth": sbsbench.indexed_files(
                    os.path.join(clip_dir, "gt_depth", "frame_*.*"), "frame_"
                ),
                "gt_right": sbsbench.indexed_files(
                    os.path.join(clip_dir, "gt_right", "frame_*.*"), "frame_"
                ),
                "gt_flow": sbsbench.indexed_files(
                    os.path.join(clip_dir, "gt_flow", "frame_*.npz"), "frame_"
                ),
            }
            artifact_paths = {}
            for frame_id in output_selection["output_frame_ids"]:
                paths = {
                    "source": source_by_id[frame_id],
                    "sbs": sbs_by_id[frame_id],
                    "depth": depth_by_id[frame_id],
                    "warp_mask": mask_by_id[frame_id],
                    "warp_disparity": disparity_by_id[frame_id],
                }
                for name, by_id in optional_artifacts.items():
                    if frame_id in by_id:
                        paths[name] = by_id[frame_id]
                artifact_paths[frame_id] = paths
            score_context["frame_gate"] = {
                "source_frame_ids": ordered_source_ids,
                "output_selection": output_selection,
                "artifact_paths": artifact_paths,
                "runtime_scene_evidence": runtime_scenes,
                "runtime_scene_evidence_path": runtime_scene_path,
                "clip_sha1": meta["clip_set_sha1"][clip],
                "harness_contract_sha256": sha256_file(contract_path),
                "expected_flat": clip_meta.get("expected_flat") is True,
                "geometry": {
                    key: clip_meta[key] for key in (
                        "source_width", "source_height",
                        "model_input_width", "model_input_height",
                        "depth_short_side", "depth_max_aspect",
                        "eye_width", "eye_height",
                        "content_scale_x", "content_scale_y",
                        "disparity_raster_width", "disparity_raster_height",
                        "color_mode",
                    )
                },
                "color": {
                    key: clip_meta[key] for key in (
                        "color_mode", "metric_preview_encoding", "hdr_source_kind",
                        "hdr_input_scale", "sdr_white_level_raw",
                    )
                },
                "pipeline": {
                    key: clip_meta[key] for key in (
                        "model", "profile", "depth_step", "depth_reuse_interval",
                        "adaptive_pop", "adaptive_pop_max", "pop_strength",
                        "artistic_style", "artistic_policy", "artistic_scale_override",
                        "policy_warp_source_sha256",
                    )
                },
            }
        score_jobs.append((
            clip, score_context, out_dir, clip_dir,
            clip_meta.get("expected_flat") is True,
            common_artifact_dir if batch_identity is not None else None,
        ))
        print(f"[{clip}] harness complete", flush=True)

    configure_score_worker_threads()
    score_executor = ProcessPoolExecutor(
        max_workers=args.score_workers, initializer=_initialize_score_worker
    )
    score_lifecycle["executor"] = score_executor
    score_queue = BoundedOrderedScoreQueue(score_executor, 2 * args.score_workers)
    score_lifecycle["queue"] = score_queue
    for (clip, context, out_dir, clip_dir, expected_flat,
         common_artifact_dir) in score_jobs:
        print(f"[{clip}] scoring queued...", flush=True)
        try:
            completed_scores = score_queue.submit(
                clip, context, out_dir, clip_dir, expected_flat,
                common_artifact_dir,
            )
        except ScoreWorkerError as exc:
            fail(str(exc))
        for scored_clip, scored_context, rows, agg in completed_scores:
            print(f"[{scored_clip}] scoring complete", flush=True)
            finalize_scored_clip(scored_clip, scored_context, rows, agg)

    try:
        for scored_clip, context, rows, agg in score_queue.drain():
            print(f"[{scored_clip}] scoring complete", flush=True)
            finalize_scored_clip(scored_clip, context, rows, agg)
    except ScoreWorkerError as exc:
        fail(str(exc))

    # Release idle workers before source revalidation/report generation.  The top-level finally
    # remains the fail-safe for every exceptional path above.
    score_lifecycle["queue"] = None
    score_executor.shutdown(wait=True)
    score_lifecycle["executor"] = None

    try:
        revalidate_clip_hashes(
            clips_dir, clips, clip_set_sha1, clip_hash_provenance,
            args.verify_clip_hashes,
        )
    except clip_hashes.ClipHashManifestError as error:
        fail(str(error))
    if expected_hdr_kind == harness_contract.HDR_SOURCE_NATIVE_PQ:
        for clip in clips:
            clip_dir = contained_component(clips_dir, clip, "clip name")
            source_by_id = sbsbench.indexed_files(
                os.path.join(clip_dir, "frame_*.*"), "frame_"
            )
            try:
                current_identity = authenticate_native_hdr_clip(
                    clip_dir, source_by_id, full=args.verify_clip_hashes
                )
            except RuntimeError as error:
                fail(f"{clip}: native-HDR source changed during evaluation: {error}")
            if current_identity != native_hdr_identities.get(clip):
                fail(f"{clip}: native-HDR source identity changed during evaluation")

    verdict = ("hard_failures" if hard_failures else "comparison_only" if args.comparison_only
               else "regressions" if regressions else "pass")
    out = {"meta": meta, "verdict": verdict, "regressions": regressions,
           "hard_failures": hard_failures, "issues": issues, "clips": results}
    res_path = os.path.join(out_root, "results.json")
    res_tmp = res_path + ".tmp"
    try:
        with open(res_tmp, "w", encoding="utf-8") as stream:
            json.dump(out, stream, indent=2)
        os.replace(res_tmp, res_path)
    finally:
        if os.path.exists(res_tmp):
            os.unlink(res_tmp)

    frame_gate_path = None
    if publish_gate_evidence:
        frame_gate_path = contained_component(
            out_root, FRAME_GATE_EVIDENCE_FILENAME, "frame-gate publication"
        )
        try:
            write_frame_gate_evidence(
                frame_gate_path, meta, gate_thresholds, frame_gate_record_groups,
                sha256_file(res_path),
                evidence_contract=(
                    SELECTED_FRAME_GATE_EVIDENCE_CONTRACT
                    if args.publish_selected_frame_gates else
                    FRAME_GATE_EVIDENCE_CONTRACT
                ),
            )
            validate_frame_gate_evidence(frame_gate_path)
        except (OSError, ValueError) as error:
            invalidate_publication_file(frame_gate_path, "frame-gate publication")
            fail(f"frame-gate evidence publication failed: {error}")

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
    if frame_gate_path:
        print(f"  frame gates: {frame_gate_path}")
    if args.update_baselines:
        print(f"  baselines updated in {base_dir} -- commit them with the change that justified it.")
    sys.exit(1 if regressions or hard_failures else 0)


def main():
    """Run one eval while guaranteeing worker cleanup and exact parent-env restoration."""
    score_lifecycle = {}
    parent_worker_environment = capture_score_worker_environment()
    try:
        return _run_main(score_lifecycle)
    finally:
        try:
            score_queue = score_lifecycle.get("queue")
            if score_queue is not None:
                score_queue.cancel_pending()
        finally:
            try:
                score_executor = score_lifecycle.get("executor")
                if score_executor is not None:
                    score_executor.shutdown(wait=True, cancel_futures=True)
            finally:
                restore_score_worker_environment(parent_worker_environment)


if __name__ == "__main__":
    main()
