#!/usr/bin/env python3
"""Generate authenticated target-only ordinal artistic-safety labels.

This orchestration is deliberately separate from the deployed scalar artistic
policy and its sparse schema-10 bootstrap.  For each clip/condition/geometry,
one offline harness process produces the authenticated label targets and
invokes the unchanged shipping warp for all 26 scales.
The authenticated scale artifacts are scored by the ordinary evaluator.
Sparse 1.00/1.30/1.50 visuals are retained alongside those same runs for human
inspection.
* ``sources`` publishes only authenticated target model inputs. Target rows
  must join safety labels one-to-one before the catalog is eligible for
  training. Native-PQ rows bind their linear scRGB FP16 model sources.

Each completed multiscale batch is validated and compacted immediately to
``results.json``, ``frame_gate_evidence.jsonl``, and per-clip runtime-scene
evidence.  One serialized GPU render is overlapped with CPU scoring of the
previous batch, with at most one rendered batch waiting behind the active
scorer.  The peak retained render storage is therefore bounded independently
of the complete scale grid.  Resume is bound
to an immutable plan, active-split hash, executable/config/metric identities,
and a source-content verification receipt.

Only active-split ``training`` and ``development`` productions are opened.
Sealed-test identifiers are recorded from the split document, but their
manifests and media paths are never resolved, statted, hashed, or rendered.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
from dataclasses import dataclass
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading


THIS_DIR = Path(__file__).resolve().parent
SBSBENCH_DIR = THIS_DIR.parent / "sbsbench"
sys.path.insert(0, str(SBSBENCH_DIR))

import artistic_policy_ordinal_contract as ordinal_contract  # noqa: E402
import build_ordinal_frame_label_bundle as bundle_builder  # noqa: E402
import depth_input_color as input_color  # noqa: E402
import build_clip_hash_manifest as clip_hashes  # noqa: E402
import ordinal_result_cache  # noqa: E402
import prepare_ordinal_full_frame_source_rows as full_sources  # noqa: E402
import preprocessing_artifact_cache  # noqa: E402
import multiscale_batch  # noqa: E402
import run_eval  # noqa: E402
import run_multiscale_eval  # noqa: E402
import sbs_harness_contract  # noqa: E402


SCHEMA = 8
CONTRACT = "apollo-ordinal-target-only-label-orchestration-v1"
CATALOG_SCHEMA = 5
CATALOG_CONTRACT = "apollo-ordinal-target-only-label-catalog-v1"
SOURCE_RECEIPT_SCHEMA = 1
SOURCE_RECEIPT_CONTRACT = "apollo-ordinal-source-verification-v1"
COMPACTION_SCHEMA = 3
COMPACTION_CONTRACT = "apollo-ordinal-safety-compaction-v3"
RUNTIME_FILENAME = "ordinal_orchestration_runtime.json"
RUNTIME_SCHEMA = 1
RUNTIME_CONTRACT = "apollo-ordinal-regeneration-runtime-v1"
DEPTH_MODEL = "depth_anything_v2_fp16"
PHASES = ("safety", "bundle", "sources", "catalog")
ARTIFACT_SCALES = (1.00, 1.30, 1.50)
DEFAULT_SCALE_SCORE_JOBS = run_multiscale_eval.DEFAULT_SCALE_SCORE_JOBS
DEFAULT_DEPTH_STATE_CACHE_ROOT = Path(
    r"E:\ApolloDev\artistic-policy\cache\v1\depth-state"
)
DEFAULT_SCORED_RESULT_CACHE_ROOT = Path(
    r"E:\ApolloDev\artistic-policy\cache\v1\ordinal-scores"
)
GEOMETRIES = (
    ("uncapped-1280x720", 1280, 720),
    ("packed-width-capped-960x540", 960, 540),
)
MONO_CONDITIONS = (
    ("sdr", None),
    ("hdr80", 1000),
    ("hdr200", 2500),
    ("hdr480", 6000),
)
NATIVE_HDR_CONDITIONS = (("native-pq", None),)
WORKING_SPLITS = ("training", "development")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
SOURCE_FRAME = re.compile(r"^frame_[0-9]+\.(?:png|jpg|jpeg)$", re.IGNORECASE)
SOURCE_FRAME_ID = re.compile(
    r"^frame_([0-9]+)\.(?:png|jpg|jpeg)$", re.IGNORECASE
)
# A batch already keeps one GPU inference stream busy and scores scales with
# CPU workers.  Running two batches concurrently adds GPU contention without
# increasing the authenticated inference throughput.
MAX_RENDER_WORKERS = 1
SAFETY_PIPELINE_CONTRACT = "apollo-ordinal-render-score-pipeline-v1"
MAX_PENDING_RENDERED_BATCHES = 1
# ``run_multiscale_eval`` appends either ``-batch`` or a scale slug to this
# prefix before passing it through run_eval's 128-character portable component
# contract.  Keep the base short enough for the longer suffix.
MAX_BATCH_LABEL_LENGTH = 128 - len("-batch")
LABEL_HASH_LENGTH = 24
WINDOWS_LEGACY_PATH_LIMIT = 259
LONGEST_BATCH_ARTIFACT = "warp_unclamped_disparity_4294967295.f32"


def canonical_bytes(value):
    return (json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")


def canonical_sha256(value):
    return hashlib.sha256(canonical_bytes(value).rstrip(b"\n")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path, description="JSON"):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not an object: {path}")
    return value


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def _update_runtime(workspace, plan_sha256, **updates):
    """Update non-authoritative progress state under the wrapper's run lock."""
    path = Path(workspace) / RUNTIME_FILENAME
    value = {}
    if path.is_file():
        current = load_json(path, "ordinal regeneration runtime")
        if (current.get("schema") == RUNTIME_SCHEMA and
                current.get("contract") == RUNTIME_CONTRACT and
                current.get("plan_sha256") == plan_sha256):
            value.update(current)
    value.update({
        "schema": RUNTIME_SCHEMA,
        "contract": RUNTIME_CONTRACT,
        "plan_sha256": plan_sha256,
        "last_update_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    })
    value.update(updates)
    write_json_atomic(path, value)


def _safe_component(value, description):
    if (not isinstance(value, str) or not SAFE_COMPONENT.fullmatch(value) or
            value in {".", ".."} or Path(value).name != value or
            "/" in value or "\\" in value):
        raise RuntimeError(f"unsafe {description}: {value!r}")
    return value


def _slug(value):
    result = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    if not result:
        raise RuntimeError("empty orchestration slug")
    return result


def _bounded_label(value, maximum_length):
    """Return one deterministic portable component without losing identity."""
    value = _slug(value)
    if len(value) <= maximum_length:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:LABEL_HASH_LENGTH]
    prefix_length = maximum_length - len(digest) - 1
    prefix = value[:prefix_length].rstrip("-._")
    if not prefix:
        raise RuntimeError("orchestration label limit is too small")
    return f"{prefix}-{digest}"


def _scale_slug(scale):
    return f"s{int(round(float(scale) * 100)):03d}"


def _inside(path, root):
    try:
        Path(path).resolve(strict=False).relative_to(
            Path(root).resolve(strict=False)
        )
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class Dataset:
    production_id: str
    source_kind: str
    split: str
    root: Path
    manifest: Path
    manifest_sha256: str
    clip_hash_manifest: Path
    clip_hash_manifest_sha256: str
    clip_hash_content_sha256: str
    clips: tuple[str, ...]
    frame_count: int
    label_frame_count: int
    output_frame_count: int

    @property
    def key(self):
        return _slug(self.production_id)


@dataclass(frozen=True)
class Condition:
    key: str
    raw_white: int | None
    input_variant: dict

    @property
    def input_variant_sha256(self):
        return input_color.input_variant_sha256(self.input_variant)

    @property
    def extra(self):
        if self.key == "native-pq":
            return ("--native-hdr-scrgb",)
        if self.raw_white is not None:
            return (
                "--simulate-hdr", "--sdr-white-level-raw",
                str(self.raw_white),
            )
        return ()


@dataclass(frozen=True)
class Geometry:
    name: str
    eye_width: int
    eye_height: int

    @property
    def key(self):
        return f"g{self.eye_width}x{self.eye_height}"


@dataclass(frozen=True)
class Step:
    key: str
    phase: str
    kind: str
    command: tuple[str, ...]
    output: Path
    metadata: dict

    def as_dict(self):
        return {
            "key": self.key,
            "phase": self.phase,
            "kind": self.kind,
            "output": str(self.output),
            "argv": list(self.command),
            "command": subprocess.list2cmdline(list(self.command)),
            "metadata": self.metadata,
        }


@dataclass
class Plan:
    repo: Path
    workspace: Path
    build_dir: Path
    conf: Path
    python: Path
    scorer_runtime_identity: dict | None
    depth_state_cache_root: Path | None
    scored_result_cache_root: Path | None
    active_split: Path
    active_split_sha256: str
    executable_sha256: str
    conf_sha256: str
    metric_sha256: str
    thresholds_sha256: str
    sbsbench_sha256: str
    run_eval_sha256: str
    code_identities: dict
    scope: str
    sealed_test_production_ids: tuple[str, ...]
    datasets: tuple[Dataset, ...]
    geometries: tuple[Geometry, ...]
    steps: tuple[Step, ...]

    def as_dict(self):
        counts = {}
        for step in self.steps:
            counts[step.kind] = counts.get(step.kind, 0) + 1
        working_frames = sum(item.frame_count for item in self.datasets)
        condition_label_frames = sum(
            item.label_frame_count * len(_conditions(item))
            for item in self.datasets
        )
        condition_output_frames = sum(
            item.output_frame_count * len(_conditions(item))
            for item in self.datasets
        )
        return {
            "schema": SCHEMA,
            "contract": CONTRACT,
            "evidence_roles": {
                "safety": (
                    "authenticated label targets, production depth, and "
                    "two-geometry scale sweeps; sole ordinal safety "
                    "supervision"
                ),
                "artifact": (
                    "sparse visual/debug evidence copied from authenticated "
                    "safety batches; never a separate render"
                ),
            },
            "scale_thresholds": list(ordinal_contract.SCALES),
            "artifact_scales": list(ARTIFACT_SCALES),
            "safety_pipeline": {
                "contract": SAFETY_PIPELINE_CONTRACT,
                "gpu_render_concurrency": MAX_RENDER_WORKERS,
                "cpu_score_batch_concurrency": 1,
                "maximum_pending_rendered_batches":
                    MAX_PENDING_RENDERED_BATCHES,
                "publication_order": "plan-order-after-authenticated-score",
                "failure_policy": "retain-rendered-batch-and-fail-closed",
            },
            "repo": str(self.repo),
            "workspace": str(self.workspace),
            "build_dir": str(self.build_dir),
            "conf": str(self.conf),
            "python": str(self.python),
            "scorer_runtime_identity": self.scorer_runtime_identity,
            "depth_state_cache_root": (
                str(self.depth_state_cache_root)
                if self.depth_state_cache_root is not None else None
            ),
            "scored_result_cache_root": (
                str(self.scored_result_cache_root)
                if self.scored_result_cache_root is not None else None
            ),
            "active_split": str(self.active_split),
            "active_split_sha256": self.active_split_sha256,
            "executable_sha256": self.executable_sha256,
            "conf_sha256": self.conf_sha256,
            "metric_sha256": self.metric_sha256,
            "thresholds_sha256": self.thresholds_sha256,
            "sbsbench_sha256": self.sbsbench_sha256,
            "run_eval_sha256": self.run_eval_sha256,
            "code_identities": self.code_identities,
            "scope": self.scope,
            "training_eligible": self.scope == "full-active-train-development",
            "sealed_test_policy": (
                "identifiers recorded from active split only; test manifests "
                "and media are never resolved, statted, hashed, or rendered"
            ),
            "sealed_test_production_ids": list(
                self.sealed_test_production_ids
            ),
            "datasets": [{
                "production_id": item.production_id,
                "source_kind": item.source_kind,
                "split": item.split,
                "root": str(item.root),
                "manifest": str(item.manifest),
                "manifest_sha256": item.manifest_sha256,
                "clip_hash_manifest": str(item.clip_hash_manifest),
                "clip_hash_manifest_sha256":
                    item.clip_hash_manifest_sha256,
                "clip_hash_content_sha256":
                    item.clip_hash_content_sha256,
                "clips": list(item.clips),
                "frame_count": item.frame_count,
                "label_frame_count": item.label_frame_count,
                "output_frame_count": item.output_frame_count,
                "conditions": [condition.key for condition in
                               _conditions(item)],
            } for item in self.datasets],
            "geometries": [{
                "name": item.name,
                "eye_width": item.eye_width,
                "eye_height": item.eye_height,
            } for item in self.geometries],
            "estimates": {
                "working_source_frames": working_frames,
                "condition_frames": condition_label_frames,
                "condition_context_frames": 0,
                "condition_label_frames": condition_label_frames,
                "condition_output_frames": condition_output_frames,
                "safety_frame_geometry_scale_visits": (
                    condition_output_frames * len(self.geometries) *
                    ordinal_contract.FRONTIER_SIZE
                ),
                "multiscale_harness_runs": counts.get("safety_batch", 0),
                "shipping_estimator_sequences_cold_cache": (
                    counts.get("safety_batch", 0) // len(self.geometries)
                ),
                "scalar_scale_score_runs": (
                    counts.get("safety_batch", 0) *
                    ordinal_contract.FRONTIER_SIZE
                ),
                "legacy_scalar_harness_runs_avoided": (
                    counts.get("safety_batch", 0) *
                    (ordinal_contract.FRONTIER_SIZE - 1)
                ),
                "artifact_render_runs": 0,
                "frame_label_bundles": counts.get("bundle", 0),
                "selected_source_bundles": counts.get("source", 0),
                "selected_source_rows": condition_label_frames,
                "selected_source_context_rows": 0,
                "selected_source_target_rows": condition_label_frames,
                "maximum_simultaneous_gpu_render_batches":
                    MAX_RENDER_WORKERS,
                "maximum_pending_rendered_batches":
                    MAX_PENDING_RENDERED_BATCHES,
                "maximum_retained_multiscale_batch_trees": 2,
            },
            "steps": [step.as_dict() for step in self.steps],
        }


def _conditions(dataset):
    if dataset.source_kind == "mono-video":
        return tuple(Condition(
            key, raw_white,
            input_color.sdr_input_variant() if raw_white is None else
            input_color.windows_hdr_input_variant(raw_white),
        ) for key, raw_white in MONO_CONDITIONS)
    if dataset.source_kind == "native-hdr-video":
        return (Condition(
            NATIVE_HDR_CONDITIONS[0][0], None,
            input_color.native_pq_input_variant(),
        ),)
    raise RuntimeError(
        f"unsupported ordinal source kind: {dataset.source_kind!r}"
    )


def _code_identities(repo):
    """Pin every Python/data contract used to publish resumable labels."""
    paths = {
        "orchestrator": Path(__file__).resolve(),
        "ordinal_contract": Path(ordinal_contract.__file__).resolve(),
        "bundle_builder": Path(bundle_builder.__file__).resolve(),
        "source_publisher": Path(full_sources.__file__).resolve(),
        "input_color": Path(input_color.__file__).resolve(),
        "clip_hash_manifest": Path(clip_hashes.__file__).resolve(),
        "run_eval": Path(run_eval.__file__).resolve(),
        "sbsbench": SBSBENCH_DIR / "sbsbench.py",
        "artistic_geometry_contract": (
            repo / "tools" / "depth_models" /
            "artistic_geometry_contract.py"
        ),
        "native_hdr_capture": (
            repo / "tools" / "depth_models" / "native_hdr_capture.py"
        ),
        "runtime_scene_evidence": SBSBENCH_DIR / "runtime_scene_evidence.py",
        "multiscale_batch": SBSBENCH_DIR / "multiscale_batch.py",
        "multiscale_driver": SBSBENCH_DIR / "run_multiscale_eval.py",
        "depth_state_cache": SBSBENCH_DIR / "depth_state_cache.py",
        "preprocessing_artifact_cache": (
            repo / "tools" / "depth_models" /
            "preprocessing_artifact_cache.py"
        ),
        "ordinal_result_cache": (
            repo / "tools" / "depth_models" / "ordinal_result_cache.py"
        ),
        "harness_contract": Path(sbs_harness_contract.__file__).resolve(),
        "harness_source": repo / "src" / "sbs_bench_harness.cpp",
        "depth_state_sequence_header": repo / "src" / "sbs_depth_state_sequence.h",
        "depth_state_sequence_source": repo / "src" / "sbs_depth_state_sequence.cpp",
        "depth_estimator_header": repo / "src" / "video_depth_estimator.h",
        "depth_estimator_source": repo / "src" / "video_depth_estimator.cpp",
        "thresholds": repo / "tools" / "sbsbench" / "thresholds.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "ordinal orchestration code input is missing: " +
            ", ".join(missing)
        )
    return {
        role: {"path": str(path), "sha256": sha256_file(path)}
        for role, path in sorted(paths.items())
    }


def _manifest_path(active_path, value, description):
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"active split lacks {description}")
    path = Path(value)
    if not path.is_absolute():
        path = active_path.parent / path
    return path.resolve()


def _sequence_frame_count(sequence, source_kind):
    key = "context_frames" if source_kind == "mono-video" else "frames"
    value = sequence.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeError(f"dataset sequence has invalid {key}")
    return value


def _selected_frame_count(dataset, clips):
    manifest = clip_hashes.load_manifest(dataset.clip_hash_manifest)
    total = 0
    for clip in clips:
        entry = manifest["clips"].get(clip)
        if not isinstance(entry, dict):
            raise RuntimeError(f"clip hash manifest lacks {clip!r}")
        records = entry.get("files")
        if not isinstance(records, list):
            raise RuntimeError(f"clip hash manifest records differ: {clip!r}")
        count = sum(
            isinstance(row, dict) and
            isinstance(row.get("path"), str) and
            SOURCE_FRAME.fullmatch(row["path"]) is not None
            for row in records
        )
        if count < 1:
            raise RuntimeError(f"clip hash manifest has no source frames: {clip!r}")
        total += count
    return total


def _clip_selection(dataset, clip):
    """Authenticate the exact target-only output selection."""
    manifest = clip_hashes.load_manifest(dataset.clip_hash_manifest)
    entry = manifest["clips"].get(clip)
    if not isinstance(entry, dict) or not isinstance(entry.get("files"), list):
        raise RuntimeError(f"clip hash manifest lacks {clip!r}")
    source_ids = []
    label_manifest_bound = False
    for row in entry["files"]:
        path = row.get("path") if isinstance(row, dict) else None
        if path == "label_frames.json":
            label_manifest_bound = True
        match = SOURCE_FRAME_ID.fullmatch(path or "")
        if match:
            source_ids.append(int(match.group(1)))
    if (not source_ids or source_ids != sorted(set(source_ids)) or
            source_ids != list(range(source_ids[0], source_ids[-1] + 1))):
        raise RuntimeError(f"clip source identities are not consecutive: {clip!r}")
    if not label_manifest_bound:
        raise RuntimeError(
            f"clip hash manifest does not bind label_frames.json: {clip!r}"
        )
    try:
        label_ids = run_eval.sbsbench.load_label_frame_ids(
            str(dataset.root / clip)
        )
    except ValueError as error:
        raise RuntimeError(
            f"invalid label-frame manifest for {clip!r}: {error}"
        ) from error
    if not label_ids:
        raise RuntimeError(f"clip has no authenticated label targets: {clip!r}")
    source_id_set = set(source_ids)
    if not set(label_ids).issubset(source_id_set):
        raise RuntimeError(f"label targets escape source cadence: {clip!r}")
    return tuple(source_ids), tuple(label_ids), tuple(label_ids)


def _selected_label_frame_count(dataset, clips):
    return sum(len(_clip_selection(dataset, clip)[1]) for clip in clips)


def _selected_output_frame_count(dataset, clips):
    return sum(len(_clip_selection(dataset, clip)[2]) for clip in clips)


def load_working_datasets(active_split_path):
    """Load train/dev only; never touch a sealed-test path."""
    active_split_path = Path(active_split_path).resolve()
    active = load_json(active_split_path, "active split")
    if active.get("schema") != 1:
        raise RuntimeError("active split schema is unsupported")
    assignments = active.get("split_productions")
    productions = active.get("productions")
    if not isinstance(assignments, dict) or not isinstance(productions, list):
        raise RuntimeError("active split assignments are missing")
    expected = {}
    for split in (*WORKING_SPLITS, "test"):
        values = assignments.get(split)
        if (not isinstance(values, list) or not values or
                any(not isinstance(value, str) or not value for value in values) or
                len(values) != len(set(values))):
            raise RuntimeError(f"active split has invalid {split} assignment")
        for production in values:
            if production in expected:
                raise RuntimeError("active split repeats a production across roles")
            expected[production] = split
    if len(assignments["test"]) < 2:
        raise RuntimeError("active split lacks two independent sealed tests")
    rows = {}
    for row in productions:
        if not isinstance(row, dict):
            raise RuntimeError("active split production row is invalid")
        production = row.get("production_id")
        if not isinstance(production, str) or not production:
            raise RuntimeError("active split production ID is invalid")
        if production in rows:
            raise RuntimeError("active split repeats a production row")
        rows[production] = row
    if set(rows) != set(expected):
        raise RuntimeError("active split production rows and assignments differ")

    datasets = []
    for split in WORKING_SPLITS:
        for production in assignments[split]:
            row = rows[production]
            source_kind = row.get("source_kind")
            if row.get("split") != split or source_kind not in {
                    "mono-video", "native-hdr-video"}:
                raise RuntimeError(
                    f"working production contract differs: {production}"
                )
            manifest_path = _manifest_path(
                active_split_path, row.get("dataset_manifest"),
                f"{production} dataset manifest",
            )
            digest = row.get("dataset_manifest_sha256")
            if (not isinstance(digest, str) or not SHA256.fullmatch(digest) or
                    sha256_file(manifest_path) != digest):
                raise RuntimeError(
                    f"working dataset manifest identity is stale: {production}"
                )
            manifest = load_json(manifest_path, "working dataset manifest")
            if (manifest.get("schema") != 2 or
                    manifest.get("production_id") != production or
                    manifest.get("source_kind") != source_kind or
                    manifest.get("split") != split):
                raise RuntimeError(
                    f"working dataset manifest disagrees: {production}"
                )
            sequences = manifest.get("sequences")
            if not isinstance(sequences, list) or not sequences:
                raise RuntimeError(f"working dataset has no sequences: {production}")
            clips = []
            frame_count = 0
            for sequence in sequences:
                if not isinstance(sequence, dict):
                    raise RuntimeError("dataset sequence row is invalid")
                clip = _safe_component(sequence.get("clip"), "clip name")
                if sequence.get("split") != split:
                    raise RuntimeError("dataset sequence split differs")
                clips.append(clip)
                frame_count += _sequence_frame_count(sequence, source_kind)
            if len(clips) != len(set(clips)):
                raise RuntimeError(f"dataset repeats a clip: {production}")
            declared_frames = (
                manifest.get("context_frame_count")
                if source_kind == "mono-video" else
                manifest.get("frame_count")
            )
            if declared_frames != frame_count:
                raise RuntimeError(
                    f"dataset frame count differs: {production}"
                )
            root = manifest_path.parent.resolve()
            clip_manifest_path = root / clip_hashes.MANIFEST_NAME
            clip_manifest = clip_hashes.load_manifest(clip_manifest_path)
            if set(clip_manifest.get("clips", {})) != set(clips):
                raise RuntimeError(
                    f"clip hash manifest coverage differs: {production}"
                )
            # Cheap stat/path verification is part of every plan build. Full
            # content verification is performed once by execute().
            clip_hashes.verify_selected_clips(
                clip_manifest_path, root, clips, full=False
            )
            dataset = Dataset(
                production_id=production,
                source_kind=source_kind,
                split=split,
                root=root,
                manifest=manifest_path,
                manifest_sha256=digest,
                clip_hash_manifest=clip_manifest_path.resolve(),
                clip_hash_manifest_sha256=sha256_file(clip_manifest_path),
                clip_hash_content_sha256=clip_manifest[
                    clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
                ],
                clips=tuple(clips),
                frame_count=frame_count,
                label_frame_count=0,
                output_frame_count=0,
            )
            label_frame_count = _selected_label_frame_count(dataset, clips)
            output_frame_count = _selected_output_frame_count(dataset, clips)
            declared_labels = manifest.get("label_frame_count")
            if (not isinstance(declared_labels, int) or
                    isinstance(declared_labels, bool) or
                    declared_labels != label_frame_count):
                raise RuntimeError(
                    f"dataset label-frame count differs: {production}"
                )
            datasets.append(Dataset(
                production_id=dataset.production_id,
                source_kind=dataset.source_kind,
                split=dataset.split,
                root=dataset.root,
                manifest=dataset.manifest,
                manifest_sha256=dataset.manifest_sha256,
                clip_hash_manifest=dataset.clip_hash_manifest,
                clip_hash_manifest_sha256=dataset.clip_hash_manifest_sha256,
                clip_hash_content_sha256=dataset.clip_hash_content_sha256,
                clips=dataset.clips,
                frame_count=dataset.frame_count,
                label_frame_count=label_frame_count,
                output_frame_count=output_frame_count,
            ))
    if not datasets:
        raise RuntimeError("active split has no ordinal working productions")
    return active, tuple(datasets), tuple(assignments["test"])


def _render_label(prefix, role, dataset, condition, geometry, scale, clip,
                  maximum_length=MAX_BATCH_LABEL_LENGTH):
    if role != "safety":
        raise RuntimeError(f"unsupported ordinal render role: {role}")
    return (
        f"{_batch_label(prefix, dataset, condition, geometry, clip, maximum_length)}-"
        f"{_scale_slug(scale)}"
    )


def _batch_label(prefix, dataset, condition, geometry, clip,
                 maximum_length=MAX_BATCH_LABEL_LENGTH):
    raw = "-".join((
        _slug(prefix), "safety", dataset.key, condition.key, geometry.key,
        _slug(clip),
    ))
    return _bounded_label(raw, maximum_length)


def _batch_label_path_limit(eval_root, clip):
    """Reserve enough legacy-Windows path space for the longest C++ artifact."""
    probe = (
        Path(eval_root).resolve() / "x-batch" / clip / "scales" / "s150" /
        LONGEST_BATCH_ARTIFACT
    )
    fixed_length = len(str(probe)) - 1
    limit = min(
        MAX_BATCH_LABEL_LENGTH,
        WINDOWS_LEGACY_PATH_LIMIT - fixed_length,
    )
    if limit <= LABEL_HASH_LENGTH + 1:
        raise RuntimeError(
            "evaluator/clip path leaves no safe multiscale label budget: "
            f"{eval_root} / {clip}"
        )
    return limit


def _clip_frame_count(dataset, clip):
    return _selected_frame_count(dataset, (clip,))


def _clip_label_frame_count(dataset, clip):
    return len(_clip_selection(dataset, clip)[1])


def _clip_output_frame_count(dataset, clip):
    return len(_clip_selection(dataset, clip)[2])


def _batch_render_command(plan_values, dataset, condition, geometry, clip,
                          label_prefix, summary):
    cache_args = (
        (
            "--depth-state-cache-root",
            str(plan_values["depth_state_cache_root"]),
            "--depth-state-cache-split", dataset.split,
        )
        if plan_values["depth_state_cache_root"] is not None else ()
    )
    command = [
        str(plan_values["python"]),
        str(plan_values["repo"] / "tools" / "sbsbench" /
            "run_multiscale_eval.py"),
        "--build-dir", str(plan_values["build_dir"]),
        "--conf", str(plan_values["conf"]),
        "--clips-root", str(dataset.root),
        "--clip", clip,
        "--label-prefix", label_prefix,
        "--scales", ",".join(
            f"{scale:.2f}" for scale in ordinal_contract.SCALES
        ),
        "--score-workers", str(plan_values["score_workers"]),
        "--selected-label-frames",
        *cache_args,
        "--summary", str(summary),
        "--extra",
        "--eye-w", str(geometry.eye_width),
        "--eye-h", str(geometry.eye_height),
        *condition.extra,
    ]
    return tuple(command)


def build_plan(args):
    repo = Path(__file__).resolve().parents[2]
    workspace = args.workspace.resolve()
    build_dir = args.build_dir.resolve()
    conf = args.conf.resolve()
    python = args.python.resolve()
    requested_cache_root = getattr(
        args, "depth_state_cache_root", DEFAULT_DEPTH_STATE_CACHE_ROOT
    )
    depth_state_cache_root = (
        requested_cache_root.resolve()
        if requested_cache_root is not None else None
    )
    requested_score_cache_root = getattr(
        args, "scored_result_cache_root", DEFAULT_SCORED_RESULT_CACHE_ROOT
    )
    scored_result_cache_root = (
        requested_score_cache_root.resolve()
        if requested_score_cache_root is not None else None
    )
    executable = build_dir / "sunshine.exe"
    if not executable.is_file():
        raise RuntimeError(f"missing evaluator executable: {executable}")
    if not conf.is_file():
        raise RuntimeError(f"missing evaluator configuration: {conf}")
    if not python.is_file():
        raise RuntimeError(f"missing Python interpreter: {python}")
    scorer_runtime = (
        ordinal_result_cache.query_scorer_runtime_identity(python)
        if scored_result_cache_root is not None else None
    )
    active_split = args.active_split.resolve()
    _active, datasets, sealed_test_ids = load_working_datasets(active_split)
    requested_productions = tuple(getattr(args, "production", ()) or ())
    if requested_productions:
        if len(requested_productions) != len(set(requested_productions)):
            raise RuntimeError("ordinal smoke selection repeats a production")
        available = {item.production_id for item in datasets}
        unknown = sorted(set(requested_productions) - available)
        if unknown:
            raise RuntimeError(
                "ordinal smoke selection is not a working production: " +
                ", ".join(unknown)
            )
        datasets = tuple(
            item for item in datasets
            if item.production_id in set(requested_productions)
        )
    clip_limit = getattr(args, "clip_limit", None)
    if clip_limit is not None:
        datasets = tuple(Dataset(
            production_id=item.production_id,
            source_kind=item.source_kind,
            split=item.split,
            root=item.root,
            manifest=item.manifest,
            manifest_sha256=item.manifest_sha256,
            clip_hash_manifest=item.clip_hash_manifest,
            clip_hash_manifest_sha256=item.clip_hash_manifest_sha256,
            clip_hash_content_sha256=item.clip_hash_content_sha256,
            clips=item.clips[:clip_limit],
            frame_count=_selected_frame_count(item, item.clips[:clip_limit]),
            label_frame_count=_selected_label_frame_count(
                item, item.clips[:clip_limit]
            ),
            output_frame_count=_selected_output_frame_count(
                item, item.clips[:clip_limit]
            ),
        ) for item in datasets)
    full_working_ids = {
        row.get("production_id") for row in _active.get("productions", ())
        if isinstance(row, dict) and row.get("split") in WORKING_SPLITS
    }
    scope = (
        "full-active-train-development"
        if ({item.production_id for item in datasets} == full_working_ids and
            clip_limit is None) else
        "smoke-subset-not-training-eligible"
    )
    # Evaluator publications live in one build-global namespace, while the
    # orchestration lock is workspace-local.  Namespace every label by the
    # immutable workspace identity so two valid workspaces cannot delete or
    # replace each other's render/score transactions.
    workspace_namespace = canonical_sha256({
        "contract": "apollo-ordinal-eval-workspace-namespace-v1",
        "workspace": str(workspace),
    })[:8]
    effective_prefix = f"{args.run_prefix}-w{workspace_namespace}"
    if scope != "full-active-train-development":
        selection_identity = [
            (item.production_id, list(item.clips)) for item in datasets
        ]
        effective_prefix = (
            f"{effective_prefix}-smoke-"
            f"{canonical_sha256(selection_identity)[:8]}"
        )
    geometries = tuple(Geometry(*value) for value in GEOMETRIES)
    eval_root = build_dir / "sbs_eval"
    values = {
        "repo": repo, "workspace": workspace, "build_dir": build_dir,
        "conf": conf, "python": python, "score_workers": args.score_workers,
        "depth_state_cache_root": depth_state_cache_root,
    }
    steps = []
    safety_outputs = {}
    bundle_outputs = {}

    for dataset in datasets:
        for condition in _conditions(dataset):
            for geometry in geometries:
                for clip in dataset.clips:
                    label_limit = _batch_label_path_limit(eval_root, clip)
                    label_prefix = _batch_label(
                        effective_prefix, dataset, condition, geometry, clip,
                        label_limit,
                    )
                    run_eval.validate_path_component(
                        label_prefix, "ordinal batch label prefix"
                    )
                    run_eval.validate_path_component(
                        label_prefix + "-batch", "ordinal batch label"
                    )
                    summary = (
                        workspace / "multiscale_summaries" /
                        f"{label_prefix}.json"
                    )
                    render_labels = {
                        _scale_slug(scale): _render_label(
                            effective_prefix, "safety", dataset, condition,
                            geometry, scale, clip, label_limit,
                        )
                        for scale in ordinal_contract.SCALES
                    }
                    for render_label in render_labels.values():
                        run_eval.validate_path_component(
                            render_label, "ordinal scale label"
                        )
                    scale_outputs = {
                        slug: eval_root / render_label
                        for slug, render_label in render_labels.items()
                    }
                    scale_metadata = {
                        _scale_slug(scale): _render_metadata(
                            dataset, condition, geometry, scale, "safety",
                            clips=(clip,),
                            frame_count=_clip_label_frame_count(dataset, clip),
                            source_frame_count=_clip_frame_count(dataset, clip),
                            output_frame_count=_clip_output_frame_count(
                                dataset, clip
                            ),
                        )
                        for scale in ordinal_contract.SCALES
                    }
                    steps.append(Step(
                        key=label_prefix, phase="safety", kind="safety_batch",
                        command=_batch_render_command(
                            values, dataset, condition, geometry, clip,
                            label_prefix, summary,
                        ),
                        output=summary,
                        metadata={
                            "evidence_role": "safety",
                            "production_id": dataset.production_id,
                            "source_kind": dataset.source_kind,
                            "split": dataset.split,
                            "dataset_manifest_sha256":
                                dataset.manifest_sha256,
                            "condition": condition.key,
                            "clip": clip,
                            "clips": [clip],
                            "clips_root": str(dataset.root),
                            "frame_count": _clip_label_frame_count(
                                dataset, clip
                            ),
                            "source_frame_count": _clip_frame_count(
                                dataset, clip
                            ),
                            "output_frame_count": _clip_output_frame_count(
                                dataset, clip
                            ),
                            "scale_outputs": {
                                key: str(value)
                                for key, value in scale_outputs.items()
                            },
                            "scale_metadata": scale_metadata,
                        },
                    ))
                    for scale in ordinal_contract.SCALES:
                        safety_outputs[(
                            dataset.production_id, condition.key,
                            geometry.key, scale, clip,
                        )] = scale_outputs[_scale_slug(scale)]

    for dataset in datasets:
        for condition in _conditions(dataset):
            for clip in dataset.clips:
                output = (
                    workspace / "bundles" / dataset.key / condition.key / clip
                )
                run_paths = [
                    safety_outputs[(
                        dataset.production_id, condition.key,
                        geometry.key, scale, clip,
                    )] / run_eval.FRAME_GATE_EVIDENCE_FILENAME
                    for geometry in geometries
                    for scale in ordinal_contract.SCALES
                ]
                manifest = output / "run_grid.json"
                command = (
                    str(python),
                    str(repo / "tools" / "depth_models" /
                        "build_ordinal_frame_label_bundle.py"),
                    "--run-manifest", str(manifest),
                    "--thresholds",
                    str(repo / "tools" / "sbsbench" / "thresholds.json"),
                    "--output", str(output / "labels.jsonl"),
                    "--summary", str(output / "summary.json"),
                )
                steps.append(Step(
                    key=(f"bundle-{dataset.key}-{condition.key}-{clip}"),
                    phase="bundle", kind="bundle", command=command,
                    output=output,
                    metadata={
                        "production_id": dataset.production_id,
                        "source_kind": dataset.source_kind,
                        "split": dataset.split,
                        "dataset_manifest_sha256": dataset.manifest_sha256,
                        "condition": condition.key,
                        "input_variant": condition.input_variant,
                        "input_variant_sha256":
                            condition.input_variant_sha256,
                        "clip": clip,
                        "frame_count": _clip_label_frame_count(dataset, clip),
                        "source_frame_count": _clip_frame_count(dataset, clip),
                        "output_frame_count": _clip_output_frame_count(
                            dataset, clip
                        ),
                        "run_manifest": str(manifest),
                        "run_paths": [str(path) for path in run_paths],
                        "expected_runs": (
                            len(geometries) *
                            ordinal_contract.FRONTIER_SIZE
                        ),
                    },
                ))
                bundle_outputs[(
                    dataset.production_id, condition.key, clip,
                )] = output / "labels.jsonl"

    for dataset in datasets:
        for condition in _conditions(dataset):
            output = workspace / "sources" / dataset.key / condition.key
            command = [
                str(python),
                str(repo / "tools" / "depth_models" /
                    "prepare_ordinal_full_frame_source_rows.py"),
                "--dataset-manifest", str(dataset.manifest),
            ]
            source_bundles = []
            for clip in dataset.clips:
                command.extend(("--clip", clip))
                bundle = bundle_outputs[(
                    dataset.production_id, condition.key, clip,
                )]
                source_bundles.append(str(bundle))
                command.extend(("--ordinal-bundle", str(bundle)))
            command.extend(("--output", str(output)))
            steps.append(Step(
                key=f"sources-{dataset.key}-{condition.key}",
                phase="sources", kind="source", command=tuple(command),
                output=output,
                metadata={
                    "production_id": dataset.production_id,
                    "source_kind": dataset.source_kind,
                    "split": dataset.split,
                    "dataset_manifest": str(dataset.manifest),
                    "dataset_manifest_sha256": dataset.manifest_sha256,
                    "condition": condition.key,
                    "input_variant": condition.input_variant,
                    "input_variant_sha256": condition.input_variant_sha256,
                    "clips": list(dataset.clips),
                    "expected_frames": dataset.label_frame_count,
                    "expected_target_frames": dataset.label_frame_count,
                    "expected_context_frames": 0,
                    "ordinal_bundles": source_bundles,
                },
            ))
    catalog_output = workspace / "ordinal_frame_label_catalog.json"
    steps.append(Step(
        key="catalog", phase="catalog", kind="catalog", command=(),
        output=catalog_output,
        metadata={"expected_bundles": sum(
            len(dataset.clips) * len(_conditions(dataset))
            for dataset in datasets
        )},
    ))
    return Plan(
        repo=repo, workspace=workspace, build_dir=build_dir, conf=conf,
        python=python, scorer_runtime_identity=scorer_runtime,
        depth_state_cache_root=depth_state_cache_root,
        scored_result_cache_root=scored_result_cache_root,
        active_split=active_split,
        active_split_sha256=sha256_file(active_split),
        executable_sha256=sha256_file(executable),
        conf_sha256=run_eval.sha256_files([str(conf)]),
        metric_sha256=run_eval.metric_contract_sha(),
        thresholds_sha256=sha256_file(
            repo / "tools" / "sbsbench" / "thresholds.json"
        ),
        sbsbench_sha256=sha256_file(
            repo / "tools" / "sbsbench" / "sbsbench.py"
        ),
        run_eval_sha256=sha256_file(
            repo / "tools" / "sbsbench" / "run_eval.py"
        ),
        code_identities=_code_identities(repo),
        scope=scope,
        sealed_test_production_ids=sealed_test_ids,
        datasets=datasets, geometries=geometries, steps=tuple(steps),
    )


def _render_metadata(dataset, condition, geometry, scale, role, *,
                     clips=None, frame_count=None, source_frame_count=None,
                     output_frame_count=None):
    variant = condition.input_variant
    selected_clips = list(dataset.clips if clips is None else clips)
    selected_frame_count = (
        dataset.label_frame_count if frame_count is None else frame_count
    )
    return {
        "evidence_role": role,
        "production_id": dataset.production_id,
        "source_kind": dataset.source_kind,
        "split": dataset.split,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "clips_root": str(dataset.root),
        "clips": selected_clips,
        "frame_count": selected_frame_count,
        "source_frame_count": (
            dataset.frame_count if source_frame_count is None else
            source_frame_count
        ),
        "output_frame_count": (
            dataset.output_frame_count if output_frame_count is None else
            output_frame_count
        ),
        "clip_hash_manifest_sha256":
            dataset.clip_hash_manifest_sha256,
        "clip_hash_content_sha256": dataset.clip_hash_content_sha256,
        "condition": condition.key,
        "raw_white": condition.raw_white,
        "input_variant": variant,
        "input_variant_sha256": condition.input_variant_sha256,
        "hdr_source_kind":
            sbs_harness_contract.input_variant_hdr_source_kind(variant),
        "metric_preview_encoding":
            sbs_harness_contract.input_variant_metric_preview_encoding(variant),
        "scale": scale,
        "eye_width": geometry.eye_width,
        "eye_height": geometry.eye_height,
        "geometry_key": geometry.key,
    }


def _validate_render_common(step, payload, expected_selection):
    if payload.get("verdict") not in {"comparison_only", "hard_failures"}:
        raise RuntimeError(f"render verdict is incomplete: {step.key}")
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise RuntimeError(f"render metadata is missing: {step.key}")
    if (meta.get("run_name") != step.output.name or
            Path(meta.get("clips_root", "")).resolve() !=
            Path(step.metadata["clips_root"]).resolve() or
            meta.get("clip_hash_manifest_sha256") !=
            step.metadata["clip_hash_manifest_sha256"] or
            meta.get("output_selection_mode") != expected_selection or
            meta.get("depth_step") != "current-once" or
            meta.get("depth_reuse_interval") != 1 or
            meta.get("artistic_policy") is not False or
            meta.get("model") != DEPTH_MODEL or
            not math.isclose(
                float(meta.get("artistic_scale_override", -1.0)),
                float(step.metadata["scale"]), abs_tol=1e-9,
            ) or set(payload.get("clips", {})) !=
            set(step.metadata["clips"])):
        raise RuntimeError(f"render contract differs: {step.key}")
    for clip_name, entry in payload["clips"].items():
        clip = entry.get("meta")
        if not isinstance(clip, dict):
            raise RuntimeError(f"clip render metadata is missing: {clip_name}")
        raw_white = step.metadata["raw_white"] or 0
        expected_hdr_scale = raw_white / 1000.0
        if step.metadata["condition"] == "native-pq":
            expected_hdr_scale = 0.0
        if (clip.get("model") != DEPTH_MODEL or
                clip.get("depth_step") != "current-once" or
                clip.get("depth_reuse_interval") != 1 or
                clip.get("depth_compensation") != "none" or
                clip.get("output_selection_mode") != expected_selection or
                clip.get("artistic_policy") is not False or
                clip.get("hdr_source_kind") !=
                step.metadata["hdr_source_kind"] or
                clip.get("metric_preview_encoding") !=
                step.metadata["metric_preview_encoding"] or
                clip.get("sdr_white_level_raw") != raw_white or
                not math.isclose(
                    float(clip.get("hdr_input_scale", -1.0)),
                    expected_hdr_scale, abs_tol=1e-9,
                ) or clip.get("eye_width") !=
                step.metadata["eye_width"] or
                clip.get("eye_height") !=
                step.metadata["eye_height"]):
            raise RuntimeError(
                f"clip render contract differs: {step.key}/{clip_name}"
            )
    return payload


def validate_safety_render(step):
    results = step.output / "results.json"
    sidecar = step.output / run_eval.FRAME_GATE_EVIDENCE_FILENAME
    if not results.is_file() or not sidecar.is_file():
        raise RuntimeError(f"selected-frame safety evidence is missing: {step.key}")
    _validate_render_common(
        step, load_json(results, "selected-frame safety result"),
        "label-frames"
    )
    records = run_eval.validate_frame_gate_evidence(sidecar)
    header = records[0]
    if (header.get("run_name") != step.output.name or
            header.get("metric_sha256") != run_eval.metric_contract_sha() or
            header.get("conf_sha256") !=
            run_eval.sha256_files([str(Path(step.command[
                step.command.index("--conf") + 1
            ]))]) or
            header.get("clip_hash_manifest_sha256") !=
            step.metadata["clip_hash_manifest_sha256"]):
        raise RuntimeError(f"selected-frame sidecar identity differs: {step.key}")
    parsed = []
    for clip in step.metadata["clips"]:
        run = bundle_builder.parse_frame_gate_evidence(sidecar, clip)
        geometry = run["geometry"]
        if (run["input_variant"] != step.metadata["input_variant"] or
                run["input_variant_sha256"] !=
                step.metadata["input_variant_sha256"] or
                run["scale"] != step.metadata["scale"] or
                geometry["eye_width"] != step.metadata["eye_width"] or
                geometry["eye_height"] != step.metadata["eye_height"] or
                run["common_identity"]["pipeline_without_scale"].get(
                    "depth_reuse_interval"
                ) != 1):
            raise RuntimeError(
                f"selected-frame sidecar semantics differ: {step.key}/{clip}"
            )
        parsed.append(run)
    if sum(len(run["frames"]) for run in parsed) != step.metadata["frame_count"]:
        raise RuntimeError(f"selected-frame safety coverage differs: {step.key}")
    return parsed


def _scalar_safety_steps(batch_step):
    outputs = batch_step.metadata.get("scale_outputs")
    metadata = batch_step.metadata.get("scale_metadata")
    if (not isinstance(outputs, dict) or not isinstance(metadata, dict) or
            set(outputs) != {_scale_slug(scale) for scale in
                             ordinal_contract.SCALES} or
            set(metadata) != set(outputs)):
        raise RuntimeError(f"multiscale plan grid is incomplete: {batch_step.key}")
    for scale in ordinal_contract.SCALES:
        slug = _scale_slug(scale)
        output = Path(outputs[slug])
        yield Step(
            key=output.name,
            phase="safety",
            kind="safety_render",
            command=batch_step.command,
            output=output,
            metadata=metadata[slug],
        )


def validate_safety_batch(step, *, require_compaction=True):
    summary = load_json(step.output, "multiscale safety summary")
    expected_clip = step.metadata["clip"]
    expected_outputs = step.metadata["scale_outputs"]
    expected_scale_score_jobs = int(
        step.command[step.command.index("--score-workers") + 1]
    )
    executable = (
        Path(step.command[step.command.index("--build-dir") + 1]) /
        "sunshine.exe"
    )
    conf = Path(step.command[step.command.index("--conf") + 1])
    if (Path(step.output).read_bytes() != canonical_bytes(summary) or
            summary.get("schema") != run_multiscale_eval.SCHEMA or
            summary.get("contract") != run_multiscale_eval.CONTRACT or
            summary.get("clip") != expected_clip or
            not isinstance(summary.get("clip_sha1"), str) or
            not re.fullmatch(r"[0-9a-f]{12}", summary["clip_sha1"]) or
            summary.get("executable_sha256") != sha256_file(executable) or
            summary.get("conf_sha256") !=
            run_eval.sha256_files([str(conf)]) or
            summary.get("metric_sha256") != run_eval.metric_contract_sha() or
            not isinstance(summary.get("render_identity_sha256"), str) or
            not SHA256.fullmatch(summary["render_identity_sha256"]) or
            not isinstance(
                summary.get("render_identity_receipt_sha256"), str
            ) or not SHA256.fullmatch(
                summary["render_identity_receipt_sha256"]
            ) or
            summary.get("scale_score_jobs") != expected_scale_score_jobs or
            summary.get("child_score_workers") !=
            run_multiscale_eval.CHILD_SCORE_WORKERS or
            summary.get(
                "rendered_batch_retained_until_success_summary"
            ) is not True or
            summary.get("batch_cleanup_policy") !=
            "best-effort-after-success-summary"):
        raise RuntimeError(f"multiscale safety summary differs: {step.key}")
    rows = summary.get("scale_results")
    if not isinstance(rows, list) or len(rows) != ordinal_contract.FRONTIER_SIZE:
        raise RuntimeError(f"multiscale safety result grid differs: {step.key}")
    by_slug = {
        row.get("scale_slug"): row for row in rows if isinstance(row, dict)
    }
    if len(by_slug) != len(rows) or set(by_slug) != set(expected_outputs):
        raise RuntimeError(f"multiscale safety scale identities differ: {step.key}")
    expected_source_ids = None
    expected_label_ids = None
    expected_output_ids = None
    expected_label_manifest_sha256 = None
    expected_clip_sha1 = None
    for scalar in _scalar_safety_steps(step):
        slug = _scale_slug(scalar.metadata["scale"])
        row = by_slug[slug]
        provenance_root = (
            scalar.output / "multiscale_provenance" / expected_clip
        )
        provenance = {
            path.name: sha256_file(path)
            for path in provenance_root.iterdir() if path.is_file()
        } if provenance_root.is_dir() else {}
        try:
            row_scale = float(row.get("scale"))
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"multiscale safety publication differs: {scalar.key}"
            ) from error
        if (Path(row.get("run", "")).resolve(strict=False) !=
                scalar.output.resolve(strict=False) or
                not math.isclose(
                    row_scale,
                    float(scalar.metadata["scale"]), abs_tol=1e-9,
                ) or row.get("scale_slug") != slug or
                row.get("results_sha256") !=
                sha256_file(scalar.output / "results.json") or
                row.get("frame_gate_evidence_sha256") != sha256_file(
                    scalar.output / run_eval.FRAME_GATE_EVIDENCE_FILENAME
                ) or row.get("provenance_sha256") != provenance or
                set(provenance) != {
                    "contract.json", multiscale_batch.MANIFEST,
                    multiscale_batch.HARNESS_MANIFEST,
                    run_multiscale_eval.RENDER_IDENTITY_FILENAME,
                }):
            raise RuntimeError(
                f"multiscale safety publication differs: {scalar.key}"
            )
        parsed = validate_safety_render(scalar)
        if len(parsed) != 1:
            raise RuntimeError(
                f"multiscale safety clip coverage differs: {scalar.key}"
            )
        run = parsed[0]
        label_ids = [frame["frame_id"] for frame in run["frames"]]
        common = run["common_identity"]
        source_ids = common["source_frame_ids"]
        output_ids = common["output_selected_frame_ids"]
        label_manifest_sha256 = common["output_label_frames_sha256"]
        expected_source_ids = expected_source_ids or source_ids
        expected_label_ids = expected_label_ids or label_ids
        expected_output_ids = expected_output_ids or output_ids
        expected_label_manifest_sha256 = (
            expected_label_manifest_sha256 or label_manifest_sha256
        )
        expected_clip_sha1 = expected_clip_sha1 or run["common_identity"][
            "clip_sha1"
        ]
        identity = run["run_identity"]
        render_receipt = load_json(
            provenance_root / run_multiscale_eval.RENDER_IDENTITY_FILENAME,
            "multiscale render identity",
        )
        if (source_ids != expected_source_ids or
                label_ids != expected_label_ids or
                common["label_frame_ids"] != expected_label_ids or
                output_ids != expected_output_ids or
                label_manifest_sha256 != expected_label_manifest_sha256 or
                run["common_identity"]["clip_sha1"] != expected_clip_sha1 or
                identity["multiscale_batch_manifest_sha256"] !=
                summary.get("batch_manifest_sha256") or
                identity["multiscale_batch_manifest_sha256"] !=
                provenance[multiscale_batch.MANIFEST] or
                identity["multiscale_harness_contract_sha256"] !=
                provenance[multiscale_batch.HARNESS_MANIFEST] or
                identity["multiscale_scale_contract_sha256"] !=
                provenance["contract.json"] or
                render_receipt.get("schema") !=
                run_multiscale_eval.RENDER_IDENTITY_SCHEMA or
                render_receipt.get("contract") !=
                run_multiscale_eval.RENDER_IDENTITY_CONTRACT or
                render_receipt.get("render_identity_sha256") !=
                summary["render_identity_sha256"] or
                render_receipt.get("batch_manifest_sha256") !=
                summary["batch_manifest_sha256"] or
                provenance[run_multiscale_eval.RENDER_IDENTITY_FILENAME] !=
                summary["render_identity_receipt_sha256"]):
            raise RuntimeError(
                f"multiscale safety provenance join differs: {scalar.key}"
            )
        if require_compaction:
            validate_compaction(scalar)
    if (summary.get("source_frame_ids") != expected_source_ids or
            summary.get("label_frame_ids") != expected_label_ids or
            summary.get("output_selected_frame_ids") != expected_output_ids or
            summary.get("output_selection_mode") != "label-frames" or
            summary.get("output_label_frames_sha256") !=
            expected_label_manifest_sha256 or
            summary.get("clip_sha1") != expected_clip_sha1):
        raise RuntimeError(f"multiscale safety source identity differs: {step.key}")
    return summary


def _tree_identities(root, relative_to):
    root = Path(root)
    if not root.is_dir():
        return {}
    return {
        path.resolve().relative_to(Path(relative_to).resolve()).as_posix():
            sha256_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def compact_safety_tree(output, clips, *, multiscale=False, scale=None):
    """Delete full rendered frames while retaining replayable label evidence."""
    output = Path(output).resolve()
    retained = {
        (output / "results.json").resolve(),
        (output / run_eval.FRAME_GATE_EVIDENCE_FILENAME).resolve(),
    }
    for clip in clips:
        retained.add((
            output / clip / bundle_builder.RUNTIME_SCENE_FILENAME
        ).resolve())
    provenance = {}
    artifacts = {}
    if multiscale:
        for clip in clips:
            provenance_root = output / "multiscale_provenance" / clip
            identities = _tree_identities(provenance_root, output)
            expected_names = {
                "contract.json", "multiscale_batch_manifest.json",
                "multiscale_contract.json",
                run_multiscale_eval.RENDER_IDENTITY_FILENAME,
            }
            if {Path(path).name for path in identities} != expected_names:
                raise RuntimeError(
                    f"multiscale provenance differs: {output.name}/{clip}"
                )
            provenance.update(identities)
            retained.update(
                path.resolve() for path in provenance_root.rglob("*")
                if path.is_file()
            )
            artifact_root = output / "artifact_evidence" / clip
            artifact_rows = _tree_identities(artifact_root, output)
            needs_artifacts = any(
                math.isclose(float(scale), expected, abs_tol=1e-9)
                for expected in ARTIFACT_SCALES
            ) if scale is not None else False
            if needs_artifacts and (
                    not artifact_rows or
                    not (artifact_root / "visual_evidence.json").is_file()):
                raise RuntimeError(
                    f"sparse multiscale evidence is missing: {output.name}/{clip}"
                )
            if not needs_artifacts and artifact_rows:
                raise RuntimeError(
                    f"unexpected sparse multiscale evidence: {output.name}/{clip}"
                )
            artifacts.update(artifact_rows)
            retained.update(
                path.resolve() for path in artifact_root.rglob("*")
                if path.is_file()
            )
    missing = [str(path) for path in retained if not path.is_file()]
    if missing:
        raise RuntimeError(
            "cannot compact incomplete safety evidence: " + ", ".join(missing)
        )
    deleted_files = 0
    deleted_bytes = 0
    marker_path = output / "ordinal_safety_compaction.json"
    retained.add(marker_path.resolve())
    for path in sorted(
            output.rglob("*"), key=lambda candidate: len(candidate.parts),
            reverse=True):
        resolved = path.resolve(strict=False)
        if path.is_file() and resolved not in retained:
            deleted_bytes += path.stat().st_size
            path.unlink()
            deleted_files += 1
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    marker = {
        "schema": COMPACTION_SCHEMA,
        "contract": COMPACTION_CONTRACT,
        "results_sha256": sha256_file(output / "results.json"),
        "frame_gate_evidence_sha256": sha256_file(
            output / run_eval.FRAME_GATE_EVIDENCE_FILENAME
        ),
        "runtime_scene_evidence_sha256": {
            clip: sha256_file(
                output / clip / bundle_builder.RUNTIME_SCENE_FILENAME
            ) for clip in clips
        },
        "multiscale_provenance_sha256": provenance,
        "artifact_evidence_sha256": artifacts,
        "retained_role": "selected-target-safety-label-evidence",
        "deleted_files": deleted_files,
        "deleted_bytes": deleted_bytes,
    }
    write_json_atomic(marker_path, marker)
    return marker


def validate_compaction(step):
    marker_path = step.output / "ordinal_safety_compaction.json"
    marker = load_json(marker_path, "ordinal safety compaction")
    if (marker.get("schema") != COMPACTION_SCHEMA or
            marker.get("contract") != COMPACTION_CONTRACT or
            marker.get("results_sha256") !=
            sha256_file(step.output / "results.json") or
            marker.get("frame_gate_evidence_sha256") != sha256_file(
                step.output / run_eval.FRAME_GATE_EVIDENCE_FILENAME
            ) or marker.get("retained_role") !=
            "selected-target-safety-label-evidence"):
        raise RuntimeError(f"ordinal compaction marker is stale: {step.key}")
    expected_scenes = {
        clip: sha256_file(
            step.output / clip / bundle_builder.RUNTIME_SCENE_FILENAME
        ) for clip in step.metadata["clips"]
    }
    if marker.get("runtime_scene_evidence_sha256") != expected_scenes:
        raise RuntimeError(f"ordinal runtime scenes changed: {step.key}")
    expected_provenance = {}
    expected_artifacts = {}
    for clip in step.metadata["clips"]:
        expected_provenance.update(_tree_identities(
            step.output / "multiscale_provenance" / clip, step.output
        ))
        expected_artifacts.update(_tree_identities(
            step.output / "artifact_evidence" / clip, step.output
        ))
    needs_artifacts = any(
        math.isclose(float(step.metadata["scale"]), expected, abs_tol=1e-9)
        for expected in ARTIFACT_SCALES
    )
    if (not expected_provenance or
            marker.get("multiscale_provenance_sha256") !=
            expected_provenance or
            marker.get("artifact_evidence_sha256") != expected_artifacts or
            needs_artifacts != bool(expected_artifacts)):
        raise RuntimeError(f"ordinal multiscale evidence changed: {step.key}")
    return marker


def _run_manifest_payload(step):
    paths = step.metadata.get("run_paths")
    if (not isinstance(paths, list) or
            len(paths) != step.metadata["expected_runs"]):
        raise RuntimeError(f"bundle run grid is incomplete: {step.key}")
    return {
        "schema": bundle_builder.RUN_GRID_SCHEMA,
        "contract": bundle_builder.RUN_GRID_CONTRACT,
        "clip": step.metadata["clip"],
        "input_variant_sha256": step.metadata["input_variant_sha256"],
        "runs": paths,
    }


def validate_bundle(step):
    labels = step.output / "labels.jsonl"
    summary_path = step.output / "summary.json"
    manifest_path = Path(step.metadata["run_manifest"])
    if not labels.is_file() or not summary_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"ordinal label bundle is incomplete: {step.key}")
    if load_json(manifest_path, "ordinal run grid") != _run_manifest_payload(step):
        raise RuntimeError(f"ordinal run grid changed: {step.key}")
    records = bundle_builder.validate_frame_label_bundle(labels)
    header = records[0]
    summary = bundle_builder.validate_frame_label_summary(
        labels, summary_path, records=records
    )
    if (header.get("clip") != step.metadata["clip"] or
            header.get("input_variant") != step.metadata["input_variant"] or
            header.get("input_variant_sha256") !=
            step.metadata["input_variant_sha256"] or
            summary.get("label_bundle_sha256") != sha256_file(labels) or
            summary.get("clip") != step.metadata["clip"] or
            summary.get("input_variant_sha256") !=
            step.metadata["input_variant_sha256"]):
        raise RuntimeError(f"ordinal label bundle identity differs: {step.key}")
    return records


def _bundle_steps(plan):
    return [step for step in plan.steps if step.kind == "bundle"]


def _source_steps(plan):
    return [step for step in plan.steps if step.kind == "source"]


def validate_source(step):
    """Authenticate one target-only source publication against its plan."""
    labels = step.output / "labels.jsonl"
    summary_path = step.output / "summary.json"
    contract_path = step.output / "source_contract.json"
    if not all(path.is_file() for path in (
            labels, summary_path, contract_path)):
        raise RuntimeError(f"ordinal source bundle is incomplete: {step.key}")
    rows = full_sources.validate_full_frame_source_bundle(
        labels, verify_media=True
    )
    summary = load_json(summary_path, "ordinal source summary")
    contract = load_json(contract_path, "ordinal source contract")
    expected_bundles = {
        str(Path(path).resolve()) for path in step.metadata["ordinal_bundles"]
    }
    observed_bundles = contract.get("ordinal_bundles")
    if not isinstance(observed_bundles, list):
        raise RuntimeError(f"ordinal source bundle list is missing: {step.key}")
    observed_paths = {
        str(Path(identity.get("path", "")).resolve(strict=False))
        for identity in observed_bundles if isinstance(identity, dict)
    }
    dataset_identity = contract.get("dataset_manifest")
    target_rows = [row for row in rows if row.get("row_role") == "target"]
    context_rows = [row for row in rows if row.get("row_role") == "context"]
    if (len(rows) != step.metadata["expected_frames"] or
            summary.get("accepted") != step.metadata["expected_frames"] or
            summary.get("row_count") != step.metadata["expected_frames"] or
            len(target_rows) != step.metadata["expected_target_frames"] or
            summary.get("target_row_count") !=
            step.metadata["expected_target_frames"] or
            len(context_rows) != step.metadata["expected_context_frames"] or
            summary.get("context_row_count") !=
            step.metadata["expected_context_frames"] or
            summary.get("production_id") !=
            step.metadata["production_id"] or
            summary.get("source_kind") != step.metadata["source_kind"] or
            summary.get("split") != step.metadata["split"] or
            summary.get("input_variant_sha256") !=
            step.metadata["input_variant_sha256"] or
            summary.get("selected_clips") != step.metadata["clips"] or
            not isinstance(dataset_identity, dict) or
            dataset_identity.get("path") !=
            str(Path(step.metadata["dataset_manifest"]).resolve()) or
            dataset_identity.get("sha256") !=
            step.metadata["dataset_manifest_sha256"] or
            contract.get("input_variant") !=
            step.metadata["input_variant"] or
            contract.get("input_variant_sha256") !=
            step.metadata["input_variant_sha256"] or
            contract.get("selected_clips") != step.metadata["clips"] or
            contract.get("scope") != summary.get("scope") or
            observed_paths != expected_bundles or
            len(observed_bundles) != len(expected_bundles)):
        raise RuntimeError(f"ordinal source identity differs: {step.key}")
    expected_clips = set(step.metadata["clips"])
    if ({row.get("clip") for row in rows} != expected_clips or
            any(
                row.get("production_id") !=
                step.metadata["production_id"] or
                row.get("source_kind") != step.metadata["source_kind"] or
                row.get("split") != step.metadata["split"] or
                row.get("input_variant") !=
                step.metadata["input_variant"] or
                row.get("input_variant_sha256") !=
                step.metadata["input_variant_sha256"] or
                str(Path(row.get("ordinal_bundle", "")).resolve(
                    strict=False
                )) not in expected_bundles
                for row in rows
            )):
        raise RuntimeError(f"ordinal source rows differ: {step.key}")
    return {
        "rows": rows,
        "summary": summary,
        "contract": contract,
        "labels": labels,
        "summary_path": summary_path,
        "contract_path": contract_path,
    }


def build_catalog(plan):
    entries = []
    for step in _bundle_steps(plan):
        records = validate_bundle(step)
        labels = step.output / "labels.jsonl"
        summary = step.output / "summary.json"
        header = records[0]
        common_identity = header.get("common_run_identity", {})
        if common_identity.get("executable_sha256") != plan.executable_sha256:
            raise RuntimeError(
                f"ordinal bundle executable identity differs: {step.key}"
            )
        batch_manifests = {}
        for geometry_sha256 in sorted({
                identity["geometry_sha256"]
                for identity in header["scale_run_identities"]
                }):
            values = {
                identity["multiscale_batch_manifest_sha256"]
                for identity in header["scale_run_identities"]
                if identity["geometry_sha256"] == geometry_sha256
            }
            if len(values) != 1:
                raise RuntimeError(
                    f"ordinal bundle batch identity differs: {step.key}"
                )
            batch_manifests[geometry_sha256] = values.pop()
        entries.append({
            "production_id": step.metadata["production_id"],
            "source_kind": step.metadata["source_kind"],
            "split": step.metadata["split"],
            "dataset_manifest_sha256":
                step.metadata["dataset_manifest_sha256"],
            "condition": step.metadata["condition"],
            "input_variant_sha256":
                step.metadata["input_variant_sha256"],
            "clip": step.metadata["clip"],
            "frame_count": len(records) - 2,
            "executable_sha256": common_identity["executable_sha256"],
            "multiscale_batch_manifest_sha256_by_geometry":
                batch_manifests,
            "labels": str(labels.resolve()),
            "labels_sha256": sha256_file(labels),
            "summary": str(summary.resolve()),
            "summary_sha256": sha256_file(summary),
        })
    entries.sort(key=lambda item: (
        item["split"], item["production_id"], item["condition"], item["clip"]
    ))
    source_entries = []
    for step in _source_steps(plan):
        publication = validate_source(step)
        rows = publication["rows"]
        labels = publication["labels"]
        summary = publication["summary_path"]
        contract = publication["contract_path"]
        target_rows = [
            row for row in rows if row["row_role"] == "target"
        ]
        context_rows = [
            row for row in rows if row["row_role"] == "context"
        ]
        source_entries.append({
            "production_id": step.metadata["production_id"],
            "source_kind": step.metadata["source_kind"],
            "split": step.metadata["split"],
            "dataset_manifest_sha256":
                step.metadata["dataset_manifest_sha256"],
            "condition": step.metadata["condition"],
            "input_variant_sha256":
                step.metadata["input_variant_sha256"],
            "scope": publication["summary"]["scope"],
            "clips": list(step.metadata["clips"]),
            "clip_counts": dict(sorted(Counter(
                row["clip"] for row in rows
            ).items())),
            "target_clip_counts": dict(sorted(Counter(
                row["clip"] for row in target_rows
            ).items())),
            "context_clip_counts": dict(sorted(Counter(
                row["clip"] for row in context_rows
            ).items())),
            "row_count": len(rows),
            "target_row_count": len(target_rows),
            "context_row_count": len(context_rows),
            "labels": str(labels.resolve()),
            "labels_sha256": sha256_file(labels),
            "summary": str(summary.resolve()),
            "summary_sha256": sha256_file(summary),
            "source_contract": str(contract.resolve()),
            "source_contract_sha256": sha256_file(contract),
        })
    source_entries.sort(key=lambda item: (
        item["split"], item["production_id"], item["condition"]
    ))
    bundle_counts = {
        (entry["production_id"], entry["condition"], entry["clip"]):
        entry["frame_count"] for entry in entries
    }
    source_target_counts = {}
    for entry in source_entries:
        for clip, count in entry["target_clip_counts"].items():
            key = (entry["production_id"], entry["condition"], clip)
            if key in source_target_counts:
                raise RuntimeError(
                    "ordinal catalog repeats a source/condition/clip"
                )
            source_target_counts[key] = count
    if source_target_counts != bundle_counts:
        raise RuntimeError(
            "ordinal catalog target rows and safety labels do not join exactly"
        )
    expected_target_rows = sum(
        dataset.label_frame_count * len(_conditions(dataset))
        for dataset in plan.datasets
    )
    expected_rows = expected_target_rows
    expected_context_rows = 0
    expected_sources = sum(
        len(_conditions(dataset)) for dataset in plan.datasets
    )
    exact_coverage = (
        len(source_entries) == expected_sources and
        sum(item["row_count"] for item in source_entries) == expected_rows and
        sum(item["target_row_count"] for item in source_entries) ==
        expected_target_rows and
        sum(item["context_row_count"] for item in source_entries) ==
        expected_context_rows and
        (
            plan.scope != "full-active-train-development" or
            all(item["scope"] == "full-dataset"
                for item in source_entries)
        )
    )
    if not exact_coverage:
        raise RuntimeError("ordinal catalog target-only coverage differs")
    return {
        "schema": CATALOG_SCHEMA,
        "contract": CATALOG_CONTRACT,
        "active_split": str(plan.active_split),
        "active_split_sha256": plan.active_split_sha256,
        "sealed_test_policy": (
            "not generated; only train/development bundle paths appear"
        ),
        "sealed_test_production_ids": list(
            plan.sealed_test_production_ids
        ),
        "scope": plan.scope,
        "training_eligible": (
            plan.scope == "full-active-train-development" and exact_coverage
        ),
        "evidence_role": (
            "authenticated sparse safety targets"
        ),
        "source_role": (
            "authenticated target-only production model inputs"
        ),
        "scale_thresholds": list(ordinal_contract.SCALES),
        "metric_sha256": plan.metric_sha256,
        "conf_sha256": plan.conf_sha256,
        "executable_sha256": plan.executable_sha256,
        "thresholds_sha256": plan.thresholds_sha256,
        "sbsbench_sha256": plan.sbsbench_sha256,
        "run_eval_sha256": plan.run_eval_sha256,
        "code_identities": plan.code_identities,
        "bundle_count": len(entries),
        "frame_count_by_split_and_regime": _catalog_counts(entries),
        "source_bundle_count": len(source_entries),
        "source_row_count": sum(
            item["row_count"] for item in source_entries
        ),
        "source_target_row_count": sum(
            item["target_row_count"] for item in source_entries
        ),
        "source_context_row_count": sum(
            item["context_row_count"] for item in source_entries
        ),
        "source_row_count_by_split_and_regime": _catalog_counts(
            source_entries, "row_count"
        ),
        "source_row_count_by_split_and_condition":
            _catalog_condition_counts(source_entries, "row_count"),
        "source_target_row_count_by_split_and_regime": _catalog_counts(
            source_entries, "target_row_count"
        ),
        "source_target_row_count_by_split_and_condition":
            _catalog_condition_counts(source_entries, "target_row_count"),
        "source_context_row_count_by_split_and_regime": _catalog_counts(
            source_entries, "context_row_count"
        ),
        "source_context_row_count_by_split_and_condition":
            _catalog_condition_counts(source_entries, "context_row_count"),
        "bundles": entries,
        "sources": source_entries,
    }


def _catalog_counts(entries, count_field="frame_count"):
    counts = {}
    for entry in entries:
        regime = "sdr" if entry["condition"] == "sdr" else "hdr"
        key = f"{entry['split']}/{regime}"
        counts[key] = counts.get(key, 0) + entry[count_field]
    return dict(sorted(counts.items()))


def _catalog_condition_counts(entries, count_field):
    counts = {}
    for entry in entries:
        key = f"{entry['split']}/{entry['condition']}"
        counts[key] = counts.get(key, 0) + entry[count_field]
    return dict(sorted(counts.items()))


def validate_catalog(plan, path, expected=None):
    path = Path(path)
    value = load_json(path, "ordinal label catalog")
    if expected is None:
        expected = build_catalog(plan)
    if (value != expected or
            value.get("bundle_count") != len(_bundle_steps(plan)) or
            value.get("source_bundle_count") != len(_source_steps(plan))):
        raise RuntimeError("ordinal label catalog is stale")
    if any(
            entry["split"] not in WORKING_SPLITS
            for field in ("bundles", "sources")
            for entry in value[field]):
        raise RuntimeError("sealed-test bundle entered ordinal catalog")
    return value


def _step_complete(step, plan):
    try:
        if step.kind == "safety_batch":
            validate_safety_batch(step, require_compaction=True)
        elif step.kind == "bundle":
            validate_bundle(step)
        elif step.kind == "source":
            validate_source(step)
        elif step.kind == "catalog":
            validate_catalog(plan, step.output)
        else:
            raise RuntimeError(f"unknown ordinal step kind: {step.kind}")
        return True
    except (KeyError, OSError, RuntimeError, TypeError, ValueError,
            json.JSONDecodeError):
        return False


def _safe_remove(path, root):
    path = Path(path).resolve(strict=False)
    root = Path(root).resolve(strict=False)
    if path == root or not _inside(path, root):
        raise RuntimeError(f"refusing to replace output outside stage root: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _verify_sources(plan):
    receipt_path = plan.workspace / "source_verification.json"
    expected_rows = [{
        "production_id": dataset.production_id,
        "split": dataset.split,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "clip_hash_manifest_sha256":
            dataset.clip_hash_manifest_sha256,
        "clip_hash_content_sha256":
            dataset.clip_hash_content_sha256,
        "clips": list(dataset.clips),
    } for dataset in plan.datasets]
    for dataset in plan.datasets:
        clip_hashes.verify_selected_clips(
            dataset.clip_hash_manifest, dataset.root, dataset.clips,
            full=True,
        )
    write_json_atomic(receipt_path, {
        "schema": SOURCE_RECEIPT_SCHEMA,
        "contract": SOURCE_RECEIPT_CONTRACT,
        "active_split_sha256": plan.active_split_sha256,
        "verification": "full-content-per-invocation",
        "datasets": expected_rows,
    })
    return receipt_path


def _staged_step_command(step, stage):
    """Insert a driver stage before ``--extra`` consumes the argv tail."""
    if stage not in {"render", "score"}:
        raise ValueError(f"unsupported ordinal safety stage: {stage!r}")
    command = list(step.command)
    if "run_multiscale_eval.py" not in Path(command[1]).name:
        raise RuntimeError(f"ordinal step is not a multiscale driver: {step.key}")
    if "--stage" in command:
        raise RuntimeError(f"ordinal step already owns a stage: {step.key}")
    insertion = command.index("--extra") if "--extra" in command else len(command)
    command[insertion:insertion] = ["--stage", stage]
    return tuple(command)


def _render_identity_for_step(plan, step):
    """Rebuild the driver's path-independent identity before cache access."""
    preprocessing_artifact_cache.require_working_split(step.metadata["split"])
    command = list(step.command)
    extra_index = command.index("--extra")
    extra = run_multiscale_eval._validated_extra(command[extra_index + 1:])
    clips_root = Path(step.metadata["clips_root"]).resolve(strict=True)
    clip = step.metadata["clip"]
    clip_dir = Path(run_eval.contained_component(
        clips_root, clip, "clip name"
    ))
    output_selection = run_multiscale_eval._resolve_output_selection(
        clip_dir, "--selected-label-frames" in command[:extra_index]
    )
    clip_set_sha1, provenance = run_eval.resolve_clip_hashes(
        str(clips_root), [clip], False
    )
    clip_content_mode, clip_content_rows = (
        run_multiscale_eval._path_independent_clip_content_rows(
            clip_dir, clip, provenance
        )
    )
    clip_content = run_multiscale_eval._clip_content_identity(
        clip_content_mode, clip_content_rows
    )
    profile = run_eval.expected_profile(str(plan.conf), extra)
    model = run_eval.expected_depth_model(str(plan.conf), profile, extra)
    depth_context = run_multiscale_eval.depth_state_identity_context(
        repo=plan.repo,
        build_dir=plan.build_dir,
        conf_sha256=plan.conf_sha256,
        executable_sha256=plan.executable_sha256,
        model=model,
        clip_dir=clip_dir,
        source_content_rows=clip_content_rows,
        source_ids=output_selection["source_frame_ids"],
        selected_frame_ids=output_selection["output_frame_ids"],
        extra=extra,
    )
    run_multiscale_eval.reverify_depth_state_inputs(
        depth_context["identity"], depth_context["identity_args"],
        depth_context["runtime_snapshot"], clips_root, clip,
        clip_set_sha1, provenance,
    )
    return run_multiscale_eval._render_identity(
        clip_content=clip_content,
        clip_sha1=clip_set_sha1[clip],
        executable_sha256=plan.executable_sha256,
        conf_sha256=plan.conf_sha256,
        metric_sha256=plan.metric_sha256,
        model=model,
        extra=extra,
        output_selection=output_selection,
        scales=ordinal_contract.SCALES,
        depth_state_identity_sha256=depth_context["identity_sha256"],
    )


def _score_cache_contracts(plan):
    """Bind every runtime/code contract that can change scored bytes."""
    code = plan.code_identities
    return {
        "driver": {
            "schema": run_multiscale_eval.SCHEMA,
            "contract": run_multiscale_eval.CONTRACT,
            "implementation_sha256": code["multiscale_driver"]["sha256"],
            "depth_state_cache_sha256":
                code["depth_state_cache"]["sha256"],
            "artifact_cache_sha256":
                code["preprocessing_artifact_cache"]["sha256"],
        },
        "transport": {
            "schema": multiscale_batch.SCHEMA,
            "contract": multiscale_batch.CONTRACT,
            "harness_schema": multiscale_batch.HARNESS_SCHEMA,
            "harness_contract": multiscale_batch.HARNESS_CONTRACT,
            "implementation_sha256": code["multiscale_batch"]["sha256"],
        },
        "evaluator": {
            "schema": run_eval.EVAL_SCHEMA,
            "metric_sha256": plan.metric_sha256,
            "run_eval_sha256": plan.run_eval_sha256,
            "sbsbench_sha256": plan.sbsbench_sha256,
            "artistic_geometry_contract_sha256":
                code["artistic_geometry_contract"]["sha256"],
            "native_hdr_capture_sha256":
                code["native_hdr_capture"]["sha256"],
            "runtime_scene_evidence_sha256":
                code["runtime_scene_evidence"]["sha256"],
            "clip_hash_manifest_sha256":
                code["clip_hash_manifest"]["sha256"],
            "harness_contract_sha256":
                code["harness_contract"]["sha256"],
        },
        "gate": {
            "schema": run_eval.FRAME_GATE_EVIDENCE_SCHEMA,
            "contract": run_eval.SELECTED_FRAME_GATE_EVIDENCE_CONTRACT,
            "selection_contract":
                run_eval.SELECTED_FRAME_GATE_OUTPUT_SELECTION_CONTRACT,
        },
        "compaction": {
            "schema": COMPACTION_SCHEMA,
            "contract": COMPACTION_CONTRACT,
        },
        "bundle_adapter": {
            "schema": bundle_builder.BUNDLE_SCHEMA,
            "contract": bundle_builder.BUNDLE_CONTRACT,
            "implementation_sha256": code["bundle_builder"]["sha256"],
        },
        "orchestrator": {
            "schema": SCHEMA,
            "contract": CONTRACT,
            "implementation_sha256": code["orchestrator"]["sha256"],
        },
        "result_cache": {
            "schema": ordinal_result_cache.PACKET_SCHEMA,
            "contract": ordinal_result_cache.PACKET_CONTRACT,
            "implementation_sha256": code["ordinal_result_cache"]["sha256"],
        },
        "scorer_runtime": plan.scorer_runtime_identity,
    }


def _score_cache_context(plan, step):
    if plan.scored_result_cache_root is None:
        return None
    split = preprocessing_artifact_cache.require_working_split(
        step.metadata["split"]
    )
    clips_root = Path(step.metadata["clips_root"]).resolve(strict=True)
    scale_outputs = {
        slug: Path(path).resolve(strict=False)
        for slug, path in step.metadata["scale_outputs"].items()
    }
    preprocessing_artifact_cache.require_disjoint_roots(
        plan.scored_result_cache_root, clips_root, step.output.parent,
        *scale_outputs.values(),
    )
    render = _render_identity_for_step(plan, step)
    identity = ordinal_result_cache.scored_cache_identity(
        split=split,
        render_identity_sha256=render["render_identity_sha256"],
        metric_sha256=plan.metric_sha256,
        thresholds_sha256=plan.thresholds_sha256,
        scales=ordinal_contract.SCALES,
        artifact_scales=ARTIFACT_SCALES,
        contracts=_score_cache_contracts(plan),
    )
    return {
        "cache": preprocessing_artifact_cache.DirectoryArtifactCache(
            plan.scored_result_cache_root
        ),
        "identity": identity,
        "summary_path": step.output,
        "scale_outputs": scale_outputs,
        "clip": step.metadata["clip"],
        "clips_root": clips_root,
        "score_workers": int(step.command[
            step.command.index("--score-workers") + 1
        ]),
    }


def _materialize_scored_cache(context):
    if context is None:
        return None
    return ordinal_result_cache.materialize(
        context["cache"], context["identity"],
        summary_path=context["summary_path"],
        scale_outputs=context["scale_outputs"],
        clips_root=context["clips_root"],
        score_workers=context["score_workers"],
    )


def _publish_scored_cache(context):
    if context is None:
        return None
    return ordinal_result_cache.publish(
        context["cache"], context["identity"],
        summary_path=context["summary_path"],
        scale_outputs=context["scale_outputs"],
        clip=context["clip"], clips_root=context["clips_root"],
    )


class _SafetyPipelineCancelled(RuntimeError):
    """Internal cooperative-cancellation signal for render/score children."""


def _terminate_subprocess_tree(process):
    """Best-effort bounded teardown of a staged evaluator process tree."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ("taskkill", "/PID", str(process.pid), "/T", "/F"),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def _run_step_process(step, logs_root, *, stage=None, cancel_event=None):
    logs_root.mkdir(parents=True, exist_ok=True)
    suffix = f".{stage}" if stage is not None else ""
    log = logs_root / f"{step.key}{suffix}.log"
    command = (
        _staged_step_command(step, stage)
        if stage is not None else step.command
    )
    stage_label = f"/{stage}" if stage is not None else ""
    print(f"\n[{step.phase}{stage_label}] {step.key}", flush=True)
    print(subprocess.list2cmdline(list(command)), flush=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    if cancel_event is not None and cancel_event.is_set():
        raise _SafetyPipelineCancelled(
            f"cancelled ordinal step before launch: {step.key}"
        )
    popen_options = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt" else {"start_new_session": True}
    )
    with log.open("w", encoding="utf-8", newline="\n") as stream:
        process = subprocess.Popen(
            list(command), cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
            env=environment, **popen_options,
        )
        assert process.stdout is not None
        pump_errors = []

        def pump_output():
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
            except (OSError, ValueError) as error:
                pump_errors.append(error)

        pump = threading.Thread(
            target=pump_output, name=f"ordinal-log-{step.key}", daemon=True
        )
        pump.start()
        wait_event = cancel_event or threading.Event()
        try:
            while process.poll() is None:
                if pump_errors:
                    raise RuntimeError(
                        f"ordinal step logging failed: {step.key}"
                    ) from pump_errors[0]
                if wait_event.wait(0.1) and cancel_event is not None:
                    raise _SafetyPipelineCancelled(
                        f"cancelled ordinal step: {step.key}"
                    )
            return_code = process.returncode
        except BaseException:
            _terminate_subprocess_tree(process)
            raise
        finally:
            pump.join(timeout=5)
            if pump.is_alive():
                _terminate_subprocess_tree(process)
                pump.join(timeout=5)
            process.stdout.close()
        if pump_errors:
            raise RuntimeError(
                f"ordinal step logging failed: {step.key}"
            ) from pump_errors[0]
    if return_code != 0:
        raise RuntimeError(
            f"ordinal step{stage_label} failed ({return_code}): "
            f"{step.key}; {log}"
        )


def _prepare_output(step, plan, args, completed):
    if step.key in completed:
        if not _step_complete(step, plan):
            raise RuntimeError(
                f"recorded ordinal step is stale/incomplete: {step.key}"
            )
        print(f"[resume] {step.key}", flush=True)
        return False
    if _step_complete(step, plan):
        print(
            f"[adopt] authenticated completed output: {step.key}",
            flush=True,
        )
        return False
    if step.output.exists():
        nonempty = step.output.is_file() or any(step.output.iterdir())
        if nonempty:
            if not args.restart:
                raise RuntimeError(
                    f"partial ordinal output blocks {step.key}; use "
                    "--repair-partials in regenerate_artistic_ordinal_corpus.py"
                )
            root = plan.workspace
            _safe_remove(step.output, root)
    return True


def _run_safety_pipeline(pending, logs_root, authenticate_scored):
    """Overlap one CPU score transaction with the next serialized GPU render."""
    score_item = None
    score_future = None
    cancel_event = threading.Event()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="ordinal-score-batch"
    )
    try:
        for item in pending:
            render_error = None
            try:
                _run_step_process(
                    item, logs_root, stage="render",
                    cancel_event=cancel_event,
                )
            except Exception as error:
                render_error = error

            # Finish and checkpoint the prior scorer even if the next render
            # failed.  Its authenticated work is independent and must remain
            # resumable rather than being silently lost.
            if score_future is not None:
                authenticate_scored(score_item, score_future)
                score_item = None
                score_future = None

            if render_error is not None:
                raise RuntimeError(
                    f"ordinal render failed at {item.key}: {render_error}"
                ) from render_error

            score_item = item
            score_future = executor.submit(
                _run_step_process, item, logs_root, stage="score",
                cancel_event=cancel_event,
            )

        if score_future is not None:
            authenticate_scored(score_item, score_future)
    finally:
        # Covers KeyboardInterrupt/SystemExit as well as ordinary validation
        # failures.  The score worker observes this before executor shutdown,
        # tears down its complete child tree, and cannot leak stale GPU work.
        cancel_event.set()
        executor.shutdown(wait=True, cancel_futures=True)


def execute(plan, args):
    plan.workspace.mkdir(parents=True, exist_ok=True)
    plan_path = plan.workspace / "ordinal_orchestration_plan.json"
    plan_value = plan.as_dict()
    plan_sha256 = canonical_sha256(plan_value)
    if plan_path.exists():
        if load_json(plan_path, "ordinal orchestration plan") != plan_value:
            raise RuntimeError(
                "ordinal orchestration plan changed; use a new workspace"
            )
    else:
        write_json_atomic(plan_path, plan_value)
    # The current binary/config/metric implementations must still equal the
    # immutable plan before any old output can be resumed.
    if (sha256_file(plan.build_dir / "sunshine.exe") !=
            plan.executable_sha256 or
            run_eval.sha256_files([str(plan.conf)]) != plan.conf_sha256 or
            run_eval.metric_contract_sha() != plan.metric_sha256 or
            sha256_file(plan.repo / "tools" / "sbsbench" /
                        "thresholds.json") != plan.thresholds_sha256 or
            sha256_file(plan.repo / "tools" / "sbsbench" /
                        "sbsbench.py") != plan.sbsbench_sha256 or
            sha256_file(plan.repo / "tools" / "sbsbench" /
                        "run_eval.py") != plan.run_eval_sha256 or
            sha256_file(plan.active_split) != plan.active_split_sha256 or
            (plan.scored_result_cache_root is not None and
             ordinal_result_cache.query_scorer_runtime_identity(plan.python) !=
             plan.scorer_runtime_identity) or
            _code_identities(plan.repo) != plan.code_identities):
        raise RuntimeError("ordinal plan input identity changed during execution")
    _update_runtime(
        plan.workspace, plan_sha256, state="running",
        current_step=None, current_phase="source-verification",
    )
    # Rehash the large native runtime once here. Child render/score processes
    # then reuse the stat-checked receipt during this invocation, while render
    # identities remain bound to the exact refreshed bytes and GPU/driver.
    run_multiscale_eval.depth_state_cache.runtime_snapshot(
        plan.build_dir, force_refresh=True
    )
    source_receipt = _verify_sources(plan)
    state_path = plan.workspace / "ordinal_orchestration_state.json"
    completed = []
    if state_path.is_file():
        state = load_json(state_path, "ordinal orchestration state")
        if (state.get("schema") != SCHEMA or
                state.get("contract") != CONTRACT or
                state.get("plan_sha256") != plan_sha256 or
                not isinstance(state.get("completed"), list)):
            raise RuntimeError("ordinal orchestration state is stale")
        completed = list(state["completed"])
        if len(completed) != len(set(completed)):
            raise RuntimeError("ordinal orchestration state repeats a step")
    known = {step.key for step in plan.steps}
    if any(key not in known for key in completed):
        raise RuntimeError("ordinal orchestration state contains an unknown step")
    stop_index = PHASES.index(args.stop_after)
    eligible = [
        step for step in plan.steps if PHASES.index(step.phase) <= stop_index
    ]
    logs_root = plan.workspace / "orchestration_logs"

    def record(step):
        if step.key not in completed:
            completed.append(step.key)
        write_json_atomic(state_path, {
            "schema": SCHEMA,
            "contract": CONTRACT,
            "plan_sha256": plan_sha256,
            "source_verification_sha256": sha256_file(source_receipt),
            "stop_after": args.stop_after,
            "completed": completed,
            "last_completed": step.key,
        })
        print(
            f"[progress] {len(completed)}/{len(eligible)} steps complete "
            f"({100.0 * len(completed) / len(eligible):.1f}%): {step.key}",
            flush=True,
        )
        _update_runtime(
            plan.workspace, plan_sha256, state="running",
            completed_steps=len(completed), total_steps=len(eligible),
            last_completed=step.key, current_step=None, current_phase=None,
        )

    index = 0
    while index < len(eligible):
        step = eligible[index]
        print(
            f"[progress] preparing step {index + 1}/{len(eligible)}: "
            f"{step.key}",
            flush=True,
        )
        _update_runtime(
            plan.workspace, plan_sha256, state="running",
            current_step=step.key, current_phase=step.phase,
            completed_steps=len(completed), total_steps=len(eligible),
        )
        if step.kind == "safety_batch":
            end = index + 1
            while (end < len(eligible) and
                   eligible[end].kind == step.kind):
                end += 1
            group = eligible[index:end]
            pending = []
            score_cache_contexts = {}
            for item in group:
                was_recorded = item.key in completed
                if _prepare_output(item, plan, args, completed):
                    if args.restart:
                        for output in item.metadata["scale_outputs"].values():
                            output = Path(output)
                            if output.exists():
                                _safe_remove(
                                    output, plan.build_dir / "sbs_eval"
                                )
                    context = _score_cache_context(plan, item)
                    score_cache_contexts[item.key] = context
                    cached = _materialize_scored_cache(context)
                    if cached is None:
                        pending.append(item)
                    else:
                        validate_safety_batch(item, require_compaction=True)
                        print(
                            f"[score-cache] authenticated hit: {item.key}",
                            flush=True,
                        )
                        record(item)
                elif not was_recorded:
                    # A fully authenticated publication can exist just before
                    # the state checkpoint if the process is interrupted.
                    # Adopt it instead of rerendering or dead-ending resume.
                    context = _score_cache_context(plan, item)
                    _publish_scored_cache(context)
                    record(item)
            # The GPU harness remains strictly serialized.  Once batch N is
            # rendered, its CPU-only scorer runs while the main thread renders
            # N+1.  We do not begin N+2 until N has scored and published, so at
            # most one complete rendered batch waits behind the active scorer.

            def authenticate_scored(item, future):
                try:
                    future.result()
                    validate_safety_batch(item, require_compaction=False)
                    for scalar in _scalar_safety_steps(item):
                        compact_safety_tree(
                            scalar.output,
                            scalar.metadata["clips"],
                            multiscale=True,
                            scale=scalar.metadata["scale"],
                        )
                        validate_compaction(scalar)
                    validate_safety_batch(item, require_compaction=True)
                    context = score_cache_contexts.get(item.key)
                    if context is None and plan.scored_result_cache_root is not None:
                        context = _score_cache_context(plan, item)
                    key = _publish_scored_cache(context)
                    if key is not None:
                        print(
                            f"[score-cache] published {key[:12]}: {item.key}",
                            flush=True,
                        )
                    record(item)
                except Exception as error:
                    raise RuntimeError(
                        f"ordinal score/publication failed at {item.key}: "
                        f"{error}"
                    ) from error

            _run_safety_pipeline(pending, logs_root, authenticate_scored)
            index = end
            continue

        if _prepare_output(step, plan, args, completed):
            if step.kind == "bundle":
                step.output.mkdir(parents=True, exist_ok=True)
                write_json_atomic(
                    step.metadata["run_manifest"],
                    _run_manifest_payload(step),
                )
                _run_step_process(step, logs_root)
                validate_bundle(step)
            elif step.kind == "source":
                step.output.parent.mkdir(parents=True, exist_ok=True)
                _run_step_process(step, logs_root)
                validate_source(step)
            elif step.kind == "catalog":
                catalog = build_catalog(plan)
                write_json_atomic(step.output, catalog)
                validate_catalog(plan, step.output, expected=catalog)
            else:
                raise RuntimeError(f"unknown ordinal step kind: {step.kind}")
        record(step)
        index += 1
    return {
        "plan": str(plan_path),
        "state": str(state_path),
        "source_verification": str(source_receipt),
        "stop_after": args.stop_after,
        "completed_steps": len(completed),
        "last_completed": completed[-1] if completed else None,
        "catalog": (
            str(plan.workspace / "ordinal_frame_label_catalog.json")
            if args.stop_after == "catalog" else None
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--active-split", required=True, type=Path)
    parser.add_argument(
        "--build-dir", type=Path,
        default=Path("cmake-build-relwithdebinfo"),
    )
    parser.add_argument(
        "--conf", type=Path, default=Path("tools/sbsbench/bench.conf")
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--depth-state-cache-root", type=Path,
        default=DEFAULT_DEPTH_STATE_CACHE_ROOT,
    )
    parser.add_argument(
        "--no-depth-state-cache", action="store_true",
        help="disable cross-workspace production depth-state reuse",
    )
    parser.add_argument(
        "--scored-result-cache-root", type=Path,
        default=DEFAULT_SCORED_RESULT_CACHE_ROOT,
    )
    parser.add_argument(
        "--no-scored-result-cache", action="store_true",
        help="disable cross-workspace compacted ordinal-result reuse",
    )
    parser.add_argument("--run-prefix", default="ordv2")
    parser.add_argument(
        "--production", action="append",
        help=(
            "working production ID to include; repeatable smoke-only subset "
            "unless all train/development productions are named"
        ),
    )
    parser.add_argument(
        "--clip-limit", type=int,
        help=(
            "use only the first N authenticated clips per selected production; "
            "marks the catalog smoke-only and not training-eligible"
        ),
    )
    parser.add_argument(
        "--score-workers", type=int, default=DEFAULT_SCALE_SCORE_JOBS,
        help=(
            "concurrent exact-scale scorers; each isolated scorer uses one "
            "metric worker"
        ),
    )
    parser.add_argument(
        "--render-workers", type=int, default=1,
        help="multiscale GPU batches; fixed at 1 to avoid inference contention",
    )
    parser.add_argument("--stop-after", choices=PHASES, default="catalog")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--restart", action="store_true",
        help="replace only stale partial outputs owned by this exact plan",
    )
    args = parser.parse_args(argv)
    if args.no_depth_state_cache:
        args.depth_state_cache_root = None
    if args.no_scored_result_cache:
        args.scored_result_cache_root = None
    if (args.score_workers < 1 or
            not 1 <= args.render_workers <= MAX_RENDER_WORKERS or
            (args.clip_limit is not None and args.clip_limit < 1)):
        parser.error(
            "score workers/clip limit must be positive and render workers "
            "must be 1"
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        # Freeze only a build that Ninja has proved current.  Otherwise the
        # first multiscale driver can rebuild sunshine.exe after the immutable
        # plan has already recorded its byte identity.
        run_eval.require_current_build(str(args.build_dir.resolve()))
        plan = build_plan(args)
        result = plan.as_dict() if args.dry_run else execute(plan, args)
    except (OSError, RuntimeError, ValueError,
            subprocess.SubprocessError) as error:
        raise SystemExit(f"ordinal orchestration failed: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
