#!/usr/bin/env python3
"""Replay a complete depth-coordinate-v2 shot timeline through Apollo's exact D3D SBS path.

Unlike the single-dump witness, this tool uploads every authenticated raw DAV2 field and cut signal
to one persistent native GPU state. Shared range/histogram passes plus six base V2 coordinate shaders
resolve the fixed raw coordinate and scene camera, apply one exact frame-local container, then
compute the vertical upper/lower envelopes, their
authenticated 75/25 share, and one row majorant before handing the final-parallax field directly
to the D3D renderer.
NumPy runs afterward as a comparison-only oracle. The native trace authenticates current-color flat
fallback, retained-camera semantics, cut attribution, calibration revisions, and the effective gain
used by each rendered output.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np
from PIL import Image

try:
    from . import direct_geometry_contract as direct_geometry
    from . import cut_state_contract
    from . import depth_coordinate_v2_dump_contract as dump_contract
    from . import whole_clip_raw_contract
    from . import prod_zipdepth_convex2x as convex2x_contract
    from . import prod_zipdepth_convex2x_diagnostics_contract as convex2x_diagnostics
    from .adaptive_state_contract import COUNTER_MAX as CUT_COUNTER_MAX
    from .depth_coordinate_v2_contract import (
        CONTRACT_CANONICAL_SHA256, CONTRACT_PATH, CONTRACT_SCHEMA)
    from .depth_mapping_v2 import (
        DIRECT_PARALLAX_SOURCE_U_LIMIT, MappingV2Config,
        decode_direct_parallax, encode_direct_parallax)
    from .depth_mapping_v2_temporal import (
        V2_STATE_TRACE_SCHEMA,
        _clip_sequence_input_contract,
        _load_cut_indices,
        _load_run_model_contract,
        _load_shape,
        _raw_files,
        generate_first_latch_exact_sequence,
        validate_v2_state_trace,
    )
except ImportError:  # Direct execution from tools/sbsbench.
    import direct_geometry_contract as direct_geometry  # type: ignore
    import cut_state_contract  # type: ignore
    import depth_coordinate_v2_dump_contract as dump_contract  # type: ignore
    import whole_clip_raw_contract  # type: ignore
    import prod_zipdepth_convex2x as convex2x_contract  # type: ignore
    import prod_zipdepth_convex2x_diagnostics_contract as convex2x_diagnostics  # type: ignore
    from adaptive_state_contract import COUNTER_MAX as CUT_COUNTER_MAX  # type: ignore
    from depth_coordinate_v2_contract import (  # type: ignore
        CONTRACT_CANONICAL_SHA256, CONTRACT_PATH, CONTRACT_SCHEMA)
    from depth_mapping_v2 import (  # type: ignore
        DIRECT_PARALLAX_SOURCE_U_LIMIT, MappingV2Config,
        decode_direct_parallax, encode_direct_parallax)
    from depth_mapping_v2_temporal import (  # type: ignore
        V2_STATE_TRACE_SCHEMA,
        _clip_sequence_input_contract,
        _load_cut_indices,
        _load_run_model_contract,
        _load_shape,
        _raw_files,
        generate_first_latch_exact_sequence,
        validate_v2_state_trace,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
# Exact replay compares a Pillow-decoded NumPy oracle with a WIC-decoded native replay. Keep this
# lossless-only so both sides attest the same decoded source pixels.
FRAME_PATTERN = re.compile(r"frame_(\d+)\.png$", re.IGNORECASE)
SEQUENCE_CONTRACT_SCHEMA = 21
GPU_INPUT_MANIFEST_SCHEMA = 10
GPU_INPUT_MANIFEST_MODE = "depth-coordinate-v2-production-gpu-sequence-v12"
SEQUENCE_MAPPING_CONFIG_KEYS = frozenset(asdict(MappingV2Config()).keys())
NUMPY_COMPARISON_SCHEMA = 4
PRODUCER_EVIDENCE_SCHEMA = 1
PRODUCER_EVIDENCE_BINDING = "self-contained-schema36-schema24-raw-producer-v1"
PRODUCER_EVIDENCE_DIR = "gpu_input/provenance"
SEQUENCE_CONTRACT_FILE = "depth_coordinate_v2_sequence_contract.json"
STATE_TRACE_FILE = "depth_coordinate_v2_state_trace.json"
GPU_INPUT_MANIFEST_FILE = "gpu_input/manifest.json"
NUMPY_COMPARISON_FILE = "depth_coordinate_v2_numpy_comparison.json"
IMPLEMENTATION_SOURCE_FILES = (
    "tools/sbsbench/contracts/depth-coordinate-v2-v1.json",
    "src/depth_coordinate_v2.h",
    "src/generated/depth_coordinate_v2_contract.h",
    "src/generated/sbs_adaptive_state_contract.h",
    "src/sbs_bench_depth_coordinate_v2.cpp",
    "src/sbs_bench_depth_coordinate_v2.h",
    "src/sbs_bench_harness.cpp",
    "src_assets/windows/assets/shaders/directx/depth_coordinate_v2_moments_cs.hlsl",
    "src_assets/windows/assets/shaders/directx/depth_coordinate_v2_frame_resolve_cs.hlsl",
    "src_assets/windows/assets/shaders/directx/depth_coordinate_v2_state_resolve_cs.hlsl",
    "src_assets/windows/assets/shaders/directx/depth_coordinate_v2_map_cs.hlsl",
    "src_assets/windows/assets/shaders/directx/depth_coordinate_v2_vertical_limit_cs.hlsl",
    "src_assets/windows/assets/shaders/directx/depth_coordinate_v2_limit_cs.hlsl",
    "src_assets/windows/assets/shaders/directx/sbs_direct_replay_ps.hlsl",
    "src_assets/windows/assets/shaders/directx/sbs_reprojection_v2_live_ps.hlsl",
    "src_assets/windows/assets/shaders/directx/sbs_reprojection_v2_diagnostics_ps.hlsl",
    "src_assets/windows/assets/shaders/directx/include/depth_color.hlsl",
    "src_assets/windows/assets/shaders/directx/include/depth_constants.hlsl",
    "src_assets/windows/assets/shaders/directx/include/depth_coordinate_v2.hlsl",
    "src_assets/windows/assets/shaders/directx/include/depth_coordinate_v2_contract.generated.hlsl",
    "src_assets/windows/assets/shaders/directx/include/sbs_adaptive_state_contract.generated.hlsl",
    "tools/sbsbench/depth_coordinate_v2_contract.py",
    "tools/sbsbench/depth_coordinate_v2_dump_contract.py",
    "tools/sbsbench/depth_mapping_v2.py",
    "tools/sbsbench/depth_mapping_v2_temporal.py",
    "tools/sbsbench/direct_geometry_contract.py",
    "tools/sbsbench/prod_zipdepth_convex2x.py",
    "tools/sbsbench/prod_zipdepth_convex2x_diagnostics_contract.py",
    "tools/sbsbench/contracts/prod-zipdepth-convex2x-v2.json",
    "tools/sbsbench/run_eval.py",
    "tools/sbsbench/cut_state_contract.py",
    "tools/sbsbench/whole_clip_raw_contract.py",
    "tools/sbsbench/replay_depth_mapping_v2_sequence.py",
)
RENDERER_SCORE_CONTRACT_FILE = "renderer_quality_score_contract.json"
RENDERER_SCORE_CONTRACT_SCHEMA = 3
RENDERER_SCORECARD_FILE = "renderer_quality_scorecard.json"
FORBIDDEN_RENDERER_SCORE_PREFIXES = ("shot_state_",)


def _implementation_sources() -> list[Dict[str, str]]:
    return [
        {
            "file": relative,
            "sha256": direct_geometry.file_sha256(str(REPO / relative)),
        }
        for relative in IMPLEMENTATION_SOURCE_FILES
    ]


def _metric_contract_evidence() -> Dict[str, object]:
    """Bind the evaluator's one canonical automatic-metric source contract.

    ``run_eval`` owns this list because baseline validity and standalone scoring must use the
    same definition.  Preserve the caller's numeric-runtime environment while importing it:
    run_eval pins CPU kernels for its own worker orchestration at module import time, whereas this
    replay tool must not leak those evaluator-only limits into the native harness child.
    """

    numeric_environment = (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    saved_environment = {key: os.environ.get(key) for key in numeric_environment}
    try:
        if __package__:
            from . import run_eval as benchmark_runner
        else:
            import run_eval as benchmark_runner  # type: ignore
    finally:
        for key, value in saved_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    paths = [Path(path).resolve() for path in benchmark_runner.metric_contract_files()]
    sources: list[Dict[str, str]] = []
    for path in paths:
        try:
            relative = path.relative_to(REPO.resolve())
        except ValueError as exc:
            raise ValueError(f"metric-contract source escapes repository: {path}") from exc
        sources.append({
            "file": relative.as_posix(),
            "sha256": benchmark_runner.sha256_files([str(path)]),
        })
    return {
        "authority": "run_eval.metric_contract_files-and-sha-v1",
        "sha256": benchmark_runner.metric_contract_sha(),
        "sources": sources,
    }


def _validate_metric_contract_evidence(value: object) -> None:
    if value != _metric_contract_evidence():
        raise ValueError("sequence automatic-metric source contract is stale or incomplete")


def _diagnostic_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "role": "non-controlling-fixed-scale-camera-audit-v4",
        "maximum_abs_candidate_center_drift_u": max(
            abs(float(row["candidate_center_drift_u"])) for row in rows),
        "maximum_abs_predicted_zero_translation_source_u": max(
            abs(float(row["predicted_zero_translation_source_u"])) for row in rows),
        "minimum_active_effective_gain": min(
            (float(row["effective_gain"]) for row in rows if row["frame_valid"]),
            default=0.0),
        "maximum_conditioner_raised_fraction": max(
            float(row["conditioner_raised_fraction"]) for row in rows),
        "maximum_conditioner_raise_source_u": max(
            float(row["conditioner_max_raise_source_u"]) for row in rows),
        "maximum_conditioner_lowered_fraction": max(
            float(row["conditioner_lowered_fraction"]) for row in rows),
        "maximum_conditioner_lower_source_u": max(
            float(row["conditioner_max_lower_source_u"]) for row in rows),
        "maximum_final_horizontal_slope": max(
            float(row["final_horizontal_slope_max"]) for row in rows),
        "maximum_final_vertical_shear": max(
            float(row["final_vertical_shear_max"]) for row in rows),
        "collapsed_frames": sum(bool(row["collapsed"]) for row in rows),
        "flat_unusable_frames": sum(not bool(row["frame_valid"]) for row in rows),
        "retained_camera_unusable_frames": sum(
            not bool(row["frame_valid"]) and bool(row["camera_valid"]) for row in rows),
        "camera_clear_unusable_frames": sum(
            not bool(row["frame_valid"]) and not bool(row["camera_valid"]) for row in rows),
        "confirmed_cuts": sum(bool(row["confirmed_cut"]) for row in rows),
        "calibration_revisions": max(int(row["calibration_revision"]) for row in rows),
    }


def _publish_renderer_quality_score(
        scorecard_path: Path,
        output: Path) -> Dict[str, object]:
    """Validate and attest the scorer's direct renderer-quality output."""

    renderer_path = output / RENDERER_SCORECARD_FILE
    if scorecard_path.resolve() != renderer_path.resolve():
        raise ValueError("renderer scorecard must use the canonical output path")
    scorecard = _read_json(scorecard_path)
    if (set(scorecard) != {"aggregate", "frames"} or
            not isinstance(scorecard["aggregate"], dict) or
            not isinstance(scorecard["frames"], list) or
            any(not isinstance(row, dict) for row in scorecard["frames"])):
        raise ValueError("renderer scorecard has an unknown schema")
    all_keys = set(scorecard["aggregate"])
    for row in scorecard["frames"]:
        all_keys.update(row)
    forbidden = sorted(
        key for key in all_keys
        if any(key.startswith(prefix) for prefix in FORBIDDEN_RENDERER_SCORE_PREFIXES))
    if forbidden:
        raise ValueError(
            f"renderer scorecard contains controller-state metrics: {forbidden}")

    metric_contract = _metric_contract_evidence()
    contract = {
        "schema": RENDERER_SCORE_CONTRACT_SCHEMA,
        "scope": "renderer-output-quality-only-v1",
        "renderer_scorecard": {
            "file": RENDERER_SCORECARD_FILE,
            "sha256": direct_geometry.file_sha256(str(renderer_path)),
        },
        "forbidden_metric_prefixes": list(FORBIDDEN_RENDERER_SCORE_PREFIXES),
        "v2_state_authority": {
            "file": STATE_TRACE_FILE,
            "schema": V2_STATE_TRACE_SCHEMA,
        },
        "metric_contract": metric_contract,
        "reason": "controller state is authenticated only by the v2 state trace",
    }
    contract_path = output / RENDERER_SCORE_CONTRACT_FILE
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "complete",
        "scope": "renderer-output-quality-only-v1",
        "scorecard_file": RENDERER_SCORECARD_FILE,
        "scorecard_sha256": direct_geometry.file_sha256(str(renderer_path)),
        "score_contract_file": RENDERER_SCORE_CONTRACT_FILE,
        "score_contract_sha256": direct_geometry.file_sha256(str(contract_path)),
        "metric_contract": metric_contract,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _new_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path is not a directory: {path}")
        try:
            next(path.iterdir())
        except StopIteration:
            return
        raise ValueError(f"output directory must be new or empty: {path}")
    path.mkdir(parents=True)


def _source_frames(path: Path) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    for candidate in path.iterdir():
        if not candidate.is_file():
            continue
        match = FRAME_PATTERN.fullmatch(candidate.name)
        if not match:
            continue
        frame_id = int(match.group(1))
        if frame_id in result:
            raise ValueError(f"duplicate source frame ID {frame_id} in {path}")
        result[frame_id] = candidate
    if not result:
        raise ValueError(f"no lossless frame_<id>.png source frames in {path}")
    return result


def _load_raw_fields(raw_sequence: Path) -> tuple[list[int], list[np.ndarray], tuple[int, int]]:
    shape = _load_shape(raw_sequence)
    files = _raw_files(raw_sequence)
    if not files:
        raise ValueError(f"{raw_sequence}: no raw_<id>.f32 fields")
    frame_ids = [frame_id for frame_id, _ in files]
    expected_ids = list(range(frame_ids[0], frame_ids[0] + len(frame_ids)))
    if frame_ids != expected_ids:
        raise ValueError(
            "sparse raw frame IDs require authenticated timestamps; "
            f"expected {expected_ids}, got {frame_ids}")
    expected_values = shape[0] * shape[1]
    fields: list[np.ndarray] = []
    for frame_id, path in files:
        values = np.fromfile(path, dtype="<f4")
        if values.size != expected_values:
            raise ValueError(
                f"{path}: expected {expected_values} float32 values, got {values.size}")
        # Preserve invalid model output exactly. The experimental frame-resolve shader owns the
        # finite-input decision and publishes flat while retaining a no-cut scene camera.
        fields.append(values.reshape(shape))
    return frame_ids, fields, shape


def _run_checked(command: list[str], cwd: Path, log_path: Path) -> None:
    environment = os.environ.copy()
    environment.setdefault("SUNSHINE_SBS_BENCH", "1")
    try:
        completed = subprocess.run(
            command, cwd=cwd, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=900, env=environment, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot run {' '.join(command)}: {exc}") from exc
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {' '.join(command)}\n{tail}")


def _load_authenticated_cut_pulses(
        raw_sequence: Path,
        frame_ids: Sequence[int],
        cut_counts: Sequence[int]) -> list[bool]:
    """Return emitted one-frame pulses without conflating them with generation recovery.

    ``_load_cut_indices`` deliberately treats a generation increment as a cut even when the
    transient pulse was dropped.  The native state shader needs both original signals so this
    replay proves that its persistent count path recovers that exact count-only case.
    """

    if len(frame_ids) != len(cut_counts):
        raise ValueError("cut pulse frame/count sequences have different lengths")
    state_path = raw_sequence / "cut_state.json"
    if not state_path.exists():
        return [False] * len(frame_ids)
    trace = cut_state_contract.load_trace(str(state_path))
    if set(trace) != set(frame_ids):
        raise ValueError(
            f"{state_path}: trace/raw frame IDs disagree while loading cut pulses")
    pulses: list[bool] = []
    for index, frame_id in enumerate(frame_ids):
        emitted_count = int(trace[frame_id]["hard_cut_count"])
        if emitted_count != int(cut_counts[index]):
            raise ValueError(
                f"{state_path}: cut-count evidence changed while loading frame {frame_id}")
        pulses.append(trace[frame_id]["hard_cut_pulse"] > 0.5)
    return pulses


def _materialize_gpu_replay_inputs(
        output: Path,
        sources: Dict[int, Path],
        raw_paths: Dict[int, Path],
        frame_ids: Sequence[int],
        cut_counts: Sequence[int],
        cut_pulses: Sequence[bool],
        cut_source: str,
        shape: tuple[int, int],
        run_model: Dict[str, object],
        config: MappingV2Config) -> tuple[list[Dict[str, object]], Path]:
    """Bind current color, exact raw bytes, and cut generations for native GPU replay."""

    if not (len(frame_ids) == len(cut_counts) == len(cut_pulses)):
        raise ValueError("GPU replay frame and cut sequences have different lengths")
    input_frames = output / "input_frames"
    replay_frames = output / "frames"
    gpu_input = output / "gpu_input"
    input_frames.mkdir()
    replay_frames.mkdir()
    gpu_input.mkdir(exist_ok=True)
    contract_copy = gpu_input / "contracts" / CONTRACT_PATH.name
    contract_copy.parent.mkdir()
    shutil.copyfile(CONTRACT_PATH, contract_copy)
    rows: list[Dict[str, object]] = []
    manifest_frames: list[Dict[str, object]] = []
    source_shape: Optional[Dict[str, int]] = None
    for index, frame_id in enumerate(frame_ids):
        frame_text = f"{frame_id:05d}"
        source = sources[frame_id]
        source_name = f"frame_{frame_text}{source.suffix.lower()}"
        input_path = input_frames / source_name
        rendered_path = replay_frames / source_name
        # One immutable byte snapshot owns the hash, both evidence copies, and the dimensions
        # later authenticated by the native SRV contract. Two independent copy/read operations
        # would leave a path-replacement window between supposedly identical authorities.
        source_bytes = source.read_bytes()
        if not source_bytes:
            raise ValueError(f"empty exact source frame: {source}")
        try:
            with Image.open(io.BytesIO(source_bytes)) as image:
                image.load()
                source_width, source_height = image.size
        except Exception as exc:
            raise ValueError(f"invalid exact PNG source frame: {source}") from exc
        current_source_shape = {"width": source_width, "height": source_height}
        if (source_width < 1 or source_height < 1 or
                source_width > 16384 or source_height > 16384):
            raise ValueError(f"unsupported exact source dimensions: {source}")
        if source_shape is None:
            source_shape = current_source_shape
        elif source_shape != current_source_shape:
            raise ValueError("exact replay source frames have mixed dimensions")
        _exact_source_capture_grid_kind(
            source_width, source_height, shape[1], shape[0])
        input_path.write_bytes(source_bytes)
        rendered_path.write_bytes(source_bytes)
        raw_name = f"raw_{frame_text}.f32"
        raw_copy = gpu_input / raw_name
        shutil.copyfile(raw_paths[frame_id], raw_copy)
        raw_sha = direct_geometry.file_sha256(str(raw_copy))
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        manifest_frames.append({
            "frame_id": frame_text,
            "raw_file": raw_name,
            "raw_sha256": raw_sha,
            "source_sha256": source_sha,
            "hard_cut_count": int(cut_counts[index]),
            "hard_cut_pulse": bool(cut_pulses[index]),
        })
        rows.append({
            "frame_id": frame_text,
            "input_source_file": str(input_path.relative_to(output)).replace("\\", "/"),
            "input_source_sha256": source_sha,
            "rendered_source_frame_id": frame_text,
            "rendered_source_file": str(rendered_path.relative_to(output)).replace("\\", "/"),
            "rendered_source_sha256": direct_geometry.file_sha256(str(rendered_path)),
            "raw_depth_sha256": raw_sha,
            "order_sha256": None,
            "parallax_sha256": None,
        })
    manifest = {
        "schema": GPU_INPUT_MANIFEST_SCHEMA,
        "mode": GPU_INPUT_MANIFEST_MODE,
        "calibration_contract": {
            "file": str(CONTRACT_PATH.relative_to(SCRIPT_DIR)).replace("\\", "/"),
            "schema": CONTRACT_SCHEMA,
            "sha256": direct_geometry.file_sha256(str(CONTRACT_PATH)),
        },
        "model_identity": {
            "calibration_id": run_model["calibration_id"],
            "model": run_model["model"],
            "depth_model_url": run_model["depth_model_url"],
            "onnx_sha256": run_model["onnx_sha256"],
            "preprocess_profile": run_model["preprocess_profile"],
            "preprocess_source_closure_sha256":
                run_model["preprocess_source_closure_sha256"],
        },
        "raw_shape": {
            "width": shape[1], "height": shape[0],
            "dtype": "float32-le", "layout": "row-major",
        },
        "source_shape": source_shape,
        "mapping_config": asdict(config),
        "cut_source": cut_source,
        "frames": manifest_frames,
    }
    manifest_path = gpu_input / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, manifest_path


def _exact_source_capture_grid_kind(
        source_width: int, source_height: int,
        grid_width: int, grid_height: int) -> str:
    """Require one exact source-derived legacy-coarse or fused-high production profile."""

    actual = grid_width, grid_height
    for scale, kind in ((1, "legacy-dav2"), (2, "single-high-convex2x")):
        expected = dump_contract.expected_capture_grid_for_source(
            source_width, source_height, scale=scale)
        if expected == actual:
            return kind
    raise ValueError(
        "replay depth grid does not match the exact source-derived production profile")


def _materialize_producer_evidence_bundle(
        output: Path,
        raw_sequence: Path,
        run_model: Dict[str, object]) -> Dict[str, Any]:
    """Copy the exact source-run authorities so replay survives deletion of the eval run."""

    evidence_dir = output / PRODUCER_EVIDENCE_DIR
    evidence_dir.mkdir(parents=True)
    source_paths = {
        "results_json": raw_sequence.parent / "results.json",
        "harness_contract": raw_sequence / "contract.json",
        "raw_shape": raw_sequence / "raw_shape.json",
        "cut_state": raw_sequence / "cut_state.json",
    }
    filenames = {
        "results_json": "results.json",
        "harness_contract": "contract.json",
        "raw_shape": "raw_shape.json",
        "cut_state": "cut_state.json",
    }
    references: Dict[str, Dict[str, str]] = {}
    for key, source in source_paths.items():
        destination = evidence_dir / filenames[key]
        shutil.copyfile(source, destination)
        references[key] = {
            "file": str(destination.relative_to(output)).replace("\\", "/"),
            "sha256": direct_geometry.file_sha256(str(destination)),
        }
    manifest = run_model["whole_clip_raw_artifacts"].get(raw_sequence.name)
    if manifest is None:
        raise ValueError("source results omit this clip's raw artifact manifest")
    manifest_path = evidence_dir / "raw_artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    references["raw_artifact_manifest"] = {
        "file": str(manifest_path.relative_to(output)).replace("\\", "/"),
        "sha256": direct_geometry.file_sha256(str(manifest_path)),
    }
    if references["results_json"]["sha256"] != run_model["results_sha256"]:
        raise ValueError("source results.json changed while materializing replay evidence")
    return {
        "schema": PRODUCER_EVIDENCE_SCHEMA,
        "binding": PRODUCER_EVIDENCE_BINDING,
        "clip": raw_sequence.name,
        **references,
    }


def _numpy_oracle_sequence(
        raw_fields: Sequence[np.ndarray],
        frame_ids: Sequence[int],
        cuts: Sequence[bool],
        cut_counts: Sequence[int],
        cut_source: str,
        config: MappingV2Config,
        ) -> tuple[list[np.ndarray], list[np.ndarray], list[Optional[Dict[str, object]]]]:
    """Produce comparison-only fields with the same retained-camera state machine."""

    mapped = generate_first_latch_exact_sequence(
        [np.asarray(field, dtype=np.float64) for field in raw_fields],
        frame_ids,
        cuts,
        cut_source,
        config,
        confirmed_cut_counts=cut_counts,
    )
    return (
        [np.asarray(field, dtype="<f4") for field in mapped.canonical_fields],
        [encode_direct_parallax(field) for field in mapped.parallax_fields],
        list(mapped.state_trace["frames"]),
    )


def _compare_gpu_with_numpy(
        output: Path,
        raw_fields: Sequence[np.ndarray],
        frame_ids: Sequence[int],
        cuts: Sequence[bool],
        cut_counts: Sequence[int],
        cut_source: str,
        config: MappingV2Config,
        gpu_trace: Dict[str, object]) -> Dict[str, object]:
    """Fail closed when native GPU fields leave explicit float32 oracle tolerances."""

    tolerances = {
        "canonical_max_abs": 5.0e-4,
        "encoded_parallax_max_abs": 1.0e-4,
        "state_scalar_max_abs": 5.0e-4,
    }
    oracle_orders, oracle_encoded, oracle_rows = _numpy_oracle_sequence(
        raw_fields, frame_ids, cuts, cut_counts, cut_source, config)
    gpu_rows = gpu_trace["frames"]
    state_fields = (
        "center", "inverse_scale", "latched_scale", "convergence_curve",
        "container_scale", "effective_gain", "observed_mean", "observed_std",
        "observed_raw_minimum", "observed_raw_maximum",
    )
    comparison_rows: list[Dict[str, object]] = []
    for index, frame_id in enumerate(frame_ids):
        frame_text = f"{frame_id:05d}"
        gpu_order_path = output / "harness" / f"depth_{frame_text}.f32"
        gpu_parallax_path = output / "harness" / f"parallax_{frame_text}.f32"
        gpu_order = np.fromfile(gpu_order_path, dtype="<f4").reshape(raw_fields[index].shape)
        gpu_encoded = np.fromfile(
            gpu_parallax_path, dtype="<f4").reshape(raw_fields[index].shape)
        order_error = float(np.max(np.abs(
            gpu_order.astype(np.float64) - oracle_orders[index].astype(np.float64))))
        parallax_error = float(np.max(np.abs(
            gpu_encoded.astype(np.float64) - oracle_encoded[index].astype(np.float64))))
        input_valid = bool(np.isfinite(raw_fields[index]).all())
        state_error = 0.0
        if input_valid:
            oracle_row = oracle_rows[index]
            if oracle_row is None:
                raise RuntimeError(f"NumPy oracle omitted finite frame {frame_text}")
            state_error = max(abs(
                float(gpu_rows[index][field]) - float(oracle_row[field]))
                for field in state_fields)
        else:
            if (gpu_rows[index].get("input_valid") is not False or
                    gpu_rows[index].get("frame_valid") is not False or
                    gpu_rows[index].get("collapsed") is not False):
                raise RuntimeError(
                    f"GPU replay did not distinguish invalid raw input at frame {frame_text}")
        within = (order_error <= tolerances["canonical_max_abs"] and
                  parallax_error <= tolerances["encoded_parallax_max_abs"] and
                  state_error <= tolerances["state_scalar_max_abs"])
        if not within:
            raise RuntimeError(
                f"GPU/NumPy v2 mismatch at frame {frame_text}: order={order_error:.7g}, "
                f"parallax={parallax_error:.7g}, state={state_error:.7g}")
        numpy_order_bytes = np.asarray(oracle_orders[index], dtype="<f4").tobytes()
        numpy_parallax_bytes = np.asarray(oracle_encoded[index], dtype="<f4").tobytes()
        comparison_rows.append({
            "frame_id": frame_text,
            "input_valid": input_valid,
            "gpu_order_sha256": direct_geometry.file_sha256(str(gpu_order_path)),
            "numpy_order_sha256": hashlib.sha256(numpy_order_bytes).hexdigest(),
            "order_exact_hash_match": hashlib.sha256(numpy_order_bytes).hexdigest() ==
                                      direct_geometry.file_sha256(str(gpu_order_path)),
            "canonical_max_abs_error": order_error,
            "gpu_parallax_sha256": direct_geometry.file_sha256(str(gpu_parallax_path)),
            "numpy_parallax_sha256": hashlib.sha256(numpy_parallax_bytes).hexdigest(),
            "parallax_exact_hash_match": hashlib.sha256(numpy_parallax_bytes).hexdigest() ==
                                         direct_geometry.file_sha256(str(gpu_parallax_path)),
            "encoded_parallax_max_abs_error": parallax_error,
            "state_scalar_max_abs_error": state_error,
            "within_float32_tolerances": within,
        })
    document = {
        "schema": NUMPY_COMPARISON_SCHEMA,
        "role": "numpy-comparison-oracle-only-v2",
        "render_authority": "native-six-shader-gpu-output",
        "tolerances": tolerances,
        "frames": comparison_rows,
        "all_within_float32_tolerances": True,
    }
    path = output / NUMPY_COMPARISON_FILE
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _validate_harness_geometry(
        harness: Path,
        frame_ids: Sequence[int],
        shape: tuple[int, int]) -> Dict[str, Any]:
    contract = _read_json(harness / "contract.json")
    try:
        validated = direct_geometry.validate_artifacts(
            str(harness), contract, set(frame_ids))
    except ValueError as exc:
        raise RuntimeError(
            f"harness did not attest the complete direct-geometry sequence: {exc}") from exc
    fields = validated["manifest"]["fields"]
    if [int(field["frame_id"]) for field in fields] != list(frame_ids):
        raise RuntimeError("harness reordered the direct-geometry sequence")
    for field, frame_id in zip(fields, frame_ids):
        frame_text = f"{frame_id:05d}"
        parallax = harness / f"parallax_{frame_text}.f32"
        order = harness / f"depth_{frame_text}.f32"
        if (field["width"] != shape[1] or field["height"] != shape[0] or
                field["parallax_sha256"] != direct_geometry.file_sha256(str(parallax)) or
                field["order_sha256"] != direct_geometry.file_sha256(str(order)) or
                validated["shapes"].get(frame_id) != shape):
            raise RuntimeError(
                f"harness did not authenticate exact fields for frame {frame_text}")
    return contract


def _relative_file(output: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"sequence contract has invalid {label} path")
    resolved = (output / value).resolve()
    try:
        resolved.relative_to(output.resolve())
    except ValueError as exc:
        raise ValueError(f"sequence contract {label} escapes output root") from exc
    return resolved


def _validate_frame_source_attestation(
        output: Path, row: Dict[str, object], frame_text: str) -> None:
    """Rehash both copies and require rendered color to remain the current source frame."""

    for file_key, hash_key, directory, label in (
            ("input_source_file", "input_source_sha256", "input_frames", "input source"),
            ("rendered_source_file", "rendered_source_sha256", "frames", "rendered source")):
        relative = Path(str(row.get(file_key, "")))
        if (len(relative.parts) != 2 or relative.parts[0] != directory or
                relative.stem != f"frame_{frame_text}" or
                relative.suffix.lower() != ".png"):
            raise ValueError(
                f"sequence contract frame {frame_text} has invalid {label} identity")
        path = _relative_file(output, row.get(file_key), label)
        if direct_geometry.file_sha256(str(path)) != row.get(hash_key):
            raise ValueError(f"sequence contract frame {frame_text} {label} hash mismatch")

    if row["input_source_file"] == row["rendered_source_file"]:
        raise ValueError(
            f"sequence contract frame {frame_text} aliases input and rendered source files")
    if (row.get("rendered_source_frame_id") != frame_text or
            row.get("input_source_sha256") != row.get("rendered_source_sha256")):
        raise ValueError(
            f"sequence contract frame {frame_text} did not render its current source color")


def _validate_gpu_input_manifest_evidence(
        output: Path,
        reference: Any,
        mapping: Dict[str, Any],
        cut_source: Any,
        input_contract: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the current manifest reference and one non-contradictory authority."""

    if (not isinstance(reference, dict) or
            set(reference) != {"file", "schema", "sha256"} or
            reference.get("file") != GPU_INPUT_MANIFEST_FILE or
            reference.get("schema") != GPU_INPUT_MANIFEST_SCHEMA):
        raise ValueError("sequence contract has invalid GPU input manifest reference")
    reference_path = _relative_file(output, reference["file"], "GPU input manifest")
    if direct_geometry.file_sha256(str(reference_path)) != reference.get("sha256"):
        raise ValueError("sequence contract GPU input manifest hash mismatch")
    manifest = _read_json(reference_path)
    expected_root = {
        "schema", "mode", "calibration_contract", "model_identity", "raw_shape",
        "source_shape", "mapping_config", "cut_source", "frames",
    }
    if (set(manifest) != expected_root or
            manifest.get("schema") != GPU_INPUT_MANIFEST_SCHEMA or
            manifest.get("mode") != GPU_INPUT_MANIFEST_MODE or
            manifest.get("mapping_config") != mapping or
            manifest.get("cut_source") != cut_source):
        raise ValueError("GPU input manifest does not bind the selected mapping/cut authority")
    expected_identity = {
        "calibration_id": input_contract.get("calibration_id"),
        "model": input_contract.get("model"),
        "depth_model_url": input_contract.get("depth_model_url"),
        "onnx_sha256": input_contract.get("onnx_sha256"),
        "preprocess_profile": input_contract.get("preprocess_profile"),
        "preprocess_source_closure_sha256":
            input_contract.get("preprocess_source_closure_sha256"),
    }
    if manifest.get("model_identity") != expected_identity:
        raise ValueError("GPU input manifest model identity disagrees with producer evidence")
    input_shape = input_contract.get("raw_shape")
    if (not isinstance(input_shape, dict) or set(input_shape) != {"height", "width"} or
            type(input_shape.get("height")) is not int or input_shape["height"] < 1 or
            type(input_shape.get("width")) is not int or input_shape["width"] < 1):
        raise ValueError("sequence input contract has an invalid raw shape")
    expected_shape = {
        "width": input_shape["width"],
        "height": input_shape["height"],
        "dtype": "float32-le", "layout": "row-major",
    }
    if manifest.get("raw_shape") != expected_shape:
        raise ValueError("GPU input manifest raw shape disagrees with producer evidence")
    source_shape = manifest.get("source_shape")
    if (not isinstance(source_shape, dict) or set(source_shape) != {"width", "height"} or
            type(source_shape.get("width")) is not int or
            type(source_shape.get("height")) is not int or
            source_shape["width"] < 1 or source_shape["width"] > 16384 or
            source_shape["height"] < 1 or source_shape["height"] > 16384):
        raise ValueError("GPU input manifest has an invalid source shape")
    _exact_source_capture_grid_kind(
        source_shape["width"], source_shape["height"],
        expected_shape["width"], expected_shape["height"])
    calibration_ref = manifest.get("calibration_contract")
    if (not isinstance(calibration_ref, dict) or
            set(calibration_ref) != {"file", "schema", "sha256"} or
            calibration_ref.get("file") !=
            str(CONTRACT_PATH.relative_to(SCRIPT_DIR)).replace("\\", "/") or
            calibration_ref.get("schema") != CONTRACT_SCHEMA or
            not isinstance(calibration_ref.get("sha256"), str) or
            re.fullmatch(r"[0-9a-f]{64}", calibration_ref["sha256"]) is None):
        raise ValueError("GPU input manifest has an invalid mapping calibration contract")
    calibration_path = (reference_path.parent / calibration_ref["file"]).resolve()
    if (reference_path.parent.resolve() not in calibration_path.parents or
            direct_geometry.file_sha256(str(calibration_path)) != calibration_ref["sha256"]):
        raise ValueError("GPU input manifest calibration contract bytes are not authenticated")
    calibration_document = _read_json(calibration_path)
    measured_canonical = hashlib.sha256(json.dumps(
        calibration_document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if (calibration_document.get("schema") != CONTRACT_SCHEMA or
            measured_canonical != CONTRACT_CANONICAL_SHA256):
        raise ValueError("GPU input manifest mapping calibration semantics are not canonical")

    frames = manifest.get("frames")
    selected = input_contract.get("selected_frame_ids")
    raw_records = input_contract.get("ordered_raw_fields")
    expected_frame_keys = {
        "frame_id", "raw_file", "raw_sha256", "source_sha256", "hard_cut_count",
        "hard_cut_pulse",
    }
    if (not isinstance(frames, list) or not isinstance(selected, list) or
            not isinstance(raw_records, list) or
            len(frames) != len(selected) or len(frames) != len(raw_records)):
        raise ValueError("GPU input manifest has an incomplete ordered frame set")
    expected_bytes = expected_shape["width"] * expected_shape["height"] * 4
    declared_raw_files = set()
    for index, (row, frame_id, raw_record) in enumerate(zip(frames, selected, raw_records)):
        expected_file = f"raw_{frame_id}.f32"
        if (not isinstance(row, dict) or set(row) != expected_frame_keys or
                row.get("frame_id") != frame_id or row.get("raw_file") != expected_file or
                row.get("raw_sha256") != raw_record.get("sha256") or
                not isinstance(row.get("source_sha256"), str) or
                re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]) is None or
                raw_record.get("frame_id") != frame_id or
                type(row.get("hard_cut_count")) is not int or
                not 0 <= row["hard_cut_count"] <= CUT_COUNTER_MAX or
                type(row.get("hard_cut_pulse")) is not bool):
            raise ValueError(f"GPU input manifest frame {index} has invalid evidence")
        raw_path = reference_path.parent / expected_file
        try:
            raw_size = raw_path.stat().st_size
        except OSError as exc:
            raise ValueError(f"GPU input manifest raw tensor is missing: {raw_path}") from exc
        if (raw_size != expected_bytes or
                direct_geometry.file_sha256(str(raw_path)) != row["raw_sha256"]):
            raise ValueError(f"GPU input manifest raw tensor {frame_id} is not authenticated")
        source_candidates = list((output / "input_frames").glob(f"frame_{frame_id}.*"))
        if (len(source_candidates) != 1 or
                direct_geometry.file_sha256(str(source_candidates[0])) !=
                row["source_sha256"]):
            raise ValueError(
                f"GPU input manifest source color {frame_id} is not authenticated")
        try:
            with Image.open(source_candidates[0]) as source_image:
                source_image.load()
                measured_source_shape = {
                    "width": source_image.width, "height": source_image.height}
        except Exception as exc:
            raise ValueError(
                f"GPU input manifest source color {frame_id} is not decodable") from exc
        if measured_source_shape != source_shape:
            raise ValueError(
                f"GPU input manifest source color {frame_id} shape is not authenticated")
        declared_raw_files.add(expected_file)
    actual_raw_files = {path.name for path in reference_path.parent.glob("raw_*.f32")}
    if actual_raw_files != declared_raw_files:
        raise ValueError("GPU input manifest raw tensor set is incomplete or ambiguous")
    return manifest


def _validate_producer_evidence_bundle(
        output: Path,
        reference: Any,
        input_contract: Dict[str, Any],
        gpu_manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild trust from copied producer bytes, without consulting the original eval tree."""

    expected_keys = {
        "schema", "binding", "clip", "results_json", "harness_contract", "raw_shape",
        "raw_artifact_manifest", "cut_state",
    }
    if (not isinstance(reference, dict) or set(reference) != expected_keys or
            reference.get("schema") != PRODUCER_EVIDENCE_SCHEMA or
            reference.get("binding") != PRODUCER_EVIDENCE_BINDING or
            not isinstance(reference.get("clip"), str) or not reference["clip"]):
        raise ValueError("sequence producer evidence bundle has missing or unknown semantics")
    documents: Dict[str, Dict[str, Any]] = {}
    document_paths: Dict[str, Path] = {}
    for key in ("results_json", "harness_contract", "raw_shape",
                "raw_artifact_manifest", "cut_state"):
        item = reference.get(key)
        if (not isinstance(item, dict) or set(item) != {"file", "sha256"} or
                not isinstance(item.get("sha256"), str) or
                re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None):
            raise ValueError(f"sequence producer evidence has an invalid {key} reference")
        path = _relative_file(output, item.get("file"), f"producer {key}")
        if direct_geometry.file_sha256(str(path)) != item["sha256"]:
            raise ValueError(f"sequence producer evidence {key} hash mismatch")
        document_paths[key] = path
        documents[key] = _read_json(path)

    results = documents["results_json"]
    results_meta = results.get("meta") if isinstance(results, dict) else None
    results_clips = results.get("clips") if isinstance(results, dict) else None
    clip = reference["clip"]
    if (not isinstance(results_meta, dict) or
            results_meta.get("eval_schema") != whole_clip_raw_contract.EVALUATOR_SCHEMA or
            not isinstance(results_clips, dict) or clip not in results_clips or
            reference["results_json"]["sha256"] !=
            input_contract.get("results_json_sha256")):
        raise ValueError("sequence copied results.json is not the authenticated source run")
    raw_manifests = results_meta.get(whole_clip_raw_contract.RESULTS_META_KEY)
    manifest = documents["raw_artifact_manifest"]
    if (not isinstance(raw_manifests, dict) or raw_manifests.get(clip) != manifest):
        raise ValueError("sequence copied raw manifest is not embedded in copied results.json")
    whole_clip_raw_contract.validate_manifest(manifest)
    clip_entry = results_clips[clip]
    expected_clip_identity = {
        "calibration_status": manifest["calibration_status"],
        "calibration_id": manifest["calibration_id"],
        "preprocess_profile": manifest["producer_model_identity"]["preprocess_profile"],
        "raw_shape": manifest["raw_shape"],
    }
    if (not isinstance(clip_entry, dict) or
            not isinstance(clip_entry.get("meta"), dict) or
            clip_entry["meta"].get("raw_model_identity") != expected_clip_identity):
        raise ValueError("sequence copied results omit the clip-level raw identity summary")
    if (manifest.get("calibration_status") != "calibrated" or
            manifest.get("calibration_id") != input_contract.get("calibration_id")):
        raise ValueError("sequence copied raw manifest is not calibrated for replay")

    contract = documents["harness_contract"]
    if (contract.get("schema") != whole_clip_raw_contract.HARNESS_CONTRACT_SCHEMA or
            reference["harness_contract"]["sha256"] !=
            input_contract.get("contract_json_sha256") or
            contract.get("raw_model_provenance") != manifest.get("producer_model_identity")):
        raise ValueError("sequence copied harness contract disagrees with raw provenance")
    raw_shape = documents["raw_shape"]
    expected_shape = manifest["raw_shape"]
    if (set(raw_shape) != {"schema", "width", "height", "dtype", "layout", "stage"} or
            raw_shape.get("schema") != whole_clip_raw_contract.RAW_SHAPE_SCHEMA or
            raw_shape.get("width") != expected_shape["width"] or
            raw_shape.get("height") != expected_shape["height"] or
            raw_shape.get("dtype") != "float32-le" or
            raw_shape.get("layout") != "row-major" or
            raw_shape.get("stage") != whole_clip_raw_contract.RAW_STAGE or
            reference["raw_shape"]["sha256"] !=
            input_contract.get("raw_shape_json_sha256")):
        raise ValueError("sequence copied raw shape has unknown or inconsistent semantics")

    supported_high = {
        (shape.width, shape.height)
        for shape in convex2x_contract.supported_high_shapes()
    }
    raw_extent = expected_shape["width"], expected_shape["height"]
    gpu_source_shape = gpu_manifest.get("source_shape")
    if (not isinstance(gpu_source_shape, dict) or
            set(gpu_source_shape) != {"width", "height"}):
        raise ValueError("sequence copied GPU evidence has an invalid source shape")
    _exact_source_capture_grid_kind(
        gpu_source_shape["width"], gpu_source_shape["height"],
        raw_extent[0], raw_extent[1])
    if raw_extent in supported_high:
        composite_runtime = results_meta.get("composite_runtime_provenance")
        fused_manifests = results_meta.get(convex2x_diagnostics.RESULTS_META_KEY)
        if (not isinstance(composite_runtime, dict) or
                not isinstance(fused_manifests, dict) or clip not in fused_manifests):
            raise ValueError(
                "sequence copied single-high raw field lacks fused runtime evidence")
        fused_manifest = fused_manifests[clip]
        frame_ids = [int(row["frame_id"]) for row in manifest["frames"]]
        convex2x_diagnostics.validate_manifest(
            fused_manifest, frame_ids, composite_runtime)
        if (fused_manifest.get("schema") != convex2x_diagnostics.MANIFEST_SCHEMA or
                fused_manifest["tensor_shapes"].get("output") != {
                    "width": raw_extent[0], "height": raw_extent[1], "channels": 1}):
            raise ValueError(
                "sequence copied fused evidence does not bind the single-high raw shape")
        expected_reference = {
            "schema": convex2x_diagnostics.SIDECAR_SCHEMA,
            "sidecar": convex2x_diagnostics.SIDECAR_FILENAME,
            "sidecar_sha256": fused_manifest["sidecar_sha256"],
            "frame_count": fused_manifest["frame_count"],
            "input_file_pattern": "model_input_<frame-id>.f32",
            "output_file_pattern": "raw_<frame-id>.f32",
            "authority": "single-high-input-output-boundary",
        }
        if contract.get("prod_zipdepth_convex2x_diagnostics") != expected_reference:
            raise ValueError(
                "sequence copied harness contract does not bind active single-high evidence")
        embedded = fused_manifest["embedded_dav2_provenance"]
        projected_identity = {
            key: manifest["producer_model_identity"][key]
            for key in (
                "model", "depth_model_url", "onnx_sha256", "preprocess_profile",
                "preprocess_source_closure_sha256")
        }
        if embedded != projected_identity:
            raise ValueError(
                "sequence copied fused evidence disagrees with embedded DAV2 provenance")
        fused_outputs = [
            {"frame_id": row["frame_id"],
             "file": row["output"]["file"], "sha256": row["output"]["sha256"]}
            for row in fused_manifest["frames"]
        ]
        expected_outputs = [
            {"frame_id": row["frame_id"], "file": row["file"], "sha256": row["sha256"]}
            for row in manifest["frames"]
        ]
        if fused_outputs != expected_outputs:
            raise ValueError(
                "sequence copied fused output hashes disagree with raw producer evidence")

    manifest_rows = manifest["frames"]
    input_rows = input_contract.get("ordered_raw_fields")
    gpu_rows = gpu_manifest.get("frames")
    if (not isinstance(input_rows, list) or not isinstance(gpu_rows, list) or
            len(manifest_rows) != len(input_rows) or len(manifest_rows) != len(gpu_rows)):
        raise ValueError("sequence copied raw manifest frame set is incomplete")
    expected_bytes = expected_shape["width"] * expected_shape["height"] * 4
    cut_control = input_contract.get("cut_control_evidence")
    if (not isinstance(cut_control, dict) or
            reference["cut_state"]["sha256"] != cut_control.get("sha256")):
        raise ValueError("sequence copied cut-state artifact is not the authenticated cut authority")
    cut_trace = cut_state_contract.load_trace(str(document_paths["cut_state"]))
    if set(cut_trace) != {int(row["frame_id"]) for row in manifest_rows}:
        raise ValueError("sequence copied cut-state frame set disagrees with raw evidence")
    for manifest_row, input_row, gpu_row in zip(manifest_rows, input_rows, gpu_rows):
        if (input_row != {"frame_id": manifest_row["frame_id"],
                         "sha256": manifest_row["sha256"]} or
                gpu_row.get("frame_id") != manifest_row["frame_id"] or
                gpu_row.get("raw_file") != manifest_row["file"] or
                gpu_row.get("raw_sha256") != manifest_row["sha256"]):
            raise ValueError("sequence raw frame evidence disagrees across producer/input/GPU")
        cut_row = cut_trace[int(manifest_row["frame_id"])]
        if (gpu_row.get("hard_cut_count") != int(cut_row["hard_cut_count"]) or
                gpu_row.get("hard_cut_pulse") != (cut_row["hard_cut_pulse"] > 0.5)):
            raise ValueError("sequence GPU cut evidence disagrees with production cut state")
        raw_path = output / "gpu_input" / manifest_row["file"]
        try:
            raw_size = raw_path.stat().st_size
        except OSError as exc:
            raise ValueError(f"sequence copied raw tensor is missing: {raw_path}") from exc
        if (raw_size != expected_bytes or
                direct_geometry.file_sha256(str(raw_path)) != manifest_row["sha256"]):
            raise ValueError(
                f"sequence copied raw tensor {manifest_row['frame_id']} is not authenticated")
    return manifest


def validate_sequence_replay_artifacts(output: Path) -> Dict[str, Any]:
    """Authenticate the state/color/geometry bundle after the harness completes."""

    document = _read_json(output / SEQUENCE_CONTRACT_FILE)
    expected_root = {
        "schema", "experiment", "input_contract", "mapping_config", "cut_source",
        "pop_strength_authority",
        "state_trace", "harness_contract", "direct_geometry_manifest", "frames",
        "unusable_depth_semantics", "mapping_implementation",
        "diagnostic_summary", "implementation_sources", "score_evidence",
        "gpu_input_manifest", "numpy_comparison",
        "producer_evidence",
    }
    if (set(document) != expected_root or document.get("schema") != SEQUENCE_CONTRACT_SCHEMA or
            document.get("experiment") != "depth-coordinate-v2-whole-clip-exact-replay" or
            document.get("mapping_implementation") !=
            "authenticated-raw-depth-plus-six-v2-compute-shaders-persistent-gpu-state-v9" or
            document.get("unusable_depth_semantics") !=
            "current-color-flat-retain-camera-unless-cut-v2"):
        raise ValueError("sequence contract has missing or unknown semantics")
    if document.get("implementation_sources") != _implementation_sources():
        raise ValueError("sequence implementation source hashes are stale or incomplete")
    input_contract = document.get("input_contract")
    expected_input_keys = {
        "contract_json_sha256", "contract_schema", "eval_schema",
        "depth_step", "depth_reuse_interval",
        "model", "depth_model_url", "preprocess_profile",
        "preprocess_source_closure_sha256", "onnx_sha256",
        "engine_name", "engine_sha256", "results_json_sha256",
        "calibration_id", "raw_coordinate_scale",
        "run_pop_strength",
        "model_hash_authority", "input_shape_authority", "raw_hash_authority", "raw_shape",
        "raw_shape_json_sha256", "selected_frame_ids", "ordered_raw_fields",
        "cut_control_evidence", "cut_expectation_evidence",
        "sequence_input_sha256",
    }
    if not isinstance(input_contract, dict) or set(input_contract) != expected_input_keys:
        raise ValueError("sequence contract has an invalid input evidence contract")
    sha_pattern = re.compile(r"[0-9a-f]{64}")
    if (input_contract.get("eval_schema") != whole_clip_raw_contract.EVALUATOR_SCHEMA or
            input_contract.get("contract_schema") !=
            whole_clip_raw_contract.HARNESS_CONTRACT_SCHEMA or
            input_contract.get("model_hash_authority") !=
            "schema-37-run-level-results-json" or
            input_contract.get("input_shape_authority") !=
            "schema-37-per-clip-raw-manifest" or
            not isinstance(input_contract.get("depth_model_url"), str) or
            not input_contract["depth_model_url"].startswith("https://") or
            not isinstance(input_contract.get("preprocess_profile"), str) or
            not input_contract["preprocess_profile"] or
            isinstance(input_contract.get("run_pop_strength"), bool) or
            not isinstance(input_contract.get("run_pop_strength"), (int, float)) or
            not 0.25 <= float(input_contract["run_pop_strength"]) <= 2.0 or
            any(not isinstance(input_contract.get(key), str) or
                sha_pattern.fullmatch(input_contract[key]) is None
                for key in ("contract_json_sha256", "onnx_sha256",
                            "preprocess_source_closure_sha256", "engine_sha256",
                            "results_json_sha256", "raw_shape_json_sha256"))):
        raise ValueError("sequence input evidence has unknown authority or invalid hashes")
    raw_hash_authority = input_contract.get("raw_hash_authority")
    if raw_hash_authority != {
            "source": "schema-37-run-level-results-json",
            "manifest_schema": whole_clip_raw_contract.MANIFEST_SCHEMA,
            "binding": whole_clip_raw_contract.BINDING,
    }:
        raise ValueError("sequence input evidence has unknown raw-hash authority")
    cut_control = input_contract.get("cut_control_evidence")
    if (not isinstance(cut_control, dict) or
            set(cut_control) != {"kind", "sha256"} or
            cut_control.get("kind") != "cut-state-hard-cut-generation" or
            not isinstance(cut_control.get("sha256"), str) or
            sha_pattern.fullmatch(cut_control["sha256"]) is None):
        raise ValueError("sequence cut controller evidence is not the live hard-cut generation")
    cut_expectation = input_contract.get("cut_expectation_evidence")
    if (not isinstance(cut_expectation, dict) or
            set(cut_expectation) != {"kind", "sha256"} or
            cut_expectation.get("kind") not in {"committed-meta-scoring-only", "none"} or
            (cut_expectation["kind"] == "committed-meta-scoring-only" and
             (not isinstance(cut_expectation.get("sha256"), str) or
              sha_pattern.fullmatch(cut_expectation["sha256"]) is None)) or
            (cut_expectation["kind"] == "none" and
             cut_expectation.get("sha256") is not None)):
        raise ValueError("sequence cut expectation evidence has unknown semantics")
    digest_payload = dict(input_contract)
    declared_input_digest = digest_payload.pop("sequence_input_sha256")
    measured_input_digest = hashlib.sha256(json.dumps(
        digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if declared_input_digest != measured_input_digest:
        raise ValueError("sequence input evidence digest mismatch")
    mapping = document.get("mapping_config")
    if (not isinstance(mapping, dict) or set(mapping) != SEQUENCE_MAPPING_CONFIG_KEYS or
            mapping.get("raw_coordinate_scale") !=
            input_contract.get("raw_coordinate_scale") or
            mapping.get("vertical_majorant_share") !=
            MappingV2Config().vertical_majorant_share):
        raise ValueError("sequence mapping does not use the authenticated model calibration")
    pop_authority = document.get("pop_strength_authority")
    if (not isinstance(pop_authority, dict) or
            set(pop_authority) != {"source", "selected", "run_configured"} or
            pop_authority.get("source") not in {
                "run-results-profile-config", "explicit-cli-override"} or
            pop_authority.get("run_configured") != input_contract.get("run_pop_strength") or
            pop_authority.get("selected") != mapping.get("pop_strength") or
            (pop_authority.get("source") == "run-results-profile-config" and
             pop_authority.get("selected") != pop_authority.get("run_configured"))):
        raise ValueError("sequence pop strength has unknown or inconsistent authority")
    trace_ref = document.get("state_trace")
    harness_ref = document.get("harness_contract")
    manifest_ref = document.get("direct_geometry_manifest")
    for reference, expected_file, expected_schema, label in (
            (trace_ref, STATE_TRACE_FILE, V2_STATE_TRACE_SCHEMA, "state trace"),
            (harness_ref, "harness/contract.json", direct_geometry.CONTRACT_SCHEMA,
             "harness contract"),
            (manifest_ref, "harness/direct_parallax_manifest.json",
             direct_geometry.MANIFEST_SCHEMA, "direct manifest")):
        if (not isinstance(reference, dict) or
                set(reference) != {"file", "schema", "sha256"} or
                reference.get("file") != expected_file or
                reference.get("schema") != expected_schema):
            raise ValueError(f"sequence contract has invalid {label} reference")
        path = _relative_file(output, reference["file"], label)
        if direct_geometry.file_sha256(str(path)) != reference.get("sha256"):
            raise ValueError(f"sequence contract {label} hash mismatch")

    gpu_input_ref = document.get("gpu_input_manifest")
    numpy_ref = document.get("numpy_comparison")
    for reference, expected_file, expected_schema, label in (
            (numpy_ref, NUMPY_COMPARISON_FILE, NUMPY_COMPARISON_SCHEMA,
             "NumPy comparison"),):
        if (not isinstance(reference, dict) or
                set(reference) != {"file", "schema", "sha256"} or
                reference.get("file") != expected_file or
                reference.get("schema") != expected_schema):
            raise ValueError(f"sequence contract has invalid {label} reference")
        reference_path = _relative_file(output, reference["file"], label)
        if direct_geometry.file_sha256(str(reference_path)) != reference.get("sha256"):
            raise ValueError(f"sequence contract {label} hash mismatch")

    gpu_input = _validate_gpu_input_manifest_evidence(
        output, gpu_input_ref, mapping, document.get("cut_source"), input_contract)
    _validate_producer_evidence_bundle(
        output, document.get("producer_evidence"), input_contract, gpu_input)
    numpy_comparison = _read_json(output / NUMPY_COMPARISON_FILE)
    if (numpy_comparison.get("schema") != NUMPY_COMPARISON_SCHEMA or
            numpy_comparison.get("role") != "numpy-comparison-oracle-only-v2" or
            numpy_comparison.get("render_authority") !=
            "native-six-shader-gpu-output" or
            numpy_comparison.get("all_within_float32_tolerances") is not True):
        raise ValueError("NumPy evidence is not a passing comparison-only oracle")

    rows = document.get("frames")
    expected_row_keys = {
        "frame_id", "input_source_file", "input_source_sha256", "rendered_source_frame_id",
        "rendered_source_file", "rendered_source_sha256", "raw_depth_sha256",
        "order_sha256", "parallax_sha256",
    }
    if not isinstance(rows, list) or not rows:
        raise ValueError("sequence contract requires a non-empty frames list")
    frame_ids: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_row_keys:
            raise ValueError(f"sequence contract frame {index} has invalid fields")
        frame_text = row.get("frame_id")
        if not isinstance(frame_text, str) or not frame_text.isdigit():
            raise ValueError(f"sequence contract frame {index} has invalid ID")
        frame_ids.append(int(frame_text))
        _validate_frame_source_attestation(output, row, frame_text)
        for kind, filename in (("order", f"depth_{frame_text}.f32"),
                               ("parallax", f"parallax_{frame_text}.f32")):
            path = output / "harness" / filename
            if direct_geometry.file_sha256(str(path)) != row[f"{kind}_sha256"]:
                raise ValueError(f"sequence contract frame {frame_text} {kind} hash mismatch")

    selected_ids = [f"{frame_id:05d}" for frame_id in frame_ids]
    if input_contract["selected_frame_ids"] != selected_ids:
        raise ValueError("sequence selected frame IDs disagree with rendered outputs")
    raw_records = input_contract["ordered_raw_fields"]
    if (not isinstance(raw_records, list) or len(raw_records) != len(rows) or
            any(not isinstance(record, dict) or set(record) != {"frame_id", "sha256"}
                for record in raw_records)):
        raise ValueError("sequence ordered raw-field contract is invalid")
    for row, raw_record in zip(rows, raw_records):
        if (raw_record["frame_id"] != row["frame_id"] or
                raw_record["sha256"] != row["raw_depth_sha256"]):
            raise ValueError(f"sequence frame {row['frame_id']} raw hash disagrees with input")

    raw_shape = input_contract.get("raw_shape")
    if (not isinstance(raw_shape, dict) or set(raw_shape) != {"height", "width"} or
            type(raw_shape.get("height")) is not int or raw_shape["height"] < 1 or
            type(raw_shape.get("width")) is not int or raw_shape["width"] < 1):
        raise ValueError("sequence raw shape is invalid")
    trace = _read_json(output / STATE_TRACE_FILE)
    validate_v2_state_trace(trace, frame_ids)
    if trace.get("producer", {}).get("tensor_shape") != {
            "width": raw_shape["width"], "height": raw_shape["height"]}:
        raise ValueError("sequence native state trace uses a different raw tensor shape")
    if trace["producer"]["manifest_sha256"] != gpu_input_ref.get("sha256"):
        raise ValueError("sequence native state trace names a different GPU input manifest")
    trace_rows = trace["frames"]
    gpu_frames = gpu_input["frames"]
    for index, (gpu_row, trace_row) in enumerate(zip(gpu_frames, trace_rows)):
        previous_count = (gpu_frames[index - 1]["hard_cut_count"] if index else
                          gpu_row["hard_cut_count"])
        expected_confirmed_cut = bool(
            index > 0 and (gpu_row["hard_cut_pulse"] or
                           gpu_row["hard_cut_count"] != previous_count))
        if (trace_row["confirmed_cut_count"] != gpu_row["hard_cut_count"] or
                trace_row["confirmed_cut"] != expected_confirmed_cut):
            raise ValueError(
                f"sequence frame {gpu_row['frame_id']} GPU cut input disagrees with state trace")
    if document.get("diagnostic_summary") != _diagnostic_summary(trace_rows):
        raise ValueError("sequence diagnostic summary disagrees with state trace")
    for row, gpu_row, trace_row in zip(rows, gpu_frames, trace_rows):
        if (gpu_row["source_sha256"] != row["input_source_sha256"] or
                row["rendered_source_frame_id"] != trace_row["rendered_source_frame_id"] or
                row["order_sha256"] != trace_row["order_sha256"] or
                row["parallax_sha256"] != trace_row["parallax_sha256"]):
            raise ValueError(f"sequence contract frame {row['frame_id']} disagrees with state")

        # Re-read the authenticated interchange bytes rather than trusting trace summaries. This
        # independently proves the pointwise soft container and both axis bounds for every field.
        frame_text = row["frame_id"]
        value_count = raw_shape["height"] * raw_shape["width"]
        order_path = output / "harness" / f"depth_{frame_text}.f32"
        parallax_path = output / "harness" / f"parallax_{frame_text}.f32"
        order = np.fromfile(order_path, dtype="<f4")
        encoded = np.fromfile(parallax_path, dtype="<f4")
        if (order.size != value_count or encoded.size != value_count or
                not np.isfinite(order).all() or not np.isfinite(encoded).all()):
            raise ValueError(f"sequence frame {frame_text} has invalid direct field bytes")
        order = order.reshape(raw_shape["height"], raw_shape["width"])
        decoded = decode_direct_parallax(
            encoded.reshape(raw_shape["height"], raw_shape["width"]))
        measured_maximum = float(np.max(np.abs(decoded)))
        measured_slope = (float(np.max(np.abs(np.diff(decoded, axis=1)))) * raw_shape["width"]
                          if raw_shape["width"] > 1 else 0.0)
        measured_vertical_shear = (
            float(np.max(np.abs(np.diff(decoded, axis=0)))) * raw_shape["width"]
            if raw_shape["height"] > 1 else 0.0)
        if measured_maximum > DIRECT_PARALLAX_SOURCE_U_LIMIT + 2.0e-7:
            raise ValueError(f"sequence frame {frame_text} exceeds the hard parallax container")
        if measured_slope > float(mapping["max_horizontal_slope"]) + 2.0e-5:
            raise ValueError(f"sequence frame {frame_text} exceeds the horizontal slope bound")
        if measured_vertical_shear > float(mapping["max_vertical_shear"]) + 2.0e-5:
            raise ValueError(f"sequence frame {frame_text} exceeds the vertical shear bound")
        if (not np.isclose(measured_maximum,
                           float(trace_row["final_max_abs_source_u"]),
                           rtol=2.0e-5, atol=2.0e-7) or
                not np.isclose(measured_slope,
                               float(trace_row["final_horizontal_slope_max"]),
                               rtol=2.0e-5, atol=2.0e-5) or
                not np.isclose(measured_vertical_shear,
                               float(trace_row["final_vertical_shear_max"]),
                               rtol=2.0e-5, atol=2.0e-5)):
            raise ValueError(f"sequence frame {frame_text} geometry summary is not remeasurable")
        if not trace_row["frame_valid"] and (
                np.any(order != 0.0) or measured_maximum > 2.0e-7):
            raise ValueError(f"sequence frame {frame_text} invalid depth was not published flat")
    harness = _read_json(output / "harness" / "contract.json")
    direct_geometry.validate_artifacts(str(output / "harness"), harness, set(frame_ids))
    gpu_execution = harness.get("depth_coordinate_v2_gpu")
    expected_cut_state = cut_state_contract.contract_reference(gpu_replay=True)
    if (not isinstance(gpu_execution, dict) or
            not np.isclose(float(harness.get("pop_strength", -1.0)),
                           float(mapping["pop_strength"]), rtol=0.0, atol=1.0e-7) or
            gpu_execution.get("enabled") is not True or
            gpu_execution.get("execution") !=
            "authenticated-raw-depth-plus-six-v2-compute-shaders-persistent-state-v9" or
            gpu_execution.get("tensorrt_executed") is not False or
            gpu_execution.get("render_authority") != "gpu-canonical-and-final-fields" or
            gpu_execution.get("numpy_role") != "comparison-only" or
            gpu_execution.get("calibration_contract_canonical_sha256") !=
            CONTRACT_CANONICAL_SHA256 or
            gpu_execution.get("pop_strength_authority") !=
            "contract.pop_strength-only" or
            gpu_execution.get("adaptive_pop_applied") is not False or
            harness.get("cut_state") != expected_cut_state or
            gpu_execution.get("input_manifest_sha256") !=
            gpu_input_ref.get("sha256") or
            gpu_execution.get("state_trace") != {
                "file": STATE_TRACE_FILE,
                "schema": V2_STATE_TRACE_SCHEMA,
                "sha256": trace_ref.get("sha256"),
            }):
        raise ValueError("harness did not attest native persistent-GPU v2 execution")
    score = document.get("score_evidence")
    if score == {"status": "not-run"}:
        pass
    elif isinstance(score, dict) and set(score) == {
            "status", "scope", "scorecard_file", "scorecard_sha256",
            "score_contract_file", "score_contract_sha256",
            "score_log_file", "score_log_sha256", "metric_contract"} and (
            score.get("status") == "complete" and
            score.get("scope") == "renderer-output-quality-only-v1"):
        for file_key, hash_key in (
                ("scorecard_file", "scorecard_sha256"),
                ("score_contract_file", "score_contract_sha256"),
                ("score_log_file", "score_log_sha256")):
            path = _relative_file(output, score[file_key], file_key)
            if direct_geometry.file_sha256(str(path)) != score[hash_key]:
                raise ValueError(f"sequence {file_key} hash mismatch")
        _validate_metric_contract_evidence(score["metric_contract"])
        scope_contract = _read_json(output / RENDERER_SCORE_CONTRACT_FILE)
        if scope_contract != {
                "schema": RENDERER_SCORE_CONTRACT_SCHEMA,
                "scope": "renderer-output-quality-only-v1",
                "renderer_scorecard": {
                    "file": RENDERER_SCORECARD_FILE,
                    "sha256": score["scorecard_sha256"]},
                "forbidden_metric_prefixes": list(FORBIDDEN_RENDERER_SCORE_PREFIXES),
                "v2_state_authority": {
                    "file": STATE_TRACE_FILE,
                    "schema": V2_STATE_TRACE_SCHEMA,
                },
                "metric_contract": score["metric_contract"],
                "reason": "controller state is authenticated only by the v2 state trace",
        }:
            raise ValueError("sequence renderer score contract has unknown semantics")
        renderer_score = _read_json(output / RENDERER_SCORECARD_FILE)
        if (set(renderer_score) != {"aggregate", "frames"} or
                not isinstance(renderer_score["aggregate"], dict) or
                not isinstance(renderer_score["frames"], list) or
                any(not isinstance(row, dict) for row in renderer_score["frames"])):
            raise ValueError("sequence renderer scorecard has an unknown schema")
        remaining = set(renderer_score["aggregate"])
        for row in renderer_score.get("frames", []):
            remaining.update(row)
        if any(key.startswith(prefix)
               for key in remaining for prefix in FORBIDDEN_RENDERER_SCORE_PREFIXES):
            raise ValueError("controller-state metrics remain in renderer-quality score")
    else:
        raise ValueError("sequence score evidence has unknown semantics")
    return document


def _resolve_pop_strength(
        run_model: Dict[str, object],
        explicit_override: Optional[float]) -> tuple[float, Dict[str, object]]:
    """Select and attest the sole v2 artistic gain authority."""

    run_configured = float(run_model["pop_strength"])
    selected = float(explicit_override) if explicit_override is not None else run_configured
    if not np.isfinite(selected) or not 0.25 <= selected <= 2.0:
        raise ValueError("v2 pop strength must be finite and between 0.25 and 2")
    return selected, {
        "source": ("explicit-cli-override" if explicit_override is not None else
                   "run-results-profile-config"),
        "selected": selected,
        "run_configured": run_configured,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-seq", required=True, type=Path,
                        help="one harness clip directory containing raw_<id>.f32")
    parser.add_argument("--frames", required=True, type=Path,
                        help="matching lossless source clip with frame_<id>.png")
    parser.add_argument("--out", required=True, type=Path, help="new or empty output directory")
    parser.add_argument("--build-dir", type=Path,
                        default=REPO / "cmake-build-relwithdebinfo")
    parser.add_argument("--conf", type=Path, default=SCRIPT_DIR / "bench.conf")
    parser.add_argument(
        "--pop-strength", type=float, default=None,
        help="explicit v2 pop override (default: resolved profile/global value in results.json)")
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args(argv)

    raw_sequence = args.raw_seq.resolve()
    source_root = args.frames.resolve()
    output = args.out.resolve()
    if not raw_sequence.is_dir() or not source_root.is_dir():
        parser.error("--raw-seq and --frames must be directories")
    try:
        _new_output_directory(output)
        frame_ids, raw_fields, shape = _load_raw_fields(raw_sequence)
        raw_paths = dict(_raw_files(raw_sequence))
        sources = _source_frames(source_root)
        if set(sources) != set(frame_ids):
            raise ValueError(
                f"source/raw frame IDs disagree: source={sorted(sources)} raw={frame_ids}")
        run_root = raw_sequence.parent
        run_model = _load_run_model_contract(run_root)
        input_contract = _clip_sequence_input_contract(
            raw_sequence, frame_ids, shape, run_model)
        producer_evidence = _materialize_producer_evidence_bundle(
            output, raw_sequence, run_model)
        cuts, cut_counts, cut_source = _load_cut_indices(raw_sequence, frame_ids)
        cut_pulses = _load_authenticated_cut_pulses(
            raw_sequence, frame_ids, cut_counts)
        selected_pop_strength, pop_strength_authority = _resolve_pop_strength(
            run_model, args.pop_strength)
        config = MappingV2Config(
            raw_coordinate_scale=float(run_model["raw_coordinate_scale"]),
            pop_strength=selected_pop_strength,
        )
        replay_frames = output / "frames"
        harness = output / "harness"
        output_rows, gpu_manifest_path = _materialize_gpu_replay_inputs(
            output, sources, raw_paths, frame_ids, cut_counts, cut_pulses, cut_source,
            shape, run_model, config)
        harness.mkdir()
        source_meta = source_root / "meta.json"
        if source_meta.is_file():
            shutil.copyfile(source_meta, replay_frames / "meta.json")
        else:
            (replay_frames / "meta.json").write_text(json.dumps({
                "name": f"{raw_sequence.name}-mapping-v2-sequence",
                "description": "Whole-clip exact depth-coordinate-v2 replay.",
                "content_type": "unclassified",
            }, indent=2) + "\n", encoding="utf-8")

        executable = args.build_dir.resolve() / "sunshine.exe"
        conf = args.conf.resolve()
        if not executable.is_file() or not conf.is_file():
            raise ValueError("build-dir must contain sunshine.exe and --conf must be readable")
        harness_command = [
            str(executable), str(conf), "--sbs-bench",
            "--frames", str(replay_frames), "--out", str(harness),
            "--pop-strength", str(config.pop_strength),
            "--depth-coordinate-v2-manifest", str(gpu_manifest_path),
        ]
        _run_checked(harness_command, args.build_dir.resolve(), output / "harness.log")
        native_trace_path = harness / STATE_TRACE_FILE
        trace_path = output / STATE_TRACE_FILE
        if not native_trace_path.is_file():
            raise RuntimeError("native harness omitted the v2 GPU state trace")
        shutil.copyfile(native_trace_path, trace_path)
        gpu_trace = _read_json(trace_path)
        validate_v2_state_trace(gpu_trace, frame_ids)
        trace_rows = gpu_trace["frames"]
        for row, trace_row in zip(output_rows, trace_rows):
            row["order_sha256"] = trace_row["order_sha256"]
            row["parallax_sha256"] = trace_row["parallax_sha256"]
        harness_contract = _validate_harness_geometry(harness, frame_ids, shape)
        if harness_contract.get("model") != run_model["model"]:
            raise RuntimeError("replay harness model differs from raw-sequence model contract")

        numpy_comparison = _compare_gpu_with_numpy(
            output, raw_fields, frame_ids, cuts, cut_counts, cut_source, config, gpu_trace)

        direct_reference = harness_contract.get("direct_parallax_manifest")
        if not isinstance(direct_reference, dict):
            raise RuntimeError("replay harness omitted direct manifest reference")
        score_evidence: Dict[str, object] = {"status": "not-run"}
        if not args.skip_score:
            score_path = output / RENDERER_SCORECARD_FILE
            score_log = output / "score.log"
            score_command = [
                sys.executable, str(SCRIPT_DIR / "sbsbench.py"),
                "--seq", str(harness), "--frames", str(replay_frames),
                "--json", str(score_path),
            ]
            _run_checked(score_command, REPO, score_log)
            score_evidence = {
                **_publish_renderer_quality_score(score_path, output),
                "score_log_file": "score.log",
                "score_log_sha256": direct_geometry.file_sha256(str(score_log)),
            }
        sequence_contract = {
            "schema": SEQUENCE_CONTRACT_SCHEMA,
            "experiment": "depth-coordinate-v2-whole-clip-exact-replay",
            "mapping_implementation":
                "authenticated-raw-depth-plus-six-v2-compute-shaders-persistent-gpu-state-v9",
            "input_contract": input_contract,
            "mapping_config": asdict(config),
            "pop_strength_authority": pop_strength_authority,
            "cut_source": cut_source,
            "state_trace": {
                "file": STATE_TRACE_FILE,
                "schema": V2_STATE_TRACE_SCHEMA,
                "sha256": direct_geometry.file_sha256(str(trace_path)),
            },
            "harness_contract": {
                "file": "harness/contract.json",
                "schema": direct_geometry.CONTRACT_SCHEMA,
                "sha256": direct_geometry.file_sha256(str(harness / "contract.json")),
            },
            "direct_geometry_manifest": {
                "file": "harness/direct_parallax_manifest.json",
                "schema": direct_geometry.MANIFEST_SCHEMA,
                "sha256": direct_geometry.file_sha256(
                    str(harness / "direct_parallax_manifest.json")),
            },
            "gpu_input_manifest": {
                "file": GPU_INPUT_MANIFEST_FILE,
                "schema": GPU_INPUT_MANIFEST_SCHEMA,
                "sha256": direct_geometry.file_sha256(str(gpu_manifest_path)),
            },
            "producer_evidence": producer_evidence,
            "numpy_comparison": {
                "file": NUMPY_COMPARISON_FILE,
                "schema": int(numpy_comparison["schema"]),
                "sha256": direct_geometry.file_sha256(
                    str(output / NUMPY_COMPARISON_FILE)),
            },
            "frames": output_rows,
            "unusable_depth_semantics":
                "current-color-flat-retain-camera-unless-cut-v2",
            "diagnostic_summary": _diagnostic_summary(trace_rows),
            "implementation_sources": _implementation_sources(),
            "score_evidence": score_evidence,
        }
        (output / SEQUENCE_CONTRACT_FILE).write_text(
            json.dumps(sequence_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_sequence_replay_artifacts(output)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))

    print(f"wrote exact whole-clip depth-coordinate-v2 replay to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
