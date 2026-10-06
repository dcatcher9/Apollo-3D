"""Strict native adaptive-camera trace and authenticated raw-sequence readers.

There is one production geometry policy. The native GPU owns zero, robust divisor, source-time
adaptation and rendered fields; this module authenticates evidence without a CPU controller.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
try:
    from .depth_mapping_v2 import DIRECT_PARALLAX_SOURCE_U_LIMIT, MappingV2Config, linear_relative_coordinate
    from .depth_coordinate_v2_contract import (CALIBRATED_DEFAULTS as V2_DEFAULTS,
        CONTRACT_CANONICAL_SHA256, CONTRACT_PATH, CONTRACT_SCHEMA, MODEL_CALIBRATIONS)
    from . import cut_state_contract, whole_clip_raw_contract
    from . import depth_coordinate_v2_dump_contract as dump_contract
    from . import prod_zipdepth_convex2x as convex2x_contract
    from . import prod_zipdepth_convex2x_diagnostics_contract as convex2x_diagnostics
except ImportError:
    from depth_mapping_v2 import DIRECT_PARALLAX_SOURCE_U_LIMIT, MappingV2Config, linear_relative_coordinate
    from depth_coordinate_v2_contract import (CALIBRATED_DEFAULTS as V2_DEFAULTS,
        CONTRACT_CANONICAL_SHA256, CONTRACT_PATH, CONTRACT_SCHEMA, MODEL_CALIBRATIONS)
    import cut_state_contract, whole_clip_raw_contract
    import depth_coordinate_v2_dump_contract as dump_contract
    import prod_zipdepth_convex2x as convex2x_contract
    import prod_zipdepth_convex2x_diagnostics_contract as convex2x_diagnostics

RAW_PATTERN = re.compile(r"raw_(\d+)\.f32$")
V2_STATE_TRACE_SCHEMA = 22
V2_STATE_TRACE_POLICY = (
    "authenticated-adaptive-camera-host-robust-linear-hard-cap-vertical-share75-row-majorant-v22"
)
V2_GPU_SHADER_SEQUENCE = (
    "depth_coordinate_v2_moments_cs.hlsl",
    "depth_coordinate_v2_frame_resolve_cs.hlsl",
    "depth_coordinate_v2_histogram_cs.hlsl",
    "depth_coordinate_v2_quantiles_cs.hlsl",
    "depth_coordinate_v2_state_resolve_cs.hlsl",
    "depth_coordinate_v2_map_cs.hlsl",
    "depth_coordinate_v2_vertical_limit_cs.hlsl",
    "depth_coordinate_v2_limit_cs.hlsl",
)
V2_STATE_TRACE_FIELDS = (
    "frame_id",
    "observation_timestamp_us",
    "state_words_u32",
    "joint_plane_mode",
    "input_source_frame_id",
    "rendered_source_frame_id",
    "calibration_revision",
    "confirmed_cut_count",
    "confirmed_cut",
    "cut_attribution",
    "input_valid",
    "frame_valid",
    "camera_valid",
    "collapsed",
    "center",
    "inverse_scale",
    "latched_scale",
    "convergence_curve",
    "requested_gain",
    "container_scale",
    "effective_gain",
    "observed_mean",
    "observed_std",
    "observed_raw_minimum",
    "observed_raw_maximum",
    "observed_percentile_low",
    "observed_percentile_high",
    "observed_percentile_valid",
    "observed_percentile_bin_width",
    "candidate_center_drift_u",
    "predicted_zero_translation_source_u",
    "pre_limiter_max_abs_source_u",
    "vertical_majorant_raised_fraction",
    "vertical_majorant_max_raise_source_u",
    "final_max_abs_source_u",
    "conditioner_raised_fraction",
    "conditioner_max_raise_source_u",
    "conditioner_lowered_fraction",
    "conditioner_max_lower_source_u",
    "final_horizontal_slope_max",
    "final_vertical_shear_max",
    "order_sha256",
    "vertical_majorant_sha256",
    "vertical_conditioned_sha256",
    "parallax_sha256",
)



def _exact_counter_increment(value: int) -> int:
    """Match the GPU counter's reserved-sentinel saturation rule."""

    return min(int(value), 0xFFFFFFFD) + 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_adaptation_semantics(mode: int) -> Dict[str, str]:
    if type(mode) is not int or mode != 3:
        raise ValueError("v2 trace requires adaptive policy identity 3")
    return {
        "coordinate": "bounded-source-time-linear-raw-mean-and-robust-amplitude-reference",
        "convergence_curve": "continuously-tracked-mean-is-zero-plane-no-cut-latch",
        "near_curve": "linear-coordinate-no-depth-curve-before-hard-representation-cap",
    }


def validate_v2_state_trace(
        trace: Dict[str, object],
        expected_frame_ids: Sequence[int]) -> None:
    """Fail closed unless every row proves the simplified persistent-camera contract."""

    root_keys = {
        "schema", "policy", "calibration_contract", "cut_source", "diagnostic_role",
        "diagnostic_method", "mapping_config", "adaptation_semantics",
        "frame_fields", "frames", "producer",
    }
    if not isinstance(trace, dict) or set(trace) != root_keys:
        raise ValueError("v2 state trace has missing or unknown root fields")
    if (trace.get("schema") != V2_STATE_TRACE_SCHEMA or
            trace.get("policy") != V2_STATE_TRACE_POLICY or
            trace.get("diagnostic_role") != "non-controlling-mode-selected-camera-audit-v6" or
            trace.get("diagnostic_method") not in {
                "frame-moment-proxies-not-matched-pixel-affine-v2",
                "gpu-frame-moments-and-rendered-fields-v6"} or
            trace.get("frame_fields") != list(V2_STATE_TRACE_FIELDS) or
            not isinstance(trace.get("cut_source"), str) or not trace["cut_source"]):
        raise ValueError("v2 state trace has missing or unknown semantics")
    contract = trace.get("calibration_contract")
    expected_contract = {
        # JSON provenance paths use a platform-independent spelling.  Native producers always
        # emit forward slashes, including when the replay runs on Windows.
        "file": CONTRACT_PATH.relative_to(Path(__file__).resolve().parent).as_posix(),
        "schema": CONTRACT_SCHEMA,
        "sha256": _file_sha256(CONTRACT_PATH),
    }
    if contract != expected_contract:
        raise ValueError("v2 state trace calibration contract is stale or unauthenticated")
    mapping_payload = trace.get("mapping_config")
    expected_mapping_keys = set(asdict(MappingV2Config()))
    if (not isinstance(mapping_payload, dict) or
            set(mapping_payload) != expected_mapping_keys | {"joint_plane_mode"}):
        raise ValueError("v2 state trace mapping config is missing or unknown")
    mode = mapping_payload.get("joint_plane_mode")
    if type(mode) is not int or mode != 3:
        raise ValueError("v2 state trace joint_plane_mode must be adaptive policy identity 3")
    if trace.get("adaptation_semantics") != _trace_adaptation_semantics(mode):
        raise ValueError("v2 state trace mode/adaptation semantics disagree")
    try:
        mapping = MappingV2Config(**{
            key: value for key, value in mapping_payload.items()
            if key != "joint_plane_mode"})
        # Trigger the complete mapping validation without coupling this module to private helpers.
        linear_relative_coordinate(np.asarray([[0.0]]), 0.0, mapping.raw_coordinate_scale, mapping)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"v2 state trace mapping config is invalid: {exc}") from exc
    producer = trace.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("v2 state trace has an invalid producer authority")
    producer_authority = producer.get("authority")
    expected_method = None
    if producer_authority == (
            "authenticated-raw-depth-plus-eight-v2-compute-shaders-persistent-gpu-state-v11"):
        digest_pattern = re.compile(r"[0-9a-f]{64}")
        if (set(producer) != {
                "authority", "manifest_sha256", "contract_canonical_sha256",
                "tensor_shape", "shader_sequence", "state_persistence", "numpy_role"} or
                not isinstance(producer.get("manifest_sha256"), str) or
                digest_pattern.fullmatch(producer["manifest_sha256"]) is None or
                producer.get("contract_canonical_sha256") != CONTRACT_CANONICAL_SHA256 or
                producer.get("shader_sequence") != list(
                    V2_GPU_SHADER_SEQUENCE) or
                producer.get("state_persistence") != "single-buffer-whole-sequence" or
                producer.get("numpy_role") != "comparison-only-not-render-authority"):
            raise ValueError("v2 state trace has invalid native GPU producer evidence")
        expected_method = "gpu-frame-moments-and-rendered-fields-v6"
        calibrated_shapes = {
            shape
            for calibration in MODEL_CALIBRATIONS
            if math.isclose(
                calibration.raw_coordinate_scale, mapping.raw_coordinate_scale,
                rel_tol=0.0, abs_tol=1.0e-12)
            for shape in calibration.calibrated_input_shapes
        }
        calibrated_shapes.update({
            (shape.width, shape.height)
            for shape in convex2x_contract.supported_high_shapes()
            if (shape.width // 2, shape.height // 2) in calibrated_shapes
        })
        tensor_shape = producer.get("tensor_shape")
        if (not isinstance(tensor_shape, dict) or
                set(tensor_shape) != {"width", "height"} or
                isinstance(tensor_shape.get("width"), bool) or
                isinstance(tensor_shape.get("height"), bool) or
                not isinstance(tensor_shape.get("width"), int) or
                not isinstance(tensor_shape.get("height"), int)):
            raise ValueError(
                "v2 native trace has an invalid replay tensor shape")
        calibrated_width = tensor_shape["width"]
        calibrated_height = tensor_shape["height"]
        if (calibrated_width, calibrated_height) not in calibrated_shapes:
            raise ValueError(
                "v2 native trace identifies an unauthenticated replay tensor shape")
    else:
        raise ValueError("v2 state trace has an invalid producer authority")
    if trace.get("diagnostic_method") != expected_method:
        raise ValueError("v2 state trace diagnostic method disagrees with its producer")

    rows = trace.get("frames")
    expected_ids = tuple(int(value) for value in expected_frame_ids)
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError("v2 state trace frame count does not match the replay sequence")
    prior_revision = 0
    prior_cut_count: Optional[int] = None
    prior_camera_initialized = False
    prior_center = 0.0
    prior_inverse_scale = 0.0
    prior_convergence_curve = V2_DEFAULTS.convergence_curve_default
    prior_native_state = None
    requested_gain: Optional[float] = None
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    for index, (row, expected_id) in enumerate(zip(rows, expected_ids)):
        if not isinstance(row, dict) or set(row) != set(V2_STATE_TRACE_FIELDS):
            raise ValueError(f"v2 state trace frame {index} has an invalid field layout")
        frame_id = row["frame_id"]
        if type(row["joint_plane_mode"]) is not int or row["joint_plane_mode"] != mode:
            raise ValueError(f"v2 state trace {frame_id} mode disagrees with mapping")
        if frame_id != f"{expected_id:05d}":
            raise ValueError(f"v2 state trace frame-id mismatch at index {index}")
        if (row["input_source_frame_id"] != frame_id or
                row["rendered_source_frame_id"] != frame_id):
            raise ValueError(f"v2 state trace {frame_id} has invalid source-frame identity")
        timestamp = row["observation_timestamp_us"]
        words = row["state_words_u32"]
        if (type(timestamp) is not int or not 0 <= timestamp <= 0xFFFFFFFFFFFFFFFF or
                not isinstance(words, list) or
                len(words) != 28 or
                any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in words)):
            raise ValueError(f"v2 state trace {frame_id} has an invalid source clock/state layout")
        native_state = dump_contract.validate_parallax_state_words(
            words, raw_coordinate_scale=mapping.raw_coordinate_scale,
            expected_joint_plane_mode=mode, requested_gain=mapping.parallax_gain)
        for key in ("center", "inverse_scale", "convergence_curve", "container_scale"):
            if np.float32(row[key]) != np.float32(native_state[key]):
                raise ValueError(f"v2 state trace {frame_id} {key} disagrees with authenticated state")
        for key in ("calibration_revision", "confirmed_cut_count"):
            if row[key] != native_state[key]:
                raise ValueError(f"v2 state trace {frame_id} {key} disagrees with authenticated state")
        if row["frame_valid"] != (native_state["frame_valid"] == 1.0):
            raise ValueError(f"v2 state trace {frame_id} validity disagrees with authenticated state")
        for key in ("calibration_revision", "confirmed_cut_count"):
            if type(row[key]) is not int or row[key] < 0 or row[key] > 0xFFFFFFFE:
                raise ValueError(f"v2 state trace {frame_id} has an invalid {key}")
        for key in ("confirmed_cut", "input_valid", "frame_valid", "camera_valid",
                    "collapsed"):
            if type(row[key]) is not bool:
                raise ValueError(f"v2 state trace {frame_id} has an invalid {key}")
        if not isinstance(row["cut_attribution"], str):
            raise ValueError(f"v2 state trace {frame_id} has invalid attribution")
        for key in (
                "center", "inverse_scale", "latched_scale", "convergence_curve",
                "requested_gain", "container_scale", "effective_gain",
                "observed_mean", "observed_std", "observed_raw_minimum",
                "observed_raw_maximum",
                "observed_percentile_low", "observed_percentile_high",
                "observed_percentile_valid", "observed_percentile_bin_width",
                 "candidate_center_drift_u", "predicted_zero_translation_source_u",
                 "pre_limiter_max_abs_source_u",
                "vertical_majorant_raised_fraction",
                "vertical_majorant_max_raise_source_u", "final_max_abs_source_u",
                "conditioner_raised_fraction", "conditioner_max_raise_source_u",
                "conditioner_lowered_fraction", "conditioner_max_lower_source_u",
                "final_horizontal_slope_max", "final_vertical_shear_max"):
            value = row[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value))):
                raise ValueError(f"v2 state trace {frame_id} has non-finite {key}")
        expected_collapsed = bool(
            row["input_valid"] and
            row["observed_std"] <= mapping.collapse_abs_epsilon)
        percentile_valid = row["observed_percentile_valid"]
        percentile_fields = ("observed_percentile_low", "observed_percentile_high",
                             "observed_percentile_bin_width")
        if percentile_valid not in (0.0, 1.0):
            raise ValueError(f"v2 state trace {frame_id} has invalid percentile validity")
        if percentile_valid == 1.0:
            if (not row["input_valid"] or
                    row["observed_percentile_bin_width"] < 0.0 or not
                    row["observed_raw_minimum"] <= row["observed_percentile_low"] <=
                    row["observed_percentile_high"] <= row["observed_raw_maximum"]):
                raise ValueError(f"v2 state trace {frame_id} has invalid percentile bounds")
        elif any(row[key] != 0.0 for key in percentile_fields):
            raise ValueError(f"v2 state trace {frame_id} has nonzero unavailable percentile tail")
        expected_frame_valid = bool(row["input_valid"] and not expected_collapsed)
        maximum = np.float32(row["observed_raw_maximum"])
        minimum = np.float32(row["observed_raw_minimum"])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            low = np.float32(row["observed_percentile_low"])
            high = np.float32(row["observed_percentile_high"])
            amplitude = np.maximum(
                np.float32(np.float32(0.5) * np.float32(high - low)),
                np.float32(mapping.raw_coordinate_scale))
            nearest_inverse = np.float32(np.float32(1.0) / amplitude)
            target_zero = np.float32(row["observed_mean"])
            fit_usable = bool(expected_frame_valid and
                              row["observed_percentile_valid"] == 1.0 and
                              maximum > minimum and
                              minimum <= target_zero <= maximum and
                              minimum <= low <= high <= maximum and
                              np.isfinite(amplitude) and
                              np.isfinite(nearest_inverse) and nearest_inverse > 0.0)
        previous_clock = (prior_native_state["gain_last_observation_low"] |
                          (prior_native_state["gain_last_observation_high"] << 32)
                          if prior_native_state is not None else 0)
        distinct_clock = timestamp > 0 and timestamp > previous_clock
        authorization_eligible = bool(fit_usable and distinct_clock and
                                      native_state["gain_seed_count"] == 1)
        # Seeding/adaptation arithmetic may deliberately fail flat after a usable target.
        # Audit the native authorization one-way, without inventing a CPU controller.
        if row["frame_valid"] and not authorization_eligible:
            raise ValueError(f"v2 state trace {frame_id} gain authorization lacks usable source evidence")
        expected_frame_valid = row["frame_valid"]
        last_clock = (native_state["gain_last_observation_low"] |
                      (native_state["gain_last_observation_high"] << 32))
        expected_clock = timestamp if distinct_clock else previous_clock
        if last_clock != expected_clock:
            raise ValueError(f"v2 state trace {frame_id} gain clock disagrees with source time")
        if native_state["gain_clock_armed"] and not (fit_usable and distinct_clock):
            raise ValueError(f"v2 state trace {frame_id} gain continuity disagrees with source evidence")
        if fit_usable and distinct_clock:
            for key, expected in (("gain_target_zero", target_zero),
                                  ("gain_target_inverse_scale", nearest_inverse),
                                  ("gain_target_nearest", amplitude)):
                if not math.isclose(native_state[key], expected, rel_tol=3.0e-5, abs_tol=3.0e-7):
                    raise ValueError(f"v2 state trace {frame_id} {key} disagrees with captured moments")
        if (row["collapsed"] != expected_collapsed or
                row["frame_valid"] != expected_frame_valid):
            raise ValueError(
                f"v2 state trace {frame_id} has inconsistent input/collapse/frame validity")
        for key in (
                "order_sha256", "vertical_majorant_sha256",
                "vertical_conditioned_sha256", "parallax_sha256"):
            if (not isinstance(row[key], str) or digest_pattern.fullmatch(row[key]) is None):
                raise ValueError(f"v2 state trace {frame_id} has invalid {key}")

        revision = row["calibration_revision"]
        cut_count = row["confirmed_cut_count"]
        if prior_cut_count is not None:
            cut_delta = cut_count - prior_cut_count
            if cut_delta not in (0, 1):
                raise ValueError(f"v2 state trace {frame_id} cut generation is inconsistent")
            if not row["confirmed_cut"] and cut_delta != 0:
                raise ValueError(
                    f"v2 state trace {frame_id} changed cut generation without a cut")
        camera_initialized = bool(
            row["inverse_scale"] > 0.0 and row["calibration_revision"] > 0)
        if row["camera_valid"] != camera_initialized:
            raise ValueError(f"v2 state trace {frame_id} camera validity is inconsistent")
        published = row["frame_valid"]
        expected_revision = (_exact_counter_increment(prior_revision)
                             if published else prior_revision)
        if revision != expected_revision:
            raise ValueError(f"v2 state trace {frame_id} calibration revision is inconsistent")
        expected_attribution = (
            "initialization" if index == 0 else
            trace["cut_source"] if row["confirmed_cut"] else "none"
        )
        if row["cut_attribution"] != expected_attribution:
            raise ValueError(f"v2 state trace {frame_id} cut attribution is inconsistent")
        if row["container_scale"] != 1.0:
            raise ValueError(f"v2 state trace {frame_id} has non-identity container scale")
        if requested_gain is None:
            requested_gain = row["requested_gain"]
        elif not math.isclose(row["requested_gain"], requested_gain,
                              rel_tol=2.0e-6, abs_tol=2.0e-8):
            raise ValueError(f"v2 state trace {frame_id} changed requested gain")
        if row["requested_gain"] <= 0.0:
            raise ValueError(f"v2 state trace {frame_id} has non-positive requested gain")
        if not math.isclose(
                row["requested_gain"], mapping.parallax_gain,
                rel_tol=2.0e-6, abs_tol=2.0e-8):
            raise ValueError(f"v2 state trace {frame_id} requested gain disagrees with mapping")
        for key in ("pre_limiter_max_abs_source_u",
                    "vertical_majorant_raised_fraction",
                    "vertical_majorant_max_raise_source_u", "final_max_abs_source_u",
                    "conditioner_raised_fraction", "conditioner_max_raise_source_u",
                    "conditioner_lowered_fraction", "conditioner_max_lower_source_u",
                    "final_horizontal_slope_max", "final_vertical_shear_max"):
            if row[key] < 0.0:
                raise ValueError(f"v2 state trace {frame_id} has negative {key}")
        for key in (
                "vertical_majorant_raised_fraction",
                "conditioner_raised_fraction",
                "conditioner_lowered_fraction"):
            if row[key] > 1.0:
                raise ValueError(f"v2 state trace {frame_id} {key} exceeds one")
        if row["frame_valid"]:
            if not camera_initialized:
                raise ValueError(f"v2 state trace {frame_id} has no camera for a usable frame")
            if row["inverse_scale"] <= 0.0 or row["latched_scale"] <= 0.0:
                raise ValueError(f"v2 state trace {frame_id} has invalid latched scale")
            if not math.isclose(
                    row["latched_scale"], 1.0 / row["inverse_scale"],
                    rel_tol=3.0e-5, abs_tol=3.0e-7):
                raise ValueError(f"v2 state trace {frame_id} scale/inverse disagree")
            if not math.isclose(
                    row["convergence_curve"], V2_DEFAULTS.convergence_curve_default,
                    rel_tol=0.0, abs_tol=2.0e-7):
                raise ValueError(
                    f"v2 state trace {frame_id} has an unknown convergence selection")
            expected_gain = row["requested_gain"]
            if not math.isclose(row["effective_gain"], expected_gain, rel_tol=3.0e-5,
                                abs_tol=3.0e-7):
                raise ValueError(f"v2 state trace {frame_id} effective gain is inconsistent")
        else:
            if row["effective_gain"] != 0.0 or row["container_scale"] != 1.0:
                raise ValueError(
                    f"v2 state trace {frame_id} unavailable frame retained active gain")
            expected_camera = prior_camera_initialized
            if camera_initialized != expected_camera:
                raise ValueError(
                    f"v2 state trace {frame_id} changed camera on unavailable depth")
            if expected_camera:
                if (not math.isclose(row["center"], prior_center,
                                     rel_tol=2.0e-6, abs_tol=2.0e-7) or
                        not math.isclose(row["inverse_scale"], prior_inverse_scale,
                                         rel_tol=2.0e-6, abs_tol=2.0e-7) or
                        not math.isclose(row["latched_scale"],
                                         1.0 / prior_inverse_scale,
                                         rel_tol=3.0e-5, abs_tol=3.0e-7) or
                        not math.isclose(row["convergence_curve"],
                                         prior_convergence_curve,
                                         rel_tol=0.0, abs_tol=2.0e-7)):
                    raise ValueError(
                        f"v2 state trace {frame_id} moved retained camera on unavailable depth")
            elif (row["center"] != 0.0 or row["inverse_scale"] != 0.0 or
                  row["latched_scale"] != 0.0 or
                  row["convergence_curve"] != V2_DEFAULTS.convergence_curve_default):
                raise ValueError(
                    f"v2 state trace {frame_id} failed to publish canonical empty camera")
        limit = DIRECT_PARALLAX_SOURCE_U_LIMIT
        if row["final_max_abs_source_u"] > limit + 1.0e-7:
            raise ValueError(f"v2 state trace {frame_id} exceeded the hard parallax container")
        if row["final_horizontal_slope_max"] > mapping.max_horizontal_slope + 1.0e-6:
            raise ValueError(f"v2 state trace {frame_id} exceeded the horizontal slope bound")
        if row["final_vertical_shear_max"] > mapping.max_vertical_shear + 1.0e-6:
            raise ValueError(f"v2 state trace {frame_id} exceeded the vertical shear bound")

        prior_revision = revision
        prior_cut_count = cut_count
        prior_camera_initialized = camera_initialized
        prior_center = row["center"]
        prior_inverse_scale = row["inverse_scale"]
        prior_convergence_curve = row["convergence_curve"]
        prior_native_state = native_state


def _load_shape(clip_dir: Path) -> Tuple[int, int]:
    path = clip_dir / "raw_shape.json"
    try:
        shape = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{clip_dir}: invalid raw_shape.json: {exc}") from exc
    expected_keys = {"schema", "width", "height", "dtype", "layout", "stage"}
    if (not isinstance(shape, dict) or set(shape) != expected_keys or
            shape.get("schema") != whole_clip_raw_contract.RAW_SHAPE_SCHEMA or
            shape.get("dtype") != "float32-le" or shape.get("layout") != "row-major" or
            shape.get("stage") != whole_clip_raw_contract.RAW_STAGE or
            type(shape.get("width")) is not int or shape["width"] <= 0 or
            type(shape.get("height")) is not int or shape["height"] <= 0):
        raise ValueError(f"{clip_dir}: raw_shape.json has missing/unknown shape semantics")
    return shape["height"], shape["width"]


def _raw_files(clip_dir: Path) -> List[Tuple[int, Path]]:
    by_id: Dict[int, Path] = {}
    for path in clip_dir.glob("raw_*.f32"):
        match = RAW_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"{clip_dir}: malformed raw-field filename {path.name!r}")
        frame_id = int(match.group(1))
        if frame_id in by_id:
            raise ValueError(
                f"{clip_dir}: duplicate numeric raw frame ID {frame_id}: "
                f"{by_id[frame_id].name}, {path.name}")
        by_id[frame_id] = path
    return sorted(by_id.items())


def _load_cut_indices(
        clip_dir: Path,
        frame_ids: Sequence[int]) -> Tuple[List[bool], List[int], str]:
    cuts = [False] * len(frame_ids)
    counts = [0] * len(frame_ids)
    if cuts:
        cuts[0] = True
    # Controller evidence must be the detector state that production actually emitted. Committed
    # expected_pulse_frames are ground truth for scoring only: allowing labels to drive this list
    # would make a missed or late live cut look perfect in the v2 replay.
    state_path = clip_dir / "cut_state.json"
    if not state_path.exists():
        return cuts, counts, "first-frame-only"
    trace = cut_state_contract.load_trace(str(state_path))
    if set(trace) != set(frame_ids):
        raise ValueError(
            f"{state_path}: trace/raw frame IDs disagree: "
            f"trace={sorted(trace)} raw={sorted(frame_ids)}")
    previous_count: Optional[int] = None
    for index, frame_id in enumerate(frame_ids):
        count = int(trace[frame_id]["hard_cut_count"])
        counts[index] = count
        pulse = trace[frame_id]["hard_cut_pulse"] > 0.5
        if previous_count is not None:
            if count < previous_count or count - previous_count > 1:
                raise ValueError(
                    f"{state_path}: hard-cut generation must advance by zero or one at frame "
                    f"{frame_id}, got {previous_count} -> {count}")
            cuts[index] = pulse or count != previous_count
        previous_count = count
    return cuts, counts, "cut-state-hard-cut-generation"


def _load_run_model_contract(run_root: Path) -> Dict[str, object]:
    """Authenticate the model identity inherited by every raw field in an evaluator run."""

    results_path = run_root / "results.json"
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{run_root}: invalid results.json model contract: {exc}") from exc
    meta = payload.get("meta") if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        raise ValueError(f"{run_root}: results.json lacks meta model contract")
    if meta.get("eval_schema") != whole_clip_raw_contract.EVALUATOR_SCHEMA:
        raise ValueError(
            f"{run_root}: stale evaluator schema {meta.get('eval_schema')!r}; expected "
            f"{whole_clip_raw_contract.EVALUATOR_SCHEMA}")
    sha_pattern = re.compile(r"[0-9a-f]{64}")
    model = meta.get("model")
    model_url = meta.get("depth_model_url")
    preprocess_profile = meta.get("preprocess_profile")
    preprocess_source_closure_sha256 = meta.get("preprocess_source_closure_sha256")
    declared_calibration_id = meta.get("depth_coordinate_v2_calibration_id")
    declared_raw_shape = meta.get("depth_coordinate_v2_raw_shape")
    onnx_sha = meta.get("onnx_sha256")
    engine_name = meta.get("engine_name")
    engine_sha = meta.get("engine_sha256")
    pop_strength = meta.get("pop_strength")
    if (not isinstance(model, str) or not model or
            not isinstance(model_url, str) or
            (preprocess_profile is not None and
             (not isinstance(preprocess_profile, str) or not preprocess_profile)) or
            not isinstance(preprocess_source_closure_sha256, str) or
            sha_pattern.fullmatch(preprocess_source_closure_sha256) is None or
            (declared_calibration_id is not None and
             (not isinstance(declared_calibration_id, str) or not declared_calibration_id)) or
            (declared_raw_shape is not None and
             (not isinstance(declared_raw_shape, dict) or
              set(declared_raw_shape) != {"height", "width"} or
              type(declared_raw_shape.get("height")) is not int or
              declared_raw_shape["height"] < 1 or
              type(declared_raw_shape.get("width")) is not int or
              declared_raw_shape["width"] < 1)) or
            not isinstance(onnx_sha, str) or sha_pattern.fullmatch(onnx_sha) is None or
            not isinstance(engine_name, str) or not engine_name or
            not isinstance(engine_sha, str) or sha_pattern.fullmatch(engine_sha) is None or
            isinstance(pop_strength, bool) or not isinstance(pop_strength, (int, float)) or
            not math.isfinite(float(pop_strength)) or not 0.25 <= float(pop_strength) <= 2.0):
        raise ValueError(
            f"{run_root}: results.json has incomplete model/engine/pop contract")
    clips = payload.get("clips")
    raw_manifests = meta.get(whole_clip_raw_contract.RESULTS_META_KEY)
    fused_manifests = meta.get(convex2x_diagnostics.RESULTS_META_KEY)
    composite_runtime = meta.get("composite_runtime_provenance")
    if (not isinstance(clips, dict) or not clips or
            not isinstance(raw_manifests, dict) or set(raw_manifests) != set(clips)):
        raise ValueError(
            f"{run_root}: schema-{whole_clip_raw_contract.EVALUATOR_SCHEMA} results do not "
            "bind raw artifacts for the exact scored clip set")
    clip_calibrations: Dict[str, object] = {}
    capture_grid_by_clip: Dict[str, str] = {}
    producer_calibrations: Dict[str, object] = {}
    for clip_name, manifest in raw_manifests.items():
        try:
            whole_clip_raw_contract.validate_manifest(manifest)
        except ValueError as exc:
            raise ValueError(f"{run_root}/{clip_name}: {exc}") from exc
        raw_shape = manifest["raw_shape"]
        producer_identity = manifest["producer_model_identity"]
        clip_entry = clips.get(clip_name)
        expected_clip_summary = {
            "calibration_status": manifest["calibration_status"],
            "calibration_id": manifest["calibration_id"],
            "preprocess_profile": producer_identity["preprocess_profile"],
            "raw_shape": raw_shape,
        }
        if (not isinstance(clip_entry, dict) or
                not isinstance(clip_entry.get("meta"), dict) or
                clip_entry["meta"].get("raw_model_identity") != expected_clip_summary):
            raise ValueError(
                f"{run_root}/{clip_name}: results omit the exact clip-level raw identity")
        if (producer_identity["model"] != model or
                producer_identity["depth_model_url"] != model_url or
                producer_identity["onnx_sha256"] != onnx_sha or
                producer_identity["preprocess_source_closure_sha256"] !=
                preprocess_source_closure_sha256):
            raise ValueError(
                f"{run_root}/{clip_name}: raw producer identity disagrees with results.json")
        matches = [
            value for value in MODEL_CALIBRATIONS
            if (value.depth_model == producer_identity["model"] and
                value.depth_model_url == producer_identity["depth_model_url"] and
                value.onnx_sha256 == producer_identity["onnx_sha256"] and
                value.preprocess.profile == producer_identity["preprocess_profile"] and
                value.preprocess.source_closure_sha256 ==
                producer_identity["preprocess_source_closure_sha256"])
        ]
        if len(matches) > 1:
            raise ValueError(f"{run_root}/{clip_name}: ambiguous raw model calibration")
        calibration = matches[0] if matches else None
        if calibration is not None:
            producer_calibrations[calibration.calibration_id] = calibration
        raw_extent = raw_shape["width"], raw_shape["height"]
        coarse_supported = (
            calibration is not None and raw_extent in calibration.calibrated_input_shapes)
        supported_high = {
            (shape.width, shape.height)
            for shape in convex2x_contract.supported_high_shapes()
        }
        single_high_candidate = (
            calibration is not None and raw_extent in supported_high and
            raw_extent[0] % 2 == 0 and raw_extent[1] % 2 == 0 and
            (raw_extent[0] // 2, raw_extent[1] // 2) in
            calibration.calibrated_input_shapes)
        single_high_authenticated = False
        if single_high_candidate:
            if (not isinstance(composite_runtime, dict) or
                    not isinstance(fused_manifests, dict) or
                    set(fused_manifests) != set(clips) or
                    clip_name not in fused_manifests):
                raise ValueError(
                    f"{run_root}/{clip_name}: single-high raw field lacks schema-2 fused evidence")
            fused_manifest = fused_manifests[clip_name]
            if (not isinstance(fused_manifest, dict) or
                    fused_manifest.get("schema") != convex2x_diagnostics.MANIFEST_SCHEMA):
                raise ValueError(
                    f"{run_root}/{clip_name}: single-high raw field has stale split fused evidence")
            try:
                convex2x_diagnostics.authenticate_manifest_files(
                    run_root / clip_name,
                    fused_manifest,
                    [int(row["frame_id"]) for row in manifest["frames"]],
                    composite_runtime,
                )
            except ValueError as exc:
                raise ValueError(
                    f"{run_root}/{clip_name}: invalid single-high fused evidence: {exc}") from exc
            output_shape = fused_manifest["tensor_shapes"]["output"]
            embedded = fused_manifest["embedded_dav2_provenance"]
            projected_identity = {
                key: producer_identity[key]
                for key in (
                    "model", "depth_model_url", "onnx_sha256", "preprocess_profile",
                    "preprocess_source_closure_sha256")
            }
            if (output_shape != {
                        "width": raw_extent[0], "height": raw_extent[1], "channels": 1} or
                    embedded != projected_identity):
                raise ValueError(
                    f"{run_root}/{clip_name}: fused evidence does not bind this high raw field")
            single_high_authenticated = True

        if calibration is None:
            expected_status = "abstain-unsupported-model-contract"
            expected_calibration_id = None
        elif not coarse_supported and not single_high_authenticated:
            expected_status = "abstain-unsupported-shape"
            expected_calibration_id = calibration.calibration_id
        else:
            expected_status = "calibrated"
            expected_calibration_id = calibration.calibration_id
        if (manifest["calibration_status"] != expected_status or
                manifest["calibration_id"] != expected_calibration_id):
            raise ValueError(
                f"{run_root}/{clip_name}: raw calibration status disagrees with the exact "
                "producer identity and shape")
        clip_calibrations[clip_name] = calibration if expected_status == "calibrated" else None
        if expected_status == "calibrated":
            capture_grid_by_clip[clip_name] = (
                "single-high-convex2x" if single_high_authenticated else "legacy-dav2")

    all_calibrated = all(
        manifest["calibration_status"] == "calibrated" for manifest in raw_manifests.values())
    run_tuples = {
        (manifest["calibration_id"],
         manifest["producer_model_identity"]["preprocess_profile"],
         manifest["raw_shape"]["width"], manifest["raw_shape"]["height"])
        for manifest in raw_manifests.values()
    }
    if len(producer_calibrations) == 1:
        run_calibration = next(iter(producer_calibrations.values()))
        expected_id = run_calibration.calibration_id
        expected_profile = run_calibration.preprocess.profile
    else:
        expected_id = expected_profile = None
    if all_calibrated and len(run_tuples) == 1:
        _, _, expected_width, expected_height = next(iter(run_tuples))
        expected_run_shape = {"height": expected_height, "width": expected_width}
    else:
        expected_run_shape = None
    if (declared_calibration_id != expected_id or preprocess_profile != expected_profile or
            declared_raw_shape != expected_run_shape):
        raise ValueError(
            f"{run_root}: run-level calibration/profile/shape overclaims mixed clip evidence")
    if len(producer_calibrations) > 1:
        raise ValueError(f"{run_root}: clips require incompatible v2 mapping calibrations")
    calibration = next(iter(producer_calibrations.values()), None)
    return {
        "results_sha256": _file_sha256(results_path),
        "eval_schema": whole_clip_raw_contract.EVALUATOR_SCHEMA,
        "model": model,
        "depth_model_url": model_url,
        "preprocess_profile": (
            calibration.preprocess.profile if calibration is not None else None),
        "preprocess_source_closure_sha256": preprocess_source_closure_sha256,
        "onnx_sha256": onnx_sha,
        "engine_name": engine_name,
        "engine_sha256": engine_sha,
        "calibration_id": calibration.calibration_id if calibration is not None else None,
        "raw_coordinate_scale": (
            calibration.raw_coordinate_scale if calibration is not None else None),
        "pop_strength": float(pop_strength),
        "calibrated_input_shapes": (
            [list(value) for value in calibration.calibrated_input_shapes]
            if calibration is not None else []),
        "calibration_by_clip": clip_calibrations,
        "capture_grid_by_clip": capture_grid_by_clip,
        "whole_clip_raw_artifacts": raw_manifests,
    }


def _clip_sequence_input_contract(
        clip_dir: Path,
        frame_ids: Sequence[int],
        shape: Tuple[int, int],
        run_model: Dict[str, object]) -> Dict[str, object]:
    """Bind ordered raw bytes, shape, model, cut evidence, and selected output IDs."""

    contract_path = clip_dir / "contract.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{clip_dir}: invalid contract.json: {exc}") from exc
    if (not isinstance(contract, dict) or
            contract.get("schema") != whole_clip_raw_contract.HARNESS_CONTRACT_SCHEMA or
            contract.get("model") != run_model["model"] or
            contract.get("pop_strength") != run_model["pop_strength"] or
            not isinstance(contract.get("depth_step"), str) or not contract["depth_step"] or
            type(contract.get("depth_reuse_interval")) is not int or
            contract["depth_reuse_interval"] < 1):
        raise ValueError(
            f"{clip_dir}: contract.json model/pop/schema/depth schedule disagrees with run contract")
    expected_ids = set(int(value) for value in frame_ids)

    def artifact_ids(pattern: str, prefix: str) -> set[int]:
        observed: Dict[int, Path] = {}
        for path in clip_dir.glob(pattern):
            suffix = path.stem[len(prefix):]
            if not suffix.isdigit():
                raise ValueError(f"{clip_dir}: malformed artifact identity {path.name!r}")
            frame_id = int(suffix)
            if frame_id in observed:
                raise ValueError(
                    f"{clip_dir}: duplicate numeric {prefix} frame ID {frame_id}")
            observed[frame_id] = path
        return set(observed)

    artifact_sets = {
        "sbs": artifact_ids("sbs_*.png", "sbs_"),
        "depth": artifact_ids("depth_*.png", "depth_"),
        "warp_map": artifact_ids("warp_map_*.f32", "warp_map_"),
    }
    mismatched = {name: sorted(ids) for name, ids in artifact_sets.items()
                  if ids != expected_ids}
    state_path = clip_dir / "cut_state.json"
    if not state_path.exists():
        mismatched["cut_state"] = []
    else:
        state_ids = set(cut_state_contract.load_trace(str(state_path)))
        if state_ids != expected_ids:
            mismatched["cut_state"] = sorted(state_ids)
    if mismatched:
        raise ValueError(
            f"{clip_dir}: selected raw IDs do not equal complete output identities: {mismatched}")
    raw_shape_path = clip_dir / "raw_shape.json"
    raw_paths = dict(_raw_files(clip_dir))
    if set(raw_paths) != expected_ids:
        raise ValueError(f"{clip_dir}: raw field IDs changed while building input contract")
    manifest = run_model["whole_clip_raw_artifacts"].get(clip_dir.name)
    if manifest is None:
        raise ValueError(f"{clip_dir}: results.json has no raw artifact binding for this clip")
    try:
        manifest_frames = whole_clip_raw_contract.authenticate_manifest_files(
            clip_dir, manifest, frame_ids)
    except ValueError as exc:
        raise ValueError(f"{clip_dir}: {exc}") from exc
    if manifest["raw_shape"] != {"height": shape[0], "width": shape[1]}:
        raise ValueError(f"{clip_dir}: loaded raw shape disagrees with its run manifest")
    if manifest["calibration_status"] != "calibrated":
        raise ValueError(
            f"{clip_dir}: v2 replay abstains for this clip: "
            f"{manifest['abstention_reason']}")
    clip_calibration = run_model["calibration_by_clip"].get(clip_dir.name)
    capture_grid = run_model["capture_grid_by_clip"].get(clip_dir.name)
    if (clip_calibration is None or
            capture_grid not in {"legacy-dav2", "single-high-convex2x"} or
            run_model["calibration_id"] != clip_calibration.calibration_id):
        raise ValueError(
            f"{clip_dir}: calibrated raw identity is inconsistent with the run mapping")
    raw_fields = [
        {"frame_id": row["frame_id"], "sha256": row["sha256"]}
        for row in manifest_frames
    ]
    committed_meta = Path(__file__).resolve().parent / "clips" / clip_dir.name / "meta.json"
    cut_control_evidence = {
        "kind": "cut-state-hard-cut-generation",
        "sha256": _file_sha256(state_path),
    }
    cut_expectation_evidence = (
        {"kind": "committed-meta-scoring-only", "sha256": _file_sha256(committed_meta)}
        if committed_meta.exists() else
        {"kind": "none", "sha256": None}
    )
    selected = [f"{frame_id:05d}" for frame_id in frame_ids]
    evidence: Dict[str, object] = {
        "contract_json_sha256": _file_sha256(contract_path),
        "contract_schema": contract["schema"],
        "eval_schema": run_model["eval_schema"],
        "depth_step": contract["depth_step"],
        "depth_reuse_interval": contract["depth_reuse_interval"],
        "model": run_model["model"],
        "depth_model_url": run_model["depth_model_url"],
        "preprocess_profile": run_model["preprocess_profile"],
        "preprocess_source_closure_sha256":
            run_model["preprocess_source_closure_sha256"],
        "onnx_sha256": run_model["onnx_sha256"],
        "engine_name": run_model["engine_name"],
        "engine_sha256": run_model["engine_sha256"],
        "calibration_id": run_model["calibration_id"],
        "raw_coordinate_scale": run_model["raw_coordinate_scale"],
        "run_pop_strength": run_model["pop_strength"],
        "results_json_sha256": run_model["results_sha256"],
        "model_hash_authority": "schema-38-run-level-results-json",
        "input_shape_authority": "schema-38-per-clip-raw-manifest",
        "raw_hash_authority": {
            "source": "schema-38-run-level-results-json",
            "manifest_schema": whole_clip_raw_contract.MANIFEST_SCHEMA,
            "binding": whole_clip_raw_contract.BINDING,
        },
        "raw_shape": {"height": shape[0], "width": shape[1]},
        "raw_shape_json_sha256": _file_sha256(raw_shape_path),
        "selected_frame_ids": selected,
        "ordered_raw_fields": raw_fields,
        "cut_control_evidence": cut_control_evidence,
        "cut_expectation_evidence": cut_expectation_evidence,
    }
    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    evidence["sequence_input_sha256"] = hashlib.sha256(canonical).hexdigest()
    return evidence
