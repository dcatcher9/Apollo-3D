#!/usr/bin/env python3
"""Whole-clip temporal reference for the host SBS depth coordinate.

This module deliberately consumes the harness' ``raw_<frame>.f32`` artifacts.  It does not
replace the live controller and it does not judge reprojection quality.  Its narrower purpose is
to answer a question that single-frame baselines cannot: when should the continuous moment
coordinate be allowed to move inside a shot?

The policy-study CLI compares one selected baseline with two rejected research alternatives:

``first``
    Seed from the first usable depth field after a confirmed cut, then hold for the shot.

``aggregate``
    Seed immediately, observe 0.25 seconds, relatch once to the weighted median center and
    then hold. The fixed authenticated scale never moves. The window is measured in seconds,
    not frames.

``slow``
    Seed immediately and continuously correct center with a time-based EMA.

``generate_first_latch_exact_sequence`` is the selected NumPy comparison reference. It is
deliberately a separate state machine: it never constructs the research controller and never
iterates the aggregate/slow policy set. It cannot select rendered geometry; the latter policies
remain here only as falsifiers for the design decision.

Every policy resets immediately on a confirmed cut.  Source-U safety is an independent fail-safe: it may
only reduce the whole-field gain during a shot and is reset at the next cut.  The report uses
pre-reprojection one-eye source-U parallax so it does not confuse coordinate motion with holes,
occlusion, or image motion.  A subsampled raw field is sufficient for policy comparisons and
makes whole-suite runs inexpensive.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .depth_mapping_v2 import (
        DIRECT_PARALLAX_SOURCE_U_LIMIT,
        MappingV2Config,
        asymmetric_curve,
        calibrate_coordinate,
        container_scale_for_curve_range,
        curve_relative_coordinate,
        dense_near_weight,
        encode_direct_parallax,
        effective_near_log_tau,
        horizontal_lipschitz_majorant,
        near_tail_count_and_coverage,
        vertical_lipschitz_majorant,
    )
    from .depth_coordinate_v2_contract import (
        CALIBRATED_DEFAULTS as V2_DEFAULTS, CONTRACT_CANONICAL_SHA256,
        CONTRACT_PATH, CONTRACT_SCHEMA, MODEL_CALIBRATIONS)
    from . import subject_state_contract, whole_clip_raw_contract
except ImportError:  # Direct execution from tools/sbsbench.
    from depth_mapping_v2 import (  # type: ignore
        DIRECT_PARALLAX_SOURCE_U_LIMIT,
        MappingV2Config,
        asymmetric_curve,
        calibrate_coordinate,
        container_scale_for_curve_range,
        curve_relative_coordinate,
        dense_near_weight,
        encode_direct_parallax,
        effective_near_log_tau,
        horizontal_lipschitz_majorant,
        near_tail_count_and_coverage,
        vertical_lipschitz_majorant,
    )
    from depth_coordinate_v2_contract import (  # type: ignore
        CALIBRATED_DEFAULTS as V2_DEFAULTS, CONTRACT_CANONICAL_SHA256,
        CONTRACT_PATH, CONTRACT_SCHEMA, MODEL_CALIBRATIONS)
    import subject_state_contract  # type: ignore
    import whole_clip_raw_contract  # type: ignore


RAW_PATTERN = re.compile(r"raw_(\d+)\.f32$")
SELECTED_POLICY = "first"
RESEARCH_POLICIES = ("aggregate", "slow")
POLICY_STUDY_POLICIES = (SELECTED_POLICY, *RESEARCH_POLICIES)
V2_STATE_TRACE_SCHEMA = 10
V2_STATE_TRACE_POLICY = (
    "latched-center-scale-near-tail-retained-camera-base-container-vertical-shear2-near-majorant-v8"
)
V2_GPU_SHADER_SEQUENCE = (
    "depth_coordinate_v2_moments_cs.hlsl",
    "depth_coordinate_v2_frame_resolve_cs.hlsl",
    "depth_coordinate_v2_near_coverage_cs.hlsl",
    "depth_coordinate_v2_state_resolve_cs.hlsl",
    "depth_coordinate_v2_map_cs.hlsl",
    "depth_coordinate_v2_vertical_limit_cs.hlsl",
    "depth_coordinate_v2_limit_cs.hlsl",
)
V2_STATE_TRACE_FIELDS = (
    "frame_id",
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
    "latched_near_tail_coverage",
    "effective_near_log_tau",
    "latched_near_tail_count",
    "requested_gain",
    "container_scale",
    "effective_gain",
    "observed_mean",
    "observed_std",
    "observed_raw_minimum",
    "observed_raw_maximum",
    "candidate_center_drift_u",
    "predicted_zero_translation_source_u",
    "pre_limiter_max_abs_source_u",
    "vertical_majorant_raised_fraction",
    "vertical_majorant_max_raise_source_u",
    "final_max_abs_source_u",
    "limiter_raised_fraction",
    "limiter_max_raise_source_u",
    "final_horizontal_slope_max",
    "final_vertical_shear_max",
    "order_sha256",
    "vertical_majorant_sha256",
    "parallax_sha256",
)


@dataclass(frozen=True)
class MomentCandidate:
    center: float
    scale: float
    observed_std: float
    raw_min: float
    raw_max: float
    collapsed: bool
    near_tail_count: int
    near_tail_coverage: float
    dense_near_weight: float
    effective_near_log_tau: float


@dataclass(frozen=True)
class CoordinateState:
    """Shot camera plus an explicit, separately latched curve-space convergence."""

    center: float
    scale: float
    convergence_curve: float = 0.0
    valid: bool = True
    near_tail_count: int = 0
    near_tail_coverage: float = 0.0
    dense_near_weight: float = 0.0
    effective_near_log_tau: float = V2_DEFAULTS.near_log_tau


@dataclass(frozen=True)
class TemporalConfig:
    """Configuration for the offline policy comparison, not the production controller."""

    policy: str
    aggregate_seconds: float = 0.25
    slow_tau_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.policy not in POLICY_STUDY_POLICIES:
            raise ValueError(f"unsupported temporal policy: {self.policy}")
        if not math.isfinite(self.aggregate_seconds) or self.aggregate_seconds <= 0.0:
            raise ValueError("aggregate_seconds must be finite and positive")
        if not math.isfinite(self.slow_tau_seconds) or self.slow_tau_seconds <= 0.0:
            raise ValueError("slow_tau_seconds must be finite and positive")


@dataclass(frozen=True)
class UpdateResult:
    state: CoordinateState
    reset: bool
    relatched: bool


@dataclass(frozen=True)
class ExactSequenceResult:
    """Fields and state attestation for an exact whole-clip direct replay.

    ``canonical_fields`` preserve DAV2 ordering before the spatial projection. ``parallax_fields``
    are final one-eye source-U displacement after frame-local hard safety, the least column-wise
    vertical near-preserving majorant, and then the least row-wise horizontal near-preserving
    majorant. An unusable field publishes flat geometry for the current color while retaining the
    scene camera unless an authenticated cut also arrives. Near-tail occupancy, its smoothstep
    weight, and its effective logarithmic tau are acquired with the camera and held for the shot.
    There is deliberately no timed or frame-counted hold policy.
    """

    frame_ids: Tuple[int, ...]
    canonical_fields: Tuple[np.ndarray, ...]
    parallax_fields: Tuple[np.ndarray, ...]
    state_trace: Dict[str, object]


def moment_candidate(
        raw_depth: np.ndarray,
        config: MappingV2Config = MappingV2Config()) -> MomentCandidate:
    """Compute the same continuous coordinate candidate as ``depth_mapping_v2``."""

    calibration = calibrate_coordinate(raw_depth, config)
    canonical = ((np.asarray(raw_depth, dtype=np.float64) - calibration.center) /
                 calibration.scale)
    tail_count, tail_coverage = near_tail_count_and_coverage(canonical, config)
    tail_weight = dense_near_weight(tail_coverage, config)
    return MomentCandidate(
        center=calibration.center,
        scale=calibration.scale,
        observed_std=calibration.observed_std,
        raw_min=calibration.raw_min,
        raw_max=calibration.raw_max,
        collapsed=calibration.collapsed,
        near_tail_count=tail_count,
        near_tail_coverage=tail_coverage,
        dense_near_weight=tail_weight,
        effective_near_log_tau=effective_near_log_tau(tail_coverage, config),
    )


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    if (values_array.ndim != 1 or weights_array.shape != values_array.shape or
            values_array.size == 0 or not np.isfinite(values_array).all() or
            not np.isfinite(weights_array).all() or np.any(weights_array < 0.0) or
            float(np.sum(weights_array)) <= 0.0):
        raise ValueError("weighted quantile requires finite values and positive total weight")
    if not 0.0 <= q <= 1.0:
        raise ValueError("weighted quantile q must be in [0, 1]")
    order = np.argsort(values_array, kind="stable")
    ordered_values = values_array[order]
    ordered_weights = weights_array[order]
    # Midpoint ranks avoid an update-rate-dependent bias toward the first/last observation.
    ranks = (np.cumsum(ordered_weights) - 0.5 * ordered_weights) / np.sum(ordered_weights)
    return float(np.interp(q, ranks, ordered_values,
                           left=ordered_values[0], right=ordered_values[-1]))


class TemporalCoordinateController:
    """Small deterministic reference state machine shared by the report and tests."""

    def __init__(
            self,
            temporal: TemporalConfig,
            mapping: MappingV2Config = MappingV2Config()) -> None:
        self.temporal = temporal
        self.mapping = mapping
        self.state: Optional[CoordinateState] = None
        self.shot_start = 0.0
        self.last_time: Optional[float] = None
        self.relatched = False
        self.samples: List[MomentCandidate] = []
        self.sample_weights: List[float] = []

    def update(self, candidate: MomentCandidate, time_seconds: float, cut: bool) -> UpdateResult:
        if not math.isfinite(time_seconds):
            raise ValueError("time_seconds must be finite")
        if self.last_time is not None and time_seconds < self.last_time:
            raise ValueError("time_seconds must be monotonic")

        if candidate.collapsed:
            if cut:
                self.state = None
            self.samples = []
            self.sample_weights = []
            self.relatched = False
            self.last_time = time_seconds
            return UpdateResult(
                self.state or CoordinateState(0.0, 1.0, 0.0, False),
                reset=cut, relatched=False)

        reset = self.state is None or cut
        if reset:
            state = CoordinateState(
                candidate.center,
                candidate.scale,
                near_tail_count=candidate.near_tail_count,
                near_tail_coverage=candidate.near_tail_coverage,
                dense_near_weight=candidate.dense_near_weight,
                effective_near_log_tau=candidate.effective_near_log_tau,
            )
            self.state = state
            self.shot_start = time_seconds
            self.relatched = False
            self.samples = [candidate]
            # The first observation represents the interval until the next update.  A tiny
            # positive seed keeps a one-frame shot well-defined without materially biasing it.
            self.sample_weights = [np.finfo(np.float64).eps]
            self.last_time = time_seconds
            return UpdateResult(self.state, reset=True, relatched=False)

        assert self.state is not None
        assert self.last_time is not None
        dt = time_seconds - self.last_time
        # Attribute elapsed time to the previously observed field.  This is a zero-order hold and
        # makes aggregate membership nearly invariant to 72/90 Hz and to depth reuse.
        if self.temporal.policy == "aggregate" and not self.relatched:
            self.sample_weights[-1] += dt
            self.samples.append(candidate)
            self.sample_weights.append(np.finfo(np.float64).eps)

        state = self.state
        did_relatch = False
        if self.temporal.policy == "aggregate":
            elapsed = time_seconds - self.shot_start
            if not self.relatched and elapsed + 1.0e-12 >= self.temporal.aggregate_seconds:
                center = _weighted_quantile(
                    [sample.center for sample in self.samples], self.sample_weights, 0.50)
                state = CoordinateState(
                    center, state.scale, state.convergence_curve, True,
                    state.near_tail_count, state.near_tail_coverage,
                    state.dense_near_weight, state.effective_near_log_tau)
                self.relatched = True
                did_relatch = True
        elif self.temporal.policy == "slow":
            alpha = -math.expm1(-dt / self.temporal.slow_tau_seconds)
            center = state.center + alpha * (candidate.center - state.center)
            state = CoordinateState(
                center, state.scale, state.convergence_curve, True,
                state.near_tail_count, state.near_tail_coverage,
                state.dense_near_weight, state.effective_near_log_tau)

        self.state = state
        self.last_time = time_seconds
        return UpdateResult(self.state, reset=False, relatched=did_relatch)


def parallax_for_state(
        raw_depth: np.ndarray,
        state: CoordinateState,
        config: MappingV2Config = MappingV2Config()) -> np.ndarray:
    """Map a raw field with an externally supplied, shot-persistent coordinate.

    This intentionally stops before the two-axis Lipschitz projection. The vertical and
    horizontal majorants are spatial safety operations; including them would make a temporal
    coordinate comparison depend on edge layout and obscure the state-induced delta being
    measured.
    """

    raw = np.asarray(raw_depth, dtype=np.float64)
    if not state.valid:
        return np.zeros(raw.shape, dtype=np.float64)
    _, curved = curve_relative_coordinate(
        raw, state.center, state.scale, config,
        convergence_curve=state.convergence_curve,
        near_log_tau=state.effective_near_log_tau)
    parallax = curved * config.parallax_gain
    if not np.isfinite(parallax).all():
        raise ValueError("temporal mapping produced non-finite parallax")
    return parallax


def map_timeline_flat_on_unusable(
        raw_fields: Sequence[np.ndarray],
        candidates: Sequence[MomentCandidate],
        states: Sequence[CoordinateState],
        cuts: Sequence[bool],
        config: MappingV2Config = MappingV2Config()) -> List[np.ndarray]:
    """Map each current color with flat geometry whenever its depth is unusable."""

    if not (len(raw_fields) == len(candidates) == len(states) == len(cuts)):
        raise ValueError("timeline fields, candidates, states, and cuts must have equal length")
    outputs: List[np.ndarray] = []
    for raw, candidate, state, _cut in zip(raw_fields, candidates, states, cuts):
        if candidate.collapsed:
            outputs.append(np.zeros(np.asarray(raw).shape, dtype=np.float64))
            continue
        outputs.append(parallax_for_state(raw, state, config))
    return outputs


def _exact_counter_increment(value: int) -> int:
    """Match the GPU counter's reserved-sentinel saturation rule."""

    return min(int(value), 0xFFFFFFFD) + 1


def _field_sha256(field: np.ndarray) -> str:
    values = np.asarray(field, dtype="<f4")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_container_scale(
        raw_minimum: float,
        raw_maximum: float,
        center: float,
        scale: float,
        convergence_curve: float,
        config: MappingV2Config) -> float:
    """Mirror the shader's conservative base-tau hard-container calculation.

    Scene occupancy may select a smaller effective tau for rendering. It must not loosen the
    representation envelope, so extrema here intentionally use ``config.near_log_tau``.
    """

    endpoint_coordinates = np.asarray([
        (float(raw_minimum) - center) / scale,
        (float(raw_maximum) - center) / scale,
    ])
    endpoints = asymmetric_curve(
        endpoint_coordinates, config,
        near_log_tau=config.near_log_tau) - convergence_curve
    container_scale, _ = container_scale_for_curve_range(
        float(endpoints[0]), float(endpoints[1]), config)
    return container_scale


def validate_v2_state_trace(
        trace: Dict[str, object],
        expected_frame_ids: Sequence[int]) -> None:
    """Fail closed unless every row proves the simplified persistent-camera contract."""

    root_keys = {
        "schema", "policy", "calibration_contract", "cut_source", "diagnostic_role",
        "diagnostic_method", "mapping_config",
        "frame_fields", "frames", "producer",
    }
    if not isinstance(trace, dict) or set(trace) != root_keys:
        raise ValueError("v2 state trace has missing or unknown root fields")
    if (trace.get("schema") != V2_STATE_TRACE_SCHEMA or
            trace.get("policy") != V2_STATE_TRACE_POLICY or
            trace.get("diagnostic_role") != "non-controlling-fixed-scale-camera-audit-v4" or
            trace.get("diagnostic_method") not in {
                "frame-moment-proxies-not-matched-pixel-affine-v2",
                "gpu-frame-moments-near-tail-and-rendered-fields-v4"} or
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
    if not isinstance(mapping_payload, dict) or set(mapping_payload) != expected_mapping_keys:
        raise ValueError("v2 state trace mapping config is missing or unknown")
    try:
        mapping = MappingV2Config(**mapping_payload)
        # Trigger the complete mapping validation without coupling this module to private helpers.
        asymmetric_curve(np.asarray([0.0], dtype=np.float64), mapping)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"v2 state trace mapping config is invalid: {exc}") from exc
    producer = trace.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("v2 state trace has an invalid producer authority")
    producer_authority = producer.get("authority")
    expected_method = None
    native_texel_count: Optional[int] = None
    if producer_authority == "numpy-reference-comparison-only-v3":
        if (set(producer) != {"authority", "numpy_role"} or
                producer.get("numpy_role") != "comparison-only-not-render-authority"):
            raise ValueError("v2 state trace has invalid NumPy producer evidence")
        expected_method = "frame-moment-proxies-not-matched-pixel-affine-v2"
    elif producer_authority == (
            "seven-experimental-shadow-compute-shaders-persistent-gpu-state-v3"):
        digest_pattern = re.compile(r"[0-9a-f]{64}")
        if (set(producer) != {
                "authority", "manifest_sha256", "contract_canonical_sha256",
                "tensor_shape", "shader_sequence", "state_persistence", "numpy_role"} or
                not isinstance(producer.get("manifest_sha256"), str) or
                digest_pattern.fullmatch(producer["manifest_sha256"]) is None or
                producer.get("contract_canonical_sha256") != CONTRACT_CANONICAL_SHA256 or
                producer.get("shader_sequence") != list(V2_GPU_SHADER_SEQUENCE) or
                producer.get("state_persistence") != "single-buffer-whole-sequence" or
                producer.get("numpy_role") != "comparison-only-not-render-authority"):
            raise ValueError("v2 state trace has invalid native GPU producer evidence")
        expected_method = "gpu-frame-moments-near-tail-and-rendered-fields-v4"
        calibrated_shapes = {
            shape
            for calibration in MODEL_CALIBRATIONS
            if math.isclose(
                calibration.raw_coordinate_scale, mapping.raw_coordinate_scale,
                rel_tol=0.0, abs_tol=1.0e-12)
            for shape in calibration.calibrated_input_shapes
        }
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
        native_texel_count = calibrated_width * calibrated_height
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
    prior_near_tail_count = 0
    prior_near_tail_coverage = 0.0
    prior_effective_near_log_tau = mapping.near_log_tau
    requested_gain: Optional[float] = None
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    for index, (row, expected_id) in enumerate(zip(rows, expected_ids)):
        if not isinstance(row, dict) or set(row) != set(V2_STATE_TRACE_FIELDS):
            raise ValueError(f"v2 state trace frame {index} has an invalid field layout")
        frame_id = row["frame_id"]
        if frame_id != f"{expected_id:05d}":
            raise ValueError(f"v2 state trace frame-id mismatch at index {index}")
        if (row["input_source_frame_id"] != frame_id or
                row["rendered_source_frame_id"] != frame_id):
            raise ValueError(f"v2 state trace {frame_id} has invalid source-frame identity")
        for key in ("calibration_revision", "confirmed_cut_count",
                    "latched_near_tail_count"):
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
                "latched_near_tail_coverage", "effective_near_log_tau",
                "requested_gain", "container_scale", "effective_gain",
                "observed_mean", "observed_std", "observed_raw_minimum",
                "observed_raw_maximum",
                "candidate_center_drift_u", "predicted_zero_translation_source_u",
                "pre_limiter_max_abs_source_u",
                "vertical_majorant_raised_fraction",
                "vertical_majorant_max_raise_source_u", "final_max_abs_source_u",
                "limiter_raised_fraction", "limiter_max_raise_source_u",
                "final_horizontal_slope_max", "final_vertical_shear_max"):
            value = row[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value))):
                raise ValueError(f"v2 state trace {frame_id} has non-finite {key}")
        expected_collapsed = bool(
            row["input_valid"] and
            row["observed_std"] <= mapping.collapse_abs_epsilon)
        expected_frame_valid = bool(row["input_valid"] and not expected_collapsed)
        if (row["collapsed"] != expected_collapsed or
                row["frame_valid"] != expected_frame_valid):
            raise ValueError(
                f"v2 state trace {frame_id} has inconsistent input/collapse/frame validity")
        for key in ("order_sha256", "vertical_majorant_sha256", "parallax_sha256"):
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
        latch = bool(
            row["frame_valid"] and
            (not prior_camera_initialized or row["confirmed_cut"]))
        expected_revision = (_exact_counter_increment(prior_revision)
                             if latch else prior_revision)
        if revision != expected_revision:
            raise ValueError(f"v2 state trace {frame_id} calibration revision is inconsistent")
        expected_attribution = (
            "initialization" if index == 0 else
            trace["cut_source"] if row["confirmed_cut"] else "none"
        )
        if row["cut_attribution"] != expected_attribution:
            raise ValueError(f"v2 state trace {frame_id} cut attribution is inconsistent")
        if not 0.0 <= row["container_scale"] <= 1.0:
            raise ValueError(f"v2 state trace {frame_id} has invalid container scale")
        if not 0.0 <= row["latched_near_tail_coverage"] <= 1.0:
            raise ValueError(f"v2 state trace {frame_id} has invalid near-tail coverage")
        if ((row["latched_near_tail_count"] == 0) !=
                (row["latched_near_tail_coverage"] == 0.0)):
            raise ValueError(f"v2 state trace {frame_id} near-tail count/coverage disagree")
        if native_texel_count is not None:
            if row["latched_near_tail_count"] > native_texel_count:
                raise ValueError(f"v2 state trace {frame_id} near-tail count exceeds shape")
            expected_coverage = row["latched_near_tail_count"] / native_texel_count
            if not math.isclose(
                    row["latched_near_tail_coverage"], expected_coverage,
                    rel_tol=0.0, abs_tol=2.0e-6):
                raise ValueError(
                    f"v2 state trace {frame_id} near-tail count/coverage disagree")
        if not (mapping.near_log_tau_dense <= row["effective_near_log_tau"] <=
                mapping.near_log_tau):
            raise ValueError(f"v2 state trace {frame_id} has invalid effective near tau")
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
                    "limiter_raised_fraction", "limiter_max_raise_source_u",
                    "final_horizontal_slope_max", "final_vertical_shear_max"):
            if row[key] < 0.0:
                raise ValueError(f"v2 state trace {frame_id} has negative {key}")
        for key in ("vertical_majorant_raised_fraction", "limiter_raised_fraction"):
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
                raise ValueError(f"v2 state trace {frame_id} changed convergence curve")
            expected_near_tau = effective_near_log_tau(
                row["latched_near_tail_coverage"], mapping)
            if not math.isclose(
                    row["effective_near_log_tau"], expected_near_tau,
                    rel_tol=3.0e-5, abs_tol=3.0e-7):
                raise ValueError(f"v2 state trace {frame_id} near-tail curve is inconsistent")
            expected_container = _frame_container_scale(
                row["observed_raw_minimum"], row["observed_raw_maximum"],
                row["center"], row["latched_scale"], row["convergence_curve"], mapping)
            if not math.isclose(row["container_scale"], expected_container,
                                rel_tol=3.0e-5, abs_tol=3.0e-7):
                raise ValueError(f"v2 state trace {frame_id} frame container is inconsistent")
            expected_gain = row["requested_gain"] * row["container_scale"]
            if not math.isclose(row["effective_gain"], expected_gain, rel_tol=3.0e-5,
                                abs_tol=3.0e-7):
                raise ValueError(f"v2 state trace {frame_id} effective gain is inconsistent")
            if latch:
                expected_scale = mapping.raw_coordinate_scale
                if (not math.isclose(row["center"], row["observed_mean"],
                                     rel_tol=2.0e-6, abs_tol=2.0e-7) or
                        not math.isclose(row["latched_scale"], expected_scale,
                                         rel_tol=3.0e-5, abs_tol=3.0e-7)):
                    raise ValueError(f"v2 state trace {frame_id} acquired the wrong camera")
            if prior_camera_initialized and not latch:
                if (not math.isclose(row["center"], prior_center,
                                     rel_tol=2.0e-6, abs_tol=2.0e-7) or
                        not math.isclose(row["inverse_scale"], prior_inverse_scale,
                                         rel_tol=2.0e-6, abs_tol=2.0e-7) or
                        not math.isclose(row["convergence_curve"], prior_convergence_curve,
                                         rel_tol=0.0, abs_tol=2.0e-7) or
                        row["latched_near_tail_count"] != prior_near_tail_count or
                        not math.isclose(row["latched_near_tail_coverage"],
                                         prior_near_tail_coverage,
                                         rel_tol=2.0e-6, abs_tol=2.0e-7) or
                        not math.isclose(row["effective_near_log_tau"],
                                         prior_effective_near_log_tau,
                                         rel_tol=2.0e-6, abs_tol=2.0e-7)):
                    raise ValueError(f"v2 state trace {frame_id} moved a latched coordinate")
        else:
            if row["effective_gain"] != 0.0 or row["container_scale"] != 1.0:
                raise ValueError(
                    f"v2 state trace {frame_id} unavailable frame retained active gain")
            expected_camera = prior_camera_initialized and not row["confirmed_cut"]
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
                                         rel_tol=0.0, abs_tol=2.0e-7) or
                        row["latched_near_tail_count"] != prior_near_tail_count or
                        not math.isclose(row["latched_near_tail_coverage"],
                                         prior_near_tail_coverage,
                                         rel_tol=2.0e-6, abs_tol=2.0e-7) or
                        not math.isclose(row["effective_near_log_tau"],
                                         prior_effective_near_log_tau,
                                         rel_tol=2.0e-6, abs_tol=2.0e-7)):
                    raise ValueError(
                        f"v2 state trace {frame_id} moved retained camera on unavailable depth")
            elif (row["center"] != 0.0 or row["inverse_scale"] != 0.0 or
                  row["latched_scale"] != 0.0 or
                  row["convergence_curve"] != V2_DEFAULTS.convergence_curve_default or
                  row["latched_near_tail_count"] != 0 or
                  row["latched_near_tail_coverage"] != 0.0 or
                  row["effective_near_log_tau"] != mapping.near_log_tau):
                raise ValueError(
                    f"v2 state trace {frame_id} failed to publish canonical empty camera")
        if row["final_max_abs_source_u"] > DIRECT_PARALLAX_SOURCE_U_LIMIT + 1.0e-7:
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
        prior_near_tail_count = row["latched_near_tail_count"]
        prior_near_tail_coverage = row["latched_near_tail_coverage"]
        prior_effective_near_log_tau = row["effective_near_log_tau"]


def generate_first_latch_exact_sequence(
        raw_fields: Sequence[np.ndarray],
        frame_ids: Sequence[int],
        cuts: Sequence[bool],
        cut_source: str,
        config: MappingV2Config = MappingV2Config(),
        *,
        confirmed_cut_counts: Optional[Sequence[int]] = None) -> ExactSequenceResult:
    """Generate comparison-only whole-clip geometry and a NumPy state trace.

    This independently mirrors only the selected policy: first usable field latches center, the
    authenticated model/shape scale stays fixed, unusable no-cut frames retain that camera while
    rendering flat, and the hard container remains frame-local. The native seven-shader sequence
    owns rendered fields; this result may only compare against them.
    """

    if not (len(raw_fields) == len(frame_ids) == len(cuts)):
        raise ValueError("raw fields, frame IDs, and cut flags must have equal length")
    if not raw_fields:
        raise ValueError("whole-clip exact replay requires at least one frame")
    ids = tuple(int(value) for value in frame_ids)
    if (len(set(ids)) != len(ids) or any(value < 0 for value in ids) or
            tuple(sorted(ids)) != ids):
        raise ValueError("whole-clip frame IDs must be unique, non-negative, and increasing")
    if not isinstance(cut_source, str) or not cut_source:
        raise ValueError("cut_source must identify the cut evidence")
    if confirmed_cut_counts is None:
        # Direct unit/research callers do not necessarily have a production subject-state trace.
        # Give them the same monotonic generation semantics, starting at zero.
        derived_counts: List[int] = []
        generation = 0
        for index, cut_flag in enumerate(cuts):
            if index > 0 and bool(cut_flag):
                generation = _exact_counter_increment(generation)
            derived_counts.append(generation)
        cut_counts = tuple(derived_counts)
    else:
        if len(confirmed_cut_counts) != len(frame_ids):
            raise ValueError("cut generations and frame IDs must have equal length")
        cut_counts = tuple(confirmed_cut_counts)
        if any(type(value) is not int or value < 0 or value > 0xFFFFFFFE
               for value in cut_counts):
            raise ValueError("cut generations must be uint32 values below the reserved sentinel")
        for index in range(1, len(cut_counts)):
            delta = cut_counts[index] - cut_counts[index - 1]
            if delta not in (0, 1) or (not bool(cuts[index]) and delta != 0):
                raise ValueError("cut generations disagree with the confirmed-cut timeline")

    fields = [np.asarray(field, dtype=np.float64) for field in raw_fields]
    shape = fields[0].shape
    if any(field.ndim != 2 or field.shape != shape or field.size == 0
           for field in fields):
        raise ValueError("whole-clip raw fields must be non-empty 2D arrays of one shape")

    canonical_fields: List[np.ndarray] = []
    parallax_fields: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    active_center = 0.0
    active_scale = 0.0
    convergence_curve = V2_DEFAULTS.convergence_curve_default
    latched_near_tail_count = 0
    latched_near_tail_coverage = 0.0
    latched_dense_near_weight = 0.0
    latched_effective_near_log_tau = config.near_log_tau
    container_scale = 1.0
    camera_valid = False
    calibration_revision = 0

    for index, (raw, frame_id, cut_flag, confirmed_cut_count) in enumerate(
            zip(fields, ids, cuts, cut_counts)):
        # _load_cut_indices marks frame zero as a scene boundary so policy simulation starts in a
        # known shot. It is initialization, not evidence that an upstream cut counter advanced.
        confirmed_cut = bool(cut_flag and index > 0)
        input_valid = bool(np.isfinite(raw).all())
        candidate = moment_candidate(raw, config) if input_valid else None
        collapsed = bool(candidate is not None and candidate.collapsed)
        frame_valid = bool(input_valid and not collapsed)
        pre_limiter_max_abs = 0.0
        vertical_majorant_raised_fraction = 0.0
        vertical_majorant_max_raise = 0.0
        final_max_abs = 0.0
        limiter_raised_fraction = 0.0
        limiter_max_raise = 0.0
        final_horizontal_slope_max = 0.0
        final_vertical_shear_max = 0.0
        if not frame_valid:
            # Never reuse stale per-pixel geometry. The scene camera is retained across an
            # unusable no-cut field, but an authenticated cut clears it so the next usable field
            # cannot inherit the previous scene's gauge.
            if confirmed_cut:
                camera_valid = False
                active_center = 0.0
                active_scale = 0.0
                convergence_curve = V2_DEFAULTS.convergence_curve_default
                latched_near_tail_count = 0
                latched_near_tail_coverage = 0.0
                latched_dense_near_weight = 0.0
                latched_effective_near_log_tau = config.near_log_tau
            container_scale = 1.0
            canonical = np.zeros(shape, dtype="<f4")
            vertical = np.zeros(shape, dtype="<f4")
            parallax = np.zeros(shape, dtype="<f4")
        else:
            assert candidate is not None
            if not camera_valid or confirmed_cut:
                active_center = candidate.center
                active_scale = config.raw_coordinate_scale
                convergence_curve = V2_DEFAULTS.convergence_curve_default
                latched_near_tail_count = candidate.near_tail_count
                latched_near_tail_coverage = candidate.near_tail_coverage
                latched_dense_near_weight = candidate.dense_near_weight
                latched_effective_near_log_tau = (
                    config.near_log_tau + latched_dense_near_weight *
                    (config.near_log_tau_dense - config.near_log_tau))
                calibration_revision = _exact_counter_increment(calibration_revision)
                camera_valid = True

            container_scale = _frame_container_scale(
                candidate.raw_min, candidate.raw_max,
                active_center, active_scale, convergence_curve, config)
            canonical64, curved = curve_relative_coordinate(
                raw, active_center, active_scale, config,
                convergence_curve=convergence_curve,
                near_log_tau=latched_effective_near_log_tau)
            effective_gain = config.parallax_gain * container_scale
            candidate_parallax = curved * effective_gain
            vertical = vertical_lipschitz_majorant(
                candidate_parallax, config.max_vertical_shear / shape[1])
            final = horizontal_lipschitz_majorant(
                vertical, config.max_horizontal_slope / shape[1])
            vertical_correction = vertical - candidate_parallax
            correction = final - candidate_parallax
            vertical_tolerance = max(
                1.0e-12, config.max_vertical_shear / shape[1] * 1.0e-9)
            tolerance = max(
                1.0e-12, config.max_horizontal_slope / shape[1] * 1.0e-9)
            pre_limiter_max_abs = float(np.max(np.abs(candidate_parallax)))
            vertical_majorant_raised_fraction = float(
                np.mean(vertical_correction > vertical_tolerance))
            vertical_majorant_max_raise = float(np.max(vertical_correction))
            final_max_abs = float(np.max(np.abs(final)))
            limiter_raised_fraction = float(np.mean(correction > tolerance))
            limiter_max_raise = float(np.max(correction))
            final_horizontal_slope_max = (
                float(np.max(np.abs(np.diff(final, axis=1)))) * shape[1]
                if shape[1] > 1 else 0.0
            )
            final_vertical_shear_max = (
                float(np.max(np.abs(np.diff(final, axis=0)))) * shape[1]
                if shape[0] > 1 else 0.0
            )
            canonical = canonical64.astype("<f4")
            vertical = vertical.astype("<f4")
            parallax = final.astype("<f4")

        encoded = encode_direct_parallax(parallax)
        order_sha = _field_sha256(canonical)
        vertical_majorant_sha = _field_sha256(vertical)
        parallax_sha = _field_sha256(encoded)
        effective_gain = (config.parallax_gain * container_scale
                          if frame_valid else 0.0)
        candidate_center_drift_u = (
            (candidate.center - active_center) / active_scale
            if frame_valid and candidate is not None else 0.0)
        predicted_zero_translation = (
            effective_gain * float(asymmetric_curve(
                np.asarray(candidate_center_drift_u), config,
                near_log_tau=latched_effective_near_log_tau) - convergence_curve)
            if frame_valid else 0.0)
        frame_id_text = f"{frame_id:05d}"
        cut_attribution = (
            "initialization" if index == 0 else cut_source if confirmed_cut else "none"
        )
        row: Dict[str, object] = {
            "frame_id": frame_id_text,
            "input_source_frame_id": frame_id_text,
            "rendered_source_frame_id": frame_id_text,
            "calibration_revision": calibration_revision,
            "confirmed_cut_count": confirmed_cut_count,
            "confirmed_cut": confirmed_cut,
            "cut_attribution": cut_attribution,
            "input_valid": input_valid,
            "frame_valid": frame_valid,
            "camera_valid": camera_valid,
            "collapsed": collapsed,
            "center": active_center if camera_valid else 0.0,
            "inverse_scale": 1.0 / active_scale if camera_valid else 0.0,
            "latched_scale": active_scale if camera_valid else 0.0,
            "convergence_curve": convergence_curve,
            "latched_near_tail_coverage": (
                latched_near_tail_coverage if camera_valid else 0.0),
            "effective_near_log_tau": (
                latched_effective_near_log_tau if camera_valid else config.near_log_tau),
            "latched_near_tail_count": (
                latched_near_tail_count if camera_valid else 0),
            "requested_gain": config.parallax_gain,
            "container_scale": container_scale,
            "effective_gain": effective_gain,
            "observed_mean": candidate.center if candidate is not None else 0.0,
            "observed_std": candidate.observed_std if candidate is not None else 0.0,
            "observed_raw_minimum": candidate.raw_min if candidate is not None else 0.0,
            "observed_raw_maximum": candidate.raw_max if candidate is not None else 0.0,
            "candidate_center_drift_u": candidate_center_drift_u,
            "predicted_zero_translation_source_u": predicted_zero_translation,
            "pre_limiter_max_abs_source_u": pre_limiter_max_abs,
            "vertical_majorant_raised_fraction": vertical_majorant_raised_fraction,
            "vertical_majorant_max_raise_source_u": vertical_majorant_max_raise,
            "final_max_abs_source_u": final_max_abs,
            "limiter_raised_fraction": limiter_raised_fraction,
            "limiter_max_raise_source_u": limiter_max_raise,
            "final_horizontal_slope_max": final_horizontal_slope_max,
            "final_vertical_shear_max": final_vertical_shear_max,
            "order_sha256": order_sha,
            "vertical_majorant_sha256": vertical_majorant_sha,
            "parallax_sha256": parallax_sha,
        }
        canonical_fields.append(canonical)
        parallax_fields.append(parallax)
        rows.append(row)

    trace: Dict[str, object] = {
        "schema": V2_STATE_TRACE_SCHEMA,
        "policy": V2_STATE_TRACE_POLICY,
        "calibration_contract": {
            "file": CONTRACT_PATH.relative_to(Path(__file__).resolve().parent).as_posix(),
            "schema": CONTRACT_SCHEMA,
            "sha256": _file_sha256(CONTRACT_PATH),
        },
        "cut_source": cut_source,
        "diagnostic_role": "non-controlling-fixed-scale-camera-audit-v4",
        "diagnostic_method": "frame-moment-proxies-not-matched-pixel-affine-v2",
        "mapping_config": asdict(config),
        "frame_fields": list(V2_STATE_TRACE_FIELDS),
        "frames": rows,
        "producer": {
            "authority": "numpy-reference-comparison-only-v3",
            "numpy_role": "comparison-only-not-render-authority",
        },
    }
    validate_v2_state_trace(trace, ids)
    return ExactSequenceResult(
        frame_ids=ids,
        canonical_fields=tuple(canonical_fields),
        parallax_fields=tuple(parallax_fields),
        state_trace=trace,
    )


def _gain_limits(
        raw_depth: np.ndarray,
        state: CoordinateState,
        config: MappingV2Config,
        max_collar_texels: Optional[int]) -> Tuple[float, float, float, float]:
    """Return absolute, collar, and combined gain limits plus unit-gain edge step."""

    raw = np.asarray(raw_depth, dtype=np.float64)
    _, unit = curve_relative_coordinate(
        raw, state.center, state.scale, config,
        convergence_curve=state.convergence_curve,
        near_log_tau=state.effective_near_log_tau)
    # The hard container uses the conservative base near curve, independently of the scene's
    # effective curve. This preserves the original representation bound under adaptation.
    _, container_unit = curve_relative_coordinate(
        raw, state.center, state.scale, config,
        convergence_curve=state.convergence_curve,
        near_log_tau=config.near_log_tau)
    _, absolute_unit = container_scale_for_curve_range(
        float(np.min(container_unit)),
        float(np.max(container_unit)),
        config,
        gain=1.0,
    )
    absolute_limit = (
        config.direct_container_limit / absolute_unit if absolute_unit > 0.0 else math.inf)
    if raw.shape[1] < 2:
        adjacent_unit = 0.0
    else:
        adjacent_unit = float(np.max(np.abs(np.diff(unit, axis=1))))
    max_step = config.max_horizontal_slope / raw.shape[1]
    max_vertical_step = config.max_vertical_shear / raw.shape[1]
    collar_limit = math.inf
    if max_collar_texels is not None and adjacent_unit > 0.0:
        collar_limit = max_collar_texels * max_step / adjacent_unit
    return absolute_limit, collar_limit, min(absolute_limit, collar_limit), adjacent_unit


def _limiter_burden(
        raw_depth: np.ndarray,
        state: CoordinateState,
        gain: float,
        config: MappingV2Config) -> Dict[str, float]:
    raw = np.asarray(raw_depth, dtype=np.float64)
    _, unit = curve_relative_coordinate(
        raw, state.center, state.scale, config,
        convergence_curve=state.convergence_curve,
        near_log_tau=state.effective_near_log_tau)
    desired = unit * gain
    max_step = config.max_horizontal_slope / raw.shape[1]
    adjacent = (float(np.max(np.abs(np.diff(desired, axis=1))))
                if raw.shape[1] > 1 else 0.0)
    collar = int(math.ceil(adjacent / max_step - 1.0e-12)) if adjacent > 0.0 else 0
    vertical = vertical_lipschitz_majorant(desired, max_vertical_step)
    limited = horizontal_lipschitz_majorant(vertical, max_step)
    correction = limited - desired
    tolerance = max(1.0e-12, max_step * 1.0e-9)
    limited_min = min(
        float(np.min(limited)),
        -config.far_tau * gain,
    )
    limited_max = float(np.max(limited))
    absolute_max = max(abs(limited_min), abs(limited_max))
    return {
        "gain": gain,
        "parallax_span_source_u": limited_max - limited_min,
        "absolute_parallax_source_u_max": absolute_max,
        "source_u_envelope_feasible": float(
            absolute_max <= config.direct_container_limit + 1.0e-12),
        "estimated_collar_texels": collar,
        "limiter_raised_fraction": float(np.mean(correction > tolerance)),
        "limiter_correction_source_u_p95": float(np.percentile(correction, 95.0)),
        "limiter_correction_source_u_max": float(np.max(correction)),
    }


def evaluate_gain_ownership(
        raw_fields: Sequence[np.ndarray],
        states: Sequence[CoordinateState],
        cuts: Sequence[bool],
        requested_gain: float,
        max_collar_texels: Optional[int],
        config: MappingV2Config) -> Dict[str, object]:
    """Compare immutable requested gain with the recoverable frame-local hard container."""

    fixed_reports: List[Dict[str, float]] = []
    capped_reports: List[Dict[str, float]] = []
    absolute_limits: List[float] = []
    collar_limits: List[float] = []
    instantaneous_limits: List[float] = []
    active_limits: List[float] = []
    absolute_violations = 0
    collar_violations = 0
    for raw, state, cut in zip(raw_fields, states, cuts):
        absolute_limit, collar_limit, safe_limit, _ = _gain_limits(
            raw, state, config, max_collar_texels)
        effective_gain = min(requested_gain, safe_limit)
        fixed = _limiter_burden(raw, state, requested_gain, config)
        capped = _limiter_burden(raw, state, effective_gain, config)
        fixed_reports.append(fixed)
        capped_reports.append(capped)
        absolute_limits.append(absolute_limit)
        collar_limits.append(collar_limit)
        instantaneous_limits.append(safe_limit)
        active_limits.append(safe_limit)
        if requested_gain > absolute_limit:
            absolute_violations += 1
        if math.isfinite(collar_limit) and requested_gain > collar_limit:
            collar_violations += 1

    def burden_summary(values: Sequence[Dict[str, float]]) -> Dict[str, float]:
        return {
            "gain_min": min(value["gain"] for value in values),
            "gain_max": max(value["gain"] for value in values),
            "parallax_span_source_u_max": max(
                value["parallax_span_source_u"] for value in values),
            "absolute_parallax_source_u_max": max(
                value["absolute_parallax_source_u_max"] for value in values),
            "source_u_envelope_violation_frames": sum(
                value["source_u_envelope_feasible"] < 0.5 for value in values),
            "estimated_collar_texels_max": max(
                value["estimated_collar_texels"] for value in values),
            "limiter_raised_fraction_p95": _percentile(
                (value["limiter_raised_fraction"] for value in values), 95.0),
            "limiter_correction_source_u_p95": _percentile(
                (value["limiter_correction_source_u_p95"] for value in values), 95.0),
            "limiter_correction_source_u_max": max(
                value["limiter_correction_source_u_max"] for value in values),
        }

    finite_absolute = [value for value in absolute_limits if math.isfinite(value)]
    finite_collar = [value for value in collar_limits if math.isfinite(value)]
    finite_safe = [value for value in instantaneous_limits if math.isfinite(value)]
    return {
        "requested_gain": requested_gain,
        "max_collar_texels": max_collar_texels,
        "fixed_requested": burden_summary(fixed_reports),
        "requested_capped_by_frame_container": burden_summary(capped_reports),
        "fixed_absolute_violation_frames": absolute_violations,
        "fixed_collar_violation_frames": collar_violations,
        "absolute_safe_gain_range": [min(finite_absolute, default=math.inf),
                                     max(finite_absolute, default=math.inf)],
        "collar_safe_gain_range": [min(finite_collar, default=math.inf),
                                   max(finite_collar, default=math.inf)],
        "instantaneous_max_safe_gain_range": [min(finite_safe, default=math.inf),
                                              max(finite_safe, default=math.inf)],
        "frame_safe_gain_min": min(active_limits),
        "frame_safe_gain_max": max(active_limits),
        "note": ("requested gain is immutable; only the current frame's hard representation "
                 "container may reduce effective gain"),
    }


def _percentile(values: Iterable[float], percentile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.percentile(array, percentile))


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
    state_path = clip_dir / "subject_state.json"
    if not state_path.exists():
        return cuts, counts, "first-frame-only"
    trace = subject_state_contract.load_trace(str(state_path))
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
    return cuts, counts, "subject-state-hard-cut-generation"


def simulate_source_timeline(
        candidates: Sequence[MomentCandidate],
        cuts: Sequence[bool],
        source_fps: float,
        temporal: TemporalConfig,
        mapping: MappingV2Config = MappingV2Config()) -> Tuple[List[CoordinateState], List[bool]]:
    controller = TemporalCoordinateController(temporal, mapping)
    states: List[CoordinateState] = []
    relatches: List[bool] = []
    for index, candidate in enumerate(candidates):
        update = controller.update(candidate, index / source_fps, bool(cuts[index]))
        states.append(update.state)
        relatches.append(update.relatched)
    return states, relatches


def simulate_stream_schedule(
        candidates: Sequence[MomentCandidate],
        cuts: Sequence[bool],
        source_fps: float,
        stream_fps: float,
        depth_reuse: int,
        temporal: TemporalConfig,
        mapping: MappingV2Config = MappingV2Config()) -> Tuple[List[CoordinateState], List[float]]:
    """Sample controller state at source-frame times under a stream/depth schedule."""

    if source_fps <= 0.0 or stream_fps <= 0.0 or depth_reuse < 1:
        raise ValueError("FPS values must be positive and depth_reuse must be >= 1")
    if not candidates:
        return [], []
    duration = (len(candidates) - 1) / source_fps
    stream_count = int(math.ceil(duration * stream_fps)) + depth_reuse + 1
    controller = TemporalCoordinateController(temporal, mapping)
    update_times: List[float] = []
    update_states: List[CoordinateState] = []
    cut_delays: List[float] = []
    last_source_index = -1
    for stream_index in range(stream_count):
        if stream_index % depth_reuse:
            continue
        time_seconds = stream_index / stream_fps
        if time_seconds > duration + depth_reuse / stream_fps + 1.0e-12:
            break
        source_index = min(int(math.floor(time_seconds * source_fps + 1.0e-9)),
                           len(candidates) - 1)
        crossed = [index for index in range(last_source_index + 1, source_index + 1)
                   if cuts[index]]
        cut = bool(crossed)
        if cut:
            cut_time = crossed[-1] / source_fps
            cut_delays.append(max(0.0, time_seconds - cut_time))
        update = controller.update(candidates[source_index], time_seconds, cut)
        update_times.append(time_seconds)
        update_states.append(update.state)
        last_source_index = max(last_source_index, source_index)

    sampled: List[CoordinateState] = []
    for source_index in range(len(candidates)):
        time_seconds = source_index / source_fps
        update_index = int(np.searchsorted(update_times, time_seconds + 1.0e-12, side="right")) - 1
        sampled.append(update_states[max(0, update_index)])
    return sampled, cut_delays


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
    if (not isinstance(clips, dict) or not clips or
            not isinstance(raw_manifests, dict) or set(raw_manifests) != set(clips)):
        raise ValueError(
            f"{run_root}: schema-{whole_clip_raw_contract.EVALUATOR_SCHEMA} results do not "
            "bind raw artifacts for the exact scored clip set")
    clip_calibrations: Dict[str, object] = {}
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
        if calibration is None:
            expected_status = "abstain-unsupported-model-contract"
            expected_calibration_id = None
        elif ((raw_shape["width"], raw_shape["height"]) not in
              calibration.calibrated_input_shapes):
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
    state_path = clip_dir / "subject_state.json"
    if not state_path.exists():
        mismatched["subject_state"] = []
    else:
        state_ids = set(subject_state_contract.load_trace(str(state_path)))
        if state_ids != expected_ids:
            mismatched["subject_state"] = sorted(state_ids)
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
    if (clip_calibration is None or
            (shape[1], shape[0]) not in clip_calibration.calibrated_input_shapes or
            run_model["calibration_id"] != clip_calibration.calibration_id):
        raise ValueError(
            f"{clip_dir}: calibrated raw identity is inconsistent with the run mapping")
    raw_fields = [
        {"frame_id": row["frame_id"], "sha256": row["sha256"]}
        for row in manifest_frames
    ]
    committed_meta = Path(__file__).resolve().parent / "clips" / clip_dir.name / "meta.json"
    cut_control_evidence = {
        "kind": "subject-state-hard-cut-generation",
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
        "model_hash_authority": "schema-36-run-level-results-json",
        "input_shape_authority": "schema-36-per-clip-raw-manifest",
        "raw_hash_authority": {
            "source": "schema-36-run-level-results-json",
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


def _state_field_delta(
        raw: np.ndarray,
        left: CoordinateState,
        right: CoordinateState,
        mapping: MappingV2Config) -> float:
    return float(np.percentile(np.abs(
        parallax_for_state(raw, left, mapping) - parallax_for_state(raw, right, mapping)
    ), 95.0))


def evaluate_clip(
        clip_dir: Path,
        source_fps: float,
        sample_stride: int,
        temporal_configs: Sequence[TemporalConfig],
        schedules: Sequence[Tuple[float, int]],
        requested_gain: float,
        max_collar_texels: Optional[int],
        mapping: MappingV2Config = MappingV2Config(),
        run_model: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    height, width = _load_shape(clip_dir)
    files = _raw_files(clip_dir)
    if not files:
        raise ValueError(f"{clip_dir}: no raw_<frame>.f32 files")
    raw_fields: List[np.ndarray] = []
    full_raw_fields: List[np.ndarray] = []
    candidates: List[MomentCandidate] = []
    expected_count = height * width
    for _, path in files:
        raw = np.fromfile(path, dtype="<f4")
        if raw.size != expected_count:
            raise ValueError(f"{path}: expected {expected_count} floats, got {raw.size}")
        full = raw.reshape(height, width)
        sampled = full[::sample_stride, ::sample_stride].astype(np.float64)
        full_raw_fields.append(full)
        raw_fields.append(sampled)
        # Candidate moments must use the complete field; sampling would change the calibration.
        candidates.append(moment_candidate(full, mapping))
    frame_ids = [frame_id for frame_id, _ in files]
    expected_ids = list(range(frame_ids[0], frame_ids[0] + len(frame_ids)))
    if frame_ids != expected_ids:
        raise ValueError(
            f"{clip_dir}: sparse frame IDs have no authenticated timestamp contract; "
            f"expected {expected_ids}, got {frame_ids}")
    cuts, _cut_counts, cut_source = _load_cut_indices(clip_dir, frame_ids)
    input_contract = (_clip_sequence_input_contract(
        clip_dir, frame_ids, (height, width), run_model)
        if run_model is not None else None)

    policies: Dict[str, object] = {}
    states_by_policy: Dict[str, List[CoordinateState]] = {}
    for temporal in temporal_configs:
        states, relatches = simulate_source_timeline(
            candidates, cuts, source_fps, temporal, mapping)
        states_by_policy[temporal.policy] = states
        oracle_states = [
            CoordinateState(candidate.center, candidate.scale,
                            valid=not candidate.collapsed,
                            near_tail_count=candidate.near_tail_count,
                            near_tail_coverage=candidate.near_tail_coverage,
                            dense_near_weight=candidate.dense_near_weight,
                            effective_near_log_tau=candidate.effective_near_log_tau)
            for candidate in candidates
        ]
        fields = map_timeline_flat_on_unusable(
            raw_fields, candidates, states, cuts, mapping)
        center_errors = [
            (0.0 if candidate.collapsed or not state.valid else
             abs(state.center - candidate.center) / state.scale)
            for state, candidate in zip(states, candidates)
        ]
        scale_errors = [
            (0.0 if candidate.collapsed or not state.valid else
             abs(math.log(state.scale / candidate.scale)))
            for state, candidate in zip(states, candidates)
        ]
        oracle_errors = [
            (0.0 if candidate.collapsed else
             _state_field_delta(raw, state, oracle, mapping))
            for raw, state, oracle, candidate in
            zip(raw_fields, states, oracle_states, candidates)
        ]
        state_center_steps: List[float] = []
        state_log_scale_steps: List[float] = []
        state_field_steps: List[float] = []
        temporal_field_steps: List[float] = []
        static_field_steps: List[float] = []
        noncut_state_field_steps: List[float] = []
        cut_state_field_steps: List[float] = []
        for index in range(1, len(states)):
            prior = states[index - 1]
            current = states[index]
            if prior.valid and current.valid:
                state_center_steps.append(abs(current.center - prior.center) / prior.scale)
                state_log_scale_steps.append(abs(math.log(current.scale / prior.scale)))
            state_delta = (0.0 if (candidates[index].collapsed or
                                  not prior.valid or not current.valid) else
                           _state_field_delta(raw_fields[index], prior, current, mapping))
            state_field_steps.append(state_delta)
            (cut_state_field_steps if cuts[index] else noncut_state_field_steps).append(state_delta)
            temporal_delta = np.abs(fields[index] - fields[index - 1])
            temporal_field_steps.append(float(np.percentile(temporal_delta, 95.0)))
            # A per-frame canonical oracle removes raw model scale/offset motion.  Pixels whose
            # oracle coordinate barely moved form a useful "geometry-static" mask.
            if not candidates[index - 1].collapsed and not candidates[index].collapsed:
                previous_u = ((raw_fields[index - 1] - candidates[index - 1].center) /
                              candidates[index - 1].scale)
                current_u = ((raw_fields[index] - candidates[index].center) /
                             candidates[index].scale)
                static = np.abs(current_u - previous_u) <= 0.01
                if np.any(static):
                    static_field_steps.append(
                        float(np.percentile(temporal_delta[static], 95.0)))

        shot_indices = np.cumsum(np.asarray(cuts, dtype=np.int64))
        post_settle_center_drift: List[float] = []
        post_settle_scale_drift: List[float] = []
        for shot in np.unique(shot_indices):
            indices = np.flatnonzero(shot_indices == shot)
            if indices.size == 0:
                continue
            first = int(indices[0])
            settled = indices[(indices - first) / source_fps >= temporal.aggregate_seconds]
            if settled.size < 2:
                continue
            shot_states = [states[int(index)] for index in settled
                           if states[int(index)].valid]
            if len(shot_states) < 2:
                continue
            reference_scale = shot_states[0].scale
            centers = [state.center for state in shot_states]
            scales = [math.log(state.scale) for state in shot_states]
            post_settle_center_drift.append((max(centers) - min(centers)) / reference_scale)
            post_settle_scale_drift.append(max(scales) - min(scales))

        schedule_reports: Dict[str, object] = {}
        for stream_fps, depth_reuse in schedules:
            scheduled, cut_delays = simulate_stream_schedule(
                candidates, cuts, source_fps, stream_fps, depth_reuse, temporal, mapping)
            schedule_center = [
                (abs(left.center - right.center) / left.scale
                 if left.valid and right.valid else 0.0)
                for left, right in zip(states, scheduled)
            ]
            schedule_scale = [
                (abs(math.log(left.scale / right.scale))
                 if left.valid and right.valid else 0.0)
                for left, right in zip(states, scheduled)
            ]
            schedule_fields = [
                (0.0 if candidate.collapsed else
                 _state_field_delta(raw, left, right, mapping))
                for raw, left, right, candidate in
                zip(raw_fields, states, scheduled, candidates)
            ]
            schedule_reports[f"{stream_fps:g}fps_reuse{depth_reuse}"] = {
                "center_error_norm_max": max(schedule_center, default=0.0),
                "log_scale_error_max": max(schedule_scale, default=0.0),
                "parallax_error_source_u_p95_max": max(schedule_fields, default=0.0),
                "center_error_norm_excluding_cut_frames_max": max(
                    (value for index, value in enumerate(schedule_center) if not cuts[index]),
                    default=0.0),
                "log_scale_error_excluding_cut_frames_max": max(
                    (value for index, value in enumerate(schedule_scale) if not cuts[index]),
                    default=0.0),
                "parallax_error_source_u_p95_excluding_cut_frames_max": max(
                    (value for index, value in enumerate(schedule_fields) if not cuts[index]),
                    default=0.0),
                "cut_delay_ms_max": 1000.0 * max(cut_delays, default=0.0),
            }

        policies[temporal.policy] = {
            "center_candidate_error_norm_p95": _percentile(center_errors, 95.0),
            "log_scale_candidate_error_p95": _percentile(scale_errors, 95.0),
            "oracle_parallax_error_source_u_p95": _percentile(oracle_errors, 95.0),
            "state_center_step_norm_max": max(state_center_steps, default=0.0),
            "state_log_scale_step_max": max(state_log_scale_steps, default=0.0),
            "state_parallax_step_source_u_p95_max": max(state_field_steps, default=0.0),
            "noncut_state_parallax_step_source_u_p95_max": max(
                noncut_state_field_steps, default=0.0),
            "cut_state_parallax_step_source_u_p95_max": max(
                cut_state_field_steps, default=0.0),
            "observed_temporal_parallax_source_u_p95": _percentile(
                temporal_field_steps, 95.0),
            "geometry_static_parallax_source_u_p95": _percentile(
                static_field_steps, 95.0),
            "post_settle_center_drift_norm_max": max(post_settle_center_drift, default=0.0),
            "post_settle_log_scale_drift_max": max(post_settle_scale_drift, default=0.0),
            "relatch_frame_ids": [frame_ids[index] for index, value in enumerate(relatches)
                                  if value],
            "convergence_curve_range": [
                min(state.convergence_curve for state in states),
                max(state.convergence_curve for state in states),
            ],
            "schedules": schedule_reports,
        }

    gain_ownership = evaluate_gain_ownership(
        full_raw_fields,
        states_by_policy["first"],
        cuts,
        requested_gain,
        max_collar_texels,
        mapping,
    )
    return {
        "clip": clip_dir.name,
        "frames": len(files),
        "shape": [height, width],
        "sample_stride": sample_stride,
        "cut_frame_ids": [frame_ids[index] for index, value in enumerate(cuts) if value],
        "cut_source": cut_source,
        "sequence_input_contract": input_contract,
        "candidate": {
            "center_range": [min(value.center for value in candidates),
                             max(value.center for value in candidates)],
            "scale_range": [min(value.scale for value in candidates),
                            max(value.scale for value in candidates)],
            "collapsed_frames": sum(value.collapsed for value in candidates),
        },
        "gain_ownership_on_production_first_coordinate": gain_ownership,
        "policies": policies,
    }


def summarize(clips: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for policy in POLICY_STUDY_POLICIES:
        entries = [clip["policies"][policy] for clip in clips]  # type: ignore[index]
        scalar_keys = [
            "center_candidate_error_norm_p95",
            "log_scale_candidate_error_p95",
            "oracle_parallax_error_source_u_p95",
            "state_parallax_step_source_u_p95_max",
            "noncut_state_parallax_step_source_u_p95_max",
            "cut_state_parallax_step_source_u_p95_max",
            "observed_temporal_parallax_source_u_p95",
            "geometry_static_parallax_source_u_p95",
            "post_settle_center_drift_norm_max",
            "post_settle_log_scale_drift_max",
        ]
        summary[policy] = {
            key: {
                "median_clip": _percentile((float(entry[key]) for entry in entries), 50.0),
                "p95_clip": _percentile((float(entry[key]) for entry in entries), 95.0),
                "max_clip": max((float(entry[key]) for entry in entries), default=0.0),
            }
            for key in scalar_keys
        }
        schedules: Dict[str, object] = {}
        schedule_names = sorted({name for entry in entries
                                 for name in entry["schedules"]})  # type: ignore[index]
        for schedule_name in schedule_names:
            schedule_entries = [entry["schedules"][schedule_name]  # type: ignore[index]
                                for entry in entries]
            schedules[schedule_name] = {
                key: max(float(item[key]) for item in schedule_entries)
                for key in schedule_entries[0]
            }
        summary[policy]["schedules_worst_clip"] = schedules  # type: ignore[index]
    gain_entries = [clip["gain_ownership_on_production_first_coordinate"]  # type: ignore[index]
                    for clip in clips]
    if gain_entries:
        summary["gain_ownership"] = {
            "fixed_absolute_violation_frames": sum(
                int(entry["fixed_absolute_violation_frames"]) for entry in gain_entries),
            "fixed_collar_violation_frames": sum(
                int(entry["fixed_collar_violation_frames"]) for entry in gain_entries),
            "fixed_requested_collar_texels_max": max(
                float(entry["fixed_requested"]["estimated_collar_texels_max"])
                for entry in gain_entries),
            "safe_capped_collar_texels_max": max(
                float(entry["requested_capped_by_shot_safety"]
                      ["estimated_collar_texels_max"])
                for entry in gain_entries),
            "fixed_requested_limiter_correction_source_u_max": max(
                float(entry["fixed_requested"]["limiter_correction_source_u_max"])
                for entry in gain_entries),
            "safe_capped_limiter_correction_source_u_max": max(
                float(entry["requested_capped_by_shot_safety"]
                      ["limiter_correction_source_u_max"])
                for entry in gain_entries),
            "fixed_absolute_parallax_source_u_max": max(
                float(entry["fixed_requested"]["absolute_parallax_source_u_max"])
                for entry in gain_entries),
            "active_shot_safe_gain_min": min(
                float(entry["active_shot_safe_gain_min"]) for entry in gain_entries),
            "active_shot_safe_gain_max": max(
                float(entry["active_shot_safe_gain_max"]) for entry in gain_entries),
        }
    return summary


def _parse_schedule(value: str) -> Tuple[float, int]:
    try:
        fps_text, reuse_text = value.split(":", 1)
        fps = float(fps_text)
        reuse = int(reuse_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("schedule must be FPS:DEPTH_REUSE") from exc
    if fps <= 0.0 or reuse < 1:
        raise argparse.ArgumentTypeError("schedule FPS must be positive and reuse >= 1")
    return fps, reuse


def _parse_optional_positive_int(value: str) -> Optional[int]:
    if value.strip().lower() in {"none", "off", "disabled"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer or 'none'") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer or 'none'")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path,
                        help="sbs_eval run containing clip directories and raw_<id>.f32")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--clip", action="append", dest="clips",
                        help="clip name (repeatable); default is every directory with raw files")
    parser.add_argument("--source-fps", type=float, default=30.0,
                        help="artifact timeline FPS assumption (default: 30)")
    parser.add_argument("--sample-stride", type=int, default=4,
                        help="spatial stride for temporal parallax diagnostics (default: 4)")
    parser.add_argument("--schedule", action="append", type=_parse_schedule,
                        default=[], help="invariance schedule FPS:DEPTH_REUSE (repeatable)")
    parser.add_argument("--aggregate-seconds", type=float, default=0.25)
    parser.add_argument("--slow-tau-seconds", type=float, default=1.0)
    parser.add_argument("--pop-strength", type=float, default=MappingV2Config().pop_strength,
                        help="artistic pop strength (default: mapping contract value)")
    parser.add_argument("--gain-per-pop", type=float, default=MappingV2Config().gain_per_pop,
                        help="one-eye source-U gain per pop unit (default: mapping contract value)")
    parser.add_argument(
        "--experimental-max-collar-texels", type=_parse_optional_positive_int, default=None,
        help=("diagnostic-only hypothetical collar cap; never part of selected geometry "
              "(default: none)"))
    args = parser.parse_args(argv)
    if (args.source_fps <= 0.0 or args.sample_stride < 1 or args.pop_strength <= 0.0 or
            args.gain_per_pop <= 0.0):
        parser.error("FPS/pop/gain must be positive and sample stride must be >= 1")
    try:
        run_model = _load_run_model_contract(args.run_root)
    except ValueError as exc:
        parser.error(str(exc))
    mapping = MappingV2Config(
        raw_coordinate_scale=float(run_model["raw_coordinate_scale"]),
        pop_strength=args.pop_strength,
        gain_per_pop=args.gain_per_pop,
    )
    requested_gain = mapping.parallax_gain
    schedules = args.schedule or [(72.0, 1), (90.0, 1), (72.0, 2), (90.0, 2)]
    temporal_configs = [
        TemporalConfig(policy, args.aggregate_seconds, args.slow_tau_seconds)
        for policy in POLICY_STUDY_POLICIES
    ]
    if args.clips:
        clip_dirs = [args.run_root / clip for clip in args.clips]
    else:
        clip_dirs = sorted(path for path in args.run_root.iterdir()
                           if path.is_dir() and _raw_files(path))
    reports: List[Dict[str, object]] = []
    for clip_dir in clip_dirs:
        print(f"temporal depth coordinate: {clip_dir.name}", flush=True)
        reports.append(evaluate_clip(
            clip_dir, args.source_fps, args.sample_stride,
            temporal_configs, schedules, requested_gain,
            args.experimental_max_collar_texels, mapping, run_model))
    payload = {
        "schema": 1,
        "description": "pre-reprojection temporal evidence for host SBS depth coordinate v2",
        "run_root": str(args.run_root.resolve()),
        "source_fps_assumption": args.source_fps,
        "mapping": asdict(mapping),
        "temporal": [asdict(value) for value in temporal_configs],
        "schedules": [{"stream_fps": fps, "depth_reuse": reuse}
                      for fps, reuse in schedules],
        "gain_evidence": {
            "pop_strength": mapping.pop_strength,
            "gain_per_pop": mapping.gain_per_pop,
            "requested_gain": requested_gain,
            "experimental_max_collar_texels": args.experimental_max_collar_texels,
            "collar_role": "diagnostic-only; never selected geometry",
            "policy": ("requested gain is immutable; the hard representation container is "
                       "derived independently for every usable frame and can recover"),
            "zero_plane_order": (
                "curve-space convergence is separately latched and currently exactly zero"),
        },
        "clips": reports,
        "summary": summarize(reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
