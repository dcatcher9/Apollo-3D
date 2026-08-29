#!/usr/bin/env python3
"""Compare coarse- versus refined-driven V2 camera statistics on one refined field.

This is an evaluation-only still-frame isolation.  Both branches map the same authenticated
ZipDepth-convex-2x ``refined_<frame>.f32`` geometry.  The control acquires its camera and collapse
decision from the paired coarse DAV2 ``raw_<frame>.f32`` field, matching the pre-single-high split
path; the treatment acquires those two values from the refined field.  Consequently, every reported
parallax delta is caused by moving camera/statistics ownership, not by changing geometry.

The oracle intentionally excludes scene latching/cuts, temporal reuse, subtitle conditioning,
reprojection, and live GPU timing.  It cannot make temporal or image-quality
claims.  It uses the production V2 curve, pointwise container, and two-axis spatial limiters from
``depth_mapping_v2`` so the remaining comparison is small and reviewable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

try:
    from . import prod_zipdepth_convex2x_diagnostics_contract as artifact_contract
    from .depth_mapping_v2 import (
        MappingV2Config,
        calibrate_coordinate,
        curve_relative_coordinate,
        horizontal_lipschitz_majorant,
        pointwise_soft_container,
        select_scene_coordinate,
        vertical_lipschitz_majorant,
        vertical_lipschitz_minorant,
        vertical_share_coefficients,
    )
except ImportError:  # Direct execution from tools/sbsbench.
    import prod_zipdepth_convex2x_diagnostics_contract as artifact_contract  # type: ignore
    from depth_mapping_v2 import (  # type: ignore
        MappingV2Config,
        calibrate_coordinate,
        curve_relative_coordinate,
        horizontal_lipschitz_majorant,
        pointwise_soft_container,
        select_scene_coordinate,
        vertical_lipschitz_majorant,
        vertical_lipschitz_minorant,
        vertical_share_coefficients,
    )


REPORT_SCHEMA = 1
COMPARISON = "same-refined-geometry-coarse-vs-refined-camera-statistics-v1"
_FRAME_FILE = re.compile(r"raw_(\d+)\.f32$")
_DEFAULT_DISPLAY_WIDTHS = (1920, 3840)


@dataclass(frozen=True)
class CameraMappedField:
    """One high-resolution field mapped from an explicitly selected statistics source."""

    center: float
    observed_std: float
    collapsed: bool
    pre_spatial: np.ndarray
    final: np.ndarray


def _require_field(name: str, value: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite, non-empty real 2D field")
    try:
        field = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite, non-empty real 2D field") from exc
    if field.ndim != 2 or field.size == 0 or not np.isfinite(field).all():
        raise ValueError(f"{name} must be a finite, non-empty real 2D field")
    return field


def map_refined_with_statistics(
        refined_depth: np.ndarray,
        statistics_source: np.ndarray,
        config: MappingV2Config = MappingV2Config(),
) -> CameraMappedField:
    """Map refined geometry while sourcing only camera validity/center from another field."""

    refined = _require_field("refined_depth", refined_depth)
    statistics = _require_field("statistics_source", statistics_source)
    calibration = calibrate_coordinate(statistics, config)
    selection = select_scene_coordinate(statistics, config)
    if calibration.collapsed:
        zero = np.zeros(refined.shape, dtype=np.float32)
        return CameraMappedField(
            center=selection.selected_center,
            observed_std=calibration.observed_std,
            collapsed=True,
            pre_spatial=zero.copy(),
            final=zero,
        )

    _, curved = curve_relative_coordinate(
        refined,
        selection.selected_center,
        config.raw_coordinate_scale,
        config,
        convergence_curve=selection.convergence_curve,
    )
    pre_spatial = pointwise_soft_container(
        curved * config.parallax_gain, config.direct_container_limit)
    width = refined.shape[1]
    vertical_majorant = vertical_lipschitz_majorant(
        pre_spatial, config.max_vertical_shear / width)
    vertical_minorant = vertical_lipschitz_minorant(
        pre_spatial, config.max_vertical_shear / width)
    majorant_share, minorant_share = vertical_share_coefficients(
        config.vertical_majorant_share)
    vertical = majorant_share * vertical_majorant + minorant_share * vertical_minorant
    final = horizontal_lipschitz_majorant(
        vertical, config.max_horizontal_slope / width)

    horizontal_slope = (
        float(np.max(np.abs(np.diff(final, axis=1)))) * width
        if width > 1 else 0.0)
    vertical_shear = (
        float(np.max(np.abs(np.diff(final, axis=0)))) * width
        if refined.shape[0] > 1 else 0.0)
    tolerance = 1.0e-8
    if horizontal_slope > config.max_horizontal_slope + tolerance:
        raise RuntimeError("high-resolution A/B oracle violated horizontal slope policy")
    if vertical_shear > config.max_vertical_shear + tolerance:
        raise RuntimeError("high-resolution A/B oracle violated vertical shear policy")

    pre32 = pre_spatial.astype(np.float32)
    final32 = final.astype(np.float32)
    if not np.isfinite(pre32).all() or not np.isfinite(final32).all():
        raise ValueError("high-resolution A/B mapping is not finite float32")
    return CameraMappedField(
        center=selection.selected_center,
        observed_std=calibration.observed_std,
        collapsed=False,
        pre_spatial=pre32,
        final=final32,
    )


def _field_statistics(field: np.ndarray) -> Dict[str, float]:
    values = _require_field("depth field", field)
    quantiles = np.quantile(values, (0.01, 0.02, 0.50, 0.98, 0.99))
    return {
        "minimum": float(np.min(values)),
        "p01": float(quantiles[0]),
        "p02": float(quantiles[1]),
        "mean": float(np.mean(values, dtype=np.float64)),
        "p50": float(quantiles[2]),
        "p98": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "maximum": float(np.max(values)),
        "population_std": float(np.std(values, dtype=np.float64)),
    }


def _delta_metrics(
        control: np.ndarray,
        treatment: np.ndarray,
        display_widths: Sequence[int],
) -> Dict[str, Any]:
    control_field = _require_field("control parallax", control)
    treatment_field = _require_field("treatment parallax", treatment)
    if control_field.shape != treatment_field.shape:
        raise ValueError("control and treatment parallax shapes must match")
    delta = treatment_field - control_field
    absolute = np.abs(delta)
    quantiles = np.quantile(absolute, (0.50, 0.90, 0.95, 0.99))
    source_u = {
        "signed_mean": float(np.mean(delta, dtype=np.float64)),
        "mean_absolute": float(np.mean(absolute, dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(delta * delta, dtype=np.float64))),
        "p50_absolute": float(quantiles[0]),
        "p90_absolute": float(quantiles[1]),
        "p95_absolute": float(quantiles[2]),
        "p99_absolute": float(quantiles[3]),
        "maximum_absolute": float(np.max(absolute)),
        "fraction_changed_float32": float(np.mean(control_field != treatment_field)),
    }
    pixels: Dict[str, Dict[str, float]] = {}
    for width in display_widths:
        pixels[str(width)] = {
            "mean_absolute": source_u["mean_absolute"] * width,
            "p95_absolute": source_u["p95_absolute"] * width,
            "p99_absolute": source_u["p99_absolute"] * width,
            "maximum_absolute": source_u["maximum_absolute"] * width,
        }
    return {"source_u": source_u, "equivalent_source_pixels": pixels}


def compare_pair(
        coarse_depth: np.ndarray,
        refined_depth: np.ndarray,
        config: MappingV2Config = MappingV2Config(),
        display_widths: Sequence[int] = _DEFAULT_DISPLAY_WIDTHS,
) -> Dict[str, Any]:
    """Evaluate the still-frame camera/statistics ownership change for one paired artifact."""

    coarse = _require_field("coarse_depth", coarse_depth)
    refined = _require_field("refined_depth", refined_depth)
    if refined.shape != (2 * coarse.shape[0], 2 * coarse.shape[1]):
        raise ValueError(
            "refined_depth must have exactly twice the coarse height and width")
    widths = _validated_display_widths(display_widths)
    control = map_refined_with_statistics(refined, coarse, config)
    treatment = map_refined_with_statistics(refined, refined, config)
    center_delta = treatment.center - control.center
    return {
        "coarse_statistics": _field_statistics(coarse),
        "refined_statistics": _field_statistics(refined),
        "camera": {
            "control_center_raw": control.center,
            "treatment_center_raw": treatment.center,
            "treatment_minus_control_center_raw": center_delta,
            "treatment_minus_control_center_canonical": (
                center_delta / config.raw_coordinate_scale),
            "control_observed_population_std": control.observed_std,
            "treatment_observed_population_std": treatment.observed_std,
            "control_collapsed": control.collapsed,
            "treatment_collapsed": treatment.collapsed,
        },
        "pre_spatial_parallax_delta": _delta_metrics(
            control.pre_spatial, treatment.pre_spatial, widths),
        "post_spatial_parallax_delta": _delta_metrics(
            control.final, treatment.final, widths),
        "control_output_range_source_u": {
            "minimum": float(np.min(control.final)),
            "maximum": float(np.max(control.final)),
        },
        "treatment_output_range_source_u": {
            "minimum": float(np.min(treatment.final)),
            "maximum": float(np.max(treatment.final)),
        },
    }


def _validated_display_widths(values: Iterable[int]) -> Tuple[int, ...]:
    widths = tuple(values)
    if (not widths or any(type(value) is not int or value <= 0 for value in widths) or
            len(set(widths)) != len(widths)):
        raise ValueError("display widths must be unique positive integers")
    return widths


def _discover_frame_ids(artifact_dir: Path) -> Tuple[int, ...]:
    identities = []
    for path in artifact_dir.glob("raw_*.f32"):
        match = _FRAME_FILE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"non-canonical raw artifact identity {path.name!r}")
        frame_id = int(match.group(1))
        if path.name != f"raw_{frame_id:05d}.f32":
            raise ValueError(f"non-canonical raw artifact identity {path.name!r}")
        identities.append(frame_id)
    ids = tuple(sorted(identities))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("artifact directory must contain unique canonical raw frame files")
    return ids


def _json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_tensor(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    try:
        field = np.fromfile(path, dtype="<f4")
    except OSError as exc:
        raise ValueError(f"cannot read authenticated tensor {path}: {exc}") from exc
    if field.size != shape[0] * shape[1]:
        raise ValueError(f"authenticated tensor {path.name} changed size during evaluation")
    field = field.reshape(shape)
    if not np.isfinite(field).all():
        raise ValueError(f"authenticated tensor {path.name} changed during evaluation")
    return field


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty A/B")

    def values(path: Tuple[str, ...]) -> Tuple[float, ...]:
        result = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            result.append(float(value))
        return tuple(result)

    center_raw = values(("camera", "treatment_minus_control_center_raw"))
    center_canonical = values(("camera", "treatment_minus_control_center_canonical"))
    pre_p99 = values(("pre_spatial_parallax_delta", "source_u", "p99_absolute"))
    post_mean = values(("post_spatial_parallax_delta", "source_u", "mean_absolute"))
    post_p95 = values(("post_spatial_parallax_delta", "source_u", "p95_absolute"))
    post_p99 = values(("post_spatial_parallax_delta", "source_u", "p99_absolute"))
    post_max = values(("post_spatial_parallax_delta", "source_u", "maximum_absolute"))
    return {
        "row_count": len(rows),
        "mean_absolute_camera_center_delta_raw": float(np.mean(np.abs(center_raw))),
        "maximum_absolute_camera_center_delta_raw": max(abs(value) for value in center_raw),
        "maximum_absolute_camera_center_delta_canonical": max(
            abs(value) for value in center_canonical),
        "maximum_pre_spatial_p99_absolute_delta_source_u": max(pre_p99),
        "mean_of_post_spatial_mean_absolute_delta_source_u": float(np.mean(post_mean)),
        "maximum_post_spatial_p95_absolute_delta_source_u": max(post_p95),
        "maximum_post_spatial_p99_absolute_delta_source_u": max(post_p99),
        "maximum_post_spatial_absolute_delta_source_u": max(post_max),
        "any_collapse_decision_changed": any(
            row["camera"]["control_collapsed"] != row["camera"]["treatment_collapsed"]
            for row in rows),
    }


def build_report(
        artifact_dir: Path,
        *,
        pop_strength_override: float | None = None,
        display_widths: Sequence[int] = _DEFAULT_DISPLAY_WIDTHS,
) -> Dict[str, Any]:
    """Authenticate a harness and build a deduplicated still/camera A/B report."""

    root = artifact_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"artifact directory does not exist: {root}")
    ids = _discover_frame_ids(root)
    sidecar = _json_object(
        root / artifact_contract.SIDECAR_FILENAME, "fused diagnostic sidecar")
    if sidecar.get("schema") != artifact_contract.LEGACY_SIDECAR_SCHEMA:
        raise ValueError(
            "this retired coarse-vs-refined isolation report requires historical diagnostic "
            "schema 1; active schema 2 has one public high-resolution output and no coarse "
            "tensor to compare")
    expected_runtime = sidecar.get("composite_runtime_provenance")
    if not isinstance(expected_runtime, dict):
        raise ValueError("fused diagnostic sidecar has no composite runtime provenance")
    manifest = artifact_contract.build_manifest(root, ids, expected_runtime)
    contract = _json_object(root / "contract.json", "harness contract")
    contract_pop = contract.get("pop_strength")
    if (not isinstance(contract_pop, (int, float)) or isinstance(contract_pop, bool) or
            not math.isfinite(float(contract_pop)) or float(contract_pop) <= 0.0):
        raise ValueError("harness contract has no finite positive pop_strength")
    pop_strength = (
        float(contract_pop) if pop_strength_override is None
        else float(pop_strength_override))
    if not math.isfinite(pop_strength) or pop_strength <= 0.0:
        raise ValueError("pop strength override must be finite and positive")
    config = replace(MappingV2Config(), pop_strength=pop_strength)
    widths = _validated_display_widths(display_widths)

    shapes = manifest["tensor_shapes"]
    coarse_shape = (shapes["raw"]["height"], shapes["raw"]["width"])
    refined_shape = (shapes["refined"]["height"], shapes["refined"]["width"])
    cached: Dict[Tuple[str, str], Tuple[str, Dict[str, Any]]] = {}
    rows = []
    unique_rows = []
    for record in manifest["frames"]:
        frame_id = record["frame_id"]
        pair_identity = (record["raw"]["sha256"], record["refined"]["sha256"])
        cached_pair = cached.get(pair_identity)
        if cached_pair is None:
            coarse = _load_tensor(root / record["raw"]["file"], coarse_shape)
            refined = _load_tensor(root / record["refined"]["file"], refined_shape)
            comparison = compare_pair(coarse, refined, config, widths)
            cached[pair_identity] = (frame_id, comparison)
            unique_rows.append(comparison)
            duplicate_of = None
        else:
            duplicate_of, comparison = cached_pair
        rows.append({
            "frame_id": frame_id,
            "duplicate_of_frame_id": duplicate_of,
            "raw_sha256": pair_identity[0],
            "refined_sha256": pair_identity[1],
            **comparison,
        })

    return {
        "schema": REPORT_SCHEMA,
        "comparison": COMPARISON,
        "scope": {
            "claim": (
                "still-frame isolation of camera/collapse-statistics ownership while both "
                "branches use the same high-resolution refined geometry"),
            "control": "coarse DAV2 camera/collapse statistics plus refined geometry",
            "treatment": "refined camera/collapse statistics plus refined geometry",
            "included_after_camera": (
                "V2 fixed-scale asymmetric curve, pointwise soft container, vertical "
                "majorant/minorant blend, and horizontal majorant"),
            "excluded": [
                "scene-latched camera state and cut handling",
                "temporal normalization, motion masks, depth reuse, and fallback",
                "subtitle conditioning",
                "live GPU arithmetic, renderer/reprojection, image quality, and latency",
            ],
            "temporal_claim_authorized": False,
            "geometry_quality_claim_authorized": False,
        },
        "artifacts": {
            "directory": str(root),
            "manifest": manifest,
            "frame_count": len(rows),
            "unique_raw_refined_pair_count": len(unique_rows),
            "duplicate_pair_frame_count": len(rows) - len(unique_rows),
            "duplicates_are_not_independent_evidence": len(unique_rows) != len(rows),
        },
        "numeric_runtime": {
            "executable": sys.executable,
            "python_version": sys.version,
            "numpy_version": np.__version__,
        },
        "mapping_config": asdict(config),
        "pop_strength_source": (
            "harness-contract" if pop_strength_override is None else "explicit-cli-override"),
        "display_widths_for_source_u_pixel_equivalents": list(widths),
        "frames": rows,
        "aggregate_all_captured_frames": _aggregate_rows(rows),
        "aggregate_unique_raw_refined_pairs": _aggregate_rows(unique_rows),
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts", type=Path, required=True,
        help="Completed fused diagnostic harness directory")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON report")
    parser.add_argument(
        "--pop-strength", type=float,
        help="Optional research override; default is the authenticated harness contract value")
    parser.add_argument(
        "--display-width", type=int, action="append", dest="display_widths",
        help="Source width for equivalent-pixel deltas (repeatable; default: 1920, 3840)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    widths = (
        tuple(args.display_widths) if args.display_widths is not None
        else _DEFAULT_DISPLAY_WIDTHS)
    report = build_report(
        args.artifacts,
        pop_strength_override=args.pop_strength,
        display_widths=widths,
    )
    _atomic_write_json(args.out, report)
    unique = report["aggregate_unique_raw_refined_pairs"]
    print(json.dumps({
        "report": str(args.out.resolve()),
        "captured_frames": report["artifacts"]["frame_count"],
        "unique_pairs": report["artifacts"]["unique_raw_refined_pair_count"],
        "maximum_camera_center_delta_raw": (
            unique["maximum_absolute_camera_center_delta_raw"]),
        "maximum_post_spatial_p99_delta_source_u": (
            unique["maximum_post_spatial_p99_absolute_delta_source_u"]),
        "maximum_post_spatial_delta_source_u": (
            unique["maximum_post_spatial_absolute_delta_source_u"]),
        "temporal_claim_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
