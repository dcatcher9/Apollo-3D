#!/usr/bin/env python3
"""Strict readers for the diagnostic depth-coordinate-v2 JSON dump files.

The JSON serialization schemas are deliberately independent of the GPU coordinate-contract
schema.  Each document also binds the exact generated contract schema/tag and complete physical
field layout, so a consumer cannot silently reinterpret a same-sized state buffer.
"""

from __future__ import annotations

import math
import struct
from typing import Any, Dict

try:
    from . import depth_coordinate_v2_contract as coordinate_contract
    from . import generate_depth_coordinate_v2_contract as generator
except ImportError:  # Direct script/module loading from tools/sbsbench.
    import depth_coordinate_v2_contract as coordinate_contract  # type: ignore
    import generate_depth_coordinate_v2_contract as generator  # type: ignore


DUMP_MANIFEST_SCHEMA = 10
SHADOW_STATE_DUMP_SCHEMA = 13
SHADOW_FRAME_STATS_DUMP_SCHEMA = 2
LIVE_RENDERER_SOURCE_CLOSURE_SHA256 = (
    "707f3866759e2514f718a7a9dae6ed08e90077f29b86d7f925be4a76a4bc3106"
)
DIAGNOSTIC_SOURCE_CLOSURE_SHA256 = (
    "7893c2464bc03c9bc878acffca29cdc6541de4bb73aa2fa57ac34dc5645aaf1e"
)

_CONTRACT = coordinate_contract.load_contract()
_CONTRACT_TAG = generator.contract_tag(_CONTRACT)
_STATE_FIELDS = tuple(_CONTRACT["shadow_state"]["fields"])
_FRAME_STATS_FIELDS = tuple(_CONTRACT["frame_stats"]["fields"])
_DEFAULTS = coordinate_contract.CALIBRATED_DEFAULTS
_RESERVED_CALIBRATION_REVISION = 0xFFFFFFFF

_STATE_ROOT_KEYS = {
    "schema", "coordinate_contract", "source", "capture", "rendered_output_selected",
    "wire_contract", "units", "constants", "fields", "named_values", "decoded",
    "adaptation_semantics",
}
_STATE_CONSTANT_KEYS = {
    "raw_coordinate_scale", "collapse_abs_epsilon", "far_tau", "near_log_tau",
    "gain_per_pop",
    "reference_pop_strength", "reference_gain_at_reference_pop", "requested_gain",
    "requested_pop_strength", "direct_container_limit", "max_horizontal_slope",
    "max_vertical_shear", "convergence_curve_default",
    "stage_valley_ratio_max",
    "vertical_majorant_share",
}
_DECODED_KEYS = {
    "frame_valid", "camera_valid", "calibration_revision", "confirmed_cut_count", "contract_tag",
    "requested_gain", "requested_pop_strength", "latched_scale",
    "convergence_curve", "container_scale", "effective_gain",
    "camera_center_integrity_bits",
}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _uint32(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{label} must be a uint32")
    return value


def camera_center_integrity_bits(
        center: float, inverse_scale: float, convergence_curve: float, revision: int) -> int:
    """Mirror the authenticated SM5 uint32 checksum over the latched camera center."""

    words = (
        struct.unpack("<I", struct.pack("<f", center))[0],
        struct.unpack("<I", struct.pack("<f", inverse_scale))[0],
        struct.unpack("<I", struct.pack("<f", convergence_curve))[0],
        revision,
    )
    checksum = 0
    for word in words:
        checksum = ((checksum ^ word) * 16777619) & 0xFFFFFFFF
    return checksum


def _calibration_revision(value: Any, label: str) -> int:
    revision = _uint32(value, label)
    if revision == _RESERVED_CALIBRATION_REVISION:
        raise ValueError(f"{label} uses the reserved calibration revision")
    return revision


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return math.isclose(float(left), float(right), rel_tol=1.0e-6, abs_tol=1.0e-8)


def _require_coordinate_binding(value: Any, count_key: str, count: int) -> None:
    shader = coordinate_contract.SHADER_IMPLEMENTATION
    expected = {
        "schema": coordinate_contract.CONTRACT_SCHEMA,
        "tag": _CONTRACT_TAG,
        "source_closure_schema": shader.source_closure_schema,
        "source_compile_flags": shader.source_compile_flags,
        "source_macro_count": shader.source_macro_count,
        "source_closure_sha256": shader.source_closure_sha256,
        count_key: count,
    }
    if value != expected:
        raise ValueError("dump has an unknown depth-coordinate-v2 contract binding")


def validate_shadow_state_document(document: Any) -> Dict[str, Any]:
    """Validate one parsed ``shadow_state.json`` and return its typed named values."""

    if not isinstance(document, dict) or set(document) != _STATE_ROOT_KEYS:
        raise ValueError("shadow_state.json has missing or unknown root fields")
    if document.get("schema") != SHADOW_STATE_DUMP_SCHEMA:
        raise ValueError("shadow_state.json has an unknown serialization schema")
    _require_coordinate_binding(
        document.get("coordinate_contract"), "state_word_count", len(_STATE_FIELDS))
    rendered = document.get("rendered_output_selected")
    expected_wire = (
        "authenticated live Host-SBS renderer input; not a client wire contract"
        if rendered is True else
        "experimental diagnostic shadow; not selected by the renderer or client"
    )
    if (document.get("source") != _CONTRACT["shadow_state"]["source"] or
            document.get("capture") != _CONTRACT["shadow_state"]["capture"] or
            not isinstance(rendered, bool) or
            document.get("wire_contract") != expected_wire):
        raise ValueError("shadow_state.json has unknown capture semantics")
    if document.get("units") != {
            "coordinate": "dimensionless canonical coordinate derived from raw depth",
            "gain": "one-eye source-U per curve unit",
            "parallax": "signed one-eye source-U"}:
        raise ValueError("shadow_state.json has unknown units")
    if document.get("adaptation_semantics") != {
            "coordinate":
                "immediate-first-usable-center-latched-until-cut-fixed-authenticated-scale-retained-across-unusable",
            "convergence_curve":
                "selected-upper-valley-or-mean-center-is-zero-plane",
            "requested_gain": "immutable-cfg-pop-strength",
            "container_scale":
                "frame-local-hard-direct-parallax-attenuation-recoverable-next-frame",
            "near_curve":
                "fixed-contract-logarithmic-tau-independent-of-content-occupancy",
            "spatial_conditioner":
                "fixed-75pct-vertical-majorant-share-then-horizontal-majorant"}:
        raise ValueError("shadow_state.json has unknown adaptation semantics")

    constants = document.get("constants")
    if not isinstance(constants, dict) or set(constants) != _STATE_CONSTANT_KEYS:
        raise ValueError("shadow_state.json has an invalid constants object")
    for key in _STATE_CONSTANT_KEYS:
        _finite_number(constants[key], f"shadow_state.json constants.{key}")
    expected_defaults = {
        "collapse_abs_epsilon": _DEFAULTS.collapse_abs_epsilon,
        "far_tau": _DEFAULTS.far_tau,
        "near_log_tau": _DEFAULTS.near_log_tau,
        "gain_per_pop": _DEFAULTS.gain_per_pop,
        "reference_pop_strength": _DEFAULTS.reference_pop_strength,
        "reference_gain_at_reference_pop":
            _DEFAULTS.gain_per_pop * _DEFAULTS.reference_pop_strength,
        "direct_container_limit": _DEFAULTS.direct_container_limit,
        "max_horizontal_slope": _DEFAULTS.max_horizontal_slope,
        "max_vertical_shear": _DEFAULTS.max_vertical_shear,
        "vertical_majorant_share": _DEFAULTS.vertical_majorant_share,
        "convergence_curve_default": _DEFAULTS.convergence_curve_default,
        "stage_valley_ratio_max": _DEFAULTS.stage_valley_ratio_max,
    }
    if any(not _same_number(constants.get(key), value)
           for key, value in expected_defaults.items()):
        raise ValueError("shadow_state.json constants disagree with the generated contract")
    if (_finite_number(constants["raw_coordinate_scale"], "raw_coordinate_scale") <= 0.0 or
            _finite_number(constants["requested_gain"], "requested_gain") < 0.0 or
            _finite_number(constants["requested_pop_strength"],
                           "requested_pop_strength") < 0.0):
        raise ValueError("shadow_state.json has invalid runtime constants")
    if not _same_number(
            constants["requested_gain"],
            float(constants["requested_pop_strength"]) * _DEFAULTS.gain_per_pop):
        raise ValueError("shadow_state.json requested gain disagrees with requested pop")

    fields = document.get("fields")
    named = document.get("named_values")
    expected_names = [field["name"] for field in _STATE_FIELDS]
    if (not isinstance(fields, list) or len(fields) != len(_STATE_FIELDS) or
            not isinstance(named, dict) or set(named) != set(expected_names)):
        raise ValueError("shadow_state.json does not contain the exact state layout")
    typed: Dict[str, Any] = {}
    for index, (serialized, descriptor) in enumerate(zip(fields, _STATE_FIELDS)):
        expected_type = ("float32" if descriptor["gpu_encoding"] == "float" else
                         "uint32-bitcast")
        if (not isinstance(serialized, dict) or
                set(serialized) != {"index", "name", "type", "value"} or
                serialized.get("index") != index or
                serialized.get("name") != descriptor["name"] or
                serialized.get("type") != expected_type):
            raise ValueError(f"shadow_state.json field {index} has unknown semantics")
        value = serialized.get("value")
        if descriptor["gpu_encoding"] == "float":
            value = _finite_number(value, f"shadow_state.json field {index}")
            named_matches = _same_number(named.get(descriptor["name"]), value)
        else:
            value = _uint32(value, f"shadow_state.json field {index}")
            named_matches = (
                _uint32(
                    named.get(descriptor["name"]),
                    f"shadow_state.json named_values.{descriptor['name']}",
                ) == value
            )
        if not named_matches:
            raise ValueError(f"shadow_state.json field {index} disagrees with named_values")
        typed[descriptor["name"]] = value
    _calibration_revision(
        typed["calibration_revision"],
        "shadow_state.json state calibration_revision",
    )
    if typed["contract_tag_bits"] != _CONTRACT_TAG:
        raise ValueError("shadow_state.json state words have the wrong contract tag")
    expected_camera_integrity = camera_center_integrity_bits(
        float(typed["center"]),
        float(typed["inverse_scale"]),
        float(typed["convergence_curve"]),
        int(typed["calibration_revision"]),
    )
    if typed["camera_center_integrity_bits"] != expected_camera_integrity:
        raise ValueError("shadow_state.json camera center integrity checksum disagrees")
    if any(typed[name] != 0 for name in (
            "mapping_state_reserved_0", "mapping_state_reserved_1",
            "mapping_state_reserved_2")):
        raise ValueError("shadow_state.json has nonzero reserved mapping state")

    decoded = document.get("decoded")
    if not isinstance(decoded, dict) or set(decoded) != _DECODED_KEYS:
        raise ValueError("shadow_state.json has an invalid decoded object")
    if not isinstance(decoded.get("frame_valid"), bool) or not isinstance(
            decoded.get("camera_valid"), bool):
        raise ValueError("shadow_state.json decoded validity fields must be boolean")
    if (typed["frame_valid"] not in (0.0, 1.0) or
            decoded["frame_valid"] != (typed["frame_valid"] > 0.5)):
        raise ValueError("shadow_state.json has an invalid or inconsistent frame_valid flag")
    _calibration_revision(
        decoded.get("calibration_revision"),
        "shadow_state.json decoded.calibration_revision",
    )
    for key in ("confirmed_cut_count", "contract_tag", "camera_center_integrity_bits"):
        _uint32(decoded.get(key), f"shadow_state.json decoded.{key}")
    for key in ("requested_gain", "requested_pop_strength", "latched_scale",
                "convergence_curve", "container_scale", "effective_gain"):
        _finite_number(decoded.get(key), f"shadow_state.json decoded.{key}")
    expected_decoded = {
        "calibration_revision": typed["calibration_revision"],
        "confirmed_cut_count": typed["confirmed_cut_count"],
        "contract_tag": _CONTRACT_TAG,
        "requested_gain": constants["requested_gain"],
        "requested_pop_strength": constants["requested_pop_strength"],
        "camera_valid": (
            typed["inverse_scale"] > 0.0 and typed["calibration_revision"] > 0),
        "latched_scale": (1.0 / typed["inverse_scale"]
                          if decoded["camera_valid"] else 0.0),
        "convergence_curve": typed["convergence_curve"],
        "container_scale": typed["container_scale"],
        "effective_gain": (constants["requested_gain"] * typed["container_scale"]
                           if decoded["frame_valid"] else 0.0),
        "camera_center_integrity_bits": expected_camera_integrity,
    }
    if any(
            (decoded.get(key) != value if key in {
                "camera_valid", "calibration_revision", "confirmed_cut_count", "contract_tag",
                "camera_center_integrity_bits",
            } else not _same_number(decoded.get(key), value))
            for key, value in expected_decoded.items()):
        raise ValueError("shadow_state.json decoded values disagree with the state words")
    convergence_valid = _same_number(
        typed["convergence_curve"], _DEFAULTS.convergence_curve_default)
    if (not 0.0 <= typed["container_scale"] <= 1.0 or
            not convergence_valid or
            decoded["effective_gain"] < 0.0 or
            decoded["effective_gain"] > float(decoded["requested_gain"]) + 1.0e-7 or
            (decoded["camera_valid"] and not _same_number(
                decoded["latched_scale"], constants["raw_coordinate_scale"])) or
            (decoded["frame_valid"] and
             (not decoded["camera_valid"] or typed["container_scale"] <= 0.0)) or
            (not decoded["frame_valid"] and typed["container_scale"] != 1.0) or
            (not decoded["camera_valid"] and
              (typed["center"] != 0.0 or typed["inverse_scale"] != 0.0 or
              not _same_number(
                  typed["convergence_curve"], _DEFAULTS.convergence_curve_default)))):
        raise ValueError("shadow_state.json decoded safety state is out of range")
    return typed


def validate_shadow_frame_stats_document(document: Any) -> Dict[str, float]:
    """Validate one parsed ``shadow_frame_stats.json`` under its independent schema."""

    expected_root = {"schema", "coordinate_contract", "source", "named_values"}
    if not isinstance(document, dict) or set(document) != expected_root:
        raise ValueError("shadow_frame_stats.json has missing or unknown root fields")
    if document.get("schema") != SHADOW_FRAME_STATS_DUMP_SCHEMA:
        raise ValueError("shadow_frame_stats.json has an unknown serialization schema")
    _require_coordinate_binding(
        document.get("coordinate_contract"),
        "frame_stats_word_count",
        len(_FRAME_STATS_FIELDS),
    )
    if document.get("source") != _CONTRACT["frame_stats"]["source"]:
        raise ValueError("shadow_frame_stats.json has an unknown source")
    named = document.get("named_values")
    names = [field["name"] for field in _FRAME_STATS_FIELDS]
    if not isinstance(named, dict) or set(named) != set(names):
        raise ValueError("shadow_frame_stats.json does not contain the exact stats layout")
    values = {name: _finite_number(named[name], f"shadow_frame_stats.json {name}")
              for name in names}
    if values["valid"] not in (0.0, 1.0):
        raise ValueError("shadow_frame_stats.json valid must be zero or one")
    valid_count = values["valid_count"]
    texel_count = values["texel_count"]
    if (valid_count < 0.0 or texel_count <= 0.0 or valid_count > texel_count or
            valid_count != math.floor(valid_count) or
            texel_count != math.floor(texel_count) or
            values["population_std"] < 0.0):
        raise ValueError("shadow_frame_stats.json has invalid counts or spread")
    if values["valid"] > 0.5:
        if (valid_count != texel_count or values["maximum"] < values["minimum"]):
            raise ValueError("shadow_frame_stats.json valid statistics are inconsistent")
    elif any(values[key] != 0.0 for key in
             ("mean", "population_std", "minimum", "maximum")):
        raise ValueError("shadow_frame_stats.json invalid statistics must be canonical zero")
    return values


def validate_v2_dump_manifest_document(document: Any) -> Dict[str, Any]:
    """Validate the schema-10 V2 geometry fragment of ``dump_manifest.json``.

    The full package contains legacy/color/model metadata owned by other contracts. This reader
    deliberately validates only the candidate -> full-resolution ownership refinement -> vertical
    envelopes/share -> row-majorant chain, including every authenticated intermediate.
    """

    if not isinstance(document, dict) or document.get("schema") != DUMP_MANIFEST_SCHEMA:
        raise ValueError("dump_manifest.json has an unknown serialization schema")
    renderer = document.get("renderer")
    shadow = document.get("parallax_v2_shadow")
    artifacts = document.get("artifacts")
    dimensions = document.get("dimensions")
    if not all(isinstance(value, dict) for value in
               (renderer, shadow, artifacts, dimensions)):
        raise ValueError("dump_manifest.json is missing its V2 geometry objects")

    active = shadow.get("active")
    selected = renderer.get("parallax_v2_render_selected")
    if not isinstance(active, bool) or not isinstance(selected, bool):
        raise ValueError("dump_manifest.json has invalid V2 availability flags")
    shadow_selected = shadow.get("rendered_output_selected")
    if not isinstance(shadow_selected, bool) or shadow_selected != selected:
        raise ValueError("dump_manifest.json V2 selection flags disagree")
    if selected and not active:
        raise ValueError("dump_manifest.json selects V2 without an active producer")
    requested = renderer.get("parallax_v2_render_requested")
    mapping_matches = renderer.get("mapping_artifacts_match_selected_renderer")
    if (not isinstance(requested, bool) or
            not isinstance(mapping_matches, bool) or
            (selected and not requested)):
        raise ValueError("dump_manifest.json has invalid V2 renderer request attribution")

    expected_position = "shadow_final_parallax" if selected else None
    expected_authority = (
        "authenticated-parallax-v2-orientation-selective-conditioned-field"
        if selected else None
    )
    expected_inverse = (
        "12-step contractive fixed point; no forward-warp owner/visibility splat and no synthetic fill"
        if selected else None
    )
    expected_coordinate_role = (
        "shadow_coordinate is diagnostic only; it has no renderer authority"
        if selected else None
    )
    expected_live_shader_source = ({
        "source_closure_schema": generator.SOURCE_CLOSURE_SCHEMA,
        "source_compile_flags": generator.SHADER_COMPILE_FLAGS,
        "source_macro_count": 0,
        "source_closure_sha256": LIVE_RENDERER_SOURCE_CLOSURE_SHA256,
        "source_file": "sbs_reprojection_v2_live_ps.hlsl",
        "entrypoint": "main_ps",
        "target": "ps_5_0",
        "diagnostic_source_closure_sha256": DIAGNOSTIC_SOURCE_CLOSURE_SHA256,
        "mapping_source_file": "sbs_reprojection_v2_diagnostics_ps.hlsl",
        "mapping_entrypoint": "mapping_ps",
        "mask_source_file": "sbs_reprojection_v2_diagnostics_ps.hlsl",
        "mask_entrypoint": "mask_ps",
    } if selected else None)
    expected_vertical_role = (
        "least column-wise upper envelope v+ >= ownership-refined candidate with adjacent-row source-U change <= "
        "max_vertical_shear/target_width; diagnostic evidence only"
        if selected else None
    )
    expected_ownership_role = (
        "conservative full-resolution source-contour foreground ownership applied to candidate "
        "before the vertical conditioner; may only raise uniquely owned far-side boundary texels"
        if selected else None
    )
    expected_conditioned_role = (
        "fixed 75/25 share of column upper/lower envelopes; may raise or lower candidate and "
        "feeds the row majorant"
        if selected else None
    )
    expected_final_role = (
        "least row-wise q >= shadow_vertical_conditioned with horizontal slope <= "
        "max_horizontal_slope and vertical shear <= max_vertical_shear; q may raise or lower "
        "candidate and is the live position authority"
        if selected else None
    )
    expected_collar_defocus = ({
        "enabled": False,
        "role": ("disabled after live hand-boundary halo regression; live color uses one "
                 "linear sample at the inverse-warped coordinate"),
        "kernel": "none",
        "hdr": "native source sample; no clamp, tone map, or gamma conversion",
    } if selected else None)
    if (renderer.get("authority") != expected_authority or
            renderer.get("parallax_v2_inverse") != expected_inverse or
            renderer.get("live_shader_source") != expected_live_shader_source or
            renderer.get("parallax_v2_coordinate_role") != expected_coordinate_role or
            renderer.get("parallax_v2_position_field") != expected_position or
            renderer.get("parallax_v2_ownership_refined_role") != expected_ownership_role or
            renderer.get("parallax_v2_vertical_majorant_role") != expected_vertical_role or
            renderer.get("parallax_v2_vertical_conditioned_role") !=
                expected_conditioned_role or
            renderer.get("parallax_v2_conditioner_role") != expected_final_role or
            renderer.get("collar_defocus") != expected_collar_defocus):
        raise ValueError("dump_manifest.json has unknown V2 conditioner attribution")

    expected_artifacts = {
        "shadow_candidate_parallax.f32": (
            "parallax-v2 pre-limiter candidate displacement", True),
        "shadow_ownership_refined_parallax.f32": (
            "parallax-v2 full-resolution contour ownership refinement", True),
        "shadow_ownership_refined_parallax_shape.json":
            ("parallax-v2 full-resolution contour ownership refinement contract", False),
        "shadow_ownership_refined_parallax.png":
            ("parallax-v2 full-resolution contour ownership refinement preview", False),
        "shadow_ownership_refined_parallax_heat.png":
            ("parallax-v2 full-resolution contour ownership refinement preview", False),
        "shadow_vertical_majorant.f32": (
            "parallax-v2 vertical shear-limiter intermediate", False),
        "shadow_vertical_majorant_shape.json":
            ("parallax-v2 vertical shear-limiter intermediate contract", False),
        "shadow_vertical_majorant.png":
            ("parallax-v2 vertical shear-limiter intermediate preview", False),
        "shadow_vertical_majorant_heat.png":
            ("parallax-v2 vertical shear-limiter intermediate preview", False),
        "shadow_vertical_conditioned.f32":
            ("parallax-v2 orientation-selective vertical conditioner", False),
        "shadow_vertical_conditioned_shape.json":
            ("parallax-v2 orientation-selective vertical conditioner contract", False),
        "shadow_vertical_conditioned.png":
            ("parallax-v2 orientation-selective vertical conditioner preview", False),
        "shadow_vertical_conditioned_heat.png":
            ("parallax-v2 orientation-selective vertical conditioner preview", False),
        "shadow_final_parallax.f32": (
            "parallax-v2 final conditioned displacement field", True),
    }
    for name, (stage, required) in expected_artifacts.items():
        descriptor = artifacts.get(name)
        if (not isinstance(descriptor, dict) or
                set(descriptor) != {"available", "required", "stage", "description"} or
                descriptor.get("available") is not active or
                descriptor.get("required") is not required or
                descriptor.get("stage") != stage or
                not isinstance(descriptor.get("description"), str) or
                not descriptor["description"]):
            raise ValueError(
                "dump_manifest.json has an invalid V2 geometry artifact contract")
    dimension_names = (
        "shadow_candidate_parallax", "shadow_ownership_refined_parallax",
        "shadow_vertical_majorant",
        "shadow_vertical_conditioned", "shadow_final_parallax")
    geometry_dimensions = [dimensions.get(name) for name in dimension_names]
    if active:
        if any(not isinstance(value, dict) for value in geometry_dimensions):
            raise ValueError("dump_manifest.json omits an active V2 geometry dimension")
        expected_dimensions = geometry_dimensions[0]
        if (set(expected_dimensions) != {"width", "height", "format", "format_value"} or
                type(expected_dimensions["width"]) is not int or
                type(expected_dimensions["height"]) is not int or
                expected_dimensions["width"] <= 0 or expected_dimensions["height"] <= 0 or
                expected_dimensions["format"] != "DXGI_FORMAT_R32_FLOAT" or
                expected_dimensions["format_value"] != 41 or
                any(value != expected_dimensions for value in geometry_dimensions[1:])):
            raise ValueError("dump_manifest.json V2 geometry dimensions disagree")
    elif any(value is not None for value in geometry_dimensions):
        raise ValueError("dump_manifest.json exposes inactive V2 geometry dimensions")
    return {
        "active": active,
        "rendered_output_selected": selected,
        "mapping_artifacts_match_selected_renderer": mapping_matches,
        "position_field": expected_position,
        "ownership_refined_available": active,
        "vertical_majorant_available": active,
        "vertical_conditioned_available": active,
    }


__all__ = [
    "DIAGNOSTIC_SOURCE_CLOSURE_SHA256",
    "DUMP_MANIFEST_SCHEMA",
    "LIVE_RENDERER_SOURCE_CLOSURE_SHA256",
    "SHADOW_FRAME_STATS_DUMP_SCHEMA",
    "SHADOW_STATE_DUMP_SCHEMA",
    "camera_center_integrity_bits",
    "validate_v2_dump_manifest_document",
    "validate_shadow_frame_stats_document",
    "validate_shadow_state_document",
]
