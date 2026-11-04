"""Typed readback of the actual shared UI warp, separate from legacy map diagnostics.

This reader checks immutable GPU artifacts and geometric invariants. It never recreates
the pin scan or substitutes a CPU renderer for the production shaders.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any


KIND = "shared-ui-pin-v1"
EVIDENCE = "actual_draw_evidence.json"
ARTIFACTS = (
    "actual_shared_pin_weights.f32", "actual_shared_pin_field.f32",
    "actual_shared_pin_rows.u32", "actual_shared_pin_geometry.u32",
    "actual_shared_pin_map.rgba32f", "actual_shared_pin_boundary.rgba32f",
)
LEGACY_ROLE = "independent analytic diagnostic; does not describe rendered shared warp"


def _expected_closure() -> str:
    try:
        from . import generate_depth_coordinate_v2_contract as shader_generator
    except ImportError:
        import generate_depth_coordinate_v2_contract as shader_generator
    return shader_generator.shared_ui_warp_runtime_closures()["closure"]


def validate_attribution(manifest: dict[str, Any]) -> dict[str, Any]:
    """Accept the optional extension without reinterpreting older schema-50 dumps."""
    value = manifest.get("renderer", {}).get("actual_draw_renderer")
    if value is None:
        return {"kind": "legacy-analytic-v1", "shared_pin_selected": False}
    if not isinstance(value, dict) or set(value) != {
            "kind", "source_closure_sha256", "legacy_mapping_role", "evidence_artifact"}:
        raise ValueError("actual draw renderer attribution is malformed")
    shared = value["kind"] == KIND
    if (value["kind"] not in {KIND, "legacy-analytic-v1"} or
            value["source_closure_sha256"] != (_expected_closure() if shared else "") or
            value["legacy_mapping_role"] != (LEGACY_ROLE if shared else
                "selected analytic renderer diagnostic") or
            value["evidence_artifact"] != EVIDENCE):
        raise ValueError("actual draw renderer has an unknown source or map attribution")
    if shared and manifest["renderer"].get("mapping_artifacts_match_selected_renderer") is not False:
        raise ValueError("legacy analytic map cannot claim shared-warp render authority")
    return {"kind": value["kind"], "shared_pin_selected": shared,
            "source_closure_sha256": value["source_closure_sha256"]}


def _read(root: Path, manifest: dict[str, Any], name: str) -> bytes:
    descriptor = manifest["artifacts"].get(name)
    if not isinstance(descriptor, dict) or descriptor.get("available") is not True:
        raise ValueError(f"{name} is not available")
    try:
        payload = (root / name).read_bytes()
    except OSError as error:
        raise ValueError(f"{name} is missing") from error
    if hashlib.sha256(payload).hexdigest() != descriptor.get("sha256"):
        raise ValueError(f"{name} content hash mismatch")
    if "byte_count" in descriptor and len(payload) != descriptor["byte_count"]:
        raise ValueError(f"{name} exact byte_count mismatch")
    return payload


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"actual draw evidence contains duplicate field: {key}")
        result[key] = value
    return result


def _same_source_word(value: Any, expected: Any) -> bool:
    return type(value) is int and type(expected) is int and 0 < value < (1 << 64) and value == expected


def _verify(root: Any, manifest: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Return the summary and its already authenticated map snapshot, when available."""
    attribution = validate_attribution(manifest)
    if not attribution["shared_pin_selected"]:
        return {**attribution, "actual_map_verified": False}, None
    root = Path(root)
    descriptor = manifest["artifacts"].get(EVIDENCE, {})
    if not isinstance(descriptor, dict):
        raise ValueError("actual draw evidence descriptor must be an object")
    if descriptor.get("available") is not True:
        return {**attribution, "actual_map_verified": False,
                "unavailable_reason": "optional actual draw evidence is unavailable"}, None
    try:
        evidence = json.loads(_read(root, manifest, EVIDENCE), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("actual draw evidence is invalid JSON") from error
    if not isinstance(evidence, dict):
        raise ValueError("actual draw evidence must be a JSON object")
    if evidence.get("captured") is False:
        return {**attribution, "actual_map_verified": False,
                "unavailable_reason": "actual draw binding observation was unavailable"}, None
    shared = evidence.get("shared_pin")
    if (not isinstance(shared, dict) or shared.get("selected") is not True or
            evidence.get("captured") is not True or
            shared.get("source_closure_sha256") != attribution["source_closure_sha256"] or
            evidence.get("pixel_shader_closure") != attribution["source_closure_sha256"] or
            not _same_source_word(evidence.get("model_frame"), manifest.get("matched_frame_id")) or
            not _same_source_word(evidence.get("observation_us"), manifest.get("matched_observation_timestamp_us")) or
            shared.get("map_artifact") != ARTIFACTS[4] or
            shared.get("boundary_artifact") != ARTIFACTS[5] or
            shared.get("map_layout") != (
                "packed-output row-major float32-le RGBA: sourceU,sourceV,pinnedRow,contentValid") or
            shared.get("boundary_layout") != "packed-output row-major float32-le RGBA: finiteSourceBoundary,0,0,1"):
        raise ValueError("shared pin evidence does not identify its exact draw and source")
    evidence_artifacts = evidence.get("artifacts")
    if not isinstance(evidence_artifacts, dict):
        raise ValueError("actual draw evidence artifacts must be an object")
    payloads = {}
    missing = []
    for name in ARTIFACTS:
        descriptor = manifest["artifacts"].get(name)
        if not isinstance(descriptor, dict) or descriptor != evidence_artifacts.get(name):
            raise ValueError(f"{name} evidence/manifest descriptor mismatch")
        if descriptor.get("available") is True:
            payloads[name] = _read(root, manifest, name)
        else:
            missing.append(name)
    if missing:
        return {**attribution, "actual_map_verified": False,
                "unavailable_reason": "optional shared readbacks unavailable", "missing": missing}, None

    import numpy as np

    if len(payloads[ARTIFACTS[3]]) != 16:
        raise ValueError("shared pin b6 must contain exactly four uint32 words")
    width, height, top, band = struct.unpack("<4I", payloads[ARTIFACTS[3]])
    source = manifest["dimensions"]["source"]
    packed = manifest["dimensions"]["packed_sbs"]
    source_shape = shared.get("source_shape")
    if (width != source["width"] or height != source["height"] or
            width == 0 or height == 0 or band == 0 or top >= height or band > height - top or
            not isinstance(source_shape, list) or any(type(word) is not int for word in source_shape) or
            source_shape != [width, height, top, band]):
        raise ValueError("shared pin b6/source shape disagrees")
    shapes = ((width, band, 41, 4), (width, band, 41, 4),
              (1, band + 1, 42, 4), None,
              (packed["width"], packed["height"], 2, 16),
              (packed["width"], packed["height"], 2, 16))
    for name, shape in zip(ARTIFACTS, shapes):
        if shape is None:
            continue
        w, h, fmt, stride = shape
        desc = manifest["artifacts"][name].get("texture", {})
        if (not isinstance(desc, dict) or desc.get("width") != w or desc.get("height") != h or
                desc.get("format_value") != fmt or
                len(payloads[name]) != w * h * stride):
            raise ValueError(f"{name} exact texture layout disagrees")
    weights = np.frombuffer(payloads[ARTIFACTS[0]], dtype="<f4").reshape(band, width)
    field = np.frombuffer(payloads[ARTIFACTS[1]], dtype="<f4").reshape(band, width)
    rows = np.frombuffer(payloads[ARTIFACTS[2]], dtype="<u4")
    if (not np.isfinite(field).all() or not np.isin(weights, (0.0, 1.0)).all() or
            not np.isin(rows, (0, 1)).all() or
            not np.array_equal(rows[:-1].astype(bool), np.any(weights > 0, axis=1)) or
            (rows[-1] == 0 and np.any(weights))):
        raise ValueError("shared pin authorized weights/rows/header are noncanonical")
    map_shape = (packed["height"], packed["width"], 4)
    mapping = np.frombuffer(payloads[ARTIFACTS[4]], dtype="<f4").reshape(map_shape)
    boundary = np.frombuffer(payloads[ARTIFACTS[5]], dtype="<f4").reshape(map_shape)
    if (not np.isfinite(mapping).all() or not np.isfinite(boundary).all() or
            not np.isin(mapping[..., 2:], (0.0, 1.0)).all() or
            not np.isin(boundary[..., 0], (0.0, 1.0)).all() or
            np.any(boundary[..., 1:3] != 0) or np.any(boundary[..., 3] != 1)):
        raise ValueError("shared pin map/boundary contains noncanonical channels")
    valid = mapping[..., 3] == 1
    if any(np.any(mapping[..., channel] != expected, where=~valid)
           for channel, expected in enumerate((-1.0, -1.0, 0.0))):
        raise ValueError("shared pin invalid content map must use its canonical sentinel")
    expected_boundary = valid & ((mapping[..., 0] < 0) | (mapping[..., 0] > 1))
    if (not np.array_equal(boundary[..., 0].astype(bool), expected_boundary) or
            np.any(mapping[..., 1][valid] < 0) or np.any(mapping[..., 1][valid] > 1)):
        raise ValueError("shared pin map and finite-source boundary disagree")
    return {**attribution, "actual_map_verified": True,
            "map_artifact": ARTIFACTS[4], "boundary_artifact": ARTIFACTS[5],
            "width": packed["width"], "height": packed["height"],
            "authorized_native_pixels": int(np.count_nonzero(weights)),
            "pinned_native_rows": int(np.count_nonzero(rows[:-1])),
            "selected_receipt_authorized": bool(rows[-1]),
            "legacy_map_has_render_authority": False,
            "pin_scan_replayed_on_cpu": False}, mapping


def verify(root: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    """Authenticate same-draw fields/maps; optional missing diagnostics remain explicit."""
    return _verify(root, manifest)[0]


def load_actual_map(root: Any, manifest: dict[str, Any]) -> Any:
    """Return the authenticated actual source-U map for shared-warp numeric analysis."""
    summary, mapping = _verify(root, manifest)
    if not summary.get("actual_map_verified"):
        raise ValueError("shared warp has no complete authenticated actual-map readback")
    return mapping[..., 0].copy()
