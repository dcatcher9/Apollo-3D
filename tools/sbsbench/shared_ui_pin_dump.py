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

try:
    from .host_sbs_shader_manifest import DIRECT_GLYPH_PREPARE_GROUP
except ImportError:
    from host_sbs_shader_manifest import DIRECT_GLYPH_PREPARE_GROUP


KIND = "shared-ui-pin-v1"
COMPLETED_KIND = "shared-ui-pin-completed-v2"
QUEUED_KIND = "shared-ui-pin-queued-v2"
LEGACY_COMBINED_CLOSURE = "326bf2defe8021e1930f46279fea27ceac12effad2ad9d2a17127d647bcdddef"
_ROUTE_CLOSURE_KEYS = {COMPLETED_KIND: "completed_closure", QUEUED_KIND: "queued_closure"}
# Recorded by the last producer with one displacement band shared by both eyes (23a4d95e).
SINGLE_BAND_SPLIT_CLOSURES = {
    COMPLETED_KIND: "50db509da2b32216404aba41fc70095ca7b82e36bc86b0417a98cd32948a1d4f",
    QUEUED_KIND: "813e5680174a7d4aeffcf434cde666f61774e2329bbc6753d52062f8dbada412",
}
SINGLE_BAND_RAW_PREPARE = "b254eba72ffaca96d899fb48ec44596d217cb8312334bea7544080cf07a4c3ad"
SINGLE_BAND_DIRECT_PREPARE = "219a6a305c05ca3ac1744286a4e87b83b61128c10e05c61ec517337f279ec940"
EVIDENCE = "actual_draw_evidence.json"
SELECTED_RECEIPT = "actual_selected_receipt.u32"
ARTIFACTS = (
    "actual_shared_pin_weights.f32", "actual_shared_pin_field.f32",
    "actual_shared_pin_rows.u32", "actual_shared_pin_geometry.u32",
    "actual_shared_pin_map.rgba32f", "actual_shared_pin_boundary.rgba32f",
)
LEGACY_ROLE = "independent analytic diagnostic; does not describe rendered shared warp"
SHARED_ROLE = "independent Base diagnostic; does not describe rendered shared warp"
BASE_ROLE = "authenticated Base inverse diagnostic"


def _runtime_closure(name: str) -> str:
    try:
        from . import generate_depth_coordinate_v2_contract as shader_generator
    except ImportError:
        import generate_depth_coordinate_v2_contract as shader_generator
    return shader_generator.shared_ui_warp_runtime_closures(names=(name,))[name]


def _expected_closure(kind: str = KIND) -> str:
    return LEGACY_COMBINED_CLOSURE if kind == KIND else _runtime_closure(_ROUTE_CLOSURE_KEYS[kind])


def is_single_band_split_draw(draw: dict) -> bool:
    kind = draw.get("kind")
    return (kind in (COMPLETED_KIND, QUEUED_KIND) and
            draw.get("source_closure_sha256") == SINGLE_BAND_SPLIT_CLOSURES[kind])


def validate_attribution(manifest: dict[str, Any]) -> dict[str, Any]:
    """Read schema-2 Base/shared attribution and retain historical unversioned attribution."""
    if "schema" in manifest and type(manifest["schema"]) is not int:
        raise ValueError("dump serialization schema must be an integer")
    current_dump = manifest.get("schema", 0) >= 51
    value = manifest.get("renderer", {}).get("actual_draw_renderer")
    if value is None:
        if current_dump:
            raise ValueError("current dump requires versioned actual draw renderer attribution")
        return {"kind": "legacy-analytic-v1", "shared_pin_selected": False}
    if not isinstance(value, dict):
        raise ValueError("actual draw renderer attribution is malformed")
    current = "schema" in value
    expected_fields = {"kind", "source_closure_sha256", "evidence_artifact"}
    expected_fields |= {"schema", "mapping_role"} if current else {"legacy_mapping_role"}
    if (set(value) != expected_fields or
            (current and (type(value["schema"]) is not int or value["schema"] != 2)) or
            (current_dump and not current)):
        raise ValueError("actual draw renderer attribution has an unknown schema")
    kind = value["kind"]
    shared_kinds = (KIND, COMPLETED_KIND, QUEUED_KIND) if current else (KIND,)
    shared = kind in shared_kinds
    role = SHARED_ROLE if shared else BASE_ROLE
    if not current:
        role = LEGACY_ROLE if shared else "selected analytic renderer diagnostic"
    if ((not shared and kind != ("ordinary-base-v1" if current else "legacy-analytic-v1")) or
            (not is_single_band_split_draw(value) and
             value["source_closure_sha256"] != (_expected_closure(kind) if shared else "")) or
            value["mapping_role" if current else "legacy_mapping_role"] != role or
            value["evidence_artifact"] != EVIDENCE):
        raise ValueError("actual draw renderer has an unknown source or map attribution")
    if shared and manifest["renderer"].get("mapping_artifacts_match_selected_renderer") is not False:
        raise ValueError("ordinary/legacy analytic map cannot claim shared-warp render authority")
    return {"kind": kind, "shared_pin_selected": shared,
            "source_closure_sha256": value["source_closure_sha256"],
            "evidence_schema": 2 if current else 1}


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


def _hybrid_receipt_kind(root: Path, manifest: dict, evidence: dict, authorized: bool,
                         direct_glyph: bool) -> str | None:
    descriptor = manifest["artifacts"].get(SELECTED_RECEIPT)
    if (evidence.get("binding_roles", {}).get("t7") != SELECTED_RECEIPT or not isinstance(descriptor, dict) or
            evidence["artifacts"].get(SELECTED_RECEIPT) != descriptor):
        raise ValueError("hybrid selected receipt lacks its exact t7 artifact binding")
    if descriptor.get("available") is not True:
        return None
    payload = _read(root, manifest, SELECTED_RECEIPT)
    if len(payload) != 64 * 4:
        raise ValueError("hybrid selected receipt has the wrong byte length")
    words = struct.unpack("<64I", payload)
    if (list(words[:4]) != evidence.get("selected_receipt_header") or
            list(words[60:]) != evidence.get("selected_receipt_token_algorithm")):
        raise ValueError("hybrid selected receipt disagrees with the draw evidence")
    if not authorized:
        return None
    tag, schema, kind, invalid = words[:4]
    token_low, token_high, algorithm, reserved = words[60:]
    roles = {(4, 3): "hybrid", (1, 1): "legacy-fallback"}
    if direct_glyph:
        roles[4, 4] = "direct-glyph"
    if ((tag, schema, invalid, reserved) != (0x31534D53, 1, 0, 0) or
            not (token_low or token_high) or (kind, algorithm) not in roles):
        raise ValueError("authorized hybrid draw has an unexpected selected receipt")
    return roles[kind, algorithm]


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
    if ((attribution["evidence_schema"] == 2 and
            (type(evidence.get("schema")) is not int or evidence["schema"] != 2)) or
            (attribution["evidence_schema"] == 1 and "schema" in evidence)):
        raise ValueError("actual draw evidence schema disagrees with renderer attribution")
    if attribution["evidence_schema"] == 2:
        roles = evidence.get("binding_roles", {})
        if not isinstance(roles, dict) or "t8" in roles:
            raise ValueError("current actual draw evidence retains the retired legacy t8 mask")
    if evidence.get("captured") is False:
        return {**attribution, "actual_map_verified": False,
                "unavailable_reason": "actual draw binding observation was unavailable"}, None
    kind = attribution["kind"]
    if kind in _ROUTE_CLOSURE_KEYS and (
            evidence.get("program") != kind or evidence.get("queued_geometry") is not (kind == QUEUED_KIND)):
        raise ValueError("actual draw evidence does not identify its attributed shared pin route")
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
    model_prepare = shared.get("model_prepare_source_closure_sha256")
    single_band_split = is_single_band_split_draw(attribution)
    direct_prepare = (SINGLE_BAND_DIRECT_PREPARE if single_band_split else
                      DIRECT_GLYPH_PREPARE_GROUP.source_closure_sha256)
    if model_prepare is not None:
        raw_prepare = SINGLE_BAND_RAW_PREPARE if single_band_split else _runtime_closure("raw_model_prepare_closure")
        if kind != COMPLETED_KIND or model_prepare not in (raw_prepare, direct_prepare):
            raise ValueError("shared pin model prepare has an unknown source or route")
        attribution["model_prepare_source_closure_sha256"] = model_prepare
    evidence_artifacts = evidence.get("artifacts")
    if not isinstance(evidence_artifacts, dict):
        raise ValueError("actual draw evidence artifacts must be an object")
    if attribution["evidence_schema"] == 2 and (
            "actual_original_legacy_M.u32" in evidence_artifacts or
            "actual_original_legacy_M.u32" in manifest["artifacts"]):
        raise ValueError("current actual draw evidence retains the retired legacy t8 mask")
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
    # The field and weight planes hold the band once (dumps before the eye-specific limiter) or
    # once per eye: the left eye's rows, then the right eye's.
    eye_copies = 1 if kind == KIND or single_band_split else 2
    shapes = ((width, band * eye_copies, 41, 4), (width, band * eye_copies, 41, 4),
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
    weights = np.frombuffer(payloads[ARTIFACTS[0]], dtype="<f4").reshape(eye_copies, band, width)
    field = np.frombuffer(payloads[ARTIFACTS[1]], dtype="<f4").reshape(eye_copies, band, width)
    rows = np.frombuffer(payloads[ARTIFACTS[2]], dtype="<u4")
    if (not np.isfinite(field).all() or not np.isin(weights, (0.0, 1.0)).all() or
            not np.isin(rows, (0, 1)).all() or
            not np.array_equal(rows[:-1].astype(bool), np.any(weights > 0, axis=(0, 2))) or
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
    if model_prepare is not None:
        attribution["verified_selected_matte_kind"] = _hybrid_receipt_kind(
            root, manifest, evidence, bool(rows[-1]), model_prepare == direct_prepare)
    return {**attribution, "actual_map_verified": True,
            "map_artifact": ARTIFACTS[4], "boundary_artifact": ARTIFACTS[5],
            "width": packed["width"], "height": packed["height"],
            "authorized_native_pixels": int(np.count_nonzero(weights[0])),
            "eye_copies": eye_copies,
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
