#!/usr/bin/env python3
"""Run/validate file-fed actual completed current-outline compositor sidecars.

This is a renderer qualification, not an extension of ordinary metric or live-capture authority.
The native harness retains its legacy sbs_* metric images and writes the selected draw separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path

from PIL import Image

import depth_coordinate_v2_contract as coordinate
import depth_coordinate_v2_dump_contract as dump_contract
import generate_depth_coordinate_v2_contract as generator
import current_outline_mask_metrics as mask_metrics
from source_observation_timeline import (
    TimelineError, prepared_observation_timestamps, read_observation_timeline, write_observation_timeline,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "current_outline_qualification.json"
ROLE = "file-fed-actual-completed-current-outline-compositor"
FRAME_NAME = re.compile(r"frame_(\d+)\.png\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
MAXIMUM_FRAMES = 64
# Read-only identities for retired qualification schemas. These do not load or
# register the deleted renderer, and cannot qualify the current shared-only ABI.
RETIRED_COMPLETED_SOURCE_CLOSURE = "c8053c9ed5b1f3969e2852f63df3f811671e88f8b1ec8e9dae3b7fc616a3ebe6"
RETIRED_SHARED_SOURCE_CLOSURE = "60febe4bfb1ab4d22142648c9898e3407af0a1f83eedbbcf5593f3ca19a90626"
COMPOSITION_ROLES = ("source5", "colour", "components", "enclosure", "roi", "outline", "inside", "selector", "stability", "hole", "coverage")
PRODUCER_STAGES = {
    "cbc_meta": (512, "current_outline_stages[2]:CBCMeta512", "BUFFER", "UNKNOWN", 4),
    "raw_meta": (512, "current_outline_stages[7]:RawMeta512", "BUFFER", "UNKNOWN", 4),
    "f2_data": (68608, "current_outline_stages[8]:F2data68608", "BUFFEREX", "R32_TYPELESS", 0),
}


class QualificationError(ValueError):
    """Incomplete, mismatched or unauthenticated compositor evidence."""


def _require(condition, message):
    if not condition:
        raise QualificationError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uint(value, label, maximum=(1 << 64) - 1, positive=False):
    _require(type(value) is int and (1 if positive else 0) <= value <= maximum,
             f"{label} is not a valid unsigned integer")
    return value


def _words(value, count, label):
    _require(isinstance(value, list) and len(value) == count, f"{label} has the wrong word count")
    return [_uint(word, f"{label}[{index}]", (1 << 32) - 1) for index, word in enumerate(value)]


def _join(words, offset):
    return words[offset] | words[offset + 1] << 32


def _within(directory, name):
    _require(isinstance(name, str) and name and "\\" not in name, "artifact path is malformed")
    relative = Path(name)
    _require(not relative.is_absolute() and ".." not in relative.parts and ":" not in name,
             "artifact path escapes its qualification directory")
    path = (directory / relative).resolve()
    _require(path.is_relative_to(directory.resolve()) and path.is_file(), "artifact is absent or escapes its owner")
    return path


def _closed(value, names, label):
    _require(isinstance(value, dict) and set(value) == set(names), f"{label} has unknown or missing fields")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate qualification field: {key}")
        result[key] = value
    return result


def source_frames(directory):
    directory = Path(directory)
    _require(directory.is_dir(), "source directory is absent")
    found = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        match = FRAME_NAME.fullmatch(path.name)
        _require(match is not None, "qualification requires canonical frame_<id>.png sources only")
        found.append((int(match[1]), path))
    found.sort(key=lambda item: (item[0], item[1].name))
    _require(1 <= len(found) <= MAXIMUM_FRAMES, "qualification requires 1..64 source frames")
    _require(len({number for number, _ in found}) == len(found), "source frame identities are duplicated")
    return [path for _, path in found]


def expected_closures(schema=1):
    # The C++ role/spec lists remain the owner. Check their source pins through the normal
    # generator rather than trusting a digest supplied by a qualification document.
    _require(generator._write_current_outline_runtime_pins(True), "current-outline source pins are stale")
    if schema >= 5:
        _require(generator._write_shared_ui_warp_runtime_pin(True), "shared warp source pin is stale")
    elif schema < 4:
        _require(generator._write_current_background_runtime_pin(True), "background source pin is stale")
    source = generator.CURRENT_OUTLINE_RUNTIME_HEADER.read_text(encoding="utf-8")
    result = {}
    if schema < 5:
        result["completed_source_closure_sha256"] = RETIRED_COMPLETED_SOURCE_CLOSURE
    for name in ("selector",):
        match = re.search(rf'inline constexpr std::string_view {name}_closure="([0-9a-f]{{64}})";', source)
        _require(match is not None, f"native {name} closure is absent")
        result[f"{name}_source_closure_sha256"] = match[1]
    if schema == 4:
        result["shared_warp_source_closure_sha256"] = RETIRED_SHARED_SOURCE_CLOSURE
        return result
    header = (generator.CURRENT_SHARED_UI_WARP_RUNTIME_HEADER if schema >= 5
              else generator.CURRENT_BACKGROUND_RUNTIME_HEADER)
    source = header.read_text(encoding="utf-8")
    match = re.search(r'inline constexpr std::string_view closure\s*=\s*"([0-9a-f]{64})";', source)
    _require(match is not None, "native renderer closure is absent")
    result["shared_warp_source_closure_sha256" if schema >= 4 else "background_source_closure_sha256"] = match[1]
    return result


def expected_composition_closures(schema=5):
    source = generator.CURRENT_OUTLINE_RUNTIME_HEADER.read_text(encoding="utf-8")
    result = {}
    roles = (*COMPOSITION_ROLES, "foreground_edge") if schema >= 6 else COMPOSITION_ROLES
    for role in roles:
        match = re.search(rf'inline constexpr std::string_view {role}_closure="([0-9a-f]{{64}})";', source)
        _require(match is not None, f"native composition {role} closure is absent")
        result[role] = match[1]
    return result


def validate_producer_stages(frame, output):
    """Diagnostic aliases are never substitutes for the selected/full-source receipt."""
    stages = frame["producer_stages"]
    _closed(stages, {"scope", "pending_queued", "selected_matte_matches_draw", "selected_receipt_matches_draw",
                     "completed_frame_id", "observation_timestamp_us", "analysis_generation", "transaction_token",
                     "source_snapshot_sha256", "buffers"}, "producer stage owner")
    _require(stages["scope"] == "diagnostic-only-exact-completed-composition" and
             stages["pending_queued"] is False and stages["selected_matte_matches_draw"] is True and
             stages["selected_receipt_matches_draw"] is True, "producer stage completed draw role differs")
    for key in ("completed_frame_id", "observation_timestamp_us", "analysis_generation", "transaction_token"):
        _require(_uint(stages[key], f"producer stage {key}") == frame[key], "producer stage source owner differs")
    _require(stages["source_snapshot_sha256"] == frame["source_snapshot_sha256"], "producer stage snapshot differs")
    buffers = stages["buffers"]
    _closed(buffers, PRODUCER_STAGES, "producer stage buffers")
    decoded = {}
    for name, (words, role, dimension, view_format, stride) in PRODUCER_STAGES.items():
        binding = buffers[name]
        _closed(binding, {"file", "sha256", "role", "word_count", "byte_count", "format", "view_dimension",
                          "view_format", "view_first_element", "view_word_count", "resource_byte_count",
                          "structure_stride_bytes"}, f"producer stage {name}")
        expected_path = f"current-outline/producer-stages/frame_{frame['source_file_id']}/{name}.u32"
        _require(binding["file"] == expected_path and binding["role"] == role and
                 binding["format"] == "uint32-little-endian" and binding["view_dimension"] == dimension and
                 binding["view_format"] == view_format, "producer stage artifact role/format differs")
        expected_numbers = {"word_count": words, "byte_count": words * 4, "view_first_element": 0,
                            "view_word_count": words, "resource_byte_count": words * 4, "structure_stride_bytes": stride}
        for key, expected in expected_numbers.items():
            _require(_uint(binding[key], f"producer stage {name} {key}") == expected,
                     "producer stage whole-buffer shape differs")
        path = _within(output, binding["file"])
        _require(path.stat().st_size == words * 4, "producer stage byte length differs")
        payload = path.read_bytes()
        _require(len(payload) == words * 4 and hashlib.sha256(payload).hexdigest() == binding["sha256"],
                 "producer stage artifact size/hash differs")
        decoded[name] = struct.unpack(f"<{words}I", payload)
    # Kind3 already authenticates these literal header/CB joins in the production selector.
    # A fallback tuple's diagnostic bytes remain unavailable evidence; no model bank is inferred.
    if frame["selected_receipt"][2:4] != [3, 0]:
        return False
    cbc, raw, f2 = (decoded[name] for name in ("cbc_meta", "raw_meta", "f2_data"))
    c24 = tuple(frame["selected_receipt"][4:28])
    _require(cbc[:4] == (0x31434E45, 5, 1, 0) and raw[:4] == (0x31474E52, 1, 1, 0) and
             f2[:4] == (0x37424643, 7, 1, 0), "producer stage current headers differ")
    _require(cbc[16:40] == raw[12:36] == f2[8:32] == c24 and
             raw[36:52] == cbc[:16] and f2[32:48] == raw[:16], "producer stage current C24/header join differs")
    _require(cbc[4] == cbc[5] == cbc[9] == 1 and cbc[80] == frame["contour_identity"][3],
             "producer stage current source/cut header differs")
    selected = frame["selected_receipt"]
    width, height, _, band = selected[4:8]
    _require(cbc[40:43] == (0x31434E45, 1, 1) and cbc[43] in {1, 2, 4} and
             cbc[44:52] == (width, band, 28, 1, width, height, 28, 1) and
             cbc[52:56] == (*selected[60:62], 1, 1 ^ 0x6F435257) and
             cbc[56:60] == (*selected[16:18], *selected[20:22]) and
             cbc[60:64] == (*selected[18:20], 0, 0) and
             cbc[64:72] == (0, 0, width, height, 5, 0, 4, 136) and cbc[72:80] == (0,) * 8,
             "producer stage current source envelope/token differs")
    return True


def validate_shared_warp(frame, width, height, crop):
    """Check the owned native pin band; row flags remain pointwise-current GPU evidence."""
    shared = frame["shared_warp"]
    _closed(shared, {"renderer", "shape", "draw_serial", "selected_receipt_header", "active_rows"}, "shared warp")
    _require(shared["renderer"] == "shared-ui-pin-band", "shared warp renderer differs")
    band_top = height - crop
    top = max(0, band_top - 1)
    shape = [width, height, top, height - top]
    _require(_words(shared["shape"], 4, "shared warp shape") == shape,
             "shared warp native source/band shape differs")
    serial = _uint(shared["draw_serial"], "shared warp draw serial", positive=True)
    header = _uint(shared["selected_receipt_header"], "shared warp selected receipt header", 1)
    selected_current = frame["selected_receipt"][2:4] == [3, 0]
    _require(header == int(selected_current), "shared warp selected receipt authority differs")
    rows = _words(shared["active_rows"], shape[3], "shared warp active rows")
    _require(all(value in {0, 1} for value in rows), "shared warp row flags are not binary")
    active = bool(header and frame["contour_identity"][2] == 1 and frame["condition_params"][4] == 2)
    _require(active or not any(rows), "inactive shared warp grants pin authority")
    _require(top == band_top or rows[0] == 0, "shared warp border row grants glyph authority")
    return serial, int(any(rows))


def _validate_frame(frame, ordinal, source, clock, output, scale, schema):
    fields = {
        "observation_ordinal", "source_file", "source_file_id", "source_snapshot_sha256", "source_size",
        "output_file", "output_sha256", "output_size", "completed_frame_id", "observation_timestamp_us",
        "transaction_token", "analysis_generation", "bindings_observed",
        "selected_receipt", "contour_identity", "condition_params", "parallax_state", "subtitle_constants",
        "publication_receipt",
    }
    fields.update({"shared_warp"} if schema >= 4 else {"background_serial", "background_proof", "background_expected"})
    if schema >= 2:
        fields.add("selected_mask")
    if schema >= 3:
        fields.add("producer_stages")
        fields.add("hole_program_ready")
        fields.add("coverage_program_ready")
    if schema >= 6:
        fields.add("foreground_edge_program_ready")
    _closed(frame, fields, "qualified frame")
    if schema >= 3:
        _require(frame["hole_program_ready"] is True, "current mask-hole programs were unavailable")
        _require(frame["coverage_program_ready"] is True, "current stroke-coverage programs were unavailable")
    if schema >= 6:
        _require(frame["foreground_edge_program_ready"] is True, "current foreground-edge program was unavailable")
    source_id = FRAME_NAME.fullmatch(source.name)[1]
    _require(_uint(frame["observation_ordinal"], "observation ordinal", positive=True) == ordinal and
             _uint(frame["completed_frame_id"], "completed frame", positive=True) == ordinal,
             "qualified frame does not own its current observation")
    _require(frame["source_file"] == source.name and frame["source_file_id"] == source_id and
             frame["source_snapshot_sha256"] == _sha(source), "source snapshot/identity differs")
    _require(_uint(frame["observation_timestamp_us"], "observation clock", positive=True) == clock and
             _uint(frame["analysis_generation"], "analysis generation") == 0,
             "qualified clock or full-source analysis generation differs")
    _require(frame["bindings_observed"] is True, "actual selected draw bindings were not observed")
    token = _uint(frame["transaction_token"], "transaction token", positive=True)
    serial = None if schema >= 4 else _uint(frame["background_serial"], "background serial", positive=True)
    with Image.open(source) as image:
        _require(image.format == "PNG", "source snapshot is not PNG")
        image.load()
        source_size = list(image.size)
    width, height = source_size
    _require(_words(frame["source_size"], 2, "source size") == source_size and max(source_size) <= 5120 and
             width * height <= 5120 * 2160, "source raster differs or exceeds the Host source boundary")
    _require(frame["output_file"] == f"current-outline/sbs_{source_id}.png", "output identity differs")
    png = _within(output, frame["output_file"])
    _require(HASH.fullmatch(str(frame["output_sha256"])) is not None and _sha(png) == frame["output_sha256"],
             "selected output hash differs")
    with Image.open(png) as image:
        image.load()
        output_size = list(image.size)
        _require(image.format == "PNG", "selected artifact is not PNG")
    _require(_words(frame["output_size"], 2, "output size") == output_size and output_size[0] % 2 == 0 and
             0 < min(output_size) <= max(output_size) <= 16384, "selected output extent differs")

    selected = _words(frame["selected_receipt"], 64, "selected receipt")
    contour = _words(frame["contour_identity"], 32, "contour identity")
    condition = _words(frame["condition_params"], 6, "condition parameters")
    render = _words(frame["subtitle_constants"], 8, "subtitle constants")
    proof = None if schema >= 4 else _words(frame["background_proof"], 96, "background proof")
    expected = None if schema >= 4 else _words(frame["background_expected"], 16, "background lease")
    values = dump_contract.validate_parallax_state_words(frame["parallax_state"], raw_coordinate_scale=scale)
    state_clock = values["gain_last_observation_low"] | values["gain_last_observation_high"] << 32
    _require(state_clock == clock, "ParallaxState belongs to a different owner clock")
    _require(render == [1, 0, 0, 0, 0, 0, width, height], "draw b3 does not bind the current full-source SDR frame")
    _require(selected[:2] == [0x31534D53, 1] and selected[2] in {1, 3} and selected[3] in {0, 1} and
             selected[62:] == [2, 0] and _join(selected, 60) == token, "selected receipt ABI/token is invalid")
    _require(_join(selected, 16) == ordinal and _join(selected, 18) == 0 and _join(selected, 20) == clock,
             "selected receipt frame/generation/clock differs")
    policy = coordinate.load_contract()["subtitle_ocr"]["contour_policy"]
    crop = min(height, (width * coordinate.SUBTITLE_OCR.source_crop_aspect_height +
                       coordinate.SUBTITLE_OCR.source_crop_aspect_width - 1) //
               coordinate.SUBTITLE_OCR.source_crop_aspect_width)
    _require(selected[4:8] == [width, height, height - crop, crop] and
             dump_contract.capture_grid_relation_for_source(width, height, selected[8], selected[9]) is not None and
             tuple(selected[10:12]) == coordinate.subtitle_ocr_dynamic_roi(width, height, selected[8], selected[9]) and
             selected[12:16] == [0, 0, selected[8], selected[9]], "selected receipt source/field geometry differs")
    if schema >= 2:
        mask_metrics.load_selected_mask(frame["selected_mask"], output, source_size, source_id,
                                        [0, height - crop, width, height])
    _require(selected[22] in {0, 1} and selected[23] == 0 and
             selected[24:28] == [1, policy["recipe"], 0, 0], "selected receipt options are invalid")
    if selected[3] == 0:
        _require(selected[28:60] == contour and contour[:2] == [policy["history_schema"], policy["history_tag"]] and
                 contour[2] in {0, 1} and contour[3] == values["confirmed_cut_count"] and
                 contour[4:8] == selected[16:20] and contour[8:20] == selected[4:16] and
                 contour[20:22] == selected[20:22] and contour[22:24] == [policy["recipe"], 1] and
                 contour[24:] == [0] * 8, "selected receipt/SCT owner join is invalid")
    else:
        _require(selected[2] == 1 and selected[28:60] == [0] * 32,
                 "unavailable selected receipt grants current authority")
    plane = struct.unpack("<f", struct.pack("<I", condition[5]))[0]
    plane_limit = struct.unpack("<f", struct.pack("<f", coordinate.CALIBRATED_DEFAULTS.direct_container_limit))[0]
    ocr = coordinate.SUBTITLE_OCR
    _require(condition[:2] == [ocr.condition_param_schema, ocr.condition_param_tag] and
             condition[2] <= ocr.plane_cover_capacity and condition[3] < (1 << condition[2]) and
             condition[4] in {0, ocr.condition_param_fully_active_value} and
             math.isfinite(plane) and 0 <= plane <= plane_limit,
             "condition parameters are invalid")
    _require((condition[4] == 0 and condition[2:] == [0] * 4) or
             (condition[4] == ocr.condition_param_fully_active_value and condition[2] > 0),
             "condition parameters are not canonical active/inactive data")
    if schema >= 4:
        serial, rendered_active = validate_shared_warp(frame, width, height, crop)
    else:
        _require(proof[:2] == [0x31504742, 1] and proof[2] in {0, 1, 2} and proof[3] <= 31 and
                 proof[4:20] == expected and proof[62:64] == [1, 0] and proof[83:] == [0] * 13 and
                 _join(expected, 0) == serial and all(_join(expected, offset) for offset in range(0, 16, 2)),
                 "background proof/immutable lease is invalid")
        if proof[2] in {1, 2}:
            _require(selected[2:4] == [3, 0] and proof[3] == 0 and proof[20:52] == contour and
                     proof[52:60] == render and _join(proof, 60) == token and
                     proof[64:72] == [width, height, width, crop, height - crop, 0, 0, 0] and
                     proof[77] == width * crop, "background proof source/owner join differs")
            if proof[2] == 1:
                _require(contour[2] == 1 and condition[2] > 0 and condition[4] == 2 and
                         proof[80] == proof[77] and proof[81] == 1 and
                         all(value <= proof[77] for value in proof[72:77] + proof[82:83]),
                         "background proof did not seal its complete current native field")
            else:
                _require(contour[2] == 0 and condition[2:5] == [0, 0, 0] and proof[80:83] == [0, 0, 0],
                         "empty background disposition grants pixels")
        else:
            _require(proof[3] != 0, "unavailable background disposition has no rejection reason")

    publication = frame["publication_receipt"]
    _closed(publication, {"estimator_generation", "publication_sequence", "domain_tag", "expected_work", "depth",
                          "depth_owner_frame_id", "depth_owner_timestamp_us", "subtitle", "subtitle_frame_id",
                          "subtitle_domain_tag"}, "publication receipt")
    for key, value in publication.items():
        _uint(value, f"publication {key}")
    generation = _uint(publication["estimator_generation"], "estimator generation", positive=True)
    sequence = _uint(publication["publication_sequence"], "publication sequence", positive=True)
    domain = _uint(publication["domain_tag"], "publication domain", positive=True)
    _require(publication["expected_work"] in {1, 2} and publication["depth"] == 2 and
             publication["depth_owner_frame_id"] == ordinal and publication["depth_owner_timestamp_us"] == clock and
             publication["subtitle"] == publication["expected_work"] and publication["subtitle_frame_id"] == ordinal and
             publication["subtitle_domain_tag"] == domain, "publication receipt is stale, suppressed or not inferred")
    stage_current = validate_producer_stages(frame, output) if schema >= 3 else False
    return generation, sequence, serial, selected[2], (rendered_active if schema >= 4 else proof[2]), stage_current


def validate_qualification(directory, frames_directory, executable, timeline=None):
    output = Path(directory)
    path = output / MANIFEST
    _require(path.is_file() and path.stat().st_size <= 16 * 1024 * 1024, "qualification manifest is absent/oversized")
    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    root_fields = {"schema", "status", "role", "ordinary_metric_artifacts", "live_capture", "scheduling_authority",
                       "source_transfer", "frame_count", "executable_sha256", "coordinate_contract_schema",
                       "coordinate_contract_sha256", "producer_source_closure_sha256",
                       "selector_source_closure_sha256", "observation_timeline",
                       "model", "model_url", "onnx_sha256", "raw_coordinate_scale", "frames"}
    schema = document.get("schema") if isinstance(document, dict) else None
    if type(schema) is not int or schema < 5:
        root_fields.add("completed_source_closure_sha256")
    root_fields.add("shared_warp_source_closure_sha256" if type(schema) is int and
                    schema >= 4 else "background_source_closure_sha256")
    if type(schema) is int and schema >= 3:
        root_fields.add("composition_source_closures")
        root_fields.add("hole_program_ready")
        root_fields.add("coverage_program_ready")
    if type(schema) is int and schema >= 6:
        root_fields.add("foreground_edge_program_ready")
    _closed(document, root_fields, "qualification manifest")
    _require(type(document["schema"]) is int and document["schema"] in {1, 2, 3, 4, 5, 6} and
             document["status"] == "COMPLETE" and document["role"] == ROLE and
             document["ordinary_metric_artifacts"] == "unchanged-legacy-native-matte-renderer" and
             document["live_capture"] is False and document["source_transfer"] == "encoded-srgb" and
             document["scheduling_authority"] == "ordered-authored-full-source-observations",
             "qualification scope/version is unsupported")
    _require(document["executable_sha256"] == _sha(executable), "qualification executable differs")
    _require(document["coordinate_contract_schema"] == coordinate.CONTRACT_SCHEMA and
             document["coordinate_contract_sha256"] == coordinate.CONTRACT_CANONICAL_SHA256 and
             document["producer_source_closure_sha256"] == coordinate.SHADER_IMPLEMENTATION.source_closure_sha256,
             "coordinate/producer contract differs")
    for key, digest in expected_closures(document["schema"]).items():
        _require(document[key] == digest, f"{key} differs from current native sources")
    if document["schema"] >= 3:
        _require(document["hole_program_ready"] is True, "current mask-hole programs were unavailable")
        _require(document["coverage_program_ready"] is True, "current stroke-coverage programs were unavailable")
        _require(document["composition_source_closures"] == expected_composition_closures(document["schema"]),
                 "producer stage composition source closures differ")
    if document["schema"] >= 6:
        _require(document["foreground_edge_program_ready"] is True, "current foreground-edge program was unavailable")
    calibrations = [item for item in coordinate.MODEL_CALIBRATIONS if item.depth_model == document["model"] and
                    item.depth_model_url == document["model_url"] and item.onnx_sha256 == document["onnx_sha256"] and
                    item.raw_coordinate_scale == document["raw_coordinate_scale"]]
    _require(calibrations, "model calibration identity differs")
    binding = document["observation_timeline"]
    _require(isinstance(binding, list) and len(binding) == 2 and binding[0] == "current-outline/observation.timeline",
             "timeline binding is malformed")
    copied_timeline = _within(output, binding[0])
    _require(binding[1] == _sha(copied_timeline), "timeline hash differs")
    clocks = read_observation_timeline(copied_timeline)
    if timeline is not None:
        _require(_sha(timeline) == binding[1] and read_observation_timeline(Path(timeline)) == clocks,
                 "qualification timeline differs from requested observations")
    sources = source_frames(frames_directory)
    _require(type(document["frame_count"]) is int and document["frame_count"] == len(sources) == len(clocks) and
             isinstance(document["frames"], list) and len(document["frames"]) == len(sources),
             "qualification source/observation coverage is incomplete")
    outcomes = [_validate_frame(frame, ordinal, source, clock, output, document["raw_coordinate_scale"], document["schema"])
                for ordinal, (frame, source, clock) in enumerate(zip(document["frames"], sources, clocks), 1)]
    _require(len({item[0] for item in outcomes}) == 1 and
             all(later[1] > earlier[1] and later[2] > earlier[2] for earlier, later in zip(outcomes, outcomes[1:])),
             "qualification producer/sequence/lease lineage is inconsistent")
    expected_outputs = {frame["output_file"] for frame in document["frames"]}
    actual_outputs = {path.relative_to(output).as_posix() for path in (output / "current-outline").glob("sbs_*.png")}
    _require(actual_outputs == expected_outputs, "selected output frame set differs")
    if document["schema"] >= 2:
        masks = {frame["selected_mask"]["file"] for frame in document["frames"]}
        actual_masks = {path.relative_to(output).as_posix() for path in (output / "current-outline").glob("selected_matte_*.u32")}
        _require(masks == actual_masks, "selected mask frame set differs")
    if document["schema"] >= 3:
        stages = {binding["file"] for frame in document["frames"] for binding in frame["producer_stages"]["buffers"].values()}
        actual_stages = {path.relative_to(output).as_posix() for path in (output / "current-outline/producer-stages").rglob("*")
                         if path.is_file()}
        _require(stages == actual_stages, "producer stage artifact coverage differs")
    return {"schema": 1, "status": "PASS", "frame_count": len(outcomes),
            "renderer_schema": document["schema"], "renderer_policy_current": document["schema"] == 6,
            "selected_current_frames": sum(item[3] == 3 for item in outcomes),
            "background_valid_frames": sum(item[4] == 1 for item in outcomes) if document["schema"] < 4 else 0,
            "shared_warp_frames": len(outcomes) if document["schema"] >= 4 else 0,
            "shared_warp_active_frames": sum(item[4] == 1 for item in outcomes) if document["schema"] >= 4 else 0,
            "scope": ROLE, "selected_mask_evidence_available": document["schema"] >= 2,
            "producer_stage_evidence_available": document["schema"] >= 3,
            "producer_stage_current_frames": sum(item[5] for item in outcomes),
            "ordinary_metrics_qualify_selected_composite": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "cmake-build-relwithdebinfo")
    parser.add_argument("--conf", type=Path, default=Path(__file__).with_name("bench.conf"))
    parser.add_argument("--observation-timeline", type=Path)
    parser.add_argument("--fps", default="30/1", help="authored cadence when no explicit timeline is supplied")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        executable = (args.build_dir / "sunshine.exe").resolve()
        sources = source_frames(args.frames)
        timeline = args.observation_timeline
        if not args.validate_only:
            _require(executable.is_file(), "build sunshine before qualification")
            _require(not args.out.exists() or (args.out.is_dir() and not any(args.out.iterdir())),
                     "qualification output must be absent or empty")
            args.out.mkdir(parents=True, exist_ok=True)
            if timeline is None:
                timeline = args.out / "requested_observation.timeline"
                write_observation_timeline(timeline, prepared_observation_timestamps(len(sources), args.fps))
            _require(len(read_observation_timeline(timeline)) == len(sources), "requested timeline count differs")
            before = {path.name: _sha(path) for path in sources}
            truth_before = mask_metrics.truth_input_identity(args.frames, sources)
            binary = _sha(executable)
            command = [str(executable), str(args.conf.resolve()), "--sbs-bench", "--frames", str(args.frames.resolve()),
                       "--out", str(args.out.resolve()), "--observation-timeline", str(timeline.resolve()),
                       "--current-outline-qualification"]
            subprocess.run(command, cwd=args.build_dir.resolve(), check=True)
            _require(binary == _sha(executable) and before == {path.name: _sha(path) for path in sources},
                     "qualification source or executable changed during capture")
            _require(truth_before == mask_metrics.truth_input_identity(args.frames, sources),
                     "source metadata/tight overlay GT changed during capture")
        result = validate_qualification(args.out, args.frames, executable, timeline)
        scores = mask_metrics.score_qualification(args.out, args.frames, executable, timeline)
        result["mask_diagnostics"] = scores["aggregate"]
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, TimelineError, subprocess.CalledProcessError) as error:
        print(f"current-outline qualification: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
