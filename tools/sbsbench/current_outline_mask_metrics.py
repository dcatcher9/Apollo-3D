#!/usr/bin/env python3
"""Diagnostic source-coordinate scoring of the literal GPU selected mask.

No stereo warp or final-RGB retention is reconstructed here. The mask is placed at its
authenticated native source crop without filtering; outside that crop it grants no protection.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
from pathlib import Path

import numpy as np
import PIL
from PIL import Image


REPORT = "current_outline_mask_scores.json"
GT_DIRECTORY = "gt_subtitle_overlay_mask"
MASK_FORMAT = "r32_uint_little_endian_0_or_1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selected_mask(binding, output, source_size, source_id, source_rect):
    """Validate/decode exact little-endian R32_UINT bytes and their source crop owner."""
    _require(isinstance(binding, dict) and set(binding) == {
        "file", "sha256", "size", "source_rect", "byte_count", "format",
    }, "selected mask binding has unknown or missing fields")
    width, height = source_size
    left, top, right, bottom = source_rect
    expected_size = [right - left, bottom - top]
    _require(binding["file"] == f"current-outline/selected_matte_{source_id}.u32" and
             binding["format"] == MASK_FORMAT, "selected mask role/format differs")
    for key, expected in (("size", expected_size), ("source_rect", list(source_rect))):
        values = binding[key]
        _require(isinstance(values, list) and len(values) == len(expected) and
                 all(type(value) is int for value in values) and values == expected,
                 f"selected mask {key} differs from its source owner")
    _require(0 <= left < right <= width and 0 <= top < bottom <= height,
             "selected mask source crop is invalid")
    byte_count = expected_size[0] * expected_size[1] * 4
    _require(type(binding["byte_count"]) is int and binding["byte_count"] == byte_count,
             "selected mask byte count differs")
    output = Path(output).resolve()
    path = (output / binding["file"]).resolve()
    _require(path.is_relative_to(output) and path.is_file(), "selected mask artifact escapes/is absent")
    payload = path.read_bytes()
    _require(len(payload) == byte_count and hashlib.sha256(payload).hexdigest() == binding["sha256"],
             "selected mask artifact size/hash differs")
    values = np.frombuffer(payload, dtype="<u4").reshape(expected_size[1], expected_size[0])
    _require(np.all((values == 0) | (values == 1)), "selected mask artifact is not binary uint32")
    source_mask = np.zeros((height, width), dtype=bool)
    source_mask[top:bottom, left:right] = values != 0
    return source_mask


def ground_truth_bindings(frames_directory, sources):
    """Optional GT is all-or-none with exact source filename, dimensions and L0/255 data."""
    directory = Path(frames_directory) / GT_DIRECTORY
    if not directory.exists():
        return None
    _require(directory.is_dir(), "tight overlay GT is not a directory")
    images = {path.name for path in directory.iterdir() if path.is_file() and
              path.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    _require(images == {source.name for source in sources}, "tight overlay GT/source identities differ")
    result = []
    for source in sources:
        path = directory / source.name
        with Image.open(source) as image:
            source_size = image.size
        payload = path.read_bytes()
        with Image.open(io.BytesIO(payload)) as image:
            _require(image.format == "PNG" and image.mode == "L" and image.size == source_size,
                     "tight overlay GT must be source-sized 8-bit L PNG")
            values = np.asarray(image)
            _require(np.all((values == 0) | (values == 255)), "tight overlay GT is not binary 0/255")
        result.append({"file": f"{GT_DIRECTORY}/{source.name}", "sha256": hashlib.sha256(payload).hexdigest(),
                       "size": list(source_size)})
    return result


def truth_input_identity(frames_directory, sources):
    metadata = Path(frames_directory) / "meta.json"
    binding = None
    required = False
    if metadata.exists():
        _require(metadata.is_file(), "source metadata is not a file")
        def unique(pairs):
            result = {}
            for key, value in pairs:
                _require(key not in result, "source metadata has duplicate fields")
                result[key] = value
            return result
        payload = metadata.read_bytes()
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=unique)
        _require(isinstance(document, dict), "source metadata is not a JSON object")
        for flag in ("required_gt_subtitle_tight_mask", "required_gt_subtitle_sanitizer_oracle"):
            if flag in document:
                _require(type(document[flag]) is bool, f"source metadata {flag} is not boolean")
                required |= document[flag]
        binding = {"file": "meta.json", "sha256": hashlib.sha256(payload).hexdigest()}
    truth = ground_truth_bindings(frames_directory, sources)
    _require(not required or truth is not None, "source metadata requires missing tight overlay GT")
    return {"metadata": binding, "tight_masks": truth}


def count_mask(prediction, ground_truth):
    _require(prediction.dtype == ground_truth.dtype == np.dtype(bool) and prediction.shape == ground_truth.shape,
             "mask arithmetic requires matched binary arrays")
    return {"predicted_pixels": int(np.count_nonzero(prediction)),
            "ground_truth_pixels": int(np.count_nonzero(ground_truth)),
            "true_positive_pixels": int(np.count_nonzero(prediction & ground_truth)),
            "false_positive_pixels": int(np.count_nonzero(prediction & ~ground_truth)),
            "false_negative_pixels": int(np.count_nonzero(~prediction & ground_truth))}


def ratios(counts):
    predicted, truth, true = (counts[name] for name in (
        "predicted_pixels", "ground_truth_pixels", "true_positive_pixels"))
    union = true + counts["false_positive_pixels"] + counts["false_negative_pixels"]
    return {"precision": true / predicted if predicted else None,
            "recall": true / truth if truth else None,
            "iou": true / union if union else None}


def count_temporal(previous_mask, current_mask, previous_rgb, current_rgb, previous_gt, current_gt):
    _require(all(value.dtype == np.dtype(bool) for value in (previous_mask, current_mask, previous_gt, current_gt)) and
             previous_rgb.dtype == current_rgb.dtype == np.dtype("uint8") and
             previous_mask.shape == current_mask.shape == previous_gt.shape == current_gt.shape and
             previous_rgb.shape == current_rgb.shape == (*current_mask.shape, 3),
             "temporal mask arithmetic requires matched source rasters")
    stable = np.all(previous_rgb == current_rgb, axis=2) & (previous_gt == current_gt)
    changed = previous_mask != current_mask
    result = {}
    for prefix, eligible in (("all", stable), ("glyph", stable & current_gt), ("background", stable & ~current_gt)):
        result[f"{prefix}_eligible_pixels"] = int(np.count_nonzero(eligible))
        result[f"{prefix}_changed_pixels"] = int(np.count_nonzero(changed & eligible))
    return result


def temporal_ratios(counts):
    return {f"{prefix}_flicker_fraction": counts[f"{prefix}_changed_pixels"] / counts[f"{prefix}_eligible_pixels"]
            if counts[f"{prefix}_eligible_pixels"] else None for prefix in ("all", "glyph", "background")}


def score_qualification(output, frames_directory, executable, timeline=None, write=True):
    # Import only at this independent entry point; the qualifier also uses the literal decoder.
    import qualify_current_outline as qualification

    output = Path(output)
    frames_directory = Path(frames_directory)
    manifest_path = output / qualification.MANIFEST
    _require(manifest_path.is_file() and manifest_path.stat().st_size <= 16 * 1024 * 1024,
             "qualification manifest is absent/oversized")
    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    scorer_digest, reader_digest = sha256(Path(__file__)), sha256(Path(qualification.__file__))
    verified = qualification.validate_qualification(output, frames_directory, executable, timeline)
    _require(sha256(manifest_path) == manifest_digest, "qualification manifest changed during validation")
    manifest = json.loads(manifest_bytes.decode("utf-8"), object_pairs_hook=qualification._unique_object)
    sources = qualification.source_frames(frames_directory)
    truth_identity = truth_input_identity(frames_directory, sources)
    truth_bindings = truth_identity["tight_masks"]
    has_masks = manifest["schema"] >= 2
    aggregate_counts = {key: 0 for key in (
        "predicted_pixels", "ground_truth_pixels", "true_positive_pixels", "false_positive_pixels", "false_negative_pixels")}
    aggregate_temporal = {f"{prefix}_{kind}_pixels": 0 for prefix in ("all", "glyph", "background")
                          for kind in ("eligible", "changed")}
    scored, empty_frames, empty_false, temporal_pairs = 0, 0, 0, 0
    previous = None
    frames = []
    for index, (source, frame) in enumerate(zip(sources, manifest["frames"])):
        item = {"source_file": source.name, "source_snapshot_sha256": frame["source_snapshot_sha256"],
                "observation_timestamp_us": frame["observation_timestamp_us"],
                "selected_kind": frame["selected_receipt"][2], "selected_mask": frame.get("selected_mask"),
                "producer_stages": frame.get("producer_stages"),
                "ground_truth": truth_bindings[index] if truth_bindings is not None else None,
                "counts": None, "precision": None, "recall": None, "iou": None,
                "empty_false_protected_pixels": None, "temporal": None}
        if has_masks and truth_bindings is not None:
            selected = frame["selected_receipt"]
            prediction = load_selected_mask(frame["selected_mask"], output, frame["source_size"],
                                            frame["source_file_id"], [0, selected[6], selected[4], selected[5]])
            truth_bytes = (frames_directory / truth_bindings[index]["file"]).read_bytes()
            source_bytes = source.read_bytes()
            _require(hashlib.sha256(truth_bytes).hexdigest() == truth_bindings[index]["sha256"] and
                     hashlib.sha256(source_bytes).hexdigest() == frame["source_snapshot_sha256"],
                     "source/GT snapshot differs during scoring")
            with Image.open(io.BytesIO(truth_bytes)) as image:
                truth = np.asarray(image) == 255
            with Image.open(io.BytesIO(source_bytes)) as image:
                rgb = np.asarray(image.convert("RGB"))
            counts = count_mask(prediction, truth)
            item.update(counts=counts, **ratios(counts))
            scored += 1
            for key, value in counts.items():
                aggregate_counts[key] += value
            if counts["ground_truth_pixels"] == 0:
                empty_frames += 1
                empty_false += counts["predicted_pixels"]
                item["empty_false_protected_pixels"] = counts["predicted_pixels"]
            if previous is not None and previous[0].shape == prediction.shape:
                temporal = count_temporal(previous[0], prediction, previous[1], rgb, previous[2], truth)
                item["temporal"] = {**temporal, **temporal_ratios(temporal)}
                temporal_pairs += 1
                for key, value in temporal.items():
                    aggregate_temporal[key] += value
            previous = (prediction, rgb, truth)
        frames.append(item)
    aggregate = {"scored_frames": scored, "counts": aggregate_counts if scored else None,
                 **(ratios(aggregate_counts) if scored else {"precision": None, "recall": None, "iou": None}),
                 "empty_frames": empty_frames, "empty_false_protected_pixels": empty_false if empty_frames else None,
                 "temporal_pairs": temporal_pairs, "temporal":
                 {**aggregate_temporal, **temporal_ratios(aggregate_temporal)} if temporal_pairs else None}
    report = {"schema": 1, "status": "DIAGNOSTIC", "qualification": verified,
              "qualification_manifest_sha256": manifest_digest, "executable_sha256": manifest["executable_sha256"],
              "scorer_source_sha256": scorer_digest, "reader_source_sha256": reader_digest,
              "numeric_runtime": {"python_executable": str(Path(sys.executable).resolve()),
                                  "python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__},
              "observation_timeline": manifest["observation_timeline"],
              "coordinate_space": "full-source-literal-selected-mask-zero-outside-native-crop",
              "ground_truth_directory": GT_DIRECTORY if truth_bindings is not None else None,
              "source_truth_identity": truth_identity,
              "selected_mask_evidence_available": has_masks,
              "final_rgb_retention_qualified": False, "live_capture_qualified": False,
              "ordinary_metrics_migrated": False, "quality_gate": None,
              "aggregate": aggregate, "frames": frames}
    # Recheck independent immutable inputs after the numeric work, before publishing scores.
    _require(truth_input_identity(frames_directory, sources) == truth_identity,
             "source metadata/tight overlay GT changed during scoring")
    _require(report["qualification_manifest_sha256"] == sha256(manifest_path) and
             sha256(Path(__file__)) == scorer_digest and sha256(Path(qualification.__file__)) == reader_digest and
             sha256(executable) == manifest["executable_sha256"] and
             sha256(output / manifest["observation_timeline"][0]) == manifest["observation_timeline"][1] and
             all(sha256(source) == frame["source_snapshot_sha256"] for source, frame in zip(sources, frames)) and
             all(sha256(output / frame["output_file"]) == frame["output_sha256"] and
                 (not has_masks or sha256(output / frame["selected_mask"]["file"]) == frame["selected_mask"]["sha256"])
                 and all(sha256(output / binding["file"]) == binding["sha256"]
                         for binding in frame.get("producer_stages", {}).get("buffers", {}).values())
                 for frame in manifest["frames"]), "qualified source/artifact/manifest/executable changed during scoring")
    if write:
        temporary = output / (REPORT + ".partial")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output / REPORT)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--build-dir", type=Path, default=Path(__file__).resolve().parents[2] / "cmake-build-relwithdebinfo")
    args = parser.parse_args(argv)
    try:
        report = score_qualification(args.out, args.frames, args.build_dir / "sunshine.exe")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"selected mask diagnostics: {error}\n")
    print(json.dumps(report["aggregate"], sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
