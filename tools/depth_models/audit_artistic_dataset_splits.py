#!/usr/bin/env python3
"""Freeze and validate the active complete-production artistic data split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import artistic_sources as sources


DEFAULT_CATALOG = Path(__file__).with_name(
    sources.DEFAULT_ACTIVE_CATALOG_NAME
)
SEQUENCE_SOURCE_CONTAINERS = {
    "image-sequence-archives",
    "derived-public-image-sequences",
}
NATIVE_HDR_SOURCE_KIND = "native-hdr-video"
NATIVE_HDR_COLLECTION_IDENTITY_KIND = "native_hdr_video_collection_sha256"


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_hash(value, label):
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise RuntimeError(f"invalid {label}")
    return value


def _verified_provenance_file(payload, dataset_path: Path, name, production):
    provenance = payload.get("source_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(f"{production}: native HDR source provenance is missing")
    value = provenance.get(name)
    expected = _required_hash(
        provenance.get(f"{name}_sha256"),
        f"{production} {name}_sha256",
    )
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{production}: native HDR {name} is missing")
    path = Path(value)
    if not path.is_absolute():
        path = dataset_path.parent / path
    path = path.resolve()
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"{production}: native HDR {name} is missing or changed")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{production}: native HDR {name} is unreadable") from error
    if not isinstance(document, dict) or not sources.schema_is(document.get("schema"), 1):
        raise RuntimeError(f"{production}: native HDR {name} has unsupported schema")
    return path, expected, document


def _unique_rows(rows, key, label, production):
    if not isinstance(rows, list):
        raise RuntimeError(f"{production}: native HDR {label} is missing")
    result = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"{production}: native HDR {label} row {index} is invalid")
        identity = row.get(key)
        if not isinstance(identity, str) or not identity or identity in result:
            raise RuntimeError(f"{production}: native HDR {label} has duplicate identity")
        result[identity] = row
    return result


def native_hdr_source_identity(payload, dataset_path: Path, production,
                               verify_media=False):
    """Authenticate one aggregate native-HDR production without a fake video hash.

    The dataset manifest pins the selection and download-receipt files.  Its
    retained sequences select the exact source videos and capture groups.  The
    resulting semantic collection identity is stable across local path changes,
    while ``verify_media`` lets the split freezer hash every selected master
    once before publishing the active split.
    """
    if (payload.get("source_kind") != NATIVE_HDR_SOURCE_KIND or
            not sources.schema_is(payload.get("schema"), 2)):
        raise RuntimeError(f"{production}: not a native HDR collection")
    split = payload.get("split")
    if split not in sources.SPLITS:
        raise RuntimeError(f"{production}: native HDR split is invalid")
    selection_path, selection_hash, selection = _verified_provenance_file(
        payload, dataset_path, "selection_manifest", production
    )
    receipt_path, receipt_hash, receipt = _verified_provenance_file(
        payload, dataset_path, "download_receipt", production
    )
    selected = _unique_rows(selection.get("clips"), "video_id",
                            "selection clips", production)
    accepted = _unique_rows(receipt.get("accepted"), "video_id",
                            "download receipt", production)
    sequences = payload.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise RuntimeError(f"{production}: native HDR sequences are missing")
    sequence_videos = {}
    clips = set()
    total_frames = 0
    total_labels = 0
    frame_rates = set()
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, dict):
            raise RuntimeError(f"{production}: native HDR sequence {index} is invalid")
        video_id = sequence.get("video_id")
        capture_group = sequence.get("capture_group_id")
        clip = sequence.get("clip")
        if (not isinstance(video_id, str) or not video_id or
                not isinstance(capture_group, str) or not capture_group or
                not isinstance(clip, str) or not clip or clip in clips or
                sequence.get("split") != split):
            raise RuntimeError(f"{production}: native HDR sequence identity is invalid")
        clips.add(clip)
        previous_group = sequence_videos.setdefault(video_id, capture_group)
        if previous_group != capture_group:
            raise RuntimeError(f"{production}: native HDR video crosses capture groups")
        try:
            frames = int(sequence.get("frames", 0))
            labels = int(sequence.get("label_frames", 0))
            frame_rate = float(sequence.get("source_frame_rate", 0.0))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{production}: native HDR sequence counts are invalid") from error
        if (frames <= 0 or labels <= 0 or not math.isfinite(frame_rate) or
                frame_rate <= 0.0):
            raise RuntimeError(f"{production}: native HDR sequence cadence is invalid")
        total_frames += frames
        total_labels += labels
        frame_rates.add(frame_rate)
    expected_video_count = payload.get("source_video_count")
    if (isinstance(expected_video_count, bool) or
            not isinstance(expected_video_count, int) or
            expected_video_count != len(sequence_videos)):
        raise RuntimeError(f"{production}: native HDR source-video count disagrees")
    if (payload.get("source_frame_count") != total_frames or
            payload.get("frame_count") != total_frames or
            payload.get("label_frame_count") != total_labels):
        raise RuntimeError(f"{production}: native HDR retained-frame counts disagree")

    records = []
    capture_groups = set()
    for video_id, capture_group in sorted(sequence_videos.items()):
        selection_row = selected.get(video_id)
        receipt_row = accepted.get(video_id)
        download = receipt_row.get("download") if isinstance(receipt_row, dict) else None
        if (not isinstance(selection_row, dict) or
                selection_row.get("split") != split or
                selection_row.get("capture_group_id") != capture_group or
                not isinstance(receipt_row, dict) or
                receipt_row.get("split") != split or
                receipt_row.get("capture_group_id") != capture_group or
                not isinstance(download, dict) or
                download.get("video_id") != video_id):
            raise RuntimeError(
                f"{production}: native HDR source {video_id} disagrees with provenance"
            )
        video_hash = _required_hash(
            download.get("sha256"), f"{production} source video sha256"
        )
        if capture_group in capture_groups:
            raise RuntimeError(f"{production}: native HDR capture group is repeated")
        capture_groups.add(capture_group)
        if verify_media:
            media = Path(download.get("path", ""))
            if not media.is_absolute():
                media = receipt_path.parent / media
            media = media.resolve()
            if not media.is_file() or sha256(media) != video_hash:
                raise RuntimeError(
                    f"{production}: native HDR source video is missing or changed: {media}"
                )
        records.append({
            "video_id": video_id,
            "video_sha256": video_hash,
            "capture_group_id": capture_group,
        })
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "source_identity_kind": NATIVE_HDR_COLLECTION_IDENTITY_KIND,
        "source_collection_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "source_videos": records,
        "source_video_ids": [row["video_id"] for row in records],
        "source_capture_group_ids": [row["capture_group_id"] for row in records],
        "source_provenance": {
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": selection_hash,
            "download_receipt": str(receipt_path),
            "download_receipt_sha256": receipt_hash,
        },
        "context_frames": total_frames,
        "context_frame_rates": sorted(frame_rates),
        "label_frames": total_labels,
        "shots": len(capture_groups),
    }


def source_identity(payload, path: Path, manifest_schema, production):
    expected_hash = payload.get("video_sha256")
    if (sources.schema_is(manifest_schema, 2) and
            payload.get("source_container") in SEQUENCE_SOURCE_CONTAINERS):
        relative_name = payload.get("source_sequence_manifest")
        if not isinstance(relative_name, str) or not relative_name.strip():
            raise RuntimeError(
                f"{production}: source sequence manifest is missing"
            )
        relative_path = Path(relative_name)
        if relative_path.is_absolute():
            raise RuntimeError(
                f"{production}: source sequence manifest must be relative"
            )
        dataset_root = path.parent.resolve()
        sequence_manifest = (dataset_root / relative_path).resolve()
        try:
            sequence_manifest.relative_to(dataset_root)
        except ValueError as error:
            raise RuntimeError(
                f"{production}: source sequence manifest escapes dataset root"
            ) from error
        if not sequence_manifest.is_file() or not expected_hash:
            raise RuntimeError(
                f"{production}: source sequence manifest or hash is missing"
            )
        current_hash = sha256(sequence_manifest)
        if current_hash != expected_hash:
            raise RuntimeError(
                f"{production}: source sequence manifest hash changed"
            )
        return current_hash

    video = Path(payload.get("video", ""))
    if not video.is_absolute():
        video = path.parent / video
    video = video.resolve()
    if not video.is_file() or not expected_hash:
        raise RuntimeError(f"{production}: source video or hash is missing")
    current_hash = sha256(video)
    if current_hash != expected_hash:
        raise RuntimeError(f"{production}: source video hash changed")
    return current_hash


def audit(catalog_path: Path, manifest_paths):
    catalog = sources.load_catalog(catalog_path)
    by_production = {
        item["production_id"]: item for item in catalog["sources"]
        if (item.get("production_id") and
            item.get("admission") == "global_policy")
    }
    productions = {}
    rows = []
    native_video_ids = {}
    source_video_hashes = {}
    native_capture_groups = {}
    for path in manifest_paths:
        path = path.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest_schema = (
            payload.get("schema") if isinstance(payload, dict) else None
        )
        if sources.schema_is(manifest_schema, 1):
            production = payload.get("film_id")
            label_frames = int(payload.get("sample_count", 0))
            manifest_kind = "authored-stereo"
        elif sources.schema_is(manifest_schema, 2):
            production = payload.get("production_id")
            label_frames = int(payload.get("label_frame_count", 0))
            manifest_kind = payload.get("source_kind")
            if manifest_kind not in sources.SOURCE_KINDS:
                raise RuntimeError(
                    f"{path}: missing or invalid source_kind"
                )
        else:
            raise RuntimeError(f"unsupported dataset manifest: {path}")
        split = payload.get("split")
        if not production or split not in sources.SPLITS:
            raise RuntimeError(
                f"{path}: missing production identity or valid split"
            )
        if production in productions:
            raise RuntimeError(
                f"production {production!r} appears in multiple active manifests"
            )
        source = by_production.get(production)
        if source is None or source.get("admission") != "global_policy":
            raise RuntimeError(f"{production}: not an admitted catalog production")
        if source["source_kind"] != manifest_kind:
            raise RuntimeError(
                f"{production}: catalog source kind {source['source_kind']} != "
                f"dataset {manifest_kind}"
            )
        if source["split"] != split:
            raise RuntimeError(
                f"{production}: catalog split {source['split']} != dataset {split}"
            )
        expected_weight = float(source["global_policy_weight"])
        actual_weight = float(payload.get("global_policy_weight", 1.0))
        if (not math.isfinite(actual_weight) or actual_weight <= 0.0 or
                abs(expected_weight - actual_weight) > 1e-9):
            raise RuntimeError(
                f"{production}: catalog weight {expected_weight} != "
                f"dataset {actual_weight}"
            )
        productions[production] = split
        native_identity = None
        if manifest_kind == NATIVE_HDR_SOURCE_KIND:
            native_identity = native_hdr_source_identity(
                payload, path, production, verify_media=True
            )
            context_frames = native_identity["context_frames"]
            label_frames = native_identity["label_frames"]
        else:
            context_frames = int(payload.get("context_frame_count", 0))
            context_fps = float(payload.get(
                "context_fps", payload.get("source_fps", 0.0)
            ))
            if (context_frames <= 0 or not math.isfinite(context_fps) or
                    context_fps <= 0.0):
                raise RuntimeError(
                    f"{production}: dataset predates full-cadence context"
                )
        if context_frames <= 0 or label_frames <= 0:
            raise RuntimeError(f"{production}: dataset has no usable frames")
        active_row = {
            "production_id": production,
            "source_id": source["id"],
            "source_kind": source["source_kind"],
            "source_group": source["source_group"],
            "split": split,
            "global_policy_weight": actual_weight,
            "dataset_manifest": str(path),
            "dataset_manifest_schema": manifest_schema,
            "dataset_manifest_sha256": sha256(path),
            "context_frames": context_frames,
            "label_frames": label_frames,
        }
        if native_identity is not None:
            active_row.update(native_identity)
            for record in native_identity["source_videos"]:
                for observed, key, value in (
                    (native_video_ids, "source video id", record["video_id"]),
                    (source_video_hashes, "source video hash", record["video_sha256"]),
                    (native_capture_groups, "capture group", record["capture_group_id"]),
                ):
                    previous = observed.setdefault(value, production)
                    if previous != production:
                        raise RuntimeError(
                            f"native HDR {key} appears in multiple productions: "
                            f"{previous!r}, {production!r}"
                        )
        else:
            current_video_hash = source_identity(
                payload, path, manifest_schema, production
            )
            previous = source_video_hashes.setdefault(
                current_video_hash, production
            )
            if previous != production:
                raise RuntimeError(
                    "source video identity appears in multiple productions: "
                    f"{previous!r}, {production!r}"
                )
            active_row.update({
                "video_sha256": current_video_hash,
                "context_fps": context_fps,
                "shots": int(payload.get("shot_count", 0)),
            })
        if (native_identity is None and
                sources.schema_is(manifest_schema, 2) and
                payload.get("source_container") in
                SEQUENCE_SOURCE_CONTAINERS):
            active_row.update({
                "source_identity_kind":
                    "source_sequence_manifest_sha256",
                "source_sequence_manifest_sha256": current_video_hash,
            })
        elif native_identity is None:
            active_row["source_identity_kind"] = "video_sha256"
        rows.append(active_row)
    split_productions = {
        split: sorted(
            production for production, assigned in productions.items()
            if assigned == split
        )
        for split in sorted(sources.SPLITS)
    }
    if not split_productions["training"]:
        raise RuntimeError("active split has no training production")
    if not split_productions["development"]:
        raise RuntimeError("active split has no development production")
    if len(split_productions["test"]) < 2:
        raise RuntimeError("active split needs two sealed test productions")
    test_groups = {
        row["source_group"] for row in rows if row["split"] == "test"
    }
    if len(test_groups) < 2:
        raise RuntimeError("sealed tests need two independent source groups")
    return {
        "schema": 1,
        "catalog": str(catalog_path.resolve()),
        "catalog_sha256": sha256(catalog_path),
        "productions": rows,
        "split_productions": split_productions,
        "totals": {
            "productions": len(rows),
            "context_frames": sum(row["context_frames"] for row in rows),
            "label_frames": sum(row["label_frames"] for row in rows),
            "shots": sum(row["shots"] for row in rows),
        },
        "split_rule": "complete productions are immutable across splits",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog", type=Path,
        default=DEFAULT_CATALOG,
    )
    parser.add_argument(
        "--dataset-manifest", action="append", type=Path, required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.catalog, args.dataset_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
