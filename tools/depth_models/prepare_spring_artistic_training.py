#!/usr/bin/env python3
"""Extract complete Spring stereo sequences for artistic-policy training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath


FRAME_RE = re.compile(r"^frame_(left|right)_(\d+)\.png$", re.IGNORECASE)
SEQUENCE_RE = re.compile(r"^[0-9]+$")
DEFAULT_HOLDOUT = ("0028", "0035", "0042")


def file_hash(path: Path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_frames(archive: zipfile.ZipFile, eye: str):
    result = {}
    expected_dir = f"frame_{eye}"
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.is_dir() or len(path.parts) != 5:
            continue
        if path.parts[:2] != ("spring", "test") or path.parts[3] != expected_dir:
            continue
        sequence = path.parts[2]
        match = FRAME_RE.fullmatch(path.name)
        if (not match or match.group(1).lower() != eye
                or not SEQUENCE_RE.fullmatch(sequence)):
            continue
        result.setdefault(sequence, {})[int(match.group(2))] = info
    return result


def extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, output.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)


def prepare(left_archive_path: Path, right_archive_path: Path, output: Path,
            sequence_filter=None, holdout_sequences=DEFAULT_HOLDOUT):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    requested = set(sequence_filter or [])
    holdout = set(holdout_sequences)
    rows = []

    with (zipfile.ZipFile(left_archive_path) as left_archive,
          zipfile.ZipFile(right_archive_path) as right_archive):
        left = archive_frames(left_archive, "left")
        right = archive_frames(right_archive, "right")
        sequences = sorted(set(left) & set(right))
        if requested:
            missing = requested - set(sequences)
            if missing:
                raise RuntimeError(f"sequences not found in archives: {sorted(missing)}")
            sequences = [sequence for sequence in sequences if sequence in requested]
        if not sequences:
            raise RuntimeError("no paired Spring stereo sequences found")
        unknown_holdouts = holdout - set(sequences)
        if unknown_holdouts and not requested:
            raise RuntimeError(
                f"holdout sequences not found in archives: {sorted(unknown_holdouts)}"
            )

        for sequence in sequences:
            left_ids = set(left[sequence])
            right_ids = set(right[sequence])
            if left_ids != right_ids:
                raise RuntimeError(
                    f"{sequence}: left/right frame mismatch: "
                    f"left-only={sorted(left_ids - right_ids)}, "
                    f"right-only={sorted(right_ids - left_ids)}"
                )
            clip = output / f"spring_{sequence}"
            selection = []
            for output_index, source_id in enumerate(sorted(left_ids)):
                name = f"frame_{output_index:05d}.png"
                extract_member(left_archive, left[sequence][source_id], clip / name)
                extract_member(
                    right_archive, right[sequence][source_id], clip / "gt_right" / name
                )
                selection.append({
                    "output_frame": output_index,
                    "dataset_frame": source_id,
                    "source_left": left[sequence][source_id].filename,
                    "source_right": right[sequence][source_id].filename,
                })
            split = "validation" if sequence in holdout else "training"
            meta = {
                "schema": 1,
                "name": f"spring-{sequence}",
                "dataset": "Spring Benchmark",
                "domain": "spring",
                "homepage": "https://www.spring-benchmark.org/download",
                "citation": (
                    "Mehl et al., Spring: A High-Resolution High-Detail Dataset and "
                    "Benchmark for Scene Flow, CVPR 2023"
                ),
                "license": "CC BY 4.0",
                "license_scope": "Spring movie assets and standard clean dataset",
                "purpose": "artistic-policy geometric stereo supervision",
                "policy_role": "local_geometry",
                "global_policy_weight": 0.0,
                "sequence": sequence,
                "source_split": "test",
                "split": split,
                "required_gt_stereo": True,
                "selection": selection,
            }
            (clip / "meta.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
            rows.append({
                "clip": clip.name,
                "sequence": sequence,
                "frames": len(selection),
                "split": split,
            })

    manifest = {
        "schema": 1,
        "dataset": "Spring Benchmark",
        "domain": "spring",
        "homepage": "https://www.spring-benchmark.org/download",
        "license": "CC BY 4.0",
        "policy_role": "local_geometry",
        "left_archive": str(left_archive_path.resolve()),
        "left_archive_size": left_archive_path.stat().st_size,
        "left_archive_sha256": file_hash(left_archive_path),
        "right_archive": str(right_archive_path.resolve()),
        "right_archive_size": right_archive_path.stat().st_size,
        "right_archive_sha256": file_hash(right_archive_path),
        "holdout_sequences": sorted(holdout & {row["sequence"] for row in rows}),
        "sequences": rows,
        "sequence_count": len(rows),
        "frame_count": sum(row["frames"] for row in rows),
        "training_frames": sum(
            row["frames"] for row in rows if row["split"] == "training"
        ),
        "validation_frames": sum(
            row["frames"] for row in rows if row["split"] == "validation"
        ),
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-archive", required=True, type=Path)
    parser.add_argument("--right-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequences", default="",
                        help="optional comma-separated exact sequence names")
    parser.add_argument(
        "--holdout-sequences", default=",".join(DEFAULT_HOLDOUT),
        help="complete sequences reserved for validation",
    )
    args = parser.parse_args()
    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    holdout = tuple(
        item.strip() for item in args.holdout_sequences.split(",") if item.strip()
    )
    manifest = prepare(
        args.left_archive, args.right_archive, args.output, sequences, holdout
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
