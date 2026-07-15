#!/usr/bin/env python3
"""Extract complete MPI Sintel stereo sequences for artistic-policy training.

The regular SBS evaluation suite deliberately uses short curated windows.  Policy
training needs more independent scenes, but must retain complete sequences so the
validation split cannot leak adjacent frames into training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath


FRAME_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)
SEQUENCE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_frames(archive: zipfile.ZipFile, render_pass: str, eye: str):
    prefix = PurePosixPath("training") / f"{render_pass}_{eye}"
    result = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.is_dir() or len(path.parts) != 4:
            continue
        if PurePosixPath(*path.parts[:2]) != prefix:
            continue
        sequence = path.parts[2]
        match = FRAME_RE.fullmatch(path.name)
        if not match or not SEQUENCE_RE.fullmatch(sequence):
            continue
        result.setdefault(sequence, {})[int(match.group(1))] = info
    return result


def extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, output.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)


def prepare(archive_path: Path, output: Path, render_pass: str,
            sequence_filter=None, holdout_prefixes=("ambush", "market")):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    requested = set(sequence_filter or [])
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        left = archive_frames(archive, render_pass, "left")
        right = archive_frames(archive, render_pass, "right")
        sequences = sorted(set(left) & set(right))
        if requested:
            missing = requested - set(sequences)
            if missing:
                raise RuntimeError(f"sequences not found in archive: {sorted(missing)}")
            sequences = [sequence for sequence in sequences if sequence in requested]
        if not sequences:
            raise RuntimeError(f"no paired {render_pass} stereo sequences in {archive_path}")

        for sequence in sequences:
            left_ids = set(left[sequence])
            right_ids = set(right[sequence])
            if left_ids != right_ids:
                raise RuntimeError(
                    f"{sequence}: left/right frame mismatch: "
                    f"left-only={sorted(left_ids - right_ids)}, "
                    f"right-only={sorted(right_ids - left_ids)}"
                )
            clip = output / f"sintel_{sequence}"
            selection = []
            for output_index, source_id in enumerate(sorted(left_ids)):
                name = f"frame_{output_index:05d}.png"
                extract_member(archive, left[sequence][source_id], clip / name)
                extract_member(archive, right[sequence][source_id],
                               clip / "gt_right" / name)
                selection.append({
                    "output_frame": output_index,
                    "dataset_frame": source_id,
                    "source_left": left[sequence][source_id].filename,
                    "source_right": right[sequence][source_id].filename,
                })
            split = ("validation" if any(
                sequence.startswith(prefix) for prefix in holdout_prefixes
            ) else "training")
            meta = {
                "schema": 1,
                "name": f"sintel-{sequence}",
                "dataset": "MPI Sintel Stereo Training Dataset",
                "domain": "sintel",
                "homepage": "https://sintel.is.tue.mpg.de/stereo",
                "citation": (
                    "Butler et al., A Naturalistic Open Source Movie for Optical "
                    "Flow Evaluation, ECCV 2012"
                ),
                "license_note": (
                    "Official research dataset derived from the open Sintel film; "
                    "prepared media remains external to Git."
                ),
                "purpose": "artistic-policy geometric stereo supervision",
                "policy_role": "local_geometry",
                "global_policy_weight": 0.0,
                "pass": render_pass,
                "sequence": sequence,
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
        "dataset": "MPI Sintel Stereo Training Dataset",
        "domain": "sintel",
        "policy_role": "local_geometry",
        "archive": str(archive_path.resolve()),
        "archive_sha256": sha256(archive_path),
        "pass": render_pass,
        "holdout_prefixes": list(holdout_prefixes),
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
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pass", dest="render_pass", choices=("clean", "final"),
                        default="final")
    parser.add_argument("--sequences", default="",
                        help="optional comma-separated exact sequence names")
    parser.add_argument("--holdout-prefixes", default="ambush,market",
                        help="whole sequence families reserved for validation")
    args = parser.parse_args()
    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    holdout = tuple(
        item.strip() for item in args.holdout_prefixes.split(",") if item.strip()
    )
    manifest = prepare(
        args.archive, args.output, args.render_pass, sequences, holdout
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
