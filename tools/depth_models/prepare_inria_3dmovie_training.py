#!/usr/bin/env python3
"""Prepare Inria 3DMovie video sequences for artistic-policy training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat


HOMEPAGE = "https://www.di.ens.fr/willow/research/stereoseg/"
LICENSE = "non-commercial research; source copyrights retained"


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stereo_pairs(sequence: Path):
    pairs = []
    for left in sorted(sequence.glob("*.jpg")):
        right = left.with_name(left.name + ".right")
        if not right.is_file():
            raise RuntimeError(f"missing right eye for {left}")
        if not left.stem.isdigit():
            raise RuntimeError(f"non-numeric Inria frame name: {left.name}")
        pairs.append((left, right))
    unexpected = sorted(
        item for item in sequence.iterdir()
        if item.is_file() and item.name.endswith(".jpg.right")
        and not item.with_name(item.name.removesuffix(".right")).is_file()
    )
    if unexpected:
        raise RuntimeError(f"orphan right eye: {unexpected[0]}")
    if not pairs:
        raise RuntimeError(f"no stereo pairs in {sequence}")
    return pairs


def write_png(source: Path, output: Path):
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"cannot write {output}")
    return image.shape[1], image.shape[0]


def write_disparity(source: Path, output: Path, dimensions):
    payload = loadmat(source)
    uv = np.asarray(payload.get("uv"), np.float32)
    if uv.ndim != 3 or uv.shape[2] < 2:
        raise RuntimeError(f"{source}: expected HxWx2 uv disparity")
    width, height = dimensions
    if uv.shape[:2] != (height, width):
        raise RuntimeError(
            f"{source}: disparity/image dimensions differ "
            f"({uv.shape[:2]} != {(height, width)})"
        )
    # Inria stores left-to-right optical flow (x_right - x_left). Apollo's
    # high-near convention is x_left - x_right, so polarity is reversed.
    disparity_px = -uv[..., 0]
    vertical_disparity_px = uv[..., 1]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        disparity_px=disparity_px.astype(np.float32),
        vertical_disparity_px=vertical_disparity_px.astype(np.float32),
    )


def prepare(frames_root: Path, disparity_root: Path, output: Path,
            source_archive: Path, disparity_archive: Path):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output must be empty: {output}")
    for archive, description in (
        (source_archive, "source archive"),
        (disparity_archive, "disparity archive"),
    ):
        if not archive.is_file():
            raise RuntimeError(f"missing {description}: {archive}")
    sequences = sorted(
        (item for item in frames_root.iterdir() if item.is_dir()),
        key=lambda item: int(item.name),
    )
    if not sequences:
        raise RuntimeError(f"no sequence directories under {frames_root}")
    output.mkdir(parents=True, exist_ok=True)
    archive_hash = sha256(source_archive)
    disparity_archive_hash = sha256(disparity_archive)
    prepared = []
    for index, sequence in enumerate(sequences, 1):
        clip_name = f"inria_3dmovie_{int(sequence.name):02d}"
        clip = output / clip_name
        pairs = stereo_pairs(sequence)
        disparity_sequence = disparity_root / sequence.name
        if not disparity_sequence.is_dir():
            raise RuntimeError(f"missing disparity sequence: {disparity_sequence}")
        dimensions = set()
        for left, right in pairs:
            suffix = left.stem
            left_dimensions = write_png(left, clip / f"frame_{suffix}.png")
            dimensions.add(left_dimensions)
            dimensions.add(write_png(
                right, clip / "gt_right" / f"frame_{suffix}.png"
            ))
            disparity_source = (
                disparity_sequence / f"{left.name}_disparity.mat"
            )
            if not disparity_source.is_file():
                raise RuntimeError(
                    f"missing reference disparity for {left}: {disparity_source}"
                )
            write_disparity(
                disparity_source,
                clip / "gt_disparity" / f"frame_{suffix}.npz",
                left_dimensions,
            )
        if len(dimensions) != 1:
            raise RuntimeError(
                f"{sequence}: inconsistent eye/frame dimensions {dimensions}"
            )
        width, height = dimensions.pop()
        meta = {
            "schema": 1,
            "name": clip_name,
            "dataset": "Inria 3DMovie Dataset v1",
            "domain": "inria_3dmovie",
            # The public archive identifies two source films but does not map
            # every numeric sequence to one of them.  Keep one mixed film id
            # so no sequence-level split can masquerade as film validation.
            "film_id": "inria_3dmovie_mixed_streetdance_pina",
            "film_identity_known": False,
            "homepage": HOMEPAGE,
            "source_archive": str(source_archive.resolve()),
            "source_archive_sha256": archive_hash,
            "disparity_archive": str(disparity_archive.resolve()),
            "disparity_archive_sha256": disparity_archive_hash,
            "source_sequence": sequence.name,
            "license": LICENSE,
            "purpose": "artistic-policy authored stereo movie training",
            "policy_role": "cinematic_training",
            "global_policy_weight": 1.0,
            "projection": "rectilinear",
            "source_width": width,
            "source_height": height,
            "frame_count": len(pairs),
            "split": "training",
            "validation_eligible": False,
            "required_gt_stereo": True,
            "reference_disparity": "Inria uv left-to-right flow, polarity reversed",
        }
        (clip / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        prepared.append({
            "clip": clip_name,
            "source_sequence": sequence.name,
            "frames": len(pairs),
            "split": "training",
        })
        print(
            f"[{index}/{len(sequences)}] {clip_name}: {len(pairs)} pairs",
            flush=True,
        )

    manifest = {
        "schema": 1,
        "dataset": "Inria 3DMovie Dataset v1",
        "homepage": HOMEPAGE,
        "license": LICENSE,
        "policy_role": "cinematic_training",
        "global_policy_weight": 1.0,
        "film_id": "inria_3dmovie_mixed_streetdance_pina",
        "film_identity_known": False,
        "validation_eligible": False,
        "source_archive": str(source_archive.resolve()),
        "source_archive_sha256": archive_hash,
        "disparity_archive": str(disparity_archive.resolve()),
        "disparity_archive_sha256": disparity_archive_hash,
        "source_frames_root": str(frames_root.resolve()),
        "sequences": prepared,
        "clip_count": len(prepared),
        "frame_count": sum(item["frames"] for item in prepared),
        "training_frames": sum(item["frames"] for item in prepared),
        "validation_frames": 0,
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--disparity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--disparity-archive", required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare(
        args.frames, args.disparity, args.output, args.source_archive,
        args.disparity_archive,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
