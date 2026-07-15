#!/usr/bin/env python3
"""Prepare static real-camera stereo domains for artistic-policy training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath


SAFE_NAME_RE = re.compile(r"[^a-z0-9]+")
FLICKR_RE = re.compile(r"^(\d+)_(L|R)\.png$", re.IGNORECASE)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value):
    return SAFE_NAME_RE.sub("_", value.lower()).strip("_")


def ensure_empty(output: Path):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def write_meta(clip: Path, payload):
    (clip / "meta.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def write_manifest(output: Path, payload):
    (output / "dataset_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def prepare_middlebury(archive_path: Path, output: Path):
    ensure_empty(output)
    pairs = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or len(path.parts) != 4:
                continue
            if path.parts[0] != "MiddEval3" or path.parts[1] not in (
                    "trainingF", "testF"):
                continue
            if path.name not in ("im0.png", "im1.png"):
                continue
            pairs.setdefault((path.parts[1], path.parts[2]), {})[path.name] = info

        rows = []
        for (source_split, scene), members in sorted(pairs.items()):
            if set(members) != {"im0.png", "im1.png"}:
                raise RuntimeError(f"Middlebury {source_split}/{scene}: incomplete pair")
            clip = output / f"middlebury_{safe_name(scene)}"
            left_path = clip / "frame_00000.png"
            right_path = clip / "gt_right" / "frame_00000.png"
            left_path.parent.mkdir(parents=True, exist_ok=True)
            right_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(members["im0.png"]) as source, left_path.open("wb") as dest:
                shutil.copyfileobj(source, dest, length=1024 * 1024)
            with archive.open(members["im1.png"]) as source, right_path.open("wb") as dest:
                shutil.copyfileobj(source, dest, length=1024 * 1024)
            split = "validation"
            write_meta(clip, {
                "schema": 1,
                "name": f"middlebury-{scene}",
                "dataset": "Middlebury Stereo Evaluation v3",
                "domain": "middlebury",
                "homepage": "https://vision.middlebury.edu/stereo/submit3/",
                "license_note": (
                    "Middlebury grants permission to use and publish its stereo "
                    "images and disparities; cite the dataset authors."
                ),
                "purpose": "artistic-policy real-camera stereo supervision",
                "policy_role": "validation_only",
                "global_policy_weight": 0.0,
                "scene": scene,
                "source_split": source_split,
                "split": split,
                "required_gt_stereo": True,
                "static_pair": True,
            })
            rows.append({"clip": clip.name, "scene": scene, "frames": 1,
                         "source_split": source_split, "split": split})

    return write_manifest(output, {
        "schema": 1,
        "dataset": "Middlebury Stereo Evaluation v3",
        "domain": "middlebury",
        "homepage": "https://vision.middlebury.edu/stereo/submit3/",
        "archive": str(archive_path.resolve()),
        "archive_size": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "pairs": rows,
        "pair_count": len(rows),
        "training_pairs": 0,
        "validation_pairs": sum(row["split"] == "validation" for row in rows),
    })


def prepare_eth3d(input_root: Path, output: Path,
                  holdout_prefixes=("forest", "terrace")):
    ensure_empty(output)
    rows = []
    for source in sorted(path for path in input_root.iterdir() if path.is_dir()):
        left_source = source / "im0.png"
        right_source = source / "im1.png"
        if not left_source.is_file() and not right_source.is_file():
            continue
        if not left_source.is_file() or not right_source.is_file():
            raise RuntimeError(f"ETH3D {source.name}: incomplete pair")
        clip = output / f"eth3d_{safe_name(source.name)}"
        left_path = clip / "frame_00000.png"
        right_path = clip / "gt_right" / "frame_00000.png"
        left_path.parent.mkdir(parents=True, exist_ok=True)
        right_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(left_source, left_path)
        shutil.copyfile(right_source, right_path)
        split = "validation"
        write_meta(clip, {
            "schema": 1,
            "name": f"eth3d-{source.name}",
            "dataset": "ETH3D Low-res Two-view Stereo Training Dataset",
            "domain": "eth3d",
            "homepage": "https://www.eth3d.net/datasets",
            "license_note": "Use under the ETH3D dataset terms and cite the authors.",
            "purpose": "artistic-policy real-camera stereo supervision",
            "policy_role": "validation_only",
            "global_policy_weight": 0.0,
            "scene": source.name,
            "source_split": "training",
            "split": split,
            "required_gt_stereo": True,
            "static_pair": True,
        })
        rows.append({"clip": clip.name, "scene": source.name, "frames": 1,
                     "split": split})
    if not rows:
        raise RuntimeError(f"no ETH3D im0.png/im1.png pairs found under {input_root}")
    return write_manifest(output, {
        "schema": 1,
        "dataset": "ETH3D Low-res Two-view Stereo Training Dataset",
        "domain": "eth3d",
        "homepage": "https://www.eth3d.net/datasets",
        "input_root": str(input_root.resolve()),
        "holdout_prefixes": list(holdout_prefixes),
        "pairs": rows,
        "pair_count": len(rows),
        "training_pairs": 0,
        "validation_pairs": sum(row["split"] == "validation" for row in rows),
    })


def prepare_flickr(archive_path: Path, output: Path):
    ensure_empty(output)
    pairs = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or len(path.parts) != 3 or path.parts[0] != "Flickr1024":
                continue
            match = FLICKR_RE.fullmatch(path.name)
            if not match or path.parts[1] not in ("Train", "Validation", "Test"):
                continue
            pairs.setdefault((path.parts[1], match.group(1)), {})[
                match.group(2).upper()
            ] = info

        rows = []
        for (source_split, pair_id), members in sorted(pairs.items()):
            if set(members) != {"L", "R"}:
                raise RuntimeError(f"Flickr1024 {source_split}/{pair_id}: incomplete pair")
            clip = output / f"flickr1024_{source_split.lower()}_{pair_id}"
            left_path = clip / "frame_00000.png"
            right_path = clip / "gt_right" / "frame_00000.png"
            left_path.parent.mkdir(parents=True, exist_ok=True)
            right_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(members["L"]) as source, left_path.open("wb") as dest:
                shutil.copyfileobj(source, dest, length=1024 * 1024)
            with archive.open(members["R"]) as source, right_path.open("wb") as dest:
                shutil.copyfileobj(source, dest, length=1024 * 1024)
            split = "training" if source_split == "Train" else "validation"
            write_meta(clip, {
                "schema": 1,
                "name": f"flickr1024-{source_split}-{pair_id}",
                "dataset": "Flickr1024",
                "domain": "flickr1024",
                "homepage": "https://yingqianwang.github.io/Flickr1024/",
                "license": "non-commercial use only",
                "license_scope": "dataset and derived data",
                "purpose": "artistic-policy real-camera local stereo supervision",
                "policy_role": "local_geometry",
                "global_policy_weight": 0.0,
                "pair_id": pair_id,
                "source_split": source_split,
                "split": split,
                "required_gt_stereo": True,
                "static_pair": True,
            })
            rows.append({"clip": clip.name, "pair_id": pair_id, "frames": 1,
                         "source_split": source_split, "split": split})

    if len(rows) != 1024:
        raise RuntimeError(f"expected 1024 Flickr1024 pairs, found {len(rows)}")
    return write_manifest(output, {
        "schema": 1,
        "dataset": "Flickr1024",
        "domain": "flickr1024",
        "homepage": "https://yingqianwang.github.io/Flickr1024/",
        "license": "non-commercial use only",
        "policy_role": "local_geometry",
        "archive": str(archive_path.resolve()),
        "archive_size": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "pairs": rows,
        "pair_count": len(rows),
        "training_pairs": sum(row["split"] == "training" for row in rows),
        "validation_pairs": sum(row["split"] == "validation" for row in rows),
    })


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    middlebury = subparsers.add_parser("middlebury")
    middlebury.add_argument("--archive", required=True, type=Path)
    middlebury.add_argument("--output", required=True, type=Path)
    eth3d = subparsers.add_parser("eth3d")
    eth3d.add_argument("--input", required=True, type=Path)
    eth3d.add_argument("--output", required=True, type=Path)
    eth3d.add_argument("--holdout-prefixes", default="forest,terrace")
    flickr = subparsers.add_parser("flickr1024")
    flickr.add_argument("--archive", required=True, type=Path)
    flickr.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.dataset == "middlebury":
        manifest = prepare_middlebury(args.archive, args.output)
    elif args.dataset == "eth3d":
        holdout = tuple(item.strip() for item in args.holdout_prefixes.split(",")
                        if item.strip())
        manifest = prepare_eth3d(args.input, args.output, holdout)
    else:
        manifest = prepare_flickr(args.archive, args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
