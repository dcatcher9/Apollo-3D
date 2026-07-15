#!/usr/bin/env python3
"""Freeze and validate the active complete-production artistic data split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fetch_artistic_stereo_sources as sources


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(catalog_path: Path, manifest_paths):
    catalog = sources.load_catalog(catalog_path)
    by_production = {
        item["production_id"]: item for item in catalog["sources"]
        if item.get("production_id")
    }
    productions = {}
    rows = []
    for path in manifest_paths:
        path = path.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != 1:
            raise RuntimeError(f"unsupported dataset manifest: {path}")
        production = payload.get("film_id")
        split = payload.get("split")
        if not production or split not in sources.SPLITS:
            raise RuntimeError(f"{path}: missing film_id or valid split")
        if production in productions:
            raise RuntimeError(
                f"production {production!r} appears in multiple active manifests"
            )
        source = by_production.get(production)
        if source is None or source.get("admission") != "global_policy":
            raise RuntimeError(f"{production}: not an admitted catalog production")
        if source["split"] != split:
            raise RuntimeError(
                f"{production}: catalog split {source['split']} != dataset {split}"
            )
        expected_weight = float(source["global_policy_weight"])
        actual_weight = float(payload.get("global_policy_weight", 1.0))
        if abs(expected_weight - actual_weight) > 1e-9:
            raise RuntimeError(
                f"{production}: catalog weight {expected_weight} != "
                f"dataset {actual_weight}"
            )
        productions[production] = split
        context_frames = int(payload.get("context_frame_count", 0))
        context_fps = float(payload.get(
            "context_fps", payload.get("source_fps", 0.0)
        ))
        if context_frames <= 0 or context_fps <= 0.0:
            raise RuntimeError(
                f"{production}: dataset predates full-cadence context"
            )
        label_frames = int(payload.get("sample_count", 0))
        if context_frames <= 0 or label_frames <= 0:
            raise RuntimeError(f"{production}: dataset has no usable frames")
        video = Path(payload.get("video", ""))
        expected_video_hash = payload.get("video_sha256")
        if not video.is_file() or not expected_video_hash:
            raise RuntimeError(f"{production}: source video or hash is missing")
        current_video_hash = sha256(video)
        if current_video_hash != expected_video_hash:
            raise RuntimeError(f"{production}: source video hash changed")
        rows.append({
            "production_id": production,
            "source_id": source["id"],
            "source_group": source["source_group"],
            "split": split,
            "global_policy_weight": actual_weight,
            "dataset_manifest": str(path),
            "dataset_manifest_sha256": sha256(path),
            "video_sha256": current_video_hash,
            "context_frames": context_frames,
            "context_fps": context_fps,
            "label_frames": label_frames,
            "shots": int(payload.get("shot_count", 0)),
        })
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
        default=Path(__file__).with_name("artistic_stereo_sources.json"),
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
