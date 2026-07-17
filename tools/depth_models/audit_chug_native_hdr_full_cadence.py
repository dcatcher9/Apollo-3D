#!/usr/bin/env python3
"""Audit one published CHUG native-PQ full-cadence train/dev corpus."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import native_hdr_capture
import prepare_chug_native_hdr_full_cadence as full
import prepare_chug_native_hdr_training as sparse


SBSBENCH = Path(__file__).resolve().parents[1] / "sbsbench"
sys.path.insert(0, str(SBSBENCH))
import build_clip_hash_manifest as clip_hashes  # noqa: E402


AUDIT_SCHEMA = 1
AUDIT_CONTRACT = "apollo-chug-native-pq-full-cadence-audit-v1"


def _verified(document_path: Path, row, path_key: str, hash_key: str,
              label: str) -> Path:
    return full._verified_reference(
        document_path, row, path_key, hash_key, label
    )


def _rows_by_video(rows, label: str):
    if not isinstance(rows, list):
        raise RuntimeError(f"{label} rows are missing")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"{label} row is invalid")
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or video_id in result:
            raise RuntimeError(f"{label} video identity is invalid")
        result[video_id] = row
    return result


def _load_source_provenance(bootstrap_path: Path, bootstrap):
    provenance = bootstrap.get("source_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("full-cadence source provenance is missing")
    selection_path = _verified(
        bootstrap_path, provenance, "selection_manifest",
        "selection_manifest_sha256", "CHUG selection manifest",
    )
    receipt_path = _verified(
        bootstrap_path, provenance, "download_receipt",
        "download_receipt_sha256", "CHUG download receipt",
    )
    selection = sparse.read_json(selection_path, "CHUG selection manifest")
    receipt = sparse.read_json(receipt_path, "CHUG download receipt")
    if (selection.get("schema") != 1 or receipt.get("schema") != 1 or
            receipt.get("license") != "CC BY-NC-SA 4.0"):
        raise RuntimeError("CHUG source provenance contract differs")
    return {
        "selection": _rows_by_video(selection.get("clips"), "selection"),
        "receipt": _rows_by_video(receipt.get("accepted"), "receipt"),
    }


def _source_row(sequence, source_provenance):
    video_id = sequence.get("video_id")
    selection = source_provenance["selection"].get(video_id)
    receipt = source_provenance["receipt"].get(video_id)
    if not isinstance(selection, dict) or not isinstance(receipt, dict):
        raise RuntimeError(f"{video_id}: current source provenance is missing")
    split = sequence.get("split")
    capture_group = sequence.get("capture_group_id")
    content_id = receipt.get("content_id")
    download = receipt.get("download")
    if (selection.get("split") != split or receipt.get("split") != split or
            selection.get("capture_group_id") != capture_group or
            receipt.get("capture_group_id") != capture_group or
            selection.get("content_id") != content_id or
            not isinstance(content_id, str) or not content_id or
            not isinstance(download, dict) or
            download.get("video_id") != video_id or
            type(download.get("bytes")) is not int or
            download["bytes"] <= 0 or
            not isinstance(download.get("sha256"), str) or
            len(download["sha256"]) != 64):
        raise RuntimeError(f"{video_id}: current source provenance differs")
    return {
        "content_id": content_id,
        "bytes": download["bytes"],
        "sha256": download["sha256"],
    }


def _clip_audit(clip_root: Path, sequence, *, verify_content: bool,
                production_id: str, conversion_hash: str,
                cut_threshold: float, source_row):
    authentication = native_hdr_capture.validate_clip(
        clip_root, full=verify_content
    )
    manifest = sparse.read_json(
        clip_root / native_hdr_capture.MANIFEST_NAME,
        "native-HDR frame manifest",
    )
    timing_path = clip_root / "source_timing.json"
    timing = sparse.read_json(timing_path, "source timing")
    cuts_path = clip_root / "source_cut_evidence.json"
    cuts = sparse.read_json(cuts_path, "source cut evidence")
    metadata = sparse.read_json(clip_root / "meta.json", "clip metadata")
    labels = sparse.read_json(clip_root / "label_frames.json", "label frames")
    if authentication["frame_count"] != sequence["frames"]:
        raise RuntimeError(f"{clip_root.name}: full-cadence sidecar differs")
    bindings = full._validate_clip_sidecar_bindings(
        metadata=metadata, labels=labels, timing=timing, cuts=cuts,
        frame_manifest=manifest, timing_path=timing_path, cut_path=cuts_path,
        split=sequence["split"], production_id=production_id,
        capture_group_id=sequence["capture_group_id"],
        video_id=sequence["video_id"], content_id=source_row["content_id"],
        source_video_bytes=source_row["bytes"],
        source_video_sha256=source_row["sha256"],
        source_frame_count=sequence["frames"],
        label_frame_ids=sequence["label_frame_ids"],
        source_timing_content_sha256=sequence[
            "source_timing_content_sha256"
        ],
        conversion_hash=conversion_hash, cut_threshold=cut_threshold,
        expected_cut_candidate_count=sequence["source_cut_candidate_count"],
    )
    timing_hash = bindings["timing_content_sha256"]
    cut_hash = bindings["cut_content_sha256"]
    if (timing["source_frame_rate"]["rational"] !=
            sequence["source_frame_rate_rational"] or
            timing["time_base"]["rational"] !=
            sequence["source_time_base_rational"]):
        raise RuntimeError(f"{clip_root.name}: cadence identity differs")
    if len(timing["frames"]) != len(manifest["frames"]):
        raise RuntimeError(f"{clip_root.name}: frame timing coverage differs")
    for timing_row, frame_row in zip(timing["frames"], manifest["frames"]):
        if (timing_row["frame"] != frame_row["frame"] or
                not math.isclose(
                    float(timing_row["timestamp_seconds"]),
                    float(frame_row["timestamp_seconds"]),
                    rel_tol=0.0, abs_tol=1e-12,
                )):
            raise RuntimeError(f"{clip_root.name}: frame timestamp differs")
    stats = [row["stats"] for row in manifest["frames"]]
    return {
        "clip": clip_root.name,
        "video_id": sequence["video_id"],
        "frames": sequence["frames"],
        "source_frame_rate_rational": sequence["source_frame_rate_rational"],
        "source_time_base_rational": sequence["source_time_base_rational"],
        "timing_content_sha256": timing_hash,
        "cut_content_sha256": cut_hash,
        "cut_candidates": cuts["cut_candidate_count"],
        "model_source_content_sha256": manifest["content_sha256"],
        "nonfinite_components": sum(row["nonfinite_components"] for row in stats),
        "maximum_luminance_nits": max(row["luminance_nits_max"] for row in stats),
        "maximum_preview_saturated_fraction": max(
            row["preview_saturated_fraction"] for row in stats
        ),
        "maximum_preview_black_fraction": max(
            row["preview_black_fraction"] for row in stats
        ),
        "visual_samples": [
            str((clip_root / f"frame_{frame_id:05d}.png").resolve())
            for frame_id in (0, sequence["frames"] // 2,
                             sequence["frames"] - 1)
        ],
    }


def _sparse_center_parity(sparse_bootstrap_path: Path, full_by_video):
    sparse_bootstrap_path = sparse_bootstrap_path.resolve(strict=True)
    bootstrap = sparse.read_json(sparse_bootstrap_path, "sparse CHUG bootstrap")
    if (bootstrap.get("schema") != full.SOURCE_BOOTSTRAP_SCHEMA or
            bootstrap.get("contract") != full.SOURCE_BOOTSTRAP_CONTRACT):
        raise RuntimeError("sparse parity bootstrap contract differs")
    comparisons = []
    for split in full.SPLITS:
        entry = bootstrap["datasets"][split]
        manifest_path = _verified(
            sparse_bootstrap_path, entry, "dataset_manifest",
            "dataset_manifest_sha256", f"sparse {split} dataset manifest",
        )
        manifest = sparse.read_json(manifest_path, f"sparse {split} dataset")
        for sequence in manifest["sequences"]:
            video_id = sequence["video_id"]
            new_root, new_manifest = full_by_video[video_id]
            old_root = manifest_path.parent / sequence["clip"]
            old_authentication = native_hdr_capture.validate_clip(
                old_root, full=False
            )
            labels = sparse.read_json(
                old_root / "label_frames.json", "sparse label frames"
            )["frame_ids"]
            if len(labels) != 1:
                raise RuntimeError("sparse CHUG window has multiple/no labels")
            old_row = old_authentication["frames"][labels[0]]
            source_frame = sequence["source_label_frame_id"]
            new_row = new_manifest["frames"][source_frame]
            if (old_row["sha256"] != new_row["sha256"] or
                    old_row["preview_sha256"] != new_row["preview_sha256"]):
                raise RuntimeError(
                    f"{video_id} frame {source_frame}: sparse/full color bytes differ"
                )
            comparisons.append({
                "split": split,
                "video_id": video_id,
                "source_frame": source_frame,
                "model_source_sha256": new_row["sha256"],
                "preview_sha256": new_row["preview_sha256"],
                "full_clip": str(new_root),
                "sparse_clip": str(old_root.resolve()),
            })
    if len(comparisons) != sum(full.EXPECTED_LABEL_FRAMES.values()):
        raise RuntimeError("sparse/full parity comparison cardinality differs")
    return comparisons


def audit(bootstrap_path: Path, *, sparse_bootstrap_path: Path | None = None,
          verify_content: bool = False):
    bootstrap_path = bootstrap_path.resolve(strict=True)
    bootstrap = sparse.read_json(bootstrap_path, "full-cadence bootstrap")
    if (bootstrap.get("schema") != full.PREPARATION_SCHEMA or
            bootstrap.get("contract") != full.PREPARATION_CONTRACT or
            bootstrap.get("dataset") != full.DATASET_NAME or
            bootstrap.get("full_cadence_contract") !=
            full.FULL_CADENCE_CONTRACT):
        raise RuntimeError("full-cadence bootstrap contract differs")
    root = Path(bootstrap["output_root"]).resolve(strict=True)
    if root / full.BOOTSTRAP_MANIFEST != bootstrap_path:
        raise RuntimeError("full-cadence bootstrap output root differs")
    source_provenance = _load_source_provenance(bootstrap_path, bootstrap)
    split_reports = {}
    full_by_video = {}
    for split in full.SPLITS:
        entry = bootstrap["datasets"][split]
        manifest_path = _verified(
            bootstrap_path, entry, "dataset_manifest",
            "dataset_manifest_sha256", f"full-cadence {split} dataset",
        )
        manifest = sparse.read_json(manifest_path, f"full-cadence {split} dataset")
        production_id = f"{full.PRODUCTION_PREFIX}_{split}"
        cut_threshold = manifest.get("source_cut_candidate_threshold")
        conversion_hash = manifest.get("conversion_contract_sha256")
        if (manifest.get("production_id") != production_id or
                manifest.get("split") != split or
                not isinstance(cut_threshold, (int, float)) or
                isinstance(cut_threshold, bool) or
                not math.isfinite(float(cut_threshold)) or
                not 0.0 < float(cut_threshold) < 1.0 or
                not isinstance(conversion_hash, str) or
                len(conversion_hash) != 64 or
                manifest.get("frame_count") != full.EXPECTED_SOURCE_FRAMES[split] or
                manifest.get("label_frame_count") !=
                full.EXPECTED_LABEL_FRAMES[split]):
            raise RuntimeError(f"full-cadence {split} dataset cardinality differs")
        clip_hash_path = _verified(
            bootstrap_path, entry["clip_hash_manifest"], "path", "sha256",
            f"full-cadence {split} clip hash manifest",
        )
        clip_names = sorted(sequence["clip"] for sequence in manifest["sequences"])
        clip_hashes.verify_selected_clips(
            clip_hash_path, manifest_path.parent, clip_names,
            full=verify_content,
        )
        clips = []
        for sequence in manifest["sequences"]:
            if sequence.get("split") != split:
                raise RuntimeError(
                    f"{sequence.get('clip')}: sequence split differs"
                )
            clip_root = manifest_path.parent / sequence["clip"]
            clip = _clip_audit(
                clip_root, sequence, verify_content=verify_content,
                production_id=production_id, conversion_hash=conversion_hash,
                cut_threshold=float(cut_threshold),
                source_row=_source_row(sequence, source_provenance),
            )
            clips.append(clip)
            frame_manifest, _, _ = native_hdr_capture.load_manifest(clip_root)
            full_by_video[sequence["video_id"]] = (clip_root, frame_manifest)
        split_reports[split] = {
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": sparse.sha256(manifest_path),
            "clip_hash_manifest": str(clip_hash_path),
            "clip_hash_manifest_sha256": sparse.sha256(clip_hash_path),
            "clips": len(clips),
            "frames": sum(item["frames"] for item in clips),
            "cut_candidates": sum(item["cut_candidates"] for item in clips),
            "nonfinite_components": sum(
                item["nonfinite_components"] for item in clips
            ),
            "maximum_luminance_nits": max(
                item["maximum_luminance_nits"] for item in clips
            ),
            "maximum_preview_saturated_fraction": max(
                item["maximum_preview_saturated_fraction"] for item in clips
            ),
            "maximum_preview_black_fraction": max(
                item["maximum_preview_black_fraction"] for item in clips
            ),
            "source_frame_rates": sorted({
                item["source_frame_rate_rational"] for item in clips
            }),
            "clip_records": clips,
        }
    parity = None
    if sparse_bootstrap_path is not None:
        parity = _sparse_center_parity(sparse_bootstrap_path, full_by_video)
    byte_count = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    return {
        "schema": AUDIT_SCHEMA,
        "contract": AUDIT_CONTRACT,
        "bootstrap_manifest": str(bootstrap_path),
        "bootstrap_manifest_sha256": sparse.sha256(bootstrap_path),
        "verification": "full-content" if verify_content else "stat-and-contract",
        "sealed_test_policy": bootstrap["sealed_test_policy"],
        "splits": split_reports,
        "totals": {
            "clips": sum(item["clips"] for item in split_reports.values()),
            "frames": sum(item["frames"] for item in split_reports.values()),
            "bytes": byte_count,
            "gib": byte_count / (1024 ** 3),
            "sparse_center_parity_checks": 0 if parity is None else len(parity),
            "sparse_center_parity_failures": 0,
        },
        "sparse_center_parity": parity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-manifest", required=True, type=Path)
    parser.add_argument("--sparse-bootstrap-manifest", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(
        args.bootstrap_manifest,
        sparse_bootstrap_path=args.sparse_bootstrap_manifest,
        verify_content=args.full,
    )
    if args.output:
        sparse.write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
