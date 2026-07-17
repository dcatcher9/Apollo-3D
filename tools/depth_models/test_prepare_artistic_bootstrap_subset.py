#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

import prepare_artistic_bootstrap_subset as bootstrap
import train_artistic_policy as train


FIXTURE_COUNTS = {
    ("reds", "training"): 2,
    ("spring", "training"): 1,
    ("reds", "development"): 1,
    ("spring", "development"): 1,
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_bytes(width, height, seed):
    image = Image.new("RGB", (width, height))
    image.putdata([
        (
            (seed + x * 17 + y * 5) % 256,
            (seed * 3 + x * 7 + y * 11) % 256,
            (seed * 5 + x * 13 + y * 3) % 256,
        )
        for y in range(height) for x in range(width)
    ])
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    return encoded.getvalue()


class ArtisticBootstrapSubsetTests(unittest.TestCase):
    def make_source_split(self, prepared, source, split, clips, size):
        root = prepared / bootstrap.SOURCE_LAYOUT[source] / split
        root.mkdir(parents=True)
        production = f"{source}_mono_v1_{split}"
        sequence_rows = []
        sequence_manifest_rows = []
        for clip_index, clip in enumerate(clips):
            clip_root = root / clip
            clip_root.mkdir()
            frame_rows = []
            for frame_id in range(3):
                path = clip_root / f"frame_{frame_id:05d}.png"
                path.write_bytes(image_bytes(
                    size[0], size[1], 20 + clip_index * 30 + frame_id
                ))
                frame_rows.append({
                    "local_frame": frame_id,
                    "source_frame": frame_id,
                    "output": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "width": size[0],
                    "height": size[1],
                })
            labels = {"schema": 1, "frame_ids": [0, 2]}
            (clip_root / "label_frames.json").write_text(
                json.dumps(labels, indent=2) + "\n", encoding="utf-8"
            )
            meta = {
                "schema": 2,
                "name": clip,
                "film_id": production,
                "production_id": production,
                "source_kind": "mono-video",
                "source_width": size[0],
                "source_height": size[1],
                "split": split,
                "dataset": f"Synthetic {source}",
                "domain": f"{source}-domain",
                "license": "CC BY 4.0",
                "license_url": "https://example.test/license",
                "homepage": f"https://example.test/{source}",
                "global_policy_weight": 1.0 if source == "reds" else 1.0 / 3.0,
                "source_color_contract": "rgb8-png-byte-exact",
                "auxiliary_disparity": (
                    "fixture-disparity" if source == "spring" else None
                ),
            }
            (clip_root / "meta.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
            record = {
                "schema": 1,
                "clip": clip,
                "source_sequence": clip.rsplit("_", 1)[-1],
                "source_first_frame": 0,
                "source_last_frame": 2,
                "source_frame_count": 3,
                "label_frame_ids": [0, 2],
                "rgb_frames": frame_rows,
                "width": size[0],
                "height": size[1],
            }
            (clip_root / "source_sequence_record.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            if source == "spring":
                disparity = clip_root / "gt_disparity"
                disparity.mkdir()
                (disparity / "frame_00000.dsp5").write_bytes(b"source-only")
            sequence_row = {
                "clip": clip,
                "source_sequence": record["source_sequence"],
                "source_start_frame": 0,
                "source_end_frame": 2,
                "context_frames": 3,
                "label_frames": 2,
                "split": split,
            }
            sequence_rows.append(sequence_row)
            sequence_manifest_rows.append(record)
        source_sequence_manifest = {
            "schema": 1,
            "production_id": production,
            "split": split,
            "sequences": sequence_manifest_rows,
        }
        sequence_path = root / "source_sequence_manifest.json"
        sequence_path.write_text(
            json.dumps(source_sequence_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        dataset = {
            "schema": 2,
            "dataset": f"Synthetic {source}",
            "domain": f"{source}-domain",
            "production_id": production,
            "source_kind": "mono-video",
            "source_sequence_manifest": sequence_path.name,
            "policy_role": "cinematic_training",
            "homepage": f"https://example.test/{source}",
            "license": "CC BY 4.0",
            "license_url": "https://example.test/license",
            "split": split,
            "context_fps": 24.0,
            "global_policy_weight": 1.0 if source == "reds" else 1.0 / 3.0,
            "color_contract": "decoded-sdr-bgr8",
            "source_color_contract": "rgb8-png-byte-exact",
            "video_sha256": sha256(sequence_path),
            # Reverse the rows so selection must sort by clip rather than trust order.
            "sequences": list(reversed(sequence_rows)),
            "shot_count": len(sequence_rows),
            "context_frame_count": 3 * len(sequence_rows),
            "label_frame_count": 2 * len(sequence_rows),
        }
        (root / "dataset_manifest.json").write_text(
            json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
        )
        bootstrap.clip_hashes.build_and_write(root, workers=1)
        return root

    def make_sealed_test(self, prepared, source):
        root = prepared / bootstrap.SOURCE_LAYOUT[source] / "test"
        root.mkdir(parents=True)
        production = f"{source}_mono_v1_test"
        sequence_path = root / "source_sequence_manifest.json"
        sequence_path.write_text(json.dumps({
            "schema": 1,
            "source_container": "image-sequence-archives",
            "production_id": production,
            "split": "test",
            "sequences": [{
                "clip": f"{source}_sealed_000",
                "source_frame_count": 5,
                "label_frame_ids": [0, 4],
            }],
            "context_frame_count": 5,
            "label_frame_count": 2,
        }, indent=2) + "\n", encoding="utf-8")
        dataset = {
            "schema": 2,
            "dataset": f"Synthetic {source}",
            "domain": f"{source}-domain",
            "production_id": production,
            "source_kind": "mono-video",
            "source_container": "image-sequence-archives",
            "source_sequence_manifest": sequence_path.name,
            "policy_role": "cinematic_training",
            "homepage": f"https://example.test/{source}",
            "license": "CC BY 4.0",
            "license_url": "https://example.test/license",
            "split": "test",
            "context_fps": 24.0,
            "global_policy_weight": (
                1.0 if source == "reds" else 1.0 / 3.0
            ),
            "color_contract": "decoded-sdr-bgr8",
            "source_color_contract": "rgb8-png-byte-exact",
            "video_sha256": sha256(sequence_path),
            "sequences": [{
                "clip": f"{source}_sealed_000",
                "context_frames": 5,
                "label_frames": 2,
                "split": "test",
            }],
            "shot_count": 1,
            "context_frame_count": 5,
            "label_frame_count": 2,
        }
        (root / "dataset_manifest.json").write_text(
            json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
        )
        return root

    def make_fixture(self, root):
        prepared = root / "prepared"
        roots = {
            ("reds", "training"): self.make_source_split(
                prepared, "reds", "training",
                ["reds_training_009", "reds_training_003", "reds_training_000"],
                (8, 4),
            ),
            ("spring", "training"): self.make_source_split(
                prepared, "spring", "training",
                ["spring_training_0005", "spring_training_0001"], (12, 6),
            ),
            ("reds", "development"): self.make_source_split(
                prepared, "reds", "development",
                ["reds_development_002", "reds_development_001"], (8, 4),
            ),
            ("spring", "development"): self.make_source_split(
                prepared, "spring", "development",
                ["spring_development_0008", "spring_development_0007"],
                (12, 6),
            ),
            ("reds", "test"): self.make_sealed_test(prepared, "reds"),
            ("spring", "test"): self.make_sealed_test(prepared, "spring"),
        }
        return prepared, roots

    def test_prepares_deterministic_subset_with_exact_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, source_roots = self.make_fixture(root)
            source_frame = (
                source_roots[("spring", "training")] /
                "spring_training_0001" / "frame_00000.png"
            )
            source_bytes = source_frame.read_bytes()
            output = root / "bootstrap"
            summary = bootstrap.prepare_bootstrap(
                prepared, output, workers=2, width=8, height=4,
                selection_counts=FIXTURE_COUNTS,
            )

            self.assertEqual(summary["totals"], {
                "productions": 4,
                "shots": 5,
                "context_frames": 15,
                "label_frames": 10,
            })
            reds_train = (
                output / "reds-mono-hdr-bootstrap-v1" / "training"
            )
            manifest = json.loads(
                (reds_train / "dataset_manifest.json").read_text()
            )
            self.assertEqual(
                [row["clip"] for row in manifest["sequences"]],
                ["reds_training_000", "reds_training_003"],
            )
            self.assertEqual(manifest["license"], "CC BY 4.0")
            self.assertEqual(manifest["canonical_source_width"], 8)
            self.assertEqual(manifest["canonical_source_height"], 4)

            spring_train = (
                output / "spring-mono-hdr-bootstrap-v1" / "training"
            )
            derived = spring_train / "spring_training_0001"
            with Image.open(derived / "frame_00000.png") as image:
                self.assertEqual(image.size, (8, 4))
                self.assertEqual(image.mode, "RGB")
            self.assertEqual(source_frame.read_bytes(), source_bytes)
            self.assertFalse((derived / "gt_disparity").exists())
            meta = json.loads((derived / "meta.json").read_text())
            self.assertEqual(meta["license"], "CC BY 4.0")
            self.assertEqual(meta["source_width"], 8)
            self.assertEqual(meta["source_height"], 4)
            self.assertIn("original raster", meta["excluded_auxiliary_data"])
            self.assertEqual(
                (derived / "label_frames.json").read_bytes(),
                (source_frame.parent / "label_frames.json").read_bytes(),
            )
            for dataset in summary["datasets"]:
                dataset_root = Path(dataset["output_root"])
                bootstrap.clip_hashes.verify_selected_clips(
                    dataset_root / bootstrap.clip_hashes.MANIFEST_NAME,
                    dataset_root, dataset["clips"], full=True,
                )

            contract = summary["training_contract"]
            catalog_path = Path(contract["source_catalog"])
            active_path = Path(contract["active_split"])
            self.assertEqual(catalog_path.name, bootstrap.SOURCE_CATALOG_NAME)
            self.assertEqual(active_path.name, bootstrap.ACTIVE_SPLIT_NAME)
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual(len(catalog["sources"]), 6)
            self.assertEqual(
                {
                    row["production_id"] for row in catalog["sources"]
                    if row["split"] != "test"
                },
                {
                    "reds_mono_hdr_bootstrap_v1_training",
                    "reds_mono_hdr_bootstrap_v1_development",
                    "spring_mono_hdr_bootstrap_v1_training",
                    "spring_mono_hdr_bootstrap_v1_development",
                },
            )
            self.assertEqual(
                {
                    row["production_id"] for row in catalog["sources"]
                    if row["split"] == "test"
                },
                {"reds_mono_v1_test", "spring_mono_v1_test"},
            )
            self.assertTrue(all(
                row["retrieval"].get("bootstrap_publication_access") ==
                "dataset and source-sequence manifests only; frames unopened"
                for row in catalog["sources"] if row["split"] == "test"
            ))
            self.assertFalse(any(
                path
                for test_root in (
                    source_roots[("reds", "test")],
                    source_roots[("spring", "test")],
                )
                for path in test_root.glob("**/frame_*.png")
            ))

            active, active_hash = train.load_active_split(active_path)
            self.assertEqual(active_hash, contract["active_split_sha256"])
            self.assertEqual(active["split_productions"], {
                "training": [
                    "reds_mono_hdr_bootstrap_v1_training",
                    "spring_mono_hdr_bootstrap_v1_training",
                ],
                "development": [
                    "reds_mono_hdr_bootstrap_v1_development",
                    "spring_mono_hdr_bootstrap_v1_development",
                ],
                "test": ["reds_mono_v1_test", "spring_mono_v1_test"],
            })
            self.assertEqual(
                {row["source_group"] for row in active["productions"]
                 if row["split"] == "test"},
                {"reds_gopro_capture", "spring_blender_movie"},
            )
            self.assertTrue(all(
                row["source_sequence_manifest_sha256"] ==
                row["video_sha256"]
                for row in active["productions"]
            ))
            train.validate_rows_against_active_split([
                {
                    "split": split,
                    "film_id": bootstrap._derived_production(source, split),
                }
                for source in bootstrap.SOURCES
                for split in bootstrap.SPLITS
            ], active, {"training", "development"})

    def test_two_outputs_have_identical_semantic_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, _roots = self.make_fixture(root)
            first = bootstrap.prepare_bootstrap(
                prepared, root / "first", workers=1, width=8, height=4,
                selection_counts=FIXTURE_COUNTS,
            )
            second = bootstrap.prepare_bootstrap(
                prepared, root / "second", workers=2, width=8, height=4,
                selection_counts=FIXTURE_COUNTS,
            )
            first_digests = {
                (row["source"], row["split"]):
                row["clip_hash_semantic_content_sha256"]
                for row in first["datasets"]
            }
            second_digests = {
                (row["source"], row["split"]):
                row["clip_hash_semantic_content_sha256"]
                for row in second["datasets"]
            }
            self.assertEqual(first_digests, second_digests)

    def test_cache_reuses_normalized_train_dev_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, _roots = self.make_fixture(root)
            cache = root / "cache"
            bootstrap.prepare_bootstrap(
                prepared, root / "first", workers=2, width=8, height=4,
                selection_counts=FIXTURE_COUNTS, preprocess_cache=cache,
            )
            with mock.patch.object(
                    bootstrap, "normalize_frame",
                    side_effect=AssertionError("cache hit normalized a frame")):
                second = bootstrap.prepare_bootstrap(
                    prepared, root / "second", workers=2,
                    width=8, height=4,
                    selection_counts=FIXTURE_COUNTS,
                    preprocess_cache=cache,
                )
            self.assertEqual(second["totals"]["context_frames"], 15)

    def test_cache_rejects_source_change_between_key_and_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, _roots = self.make_fixture(root)
            original = bootstrap.normalize_frame
            changed = False

            def mutate_then_normalize(source, destination, width, height):
                nonlocal changed
                if not changed:
                    changed = True
                    source.write_bytes(image_bytes(8, 4, 249))
                return original(source, destination, width, height)

            with mock.patch.object(
                    bootstrap, "normalize_frame",
                    side_effect=mutate_then_normalize):
                with self.assertRaisesRegex(
                        RuntimeError, "changed between cache keying and resize"):
                    bootstrap.prepare_bootstrap(
                        prepared, root / "changed", workers=1,
                        width=8, height=4,
                        selection_counts=FIXTURE_COUNTS,
                        preprocess_cache=root / "cache",
                    )

    def test_rejects_overlap_existing_output_and_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, _roots = self.make_fixture(root)
            with self.assertRaisesRegex(RuntimeError, "overlap"):
                bootstrap.prepare_bootstrap(
                    prepared, prepared / "derived", workers=1,
                    selection_counts=FIXTURE_COUNTS,
                )
            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                bootstrap.prepare_bootstrap(
                    prepared, output, workers=1,
                    selection_counts=FIXTURE_COUNTS,
                )
            with self.assertRaisesRegex(RuntimeError, "outside the admitted"):
                bootstrap._source_dataset(
                    prepared, "reds", "test", 1, False
                )


if __name__ == "__main__":
    unittest.main()
