#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import artistic_sources as sources
import audit_artistic_dataset_splits as split_audit


class ArtisticDatasetSplitAuditTests(unittest.TestCase):
    @staticmethod
    def source(production, split, kind="mono-video", group=None):
        row = {
            "id": production,
            "production_id": production,
            "source_kind": kind,
            "source_group": group or production,
            "split": split,
            "admission": "global_policy",
            "complete_production": True,
            "global_policy_weight": 1.0,
            "license": "CC BY",
            "license_url": "https://example.test/license",
            "retrieval": {"kind": "unavailable"},
        }
        if kind == "authored-stereo":
            row.update({
                "eye_order": "first-left",
                "eye_display_aspect_ratio": 16 / 9,
            })
        return row

    @staticmethod
    def assignments():
        return (
            ("train", "training"),
            ("dev", "development"),
            ("test_a", "test"),
            ("test_b", "test"),
        )

    @staticmethod
    def write_manifest(root, production, split, schema, kind):
        manifest = root / f"{production}.json"
        video = root / f"{production}.mp4"
        video.write_bytes(production.encode("ascii"))
        payload = {
            "schema": schema,
            "split": split,
            "global_policy_weight": 1.0,
            "context_frame_count": 10,
            "context_fps": 24.0,
            "shot_count": 1,
            "video": video.name,
            "video_sha256": split_audit.sha256(video),
        }
        if schema == 1:
            payload.update({"film_id": production, "sample_count": 2})
        else:
            payload.update({
                "production_id": production,
                "source_kind": kind,
                "label_frame_count": 2,
            })
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return manifest

    @staticmethod
    def write_sequence_manifest(
            root, production, split,
            source_container="image-sequence-archives"):
        sequence_manifest = root / f"{production}_sequences.json"
        sequence_manifest.write_text(json.dumps({
            "schema": 1,
            "production_id": production,
            "sequences": [{"id": f"{production}_000"}],
        }), encoding="utf-8")
        manifest = root / f"{production}.json"
        manifest.write_text(json.dumps({
            "schema": 2,
            "production_id": production,
            "source_kind": "mono-video",
            "source_container": source_container,
            "source_sequence_manifest": sequence_manifest.name,
            "split": split,
            "global_policy_weight": 1.0,
            "context_frame_count": 10,
            "context_fps": 24.0,
            "label_frame_count": 2,
            "shot_count": 1,
            "video_sha256": split_audit.sha256(sequence_manifest),
        }), encoding="utf-8")
        return manifest, sequence_manifest

    @staticmethod
    def write_native_manifest(root, production, split, video_id=None,
                              capture_group=None, media_bytes=None):
        video_id = video_id or f"{production}_video"
        capture_group = capture_group or f"{production}_capture"
        media = root / f"{production}_{video_id}.mp4"
        media.write_bytes(media_bytes or production.encode("ascii"))
        video_hash = split_audit.sha256(media)
        selection = root / f"{production}_selection.json"
        selection.write_text(json.dumps({
            "schema": 1,
            "clips": [{
                "video_id": video_id,
                "capture_group_id": capture_group,
                "split": split,
            }],
        }), encoding="utf-8")
        receipt = root / f"{production}_receipt.json"
        receipt.write_text(json.dumps({
            "schema": 1,
            "license": "CC BY",
            "license_url": "https://example.test/license",
            "accepted": [{
                "video_id": video_id,
                "capture_group_id": capture_group,
                "split": split,
                "download": {
                    "video_id": video_id,
                    "path": str(media),
                    "sha256": video_hash,
                },
            }],
        }), encoding="utf-8")
        manifest = root / f"{production}.json"
        manifest.write_text(json.dumps({
            "schema": 2,
            "production_id": production,
            "source_kind": "native-hdr-video",
            "dataset": "fixture-native-pq",
            "domain": "native_hdr_cinematic",
            "license": "CC BY",
            "policy_role": "cinematic_training",
            "split": split,
            "global_policy_weight": 1.0,
            "source_video_count": 1,
            "source_frame_count": 3,
            "frame_count": 3,
            "label_frame_count": 1,
            "source_provenance": {
                "selection_manifest": str(selection),
                "selection_manifest_sha256": split_audit.sha256(selection),
                "download_receipt": str(receipt),
                "download_receipt_sha256": split_audit.sha256(receipt),
            },
            "sequences": [{
                "clip": f"{production}_window",
                "video_id": video_id,
                "capture_group_id": capture_group,
                "split": split,
                "frames": 3,
                "label_frames": 1,
                "source_frame_rate": 24.0,
            }],
        }), encoding="utf-8")
        return manifest, selection, receipt

    def test_repository_active_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_rows = []
            manifests = []
            for production, split in self.assignments():
                source_rows.append({
                    "id": production,
                    "production_id": production,
                    "source_group": production,
                    "split": split,
                    "admission": "global_policy",
                    "complete_production": True,
                    "global_policy_weight": 1.0,
                    "license": "CC BY",
                    "license_url": "https://example.test/license",
                    "eye_order": "first-left",
                    "eye_display_aspect_ratio": 16 / 9,
                    "retrieval": {"kind": "unavailable"},
                })
                manifest = root / f"{production}.json"
                video = root / f"{production}.mp4"
                video.write_bytes(production.encode("ascii"))
                manifest.write_text(json.dumps({
                    "schema": 1,
                    "film_id": production,
                    "split": split,
                    "global_policy_weight": 1.0,
                    "context_frame_count": 10,
                    "context_fps": 24.0,
                    "sample_count": 2,
                    "shot_count": 1,
                    "video": str(video),
                    "video_sha256": split_audit.sha256(video),
                }), encoding="utf-8")
                manifests.append(manifest)
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "schema": 1, "sources": source_rows,
            }), encoding="utf-8")
            result = split_audit.audit(catalog, manifests)
            self.assertEqual(result["totals"]["productions"], 4)
            self.assertEqual(len(result["split_productions"]["test"]), 2)
            self.assertEqual(
                {row["source_kind"] for row in result["productions"]},
                {"authored-stereo"},
            )

    def test_repository_legacy_catalog_remains_compatible(self):
        catalog = sources.load_catalog(
            Path(__file__).with_name("artistic_stereo_sources.json")
        )
        self.assertEqual(catalog["schema"], 1)
        self.assertEqual(
            {row["source_kind"] for row in catalog["sources"]},
            {"authored-stereo"},
        )

    def test_repository_default_catalog_is_generic_and_keeps_stereo_auxiliary(self):
        self.assertEqual(
            split_audit.DEFAULT_CATALOG.name,
            sources.DEFAULT_ACTIVE_CATALOG_NAME,
        )
        catalog = sources.load_catalog(split_audit.DEFAULT_CATALOG)
        self.assertEqual(catalog["schema"], 2)
        self.assertTrue(catalog["sources"])
        self.assertEqual(
            {row["source_kind"] for row in catalog["sources"]},
            {"authored-stereo"},
        )
        self.assertEqual(
            {row["admission"] for row in catalog["sources"]},
            {"stereo_auxiliary"},
        )
        self.assertFalse(any(
            row["admission"] == "global_policy"
            for row in catalog["sources"]
        ))

    def test_schema_2_auxiliary_migration_catalog_needs_no_fake_active_split(self):
        catalog = sources.validate_catalog({
            "schema": 2,
            "sources": [{
                "id": "stereo_aux",
                "production_id": "stereo_aux",
                "source_kind": "authored-stereo",
                "source_group": "archive",
                "split": None,
                "admission": "stereo_auxiliary",
                "license": "CC BY",
                "license_url": "https://example.test/license",
            }],
        })
        self.assertEqual(catalog["sources"][0]["admission"], "stereo_auxiliary")

    def test_schema_2_mono_catalog_and_manifests_need_no_eye_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.source(production, split)
                for production, split in self.assignments()
            ]
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "schema": 2,
                "sources": rows,
            }), encoding="utf-8")
            manifests = [
                self.write_manifest(root, production, split, 2, "mono-video")
                for production, split in self.assignments()
            ]

            result = split_audit.audit(catalog, manifests)

            self.assertEqual(result["totals"]["label_frames"], 8)
            self.assertEqual(
                {row["dataset_manifest_schema"]
                 for row in result["productions"]},
                {2},
            )
            self.assertEqual(
                {row["source_kind"] for row in result["productions"]},
                {"mono-video"},
            )

    def test_schema_2_image_sequences_use_manifest_compatibility_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.source(production, split)
                for production, split in self.assignments()
            ]
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "schema": 2,
                "sources": rows,
            }), encoding="utf-8")
            prepared = [
                self.write_sequence_manifest(root, production, split)
                for production, split in self.assignments()
            ]

            result = split_audit.audit(
                catalog, [manifest for manifest, _ in prepared]
            )

            identities = {
                row["production_id"]: row["video_sha256"]
                for row in result["productions"]
            }
            self.assertEqual(
                identities,
                {
                    production: split_audit.sha256(sequence_manifest)
                    for (production, _), (_, sequence_manifest) in zip(
                        self.assignments(), prepared
                    )
                },
            )
            self.assertFalse(any(
                "source_sequence_manifest" in row
                for row in result["productions"]
            ))
            self.assertTrue(all(
                row["source_sequence_manifest_sha256"] ==
                row["video_sha256"]
                for row in result["productions"]
            ))

    def test_schema_2_derived_image_sequences_use_manifest_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.source(production, split)
                for production, split in self.assignments()
            ]
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "schema": 2,
                "sources": rows,
            }), encoding="utf-8")
            prepared = [
                self.write_sequence_manifest(
                    root, production, split,
                    source_container="derived-public-image-sequences",
                )
                for production, split in self.assignments()
            ]

            result = split_audit.audit(
                catalog, [manifest for manifest, _ in prepared]
            )

            self.assertEqual(
                {
                    row["production_id"]: row["video_sha256"]
                    for row in result["productions"]
                },
                {
                    production: split_audit.sha256(sequence_manifest)
                    for (production, _), (_, sequence_manifest) in zip(
                        self.assignments(), prepared
                    )
                },
            )
            self.assertEqual(
                {row["source_identity_kind"]
                 for row in result["productions"]},
                {"source_sequence_manifest_sha256"},
            )

    def test_schema_2_image_sequence_identity_detects_manifest_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.source(production, split)
                for production, split in self.assignments()
            ]
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "schema": 2,
                "sources": rows,
            }), encoding="utf-8")
            prepared = [
                self.write_sequence_manifest(root, production, split)
                for production, split in self.assignments()
            ]
            prepared[0][1].write_text(
                '{"schema": 1, "sequences": []}', encoding="utf-8"
            )

            with self.assertRaisesRegex(
                    RuntimeError, "source sequence manifest hash changed"):
                split_audit.audit(
                    catalog, [manifest for manifest, _ in prepared]
                )

    def test_schema_2_image_sequence_manifest_must_be_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.source(production, split)
                for production, split in self.assignments()
            ]
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "schema": 2,
                "sources": rows,
            }), encoding="utf-8")
            prepared = [
                self.write_sequence_manifest(root, production, split)
                for production, split in self.assignments()
            ]
            manifest = prepared[0][0]
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["source_sequence_manifest"] = str(prepared[0][1].resolve())
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                    RuntimeError, "source sequence manifest must be relative"):
                split_audit.audit(
                    catalog, [item[0] for item in prepared]
                )

    def test_schema_2_accepts_all_generic_source_kinds(self):
        temporal_kinds = (
            "mono-video",
            "native-hdr-video",
            "authored-stereo",
            "gt-depth-flow",
        )
        rows = [
            self.source(production, split, kind)
            for (production, split), kind in zip(
                self.assignments(), temporal_kinds
            )
        ]
        rows.append({
            "id": "spatial_aux",
            "source_kind": "still-spatial",
            "source_group": "spatial_collection",
            "admission": "spatial_auxiliary",
            "license": "CC BY",
            "license_url": "https://example.test/license",
        })
        catalog = sources.validate_catalog({"schema": 2, "sources": rows})
        self.assertEqual(
            {row["source_kind"] for row in catalog["sources"]},
            sources.SOURCE_KINDS,
        )

    def test_native_hdr_collection_uses_authenticated_leaf_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.source(production, split, "native-hdr-video")
                for production, split in self.assignments()
            ]
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "schema": 2, "sources": rows,
            }), encoding="utf-8")
            prepared = [
                self.write_native_manifest(root, production, split)
                for production, split in self.assignments()
            ]

            result = split_audit.audit(
                catalog, [manifest for manifest, _, _ in prepared]
            )

            for row in result["productions"]:
                self.assertEqual(
                    row["source_identity_kind"],
                    split_audit.NATIVE_HDR_COLLECTION_IDENTITY_KIND,
                )
                self.assertEqual(len(row["source_videos"]), 1)
                self.assertNotIn("video_sha256", row)
                self.assertEqual(row["context_frame_rates"], [24.0])
                self.assertEqual(row["shots"], 1)

    def test_native_hdr_collection_rejects_cross_split_leaf_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.source(production, split, "native-hdr-video")
                for production, split in self.assignments()
            ]
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "schema": 2, "sources": rows,
            }), encoding="utf-8")
            prepared = []
            for production, split in self.assignments():
                shared = production in {"train", "dev"}
                prepared.append(self.write_native_manifest(
                    root, production, split,
                    video_id="shared_video" if shared else None,
                    capture_group="shared_capture" if shared else None,
                    media_bytes=b"shared" if shared else None,
                ))

            with self.assertRaisesRegex(
                    RuntimeError, "multiple productions"):
                split_audit.audit(
                    catalog, [manifest for manifest, _, _ in prepared]
                )

    def test_native_hdr_collection_rejects_changed_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.source(production, split, "native-hdr-video")
                for production, split in self.assignments()
            ]
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "schema": 2, "sources": rows,
            }), encoding="utf-8")
            prepared = [
                self.write_native_manifest(root, production, split)
                for production, split in self.assignments()
            ]
            prepared[0][2].write_text('{"schema":1,"accepted":[]}',
                                      encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "missing or changed"):
                split_audit.audit(
                    catalog, [manifest for manifest, _, _ in prepared]
                )

    def test_catalog_rejects_boolean_schema(self):
        rows = [
            self.source(production, split, "authored-stereo")
            for production, split in self.assignments()
        ]
        with self.assertRaisesRegex(RuntimeError, "source catalog schema"):
            sources.validate_catalog({"schema": True, "sources": rows})

    def test_audit_rejects_boolean_dataset_manifest_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.source(production, split, "authored-stereo")
                for production, split in self.assignments()
            ]
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "schema": 2,
                "sources": rows,
            }), encoding="utf-8")
            manifests = [
                self.write_manifest(
                    root, production, split, 1, "authored-stereo"
                )
                for production, split in self.assignments()
            ]
            payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            payload["schema"] = True
            manifests[0].write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "dataset manifest"):
                split_audit.audit(catalog, manifests)

    def test_audit_rejects_nonfinite_context_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.source(production, split)
                for production, split in self.assignments()
            ]
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "schema": 2,
                "sources": rows,
            }), encoding="utf-8")
            manifests = [
                self.write_manifest(root, production, split, 2, "mono-video")
                for production, split in self.assignments()
            ]
            payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            payload["context_fps"] = float("nan")
            manifests[0].write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "full-cadence context"):
                split_audit.audit(catalog, manifests)

    def test_still_spatial_cannot_enter_global_policy(self):
        rows = [
            self.source(production, split)
            for production, split in self.assignments()
        ]
        rows[0]["source_kind"] = "still-spatial"
        with self.assertRaisesRegex(RuntimeError, "spatial_auxiliary"):
            sources.validate_catalog({"schema": 2, "sources": rows})

    def test_authored_stereo_alone_requires_verified_eyes(self):
        rows = [
            self.source(production, split)
            for production, split in self.assignments()
        ]
        rows[0]["source_kind"] = "authored-stereo"
        with self.assertRaisesRegex(RuntimeError, "eye order"):
            sources.validate_catalog({"schema": 2, "sources": rows})

    def test_catalog_rejects_production_split_leak(self):
        rows = [
            self.source(production, split)
            for production, split in self.assignments()
        ]
        leaked = self.source("train_copy", "development")
        leaked["production_id"] = "train"
        rows.append(leaked)
        with self.assertRaisesRegex(RuntimeError, "leaks across"):
            sources.validate_catalog({"schema": 2, "sources": rows})

    def test_catalog_rejects_nonpositive_weight_and_missing_license(self):
        rows = [
            self.source(production, split)
            for production, split in self.assignments()
        ]
        rows[0]["global_policy_weight"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "weight is not positive"):
            sources.validate_catalog({"schema": 2, "sources": rows})
        rows[0]["global_policy_weight"] = 1.0
        rows[0].pop("license_url")
        with self.assertRaisesRegex(RuntimeError, "license provenance"):
            sources.validate_catalog({"schema": 2, "sources": rows})

    def test_catalog_requires_independent_sealed_test_groups(self):
        rows = [
            self.source(production, split)
            for production, split in self.assignments()
        ]
        rows[2]["source_group"] = "shared"
        rows[3]["source_group"] = "shared"
        with self.assertRaisesRegex(RuntimeError, "independent source groups"):
            sources.validate_catalog({"schema": 2, "sources": rows})


if __name__ == "__main__":
    unittest.main()
