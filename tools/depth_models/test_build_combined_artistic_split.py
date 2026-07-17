#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import audit_artistic_dataset_splits as split_audit
import build_combined_artistic_split as combined
from test_audit_artistic_dataset_splits import ArtisticDatasetSplitAuditTests


class CombinedArtisticSplitTests(unittest.TestCase):
    def fixture(self, root):
        assignments = ArtisticDatasetSplitAuditTests.assignments()
        base_catalog = root / "base_catalog.json"
        base_catalog.write_text(json.dumps({
            "schema": 2,
            "sealed_test_policy": "fixture sealed tests stay unopened",
            "sources": [
                ArtisticDatasetSplitAuditTests.source(production, split)
                for production, split in assignments
            ],
        }), encoding="utf-8")
        base_manifests = [
            ArtisticDatasetSplitAuditTests.write_manifest(
                root, production, split, 2, "mono-video"
            )
            for production, split in assignments
        ]
        base_payload = split_audit.audit(base_catalog, base_manifests)
        base_active = root / "base_active.json"
        base_active.write_text(json.dumps(base_payload), encoding="utf-8")

        native_root = root / "native"
        native_root.mkdir()
        native_entries = {}
        for split in ("training", "development"):
            production = f"native_{split}"
            manifest, _, _ = (
                ArtisticDatasetSplitAuditTests.write_native_manifest(
                    native_root, production, split
                )
            )
            native_entries[split] = {
                "dataset_manifest": str(manifest),
                "dataset_manifest_sha256": split_audit.sha256(manifest),
            }
        bootstrap = native_root / "bootstrap.json"
        bootstrap.write_text(json.dumps({
            "schema": combined.NATIVE_BOOTSTRAP_SCHEMA,
            "contract": combined.NATIVE_BOOTSTRAP_CONTRACT,
            "datasets": native_entries,
        }), encoding="utf-8")
        return base_active, bootstrap

    def test_builds_one_fail_closed_combined_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_active, bootstrap = self.fixture(root)
            output_catalog = root / "combined_catalog.json"
            output_active = root / "combined_active.json"

            catalog, active = combined.build(
                base_active, bootstrap, output_catalog, output_active
            )

            self.assertTrue(output_catalog.is_file())
            self.assertTrue(output_active.is_file())
            self.assertEqual(len(catalog["sources"]), 6)
            self.assertEqual(
                active["split_productions"]["training"],
                ["native_training", "train"],
            )
            self.assertEqual(
                active["split_productions"]["development"],
                ["dev", "native_development"],
            )
            self.assertEqual(active["split_productions"]["test"],
                             ["test_a", "test_b"])
            native = [
                row for row in active["productions"]
                if row["source_kind"] == "native-hdr-video"
            ]
            self.assertEqual(len(native), 2)
            self.assertTrue(all(
                row["source_identity_kind"] ==
                split_audit.NATIVE_HDR_COLLECTION_IDENTITY_KIND
                for row in native
            ))

    def test_rejects_stale_native_dataset_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_active, bootstrap = self.fixture(root)
            payload = json.loads(bootstrap.read_text(encoding="utf-8"))
            payload["datasets"]["training"]["dataset_manifest_sha256"] = "0" * 64
            bootstrap.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "stale native HDR"):
                combined.build(
                    base_active, bootstrap,
                    root / "combined_catalog.json",
                    root / "combined_active.json",
                )


if __name__ == "__main__":
    unittest.main()
