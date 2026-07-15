#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import audit_artistic_dataset_splits as split_audit


class ArtisticDatasetSplitAuditTests(unittest.TestCase):
    def test_repository_active_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_rows = []
            manifests = []
            assignments = (
                ("train", "training"),
                ("dev", "development"),
                ("test_a", "test"),
                ("test_b", "test"),
            )
            for production, split in assignments:
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


if __name__ == "__main__":
    unittest.main()
