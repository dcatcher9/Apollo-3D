#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import fetch_artistic_stereo_sources as fetch


class ArtisticStereoSourceTests(unittest.TestCase):
    def write_catalog(self, root, sources):
        path = root / "sources.json"
        path.write_text(
            json.dumps({"schema": 1, "sources": sources}),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def source(identifier, production, split):
        return {
            "id": identifier,
            "production_id": production,
            "source_group": identifier,
            "split": split,
            "admission": "global_policy",
            "complete_production": True,
            "global_policy_weight": 1.0,
            "license": "CC BY",
            "license_url": "https://example.test/license",
            "eye_order": "first-left",
            "eye_display_aspect_ratio": 1.7777777778,
            "retrieval": {"kind": "unavailable"},
        }

    def test_catalog_requires_two_sealed_productions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [
                self.source("train", "train", "training"),
                self.source("test", "test", "test"),
            ]
            path = self.write_catalog(root, sources)
            with self.assertRaisesRegex(RuntimeError, "two sealed"):
                fetch.load_catalog(path)

    def test_catalog_rejects_production_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [
                self.source("train", "same", "training"),
                self.source("test_a", "same", "test"),
                self.source("test_b", "other", "test"),
            ]
            path = self.write_catalog(root, sources)
            with self.assertRaisesRegex(RuntimeError, "leaks across"):
                fetch.load_catalog(path)

    def test_catalog_requires_independent_sealed_source_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [
                self.source("train", "train", "training"),
                self.source("test_a", "test_a", "test"),
                self.source("test_b", "test_b", "test"),
            ]
            sources[1]["source_group"] = "same_publisher"
            sources[2]["source_group"] = "same_publisher"
            path = self.write_catalog(root, sources)
            with self.assertRaisesRegex(RuntimeError, "independent source"):
                fetch.load_catalog(path)

    def test_repository_catalog_is_valid(self):
        path = Path(fetch.__file__).with_name("artistic_stereo_sources.json")
        catalog = fetch.load_catalog(path)
        self.assertEqual(catalog["schema"], 1)

    def test_global_source_rejects_unrectified_stereo(self):
        source = self.source("test", "test", "test")
        audit = {
            "absolute_vertical_mismatch_pct": {"p50": 8.0, "p95": 58.0},
            "sample_vertical_median_pct": {"p50": 8.0, "p95": 12.0},
        }
        with self.assertRaisesRegex(RuntimeError, "rectification failed"):
            fetch.validate_audit(source, audit)

    def test_global_source_accepts_rectified_stereo(self):
        source = self.source("test", "test", "test")
        audit = {
            "absolute_vertical_mismatch_pct": {"p50": 0.2, "p95": 0.9},
            "sample_vertical_median_pct": {"p50": 0.1, "p95": 0.2},
        }
        fetch.validate_audit(source, audit)


if __name__ == "__main__":
    unittest.main()
