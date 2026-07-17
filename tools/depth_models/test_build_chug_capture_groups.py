#!/usr/bin/env python3

import base64
import tempfile
import unittest
from pathlib import Path

import numpy as np

import build_chug_capture_groups as groups


class ChugCaptureGroupTests(unittest.TestCase):
    @staticmethod
    def row(video_id, content_name):
        return {
            "video_id": video_id,
            "content_name": content_name,
        }

    @staticmethod
    def review(positives=None, negatives=None):
        return {
            "visual_identity": {
                "anchor_fractions": [0.1, 0.3, 0.5, 0.7, 0.9],
                "fingerprint_transform": "test-transform-v1",
                "feature_contract": "test-feature-contract-v1",
                "depth_weights_sha256": "1" * 64,
            },
            "capture_session": {
                "img_max_consecutive_gap": 2,
                "img_max_group_span": 2,
                "timestamp_max_consecutive_seconds": 120,
                "timestamp_max_group_span_seconds": 120,
                "perceptual_img_max_delta": 150,
                "perceptual_timestamp_max_seconds": 7200,
            },
            "perceptual_thresholds": {
                "bidirectional_nearest_median_min": 0.90,
                "manual_review_margin": 0.002,
            },
            "confirmed_positive_groups": positives or [],
            "manual_capture_session_groups": [],
            "perceptual_boundary_overrides": [],
            "hard_negative_pairs": negatives or [],
        }

    @staticmethod
    def embedding(axis, perturbation=0.0):
        values = np.zeros((groups.ANCHOR_COUNT, groups.FEATURE_DIMENSION), dtype=np.float32)
        values[:, axis] = 1.0
        if perturbation:
            values[:, (axis + 1) % groups.FEATURE_DIMENSION] = perturbation
            values /= np.linalg.norm(values, axis=1, keepdims=True)
        quantized = np.rint(values * 32767.0).astype("<i2")
        yy, xx = np.indices((64, 64))
        gray = np.stack([
            ((xx * (axis + 3) + yy * (axis + 7) + index * 19) % 256).astype(np.uint8)
            for index in range(groups.ANCHOR_COUNT)
        ])
        return {
            "identity": {
                "embedding_shape": [groups.ANCHOR_COUNT, groups.FEATURE_DIMENSION],
                "embedding_q15": base64.b64encode(quantized.tobytes()).decode("ascii"),
                "canonical_gray_shape": [groups.ANCHOR_COUNT, 64, 64],
                "canonical_gray_u8": base64.b64encode(gray.tobytes()).decode("ascii"),
                "phash64": [groups.perceptual_hash(frame) for frame in gray],
            },
        }

    def test_img_session_grouping_catches_confirmed_filename_leaks(self):
        rows = [
            self.row("a", "Files_Dec21_IMG_1164_0_6.mp4"),
            self.row("b", "Files_Dec21_IMG_1165_0_8.mp4"),
            self.row("c", "Files_Dec21_IMG_1181_0_5.mp4"),
            self.row("d", "Files_Dec21_IMG_1300_0_5.mp4"),
        ]
        edges = groups.capture_session_edges(rows, self.review())
        pairs = {tuple(sorted((edge["first"], edge["second"]))) for edge in edges}
        self.assertEqual(pairs, {("a", "b")})

    def test_img_session_span_prevents_transitive_day_wide_chain(self):
        rows = [
            self.row("a", "Files_Dec21_IMG_1000_0_6.mp4"),
            self.row("b", "Files_Dec21_IMG_1002_0_6.mp4"),
            self.row("c", "Files_Dec21_IMG_1004_0_6.mp4"),
            self.row("d", "Files_Dec21_IMG_1006_0_6.mp4"),
        ]
        edges = groups.capture_session_edges(rows, self.review())
        pairs = {tuple(sorted((edge["first"], edge["second"]))) for edge in edges}
        self.assertEqual(pairs, {("a", "b"), ("c", "d")})

    def test_timestamp_session_catches_lake_pair(self):
        rows = [
            self.row("a", "Files_Dec5_PXL_20231205_135710551_5_15.mp4"),
            self.row("b", "Files_Dec5_PXL_20231205_135746603_2_12.mp4"),
            self.row("c", "Files_Dec5_PXL_20231205_145746603_2_12.mp4"),
        ]
        edges = groups.capture_session_edges(rows, self.review())
        self.assertEqual(len(edges), 1)
        self.assertEqual({edges[0]["first"], edges[0]["second"]}, {"a", "b"})

    def test_pair_score_matches_anchor_permutation(self):
        first = np.zeros((groups.ANCHOR_COUNT, groups.FEATURE_DIMENSION), dtype=np.float32)
        for index in range(groups.ANCHOR_COUNT):
            first[index, index] = 1.0
        second = first[::-1].copy()
        score = groups.pair_score(first, second)
        self.assertAlmostEqual(score["best_anchor_cosine"], 1.0)
        self.assertAlmostEqual(score["top3_anchor_cosine_mean"], 1.0)

    def test_perceptual_auto_merge_requires_plausible_capture_neighborhood(self):
        rows = [
            self.row("a", "Files_Oct30_IMG_1000_0_6.mp4"),
            self.row("b", "Files_Oct30_IMG_1100_0_6.mp4"),
            self.row("c", "Files_Oct30_IMG_2000_0_6.mp4"),
        ]
        identities = {video_id: self.embedding(0) for video_id in ("a", "b", "c")}
        _groups, membership, _evidence = groups.build_groups(rows, identities, self.review())
        self.assertEqual(membership["a"], membership["b"])
        self.assertNotEqual(membership["a"], membership["c"])

    def test_manual_accept_overrides_perceptual_threshold_and_neighborhood(self):
        rows = [
            self.row("a", "Files_Oct30_IMG_1000_0_6.mp4"),
            self.row("b", "Files_Oct30_IMG_2000_0_6.mp4"),
        ]
        review = self.review()
        review["perceptual_boundary_overrides"] = [{
            "name": "reviewed_same_session",
            "video_ids": ["a", "b"],
            "decision": "accept",
        }]
        identities = {"a": self.embedding(0), "b": self.embedding(2)}
        _groups, membership, _evidence = groups.build_groups(rows, identities, review)
        self.assertEqual(membership["a"], membership["b"])

    def test_build_groups_is_order_independent_and_keeps_hard_negative_apart(self):
        rows = [
            self.row("a", "Files_Dec21_IMG_1164_0_6.mp4"),
            self.row("b", "Files_Dec21_IMG_1165_0_8.mp4"),
            self.row("c", "Files_Nov1_IMG_5000_0_8.mp4"),
        ]
        review = self.review(
            positives=[{"name": "same", "video_ids": ["a", "b"]}],
            negatives=[{"name": "different", "video_ids": ["a", "c"]}],
        )
        identities = {
            "a": self.embedding(0),
            "b": self.embedding(0, 0.01),
            "c": self.embedding(2),
        }
        first, membership, evidence = groups.build_groups(rows, identities, review)
        second, reversed_membership, _ = groups.build_groups(
            list(reversed(rows)), identities, review
        )
        self.assertEqual(first, second)
        self.assertEqual(membership, reversed_membership)
        self.assertEqual(membership["a"], membership["b"])
        self.assertNotEqual(membership["a"], membership["c"])
        self.assertEqual(
            {item["label"] for item in evidence["calibration_pairs"]},
            {"positive", "negative"},
        )

    def test_visual_cache_is_bound_and_requires_thumbnail_identity(self):
        embedding = self.embedding(0)["identity"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            thumbnail = root / "visual_thumbnails" / "a.png"
            thumbnail.parent.mkdir(parents=True)
            thumbnail.write_bytes(b"png")
            embedding["thumbnail"] = {
                "path": "visual_thumbnails/a.png",
                "bytes": 3,
                "sha256": groups.chug.sha256(thumbnail),
            }
            cache = root / "visual_identity" / "a.json"
            binding = {"video_id": "a", "source_url": "https://example.test/a"}
            groups.chug.atomic_write_json(cache, {
                "schema": groups.VISUAL_CACHE_SCHEMA,
                "binding": binding,
                "identity": embedding,
            })
            self.assertIsNotNone(groups.load_visual_cache(cache, binding, root))
            self.assertIsNone(groups.load_visual_cache(cache, {**binding, "source_url": "changed"}, root))
            thumbnail.write_bytes(b"bad")
            self.assertIsNone(groups.load_visual_cache(cache, binding, root))


if __name__ == "__main__":
    unittest.main()
