"""Focused tests for authoritative metadata used by metric-only rescoring."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import rescore_run  # noqa: E402
import run_eval  # noqa: E402


class RescoreMetadataTests(unittest.TestCase):
    def _fixture(self, root):
        clips_root = os.path.join(root, "clips")
        source_dir = os.path.join(clips_root, "demo")
        run_dir = os.path.join(root, "run")
        artifact_dir = os.path.join(run_dir, "demo")
        os.makedirs(source_dir)
        os.makedirs(artifact_dir)
        source = np.full((8, 12, 3), 72, np.uint8)
        Image.fromarray(source, "RGB").save(
            os.path.join(source_dir, "frame_00000.png"))
        with open(os.path.join(source_dir, "meta.json"), "w", encoding="utf-8") as stream:
            json.dump({
                "name": "source authority",
                "expected_flat": False,
                "content_type": "synthetic",
            }, stream)

        Image.fromarray(np.concatenate((source, source), axis=1), "RGB").save(
            os.path.join(artifact_dir, "sbs_00000.png"))
        Image.fromarray(np.full((8, 12), 128, np.uint8), "L").save(
            os.path.join(artifact_dir, "depth_00000.png"))
        Image.fromarray(np.zeros((8, 24, 3), np.uint8), "RGB").save(
            os.path.join(artifact_dir, "warp_mask_00000.png"))
        np.zeros((8, 24), np.float32).tofile(
            os.path.join(artifact_dir, "warp_map_00000.f32"))
        contract = {
            "schema": 16,
            "model": "depth_anything_v2_fp16",
            "profile": "apollo",
            "depth_step": "current-once",
            "depth_reuse_interval": 1,
            "depth_compensation": "none",
            "literal_bestv2": False,
            "cuda_graph": True,
            "adaptive_pop": True,
            "adaptive_pop_max": 1.3,
            "zero_plane": "legacy",
        }
        with open(os.path.join(artifact_dir, "contract.json"), "w",
                  encoding="utf-8") as stream:
            json.dump(contract, stream)
        artifact_hash = run_eval.scored_artifact_sha256(artifact_dir)
        data = {
            "meta": {
                "suite": "core",
                "clip_set_sha1": {"demo": run_eval.sha1_dir(source_dir)},
                "scored_artifact_sha256": {"demo": artifact_hash},
                **{key: contract[key] for key in (
                    "model", "profile", "depth_step", "depth_reuse_interval",
                    "depth_compensation", "literal_bestv2", "cuda_graph",
                    "adaptive_pop", "adaptive_pop_max", "zero_plane")},
            },
            "clips": {"demo": {"meta": {
                "name": "forged cache",
                "expected_flat": True,
                "source_frame_count": 99,
                "model": "forged-model",
            }}},
        }
        return data, clips_root, run_dir, source_dir

    def test_rescore_rebuilds_meta_without_cached_values(self):
        with tempfile.TemporaryDirectory() as root:
            data, clips_root, run_dir, _ = self._fixture(root)
            rebuilt = rescore_run.authoritative_clip_meta(
                data, "demo", clips_root, run_dir)
            self.assertEqual(rebuilt["name"], "source authority")
            self.assertIs(rebuilt["expected_flat"], False)
            self.assertEqual(rebuilt["source_frame_count"], 1)
            self.assertEqual(rebuilt["model"], "depth_anything_v2_fp16")

    def test_report_authentication_rejects_cached_scoring_meta(self):
        with tempfile.TemporaryDirectory() as root:
            data, clips_root, run_dir, _ = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "clips.demo.meta"):
                run_eval._authenticated_remeasurement_clip_meta(
                    data, "demo", clips_root, run_dir)

    def test_rescore_rejects_incomplete_image_frame_set(self):
        with tempfile.TemporaryDirectory() as root:
            data, clips_root, run_dir, source_dir = self._fixture(root)
            Image.fromarray(np.full((8, 12, 3), 96, np.uint8), "RGB").save(
                os.path.join(source_dir, "frame_00001.png"))
            with self.assertRaisesRegex(SystemExit, "frame-id mismatch"):
                rescore_run.authoritative_clip_meta(data, "demo", clips_root, run_dir)


if __name__ == "__main__":
    unittest.main()
