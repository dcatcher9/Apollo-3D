"""Focused tests for authoritative metadata used by metric-only rescoring."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import rescore_run  # noqa: E402
import run_eval  # noqa: E402
import sbsbench  # noqa: E402


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
            "schema": 17,
            "model": "depth_anything_v2_fp16",
            "profile": "apollo",
            "depth_step": "current-once",
            "depth_reuse_interval": 1,
            "depth_compensation": "none",
            "literal_bestv2": False,
            "cuda_graph": True,
            "adaptive_pop": True,
            "adaptive_pop_max": 1.3,
            "zero_plane": "median",
            "subject_state": {
                "file": "subject_state.json", "schema": 1,
                "capture": "every-source-frame-after-estimator-update",
            },
        }
        with open(os.path.join(artifact_dir, "contract.json"), "w",
                  encoding="utf-8") as stream:
            json.dump(contract, stream)
        with open(os.path.join(artifact_dir, "subject_state.json"), "w",
                  encoding="utf-8") as stream:
            json.dump({
                "schema": 1,
                "source": "depth_subject_resolve_cs.SubjectState",
                "capture": "every-source-frame-after-estimator-update",
                "fields": list(sbsbench.SUBJECT_STATE_FIELDS),
                "frames": [{"frame_id": "00000", "values": [
                    0.0, 0.0, 0.5, 1.0, 0.0, 1.0,
                    0.0, 1.0, 0.0, 1.0, 3.0, 1.0,
                ]}],
            }, stream)
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

    def _two_clip_main_fixture(self, root):
        clips_root = os.path.join(root, "clips")
        run_dir = os.path.join(root, "run")
        os.makedirs(clips_root)
        os.makedirs(run_dir)
        clip_names = ["zeta", "alpha"]
        data = {
            "meta": {
                "run_kind": "comparison-only",
                "eval_schema": run_eval.EVAL_SCHEMA,
                "clips_root": clips_root,
                "clip_set_sha1": {
                    clip: f"source-sha1-{clip}" for clip in clip_names},
                "scored_artifact_sha256": {
                    clip: f"artifact-sha256-{clip}" for clip in clip_names},
            },
            "clips": {
                clip: {"meta": {}, "perf_ms": {"forged": 999.0}}
                for clip in clip_names
            },
        }
        result_path = os.path.join(run_dir, "results.json")
        with open(result_path, "w", encoding="utf-8") as stream:
            json.dump(data, stream)
        return clip_names, clips_root, run_dir, result_path

    def test_main_parallelizes_in_input_order_and_replaces_output_atomically(self):
        with tempfile.TemporaryDirectory() as root:
            clip_names, clips_root, run_dir, result_path = (
                self._two_clip_main_fixture(root))
            source_calls = []
            artifact_calls = []
            measurement_calls = []

            def source_digests(path):
                clip = os.path.basename(path)
                source_calls.append(clip)
                return f"source-sha1-{clip}", f"source-sha256-{clip}"

            def artifact_digests(path):
                clip = os.path.basename(path)
                artifact_calls.append(clip)
                return f"artifact-sha256-{clip}", f"numeric-sha256-{clip}"

            def clip_meta(_data, clip, _clips_root, _run_dir,
                          source_sha1=None, artifact_sha256=None):
                self.assertEqual(source_sha1, f"source-sha1-{clip}")
                self.assertEqual(artifact_sha256, f"artifact-sha256-{clip}")
                return {
                    "scored_artifact_sha256": artifact_sha256,
                    "source_frame_count": 1,
                }

            def measure(sequence_jobs, jobs):
                measurement_calls.append((list(sequence_jobs), jobs))
                return [
                    (clip, ([{"_frame_id": index, "quality": index + 1.0}],
                            {"quality": index + 1.0}))
                    for index, (clip, _artifact_dir, _source_dir)
                    in enumerate(sequence_jobs)
                ]

            output_path = os.path.join(run_dir, "results.rescored.json")
            real_replace = os.replace
            argv = [
                "rescore_run.py", run_dir, "--clips-root", clips_root,
                "--jobs", "2",
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(
                        rescore_run.run_eval, "source_evidence_digests",
                        side_effect=source_digests), \
                    mock.patch.object(
                        rescore_run.run_eval, "scored_artifact_digests",
                        side_effect=artifact_digests), \
                    mock.patch.object(
                        rescore_run.run_eval, "load_perf_metrics",
                        return_value={"depth_infer": 1.25}), \
                    mock.patch.object(
                        rescore_run, "authoritative_clip_meta",
                        side_effect=clip_meta), \
                    mock.patch.object(
                        rescore_run.eval_parallel, "measure_clip_sequences",
                        side_effect=measure), \
                    mock.patch.object(
                        rescore_run.sbsbench, "filter_aggregate_by_evidence",
                        side_effect=lambda _rows, aggregate, _specs, _meta: aggregate), \
                    mock.patch.object(
                        rescore_run.run_eval, "score_clip_gates",
                        return_value=({}, [], [])), \
                    mock.patch.object(
                        rescore_run.run_eval, "primary_evidence_failures",
                        return_value=[]), \
                    mock.patch.object(
                        rescore_run.run_eval, "perf_evidence_failures",
                        return_value=[]), \
                    mock.patch.object(
                        rescore_run.run_eval, "build_frame_records",
                        side_effect=lambda rows, _thresholds, _meta: rows), \
                    mock.patch.object(
                        rescore_run.run_eval, "summarize_frame_labels",
                        return_value={}), \
                    mock.patch.object(
                        rescore_run.run_eval, "bind_training_labels_to_evidence_gate"), \
                    mock.patch.object(
                        rescore_run.os, "replace", wraps=real_replace) as replace:
                rescore_run.main()

            expected_jobs = [
                (clip, os.path.join(run_dir, clip), os.path.join(clips_root, clip))
                for clip in clip_names
            ]
            self.assertEqual(measurement_calls, [(expected_jobs, 2)])
            self.assertEqual(source_calls, clip_names + clip_names)
            self.assertEqual(artifact_calls, clip_names + clip_names)
            replace.assert_called_once_with(output_path + ".tmp", output_path)
            self.assertTrue(os.path.exists(output_path))
            self.assertFalse(os.path.exists(output_path + ".tmp"))

            with open(output_path, encoding="utf-8") as stream:
                rescored = json.load(stream)
            self.assertEqual(list(rescored["clips"]), clip_names)
            self.assertEqual(rescored["meta"]["scoring_jobs"], 2)
            self.assertEqual(
                [rescored["clips"][clip]["aggregate"]["quality"]
                 for clip in clip_names],
                [1.0, 2.0])
            self.assertTrue(all(
                rescored["clips"][clip]["perf_ms"] == {"depth_infer": 1.25}
                for clip in clip_names))
            with open(result_path, encoding="utf-8") as stream:
                original = json.load(stream)
            self.assertTrue(all(
                "aggregate" not in original["clips"][clip] for clip in clip_names))

    def test_main_rejects_post_measure_source_mutation_without_output(self):
        with tempfile.TemporaryDirectory() as root:
            clip_names, clips_root, run_dir, result_path = (
                self._two_clip_main_fixture(root))
            source_counts = {clip: 0 for clip in clip_names}

            def source_digests(path):
                clip = os.path.basename(path)
                source_counts[clip] += 1
                suffix = "-changed" if clip == "alpha" and source_counts[clip] > 1 else ""
                return (
                    f"source-sha1-{clip}{suffix}",
                    f"source-sha256-{clip}{suffix}",
                )

            def artifact_digests(path):
                clip = os.path.basename(path)
                return f"artifact-sha256-{clip}", f"numeric-sha256-{clip}"

            def clip_meta(_data, clip, _clips_root, _run_dir,
                          source_sha1=None, artifact_sha256=None):
                return {
                    "scored_artifact_sha256": artifact_sha256,
                    "source_frame_count": 1,
                }

            measured = [
                (clip, ([{"_frame_id": index}], {"quality": index + 1.0}))
                for index, clip in enumerate(clip_names)
            ]
            output_path = os.path.join(run_dir, "results.rescored.json")
            argv = [
                "rescore_run.py", run_dir, "--clips-root", clips_root,
                "--jobs", "2",
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(
                        rescore_run.run_eval, "source_evidence_digests",
                        side_effect=source_digests), \
                    mock.patch.object(
                        rescore_run.run_eval, "scored_artifact_digests",
                        side_effect=artifact_digests), \
                    mock.patch.object(
                        rescore_run, "authoritative_clip_meta",
                        side_effect=clip_meta), \
                    mock.patch.object(
                        rescore_run.eval_parallel, "measure_clip_sequences",
                        return_value=measured), \
                    self.assertRaisesRegex(
                        SystemExit, "alpha: source evidence changed during rescoring"):
                rescore_run.main()

            self.assertFalse(os.path.exists(output_path))
            self.assertFalse(os.path.exists(output_path + ".tmp"))
            with open(result_path, encoding="utf-8") as stream:
                original = json.load(stream)
            self.assertTrue(all(
                "aggregate" not in original["clips"][clip] for clip in clip_names))


if __name__ == "__main__":
    unittest.main()
