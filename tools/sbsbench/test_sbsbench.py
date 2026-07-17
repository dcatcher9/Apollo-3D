import io
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import argparse
import unittest
from unittest import mock
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "depth_models"))
import audit_depth_transform  # noqa: E402
import audit_depth_confidence  # noqa: E402
import generate_artistic_depth_run  # noqa: E402
import prepare_spring_artistic_training  # noqa: E402
import prepare_static_stereo_training  # noqa: E402
import prepare_sintel_artistic_training  # noqa: E402
import prepare_public_datasets  # noqa: E402
import prepare_flow_ema_reference  # noqa: E402
import run_eval  # noqa: E402
import rescore_run  # noqa: E402
import sbsbench  # noqa: E402


class EvalContractTests(unittest.TestCase):
    @staticmethod
    def png_bytes(value=64, mode="RGB"):
        shape = (8, 12, 3) if mode == "RGB" else (8, 12)
        array = np.full(shape, value, np.uint8)
        stream = io.BytesIO()
        Image.fromarray(array, mode=mode).save(stream, "PNG")
        return stream.getvalue()

    def test_metric_hash_is_independent_of_text_line_endings(self):
        paths = []
        try:
            for data in (b"alpha\nbeta\n", b"alpha\r\nbeta\r\n"):
                with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as fh:
                    fh.write(data)
                    paths.append(fh.name)
            # sha256_files includes the basename, so give both temp files the same logical name.
            with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
                left_path = os.path.join(left, "metric.py")
                right_path = os.path.join(right, "metric.py")
                with open(paths[0], "rb") as src, open(left_path, "wb") as dst:
                    dst.write(src.read())
                with open(paths[1], "rb") as src, open(right_path, "wb") as dst:
                    dst.write(src.read())
                self.assertEqual(run_eval.sha256_files([left_path]),
                                 run_eval.sha256_files([right_path]))
        finally:
            for path in paths:
                os.unlink(path)

    def test_hdr_source_kind_flags_are_mutually_exclusive_and_native_is_unscaled(self):
        self.assertEqual(
            run_eval.expected_hdr_source_kind([]),
            run_eval.harness_contract.HDR_SOURCE_SDR,
        )
        self.assertEqual(
            run_eval.expected_hdr_source_kind(["--simulate-hdr"]),
            run_eval.harness_contract.HDR_SOURCE_SIMULATED,
        )
        self.assertEqual(
            run_eval.expected_hdr_source_kind(["--native-hdr-scrgb"]),
            run_eval.harness_contract.HDR_SOURCE_NATIVE_PQ,
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            run_eval.expected_hdr_source_kind([
                "--simulate-hdr", "--native-hdr-scrgb",
            ])
        with self.assertRaisesRegex(ValueError, "cannot carry"):
            run_eval.expected_hdr_source_kind([
                "--native-hdr-scrgb", "--sdr-white-level-raw", "1000",
            ])

    def test_hdr_contract_provenance_separates_native_pq_simulated_and_sdr(self):
        cases = (
            ({
                "color_mode": "sdr-srgb-8bit",
                "hdr_source_kind": "native-sdr",
                "metric_preview_encoding": "native-srgb-v1",
                "hdr_input_scale": 0.0,
                "sdr_white_level_raw": 0,
            }, run_eval.harness_contract.HDR_SOURCE_SDR),
            ({
                "color_mode": "hdr-scrgb-fp16",
                "hdr_source_kind": "sdr-in-windows-hdr",
                "metric_preview_encoding":
                    "source-relative-srgb-from-scrgb-white-normalized-v1",
                "hdr_input_scale": 2.5,
                "sdr_white_level_raw": 2500,
            }, run_eval.harness_contract.HDR_SOURCE_SIMULATED),
            ({
                "color_mode": "hdr-scrgb-fp16",
                "hdr_source_kind": "native-pq-in-windows-hdr",
                "metric_preview_encoding":
                    "perceptual-srgb-from-native-scrgb-reinhard-v1",
                "hdr_input_scale": 0.0,
                "sdr_white_level_raw": 0,
            }, run_eval.harness_contract.HDR_SOURCE_NATIVE_PQ),
        )
        for payload, kind in cases:
            with self.subTest(kind=kind):
                encoding, scale, white = run_eval.validate_hdr_contract_provenance(
                    payload, kind, "clip"
                )
                self.assertEqual(encoding, payload["metric_preview_encoding"])
                self.assertEqual(scale, float(payload["hdr_input_scale"]))
                self.assertEqual(white, payload["sdr_white_level_raw"])

        stale_native = dict(cases[2][0], hdr_input_scale=1.0)
        with self.assertRaisesRegex(RuntimeError, "zero SDR-white"):
            run_eval.validate_hdr_contract_provenance(
                stale_native, run_eval.harness_contract.HDR_SOURCE_NATIVE_PQ,
                "clip",
            )
        wrong_kind = dict(cases[2][0], hdr_source_kind="sdr-in-windows-hdr")
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            run_eval.validate_hdr_contract_provenance(
                wrong_kind, run_eval.harness_contract.HDR_SOURCE_NATIVE_PQ,
                "clip",
            )

    def test_native_hdr_sidecar_authentication_binds_every_preview(self):
        with tempfile.TemporaryDirectory() as root:
            preview = os.path.join(root, "frame_00000.png")
            Path(preview).write_bytes(self.png_bytes())
            authentication = {
                "manifest": os.path.join(root, "frame_model_sources.json"),
                "manifest_sha256": "a" * 64,
                "content_sha256": "b" * 64,
                "width": 12,
                "height": 8,
                "frame_count": 1,
                "frames": {0: {"preview_path": Path(preview).resolve()}},
            }
            with mock.patch.object(
                    run_eval.native_hdr_capture, "validate_clip",
                    return_value=authentication) as validate:
                identity = run_eval.authenticate_native_hdr_clip(
                    root, {0: preview}, full=True
                )
            validate.assert_called_once_with(root, full=True)
            self.assertEqual(identity["content_sha256"], "b" * 64)

            incomplete = dict(authentication, frames={})
            with mock.patch.object(
                    run_eval.native_hdr_capture, "validate_clip",
                    return_value=incomplete):
                with self.assertRaisesRegex(RuntimeError, "exactly cover"):
                    run_eval.authenticate_native_hdr_clip(root, {0: preview})

    def test_clip_hash_covers_stereo_reference_and_requirement(self):
        with tempfile.TemporaryDirectory() as clip:
            gt_right = os.path.join(clip, "gt_right")
            os.makedirs(gt_right)
            Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(
                os.path.join(clip, "frame_00000.png"))
            reference_path = os.path.join(gt_right, "frame_00000.png")
            Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(reference_path)
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"required_gt_stereo": True}, fh)
            original = run_eval.sha1_dir(clip)
            Image.fromarray(np.full((8, 12, 3), 255, np.uint8)).save(reference_path)
            self.assertNotEqual(original, run_eval.sha1_dir(clip))
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"required_gt_stereo": False}, fh)
            changed_pixels = run_eval.sha1_dir(clip)
            with open(os.path.join(clip, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"required_gt_stereo": True}, fh)
            self.assertNotEqual(changed_pixels, run_eval.sha1_dir(clip))

    def test_clip_hash_manifest_is_used_after_cheap_verification(self):
        with tempfile.TemporaryDirectory() as root:
            clip = os.path.join(root, "shot")
            os.makedirs(clip)
            with open(os.path.join(clip, "frame_00000.png"), "wb") as stream:
                stream.write(b"source")
            manifest, _output = run_eval.clip_hashes.build_and_write(
                root, workers=1
            )
            with mock.patch.object(
                    run_eval, "sha1_dir", side_effect=AssertionError("direct hash")):
                identities, provenance = run_eval.resolve_clip_hashes(
                    root, ["shot"]
                )
            self.assertEqual(
                identities, {"shot": manifest["clips"]["shot"]["clip_sha1"]}
            )
            self.assertEqual(provenance["clip_hash_source"], "manifest")
            self.assertEqual(provenance["clip_hash_verification"], "stat")
            self.assertEqual(len(provenance["clip_hash_manifest_sha256"]), 64)

    def test_existing_clip_hash_manifest_fails_closed_when_stale(self):
        with tempfile.TemporaryDirectory() as root:
            clip = os.path.join(root, "shot")
            os.makedirs(clip)
            frame = os.path.join(clip, "frame_00000.png")
            with open(frame, "wb") as stream:
                stream.write(b"source")
            run_eval.clip_hashes.build_and_write(root, workers=1)
            with open(frame, "ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(
                    run_eval.clip_hashes.ClipHashManifestError, "size changed"):
                run_eval.resolve_clip_hashes(root, ["shot"])

    def test_clip_hashes_fall_back_to_direct_content_without_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            clip = os.path.join(root, "shot")
            os.makedirs(clip)
            with mock.patch.object(
                    run_eval, "sha1_dir", return_value="123456789abc") as direct:
                identities, provenance = run_eval.resolve_clip_hashes(
                    root, ["shot"]
                )
            direct.assert_called_once_with(clip)
            self.assertEqual(identities, {"shot": "123456789abc"})
            self.assertEqual(provenance["clip_hash_source"], "direct")
            self.assertEqual(
                provenance["clip_hash_verification"], "direct-content"
            )

    def test_direct_clip_hash_selection_rejects_duplicates_and_unsafe_names(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "shot"))
            with self.assertRaisesRegex(
                    run_eval.clip_hashes.ClipHashManifestError, "duplicates"):
                run_eval.resolve_clip_hashes(root, ["shot", "shot"])
            for name in ("../shot", "nested/shot", "nested\\shot", ".hidden", "NUL"):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(
                            run_eval.clip_hashes.ClipHashManifestError, "invalid clip name"):
                        run_eval.resolve_clip_hashes(root, [name])

    def test_run_label_and_clip_paths_are_single_contained_components(self):
        with tempfile.TemporaryDirectory() as root:
            expected = os.path.abspath(os.path.join(root, "eval-safe_1.0"))
            self.assertEqual(
                run_eval.contained_component(root, "eval-safe_1.0", "run label"),
                expected,
            )
            for value in ("", ".", "..", "../escape", "sub/escape", "sub\\escape",
                          "C:escape", "label with spaces", "trailing.", "COM1.log"):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "invalid run label"):
                        run_eval.contained_component(root, value, "run label")

    def test_direct_clip_sources_are_revalidated_after_scoring(self):
        with tempfile.TemporaryDirectory() as root:
            clip = os.path.join(root, "shot")
            os.makedirs(clip)
            frame = os.path.join(clip, "frame_00000.png")
            with open(frame, "wb") as stream:
                stream.write(b"source")
            identities, provenance = run_eval.resolve_clip_hashes(root, ["shot"])
            with open(frame, "ab") as stream:
                stream.write(b"-changed")
            with self.assertRaisesRegex(
                    run_eval.clip_hashes.ClipHashManifestError,
                    "source identities changed"):
                run_eval.revalidate_clip_hashes(
                    root, ["shot"], identities, provenance
                )

    def test_frozen_manifest_identity_is_revalidated_after_scoring(self):
        with tempfile.TemporaryDirectory() as root:
            clip = os.path.join(root, "shot")
            os.makedirs(clip)
            with open(os.path.join(clip, "frame_00000.png"), "wb") as stream:
                stream.write(b"source")
            _manifest, manifest_path = run_eval.clip_hashes.build_and_write(
                root, workers=1
            )
            identities, provenance = run_eval.resolve_clip_hashes(root, ["shot"])
            with open(manifest_path, "a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(
                    run_eval.clip_hashes.ClipHashManifestError,
                    "provenance changed"):
                run_eval.revalidate_clip_hashes(
                    root, ["shot"], identities, provenance
                )

    def test_eval_cleanup_cancels_workers_and_restores_parent_environment(self):
        original = run_eval.capture_score_worker_environment()
        first, second = run_eval.SCORE_WORKER_THREAD_ENV[:2]
        queue = mock.Mock()
        executor = mock.Mock()
        try:
            os.environ[first] = "parent-value"
            os.environ.pop(second, None)

            def abort(lifecycle):
                lifecycle["queue"] = queue
                lifecycle["executor"] = executor
                run_eval.configure_score_worker_threads()
                raise RuntimeError("stop")

            with mock.patch.object(run_eval, "_run_main", side_effect=abort):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    run_eval.main()
            self.assertEqual(os.environ[first], "parent-value")
            self.assertNotIn(second, os.environ)
            queue.cancel_pending.assert_called_once_with()
            executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        finally:
            run_eval.restore_score_worker_environment(original)

    def test_verify_clip_hashes_detects_same_stat_content_change(self):
        with tempfile.TemporaryDirectory() as root:
            clip = os.path.join(root, "shot")
            os.makedirs(clip)
            frame = os.path.join(clip, "frame_00000.png")
            with open(frame, "wb") as stream:
                stream.write(b"source")
            run_eval.clip_hashes.build_and_write(root, workers=1)
            previous = os.stat(frame)
            with open(frame, "wb") as stream:
                stream.write(b"tamper")
            os.utime(frame, ns=(previous.st_atime_ns, previous.st_mtime_ns))

            run_eval.resolve_clip_hashes(root, ["shot"], False)
            with self.assertRaisesRegex(
                    run_eval.clip_hashes.ClipHashManifestError,
                    "content hash changed"):
                run_eval.resolve_clip_hashes(root, ["shot"], True)

    def test_label_frame_manifest_is_strict_and_authenticated(self):
        with tempfile.TemporaryDirectory() as clip:
            path = os.path.join(clip, "label_frames.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump({"schema": 1, "frame_ids": [0, 3, 9]}, stream)
            frame_ids, identity = run_eval.load_label_frame_manifest(
                clip, list(range(10))
            )
            self.assertEqual(frame_ids, [0, 3, 9])
            self.assertEqual(identity, run_eval.sha256_file(path))

            selection = run_eval.resolve_output_selection(
                clip, list(range(10)), 1, False, True
            )
            self.assertEqual(selection, {
                "mode": "label-frames",
                "label_frame_ids": [0, 3, 9],
                "output_frame_ids": [0, 3, 9],
                "label_frames_sha256": identity,
            })

    def test_label_frame_manifest_rejects_ambiguous_or_missing_ids(self):
        invalid = (
            {"schema": 1, "frame_ids": []},
            {"schema": 1, "frame_ids": [0, 0]},
            {"schema": 1, "frame_ids": [2, 1]},
            {"schema": 1, "frame_ids": [False]},
            {"schema": 1, "frame_ids": [-1]},
            {"schema": 1, "frame_ids": [0], "unversioned": True},
        )
        with tempfile.TemporaryDirectory() as clip:
            path = os.path.join(clip, "label_frames.json")
            for payload in invalid:
                with self.subTest(payload=payload):
                    with open(path, "w", encoding="utf-8") as stream:
                        json.dump(payload, stream)
                    with self.assertRaises(ValueError):
                        run_eval.load_label_frame_manifest(clip, [0, 1, 2])
            with open(path, "w", encoding="utf-8") as stream:
                json.dump({"schema": 1, "frame_ids": [0, 4]}, stream)
            with self.assertRaisesRegex(ValueError, "missing source frames"):
                run_eval.load_label_frame_manifest(clip, [0, 1, 2])

    def test_optional_clip_metadata_is_absent_or_strict(self):
        with tempfile.TemporaryDirectory() as clip:
            self.assertEqual(run_eval.load_optional_clip_metadata(clip), {})
            meta_path = os.path.join(clip, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as stream:
                json.dump({"name": "shot", "expected_flat": False,
                           "ignored_future_field": 4}, stream)
            self.assertEqual(
                run_eval.load_optional_clip_metadata(clip),
                {"name": "shot", "expected_flat": False},
            )
            for invalid in ([], False, "not-an-object"):
                with self.subTest(invalid=invalid):
                    with open(meta_path, "w", encoding="utf-8") as stream:
                        json.dump(invalid, stream)
                    with self.assertRaisesRegex(ValueError, "root must be an object"):
                        run_eval.load_optional_clip_metadata(clip)
            with open(meta_path, "w", encoding="utf-8") as stream:
                stream.write("{")
            with self.assertRaisesRegex(ValueError, "invalid clip metadata"):
                run_eval.load_optional_clip_metadata(clip)
            with open(meta_path, "w", encoding="utf-8") as stream:
                json.dump({"expected_flat": "false"}, stream)
            with self.assertRaisesRegex(ValueError, "flags must be booleans"):
                run_eval.load_optional_clip_metadata(clip)

    def test_run_publication_invalidation_never_deletes_directories(self):
        with tempfile.TemporaryDirectory() as root:
            published = os.path.join(root, "results.json")
            with open(published, "w", encoding="utf-8") as stream:
                stream.write("old")
            run_eval.invalidate_publication_file(published, "result")
            self.assertFalse(os.path.lexists(published))
            os.makedirs(published)
            with self.assertRaisesRegex(ValueError, "is a directory"):
                run_eval.invalidate_publication_file(published, "result")
            self.assertTrue(os.path.isdir(published))

    def test_label_frame_selection_rejects_other_sampling_modes(self):
        with tempfile.TemporaryDirectory() as clip:
            with open(os.path.join(clip, "label_frames.json"), "w",
                      encoding="utf-8") as stream:
                json.dump({"schema": 1, "frame_ids": [0]}, stream)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                run_eval.resolve_output_selection(clip, [0], 1, True, True)
            with self.assertRaisesRegex(ValueError, "requires --output-every 1"):
                run_eval.resolve_output_selection(clip, [0], 2, False, True)
            self.assertEqual(
                run_eval.resolve_output_selection(clip, [0], 1, False, True)[
                    "output_frame_ids"
                ],
                [0],
            )

    def test_metric_contract_includes_runner_gating_semantics(self):
        metric_files = [os.path.join(run_eval.SCRIPT_DIR, "sbsbench.py"),
                        os.path.join(run_eval.SCRIPT_DIR, "thresholds.json"),
                        os.path.abspath(run_eval.__file__)]
        self.assertEqual(run_eval.metric_contract_sha(),
                         run_eval.sha256_files(metric_files))

    def test_missing_hard_metric_fails_closed(self):
        thresholds = {"metrics": {
            "negative_disparity_pct": {
                "role": "hard", "axis": "comfort", "better": "lower",
                "hard_max": 3.0,
            }
        }}
        _worst, _issues, hard = run_eval.score_clip_gates(
            [], {}, thresholds, {}
        )
        self.assertEqual(len(hard), 1)
        self.assertEqual(hard[0]["metric"], "negative_disparity_pct")
        self.assertEqual(hard[0]["reason"], "missing")

    def test_frame_gate_coverage_rejects_every_sparse_form(self):
        full = {
            "mode": "interval", "output_frame_ids": [4, 5, 6],
            "label_frame_ids": [], "label_frames_sha256": "",
        }
        self.assertEqual(
            run_eval.validate_full_frame_gate_coverage(
                [4, 5, 6], full,
                [{"_frame_id": 4}, {"_frame_id": 5}, {"_frame_id": 6}],
            ),
            [4, 5, 6],
        )
        with self.assertRaisesRegex(ValueError, "consecutive source"):
            run_eval.validate_full_frame_gate_coverage(
                [4, 6], {**full, "output_frame_ids": [4, 6]}
            )
        with self.assertRaisesRegex(ValueError, "sparse/GT-only"):
            run_eval.validate_full_frame_gate_coverage(
                [4, 5, 6], {**full, "mode": "label-frames"}
            )
        with self.assertRaisesRegex(ValueError, "every source frame"):
            run_eval.validate_full_frame_gate_coverage(
                [4, 5, 6], {**full, "output_frame_ids": [4, 6]}
            )
        with self.assertRaisesRegex(ValueError, "one ordered metric row"):
            run_eval.validate_full_frame_gate_coverage(
                [4, 5, 6], full,
                [{"_frame_id": 4}, {"_frame_id": 6}],
            )

    def test_selected_frame_gate_coverage_requires_exact_targets_only(self):
        source_ids = list(range(4, 11))
        selected = {
            "mode": "label-frames",
            "label_frame_ids": [5, 9],
            "output_frame_ids": [5, 9],
            "label_frames_sha256": "a" * 64,
        }
        rows = [{"_frame_id": frame_id} for frame_id in selected["output_frame_ids"]]
        self.assertEqual(
            run_eval.validate_selected_frame_gate_coverage(
                source_ids, selected, rows
            ),
            (source_ids, [5, 9], [5, 9]),
        )
        with self.assertRaisesRegex(ValueError, "authenticated label manifest"):
            run_eval.validate_selected_frame_gate_coverage(
                source_ids, {**selected, "label_frames_sha256": ""}
            )
        with self.assertRaisesRegex(ValueError, "targets only"):
            run_eval.validate_selected_frame_gate_coverage(
                source_ids, {**selected, "output_frame_ids": [5]}
            )
        with self.assertRaisesRegex(ValueError, "targets only"):
            run_eval.validate_selected_frame_gate_coverage(
                source_ids, {**selected, "output_frame_ids": [5, 9, 10]}
            )
        with self.assertRaisesRegex(ValueError, "one ordered metric row"):
            run_eval.validate_selected_frame_gate_coverage(
                source_ids, selected, rows[:-1]
            )
        self.assertEqual(
            run_eval.validate_selected_frame_gate_coverage(
                [4], {
                    "mode": "label-frames",
                    "label_frame_ids": [4],
                    "output_frame_ids": [4],
                    "label_frames_sha256": "b" * 64,
                },
            ),
            ([4], [4], [4]),
        )
        with self.assertRaisesRegex(ValueError, "sparse/GT-only"):
            run_eval.validate_full_frame_gate_coverage(source_ids, selected)

    @staticmethod
    def _frame_gate_thresholds():
        return {"metrics": {
            "exact_pop_spread_pct": {
                "role": "primary", "required_evidence": True,
            },
            "source_halo_p95": {
                "role": "primary", "required_evidence": True, "trigger": 5.0,
                "ordinal_hard_max": 8.0,
            },
            "flow_temporal_p95": {
                "role": "primary", "required_evidence": True, "min_frames": 2,
                "temporal_evidence": True,
            },
            "positive_disparity_pct": {
                "role": "hard", "hard_max": 3.0,
            },
            "diagnostic_only": {"role": "diagnostic", "trigger": 0.0},
        }}

    def test_frame_gate_metric_evidence_is_per_frame_and_fails_closed(self):
        thresholds = self._frame_gate_thresholds()
        first, violations = run_eval.frame_gate_metric_evidence({
            "exact_pop_spread_pct": 1.25,
            "source_halo_p95": 4.0,
            "positive_disparity_pct": 2.0,
        }, 0, thresholds)
        self.assertEqual(first["primary"]["flow_temporal_p95"], None)
        self.assertEqual(violations, [])

        _bounded, ordinal_violations = run_eval.frame_gate_metric_evidence({
            "exact_pop_spread_pct": 1.4,
            "source_halo_p95": 9.0,
            "flow_temporal": 2.0,
            "positive_disparity_pct": 2.0,
        }, 1, thresholds)
        self.assertEqual(
            [(item["metric"], item["kind"])
             for item in ordinal_violations],
            [("source_halo_p95", "trigger_max"),
             ("source_halo_p95", "ordinal_hard_max")],
        )

        second, violations = run_eval.frame_gate_metric_evidence({
            "exact_pop_spread_pct": 1.4,
            "source_halo_p95": 7.0,
            "flow_temporal": 2.0,
            "positive_disparity_pct": 3.5,
        }, 1, thresholds)
        self.assertEqual(second["primary"]["flow_temporal_p95"], 2.0)
        self.assertEqual(
            [(item["metric"], item["kind"]) for item in violations],
            [("source_halo_p95", "trigger_max"),
             ("positive_disparity_pct", "hard_max")],
        )
        with self.assertRaisesRegex(ValueError, "required per-frame hard metric"):
            run_eval.frame_gate_metric_evidence({
                "exact_pop_spread_pct": 1.0,
                "source_halo_p95": 1.0,
            }, 0, thresholds)
        missing_temporal, violations = run_eval.frame_gate_metric_evidence({
            "exact_pop_spread_pct": 1.0,
            "source_halo_p95": 1.0,
            "positive_disparity_pct": 1.0,
        }, 1, thresholds)
        self.assertIsNone(missing_temporal["primary"]["flow_temporal_p95"])
        self.assertEqual(violations, [])

    def test_target_only_gates_exclude_only_explicit_temporal_metrics(self):
        thresholds = self._frame_gate_thresholds()
        filtered = run_eval.target_only_gate_thresholds(thresholds)
        self.assertNotIn("flow_temporal_p95", filtered["metrics"])
        self.assertIn("exact_pop_spread_pct", filtered["metrics"])
        self.assertIn("source_halo_p95", filtered["metrics"])
        self.assertIn("positive_disparity_pct", filtered["metrics"])
        with self.assertRaisesRegex(ValueError, "temporal metric markers"):
            run_eval.target_only_gate_thresholds({
                "metrics": {"spatial": {"role": "primary"}}
            })

    def test_frame_gate_sidecar_binds_artifacts_results_and_detects_tampering(self):
        thresholds = self._frame_gate_thresholds()
        with tempfile.TemporaryDirectory() as root:
            artifacts = {}
            for frame_id in (4, 5):
                artifacts[frame_id] = {}
                for name in ("source", "sbs", "depth", "warp_mask", "warp_disparity"):
                    path = os.path.join(root, f"{name}_{frame_id}")
                    Path(path).write_bytes(f"{name}:{frame_id}".encode("ascii"))
                    artifacts[frame_id][name] = path
            context = {
                "source_frame_ids": [4, 5],
                "output_selection": {
                    "mode": "interval", "output_frame_ids": [4, 5],
                    "label_frame_ids": [], "label_frames_sha256": "",
                },
                "artifact_paths": artifacts,
                "clip_sha1": "a" * 12,
                "harness_contract_sha256": "b" * 64,
                "expected_flat": False,
                "geometry": {
                    "source_width": 32, "source_height": 16,
                    "model_input_width": 28, "model_input_height": 14,
                    "depth_short_side": 196, "depth_max_aspect": 4.0,
                    "eye_width": 32, "eye_height": 16,
                    "content_scale_x": 1.0, "content_scale_y": 1.0,
                    "disparity_raster_width": 32, "disparity_raster_height": 16,
                    "color_mode": "sdr-srgb-8bit",
                },
                "color": {"color_mode": "sdr-srgb-8bit"},
                "pipeline": {"artistic_scale_override": 1.25},
            }
            scene_payload = {
                "schema": 1,
                "contract": run_eval.runtime_scene_evidence.CONTRACT,
                "evidence_source": (
                    "SubjectState[0].y after completed depth postprocess"
                ),
                "cut_rule": (
                    "prior_scene_age_gte_7_and_current_scene_age_eq_0"
                ),
                "cadence": "completed-depth-frames-only",
                "completion_sequence_contract": (
                    "exact for this synchronous harness sequence; live "
                    "busy-drop cadence is not replayed"
                ),
                "depth_reuse_interval": 1,
                "source_frame_ids": [4, 5],
                "completed_source_frame_ids": [4, 5],
                "completed_depth_frame_count": 2,
                "frames": [
                    {
                        "source_frame_ordinal": 0,
                        "source_frame_id": 4,
                        "runtime_scene_id": 0,
                        "scene_age": 0.0,
                        "subject_initialized": True,
                        "hard_cut": False,
                        "scene_start": True,
                    },
                    {
                        "source_frame_ordinal": 1,
                        "source_frame_id": 5,
                        "runtime_scene_id": 0,
                        "scene_age": 1.0,
                        "subject_initialized": True,
                        "hard_cut": False,
                        "scene_start": False,
                    },
                ],
            }
            scene_path = os.path.join(root, "runtime_scene_evidence.json")
            Path(scene_path).write_text(
                json.dumps(scene_payload, sort_keys=True), encoding="utf-8"
            )
            context["runtime_scene_evidence"] = scene_payload
            context["runtime_scene_evidence_path"] = scene_path
            rows = [
                {"_frame_id": 4, "exact_pop_spread_pct": 1.0,
                 "source_halo_p95": 1.0, "positive_disparity_pct": 1.0},
                {"_frame_id": 5, "exact_pop_spread_pct": 1.2,
                 "source_halo_p95": 1.5, "flow_temporal": 0.5,
                 "positive_disparity_pct": 1.0},
            ]
            incomplete_geometry = dict(context)
            incomplete_geometry["geometry"] = dict(context["geometry"])
            del incomplete_geometry["geometry"]["model_input_width"]
            with self.assertRaisesRegex(ValueError, "invalid frame-gate artistic geometry"):
                run_eval.build_frame_gate_clip_records(
                    "shot", rows, thresholds, incomplete_geometry
                )
            clip_records = run_eval.build_frame_gate_clip_records(
                "shot", rows, thresholds, context
            )
            output = os.path.join(root, run_eval.FRAME_GATE_EVIDENCE_FILENAME)
            run_meta = {
                "metric_sha256": "c" * 16,
                "conf_sha256": "d" * 16,
                "clip_hash_manifest_sha256": "e" * 64,
                "clip_set_sha1": {"shot": "a" * 12},
                "run_name": "test",
                "suite": "core",
                "hdr_source_kind": "native-sdr",
            }
            identity = run_eval.write_frame_gate_evidence(
                output, run_meta, thresholds, [clip_records], "f" * 64
            )
            records = run_eval.validate_frame_gate_evidence(output)
            self.assertEqual(identity, run_eval.sha256_file(output))
            self.assertEqual(
                records[0]["contract"], run_eval.FRAME_GATE_EVIDENCE_CONTRACT
            )
            self.assertEqual(records[0]["results_sha256"], "f" * 64)
            clip_record = next(record for record in records if record["record"] == "clip")
            self.assertEqual(
                set(clip_record["geometry"]),
                set(run_eval.artistic_geometry_contract.GEOMETRY_KEYS),
            )
            self.assertEqual(
                clip_record["geometry_contract"],
                run_eval.artistic_geometry_contract.GEOMETRY_CONTRACT,
            )
            self.assertEqual(len(clip_record["geometry_sha256"]), 64)
            self.assertEqual(clip_record["runtime_scene_count"], 1)
            self.assertEqual(
                clip_record["runtime_scene_evidence_sha256"],
                run_eval.sha256_file(scene_path),
            )
            frames = [record for record in records if record["record"] == "frame"]
            self.assertEqual([record["frame_id"] for record in frames], [4, 5])
            self.assertTrue(frames[0]["runtime_scene"]["scene_start"])
            self.assertEqual(
                frames[0]["artifact_sha256"]["source"],
                run_eval.sha256_file(artifacts[4]["source"]),
            )

            records[1]["clip_sha1"] = "tampered"
            with open(output, "wb") as stream:
                for record in records:
                    stream.write(run_eval.canonical_json_bytes(record))
            with self.assertRaisesRegex(ValueError, "payload digest"):
                run_eval.validate_frame_gate_evidence(output)

    def test_selected_frame_gate_sidecar_preserves_full_source_ordinals(self):
        thresholds = self._frame_gate_thresholds()
        source_ids = list(range(4, 11))
        label_ids = [5, 9]
        output_ids = [5, 9]
        with tempfile.TemporaryDirectory() as root:
            manifest_path = os.path.join(root, "label_frames.json")
            Path(manifest_path).write_text(
                json.dumps({"schema": 1, "frame_ids": label_ids}), encoding="utf-8"
            )
            artifacts = {}
            for frame_id in output_ids:
                artifacts[frame_id] = {}
                for name in ("source", "sbs", "depth", "warp_mask", "warp_disparity"):
                    path = os.path.join(root, f"{name}_{frame_id}")
                    Path(path).write_bytes(f"{name}:{frame_id}".encode("ascii"))
                    artifacts[frame_id][name] = path

            scene_rows = []
            for ordinal, frame_id in enumerate(source_ids):
                scene_rows.append({
                    "source_frame_ordinal": ordinal,
                    "source_frame_id": frame_id,
                    "runtime_scene_id": 0,
                    "scene_age": float(ordinal),
                    "subject_initialized": True,
                    "hard_cut": False,
                    "scene_start": ordinal == 0,
                })
            scene_payload = {
                "schema": 1,
                "contract": run_eval.runtime_scene_evidence.CONTRACT,
                "evidence_source": "SubjectState[0].y after completed depth postprocess",
                "cut_rule": "prior_scene_age_gte_7_and_current_scene_age_eq_0",
                "cadence": "completed-depth-frames-only",
                "completion_sequence_contract": (
                    "exact for this synchronous harness sequence; live busy-drop cadence "
                    "is not replayed"
                ),
                "depth_reuse_interval": 1,
                "source_frame_ids": source_ids,
                "completed_source_frame_ids": source_ids,
                "completed_depth_frame_count": len(source_ids),
                "frames": scene_rows,
            }
            scene_path = os.path.join(root, "runtime_scene_evidence.json")
            Path(scene_path).write_text(
                json.dumps(scene_payload, sort_keys=True), encoding="utf-8"
            )
            context = {
                "source_frame_ids": source_ids,
                "output_selection": {
                    "mode": "label-frames",
                    "label_frame_ids": label_ids,
                    "output_frame_ids": output_ids,
                    "label_frames_sha256": run_eval.sha256_file(manifest_path),
                },
                "artifact_paths": artifacts,
                "runtime_scene_evidence": scene_payload,
                "runtime_scene_evidence_path": scene_path,
                "clip_sha1": "a" * 12,
                "harness_contract_sha256": "b" * 64,
                "expected_flat": False,
                "geometry": {
                    "source_width": 32, "source_height": 16,
                    "model_input_width": 28, "model_input_height": 14,
                    "depth_short_side": 196, "depth_max_aspect": 4.0,
                    "eye_width": 32, "eye_height": 16,
                    "content_scale_x": 1.0, "content_scale_y": 1.0,
                    "disparity_raster_width": 32, "disparity_raster_height": 16,
                    "color_mode": "sdr-srgb-8bit",
                },
                "color": {"color_mode": "sdr-srgb-8bit"},
                "pipeline": {"artistic_scale_override": 1.25},
            }
            rows = [{
                "_frame_id": frame_id,
                "exact_pop_spread_pct": 1.0,
                "source_halo_p95": 1.0,
                "positive_disparity_pct": 1.0,
                **({"flow_temporal": 0.5} if frame_id in label_ids else {}),
            } for frame_id in output_ids]
            clip_records = run_eval.build_frame_gate_clip_records(
                "shot", rows, thresholds, context
            )
            frame_records = [
                record for record in clip_records if record["record"] == "frame"
            ]
            self.assertEqual([record["frame_id"] for record in frame_records], output_ids)
            self.assertEqual([record["ordinal"] for record in frame_records], [1, 5])
            self.assertEqual(
                [record["runtime_scene"]["source_frame_ordinal"]
                 for record in frame_records],
                [1, 5],
            )

            output = os.path.join(root, run_eval.FRAME_GATE_EVIDENCE_FILENAME)
            run_meta = {
                "metric_sha256": "c" * 16,
                "conf_sha256": "d" * 16,
                "clip_hash_manifest_sha256": "e" * 64,
                "clip_set_sha1": {"shot": "a" * 12},
                "run_name": "test-selected",
                "suite": "core",
                "hdr_source_kind": "native-sdr",
            }
            with self.assertRaisesRegex(ValueError, "differs from publication header"):
                run_eval.write_frame_gate_evidence(
                    output, run_meta, thresholds, [clip_records], "f" * 64
                )
            run_eval.write_frame_gate_evidence(
                output, run_meta, thresholds, [clip_records], "f" * 64,
                evidence_contract=run_eval.SELECTED_FRAME_GATE_EVIDENCE_CONTRACT,
            )
            records = run_eval.validate_frame_gate_evidence(output)
            self.assertEqual(
                records[0]["contract"],
                run_eval.SELECTED_FRAME_GATE_EVIDENCE_CONTRACT,
            )
            clip_record = next(
                record for record in records if record["record"] == "clip"
            )
            self.assertEqual(clip_record["full_source_frame_count"], len(source_ids))
            self.assertEqual(clip_record["full_source_frame_ids"], source_ids)
            self.assertEqual(clip_record["label_frame_ids"], label_ids)
            self.assertEqual(clip_record["output_selected_frame_ids"], output_ids)
            self.assertEqual(
                clip_record["output_label_frames_sha256"],
                run_eval.sha256_file(manifest_path),
            )
            clip_record["output_selected_frame_ids"] = output_ids[:-1]
            payload_bytes = [
                run_eval.canonical_json_bytes(record) for record in records[:-1]
            ]
            records[-1]["payload_sha256"] = hashlib.sha256(
                b"".join(payload_bytes)
            ).hexdigest()
            with open(output, "wb") as stream:
                for record in records:
                    stream.write(run_eval.canonical_json_bytes(record))
            with self.assertRaisesRegex(ValueError, "targets only"):
                run_eval.validate_frame_gate_evidence(output)

    def test_baseline_context_covers_sampling_and_artistic_controls(self):
        self.assertEqual(run_eval.EVAL_SCHEMA, 31)
        for field in (
                "output_interval", "output_gt_right_only", "artistic_policy",
                "artistic_style", "artistic_scale_override", "depth_override_frames",
                "policy_warp_source_sha256"):
            self.assertIn(field, run_eval.BASELINE_CONTEXT_FIELDS)
        for field in ("model", "conf_sha256", "artistic_policy_consumed",
                      "artistic_policy_authorization", "model_onnx_sha256"):
            self.assertIn(field, run_eval.POLICY_CANDIDATE_TREATMENT_FIELDS)
        self.assertEqual(run_eval.LABEL_SELECTION_CONTEXT_FIELDS, (
            "output_selection_mode", "label_frame_ids", "output_selected_frame_ids",
            "output_label_frames_sha256",
        ))

    def test_bounded_score_queue_preserves_order_and_outstanding_limit(self):
        executor = mock.Mock()

        def submit(function, seq_dir, frames_dir, expected_flat):
            self.assertIs(function, run_eval.score_clip_artifacts)
            future = mock.Mock()
            future.result.return_value = ([{"seq": seq_dir}], {"seq": seq_dir})
            return future

        executor.submit.side_effect = submit
        queue = run_eval.BoundedOrderedScoreQueue(executor, max_outstanding=2)

        self.assertEqual(queue.submit("a", {"ordinal": 0}, "seq-a", "frames", False), [])
        self.assertEqual(queue.submit("b", {"ordinal": 1}, "seq-b", "frames", False), [])
        self.assertEqual(queue.outstanding, 2)
        completed = queue.submit("c", {"ordinal": 2}, "seq-c", "frames", True)
        self.assertEqual(queue.outstanding, 2)
        self.assertEqual(completed[0][0], "a")
        self.assertEqual([item[0] for item in queue.drain()], ["b", "c"])

    def test_bounded_score_queue_surfaces_worker_exception_with_clip(self):
        executor = mock.Mock()
        failed = mock.Mock()
        failed.result.side_effect = ValueError("invalid evidence")
        executor.submit.return_value = failed
        queue = run_eval.BoundedOrderedScoreQueue(executor, max_outstanding=1)
        queue.submit("bad_clip", {}, "seq", "frames", False)

        with self.assertRaisesRegex(
                run_eval.ScoreWorkerError, "bad_clip.*ValueError.*invalid evidence"):
            queue.submit("next_clip", {}, "next", "frames", False)

    def test_score_worker_environment_disables_nested_thread_pools(self):
        overridden = {name: "12" for name in run_eval.SCORE_WORKER_THREAD_ENV}
        with mock.patch.dict(os.environ, overridden):
            run_eval.configure_score_worker_threads()
            self.assertTrue(all(os.environ[name] == "1"
                                for name in run_eval.SCORE_WORKER_THREAD_ENV))

    def test_score_worker_process_matches_direct_measurement(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            source = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
            Image.fromarray(source).save(os.path.join(frames, "frame_00000.png"))
            Image.fromarray(np.concatenate((source, source), axis=1)).save(
                os.path.join(seq, "sbs_00000.png")
            )
            expected = sbsbench.measure_sequence(seq, frames)
            with mock.patch.dict(os.environ, {}):
                run_eval.configure_score_worker_threads()
                with run_eval.ProcessPoolExecutor(
                        max_workers=1,
                        initializer=run_eval._initialize_score_worker) as executor:
                    actual = executor.submit(
                        run_eval.score_clip_artifacts, seq, frames, False
                    ).result(timeout=30)
            self.assertEqual(actual, expected)

    def test_exact_disparity_requires_full_output_eye_raster_shape(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "sbs.png")
            # Two 8x6 eyes; source has the same aspect, so the exact field must be 8x6.
            Image.fromarray(np.full((6, 16, 3), 80, np.uint8)).save(path)
            source = np.full((6, 8), 0.5, np.float32)
            with self.assertRaisesRegex(ValueError, "raster shape mismatch"):
                sbsbench.measure_seq_frame(
                    path, src_gray=source,
                    warp_disparity=np.zeros((3, 4), np.float32),
                )

    def test_source_integrity_excludes_letterbox_bars(self):
        rng = np.random.default_rng(27)
        source = rng.random((4, 8), dtype=np.float32)
        eye = np.zeros((8, 8), dtype=np.float32)
        eye[2:6] = source
        metrics = sbsbench.source_relative_metrics(eye, source, max_shift=1)
        self.assertGreater(metrics["source_coverage_pct"], 99.0)
        self.assertGreater(metrics["image_integrity_pct"], 99.0)

    def test_warp_mask_keeps_full_eye_alignment_with_fractional_bars(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "sbs.png")
            source = np.linspace(0.0, 1.0, 5 * 9, dtype=np.float32).reshape(5, 9)
            content = np.asarray(
                Image.fromarray(source, mode="F").resize((8, 4), Image.BILINEAR),
                dtype=np.float32,
            )
            eye = np.zeros((8, 8), dtype=np.float32)
            eye[2:6] = content
            packed = np.concatenate((eye, eye), axis=1)
            Image.fromarray(np.uint8(np.clip(packed, 0.0, 1.0) * 255.0)).save(path)
            row, _, _ = sbsbench.measure_seq_frame(
                path,
                src_gray=source,
                warp_mask=np.zeros((8, 16, 3), dtype=np.float32),
                warp_disparity=np.zeros((8, 8), dtype=np.float32),
            )
        self.assertEqual(row["warp_hole_pct"], 0.0)

    def test_content_mask_matches_float32_hlsl_boundary(self):
        source = np.zeros((192, 100), dtype=np.float32)
        mask = sbsbench.source_content_pixel_mask(source, 1920, 1200)
        columns = np.flatnonzero(np.any(mask, axis=0))
        self.assertEqual((columns[0], columns[-1], columns.size), (647, 1271, 625))

    def test_named_profiles_and_explicit_overrides_share_production_precedence(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("sbs_3d_profile = cinema\n")
            path = fh.name
        try:
            self.assertEqual(run_eval.expected_profile(path, []), "cinema")
        finally:
            os.unlink(path)

    def test_apollo_is_the_unconfigured_default_profile(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            path = fh.name
        try:
            self.assertEqual(run_eval.expected_profile(path, []), "apollo")
        finally:
            os.unlink(path)

    def test_committed_gate_tracks_the_production_default_profile(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        bench_conf = os.path.join(repo, "tools", "sbsbench", "bench.conf")
        self.assertEqual(run_eval.expected_profile(bench_conf, []), "apollo")
        with open(os.path.join(repo, "src", "config.h"), encoding="utf-8") as fh:
            self.assertIn('std::string profile = "apollo"', fh.read())

    def test_baseline_update_refuses_gpu_contention(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            evaluator = fh.read()
        self.assertIn("if args.update_baselines:", evaluator)
        self.assertIn("refusing --update-baselines while another sunshine.exe is running",
                      evaluator)

    def test_custom_profile_values_need_no_evaluator_code_change(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("sbs_3d_profile = Cinema\n"
                     "sbs_3d_profile_Cinema_depth_model = depth_anything_v2_base_fp16\n")
            path = fh.name
        try:
            self.assertEqual(run_eval.expected_profile(path, []), "Cinema")
            self.assertEqual(run_eval.expected_depth_model(path, "Cinema", []),
                             "depth_anything_v2_base_fp16")
            self.assertEqual(
                run_eval.expected_depth_model(
                    path, "Cinema", ["--model", "depth_anything_v2_fp8"]),
                "depth_anything_v2_fp8")
        finally:
            os.unlink(path)

    def test_live_sbs_contract_is_off_ai_with_startup_profile(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "video.h"), encoding="utf-8") as fh:
            video_header = fh.read()
        self.assertIn("SBS_AI = 1", video_header)
        self.assertNotIn("SBS_GAME", video_header)
        self.assertNotIn("SBS_MOVIE", video_header)

        with open(os.path.join(repo, "src", "config.cpp"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertIn('apply_sbs_values(video.sbs, "sbs_3d_profile_" + sbs_profile + "_")', config)
        self.assertNotIn("video.sbs_profiles", config)
        self.assertIn('if (sbs_profile == "vd3d")', config)

        with open(os.path.join(repo, "src", "stream.cpp"), encoding="utf-8") as fh:
            stream = fh.read()
        self.assertNotIn("IDX_SET_SBS_PROFILE", stream)
        self.assertNotIn("IDX_SBS_PROFILE_LIST", stream)
        self.assertIn("mail::sbs_depth_status", stream)
        self.assertNotIn("depth_engine_phase", stream)
        self.assertNotIn("set_active_depth_model(id)", stream)

        with open(os.path.join(repo, "src", "main.cpp"), encoding="utf-8") as fh:
            main = fh.read()
        self.assertIn("prepare_tensorrt_model", main)
        self.assertIn("std::jthread model_prepare_thread", main)
        self.assertLess(main.index("if (!config::sunshine.cmd.name.empty())"),
                        main.index("std::jthread model_prepare_thread"))

        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn("cuda_device_for_configured_adapter", estimator)
        self.assertIn("if (!warmup_execution_context", estimator)
        self.assertIn("const bool synchronized = enqueued", estimator)

    def test_relative_cli_paths_are_resolved_before_subprocess_cwd(self):
        args = argparse.Namespace(build_dir="cmake-build-relwithdebinfo", conf="bench.conf",
                                  clips_root=None, baseline_dir=None,
                                  report_control=None, report_out=None)
        run_eval.normalize_cli_paths(args)
        self.assertTrue(os.path.isabs(args.build_dir))
        self.assertTrue(os.path.isabs(args.conf))

    def test_eval_builds_production_binary_and_fails_closed_on_build_error(self):
        current = mock.Mock(returncode=0, stdout="ninja: no work to do.\n", stderr="")
        with mock.patch.object(run_eval.shutil, "which", return_value="ninja"), \
                mock.patch.object(run_eval.subprocess, "run", return_value=current) as run:
            run_eval.require_current_build("build")
        self.assertEqual(run.call_args.args[0], ["ninja", "-C", "build", "sunshine"])
        failed = mock.Mock(returncode=1, stdout="compile failed\n", stderr="")
        with mock.patch.object(run_eval.shutil, "which", return_value="ninja"), \
                mock.patch.object(run_eval.subprocess, "run", return_value=failed), \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                run_eval.require_current_build("build")

    def test_apollo_bestv2_normalizes_pixel_shifts_by_source_geometry(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "sbs_reprojection_ps.hlsl")
        with open(shader, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("LeftColorTexture.GetDimensions(sourceWidth, sourceHeight)", text)
        self.assertIn(
            "Bestv2SearchRadius((float)sourceWidth, (float)sourceHeight, s2)", text
        )
        self.assertIn("s0, s1, s2, (float)sourceWidth, (float)sourceHeight", text)
        self.assertEqual(text.count("DepthParallax("), 2)
        self.assertNotIn("Bestv2SearchRadius((float)dw)", text)

    def test_forward_coverage_diagnostic_uses_source_geometry(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx")
        with open(os.path.join(shader_dir, "sbs_forward_coverage_cs.hlsl"),
                  encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("LeftColorTexture.GetDimensions(source_w, source_h)", text)
        self.assertIn("s0, s1, s2, (float)source_w, (float)source_h", text)
        self.assertNotIn("s0, s1, (float)eye_w", text)

    def test_bestv2_scales_wide_sources_from_validated_calibration_width(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        common = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "include", "sbs_warp_common.hlsl")
        with open(common, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("BESTV2_CALIBRATION_WIDTH = 854.0f", text)
        self.assertIn("return min(max(source_width, 1.0f), BESTV2_CALIBRATION_WIDTH)", text)
        self.assertGreaterEqual(text.count("/ parallax_width"), 2)

    def test_bestv2_preserves_angular_pop_across_source_aspects(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        common = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "include", "sbs_warp_common.hlsl")
        with open(common, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("BESTV2_REFERENCE_ASPECT = 5120.0f / 2160.0f", text)
        self.assertIn("BESTV2_REFERENCE_ASPECT / aspect", text)
        self.assertNotIn("fixed_height", text)
        reference_aspect = 5120.0 / 2160.0
        self.assertAlmostEqual(reference_aspect / (5120.0 / 2160.0), 1.0)
        self.assertAlmostEqual(reference_aspect / (3840.0 / 2160.0), 4.0 / 3.0)
        self.assertAlmostEqual(reference_aspect / (3552.0 / 3840.0), 2.562562563, places=6)

    def test_pop_strength_scales_shared_parallax_and_apollo_probe_radius(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        common = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "include", "sbs_warp_common.hlsl")
        with open(common, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("float pop_strength;", text)
        self.assertIn("pop_strength * adaptive_ratio", text)
        self.assertIn("return parallax * p.output_scale;", text)
        self.assertIn("return clamp(DepthParallaxUnclamped", text)
        self.assertIn("p.clamp_abs = 0.071f * aspect_scale;", text)

        with open(os.path.join(repo, "src", "config.cpp"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertIn('prefix + "pop_strength", target.pop_strength, {0.25, 2.0}', config)
        with open(os.path.join(repo, "src", "config.h"), encoding="utf-8") as fh:
            config_header = fh.read()
        self.assertIn("double pop_strength = 1.25;", config_header)
        self.assertIn("bool adaptive_pop = true;", config_header)
        self.assertIn("double adaptive_pop_max = 1.30;", config_header)
        self.assertIn('std::string zero_plane = "legacy";', config_header)
        self.assertIn('std::string artistic_style = "immersive";', config_header)

        with open(os.path.join(repo, "src", "platform", "windows", "display_vram.cpp"),
                  encoding="utf-8") as fh:
            production = fh.read()
        self.assertIn("(float) sbs_config.pop_strength", production)
        self.assertIn("sbs_config.adaptive_pop ? 1.0f : 0.0f", production)

        with open(os.path.join(repo, "src_assets", "windows", "assets", "shaders",
                               "directx", "depth_subject_resolve_cs.hlsl"),
                  encoding="utf-8") as fh:
            adaptive = fh.read()
        self.assertIn("change_fraction >= 0.65f", adaptive)
        self.assertIn("scene_age >= 8.0f", adaptive)
        self.assertIn("smoothstep(0.007f, 0.016f, edge_fraction)", adaptive)
        self.assertNotIn("lerp(pop_ratio, target_ratio", adaptive)
        self.assertIn("Bestv2RawShiftPxFast(zero_anchor_shaped)", adaptive)
        self.assertIn("s2 = float4(zero_anchor_shift, zero_valid, zero_plane_mode", adaptive)
        self.assertIn("float safe_ceiling = ArtisticGlobal[0]", adaptive)
        self.assertIn("lerp(1.0f, safe_ceiling, style_mix)", adaptive)
        self.assertIn("ceiling_confidence >= 0.5f", adaptive)
        self.assertNotIn(
            "lerp(1.0f, predicted_scale, predicted_confidence)", adaptive
        )
        self.assertIn("artistic_policy > 0.5f", adaptive)

        self.assertNotIn("0.015f", text)
        self.assertIn("pop_strength * adaptive_ratio * artistic_ratio", text)

    def test_adaptive_pop_last_flag_wins(self):
        conf = os.path.join(os.path.dirname(__file__), "bench.conf")
        self.assertFalse(run_eval.expected_adaptive_pop(
            conf, "apollo", ["--adaptive-pop", "--no-adaptive-pop"]))
        self.assertTrue(run_eval.expected_adaptive_pop(
            conf, "apollo", ["--no-adaptive-pop", "--adaptive-pop"]))

    def test_literal_bestv2_is_harness_only_and_machine_verified(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                               "include", "sbs_warp_common.hlsl"), encoding="utf-8") as fh:
            shader = fh.read()
        self.assertIn("float literal_bestv2;", shader)
        self.assertIn("literal_mode > 0.5f", shader)

        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"), encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn('a == "--literal-bestv2"', harness)
        self.assertIn('target_directory / "contract.json"', harness)
        self.assertIn('"apollo-harness-artistic-multiscale-v5"', harness)
        self.assertIn('"  \\"schema\\": 28,\\n"', harness)
        self.assertIn('\\"metric_preview_encoding\\"', harness)
        self.assertIn('a == "--no-artistic-policy"', harness)
        self.assertIn('a == "--artistic-scale-override"', harness)
        self.assertIn('a == "--output-every"', harness)
        self.assertIn('a == "--output-gt-right-only"', harness)
        self.assertIn('a == "--output-label-frames"', harness)
        self.assertIn('fs::path(o.frames) / "label_frames.json"', harness)
        self.assertIn('\\"label_frame_ids\\"', harness)
        self.assertIn('\\"output_selected_frame_ids\\"', harness)
        self.assertIn('\\"output_label_frames_sha256\\"', harness)
        multiscale_start = harness.index("if (!o.artistic_scale_grid.empty())")
        multiscale_gate = harness[
            multiscale_start:
            harness.index("if (o.literal_bestv2", multiscale_start)
        ]
        self.assertNotIn(
            "o.output_label_frames || !o.depth_override_root.empty()",
            multiscale_gate,
        )
        self.assertIn("--output-label-frames may select emitted artifacts", multiscale_gate)
        self.assertIn('{"schema", 5}', harness)
        self.assertIn('{"label_frame_ids", label_frame_ids}', harness)
        self.assertIn(
            '{"output_label_frames_sha256", output_label_frames_sha256}',
            harness,
        )
        self.assertIn("artistic_policy_consumed", harness)
        self.assertIn("model_onnx_sha256", harness)
        self.assertIn("policy_metadata_sha256", harness)
        self.assertNotIn("std::string frame_id(", harness)
        self.assertIn(
            "const std::string &output_id = source_frame_suffixes[fi]", harness
        )
        self.assertIn(
            "if (!emit_frame && (o.depth_only || sparse_output_selection))",
            harness,
        )
        self.assertIn("--depth-override-root cannot be combined with ", harness)
        hoist = harness.index(
            "// The completed depth and its silhouette prefilter are invariant"
        )
        prefilter = harness.index(
            "ID3D11ShaderResourceView *warp_depth", hoist
        )
        render_loop = harness.index(
            "const size_t render_scale_count", prefilter
        )
        composite = harness.index(
            "// Composite (mirrors display_vram::convert()", render_loop
        )
        diagnostics = harness.index(
            "// Export the exact full-binocular disparity", composite
        )
        self.assertIn("if (est.depth)", harness[prefilter:render_loop])
        self.assertNotIn(
            "ctx->CSSetShader(warp_prefilter_cs", harness[render_loop:composite]
        )
        self.assertIn("if (emit_frame && est.depth)", harness[diagnostics:])
        raw_diagnostic = harness.index(
            "// Also preserve the unclamped, scale-1 baseline", diagnostics
        )
        coverage = harness.index("auto dispatch_coverage", raw_diagnostic)
        self.assertIn(
            "if (!multiscale || render_scale_index == 0)",
            harness[raw_diagnostic:coverage],
        )
        self.assertIn(
            "fs::create_hard_link", harness[raw_diagnostic:coverage]
        )
        perf_tick = harness.index("sbs_perf::tick();", diagnostics)
        self.assertIn("if (emit_frame)", harness[perf_tick:])
        writer_start = harness.index("class bounded_multiscale_artifact_writer")
        writer_end = harness.index("bool save_gray16_png", writer_start)
        writer = harness[writer_start:writer_end]
        self.assertIn("CoInitializeEx(nullptr, COINIT_MULTITHREADED)", writer)
        self.assertIn("CLSID_WICImagingFactory", writer)
        self.assertIn("jobs_.size() < queue_capacity_", writer)
        self.assertIn("native_hdr_metric_bgra", writer)
        self.assertIn("simulated_hdr_metric_bgra", writer)
        self.assertNotIn("g_wic", writer)
        drain = harness.index(
            "multiscale_artifact_writer_result = multiscale_artifact_writer->finish()"
        )
        perf_publication = harness.index("sbs_perf::dump_json", drain)
        contract_publication = harness.index(
            "const nlohmann::json batch_contract", drain
        )
        self.assertLess(drain, perf_publication)
        self.assertLess(drain, contract_publication)
        self.assertIn('"apollo-bounded-multiscale-png-writer-v1"', harness)
        self.assertIn('{"drained_before_publication", true}', harness)
        self.assertIn('\\"artistic_policy\\"', harness)
        self.assertIn('\\"depth_override_frames\\"', harness)
        self.assertIn('\\"zero_plane\\"', harness)
        self.assertIn('\\"artistic_style\\"', harness)

        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            evaluator = fh.read()
        self.assertIn('contract_path = os.path.join(out_dir, "contract.json")', evaluator)
        self.assertIn("authenticate_native_hdr_clip(", evaluator)
        self.assertIn('"hdr_source_kind": expected_hdr_kind', evaluator)
        self.assertIn('"hdr_source_kind": contract["hdr_source_kind"]', evaluator)
        self.assertNotIn("profile ([a-z0-9_-]+)", evaluator)

        with open(os.path.join(repo, "src", "stream.cpp"), encoding="utf-8") as fh:
            stream = fh.read()
        self.assertNotIn("SBS_PRESENTATION_FIXED_HEIGHT", stream)

    def test_depth_reuse_cadence_is_explicit_and_machine_verified(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"), encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn('a == "--depth-every"', harness)
        self.assertIn('a == "--depth-override-root"', harness)
        self.assertIn("depth_compensation", harness)
        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            evaluator = fh.read()
        self.assertIn('"depth_compensation": depth_compensation', evaluator)
        self.assertIn("depth_reuse_interval", harness)
        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            evaluator = fh.read()
        self.assertIn('extra_value(args.extra, "--depth-every", 1)', evaluator)
        self.assertIn('f"reuse-{depth_reuse_interval}"', evaluator)
        self.assertIn("HARNESS_SCHEMA = harness_contract.HARNESS_SCHEMA", evaluator)
        self.assertIn('extra_value(args.extra, "--output-every", 1)', evaluator)
        self.assertIn("resolve_output_selection(", evaluator)
        self.assertIn('"label-frames" if output_label_frames', evaluator)
        self.assertIn('depth_override_root and not args.comparison_only', evaluator)
        self.assertIn('"--score-workers"', evaluator)
        self.assertIn("ProcessPoolExecutor(", evaluator)
        self.assertIn("2 * args.score_workers", evaluator)
        self.assertIn("env=harness_environment", evaluator)

    def test_zero_plane_modes_are_shot_latched_and_machine_verified(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src_assets", "windows", "assets", "shaders",
                               "directx", "include", "sbs_warp_common.hlsl"),
                  encoding="utf-8") as fh:
            common = fh.read()
        self.assertIn("p.explicit_zero_plane > 0.5f ? p.zero_anchor_shift_px", common)
        self.assertIn("p.explicit_zero_plane > 0.5f ? 0.0f", common)
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn('a == "--zero-plane"', harness)
        self.assertIn('o.zero_plane != "background"', harness)
        conf = os.path.join(os.path.dirname(__file__), "bench.conf")
        self.assertEqual(run_eval.expected_profile_string(
            conf, "apollo", "zero_plane", "legacy", ["--zero-plane", "median"],
            "--zero-plane"), "median")

    def test_live_depth_pairing_is_bounded_and_sync_is_evaluation_only(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "config.cpp"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertNotIn('"depth_frame_mode"', config)
        self.assertNotIn('"depth_fps"', config)

        with open(os.path.join(repo, "src", "platform", "windows", "display_vram.cpp"),
                  encoding="utf-8") as fh:
            production = fh.read()
        self.assertIn("std::array<matched_frame_slot_t, 2>", production)
        self.assertIn("repeat_matched_output", production)
        self.assertNotIn("finish_pending_depth", production)
        self.assertNotIn("depth_frame_mode", production)

        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn("finish_pending_depth_for_evaluation", harness)

    def test_cuda_graph_replay_is_signature_safe_and_falls_back(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "config.h"), encoding="utf-8") as fh:
            config = fh.read()
        self.assertIn("bool cuda_graph = true;", config)
        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn("input != graph_input || output != graph_output", estimator)
        self.assertIn("artistic_output != graph_artistic_output", estimator)
        self.assertIn("target_w != graph_width || target_h != graph_height", estimator)
        self.assertIn("if (!graph_signature_warmed)", estimator)
        self.assertIn("destroy_inference_graph(cuda);", estimator)
        self.assertIn("return exec_context->enqueueV3(cu_stream);", estimator)
        with open(os.path.join(repo, "src", "cuda_driver_api.h"), encoding="utf-8") as fh:
            driver = fh.read()
        for symbol in ("cuStreamBeginCapture", "cuStreamEndCapture",
                       "cuGraphInstantiateWithFlags", "cuGraphLaunch",
                       "cuGraphExecDestroy"):
            self.assertIn(symbol, driver)

    def test_cuda_graph_eval_override_matches_profile_precedence(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("sbs_3d_profile = cinema\n"
                     "sbs_3d_profile_cinema_cuda_graph = false\n")
            path = fh.name
        try:
            self.assertFalse(run_eval.expected_profile_bool(
                path, "cinema", "cuda_graph", True, [], "--cuda-graph"))
            self.assertTrue(run_eval.expected_profile_bool(
                path, "cinema", "cuda_graph", True,
                ["--cuda-graph", "on"], "--cuda-graph"))
        finally:
            os.unlink(path)

    def test_edge_selective_ema_uses_immutable_history_and_exports_locality_mask(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders",
                                  "directx")
        with open(os.path.join(shader_dir, "depth_ema_motion_cs.hlsl"),
                  encoding="utf-8") as fh:
            mask_shader = fh.read()
        self.assertIn("PreviousDepth", mask_shader)
        self.assertIn("ema_edge_change", mask_shader)
        self.assertNotIn("ema_edge_dilation", mask_shader)
        self.assertIn("MotionMask[DTid.xy] = IsMovingEdge", mask_shader)
        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn("CopyResource(depth_previous_tex.Get(), depth_tex.Get())", estimator)
        self.assertIn("ema_motion_mask_srv", estimator)
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn('"ema_mask_%s.png"', harness)

    def test_depth_override_manifest_requires_exact_frames_and_source_hash(self):
        with tempfile.TemporaryDirectory() as root:
            clips_root = os.path.join(root, "clips")
            clip_dir = os.path.join(clips_root, "sample")
            override_root = os.path.join(root, "override")
            override_clip = os.path.join(override_root, "sample")
            os.makedirs(clip_dir)
            os.makedirs(override_clip)
            for frame_id in range(3):
                Image.fromarray(np.full((8, 12, 3), frame_id, np.uint8)).save(
                    os.path.join(clip_dir, f"frame_{frame_id:05d}.png"))
            Image.fromarray(np.full((4, 6), 32768, np.uint16)).save(
                os.path.join(override_clip, "depth_00001.png"))
            manifest = {
                "schema": 3,
                "method": "classical-tile-phase-flow",
                "frame_policy": "held",
                "depth_every": 2,
                "clips": {"sample": {
                    "override_frames": 1,
                    "override_frame_ids": [1],
                    "clip_sha1": run_eval.sha1_dir(clip_dir),
                }},
            }
            with open(os.path.join(override_root, "manifest.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(manifest, fh)
            self.assertEqual(run_eval.validate_depth_override_manifest(
                override_root, clips_root, ["sample"], 2), {"sample": 1})
            os.remove(os.path.join(override_clip, "depth_00001.png"))
            original_stderr = sys.stderr
            try:
                sys.stderr = io.StringIO()
                with self.assertRaises(SystemExit):
                    run_eval.validate_depth_override_manifest(
                        override_root, clips_root, ["sample"], 2)
            finally:
                sys.stderr = original_stderr

    def test_all_frame_depth_treatment_manifest_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            clips_root = os.path.join(root, "clips")
            clip_dir = os.path.join(clips_root, "sample")
            override_root = os.path.join(root, "override")
            override_clip = os.path.join(override_root, "sample")
            os.makedirs(clip_dir)
            os.makedirs(override_clip)
            frame_ids = list(range(3))
            for frame_id in frame_ids:
                Image.fromarray(np.full((8, 12, 3), frame_id, np.uint8)).save(
                    os.path.join(clip_dir, f"frame_{frame_id:05d}.png"))
                Image.fromarray(np.full((4, 6), 32768, np.uint16)).save(
                    os.path.join(override_clip, f"depth_{frame_id:05d}.png"))
            manifest = {
                "schema": 3,
                "method": "flow-aware-ema-oracle",
                "frame_policy": "all",
                "depth_every": 1,
                "clips": {"sample": {
                    "override_frames": 3,
                    "override_frame_ids": frame_ids,
                    "clip_sha1": run_eval.sha1_dir(clip_dir),
                }},
            }
            with open(os.path.join(override_root, "manifest.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(manifest, fh)
            self.assertEqual(run_eval.validate_depth_override_manifest(
                override_root, clips_root, ["sample"], 1, True), {"sample": 3})
            os.remove(os.path.join(override_clip, "depth_00002.png"))
            original_stderr = sys.stderr
            try:
                sys.stderr = io.StringIO()
                with self.assertRaises(SystemExit):
                    run_eval.validate_depth_override_manifest(
                        override_root, clips_root, ["sample"], 1, True)
            finally:
                sys.stderr = original_stderr

    def test_rescore_derives_depth_compensation_for_schema_upgrade(self):
        self.assertEqual(rescore_run.depth_compensation_from_meta({}), "none")
        self.assertEqual(rescore_run.depth_compensation_from_meta(
            {"extra_args": ["--depth-override-root", "reference"]}),
            "external-reference")
        self.assertEqual(rescore_run.depth_compensation_from_meta(
            {"depth_compensation": "nvof-1x1"}), "nvof-1x1")
        self.assertEqual(rescore_run.depth_compensation_from_meta(
            {"extra_args": ["--depth-override-root", "reference",
                            "--depth-override-all"]}),
            "external-treatment")

    def test_warp_and_coverage_apply_per_eye_aspect_mapping(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx")
        for name in ("sbs_reprojection_ps.hlsl", "sbs_forward_coverage_cs.hlsl"):
            with self.subTest(shader=name), open(os.path.join(shader_dir, name), encoding="utf-8") as fh:
                self.assertIn("ContentToSourceUV", fh.read())

    def test_hdr_depth_input_uses_validated_color_transform(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx")
        with open(os.path.join(shader_dir, "include", "depth_color.hlsl"), encoding="utf-8") as fh:
            color = fh.read()
        self.assertIn("DepthHdrScRgbToSrgb", color)
        self.assertIn("dot(c, float3(0.2126f, 0.7152f, 0.0722f))", color)
        self.assertNotIn("c / (1.0f + c)", color)
        for name in ("rgb_to_nchw_cs.hlsl",):
            with self.subTest(shader=name), open(os.path.join(shader_dir, name), encoding="utf-8") as fh:
                text = fh.read()
                self.assertIn('include/depth_color.hlsl', text)
                self.assertIn("DepthColorToSrgb", text)

    def test_hdr_warp_stays_linear_fp16_until_pq_conversion(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        display = os.path.join(repo, "src", "platform", "windows", "display_vram.cpp")
        with open(display, encoding="utf-8") as fh:
            pipeline = fh.read()
        self.assertIn("tex_desc.Format = sbs_intermediate_linear ? DXGI_FORMAT_R16G16B16A16_FLOAT", pipeline)
        self.assertNotIn("sbs_sharpen", pipeline)
        self.assertIn("input_is_linear ? convert_Y_or_YUV_fp16_ps.get()", pipeline)
        self.assertIn("models::input_color_space::linear_sdr", pipeline)
        common = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "include", "common.hlsl")
        with open(common, encoding="utf-8") as fh:
            color = fh.read()
        self.assertIn("rgb = Rec709toRec2020(rgb)", color)
        self.assertIn("rgb *= 80", color)
        self.assertIn("return NitsToPQ(rgb)", color)

    def test_hdr_debug_preview_preserves_hue(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(repo, "src", "platform", "windows", "sbs_debug_dump.cpp")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("const float luminance = std::max(0.2126f * r + 0.7152f * g", text)
        self.assertIn("const float tone_scale = 1.0f / (1.0f + luminance)", text)
        self.assertNotIn("c = c / (1.0f + c)", text)

    def test_report_reuses_one_aggregate_decision_and_writes_sidecar(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(repo, "tools", "sbsbench", "build_report.py")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("AB_DECISION = sbsbench.evaluate_ab_decision(\n    ctrl_agg, treat_agg", text)
        self.assertIn("decision = AB_DECISION", text)
        self.assertIn('"decision_clips": DECISION_CLIPS', text)
        self.assertIn('"decision_scope": DECISION_SCOPE', text)
        self.assertIn('"source_artifact_clips": SOURCE_ARTIFACT_CLIPS', text)
        self.assertIn('"schema": 3', text)
        self.assertIn('"report_sha256": REPORT_SHA', text)
        self.assertIn('AB_DECISION["verdict"]', text)
        self.assertIn("IS_PROFILE_CMP", text)
        self.assertIn("IS_TRADEOFF_CMP = IS_MODE_CMP or IS_PROFILE_CMP", text)
        self.assertIn("def _paired_mean_aggregate", text)
        self.assertIn("a, b = _paired_mean_aggregate(k)", text)
        self.assertIn("if not any(value is not None for value in ", text)
        evidence = text[text.index("def visual_evidence_section"):text.index(
            "def source_artifact_section")]
        self.assertNotIn("stereo_art_scale_std_error_pct", evidence)
        self.assertNotIn("stereo_art_zero_std_error_pct", evidence)

        conclusion = text[text.index("def conclusion_section"):text.index("def gate_strip")]
        card = text[text.index("def _evidence_card"):text.index("def _strongest_change")]
        self.assertIn('elif state == "inconclusive":', conclusion)
        self.assertNotIn('state == "inconclusive"', card)

    def test_live_trt_contexts_are_bounded_and_engine_io_fails_closed(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(repo, "src", "video_depth_estimator.cpp")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("kMaxContextsPerEngine = 4", text)
        self.assertIn("slot.context_count >= kMaxContextsPerEngine", text)
        self.assertIn("g_trt_context_available.wait_for", text)
        self.assertIn("slot.io_compatible = have_in && have_out && input_fp32 && output_fp32", text)
        self.assertIn("validate_engine_io_locked", text)

    def test_artistic_policy_metadata_and_config_fail_closed_without_losing_depth(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(repo, "src", "video_depth_estimator.cpp")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("validate_artistic_policy_metadata", text)
        self.assertIn('metadata.value("schema", 0) != 5', text)
        self.assertIn('metadata.value("deployed_model", "") != model.name', text)
        self.assertIn('metadata.value("base_depth_model", "")', text)
        self.assertIn('"safe-frontier-multistyle-apollo-v1"', text)
        self.assertIn('"multiscale-dino-depth-dpt-stats-v1"', text)
        self.assertIn('"safe_scale_ceiling"', text)
        self.assertIn('"safe_ceiling_confidence"', text)
        self.assertGreaterEqual(text.count('"hard actionable probability"'), 2)
        self.assertNotIn('"soft safe-ceiling action probability"', text)
        self.assertIn("engine_source_marker_path", text)
        self.assertIn('marker.value("onnx_sha256", "") != onnx_sha256', text)
        self.assertIn("std::string engine_sha256;", text)
        self.assertIn("sha256_bytes(std::string_view(blob.data(), blob.size()))", text)
        self.assertIn(
            'marker.value("engine_sha256", "") != resident_engine_sha256', text)
        self.assertIn("validate_artistic_policy_metadata_impl", text)
        self.assertIn(") noexcept {", text)
        self.assertIn("artistic policy metadata validation threw", text)
        self.assertIn("sha256_file_cached(onnx_path, onnx_sha256)", text)
        self.assertIn("build_source_onnx_sha256", text)
        self.assertIn("Source ONNX changed while TensorRT was building", text)
        self.assertIn('const nlohmann::json expected_baseline', text)
        self.assertIn('"sealed-test-artistic-policy-v3"', text)
        self.assertIn('approval->value("evaluation_schema", 0) != 13', text)
        self.assertIn('approval->value("split", "") != "test"', text)
        self.assertIn('decision_accepted->is_boolean()', text)
        self.assertIn('"checkpoint_sha256"', text)
        self.assertIn('"active_split_sha256"', text)
        self.assertIn('"label_fitter_identity_sha256"', text)
        self.assertIn('"test_labels_sha256"', text)
        self.assertIn('"sealed_test_productions"', text)
        self.assertIn("artistic_policy_status() const", text)
        self.assertIn("artistic_model_onnx_sha256", text)
        self.assertIn("artistic_policy_metadata_sha256", text)
        self.assertIn("deployment_geometry_allowlist_sha256", text)
        self.assertIn("apollo-artistic-policy-deployment-v1", text)
        self.assertIn("validate_artistic_live_geometry", text)
        self.assertIn("validate_artistic_input_variant_manifest", text)
        self.assertIn("kArtisticDepthInputColorContractSha256", text)
        self.assertIn("kArtisticInputVariantManifestSha256", text)
        self.assertIn("valid_runtime_regime_acceptance", text)
        live_contract = text[text.index("static std::string_view artistic_live_color_mode"):
                             text.index("static bool valid_runtime_regime_acceptance")]
        self.assertIn("input_color_space::srgb", live_contract)
        self.assertIn("DXGI_FORMAT_B8G8R8A8_UNORM", live_contract)
        self.assertIn("input_color_space::scrgb_hdr", live_contract)
        self.assertIn("DXGI_FORMAT_R16G16B16A16_FLOAT", live_contract)
        self.assertNotIn("input_color_space::linear_sdr &&", live_contract)
        self.assertIn("pending_input_contract_exact", text)
        self.assertIn("artistic_policy_consumed_once", text)
        self.assertIn("consume_artistic_policy = false;", text)
        self.assertIn("normal depth inference remains enabled", text)

    def test_live_gpu_timer_tail_is_bounded_and_generation_safe(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        display_path = os.path.join(repo, "src", "platform", "windows", "display_vram.cpp")
        with open(display_path, encoding="utf-8") as fh:
            display = fh.read()
        with open(os.path.join(repo, "src", "sbs_perf.cpp"), encoding="utf-8") as fh:
            perf = fh.read()
        self.assertIn("drain_sbs_gpu_timers();", display)
        self.assertIn("std::chrono::milliseconds(100)", display)
        self.assertIn("sbs_perf::add_sample_ms_if_current", display)
        self.assertIn("g_generation.fetch_add", perf)

    def test_depth_transform_audit_preserves_16bit_precision(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "depth.png")
            values = np.linspace(0, 65535, 100, dtype=np.uint16).reshape(10, 10)
            Image.fromarray(values).save(path)
            stats = audit_depth_transform.frame_stats(path)
        self.assertAlmostEqual(stats["spread_p95_p05"], 0.9, delta=0.02)
        self.assertGreater(stats["saturated_low_pct"], 0.0)
        self.assertGreater(stats["saturated_high_pct"], 0.0)

    def test_depth_transform_audit_uses_image_mode_for_dark_16bit_png(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            path = fh.name
        try:
            values = np.linspace(0, 255, 100, dtype=np.uint16).reshape(10, 10)
            Image.fromarray(values).save(path)
            stats = audit_depth_transform.frame_stats(path)
            self.assertLess(stats["p99"], 0.005)
            self.assertLess(stats["spread_p95_p05"], 0.005)
        finally:
            os.unlink(path)

    def test_expected_flat_exemption_is_derived_from_stereo_axis(self):
        flat = {"expected_flat": True}
        self.assertTrue(run_eval.metric_exempt_for_clip({"axis": "stereo"}, flat))
        self.assertFalse(run_eval.metric_exempt_for_clip({"axis": "comfort"}, flat))
        self.assertFalse(run_eval.metric_exempt_for_clip({"axis": "stereo"}, {}))
        self.assertFalse(run_eval.metric_exempt_for_clip(
            {"axis": "stereo"}, {"expected_flat": "true"}
        ))

    def test_rejected_processors_and_ema_order_are_permanently_removed(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx")
        for name in ("depth_guided_upsample_cs.hlsl", "depth_guide_downsample_cs.hlsl",
                     "depth_curvature_cs.hlsl", os.path.join("include", "band_curve.hlsl")):
            self.assertFalse(os.path.exists(os.path.join(shader_dir, name)))
        with open(os.path.join(repo, "src", "config.cpp"), encoding="utf-8") as fh:
            config = fh.read()
        for key in ("sbs_3d_ema_pixel_first", "sbs_3d_guided_upsample",
                    "sbs_3d_foreground_curvature", "sbs_3d_minmax_snap",
                    "sbs_3d_range_floor", "sbs_3d_shift_profile",
                    "sbs_3d_subject_track"):
            self.assertNotIn(key, config)

    def test_bestv2_subject_pipeline_is_mandatory_and_validated(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        paths = {
            "estimator": os.path.join(repo, "src", "video_depth_estimator.cpp"),
            "display": os.path.join(repo, "src", "platform", "windows", "display_vram.cpp"),
            "harness": os.path.join(repo, "src", "sbs_bench_harness.cpp"),
        }
        text = {}
        for key, path in paths.items():
            with open(path, encoding="utf-8") as fh:
                text[key] = fh.read()
        self.assertIn("const bool core_shaders_ok", text["estimator"])
        self.assertIn("if (!valid || !input_srv)", text["estimator"])
        self.assertIn("depth_estimator->is_valid()", text["display"])
        self.assertNotIn("--no-subject-track", text["harness"])
        self.assertNotIn("sbs_cfg.subject_track", text["harness"])

    def test_bestv2_fast_curve_is_subpixel_and_live_only(self):
        depth = np.linspace(0.0, 1.0, 100001, dtype=np.float64)
        near = np.exp(-0.5 * ((depth - 0.85) / 0.24) ** 2)
        middle = np.exp(-0.5 * ((depth - 0.50) / 0.28) ** 2)
        far = np.exp(-0.5 * ((depth - 0.15) / 0.24) ** 2)
        exact = (near * 9.99 + middle * 3.0 - far * 2.52) / (near + middle + far + 1e-6)
        coeffs = (-1.39635933, 2.776208766, 21.04503417, -94.6673759,
                  376.6610774, -645.141824, 482.8701123, -133.5645677)
        approx = np.full_like(depth, coeffs[-1])
        for coefficient in reversed(coeffs[:-1]):
            approx = approx * depth + coefficient
        self.assertLess(np.max(np.abs(approx - exact)), 0.01)

        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders",
                                  "directx")
        with open(os.path.join(shader_dir, "include", "sbs_warp_common.hlsl"),
                  encoding="utf-8") as fh:
            warp_common = fh.read()
        with open(os.path.join(shader_dir, "include", "bestv2_curve.hlsl"),
                  encoding="utf-8") as fh:
            curve = fh.read()
        self.assertIn("Bestv2RawShiftPxFast(shaped_depth)", warp_common)
        self.assertNotIn("Bestv2RawShiftPx(float", curve)

    def test_tensorrt_level_is_part_of_engine_recipe(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "model_manager.h"), encoding="utf-8") as fh:
            manager = fh.read()
        with open(os.path.join(repo, "src", "video_depth_estimator.cpp"),
                  encoding="utf-8") as fh:
            estimator = fh.read()
        self.assertIn("depth_engine_builder_level = 5", manager)
        self.assertIn("trt-opt770x434-level5-v5", manager)
        self.assertIn("setBuilderOptimizationLevel(depth_engine_builder_level)", estimator)
        self.assertIn('std::string_view(name) == "artistic_global"', estimator)
        self.assertNotIn('artistic_local', estimator)
        self.assertIn("cuda_artistic_res", estimator)
        self.assertNotIn("CUdeviceptr artistic_global_out", estimator)

        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn('return config::depth_model_info {want, ""};', harness)
        self.assertNotIn("not in registry; using active model", harness)

    def test_live_and_eval_shaders_use_level3_optimization(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "src", "platform", "windows", "display_vram.cpp"),
                  encoding="utf-8") as fh:
            live = fh.read()
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness = fh.read()
        self.assertIn("flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3", live)
        self.assertIn("D3DCOMPILE_OPTIMIZATION_LEVEL3", harness)

    def test_production_warp_has_no_retired_plane_lock_path(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shader = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "sbs_reprojection_ps.hlsl")
        with open(shader, encoding="utf-8") as fh:
            text = fh.read()
        self.assertNotIn("PlaneLockTexture", text)
        self.assertNotIn("subject_plane_lock", text)

        common = os.path.join(repo, "src_assets", "windows", "assets", "shaders", "directx",
                              "include", "sbs_warp_common.hlsl")
        with open(common, encoding="utf-8") as fh:
            common_text = fh.read()
        self.assertNotIn("use_plane_lock", common_text)
        self.assertIn("sample_uv = Reproject(src_uv, eyeSign, true)", text)
        self.assertIn("sample_uv = Reproject(src_uv, eyeSign, false)", text)
        self.assertIn("MakeBestv2Params", text)
        self.assertIn(
            "DepthParallax(\n            d, s0, s1, params, use_subject_stretch)",
            text)
        self.assertNotIn("DepthParallax(d, s0, s1, shaped", text)

    def test_retired_geometry_is_absent_but_forward_coverage_diagnostic_remains(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        display = os.path.join(repo, "src", "platform", "windows", "display_vram.cpp")
        with open(display, encoding="utf-8") as fh:
            display_text = fh.read()
        self.assertNotIn("sbs_vd3d", display_text)
        self.assertNotIn("sbs_sharpen", display_text)
        with open(os.path.join(repo, "src", "sbs_bench_harness.cpp"),
                  encoding="utf-8") as fh:
            harness_text = fh.read()
        self.assertIn("sbs_forward_coverage_cs.hlsl", harness_text)
        self.assertIn("dispatch_coverage", harness_text)
        for retired in ("subject_plane_lock", "subject_plane_width", "bestv2_sharpen",
                        "ema_edge_dilation"):
            self.assertNotIn(retired, harness_text)
        shader_dir = os.path.join(repo, "src_assets", "windows", "assets", "shaders",
                                  "directx")
        for retired_shader in ("depth_plane_band_cs.hlsl", "depth_plane_combine_cs.hlsl",
                               "depth_plane_filter_cs.hlsl", "depth_plane_reduce_cs.hlsl",
                               "depth_plane_resolve_cs.hlsl", "sbs_sharpen_ps.hlsl"):
            self.assertFalse(os.path.exists(os.path.join(shader_dir, retired_shader)))
        self.assertFalse(os.path.exists(os.path.join(
            shader_dir, "include", "depth_plane_constants.hlsl")))

    def test_report_evidence_is_bounded_and_accepts_zero_based_frames(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        report = os.path.join(repo, "tools", "sbsbench", "build_report.py")
        with open(report, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("match_scale = min(1.0, 256.0 / ew)", text)
        self.assertIn("if prev_idx < 0:", text)
        self.assertNotIn("if prev_idx < 1:", text)

    def test_phase_shift_recovers_known_translation(self):
        rng = np.random.default_rng(1234)
        a = rng.random((64, 64))
        b = np.roll(a, shift=(2, -5), axis=(0, 1))
        dy, dx = sbsbench.phase_shift(a, b)
        self.assertAlmostEqual(dy, -2.0, places=5)
        self.assertAlmostEqual(dx, 5.0, places=5)

    def test_translation_residual_validates_nonwrapping_alignment(self):
        rng = np.random.default_rng(4321)
        source = rng.random((48, 64), dtype=np.float32)
        shifted = np.zeros_like(source)
        shifted[:, 7:] = source[:, :-7]
        self.assertLess(sbsbench.translation_residual(source, shifted, 0, -7),
                        sbsbench.translation_residual(source, shifted, 0, 0) * 0.1)

    def test_exact_warp_comfort_is_not_fooled_by_repetitive_image_alias(self):
        # At a square eye, an impossible -8% image alias would report about 3.38% in the
        # reference-aspect convention. The exact shader field is -2%, so the hard evidence is
        # only about 0.84% and cannot be replaced by the alias.
        exact = np.full((8, 8), -0.02, np.float32)
        positive, negative = sbsbench.exact_warp_comfort(exact, 64, 64)
        _, alias_negative = sbsbench.comfort_disparity(
            np.asarray([-5.12]), np.asarray([1.0]), 64, 64)
        self.assertLess(negative, 1.0)
        self.assertGreater(alias_negative, 3.0)
        self.assertEqual(positive, 0.0)

    def test_exact_warp_pop_spread_uses_signed_hlsl_percentiles(self):
        exact = np.linspace(-0.02, 0.03, 101, dtype=np.float32).reshape(1, -1)
        lo, hi = np.percentile(exact, (5.0, 95.0))
        spread = sbsbench.exact_warp_pop_spread(exact, 5120, 2160)
        self.assertAlmostEqual(spread, (hi - lo) * 100.0, places=6)

    def test_exact_warp_metrics_include_horizontal_letterbox_scale(self):
        exact = np.linspace(-0.02, 0.03, 101, dtype=np.float32).reshape(1, -1)
        square_source = np.zeros((100, 100), np.float32)
        content_scale = sbsbench.source_content_scale_x(square_source, 200, 100)
        full = sbsbench.exact_warp_pop_spread(exact, 200, 100)
        letterboxed = sbsbench.exact_warp_pop_spread(
            exact * content_scale, 200, 100)
        self.assertEqual(content_scale, 0.5)
        self.assertAlmostEqual(letterboxed, full * 0.5, places=6)

    def test_exact_pop_is_primary_and_image_pop_is_diagnostic(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "tools", "sbsbench", "thresholds.json"),
                  encoding="utf-8") as fh:
            metrics = json.load(fh)["metrics"]
        self.assertEqual(metrics["exact_pop_spread_pct"]["role"], "primary")
        self.assertEqual(metrics["exact_pop_spread_pct"]["evidence"],
                         "exact_hlsl_full_binocular_disparity")
        self.assertEqual(metrics["pop_spread_pct"]["role"], "diagnostic")
        with open(os.path.join(repo, "tools", "sbsbench", "run_eval.py"),
                  encoding="utf-8") as fh:
            runner = fh.read()
        self.assertIn("exact_clamped_full_binocular_normalized", runner)
        self.assertIn("unclamped_full_binocular_normalized_at_artistic_scale_1", runner)
        self.assertIn("expected_output_ids != unclamped_disparity_ids", runner)

    def test_sequence_frame_uses_exact_field_for_primary_pop(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "sbs_00000.png")
            Image.fromarray(np.zeros((32, 64, 3), np.uint8)).save(path)
            source = np.zeros((32, 16), np.float32)  # half-width content in a square eye
            exact = np.zeros((32, 32), dtype=np.float32)
            exact[:, 8:24] = np.linspace(
                -0.02, 0.03, 32 * 16, dtype=np.float32
            ).reshape(32, 16)
            row, _, _ = sbsbench.measure_seq_frame(
                path, src_gray=source, warp_disparity=exact)
        expected = sbsbench.exact_warp_pop_spread(exact[:, 8:24], 32, 32)
        self.assertAlmostEqual(row["exact_pop_spread_pct"], expected, places=6)

    def test_disparity_field_rejects_photometrically_invalid_peak(self):
        rng = np.random.default_rng(987)
        left = rng.random((64, 64), dtype=np.float32)
        with mock.patch.object(sbsbench, "phase_shift", return_value=(15.0, -30.0)):
            field = sbsbench.disparity_field(left, left.copy(), tile=64, stride=64)
        self.assertIsNone(field)

    def test_disparity_field_covers_tile_sized_frame_and_final_borders(self):
        rng = np.random.default_rng(2026)
        left = rng.random((192, 320), dtype=np.float32)
        right = np.roll(left, 3, axis=1)
        field = sbsbench.disparity_field(left, right, tile=192, stride=128)
        self.assertIsNotNone(field)
        self.assertEqual(len(field[0]), 2)  # x=0 and the border-aligned x=128 tile
        self.assertEqual(sbsbench._tile_positions(320, 192, 128), [0, 128])

    def test_sequence_joins_by_frame_identity(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            sbs = np.zeros((16, 32, 3), dtype=np.uint8)
            src = np.zeros((16, 16, 3), dtype=np.uint8)
            Image.fromarray(sbs).save(os.path.join(seq, "sbs_00007.png"))
            Image.fromarray(src).save(os.path.join(frames, "frame_00007.png"))
            rows, agg = sbsbench.measure_sequence(seq, frames)
            self.assertEqual(rows[0]["_frame_id"], 7)
            self.assertEqual(agg["_n"], 1)

    def test_sequence_rejects_positional_mispairing(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            blank = np.zeros((16, 32, 3), dtype=np.uint8)
            Image.fromarray(blank).save(os.path.join(seq, "sbs_00008.png"))
            Image.fromarray(blank[:, :16]).save(os.path.join(frames, "frame_00007.png"))
            with self.assertRaisesRegex(ValueError, "frame-id mismatch"):
                sbsbench.measure_sequence(seq, frames)

    def test_sampled_sequence_accepts_source_and_stereo_reference_supersets(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            gt_right = os.path.join(frames, "gt_right")
            os.makedirs(seq)
            os.makedirs(gt_right)
            rng = np.random.default_rng(20260715)
            for frame_id in range(5):
                src = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(src).save(os.path.join(frames, f"frame_{frame_id:05d}.png"))
                Image.fromarray(src).save(os.path.join(
                    gt_right, f"frame_{frame_id:05d}.png"))
                if frame_id % 2 == 0:
                    Image.fromarray(np.concatenate((src, src), axis=1)).save(
                        os.path.join(seq, f"sbs_{frame_id:05d}.png"))
                    depth = np.tile(np.linspace(0, 65535, 16, dtype=np.uint16), (16, 1))
                    Image.fromarray(depth).save(
                        os.path.join(seq, f"depth_{frame_id:05d}.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"dataset": "Authored Movie", "required_gt_stereo": True}, fh)
            rows, agg = sbsbench.measure_sequence(seq, frames)
            self.assertEqual([row["_frame_id"] for row in rows], [0, 2, 4])
            self.assertEqual(agg["_n"], 3)
            self.assertIn("stereo_gt_psnr", agg)

    def test_sparse_label_segments_reset_temporal_metrics_across_gaps(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            for frame_id, value in ((0, 0), (1, 0), (10, 255), (11, 255)):
                eye = np.full((32, 32, 3), value, np.uint8)
                Image.fromarray(np.concatenate((eye, eye), axis=1)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png")
                )
                Image.fromarray(eye).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png")
                )
            with open(os.path.join(frames, "label_frames.json"), "w",
                      encoding="utf-8") as stream:
                json.dump({"schema": 1, "frame_ids": [1, 11]}, stream)
            with open(os.path.join(frames, "meta.json"), "w",
                      encoding="utf-8") as stream:
                json.dump({"required_temporal_evidence": True}, stream)

            rows, agg = sbsbench.measure_sequence(seq, frames)

            by_id = {row["_frame_id"]: row for row in rows}
            self.assertNotIn("flicker", by_id[10])
            self.assertEqual(agg["flicker_p95"], 0.0)

    def test_sparse_labels_do_not_score_adjacent_bridge_between_selected_pairs(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            # Labels 1 and 3 select exact pairs 0->1 and 2->3. The materialized artifact set is
            # nevertheless contiguous, so a naive consecutive-ID scan would also score 1->2.
            for frame_id, value in enumerate((0, 0, 255, 255)):
                eye = np.full((32, 32, 3), value, np.uint8)
                Image.fromarray(np.concatenate((eye, eye), axis=1)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png")
                )
                Image.fromarray(eye).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png")
                )
            with open(os.path.join(frames, "label_frames.json"), "w",
                      encoding="utf-8") as stream:
                json.dump({"schema": 1, "frame_ids": [1, 3]}, stream)
            with open(os.path.join(frames, "meta.json"), "w",
                      encoding="utf-8") as stream:
                json.dump({"required_temporal_evidence": True}, stream)

            rows, agg = sbsbench.measure_sequence(seq, frames)

            by_id = {row["_frame_id"]: row for row in rows}
            self.assertIn("flicker", by_id[1])
            self.assertNotIn("flicker", by_id[2])
            self.assertIn("flicker", by_id[3])
            self.assertEqual(agg["flicker_p95"], 0.0)

    def test_sparse_label_manifest_requires_valid_clip_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            for frame_id in range(2):
                eye = np.zeros((16, 16, 3), np.uint8)
                Image.fromarray(np.concatenate((eye, eye), axis=1)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png")
                )
                Image.fromarray(eye).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png")
                )
            with open(os.path.join(frames, "label_frames.json"), "w",
                      encoding="utf-8") as stream:
                json.dump({"schema": 1, "frame_ids": [1]}, stream)

            with self.assertRaisesRegex(ValueError, "missing required clip metadata"):
                sbsbench.measure_sequence(seq, frames)

            meta_path = os.path.join(frames, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as stream:
                stream.write("{broken")
            with self.assertRaisesRegex(ValueError, "invalid clip metadata"):
                sbsbench.measure_sequence(seq, frames)

            with open(meta_path, "w", encoding="utf-8") as stream:
                json.dump({"required_gt_flow": "false"}, stream)
            with self.assertRaisesRegex(ValueError, "evidence flags must be booleans"):
                sbsbench.measure_sequence(seq, frames)

            with open(meta_path, "w", encoding="utf-8") as stream:
                json.dump({"dataset": 0}, stream)
            with self.assertRaisesRegex(ValueError, "dataset must be a nonempty string"):
                sbsbench.measure_sequence(seq, frames)

    def test_plain_sequence_keeps_metadata_optional_but_rejects_malformed_file(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            eye = np.zeros((16, 16, 3), np.uint8)
            Image.fromarray(np.concatenate((eye, eye), axis=1)).save(
                os.path.join(seq, "sbs_00000.png")
            )
            Image.fromarray(eye).save(os.path.join(frames, "frame_00000.png"))

            rows, _agg = sbsbench.measure_sequence(seq, frames)
            self.assertEqual([row["_frame_id"] for row in rows], [0])

            with open(os.path.join(frames, "meta.json"), "w",
                      encoding="utf-8") as stream:
                json.dump([], stream)
            with self.assertRaisesRegex(ValueError, "root must be an object"):
                sbsbench.measure_sequence(seq, frames)

    def test_sparse_authored_stereo_requires_sidecars_only_for_label_targets(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            gt_right = os.path.join(frames, "gt_right")
            os.makedirs(seq)
            os.makedirs(gt_right)
            rng = np.random.default_rng(20260715)
            for frame_id in (0, 1, 10, 11):
                src = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(src).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png")
                )
                Image.fromarray(np.concatenate((src, src), axis=1)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png")
                )
                depth = np.tile(np.linspace(0, 65535, 16, dtype=np.uint16), (16, 1))
                Image.fromarray(depth).save(
                    os.path.join(seq, f"depth_{frame_id:05d}.png")
                )
                if frame_id in (1, 11):
                    Image.fromarray(src).save(
                        os.path.join(gt_right, f"frame_{frame_id:05d}.png")
                    )
            with open(os.path.join(frames, "label_frames.json"), "w",
                      encoding="utf-8") as stream:
                json.dump({"schema": 1, "frame_ids": [1, 11]}, stream)
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as stream:
                json.dump({"required_gt_stereo": True}, stream)

            rows, agg = sbsbench.measure_sequence(seq, frames)

            stereo_rows = [row["_frame_id"] for row in rows if "stereo_gt_psnr" in row]
            self.assertEqual(stereo_rows, [1, 11])
            self.assertIn("stereo_art_polarity_ok", agg)
            os.remove(os.path.join(gt_right, "frame_00011.png"))
            with self.assertRaisesRegex(ValueError, "every label target"):
                sbsbench.measure_sequence(seq, frames)

    def test_sampled_sequence_rejects_one_frame_flow_as_invalid_for_gaps(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            flow = os.path.join(frames, "gt_flow")
            os.makedirs(seq)
            os.makedirs(flow)
            blank_sbs = np.zeros((16, 32, 3), np.uint8)
            blank_src = np.zeros((16, 16, 3), np.uint8)
            for frame_id in range(3):
                Image.fromarray(blank_src).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png"))
            for frame_id in (0, 2):
                Image.fromarray(blank_sbs).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png"))
            np.savez(os.path.join(flow, "frame_00001.npz"),
                     flow=np.zeros((16, 16, 2), np.float32), valid=np.ones((16, 16), bool))
            np.savez(os.path.join(flow, "frame_00002.npz"),
                     flow=np.zeros((16, 16, 2), np.float32), valid=np.ones((16, 16), bool))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"required_gt_flow": True}, fh)
            with self.assertRaisesRegex(ValueError, "requires GT optical flow"):
                sbsbench.measure_sequence(seq, frames)

    def test_public_clip_rejects_missing_required_ground_truth(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                os.path.join(seq, "sbs_00000.png"))
            Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                os.path.join(frames, "frame_00000.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                fh.write('{"dataset":"Example Public Dataset","required_gt_depth":true}')
            with self.assertRaisesRegex(ValueError, "requires GT depth"):
                sbsbench.measure_sequence(seq, frames)

    def test_required_gt_depth_rejects_partial_target_or_temporal_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            gt_dir = os.path.join(frames, "gt_depth")
            os.makedirs(seq)
            os.makedirs(gt_dir)
            for frame_id in range(2):
                Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png"))
                Image.fromarray(np.full((8, 16), 32768, np.uint16)).save(
                    os.path.join(seq, f"depth_{frame_id:05d}.png"))
                Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png"))
            Image.fromarray(np.full((8, 16), 32768, np.uint16)).save(
                os.path.join(gt_dir, "frame_00000.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"required_gt_depth": True, "gt_depth_kind": "disparity"}, fh)

            with self.assertRaisesRegex(
                    ValueError, "GT-depth/SBS frame-id mismatch|every target/temporal endpoint"):
                sbsbench.measure_sequence(seq, frames)

    def test_public_clip_rejects_missing_required_optical_flow(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            for frame_id in range(2):
                Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png"))
                Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                fh.write('{"required_gt_flow":true}')
            with self.assertRaisesRegex(ValueError, "requires GT optical flow"):
                sbsbench.measure_sequence(seq, frames)

    def test_public_clip_rejects_missing_required_stereo_reference(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                os.path.join(seq, "sbs_00000.png"))
            Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                os.path.join(frames, "frame_00000.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                fh.write('{"required_gt_stereo":true}')
            with self.assertRaisesRegex(ValueError, "requires GT stereo"):
                sbsbench.measure_sequence(seq, frames)

    def test_public_clip_rejects_missing_required_depth_lag_metric(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            gt_dir = os.path.join(frames, "gt_depth")
            os.makedirs(seq)
            os.makedirs(gt_dir)
            for frame_id in range(2):
                Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(
                    os.path.join(seq, f"sbs_{frame_id:05d}.png"))
                Image.fromarray(np.full((8, 16), 32768, np.uint16)).save(
                    os.path.join(seq, f"depth_{frame_id:05d}.png"))
                Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                    os.path.join(frames, f"frame_{frame_id:05d}.png"))
                Image.fromarray(np.full((8, 16), 32768, np.uint16)).save(
                    os.path.join(gt_dir, f"frame_{frame_id:05d}.png"))
            with open(os.path.join(frames, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"required_gt_depth": True, "gt_depth_kind": "disparity"}, fh)
            with mock.patch.object(sbsbench, "depth_ground_truth_lag", return_value=None):
                with self.assertRaisesRegex(ValueError, "depth_gt_lag_f1_p95"):
                    sbsbench.measure_sequence(seq, frames)

    def test_duplicate_numeric_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(os.path.join(root, "frame_1.png"))
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(os.path.join(root, "frame_01.jpg"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                sbsbench.indexed_files(os.path.join(root, "frame_*.*"), "frame_")

    def test_disocclusion_ratio_requires_minimum_support(self):
        eye = np.zeros((64, 64), dtype=np.float32)
        depth = np.full((16, 16), 0.5, dtype=np.float32)
        frac, smear = sbsbench.disocclusion_metrics(eye, depth)
        self.assertLess(frac, sbsbench.MIN_DISOCC_FRAC)
        self.assertIsNone(smear)

    def test_exact_warp_mask_restricts_fill_error_and_artifact_overlap(self):
        source = np.zeros((64, 64), dtype=np.float32)
        left = source.copy()
        right = source.copy()
        left[20:30, 25:35] = 1.0
        right[20:30, 25:35] = 1.0
        mask = np.zeros((64, 128, 3), dtype=np.float32)
        mask[20:30, 25:35, 0] = 1.0
        mask[20:30, 64 + 25:64 + 35, 0] = 1.0
        metrics = sbsbench.warp_hole_metrics(left, right, mask, source)
        self.assertGreater(metrics["warp_hole_pct"], 1.0)
        self.assertNotIn("warp_unresolved_pct", metrics)
        self.assertGreater(metrics["hole_source_residual_p95"], 100.0)
        self.assertGreater(metrics["hole_bad_fill_pct"], 80.0)
        self.assertGreater(metrics["artifact_in_hole_pct"], 80.0)

    def test_depth_is_diagnostic_not_part_of_artifact_score(self):
        clean = {"exact_pop_spread_pct": 0.0}
        false_stereo = {"exact_pop_spread_pct": 0.2}
        self.assertGreater(
            sbsbench.sbs_score(clean, expected_flat=True)["q_depth"],
            sbsbench.sbs_score(false_stereo, expected_flat=True)["q_depth"])
        self.assertLess(
            sbsbench.sbs_score(clean)["q_depth"],
            sbsbench.sbs_score(false_stereo)["q_depth"])
        self.assertEqual(sbsbench.sbs_score(clean)["score"],
                         sbsbench.sbs_score(false_stereo)["score"])

    def test_metric_delta_class_uses_gate_tolerance_and_direction(self):
        lower = {"better": "lower", "rel_tol": 0.25, "abs_floor": 0.5}
        self.assertEqual(sbsbench.metric_delta_class(2.0, 2.4, lower), "noise")
        self.assertEqual(sbsbench.metric_delta_class(2.0, 2.6, lower), "regressed")
        self.assertEqual(sbsbench.metric_delta_class(2.0, 1.4, lower), "improved")

    def test_metric_roles_control_committed_gate(self):
        diagnostic = {"role": "diagnostic", "better": "lower",
                      "rel_tol": 0.0, "abs_floor": 0.1}
        hard = {"role": "hard", "better": "lower", "hard_max": 0.5,
                "rel_tol": 0.0, "abs_floor": 0.1}
        self.assertFalse(sbsbench.metric_gate_failed(0.0, 99.0, diagnostic))
        self.assertFalse(sbsbench.metric_gate_failed(0.0, 0.49, hard))
        self.assertTrue(sbsbench.metric_gate_failed(0.0, 0.51, hard))
        hard_min = {"role": "hard", "better": "higher", "hard_min": 90.0,
                    "rel_tol": 0.0, "abs_floor": 1.0}
        self.assertFalse(sbsbench.metric_gate_failed(95.0, 91.0, hard_min))
        self.assertTrue(sbsbench.metric_gate_failed(95.0, 89.0, hard_min))

    def test_artistic_stereo_metrics_cannot_drive_committed_gate(self):
        with open(os.path.join(run_eval.SCRIPT_DIR, "thresholds.json"),
                  encoding="utf-8") as fh:
            specs = json.load(fh)["metrics"]
        artistic = {key: spec for key, spec in specs.items()
                    if key.startswith("stereo_art_")}
        self.assertTrue(artistic)
        self.assertTrue(all(spec["role"] == "diagnostic"
                            and spec["axis"] == "artistic-style"
                            for spec in artistic.values()))

    def test_ab_decision_preserves_primary_axis_tradeoff(self):
        specs = {
            "pop": {"role": "primary", "axis": "stereo", "better": "higher",
                    "rel_tol": 0.0, "abs_floor": 0.5},
            "halo": {"role": "primary", "axis": "warp", "better": "lower",
                     "rel_tol": 0.0, "abs_floor": 0.5},
            "legacy_proxy": {"role": "diagnostic", "axis": "warp", "better": "lower",
                             "rel_tol": 0.0, "abs_floor": 0.1},
        }
        result = sbsbench.evaluate_ab_decision(
            {"clip": {"pop": 4.0, "halo": 2.0, "legacy_proxy": 0.0}},
            {"clip": {"pop": 5.0, "halo": 3.0, "legacy_proxy": 99.0}},
            ["clip"], specs)
        self.assertEqual(result["verdict"], "tradeoff")
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["regressed"], 1)

    def test_ab_decision_hard_constraint_cannot_be_traded(self):
        specs = {
            "vmis": {"role": "hard", "axis": "comfort", "hard_max": 0.5,
                     "better": "lower", "rel_tol": 0.0, "abs_floor": 0.1},
            "pop": {"role": "primary", "axis": "stereo", "better": "higher",
                    "rel_tol": 0.0, "abs_floor": 0.5},
        }
        result = sbsbench.evaluate_ab_decision(
            {"clip": {"vmis": 0.1, "pop": 4.0}},
            {"clip": {"vmis": 0.6, "pop": 8.0}}, ["clip"], specs)
        self.assertEqual(result["verdict"], "reject_hard")

    def test_ab_decision_skips_only_two_sided_optional_absence(self):
        optional = {
            "optional_gt": {"role": "primary", "axis": "depth", "better": "lower",
                            "rel_tol": 0.0, "abs_floor": 0.1},
        }
        both_absent = sbsbench.evaluate_ab_decision(
            {"clip": {"_n": 2}}, {"clip": {"_n": 2}}, ["clip"], optional)
        self.assertEqual(both_absent["verdict"], "neutral")
        self.assertEqual(both_absent["missing_evidence"], [])

        one_absent = sbsbench.evaluate_ab_decision(
            {"clip": {"_n": 2, "optional_gt": 1.0}},
            {"clip": {"_n": 2}}, ["clip"], optional)
        self.assertEqual(one_absent["verdict"], "inconclusive")
        self.assertEqual(len(one_absent["missing_evidence"]), 1)

    def test_ab_decision_fails_closed_for_two_sided_required_absence(self):
        required = {
            "exact": {"role": "primary", "axis": "stereo", "better": "higher",
                      "rel_tol": 0.0, "abs_floor": 0.1, "required_evidence": True},
        }
        result = sbsbench.evaluate_ab_decision(
            {"clip": {"_n": 2}}, {"clip": {"_n": 2}}, ["clip"], required)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertEqual(len(result["missing_evidence"]), 1)

    def test_required_primary_evidence_respects_frame_and_flat_applicability(self):
        thresholds = {"metrics": {
            "exact": {"role": "primary", "axis": "stereo",
                      "required_evidence": True},
            "temporal": {"role": "primary", "axis": "stability",
                         "required_evidence": True, "min_frames": 2},
        }}
        self.assertEqual(
            run_eval.missing_required_metric_evidence({"_n": 1}, thresholds, {}),
            ["exact"],
        )
        self.assertEqual(
            run_eval.missing_required_metric_evidence({"_n": 2}, thresholds, {}),
            ["exact", "temporal"],
        )
        self.assertEqual(
            run_eval.missing_required_metric_evidence(
                {"_n": 2, "temporal": 0.0}, thresholds, {"expected_flat": True}
            ),
            [],
        )

    def test_source_residual_accepts_horizontal_parallax_and_detects_corruption(self):
        rng = np.random.default_rng(42)
        src = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        shifted = sbsbench._shift_x_edge(src, 5)
        clean = sbsbench.source_match_residual(shifted, src, max_shift=8)
        corrupted = shifted.copy()
        corrupted[24:72, 60:100] = 0.0
        damaged = sbsbench.source_match_residual(corrupted, src, max_shift=8)
        self.assertLess(clean[1], 0.01)
        self.assertGreater(damaged[1], clean[1] + 5.0)

    def test_static_region_jitter_ignores_source_motion_but_detects_static_warp_change(self):
        rng = np.random.default_rng(9)
        src = np.round(rng.random((64, 96), dtype=np.float32) * 255.0) / 255.0
        stable, support = sbsbench.static_region_jitter(src, src, src, src, src, src,
                                                        min_support=0.5)
        self.assertAlmostEqual(stable, 0.0)
        self.assertEqual(support, 1.0)
        changed = src.copy()
        changed[16:48, 30:66] = np.clip(changed[16:48, 30:66] + 0.2, 0, 1)
        jitter, _ = sbsbench.static_region_jitter(changed, changed, src, src, src, src,
                                                  min_support=0.5)
        self.assertGreater(jitter, 20.0)
        moving_src = np.roll(src, 8, axis=1)
        skipped, moving_support = sbsbench.static_region_jitter(
            moving_src, moving_src, src, src, moving_src, src, min_support=0.5)
        self.assertIsNone(skipped)
        self.assertLess(moving_support, 0.5)

    def test_comfort_disparity_reports_both_signed_tails(self):
        dx = np.array([-12.0, -8.0, 0.0, 6.0, 10.0])
        weights = np.ones_like(dx)
        positive, negative = sbsbench.comfort_disparity(
            dx, weights, eye_width=400,
            eye_height=400 / sbsbench.REFERENCE_STREAM_ASPECT, tail=0.8)
        self.assertAlmostEqual(positive, 1.5)
        self.assertAlmostEqual(negative, 3.0)

    def test_perceived_disparity_is_client_aspect_invariant(self):
        ref = sbsbench.perceived_disparity_pct(51.2, 5120, 2160)
        # The aspect correction keeps pixel disparity constant when pixel height is unchanged;
        # at a taller raster it grows in direct proportion to height.
        uhd = sbsbench.perceived_disparity_pct(51.2, 3840, 2160)
        tall = sbsbench.perceived_disparity_pct(51.2 * 3840.0 / 2160.0, 3552, 3840)
        self.assertAlmostEqual(ref, uhd, places=6)
        self.assertAlmostEqual(ref, tall, places=6)

    def test_hard_integrity_aggregates_worst_frame_not_mean(self):
        agg = sbsbench.aggregate([
            {"source_coverage_pct": 100.0, "positive_disparity_pct": 0.5,
             "vmisalign_pct": 0.01},
            {"source_coverage_pct": 70.0, "positive_disparity_pct": 4.0,
             "vmisalign_pct": 0.2},
        ])
        self.assertEqual(agg["source_coverage_pct"], 70.0)
        self.assertEqual(agg["positive_disparity_pct"], 4.0)
        self.assertEqual(agg["vmisalign_pct"], 0.2)

    def test_resolution_independent_metrics_preserve_normalized_geometry(self):
        dx_small = np.array([-4.0, 0.0, 4.0])
        dx_large = dx_small * 2.0
        weights = np.ones(3)
        spread_small = sbsbench.pop_spread(dx_small, weights) / 400.0 * 100.0
        spread_large = sbsbench.pop_spread(dx_large, weights) / 800.0 * 100.0
        self.assertAlmostEqual(spread_small, spread_large)

    def test_source_coverage_and_integrity_detect_missing_content(self):
        rng = np.random.default_rng(22)
        src = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        clean = sbsbench._shift_x_edge(src, 5)
        good = sbsbench.source_relative_metrics(clean, src, max_shift=8)
        damaged = clean.copy()
        damaged[20:76, 60:110] = 0.0
        bad = sbsbench.source_relative_metrics(damaged, src, max_shift=8)
        self.assertGreater(good["source_coverage_pct"], 99.0)
        self.assertGreater(good["image_integrity_pct"], 99.0)
        self.assertLess(bad["source_coverage_pct"], good["source_coverage_pct"] - 15.0)
        self.assertLess(bad["image_integrity_pct"], good["image_integrity_pct"] - 10.0)

    def test_source_relative_halo_and_stretch_subtract_real_source_structure(self):
        y, x = np.mgrid[:96, :160]
        src = (0.35 + 0.2 * np.sin(x * 0.55) + 0.15 * np.sin(y * 0.3)).astype(np.float32)
        depth = np.full((24, 40), 0.2, np.float32)
        depth[:, 20:] = 0.8
        clean = sbsbench.source_relative_metrics(src, src, depth, max_shift=4)
        halo_eye = src.copy()
        halo_eye[:, 79:82] = 1.0
        halo = sbsbench.source_relative_metrics(halo_eye, src, depth, max_shift=4)
        stretch_eye = src.copy()
        stretch_eye[:, 82:115] = stretch_eye[:, 82:83]
        stretch = sbsbench.source_relative_metrics(stretch_eye, src, depth, max_shift=4)
        self.assertGreater(halo["source_halo_p95"], clean["source_halo_p95"] + 3.0)
        self.assertGreater(stretch["source_stretch_pct"], clean["source_stretch_pct"] + 10.0)

    def test_ground_truth_depth_metrics_reward_aligned_structure(self):
        gt = np.full((96, 160), 0.25, np.float32)
        gt[:, 80:] = 0.75
        equivalent = gt * 0.8 + 0.1  # monocular scale/shift ambiguity is intentionally free
        flat = np.full_like(gt, 0.5)
        good = sbsbench.depth_ground_truth_metrics(equivalent, gt)
        bad = sbsbench.depth_ground_truth_metrics(flat, gt)
        self.assertLess(good["depth_gt_si_rmse"], 0.01)
        self.assertGreater(good["depth_gt_edge_f1"], 99.0)
        self.assertGreater(bad["depth_gt_si_rmse"], 40.0)
        self.assertLess(bad["depth_gt_edge_f1"], 1.0)

    def test_ground_truth_depth_metrics_reject_inverted_polarity(self):
        gt = np.full((96, 160), 0.2, np.float32)
        gt[:, 80:] = 0.8
        inverted = 1.0 - gt
        metrics = sbsbench.depth_ground_truth_metrics(inverted, gt)
        self.assertGreater(metrics["depth_gt_si_rmse"], 20.0)
        self.assertLess(metrics["depth_gt_edge_f1"], 1.0)

    def test_ground_truth_stereo_ignores_only_global_horizontal_offset(self):
        rng = np.random.default_rng(73)
        reference = rng.random((96, 160), dtype=np.float32)
        shifted = sbsbench._shift_x_edge(reference, 7)
        good = sbsbench.stereo_ground_truth_metrics(shifted, reference)
        vertically_wrong = np.roll(shifted, 4, axis=0)
        bad = sbsbench.stereo_ground_truth_metrics(vertically_wrong, reference)
        self.assertGreater(good["stereo_gt_psnr"], 80.0)
        self.assertGreater(good["stereo_gt_ssim"], 0.999)
        self.assertLess(good["stereo_gt_residual_p95"], 1.0)
        self.assertGreater(good["stereo_gt_coverage_pct"], 99.9)
        self.assertLess(bad["stereo_gt_psnr"], good["stereo_gt_psnr"] - 20.0)
        self.assertLess(bad["stereo_gt_ssim"], good["stereo_gt_ssim"] - 0.2)
        self.assertGreater(bad["stereo_gt_residual_p95"],
                           good["stereo_gt_residual_p95"] + 10.0)

    def test_ground_truth_stereo_detects_local_content_corruption(self):
        rng = np.random.default_rng(91)
        reference = rng.random((96, 160), dtype=np.float32)
        clean = sbsbench._shift_x_edge(reference, -5)
        corrupted = clean.copy()
        corrupted[24:72, 60:110] = 0.0
        good = sbsbench.stereo_ground_truth_metrics(clean, reference)
        bad = sbsbench.stereo_ground_truth_metrics(corrupted, reference)
        self.assertLess(bad["stereo_gt_psnr"], good["stereo_gt_psnr"] - 10.0)
        self.assertLess(bad["stereo_gt_ssim"], good["stereo_gt_ssim"] - 0.05)
        self.assertLess(bad["stereo_gt_coverage_pct"],
                        good["stereo_gt_coverage_pct"] - 5.0)

    def test_artistic_stereo_metrics_recover_positive_global_style(self):
        rng = np.random.default_rng(119)
        source = rng.random((96, 160), dtype=np.float32)
        depth_row = np.repeat(np.array([0.1, 0.5, 0.9], np.float32), [54, 53, 53])
        depth = np.broadcast_to(depth_row, source.shape)

        def render(scale, offset, eye_fraction):
            disparity = scale * depth + offset
            shift = np.rint(-disparity * eye_fraction).astype(np.int32)
            x = np.arange(source.shape[1])[None, :] - shift
            x = np.clip(x, 0, source.shape[1] - 1)
            return np.take_along_axis(source, np.broadcast_to(x, source.shape), axis=1)

        reference = render(12.0, -3.0, 1.0)
        matching_right = render(12.0, -3.0, 0.5)
        weak_right = render(5.0, 1.0, 0.5)
        good = sbsbench.artistic_stereo_metrics(
            source, matching_right, reference, depth)
        bad = sbsbench.artistic_stereo_metrics(source, weak_right, reference, depth)
        self.assertEqual(good["stereo_art_polarity_ok"], 100.0)
        self.assertLess(good["stereo_art_scale_error_pct"],
                        bad["stereo_art_scale_error_pct"])
        self.assertLess(good["stereo_art_zero_error_pct"],
                        bad["stereo_art_zero_error_pct"])
        self.assertGreater(good["stereo_art_ddc_iou"], 0.0)

    def test_artistic_stereo_metrics_reject_inverted_polarity(self):
        rng = np.random.default_rng(127)
        source = rng.random((96, 160), dtype=np.float32)
        depth_row = np.repeat(np.array([0.1, 0.5, 0.9], np.float32), [54, 53, 53])
        depth = np.broadcast_to(depth_row, source.shape)

        def render(disparity):
            shift = np.rint(-disparity).astype(np.int32)
            x = np.arange(source.shape[1])[None, :] - shift
            x = np.clip(x, 0, source.shape[1] - 1)
            return np.take_along_axis(source, np.broadcast_to(x, source.shape), axis=1)

        reference = render(10.0 * depth - 2.0)
        inverted_right = render(-(5.0 * depth - 1.0))
        metrics = sbsbench.artistic_stereo_metrics(
            source, inverted_right, reference, depth)
        self.assertEqual(metrics["stereo_art_polarity_ok"], 0.0)

    def test_artistic_aggregate_suppresses_partial_validity(self):
        complete = {
            "stereo_art_polarity_ok": 100.0,
            "stereo_art_scale_pct": 1.0,
            "stereo_art_zero_pct": 0.2,
            "stereo_ref_scale_pct": 2.0,
            "stereo_ref_zero_pct": 0.4,
            "stereo_art_scale_error_pct": 1.0,
            "stereo_art_zero_error_pct": 0.2,
            "stereo_art_support_pct": 80.0,
            "stereo_art_ddc_iou": 10.0,
            "stereo_ref_ddc_iou": 20.0,
        }
        invalid = {"stereo_art_polarity_ok": 0.0}
        rows = [complete, invalid]
        agg = sbsbench.aggregate(rows)
        sbsbench.finalize_artistic_stereo_aggregate(rows, agg)
        self.assertEqual(agg["stereo_art_polarity_ok"], 50.0)
        for key in sbsbench.ARTISTIC_STEREO_FRAME_METRICS:
            self.assertNotIn(key, agg)

    def test_ground_truth_depth_lag_detects_previous_frame_geometry(self):
        previous = np.zeros((32, 48), np.float32)
        previous[8:24, 8:20] = 1.0
        current = np.zeros_like(previous)
        current[8:24, 18:30] = 1.0
        self.assertGreater(
            sbsbench.depth_ground_truth_lag(previous, current, previous), 50.0)
        self.assertEqual(
            sbsbench.depth_ground_truth_lag(current, current, previous), 0.0)

    def test_ground_truth_ghost_detects_previous_only_boundary(self):
        previous = np.zeros((32, 48), np.float32)
        previous[8:24, 8:20] = 1.0
        current = np.zeros_like(previous)
        current[8:24, 18:30] = 1.0
        self.assertEqual(
            sbsbench.depth_ground_truth_ghost(current, current, previous), 0.0)
        self.assertGreater(
            sbsbench.depth_ground_truth_ghost(previous, current, previous), 40.0)

    def test_ground_truth_edge_tolerance_works_in_both_axes(self):
        gt = np.full((96, 160), 0.25, np.float32)
        gt[48:, :] = 0.75
        shifted = np.full_like(gt, 0.25)
        shifted[49:, :] = 0.75
        metrics = sbsbench.depth_ground_truth_metrics(shifted, gt)
        self.assertGreater(metrics["depth_gt_edge_f1"], 99.0)

    def test_metric_depth_resize_does_not_invert_interpolated_invalid_holes(self):
        gt = np.full((48, 80), 2.0, np.float32)
        gt[12:36, 38:42] = 0.0
        resized, valid = sbsbench.resize_metric_depth(gt, 40, 24)
        self.assertTrue(np.all(resized[valid] > 1.9))
        self.assertTrue(np.all(resized[valid] < 2.1))
        prediction = np.full((24, 40), 0.5, np.float32)
        metrics = sbsbench.depth_ground_truth_metrics(prediction, gt, "metric")
        self.assertLess(metrics["depth_gt_si_rmse"], 0.01)
        self.assertGreater(metrics["depth_gt_edge_f1"], 99.0)

    def test_optical_flow_temporal_metric_compensates_motion(self):
        rng = np.random.default_rng(31)
        previous = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        current = np.roll(previous, 5, axis=1)
        stable, _, support = sbsbench.flow_temporal_metrics(
            current, current, previous, previous, current, previous, min_support=0.1)
        corrupted = current.copy()
        corrupted[20:75, 60:110] = 0.0
        unstable, _, _ = sbsbench.flow_temporal_metrics(
            corrupted, corrupted, previous, previous, current, previous, min_support=0.1)
        self.assertGreater(support, 0.8)
        self.assertLess(stable, 2.0)
        self.assertGreater(unstable, stable + 100.0)

    def test_nearest_flow_warp_preserves_depth_steps(self):
        previous = np.zeros((16, 24), np.float32)
        previous[:, 8:] = 1.0
        u = np.full_like(previous, 3.0)
        v = np.zeros_like(previous)
        warped, valid = sbsbench.warp_previous_nearest_with_flow(previous, u, v)
        self.assertTrue(valid[:, 3:].all())
        self.assertEqual(set(np.unique(warped)), {0.0, 1.0})
        self.assertTrue((warped[:, 11:] == 1.0).all())

    def test_flow_aware_ema_tracks_translated_depth_edge(self):
        height, width = 24, 40
        previous = np.zeros((height, width), np.float32)
        previous[:, 12:] = 1.0
        current = np.zeros_like(previous)
        current[:, 15:] = 1.0
        flow = np.zeros((height, width, 2), np.float32)
        flow[..., 0] = 3.0
        valid = np.ones((height, width), bool)
        filtered, reliable, _ = prepare_flow_ema_reference.flow_aware_ema(
            current, previous, previous, current, flow, valid,
            0.5, 0.05, 0.02, 0.25)
        self.assertGreater(float(reliable.mean()), 0.85)
        self.assertTrue(np.array_equal(filtered, current))

    def test_depth_confidence_ignores_flat_depth(self):
        source = np.tile(np.linspace(0.0, 1.0, 96, dtype=np.float32), (48, 1))
        depth = np.full((48, 96), 0.5, np.float32)
        result = audit_depth_confidence.depth_confidence_map(depth, source)
        self.assertFalse(result["band"].any())
        self.assertTrue(np.all(result["risk"] == 0.0))
        self.assertTrue(np.all(result["confidence"] == 1.0))

    def test_depth_confidence_prefers_sharp_aligned_edges(self):
        source = np.zeros((48, 96), np.float32)
        source[:, 48:] = 1.0
        sharp = np.zeros_like(source)
        sharp[:, 48:] = 1.0
        shifted = np.zeros_like(source)
        shifted[:, 56:] = 1.0
        soft = np.zeros_like(source)
        soft[:, 44:53] = np.linspace(0.0, 1.0, 9, dtype=np.float32)
        soft[:, 53:] = 1.0
        sharp_result = audit_depth_confidence.depth_confidence_map(sharp, source)
        shifted_result = audit_depth_confidence.depth_confidence_map(shifted, source)
        soft_result = audit_depth_confidence.depth_confidence_map(soft, source)
        sharp_risk = sharp_result["model_risk"]
        shifted_risk = shifted_result["model_risk"]
        soft_risk = soft_result["model_risk"]
        self.assertLess(float(sharp_risk.max()), 0.1)
        self.assertGreater(float(shifted_risk.max()), 0.6)
        self.assertGreater(float(soft_risk.max()), float(sharp_risk.max()) + 0.2)
        self.assertGreater(float(sharp_result["warp_risk"].max()), 0.5)

    def test_depth_confidence_detects_flow_compensated_temporal_change(self):
        rng = np.random.default_rng(91)
        source = rng.random((64, 128), dtype=np.float32)
        previous = np.zeros_like(source)
        previous[:, 48:] = 1.0
        current = previous.copy()
        current[16:48, 48:] = 0.25
        stable = audit_depth_confidence.depth_confidence_map(
            previous, source, previous_depth=previous, previous_src=source)
        changed = audit_depth_confidence.depth_confidence_map(
            current, source, previous_depth=previous, previous_src=source)
        valid = changed["band"] & changed["temporal_valid"]
        self.assertTrue(valid.any())
        self.assertLess(float(stable["temporal"].max()), 0.01)
        self.assertGreater(float(changed["temporal"][valid].max()), 0.9)

    def test_confidence_audit_auc_is_tie_aware(self):
        labels = np.array([False, True, False, True])
        self.assertEqual(
            audit_depth_confidence.rank_auc(np.array([0.0, 1.0, 0.0, 1.0]), labels), 1.0)
        self.assertEqual(
            audit_depth_confidence.rank_auc(np.ones(4), labels), 0.5)
        self.assertIsNone(
            audit_depth_confidence.rank_auc(np.arange(4), np.zeros(4, bool)))

    def test_confidence_audit_rejects_tiny_pixel_classes(self):
        risk = np.zeros((16, 16), np.float32)
        risk[:, 8:] = 1.0
        confidence = {"risk": risk, "band": np.ones_like(risk, bool)}
        severity = np.zeros_like(risk)
        severity[:2, :8] = 2.0  # only 16 artifact pixels despite perfect ranking
        row, _, _, _ = audit_depth_confidence.validation_row(
            confidence, severity, np.ones_like(risk, bool))
        self.assertEqual(row["artifact_positive_px"], 16)
        self.assertIsNone(row["artifact_auc"])

    def test_confidence_audit_fails_closed_when_gt_evidence_is_missing(self):
        rows = [{"artifact_auc": 0.8, "artifact_capture_pct": 90.0} for _ in range(4)]
        stats = audit_depth_confidence.calibration_decision(rows, 4, 4)
        self.assertTrue(stats["warp_screening_validated"])
        self.assertFalse(stats["model_boundary_validated"])
        self.assertEqual(stats["gt_auc_frames"], 0)
        rows[0]["gt_bad_edge_auc"] = 0.7
        rows[1]["gt_bad_edge_auc"] = 0.6
        stats = audit_depth_confidence.calibration_decision(rows, 4, 4)
        self.assertTrue(stats["model_boundary_validated"])

    def test_confidence_audit_allows_flat_gt_without_boundary_auc(self):
        rows = [{"artifact_auc": 0.8, "artifact_capture_pct": 90.0} for _ in range(4)]
        stats = audit_depth_confidence.calibration_decision(rows, 0, 4)
        self.assertTrue(stats["warp_screening_validated"])
        self.assertIsNone(stats["model_boundary_validated"])
        self.assertEqual(stats["gt_frames_available"], 4)
        self.assertEqual(stats["gt_frames_eligible"], 0)

    def test_confidence_audit_rejects_frame_identity_drift(self):
        with self.assertRaisesRegex(ValueError, "missing=\\[2\\], extra=\\[3\\]"):
            audit_depth_confidence.require_frame_ids("depth", [1, 2], [1, 3])

    def test_exact_forward_flow_temporal_metric_compensates_motion(self):
        rng = np.random.default_rng(71)
        previous = np.round(rng.random((96, 160), dtype=np.float32) * 255.0) / 255.0
        current = np.zeros_like(previous)
        current[:, 5:] = previous[:, :-5]
        flow = np.zeros((96, 160, 2), np.float32)
        flow[..., 0] = 5.0
        valid = np.ones((96, 160), bool)
        valid[:, -5:] = False
        stable, _, support = sbsbench.flow_temporal_metrics(
            current, current, previous, previous, current, previous, min_support=0.1,
            reference_flow=flow, reference_valid=valid)
        self.assertGreater(support, 0.8)
        self.assertLess(stable, 2.0)

    def test_npy_metric_depth_preserves_native_values(self):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as fh:
            path = fh.name
        try:
            expected = np.array([[0.25, 4.0], [12.5, 200.0]], np.float32)
            np.save(path, expected)
            np.testing.assert_array_equal(sbsbench.load_depth(path), expected)
        finally:
            os.unlink(path)

    def test_public_dataset_timestamp_association_is_nearest_and_unique(self):
        rgb = [(0.00, "r0"), (0.10, "r1"), (0.20, "r2")]
        depth = [(0.009, "d0"), (0.105, "d1"), (0.35, "far")]
        pairs = prepare_public_datasets.associate_timestamps(rgb, depth, 0.03)
        self.assertEqual([(p[1], p[3]) for p in pairs], [("r0", "d0"), ("r1", "d1")])

    def test_suite_defaults_keep_core_and_extended_baselines_separate(self):
        core_clips, core_baselines = run_eval.suite_defaults("core")
        extended_clips, extended_baselines = run_eval.suite_defaults("extended")
        self.assertTrue(core_clips.endswith(os.path.join("sbsbench", "clips")))
        self.assertTrue(core_baselines.endswith(os.path.join("sbsbench", "baselines")))
        self.assertIn(os.path.join("prepared", "extended-v3"), extended_clips)
        self.assertTrue(extended_baselines.endswith("baselines_extended"))

    def test_rescore_uses_canonical_metric_contract_hash(self):
        artifact_hash = "a" * 16
        current_hash = "c" * 16
        data = {"meta": {
            "eval_schema": run_eval.EVAL_SCHEMA,
            "clip_set_sha1": {"shot": "original"},
            "metric_sha256": artifact_hash,
        }, "clips": {"shot": {"meta": {"metric_sha256": artifact_hash}}}}
        with mock.patch.object(run_eval, "metric_contract_sha", return_value=current_hash):
            rescore_run.refresh_contract_metadata(data)
        self.assertEqual(data["meta"]["metric_sha256"], current_hash)
        self.assertEqual(data["meta"]["artifact_metric_sha256"], artifact_hash)
        self.assertEqual(data["clips"]["shot"]["meta"]["metric_sha256"], current_hash)
        self.assertEqual(
            data["clips"]["shot"]["meta"]["artifact_metric_sha256"], artifact_hash)
        self.assertEqual(data["meta"]["eval_schema"], run_eval.EVAL_SCHEMA)
        self.assertEqual(data["meta"]["clip_set_sha1"], {"shot": "original"})

    def _current_rescore_fixture(self, root):
        clips_root = os.path.join(root, "clips")
        run_dir = os.path.join(root, "run")
        source_clip = os.path.join(clips_root, "shot")
        run_clip = os.path.join(run_dir, "shot")
        os.makedirs(source_clip)
        os.makedirs(run_clip)
        for frame_id in (0, 1):
            Image.fromarray(np.full((32, 64, 3), 40 + frame_id, np.uint8)).save(
                os.path.join(source_clip, f"frame_{frame_id:05d}.png")
            )
            for name, extension in (
                    ("sbs", "png"), ("depth", "png"), ("raw", "f32"),
                    ("warp_mask", "png"), ("warp_disparity", "f32"),
                    ("warp_unclamped_disparity", "f32")):
                open(os.path.join(run_clip, f"{name}_{frame_id:05d}.{extension}"),
                     "wb").close()
        policy_hash = "a" * 64
        contract = {
            "schema": run_eval.HARNESS_SCHEMA,
            "artifact_mode": "full",
            "source_width": 64,
            "source_height": 32,
            "model_input_width": 64,
            "model_input_height": 28,
            "eye_width": 64,
            "eye_height": 32,
            "color_mode": "sdr-srgb-8bit",
            "metric_preview_encoding": "native-srgb-v1",
            "disparity_raster_width": 64,
            "disparity_raster_height": 32,
            "output_interval": 1,
            "output_gt_right_only": False,
            "output_selection_mode": "interval",
            "label_frame_ids": [],
            "output_selected_frame_ids": [0, 1],
            "output_label_frames_sha256": "",
            "policy_warp_source_sha256": policy_hash,
            "metric_sha256": "b" * 16,
            "artistic_policy_consumed": False,
            "artistic_policy_authorization": "none",
            "model_onnx_sha256": "",
            "policy_metadata_sha256": "",
            "deployment_geometry_allowlist_sha256": "",
            "content_scale_x": 1.0,
            "content_scale_y": 1.0,
            "artistic_full_clamp_abs": 0.142,
            "warp_mask": {"red": "forward_disocclusion_before_fill"},
            "warp_disparity": rescore_run.EXACT_DISPARITY_SEMANTICS,
            "warp_unclamped_disparity": rescore_run.UNCLAMPED_DISPARITY_SEMANTICS,
            "artistic_disparity_contract": rescore_run.ARTISTIC_DISPARITY_CONTRACT,
        }
        with open(os.path.join(run_clip, "contract.json"), "w", encoding="utf-8") as fh:
            json.dump(contract, fh)
        clip_hash = run_eval.sha1_dir(source_clip)
        data = {
            "meta": {
                "eval_schema": run_eval.EVAL_SCHEMA,
                "clip_set_sha1": {"shot": clip_hash},
                "output_interval": 1,
                "output_gt_right_only": False,
                "output_selection_mode": "interval",
                "policy_warp_source_sha256": policy_hash,
                "metric_sha256": "b" * 16,
                "artistic_policy_consumed": False,
                "artistic_policy_authorization": "none",
                "model_onnx_sha256": "",
                "policy_metadata_sha256": "",
                "deployment_geometry_allowlist_sha256": "",
            },
            "clips": {"shot": {"meta": {
                "clip_sha1": clip_hash,
                "metric_sha256": "b" * 16,
                "policy_warp_source_sha256": policy_hash,
                "artistic_policy_consumed": False,
                "artistic_policy_authorization": "none",
                "model_onnx_sha256": "",
                "policy_metadata_sha256": "",
                "deployment_geometry_allowlist_sha256": "",
                "output_selection_mode": "interval",
                "label_frame_ids": [],
                "output_selected_frame_ids": [0, 1],
                "output_label_frames_sha256": "",
                "source_width": 64,
                "source_height": 32,
                "model_input_width": 64,
                "model_input_height": 28,
                "eye_width": 64,
                "eye_height": 32,
                "disparity_raster_width": 64,
                "disparity_raster_height": 32,
                "content_scale_x": 1.0,
                "content_scale_y": 1.0,
                "artistic_full_clamp_abs": 0.142,
                "color_mode": "sdr-srgb-8bit",
                "metric_preview_encoding": "native-srgb-v1",
            }}},
        }
        return data, run_dir, clips_root

    def test_rescore_refuses_to_upgrade_old_eval_schema(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            data["meta"]["eval_schema"] = run_eval.EVAL_SCHEMA - 1
            with self.assertRaisesRegex(RuntimeError, "fresh GPU run"):
                rescore_run.validate_current_artifact_contract(
                    data, run_dir, clips_root)

    def test_rescore_rejects_schema25_preview_artifacts_as_stale(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            contract_path = os.path.join(run_dir, "shot", "contract.json")
            with open(contract_path, encoding="utf-8") as stream:
                contract = json.load(stream)
            contract["schema"] = 25
            contract.pop("metric_preview_encoding", None)
            with open(contract_path, "w", encoding="utf-8") as stream:
                json.dump(contract, stream)
            with self.assertRaisesRegex(RuntimeError, "schema.*28.*25"):
                rescore_run.validate_current_artifact_contract(
                    data, run_dir, clips_root
                )

    def test_rescore_requires_exact_artifact_identities(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            os.remove(os.path.join(
                run_dir, "shot", "warp_unclamped_disparity_00001.f32"))
            with self.assertRaisesRegex(RuntimeError, "exact artifact identities"):
                rescore_run.validate_current_artifact_contract(
                    data, run_dir, clips_root)

    def test_rescore_requires_authenticated_output_selection(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            del data["clips"]["shot"]["meta"]["output_selected_frame_ids"]
            with self.assertRaisesRegex(RuntimeError, "authenticated output selection"):
                rescore_run.validate_current_artifact_contract(
                    data, run_dir, clips_root)

    def test_rescore_requires_matching_policy_source_hash(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            data["clips"]["shot"]["meta"]["policy_warp_source_sha256"] = "b" * 64
            with self.assertRaisesRegex(RuntimeError, "clip policy-warp source hash"):
                rescore_run.validate_current_artifact_contract(
                    data, run_dir, clips_root)

    def test_rescore_requires_original_clip_source_identity(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            Image.fromarray(np.full((32, 64, 3), 200, np.uint8)).save(
                os.path.join(clips_root, "shot", "frame_00000.png")
            )
            with self.assertRaisesRegex(RuntimeError, "differs from the original GPU run"):
                rescore_run.validate_current_artifact_contract(
                    data, run_dir, clips_root)

    def test_rescore_requires_recorded_identity_for_every_clip(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            del data["meta"]["clip_set_sha1"]
            with self.assertRaisesRegex(RuntimeError, "clip_set_sha1"):
                rescore_run.validate_current_artifact_contract(
                    data, run_dir, clips_root)

    def test_rescore_rejects_non_boolean_expected_flat(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            data["clips"]["shot"]["meta"]["expected_flat"] = "false"
            with self.assertRaisesRegex(RuntimeError, "expected_flat must be a boolean"):
                rescore_run.validate_current_artifact_contract(
                    data, run_dir, clips_root)

    def test_rescore_rejects_unsafe_clip_paths(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            data["clips"]["../shot"] = data["clips"].pop("shot")
            data["meta"]["clip_set_sha1"] = {
                "../shot": data["meta"]["clip_set_sha1"].pop("shot")
            }
            with self.assertRaisesRegex(RuntimeError, "invalid source clip selection"):
                rescore_run.validate_current_artifact_contract(
                    data, run_dir, clips_root)

    def test_rescore_requires_source_dimensions_from_harness_contract(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            contract_path = os.path.join(run_dir, "shot", "contract.json")
            with open(contract_path, encoding="utf-8") as fh:
                contract = json.load(fh)
            contract["source_width"] = 63
            with open(contract_path, "w", encoding="utf-8") as fh:
                json.dump(contract, fh)
            with self.assertRaisesRegex(RuntimeError, "dimensions.*harness contract"):
                rescore_run.validate_current_artifact_contract(
                    data, run_dir, clips_root)

    def test_rescore_accepts_current_exact_artifact_contract(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            self.assertEqual(
                rescore_run.validate_current_artifact_contract(
                    data, run_dir, clips_root),
                "b" * 16,
            )

    def test_rescore_keeps_dual_metric_provenance_across_repeat_validation(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            artifact_hash = rescore_run.validate_current_artifact_contract(
                data, run_dir, clips_root)
            current_hash = "c" * 16
            with mock.patch.object(
                    run_eval, "metric_contract_sha", return_value=current_hash):
                rescore_run.refresh_contract_metadata(data, artifact_hash)
            self.assertEqual(data["meta"]["artifact_metric_sha256"], "b" * 16)
            self.assertEqual(data["meta"]["metric_sha256"], current_hash)
            self.assertEqual(
                data["clips"]["shot"]["meta"]["artifact_metric_sha256"], "b" * 16)
            self.assertEqual(
                data["clips"]["shot"]["meta"]["metric_sha256"], current_hash)
            self.assertEqual(
                rescore_run.validate_current_artifact_contract(
                    data, run_dir, clips_root),
                "b" * 16,
            )

    def test_rescore_authenticates_contract_against_preserved_artifact_metric_hash(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            artifact_hash = rescore_run.validate_current_artifact_contract(
                data, run_dir, clips_root)
            with mock.patch.object(
                    run_eval, "metric_contract_sha", return_value="c" * 16):
                rescore_run.refresh_contract_metadata(data, artifact_hash)
            contract_path = os.path.join(run_dir, "shot", "contract.json")
            with open(contract_path, encoding="utf-8") as fh:
                contract = json.load(fh)
            contract["metric_sha256"] = "d" * 16
            with open(contract_path, "w", encoding="utf-8") as fh:
                json.dump(contract, fh)
            with self.assertRaisesRegex(RuntimeError, "artifact metric hash differs"):
                rescore_run.validate_current_artifact_contract(
                    data, run_dir, clips_root)

    def test_rescore_migrates_legacy_half_updated_metric_hash(self):
        with tempfile.TemporaryDirectory() as root:
            data, run_dir, clips_root = self._current_rescore_fixture(root)
            data["meta"]["metric_sha256"] = "c" * 16
            artifact_hash = rescore_run.validate_current_artifact_contract(
                data, run_dir, clips_root)
            self.assertEqual(artifact_hash, "b" * 16)
            with mock.patch.object(
                    run_eval, "metric_contract_sha", return_value="d" * 16):
                rescore_run.refresh_contract_metadata(data, artifact_hash)
            self.assertEqual(data["meta"]["artifact_metric_sha256"], "b" * 16)
            self.assertEqual(data["meta"]["metric_sha256"], "d" * 16)

    def test_sintel_adapter_preserves_left_and_rendered_right_frames(self):
        with tempfile.TemporaryDirectory() as root:
            archive = os.path.join(root, "sintel.zip")
            with zipfile.ZipFile(archive, "w") as zf:
                for i in range(3):
                    for eye, value in (("left", 40 + i), ("right", 80 + i)):
                        zf.writestr(f"training/final_{eye}/demo/frame_{i + 1:04d}.png",
                                    self.png_bytes(value))
                    zf.writestr(f"training/disparities/demo/frame_{i + 1:04d}.png",
                                self.png_bytes(10 + i))
            out = os.path.join(root, "out")
            os.makedirs(out)
            clip = {"archives": ["stereo"], "sequence": "demo", "pass": "final",
                    "start": 0, "stride": 1, "count": 2}
            rows = prepare_public_datasets.prepare_sintel(
                "demo", clip, {}, {"stereo": archive}, out, "test")
            self.assertEqual(len(rows), 2)
            self.assertTrue(os.path.exists(os.path.join(out, "frame_00000.png")))
            self.assertTrue(os.path.exists(os.path.join(out, "gt_right", "frame_00001.png")))
            self.assertTrue(os.path.exists(os.path.join(out, "gt_depth", "frame_00001.npy")))

    def test_artistic_sintel_suite_extracts_complete_sequence_holdouts(self):
        with tempfile.TemporaryDirectory() as root:
            archive = os.path.join(root, "sintel.zip")
            with zipfile.ZipFile(archive, "w") as zf:
                for sequence in ("alley_1", "ambush_2"):
                    for frame in (1, 2):
                        for eye, value in (("left", 40), ("right", 80)):
                            zf.writestr(
                                f"training/final_{eye}/{sequence}/frame_{frame:04d}.png",
                                self.png_bytes(value + frame),
                            )
            output = os.path.join(root, "suite")
            manifest = prepare_sintel_artistic_training.prepare(
                Path(archive), Path(output),
                "final",
            )
            self.assertEqual(manifest["frame_count"], 4)
            self.assertEqual(manifest["training_frames"], 2)
            self.assertEqual(manifest["validation_frames"], 2)
            self.assertTrue(os.path.exists(
                os.path.join(output, "sintel_alley_1", "frame_00001.png")
            ))
            self.assertTrue(os.path.exists(
                os.path.join(output, "sintel_ambush_2", "gt_right", "frame_00001.png")
            ))

    def test_artistic_spring_suite_pairs_archives_and_holds_out_sequences(self):
        with tempfile.TemporaryDirectory() as root:
            archives = []
            for eye, value in (("left", 40), ("right", 80)):
                archive = os.path.join(root, f"spring-{eye}.zip")
                with zipfile.ZipFile(archive, "w") as zf:
                    for sequence in ("0003", "0042"):
                        for frame in (1, 2):
                            zf.writestr(
                                f"spring/test/{sequence}/frame_{eye}/"
                                f"frame_{eye}_{frame:04d}.png",
                                self.png_bytes(value + frame),
                            )
                archives.append(Path(archive))
            output = Path(root) / "suite"
            manifest = prepare_spring_artistic_training.prepare(
                archives[0], archives[1], output,
                holdout_sequences=("0042",),
            )
            self.assertEqual(manifest["frame_count"], 4)
            self.assertEqual(manifest["training_frames"], 2)
            self.assertEqual(manifest["validation_frames"], 2)
            self.assertTrue((output / "spring_0003" / "frame_00001.png").is_file())
            self.assertTrue(
                (output / "spring_0042" / "gt_right" / "frame_00001.png").is_file()
            )

    def test_artistic_middlebury_suite_uses_official_split(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "middlebury.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                for split, scene in (("trainingF", "Train"), ("testF", "Test")):
                    for image, value in (("im0.png", 40), ("im1.png", 80)):
                        zf.writestr(
                            f"MiddEval3/{split}/{scene}/{image}", self.png_bytes(value)
                        )
            output = Path(root) / "suite"
            manifest = prepare_static_stereo_training.prepare_middlebury(
                archive, output
            )
            self.assertEqual(manifest["pair_count"], 2)
            self.assertEqual(manifest["training_pairs"], 0)
            self.assertEqual(manifest["validation_pairs"], 2)
            self.assertTrue(
                (output / "middlebury_test" / "gt_right" / "frame_00000.png")
                .is_file()
            )

    def test_artistic_eth3d_suite_holds_out_complete_scenes(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "eth3d"
            for scene in ("delivery_area_1s", "forest_1s"):
                scene_root = source / scene
                scene_root.mkdir(parents=True)
                (scene_root / "im0.png").write_bytes(self.png_bytes(40))
                (scene_root / "im1.png").write_bytes(self.png_bytes(80))
            output = Path(root) / "suite"
            manifest = prepare_static_stereo_training.prepare_eth3d(source, output)
            self.assertEqual(manifest["pair_count"], 2)
            self.assertEqual(manifest["training_pairs"], 0)
            self.assertEqual(manifest["validation_pairs"], 2)

    def test_artistic_depth_run_requires_exact_depth_only_contract(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            output = Path(root) / "output"
            source.mkdir()
            output.mkdir()
            for frame in range(2):
                (source / f"frame_{frame:05d}.png").write_bytes(self.png_bytes(frame))
                (output / f"depth_{frame:05d}.png").write_bytes(self.png_bytes(frame, "L"))
                (output / f"baseline_disparity_{frame:05d}.f32").write_bytes(
                    b"\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00"
                )
                (output / f"baseline_unclamped_disparity_{frame:05d}.f32").write_bytes(
                    b"\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00"
                )
            selection = generate_artistic_depth_run.output_selection(source)
            (output / "contract.json").write_text(json.dumps({
                "schema": generate_artistic_depth_run.harness_contract.HARNESS_SCHEMA,
                "model": "depth_anything_v2_fp16",
                "source_width": 1,
                "source_height": 1,
                "model_input_width": 14,
                "model_input_height": 14,
                "eye_width": 1,
                "eye_height": 1,
                "color_mode": "sdr-srgb-8bit",
                "hdr_source_kind": "native-sdr",
                "metric_preview_encoding": "native-srgb-v1",
                "content_scale_x": 1.0,
                "content_scale_y": 1.0,
                "disparity_raster_width": 1,
                "disparity_raster_height": 1,
                "policy_warp_source_sha256": "0" * 64,
                "metric_sha256": "1" * 16,
                "artistic_full_clamp_abs": 0.142,
                "artifact_mode": "depth+baseline-disparity",
                "depth_step": "current-once",
                "artistic_policy": False,
                "artistic_policy_consumed": False,
                "artistic_policy_authorization": "none",
                "model_onnx_sha256": "",
                "policy_metadata_sha256": "",
                "deployment_geometry_allowlist_sha256": "",
                "artistic_scale_override": 0.0,
                "warp_disparity":
                    "exact_clamped_full_binocular_normalized_at_output_eye_raster_zero_bars",
                "warp_unclamped_disparity":
                    "unclamped_full_binocular_normalized_at_artistic_scale_1_"
                    "output_eye_raster_zero_bars",
                "artistic_disparity_contract":
                    "clamp(raw_baseline_times_scale_to_plus_or_minus_0.142_"
                    "times_aspect_scale_times_content_scale_x)",
                "output_selection_mode": selection["mode"],
                "label_frame_ids": selection["label_frame_ids"],
                "output_selected_frame_ids": selection["output_frame_ids"],
                "output_label_frames_sha256": selection["label_frames_sha256"],
            }), encoding="utf-8")
            source_identity = {
                "source_identity_method":
                    generate_artistic_depth_run.SOURCE_IDENTITY_FINGERPRINT,
                "source_identity_value":
                    generate_artistic_depth_run.source_fingerprint(source),
            }
            identity = generate_artistic_depth_run.generation_identity(
                source_identity, selection, "e" * 64, "c" * 16,
                "depth_anything_v2_fp16"
            )
            (output / "generation_identity.json").write_text(
                json.dumps(identity), encoding="utf-8"
            )
            self.assertTrue(generate_artistic_depth_run.valid_completed_clip(
                source, output, "depth_anything_v2_fp16"
            ))
            (output / "depth_00001.png").unlink()
            self.assertFalse(generate_artistic_depth_run.valid_completed_clip(
                source, output, "depth_anything_v2_fp16"
            ))

    def test_spring_adapter_range_selects_matching_stereo_frames(self):
        with tempfile.TemporaryDirectory() as root:
            archives = {}
            for side, value in (("left", 40), ("right", 80)):
                path = os.path.join(root, side + ".zip")
                with zipfile.ZipFile(path, "w") as zf:
                    for i in range(3):
                        zf.writestr(f"spring/test/0003/frame_{side}/frame_{side}_{i + 1:04d}.png",
                                    self.png_bytes(value + i))
                archives["test_" + side] = {"url": path, "size": os.path.getsize(path)}
            out = os.path.join(root, "out")
            os.makedirs(out)
            clip = {"sequence": "0003", "start": 1, "stride": 1, "count": 2}

            def memory_reader(url, expected_size):
                with open(url, "rb") as archive_file:
                    return io.BytesIO(archive_file.read())

            with mock.patch.object(
                    prepare_public_datasets, "HTTPRangeReader",
                    side_effect=memory_reader):
                rows = prepare_public_datasets.prepare_spring(
                    "demo", clip, {}, archives, out, "test")
            self.assertEqual([row["dataset_frame"] for row in rows], [2, 3])
            self.assertTrue(os.path.exists(os.path.join(out, "frame_00000.png")))
            self.assertTrue(os.path.exists(os.path.join(out, "gt_right", "frame_00001.png")))

    def test_vkitti_adapter_selects_matching_rgb_and_depth(self):
        with tempfile.TemporaryDirectory() as root:
            archives = {}
            for modality in ("rgb", "depth"):
                path = os.path.join(root, modality + ".tar")
                archives[modality] = path
                with tarfile.open(path, "w") as tf:
                    for i in range(3):
                        suffix = f"rgb_{i:05d}.jpg" if modality == "rgb" else f"depth_{i:05d}.png"
                        folder = "rgb" if modality == "rgb" else "depth"
                        data = self.png_bytes(50 + i, "RGB" if modality == "rgb" else "L")
                        info = tarfile.TarInfo(
                            f"vkitti/{'Scene01'}/clone/frames/{folder}/Camera_0/{suffix}")
                        info.size = len(data)
                        tf.addfile(info, io.BytesIO(data))
            out = os.path.join(root, "out")
            os.makedirs(out)
            clip = {"scene": "Scene01", "variant": "clone", "camera": "Camera_0",
                    "start": 1, "stride": 1, "count": 2}
            rows = prepare_public_datasets.prepare_vkitti2(
                "demo", clip, {}, archives, out, "test")
            self.assertEqual([r["dataset_frame"] for r in rows], [1, 2])
            self.assertTrue(os.path.exists(os.path.join(out, "frame_00000.png")))
            self.assertTrue(os.path.exists(os.path.join(out, "gt_depth", "frame_00001.png")))


if __name__ == "__main__":
    unittest.main()
