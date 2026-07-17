#!/usr/bin/env python3

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import multiscale_batch as batch  # noqa: E402
import run_multiscale_eval as driver  # noqa: E402
import sbsbench  # noqa: E402


class MultiscaleBatchTests(unittest.TestCase):
    def test_measured_default_scale_concurrency(self):
        self.assertEqual(driver.DEFAULT_SCALE_SCORE_JOBS, 8)

    def _fixture(self, root, *, sparse=True):
        root = Path(root)
        scales = (1.0, 1.3)
        source_frame_ids = list(range(6))
        label_frame_ids = [2, 5] if sparse else []
        output_frame_ids = list(label_frame_ids) if sparse else source_frame_ids
        selection_mode = "label-frames" if sparse else "interval"
        label_sha256 = "3" * 64 if sparse else ""
        common = root / "common"
        common.mkdir(parents=True)
        for frame_id in output_frame_ids:
            (common / f"depth_{frame_id:05d}.png").write_bytes(
                f"depth-{frame_id}".encode()
            )
            (common / f"raw_{frame_id:05d}.f32").write_bytes(
                f"raw-{frame_id}".encode()
            )
        (common / "raw_shape.json").write_text("{}", encoding="utf-8")
        (common / "runtime_scene_evidence.json").write_text(json.dumps({
            "schema": 1,
            "contract": "apollo-subject-state-runtime-scenes-v1",
            "evidence_source":
                "SubjectState[0].y after completed depth postprocess",
            "cut_rule": "prior_scene_age_gte_7_and_current_scene_age_eq_0",
            "cadence": "completed-depth-frames-only",
            "completion_sequence_contract": (
                "exact for this synchronous harness sequence; live busy-drop "
                "cadence is not replayed"
            ),
            "depth_reuse_interval": 1,
            "source_frame_ids": source_frame_ids,
            "completed_source_frame_ids": source_frame_ids,
            "completed_depth_frame_count": len(source_frame_ids),
            "frames": [{
                "source_frame_ordinal": ordinal,
                "source_frame_id": frame_id,
                "runtime_scene_id": 0,
                "scene_age": 0.0,
                "subject_initialized": False,
                "hard_cut": False,
                "scene_start": ordinal == 0,
            } for ordinal, frame_id in enumerate(source_frame_ids)],
        }), encoding="utf-8")
        rows = []
        metric = "0123456789abcdef"
        for index, scale in enumerate(scales):
            slug = batch.scale_slug(scale)
            relative = f"scales/{slug}"
            directory = root / relative
            directory.mkdir(parents=True)
            for frame_id in output_frame_ids:
                for prefix, suffix in (
                        ("sbs_", ".png"), ("warp_mask_", ".png"),
                        ("warp_disparity_", ".f32"),
                        ("warp_unclamped_disparity_", ".f32")):
                    (directory / f"{prefix}{frame_id:05d}{suffix}").write_bytes(
                        f"{slug}-{prefix}-{frame_id}".encode()
                    )
            (directory / "contract.json").write_text(json.dumps({
                "multiscale_batch": True,
                "multiscale_batch_contract": batch.HARNESS_CONTRACT,
                "multiscale_scale_index": index,
                "multiscale_scale_float32_bits":
                    batch.scale_float32_bits(scale),
                "artistic_scale_override": scale,
                "output_selection_mode": selection_mode,
                "label_frame_ids": label_frame_ids,
                "output_selected_frame_ids": output_frame_ids,
                "output_label_frames_sha256": label_sha256,
                "metric_sha256": metric,
            }), encoding="utf-8")
            rows.append({
                "index": index,
                "scale": scale,
                "float32_bits": batch.scale_float32_bits(scale),
                "directory": relative,
            })
        (root / batch.HARNESS_MANIFEST).write_text(json.dumps({
            "schema": batch.HARNESS_SCHEMA,
            "contract": batch.HARNESS_CONTRACT,
            "scope": "offline-sbs-bench-only",
            "shipping_estimator_calls_per_source_frame": 1,
            "depth_state_cache": {
                "mode": "disabled",
                "key_sha256": "",
                "manifest_sha256": "",
                "boundary":
                    "completed-production-depth-state-before-warp-prefilter",
                "selected_state_frame_count": len(output_frame_ids),
                "runtime_scene_frame_count": len(source_frame_ids),
            },
            "scale_rows": rows,
            "common_directory": "common",
            "source_frame_ids": source_frame_ids,
            "label_frame_ids": label_frame_ids,
            "output_selected_frame_ids": output_frame_ids,
            "output_selection_mode": selection_mode,
            "output_label_frames_sha256": label_sha256,
            "source_frame_count": len(source_frame_ids),
            "output_frame_count_per_scale": len(output_frame_ids),
            "artifact_writer": {
                "contract": batch.ARTIFACT_WRITER_CONTRACT,
                "mode": "bounded-async-worker-owned-buffers",
                "d3d_readback_thread": "harness-main",
                "png_factory_scope": "per-worker-com-mta",
                "worker_count": 4,
                "queue_capacity": 8,
                "maximum_inflight_job_bound": 13,
                "submitted_jobs": len(scales) * len(output_frame_ids) * 2,
                "completed_jobs": len(scales) * len(output_frame_ids) * 2,
                "sbs_png_jobs": len(scales) * len(output_frame_ids),
                "mask_png_jobs": len(scales) * len(output_frame_ids),
                "deterministic_unique_output_paths": True,
                "drained_before_publication": True,
            },
        }), encoding="utf-8")
        identities = {
            "clip": "clip",
            "clip_sha1": "0123456789ab",
            "executable_sha256": "1" * 64,
            "conf_sha256": "2" * 16,
            "metric_sha256": metric,
        }
        return scales, identities

    def test_publish_and_validate_exact_selected_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scales, identities = self._fixture(root)
            payload = batch.publish(root, scales=scales, **identities)

            selected = batch.validate(
                root, scale=1.3, **identities
            )

            self.assertEqual(selected["manifest"], payload)
            self.assertEqual(selected["scale_root"].name, "s130")
            self.assertEqual(selected["common_root"].name, "common")
            self.assertEqual(payload["source_frame_ids"], list(range(6)))
            self.assertEqual(payload["label_frame_ids"], [2, 5])
            self.assertEqual(
                payload["output_selected_frame_ids"], [2, 5]
            )
            self.assertEqual(
                (root / batch.MANIFEST).read_bytes(),
                batch.canonical_bytes(payload),
            )

    def test_selected_artifact_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scales, identities = self._fixture(root)
            batch.publish(root, scales=scales, **identities)
            (root / "scales" / "s130" / "sbs_00000.png").write_bytes(
                b"tampered"
            )
            with self.assertRaisesRegex(ValueError, "artifact.*changed"):
                batch.validate(root, scale=1.3, **identities)

    def test_manifest_must_be_canonical_and_exact_scale_present(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scales, identities = self._fixture(root)
            batch.publish(root, scales=scales, **identities)
            manifest = root / batch.MANIFEST
            value = json.loads(manifest.read_text(encoding="utf-8"))
            manifest.write_text(json.dumps(value, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not canonical"):
                batch.validate(root, scale=1.0, **identities)
            manifest.write_bytes(batch.canonical_bytes(value))
            with self.assertRaisesRegex(ValueError, "lacks the requested"):
                batch.validate(root, scale=1.1, **identities)

    def test_full_output_mode_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scales, identities = self._fixture(root, sparse=False)
            payload = batch.publish(root, scales=scales, **identities)
            self.assertEqual(payload["label_frame_ids"], [])
            self.assertEqual(
                payload["source_frame_ids"],
                payload["output_selected_frame_ids"],
            )
            self.assertEqual(payload["output_selection_mode"], "interval")

    def test_sparse_common_artifacts_must_match_selected_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scales, identities = self._fixture(root)
            (root / "common" / "depth_00000.png").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "common artifacts"):
                batch.publish(root, scales=scales, **identities)

    def test_runtime_scene_evidence_must_cover_every_source_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scales, identities = self._fixture(root)
            scene_path = root / "common" / "runtime_scene_evidence.json"
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            scene["source_frame_ids"] = scene["source_frame_ids"][:-1]
            scene_path.write_text(json.dumps(scene), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime scene evidence"):
                batch.publish(root, scales=scales, **identities)

    def test_artifact_writer_must_be_bounded_and_fully_drained(self):
        for field, value, message in (
                ("maximum_inflight_job_bound", 12, "bound evidence"),
                ("completed_jobs", 15, "completion evidence"),
                ("drained_before_publication", False, "contract differs")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scales, identities = self._fixture(root)
                harness_path = root / batch.HARNESS_MANIFEST
                harness = json.loads(harness_path.read_text(encoding="utf-8"))
                harness["artifact_writer"][field] = value
                harness_path.write_text(json.dumps(harness), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    batch.publish(root, scales=scales, **identities)

    def test_sparse_outputs_require_exact_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scales, identities = self._fixture(root)
            harness_path = root / batch.HARNESS_MANIFEST
            harness = json.loads(harness_path.read_text(encoding="utf-8"))
            harness["output_selected_frame_ids"] = [2, 4, 5]
            harness_path.write_text(json.dumps(harness), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected output frame coverage"):
                batch.publish(root, scales=scales, **identities)

    def test_scale_contract_must_use_canonical_lattice(self):
        for invalid in (0.49, 1.501, 1.005, float("nan")):
            with self.assertRaises(ValueError):
                batch.scale_slug(invalid)
        self.assertEqual(batch.scale_slug(1.30), "s130")

    def test_sequence_scores_shared_depth_from_common_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scale = root / "scale"
            common = root / "common"
            frames = root / "frames"
            scale.mkdir()
            common.mkdir()
            frames.mkdir()
            eye = np.zeros((16, 16, 3), dtype=np.uint8)
            Image.fromarray(np.concatenate((eye, eye), axis=1)).save(
                scale / "sbs_00000.png"
            )
            Image.fromarray(eye).save(frames / "frame_00000.png")
            depth = np.tile(
                np.linspace(0, 65535, 16, dtype=np.uint16), (16, 1)
            )
            Image.fromarray(depth).save(common / "depth_00000.png")

            rows, aggregate = sbsbench.measure_sequence(
                scale, frames, common_artifact_dir=common
            )

            self.assertEqual(rows[0]["_frame_id"], 0)
            self.assertIn("depth_spread", rows[0])
            self.assertEqual(aggregate["_n"], 1)

    def test_driver_rejects_every_driver_owned_extra_spelling(self):
        for value in ("--frames", "--frames=alternate", "--depth-every",
                      "--artistic-scale-override=1.50",
                      "--output-label-frames"):
            with self.subTest(value=value), self.assertRaisesRegex(
                    RuntimeError, "driver-owned option"):
                driver._validated_extra([value])
        self.assertEqual(
            driver._validated_extra(["--eye-w", "1280", "--simulate-hdr"]),
            ["--eye-w", "1280", "--simulate-hdr"],
        )

    def test_driver_invalidates_stale_summary_before_output_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            temporary = Path(directory) / "summary.json.tmp"
            summary.write_text("old success", encoding="utf-8")
            temporary.write_text("old partial", encoding="utf-8")

            resolved = driver._invalidate_summary(summary)

            self.assertEqual(resolved, summary.resolve())
            self.assertFalse(summary.exists())
            self.assertFalse(temporary.exists())

    def test_driver_refuses_directory_summary_path(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "not a file"):
                driver._invalidate_summary(Path(directory))

    def test_success_summary_is_durable_before_batch_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch_root = root / "batch"
            batch_root.mkdir()
            (batch_root / "render.bin").write_bytes(b"render")
            summary = root / "summary.json"
            payload = {"schema": driver.SCHEMA, "contract": driver.CONTRACT}

            driver._publish_summary_then_cleanup(
                summary, payload, batch_root, "fixture"
            )

            self.assertEqual(summary.read_bytes(), driver.canonical_bytes(payload))
            self.assertFalse(batch_root.exists())

    def test_summary_publication_failure_retains_rendered_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch_root = root / "batch"
            batch_root.mkdir()
            (batch_root / "render.bin").write_bytes(b"render")
            summary = root / "summary.json"
            with mock.patch.object(
                    driver.os, "replace", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    driver._publish_summary_then_cleanup(
                        summary, {"complete": True}, batch_root, "fixture"
                    )
            self.assertTrue(batch_root.is_dir())
            self.assertFalse(summary.exists())

    def test_cleanup_failure_keeps_durable_success_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch_root = root / "batch"
            batch_root.mkdir()
            summary = root / "summary.json"
            payload = {"complete": True}
            with mock.patch.object(
                    driver.shutil, "rmtree",
                    side_effect=OSError("injected cleanup")):
                driver._publish_summary_then_cleanup(
                    summary, payload, batch_root, "fixture"
                )
            self.assertEqual(summary.read_bytes(), driver.canonical_bytes(payload))
            self.assertTrue(batch_root.is_dir())

    def test_render_identity_is_path_independent_and_semantic(self):
        with tempfile.TemporaryDirectory() as first_directory, \
                tempfile.TemporaryDirectory() as second_directory:
            roots = [Path(first_directory), Path(second_directory)]
            for root in roots:
                (root / "frame_00000.png").write_bytes(b"same-frame")
                (root / "label_frames.json").write_text(
                    '{"schema":1,"frame_ids":[0]}', encoding="utf-8"
                )
            contents = [
                driver._path_independent_clip_content_identity(
                    root, "clip", {"clip_hash_manifest": None}
                ) for root in roots
            ]
            self.assertEqual(contents[0], contents[1])
            selection = {
                "source_frame_ids": [0],
                "label_frame_ids": [0],
                "output_frame_ids": [0],
                "mode": "label-frames",
                "label_frames_sha256": "4" * 64,
            }
            common = dict(
                clip_content=contents[0], clip_sha1="0" * 12,
                executable_sha256="1" * 64, conf_sha256="2" * 16,
                metric_sha256="3" * 16, model="depth-model",
                output_selection=selection, scales=(1.0, 1.3),
                depth_state_identity_sha256="5" * 64,
            )
            first = driver._render_identity(extra=["--eye-w", "1280"], **common)
            same = driver._render_identity(extra=["--eye-w", "1280"], **common)
            changed = driver._render_identity(extra=["--eye-w", "960"], **common)
            runtime_changed = driver._render_identity(
                extra=["--eye-w", "1280"],
                **{**common, "depth_state_identity_sha256": "6" * 64},
            )

            self.assertEqual(first, same)
            self.assertNotEqual(
                first["render_identity_sha256"],
                changed["render_identity_sha256"],
            )
            self.assertNotEqual(
                first["render_identity_sha256"],
                runtime_changed["render_identity_sha256"],
            )
            serialized = json.dumps(first)
            self.assertNotIn(first_directory, serialized)
            self.assertNotIn(second_directory, serialized)

    def test_depth_input_reverification_refreshes_rows_and_runtime_receipt(self):
        expected = {"cache": "identity"}
        identity_args = {
            "clip_dir": Path("clip-root"),
            "source_content_rows": [{"path": "old"}],
        }
        refreshed = [{"path": "frame_00000.png", "size": 1, "sha256": "a" * 64}]
        with mock.patch.object(
                driver.run_eval, "revalidate_clip_hashes") as revalidate, \
                mock.patch.object(
                    driver, "_path_independent_clip_content_rows",
                    return_value=("frozen", refreshed),
                ), \
                mock.patch.object(
                    driver.depth_state_cache, "verify_runtime_snapshot",
                    return_value={"runtime": "identity"},
                ) as verify_runtime, \
                mock.patch.object(
                    driver.depth_state_cache, "identity",
                    return_value=expected,
                ) as make_identity:
            self.assertEqual(
                driver.reverify_depth_state_inputs(
                    expected, identity_args, {"snapshot": True},
                    Path("clips"), "clip", {"clip": "abc"},
                    {"clip_hash_manifest": None},
                ),
                expected,
            )
        revalidate.assert_called_once()
        verify_runtime.assert_called_once_with({"snapshot": True})
        self.assertEqual(
            make_identity.call_args.kwargs["source_content_rows"], refreshed,
        )

    def test_depth_input_reverification_fails_on_identity_change(self):
        with mock.patch.object(
                driver.run_eval, "revalidate_clip_hashes"), \
                mock.patch.object(
                    driver, "_path_independent_clip_content_rows",
                    return_value=("frozen", []),
                ), \
                mock.patch.object(
                    driver.depth_state_cache, "verify_runtime_snapshot",
                    return_value={"runtime": "identity"},
                ), \
                mock.patch.object(
                    driver.depth_state_cache, "identity",
                    return_value={"different": True},
                ):
            with self.assertRaisesRegex(RuntimeError, "inputs changed"):
                driver.reverify_depth_state_inputs(
                    {"expected": True},
                    {"clip_dir": Path("clip"), "source_content_rows": []},
                    {}, Path("clips"), "clip", {"clip": "abc"}, {},
                )

    def test_native_depth_admission_rehashes_sidecars(self):
        expected = {"cache": "identity"}
        identity_args = {
            "clip_dir": Path("clip-root"),
            "source_content_rows": [{"path": "old"}],
            "extra": ["--native-hdr-scrgb"],
        }
        refreshed = [{"path": "frame_00000.png", "size": 1,
                      "sha256": "a" * 64}]
        with mock.patch.object(
                driver.run_eval, "revalidate_clip_hashes"), \
                mock.patch.object(
                    driver, "_path_independent_clip_content_rows",
                    return_value=("frozen", refreshed),
                ), \
                mock.patch.object(
                    driver.depth_state_cache.native_hdr_capture,
                    "validate_clip", return_value={"authenticated": True},
                ) as validate_native, \
                mock.patch.object(
                    driver.depth_state_cache, "verify_runtime_snapshot",
                    return_value={"runtime": "identity"},
                ), \
                mock.patch.object(
                    driver.depth_state_cache, "identity",
                    return_value=expected,
                ):
            driver.reverify_depth_state_inputs(
                expected, identity_args, {}, Path("clips"), "clip",
                {"clip": "abc"}, {"clip_hash_manifest": None},
            )
        validate_native.assert_called_once_with(Path("clip-root"), full=True)

    def test_render_receipt_binds_authenticated_batch_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scales, identities = self._fixture(root)
            batch.publish(root, scales=scales, **identities)
            identity = {
                "schema": driver.RENDER_IDENTITY_SCHEMA,
                "contract": driver.RENDER_IDENTITY_CONTRACT,
                "render_identity_sha256": "a" * 64,
                "inputs": {"fixture": True},
            }
            receipt = driver._write_render_receipt(root, identity)

            self.assertEqual(
                driver._validate_render_receipt(root, identity), receipt
            )
            manifest = root / batch.MANIFEST
            manifest.write_bytes(manifest.read_bytes() + b" ")
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                driver._validate_render_receipt(root, identity)

    def test_driver_selected_mode_uses_targets_only(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory)
            for frame_id in range(6):
                (clip / f"frame_{frame_id:05d}.png").write_bytes(b"frame")
            (clip / "label_frames.json").write_text(json.dumps({
                "schema": 1, "frame_ids": [2, 5],
            }), encoding="utf-8")

            selection = driver._resolve_output_selection(clip, True)

            self.assertEqual(selection["label_frame_ids"], [2, 5])
            self.assertEqual(selection["output_frame_ids"], [2, 5])
            self.assertEqual(
                driver._harness_selection_args(True),
                ["--output-label-frames"],
            )
            self.assertEqual(
                driver._frame_gate_args(True),
                ["--publish-selected-frame-gates"],
            )
            self.assertEqual(
                driver._frame_gate_args(False), ["--publish-frame-gates"]
            )

    def test_scale_score_child_is_single_worker(self):
        command = driver._scale_score_command(
            build_dir=Path("build"),
            conf=Path("bench.conf"),
            clips_root=Path("clips"),
            clip="clip",
            label="run-s100",
            batch_group_root=Path("batch"),
            executable_sha256="1" * 64,
            extra=["--eye-w", "1280"],
            selected_label_frames=True,
            scale=1.0,
        )

        worker_index = command.index("--score-workers")
        self.assertEqual(command[worker_index + 1], "1")
        self.assertIn("--publish-selected-frame-gates", command)

    def test_scale_scoring_is_bounded_and_returns_lattice_order(self):
        jobs = [{
            "scale_index": index,
            "scale": 1.0 + 0.02 * index,
            "command": ["score", str(index)],
        } for index in range(4)]
        lock = threading.Lock()
        release_first = threading.Event()
        second_finished = threading.Event()
        third_started = threading.Event()
        active = 0
        maximum_active = 0
        completion_order = []

        def score(job, _cwd):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                if job["scale_index"] == 0:
                    self.assertTrue(release_first.wait(1.0))
                elif job["scale_index"] == 1:
                    second_finished.set()
                elif job["scale_index"] == 2:
                    self.assertTrue(second_finished.wait(1.0))
                    third_started.set()
                    release_first.set()
                completion_order.append(job["scale_index"])
                return job
            finally:
                with lock:
                    active -= 1

        with mock.patch.object(driver, "_score_scale_child", side_effect=score):
            result = driver._score_scales(jobs, 2, Path("."))

        self.assertTrue(third_started.is_set())
        self.assertEqual(maximum_active, 2)
        self.assertNotEqual(completion_order, list(range(4)))
        self.assertEqual(
            [job["scale_index"] for job in result], list(range(4))
        )

    def test_scale_scoring_failure_waits_running_worker(self):
        jobs = [{
            "scale_index": index,
            "scale": 1.0 + 0.02 * index,
            "command": ["score", str(index)],
        } for index in range(3)]
        second_started = threading.Event()
        second_finished = threading.Event()

        def score(job, _cwd):
            if job["scale_index"] == 0:
                self.assertTrue(second_started.wait(1.0))
                raise RuntimeError("scale failed")
            if job["scale_index"] == 1:
                second_started.set()
                time.sleep(0.05)
                second_finished.set()
            return job

        with mock.patch.object(driver, "_score_scale_child", side_effect=score):
            with self.assertRaisesRegex(RuntimeError, "scale failed"):
                driver._score_scales(jobs, 2, Path("."))

        self.assertTrue(second_finished.is_set())

    def test_visual_evidence_uses_numeric_ids_and_replaces_stale_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            clip = root / "clip"
            destination = root / "destination"
            source.mkdir()
            clip.mkdir()
            destination.mkdir()
            frame_id = 123456
            (clip / "label_frames.json").write_text(json.dumps({
                "schema": 1, "frame_ids": [frame_id],
            }), encoding="utf-8")
            for prefix, suffix in (
                    ("sbs_", ".png"), ("warp_mask_", ".png"),
                    ("warp_disparity_", ".f32")):
                (source / f"{prefix}{frame_id:06d}{suffix}").write_bytes(
                    prefix.encode("ascii")
                )
            (destination / "stale.png").write_bytes(b"stale")

            driver._copy_visual_evidence(
                source, destination, clip, 1.0
            )

            self.assertFalse((destination / "stale.png").exists())
            evidence = json.loads(
                (destination / "visual_evidence.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(evidence["frame_ids"], [frame_id])
            self.assertEqual(len(evidence["files"]), 3)


if __name__ == "__main__":
    unittest.main()
