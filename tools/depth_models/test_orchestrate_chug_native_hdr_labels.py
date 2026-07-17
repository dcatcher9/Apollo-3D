import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import orchestrate_chug_native_hdr_labels as orchestration  # noqa: E402


class NativeHdrLabelOrchestrationTests(unittest.TestCase):
    def make_publication(self, root):
        dataset_root = root / "native"
        dataset_root.mkdir()
        conversion = {"schema": 1, "contract": "test-native-conversion"}
        conversion_sha = orchestration.native_hdr_capture.canonical_sha256(
            conversion
        )
        conversion_path = dataset_root / "conversion_contract.json"
        conversion_path.write_text(json.dumps(conversion), encoding="utf-8")
        datasets = {}
        retention = {}
        for split in ("training", "development"):
            split_root = dataset_root / split
            split_root.mkdir()
            sequences = []
            clips = []
            groups = []
            source_count = orchestration.EXPECTED_SOURCE_CLIPS[split]
            for source_index in range(source_count):
                video_id = f"{split}-video-{source_index:02d}"
                capture_group = f"{split}-capture-{source_index:02d}"
                groups.append(capture_group)
                for window_index in range(
                        orchestration.chug_prepare.LABELS_PER_CLIP):
                    clip = orchestration.chug_prepare._window_clip_name(
                        video_id, window_index
                    )
                    clips.append(clip)
                    clip_root = split_root / clip
                    clip_root.mkdir()
                    source_label = 10 + window_index * 10
                    evidence = {
                        "contract": orchestration.chug_prepare.
                        FLOW_SUPPORT_SELECTION_CONTRACT,
                        "flow_support_contract": orchestration.chug_prepare.
                        FLOW_SUPPORT_CONTRACT,
                        "flow_support_metric_sha256": orchestration.
                        chug_prepare.flow_support_metric_sha256(),
                        "preferred_pair":
                            "previous-source-frame-to-label-frame",
                        "minimum_support": orchestration.chug_prepare.
                        FLOW_TEMPORAL_MIN_SUPPORT,
                        "search_radius_frames": orchestration.chug_prepare.
                        FLOW_SUPPORT_SEARCH_RADIUS_FRAMES,
                        "search_order":
                            "nominal-then-negative-positive-by-distance",
                        "nominal_source_label_frame_id": source_label,
                        "selected_source_label_frame_id": source_label,
                        "selected_offset_frames": 0,
                        "selected_previous_source_frame_id": source_label - 1,
                        "selected_pair_flow_support": 0.5,
                    }
                    frames = [{
                        "frame": frame,
                        "source_frame": source_label - 1 + frame,
                        "source_timestamp_seconds": (source_label - 1 + frame) / 24,
                    } for frame in range(orchestration.WINDOW_FRAME_COUNT)]
                    (clip_root / "label_frames.json").write_text(json.dumps({
                        "schema": 1, "frame_ids": [1],
                    }), encoding="utf-8")
                    (clip_root / "meta.json").write_text(json.dumps({
                        "preparation_contract":
                            orchestration.chug_prepare.PREPARATION_CONTRACT,
                        "split": split,
                        "capture_group_id": capture_group,
                        "source_video_id": video_id,
                        "source_kind": "native-hdr-video",
                        "native_hdr": True,
                        "frame_selection": {
                            "contract": orchestration.chug_prepare.
                            FRAME_SELECTION_CONTRACT,
                            "source_frame_count": 100,
                            "source_frame_rate": 24.0,
                            "window_index": window_index,
                            "retained_frame_count":
                                orchestration.WINDOW_FRAME_COUNT,
                            "temporal_window_radius": 1,
                            "label_frame_ids": [1],
                            "source_label_frame_id": source_label,
                            "frames": frames,
                            "temporal_evidence_selection": evidence,
                        },
                    }), encoding="utf-8")
                    sequences.append({
                        "clip": clip,
                        "frames": orchestration.WINDOW_FRAME_COUNT,
                        "source_frames": orchestration.WINDOW_FRAME_COUNT,
                        "master_source_frames": 100,
                        "source_frame_rate": 24.0,
                        "label_frames": 1,
                        "split": split,
                        "capture_group_id": capture_group,
                        "video_id": video_id,
                        "window_index": window_index,
                        "source_label_frame_id": source_label,
                        "nominal_source_label_frame_id": source_label,
                        "selected_pair_flow_support": 0.5,
                        "temporal_evidence_selection": evidence,
                    })
            sequences.sort(key=lambda row: row["clip"])
            clips.sort()
            context = len(clips) * orchestration.WINDOW_FRAME_COUNT
            manifest = {
                "schema": 2,
                "dataset": "chug-native-pq-v1",
                "domain": "native_hdr_cinematic",
                "source_kind": "native-hdr-video",
                "split": split,
                "source_split": split,
                "preparation_contract":
                    orchestration.chug_prepare.PREPARATION_CONTRACT,
                "temporal_evidence_selection_contract": orchestration.
                    chug_prepare.FLOW_SUPPORT_SELECTION_CONTRACT,
                "source_flow_support_contract": orchestration.chug_prepare.
                    FLOW_SUPPORT_CONTRACT,
                "source_flow_metric_sha256": orchestration.chug_prepare.
                    flow_support_metric_sha256(),
                "source_flow_support_minimum": orchestration.chug_prepare.
                    FLOW_TEMPORAL_MIN_SUPPORT,
                "conversion_contract_sha256": conversion_sha,
                "sequences": sequences,
                "frame_count": context,
                "source_frame_count": context,
                "master_source_frame_count": source_count * 100,
                "label_frame_count": len(clips),
            }
            manifest_path = split_root / "dataset_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            semantic_sha = ("a" if split == "training" else "b") * 64
            clip_manifest = {
                "clips": {clip: {} for clip in clips},
                orchestration.depth_run.clip_hashes.
                MANIFEST_CONTENT_SHA256_FIELD: semantic_sha,
            }
            clip_manifest_path = split_root / "clip_hash_manifest.json"
            clip_manifest_path.write_text(
                json.dumps(clip_manifest), encoding="utf-8"
            )
            datasets[split] = {
                "root": str(split_root.resolve()),
                "dataset_manifest": str(manifest_path.resolve()),
                "dataset_manifest_sha256":
                    orchestration.common.sha256(manifest_path),
                "clip_hash_manifest": {
                    "path": str(clip_manifest_path.resolve()),
                    "sha256": orchestration.common.sha256(clip_manifest_path),
                    "semantic_content_sha256": semantic_sha,
                },
                "clips": clips,
                "context_frame_count": context,
                "source_context_frame_count": context,
                "master_source_frame_count": source_count * 100,
                "label_frame_count": len(clips),
                "capture_group_ids": sorted(groups),
            }
            retention[split] = {
                "source_clips": source_count,
                "window_clips": len(clips),
                "source_frames": source_count * 100,
                "retained_frames": context,
                "label_frames": len(clips),
            }
        bootstrap = {
            "schema": orchestration.chug_prepare.PREPARATION_SCHEMA,
            "contract": orchestration.chug_prepare.PREPARATION_CONTRACT,
            "output_root": str(dataset_root.resolve()),
            "conversion_contract": str(conversion_path.resolve()),
            "conversion_contract_sha256": conversion_sha,
            "temporal_evidence_selection_contract": (
                orchestration.chug_prepare.FLOW_SUPPORT_SELECTION_CONTRACT
            ),
            "source_flow_support_contract": (
                orchestration.chug_prepare.FLOW_SUPPORT_CONTRACT
            ),
            "source_flow_metric_sha256": (
                orchestration.chug_prepare.flow_support_metric_sha256()
            ),
            "source_flow_support_minimum": (
                orchestration.chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT
            ),
            "sealed_test_policy":
                "CHUG test masters were not decoded or opened",
            "retention": {
                "contract": orchestration.chug_prepare.FRAME_SELECTION_CONTRACT,
                "temporal_window_radius": 1,
                "temporal_evidence_selection": {
                    "contract": (
                        orchestration.chug_prepare.FLOW_SUPPORT_SELECTION_CONTRACT
                    ),
                    "flow_support_contract": (
                        orchestration.chug_prepare.FLOW_SUPPORT_CONTRACT
                    ),
                    "flow_support_metric_sha256": (
                        orchestration.chug_prepare.flow_support_metric_sha256()
                    ),
                    "minimum_support": (
                        orchestration.chug_prepare.FLOW_TEMPORAL_MIN_SUPPORT
                    ),
                    "search_radius_frames": (
                        orchestration.chug_prepare.FLOW_SUPPORT_SEARCH_RADIUS_FRAMES
                    ),
                    "search_order":
                        "nominal-then-negative-positive-by-distance",
                    "preferred_pair":
                        "previous-source-frame-to-label-frame",
                },
                "stored_identity":
                    "independent-contiguous-window-clip-with-source-frame-map",
                "splits": retention,
            },
            "datasets": datasets,
            "summary": {
                "training_clips": 60,
                "development_clips": 20,
                "training_policy_samples": 60,
                "development_policy_samples": 20,
            },
        }
        bootstrap_path = (
            dataset_root / orchestration.chug_prepare.BOOTSTRAP_MANIFEST
        )
        bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")
        return dataset_root, conversion_sha

    def publication_mocks(self, conversion_sha):
        def load_clip_manifest(path):
            return json.loads(Path(path).read_text(encoding="utf-8"))

        def load_native_manifest(clip_root):
            clip_root = Path(clip_root)
            metadata = json.loads(
                (clip_root / "meta.json").read_text(encoding="utf-8")
            )
            selection = metadata["frame_selection"]
            source_video_id = metadata["source_video_id"]
            source_video = {
                "dataset": "CHUG",
                "video_id": source_video_id,
                "split": metadata["split"],
                "capture_group_id": metadata["capture_group_id"],
                "license": "CC BY-NC-SA 4.0",
                "source_frame_count": selection["source_frame_count"],
                "source_frame_rate": selection["source_frame_rate"],
                "frame_selection_contract": selection["contract"],
                "window_index": selection["window_index"],
                "source_window_frame_ids": [
                    row["source_frame"] for row in selection["frames"]
                ],
                "source_label_frame_id":
                    selection["source_label_frame_id"],
                "source_window_timestamps_seconds": [
                    row["source_timestamp_seconds"]
                    for row in selection["frames"]
                ],
                "temporal_evidence_selection": selection[
                    "temporal_evidence_selection"
                ],
            }
            frames = {
                row["frame"]: {
                    "timestamp_seconds": row["source_timestamp_seconds"],
                }
                for row in selection["frames"]
            }
            return ({
                "conversion": {"contract_sha256": conversion_sha},
                "source_video": source_video,
            }, frames, clip_root / "frame_model_sources.json")

        return (
            mock.patch.object(
                orchestration.depth_run.clip_hashes, "load_manifest",
                side_effect=load_clip_manifest,
            ),
            mock.patch.object(
                orchestration.depth_run.clip_hashes,
                "verify_selected_clips", return_value={},
            ),
            mock.patch.object(
                orchestration.native_hdr_capture, "validate_clip",
                return_value={
                    "frame_count": orchestration.WINDOW_FRAME_COUNT,
                    "width": orchestration.chug_prepare.TARGET_WIDTH,
                    "height": orchestration.chug_prepare.TARGET_HEIGHT,
                },
            ),
            mock.patch.object(
                orchestration.native_hdr_capture, "load_manifest",
                side_effect=load_native_manifest,
            ),
            mock.patch.object(
                orchestration.selector.sbsbench, "load_gray",
                return_value=None,
            ),
            mock.patch.object(
                orchestration.selector.sbsbench, "flow_temporal_metrics",
                return_value=(None, None, 0.5),
            ),
        )

    def make_args(self, root, dataset_root):
        build = root / "build"
        build.mkdir(exist_ok=True)
        (build / "sunshine.exe").write_bytes(b"test executable")
        conf = root / "bench.conf"
        conf.write_text("sbs_3d_profile = apollo\n", encoding="utf-8")
        return orchestration.parse_args([
            "--dataset-root", str(dataset_root),
            "--workspace", str(root / "labels"),
            "--build-dir", str(build),
            "--conf", str(conf),
            "--python", sys.executable,
        ])

    def build_valid_plan(self, root):
        dataset_root, conversion_sha = self.make_publication(root)
        patches = self.publication_mocks(conversion_sha)
        with (patches[0], patches[1], patches[2], patches[3], patches[4],
              patches[5], mock.patch.object(
                orchestration.depth_run, "selected_depth_model_identity",
                return_value={"model": "test-depth"})):
            plan = orchestration.build_plan(
                self.make_args(root, dataset_root)
            )
        return plan, dataset_root

    def test_plan_authenticates_sparse_layout_and_stops_at_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, dataset_root = self.build_valid_plan(root)
            self.assertEqual(
                self.make_args(root, dataset_root).render_workers, 1
            )
            payload = plan.as_dict()
            self.assertEqual(len(plan.steps), 38)
            self.assertEqual(payload["terminal_phase"], "merge")
            self.assertFalse(payload["training_command_present"])
            self.assertEqual(payload["expected_harness_schema"], 28)
            self.assertEqual(
                [len(dataset.clips) for dataset in plan.datasets], [60, 20]
            )
            self.assertEqual(len(plan.geometry_manifest["tuples"]), 4)
            self.assertEqual(len(plan.input_variant_manifest["variants"]), 5)
            phase_indices = [
                orchestration.PHASES.index(step.phase) for step in plan.steps
            ]
            self.assertEqual(phase_indices, sorted(phase_indices))
            self.assertEqual(
                sum(step.phase == "identity" for step in plan.steps), 4
            )
            self.assertEqual(
                sum(step.phase == "render" for step in plan.steps), 24
            )
            self.assertFalse(any(
                step.kind in {"train", "evaluate"} or
                "train_artistic_policy" in " ".join(step.command) or
                "evaluate_artistic_policy" in " ".join(step.command)
                for step in plan.steps
            ))
            render_steps = [
                step for step in plan.steps if step.kind == "render"
            ]
            depth_by_dataset = {
                step.metadata["dataset"]: str(step.output)
                for step in plan.steps if step.kind == "depth"
            }
            self.assertTrue(all(
                step.metadata["depth_run"] ==
                depth_by_dataset[step.metadata["dataset"]]
                for step in plan.steps if step.kind == "source"
            ))
            self.assertTrue(all(
                "--native-hdr-scrgb" in step.command and
                "--simulate-hdr" not in step.command and
                step.metadata["expected_harness_schema"] == 28 and
                step.metadata["hdr_source_kind"] ==
                "native-pq-in-windows-hdr"
                for step in render_steps
            ))

    def test_bootstrap_schema_and_center_label_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root, conversion_sha = self.make_publication(root)
            bootstrap_path = (
                dataset_root / orchestration.chug_prepare.BOOTSTRAP_MANIFEST
            )
            bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
            bootstrap["schema"] -= 1
            bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "stale or invalid"):
                orchestration._validate_dataset_bootstrap(dataset_root)

            bootstrap["schema"] = orchestration.chug_prepare.PREPARATION_SCHEMA
            bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")
            clip = bootstrap["datasets"]["training"]["clips"][0]
            labels_path = dataset_root / "training" / clip / "label_frames.json"
            labels_path.write_text(json.dumps({
                "schema": 1, "frame_ids": [0],
            }), encoding="utf-8")
            patches = self.publication_mocks(conversion_sha)
            with patches[0], patches[1], patches[2], patches[3], \
                    patches[4], patches[5], \
                    self.assertRaisesRegex(RuntimeError, "center label"):
                orchestration._validate_dataset_bootstrap(dataset_root)

    def test_source_frame_rate_is_cross_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root, conversion_sha = self.make_publication(root)
            bootstrap = json.loads((
                dataset_root / orchestration.chug_prepare.BOOTSTRAP_MANIFEST
            ).read_text(encoding="utf-8"))
            clip = bootstrap["datasets"]["training"]["clips"][0]
            metadata_path = dataset_root / "training" / clip / "meta.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["frame_selection"]["source_frame_rate"] = 30.0
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            patches = self.publication_mocks(conversion_sha)
            with patches[0], patches[1], patches[2], patches[3], \
                    patches[4], patches[5], \
                    self.assertRaisesRegex(
                        RuntimeError, "frame-selection contract differs"
                    ):
                orchestration._validate_dataset_bootstrap(dataset_root)

    def test_flow_metric_contract_is_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root, conversion_sha = self.make_publication(root)
            bootstrap = json.loads((
                dataset_root / orchestration.chug_prepare.BOOTSTRAP_MANIFEST
            ).read_text(encoding="utf-8"))
            clip = bootstrap["datasets"]["training"]["clips"][0]
            metadata_path = dataset_root / "training" / clip / "meta.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["frame_selection"]["temporal_evidence_selection"][
                "flow_support_metric_sha256"
            ] = "0" * 64
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            patches = self.publication_mocks(conversion_sha)
            with patches[0], patches[1], patches[2], patches[3], \
                    patches[4], patches[5], \
                    self.assertRaisesRegex(
                        RuntimeError, "flow-selection contract differs"
                    ):
                orchestration._validate_dataset_bootstrap(dataset_root)

    def test_resume_rejects_changed_plan_without_running_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, _dataset_root = self.build_valid_plan(root)
            args = self.make_args(root, plan.datasets[0].root.parent)
            args.compact_renders = False
            with mock.patch.object(
                    orchestration.common, "step_complete", return_value=True), \
                    mock.patch.object(
                        orchestration, "identity_screen",
                        return_value={"blocked_splits": []}):
                result = orchestration.execute(plan, args)
            self.assertEqual(result["completed_steps"], 38)
            plan_path = plan.workspace / "native_hdr_label_plan.json"
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload["terminal_phase"] = "train"
            plan_path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(
                    orchestration.common, "step_complete", return_value=True), \
                    mock.patch.object(
                        orchestration, "identity_screen",
                        return_value={"blocked_splits": []}), \
                    self.assertRaisesRegex(RuntimeError, "different immutable"):
                orchestration.execute(plan, args)

    def test_parallel_native_render_groups_keep_identity_barrier_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, _dataset_root = self.build_valid_plan(root)
            args = self.make_args(root, plan.datasets[0].root.parent)
            args.compact_renders = False
            args.render_workers = 2
            args.stop_after = "render"
            done = set()
            events = []

            def fake_run(step, plan, logs):
                done.add(step.key)

            def fake_batch(group, plan, logs, workers):
                events.append(f"batch:{group[0].phase}")
                done.update(step.key for step in reversed(group))

            def fake_screen(plan):
                events.append("screen")
                return {"blocked_splits": [], "decision": "proceed"}

            with mock.patch.object(
                    orchestration.common, "step_complete",
                    side_effect=lambda step: step.key in done), \
                    mock.patch.object(
                        orchestration.common, "_run_step",
                        side_effect=fake_run), \
                    mock.patch.object(
                        orchestration.common, "_run_render_batch",
                        side_effect=fake_batch), \
                    mock.patch.object(
                        orchestration, "identity_screen",
                        side_effect=fake_screen):
                result = orchestration.execute(plan, args)

            self.assertEqual(
                events[:3], ["batch:identity", "screen", "batch:render"]
            )
            eligible = [
                step.key for step in plan.steps
                if orchestration.PHASES.index(step.phase) <=
                orchestration.PHASES.index("render")
            ]
            state = json.loads(
                (plan.workspace / "native_hdr_label_state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(state["completed"], eligible)
            self.assertEqual(result["completed_steps"], len(eligible))

    def test_path_escape_and_compaction_marker_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, _dataset_root = self.build_valid_plan(root)
            step = next(item for item in plan.steps if item.kind == "render")
            outside = orchestration.common.Step(
                key=step.key, phase=step.phase, kind=step.kind,
                command=step.command, output=root / "outside",
                metadata=step.metadata,
            )
            steps = list(plan.steps)
            steps[steps.index(step)] = outside
            with self.assertRaisesRegex(RuntimeError, "escapes its stage root"):
                orchestration._validate_label_only_plan(
                    orchestration.Plan(
                        **{**plan.__dict__, "steps": tuple(steps)}
                    )
                )
            step.output.mkdir(parents=True)
            results = step.output / "results.json"
            results.write_text("{}", encoding="utf-8")
            marker = step.output / "bootstrap_compaction.json"
            marker.write_text(json.dumps({
                "schema": 1,
                "contract": "artistic-bootstrap-render-compaction-v1",
                "identity": bool(step.metadata["identity"]),
                "results_sha256": orchestration.common.sha256(results),
            }), encoding="utf-8")
            self.assertTrue(orchestration._render_compaction_complete(step))
            marker.write_text("{}", encoding="utf-8")
            self.assertFalse(orchestration._render_compaction_complete(step))


if __name__ == "__main__":
    unittest.main()
