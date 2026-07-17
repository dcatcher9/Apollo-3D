#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sbsbench"))

import build_clip_hash_manifest as clip_hashes  # noqa: E402
import artistic_policy_ordinal_contract as ordinal_contract  # noqa: E402
import orchestrate_artistic_ordinal_labels as orchestration  # noqa: E402


class OrdinalLabelOrchestrationTests(unittest.TestCase):
    def _dataset(self, root, production, split, source_kind, frames):
        dataset = root / production
        clip = f"{production}_clip"
        clip_root = dataset / clip
        clip_root.mkdir(parents=True)
        for frame_id in range(frames):
            (clip_root / f"frame_{frame_id:05d}.png").write_bytes(
                f"frame-{frame_id}".encode("ascii")
            )
        (clip_root / "label_frames.json").write_text(
            json.dumps({"schema": 1, "frame_ids": [frames // 2]}),
            encoding="utf-8"
        )
        (clip_root / "meta.json").write_text(
            json.dumps({
                "fps": 24.0,
                "expected_flat": False,
                "label_frame_ids": [frames // 2],
            }), encoding="utf-8",
        )
        clip_hashes.build_and_write(dataset, output=dataset /
                                    clip_hashes.MANIFEST_NAME)
        sequence = {"clip": clip, "split": split}
        if source_kind == "mono-video":
            sequence["context_frames"] = frames
            count_field = {"context_frame_count": frames}
        else:
            sequence["frames"] = frames
            count_field = {"frame_count": frames}
        manifest = {
            "schema": 2,
            "production_id": production,
            "source_kind": source_kind,
            "split": split,
            "global_policy_weight": 1.0,
            "sequences": [sequence],
            "label_frame_count": 1,
            **count_field,
        }
        path = dataset / "dataset_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, clip

    def fixture(self, root):
        mono_path, mono_clip = self._dataset(
            root, "mono_train", "training", "mono-video", 2
        )
        pq_path, pq_clip = self._dataset(
            root, "pq_dev", "development", "native-hdr-video", 3
        )
        active = root / "active.json"
        active.write_text(json.dumps({
            "schema": 1,
            "productions": [{
                "production_id": "mono_train",
                "source_kind": "mono-video",
                "split": "training",
                "dataset_manifest": str(mono_path),
                "dataset_manifest_sha256":
                    orchestration.sha256_file(mono_path),
            }, {
                "production_id": "pq_dev",
                "source_kind": "native-hdr-video",
                "split": "development",
                "dataset_manifest": str(pq_path),
                "dataset_manifest_sha256":
                    orchestration.sha256_file(pq_path),
            }, {
                "production_id": "sealed_a",
                "dataset_manifest": str(root / "MUST_NOT_OPEN_A.json"),
                "split": "test",
            }, {
                "production_id": "sealed_b",
                "dataset_manifest": str(root / "MUST_NOT_OPEN_B.json"),
                "split": "test",
            }],
            "split_productions": {
                "training": ["mono_train"],
                "development": ["pq_dev"],
                "test": ["sealed_a", "sealed_b"],
            },
        }), encoding="utf-8")
        build = root / "build"
        build.mkdir()
        (build / "sunshine.exe").write_bytes(b"binary")
        conf = root / "bench.conf"
        conf.write_text("sbs_3d_profile = apollo\n", encoding="utf-8")
        args = argparse.Namespace(
            workspace=root / "work",
            active_split=active,
            build_dir=build,
            conf=conf,
            python=Path(sys.executable),
            run_prefix="fixture",
            score_workers=2,
            scored_result_cache_root=None,
        )
        return args, mono_clip, pq_clip

    def test_main_proves_build_current_before_freezing_plan(self):
        events = []
        args = argparse.Namespace(
            build_dir=Path("build"), dry_run=True,
        )
        plan = mock.Mock()
        plan.as_dict.return_value = {"plan": "frozen"}

        with mock.patch.object(
                orchestration, "parse_args", return_value=args), \
                mock.patch.object(
                    orchestration.run_eval, "require_current_build",
                    side_effect=lambda _path: events.append("build"),
                ), mock.patch.object(
                    orchestration, "build_plan",
                    side_effect=lambda _args: (
                        events.append("plan") or plan
                    ),
                ), mock.patch("builtins.print"):
            orchestration.main([])

        self.assertEqual(events, ["build", "plan"])

    def test_long_real_labels_remain_portable_and_collision_bound(self):
        dataset = orchestration.Dataset(
            production_id=(
                "chug_native_pq_full_cadence_v3_development_with_a_very_"
                "long_authenticated_production_identity"
            ),
            source_kind="native-hdr-video",
            split="development",
            root=Path("dataset"),
            manifest=Path("dataset.json"),
            manifest_sha256="a" * 64,
            clip_hash_manifest=Path("clip_hash_manifest.json"),
            clip_hash_manifest_sha256="b" * 64,
            clip_hash_content_sha256="c" * 64,
            clips=(),
            frame_count=1,
            label_frame_count=1,
            output_frame_count=1,
        )
        condition = orchestration.Condition(
            "native-pq", None, {"contract": "fixture"}
        )
        geometry = orchestration.Geometry("fixture", 1280, 720)
        clip_a = "chug_pq_full_31eebcd1eb732715db4032922c3d355e"
        clip_b = clip_a + "-different"
        batch_a = orchestration._batch_label(
            "ordv2-smoke-6dfebc56", dataset, condition, geometry, clip_a
        )
        batch_b = orchestration._batch_label(
            "ordv2-smoke-6dfebc56", dataset, condition, geometry, clip_b
        )

        self.assertLessEqual(
            len(batch_a), orchestration.MAX_BATCH_LABEL_LENGTH
        )
        self.assertNotEqual(batch_a, batch_b)

        orchestration.run_eval.validate_path_component(
            batch_a + "-batch", "test batch label"
        )
        render = orchestration._render_label(
            "ordv2-smoke-6dfebc56", "safety", dataset, condition,
            geometry, 1.5, clip_a,
        )
        self.assertEqual(render, batch_a + "-s150")
        orchestration.run_eval.validate_path_component(
            render, "test render label"
        )

        eval_root = Path("E:/Git/Repo/Apollo-3D/build/sbs_eval")
        path_limit = orchestration._batch_label_path_limit(eval_root, clip_a)
        path_safe_batch = orchestration._batch_label(
            "ordv2-smoke-6dfebc56", dataset, condition, geometry, clip_a,
            path_limit,
        )
        longest = (
            eval_root.resolve() / f"{path_safe_batch}-batch" / clip_a /
            "scales" / "s150" / orchestration.LONGEST_BATCH_ARTIFACT
        )
        self.assertLessEqual(
            len(str(longest)), orchestration.WINDOWS_LEGACY_PATH_LIMIT
        )

    def test_eval_labels_are_namespaced_per_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_args, _mono, _pq = self.fixture(root)
            second_args = argparse.Namespace(**vars(first_args))
            second_args.workspace = root / "other-work"

            first = orchestration.build_plan(first_args)
            second = orchestration.build_plan(second_args)
            first_labels = {
                step.key for step in first.steps if step.kind == "safety_batch"
            }
            second_labels = {
                step.key for step in second.steps if step.kind == "safety_batch"
            }

            self.assertTrue(first_labels)
            self.assertTrue(second_labels)
            self.assertTrue(first_labels.isdisjoint(second_labels))

    def test_smoke_eval_labels_are_namespaced_per_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_args, _mono, _pq = self.fixture(root)
            first_args.clip_limit = 1
            second_args = argparse.Namespace(**vars(first_args))
            second_args.workspace = root / "other-smoke-work"

            first = orchestration.build_plan(first_args)
            second = orchestration.build_plan(second_args)
            first_labels = {
                step.key for step in first.steps if step.kind == "safety_batch"
            }
            second_labels = {
                step.key for step in second.steps if step.kind == "safety_batch"
            }

            self.assertEqual(first.scope, "smoke-subset-not-training-eligible")
            self.assertEqual(second.scope, "smoke-subset-not-training-eligible")
            self.assertTrue(first_labels.isdisjoint(second_labels))

    def test_scored_cache_identity_ignores_workspace_and_run_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_args, _mono, _pq = self.fixture(root)
            first_args.scored_result_cache_root = root / "shared-score-cache"
            second_args = argparse.Namespace(**vars(first_args))
            second_args.workspace = root / "other-work"
            first = orchestration.build_plan(first_args)
            second = orchestration.build_plan(second_args)
            first_step = next(
                step for step in first.steps if step.kind == "safety_batch"
            )
            second_step = next(
                step for step in second.steps if step.kind == "safety_batch"
            )
            depth_context = {
                "identity": {"depth": "identity"},
                "identity_sha256": "d" * 64,
                "identity_args": {},
                "runtime_snapshot": {},
            }
            with mock.patch.object(
                    orchestration.run_multiscale_eval,
                    "depth_state_identity_context",
                    return_value=depth_context,
                    ), mock.patch.object(
                        orchestration.run_multiscale_eval,
                        "reverify_depth_state_inputs",
                    ):
                first_context = orchestration._score_cache_context(
                    first, first_step
                )
                second_context = orchestration._score_cache_context(
                    second, second_step
                )
            key = (
                orchestration.preprocessing_artifact_cache.
                DirectoryArtifactCache.key
            )
            self.assertNotEqual(first_step.key, second_step.key)
            self.assertEqual(
                key(first_context["identity"]),
                key(second_context["identity"]),
            )
            self.assertEqual(
                first.scorer_runtime_identity,
                second.scorer_runtime_identity,
            )

    def test_plan_batches_all_scales_and_reuses_sparse_visuals(self):
        with tempfile.TemporaryDirectory() as directory:
            args, mono_clip, pq_clip = self.fixture(Path(directory))
            plan = orchestration.build_plan(args)
            safety = [step for step in plan.steps
                      if step.kind == "safety_batch"]
            bundles = [step for step in plan.steps if step.kind == "bundle"]
            sources = [step for step in plan.steps if step.kind == "source"]

            self.assertEqual(len(safety), 5 * 2)
            self.assertEqual(len(bundles), 4 + 1)
            self.assertEqual(len(sources), 4 + 1)
            self.assertEqual({mono_clip, pq_clip}, {
                step.metadata["clip"] for step in bundles
            })
            for step in safety:
                self.assertIn("run_multiscale_eval.py", step.command[1])
                self.assertIn("--scales", step.command)
                self.assertEqual(
                    len(step.command[step.command.index("--scales") + 1].split(",")),
                    ordinal_contract.FRONTIER_SIZE,
                )
                self.assertNotIn("--depth-every", step.command)
                self.assertNotIn("--output-every", step.command)
                self.assertNotIn("--no-artistic-policy", step.command)
                self.assertEqual(step.metadata["evidence_role"], "safety")
                self.assertEqual(len(step.metadata["scale_outputs"]),
                                 ordinal_contract.FRONTIER_SIZE)
            native = [
                step for step in safety
                if step.metadata["condition"] == "native-pq"
            ]
            simulated = [
                step for step in safety
                if step.metadata["condition"].startswith("hdr")
            ]
            self.assertTrue(native)
            self.assertTrue(all("--native-hdr-scrgb" in step.command
                                for step in native))
            self.assertTrue(simulated)
            self.assertTrue(all("--simulate-hdr" in step.command
                                for step in simulated))
            for step in sources:
                self.assertIn(
                    "prepare_ordinal_full_frame_source_rows.py",
                    step.command[1],
                )
                self.assertEqual(step.command.count("--ordinal-bundle"), 1)
                self.assertEqual(step.command.count("--clip"), 1)
                self.assertEqual(
                    len(step.metadata["ordinal_bundles"]), 1
                )
            estimates = plan.as_dict()["estimates"]
            self.assertEqual(estimates["selected_source_bundles"], 5)
            self.assertEqual(estimates["selected_source_rows"], 5)
            self.assertEqual(estimates["selected_source_target_rows"], 5)
            self.assertEqual(estimates["selected_source_context_rows"], 0)
            self.assertEqual(estimates["condition_context_frames"], 0)
            self.assertEqual(estimates["condition_output_frames"], 5)
            self.assertEqual(estimates["multiscale_harness_runs"], 10)
            self.assertEqual(estimates["scalar_scale_score_runs"], 260)
            self.assertEqual(estimates["artifact_render_runs"], 0)
            self.assertEqual(
                estimates["maximum_simultaneous_gpu_render_batches"], 1
            )
            self.assertEqual(
                estimates["maximum_pending_rendered_batches"], 1
            )
            self.assertEqual(
                plan.as_dict()["safety_pipeline"]["contract"],
                orchestration.SAFETY_PIPELINE_CONTRACT,
            )
            self.assertIn("source_publisher", plan.code_identities)
            score_contracts = orchestration._score_cache_contracts(plan)
            self.assertEqual(
                score_contracts["driver"]["depth_state_cache_sha256"],
                plan.code_identities["depth_state_cache"]["sha256"],
            )
            self.assertEqual(
                score_contracts["driver"]["artifact_cache_sha256"],
                plan.code_identities[
                    "preprocessing_artifact_cache"
                ]["sha256"],
            )
            evaluator = score_contracts["evaluator"]
            for role in (
                    "artistic_geometry_contract", "native_hdr_capture",
                    "runtime_scene_evidence", "clip_hash_manifest",
                    "harness_contract"):
                self.assertEqual(
                    evaluator[f"{role}_sha256"],
                    plan.code_identities[role]["sha256"],
                )
            self.assertRegex(plan.thresholds_sha256, r"^[0-9a-f]{64}$")
            self.assertRegex(plan.sbsbench_sha256, r"^[0-9a-f]{64}$")
            self.assertRegex(plan.run_eval_sha256, r"^[0-9a-f]{64}$")

    def test_plan_never_resolves_or_serializes_sealed_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _mono, _pq = self.fixture(Path(directory))
            plan = orchestration.build_plan(args)
            serialized = json.dumps(plan.as_dict())
            self.assertNotIn("MUST_NOT_OPEN_A", serialized)
            self.assertNotIn("MUST_NOT_OPEN_B", serialized)
            self.assertEqual(
                plan.sealed_test_production_ids, ("sealed_a", "sealed_b")
            )
            self.assertEqual(
                {item.split for item in plan.datasets},
                {"training", "development"},
            )

    def test_production_clip_limit_is_explicit_nontraining_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _mono, pq_clip = self.fixture(Path(directory))
            args.production = ["pq_dev"]
            args.clip_limit = 1

            plan = orchestration.build_plan(args)

            self.assertEqual(plan.scope, "smoke-subset-not-training-eligible")
            self.assertEqual([item.production_id for item in plan.datasets],
                             ["pq_dev"])
            self.assertEqual(plan.datasets[0].clips, (pq_clip,))
            value = plan.as_dict()
            self.assertFalse(value["training_eligible"])
            self.assertEqual(value["estimates"]["condition_frames"], 1)
            self.assertEqual(value["estimates"]["multiscale_harness_runs"], 2)
            self.assertEqual(value["estimates"]["scalar_scale_score_runs"], 52)
            self.assertEqual(value["estimates"]["artifact_render_runs"], 0)
            self.assertEqual(
                value["estimates"]["selected_source_bundles"], 1
            )
            self.assertEqual(
                value["estimates"]["selected_source_rows"], 1
            )

    def test_each_clip_condition_bundle_has_exact_52_run_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _mono, _pq = self.fixture(Path(directory))
            plan = orchestration.build_plan(args)
            for step in plan.steps:
                if step.kind != "bundle":
                    continue
                payload = orchestration._run_manifest_payload(step)
                self.assertEqual(
                    payload["contract"],
                    "apollo-ordinal-frame-run-grid-v1",
                )
                self.assertEqual(
                    len(payload["runs"]),
                    2 * ordinal_contract.FRONTIER_SIZE,
                )
                self.assertEqual(len(payload["runs"]), len(set(payload["runs"])))
                self.assertTrue(all(
                    path.endswith("frame_gate_evidence.jsonl")
                    for path in payload["runs"]
                ))

    def test_safety_compaction_retains_only_label_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            clip = "clip"
            (output / clip).mkdir(parents=True)
            (output / "results.json").write_text("{}", encoding="utf-8")
            (output / "frame_gate_evidence.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            (output / clip / "runtime_scene_evidence.json").write_text(
                "{}", encoding="utf-8"
            )
            disposable = output / clip / "sbs_00000.png"
            disposable.write_bytes(b"large-render")

            marker = orchestration.compact_safety_tree(output, [clip])

            self.assertFalse(disposable.exists())
            self.assertTrue((output / "results.json").is_file())
            self.assertTrue((output / "frame_gate_evidence.jsonl").is_file())
            self.assertTrue((
                output / clip / "runtime_scene_evidence.json"
            ).is_file())
            self.assertEqual(marker["deleted_files"], 1)
            self.assertEqual(marker["retained_role"],
                             "selected-target-safety-label-evidence")

    def test_multiscale_compaction_retains_provenance_and_sparse_visuals(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run-s100"
            clip = "clip"
            (output / clip).mkdir(parents=True)
            (output / "results.json").write_text("{}", encoding="utf-8")
            (output / "frame_gate_evidence.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            (output / clip / "runtime_scene_evidence.json").write_text(
                "{}", encoding="utf-8"
            )
            provenance = output / "multiscale_provenance" / clip
            provenance.mkdir(parents=True)
            for name in (
                    "contract.json", "multiscale_batch_manifest.json",
                    "multiscale_contract.json", "render_identity.json"):
                (provenance / name).write_text("{}", encoding="utf-8")
            artifacts = output / "artifact_evidence" / clip
            artifacts.mkdir(parents=True)
            (artifacts / "visual_evidence.json").write_text(
                "{}", encoding="utf-8"
            )
            (artifacts / "sbs_00000.png").write_bytes(b"visual")

            marker = orchestration.compact_safety_tree(
                output, [clip], multiscale=True, scale=1.0
            )
            step = orchestration.Step(
                key=output.name, phase="safety", kind="safety_render",
                command=(), output=output,
                metadata={"clips": [clip], "scale": 1.0},
            )

            orchestration.validate_compaction(step)
            self.assertEqual(len(marker["multiscale_provenance_sha256"]), 4)
            self.assertEqual(len(marker["artifact_evidence_sha256"]), 2)
            contract_path = provenance / "contract.json"
            contract_path.write_text('{"mutated":true}', encoding="utf-8")
            with self.assertRaisesRegex(
                    RuntimeError, "multiscale evidence changed"):
                orchestration.validate_compaction(step)

    def test_source_step_requires_exact_planned_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "source"
            output.mkdir()
            labels = output / "labels.jsonl"
            labels.write_text("{}\n", encoding="utf-8")
            manifest = root / "dataset.json"
            manifest.write_text("{}", encoding="utf-8")
            bundle = root / "bundle.jsonl"
            bundle.write_text("{}\n", encoding="utf-8")
            variant = orchestration.input_color.sdr_input_variant()
            variant_hash = (
                orchestration.input_color.input_variant_sha256(variant)
            )
            (output / "summary.json").write_text(json.dumps({
                "accepted": 1,
                "row_count": 1,
                "target_row_count": 1,
                "context_row_count": 0,
                "production_id": "production",
                "source_kind": "mono-video",
                "split": "training",
                "scope": "full-dataset",
                "selected_clips": ["clip"],
                "input_variant_sha256": variant_hash,
            }), encoding="utf-8")
            (output / "source_contract.json").write_text(json.dumps({
                "dataset_manifest": {
                    "path": str(manifest.resolve()),
                    "sha256": orchestration.sha256_file(manifest),
                },
                "input_variant": variant,
                "input_variant_sha256": variant_hash,
                "scope": "full-dataset",
                "selected_clips": ["clip"],
                "ordinal_bundles": [{"path": str(bundle.resolve())}],
            }), encoding="utf-8")
            row = {
                "production_id": "production",
                "source_kind": "mono-video",
                "split": "training",
                "input_variant": variant,
                "input_variant_sha256": variant_hash,
                "clip": "clip",
                "row_role": "target",
                "ordinal_bundle": str(bundle.resolve()),
            }
            step = orchestration.Step(
                key="sources-production-sdr", phase="sources",
                kind="source", command=(), output=output,
                metadata={
                    "production_id": "production",
                    "source_kind": "mono-video",
                    "split": "training",
                    "dataset_manifest": str(manifest),
                    "dataset_manifest_sha256":
                        orchestration.sha256_file(manifest),
                    "condition": "sdr",
                    "input_variant": variant,
                    "input_variant_sha256": variant_hash,
                    "clips": ["clip"],
                    "expected_frames": 1,
                    "expected_target_frames": 1,
                    "expected_context_frames": 0,
                    "ordinal_bundles": [str(bundle)],
                },
            )
            with mock.patch.object(
                    orchestration.full_sources,
                    "validate_full_frame_source_bundle",
                    return_value=[row]):
                result = orchestration.validate_source(step)
                self.assertEqual(result["rows"], [row])
                broken = dict(row, production_id="wrong")
                with mock.patch.object(
                        orchestration.full_sources,
                        "validate_full_frame_source_bundle",
                        return_value=[broken]):
                    with self.assertRaisesRegex(
                            RuntimeError, "source rows differ"):
                        orchestration.validate_source(step)

    def test_bundle_step_delegates_to_exact_summary_validator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bundle"
            output.mkdir()
            labels = output / "labels.jsonl"
            summary = output / "summary.json"
            manifest = output / "run_grid.json"
            labels.write_text("{}\n", encoding="utf-8")
            summary.write_text("{}\n", encoding="utf-8")
            step = orchestration.Step(
                key="bundle-fixture", phase="bundle", kind="bundle",
                command=(), output=output,
                metadata={
                    "clip": "clip", "input_variant": {"kind": "fixture"},
                    "input_variant_sha256": "a" * 64,
                    "run_manifest": str(manifest),
                    "run_paths": [str(root / "run")],
                    "expected_runs": 1,
                },
            )
            orchestration.write_json_atomic(
                manifest, orchestration._run_manifest_payload(step)
            )
            records = [{"record": "header"}, {"record": "trailer"}]
            with mock.patch.object(
                    orchestration.bundle_builder,
                    "validate_frame_label_bundle",
                    return_value=records), mock.patch.object(
                        orchestration.bundle_builder,
                        "validate_frame_label_summary",
                        side_effect=RuntimeError(
                            "ordinal frame label summary differs from its "
                            "authenticated bundle"
                        ),
                    ) as validate_summary, self.assertRaisesRegex(
                        RuntimeError, "authenticated bundle"):
                orchestration.validate_bundle(step)
            validate_summary.assert_called_once_with(
                labels, summary, records=records
            )

    def test_authenticated_output_after_interruption_is_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "complete.json"
            output.write_text("{}", encoding="utf-8")
            step = orchestration.Step(
                key="safety-a", phase="safety", kind="safety_batch",
                command=(), output=output, metadata={},
            )
            args = argparse.Namespace(restart=False)

            with mock.patch.object(
                    orchestration, "_step_complete", return_value=True):
                should_run = orchestration._prepare_output(
                    step, mock.Mock(), args, []
                )

            self.assertFalse(should_run)

    def test_partial_output_names_public_repair_option(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "partial"
            output.mkdir()
            (output / "fragment").write_text("x", encoding="utf-8")
            step = orchestration.Step(
                key="safety-a", phase="safety", kind="safety_batch",
                command=(), output=output, metadata={},
            )
            plan = mock.Mock(workspace=root)
            args = argparse.Namespace(restart=False)

            with mock.patch.object(
                    orchestration, "_step_complete", return_value=False), \
                    self.assertRaisesRegex(
                        RuntimeError, "--repair-partials"):
                orchestration._prepare_output(step, plan, args, [])

    def test_staged_command_precedes_remainder_extra(self):
        step = orchestration.Step(
            key="safety-a", phase="safety", kind="safety_batch",
            command=(
                "python", "run_multiscale_eval.py", "--summary", "out",
                "--extra", "--eye-w", "1280",
            ),
            output=Path("out"), metadata={},
        )

        command = orchestration._staged_step_command(step, "render")

        self.assertEqual(
            command[command.index("--stage") + 1], "render"
        )
        self.assertLess(command.index("--stage"), command.index("--extra"))
        self.assertEqual(command[-2:], ("--eye-w", "1280"))

    def test_safety_pipeline_overlaps_score_with_next_serial_render(self):
        steps = [orchestration.Step(
            key=f"safety-{name}", phase="safety", kind="safety_batch",
            command=(), output=Path(name), metadata={},
        ) for name in ("a", "b")]
        score_a_started = threading.Event()
        render_b_started = threading.Event()
        events = []

        def run(item, _logs, *, stage=None, cancel_event=None):
            events.append((stage, item.key, "start"))
            if stage == "score" and item.key == "safety-a":
                score_a_started.set()
                self.assertTrue(render_b_started.wait(1.0))
            if stage == "render" and item.key == "safety-b":
                self.assertTrue(score_a_started.wait(1.0))
                render_b_started.set()
            events.append((stage, item.key, "end"))

        authenticated = []

        def authenticate(item, future):
            future.result()
            authenticated.append(item.key)

        with mock.patch.object(
                orchestration, "_run_step_process", side_effect=run):
            orchestration._run_safety_pipeline(
                steps, Path("logs"), authenticate
            )

        self.assertTrue(render_b_started.is_set())
        self.assertEqual(authenticated, ["safety-a", "safety-b"])
        self.assertLess(
            events.index(("score", "safety-a", "start")),
            events.index(("render", "safety-b", "end")),
        )

    def test_safety_pipeline_checkpoints_prior_score_before_render_failure(self):
        steps = [orchestration.Step(
            key=f"safety-{name}", phase="safety", kind="safety_batch",
            command=(), output=Path(name), metadata={},
        ) for name in ("a", "b")]
        score_a_started = threading.Event()
        authenticated = []

        def run(item, _logs, *, stage=None, cancel_event=None):
            if stage == "score" and item.key == "safety-a":
                score_a_started.set()
                return
            if stage == "render" and item.key == "safety-b":
                self.assertTrue(score_a_started.wait(1.0))
                raise RuntimeError("render boom")

        def authenticate(item, future):
            future.result()
            authenticated.append(item.key)

        with mock.patch.object(
                orchestration, "_run_step_process", side_effect=run), \
                self.assertRaisesRegex(RuntimeError, "render boom"):
            orchestration._run_safety_pipeline(
                steps, Path("logs"), authenticate
            )

        self.assertEqual(authenticated, ["safety-a"])

    def test_safety_pipeline_interrupt_cancels_active_score(self):
        steps = [orchestration.Step(
            key=f"safety-{name}", phase="safety", kind="safety_batch",
            command=(), output=Path(name), metadata={},
        ) for name in ("a", "b")]
        score_started = threading.Event()
        score_cancelled = threading.Event()

        def run(item, _logs, *, stage=None, cancel_event=None):
            self.assertIsNotNone(cancel_event)
            if stage == "score" and item.key == "safety-a":
                score_started.set()
                self.assertTrue(cancel_event.wait(1.0))
                score_cancelled.set()
                return
            if stage == "render" and item.key == "safety-b":
                self.assertTrue(score_started.wait(1.0))
                raise KeyboardInterrupt()

        def authenticate(item, future):
            future.result()

        with mock.patch.object(
                orchestration, "_run_step_process", side_effect=run), \
                self.assertRaises(KeyboardInterrupt):
            orchestration._run_safety_pipeline(
                steps, Path("logs"), authenticate
            )

        self.assertTrue(score_cancelled.is_set())

    def test_catalog_requires_one_source_row_per_safety_label(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _mono, _pq = self.fixture(Path(directory))
            plan = orchestration.build_plan(args)
            for step in plan.steps:
                if step.kind not in {"bundle", "source"}:
                    continue
                step.output.mkdir(parents=True, exist_ok=True)
                (step.output / "labels.jsonl").write_text(
                    "{}\n", encoding="utf-8"
                )
                (step.output / "summary.json").write_text(
                    "{}", encoding="utf-8"
                )
                if step.kind == "source":
                    (step.output / "source_contract.json").write_text(
                        "{}", encoding="utf-8"
                    )

            def fake_bundle(step):
                count = step.metadata["frame_count"]
                identities = [{
                    "geometry_sha256": geometry,
                    "multiscale_batch_manifest_sha256": batch,
                } for geometry, batch in (
                    ("1" * 64, "3" * 64), ("2" * 64, "4" * 64)
                ) for _scale in ordinal_contract.SCALES]
                return [{
                    "common_run_identity": {
                        "executable_sha256": plan.executable_sha256,
                    },
                    "scale_run_identities": identities,
                }, *({} for _ in range(count)), {}]

            def fake_source(step, short=False):
                target_count = (
                    step.metadata["expected_target_frames"] - int(short)
                )
                rows = [{
                    "clip": step.metadata["clips"][0],
                    "row_role": "target",
                } for _index in range(target_count)]
                return {
                    "rows": rows,
                    "summary": {"scope": "full-dataset"},
                    "labels": step.output / "labels.jsonl",
                    "summary_path": step.output / "summary.json",
                    "contract_path": step.output / "source_contract.json",
                }

            with mock.patch.object(
                    orchestration, "validate_bundle",
                    side_effect=fake_bundle), mock.patch.object(
                        orchestration, "validate_source",
                        side_effect=fake_source):
                catalog = orchestration.build_catalog(plan)
            self.assertTrue(catalog["training_eligible"])
            self.assertEqual(catalog["source_bundle_count"], 5)
            self.assertEqual(catalog["source_row_count"], 5)
            self.assertEqual(catalog["source_target_row_count"], 5)
            self.assertEqual(catalog["source_context_row_count"], 0)
            self.assertEqual(
                catalog["source_row_count_by_split_and_regime"],
                {"development/hdr": 1, "training/hdr": 3,
                 "training/sdr": 1},
            )

            first = True

            def one_short(step):
                nonlocal first
                short = first
                first = False
                return fake_source(step, short=short)

            with mock.patch.object(
                    orchestration, "validate_bundle",
                    side_effect=fake_bundle), mock.patch.object(
                        orchestration, "validate_source",
                        side_effect=one_short):
                with self.assertRaisesRegex(RuntimeError, "join exactly"):
                    orchestration.build_catalog(plan)


if __name__ == "__main__":
    unittest.main()
