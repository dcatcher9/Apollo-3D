import json
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

import orchestrate_artistic_hdr_bootstrap as orchestration  # noqa: E402


class BootstrapOrchestrationTests(unittest.TestCase):
    def make_workspace(self, root):
        workspace = root / "workspace"
        datasets_root = workspace / "datasets"
        datasets_root.mkdir(parents=True)
        rows = []
        productions = {"training": [], "development": [], "test": [
            "sealed-reds", "sealed-spring",
        ]}
        settings = (
            ("reds", "training", 800, 40, 8),
            ("reds", "development", 200, 10, 2),
            ("spring", "training", 593, 20, 4),
            ("spring", "development", 325, 10, 2),
        )
        for source, split, context, labels, shots in settings:
            production = f"{source}-bootstrap-{split}"
            dataset = datasets_root / source / split
            dataset.mkdir(parents=True)
            manifest = dataset / "dataset_manifest.json"
            manifest.write_text(json.dumps({
                "schema": 2,
                "production_id": production,
                "split": split,
            }), encoding="utf-8")
            clip_hash_manifest = dataset / "clip_hash_manifest.json"
            clip_hash_manifest.write_text(json.dumps({
                "schema": 2,
                "clips": {},
            }), encoding="utf-8")
            clips = [f"{source}_{split}_{index:03d}" for index in range(shots)]
            rows.append({
                "source": source,
                "split": split,
                "production_id": production,
                "output_root": str(dataset),
                "clips": clips,
                "context_frame_count": context,
                "label_frame_count": labels,
                "dataset_manifest_sha256": orchestration.sha256(manifest),
                "clip_hash_manifest_sha256":
                    orchestration.sha256(clip_hash_manifest),
            })
            productions[split].append(production)
        active = datasets_root / "active_artistic_split_bootstrap.json"
        active.write_text(json.dumps({
            "schema": 2,
            "split_productions": productions,
        }), encoding="utf-8")
        bootstrap = datasets_root / "bootstrap_manifest.json"
        bootstrap.write_text(json.dumps({
            "schema": 1,
            "preparation_contract": orchestration.BOOTSTRAP_CONTRACT,
            "normalization": {"target_width": 1280, "target_height": 720},
            "datasets": rows,
            "training_contract": {
                "active_split": str(active),
                "active_split_sha256": orchestration.sha256(active),
            },
        }), encoding="utf-8")
        return workspace

    def make_args(self, root, workspace):
        build = root / "build"
        assets = build / "assets"
        assets.mkdir(parents=True)
        (build / "sunshine.exe").write_bytes(b"benchmark")
        recipe = orchestration.depth_run.selected_depth_engine_recipe()
        (assets / f"{orchestration.DEPTH_MODEL}.{recipe}.engine").write_bytes(
            b"engine"
        )
        depth_root = root / "Depth-Anything-V2"
        depth_root.mkdir()
        weights = root / "depth.pth"
        weights.write_bytes(b"weights")
        return orchestration.parse_args([
            "--workspace", str(workspace),
            "--build-dir", str(build),
            "--conf", str(REPO / "tools" / "sbsbench" / "bench.conf"),
            "--python", sys.executable,
            "--depth-anything-root", str(depth_root),
            "--depth-weights", str(weights),
        ])

    def test_plan_is_complete_identity_first_and_never_supplies_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self.make_workspace(root)
            args = self.make_args(root, workspace)
            self.assertEqual(args.render_workers, 1)
            plan = orchestration.build_plan(args)
            payload = plan.as_dict()

            self.assertEqual(payload["counts"], {
                "depth": 16,
                "source": 16,
                "render": 224,
                "select": 32,
                "merge": 4,
                "train": 1,
                "evaluate": 1,
            })
            self.assertEqual(payload["subprocess_steps"], 294)
            self.assertEqual(
                payload["estimates"]["depth_generation_runs"], 16
            )
            self.assertEqual(
                payload["estimates"]["full_cadence_frame_visits"], 107408
            )
            self.assertEqual(
                payload["estimates"]["sparse_output_frame_artifact_sets"],
                8960,
            )
            self.assertEqual(
                payload["estimates"]["exact_two_disparity_rasters_written_gib"],
                48.07,
            )
            self.assertEqual(
                payload["estimates"]["training_policy_samples"], 240
            )
            self.assertEqual(
                payload["estimates"]["development_policy_samples"], 80
            )
            self.assertEqual(
                payload["estimates"]["safety_geometries_per_policy_sample"],
                2,
            )
            self.assertEqual(
                {(item.eye_width, item.eye_height) for item in plan.geometries},
                {(1280, 720), (960, 540)},
            )
            self.assertTrue(all(
                orchestration._geometry_value(
                    1280, 720, item,
                    orchestration.input_color.COLOR_MODE_SDR,
                )["content_scale_x"] == 1.0 and
                orchestration._geometry_value(
                    1280, 720, item,
                    orchestration.input_color.COLOR_MODE_SDR,
                )["content_scale_y"] == 1.0
                for item in plan.geometries
            ))
            self.assertEqual(
                {
                    value["windows_sdr_white_level_raw"]
                    for value in plan.input_variant_manifest["variants"]
                },
                {None, 1000, 2500, 6000},
            )
            self.assertEqual(
                {
                    value["color_mode"]
                    for value in plan.geometry_manifest["tuples"]
                },
                {
                    orchestration.input_color.COLOR_MODE_SDR,
                    orchestration.input_color.COLOR_MODE_HDR,
                },
            )
            self.assertEqual(len(plan.geometry_manifest["tuples"]), 4)

            phases = [step.phase for step in plan.steps]
            first_render = phases.index("render")
            self.assertTrue(all(
                phase in {"depth", "sources", "identity"}
                for phase in phases[:first_render]
            ))
            identity = [
                step for step in plan.steps if step.phase == "identity"
            ]
            self.assertEqual(len(identity), 32)
            self.assertTrue(all(step.metadata["scale"] == 1.0 for step in identity))
            depth_steps = [
                step for step in plan.steps if step.kind == "depth"
            ]
            self.assertEqual(len(depth_steps), 16)
            sdr = [
                step for step in depth_steps
                if step.metadata["condition"] == "sdr"
            ]
            hdr = [
                step for step in depth_steps
                if step.metadata["condition"] != "sdr"
            ]
            self.assertEqual((len(sdr), len(hdr)), (4, 12))
            self.assertTrue(all(
                "--simulate-hdr" not in step.command and
                step.metadata["input_variant"] ==
                orchestration.input_color.sdr_input_variant() and
                step.metadata["metric_preview_encoding"] ==
                orchestration.selector.sbs_contract.
                expected_metric_preview_encoding(
                    orchestration.input_color.COLOR_MODE_SDR
                )
                for step in sdr
            ))
            self.assertTrue(all(
                "--simulate-hdr" in step.command and
                step.metadata["metric_preview_encoding"] ==
                orchestration.selector.sbs_contract.
                expected_metric_preview_encoding(
                    orchestration.input_color.COLOR_MODE_HDR
                )
                for step in hdr
            ))
            full = [
                step for step in depth_steps
                if step.metadata["clip_hash_verification"] == "full"
            ]
            stat = [
                step for step in depth_steps
                if step.metadata["clip_hash_verification"] == "stat"
            ]
            self.assertEqual((len(full), len(stat)), (4, 12))
            self.assertTrue(all(
                "--verify-clip-hashes" in step.command for step in full
            ))
            self.assertTrue(all(
                "--verify-clip-hashes" not in step.command for step in stat
            ))

            train = next(step for step in plan.steps if step.kind == "train")
            evaluate = next(
                step for step in plan.steps if step.kind == "evaluate"
            )
            self.assertNotIn("sealed-reds", " ".join(train.command))
            self.assertNotIn("sealed-spring", " ".join(train.command))
            self.assertNotIn("sealed-reds", " ".join(evaluate.command))
            self.assertNotIn("sealed-spring", " ".join(evaluate.command))
            self.assertEqual(evaluate.command[-2:], ("--split", "development"))
            self.assertFalse(plan.geometry_manifest_path.exists())
            self.assertFalse(plan.input_variant_manifest_path.exists())

    def test_scale_grid_and_selector_reuse_explicit_identity_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self.make_workspace(root)
            plan = orchestration.build_plan(
                self.make_args(root, workspace)
            )
            select = next(step for step in plan.steps if step.kind == "select")
            command = list(select.command)
            control = command[command.index("--control") + 1]
            candidates = [
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--candidate"
            ]
            self.assertEqual(
                [float(value.split("=", 1)[0]) for value in candidates],
                list(orchestration.SCALES),
            )
            identity = next(
                value.split("=", 1)[1] for value in candidates
                if value.startswith("1.0=")
            )
            self.assertEqual(identity, control)

    def test_depth_resume_requires_schema28_executable_and_preview_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self.make_workspace(root)
            plan = orchestration.build_plan(self.make_args(root, workspace))
            step = next(item for item in plan.steps if item.kind == "depth")
            step.output.mkdir(parents=True)
            manifest = {
                "schema": orchestration.depth_run.DEPTH_RUN_MANIFEST_SCHEMA,
                "harness_schema":
                    orchestration.selector.EXPECTED_HARNESS_SCHEMA,
                "purpose": "artistic-policy depth supervision",
                "suite": step.metadata["dataset_root"],
                "suite_manifest_sha256":
                    step.metadata["dataset_manifest_sha256"],
                "clip_hash_manifest_file_sha256":
                    step.metadata["clip_hash_manifest_sha256"],
                "clip_hash_verification":
                    step.metadata["clip_hash_verification"],
                "executable_sha256": step.metadata["executable_sha256"],
                "conf_sha256": step.metadata["conf_sha256"],
                "model": step.metadata["model"],
                "model_asset_identity":
                    step.metadata["model_asset_identity"],
                "model_asset_identity_sha256":
                    step.metadata["model_asset_identity_sha256"],
                "input_variant": step.metadata["input_variant"],
                "input_variant_sha256":
                    orchestration.input_color.input_variant_sha256(
                        step.metadata["input_variant"]
                    ),
                "metric_preview_encoding":
                    step.metadata["metric_preview_encoding"],
                "output_gt_right_only": False,
                "source_identities": {
                    clip: {"source_identity_method": "fixture"}
                    for clip in step.metadata["clips"]
                },
                "clips": [{
                    "clip": clip,
                    "metric_preview_encoding":
                        step.metadata["metric_preview_encoding"],
                } for clip in step.metadata["clips"]],
                "clip_count": len(step.metadata["clips"]),
            }

            def publish():
                (step.output / "depth_run_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

            publish()
            with mock.patch.object(
                    orchestration.depth_run, "valid_completed_clip",
                    return_value=True):
                self.assertTrue(orchestration._depth_complete(step))
                manifest["harness_schema"] = 25
                publish()
                self.assertFalse(orchestration._depth_complete(step))
                manifest["harness_schema"] = (
                    orchestration.selector.EXPECTED_HARNESS_SCHEMA
                )
                manifest["clips"][0]["metric_preview_encoding"] = (
                    "source-relative-srgb-from-scrgb-white-normalized-v1"
                )
                publish()
                self.assertFalse(orchestration._depth_complete(step))

    def test_source_resume_is_bound_to_current_depth_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self.make_workspace(root)
            plan = orchestration.build_plan(self.make_args(root, workspace))
            step = next(item for item in plan.steps if item.kind == "source")
            depth_run = Path(step.metadata["depth_run"])
            depth_run.mkdir(parents=True)
            depth_manifest = depth_run / "depth_run_manifest.json"
            depth_manifest.write_text(
                json.dumps({"generation": 1}), encoding="utf-8"
            )
            step.output.mkdir(parents=True)
            labels = step.output / "labels.jsonl"
            labels.write_text("{}\n", encoding="utf-8")

            def publish_source(run_path=depth_run):
                manifest_hash = orchestration.sha256(depth_manifest)
                contract = {
                    "run": str(run_path),
                    "clips": step.metadata["dataset_root"],
                    "input_variant": step.metadata["input_variant"],
                    "input_variant_sha256":
                        orchestration.input_color.input_variant_sha256(
                            step.metadata["input_variant"]
                        ),
                    "metric_preview_encoding":
                        step.metadata["metric_preview_encoding"],
                    "run_contract": {
                        "kind": "depth_run_manifest",
                        "sha256": manifest_hash,
                        "suite_manifest_sha256":
                            step.metadata["dataset_manifest_sha256"],
                    },
                    "depth_authentication": {
                        "manifest_sha256": manifest_hash,
                    },
                }
                contract_path = step.output / "source_contract.json"
                contract_path.write_text(
                    json.dumps(contract), encoding="utf-8"
                )
                (step.output / "summary.json").write_text(json.dumps({
                    "accepted": step.metadata["expected_labels"],
                    "labels_sha256": orchestration.sha256(labels),
                    "source_contract_sha256":
                        orchestration.sha256(contract_path),
                }), encoding="utf-8")

            publish_source()
            self.assertTrue(orchestration._source_complete(step))

            depth_manifest.write_text(
                json.dumps({"generation": 2}), encoding="utf-8"
            )
            self.assertFalse(orchestration._source_complete(step))

            publish_source(root / "another-depth-run")
            self.assertFalse(orchestration._source_complete(step))

            publish_source()
            self.assertTrue(orchestration._source_complete(step))

    def test_selected_bundle_resume_authenticates_current_code_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selected"
            output.mkdir()
            labels = output / "labels.jsonl"
            labels.write_text("{}\n", encoding="utf-8")
            code_path = output / "frozen.py"
            code_path.write_text("# frozen\n", encoding="utf-8")
            identity = {
                "path": str(code_path),
                "sha256": orchestration.sha256(code_path),
            }
            roles = (
                orchestration.label_merge.SELECTED_LABEL_FITTER_CODE_ROLES
            )
            contract = {
                "code": {
                    role: dict(identity)
                    for role in roles
                },
            }
            contract_path = output / "label_fitter_contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            summary = {
                "schema": orchestration.label_merge.SOURCE_LABEL_SCHEMA,
                "accepted": 1,
                "labels_sha256": orchestration.sha256(labels),
                "label_fitter_contract_sha256":
                    orchestration.sha256(contract_path),
            }
            (output / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            step = SimpleNamespace(
                output=output, metadata={"expected_labels": 1}
            )

            self.assertTrue(orchestration._bundle_complete(
                step, orchestration.label_merge.SOURCE_LABEL_SCHEMA
            ))
            code_path.write_text("# changed\n", encoding="utf-8")
            self.assertFalse(orchestration._bundle_complete(
                step, orchestration.label_merge.SOURCE_LABEL_SCHEMA
            ))

    def test_render_compaction_keeps_only_replayable_identity_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_root = root / "sbs_eval"
            output = eval_root / "identity"
            clip_root = output / "clip"
            clip_root.mkdir(parents=True)
            payload = {
                "verdict": "comparison_only",
                "meta": {
                    "run_name": "identity",
                    "artistic_scale_override": 1.0,
                    "output_selection_mode": "label-frames",
                    "artistic_policy": False,
                    "clips_root": str(root / "clips"),
                    "clip_hash_manifest_sha256": "a" * 64,
                },
                "clips": {"clip": {"meta": {
                    "harness_schema":
                        orchestration.selector.EXPECTED_HARNESS_SCHEMA,
                    "color_mode": orchestration.input_color.COLOR_MODE_HDR,
                    "hdr_source_kind":
                        orchestration.selector.sbs_contract.HDR_SOURCE_SIMULATED,
                    "metric_preview_encoding":
                        orchestration._condition_preview_encoding(1000),
                    "sdr_white_level_raw": 1000,
                    "hdr_input_scale": 1.0,
                    "eye_width": 1280,
                    "eye_height": 720,
                    "depth_compensation": "none",
                    "output_selection_mode": "label-frames",
                    "output_selected_frame_ids": [0],
                }}},
            }
            (output / "results.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            for name in (
                    "contract.json", "warp_disparity_00000.f32",
                    "warp_unclamped_disparity_00000.f32", "sbs_00000.png"):
                (clip_root / name).write_bytes(b"artifact")
            step = orchestration.Step(
                key="identity", phase="identity", kind="render",
                command=("python",), output=output,
                metadata={
                    "clips": ["clip"], "condition": "w1000",
                    "raw_white": 1000,
                    "color_mode": orchestration.input_color.COLOR_MODE_HDR,
                    "input_variant":
                        orchestration.input_color.windows_hdr_input_variant(1000),
                    "metric_preview_encoding":
                        orchestration._condition_preview_encoding(1000),
                    "clips_root": str(root / "clips"),
                    "clip_hash_manifest_sha256": "a" * 64,
                    "scale": 1.0, "eye_width": 1280, "eye_height": 720,
                    "identity": True,
                },
            )

            orchestration.compact_render(step, eval_root)

            self.assertTrue((output / "results.json").is_file())
            self.assertTrue((clip_root / "contract.json").is_file())
            self.assertTrue(
                (clip_root / "warp_disparity_00000.f32").is_file()
            )
            self.assertTrue(
                (clip_root / "warp_unclamped_disparity_00000.f32").is_file()
            )
            self.assertFalse((clip_root / "sbs_00000.png").exists())
            marker = json.loads(
                (output / "bootstrap_compaction.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(marker["identity"])

    def test_identity_screen_blocks_only_the_failed_input_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self.make_workspace(root)
            plan = orchestration.build_plan(
                self.make_args(root, workspace)
            )
            identity_steps = [
                step for step in plan.steps if step.phase == "identity"
            ]
            for step in identity_steps:
                step.output.mkdir(parents=True)
                failures = []
                clips = {}
                for clip_name in step.metadata["clips"]:
                    clip_root = step.output / clip_name
                    clip_root.mkdir()
                    (clip_root / "contract.json").write_text(
                        "{}", encoding="utf-8"
                    )
                    for prefix in (
                            "warp_disparity", "warp_unclamped_disparity"):
                        (clip_root / f"{prefix}_00000.f32").write_bytes(
                            b"evidence"
                        )
                    clips[clip_name] = {"meta": {
                        "harness_schema":
                            orchestration.selector.EXPECTED_HARNESS_SCHEMA,
                        "color_mode": step.metadata["color_mode"],
                        "hdr_source_kind": (
                            orchestration.selector.sbs_contract.
                            input_variant_hdr_source_kind(
                                step.metadata["input_variant"]
                            )
                        ),
                        "metric_preview_encoding":
                            step.metadata["metric_preview_encoding"],
                        "sdr_white_level_raw": (
                            step.metadata["raw_white"] or 0
                        ),
                        "hdr_input_scale": (
                            (step.metadata["raw_white"] or 0) / 1000.0
                        ),
                        "eye_width": step.metadata["eye_width"],
                        "eye_height": step.metadata["eye_height"],
                        "depth_compensation": "none",
                        "output_selection_mode": "label-frames",
                        "output_selected_frame_ids": [0],
                    }}
                    if step.metadata["raw_white"] == 2500:
                        failures.append({
                            "clip": clip_name,
                            "metric": "source_coverage_pct",
                        })
                payload = {
                    "verdict": (
                        "hard_failures" if failures else "comparison_only"
                    ),
                    "meta": {
                        "run_name": step.output.name,
                        "artistic_scale_override": 1.0,
                        "output_selection_mode": "label-frames",
                        "artistic_policy": False,
                        "clips_root": step.metadata["clips_root"],
                        "clip_hash_manifest_sha256":
                            step.metadata["clip_hash_manifest_sha256"],
                    },
                    "clips": clips,
                    "hard_failures": failures,
                }
                (step.output / "results.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            screen = orchestration.identity_screen(plan)

            self.assertEqual(
                screen["blocked_splits"], ["development", "training"]
            )
            self.assertEqual(screen["blocked_conditions"], [
                "development:w2500", "training:w2500",
            ])
            self.assertEqual(screen["blocked_regimes"], [
                "development:hdr", "training:hdr",
            ])
            self.assertEqual(screen["decision"], "stop-before-candidate-grid")
            self.assertTrue(all(
                value["conditions"]["w2500"][
                    "identity_feasible_across_two_geometries"
                ] == 0 and
                value["conditions"]["sdr"][
                    "identity_feasible_across_two_geometries"
                ] > 0 and
                value["conditions"]["w1000"][
                    "identity_feasible_across_two_geometries"
                ] > 0
                for value in screen["datasets"].values()
            ))

    def test_parallel_render_runner_is_bounded_and_phase_restricted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            eval_root = root / "build" / "sbs_eval"
            workspace.mkdir()
            eval_root.mkdir(parents=True)
            run_eval = REPO / "tools" / "sbsbench" / "run_eval.py"
            steps = tuple(
                orchestration.Step(
                    key=f"render-{index}", phase="render", kind="render",
                    command=(sys.executable, str(run_eval)),
                    output=eval_root / f"run-{index}",
                    metadata={},
                )
                for index in range(6)
            )
            active = 0
            peak = 0
            lock = threading.Lock()

            def fake_worker(step, plan, logs, cancel_event):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.02)
                with lock:
                    active -= 1

            with mock.patch.object(
                    orchestration, "_run_render_step_buffered",
                    side_effect=fake_worker):
                orchestration._run_render_batch(
                    steps, SimpleNamespace(
                        repo=REPO, workspace=workspace,
                        build_dir=root / "build",
                    ), workspace / "logs", 2
                )
            self.assertEqual(peak, 2)
            with self.assertRaisesRegex(RuntimeError, "between 1 and 2"):
                orchestration._run_render_batch(
                    steps, SimpleNamespace(
                        repo=REPO, workspace=workspace,
                        build_dir=root / "build",
                    ), workspace / "logs", 3
                )

            mixed = list(steps[:2])
            mixed[1] = orchestration.Step(
                key="select", phase="select", kind="select",
                command=("ignored",), output=root / "select", metadata={},
            )
            with self.assertRaisesRegex(RuntimeError, "one render phase"):
                orchestration._run_render_batch(
                    mixed, SimpleNamespace(
                        repo=REPO, workspace=workspace,
                        build_dir=root / "build",
                    ), workspace / "logs", 2
                )
            outside = orchestration.Step(
                key="outside", phase="render", kind="render",
                command=(sys.executable, str(run_eval)),
                output=root / "outside", metadata={},
            )
            with self.assertRaisesRegex(RuntimeError, "escapes evaluator"):
                orchestration._run_render_batch(
                    (outside,), SimpleNamespace(
                        repo=REPO, workspace=workspace,
                        build_dir=root / "build",
                    ), workspace / "logs", 2
                )
            wrong_command = orchestration.Step(
                key="wrong", phase="render", kind="render",
                command=(sys.executable, str(REPO / "wrong.py")),
                output=eval_root / "wrong", metadata={},
            )
            with self.assertRaisesRegex(RuntimeError, "non-run_eval"):
                orchestration._run_render_batch(
                    (wrong_command,), SimpleNamespace(
                        repo=REPO, workspace=workspace,
                        build_dir=root / "build",
                    ), workspace / "logs", 2
                )

    def test_parallel_render_failure_cancels_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            eval_root = root / "build" / "sbs_eval"
            workspace.mkdir()
            eval_root.mkdir(parents=True)
            run_eval = REPO / "tools" / "sbsbench" / "run_eval.py"
            steps = tuple(
                orchestration.Step(
                    key=f"render-{index}", phase="identity", kind="render",
                    command=(sys.executable, str(run_eval)),
                    output=eval_root / f"run-{index}",
                    metadata={},
                )
                for index in range(6)
            )
            cancelled = threading.Event()

            def fake_worker(step, plan, logs, cancel_event):
                if step.key == "render-0":
                    time.sleep(0.02)
                    raise RuntimeError("intentional failure")
                if cancel_event.wait(1.0):
                    cancelled.set()
                    raise orchestration._ParallelRenderCancelled(step.key)
                raise AssertionError("sibling render was not cancelled")

            with mock.patch.object(
                    orchestration, "_run_render_step_buffered",
                    side_effect=fake_worker), self.assertRaisesRegex(
                        RuntimeError, "intentional failure"):
                orchestration._run_render_batch(
                    steps, SimpleNamespace(
                        repo=REPO, workspace=workspace,
                        build_dir=root / "build",
                    ), workspace / "logs", 2
                )
            self.assertTrue(cancelled.is_set())

    def test_execute_batches_only_render_phases_and_records_plan_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            build = root / "build"
            workspace.mkdir()
            (build / "sbs_eval").mkdir(parents=True)
            specifications = (
                ("depth", "depth", "depth"),
                ("identity-a", "identity", "render"),
                ("identity-b", "identity", "render"),
                ("candidate-a", "render", "render"),
                ("candidate-b", "render", "render"),
                ("select", "select", "select"),
            )
            steps = tuple(
                orchestration.Step(
                    key=key, phase=phase, kind=kind,
                    command=(
                        (sys.executable, str(
                            REPO / "tools" / "sbsbench" / "run_eval.py"
                        )) if kind == "render" else ("ignored",)
                    ),
                    output=((build / "sbs_eval" / key)
                            if kind == "render" else workspace / key),
                    metadata={"identity": phase == "identity"},
                )
                for key, phase, kind in specifications
            )
            plan = SimpleNamespace(
                repo=REPO, workspace=workspace, build_dir=build,
                geometry_manifest_path=workspace / "geometry.json",
                geometry_manifest={"schema": 1},
                input_variant_manifest_path=workspace / "input.json",
                input_variant_manifest={"schema": 1}, steps=steps,
                as_dict=lambda: {"steps": [step.key for step in steps]},
            )
            args = SimpleNamespace(
                stop_after="select", restart=False, compact_renders=False,
                render_workers=2,
            )
            done = set()
            events = []

            def fake_run(step, plan, logs):
                done.add(step.key)

            def fake_batch(group, plan, logs, workers):
                events.append(f"batch:{group[0].phase}")
                done.update(step.key for step in reversed(group))

            def fake_screen(plan):
                events.append("screen")
                return {"blocked_splits": [], "blocked_conditions": []}

            with mock.patch.object(
                    orchestration, "step_complete",
                    side_effect=lambda step: step.key in done), \
                    mock.patch.object(
                        orchestration, "_run_step", side_effect=fake_run), \
                    mock.patch.object(
                        orchestration, "_run_render_batch",
                        side_effect=fake_batch), \
                    mock.patch.object(
                        orchestration, "identity_screen",
                        side_effect=fake_screen):
                result = orchestration.execute(plan, args)

            self.assertEqual(
                events[:3], ["batch:identity", "screen", "batch:render"]
            )
            state = json.loads((workspace / "orchestration_state.json").read_text(
                encoding="utf-8"
            ))
            self.assertEqual(state["completed"], [step.key for step in steps])
            self.assertEqual(result["completed_steps"], len(steps))


if __name__ == "__main__":
    unittest.main()
