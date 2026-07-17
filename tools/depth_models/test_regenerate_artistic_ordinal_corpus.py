#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sbsbench"))

import regenerate_artistic_ordinal_corpus as regeneration  # noqa: E402


class RegenerateArtisticOrdinalCorpusTests(unittest.TestCase):
    def test_unattended_defaults_use_fresh_workspace_and_measured_concurrency(self):
        args = regeneration.parse_args(["plan"])
        self.assertEqual(args.workspace.name, "ordinal-v2-target-only-v9")
        self.assertEqual(args.run_prefix, "ordv2-target-only-v9")
        self.assertEqual(args.score_workers, 8)
        self.assertEqual(
            args.scored_result_cache_root,
            regeneration.DEFAULT_SCORED_RESULT_CACHE_ROOT,
        )
        self.assertEqual(
            args.score_workers,
            regeneration.ordinal.run_multiscale_eval.DEFAULT_SCALE_SCORE_JOBS,
        )

        verify = regeneration.parse_args(["verify"])
        status = regeneration.parse_args(["status"])
        self.assertIsNone(verify.workers)
        self.assertIsNone(status.workers)

    def test_verify_workers_must_be_positive(self):
        with self.assertRaises(SystemExit):
            regeneration.parse_args([
                "verify", "--workers", "0",
            ])

    def _request(self, workspace, **updates):
        value = {
            "schema": regeneration.SCHEMA,
            "contract": regeneration.CONTRACT,
            "workspace": str(Path(workspace).resolve()),
            "active_split": str(Path(workspace) / "active.json"),
            "build_dir": str(Path(workspace) / "build"),
            "conf": str(Path(workspace) / "bench.conf"),
            "python": sys.executable,
            "run_prefix": "fixture",
            "production": [],
            "clip_limit": None,
            "score_workers": 3,
            "depth_state_cache_root": None,
            "scored_result_cache_root": None,
            "plan_sha256": "a" * 64,
            "plan_contract": "fixture-plan-v1",
            "scope": "complete-active-train-development",
            "training_eligible": True,
            "estimates": {"label_frames": 400},
        }
        value.update(updates)
        return value

    def _write_workspace(self, workspace, request=None, steps=None,
                         completed=None):
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        plan = {
            "schema": 1,
            "contract": "fixture-plan-v1",
            "steps": steps or [
                {"key": "safety-a"},
                {"key": "bundle-a"},
                {"key": "catalog"},
            ],
        }
        request = request or self._request(workspace)
        request["plan_sha256"] = regeneration.ordinal.canonical_sha256(plan)
        (workspace / regeneration.REQUEST_FILENAME).write_text(
            json.dumps(request), encoding="utf-8"
        )
        (workspace / regeneration.PLAN_FILENAME).write_text(
            json.dumps(plan), encoding="utf-8"
        )
        if completed is not None:
            (workspace / regeneration.STATE_FILENAME).write_text(
                json.dumps({
                    "schema": plan["schema"],
                    "contract": plan["contract"],
                    "plan_sha256": request["plan_sha256"],
                    "completed": completed,
                }), encoding="utf-8"
            )
        return request, plan

    @staticmethod
    def _fake_plan(workspace, value=None):
        workspace = Path(workspace).resolve()
        plan_value = value or {
            "schema": 1,
            "contract": "fixture-plan-v1",
            "scope": "complete-active-train-development",
            "training_eligible": True,
            "estimates": {"label_frames": 400},
            "steps": [{"key": "safety-a"}],
        }
        return SimpleNamespace(
            workspace=workspace,
            active_split=workspace / "active.json",
            build_dir=workspace / "build",
            conf=workspace / "bench.conf",
            python=Path(sys.executable),
            depth_state_cache_root=None,
            scored_result_cache_root=None,
            steps=tuple(),
            as_dict=lambda: plan_value,
        )

    def test_plan_checks_build_then_freezes_request_and_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            plan = self._fake_plan(workspace)
            args = argparse.Namespace(
                workspace=workspace,
                active_split=workspace / "active.json",
                build_dir=workspace / "build",
                conf=workspace / "bench.conf",
                python=Path(sys.executable),
                run_prefix="fixture",
                production=None,
                clip_limit=None,
                score_workers=4,
            )
            events = []

            with mock.patch.object(
                    regeneration.ordinal.run_eval, "require_current_build",
                    side_effect=lambda _path: events.append("build")), \
                    mock.patch.object(
                        regeneration.ordinal, "build_plan",
                        side_effect=lambda _args: (
                            events.append("plan") or plan
                        )):
                result = regeneration.command_plan(args)

            self.assertEqual(events, ["build", "plan"])
            request = json.loads((
                workspace / regeneration.REQUEST_FILENAME
            ).read_text(encoding="utf-8"))
            frozen_plan = json.loads((
                workspace / regeneration.PLAN_FILENAME
            ).read_text(encoding="utf-8"))
            self.assertEqual(frozen_plan, plan.as_dict())
            self.assertEqual(request["plan_sha256"], result["plan_sha256"])
            self.assertIn(
                f"--accept-plan-sha256 {result['plan_sha256']}",
                result["next_command"],
            )

    def test_plan_refuses_to_replace_a_different_frozen_request(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            path = workspace / regeneration.REQUEST_FILENAME
            path.write_text('{"old":true}', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "differs"):
                regeneration._freeze(
                    path, {"new": True}, "regeneration request"
                )
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), {"old": True}
            )

    def test_request_is_bound_to_the_workspace_containing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            other = Path(directory) / "other"
            request = self._request(workspace)
            request["workspace"] = str(other)
            self._write_workspace(
                workspace, request=request
            )

            with self.assertRaisesRegex(RuntimeError, "does not match"):
                regeneration._load_request(workspace)

    def test_run_rejects_wrong_digest_before_rebuilding_or_executing(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            self._write_workspace(workspace)
            args = argparse.Namespace(
                workspace=workspace,
                accept_plan_sha256="b" * 64,
                stop_after="catalog",
                repair_partials=False,
            )

            with mock.patch.object(regeneration, "_rebuild_plan") as rebuild, \
                    mock.patch.object(
                        regeneration.ordinal, "execute"
                    ) as execute, self.assertRaisesRegex(
                        RuntimeError, "does not match"):
                regeneration.command_run(args)

            rebuild.assert_not_called()
            execute.assert_not_called()

    def test_run_delegates_resume_and_repair_to_the_orchestrator(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            request, _plan_value = self._write_workspace(workspace)
            plan = self._fake_plan(workspace)
            args = argparse.Namespace(
                workspace=workspace,
                accept_plan_sha256=request["plan_sha256"],
                stop_after="sources",
                repair_partials=True,
            )

            with mock.patch.object(
                    regeneration, "_rebuild_plan", return_value=plan
                    ) as rebuild, mock.patch.object(
                        regeneration.ordinal, "execute",
                        return_value={"complete": True},
                    ) as execute:
                result = regeneration.command_run(args)

            self.assertEqual(result, {"complete": True})
            rebuild.assert_called_once_with(request, require_build=True)
            namespace = execute.call_args.args[1]
            self.assertEqual(namespace.stop_after, "sources")
            self.assertTrue(namespace.restart)
            self.assertEqual(namespace.render_workers, 1)
            runtime = json.loads((
                workspace / regeneration.RUNTIME_FILENAME
            ).read_text(encoding="utf-8"))
            self.assertEqual(runtime["state"], "completed")
            self.assertFalse(regeneration._run_lock_active(workspace))

    def test_status_is_read_only_and_reports_resume_position(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            self._write_workspace(workspace, completed=["safety-a"])
            logs = workspace / "orchestration_logs"
            logs.mkdir()
            recent = logs / "safety-a.log"
            recent.write_text("progress", encoding="utf-8")
            (workspace / "ordinal_frame_label_catalog.json").write_text(
                "{}", encoding="utf-8"
            )

            result = regeneration.command_status(argparse.Namespace(
                workspace=workspace, verify_completed=False
            ))

            self.assertEqual(result["completed_steps"], 1)
            self.assertEqual(result["total_steps"], 3)
            self.assertEqual(result["percent_complete"], 33.33)
            self.assertEqual(result["last_completed"], "safety-a")
            self.assertEqual(result["next_step"], "bundle-a")
            self.assertEqual(result["recent_log"], str(recent))
            self.assertIsNotNone(result["catalog"])
            self.assertEqual(result["execution_state"], "stopped")
            self.assertEqual(
                result["progress_basis"],
                "completed-plan-steps-unweighted",
            )
            self.assertIn("--accept-plan-sha256", result["resume_command"])

    def test_run_failure_is_visible_after_the_lock_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            request, _plan_value = self._write_workspace(workspace)
            plan = self._fake_plan(workspace)
            args = argparse.Namespace(
                workspace=workspace,
                accept_plan_sha256=request["plan_sha256"],
                stop_after="catalog", repair_partials=False,
            )

            with mock.patch.object(
                    regeneration, "_rebuild_plan", return_value=plan), \
                    mock.patch.object(
                        regeneration.ordinal, "execute",
                        side_effect=RuntimeError("fixture failure"),
                    ), self.assertRaisesRegex(RuntimeError, "fixture failure"):
                regeneration.command_run(args)

            result = regeneration.command_status(argparse.Namespace(
                workspace=workspace, verify_completed=False
            ))
            self.assertEqual(result["execution_state"], "failed")
            self.assertIn("fixture failure", result["last_failure"])

    def test_status_rejects_a_stale_state_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            self._write_workspace(workspace, completed=["safety-a"])
            state_path = workspace / regeneration.STATE_FILENAME
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["plan_sha256"] = "f" * 64
            state_path.write_text(json.dumps(state), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "state is stale"):
                regeneration.command_status(argparse.Namespace(
                    workspace=workspace, verify_completed=False
                ))

    def test_verify_completed_before_first_run_is_valid_and_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            _request, plan_value = self._write_workspace(workspace)
            step = SimpleNamespace(key="safety-a")
            plan = self._fake_plan(workspace, plan_value)
            plan.steps = (step,)

            with mock.patch.object(
                    regeneration, "_rebuild_plan", return_value=plan), \
                    mock.patch.object(
                        regeneration.ordinal, "_step_complete"
                    ) as complete:
                result = regeneration.command_status(argparse.Namespace(
                    workspace=workspace, verify_completed=True, workers=None
                ))

            self.assertEqual(result["verified_completed_steps"], 0)
            complete.assert_not_called()

    def test_completed_only_verification_does_not_probe_unstarted_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            _request, plan_value = self._write_workspace(
                workspace, completed=["safety-a"]
            )
            steps = tuple(
                SimpleNamespace(key=key)
                for key in ("safety-a", "bundle-a", "catalog")
            )
            plan = self._fake_plan(workspace, plan_value)
            plan.steps = steps
            args = argparse.Namespace(
                workspace=workspace,
                completed_only=True,
                workers=2,
            )

            with mock.patch.object(
                    regeneration, "_rebuild_plan", return_value=plan), \
                    mock.patch.object(
                        regeneration.ordinal, "_verify_sources"
                    ) as verify_sources, mock.patch.object(
                        regeneration, "_verify_step_results",
                        return_value=[True],
                    ) as verify_steps:
                result = regeneration.command_verify(args)

            verify_sources.assert_called_once_with(plan)
            self.assertEqual(
                [step.key for step in verify_steps.call_args.args[0]],
                ["safety-a"],
            )
            self.assertIs(verify_steps.call_args.args[1], plan)
            self.assertEqual(verify_steps.call_args.args[2], 2)
            self.assertTrue(result["verified"])
            self.assertEqual(result["verification_scope"], "completed-only")

    def test_verify_uses_frozen_score_workers_and_preserves_failure_order(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            _request, plan_value = self._write_workspace(workspace)
            plan = self._fake_plan(workspace, plan_value)
            plan.steps = tuple(
                SimpleNamespace(key=key)
                for key in ("first", "middle", "last")
            )
            events = []

            with mock.patch.object(
                    regeneration, "_rebuild_plan", return_value=plan), \
                    mock.patch.object(
                        regeneration.ordinal, "_verify_sources",
                        side_effect=lambda _plan: events.append("sources"),
                    ), mock.patch.object(
                        regeneration, "_verify_step_results",
                        side_effect=lambda steps, _plan, workers: (
                            events.append((
                                "steps", tuple(step.key for step in steps),
                                workers,
                            )) or [False, True, False]
                        ),
                    ), self.assertRaisesRegex(
                        RuntimeError,
                        r"incomplete/invalid \(2 steps\); first: first",
                    ):
                regeneration.command_verify(argparse.Namespace(
                    workspace=workspace, completed_only=False, workers=None
                ))

            self.assertEqual(events, [
                "sources", ("steps", ("first", "middle", "last"), 3),
            ])

    def test_source_verification_failure_prevents_step_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            _request, plan_value = self._write_workspace(workspace)
            plan = self._fake_plan(workspace, plan_value)
            plan.steps = (SimpleNamespace(key="safety-a"),)

            with mock.patch.object(
                    regeneration, "_rebuild_plan", return_value=plan), \
                    mock.patch.object(
                        regeneration.ordinal, "_verify_sources",
                        side_effect=RuntimeError("source hash mismatch"),
                    ), mock.patch.object(
                        regeneration, "_verify_step_results",
                    ) as verify_steps, self.assertRaisesRegex(
                        RuntimeError, "source hash mismatch"):
                regeneration.command_verify(argparse.Namespace(
                    workspace=workspace, completed_only=False, workers=2
                ))

            verify_steps.assert_not_called()

    def test_complete_verification_fails_closed_on_missing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            _request, plan_value = self._write_workspace(workspace)
            plan = self._fake_plan(workspace, plan_value)
            plan.steps = tuple(
                SimpleNamespace(key=key)
                for key in ("safety-a", "bundle-a")
            )
            args = argparse.Namespace(
                workspace=workspace,
                completed_only=False,
                workers=3,
            )

            with mock.patch.object(
                    regeneration, "_rebuild_plan", return_value=plan), \
                    mock.patch.object(
                        regeneration.ordinal, "_verify_sources"
                    ) as verify_sources, mock.patch.object(
                        regeneration, "_verify_step_results",
                        return_value=[True, False],
                    ), self.assertRaisesRegex(
                        RuntimeError, "incomplete/invalid.*bundle-a"):
                regeneration.command_verify(args)

            verify_sources.assert_called_once_with(plan)

    def test_parallel_step_verification_is_bounded_exact_and_ordered(self):
        steps = tuple(
            SimpleNamespace(key=key) for key in ("slow", "fast", "last")
        )

        class WorkingPlan:
            @property
            def sealed_test_production_ids(self):
                raise AssertionError("parallel verification read sealed tests")

        plan = WorkingPlan()
        lock = threading.Lock()
        active = 0
        maximum_active = 0
        calls = []

        def complete(step, received_plan):
            nonlocal active, maximum_active
            self.assertIs(received_plan, plan)
            with lock:
                calls.append(step.key)
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep({"slow": 0.06, "fast": 0.01, "last": 0.02}[step.key])
            with lock:
                active -= 1
            return step.key != "fast"

        with mock.patch.object(
                regeneration.ordinal, "_step_complete",
                side_effect=complete):
            results = regeneration._verify_step_results(steps, plan, 2)

        self.assertEqual(results, [True, False, True])
        self.assertCountEqual(calls, [step.key for step in steps])
        self.assertEqual(len(calls), len(steps))
        self.assertEqual(maximum_active, 2)

    def test_parallel_step_verification_reports_plan_order_failure(self):
        steps = tuple(
            SimpleNamespace(key=key) for key in ("first", "second")
        )
        plan = SimpleNamespace()

        def fail(step, _plan):
            if step.key == "first":
                time.sleep(0.05)
            raise AssertionError(f"{step.key}-failure")

        with mock.patch.object(
                regeneration.ordinal, "_step_complete", side_effect=fail), \
                self.assertRaisesRegex(AssertionError, "first-failure"):
            regeneration._verify_step_results(steps, plan, 2)

    def test_parallel_step_verification_validates_workers_before_work(self):
        steps = (SimpleNamespace(key="safety-a"),)
        with mock.patch.object(
                regeneration.ordinal, "_step_complete") as complete, \
                self.assertRaisesRegex(ValueError, "workers"):
            regeneration._verify_step_results(steps, SimpleNamespace(), 0)
        complete.assert_not_called()

    def test_parallel_step_verification_empty_selection_does_no_work(self):
        with mock.patch.object(
                regeneration.ordinal, "_step_complete") as complete:
            result = regeneration._verify_step_results(
                (), SimpleNamespace(), 4
            )
        self.assertEqual(result, [])
        complete.assert_not_called()

    def test_workspace_lock_rejects_a_second_runner_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            digest = "a" * 64

            with regeneration._exclusive_run_lock(workspace, digest):
                self.assertTrue(regeneration._run_lock_active(workspace))
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with regeneration._exclusive_run_lock(workspace, digest):
                        pass

            self.assertFalse(regeneration._run_lock_active(workspace))

    def test_stale_running_runtime_is_reported_as_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            request, _plan = self._write_workspace(workspace)
            regeneration._write_runtime(
                workspace, request["plan_sha256"], "running",
                current_step="safety-a", current_phase="safety",
            )

            result = regeneration.command_status(argparse.Namespace(
                workspace=workspace, verify_completed=False
            ))

            self.assertEqual(result["execution_state"], "interrupted")
            self.assertEqual(result["current_step"], "safety-a")
            self.assertFalse(result["lock_active"])


if __name__ == "__main__":
    unittest.main()
