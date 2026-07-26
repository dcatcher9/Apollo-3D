import argparse
import ast
import builtins
import io
import os
import sys
import types
import unittest
from unittest import mock


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import generate_report  # noqa: E402


def _build_report_functions(*names):
    path = os.path.join(SCRIPT_DIR, "build_report.py")
    with open(path, encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=path)
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "argparse": argparse,
        "REPORT_SCORING_JOBS_ENV": "SBSBENCH_REPORT_SCORING_JOBS",
        "REPORT_SCORING_MAX_JOBS": 16,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, "exec"), namespace)
    return tuple(namespace[name] for name in names)


class ReportParallelIntegrationTests(unittest.TestCase):
    def test_launcher_scoring_jobs_defaults_bounded_and_is_position_independent(self):
        with mock.patch.object(os, "cpu_count", return_value=24):
            self.assertEqual(generate_report.report_scoring_jobs(
                ["control", "treatment", "report.html"]), 8)
        self.assertEqual(generate_report.report_scoring_jobs(
            ["control", "--scoring-jobs", "3", "treatment", "report.html"]), 3)
        self.assertEqual(generate_report.report_scoring_jobs(
            ["control", "treatment", "report.html", "--scoring-jobs=5"]), 5)
        for invalid in ("0", "-1", "17", "many"):
            with self.subTest(invalid=invalid), mock.patch(
                    "sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                generate_report.report_scoring_jobs(["--scoring-jobs", invalid])

    def test_guarded_launcher_sets_process_backend_and_job_handoff_before_import(self):
        observed = {}
        real_import = builtins.__import__

        def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "build_report":
                observed["backend"] = os.environ.get("SBSBENCH_SPATIAL_BACKEND")
                observed["jobs"] = os.environ.get(
                    generate_report.REPORT_SCORING_JOBS_ENV)
                observed["argv"] = list(sys.argv)
                return types.ModuleType("build_report")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch.object(
                sys, "argv", ["generate_report.py", "control", "--scoring-jobs", "4",
                              "treatment", "report.html"]), \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(builtins, "__import__", side_effect=recording_import):
            generate_report.main()

        self.assertEqual(observed, {
            "backend": "process",
            "jobs": "4",
            "argv": ["generate_report.py", "control", "treatment", "report.html"],
        })

    def test_builder_resolves_cli_then_launcher_environment(self):
        positive_count, report_jobs = _build_report_functions(
            "_positive_job_count", "_report_scoring_jobs")
        self.assertEqual(positive_count("2"), 2)
        self.assertIsNone(report_jobs([], {}))
        self.assertEqual(report_jobs(
            [], {"SBSBENCH_REPORT_SCORING_JOBS": "6"}), 6)
        self.assertEqual(report_jobs(
            ["--scoring-jobs", "3"], {"SBSBENCH_REPORT_SCORING_JOBS": "6"}), 3)
        with self.assertRaisesRegex(SystemExit, "SBSBENCH_REPORT_SCORING_JOBS"):
            report_jobs([], {"SBSBENCH_REPORT_SCORING_JOBS": "0"})
        with self.assertRaisesRegex(SystemExit, "SBSBENCH_REPORT_SCORING_JOBS"):
            report_jobs([], {"SBSBENCH_REPORT_SCORING_JOBS": "17"})

    def test_builder_forwards_scoring_jobs_to_authoritative_verification(self):
        (validate_results,) = _build_report_functions("_validate_authoritative_results")
        artifact_runtime = mock.Mock()
        artifact_runtime.verify_results_against_artifacts.return_value = {
            "passed": True, "clips": ["clip"], "frame_count": 1}
        namespace = validate_results.__globals__
        namespace["run_eval"] = artifact_runtime
        namespace["THRESHOLD_CFG"] = {"metrics": {}}
        run = {"meta": {}, "clips": {"clip": {}}}
        session = object()

        validate_results(
            run, "run", "control", "clips",
            remeasurement_session=session, scoring_jobs=4)

        artifact_runtime.verify_results_against_artifacts.assert_called_once_with(
            run, "run", "clips", {"metrics": {}},
            remeasurement_session=session, scoring_jobs=4)

    def test_builder_omits_unset_jobs_and_preserves_direct_thread_fallback(self):
        (validate_results,) = _build_report_functions("_validate_authoritative_results")
        artifact_runtime = mock.Mock()
        artifact_runtime.verify_results_against_artifacts.return_value = {
            "passed": True, "clips": ["clip"], "frame_count": 1}
        namespace = validate_results.__globals__
        namespace["run_eval"] = artifact_runtime
        namespace["THRESHOLD_CFG"] = {"metrics": {}}
        run = {"meta": {}, "clips": {"clip": {}}}

        validate_results(run, "run", "control", "clips")

        artifact_runtime.verify_results_against_artifacts.assert_called_once_with(
            run, "run", "clips", {"metrics": {}}, remeasurement_session=None)
        with open(os.path.join(SCRIPT_DIR, "build_report.py"), encoding="utf-8") as stream:
            report_source = stream.read()
        self.assertIn('if __name__ == "__main__":', report_source)
        self.assertIn(
            'os.environ[sbsbench.SEQUENCE_SPATIAL_BACKEND_ENV] = "thread"',
            report_source)


if __name__ == "__main__":
    unittest.main()
