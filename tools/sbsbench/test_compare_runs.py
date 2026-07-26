"""Fail-closed provenance tests for the SBS run comparator."""

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import compare_runs


class CompareRunsTests(unittest.TestCase):
    @staticmethod
    def _run(value=90.0, content_type="real-capture"):
        return {
            "meta": {
                "clip_set_sha1": {"clip": "0123456789ab"},
                "eval_schema": compare_runs.run_eval.EVAL_SCHEMA,
                "suite": "core",
                "mode": "profile",
                "run_kind": "comparison-only",
                "metric_sha256": compare_runs.run_eval.metric_contract_sha(),
                "label_contract_sha256": compare_runs.run_eval.label_contract_sha(),
                "metric_runtime": compare_runs.run_eval.metric_runtime_provenance(),
            },
            "clips": {
                "clip": {
                    "aggregate": {"exact_binocular_support_pct": value},
                    "meta": {"content_type": content_type},
                },
            },
        }

    def _main(self, control, treatment, *extra):
        with tempfile.TemporaryDirectory() as temporary:
            control_dir = os.path.join(temporary, "control")
            treatment_dir = os.path.join(temporary, "treatment")
            os.makedirs(control_dir)
            os.makedirs(treatment_dir)
            with open(os.path.join(control_dir, "results.json"), "w",
                      encoding="utf-8") as stream:
                json.dump(control, stream)
            with open(os.path.join(treatment_dir, "results.json"), "w",
                      encoding="utf-8") as stream:
                json.dump(treatment, stream)
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = ["compare_runs.py", control_dir, treatment_dir, *extra]
            with mock.patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = compare_runs.main()
            return status, stdout.getvalue(), stderr.getvalue()

    def test_current_compatible_runs_compare(self):
        status, stdout, stderr = self._main(
            self._run(90.0), self._run(91.0),
            "--metrics", "exact_binocular_support_pct")
        self.assertEqual(status, 0, stderr)
        self.assertIn("real-capture(1)", stdout)
        self.assertIn("exact_binocular_support_pct", stdout)

    def test_equal_but_stale_provenance_is_rejected(self):
        for key, stale in (
                ("eval_schema", compare_runs.run_eval.EVAL_SCHEMA - 1),
                ("metric_sha256", "stale-metric"),
                ("label_contract_sha256", "stale-label"),
                ("metric_runtime", {"python": "different"})):
            with self.subTest(key=key):
                control = self._run()
                treatment = self._run()
                control["meta"][key] = stale
                treatment["meta"][key] = copy.deepcopy(stale)
                status, _, stderr = self._main(control, treatment)
                self.assertEqual(status, 2)
                self.assertIn(key, stderr)
                self.assertIn("stale or incompatible", stderr)

    def test_shared_run_headers_must_be_present_and_current_contract_values(self):
        for key, malformed in (("mode", None), ("suite", "future-suite"),
                               ("run_kind", "ad-hoc")):
            with self.subTest(key=key):
                control = self._run()
                treatment = self._run()
                control["meta"][key] = malformed
                treatment["meta"][key] = malformed
                status, _, stderr = self._main(control, treatment)
                self.assertEqual(status, 2)
                self.assertIn(key, stderr)

    def test_unknown_requested_metric_is_an_error(self):
        status, stdout, stderr = self._main(
            self._run(), self._run(), "--metrics", "definitely_not_a_metric")
        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("unknown requested metric", stderr)

    def test_content_classification_must_be_explicit_and_equal(self):
        status, _, stderr = self._main(
            self._run(content_type="synthetic"), self._run(content_type="real-capture"))
        self.assertEqual(status, 2)
        self.assertIn("content_type differs", stderr)

        control = self._run()
        del control["clips"]["clip"]["meta"]["content_type"]
        status, _, stderr = self._main(control, self._run())
        self.assertEqual(status, 2)
        self.assertIn("no explicit content_type", stderr)

    def test_unknown_content_classification_is_rejected(self):
        status, _, stderr = self._main(
            self._run(content_type="capture"), self._run(content_type="capture"))
        self.assertEqual(status, 2)
        self.assertIn("content_type 'capture' is not one of", stderr)

    def test_unclassified_content_is_excluded_from_the_decisive_aggregate(self):
        status, stdout, stderr = self._main(
            self._run(90.0, content_type="unclassified"),
            self._run(91.0, content_type="unclassified"),
            "--metrics", "exact_binocular_support_pct")
        self.assertEqual(status, 0, stderr)
        self.assertIn("NOT DECISIVE (unclassified content)", stdout)
        self.assertNotIn("ALL CLASSIFIED NON-PROBE", stdout)
        self.assertNotIn("worst clip:", stdout)

    def test_clip_entries_must_match_authenticated_clip_set(self):
        treatment = self._run()
        treatment["clips"]["extra"] = copy.deepcopy(treatment["clips"]["clip"])
        status, _, stderr = self._main(self._run(), treatment)
        self.assertEqual(status, 2)
        self.assertIn("clips do not match clip_set_sha1", stderr)

    def test_clip_hashes_must_be_lowercase_12_hex(self):
        for digest in ("0123456789a", "0123456789AB", "0123456789az", 123456789012):
            with self.subTest(digest=digest):
                treatment = self._run()
                treatment["meta"]["clip_set_sha1"]["clip"] = digest
                control = copy.deepcopy(treatment)
                status, _, stderr = self._main(control, treatment)
                self.assertEqual(status, 2)
                self.assertIn("invalid lowercase 12-hex", stderr)

    def test_malformed_clip_entry_is_rejected_without_traceback(self):
        treatment = self._run()
        treatment["clips"]["clip"] = None
        status, _, stderr = self._main(self._run(), treatment)
        self.assertEqual(status, 2)
        self.assertIn("clip 'clip' is not an object", stderr)


if __name__ == "__main__":
    unittest.main()
