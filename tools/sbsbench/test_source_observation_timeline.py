"""Authored source cadence and explicit timeline admission; no wall-clock source."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval
import source_observation_timeline as timeline


class SourceObservationTimelineTests(unittest.TestCase):
    def test_fractional_cadence_uses_integer_source_times_without_accumulation(self):
        self.assertEqual(timeline.prepared_observation_timestamps(4, "30000/1001"),
                         [1, 33367, 66734, 100101])
        self.assertEqual(timeline.prepared_observation_timestamps(3, "30/1"),
                         [1, 33334, 66667])

    def test_roundtrip_preserves_repeated_source_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.timeline"
            timeline.write_observation_timeline(path, [1, 1, 50001])
            self.assertEqual(timeline.read_observation_timeline(path), [1, 1, 50001])

    def test_invalid_or_regressed_wire_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.timeline"
            for values in ([], [0], [2, 1], [True], [1 << 64]):
                with self.subTest(values=values), self.assertRaises(timeline.TimelineError):
                    timeline.write_observation_timeline(path, values)
            valid = struct.pack("<8sIIQ2Q", b"SBSOTL1\0", 1, 24, 2, 1, 2)
            for payload in (valid[:-1], valid + b"x", valid.replace(b"SBSOTL1", b"INVALID"),
                            struct.pack("<8sIIQ2Q", b"SBSOTL1\0", 1, 24, 2, 2, 1)):
                path.write_bytes(payload)
                with self.subTest(payload=payload), self.assertRaises(timeline.TimelineError):
                    timeline.read_observation_timeline(path)

    @staticmethod
    def native_contract(record):
        return {"observation_timeline": {
            "schema": 1, "timestamp_unit": "monotonic-source-us-plus-one",
            "count": record["count"], "sha256": record["sha256"]}}

    def test_default_cadence_is_declared_authored_30hz_not_measured_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            record = run_eval.prepare_clip_observation_timeline(path, 3, {})
            self.assertEqual(record["time_provenance"], {
                "kind": "authored-prepared-source-cadence", "authored_cadence": "30/1",
                "cadence_source": "declared-evaluator-default-30Hz"})
            run_eval.validate_run_observation_timeline(
                path, {"observation_timeline_artifacts": {"clip": record}}, "clip",
                self.native_contract(record), 3, {})

    def test_authored_cadence_must_match_authenticated_source_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            record = run_eval.prepare_clip_observation_timeline(path, 3, {"fps": 24})
            run_eval.validate_run_observation_timeline(
                path, {"observation_timeline_artifacts": {"clip": record}}, "clip",
                self.native_contract(record), 3, {"fps": 24})
            for metadata in ({"fps": 30}, {}):
                with self.subTest(metadata=metadata), self.assertRaisesRegex(
                        ValueError, "authored source cadence"):
                    run_eval.validate_run_observation_timeline(
                        path, {"observation_timeline_artifacts": {"clip": record}}, "clip",
                        self.native_contract(record), 3, metadata)

    def test_explicit_clock_is_preserved_and_bound_to_native_attestation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "custom.timeline"
            timeline.write_observation_timeline(source, [7, 400007])
            record = run_eval.prepare_clip_observation_timeline(path, 2, {}, source)
            self.assertEqual((path / record["file"]).read_bytes(), source.read_bytes())
            run_eval.validate_run_observation_timeline(
                path, {"observation_timeline_artifacts": {"clip": record}}, "clip",
                self.native_contract(record), 2, {})
            bad = copy.deepcopy(self.native_contract(record))
            bad["observation_timeline"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "native source observation"):
                run_eval.validate_run_observation_timeline(
                    path, {"observation_timeline_artifacts": {"clip": record}}, "clip", bad, 2, {})
            with self.assertRaisesRegex(ValueError, "count differs"):
                run_eval.prepare_clip_observation_timeline(path, 3, {}, source)

    def test_removed_selectors_and_ambiguous_clock_overrides_are_rejected(self):
        for token in ("--joint-plane-mode=3", "--joint-plane-experiment=on"):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "removed"):
                run_eval.validate_unified_policy_options([token])
        for extra in (["--observation-timeline"],
                      ["--observation-timeline", "a", "--observation-timeline", "b"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                run_eval.observation_timeline_override(extra)

    def test_missing_explicit_clock_is_a_setup_failure_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.timeline"
            command = [sys.executable, "-B", str(Path(run_eval.__file__)),
                       "--comparison-only", "--clips", "flat_page",
                       "--extra", "--observation-timeline", str(path)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("invalid source observation timeline", result.stderr)
            self.assertIn(repr(str(path)), result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_unreadable_explicit_clock_retains_the_path_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unreadable.timeline"
            error = PermissionError(13, "Permission denied", str(path))
            with mock.patch.object(run_eval, "read_observation_timeline", side_effect=error):
                with self.assertRaisesRegex(
                        ValueError, "invalid source observation timeline") as caught:
                    run_eval.observation_timeline_override(["--observation-timeline", str(path)])
            self.assertIs(caught.exception.__cause__, error)
            self.assertIn(repr(str(path)), str(caught.exception))

    def test_same_pixels_with_different_clocks_fail_every_comparison_context(self):
        import compare_runs
        from test_compare_runs import CompareRunsTests
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index, timestamps in enumerate(([1, 33334], [1, 100001])):
                payload = timeline.observation_timeline_payload(timestamps)
                records.append({"file": "observation.timeline", "schema": 1, "count": 2,
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "time_provenance": {"kind": "explicit-source-timeline",
                                                    "authored_cadence": None}})
            control = CompareRunsTests._run()
            treatment = copy.deepcopy(control)
            for run, record in zip((control, treatment), records):
                run["meta"]["observation_timeline_artifacts"] = {"clip": record}
            self.assertIn("observation_timeline_artifacts", compare_runs.compatibility_error(
                control, treatment))
            self.assertNotEqual(run_eval.label_context_sha(control["meta"]),
                                run_eval.label_context_sha(treatment["meta"]))
            baseline_meta = {**run_eval.baseline_required_context(control["meta"]),
                             "extra_args": [], "git_dirty": False, "clip_sha1": "same-pixels"}
            baseline = {"meta": baseline_meta, "aggregate": {}, "perf_ms": {}}
            run_eval._validate_baseline_manifest(baseline, "clip", "authored test",
                run_eval.baseline_required_context(control["meta"]), {"clip": "same-pixels"})
            with self.assertRaisesRegex(ValueError, "observation_timeline_artifacts"):
                run_eval._validate_baseline_manifest(baseline, "clip", "authored test",
                    run_eval.baseline_required_context(treatment["meta"]), {"clip": "same-pixels"})
            paths = []
            for name, run in (("control", control), ("treatment", treatment)):
                path = root / name
                path.mkdir()
                (path / "results.json").write_text(json.dumps(run))
                paths.append(path)
            report = root / "must-not-exist.html"
            result = subprocess.run([sys.executable, "-B", str(Path(run_eval.SCRIPT_DIR) /
                "build_report.py"), *(str(path) for path in paths), str(report),
                "--allow-executable-diff", "--allow-config-diff"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("observation_timeline_artifacts", result.stderr)
            self.assertFalse(report.exists())

    def test_fps_drift_changes_source_identity_even_when_pixels_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frame_00001.png").write_bytes(b"same authored source bytes")
            (root / "meta.json").write_text(json.dumps({"fps": 30}))
            control = run_eval.source_evidence_digests(str(root))
            (root / "meta.json").write_text(json.dumps({"fps": 10}))
            self.assertNotEqual(control, run_eval.source_evidence_digests(str(root)))


if __name__ == "__main__":
    unittest.main()
