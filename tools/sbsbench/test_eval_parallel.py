import copy
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbsbench  # noqa: E402
import eval_parallel  # noqa: E402


class EvalClipParallelTests(unittest.TestCase):
    def setUp(self):
        sbsbench.disable_reusable_spatial_executor()

    def tearDown(self):
        sbsbench.disable_reusable_spatial_executor()

    def _write_sequence_fixture(self, root, name, offset):
        sequence = os.path.join(root, name, "artifacts")
        sources = os.path.join(root, name, "sources")
        os.makedirs(sequence)
        os.makedirs(sources)
        yy, xx = np.mgrid[:24, :32]
        for frame_id in range(8):
            source = np.stack((
                (xx * 5 + frame_id * 7 + offset) % 256,
                (yy * 9 + xx * 2 + frame_id * 3) % 256,
                ((xx + yy) * 4 + frame_id * 11) % 256,
            ), axis=2).astype(np.uint8)
            sbs = np.concatenate((source, source), axis=1)
            Image.fromarray(sbs, "RGB").save(
                os.path.join(sequence, f"sbs_{frame_id:05d}.png"))
            Image.fromarray(source, "RGB").save(
                os.path.join(sources, f"frame_{frame_id:05d}.png"))
        with open(os.path.join(sources, "meta.json"), "w", encoding="utf-8") as stream:
            json.dump({}, stream)
        return sequence, sources

    def test_clip_worker_count_is_positive_bounded_and_clip_limited(self):
        self.assertEqual(eval_parallel.clip_scoring_worker_count(4, 13), 4)
        self.assertEqual(eval_parallel.clip_scoring_worker_count(4, 2), 2)
        self.assertEqual(eval_parallel.clip_scoring_worker_count(4, 0), 0)

        for invalid in (
                0, -1, True, "4", eval_parallel.CLIP_SCORING_MAX_WORKERS + 1):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ValueError, "clip scoring jobs"):
                eval_parallel.clip_scoring_worker_count(invalid, 2)
        for invalid_count in (-1, True, 1.5):
            with self.subTest(invalid_count=invalid_count), self.assertRaisesRegex(
                    ValueError, "clip count"):
                eval_parallel.clip_scoring_worker_count(1, invalid_count)

    def test_ordered_clip_pool_overlaps_work_without_mutating_jobs(self):
        sequence_jobs = [
            ("first", "slow", "source-a"),
            ("second", "fast", "source-b"),
            ("third", "medium", "source-c"),
        ]
        original_jobs = copy.deepcopy(sequence_jobs)
        delays = {"slow": 0.06, "fast": 0.01, "medium": 0.03}
        state_lock = threading.Lock()
        active = 0
        maximum_active = 0

        def fake_measure(seq_dir, frames_dir, compact=False):
            self.assertTrue(compact)
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(delays[seq_dir])
                return ([{"sequence": seq_dir}], {"source": frames_dir})
            finally:
                with state_lock:
                    active -= 1

        with mock.patch.object(
                sbsbench, "enable_reusable_spatial_executor"), mock.patch.object(
                    sbsbench, "measure_sequence", side_effect=fake_measure):
            measured = eval_parallel.measure_clip_sequences(sequence_jobs, jobs=3)

        self.assertEqual(sequence_jobs, original_jobs)
        self.assertGreaterEqual(maximum_active, 2)
        self.assertEqual([clip for clip, _result in measured],
                         ["first", "second", "third"])
        self.assertEqual(
            [result[1]["source"] for _clip, result in measured],
            ["source-a", "source-b", "source-c"])

    def test_jobs_one_preserves_serial_input_order(self):
        calls = []

        def fake_measure(seq_dir, frames_dir, compact=False):
            self.assertTrue(compact)
            calls.append((seq_dir, frames_dir))
            return ([{"sequence": seq_dir}], {"source": frames_dir})

        sequence_jobs = [
            ("first", "artifact-a", "source-a"),
            ("second", "artifact-b", "source-b"),
        ]
        with mock.patch.object(
                sbsbench, "enable_reusable_spatial_executor") as enable, mock.patch.object(
                    sbsbench, "measure_sequence", side_effect=fake_measure):
            measured = eval_parallel.measure_clip_sequences(sequence_jobs, jobs=1)

        enable.assert_not_called()
        self.assertEqual(calls, [
            ("artifact-a", "source-a"),
            ("artifact-b", "source-b"),
        ])
        self.assertEqual([clip for clip, _result in measured], ["first", "second"])

    def test_real_process_spatial_pool_is_equivalent_under_clip_threads(self):
        with tempfile.TemporaryDirectory() as root:
            first = self._write_sequence_fixture(root, "first", 0)
            second = self._write_sequence_fixture(root, "second", 19)
            sequence_jobs = [
                ("first", first[0], first[1]),
                ("second", second[0], second[1]),
            ]
            with mock.patch.dict(os.environ, {
                    sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV: "1",
                    sbsbench.SEQUENCE_SPATIAL_BACKEND_ENV: "process"}):
                serial = [
                    (clip, sbsbench.measure_sequence(sequence, sources))
                    for clip, sequence, sources in sequence_jobs
                ]

            sbsbench.disable_reusable_spatial_executor()
            with mock.patch.dict(os.environ, {
                    sbsbench.SEQUENCE_SPATIAL_WORKERS_ENV: "2",
                    sbsbench.SEQUENCE_SPATIAL_BACKEND_ENV: "process"}):
                parallel = eval_parallel.measure_clip_sequences(sequence_jobs, jobs=2)

        self.assertEqual(parallel, serial)

    def test_first_failure_is_deterministic_by_input_order_and_names_clip(self):
        both_started = threading.Barrier(2)

        def fail_out_of_order(seq_dir, _frames_dir, compact=False):
            self.assertTrue(compact)
            both_started.wait(timeout=2)
            if seq_dir == "first-artifacts":
                time.sleep(0.04)
                raise ValueError("first failure")
            raise ValueError("later failure")

        sequence_jobs = [
            ("first-clip", "first-artifacts", "source-a"),
            ("later-clip", "later-artifacts", "source-b"),
        ]
        with mock.patch.object(
                sbsbench, "enable_reusable_spatial_executor"), mock.patch.object(
                    sbsbench, "measure_sequence", side_effect=fail_out_of_order):
            with self.assertRaisesRegex(
                    RuntimeError, "first-clip.*first failure") as raised:
                eval_parallel.measure_clip_sequences(sequence_jobs, jobs=2)

        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_parallel_pool_is_ready_before_clip_threads_start(self):
        events = []
        state_lock = threading.Lock()

        def fake_enable():
            with state_lock:
                events.append(("enable", threading.current_thread().name))

        def fake_measure(seq_dir, frames_dir, compact=False):
            self.assertTrue(compact)
            with state_lock:
                events.append((seq_dir, threading.current_thread().name))
            return ([{"sequence": seq_dir}], {"source": frames_dir})

        sequence_jobs = [
            ("first", "artifact-a", "source-a"),
            ("second", "artifact-b", "source-b"),
        ]
        with mock.patch.object(
                sbsbench, "enable_reusable_spatial_executor",
                side_effect=fake_enable), mock.patch.object(
                    sbsbench, "measure_sequence", side_effect=fake_measure):
            measured = eval_parallel.measure_clip_sequences(sequence_jobs, jobs=2)

        self.assertEqual(events[0][0], "enable")
        self.assertEqual(events[0][1], threading.current_thread().name)
        self.assertTrue(all(name.startswith("sbsbench-clip")
                            for _event, name in events[1:]))
        self.assertEqual([clip for clip, _result in measured], ["first", "second"])

    def test_malformed_jobs_fail_before_starting_workers(self):
        malformed = [
            ("missing-source", "artifact"),
            ("", "artifact", "source"),
        ]
        for job in malformed:
            with self.subTest(job=job), mock.patch.object(
                    sbsbench, "enable_reusable_spatial_executor") as enable:
                with self.assertRaisesRegex(ValueError, "clip scoring job"):
                    eval_parallel.measure_clip_sequences([job], jobs=2)
                enable.assert_not_called()


if __name__ == "__main__":
    unittest.main()
