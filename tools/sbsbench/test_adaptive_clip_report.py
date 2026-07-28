import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import adaptive_clip_report as report  # noqa: E402


def trace_header(**config_overrides):
    config = {
        "model": "depth_anything_v2_fp16",
        "profile": "apollo",
        "pop_strength": 1.2,
        "adaptive_pop": True,
        "adaptive_pop_max": 2.0,
        "zero_plane": "median",
        "depth_reuse_interval": 1,
    }
    config.update(config_overrides)
    return {
        "record": "header",
        "schema": report.TRACE_SCHEMA,
        "source": "depth_subject_resolve_cs.SubjectState",
        "capture": "every-source-frame-after-estimator-update",
        "fields": [
            {"name": name, "type": kind}
            for name, kind in report.FIELD_SPECS
        ],
        "analysis_flag_bits": report.ANALYSIS_FLAG_BITS,
        "config": config,
    }


def frame_record(
    source_index,
    *,
    age=0,
    flags=3,
    pulse=False,
    hard_cuts=0,
    external_cuts=0,
    empty_raw=0,
    collapsed_raw=0,
    ratio=1.0,
    edge=-1.0,
    change=0.02,
    anchor=2.5,
    initialized=True,
    anchor_valid=True,
    range_collapsed=False,
    depth_ready=True,
    depth_updated=True,
    valid_fraction=1.0,
):
    values = [
        0.0,
        float(age),
        0.55,
        1.0 if initialized else 0.0,
        0.05,
        1.1,
        0.08,
        float(ratio),
        float(anchor),
        1.0 if anchor_valid else 0.0,
        float(flags),
        1.0,
        float(edge),
        float(change),
        float(valid_fraction),
        0.4,
        int(hard_cuts),
        int(external_cuts),
        int(empty_raw),
        int(collapsed_raw),
        1.0 if range_collapsed else 0.0,
        1.0 if depth_ready else 0.0,
        1.0 if pulse else 0.0,
        0.0,
        float(edge),
        float(anchor),
        0.03,
        0.70,
        0.65,
        0.62,
        0.58,
        0,
    ]
    return {
        "record": "frame",
        "frame_id": f"{source_index + 1:010d}",
        "source_index": source_index,
        "depth_updated": depth_updated,
        "absolute_effective_pop": 1.2 * max(float(ratio), 1.0),
        "scene_camera_override": False,
        "resolved_zero_anchor_shift_px": float(anchor),
        "geometry_armed": bool(flags & 1),
        "appearance_armed": bool(flags & 2),
        "geometry_low_once": bool(flags & 4),
        "appearance_quiet_once": bool(flags & 8),
        "cut_latched": bool(flags & 16),
        "appearance_recovery": bool(flags & 32),
        "hard_cut_pulse": pulse,
        "hard_cut_count": hard_cuts,
        "external_cut_count": external_cuts,
        "empty_raw_count": empty_raw,
        "collapsed_raw_count": collapsed_raw,
        "values": values,
    }


def write_trace(path, frames, header=None):
    records = [header or trace_header(), *frames]
    with open(path, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def wrapper_timeline(frames, seconds_per_frame=0.1, first_pts=7.5):
    return {
        "schema": 1,
        "clock": "source-video-presentation",
        "time_base": {"num": 1, "den": 1000},
        "first_pts": round(first_pts * 1000),
        "first_pts_time": first_pts,
        "nominal_fps": 1.0 / seconds_per_frame,
        "variable_frame_rate": False,
        "frame_count": len(frames),
        "frames": [
            {
                "index": index,
                "frame_id": frame["frame_id"],
                "pts": round((first_pts + index * seconds_per_frame) * 1000),
                "pts_time": first_pts + index * seconds_per_frame,
                "pts_time_text": f"{first_pts + index * seconds_per_frame:.9f}",
                "duration": round(seconds_per_frame * 1000),
                "duration_time": seconds_per_frame,
            }
            for index, frame in enumerate(frames)
        ],
    }


class AdaptiveTraceContractTests(unittest.TestCase):
    def test_incremental_decoder_matches_full_trace_validation(self):
        header_value = trace_header()
        frames = [
            frame_record(0, age=1),
            frame_record(1, age=2),
        ]
        decoder = report.IncrementalTraceDecoder()
        self.assertIsNone(decoder.feed_line(json.dumps(header_value) + "\n"))
        decoded = [
            decoder.feed_line(json.dumps(frame) + "\n")
            for frame in frames
        ]
        self.assertEqual(decoder.finalize(), header_value)
        self.assertEqual(decoder.frame_count, 2)
        self.assertEqual(
            [frame["frame_id"] for frame in decoded],
            ["0000000001", "0000000002"],
        )
        with self.assertRaisesRegex(
                report.TraceContractError, "after finalization"):
            decoder.feed_line(json.dumps(frame_record(2)) + "\n")

    def test_rejects_duplicate_counter_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive_state.jsonl"
            frame = frame_record(0)
            frame["hard_cut_count"] = 1
            write_trace(path, [frame])
            with self.assertRaisesRegex(
                    report.TraceContractError, "hard_cut_count duplicate disagrees"):
                report.load_trace(path)

    def test_accepts_native_appearance_recovery_and_rejects_a_wrong_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive_state.jsonl"
            recovery = frame_record(0, flags=32)
            write_trace(path, [recovery])
            _header, frames = report.load_trace(path)
            self.assertTrue(frames[0]["appearance_recovery"])
            output = Path(directory) / "report"
            report.generate_outputs(
                path,
                output,
                wrapper_timeline([recovery]),
                "appearance-recovery",
            )
            with (output / "frames.csv").open(
                    newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["appearance_recovery"], "1")

            recovery["appearance_recovery"] = False
            write_trace(path, [recovery])
            with self.assertRaisesRegex(
                    report.TraceContractError,
                    "appearance_recovery duplicate disagrees"):
                report.load_trace(path)

    def test_rejects_unknown_cut_and_analysis_flag_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive_state.jsonl"
            unknown_cut = frame_record(0, flags=64)
            write_trace(path, [unknown_cut])
            with self.assertRaisesRegex(
                    report.TraceContractError, "cut_flags.*unknown schema bits"):
                report.load_trace(path)

            unknown_analysis = frame_record(0)
            unknown_analysis["values"][31] = 64
            write_trace(path, [unknown_analysis])
            with self.assertRaisesRegex(
                    report.TraceContractError,
                    "analysis_flags.*unknown schema bits"):
                report.load_trace(path)

    def test_rejects_nonfinite_float_and_wrong_header_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive_state.jsonl"
            frame = frame_record(0)
            frame["values"][0] = math.inf
            write_trace(path, [frame])
            with self.assertRaisesRegex(report.TraceContractError, "must be finite"):
                report.load_trace(path)

            header = trace_header()
            header["fields"][0]["type"] = "float64"
            write_trace(path, [frame_record(0)], header)
            with self.assertRaisesRegex(report.TraceContractError, "header contract mismatch"):
                report.load_trace(path)

    def test_rejects_duplicate_json_keys_and_nonincreasing_frame_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive_state.jsonl"
            header_text = json.dumps(trace_header(), separators=(",", ":"))
            frame_text = json.dumps(frame_record(0), separators=(",", ":"))
            path.write_text(
                header_text + "\n" +
                frame_text[:-1] + ',"frame_id":"0000000002"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(report.TraceContractError, "duplicate key"):
                report.load_trace(path)

            first = frame_record(0)
            second = frame_record(1)
            second["frame_id"] = first["frame_id"]
            write_trace(path, [first, second])
            with self.assertRaisesRegex(report.TraceContractError, "duplicate trace frame_id"):
                report.load_trace(path)

    def test_rejects_depth_update_outside_declared_reuse_cadence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive_state.jsonl"
            header = trace_header(depth_reuse_interval=3)
            frames = [
                frame_record(0, depth_updated=True),
                frame_record(1, depth_updated=True),
            ]
            write_trace(path, frames, header)
            with self.assertRaisesRegex(
                    report.TraceContractError, "disagrees with depth_reuse_interval=3"):
                report.load_trace(path)

    def test_accepts_unclamped_pop_then_flags_out_of_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "adaptive_state.jsonl"
            output = root / "report"
            frames = [frame_record(0, age=8, ratio=2.0, edge=0.01)]
            write_trace(trace, frames)

            header, parsed = report.load_trace(trace)
            self.assertEqual(parsed[0]["absolute_effective_pop"], 2.4)
            summary = report.generate_outputs(
                trace, output, wrapper_timeline(frames), "out-of-band")
            self.assertIn(
                "adaptive_pop_out_of_configured_range",
                summary["anomaly_counts"],
            )
            self.assertEqual(
                summary["parameter_policy"]["pop_ceiling_recommendation"],
                "not_inferred",
            )

    def test_first_frame_health_counters_keep_their_specific_cause(self):
        cases = (
            (
                "empty",
                frame_record(
                    0, initialized=False, depth_ready=False, empty_raw=1,
                    valid_fraction=0.0),
                "empty_raw_depth",
            ),
            (
                "collapsed",
                frame_record(
                    0, collapsed_raw=1, range_collapsed=True),
                "collapsed_raw_depth",
            ),
        )
        for label, frame, expected_kind in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trace = root / "adaptive_state.jsonl"
                write_trace(trace, [frame])
                summary = report.generate_outputs(trace, root / "out")
                self.assertIn(expected_kind, summary["anomaly_counts"])
                self.assertNotIn("nonzero_initial_counter", summary["anomaly_counts"])
                event = next(
                    item for item in summary["anomalies"]
                    if item["kind"] == expected_kind
                )
                self.assertEqual(event["frame_id"], "0000000001")
                self.assertEqual(event["count_delta"], 1)


class AdaptiveReportTests(unittest.TestCase):
    def test_segments_explicit_pulses_and_emits_endpoint_quantiles(self):
        frames = []
        hard_cuts = 0
        source_index = 0
        for shot in range(12):
            edge = 0.02 + shot * 0.01
            ratio = 1.0 + shot * 0.02
            for position in range(10):
                pulse = shot > 0 and position == 0
                if pulse:
                    hard_cuts += 1
                # Keep the first accepted cut's recovery fully disarmed through the next
                # accepted cut. The next pulse is therefore the important 16 -> 16 case.
                flags = 16 if shot in (1, 2) else (16 if pulse else 19)
                settled = position >= 8
                frames.append(frame_record(
                    source_index,
                    age=position,
                    flags=flags,
                    pulse=pulse,
                    hard_cuts=hard_cuts,
                    ratio=ratio if settled else 1.0,
                    edge=edge if settled else -1.0,
                    anchor=2.5 + shot * 0.01,
                ))
                source_index += 1

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "adaptive_state.jsonl"
            output = root / "report"
            write_trace(trace, frames)
            summary = report.generate_outputs(
                trace,
                output,
                timeline=wrapper_timeline(frames),
                source_name="twelve-shot.mp4",
            )

            self.assertEqual(summary["shot_count"], 12)
            self.assertEqual(summary["hard_cut_pulse_count"], 11)
            advisory = summary["pop_risk_endpoint_advisory"]
            self.assertEqual(advisory["status"], "advisory")
            self.assertEqual(advisory["classified_shots"], 12)
            self.assertAlmostEqual(advisory["pop_risk_low_q10"], 0.031)
            self.assertAlmostEqual(advisory["pop_risk_high_q90"], 0.119)
            self.assertEqual(summary["parameter_policy"], {
                "pop_floor_recommendation": "not_inferred",
                "pop_ceiling_recommendation": "not_inferred",
                "zero_plane_recommendation": "not_inferred",
                "reason": (
                    "Controller output is circular evidence for its own artistic/safety endpoints. "
                    "Use final-SBS artifact, comfort, and headset evidence."
                ),
            })
            for filename in ("frames.csv", "shots.csv", "summary.json", "report.html"):
                self.assertTrue((output / filename).is_file(), filename)
            with (output / "frames.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), len(frames))
            with (output / "shots.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 12)
            html_text = (output / "report.html").read_text(encoding="utf-8")
            self.assertIn("Host SBS whole-clip adaptive report", html_text)
            self.assertIn("Red dashed lines are accepted hard-cut pulses", html_text)

    def test_insufficient_shots_never_emits_endpoint_values(self):
        frames = [
            frame_record(index, age=index, edge=0.08 if index >= 8 else -1.0)
            for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "adaptive_state.jsonl"
            write_trace(trace, frames)
            summary = report.generate_outputs(trace, root / "out")
            advisory = summary["pop_risk_endpoint_advisory"]
            self.assertEqual(advisory["status"], "insufficient_shots")
            self.assertIsNone(advisory["pop_risk_low_q10"])
            self.assertIsNone(advisory["pop_risk_high_q90"])

    def test_reused_pulse_state_does_not_create_duplicate_shots(self):
        frames = [
            frame_record(0, age=7, flags=3, pulse=False, hard_cuts=0),
            frame_record(
                1, age=7, flags=3, pulse=False, hard_cuts=0, depth_updated=False),
            frame_record(
                2, age=7, flags=3, pulse=False, hard_cuts=0, depth_updated=False),
            frame_record(3, age=0, flags=16, pulse=True, hard_cuts=1),
            # --depth-every 3 reuses the exact GPU state on two intervening color frames.
            frame_record(
                4, age=0, flags=16, pulse=True, hard_cuts=1, depth_updated=False),
            frame_record(
                5, age=0, flags=16, pulse=True, hard_cuts=1, depth_updated=False),
            frame_record(6, age=1, flags=16, pulse=False, hard_cuts=1),
        ]
        header = trace_header(depth_reuse_interval=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "adaptive_state.jsonl"
            write_trace(trace, frames, header)
            summary = report.generate_outputs(trace, root / "out")
            self.assertEqual(summary["hard_cut_pulse_count"], 1)
            self.assertEqual(summary["shot_count"], 2)
            self.assertNotIn(
                "hard_cut_counter_pulse_mismatch",
                summary["anomaly_counts"],
            )

    def test_flags_health_drift_burst_and_disarmed_intervals(self):
        frames = []
        hard_cuts = 0
        for index in range(70):
            pulse = index in (10, 14)
            if pulse:
                hard_cuts += 1
            shot_age = index if index < 10 else (
                index - 10 if index < 14 else index - 14)
            ratio = 1.1 if index >= 23 else 1.0
            anchor = 2.7 if index >= 23 else 2.5
            frames.append(frame_record(
                index,
                age=shot_age,
                flags=16,
                pulse=pulse,
                hard_cuts=hard_cuts,
                ratio=ratio,
                edge=0.05 if shot_age >= 8 else -1.0,
                anchor=anchor,
                empty_raw=1 if index >= 25 else 0,
                collapsed_raw=1 if index >= 30 else 0,
                range_collapsed=index == 30,
                depth_ready=index != 40,
            ))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "adaptive_state.jsonl"
            write_trace(trace, frames)
            summary = report.generate_outputs(
                trace,
                root / "out",
                wrapper_timeline(frames, seconds_per_frame=0.05),
            )
            kinds = {item["kind"] for item in summary["anomalies"]}
            self.assertTrue({
                "cut_burst",
                "both_cut_arms_disarmed_long",
                "empty_raw_depth",
                "collapsed_raw_depth",
                "range_collapsed",
                "depth_not_ready_after_initialization",
                "post_settle_anchor_drift",
                "post_settle_adaptive_pop_drift",
            }.issubset(kinds))

    def test_wrapper_pts_are_normalized_but_absolute_pts_is_preserved(self):
        frames = [frame_record(index) for index in range(3)]
        timeline = wrapper_timeline(frames, seconds_per_frame=0.04, first_pts=11.25)
        joined, timing = report.join_timeline(
            [
                report._validate_frame(frame, index + 2, trace_header()["config"])
                for index, frame in enumerate(frames)
            ],
            timeline,
        )
        self.assertEqual(joined[0]["source_pts_seconds"], 11.25)
        self.assertEqual(joined[0]["timestamp_seconds"], 0.0)
        self.assertAlmostEqual(joined[2]["timestamp_seconds"], 0.08)
        self.assertEqual(timing["first_pts_seconds"], 11.25)
        self.assertEqual(timing["source"], "source-video-presentation")

    def test_timeline_rejects_mismatched_frame_id_even_when_index_matches(self):
        frames = [frame_record(index) for index in range(2)]
        parsed = [
            report._validate_frame(frame, index + 2, trace_header()["config"])
            for index, frame in enumerate(frames)
        ]
        timeline = wrapper_timeline(frames)
        timeline["frames"][0]["frame_id"] = "9999999999"
        with self.assertRaisesRegex(
                report.TraceContractError, "frame_id .* disagrees with trace frame"):
            report.join_timeline(parsed, timeline)


if __name__ == "__main__":
    unittest.main()
