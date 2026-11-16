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
        "pop_strength": 1.2,
        "depth_reuse_interval": 1,
    }
    config.update(config_overrides)
    return {
        "record": "header",
        "schema": report.TRACE_SCHEMA,
        "source": report.TRACE_SOURCE,
        "capture": report.TRACE_CAPTURE,
        "fields": [
            {"name": name, "type": kind}
            for name, kind in report.FIELD_SPECS
        ],
        "config": config,
    }


def frame_record(
    source_index,
    *,
    empty_raw=0,
    collapsed_raw=0,
    range_collapsed=False,
    depth_ready=True,
    depth_updated=True,
    valid_fraction=1.0,
):
    values_by_name = {
        name: report.CONTRACT_TAG if initial == "contract_tag" else initial
        for name, initial in report.INITIAL_VALUES.items()
    }
    values_by_name.update({
        "valid_depth_fraction": float(valid_fraction),
        "effective_raw_range_width": 0.4,
        "empty_raw_count": int(empty_raw),
        "collapsed_raw_count": int(collapsed_raw),
        "range_collapsed": 1.0 if range_collapsed else 0.0,
        "depth_ready": 1.0 if depth_ready else 0.0,
    })
    values = [
        int(values_by_name[name])
        if encoding in {"uint_bits", "uint_valued_float"}
        else float(values_by_name[name])
        for (name, _), encoding in zip(report.FIELD_SPECS, report.FIELD_ENCODINGS)
    ]
    return {
        "record": "frame",
        "frame_id": f"{source_index + 1:010d}",
        "source_index": source_index,
        "depth_updated": depth_updated,
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
            frame_record(0),
            frame_record(1),
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
        self.assertEqual(set(decoded[0]), {
            "frame_id", "source_index", "depth_updated", *report.FIELD_NAMES})
        with self.assertRaisesRegex(
                report.TraceContractError, "after finalization"):
            decoder.feed_line(json.dumps(frame_record(2)) + "\n")

    def test_rejects_duplicate_counter_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive_state.jsonl"
            for counter in ("empty_raw_count", "collapsed_raw_count"):
                with self.subTest(counter=counter):
                    frame = frame_record(0)
                    frame[counter] = 1
                    write_trace(path, [frame])
                    with self.assertRaisesRegex(
                            report.TraceContractError,
                            f"{counter} duplicate disagrees"):
                        report.load_trace(path)

    def test_rejects_nonzero_reserved_word(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive_state.jsonl"
            reserved = frame_record(0)
            reserved["values"][report.FIELD_NAMES.index("reserved_3")] = 1
            write_trace(path, [reserved])
            with self.assertRaisesRegex(
                    report.TraceContractError,
                    "reserved field reserved_3.*non-default"):
                report.load_trace(path)

    def test_rejects_frame_and_header_keys_outside_the_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive_state.jsonl"
            frame = frame_record(0)
            frame["unknown_key"] = False
            write_trace(path, [frame])
            with self.assertRaisesRegex(report.TraceContractError, "exact keys"):
                report.load_trace(path)

            header = trace_header()
            header["analysis_flag_bits"] = {}
            write_trace(path, [frame_record(0)], header)
            with self.assertRaisesRegex(report.TraceContractError, "exact keys"):
                report.load_trace(path)

    def test_rejects_nonfinite_float_and_wrong_header_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive_state.jsonl"
            frame = frame_record(0)
            frame["values"][report.FIELD_NAMES.index("valid_depth_fraction")] = math.inf
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

            first = frame_record(0)
            first["frame_id"] = "1"
            second = frame_record(1)
            second["frame_id"] = "00001"
            write_trace(path, [first, second])
            with self.assertRaisesRegex(report.TraceContractError, "duplicate trace frame_id"):
                report.load_trace(path)

    def test_rejects_same_sized_foreign_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive_state.jsonl"
            frame = frame_record(0)
            frame["values"][0] ^= 1
            write_trace(path, [frame])
            with self.assertRaisesRegex(
                    report.TraceContractError, "contract tag"):
                report.load_trace(path)

    def test_trace_frame_id_rejects_non_ascii_decimal_and_uint64_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive_state.jsonl"
            for invalid_id, message in (
                ("1x", "ASCII decimal string"),
                ("１", "ASCII decimal string"),
                (str(report.UINT64_MAX + 1), "uint64 range"),
            ):
                with self.subTest(frame_id=invalid_id):
                    frame = frame_record(0)
                    frame["frame_id"] = invalid_id
                    write_trace(path, [frame])
                    with self.assertRaisesRegex(report.TraceContractError, message):
                        report.load_trace(path)

    def test_rejects_retired_depth_reuse_cadence(self):
        with self.assertRaisesRegex(report.TraceContractError, "exactly 1"):
            report._validate_header(trace_header(depth_reuse_interval=2))

    def test_first_frame_health_counters_keep_their_specific_cause(self):
        cases = (
            (
                "empty",
                frame_record(
                    0, depth_ready=False, empty_raw=1, valid_fraction=0.0),
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
                event = next(
                    item for item in summary["anomalies"]
                    if item["kind"] == expected_kind
                )
                self.assertEqual(event["frame_id"], "0000000001")
                self.assertEqual(event["count_delta"], 1)


class AdaptiveReportTests(unittest.TestCase):
    def test_emits_depth_health_report(self):
        frames = [
            frame_record(index, valid_fraction=0.5 if index == 7 else 1.0)
            for index in range(120)
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "adaptive_state.jsonl"
            output = root / "report"
            write_trace(trace, frames)
            summary = report.generate_outputs(
                trace,
                output,
                timeline=wrapper_timeline(frames),
                source_name="healthy.mp4",
            )

            self.assertEqual(summary["schema"], 2)
            self.assertEqual(summary["frame_count"], 120)
            self.assertEqual(summary["depth_update_count"], 120)
            self.assertEqual(summary["min_valid_depth_fraction"], 0.5)
            self.assertEqual(summary["anomaly_count"], 0)
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["frames.csv", "report.html", "summary.json"])
            with (output / "frames.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(frames))
            self.assertEqual(tuple(rows[0]), report.FRAME_CSV_FIELDS)
            html_text = (output / "report.html").read_text(encoding="utf-8")
            self.assertIn("Host SBS whole-clip depth-health report", html_text)
            self.assertIn("Valid depth fraction", html_text)

    def test_flags_health_counters_collapse_and_lost_readiness(self):
        frames = [
            frame_record(
                index,
                empty_raw=1 if index >= 25 else 0,
                collapsed_raw=1 if index >= 30 else 0,
                range_collapsed=index == 30,
                depth_ready=index != 40,
            )
            for index in range(70)
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "adaptive_state.jsonl"
            write_trace(trace, frames)
            summary = report.generate_outputs(
                trace,
                root / "out",
                wrapper_timeline(frames, seconds_per_frame=0.05),
            )
            self.assertEqual(summary["anomaly_counts"], {
                "collapsed_raw_depth": 1,
                "depth_not_ready_after_ready": 1,
                "empty_raw_depth": 1,
                "range_collapsed": 1,
            })
            lost = next(
                item for item in summary["anomalies"]
                if item["kind"] == "depth_not_ready_after_ready")
            self.assertEqual(lost["frame_id"], "0000000041")

    def test_flags_delayed_and_absent_depth_readiness(self):
        delayed = [frame_record(index, depth_ready=index >= 3) for index in range(6)]
        never = [frame_record(index, depth_ready=False) for index in range(6)]
        for label, frames, expected in (
            ("delayed", delayed, {"depth_ready_delay": 1}),
            ("never", never, {"depth_never_ready": 1}),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trace = root / "adaptive_state.jsonl"
                write_trace(trace, frames)
                summary = report.generate_outputs(trace, root / "out")
                self.assertEqual(summary["anomaly_counts"], expected)

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

    def test_timeline_frame_ids_match_by_numeric_value_across_zero_padding(self):
        frames = [frame_record(index) for index in range(2)]
        parsed = [
            report._validate_frame(frame, index + 2, trace_header()["config"])
            for index, frame in enumerate(frames)
        ]
        timeline = wrapper_timeline(frames)
        timeline["frames"][0]["frame_id"] = "00001"
        timeline["frames"][1]["frame_id"] = "2"

        joined, _timing = report.join_timeline(parsed, timeline)

        self.assertEqual(
            [frame["frame_id"] for frame in joined],
            ["0000000001", "0000000002"],
        )

    def test_timeline_frame_ids_reject_malformed_overflow_and_numeric_duplicates(self):
        frame = frame_record(0)
        parsed = [
            report._validate_frame(frame, 2, trace_header()["config"])
        ]
        for invalid_id, message in (
            ("1x", "ASCII decimal string"),
            ("１", "ASCII decimal string"),
            (str(report.UINT64_MAX + 1), "uint64 range"),
        ):
            with self.subTest(frame_id=invalid_id):
                timeline = wrapper_timeline([frame])
                timeline["frames"][0]["frame_id"] = invalid_id
                with self.assertRaisesRegex(report.TraceContractError, message):
                    report.join_timeline(parsed, timeline)

        frames = [frame_record(index) for index in range(2)]
        parsed = [
            report._validate_frame(item, index + 2, trace_header()["config"])
            for index, item in enumerate(frames)
        ]
        timeline = wrapper_timeline(frames)
        timeline["frames"][0]["frame_id"] = "1"
        timeline["frames"][1]["frame_id"] = "00001"
        with self.assertRaisesRegex(
                report.TraceContractError, "invalid/duplicate frame_id"):
            report.join_timeline(parsed, timeline)

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
