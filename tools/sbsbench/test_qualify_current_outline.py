#!/usr/bin/env python3
"""Independent provenance/ABI reader tests; these fixtures do not model rendering."""

import copy
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import depth_coordinate_v2_contract as coordinate  # noqa: E402
import depth_coordinate_v2_dump_contract as dump_contract  # noqa: E402
import generate_depth_coordinate_v2_contract as generator  # noqa: E402
import qualify_current_outline as qualification  # noqa: E402
from source_observation_timeline import write_observation_timeline  # noqa: E402


def float_word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def state_words(clock, scale):
    contract = coordinate.load_contract()
    fields = contract["shadow_state"]["fields"]
    values = {field["name"]: 0 for field in fields}
    tag = generator.contract_tag(contract)
    values.update({
        "center": 2.0, "inverse_scale": 1.0 / scale,
        "container_scale": 1.0, "calibration_revision": 1,
        "frame_valid": 1.0, "contract_tag_bits": tag,
        "renderer_authorization_bits": tag, "joint_plane_mode_bits": 3,
        "gain_last_observation_low": clock & 0xFFFFFFFF,
        "gain_last_observation_high": clock >> 32,
        "gain_clock_armed": 1, "gain_seed_count": 1,
        "gain_target_zero": 2.0, "gain_target_inverse_scale": 1.0 / scale,
        "gain_target_nearest": scale,
        "gain_display_limit": coordinate.CALIBRATED_DEFAULTS.direct_container_limit,
        "gain_seed_first_low": clock & 0xFFFFFFFF,
        "gain_seed_first_high": clock >> 32,
        "gain_seed_last_low": clock & 0xFFFFFFFF,
        "gain_seed_last_high": clock >> 32,
        "gain_seed_mean_nearest": scale, "gain_seed_mean_zero": 2.0,
    })
    for field in fields:
        if field["gpu_encoding"] == "float":
            name = field["name"]
            values[name] = struct.unpack("<f", struct.pack("<f", values[name]))[0]
    values["camera_center_integrity_bits"] = dump_contract.camera_center_integrity_for_state_values(values)
    return [float_word(values[field["name"]]) if field["gpu_encoding"] == "float"
            else values[field["name"]] for field in fields]


class CurrentOutlineQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.closures = qualification.expected_closures()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frames = self.root / "frames"
        self.frames.mkdir()
        self.output = self.root / "output"
        (self.output / "current-outline").mkdir(parents=True)
        self.executable = self.root / "sunshine.exe"
        self.executable.write_bytes(b"fixture native executable identity")
        self.timeline = self.root / "requested.timeline"
        self.clocks = [1000001, 1033334]
        write_observation_timeline(self.timeline, self.clocks)
        timeline = self.output / "current-outline/observation.timeline"
        timeline.write_bytes(self.timeline.read_bytes())
        self.calibration = coordinate.MODEL_CALIBRATIONS[0]
        policy = coordinate.load_contract()["subtitle_ocr"]["contour_policy"]
        width, height, field_width, field_height = 960, 540, 1540, 868
        crop = 160
        roi = coordinate.subtitle_ocr_dynamic_roi(width, height, field_width, field_height)
        self.assertIsNotNone(roi)
        frames = []
        for ordinal, clock in enumerate(self.clocks, 1):
            source_id = f"{ordinal:05d}"
            source = self.frames / f"frame_{source_id}.png"
            Image.new("RGB", (width, height), (ordinal, 2, 3)).save(source)
            png = self.output / f"current-outline/sbs_{source_id}.png"
            Image.new("RGB", (2 * width, height), (ordinal, 4, 5)).save(png)
            token = 100 + ordinal
            geometry = [width, height, height - crop, crop, field_width, field_height,
                        *roi, 0, 0, field_width, field_height]
            contour = [policy["history_schema"], policy["history_tag"], 1, 0,
                       ordinal, 0, 0, 0, *geometry, clock, 0, policy["recipe"], 1, *([0] * 8)]
            selected = [0x31534D53, 1, 3, 0, *geometry, ordinal, 0, 0, 0, clock, 0,
                        0, 0, 1, policy["recipe"], 0, 0, *contour, token, 0, 2, 0]
            render = [1, 0, 0, 0, 0, 0, width, height]
            expected = [ordinal, 0, 17, 0, 19, 0, 23, 0, 29, 0, 31, 0, 37, 0, 41, 0]
            proof = [0] * 96
            proof[:20] = [0x31504742, 1, 1, 0, *expected]
            proof[20:52] = contour
            proof[52:64] = [*render, token, 0, 1, 0]
            proof[64:72] = [width, height, width, crop, height - crop, 0, 0, 0]
            proof[77], proof[80], proof[81] = width * crop, width * crop, 1
            frames.append({
                "observation_ordinal": ordinal, "source_file": source.name, "source_file_id": source_id,
                "source_snapshot_sha256": qualification._sha(source), "source_size": [width, height],
                "output_file": png.relative_to(self.output).as_posix(), "output_sha256": qualification._sha(png),
                "output_size": [2 * width, height], "completed_frame_id": ordinal,
                "observation_timestamp_us": clock, "transaction_token": token, "analysis_generation": 0,
                "bindings_observed": True, "background_serial": ordinal, "selected_receipt": selected,
                "contour_identity": contour,
                "condition_params": [coordinate.SUBTITLE_OCR.condition_param_schema,
                                     coordinate.SUBTITLE_OCR.condition_param_tag, 1, 0, 2, float_word(0.01)],
                "parallax_state": state_words(clock, self.calibration.raw_coordinate_scale),
                "subtitle_constants": render, "background_proof": proof, "background_expected": expected,
                "publication_receipt": {"estimator_generation": 7, "publication_sequence": ordinal,
                                        "domain_tag": 9, "expected_work": 1, "depth": 2,
                                        "depth_owner_frame_id": ordinal, "depth_owner_timestamp_us": clock,
                                        "subtitle": 1, "subtitle_frame_id": ordinal, "subtitle_domain_tag": 9},
            })
        self.document = {
            "schema": 1, "status": "COMPLETE", "role": qualification.ROLE,
            "ordinary_metric_artifacts": "unchanged-legacy-native-matte-renderer", "live_capture": False,
            "scheduling_authority": "ordered-authored-full-source-observations", "source_transfer": "encoded-srgb",
            "frame_count": len(frames), "executable_sha256": qualification._sha(self.executable),
            "coordinate_contract_schema": coordinate.CONTRACT_SCHEMA,
            "coordinate_contract_sha256": coordinate.CONTRACT_CANONICAL_SHA256,
            "producer_source_closure_sha256": coordinate.SHADER_IMPLEMENTATION.source_closure_sha256,
            **self.closures,
            "observation_timeline": ["current-outline/observation.timeline", qualification._sha(timeline)],
            "model": self.calibration.depth_model, "model_url": self.calibration.depth_model_url,
            "onnx_sha256": self.calibration.onnx_sha256, "raw_coordinate_scale": self.calibration.raw_coordinate_scale,
            "frames": frames,
        }

    def validate(self, document=None, **kwargs):
        (self.output / qualification.MANIFEST).write_text(json.dumps(document or self.document), encoding="utf-8")
        return qualification.validate_qualification(self.output, self.frames, self.executable, **kwargs)

    def test_current_selected_proof_and_source_sequence_are_qualified(self):
        result = self.validate(timeline=self.timeline)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["selected_current_frames"], 2)
        self.assertEqual(result["background_valid_frames"], 2)
        self.assertFalse(result["ordinary_metrics_qualify_selected_composite"])

    def test_rejects_corrupt_identity_receipts_and_hashes(self):
        changes = {
            "source snapshot": lambda d: d["frames"][0].update(source_snapshot_sha256="0" * 64),
            "output digest": lambda d: d["frames"][0].update(output_sha256="0" * 64),
            "source identity": lambda d: d["frames"][0].update(source_file_id="00002"),
            "frame identity": lambda d: d["frames"][0].update(completed_frame_id=2),
            "observation clock": lambda d: d["frames"][0].update(observation_timestamp_us=42),
            "selected owner": lambda d: d["frames"][0]["selected_receipt"].__setitem__(16, 2),
            "selected clock": lambda d: d["frames"][0]["selected_receipt"].__setitem__(20, 42),
            "selected token": lambda d: d["frames"][0]["selected_receipt"].__setitem__(60, 42),
            "selected reserved": lambda d: d["frames"][0]["selected_receipt"].__setitem__(63, 1),
            "selected ROI": lambda d: d["frames"][0]["selected_receipt"].__setitem__(10, 1),
            "selected field": lambda d: d["frames"][0]["selected_receipt"].__setitem__(8, 868),
            "SCT body": lambda d: d["frames"][0]["contour_identity"].__setitem__(24, 1),
            "inactive SCP payload": lambda d: d["frames"][0]["condition_params"].__setitem__(4, 0),
            "active SCP without covers": lambda d: d["frames"][0]["condition_params"].__setitem__(2, 0),
            "background lease": lambda d: d["frames"][0]["background_expected"].__setitem__(4, 41),
            "background token": lambda d: d["frames"][0]["background_proof"].__setitem__(60, 42),
            "background SCT": lambda d: d["frames"][0]["background_proof"].__setitem__(24, 2),
            "background completion": lambda d: d["frames"][0]["background_proof"].__setitem__(80, 1),
            "publication owner": lambda d: d["frames"][0]["publication_receipt"].update(depth_owner_frame_id=2),
            "publication reused": lambda d: d["frames"][0]["publication_receipt"].update(depth=1),
            "stale producer": lambda d: d["frames"][1]["publication_receipt"].update(estimator_generation=8),
            "duplicate sequence": lambda d: d["frames"][1]["publication_receipt"].update(publication_sequence=1),
            "missing binding observation": lambda d: d["frames"][0].update(bindings_observed=False),
            "boolean word": lambda d: d["frames"][0]["selected_receipt"].__setitem__(4, True),
            "boolean frame": lambda d: d["frames"][0].update(completed_frame_id=True),
            "boolean publication": lambda d: d["frames"][0]["publication_receipt"].update(depth_owner_frame_id=True),
            "wrong shape": lambda d: d["frames"][0]["selected_receipt"].pop(),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                document = copy.deepcopy(self.document)
                change(document)
                with self.assertRaises(ValueError):
                    self.validate(document)

    def test_rejects_contract_scope_and_frame_coverage_changes(self):
        for key, value in {
            "schema": True, "executable_sha256": "0" * 64, "completed_source_closure_sha256": "0" * 64,
            "selector_source_closure_sha256": "0" * 64, "background_source_closure_sha256": "0" * 64,
            "producer_source_closure_sha256": "0" * 64, "coordinate_contract_sha256": "0" * 64,
            "model": "unrecognized", "live_capture": True, "ordinary_metric_artifacts": "selected-renderer",
            "frame_count": 1, "frames": [],
        }.items():
            with self.subTest(key=key):
                document = copy.deepcopy(self.document)
                document[key] = value
                with self.assertRaises(ValueError):
                    self.validate(document)

    def test_rejects_artifact_mutation_extra_output_and_timeline_substitution(self):
        source = self.frames / "frame_00001.png"
        Image.new("RGB", tuple(self.document["frames"][0]["source_size"]), "white").save(source)
        with self.assertRaisesRegex(ValueError, "source snapshot"):
            self.validate()
        self.document["frames"][0]["source_snapshot_sha256"] = qualification._sha(source)
        png = self.output / "current-outline/sbs_00001.png"
        png.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "output hash"):
            self.validate()
        Image.new("RGB", tuple(self.document["frames"][0]["output_size"])).save(png)
        self.document["frames"][0]["output_sha256"] = qualification._sha(png)
        Image.new("RGB", (2, 2)).save(self.output / "current-outline/sbs_99999.png")
        with self.assertRaisesRegex(ValueError, "frame set"):
            self.validate()
        (self.output / "current-outline/sbs_99999.png").unlink()
        write_observation_timeline(self.timeline, [1000002, 1033335])
        with self.assertRaisesRegex(ValueError, "requested observations"):
            self.validate(timeline=self.timeline)

    def test_accepts_native_float32_direct_limit_and_rejects_next_float(self):
        for frame in self.document["frames"]:
            frame["condition_params"][5] = float_word(coordinate.CALIBRATED_DEFAULTS.direct_container_limit)
        self.assertEqual(self.validate()["status"], "PASS")
        self.document["frames"][0]["condition_params"][5] += 1
        with self.assertRaisesRegex(ValueError, "condition parameters"):
            self.validate()

    def test_rejects_truncated_source_even_with_matching_rewritten_digest(self):
        source = self.frames / "frame_00001.png"
        source.write_bytes(source.read_bytes()[:64])
        with Image.open(source) as image:
            self.assertEqual(image.size, tuple(self.document["frames"][0]["source_size"]))
        self.document["frames"][0]["source_snapshot_sha256"] = qualification._sha(source)
        with self.assertRaises(OSError):
            self.validate()

    def test_empty_current_background_and_unavailable_fallback_remain_explicit(self):
        empty, fallback = self.document["frames"]
        empty["contour_identity"][2] = 0
        empty["selected_receipt"][30] = 0
        empty["condition_params"][2:5] = [0, 0, 0]
        empty["condition_params"][5] = 0
        empty["background_proof"][2] = 2
        empty["background_proof"][22] = 0
        empty["background_proof"][80:83] = [0, 0, 0]
        fallback["selected_receipt"][2:4] = [1, 1]
        fallback["selected_receipt"][28:60] = [0] * 32
        fallback["background_proof"][2:4] = [0, 1]
        result = self.validate()
        self.assertEqual(result["selected_current_frames"], 1)
        self.assertEqual(result["background_valid_frames"], 0)
        fallback["selected_receipt"][2] = 3
        with self.assertRaisesRegex(ValueError, "unavailable selected receipt"):
            self.validate()

    def test_rejects_path_escape_unknown_fields_and_duplicate_json_fields(self):
        for path in ("../outside.png", "C:/outside.png", "current-outline\\sbs_00001.png"):
            with self.subTest(path=path):
                document = copy.deepcopy(self.document)
                document["frames"][0]["output_file"] = path
                with self.assertRaises(ValueError):
                    self.validate(document)
        document = copy.deepcopy(self.document)
        document["unrecognized"] = 1
        with self.assertRaisesRegex(ValueError, "unknown or missing"):
            self.validate(document)
        (self.output / qualification.MANIFEST).write_text('{"schema":1,"schema":1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate qualification field"):
            qualification.validate_qualification(self.output, self.frames, self.executable)

    def test_source_discovery_rejects_ambiguous_noncanonical_and_duplicate_ids(self):
        for name in ("frame_1.png", "frame_extra.png", "frame_00003.PNG", "frame_00003.jpg"):
            with self.subTest(name=name):
                path = self.frames / name
                Image.new("RGB", (2, 2)).save(path)
                try:
                    with self.assertRaises(ValueError):
                        qualification.source_frames(self.frames)
                finally:
                    path.unlink()

    def test_launcher_invokes_explicit_native_selected_route(self):
        build = self.root / "build"
        build.mkdir()
        (build / "sunshine.exe").write_bytes(self.executable.read_bytes())
        target = self.root / "launch-output"
        with mock.patch.object(qualification.subprocess, "run") as run, \
                mock.patch.object(qualification, "validate_qualification", return_value={"status": "PASS"}) as validate, \
                mock.patch.object(qualification.mask_metrics, "score_qualification", return_value={"aggregate": None}):
            self.assertEqual(qualification.main(["--frames", str(self.frames), "--out", str(target),
                                                 "--build-dir", str(build)]), 0)
        command = run.call_args.args[0]
        self.assertIn("--current-outline-qualification", command)
        self.assertIn("--observation-timeline", command)
        self.assertEqual(command[0], str((build / "sunshine.exe").resolve()))
        self.assertEqual(run.call_args.kwargs, {"cwd": build.resolve(), "check": True})
        self.assertEqual(validate.call_args.args[:3], (target, self.frames, (build / "sunshine.exe").resolve()))


if __name__ == "__main__":
    unittest.main()
