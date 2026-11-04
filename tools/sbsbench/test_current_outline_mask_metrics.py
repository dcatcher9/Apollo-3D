#!/usr/bin/env python3
"""Literal-mask provenance and independently authored count/transition arithmetic."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import current_outline_mask_metrics as metrics  # noqa: E402
import qualify_current_outline as qualification  # noqa: E402
import test_qualify_current_outline as reader_fixtures  # noqa: E402


class CurrentOutlineMaskArithmeticTests(unittest.TestCase):
    def test_precision_recall_iou_and_missing_denominators(self):
        predicted = np.array([[1, 1, 0, 0], [0, 0, 1, 0]], dtype=bool)
        truth = np.array([[1, 0, 1, 0], [0, 0, 1, 0]], dtype=bool)
        counts = metrics.count_mask(predicted, truth)
        self.assertEqual(counts, {"predicted_pixels": 3, "ground_truth_pixels": 3,
                                  "true_positive_pixels": 2, "false_positive_pixels": 1,
                                  "false_negative_pixels": 1})
        self.assertEqual(metrics.ratios(counts), {"precision": 2 / 3, "recall": 2 / 3, "iou": 0.5})
        empty = np.zeros_like(predicted)
        self.assertEqual(metrics.ratios(metrics.count_mask(empty, empty)),
                         {"precision": None, "recall": None, "iou": None})
        self.assertEqual(metrics.ratios(metrics.count_mask(empty, truth)),
                         {"precision": None, "recall": 0, "iou": 0})
        self.assertEqual(metrics.ratios(metrics.count_mask(predicted, empty)),
                         {"precision": 0, "recall": None, "iou": 0})

    def test_temporal_excludes_rgb_motion_and_ground_truth_replacement(self):
        previous_gt = np.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
        current_gt = np.array([[1, 1, 1, 0], [0, 0, 0, 0]], dtype=bool)
        previous_mask = np.zeros_like(previous_gt)
        current_mask = np.array([[1, 0, 1, 1], [1, 1, 0, 0]], dtype=bool)
        previous_rgb = np.zeros((2, 4, 3), dtype=np.uint8)
        current_rgb = previous_rgb.copy()
        current_rgb[1, 0, 0] = 1
        counts = metrics.count_temporal(previous_mask, current_mask, previous_rgb, current_rgb, previous_gt, current_gt)
        self.assertEqual(counts, {"all_eligible_pixels": 6, "all_changed_pixels": 3,
                                  "glyph_eligible_pixels": 2, "glyph_changed_pixels": 1,
                                  "background_eligible_pixels": 4, "background_changed_pixels": 2})
        self.assertEqual(metrics.temporal_ratios(counts), {"all_flicker_fraction": 0.5,
                         "glyph_flicker_fraction": 0.5, "background_flicker_fraction": 0.5})
        changed_rgb = previous_rgb + 1
        none = metrics.count_temporal(previous_mask, current_mask, previous_rgb, changed_rgb, previous_gt, current_gt)
        self.assertEqual(metrics.temporal_ratios(none), {"all_flicker_fraction": None,
                         "glyph_flicker_fraction": None, "background_flicker_fraction": None})


class CurrentOutlineMaskEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.closures = qualification.expected_closures()

    def setUp(self):
        self.reader = reader_fixtures.CurrentOutlineQualificationTests("test_current_selected_proof_and_source_sequence_are_qualified")
        self.reader.closures = self.closures
        self.reader.setUp()
        self.addCleanup(self.reader.doCleanups)
        self.output, self.frames, self.executable = self.reader.output, self.reader.frames, self.reader.executable
        self.document = self.reader.document
        self.document["schema"] = 2
        self.gt_directory = self.frames / metrics.GT_DIRECTORY
        self.gt_directory.mkdir()
        (self.frames / "meta.json").write_text(json.dumps({"name": "literal-mask-test", "fps": "30/1"}), encoding="utf-8")
        for index, frame in enumerate(self.document["frames"]):
            selected = frame["selected_receipt"]
            width, height, top, crop = selected[4:8]
            values = np.zeros((crop, width), dtype="<u4")
            values[0, :2] = 1
            name = f"current-outline/selected_matte_{frame['source_file_id']}.u32"
            path = self.output / name
            path.write_bytes(values.tobytes())
            frame["selected_mask"] = {"file": name, "sha256": metrics.sha256(path), "size": [width, crop],
                                      "source_rect": [0, top, width, height], "byte_count": values.nbytes,
                                      "format": metrics.MASK_FORMAT}
            truth = np.zeros((height, width), dtype=np.uint8)
            if index == 1:
                truth[top, 0], truth[top, 2] = 255, 255
            Image.fromarray(truth).save(self.gt_directory / frame["source_file"])

    def save(self):
        (self.output / qualification.MANIFEST).write_text(json.dumps(self.document), encoding="utf-8")

    def score(self):
        self.save()
        return metrics.score_qualification(self.output, self.frames, self.executable, self.reader.timeline)

    def upgrade_to_schema3(self):
        self.document["schema"] = 3
        self.document["composition_source_closures"] = qualification.expected_composition_closures()
        self.document["hole_program_ready"] = True
        self.document["coverage_program_ready"] = True
        for frame in self.document["frames"]:
            frame["hole_program_ready"] = True
            frame["coverage_program_ready"] = True
        for frame in self.document["frames"]:
            c24 = frame["selected_receipt"][4:28]
            cbc, raw, f2 = ([0] * count for count in (512, 512, 68608))
            cbc[:4] = [0x31434E45, 5, 1, 0]
            cbc[4] = cbc[5] = cbc[9] = 1
            cbc[16:40] = c24
            selected = frame["selected_receipt"]
            width, height, _, band = selected[4:8]
            cbc[40:44] = [0x31434E45, 1, 1, 2]
            cbc[44:52] = [width, band, 28, 1, width, height, 28, 1]
            cbc[52:56] = [*selected[60:62], 1, 1 ^ 0x6F435257]
            cbc[56:60] = [*selected[16:18], *selected[20:22]]
            cbc[60:64] = [*selected[18:20], 0, 0]
            cbc[64:72] = [0, 0, width, height, 5, 0, 4, 136]
            cbc[80] = frame["contour_identity"][3]
            raw[:4] = [0x31474E52, 1, 1, 0]
            raw[12:36], raw[36:52] = c24, cbc[:16]
            f2[:4] = [0x37424643, 7, 1, 0]
            f2[8:32], f2[32:48] = c24, raw[:16]
            directory = self.output / f"current-outline/producer-stages/frame_{frame['source_file_id']}"
            directory.mkdir(parents=True)
            buffers = {}
            for (name, (words, role, dimension, view_format, stride)), data in zip(
                    qualification.PRODUCER_STAGES.items(), (cbc, raw, f2)):
                path = directory / f"{name}.u32"
                path.write_bytes(np.asarray(data, dtype="<u4").tobytes())
                buffers[name] = {"file": path.relative_to(self.output).as_posix(), "sha256": metrics.sha256(path),
                                 "word_count": words, "byte_count": words * 4, "format": "uint32-little-endian",
                                 "role": role, "view_dimension": dimension, "view_format": view_format,
                                 "view_first_element": 0, "view_word_count": words, "resource_byte_count": words * 4,
                                 "structure_stride_bytes": stride}
            frame["producer_stages"] = {"scope": "diagnostic-only-exact-completed-composition", "pending_queued": False,
                                        "selected_matte_matches_draw": True, "selected_receipt_matches_draw": True,
                                        **{key: frame[key] for key in ("completed_frame_id", "observation_timestamp_us",
                                                                      "analysis_generation", "transaction_token",
                                                                      "source_snapshot_sha256")}, "buffers": buffers}

    def test_schema2_scores_literal_crop_and_binds_gt_meta_timeline_and_runtime(self):
        report = self.score()
        aggregate = report["aggregate"]
        self.assertEqual(aggregate["scored_frames"], 2)
        self.assertEqual(aggregate["counts"], {"predicted_pixels": 4, "ground_truth_pixels": 2,
                                              "true_positive_pixels": 1, "false_positive_pixels": 3,
                                              "false_negative_pixels": 1})
        self.assertEqual((aggregate["precision"], aggregate["recall"], aggregate["iou"]), (0.25, 0.5, 0.2))
        self.assertEqual(aggregate["empty_false_protected_pixels"], 2)
        self.assertEqual(report["frames"][0]["empty_false_protected_pixels"], 2)
        self.assertIsNone(report["frames"][0]["recall"])
        self.assertIsNone(report["frames"][1]["empty_false_protected_pixels"])
        self.assertIsNone(aggregate["temporal"]["all_flicker_fraction"])
        self.assertEqual(report["source_truth_identity"]["metadata"]["sha256"], metrics.sha256(self.frames / "meta.json"))
        self.assertEqual(report["observation_timeline"], self.document["observation_timeline"])
        self.assertEqual(report["numeric_runtime"]["python_executable"], str(Path(sys.executable).resolve()))
        self.assertEqual(len(report["source_truth_identity"]["tight_masks"]), 2)
        self.assertFalse(report["final_rgb_retention_qualified"])
        self.assertIsNone(report["quality_gate"])
        self.assertTrue((self.output / metrics.REPORT).is_file())

    def test_full_source_truth_outside_crop_is_a_false_negative(self):
        frame = self.document["frames"][1]
        path = self.gt_directory / frame["source_file"]
        with Image.open(path) as image:
            truth = np.array(image)
        truth[0, 0] = 255
        Image.fromarray(truth).save(path)
        report = self.score()
        self.assertEqual(report["aggregate"]["counts"]["false_negative_pixels"], 2)

    def test_schema1_and_absent_optional_truth_do_not_fabricate_scores(self):
        for frame in self.document["frames"]:
            del frame["selected_mask"]
        self.document["schema"] = 1
        report = self.score()
        self.assertFalse(report["selected_mask_evidence_available"])
        self.assertEqual(report["aggregate"]["scored_frames"], 0)
        self.assertIsNone(report["aggregate"]["precision"])
        self.assertIsNone(report["aggregate"]["counts"])
        for path in self.gt_directory.iterdir():
            path.unlink()
        self.gt_directory.rmdir()
        self.assertIsNone(self.score()["ground_truth_directory"])

    def test_corrupt_hash_nonbinary_bytes_and_wrong_crop_are_rejected(self):
        original = copy.deepcopy(self.document)
        frame = self.document["frames"][0]
        path = self.output / frame["selected_mask"]["file"]
        frame["selected_mask"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "size/hash"):
            self.score()
        self.document = copy.deepcopy(original)
        raw = bytearray(path.read_bytes())
        raw[0] = 2
        path.write_bytes(raw)
        self.document["frames"][0]["selected_mask"]["sha256"] = metrics.sha256(path)
        with self.assertRaisesRegex(ValueError, "not binary"):
            self.score()
        path.write_bytes(bytes([1]) + bytes(raw[1:]))
        self.document = copy.deepcopy(original)
        self.document["frames"][0]["selected_mask"]["source_rect"][1] -= 1
        with self.assertRaisesRegex(ValueError, "source_rect"):
            self.score()

    def test_gt_identity_dimensions_binary_values_and_mode_are_required(self):
        path = self.gt_directory / self.document["frames"][0]["source_file"]
        valid = path.read_bytes()
        cases = (("L", (960, 540), 127), ("L", (960, 541), 0), ("RGB", (960, 540), 0))
        for mode, size, value in cases:
            with self.subTest(mode=mode, size=size, value=value):
                Image.new(mode, size, value).save(path)
                with self.assertRaises(ValueError):
                    self.score()
        path.write_bytes(valid)
        path.rename(path.with_name("frame_99999.png"))
        with self.assertRaisesRegex(ValueError, "identities"):
            self.score()

    def test_declared_required_truth_cannot_be_missing_or_mistyped(self):
        for path in self.gt_directory.iterdir():
            path.unlink()
        self.gt_directory.rmdir()
        metadata = self.frames / "meta.json"
        for flag in ("required_gt_subtitle_tight_mask", "required_gt_subtitle_sanitizer_oracle"):
            with self.subTest(flag=flag):
                metadata.write_text(json.dumps({flag: True}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "requires missing"):
                    self.score()
                metadata.write_text(json.dumps({flag: 1}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "not boolean"):
                    self.score()
                metadata.write_text(json.dumps({flag: False}), encoding="utf-8")
                self.assertEqual(self.score()["aggregate"]["scored_frames"], 0)

    def test_manifest_replaced_after_validation_is_rejected(self):
        self.save()
        real_validate = qualification.validate_qualification
        def replace_after_validation(*args, **kwargs):
            result = real_validate(*args, **kwargs)
            changed = copy.deepcopy(self.document)
            changed["frames"][0]["bindings_observed"] = False
            (self.output / qualification.MANIFEST).write_text(json.dumps(changed), encoding="utf-8")
            return result
        with mock.patch.object(qualification, "validate_qualification", side_effect=replace_after_validation):
            with self.assertRaisesRegex(ValueError, "changed during validation"):
                metrics.score_qualification(self.output, self.frames, self.executable)

    def test_schema3_producer_snapshots_preserve_mask_scores_and_source_roles(self):
        before = self.score()["aggregate"]
        self.upgrade_to_schema3()
        report = self.score()
        self.assertEqual(report["aggregate"], before)
        self.assertTrue(report["qualification"]["producer_stage_evidence_available"])
        self.assertEqual(report["qualification"]["producer_stage_current_frames"], 2)
        self.assertEqual(report["frames"][0]["producer_stages"], self.document["frames"][0]["producer_stages"])

    def test_schema3_rejects_owner_descriptor_closure_and_coverage_mutations(self):
        self.upgrade_to_schema3()
        original = copy.deepcopy(self.document)
        changes = {
            "queued alias": lambda d: d["frames"][0]["producer_stages"].update(pending_queued=True),
            "M role mismatch": lambda d: d["frames"][0]["producer_stages"].update(selected_matte_matches_draw=False),
            "receipt role mismatch": lambda d: d["frames"][0]["producer_stages"].update(selected_receipt_matches_draw=False),
            "stale owner": lambda d: d["frames"][0]["producer_stages"].update(completed_frame_id=2),
            "stale clock": lambda d: d["frames"][0]["producer_stages"].update(observation_timestamp_us=42),
            "wrong snapshot": lambda d: d["frames"][0]["producer_stages"].update(source_snapshot_sha256="0" * 64),
            "unknown scope": lambda d: d["frames"][0]["producer_stages"].update(scope="rendering-authority"),
            "subrange": lambda d: d["frames"][0]["producer_stages"]["buffers"]["cbc_meta"].update(view_first_element=1),
            "oversized resource": lambda d: d["frames"][0]["producer_stages"]["buffers"]["cbc_meta"].update(resource_byte_count=2052),
            "raw as structured": lambda d: d["frames"][0]["producer_stages"]["buffers"]["f2_data"].update(view_dimension="BUFFER"),
            "raw stride": lambda d: d["frames"][0]["producer_stages"]["buffers"]["f2_data"].update(structure_stride_bytes=4),
            "structured format": lambda d: d["frames"][0]["producer_stages"]["buffers"]["raw_meta"].update(view_format="R32_TYPELESS"),
            "wrong role": lambda d: d["frames"][0]["producer_stages"]["buffers"]["cbc_meta"].update(role="RawMeta512"),
            "wrong words": lambda d: d["frames"][0]["producer_stages"]["buffers"]["f2_data"].update(word_count=512),
            "wrong hash": lambda d: d["frames"][0]["producer_stages"]["buffers"]["raw_meta"].update(sha256="0" * 64),
            "wrong closure": lambda d: d["composition_source_closures"].update(enclosure="0" * 64),
            "missing optional hole inventory": lambda d: d["composition_source_closures"].pop("hole"),
            "missing optional coverage inventory": lambda d: d["composition_source_closures"].pop("coverage"),
            "coverage programs unavailable": lambda d: d.update(coverage_program_ready=False),
            "frame coverage programs unavailable": lambda d: d["frames"][0].update(coverage_program_ready=False),
            "coverage availability is not a numeric flag": lambda d: d["frames"][0].update(coverage_program_ready=1),
            "hole programs unavailable": lambda d: d.update(hole_program_ready=False),
            "frame hole programs unavailable": lambda d: d["frames"][0].update(hole_program_ready=False),
            "hole availability is not a numeric flag": lambda d: d["frames"][0].update(hole_program_ready=1),
            "missing frame evidence": lambda d: d["frames"][0].pop("producer_stages"),
            "missing stage role": lambda d: d["frames"][0]["producer_stages"]["buffers"].pop("raw_meta"),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                self.document = copy.deepcopy(original)
                change(self.document)
                with self.assertRaises(ValueError):
                    self.score()
        self.document = original
        extra = self.output / "current-outline/producer-stages/unbound.u32"
        extra.write_bytes(b"unbound")
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.score()

    def test_schema3_rejects_corrupt_stage_headers_c24_joins_and_file_lengths(self):
        self.upgrade_to_schema3()
        for name, word in (("cbc_meta", 2), ("raw_meta", 3), ("f2_data", 1),
                           ("cbc_meta", 16), ("raw_meta", 12), ("f2_data", 8),
                           ("raw_meta", 36), ("f2_data", 32), ("cbc_meta", 80),
                           ("cbc_meta", 40), ("cbc_meta", 44), ("cbc_meta", 52), ("cbc_meta", 55),
                           ("cbc_meta", 56), ("cbc_meta", 58), ("cbc_meta", 60), ("cbc_meta", 64), ("cbc_meta", 72)):
            with self.subTest(name=name, word=word):
                binding = self.document["frames"][0]["producer_stages"]["buffers"][name]
                path = self.output / binding["file"]
                original = path.read_bytes()
                values = np.frombuffer(original, dtype="<u4").copy()
                values[word] ^= 1
                path.write_bytes(values.tobytes())
                binding["sha256"] = metrics.sha256(path)
                with self.assertRaisesRegex(ValueError, "producer stage current"):
                    self.score()
                path.write_bytes(original)
                binding["sha256"] = metrics.sha256(path)
        binding = self.document["frames"][0]["producer_stages"]["buffers"]["raw_meta"]
        path = self.output / binding["file"]
        path.write_bytes(path.read_bytes()[:-4])
        binding["sha256"] = metrics.sha256(path)
        with self.assertRaisesRegex(ValueError, "byte length"):
            self.score()

    def test_schema3_fallback_diagnostic_bytes_do_not_become_current_proof(self):
        self.upgrade_to_schema3()
        frame = self.document["frames"][0]
        frame["selected_receipt"][2:4] = [1, 1]
        frame["selected_receipt"][28:60] = [0] * 32
        frame["background_proof"][2:4] = [0, 1]
        for binding in frame["producer_stages"]["buffers"].values():
            path = self.output / binding["file"]
            path.write_bytes(b"\0" * binding["byte_count"])
            binding["sha256"] = metrics.sha256(path)
        result = self.score()["qualification"]
        self.assertEqual(result["selected_current_frames"], 1)
        self.assertEqual(result["producer_stage_current_frames"], 1)


if __name__ == "__main__":
    unittest.main()
