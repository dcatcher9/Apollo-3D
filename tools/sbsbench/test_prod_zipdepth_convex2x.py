#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import prod_zipdepth_convex2x as convex  # noqa: E402


class ProdZipDepthConvex2xTests(unittest.TestCase):
    def _assert_contract_rejected(self, mutate, expected="contract"):
        document = copy.deepcopy(convex.load_contract())
        mutate(document)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with mock.patch.object(convex, "CONTRACT_PATH", path):
                with self.assertRaisesRegex(ValueError, expected):
                    convex.load_contract()

    def test_contract_exposes_only_single_high_input_and_output(self):
        contract = convex.load_contract()
        io = contract["engine_io"]
        self.assertEqual(
            convex.tensor_names(io["inputs"]),
            ("pixel_values",),
        )
        self.assertEqual(
            convex.tensor_names(io["outputs"]),
            ("refined_depth",),
        )
        self.assertEqual(
            convex.tensor_names(io["internal_tensors"]),
            ("dav2_pixel_values", "predicted_depth"),
        )
        downsample = contract["operator"]["input_downsample"]
        self.assertEqual(downsample["operator"], "AveragePool")
        self.assertEqual(downsample["dtype"], "float32")
        self.assertEqual(downsample["kernel"], [2, 2])
        self.assertEqual(downsample["stride"], [2, 2])
        self.assertEqual(contract["operator"]["pixel_gate"], "none")
        self.assertEqual(contract["sources"]["zipdepth"]["variant"], "base")
        self.assertEqual(contract["operator"]["temperature"], 1.0)
        self.assertIs(contract["operator"]["zipdepth_use_unfold"], True)
        self.assertEqual(
            contract["sources"]["fused_onnx"]["sha256"],
            "26684c5da8fdd4bdc5f1c9cf919cec8d1e2d027fbe95705a454f85d31eee2c23",
        )
        optimization = contract["export"]["model_optimization"]
        self.assertEqual(
            optimization["recipe"],
            "zipdepth-selective-fp16-project-before-resize-dense-group4-v1",
        )
        self.assertEqual(optimization["selective_fp16_initializer_count"], 88)
        self.assertEqual(len(optimization["project_before_resize_nodes"]), 4)
        self.assertEqual(len(optimization["dense_group4_nodes"]), 10)
        self.assertEqual(
            optimization["precision"]["predicted_depth_and_convex_tail"],
            "float32",
        )
        self.assertIn("graph-cut", contract["authority"]["forbidden"])
        self.assertIn("adaptive-j", contract["authority"]["forbidden"])

    def test_all_production_shapes_scale_exactly(self):
        expected = {
            (770, 434): (1540, 868),
            (1022, 434): (2044, 868),
            (1036, 434): (2072, 868),
            (434, 770): (868, 1540),
            (434, 1022): (868, 2044),
            (434, 1036): (868, 2072),
        }
        actual = {
            (shape.width, shape.height): (
                convex.refined_shape(shape).width,
                convex.refined_shape(shape).height,
            )
            for shape in convex.supported_coarse_shapes()
        }
        self.assertEqual(actual, expected)

    def test_tensorrt_uses_one_fixed_high_point_profile_per_shape(self):
        contract = convex.load_contract()
        tensorrt = contract["tensorrt"]
        self.assertEqual(
            tensorrt["profile_strategy"],
            "one-engine-six-fixed-high-point-profiles",
        )
        self.assertEqual(tensorrt["profile_order"], "high_shapes_wh")
        self.assertEqual(tensorrt["builder_optimization_level"], 5)
        self.assertEqual(len(contract["high_shapes_wh"]), 6)
        self.assertEqual(
            [(shape.width, shape.height) for shape in convex.supported_high_shapes()],
            [
                (1540, 868),
                (2044, 868),
                (2072, 868),
                (868, 1540),
                (868, 2044),
                (868, 2072),
            ],
        )
        for index, shape in enumerate(convex.supported_high_shapes()):
            self.assertEqual(convex.fixed_profile_index(shape), index)
        with self.assertRaisesRegex(ValueError, "unsupported high shape"):
            convex.fixed_profile_index(convex.Shape(434, 434))

    def test_contract_rejects_nonexact_or_coerced_profile_tables(self):
        mutations = {
            "missing": lambda value: value["high_shapes_wh"].pop(),
            "duplicate": lambda value: value["high_shapes_wh"].__setitem__(
                5, list(value["high_shapes_wh"][4])),
            "wrong-order": lambda value: value["high_shapes_wh"].__setitem__(
                slice(0, 2), list(reversed(value["high_shapes_wh"][:2]))),
            "wrong-calibrated-half": lambda value: value["high_shapes_wh"][0].__setitem__(
                0, 1568),
            "odd": lambda value: value["high_shapes_wh"][0].__setitem__(0, 1539),
            "string": lambda value: value["high_shapes_wh"][0].__setitem__(0, "1540"),
            "float": lambda value: value["high_shapes_wh"][0].__setitem__(0, 1540.0),
            "boolean": lambda value: value["high_shapes_wh"][0].__setitem__(0, True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                self._assert_contract_rejected(mutate, "exact ordered six calibrated")

    def test_contract_rejects_profile_strategy_and_io_drift(self):
        for name, mutate in {
                "profile-strategy": lambda value: value["tensorrt"].update(
                    {"profile_strategy": "one-ranged-profile"}),
                "profile-order": lambda value: value["tensorrt"].update(
                    {"profile_order": "sorted"}),
                "input-shape": lambda value: value["engine_io"]["inputs"][0].update(
                    {"shape": [1, 3, "H", "W"]}),
                "output-shape": lambda value: value["engine_io"]["outputs"][0].update(
                    {"shape": [1, "H", "W"]}),
                "bool-optimization-level": lambda value: value["tensorrt"].update(
                    {"builder_optimization_level": True}),
                }.items():
            with self.subTest(name=name):
                self._assert_contract_rejected(mutate)

    def test_contract_rejects_model_optimization_drift(self):
        mutations = {
            "recipe": lambda value: value["export"]["model_optimization"].update(
                {"recipe": "generic-autocast"}
            ),
            "order": lambda value: value["export"]["model_optimization"].update(
                {"order": list(reversed(
                    value["export"]["model_optimization"]["order"]
                ))}
            ),
            "feature-precision": lambda value: value["export"][
                "model_optimization"
            ]["precision"].update({"zipdepth_feature_mask": "float32"}),
            "dense-target": lambda value: value["export"]["model_optimization"][
                "dense_group4_nodes"
            ].pop(),
            "tail-target": lambda value: value["export"]["model_optimization"][
                "frozen_convex_tail_nodes"
            ].append("node_extra"),
            "raw-hash": lambda value: value["export"].update(
                {"raw_zipdepth_branch_onnx_sha256": "0" * 63}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                self._assert_contract_rejected(mutate)

    def test_content_edges_are_scaled_not_refitted(self):
        shape = convex.Shape(770, 434)
        content = convex.ContentRect(17, 9, 752, 421)
        self.assertEqual(
            convex.refined_content_rect(shape, content),
            convex.ContentRect(34, 18, 1504, 842),
        )

    def test_uniform_logits_are_a_convex_local_average(self):
        depth = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        logits = np.zeros((1, 36, 2, 2), dtype=np.float32)
        output = convex.convex2x(depth, logits)
        lower, upper = convex.local_bounds2x(depth)
        self.assertEqual(output.shape, (1, 4, 4))
        self.assertTrue(np.all(output >= lower - 1.0e-6))
        self.assertTrue(np.all(output <= upper + 1.0e-6))

    def test_subpixel_layout_is_row_major_pixel_shuffle(self):
        depth = np.asarray(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
            dtype=np.float32,
        )
        logits = np.full((1, 9, 4, 3, 3), -80.0, dtype=np.float32)
        # Each 2x2 phase selects a different 3x3 neighbour at the center cell.
        for phase, neighbor in enumerate((0, 2, 6, 8)):
            logits[:, neighbor, phase, :, :] = 80.0
        output = convex.convex2x(depth, logits)
        center = output[0, 2:4, 2:4]
        np.testing.assert_array_equal(center, np.asarray([[1.0, 3.0], [7.0, 9.0]]))

    def test_nonfinite_input_rejects_the_whole_frame(self):
        depth = np.ones((1, 2, 2), dtype=np.float32)
        logits = np.zeros((1, 36, 2, 2), dtype=np.float32)
        depth[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "one unit"):
            convex.convex2x(depth, logits)

    def test_evaluation_harness_exports_one_single_high_model_boundary(self):
        repo = SCRIPT_DIR.parent.parent
        harness = (repo / "src" / "sbs_bench_harness.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn('"model_input_" + output_id + ".f32"', harness)
        self.assertIn("prod_zipdepth_convex2x_diagnostics.json", harness)
        self.assertIn("single-high-input-output-boundary", harness)
        self.assertIn("est.model_input_snapshot.Get()", harness)
        self.assertIn(
            "est.guidance_model_input_snapshot.Get() ==",
            harness,
        )
        self.assertIn(
            "est.refined_model_depth_snapshot.Get() ==",
            harness,
        )
        self.assertNotIn('"refined_" + output_id + ".f32"', harness)
        self.assertNotIn('"zip_model_input_" + output_id + ".f32"', harness)
        self.assertNotIn(
            "warp_depth = est.refined_model_depth_snapshot",
            harness,
        )
        self.assertNotIn(
            "est.raw_model_depth = est.refined_model_depth_snapshot",
            harness,
        )


if __name__ == "__main__":
    unittest.main()
