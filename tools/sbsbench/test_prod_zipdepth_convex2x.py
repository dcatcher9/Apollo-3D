#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import prod_zipdepth_convex2x as convex  # noqa: E402


class ProdZipDepthConvex2xTests(unittest.TestCase):
    def test_contract_keeps_coarse_and_refined_outputs_in_one_engine(self):
        contract = convex.load_contract()
        io = contract["engine_io"]
        self.assertEqual(
            convex.tensor_names(io["inputs"]),
            ("pixel_values", "zip_pixel_values"),
        )
        self.assertEqual(
            convex.tensor_names(io["outputs"]),
            ("predicted_depth", "refined_depth"),
        )
        self.assertEqual(contract["operator"]["pixel_gate"], "none")
        self.assertEqual(contract["sources"]["zipdepth"]["variant"], "base")
        self.assertEqual(contract["operator"]["temperature"], 1.0)
        self.assertIs(contract["operator"]["zipdepth_use_unfold"], True)
        self.assertEqual(
            contract["sources"]["fused_onnx"]["sha256"],
            "959fc90097d7055b9c56cb140f432e0f5aed533476e8cedd6ec2baae097b287f",
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

    def test_tensorrt_uses_one_fixed_point_profile_per_shape(self):
        contract = convex.load_contract()
        tensorrt = contract["tensorrt"]
        self.assertEqual(
            tensorrt["profile_strategy"],
            "one-engine-six-fixed-point-profiles",
        )
        self.assertEqual(tensorrt["profile_order"], "coarse_shapes_wh")
        self.assertEqual(tensorrt["builder_optimization_level"], 5)
        self.assertEqual(len(contract["coarse_shapes_wh"]), 6)
        for index, shape in enumerate(convex.supported_coarse_shapes()):
            self.assertEqual(convex.fixed_profile_index(shape), index)
        with self.assertRaisesRegex(ValueError, "unsupported coarse shape"):
            convex.fixed_profile_index(convex.Shape(434, 434))

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

    def test_evaluation_harness_exports_fused_diagnostics_without_authority(self):
        repo = SCRIPT_DIR.parent.parent
        harness = (repo / "src" / "sbs_bench_harness.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn('"refined_" + output_id + ".f32"', harness)
        self.assertIn('"zip_model_input_" + output_id + ".f32"', harness)
        self.assertIn("prod_zipdepth_convex2x_diagnostics.json", harness)
        self.assertIn("diagnostic-only-never-live-or-scoring-depth", harness)
        self.assertIn("est.refined_model_depth_snapshot.Get()", harness)
        self.assertIn("est.guidance_model_input_snapshot.Get()", harness)
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
