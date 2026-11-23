#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import prod_zipdepth_tuning as tuning  # noqa: E402

FEATURES = 4


def make_composite() -> onnx.ModelProto:
    """The composite's shape in miniature: 2x2 average -> "ViT" (one 1x1 convolution) -> final
    feature resize -> output convolution -> ReLU."""
    rng = np.random.default_rng(7)
    graph = helper.make_graph(
        [
            helper.make_node(
                "AveragePool", ["pixel_values"], ["dav2_pixel_values"], name=tuning.AVERAGE_POOL_NODE,
                kernel_shape=[2, 2], strides=[2, 2]),
            helper.make_node("Conv", ["dav2_pixel_values", "vit_weight"], ["features"], name="vit"),
            helper.make_node(
                "Resize", ["features", "", "upscale"], ["features_resized"], name="final_feature_resize",
                mode="linear", coordinate_transformation_mode="align_corners"),
            helper.make_node(
                "Conv", ["features_resized", tuning.FINAL_CONV_WEIGHT], ["depth"], name="output_conv"),
            helper.make_node("Relu", ["depth"], ["depth_relu"], name="output_relu"),
            helper.make_node("Squeeze", ["depth_relu", "channel_axis"], ["refined_depth"], name="output_squeeze"),
        ],
        "composite",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, "high_h", "high_w"])],
        [helper.make_tensor_value_info("refined_depth", TensorProto.FLOAT, [1, "h", "w"])],
        [
            numpy_helper.from_array(rng.normal(size=(FEATURES, 3, 1, 1)).astype(np.float32), "vit_weight"),
            numpy_helper.from_array(np.array([1, 1, 1, 1], np.float32), "upscale"),
            numpy_helper.from_array(
                np.abs(rng.normal(size=(1, FEATURES, 1, 1))).astype(np.float32), tuning.FINAL_CONV_WEIGHT),
            numpy_helper.from_array(np.array([1], np.int64), "channel_axis"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    return model


def run(model: onnx.ModelProto, image: np.ndarray) -> np.ndarray:
    onnx.checker.check_model(model, full_check=True)
    return ReferenceEvaluator(model).run(None, {"pixel_values": image})[0]


class VitInputTests(unittest.TestCase):
    def test_size_is_a_whole_number_of_patches_and_matches_the_production_grids(self):
        self.assertEqual(tuning.vit_input_size(1540, 0.6875), 1064)
        self.assertEqual(tuning.vit_input_size(868, 0.6875), 602)
        self.assertEqual(tuning.vit_input_size(2072, 0.6875), 1428)
        for high in (868, 1148, 1232, 1260, 1316, 1400, 1540, 2044, 2072):
            self.assertEqual(tuning.vit_input_size(high, 0.6875) % tuning.VIT_PATCH, 0)
            # the graph computes the same value with a float32 product and ONNX Round
            patches = np.round(np.float32(high) * np.float32(0.6875 / tuning.VIT_PATCH))
            self.assertEqual(int(patches) * tuning.VIT_PATCH, tuning.vit_input_size(high, 0.6875))

    def test_retarget_keeps_the_output_grid_and_feeds_the_vit_the_scaled_grid(self):
        model = make_composite()
        image = np.random.default_rng(3).random((1, 3, 28, 56), dtype=np.float32)
        before = run(model, image)
        self.assertEqual(before.shape, (1, 14, 28))
        report = tuning.retarget_vit_input(model, 0.75)
        self.assertEqual(report["final_feature_resize_node"], "final_feature_resize")
        self.assertEqual(report["feature_channels"], FEATURES)
        self.assertFalse(any(node.name == tuning.AVERAGE_POOL_NODE for node in model.graph.node))
        after = run(model, image)
        self.assertEqual(after.shape, before.shape)
        probe = onnx.ModelProto()
        probe.CopyFrom(model)
        probe.graph.output.append(helper.make_tensor_value_info("dav2_pixel_values", TensorProto.FLOAT, None))
        seen = ReferenceEvaluator(probe).run(["dav2_pixel_values"], {"pixel_values": image})[0]
        self.assertEqual(seen.shape, (1, 3, tuning.vit_input_size(28, 0.75), tuning.vit_input_size(56, 0.75)))
        self.assertEqual(seen.shape[2:], (28, 42))
        # another bound shape, same graph
        other = np.random.default_rng(4).random((1, 3, 56, 28), dtype=np.float32)
        self.assertEqual(run(model, other).shape, (1, 28, 14))

    def test_retarget_refuses_a_graph_without_the_average_or_a_second_application(self):
        model = make_composite()
        tuning.retarget_vit_input(model, 0.75)
        with self.assertRaisesRegex(ValueError, "2x2 input average was not found"):
            tuning.retarget_vit_input(model, 0.75)
        with self.assertRaisesRegex(ValueError, "outside"):
            tuning.retarget_vit_input(make_composite(), 1.5)

    def test_a_constant_image_passes_unchanged_through_either_input_path(self):
        image = np.full((1, 3, 28, 28), 0.25, np.float32)
        model = make_composite()
        before = run(model, image)
        tuning.retarget_vit_input(model, 0.5)
        np.testing.assert_allclose(run(model, image), before, rtol=1e-6, atol=1e-6)


class GainTests(unittest.TestCase):
    def test_gain_scales_the_public_output_once(self):
        model = make_composite()
        image = np.random.default_rng(5).random((1, 3, 28, 28), dtype=np.float32)
        before = run(model, image)
        tuning.apply_output_gain(model, 1.25)
        np.testing.assert_allclose(run(model, image), before * np.float32(1.25), rtol=1e-6)
        self.assertEqual([output.name for output in model.graph.output], ["refined_depth"])
        with self.assertRaisesRegex(ValueError, "already has an output gain"):
            tuning.apply_output_gain(model, 1.25)
        with self.assertRaisesRegex(ValueError, "not positive"):
            tuning.apply_output_gain(make_composite(), 0.0)


class WeightSwapTests(unittest.TestCase):
    def make_dav2(self):
        rng = np.random.default_rng(11)
        stock = {
            "pretrained.blocks.0.attn.qkv.weight": rng.normal(size=(6, 4)).astype(np.float32),
            "pretrained.blocks.0.mlp.fc1.weight": rng.normal(size=(8, 4)).astype(np.float32),
            "pretrained.blocks.0.attn.qkv.bias": rng.normal(size=(6,)).astype(np.float32),
            "depth_head.projects.0.weight": rng.normal(size=(2, 4, 1, 1)).astype(np.float32),
        }
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"], name="identity")],
            "dav2",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            [
                # the ONNX keeps linear weights transposed, unnamed and in half precision
                numpy_helper.from_array(stock["pretrained.blocks.0.attn.qkv.weight"].T.astype(np.float16), "anon_1"),
                numpy_helper.from_array(stock["pretrained.blocks.0.mlp.fc1.weight"].T.astype(np.float16), "anon_2"),
                numpy_helper.from_array(stock["depth_head.projects.0.weight"], "depth_head.projects.0.weight"),
            ],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), stock

    def test_head_tensors_are_written_by_name_and_missing_ones_reported(self):
        model, stock = self.make_dav2()
        tuned = dict(stock)
        tuned["depth_head.projects.0.weight"] = stock["depth_head.projects.0.weight"] + 1
        tuned["depth_head.folded.weight"] = np.zeros((1,), np.float32)
        written, absent = tuning.swap_head_initializers(model, tuned)
        self.assertEqual((written, absent), (1, ["depth_head.folded.weight"]))
        stored = numpy_helper.to_array(
            next(i for i in model.graph.initializer if i.name == "depth_head.projects.0.weight"))
        np.testing.assert_array_equal(stored, tuned["depth_head.projects.0.weight"])
        tuned["depth_head.projects.0.weight"] = np.zeros((3, 4, 1, 1), np.float32)
        with self.assertRaisesRegex(ValueError, "tuned shape"):
            tuning.swap_head_initializers(model, tuned)

    def test_changed_vit_weights_replace_the_initializer_that_holds_the_stock_weight(self):
        model, stock = self.make_dav2()
        tuned = dict(stock)
        tuned["pretrained.blocks.0.attn.qkv.weight"] = stock["pretrained.blocks.0.attn.qkv.weight"] * 2
        self.assertEqual(tuning.swap_vit_linear_initializers(model, stock, tuned), 1)
        stored = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        np.testing.assert_array_equal(
            stored["anon_1"], tuned["pretrained.blocks.0.attn.qkv.weight"].T.astype(np.float16))
        np.testing.assert_array_equal(
            stored["anon_2"], stock["pretrained.blocks.0.mlp.fc1.weight"].T.astype(np.float16))
        self.assertEqual(stored["anon_1"].dtype, np.float16)

    def test_a_changed_vit_tensor_the_export_cannot_carry_is_an_error(self):
        model, stock = self.make_dav2()
        tuned = dict(stock)
        tuned["pretrained.blocks.0.attn.qkv.bias"] = stock["pretrained.blocks.0.attn.qkv.bias"] + 1
        with self.assertRaisesRegex(ValueError, "cannot carry"):
            tuning.swap_vit_linear_initializers(model, stock, tuned)
        tuned = dict(stock)
        tuned["pretrained.pos_embed"] = np.zeros((1, 2, 4), np.float32)
        with self.assertRaisesRegex(ValueError, "cannot carry"):
            tuning.swap_vit_linear_initializers(model, stock, tuned)

    def test_a_weight_without_exactly_one_home_is_an_error(self):
        model, stock = self.make_dav2()
        tuned = dict(stock)
        tuned["pretrained.blocks.0.attn.qkv.weight"] = stock["pretrained.blocks.0.attn.qkv.weight"] * 2
        other = {name: value + 1 for name, value in stock.items()}
        with self.assertRaisesRegex(ValueError, "expected one matching initializer, found 0"):
            tuning.swap_vit_linear_initializers(model, other, tuned)
        duplicate = onnx.ModelProto()
        duplicate.CopyFrom(model)
        duplicate.graph.initializer.append(numpy_helper.from_array(
            stock["pretrained.blocks.0.attn.qkv.weight"].T.astype(np.float16), "anon_3"))
        with self.assertRaisesRegex(ValueError, "found 2"):
            tuning.swap_vit_linear_initializers(duplicate, stock, tuned)


if __name__ == "__main__":
    unittest.main()
