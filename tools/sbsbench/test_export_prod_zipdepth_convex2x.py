#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import onnx
from onnx import helper


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import export_prod_zipdepth_convex2x as exporter  # noqa: E402
import prod_zipdepth_convex2x as contract_api  # noqa: E402


def make_dummy_dav2() -> onnx.ModelProto:
    pixel_values = helper.make_tensor_value_info(
        "pixel_values", onnx.TensorProto.FLOAT, [1, 3, "height", "width"]
    )
    predicted_depth = helper.make_tensor_value_info(
        "predicted_depth", onnx.TensorProto.FLOAT, [1, "height", "width"]
    )
    channel = helper.make_tensor("channel", onnx.TensorProto.INT64, [], [0])
    graph = helper.make_graph(
        [helper.make_node("Gather", ["pixel_values", "channel"], ["predicted_depth"], axis=1)],
        "dummy_dav2",
        [pixel_values],
        [predicted_depth],
        [channel],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])


def make_dummy_zip_branch() -> onnx.ModelProto:
    guidance = helper.make_tensor_value_info(
        "zip_pixel_values", onnx.TensorProto.FLOAT, [1, 3, "2*height", "2*width"]
    )
    predicted_depth = helper.make_tensor_value_info(
        "predicted_depth", onnx.TensorProto.FLOAT, [1, "height", "width"]
    )
    refined_depth = helper.make_tensor_value_info(
        "refined_depth", onnx.TensorProto.FLOAT, [1, "2*height", "2*width"]
    )
    channel = helper.make_tensor("channel", onnx.TensorProto.INT64, [], [0])
    graph = helper.make_graph(
        [helper.make_node("Gather", ["zip_pixel_values", "channel"], ["refined_depth"], axis=1)],
        "dummy_zip_branch",
        [guidance, predicted_depth],
        [refined_depth],
        [channel],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", exporter.OPSET_VERSION)]
    )


class ProdZipDepthExporterTests(unittest.TestCase):
    def test_nonempty_output_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "prior.txt").write_text("prior run", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing nonempty"):
                exporter.prepare_output_directory(occupied)

            empty = root / "empty"
            self.assertEqual(exporter.prepare_output_directory(empty), empty.resolve())
            self.assertTrue(empty.is_dir())

    def test_source_hash_check_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset.bin"
            path.write_bytes(b"frozen asset")
            expected = hashlib.sha256(b"frozen asset").hexdigest()
            self.assertEqual(exporter.require_file_hash(path, expected, "asset"), expected)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                exporter.require_file_hash(path, "0" * 64, "asset")

    def test_generated_artifact_identity_is_frozen_by_bytes_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.onnx"
            path.write_bytes(b"canonical graph")
            digest = hashlib.sha256(b"canonical graph").hexdigest()
            observed = exporter.require_artifact_identity(
                path, len(b"canonical graph"), digest, "fused ONNX"
            )
            self.assertEqual(observed["sha256"], digest)
            with self.assertRaisesRegex(ValueError, "not the frozen release artifact"):
                exporter.require_artifact_identity(
                    path, len(b"canonical graph") + 1, digest, "fused ONNX"
                )

        fused_contract = contract_api.load_contract()["sources"]["fused_onnx"]
        self.assertEqual(
            fused_contract["sha256"],
            "0547dd046dead55057bb34a356d987559b2d93248e84600245f02df828d8bbb7",
        )
        self.assertEqual(fused_contract["bytes"], 74279879)

    def test_tensorrt_arguments_preserve_six_fixed_profile_order(self):
        command = exporter.point_profile_build_arguments(
            Path("trtexec.exe"),
            Path("fused.onnx"),
            Path("fused.plan"),
            5,
        )
        profiles = [item for item in command if item.startswith("--profile=")]
        self.assertEqual(profiles, [f"--profile={index}" for index in range(6)])
        self.assertIn("--builderOptimizationLevel=5", command)
        self.assertIn(
            "--minShapes=pixel_values:1x3x868x1540",
            command,
        )
        self.assertIn(
            "--maxShapes=pixel_values:1x3x2072x868",
            command,
        )
        self.assertFalse(any("zip_pixel_values" in item for item in command))
        expected = [
            (shape.width, shape.height)
            for shape in contract_api.supported_high_shapes()
        ]
        self.assertEqual(
            expected,
            [
                (1540, 868),
                (2044, 868),
                (2072, 868),
                (868, 1540),
                (868, 2044),
                (868, 2072),
            ],
        )

    def test_fused_composition_is_deterministic_and_has_single_high_io(self):
        first = exporter.compose_fused_models(make_dummy_dav2(), make_dummy_zip_branch())
        second = exporter.compose_fused_models(make_dummy_dav2(), make_dummy_zip_branch())
        self.assertEqual(
            exporter.deterministic_model_bytes(first),
            exporter.deterministic_model_bytes(second),
        )
        self.assertEqual(
            tuple(item.name for item in first.graph.input),
            ("pixel_values",),
        )
        self.assertEqual(
            tuple(item.name for item in first.graph.output),
            ("refined_depth",),
        )
        self.assertEqual(
            exporter.shape_of(first.graph.input[0]),
            [1, 3, "2*height", "2*width"],
        )
        self.assertEqual(
            exporter.shape_of(first.graph.output[0]), [1, "2*height", "2*width"]
        )
        pools = [node for node in first.graph.node if node.op_type == "AveragePool"]
        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0].name, "single_high_io_average_pool_2x2")
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in pools[0].attribute
        }
        self.assertEqual(attributes["kernel_shape"], [2, 2])
        self.assertEqual(attributes["strides"], [2, 2])
        self.assertEqual(attributes["pads"], [0, 0, 0, 0])
        self.assertEqual(attributes["ceil_mode"], 0)
        self.assertEqual(attributes["count_include_pad"], 0)
        internal = next(
            value
            for value in first.graph.value_info
            if value.name == "dav2_pixel_values"
        )
        self.assertEqual(internal.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        self.assertEqual(
            exporter.shape_of(internal), [1, 3, "height", "width"]
        )
        self.assertNotIn("predicted_depth", {item.name for item in first.graph.output})
        self.assertEqual(
            [(item.domain, item.version) for item in first.opset_import],
            [("", exporter.OPSET_VERSION)],
        )
        onnx.checker.check_model(first, full_check=True)

    def test_composition_rejects_a_partial_zip_boundary(self):
        branch = make_dummy_zip_branch()
        branch.graph.output[0].name = "convex_logits"
        with self.assertRaisesRegex(ValueError, "unexpected ZipDepth convex branch boundary"):
            exporter.compose_fused_models(make_dummy_dav2(), branch)


if __name__ == "__main__":
    unittest.main()
