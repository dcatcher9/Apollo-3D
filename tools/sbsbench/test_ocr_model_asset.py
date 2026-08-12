"""Integrity tests for the packaged ModelOpt FP16 PP-OCRv6 detector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import onnx
from onnx import TensorProto


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "tools" / "sbsbench" / "contracts" / "depth-coordinate-v2-v1.json"
ASSETS_ROOT = REPO_ROOT / "src_assets" / "windows" / "assets"


class OcrModelAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["subtitle_ocr"]
        cls.asset_path = ASSETS_ROOT / cls.contract["asset_path"]
        cls.provenance_path = cls.asset_path.with_suffix(".provenance.json")

    def test_packaged_artifact_and_provenance_match_the_authenticated_contract(self):
        payload = self.asset_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            self.contract["artifact_onnx_sha256"],
        )
        provenance = json.loads(self.provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(provenance["artifact"]["path"], self.contract["asset_path"])
        self.assertEqual(provenance["artifact"]["bytes"], len(payload))
        self.assertEqual(
            provenance["artifact"]["sha256"],
            self.contract["artifact_onnx_sha256"],
        )
        self.assertEqual(provenance["source"]["url"], self.contract["source_url"])
        self.assertEqual(
            provenance["source"]["sha256"],
            self.contract["source_onnx_sha256"],
        )
        self.assertEqual(provenance["conversion"]["tool"], self.contract["conversion_tool"])
        self.assertEqual(
            provenance["conversion"]["version"],
            self.contract["conversion_version"],
        )
        self.assertEqual(
            provenance["conversion"]["recipe"],
            self.contract["conversion_recipe"],
        )
        self.assertEqual(
            provenance["conversion"]["calibration_profile"],
            self.contract["conversion_calibration_profile"],
        )
        self.assertTrue((ASSETS_ROOT / "licenses" / "PP-OCRv6-Apache-2.0.txt").is_file())

    def test_model_is_self_contained_fp16_compute_with_fp32_boundaries(self):
        model = onnx.load(self.asset_path, load_external_data=False)
        onnx.checker.check_model(model)
        self.assertFalse(any(initializer.external_data for initializer in model.graph.initializer))

        self.assertEqual(len(model.graph.input), 1)
        self.assertEqual(len(model.graph.output), 1)
        model_input = model.graph.input[0]
        model_output = model.graph.output[0]
        self.assertEqual(model_input.name, "x")
        self.assertEqual(model_output.name, "fetch_name_0")
        self.assertEqual(model_input.type.tensor_type.elem_type, TensorProto.FLOAT)
        self.assertEqual(model_output.type.tensor_type.elem_type, TensorProto.FLOAT)
        input_dimensions = model_input.type.tensor_type.shape.dim
        output_dimensions = model_output.type.tensor_type.shape.dim
        self.assertEqual(len(input_dimensions), 4)
        self.assertEqual(len(output_dimensions), 4)
        self.assertEqual(input_dimensions[1].dim_value, 3)
        self.assertEqual(output_dimensions[1].dim_value, 1)
        for dimension in (
            input_dimensions[0],
            input_dimensions[2],
            input_dimensions[3],
            output_dimensions[0],
            output_dimensions[2],
            output_dimensions[3],
        ):
            self.assertEqual(dimension.dim_value, 0)
            self.assertTrue(dimension.dim_param)

        # The authenticated ONNX deliberately retains Paddle's dynamic N/H/W
        # boundary. TensorRT fixes the only production optimization profile via
        # the generated contract and the engine recipe, while channel counts and
        # FP32 boundary types remain intrinsic to the model.
        self.assertEqual(self.contract["input_tensor"]["shape"], [1, 3, 160, 960])
        self.assertEqual(self.contract["output_tensor"]["shape"], [1, 1, 160, 960])
        self.assertIn("fixed960x160", self.contract["engine_recipe"])

        half_initializers = sum(
            initializer.data_type == TensorProto.FLOAT16
            for initializer in model.graph.initializer
        )
        self.assertGreaterEqual(half_initializers, 200)
        casts = [node for node in model.graph.node if node.op_type == "Cast"]
        self.assertEqual(len(casts), 2)
        cast_targets = sorted(
            attribute.i
            for node in casts
            for attribute in node.attribute
            if attribute.name == "to"
        )
        self.assertEqual(cast_targets, [TensorProto.FLOAT, TensorProto.FLOAT16])


if __name__ == "__main__":
    unittest.main()
