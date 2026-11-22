"""Integrity tests for the packaged DAV2 + ZipDepth composite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

import onnx
from onnx import TensorProto


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import prod_zipdepth_convex2x as contract_api  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parents[1]
ASSETS_ROOT = REPO_ROOT / "src_assets" / "windows" / "assets"


class CompositeAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = contract_api.load_contract()["sources"]
        cls.asset_path = ASSETS_ROOT / (cls.sources["fused_onnx"]["logical_model"] + ".onnx")
        cls.provenance = json.loads(
            cls.asset_path.with_suffix(".provenance.json").read_text(encoding="utf-8")
        )

    def test_packaged_file_is_the_one_the_contract_and_the_host_name(self):
        payload = self.asset_path.read_bytes()
        fused = self.sources["fused_onnx"]
        self.assertEqual(len(payload), fused["bytes"])
        self.assertEqual(hashlib.sha256(payload).hexdigest(), fused["sha256"])
        header = (REPO_ROOT / "src" / "prod_zipdepth_convex2x.h").read_text(encoding="utf-8")
        self.assertIn(fused["sha256"], header)
        self.assertIn(fused["logical_model"], header)

    def test_provenance_matches_the_contract(self):
        artifact = self.provenance["artifact"]
        fused = self.sources["fused_onnx"]
        self.assertEqual(artifact["path"], self.asset_path.name)
        self.assertEqual((artifact["bytes"], artifact["sha256"]), (fused["bytes"], fused["sha256"]))
        sources = self.provenance["sources"]
        tuning = self.sources["tuning"]
        self.assertEqual(sources["dav2"]["sha256"], self.sources["dav2"]["onnx_sha256"])
        self.assertEqual(
            sources["zipdepth"]["checkpoint_sha256"], self.sources["zipdepth"]["checkpoint_sha256"]
        )
        self.assertEqual(
            sources["zipdepth"]["repository_commit"], self.sources["zipdepth"]["repository_commit"]
        )
        self.assertEqual(sources["tuned_checkpoint"]["sha256"], tuning["checkpoint_sha256"])
        self.assertEqual(sources["tuned_checkpoint"]["bytes"], tuning["checkpoint_bytes"])
        self.assertEqual(
            sources["dav2_torch_checkpoint"]["sha256"], tuning["dav2_torch_checkpoint_sha256"]
        )
        recorded = self.provenance["tuning"]
        self.assertEqual(recorded["vit_input_scale"], tuning["vit_input_scale"])
        self.assertEqual(recorded["output_gain"], tuning["output_gain"])
        self.assertEqual(recorded["changed_tensors"]["dav2_decoder_head"], tuning["head_tensors"])
        self.assertEqual(
            recorded["changed_tensors"]["dav2_vit_linear_weights"], tuning["vit_linear_tensors"]
        )
        for name in ("Depth-Anything-V2-Small-Apache-2.0.txt", "ZipDepth-MIT.txt"):
            self.assertTrue((ASSETS_ROOT / "licenses" / name).is_file(), name)

    def test_model_is_self_contained_with_the_public_boundary(self):
        model = onnx.load(self.asset_path, load_external_data=False)
        graph = self.provenance["graph"]
        self.assertEqual([(i.domain, i.version) for i in model.opset_import], [("", graph["opset"])])
        self.assertEqual(len(model.graph.node), graph["nodes"])
        self.assertEqual(len(model.graph.initializer), graph["initializers"])
        self.assertEqual(
            [(v.name, v.type.tensor_type.elem_type) for v in model.graph.input],
            [(graph["input"]["name"], TensorProto.FLOAT)],
        )
        self.assertEqual(
            [(v.name, v.type.tensor_type.elem_type) for v in model.graph.output],
            [(graph["output"]["name"], TensorProto.FLOAT)],
        )
        for initializer in model.graph.initializer:
            self.assertEqual(initializer.data_location, TensorProto.DEFAULT, initializer.name)
        float_values = {TensorProto.FLOAT16: 0, TensorProto.FLOAT: 0}
        for initializer in model.graph.initializer:
            if initializer.data_type in float_values:
                count = 1
                for extent in initializer.dims:
                    count *= extent
                float_values[initializer.data_type] += count
        # All learned weights are FP16; the FP32 tensors are a handful of small constants.
        self.assertGreater(float_values[TensorProto.FLOAT16], 30_000_000)
        self.assertLess(float_values[TensorProto.FLOAT], 1000)


if __name__ == "__main__":
    unittest.main()
