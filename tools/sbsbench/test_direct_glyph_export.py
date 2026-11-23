import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class DirectGlyphExportTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "requires the ONNX Runtime CPU provider")
    def test_folded_encoder_preserves_owned_scores_including_edge_strokes(self):
        import onnxruntime as ort

        models = Path(__file__).resolve().parents[2] / "src_assets/windows/assets/models"
        rgb = np.random.default_rng(401).random((1, 3, 28, 512), dtype=np.float32)
        rgb[:, :, [0, 1, 26, 27], :] = 0
        rgb[:, :, [0, 27], ::2] = 1

        def owned_logits(filename, top, bottom):
            session = ort.InferenceSession(str(models / filename), providers=["CPUExecutionProvider"])
            tile = np.pad(rgb, ((0, 0), (0, 0), (top, bottom), (0, 0)), mode="edge")
            logits, = session.run(["logits"], {"rgb": tile})
            return logits[:, :, top:top + 28, 36:476]

        reference = owned_logits("direct_glyph_stroke_h144_w512_mixed.onnx", 56, 60)
        folded = owned_logits("direct_glyph_stroke_h52_w512_mixed_batch.onnx", 12, 12)
        np.testing.assert_allclose(folded, reference, rtol=0, atol=1e-5)

    def test_reproduce_serial_and_batched_mixed_models(self):
        repo = Path(__file__).resolve().parents[2]
        assets = repo / "src_assets/windows/assets"
        with tempfile.TemporaryDirectory() as directory:
            for height, variant in ((144, "mixed"), (52, "mixed_batch")):
                with self.subTest(height=height, variant=variant):
                    receipt = assets / "models" / f"direct_glyph_stroke_h{height}_w512_{variant}.provenance.json"
                    spec = json.loads(receipt.read_text(encoding="utf-8"))
                    source = receipt.parent / spec["source_onnx"]["file"]
                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), spec["source_onnx"]["sha256"])
                    output = Path(directory) / f"{variant}.onnx"
                    subprocess.run([sys.executable, repo / spec["exporter"], source, output,
                                    *spec.get("export_args", [])], check=True)
                    model = output.read_bytes()
                    self.assertEqual(model, (assets / spec["artifact"]["path"]).read_bytes())
                    self.assertEqual(len(model), spec["artifact"]["bytes"])
                    self.assertEqual(hashlib.sha256(model).hexdigest(), spec["artifact"]["sha256"])
