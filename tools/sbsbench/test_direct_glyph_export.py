import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DirectGlyphExportTests(unittest.TestCase):
    def test_reproduce_serial_and_batched_mixed_models(self):
        repo = Path(__file__).resolve().parents[2]
        assets = repo / "src_assets/windows/assets"
        with tempfile.TemporaryDirectory() as directory:
            for variant in ("mixed", "mixed_batch"):
                with self.subTest(variant=variant):
                    receipt = assets / "models" / f"direct_glyph_stroke_h144_w512_{variant}.provenance.json"
                    spec = json.loads(receipt.read_text(encoding="utf-8"))
                    source = receipt.parent / spec["source_onnx"]["file"]
                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), spec["source_onnx"]["sha256"])
                    output = Path(directory) / f"{variant}.onnx"
                    subprocess.run([sys.executable, repo / spec["exporter"], source, output,
                                    *spec.get("export_args", [])], check=True)
                    model = output.read_bytes()
                    self.assertEqual(model, (assets / spec["artifact"]["path"]).read_bytes())
                    self.assertEqual(hashlib.sha256(model).hexdigest(), spec["artifact"]["sha256"])
