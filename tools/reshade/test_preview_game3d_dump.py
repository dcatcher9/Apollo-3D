"""End-to-end CPU tests of the native Game 3D preview CLI and manifest replacement."""

import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


class PreviewCliTests(unittest.TestCase):
    executable = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sunshine-preview-cli-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def package(self):
        color = bytes([8, 127, 244, 255, 1, 2, 3, 255])
        depth = struct.pack("<2f", 0.25, 0.75)
        descriptions = [
            ("source_color", 2, 1, 28, color),
            ("raw_depth", 2, 1, 41, depth),
            ("sbs", 4, 1, 28, color + color),
        ]
        self.native = {}
        artifacts = []
        for index, (kind, width, height, fmt, data) in enumerate(descriptions):
            name = f"{index}_{kind}.bin"
            self.native[name] = data
            (self.root / name).write_bytes(data)
            artifacts.append({"kind": kind, "file": name, "width": width, "height": height,
                              "dxgi_format": fmt, "row_bytes": width * 4, "byte_count": len(data)})
        self.original = {
            "schema": "sunshine.game3d.dump.v1", "status": "complete",
            "capture_id": 73, "artifacts": artifacts,
            "producer_metadata": {
                "color_space": 1, "replay": {"exact_fixture_parameter": 123},
                "render_parameters": {"depth_rect": [0, 0, 1, 1], "jitter_uv": [0, 0]},
            },
        }
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(self.original), encoding="utf-8")

    def run_cli(self):
        return subprocess.run([str(self.executable), str(self.root)], capture_output=True,
                              text=True, timeout=30, check=False)

    def test_replaces_read_manifest_and_preserves_replay_and_native_bytes(self):
        self.package()
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        updated = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(updated.pop("visualizations")["status"], "complete")
        self.assertEqual(updated, self.original)
        for name, data in self.native.items():
            self.assertEqual((self.root / name).read_bytes(), data)
        for name, expected_dimensions in (("color.png", (2, 1)), ("raw_depth.png", (2, 1)),
                                           ("final_sbs.png", (4, 1))):
            png = (self.root / name).read_bytes()
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", png[16:24]), expected_dimensions)
        self.assertTrue((self.root / "index.html").is_file())
        self.assertFalse((self.root / "manifest.preview.tmp").exists())
        # Re-running must also replace existing preview files and manifest safely.
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_consumed_ui_source_is_preserved_and_cannot_be_silently_skipped(self):
        self.package()
        payload = bytes([64, 128, 255, 0, 64, 128, 255, 128])
        ui = dict(kind="ui_source_color", artifact_id=33, file="ui_source_color.bin",
                  width=2, height=1, dxgi_format=28, row_bytes=8, byte_count=8)
        (self.root / ui["file"]).write_bytes(payload)
        self.original["artifacts"].append(ui)
        self.original["producer_metadata"]["replay"]["ui_alpha_source"] = "ui_source_color"
        self.manifest.write_text(json.dumps(self.original), encoding="utf-8")
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        from PIL import Image
        with Image.open(self.root / "ui_source_alpha.png") as image:
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
            self.assertEqual(image.getpixel((1, 0)), (128, 128, 128))
        self.assertEqual((self.root / ui["file"]).read_bytes(), payload)
        # A required renderer input is not a best-effort optional UI observation.
        (self.root / ui["file"]).unlink()
        result = self.run_cli()
        self.assertEqual(result.returncode, 2)

    def test_rejects_input_named_like_generated_file_before_writing(self):
        self.package()
        artifact = self.original["artifacts"][0]
        data = self.native[artifact["file"]]
        artifact["file"] = "COLOR.PNG"
        (self.root / "COLOR.PNG").write_bytes(data)
        self.manifest.write_text(json.dumps(self.original), encoding="utf-8")
        original_bytes = self.manifest.read_bytes()
        result = self.run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unsafe", result.stderr)
        self.assertEqual((self.root / "COLOR.PNG").read_bytes(), data)
        self.assertEqual(self.manifest.read_bytes(), original_bytes)
        self.assertFalse((self.root / "index.html").exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    PreviewCliTests.executable = args.executable.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining])
