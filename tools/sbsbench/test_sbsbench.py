import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbsbench
import run_eval


class EvalContractTests(unittest.TestCase):
    def test_warp_override_uses_last_explicit_value(self):
        self.assertEqual(run_eval.extra_value(
            ["--warp", "apollo", "--warp", "vd3d"], "--warp", "apollo"), "vd3d")

    def test_warp_is_read_from_config(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("# sbs_3d_warp = apollo\nsbs_3d_warp = vd3d # active\n")
            path = fh.name
        try:
            self.assertEqual(run_eval.conf_value(path, "sbs_3d_warp", "apollo"), "vd3d")
        finally:
            os.unlink(path)

    def test_phase_shift_recovers_known_translation(self):
        rng = np.random.default_rng(1234)
        a = rng.random((64, 64))
        b = np.roll(a, shift=(2, -5), axis=(0, 1))
        dy, dx = sbsbench.phase_shift(a, b)
        self.assertAlmostEqual(dy, -2.0, places=5)
        self.assertAlmostEqual(dx, 5.0, places=5)

    def test_sequence_joins_by_frame_identity(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            sbs = np.zeros((16, 32, 3), dtype=np.uint8)
            src = np.zeros((16, 16, 3), dtype=np.uint8)
            Image.fromarray(sbs).save(os.path.join(seq, "sbs_00007.png"))
            Image.fromarray(src).save(os.path.join(frames, "frame_00007.png"))
            rows, agg = sbsbench.measure_sequence(seq, frames)
            self.assertEqual(rows[0]["_frame_id"], 7)
            self.assertEqual(agg["_n"], 1)

    def test_sequence_rejects_positional_mispairing(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seq")
            frames = os.path.join(root, "frames")
            os.makedirs(seq)
            os.makedirs(frames)
            blank = np.zeros((16, 32, 3), dtype=np.uint8)
            Image.fromarray(blank).save(os.path.join(seq, "sbs_00008.png"))
            Image.fromarray(blank[:, :16]).save(os.path.join(frames, "frame_00007.png"))
            with self.assertRaisesRegex(ValueError, "frame-id mismatch"):
                sbsbench.measure_sequence(seq, frames)

    def test_duplicate_numeric_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(os.path.join(root, "frame_1.png"))
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(os.path.join(root, "frame_01.jpg"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                sbsbench.indexed_files(os.path.join(root, "frame_*.*"), "frame_")


if __name__ == "__main__":
    unittest.main()
