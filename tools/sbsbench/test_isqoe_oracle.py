from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.sbsbench import isqoe_oracle


class IsqoeOracleTests(unittest.TestCase):
    def test_dropout_api_adapter_accepts_module_and_float(self):
        class Dropout:
            p = 0.25

        self.assertEqual(isqoe_oracle._dropout_probability(Dropout()), 0.25)
        self.assertEqual(isqoe_oracle._dropout_probability(0.5), 0.5)

    def test_missing_official_checkout_fails_before_model_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(
                    isqoe_oracle.IsqoeUnavailable, "not an Apple ml-isqoe"):
                isqoe_oracle.IsqoeModel(root, checkpoint, "cpu")

    def test_repository_revision_reads_loose_ref_without_invoking_git(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref = root / ".git" / "refs" / "heads" / "main"
            ref.parent.mkdir(parents=True)
            (root / ".git" / "HEAD").write_text(
                "ref: refs/heads/main\n", encoding="ascii")
            revision = "0123456789abcdef0123456789abcdef01234567"
            ref.write_text(revision + "\n", encoding="ascii")
            self.assertEqual(isqoe_oracle._repository_revision(root), revision)

    def test_packed_eye_order_is_measured_both_ways(self):
        import torch

        model = isqoe_oracle.IsqoeModel.__new__(isqoe_oracle.IsqoeModel)
        model.torch = torch
        model.device = "cpu"
        model.preprocess = lambda image: torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        model.model = lambda left, right: left.mean((1, 2, 3)) + 2.0 * right.mean(
            (1, 2, 3))

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sbs.png"
            left = np.full((4, 6, 3), 51, dtype=np.uint8)
            right = np.full((4, 6, 3), 102, dtype=np.uint8)
            Image.fromarray(np.concatenate((left, right), axis=1)).save(path)
            payload = model.evaluate_path(path)

        metrics = payload["metrics"]
        self.assertAlmostEqual(metrics["isqoe_score"], 1.0, places=6)
        self.assertAlmostEqual(metrics["isqoe_swapped_score"], 0.8, places=6)
        self.assertAlmostEqual(metrics["isqoe_mean_score"], 0.9, places=6)
        self.assertAlmostEqual(metrics["isqoe_worst_score"], 1.0, places=6)
        self.assertAlmostEqual(metrics["isqoe_eye_order_delta"], 0.2, places=6)
        self.assertFalse(payload["training_label_eligible"])
        self.assertEqual(len(payload["input_sha256"]), 64)

    def test_non_finite_model_score_is_rejected(self):
        import torch

        model = isqoe_oracle.IsqoeModel.__new__(isqoe_oracle.IsqoeModel)
        model.torch = torch
        model.device = "cpu"
        model.preprocess = lambda image: torch.zeros((3, 2, 2))
        model.model = lambda left, right: torch.tensor([float("nan")])
        image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            model.evaluate(image, image)

    def test_provenance_property_does_not_rehash_checkpoint(self):
        model = isqoe_oracle.IsqoeModel.__new__(isqoe_oracle.IsqoeModel)
        model._provenance = {"checkpoint_sha256": "a" * 64}
        first = model.provenance
        first["checkpoint_sha256"] = "mutated"
        self.assertEqual(model.provenance["checkpoint_sha256"], "a" * 64)

    def test_odd_packed_width_is_rejected(self):
        model = isqoe_oracle.IsqoeModel.__new__(isqoe_oracle.IsqoeModel)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "odd.png"
            Image.fromarray(np.zeros((4, 13, 3), dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, "width must be even"):
                model.evaluate_path(path)


if __name__ == "__main__":
    unittest.main()
