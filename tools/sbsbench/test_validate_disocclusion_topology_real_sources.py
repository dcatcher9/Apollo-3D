"""Contract tests for the bounded authenticated-source topology validator."""

import os
import sys
import unittest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import validate_disocclusion_topology_real_sources as validator  # noqa: E402


class AuthenticatedTopologyValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = validator.build_report(
            [validator.real_suite.DEFAULT_CLIPS_ROOT], frames_per_clip=1,
            max_width=96, max_clips_per_suite=1, workers=1)

    def test_bounded_core_run_passes_with_only_expected_abstentions(self):
        summary = self.report["summary"]
        self.assertEqual(summary["clips"], 1)
        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["checks"], 36)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["unexpected_abstentions"], 0)
        self.assertEqual(summary["abstained"], summary["expected_abstentions"])
        self.assertEqual(summary["acceptable"], summary["checks"])

    def test_report_never_qualifies_training_labels(self):
        self.assertEqual(self.report["training_label_qualification"], "blocked")
        self.assertFalse(self.report["auto_promotes_labels"])
        self.assertIn("simulated HDR", " ".join(self.report["limitations"]))

    def test_false_masks_are_explicit_expected_abstentions(self):
        checks = self.report["samples"][0]["checks"]
        false_masks = [check for check in checks
                       if check["family"] == "false_forward_mask"]
        self.assertEqual(len(false_masks), len(validator.VARIANTS))
        self.assertTrue(all(check["status"] == "abstain" for check in false_masks))
        self.assertTrue(all(check["expected_status"] == "abstain"
                            for check in false_masks))
        self.assertTrue(all(check["acceptable"] for check in false_masks))

    def test_every_deployment_variant_is_exercised(self):
        sample = self.report["samples"][0]
        self.assertEqual(set(sample["variants"]), set(validator.VARIANTS))
        for name, variant in validator.VARIANTS.items():
            measured = sample["variants"][name]
            self.assertEqual(measured["geometry"], variant["geometry"])
            self.assertEqual(measured["regime"], variant["regime"])


if __name__ == "__main__":
    unittest.main()
