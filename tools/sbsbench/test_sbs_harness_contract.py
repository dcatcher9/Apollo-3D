#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sbs_harness_contract as contract  # noqa: E402


class HarnessContractTests(unittest.TestCase):
    def test_schema_and_color_mapping(self):
        self.assertEqual(contract.HARNESS_SCHEMA, 28)
        self.assertEqual(
            contract.expected_metric_preview_encoding(
                contract.COLOR_MODE_HDR, contract.HDR_SOURCE_SIMULATED
            ),
            "source-relative-srgb-from-scrgb-white-normalized-v1",
        )
        self.assertEqual(
            contract.expected_metric_preview_encoding(
                contract.COLOR_MODE_HDR, contract.HDR_SOURCE_NATIVE_PQ
            ),
            "perceptual-srgb-from-native-scrgb-reinhard-v1",
        )
        for color_mode in (
                contract.COLOR_MODE_SDR, contract.COLOR_MODE_LINEAR_SDR):
            self.assertEqual(
                contract.expected_metric_preview_encoding(
                    color_mode, contract.HDR_SOURCE_SDR
                ),
                "native-srgb-v1",
            )

    def test_missing_wrong_and_unknown_values_fail_closed(self):
        for encoding in (None, "native-srgb-v1"):
            with self.assertRaisesRegex(RuntimeError, "metric preview encoding"):
                contract.validate_metric_preview_encoding(
                    contract.COLOR_MODE_HDR, encoding, "clip",
                    contract.HDR_SOURCE_SIMULATED,
                )
        with self.assertRaisesRegex(RuntimeError, "unsupported HDR source kind"):
            contract.expected_metric_preview_encoding(
                contract.COLOR_MODE_HDR, contract.HDR_SOURCE_SDR
            )
        with self.assertRaisesRegex(RuntimeError, "unsupported harness color mode"):
            contract.expected_metric_preview_encoding("future-color")


if __name__ == "__main__":
    unittest.main()
