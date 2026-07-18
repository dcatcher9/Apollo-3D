"""Deterministic tests for metric-specific report evidence helpers."""

import ast
import os
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(SCRIPT_DIR, "build_report.py")


def load_report_functions(names, namespace=None):
    """Load selected pure functions without executing build_report's CLI entry point."""
    with open(REPORT_PATH, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), REPORT_PATH)
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names]
    missing = set(names) - {node.name for node in selected}
    if missing:
        raise AssertionError(f"missing report helpers: {sorted(missing)}")
    scope = {"np": np, "Image": Image, "ImageDraw": ImageDraw, "ImageFilter": ImageFilter}
    if namespace:
        scope.update(namespace)
    exec(compile(ast.Module(body=selected, type_ignores=[]), REPORT_PATH, "exec"), scope)
    return scope


class ReportVisualEvidenceTests(unittest.TestCase):
    def test_detector_heat_is_deterministic_and_separates_support_from_severity(self):
        scope = load_report_functions({"_expanded_map", "_artifact_analysis_rgb"})
        values = np.zeros((13, 17), np.float32)
        support = np.zeros(values.shape, bool)
        support[2, 3] = True
        support[9, 12] = True
        values[9, 12] = 4.0
        first = scope["_artifact_analysis_rgb"](values, support)
        second = scope["_artifact_analysis_rgb"](values, support)
        np.testing.assert_array_equal(first, second)
        # Support-only evidence is cyan; the actual strongest contribution is red/yellow.
        self.assertGreater(first[2, 3, 2], first[2, 3, 0])
        self.assertEqual(int(first[9, 12, 0]), 255)
        self.assertGreater(int(first[9, 12, 1]), 0)

    def test_static_support_visual_never_modifies_source_pixels(self):
        scope = load_report_functions({"_support_analysis_rgb", "_label_analysis_image"})
        rng = np.random.default_rng(83)
        source = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
        original = source.copy()
        support = np.zeros((20, 30), bool)
        support[4:12, 7:19] = True
        analysis = scope["_support_analysis_rgb"](support)
        scope["_label_analysis_image"](
            Image.fromarray(analysis), ["analysis mask (not source content)"])
        np.testing.assert_array_equal(source, original)
        np.testing.assert_array_equal(analysis[5, 8], (0, 210, 235))
        np.testing.assert_array_equal(analysis[0, 0], (0, 0, 0))

    def test_report_routes_row_shear_to_localized_detector_and_never_paints_source(self):
        with open(REPORT_PATH, encoding="utf-8") as fh:
            text = fh.read()
        dispatch = text[text.index("def visual_evidence_images"):text.index("def run_label")]
        self.assertIn('"warp_cross_row_shear_severity_pct"', dispatch)
        self.assertIn("return cross_row_shear_evidence(clip, idx, metric)", dispatch)
        self.assertNotIn("ARTIFACT_VISUAL_MAPS", dispatch)
        static = text[text.index("def static_jitter_evidence"):
                      text.index("def ground_truth_depth_evidence")]
        self.assertNotIn("source_a[", static)
        self.assertIn("durl(source.crop(crop)", static)
        self.assertIn("analysis mask (not source content)", static)


if __name__ == "__main__":
    unittest.main()
