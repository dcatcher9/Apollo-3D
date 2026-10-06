"""The evaluator admits one adaptive pipeline and rejects retired selectors."""
import unittest
import run_eval


class UnifiedPipelinePolicyTests(unittest.TestCase):
    def test_default_has_no_selector(self):
        run_eval.validate_unified_policy_options([])

    def test_removed_selectors_fail_before_gpu_work(self):
        for option in ("--joint-plane-experiment", "--joint-plane-mode"):
            for value in ("on", "off", "0", "3"):
                with self.subTest(option=option, value=value), self.assertRaisesRegex(ValueError, "removed"):
                    run_eval.validate_unified_policy_options([option, value])

    def test_other_artistic_and_execution_controls_remain_independent(self):
        run_eval.validate_unified_policy_options(["--pop-strength", "1.0", "--limit", "1"])
