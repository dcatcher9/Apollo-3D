"""Keep the production control distinct from the continuous Host adapter."""
import unittest
from unittest import mock

import run_eval


class JointPlaneGainPolicyTests(unittest.TestCase):
    def mode(self, configured, *extra):
        with mock.patch.object(run_eval, "conf_value", return_value=configured):
            return run_eval.expected_joint_plane_mode("unused.conf", list(extra))

    def test_live_opt_in_selects_gain_policy(self):
        self.assertEqual(self.mode("false"), 0)
        self.assertEqual(self.mode("enabled"), 3)
        self.assertEqual(self.mode("false", "--joint-plane-experiment", "on"), 3)

    def test_supported_modes_are_explicit(self):
        self.assertEqual(self.mode("false", "--joint-plane-mode", "3"), 3)
        self.assertEqual(self.mode("true", "--joint-plane-mode", "0"), 0)

    def test_conflicting_overrides_cannot_mislabel_an_arm(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.mode("false", "--joint-plane-mode", "3",
                      "--joint-plane-experiment", "off")

    def test_unknown_policy_is_rejected(self):
        for value in ("1", "2", "4", "true", "-1", "1.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.mode("false", "--joint-plane-mode", value)


if __name__ == "__main__":
    unittest.main()
