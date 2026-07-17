#!/usr/bin/env python3

import unittest

import artistic_policy_ordinal_contract as ordinal_contract
import select_ordinal_render_frontiers as selector


SPECS = {
    "source_halo_p95": {
        "role": "primary", "axis": "warp", "better": "lower",
        "rel_tol": 0.2, "abs_floor": 0.5,
    },
    "static_jitter_p95": {
        "role": "primary", "axis": "stability", "better": "lower",
        "rel_tol": 0.2, "abs_floor": 0.5,
    },
    "source_coverage_pct": {
        "role": "hard", "axis": "integrity", "better": "higher",
        "rel_tol": 0.0, "abs_floor": 1.0, "hard_min": 90.0,
    },
}


def aggregate(pop, halo=4.0, jitter=2.0, coverage=99.0):
    return {
        "exact_pop_spread_pct": pop,
        "source_halo_p95": halo,
        "static_jitter_p95": jitter,
        "source_coverage_pct": coverage,
    }


def full_safe_grid():
    return {
        scale: aggregate(2.0 + (scale - 1.0) * 2.0)
        for scale in ordinal_contract.SCALES
    }


class OrdinalRenderFrontierSelectorTests(unittest.TestCase):
    def test_first_failure_is_retained_and_farther_bins_are_unknown(self):
        candidates = full_safe_grid()
        candidates[1.06] = aggregate(2.3, halo=6.0)
        # A disconnected later candidate may look safe; it must not become a
        # negative or positive label after the first frontier failure.
        candidates[1.08] = aggregate(2.4, halo=4.0)
        result = selector.select_clip_frontier(
            aggregate(2.0), candidates, SPECS
        )
        self.assertEqual(result["highest_proven_safe_scale"], 1.04)
        self.assertEqual(result["first_proven_unsafe_scale"], 1.06)
        self.assertEqual(result["states"][:5], [
            "safe", "safe", "safe", "unsafe", "unknown",
        ])
        self.assertEqual(result["first_unsafe_failure_causes"], [
            "source_halo_p95:regression",
        ])
        self.assertEqual(len(result["tested_bins"]), 4)

    def test_safe_maximum_is_right_censored(self):
        result = selector.select_clip_frontier(
            aggregate(2.0), full_safe_grid(), SPECS
        )
        self.assertTrue(result["right_censored"])
        self.assertEqual(result["highest_proven_safe_scale"], 1.5)
        self.assertIsNone(result["first_proven_unsafe_scale"])
        self.assertAlmostEqual(
            result["realized_pop_gain_over_identity_pct"], 1.0
        )

    def test_identity_hard_failure_is_left_censored(self):
        result = selector.select_clip_frontier(
            aggregate(2.0, coverage=82.0),
            {1.0: aggregate(2.0, coverage=82.0)},
            SPECS,
        )
        self.assertTrue(result["left_censored"])
        self.assertFalse(result["identity_feasible"])
        self.assertEqual(result["first_proven_unsafe_scale"], 1.0)
        self.assertEqual(result["first_unsafe_failure_causes"], [
            "source_coverage_pct:hard",
        ])

    def test_identity_missing_or_relative_mismatch_is_not_a_label(self):
        missing_pop = aggregate(2.0)
        del missing_pop["exact_pop_spread_pct"]
        with self.assertRaisesRegex(RuntimeError, "no finite"):
            selector.select_clip_frontier(
                aggregate(2.0), {1.0: missing_pop}, SPECS
            )
        with self.assertRaisesRegex(
                RuntimeError, "incomplete or inconsistent"):
            selector.select_clip_frontier(
                aggregate(2.0),
                {1.0: aggregate(2.0, halo=6.0)}, SPECS,
            )

    def test_safe_incomplete_prefix_and_scale_gaps_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "stops without"):
            selector.select_clip_frontier(
                aggregate(2.0),
                {1.0: aggregate(2.0), 1.02: aggregate(2.1)},
                SPECS,
            )
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            selector.select_clip_frontier(
                aggregate(2.0),
                {1.0: aggregate(2.0), 1.04: aggregate(2.2, halo=6.0)},
                SPECS,
            )

    def test_missing_candidate_pop_is_broken_evidence_not_a_label(self):
        missing_pop = aggregate(2.1)
        del missing_pop["exact_pop_spread_pct"]
        with self.assertRaisesRegex(RuntimeError, "no finite"):
            selector.select_clip_frontier(
                aggregate(2.0),
                {1.0: aggregate(2.0), 1.02: missing_pop},
                SPECS,
            )

    def test_report_trigger_is_diagnostic_but_ordinal_hard_is_safety(self):
        diagnostic_specs = {
            **SPECS,
            "source_halo_p95": {
                **SPECS["source_halo_p95"], "trigger": 4.5,
            },
        }
        # Trigger-only evidence does not authorize an ordinal hard boundary.
        result = selector.select_clip_frontier(
            aggregate(2.0), full_safe_grid(), diagnostic_specs
        )
        self.assertTrue(result["right_censored"])

        hard_specs = {
            **diagnostic_specs,
            "source_halo_p95": {
                **diagnostic_specs["source_halo_p95"],
                "ordinal_hard_max": 4.5,
            },
        }
        candidates = full_safe_grid()
        candidates[1.04] = aggregate(2.2, halo=4.6)
        result = selector.select_clip_frontier(
            aggregate(2.0), candidates, hard_specs
        )
        self.assertEqual(result["first_proven_unsafe_scale"], 1.04)
        self.assertEqual(result["first_unsafe_failure_causes"], [
            "source_halo_p95:ordinal-hard-max",
        ])


if __name__ == "__main__":
    unittest.main()
