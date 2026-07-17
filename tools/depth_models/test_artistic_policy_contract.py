#!/usr/bin/env python3

import copy
import unittest

import artistic_policy_ordinal_contract as contract


def tested(scale, safe=True, pop=None, causes=None):
    return {
        "scale": scale,
        "safe": safe,
        "realized_pop_pct": scale if pop is None else pop,
        "failure_causes": [] if safe else (causes or ["source_halo_p95:hard"]),
    }


class ArtisticOrdinalFrontierContractTests(unittest.TestCase):
    def test_lattice_is_exact_identity_to_maximum(self):
        self.assertEqual(contract.FRONTIER_SIZE, 26)
        self.assertEqual(contract.SCALES[0], 1.0)
        self.assertEqual(contract.SCALES[-1], 1.5)
        self.assertEqual(contract.scale_index(1.26), 13)
        for value in (0.9, 1.01, 1.5001, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    contract.scale_index(value)

    def test_first_unsafe_is_an_interval_not_an_exact_ceiling(self):
        evidence = contract.build_frontier_evidence([
            tested(1.00, pop=2.0),
            tested(1.02, pop=2.1),
            tested(1.04, pop=2.2),
            tested(1.06, safe=False, pop=2.3,
                   causes=["source_halo_p95:hard"]),
        ])
        self.assertEqual(evidence["highest_proven_safe_scale"], 1.04)
        self.assertEqual(evidence["first_proven_unsafe_scale"], 1.06)
        self.assertFalse(evidence["right_censored"])
        self.assertEqual(evidence["states"][:5], [
            "safe", "safe", "safe", "unsafe", "unknown",
        ])
        self.assertAlmostEqual(
            evidence["realized_pop_gain_over_identity_pct"], 0.2
        )

    def test_safe_maximum_is_right_censored_and_preserves_plateau(self):
        rows = []
        for scale in contract.SCALES:
            rows.append(tested(scale, pop=min(scale, 1.30)))
        evidence = contract.build_frontier_evidence(rows)
        self.assertTrue(evidence["right_censored"])
        self.assertIsNone(evidence["first_proven_unsafe_scale"])
        self.assertEqual(evidence["highest_proven_safe_scale"], 1.5)
        self.assertEqual(evidence["first_maximum_pop_scale"], 1.3)
        self.assertTrue(all(state == "safe" for state in evidence["states"]))

    def test_identity_hard_failure_has_no_proven_safe_action(self):
        evidence = contract.build_frontier_evidence([
            tested(1.0, safe=False, pop=1.0,
                   causes=["source_coverage_pct:hard"]),
        ])
        self.assertFalse(evidence["identity_feasible"])
        self.assertTrue(evidence["left_censored"])
        self.assertIsNone(evidence["highest_proven_safe_scale"])
        self.assertEqual(evidence["first_proven_unsafe_scale"], 1.0)
        self.assertEqual(evidence["states"][0], "unsafe")
        self.assertTrue(all(
            state == "unknown" for state in evidence["states"][1:]
        ))

    def test_incomplete_or_invented_frontiers_fail_closed(self):
        cases = (
            [tested(1.0), tested(1.04, safe=False)],
            [tested(1.0), tested(1.02, safe=False), tested(1.04)],
            [{
                "scale": 1.0,
                "safe": False,
                "realized_pop_pct": 1.0,
                "failure_causes": [],
            }],
            [tested(1.0, pop=2.0), tested(1.02, pop=1.9)],
            [tested(1.0), tested(1.02)],
        )
        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaises(RuntimeError):
                    contract.build_frontier_evidence(rows)

    def test_canonical_validator_rejects_tampering(self):
        evidence = contract.build_frontier_evidence([
            tested(1.0), tested(1.02, safe=False),
        ])
        contract.validate_frontier_evidence(evidence)
        tampered = copy.deepcopy(evidence)
        tampered["states"][2] = "unsafe"
        with self.assertRaisesRegex(RuntimeError, "canonical"):
            contract.validate_frontier_evidence(tampered)

    def test_runtime_selection_stops_at_first_probability_failure(self):
        probabilities = [0.99] * contract.FRONTIER_SIZE
        probabilities[4] = 0.40
        probabilities[5] = 0.99
        self.assertEqual(
            contract.select_contiguous_safe_scale(probabilities, 0.95),
            1.06,
        )
        probabilities[0] = 0.40
        self.assertIsNone(
            contract.select_contiguous_safe_scale(probabilities, 0.95)
        )
        with self.assertRaises(RuntimeError):
            contract.select_contiguous_safe_scale(
                probabilities[:-1], 0.95
            )


if __name__ == "__main__":
    unittest.main()
