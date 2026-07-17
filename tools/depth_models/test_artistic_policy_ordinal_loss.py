#!/usr/bin/env python3

import math
import unittest

import torch

import artistic_policy_ordinal_contract as contract
import artistic_policy_ordinal_loss as loss


def tested(scale, safe=True):
    return {
        "scale": scale,
        "safe": safe,
        "realized_pop_pct": scale,
        "failure_causes": [] if safe else ["source_halo_p95:hard"],
    }


def finite_evidence():
    return contract.build_frontier_evidence([
        tested(1.00),
        tested(1.02, safe=False),
    ])


def right_censored_evidence():
    return contract.build_frontier_evidence([
        tested(scale) for scale in contract.SCALES
    ])


def left_censored_evidence():
    return contract.build_frontier_evidence([
        tested(1.00, safe=False),
    ])


class ArtisticPolicyOrdinalLossTests(unittest.TestCase):
    def test_interval_nll_uses_finite_bracket_mass(self):
        probabilities = torch.linspace(0.9, 0.1, contract.FRONTIER_SIZE,
                                       dtype=torch.float64).unsqueeze(0)
        value = loss.interval_censored_nll(
            probabilities, [finite_evidence()]
        )
        expected = -math.log(float(probabilities[0, 0] -
                                   probabilities[0, 1]))
        self.assertAlmostEqual(float(value), expected)

    def test_interval_nll_handles_right_and_left_censoring(self):
        right = torch.linspace(0.95, 0.55, contract.FRONTIER_SIZE,
                               dtype=torch.float64)
        left = torch.linspace(0.30, 0.05, contract.FRONTIER_SIZE,
                              dtype=torch.float64)
        values = loss.interval_censored_nll(
            torch.stack((right, left)),
            [right_censored_evidence(), left_censored_evidence()],
            reduction="none",
        )
        self.assertAlmostEqual(float(values[0]), -math.log(float(right[-1])))
        self.assertAlmostEqual(float(values[1]),
                               -math.log(1.0 - float(left[0])))

    def test_brier_masks_unknown_bins_and_weights_unsafe_prediction(self):
        first = torch.linspace(0.8, 0.1, contract.FRONTIER_SIZE,
                               dtype=torch.float64)
        changed_unknown = first.clone()
        changed_unknown[2:] = torch.linspace(
            0.70, 0.05, contract.FRONTIER_SIZE - 2,
            dtype=torch.float64,
        )
        probabilities = torch.stack((first, changed_unknown))
        evidence = [finite_evidence(), finite_evidence()]
        values = loss.known_bin_asymmetric_brier(
            probabilities,
            evidence,
            unsafe_overprediction_weight=4.0,
            reduction="none",
        )
        expected = ((float(first[0]) - 1.0) ** 2 +
                    4.0 * float(first[1]) ** 2) / 5.0
        self.assertAlmostEqual(float(values[0]), expected)
        self.assertAlmostEqual(float(values[1]), expected)

    def test_unsafe_overprediction_receives_larger_gradient(self):
        probabilities = torch.linspace(
            0.8, 0.1, contract.FRONTIER_SIZE,
            dtype=torch.float64,
        ).unsqueeze(0).requires_grad_()
        value = loss.known_bin_asymmetric_brier(
            probabilities,
            [finite_evidence()],
            unsafe_overprediction_weight=4.0,
        )
        value.backward()
        safe_gradient = abs(float(probabilities.grad[0, 0]))
        unsafe_gradient = abs(float(probabilities.grad[0, 1]))
        self.assertGreater(unsafe_gradient, safe_gradient)
        self.assertTrue(torch.all(probabilities.grad[0, 2:] == 0.0))

    def test_combined_loss_preserves_useful_interval_gradients(self):
        probabilities = torch.linspace(
            0.9, 0.1, contract.FRONTIER_SIZE,
            dtype=torch.float64,
        ).unsqueeze(0).requires_grad_()
        values = loss.ordinal_frontier_loss(
            probabilities,
            [finite_evidence()],
            interval_weight=1.0,
            brier_weight=0.25,
        )
        self.assertEqual(set(values), {
            "loss", "interval_nll", "known_bin_brier",
        })
        values["loss"].backward()
        self.assertLess(float(probabilities.grad[0, 0]), 0.0)
        self.assertGreater(float(probabilities.grad[0, 1]), 0.0)

    def test_collapsed_finite_bracket_has_recovery_gradient(self):
        probabilities = torch.full(
            (1, contract.FRONTIER_SIZE),
            0.5,
            dtype=torch.float64,
            requires_grad=True,
        )
        value = loss.interval_censored_nll(
            probabilities, [finite_evidence()]
        )
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertLess(float(probabilities.grad[0, 0]), 0.0)
        self.assertGreater(float(probabilities.grad[0, 1]), 0.0)

    def test_invalid_probability_or_evidence_contract_fails_closed(self):
        valid = torch.linspace(
            0.9, 0.1, contract.FRONTIER_SIZE
        ).unsqueeze(0)
        invalid_cases = (
            valid[:, :-1],
            torch.flip(valid, dims=(1,)),
            valid.clone().fill_(float("nan")),
        )
        for probabilities in invalid_cases:
            with self.subTest(shape=tuple(probabilities.shape)):
                with self.assertRaises(RuntimeError):
                    loss.interval_censored_nll(
                        probabilities, [finite_evidence()]
                    )
        with self.assertRaises(RuntimeError):
            loss.interval_censored_nll(valid, [])
        with self.assertRaises(RuntimeError):
            loss.known_bin_asymmetric_brier(
                valid, [finite_evidence()],
                unsafe_overprediction_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
