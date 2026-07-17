"""Losses for the experimental ordinal artistic-safety frontier.

This module is intentionally separate from ``train_artistic_policy.py`` and
the shipping two-channel scalar policy.  It trains survival probabilities
``S[i] = P(scale i is safe)`` from canonical interval/censoring evidence.
"""

from __future__ import annotations

import math

import torch

from artistic_policy_ordinal_contract import (
    FRONTIER_SIZE,
    scale_index,
    validate_frontier_evidence,
)


DEFAULT_UNSAFE_OVERPREDICTION_WEIGHT = 4.0


def _finite_nonnegative(value, description, *, positive=False):
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or
            not math.isfinite(float(value))):
        raise RuntimeError(f"{description} is not finite")
    value = float(value)
    if value < 0.0 or (positive and value <= 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise RuntimeError(f"{description} must be {qualifier}")
    return value


def _validate_reduction(reduction):
    if reduction not in {"none", "mean", "sum"}:
        raise RuntimeError("ordinal loss reduction must be none, mean, or sum")


def _reduce(values, reduction):
    _validate_reduction(reduction)
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    return values.mean()


def validate_frontier_probabilities(probabilities):
    """Fail closed unless probabilities satisfy the ordinal model contract."""
    if (not isinstance(probabilities, torch.Tensor) or
            probabilities.ndim != 2 or
            probabilities.shape[1] != FRONTIER_SIZE or
            not probabilities.is_floating_point()):
        raise RuntimeError(
            "frontier probabilities must be a floating [N,26] tensor"
        )
    if probabilities.shape[0] < 1:
        raise RuntimeError("frontier probability batch is empty")
    detached = probabilities.detach()
    if not bool(torch.isfinite(detached).all()):
        raise RuntimeError("frontier probabilities are not finite")
    if bool(((detached < 0.0) | (detached > 1.0)).any()):
        raise RuntimeError("frontier probabilities are outside [0,1]")
    if bool((detached[:, 1:] > detached[:, :-1]).any()):
        raise RuntimeError("frontier probabilities are not non-increasing")
    return probabilities


def _validate_evidence_batch(evidence_batch, batch_size):
    if (not isinstance(evidence_batch, (list, tuple)) or
            len(evidence_batch) != batch_size):
        raise RuntimeError(
            "frontier evidence count does not match the probability batch"
        )
    return [validate_frontier_evidence(value) for value in evidence_batch]


def _likelihood_mass(probabilities, evidence_batch):
    masses = []
    for row, evidence in zip(probabilities, evidence_batch):
        if evidence["left_censored"]:
            # Identity itself failed: the latent safe frontier is below 1.00.
            mass = 1.0 - row[0]
        elif evidence["right_censored"]:
            # Every tested action through the endpoint was safe.
            lower = scale_index(evidence["highest_proven_safe_scale"])
            mass = row[lower]
        else:
            # The safe frontier is in [last safe, first unsafe).
            lower = scale_index(evidence["highest_proven_safe_scale"])
            upper = scale_index(evidence["first_proven_unsafe_scale"])
            mass = row[lower] - row[upper]
        masses.append(mass)
    return torch.stack(masses)


def interval_censored_nll(probabilities, evidence_batch, *, reduction="mean",
                          epsilon=None):
    """Negative log likelihood for finite, right-, and left-censored labels."""
    probabilities = validate_frontier_probabilities(probabilities)
    evidence_batch = _validate_evidence_batch(
        evidence_batch, probabilities.shape[0]
    )
    if epsilon is None:
        epsilon = torch.finfo(probabilities.dtype).eps
    epsilon = _finite_nonnegative(
        epsilon, "ordinal likelihood epsilon", positive=True
    )
    masses = _likelihood_mass(probabilities, evidence_batch)
    # Adding epsilon, rather than clamping the mass, preserves a recovery
    # gradient if two adjacent survival bins have numerically collapsed.
    losses = -torch.log(masses + epsilon)
    return _reduce(losses, reduction)


def known_bin_asymmetric_brier(
        probabilities, evidence_batch, *,
        unsafe_overprediction_weight=DEFAULT_UNSAFE_OVERPREDICTION_WEIGHT,
        reduction="mean"):
    """Brier score over measured bins only, emphasizing unsafe confidence.

    Safe bins have target one and unit weight.  A measured unsafe bin has
    target zero and the configured larger weight.  Unknown bins contribute
    neither a target nor a loss.  Each sample is normalized by its known-bin
    weights before applying the requested batch reduction.
    """
    probabilities = validate_frontier_probabilities(probabilities)
    evidence_batch = _validate_evidence_batch(
        evidence_batch, probabilities.shape[0]
    )
    unsafe_weight = _finite_nonnegative(
        unsafe_overprediction_weight,
        "unsafe overprediction weight",
        positive=True,
    )
    sample_losses = []
    for row, evidence in zip(probabilities, evidence_batch):
        targets = []
        predictions = []
        weights = []
        for index, state in enumerate(evidence["states"]):
            if state == "unknown":
                continue
            predictions.append(row[index])
            if state == "safe":
                targets.append(row.new_tensor(1.0))
                weights.append(row.new_tensor(1.0))
            else:
                targets.append(row.new_tensor(0.0))
                weights.append(row.new_tensor(unsafe_weight))
        prediction = torch.stack(predictions)
        target = torch.stack(targets)
        weight = torch.stack(weights)
        sample_losses.append(
            (weight * (prediction - target).square()).sum() / weight.sum()
        )
    return _reduce(torch.stack(sample_losses), reduction)


def ordinal_frontier_loss(
        probabilities, evidence_batch, *, interval_weight=1.0,
        brier_weight=0.25,
        unsafe_overprediction_weight=DEFAULT_UNSAFE_OVERPREDICTION_WEIGHT,
        reduction="mean", epsilon=None):
    """Return the independent and combined experimental frontier losses."""
    interval_weight = _finite_nonnegative(
        interval_weight, "ordinal interval-loss weight"
    )
    brier_weight = _finite_nonnegative(
        brier_weight, "ordinal Brier-loss weight"
    )
    nll = interval_censored_nll(
        probabilities,
        evidence_batch,
        reduction=reduction,
        epsilon=epsilon,
    )
    brier = known_bin_asymmetric_brier(
        probabilities,
        evidence_batch,
        unsafe_overprediction_weight=unsafe_overprediction_weight,
        reduction=reduction,
    )
    return {
        "loss": interval_weight * nll + brier_weight * brier,
        "interval_nll": nll,
        "known_bin_brier": brier,
    }


__all__ = [
    "DEFAULT_UNSAFE_OVERPREDICTION_WEIGHT",
    "validate_frontier_probabilities",
    "interval_censored_nll",
    "known_bin_asymmetric_brier",
    "ordinal_frontier_loss",
]
