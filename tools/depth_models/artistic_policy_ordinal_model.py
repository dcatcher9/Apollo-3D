"""Experimental monotone ordinal safety head on frozen DA-V2 features.

This module deliberately does not alter the shipping two-channel artistic
policy.  It reuses the exact frozen multiscale/depth/DPT feature path and
replaces only the small global head for offline frontier experiments.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from artistic_policy_model import (
    POLICY_FEATURE_CONTRACT,
    ArtisticPolicyModel,
    load_depth_anything_small,
    use_dynamic_onnx_position_encoding,
)
from artistic_policy_ordinal_contract import FRONTIER_SIZE, SCALES


ORDINAL_POLICY_CONTRACT = "safe-frontier-ordinal-apollo-v2"
ORDINAL_CHECKPOINT_SCHEMA = 3
ORDINAL_OUTPUT_NAME = "artistic_safety_frontier"
ORDINAL_OUTPUT_SEMANTICS = {
    "contract": ORDINAL_POLICY_CONTRACT,
    "output": ORDINAL_OUTPUT_NAME,
    "scales": list(SCALES),
    "channels": "monotone calibrated point probability that each connected scale is safe",
    "selection": (
        "highest contiguous calibrated point probability; validated lower "
        "bounds may be substituted explicitly"
    ),
    "abstention": "identity is unauthorized when its probability misses the threshold",
}


def _logit(value):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class OrdinalArtisticPolicyModel(ArtisticPolicyModel):
    """DA-V2 Small plus a behavior-neutral 26-threshold safety head."""

    def __init__(self, depth_model):
        super().__init__(depth_model)
        self.ordinal_head = nn.Sequential(
            nn.LayerNorm(self.policy_feature_size),
            nn.Linear(self.policy_feature_size, 192),
            nn.GELU(),
            nn.Linear(192, FRONTIER_SIZE),
        )
        # Remove the scalar head so checkpoints cannot accidentally contain or
        # train both policy interpretations.
        del self.global_head
        self._initialize_neutral_ordinal_policy()

    def _initialize_neutral_ordinal_policy(self):
        last = self.ordinal_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, -8.0)
        with torch.no_grad():
            # Every threshold starts below any deployable confidence gate.
            last.bias[0] = _logit(0.02)

    @staticmethod
    def monotone_probabilities(raw):
        """Map arbitrary logits to non-increasing connected-safe probabilities."""
        if raw.ndim != 2 or raw.shape[1] != FRONTIER_SIZE:
            raise RuntimeError("ordinal policy logits must have shape [N,26]")
        base = raw[:, :1]
        decrements = F.softplus(raw[:, 1:]) + 1e-4
        logits = torch.cat((
            base,
            base - torch.cumsum(decrements, dim=1),
        ), dim=1)
        return torch.sigmoid(logits)

    def _policy_from_features(self, features, depth, dpt_context):
        raw = self.ordinal_head(
            self._policy_features(features, depth, dpt_context)
        )
        return self.monotone_probabilities(raw)

    def forward_policy_features(self, pooled):
        """Train the ordinal head from cached frozen DA-V2 features."""
        return self.monotone_probabilities(self.ordinal_head(pooled))


def ordinal_policy_state_dict(model):
    """Return only trainable ordinal-head weights, never frozen DA-V2 weights."""
    return {
        f"ordinal_head.{key}": value.detach().cpu()
        for key, value in model.ordinal_head.state_dict().items()
    }


def load_ordinal_policy_state_dict(model, state):
    prefix = "ordinal_head."
    if (not isinstance(state, dict) or not state or
            any(not key.startswith(prefix) for key in state)):
        raise RuntimeError("checkpoint does not contain only ordinal-head state")
    model.ordinal_head.load_state_dict({
        key[len(prefix):]: value for key, value in state.items()
    }, strict=True)


__all__ = [
    "POLICY_FEATURE_CONTRACT",
    "ORDINAL_POLICY_CONTRACT",
    "ORDINAL_CHECKPOINT_SCHEMA",
    "ORDINAL_OUTPUT_NAME",
    "ORDINAL_OUTPUT_SEMANTICS",
    "OrdinalArtisticPolicyModel",
    "load_depth_anything_small",
    "use_dynamic_onnx_position_encoding",
    "ordinal_policy_state_dict",
    "load_ordinal_policy_state_dict",
]
