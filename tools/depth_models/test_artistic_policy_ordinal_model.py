#!/usr/bin/env python3

import unittest

import torch

import artistic_policy_ordinal_contract as frontier
import artistic_policy_ordinal_model as ordinal


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = type("Pretrained", (), {"embed_dim": 4})()


class OrdinalArtisticPolicyModelTests(unittest.TestCase):
    def test_neutral_head_is_below_action_threshold_and_monotone(self):
        model = ordinal.OrdinalArtisticPolicyModel(Backbone())
        output = model.forward_policy_features(torch.zeros((3, 40)))
        self.assertEqual(tuple(output.shape), (3, frontier.FRONTIER_SIZE))
        self.assertTrue(torch.all(output < 0.5))
        self.assertTrue(torch.all(output[:, 1:] <= output[:, :-1]))
        self.assertTrue(torch.allclose(
            output[:, 0], torch.full((3,), 0.02), atol=1e-6
        ))

    def test_arbitrary_logits_are_monotone_with_useful_gradients(self):
        raw = torch.randn((4, frontier.FRONTIER_SIZE), requires_grad=True)
        probabilities = ordinal.OrdinalArtisticPolicyModel.monotone_probabilities(
            raw
        )
        self.assertTrue(torch.all(
            probabilities[:, 1:] <= probabilities[:, :-1]
        ))
        probabilities.sum().backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.abs().sum()), 0.0)

    def test_checkpoint_contains_only_ordinal_head(self):
        model = ordinal.OrdinalArtisticPolicyModel(Backbone())
        state = ordinal.ordinal_policy_state_dict(model)
        self.assertTrue(state)
        self.assertTrue(all(key.startswith("ordinal_head.") for key in state))
        self.assertFalse(any("depth_model" in key for key in state))
        restored = ordinal.OrdinalArtisticPolicyModel(Backbone())
        ordinal.load_ordinal_policy_state_dict(restored, state)
        for key, value in state.items():
            self.assertTrue(torch.equal(
                value,
                ordinal.ordinal_policy_state_dict(restored)[key],
            ))
        with self.assertRaises(RuntimeError):
            ordinal.load_ordinal_policy_state_dict(
                restored, {"depth_model.weight": torch.ones(1)}
            )

    def test_scalar_shipping_model_is_not_replaced(self):
        model = ordinal.OrdinalArtisticPolicyModel(Backbone())
        self.assertFalse(hasattr(model, "global_head"))
        self.assertEqual(ordinal.ORDINAL_OUTPUT_NAME,
                         "artistic_safety_frontier")
        self.assertEqual(ordinal.ORDINAL_CHECKPOINT_SCHEMA, 3)


if __name__ == "__main__":
    unittest.main()
