"""Shared-feature Depth Anything V2 Small artistic-policy model.

The frozen DA-V2 encoder is evaluated once.  Its normal DPT decoder produces the
unchanged relative-depth tensor while one deliberately small head predicts:

* a shot-level safe scale ceiling for Apollo's current full binocular disparity; and
* the probability that the learned ceiling is actionable by the later shot latch.

The policy head consumes pooled features from every DINO stage plus detached,
scale-invariant summaries of the unchanged depth and DPT decoder fields.  It
therefore sees both scene semantics and geometric/edge risk without adding a
dense output or allowing policy training to modify the production depth path.

The ceiling is dimensionless. A runtime preset requests its artistic multiplier,
then clamps that request to the learned ceiling. The bounds here define the model's numerical
contract, not Apollo's comfort contract;
production still clamps the rendered result against its validated comfort envelope.
"""

from __future__ import annotations

import importlib
import math
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from artistic_policy_contract import (
    ART_SCALE_DELTA_MAX,
    ARTISTIC_GLOBAL_SIZE,
)


POLICY_FEATURE_CONTRACT = "multiscale-dino-depth-dpt-stats-v1"
POLICY_STAT_SIZE = 8
POLICY_CHECKPOINT_SCHEMA = 8
POLICY_CONTRACT = "safe-frontier-multistyle-apollo-v1"
POLICY_OUTPUT_SEMANTICS = {
    "artistic_global_0": "safe_scale_ceiling",
    "artistic_global_1": "safe_ceiling_confidence",
    "confidence_semantics": "hard actionable probability",
    "action_threshold": 0.5,
    "preset_rules": {
        "safe_cap": (
            "safe_ceiling_confidence >= 0.5 ? "
            "clamp(safe_scale_ceiling, 1.0, 1.5) : 1.0"
        ),
        "clean": "1.0",
        "balanced": "1.0 + 0.5 * (safe_cap - 1.0)",
        "immersive": "safe_cap",
    },
}


class _RecoverableUnitClamp(torch.autograd.Function):
    """Exact [0,1] forward clamp with a straight-through training gradient.

    A normal clamp strands samples after a negative update, while a sigmoid or
    rational asymptote cannot represent Apollo's labelled maximum at a finite
    logit. The exported graph is an ordinary ONNX Clip; only training uses the
    recoverable straight-through derivative.
    """

    @staticmethod
    def forward(ctx, raw):
        return torch.clamp(raw, 0.0, 1.0)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def _load_depth_anything_class(depth_anything_root: Path):
    root = str(depth_anything_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("depth_anything_v2.dpt")
    return module.DepthAnythingV2


def load_depth_anything_small(depth_anything_root: Path, weights: Path):
    """Load the official Apache-2.0 DA-V2 Small model."""
    depth_anything_v2 = _load_depth_anything_class(depth_anything_root)
    model = depth_anything_v2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
    )
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def _logit(value: float) -> float:
    value = min(max(value, 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


class ArtisticPolicyModel(nn.Module):
    """DA-V2 Small with a behavior-neutral global stereo-control head."""

    def __init__(self, depth_model: nn.Module):
        super().__init__()
        self.depth_model = depth_model
        embed_dim = int(depth_model.pretrained.embed_dim)
        intermediate_count = len(
            depth_model.intermediate_layer_idx[depth_model.encoder]
        ) if hasattr(depth_model, "intermediate_layer_idx") else 4
        self.policy_feature_size = embed_dim * 2 * intermediate_count + POLICY_STAT_SIZE
        self.global_head = nn.Sequential(
            nn.LayerNorm(self.policy_feature_size),
            nn.Linear(self.policy_feature_size, 192),
            nn.GELU(),
            nn.Linear(192, ARTISTIC_GLOBAL_SIZE),
        )
        self.base_frozen = False
        self._initialize_neutral_policy()

    def _initialize_neutral_policy(self):
        # Identity ceiling and low confidence. An untrained checkpoint therefore
        # exactly preserves Apollo's current symmetric warp.
        global_last = self.global_head[-1]
        nn.init.zeros_(global_last.weight)
        nn.init.zeros_(global_last.bias)
        with torch.no_grad():
            global_last.bias[1] = _logit(0.02)

    def freeze_base(self):
        self.base_frozen = True
        self.depth_model.requires_grad_(False)
        self.depth_model.eval()

    def _features(self, x: torch.Tensor):
        indices = self.depth_model.intermediate_layer_idx[self.depth_model.encoder]
        if self.base_frozen:
            with torch.no_grad():
                return self.depth_model.pretrained.get_intermediate_layers(
                    x, indices, return_class_token=True
                )
        return self.depth_model.pretrained.get_intermediate_layers(
            x, indices, return_class_token=True
        )

    @staticmethod
    def _spatial_stats(field: torch.Tensor):
        """Return four detached, scale-invariant field/risk summaries."""
        field = field.detach()
        if field.ndim == 3:
            field = field[:, None]
        rms = torch.sqrt(field.square().mean(dim=(1, 2, 3)) + 1e-6)
        mean = field.mean(dim=(1, 2, 3)) / rms
        centered = field - field.mean(dim=(1, 2, 3), keepdim=True)
        variation = torch.sqrt(
            centered.square().mean(dim=(1, 2, 3)) + 1e-6
        ) / rms
        grad_x = (field[:, :, :, 1:] - field[:, :, :, :-1]).abs()
        grad_y = (field[:, :, 1:, :] - field[:, :, :-1, :]).abs()
        edge_x = grad_x.mean(dim=(1, 2, 3)) / rms
        edge_y = grad_y.mean(dim=(1, 2, 3)) / rms
        return torch.stack((mean, variation, edge_x, edge_y), dim=1)

    def _policy_features(self, features, depth, dpt_context):
        pooled = []
        for tokens, class_token in features:
            pooled.extend((class_token, tokens.mean(dim=1)))
        pooled.extend((
            self._spatial_stats(depth),
            self._spatial_stats(dpt_context),
        ))
        return torch.cat(pooled, dim=1)

    def _policy_from_features(self, features, depth, dpt_context):
        raw_global = self.global_head(
            self._policy_features(features, depth, dpt_context)
        )
        return torch.stack(
            (
                self._safe_ceiling(raw_global[:, 0]),
                torch.sigmoid(raw_global[:, 1]),
            ),
            dim=1,
        )

    def forward_policy(self, pixel_values: torch.Tensor):
        features = self._features(pixel_values)
        height, width = pixel_values.shape[-2:]
        depth, dpt_context = self._decode_depth(
            features, height // 14, width // 14, height, width,
            return_policy_context=True,
        )
        depth = F.relu(depth).squeeze(1)
        return self._policy_from_features(features, depth, dpt_context)

    def policy_features(self, pixel_values: torch.Tensor):
        """Return frozen multiscale, depth, and decoder features used by the head."""
        features = self._features(pixel_values)
        height, width = pixel_values.shape[-2:]
        depth, dpt_context = self._decode_depth(
            features, height // 14, width // 14, height, width,
            return_policy_context=True,
        )
        depth = F.relu(depth).squeeze(1)
        return self._policy_features(features, depth, dpt_context)

    def last_policy_features(self, pixel_values: torch.Tensor):
        """Compatibility alias for the upgraded frozen policy feature vector."""
        return self.policy_features(pixel_values)

    def forward_policy_features(self, pooled: torch.Tensor):
        """Train policy heads from cached frozen-backbone features."""
        raw_global = self.global_head(pooled)
        return torch.stack(
            (
                self._safe_ceiling(raw_global[:, 0]),
                torch.sigmoid(raw_global[:, 1]),
            ),
            dim=1,
        )

    @staticmethod
    def _safe_ceiling(raw: torch.Tensor):
        """Map finite logits exactly onto the full identity-to-maximum interval."""
        return 1.0 + _RecoverableUnitClamp.apply(raw) * ART_SCALE_DELTA_MAX

    def _decode_depth(self, features, patch_h, patch_w, height, width,
                      return_policy_context=False):
        """DA-V2's DPT decoder with a dynamic-shape-safe final resize."""
        head = self.depth_model.depth_head
        projected = []
        for index, feature in enumerate(features):
            tokens = feature[0]
            tokens = tokens.permute(0, 2, 1).reshape(
                tokens.shape[0], tokens.shape[-1], patch_h, patch_w
            )
            tokens = head.projects[index](tokens)
            projected.append(head.resize_layers[index](tokens))
        layer_1, layer_2, layer_3, layer_4 = projected
        layer_1 = head.scratch.layer1_rn(layer_1)
        layer_2 = head.scratch.layer2_rn(layer_2)
        layer_3 = head.scratch.layer3_rn(layer_3)
        layer_4 = head.scratch.layer4_rn(layer_4)
        path_4 = head.scratch.refinenet4(layer_4, size=layer_3.shape[2:])
        path_3 = head.scratch.refinenet3(path_4, layer_3, size=layer_2.shape[2:])
        path_2 = head.scratch.refinenet2(path_3, layer_2, size=layer_1.shape[2:])
        path_1 = head.scratch.refinenet1(path_2, layer_1)
        depth = head.scratch.output_conv1(path_1)
        depth = F.interpolate(
            depth, (height, width), mode="bilinear", align_corners=True
        )
        depth = head.scratch.output_conv2(depth)
        if return_policy_context:
            return depth, path_1
        return depth

    def forward(self, pixel_values: torch.Tensor):
        height, width = pixel_values.shape[-2:]
        features = self._features(pixel_values)
        depth, dpt_context = self._decode_depth(
            features, height // 14, width // 14, height, width,
            return_policy_context=True,
        )
        depth = F.relu(depth).squeeze(1)
        global_policy = self._policy_from_features(features, depth, dpt_context)
        return depth, global_policy


def use_dynamic_onnx_position_encoding(model: ArtisticPolicyModel):
    """Replace DINOv2's Python-float resize with an ONNX dynamic-shape resize.

    The official implementation computes a bicubic scale factor through Python floats,
    which legacy ONNX tracing freezes to the example resolution.  Size-based interpolation
    is numerically equivalent for patch-aligned Apollo inputs and exports Shape/Gather nodes.
    """
    backbone = model.depth_model.pretrained

    def interpolate_pos_encoding_dynamic(self, x, image_h, image_w):
        previous_dtype = x.dtype
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, :1]
        patch_pos_embed = pos_embed[:, 1:]
        grid = int(math.sqrt(patch_pos_embed.shape[1]))
        patch_pos_embed = patch_pos_embed.reshape(
            1, grid, grid, x.shape[-1]
        ).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(image_h // self.patch_size, image_w // self.patch_size),
            mode="bicubic",
            align_corners=False,
            antialias=self.interpolate_antialias,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(previous_dtype)

    backbone.interpolate_pos_encoding = types.MethodType(
        interpolate_pos_encoding_dynamic, backbone
    )


def policy_state_dict(model: ArtisticPolicyModel):
    """Return only trainable policy weights; official DA-V2 weights stay external."""
    return {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith("global_head.")
    }


def load_policy_state(model: ArtisticPolicyModel, checkpoint: Path, payload=None):
    if payload is None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != POLICY_CHECKPOINT_SCHEMA:
        raise RuntimeError("incompatible artistic-policy checkpoint schema")
    if payload.get("policy_contract") != POLICY_CONTRACT:
        raise RuntimeError("incompatible artistic-policy objective contract")
    if payload.get("policy_feature_contract") != POLICY_FEATURE_CONTRACT:
        raise RuntimeError("incompatible artistic-policy feature contract")
    if payload.get("output_semantics") != POLICY_OUTPUT_SEMANTICS:
        raise RuntimeError("incompatible artistic-policy output semantics")
    if not isinstance(payload.get("policy_baseline"), dict):
        raise RuntimeError("artistic-policy checkpoint lacks baseline provenance")
    metric_sha256 = payload.get("metric_sha256")
    if (not isinstance(metric_sha256, str) or len(metric_sha256) != 16 or
            any(character not in "0123456789abcdef" for character in metric_sha256)):
        raise RuntimeError("artistic-policy checkpoint lacks metric provenance")
    state = payload.get("policy_state", payload)
    if not isinstance(state, dict):
        raise RuntimeError("artistic-policy checkpoint has invalid policy state")
    depth_keys = sorted(
        key for key in state
        if isinstance(key, str) and key.startswith("depth_model.")
    )
    if depth_keys:
        raise RuntimeError(
            "artistic-policy checkpoint must not contain frozen depth-model weights: "
            + ", ".join(depth_keys)
        )
    expected_keys = set(policy_state_dict(model))
    actual_keys = set(state)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(
            (str(key) for key in actual_keys - expected_keys)
        )
        raise RuntimeError(
            f"incompatible artistic-policy checkpoint: missing={missing}, "
            f"unexpected={unexpected}"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith("depth_model.")]
    if missing or unexpected:
        raise RuntimeError(
            f"incompatible artistic-policy checkpoint: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return payload
