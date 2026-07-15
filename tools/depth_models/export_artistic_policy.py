#!/usr/bin/env python3
"""Export the shared-feature artistic DA-V2 model to TensorRT-ready ONNX."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import onnx
import torch
from torch import nn

from artistic_policy_model import (
    ART_SCALE_DELTA_MAX,
    ArtisticPolicyModel,
    load_depth_anything_small,
    load_policy_state,
    use_dynamic_onnx_position_encoding,
)
from artistic_geometry_contract import allowlist_sha256, validate_allowlist


SEALED_TEST_APPROVAL_CONTRACT = "sealed-test-artistic-policy-v2"
EVALUATION_SCHEMA = 11
MAX_UNSAFE_CEILING_OVERSHOOT_SCALE = 0.05
MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE = 0.01


class Fp16InternalFp32Io(nn.Module):
    """Keep Apollo's FP32 buffer contract while exporting FP16 model math."""

    def __init__(self, model):
        super().__init__()
        self.model = model.half()

    def forward(self, pixel_values):
        outputs = self.model(pixel_values.half())
        return tuple(output.float() for output in outputs)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_export_provenance(checkpoint, depth_weights):
    actual_depth_hash = sha256(depth_weights)
    expected_depth_hash = checkpoint.get("depth_weights_sha256")
    if expected_depth_hash != actual_depth_hash:
        raise RuntimeError(
            "supplied DA-V2 weights do not match the policy checkpoint: "
            f"{actual_depth_hash} != {expected_depth_hash}"
        )
    metric_sha256 = checkpoint.get("metric_sha256")
    if (not isinstance(metric_sha256, str) or len(metric_sha256) != 16 or
            any(character not in "0123456789abcdef" for character in metric_sha256)):
        raise RuntimeError("policy checkpoint lacks exact metric provenance")
    geometry_allowlist = checkpoint.get("deployment_geometry_allowlist")
    validate_allowlist(geometry_allowlist)
    geometry_hash = allowlist_sha256(geometry_allowlist)
    if checkpoint.get("deployment_geometry_allowlist_sha256") != geometry_hash:
        raise RuntimeError("policy checkpoint has stale deployment geometry provenance")
    return actual_depth_hash, metric_sha256, geometry_allowlist, geometry_hash


def _require_hash(payload, key, length=64):
    value = payload.get(key)
    if (not isinstance(value, str) or len(value) != length or
            any(character not in "0123456789abcdef" for character in value)):
        raise RuntimeError(f"sealed-test evaluation has invalid {key}")
    return value


def _validate_unsafe_ceiling_overshoot(payload, decision):
    evidence = payload.get("unsafe_ceiling_overshoot")
    if not isinstance(evidence, dict):
        raise RuntimeError("sealed-test evaluation lacks unsafe-ceiling evidence")
    numeric = {}
    for key in (
            "maximum_scale", "maximum_limit_scale",
            "film_balanced_mean_scale", "film_balanced_mean_limit_scale",
            "film_balanced_overshoot_rate_pct"):
        value = evidence.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(float(value))):
            raise RuntimeError(
                f"sealed-test evaluation has invalid unsafe-ceiling {key}"
            )
        numeric[key] = float(value)
    if (numeric["maximum_scale"] < 0.0 or
            numeric["film_balanced_mean_scale"] < 0.0 or
            not 0.0 <= numeric["film_balanced_overshoot_rate_pct"] <= 100.0):
        raise RuntimeError("sealed-test evaluation has invalid unsafe-ceiling evidence")
    if (abs(numeric["maximum_limit_scale"] -
            MAX_UNSAFE_CEILING_OVERSHOOT_SCALE) > 1e-12 or
            abs(numeric["film_balanced_mean_limit_scale"] -
                MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE) > 1e-12):
        raise RuntimeError("sealed-test evaluation uses different unsafe-ceiling limits")
    if (evidence.get("maximum_pass") is not True or
            evidence.get("film_balanced_mean_pass") is not True or
            numeric["maximum_scale"] > MAX_UNSAFE_CEILING_OVERSHOOT_SCALE + 1e-9 or
            numeric["film_balanced_mean_scale"] >
            MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE + 1e-9):
        raise RuntimeError("sealed-test evaluation failed unsafe-ceiling guards")
    guards = decision.get("guards")
    if (not isinstance(guards, dict) or
            decision.get("unsafe_overshoot_guard_required") is not True or
            guards.get("unsafe_ceiling_maximum") is not True or
            guards.get("unsafe_ceiling_film_balanced_mean") is not True):
        raise RuntimeError("sealed-test decision lacks unsafe-ceiling guards")
    if decision.get("unsafe_ceiling_overshoot") != evidence:
        raise RuntimeError("sealed-test decision has inconsistent unsafe-ceiling evidence")
    return evidence


def validate_sealed_test_approval(checkpoint, checkpoint_sha256, evaluation):
    """Bind export to an accepted sealed-test result for this checkpoint."""
    try:
        evaluation_bytes = evaluation.read_bytes()
        evaluation_sha256 = hashlib.sha256(evaluation_bytes).hexdigest()
        payload = json.loads(evaluation_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read sealed-test evaluation: {evaluation}"
        ) from error
    if not isinstance(payload, dict) or payload.get("schema") != EVALUATION_SCHEMA:
        raise RuntimeError("incompatible sealed-test evaluation schema")
    if payload.get("split") != "test":
        raise RuntimeError("export approval requires the sealed test split")
    decision = payload.get("decision")
    if not isinstance(decision, dict) or decision.get("accepted") is not True:
        raise RuntimeError("sealed-test evaluation did not accept this checkpoint")
    unsafe_ceiling_overshoot = _validate_unsafe_ceiling_overshoot(
        payload, decision
    )

    identities = {
        "checkpoint_sha256": checkpoint_sha256,
        "active_split_sha256": checkpoint.get("active_split_sha256"),
        "metric_sha256": checkpoint.get("metric_sha256"),
        "label_fitter_identity_sha256": checkpoint.get(
            "label_fitter_identity_sha256"
        ),
        "deployment_geometry_allowlist_sha256": checkpoint.get(
            "deployment_geometry_allowlist_sha256"
        ),
    }
    _require_hash(identities, "checkpoint_sha256")
    _require_hash(identities, "active_split_sha256")
    _require_hash(identities, "metric_sha256", length=16)
    _require_hash(identities, "label_fitter_identity_sha256")
    _require_hash(identities, "deployment_geometry_allowlist_sha256")
    for key, expected in identities.items():
        actual = _require_hash(
            payload, key, length=16 if key == "metric_sha256" else 64
        )
        if actual != expected:
            raise RuntimeError(
                f"sealed-test evaluation {key} does not match the checkpoint: "
                f"{actual} != {expected}"
            )

    test_labels_sha256 = _require_hash(payload, "test_labels_sha256")
    geometry_allowlist = payload.get("deployment_geometry_allowlist")
    validate_allowlist(geometry_allowlist)
    if (geometry_allowlist != checkpoint.get("deployment_geometry_allowlist") or
            allowlist_sha256(geometry_allowlist) != identities[
                "deployment_geometry_allowlist_sha256"
            ]):
        raise RuntimeError(
            "sealed-test evaluation deployment geometry allow-list does not match checkpoint"
        )
    expected_films = checkpoint.get("sealed_test_productions")
    if (not isinstance(expected_films, (list, tuple)) or not expected_films or
            any(not isinstance(film, str) or not film for film in expected_films) or
            len(set(expected_films)) != len(expected_films)):
        raise RuntimeError("checkpoint lacks a valid sealed-test production set")
    expected_films = sorted(expected_films)
    actual_films = payload.get("val_films")
    if (not isinstance(actual_films, list) or
            any(not isinstance(film, str) for film in actual_films) or
            sorted(actual_films) != expected_films):
        raise RuntimeError(
            "sealed-test evaluation productions do not match the checkpoint"
        )
    return evaluation_sha256, {
        "contract": SEALED_TEST_APPROVAL_CONTRACT,
        "evaluation_sha256": evaluation_sha256,
        "evaluation_schema": EVALUATION_SCHEMA,
        "split": "test",
        "decision_accepted": True,
        "unsafe_ceiling_overshoot": unsafe_ceiling_overshoot,
        **identities,
        "test_labels_sha256": test_labels_sha256,
        "sealed_test_productions": expected_films,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-anything-root", required=True, type=Path)
    parser.add_argument("--depth-weights", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint_bytes = args.policy.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
    )
    (depth_weights_sha256, metric_sha256, deployment_geometry_allowlist,
     deployment_geometry_allowlist_sha256) = validate_export_provenance(
        checkpoint, args.depth_weights
    )
    evaluation_sha256, approval_contract = validate_sealed_test_approval(
        checkpoint, checkpoint_sha256, args.evaluation
    )
    depth_model = load_depth_anything_small(
        args.depth_anything_root, args.depth_weights
    )
    model = ArtisticPolicyModel(depth_model)
    checkpoint = load_policy_state(model, args.policy, checkpoint)
    use_dynamic_onnx_position_encoding(model)
    model = Fp16InternalFp32Io(model.eval()).eval()

    # Apollo's builder does not globally force reduced precision. Match the shipping
    # DA-V2 ONNX explicitly: FP32 I/O around FP16 weights and internal computation.
    example = torch.zeros((1, 3, 434, 770), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        args.output,
        input_names=["pixel_values"],
        output_names=["predicted_depth", "artistic_global"],
        dynamic_axes={
            "pixel_values": {2: "height", 3: "width"},
            "predicted_depth": {1: "height", 2: "width"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    metadata = {
        "schema": 4,
        "deployed_model": args.output.stem,
        "base_depth_model": checkpoint["policy_baseline"]["depth_model"],
        "onnx_sha256": sha256(args.output),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "policy_contract": checkpoint.get("policy_contract"),
        "policy_feature_contract": checkpoint.get("policy_feature_contract"),
        "policy_baseline": checkpoint.get("policy_baseline"),
        "output_semantics": checkpoint.get("output_semantics"),
        "checkpoint_selection": {
            "key": checkpoint.get("checkpoint_selection_key"),
            "development_acceptance": checkpoint.get("development_acceptance"),
        },
        "depth_weights_sha256": depth_weights_sha256,
        "metric_sha256": metric_sha256,
        "deployment_geometry_allowlist": deployment_geometry_allowlist,
        "deployment_geometry_allowlist_sha256": (
            deployment_geometry_allowlist_sha256
        ),
        "evaluation_sha256": evaluation_sha256,
        "approval_contract": approval_contract,
        "labels_sha256": checkpoint.get("labels_sha256"),
        "internal_precision": "float16",
        "input": {"name": "pixel_values", "dtype": "float32", "shape": [1, 3, "H", "W"]},
        "outputs": {
            "predicted_depth": {"dtype": "float32", "shape": [1, "H", "W"]},
            "artistic_global": {
                "dtype": "float32",
                "shape": [1, 2],
                "channels": [
                    "safe_scale_ceiling",
                    "safe_ceiling_confidence",
                ],
            },
        },
        "bounds": {
            "scale_delta_max": ART_SCALE_DELTA_MAX,
        },
        "runtime": {
            "confidence_semantics": "hard actionable probability",
            "action_threshold": 0.5,
            "inactive_ceiling": 1.0,
            "ceiling_bounds": [1.0, 1.0 + ART_SCALE_DELTA_MAX],
            "preset_rules": {
                "safe_cap": "confidence >= 0.5 ? clamp(ceiling, 1.0, 1.5) : 1.0",
                "clean": "1.0",
                "balanced": "1.0 + 0.5 * (safe_cap - 1.0)",
                "immersive": "safe_cap",
            },
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
