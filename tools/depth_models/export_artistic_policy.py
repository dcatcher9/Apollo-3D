#!/usr/bin/env python3
"""Export the shared-feature artistic DA-V2 model to TensorRT-ready ONNX."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import onnx
import torch
from torch import nn

import artistic_policy_evaluation_contract as evaluation_contract
from artistic_policy_model import (
    ART_SCALE_DELTA_MAX,
    ArtisticPolicyModel,
    load_depth_anything_small,
    load_policy_state,
    use_dynamic_onnx_position_encoding,
)
from artistic_geometry_contract import allowlist_sha256, validate_allowlist
import depth_input_color as input_color
import merge_artistic_geometry_labels as label_merge


SEALED_TEST_APPROVAL_CONTRACT = evaluation_contract.SEALED_APPROVAL_CONTRACT
EVALUATION_SCHEMA = evaluation_contract.EVALUATION_SCHEMA
EXPORT_METADATA_SCHEMA = evaluation_contract.EXPORT_METADATA_SCHEMA
MAX_UNSAFE_CEILING_OVERSHOOT_SCALE = (
    evaluation_contract.MAX_UNSAFE_CEILING_OVERSHOOT_SCALE
)
MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE = (
    evaluation_contract.MAX_FILM_BALANCED_UNSAFE_CEILING_OVERSHOOT_SCALE
)


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
    input_manifest = checkpoint.get("input_variant_manifest")
    label_merge.validate_input_variant_manifest(input_manifest)
    input_manifest_hash = label_merge.input_variant_manifest_sha256(input_manifest)
    color_contract_hash = input_color.color_contract_sha256()
    if (checkpoint.get("input_variant_manifest_sha256") != input_manifest_hash or
            checkpoint.get("depth_input_color_contract_sha256") !=
            color_contract_hash):
        raise RuntimeError("policy checkpoint has stale input color provenance")
    condition_target_contract = checkpoint.get("condition_target_contract")
    if condition_target_contract != label_merge.CONDITION_TARGET_CONTRACT:
        raise RuntimeError("policy checkpoint has stale shared-target provenance")
    return (actual_depth_hash, metric_sha256, geometry_allowlist, geometry_hash,
            input_manifest, input_manifest_hash, color_contract_hash,
            condition_target_contract)


def _require_hash(payload, key, length=64):
    value = payload.get(key)
    if (not isinstance(value, str) or len(value) != length or
            any(character not in "0123456789abcdef" for character in value)):
        raise RuntimeError(f"sealed-test evaluation has invalid {key}")
    return value


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
    unsafe_ceiling_overshoot = (
        evaluation_contract.validate_unsafe_ceiling_overshoot(
            payload, decision, "sealed-test evaluation"
        )
    )
    condition_target_contract = checkpoint.get("condition_target_contract")
    if (condition_target_contract != label_merge.CONDITION_TARGET_CONTRACT or
            payload.get("condition_target_contract") !=
            condition_target_contract):
        raise RuntimeError(
            "sealed-test evaluation uses a stale shared-target contract"
        )
    input_manifest = checkpoint.get("input_variant_manifest")
    label_merge.validate_input_variant_manifest(input_manifest)
    expected_hdr_whites = sorted(
        int(variant["windows_sdr_white_level_raw"])
        for variant in input_manifest["variants"]
        if variant["kind"] == input_color.INPUT_KIND_WINDOWS_HDR
    )
    if tuple(expected_hdr_whites) != (
            evaluation_contract.EXPECTED_HDR_WHITE_LEVELS_RAW):
        raise RuntimeError("checkpoint lacks the production HDR white anchors")
    runtime_regime_acceptance = (
        evaluation_contract.validate_runtime_regime_acceptance(
            payload, decision, condition_target_contract, expected_hdr_whites
        )
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
        "input_variant_manifest_sha256": checkpoint.get(
            "input_variant_manifest_sha256"
        ),
        "depth_input_color_contract_sha256": checkpoint.get(
            "depth_input_color_contract_sha256"
        ),
    }
    _require_hash(identities, "checkpoint_sha256")
    _require_hash(identities, "active_split_sha256")
    _require_hash(identities, "metric_sha256", length=16)
    _require_hash(identities, "label_fitter_identity_sha256")
    _require_hash(identities, "deployment_geometry_allowlist_sha256")
    _require_hash(identities, "input_variant_manifest_sha256")
    _require_hash(identities, "depth_input_color_contract_sha256")
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
    if (payload.get("input_variant_manifest") != input_manifest or
            label_merge.input_variant_manifest_sha256(input_manifest) !=
            identities["input_variant_manifest_sha256"] or
            payload.get("depth_input_color_contract_sha256") !=
            identities["depth_input_color_contract_sha256"]):
        raise RuntimeError(
            "sealed-test evaluation input color contract does not match checkpoint"
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
        "runtime_regime_acceptance": runtime_regime_acceptance,
        "condition_target_contract": condition_target_contract,
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
     deployment_geometry_allowlist_sha256, input_variant_manifest,
     input_variant_manifest_sha256,
     depth_input_color_contract_sha256,
     condition_target_contract) = validate_export_provenance(
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
        "schema": EXPORT_METADATA_SCHEMA,
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
        "input_variant_manifest": input_variant_manifest,
        "input_variant_manifest_sha256": input_variant_manifest_sha256,
        "depth_input_color_contract_sha256": depth_input_color_contract_sha256,
        "condition_target_contract": condition_target_contract,
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
