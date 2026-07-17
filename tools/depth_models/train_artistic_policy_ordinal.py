#!/usr/bin/env python3
"""Train the separate offline DA-V2 ordinal artistic-safety head.

This experiment never reads or writes the shipping scalar-policy checkpoint.
It joins authenticated sparse ordinal render targets to authenticated
target-only source rows, freezes DA-V2, and optimizes only
``OrdinalArtisticPolicyModel.ordinal_head``.  Development selects checkpoints;
sealed-test rows are rejected before any image is decoded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

import artistic_geometry_contract as geometry_contract
import depth_input_color as input_color
import build_ordinal_frame_label_bundle as ordinal_bundle
import merge_ordinal_geometry_frontiers as ordinal_merge
import prepare_ordinal_full_frame_source_rows as source_rows
import train_artistic_policy as scalar_training
import run_eval
from artistic_policy_ordinal_contract import (
    SCALES,
    build_frontier_evidence,
    scale_index,
    select_contiguous_safe_scale,
)
from artistic_policy_ordinal_loss import ordinal_frontier_loss
from artistic_policy_ordinal_model import (
    ORDINAL_CHECKPOINT_SCHEMA,
    ORDINAL_OUTPUT_SEMANTICS,
    ORDINAL_POLICY_CONTRACT,
    POLICY_FEATURE_CONTRACT,
    OrdinalArtisticPolicyModel,
    load_depth_anything_small,
    ordinal_policy_state_dict,
    use_dynamic_onnx_position_encoding,
)


TRAINING_CONTRACT = "offline-ordinal-artistic-policy-training-v5"
CATALOG_SCHEMA = 5
CATALOG_CONTRACT = "apollo-ordinal-target-only-label-catalog-v1"
CHECKPOINT_SELECTION_CONTRACT = "ordinal-safety-first-selection-key-v1"
HISTORY_SCHEMA = 1
HISTORY_CONTRACT = "offline-ordinal-artistic-policy-history-v1"
SELECTION_CONFIDENCE = 0.95
CALIBRATION_ALPHA = 0.05
CALIBRATION_MAX_FAILURE_RATE = 0.05
CALIBRATION_STEPS = 150
CALIBRATION_LEARNING_RATE = 0.05
APPROVED_MONO_RUNTIME_VARIANTS = {
    "native_sdr", "hdr_raw1000", "hdr_raw2500", "hdr_raw6000",
}
THRESHOLDS_PATH = ordinal_bundle.THRESHOLDS_PATH
FEATURE_NEAR_EQUAL_RTOL = 1e-6
FEATURE_NEAR_EQUAL_ATOL = 1e-7
PLATEAU_POP_TOLERANCE_PCT = 1e-6


def sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not an object")
    return value


def _is_hex_digest(value, length):
    return (
        isinstance(value, str) and len(value) == length and
        all(character in "0123456789abcdef" for character in value)
    )


def _is_sha256(value):
    return _is_hex_digest(value, 64)


def _is_contract_sha256(value):
    """Validate the evaluator's intentionally truncated contract digest."""
    return _is_hex_digest(value, 16)


def current_metric_specs_sha256():
    thresholds = _load_json_object(THRESHOLDS_PATH, "metric thresholds")
    metrics = thresholds.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise RuntimeError("metric thresholds have no metric specifications")
    return ordinal_bundle.canonical_sha256(metrics)


def validate_training_catalog(path, frontier_paths, source_paths,
                              split_manifest):
    """Authenticate the exact orchestration publication admitted to training."""
    path = Path(path).resolve(strict=True)
    value = _load_json_object(path, "ordinal orchestration catalog")
    if (value.get("schema") != CATALOG_SCHEMA or
            value.get("contract") != CATALOG_CONTRACT or
            value.get("training_eligible") is not True or
            value.get("scope") != "full-active-train-development"):
        raise RuntimeError("ordinal orchestration catalog is not training-eligible")
    for field in (
            "active_split_sha256", "thresholds_sha256",
            "sbsbench_sha256", "run_eval_sha256", "executable_sha256"):
        if not _is_sha256(value.get(field)):
            raise RuntimeError(f"ordinal catalog lacks authenticated {field}")
    if not _is_contract_sha256(value.get("conf_sha256")):
        raise RuntimeError("ordinal catalog lacks authenticated conf_sha256")
    split_manifest = Path(split_manifest).resolve(strict=True)
    if (Path(value.get("active_split", "")).resolve(strict=False) !=
            split_manifest or
            value.get("active_split_sha256") != sha256(split_manifest)):
        raise RuntimeError("ordinal catalog active split is stale")
    if (value.get("metric_sha256") != run_eval.metric_contract_sha() or
            value.get("thresholds_sha256") != sha256(THRESHOLDS_PATH) or
            value.get("sbsbench_sha256") != sha256(
                ordinal_bundle.SBSBENCH_DIR / "sbsbench.py"
            ) or
            value.get("run_eval_sha256") != sha256(Path(run_eval.__file__))):
        raise RuntimeError("ordinal catalog metric contract is stale")
    identities = value.get("code_identities")
    if not isinstance(identities, dict) or not identities:
        raise RuntimeError("ordinal catalog lacks code identities")
    for role, identity in identities.items():
        if not isinstance(identity, dict):
            raise RuntimeError(f"ordinal catalog code identity is invalid: {role}")
        identity_path = Path(identity.get("path", ""))
        if (not identity_path.is_file() or
                identity.get("sha256") != sha256(identity_path)):
            raise RuntimeError(f"ordinal catalog code identity is stale: {role}")

    def exact_publication_paths(field, supplied):
        entries = value.get(field)
        if not isinstance(entries, list) or not entries:
            raise RuntimeError(f"ordinal catalog lacks {field}")
        catalog_paths = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError(f"ordinal catalog {field} entry is invalid")
            artifact_fields = [
                ("labels", "labels_sha256"),
                ("summary", "summary_sha256"),
            ]
            if field == "sources":
                artifact_fields.append(
                    ("source_contract", "source_contract_sha256")
                )
            artifacts = {}
            for path_field, hash_field in artifact_fields:
                artifact = Path(entry.get(path_field, "")).resolve(
                    strict=True
                )
                if entry.get(hash_field) != sha256(artifact):
                    raise RuntimeError(
                        f"ordinal catalog {field} publication is stale"
                    )
                artifacts[path_field] = artifact
            publication = artifacts["labels"]
            catalog_paths.append(publication)
        supplied_paths = [Path(item).resolve(strict=True) for item in supplied]
        if (len(catalog_paths) != len(set(catalog_paths)) or
                len(supplied_paths) != len(set(supplied_paths)) or
                set(catalog_paths) != set(supplied_paths)):
            raise RuntimeError(
                f"ordinal catalog {field} paths differ from trainer inputs"
            )
        return sorted(str(item) for item in catalog_paths)

    bundle_paths = exact_publication_paths("bundles", frontier_paths)
    admitted_source_paths = exact_publication_paths("sources", source_paths)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "active_split_sha256": value["active_split_sha256"],
        "metric_sha256": value["metric_sha256"],
        "thresholds_sha256": value["thresholds_sha256"],
        "sbsbench_sha256": value["sbsbench_sha256"],
        "run_eval_sha256": value["run_eval_sha256"],
        "conf_sha256": value["conf_sha256"],
        "executable_sha256": value["executable_sha256"],
        "bundle_paths": bundle_paths,
        "source_paths": admitted_source_paths,
    }


def runtime_regime(input_variant):
    input_color.validate_input_variant(input_variant)
    return (
        "sdr" if input_variant["kind"] == input_color.INPUT_KIND_SDR
        else "hdr"
    )


def runtime_variant_name(input_variant):
    """Return a report-only condition name derived from image provenance."""
    input_color.validate_input_variant(input_variant)
    kind = input_variant["kind"]
    if kind == input_color.INPUT_KIND_SDR:
        return "native_sdr"
    if kind == input_color.INPUT_KIND_NATIVE_PQ:
        return "native_pq"
    if kind == input_color.INPUT_KIND_WINDOWS_HDR:
        return "hdr_raw" + str(input_variant["windows_sdr_white_level_raw"])
    raise RuntimeError("unsupported ordinal input variant")


def validate_gain_rationale(value):
    if not isinstance(value, str):
        raise RuntimeError("minimum development pop-gain rationale is missing")
    normalized = value.strip()
    if (not 16 <= len(normalized) <= 500 or
            any(ord(character) < 32 for character in normalized)):
        raise RuntimeError(
            "minimum development pop-gain rationale must be 16..500 printable characters"
        )
    return normalized


def row_role(row):
    role = row.get("row_role")
    if role != "target":
        raise RuntimeError("ordinal policy input is not a target row")
    return role


def row_is_target(row):
    return row_role(row) == "target"


def row_is_proven(row):
    row_role(row)
    evidence = row.get("_ordinal_safety_evidence")
    if (not isinstance(evidence, dict) or
            evidence.get("status") not in {"proven", "unproven"}):
        raise RuntimeError("ordinal row lacks canonical frame-safety evidence")
    proven = evidence["status"] == "proven"
    if proven != (row.get("_ordinal_intersection") is not None):
        raise RuntimeError("ordinal row safety status and target disagree")
    return proven


def _failure_families(intersection):
    families = set()
    for geometry in intersection["first_unsafe_failures"]:
        for cause in geometry["failure_causes"]:
            families.add(cause.split(":", 1)[0])
    return tuple(sorted(families))


def frontier_stratum(row):
    intersection = row["_ordinal_intersection"]
    ordinal_merge.validate_geometry_intersection(intersection)
    families = _failure_families(intersection)
    if intersection["left_censored"]:
        if not families:
            raise RuntimeError("identity failure has no failure stratum")
        return "left_censored:identity:" + "+".join(families)
    elif intersection["right_censored"]:
        return "right_censored:safe-1.50"
    if not families:
        raise RuntimeError("measured ordinal failure has no failure stratum")
    highest = intersection["highest_proven_safe_scale"]
    if highest is None:
        raise RuntimeError("finite ordinal failure has no highest-safe bin")
    return f"finite:safe-{highest:.2f}:" + "+".join(families)


def intersection_loss_evidence(intersection):
    """Adapt a canonical two-geometry intersection to the loss contract."""
    ordinal_merge.validate_geometry_intersection(intersection)
    failures = sorted({
        cause
        for geometry in intersection["first_unsafe_failures"]
        for cause in geometry["failure_causes"]
    })
    tested = []
    previous_pop = 0.0
    gains = intersection["conservative_safe_pop_gain_over_identity_pct"]
    for index, state in enumerate(intersection["states"]):
        if state == "unknown":
            break
        if state == "safe":
            pop = float(gains[index])
            previous_pop = pop
            causes = []
        else:
            pop = previous_pop
            causes = failures
        tested.append({
            "scale": SCALES[index],
            "safe": state == "safe",
            "realized_pop_pct": pop,
            "failure_causes": causes,
        })
    return build_frontier_evidence(tested)


def balanced_ordinal_sample_weights(rows):
    """Balance regimes, frontier/failure strata, variants, domains and clips.

    The three simulated Windows-HDR white anchors share the HDR half of the
    mass rather than giving HDR three times the native-SDR influence.
    """
    if not rows:
        raise RuntimeError("cannot balance an empty ordinal dataset")
    regimes = set()
    strata = {}
    variants = {}
    domains = {}
    clips = {}
    frames = {}
    keys = []
    for row in rows:
        regime = runtime_regime(row["_input_variant"])
        stratum = frontier_stratum(row)
        variant = row["_input_variant_sha256"]
        domain = row.get("domain") or "unknown"
        film = row["film_id"]
        clip = row["clip"]
        regimes.add(regime)
        strata.setdefault(regime, set()).add(stratum)
        variants.setdefault((regime, stratum), set()).add(variant)
        domains.setdefault((regime, stratum, variant), set()).add(domain)
        clips.setdefault((regime, stratum, variant, domain), set()).add(
            (film, clip)
        )
        key = (regime, stratum, variant, domain, film, clip)
        frames[key] = frames.get(key, 0) + 1
        keys.append(key)
    weights = []
    for row, key in zip(rows, keys):
        regime, stratum, variant, domain, _film, _clip = key
        denominator = (
            len(regimes) * len(strata[regime]) *
            len(variants[(regime, stratum)]) *
            len(domains[(regime, stratum, variant)]) *
            len(clips[(regime, stratum, variant, domain)]) * frames[key]
        )
        base = float(row.get("global_policy_weight", 1.0))
        if not math.isfinite(base) or base <= 0.0:
            raise RuntimeError("ordinal sample has invalid policy weight")
        weights.append(base / denominator)
    # Production weights may differ across catalogs. Preserve them within a
    # frontier cell, but never let them undo the explicit equal regime/stratum
    # contract.
    cell_totals = {}
    for key, weight in zip(keys, weights):
        cell = key[:2]
        cell_totals[cell] = cell_totals.get(cell, 0.0) + weight
    normalized = []
    for key, weight in zip(keys, weights):
        regime, stratum = key[:2]
        target = 1.0 / (len(regimes) * len(strata[regime]))
        normalized.append(weight * target / cell_totals[(regime, stratum)])
    weights = normalized
    return weights


def validate_ordinal_coverage(rows, split):
    """Require safety and useful-action evidence in both runtime regimes."""
    coverage = {}
    for row in rows:
        regime = runtime_regime(row["_input_variant"])
        intersection = row["_ordinal_intersection"]
        ordinal_merge.validate_geometry_intersection(intersection)
        bucket = coverage.setdefault(regime, {
            "measured_failure": False,
            "positive_safe_gain": False,
        })
        bucket["measured_failure"] |= not intersection["right_censored"]
        gain = intersection["maximum_conservative_safe_pop_gain_pct"]
        bucket["positive_safe_gain"] |= bool(
            intersection["identity_feasible"] and gain is not None and gain > 0.0
        )
    if set(coverage) != {"sdr", "hdr"}:
        raise RuntimeError(
            f"{split} ordinal coverage requires native SDR and HDR regimes"
        )
    for regime, values in sorted(coverage.items()):
        missing = [name for name, present in values.items() if not present]
        if missing:
            raise RuntimeError(
                f"{split} {regime} ordinal coverage lacks: " + ", ".join(missing)
            )


def load_ordinal_active_split(path):
    """Authenticate train/dev split metadata without opening sealed-test data.

    The orchestration catalog authenticates this manifest as a whole before
    this loader runs.  Training and development dataset manifests are checked
    directly because they are admitted inputs.  Sealed-test paths and hashes
    remain opaque frozen identifiers: opening them during model selection
    would violate the holdout contract.
    """
    path = Path(path).resolve(strict=True)
    payload = _load_json_object(path, "active ordinal split")
    if payload.get("schema") != 1:
        raise RuntimeError(f"unsupported active ordinal split: {path}")

    def require_sha(value, label):
        if not _is_sha256(value):
            raise RuntimeError(f"active ordinal split has invalid {label}")
        return value

    def resolve_reference(value, label, *, require_file=True):
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"active ordinal split has no {label}")
        referenced = Path(value)
        if not referenced.is_absolute():
            referenced = path.parent / referenced
        referenced = referenced.resolve(strict=False)
        if require_file and not referenced.is_file():
            raise RuntimeError(
                f"active ordinal split {label} is missing: {referenced}"
            )
        return referenced

    catalog = resolve_reference(payload.get("catalog"), "source catalog")
    if sha256(catalog) != require_sha(
            payload.get("catalog_sha256"), "catalog_sha256"):
        raise RuntimeError("active ordinal split source catalog hash is stale")

    split_productions = payload.get("split_productions")
    if not isinstance(split_productions, dict):
        raise RuntimeError("active ordinal split lacks split_productions")
    assigned = {}
    for split in ("training", "development", "test"):
        productions = split_productions.get(split)
        if (not isinstance(productions, list) or not productions or
                any(not isinstance(item, str) or not item
                    for item in productions) or
                len(productions) != len(set(productions))):
            raise RuntimeError(
                f"active ordinal split has invalid {split} productions"
            )
        for production in productions:
            if production in assigned:
                raise RuntimeError(
                    "active ordinal split assigns one production twice"
                )
            assigned[production] = split

    productions = payload.get("productions")
    if not isinstance(productions, list) or not productions:
        raise RuntimeError("active ordinal split lacks production evidence")
    observed = {}
    expected_sequences = {}
    for index, item in enumerate(productions):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"active ordinal split production {index} is invalid"
            )
        production = item.get("production_id")
        split = item.get("split")
        source_kind = item.get("source_kind")
        context_frames = item.get("context_frames")
        label_frames = item.get("label_frames")
        if (not isinstance(production, str) or not production or
                assigned.get(production) != split or
                production in observed or
                source_kind not in {"mono-video", "native-hdr-video"} or
                not isinstance(context_frames, int) or
                isinstance(context_frames, bool) or context_frames < 1 or
                not isinstance(label_frames, int) or
                isinstance(label_frames, bool) or label_frames < 1 or
                label_frames > context_frames):
            raise RuntimeError(
                f"active ordinal split production {index} is invalid"
            )
        observed[production] = split
        manifest_value = item.get("dataset_manifest")
        manifest_hash = require_sha(
            item.get("dataset_manifest_sha256"),
            f"dataset_manifest_sha256 for {production}",
        )
        # Preserve but never dereference a sealed-test manifest.
        manifest = resolve_reference(
            manifest_value, f"dataset manifest for {production}",
            require_file=split != "test",
        )
        if split == "test":
            continue
        if sha256(manifest) != manifest_hash:
            raise RuntimeError(
                f"active ordinal split dataset manifest is stale for {production}"
            )
        dataset = _load_json_object(
            manifest, f"dataset manifest for {production}"
        )
        dataset_production = (
            dataset.get("film_id") if dataset.get("schema") == 1
            else dataset.get("production_id")
        )
        if (dataset.get("schema") not in {1, 2} or
                dataset_production != production or
                dataset.get("split") != split or
                (dataset.get("schema") == 2 and
                 dataset.get("source_kind") != source_kind)):
            raise RuntimeError(
                f"active ordinal split dataset identity disagrees for {production}"
            )
        sequences = dataset.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise RuntimeError(
                f"active ordinal split dataset has no sequences for {production}"
            )
        production_sequences = {}
        for sequence in sequences:
            if not isinstance(sequence, dict):
                raise RuntimeError(
                    f"active ordinal split sequence is invalid for {production}"
                )
            clip = sequence.get("clip")
            frame_count = sequence.get(
                "source_frames", sequence.get("context_frames")
            )
            source_first_frame = sequence.get("source_start_frame", 0)
            target_count = sequence.get("label_frames")
            if (not isinstance(clip, str) or not clip or
                    clip in production_sequences or
                    not isinstance(frame_count, int) or
                    isinstance(frame_count, bool) or frame_count < 1 or
                    not isinstance(source_first_frame, int) or
                    isinstance(source_first_frame, bool) or
                    source_first_frame < 0 or
                    not isinstance(target_count, int) or
                    isinstance(target_count, bool) or target_count < 1 or
                    target_count > frame_count):
                raise RuntimeError(
                    f"active ordinal split sequence is invalid for {production}"
                )
            target_ids = sequence.get("label_frame_ids")
            if target_ids is not None:
                if (not isinstance(target_ids, list) or
                        len(target_ids) != target_count or
                        any(not isinstance(value, int) or
                            isinstance(value, bool) or
                            not 0 <= value < frame_count
                            for value in target_ids) or
                        len(target_ids) != len(set(target_ids))):
                    raise RuntimeError(
                        "active ordinal split target frame ids are invalid for "
                        f"{production}/{clip}"
                    )
                target_ids = sorted(target_ids)
            production_sequences[clip] = {
                # Prepared clip filenames and all published row/frame IDs are
                # shot-local and zero based. source_start_frame is provenance
                # in the original movie/archive (Spring begins at source 1),
                # not the prepared clip coordinate system.
                "first_frame": 0,
                "source_first_frame": source_first_frame,
                "frame_count": frame_count,
                "target_count": target_count,
                "target_frame_ids": target_ids,
            }
        if (sum(item["frame_count"]
                for item in production_sequences.values()) != context_frames or
                sum(item["target_count"]
                    for item in production_sequences.values()) != label_frames):
            raise RuntimeError(
                f"active ordinal split sequence totals disagree for {production}"
            )
        expected_sequences[production] = production_sequences
    if observed != assigned:
        raise RuntimeError(
            "active ordinal split production evidence is incomplete"
        )
    payload["_ordinal_expected_sequences"] = expected_sequences
    return payload, sha256(path)


def validate_complete_active_cardinality(rows, active_split):
    """Validate exact sparse targets; unlabeled cadence rows are forbidden."""
    productions = active_split.get("productions")
    if not isinstance(productions, list):
        raise RuntimeError("active split lacks production cardinality evidence")
    working_splits = {"training", "development"}
    expected = {}
    production_split = {}
    production_variants = {}
    for item in productions:
        if not isinstance(item, dict):
            raise RuntimeError("active split production cardinality is invalid")
        if item.get("split") not in working_splits:
            continue
        production = item.get("production_id")
        split = item.get("split")
        source_kind = item.get("source_kind")
        target_count = item.get("label_frames")
        if (not isinstance(production, str) or not production or
                source_kind not in {"mono-video", "native-hdr-video"} or
                not isinstance(target_count, int) or
                isinstance(target_count, bool) or target_count < 1 or
                production in production_split):
            raise RuntimeError("active split production cardinality is invalid")
        variant_names = (
            APPROVED_MONO_RUNTIME_VARIANTS
            if source_kind == "mono-video" else {"native_pq"}
        )
        production_split[production] = split
        production_variants[production] = set(variant_names)
        for variant in variant_names:
            expected[(production, variant, "target")] = target_count
    if (not expected or set(production_split.values()) != working_splits):
        raise RuntimeError(
            "active split requires non-empty training/development cardinality"
        )
    actual = {key: 0 for key in expected}
    row_identities = set()
    for row in rows:
        production = row.get("film_id")
        split = row.get("split")
        clip = row.get("clip")
        frame = row.get("frame")
        role = row_role(row)
        if (production not in production_split or
                split != production_split.get(production) or
                not isinstance(clip, str) or not clip or
                not isinstance(frame, int) or isinstance(frame, bool) or
                frame < 0):
            raise RuntimeError("ordinal row is outside the active train/dev scope")
        variant = runtime_variant_name(row.get("_input_variant"))
        if variant not in production_variants[production]:
            raise RuntimeError("ordinal row has an unexpected runtime condition")
        row_identity = (production, clip, frame, variant)
        if row_identity in row_identities:
            raise RuntimeError(
                "ordinal train/dev cardinality repeats a frame/condition row: "
                f"{row_identity!r}"
            )
        row_identities.add(row_identity)
        key = (production, variant, role)
        actual[key] = actual.get(key, 0) + 1
        row_is_proven(row)
    if actual != expected or len(row_identities) != len(rows):
        raise RuntimeError(
            "ordinal train/dev cardinality is incomplete: "
            f"actual={actual}, expected={expected}"
        )
    expected_sequences = active_split.get("_ordinal_expected_sequences")
    if not isinstance(expected_sequences, dict):
        raise RuntimeError("active split lacks authenticated target selection")
    for production, variants in production_variants.items():
        sequences = expected_sequences.get(production)
        if not isinstance(sequences, dict):
            raise RuntimeError("active split target selection is incomplete")
        expected_ids = {
            (clip, frame)
            for clip, sequence in sequences.items()
            for frame in (sequence["target_frame_ids"] or ())
        }
        if not expected_ids:
            # Older manifests may authenticate only target cardinality. The
            # frontier/source one-to-one join still freezes exact frame IDs.
            continue
        for variant in variants:
            actual_ids = {
                (row["clip"], int(row["frame"])) for row in rows
                if row["film_id"] == production and
                runtime_variant_name(row["_input_variant"]) == variant
            }
            if actual_ids != expected_ids:
                raise RuntimeError(
                    "ordinal target sequence coverage is incomplete: " +
                    repr((production, variant))
                )
    return build_expected_runtime_evidence(rows)


def build_expected_runtime_evidence(rows):
    """Freeze exact validated target evidence by split and condition."""
    result = {}
    for split in ("training", "development"):
        split_rows = [row for row in rows if row.get("split") == split]
        variants = sorted({
            runtime_variant_name(row["_input_variant"]) for row in split_rows
        })
        conditions = {}
        for row in split_rows:
            name = runtime_variant_name(row["_input_variant"])
            identity = {
                "name": name,
                "input_variant": row["_input_variant"],
                "input_variant_sha256": row["_input_variant_sha256"],
            }
            previous = conditions.setdefault(name, identity)
            if previous != identity:
                raise RuntimeError(
                    "ordinal split maps one condition name to multiple variants"
                )
        split_value = {
            "expected_variants": variants,
            "expected_conditions": [
                conditions[name] for name in sorted(conditions)
            ],
            "regimes": {},
            "variants": {},
        }

        def counts(members):
            return {
                "target_rows": len(members),
                "proven_targets": sum(row_is_proven(row) for row in members),
                "unproven_targets": sum(
                    not row_is_proven(row) for row in members
                ),
            }

        for regime in ("sdr", "hdr"):
            split_value["regimes"][regime] = counts([
                row for row in split_rows
                if runtime_regime(row["_input_variant"]) == regime
            ])
        for variant in variants:
            split_value["variants"][variant] = counts([
                row for row in split_rows
                if runtime_variant_name(row["_input_variant"]) == variant
            ])
        result[split] = split_value
    return result


def attach_active_source_groups(rows, active_split):
    """Attach report-only independence groups from the authenticated split."""
    productions = active_split.get("productions")
    if not isinstance(productions, list) or not productions:
        raise RuntimeError("active split lacks source-group provenance")
    mapping = {}
    for record in productions:
        if not isinstance(record, dict):
            raise RuntimeError("active split has invalid source-group provenance")
        production = record.get("production_id")
        group = record.get("source_group")
        if (not isinstance(production, str) or not production or
                not isinstance(group, str) or not group or
                production in mapping):
            raise RuntimeError("active split source-group provenance is ambiguous")
        mapping[production] = group
    for row in rows:
        group = mapping.get(row["film_id"])
        if group is None:
            raise RuntimeError(
                f"ordinal row production is absent from active split: "
                f"{row['film_id']}"
            )
        row["_independent_source_group"] = group


@dataclass(frozen=True)
class SharedAffineCalibration:
    """One positive affine logit transform shared by all 26 thresholds."""

    slope: float = 1.0
    bias: float = 0.0

    def __post_init__(self):
        if (not math.isfinite(self.slope) or self.slope <= 0.0 or
                not math.isfinite(self.bias)):
            raise RuntimeError("ordinal calibration must be finite and positive")

    def apply(self, probabilities):
        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
        epsilon = torch.finfo(probabilities.dtype).eps
        logits = torch.logit(probabilities.clamp(epsilon, 1.0 - epsilon))
        return torch.sigmoid(logits * self.slope + self.bias)

    def as_dict(self):
        return {
            "contract": "shared-positive-affine-logit-v1",
            "slope": self.slope,
            "bias": self.bias,
            "fit_split": "training",
        }


def fit_shared_affine_calibration(probabilities, evidence, *,
                                  sample_weights=None,
                                  steps=CALIBRATION_STEPS,
                                  learning_rate=CALIBRATION_LEARNING_RATE):
    """Fit post-hoc calibration on training evidence without touching DA/head."""
    probabilities = torch.as_tensor(
        probabilities, dtype=torch.float64
    ).detach().cpu()
    if probabilities.ndim != 2 or probabilities.shape[0] != len(evidence):
        raise RuntimeError("ordinal calibration inputs have inconsistent shapes")
    if steps < 0 or learning_rate <= 0.0:
        raise RuntimeError("ordinal calibration optimizer settings are invalid")
    raw_slope = torch.tensor(
        math.log(math.expm1(1.0)), dtype=torch.float64, requires_grad=True
    )
    bias = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam((raw_slope, bias), lr=learning_rate)
    if sample_weights is None:
        weights = torch.ones(probabilities.shape[0], dtype=torch.float64)
    else:
        weights = torch.as_tensor(sample_weights, dtype=torch.float64)
        if (weights.shape != (probabilities.shape[0],) or
                not torch.isfinite(weights).all() or
                bool((weights <= 0.0).any())):
            raise RuntimeError("ordinal calibration weights are invalid")
    weights = weights / weights.sum()
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        slope = F.softplus(raw_slope) + 1e-4
        epsilon = torch.finfo(probabilities.dtype).eps
        logits = torch.logit(probabilities.clamp(epsilon, 1.0 - epsilon))
        calibrated = torch.sigmoid(logits * slope + bias)
        per_sample = ordinal_frontier_loss(
            calibrated, evidence, reduction="none"
        )["loss"]
        loss = (weights * per_sample).sum()
        loss.backward()
        optimizer.step()
    return SharedAffineCalibration(
        float((F.softplus(raw_slope.detach()) + 1e-4).item()),
        float(bias.detach().item()),
    )


def select_scale(probabilities, confidence=SELECTION_CONFIDENCE):
    """Validate monotonic model output, then use the canonical selector."""
    values = np.asarray(probabilities, dtype=np.float64)
    if (values.shape != (len(SCALES),) or not np.isfinite(values).all() or
            np.any(values < 0.0) or np.any(values > 1.0) or
            np.any(values[1:] > values[:-1] + 1e-12)):
        raise RuntimeError("ordinal selection probabilities are invalid")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise RuntimeError("ordinal selection confidence is invalid")
    return select_contiguous_safe_scale(values.tolist(), confidence)


def _mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def known_bin_ece(probabilities, intersection, bin_count=10):
    """Return per-frame ECE over only measured ordinal safety bins."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(SCALES),) or not np.isfinite(values).all():
        raise RuntimeError("known-bin ECE received invalid probabilities")
    states = intersection["states"]
    known = [(float(probability), 1.0 if state == "safe" else 0.0)
             for probability, state in zip(values, states)
             if state in {"safe", "unsafe"}]
    if not known:
        raise RuntimeError("known-bin ECE has no measured bins")
    total = len(known)
    result = 0.0
    for index in range(bin_count):
        low = index / bin_count
        high = (index + 1) / bin_count
        members = [(probability, target) for probability, target in known
                   if probability >= low and
                   (probability < high or index == bin_count - 1)]
        if not members:
            continue
        confidence = np.mean([item[0] for item in members])
        accuracy = np.mean([item[1] for item in members])
        result += len(members) / total * abs(float(confidence - accuracy))
    return float(result)


def _macro_rows(records, group_fields):
    """Average frames into scenes, scenes into films, then films equally."""
    hierarchy = {}
    for record in records:
        row = record["row"]
        scene = row.get("_runtime_scene_id")
        if scene is None:
            scene = row["clip"]
        key = (row["film_id"], row["clip"], str(scene))
        hierarchy.setdefault(key, []).append(record)
    scenes = []
    for (film, _clip, _scene), members in hierarchy.items():
        scenes.append({
            "film": film,
            **{field: _mean(member[field] for member in members)
               for field in group_fields},
        })
    films = {}
    for scene in scenes:
        films.setdefault(scene["film"], []).append(scene)
    film_metrics = {
        film: {
            field: _mean(scene[field] for scene in film_scenes)
            for field in group_fields
        }
        for film, film_scenes in films.items()
    }
    macro = {
        field: _mean(metrics[field] for metrics in film_metrics.values())
        for field in group_fields
    }
    return macro, film_metrics


def _evaluate_group(records):
    if not records:
        raise RuntimeError("ordinal evaluation group is empty")
    if any(not row_is_target(record["row"]) for record in records):
        raise RuntimeError("ordinal target metrics include controller context")
    fields = (
        "unproven_overshoot", "selected_unproven_overshoot",
        "realized_pop_gain_pct", "selected_realized_pop_gain_pct",
        "interval_nll", "known_bin_brier", "known_bin_ece",
        "scale_underreach", "abstained",
        "identity_hard_failure", "plateau_excess_scale",
        "selected_plateau_excess_scale", "safety_evidence_unproven",
    )
    macro, films = _macro_rows(records, fields)
    macro.update({
        "samples": len(records),
        "proven_samples": int(sum(row_is_proven(item["row"])
                                  for item in records)),
        "unproven_samples": int(sum(not row_is_proven(item["row"])
                                    for item in records)),
        "shots": len({(
            item["row"]["film_id"], item["row"]["clip"],
            str(item["row"].get("_runtime_scene_id")
                if item["row"].get("_runtime_scene_id") is not None
                else item["row"]["clip"]),
        ) for item in records}),
        "films": len(films),
        "unproven_overshoot_count": int(sum(
            item["unproven_overshoot"] > 0.0 for item in records
        )),
        "selected_unproven_overshoot_count": int(sum(
            item["selected_unproven_overshoot"] > 0.0
            for item in records
        )),
        "identity_hard_failure_count": int(sum(
            item["identity_hard_failure"] > 0.0 for item in records
        )),
        "plateau_excess_selection_count": int(sum(
            item["selected_plateau_excess_scale"] > 0.0 for item in records
        )),
        "plateau_excess_applied_count": int(sum(
            item["plateau_excess_scale"] > 0.0 for item in records
        )),
    })
    independent = {}
    for item in records:
        row = item["row"]
        if not row_is_proven(row):
            continue
        group = row.get("_independent_source_group", row["film_id"])
        independent[group] = bool(
            independent.get(group, False) or
            item["selected_unproven_overshoot"] > 0.0
        )
    macro.update({
        "independent_source_groups": len(independent),
        "independent_source_groups_with_overshoot": sum(
            independent.values()
        ),
    })
    return {"macro": macro, "films": films}


def _scale_result(scale, intersection):
    index = scale_index(scale)
    highest = intersection["highest_proven_safe_scale"]
    highest_index = None if highest is None else scale_index(highest)
    overshoot = float(highest_index is None or index > highest_index)
    gain = 0.0
    gains = intersection["conservative_safe_pop_gain_over_identity_pct"]
    if not overshoot and gains[index] is not None:
        gain = float(gains[index])
    return overshoot, gain, highest_index


def _plateau_excess_scale(scale, intersection):
    """Measure avoidable multiplier above the first equal-pop safe action."""
    if scale is None or intersection["left_censored"]:
        return 0.0
    index = scale_index(scale)
    first = intersection["first_maximum_conservative_gain_scale"]
    maximum = intersection["maximum_conservative_safe_pop_gain_pct"]
    gains = intersection["conservative_safe_pop_gain_over_identity_pct"]
    if first is None or maximum is None or gains[index] is None:
        return 0.0
    if float(gains[index]) + PLATEAU_POP_TOLERANCE_PCT < float(maximum):
        return 0.0
    return max(0.0, float(scale) - float(first))


def _one_sided_calibration_evidence(group):
    failures = group["macro"][
        "independent_source_groups_with_overshoot"
    ]
    independent_groups = group["macro"]["independent_source_groups"]
    required = math.ceil(
        math.log(CALIBRATION_ALPHA) /
        math.log(1.0 - CALIBRATION_MAX_FAILURE_RATE)
    )
    upper = (
        1.0 - CALIBRATION_ALPHA ** (1.0 / independent_groups)
        if independent_groups and failures == 0 else 1.0
    )
    return {
        "contract": "zero-failure-one-sided-binomial-bound-v1",
        "confidence": 1.0 - CALIBRATION_ALPHA,
        "maximum_failure_rate": CALIBRATION_MAX_FAILURE_RATE,
        "independent_development_groups": independent_groups,
        "observed_groups_with_overshoot": failures,
        "zero-failure_upper_bound": upper,
        "minimum_independent_groups_required": required,
        "deployable": failures == 0 and independent_groups >= required,
    }


def build_prediction_examples(rows, calibrated, records, limit=3):
    """Publish deterministic low/middle/high frontier examples for review."""
    candidates = []
    for index, (row, record) in enumerate(zip(rows, records)):
        if not row_is_proven(row):
            continue
        highest = row["_ordinal_intersection"]["highest_proven_safe_scale"]
        candidates.append((
            -1.0 if highest is None else float(highest),
            str(row["film_id"]), str(row["clip"]), int(row["frame"]),
            index, row, record,
        ))
    candidates.sort(key=lambda item: item[:5])
    if not candidates:
        return []
    count = min(limit, len(candidates))
    positions = sorted({
        round(position * (len(candidates) - 1) / max(1, count - 1))
        for position in range(count)
    })
    examples = []
    for position in positions:
        _highest_key, _film, _clip, _frame, index, row, record = (
            candidates[position]
        )
        intersection = row["_ordinal_intersection"]
        examples.append({
            "source": row.get("source", ""),
            "film_id": row["film_id"],
            "clip": row["clip"],
            "frame": int(row["frame"]),
            "input_variant": runtime_variant_name(row["_input_variant"]),
            "highest_proven_safe_scale": intersection[
                "highest_proven_safe_scale"
            ],
            "selected_scale": record["selected_scale"],
            "selection_confidence": SELECTION_CONFIDENCE,
            "channels": [
                {"scale": scale, "safe_probability": float(probability)}
                for scale, probability in zip(
                    SCALES, calibrated[index].tolist()
                )
            ],
            "target_states": list(intersection["states"]),
        })
    return examples


def evaluate_predictions(
        probabilities, rows, calibration=None,
        confidence=SELECTION_CONFIDENCE,
        minimum_development_pop_gain_pct=None,
        minimum_development_pop_gain_rationale=None):
    """Return safety-first, film-balanced development acceptance metrics."""
    tensor = torch.as_tensor(probabilities, dtype=torch.float64)
    if tensor.ndim != 2 or tensor.shape[0] != len(rows):
        raise RuntimeError("ordinal predictions do not match development rows")
    calibration = calibration or SharedAffineCalibration()
    calibrated = calibration.apply(tensor)
    proven_indices = [
        index for index, row in enumerate(rows) if row_is_proven(row)
    ]
    if not proven_indices:
        raise RuntimeError("ordinal development evidence has no proven frames")
    evidence = [
        intersection_loss_evidence(rows[index]["_ordinal_intersection"])
        for index in proven_indices
    ]
    loss_rows = ordinal_frontier_loss(
        calibrated[proven_indices], evidence, reduction="none"
    )
    loss_index = {
        row_index: evidence_index
        for evidence_index, row_index in enumerate(proven_indices)
    }
    records = []
    for index, row in enumerate(rows):
        intersection = row["_ordinal_intersection"]
        target = row_is_target(row)
        proven = row_is_proven(row)
        selected_scale = (
            select_scale(calibrated[index].tolist(), confidence)
            if not target or proven else None
        )
        selected_index = (
            None if selected_scale is None else scale_index(selected_scale)
        )
        if not target or not proven:
            overshoot, gain, highest_index = 0.0, 0.0, None
        elif selected_index is None:
            overshoot, gain, highest_index = (0.0, 0.0, (
                None if intersection["highest_proven_safe_scale"] is None
                else scale_index(intersection["highest_proven_safe_scale"])
            ))
        else:
            overshoot, gain, highest_index = _scale_result(
                selected_scale, intersection
            )
        underreach = float(
            max(0, (highest_index or 0) - (selected_index or 0)) * 0.02
        ) if highest_index is not None else 0.0
        evidence_index = loss_index.get(index)
        identity_failure = float(
            proven and intersection["left_censored"]
        )
        records.append({
            "row": row,
            "selected_unproven_overshoot": overshoot,
            "selected_realized_pop_gain_pct": gain,
            # Temporal replay is intentionally out of scope for this phase.
            # The model-selected action is evaluated directly on this target.
            "unproven_overshoot": overshoot,
            "realized_pop_gain_pct": gain,
            "interval_nll": (
                None if evidence_index is None else
                float(loss_rows["interval_nll"][evidence_index])
            ),
            "known_bin_brier": (
                None if evidence_index is None else
                float(loss_rows["known_bin_brier"][evidence_index])
            ),
            "known_bin_ece": (
                None if not proven else known_bin_ece(
                    calibrated[index].tolist(), intersection
                )
            ),
            "scale_underreach": underreach,
            "abstained": float(selected_scale is None),
            "selected_scale": selected_scale,
            "identity_hard_failure": identity_failure,
            "selected_plateau_excess_scale": (
                _plateau_excess_scale(selected_scale, intersection)
                if proven else 0.0
            ),
            "plateau_excess_scale": (
                _plateau_excess_scale(selected_scale, intersection)
                if proven else 0.0
            ),
            "safety_evidence_unproven": float(not proven),
            "applied_scale": (
                selected_scale if selected_scale is not None else 1.0
            ),
        })
    target_records = [
        record for record in records if row_is_target(record["row"])
    ]
    regimes = {
        regime: _evaluate_group([
            record for record in target_records
            if runtime_regime(record["row"]["_input_variant"]) == regime
        ])
        for regime in ("sdr", "hdr")
    }
    variants = {}
    for name in sorted({
            runtime_variant_name(record["row"]["_input_variant"])
            for record in target_records
    }):
        variants[name] = _evaluate_group([
            record for record in target_records
            if runtime_variant_name(record["row"]["_input_variant"]) == name
        ])
    overall = _evaluate_group(target_records)
    # Overall is a presentation convenience only; make it exactly regime
    # balanced so three HDR anchors never outweigh native SDR.
    balanced_fields = (
        "unproven_overshoot", "selected_unproven_overshoot",
        "realized_pop_gain_pct", "selected_realized_pop_gain_pct",
        "interval_nll", "known_bin_brier", "scale_underreach", "abstained",
        "known_bin_ece", "identity_hard_failure",
        "plateau_excess_scale", "selected_plateau_excess_scale",
        "safety_evidence_unproven",
    )
    for field in balanced_fields:
        overall["macro"][field] = _mean(
            regimes[regime]["macro"][field] for regime in ("sdr", "hdr")
        )
    bound_groups = {
        "sdr": regimes["sdr"],
        "hdr": regimes["hdr"],
        **{
            "variant:" + name: group for name, group in variants.items()
            if name != "native_sdr"
        },
    }
    calibration_evidence = {
        name: _one_sided_calibration_evidence(group)
        for name, group in bound_groups.items()
    }
    result = {
        "selection_confidence": confidence,
        "calibration": calibration.as_dict(),
        "overall": overall,
        "regimes": regimes,
        "input_variants": variants,
        "application": {
            "mode": "direct-target-only",
            "target_rows": len(target_records),
            "context_rows": 0,
            "validation_scope": "independent authenticated safety targets",
        },
        "prediction_examples": build_prediction_examples(
            rows, calibrated, records
        ),
        "promotion_prerequisites": {
            "independent_target_render_gate_passed": False,
            "sealed_test_target_gate_passed": False,
            "promotion_blocked_until_both_true": True,
        },
        "calibration_evidence": calibration_evidence,
        "calibration_deployable": all(
            value["deployable"] for value in calibration_evidence.values()
        ),
    }
    result["development_candidate_status"] = development_candidate_status(
        result, minimum_development_pop_gain_pct,
        minimum_development_pop_gain_rationale,
    )
    return result


def development_candidate_status(evaluation, minimum_pop_gain_pct,
                                 minimum_pop_gain_rationale=None):
    """Gate a training checkpoint without claiming production acceptance."""
    group_metrics = {
        "sdr": evaluation["regimes"]["sdr"]["macro"],
        "hdr": evaluation["regimes"]["hdr"]["macro"],
        **{
            "variant:" + name: group["macro"]
            for name, group in evaluation["input_variants"].items()
            if name != "native_sdr"
        },
    }
    gains = {
        name: metrics["realized_pop_gain_pct"]
        for name, metrics in group_metrics.items()
    }
    unproven_samples = {
        name: int(metrics["unproven_samples"])
        for name, metrics in group_metrics.items()
    }
    complete_evidence_pass = all(
        count == 0 for count in unproven_samples.values()
    )
    identity_failures = {
        name: int(metrics["identity_hard_failure_count"])
        for name, metrics in group_metrics.items()
    }
    if minimum_pop_gain_pct is None:
        if minimum_pop_gain_rationale is not None:
            raise RuntimeError(
                "pop-gain rationale was supplied without a minimum"
            )
        threshold = None
        rationale = None
        material_gain_pass = False
    else:
        if (not isinstance(minimum_pop_gain_pct, (int, float)) or
                isinstance(minimum_pop_gain_pct, bool) or
                not math.isfinite(float(minimum_pop_gain_pct)) or
                float(minimum_pop_gain_pct) <= 0.0):
            raise RuntimeError(
                "minimum development pop gain must be finite and positive"
            )
        threshold = float(minimum_pop_gain_pct)
        rationale = validate_gain_rationale(minimum_pop_gain_rationale)
        material_gain_pass = all(
            value is not None and value >= threshold for value in gains.values()
        )
    key = checkpoint_selection_key(evaluation)
    zero_identity_failure_pass = all(
        count == 0 for count in identity_failures.values()
    )
    zero_overshoot_pass = key[2] == 0 and key[3] == 0
    return {
        "contract": "ordinal-development-candidate-status-v2",
        "minimum_realized_pop_gain_pct": threshold,
        "minimum_realized_pop_gain_rationale": rationale,
        "group_realized_pop_gain_pct": gains,
        "worst_group_realized_pop_gain_pct": min(gains.values()),
        "group_unproven_samples": unproven_samples,
        "group_identity_hard_failures": identity_failures,
        "complete_evidence_pass": complete_evidence_pass,
        "zero_identity_failure_pass": zero_identity_failure_pass,
        "zero_overshoot_pass": zero_overshoot_pass,
        "material_gain_pass": material_gain_pass,
        "training_checkpoint_eligible": (
            complete_evidence_pass and zero_identity_failure_pass and
            zero_overshoot_pass and material_gain_pass
        ),
        # Development can select a checkpoint, never approve a live policy.
        "production_policy_accepted": False,
    }


def checkpoint_selection_key(evaluation):
    """Order epochs: safety proof, worst-regime gain, then fit quality."""
    regimes = evaluation["regimes"]
    if set(regimes) != {"sdr", "hdr"}:
        raise RuntimeError("ordinal development evaluation lacks a runtime regime")
    hdr_variants = [
        value["macro"]
        for name, value in evaluation["input_variants"].items()
        if name != "native_sdr"
    ]
    metrics = [regimes[name]["macro"] for name in ("sdr", "hdr")]
    metrics.extend(hdr_variants)
    overshoots = [
        count
        for item in metrics
        for count in (
            item["selected_unproven_overshoot_count"],
            item["unproven_overshoot_count"],
        )
    ]
    identity_failures = [
        item["identity_hard_failure_count"] for item in metrics
    ]
    selected_plateau_excess = [
        item["selected_plateau_excess_scale"] for item in metrics
    ]
    applied_plateau_excess = [
        item["plateau_excess_scale"] for item in metrics
    ]
    return (
        max(identity_failures),
        sum(identity_failures),
        max(overshoots),
        sum(overshoots),
        -min(item["realized_pop_gain_pct"] for item in metrics),
        -evaluation["overall"]["macro"]["realized_pop_gain_pct"],
        max(selected_plateau_excess),
        max(applied_plateau_excess),
        evaluation["overall"]["macro"]["interval_nll"],
        evaluation["overall"]["macro"]["known_bin_brier"],
        evaluation["overall"]["macro"]["known_bin_ece"],
        evaluation["overall"]["macro"]["scale_underreach"],
    )


def load_source_bundles(paths):
    """Load only authenticated target-only ordinal source bundles."""
    rows = []
    identities = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        bundle_rows = source_rows.validate_full_frame_source_bundle(
            path, verify_media=True
        )
        summary_path = path.parent / "summary.json"
        contract_path = path.parent / "source_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        identities.append({
            "labels": {"path": str(path), "sha256": sha256(path)},
            "summary": {
                "path": str(summary_path), "sha256": sha256(summary_path),
            },
            "contract": {
                "path": str(contract_path), "sha256": sha256(contract_path),
            },
            "payload": contract,
        })
        rows.extend(bundle_rows)
    if not rows:
        raise RuntimeError("source-row bundles are empty")
    return rows, identities


def _source_join_key(row):
    return (
        row["source_sha256"], row["clip"], int(row["frame"]),
        row["input_variant_sha256"],
    )


def _source_model_identity(row):
    return {
        key: row.get(key) for key in (
            "source", "source_sha256", "source_width", "source_height",
            "model_input_width", "model_input_height", "model_source",
            "model_source_sha256", "input_variant", "input_variant_sha256",
            "clip", "frame", "split", "film_id", "domain", "source_kind",
            "global_policy_weight", "source_frame_rate", "row_role",
            "runtime_scene_evidence", "runtime_scene_trace_sha256",
        )
    }


def geometry_structure_sha256(allowlist):
    """Hash deployment geometry while intentionally excluding color mode."""
    geometry_contract.validate_allowlist(allowlist)
    structural_tuples = sorted(
        json.dumps(
            {key: value for key, value in item.items()
             if key != "color_mode"},
            sort_keys=True, separators=(",", ":"),
        )
        for item in allowlist["tuples"]
    )
    return canonical_sha256(structural_tuples)


def source_row_map(rows):
    """Collapse two destination geometries only when model inputs are exact."""
    result = {}
    for row in rows:
        key = _source_join_key(row)
        previous = result.get(key)
        if (previous is not None and
                _source_model_identity(previous) != _source_model_identity(row)):
            raise RuntimeError(
                "duplicate ordinal source rows disagree on model input: " +
                repr(key)
            )
        result[key] = row
    return result


def load_ordinal_bundles(paths):
    """Load only canonical, fully authenticated per-frame frontier bundles."""
    frames = []
    identities = []
    common = None
    seen = set()
    all_geometry_tuples = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        records = ordinal_bundle.validate_frame_label_bundle(path)
        header = records[0]
        identity = {
            "path": str(path),
            "sha256": sha256(path),
            "clip": header["clip"],
            "input_variant": header["input_variant"],
            "input_variant_sha256": header["input_variant_sha256"],
            "metric_specs_sha256": header["metric_specs_sha256"],
            "metric_contract_sha256": header["metric_contract_sha256"],
            "thresholds_sha256": header["thresholds_sha256"],
            "deployment_geometry_allowlist": header[
                "deployment_geometry_allowlist"
            ],
            "deployment_geometry_allowlist_sha256": header[
                "deployment_geometry_allowlist_sha256"
            ],
            "code_identity_sha256": header["code_identity_sha256"],
        }
        tuples = header["deployment_geometry_allowlist"]["tuples"]
        if any(
                item["color_mode"] != header["input_variant"]["color_mode"]
                for item in tuples):
            raise RuntimeError(
                "ordinal bundle geometry color differs from its input variant"
            )
        bundle_common = {
            "metric_specs_sha256": identity["metric_specs_sha256"],
            "metric_contract_sha256": identity["metric_contract_sha256"],
            "thresholds_sha256": identity["thresholds_sha256"],
            "deployment_geometry_structure_sha256":
                geometry_structure_sha256(
                    header["deployment_geometry_allowlist"]
                ),
        }
        if common is None:
            common = bundle_common
        elif common != bundle_common:
            raise RuntimeError(
                "ordinal bundles do not share one metric/geometry contract"
            )
        all_geometry_tuples.extend(tuples)
        identities.append(identity)
        for frame in records[1:-1]:
            provenance = frame["model_input_provenance"]
            key = (
                provenance["source_artifact_sha256"], frame["clip"],
                int(frame["frame_id"]), header["input_variant_sha256"],
            )
            if key in seen:
                raise RuntimeError("duplicate ordinal frame target: " + repr(key))
            seen.add(key)
            frames.append({
                "row_role": "target",
                "_join_key": key,
                "_input_variant": header["input_variant"],
                "_input_variant_sha256": header["input_variant_sha256"],
                "_runtime_scene_id": frame["runtime_scene_id"],
                "_runtime_scene_evidence": frame["runtime_scene_evidence"],
                "_ordinal_safety_evidence": frame["frame_safety_evidence"],
                "_ordinal_diagnostic_absolute_violations": frame[
                    "diagnostic_absolute_violations"
                ],
                "_ordinal_intersection": frame["geometry_intersection"],
                "_model_input_provenance": provenance,
                "_model_input_provenance_sha256": frame[
                    "model_input_provenance_sha256"
                ],
                "_model_depth_artifact_sha256": frame[
                    "model_depth_artifact_sha256"
                ],
                "_ordinal_bundle_sha256": identity["sha256"],
            })
    if not frames:
        raise RuntimeError("ordinal frame-label bundles are empty")
    combined_allowlist = geometry_contract.build_allowlist(all_geometry_tuples)
    common.update({
        "deployment_geometry_allowlist": combined_allowlist,
        "deployment_geometry_allowlist_sha256":
            geometry_contract.allowlist_sha256(combined_allowlist),
    })
    if (common["metric_specs_sha256"] != current_metric_specs_sha256() or
            common["metric_contract_sha256"] !=
            run_eval.metric_contract_sha() or
            common["thresholds_sha256"] != sha256(THRESHOLDS_PATH)):
        raise RuntimeError("ordinal bundles do not match current metric thresholds")
    return frames, identities, common


def join_ordinal_sources(source_rows_value, frontier_frames):
    """Join target-only sources to the exact sparse safety targets."""
    sources = source_row_map(source_rows_value)
    frontier_map = {row["_join_key"]: row for row in frontier_frames}
    if len(frontier_map) != len(frontier_frames):
        raise RuntimeError("ordinal frame targets repeat a source/input identity")
    target_sources = set(sources)
    missing_sources = sorted(set(frontier_map) - target_sources)
    unexpected_targets = sorted(target_sources - set(frontier_map))
    if missing_sources or unexpected_targets:
        raise RuntimeError(
            "ordinal/source target subset differs: "
            f"missing_sources={missing_sources[:4]}, "
            f"unexpected_targets={unexpected_targets[:4]}"
        )
    joined = []
    validated_variants = {}
    for key, source_value in sources.items():
        source = dict(source_value)
        variant = source.get("input_variant")
        variant_sha256 = source.get("input_variant_sha256")
        if not _is_sha256(variant_sha256):
            raise RuntimeError("ordinal/source input variant identity differs")
        previous_variant = validated_variants.get(variant_sha256)
        if previous_variant is None:
            input_color.validate_input_variant(variant)
            if input_color.input_variant_sha256(variant) != variant_sha256:
                raise RuntimeError(
                    "ordinal/source input variant identity differs"
                )
            validated_variants[variant_sha256] = variant
        elif previous_variant != variant:
            raise RuntimeError("ordinal/source input variant identity differs")
        row_role(source)
        scene = source.get("runtime_scene_evidence")
        if not isinstance(scene, dict):
            raise RuntimeError(
                "ordinal/source row lacks authenticated runtime scene evidence"
            )
        source["_runtime_scene_id"] = scene.get("runtime_scene_id")
        source["_runtime_scene_evidence"] = scene
        source["_input_variant"] = variant
        source["_input_variant_sha256"] = variant_sha256
        frontier = frontier_map[key]
        if (source["input_variant"] != frontier["_input_variant"] or
                source["input_variant_sha256"] !=
                frontier["_input_variant_sha256"]):
            raise RuntimeError("ordinal/source input variant differs")
        provenance = frontier["_model_input_provenance"]
        if (provenance["source_artifact_sha256"] != source["source_sha256"] or
                provenance["input_variant_sha256"] !=
                source["input_variant_sha256"] or
                source["ordinal_bundle_sha256"] !=
                frontier["_ordinal_bundle_sha256"] or
                source["ordinal_frame_model_input_provenance"] != provenance or
                source["ordinal_frame_model_input_provenance_sha256"] !=
                frontier["_model_input_provenance_sha256"] or
                source["ordinal_model_depth_artifact_sha256"] !=
                frontier["_model_depth_artifact_sha256"] or
                scene != frontier["_runtime_scene_evidence"]):
            raise RuntimeError("ordinal/source model-input provenance differs")
        expected_model_geometry = {
            "source_width": int(source["source_width"]),
            "source_height": int(source["source_height"]),
            "model_input_width": int(source["model_input_width"]),
            "model_input_height": int(source["model_input_height"]),
            "color_mode": source["input_variant"]["color_mode"],
        }
        if row_is_proven(frontier):
            for geometry_frontier in frontier[
                    "_ordinal_intersection"]["geometry_frontiers"]:
                geometry = geometry_frontier["deployment_geometry"]
                observed = {
                    field: geometry[field] for field in expected_model_geometry
                }
                if observed != expected_model_geometry:
                    raise RuntimeError(
                        "ordinal/source production model geometry differs"
                    )
        for field, value in frontier.items():
            if field.startswith("_"):
                source[field] = value
        joined.append(source)
    return joined


class OrdinalImageDataset(Dataset):
    """Decode the exact production model input for joined ordinal rows."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        variant = row["_input_variant"]
        width = int(row["model_input_width"])
        height = int(row["model_input_height"])
        if variant["kind"] == input_color.INPUT_KIND_NATIVE_PQ:
            model_payload = Path(row["model_source"]).read_bytes()
            if hashlib.sha256(model_payload).hexdigest() != row[
                    "model_source_sha256"]:
                raise RuntimeError(
                    "native-PQ model source changed after admission"
                )
            values = np.frombuffer(model_payload, dtype="<f2")
            expected = row["source_width"] * row["source_height"] * 4
            if values.size != expected:
                raise RuntimeError("native-PQ model source changed after admission")
            values = values.reshape(row["source_height"], row["source_width"], 4)
            image = input_color.preprocess_scrgb_f16_to_nchw(
                values, width, height, variant
            )
        else:
            source_payload = Path(row["source"]).read_bytes()
            if hashlib.sha256(source_payload).hexdigest() != row[
                    "source_sha256"]:
                raise RuntimeError("ordinal source changed after admission")
            bgr = cv2.imdecode(
                np.frombuffer(source_payload, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if bgr is None or (bgr.shape[1], bgr.shape[0]) != (
                    row["source_width"], row["source_height"]):
                raise RuntimeError("ordinal source image changed after admission")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = input_color.preprocess_rgb8_to_nchw(
                rgb, width, height, variant
            )
        return torch.from_numpy(image.copy()), index


class CachedOrdinalDataset(Dataset):
    def __init__(self, features, rows):
        if (not isinstance(features, torch.Tensor) or features.ndim != 2 or
                features.shape[0] != len(rows) or not torch.isfinite(features).all()):
            raise RuntimeError("ordinal cached features are invalid")
        self.features = features
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        intersection = self.rows[index]["_ordinal_intersection"]
        return (
            self.features[index],
            None if intersection is None else
            intersection_loss_evidence(intersection),
        )

    def subset(self, indices):
        indices = list(indices)
        if any(not isinstance(index, int) or not 0 <= index < len(self.rows)
               for index in indices):
            raise RuntimeError("ordinal cached subset indices are invalid")
        return CachedOrdinalDataset(
            self.features[indices], [self.rows[index] for index in indices]
        )


def collate_ordinal_samples(batch):
    features, evidence = zip(*batch)
    return torch.stack(features), list(evidence)


def cache_ordinal_dataset(model, rows, device, *, batch_size=16, workers=0):
    """Cache frozen DA-V2 features in shape-homogeneous mixed-FP16 batches."""
    if batch_size < 1 or workers < 0:
        raise RuntimeError("ordinal cache batching settings are invalid")
    dataset = OrdinalImageDataset(rows)
    groups = {}
    for index, row in enumerate(rows):
        shape = (int(row["model_input_width"]), int(row["model_input_height"]))
        groups.setdefault(shape, []).append(index)
    features = [None] * len(dataset)
    completed = 0
    model.eval()
    with torch.inference_mode():
        for _shape, indices in sorted(groups.items()):
            loader = DataLoader(
                Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=device.type == "cuda",
            )
            for images, row_indices in loader:
                images = images.to(
                    device, non_blocking=device.type == "cuda"
                )
                with torch.amp.autocast(
                        device_type=device.type, dtype=torch.float16,
                        enabled=device.type == "cuda"):
                    batch_features = model.policy_features(images)
                for row_index, feature in zip(
                        row_indices.tolist(), batch_features.float().cpu()):
                    features[row_index] = feature
                completed += len(row_indices)
                if completed % 100 < len(row_indices) or completed == len(dataset):
                    print(
                        f"ordinal cache {completed}/{len(dataset)}", flush=True
                    )
    if any(feature is None for feature in features):
        raise RuntimeError("ordinal feature cache did not cover every row")
    return CachedOrdinalDataset(torch.stack(features), rows)


def audit_paired_variant_identifiability(dataset, split):
    """Report paired feature/target ambiguity without overriding safety labels."""
    if not isinstance(dataset, CachedOrdinalDataset):
        raise RuntimeError("identifiability audit requires cached ordinal features")
    groups = {}
    for index, row in enumerate(dataset.rows):
        if row.get("source_kind") != "mono-video" or not row_is_proven(row):
            continue
        key = (row["film_id"], row["clip"], int(row["frame"]))
        variant = runtime_variant_name(row["_input_variant"])
        members = groups.setdefault(key, {})
        if variant in members:
            raise RuntimeError("paired identifiability audit repeats an input variant")
        members[variant] = (index, row)
    comparisons = 0
    near_identical = 0
    exact_depth = 0
    contradictory_pairs = 0
    contradiction_examples = []
    for key, variants in groups.items():
        for (left_name, (left_index, left)), (
                right_name, (right_index, right)) in combinations(
                    sorted(variants.items()), 2):
            comparisons += 1
            if left["source_sha256"] != right["source_sha256"]:
                raise RuntimeError(
                    "paired mono variants do not share one source frame: " + repr(key)
                )
            left_feature = dataset.features[left_index]
            right_feature = dataset.features[right_index]
            near = torch.allclose(
                left_feature, right_feature,
                rtol=FEATURE_NEAR_EQUAL_RTOL,
                atol=FEATURE_NEAR_EQUAL_ATOL,
            )
            if not near:
                continue
            near_identical += 1
            depth_equal = (
                left["_model_depth_artifact_sha256"] ==
                right["_model_depth_artifact_sha256"]
            )
            exact_depth += int(depth_equal)
            left_states = left["_ordinal_intersection"]["states"]
            right_states = right["_ordinal_intersection"]["states"]
            contradictory = [
                scale for scale, left_state, right_state in zip(
                    SCALES, left_states, right_states
                )
                if {left_state, right_state} == {"safe", "unsafe"}
            ]
            if contradictory:
                contradictory_pairs += 1
                if len(contradiction_examples) < 16:
                    contradiction_examples.append({
                        "film_id": key[0], "clip": key[1], "frame": key[2],
                        "variants": [left_name, right_name],
                        "first_conflicting_scale": contradictory[0],
                        "depth_artifact_equal": depth_equal,
                    })
    return {
        "contract": "paired-image-evidence-identifiability-v1",
        "split": split,
        "paired_source_frames": len(groups),
        "variant_pair_comparisons": comparisons,
        "near_identical_feature_pairs": near_identical,
        "near_identical_pairs_with_exact_depth": exact_depth,
        "contradictory_near_identical_pairs": contradictory_pairs,
        "contradiction_examples": contradiction_examples,
        "feature_rtol": FEATURE_NEAR_EQUAL_RTOL,
        "feature_atol": FEATURE_NEAR_EQUAL_ATOL,
        "target_rule": (
            "each image-derived condition retains its own safety interval even "
            "when frozen evidence is similar; contradictions are diagnostic "
            "calibration risk, never a label-admission override"
        ),
        "variant_specific_safety_targets_retained": True,
        "label_admission_blocked_by_feature_similarity": False,
        "runtime_condition_metadata_model_input": False,
    }


def run_epoch(model, loader, device, optimizer, scaler):
    training = optimizer is not None
    model.train(training)
    model.depth_model.eval()
    totals = {"loss": 0.0, "interval_nll": 0.0, "known_bin_brier": 0.0}
    batches = 0
    for features, evidence in loader:
        if any(item is None for item in evidence):
            raise RuntimeError("unproven ordinal row entered supervised loss")
        features = features.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=device.type == "cuda"):
                probabilities = model.forward_policy_features(features)
            # Adjacent ordinal masses can be much smaller than fp16 epsilon.
            # Keep the cached head forward mixed-precision, but always compute
            # censoring/calibration losses in fp32.
            parts = ordinal_frontier_loss(probabilities.float(), evidence)
            loss = parts["loss"]
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        for key in totals:
            value = loss if key == "loss" else parts[key]
            totals[key] += float(value.detach())
        batches += 1
    if not batches:
        raise RuntimeError("ordinal epoch has no batches")
    return {key: value / batches for key, value in totals.items()}


def collect_probabilities(model, dataset, device, batch_size=256):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_ordinal_samples,
    )
    outputs = []
    model.eval()
    with torch.inference_mode():
        for features, _evidence in loader:
            outputs.append(
                model.forward_policy_features(features.to(device)).float().cpu()
            )
    return torch.cat(outputs)


def canonical_sha256(value):
    payload = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_history_envelope(history, epoch_count, training_contract_hash,
                           checkpoint_path):
    """Build a completed, self-authenticating training receipt."""
    if (not isinstance(history, list) or
            not isinstance(epoch_count, int) or
            isinstance(epoch_count, bool) or epoch_count < 1 or
            not _is_sha256(training_contract_hash) or
            [record.get("epoch") for record in history] !=
            list(range(1, epoch_count + 1)) or
            any(record.get("checkpoint_selection_contract") !=
                CHECKPOINT_SELECTION_CONTRACT for record in history)):
        raise RuntimeError("ordinal history does not contain exact epochs")
    checkpoint_path = Path(checkpoint_path)
    receipt = None
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint_epoch = checkpoint.get("epoch")
        if (checkpoint.get("schema") != ORDINAL_CHECKPOINT_SCHEMA or
                checkpoint.get("training_contract_sha256") !=
                training_contract_hash or
                checkpoint.get("checkpoint_selection_contract") !=
                CHECKPOINT_SELECTION_CONTRACT or
                not isinstance(checkpoint_epoch, int) or
                isinstance(checkpoint_epoch, bool) or
                not 1 <= checkpoint_epoch <= epoch_count):
            raise RuntimeError("published ordinal checkpoint receipt is invalid")
        receipt = {
            "path": checkpoint_path.name,
            "sha256": sha256(checkpoint_path),
            "epoch": checkpoint_epoch,
        }
    return {
        "schema": HISTORY_SCHEMA,
        "contract": HISTORY_CONTRACT,
        "training_contract_sha256": training_contract_hash,
        "completed": True,
        "epoch_count": len(history),
        "epochs": history,
        "best_checkpoint": receipt,
    }


def build_training_contract(args, source_identities, frontier_identities,
                            frontier_common, active_split, active_split_hash,
                            train_rows, development_rows,
                            train_supervised_rows,
                            development_supervised_rows, catalog_identity,
                            identifiability_audit,
                            expected_runtime_evidence):
    target_rows = [
        row for row in (*train_rows, *development_rows)
        if row_is_target(row)
    ]
    depth_models = {
        row["_model_input_provenance"]["depth_model"]
        for row in target_rows
    }
    if len(depth_models) != 1:
        raise RuntimeError("ordinal labels mix depth-model identities")
    code_paths = (
        Path(__file__), Path(ordinal_bundle.__file__),
        Path(ordinal_merge.__file__),
        Path(__file__).with_name("artistic_policy_ordinal_contract.py"),
        Path(__file__).with_name("artistic_policy_ordinal_loss.py"),
        Path(__file__).with_name("artistic_policy_ordinal_model.py"),
        Path(__file__).with_name("artistic_policy_model.py"),
        Path(input_color.__file__),
    )
    code = {
        path.name: {"path": str(path.resolve()), "sha256": sha256(path)}
        for path in code_paths
    }
    return {
        "schema": ORDINAL_CHECKPOINT_SCHEMA,
        "training_contract": TRAINING_CONTRACT,
        "checkpoint_selection_contract": CHECKPOINT_SELECTION_CONTRACT,
        "policy_contract": ORDINAL_POLICY_CONTRACT,
        "output_semantics": ORDINAL_OUTPUT_SEMANTICS,
        "policy_feature_contract": POLICY_FEATURE_CONTRACT,
        "source_bundles": source_identities,
        "source_bundles_sha256": canonical_sha256(source_identities),
        "frontier_bundles": frontier_identities,
        "frontier_bundles_sha256": canonical_sha256(frontier_identities),
        "orchestration_catalog": catalog_identity,
        "metric_specs_sha256": frontier_common["metric_specs_sha256"],
        "metric_contract_sha256": frontier_common["metric_contract_sha256"],
        "thresholds_sha256": frontier_common["thresholds_sha256"],
        "deployment_geometry_allowlist": frontier_common[
            "deployment_geometry_allowlist"
        ],
        "deployment_geometry_allowlist_sha256": frontier_common[
            "deployment_geometry_allowlist_sha256"
        ],
        "deployment_geometry_structure_sha256": frontier_common[
            "deployment_geometry_structure_sha256"
        ],
        "depth_input_color_contract_sha256":
            input_color.color_contract_sha256(),
        "depth_model": next(iter(depth_models)),
        "depth_weights_sha256": sha256(args.depth_weights),
        "active_split": str(args.split_manifest.resolve()),
        "active_split_sha256": active_split_hash,
        "training_productions": active_split["split_productions"]["training"],
        "development_productions": active_split[
            "split_productions"
        ]["development"],
        # Names are retained to prove the held-out set was frozen, but no test
        # image/label path or content hash is admitted to training.
        "sealed_test_productions": active_split["split_productions"]["test"],
        "train_samples": len(train_rows),
        "development_samples": len(development_rows),
        "train_supervised_samples": len(train_supervised_rows),
        "development_supervised_samples": len(development_supervised_rows),
        "expected_runtime_evidence": expected_runtime_evidence,
        "paired_variant_identifiability": identifiability_audit,
        "unproven_frame_policy": (
            "use authenticated sparse targets only; exclude unproven targets "
            "from loss/calibration and force target identity"
        ),
        "preprocessing": (
            "exact authenticated production SDR/HDR input transform, source-"
            "bounded patch-grid geometry, dynamic positional interpolation"
        ),
        "sampling": (
            "equal native-SDR/HDR runtime mass, then exact highest-safe "
            "ceiling/censor bucket x failure-family strata, input variants, "
            "domains, clips, and frames"
        ),
        "objective": (
            "interval-censored likelihood plus known-bin asymmetric Brier; "
            "no unconditional same-shot consistency"
        ),
        "calibration": (
            "one shared positive affine logit transform fit on training only"
        ),
        "checkpoint_selection": (
            "development-only; complete required evidence plus zero unproven "
            "direct-selection overshoot, zero identity hard failures, and the "
            "frozen material-gain floor in SDR, HDR aggregate, and every HDR "
            "variant; then worst-regime and overall realized-pop gain, "
            "natural first-maximum plateau preference, NLL, Brier, ECE, "
            "underreach"
        ),
        "selection_confidence": SELECTION_CONFIDENCE,
        "temporal_evaluation": "out of scope; independent target frames only",
        "plateau_policy": {
            "safety_bins_remain_pure_safety": True,
            "oracle_pop_tolerance_pct": PLATEAU_POP_TOLERANCE_PCT,
            "selection_rule": (
                "prefer checkpoints whose naturally selected safe scale is the "
                "first scale attaining equal maximum realized pop; never use "
                "oracle pop to alter direct scale selection"
            ),
        },
        "minimum_development_realized_pop_gain_pct":
            args.minimum_development_pop_gain_pct,
        "minimum_development_realized_pop_gain_rationale":
            validate_gain_rationale(
                args.minimum_development_pop_gain_rationale
            ),
        "promotion_prerequisites": {
            "independent_target_render_gate_passed": False,
            "sealed_test_target_gate_passed": False,
            "promotion_blocked_until_both_true": True,
        },
        "code": code,
        "seed": args.seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontiers", required=True, type=Path, nargs="+")
    parser.add_argument("--source-rows", required=True, type=Path, nargs="+")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--depth-anything-root", required=True, type=Path)
    parser.add_argument("--depth-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-batch-size", type=int, default=16)
    parser.add_argument("--cache-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--minimum-development-pop-gain-pct", type=float, required=True,
        help=(
            "frozen positive minimum film-balanced realized-pop gain required "
            "in SDR, HDR aggregate, and every HDR input variant"
        ),
    )
    parser.add_argument(
        "--minimum-development-pop-gain-rationale", required=True,
        help="reviewed pre-development rationale frozen into candidate metadata",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.minimum_development_pop_gain_rationale = validate_gain_rationale(
        args.minimum_development_pop_gain_rationale
    )
    if (args.epochs < 1 or args.batch_size < 1 or
            args.cache_batch_size < 1 or args.cache_workers < 0 or
            args.learning_rate <= 0.0 or
            not math.isfinite(args.minimum_development_pop_gain_pct) or
            args.minimum_development_pop_gain_pct <= 0.0):
        raise RuntimeError("ordinal training settings are invalid")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.output.exists() and (
            not args.output.is_dir() or any(args.output.iterdir())):
        raise RuntimeError(
            "ordinal output must be a new or empty directory; stale checkpoints "
            "are never overwritten"
        )

    catalog_identity = validate_training_catalog(
        args.catalog, args.frontiers, args.source_rows, args.split_manifest
    )
    all_source_rows, source_identities = load_source_bundles(args.source_rows)
    frontier_frames, frontier_identities, frontier_common = (
        load_ordinal_bundles(args.frontiers)
    )
    rows = join_ordinal_sources(all_source_rows, frontier_frames)
    active_split, active_split_hash = load_ordinal_active_split(
        args.split_manifest
    )
    expected_runtime_evidence = validate_complete_active_cardinality(
        rows, active_split
    )
    attach_active_source_groups(rows, active_split)
    scalar_training.validate_rows_against_active_split(
        rows, active_split, {"training", "development"}
    )
    train_rows = [row for row in rows if row["split"] == "training"]
    development_rows = [
        row for row in rows if row["split"] == "development"
    ]
    if not train_rows or not development_rows:
        raise RuntimeError(
            "ordinal training requires non-empty training/development splits"
        )
    scalar_training.validate_global_film_split(train_rows, development_rows)
    train_supervised_rows = [row for row in train_rows if row_is_proven(row)]
    development_supervised_rows = [
        row for row in development_rows if row_is_proven(row)
    ]
    if not train_supervised_rows or not development_supervised_rows:
        raise RuntimeError(
            "ordinal training requires proven safety evidence in both splits"
        )
    validate_ordinal_coverage(train_supervised_rows, "training")
    validate_ordinal_coverage(development_supervised_rows, "development")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrdinalArtisticPolicyModel(load_depth_anything_small(
        args.depth_anything_root, args.depth_weights
    ))
    use_dynamic_onnx_position_encoding(model)
    model.freeze_base()
    model.to(device)
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable or any(
            not name.startswith("ordinal_head.") for name in trainable):
        raise RuntimeError(
            "ordinal experiment would train parameters outside ordinal_head"
        )
    optimizer = torch.optim.AdamW(
        model.ordinal_head.parameters(), lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    print("Caching production-shaped frozen DA-V2 ordinal features...", flush=True)
    train_dataset = cache_ordinal_dataset(
        model, train_supervised_rows, device,
        batch_size=args.cache_batch_size, workers=args.cache_workers,
    )
    development_full_dataset = cache_ordinal_dataset(
        model, development_rows, device,
        batch_size=args.cache_batch_size, workers=args.cache_workers,
    )
    development_proven_indices = [
        index for index, row in enumerate(development_rows)
        if row_is_proven(row)
    ]
    development_dataset = development_full_dataset.subset(
        development_proven_indices
    )
    identifiability_audit = {
        "training": audit_paired_variant_identifiability(
            train_dataset, "training"
        ),
        "development": audit_paired_variant_identifiability(
            development_full_dataset, "development"
        ),
    }
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        balanced_ordinal_sample_weights(train_supervised_rows),
        len(train_supervised_rows),
        replacement=True, generator=generator,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate_ordinal_samples,
    )
    development_loader = DataLoader(
        development_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_ordinal_samples,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    contract = build_training_contract(
        args, source_identities, frontier_identities, frontier_common,
        active_split, active_split_hash, train_rows, development_rows,
        train_supervised_rows, development_supervised_rows,
        catalog_identity, identifiability_audit, expected_runtime_evidence,
    )
    training_contract_path = args.output / "training_contract.json"
    atomic_json_save(contract, training_contract_path)
    training_contract_hash = sha256(training_contract_path)

    history = []
    best_key = None
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(
            model, train_loader, device, optimizer, scaler
        )
        with torch.no_grad():
            development = run_epoch(
                model, development_loader, device, None, scaler
            )
        train_probability = collect_probabilities(
            model, train_dataset, device
        )
        calibration = fit_shared_affine_calibration(
            train_probability,
            [intersection_loss_evidence(row["_ordinal_intersection"])
             for row in train_supervised_rows],
            sample_weights=balanced_ordinal_sample_weights(
                train_supervised_rows
            ),
        )
        development_probability = collect_probabilities(
            model, development_full_dataset, device
        )
        acceptance = evaluate_predictions(
            development_probability, development_rows, calibration,
            minimum_development_pop_gain_pct=(
                args.minimum_development_pop_gain_pct
            ),
            minimum_development_pop_gain_rationale=(
                args.minimum_development_pop_gain_rationale
            ),
        )
        development["acceptance"] = acceptance
        key = checkpoint_selection_key(acceptance)
        eligible = acceptance["development_candidate_status"][
            "training_checkpoint_eligible"
        ]
        record = {
            "epoch": epoch,
            "checkpoint_selection_contract": CHECKPOINT_SELECTION_CONTRACT,
            "training": training,
            "development": development,
            "checkpoint_eligible": eligible,
            "checkpoint_selection_key": key,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if eligible and (best_key is None or key < best_key):
            best_key = key
            atomic_torch_save({
                "schema": ORDINAL_CHECKPOINT_SCHEMA,
                "training_contract": TRAINING_CONTRACT,
                "training_contract_sha256": training_contract_hash,
                "checkpoint_selection_contract":
                    CHECKPOINT_SELECTION_CONTRACT,
                "policy_contract": ORDINAL_POLICY_CONTRACT,
                "output_semantics": ORDINAL_OUTPUT_SEMANTICS,
                "policy_feature_contract": POLICY_FEATURE_CONTRACT,
                "ordinal_policy_state": ordinal_policy_state_dict(model),
                "calibration": calibration.as_dict(),
                "selection_confidence": SELECTION_CONFIDENCE,
                "epoch": epoch,
                "development_loss": development["loss"],
                "development_acceptance": acceptance,
                "calibration_deployable": acceptance[
                    "calibration_deployable"
                ],
                "calibration_evidence": acceptance[
                    "calibration_evidence"
                ],
                "checkpoint_selection_key": key,
                "active_split_sha256": active_split_hash,
                "source_bundles_sha256": contract[
                    "source_bundles_sha256"
                ],
                "frontier_bundles_sha256": contract[
                    "frontier_bundles_sha256"
                ],
                "depth_weights_sha256": contract["depth_weights_sha256"],
                "metric_specs_sha256": contract["metric_specs_sha256"],
                "metric_contract_sha256": contract["metric_contract_sha256"],
                "thresholds_sha256": contract["thresholds_sha256"],
                "orchestration_catalog_sha256": catalog_identity["sha256"],
                "deployment_geometry_allowlist_sha256": contract[
                    "deployment_geometry_allowlist_sha256"
                ],
                "deployment_geometry_structure_sha256": contract[
                    "deployment_geometry_structure_sha256"
                ],
                "sealed_test_productions": contract[
                    "sealed_test_productions"
                ],
            }, args.output / "artistic_policy_ordinal_best.pt")
    checkpoint_path = args.output / "artistic_policy_ordinal_best.pt"
    history_envelope = build_history_envelope(
        history, args.epochs, training_contract_hash, checkpoint_path
    )
    atomic_json_save(history_envelope, args.output / "history.json")
    if best_key is None:
        raise RuntimeError(
            "no ordinal epoch achieved zero unproven development overshoot and "
            "the frozen material realized-pop gain floor in every SDR/HDR "
            "group; no checkpoint was published"
        )


if __name__ == "__main__":
    main()
