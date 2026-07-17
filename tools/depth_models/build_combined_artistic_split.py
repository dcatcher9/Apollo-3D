#!/usr/bin/env python3
"""Publish one authenticated SDR-origin + native-PQ active split.

The existing dataset manifests remain immutable.  This builder appends the two
native-PQ productions to a previously frozen SDR active split, then delegates
all source-file, provenance, collection-overlap and sealed-test checks to the
canonical split auditor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import artistic_sources as sources
import audit_artistic_dataset_splits as split_audit


NATIVE_BOOTSTRAP_SCHEMA = 3
NATIVE_BOOTSTRAP_CONTRACT = "apollo-chug-native-pq-training-v3"


def _load(path: Path, label):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid {label}: {path}")
    return value


def _referenced(document_path: Path, value, label):
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"missing {label}")
    path = Path(value)
    if not path.is_absolute():
        path = document_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    return path


def _verified(document_path: Path, row, path_key, hash_key, label):
    path = _referenced(document_path, row.get(path_key), label)
    expected = split_audit._required_hash(row.get(hash_key), hash_key)
    if split_audit.sha256(path) != expected:
        raise RuntimeError(f"stale {label}: {path}")
    return path


def build(base_active_path: Path, native_bootstrap_path: Path,
          output_catalog: Path, output_active: Path):
    base_active_path = base_active_path.resolve()
    native_bootstrap_path = native_bootstrap_path.resolve()
    output_catalog = output_catalog.resolve()
    output_active = output_active.resolve()
    if output_catalog == output_active:
        raise RuntimeError("catalog and active-split outputs must differ")

    base = _load(base_active_path, "base active split")
    if not sources.schema_is(base.get("schema"), 1):
        raise RuntimeError("base active split has unsupported schema")
    base_catalog_path = _verified(
        base_active_path, base, "catalog", "catalog_sha256", "base catalog"
    )
    base_catalog = sources.load_catalog(base_catalog_path)
    base_manifests = []
    for index, row in enumerate(base.get("productions", ())):
        if not isinstance(row, dict):
            raise RuntimeError(f"base production {index} is invalid")
        base_manifests.append(_verified(
            base_active_path, row, "dataset_manifest",
            "dataset_manifest_sha256", "base dataset manifest",
        ))
    if ({row.get("production_id") for row in base.get("productions", ())} !=
            {production for values in base.get("split_productions", {}).values()
             for production in values}):
        raise RuntimeError("base active split production assignment is incomplete")

    bootstrap = _load(native_bootstrap_path, "native HDR bootstrap")
    if (not sources.schema_is(bootstrap.get("schema"), NATIVE_BOOTSTRAP_SCHEMA) or
            bootstrap.get("contract") != NATIVE_BOOTSTRAP_CONTRACT):
        raise RuntimeError("native HDR bootstrap has unsupported contract")
    native_manifests = []
    native_rows = []
    bootstrap_datasets = bootstrap.get("datasets")
    if not isinstance(bootstrap_datasets, dict):
        raise RuntimeError("native HDR bootstrap datasets are missing")
    for split in ("training", "development"):
        entry = bootstrap_datasets.get(split)
        if not isinstance(entry, dict):
            raise RuntimeError(f"native HDR bootstrap lacks {split}")
        manifest = _verified(
            native_bootstrap_path, entry, "dataset_manifest",
            "dataset_manifest_sha256", f"native HDR {split} dataset manifest",
        )
        dataset = _load(manifest, f"native HDR {split} dataset")
        production = dataset.get("production_id")
        if (dataset.get("source_kind") != split_audit.NATIVE_HDR_SOURCE_KIND or
                dataset.get("split") != split or
                not isinstance(production, str) or not production):
            raise RuntimeError(f"native HDR {split} dataset identity disagrees")
        identity = split_audit.native_hdr_source_identity(
            dataset, manifest, production, verify_media=False
        )
        receipt_path = Path(identity["source_provenance"]["download_receipt"])
        receipt = _load(receipt_path, "native HDR download receipt")
        license_name = dataset.get("license")
        license_url = receipt.get("license_url")
        if (not isinstance(license_name, str) or not license_name or
                receipt.get("license") != license_name or
                not isinstance(license_url, str) or not license_url):
            raise RuntimeError("native HDR license provenance disagrees")
        native_rows.append({
            "id": production,
            "production_id": production,
            "source_kind": split_audit.NATIVE_HDR_SOURCE_KIND,
            "source_group": "chug_hdr_reference_capture_groups",
            "split": split,
            "admission": "global_policy",
            "complete_production": True,
            "global_policy_weight": float(dataset.get("global_policy_weight", 0.0)),
            "license": license_name,
            "license_url": license_url,
            "dataset": dataset.get("dataset"),
            "domain": dataset.get("domain"),
            "policy_role": dataset.get("policy_role"),
            "retrieval": {
                "kind": "authenticated-native-hdr-video-collection",
                "bootstrap_manifest": str(native_bootstrap_path),
                "bootstrap_manifest_sha256": split_audit.sha256(
                    native_bootstrap_path
                ),
                "dataset_manifest": str(manifest),
                "dataset_manifest_sha256": split_audit.sha256(manifest),
            },
        })
        native_manifests.append(manifest)

    catalog_sources = list(base_catalog["sources"]) + native_rows
    identifiers = [row.get("id") for row in catalog_sources]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("combined source catalog repeats a source id")
    catalog = {
        "schema": 2,
        "purpose": (
            "Authenticated SDR-origin plus native-PQ artistic-policy training, "
            "development and independent sealed tests"
        ),
        "sealed_test_policy": base_catalog.get("sealed_test_policy"),
        "sources": catalog_sources,
    }
    sources.validate_catalog(catalog, "combined source catalog")
    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    output_catalog.write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    active = split_audit.audit(
        output_catalog, base_manifests + native_manifests
    )
    output_active.parent.mkdir(parents=True, exist_ok=True)
    output_active.write_text(
        json.dumps(active, indent=2) + "\n", encoding="utf-8"
    )
    return catalog, active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-active-split", required=True, type=Path)
    parser.add_argument("--native-bootstrap-manifest", required=True, type=Path)
    parser.add_argument("--output-catalog", required=True, type=Path)
    parser.add_argument("--output-active-split", required=True, type=Path)
    args = parser.parse_args()
    _, active = build(
        args.base_active_split, args.native_bootstrap_manifest,
        args.output_catalog, args.output_active_split,
    )
    print(json.dumps({
        "active_split": str(args.output_active_split.resolve()),
        "active_split_sha256": split_audit.sha256(
            args.output_active_split.resolve()
        ),
        "split_productions": active["split_productions"],
        "totals": active["totals"],
    }, indent=2))


if __name__ == "__main__":
    main()
