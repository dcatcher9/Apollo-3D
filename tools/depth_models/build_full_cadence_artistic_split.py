#!/usr/bin/env python3
"""Replace sparse CHUG windows with a separate full-cadence CHUG production.

The input catalog/active split and all referenced datasets remain immutable.
Only the two native-HDR train/dev productions are replaced; sealed test rows
are copied by metadata reference and their media is never opened.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import artistic_sources as sources
import audit_artistic_dataset_splits as split_audit
import build_combined_artistic_split as combined
import prepare_chug_native_hdr_full_cadence as full


EXPECTED_RETIRED_PRODUCTIONS = {
    "chug_native_pq_v1_training",
    "chug_native_pq_v1_development",
}


def _write_new_or_equal(path: Path, payload, label: str):
    path = path.resolve()
    if path.exists():
        existing = combined._load(path, label)
        if existing != payload:
            raise RuntimeError(f"existing {label} differs; refusing replacement: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _base_without_sparse_native(base_active_path: Path):
    base_active_path = base_active_path.resolve(strict=True)
    active = combined._load(base_active_path, "base active split")
    if not sources.schema_is(active.get("schema"), 1):
        raise RuntimeError("base active split has unsupported schema")
    catalog_path = combined._verified(
        base_active_path, active, "catalog", "catalog_sha256", "base catalog"
    )
    catalog = sources.load_catalog(catalog_path)
    productions = active.get("productions")
    if not isinstance(productions, list):
        raise RuntimeError("base active split productions are missing")
    removed = {
        row.get("production_id") for row in productions
        if isinstance(row, dict) and
        row.get("source_kind") == split_audit.NATIVE_HDR_SOURCE_KIND
    }
    if removed != EXPECTED_RETIRED_PRODUCTIONS:
        raise RuntimeError(
            "base active split does not contain exactly the retired sparse "
            f"CHUG train/dev productions: {sorted(removed)}"
        )
    kept_rows = [
        row for row in productions
        if row.get("production_id") not in removed
    ]
    kept_manifests = [combined._verified(
        base_active_path, row, "dataset_manifest", "dataset_manifest_sha256",
        f"base dataset manifest {row.get('production_id')}",
    ) for row in kept_rows]
    catalog_rows = [
        row for row in catalog["sources"]
        if row.get("production_id") not in removed
    ]
    if (len(catalog_rows) != len(catalog["sources"]) - len(removed) or
            any(row.get("source_kind") == split_audit.NATIVE_HDR_SOURCE_KIND
                for row in catalog_rows)):
        raise RuntimeError("base catalog native-HDR replacement scope differs")
    return active, catalog, catalog_rows, kept_manifests


def _full_cadence_rows(bootstrap_path: Path):
    bootstrap_path = bootstrap_path.resolve(strict=True)
    bootstrap = combined._load(bootstrap_path, "full-cadence CHUG bootstrap")
    if (bootstrap.get("schema") != full.PREPARATION_SCHEMA or
            bootstrap.get("contract") != full.PREPARATION_CONTRACT or
            bootstrap.get("dataset") != full.DATASET_NAME or
            bootstrap.get("full_cadence_contract") !=
            full.FULL_CADENCE_CONTRACT):
        raise RuntimeError("unsupported full-cadence CHUG bootstrap contract")
    datasets = bootstrap.get("datasets")
    if not isinstance(datasets, dict):
        raise RuntimeError("full-cadence CHUG bootstrap lacks datasets")
    manifests = []
    catalog_rows = []
    for split in ("training", "development"):
        entry = datasets.get(split)
        if not isinstance(entry, dict):
            raise RuntimeError(f"full-cadence CHUG bootstrap lacks {split}")
        manifest_path = combined._verified(
            bootstrap_path, entry, "dataset_manifest", "dataset_manifest_sha256",
            f"full-cadence CHUG {split} manifest",
        )
        dataset = combined._load(
            manifest_path, f"full-cadence CHUG {split} dataset"
        )
        production = f"{full.PRODUCTION_PREFIX}_{split}"
        if (dataset.get("schema") != 2 or
                dataset.get("dataset") != full.DATASET_NAME or
                dataset.get("production_id") != production or
                dataset.get("source_kind") != split_audit.NATIVE_HDR_SOURCE_KIND or
                dataset.get("split") != split or
                dataset.get("full_cadence_contract") !=
                full.FULL_CADENCE_CONTRACT or
                dataset.get("frame_count") != full.EXPECTED_SOURCE_FRAMES[split]):
            raise RuntimeError(f"full-cadence CHUG {split} dataset differs")
        identity = split_audit.native_hdr_source_identity(
            dataset, manifest_path, production, verify_media=False
        )
        receipt = combined._load(
            Path(identity["source_provenance"]["download_receipt"]),
            "native HDR download receipt",
        )
        license_name = dataset.get("license")
        license_url = receipt.get("license_url")
        if (receipt.get("license") != license_name or
                not isinstance(license_url, str) or not license_url):
            raise RuntimeError("full-cadence CHUG license provenance differs")
        catalog_rows.append({
            "id": production,
            "production_id": production,
            "source_kind": split_audit.NATIVE_HDR_SOURCE_KIND,
            "source_group": "chug_hdr_reference_capture_groups",
            "split": split,
            "admission": "global_policy",
            "complete_production": True,
            "global_policy_weight": float(dataset["global_policy_weight"]),
            "license": license_name,
            "license_url": license_url,
            "dataset": dataset.get("dataset"),
            "domain": dataset.get("domain"),
            "policy_role": dataset.get("policy_role"),
            "retrieval": {
                "kind": "authenticated-native-hdr-full-cadence-video-collection",
                "bootstrap_manifest": str(bootstrap_path),
                "bootstrap_manifest_sha256": split_audit.sha256(bootstrap_path),
                "dataset_manifest": str(manifest_path),
                "dataset_manifest_sha256": split_audit.sha256(manifest_path),
                "full_cadence_contract": full.FULL_CADENCE_CONTRACT,
            },
        })
        manifests.append(manifest_path)
    return catalog_rows, manifests


def build(base_active_path: Path, bootstrap_path: Path,
          output_catalog: Path, output_active: Path):
    output_catalog = output_catalog.resolve()
    output_active = output_active.resolve()
    if output_catalog == output_active:
        raise RuntimeError("catalog and active split outputs must differ")
    active, base_catalog, catalog_rows, base_manifests = (
        _base_without_sparse_native(base_active_path)
    )
    native_rows, native_manifests = _full_cadence_rows(bootstrap_path)
    merged_rows = catalog_rows + native_rows
    identifiers = [row.get("id") for row in merged_rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("full-cadence catalog repeats a source id")
    catalog = {
        "schema": 2,
        "purpose": (
            "Authenticated SDR-origin plus full-cadence native-PQ artistic-policy "
            "training, development and independent sealed tests"
        ),
        "sealed_test_policy": base_catalog.get("sealed_test_policy"),
        "sources": merged_rows,
    }
    sources.validate_catalog(catalog, "full-cadence combined source catalog")
    _write_new_or_equal(output_catalog, catalog, "full-cadence source catalog")
    updated = split_audit.audit(
        output_catalog, base_manifests + native_manifests
    )
    expected_tests = active.get("split_productions", {}).get("test")
    if updated.get("split_productions", {}).get("test") != expected_tests:
        raise RuntimeError("sealed test production assignment changed")
    _write_new_or_equal(output_active, updated, "full-cadence active split")
    return catalog, updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-active-split", required=True, type=Path)
    parser.add_argument("--full-cadence-bootstrap-manifest", required=True, type=Path)
    parser.add_argument("--output-catalog", required=True, type=Path)
    parser.add_argument("--output-active-split", required=True, type=Path)
    args = parser.parse_args()
    _, active = build(
        args.base_active_split, args.full_cadence_bootstrap_manifest,
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
