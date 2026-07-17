#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ordinal_result_cache as result_cache
import preprocessing_artifact_cache as artifact_cache


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def harness_bytes(value):
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def identity(split="training", metric=None, thresholds=None):
    return result_cache.scored_cache_identity(
        split=split,
        render_identity_sha256=digest("render"),
        metric_sha256=metric or digest("metric")[:16],
        thresholds_sha256=thresholds or digest("thresholds"),
        scales=(1.0, 1.3),
        artifact_scales=(1.0, 1.3),
        contracts={
            "driver": {"schema": 4, "contract": "driver-v4"},
            "evaluator": {"schema": 30, "contract": "eval-v30"},
            "gate": {"schema": 1, "contract": "selected-gate-v1"},
            "compaction": {
                "schema": 3,
                "contract": "apollo-ordinal-safety-compaction-v3",
            },
            "report": {"schema": 1, "contract": "report-v1"},
        },
    )


def tree_rows(root, relative_to):
    return {
        path.resolve().relative_to(relative_to.resolve()).as_posix():
            artifact_cache.sha256_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def depth_origin(mode):
    return {
        "mode": mode,
        "key_sha256": digest("depth-key"),
        "manifest_sha256": digest("depth-manifest"),
        "boundary": "completed-production-depth-state-before-warp-prefilter",
        "selected_state_frame_count": 1,
        "runtime_scene_frame_count": 1,
    }


def scale_contract(scale, mode):
    return {
        "schema": 31,
        "metric_sha256": digest("metric")[:16],
        "multiscale_batch": True,
        "artistic_scale_override": scale,
        "depth_state_cache_mode": mode,
        "depth_state_cache_key_sha256": digest("depth-key"),
        "depth_state_manifest_sha256": digest("depth-manifest"),
    }


def origin_documents(mode):
    calls = 0 if mode == "authenticated-replay" else 1
    harness = {
        "schema": 5,
        "contract": "apollo-harness-artistic-multiscale-v5",
        "shipping_estimator_calls_per_source_frame": calls,
        "depth_state_cache": depth_origin(mode),
    }
    encoded_harness = harness_bytes(harness)
    rows = []
    for index, scale in enumerate((1.0, 1.3)):
        slug = result_cache._scale_slug(scale)
        contract = scale_contract(scale, mode)
        encoded = harness_bytes(contract)
        contract_sha = hashlib.sha256(encoded).hexdigest()
        rows.append({
            "index": index,
            "scale": scale,
            "directory": f"scales/{slug}",
            "contract_sha256": contract_sha,
            "artifacts": [{
                "path": f"scales/{slug}/contract.json",
                "size": len(encoded),
                "sha256": contract_sha,
            }],
        })
    manifest = {
        "schema": 3,
        "contract": "apollo-authenticated-artistic-multiscale-batch-v3",
        "harness_contract_sha256": hashlib.sha256(encoded_harness).hexdigest(),
        "depth_state_cache": depth_origin(mode),
        "scale_rows": rows,
    }
    manifest_sha = hashlib.sha256(
        artifact_cache.canonical_bytes(manifest)
    ).hexdigest()
    receipt = {
        "schema": 1,
        "contract": "apollo-multiscale-render-identity-v1",
        "render_identity_sha256": digest("render"),
        "batch_manifest_sha256": manifest_sha,
    }
    return harness, manifest, receipt


def make_gate(path, run_name, results_sha, manifest_sha, contract_sha):
    records = [
        {
            "record": "header",
            "schema": 1,
            "contract": "apollo-target-frame-gate-evidence-v2",
            "metric_sha256": digest("metric")[:16],
            "thresholds_sha256": digest("thresholds"),
            "results_sha256": results_sha,
            "run_name": run_name,
            "multiscale_batch_manifest_sha256": manifest_sha,
        },
        {
            "record": "clip", "clip": "clip", "frame_count": 1,
            "harness_contract_sha256": contract_sha,
        },
        {"record": "frame", "clip": "clip", "frame_id": 7},
        {
            "record": "clip_end",
            "clip": "clip",
            "frame_count": 1,
            "frame_records_sha256": digest("frame"),
        },
        {
            "record": "trailer",
            "payload_record_count": 0,
            "clip_count": 1,
            "frame_count": 1,
            "payload_sha256": "",
        },
    ]
    result_cache._write_gate(path, records)


def make_scale(root, clips_root, scale, mode="cold-export",
               with_artifacts=True):
    root.mkdir(parents=True)
    harness, manifest, receipt = origin_documents(mode)
    manifest_sha = hashlib.sha256(
        artifact_cache.canonical_bytes(manifest)
    ).hexdigest()
    contract = scale_contract(scale, mode)
    contract_sha = hashlib.sha256(harness_bytes(contract)).hexdigest()
    result = {
        "meta": {
            "run_name": root.name,
            "clips_root": str(clips_root.resolve()),
            "clip_hash_manifest": str(
                (clips_root / "clip_hash_manifest.json").resolve()
            ),
            "metric_sha256": digest("metric")[:16],
            "timestamp": "2026-07-16T12:00:00",
            "git_sha": "abc1234",
            "git_dirty": True,
            "multiscale_batch_manifest_sha256": manifest_sha,
        },
        "verdict": "comparison_only",
        "clips": {"clip": {"meta": {"scale": scale}}},
    }
    results = root / result_cache.RESULTS_FILENAME
    results.write_bytes(artifact_cache.canonical_bytes(result))
    gate = root / result_cache.FRAME_GATE_FILENAME
    make_gate(
        gate, root.name, artifact_cache.sha256_file(results),
        manifest_sha, contract_sha,
    )

    runtime = root / "clip" / "runtime_scene_evidence.json"
    runtime.parent.mkdir()
    runtime.write_bytes(artifact_cache.canonical_bytes({
        "schema": 1, "scale": scale,
    }))
    provenance = root / "multiscale_provenance" / "clip"
    provenance.mkdir(parents=True)
    for name, value in {
            "contract.json": contract,
            "multiscale_batch_manifest.json": manifest,
            "multiscale_contract.json": harness,
            "render_identity.json": receipt,
            }.items():
        encoded = (
            harness_bytes(value)
            if name in {"contract.json", "multiscale_contract.json"} else
            artifact_cache.canonical_bytes(value)
        )
        (provenance / name).write_bytes(encoded)
    if with_artifacts:
        evidence = root / "artifact_evidence" / "clip"
        evidence.mkdir(parents=True)
        (evidence / "visual_evidence.json").write_bytes(
            artifact_cache.canonical_bytes({"scale": scale})
        )
        (evidence / "sbs_00007.png").write_bytes(b"small-sparse-evidence")
    provenance_rows = tree_rows(provenance, root)
    artifact_rows = tree_rows(root / "artifact_evidence" / "clip", root)
    marker = {
        "schema": 3,
        "contract": "apollo-ordinal-safety-compaction-v3",
        "results_sha256": artifact_cache.sha256_file(results),
        "frame_gate_evidence_sha256": artifact_cache.sha256_file(gate),
        "runtime_scene_evidence_sha256": {
            "clip": artifact_cache.sha256_file(runtime),
        },
        "multiscale_provenance_sha256": provenance_rows,
        "artifact_evidence_sha256": artifact_rows,
        "retained_role": "selected-target-safety-label-evidence",
        "deleted_files": 20,
        "deleted_bytes": 123456,
    }
    (root / result_cache.COMPACTION_FILENAME).write_bytes(
        artifact_cache.canonical_bytes(marker)
    )
    return {
        "scale": scale,
        "scale_slug": result_cache._scale_slug(scale),
        "run": str(root.resolve()),
        "results_sha256": artifact_cache.sha256_file(results),
        "frame_gate_evidence_sha256": artifact_cache.sha256_file(gate),
        "provenance_sha256": {
            Path(path).name: value for path, value in provenance_rows.items()
        },
    }


class OrdinalResultCacheTest(unittest.TestCase):
    def fixture(self, root, mode="cold-export"):
        clips_root = root / "old-dataset" / "training"
        clips_root.mkdir(parents=True)
        (clips_root / "clip_hash_manifest.json").write_text(
            "{}", encoding="utf-8"
        )
        outputs = {}
        rows = []
        for scale in (1.0, 1.3):
            slug = result_cache._scale_slug(scale)
            output = root / "old-eval" / f"old-prefix-{slug}"
            outputs[slug] = output
            rows.append(make_scale(output, clips_root, scale, mode=mode))
        _, manifest, receipt = origin_documents(mode)
        summary = {
            "schema": 4,
            "contract": "driver-v4",
            "clip": "clip",
            "render_identity_sha256": digest("render"),
            "metric_sha256": digest("metric")[:16],
            "scale_score_jobs": 8,
            "scale_results": rows,
            "batch_artifacts_deleted_after_authenticated_scoring": True,
            "batch_manifest_sha256": hashlib.sha256(
                artifact_cache.canonical_bytes(manifest)
            ).hexdigest(),
            "render_identity_receipt_sha256": hashlib.sha256(
                artifact_cache.canonical_bytes(receipt)
            ).hexdigest(),
        }
        summary_path = root / "old-summary.json"
        summary_path.write_bytes(artifact_cache.canonical_bytes(summary))
        return clips_root, outputs, summary_path

    def test_identity_is_path_independent_and_contract_sensitive(self):
        baseline = identity()
        self.assertEqual(
            artifact_cache.DirectoryArtifactCache.key(baseline),
            artifact_cache.DirectoryArtifactCache.key(identity()),
        )
        self.assertNotEqual(
            artifact_cache.DirectoryArtifactCache.key(baseline),
            artifact_cache.DirectoryArtifactCache.key(
                identity(metric=digest("metric-v2")[:16])
            ),
        )
        self.assertNotEqual(
            artifact_cache.DirectoryArtifactCache.key(baseline),
            artifact_cache.DirectoryArtifactCache.key(
                identity(thresholds=digest("thresholds-v2"))
            ),
        )

    def test_sealed_split_is_rejected_before_cache_access(self):
        with self.assertRaisesRegex(RuntimeError, "train/development only"):
            identity(split="sealed-test")

    def test_compacted_grid_round_trips_to_new_paths_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_clips, old_outputs, old_summary = self.fixture(root)
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            key = result_cache.publish(
                cache,
                identity(),
                summary_path=old_summary,
                scale_outputs=old_outputs,
                clip="clip",
                clips_root=old_clips,
            )
            cache_bytes = b"".join(
                path.read_bytes() for path in sorted(
                    cache._entry(key).rglob("*")) if path.is_file()
            )
            self.assertNotIn(b"old-prefix", cache_bytes)
            self.assertNotIn(str(old_clips).encode("utf-8"), cache_bytes)

            new_clips = root / "new-dataset" / "training"
            new_clips.mkdir(parents=True)
            (new_clips / "clip_hash_manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            new_outputs = {
                slug: root / "new-eval" / f"new-prefix-{slug}"
                for slug in old_outputs
            }
            new_summary = root / "new-summary.json"
            with mock.patch.object(
                    artifact_cache, "inheriting_temporary_directory",
                    wraps=artifact_cache.inheriting_temporary_directory,
                    ) as inheriting_staging:
                summary = result_cache.materialize(
                    cache,
                    identity(),
                    summary_path=new_summary,
                    scale_outputs=new_outputs,
                    clips_root=new_clips,
                    score_workers=3,
                )
            self.assertGreaterEqual(
                inheriting_staging.call_count, len(new_outputs)
            )
            self.assertIsNotNone(summary)
            self.assertEqual(summary["scale_score_jobs"], 3)
            for row in summary["scale_results"]:
                output = new_outputs[row["scale_slug"]]
                self.assertEqual(Path(row["run"]), output.resolve())
                result = json.loads(
                    (output / result_cache.RESULTS_FILENAME).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(result["meta"]["run_name"], output.name)
                self.assertTrue(
                    result["meta"]["git_sha"].startswith("score-cache-")
                )
                self.assertFalse(result["meta"]["git_dirty"])
                self.assertEqual(
                    result["meta"]["ordinal_score_cache"]["contract"],
                    result_cache.PACKET_CONTRACT,
                )
                self.assertEqual(
                    Path(result["meta"]["clips_root"]), new_clips.resolve()
                )
                gate = result_cache._read_gate(
                    output / result_cache.FRAME_GATE_FILENAME
                )
                self.assertEqual(gate[0]["run_name"], output.name)
                self.assertEqual(
                    gate[0]["results_sha256"],
                    artifact_cache.sha256_file(
                        output / result_cache.RESULTS_FILENAME
                    ),
                )
                marker = json.loads((
                    output / result_cache.COMPACTION_FILENAME
                ).read_text(encoding="utf-8"))
                self.assertTrue(marker["materialized_from_scored_cache"])
                self.assertEqual(
                    marker["frame_gate_evidence_sha256"],
                    artifact_cache.sha256_file(
                        output / result_cache.FRAME_GATE_FILENAME
                    ),
                )
                provenance = (
                    output / "multiscale_provenance" / "clip"
                )
                contract = json.loads((
                    provenance / "contract.json"
                ).read_text(encoding="utf-8"))
                harness = json.loads((
                    provenance / "multiscale_contract.json"
                ).read_text(encoding="utf-8"))
                self.assertEqual(
                    contract["depth_state_cache_mode"],
                    result_cache.CACHED_DEPTH_MODE,
                )
                self.assertEqual(
                    harness["depth_state_cache"]["mode"],
                    result_cache.CACHED_DEPTH_MODE,
                )
                self.assertEqual(
                    harness["shipping_estimator_calls_per_source_frame"], 0
                )
            self.assertEqual(
                summary["scored_result_cache"]["contract"],
                result_cache.PACKET_CONTRACT,
            )

    def test_large_or_uncompacted_leftover_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips_root, outputs, summary = self.fixture(root)
            next(iter(outputs.values()), None).joinpath(
                "sbs_00007.png"
            ).write_bytes(b"uncompacted")
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            with self.assertRaisesRegex(RuntimeError, "exactly compacted"):
                result_cache.publish(
                    cache,
                    identity(),
                    summary_path=summary,
                    scale_outputs=outputs,
                    clip="clip",
                    clips_root=clips_root,
                )

    def test_reparse_evidence_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips_root, outputs, summary = self.fixture(root)
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            evidence = outputs["s100"] / "artifact_evidence" / "clip"
            original = artifact_cache.is_link_or_junction

            def report_reparse(path):
                path = Path(path)
                return path == evidence or original(path)

            with mock.patch.object(
                    artifact_cache, "is_link_or_junction",
                    side_effect=report_reparse):
                with self.assertRaisesRegex(RuntimeError, "contains a link"):
                    result_cache.publish(
                        cache, identity(), summary_path=summary,
                        scale_outputs=outputs, clip="clip",
                        clips_root=clips_root,
                    )

    def test_same_identity_with_different_compacted_bytes_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips_root, outputs, summary = self.fixture(root)
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            result_cache.publish(
                cache,
                identity(),
                summary_path=summary,
                scale_outputs=outputs,
                clip="clip",
                clips_root=clips_root,
            )
            output = outputs["s100"]
            runtime = output / "clip" / "runtime_scene_evidence.json"
            runtime.write_bytes(artifact_cache.canonical_bytes({
                "schema": 1, "scale": 1.0, "changed": True,
            }))
            marker_path = output / result_cache.COMPACTION_FILENAME
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["runtime_scene_evidence_sha256"]["clip"] = \
                artifact_cache.sha256_file(runtime)
            marker_path.write_bytes(artifact_cache.canonical_bytes(marker))
            with self.assertRaisesRegex(
                    RuntimeError, "payload differs|different packet bytes"):
                result_cache.publish(
                    cache,
                    identity(),
                    summary_path=summary,
                    scale_outputs=outputs,
                    clip="clip",
                    clips_root=clips_root,
                )

    def test_source_mutation_during_packet_copy_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips_root, outputs, summary = self.fixture(root)
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            original = result_cache._copy_static_tree
            changed = False

            def mutate_then_copy(source, destination, clip, **kwargs):
                nonlocal changed
                if not changed:
                    changed = True
                    runtime = Path(source) / clip / \
                        "runtime_scene_evidence.json"
                    runtime.write_bytes(artifact_cache.canonical_bytes({
                        "schema": 1, "scale": 1.0,
                        "changed_during_copy": True,
                    }))
                return original(source, destination, clip, **kwargs)

            with mock.patch.object(
                    result_cache, "_copy_static_tree",
                    side_effect=mutate_then_copy):
                with self.assertRaisesRegex(
                        RuntimeError, "changed while copying static evidence"):
                    result_cache.publish(
                        cache, identity(), summary_path=summary,
                        scale_outputs=outputs, clip="clip",
                        clips_root=clips_root,
                    )

    def test_nondeterministic_run_origin_is_normalized_out_of_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips_root, outputs, summary_path = self.fixture(root)
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            first = result_cache.publish(
                cache,
                identity(),
                summary_path=summary_path,
                scale_outputs=outputs,
                clip="clip",
                clips_root=clips_root,
            )
            output = outputs["s100"]
            results_path = output / result_cache.RESULTS_FILENAME
            result = json.loads(results_path.read_text(encoding="utf-8"))
            result["meta"].update({
                "timestamp": "2026-07-17T01:02:03",
                "git_sha": "different",
                "git_dirty": False,
            })
            results_path.write_bytes(artifact_cache.canonical_bytes(result))
            results_sha = artifact_cache.sha256_file(results_path)
            gate_path = output / result_cache.FRAME_GATE_FILENAME
            gate = result_cache._read_gate(gate_path)
            gate[0]["results_sha256"] = results_sha
            result_cache._write_gate(gate_path, gate)
            gate_sha = artifact_cache.sha256_file(gate_path)
            marker_path = output / result_cache.COMPACTION_FILENAME
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["results_sha256"] = results_sha
            marker["frame_gate_evidence_sha256"] = gate_sha
            marker_path.write_bytes(artifact_cache.canonical_bytes(marker))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row = next(
                item for item in summary["scale_results"]
                if item["scale_slug"] == "s100"
            )
            row["results_sha256"] = results_sha
            row["frame_gate_evidence_sha256"] = gate_sha
            summary_path.write_bytes(artifact_cache.canonical_bytes(summary))

            second = result_cache.publish(
                cache,
                identity(),
                summary_path=summary_path,
                scale_outputs=outputs,
                clip="clip",
                clips_root=clips_root,
            )
            self.assertEqual(first, second)

    def test_cold_export_and_authenticated_replay_share_one_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            cold_clips, cold_outputs, cold_summary = self.fixture(
                root / "cold", mode="cold-export"
            )
            replay_clips, replay_outputs, replay_summary = self.fixture(
                root / "replay", mode="authenticated-replay"
            )
            first = result_cache.publish(
                cache, identity(), summary_path=cold_summary,
                scale_outputs=cold_outputs, clip="clip",
                clips_root=cold_clips,
            )
            second = result_cache.publish(
                cache, identity(), summary_path=replay_summary,
                scale_outputs=replay_outputs, clip="clip",
                clips_root=replay_clips,
            )
            self.assertEqual(first, second)

    def test_interrupted_scale_commit_is_authenticated_and_resumed(self):
        class SimulatedProcessDeath(BaseException):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips_root, outputs, summary = self.fixture(root)
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            result_cache.publish(
                cache, identity(), summary_path=summary,
                scale_outputs=outputs, clip="clip", clips_root=clips_root,
            )
            destination_clips = root / "destination" / "training"
            destination_clips.mkdir(parents=True)
            (destination_clips / "clip_hash_manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            destination_outputs = {
                slug: root / "destination-eval" / f"new-{slug}"
                for slug in outputs
            }
            destination_summary = root / "destination-summary.json"

            def die_after_rename(staging, output):
                Path(staging).rename(output)
                raise SimulatedProcessDeath()

            with mock.patch.object(
                    result_cache, "_commit_scale_output",
                    side_effect=die_after_rename):
                with self.assertRaises(SimulatedProcessDeath):
                    result_cache.materialize(
                        cache, identity(), summary_path=destination_summary,
                        scale_outputs=destination_outputs,
                        clips_root=destination_clips, score_workers=2,
                    )
            self.assertEqual(
                sum(path.exists() for path in destination_outputs.values()), 1
            )
            recovered = result_cache.materialize(
                cache, identity(), summary_path=destination_summary,
                scale_outputs=destination_outputs,
                clips_root=destination_clips, score_workers=2,
            )
            self.assertEqual(
                recovered["scored_result_cache"][
                    "recovered_interrupted_scale_outputs"
                ],
                1,
            )
            self.assertTrue(all(
                path.is_dir() for path in destination_outputs.values()
            ))
            transaction_ids = set()
            for output in destination_outputs.values():
                marker = json.loads((
                    output / result_cache.COMPACTION_FILENAME
                ).read_text(encoding="utf-8"))
                transaction_ids.add(marker["score_cache_transaction_id"])
            self.assertEqual(len(transaction_ids), 1)

    def test_hard_kill_orphan_temps_are_owned_and_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips_root, outputs, summary = self.fixture(root)
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            result_cache.publish(
                cache, identity(), summary_path=summary,
                scale_outputs=outputs, clip="clip", clips_root=clips_root,
            )
            destination_clips = root / "destination" / "training"
            destination_clips.mkdir(parents=True)
            (destination_clips / "clip_hash_manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            destination_outputs = {
                slug: root / "destination-eval" / f"new-{slug}"
                for slug in outputs
            }
            destination_summary = root / "destination-summary.json"
            key = artifact_cache.DirectoryArtifactCache.key(identity())
            stale_lock = result_cache._MaterializationTransaction(
                destination_summary, destination_outputs, key
            )
            transaction = stale_lock.acquire()["current"]
            stale_id = transaction["transaction_id"]
            packet = result_cache._packet_transaction_path(
                destination_summary, stale_id
            )
            packet.mkdir(parents=True)
            packet_partial = packet.with_name(
                packet.name + ".cache-partial-dead"
            )
            packet_partial.mkdir()
            summary_temp = result_cache._summary_transaction_temporary(
                destination_summary, stale_id
            )
            summary_temp.write_bytes(b"partial")
            scale_temps = []
            for output in destination_outputs.values():
                output.parent.mkdir(parents=True, exist_ok=True)
                temp = output.parent / (
                    result_cache._scale_transaction_prefix(output, stale_id) +
                    "dead"
                )
                temp.mkdir()
                scale_temps.append(temp)
            unrelated = next(iter(destination_outputs.values())).parent / (
                result_cache._scale_transaction_prefix(
                    next(iter(destination_outputs.values())), "f" * 32
                ) + "keep"
            )
            unrelated.mkdir()
            stale_paths = [
                packet, packet_partial, summary_temp, *scale_temps,
            ]
            # Preserve the active receipt while releasing only the OS lock:
            # this is the durable state left by an abrupt process death.
            stale_lock.release(preserve_active=True)

            recovered = result_cache.materialize(
                cache, identity(), summary_path=destination_summary,
                scale_outputs=destination_outputs,
                clips_root=destination_clips, score_workers=2,
            )
            self.assertEqual(
                recovered["scored_result_cache"]
                ["recovered_interrupted_temp_paths"],
                len(stale_paths),
            )
            self.assertTrue(all(not path.exists() for path in stale_paths))
            self.assertTrue(unrelated.is_dir())
            self.assertTrue(all(
                output.is_dir() for output in destination_outputs.values()
            ))

    def test_live_materialization_transaction_blocks_second_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips_root, outputs, summary = self.fixture(root)
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            result_cache.publish(
                cache, identity(), summary_path=summary,
                scale_outputs=outputs, clip="clip", clips_root=clips_root,
            )
            destination_clips = root / "destination" / "training"
            destination_clips.mkdir(parents=True)
            (destination_clips / "clip_hash_manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            destination_outputs = {
                slug: root / "destination-eval" / f"new-{slug}"
                for slug in outputs
            }
            destination_summary = root / "destination-summary.json"
            key = artifact_cache.DirectoryArtifactCache.key(identity())
            transaction = result_cache._MaterializationTransaction(
                destination_summary, destination_outputs, key
            )
            transaction.acquire()
            try:
                with self.assertRaisesRegex(
                        RuntimeError, "already active"):
                    result_cache.materialize(
                        cache, identity(), summary_path=destination_summary,
                        scale_outputs=destination_outputs,
                        clips_root=destination_clips, score_workers=2,
                    )
                self.assertFalse(any(
                    path.exists() for path in destination_outputs.values()
                ))
                child = subprocess.run(
                    [
                        sys.executable, "-c",
                        (
                            "import json, pathlib, sys; "
                            "import ordinal_result_cache as c; "
                            "outputs={k:pathlib.Path(v) for k,v in "
                            "json.loads(sys.argv[2]).items()}; "
                            "lock=c._MaterializationTransaction("
                            "pathlib.Path(sys.argv[1]),outputs,sys.argv[3]); "
                            "\ntry:\n lock.acquire()\n"
                            "except RuntimeError as e:\n"
                            " print(e)\n"
                            " raise SystemExit(0 if 'already active' in "
                            "str(e) else 2)\n"
                            "else:\n lock.release(preserve_active=False)\n"
                            " raise SystemExit(3)\n"
                        ),
                        str(destination_summary),
                        json.dumps({
                            slug: str(path)
                            for slug, path in destination_outputs.items()
                        }),
                        key,
                    ],
                    cwd=Path(__file__).resolve().parent,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(
                    child.returncode, 0,
                    msg=f"stdout={child.stdout}\nstderr={child.stderr}",
                )
            finally:
                transaction.release(preserve_active=False)

    def test_scorer_runtime_identity_is_stable_and_path_free(self):
        python = Path(
            r"E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe"
        )
        if not python.is_file():
            self.skipTest("artistic-policy scorer runtime is unavailable")
        first = result_cache.query_scorer_runtime_identity(python)
        second = result_cache.query_scorer_runtime_identity(python)
        self.assertEqual(first, second)
        encoded = artifact_cache.canonical_bytes(first).decode("utf-8")
        self.assertNotIn(str(python), encoded)
        self.assertNotIn("\\", encoded)
        runtime_names = {
            row["filename"].lower()
            for row in first["python"]["runtime_binaries"]
        }
        self.assertIn("python.exe", runtime_names)
        if sys.platform == "win32":
            self.assertTrue(any(
                name.startswith("python") and name.endswith(".dll")
                for name in runtime_names
            ))
        for package in ("numpy", "pillow"):
            binaries = first["packages"][package]["native_binaries"]
            self.assertTrue(all(
                row["filename_suffix"].endswith(
                    (".pyd", ".so", ".dll", ".dylib")
                )
                for row in binaries
            ))
            sources = first["packages"][package]["python_sources"]
            self.assertTrue(sources)
            self.assertTrue(all(
                row["path"].endswith(".py") for row in sources
            ))
        numpy_names = {
            row["filename"].lower()
            for row in first["packages"]["numpy"]["native_binaries"]
        }
        self.assertTrue(any("pocketfft" in name for name in numpy_names))
        self.assertTrue(any("umath_linalg" in name for name in numpy_names))
        self.assertEqual(set(first["packages"]), {"numpy", "pillow"})

    def test_cache_miss_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips_root = root / "dataset" / "training"
            clips_root.mkdir(parents=True)
            outputs = {
                result_cache._scale_slug(scale):
                    root / "eval" / f"prefix-{result_cache._scale_slug(scale)}"
                for scale in (1.0, 1.3)
            }
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            summary = root / "absent-summary-parent" / "summary.json"
            value = result_cache.materialize(
                cache,
                identity(),
                summary_path=summary,
                scale_outputs=outputs,
                clips_root=clips_root,
                score_workers=8,
            )
            self.assertIsNone(value)
            self.assertFalse(any(path.exists() for path in outputs.values()))
            self.assertFalse(summary.parent.exists())


if __name__ == "__main__":
    unittest.main()
