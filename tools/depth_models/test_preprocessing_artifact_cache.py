import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import preprocessing_artifact_cache as artifact_cache


def _identity(source_hash="1" * 64, width=1280):
    return artifact_cache.cache_identity(
        artifact_kind="fixture",
        source={"bytes": 4, "sha256": source_hash},
        selection={"start": 0, "end": 10},
        preprocessing={"width": width, "algorithm": "fixture-v1"},
        color_contract={"transfer": "srgb"},
        code={"schema": 1, "sha256": "2" * 64},
    )


class PreprocessingArtifactCacheTests(unittest.TestCase):
    def test_key_covers_source_selection_preprocessing_color_and_code(self):
        baseline = artifact_cache.DirectoryArtifactCache.key(_identity())
        variants = [
            _identity(source_hash="3" * 64),
            artifact_cache.cache_identity(
                artifact_kind="fixture",
                source={"bytes": 4, "sha256": "1" * 64},
                selection={"start": 1, "end": 10},
                preprocessing={"width": 1280, "algorithm": "fixture-v1"},
                color_contract={"transfer": "srgb"},
                code={"schema": 1, "sha256": "2" * 64},
            ),
            _identity(width=960),
        ]
        self.assertTrue(all(
            artifact_cache.DirectoryArtifactCache.key(value) != baseline
            for value in variants
        ))

    def test_publish_and_materialize_rehash_every_payload_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "a.bin").write_bytes(b"abcd")
            (source / "nested").mkdir()
            (source / "nested" / "b.json").write_text(
                json.dumps({"value": 1}), encoding="utf-8"
            )
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            identity = _identity()
            key = cache.publish(identity, source)
            destination = root / "destination"
            self.assertTrue(cache.materialize(identity, destination))
            self.assertEqual((destination / "a.bin").read_bytes(), b"abcd")

            payload = cache._entry(key) / artifact_cache.PAYLOAD_DIRECTORY
            (payload / "a.bin").write_bytes(b"abce")
            with self.assertRaisesRegex(RuntimeError, "payload bytes differ"):
                cache.materialize(identity, root / "corrupt-copy")
            self.assertFalse((root / "corrupt-copy").exists())

    def test_validated_payload_receipt_pins_inner_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            inner = source / "depth_state_manifest.json"
            inner.write_bytes(b"authenticated-inner-manifest\n")
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            identity = _identity()
            cache.publish(identity, source)

            payload, receipt = cache.validated_payload_receipt(identity)
            row = next(
                value for value in receipt["files"]
                if value["path"] == inner.name
            )
            self.assertEqual(
                row["sha256"], artifact_cache.sha256_file(payload / inner.name)
            )
            self.assertEqual(
                receipt["payload_manifest_sha256"],
                artifact_cache.canonical_sha256(receipt["files"]),
            )

    def test_same_key_producers_must_publish_identical_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "value.bin").write_bytes(b"first")
            (second / "value.bin").write_bytes(b"second")
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            identity = _identity()
            cache.publish(identity, first)
            with self.assertRaisesRegex(
                    RuntimeError, "concurrent producer payload differs"):
                cache.publish(identity, second)

    def test_cache_entry_rejects_unmanifested_outer_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "value.bin").write_bytes(b"value")
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            identity = _identity()
            key = cache.publish(identity, source)
            (cache._entry(key) / "untracked.bin").write_bytes(b"untracked")
            with self.assertRaisesRegex(RuntimeError, "layout is invalid"):
                cache.materialize(identity, root / "destination")

    def test_miss_leaves_destination_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "destination"
            cache = artifact_cache.DirectoryArtifactCache(root / "cache")
            self.assertFalse(cache.materialize(_identity(), destination))
            self.assertFalse(destination.exists())

    def test_rejects_sealed_test_before_caller_source_access(self):
        with self.assertRaisesRegex(RuntimeError, "train/development only"):
            artifact_cache.require_working_split("test")
        self.assertEqual(
            artifact_cache.require_working_split("training"), "training"
        )

    def test_code_snapshot_fails_if_implementation_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "prepare.py"
            script.write_text("version = 1\n", encoding="utf-8")
            paths = {"preparer": script}
            expected = artifact_cache.code_identities(paths)
            artifact_cache.verify_code_identities(paths, expected)
            script.write_text("version = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                artifact_cache.verify_code_identities(paths, expected)

    def test_source_snapshot_fails_after_source_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            source.write_bytes(b"first source bytes")
            expected = artifact_cache.source_file_snapshot(source)
            artifact_cache.verify_source_file_snapshot(source, expected)
            source.write_bytes(b"second source bytes")
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                artifact_cache.verify_source_file_snapshot(source, expected)

    def test_cache_roots_must_be_disjoint_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            artifact_cache.require_disjoint_roots(
                cache, root / "source", root / "output"
            )
            for overlap in (cache, cache / "nested", root):
                with self.subTest(overlap=overlap), self.assertRaisesRegex(
                        RuntimeError, "must not overlap"):
                    artifact_cache.require_disjoint_roots(cache, overlap)

    def test_payload_root_rejects_junction_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "file.bin").write_bytes(b"payload")
            with mock.patch.object(
                    artifact_cache, "is_link_or_junction",
                    side_effect=lambda path: Path(path) == root):
                with self.assertRaisesRegex(RuntimeError, "plain directory"):
                    artifact_cache._payload_rows(root)


if __name__ == "__main__":
    unittest.main()
