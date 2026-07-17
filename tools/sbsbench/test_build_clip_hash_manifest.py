#!/usr/bin/env python3

import glob
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import build_clip_hash_manifest as clip_hashes


def legacy_sha1_dir(path):
    digest = hashlib.sha1()
    files = (
        glob.glob(os.path.join(path, "frame_*")) +
        glob.glob(os.path.join(path, "gt_depth", "frame_*")) +
        glob.glob(os.path.join(path, "gt_flow", "frame_*")) +
        glob.glob(os.path.join(path, "gt_right", "frame_*")) +
        glob.glob(os.path.join(path, "label_frames.json"))
    )
    for file_path in sorted(files):
        with open(file_path, "rb") as stream:
            digest.update(
                os.path.relpath(file_path, path).replace("\\", "/").encode()
            )
            digest.update(stream.read())
    try:
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as stream:
            meta = json.load(stream)
        semantic = {
            key: meta[key] for key in (
                "expected_flat", "gt_depth_kind", "dataset",
                "required_gt_depth", "required_gt_flow",
                "required_gt_stereo",
            ) if key in meta
        }
        digest.update(json.dumps(semantic, sort_keys=True).encode())
    except (OSError, ValueError):
        pass
    return digest.hexdigest()[:12]


class ClipHashManifestTests(unittest.TestCase):
    @staticmethod
    def make_clip(root, name, value=b"source"):
        clip = root / name
        (clip / "gt_depth").mkdir(parents=True)
        (clip / "frame_00000.png").write_bytes(value)
        (clip / "gt_depth" / "frame_00000.png").write_bytes(b"depth")
        (clip / "label_frames.json").write_text(
            json.dumps({"schema": 1, "frame_ids": [0]}), encoding="utf-8"
        )
        (clip / "meta.json").write_text(json.dumps({
            "name": "human-only",
            "description": "not identity",
            "dataset": "fixture",
            "required_gt_depth": True,
        }), encoding="utf-8")
        return clip

    def test_hash_matches_exact_legacy_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = self.make_clip(Path(directory), "shot")
            self.assertEqual(
                clip_hashes.sha1_dir(clip), legacy_sha1_dir(str(clip))
            )
            original = clip_hashes.sha1_dir(clip)
            meta = json.loads((clip / "meta.json").read_text(encoding="utf-8"))
            meta["description"] = "changed but non-semantic"
            (clip / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            self.assertEqual(clip_hashes.sha1_dir(clip), original)
            meta["required_gt_depth"] = False
            (clip / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            self.assertNotEqual(clip_hashes.sha1_dir(clip), original)

    def test_parallel_build_records_all_files_and_writes_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.make_clip(root, "first")
            self.make_clip(root, "second", b"other")
            manifest, output = clip_hashes.build_and_write(root, workers=2)

            self.assertTrue(output.is_file())
            self.assertEqual(sorted(manifest["clips"]), ["first", "second"])
            self.assertEqual(
                manifest["clips"]["first"]["clip_sha1"],
                legacy_sha1_dir(str(first)),
            )
            records = manifest["clips"]["first"]["files"]
            self.assertEqual(
                {record["path"] for record in records},
                {
                    "frame_00000.png",
                    "gt_depth/frame_00000.png",
                    "label_frames.json",
                },
            )
            for record in records + [manifest["clips"]["first"]["meta_file"]]:
                self.assertEqual(len(record["sha256"]), 64)
                self.assertGreaterEqual(record["size"], 0)
                self.assertIsInstance(record["mtime_ns"], int)
                self.assertTrue(os.path.isabs(record["real_path"]))
            self.assertFalse(list(root.glob(".clip_hash_manifest.json.*.partial")))

    def test_semantic_content_digest_ignores_time_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_clip(root, "first")
            first = clip_hashes.build_manifest(root, workers=1)
            second = clip_hashes.build_manifest(root, workers=1)
            second["created_utc"] = "2099-01-01T00:00:00+00:00"
            second["clips_root"]["path"] = "different-provenance"
            second["clips"]["first"]["clip_path"] = "different-provenance"
            second["clips"]["first"]["files"][0]["mtime_ns"] += 1
            self.assertEqual(
                first[clip_hashes.MANIFEST_CONTENT_SHA256_FIELD],
                clip_hashes.semantic_content_sha256(second),
            )

    def test_semantic_content_digest_changes_with_input_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = self.make_clip(root, "first")
            first = clip_hashes.build_manifest(root, workers=1)
            (clip / "frame_00000.png").write_bytes(b"changed")
            second = clip_hashes.build_manifest(root, workers=1)
            self.assertNotEqual(
                first[clip_hashes.MANIFEST_CONTENT_SHA256_FIELD],
                second[clip_hashes.MANIFEST_CONTENT_SHA256_FIELD],
            )

    def test_load_rejects_tampered_semantic_content_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_clip(root, "first")
            manifest, output = clip_hashes.build_and_write(root, workers=1)
            manifest["clips"]["first"]["files"][0]["sha256"] = "0" * 64
            clip_hashes.write_manifest_atomic(manifest, output)
            with self.assertRaisesRegex(
                    clip_hashes.ClipHashManifestError,
                    "semantic content digest is invalid"):
                clip_hashes.load_manifest(output)

    def test_cheap_and_full_verification_return_selected_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_clip(root, "first")
            self.make_clip(root, "second")
            manifest, output = clip_hashes.build_and_write(root, workers=1)
            expected = {"second": manifest["clips"]["second"]["clip_sha1"]}
            self.assertEqual(
                clip_hashes.verify_selected_clips(
                    output, root, ["second"], full=False
                ), expected
            )
            self.assertEqual(
                clip_hashes.verify_selected_clips(
                    output, root, ["second"], full=True
                ), expected
            )

    def test_manifest_fails_closed_for_missing_entry_or_changed_file_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = self.make_clip(root, "first")
            _manifest, output = clip_hashes.build_and_write(root, workers=1)
            self.make_clip(root, "second")
            with self.assertRaisesRegex(
                    clip_hashes.ClipHashManifestError, "no exact entry"):
                clip_hashes.verify_selected_clips(output, root, ["second"])
            (clip / "frame_00001.png").write_bytes(b"new")
            with self.assertRaisesRegex(
                    clip_hashes.ClipHashManifestError, "file set changed"):
                clip_hashes.verify_selected_clips(output, root, ["first"])

    def test_cheap_verification_detects_stat_and_resolved_path_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = self.make_clip(root, "first")
            _manifest, output = clip_hashes.build_and_write(root, workers=1)
            frame = clip / "frame_00000.png"
            frame.write_bytes(b"different-size")
            with self.assertRaisesRegex(
                    clip_hashes.ClipHashManifestError, "size changed"):
                clip_hashes.verify_selected_clips(output, root, ["first"])

            clip_hashes.build_and_write(root, workers=1)
            payload = json.loads(output.read_text(encoding="utf-8"))
            payload["clips"]["first"]["files"][0]["real_path"] = "wrong"
            clip_hashes.write_manifest_atomic(payload, output)
            with self.assertRaisesRegex(
                    clip_hashes.ClipHashManifestError, "real_path changed"):
                clip_hashes.verify_selected_clips(output, root, ["first"])

    def test_full_verification_catches_same_stat_content_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = self.make_clip(root, "first", b"source")
            _manifest, output = clip_hashes.build_and_write(root, workers=1)
            frame = clip / "frame_00000.png"
            previous = frame.stat()
            frame.write_bytes(b"tamper")
            os.utime(frame, ns=(previous.st_atime_ns, previous.st_mtime_ns))

            clip_hashes.verify_selected_clips(output, root, ["first"], full=False)
            with self.assertRaisesRegex(
                    clip_hashes.ClipHashManifestError, "content hash changed"):
                clip_hashes.verify_selected_clips(
                    output, root, ["first"], full=True
                )

    def test_manifest_is_bound_to_physical_root_not_only_text_path(self):
        with tempfile.TemporaryDirectory() as first_dir, \
                tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir)
            second = Path(second_dir)
            self.make_clip(first, "shot")
            self.make_clip(second, "shot")
            _manifest, output = clip_hashes.build_and_write(first, workers=1)
            copied = second / clip_hashes.MANIFEST_NAME
            copied.write_bytes(output.read_bytes())
            with self.assertRaisesRegex(
                    clip_hashes.ClipHashManifestError, "clips root changed"):
                clip_hashes.verify_selected_clips(copied, second, ["shot"])


if __name__ == "__main__":
    unittest.main()
