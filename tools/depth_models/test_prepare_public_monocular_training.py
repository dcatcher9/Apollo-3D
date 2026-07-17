#!/usr/bin/env python3

from __future__ import annotations

import io
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import zipfile

from PIL import Image

import prepare_public_monocular_training as public_mono


def png_bytes(value, width=8, height=6, grayscale=False):
    mode = "L" if grayscale else "RGB"
    color = value if grayscale else (value, value, value)
    image = Image.new(mode, (width, height), color)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    return encoded.getvalue()


class FakeResponse(io.BytesIO):
    def __init__(self, value, status, headers):
        super().__init__(value)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class PublicMonocularPreparationTests(unittest.TestCase):
    @staticmethod
    def _write_zip(path, members):
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            for name, data in members:
                member = zipfile.ZipInfo(name, date_time=(2025, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_STORED
                archive.writestr(member, data)

    @staticmethod
    def _archive_spec(root, relative, url, pattern, algorithm="sha256"):
        path = root / relative
        return {
            "relative_path": relative,
            "url": url,
            "bytes": path.stat().st_size,
            "hash_algorithm": algorithm,
            "hash": public_mono.hash_file(path, algorithm),
            "member_pattern": pattern,
        }

    def _fixture(self, root, omit_spring_disparity=None):
        downloads = root / "downloads"
        source_bytes = {}

        def rgb_members(prefix, sequences, digits):
            rows = []
            for sequence_index, sequence in enumerate(sequences):
                for frame in range(6):
                    member = prefix.format(
                        sequence=sequence,
                        frame=f"{frame:0{digits}d}",
                    )
                    data = png_bytes(20 + sequence_index * 40 + frame)
                    rows.append((member, data))
                    source_bytes[member] = data
            return rows

        reds_train = "reds-v1/train_sharp.zip"
        reds_test = "reds-v1/val_sharp.zip"
        spring_train = "spring-v1/train_frame_left.zip"
        spring_disp = "spring-v1/train_disp1_left.zip"
        spring_test = "spring-v1/test_frame_left.zip"
        self._write_zip(downloads / reds_train, rgb_members(
            "train/train_sharp/{sequence}/{frame}.png", ("000", "001"), 8
        ))
        self._write_zip(downloads / reds_test, rgb_members(
            "val/val_sharp/{sequence}/{frame}.png", ("010",), 8
        ))
        self._write_zip(downloads / spring_train, rgb_members(
            "spring/train/{sequence}/frame_left/frame_left_{frame}.png",
            ("0001", "0002"), 4,
        ))
        disparity_members = []
        for sequence in ("0001", "0002"):
            for frame in range(6):
                if omit_spring_disparity == (sequence, frame):
                    continue
                member = (
                    f"spring/train/{sequence}/disp1_left/"
                    f"disp1_left_{frame:04d}.dsp5"
                )
                data = f"dsp5:{sequence}:{frame}".encode("ascii")
                disparity_members.append((member, data))
                source_bytes[member] = data
        self._write_zip(downloads / spring_disp, disparity_members)
        self._write_zip(downloads / spring_test, rgb_members(
            "spring/test/{sequence}/frame_left/frame_left_{frame}.png",
            ("0010",), 4,
        ))

        manifest = {
            "schema": 1,
            "purpose": "synthetic public mono fixture",
            "sources": {
                "reds": {
                    "version": "reds-sharp-v1",
                    "dataset": "Synthetic REDS",
                    "domain": "real_dynamic_scenes",
                    "source_group": "synthetic_reds",
                    "homepage": "https://example.test/reds",
                    "license": "CC BY 4.0",
                    "license_url": "https://example.test/license",
                    "context_fps": 24.0,
                    "global_policy_weight": 1.0,
                    "development_count": 1,
                    "split_salt": "synthetic-reds",
                    "expected_sequence_counts": {
                        "train_rgb": 2, "test_rgb": 1,
                    },
                    "archives": {
                        "train_rgb": self._archive_spec(
                            downloads, reds_train, "https://example.test/reds-train",
                            (r"^train/train_sharp/(?P<sequence>[0-9]{3})/"
                             r"(?P<frame>[0-9]{8})[.]png$")
                        ),
                        "test_rgb": self._archive_spec(
                            downloads, reds_test, "https://example.test/reds-test",
                            (r"^val/val_sharp/(?P<sequence>[0-9]{3})/"
                             r"(?P<frame>[0-9]{8})[.]png$")
                        ),
                    },
                },
                "spring": {
                    "version": "spring-left-v1",
                    "dataset": "Synthetic Spring",
                    "domain": "cinematic_animation",
                    "source_group": "synthetic_spring",
                    "homepage": "https://example.test/spring",
                    "license": "CC BY 4.0",
                    "license_url": "https://example.test/license",
                    "context_fps": 24.0,
                    "global_policy_weight": 1.0 / 3.0,
                    "development_count": 1,
                    "split_salt": "synthetic-spring",
                    "expected_sequence_counts": {
                        "train_rgb": 2, "test_rgb": 1,
                    },
                    "archives": {
                        "train_rgb": self._archive_spec(
                            downloads, spring_train,
                            "https://example.test/spring-train",
                            (r"^spring/train/(?P<sequence>[0-9]{4})/frame_left/"
                             r"frame_left_(?P<frame>[0-9]{4})[.]png$"),
                            "md5",
                        ),
                        "train_disparity": {
                            **self._archive_spec(
                                downloads, spring_disp,
                                "https://example.test/spring-disp",
                                (r"^spring/train/(?P<sequence>[0-9]{4})/disp1_left/"
                                 r"disp1_left_(?P<frame>[0-9]{4})[.]dsp5$"),
                                "md5",
                            ),
                            "admission": "sparse_auxiliary_only",
                        },
                        "test_rgb": self._archive_spec(
                            downloads, spring_test,
                            "https://example.test/spring-test",
                            (r"^spring/test/(?P<sequence>[0-9]{4})/frame_left/"
                             r"frame_left_(?P<frame>[0-9]{4})[.]png$"),
                            "md5",
                        ),
                    },
                },
            },
        }
        sources = root / "public_sources.json"
        sources.write_text(json.dumps(manifest), encoding="utf-8")
        return sources, downloads, source_bytes

    def _run(self, root, **kwargs):
        sources, downloads, source_bytes = self._fixture(
            root, kwargs.get("omit_spring_disparity")
        )
        prepared = root / "prepared"
        catalog = root / "metadata" / "catalog.json"
        active = root / "metadata" / "active.json"
        result = public_mono.prepare_public_sources(
            sources, downloads, prepared, catalog, active,
            workers=2, download_workers=2,
            overwrite_stale=kwargs.get("overwrite_stale", False),
        )
        return sources, downloads, prepared, catalog, active, source_bytes, result

    def test_prepares_six_byte_exact_full_cadence_aggregate_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (_sources, _downloads, prepared, catalog, active,
             source_bytes, result) = self._run(root)

            self.assertEqual(len(result["dataset_manifests"]), 6)
            self.assertEqual(result["context_frames"], 36)
            self.assertEqual(result["label_frames"], 30)
            self.assertEqual(len(json.loads(
                catalog.read_text(encoding="utf-8")
            )["sources"]), 6)
            active_payload = json.loads(active.read_text(encoding="utf-8"))
            self.assertEqual(active_payload["totals"]["productions"], 6)
            self.assertEqual(len(active_payload["split_productions"]["test"]), 2)

            for source in ("reds", "spring"):
                for split in public_mono.SPLITS:
                    split_root = prepared / f"{source}-mono-v1" / split
                    dataset = json.loads((
                        split_root / public_mono.DATASET_MANIFEST
                    ).read_text(encoding="utf-8"))
                    self.assertEqual(dataset["source_container"],
                                     "image-sequence-archives")
                    self.assertEqual(dataset["label_frame_count"], 5)
                    self.assertEqual(
                        dataset["video_sha256"],
                        public_mono.hash_file(
                            split_root / public_mono.SOURCE_SEQUENCE_MANIFEST
                        ),
                    )
                    self.assertFalse(any(split_root.glob(".*.partial")))
                    clip = split_root / dataset["sequences"][0]["clip"]
                    labels = json.loads((clip / "label_frames.json").read_text(
                        encoding="utf-8"
                    ))
                    self.assertEqual(labels["frame_ids"], [0, 1, 3, 4, 5])
                    record = json.loads((
                        clip / public_mono.SEQUENCE_RECORD
                    ).read_text(encoding="utf-8"))
                    for frame in record["rgb_frames"]:
                        self.assertEqual(
                            (clip / frame["output"]).read_bytes(),
                            source_bytes[frame["member"]],
                        )
                    if source == "spring" and split != "test":
                        self.assertEqual(len(record["disparity_frames"]), 5)
                        self.assertEqual(
                            len(list((clip / "gt_disparity").glob("*.dsp5"))), 5
                        )
                    else:
                        self.assertFalse((clip / "gt_disparity").exists())

    def test_published_split_is_reused_and_stale_bytes_need_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = self._run(root)
            prepared = values[2]
            clip = next((prepared / "reds-mono-v1" / "training").glob(
                "reds_training_*"
            ))
            frame = clip / "frame_00000.png"
            original = frame.read_bytes()

            # A complete valid result is idempotently reused.
            self._run(root)
            self.assertEqual(frame.read_bytes(), original)

            tampered = bytearray(original)
            tampered[-1] ^= 1
            frame.write_bytes(tampered)
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                self._run(root)
            self._run(root, overwrite_stale=True)
            self.assertEqual(frame.read_bytes(), original)

    def test_missing_sparse_spring_label_disparity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Frame 0 is always one of the five label endpoints.
            with self.assertRaisesRegex(
                    RuntimeError, "disparity (missing label|frames differ)"):
                self._run(root, omit_spring_disparity=("0001", 0))
            self.assertFalse(
                (root / "prepared" / "spring-mono-v1" / "training").exists()
            )

    def test_prepare_only_can_stage_spring_without_accessing_reds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, downloads, _source_bytes = self._fixture(root)
            (downloads / "reds-v1" / "train_sharp.zip").unlink()
            (downloads / "reds-v1" / "val_sharp.zip").unlink()
            prepared = root / "prepared"

            result = public_mono.prepare_public_sources(
                sources, downloads, prepared, None, None,
                workers=2, download_workers=2,
                selected_sources=("spring",), generate_metadata=False,
            )

            self.assertEqual(len(result["dataset_manifests"]), 3)
            self.assertIsNone(result["catalog"])
            self.assertTrue((prepared / "spring-mono-v1" / "test").is_dir())
            self.assertFalse((prepared / "reds-mono-v1").exists())

    def test_resume_download_appends_only_requested_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"0123456789abcdef"
            spec = {
                "relative": Path("dataset/archive.zip"),
                "url": "https://example.test/archive.zip",
                "bytes": len(content),
                "hash_algorithm": "sha256",
                "hash": hashlib.sha256(content).hexdigest(),
            }
            partial = root / "dataset" / "archive.zip.partial"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(content[:7])

            def opener(request):
                self.assertEqual(request.headers["Range"], "bytes=7-")
                return FakeResponse(
                    content[7:], 206,
                    {"Content-Range": f"bytes 7-{len(content)-1}/{len(content)}"},
                )

            result = public_mono.download_archive(root, spec, opener)
            self.assertEqual(result.read_bytes(), content)
            self.assertFalse(partial.exists())

    def test_incomplete_final_archive_is_adopted_as_resume_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"complete pinned archive"
            spec = {
                "relative": Path("dataset/archive.zip"),
                "url": "https://example.test/archive.zip",
                "bytes": len(content),
                "hash_algorithm": "sha256",
                "hash": hashlib.sha256(content).hexdigest(),
            }
            destination = root / spec["relative"]
            destination.parent.mkdir(parents=True)
            destination.write_bytes(content[:5])

            def opener(request):
                self.assertEqual(request.headers["Range"], "bytes=5-")
                return FakeResponse(
                    content[5:], 206,
                    {"Content-Range": f"bytes 5-{len(content)-1}/{len(content)}"},
                )

            result = public_mono.download_archive(root, spec, opener)
            self.assertEqual(result.read_bytes(), content)

    def test_zip_gap_and_non_rgb8_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "gap.zip"
            self._write_zip(archive_path, [
                ("x/000/0000.png", png_bytes(1)),
                ("x/000/0002.png", png_bytes(2)),
            ])
            spec = {
                "role": "rgb",
                "pattern": __import__("re").compile(
                    r"^x/(?P<sequence>[0-9]{3})/"
                    r"(?P<frame>[0-9]{4})[.]png$"
                ),
            }
            with self.assertRaisesRegex(RuntimeError, "frame gaps"):
                public_mono.index_archive(archive_path, spec)
            with self.assertRaisesRegex(RuntimeError, "RGB8"):
                public_mono._png_contract(png_bytes(3, grayscale=True))

    def test_rejects_overlapping_download_and_prepared_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources.json"
            sources.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "roots overlap"):
                public_mono._validate_roots(
                    sources, root / "data", root / "data" / "prepared",
                    root / "catalog.json", root / "active.json",
                )


if __name__ == "__main__":
    unittest.main()
