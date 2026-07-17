#!/usr/bin/env python3

import csv
import hashlib
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import fetch_chug_hdr_references as chug


class FakeResponse(io.BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class ChugHdrReferenceTests(unittest.TestCase):
    @staticmethod
    def raw_row(video_id="0" * 32, reference="1", orientation="Landscape", frame_rate="30.0"):
        width, height = (1920, 1080) if orientation == "Landscape" else (1080, 1920)
        return {
            "Video": video_id,
            "mos_j": "50",
            "sos_j": "2",
            "ref": reference,
            "name": "1080p_ref_#source.mp4" if reference == "1" else "720p_bad_source.mp4",
            "bitladder": "1080p_ref_" if reference == "1" else "720p_2mbps_",
            "resolution": "ref" if reference == "1" else "720p",
            "bitrate": "ref" if reference == "1" else "2mbps",
            "orientation": orientation,
            "framerate": frame_rate,
            "content_name": f"source_{video_id}.mp4",
            "height": str(height),
            "width": str(width),
        }

    @staticmethod
    def row(index, orientation="Landscape", frame_rate=30.0, content_name=None):
        video_id = f"{index:032x}"
        return {
            "video_id": video_id,
            "content_name": content_name or f"source_{index}.mp4",
            "orientation": orientation,
            "catalog_frame_rate": frame_rate,
            "probed_frame_rate": frame_rate,
            "width": 1920 if orientation == "Landscape" else 1080,
            "height": 1080 if orientation == "Landscape" else 1920,
            "csv_row": index + 2,
        }

    @staticmethod
    def selection_contract():
        return {
            "hash_salt": "test-selection-v1",
            "framerate_buckets": [
                {"name": "24_25", "max_exclusive": 27.5},
                {"name": "30", "max_exclusive": 45.0},
                {"name": "60", "max_exclusive": 90.0},
                {"name": "120_plus"},
            ],
        }

    @staticmethod
    def split_contract():
        return {
            "hash_salt": "test-split-v1",
            "weights": {"training": 6, "development": 1, "test": 1},
        }

    @staticmethod
    def probe_payload(**overrides):
        stream = {
            "codec_type": "video",
            "codec_name": "hevc",
            "pix_fmt": "yuv420p10le",
            "color_range": "tv",
            "color_space": "bt2020nc",
            "color_transfer": "smpte2084",
            "color_primaries": "bt2020",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "30000/1001",
            "duration": "10.0",
        }
        stream.update(overrides)
        return {
            "streams": [stream],
            "format": {"format_name": "mov,mp4", "duration": "10.0", "size": "100"},
        }

    @staticmethod
    def color_contract():
        return {
            "codec": "hevc",
            "minimum_bit_depth": 10,
            "color_range": "tv",
            "color_primaries": "bt2020",
            "color_space": "bt2020nc",
            "color_transfer": "smpte2084",
            "minimum_duration_seconds": 5.0,
            "maximum_duration_seconds": 15.0,
        }

    def candidate_manifest(self):
        return {
            "id": "test-chug",
            "repository_commit": "1" * 40,
            "videos": {
                "url_template": "https://example.test/videos/{video_id}.mp4",
            },
            "native_color_contract": self.color_contract(),
        }

    def write_candidate_cache(self, cache, row, payload, manifest=None, version="ffprobe test"):
        manifest = manifest or self.candidate_manifest()
        binding = chug.candidate_probe_binding(row, manifest, version)
        chug.atomic_write_json(
            cache,
            chug.candidate_probe_cache_payload(binding, payload),
        )

    def write_csv(self, root, rows):
        path = root / "chug.csv"
        fieldnames = list(self.raw_row().keys())
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_repository_source_manifest_is_valid_and_pinned(self):
        path = Path(chug.__file__).with_name("chug_hdr_reference_sources.json")
        manifest = chug.load_source_manifest(path)
        self.assertEqual(manifest["repository_commit"], "6e4cc0631ea7faa731992f1fed3edc3efce53593")
        self.assertEqual(manifest["license"], "CC BY-NC-SA 4.0")
        self.assertEqual(manifest["selection"]["default_limit"], 96)

    def test_csv_excludes_degraded_rows_and_authenticates_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.raw_row("1" * 32)
            degraded = self.raw_row("2" * 32, reference="0")
            degraded["content_name"] = first["content_name"]
            path = self.write_csv(root, [first, degraded])
            rows, stats = chug.load_chug_references(path, {
                "total_rows": 2,
                "reference_rows": 1,
                "degraded_rows": 1,
            })
            self.assertEqual([row["video_id"] for row in rows], ["1" * 32])
            self.assertEqual(stats["degraded_rows_excluded"], 1)

    def test_explicit_degraded_row_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "degraded/non-reference"):
            chug.parse_reference_row(self.raw_row(reference="0"), 7)

    def test_reference_row_rejects_degraded_ladder_labels(self):
        row = self.raw_row()
        row["bitladder"] = "1080p_10mbps_"
        with self.assertRaisesRegex(RuntimeError, "degraded bit-ladder"):
            chug.parse_reference_row(row, 2)

    def test_hash_stratification_is_order_independent_and_covers_strata(self):
        rows = []
        index = 1
        for orientation in ("Landscape", "Portrait"):
            for frame_rate in (24.0, 30.0, 59.94, 120.0):
                for _ in range(5):
                    rows.append(self.row(index, orientation, frame_rate))
                    index += 1
        first = chug.select_hash_stratified(rows, 16, self.selection_contract())
        second = chug.select_hash_stratified(list(reversed(rows)), 16, self.selection_contract())
        self.assertEqual(
            [row["video_id"] for row in first],
            [row["video_id"] for row in second],
        )
        self.assertEqual(len({row["selection_stratum"] for row in first}), 8)
        self.assertEqual(len(first), 16)

    def test_zero_limit_selects_every_reference(self):
        rows = [self.row(index) for index in range(1, 8)]
        selected = chug.select_hash_stratified(rows, 0, self.selection_contract())
        self.assertEqual(len(selected), len(rows))

    def test_selection_uses_probed_not_catalog_frame_rate(self):
        row = self.row(1, frame_rate=120.0)
        row["catalog_frame_rate"] = 27.31
        selected = chug.select_hash_stratified([row], 1, self.selection_contract())
        self.assertEqual(selected[0]["frame_rate_bucket"], "120_plus")

    def test_contract_rejection_is_deterministically_backfilled(self):
        rows = [self.row(index, frame_rate=30.0) for index in range(1, 12)]
        original = chug.select_hash_stratified(rows, 6, self.selection_contract())
        rejected_id = original[0]["video_id"]
        valid = [row for row in rows if row["video_id"] != rejected_id]
        first = chug.select_hash_stratified(valid, 6, self.selection_contract())
        second = chug.select_hash_stratified(list(reversed(valid)), 6, self.selection_contract())
        self.assertEqual(len(first), 6)
        self.assertNotIn(rejected_id, {row["video_id"] for row in first})
        self.assertEqual(
            [row["video_id"] for row in first],
            [row["video_id"] for row in second],
        )

    def test_exact_content_split_is_72_12_12_for_default_pool(self):
        rows = [self.row(index) for index in range(1, 97)]
        selected = chug.select_hash_stratified(rows, 96, self.selection_contract())
        assigned = chug.assign_exact_content_splits(selected, self.split_contract())
        self.assertEqual(
            chug.count_by(assigned, "split"),
            {"development": 12, "test": 12, "training": 72},
        )
        self.assertEqual(len({row["content_id"] for row in assigned}), 96)

    def test_exact_content_split_rejects_duplicate_content_identity(self):
        rows = [
            {**self.row(1, content_name="same.mp4"), "selection_hash": "1"},
            {**self.row(2, content_name="same.mp4"), "selection_hash": "2"},
        ]
        with self.assertRaisesRegex(RuntimeError, "duplicate selected content"):
            chug.assign_exact_content_splits(rows, self.split_contract())

    def test_capture_group_manifest_collapses_to_one_deterministic_representative(self):
        candidates = [self.row(index) for index in range(1, 98)]
        candidate_ids = [row["video_id"] for row in candidates]
        groups = [{
            "capture_group_id": "group-pair",
            "members": sorted(candidate_ids[:2]),
            "member_count": 2,
        }]
        groups.extend({
            "capture_group_id": f"group-{video_id}",
            "members": [video_id],
            "member_count": 1,
        } for video_id in candidate_ids[2:])
        semantic = {
            "schema": 1,
            "dataset": "test-chug",
            "repository_commit": "1" * 40,
            "candidate_probe_semantic_sha256": "2" * 64,
            "valid_reference_count": len(candidates),
            "visual_identities": [
                {"video_id": video_id} for video_id in candidate_ids
            ],
            "capture_groups": groups,
            "evidence": {
                "calibration_pairs": [],
                "review_required_pairs": [],
            },
            "status": "decision_ready",
        }
        payload = {**semantic, "semantic_sha256": chug.canonical_sha256(semantic)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture_group_manifest.json"
            chug.atomic_write_json(path, payload)
            loaded = chug.load_capture_group_manifest(
                path,
                {"id": "test-chug", "repository_commit": "1" * 40},
                "2" * 64,
                candidates,
            )
        first = chug.collapse_capture_group_representatives(
            candidates,
            loaded,
            self.selection_contract(),
        )
        second = chug.collapse_capture_group_representatives(
            list(reversed(candidates)),
            loaded,
            self.selection_contract(),
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 96)
        self.assertEqual(len({row["capture_group_id"] for row in first}), 96)
        selected = chug.prepare_selection(
            first,
            {"selection": self.selection_contract(), "split": self.split_contract()},
            96,
        )
        self.assertEqual(
            chug.count_by(selected, "split"),
            {"development": 12, "test": 12, "training": 72},
        )

    def test_capture_group_manifest_rejects_unresolved_review(self):
        candidates = [self.row(1)]
        semantic = {
            "schema": 1,
            "dataset": "test-chug",
            "repository_commit": "1" * 40,
            "candidate_probe_semantic_sha256": "2" * 64,
            "valid_reference_count": 1,
            "visual_identities": [{"video_id": candidates[0]["video_id"]}],
            "capture_groups": [{
                "capture_group_id": "group-1",
                "members": [candidates[0]["video_id"]],
                "member_count": 1,
            }],
            "evidence": {
                "calibration_pairs": [],
                "review_required_pairs": [{"first": "a", "second": "b"}],
            },
            "status": "review_required",
        }
        payload = {**semantic, "semantic_sha256": chug.canonical_sha256(semantic)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture_group_manifest.json"
            chug.atomic_write_json(path, payload)
            with self.assertRaisesRegex(RuntimeError, "not decision-ready"):
                chug.load_capture_group_manifest(
                    path,
                    {"id": "test-chug", "repository_commit": "1" * 40},
                    "2" * 64,
                    candidates,
                )

    def test_existing_selection_is_reassigned_as_exact_indivisible_groups(self):
        rows = chug.select_hash_stratified(
            [self.row(index) for index in range(1, 97)],
            96,
            self.selection_contract(),
        )
        for index, row in enumerate(rows):
            row["capture_group_id"] = (
                f"paired-{index // 2}" if index < 20 else f"single-{index}"
            )
        assigned = chug.assign_exact_capture_group_splits(rows, self.split_contract())
        self.assertEqual(
            chug.count_by(assigned, "split"),
            {"development": 12, "test": 12, "training": 72},
        )
        group_splits = {}
        for row in assigned:
            group_splits.setdefault(row["capture_group_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))

    def test_existing_selection_fails_when_groups_cannot_meet_exact_quota(self):
        rows = chug.select_hash_stratified(
            [self.row(index) for index in range(1, 97)],
            96,
            self.selection_contract(),
        )
        for index, row in enumerate(rows):
            row["capture_group_id"] = f"group-{index // 32}"
        with self.assertRaisesRegex(RuntimeError, "cannot satisfy"):
            chug.assign_exact_capture_group_splits(rows, self.split_contract())

    def test_valid_native_pq_probe_is_accepted_without_frame_decode(self):
        audit = chug.validate_native_pq_probe(
            self.probe_payload(),
            self.row(1),
            self.color_contract(),
        )
        self.assertEqual(audit["bit_depth"], 10)
        self.assertEqual(audit["color_transfer"], "smpte2084")
        self.assertEqual(audit["frame_decode"], "not_performed")

    def test_native_pq_probe_rejects_every_required_contract_violation(self):
        cases = {
            "codec": ({"codec_name": "h264"}, "not HEVC"),
            "depth": ({"pix_fmt": "yuv420p"}, "yuv420p10"),
            "range": ({"color_range": "pc"}, "color_range"),
            "primaries": ({"color_primaries": "bt709"}, "color_primaries"),
            "matrix": ({"color_space": "bt2020c"}, "color_space"),
            "transfer": ({"color_transfer": "arib-std-b67"}, "color_transfer"),
            "short": ({"duration": "4.99"}, "outside"),
            "long": ({"duration": "15.01"}, "outside"),
        }
        for name, (override, pattern) in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, pattern):
                    chug.validate_native_pq_probe(
                        self.probe_payload(**override),
                        self.row(1),
                        self.color_contract(),
                    )

    def test_candidate_probe_reuses_and_revalidates_valid_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = self.row(1)
            cache = output / "candidate_ffprobe" / f"{row['video_id']}.json"
            self.write_candidate_cache(cache, row, self.probe_payload())
            with mock.patch.object(chug, "probe_video") as probe:
                candidate, rejection = chug.probe_reference_candidate(
                    row,
                    output,
                    self.candidate_manifest(),
                    Path("ffprobe"),
                    "ffprobe test",
                )
            probe.assert_not_called()
            self.assertIsNone(rejection)
            self.assertEqual(candidate["probe_source"], "cache")
            self.assertEqual(candidate["candidate_audit"]["color_range"], "tv")

    def test_candidate_probe_cached_contract_failure_is_not_hidden_by_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = self.row(1)
            cache = output / "candidate_ffprobe" / f"{row['video_id']}.json"
            self.write_candidate_cache(cache, row, self.probe_payload(color_range="pc"))
            with mock.patch.object(chug, "probe_video") as probe:
                candidate, rejection = chug.probe_reference_candidate(
                    row,
                    output,
                    self.candidate_manifest(),
                    Path("ffprobe"),
                    "ffprobe test",
                )
            probe.assert_not_called()
            self.assertIsNone(candidate)
            self.assertEqual(rejection["probe_source"], "cache")
            self.assertIn("color_range", rejection["reason"])

    def test_candidate_probe_replaces_corrupt_cache_from_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = self.row(1)
            cache = output / "candidate_ffprobe" / f"{row['video_id']}.json"
            cache.parent.mkdir(parents=True)
            cache.write_text("{not-json", encoding="utf-8")
            payload = self.probe_payload()
            with mock.patch.object(chug, "probe_video", return_value=payload) as probe:
                candidate, rejection = chug.probe_reference_candidate(
                    row,
                    output,
                    self.candidate_manifest(),
                    Path("ffprobe"),
                    "ffprobe test",
                )
            probe.assert_called_once()
            self.assertIsNone(rejection)
            self.assertEqual(candidate["probe_source"], "remote")
            cached = json.loads(cache.read_text(encoding="utf-8"))
            self.assertEqual(cached["probe"], payload)
            self.assertEqual(cached["binding"]["repository_commit"], "1" * 40)

    def test_candidate_probe_refresh_ignores_existing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = self.row(1)
            cache = output / "candidate_ffprobe" / f"{row['video_id']}.json"
            self.write_candidate_cache(cache, row, self.probe_payload(color_range="pc"))
            payload = self.probe_payload()
            with mock.patch.object(chug, "probe_video", return_value=payload) as probe:
                candidate, rejection = chug.probe_reference_candidate(
                    row,
                    output,
                    self.candidate_manifest(),
                    Path("ffprobe"),
                    "ffprobe test",
                    refresh_probe=True,
                )
            probe.assert_called_once()
            self.assertIsNone(rejection)
            self.assertEqual(candidate["probe_source"], "remote")
            self.assertEqual(json.loads(cache.read_text(encoding="utf-8"))["probe"], payload)

    def test_candidate_probe_replaces_structurally_corrupt_json_object(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = self.row(1)
            cache = output / "candidate_ffprobe" / f"{row['video_id']}.json"
            chug.atomic_write_json(cache, {})
            payload = self.probe_payload()
            with mock.patch.object(chug, "probe_video", return_value=payload) as probe:
                candidate, rejection = chug.probe_reference_candidate(
                    row,
                    output,
                    self.candidate_manifest(),
                    Path("ffprobe"),
                    "ffprobe test",
                )
            probe.assert_called_once()
            self.assertIsNone(rejection)
            self.assertEqual(candidate["probe_cache_status"], "unbound_or_incompatible")
            self.assertEqual(json.loads(cache.read_text(encoding="utf-8"))["probe"], payload)

    def test_candidate_probe_refetches_cache_bound_to_an_old_source(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = self.row(1)
            cache = output / "candidate_ffprobe" / f"{row['video_id']}.json"
            old_manifest = self.candidate_manifest()
            old_manifest["repository_commit"] = "0" * 40
            self.write_candidate_cache(cache, row, self.probe_payload(), manifest=old_manifest)
            payload = self.probe_payload()
            with mock.patch.object(chug, "probe_video", return_value=payload) as probe:
                candidate, rejection = chug.probe_reference_candidate(
                    row,
                    output,
                    self.candidate_manifest(),
                    Path("ffprobe"),
                    "ffprobe test",
                )
            probe.assert_called_once()
            self.assertIsNone(rejection)
            self.assertEqual(candidate["probe_cache_status"], "binding_mismatch")
            cached = json.loads(cache.read_text(encoding="utf-8"))
            self.assertEqual(cached["binding"]["repository_commit"], "1" * 40)

    def test_candidate_probe_refetches_cache_from_an_old_ffprobe_version(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = self.row(1)
            cache = output / "candidate_ffprobe" / f"{row['video_id']}.json"
            self.write_candidate_cache(cache, row, self.probe_payload(), version="ffprobe old")
            payload = self.probe_payload()
            with mock.patch.object(chug, "probe_video", return_value=payload) as probe:
                candidate, rejection = chug.probe_reference_candidate(
                    row,
                    output,
                    self.candidate_manifest(),
                    Path("ffprobe"),
                    "ffprobe test",
                )
            probe.assert_called_once()
            self.assertIsNone(rejection)
            self.assertEqual(candidate["probe_cache_status"], "binding_mismatch")
            cached = json.loads(cache.read_text(encoding="utf-8"))
            self.assertEqual(cached["binding"]["producer_ffprobe_version"], "ffprobe test")

    def test_candidate_semantic_identity_ignores_cache_execution_path(self):
        manifest = self.candidate_manifest()
        candidate = {
            **self.row(1),
            "candidate_audit": {"frame_rate": 30.0},
            "probe_source": "remote",
            "probe_cache_status": "missing",
        }
        cached = {
            **candidate,
            "probe_source": "cache",
            "probe_cache_status": "cache_hit",
        }
        first = chug.build_candidate_semantic_payload(
            manifest,
            "ffprobe test",
            1,
            [candidate],
            [],
        )
        second = chug.build_candidate_semantic_payload(
            manifest,
            "ffprobe test",
            1,
            [cached],
            [],
        )
        self.assertEqual(first, second)
        self.assertEqual(chug.canonical_sha256(first), chug.canonical_sha256(second))

    def test_resume_appends_only_a_valid_content_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "clip.mp4"
            partial = root / "clip.mp4.part"
            marker = root / "clip.mp4.part.url"
            partial.write_bytes(b"abc")
            marker.write_text("https://example.test/clip.mp4\n", encoding="utf-8")
            response = FakeResponse(
                b"def",
                status=206,
                headers={"Content-Length": "3", "Content-Range": "bytes 3-5/6"},
            )
            with mock.patch.object(chug.urllib.request, "urlopen", return_value=response) as opener:
                receipt = chug.download_http("https://example.test/clip.mp4", destination)
            self.assertEqual(destination.read_bytes(), b"abcdef")
            self.assertEqual(receipt["status"], "downloaded_resumed")
            self.assertEqual(opener.call_args.args[0].headers["Range"], "bytes=3-")

    def test_server_ignoring_range_restarts_instead_of_duplicating(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "clip.mp4"
            (root / "clip.mp4.part").write_bytes(b"abc")
            (root / "clip.mp4.part.url").write_text(
                "https://example.test/clip.mp4\n",
                encoding="utf-8",
            )
            response = FakeResponse(b"abcdef", status=200, headers={"Content-Length": "6"})
            with mock.patch.object(chug.urllib.request, "urlopen", return_value=response):
                receipt = chug.download_http("https://example.test/clip.mp4", destination)
            self.assertEqual(destination.read_bytes(), b"abcdef")
            self.assertEqual(receipt["status"], "downloaded_restarted")

    def test_locked_download_rejects_wrong_hash_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "chug.csv"
            payload = b"wrong"
            response = FakeResponse(payload, headers={"Content-Length": str(len(payload))})
            expected = hashlib.sha256(b"right").hexdigest()
            with mock.patch.object(chug.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    chug.download_http(
                        "https://example.test/chug.csv",
                        destination,
                        expected_bytes=len(payload),
                        expected_sha256=expected,
                    )
            self.assertFalse(destination.exists())
            self.assertFalse((Path(directory) / "chug.csv.part").exists())

    def test_parallel_downloads_never_exceed_the_configured_bound(self):
        active = 0
        peak = 0
        lock = threading.Lock()

        def fake_download(row, output, manifest, timeout, audit_only):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return {"video_id": row["video_id"], "bytes": 1, "sha256": "0" * 64}

        rows = [self.row(index) for index in range(1, 10)]
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(chug, "download_video", side_effect=fake_download):
                completed, rejected = chug.download_selected(
                    rows,
                    Path(directory),
                    {},
                    jobs=3,
                )
        self.assertEqual(len(completed), len(rows))
        self.assertFalse(rejected)
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, 3)

    def test_audit_invokes_ffprobe_for_every_downloaded_clip(self):
        rows = [self.row(index) for index in range(1, 4)]
        downloads = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for row in rows:
                row["candidate_audit"] = chug.validate_native_pq_probe(
                    self.probe_payload(), row, self.color_contract()
                )
                path = root / f"{row['video_id']}.mp4"
                downloads.append({
                    "video_id": row["video_id"],
                    "path": str(path),
                    "relative_path": path.name,
                    "url": "https://example.test",
                    "bytes": 100,
                    "sha256": "0" * 64,
                    "status": "test",
                })
            manifest = {"native_color_contract": self.color_contract()}
            with mock.patch.object(
                    chug,
                    "probe_video",
                    side_effect=[self.probe_payload() for _ in rows],
            ) as probe:
                accepted, rejected = chug.audit_downloads(
                    downloads,
                    rows,
                    root,
                    manifest,
                    Path("ffprobe"),
                )
            self.assertEqual(probe.call_count, len(rows))
            self.assertEqual(len(accepted), len(rows))
            self.assertFalse(rejected)


if __name__ == "__main__":
    unittest.main()
