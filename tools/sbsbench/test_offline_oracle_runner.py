import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from tools.sbsbench import run_offline_oracles as runner


class DummySeaModel:
    load_count = 0

    def __init__(self, repo, checkpoint, config, device):
        type(self).load_count += 1
        self.repo = repo
        self.checkpoint = checkpoint
        self.config = config
        self.device = device or "cpu"


class DummyIsqoeModel:
    load_count = 0

    def __init__(self, repo, checkpoint, device):
        type(self).load_count += 1
        self.repo = repo
        self.checkpoint = checkpoint
        self.device = device or "cpu"
        self.provenance = {"implementation": "unit-test"}

    def evaluate_path(self, path):
        return {
            "schema": 1,
            "oracle": "apple-isqoe",
            "status": "ok",
            "training_label_eligible": False,
            "path": str(path.resolve()),
            "metrics": {"isqoe_mean_score": 0.2},
        }


class OfflineOracleRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.clips_root = self.root / "clips"
        self.dependencies = self.root / "dependencies"
        self.run_dir.mkdir()
        self.clips_root.mkdir()
        self.dependencies.mkdir()
        for name in ("clip_a", "clip_b"):
            output = self.run_dir / name
            source = self.clips_root / name
            output.mkdir()
            source.mkdir()
            rgb = np.full((8, 16, 3), 80, dtype=np.uint8)
            Image.fromarray(np.concatenate((rgb, rgb), axis=1)).save(
                output / "sbs_00000.png")
            source_u = np.broadcast_to(
                (np.arange(16, dtype=np.float32) + 0.5)[None, :] / 16.0,
                (8, 16),
            )
            np.concatenate((source_u, source_u), axis=1).astype(
                np.float32).tofile(output / "warp_map_00000.f32")
            (output / "warp_map_shape.json").write_text(json.dumps({
                "width": 32,
                "height": 8,
                "eye_width": 16,
                "eye_height": 8,
                "source_width": 16,
                "source_height": 8,
                "content_scale_x": 1.0,
                "content_scale_y": 1.0,
            }), encoding="utf-8")
            Image.fromarray(rgb).save(source / "frame_00000.png")
            Image.fromarray(rgb).save(source / "frame_00001.png")
        results = {
            "meta": {"run_name": "unit-run", "eval_schema": 30,
                     "clips_root": str(self.clips_root)},
            "clips": {"clip_a": {}, "clip_b": {}},
        }
        (self.run_dir / "results.json").write_text(
            json.dumps(results), encoding="utf-8")
        self.raft_repo = self.dependencies / "raft"
        self.sea_repo = self.dependencies / "sea"
        self.raft_repo.mkdir()
        self.sea_repo.mkdir()
        self.raft_checkpoint = self.dependencies / "raft.pth"
        self.sea_checkpoint = self.dependencies / "sea.pth"
        self.sea_config = self.dependencies / "sea.json"
        self.isqoe_repo = self.dependencies / "isqoe"
        self.isqoe_repo.mkdir()
        self.isqoe_checkpoint = self.dependencies / "isqoe.ckpt"
        for path in (self.raft_checkpoint, self.sea_checkpoint, self.sea_config,
                     self.isqoe_checkpoint):
            path.write_bytes(b"test")

    def tearDown(self):
        self.temporary.cleanup()

    def arguments(self, *extra):
        base = [
            "--run-dir", str(self.run_dir),
            "--raft-repo", str(self.raft_repo),
            "--raft-checkpoint", str(self.raft_checkpoint),
            "--sea-repo", str(self.sea_repo),
            "--sea-checkpoint", str(self.sea_checkpoint),
            "--sea-config", str(self.sea_config),
            "--isqoe-repo", str(self.isqoe_repo),
            "--isqoe-checkpoint", str(self.isqoe_checkpoint),
        ]
        return runner.build_parser().parse_args(base + list(extra))

    def test_models_load_once_and_results_are_split_by_clip(self):
        def fake_raft(inputs, repo, checkpoint, device, valid_iters,
                      max_eye_width, save_fields):
            del repo, checkpoint, device, valid_iters, max_eye_width, save_fields
            return {
                "schema": 1,
                "oracle": "RAFT-Stereo",
                "role": "diagnostic_experimental",
                "frames": [
                    {"path": str(path), "metrics": {"status": "ok"}}
                    for path in inputs
                ],
            }

        def fake_sea(source_dir, output_dir, model, source_pattern, sbs_pattern,
                     evidence_dir, eye_flow):
            del source_dir, output_dir, model, source_pattern, sbs_pattern
            del evidence_dir, eye_flow
            return {
                "schema": 1,
                "oracle": "SEA-RAFT",
                "pairs_total": 1,
                "pairs_measured": 1,
                "pairs_cut": 0,
                "pairs_abstained": 0,
                "aggregate": {"sea_flow_edge_ghost_p95_p50": 2.0},
                "frames": [],
            }

        def fake_flip(*unused, **options):
            payload = {
                "schema": 1,
                "oracle": "nvidia-flip-exact-appearance",
                "status": "ok",
                "role": "optional_eval_only_experimental_diagnostic",
                "qualification": "experimental_diagnostic_only",
                "training_label_eligible": False,
                "support": {"pct": 80.0},
                "metrics": {"flip_worst_eye_p99": 0.04},
            }
            return (payload, {}) if options.get("return_maps") else payload

        DummySeaModel.load_count = 0
        DummyIsqoeModel.load_count = 0
        with mock.patch.object(
                runner.raft_stereo_oracle, "evaluate_paths", side_effect=fake_raft
        ) as raft_call, mock.patch.object(
                runner.sea_raft_temporal_oracle, "SeaRaftModel", DummySeaModel
        ), mock.patch.object(
                runner.sea_raft_temporal_oracle, "evaluate_sequence", side_effect=fake_sea
        ) as sea_call, mock.patch.object(
                runner.flip_appearance_oracle, "measure_flip_appearance",
                side_effect=fake_flip
        ) as flip_call, mock.patch.object(
                runner.isqoe_oracle, "IsqoeModel", DummyIsqoeModel
        ):
            root, exit_code = runner.run(self.arguments())

        self.assertEqual(exit_code, 0)
        self.assertEqual(raft_call.call_count, 1)
        self.assertEqual(DummySeaModel.load_count, 1)
        self.assertEqual(sea_call.call_count, 2)
        self.assertEqual(flip_call.call_count, 2)
        self.assertEqual(DummyIsqoeModel.load_count, 1)
        self.assertEqual(root["status"], "complete")
        self.assertFalse(root["training_label_eligible"])
        for clip in ("clip_a", "clip_b"):
            raft_path = self.run_dir / clip / "offline_oracles" / "raft_stereo.json"
            sea_path = self.run_dir / clip / "offline_oracles" / "sea_raft_temporal.json"
            flip_path = self.run_dir / clip / "offline_oracles" / "nvidia_flip_appearance.json"
            isqoe_path = self.run_dir / clip / "offline_oracles" / "apple_isqoe.json"
            raft_payload = json.loads(raft_path.read_text(encoding="utf-8"))
            sea_payload = json.loads(sea_path.read_text(encoding="utf-8"))
            flip_payload = json.loads(flip_path.read_text(encoding="utf-8"))
            isqoe_payload = json.loads(isqoe_path.read_text(encoding="utf-8"))
            self.assertEqual(raft_payload["clip"], clip)
            self.assertEqual(len(raft_payload["frames"]), 1)
            self.assertFalse(raft_payload["training_label_eligible"])
            self.assertEqual(sea_payload["clip"], clip)
            self.assertFalse(sea_payload["training_label_eligible"])
            self.assertEqual(flip_payload["status"], "ok")
            self.assertEqual(len(flip_payload["frames"]), 1)
            self.assertFalse(flip_payload["training_label_eligible"])
            self.assertEqual(isqoe_payload["status"], "ok")
            self.assertEqual(len(isqoe_payload["frames"]), 1)
            self.assertFalse(isqoe_payload["training_label_eligible"])

    def test_allow_unavailable_writes_explicit_root_and_clip_records(self):
        args = self.arguments("--oracles", "raft-stereo", "--allow-unavailable")
        args.raft_repo = None
        args.raft_checkpoint = None
        with mock.patch.dict("os.environ", {}, clear=True):
            root, exit_code = runner.run(args)
        self.assertEqual(exit_code, 0)
        self.assertEqual(root["status"], "unavailable")
        self.assertEqual(root["oracles"]["raft-stereo"]["status"], "unavailable")
        payload = json.loads((
            self.run_dir / "clip_a" / "offline_oracles" / "raft_stereo.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "unavailable")
        self.assertFalse(payload["training_label_eligible"])

    def test_missing_dependency_fails_closed_by_default_after_manifest(self):
        args = self.arguments("--oracles", "raft-stereo")
        args.raft_repo = None
        args.raft_checkpoint = None
        with mock.patch.dict("os.environ", {}, clear=True):
            root, exit_code = runner.run(args)
        self.assertEqual(exit_code, 2)
        self.assertEqual(root["status"], "unavailable")
        self.assertTrue((self.run_dir / "offline_oracles.json").is_file())

    def test_runtime_model_failure_is_never_hidden_by_allow_unavailable(self):
        args = self.arguments("--oracles", "raft-stereo", "--allow-unavailable")
        with mock.patch.object(
                runner.raft_stereo_oracle, "evaluate_paths",
                side_effect=RuntimeError("checkpoint ABI mismatch")):
            root, exit_code = runner.run(args)
        self.assertEqual(exit_code, 2)
        self.assertEqual(root["status"], "failed")
        reason = root["oracles"]["raft-stereo"]["reason"]
        self.assertIn("checkpoint ABI mismatch", reason)

    def test_flip_hdr_preview_abstains_without_loading_dependency(self):
        for clip in ("clip_a", "clip_b"):
            (self.run_dir / clip / "hdr_output_stats.json").write_text(json.dumps({
                "format": "linear-scRGB-fp16",
                "hdr_source_kind": "native-pq-in-windows-hdr",
                "png_is_preview": True,
            }), encoding="utf-8")
        args = self.arguments("--oracles", "nvidia-flip")
        with mock.patch.object(
                runner.flip_appearance_oracle, "load_official_flip",
                side_effect=AssertionError("HDR must abstain before dependency load")):
            root, exit_code = runner.run(args)
        self.assertEqual(exit_code, 0)
        self.assertEqual(root["status"], "complete")
        self.assertEqual(root["oracles"]["nvidia-flip"]["status"], "abstained")
        payload = json.loads((
            self.run_dir / "clip_a" / "offline_oracles" /
            "nvidia_flip_appearance.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "abstained")
        self.assertEqual(payload["summary"]["frames_abstained"], 1)
        self.assertIn("preview", payload["frames"][0]["reason"].lower())

    def test_flip_missing_dependency_fails_closed_unless_allowed(self):
        args = self.arguments("--oracles", "nvidia-flip")
        with mock.patch.object(
                runner.flip_appearance_oracle, "load_official_flip",
                side_effect=runner.flip_appearance_oracle.FlipUnavailable("not installed")):
            root, exit_code = runner.run(args)
        self.assertEqual(exit_code, 2)
        self.assertEqual(root["status"], "unavailable")

    def test_isqoe_missing_dependency_is_explicitly_unavailable(self):
        args = self.arguments("--oracles", "apple-isqoe")
        args.isqoe_repo = None
        args.isqoe_checkpoint = None
        with mock.patch.dict("os.environ", {}, clear=True):
            root, exit_code = runner.run(args)
        self.assertEqual(exit_code, 2)
        self.assertEqual(root["status"], "unavailable")
        self.assertEqual(root["oracles"]["apple-isqoe"]["status"], "unavailable")

    def test_isqoe_hdr_preview_abstains_before_dependency_or_model_load(self):
        for clip in ("clip_a", "clip_b"):
            (self.run_dir / clip / "hdr_output_stats.json").write_text(json.dumps({
                "format": "linear-scRGB-fp16",
                "hdr_source_kind": "native-pq-in-windows-hdr",
                "png_is_preview": True,
            }), encoding="utf-8")
        args = self.arguments("--oracles", "apple-isqoe")
        args.isqoe_repo = None
        args.isqoe_checkpoint = None
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(
                runner.isqoe_oracle, "IsqoeModel",
                side_effect=AssertionError("HDR must abstain before model loading")):
            root, exit_code = runner.run(args)
        self.assertEqual(exit_code, 0)
        self.assertEqual(root["status"], "complete")
        self.assertEqual(root["oracles"]["apple-isqoe"]["status"], "abstained")
        payload = json.loads((
            self.run_dir / "clip_a" / "offline_oracles" / "apple_isqoe.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["frames_abstained"], 1)
        self.assertFalse(payload["training_label_eligible"])

    def test_isqoe_mixed_hdr_preserves_abstention_when_ldr_dependency_is_missing(self):
        (self.run_dir / "clip_a" / "hdr_output_stats.json").write_text(json.dumps({
            "format": "linear-scRGB-fp16",
            "hdr_source_kind": "native-pq-in-windows-hdr",
            "png_is_preview": True,
        }), encoding="utf-8")
        args = self.arguments("--oracles", "apple-isqoe", "--allow-unavailable")
        args.isqoe_repo = None
        args.isqoe_checkpoint = None
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(
                runner.isqoe_oracle, "IsqoeModel",
                side_effect=AssertionError("missing dependency must not load a model")):
            root, exit_code = runner.run(args)
        self.assertEqual(exit_code, 0)
        self.assertEqual(root["status"], "unavailable")
        self.assertEqual(root["oracles"]["apple-isqoe"]["status"], "unavailable")
        self.assertEqual(
            root["clips"]["clip_a"]["apple-isqoe"]["status"], "abstained")
        self.assertEqual(
            root["clips"]["clip_b"]["apple-isqoe"]["status"], "unavailable")
        hdr_payload = json.loads((
            self.run_dir / "clip_a" / "offline_oracles" / "apple_isqoe.json"
        ).read_text(encoding="utf-8"))
        ldr_payload = json.loads((
            self.run_dir / "clip_b" / "offline_oracles" / "apple_isqoe.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(hdr_payload["summary"]["frames_abstained"], 1)
        self.assertEqual(ldr_payload["summary"]["frames_unavailable"], 1)
        self.assertFalse(hdr_payload["frames"][0]["training_label_eligible"])
        self.assertFalse(ldr_payload["frames"][0]["training_label_eligible"])

    def test_flip_requires_same_id_exact_map_without_hiding_other_clip(self):
        (self.run_dir / "clip_a" / "warp_map_00000.f32").unlink()
        args = self.arguments("--oracles", "nvidia-flip")

        def fake_flip(*unused, **options):
            del options
            return {
                "schema": 1,
                "oracle": "nvidia-flip-exact-appearance",
                "status": "ok",
                "training_label_eligible": False,
                "support": {"pct": 80.0},
                "metrics": {},
            }

        with mock.patch.object(
                runner.flip_appearance_oracle, "measure_flip_appearance",
                side_effect=fake_flip) as flip_call:
            root, exit_code = runner.run(args)
        self.assertEqual(exit_code, 2)
        self.assertEqual(root["status"], "failed")
        self.assertEqual(root["clips"]["clip_a"]["nvidia-flip"]["status"], "failed")
        self.assertEqual(root["clips"]["clip_b"]["nvidia-flip"]["status"], "ok")
        self.assertEqual(flip_call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
