import json
from pathlib import Path
import tempfile
import unittest

from tools.sbsbench import offline_oracle_report as report


class OfflineOracleReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name)
        self.clip_dir = self.run_dir / "clip_a" / "offline_oracles"
        self.clip_dir.mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write_json(path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_no_manifest_produces_no_report_section(self):
        self.assertEqual(report.build_section(self.run_dir, ["clip_a"], str), "")

    def test_section_is_collapsed_diagnostic_and_summarizes_all_oracles(self):
        self.write_json(self.run_dir / "offline_oracles.json", {
            "status": "complete",
            "selected_oracles": ["raft-stereo", "sea-raft", "nvidia-flip"],
            "training_label_eligible": False,
            "oracles": {},
        })
        self.write_json(self.clip_dir / "raft_stereo.json", {
            "status": "ok",
            "training_label_eligible": False,
            "summary": {"frames_total": 2, "frames_measured": 1,
                        "frames_abstained": 1},
            "frames": [{
                "metrics": {
                    "status": "ok",
                    "raft_supported_texture_pct": 81.25,
                    "raft_correspondence_residual_p95": 7.5,
                    "raft_exact_p95_px": 1.25,
                    "raft_exact_bad_1px_pct": 22.0,
                    "raft_vertical_abs_p95_px": 0.5,
                },
            }],
        })
        self.write_json(self.clip_dir / "sea_raft_temporal.json", {
            "status": "ok",
            "training_label_eligible": False,
            "pairs_total": 4,
            "pairs_measured": 2,
            "pairs_cut": 1,
            "pairs_abstained": 1,
            "aggregate": {
                "sea_flow_edge_ghost_p95_p95": 9.0,
                "sea_flow_flicker_p95_p95": 4.0,
                "sea_static_jitter_p95_p50": 2.0,
                "sea_left_motion_mismatch_p95_px_p95": 0.75,
                "sea_right_motion_mismatch_p95_px_p95": 1.25,
            },
        })
        self.write_json(self.clip_dir / "nvidia_flip_appearance.json", {
            "status": "ok",
            "training_label_eligible": False,
            "summary": {
                "frames_total": 3,
                "frames_measured": 2,
                "frames_abstained": 1,
                "frames_unavailable": 0,
                "frames_failed": 0,
            },
            "frames": [{
                "status": "ok",
                "support": {"pct": 75.0},
                "metrics": {
                    "flip_worst_eye_p99": 0.12,
                    "flip_worst_eye_area_gt_050_pct": 3.0,
                    "flip_interocular_error_imbalance_p99": 0.08,
                    "flip_interocular_area_imbalance_gt_050_pct": 1.0,
                },
            }, {
                "status": "ok",
                "support": {"pct": 85.0},
                "metrics": {
                    "flip_worst_eye_p99": 0.20,
                    "flip_worst_eye_area_gt_050_pct": 5.0,
                    "flip_interocular_error_imbalance_p99": 0.10,
                    "flip_interocular_area_imbalance_gt_050_pct": 2.0,
                },
            }, {
                "status": "abstained",
                "reason": "HDR preview",
                "metrics": {},
            }],
        })

        rendered = report.build_section(
            self.run_dir, ["clip_a"], lambda unused: "Movie <scene>")

        self.assertIn('<details class="fold learned-oracles">', rendered)
        self.assertNotIn('<details class="fold learned-oracles" open', rendered)
        self.assertIn("Diagnostic only", rendered)
        self.assertIn("not used by gates", rendered)
        self.assertIn("or training labels", rendered)
        self.assertIn("RAFT-Stereo correspondence", rendered)
        self.assertIn("SEA-RAFT temporal diagnostics", rendered)
        self.assertIn("NVIDIA FLIP registered appearance", rendered)
        self.assertIn("HDR preview PNGs abstain", rendered)
        self.assertIn("Movie &lt;scene&gt;", rendered)
        self.assertIn("1/2 · 1 abstain", rendered)
        self.assertIn("1.25 px / 22.00% bad&gt;1", rendered)
        self.assertIn("2/4 · 1 abstain, 1 cut", rendered)
        self.assertIn("9.00/255", rendered)
        self.assertIn("1.25 px", rendered)
        self.assertIn("2/3", rendered)
        self.assertIn("80.00%", rendered)
        self.assertIn("0.16", rendered)
        self.assertIn("4.00%", rendered)

    def test_missing_payload_and_label_contract_violation_are_visible_not_fatal(self):
        self.write_json(self.run_dir / "offline_oracles.json", {
            "status": "partial",
            "selected_oracles": ["raft-stereo"],
            "training_label_eligible": True,
            "oracles": {},
        })
        rendered = report.build_section(self.run_dir, ["clip_a"], str)
        self.assertIn("contract warning", rendered)
        self.assertIn("Root manifest lacks training_label_eligible=false", rendered)
        self.assertIn("1 missing payloads", rendered)
        self.assertIn("No valid per-clip payload", rendered)

    def test_isqoe_is_diagnostic_only_and_records_official_checkpoint(self):
        sha256 = "1a4a367ac2bb03125cd5df9e507856dc50338f06315def4224d59a5ab55b5ed3"
        self.write_json(self.run_dir / "offline_oracles.json", {
            "status": "complete",
            "selected_oracles": ["apple-isqoe"],
            "training_label_eligible": False,
            "oracles": {
                "apple-isqoe": {
                    "status": "ok",
                    "training_label_eligible": False,
                    "summary": {"frames_measured": 1},
                    "provenance": {
                        "official_checkpoint_id": "isqoe_1_1",
                        "official_checkpoint_url": (
                            "https://ml-site.cdn-apple.com/models/isqoe/isqoe_1_1.ckpt"
                        ),
                        "checkpoint_sha256": sha256,
                        "checkpoint_matches_known_official_sha256": True,
                        "repository_revision": "2162283f12dac459721c5bc1a9187f2b590847a2",
                    },
                },
            },
        })
        self.write_json(self.clip_dir / "apple_isqoe.json", {
            "status": "ok",
            "training_label_eligible": False,
            "summary": {
                "frames_total": 1,
                "frames_measured": 1,
                "frames_abstained": 0,
                "frames_unavailable": 0,
                "frames_failed": 0,
            },
            "frames": [{
                "status": "ok",
                "training_label_eligible": False,
                "metrics": {
                    "isqoe_mean_score": 0.8,
                    "isqoe_worst_score": 0.9,
                    "isqoe_eye_order_delta": 0.2,
                },
            }],
        })

        rendered = report.build_section(self.run_dir, ["clip_a"], str)
        self.assertIn("Apple iSQoE headset-preference cross-check", rendered)
        self.assertIn("not an Apollo style target or a training label", rendered)
        self.assertIn("absolute cross-clip ranking is unsupported", rendered)
        self.assertIn("checkpoint isqoe_1_1", rendered)
        self.assertIn(f"SHA-256 {sha256}", rendered)
        self.assertIn("official download", rendered)
        self.assertNotIn("contract warning", rendered)

        payload = json.loads((self.clip_dir / "apple_isqoe.json").read_text(
            encoding="utf-8"))
        payload["frames"][0]["training_label_eligible"] = True
        self.write_json(self.clip_dir / "apple_isqoe.json", payload)
        rendered = report.build_section(self.run_dir, ["clip_a"], str)
        self.assertIn("contract warning", rendered)
        self.assertIn("has a label-eligible frame", rendered)

    def test_build_report_contains_optional_appendix_hook(self):
        build_report = Path(__file__).with_name("build_report.py").read_text(encoding="utf-8")
        self.assertIn("offline_oracle_report.build_section(treat_dir, CLIPS, name)", build_report)
        self.assertIn("__LEARNED_ORACLES__", build_report)


if __name__ == "__main__":
    unittest.main()
