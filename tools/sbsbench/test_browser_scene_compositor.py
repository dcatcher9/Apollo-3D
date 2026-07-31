#!/usr/bin/env python3

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import browser_scene_compositor as compositor  # noqa: E402


class BrowserSceneCompositorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = compositor.load_manifest()

    def test_committed_manifest_has_locked_adversarial_coverage(self):
        names = {
            scenario["name"] for scenario in self.manifest["scenarios"]
        }
        self.assertEqual(
            names,
            {
                "video_sidebar_ad",
                "paused_video_animated_sidebar",
                "cold_start_static_media",
                "collage_scroll",
                "ambiguous_dual_video",
                "partial_player",
                "embedded_to_fullscreen",
                "video_sidebar_style_swap",
                "static_dense_collage",
                "stable_relocated_player",
                "cross_axis_disjoint_motion",
                "sparse_three_side_candidate",
                "cold_paused_video",
                "ambiguous_candidate_during_scroll",
                "animated_sidebar_no_dominant_player",
                "interrupted_relocation",
                "grayscale_static_media",
            },
        )
        dimensions = {
            (scenario["width"], scenario["height"])
            for scenario in self.manifest["scenarios"]
        }
        self.assertGreaterEqual(len(dimensions), 6)
        self.assertTrue(any(width < height for width, height in dimensions))
        counterstyle = next(
            scenario for scenario in self.manifest["scenarios"]
            if scenario["name"] == "video_sidebar_style_swap"
        )
        patterns = {
            layer["role"]: layer["pattern"]
            for layer in counterstyle["layers"]
            if layer["role"] in {"primary_video", "ad_unsafe"}
        }
        self.assertEqual(
            patterns,
            {"primary_video": "cards", "ad_unsafe": "video"},
        )
        by_name = {
            scenario["name"]: scenario
            for scenario in self.manifest["scenarios"]
        }

        def contracted_layouts(scenario):
            layouts = {scenario["expected_layout"]}
            layouts.update(
                transition["layout"]
                for transition in scenario.get(
                    "expected_layout_transitions",
                    [],
                )
            )
            if "expected_layout_after_geometry_reset" in scenario:
                layouts.add(
                    scenario["expected_layout_after_geometry_reset"]
                )
            return layouts

        required_targets = {
            name for name, scenario in by_name.items()
            if contracted_layouts(scenario) & {
                "primary_video", "content_collage"
            }
        }
        self.assertEqual(
            required_targets,
            {
                "video_sidebar_ad",
                "paused_video_animated_sidebar",
                "collage_scroll",
                "embedded_to_fullscreen",
                "video_sidebar_style_swap",
                "static_dense_collage",
                "stable_relocated_player",
                "cold_paused_video",
                "interrupted_relocation",
            },
        )
        self.assertEqual(
            {
                name for name, scenario in by_name.items()
                if contracted_layouts(scenario) == {"ambiguous"}
            },
            {
                "cold_start_static_media",
                "ambiguous_dual_video",
                "partial_player",
                "cross_axis_disjoint_motion",
                "sparse_three_side_candidate",
                "ambiguous_candidate_during_scroll",
                "animated_sidebar_no_dominant_player",
                "grayscale_static_media",
            },
        )
        for name in (
            "stable_relocated_player",
            "ambiguous_dual_video",
            "video_sidebar_style_swap",
        ):
            scenario = by_name[name]
            dwell_updates = math.ceil(
                0.75 * scenario["fps_num"] / scenario["fps_den"]
            )
            self.assertGreaterEqual(
                scenario["frames"] - dwell_updates,
                20,
                f"{name} needs a meaningful post-dwell observation tail",
            )
        style_swap = by_name["video_sidebar_style_swap"]
        cut_frame = style_swap["events"]["content_cut_frames"][0]
        self.assertGreater(
            cut_frame,
            math.ceil(
                0.75 * style_swap["fps_num"] / style_swap["fps_den"]
            ) + 2,
        )
        self.assertGreaterEqual(style_swap["frames"] - cut_frame, 12)

    def test_manifest_rejects_boolean_integer_and_unknown_fields(self):
        wrong_type = copy.deepcopy(self.manifest)
        wrong_type["scenarios"][0]["frames"] = True
        with self.assertRaisesRegex(compositor.CompositorError, "frames"):
            compositor.validate_manifest(wrong_type)

        wrong_schema = copy.deepcopy(self.manifest)
        wrong_schema["schema"] = True
        with self.assertRaisesRegex(compositor.CompositorError, "schema"):
            compositor.validate_manifest(wrong_schema)

        unknown = copy.deepcopy(self.manifest)
        unknown["scenarios"][0]["events"]["magic_cut"] = []
        with self.assertRaisesRegex(compositor.CompositorError, "unknown keys"):
            compositor.validate_manifest(unknown)

        invalid_pause_mode = copy.deepcopy(self.manifest)
        paused = invalid_pause_mode["scenarios"][1]["layers"][2]
        paused["animation"] = "static"
        with self.assertRaisesRegex(
            compositor.CompositorError,
            "animation_pause_frames requires frame animation",
        ):
            compositor.validate_manifest(invalid_pause_mode)

        invalid_pause_frame = copy.deepcopy(self.manifest)
        invalid_pause_frame["scenarios"][1]["layers"][2][
            "animation_pause_frames"
        ][1] = invalid_pause_frame["scenarios"][1]["frames"]
        with self.assertRaisesRegex(
            compositor.CompositorError,
            "animation_pause_frames",
        ):
            compositor.validate_manifest(invalid_pause_frame)

        invalid_relocation = copy.deepcopy(self.manifest)
        relocated = next(
            scenario for scenario in invalid_relocation["scenarios"]
            if scenario["name"] == "stable_relocated_player"
        )
        relocated["layers"][2]["rect_after_relocation"] = (
            relocated["layers"][2]["rect"]
        )
        with self.assertRaisesRegex(
            compositor.CompositorError,
            "relocation must move exactly one selected target",
        ):
            compositor.validate_manifest(invalid_relocation)

        invalid_window = copy.deepcopy(self.manifest)
        relocated = next(
            scenario for scenario in invalid_window["scenarios"]
            if scenario["name"] == "stable_relocated_player"
        )
        relocated["events"]["relocation_acceptance_windows"] = []
        with self.assertRaisesRegex(
            compositor.CompositorError,
            "one window per relocation frame",
        ):
            compositor.validate_manifest(invalid_window)

    def test_render_rejects_declared_event_without_visible_scoped_change(self):
        invisible = copy.deepcopy(self.manifest)
        scenario = invisible["scenarios"][0]
        player = next(
            layer for layer in scenario["layers"]
            if layer["role"] == "primary_video"
        )
        player["pattern"] = "solid"
        player["animation"] = "static"
        with self.assertRaisesRegex(
            compositor.CompositorError,
            "does not create a visible scoped pixel change",
        ):
            compositor.render_scenario(invisible, scenario["name"])

    def test_render_is_byte_deterministic(self):
        first = compositor.render_scenario(self.manifest, "video_sidebar_ad")
        second = compositor.render_scenario(self.manifest, "video_sidebar_ad")
        self.assertEqual(first.timeline, second.timeline)
        self.assertEqual(set(first.arrays), set(second.arrays))
        for name in first.arrays:
            self.assertTrue(np.array_equal(first.arrays[name], second.arrays[name]))

    def test_sidebar_cuts_are_not_content_cuts_and_never_enter_target(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        timeline = {item["accepted_depth_update_index"]: item
                    for item in rendered.timeline}
        self.assertTrue(timeline[9]["exterior_cut"])
        self.assertFalse(timeline[9]["content_cut"])
        self.assertTrue(timeline[19]["exterior_cut"])
        self.assertTrue(timeline[19]["content_cut"])
        self.assertTrue(timeline[14]["exposure_only"])
        self.assertFalse(timeline[14]["content_cut"])
        self.assertTrue(timeline[15]["exposure_only"])
        self.assertEqual(timeline[15]["exposure_gain_percent"], 100)

        target = rendered.arrays["target"]
        exclusion = rendered.arrays["exclusion"]
        self.assertFalse(np.any(target & exclusion))
        self.assertTrue(np.all(target[:, 31:158, 16:220]))
        self.assertFalse(np.any(target[:, :, 233:]))

    def test_ambiguous_layout_abstains_with_explicit_candidates(self):
        rendered = compositor.render_scenario(
            self.manifest, "ambiguous_dual_video"
        )
        self.assertFalse(np.any(rendered.arrays["target"]))
        self.assertTrue(np.any(rendered.arrays["ambiguity"]))
        self.assertTrue(all(
            frame["expected_layout"] == "ambiguous" and
            frame["selected_instance_id"] == 0 and
            frame["expected_roi_px"] is None
            for frame in rendered.timeline
        ))

    def test_scroll_and_geometry_reset_have_exact_timelines(self):
        scroll = compositor.render_scenario(self.manifest, "collage_scroll")
        active = [
            frame["accepted_depth_update_index"]
            for frame in scroll.timeline if frame["scroll"]
        ]
        self.assertEqual(active, list(range(8, 18)))
        self.assertEqual(scroll.timeline[6]["scroll_offset_px"], [0, 0])
        self.assertEqual(scroll.timeline[7]["scroll_offset_px"], [0, 5])
        self.assertEqual(scroll.timeline[-1]["scroll_offset_px"], [0, 50])

        fullscreen = compositor.render_scenario(
            self.manifest, "embedded_to_fullscreen"
        )
        self.assertEqual(fullscreen.timeline[43]["geometry_generation"], 0)
        self.assertTrue(fullscreen.timeline[44]["geometry_reset"])
        self.assertEqual(fullscreen.timeline[44]["geometry_generation"], 1)
        self.assertEqual(
            fullscreen.timeline[44]["expected_roi_px"], [0, 0, 384, 216]
        )
        self.assertFalse(np.any(fullscreen.arrays["exclusion"][44]))

        relocated = compositor.render_scenario(
            self.manifest,
            "stable_relocated_player",
        )
        self.assertFalse(relocated.timeline[35]["relocation"])
        self.assertTrue(relocated.timeline[36]["relocation_observed"])
        self.assertFalse(relocated.timeline[36]["relocation"])
        self.assertEqual(
            relocated.timeline[35]["expected_roi_px"],
            [18, 42, 170, 180],
        )
        self.assertEqual(
            relocated.timeline[36]["expected_roi_px"],
            [18, 42, 170, 180],
        )
        self.assertFalse(relocated.timeline[58]["relocation"])
        self.assertTrue(relocated.timeline[59]["relocation"])
        self.assertEqual(
            relocated.timeline[59]["expected_roi_px"],
            [190, 42, 342, 180],
        )

    def test_playing_video_pauses_and_resumes_while_sidebar_changes(self):
        rendered = compositor.render_scenario(
            self.manifest, "paused_video_animated_sidebar"
        )
        rgb = rendered.arrays["rgb"]
        player = np.s_[41:211, 18:248]
        sidebar = np.s_[41:225, 262:351]
        self.assertFalse(np.array_equal(
            rgb[0][player],
            rgb[1][player],
        ))
        self.assertTrue(np.array_equal(
            rgb[23][player],
            rgb[24][player],
        ))
        self.assertTrue(np.array_equal(
            rgb[24][player],
            rgb[43][player],
        ))
        self.assertFalse(np.array_equal(
            rgb[43][player],
            rgb[44][player],
        ))
        self.assertFalse(np.array_equal(
            rgb[24][sidebar],
            rgb[-1][sidebar],
        ))

    def test_playing_video_changes_locally_until_the_declared_cut(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_ad"
        )
        rgb = rendered.arrays["rgb"]
        player = rendered.arrays["target"]
        before = rgb[0][player[0]]
        next_frame = rgb[1][player[1]]
        changed_fraction = float(np.mean(np.any(before != next_frame, axis=1)))
        self.assertGreater(changed_fraction, 0.0)
        self.assertLess(changed_fraction, 0.15)
        pre_cut = rgb[17][player[17]]
        at_cut = rgb[18][player[18]]
        cut_fraction = float(np.mean(np.any(pre_cut != at_cut, axis=1)))
        self.assertGreater(cut_fraction, 0.90)

    def test_counterstyle_animation_cannot_cancel_its_declared_cut(self):
        rendered = compositor.render_scenario(
            self.manifest, "video_sidebar_style_swap"
        )
        rgb = rendered.arrays["rgb"]
        target = rendered.arrays["target"]
        ordinary_fraction = float(np.mean(np.any(
            rgb[15][target[15]] != rgb[14][target[14]],
            axis=1,
        )))
        cut_fraction = float(np.mean(np.any(
            rgb[32][target[32]] != rgb[31][target[31]],
            axis=1,
        )))
        self.assertGreater(ordinary_fraction, 0.0)
        self.assertLess(ordinary_fraction, 0.15)
        self.assertGreater(cut_fraction, 0.60)
        self.assertGreater(cut_fraction, ordinary_fraction * 4.0)

    def test_cold_start_static_media_has_no_playback_evidence(self):
        rendered = compositor.render_scenario(
            self.manifest, "cold_start_static_media"
        )
        rgb = rendered.arrays["rgb"]
        target = rendered.arrays["target"]
        self.assertTrue(all(
            frame["expected_layout"] == "ambiguous"
            for frame in rendered.timeline
        ))
        self.assertFalse(np.any(target))
        static_media = np.s_[38:202, 18:245]
        for frame in range(1, rgb.shape[0]):
            self.assertTrue(np.array_equal(
                rgb[0][static_media],
                rgb[frame][static_media],
            ))
        sidebar = np.s_[38:211, 261:343]
        self.assertFalse(np.array_equal(
            rgb[0][sidebar],
            rgb[-1][sidebar],
        ))

    def test_all_unsupported_scenes_have_empty_targets(self):
        unsupported = {
            scenario["name"]
            for scenario in self.manifest["scenarios"]
            if (
                scenario["expected_layout"] == "ambiguous" and
                not scenario.get("expected_layout_transitions")
            )
        }
        for name in unsupported:
            with self.subTest(scenario=name):
                rendered = compositor.render_scenario(
                    self.manifest,
                    name,
                )
                self.assertFalse(np.any(rendered.arrays["target"]))
                self.assertTrue(all(
                    row["expected_roi_px"] is None
                    for row in rendered.timeline
                ))

        grayscale = compositor.render_scenario(
            self.manifest,
            "grayscale_static_media",
        ).arrays["rgb"][:, 38:184, 30:290]
        self.assertTrue(np.array_equal(
            grayscale[..., 0],
            grayscale[..., 1],
        ))
        self.assertTrue(np.array_equal(
            grayscale[..., 1],
            grayscale[..., 2],
        ))

    def test_cold_static_envelope_is_content_without_video_authority(self):
        rendered = compositor.render_scenario(
            self.manifest,
            "cold_paused_video",
        )
        self.assertTrue(all(
            row["expected_layout"] == "content_collage"
            for row in rendered.timeline
        ))
        self.assertTrue(np.any(rendered.arrays["target"]))
        self.assertEqual(
            rendered.timeline[0]["expected_roi_px"],
            [34, 42, 286, 184],
        )

    def test_ambiguous_candidate_during_scroll_remains_full_frame(self):
        rendered = compositor.render_scenario(
            self.manifest,
            "ambiguous_candidate_during_scroll",
        )
        self.assertTrue(all(
            row["expected_layout"] == "ambiguous"
            for row in rendered.timeline
        ))
        self.assertTrue(all(
            row["expected_roi_px"] is None
            for row in rendered.timeline
        ))
        self.assertFalse(np.any(rendered.arrays["target"]))
        self.assertTrue(
            np.any(rendered.arrays["instance"] == 851),
            "the ambiguous candidate must remain visually present",
        )

    def test_interrupted_challenger_never_replaces_the_incumbent(self):
        rendered = compositor.render_scenario(
            self.manifest,
            "interrupted_relocation",
        )
        instances = rendered.arrays["instance"]
        self.assertFalse(np.any(instances[16] == 872))
        self.assertTrue(np.any(instances[17] == 872))
        self.assertTrue(np.any(instances[29] == 872))
        self.assertFalse(np.any(instances[30] == 872))
        self.assertTrue(all(
            row["expected_layout"] == "primary_video" and
            row["expected_roi_px"] == [16, 40, 170, 184]
            for row in rendered.timeline
        ))

    def test_materialization_is_deterministic_and_self_authenticating(self):
        rendered = compositor.render_scenario(
            self.manifest, "partial_player"
        )
        with tempfile.TemporaryDirectory() as root:
            first = compositor.materialize(
                rendered, Path(root) / "first", emit_frames=True
            )
            second = compositor.materialize(rendered, Path(root) / "second")
            first_artifact = json.loads(
                (first / "artifact.json").read_text(encoding="utf-8")
            )
            second_artifact = json.loads(
                (second / "artifact.json").read_text(encoding="utf-8")
            )
            self.assertNotEqual(
                first_artifact["emitted_frame_files"],
                second_artifact["emitted_frame_files"],
            )
            comparable_first = {
                key: value for key, value in first_artifact.items()
                if key != "emitted_frame_files"
            }
            comparable_second = {
                key: value for key, value in second_artifact.items()
                if key != "emitted_frame_files"
            }
            self.assertEqual(comparable_first, comparable_second)
            self.assertEqual(
                (first / "sequence.npz").read_bytes(),
                (second / "sequence.npz").read_bytes(),
            )
            self.assertTrue((first / "frame_00001.png").is_file())
            meta = json.loads((first / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(
                meta["scene_controller_contract"]["scenario"],
                "partial_player",
            )
            self.assertEqual(meta["frame_rate"], "30/1")
            canonical_recipe = json.dumps(
                rendered.recipe, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            self.assertEqual(
                meta["scene_controller_contract"]["recipe_sha256"],
                hashlib.sha256(canonical_recipe).hexdigest(),
            )
            self.assertEqual(
                first_artifact["recipe_sha256"],
                meta["scene_controller_contract"]["recipe_sha256"],
            )
            contract = meta["scene_controller_contract"]
            for name, hash_key in (
                ("recipe.json", "recipe_file_sha256"),
                ("sequence.npz", "labels_sha256"),
                ("timeline.jsonl", "timeline_sha256"),
                ("artifact.json", "artifact_sha256"),
            ):
                self.assertEqual(
                    hashlib.sha256((first / name).read_bytes()).hexdigest(),
                    contract[hash_key],
                )
            with np.load(first / "sequence.npz", allow_pickle=False) as sequence:
                self.assertEqual(
                    tuple(sequence["rgb"].shape), (48, 320, 240, 3)
                )
                self.assertEqual(sequence["instance"].dtype, np.uint16)
                self.assertEqual(sequence["target"].dtype, np.bool_)

    def test_materialization_refuses_to_replace_existing_output(self):
        rendered = compositor.render_scenario(
            self.manifest, "partial_player"
        )
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "exists"
            output.mkdir()
            with self.assertRaisesRegex(
                compositor.CompositorError, "already exists"
            ):
                compositor.materialize(rendered, output)


if __name__ == "__main__":
    unittest.main()
