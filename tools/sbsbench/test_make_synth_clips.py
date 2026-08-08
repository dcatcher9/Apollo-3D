import json
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_synth_clips  # noqa: E402


class SubtitleSyntheticClipTests(unittest.TestCase):
    @staticmethod
    def _generate(root):
        original_clips = make_synth_clips.CLIPS
        make_synth_clips.CLIPS = root
        try:
            for clip in make_synth_clips.SUBTITLE_CLIPS:
                make_synth_clips.GENERATORS[clip]()
                make_synth_clips.write_meta(
                    clip, **make_synth_clips.clip_metadata(clip))
        finally:
            make_synth_clips.CLIPS = original_clips

    @staticmethod
    def _files_below(root):
        return sorted(
            os.path.relpath(os.path.join(directory, filename), root)
            for directory, _, filenames in os.walk(root)
            for filename in filenames
        )

    @staticmethod
    def _component_count(mask):
        mask = np.asarray(mask, dtype=bool)
        visited = np.zeros_like(mask)
        components = 0
        height, width = mask.shape
        for y in range(height):
            for x in range(width):
                if not mask[y, x] or visited[y, x]:
                    continue
                components += 1
                visited[y, x] = True
                pending = [(y, x)]
                while pending:
                    current_y, current_x = pending.pop()
                    for next_y, next_x in (
                            (current_y - 1, current_x),
                            (current_y + 1, current_x),
                            (current_y, current_x - 1),
                            (current_y, current_x + 1)):
                        if (0 <= next_y < height and 0 <= next_x < width and
                                mask[next_y, next_x] and not visited[next_y, next_x]):
                            visited[next_y, next_x] = True
                            pending.append((next_y, next_x))
        return components

    def test_authored_layer_is_piecewise_static_over_moving_background(self):
        for variant, clip in enumerate(make_synth_clips.SUBTITLE_STANDARD_CLIPS):
            with self.subTest(clip=clip):
                first = make_synth_clips._subtitle_layers(clip, 1)
                same_cue = make_synth_clips._subtitle_layers(clip, 6)
                next_cue = make_synth_clips._subtitle_layers(clip, 7)
                for first_layer, same_cue_layer in zip(first, same_cue):
                    self.assertTrue(np.array_equal(first_layer, same_cue_layer))
                self.assertFalse(np.array_equal(first[1], next_cue[1]))

                first_background = make_synth_clips._movie_background(1, variant)
                same_cue_background = make_synth_clips._movie_background(6, variant)
                outside_region = first[3] == 0
                self.assertTrue(np.any(first_background[outside_region] !=
                                       same_cue_background[outside_region]))

                # A loose region contains substantially more support than the authored strokes.
                self.assertGreater(np.count_nonzero(first[3]), 4 * np.count_nonzero(first[1]))

    def test_layouts_cover_disjoint_and_tall_stress_cases(self):
        for frame_id in range(1, make_synth_clips.N + 1):
            with self.subTest(layout="top-bottom", frame_id=frame_id):
                region = make_synth_clips._subtitle_layers(
                    "subtitle_top_bottom_disjoint", frame_id)[3]
                self.assertEqual(self._component_count(region), 2)

            with self.subTest(layout="tall-stack", frame_id=frame_id):
                region = make_synth_clips._subtitle_layers(
                    "subtitle_bilingual_tall_stack", frame_id)[3]
                active_rows = np.flatnonzero(np.any(region != 0, axis=1))
                self.assertGreaterEqual(active_rows[-1] - active_rows[0] + 1, 160)

    def test_highres_transition_schedule_and_thin_strokes_survive_as_evidence(self):
        clip = make_synth_clips.SUBTITLE_HIGHRES_CLIP
        metadata = make_synth_clips.clip_metadata(clip)
        contract = metadata["subtitle_transition_contract"]
        self.assertEqual(contract["source_size_px"], [2560, 1440])
        self.assertEqual(contract["detector_target_size_px"], [770, 434])
        self.assertEqual(contract["authored_glyph_stroke_width_px"], 3)
        self.assertEqual(contract["authored_outline_radius_px"], 2)
        self.assertEqual(contract["empty_frame_ranges"], [[1, 4], [13, 16]])
        self.assertEqual(contract["appear_frames"], [5, 17])
        self.assertEqual(contract["subtitle_only_replacement_frames"], [9])
        self.assertEqual(contract["disappear_frames"], [13])
        self.assertEqual(contract["broad_scene_cut_frames"], [21])
        self.assertEqual(contract["overlay_replacement_at_scene_cut_frames"], [21])
        self.assertEqual(
            contract["subtitle_state_by_frame"],
            list(make_synth_clips.SUBTITLE_HIGHRES_STATE_BY_FRAME))
        self.assertEqual(
            contract["scene_state_by_frame"],
            list(make_synth_clips.SUBTITLE_HIGHRES_SCENE_BY_FRAME))
        self.assertEqual(metadata["subtitle_target_disparity_pct"], 0.0)
        self.assertEqual(metadata["shot_state_contract"], {
            "kind": "hard-cut",
            "monitor_from_frame": 2,
            "expected_pulse_frames": [21],
        })

        for frame_id, state in enumerate(
                make_synth_clips.SUBTITLE_HIGHRES_STATE_BY_FRAME, 1):
            with self.subTest(frame_id=frame_id, state=state):
                _, glyph, outline, region = (
                    make_synth_clips._highres_subtitle_layers(frame_id))
                if state == "empty":
                    self.assertFalse(np.any(glyph))
                    self.assertFalse(np.any(outline))
                    self.assertEqual(set(np.unique(region).tolist()), {0})
                else:
                    self.assertTrue(np.any(glyph))
                    self.assertTrue(np.any(outline))
                    self.assertEqual(set(np.unique(region).tolist()), {0, 255})
                    self.assertGreater(np.count_nonzero(region), 4 * np.count_nonzero(glyph))

        # The first replacement changes only burned-in pixels over the same slowly moving shot.
        before_replace = make_synth_clips._highres_subtitle_layers(8)
        after_replace = make_synth_clips._highres_subtitle_layers(9)
        self.assertFalse(np.array_equal(before_replace[1], after_replace[1]))
        replacement_outside = ~((before_replace[3] != 0) | (after_replace[3] != 0))
        replacement_left = make_synth_clips._highres_movie_background(8, "dusk-city")
        replacement_right = make_synth_clips._highres_movie_background(9, "dusk-city")
        replacement_delta = np.max(np.abs(
            replacement_left.astype(np.int16) - replacement_right.astype(np.int16)), axis=2)
        self.assertLess(np.mean(replacement_delta[replacement_outside] > 40), 0.05)

        # Frame 21 replaces almost the entire scene outside both loose overlay rectangles, so a
        # future subtitle exclusion cannot turn the genuine cut into an apparent false positive.
        before_cut = make_synth_clips._highres_subtitle_layers(20)
        after_cut = make_synth_clips._highres_subtitle_layers(21)
        self.assertFalse(np.array_equal(before_cut[1], after_cut[1]))
        cut_outside = ~((before_cut[3] != 0) | (after_cut[3] != 0))
        cut_left = make_synth_clips._highres_movie_background(20, "dusk-city")
        cut_right = make_synth_clips._highres_movie_background(21, "warm-interior")
        cut_delta = np.max(np.abs(
            cut_left.astype(np.int16) - cut_right.astype(np.int16)), axis=2)
        self.assertGreater(np.mean(cut_delta[cut_outside] > 40), 0.90)

        # The authored three-pixel strokes become sub-pixel at 770x434. A round trip back to the
        # source canvas makes the lost high-frequency energy directly comparable at one size.
        _, glyph, _, _ = make_synth_clips._highres_subtitle_layers(5)
        source = glyph.astype(np.uint8) * 255
        downscaled = np.asarray(Image.fromarray(source).resize(
            (make_synth_clips.SUBTITLE_DETECTOR_TARGET_W,
             make_synth_clips.SUBTITLE_DETECTOR_TARGET_H),
            Image.Resampling.BILINEAR))
        restored = np.asarray(Image.fromarray(downscaled).resize(
            (make_synth_clips.SUBTITLE_HIGHRES_W,
             make_synth_clips.SUBTITLE_HIGHRES_H),
            Image.Resampling.BILINEAR))

        def edge_energy(image):
            image = image.astype(np.float32)
            return (np.abs(np.diff(image, axis=1)).sum() +
                    np.abs(np.diff(image, axis=0)).sum())

        energy_ratio = edge_energy(restored) / edge_energy(source)
        self.assertLess(energy_ratio, 0.60)
        self.assertLess(int(downscaled.max()), 255)

    def test_committed_pngs_match_generator_and_sidecar_contract(self):
        committed_root = os.path.join(HERE, "clips")
        with tempfile.TemporaryDirectory() as generated_root:
            self._generate(generated_root)
            expected_ids = [f"frame_{frame_id:05d}.png" for frame_id in range(1, 25)]
            for clip in make_synth_clips.SUBTITLE_CLIPS:
                with self.subTest(clip=clip):
                    committed = os.path.join(committed_root, clip)
                    generated = os.path.join(generated_root, clip)
                    committed_files = self._files_below(committed)
                    generated_files = self._files_below(generated)
                    self.assertEqual(committed_files, generated_files)
                    for relative_path in generated_files:
                        with open(os.path.join(committed, relative_path), "rb") as stream:
                            committed_bytes = stream.read()
                        with open(os.path.join(generated, relative_path), "rb") as stream:
                            generated_bytes = stream.read()
                        self.assertEqual(committed_bytes, generated_bytes, relative_path)

                    source_ids = sorted(
                        filename for filename in os.listdir(generated)
                        if filename.startswith("frame_") and filename.endswith(".png"))
                    region_dir = os.path.join(generated, "gt_subtitle_region")
                    region_ids = sorted(os.listdir(region_dir))
                    self.assertEqual(source_ids, expected_ids)
                    self.assertEqual(region_ids, expected_ids)

                    with open(os.path.join(generated, "meta.json"), encoding="utf-8") as stream:
                        metadata = json.load(stream)
                    self.assertIs(metadata["required_gt_subtitle_region"], True)
                    self.assertEqual(metadata["subtitle_target_disparity_pct"], 0.0)
                    if clip == make_synth_clips.SUBTITLE_HIGHRES_CLIP:
                        expected_size = (
                            make_synth_clips.SUBTITLE_HIGHRES_W,
                            make_synth_clips.SUBTITLE_HIGHRES_H)
                        self.assertEqual(
                            metadata["subtitle_transition_contract"]["broad_scene_cut_frames"],
                            [21])
                    else:
                        expected_size = (make_synth_clips.W, make_synth_clips.H)
                        self.assertEqual(
                            metadata["subtitle_cue_start_frames"],
                            list(make_synth_clips.SUBTITLE_CUE_START_FRAMES))

                    for frame_id, filename in enumerate(expected_ids, 1):
                        with Image.open(os.path.join(generated, filename)) as source:
                            self.assertEqual(source.mode, "RGB")
                            self.assertEqual(source.size, expected_size)
                            self.assertEqual(source.format, "PNG")
                        with Image.open(os.path.join(region_dir, filename)) as region_image:
                            self.assertEqual(region_image.mode, "L")
                            self.assertEqual(region_image.size, expected_size)
                            self.assertEqual(region_image.format, "PNG")
                            values = set(np.unique(np.asarray(region_image)).tolist())
                            if (clip == make_synth_clips.SUBTITLE_HIGHRES_CLIP and
                                    make_synth_clips.SUBTITLE_HIGHRES_STATE_BY_FRAME[
                                        frame_id - 1] == "empty"):
                                self.assertEqual(values, {0})
                            else:
                                self.assertEqual(values, {0, 255})


if __name__ == "__main__":
    unittest.main()
