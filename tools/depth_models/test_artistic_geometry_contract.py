import copy
import unittest

import artistic_geometry_contract as geometry


class ArtisticGeometryContractTests(unittest.TestCase):
    def test_ultrawide_and_portrait_model_dims_use_resolved_aspect_cap(self):
        self.assertEqual(
            geometry.aspect_aligned_dims(
                8000, 1000, depth_short_side=432, depth_max_aspect=4.0
            ),
            (1008, 252),
        )
        self.assertEqual(
            geometry.aspect_aligned_dims(
                1000, 8000, depth_short_side=432, depth_max_aspect=4.0
            ),
            (252, 1008),
        )
        self.assertEqual(
            geometry.aspect_aligned_dims(
                8000, 1000, depth_short_side=280, depth_max_aspect=2.0
            ),
            (560, 280),
        )

    def test_tuple_binds_configured_preprocessing_and_rejects_stale_dims(self):
        scale_x, scale_y = geometry.source_content_scales(
            8000, 1000, 3840, 1080
        )
        row = {
            "source_width": 8000, "source_height": 1000,
            "eye_width": 3840, "eye_height": 1080,
            "content_scale_x": scale_x, "content_scale_y": scale_y,
            "disparity_raster_width": 3840,
            "disparity_raster_height": 1080,
        }
        value = geometry.geometry_tuple(
            row, depth_short_side=280, depth_max_aspect=2.0
        )
        self.assertEqual(
            (value["model_input_width"], value["model_input_height"]),
            (560, 280),
        )
        self.assertEqual(value["depth_short_side"], 280)
        self.assertEqual(value["depth_max_aspect"], 2.0)
        geometry.validate_geometry_tuple(value)

        stale = copy.deepcopy(value)
        stale["model_input_height"] = 266
        with self.assertRaisesRegex(RuntimeError, "stale model-input"):
            geometry.validate_geometry_tuple(stale)

    def test_geometry_admits_exact_sdr_and_hdr_runtime_color_modes(self):
        row = {
            "source_width": 1280, "source_height": 720,
            "eye_width": 1920, "eye_height": 1080,
            "content_scale_x": 1.0, "content_scale_y": 1.0,
            "disparity_raster_width": 1920,
            "disparity_raster_height": 1080,
        }
        for color_mode in (geometry.COLOR_MODE_SDR, geometry.COLOR_MODE_HDR):
            with self.subTest(color_mode=color_mode):
                value = geometry.geometry_tuple(row, color_mode=color_mode)
                self.assertEqual(value["color_mode"], color_mode)
                geometry.validate_geometry_tuple(value)

        stale = geometry.geometry_tuple(row)
        stale["color_mode"] = "hdr-pq-encoded"
        with self.assertRaisesRegex(RuntimeError, "not validated"):
            geometry.validate_geometry_tuple(stale)


if __name__ == "__main__":
    unittest.main()
