#!/usr/bin/env python3

import unittest

import video_color_contract as color


class VideoColorContractTests(unittest.TestCase):
    def test_accepts_probed_eight_bit_sdr(self):
        result = color.classify_sdr_stream({
            "pix_fmt": "yuv420p",
            "color_transfer": "bt709",
            "color_primaries": "bt709",
            "color_space": "bt709",
        })
        self.assertEqual(result["source_bit_depth"], 8)
        self.assertEqual(result["admission"], "probed-no-hdr-signals")

    def test_rejects_pq_hlg_and_wide_gamut(self):
        for stream in (
                {"pix_fmt": "yuv420p10le", "color_transfer": "smpte2084"},
                {"pix_fmt": "yuv420p10le", "color_transfer": "arib-std-b67"},
                {"pix_fmt": "yuv420p", "color_primaries": "bt2020"}):
            with self.subTest(stream=stream):
                with self.assertRaisesRegex(RuntimeError, "HDR|wide-gamut"):
                    color.classify_sdr_stream(stream)

    def test_rejects_unreviewed_high_bit_depth_but_records_override(self):
        stream = {
            "pix_fmt": "yuv420p10le",
            "color_transfer": "bt709",
            "color_primaries": "bt709",
        }
        with self.assertRaisesRegex(RuntimeError, "10-bit"):
            color.classify_sdr_stream(stream)
        result = color.classify_sdr_stream(stream, "sdr")
        self.assertEqual(result["admission"], "user-reviewed-sdr")
        self.assertEqual(result["source_bit_depth"], 10)

    def test_recognizes_common_high_bit_depth_pixel_format_families(self):
        expected = {
            "gray10le": 10,
            "gray12be": 12,
            "gray16le": 16,
            "yuv422p12le": 12,
            "gbrap14be": 14,
            "gbrpf32le": 32,
            "bayer_bggr16le": 16,
            "rgb48le": 16,
            "bgr48be": 16,
            "rgba64le": 16,
            "ayuv64be": 16,
            "xyz12le": 12,
            "p010le": 10,
            "p012le": 12,
            "p016le": 16,
            "p210le": 10,
            "p416be": 16,
            "x2rgb10le": 10,
            "y212le": 12,
            "v210": 10,
            "nv20le": 10,
        }
        for pixel_format, bit_depth in expected.items():
            with self.subTest(pixel_format=pixel_format):
                stream = {
                    "pix_fmt": pixel_format,
                    "color_transfer": "bt709",
                    "color_primaries": "bt709",
                }
                self.assertEqual(color.pixel_bit_depth(stream), bit_depth)
                with self.assertRaisesRegex(RuntimeError, f"{bit_depth}-bit"):
                    color.classify_sdr_stream(stream)

    def test_uses_conservative_depth_when_raw_sample_metadata_disagrees(self):
        self.assertEqual(color.pixel_bit_depth({
            "pix_fmt": "yuv420p",
            "bits_per_raw_sample": "12",
        }), 12)
        self.assertEqual(color.pixel_bit_depth({
            "pix_fmt": "p016le",
            "bits_per_raw_sample": "8",
        }), 16)
        self.assertEqual(color.pixel_bit_depth({
            "pix_fmt": "yuv420p10le",
            "bits_per_raw_sample": 12,
        }), 12)
        self.assertEqual(color.pixel_bit_depth({
            "pix_fmt": "yuv420p",
            "bits_per_raw_sample": True,
        }), 8)

    def test_manual_override_cannot_hide_explicit_hdr(self):
        with self.assertRaisesRegex(RuntimeError, "HDR transfer"):
            color.classify_sdr_stream({
                "pix_fmt": "yuv420p10le",
                "color_transfer": "smpte2084",
            }, "sdr")


if __name__ == "__main__":
    unittest.main()
