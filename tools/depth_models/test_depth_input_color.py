#!/usr/bin/env python3

import copy
import unittest
from unittest import mock

import numpy as np

import depth_input_color as color


EXPECTED_CONTRACT_SHA256 = (
    "939de61a85f001d14a5d3ca3e3b77e69ceb601fc30b7164b8b8eeb058f4dfc10"
)
MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

try:
    import torch
except (ImportError, OSError):
    torch = None


def denormalize(nchw):
    return nchw.transpose(1, 2, 0) * STD + MEAN


def independent_srgb_eotf(rgb):
    return np.where(
        rgb <= np.float32(0.04045),
        rgb / np.float32(12.92),
        np.power(
            (rgb + np.float32(0.055)) / np.float32(1.055),
            np.float32(2.4),
        ),
    ).astype(np.float32)


def independent_srgb_oetf(rgb):
    rgb = np.clip(rgb, np.float32(0.0), np.float32(1.0))
    return np.where(
        rgb <= np.float32(0.0031308),
        rgb * np.float32(12.92),
        np.float32(1.055) * np.power(
            rgb, np.float32(1.0 / 2.4)
        ) - np.float32(0.055),
    ).astype(np.float32)


class DepthInputColorContractTests(unittest.TestCase):
    def test_contract_is_authenticated_and_locks_canonical_white_anchors(self):
        contract = color.load_color_contract()
        anchors = contract[
            "simulated_sdr_in_windows_hdr"
        ]["windows_sdr_white_level"]["raw_anchors"]
        self.assertEqual(anchors, [1000, 2500, 3750, 5000, 6000])
        self.assertEqual(tuple(anchors), color.RAW_WHITE_ANCHORS)
        native = contract["native_pq_in_windows_hdr"]
        self.assertEqual(native["source_contract"], {
            "color_range": "limited",
            "color_primaries": "bt2020",
            "color_matrix": "bt2020-non-constant-luminance",
            "color_transfer": "smpte-st-2084",
            "minimum_component_bits": 10,
        })
        self.assertEqual(native["scrgb_reference_white_nits"], 80.0)
        self.assertEqual(
            native["pq_eotf"]["scale_policy"],
            "absolute-pq-nits-no-mastering-peak-rescale",
        )
        self.assertEqual(
            color.color_contract_sha256(contract), EXPECTED_CONTRACT_SHA256
        )

        stale = copy.deepcopy(contract)
        stale["resize"]["post_interpolation_quantization"] = "uint8"
        with self.assertRaisesRegex(RuntimeError, "color contract"):
            color.validate_color_contract(stale)
        wrong_type = copy.deepcopy(contract)
        wrong_type["schema"] = True
        with self.assertRaisesRegex(RuntimeError, "color contract"):
            color.validate_color_contract(wrong_type)

    def test_variants_bind_color_mode_white_math_and_contract_hash(self):
        sdr = color.sdr_input_variant()
        self.assertEqual(sdr["color_mode"], color.COLOR_MODE_SDR)
        self.assertIsNone(sdr["windows_sdr_white_level_raw"])
        self.assertIs(color.validate_input_variant(sdr), sdr)

        hashes = set()
        for raw in color.RAW_WHITE_ANCHORS:
            with self.subTest(raw=raw):
                variant = color.windows_hdr_input_variant(raw)
                self.assertEqual(variant["color_mode"], color.COLOR_MODE_HDR)
                self.assertEqual(
                    variant["windows_sdr_white_nits"], raw * 80.0 / 1000.0
                )
                self.assertEqual(variant["scrgb_white_scale"], raw / 1000.0)
                self.assertEqual(
                    variant["color_contract_sha256"], EXPECTED_CONTRACT_SHA256
                )
                self.assertIs(color.validate_input_variant(variant), variant)
                hashes.add(color.input_variant_sha256(variant))
        self.assertEqual(len(hashes), len(color.RAW_WHITE_ANCHORS))
        self.assertNotIn(color.input_variant_sha256(sdr), hashes)

        native = color.native_pq_input_variant()
        self.assertEqual(native["kind"], color.INPUT_KIND_NATIVE_PQ)
        self.assertEqual(native["color_mode"], color.COLOR_MODE_HDR)
        self.assertEqual(
            native["source_encoding"], color.NATIVE_PQ_SOURCE_ENCODING
        )
        self.assertEqual(
            native["capture_encoding"], color.HDR_CAPTURE_ENCODING
        )
        self.assertIsNone(native["windows_sdr_white_level_raw"])
        self.assertIsNone(native["windows_sdr_white_nits"])
        self.assertIsNone(native["scrgb_white_scale"])
        self.assertIs(color.validate_input_variant(native), native)
        self.assertNotIn(color.input_variant_sha256(native), hashes)
        self.assertNotEqual(
            color.input_variant_sha256(native),
            color.input_variant_sha256(sdr),
        )

    def test_variant_validation_fails_closed(self):
        for invalid in (999, 1000.0, True, None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(RuntimeError, "canonical raw anchor"):
                    color.windows_hdr_input_variant(invalid)

        for key, value in (
                ("windows_sdr_white_nits", 81.0),
                ("scrgb_white_scale", 1.1),
                ("color_contract_sha256", "0" * 64)):
            with self.subTest(key=key):
                stale = color.windows_hdr_input_variant(1000)
                stale[key] = value
                with self.assertRaisesRegex(RuntimeError, "not canonical"):
                    color.validate_input_variant(stale)

        unknown = color.sdr_input_variant()
        unknown["note"] = "not part of identity"
        with self.assertRaisesRegex(RuntimeError, "unknown fields"):
            color.validate_input_variant(unknown)
        wrong_type = color.sdr_input_variant()
        wrong_type["schema"] = True
        with self.assertRaisesRegex(RuntimeError, "not canonical"):
            color.validate_input_variant(wrong_type)
        stale_native = color.native_pq_input_variant()
        stale_native["windows_sdr_white_level_raw"] = 1000
        with self.assertRaisesRegex(RuntimeError, "not canonical"):
            color.validate_input_variant(stale_native)

    def test_canonical_hash_does_not_depend_on_mapping_insertion_order(self):
        variant = color.windows_hdr_input_variant(3750)
        reversed_variant = dict(reversed(list(variant.items())))
        self.assertEqual(
            color.input_variant_sha256(variant),
            color.input_variant_sha256(reversed_variant),
        )


class DepthInputColorNumpyTests(unittest.TestCase):
    def test_sdr_resize_preserves_fractional_unorm_samples(self):
        rgb = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.uint8)
        output = color.preprocess_rgb8_to_nchw(
            rgb, 3, 1, color.sdr_input_variant()
        )
        model_srgb = denormalize(output)
        expected_middle = np.float32(0.5 / 255.0)
        np.testing.assert_allclose(
            model_srgb[0, 1], expected_middle, rtol=0.0, atol=6e-8
        )
        self.assertNotEqual(float(model_srgb[0, 1, 0]), 0.0)
        self.assertNotEqual(float(model_srgb[0, 1, 0]), 1.0 / 255.0)

    def test_windows_hdr_rounds_to_fp16_before_bilinear_resize(self):
        rgb = np.array(
            [
                [[3, 47, 129], [31, 91, 201], [67, 143, 251]],
                [[11, 59, 137], [43, 107, 211], [79, 157, 255]],
            ],
            dtype=np.uint8,
        )
        variant = color.windows_hdr_input_variant(3750)
        real_resize = color._bilinear_resize_numpy
        with mock.patch.object(
                color, "_bilinear_resize_numpy", wraps=real_resize) as resize:
            color.preprocess_rgb8_to_nchw(rgb, 2, 3, variant)
        resized_source = resize.call_args.args[0]

        decoded = rgb.astype(np.float32) / np.float32(255.0)
        expected = (
            independent_srgb_eotf(decoded) * np.float32(3.75)
        ).astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(resized_source, expected)
        self.assertTrue(np.any(
            resized_source != independent_srgb_eotf(decoded) * np.float32(3.75)
        ))

    def test_windows_hdr_matches_shader_operation_order(self):
        rgb = np.array([[[255, 128, 16]]], dtype=np.uint8)
        output = color.preprocess_rgb8_to_nchw_numpy(
            rgb, 1, 1, color.windows_hdr_input_variant(6000)
        )

        decoded = rgb.astype(np.float32) / np.float32(255.0)
        scrgb = (
            independent_srgb_eotf(decoded) * np.float32(6.0)
        ).astype(np.float16).astype(np.float32)
        scrgb = np.maximum(scrgb, np.float32(0.0))
        luminance = np.sum(
            scrgb * np.array((0.2126, 0.7152, 0.0722), dtype=np.float32),
            axis=2,
            keepdims=True,
            dtype=np.float32,
        )
        compressed = scrgb / (np.float32(1.0) + luminance)
        compressed /= np.maximum(
            np.max(compressed, axis=2, keepdims=True), np.float32(1.0)
        )
        expected_srgb = independent_srgb_oetf(compressed)
        expected = ((expected_srgb - MEAN) / STD).transpose(2, 0, 1)
        np.testing.assert_allclose(output, expected, rtol=0.0, atol=2e-6)

    def test_raw_white_anchor_changes_hdr_result_but_not_shape_or_type(self):
        rgb = np.array([[[255, 255, 255]]], dtype=np.uint8)
        low = color.preprocess_rgb8_to_nchw(
            rgb, 4, 3, color.windows_hdr_input_variant(1000)
        )
        high = color.preprocess_rgb8_to_nchw(
            rgb, 4, 3, color.windows_hdr_input_variant(6000)
        )
        self.assertEqual(low.shape, (3, 3, 4))
        self.assertEqual(low.dtype, np.float32)
        self.assertTrue(low.flags.c_contiguous)
        self.assertGreater(float(high.mean()), float(low.mean()))

    def test_numpy_preprocessing_is_repeatable(self):
        rgb = np.random.default_rng(17).integers(
            0, 256, size=(7, 11, 3), dtype=np.uint8
        )
        for variant in (
                color.sdr_input_variant(),
                color.windows_hdr_input_variant(2500)):
            with self.subTest(kind=variant["kind"]):
                first = color.preprocess_rgb8_to_nchw(rgb, 6, 5, variant)
                second = color.preprocess_rgb8_to_nchw(rgb, 6, 5, variant)
                np.testing.assert_array_equal(first, second)
                self.assertTrue(np.isfinite(first).all())

    def test_native_pq_scrgb_matches_production_hdr_operation_order(self):
        scrgb = np.array(
            [[[-0.25, 0.5, 4.0, 0.0], [1.5, 0.25, 0.125, 1.0]]],
            dtype=np.float16,
        )
        variant = color.native_pq_input_variant()
        real_resize = color._bilinear_resize_numpy
        with mock.patch.object(
                color, "_bilinear_resize_numpy", wraps=real_resize) as resize:
            output = color.preprocess_scrgb_f16_to_nchw(
                scrgb, 3, 1, variant
            )
        resized_source = resize.call_args.args[0]
        self.assertEqual(resized_source.dtype, np.float32)
        np.testing.assert_array_equal(
            resized_source, scrgb[..., :3].astype(np.float32)
        )

        resized = real_resize(resized_source, 3, 1)
        mapped = np.maximum(resized, np.float32(0.0))
        luminance = np.maximum(np.sum(
            mapped * np.array((0.2126, 0.7152, 0.0722), dtype=np.float32),
            axis=2, keepdims=True, dtype=np.float32,
        ), np.float32(0.0))
        mapped /= np.float32(1.0) + luminance
        mapped /= np.maximum(
            np.max(mapped, axis=2, keepdims=True), np.float32(1.0)
        )
        expected_srgb = independent_srgb_oetf(mapped)
        expected = ((expected_srgb - MEAN) / STD).transpose(2, 0, 1)
        np.testing.assert_allclose(output, expected, rtol=0.0, atol=2e-6)
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(output.flags.c_contiguous)

    def test_native_pq_scrgb_is_repeatable_and_rgb_rgba_are_equivalent(self):
        rgb = np.random.default_rng(71).uniform(
            -0.5, 12.0, size=(7, 11, 3)
        ).astype(np.float16)
        rgba = np.concatenate((
            rgb,
            np.random.default_rng(72).uniform(
                -100.0, 100.0, size=(7, 11, 1)
            ).astype(np.float16),
        ), axis=2)
        variant = color.native_pq_input_variant()
        first = color.preprocess_scrgb_f16_to_nchw(rgb, 6, 5, variant)
        second = color.preprocess_scrgb_f16_to_nchw(rgba, 6, 5, variant)
        third = color.preprocess_scrgb_f16_to_nchw(rgb, 6, 5, variant)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, third)

    def test_native_pq_scrgb_rejects_lossy_or_ambiguous_inputs(self):
        variant = color.native_pq_input_variant()
        with self.assertRaisesRegex(TypeError, "FP16 scRGB"):
            color.preprocess_scrgb_f16_to_nchw(
                np.zeros((2, 2, 3), dtype=np.float32), 2, 2, variant
            )
        with self.assertRaisesRegex(ValueError, "HxWx3 or HxWx4"):
            color.preprocess_scrgb_f16_to_nchw(
                np.zeros((2, 2, 2), dtype=np.float16), 2, 2, variant
            )
        nonfinite = np.zeros((2, 2, 3), dtype=np.float16)
        nonfinite[0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            color.preprocess_scrgb_f16_to_nchw(
                nonfinite, 2, 2, variant
            )
        with self.assertRaisesRegex(TypeError, "native PQ input variant"):
            color.preprocess_scrgb_f16_to_nchw(
                np.zeros((2, 2, 3), dtype=np.float16), 2, 2,
                color.sdr_input_variant(),
            )
        with self.assertRaisesRegex(TypeError, "native PQ input requires"):
            color.preprocess_rgb8_to_nchw(
                np.zeros((2, 2, 3), dtype=np.uint8), 2, 2, variant
            )

    def test_rejects_ambiguous_sources_and_invalid_output_sizes(self):
        variant = color.sdr_input_variant()
        with self.assertRaisesRegex(TypeError, "RGB uint8"):
            color.preprocess_rgb8_to_nchw(
                np.zeros((2, 2, 3), dtype=np.float32), 2, 2, variant
            )
        with self.assertRaisesRegex(ValueError, "HxWx3"):
            color.preprocess_rgb8_to_nchw(
                np.zeros((2, 2, 4), dtype=np.uint8), 2, 2, variant
            )
        for width, height in ((0, 2), (2, -1), (2.0, 2), (True, 2)):
            with self.subTest(width=width, height=height):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    color.preprocess_rgb8_to_nchw(
                        np.zeros((2, 2, 3), dtype=np.uint8),
                        width,
                        height,
                        variant,
                    )


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class DepthInputColorTorchTests(unittest.TestCase):
    def test_torch_matches_numpy_and_is_repeatable(self):
        rgb = np.random.default_rng(29).integers(
            0, 256, size=(9, 13, 3), dtype=np.uint8
        )
        tensor = torch.from_numpy(rgb.copy())
        for variant in (
                color.sdr_input_variant(),
                color.windows_hdr_input_variant(5000)):
            with self.subTest(kind=variant["kind"]):
                expected = color.preprocess_rgb8_to_nchw_numpy(
                    rgb, 7, 6, variant
                )
                first = color.preprocess_rgb8_to_nchw(
                    tensor, 7, 6, variant
                )
                second = color.preprocess_rgb8_to_nchw_torch(
                    tensor, 7, 6, variant
                )
                self.assertEqual(first.dtype, torch.float32)
                self.assertEqual(tuple(first.shape), (3, 6, 7))
                self.assertTrue(first.is_contiguous())
                self.assertTrue(torch.equal(first, second))
                np.testing.assert_allclose(
                    first.cpu().numpy(), expected, rtol=0.0, atol=3e-6
                )

    def test_torch_rejects_non_uint8_and_wrong_channel_order(self):
        variant = color.sdr_input_variant()
        with self.assertRaisesRegex(TypeError, "RGB uint8"):
            color.preprocess_rgb8_to_nchw(
                torch.zeros((2, 2, 3), dtype=torch.float32), 2, 2, variant
            )
        with self.assertRaisesRegex(ValueError, "HxWx3"):
            color.preprocess_rgb8_to_nchw(
                torch.zeros((3, 2, 2), dtype=torch.uint8), 2, 2, variant
            )

    def test_torch_native_pq_scrgb_matches_numpy(self):
        scrgb = np.random.default_rng(73).uniform(
            -0.75, 16.0, size=(9, 13, 4)
        ).astype(np.float16)
        variant = color.native_pq_input_variant()
        expected = color.preprocess_scrgb_f16_to_nchw_numpy(
            scrgb, 7, 6, variant
        )
        tensor = torch.from_numpy(scrgb.copy())
        first = color.preprocess_scrgb_f16_to_nchw(
            tensor, 7, 6, variant
        )
        second = color.preprocess_scrgb_f16_to_nchw_torch(
            tensor, 7, 6, variant
        )
        self.assertEqual(first.dtype, torch.float32)
        self.assertEqual(tuple(first.shape), (3, 6, 7))
        self.assertTrue(first.is_contiguous())
        self.assertTrue(torch.equal(first, second))
        np.testing.assert_allclose(
            first.cpu().numpy(), expected, rtol=0.0, atol=3e-6
        )

    def test_torch_native_pq_scrgb_rejects_invalid_input(self):
        variant = color.native_pq_input_variant()
        with self.assertRaisesRegex(TypeError, "FP16 scRGB"):
            color.preprocess_scrgb_f16_to_nchw(
                torch.zeros((2, 2, 3), dtype=torch.float32), 2, 2, variant
            )
        nonfinite = torch.zeros((2, 2, 3), dtype=torch.float16)
        nonfinite[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            color.preprocess_scrgb_f16_to_nchw(
                nonfinite, 2, 2, variant
            )

    def test_cuda_matches_numpy_within_sampler_tolerance(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        rgb = np.random.default_rng(41).integers(
            0, 256, size=(31, 47, 3), dtype=np.uint8
        )
        tensor = torch.from_numpy(rgb.copy()).cuda()
        for variant in (
                color.sdr_input_variant(),
                color.windows_hdr_input_variant(1000),
                color.windows_hdr_input_variant(6000)):
            with self.subTest(
                    kind=variant["kind"],
                    raw=variant["windows_sdr_white_level_raw"]):
                expected = color.preprocess_rgb8_to_nchw_numpy(
                    rgb, 29, 23, variant
                )
                first = color.preprocess_rgb8_to_nchw(
                    tensor, 29, 23, variant
                )
                second = color.preprocess_rgb8_to_nchw(
                    tensor, 29, 23, variant
                )
                self.assertTrue(torch.equal(first, second))
                np.testing.assert_allclose(
                    first.cpu().numpy(), expected, rtol=0.0, atol=2e-4
                )


if __name__ == "__main__":
    unittest.main()
