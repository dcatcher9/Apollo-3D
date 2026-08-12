#!/usr/bin/env python3
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import depth_coordinate_v2_contract as python_contract  # noqa: E402
import generate_depth_coordinate_v2_contract as generator  # noqa: E402
from depth_mapping_v2 import DIRECT_PARALLAX_SOURCE_U_LIMIT, MappingV2Config  # noqa: E402


class DepthCoordinateV2ContractTests(unittest.TestCase):
    def test_schema_pins_complete_manifest_semantics(self):
        expected_digest_by_schema = {
            1: "ae4210fe9ec8c058f3ad25924a1729a63e594b2b49f456e8f7483281074ee6f9",
            2: "d3a8d6235f992223394412bf01d21b7d5caf81ef9d00296a84bafb9511fccb28",
            3: "1e228924b4b9bb440abc8458d44e3d12935557fc4052df5dacc2059275bf08f5",
            4: "20e0db74ca44a5f92d2033a974abd7e7e3b69d7e706496e1a44e20fd15e09dcc",
            5: "98176528d8223c390301b857e953617841054670b8612b161d10a34fb5054509",
            6: "0accb3048852a6ec27e205462a76cc80e3d632472ec070246bf4f32178dd22d0",
            7: "62dddac0f66c6f6da812607da3b356891ec053095438ff95d124d1944cafd798",
            8: "cbbc94e4b39976042800ca6afa7555061568a1bbf5579487a30af17c91591073",
            9: "2f0b6213eca611f6c1de2ca7fd1cfa6297dfae5ffe7eee50d8e5a40a105a4494",
            10: "61d305a57c5b3a08e0b8550a9f2573339017c018c5d4b889fce978aae03a1699",
            11: "349774fcff9be7d1959bd95388c469451c17dbaa835df1b863061e39546e2117",
            12: "75486b54117c99baefa38ee6ff821941739da2abecd6fe4335cd571ed3d651da",
            13: "0bcb9598d3e1795d789ae807cddfcd85fa0c42a51f071874b28a48cd6ceb9161",
            14: "fdfda53e49ede50fc3408c7ecdafe7076a37b8468f5d828dc4a0d47e0656f458",
            15: "09f4eae02ddfd437dbf29116c6f7f4f5c754af40a7a9124edee3c071adfc8ed6",
            16: "ee61c7c2d02e4251cc485331b875477c165ea62c83177ae593e8bcbe1be5ca5a",
            17: "53c9e4cbb84e9d7f44d268454c92c33b01ffa8f62b93c6ce73fdfd7ca74485b1",
            18: "cad7fe8348bea0d5bafbbcfcc3a4b186f85e9ec0b9072c85f6471cdc171c8917",
            19: "ae7a329e9b9c8b03535bce41daf61248702d06ce61aede0d7a4940c5a40b6209",
            20: "a5cd4d6e2529b82f76f2b2673b3fab979720f7b6a7dca09702967f55ab440b06",
            21: "f2d736f76207df5bf4ac624e04c0c4f977278551463013edb85a70def5172b6d",
            22: "800a82953af68f903ddf386cc0e1f49cf6fb3a1f44e56339bc9ca2122e393849",
            23: "6a27019e526298fc708b400dbb9bfc66238c7f40978d0dcc254aaf27b7b8fa13",
            24: "af622986d64d49c3a084730c20838189fd579c1af54d7bb61f2c681647c155bb",
            25: "328d8f71424d6005ac0bd5025b4857efe95aa36fa683ab3c554a33b5309dcd05",
            26: "abb75ebabf6928c39771621da43b88f739fa3cf2ea83b18c02cdb20745aaa4b8",
            27: "6ecebd79f92c7bce3fbaf80c5c9e52fe2193e65baf1574f5b99fbf3a6f0a681e",
            28: "3946f21026d74d888463ca1c154a137fd0743a23f25a1bd94c3aa1d4d2348196",
            29: "95625e562bad0e079a2171cded69eabc1f172bf8531936e42945e1729e5fefa6",
            30: "2cd2eb2cb33c6d1d4f848c09ee8fb1086dbbe1d8fd6402c63fc091033d137414",
            31: "ee65f33be2c6be14efb97326320f2319d2f7fe4c1020e2968cfbc5b564300762",
            32: "05f89957e3475aa541363e6d0c3bdf0bf4e07258a14ac18fe67e954d91174d21",
            33: "8df1cfe9f1a7171e6bd17a8105dcd4cd3f7159c99af1a3f3154bde676859a59a",
            34: "1924d9f30129b2d0e24feb7bd5b7ca3e030a09fb44cee9b4bda291b7efeb4bf6",
            35: "d2086c46537f9f9b356fddad55a2eb9af205eda7c1406d416b5d50db777e342e",
            36: "2662745d69bac7b27881712427aa5eaebae78ce754b534147684995fb50e4ba2",
            37: "0e36d9d34ffac20c7f20686926b3450676123ea10d757d4c5f04d04c8225db67",
            38: "ba4940356916270d435961d28ff0b4a52442cd45f718a0dc48491927d3f8a58e",
            39: "b6c7b68e27009b35317f14d3aca1aba7f461d68e00dc2d5352195a815d75b81a",
            40: "5c7116be0004e33e24f150430d85e06b9eb66782b4fc3b059501e04093835d9e",
            41: "2f8a3bba1477cdcba58b1b28de8c0353b8f0aca13528b7eefb8f2e101078c51a",
            42: "b2932026a49f6dbd6ca2036f87aa095a2286850d283b52bd98328973f923e2a6",
            43: "026eea4c5e3280af0f7945d6f84fce7d54e74d009c6d89dafb6d725bf003aba1",
            44: "7ab9db96b7af454e10cfd004f878c0847d66913059005578e212474ef2a0a5fb",
            45: "8515cf7bc352c2e9e56e6a5fd9dad9802e1e7cd02f705fd8a957617c7ba94e9a",
        }
        contract = generator.load_contract()
        self.assertEqual(
            expected_digest_by_schema.get(contract["schema"]),
            generator.contract_digest(contract),
            "v2 semantics changed without a reviewed schema version",
        )
        self.assertEqual(generator.contract_tag(contract), 0xFBD3CDB1)
        self.assertEqual(
            generator.contract_tag_semantic_digest(contract),
            "fbd3cdb175e260327046e63a18a8278428a65a9b2103a23b5a6ceae7c5abcbcd",
        )
        self.assertEqual(
            contract["shader_implementation"]["source_closure_sha256"],
            "11bd8c0ab14d22caab83044e5f0d38cca10f5eef5df5d5e671cbedf64e256a1f",
        )
        self.assertTrue(generator.tag_is_finite_normal(generator.contract_tag(contract)))
        self.assertEqual(
            python_contract.CONTRACT_CANONICAL_SHA256,
            expected_digest_by_schema[contract["schema"]],
        )
        self.assertIn(
            f'contract_canonical_sha256 = "{python_contract.CONTRACT_CANONICAL_SHA256}"',
            generator.render_cpp(contract),
        )

    def test_manifest_pins_every_cross_language_layout(self):
        contract = generator.load_contract()
        self.assertEqual(contract["vector_width"], 4)
        self.assertEqual(
            [field["name"] for field in contract["constant_buffer"]["fields"]],
            [
                "raw_coordinate_scale", "collapse_abs_epsilon", "far_tau", "near_log_tau",
                "requested_gain", "max_horizontal_slope", "direct_container_limit",
                "convergence_curve_default",
            ],
        )
        self.assertEqual(
            [field["name"] for field in contract["frame_stats"]["fields"]],
            [
                "mean", "population_std", "minimum", "maximum", "valid_count",
                "texel_count", "valid", "reserved",
            ],
        )
        fields = contract["shadow_state"]["fields"]
        self.assertEqual(
            [field["name"] for field in fields],
            [
                "center", "inverse_scale", "convergence_curve", "container_scale",
                "calibration_revision", "frame_valid", "confirmed_cut_count", "contract_tag_bits",
                "camera_center_integrity_bits", "renderer_authorization_bits",
                "mapping_state_reserved_1", "mapping_state_reserved_2",
            ],
        )
        uint_fields = [field for field in fields if field["gpu_encoding"] == "uint_bits"]
        self.assertEqual(
            [(field["name"], field["initial"]) for field in uint_fields],
            [
                ("calibration_revision", 0),
                ("confirmed_cut_count", 0),
                ("contract_tag_bits", generator.CONTRACT_TAG_SENTINEL),
                ("camera_center_integrity_bits", 0),
                ("renderer_authorization_bits", 0),
                ("mapping_state_reserved_1", 0),
                ("mapping_state_reserved_2", 0),
            ],
        )
        self.assertEqual(fields[2]["initial"], 0.0)
        self.assertEqual(fields[3]["initial"], 1.0)

    def test_python_defaults_are_loaded_from_the_manifest(self):
        defaults = python_contract.CALIBRATED_DEFAULTS
        manifest_defaults = generator.load_contract()["calibrated_defaults"]
        for name in python_contract.CALIBRATED_DEFAULT_NAMES:
            self.assertEqual(getattr(defaults, name), manifest_defaults[name])
        mapping = MappingV2Config()
        self.assertEqual(
            mapping.raw_coordinate_scale,
            python_contract.MODEL_CALIBRATIONS[0].raw_coordinate_scale)
        self.assertEqual(mapping.collapse_abs_epsilon, defaults.collapse_abs_epsilon)
        self.assertEqual(mapping.far_tau, defaults.far_tau)
        self.assertEqual(mapping.near_log_tau, defaults.near_log_tau)
        self.assertEqual(mapping.pop_strength, defaults.reference_pop_strength)
        self.assertEqual(mapping.gain_per_pop, defaults.gain_per_pop)
        self.assertEqual(
            mapping.vertical_majorant_share, defaults.vertical_majorant_share)
        self.assertEqual(mapping.max_horizontal_slope, defaults.max_horizontal_slope)
        self.assertEqual(mapping.max_vertical_shear, defaults.max_vertical_shear)
        self.assertEqual(mapping.direct_container_limit, defaults.direct_container_limit)
        self.assertEqual(defaults.convergence_curve_default, 0.0)
        self.assertEqual(defaults.far_tau, 0.75)
        self.assertEqual(defaults.near_log_tau, 0.5)
        self.assertEqual(defaults.reference_pop_strength, 1.0)
        self.assertEqual(DIRECT_PARALLAX_SOURCE_U_LIMIT, defaults.direct_container_limit)

    def test_model_calibration_binds_identity_preprocess_and_exact_shape(self):
        calibration = python_contract.MODEL_CALIBRATIONS[0]
        self.assertEqual(calibration.depth_model, "depth_anything_v2_fp16")
        self.assertEqual(
            calibration.onnx_sha256,
            "2df6223f206b5164e21f664ace61dabeb9bb6a49b8b5a3e00510b4807d0f5b04")
        self.assertEqual(calibration.raw_coordinate_scale, 2.25)
        self.assertEqual(
            calibration.preprocess.profile, "apollo-dav2-area-hdr-srgb-imagenet-v1")
        self.assertEqual(calibration.preprocess.source_closure_schema, 2)
        self.assertEqual(calibration.preprocess.source_file, "rgb_to_nchw_cs.hlsl")
        self.assertEqual(calibration.preprocess.source_entrypoint, "main")
        self.assertEqual(calibration.preprocess.source_target, "cs_5_0")
        self.assertEqual(calibration.preprocess.source_compile_flags, 0x00008800)
        self.assertEqual(calibration.preprocess.source_macro_count, 0)
        self.assertEqual(
            calibration.preprocess.source_closure_sha256,
            "d765bf19eb493f302752b1881f62e5d07fa82a073cfba3811fed50f5a5264124")
        self.assertEqual(
            calibration.preprocess.source_closure_sha256,
            generator.shader_source_closure_sha256())
        self.assertEqual(
            calibration.calibration_id,
            "dav2-small-fp16-standardized-ui-shapes-v3",
        )
        self.assertEqual(
            calibration.calibrated_input_shapes,
            (
                (770, 434),
                (1022, 434),
                (1036, 434),
                (434, 770),
                (434, 1022),
                (434, 1036),
            ),
        )
        self.assertIs(
            python_contract.find_model_calibration(
                calibration.depth_model, calibration.depth_model_url,
                calibration.onnx_sha256),
            calibration)
        self.assertIsNone(python_contract.find_model_calibration(
            calibration.depth_model, calibration.depth_model_url, "0" * 64))

    def test_subtitle_ocr_contract_binds_model_profile_tensor_and_record_abis(self):
        ocr = python_contract.SUBTITLE_OCR
        self.assertEqual(ocr.schema, 5)
        self.assertEqual(ocr.logical_model, "ppocrv6_tiny_det_modelopt_fp16")
        self.assertEqual(
            ocr.asset_path,
            "models/ppocrv6_tiny_det_modelopt045_mixed_fp16_fp32io.onnx")
        self.assertEqual(
            ocr.artifact_onnx_sha256,
            "169a233ba0ff7cac27f8ec7dccb6a406e614b25b21fe6a5638c423bf2118bb44")
        self.assertEqual(
            ocr.source_onnx_sha256,
            "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8")
        self.assertIn("2ba1506c0380b8f0b03dd142459aac66d4421f6c", ocr.source_url)
        self.assertEqual((ocr.conversion_tool, ocr.conversion_version),
                         ("nvidia-modelopt", "0.45.0"))
        self.assertEqual(
            ocr.conversion_recipe,
            "nvidia-modelopt-autocast-fp16-keep-io-fp32-v1")
        self.assertEqual(
            ocr.conversion_calibration_profile,
            "apollo-live8-bottom960x160-v1")
        self.assertEqual(
            ocr.engine_recipe,
            "trt-strong-modelopt045-fp16-iofp32-tf32-fixed960x160-level5-v2")
        self.assertEqual(
            ocr.preprocess_profile,
            "apollo-ppocrv6-bottom-6x1-bgr-imagenet-v1")
        self.assertEqual(ocr.source_crop, "bottom-6:1")
        self.assertEqual(ocr.input_name, "x")
        self.assertEqual((ocr.input_dtype, ocr.input_layout), ("float32", "NCHW"))
        self.assertEqual(ocr.input_shape, (1, 3, 160, 960))
        self.assertEqual(ocr.input_channels, ("B", "G", "R"))
        self.assertEqual(ocr.imagenet_mean, (0.485, 0.456, 0.406))
        self.assertEqual(ocr.imagenet_std, (0.229, 0.224, 0.225))
        self.assertEqual(ocr.output_name, "fetch_name_0")
        self.assertEqual((ocr.output_dtype, ocr.output_layout), ("float32", "NCHW"))
        self.assertEqual(ocr.output_shape, (1, 1, 160, 960))
        self.assertEqual(
            (ocr.locator_max_width_numerator, ocr.locator_max_width_denominator,
             ocr.ocr_safe_row_top, ocr.ocr_safe_row_bottom,
             ocr.source_crop_aspect_width, ocr.source_crop_aspect_height),
            (9, 10, 24, 155, 6, 1))
        self.assertEqual(
            (ocr.text_join_gap_cells, ocr.ribbon_join_gap_cells),
            (4, 12))
        self.assertGreater(ocr.text_join_gap_cells, 0)
        self.assertLess(ocr.text_join_gap_cells, ocr.ribbon_join_gap_cells)
        self.assertGreaterEqual(
            ocr.ribbon_join_gap_cells,
            ocr.ribbon_structural_gap_min_cells)
        self.assertEqual((ocr.record_schema, ocr.record_tag), (3, 0x3852434F))
        self.assertEqual((ocr.record_word_count, ocr.raw_box_offset), (208, 16))
        self.assertEqual((ocr.final_box_offset, ocr.final_box_capacity), (144, 8))
        self.assertEqual((ocr.locator_schema, ocr.locator_tag), (8, 0x38524C53))
        self.assertEqual(
            (ocr.locator_word_count, ocr.locator_owner_offset,
             ocr.locator_pending_offset, ocr.locator_current_offset),
            (80, 32, 48, 64))

        contract = generator.load_contract(verify_shader_source_closure=False)
        cpp = generator.render_cpp(contract)
        hlsl = generator.render_hlsl(contract)
        for token in (
                'subtitle_ocr_model_name = "ppocrv6_tiny_det_modelopt_fp16"',
                'subtitle_ocr_asset_path = '
                '"models/ppocrv6_tiny_det_modelopt045_mixed_fp16_fp32io.onnx"',
                'subtitle_ocr_artifact_onnx_sha256 = '
                '"169a233ba0ff7cac27f8ec7dccb6a406e614b25b21fe6a5638c423bf2118bb44"',
                'subtitle_ocr_source_onnx_sha256 = '
                '"193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8"',
                'subtitle_ocr_conversion_tool = "nvidia-modelopt"',
                'subtitle_ocr_conversion_version = "0.45.0"',
                'subtitle_ocr_conversion_recipe = '
                '"nvidia-modelopt-autocast-fp16-keep-io-fp32-v1"',
                'subtitle_ocr_preprocess_profile = '
                '"apollo-ppocrv6-bottom-6x1-bgr-imagenet-v1"',
                'subtitle_ocr_input_name = "x"',
                'subtitle_ocr_output_name = "fetch_name_0"',
                'std::array<double, 3> subtitle_ocr_imagenet_mean {{0.485, 0.456, 0.406}}',
                'subtitle_ocr_output_width = 960u',
                'subtitle_ocr_record_tag = 0x3852434Fu',
                'subtitle_locator_state_tag = 0x38524C53u',
                'subtitle_ocr_safe_row_top = 24u',
                'subtitle_ocr_safe_row_bottom = 155u',
                'subtitle_ocr_crop_aspect_width = 6u',
                'subtitle_ocr_crop_aspect_height = 1u',
                'subtitle_ocr_text_join_gap_cells = 4u',
                'subtitle_ocr_ribbon_join_gap_cells = 12u',
                'subtitle_locator_max_width_numerator = 9u',
                'subtitle_locator_max_width_denominator = 10u',
                'constexpr bool subtitle_ocr_field_is_calibrated('):
            self.assertIn(token, cpp)
        for token in (
                '#define V2_OCR_INPUT_WIDTH 960u',
                '#define V2_OCR_OUTPUT_WIDTH 960u',
                '#define V2_OCR_IMAGENET_MEAN_B 0.485f',
                '#define V2_OCR_IMAGENET_STD_R 0.225f',
                '#define V2_OCR_RECORD_TAG 0x3852434Fu',
                '#define V2_SUBTITLE_LOCATOR_STATE_TAG 0x38524C53u',
                '#define V2_MODEL_CALIBRATED_SHAPE_COUNT 6u',
                '#define V2_MODEL_CALIBRATED_SHAPE_WIDTH_5 434u',
                '#define V2_MODEL_CALIBRATED_SHAPE_HEIGHT_5 1036u',
                '#define V2_OCR_SAFE_ROW_TOP 24u',
                '#define V2_OCR_SAFE_ROW_BOTTOM 155u',
                '#define V2_OCR_CROP_ASPECT_WIDTH 6u',
                '#define V2_OCR_CROP_ASPECT_HEIGHT 1u',
                '#define V2_OCR_TEXT_JOIN_GAP_CELLS 4u',
                '#define V2_OCR_RIBBON_JOIN_GAP_CELLS 12u',
                '#define V2_SUBTITLE_LOCATOR_MAX_WIDTH_NUMERATOR 9u',
                '#define V2_SUBTITLE_LOCATOR_MAX_WIDTH_DENOMINATOR 10u',
                'bool V2SubtitleOcrFieldIsCalibrated('):
            self.assertIn(token, hlsl)
        self.assertNotIn("subtitle_ocr_join_gap_cells", cpp)
        self.assertNotIn("V2_OCR_JOIN_GAP_CELLS", hlsl)

    def test_subtitle_ocr_dynamic_roi_covers_all_calibrated_fields(self):
        cases = (
            ((1920, 1080, 770, 434), (325, 430), 429),
            ((2560, 1080, 1022, 434), (289, 429), 427),
            ((3440, 1440, 1036, 434), (287, 429), 427),
            ((1080, 1920, 434, 770), (709, 768), 767),
            ((1080, 2560, 434, 1022), (961, 1020), 1019),
            ((1440, 3440, 434, 1036), (975, 1034), 1033),
        )
        for arguments, expected_roi, expected_ribbon_bottom in cases:
            with self.subTest(arguments=arguments):
                self.assertTrue(
                    python_contract.subtitle_ocr_field_is_calibrated(
                        arguments[2], arguments[3]))
                self.assertEqual(
                    python_contract.subtitle_ocr_dynamic_roi(*arguments), expected_roi)
                self.assertEqual(
                    python_contract.subtitle_ocr_ribbon_min_bottom(*arguments),
                    expected_ribbon_bottom)
                self.assertEqual(
                    python_contract.subtitle_ocr_project_row_ceil(
                        *arguments,
                        python_contract.SUBTITLE_OCR.ocr_safe_row_bottom -
                        python_contract.SUBTITLE_OCR.ribbon_bottom_tolerance_pixels),
                    expected_ribbon_bottom)
        self.assertFalse(python_contract.subtitle_ocr_field_is_calibrated(768, 432))
        self.assertIsNone(
            python_contract.subtitle_ocr_dynamic_roi(1920, 1080, 768, 432))
        self.assertIsNone(
            python_contract.subtitle_ocr_ribbon_min_bottom(1920, 1080, 768, 432))

        shader_root = (REPO / "src_assets" / "windows" / "assets" /
                       "shaders" / "directx")
        preprocess_source = (shader_root / "host_sbs_ocr_preprocess_cs.hlsl").read_text(
            encoding="utf-8")
        boxes_source = (shader_root / "host_sbs_ocr_boxes_cs.hlsl").read_text(
            encoding="utf-8")
        locator_source = (shader_root / "host_sbs_subtitle_locator_cs.hlsl").read_text(
            encoding="utf-8")
        for source in (preprocess_source, boxes_source, locator_source):
            self.assertIn(
                '#include "include/depth_coordinate_v2_contract.generated.hlsl"', source)
        for token in (
                "#define OCR_WIDTH V2_OCR_INPUT_WIDTH",
                "V2_OCR_IMAGENET_MEAN_B", "V2_OCR_IMAGENET_STD_R"):
            self.assertIn(token, preprocess_source)
        for token in (
                "#define OCR_WIDTH V2_OCR_OUTPUT_WIDTH",
                "#define OCR8_WORDS V2_OCR_RECORD_WORD_COUNT",
                "#define OCR8_TAG V2_OCR_RECORD_TAG"):
            self.assertIn(token, boxes_source)
        for token in (
                "MAX_LINES = V2_SUBTITLE_LOCATOR_RECTANGLE_CAPACITY",
                "V2_SUBTITLE_LOCATOR_STATE_SCHEMA",
                "V2_SUBTITLE_LOCATOR_CURRENT_OFFSET"):
            self.assertIn(token, locator_source)

    def test_subtitle_ocr_contract_rejects_model_tensor_profile_and_abi_drift(self):
        original = generator.load_contract(verify_shader_source_closure=False)
        mutations = {
            "artifact-hash": lambda value: value["subtitle_ocr"].update(
                {"artifact_onnx_sha256": "0" * 64}),
            "source-hash": lambda value: value["subtitle_ocr"].update(
                {"source_onnx_sha256": "0" * 64}),
            "conversion-recipe": lambda value: value["subtitle_ocr"].update(
                {"conversion_recipe": "unreviewed-fp16"}),
            "profile": lambda value: value["subtitle_ocr"].update(
                {"preprocess_profile": "generic-rgb"}),
            "channels": lambda value: value["subtitle_ocr"]["input_tensor"].update(
                {"channels": ["R", "G", "B"]}),
            "output-width": lambda value: value["subtitle_ocr"]["output_tensor"].update(
                {"shape": [1, 1, 160, 120]}),
            "safe-row": lambda value: value["subtitle_ocr"]["field_policy"].update(
                {"ocr_safe_row_top": 23}),
            "crop-aspect": lambda value: value["subtitle_ocr"]["field_policy"].update(
                {"source_crop_aspect_width": 7}),
            "locator-max-width": lambda value: value["subtitle_ocr"]["field_policy"].update(
                {"locator_max_width_numerator": 8}),
            "text-join-gap": lambda value: value["subtitle_ocr"]["field_policy"].update(
                {"text_join_gap_cells": 5}),
            "ribbon-join-gap": lambda value: value["subtitle_ocr"]["field_policy"].update(
                {"ribbon_join_gap_cells": 11}),
            "ocr-tag": lambda value: value["subtitle_ocr"]["ocr_record"].update(
                {"tag": 0}),
            "locator-words": lambda value: value["subtitle_ocr"]["locator_state"].update(
                {"word_count": 96}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(original)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "subtitle_ocr"):
                    generator.validate_contract(
                        changed, verify_shader_source_closure=False)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "contract.json"
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "subtitle_ocr"):
                        python_contract.load_contract(path)

    def test_generated_native_and_hlsl_contracts_are_current(self):
        result = subprocess.run(
            [sys.executable, str(generator.__file__), "--check"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_same_count_semantic_reorders_are_rejected_before_generation(self):
        original = generator.load_contract()
        for position, expected_error in enumerate((
                "constant_buffer physical field order",
                "frame_stats physical field order",
                "shadow_state physical field order")):
            with self.subTest(layout=position):
                reordered = copy.deepcopy(original)
                target = (
                    reordered["constant_buffer"]["fields"],
                    reordered["frame_stats"]["fields"],
                    reordered["shadow_state"]["fields"],
                )[position]
                left, right = dict(target[0]), dict(target[1])
                target[0] = {**right, "index": 0}
                target[1] = {**left, "index": 1}
                with self.assertRaisesRegex(ValueError, expected_error):
                    generator.validate_contract(reordered)

    def test_calibrated_value_change_changes_tag_and_generated_defaults(self):
        original = generator.load_contract()
        changed = copy.deepcopy(original)
        changed["calibrated_defaults"]["far_tau"] = 0.16
        generator.validate_contract(changed)
        self.assertTrue(generator.tag_is_finite_normal(generator.contract_tag(changed)))
        self.assertNotEqual(generator.contract_tag(original), generator.contract_tag(changed))
        self.assertNotEqual(generator.render_cpp(original), generator.render_cpp(changed))
        self.assertNotEqual(generator.render_hlsl(original), generator.render_hlsl(changed))
        self._assert_stale_check_fails(changed)

    def test_vertical_share_rejects_values_that_round_to_float32_endpoints(self):
        original = generator.load_contract()
        for value in (1.0e-50, 1.0 - 1.0e-12):
            with self.subTest(value=value):
                changed = copy.deepcopy(original)
                changed["calibrated_defaults"]["vertical_majorant_share"] = value
                with self.assertRaisesRegex(ValueError, "positive in float32"):
                    generator.validate_contract(changed)

    def test_shader_digest_is_independent_and_generation_converges(self):
        original = generator.load_contract()
        mutated_digest = copy.deepcopy(original)
        mutated_digest["shader_implementation"]["source_closure_sha256"] = "0" * 64

        # Shader bodies are authenticated by the independent closure digest.  The GPU tag omits
        # only that self-referential value, while the complete manifest digest still binds it.
        self.assertEqual(
            generator.contract_tag(original), generator.contract_tag(mutated_digest))
        self.assertEqual(
            generator.contract_tag_semantic_digest(original),
            generator.contract_tag_semantic_digest(mutated_digest))
        self.assertNotEqual(
            generator.contract_digest(original), generator.contract_digest(mutated_digest))
        self.assertEqual(
            generator.render_hlsl(original), generator.render_hlsl(mutated_digest))

        # Replacing the recorded digest with the closure of the generated shader leaves the HLSL
        # unchanged.  Therefore a second closure/generation pass is exactly idempotent.
        closure = generator.shader_source_closure_sha256(
            generator.PREPROCESS_SHADER_ROOT,
            generator.PARALLAX_V2_SHADER_SPECS,
        )
        rebound = copy.deepcopy(original)
        rebound["shader_implementation"]["source_closure_sha256"] = closure
        self.assertEqual(generator.render_hlsl(original), generator.render_hlsl(rebound))
        self.assertEqual(
            closure,
            generator.shader_source_closure_sha256(
                generator.PREPROCESS_SHADER_ROOT,
                generator.PARALLAX_V2_SHADER_SPECS,
            ),
        )

    def test_tag_derivation_rehashes_non_normal_float_bit_patterns(self):
        candidate = None
        first_prefix = None
        for offset in range(1, 4097):
            mutated = copy.deepcopy(generator.load_contract())
            mutated["calibrated_defaults"]["reference_pop_strength"] = 1.0 + offset * 1.0e-6
            prefix = int.from_bytes(
                hashlib.sha256(generator.canonical_bytes(mutated)).digest()[:4], "big")
            if not generator.tag_is_finite_normal(prefix):
                candidate, first_prefix = mutated, prefix
                break
        self.assertIsNotNone(candidate, "test search failed to find a non-normal SHA prefix")
        resolved = generator.contract_tag(candidate)
        self.assertTrue(generator.tag_is_finite_normal(resolved))
        self.assertNotEqual(resolved, first_prefix)

    def test_physically_reordered_or_same_index_fields_fail_closed(self):
        locations = (
            ("constant_buffer", "constant_buffer fields"),
            ("frame_stats", "frame_stats fields"),
            ("shadow_state", "manifest fields"),
        )
        for location, message in locations:
            for mutation in ("swap", "duplicate"):
                with self.subTest(location=location, mutation=mutation):
                    contract = copy.deepcopy(generator.load_contract())
                    target = contract[location]["fields"]
                    if mutation == "swap":
                        target[0], target[1] = target[1], target[0]
                    else:
                        target[1]["index"] = 0
                    with self.assertRaisesRegex(ValueError, message + ".*ordered"):
                        generator.validate_contract(contract)

    def test_generated_outputs_share_tag_and_cover_all_three_gpu_layouts(self):
        contract = generator.load_contract()
        tag = generator.contract_tag(contract)
        cpp = generator.CPP_TARGET.read_text(encoding="utf-8")
        hlsl = generator.HLSL_TARGET.read_text(encoding="utf-8")
        self.assertIn(f"contract_tag = 0x{tag:08X}u", cpp)
        self.assertIn(f"V2_CONTRACT_TAG 0x{tag:08X}u", hlsl)
        self.assertIn("struct alignas(16) constants_t", cpp)
        self.assertIn("cbuffer DepthCoordinateV2Constants : register(b1)", hlsl)
        self.assertIn("V2_FRAME_STATS_POPULATION_STD(value)", hlsl)
        self.assertIn("V2_STATE_CONTRACT_TAG_BITS(value)", hlsl)
        self.assertIn("V2_CONSTANT_WORD_COUNT 8u", hlsl)
        self.assertIn("V2_FRAME_STATS_WORD_COUNT 8u", hlsl)
        self.assertIn("V2_SHADOW_STATE_WORD_COUNT 12u", hlsl)
        self.assertIn("#define V2_MAX_VERTICAL_SHEAR 2.0f", hlsl)
        self.assertIn(
            "static const float v2_max_vertical_shear = V2_MAX_VERTICAL_SHEAR;", hlsl)
        self.assertNotIn("V2_STAGE_VALLEY_RATIO_MAX", hlsl)
        self.assertNotIn("V2_STAGE_CONVERGENCE_CURVE", hlsl)
        self.assertIn(
            f"V2_DIRECT_CONTAINER_LIMIT {contract['calibrated_defaults']['direct_container_limit']}f",
            hlsl,
        )
        self.assertIn("capture_provenance_schema = 3u", cpp)
        self.assertIn("contract_tag_semantic_sha256", cpp)
        self.assertIn("shader_source_closure_sha256", cpp)
        self.assertIn("source_closure_sha256", cpp)
        self.assertIn("model_calibration_supports_shape", cpp)
        for width, height in python_contract.MODEL_CALIBRATIONS[0].calibrated_input_shapes:
            self.assertIn(
                '{"dav2-small-fp16-standardized-ui-shapes-v3", '
                f'{width}u, {height}u}}',
                cpp,
            )

    def test_gpu_counter_updates_preserve_the_reserved_sentinel(self):
        source = (REPO / "src_assets" / "windows" / "assets" / "shaders" / "directx" /
                  "depth_coordinate_v2_state_resolve_cs.hlsl").read_text(encoding="utf-8")
        self.assertIn("return min(value, 0xfffffffdu) + 1u;", source)
        self.assertEqual(source.count("IncrementExactCounter(asuint("), 1)

    def test_preprocess_source_identity_covers_transitive_includes_and_specs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "root.hlsl").write_text(
                '#include "shared.hlsl"\nvoid main() {}\n', encoding="utf-8")
            include = root / "shared.hlsl"
            include.write_text("#define VALUE 1\n", encoding="utf-8")
            specs = (("root.hlsl", "main", "cs_5_0"),)
            initial = generator.shader_source_closure_sha256(root, specs)
            include.write_text("#define VALUE 2\n", encoding="utf-8")
            self.assertNotEqual(
                initial, generator.shader_source_closure_sha256(root, specs))
            include.write_text("#define VALUE 1\n", encoding="utf-8")
            self.assertEqual(initial, generator.shader_source_closure_sha256(root, specs))
            (root / "root.hlsl").write_bytes(
                b'#include "shared.hlsl"\rvoid main() {}\r')
            include.write_bytes(b"#define VALUE 1\r")
            self.assertEqual(initial, generator.shader_source_closure_sha256(root, specs))
            self.assertNotEqual(
                initial,
                generator.shader_source_closure_sha256(
                    root, (("root.hlsl", "different", "cs_5_0"),)))

            nested = root / "nested"
            nested.mkdir()
            (root / "root.hlsl").write_text(
                '#include "nested/parent.hlsl"\nvoid main() {}\n', encoding="utf-8")
            (nested / "parent.hlsl").write_text(
                '#include "shared.hlsl"\n', encoding="utf-8")
            (nested / "shared.hlsl").write_text(
                "#define VALUE 2\n", encoding="utf-8")
            (root / "shared.hlsl").write_text(
                "#define VALUE 1\n", encoding="utf-8")
            nested_digest = generator.shader_source_closure_sha256(root, specs)
            (root / "shared.hlsl").write_text(
                "#define VALUE 9\n", encoding="utf-8")
            self.assertEqual(
                nested_digest, generator.shader_source_closure_sha256(root, specs))
            (nested / "shared.hlsl").write_text(
                "#define VALUE 3\n", encoding="utf-8")
            self.assertNotEqual(
                nested_digest, generator.shader_source_closure_sha256(root, specs))

    def _assert_stale_check_fails(self, contract):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "changed.json"
            manifest.write_text(json.dumps(contract), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(generator.__file__), "--check", "--manifest", str(manifest)],
                cwd=REPO,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("stale generated contract", result.stderr)


if __name__ == "__main__":
    unittest.main()
