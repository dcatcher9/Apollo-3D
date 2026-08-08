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
        }
        contract = generator.load_contract()
        self.assertEqual(
            expected_digest_by_schema.get(contract["schema"]),
            generator.contract_digest(contract),
            "v2 semantics changed without a reviewed schema version",
        )
        self.assertEqual(generator.contract_tag(contract), 0x1A28FDA1)
        self.assertEqual(
            generator.contract_tag_semantic_digest(contract),
            "1a28fda150d9ab42c7b6bc866168c6604a168b48044a563c657f95f44dbd3cdb",
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
