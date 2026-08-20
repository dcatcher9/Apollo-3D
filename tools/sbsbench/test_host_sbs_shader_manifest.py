#!/usr/bin/env python3
import copy
import json
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import generate_host_sbs_shader_manifest as generator  # noqa: E402


class HostSbsShaderManifestTests(unittest.TestCase):
    def test_all_generated_views_are_exact_manifest_projections(self):
        manifest = generator.load_manifest()
        self.assertEqual(
            generator.CPP_TARGET.read_text(encoding="utf-8"),
            generator.render_cpp(manifest),
        )
        self.assertEqual(
            generator.PYTHON_TARGET.read_text(encoding="utf-8"),
            generator.render_python(manifest),
        )
        documentation = generator.DOC_TARGET.read_text(encoding="utf-8")
        self.assertEqual(
            documentation,
            generator.render_documentation(manifest, documentation),
        )
        dvc_contract = generator.DVC_MANIFEST.read_text(encoding="utf-8")
        self.assertEqual(
            dvc_contract,
            generator.render_dvc_manifest(manifest, dvc_contract),
        )
        self.assertEqual(generator.main(["--check"]), 0)

    def test_generated_dvc_projection_preserves_non_shader_semantics(self):
        manifest = generator.load_manifest()
        original = json.loads(generator.DVC_MANIFEST.read_text(encoding="utf-8"))
        projected = json.loads(generator.render_dvc_manifest(
            manifest, json.dumps(original)
        ))
        original_without_projection = copy.deepcopy(original)
        projected_without_projection = copy.deepcopy(projected)
        del original_without_projection["shader_implementation"]
        del projected_without_projection["shader_implementation"]
        for calibration in original_without_projection["model_calibrations"]:
            for key in (
                    "source_closure_schema", "source_file", "source_entrypoint",
                    "source_target", "source_compile_flags", "source_macro_count",
                    "source_closure_sha256"):
                del calibration["preprocess"][key]
        for calibration in projected_without_projection["model_calibrations"]:
            for key in (
                    "source_closure_schema", "source_file", "source_entrypoint",
                    "source_target", "source_compile_flags", "source_macro_count",
                    "source_closure_sha256"):
                del calibration["preprocess"][key]
        self.assertEqual(original_without_projection, projected_without_projection)

    def test_group_taxonomy_and_spec_coverage_are_closed(self):
        manifest = generator.load_manifest()
        self.assertEqual(
            tuple(group["name"] for group in manifest["groups"]),
            generator.EXPECTED_GROUP_NAMES,
        )
        referenced = {
            name
            for group in manifest["groups"]
            for name in group["specs"]
        }
        self.assertEqual(
            referenced,
            {spec["name"] for spec in manifest["specs"]},
        )

        reordered = copy.deepcopy(manifest)
        reordered["groups"][0], reordered["groups"][1] = (
            reordered["groups"][1], reordered["groups"][0]
        )
        with self.assertRaisesRegex(ValueError, "canonical order"):
            generator.validate_manifest(reordered)

        orphaned = copy.deepcopy(manifest)
        orphaned["specs"].append({
            "name": "unreferenced_root",
            "source_file": "buffer_to_tex_cs.hlsl",
            "source_entrypoint": "main",
            "source_target": "cs_5_0",
        })
        with self.assertRaisesRegex(ValueError, "unreferenced shader specs"):
            generator.validate_manifest(orphaned)

    def test_generated_python_names_keep_specs_and_groups_distinct(self):
        manifest = generator.load_manifest()
        python_view = generator.render_python(manifest)
        self.assertIn(
            "PARALLAX_V2_LIVE_RENDERER = ShaderSpec(",
            python_view,
        )
        self.assertIn(
            "PARALLAX_V2_LIVE_RENDERER_GROUP = ClosureGroup(",
            python_view,
        )

        colliding = copy.deepcopy(manifest)
        trace_spec = next(
            spec for spec in colliding["specs"]
            if spec["name"] == "host_sbs_gpu_trace"
        )
        trace_spec["name"] = "gpu_trace_group"
        generator.closure_group(colliding, "gpu_trace")["specs"] = [
            "gpu_trace_group",
        ]
        with self.assertRaisesRegex(ValueError, "generated Python identifiers collide"):
            generator.validate_manifest(colliding)

    def test_order_or_pin_change_requires_closure_refresh(self):
        manifest = generator.load_manifest()
        reordered = copy.deepcopy(manifest)
        producer = generator.closure_group(reordered, "parallax_v2_producer")
        producer["specs"][0], producer["specs"][1] = (
            producer["specs"][1], producer["specs"][0]
        )
        with self.assertRaisesRegex(ValueError, "source closure pin is stale"):
            generator.validate_group_pins(reordered)

        stale_pin = copy.deepcopy(manifest)
        generator.closure_group(stale_pin, "gpu_trace")[
            "source_closure_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source closure pin is stale"):
            generator.validate_group_pins(stale_pin)


if __name__ == "__main__":
    unittest.main()
