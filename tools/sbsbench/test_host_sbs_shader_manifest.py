#!/usr/bin/env python3
import copy
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


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

    def test_subtitle_cohort_experiment_is_separate_from_production(self):
        manifest = generator.load_manifest()
        research = generator.closure_group(manifest, "subtitle_cohort_research")
        members = set(research["specs"])
        self.assertEqual(members, {
            "host_sbs_subtitle_cohort_prepare", "host_sbs_subtitle_cohort_members",
            "host_sbs_subtitle_cohort_filter", "host_sbs_subtitle_cohort_snapshot",
        })
        for group in manifest["groups"]:
            if group is not research and group["name"] != "subtitle_cohort_prototype_research":
                self.assertTrue(members.isdisjoint(group["specs"]))

    def test_opt_in_diagnostic_groups_pin_the_exact_loader_specs(self):
        manifest = generator.load_manifest()
        prototype = generator.group_specs(manifest, "subtitle_cohort_prototype_research")
        cohort = generator.group_specs(manifest, "subtitle_cohort_research")
        self.assertEqual(tuple(prototype), tuple(cohort)[:3])
        self.assertEqual(tuple(generator.group_specs(manifest, "subtitle_inner_colour_research")), tuple(
            ("host_sbs_subtitle_inner_colour_global_validation_cs.hlsl", entry, "cs_5_0")
            for entry in ("palette_main", "emit_main", "clear_main", "accumulate_main", "validate_main")
        ))
        self.assertEqual(tuple(generator.group_specs(manifest, "subtitle_inner_components_research")), tuple(
            ("host_sbs_subtitle_seeded_components_cs.hlsl", entry, "cs_5_0")
            for entry in ("init_main", "union_main", "reduce_main", "resolve_main", "dilate_main", "dilate_box_main")
        ))
        generator.validate_group_pins(manifest)
        for name in ("subtitle_cohort_prototype_research", "subtitle_inner_colour_research", "subtitle_inner_components_research"):
            stale = copy.deepcopy(manifest)
            generator.closure_group(stale, name)["source_closure_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "source closure pin is stale"):
                generator.validate_group_pins(stale)

    def test_opt_in_diagnostic_specs_never_join_production_groups(self):
        manifest = generator.load_manifest()
        names = ("subtitle_cohort_prototype_research", "subtitle_cohort_research", "subtitle_inner_colour_research", "subtitle_inner_components_research", "subtitle_matte_guard_research")
        diagnostic_specs = set().union(*(set(generator.closure_group(manifest, name)["specs"]) for name in names))
        for group in manifest["groups"]:
            if group["name"] not in names:
                self.assertTrue(diagnostic_specs.isdisjoint(group["specs"]))

    def test_final_matte_guard_is_separate_from_cohort_and_production(self):
        manifest = generator.load_manifest()
        guard = generator.closure_group(manifest, "subtitle_matte_guard_research")
        self.assertEqual(guard["specs"], ["host_sbs_subtitle_matte_guard"])
        for group in manifest["groups"]:
            if group is not guard:
                self.assertNotIn("host_sbs_subtitle_matte_guard", group["specs"])

    def test_optional_current_mask_holes_do_not_change_the_mandatory_producer(self):
        manifest = generator.load_manifest()
        holes = generator.closure_group(manifest, "current_mask_holes")
        entries = ("clear_main", "capture_unit_main", "prepare_holes_main", "freeze_final_main",
                   "init_main", "union_main", "reduce_main", "publish_main", "counter_main")
        self.assertEqual(tuple(generator.group_specs(manifest, "current_mask_holes")), tuple(
            ("current_outline/hole/host_sbs_current_mask_holes_cs.hlsl", entry, "cs_5_0")
            for entry in entries
        ))
        for group in manifest["groups"]:
            if group is not holes:
                self.assertTrue(set(holes["specs"]).isdisjoint(group["specs"]))
        original = json.loads(generator.DVC_MANIFEST.read_text(encoding="utf-8"))
        projected = json.loads(generator.render_dvc_manifest(manifest, json.dumps(original)))
        self.assertEqual(original["shader_implementation"], projected["shader_implementation"])

    def test_optional_stroke_coverage_has_separate_specs_and_preserves_producer_identity(self):
        manifest = generator.load_manifest()
        coverage = generator.closure_group(manifest, "current_stroke_coverage")
        self.assertEqual(tuple(generator.group_specs(manifest, "current_stroke_coverage")), tuple(
            ("current_outline/hole/host_sbs_current_stroke_coverage_cs.hlsl", entry, "cs_5_0")
            for entry in ("clear_offers_main", "prepare_main", "support_main")
        ))
        for group in manifest["groups"]:
            if group is not coverage:
                self.assertTrue(set(coverage["specs"]).isdisjoint(group["specs"]))
        original = json.loads(generator.DVC_MANIFEST.read_text(encoding="utf-8"))
        projected = json.loads(generator.render_dvc_manifest(manifest, json.dumps(original)))
        self.assertEqual(original["shader_implementation"], projected["shader_implementation"])

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


class HostSbsShaderManifestWriteTests(unittest.TestCase):
    def test_identical_refresh_preserves_mtime_and_exact_utf8_lf_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "view.h"
            rendered = "// caf\u00e9\n#define VALUE 1\n"
            path.write_bytes(rendered.encode("utf-8"))
            os.utime(path, ns=(1600000000000000000, 1600000000000000000))
            before = path.stat().st_mtime_ns
            self.assertTrue(generator._write_or_check(path, rendered, False))
            self.assertTrue(generator._write_or_check(path, rendered, False))
            self.assertTrue(generator._write_or_check(path, rendered, True))
            self.assertEqual(path.stat().st_mtime_ns, before)
            self.assertEqual(path.read_bytes(), rendered.encode("utf-8"))

    def test_changed_and_missing_outputs_write_exact_utf8_lf_bytes(self):
        rendered = "// caf\u00e9\n#define VALUE 2\n"
        for initial in (None, b"old\r\n", rendered.replace("\n", "\r\n").encode("utf-8")):
            with self.subTest(initial=initial), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "nested" / "view.h"
                if initial is not None:
                    path.parent.mkdir()
                    path.write_bytes(initial)
                    os.utime(path, ns=(1600000000000000000, 1600000000000000000))
                self.assertTrue(generator._write_or_check(path, rendered, False))
                self.assertEqual(path.read_bytes(), rendered.encode("utf-8"))

    def test_check_mode_preserves_bytes_mtime_and_newline_tolerance(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch("sys.stderr"):
            root = Path(temporary)
            path = root / "view.h"
            path.write_bytes(b"old\r\n")
            os.utime(path, ns=(1600000000000000000, 1600000000000000000))
            before = path.stat().st_mtime_ns
            self.assertFalse(generator._write_or_check(path, "expected\n", True))
            self.assertEqual(path.read_bytes(), b"old\r\n")
            self.assertEqual(path.stat().st_mtime_ns, before)
            self.assertFalse(generator._write_or_check(root / "missing" / "view.h", "expected\n", True))
            self.assertFalse((root / "missing").exists())
            # Check mode has always compared decoded universal-newline text.
            self.assertTrue(generator._write_or_check(path, "old\n", True))
            self.assertEqual(path.stat().st_mtime_ns, before)

    def test_full_manifest_pin_refresh_preserves_all_identical_output_mtimes(self):
        original = generator.load_manifest()
        loader = generator.load_manifest
        documentation = generator.DOC_TARGET.read_text(encoding="utf-8")
        dvc = generator.DVC_MANIFEST.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as patches:
            root = Path(temporary)
            paths = {name: root / name for name in (
                "MANIFEST", "CPP_TARGET", "PYTHON_TARGET", "DOC_TARGET", "DVC_MANIFEST")}
            paths["MANIFEST"].write_text(
                json.dumps(original, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
            paths["DOC_TARGET"].write_text(documentation, encoding="utf-8", newline="\n")
            paths["DVC_MANIFEST"].write_text(dvc, encoding="utf-8", newline="\n")
            for name, path in paths.items():
                patches.enter_context(mock.patch.object(generator, name, path))
            patches.enter_context(mock.patch.object(
                generator, "load_manifest", side_effect=lambda: loader(paths["MANIFEST"])))
            # This IO regression uses the supplied authenticated pins; source-closure validation
            # has independent tests and must not make this test refresh active repository pins.
            pins = {group["name"]: group["source_closure_sha256"] for group in original["groups"]}
            patches.enter_context(mock.patch.object(generator, "computed_group_pins", return_value=pins))
            self.assertTrue(generator.generate(False, True))
            expected = {path: path.read_bytes() for path in paths.values()}
            for path in paths.values():
                os.utime(path, ns=(1600000000000000000, 1600000000000000000))
            before = {path: path.stat().st_mtime_ns for path in paths.values()}
            self.assertTrue(generator.generate(False, True))
            self.assertTrue(generator.generate(True, False))
            for path in paths.values():
                self.assertEqual(path.read_bytes(), expected[path])
                self.assertEqual(path.stat().st_mtime_ns, before[path])
            cpp = paths["CPP_TARGET"]
            cpp.write_bytes(b"stale\r\n")
            os.utime(cpp, ns=(1600000000000000000, 1600000000000000000))
            with mock.patch("sys.stderr"):
                self.assertFalse(generator.generate(True, False))
            self.assertEqual(cpp.read_bytes(), b"stale\r\n")
            self.assertEqual(cpp.stat().st_mtime_ns, before[cpp])


if __name__ == "__main__":
    unittest.main()
