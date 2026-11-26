#!/usr/bin/env python3
import copy
import json
import os
import shutil
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
        generator.closure_group(stale_pin, "parallax_v2_live_renderer")[
            "source_closure_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source closure pin is stale"):
            generator.validate_group_pins(stale_pin)

    def test_only_provenance_groups_carry_checked_in_pins(self):
        manifest = generator.load_manifest()
        self.assertEqual(
            tuple(group["name"] for group in manifest["groups"]
                  if "source_closure_sha256" in group),
            generator.PINNED_GROUP_NAMES,
        )
        unpinned_with_pin = copy.deepcopy(manifest)
        generator.closure_group(unpinned_with_pin, "gpu_trace")["source_closure_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "keys differ"):
            generator.validate_manifest(unpinned_with_pin)
        pinned_without_pin = copy.deepcopy(manifest)
        del generator.closure_group(
            pinned_without_pin, "parallax_v2_live_renderer")["source_closure_sha256"]
        with self.assertRaisesRegex(ValueError, "keys differ"):
            generator.validate_manifest(pinned_without_pin)

    def test_python_view_derives_unpinned_closures_from_sources(self):
        import host_sbs_shader_manifest as view

        computed = generator.computed_group_pins(generator.load_manifest())
        self.assertEqual(set(view.CLOSURE_GROUPS), set(computed))
        for name, group in view.CLOSURE_GROUPS.items():
            with self.subTest(group=name):
                self.assertEqual(group.source_closure_sha256, computed[name])
                self.assertEqual(
                    group.pinned_source_closure_sha256 is not None,
                    name in generator.PINNED_GROUP_NAMES,
                )

    def test_unpinned_edit_changes_only_build_identity_and_pinned_edit_needs_a_pin(self):
        manifest = generator.load_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "directx"
            shutil.copytree(generator.SHADER_ROOT, root)
            header = Path(temporary) / "identity.h"
            depfile = Path(temporary) / "identity.d"
            generator.write_build_identity(header, depfile, root)
            identity = header.read_text(encoding="utf-8")
            self.assertEqual(
                identity,
                generator.render_build_identity(manifest, generator.computed_group_pins(manifest)),
            )
            for group in manifest["groups"]:
                self.assertEqual(
                    f" {group['name']}_source_closure_sha256 =" in identity,
                    not generator.is_pinned(group),
                )
            dependencies = depfile.read_text(encoding="utf-8")
            self.assertTrue(dependencies.startswith(f"{header.as_posix()}:"))
            for edited, groups in (
                    ("host_sbs_gpu_trace_cs.hlsl", {"gpu_trace"}),
                    ("direct_glyph_source/direct_source_audit.hlsl",
                     {"direct_glyph_source_audit"})):
                with self.subTest(source=edited):
                    source = root / edited
                    self.assertIn(f" \\\n  {source.resolve().as_posix()}", dependencies)
                    before = generator.computed_group_pins(manifest, root)
                    source.write_bytes(source.read_bytes() + b"\n// one-line edit\n")
                    after = generator.computed_group_pins(manifest, root)
                    self.assertEqual({name for name in before if before[name] != after[name]}, groups)
                    # No checked-in pin covers it: only the build-time header changes.
                    generator.validate_group_pins(manifest, root)
                    generator.write_build_identity(header, depfile, root)
                    self.assertNotEqual(header.read_text(encoding="utf-8"), identity)
                    identity = header.read_text(encoding="utf-8")
            renderer = root / "sbs_reprojection_v2_live_ps.hlsl"
            renderer.write_bytes(renderer.read_bytes() + b"\n// one-line edit\n")
            with self.assertRaisesRegex(
                    ValueError, "parallax_v2_live_renderer source closure pin is stale"):
                generator.validate_group_pins(manifest, root)


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
            pins = {group["name"]: group["source_closure_sha256"]
                    for group in original["groups"] if "source_closure_sha256" in group}
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
