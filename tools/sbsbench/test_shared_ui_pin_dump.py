"""Actual-map readbacks must never inherit the legacy analytic map's authority."""

import copy
import hashlib
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    from . import shared_ui_pin_dump as pin
except ImportError:
    import shared_ui_pin_dump as pin


class SharedUiPinDumpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.closure = "a" * 64
        patch = mock.patch.object(pin, "_expected_closure", return_value=self.closure)
        patch.start()
        self.addCleanup(patch.stop)
        self.manifest = {
            "matched_frame_id": 42, "matched_observation_timestamp_us": 100001,
            "renderer": {"mapping_artifacts_match_selected_renderer": False,
                "actual_draw_renderer": {"kind": pin.KIND,
                    "source_closure_sha256": self.closure,
                    "legacy_mapping_role": pin.LEGACY_ROLE, "evidence_artifact": pin.EVIDENCE}},
            "dimensions": {"source": {"width": 8, "height": 4},
                "packed_sbs": {"width": 16, "height": 4}}, "artifacts": {},
        }
        self.evidence = {"captured": True, "conversion_frame": 999, "model_frame": 42,
            "observation_us": 100001, "pixel_shader_closure": self.closure,
            "shared_pin": {"selected": True, "source_closure_sha256": self.closure,
                "source_shape": [8, 4, 1, 2], "map_artifact": pin.ARTIFACTS[4],
                "boundary_artifact": pin.ARTIFACTS[5],
                "map_layout": "packed-output row-major float32-le RGBA: sourceU,sourceV,pinnedRow,contentValid",
                "boundary_layout": "packed-output row-major float32-le RGBA: finiteSourceBoundary,0,0,1"},
            "artifacts": {}}
        weights = np.zeros((2, 8), dtype="<f4")
        weights[0, 3] = 1
        field = np.zeros_like(weights)
        rows = np.array([1, 0, 1], dtype="<u4")
        mapping = np.zeros((4, 16, 4), dtype="<f4")
        mapping[..., 0] = (np.arange(16) % 8 + 0.5) / 8
        mapping[..., 1] = (np.arange(4)[:, None] + 0.5) / 4
        mapping[..., 3] = 1
        mapping[1, :, 2] = 1
        boundary = np.zeros_like(mapping)
        boundary[..., 3] = 1
        for name, value, shape in zip(pin.ARTIFACTS,
                (weights.tobytes(), field.tobytes(), rows.tobytes(), struct.pack("<4I", 8, 4, 1, 2),
                 mapping.tobytes(), boundary.tobytes()),
                ((8, 2, 41), (8, 2, 41), (1, 3, 42), None, (16, 4, 2), (16, 4, 2))):
            self.save(name, value, shape)
        self.seal()

    def save(self, name, payload, shape=None):
        (self.root / name).write_bytes(payload)
        descriptor = {"available": True, "sha256": hashlib.sha256(payload).hexdigest(),
                      "byte_count": len(payload)}
        if shape is not None:
            w, h, fmt = shape
            descriptor["texture"] = {"width": w, "height": h, "format_value": fmt}
        self.manifest["artifacts"][name] = descriptor
        self.evidence["artifacts"][name] = copy.deepcopy(descriptor)

    def seal(self, payload=None):
        if payload is None:
            payload = json.dumps(self.evidence).encode()
        (self.root / pin.EVIDENCE).write_bytes(payload)
        self.manifest["artifacts"][pin.EVIDENCE] = {
            "available": True, "sha256": hashlib.sha256(payload).hexdigest()}

    def test_exact_readbacks_and_actual_map_selection(self):
        summary = pin.verify(self.root, self.manifest)
        self.assertTrue(summary["actual_map_verified"])
        self.assertEqual(summary["authorized_native_pixels"], 1)
        self.assertFalse(summary["legacy_map_has_render_authority"])
        self.assertEqual(pin.load_actual_map(self.root, self.manifest).shape, (4, 16))

    def test_legacy_packages_keep_their_original_attribution(self):
        self.assertFalse(pin.verify(self.root, {"renderer": {}})["shared_pin_selected"])

    def current_attribution(self):
        self.manifest["schema"] = 51
        self.manifest["renderer"]["actual_draw_renderer"] = {
            "schema": 2, "kind": pin.KIND, "source_closure_sha256": self.closure,
            "mapping_role": pin.SHARED_ROLE, "evidence_artifact": pin.EVIDENCE}
        self.evidence["schema"] = 2
        self.evidence["binding_roles"] = {"t3": "actual_draw_M.u32"}
        self.seal()

    def test_current_shared_draw_has_versioned_base_diagnostic_attribution(self):
        self.current_attribution()
        summary = pin.verify(self.root, self.manifest)
        self.assertTrue(summary["actual_map_verified"])
        self.assertEqual(summary["evidence_schema"], 2)

    def test_current_ordinary_base_cannot_be_relabelled_as_legacy_fallback(self):
        self.current_attribution()
        value = self.manifest["renderer"]["actual_draw_renderer"]
        value.update(kind="ordinary-base-v1", source_closure_sha256="", mapping_role=pin.BASE_ROLE)
        summary = pin.verify(self.root, self.manifest)
        self.assertEqual(summary["kind"], "ordinary-base-v1")
        self.assertFalse(summary["shared_pin_selected"])
        self.assertFalse(summary["actual_map_verified"])
        value["kind"] = "legacy-analytic-v1"
        with self.assertRaisesRegex(ValueError, "unknown source or map"):
            pin.verify(self.root, self.manifest)

    def test_current_dump_requires_versioned_renderer_and_evidence(self):
        historical = copy.deepcopy(self.manifest)
        historical["schema"] = 51
        with self.assertRaisesRegex(ValueError, "unknown schema"):
            pin.verify(self.root, historical)
        self.current_attribution()
        for schema in (None, True, 1, 3):
            with self.subTest(schema=schema):
                self.evidence["schema"] = schema
                self.seal()
                with self.assertRaisesRegex(ValueError, "evidence schema"):
                    pin.verify(self.root, self.manifest)

    def test_current_evidence_rejects_retired_t8_role_and_artifact(self):
        self.current_attribution()
        self.evidence["binding_roles"]["t8"] = "actual_original_legacy_M.u32"
        self.seal()
        with self.assertRaisesRegex(ValueError, "retired legacy t8"):
            pin.verify(self.root, self.manifest)
        self.evidence["binding_roles"].pop("t8")
        for container in (self.evidence["artifacts"], self.manifest["artifacts"]):
            container["actual_original_legacy_M.u32"] = {"available": False}
            self.seal()
            with self.assertRaisesRegex(ValueError, "retired legacy t8"):
                pin.verify(self.root, self.manifest)
            container.pop("actual_original_legacy_M.u32")

    def test_shared_renderer_cannot_claim_legacy_map(self):
        self.manifest["renderer"]["mapping_artifacts_match_selected_renderer"] = True
        with self.assertRaisesRegex(ValueError, "legacy analytic map"):
            pin.verify(self.root, self.manifest)

    def test_source_clock_and_closure_are_independent_joins(self):
        for key, value in (("model_frame", 41), ("observation_us", 100000),
                           ("pixel_shader_closure", "b" * 64)):
            with self.subTest(key=key):
                previous = self.evidence[key]
                self.evidence[key] = value
                self.seal()
                with self.assertRaisesRegex(ValueError, "exact draw and source"):
                    pin.verify(self.root, self.manifest)
                self.evidence[key] = previous

    def test_payload_hash_and_row_flags_reject_mutations(self):
        name = pin.ARTIFACTS[2]
        (self.root / name).write_bytes(struct.pack("<3I", 0, 0, 1))
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            pin.verify(self.root, self.manifest)
        self.save(name, struct.pack("<3I", 0, 0, 1), (1, 3, 42))
        self.seal()
        with self.assertRaisesRegex(ValueError, "weights/rows/header"):
            pin.verify(self.root, self.manifest)

    def test_boundary_must_come_from_same_map(self):
        name = pin.ARTIFACTS[5]
        values = np.frombuffer((self.root / name).read_bytes(), dtype="<f4").copy()
        values[0] = 1
        self.save(name, values.tobytes(), (16, 4, 2))
        self.seal()
        with self.assertRaisesRegex(ValueError, "finite-source boundary disagree"):
            pin.verify(self.root, self.manifest)

    def test_evidence_requires_one_unambiguous_json_object(self):
        duplicate = b'{"captured":false,' + json.dumps(self.evidence).encode()[1:]
        for payload in (b"[]", b"null", duplicate):
            with self.subTest(payload=payload[:30]):
                self.seal(payload)
                with self.assertRaisesRegex(ValueError, "object|duplicate"):
                    pin.verify(self.root, self.manifest)

    def test_source_identity_and_shape_reject_boolean_aliases(self):
        for field, manifest_field in (("model_frame", "matched_frame_id"),
                                      ("observation_us", "matched_observation_timestamp_us")):
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                manifest[manifest_field] = 1
                previous = self.evidence[field]
                self.evidence[field] = True
                self.seal()
                manifest["artifacts"][pin.EVIDENCE] = self.manifest["artifacts"][pin.EVIDENCE]
                try:
                    with self.assertRaisesRegex(ValueError, "exact draw and source"):
                        pin.verify(self.root, manifest)
                finally:
                    self.evidence[field] = previous
        self.evidence["shared_pin"]["source_shape"][2] = True
        self.seal()
        with self.assertRaisesRegex(ValueError, "source shape"):
            pin.verify(self.root, self.manifest)

    def test_invalid_content_map_uses_canonical_sentinel(self):
        name = pin.ARTIFACTS[4]
        values = np.frombuffer((self.root / name).read_bytes(), dtype="<f4").copy()
        values[:4] = (0, 0, 1, 0)
        self.save(name, values.tobytes(), (16, 4, 2))
        self.seal()
        with self.assertRaisesRegex(ValueError, "invalid content"):
            pin.verify(self.root, self.manifest)

    def test_malformed_descriptors_report_validation_errors(self):
        previous = self.manifest["artifacts"][pin.EVIDENCE]
        self.manifest["artifacts"][pin.EVIDENCE] = []
        with self.assertRaisesRegex(ValueError, "descriptor must be an object"):
            pin.verify(self.root, self.manifest)
        self.manifest["artifacts"][pin.EVIDENCE] = previous
        name = pin.ARTIFACTS[4]
        self.manifest["artifacts"][name]["texture"] = None
        self.evidence["artifacts"][name]["texture"] = None
        self.seal()
        with self.assertRaisesRegex(ValueError, "exact texture layout"):
            pin.verify(self.root, self.manifest)

    def test_evidence_artifact_map_must_be_an_object(self):
        for artifacts in (None, []):
            with self.subTest(artifacts=artifacts):
                self.evidence["artifacts"] = artifacts
                self.seal()
                with self.assertRaisesRegex(ValueError, "artifacts must be an object"):
                    pin.verify(self.root, self.manifest)

    def test_optional_readback_failure_never_substitutes_legacy_map(self):
        name = pin.ARTIFACTS[4]
        self.manifest["artifacts"][name] = {"available": False}
        self.evidence["artifacts"][name] = {"available": False}
        self.seal()
        summary = pin.verify(self.root, self.manifest)
        self.assertFalse(summary["actual_map_verified"])
        with self.assertRaisesRegex(ValueError, "no complete authenticated"):
            pin.load_actual_map(self.root, self.manifest)

    def test_optional_binding_observation_failure_has_no_map_authority(self):
        self.evidence["captured"] = False
        self.evidence["shared_pin"] = {}
        self.seal()
        summary = pin.verify(self.root, self.manifest)
        self.assertFalse(summary["actual_map_verified"])


class SharedUiPinDumpOwnerTests(unittest.TestCase):
    def test_unmocked_source_closure_matches_actual_runtime_owner(self):
        header = Path(__file__).resolve().parents[2] / "src" / "host_sbs_shared_ui_warp_runtime.h"
        expected = re.search(r'inline constexpr std::string_view closure\s*=\s*"([0-9a-f]{64})"',
                             header.read_text(encoding="utf-8"))
        self.assertIsNotNone(expected)
        self.assertEqual(pin._expected_closure(), expected.group(1))


if __name__ == "__main__":
    unittest.main()
