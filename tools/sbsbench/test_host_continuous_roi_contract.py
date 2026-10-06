"""Document admission for one continuous ROI field; native shaders own rendering proof."""
import copy
import hashlib
import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import depth_coordinate_v2_dump_contract as contract
import test_depth_coordinate_v2_dump_contract as fixtures


class HostContinuousRoiContractTests(unittest.TestCase):
    def fixture(self, root):
        legacy = fixtures.DepthCoordinateV2DumpContractTests("runTest")
        legacy.setUp()
        manifest, region_document, mapping = legacy._write_synthetic_roi_geometry_dump(root)
        region = contract.validate_depth_input_region_document(region_document)
        dimensions = manifest["dimensions"]["shadow_final_parallax"]
        final = np.fromfile(root / "shadow_final_parallax.f32", dtype="<f4").reshape(
            dimensions["height"], dimensions["width"])
        mask = np.asarray(Image.open(root / "warp_mask.png").convert("RGB")).copy()
        return manifest, region, mapping, final, mask

    @staticmethod
    def save(root, manifest, name, payload):
        (root / name).write_bytes(payload)
        manifest["artifacts"][name]["sha256"] = hashlib.sha256(payload).hexdigest()

    def save_mask(self, root, manifest, mask):
        Image.fromarray(mask).save(root / "warp_mask.png")
        manifest["artifacts"]["warp_mask.png"]["sha256"] = hashlib.sha256(
            (root / "warp_mask.png").read_bytes()).hexdigest()

    def test_adaptive_policy_uses_canonical_continuous_map_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _, _, _ = self.fixture(Path(temporary))
            self.assertEqual(manifest["warp_map_contract"]["schema"], 2)
            for mode in (3,):
                with self.subTest(mode=mode):
                    admitted = contract._validate_warp_map_manifest(
                        manifest, manifest["artifacts"], manifest["dimensions"], "window-region", mode)
                    self.assertTrue(admitted["available"])

    def test_retired_camera_modes_do_not_enter_current_map_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _, _, _ = self.fixture(Path(temporary))
            for mode in (0, 1, 2, True):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "mode"):
                    contract._validate_warp_map_manifest(
                        manifest, manifest["artifacts"], manifest["dimensions"], "window-region", mode)

    def test_layered_schema_and_selected_kind_semantics_cannot_masquerade_as_continuous(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _, _, _ = self.fixture(Path(temporary))
            for mutation in ("schema", "channel", "mask"):
                changed = copy.deepcopy(manifest)
                shape = changed["warp_map_contract"]
                if mutation == "schema":
                    shape["schema"] = 4
                elif mutation == "channel":
                    shape["channels"] = ["selected_source_u_normalized"]
                else:
                    shape["validity"]["mask"] = "green selects video, browser fill or static rim"
                with self.subTest(mutation=mutation), self.assertRaisesRegex(
                        ValueError, "warp-map contract|unknown warp-map semantics"):
                    contract._validate_warp_map_manifest(
                        changed, changed["artifacts"], changed["dimensions"], "window-region", 3)

    def test_nonzero_collar_is_admitted_while_far_exterior_zero_is_proven(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, region, mapping, final, _ = self.fixture(root)
            evidence = contract._verify_roi_exterior_zero_warp_map(root, manifest, region, final)
            self.assertTrue(evidence["has_exterior_zero_plane"])
            self.assertGreater(evidence["beyond_collar_sample_count"], 0)
            self.assertEqual(evidence["max_abs_identity_error_output_eye_px"], 0)
            left, top, right, bottom = region["inference_rect"]
            width = region["source_width"]
            packed = (np.arange(2 * width, dtype=np.float32) + np.float32(.5)) / np.float32(2 * width)
            identity = np.where(packed > np.float32(.5), (packed - np.float32(.5)) * 2,
                                packed * 2).astype(np.float32)
            exterior = (identity < np.float32(left / width)) | (identity >= np.float32(right / width))
            row = (top + bottom) // 2
            moved = np.abs(mapping[row] - identity) * width > .01
            self.assertGreater(np.count_nonzero(exterior & moved), 0)

    def test_authenticated_far_exterior_displacement_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, region, mapping, final, _ = self.fixture(root)
            changed = mapping.copy()
            changed[0, 0] += np.float32(.01)
            self.save(root, manifest, "warp_map.f32", changed.astype("<f4").tobytes())
            with self.assertRaisesRegex(ValueError, "nonzero beyond conservative collar support"):
                contract._verify_roi_exterior_zero_warp_map(root, manifest, region, final)

    def test_ownership_and_nonbinary_boundary_mask_channels_fail_closed(self):
        for channel, value in ((1, 85), (1, 170), (1, 255), (2, 255), (0, 1)):
            with self.subTest(channel=channel, value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, region, _, final, mask = self.fixture(root)
                mask[1, 1, channel] = value
                self.save_mask(root, manifest, mask)
                with self.assertRaisesRegex(ValueError, "boundary mask has noncanonical channels"):
                    contract._verify_roi_exterior_zero_warp_map(root, manifest, region, final)


if __name__ == "__main__":
    unittest.main()
