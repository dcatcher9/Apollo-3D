"""Math/source contracts for the authenticated final-parallax inverse."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import numpy as np

try:
    from . import direct_geometry_contract
except ImportError:
    import direct_geometry_contract  # type: ignore


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SHADER_ROOT = REPO_ROOT / "src_assets" / "windows" / "assets" / "shaders" / "directx"
ITERATIONS = 11
MAX_SLOPE = direct_geometry_contract.MAX_HORIZONTAL_SLOPE
SOURCE_U_LIMIT = direct_geometry_contract.SOURCE_U_LIMIT


def _clamped_linear_parallax(
    x: float,
    *,
    slope: float,
    center: float,
    offset: float,
) -> float:
    sampled_x = min(max(x, 0.0), 1.0)  # HLSL uses the clamp sampler.
    value = slope * (sampled_x - center) + offset
    return min(max(value, -SOURCE_U_LIMIT), SOURCE_U_LIMIT)


def _fixed_point_inverse(destination: float, eye_sign: float, parallax) -> float:
    source = destination
    for _ in range(ITERATIONS):
        source = destination + eye_sign * parallax(source)
    return source


def _bisected_inverse(destination: float, eye_sign: float, parallax) -> float:
    lo = destination - SOURCE_U_LIMIT
    hi = destination + SOURCE_U_LIMIT
    for _ in range(80):
        mid = (lo + hi) * 0.5
        residual = mid - destination - eye_sign * parallax(mid)
        if residual < 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


class DirectInverseWarpMathTest(unittest.TestCase):
    def test_native_float32_manifest_round_trip_is_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            order_path = root / "depth_00001.f32"
            parallax_path = root / "parallax_00001.f32"
            np.zeros((2, 2), dtype="<f4").tofile(order_path)
            np.full((2, 2), 0.5, dtype="<f4").tofile(parallax_path)
            manifest = {
                **direct_geometry_contract.MANIFEST_HEADER,
                "fields": [{
                    "frame_id": "00001",
                    "width": 2,
                    "height": 2,
                    "parallax_sha256": direct_geometry_contract.file_sha256(
                        str(parallax_path)),
                    "maximum_absolute_source_u": 0.0,
                    "order_sha256": direct_geometry_contract.file_sha256(str(order_path)),
                    "order_minimum": 0.0,
                    "order_maximum": 0.0,
                }],
            }
            manifest_path = root / "direct_parallax_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            contract = {
                "schema": direct_geometry_contract.CONTRACT_SCHEMA,
                "warp_input": direct_geometry_contract.WARP_INPUT,
                "direct_parallax": direct_geometry_contract.GEOMETRY_DESCRIPTOR,
                "direct_parallax_frames": 1,
                "direct_parallax_manifest": {
                    "file": manifest_path.name,
                    "schema": direct_geometry_contract.MANIFEST_SCHEMA,
                    "sha256": direct_geometry_contract.file_sha256(str(manifest_path)),
                },
            }

            validated = direct_geometry_contract.validate_artifacts(str(root), contract, [1])
            self.assertEqual(validated["shapes"][1], (2, 2))
            self.assertEqual(
                manifest["maximum_horizontal_source_u_slope"],
                float(np.float32(MAX_SLOPE)),
            )

            manifest["maximum_horizontal_source_u_slope"] = 0.41
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            contract["direct_parallax_manifest"]["sha256"] = (
                direct_geometry_contract.file_sha256(str(manifest_path)))
            with self.assertRaisesRegex(ValueError, "missing/unknown schema-6 semantics"):
                direct_geometry_contract.validate_artifacts(str(root), contract, [1])

    def test_eye_sign_is_the_inverse_of_forward_warp(self) -> None:
        parallax = lambda _x: 0.02
        destination = 0.5
        for eye_sign in (-1.0, 1.0):
            source = _fixed_point_inverse(destination, eye_sign, parallax)
            self.assertAlmostEqual(source, destination + eye_sign * 0.02)
            self.assertAlmostEqual(
                source - eye_sign * parallax(source),
                destination,
            )

    def test_eleven_iterations_meet_the_contraction_error_bound(self) -> None:
        # Include both slope signs, the authenticated worst magnitude, clipping plateaus,
        # both eyes, and destinations whose fixed point samples beyond a texture edge.
        worst_error = 0.0
        for slope in (-MAX_SLOPE, -0.23, 0.0, 0.23, MAX_SLOPE):
            for center in (0.15, 0.5, 0.85):
                for offset in (-0.025, 0.0, 0.025):
                    parallax = lambda x, slope=slope, center=center, offset=offset: (
                        _clamped_linear_parallax(
                            x,
                            slope=slope,
                            center=center,
                            offset=offset,
                        )
                    )
                    for eye_sign in (-1.0, 1.0):
                        for destination in (0.0, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0):
                            actual = _fixed_point_inverse(
                                destination,
                                eye_sign,
                                parallax,
                            )
                            expected = _bisected_inverse(
                                destination,
                                eye_sign,
                                parallax,
                            )
                            error = abs(actual - expected)
                            worst_error = max(worst_error, error)
                            self.assertAlmostEqual(
                                actual - eye_sign * parallax(actual),
                                destination,
                                delta=(1.0 + MAX_SLOPE) *
                                      (MAX_SLOPE ** ITERATIONS) * SOURCE_U_LIMIT,
                            )

        theoretical_bound = (MAX_SLOPE ** ITERATIONS) * SOURCE_U_LIMIT
        self.assertLessEqual(worst_error, theoretical_bound + 1.0e-12)
        # At the supported 5120-pixel source limit the conservative bound is 0.1
        # source pixel (0.075 at 3840), below a visible sampling displacement.
        self.assertLessEqual(theoretical_bound * 5120.0, 0.1)

class DirectInverseWarpSourceTest(unittest.TestCase):
    def test_external_final_path_uses_only_the_contractive_inverse(self) -> None:
        reprojection = (SHADER_ROOT / "sbs_direct_replay_ps.hlsl").read_text(
            encoding="utf-8"
        )
        function = reprojection[reprojection.index("float2 Reproject"):]
        direct_end = function.index("float4 main_ps")
        direct = function[:direct_end]

        self.assertIn("[unroll]", direct)
        self.assertIn(
            "for (int iteration = 0; iteration < 11; ++iteration)",
            direct,
        )
        self.assertEqual(direct.count("SampleParallax(source_x, destination_uv.y)"), 1)
        predicate = "asuint(next_source_x) == asuint(source_x)"
        assignment = "source_x = next_source_x;"
        guarded_break = "if (exactly_settled)"
        self.assertEqual(direct.count(predicate), 1)
        self.assertEqual(direct.count(assignment), 1)
        self.assertEqual(direct.count(guarded_break), 1)
        self.assertLess(direct.index(predicate), direct.index(assignment))
        self.assertLess(direct.index(assignment), direct.index(guarded_break))
        self.assertIn("#if !defined(HOST_SBS_TEST_FIXED_ELEVEN_REFERENCE)", direct)
        self.assertNotIn("next_source_x == source_x", direct)
        self.assertNotIn("abs(next_source_x", direct)
        for forbidden in (
            "DirectOrderTexture",
            "ForwardCoverageTexture",
            "probeStart",
            "foundSurface",
            "bgOrder",
        ):
            self.assertNotIn(forbidden, direct)
        for forbidden in (
            "SBS_DIRECT_CANDIDATE_PARALLAX",
            "SBS_CANDIDATE_GAP_FILL_MIRROR",
            "SBS_CANDIDATE_GAP_FILL_EDGE_CLAMP",
            "CanonicalOrderTexture",
            "backgroundOwner",
            "gap_fill",
        ):
            self.assertNotIn(forbidden, reprojection)

if __name__ == "__main__":
    unittest.main()
