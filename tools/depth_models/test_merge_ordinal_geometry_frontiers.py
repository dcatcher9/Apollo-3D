#!/usr/bin/env python3

import copy
import unittest

import artistic_geometry_contract as geometry_contract
import artistic_policy_ordinal_contract as ordinal_contract
import merge_ordinal_geometry_frontiers as merger


SOURCE = "a" * 64
CONDITION = "b" * 64


def geometry(eye_width, eye_height):
    scale_x, scale_y = geometry_contract.source_content_scales(
        1920, 1080, eye_width, eye_height
    )
    return geometry_contract.geometry_tuple({
        "source_width": 1920,
        "source_height": 1080,
        "eye_width": eye_width,
        "eye_height": eye_height,
        "content_scale_x": scale_x,
        "content_scale_y": scale_y,
        "disparity_raster_width": eye_width,
        "disparity_raster_height": eye_height,
    })


def frontier(identity_pop, gain_per_step, failure_index=None, cause="halo:hard"):
    tested = []
    final_index = (
        failure_index if failure_index is not None
        else ordinal_contract.FRONTIER_SIZE - 1
    )
    for index in range(final_index + 1):
        safe = failure_index is None or index < failure_index
        tested.append({
            "scale": ordinal_contract.SCALES[index],
            "safe": safe,
            "realized_pop_pct": identity_pop + gain_per_step * index,
            "failure_causes": [] if safe else [cause],
        })
    return ordinal_contract.build_frontier_evidence(tested)


def record(eye_width, eye_height, evidence, source=SOURCE,
           condition=CONDITION):
    return merger.build_geometry_frontier(
        source, condition, geometry(eye_width, eye_height), evidence
    )


class OrdinalGeometryIntersectionTests(unittest.TestCase):
    def test_earliest_geometry_failure_bounds_the_intersection(self):
        first = record(1280, 720, frontier(2.0, 0.1, failure_index=4))
        second = record(1920, 1080, frontier(
            4.0, 0.2, failure_index=2, cause="coverage:hard"
        ))
        result = merger.intersect_geometry_frontiers([first, second])
        self.assertEqual(result["states"][:4], [
            "safe", "safe", "unsafe", "unknown",
        ])
        self.assertEqual(result["highest_proven_safe_scale"], 1.02)
        self.assertEqual(result["first_proven_unsafe_scale"], 1.04)
        self.assertEqual(result["first_unsafe_failures"], [{
            "deployment_geometry_sha256":
                second["deployment_geometry_sha256"],
            "failure_causes": ["coverage:hard"],
        }])
        self.assertTrue(all(
            value is None for value in
            result["conservative_safe_pop_gain_over_identity_pct"][2:]
        ))

    def test_same_scale_failures_retain_both_geometry_causes(self):
        records = [
            record(1280, 720, frontier(
                2.0, 0.1, failure_index=2, cause="halo:hard"
            )),
            record(1920, 1080, frontier(
                3.0, 0.2, failure_index=2, cause="coverage:hard"
            )),
        ]
        result = merger.intersect_geometry_frontiers(records)
        self.assertEqual(len(result["first_unsafe_failures"]), 2)
        self.assertEqual({
            item["failure_causes"][0]
            for item in result["first_unsafe_failures"]
        }, {"halo:hard", "coverage:hard"})

    def test_gain_uses_same_geometry_deltas_before_taking_minimum(self):
        # At index 1, min(current pop)=1.10 and min(identity pop)=1.00,
        # whose unrelated-minima subtraction would claim 0.10. Geometry B
        # actually gains only 0.05, so the conservative result must be 0.05.
        records = [
            record(1280, 720, frontier(1.0, 0.10)),
            record(1920, 1080, frontier(10.0, 0.05)),
        ]
        result = merger.intersect_geometry_frontiers(records)
        self.assertTrue(result["right_censored"])
        self.assertAlmostEqual(
            result["conservative_safe_pop_gain_over_identity_pct"][1], 0.05
        )
        self.assertAlmostEqual(
            result["maximum_conservative_safe_pop_gain_pct"], 1.25
        )
        for component, original in zip(
                result["geometry_frontiers"],
                sorted(records, key=lambda item:
                       item["deployment_geometry_sha256"])):
            self.assertEqual(
                component["frontier"]["realized_pop_pct"],
                original["frontier"]["realized_pop_pct"],
            )

    def test_source_condition_and_geometry_identity_fail_closed(self):
        base = record(1280, 720, frontier(1.0, 0.1))
        cases = (
            (base, record(1920, 1080, frontier(1.0, 0.1), source="c" * 64),
             "source identities"),
            (base, record(1920, 1080, frontier(1.0, 0.1),
                          condition="d" * 64), "input conditions"),
            (base, copy.deepcopy(base), "repeats a geometry"),
        )
        for first, second, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(RuntimeError, pattern):
                    merger.intersect_geometry_frontiers([first, second])

    def test_identity_failure_is_left_censored(self):
        result = merger.intersect_geometry_frontiers([
            record(1280, 720, frontier(1.0, 0.1, failure_index=0)),
            record(1920, 1080, frontier(2.0, 0.1, failure_index=3)),
        ])
        self.assertFalse(result["identity_feasible"])
        self.assertTrue(result["left_censored"])
        self.assertIsNone(result["highest_proven_safe_scale"])
        self.assertEqual(result["first_proven_unsafe_scale"], 1.0)

    def test_validator_rejects_tampering(self):
        result = merger.intersect_geometry_frontiers([
            record(1280, 720, frontier(1.0, 0.1)),
            record(1920, 1080, frontier(2.0, 0.1)),
        ])
        merger.validate_geometry_intersection(result)
        changed = copy.deepcopy(result)
        changed["states"][2] = "unsafe"
        with self.assertRaisesRegex(RuntimeError, "canonical"):
            merger.validate_geometry_intersection(changed)


if __name__ == "__main__":
    unittest.main()
