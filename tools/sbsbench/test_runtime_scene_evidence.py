import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_scene_evidence as evidence  # noqa: E402


def valid_payload():
    source_ids = list(range(100, 120))
    frames = []
    ages = list(range(8)) + [0, 1]
    for index, age in zip(range(0, 20, 2), ages):
        hard_cut = index == 16
        frames.append({
            "source_frame_ordinal": index,
            "source_frame_id": source_ids[index],
            "runtime_scene_id": 1 if hard_cut or index > 16 else 0,
            "scene_age": age,
            "subject_initialized": True,
            "hard_cut": hard_cut,
            "scene_start": index == 0 or hard_cut,
        })
    return {
        "schema": 1,
        "contract": evidence.CONTRACT,
        "evidence_source": "SubjectState[0].y after completed depth postprocess",
        "cut_rule": "prior_scene_age_gte_7_and_current_scene_age_eq_0",
        "cadence": "completed-depth-frames-only",
        "completion_sequence_contract": (
            "exact for this synchronous harness sequence; live busy-drop cadence is not replayed"
        ),
        "depth_reuse_interval": 2,
        "source_frame_ids": source_ids,
        "completed_source_frame_ids": source_ids[::2],
        "completed_depth_frame_count": len(frames),
        "frames": frames,
    }


class RuntimeSceneEvidenceTest(unittest.TestCase):
    def test_accepts_authoritative_age_reset(self):
        self.assertEqual(evidence.validate(valid_payload())["frames"][-2]["runtime_scene_id"], 1)

    def test_rejects_cut_without_age_threshold(self):
        payload = valid_payload()
        payload["frames"][7].update({
            "runtime_scene_id": 1, "scene_age": 0,
            "hard_cut": True, "scene_start": True,
        })
        payload["frames"][8].update({
            "runtime_scene_id": 1, "scene_age": 1,
            "hard_cut": False, "scene_start": False,
        })
        payload["frames"][9]["scene_age"] = 2
        with self.assertRaisesRegex(ValueError, "authoritative age reset"):
            evidence.validate(payload)

    def test_rejects_missing_completed_depth_frame(self):
        payload = valid_payload()
        del payload["frames"][3]
        with self.assertRaisesRegex(ValueError, "count differs"):
            evidence.validate(payload)

    def test_rejects_scene_change_without_cut(self):
        payload = valid_payload()
        payload["frames"][4]["runtime_scene_id"] = 2
        with self.assertRaisesRegex(ValueError, "changed without a hard cut"):
            evidence.validate(payload)

    def test_rejects_noncanonical_extra_field(self):
        payload = copy.deepcopy(valid_payload())
        payload["frames"][0]["future"] = True
        with self.assertRaisesRegex(ValueError, "noncanonical keys"):
            evidence.validate(payload)

    def test_rejects_nonzero_first_age(self):
        payload = valid_payload()
        payload["frames"][0]["scene_age"] = 1
        with self.assertRaisesRegex(ValueError, "first runtime scene row"):
            evidence.validate(payload)

    def test_rejects_age_jump_or_deinitialization(self):
        jump = valid_payload()
        jump["frames"][3]["scene_age"] = 4
        with self.assertRaisesRegex(ValueError, "increment once"):
            evidence.validate(jump)
        deinitialized = valid_payload()
        deinitialized["frames"][3]["subject_initialized"] = False
        deinitialized["frames"][3]["scene_age"] = 0
        with self.assertRaisesRegex(ValueError, "deinitialized"):
            evidence.validate(deinitialized)

    def test_uninitialized_state_requires_zero_age(self):
        payload = valid_payload()
        payload["frames"][0]["subject_initialized"] = False
        payload["frames"][1]["subject_initialized"] = False
        payload["frames"][1]["scene_age"] = 1
        with self.assertRaisesRegex(ValueError, "uninitialized.*zero age"):
            evidence.validate(payload)


if __name__ == "__main__":
    unittest.main()
