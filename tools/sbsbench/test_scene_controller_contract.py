import copy
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import scene_controller_contract as contract  # noqa: E402


class SceneControllerContractTests(unittest.TestCase):
    def test_schema_version_pins_the_complete_contract(self):
        expected_digest_by_schema = {
            1: "bf62f184f787207aab6b5f59269a3126b0fb1890648cdcb5631bbd5d67c4a570",
        }
        canonical = json.dumps(
            contract.CONTRACT,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            expected_digest_by_schema.get(contract.SCHEMA_VERSION),
            hashlib.sha256(canonical).hexdigest(),
            "Scene Controller ABI semantics changed without a reviewed schema version",
        )

    def test_ordered_hash_covers_same_count_reorders_and_meaning_changes(self):
        self.assertEqual(
            contract.ORDERED_ABI_HASH,
            contract.compute_ordered_abi_hash(contract.CONTRACT),
        )

        reordered = copy.deepcopy(contract.CONTRACT)
        reordered["analysis_grid"][1]["name"], reordered["analysis_grid"][2]["name"] = (
            reordered["analysis_grid"][2]["name"],
            reordered["analysis_grid"][1]["name"],
        )
        self.assertNotEqual(
            contract.ORDERED_ABI_HASH,
            contract.compute_ordered_abi_hash(reordered),
        )
        with self.assertRaisesRegex(RuntimeError, "ordered ABI hash mismatch"):
            contract.validate_contract(reordered)

        changed_meaning = copy.deepcopy(contract.CONTRACT)
        changed_meaning["rule_state"][34]["name"] = "different_winner_meaning"
        self.assertNotEqual(
            contract.ORDERED_ABI_HASH,
            contract.compute_ordered_abi_hash(changed_meaning),
        )
        with self.assertRaisesRegex(RuntimeError, "ordered ABI hash mismatch"):
            contract.validate_contract(changed_meaning)

    def test_generated_native_and_shader_contracts_are_current(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "generate_scene_controller_contract.py"),
                "--check",
            ],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_canonical_dimensions_and_tensors_match_the_design(self):
        self.assertEqual(contract.DIMENSIONS["appearance_canvas_size"], 256)
        self.assertEqual(contract.DIMENSIONS["analysis_canvas_size"], 128)
        self.assertEqual(contract.DIMENSIONS["recurrent_canvas_size"], 32)
        self.assertEqual(contract.DIMENSIONS["analysis_grid_channel_count"], 10)
        self.assertEqual(contract.DIMENSIONS["layout_history_channel_count"], 12)
        self.assertEqual(contract.DIMENSIONS["depth_history_channel_count"], 10)
        self.assertEqual(contract.DIMENSIONS["dense_out_channel_count"], 14)
        self.assertEqual(contract.DIMENSIONS["global_out_word_count"], 41)
        self.assertEqual(contract.DIMENSIONS["meta_word_count"], 32)
        self.assertEqual(contract.DIMENSIONS["hidden_channel_count"], 24)
        self.assertEqual(contract.INPUT_NAMES, (
            "scene_rgb",
            "analysis_grid",
            "roi_rgb_tensor",
            "roi_depth_raw",
            "layout_history",
            "depth_history",
            "hidden_in",
            "meta",
        ))
        self.assertEqual(
            contract.OUTPUT_NAMES,
            ("dense_out", "global_out", "hidden_out"),
        )

    def test_rule_state_is_a_complete_typed_layout(self):
        self.assertEqual(contract.RULE_REVISION, "rules_v1")
        self.assertEqual(len(contract.RULE_STATE_NAMES), 64)
        self.assertEqual(contract.RULE_STATE_NAMES[:8], (
            "schema_version",
            "state_kind",
            "output_valid",
            "backend_generation",
            "roi_generation",
            "update_count",
            "state_flags",
            "reset_flags",
        ))
        self.assertEqual(contract.RULE_STATE_NAMES[-4:], (
            "last_external_cut_count",
            "reserved_1",
            "reserved_2",
            "reserved_3",
        ))
        self.assertEqual(
            contract.CONTRACT["rule_state"][-4]["gpu_encoding"],
            "uint_valued_float",
        )
        for field in contract.CONTRACT["rule_state"][-3:]:
            self.assertTrue(field["required_zero"])
            self.assertEqual(field["initial"], 0.0)

    def test_reserved_tensor_outputs_are_explicitly_required_zero(self):
        self.assertTrue(
            all(
                field.get("required_zero") is True
                for field in contract.CONTRACT["meta"][28:]
            )
        )
        self.assertTrue(
            all(
                field.get("required_zero") is True
                for field in contract.CONTRACT["global_out"][35:]
            )
        )


if __name__ == "__main__":
    unittest.main()
