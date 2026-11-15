import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import adaptive_state_contract as contract  # noqa: E402


class AdaptiveStateContractTests(unittest.TestCase):
    def test_schema_version_pins_the_complete_contract(self):
        # A stored schema number promises every meaning in this manifest, not merely its word
        # count. Intentional layout/key changes must therefore bump the schema and add a new
        # reviewed golden digest instead of silently redefining the current schema.
        expected_digest_by_schema = {
            7: "a99015fadbd63a0e2964a4292583b9f04465b1120b82b8ac3fe62f9d20facf87",
        }
        canonical = json.dumps(
            contract.CONTRACT, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.assertEqual(
            expected_digest_by_schema.get(contract.TRACE_SCHEMA),
            hashlib.sha256(canonical).hexdigest(),
            "adaptive-state semantics changed without a reviewed schema version",
        )

    def test_contract_consumers_support_package_imports(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import tools.sbsbench.adaptive_clip_report;"
                    "import tools.sbsbench.run_eval"
                ),
            ],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_generated_native_and_shader_contracts_are_current(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "generate_adaptive_state_contract.py"),
                "--check",
            ],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_manifest_is_exact(self):
        self.assertEqual(contract.TRACE_SCHEMA, 7)
        self.assertEqual(contract.CUT_CONTRACT_TAG, 0xA99015FA)
        self.assertEqual(
            contract.TRACE_SOURCE,
            "video_depth_estimator.CutBridgeState",
        )
        self.assertEqual(
            contract.TRACE_CAPTURE,
            "every-source-frame-after-complete-estimator-update",
        )
        self.assertEqual(
            set(contract.CONTRACT),
            {
                "schema", "source", "capture", "counter_max", "header_keys", "config_keys",
                "frame_keys", "retired_zero_fields", "fields",
            },
        )
        self.assertEqual(len(contract.FIELD_SPECS), 32)
        self.assertEqual(
            contract.HEADER_KEYS,
            frozenset({"record", "schema", "source", "capture", "fields", "config"}),
        )
        self.assertEqual(
            contract.CONFIG_KEYS,
            frozenset({"model", "pop_strength", "depth_reuse_interval"}),
        )
        self.assertEqual(
            contract.FRAME_KEYS,
            frozenset({
                "record", "frame_id", "source_index", "depth_updated", "empty_raw_count",
                "collapsed_raw_count", "values",
            }),
        )
        named = {
            index: (name, kind, encoding)
            for index, ((name, kind), encoding) in enumerate(
                zip(contract.FIELD_SPECS, contract.FIELD_ENCODINGS))
            if not name.startswith("reserved_cut_bridge_")
        }
        self.assertEqual(named, {
            0: ("cut_contract_tag_bits", "uint32", "uint_bits"),
            14: ("valid_depth_fraction", "float32", "float"),
            15: ("effective_raw_range_width", "float32", "float"),
            16: ("hard_cut_count", "uint32", "uint_bits"),
            18: ("empty_raw_count", "uint32", "uint_bits"),
            19: ("collapsed_raw_count", "uint32", "uint_bits"),
            20: ("range_collapsed", "float32", "float"),
            21: ("depth_ready", "float32", "float"),
        })
        for index, (name, _kind) in enumerate(contract.FIELD_SPECS):
            if index not in named:
                self.assertEqual(name, f"reserved_cut_bridge_{index}")
                self.assertEqual(contract.INITIAL_VALUES[name], 0)
        # The retired word keeps its slot for layout-stable joins; no producer writes it.
        self.assertEqual(contract.RETIRED_ZERO_FIELDS, ("hard_cut_count",))
        self.assertEqual(contract.INITIAL_VALUES["hard_cut_count"], 0)
        for retired_export in (
                "CUT_FLAG_BITS", "ANALYSIS_FLAG_BITS", "COMPACT_CUT_TRACE_FIELDS"):
            self.assertFalse(hasattr(contract, retired_export), retired_export)


if __name__ == "__main__":
    unittest.main()
