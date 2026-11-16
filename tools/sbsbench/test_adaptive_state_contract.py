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
            8: "a1f3113ea102d43e968112f930ccf66697b09f9067097256c092f7e8c429f605",
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
        self.assertEqual(contract.TRACE_SCHEMA, 8)
        self.assertEqual(contract.CONTRACT_TAG, 0xA1F3113E)
        self.assertEqual(
            contract.TRACE_SOURCE,
            "video_depth_estimator.DepthHealthState",
        )
        self.assertEqual(
            contract.TRACE_CAPTURE,
            "every-source-frame-after-complete-estimator-update",
        )
        self.assertEqual(
            set(contract.CONTRACT),
            {
                "schema", "source", "capture", "counter_max", "header_keys", "config_keys",
                "frame_keys", "fields",
            },
        )
        self.assertEqual(len(contract.FIELD_SPECS), 8)
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
        self.assertEqual(
            [(name, kind, encoding) for (name, kind), encoding in zip(
                contract.FIELD_SPECS, contract.FIELD_ENCODINGS)],
            [
                ("contract_tag_bits", "uint32", "uint_bits"),
                ("empty_raw_count", "uint32", "uint_bits"),
                ("collapsed_raw_count", "uint32", "uint_bits"),
                ("reserved_3", "uint32", "uint_bits"),
                ("valid_depth_fraction", "float32", "float"),
                ("effective_raw_range_width", "float32", "float"),
                ("range_collapsed", "float32", "float"),
                ("depth_ready", "float32", "float"),
            ])
        self.assertEqual(contract.INITIAL_VALUES["contract_tag_bits"], "contract_tag")
        for name in contract.FIELD_NAMES[1:]:
            self.assertEqual(contract.INITIAL_VALUES[name], 0)


if __name__ == "__main__":
    unittest.main()
