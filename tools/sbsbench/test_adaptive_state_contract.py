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
        # count. Intentional layout/key/flag changes must therefore bump the schema and add a new
        # reviewed golden digest instead of silently redefining schema 3.
        expected_digest_by_schema = {
            3: "4fb65e50c9ad7039c9b6407f2e772a30052a0dea06499fd3b49b260336c1c4d1",
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
                    "import tools.sbsbench.run_whole_clip;"
                    "import tools.sbsbench.scene_plan"
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

    def test_manifest_is_exact_and_carries_recovery_state(self):
        self.assertEqual(contract.TRACE_SCHEMA, 3)
        self.assertEqual(len(contract.FIELD_SPECS), 32)
        self.assertEqual(contract.FIELD_SPECS[10], ("cut_flags", "float32"))
        self.assertEqual(contract.FIELD_SPECS[31], ("analysis_flags", "uint32"))
        self.assertEqual(contract.CUT_FLAG_APPEARANCE_RECOVERY, 32)
        self.assertIn("appearance_recovery", contract.FRAME_KEYS)


if __name__ == "__main__":
    unittest.main()
