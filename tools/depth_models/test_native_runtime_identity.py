#!/usr/bin/env python3

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import native_runtime_identity as runtime


class NativeRuntimeIdentityTests(unittest.TestCase):
    def test_identity_hashes_binary_bytes_without_installation_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fake_package"
            root.mkdir()
            binary = root / "implementation.pyd"
            binary.write_bytes(b"native implementation")
            module = SimpleNamespace(__path__=[str(root)], __file__=None)
            rows = runtime.module_native_identity("fake", module)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["bytes"], len(b"native implementation"))
            self.assertEqual(len(rows[0]["sha256"]), 64)
            self.assertEqual(rows[0]["role"], "fake/package0/implementation.pyd")
            self.assertNotIn(str(root), rows[0]["role"])

    def test_identity_rejects_pure_python_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pure"
            root.mkdir()
            (root / "__init__.py").write_text("value = 1\n", encoding="utf-8")
            module = SimpleNamespace(__path__=[str(root)], __file__=None)
            with self.assertRaisesRegex(RuntimeError, "no discoverable native"):
                runtime.module_native_identity("pure", module)

    def test_fresh_identity_bypasses_cached_native_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fake_package"
            root.mkdir()
            binary = root / "implementation.pyd"
            binary.write_bytes(b"first implementation")
            module = SimpleNamespace(__path__=[str(root)], __file__=None)
            expected = runtime.module_native_identity("changing", module)
            binary.write_bytes(b"second implementation")
            self.assertEqual(
                runtime.module_native_identity("changing", module), expected
            )
            with self.assertRaisesRegex(RuntimeError, "changed"):
                runtime.verify_module_native_identity(
                    "changing", module, expected
                )


if __name__ == "__main__":
    unittest.main()
