#!/usr/bin/env python3
"""The embedded Game 3D mirror must never become a separately edited algorithm."""

from pathlib import Path
import subprocess
import tempfile
import unittest

import generate_ui_pin_band as generator


class SharedUIPinBandTests(unittest.TestCase):
    def test_repository_mirror_is_current(self):
        self.assertTrue(generator.update(generator.OWNER, generator.GAME3D, True))

    def test_owner_changes_are_materialized_without_changing_surrounding_source(self):
        owner = "float owner = 1.0;\n"
        before = "prefix\n" + generator.BEGIN + "\nstale\n" + generator.END + "\nsuffix\n"
        after = generator.materialized(before, owner)
        self.assertEqual(after, "prefix\n" + generator.BEGIN + "\n" + owner + generator.END + "\nsuffix\n")
        self.assertEqual(generator.materialized(after, owner), after)

    def test_missing_duplicate_and_reversed_markers_are_rejected(self):
        begin, end = generator.BEGIN, generator.END
        for source in ("", begin, end, begin + end + begin, begin + end + end, end + begin):
            with self.subTest(source=source), self.assertRaises(ValueError):
                generator.materialized(source, "owner\n")

    def test_invalid_owner_is_rejected(self):
        source = generator.BEGIN + "\nold\n" + generator.END
        for owner in ("no final newline", generator.BEGIN + "\n", generator.END + "\n"):
            with self.subTest(owner=owner), self.assertRaises(ValueError):
                generator.materialized(source, owner)

    def test_check_rejects_drift_without_writing_and_update_preserves_crlf(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner, mirror = Path(temporary) / "owner.hlsl", Path(temporary) / "game.hlsl"
            owner.write_bytes(b"new owner\n")
            original = ("prefix\n" + generator.BEGIN + "\nold owner\n" + generator.END + "\nsuffix\n").replace("\n", "\r\n").encode()
            mirror.write_bytes(original)
            self.assertFalse(generator.update(owner, mirror, True))
            self.assertEqual(mirror.read_bytes(), original)
            self.assertTrue(generator.update(owner, mirror, False))
            self.assertTrue(generator.update(owner, mirror, True))
            self.assertIn(b"new owner\r\n", mirror.read_bytes())
            self.assertNotIn(b"\n", mirror.read_bytes().replace(b"\r\n", b""))
            after = mirror.read_bytes()
            self.assertTrue(generator.update(owner, mirror, False))
            self.assertEqual(mirror.read_bytes(), after)

    def test_cmake_check_rejects_drift_and_duplicate_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner, mirror = Path(temporary) / "owner.hlsl", Path(temporary) / "game.hlsl"
            owner.write_bytes(b"current owner\n")
            good = generator.BEGIN + "\ncurrent owner\n" + generator.END + "\n"
            for source, expected in ((good, 0), (good.replace("current owner", "drift"), 1),
                                     (good + generator.BEGIN, 1), (good + generator.END, 1)):
                with self.subTest(source=source):
                    mirror.write_bytes(source.encode())
                    result = subprocess.run(["cmake", f"-DSHARED_UI_PIN_OWNER={owner}",
                                             f"-DSHARED_UI_PIN_GAME3D={mirror}", "-P",
                                             str(Path(generator.__file__).with_name("check_ui_pin_band.cmake"))],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
