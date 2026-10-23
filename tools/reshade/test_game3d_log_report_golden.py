# SPDX-License-Identifier: GPL-3.0-only
"""Golden outputs of game3d_log_report over saved first-launch logs of real games.

Each fixture in log_report_goldens/manifest.json is a game's scrubbed ReShade.log, with the host log of its session
when it was saved, made by scrub_game3d_log.py. Its golden is the report's exact output on it and then its exit code.
A report change that moves a golden fails here; when the change is intended, rewrite the goldens and review their
diff:

    python tools/reshade/test_game3d_log_report_golden.py --update
"""

import contextlib
import gzip
import io
import os
import shutil
import sys
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game3d_log_report as report  # noqa: E402
import scrub_game3d_log as scrub  # noqa: E402

UPDATE = False


def run_report(fixture, directory: Path) -> str:
    """The report's output on the fixture and then its exit code, from a ReShade.log with the saved log's modification
    time (the report dates the host log and tells a running game from it)."""
    game, *host = scrub.fixture_logs(fixture)
    log = directory / 'ReShade.log'
    log.write_bytes(gzip.decompress(game.read_bytes()))
    written = datetime.fromisoformat(fixture['mtime']).timestamp()
    os.utime(log, (written, written))
    argv = [str(log)]
    if host:
        (directory / 'sunshine.log').write_bytes(gzip.decompress(host[0].read_bytes()))
        argv += ['--host-log', str(directory / 'sunshine.log')]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = report.main(argv)
    return f'{output.getvalue()}Exit code: {code}\n'


class GoldenReports(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.fixtures = scrub.load_manifest()
        self.assertTrue(self.fixtures, f'no fixtures in {scrub.MANIFEST}')

    def test_reports_match_their_goldens(self):
        for fixture in self.fixtures:
            golden = scrub.GOLDENS / f'{fixture["name"]}.txt'
            with self.subTest(fixture['name'], game=fixture['game']), TemporaryDirectory() as directory:
                text = run_report(fixture, Path(directory))
                if UPDATE:
                    golden.write_text(text, encoding='utf-8')
                    continue
                self.assertTrue(golden.exists(), f'{golden.name} is missing: run with --update')
                self.assertEqual(golden.read_text(encoding='utf-8'), text)

    def test_fixtures_are_scrubbed(self):
        # The repository is public: a fixture holds nothing that the line rules would still replace. The check is the
        # same on every machine, whatever its user and computer names.
        for fixture in self.fixtures:
            for path in scrub.fixture_logs(fixture):
                with self.subTest(path.name):
                    self.assertEqual(scrub.leftovers(scrub.read_gzip(path))[:5], [])

    def test_local_names_keep_words_of_the_logs(self):
        # A full scrub replaces the user and computer names of the machine that wrote the log, but not a word the logs
        # already use: an account named Owner keeps the add-on's 'shared copy owner', one named dev 'gyan.dev'.
        with mock.patch.dict(os.environ, {'USERNAME': 'Owner', 'USER': 'dev', 'COMPUTERNAME': 'Q7-WORKSTATION'}):
            names = scrub.local_names()
        self.assertEqual(names, (('Q7-WORKSTATION', '<MACHINE>'),))
        self.assertEqual(scrub.scrub_line(r'share \\q7-workstation\games owner', None, names),
                         r'share \\<MACHINE>\games owner')

    def test_fixtures_scrub_again_unchanged(self):
        # The raw logs are not kept, so a new rule reaches the fixtures by scrubbing them again. A fixture's <GAME> is
        # no install folder to replace, a full scrub refuses the fixture, and --rescrub rewrites none of its bytes.
        with TemporaryDirectory() as directory:
            copy = Path(directory) / 'goldens'
            shutil.copytree(scrub.GOLDENS, copy)
            manifest = copy / 'manifest.json'
            with mock.patch.object(scrub, 'GOLDENS', copy), mock.patch.object(scrub, 'MANIFEST', manifest):
                for path in (path for fixture in self.fixtures for path in scrub.fixture_logs(fixture)):
                    with self.subTest(path.name):
                        lines = scrub.read_gzip(path).splitlines()
                        self.assertIsNone(scrub.install_folder(lines))
                        self.assertEqual([line for line in lines if scrub.scrub_line(line, '<GAME>') != line][:5], [])
                log = Path(directory) / 'ReShade.log'
                log.write_bytes(gzip.decompress(scrub.fixture_logs(self.fixtures[0])[0].read_bytes()))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(scrub.main(['probe', 'Probe', str(log)]), 2)
                    self.assertEqual(scrub.rescrub(), 0)
            self.assertEqual(sorted(path.name for path in copy.iterdir()),
                             sorted(path.name for path in scrub.GOLDENS.iterdir()))
            for path in scrub.GOLDENS.iterdir():
                self.assertEqual((copy / path.name).read_bytes(), path.read_bytes(), path.name)

    def test_every_file_belongs_to_a_fixture(self):
        if UPDATE:
            self.skipTest('--update writes the goldens this checks')
        names = [fixture['name'] for fixture in self.fixtures]
        self.assertEqual(len(set(names)), len(names))
        expected = {'manifest.json', '.gitattributes'}
        for fixture in self.fixtures:
            expected |= {path.name for path in scrub.fixture_logs(fixture)} | {f'{fixture["name"]}.txt'}
        self.assertEqual(sorted(path.name for path in scrub.GOLDENS.iterdir()), sorted(expected))


if __name__ == '__main__':
    if '--update' in sys.argv:
        sys.argv.remove('--update')
        UPDATE = True
    unittest.main()
