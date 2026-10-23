# SPDX-License-Identifier: GPL-3.0-only
"""Golden outputs of game3d_log_report over saved first-launch logs of real games.

Each fixture in log_report_goldens/manifest.json is a game's scrubbed ReShade.log, with the host log of its session
when it was saved, made by scrub_game3d_log.py. Its golden is the report's exact output on it. A report change that
moves a golden fails here; when the change is intended, rewrite the goldens and review their diff:

    python tools/reshade/test_game3d_log_report_golden.py --update
"""

import contextlib
import gzip
import io
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game3d_log_report as report  # noqa: E402
import scrub_game3d_log as scrub  # noqa: E402

UPDATE = False


def logs(fixture):
    """The fixture's gzipped logs as (game log, host log or None) paths."""
    host = scrub.GOLDENS / f'{fixture["name"]}.sunshine.log.gz'
    return scrub.GOLDENS / f'{fixture["name"]}.ReShade.log.gz', host if fixture['host_log'] else None


def run_report(fixture, directory: Path) -> tuple[str, int]:
    """The report's output and exit code on the fixture, from a ReShade.log with the saved log's modification time
    (the report dates the host log and tells a running game from it)."""
    game, host = logs(fixture)
    log = directory / 'ReShade.log'
    log.write_bytes(gzip.decompress(game.read_bytes()))
    written = datetime.fromisoformat(fixture['mtime']).timestamp()
    os.utime(log, (written, written))
    argv = [str(log)]
    if host:
        (directory / 'sunshine.log').write_bytes(gzip.decompress(host.read_bytes()))
        argv += ['--host-log', str(directory / 'sunshine.log')]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = report.main(argv)
    return output.getvalue(), code


class GoldenReports(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.fixtures = scrub.load_manifest()
        self.assertTrue(self.fixtures, f'no fixtures in {scrub.MANIFEST}')

    def test_reports_match_their_goldens(self):
        for fixture in self.fixtures:
            golden = scrub.GOLDENS / f'{fixture["name"]}.txt'
            with self.subTest(fixture['name'], game=fixture['game']), TemporaryDirectory() as directory:
                text, code = run_report(fixture, Path(directory))
                self.assertEqual(code, 1 if '\nOverall: FAIL' in text else 0)
                if UPDATE:
                    golden.write_text(text, encoding='utf-8')
                    continue
                self.assertTrue(golden.exists(), f'{golden.name} is missing: run with --update')
                self.assertEqual(golden.read_text(encoding='utf-8'), text)

    def test_fixtures_are_scrubbed(self):
        # The repository is public: a fixture holds nothing that scrubbing would still replace, nor this machine's
        # user or computer name.
        names = scrub.local_names()
        for fixture in self.fixtures:
            for path in filter(None, logs(fixture)):
                with self.subTest(path.name):
                    text = gzip.decompress(path.read_bytes()).decode('utf-8')
                    self.assertEqual(scrub.leftovers(text, names)[:5], [])

    def test_every_file_belongs_to_a_fixture(self):
        if UPDATE:
            self.skipTest('--update writes the goldens this checks')
        names = [fixture['name'] for fixture in self.fixtures]
        self.assertEqual(len(set(names)), len(names))
        expected = {'manifest.json', '.gitattributes'}
        for fixture in self.fixtures:
            expected |= {path.name for path in filter(None, logs(fixture))} | {f'{fixture["name"]}.txt'}
        self.assertEqual(sorted(path.name for path in scrub.GOLDENS.iterdir()), sorted(expected))


if __name__ == '__main__':
    if '--update' in sys.argv:
        sys.argv.remove('--update')
        UPDATE = True
    unittest.main()
