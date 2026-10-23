# SPDX-License-Identifier: GPL-3.0-only
"""Turns a game's saved ReShade.log (and the host log of its session) into a public report fixture.

The fixtures in log_report_goldens/ are committed to a public repository. This replaces the game's install folder, any
other drive path but the Windows folder, user-profile paths, the user and machine names, IP and MAC addresses,
adapter LUIDs, device instance paths, DirectInput device GUIDs, audio endpoint ids, client names and the host's display
list with stable placeholders. It keeps only the host log lines the report can read for the session (its window, dated
as game3d_log_report.py dates it, and the stream rates set before it) and stores both logs gzipped, then records the
fixture in manifest.json. Record its report afterwards with
`python tools/reshade/test_game3d_log_report_golden.py --update`.

    python tools/reshade/scrub_game3d_log.py <name> "<game>" <ReShade.log> [--host-log sunshine.log]

After a rule is added, `--rescrub` applies it to the committed fixtures in place; their goldens should not move.

Exit code: 0 when the fixture was written, 2 when a log is unusable or already scrubbed or the game's install folder is
unknown.
"""

from __future__ import annotations

import argparse
import gzip
import ipaddress
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game3d_log_report as report  # noqa: E402

GOLDENS = Path(__file__).resolve().parent / 'log_report_goldens'
MANIFEST = GOLDENS / 'manifest.json'

# A path separator: '\', '/' or the JSON-escaped '\\' of the host log's quoted paths.
SEP = r'(?:\\\\|\\|/)'
# Folders between an install folder and its executable: Unreal's <Project>\Binaries\Win64, bin\x64_dx12 and the like.
BUILD_FOLDERS = frozenset(('binaries', 'bin', 'win64', 'wingdk', 'x64', 'retail'))
# Host lines this far outside the session's window are dropped; the report reads one minute around it.
HOST_MARGIN = timedelta(minutes=2)

USER_PROFILE = re.compile(rf'(?<![A-Za-z0-9])[A-Za-z]:{SEP}+Users{SEP}+[^\\/\'"\s]+', re.I)
QUOTED_PATH = re.compile(rf'([\'"])((?<![A-Za-z0-9])[A-Za-z]:{SEP}[^\'"]*)\1')
BARE_PATH = re.compile(rf'(?<![A-Za-z0-9<])[A-Za-z]:{SEP}[^\s\'"|<>,;\])]*')
SYSTEM_PATH = re.compile(rf'[A-Za-z]:{SEP}+windows(?:{SEP}|$)', re.I)
DEVICE_PATH = re.compile(r'(?:\\\\){1,2}\?(?:\\\\|\\)[A-Za-z]+#[^\s\]\'",]+')
OCTET = r'(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)'
IPV4 = re.compile(rf'(?<![\w.])(?:{OCTET}\.){{3}}{OCTET}(?![\w.])')
# Dotted versions look like addresses: 'fixed-product=2.7.30.0', strings='2.7.30.0' or '(fixed 2.7.30.0)'. There
# only a local network's address is one.
VERSION_CONTEXT = re.compile(r'(?:[=\'"v]|fixed |version )$', re.I)
LOCAL_NETWORKS = tuple(ipaddress.ip_network(net) for net in
                       ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '169.254.0.0/16', '100.64.0.0/10'))
IPV6 = re.compile(r'(?<![\w:])[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}(?![\w:])')
MAC = re.compile(r'(?<![\w:-])(?:[0-9A-Fa-f]{2}([:-]))(?:[0-9A-Fa-f]{2}\1){4}[0-9A-Fa-f]{2}(?![\w:-])')
LUID = re.compile(r'(\bluid\b\W{0,3})(?:0x[0-9A-Fa-f]+(?:[:_-]0x[0-9A-Fa-f]+)?|[0-9A-Fa-f]{4,}(?:[:_-][0-9A-Fa-f]+)?)',
                  re.I)
# A client's name: 'client [<name>]', or 'for [<name>]' after a control message type ('type [IDX_...]: ...').
CLIENT = re.compile(r'(\bclient \[|\[IDX_\w+\]:.* for \[)([^\]]+)(\])')
AUDIO_ENDPOINT = re.compile(r'\{\d+\.\d+\.\d+\.\d+\}\.\{[0-9A-Fa-f-]{36}\}')
# A DirectInput device's instance GUID: a version-1 GUID on DirectInput's 'DEST' node (444553540000) that dates when
# this machine first saw the device. DirectInput's own constants (GUID_SysMouse, GUID_Joystick, its axes) share the
# node with clock sequence BFC7 and stay.
DEVICE_GUID = re.compile(r'\{[0-9A-F]{8}-[0-9A-F]{4}-1[0-9A-F]{3}-(?!BFC7-)[0-9A-F]{4}-444553540000\}', re.I)
# Host settings that name this machine or its displays.
HOST_SETTINGS = {'sunshine_name': '<MACHINE>', 'ar_glass_devices': '<DISPLAYS>'}
HOST_SETTING = re.compile(r"(config: '(\w+)' = )(.*)")
EMAIL = re.compile(r'\b[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|net|org|edu|gov|io|dev|me|cn|de|uk|jp)\b')


def install_folder(lines: list[str]) -> str | None:
    """The game's install folder: its executable's folder (ReShade's 'loaded ... into' line) above build folders."""
    for line in lines:
        if found := report.LOADED.search(line):
            folder = PureWindowsPath(found.group(1)).parent
            if not folder.drive:
                return None  # A scrubbed log's '<GAME>\...' or a relative path: no folder to replace.
            unreal = False
            while folder.name.lower() in BUILD_FOLDERS or folder.name.lower().startswith('x64_'):
                unreal = unreal or folder.name.lower() == 'binaries'
                folder = folder.parent
            if unreal and folder.parent != folder:
                folder = folder.parent  # <install>\<Project>\Binaries\Win64
            return None if folder == PureWindowsPath(folder.anchor) else str(folder)
    return None


def local_names() -> tuple[tuple[str, str], ...]:
    """This machine's user and computer names, which no fixture may contain, with their placeholders."""
    names = ((os.environ.get('USERNAME'), '<USER>'), (os.environ.get('USER'), '<USER>'),
             (os.environ.get('COMPUTERNAME'), '<MACHINE>'))
    return tuple(dict((name, placeholder) for name, placeholder in names if name and len(name) >= 3).items())


def drive_path(path: str) -> str:
    """A drive path outside the Windows folder keeps only its file name."""
    if SYSTEM_PATH.match(path):
        return path
    found = re.search(r'(\\\\|\\|/)([^\\/]*)$', path)
    return f'<PATH>{found.group(1)}{found.group(2)}' if found else '<PATH>'


def ipv4(found: re.Match) -> str:
    """An IPv4 address's placeholder, unless the dotted number is a version."""
    local = any(ipaddress.ip_address(found.group(0)) in net for net in LOCAL_NETWORKS)
    return found.group(0) if VERSION_CONTEXT.search(found.string[:found.start()]) and not local else '<IP>'


def ipv6(found: re.Match) -> str:
    """An IPv6 address's placeholder; colon-separated times and C++ scopes stay."""
    text = found.group(0)
    if '::' not in text and text.count(':') != 7 or not any(c.isdigit() for c in text):
        return text
    try:
        ipaddress.IPv6Address(text)
    except ValueError:
        return text
    return '<IP>'


def scrub_line(line: str, install: str | None = None, names: tuple[tuple[str, str], ...] = ()) -> str:
    """One log line with every identifying value replaced by its placeholder; idempotent. An install folder replaces
    nothing unless it has a drive and a folder below it: a bare root's pattern would match any separator."""
    folder = PureWindowsPath(install or '')
    if folder.drive and len(folder.parts) > 1:
        parts = [re.escape(part) for part in folder.parts]
        root = re.escape(folder.drive) + SEP + SEP.join(parts[1:])
        line = re.sub(rf'(?<![A-Za-z0-9]){root}(?=$|{SEP}|[\'"\s])', '<GAME>', line, flags=re.I)
    line = USER_PROFILE.sub('<USER_PROFILE>', line)
    line = DEVICE_PATH.sub('<DEVICE_PATH>', line)
    line = QUOTED_PATH.sub(lambda m: m.group(1) + drive_path(m.group(2)) + m.group(1), line)
    line = BARE_PATH.sub(lambda m: drive_path(m.group(0)), line)
    for name, placeholder in names:
        line = re.sub(rf'(?<!\w){re.escape(name)}(?!\w)', placeholder, line, flags=re.I)
    line = HOST_SETTING.sub(lambda m: m.group(1) + HOST_SETTINGS.get(m.group(2), m.group(3)), line)
    line = AUDIO_ENDPOINT.sub('<ENDPOINT>', line)
    line = DEVICE_GUID.sub('<DEVICE_GUID>', line)
    line = CLIENT.sub(r'\1<CLIENT>\3', line)
    line = EMAIL.sub('<EMAIL>', line)
    line = MAC.sub('<MAC>', line)
    line = LUID.sub(r'\1<LUID>', line)
    line = IPV4.sub(ipv4, line)
    return IPV6.sub(ipv6, line)


def leftovers(text: str, names: tuple[tuple[str, str], ...] = ()) -> list[str]:
    """The lines of a fixture that scrubbing would still change: a committed fixture has none."""
    return [line for line in text.splitlines() if scrub_line(line, None, names) != line]


def session_window(lines: list[str], written: datetime) -> tuple[datetime, datetime]:
    """The session's start and end on the host log's clock, dated by the log's last write as the report dates it."""
    session = report.parse(lines)
    if session.first is None:
        raise ValueError('no ReShade log lines')
    end = datetime.combine(written.date(), datetime.min.time()) + timedelta(seconds=session.last % 86400)
    if end > written + timedelta(minutes=5):
        end -= timedelta(days=1)
    return end - timedelta(seconds=session.last - session.first), end


def host_lines(lines: list[str], start: datetime, end: datetime) -> list[str]:
    """The host lines the report can read for this session: its window, and the stream rates set up to its end."""
    kept = []
    for line in lines:
        m = report.HOST_LINE.match(line)
        if not m:
            continue
        when = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
        rate = report.HOST_FPS.search(m.group(4)) or report.HOST_LIVE_FPS.search(m.group(4))
        if start - HOST_MARGIN <= when <= end + HOST_MARGIN or (rate and when <= end + HOST_MARGIN):
            kept.append(line)
    return kept


def write_gzip(path: Path, lines: list[str]) -> None:
    # A zero timestamp and no file name keep the bytes reproducible.
    path.write_bytes(gzip.compress(''.join(f'{line}\n' for line in lines).encode('utf-8'), 9, mtime=0))


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else []


def rescrub() -> int:
    """Applies the current rules to the committed fixtures in place. The raw logs are not kept; a fixture's install
    folder is already <GAME> and its host lines are already the session's, so only the line rules run again."""
    names = local_names()
    for fixture in load_manifest():
        paths = [GOLDENS / f'{fixture["name"]}.ReShade.log.gz']
        if fixture['host_log']:
            paths.append(GOLDENS / f'{fixture["name"]}.sunshine.log.gz')
        for path in paths:
            lines = gzip.decompress(path.read_bytes()).decode('utf-8').splitlines()
            scrubbed = [scrub_line(line, None, names) for line in lines]
            if scrubbed != lines:
                write_gzip(path, scrubbed)
                print(f'{path.name}: {sum(a != b for a, b in zip(lines, scrubbed))} lines scrubbed')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('name', nargs='?', help='fixture name, for example stellar_blade_1022')
    parser.add_argument('game', nargs='?', help='the game, for example "Stellar Blade"')
    parser.add_argument('log', nargs='?', type=Path,
                        help="the game's saved ReShade.log, with its original modification time")
    parser.add_argument('--host-log', type=Path, help="Sunshine's log of the same session")
    parser.add_argument('--install', help="the game's install folder, when it is not found from its executable")
    parser.add_argument('--rescrub', action='store_true',
                        help='apply the current rules to the committed fixtures in place, after a rule is added')
    args = parser.parse_args(argv)
    if args.rescrub:
        if args.name or args.host_log or args.install:
            parser.error('--rescrub takes no other arguments')
        return rescrub()
    if not args.log:
        parser.error('name, game and log are required')
    if not re.fullmatch(r'[a-z0-9_]+', args.name):
        parser.error('name: use lower-case letters, digits and underscores')
    try:
        lines = args.log.read_text(encoding='utf-8', errors='replace').splitlines()
        written = datetime.fromtimestamp(args.log.stat().st_mtime).replace(microsecond=0)
        host = args.host_log.read_text(encoding='utf-8', errors='replace').splitlines() if args.host_log else None
        start, end = session_window(lines, written)
    except (OSError, ValueError) as error:
        print(f'cannot read the logs: {error}')
        return 2
    if any('<GAME>' in line for line in lines):
        # Its install folder is already <GAME> and its host lines are cut to the session: --rescrub reruns the rules.
        print(f'{args.log} is already a scrubbed fixture: run with --rescrub to apply new rules to the fixtures')
        return 2
    install = args.install or install_folder(lines)
    if not install:
        print('the install folder is unknown: pass --install')
        return 2
    names = local_names()
    GOLDENS.mkdir(exist_ok=True)
    write_gzip(GOLDENS / f'{args.name}.ReShade.log.gz', [scrub_line(line, install, names) for line in lines])
    host_path = GOLDENS / f'{args.name}.sunshine.log.gz'
    if host is not None:
        host = host_lines(host, start, end)
        write_gzip(host_path, [scrub_line(line, install, names) for line in host])
    else:
        host_path.unlink(missing_ok=True)
    entry = {'name': args.name, 'game': args.game, 'mtime': written.isoformat(), 'host_log': host is not None}
    manifest = sorted([e for e in load_manifest() if e['name'] != args.name] + [entry], key=lambda e: e['name'])
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    kept = f', {len(host)} host lines' if host is not None else ''
    print(f'{args.name}: {len(lines)} game lines{kept}; install folder {install} is <GAME>')
    return 0


if __name__ == '__main__':
    sys.exit(main())
