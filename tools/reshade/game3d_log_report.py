# SPDX-License-Identifier: GPL-3.0-only
"""First-run readiness report for a Game 3D session.

Reads a game's ReShade.log (and optionally Sunshine's host log) and reports
PASS/WARN/FAIL for each known failure signature, with the times to look at.
It only reads existing log lines; it adds no diagnostics to the add-on.

    python tools/reshade/game3d_log_report.py <ReShade.log | game folder> [--host-log sunshine.log]

Exit code: 0 when nothing failed, 1 when a check failed, 2 when the log is unusable.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

LINE = re.compile(r'^(\d{2}):(\d{2}):(\d{2}):(\d{3}) \[\s*\d+\] \| (\w+)\s*\| (.*)$')
OUTPUT = re.compile(
    r'Sunshine SBS output: published=(\d+) fg=(\d+) scene_flat=(\d+) scene_fading=(\d+) '
    r'published_fresh_depth=(\d+) published_reused_depth=(\d+) published_depth_missing=(\d+) '
    r'\(unavailable=(\d+) reuse_after_gap=(\d+) reuse_other_source=(\d+)\) runtime=(0x[0-9a-fA-F]+)')
COVERAGE = re.compile(
    r'Sunshine list lifecycle: admissions covered=(\d+) \(states observed=(\d+) declared=(\d+)\) '
    r'not_open=\{unknown=(\d+) closed=(\d+) pass=(\d+)')
GENERATION = re.compile(r'Sunshine SBS: generation (\d+), (\d+)x(\d+) full SBS, DXGI (\d+), (\w+), (\w+)')
INACTIVE = re.compile(r'Sunshine SBS: export inactive \((\w+)\)')
FG_MODE = re.compile(r'Sunshine Streamline frame generation: viewport=\d+ mode=(\d+)')
SCALE = re.compile(r'Sunshine 3D Streamline scale: (\w+);')
RAW = re.compile(r'Sunshine 3D raw automation: (\w+);.*?stereo_scale=([0-9.e+-]+)')
CAMERA = re.compile(r'Sunshine Streamline camera availability: .*recent_valid_projection=(\d+)')
UI = re.compile(
    r'Sunshine UI protection: runtime=\S+ .*?detection=(\w+) .*?sampled_source=(\d+) sampled_covered=(\d+) '
    r'sampled_pixels=(\d+) sampled_candidates=(0x[0-9a-fA-F]+) sampled_alpha_covered=(\d+)/(\d+)/(\d+)/(\d+) '
    r'trusted_alpha=(0x[0-9a-fA-F]+)')
TRUST = re.compile(r'Sunshine UI protection: (restored alpha trust|alpha trust is now) (0x[0-9a-fA-F]+)')
LOSS = re.compile(r'sampled_only=\{revision=(\d+) found=1 cause=(\w+)')
READINESS = re.compile(r'Sunshine depth readiness: (lost|recovered) reason=(\w+)')
DEPTH_STATUS = re.compile(r'Sunshine Streamline depth: (\w+);')
HITCH = re.compile(r'Game 3D hitch: (.+?) took ([0-9.]+) ms')
TIMING = re.compile(r'Sunshine Game 3D timing: presents=(\d+) cpu_ms=\{mean=([0-9.]+) max=([0-9.]+)\}'
                    r'.*?gpu_frames=(\d+)'
                    r' gpu_ms mean/max=\{total=([0-9.]+)/([0-9.]+)')
LOADED = re.compile(r"loaded from '.*' into '(.*)'")
HOST_LINE = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.(\d{3})\]: (\w+): (.*)$')

# ReShade and game noise that says nothing about Game 3D.
BENIGN = (
    'Successfully compiled', 'IDirectInput8W::CreateDevice failed', 'is inconsistent',
    'Add-ons are still loaded', 'Game 3D hitch',
)
# Export pauses that are part of normal play rather than faults.
ROUTINE_INACTIVE = {'not_foreground', 'runtime_reset', 'no_consumer', 'present_without_render', 'runtime_gone'}
SETTLE_S = 3.0  # Recalibration and holds after an FG switch or runtime reset.
DEDICATED, PRESENTED = (0, 1), (2, 3)


def seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def clock(value: float) -> str:
    value %= 86400
    return f'{int(value // 3600):02}:{int(value % 3600 // 60):02}:{int(value % 60):02}'


@dataclass
class Interval:
    start: float
    end: float
    published: int
    fg: int
    flat: int
    missing: int
    fresh: int


@dataclass
class Session:
    exe: str = ''
    first: float | None = None
    last: float | None = None
    exited: bool = False
    addon: bool = False
    renderer: bool = False
    generation: str = ''
    intervals: list[Interval] = field(default_factory=list)
    settle: list[float] = field(default_factory=list)
    fg_switches: list[tuple[float, int]] = field(default_factory=list)
    inactive: Counter = field(default_factory=Counter)
    inactive_first: dict[str, float] = field(default_factory=dict)
    coverage: tuple[int, ...] | None = None
    projection_ready: int = 0
    raw_ready: int = 0
    raw_scales: list[float] = field(default_factory=list)
    camera_valid: bool = False
    ui: list[tuple[float, ...]] = field(default_factory=list)
    trust_events: list[tuple[float, str, int]] = field(default_factory=list)
    losses: dict[int, tuple[float, str]] = field(default_factory=dict)
    readiness: Counter = field(default_factory=Counter)
    statuses: Counter = field(default_factory=Counter)
    hitches: list[tuple[float, str, float]] = field(default_factory=list)
    timing: tuple | None = None
    warnings: list[tuple[float, str]] = field(default_factory=list)


def parse(lines) -> Session:
    s = Session()
    previous_output: dict[str, tuple] = {}
    offset = 0.0
    last_raw = None
    for line in lines:
        m = LINE.match(line.rstrip('\n'))
        if not m:
            continue
        t = seconds(*m.group(1, 2, 3, 4)) + offset
        if last_raw is not None and t < last_raw - 43200:  # Past midnight.
            offset += 86400
            t += 86400
        last_raw = t
        level, text = m.group(5), m.group(6)
        s.first = t if s.first is None else s.first
        s.last = t
        if level in ('WARN', 'ERROR') and not any(b in text for b in BENIGN):
            s.warnings.append((t, f'{level} {text[:160]}'))
        if (found := LOADED.search(text)) and not s.exe:
            s.exe = os.path.basename(found.group(1))
        if 'Finished exiting' in text:
            s.exited = True
        if 'Registered add-on "Sunshine 3D"' in text:
            s.addon = True
        if 'add-on GPU renderer ready' in text:
            s.renderer = True
        if found := GENERATION.search(text):
            width, height, api, color = found.group(2), found.group(3), found.group(5), found.group(6)
            s.generation = f'{width}x{height} {color} {api} (generation {found.group(1)})'
            s.settle.append(t)
        if found := INACTIVE.search(text):
            reason = found.group(1)
            s.inactive[reason] += 1
            s.inactive_first.setdefault(reason, t)
            if reason == 'runtime_reset':
                s.settle.append(t)
        if found := FG_MODE.search(text):
            mode = int(found.group(1))
            if not s.fg_switches or s.fg_switches[-1][1] != mode:
                s.fg_switches.append((t, mode))
                s.settle.append(t)
        if found := OUTPUT.search(text):
            values = tuple(int(v) for v in found.groups()[:7])
            runtime = found.group(11)
            before = previous_output.get(runtime)
            if before and values[0] >= before[1][0]:
                d = [a - b for a, b in zip(values, before[1])]
                if d[0]:
                    s.intervals.append(Interval(before[0], t, d[0], d[1], d[2], d[6], d[4]))
            previous_output[runtime] = (t, values)
        if found := COVERAGE.search(text):
            s.coverage = tuple(int(v) for v in found.groups())
        if found := SCALE.search(text):
            s.projection_ready += found.group(1) == 'ready'
        if found := RAW.search(text):
            if found.group(1) == 'ready':
                s.raw_ready += 1
                s.raw_scales.append(float(found.group(2)))
        if found := CAMERA.search(text):
            s.camera_valid |= found.group(1) == '1'
        if found := UI.search(text):
            g = found.groups()
            s.ui.append((t, g[0], int(g[1]), int(g[2]), int(g[3]), int(g[4], 16),
                         tuple(int(v) for v in g[5:9]), int(g[9], 16)))
        if found := TRUST.search(text):
            s.trust_events.append((t, found.group(1), int(found.group(2), 16)))
        if found := LOSS.search(text):
            s.losses.setdefault(int(found.group(1)), (t, found.group(2)))
        if (found := READINESS.search(text)) and found.group(1) == 'lost':
            s.readiness[found.group(2)] += 1
        if found := DEPTH_STATUS.search(text):
            s.statuses[found.group(1)] += 1
        if found := HITCH.search(text):
            s.hitches.append((t, found.group(1), float(found.group(2))))
        if found := TIMING.search(text):
            s.timing = (int(found.group(1)), float(found.group(2)), float(found.group(3)),
                        int(found.group(4)), float(found.group(5)), float(found.group(6)))
    return s


@dataclass
class Check:
    status: str
    name: str
    detail: str
    times: list[str] = field(default_factory=list)


def settled(s: Session, interval: Interval) -> bool:
    """Less than half of the interval lies in the settle time after an FG switch or reset."""
    covered, cursor = 0.0, interval.start
    for mark in sorted(s.settle):
        a, b = max(mark, cursor), min(mark + SETTLE_S, interval.end)
        if b > a:
            covered += b - a
            cursor = b
    return covered * 2 < interval.end - interval.start


def windows(s: Session, predicate) -> list[str]:
    """Time ranges of consecutive settled intervals matching predicate."""
    ranges: list[list[float]] = []
    for interval in s.intervals:
        if not settled(s, interval) or not predicate(interval):
            continue
        if ranges and interval.start - ranges[-1][1] < 1.0:
            ranges[-1][1] = interval.end
        else:
            ranges.append([interval.start, interval.end])
    return [f'{clock(a)}-{clock(b)}' for a, b in ranges]


def evaluate(s: Session) -> list[Check]:
    checks: list[Check] = []
    add = checks.append
    if not s.addon or not s.renderer:
        add(Check('FAIL', 'Add-on', 'Sunshine 3D add-on ' + ('loaded but the native renderer never became ready'
                                                             if s.addon else 'never registered')))
        return checks
    add(Check('PASS', 'Add-on', f'loaded; output {s.generation or "unknown"}'))

    published = sum(i.published for i in s.intervals)
    # Depth has started once most of an interval's publications carry it.
    depth_start = next((i for i in s.intervals if i.fresh * 2 >= i.published), None)
    if not published:
        add(Check('FAIL', 'Depth', 'nothing was published (export never active)'))
    elif not depth_start:
        add(Check('FAIL', 'Depth', f'{published} publications, none with fresh depth'))
    else:
        after = [i for i in s.intervals if i.start >= depth_start.start]
        pub = sum(i.published for i in after) or 1
        missing = sum(i.missing for i in after)
        gaps = windows(s, lambda i: i.start >= depth_start.start and i.missing * 10 > i.published)
        add(Check('WARN' if gaps else 'PASS', 'Depth',
                  f'fresh from {clock(depth_start.start)}; '
                  f'{100 * missing / pub:.1f}% of later publications without depth'
                  + ('; gaps outside FG switches and resets' if gaps else ''), gaps))
        flats = windows(s, lambda i: i.start >= depth_start.start and i.flat * 4 > i.published)
        flat = sum(i.flat for i in after)
        add(Check('WARN' if flats else 'PASS', 'Placement flat',
                  f'{100 * flat / pub:.1f}% of publications showed the colour frame with depth'
                  + ('; flat windows outside FG switches and resets' if flats else ''), flats))

    if s.coverage:
        covered, observed, declared, unknown, closed, passes = s.coverage
        rejected = unknown + closed + passes
        total = covered + rejected
        status = 'FAIL' if total and not covered else 'WARN' if rejected * 100 > total else 'PASS'
        add(Check(status, 'Capture coverage',
                  f'{covered} covered ({observed} observed, {declared} declared states); '
                  f'not open: unknown {unknown}, closed {closed}, render pass {passes}'))
    else:
        add(Check('INFO', 'Capture coverage', 'no capture admissions logged (no API depth source)'))

    if s.projection_ready:
        add(Check('PASS', 'Placement', 'camera projection drives placement'))
    elif s.raw_ready:
        spread = max(s.raw_scales) / max(min(s.raw_scales), 1e-9) if s.raw_scales else 1.0
        add(Check('WARN' if s.camera_valid else 'INFO', 'Placement',
                  'raw controller only' + (', although the game supplies a valid camera' if s.camera_valid else '')
                  + f'; its gain varied {spread:.0f}x'))
    else:
        add(Check('WARN', 'Placement', 'neither projection nor raw placement became ready'))

    ui_checks(s, add)

    tokens = sum(1 for _, cause in s.losses.values() if cause == 'tokens_busy')
    causes = Counter(cause for _, cause in s.losses.values())
    minutes = max((s.last - s.first) / 60.0, 1.0) if s.first is not None else 1.0
    bad = {c: n for c, n in causes.items() if c != 'lifecycle'}
    add(Check('WARN' if tokens > minutes / 2 or any(c != 'tokens_busy' for c in bad) else 'PASS',
              'Observation losses', ', '.join(f'{c} {n}' for c, n in sorted(causes.items())) or 'none',
              [f'{clock(t)} {c}' for r, (t, c) in sorted(s.losses.items()) if c != 'lifecycle'][:6]))

    failing = ('conflicting_state', 'failed', 'unsupported_state', 'unsupported_resource', 'unsupported_lifetime')
    conflicts = {k: v for k, v in s.statuses.items() if k in failing}
    if conflicts:
        add(Check('WARN', 'Capture status', ', '.join(f'{k} {v}' for k, v in sorted(conflicts.items()))))
    if s.readiness:
        add(Check('INFO', 'Depth losses', ', '.join(f'{k} {v}' for k, v in s.readiness.most_common())))

    unusual = {k: v for k, v in s.inactive.items() if k not in ROUTINE_INACTIVE}
    add(Check('FAIL' if 'device_removed' in unusual else 'WARN' if unusual else 'PASS', 'Export',
              ', '.join(f'{k} {v}' for k, v in s.inactive.most_common()) or 'never paused',
              [f'{clock(s.inactive_first[k])} {k}' for k in unusual]))

    if s.hitches:
        worst = sorted(s.hitches, key=lambda h: -h[2])[:3]
        add(Check('WARN', 'Present hitches', f'{len(s.hitches)}, worst {worst[0][2]:.1f} ms',
                  [f'{clock(t)} {what} {ms:.1f} ms' for t, what, ms in worst]))
    else:
        add(Check('PASS', 'Present hitches', 'none'))

    if s.timing:
        presents, cpu_mean, cpu_max, gpu_frames, gpu_mean, gpu_max = s.timing
        add(Check('INFO' if gpu_frames else 'WARN', 'Game 3D cost',
                  f'CPU {cpu_mean:.2f} ms mean ({cpu_max:.1f} max); '
                  + (f'GPU {gpu_mean:.2f} ms mean ({gpu_max:.1f} max)' if gpu_frames else 'no GPU timing samples')))
    if s.fg_switches:
        add(Check('INFO', 'Frame generation',
                  ', '.join(f'{clock(t)} {"on" if mode else "off"}' for t, mode in s.fg_switches)))
    if s.warnings:
        add(Check('WARN', 'Log warnings', f'{len(s.warnings)} unexpected WARN/ERROR lines',
                  [f'{clock(t)} {w}' for t, w in s.warnings[:5]]))
    return checks


def ui_checks(s: Session, add) -> None:
    for t, kind, bits in s.trust_events:
        add(Check('INFO', 'UI trust', f'{clock(t)} {kind} 0x{bits:x}'))
    if not s.ui:
        add(Check('INFO', 'UI protection', 'no UI protection samples'))
        return
    # A presented alpha (sources 3 and 4) must never decide while a trusted
    # dedicated UI channel is offered: that flattens scene as UI.
    overrides, disputes = [], []
    for t, _, source, covered, pixels, candidates, alpha, trusted in s.ui:
        if not pixels:
            continue
        dedicated = [alpha[c] for c in DEDICATED if candidates & trusted & (1 << c)]
        if dedicated and source in (3, 4):
            overrides.append(f'{clock(t)} source {source} covered {100 * covered / pixels:.0f}%')
        if dedicated and max(dedicated) * 10 < pixels * 9:
            for c in PRESENTED:
                if trusted & (1 << c) and alpha[c] * 10 >= max(dedicated) * 10 + pixels:
                    disputes.append(f'{clock(t)} channel {c} {100 * alpha[c] / pixels:.0f}% vs UI '
                                    f'{100 * max(dedicated) / pixels:.1f}%')
    states = Counter(sample[1] for sample in s.ui)
    add(Check('FAIL' if overrides else 'WARN' if disputes else 'PASS', 'UI protection',
              ', '.join(f'{k} {v}' for k, v in states.most_common())
              + ('; presented alpha decided over a trusted UI channel' if overrides else '')
              + ('; trusted presented alpha disagrees with the UI channel' if disputes else ''),
              (overrides or disputes)[:6]))


def host_checks(path: Path, start: datetime, end: datetime) -> list[Check]:
    checks: list[Check] = []
    stalls, connects, errors, fits = [], 0, [], []
    span = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        m = HOST_LINE.match(line)
        if not m:
            continue
        when = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
        span.append(when)
        if not start - timedelta(minutes=1) <= when <= end + timedelta(minutes=1):
            continue
        t = when.hour * 3600 + when.minute * 60 + when.second
        level, text = m.group(3), m.group(4)
        if 'input_producer=' in text:
            stalls.append(f'{clock(t)} ' + ('encoder-side' if 'done_by' in text else
                                            'upstream GPU' if 'upstream' in text else 'unknown'))
        if 'ReShade SBS connected' in text:
            connects += 1
        if 'ReShade SBS: game eyes' in text:
            fits.append((level, f'{clock(t)} {text[:140]}'))
        if level == 'Error':
            errors.append(f'{clock(t)} {text[:140]}')
    if span and not any(start - timedelta(minutes=1) <= w <= end + timedelta(minutes=1) for w in span):
        covers = f'{span[0]:%m-%d %H:%M} to {span[-1]:%m-%d %H:%M}'
        return [Check('INFO', 'Host log', f'no lines from this session (it covers {covers})')]
    link = f'ReShade SBS connected {connects} time(s)' if connects else 'the host never connected to Game 3D'
    checks.append(Check('PASS' if connects else 'WARN', 'Host link', link))
    for level, text in fits:
        checks.append(Check('FAIL' if level == 'Warning' else 'WARN', 'Host size', text))
    checks.append(Check('WARN' if stalls else 'PASS', 'Encoder stalls', str(len(stalls)) if stalls else 'none',
                        stalls[:6]))
    if errors:
        checks.append(Check('WARN', 'Host errors', f'{len(errors)}', errors[:5]))
    return checks


def find_log(path: Path) -> Path:
    if path.is_file():
        return path
    found = sorted(path.rglob('ReShade.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not found:
        raise FileNotFoundError(f'no ReShade.log under {path}')
    return found[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('log', type=Path, help="ReShade.log or the game's folder")
    parser.add_argument('--host-log', type=Path, help="Sunshine's log (for example E:/ApolloDev/config/sunshine.log)")
    args = parser.parse_args(argv)
    try:
        log = find_log(args.log)
        session = parse(log.read_text(encoding='utf-8', errors='replace').splitlines())
    except OSError as error:
        print(f'cannot read the log: {error}')
        return 2
    if session.first is None:
        print(f'{log}: no ReShade log lines')
        return 2
    running = not session.exited and time.time() - log.stat().st_mtime < 60
    duration = session.last - session.first
    state = 'still running' if running else 'exited' if session.exited else 'ended without exiting (crash or kill?)'
    print(f'Game 3D readiness: {session.exe or log}  {clock(session.first)} to {clock(session.last)} '
          f'({int(duration // 60)}m{int(duration % 60):02}s, {state})')
    checks = evaluate(session)
    if args.host_log:
        # ReShade logs times without a date; the log's last write dates the session.
        written = datetime.fromtimestamp(log.stat().st_mtime)
        end = datetime.combine(written.date(), datetime.min.time()) + timedelta(seconds=session.last % 86400)
        if end > written + timedelta(minutes=5):
            end -= timedelta(days=1)
        checks += host_checks(args.host_log, end - timedelta(seconds=duration), end)
    if not running and not session.exited:
        checks.append(Check('WARN', 'Session', 'the log ends without ReShade exiting'))
    for check in checks:
        print(f'[{check.status:4}] {check.name}: {check.detail}')
        for when in check.times:
            print(f'         {when}')
    fails = sum(c.status == 'FAIL' for c in checks)
    warns = sum(c.status == 'WARN' for c in checks)
    print(f'Overall: {"FAIL" if fails else "WARN" if warns else "PASS"} ({fails} fail, {warns} warn)')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
