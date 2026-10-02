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
from typing import NamedTuple

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
    r'Sunshine UI protection: runtime=\S+ .*?detection=(?P<detection>\w+) .*?sampled_source=(?P<source>\d+) '
    r'sampled_covered=(?P<covered>\d+) sampled_pixels=(?P<pixels>\d+) '
    r'sampled_candidates=(?P<candidates>0x[0-9a-fA-F]+) sampled_alpha_covered=(?P<alpha>\d+/\d+/\d+/\d+) '
    r'(?:sampled_alpha_invalid=(?P<invalid>\d+/\d+/\d+/\d+) )?trusted_alpha=(?P<trusted>0x[0-9a-fA-F]+)'
    r'(?: sampled_ui_layer=(?P<layer>\d))? '
    r'sampled_hudless=\{changed=(?P<changed>\d+) unchanged=(?P<unchanged>\d+) invalid=(?P<hudless_invalid>\d+)')
# Hidden-scene fields of the same line, absent from older logs.
SCENE = re.compile(
    r'sampled_alpha_opaque=(?P<opaque>\d+/\d+) sampled_scene=\{n=(?P<n>\d+) d=(?P<d>-?[0-9.]+) valid=(?P<valid>\d) '
    r'ran=(?P<ran>\d) verdict=(?P<verdict>\w+)\} sampled_hudless_scene=\{n=\d+ d=(?P<hudless_d>-?[0-9.]+) '
    r'valid=(?P<hudless_valid>\d)\} scene_hold=(?P<hold>\d+) shadow=(?P<shadow>\d) shadow_hidden_ms=(?P<hidden_ms>\d+)')
TRUST = re.compile(r'Sunshine UI protection: (restored alpha trust|alpha trust is now) (0x[0-9a-fA-F]+)')
LOSS = re.compile(r'sampled_only=\{revision=(\d+) found=1 cause=(\w+)')
READINESS = re.compile(r'Sunshine depth readiness: (lost|recovered) reason=(\w+)')
UNAVAILABLE_MS = re.compile(r'\bunavailable_ms=(\d+)')
PROVIDER = re.compile(r'\bprovider=(\w+)')
SELECTION = re.compile(r'\bselection=(\w+)')
RUNTIME = re.compile(r'\bruntime=(0x[0-9a-fA-F]+)')
# Either placement controller; the first word after the colon is its state.
PLACEMENT = re.compile(r'Sunshine 3D (raw automation|Streamline scale): (\w+);')
DEPTH_STATUS = re.compile(r'Sunshine Streamline depth: (\w+);')
HITCH = re.compile(r'Game 3D hitch: (.+?) took ([0-9.]+) ms')
NGX = re.compile(r'Sunshine NGX depth: .*?evaluations=(\d+) nominations=(\d+) copy_recorded=(\d+) '
                 r'metadata_only=(\d+)')
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
LOG_GATE_S = 1.0  # A changed controller state waits this long after its previous line (diagnostic_log_gate.h).
RAW_PLACED = ('ready', 'holding_reference')  # Raw automation states that place the scene (raw_scene_policy.h).
# A depth selection that names a game-side cause.
SELECTION_HINTS = {'inactive_views': 'the game supplied no depth'}
RESOLVE_S = 10.0  # A trust dispute is handled when that channel's trust is revoked this soon.
DEDICATED, PRESENTED = (0, 1), (2, 3)
UI_LAYER_SOURCE = 4  # alpha_auto_policy source of the offscreen UI layer in slot 1.
# Hidden-scene evidence (docs/reshade-sbs.md): a hidden run without a decided source this long is an uncovered hidden
# scene.
SHADOW_HIDDEN_WARN_MS = 500


def seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def clock(value: float, ms: bool = False) -> str:
    millis = round(value * 1000) % 86_400_000 if ms else int(value % 86400) * 1000
    text = f'{millis // 3_600_000:02}:{millis // 60_000 % 60:02}:{millis // 1000 % 60:02}'
    return f'{text}.{millis % 1000:03}' if ms else text


class Scene(NamedTuple):
    """A sample's hidden-scene evidence and the render's holds (docs/reshade-sbs.md, hidden-scene evidence)."""
    opaque: tuple[int, ...]  # Alpha slots 0 and 1 at least 254/255.
    n: int
    d: float
    valid: bool
    ran: bool
    verdict: str  # Presented image: none, hidden, ambiguous or visible.
    hudless_d: float
    hudless_valid: bool
    hold: int  # Routes held by the render that logged: 1 layer (source 8), 2 HUD-less (source 9).
    shadow: bool  # First-run shadow measuring with the gates closed.
    hidden_ms: int  # Longest hidden run without a decided source since the previous line.


class UISample(NamedTuple):
    t: float
    detection: str
    # 1-4: alpha slot 0-3 decided, 5: HUD-less difference, 6: full frame flat (exact pair), 8: full frame over a
    # hidden scene by the layer route, 9: by the HUD-less route. 7 is retired.
    source: int
    covered: int
    pixels: int
    candidates: int
    alpha: tuple[int, ...]
    trusted: int  # Slot bits the GPU treated as trusted.
    ui_layer: bool  # Slot 1 held the offscreen UI layer rather than a tagged UI color.
    hudless: tuple[int, ...]  # changed, unchanged, invalid.
    invalid: tuple[int, ...] | None = None  # Per alpha slot; absent in older logs.
    scene: Scene | None = None  # Absent in older logs.

    def trust_source(self, slot: int) -> int:
        return UI_LAYER_SOURCE if slot == 1 and self.ui_layer else slot


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
class DepthEpisode:
    """One 'Sunshine depth readiness' loss while the export streamed, to its recovery."""
    start: float
    reason: str
    provider: str
    selection: str
    end: float | None = None
    ending: str = 'open'  # recovered, paused (export inactive) or open (the log ended).
    unavailable_ms: int | None = None  # The add-on's own duration, from its recovered line.


@dataclass
class Unplaced:
    """A run of placement-controller lines that did not place the scene, to the next line that did.

    Its lines can be held up to a second by the add-on's diagnostic log gate, so its ends are approximate."""
    start: float
    end: float
    before: str  # Depth provider when placement was last ready, or '' when unknown.
    after: str = ''  # Depth provider when this run ended placed, or '' when unknown or it never did.
    states: list[tuple[float, str]] = field(default_factory=list)  # Each line's time and controller state.
    with_depth: list[tuple[float, float]] = field(default_factory=list)  # Parts streamed with depth available.
    cause: str = ''  # The switch, loss or export start that began it, or '' when the log names none.

    def depth_s(self) -> float:
        return sum(b - a for a, b in self.with_depth)

    def long(self) -> bool:
        """Longer with depth than the designed recalibration after a switch."""
        return self.depth_s() > SETTLE_S

    def calibration(self) -> bool:
        """The designed recalibration: it followed a named cause and had depth for at most SETTLE_S."""
        return bool(self.with_depth and self.cause) and not self.long()

    def state(self) -> str:
        """The state its lines reported for the longest time."""
        held = Counter()
        for (t, state), (until, _) in zip(self.states, self.states[1:] + [(self.end, '')]):
            held[state] += until - t
        return held.most_common(1)[0][0]


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
    overlay: list[float] = field(default_factory=list)
    fg_switches: list[tuple[float, int]] = field(default_factory=list)
    inactive: Counter = field(default_factory=Counter)
    inactive_first: dict[str, float] = field(default_factory=dict)
    coverage: tuple[int, ...] | None = None
    projection_ready: int = 0
    raw_ready: int = 0
    raw_scales: list[float] = field(default_factory=list)
    camera_valid: bool = False
    ui: list[UISample] = field(default_factory=list)
    trust_events: list[tuple[float, str, int]] = field(default_factory=list)
    losses: dict[int, tuple[float, str]] = field(default_factory=dict)
    readiness: Counter = field(default_factory=Counter)  # Losses while the export streamed.
    depth_episodes: list[DepthEpisode] = field(default_factory=list)
    streamed: list[list] = field(default_factory=list)  # From a generation to the next export inactive.
    placement: list[tuple[float, bool, str, str]] = field(default_factory=list)  # t, placed, state, provider.
    statuses: Counter = field(default_factory=Counter)
    hitches: list[tuple[float, str, float]] = field(default_factory=list)
    timing: tuple | None = None
    ngx: tuple[int, ...] | None = None
    warnings: list[tuple[float, str]] = field(default_factory=list)


def parse(lines) -> Session:
    s = Session()
    previous_output: dict[str, tuple] = {}
    offset = 0.0
    last_raw = None
    provider = ''
    open_episodes: dict[str, DepthEpisode] = {}  # Each runtime traces its own loss and recovery.

    def pause(t: float, ending: str) -> None:
        if s.streamed and s.streamed[-1][1] is None:
            s.streamed[-1][1] = t
        for episode in open_episodes.values():
            episode.end, episode.ending = t, ending
        open_episodes.clear()

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
        if 'ReShade overlay opened' in text:
            s.overlay.append(t)
        if 'Registered add-on "Sunshine 3D"' in text:
            s.addon = True
        if 'add-on GPU renderer ready' in text:
            s.renderer = True
        if found := GENERATION.search(text):
            width, height, api, color = found.group(2), found.group(3), found.group(5), found.group(6)
            s.generation = f'{width}x{height} {color} {api} (generation {found.group(1)})'
            s.settle.append(t)
            if not s.streamed or s.streamed[-1][1] is not None:
                s.streamed.append([t, None])
        if found := INACTIVE.search(text):
            pause(t, 'paused')  # Nothing is streamed from here until the next generation.
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
            g = found.groupdict()

            def counts(text: str) -> tuple[int, ...]:
                return tuple(int(v) for v in text.split('/'))
            scene = None
            if evidence := SCENE.search(text):
                e = evidence.groupdict()
                scene = Scene(counts(e['opaque']), int(e['n']), float(e['d']), e['valid'] == '1', e['ran'] == '1',
                              e['verdict'], float(e['hudless_d']), e['hudless_valid'] == '1', int(e['hold']),
                              e['shadow'] == '1', int(e['hidden_ms']))
            s.ui.append(UISample(t, g['detection'], int(g['source']), int(g['covered']), int(g['pixels']),
                                 int(g['candidates'], 16), counts(g['alpha']), int(g['trusted'], 16),
                                 g['layer'] == '1',
                                 (int(g['changed']), int(g['unchanged']), int(g['hudless_invalid'])),
                                 counts(g['invalid']) if g['invalid'] else None, scene))
        if found := TRUST.search(text):
            s.trust_events.append((t, found.group(1), int(found.group(2), 16)))
        if found := LOSS.search(text):
            s.losses.setdefault(int(found.group(1)), (t, found.group(2)))
        if found := READINESS.search(text):
            named = PROVIDER.search(text)
            if named and named.group(1) != 'unknown':
                provider = named.group(1)
            runtime = RUNTIME.search(text)
            key = runtime.group(1).lower() if runtime else ''
            if found.group(1) == 'lost' and s.streamed and s.streamed[-1][1] is None:
                s.readiness[found.group(2)] += 1
                selection = SELECTION.search(text)
                if key not in open_episodes:
                    open_episodes[key] = DepthEpisode(t, found.group(2), named.group(1) if named else 'unknown',
                                                      selection.group(1) if selection else 'unknown')
                    s.depth_episodes.append(open_episodes[key])
            elif found.group(1) == 'recovered' and (episode := open_episodes.pop(key, None)):
                unavailable = UNAVAILABLE_MS.search(text)
                episode.end, episode.ending = t, 'recovered'
                episode.unavailable_ms = int(unavailable.group(1)) if unavailable else None
        if found := PLACEMENT.search(text):
            state, raw = found.group(2), found.group(1) == 'raw automation'
            s.placement.append((t, state in RAW_PLACED if raw else state == 'ready',
                                ('raw ' if raw else 'projection ') + state, provider))
        if found := DEPTH_STATUS.search(text):
            s.statuses[found.group(1)] += 1
        if found := NGX.search(text):
            s.ngx = tuple(int(v) for v in found.groups())
        if found := HITCH.search(text):
            s.hitches.append((t, found.group(1), float(found.group(2))))
        if found := TIMING.search(text):
            s.timing = (int(found.group(1)), float(found.group(2)), float(found.group(3)),
                        int(found.group(4)), float(found.group(5)), float(found.group(6)))
    if s.last is not None:
        pause(s.last, 'open')
    return s


def unplaced(s: Session) -> list[Unplaced]:
    """Runs without placement, each with the parts that streamed while depth was available."""
    runs: list[Unplaced] = []
    current: Unplaced | None = None
    ready_provider = ''
    for t, placed, state, provider in s.placement:
        if not placed:
            current = current or Unplaced(t, t, ready_provider)
            current.states.append((t, state))
        else:
            if current:
                current.end, current.after = t, provider
                runs.append(current)
                current = None
            ready_provider = provider
    if current and s.last is not None:
        current.end = s.last
        runs.append(current)
    episodes = sorted(s.depth_episodes, key=lambda e: e.start)
    for run in runs:
        for a, b in s.streamed:
            a, b = max(a, run.start), min(b, run.end)
            for e in episodes:
                if e.start < b and e.end > a:
                    if e.start > a:
                        run.with_depth.append((a, e.start))
                    a = max(a, e.end)
            if b > a:
                run.with_depth.append((a, b))
        run.cause = cause(s, run)
    return runs


def cause(s: Session, run: Unplaced) -> str:
    """What the log names as starting a run: a depth-provider or FG switch, a depth loss or the export starting.

    The controller line that opened the run may have been held for LOG_GATE_S, so the cause may lead it by that."""
    lead = run.start - LOG_GATE_S
    if run.before and run.after and run.before != run.after:
        return f'after provider switch ({run.before} to {run.after})'
    if any(lead <= t <= run.start for t, _ in s.fg_switches):
        return 'after frame generation switch'
    if any(e.start <= run.start and e.end >= lead for e in s.depth_episodes):
        return 'after depth loss'
    if any(lead <= a < run.end for a, _ in s.streamed):
        return 'after the export started'
    return ''


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


def windows(s: Session, predicate) -> list[list[float]]:
    """Time ranges of consecutive settled intervals matching predicate."""
    ranges: list[list[float]] = []
    for interval in s.intervals:
        if not settled(s, interval) or not predicate(interval):
            continue
        if ranges and interval.start - ranges[-1][1] < 1.0:
            ranges[-1][1] = interval.end
        else:
            ranges.append([interval.start, interval.end])
    return ranges


def span(a: float, b: float, ms: bool = False) -> str:
    return f'{clock(a, ms)}-{clock(b, ms)}'


def depth_gaps(s: Session, ranges: list[list[float]]) -> list[str]:
    """Each gap window as the add-on's own loss episodes inside it, or the window when it logged none."""
    times = []
    for a, b in ranges:
        inside = [e for e in s.depth_episodes if e.start < b and e.end > a]
        if not inside:
            times.append(span(a, b))
        for e in inside:
            hint = SELECTION_HINTS.get(e.selection)
            took = f' ({e.unavailable_ms} ms)' if e.unavailable_ms is not None else ''
            times.append(f'{span(e.start, e.end, True)}{took} {e.reason}, {e.provider} selection={e.selection}'
                         + (f' ({hint})' if hint else '')
                         + {'paused': ', until the export paused', 'open': ', until the log ended'}.get(e.ending, ''))
    return times


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
        gaps = depth_gaps(s, windows(s, lambda i: i.start >= depth_start.start and i.missing * 10 > i.published))
        add(Check('WARN' if gaps else 'PASS', 'Depth',
                  f'fresh from {clock(depth_start.start)}; '
                  f'{100 * missing / pub:.1f}% of later publications without depth'
                  + ('; gaps outside FG switches and resets' if gaps else ''), gaps))
        # Flat output is depth without placement. Within runs without placement that each followed a named
        # cause and had depth for at most SETTLE_S in total it is the designed calibration; a longer or
        # unexplained run, or flat output while placed, warns.
        runs = [r for r in unplaced(s) if r.with_depth]

        def calibrating(i: Interval) -> bool:
            found = [r for r in runs if r.start < i.end and r.end > i.start]
            return bool(found) and all(r.calibration() for r in found)

        def flagged(i: Interval) -> bool:
            return i.start >= depth_start.start and i.flat * 4 > i.published

        def why(r: Unplaced) -> str:
            unexplained = '' if r.cause else ', no switch, depth loss or export start before it'
            return f'{r.state()} {span(r.start, r.end)} ({r.depth_s():.1f} s with depth{unexplained})'

        flats = windows(s, lambda i: flagged(i) and not calibrating(i))
        calibrations = windows(s, lambda i: flagged(i) and calibrating(i))
        flat = sum(i.flat for i in after)
        add(Check('WARN' if flats else 'PASS', 'Placement flat',
                  f'{100 * flat / pub:.1f}% of publications showed the colour frame with depth'
                  + ('; flat windows outside FG switches and resets' if flats else ''),
                  [span(a, b) + ''.join(f', {why(r)}' for r in runs
                                        if not r.calibration() and r.start < b and r.end > a) for a, b in flats]))
        if calibrations:
            seen = [r for r in runs if any(r.start < b and r.end > a for a, b in calibrations)]
            add(Check('INFO', 'Placement calibration',
                      ('1 flat window was a calibration' if len(calibrations) == 1 else
                       f'{len(calibrations)} flat windows were calibrations')
                      + f' with depth for at most {SETTLE_S:g} s',
                      [f'{span(r.start, r.end, True)} {r.cause}' for r in seen][:6]))

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

    failing = ('conflicting_state', 'incomplete_state', 'missing_state', 'failed', 'unsupported_state',
               'unsupported_resource', 'unsupported_lifetime')
    conflicts = {k: v for k, v in s.statuses.items() if k in failing}
    if conflicts:
        named = ', '.join(f'{k} {v}' for k, v in sorted(conflicts.items()))
        if 'incomplete_state' in conflicts:
            named += ' (incomplete_state: the source was blocked by a split barrier or a layout without a legacy state)'
        add(Check('WARN', 'Capture status', named))
    if s.ngx and s.ngx[0] and not s.ngx[2] and not s.ngx[3]:
        add(Check('WARN', 'NGX depth', f'{s.ngx[0]} DLSS evaluations recorded no depth copy'))
    if s.readiness:
        add(Check('INFO', 'Depth losses', ', '.join(f'{k} {v}' for k, v in s.readiness.most_common())))

    unusual = {k: v for k, v in s.inactive.items() if k not in ROUTINE_INACTIVE}
    add(Check('FAIL' if 'device_removed' in unusual else 'WARN' if unusual else 'PASS', 'Export',
              ', '.join(f'{k} {v}' for k, v in s.inactive.most_common()) or 'never paused',
              [f'{clock(s.inactive_first[k])} {k}' for k in unusual]))

    # A runtime reset or FG switch rebuilds the export ring and opening the
    # ReShade overlay prepares its compositor: one slow present each is expected.
    expected_marks = s.settle + s.overlay
    expected = [h for h in s.hitches if any(0 <= h[0] - mark <= SETTLE_S for mark in expected_marks)]
    unexpected = [h for h in s.hitches if h not in expected]
    if unexpected:
        worst = sorted(unexpected, key=lambda h: -h[2])[:3]
        add(Check('WARN', 'Present hitches', f'{len(unexpected)} outside resets, FG switches and overlay opening, '
                                             f'worst {worst[0][2]:.1f} ms',
                  [f'{clock(t)} {what} {ms:.1f} ms' for t, what, ms in worst]))
    else:
        add(Check('PASS', 'Present hitches', f'{len(expected)} at resets, FG switches or overlay opening' if expected
                  else 'none'))

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
    # dedicated UI channel is offered: that flattens scene as UI. Nor may a
    # trusted channel cover the whole frame while an exact HUD-less pair shows
    # the scene (Stellar Blade's opaque tagged UI color, before 2026-10).
    overrides, flattened, disputes, handled = [], [], [], []

    def revoked(t: float, source: int) -> float | None:
        return next((when for when, _, bits in s.trust_events
                     if 0 <= when - t <= RESOLVE_S and not bits & (1 << source)), None)

    def judge(sample: UISample, text: str, slot: int, unresolved: list) -> None:
        when = revoked(sample.t, sample.trust_source(slot))
        (handled if when is not None else unresolved).append(
            text + (f', trust revoked {clock(when)}' if when is not None else ''))

    for u in s.ui:
        if not u.pixels:
            continue
        dedicated = [u.alpha[c] for c in DEDICATED if u.candidates & u.trusted & (1 << c)]
        if dedicated and u.source in (3, 4):
            overrides.append(f'{clock(u.t)} source {u.source} covered {100 * u.covered / u.pixels:.0f}%')
        for c in PRESENTED:
            if dedicated and u.trusted & (1 << c) and min(abs(u.alpha[c] - d) for d in dedicated) * 10 >= u.pixels:
                judge(u, f'{clock(u.t)} channel {c} {100 * u.alpha[c] / u.pixels:.0f}% vs UI '
                         f'{100 * dedicated[0] / u.pixels:.1f}%', c, disputes)
        _, unchanged, invalid = u.hudless
        slot = u.source - 1
        exact_pair = u.candidates & 48 == 48 and not invalid
        if (0 <= slot < 4 and u.trusted & (1 << slot) and u.covered * 100 >= u.pixels * 99 and exact_pair
                and unchanged * 2 >= u.pixels):
            judge(u, f'{clock(u.t)} channel {slot}{" (UI layer)" if u.ui_layer and slot == 1 else ""} covered '
                     f'{100 * u.covered / u.pixels:.0f}% while HUD-less showed {100 * unchanged / u.pixels:.0f}% '
                     f'of the scene', slot, flattened)
    # A selective dedicated UI channel rejected for invalid pixels (for the UI
    # layer, color beyond twice its alpha) while no mask protected the frame.
    rejected = []
    for u in s.ui:
        if u.invalid is None or u.source or not u.pixels:
            continue
        for c in DEDICATED:
            if u.candidates & (1 << c) and u.invalid[c] and 0 < u.alpha[c] and u.alpha[c] * 10 < u.pixels * 9:
                rejected.append(f'{clock(u.t)} channel {c}{" (UI layer)" if u.ui_layer and c == 1 else ""} '
                                f'{100 * u.alpha[c] / u.pixels:.1f}% covered, {u.invalid[c]} invalid pixels'
                                f'{", trusted" if u.trusted & (1 << c) else ""}')
    if any(u.invalid is not None for u in s.ui):
        add(Check('WARN' if rejected else 'PASS', 'UI channel admission',
                  f'{len(rejected)} samples left the frame unprotected after rejecting a selective UI channel '
                  'for invalid pixels' if rejected else 'no selective UI channel rejected for invalid pixels',
                  rejected[:6]))
    states = Counter(u.detection for u in s.ui)
    add(Check('FAIL' if overrides or flattened else 'WARN' if disputes else 'PASS', 'UI protection',
              ', '.join(f'{k} {v}' for k, v in states.most_common())
              + ('; presented alpha decided over a trusted UI channel' if overrides else '')
              + ('; a trusted UI channel flattened the visible scene' if flattened else '')
              + ('; trusted presented alpha disagrees with the UI channel' if disputes else '')
              + ('; a contradicted channel lost its trust' if handled and not disputes and not flattened else ''),
              (overrides + flattened or disputes or handled)[:6]))
    scene_checks(s, add)


def scene_checks(s: Session, add) -> None:
    """Hidden-scene evidence: how often full-frame UI covered a hidden scene, and runs of the presented frame
    reading hidden while no UI source decided, which nothing protected (the first-run shadow reports them).

    A sample whose frame was full-frame UI (8 or 9) and whose own evidence read the presented frame visible is a
    route's exit: that verdict releases the hold at once, so each hidden scene that ends shows one, and a false
    hidden verdict shows one too. Exits are counted, not judged; many short episodes deserve a look."""
    scenes = [u for u in s.ui if u.scene]
    if not scenes:
        return
    covered = Counter(u.source for u in scenes if u.source in (8, 9))
    exits = sum(1 for u in scenes if u.source in (8, 9) and u.scene.valid and u.scene.verdict == 'visible')
    uncovered = [f'{clock(u.t)} {u.scene.hidden_ms} ms{" (first-run shadow)" if u.scene.shadow else ""}'
                 for u in scenes if not u.source and u.scene.hidden_ms >= SHADOW_HIDDEN_WARN_MS]
    detail = (f'layer route (8) in {covered[8]} samples, HUD-less route (9) in {covered[9]}, '
              f'{exits} released by a visible verdict; {sum(u.scene.ran for u in scenes)} of {len(scenes)} '
              'samples measured')
    if uncovered:
        detail += f'; the presented frame read hidden for at least {SHADOW_HIDDEN_WARN_MS} ms with no UI source'
    add(Check('WARN' if uncovered else 'INFO', 'Hidden scene', detail, uncovered[:6]))


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
