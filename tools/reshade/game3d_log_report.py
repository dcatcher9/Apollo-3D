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
    r'(?:sampled_alpha_invalid=(?P<invalid>\d+/\d+/\d+/\d+) )?'
    # Since S1 the accepted candidate bits and the offscreen UI layer's own counts; before it the trusted alpha slot
    # bits, with the layer in the UI colour slot when sampled_ui_layer=1.
    r'(?:accepted=(?P<accepted>0x[0-9a-fA-F]+) sampled_layer=\{covered=(?P<layer_covered>\d+) '
    r'invalid=(?P<layer_invalid>\d+) opaque=\d+\}'
    r'|trusted_alpha=(?P<trusted>0x[0-9a-fA-F]+)(?: sampled_ui_layer=(?P<layer>\d))?) '
    r'sampled_hudless=\{changed=(?P<changed>\d+) unchanged=(?P<unchanged>\d+) invalid=(?P<hudless_invalid>\d+)')
# Hidden-scene fields of the same line, absent from older logs.
SCENE = re.compile(
    r'sampled_alpha_opaque=(?P<opaque>\d+/\d+) sampled_scene=\{n=(?P<n>\d+) d=(?P<d>-?[0-9.]+) valid=(?P<valid>\d) '
    r'ran=(?P<ran>\d) verdict=(?P<verdict>\w+)\} sampled_hudless_scene=\{n=\d+ d=(?P<hudless_d>-?[0-9.]+) '
    r'valid=(?P<hudless_valid>\d)\} scene_hold=(?P<hold>\d+) shadow=(?P<shadow>\d) shadow_hidden_ms=(?P<hidden_ms>\d+)')
# Presentation fields of the same line; older logs lack some of them.
UI_RUNTIME = re.compile(r'Sunshine UI protection: runtime=(\S+)')
UI_MODE = re.compile(r'\bmode=(\w+)')
UI_RENDERED = re.compile(r'\brendered=(\d)')
UI_FG = re.compile(r'\bfg=(\d)')
UI_AVAILABILITY = re.compile(r'\bsource_availability=(\w+)')
# Acceptance changes: since S1 the accepted source signatures (docs/reshade-sbs.md, UI decision framework, A1), before
# it trusted source bits.
TRUST = re.compile(r'Sunshine UI protection: (restored accepted UI sources|accepted UI sources are now|'
                   r'restored alpha trust|alpha trust is now) ([^\s;]+)')
LEGACY_TRUST = re.compile(r'Sunshine UI protection: discarded (\d+) legacy UI trust entries')
# The session's cumulative exact UI counters (docs/reshade-sbs.md, UI counters), absent from older logs.
COUNTERS = re.compile(r'Sunshine UI counters: (.*)$')
COUNTER_FIELD = re.compile(r'(\w+)=(?:\{([^}]*)\}|(\S+))')
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
RESOLVE_S = 10.0  # A dispute is handled when that source's acceptance is revoked this soon.
# Alpha candidates in UISample order: UIAlpha, the UI colour tag, Backbuffer and current alpha (the log's
# sampled_alpha_covered), then the offscreen UI layer (sampled_layer); their candidate bits (ui_detection::candidate),
# acceptance kinds (ui_selection::kind) and the source each decides as.
ALPHA_NAMES = ('UI alpha', 'UI colour', 'Backbuffer alpha', 'current alpha', 'UI layer')
ALPHA_BITS = (0x1, 0x2, 0x4, 0x8, 0x40)
ALPHA_KINDS = ('ui_alpha', 'ui_color', 'backbuffer', 'current', 'ui_layer')
ALPHA_SOURCES = (1, 2, 3, 4, 10)
DRAW_ORDER = (0, 1, 4, 2, 3)  # S1: declared before inferred (ui_selection::draw_order).
DECLARED, INFERRED = (0, 1), (2, 3, 4)  # The game's own UI contract, and guesses.
DEDICATED, PRESENTED = (0, 1, 4), (2, 3)  # UI channels, whose clean empty pixels show no UI, and presented alpha.
# Trusted source bits of logs before S1, by acceptance kind.
LEGACY_TRUST_KINDS = {0x1: 'ui_alpha', 0x2: 'ui_color', 0x4: 'backbuffer', 0x8: 'current', 0x10: 'ui_layer'}
HUDLESS_PAIR = 48  # Candidate bits of the HUD-less pair.
# Unprotected time shorter than this is counted, not listed: alpha_trust_span_ms (game3d_alpha_auto.h), the span over
# which detection itself earns or loses confidence.
UNPROTECTED_MIN_S = 2.0
UI_LINE_PERIOD_S = 10.0  # An unchanged UI protection state is logged again this often while presenting (exporter.cpp).
# Hidden-scene evidence (docs/reshade-sbs.md): a hidden run without a decided source this long is an uncovered hidden
# scene.
SHADOW_HIDDEN_WARN_MS = 500
# Decided sources by number (decision texel 0, docs/reshade-sbs.md); 7 is retired.
SOURCE_NAMES = {0: 'no mask', 1: 'UI alpha', 2: 'UI colour', 3: 'Backbuffer alpha', 4: 'current alpha',
                5: 'HUD-less difference', 6: 'full frame (exact pair)', 8: 'full frame (layer route)',
                9: 'full frame (HUD-less route)', 10: 'UI layer'}
NO_MASK_REASONS = ('layer_aside', 'trusted_invalid', 'presented_blocked', 'ambiguous', 'difference_failed',
                   'gate_no_hold', 'no_candidate', 'other', 'unaccepted')
HOLD_KINDS = ('generated', 'inexact_after_exact', 'trusted_missing')
INACTIVE_REASONS = ('no_candidates', 'size', 'unprepared')


def seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def clock(value: float, ms: bool = False) -> str:
    millis = round(value * 1000) % 86_400_000 if ms else int(value % 86400) * 1000
    text = f'{millis // 3_600_000:02}:{millis // 60_000 % 60:02}:{millis // 1000 % 60:02}'
    return f'{text}.{millis % 1000:03}' if ms else text


def percent(n: int, total: int) -> str:
    value = 100 * n / total if total else 0.0
    return f'{value:.0f}%' if value >= 10 or not value else f'{value:.1f}%' if value >= 1 else f'{value:.2f}%'


def accepted_keys(value: str) -> tuple[str, ...]:
    """The accepted source keys of a logged acceptance value: since S1 'kind:format:space' signatures joined by
    commas, or 'none'; before it trusted source bits, read as their kinds."""
    if value.startswith('0x'):
        bits = int(value, 16)
        return tuple(kind for bit, kind in LEGACY_TRUST_KINDS.items() if bits & bit)
    return () if value == 'none' else tuple(value.split(','))


def counter_fields(text: str) -> dict[str, int]:
    """The numeric fields of a 'Sunshine UI counters' line, a group's fields as 'group.key'."""
    values: dict[str, int] = {}
    for key, group, value in COUNTER_FIELD.findall(text):
        if group:
            for inner, _, number in COUNTER_FIELD.findall(group):
                if number.isdigit():
                    values[f'{key}.{inner}'] = int(number)
        elif value.isdigit() and key != 'runtime':
            values[key] = int(value)
    return values


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
    # 1-4: UI alpha, UI colour tag, Backbuffer or current alpha decided, 10: the UI layer, 5: HUD-less difference, 6:
    # full frame flat (exact pair), 8: full frame over a hidden scene by the layer route, 9: by the HUD-less route. 7
    # is retired. Before S1 the layer decided as 2; parse() reads it as 10.
    source: int
    covered: int
    pixels: int
    candidates: int  # Candidate bits (ui_detection::candidate), the layer as 0x40 also before S1.
    alpha: tuple[int, ...]  # Covered pixels in ALPHA_NAMES order.
    accepted: int  # Accepted candidate bits pushed with the detection; before S1 the trusted alpha channels.
    hudless: tuple[int, ...]  # changed, unchanged, invalid.
    invalid: tuple[int, ...] | None = None  # In ALPHA_NAMES order; absent in older logs.
    scene: Scene | None = None  # Absent in older logs.
    runtime: str = ''
    mode: str = 'auto'
    rendered: bool = True
    availability: str = ''  # source_availability; absent in older logs.
    fg: bool = False  # Frame generation active.
    legacy: bool = False  # Logged before S1, when a trusted UI channel, the layer included, kept presented alpha out.

    def offered(self, c: int) -> bool:
        return bool(self.candidates & ALPHA_BITS[c])

    def is_accepted(self, c: int) -> bool:
        return bool(self.accepted & ALPHA_BITS[c])

    def valid(self, c: int) -> bool:
        """V1: at most 1% invalid pixels (unknown in older logs)."""
        return self.invalid is None or self.invalid[c] * 100 <= self.pixels

    def layer_without_alpha(self) -> bool:
        """An offscreen UI layer with no alpha anywhere but colour on more than 1% of pixels: premultiplied UI over
        transparent black cannot have colour without alpha, so V1 rejects it (Stellar Blade's SDR scene image).
        Mirrors source_alpha_ui_decision::layer_without_alpha (game3d_controls.h)."""
        return self.offered(4) and self.invalid is not None and not self.alpha[4] and not self.valid(4)

    def blocking(self, admitted: bool = False) -> bool:
        """An offered, accepted declared alpha keeps inferred alpha out (S1). Before S1 any trusted UI channel did;
        a layer without alpha was then set aside, and with admitted it does not count."""
        if not self.legacy:
            return any(self.offered(c) and self.is_accepted(c) for c in DECLARED)
        return any(self.offered(c) and self.is_accepted(c) and not (admitted and c == 4 and self.layer_without_alpha())
                   for c in DEDICATED)

    def kept_out(self) -> tuple[int, ...]:
        """The alpha candidates a blocking UI channel keeps out."""
        return PRESENTED if self.legacy else INFERRED

    def shows_no_ui(self, c: int) -> bool:
        """A UI channel offered clean and empty: the game's UI channel says there is no UI on screen."""
        if not self.offered(c) or self.alpha[c]:
            return False
        if self.invalid is not None:
            return not self.invalid[c]
        # Older logs lack invalid counts. A clean accepted channel at 0 decides, so with no source it was rejected.
        return not self.is_accepted(c)

    def unprotected(self) -> bool:
        """Auto rendered this frame without a UI mask, and no UI channel of the game showed that there was no UI.
        Mirrors source_alpha_ui_decision::unprotected() (game3d_controls.h): a source that decides, even empty, is a
        mask, also when source_availability, which names a missing source first, reads source_unavailable;
        'checking', unrendered frames and manual modes are not unprotected."""
        if self.mode != 'auto' or not self.rendered or self.detection == 'detected':
            return False
        if self.availability == 'source_unavailable':
            return True
        return self.detection == 'no_usable_mask' and not any(self.shows_no_ui(c) for c in DEDICATED)

    def pending(self) -> bool:
        """Auto rendered while its status sample was still collected or a source searched: as next_unprotected_since
        (game3d_controls.h), such a line neither starts nor ends an unprotected run."""
        return (self.mode == 'auto' and self.rendered and self.detection in ('checking', 'searching')
                and not self.unprotected())

    def why_unprotected(self) -> str:
        """Each offered candidate, in draw order, and why it gave no mask."""
        fg = 'FG on' if self.fg else 'FG off'
        if self.availability == 'source_unavailable' or not self.candidates:
            return f'{fg}: no UI source offered'
        blocking = self.blocking()
        parts = []
        for c in DRAW_ORDER:
            if not self.offered(c):
                continue
            covered, name = self.alpha[c], ALPHA_NAMES[c]
            invalid = self.invalid[c] if self.invalid is not None else 0
            if c == 4 and self.layer_without_alpha():
                parts.append(f'UI layer has colour but no alpha ({percent(invalid, self.pixels)} of pixels)')
            elif invalid:
                parts.append(f'{name} rejected ({percent(covered, self.pixels)} covered, '
                             f'{percent(invalid, self.pixels)} invalid)')
            elif c in self.kept_out() and blocking and (not self.legacy or covered * 10 < self.pixels * 9):
                parts.append(f'{name} kept out beside an accepted UI channel')
            else:
                parts.append(f'{name} covers {percent(covered, self.pixels)}'
                             + ('' if self.is_accepted(c) else ' (not accepted)'))
        if self.candidates & HUDLESS_PAIR == HUDLESS_PAIR:
            parts.append('HUD-less difference rejected')
        return f'{fg}: ' + '; '.join(parts)


def ui_sample(t: float, text: str, g: dict[str, str | None], scene: Scene | None) -> UISample:
    """One 'Sunshine UI protection' line; a line logged before S1 is normalized to the S1 candidates."""
    def counts(value: str) -> tuple[int, ...]:
        return tuple(int(v) for v in value.split('/'))

    def field_of(pattern: re.Pattern, default: str) -> str:
        return hit.group(1) if (hit := pattern.search(text)) else default
    alpha = counts(g['alpha'])
    invalid = counts(g['invalid']) if g['invalid'] else None
    candidates, source = int(g['candidates'], 16), int(g['source'])
    legacy = g['accepted'] is None
    if not legacy:
        accepted = int(g['accepted'], 16)
        alpha += (int(g['layer_covered']),)
        invalid = invalid + (int(g['layer_invalid']),) if invalid is not None else None
    elif g['layer'] == '1':
        # Before S1 the layer filled the UI colour slot: slot 1, candidate bit 2, source 2.
        def moved(bits: int) -> int:
            return bits & ~0x2 | (0x40 if bits & 0x2 else 0)
        alpha = (alpha[0], 0, alpha[2], alpha[3], alpha[1])
        invalid = (invalid[0], 0, invalid[2], invalid[3], invalid[1]) if invalid is not None else None
        candidates, accepted = moved(candidates), moved(int(g['trusted'], 16))
        source = 10 if source == 2 else source
    else:
        accepted = int(g['trusted'], 16)
        alpha += (0,)
        invalid = invalid + (0,) if invalid is not None else None
    return UISample(t, g['detection'], source, int(g['covered']), int(g['pixels']), candidates, alpha, accepted,
                    (int(g['changed']), int(g['unchanged']), int(g['hudless_invalid'])), invalid, scene,
                    field_of(UI_RUNTIME, ''), field_of(UI_MODE, 'auto'), field_of(UI_RENDERED, '1') == '1',
                    field_of(UI_AVAILABILITY, ''), field_of(UI_FG, '0') == '1', legacy)


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
    # Time, logged event, logged value and the accepted source keys after it (None for a discard of legacy entries).
    trust_events: list[tuple[float, str, str, tuple[str, ...] | None]] = field(default_factory=list)
    # The last 'Sunshine UI counters' line: the session's totals are cumulative over all its runtimes.
    counters: dict[str, int] | None = None
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
            scene = None
            if evidence := SCENE.search(text):
                e = evidence.groupdict()
                scene = Scene(tuple(int(v) for v in e['opaque'].split('/')), int(e['n']), float(e['d']),
                              e['valid'] == '1', e['ran'] == '1', e['verdict'], float(e['hudless_d']),
                              e['hudless_valid'] == '1', int(e['hold']), e['shadow'] == '1', int(e['hidden_ms']))
            s.ui.append(ui_sample(t, text, found.groupdict(), scene))
        if found := TRUST.search(text):
            s.trust_events.append((t, found.group(1), found.group(2), accepted_keys(found.group(2))))
        if found := LEGACY_TRUST.search(text):
            s.trust_events.append((t, 'discarded legacy UI trust entries', found.group(1), None))
        if (found := COUNTERS.search(text)) and 'auto_frames' in (values := counter_fields(found.group(1))):
            s.counters = values
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
    for t, kind, value, _ in s.trust_events:
        add(Check('INFO', 'UI trust', f'{clock(t)} {kind} {value}'))
    if not s.ui and s.counters is None:
        add(Check('INFO', 'UI protection', 'no UI protection samples'))
        return
    # Inferred alpha (Backbuffer, current alpha and the UI layer) must never decide while an accepted declared UI
    # channel is offered (S1): that flattens scene as UI. Nor may an accepted alpha cover the whole frame while an
    # exact HUD-less pair shows the scene (Stellar Blade's opaque tagged UI color, before 2026-10).
    overrides, flattened, disputes, handled = [], [], [], []

    def revoked(t: float, kind: str) -> float | None:
        """When an acceptance change within RESOLVE_S after t left fewer accepted sources of this kind."""
        before = None
        for when, _, _, keys in s.trust_events:
            if keys is None:
                continue
            n = sum(1 for key in keys if key.split(':')[0] == kind)
            if when < t:
                before = n
                continue
            if when - t > RESOLVE_S:
                break
            if not n or (before is not None and n < before):
                return when
            before = n
        return None

    def judge(sample: UISample, text: str, c: int, unresolved: list) -> None:
        when = revoked(sample.t, ALPHA_KINDS[c])
        (handled if when is not None else unresolved).append(
            text + (f', acceptance revoked {clock(when)}' if when is not None else ''))

    for u in s.ui:
        if not u.pixels:
            continue
        if u.blocking(admitted=True) and u.source in tuple(ALPHA_SOURCES[c] for c in u.kept_out()):
            overrides.append(f'{clock(u.t)} source {u.source} covered {100 * u.covered / u.pixels:.0f}%')
        # As the ledger's disagreement (A2): only an accepted UI channel with at most 1% invalid pixels and clean
        # accepted presented alpha dispute.
        masks = [u.alpha[c] for c in DEDICATED if u.offered(c) and u.is_accepted(c) and u.valid(c)]
        for c in PRESENTED:
            if (masks and u.offered(c) and u.is_accepted(c) and (u.invalid is None or not u.invalid[c])
                    and min(abs(u.alpha[c] - d) for d in masks) * 10 >= u.pixels):
                judge(u, f'{clock(u.t)} {ALPHA_NAMES[c]} {100 * u.alpha[c] / u.pixels:.0f}% vs UI '
                         f'{100 * masks[0] / u.pixels:.1f}%', c, disputes)
        _, unchanged, invalid = u.hudless
        exact_pair = u.candidates & HUDLESS_PAIR == HUDLESS_PAIR and not invalid
        if u.source in ALPHA_SOURCES:
            c = ALPHA_SOURCES.index(u.source)
            if u.is_accepted(c) and u.covered * 100 >= u.pixels * 99 and exact_pair and unchanged * 2 >= u.pixels:
                judge(u, f'{clock(u.t)} {ALPHA_NAMES[c]} covered {100 * u.covered / u.pixels:.0f}% while HUD-less '
                         f'showed {100 * unchanged / u.pixels:.0f}% of the scene', c, flattened)
    # A selective UI channel rejected for invalid pixels (for the UI layer, colour beyond its alpha headroom) while
    # no mask protected the frame. A channel without alpha carried no UI to reject (a UI layer with colour but no
    # alpha); the time such frames went unprotected is 'UI protection gaps'.
    rejected = []
    for u in s.ui:
        if u.invalid is None or u.source or not u.pixels:
            continue
        for c in DEDICATED:
            if u.offered(c) and u.invalid[c] and 0 < u.alpha[c] and u.alpha[c] * 10 < u.pixels * 9:
                rejected.append(f'{clock(u.t)} {ALPHA_NAMES[c]} {100 * u.alpha[c] / u.pixels:.1f}% covered, '
                                f'{u.invalid[c]} invalid pixels{", accepted" if u.is_accepted(c) else ""}')
    if any(u.invalid is not None for u in s.ui):
        add(Check('WARN' if rejected else 'PASS', 'UI channel admission',
                  f'{len(rejected)} samples left the frame unprotected after rejecting a selective UI channel '
                  'for invalid pixels' if rejected else 'no selective UI channel rejected for invalid pixels',
                  rejected[:6]))
    if s.counters is not None:
        counter_checks(s.counters, add, overrides, flattened, disputes, handled)
        gap_checks(s, add)
        scene_checks(s, add)
        return
    states = Counter(u.detection for u in s.ui)
    add(Check('FAIL' if overrides or flattened else 'WARN' if disputes else 'PASS', 'UI protection',
              ', '.join(f'{k} {v}' for k, v in states.most_common())
              + ('; inferred alpha decided beside an accepted declared UI channel' if overrides else '')
              + ('; an accepted UI source flattened the visible scene' if flattened else '')
              + ('; accepted presented alpha disagrees with an accepted UI channel' if disputes else '')
              + ('; a contradicted source lost its acceptance' if handled and not disputes and not flattened else ''),
              (overrides + flattened or disputes or handled)[:6]))
    gap_checks(s, add)
    scene_checks(s, add)


def counter_checks(c: dict[str, int], add, overrides_sampled: list[str], flattened_sampled: list[str],
                   disputes: list[str], handled: list[str]) -> None:
    """Checks from the add-on's exact per-frame UI counters (docs/reshade-sbs.md, UI counters).

    They replace the sampled invariants: every Auto frame is counted, not one per 100 ms sample. Sampled lines still
    give the times of examples, acceptance disputes, which no counter records, whether an accepted full-frame claim
    over the scene was resolved by a revocation, and the time-based gap and scene checks. Stages (S1-S6) and rule IDs
    (such as H1 and P1) named in the output are the UI decision framework's (docs/reshade-sbs.md, UI decision
    framework)."""
    def get(key: str) -> int:
        return c.get(key, 0)

    auto, detected = get('auto_frames'), get('detection_frames')
    held = sum(get(f'held.{k}') for k in HOLD_KINDS)
    inactive = sum(get(f'inactive.{k}') for k in INACTIVE_REASONS)
    reconciled = auto == detected + held + inactive
    add(Check('PASS' if reconciled else 'FAIL', 'UI counters',
              f'{auto} Auto frames through the last of {get("samples")} committed samples: {detected} detected, '
              f'{held} held, {inactive} without detection'
              + ('' if reconciled else f"; the add-on's own accounting does not reconcile ({auto} Auto frames, "
                                       f'{detected + held + inactive} detected, held or inactive)')))

    overrides, flattened = get('presented_over_dedicated'), get('trusted_full')
    # An accepted alpha covering the frame over an exact pair that shows the scene keeps deciding until repeated
    # contradictions revoke its acceptance (about 2 s), so a handled case counts trusted_full frames first. Such
    # frames fail only when a sample of them was not resolved by a revocation, or when no full claim was revoked.
    revoked = get('trust.revoked_full')
    unresolved = flattened and (flattened_sampled or not revoked)
    sources = [(source, get(f'decided.{source}')) for source in SOURCE_NAMES]
    add(Check('FAIL' if overrides or unresolved else 'WARN' if disputes else 'PASS', 'UI protection',
              'decided ' + (', '.join(f'{SOURCE_NAMES[k]} ({k}) {percent(n, detected)}' for k, n in sources if n)
                            or 'nothing') + ' of detection frames'
              + (f'; inferred alpha decided beside an accepted declared UI channel in {overrides} frames'
                 if overrides else '')
              + (f'; an accepted UI source covered the frame while an exact HUD-less pair showed the scene in '
                 f'{flattened} frames' + ('' if unresolved else f' before {revoked} revocations of such a source')
                 if flattened else '')
              + ('; accepted presented alpha disagrees with an accepted UI channel' if disputes else '')
              + ('; a contradicted source lost its acceptance' if handled and not disputes and not unresolved else ''),
              (overrides_sampled + flattened_sampled or disputes or handled)[:6]))

    full = {k: get(f'decided.{k}') for k in (6, 8, 9)}
    visible, routes = get('full_d.visible'), full[8] + full[9]
    # full_d counts the exact full change-set (6) with the held routes (8, 9). An exact full set decides without a
    # hold, and over a visible scene it is intended for an accepted exact pair (P1); it is wrong only before the
    # pair's first selective sample, which is an open question of the framework. So visible samples warn only when a
    # held route decided: the sample that releases a held route decided under the hold before its own evidence read
    # the scene visible, so each release counts one, and the counters cannot tell a release from a held route over a
    # visible scene (H1) until S2b counts releases.
    exact_note = ('an exact full change-set (6) over a visible scene is intended for an accepted exact pair (P1); '
                  'whether a pair decides one before its first selective sample is an open question')
    add(Check('WARN' if visible and routes else 'INFO' if visible else 'PASS', 'UI full frame',
              f'{sum(full.values())} full-frame frames (6: {full[6]}, 8: {full[8]}, 9: {full[9]}); samples that '
              f'decided one read the scene hidden {get("full_d.hidden")}, ambiguous {get("full_d.ambiguous")}, '
              f'visible {visible}, invalid or unmeasured {get("full_d.invalid")}; depth not current on '
              f'{get("full.depth_not_current")} detection frames'
              + ('' if not visible else
                 f'; {visible} full-frame samples read the scene visible: each release of a held hidden-scene '
                 'route (8, 9) shows one, more than one per hidden scene is a held route over a visible scene (H1); '
                 'releases are counted apart from S2b' + (f'; samples of source 6 are counted with them, and '
                                                          f'{exact_note}' if full[6] else '')
                 if routes else f'; {visible} samples of source 6 read the scene visible: {exact_note}')))

    # A whole-frame mask from alpha (an accepted source 1-4 or 10 covering at least 99%; only accepted sources decide)
    # pins the frame flat even over a visible scene: an accepted source's pin weight is saturate(8 alpha) at any
    # coverage (P1, the opacity ruling), so it is reported, not warned. The wrong cases have their own checks: an
    # accepted full claim over an exact pair that shows the scene and no revocation resolved (UI protection), and
    # unaccepted inferred alpha (UI inferred alpha). The add-on measures D as a diagnostic on the samples after one that
    # decided it, so the first sample of each episode is unmeasured.
    full_alpha, alpha_visible = get('full_alpha'), get('full_alpha_d.visible')
    if full_alpha:
        add(Check('INFO', 'UI full alpha',
                  f'{full_alpha} frames ({percent(full_alpha, detected)} of detection frames) decided a whole-frame '
                  f'alpha; samples that decided one read the scene hidden {get("full_alpha_d.hidden")}, ambiguous '
                  f'{get("full_alpha_d.ambiguous")}, visible {alpha_visible}, invalid or unmeasured '
                  f'{get("full_alpha_d.invalid")}'
                  + (f'; {alpha_visible} samples pinned an accepted whole-frame alpha flat over a visible scene, as '
                     'intended (P1)' if alpha_visible else '')))
    elif 'full_alpha' in c:
        add(Check('PASS', 'UI full alpha', 'no frame decided a whole-frame alpha'))

    # Since S1 only accepted candidates decide, so the word is zero by construction: a count is a defect. Counters
    # from before S1 (trust.opaque_set rather than trust.discarded) still let the untrusted pass decide.
    inferred, s1 = get('untrusted_inferred'), 'trust.discarded' in c
    add(Check(('FAIL' if s1 else 'WARN') if inferred else 'PASS', 'UI inferred alpha',
              f'{inferred} frames ({percent(inferred, detected)} of detection frames) decided from unaccepted inferred '
              'alpha (UI layer, Backbuffer or current alpha)'
              + ('; only accepted candidates decide since S1' if s1 else '; expected before S1') if inferred else
              'no frame decided from unaccepted inferred alpha'))
    inexact = get('inexact_difference')
    add(Check('WARN' if inexact else 'PASS', 'UI inexact difference',
              f'{inexact} frames ({percent(inexact, detected)} of detection frames) decided a HUD-less difference '
              'from an inexact pair; expected until S3' if inexact else
              'no HUD-less difference decided from an inexact pair'))

    add(Check('INFO', 'UI holds',
              ', '.join(f'{k.replace("_", " ")} {get(f"held.{k}")}' for k in HOLD_KINDS)
              + f' ({percent(held, auto)} of Auto frames); {get("held.cap")} frames wanted a hold past the cap'))
    none = [(reason.replace('_', ' '), get(f'none.{reason}')) for reason in NO_MASK_REASONS]
    skipped = [(reason.replace('_', ' '), get(f'inactive.{reason}')) for reason in INACTIVE_REASONS]

    def shares(parts: list[tuple[str, int]]) -> str:
        return ', '.join(f'{name} {percent(n, auto)}' for name, n in parts if n)
    add(Check('INFO', 'UI no mask',
              f'{percent(sum(n for _, n in none + skipped), auto)} of Auto frames had no mask'
              + (f'; decided none: {shares(none)}' if shares(none) else '')
              + (f'; without detection: {shares(skipped)}' if shares(skipped) else '')))
    add(Check('INFO', 'UI trust events',
              f'earned {get("trust.earned")}, revoked by a full claim over the scene {get("trust.revoked_full")}, '
              f'revoked by presented disagreement {get("trust.revoked_presented")}, lapsed {get("trust.lapsed")}, '
              f'restored {get("trust.restored")}, '
              + (f'legacy entries discarded {get("trust.discarded")}' if 'trust.discarded' in c else
                 f'opaque proof set {get("trust.opaque_set")} and cleared {get("trust.opaque_cleared")}')))


def outside_settle(s: Session, a: float, b: float) -> list[tuple[float, float]]:
    """The parts of a..b outside the settle time after each FG switch, runtime reset and export start."""
    parts = [(a, b)]
    for mark in s.settle:
        parts = [(p, q) for x, y in parts for p, q in ((x, min(y, mark)), (max(x, mark + SETTLE_S), y)) if q > p]
    return parts


@dataclass
class Gap:
    start: float
    end: float
    reasons: Counter = field(default_factory=Counter)  # Seconds each description held.


def unprotected_gaps(s: Session) -> list[Gap]:
    """Streamed time in which Auto left rendered frames without a UI mask, outside settle times.

    A UI protection line's state holds until that runtime's next line (written on a change, at most once per
    LOG_GATE_S, else every UI_LINE_PERIOD_S), so its ends are approximate by up to a second. Gaps less than a
    second apart merge, as windows() merges counter intervals. Lines whose status sample was pending continue a
    run with its reason, as the overlay's run does."""
    runtimes: dict[str, list[UISample]] = {}
    for u in s.ui:
        runtimes.setdefault(u.runtime, []).append(u)
    pieces = []
    for samples in runtimes.values():
        reason = ''
        for u, following in zip(samples, samples[1:] + [None]):
            if u.unprotected():
                reason = u.why_unprotected()
            elif not (reason and u.pending()):
                reason = ''
                continue
            end = min(following.t if following else s.last, u.t + UI_LINE_PERIOD_S + LOG_GATE_S)
            for a, b in s.streamed:
                pieces += [(p, q, reason) for p, q in outside_settle(s, max(a, u.t), min(b, end))]
    gaps: list[Gap] = []
    for a, b, reason in sorted(pieces):
        if not gaps or a - gaps[-1].end >= 1.0:
            gaps.append(Gap(a, b))
        gaps[-1].end = max(gaps[-1].end, b)
        gaps[-1].reasons[reason] += b - a
    return gaps


def gap_checks(s: Session, add) -> None:
    """Warn where Auto left streamed frames without a UI mask: HUD and menus then take the scene's depth."""
    if not any(u.mode == 'auto' and u.rendered for u in s.ui):
        add(Check('INFO', 'UI protection gaps', 'no rendered Auto samples'))
        return
    gaps = unprotected_gaps(s)
    listed = [g for g in gaps if g.end - g.start >= UNPROTECTED_MIN_S]
    brief = [g for g in gaps if g.end - g.start < UNPROTECTED_MIN_S]
    note = (f'{len(brief)} {"gap" if len(brief) == 1 else "gaps"} shorter than {UNPROTECTED_MIN_S:g} s '
            f'({sum(g.end - g.start for g in brief):.1f} s)') if brief else ''
    if not listed:
        add(Check('PASS', 'UI protection gaps', 'every streamed Auto frame had a UI mask or a UI channel showing no UI'
                  + (f' except {note}' if note else '')))
        return
    total = sum(g.end - g.start for g in listed)
    add(Check('WARN', 'UI protection gaps',
              f'no usable UI mask for {total:.0f} s in {len(listed)} {"window" if len(listed) == 1 else "windows"}; '
              "HUD and menus took the scene's depth there" + (f'; {note} not listed' if note else ''),
              [f'{span(g.start, g.end)} ({g.end - g.start:.0f} s) {g.reasons.most_common(1)[0][0]}'
               for g in listed][:6]))


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
