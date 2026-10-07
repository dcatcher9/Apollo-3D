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
# Since the export-slot counters (dropped, overwritten_unconsumed): what the consumer actually took of the published
# frames. published - overwritten_unconsumed is the frames the host claimed; published + dropped is what the game
# offered (Presents that tried to export).
DELIVERY = re.compile(r'Sunshine SBS output: published=(\d+) .*?runtime=(0x[0-9a-fA-F]+) generation=(\d+) '
                      r'dropped=(\d+) overwritten_unconsumed=(\d+)')
# The host's stream frame rate (display_base.cpp): '[90/1 exactly 90fps]' or '[60fps]'.
HOST_FPS = re.compile(r'Requested frame rate \[(?:(\d+)/(\d+) exactly [0-9.]+fps|(\d+)fps)\]')
# A live video-mode change applied mid-stream (the headset may move its panel between 90 and 72 Hz): the host logs
# the new capture pacing or the live virtual-display resize.
HOST_LIVE_FPS = re.compile(r'Capture pacing updated to (\d+)fps for a live video-mode change|'
                           r'Virtual display resized live to \d+x\d+ @ (\d+) Hz')
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
    # Since S2a the one-way judgment counts of the layer, Backbuffer and current alpha (A2; after be7788bf only
    # Backbuffer and current alpha: the layer's column was 0 on every line, as every layer was the never-judged
    # one-frame-late copy), the sample's own-decision reason, the candidate it refused and whether the T1 grace reused
    # the previous real frame's decision (F1). Lines from S2a through 180f1842 also carry sampled_late_layer, which
    # only repeated that a layer was offered and is skipped.
    r'(?:sampled_one_way=\{strong=(?P<strong>\d+/\d+(?:/\d+)?) contradicted=(?P<contradicted>\d+/\d+(?:/\d+)?)\} '
    r'sampled_reason=(?P<reason>\w+) sampled_refused=(?P<refused>\w+) sampled_reused=(?P<reused>\d) '
    r'(?:sampled_late_layer=\d )?)?'
    r'sampled_hudless=\{changed=(?P<changed>\d+) unchanged=(?P<unchanged>\d+) invalid=(?P<hudless_invalid>\d+)'
    r'(?: matching_tiles=(?P<tiles>\d+) lit=(?P<lit>\d+))?')
# Hidden-scene fields of the same line, absent from older logs. Before S2b the HUD-less image's D and the held routes
# (scene_hold); since S2b (SCENE_S2B) the Backbuffer and current alpha's opaque pixels, the informative full claims, the
# H1 word, the pre-UI scene image's D and the scene guard's holds (docs/reshade-sbs.md, hidden-scene evidence); since
# fix 1 (selection revision 4) also the offscreen UI layer against the presented frame (decision texel 11): matching and
# lit layer pixels, which prove the layer's signature the pre-UI scene image. Logs of fix 1 to revision 8 also carry
# lit presented pixels and those that differ from the layer, and the first-run shadow's fields (shadow,
# shadow_hidden_ms); both were shadow statistics, removed in selection revision 9 and logged as 0 from it through
# be7788bf, so they are optional here.
SCENE = re.compile(
    r'sampled_alpha_opaque=(?P<opaque>\d+/\d+) sampled_scene=\{n=(?P<n>\d+) d=(?P<d>-?[0-9.]+) valid=(?P<valid>\d) '
    r'ran=(?P<ran>\d) verdict=(?P<verdict>\w+)\} sampled_hudless_scene=\{n=\d+ d=(?P<hudless_d>-?[0-9.]+) '
    r'valid=(?P<hudless_valid>\d)\} scene_hold=(?P<hold>\d+)'
    r'(?: shadow=(?P<shadow>\d) shadow_hidden_ms=(?P<hidden_ms>\d+))?')
SCENE_S2B = re.compile(
    r'sampled_alpha_opaque=(?P<opaque>\d+/\d+) sampled_inferred_opaque=(?P<inferred>\d+/\d+) '
    r'sampled_claims=(?P<claims>0x[0-9a-fA-F]+) sampled_h1=\{applied=(?P<applied>\d) winner=(?P<winner>\d+)\} '
    r'sampled_scene=\{n=(?P<n>\d+) d=(?P<d>-?[0-9.]+) valid=(?P<valid>\d) ran=(?P<ran>\d) verdict=(?P<verdict>\w+)\} '
    r'sampled_pre_ui_scene=\{image=(?P<image>\w+) n=\d+ d=(?P<pre_ui_d>-?[0-9.]+) valid=(?P<pre_ui_valid>\d)\} '
    r'scene_guard=\{hidden=(?P<hidden>\d) pre_ui=(?P<pre_ui>\d) refuted=(?P<refuted>\d+)(?: proven=(?P<proven>\d))?\}'
    r'(?: shadow=(?P<shadow>\d) shadow_hidden_ms=(?P<hidden_ms>\d+))?'
    r'(?: sampled_pre_ui_pixels=\{match=(?P<match>\d+) image_lit=(?P<image_lit>\d+)'
    r'(?: presented_lit=(?P<presented_lit>\d+) presented_lit_differs=(?P<differs>\d+))?\})?')
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
# Since fix 1 the same ledger also holds, per offscreen layer signature, the proof that the layer is the pre-UI scene
# image (H1 d): the key 'pre_ui:<format>:<space>', listed with the accepted sources but never a UI coverage source.
PRE_UI_KIND = 'pre_ui'
LEGACY_TRUST = re.compile(r'Sunshine UI protection: discarded (\d+) legacy UI trust entries')
# Since S2a: the panel's Forget action (A3). Lines of removed features (the first-run shadow, rule H2's still screens,
# the S3 identity shadow and its interposer lines) are ignored.
FORGET = re.compile(r'Sunshine UI protection: forgot learned UI sources (\S+) for this game')
# The session's cumulative exact UI counters (docs/reshade-sbs.md, UI counters), absent from older logs.
COUNTERS = re.compile(r'Sunshine UI counters: (.*)$')
COUNTER_FIELD = re.compile(r'(\w+)=(?:\{([^}]*)\}|(\S+))')
# The UI capture gate's HUD-less pairings per log interval (game3d_ui_input_provider.cpp): batch is a same-batch
# Backbuffer pair, the only exact one; real and late are Present-counted proposals.
GATE = re.compile(r'Sunshine UI capture gate: .*?hudless_presents=\{batch=(\d+) real=(\d+) late=(\d+) generated=(\d+)'
                  r'(?: reoffered=(\d+))?')
# Since selection revision 10 the exact one-way judge (A2) also reads the declared alphas, UIAlpha and the UI colour
# tag, logged as their own group before status_revision.
DECLARED_ONE_WAY = re.compile(r'sampled_declared_one_way=\{strong=(\d+)/(\d+) contradicted=(\d+)/(\d+)\}')
LOSS = re.compile(r'sampled_only=\{revision=(\d+) found=1 cause=(\w+)')
READINESS = re.compile(r'Sunshine depth readiness: (lost|recovered) reason=(\w+)')
UNAVAILABLE_MS = re.compile(r'\bunavailable_ms=(\d+)')
PROVIDER = re.compile(r'\bprovider=(\w+)')
SELECTION = re.compile(r'\bselection=(\w+)')
RUNTIME = re.compile(r'\bruntime=(0x[0-9a-fA-F]+)')
# Either placement controller; the first word after the colon is its state.
PLACEMENT = re.compile(r'Sunshine 3D (raw automation|Streamline scale): (\w+);')
# Every API depth provider's status line (streamline_depth_provider.cpp): Streamline, NGX and the API fallback.
DEPTH_STATUS = re.compile(r'Sunshine (Streamline|NGX|API) depth: (\w+);')
HITCH = re.compile(r'Game 3D hitch: (.+?) took ([0-9.]+) ms')
# The present-thread steps of a depth-source flip (Streamline <-> NGX <-> Generic) that each log their own hitch
# line (docs/reshade-sbs.md, depth handoff): named in the hitch details.
DEPTH_FLIP_STEPS = frozenset((
    'native depth frame', 'depth display retirement', 'depth present observation', 'depth list coverage report',
    'frame generation query', 'depth readiness trace', 'API depth binding', 'generic capture switch',
    'generic challenger release', 'generic depth rebinding', 'depth capture acquisition', 'depth display preparation'))
# The add-on's Diagnostics switch (game3d_diagnostics.h): off by default, which records no per-pass GPU timing.
DIAGNOSTICS = re.compile(r'Sunshine Game 3D: diagnostics (on|off) \(Diagnostics=(\d)\)')
GPU_PROFILE = re.compile(r'\bgpu_profile=(\w+)')
# fg_owned (since the final add-on review): evaluations that copied nothing while Streamline FG owned depth.
NGX = re.compile(r'Sunshine NGX depth: .*?evaluations=(\d+) nominations=(\d+) copy_recorded=(\d+) '
                 r'metadata_only=(\d+)(?: fg_owned=(\d+))?')
TIMING = re.compile(r'Sunshine Game 3D timing: presents=(\d+) cpu_ms=\{mean=([0-9.]+) max=([0-9.]+)\}'
                    r'.*?gpu_frames=(\d+)'
                    r' gpu_ms mean/max=\{total=([0-9.]+)/([0-9.]+)')
# The timing line's offscreen UI layer group (since 10-06): live copies recorded, skipped by a full ring and offered.
# Since 10-07 the group ends with the longest time the layer went unrefreshed across skipped copies (skip_gap_ms,
# ui_layer::skip_gap), and the line with its window's length; a runtime reset logs its last, shorter window. Older lines
# cover the periodic 10 s (exporter.cpp) and have no gap.
LAYER_TIMING = re.compile(r'ui_layer=\{copies=(\d+) skipped=(\d+) offers=(\d+) '
                          r'presents_since_copy=\{mean=([0-9.]+) max=(\d+)\}(?: skip_gap_ms=([0-9.]+))?\}'
                          r'(?: window_ms=(\d+))?')
TIMING_WINDOW_S = 10.0
# The layer's recency gate (game3d_ui_layer.h max_clear_gap_ms): this long without a recorded copy, the layer is no
# longer offered at all.
LAYER_RECENCY_MS = 250.0
# Logged once per ring when a copy is first skipped: a WARN before 10-07, INFO since. UI layer copies judges it.
LAYER_SATURATED = 'Sunshine UI layer: every live copy is offered, held or still being read'
LOADED = re.compile(r"loaded from '.*' into '(.*)'")
# ReShade tears its runtimes down when the game closes; some games (Unreal) end the process before ReShade logs its own
# exit, so a teardown at the end of the log is a normal exit.
TEARDOWN = 'Destroyed runtime environment on runtime'
TEARDOWN_TAIL_S = 10.0
# Windows display scaling holds a DPI-unaware game below its display's resolution (game3d_display_scale.h).
DISPLAY_SCALING = re.compile(r'display scaling limits the game: it renders at (\d+)x(\d+) on a (\d+)x(\d+) display '
                             r'because Windows display scaling is (\d+)%')
HOST_LINE = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.(\d{3})\]: (\w+): (.*)$')
# An NVENC stall logs its warning and then its completion or timeout line for the same frame, each with the
# producer's state (nvenc_base.cpp): one stall per frame, classified by its last line.
NVENC_FRAME = re.compile(r'NvEnc: frame (\d+) ')
NVENC_SAME_STALL_S = 5

# ReShade and game noise that says nothing about Game 3D.
BENIGN = (
    'Successfully compiled', 'IDirectInput8W::CreateDevice failed', 'is inconsistent',
    'Add-ons are still loaded', 'Game 3D hitch', 'display scaling limits the game', LAYER_SATURATED,
)
# Export pauses that are part of normal play rather than faults (consumer_transfer_changed: a host attached or
# left between a Present's render and its export, which the next Present renders for).
ROUTINE_INACTIVE = {'not_foreground', 'runtime_reset', 'no_consumer', 'present_without_render', 'runtime_gone',
                    'consumer_transfer_changed'}
SETTLE_S = 3.0  # Recalibration and holds after an FG switch or runtime reset.
CALIBRATION_MIN_S = 0.1  # A run without placement that had less depth than this calibrated nothing.
LOG_GATE_S = 1.0  # A changed controller state waits this long after its previous line (diagnostic_log_gate.h).
RAW_PLACED = ('ready', 'holding_reference')  # Raw automation states that place the scene (raw_scene_policy.h).
# A depth selection that names a game-side cause.
SELECTION_HINTS = {'inactive_views': 'the game supplied no depth'}
RESOLVE_S = 10.0  # A dispute is handled when that source's acceptance is revoked this soon.
# A2 revokes an accepted source by this many contradicting samples within this span (alpha_trust_samples and
# alpha_trust_span_ms in game3d_alpha_auto.h); shorter contradictions are never revoked, by design.
A2_SAMPLES, A2_SPAN_S = 3, 2.0
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
HUDLESS = 0x10  # The HUD-less candidate bit; 0x20 marks the pair exact.
# The inferred alpha kinds the one-way test judges (A2), in the order of sampled_one_way (ui_selection::judged_kinds):
# the UI layer, Backbuffer and current alpha, by their index in ALPHA_NAMES.
JUDGED = (4, 2, 3)
# Candidate kind names of sampled_refused (ui_selection::kind_names) and how the report names them.
KIND_NAMES = dict(zip(ALPHA_KINDS, ALPHA_NAMES)) | {'hudless': 'HUD-less difference'}
# Unprotected time shorter than this is counted, not listed: alpha_trust_span_ms (game3d_alpha_auto.h), the span over
# which detection itself earns or loses confidence.
UNPROTECTED_MIN_S = 2.0
# Stream delivery: the stream follows the game, so the host should take min(what the game offers, the stream fps) new
# frames per second. A window below this share of that target for this long warns. Without a host log the stream rate
# is unknown (a 72 Hz or 60 Hz stream that takes every frame it can reads below any assumed rate), so the rates are
# reported as INFO, not judged.
DELIVERY_SHARE, DELIVERY_WARN_S = 0.85, 10.0
# Presents held without a decision to show (held.none, T1) in a counter interval: a share of Auto frames from which
# the window is listed, and a share and length at which it fails (UI detection effectively never ran).
HELD_NONE_WARN, HELD_NONE_FAIL, HELD_NONE_FAIL_S = 0.5, 0.9, 10.0
UI_LINE_PERIOD_S = 10.0  # An unchanged UI protection state is logged again this often while presenting (exporter.cpp).
# Hidden-scene evidence (docs/reshade-sbs.md): a hidden run without a decided source this long is an uncovered hidden
# scene.
SHADOW_HIDDEN_WARN_MS = 500
# Since fix 1 an offscreen layer without coverage is proven the pre-UI scene image by A2_SAMPLES samples over A2_SPAN_S
# (alpha_trust_samples and alpha_trust_span_ms) in which it equals the presented frame on at least 90% of pixels
# (ui_selection::full) and is lit on at least half (ui_selection::pre_ui_match); a restored proof lapses after
# alpha_trust_reconfirm_ms of testable time without one (game3d_alpha_auto.h).
PRE_UI_PROOF_RULE = ('3 samples over 2 s whose layer equals the presented frame on at least 90% of pixels and is lit '
                     'on at least half')
PRE_UI_RECONFIRM_S = 60
# Decided sources by number (decision texel 0, docs/reshade-sbs.md); 7 is retired. Before S2b 8 and 9 were the
# hidden-scene layer and HUD-less routes; since S2b 8 is H1 and 9 is retired. 11 (rule H2's still screen, fix 2) and
# 12 (fix 3's pre-UI change set) were removed and are reserved; logs of those builds may still carry their counter
# words.
SOURCE_NAMES = {0: 'no mask', 1: 'UI alpha', 2: 'UI colour', 3: 'Backbuffer alpha', 4: 'current alpha',
                5: 'HUD-less difference', 6: 'full frame (exact pair)', 8: 'full frame (layer route)',
                9: 'HUD-less route (before S2b)', 10: 'UI layer'}
S2B_SOURCE_NAMES = SOURCE_NAMES | {8: 'full frame over a hidden scene (H1)',
                                   11: 'still screen without a UI source (H2, removed)'}
CLAIM_PRE_UI = 0x80  # The pre-UI scene image's informative claim (ui_detection::claim_pre_ui), since S2b.
NO_MASK_REASONS = ('layer_aside', 'trusted_invalid', 'presented_blocked', 'ambiguous', 'difference_failed',
                   'gate_no_hold', 'no_candidate', 'other', 'unaccepted')
# Held frames by kind: since S2a a generated Present showing a real frame's decision, or one without such a decision
# (T1); before it three hold kinds and a cap of three Presents.
HOLD_KINDS = ('generated', 'none')
S1_HOLD_KINDS = ('generated', 'inexact_after_exact', 'trusted_missing')
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
    hudless_d: float  # The second image's D: before S2b the HUD-less image's, since S2b the pre-UI scene image's.
    hudless_valid: bool
    hold: int  # Before S2b the routes held by the render that logged: 1 layer (source 8), 2 HUD-less (source 9).
    # Logs before selection revision 9 only (the removed first-run shadow): whether it measured with the gates closed,
    # and the longest hidden run without a decided source since the previous line; False and 0 otherwise.
    shadow: bool
    hidden_ms: int
    # Since S2b (absent before): the scene guard's holds this render (hidden, pre-UI) and its refuted signatures, the
    # pre-UI scene image (none, hudless or layer), the informative full claims (candidate bits, 0x80 the pre-UI
    # image), whether H1 applied over the S1 winner and that winner's source, and the Backbuffer and current alpha's
    # opaque pixels.
    guard: tuple[int, int, int] | None = None
    pre_ui_image: str = ''
    claims: int = 0
    h1: bool = False
    winner: int = 0
    inferred_opaque: tuple[int, ...] = ()
    # Whether the offered layer was proven the presented frame without its UI, so that its pre-UI image may act
    # (H1 d): in S2b by the scene guard's D similarity, since fix 1 the ledger's key pre_ui:<format>:<space>; None on
    # lines without the field.
    proven: bool | None = None
    # Since fix 1 (absent before): the layer against the presented frame (decision texel 11): matching pixels and lit
    # layer pixels.
    pre_ui_pixels: tuple[int, int] | None = None

    @property
    def s2b(self) -> bool:
        return self.guard is not None

    @property
    def fix1(self) -> bool:
        """Logged since fix 1, whose layer proof is by pixels and a ledger key."""
        return self.pre_ui_pixels is not None

    def pre_ui_claim(self) -> bool:
        """The pre-UI image's claim could act: claimed this sample while the scene guard held its pre-UI hold."""
        return bool(self.claims & CLAIM_PRE_UI) and bool(self.guard and self.guard[1])


class UISample(NamedTuple):
    t: float
    detection: str
    # 1-4: UI alpha, UI colour tag, Backbuffer or current alpha decided, 10: the UI layer, 5: HUD-less difference, 6:
    # full frame flat (exact pair), 8: full frame over a hidden scene by the layer route, 9: by the HUD-less route, 11
    # (fix 2 to revision 8): a still screen without a UI source shown flat (H2, removed). 7 is retired. Before S1 the
    # layer decided as 2; parse() reads it as 10.
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
    # Since S2a (absent before): strong and one-way contradicted pixels in JUDGED order (A2), the own decision's
    # ui_no_mask reason or 'decided', the refused candidate's kind or 'none', and whether the T1 grace reused the
    # previous real frame's decision (then source and covered are that decision's).
    strong: tuple[int, ...] | None = None
    contradicted: tuple[int, ...] | None = None
    reason: str = ''
    refused: str = ''
    reused: bool = False
    tiles: int = 0  # HUD-less matching tiles and lit pixels (V2).
    lit: int = 0
    # Since selection revision 10: the declared alphas' (UIAlpha, UI colour tag) strong and one-way contradicted
    # pixels, ((strong...), (contradicted...)); None before.
    declared_one_way: tuple[tuple[int, int], tuple[int, int]] | None = None

    @property
    def s2a(self) -> bool:
        """Logged since S2a, whose ledger judges by provenance (A2)."""
        return self.strong is not None

    def change_set_valid(self) -> bool:
        """V2 from the logged HUD-less counts, as ui_selection::change_set_selective, change_set_full and (since
        selection revision 8) change_set_empty: a lit pair without any changed pixel in clean tiles."""
        changed, unchanged, nonfinite = self.hudless
        if nonfinite or not self.candidates & HUDLESS:
            return False
        selective = changed and changed * 4 < self.pixels and unchanged * 100 >= self.pixels * 75 and self.tiles >= 128
        full = (self.candidates & HUDLESS_PAIR == HUDLESS_PAIR and changed * 100 >= self.pixels * 98
                and self.lit * 2 >= self.pixels)
        empty = not changed and self.pixels and self.lit * 2 >= self.pixels and self.tiles >= 128
        return bool(selective or full or empty)

    def exact_judge(self) -> bool:
        """A2 judge (a): an offered exact pair whose change set is valid this sample (ui_selection::exact_judge)."""
        return self.candidates & HUDLESS_PAIR == HUDLESS_PAIR and self.change_set_valid()

    def one_way(self, c: int) -> bool:
        """A2: at least a tenth of judged alpha c's strong pixels lie where the exact pair's HUD-less image is lit and
        unchanged (ui_selection::one_way_contradicted); only with a valid exact pair."""
        if not self.s2a or not self.exact_judge():
            return False
        strong, contradicted = self.one_way_counts(c)
        return bool(strong) and contradicted * 10 >= strong

    def one_way_counts(self, c: int) -> tuple[int, int]:
        """Strong and one-way contradicted pixels of alpha c: the judged inferred kinds since S2a, the declared ones
        since selection revision 10; (0, 0) where the line has none."""
        if c in JUDGED and self.strong is not None:
            return self.strong[JUDGED.index(c)], self.contradicted[JUDGED.index(c)]
        if c in DECLARED and self.declared_one_way is not None:
            return self.declared_one_way[0][c], self.declared_one_way[1][c]
        return 0, 0

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

    def fg_label(self) -> str:
        return 'FG on' if self.fg else 'FG off'

    def why_unprotected(self) -> str:
        """Each offered candidate, in draw order, and why it gave no mask, after the line's own FG state."""
        return f'{self.fg_label()}: {self.why_body()}'

    def why_body(self) -> str:
        if self.availability == 'source_unavailable' or not self.candidates:
            return 'no UI source offered'
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
        return '; '.join(parts) + self.named_reason()

    def bare_layer(self) -> bool:
        """The offscreen UI layer offered without coverage: the layer a pre-UI proof can be about (H1 d)."""
        return self.offered(4) and not self.alpha[4]

    def hidden(self) -> bool:
        """The presented frame's valid evidence read hidden."""
        return bool(self.scene and self.scene.valid and self.scene.verdict == 'hidden')

    def named_reason(self) -> str:
        """The add-on's own reason for a sample without a mask and the candidate it refused (F1), since S2a."""
        if not self.s2a or self.reason in ('', 'decided'):
            return ''
        refused = KIND_NAMES.get(self.refused)
        return f' (reason {self.reason.replace("_", " ")}' + (f': {refused})' if refused else ')')


def ui_sample(t: float, text: str, g: dict[str, str | None], scene: Scene | None) -> UISample:
    """One 'Sunshine UI protection' line; a line logged before S1 is normalized to the S1 candidates."""
    def counts(value: str) -> tuple[int, ...]:
        return tuple(int(v) for v in value.split('/'))

    def judged(value: str) -> tuple[int, ...]:
        # JUDGED order; a line after be7788bf omits the layer's column, which no build has counted since revision 10.
        found = counts(value)
        return (0,) * (len(JUDGED) - len(found)) + found

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
    s2a = g['strong'] is not None
    return UISample(t, g['detection'], source, int(g['covered']), int(g['pixels']), candidates, alpha, accepted,
                    (int(g['changed']), int(g['unchanged']), int(g['hudless_invalid'])), invalid, scene,
                    field_of(UI_RUNTIME, ''), field_of(UI_MODE, 'auto'), field_of(UI_RENDERED, '1') == '1',
                    field_of(UI_AVAILABILITY, ''), field_of(UI_FG, '0') == '1', legacy,
                    judged(g['strong']) if s2a else None, judged(g['contradicted']) if s2a else None,
                    g['reason'] or '', g['refused'] or '', g['reused'] == '1', int(g['tiles'] or 0),
                    int(g['lit'] or 0))


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
    # A Forget is listed with None too: the acceptance line logged just before it records the change.
    trust_events: list[tuple[float, str, str, tuple[str, ...] | None]] = field(default_factory=list)
    # The last 'Sunshine UI counters' line: the session's totals are cumulative over all its runtimes. Every line is
    # kept with its time for the per-interval checks.
    counters: dict[str, int] | None = None
    counter_history: list[tuple[float, dict[str, int]]] = field(default_factory=list)
    # The UI capture gate's HUD-less pairings summed over the session: same-batch (exact) and Present-counted.
    gate_batch: int = 0
    gate_counted: int = 0
    losses: dict[int, tuple[float, str]] = field(default_factory=dict)
    readiness: Counter = field(default_factory=Counter)  # Losses while the export streamed.
    depth_episodes: list[DepthEpisode] = field(default_factory=list)
    streamed: list[list] = field(default_factory=list)  # From a generation to the next export inactive.
    placement: list[tuple[float, bool, str, str]] = field(default_factory=list)  # t, placed, state, provider.
    statuses: list[tuple[float, str, str]] = field(default_factory=list)  # Each depth status: time, status, provider.
    hitches: list[tuple[float, str, float]] = field(default_factory=list)
    timings: list[tuple] = field(default_factory=list)  # Every timing line's window.
    # Each timing line's UI layer group: (end, length s, Presents, copies, skipped, offers, presents_since_copy mean
    # and max, longest skip gap in ms or None before 10-07); and each logged-once full-ring line.
    layer_windows: list[tuple] = field(default_factory=list)
    layer_saturated: list[float] = field(default_factory=list)
    # Export delivery counters: (t, runtime, generation, published, dropped, overwritten_unconsumed); and the
    # host's stream fps when a host log was read.
    delivery: list[tuple[float, str, int, int, int, int]] = field(default_factory=list)
    stream_fps: float | None = None
    # The stream fps over the session, (t, fps) on this log's clock, when a host log was read.
    stream_fps_timeline: list[tuple[float, float]] = field(default_factory=list)
    timing_profile: str = ''  # The last timing line's gpu_profile state ('disabled' with Diagnostics off).
    teardown: float | None = None  # The last runtime teardown line.
    diagnostics: list[tuple[float, bool]] = field(default_factory=list)  # Each Diagnostics switch line.
    ngx: tuple[int, ...] | None = None
    display_scaling: list[tuple[float, tuple[int, ...]]] = field(default_factory=list)
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
        if found := DISPLAY_SCALING.search(text):
            s.display_scaling.append((t, tuple(int(v) for v in found.groups())))
        if (found := LOADED.search(text)) and not s.exe:
            s.exe = os.path.basename(found.group(1))
        if 'Finished exiting' in text:
            s.exited = True
        if TEARDOWN in text:
            s.teardown = t
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
        if found := DELIVERY.search(text):
            published, runtime, generation, dropped, overwritten = found.groups()
            s.delivery.append((t, runtime.lower(), int(generation), int(published), int(dropped), int(overwritten)))
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
                              e['hudless_valid'] == '1', int(e['hold']), e['shadow'] == '1', int(e['hidden_ms'] or 0))
            elif evidence := SCENE_S2B.search(text):
                e = evidence.groupdict()
                scene = Scene(tuple(int(v) for v in e['opaque'].split('/')), int(e['n']), float(e['d']),
                              e['valid'] == '1', e['ran'] == '1', e['verdict'], float(e['pre_ui_d']),
                              e['pre_ui_valid'] == '1', 0, e['shadow'] == '1', int(e['hidden_ms'] or 0),
                              (int(e['hidden']), int(e['pre_ui']), int(e['refuted'])), e['image'],
                              int(e['claims'], 16), e['applied'] == '1', int(e['winner']),
                              tuple(int(v) for v in e['inferred'].split('/')),
                              None if e['proven'] is None else e['proven'] == '1',
                              None if e['match'] is None else (int(e['match']), int(e['image_lit'])))
            sample = ui_sample(t, text, found.groupdict(), scene)
            if declared := DECLARED_ONE_WAY.search(text):
                strong0, strong1, against0, against1 = (int(v) for v in declared.groups())
                sample = sample._replace(declared_one_way=((strong0, strong1), (against0, against1)))
            s.ui.append(sample)
        if found := GATE.search(text):
            batch, real, late, _, reoffered = (int(v or 0) for v in found.groups())
            s.gate_batch += batch
            s.gate_counted += real + late + reoffered
        if found := TRUST.search(text):
            s.trust_events.append((t, found.group(1), found.group(2), accepted_keys(found.group(2))))
        if found := LEGACY_TRUST.search(text):
            s.trust_events.append((t, 'discarded legacy UI trust entries', found.group(1), None))
        if found := FORGET.search(text):
            s.trust_events.append((t, 'forgot learned UI sources', found.group(1), None))
        if (found := COUNTERS.search(text)) and 'auto_frames' in (values := counter_fields(found.group(1))):
            s.counters = values
            s.counter_history.append((t, values))
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
            s.statuses.append((t, found.group(2), found.group(1)))
        if found := NGX.search(text):
            s.ngx = tuple(int(v or 0) for v in found.groups())
        if found := HITCH.search(text):
            s.hitches.append((t, found.group(1), float(found.group(2))))
        if found := TIMING.search(text):
            s.timings.append((int(found.group(1)), float(found.group(2)), float(found.group(3)),
                              int(found.group(4)), float(found.group(5)), float(found.group(6))))
            profile = GPU_PROFILE.search(text)
            s.timing_profile = profile.group(1) if profile else ''
            if layer := LAYER_TIMING.search(text):
                copies, skipped, offers, mean, peak, gap, window_ms = layer.groups()
                s.layer_windows.append((t, int(window_ms) / 1000.0 if window_ms else TIMING_WINDOW_S,
                                        int(found.group(1)), int(copies), int(skipped), int(offers), float(mean),
                                        int(peak), float(gap) if gap is not None else None))
        if LAYER_SATURATED in text:
            s.layer_saturated.append(t)
        if found := DIAGNOSTICS.search(text):
            s.diagnostics.append((t, found.group(1) == 'on'))
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
    if s.display_scaling:
        gw, gh, dw, dh, pct = s.display_scaling[-1][1]
        add(Check('WARN', 'Display scaling', f'the game rendered at {gw}x{gh} on a {dw}x{dh} display because Windows '
                  f"display scaling is {pct}%; set scaling to 100% or the game's high-DPI override to Application",
                  [clock(t) for t, _ in s.display_scaling][:6]))

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
        # Each calibration run is listed once; a run with almost no depth (a depth loss inside the window) is none.
        seen = [r for r in runs if r.depth_s() >= CALIBRATION_MIN_S
                and any(r.start < b and r.end > a for a, b in calibrations)]
        if seen:
            add(Check('INFO', 'Placement calibration',
                      ('1 flat run was a calibration' if len(seen) == 1 else f'{len(seen)} flat runs were calibrations')
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

    # A capture status during the settle time after an export start, FG switch or reset is the source settling.
    failing = ('conflicting_state', 'incomplete_state', 'missing_state', 'failed', 'unsupported_state',
               'unsupported_resource', 'unsupported_lifetime')

    def settles(t: float) -> bool:
        return any(0 <= t - m <= SETTLE_S for m in s.settle)
    outside = [(t, k, p) for t, k, p in s.statuses if k in failing and not settles(t)]
    settling = Counter(k for t, k, _ in s.statuses if k in failing and settles(t))
    conflicts = Counter(k for _, k, _ in outside)
    if conflicts:
        named = ', '.join(f'{k} {v}' for k, v in sorted(conflicts.items()))
        if 'incomplete_state' in conflicts:
            named += ' (incomplete_state: the source was blocked by a split barrier or a layout without a legacy state)'
        add(Check('WARN', 'Capture status', named + (f'; {sum(settling.values())} more while settling'
                                                     if settling else ''),
                  [f'{clock(t)} {p} {k}' for t, k, p in outside][:6]))
    elif settling:
        add(Check('INFO', 'Capture status', ', '.join(f'{k} {v}' for k, v in sorted(settling.items()))
                  + f' within {SETTLE_S:g} s of an export start, FG switch or reset (settling)'))
    # Evaluations while Streamline FG owned depth copy nothing by design.
    if s.ngx and s.ngx[0] > s.ngx[4] and not s.ngx[2] and not s.ngx[3]:
        add(Check('WARN', 'NGX depth', f'{s.ngx[0] - s.ngx[4]} DLSS evaluations recorded no depth copy'))
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
                  [f'{clock(t)} {what} {ms:.1f} ms' + (' (depth-source flip step)' if what in DEPTH_FLIP_STEPS else '')
                   for t, what, ms in worst]))
    else:
        add(Check('PASS', 'Present hitches', f'{len(expected)} at resets, FG switches or overlay opening' if expected
                  else 'none'))

    if s.diagnostics:
        on = s.diagnostics[-1][1]
        add(Check('INFO', 'Diagnostics', f'Diagnostics={int(on)}: ' + (
            'per-pass GPU timing is recorded' if on else 'no per-pass GPU timing (its lines are absent by design)')))
    if s.timings:
        # Each timing line covers its own window: means weighted by that window's Presents (CPU) and GPU frames.
        presents = sum(w[0] for w in s.timings)
        gpu_frames = sum(w[3] for w in s.timings)
        cpu_mean = sum(w[0] * w[1] for w in s.timings) / presents if presents else 0.0
        gpu_mean = sum(w[3] * w[4] for w in s.timings) / gpu_frames if gpu_frames else 0.0
        cpu_max, gpu_max = max(w[2] for w in s.timings), max(w[5] for w in s.timings)
        disabled = s.timing_profile == 'disabled'
        add(Check('INFO' if gpu_frames or disabled else 'WARN', 'Game 3D cost',
                  f'CPU {cpu_mean:.2f} ms mean ({cpu_max:.1f} max) over {presents} Presents; '
                  + (f'GPU {gpu_mean:.2f} ms mean ({gpu_max:.1f} max)' if gpu_frames else
                     'GPU timing off (Diagnostics=0)' if disabled else 'no GPU timing samples')))
    delivery_checks(s, add)
    layer_copy_checks(s, add)
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
    for t, change, key in pre_ui_proofs(s):
        add(Check('INFO', 'Pre-UI proof', f'{clock(t)} {key} ' + {
            'restored': f'restored from an earlier session, provisional: {PRE_UI_PROOF_RULE} confirm it, and it lapses '
                        f'after {PRE_UI_RECONFIRM_S} s of testable time (the layer offered without coverage while the '
                        'presented frame reads visible) without them',
            'earned': f'earned by {PRE_UI_PROOF_RULE}',
            'lapsed': f'lapsed: restored and not confirmed within {PRE_UI_RECONFIRM_S} s of testable time',
            'forgotten': 'forgotten by Forget'}[change]))
    if not s.ui and s.counters is None:
        add(Check('INFO', 'UI protection', 'no UI protection samples'))
        return
    # Inferred alpha (Backbuffer, current alpha and the UI layer) must never decide while an accepted declared UI
    # channel is offered (S1): that flattens scene as UI. Nor may an accepted alpha cover the whole frame while an
    # exact HUD-less pair shows the scene (Stellar Blade's opaque tagged UI color, before 2026-10); since S2a, nor may
    # an accepted inferred alpha stay accepted once contradictions met A2's revocation condition.
    overrides, flattened, contradicted, disputes, handled, short = [], [], [], [], [], []
    s2a = any(u.s2a for u in s.ui)

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

    def judge(sample: UISample, text: str, c: int, unresolved: list, a2: str = '') -> None:
        """A revocation within RESOLVE_S handles the sample; since S2a it names the A2 contradiction kind."""
        when = revoked(sample.t, ALPHA_KINDS[c])
        (handled if when is not None else unresolved).append(
            text + (f', acceptance revoked {clock(when)}' + (f' ({a2})' if a2 else '') if when is not None else ''))

    # Since S2a each accepted inferred alpha's sampled contradictions, as the ledger judges them (A2), and since
    # selection revision 10 the declared alphas' one-way contradictions too: (time, text, judge, one way).
    a2: dict[int, list[tuple[float, str, str, bool]]] = {c: [] for c in DECLARED + INFERRED}
    for u in s.ui:
        if not u.pixels:
            continue
        if u.blocking(admitted=True) and u.source in tuple(ALPHA_SOURCES[c] for c in u.kept_out()):
            overrides.append(f'{clock(u.t)} source {u.source} covered {100 * u.covered / u.pixels:.0f}%')
        if u.s2a:
            # A2 as the ledger judges it since S2a: an accepted, V1-valid inferred alpha (Backbuffer or current alpha;
            # never the layer, the one-frame-late copy, E2) offered in the sample, contradicted (a) one way
            # by a valid exact pair or (b) by coverage differing by at least 10% of the frame from every accepted,
            # valid UIAlpha or UI color tag offered in the same sample.
            masks = [u.alpha[c] for c in DECLARED if u.offered(c) and u.is_accepted(c) and u.valid(c)]
            judged = INFERRED + (DECLARED if u.declared_one_way is not None else ())
            for c in judged:
                if not (u.offered(c) and u.is_accepted(c) and u.valid(c)) or c == 4:
                    continue
                if u.one_way(c):
                    strong, against = u.one_way_counts(c)
                    state = 'decided' if u.source == ALPHA_SOURCES[c] and not u.reused else 'accepted'
                    a2[c].append((u.t, f'{clock(u.t)} {ALPHA_NAMES[c]} {state} while an exact HUD-less pair showed '
                                  f'{percent(against, strong)} of its strong pixels as lit, unchanged scene',
                                  'A2, one-way by an exact pair', True))
                elif c in INFERRED and masks and min(abs(u.alpha[c] - d) for d in masks) * 10 >= u.pixels:
                    a2[c].append((u.t, f'{clock(u.t)} {ALPHA_NAMES[c]} {100 * u.alpha[c] / u.pixels:.0f}% vs declared '
                                  f'UI {100 * masks[0] / u.pixels:.1f}%', 'A2, declared coverage', False))
            continue
        # As the ledger's disagreement before S2a (A2): only an accepted UI channel with at most 1% invalid pixels
        # and clean accepted presented alpha dispute.
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
    # A2 revokes on its third contradiction within 2 s, so only such a run of sampled contradictions (a subset of the
    # ledger's samples) that no revocation of the kind followed is a defect: FAIL when it holds a one-way
    # contradiction, WARN for declared coverage alone. A contradiction a revocation followed is handled; a shorter one
    # is by design and only noted.
    for c, events in a2.items():
        events.sort(key=lambda e: e[0])
        in_run = [False] * len(events)
        for i in range(A2_SAMPLES - 1, len(events)):
            if events[i][0] - events[i - A2_SAMPLES + 1][0] <= A2_SPAN_S:
                for j in range(i - A2_SAMPLES + 1, i + 1):
                    in_run[j] = True
        i = 0
        while i < len(events):
            j = i + 1
            if in_run[i]:
                while j < len(events) and in_run[j] and events[j][0] - events[j - 1][0] <= A2_SPAN_S:
                    j += 1
            run_events = events[i:j]
            when = revoked(run_events[0][0], ALPHA_KINDS[c])
            if when is not None:
                handled.extend(f'{text}, acceptance revoked {clock(when)} ({a2_kind})'
                               for _, text, a2_kind, _ in run_events)
            elif in_run[i]:
                (contradicted if any(one for *_, one in run_events) else disputes).extend(
                    text + f' ({a2_kind}, {len(run_events)} contradictions within {A2_SPAN_S:.0f} s, not revoked)'
                    for _, text, a2_kind, _ in run_events)
            else:
                short.extend(text for _, text, _, _ in run_events)
            i = j
    # A selective UI channel rejected for invalid pixels (V1: more than 1% of the frame; for the UI layer also colour
    # beyond its alpha headroom) while no mask protected the frame. A channel without alpha carried no UI to reject (a
    # UI layer with colour but no alpha); the time such frames went unprotected is 'UI protection gaps'.
    rejected = []
    for u in s.ui:
        if u.invalid is None or u.source or not u.pixels:
            continue
        for c in DEDICATED:
            if u.offered(c) and not u.valid(c) and 0 < u.alpha[c] and u.alpha[c] * 10 < u.pixels * 9:
                rejected.append(f'{clock(u.t)} {ALPHA_NAMES[c]} {100 * u.alpha[c] / u.pixels:.1f}% covered, '
                                f'{u.invalid[c]} invalid pixels{", accepted" if u.is_accepted(c) else ""}')
    if any(u.invalid is not None for u in s.ui):
        add(Check('WARN' if rejected else 'PASS', 'UI channel admission',
                  f'{len(rejected)} samples left the frame unprotected after rejecting a selective UI channel '
                  'for invalid pixels' if rejected else 'no selective UI channel rejected for invalid pixels',
                  rejected[:6]))
    dispute_text = ('; accepted inferred alpha disagreed with an accepted UI alpha or UI color tag 3 times within 2 s '
                    'without a revocation (A2)' if s2a else
                    '; accepted presented alpha disagrees with an accepted UI channel')
    short_text = (f'; {len(short)} sampled A2 contradictions shorter than the revocation condition (3 within 2 s) '
                  'were not revoked, as designed' if short else '')
    if s.counters is not None:
        counter_checks(s.counters, add, overrides, flattened, contradicted, disputes, handled, dispute_text,
                       refusals(s), short, short_text, s)
        held_none_checks(s, add)
        gap_checks(s, add)
        scene_checks(s, add)
        return
    states = Counter(u.detection for u in s.ui)
    add(Check('FAIL' if overrides or flattened or contradicted else 'WARN' if disputes else 'PASS', 'UI protection',
              ', '.join(f'{k} {v}' for k, v in states.most_common())
              + ('; inferred alpha decided beside an accepted declared UI channel' if overrides else '')
              + ('; an accepted UI source flattened the visible scene' if flattened else '')
              + ('; an exact HUD-less pair contradicted an accepted alpha one way 3 times within 2 s without '
                 'a revocation (A2)' if contradicted else '')
              + (dispute_text if disputes else '') + short_text
              + ('; a contradicted source lost its acceptance'
                 if handled and not disputes and not flattened and not contradicted else ''),
              (overrides + flattened + contradicted or disputes or handled or short)[:6]))
    gap_checks(s, add)
    scene_checks(s, add)


def pre_ui_proofs(s: Session) -> list[tuple[float, str, str]]:
    """Since fix 1: each pre-UI proof key (pre_ui:<format>:<space>) that the logged acceptance changes restored,
    added (earned) or removed (lapsed, or forgotten when the Forget line beside the change names it: Forget logs the
    change first, then the cleared keys)."""
    held: set[str] = set()
    events = []
    for t, kind, _, keys in s.trust_events:
        if keys is None:
            continue
        now = {key for key in keys if key.split(':')[0] == PRE_UI_KIND}
        restored = kind.startswith('restored')
        events += [(t, 'restored' if restored else 'earned', key) for key in sorted(now - held)]
        for key in sorted(held - now):
            forgot = any(event == 'forgot learned UI sources' and abs(when - t) <= LOG_GATE_S
                         and key in value.split(',') for when, event, value, _ in s.trust_events)
            events.append((t, 'forgotten' if forgot else 'lapsed', key))
        held = now
    return events


def refusals(s: Session) -> Counter:
    """Sampled Auto samples without a mask by the add-on's own reason and refused candidate (F1), since S2a."""
    return Counter((u.reason, u.refused) for u in s.ui
                   if u.s2a and u.mode == 'auto' and not u.source and u.reason not in ('', 'decided'))


def counter_checks(c: dict[str, int], add, overrides_sampled: list[str], flattened_sampled: list[str],
                   contradicted_sampled: list[str], disputes: list[str], handled: list[str], dispute_text: str,
                   refused: Counter, short: list[str] = (), short_text: str = '', s: Session | None = None) -> None:
    """Checks from the add-on's exact per-frame UI counters (docs/reshade-sbs.md, UI counters).

    They replace the sampled invariants: every Auto frame is counted, not one per 100 ms sample. Sampled lines still
    give the times of examples, acceptance disputes, which no counter records, whether a contradicted accepted source
    (since S2a one-way by an exact pair, before it a full-frame claim over the scene) was resolved by a revocation,
    the refused candidates, and the time-based gap and scene checks. Stages (S1-S6) and rule IDs (such as H1 and P1)
    named in the output are the UI decision framework's (docs/reshade-sbs.md, UI decision framework)."""
    def get(key: str) -> int:
        return c.get(key, 0)

    # Counter lines since S2a hold held={generated none}, reused and contradicted (T1, A2); before it three hold kinds
    # with a cap and trusted_full. Since S2b they hold the scene guard's group scene={entered released refuted} (H1).
    s2a, s2b = 'held.none' in c, 'scene.entered' in c
    auto, detected = get('auto_frames'), get('detection_frames')
    held = sum(get(f'held.{k}') for k in (HOLD_KINDS if s2a else S1_HOLD_KINDS))
    inactive = sum(get(f'inactive.{k}') for k in INACTIVE_REASONS)
    reconciled = auto == detected + held + inactive
    add(Check('PASS' if reconciled else 'FAIL', 'UI counters',
              f'{auto} Auto frames through the last of {get("samples")} committed samples: {detected} detected, '
              f'{held} held, {inactive} without detection'
              + ('' if reconciled else f"; the add-on's own accounting does not reconcile ({auto} Auto frames, "
                                       f'{detected + held + inactive} detected, held or inactive)')))

    # A contradicted accepted source keeps deciding until repeated contradictions revoke its acceptance, so a handled
    # case counts its frames first. Since S2a the counter is contradicted: an accepted inferred alpha (3, 4 or 10)
    # decided by itself while the same frame's valid exact pair contradicted it one way. A2 revokes on the third
    # contradiction within 2 s and, by design, never on shorter ones, which the counter cannot tell apart; so the
    # counted frames only fail with a sampled run of three within 2 s that no revocation followed. Before S2a
    # trusted_full: an accepted alpha covering the frame over an exact pair that showed the scene, failing when a
    # sample of them was not resolved by a revocation (trust.revoked_full), or when nothing of that kind was revoked.
    # presented_over_dedicated (an inferred alpha deciding beside an accepted declared one, zero by construction since
    # S1) is on counter lines through dc7e3c77 only, so a current line reads it as 0.
    overrides = get('presented_over_dedicated')
    if s2a:
        flattened, revoked, sampled = get('contradicted'), get('trust.revoked_exact'), contradicted_sampled
        # Inferred alpha per frame; since selection revision 10 any accepted alpha, declared included, per sample.
        contradiction = (f'; an exact HUD-less pair contradicted a deciding accepted alpha one way (A2) in '
                         f'{flattened} frames')
        resolution = (f' before {revoked} one-way revocations' if revoked else
                      ', no sampled run reaching the revocation condition (3 within 2 s)')
        unresolved = bool(sampled)
    else:
        flattened, revoked, sampled = get('trusted_full'), get('trust.revoked_full'), flattened_sampled
        contradiction = (f'; an accepted UI source covered the frame while an exact HUD-less pair showed the scene in '
                         f'{flattened} frames')
        resolution = f' before {revoked} revocations of such a source'
        unresolved = flattened and (sampled or not revoked)
    names = S2B_SOURCE_NAMES if s2b else SOURCE_NAMES
    sources = [(source, get(f'decided.{source}')) for source in names]
    add(Check('FAIL' if overrides or unresolved else 'WARN' if disputes else 'PASS', 'UI protection',
              'decided ' + (', '.join(f'{names[k]} ({k}) {percent(n, detected)}' for k, n in sources if n)
                            or 'nothing') + ' of detection frames'
              + (f'; inferred alpha decided beside an accepted declared UI channel in {overrides} frames'
                 if overrides else '')
              + (contradiction + ('' if unresolved else resolution) if flattened else
                 '; an exact HUD-less pair contradicted an accepted alpha one way 3 times within 2 s without '
                 'a revocation (A2)' if sampled else '')
              + (dispute_text if disputes else '') + short_text
              + ('; a contradicted source lost its acceptance' if handled and not disputes and not unresolved else ''),
              (overrides_sampled + sampled or disputes or handled or list(short))[:6]))

    if s2b:
        full_frame_s2b(c, add, detected)
    else:
        full = {k: get(f'decided.{k}') for k in (6, 8, 9)}
        visible, routes = get('full_d.visible'), full[8] + full[9]
        # Before S2b full_d counts the exact full change-set (6) with the held routes (8, 9). An exact full set decides
        # without a hold, and over a visible scene it is intended for an accepted exact pair (P1); it is wrong only
        # before the pair's first selective sample, which was an open question of the framework. So visible samples
        # warn only when a held route decided: the sample that releases a held route decided under the hold before its
        # own evidence read the scene visible, so each release counts one, and these counters cannot tell a release
        # from a held route over a visible scene (H1); only since S2b are releases counted.
        exact_note = ('an exact full change-set (6) over a visible scene is intended for an accepted exact pair (P1); '
                      'whether a pair decides one before its first selective sample is an open question')
        add(Check('WARN' if visible and routes else 'INFO' if visible else 'PASS', 'UI full frame',
                  f'{sum(full.values())} full-frame frames (6: {full[6]}, 8: {full[8]}, 9: {full[9]}); samples that '
                  f'decided one read the scene hidden {get("full_d.hidden")}, ambiguous {get("full_d.ambiguous")}, '
                  f'visible {visible}, invalid or unmeasured {get("full_d.invalid")}; depth not current on '
                  f'{get("full.depth_not_current")} detection frames'
                  + ('' if not visible else
                     f'; {visible} full-frame samples read the scene visible: each release of a held hidden-scene '
                     'route (8, 9) shows one, more than one per hidden scene is a held route over a visible scene '
                     '(H1); logs before S2b do not count releases' + (f'; samples of source 6 are counted with them, '
                                                                      f'and {exact_note}' if full[6] else '')
                     if routes else f'; {visible} samples of source 6 read the scene visible: {exact_note}')))

    # A whole-frame mask from alpha (an accepted source 1-4 or 10 covering at least 99%; only accepted sources decide)
    # pins the frame flat even over a visible scene: an accepted source's pin weight is saturate(8 alpha) at any
    # coverage (P1, the opacity ruling), so it is reported, not warned. The wrong cases have their own checks: an
    # accepted source an exact pair contradicts and no revocation resolved (UI protection; one way since S2a, a full
    # claim over the scene before it), and unaccepted inferred alpha (UI inferred alpha). Dims and tints over dark or
    # changed pixels never meet the one-way test. Since S2b an exact full change-set (6) is counted with these accepted
    # whole-frame decisions. Builds before selection revision 9 measured D on the samples after such a decision
    # (full_alpha_d, listed when it measured something); builds from revision 9 through dc7e3c77 logged it as 0 and
    # later lines omit it.
    #
    # Only a same-batch Backbuffer pair is exact (E2): a Present-counted pair can belong to another frame and then
    # differs everywhere, so a session whose gate offered HUD-less pairs but never a same-batch one cannot have decided
    # 6 correctly (Hogwarts Legacy 10-05, which tagged only HUDLessColor: whole gameplay seconds flat).
    full_alpha, alpha_visible = get('full_alpha'), get('full_alpha_d.visible')
    measured = sum(get(f'full_alpha_d.{k}') for k in ('hidden', 'ambiguous', 'visible', 'invalid'))
    exact = get('decided.6') if s2b else 0
    counted_only = bool(exact and s is not None and s.gate_counted and not s.gate_batch)
    intended = (f'; {alpha_visible} samples pinned an accepted whole-frame decision flat over a visible scene, as '
                'intended (P1)' if alpha_visible and not counted_only else '')
    breakdown = (f'; samples of them read the scene hidden {get("full_alpha_d.hidden")}, ambiguous '
                 f'{get("full_alpha_d.ambiguous")}, visible {alpha_visible}, invalid or unmeasured '
                 f'{get("full_alpha_d.invalid")}{intended}' if measured else '')
    if counted_only:
        add(Check('FAIL', 'UI full frame pairing',
                  f'{exact} frames ({percent(exact, detected)} of detection frames) decided the whole frame flat (6) '
                  f'while the game offered no same-batch Backbuffer pair ({s.gate_counted} Present-counted HUD-less '
                  'pairings, 0 same-batch): only a same-batch pair is exact (E2), so these flattened frames whose '
                  'pair may belong to another frame'))
    if s2b and full_alpha + exact:
        add(Check('INFO', 'UI full alpha',
                  f'{full_alpha} frames decided a whole-frame alpha and {exact} an exact full change-set (6) '
                  f'({percent(full_alpha + exact, detected)} of detection frames)' + breakdown))
    elif full_alpha:
        add(Check('INFO', 'UI full alpha',
                  f'{full_alpha} frames ({percent(full_alpha, detected)} of detection frames) decided a whole-frame '
                  'alpha' + breakdown))
    elif 'full_alpha' in c:
        add(Check('PASS', 'UI full alpha', 'no frame decided a whole-frame alpha'
                  + (' or an exact full change-set (6)' if s2b else '')))

    # Since S1 only accepted candidates decide, so the word is zero by construction: a count is a defect. Counters
    # from before S1 (trust.opaque_set rather than trust.discarded) still let the untrusted pass decide. Only counter
    # lines through dc7e3c77 carry the word; a later line no longer measures it, so it has no check.
    if 'untrusted_inferred' in c:
        inferred, s1 = get('untrusted_inferred'), 'trust.discarded' in c
        add(Check(('FAIL' if s1 else 'WARN') if inferred else 'PASS', 'UI inferred alpha',
                  f'{inferred} frames ({percent(inferred, detected)} of detection frames) decided from unaccepted '
                  'inferred alpha (UI layer, Backbuffer or current alpha)'
                  + ('; only accepted candidates decide since S1' if s1 else '; expected before S1') if inferred else
                  'no frame decided from unaccepted inferred alpha'))
    # A Present-counted (inexact) pair proposes the frame; its pixels decide whether the difference is UI (V2: a partial
    # change set within broad unchanged scene in clean tiles), so a partial difference from it is expected wherever a
    # game offers no same-batch pair. Only its whole-frame claim needs the hidden-scene guard (H1 d).
    inexact = get('inexact_difference')
    add(Check('INFO' if inexact else 'PASS', 'UI inexact difference',
              f'{inexact} frames ({percent(inexact, detected)} of detection frames) decided a HUD-less difference '
              'from a Present-counted pair, validated by its own pixels (V2)' if inexact else
              'no HUD-less difference decided from an inexact pair'))

    if s2a:
        # T1: a generated Present shows the decision of the real frame it shows (generated) or has none (none); a real
        # frame without a decision of its own reuses the previous real frame's once, counted as a detection frame.
        reused = get('reused')
        add(Check('INFO', 'UI holds',
                  f'generated {get("held.generated")}, none {get("held.none")} ({percent(held, auto)} of Auto frames); '
                  f'{reused} detection frames ({percent(reused, detected)}) reused the previous real frame\'s decision '
                  '(T1)'))
    else:
        add(Check('INFO', 'UI holds',
                  ', '.join(f'{k.replace("_", " ")} {get(f"held.{k}")}' for k in S1_HOLD_KINDS)
                  + f' ({percent(held, auto)} of Auto frames); {get("held.cap")} frames wanted a hold past the cap'))
    none = [(reason.replace('_', ' '), get(f'none.{reason}')) for reason in NO_MASK_REASONS]
    skipped = [(reason.replace('_', ' '), get(f'inactive.{reason}')) for reason in INACTIVE_REASONS]
    unheld = [('generated Presents without a decision', get('held.none'))] if s2a else []

    def shares(parts: list[tuple[str, int]]) -> str:
        return ', '.join(f'{name} {percent(n, auto)}' for name, n in parts if n)
    # Since S2a the sampled lines name each own decision's refused candidate (F1); the counters count reasons only.
    named = ', '.join(f'{reason.replace("_", " ")}' + (f' ({KIND_NAMES[kind]})' if kind in KIND_NAMES else '')
                      + f' {n}' for (reason, kind), n in refused.most_common(6))
    add(Check('INFO', 'UI no mask',
              f'{percent(sum(n for _, n in none + skipped + unheld), auto)} of Auto frames had no mask'
              + (f'; decided none: {shares(none)}' if shares(none) else '')
              + (f'; without detection: {shares(skipped)}' if shares(skipped) else '')
              + (f'; {shares(unheld)}' if shares(unheld) else '')
              + (f'; sampled reasons with the refused candidate: {named}' if named else '')))
    if s2a:
        add(Check('INFO', 'UI trust events',
                  f'earned {get("trust.earned")}, revoked one way by an exact pair {get("trust.revoked_exact")}, '
                  f'revoked by a declared alpha\'s coverage {get("trust.revoked_declared")}, lapsed '
                  f'{get("trust.lapsed")}, restored {get("trust.restored")}, legacy entries discarded '
                  f'{get("trust.discarded")}, forgotten {get("trust.forgotten")}'))
    else:
        add(Check('INFO', 'UI trust events',
                  f'earned {get("trust.earned")}, revoked by a full claim over the scene {get("trust.revoked_full")}, '
                  f'revoked by presented disagreement {get("trust.revoked_presented")}, lapsed {get("trust.lapsed")}, '
                  f'restored {get("trust.restored")}, '
                  + (f'legacy entries discarded {get("trust.discarded")}' if 'trust.discarded' in c else
                     f'opaque proof set {get("trust.opaque_set")} and cleared {get("trust.opaque_cleared")}')))


def stream_fps_at(s: Session, t: float) -> float:
    """The host's stream fps in effect at t on this log's clock (infinite when no host log was read)."""
    current = s.stream_fps_timeline[0][1] if s.stream_fps_timeline else s.stream_fps or float('inf')
    for when, value in s.stream_fps_timeline:
        if when <= t:
            current = value
    return current


def delivery_checks(s: Session, add) -> None:
    """'Stream delivery': the host should take every new frame the game offers, up to the stream fps (the stream
    follows the game; no repeats). Per counter window, claimed = published - overwritten_unconsumed and offered =
    published + dropped; the target is min(offered, stream fps). Windows that start in the settle time after an FG
    switch, reset or export start are skipped. Without the host log's stream rate the rates are INFO only."""
    fps = s.stream_fps or float('inf')
    rows = []
    for a, b in zip(s.delivery, s.delivery[1:]):
        dt = b[0] - a[0]
        if (b[1], b[2]) != (a[1], a[2]) or b[3] < a[3] or dt < 2.0 or any(0 <= a[0] - m < SETTLE_S for m in s.settle):
            continue
        claimed = (b[3] - a[3]) - (b[5] - a[5])
        offered = (b[3] - a[3]) + (b[4] - a[4])
        if offered <= 0:
            continue
        rows.append((a[0], b[0], claimed / dt, min(offered / dt, stream_fps_at(s, a[0])), offered / dt))
    if not rows:
        return
    total = sum(b - a for a, b, *_ in rows)
    claimed = sum((b - a) * c for a, b, c, _, _ in rows) / total
    if not s.stream_fps:
        offered = sum((b - a) * o for a, b, _, _, o in rows) / total
        add(Check('INFO', 'Stream delivery', f'stream rate unknown (pass --host-log to judge it): the host took '
                                             f'{claimed:.1f} new frames/s, the game offered {offered:.1f}/s'))
        return
    target = sum((b - a) * g for a, b, _, g, _ in rows) / total
    windows: list[list[float]] = []
    for a, b, c, g, o in rows:
        if c >= g * DELIVERY_SHARE:
            continue
        if windows and a - windows[-1][1] < 1.0:
            windows[-1][1] = b
            windows[-1][2:] = [windows[-1][2] + c * (b - a), windows[-1][3] + g * (b - a)]
        else:
            windows.append([a, b, c * (b - a), g * (b - a)])
    long = [w for w in windows if w[1] - w[0] >= DELIVERY_WARN_S]
    # The rates in effect during this session: the one at its start and each later change.
    start = s.first if s.first is not None else 0.0
    rates = sorted({stream_fps_at(s, start)} | {value for when, value in s.stream_fps_timeline if when > start})
    source = f'stream {rates[0]:g}-{rates[-1]:g} fps, followed live' if len(rates) > 1 else f'stream {fps:g} fps'
    add(Check('WARN' if long else 'PASS', 'Stream delivery',
              f'the host took {claimed:.1f} new frames/s of a target {target:.1f}/s (what the game offered, capped at '
              f'the {source})' + (f'; {len(long)} {"window" if len(long) == 1 else "windows"} below '
                                  f'{DELIVERY_SHARE:.0%} of the target for {DELIVERY_WARN_S:g} s or longer' if long
                                  else ''),
              [f'{span(a, b)} ({b - a:.0f} s): {c / (b - a):.1f} of {g / (b - a):.1f} new frames/s'
               for a, b, c, g in long][:6]))


def layer_copy_checks(s: Session, add) -> None:
    """'UI layer copies': the offscreen UI layer's ring of live copies (game3d_ui_layer.h) skips a copy when every
    entry is offered, held for its fence or still read, and the offered copy stays offered: Presents keep reading an
    older layer until a copy is recorded again, and the host publishes them like any other, whatever the game's or the
    stream's rate. A skip's cost is that time, not a count or a rate, so the check judges each timing window's
    skip_gap_ms: the longest time the layer went unrefreshed across skipped copies. While streamed (overlapping a
    streamed span), a gap longer than one stream frame interval (the stream fps at the window's start) warns: streamed
    frames could read a UI layer that missed a whole stream frame of UI changes (a UI distortion risk). A gap of
    LAYER_RECENCY_MS or more warns in any window: the layer was then no longer offered at all (a ring that stayed
    full, which backpressure never causes). Shorter gaps (The Witcher 3's ~590 Presents/s exit screen 10-07: one skip
    costs about 3 ms), skips outside the streamed export or without the host log's stream rate, and skips on lines
    before 10-07, whose gap is not logged, are INFO. A full-ring line after the last timing line names copies whose
    count was never logged (before 10-07 a runtime reset dropped its partial window). The gap is a maximum per window,
    so neither the window's other play nor its average rates can hide a burst of skips."""
    windows = [w for w in s.layer_windows if w[3] or w[4]]
    unlogged = [t for t in s.layer_saturated if not any(w[0] >= t for w in s.layer_windows)]
    if not windows and not unlogged:
        return

    def windows_of(n: int) -> str:
        return f'{n} timing {"window" if n == 1 else "windows"}'
    stale, stuck, bounded, unstreamed, unjudged, unlogged_gap = [], [], [], [], [], []
    for end, length, presents, copies, skipped, _, mean, peak, gap in windows:
        if not skipped:
            continue
        length = max(length, 0.001)
        start = end - length
        frame_ms = 1000.0 / stream_fps_at(s, start) if s.stream_fps else None
        streamed = any(a < end and (b is None or b > start) for a, b in s.streamed)
        cost = ('refresh gap not logged' if gap is None else
                f'longest refresh gap {gap:.1f} ms' + (f' of a {frame_ms:.1f} ms stream frame' if frame_ms else ''))
        row = (f'{span(start, end)} ({length:.1f} s): skipped {skipped} of {copies + skipped} copies, {cost} '
               f'({presents / length:.0f} Presents/s, presents_since_copy mean {mean:.2f} max {peak})'
               + ('' if streamed else ', not streamed'))
        (unlogged_gap if gap is None else stuck if gap >= LAYER_RECENCY_MS else unstreamed if not streamed else
         unjudged if not frame_ms else stale if gap > frame_ms else bounded).append(row)
    if stale or stuck:
        parts = []
        if stale:
            parts.append(f'{windows_of(len(stale))} streamed with skipped layer copies leaving the layer unrefreshed '
                         'longer than a stream frame (a full ring: streamed frames read an older UI layer)')
        if stuck:
            parts.append(f'{windows_of(len(stuck))} with skipped layer copies leaving the layer unrefreshed for '
                         f'{LAYER_RECENCY_MS:.0f} ms or more, after which it is no longer offered (a ring that stayed '
                         'full)')
        add(Check('WARN', 'UI layer copies', '; '.join(parts),
                  (stuck + stale + unjudged + unstreamed + bounded + unlogged_gap)[:6]))
        return
    kept = unjudged + unstreamed + bounded + unlogged_gap
    groups = [(bounded, 'every refresh gap within a stream frame'), (unstreamed, 'outside the streamed export'),
              (unjudged, 'stream rate unknown (pass --host-log to judge them)'),
              (unlogged_gap, 'refresh gap not logged (an add-on before 10-07)')]
    parts = ['no logged timing window counted layer copies' if not windows else
             f'no layer copy skipped in {windows_of(len(windows))}' if not kept else
             f'{windows_of(len(kept))} skipped layer copies (a full ring): '
             + ', '.join(f'{len(rows)} {label}' for rows, label in groups if rows)]
    if unlogged:
        parts.append(f'copies skipped from {clock(unlogged[0])}, after the last timing line, were not counted in any '
                     "logged window, so their refresh gap is unknown (an add-on before 10-07 dropped a runtime "
                     "reset's partial window, or the process ended first)")
    add(Check('INFO' if kept or unlogged else 'PASS', 'UI layer copies', '; '.join(parts), kept[:6]))


def held_none_checks(s: Session, add) -> None:
    """'UI holds without a decision' (T1): a Present held without a real-frame decision to show (held.none) has no UI
    mask. The counters advance only when a sample commits, so each window runs from one counter line that advanced to
    the next; a window in which at least HELD_NONE_WARN of the Auto frames were held so warns, and one with at least
    HELD_NONE_FAIL for HELD_NONE_FAIL_S fails: UI detection effectively never ran (Hogwarts Legacy 10-05 at 4x frame
    generation, every Present classified generated before the HUD-less fix)."""
    changed = [(t, c) for i, (t, c) in enumerate(s.counter_history)
               if i == 0 or c.get('auto_frames', 0) != s.counter_history[i - 1][1].get('auto_frames', 0)]
    if not changed or 'held.none' not in changed[-1][1]:
        return
    windows: list[list] = []
    for (a, before), (b, after) in zip(changed, changed[1:]):
        auto = after.get('auto_frames', 0) - before.get('auto_frames', 0)
        none = after.get('held.none', 0) - before.get('held.none', 0)
        if auto <= 0 or none < auto * HELD_NONE_WARN:
            continue
        if windows and a - windows[-1][1] < 1.0:
            windows[-1][1:] = [b, windows[-1][2] + none, windows[-1][3] + auto]
        else:
            windows.append([a, b, none, auto])
    listed = [w for w in windows if w[1] - w[0] >= UNPROTECTED_MIN_S]
    if not listed:
        add(Check('PASS', 'UI holds without a decision', 'no window held most Auto frames without a decision'))
        return

    def fg(a: float, b: float) -> str:
        mode = 0
        for t, m in s.fg_switches:
            if t <= (a + b) / 2:
                mode = m
        return 'FG on' if mode else 'FG off'
    failed = any(n >= auto * HELD_NONE_FAIL and b - a >= HELD_NONE_FAIL_S for a, b, n, auto in listed)
    add(Check('FAIL' if failed else 'WARN', 'UI holds without a decision',
              f'{len(listed)} {"window" if len(listed) == 1 else "windows"} held at least '
              f'{HELD_NONE_WARN:.0%} of the Auto frames without a real-frame decision to show (T1), so those frames '
              'had no UI mask',
              [f'{span(a, b)} ({b - a:.0f} s) {fg(a, b)}: {percent(n, auto)} of {auto} Auto frames held without a '
               'decision' for a, b, n, auto in listed][:6]))


def full_frame_s2b(c: dict[str, int], add, detected: int) -> None:
    """'UI full frame' from counter lines since S2b: H1 (source 8) and its scene guard (docs/reshade-sbs.md, UI
    decision framework, H1 and M5).

    full_d counts H1 samples only. H1 shows the frame flat only under a held hidden verdict, and one valid visible
    sample releases that hold at once: the releasing sample decided 8 under the hold before its own evidence read the
    scene visible, and the guard counts it as a release (scene.released). So full_d.visible never exceeds
    scene.released unless an H1 hold acted over a visible scene. An accepted exact full change-set (6) decides without
    D (P1) and is reported with the accepted whole-frame decisions (UI full alpha)."""
    def get(key: str) -> int:
        return c.get(key, 0)
    h1, exact = get('decided.8'), get('decided.6')
    visible, released = get('full_d.visible'), get('scene.released')
    acted = visible > released
    add(Check('WARN' if acted else 'PASS', 'UI full frame',
              f'{h1 + exact} full-frame frames (6: {exact}, 8: {h1}; {percent(h1 + exact, detected)} of detection '
              f'frames); H1 samples (8) read the scene hidden {get("full_d.hidden")}, ambiguous '
              f'{get("full_d.ambiguous")}, visible {visible}, invalid or unmeasured {get("full_d.invalid")}; scene '
              f'guard entered {get("scene.entered")}, released {released}, refuted {get("scene.refuted")}; depth not '
              f'current on {get("full.depth_not_current")} detection frames'
              + (f'; an H1 hold acted over a visible scene: {visible} visible H1 samples for {released} releases'
                 if acted else '')))


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
    second apart merge, as windows() merges counter intervals. Lines whose status sample was pending, and unrendered
    lines, continue a run as the overlay's run does (next_unprotected_since in game3d_controls.h); each piece is
    labelled with its own line's FG state, a pending one as searching after the last sample's reason."""
    runtimes: dict[str, list[UISample]] = {}
    for u in s.ui:
        runtimes.setdefault(u.runtime, []).append(u)
    pieces = []
    for samples in runtimes.values():
        last = ''
        for u, following in zip(samples, samples[1:] + [None]):
            if u.unprotected():
                last = u.why_body()
                reason = u.why_unprotected()
            elif last and (u.pending() or (u.mode == 'auto' and not u.rendered)):
                reason = f'{u.fg_label()}: {u.detection if u.rendered else "not rendered"}; last sample: {last}'
            else:
                last = ''
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
    reading hidden while no UI source decided, which nothing protected (reported by the first-run shadow of builds
    before selection revision 9, removed since).

    Before S2b a sample whose frame was full-frame UI (8 or 9) and whose own evidence read the presented frame
    visible is a route's exit: that verdict releases the hold at once, so each hidden scene that ends shows one, and a
    false hidden verdict shows one too. Exits are counted, not judged; many short episodes deserve a look. Since S2b
    the H1 samples (8) are listed with the pre-UI scene image whose claim acted (H1 (d)), and the scene guard's
    entries, releases and refutations come from the counter line (M5); hidden samples whose layer pre-UI claim could
    not act because the layer was not proven are counted: in S2b lines a layer claim (0x80) without the guard's D
    proof, since fix 1 an offered layer without coverage whose signature has no ledger proof yet (an unproven layer no
    longer claims)."""
    scenes = [u for u in s.ui if u.scene]
    if not scenes:
        return
    # Logs before selection revision 9: the first-run shadow's hidden runs without a decided source. Consecutive lines
    # report the same run with a growing length, so each run is listed once, by its start, at its longest.
    runs: dict[float, tuple[float, UISample]] = {}
    for u in scenes:
        if not u.source and u.scene.hidden_ms >= SHADOW_HIDDEN_WARN_MS:
            start = u.t - u.scene.hidden_ms / 1000.0
            key = next((k for k in runs if abs(k - start) <= UI_LINE_PERIOD_S / 10), start)
            if key not in runs or u.scene.hidden_ms > runs[key][1].scene.hidden_ms:
                runs[key] = (start, u)
    uncovered = [f'{clock(start)} {u.scene.hidden_ms} ms{" (first-run shadow)" if u.scene.shadow else ""}'
                 for start, u in sorted(runs.values(), key=lambda r: r[0])]
    guarded = [u for u in scenes if u.scene.s2b]
    if guarded:
        h1 = [u for u in guarded if u.source == 8]
        pre_ui = Counter(u.scene.pre_ui_image for u in h1 if u.scene.pre_ui_claim())
        c = s.counters if s.counters is not None and 'scene.entered' in s.counters else None
        guard = (f'entered {c["scene.entered"]}, released {c["scene.released"]}, refuted {c["scene.refuted"]}' if c
                 else f'hidden hold held in {sum(1 for u in guarded if u.scene.guard[0])} samples (no counter line)')
        detail = (f'H1 hidden scene (8) in {len(h1)} samples (pre-UI image: hudless {pre_ui["hudless"]}, layer '
                  f'{pre_ui["layer"]}), {guard}; {sum(u.scene.ran for u in scenes)} of {len(scenes)} samples '
                  'measured')
        unproven = [u for u in guarded if u.scene.pre_ui_image == 'layer' and u.scene.proven is False and u.hidden()
                    and (u.bare_layer() if u.scene.fix1 else u.scene.claims & CLAIM_PRE_UI)]
        fix1 = sum(1 for u in unproven if u.scene.fix1)
        if fix1:
            detail += (f'; {fix1} hidden samples had an unproven pre-UI layer (no proof yet: {PRE_UI_PROOF_RULE})')
        if len(unproven) > fix1:
            detail += (f'; {len(unproven) - fix1} hidden samples had an unproven pre-UI layer (no gameplay sample had '
                       'read it within 0.03 of the presented frame, the proof of S2b lines)')
    else:
        covered = Counter(u.source for u in scenes if u.source in (8, 9))
        exits = sum(1 for u in scenes if u.source in (8, 9) and u.scene.valid and u.scene.verdict == 'visible')
        detail = (f'layer route (8) in {covered[8]} samples, HUD-less route (9) in {covered[9]}, '
                  f'{exits} released by a visible verdict; {sum(u.scene.ran for u in scenes)} of {len(scenes)} '
                  'samples measured')
    if uncovered:
        detail += f'; the presented frame read hidden for at least {SHADOW_HIDDEN_WARN_MS} ms with no UI source'
    add(Check('WARN' if uncovered else 'INFO', 'Hidden scene', detail, uncovered[:6]))


def host_stream_fps(path: Path, start: datetime, end: datetime) -> list[tuple[datetime, float]]:
    """The stream frame rate the host log set up to the session's end: each requested rate at stream start and
    each live video-mode change, in time order."""
    timeline: list[tuple[datetime, float]] = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        m = HOST_LINE.match(line)
        if not m:
            continue
        when = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
        if when > end + timedelta(minutes=1):
            continue
        if found := HOST_FPS.search(m.group(4)):
            num, den, whole = found.groups()
            timeline.append((when, float(whole) if whole else float(num) / max(1.0, float(den))))
        elif found := HOST_LIVE_FPS.search(m.group(4)):
            timeline.append((when, float(found.group(1) or found.group(2))))
    return timeline


def host_checks(path: Path, start: datetime, end: datetime) -> list[Check]:
    checks: list[Check] = []
    stalls, connects, errors, fits = [], 0, [], []
    open_stalls: dict[int, int] = {}  # NVENC frame -> index in stalls of its stall.
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
            kind = 'encoder-side' if 'done_by' in text else 'upstream GPU' if 'upstream' in text else 'unknown'
            frame = NVENC_FRAME.search(text)
            index = open_stalls.get(int(frame.group(1))) if frame else None
            if index is not None and 0 <= t - stalls[index][0] <= NVENC_SAME_STALL_S:
                stalls[index] = (stalls[index][0], kind)
            else:
                if frame:
                    open_stalls[int(frame.group(1))] = len(stalls)
                stalls.append((t, kind))
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
                        [f'{clock(at)} {kind}' for at, kind in stalls[:6]]))
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
    # A runtime teardown at the very end is a normal exit whose process ended before ReShade logged its own.
    torn_down = session.teardown is not None and session.last - session.teardown <= TEARDOWN_TAIL_S
    state = ('still running' if running else 'exited' if session.exited else
             'exited (the process ended before ReShade logged its exit)' if torn_down else
             'ended without exiting (crash or kill?)')
    print(f'Game 3D readiness: {session.exe or log}  {clock(session.first)} to {clock(session.last)} '
          f'({int(duration // 60)}m{int(duration % 60):02}s, {state})')
    end = None
    if args.host_log:
        # ReShade logs times without a date; the log's last write dates the session.
        written = datetime.fromtimestamp(log.stat().st_mtime)
        end = datetime.combine(written.date(), datetime.min.time()) + timedelta(seconds=session.last % 86400)
        if end > written + timedelta(minutes=5):
            end -= timedelta(days=1)
        start = end - timedelta(seconds=duration)
        timeline = host_stream_fps(args.host_log, start, end)
        if timeline:
            # Host times on this log's clock; the last rate before the session is in effect at its start.
            session.stream_fps = timeline[-1][1]
            session.stream_fps_timeline = [(session.first + (when - start).total_seconds(), fps)
                                           for when, fps in timeline]
    checks = evaluate(session)
    if end is not None:
        checks += host_checks(args.host_log, end - timedelta(seconds=duration), end)
    if not running and not session.exited and not torn_down:
        checks.append(Check('WARN', 'Session', 'the log ends without ReShade exiting or tearing its runtimes down'))
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
