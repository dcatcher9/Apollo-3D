# SPDX-License-Identifier: GPL-3.0-only
"""Unit tests for game3d_log_report: python -m unittest tools/reshade/test_game3d_log_report.py"""

import re
import sys
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game3d_log_report as report  # noqa: E402


def line(t, text, level='INFO'):
    return f'{t}:000 [ 1234] | {level:5} | {text}'


def output(t, published, fg, flat, fresh, missing, runtime='0x1'):
    return line(t, f'[Sunshine 3D] Sunshine SBS output: published={published} fg={fg} scene_flat={flat} '
                   f'scene_fading=0 published_fresh_depth={fresh} published_reused_depth=0 '
                   f'published_depth_missing={missing} (unavailable={missing} reuse_after_gap=0 reuse_other_source=0) '
                   f'runtime={runtime} generation=1; cumulative')


def ui(t, source, covered, alpha, candidates, trusted, pixels=1000, hudless=(0, 0), layer=0, invalid=(0, 0, 0, 0),
       scene=None, detection=None, availability=None, mode='auto', rendered=1, fg=0, runtime='0000000000000001'):
    detection = detection or ('detected' if source else 'no_usable_mask')
    availability = availability or ('detected' if source else 'quality_rejected')
    a = '/'.join(str(v) for v in alpha)
    i = '/'.join(str(v) for v in invalid)
    # scene: (d, verdict, hold, shadow, hidden_ms) of a log with hidden-scene evidence.
    evidence = ''
    if scene:
        d, verdict, hold, shadow, hidden_ms = scene
        evidence = (f'sampled_alpha_opaque=0/{pixels} '
                    f'sampled_scene={{n=400 d={d:.3f} valid=1 ran=1 verdict={verdict}}} '
                    f'sampled_hudless_scene={{n=0 d=0.000 valid=0}} scene_hold={hold} shadow={shadow} '
                    f'shadow_hidden_ms={hidden_ms} ')
    return line(t, f'[Sunshine 3D] Sunshine UI protection: runtime={runtime} mode={mode} rendered={rendered} '
                   f'mask_path=1 input=automatic_gpu_mask retained=1 fg={fg} fg_known=1 fg_enabled={fg} '
                   f'input_state=input_seen detection={detection} selected=automatic source=automatic '
                   f'source_availability={availability} '
                   f'sampled_source={source} sampled_covered={covered} sampled_pixels={pixels} '
                   f'sampled_candidates=0x{candidates:x} sampled_alpha_covered={a} sampled_alpha_invalid={i} '
                   f'trusted_alpha=0x{trusted:x} '
                   f'sampled_ui_layer={layer} sampled_hudless={{changed={hudless[0]} unchanged={hudless[1]} '
                   f'invalid=0 matching_tiles=0 lit=0}} {evidence}status_revision=1')


def ui1(t, source, covered, alpha, candidates, accepted, pixels=1000, hudless=(0, 0), layer=(0, 0),
        invalid=(0, 0, 0, 0), detection=None, fg=0):
    """A 'Sunshine UI protection' line since S1: candidate bits with the layer as 0x40, accepted candidates, and the
    layer's (covered, invalid) pixels."""
    detection = detection or ('detected' if source else 'no_usable_mask')
    a = '/'.join(str(v) for v in alpha)
    i = '/'.join(str(v) for v in invalid)
    return line(t, f'[Sunshine 3D] Sunshine UI protection: runtime=0000000000000001 mode=auto rendered=1 '
                   f'mask_path=1 input=automatic_gpu_mask retained=0 fg={fg} fg_known=1 fg_enabled={fg} '
                   f'input_state=input_seen detection={detection} selected=automatic source=automatic '
                   f'source_availability={"detected" if source else "quality_rejected"} '
                   f'sampled_source={source} sampled_covered={covered} sampled_pixels={pixels} '
                   f'sampled_candidates=0x{candidates:x} sampled_alpha_covered={a} sampled_alpha_invalid={i} '
                   f'accepted=0x{accepted:x} sampled_layer={{covered={layer[0]} invalid={layer[1]} opaque=0}} '
                   f'sampled_hudless={{changed={hudless[0]} unchanged={hudless[1]} invalid=0 matching_tiles=0 lit=0}} '
                   f'status_revision=1')


def ui2(t, source, covered, alpha, candidates, accepted, pixels=1000, hudless=(0, 0), layer=(0, 0),
        invalid=(0, 0, 0, 0), strong=(0, 0, 0), contradicted=(0, 0, 0), reason=None, refused='none', reused=0,
        tiles=0, lit=0, detection=None, fg=0, late=0):
    """A 'Sunshine UI protection' line since S2a: the S1 fields, then the one-way counts of the layer, Backbuffer and
    current alpha, the own decision's reason, the refused candidate, the T1 reuse and whether the layer was the
    one-frame-late copy (exporter.cpp through 180f1842; late=None omits it, as later lines do), and the HUD-less
    matching tiles and lit pixels."""
    detection = detection or ('detected' if source else 'no_usable_mask')
    reason = reason or ('decided' if source and not reused else 'unaccepted')
    a = '/'.join(str(v) for v in alpha)
    i = '/'.join(str(v) for v in invalid)
    s = '/'.join(str(v) for v in strong)
    c = '/'.join(str(v) for v in contradicted)
    return line(t, f'[Sunshine 3D] Sunshine UI protection: runtime=0000000000000001 mode=auto rendered=1 '
                   f'mask_path=1 input=automatic_gpu_mask retained=0 fg={fg} fg_known=1 fg_enabled={fg} '
                   f'input_state=input_seen detection={detection} selected=automatic source=automatic '
                   f'source_availability={"detected" if source else "quality_rejected"} '
                   f'sampled_source={source} sampled_covered={covered} sampled_pixels={pixels} '
                   f'sampled_candidates=0x{candidates:x} sampled_alpha_covered={a} sampled_alpha_invalid={i} '
                   f'accepted=0x{accepted:x} sampled_layer={{covered={layer[0]} invalid={layer[1]} opaque=0}} '
                   f'sampled_one_way={{strong={s} contradicted={c}}} sampled_reason={reason} '
                   f'sampled_refused={refused} sampled_reused={reused} '
                   + ('' if late is None else f'sampled_late_layer={late} ') +
                   f'sampled_hudless={{changed={hudless[0]} unchanged={hudless[1]} invalid=0 '
                   f'matching_tiles={tiles} lit={lit}}} sampled_alpha_opaque=0/0 '
                   f'sampled_scene={{n=0 d=0.000 valid=0 ran=0 verdict=none}} '
                   f'sampled_hudless_scene={{n=0 d=0.000 valid=0}} scene_hold=0 shadow=0 shadow_hidden_ms=0 '
                   f'status_revision=1')


def ui3(t, source, covered, alpha, candidates, accepted, pixels=1000, layer=(0, 0), invalid=(0, 0, 0, 0),
        reason=None, refused='none', scene=(0.6, 'visible'), claims=0, h1=0, winner=None, pre_ui=('none', 0.0, 0),
        guard=(0, 0, 0), ran=1, shadow=0, hidden_ms=0, detection=None, fg=0, pre_ui_pixels=None, still=None):
    """A 'Sunshine UI protection' line since S2b (exporter.cpp): the S2a fields, then the Backbuffer and current
    alpha's opaque pixels, the informative full claims, the H1 word, the presented frame's D, the pre-UI scene image's
    D (image, d, valid) and the scene guard's holds (hidden, pre-UI, refuted signatures). Since fix 1 pre_ui_pixels
    (match, image lit, presented lit, presented lit and different) follows shadow_hidden_ms, and since fix 2 still
    (scope, enabled, phase, run_ms, still cells, compared cells, short_max_ms), rule H2's group, follows it."""
    detection = detection or ('detected' if source else 'no_usable_mask')
    reason = reason or ('decided' if source else 'unaccepted')
    winner = source if winner is None else winner
    a = '/'.join(str(v) for v in alpha)
    i = '/'.join(str(v) for v in invalid)
    d, verdict = scene
    image, pre_ui_d, pre_ui_valid = pre_ui
    return line(t, f'[Sunshine 3D] Sunshine UI protection: runtime=0000000000000001 mode=auto rendered=1 '
                   f'mask_path=1 input=automatic_gpu_mask retained=0 fg={fg} fg_known=1 fg_enabled={fg} '
                   f'input_state=input_seen detection={detection} selected=automatic source=automatic '
                   f'source_availability={"detected" if source else "quality_rejected"} '
                   f'sampled_source={source} sampled_covered={covered} sampled_pixels={pixels} '
                   f'sampled_candidates=0x{candidates:x} sampled_alpha_covered={a} sampled_alpha_invalid={i} '
                   f'accepted=0x{accepted:x} sampled_layer={{covered={layer[0]} invalid={layer[1]} opaque=0}} '
                   f'sampled_one_way={{strong=0/0/0 contradicted=0/0/0}} sampled_reason={reason} '
                   f'sampled_refused={refused} sampled_reused=0 sampled_late_layer=0 '
                   f'sampled_hudless={{changed=0 unchanged=0 invalid=0 matching_tiles=0 lit=0}} '
                   f'sampled_alpha_opaque=0/0 sampled_inferred_opaque=0/{alpha[3]} sampled_claims=0x{claims:x} '
                   f'sampled_h1={{applied={h1} winner={winner}}} '
                   f'sampled_scene={{n=463 d={d:.3f} valid=1 ran={ran} verdict={verdict}}} '
                   f'sampled_pre_ui_scene={{image={image} n={463 if pre_ui_valid else 0} d={pre_ui_d:.3f} '
                   f'valid={pre_ui_valid}}} scene_guard={{hidden={guard[0]} pre_ui={guard[1]} refuted={guard[2]}'
                   f'{f" proven={guard[3]}" if len(guard) > 3 else ""}}} '
                   f'shadow={shadow} shadow_hidden_ms={hidden_ms} '
                   + ('' if pre_ui_pixels is None else
                      'sampled_pre_ui_pixels={{match={} image_lit={} presented_lit={} presented_lit_differs={}}} '
                      .format(*pre_ui_pixels))
                   + ('' if still is None else
                      'still={{scope={} enabled={} phase={} run_ms={} sampled={}/{} short_max_ms={}}} '.format(*still))
                   + 'status_revision=1')


def accepted_now(t, value):
    return line(t, f'[Sunshine 3D] Sunshine UI protection: accepted UI sources are now {value}; remembered for later '
                   'sessions of this game')


# The groups and keys of format_ui_counters (game3d_ui_counters.h), in its order since S2a.
COUNTER_GROUPS = (
    ('auto_frames', None), ('detection_frames', None),
    ('held', ('generated', 'none')), ('reused', None),
    ('inactive', ('no_candidates', 'size', 'unprepared')),
    ('decided', ('0', '1', '2', '3', '4', '5', '6', '8', '9', '10')),
    ('none', ('layer_aside', 'trusted_invalid', 'presented_blocked', 'ambiguous', 'difference_failed',
              'gate_no_hold', 'no_candidate', 'other', 'unaccepted')),
    ('full', ('6', '8', '9', 'depth_not_current')),
    ('full_d', ('hidden', 'ambiguous', 'visible', 'invalid')),
    ('untrusted_inferred', None), ('inexact_difference', None), ('contradicted', None),
    ('presented_over_dedicated', None), ('full_alpha', None),
    ('full_alpha_d', ('hidden', 'ambiguous', 'visible', 'invalid')),
    ('trust', ('earned', 'revoked_exact', 'revoked_declared', 'lapsed', 'restored', 'discarded', 'forgotten')),
    ('samples', None), ('through_ms', None),
)
# The same since S2b: decided and full omit the retired 9, and the scene guard's group follows full_d.
S2B_COUNTER_GROUPS = tuple(
    group for key, inner in COUNTER_GROUPS
    for group in ((key, tuple(k for k in inner if k != '9') if key in ('decided', 'full') else inner),)
    + ((('scene', ('entered', 'released', 'refuted')),) if key == 'full_d' else ()))
# The same after dc7e3c77: without the counters that only ever logged 0 there (untrusted_inferred,
# presented_over_dedicated and full_alpha_d).
CURRENT_COUNTER_GROUPS = tuple(
    (key, inner) for key, inner in S2B_COUNTER_GROUPS
    if key not in ('untrusted_inferred', 'presented_over_dedicated', 'full_alpha_d'))
# The same since 10-22 (format_ui_counters today): trust adds uncovered_color, the selective layer samples that
# earned nothing for colour without coverage.
UNCOVERED_COUNTER_GROUPS = tuple(
    (key, inner + ('uncovered_color',) if key == 'trust' else inner) for key, inner in CURRENT_COUNTER_GROUPS)
# The same since fix 2: decided adds 11 (H2's still screen), and H2's group follows the scene guard's.
FIX2_COUNTER_GROUPS = tuple(
    group for key, inner in S2B_COUNTER_GROUPS
    for group in ((key, inner + ('11',) if key == 'decided' else inner),)
    + ((('still', ('entered', 'released', 'short')),) if key == 'scene' else ()))
# The same in S1: three hold kinds and the cap, no reused, trusted_full and the S1 trust events.
S1_COUNTER_GROUPS = tuple(
    ('held', ('generated', 'inexact_after_exact', 'trusted_missing', 'cap')) if key == 'held' else
    ('trusted_full', None) if key == 'contradicted' else
    ('trust', ('earned', 'revoked_full', 'revoked_presented', 'lapsed', 'restored', 'discarded')) if key == 'trust'
    else (key, inner) for key, inner in COUNTER_GROUPS if key != 'reused')
# The same before S1: sources 0-9, no unaccepted reason, and the tagged UI colour's opaque proof.
S0_COUNTER_GROUPS = tuple(
    (key, tuple(k for k in inner if k not in ('10', 'unaccepted', 'discarded'))
     + (('opaque_set', 'opaque_cleared') if key == 'trust' else ()) if inner else None)
    for key, inner in S1_COUNTER_GROUPS)


def counters(t, values, runtime='0000000000000001', groups=COUNTER_GROUPS):
    """A 'Sunshine UI counters' line; values maps 'key' or 'group.key' to a count, every other field is 0."""
    fields = []
    for key, inner in groups:
        if inner is None:
            fields.append(f'{key}={values.get(key, 0)}')
        else:
            fields.append(f'{key}={{' + ' '.join(f'{k}={values.get(f"{key}.{k}", 0)}' for k in inner) + '}')
    return line(t, f'[Sunshine 3D] Sunshine UI counters: runtime={runtime} ' + ' '.join(fields))


# 100 Auto frames: 80 detected (60 by the UI layer, 2 of them reused by the T1 grace, and 20 without a mask, all for
# an ambiguous channel), 15 held (12 generated Presents showing a real frame's decision, 3 without one) and 5 without
# a candidate, over 30 committed samples.
CLEAN_COUNTERS = {'auto_frames': 100, 'detection_frames': 80, 'held.generated': 12, 'held.none': 3, 'reused': 2,
                  'inactive.no_candidates': 5, 'decided.10': 60, 'decided.0': 20, 'none.ambiguous': 20,
                  'trust.earned': 1, 'samples': 30, 'through_ms': 3000}
# The same session counted in S1: 15 held by the three S1 kinds.
S1_CLEAN_COUNTERS = {'auto_frames': 100, 'detection_frames': 80, 'held.generated': 10, 'held.inexact_after_exact': 3,
                     'held.trusted_missing': 2, 'inactive.no_candidates': 5, 'decided.10': 60, 'decided.0': 20,
                     'none.ambiguous': 20, 'trust.earned': 1, 'samples': 30, 'through_ms': 3000}


BASE = [
    line('10:00:00', "Initializing crosire's ReShade version '6.8' loaded from 'D:\\G\\dxgi.dll' "
                     "into 'D:\\G\\game.exe' ..."),
    line('10:00:01', 'Registered add-on "Sunshine 3D" v0.0.0.0 using ReShade API version 20.'),
    line('10:00:02', '[Sunshine 3D] Sunshine Game 3D: add-on GPU renderer ready (no FX file required)'),
    line('10:00:02', '[Sunshine 3D] Sunshine SBS: generation 1, 7680x2160 full SBS, DXGI 10, D3D12, scRGB '
                     '(source color 3)'),
    line('10:00:03', '[Sunshine 3D] Sunshine 3D Streamline scale: ready; viewport=0 encoding=device'),
    line('10:00:04', '[Sunshine 3D] Sunshine list lifecycle: admissions covered=900 (states observed=900 declared=0) '
                     'not_open={unknown=0 closed=0 pass=0}'),
    output('10:00:05', 100, 0, 0, 100, 0),
    output('10:00:10', 400, 0, 0, 400, 0),
    output('10:00:15', 700, 0, 0, 700, 0),
]


def run(lines):
    session = report.parse(lines)
    return {c.name: c for c in report.evaluate(session)}


# A real S2b line: Stellar Blade's SDR settings page with FG suspended (session 2026-10-03 07:20:31). The presented
# frame reads hidden, the cleared output target (the layer, colour without alpha) visible, and the layer is unproven,
# so H1 does not apply.
SB_S2B_SETTINGS = (
    '07:20:31:065 [71908] | INFO  | [Sunshine 3D] Sunshine UI protection: runtime=0000000102c619a0 mode=auto '
    'rendered=1 mask_path=1 input=automatic_gpu_mask retained=1 fg=0 fg_known=1 fg_enabled=0 '
    'input_state=not_observed detection=no_usable_mask selected=automatic source=automatic '
    'source_availability=quality_rejected sampled_source=0 sampled_covered=0 sampled_pixels=8294400 '
    'sampled_candidates=0x48 sampled_alpha_covered=0/0/0/8294400 sampled_alpha_invalid=0/0/0/0 accepted=0x0 '
    'sampled_layer={covered=0 invalid=4690222 opaque=0} sampled_one_way={strong=0/0/8294400 contradicted=0/0/0} '
    'sampled_reason=layer_aside sampled_refused=ui_layer sampled_reused=0 sampled_late_layer=1 '
    'sampled_hudless={changed=8239431 unchanged=1089 invalid=0 matching_tiles=0 lit=0} sampled_alpha_opaque=0/0 '
    'sampled_inferred_opaque=0/8294400 sampled_claims=0x80 sampled_h1={applied=0 winner=0} '
    'sampled_scene={n=911 d=-0.001 valid=1 ran=1 verdict=hidden} sampled_pre_ui_scene={image=layer n=911 d=0.544 '
    'valid=1} scene_guard={hidden=1 pre_ui=0 refuted=0 proven=0} shadow=1 shadow_hidden_ms=656 status_revision=17')
# The same sample as fix 1 logs it: an unproven layer no longer claims (0x80), and texel 11 follows shadow_hidden_ms
# (the counts of settings dump 468: 32.8% matching, 25.9% lit layer).
SB_FIX1_SETTINGS = SB_S2B_SETTINGS.replace('sampled_claims=0x80', 'sampled_claims=0x0').replace(
    'shadow_hidden_ms=656 ', 'shadow_hidden_ms=656 sampled_pre_ui_pixels={match=2719720 image_lit=2148784 '
    'presented_lit=5994692 presented_lit_differs=5238234} ')


# Real lines of Clair Obscur: Expedition 33 (2026-10-22, generation 2, FG on at 4x): the game stopped tagging depth and
# its UI buffers together (they share one Streamline tag batch) and nearly stalled. Depth readiness was lost from
# 13:47:15.502 (Present 21830) to 17.373 (Present 21899), and the output counter lines around it count those 69
# Presents as published without depth, so they were packed flat; the UI lines read no UI source from 15.633 until
# detection resumed (logged at 18.376, held by the 1 s log gate). The second no-source line, written by the recovering
# Present, differs from the first only in its time and status_revision.
E33_DEPTH_LOSS = [
    '13:45:06:786 [19344] | INFO  | Registered add-on "Sunshine 3D" v0.0.0.0 using ReShade API version 20.',
    '13:47:05:005 [51480] | INFO  | [Sunshine 3D] Sunshine Game 3D: add-on GPU renderer ready (built off the present '
    'thread in 35 ms, longest step adaptive probe resources 9.5 ms; present-thread adoption 0.002 ms; 3840x2160, '
    'colour 1)',
    '13:47:09:062 [40860] | INFO  | [Sunshine 3D] Sunshine Streamline frame generation: viewport=1 mode=1 '
    'generated_frames=3 supported=1 success=1',
    '13:47:09:353 [40860] | INFO  | [Sunshine 3D] Sunshine SBS: generation 2, 7680x2160 full SBS, DXGI 24, D3D12, '
    'sRGB, protocol 4 (source color 1)',
    '13:47:14:412 [40860] | INFO  | [Sunshine 3D] Sunshine SBS output: published=1343 fg=1343 scene_flat=266 '
    'scene_fading=128 published_fresh_depth=336 published_reused_depth=1007 published_depth_missing=0 (unavailable=0 '
    'reuse_after_gap=0 reuse_other_source=0) runtime=0x2df57f5f5a0 generation=2 dropped=0 overwritten_unconsumed=898; '
    'cumulative Game 3D publications; scene_flat had depth but showed the colour frame; reused depth is bounded and '
    'does not advance calibration; unavailable reasons are in Sunshine depth readiness; dropped found no reusable '
    'export slot, overwritten_unconsumed replaced a ready slot the consumer never claimed',
    '13:47:14:632 [40860] | INFO  | [Sunshine 3D] Sunshine UI protection: runtime=000002df57f5f5a0 mode=auto '
    'rendered=1 mask_path=1 input=automatic_gpu_mask retained=1 fg=1 fg_known=1 fg_enabled=1 input_state=input_seen '
    'detection=detected selected=automatic source=automatic source_availability=detected sampled_source=8 '
    'sampled_covered=8294400 sampled_pixels=8294400 sampled_candidates=0x14 sampled_alpha_covered=0/0/8294400/0 '
    'sampled_alpha_invalid=0/0/0/0 accepted=0x14 sampled_layer={covered=0 invalid=0 opaque=0} '
    'sampled_one_way={strong=0/0 contradicted=0/0} sampled_reason=decided sampled_refused=none sampled_reused=0 '
    'sampled_hudless={changed=7874482 unchanged=144608 invalid=0 matching_tiles=0 lit=1190451} '
    'sampled_alpha_opaque=0/0 sampled_inferred_opaque=8294382/0 sampled_claims=0x84 sampled_h1={applied=1 winner=3} '
    'sampled_scene={n=0 d=0.000 valid=0 ran=0 verdict=none} sampled_pre_ui_scene={image=none n=0 d=0.000 valid=0} '
    'scene_guard={hidden=1 pre_ui=1 refuted=0 proven=0} sampled_pre_ui_pixels={match=0 image_lit=0} '
    'sampled_declared_one_way={strong=0/0 contradicted=0/0} status_revision=4',
    '13:47:15:502 [40860] | INFO  | [Sunshine 3D] Sunshine depth readiness: lost reason=retained_depth_expired '
    'runtime=0x2df57f5f5a0 native_swapchain=0x2df4a8b7bf0 present=21830 unavailable_ms=0 suppressed_episodes=0; '
    'provider=Streamline prior_provider=Streamline selected=0 snapshot_ready=0 copied=0 output_reused=0 '
    'display=not_attempted selection=inactive_views status=stale failure=none capture=994596 sequence=90117 '
    'newest_sequence=90117 epoch=4 source=9223372036854775809 viewport=1 selection_tick_ms=59583578 source_age_ms=250 '
    'newest_view_age_ms=250 active_views=0 finished=1 success=1 submitted=1 invalid=0 capture_pending=0 '
    'capture_repeated=0 fence=30597 completed=30598 completion_valid=1 producer_retired=1; FG_query=3 FG_known=1 '
    'FG_enabled=1 FG_mode=1 FG_generated=3 FG_epoch=4 FG_viewport=1 require_FG=1 policy_epoch=4; prior_output_ready=1 '
    'prior_retained=1 prior_age_ms=250 prior_present=21814 prior_capture=994596 prior_sequence=90117 prior_epoch=4 '
    'prior_source=9223372036854775809 prior_viewport=1 prior_observation=4 current_observation=0 prior_feedback=6 '
    'current_feedback=0 prior_reset=0 current_reset=0 retained_clear=retained_depth_expired; source_action=invalidate '
    'source_reason=retained_depth_expired cache_reason=retained_depth_expired consumer=not_attempted '
    'consumer_loss=none cookie=0 slot_invalid=0',
    '13:47:15:633 [40860] | INFO  | [Sunshine 3D] Sunshine UI protection: runtime=000002df57f5f5a0 mode=auto '
    'rendered=1 mask_path=0 input=none retained=0 fg=1 fg_known=1 fg_enabled=1 input_state=not_observed '
    'detection=searching selected=automatic source=automatic source_availability=source_unavailable sampled_source=0 '
    'sampled_covered=0 sampled_pixels=0 sampled_candidates=0x0 sampled_alpha_covered=0/0/0/0 '
    'sampled_alpha_invalid=0/0/0/0 accepted=0x0 sampled_layer={covered=0 invalid=0 opaque=0} '
    'sampled_one_way={strong=0/0 contradicted=0/0} sampled_reason=decided sampled_refused=none sampled_reused=0 '
    'sampled_hudless={changed=0 unchanged=0 invalid=0 matching_tiles=0 lit=0} sampled_alpha_opaque=0/0 '
    'sampled_inferred_opaque=0/0 sampled_claims=0x0 sampled_h1={applied=0 winner=0} sampled_scene={n=0 d=0.000 '
    'valid=0 ran=0 verdict=none} sampled_pre_ui_scene={image=none n=0 d=0.000 valid=0} scene_guard={hidden=0 pre_ui=0 '
    'refuted=0 proven=0} sampled_pre_ui_pixels={match=0 image_lit=0} sampled_declared_one_way={strong=0/0 '
    'contradicted=0/0} status_revision=5',
    '13:47:17:373 [40860] | INFO  | [Sunshine 3D] Sunshine depth readiness: recovered reason=fresh_copy '
    'runtime=0x2df57f5f5a0 native_swapchain=0x2df4a8b7bf0 present=21899 unavailable_ms=1859 suppressed_episodes=0; '
    'provider=Streamline prior_provider=Streamline selected=1 snapshot_ready=1 copied=1 output_reused=0 display=ready '
    'selection=current_capture status=ready failure=none capture=995532 sequence=90124 newest_sequence=90124 epoch=4 '
    'source=9223372036854775809 viewport=1 selection_tick_ms=59585437 source_age_ms=141 newest_view_age_ms=141 '
    'active_views=1 finished=1 success=1 submitted=1 invalid=0 capture_pending=0 capture_repeated=0 fence=30599 '
    'completed=30599 completion_valid=1 producer_retired=1; FG_query=3 FG_known=1 FG_enabled=1 FG_mode=1 '
    'FG_generated=3 FG_epoch=4 FG_viewport=1 require_FG=1 policy_epoch=4; prior_output_ready=0 prior_retained=0 '
    'prior_age_ms=18446744073709551615 prior_present=0 prior_capture=0 prior_sequence=0 prior_epoch=0 prior_source=0 '
    'prior_viewport=0 prior_observation=0 current_observation=4 prior_feedback=0 current_feedback=6 prior_reset=0 '
    'current_reset=0 retained_clear=none; source_action=copy_fresh source_reason=fresh_copy_candidate '
    'cache_reason=none consumer=ready consumer_loss=none cookie=995421 slot_invalid=0',
    '13:47:18:376 [40860] | INFO  | [Sunshine 3D] Sunshine UI protection: runtime=000002df57f5f5a0 mode=auto '
    'rendered=1 mask_path=1 input=automatic_gpu_mask retained=1 fg=1 fg_known=1 fg_enabled=1 input_state=input_seen '
    'detection=detected selected=automatic source=automatic source_availability=detected sampled_source=3 '
    'sampled_covered=232 sampled_pixels=8294400 sampled_candidates=0x34 sampled_alpha_covered=0/0/232/0 '
    'sampled_alpha_invalid=0/0/0/0 accepted=0x14 sampled_layer={covered=0 invalid=0 opaque=0} '
    'sampled_one_way={strong=232/0 contradicted=0/0} sampled_reason=decided sampled_refused=none sampled_reused=0 '
    'sampled_hudless={changed=245 unchanged=8294138 invalid=0 matching_tiles=256 lit=8137908} '
    'sampled_alpha_opaque=0/0 sampled_inferred_opaque=0/0 sampled_claims=0x0 sampled_h1={applied=0 winner=3} '
    'sampled_scene={n=0 d=0.000 valid=0 ran=0 verdict=none} sampled_pre_ui_scene={image=none n=0 d=0.000 valid=0} '
    'scene_guard={hidden=0 pre_ui=0 refuted=0 proven=0} sampled_pre_ui_pixels={match=0 image_lit=0} '
    'sampled_declared_one_way={strong=0/0 contradicted=0/0} status_revision=7',
    '13:47:19:407 [40860] | INFO  | [Sunshine 3D] Sunshine SBS output: published=1983 fg=1983 scene_flat=270 '
    'scene_fading=197 published_fresh_depth=476 published_reused_depth=1438 published_depth_missing=69 '
    '(unavailable=69 reuse_after_gap=0 reuse_other_source=0) runtime=0x2df57f5f5a0 generation=2 dropped=0 '
    'overwritten_unconsumed=1202; cumulative Game 3D publications; scene_flat had depth but showed the colour frame; '
    'reused depth is bounded and does not advance calibration; unavailable reasons are in Sunshine depth readiness; '
    'dropped found no reusable export slot, overwritten_unconsumed replaced a ready slot the consumer never claimed',
]
E33_DEPTH_LOSS.insert(-2, E33_DEPTH_LOSS[7].replace('13:47:15:633', '13:47:17:377')
                      .replace('status_revision=5', 'status_revision=6'))


# Real lines of E33 10-22 session B (14:35-14:37): the title screen into the first scene (NGX depth, no frame
# generation), the render-section hitch window, and the host's two upstream stalls.
E33B_TITLE = [
    '14:35:02:546 [44960] | INFO  | Registered add-on "Sunshine 3D" v0.0.0.0 using ReShade API version 20.',
    '14:35:09:015 [42172] | INFO  | [Sunshine 3D] Sunshine Game 3D: add-on GPU renderer ready (built off the present '
    'thread in 22 ms, longest step SunshineApplyUICS 4.5 ms; present-thread adoption 0.002 ms; 3840x2160, colour 1)',
    '14:35:09:246 [41592] | INFO  | [Sunshine 3D] Sunshine SBS: generation 1, 7680x2160 full SBS, DXGI 24, D3D12, '
    'sRGB, protocol 4 (source color 1)',
    '14:35:10:016 [41592] | INFO  | [Sunshine 3D] Sunshine 3D raw automation: calibrating; source=1; samples=0 '
    'stereo_scale=0 target_scale=0 t0=0 target_t0=0 reference_Q=0 normalization_L=7.67999971 reference_valid=0 '
    'plane_state=holding_stereo_controls; uncalibrated relative raw basis; depth_path=NGX feedback_revision=3',
    '14:35:14:233 [53312] | INFO  | [Sunshine 3D] Sunshine SBS output: published=1077 fg=0 scene_flat=1077 '
    'scene_fading=0 published_fresh_depth=964 published_reused_depth=113 published_depth_missing=0 (unavailable=0 '
    'reuse_after_gap=0 reuse_other_source=0) runtime=0x1ca461a32d0 generation=1 dropped=0 overwritten_unconsumed=644; '
    'cumulative Game 3D publications; scene_flat had depth but showed the colour frame; reused depth is bounded and '
    'does not advance calibration; unavailable reasons are in Sunshine depth readiness; dropped found no reusable '
    'export slot, overwritten_unconsumed replaced a ready slot the consumer never claimed',
    '14:35:19:234 [53312] | INFO  | [Sunshine 3D] Sunshine SBS output: published=2253 fg=0 scene_flat=2253 '
    'scene_fading=0 published_fresh_depth=2056 published_reused_depth=197 published_depth_missing=0 (unavailable=0 '
    'reuse_after_gap=0 reuse_other_source=0) runtime=0x1ca461a32d0 generation=1 dropped=0 overwritten_unconsumed=1370; '
    'cumulative Game 3D publications; scene_flat had depth but showed the colour frame; reused depth is bounded and '
    'does not advance calibration; unavailable reasons are in Sunshine depth readiness; dropped found no reusable '
    'export slot, overwritten_unconsumed replaced a ready slot the consumer never claimed',
    '14:35:24:236 [53312] | INFO  | [Sunshine 3D] Sunshine SBS output: published=3423 fg=0 scene_flat=3423 '
    'scene_fading=0 published_fresh_depth=3185 published_reused_depth=238 published_depth_missing=0 (unavailable=0 '
    'reuse_after_gap=0 reuse_other_source=0) runtime=0x1ca461a32d0 generation=1 dropped=0 overwritten_unconsumed=2089; '
    'cumulative Game 3D publications; scene_flat had depth but showed the colour frame; reused depth is bounded and '
    'does not advance calibration; unavailable reasons are in Sunshine depth readiness; dropped found no reusable '
    'export slot, overwritten_unconsumed replaced a ready slot the consumer never claimed',
    '14:35:29:234 [53312] | INFO  | [Sunshine 3D] Sunshine SBS output: published=4579 fg=0 scene_flat=4579 '
    'scene_fading=0 published_fresh_depth=4278 published_reused_depth=301 published_depth_missing=0 (unavailable=0 '
    'reuse_after_gap=0 reuse_other_source=0) runtime=0x1ca461a32d0 generation=1 dropped=0 overwritten_unconsumed=2796; '
    'cumulative Game 3D publications; scene_flat had depth but showed the colour frame; reused depth is bounded and '
    'does not advance calibration; unavailable reasons are in Sunshine depth readiness; dropped found no reusable '
    'export slot, overwritten_unconsumed replaced a ready slot the consumer never claimed',
    '14:35:30:766 [53312] | INFO  | [Sunshine 3D] Sunshine depth readiness: lost reason=retained_depth_expired '
    'runtime=0x1ca461a32d0 native_swapchain=0x1ca46818980 present=4919 unavailable_ms=0 suppressed_episodes=0; '
    'provider=NGX prior_provider=NGX selected=0 snapshot_ready=0 copied=0 output_reused=0 display=not_attempted '
    'selection=inactive_views status=stale failure=none capture=352847 sequence=4900 newest_sequence=4900 epoch=4 '
    'source=1 viewport=0 selection_tick_ms=62478828 source_age_ms=250 newest_view_age_ms=250 active_views=0 finished=1 '
    'success=1 submitted=1 invalid=0 capture_pending=0 capture_repeated=0 fence=4896 completed=4896 completion_valid=1 '
    'producer_retired=1; FG_query=3 FG_known=1 FG_enabled=0 FG_mode=0 FG_generated=1 FG_epoch=4 FG_viewport=1 '
    'require_FG=0 policy_epoch=0; prior_output_ready=1 prior_retained=1 prior_age_ms=250 prior_present=4901 '
    'prior_capture=352847 prior_sequence=4900 prior_epoch=4 prior_source=1 prior_viewport=0 prior_observation=0 '
    'current_observation=0 prior_feedback=3 current_feedback=0 prior_reset=0 current_reset=0 '
    'retained_clear=retained_depth_expired; source_action=invalidate source_reason=retained_depth_expired '
    'cache_reason=retained_depth_expired consumer=not_attempted consumer_loss=none cookie=0 slot_invalid=0',
    '14:35:30:768 [53312] | INFO  | [Sunshine 3D] Sunshine 3D raw automation: depth_unavailable; source=0; samples=0 '
    'stereo_scale=0 target_scale=0 t0=0 target_t0=0 reference_Q=0 normalization_L=7.67999971 reference_valid=0 '
    'plane_state=holding_stereo_controls; uncalibrated relative raw basis; depth_path=NGX feedback_revision=3',
    '14:35:31:598 [41592] | INFO  | [Sunshine 3D] Sunshine depth readiness: recovered reason=fresh_copy '
    'runtime=0x1ca461a32d0 native_swapchain=0x1ca46818980 present=5131 unavailable_ms=828 suppressed_episodes=0; '
    'provider=NGX prior_provider=NGX selected=1 snapshot_ready=1 copied=1 output_reused=0 display=ready '
    'selection=completed_snapshot status=ready failure=none capture=354848 sequence=4902 newest_sequence=4903 epoch=4 '
    'source=1 viewport=0 selection_tick_ms=62479656 source_age_ms=16 newest_view_age_ms=0 active_views=1 finished=1 '
    'success=1 submitted=1 invalid=0 capture_pending=0 capture_repeated=0 fence=4898 completed=4898 completion_valid=1 '
    'producer_retired=1; FG_query=3 FG_known=1 FG_enabled=0 FG_mode=0 FG_generated=1 FG_epoch=4 FG_viewport=1 '
    'require_FG=0 policy_epoch=0; prior_output_ready=0 prior_retained=0 prior_age_ms=18446744073709551615 '
    'prior_present=0 prior_capture=0 prior_sequence=0 prior_epoch=0 prior_source=0 prior_viewport=0 '
    'prior_observation=0 current_observation=0 prior_feedback=0 current_feedback=5 prior_reset=0 current_reset=1 '
    'retained_clear=none; source_action=copy_fresh source_reason=fresh_copy_candidate cache_reason=none consumer=ready '
    'consumer_loss=none cookie=354872 slot_invalid=0',
    '14:35:31:772 [41592] | INFO  | [Sunshine 3D] Sunshine 3D raw automation: calibrating; source=1; samples=1 '
    'stereo_scale=0 target_scale=0 t0=0 target_t0=0 reference_Q=0 normalization_L=7.67999971 reference_valid=0 '
    'plane_state=holding_stereo_controls; uncalibrated relative raw basis; depth_path=NGX feedback_revision=5',
    '14:35:32:770 [41592] | INFO  | [Sunshine 3D] Sunshine 3D raw automation: ready; source=1; samples=4 '
    'stereo_scale=1760.48376 target_scale=1736.10669 t0=0.00153324299 target_t0=0.0015611807 reference_Q=0.00442369096 '
    'normalization_L=7.67999971 reference_valid=1 plane_state=adjusting_stereo_controls; uncalibrated relative raw '
    'basis; depth_path=NGX feedback_revision=5',
    '14:35:34:238 [41592] | INFO  | [Sunshine 3D] Sunshine SBS output: published=5527 fg=0 scene_flat=5071 '
    'scene_fading=73 published_fresh_depth=4855 published_reused_depth=460 published_depth_missing=212 '
    '(unavailable=212 reuse_after_gap=0 reuse_other_source=0) runtime=0x1ca461a32d0 generation=1 dropped=0 '
    'overwritten_unconsumed=3306; cumulative Game 3D publications; scene_flat had depth but showed the colour frame; '
    'reused depth is bounded and does not advance calibration; unavailable reasons are in Sunshine depth readiness; '
    'dropped found no reusable export slot, overwritten_unconsumed replaced a ready slot the consumer never claimed',
]

E33B_RENDER_HITCH = [
    '14:36:23:771 [53312] | INFO  | [Sunshine 3D] Sunshine Streamline frame generation: viewport=1 mode=0 '
    'generated_frames=3 supported=1 success=1',
    '14:36:27:488 [41592] | INFO  | [Sunshine 3D] Sunshine Game 3D timing: presents=1031 cpu_ms={mean=0.205 '
    'max=47.306} cpu_worst_ms={setup=0.005 depth=0.055 ui=0.025 render=47.097 export=0.124} gpu_frames=0 gpu_ms '
    'mean/max={total=0.000/0.000 inputs=0.000/0.000 source=0.000/0.000 detection=0.000/0.000 linearize=0.000/0.000 '
    'candidate=0.000/0.000 vertical=0.000/0.000 horizontal=0.000/0.000 eyes=0.000/0.000 pack=0.000/0.000} '
    'gpu_profile=disabled dropped_fence_pending=0 dropped_unresolved=0 incomplete=0 ui_layer={copies=0 skipped=0 '
    'offers=0 presents_since_copy={mean=0.00 max=0} skip_gap_ms=0.0} window_ms=10000',
]

E33B_HOST_STALLS = [
    '[2026-10-22 14:35:09.251]: Info: ReShade SBS connected: process 46096, 7680x2160, sRGB SDR, protocol 4, '
    'generation 1.',
    '[2026-10-22 14:35:58.793]: Warning: NvEnc: frame 5931 exceeded the 100 ms completion wait; its input is still '
    "being produced, so the encoder's 250 ms budget starts when the input completes (2000 ms at most); "
    'input_producer=pending_at_99ms (upstream GPU delay)',
    '[2026-10-22 14:35:58.922]: Info: NvEnc: frame 5931 completed after soft timeout in 229 ms; '
    'input_producer=done_at_225ms (upstream GPU delay)',
    '[2026-10-22 14:36:30.622]: Warning: NvEnc: frame 8615 exceeded the 100 ms completion wait; its input is still '
    "being produced, so the encoder's 250 ms budget starts when the input completes (2000 ms at most); "
    'input_producer=pending_at_100ms (upstream GPU delay)',
    '[2026-10-22 14:36:30.863]: Info: NvEnc: frame 8615 completed after soft timeout in 341 ms; '
    'input_producer=done_at_341ms (upstream GPU delay)',
]


class ReadinessReport(unittest.TestCase):
    def test_clean_session_passes(self):
        checks = run(BASE + [ui('10:00:12', 2, 5, (0, 5, 0, 0), 0x2, 0x2), line('10:00:20', 'Finished exiting.')])
        self.assertTrue(all(c.status in ('PASS', 'INFO') for c in checks.values()), checks)

    def test_display_scaling_warns(self):
        # Witcher 3 at 150% scaling on a 3840x2160 virtual display (2026-10-04): the add-on's line, a WARN check.
        text = ('[Sunshine 3D] Sunshine Game 3D: display scaling limits the game: it renders at 2560x1440 on a '
                '3840x2160 display because Windows display scaling is 150%; set scaling to 100% or this '
                "game's high-DPI override to Application")
        scaled = line('10:00:03', text, 'WARN')
        checks = run(BASE + [scaled])
        self.assertEqual(checks['Display scaling'].status, 'WARN')
        self.assertIn('2560x1440 on a 3840x2160 display', checks['Display scaling'].detail)
        self.assertIn('150%', checks['Display scaling'].detail)
        self.assertNotIn('Log warnings', checks)
        self.assertNotIn('Display scaling', run(BASE))

    def test_presented_alpha_over_trusted_ui_channel_fails(self):
        # Resident Evil Requiem before 74039d17: current alpha (source 4) decided while
        # the trusted UI color channel was offered, flattening most of the scene.
        checks = run(BASE + [ui('10:00:12', 4, 400, (0, 2, 0, 400), 0xa, 0xa)])
        self.assertEqual(checks['UI protection'].status, 'FAIL')
        disputed = run(BASE + [ui('10:00:12', 2, 2, (0, 2, 0, 400), 0xa, 0xa)])
        self.assertEqual(disputed['UI protection'].status, 'WARN')
        # The same disagreement followed by the revocation it caused is handled.
        handled = run(BASE + [ui('10:00:12', 2, 2, (0, 2, 0, 400), 0xa, 0xa),
                              line('10:00:14', '[Sunshine 3D] Sunshine UI protection: alpha trust is now 0x2; '
                                               'remembered for later sessions of this game')])
        self.assertEqual(handled['UI protection'].status, 'PASS')

    def test_trusted_channel_flattening_the_visible_scene_fails(self):
        # Stellar Blade: the opaque tagged UI color inherited the UI layer's trust
        # and covered the frame while the exact HUD-less pair showed the scene.
        flat = ui('10:00:12', 2, 1000, (0, 1000, 1000, 1000), 0x36, 0x2, hudless=(400, 600))
        self.assertEqual(run(BASE + [flat])['UI protection'].status, 'FAIL')
        # A full-screen menu: the HUD-less pair differs almost everywhere.
        menu = ui('10:00:12', 2, 1000, (0, 1000, 1000, 1000), 0x36, 0x2, hudless=(950, 50))
        self.assertEqual(run(BASE + [menu])['UI protection'].status, 'PASS')
        # Revoking the UI layer's trust (source 4, not slot 1) soon after handles it.
        layer = ui('10:00:12', 2, 1000, (0, 1000, 1000, 1000), 0x36, 0x2, hudless=(400, 600), layer=1)
        revoke = line('10:00:14', '[Sunshine 3D] Sunshine UI protection: alpha trust is now 0x2; '
                                  'remembered for later sessions of this game')
        self.assertEqual(run(BASE + [layer, revoke])['UI protection'].status, 'PASS')
        self.assertEqual(run(BASE + [flat, revoke])['UI protection'].status, 'FAIL')

    def test_rejected_selective_ui_channel_warns(self):
        # Stellar Blade before the factor-two headroom: its real UI layer covered
        # 1.8% but pulsing markers made it invalid, so no mask protected the HUD.
        rejected = ui('10:00:12', 0, 0, (0, 18, 0, 1000), 0xa, 0x0, layer=1, invalid=(0, 15, 0, 0))
        checks = run(BASE + [rejected])
        self.assertEqual(checks['UI channel admission'].status, 'WARN')
        self.assertIn('UI layer 1.8% covered', checks['UI channel admission'].times[0])
        # A full-frame channel is ambiguous anyway, and another mask may decide.
        full = ui('10:00:12', 0, 0, (0, 1000, 0, 1000), 0xa, 0x0, invalid=(0, 5, 0, 0))
        decided = ui('10:00:12', 5, 30, (0, 18, 0, 1000), 0x1a, 0x0, invalid=(0, 5, 0, 0))
        self.assertEqual(run(BASE + [full, decided])['UI channel admission'].status, 'PASS')
        # Older logs without invalid counts skip the check.
        old = line('10:00:12', '[Sunshine 3D] Sunshine UI protection: runtime=0000000000000001 mode=auto '
                               'detection=no_usable_mask sampled_source=0 sampled_covered=0 sampled_pixels=1000 '
                               'sampled_candidates=0xa sampled_alpha_covered=0/18/0/1000 trusted_alpha=0x0 '
                               'sampled_hudless={changed=0 unchanged=0 invalid=0 matching_tiles=0 lit=0}')
        self.assertNotIn('UI channel admission', run(BASE + [old]))
        self.assertIn('UI protection', run(BASE + [old]))
        # Stellar Blade SDR: the cleared layer target held the scene image, colour without any alpha. That is no UI
        # layer, so admission rejected no UI; 'UI protection gaps' reports the unprotected time.
        colour_only = ui('10:00:12', 0, 0, (0, 0, 0, 1000), 0xa, 0x2, layer=1, invalid=(0, 600, 0, 0))
        self.assertEqual(run(BASE + [colour_only])['UI channel admission'].status, 'PASS')

    def test_set_aside_ui_layer_does_not_count_as_a_trusted_channel(self):
        # Stellar Blade SDR with FG on: the trusted layer had colour but no alpha, so the trusted Backbuffer decided.
        decided = ui('10:00:12', 3, 2, (0, 0, 2, 1000), 0x6, 0x6, layer=1, invalid=(0, 996, 0, 0), fg=1)
        self.assertEqual(run(BASE + [decided])['UI protection'].status, 'PASS')
        # A layer with alpha is a UI layer still: presented alpha deciding beside it fails.
        beside = ui('10:00:12', 3, 2, (0, 5, 2, 1000), 0x6, 0x6, layer=1, fg=1)
        self.assertEqual(run(BASE + [beside])['UI protection'].status, 'FAIL')
        # As the CPU's disagreement(), a UI channel rejected for more than 1% invalid pixels disputes nothing.
        ambiguous = ui('10:00:12', 0, 0, (0, 300, 900, 1000), 0x6, 0x6, layer=1, invalid=(0, 50, 0, 0))
        self.assertEqual(run(BASE + [ambiguous])['UI protection'].status, 'PASS')
        clean = ui('10:00:12', 2, 300, (0, 300, 900, 1000), 0x6, 0x6, layer=1)
        self.assertEqual(run(BASE + [clean])['UI protection'].status, 'WARN')

    def test_s1_lines_read_the_layer_and_accepted_candidates(self):
        # Since S1 the layer has its own candidate bit (0x40), counts and source (10), beside a tagged UI colour.
        layer = ui1('10:00:12', 10, 5, (0, 1000, 0, 0), 0x46, 0x40, layer=(5, 0))
        sample = report.parse(BASE + [layer]).ui[0]
        self.assertEqual((sample.source, sample.alpha, sample.accepted, sample.legacy),
                         (10, (0, 1000, 0, 0, 5), 0x40, False))
        self.assertTrue(all(c.status in ('PASS', 'INFO') for c in run(BASE + [layer]).values()))
        # A logged line before S1 with the layer in the UI colour slot reads the same.
        old = report.parse(BASE + [ui('10:00:12', 2, 5, (0, 5, 0, 0), 0x6, 0x2, layer=1)]).ui[0]
        self.assertEqual((old.source, old.candidates, old.alpha, old.accepted, old.legacy),
                         (10, 0x44, (0, 0, 0, 0, 5), 0x40, True))
        # Inferred alpha beside an accepted declared alpha fails (S1); beside an accepted layer it may decide.
        self.assertEqual(run(BASE + [ui1('10:00:12', 10, 5, (0, 2, 0, 0), 0x42, 0x42, layer=(5, 0))])
                         ['UI protection'].status, 'FAIL')
        self.assertEqual(run(BASE + [ui1('10:00:12', 3, 2, (0, 0, 2, 0), 0x44, 0x44, layer=(0, 0))])
                         ['UI protection'].status, 'PASS')
        # Stellar Blade SDR: the layer without alpha beside the Backbuffer, not yet accepted.
        sdr = ui1('10:00:06', 0, 0, (0, 0, 3, 0), 0x44, 0x0, layer=(0, 996), fg=1)
        self.assertEqual(run(BASE + [sdr, layer])['UI protection gaps'].times,
                         ['10:00:06-10:00:12 (6 s) FG on: UI layer has colour but no alpha (100% of pixels); '
                          'Backbuffer alpha covers 0.30% (not accepted)'])
        # An accepted tag that is invalid keeps the accepted Backbuffer out (S1, the declared-alpha block).
        blocked = ui1('10:00:06', 0, 0, (0, 0, 3, 0), 0x6, 0x6, invalid=(0, 50, 0, 0), fg=1)
        self.assertEqual(run(BASE + [blocked, layer])['UI protection gaps'].times,
                         ['10:00:06-10:00:12 (6 s) FG on: UI colour rejected (0% covered, 5.0% invalid); '
                          'Backbuffer alpha kept out beside an accepted UI channel'])

    def test_s1_acceptance_lines_resolve_disputes(self):
        # An accepted Backbuffer disagrees with the accepted tag; the next acceptance line drops it.
        dispute = ui1('10:00:12', 2, 2, (0, 2, 400, 0), 0x6, 0x6)
        checks = run(BASE + [accepted_now('10:00:10', 'backbuffer:24:srgb,ui_color:87:srgb'), dispute])
        self.assertEqual(checks['UI protection'].status, 'WARN')
        handled = run(BASE + [accepted_now('10:00:10', 'backbuffer:24:srgb,ui_color:87:srgb'), dispute,
                              accepted_now('10:00:14', 'ui_color:87:srgb')])
        self.assertEqual(handled['UI protection'].status, 'PASS')
        # Another signature of the same kind leaving is a revocation too; one joining is not.
        self.assertEqual(run(BASE + [accepted_now('10:00:10', 'backbuffer:24:pq,backbuffer:24:srgb,ui_color:87:srgb'),
                                     dispute, accepted_now('10:00:14', 'backbuffer:24:pq,ui_color:87:srgb')])
                         ['UI protection'].status, 'PASS')
        self.assertEqual(run(BASE + [accepted_now('10:00:10', 'backbuffer:24:srgb,ui_color:87:srgb'), dispute,
                                     accepted_now('10:00:14', 'backbuffer:24:pq,backbuffer:24:srgb,ui_color:87:srgb')])
                         ['UI protection'].status, 'WARN')
        discarded = line('10:00:01', '[Sunshine 3D] Sunshine UI protection: discarded 1 legacy UI trust entries '
                                     '(TrustedUISources=5); each source is accepted again by its own evidence')
        restored = line('10:00:01', '[Sunshine 3D] Sunshine UI protection: restored accepted UI sources '
                                    'ui_color:87:srgb from an earlier session of this game')
        trust = [c.detail for c in report.evaluate(report.parse(BASE + [discarded, restored])) if c.name == 'UI trust']
        self.assertEqual(trust, ['10:00:01 discarded legacy UI trust entries 1',
                                 '10:00:01 restored accepted UI sources ui_color:87:srgb'])

    def test_unprotected_streamed_time_warns_with_its_reason(self):
        protected = ui('10:00:12', 2, 5, (0, 5, 0, 0), 0x2, 0x2)
        # Stellar Blade SDR with FG off: the layer held the scene image and current alpha was opaque.
        colour_only = ui('10:00:06', 0, 0, (0, 0, 0, 1000), 0xa, 0x2, layer=1, invalid=(0, 600, 0, 0))
        checks = run(BASE + [colour_only, protected])
        self.assertEqual(checks['UI protection gaps'].status, 'WARN')
        self.assertEqual(checks['UI protection gaps'].times,
                         ['10:00:06-10:00:12 (6 s) FG off: UI layer has colour but no alpha (60% of pixels); '
                          'current alpha covers 100% (not accepted)'])
        # FG on: the trusted layer kept the selective Backbuffer alpha out.
        fg_on = ui('10:00:06', 0, 0, (0, 0, 3, 1000), 0x6, 0x6, layer=1, invalid=(0, 996, 0, 0), fg=1)
        self.assertEqual(run(BASE + [fg_on, protected])['UI protection gaps'].times,
                         ['10:00:06-10:00:12 (6 s) FG on: UI layer has colour but no alpha (100% of pixels); '
                          'Backbuffer alpha kept out beside an accepted UI channel'])
        # No candidate offered while rendering.
        searching = ui('10:00:06', 0, 0, (0, 0, 0, 0), 0x0, 0x0, pixels=0, detection='searching',
                       availability='source_unavailable')
        self.assertEqual(run(BASE + [searching, protected])['UI protection gaps'].times,
                         ['10:00:06-10:00:12 (6 s) FG off: no UI source offered'])
        # A rejected UI channel names its coverage and invalid share.
        rejected = ui('10:00:06', 0, 0, (0, 18, 0, 1000), 0xa, 0x0, layer=1, invalid=(0, 5, 0, 0))
        self.assertIn('UI layer rejected (1.8% covered, 0.50% invalid)',
                      run(BASE + [rejected, protected])['UI protection gaps'].times[0])

    def test_no_ui_on_screen_is_protected(self):
        protected = ui('10:00:12', 2, 5, (0, 5, 0, 0), 0x2, 0x2)
        # A clean empty UI channel says there is no UI, whether or not it is trusted.
        empty = ui('10:00:06', 0, 0, (0, 0, 0, 1000), 0xa, 0x0, layer=1)
        self.assertEqual(run(BASE + [empty, protected])['UI protection gaps'].status, 'PASS')
        # A trusted channel that decides with no coverage is a mask.
        decided = ui('10:00:06', 2, 0, (0, 0, 0, 1000), 0xa, 0x2, layer=1)
        self.assertEqual(run(BASE + [decided, protected])['UI protection gaps'].status, 'PASS')
        # A decided mask is protection even where source_availability names a missing source first.
        unavailable = ui('10:00:06', 2, 5, (0, 5, 0, 1000), 0xa, 0x2, layer=1, availability='source_unavailable')
        self.assertEqual(run(BASE + [unavailable, protected])['UI protection gaps'].status, 'PASS')

        # Older logs lack invalid counts: an untrusted empty channel showed no UI, but a trusted one at 0 would have
        # decided, so with no source it was rejected.
        def old(t, trusted):
            return line(t, '[Sunshine 3D] Sunshine UI protection: runtime=0000000000000001 mode=auto rendered=1 '
                           'detection=no_usable_mask source_availability=quality_rejected sampled_source=0 '
                           'sampled_covered=0 sampled_pixels=1000 sampled_candidates=0xa '
                           f'sampled_alpha_covered=0/0/0/1000 trusted_alpha=0x{trusted:x} '
                           'sampled_hudless={changed=0 unchanged=0 invalid=0 matching_tiles=0 lit=0}')
        self.assertEqual(run(BASE + [old('10:00:06', 0x0), protected])['UI protection gaps'].status, 'PASS')
        self.assertEqual(run(BASE + [old('10:00:06', 0x2), protected])['UI protection gaps'].status, 'WARN')

    def test_unprotected_time_counts_only_while_streamed_settled_and_long(self):
        protected = ui('10:00:12', 2, 5, (0, 5, 0, 0), 0x2, 0x2)
        bare = ui('10:00:06', 0, 0, (0, 0, 0, 1000), 0x8, 0x0)
        # The settle time after an FG switch is not counted.
        fg = line('10:00:06', '[Sunshine 3D] Sunshine Streamline frame generation: viewport=0 mode=1')
        self.assertEqual(run(BASE + [fg, bare, protected])['UI protection gaps'].times,
                         ['10:00:09-10:00:12 (3 s) FG off: current alpha covers 100% (not accepted)'])
        # The export pausing ends it.
        inactive = line('10:00:09', '[Sunshine 3D] Sunshine SBS: export inactive (not_foreground); waiting')
        self.assertEqual(run(BASE + [bare, inactive, protected])['UI protection gaps'].times[0][:23],
                         '10:00:06-10:00:09 (3 s)')
        # Manual modes, unrendered frames and gaps shorter than 2 s do not warn; short gaps are counted.
        for quiet in (ui('10:00:06', 0, 0, (0, 0, 0, 1000), 0x8, 0x0, mode='off'),
                      ui('10:00:06', 0, 0, (0, 0, 0, 1000), 0x8, 0x0, rendered=0)):
            self.assertEqual(run(BASE + [quiet, protected])['UI protection gaps'].detail,
                             'every streamed Auto frame had a UI mask or a UI channel showing no UI')
        brief = run(BASE + [bare, ui('10:00:07', 2, 5, (0, 5, 0, 0), 0x2, 0x2)])['UI protection gaps']
        self.assertEqual(brief.status, 'PASS')
        self.assertTrue(brief.detail.endswith('except 1 gap shorter than 2 s (1.0 s)'), brief.detail)
        # Each runtime's state holds until that runtime's next line.
        other = ui('10:00:08', 2, 5, (0, 5, 0, 0), 0x2, 0x2, runtime='0000000000000002')
        self.assertEqual(run(BASE + [bare, other, protected])['UI protection gaps'].times[0][:23],
                         '10:00:06-10:00:12 (6 s)')
        manual = ui('10:00:06', 0, 0, (0, 0, 0, 1000), 0x8, 0x0, mode='on')
        self.assertEqual(run(BASE + [manual])['UI protection gaps'].status, 'INFO')
        # A pending status sample neither starts nor ends a run, as the overlay's run (next_unprotected_since).
        checking = ui('10:00:07', 0, 0, (0, 0, 0, 0), 0x0, 0x0, pixels=0, detection='checking',
                      availability='checking_quality')
        self.assertEqual(run(BASE + [bare, checking, ui('10:00:08', 0, 0, (0, 0, 0, 1000), 0x8, 0x0), protected])
                         ['UI protection gaps'].times,
                         ['10:00:06-10:00:12 (6 s) FG off: current alpha covers 100% (not accepted)'])
        self.assertEqual(run(BASE + [checking, protected])['UI protection gaps'].detail,
                         'every streamed Auto frame had a UI mask or a UI channel showing no UI')

    def test_unprotected_time_without_depth_is_flat_not_a_gap(self):
        # E33 2026-10-22: no UI source from 13:47:15.633 to 18.376 was a 3 s WARN, but depth was lost until 17.373
        # and the output counters show its 69 Presents all went out without depth, so every frame then was flat. What
        # remains is the log gate's second after the recovery (its 4 Presents were flat too, at the strength ramp's
        # start, but no line shows that), a brief gap.
        gaps = run(E33_DEPTH_LOSS)['UI protection gaps']
        self.assertEqual((gaps.status, gaps.detail), (
            'PASS', 'every streamed Auto frame had a UI mask or a UI channel showing no UI except 1 gap shorter than '
                    '2 s (1.0 s); 1.7 s without a mask while depth was lost (every frame flat by the output counters) '
                    'not counted'))
        # The same UI lines without the depth loss, or without the counters that show it flat, are the 3 s gap.
        for kept in ('Sunshine SBS output', 'depth readiness'):
            shown = run([u for u in E33_DEPTH_LOSS if kept not in u])['UI protection gaps']
            self.assertEqual((shown.status, shown.times),
                             ('WARN', ['13:47:15-13:47:18 (3 s) FG on: no UI source offered']), kept)
        # Counters that show some of its Presents with depth (the generic capture can serve it) excuse nothing.
        served = [u.replace('published_depth_missing=69 (unavailable=69', 'published_depth_missing=68 (unavailable=68')
                  for u in E33_DEPTH_LOSS]
        self.assertEqual(run(served)['UI protection gaps'].status, 'WARN')

        # Time with depth still counts around a loss, and only the runtime that lost depth is excused.
        def readiness(t, kind, present, runtime='0x1', reason='retained_depth_expired'):
            return line(t, f'[Sunshine 3D] Sunshine depth readiness: {kind} reason={reason} runtime={runtime} '
                           f'present={present} unavailable_ms=0; provider=Streamline prior_present=1 '
                           f'selection=inactive_views')
        head = BASE[:-3]  # Without its output counter lines.
        before = output('10:00:05', 100, 0, 0, 100, 0)
        bare = ui('10:00:06', 0, 0, (0, 0, 0, 0), 0x0, 0x0, pixels=0, detection='searching',
                  availability='source_unavailable')
        protected = ui('10:00:12', 2, 5, (0, 5, 0, 0), 0x2, 0x2)
        flat = output('10:00:15', 700, 0, 0, 640, 60)  # The loss's 60 Presents went out without depth.
        loss = [readiness('10:00:10', 'lost', 1000), readiness('10:00:11', 'recovered', 1060)]
        gaps = run(head + [before, bare] + loss + [protected, flat])['UI protection gaps']
        self.assertEqual((gaps.status, gaps.times), ('WARN', ['10:00:06-10:00:10 (4 s) FG off: no UI source offered']))
        self.assertTrue(gaps.detail.endswith('; 1 gap shorter than 2 s (1.0 s) not listed; 1.0 s without a mask '
                                             'while depth was lost (every frame flat by the output counters) not '
                                             'counted'), gaps.detail)
        whole = ['10:00:06-10:00:12 (6 s) FG off: no UI source offered']
        other = run(head + [before, bare, readiness('10:00:07', 'lost', 1000, '0x2'),
                            readiness('10:00:11', 'recovered', 1060, '0x2'), protected, flat])['UI protection gaps']
        self.assertEqual((other.status, other.times), ('WARN', whole))
        # The loss counts as flat only when the counters show it: not when they count fewer Presents without depth
        # than it had (here the generic depth fallback served them), restart in between or do not follow it.
        handoff = [readiness('10:00:10', 'lost', 1000, reason='source_handoff'),
                   line('10:00:10', '[Sunshine 3D] Sunshine API depth: stale; capture=1 cause=none; using generic '
                                    'depth fallback'),
                   readiness('10:00:11', 'recovered', 1060)]
        for counters in ([output('10:00:15', 700, 0, 0, 700, 0)], [flat.replace('generation=1', 'generation=2')],
                         []):
            served = run(head + [before, bare] + handoff + [protected] + counters)['UI protection gaps']
            self.assertEqual((served.status, served.times, 'not counted' in served.detail), ('WARN', whole, False),
                             counters)
        # A loss that never recovers, or a loss line that names no runtime, excuses nothing, even with every later
        # publication without depth: after source_not_associated the generic capture can show depth unlogged.
        searching = ui('10:00:12', 0, 0, (0, 0, 0, 0), 0x0, 0x0, pixels=0, detection='searching',
                       availability='source_unavailable')
        unnamed = line('10:00:07', '[Sunshine 3D] Sunshine depth readiness: lost reason=no_retained_depth '
                                   'present=1000 unavailable_ms=0; provider=Streamline selection=inactive_views')
        for lost in (readiness('10:00:07', 'lost', 1000, reason='source_not_associated'), unnamed):
            held = run(head + [before, bare, lost, searching, output('10:00:15', 700, 0, 0, 0, 600)])
            self.assertEqual(held['UI protection gaps'].status, 'WARN', lost)
            self.assertNotIn('not counted', held['UI protection gaps'].detail)

    def test_hidden_scene_routes_and_uncovered_hidden_runs(self):
        # Stellar Blade's notice splash: an opaque untrusted UI layer over a scene its picture hides.
        splash = ui('10:00:12', 8, 1000, (0, 1000, 0, 1000), 0xa, 0x0, layer=1, scene=(-0.044, 'hidden', 1, 0, 0))
        checks = run(BASE + [splash])
        self.assertEqual(checks['UI protection'].status, 'PASS')
        self.assertEqual(checks['Hidden scene'].status, 'INFO')
        self.assertIn('layer route (8) in 1 samples', checks['Hidden scene'].detail)
        # The splash closes: the frame the visible verdict was read on is still 8, and that verdict releases the
        # hold at once. Each hidden scene that ends shows one such exit; they are counted, never judged.
        closing = ui('10:00:13', 8, 1000, (0, 1000, 0, 1000), 0xa, 0x0, layer=1, scene=(0.6, 'visible', 0, 0, 0))
        checks = run(BASE + [splash, closing])
        self.assertEqual((checks['UI protection'].status, checks['Hidden scene'].status), ('PASS', 'INFO'))
        self.assertIn('1 released by a visible verdict', checks['Hidden scene'].detail)
        # The verdict word decides, not D rounded to three places: 0.2496 prints as 0.250 but is ambiguous.
        rounded = ui('10:00:13', 9, 1000, (0, 0, 0, 1000), 0x18, 0x0, hudless=(950, 50),
                     scene=(0.2496, 'ambiguous', 2, 0, 0))
        checks = run(BASE + [rounded])
        self.assertEqual(checks['UI protection'].status, 'PASS')
        self.assertIn('0 released by a visible verdict', checks['Hidden scene'].detail)
        # The first-run shadow (builds before revision 9): the presented frame read hidden for 600 ms while no UI
        # source decided. The run is listed by its start, once, at its longest length.
        uncovered = ui('10:00:14', 0, 0, (0, 0, 0, 1000), 0x8, 0x0, scene=(0.01, 'hidden', 0, 1, 600))
        checks = run(BASE + [uncovered])
        self.assertEqual(checks['Hidden scene'].status, 'WARN')
        self.assertEqual(checks['Hidden scene'].times, ['10:00:13 600 ms (first-run shadow)'])
        longer = ui('10:00:15', 0, 0, (0, 0, 0, 1000), 0x8, 0x0, scene=(0.01, 'hidden', 0, 1, 1600))
        self.assertEqual(run(BASE + [uncovered, longer])['Hidden scene'].times, ['10:00:13 1600 ms (first-run shadow)'])
        brief = ui('10:00:14', 0, 0, (0, 0, 0, 1000), 0x8, 0x0, scene=(0.01, 'hidden', 0, 1, 400))
        self.assertEqual(run(BASE + [brief])['Hidden scene'].status, 'INFO')
        # A decided source covers the run; older logs have no hidden-scene check.
        decided = ui('10:00:14', 2, 20, (0, 20, 0, 1000), 0xa, 0x2, scene=(0.01, 'hidden', 0, 0, 900))
        self.assertEqual(run(BASE + [decided])['Hidden scene'].status, 'INFO')
        self.assertNotIn('Hidden scene', run(BASE + [ui('10:00:12', 2, 5, (0, 5, 0, 0), 0x2, 0x2)]))

    def test_raw_placement_with_a_valid_camera_warns(self):
        lines = [x for x in BASE if 'Streamline scale: ready' not in x] + [
            line('10:00:06', '[Sunshine 3D] Sunshine Streamline camera availability: constants_calls=9 cameras=1 '
                             'recent_valid_projection=1; NGX_frame_match=unproven'),
            line('10:00:07', '[Sunshine 3D] Sunshine 3D raw automation: ready; source=1; samples=4 stereo_scale=50 '
                             'target_scale=50'),
            line('10:00:17', '[Sunshine 3D] Sunshine 3D raw automation: ready; source=1; samples=4 stereo_scale=2000 '
                             'target_scale=2000')]
        checks = run(lines)
        self.assertEqual(checks['Placement'].status, 'WARN')
        self.assertIn('40x', checks['Placement'].detail)

    def test_missing_addon_fails_and_depth_gaps_are_named(self):
        self.assertEqual(run(BASE[:1])['Add-on'].status, 'FAIL')
        checks = run(BASE + [output('10:00:20', 1000, 0, 0, 750, 250)])
        self.assertEqual(checks['Depth'].status, 'WARN')
        self.assertEqual(checks['Depth'].times, ['10:00:15-10:00:20'])

    def test_hitches_at_resets_and_overlay_are_expected(self):
        hitch = '[Sunshine 3D] Sunshine Game 3D hitch: stereo export took 43.7 ms on the present thread'
        expected = run(BASE + [line('10:00:06', '[Sunshine 3D] Sunshine SBS: export inactive (runtime_reset); waiting'),
                               line('10:00:07', hitch, 'WARN'),
                               line('10:00:10', '[Sunshine 3D] Sunshine SBS: ReShade overlay opened; composing'),
                               line('10:00:10', hitch, 'WARN')])
        self.assertEqual(expected['Present hitches'].status, 'PASS')
        self.assertEqual(run(BASE + [line('10:00:12', hitch, 'WARN')])['Present hitches'].status, 'WARN')

    def test_export_pauses_are_judged_by_their_class(self):
        def pause(t, reason, kind=None):
            return line(t, f'[Sunshine 3D] Sunshine SBS: export inactive ({reason}' + (f', {kind}' if kind else '')
                        + '); waiting for a valid focused technique')
        # The host's and the game's own pauses are routine: a host that reconnects replaces the ring, an HDR toggle
        # changes the colour space, a host attaching between a render and its export changes the transfer.
        for reason in ('ring_replaced', 'color_space_changed', 'consumer_transfer_changed'):
            self.assertEqual(run(BASE + [pause('10:00:06', reason, 'routine')])['Export'].status, 'PASS', reason)
        # Lines before the class tag: the same reasons by the report's copy of the routine list, the rest faults.
        for reason in ('ring_replaced', 'color_space_changed', 'consumer_without_pq', 'swapchain_unknown'):
            self.assertEqual(run(BASE + [pause('10:00:06', reason)])['Export'].status, 'PASS', reason)
        old = run(BASE + [pause('10:00:06', 'generation_limit')])['Export']
        self.assertEqual((old.status, old.times), ('WARN', ['10:00:06 generation_limit']))
        # The add-on's class decides for a reason this report does not know.
        new = run(BASE + [pause('10:00:06', 'future_reason', 'fault')])['Export']
        self.assertEqual((new.status, new.detail), ('WARN', 'future_reason 1; faults: future_reason'))
        self.assertEqual(run(BASE + [pause('10:00:06', 'future_pause', 'routine')])['Export'].status, 'PASS')
        self.assertEqual(run(BASE + [pause('10:00:06', 'device_removed', 'fault')])['Export'].status, 'FAIL')
        # Routine pauses more often than once per 10 s for a minute are a storm: seven within 60 s warn, six do not,
        # and neither do seven spread over 66 s.
        times = ['10:00:20', '10:00:29', '10:00:38', '10:00:47', '10:00:56', '10:01:05', '10:01:14']
        storm = run(BASE + [pause(t, 'ring_replaced', 'routine') for t in times])['Export']
        self.assertEqual((storm.status, storm.times), ('WARN', ['10:00:20-10:01:14 7 routine pauses']))
        self.assertIn('more often than once per 10 s for a minute', storm.detail)
        six = run(BASE + [pause(t, 'not_foreground', 'routine') for t in times[:6]])['Export']
        self.assertEqual(six.status, 'PASS')
        spread = ['10:00:20', '10:00:31', '10:00:42', '10:00:53', '10:01:04', '10:01:15', '10:01:26']
        self.assertEqual(run(BASE + [pause(t, 'no_consumer') for t in spread])['Export'].status, 'PASS')

    def test_every_capture_status_but_the_settled_ones_fails(self):
        def status(t, name):
            return line(t, f'[Sunshine 3D] Sunshine Streamline depth: {name}; source_selected=0 provider_held=1')
        # A queue the capture cannot serve, a malformed tag and an exhausted table fail like the state failures, and so
        # do 'unknown' (an out-of-range status) and a status this report does not know; inside the settle time after
        # the export start each is INFO.
        for name in ('unsupported_queue', 'malformed', 'exhausted', 'unknown', 'future_status'):
            later = run(BASE + [status('10:00:12', name)])['Capture status']
            self.assertEqual((later.status, later.times), ('WARN', [f'10:00:12 Streamline {name}']), name)
            self.assertEqual(run(BASE + [status('10:00:03', name)])['Capture status'].status, 'INFO', name)
        for name in report.CAPTURE_OK:
            self.assertNotIn('Capture status', run(BASE + [status('10:00:12', name)]), name)

    def test_observation_losses_warn_by_origin_and_rate(self):
        def loss(t, revision, cause):
            return line(t, '[Sunshine 3D] Sunshine depth observation evidence: lost runtime=0x1; '
                           f'sampled_only={{revision={revision} found=1 cause={cause}}}')
        end = [line('10:20:00', '[Sunshine 3D] Sunshine SBS: still running')]  # A 20-minute session.
        # One capacity loss (a full bounded table) passes, also in a short session; five within a minute warn.
        self.assertEqual(run(BASE + [loss('10:00:10', 1, 'records_busy')])['Observation losses'].status, 'PASS')
        burst = run(BASE + [loss(f'10:00:{10 + 5 * i}', i + 1, 'records_busy') for i in range(5)] + end)
        self.assertEqual(burst['Observation losses'].status, 'WARN')
        self.assertIn('records_busy more than 3 times within 60 s', burst['Observation losses'].detail)
        # Three of one cause within a minute, or one of each capacity cause, stay below the rate of a long session.
        three = run(BASE + [loss(f'10:00:{10 + 5 * i}', i + 1, 'tokens_busy') for i in range(3)] + end)
        self.assertEqual(three['Observation losses'].status, 'PASS')
        each = run(BASE + [loss(f'10:0{i}:10', i + 1, c) for i, c in enumerate(report.LOSS_CAPACITY)] + end)
        self.assertEqual(each['Observation losses'].status, 'PASS')
        # More than one capacity loss per two minutes over the session warns.
        often = run(BASE + [loss(f'10:{i:02}:10', i + 1, 'commands_busy') for i in range(11)] + end)
        self.assertEqual(often['Observation losses'].status, 'WARN')
        self.assertIn('11 capacity losses over 20.0 min', often['Observation losses'].detail)
        # The game's own lifecycle and a reused token are routine; a fault and an unknown cause warn at any count.
        routine = run(BASE + [loss('10:00:10', 1, 'lifecycle'), loss('10:00:11', 2, 'token_replaced')])
        self.assertEqual((routine['Observation losses'].status, routine['Observation losses'].times), ('PASS', []))
        for cause in ('sdk_failure', 'invalid_input', 'future_cause'):
            fault = run(BASE + [loss('10:00:10', 1, 'records_busy'), loss('10:00:12', 2, cause)] + end)
            self.assertEqual((fault['Observation losses'].status, fault['Observation losses'].times),
                             ('WARN', [f'10:00:12 {cause}', '10:00:10 records_busy']), cause)

    def test_log_warnings_are_judged_by_origin(self):
        compile_error = line('10:00:06', "Failed to compile 'D:\\G\\reshade-shaders\\Shaders\\SMAA.fx':", 'ERROR')
        refcount = line('10:00:07', 'Reference count for ID3D12CommandQueue1 object 0000020B (0000020A) is '
                                    'inconsistent (3).', 'WARN')
        checks = run(BASE + [compile_error, refcount])
        self.assertNotIn('Log warnings', checks)
        other = checks['Other log warnings']
        self.assertEqual((other.status, len(other.times)), ('INFO', 2))
        self.assertTrue(other.times[0].startswith("10:00:06 ERROR Failed to compile 'D:"))
        # The add-on's own lines warn, a known phrase in them included; its hitches belong to Present hitches.
        owned = line('10:00:08', '[Sunshine 3D] Sunshine SBS: the depth state is inconsistent with its tag', 'WARN')
        self.assertEqual(run(BASE + [owned])['Log warnings'].status, 'WARN')
        hitch = line('10:00:08', '[Sunshine 3D] Sunshine Game 3D hitch: stereo export took 9.1 ms', 'WARN')
        self.assertNotIn('Log warnings', run(BASE + [hitch]))
        # ReShade's lines about the add-on: one naming it, a failed add-on load and a failed effect runtime.
        for text in ('Add-on "Sunshine 3D" was not unregistered!',
                     "Failed to load add-on from 'D:\\G\\Other.addon64' with error code 126!",
                     'Failed to create back buffer render targets!'):
            self.assertEqual(run(BASE + [line('10:00:09', text, 'ERROR')])['Log warnings'].status, 'WARN', text)

    def test_depth_flip_hitches_are_named(self):
        hitch = '[Sunshine 3D] Sunshine Game 3D hitch: generic challenger release took 21.5 ms on the present thread'
        checks = run(BASE + [line('10:00:12', hitch, 'WARN')])
        self.assertEqual(checks['Present hitches'].status, 'WARN')
        self.assertIn('generic challenger release 21.5 ms (depth-source flip step)', checks['Present hitches'].times[0])

    def test_diagnostics_off_reports_no_gpu_timing_without_a_warning(self):
        timing = ('[Sunshine 3D] Sunshine Game 3D timing: presents=600 cpu_ms={mean=0.400 max=1.200} '
                  'cpu_worst_ms={setup=0.1 depth=0.1 ui=0.1 render=0.1 export=0.1} gpu_frames=0 gpu_ms mean/max={'
                  'total=0.000/0.000 inputs=0.000/0.000} gpu_profile=disabled dropped_fence_pending=0 '
                  'dropped_unresolved=0 incomplete=0')
        off = run(BASE + [line('10:00:01', '[Sunshine 3D] Sunshine Game 3D: diagnostics off (Diagnostics=0)'),
                          line('10:00:12', timing)])
        self.assertEqual(off['Game 3D cost'].status, 'INFO')
        self.assertIn('GPU timing off (Diagnostics=0)', off['Game 3D cost'].detail)
        self.assertIn('Diagnostics=0', off['Diagnostics'].detail)
        self.assertNotIn('UI identity (S3)', off)
        on = run(BASE + [line('10:00:01', '[Sunshine 3D] Sunshine Game 3D: diagnostics on (Diagnostics=1)'),
                         line('10:00:12', timing.replace('gpu_profile=disabled', 'gpu_profile=ready'))])
        self.assertEqual(on['Game 3D cost'].status, 'WARN')
        self.assertIn('Diagnostics=1', on['Diagnostics'].detail)

    def test_blocked_captures_and_copyless_ngx_are_named(self):
        checks = run(BASE + [
            line('10:00:06', '[Sunshine 3D] Sunshine Streamline depth: incomplete_state; source_selected=1'),
            line('10:00:07', '[Sunshine 3D] Sunshine NGX depth: confirmed_features=1 capture_eligible=1 '
                             'recovered_features=0 evaluations=40 nominations=40 copy_recorded=0 metadata_only=0 '
                             'unknown_feature=0')])
        self.assertIn('blocked', checks['Capture status'].detail)
        self.assertEqual(checks['NGX depth'].status, 'WARN')
        # Evaluations while Streamline FG owned depth copy nothing by design.
        owned = run(BASE + [
            line('10:00:07', '[Sunshine 3D] Sunshine NGX depth: confirmed_features=1 capture_eligible=1 '
                             'recovered_features=0 evaluations=40 nominations=0 copy_recorded=0 metadata_only=0 '
                             'fg_owned=40 unknown_feature=0')])
        self.assertNotIn('NGX depth', owned)

    def test_dlssg_inputs_name_copies_and_refusals(self):
        def ngx(depth, ui, evaluations=300):
            return line('10:00:07', '[Sunshine 3D] Sunshine NGX depth: confirmed_features=2 capture_eligible=1 '
                                    'recovered_features=0 evaluations=40 nominations=40 copy_recorded=40 '
                                    'metadata_only=0 fg_owned=0 unknown_feature=0 missing_parameters=0 '
                                    f'missing_by_feature=none fg_feature={evaluations} dlssg_depth={{{depth}}} '
                                    f'dlssg_ui={{{ui}}} failed=0 feature_capacity_loss=0; D3D12 NGX')
        # Assassin's Creed Shadows if DLSS-G's list observes no state for its depth: a WARN naming the reason.
        refused = run(BASE + [ngx('copied=0 inner_frames=0 refused=missing_state:300',
                                  'offered=600 captured=0 refused=missing_state:600')])
        self.assertEqual(refused['DLSS-G inputs'].status, 'WARN')
        self.assertIn('missing_state 300', refused['DLSS-G inputs'].detail)
        self.assertIn('observed no state', refused['DLSS-G inputs'].detail)
        self.assertIn('captured 0 of 600', refused['DLSS-G inputs'].detail)
        # Accepted copies (MFG 3x: two inner frames per real frame skipped).
        copied = run(BASE + [ngx('copied=98 inner_frames=200 refused=missing_state:2',
                                 'offered=200 captured=196 refused=no_request:4')])
        self.assertEqual(copied['DLSS-G inputs'].status, 'INFO')
        self.assertIn('supplied 98 FG-scope depth copies in 100 real-frame', copied['DLSS-G inputs'].detail)
        self.assertIn('captured 196 of 200 (refused no_request 4)', copied['DLSS-G inputs'].detail)
        # A Streamline FG tag that copies keeps precedence: no warning.
        tagged = run(BASE + [ngx('copied=0 inner_frames=0 refused=tag_copied:300',
                                 'offered=0 captured=0 refused=none')])
        self.assertEqual(tagged['DLSS-G inputs'].status, 'INFO')
        self.assertIn('Streamline FG tags supplied the depth; no UI input offered', tagged['DLSS-G inputs'].detail)
        # Per kind, Streamline UI tags that copy keep precedence too: not a refusal of the inputs offered.
        ui_tagged = run(BASE + [ngx('copied=0 inner_frames=0 refused=tag_copied:300',
                                    'offered=0 captured=0 refused=tag_copied:600')])
        self.assertEqual(ui_tagged['DLSS-G inputs'].status, 'INFO')
        self.assertIn('Streamline FG tags supplied the depth; Streamline UI tags copied 600 of its UI inputs first',
                      ui_tagged['DLSS-G inputs'].detail)
        self.assertNotIn('no UI input offered', ui_tagged['DLSS-G inputs'].detail)
        ui_partly = run(BASE + [ngx('copied=100 inner_frames=0 refused=none',
                                    'offered=100 captured=100 refused=tag_copied:100')])
        self.assertIn('UI inputs captured 100 of 100; Streamline UI tags copied 100 of its UI inputs first',
                      ui_partly['DLSS-G inputs'].detail)
        # Mostly tag copies with a few refused evaluations between them stays INFO; mostly refusals warns.
        mixed = run(BASE + [ngx('copied=0 inner_frames=0 refused=tag_copied:290,missing_state:10',
                                'offered=0 captured=0 refused=none')])
        self.assertEqual(mixed['DLSS-G inputs'].status, 'INFO')
        self.assertIn('(refused missing_state 10)', mixed['DLSS-G inputs'].detail)
        mostly_refused = run(BASE + [ngx('copied=0 inner_frames=0 refused=tag_copied:10,observer_coverage:290',
                                         'offered=0 captured=0 refused=none')])
        self.assertEqual(mostly_refused['DLSS-G inputs'].status, 'WARN')
        self.assertIn('does not cover', mostly_refused['DLSS-G inputs'].detail)
        # No DLSS-G evaluation, or an add-on before 10-23: no check.
        self.assertNotIn('DLSS-G inputs', run(BASE + [ngx('copied=0 inner_frames=0 refused=none',
                                                          'offered=0 captured=0 refused=none', 0)]))
        self.assertNotIn('DLSS-G inputs', run(BASE + [
            line('10:00:07', '[Sunshine 3D] Sunshine NGX depth: confirmed_features=1 capture_eligible=1 '
                             'recovered_features=0 evaluations=40 nominations=40 copy_recorded=40 metadata_only=0 '
                             'fg_owned=0 unknown_feature=0 missing_parameters=0 missing_by_feature=none '
                             'fg_feature=12 failed=0')]))

    def test_fg_switch_settles_before_flat_counts(self):
        checks = run(BASE + [line('10:00:16', '[Sunshine 3D] Sunshine Streamline frame generation: viewport=0 mode=1'),
                             output('10:00:18', 900, 200, 150, 900, 0)])
        self.assertEqual(checks['Placement flat'].status, 'PASS')

    def test_depth_gaps_name_the_addons_own_loss_episodes(self):
        # Stellar Blade 2026-10-02: the counter interval read 13:27:29-13:27:34; the add-on lost depth at
        # 32.988 because the game supplied no depth view and recovered 1859 ms later.
        lost = line('10:00:17', '[Sunshine 3D] Sunshine depth readiness: lost reason=retained_depth_expired '
                                'runtime=0x1 present=10 unavailable_ms=0 suppressed_episodes=0; provider=NGX '
                                'prior_provider=NGX selected=0 display=not_attempted selection=inactive_views '
                                'status=stale')
        recovered = line('10:00:18', '[Sunshine 3D] Sunshine depth readiness: recovered reason=fresh_copy runtime=0x1 '
                                     'present=20 unavailable_ms=1859 suppressed_episodes=0; provider=NGX '
                                     'prior_provider=NGX selected=1 display=ready selection=completed_snapshot')
        checks = run(BASE + [lost, recovered, output('10:00:20', 1000, 0, 0, 750, 250)])
        self.assertEqual(checks['Depth'].status, 'WARN')
        self.assertEqual(checks['Depth'].times, [
            '10:00:17.000-10:00:18.000 (1859 ms) retained_depth_expired, NGX selection=inactive_views '
            '(the game supplied no depth)'])
        self.assertEqual(checks['Depth losses'].detail, 'retained_depth_expired 1')
        # A loss inside a clean interval is counted but names no window; one never recovered says so.
        quiet = run(BASE + [lost, recovered])
        self.assertEqual((quiet['Depth'].status, quiet['Depth'].times), ('PASS', []))
        open_loss = run(BASE + [lost, output('10:00:20', 1000, 0, 0, 750, 250)])
        self.assertTrue(open_loss['Depth'].times[0].endswith('until the log ended'))

    def test_dynamic_resolution_is_no_depth_gap(self):
        # Assassin's Creed Shadows with FG on (10-23): each SR depth render-size change was a 125-438 ms
        # depth_shape_changed loss. A render-size change of one source now keeps depth: the add-on counts the
        # changes on its render size line, and the report lists them as information, never as a gap or a loss.
        size = line('10:00:17', '[Sunshine 3D] Sunshine depth render size: changed=6 held=5 total=6 '
                                'active=2206x1240..2442x1374 now allocation=2442x1374 active=0,0,2442,1374; '
                                'provider=Streamline epoch=2 source=0 viewport=0; dynamic resolution of one source: '
                                'each copy keeps its own size, a new size is used at once and a pending one holds the '
                                'previous copy, so depth stays ready')
        later = size.replace('10:00:17', '10:00:22').replace('changed=6 held=5 total=6', 'changed=2 held=0 total=8')
        later = later.replace('active=2206x1240..2442x1374', 'active=2206x1240..2658x1494')
        checks = run(BASE + [size, output('10:00:20', 1000, 500, 0, 1000, 0), later])
        self.assertEqual((checks['Depth'].status, checks['Depth'].times), ('PASS', []))
        self.assertNotIn('Depth losses', checks)
        self.assertEqual(checks['Depth render size'].status, 'INFO')
        self.assertEqual(checks['Depth render size'].detail,
                         '8 render-size changes of one Streamline depth source (dynamic resolution, active 2206x1240 '
                         'to 2658x1494) kept depth ready; 5 Presents held the previous size while the next was '
                         'pending')
        self.assertEqual(checks['Depth render size'].times, ['10:00:17', '10:00:22'])
        self.assertNotIn('Depth render size', run(BASE))

    def test_a_depth_loss_logged_just_before_its_fg_switch_belongs_to_the_switch(self):
        # Stellar Blade 2026-10-22: turning FG on handed depth from NGX to Streamline. The loss was logged 42 ms
        # before the switch's own line (the add-on logs asynchronously) and recovered 158 ms later.
        def readiness(t, kind, selection):
            return (f'{t} [ 1234] | INFO  | [Sunshine 3D] Sunshine depth readiness: {kind} reason='
                    f'{"retained_depth_expired" if kind == "lost" else "fresh_copy"} runtime=0x1 present=10 '
                    f'unavailable_ms=0 suppressed_episodes=0; provider=NGX prior_provider=NGX selection={selection}')
        lost = readiness('10:00:17:358', 'lost', 'no_completed_snapshot')
        recovered = readiness('10:00:17:516', 'recovered', 'completed_snapshot')
        fg = ('10:00:17:400 [ 1234] | INFO  | [Sunshine 3D] Sunshine Streamline frame generation: viewport=1 mode=1 '
              'generated_frames=2 supported=1 success=1')
        tail = [output('10:00:25', 1000, 0, 0, 750, 250)]
        checks = run(BASE + [lost, fg, recovered] + tail)
        self.assertEqual((checks['Depth'].status, checks['Depth'].times), ('PASS', []))
        # Without the switch the same loss is a gap.
        self.assertEqual(run(BASE + [lost, recovered] + tail)['Depth'].times, [
            '10:00:17.358-10:00:17.516 (0 ms) retained_depth_expired, NGX selection=no_completed_snapshot'])
        # A loss at the switch that outlasts its settle time, or one logged well before the switch, is still a gap.
        outlasting = run(BASE + [lost, fg, readiness('10:00:21:000', 'recovered', 'completed_snapshot')] + tail)
        self.assertEqual(outlasting['Depth'].status, 'WARN')
        early = run(BASE + [readiness('10:00:17:000', 'lost', 'no_completed_snapshot'), fg, recovered] + tail)
        self.assertEqual(early['Depth'].status, 'WARN')

    def test_a_depth_gap_names_the_games_own_freeze_inside_it(self):
        # E33 2026-10-22 15:43:35: the game stopped presenting for 773 ms while it streamed in assets and supplied no
        # depth for 1.9 s around it. The gap still warns, and names the freeze; a Present gap before the loss does not
        # belong to it.
        def readiness(t, kind):
            return (f'{t} [ 1234] | INFO  | [Sunshine 3D] Sunshine depth readiness: {kind} reason='
                    f'{"retained_depth_expired" if kind == "lost" else "fresh_copy"} runtime=0x1 present=10 '
                    f'unavailable_ms={0 if kind == "lost" else 1907} suppressed_episodes=0; provider=Streamline '
                    f'prior_provider=Streamline selection={"inactive_views" if kind == "lost" else "current_capture"}')

        def gap(t, ms):
            return (f'{t} [ 1234] | INFO  | [Sunshine 3D] Sunshine Game 3D present gap: {ms} ms since the previous '
                    f'Present; fg=1 foreground=1 suppressed=0 vram_local_mib=8698/12084 nonlocal_mib=460')
        lines = [gap('10:00:14:776', 243), readiness('10:00:15:784', 'lost'), gap('10:00:17:694', 773),
                 readiness('10:00:17:694', 'recovered'), output('10:00:20', 1000, 0, 0, 750, 250)]
        depth = run(BASE + lines)['Depth']
        self.assertEqual((depth.status, depth.times), ('WARN', [
            '10:00:15.784-10:00:17.694 (1907 ms) retained_depth_expired, Streamline selection=inactive_views '
            '(the game supplied no depth); the game froze 773 ms']))
        two = run(BASE + lines[:2] + [gap('10:00:16:500', 300)] + lines[2:])['Depth']
        self.assertTrue(two.times[0].endswith('; the game froze 773 ms (2 Present gaps)'))
        self.assertFalse(run(BASE + lines[1:2] + lines[3:])['Depth'].times[0].endswith('ms'))

    def test_an_observation_loss_is_dated_by_its_own_tick(self):
        # RE Requiem 2026-10-22: a metadata write lost at finish_evaluation (records_busy) was sampled 7.8 s later
        # with the FG switch's depth loss; the report names the loss's own time and site.
        sampled = line('10:00:17', '[Sunshine 3D] Sunshine depth observation evidence: lost runtime=0x1 '
                                   'loss_present=1970 loss_tick_ms=63342234 sampled_tick_ms=63342234 '
                                   'origin=episode_start; sampled_only={revision=4 found=1 cause=records_busy '
                                   'site=finish_evaluation line=1176 tick_ms=63334468 thread=9284 sequence=13651 '
                                   'viewport=0 feature=0 sdk_known=1 sdk_result=0 reset=4294967295}')
        session = report.parse(BASE + [sampled])
        at, cause, site = session.losses[4]
        self.assertAlmostEqual(at, report.seconds('10', '00', '17', '000') - 7.766)
        self.assertEqual((cause, site), ('records_busy', 'finish_evaluation'))
        checks = run(BASE + [sampled])
        self.assertEqual((checks['Observation losses'].status, checks['Observation losses'].times),
                         ('PASS', ['10:00:09 records_busy at finish_evaluation']))
        # An older line without the site or tick keeps the line's own time.
        old = line('10:00:17', '[Sunshine 3D] Sunshine depth observation evidence: lost runtime=0x1; '
                               'sampled_only={revision=5 found=1 cause=tokens_busy}')
        self.assertEqual(report.parse(BASE + [old]).losses[5],
                         (report.seconds('10', '00', '17', '000'), 'tokens_busy', None))

    def test_streamline_out_of_vram_warnings_are_named(self):
        # Battlefield 6 2026-10-22 (bf6_sp_1022c): over its video-memory budget, Streamline answered every DLSS
        # evaluation with eWarnOutOfVRAM (39) and the add-on of that day lost the observation on each one.
        lost = line('10:00:17', '[Sunshine 3D] Sunshine depth observation evidence: lost runtime=0x2424dd60e0 '
                                'loss_present=23268 loss_tick_ms=87051296 sampled_tick_ms=87051296 '
                                'origin=episode_start; sampled_only={revision=94 found=1 cause=sdk_failure '
                                'site=finish_evaluation line=1171 tick_ms=87051296 thread=19736 sequence=349634 '
                                'viewport=0 feature=0 sdk_known=1 sdk_result=39 reset=0}')
        self.assertEqual(report.parse(BASE + [lost]).loss_sdk, {94: 39})
        checks = run(BASE + [lost])
        self.assertEqual(checks['Observation losses'].times,
                         ['10:00:17 sdk_failure at finish_evaluation sdk_result=39 (eWarnOutOfVRAM)'])
        self.assertEqual(checks['Streamline VRAM warnings'].status, 'WARN')
        # Another SDK error is shown but is not the warning.
        error = lost.replace('sdk_result=39', 'sdk_result=20')
        checks = run(BASE + [error])
        self.assertEqual(checks['Observation losses'].times,
                         ['10:00:17 sdk_failure at finish_evaluation sdk_result=20'])
        self.assertNotIn('Streamline VRAM warnings', checks)
        # Since 10-22 the add-on accepts the warning and counts it on the camera availability line.
        counted = line('10:00:12', '[Sunshine 3D] Sunshine Streamline camera availability: constants_calls=18608 '
                                   'cameras=1 recent_valid_projection=1 vram_warnings=4211; NGX_frame_match=unproven; '
                                   'availability does not authorize cross-API scale')
        checks = run(BASE + [counted])
        self.assertEqual(checks['Streamline VRAM warnings'].status, 'INFO')
        self.assertIn('4211 call(s)', checks['Streamline VRAM warnings'].detail)
        none = counted.replace('vram_warnings=4211', 'vram_warnings=0')
        self.assertNotIn('Streamline VRAM warnings', run(BASE + [none]))
        self.assertNotIn('Streamline VRAM warnings', run(BASE))

    def test_nothing_counts_while_the_export_is_inactive(self):
        # The game in the background: no output is streamed from 'export inactive' to the next generation,
        # so its depth losses are not losses the viewer saw.
        inactive = line('10:00:16', '[Sunshine 3D] Sunshine SBS: export inactive (not_foreground); waiting')
        lost = line('10:00:17', '[Sunshine 3D] Sunshine depth readiness: lost reason=no_retained_depth runtime=0x1 '
                                'unavailable_ms=0; provider=Streamline prior_provider=Streamline '
                                'selection=fg_scope_mismatch')
        resumed = line('10:00:19', '[Sunshine 3D] Sunshine SBS: generation 2, 7680x2160 full SBS, DXGI 10, D3D12, '
                                   'scRGB (source color 3)')
        checks = run(BASE + [inactive, lost, lost, resumed, line('10:00:20', 'Finished exiting.')])
        self.assertNotIn('Depth losses', checks)
        session = report.parse(BASE + [inactive, lost, resumed, lost])
        self.assertEqual(session.readiness['no_retained_depth'], 1)
        # A loss open when the export pauses ends there.
        paused = report.parse(BASE + [lost, inactive])
        self.assertEqual((paused.depth_episodes[0].end, paused.depth_episodes[0].ending),
                         (report.seconds('10', '00', '16', '000'), 'paused'))

    def test_the_export_stops_own_invalidation_is_the_pause(self):
        # Assassin's Creed Shadows 10-23: add-ons before then logged the export stop's one reuse invalidation in the
        # millisecond of its pause line, just before it, and the report listed it as a depth gap 'until the export
        # paused'. Since then it follows the pause line, which already excludes it.
        def lost(t, reason):
            return line(t, f'[Sunshine 3D] Sunshine depth readiness: lost reason={reason} runtime=0x1 '
                           'unavailable_ms=0; provider=unknown prior_provider=Streamline selection=not_attempted')
        inactive = line('10:00:16', '[Sunshine 3D] Sunshine SBS: export inactive (not_foreground, routine); waiting')
        session = report.parse(BASE + [lost('10:00:16', 'explicit_reuse_invalidation'), inactive])
        self.assertEqual(session.depth_episodes, [])
        self.assertNotIn('explicit_reuse_invalidation', session.readiness)
        # An earlier invalidation, or another loss in the pause's millisecond, still ends at the pause.
        for loss in (lost('10:00:15', 'explicit_reuse_invalidation'), lost('10:00:16', 'no_retained_depth')):
            session = report.parse(BASE + [loss, inactive])
            self.assertEqual([(e.end, e.ending) for e in session.depth_episodes],
                             [(report.seconds('10', '00', '16', '000'), 'paused')])

    def test_depth_episodes_pair_within_each_runtime(self):
        # Each runtime traces its own loss and recovery, so another runtime's recovery cannot close it, and a
        # session crossing midnight keeps its order.
        def readiness(t, kind, runtime):
            return line(t, f'[Sunshine 3D] Sunshine depth readiness: {kind} reason=x runtime={runtime} '
                           f'unavailable_ms=0; provider=NGX selection=inactive_views')
        start = [line('23:59:58', '[Sunshine 3D] Sunshine SBS: generation 1, 7680x2160 full SBS, DXGI 10, D3D12, '
                                  'scRGB (source color 3)')]
        paused = line('00:00:02', '[Sunshine 3D] Sunshine SBS: export inactive (not_foreground)')
        session = report.parse(start + [readiness('23:59:59', 'lost', '0x1'), readiness('00:00:00', 'lost', '0x2'),
                                        readiness('00:00:01', 'recovered', '0x1'), paused])
        self.assertEqual([(report.clock(e.start, True), report.clock(e.end, True), e.ending)
                          for e in session.depth_episodes],
                         [('23:59:59.000', '00:00:01.000', 'recovered'), ('00:00:00.000', '00:00:02.000', 'paused')])

    def test_short_calibration_after_a_provider_switch_is_info(self):
        # Stellar Blade 2026-10-02: FG on at 13:27:37 covered too little of the 5 s interval for the settle
        # rule, but its projection controller calibrated within a second of the switch to Streamline depth.
        def readiness(t, kind, provider, prior):
            return line(t, f'[Sunshine 3D] Sunshine depth readiness: {kind} reason=x runtime=0x1 unavailable_ms=0; '
                           f'provider={provider} prior_provider={prior} selection=completed_snapshot')
        switch = [
            readiness('10:00:05', 'lost', 'NGX', 'NGX'), readiness('10:00:05', 'recovered', 'NGX', 'NGX'),
            line('10:00:06', '[Sunshine 3D] Sunshine 3D raw automation: ready; source=3; samples=4 stereo_scale=5'),
            readiness('10:00:16', 'lost', 'Streamline', 'NGX'),
            line('10:00:16', '[Sunshine 3D] Sunshine 3D raw automation: depth_unavailable; source=0; samples=0'),
            readiness('10:00:16', 'recovered', 'Streamline', 'Streamline'),
            line('10:00:16', '[Sunshine 3D] Sunshine 3D Streamline scale: waiting_for_depth; viewport=1; retaining'),
            line('10:00:17', '[Sunshine 3D] Sunshine 3D Streamline scale: ready; viewport=1 encoding=device'),
            output('10:00:20', 1000, 0, 150, 1000, 0)]
        checks = run(BASE + switch)
        self.assertEqual(checks['Placement flat'].status, 'PASS')
        self.assertEqual(checks['Placement calibration'].status, 'INFO')
        self.assertEqual(checks['Placement calibration'].times,
                         ['10:00:16.000-10:00:17.000 after provider switch (NGX to Streamline)'])
        # A calibration that had depth for longer than the settle time is a fault, and names its state.
        slow = [x.replace('10:00:17', '10:00:20') if 'scale: ready' in x else x for x in switch]
        checks = run(BASE + slow)
        self.assertEqual(checks['Placement flat'].status, 'WARN')
        self.assertNotIn('Placement calibration', checks)
        self.assertEqual(checks['Placement flat'].times,
                         ['10:00:15-10:00:20, projection waiting_for_depth 10:00:16-10:00:20 (4.0 s with depth)'])
        # Depth time adds up across losses: two 2 s attempts are 4 s of flat output with depth. A loss that
        # began after the run did not start it.
        flicker = switch[:3] + [
            line('10:00:16', '[Sunshine 3D] Sunshine 3D raw automation: calibrating; source=3; samples=0'),
            readiness('10:00:18', 'lost', 'NGX', 'NGX'), readiness('10:00:19', 'recovered', 'NGX', 'NGX'),
            output('10:00:20', 1000, 0, 150, 1000, 0),
            line('10:00:21', '[Sunshine 3D] Sunshine 3D raw automation: ready; source=3; samples=4 stereo_scale=5')]
        self.assertEqual(run(BASE + flicker)['Placement flat'].times,
                         ['10:00:15-10:00:20, raw calibrating 10:00:16-10:00:21 (4.0 s with depth, no switch, '
                          'depth loss or export start before it)'])
        # A short run that nothing in the log started is not a designed calibration: placement fell back
        # on its own, so it warns however briefly it lasted.
        unexplained = switch[:3] + [
            line('10:00:18', '[Sunshine 3D] Sunshine 3D raw automation: calibrating; source=3; samples=0'),
            line('10:00:19', '[Sunshine 3D] Sunshine 3D raw automation: ready; source=3; samples=4 stereo_scale=5'),
            output('10:00:20', 1000, 0, 150, 1000, 0)]
        checks = run(BASE + unexplained)
        self.assertNotIn('Placement calibration', checks)
        self.assertEqual(checks['Placement flat'].times,
                         ['10:00:15-10:00:20, raw calibrating 10:00:18-10:00:19 (1.0 s with depth, no switch, '
                          'depth loss or export start before it)'])
        # The same run is designed when an FG switch began it, up to the add-on's one-second log gate earlier.
        fg = line('10:00:18', '[Sunshine 3D] Sunshine Streamline frame generation: viewport=1 mode=1 '
                              'generated_frames=2 supported=1 success=1')
        checks = run(BASE + unexplained[:3] + [fg] + unexplained[3:])
        self.assertEqual(checks['Placement calibration'].times,
                         ['10:00:18.000-10:00:19.000 after frame generation switch'])
        # Flat output while placement claimed ready has no calibration in progress.
        placed = run(BASE + switch[:3] + [output('10:00:20', 1000, 0, 150, 1000, 0)])
        self.assertEqual((placed['Placement flat'].status, placed['Placement flat'].times),
                         ('WARN', ['10:00:15-10:00:20']))

    def test_counter_line_is_parsed_and_the_last_one_counts(self):
        # An S2a counter line, as format_ui_counters wrote it before S2b.
        text = ('auto_frames=10 detection_frames=6 held={generated=2 none=1} reused=1 '
                'inactive={no_candidates=1 size=0 unprepared=0} decided={0=2 1=0 2=0 3=0 4=0 5=3 6=1 8=0 9=0 10=4} '
                'none={layer_aside=0 trusted_invalid=0 presented_blocked=0 ambiguous=0 difference_failed=1 '
                'gate_no_hold=1 no_candidate=0 other=0 unaccepted=2} full={6=1 8=0 9=0 depth_not_current=1} '
                'full_d={hidden=1 ambiguous=0 visible=0 invalid=0} untrusted_inferred=0 inexact_difference=3 '
                'contradicted=2 presented_over_dedicated=0 full_alpha=2 '
                'full_alpha_d={hidden=0 ambiguous=0 visible=1 invalid=0} trust={earned=1 revoked_exact=1 '
                'revoked_declared=0 lapsed=0 restored=0 discarded=2 forgotten=3} samples=4 through_ms=12345')
        fields = report.counter_fields('runtime=0000000000000001 ' + text)
        self.assertEqual((fields['auto_frames'], fields['held.none'], fields['reused'], fields['decided.10'],
                          fields['none.unaccepted'], fields['full.depth_not_current'], fields['contradicted'],
                          fields['full_alpha_d.visible'], fields['trust.revoked_exact'], fields['trust.forgotten'],
                          fields['through_ms']),
                         (10, 1, 1, 4, 2, 1, 2, 1, 1, 3, 12345))
        self.assertNotIn('runtime', fields)
        # The helper writes the same grammar.
        self.assertEqual(report.counter_fields(report.COUNTERS.search(counters('10:00:00', fields)).group(1)), fields)
        # The S1 text (three hold kinds and the cap, trusted_full) and a line before S1 (sources 0-9, the opaque
        # proof) parse the same way.
        s1 = report.counter_fields(counters('10:00:00', {'held.cap': 1, 'trusted_full': 2}, groups=S1_COUNTER_GROUPS))
        self.assertEqual((s1['held.cap'], s1['trusted_full'], 'held.none' in s1, 'reused' in s1), (1, 2, False, False))
        old = report.counter_fields(counters('10:00:00', {'decided.2': 4}, groups=S0_COUNTER_GROUPS))
        self.assertEqual((old['decided.2'], 'decided.10' in old, 'trust.opaque_set' in old), (4, False, True))
        # Totals are cumulative: the last line of the log is the session's.
        session = report.parse(BASE + [counters('10:00:12', {'auto_frames': 1, 'detection_frames': 1}),
                                       counters('10:00:13', CLEAN_COUNTERS, runtime='0000000000000002')])
        self.assertEqual(session.counters['auto_frames'], 100)

    def test_counters_replace_the_sampled_invariants(self):
        sample = ui('10:00:12', 2, 5, (0, 5, 0, 0), 0x2, 0x2)
        checks = run(BASE + [sample, counters('10:00:12', CLEAN_COUNTERS), line('10:00:20', 'Finished exiting.')])
        self.assertTrue(all(c.status in ('PASS', 'INFO') for c in checks.values()), checks)
        self.assertEqual(checks['UI counters'].detail, '100 Auto frames through the last of 30 committed samples: '
                                                       '80 detected, 15 held, 5 without detection')
        self.assertIn('UI layer (10) 75%', checks['UI protection'].detail)
        # T1: generated Presents hold the decision of the real frame they show or have none; a real frame without
        # its own decision reuses the previous real frame's once.
        self.assertEqual(checks['UI holds'].detail, "generated 12, none 3 (15% of Auto frames); 2 detection frames "
                                                    "(2.5%) reused the previous real frame's decision (T1)")
        self.assertEqual(checks['UI no mask'].detail,
                         '28% of Auto frames had no mask; decided none: ambiguous 20%; '
                         'without detection: no candidates 5.0%; generated Presents without a decision 3.0%')
        self.assertEqual(checks['UI trust events'].detail,
                         "earned 1, revoked one way by an exact pair 0, revoked by a declared alpha's coverage 0, "
                         'lapsed 0, restored 0, legacy entries discarded 0, forgotten 0')
        # Without counters (older logs) none of these checks exists and the sampled ones decide.
        sampled = run(BASE + [sample])
        for name in ('UI counters', 'UI full frame', 'UI inferred alpha', 'UI inexact difference', 'UI holds',
                     'UI no mask', 'UI trust events'):
            self.assertNotIn(name, sampled)
        self.assertEqual(sampled['UI protection'].detail, 'detected 1')

    def test_s1_counter_lines_keep_their_checks(self):
        # Counter lines logged in S1: three hold kinds with the cap, trusted_full and the S1 trust events.
        checks = run(BASE + [counters('10:00:12', S1_CLEAN_COUNTERS, groups=S1_COUNTER_GROUPS)])
        self.assertTrue(all(c.status in ('PASS', 'INFO') for c in checks.values()), checks)
        self.assertEqual(checks['UI counters'].detail, '100 Auto frames through the last of 30 committed samples: '
                                                       '80 detected, 15 held, 5 without detection')
        self.assertEqual(checks['UI holds'].detail, 'generated 10, inexact after exact 3, trusted missing 2 '
                                                    '(15% of Auto frames); 0 frames wanted a hold past the cap')
        self.assertEqual(checks['UI no mask'].detail,
                         '25% of Auto frames had no mask; decided none: ambiguous 20%; '
                         'without detection: no candidates 5.0%')
        self.assertIn('revoked by a full claim over the scene 0', checks['UI trust events'].detail)

    def test_counters_that_do_not_reconcile_fail(self):
        checks = run(BASE + [counters('10:00:12', {**CLEAN_COUNTERS, 'auto_frames': 101})])
        self.assertEqual(checks['UI counters'].status, 'FAIL')
        self.assertIn('does not reconcile', checks['UI counters'].detail)
        # Since S2a a generated Present without a decision is a held frame (held.none), not an inactive one.
        self.assertEqual(run(BASE + [counters('10:00:12', {**CLEAN_COUNTERS, 'held.none': 4})])
                         ['UI counters'].status, 'FAIL')
        self.assertEqual(run(BASE + [counters('10:00:12', {**CLEAN_COUNTERS, 'held.none': 4, 'auto_frames': 101})])
                         ['UI counters'].status, 'PASS')
        # A reused frame is a detection frame: it adds nothing to the identity.
        self.assertEqual(run(BASE + [counters('10:00:12', {**CLEAN_COUNTERS, 'reused': 40})])['UI counters'].status,
                         'PASS')
        # S1: a frame that wanted a hold past the cap detected instead: it is a detection frame, not a held one.
        capped = {**S1_CLEAN_COUNTERS, 'held.cap': 4}
        self.assertEqual(run(BASE + [counters('10:00:12', capped, groups=S1_COUNTER_GROUPS)])['UI counters'].status,
                         'PASS')
        self.assertEqual(run(BASE + [counters('10:00:12', {**S1_CLEAN_COUNTERS, 'auto_frames': 101},
                                              groups=S1_COUNTER_GROUPS)])['UI counters'].status, 'FAIL')

    def test_counted_invariant_breaks_fail_without_a_sample(self):
        clean = ui2('10:00:12', 2, 5, (0, 5, 0, 0), 0x2, 0x2)
        checks = run(BASE + [clean, counters('10:00:12', {**CLEAN_COUNTERS, 'presented_over_dedicated': 1})])
        self.assertEqual(checks['UI protection'].status, 'FAIL')
        # A2 revokes on the third contradiction within 2 s and never on shorter ones, which the counter cannot tell
        # apart, so contradicted frames alone are reported, not failed.
        checks = run(BASE + [clean, counters('10:00:12', {**CLEAN_COUNTERS, 'contradicted': 1})])
        self.assertEqual(checks['UI protection'].status, 'PASS')
        self.assertIn('one way (A2) in 1 frames, no sampled run reaching the revocation condition (3 within 2 s)',
                      checks['UI protection'].detail)
        # An accepted full Backbuffer over a valid exact pair (40 tiles of 256, 95% unchanged: a partial change set)
        # whose HUD-less image is lit and unchanged on 80% of its strong pixels. One or two such samples within 2 s
        # are noted; three within 2 s without a revocation fail, with or without counters.

        def flat(t):
            return ui2(t, 3, 1000, (0, 0, 1000, 0), 0x34, 0x4, hudless=(50, 950), tiles=200, lit=900,
                       strong=(0, 1000, 0), contradicted=(0, 800, 0))

        checks = run(BASE + [flat('10:00:12'), flat('10:00:13')])
        self.assertEqual(checks['UI protection'].status, 'PASS')
        self.assertIn('2 sampled A2 contradictions shorter than the revocation condition (3 within 2 s) were not '
                      'revoked, as designed', checks['UI protection'].detail)
        self.assertEqual(checks['UI protection'].times[0],
                         '10:00:12 Backbuffer alpha decided while an exact HUD-less pair showed 80% of its strong '
                         'pixels as lit, unchanged scene')
        # Three spread over more than 2 s are as short.
        self.assertEqual(run(BASE + [flat('10:00:10'), flat('10:00:12'), flat('10:00:13')])['UI protection'].status,
                         'PASS')
        run3 = [flat('10:00:12'), flat('10:00:13'), flat('10:00:14')]
        self.assertEqual(run(BASE + run3)['UI protection'].status, 'FAIL')
        checks = run(BASE + run3 + [counters('10:00:14', {**CLEAN_COUNTERS, 'contradicted': 2})])
        self.assertEqual((checks['UI protection'].status, len(checks['UI protection'].times)), ('FAIL', 3))
        self.assertIn('one way (A2) in 2 frames', checks['UI protection'].detail)
        self.assertEqual(checks['UI protection'].times[2],
                         '10:00:14 Backbuffer alpha decided while an exact HUD-less pair showed 80% of its strong '
                         'pixels as lit, unchanged scene (A2, one-way by an exact pair, 3 contradictions within 2 s, '
                         'not revoked)')
        # S1 counter lines keep the full-claim rule and its trusted_full counter.
        s1_flat = ui('10:00:12', 2, 1000, (0, 1000, 1000, 1000), 0x36, 0x2, hudless=(400, 600))
        checks = run(BASE + [s1_flat, counters('10:00:12', {**S1_CLEAN_COUNTERS, 'trusted_full': 2},
                                               groups=S1_COUNTER_GROUPS)])
        self.assertEqual((checks['UI protection'].status, len(checks['UI protection'].times)), ('FAIL', 1))
        self.assertIn('showed the scene in 2 frames', checks['UI protection'].detail)

    def test_one_way_contradiction_needs_a_valid_exact_pair_and_lit_unchanged_pixels(self):
        def thrice(**kw):
            """The same sample three times within 2 s, which A2 revokes when each is contradicted."""
            source, alpha, accepted = kw.pop('source', 3), kw.pop('alpha', (0, 0, 1000, 0)), kw.pop('accepted', 0x4)
            candidates, strong = kw.pop('candidates', 0x34), kw.pop('strong', (0, 1000, 0))
            return [ui2(t, source, 1000 if source else 0, alpha, candidates, accepted, strong=strong, **kw)
                    for t in ('10:00:12', '10:00:13', '10:00:14')]
        # A dim or tint over dark or changed pixels (Expedition 33 pause, The Witcher 3 sign wheel) is never
        # contradicted: the HUD-less image is not lit and unchanged under its strong pixels.
        dim = thrice(hudless=(50, 950), tiles=200, lit=900, contradicted=(0, 99, 0))
        self.assertEqual(run(BASE + dim)['UI protection'].status, 'PASS')
        # A middle-band pair (V2-invalid), an inexact pair (no 0x20) and a partial set in too few clean tiles never
        # judge.
        for judged in (thrice(hudless=(500, 500), tiles=200, lit=900, contradicted=(0, 800, 0)),
                       thrice(candidates=0x14, hudless=(50, 950), tiles=200, lit=900, contradicted=(0, 800, 0)),
                       thrice(hudless=(50, 950), tiles=100, lit=900, contradicted=(0, 800, 0))):
            self.assertEqual(run(BASE + judged)['UI protection'].status, 'PASS')
        # A full change set from an exact pair (at least 98% changed over a lit HUD-less image) judges too.
        full = thrice(hudless=(990, 10), lit=900, strong=(0, 100, 0), contradicted=(0, 10, 0))
        self.assertEqual(run(BASE + full)['UI protection'].status, 'FAIL')
        # An unaccepted source and a declared alpha are never judged; an accepted one is judged whatever decided,
        # the T1 reuse of a previous decision included.
        self.assertEqual(run(BASE + thrice(source=0, accepted=0x0, hudless=(50, 950), tiles=200, lit=900,
                                           contradicted=(0, 800, 0)))['UI protection'].status, 'PASS')
        reused = thrice(hudless=(50, 950), tiles=200, lit=900, contradicted=(0, 800, 0), reused=1,
                        reason='difference_failed', refused='hudless')
        self.assertEqual(run(BASE + reused)['UI protection'].status, 'FAIL')
        declared = thrice(source=1, alpha=(1000, 0, 0, 0), candidates=0x31, accepted=0x1, strong=(0, 0, 0),
                          hudless=(50, 950), tiles=200, lit=900)
        self.assertEqual(run(BASE + declared)['UI protection'].status, 'PASS')
        # The one-frame-late layer copy is not same-sample evidence (E2): its GPU counts no strong pixel, and no judge
        # reads it, a disagreeing accepted UIAlpha included.
        late = [ui2(t, 1, 20, (20, 0, 0, 0), 0x41, 0x41, layer=(400, 0), late=1)
                for t in ('10:00:12', '10:00:13', '10:00:14')]
        self.assertEqual(run(BASE + late)['UI protection'].status, 'PASS')
        # Lines after 180f1842 no longer carry sampled_late_layer; the layer is never judged all the same.
        unmarked = [ui2(t, 1, 20, (20, 0, 0, 0), 0x41, 0x41, layer=(400, 0), late=None)
                    for t in ('10:00:12', '10:00:13', '10:00:14')]
        self.assertNotIn('sampled_late_layer', unmarked[0])
        parsed = report.parse(BASE + unmarked).ui
        self.assertEqual((len(parsed), parsed[0].s2a), (3, True))
        # A line that still carries the field parses to the same samples.
        self.assertEqual(report.parse(BASE + late).ui, parsed)
        self.assertEqual(run(BASE + unmarked)['UI protection'].status, 'PASS')

    def test_counted_contradiction_resolved_by_revocation_passes(self):
        # The designed safety net (A2): an accepted Backbuffer that a valid exact pair contradicts one way keeps
        # deciding until three contradicting samples within 2 s revoke it, so its frames count as contradicted first.
        accepted = accepted_now('10:00:10', 'backbuffer:24:srgb,hudless:24:srgb')

        def flat(t='10:00:12'):
            return ui2(t, 3, 1000, (0, 0, 1000, 0), 0x34, 0x4, hudless=(50, 950), tiles=200, lit=900,
                       strong=(0, 1000, 0), contradicted=(0, 800, 0))
        revoke = accepted_now('10:00:14', 'hudless:24:srgb')
        resolved = {**CLEAN_COUNTERS, 'contradicted': 120, 'trust.revoked_exact': 1}
        checks = run(BASE + [accepted, flat(), revoke, counters('10:00:15', resolved)])
        self.assertEqual(checks['UI protection'].status, 'PASS')
        self.assertIn('in 120 frames before 1 one-way revocations', checks['UI protection'].detail)
        self.assertIn('a contradicted source lost its acceptance', checks['UI protection'].detail)
        self.assertEqual(checks['UI protection'].times,
                         ['10:00:12 Backbuffer alpha decided while an exact HUD-less pair showed 80% of its strong '
                          'pixels as lit, unchanged scene, acceptance revoked 10:00:14 (A2, one-way by an exact pair)'])
        # Without a sample, counted frames never fail: a one-way revocation resolves them, and without one they may
        # be contradictions shorter than the revocation condition.
        self.assertEqual(run(BASE + [counters('10:00:15', resolved)])['UI protection'].status, 'PASS')
        declared = {**CLEAN_COUNTERS, 'contradicted': 120, 'trust.revoked_declared': 1}
        self.assertEqual(run(BASE + [counters('10:00:15', declared)])['UI protection'].status, 'PASS')
        # A sampled run of three within 2 s that no revocation of its kind followed fails, whatever else was revoked.
        run3 = [flat('10:00:12'), flat('10:00:13'), flat('10:00:14')]
        self.assertEqual(run(BASE + run3 + [counters('10:00:15', resolved)])['UI protection'].status, 'FAIL')
        self.assertEqual(run(BASE + [accepted] + run3 + [revoke, counters('10:00:15', resolved)])
                         ['UI protection'].status, 'PASS')

    def test_s1_counted_full_claim_resolved_by_revocation_passes(self):
        # S1 logs: a trusted channel covering the frame over an exact pair that shows the scene keeps deciding until
        # repeated contradictions revoke its trust, so its frames count as trusted_full first.
        layer = ui('10:00:12', 2, 1000, (0, 1000, 1000, 1000), 0x36, 0x2, hudless=(400, 600), layer=1)
        revoke = line('10:00:14', '[Sunshine 3D] Sunshine UI protection: alpha trust is now 0x2; '
                                  'remembered for later sessions of this game')
        resolved = {**S1_CLEAN_COUNTERS, 'trusted_full': 120, 'trust.revoked_full': 1}
        checks = run(BASE + [layer, revoke, counters('10:00:15', resolved, groups=S1_COUNTER_GROUPS)])
        self.assertEqual(checks['UI protection'].status, 'PASS')
        self.assertIn('in 120 frames before 1 revocations of such a source', checks['UI protection'].detail)
        self.assertIn('a contradicted source lost its acceptance', checks['UI protection'].detail)
        # Without a sample, a revocation of a full claim also resolves the counted frames.
        self.assertEqual(run(BASE + [counters('10:00:15', resolved, groups=S1_COUNTER_GROUPS)])
                         ['UI protection'].status, 'PASS')
        # A sample no revocation resolved still fails, whatever else was revoked.
        flat = ui('10:00:12', 2, 1000, (0, 1000, 1000, 1000), 0x36, 0x2, hudless=(400, 600))
        self.assertEqual(run(BASE + [flat, revoke, counters('10:00:15', resolved, groups=S1_COUNTER_GROUPS)])
                         ['UI protection'].status, 'FAIL')

    def test_s2a_declared_coverage_disputes_name_their_revocation(self):
        # A2 (b) since S2a: an accepted inferred alpha disagrees with an accepted UIAlpha. Three such samples within
        # 2 s without a revocation warn; fewer are noted. The layer is never judged (the one-frame-late copy, E2).

        def disagreeing(t='10:00:12'):
            return ui2(t, 1, 20, (20, 0, 400, 0), 0x45, 0x45)
        backbuffer = disagreeing()
        checks = run(BASE + [accepted_now('10:00:10', 'ui_alpha:61:srgb,backbuffer:28:srgb'), backbuffer])
        self.assertEqual(checks['UI protection'].status, 'PASS')
        self.assertIn('1 sampled A2 contradictions shorter than the revocation condition',
                      checks['UI protection'].detail)
        checks = run(BASE + [accepted_now('10:00:10', 'ui_alpha:61:srgb,backbuffer:28:srgb')]
                     + [disagreeing(t) for t in ('10:00:12', '10:00:13', '10:00:14')])
        self.assertEqual(checks['UI protection'].status, 'WARN')
        self.assertIn('accepted inferred alpha disagreed with an accepted UI alpha or UI color tag 3 times within 2 s',
                      checks['UI protection'].detail)
        handled = run(BASE + [accepted_now('10:00:10', 'ui_alpha:61:srgb,backbuffer:28:srgb'), backbuffer,
                              accepted_now('10:00:14', 'ui_alpha:61:srgb')])
        self.assertEqual(handled['UI protection'].status, 'PASS')
        self.assertEqual(handled['UI protection'].times,
                         ['10:00:12 Backbuffer alpha 40% vs declared UI 2.0%, acceptance revoked 10:00:14 '
                          '(A2, declared coverage)'])
        # The layer no longer judges presented alpha, and an invalid declared alpha judges nothing.
        beside = ui2('10:00:12', 10, 5, (0, 0, 400, 0), 0x44, 0x44, layer=(5, 0))
        self.assertEqual(run(BASE + [beside])['UI protection'].status, 'PASS')
        invalid = ui2('10:00:12', 0, 0, (20, 0, 400, 0), 0x5, 0x5, invalid=(50, 0, 0, 0),
                      reason='presented_blocked', refused='backbuffer')
        self.assertEqual(run(BASE + [invalid])['UI protection'].status, 'PASS')

    def test_s2a_lines_name_the_reason_refused_candidate_forget_and_shadow(self):
        sample = report.parse(BASE + [ui2('10:00:06', 0, 0, (0, 0, 30, 0), 0x4, 0x0, strong=(0, 30, 0),
                                          reason='unaccepted', refused='backbuffer', tiles=7, lit=9)]).ui[0]
        self.assertEqual((sample.s2a, sample.strong, sample.contradicted, sample.reason, sample.refused, sample.reused,
                          sample.tiles, sample.lit),
                         (True, (0, 30, 0), (0, 0, 0), 'unaccepted', 'backbuffer', False, 7, 9))
        self.assertFalse(report.parse(BASE + [ui1('10:00:06', 0, 0, (0, 0, 3, 0), 0x4, 0x0)]).ui[0].s2a)
        # The gap names the add-on's own reason and the candidate it refused (F1).
        learning = ui2('10:00:06', 0, 0, (0, 0, 30, 0), 0x4, 0x0, reason='unaccepted', refused='backbuffer')
        protected = ui2('10:00:12', 3, 30, (0, 0, 30, 0), 0x4, 0x4)
        checks = run(BASE + [learning, protected])
        self.assertEqual(checks['UI protection gaps'].times,
                         ['10:00:06-10:00:12 (6 s) FG off: Backbuffer alpha covers 3.0% (not accepted) '
                          '(reason unaccepted: Backbuffer alpha)'])
        # With counters, the sampled reasons and refused candidates are listed beside the counted ones.
        checks = run(BASE + [learning, protected, counters('10:00:12', CLEAN_COUNTERS)])
        self.assertTrue(checks['UI no mask'].detail.endswith(
            '; sampled reasons with the refused candidate: unaccepted (Backbuffer alpha) 1'), checks['UI no mask'])
        # A reused sample shows the previous real frame's decision: it is a mask, not a refusal.
        reused = ui2('10:00:06', 3, 30, (0, 0, 0, 0), 0x0, 0x0, reused=1, reason='no_candidate')
        self.assertNotIn('sampled reasons', run(BASE + [reused, counters('10:00:12', CLEAN_COUNTERS)])
                         ['UI no mask'].detail)
        # The panel's Forget (A3) is listed.
        forget = line('10:00:07', '[Sunshine 3D] Sunshine UI protection: forgot learned UI sources '
                                  'backbuffer:24:srgb,ui_layer:28:srgb for this game; each source is accepted again '
                                  'by its own evidence')
        shadow = line('10:00:01', '[Sunshine 3D] Sunshine UI protection: first-run shadow measures this session '
                                  '(UISceneShadow=absent)')
        lines = BASE + [shadow, accepted_now('10:00:05', 'backbuffer:24:srgb,ui_layer:28:srgb'), forget,
                        accepted_now('10:00:07', 'none')]
        session = report.parse(lines)
        self.assertEqual([(kind, keys) for _, kind, _, keys in session.trust_events],
                         [('accepted UI sources are now', ('backbuffer:24:srgb', 'ui_layer:28:srgb')),
                          ('forgot learned UI sources', None), ('accepted UI sources are now', ())])
        trust = [c.detail for c in report.evaluate(session) if c.name == 'UI trust']
        self.assertIn('10:00:07 forgot learned UI sources backbuffer:24:srgb,ui_layer:28:srgb', trust)
        # The first-run shadow was removed: its line in older logs is ignored.
        self.assertNotIn('UI first-run shadow', run(lines))
        checks = run(BASE + [counters('10:00:12', {**CLEAN_COUNTERS, 'trust.forgotten': 2})])
        self.assertTrue(checks['UI trust events'].detail.endswith('forgotten 2'))
        # Since 10-22 the line counts the selective layer samples refused for colour without coverage (A1).
        uncovered = {**CLEAN_COUNTERS, 'trust.forgotten': 2, 'trust.uncovered_color': 21}
        text = report.COUNTERS.search(counters('10:00:12', uncovered, groups=UNCOVERED_COUNTER_GROUPS)).group(1)
        self.assertEqual(report.counter_fields(text)['trust.uncovered_color'], 21)
        checks = run(BASE + [counters('10:00:12', uncovered, groups=UNCOVERED_COUNTER_GROUPS)])
        self.assertTrue(checks['UI trust events'].detail.endswith(
            'forgotten 2; 21 selective layer samples earned nothing for colour without coverage (A1)'))
        self.assertEqual(checks['UI counters'].status, 'PASS')

    def test_counted_full_frame_over_a_visible_scene_warns(self):
        full = {**CLEAN_COUNTERS, 'decided.10': 50, 'decided.8': 10, 'full_d.hidden': 3}
        self.assertEqual(run(BASE + [counters('10:00:12', full)])['UI full frame'].status, 'PASS')
        # The sample releasing a held route decided under the hold, so a visible one is not by itself a fault.
        checks = run(BASE + [counters('10:00:12', {**full, 'full_d.visible': 1})])
        self.assertEqual(checks['UI full frame'].status, 'WARN')
        self.assertIn('H1', checks['UI full frame'].detail)
        self.assertIn('each release of a held hidden-scene route (8, 9) shows one', checks['UI full frame'].detail)
        self.assertIn('logs before S2b do not count releases', checks['UI full frame'].detail)
        self.assertNotIn('source 6', checks['UI full frame'].detail)
        # An exact full change-set (6) decides without a hold; over a visible scene it is intended for an accepted
        # exact pair (P1), so with no held route its visible samples are reported, not warned.
        exact = {**CLEAN_COUNTERS, 'decided.10': 50, 'decided.6': 10, 'full_d.visible': 2}
        checks = run(BASE + [counters('10:00:12', exact)])
        self.assertEqual(checks['UI full frame'].status, 'INFO')
        self.assertIn('2 samples of source 6 read the scene visible', checks['UI full frame'].detail)
        self.assertIn('intended for an accepted exact pair (P1)', checks['UI full frame'].detail)
        self.assertIn('open question', checks['UI full frame'].detail)
        # With a held route as well, the shared counter still warns and says what source 6 contributes.
        checks = run(BASE + [counters('10:00:12', {**exact, 'decided.9': 5})])
        self.assertEqual(checks['UI full frame'].status, 'WARN')
        self.assertIn('H1', checks['UI full frame'].detail)
        self.assertIn('samples of source 6 are counted with them', checks['UI full frame'].detail)
        self.assertIn('full frame (exact pair) (6) 12%, HUD-less route (before S2b) (9) 6.2%',
                      checks['UI protection'].detail)

    def test_s2b_counter_lines_check_the_h1_release_invariant(self):
        # A counter line from S2b through dc7e3c77: decided and full omit 9, and the scene guard's group follows
        # full_d.
        text = ('auto_frames=10 detection_frames=6 held={generated=2 none=1} reused=1 '
                'inactive={no_candidates=1 size=0 unprepared=0} decided={0=2 1=0 2=0 3=0 4=0 5=3 6=1 8=0 10=4} '
                'none={layer_aside=0 trusted_invalid=0 presented_blocked=0 ambiguous=0 difference_failed=1 '
                'gate_no_hold=1 no_candidate=0 other=0 unaccepted=2} full={6=1 8=0 depth_not_current=1} '
                'full_d={hidden=1 ambiguous=0 visible=0 invalid=0} scene={entered=1 released=1 refuted=2} '
                'untrusted_inferred=0 inexact_difference=3 contradicted=2 presented_over_dedicated=0 full_alpha=2 '
                'full_alpha_d={hidden=0 ambiguous=0 visible=1 invalid=0} trust={earned=1 revoked_exact=1 '
                'revoked_declared=0 lapsed=0 restored=0 discarded=2 forgotten=3} samples=4 through_ms=12345')
        fields = report.counter_fields('runtime=0000000000000001 ' + text)
        self.assertEqual((fields['scene.entered'], fields['scene.released'], fields['scene.refuted'],
                          'decided.9' in fields, 'full.9' in fields), (1, 1, 2, False, False))
        self.assertEqual(report.counter_fields(report.COUNTERS.search(
            counters('10:00:00', fields, groups=S2B_COUNTER_GROUPS)).group(1)), fields)
        # Stellar Blade SDR settings, FG suspended: H1 holds two menu visits flat, each released by one visible
        # sample that decided 8 under the hold. Visible H1 samples up to the releases are those releases.
        menus = {**CLEAN_COUNTERS, 'decided.10': 40, 'decided.8': 20, 'full_d.hidden': 6, 'full_d.visible': 2,
                 'scene.entered': 2, 'scene.released': 2}
        checks = run(BASE + [counters('10:00:12', menus, groups=S2B_COUNTER_GROUPS)])
        self.assertTrue(all(c.status in ('PASS', 'INFO') for c in checks.values()), checks)
        self.assertEqual(checks['UI full frame'].detail,
                         '20 full-frame frames (6: 0, 8: 20; 25% of detection frames); H1 samples (8) read the scene '
                         'hidden 6, ambiguous 0, visible 2, invalid or unmeasured 0; scene guard entered 2, released '
                         '2, refuted 0; depth not current on 0 detection frames')
        self.assertIn('full frame over a hidden scene (H1) (8) 25%', checks['UI protection'].detail)
        # More visible H1 samples than releases: an H1 hold acted over a visible scene.
        checks = run(BASE + [counters('10:00:12', {**menus, 'full_d.visible': 3}, groups=S2B_COUNTER_GROUPS)])
        self.assertEqual(checks['UI full frame'].status, 'WARN')
        self.assertTrue(checks['UI full frame'].detail.endswith(
            '; an H1 hold acted over a visible scene: 3 visible H1 samples for 2 releases'), checks['UI full frame'])
        # An accepted exact full change-set (6) is an accepted whole-frame decision (P1): its samples are counted in
        # full_alpha_d, and over a visible scene it is reported, not warned.
        exact = {**CLEAN_COUNTERS, 'decided.10': 50, 'decided.6': 10, 'full_alpha_d.visible': 2}
        checks = run(BASE + [counters('10:00:12', exact, groups=S2B_COUNTER_GROUPS)])
        self.assertEqual((checks['UI full frame'].status, checks['UI full alpha'].status), ('PASS', 'INFO'))
        self.assertEqual(checks['UI full alpha'].detail,
                         '0 frames decided a whole-frame alpha and 10 an exact full change-set (6) (12% of detection '
                         'frames); samples of them read the scene hidden 0, ambiguous 0, visible 2, invalid or '
                         'unmeasured 0; 2 samples pinned an accepted whole-frame decision flat over a visible scene, '
                         'as intended (P1)')
        clean = run(BASE + [counters('10:00:12', CLEAN_COUNTERS, groups=S2B_COUNTER_GROUPS)])
        self.assertEqual(clean['UI full alpha'].detail,
                         'no frame decided a whole-frame alpha or an exact full change-set (6)')

    def test_current_counter_line_omits_the_counters_that_logged_zero(self):
        # The text test_game3d_alpha_auto pins for format_ui_counters: builds after dc7e3c77 no longer log
        # untrusted_inferred, presented_over_dedicated or full_alpha_d, which had logged only 0 since revision 9.
        text = ('auto_frames=15 detection_frames=11 held={generated=2 none=1} reused=1 '
                'inactive={no_candidates=1 size=0 unprepared=0} decided={0=2 1=0 2=0 3=0 4=0 5=3 6=1 8=0 10=4} '
                'none={layer_aside=0 trusted_invalid=0 presented_blocked=0 ambiguous=0 difference_failed=1 '
                'gate_no_hold=1 no_candidate=0 other=0 unaccepted=2} full={6=1 8=0 depth_not_current=1} '
                'full_d={hidden=1 ambiguous=0 visible=0 invalid=0} scene={entered=1 released=1 refuted=2} '
                'inexact_difference=3 contradicted=2 full_alpha=2 trust={earned=1 revoked_exact=1 '
                'revoked_declared=0 lapsed=0 restored=0 discarded=2 forgotten=3} samples=4 through_ms=12345')
        fields = report.counter_fields('runtime=0000000000000001 ' + text)
        self.assertEqual((fields['scene.refuted'], fields['inexact_difference'], fields['full_alpha'],
                          fields['trust.forgotten']), (2, 3, 2, 3))
        self.assertFalse([k for k in fields
                          if k.split('.')[0] in ('untrusted_inferred', 'presented_over_dedicated', 'full_alpha_d')])
        self.assertEqual(report.counter_fields(report.COUNTERS.search(
            counters('10:00:00', fields, groups=CURRENT_COUNTER_GROUPS)).group(1)), fields)
        # A whole-frame alpha is reported without a breakdown, and the line no longer measures unaccepted inferred
        # alpha, so it has no such check.
        checks = run(BASE + [counters('10:00:12', {**CLEAN_COUNTERS, 'full_alpha': 40}, groups=CURRENT_COUNTER_GROUPS)])
        self.assertEqual((checks['UI full alpha'].status, checks['UI full alpha'].detail),
                         ('INFO', '40 frames decided a whole-frame alpha and 0 an exact full change-set (6) (50% of '
                          'detection frames)'))
        self.assertNotIn('UI inferred alpha', checks)
        self.assertEqual((checks['UI counters'].status, checks['UI protection'].status), ('PASS', 'PASS'))

    def test_every_inactive_counter_reason_counts(self):
        # A later add-on may add a reason for frames without detection: it reconciles and is listed after the known
        # ones, the new ones by name.
        groups = tuple((key, inner + ('zeta', 'future') if key == 'inactive' else inner)
                       for key, inner in UNCOVERED_COUNTER_GROUPS)
        values = {**CLEAN_COUNTERS, 'auto_frames': 112, 'inactive.zeta': 7, 'inactive.future': 5}
        checks = run(BASE + [counters('10:00:12', values, groups=groups)])
        self.assertEqual(checks['UI counters'].status, 'PASS', checks['UI counters'].detail)
        self.assertIn('17 without detection', checks['UI counters'].detail)
        self.assertIn('without detection: no candidates 4.5%, future 4.5%, zeta 6.2%', checks['UI no mask'].detail)

    def test_s2b_lines_parse_the_pre_ui_scene_and_the_scene_guard(self):
        # Stellar Blade SDR settings with FG suspended: the cleared output target holds the pre-UI scene (colour
        # without alpha, V1-invalid) and reads visible while the presented menu reads hidden; under both held
        # verdicts H1 (d) decides 8 over the S1 winner (no mask).
        menu = ui3('10:00:12', 8, 1000, (0, 0, 0, 1000), 0x48, 0x0, layer=(0, 600), invalid=(0, 0, 0, 0),
                   scene=(0.031, 'hidden'), claims=0x80, h1=1, winner=0, pre_ui=('layer', 0.588, 1), guard=(1, 1, 0))
        u = report.parse(BASE + [menu]).ui[0]
        self.assertTrue(u.scene.s2b)
        self.assertEqual((u.source, u.scene.verdict, u.scene.d, u.scene.pre_ui_image, u.scene.hudless_d,
                          u.scene.hudless_valid, u.scene.claims, u.scene.h1, u.scene.winner, u.scene.guard,
                          u.scene.inferred_opaque, u.scene.pre_ui_claim()),
                         (8, 'hidden', 0.031, 'layer', 0.588, True, 0x80, True, 0, (1, 1, 0), (0, 1000), True))
        # Older lines have no scene guard.
        old = report.parse(BASE + [ui('10:00:12', 0, 0, (0, 0, 0, 1000), 0x8, 0x0,
                                      scene=(0.01, 'hidden', 0, 1, 0))]).ui[0]
        self.assertEqual((old.scene.s2b, old.scene.guard, old.scene.pre_ui_claim()), (False, None, False))
        # The hidden-scene check lists H1 samples by the pre-UI image whose claim acted, and the guard's entries,
        # releases and refutations from the counter line. A layer claim (b) acting without the pre-UI hold is H1 but
        # not a pre-UI sample.
        release = ui3('10:00:13', 8, 1000, (0, 0, 0, 1000), 0x48, 0x0, layer=(0, 600), claims=0x80, h1=1, winner=0,
                      pre_ui=('layer', 0.575, 1), guard=(1, 1, 0))
        splash = ui3('10:00:14', 8, 1000, (0, 0, 0, 1000), 0x48, 0x0, layer=(1000, 0), scene=(-0.044, 'hidden'),
                     claims=0x40, h1=1, winner=0, pre_ui=('layer', -0.04, 1), guard=(1, 0, 0))
        gameplay = ui3('10:00:15', 0, 0, (0, 0, 0, 1000), 0x48, 0x0, layer=(0, 600), reason='layer_aside',
                       refused='ui_layer', claims=0x80, pre_ui=('layer', 0.518, 1), scene=(0.515, 'visible'))
        guard = {**CLEAN_COUNTERS, 'decided.8': 3, 'full_d.hidden': 2, 'full_d.visible': 1, 'scene.entered': 2,
                 'scene.released': 1, 'scene.refuted': 1}
        checks = run(BASE + [menu, release, splash, gameplay, counters('10:00:15', guard, groups=S2B_COUNTER_GROUPS)])
        self.assertEqual((checks['Hidden scene'].status, checks['UI full frame'].status), ('INFO', 'PASS'))
        self.assertEqual(checks['Hidden scene'].detail,
                         'H1 hidden scene (8) in 3 samples (pre-UI image: hudless 0, layer 2), entered 2, released 1, '
                         'refuted 1; 4 of 4 samples measured')
        # H1 is a mask: it leaves no protection gap.
        self.assertEqual(checks['UI protection gaps'].status, 'PASS')
        # Without a counter line the guard's holds are counted from the samples.
        self.assertIn('hidden hold held in 3 samples (no counter line)',
                      run(BASE + [menu, release, splash, gameplay])['Hidden scene'].detail)
        # A hidden run with no decided source still warns, as before S2b.
        dark = ui3('10:00:16', 0, 0, (0, 0, 0, 1000), 0x48, 0x0, layer=(0, 600), reason='layer_aside',
                   refused='ui_layer', scene=(0.02, 'hidden'), claims=0x80, pre_ui=('layer', 0.03, 1),
                   guard=(1, 0, 0), shadow=1, hidden_ms=900)
        checks = run(BASE + [dark, counters('10:00:16', guard, groups=S2B_COUNTER_GROUPS)])
        self.assertEqual((checks['Hidden scene'].status, checks['Hidden scene'].times),
                         ('WARN', ['10:00:15 900 ms (first-run shadow)']))
        self.assertIn('H1 hidden scene (8) in 0 samples', checks['Hidden scene'].detail)
        self.assertNotIn('unproven', checks['Hidden scene'].detail)
        # Since the layer proof: a settings menu opened before any gameplay sample proved the scene layer reads hidden
        # with the layer visible, and stays 3D; the report counts it and parses proven.
        unproven = ui3('10:00:17', 0, 0, (0, 0, 0, 1000), 0x48, 0x0, layer=(0, 600), reason='layer_aside',
                       refused='ui_layer', scene=(0.031, 'hidden'), claims=0x80, pre_ui=('layer', 0.588, 1),
                       guard=(1, 0, 0, 0))
        proven = ui3('10:00:18', 8, 1000, (0, 0, 0, 1000), 0x48, 0x0, layer=(0, 600), scene=(0.031, 'hidden'),
                     claims=0x80, h1=1, winner=0, pre_ui=('layer', 0.588, 1), guard=(1, 1, 0, 1))
        parsed = report.parse(BASE + [unproven, proven]).ui
        self.assertEqual([u.scene.proven for u in parsed], [False, True])
        self.assertIsNone(report.parse(BASE + [menu]).ui[0].scene.proven)
        detail = run(BASE + [unproven, proven])['Hidden scene'].detail
        self.assertIn('1 hidden samples had an unproven pre-UI layer', detail)
        self.assertIn('pre-UI image: hudless 0, layer 1', detail)

    def test_fix1_lines_parse_the_pre_ui_pixels_and_keep_s2b_lines(self):
        # A real S2b line parses as before: no pixel counts, and its unproven layer claim is named by the S2b proof.
        old = report.parse([SB_S2B_SETTINGS]).ui[0]
        self.assertEqual((old.scene.s2b, old.scene.fix1, old.scene.pre_ui_pixels, old.scene.proven, old.scene.claims,
                          old.scene.pre_ui_image, old.scene.hudless_d),
                         (True, False, None, False, 0x80, 'layer', 0.544))
        detail = run(BASE + [SB_S2B_SETTINGS])['Hidden scene'].detail
        self.assertIn('1 hidden samples had an unproven pre-UI layer (no gameplay sample had read it within 0.03 of '
                      'the presented frame, the proof of S2b lines)', detail)
        # Since fix 1 the same sample carries texel 11 and no claim; an offered layer without coverage whose signature
        # has no proof yet is still counted, named by the pixel rule.
        new = report.parse([SB_FIX1_SETTINGS]).ui[0]
        self.assertEqual((new.scene.fix1, new.scene.pre_ui_pixels, new.scene.claims, new.scene.proven,
                          new.scene.hidden_ms, new.bare_layer(), new.hidden()),
                         (True, (2719720, 2148784), 0, False, 656, True, True))
        detail = run(BASE + [SB_FIX1_SETTINGS])['Hidden scene'].detail
        self.assertIn('1 hidden samples had an unproven pre-UI layer (no proof yet: 3 samples over 2 s whose layer '
                      'equals the presented frame on at least 90% of pixels and is lit on at least half)', detail)
        self.assertNotIn('0.03', detail)
        # A proven layer is not counted, nor a layer with coverage (a real UI layer, as in HDR) or a visible frame.
        for line_text in (SB_FIX1_SETTINGS.replace('proven=0', 'proven=1'),
                          SB_FIX1_SETTINGS.replace('sampled_layer={covered=0', 'sampled_layer={covered=14519'),
                          SB_FIX1_SETTINGS.replace('verdict=hidden', 'verdict=visible')):
            self.assertNotIn('unproven', run(BASE + [line_text])['Hidden scene'].detail)
        # The fixture helper writes the same field.
        flat = ui3('10:00:12', 8, 1000, (0, 0, 0, 1000), 0x48, 0x0, layer=(0, 600), scene=(0.031, 'hidden'),
                   claims=0x80, h1=1, winner=0, pre_ui=('layer', 0.588, 1), guard=(1, 1, 0, 1),
                   pre_ui_pixels=(330, 260, 600, 520))
        self.assertEqual(report.parse(BASE + [flat]).ui[0].scene.pre_ui_pixels, (330, 260))
        # A line without the removed shadow fields and the last two pixel counts (shadow statistics; since selection
        # revision 9 they are logged as 0) parses the same.
        current = flat.replace('shadow=0 shadow_hidden_ms=0 ', '').replace(
            ' presented_lit=600 presented_lit_differs=520', '')
        self.assertNotIn('shadow=', current)
        parsed = report.parse(BASE + [current]).ui[0].scene
        self.assertEqual((parsed.s2b, parsed.pre_ui_pixels, parsed.shadow, parsed.hidden_ms, parsed.proven),
                         (True, (330, 260), False, 0, True))
        self.assertEqual(run(BASE + [flat])['Hidden scene'].detail,
                         'H1 hidden scene (8) in 1 samples (pre-UI image: hudless 0, layer 1), hidden hold held in 1 '
                         'samples (no counter line); 1 of 1 samples measured')

    def test_fix1_pre_ui_proof_keys_are_reported_and_are_no_coverage_source(self):
        restored = line('10:00:01', '[Sunshine 3D] Sunshine UI protection: restored accepted UI sources '
                                    'ui_color:87:srgb,pre_ui:87:srgb from an earlier session of this game')
        session = report.parse(BASE + [restored, accepted_now('10:00:20', 'ui_color:87:srgb'),
                                       accepted_now('10:00:30', 'ui_color:87:srgb,pre_ui:87:srgb,pre_ui:87:pq')])
        self.assertEqual(session.trust_events[0][3], ('ui_color:87:srgb', 'pre_ui:87:srgb'))
        self.assertEqual(report.pre_ui_proofs(session),
                         [(36001.0, 'restored', 'pre_ui:87:srgb'), (36020.0, 'lapsed', 'pre_ui:87:srgb'),
                          (36030.0, 'earned', 'pre_ui:87:pq'), (36030.0, 'earned', 'pre_ui:87:srgb')])
        proofs = [c for c in report.evaluate(session) if c.name == 'Pre-UI proof']
        self.assertTrue(all(c.status == 'INFO' for c in proofs))
        self.assertEqual([c.detail for c in proofs], [
            '10:00:01 pre_ui:87:srgb restored from an earlier session, provisional: 3 samples over 2 s whose layer '
            'equals the presented frame on at least 90% of pixels and is lit on at least half confirm it, and it '
            'lapses after 60 s of testable time (the layer offered without coverage while the presented frame reads '
            'visible) without them',
            '10:00:20 pre_ui:87:srgb lapsed: restored and not confirmed within 60 s of testable time',
            '10:00:30 pre_ui:87:pq earned by 3 samples over 2 s whose layer equals the presented frame on at least 90% '
            'of pixels and is lit on at least half',
            '10:00:30 pre_ui:87:srgb earned by 3 samples over 2 s whose layer equals the presented frame on at least '
            '90% of pixels and is lit on at least half'])
        # Forget logs the acceptance change, then the Forget line naming the cleared keys.
        forget = line('10:00:40', '[Sunshine 3D] Sunshine UI protection: forgot learned UI sources '
                                  'ui_color:87:srgb,pre_ui:87:srgb for this game; each source is accepted again by its '
                                  'own evidence')
        session = report.parse(BASE + [accepted_now('10:00:30', 'ui_color:87:srgb,pre_ui:87:srgb'),
                                       accepted_now('10:00:40', 'none'), forget])
        self.assertEqual(report.pre_ui_proofs(session)[1:], [(36040.0, 'forgotten', 'pre_ui:87:srgb')])
        # S2b logs have no proof keys, and so no such line.
        self.assertNotIn('Pre-UI proof', run(BASE + [restored.replace(',pre_ui:87:srgb', '')]))

        # A pre-UI proof key is not a UI coverage source: an accepted source's revocation beside a proof of the same
        # format resolves its dispute, and a proof that lapses alone resolves nothing. The layer the proof is about is
        # never judged (the one-frame-late copy, E2), so its disagreement is no dispute either way.
        def disagreeing(t):
            return ui2(t, 1, 20, (20, 0, 400, 0), 0x45, 0x45)
        samples = [disagreeing(t) for t in ('10:00:12', '10:00:13', '10:00:14')]
        start = accepted_now('10:00:10', 'ui_alpha:61:srgb,backbuffer:28:srgb,pre_ui:28:srgb')
        handled = run(BASE + [start] + samples + [accepted_now('10:00:15', 'ui_alpha:61:srgb,pre_ui:28:srgb')])
        self.assertEqual(handled['UI protection'].status, 'PASS')
        self.assertIn('acceptance revoked 10:00:15', handled['UI protection'].times[0])
        unresolved = run(BASE + [start] + samples + [accepted_now('10:00:15', 'ui_alpha:61:srgb,backbuffer:28:srgb')])
        self.assertEqual(unresolved['UI protection'].status, 'WARN')
        layer = [ui2(t, 1, 20, (20, 0, 0, 0), 0x41, 0x41, layer=(400, 0), late=None)
                 for t in ('10:00:12', '10:00:13', '10:00:14')]
        start = accepted_now('10:00:10', 'ui_alpha:61:srgb,ui_layer:28:srgb,pre_ui:28:srgb')
        kept = run(BASE + [start] + layer + [accepted_now('10:00:15', 'ui_alpha:61:srgb,ui_layer:28:srgb')])
        self.assertEqual((kept['UI protection'].status, kept['UI protection'].times), ('PASS', []))

    def test_empty_change_set_is_valid_v2_evidence(self):
        # Since selection revision 8 a lit pair without any changed pixel in clean tiles is a valid empty change set
        # (ui_selection::change_set_empty), so an exact empty pair judges like any other valid exact pair.
        empty = report.parse(BASE + [ui2('10:00:06', 0, 0, (0, 0, 0, 0), 0x30, 0x10, hudless=(0, 1000), tiles=200,
                                         lit=600)]).ui[0]
        self.assertTrue((empty.change_set_valid(), empty.exact_judge()) == (True, True))
        for tiles, lit in ((127, 600), (200, 499)):
            sample = report.parse(BASE + [ui2('10:00:06', 0, 0, (0, 0, 0, 0), 0x30, 0x10, hudless=(0, 1000),
                                              tiles=tiles, lit=lit)]).ui[0]
            self.assertFalse(sample.change_set_valid())

    def test_current_ui_line_has_no_layer_column_or_zero_fields(self):
        # The line exporter.cpp writes after be7788bf: sampled_one_way holds Backbuffer and current alpha only,
        # and the removed first-run shadow's fields and texel 11's shadow statistics are gone.
        def flat(t):
            return line(t, '[Sunshine 3D] Sunshine UI protection: runtime=0000000000000001 mode=auto rendered=1 '
                           'mask_path=1 input=automatic_gpu_mask retained=0 fg=0 fg_known=1 fg_enabled=0 '
                           'input_state=input_seen detection=detected selected=automatic source=automatic '
                           'source_availability=detected sampled_source=3 sampled_covered=1000 sampled_pixels=1000 '
                           'sampled_candidates=0x34 sampled_alpha_covered=0/0/1000/0 sampled_alpha_invalid=0/0/0/0 '
                           'accepted=0x4 sampled_layer={covered=0 invalid=0 opaque=0} '
                           'sampled_one_way={strong=1000/0 contradicted=800/0} sampled_reason=decided '
                           'sampled_refused=none sampled_reused=0 sampled_hudless={changed=50 unchanged=950 invalid=0 '
                           'matching_tiles=200 lit=900} sampled_alpha_opaque=0/0 sampled_inferred_opaque=0/0 '
                           'sampled_claims=0x0 sampled_h1={applied=0 winner=3} '
                           'sampled_scene={n=463 d=0.600 valid=1 ran=1 verdict=visible} '
                           'sampled_pre_ui_scene={image=none n=0 d=0.000 valid=0} '
                           'scene_guard={hidden=0 pre_ui=0 refuted=0 proven=0} '
                           'sampled_pre_ui_pixels={match=7 image_lit=9} '
                           'sampled_declared_one_way={strong=0/0 contradicted=0/0} status_revision=1')
        sample = report.parse(BASE + [flat('10:00:12')]).ui[0]
        self.assertEqual((sample.s2a, sample.strong, sample.contradicted, sample.one_way_counts(2), sample.one_way(2),
                          sample.declared_one_way, sample.scene.s2b, sample.scene.pre_ui_pixels,
                          sample.scene.hidden_ms),
                         (True, (0, 1000, 0), (0, 800, 0), (1000, 800), True, ((0, 0), (0, 0)), True, (7, 9), 0))
        # The same Backbuffer contradiction as on older lines: three within 2 s that no revocation followed fail.
        resolved = {**CLEAN_COUNTERS, 'contradicted': 120, 'trust.revoked_exact': 1}
        run3 = [flat('10:00:12'), flat('10:00:13'), flat('10:00:14')]
        checks = run(BASE + run3 + [counters('10:00:15', resolved, groups=CURRENT_COUNTER_GROUPS)])
        self.assertEqual(checks['UI protection'].status, 'FAIL')
        self.assertIn('10:00:12 Backbuffer alpha decided while an exact HUD-less pair showed 80% of its strong pixels',
                      checks['UI protection'].times[0])

    def test_declared_alpha_one_way_contradictions_since_revision_10(self):
        # Since selection revision 10 the exact one-way judge also reads UIAlpha and the UI colour tag
        # (sampled_declared_one_way); three contradictions within 2 s that no revocation followed fail.
        def wheel(t, against):
            text = ui2(t, 1, 800, (800, 0, 0, 0), 0x31, 0x11, hudless=(10, 980), tiles=200, lit=600)
            declared = f' sampled_declared_one_way={{strong=800/0 contradicted={against}/0}} status_revision=1'
            return text.replace(' status_revision=1', declared)
        sample = report.parse(BASE + [wheel('10:00:06', 400)]).ui[0]
        self.assertEqual((sample.declared_one_way, sample.one_way_counts(0), sample.one_way(0)),
                         (((800, 0), (400, 0)), (800, 400), True))
        self.assertFalse(report.parse(BASE + [wheel('10:00:06', 40)]).ui[0].one_way(0))
        run_of_three = [wheel(f'10:00:{6 + i:02}', 400) for i in range(3)]
        unresolved = run(BASE + run_of_three + [counters('10:00:12', {**CLEAN_COUNTERS, 'contradicted': 3})])
        self.assertEqual(unresolved['UI protection'].status, 'FAIL')
        self.assertIn('UI alpha decided while an exact HUD-less pair showed 50% of its strong pixels',
                      unresolved['UI protection'].times[0])
        revoked = run(BASE + run_of_three + [accepted_now('10:00:09', 'none'),
                                             counters('10:00:12', {**CLEAN_COUNTERS, 'contradicted': 3})])
        self.assertEqual(revoked['UI protection'].status, 'PASS')
        # The gate's re-offered pairings are Present-counted pairings too.
        gate = ('Sunshine UI capture gate: runtime=1 hook={state=admitted} hudless_presents={batch=0 real=2 late=1 '
                'generated=0 reoffered=7 stale=0 other=0 none=0}; gate metadata does not authorize pixels')
        self.assertEqual(report.parse(BASE + [line('10:00:06', '[Sunshine 3D] ' + gate)]).gate_counted, 10)

    def test_stream_delivery_compares_claimed_frames_with_the_games_offer(self):
        # Stellar Blade 10-05 FG 4x before the export-throughput fix: the game offered ~220 frames/s, the stream ran at
        # 90 fps, and the host claimed ~50/s (published - overwritten_unconsumed).
        def out(t, published, dropped, overwritten):
            return line(t, f'[Sunshine 3D] Sunshine SBS output: published={published} fg=0 scene_flat=0 '
                           f'scene_fading=0 published_fresh_depth={published} published_reused_depth=0 '
                           'published_depth_missing=0 '
                           '(unavailable=0 reuse_after_gap=0 reuse_other_source=0) runtime=0x1 generation=7 '
                           f'dropped={dropped} overwritten_unconsumed={overwritten}; cumulative')
        slow = [out(f'10:00:{5 * i:02}', 600 * i, 500 * i, 350 * i) for i in range(1, 6)]

        def judged(lines, fps):
            session = report.parse(BASE + lines)
            session.stream_fps = fps
            return [c for c in report.evaluate(session) if c.name == 'Stream delivery'][0]
        check = judged(slow, 90.0)
        self.assertEqual(check.status, 'WARN')
        self.assertTrue(check.detail.startswith('the host took 50.0 new frames/s of a target 90.0/s'), check.detail)
        self.assertEqual(check.times, ['10:00:05-10:00:25 (20 s): 50.0 of 90.0 new frames/s'])
        # Without the host log the stream rate is unknown, so nothing is judged: a 72 Hz stream that takes every frame
        # it can (RE9 and Hogwarts 10-06) read as a WARN against an assumed 90 fps.
        unknown = run(BASE + slow)['Stream delivery']
        self.assertEqual(unknown.status, 'INFO')
        self.assertEqual(unknown.detail, 'stream rate unknown (pass --host-log to judge it): the host took 50.0 new '
                                         'frames/s, the game offered 220.0/s')
        self.assertFalse(unknown.times)
        # A game offering 60/s and a host claiming all of them passes; the target follows the game.
        follows = [out(f'10:00:{5 * i:02}', 300 * i, 0, 0) for i in range(1, 6)]
        check = judged(follows, 90.0)
        self.assertEqual(check.status, 'PASS')
        self.assertTrue(check.detail.startswith('the host took 60.0 new frames/s of a target 60.0/s'), check.detail)
        # The host log's stream rate is the cap.
        self.assertIn('target 72.0/s', judged(slow, 72.0).detail)
        exact = report.HOST_FPS.search('Requested frame rate [90/1 exactly 90fps]')
        self.assertEqual(exact.groups(), ('90', '1', None))
        self.assertEqual(report.HOST_FPS.search('Requested frame rate [60fps]').groups(), (None, None, '60'))
        # Older logs without the slot counters have no such check.
        self.assertNotIn('Stream delivery', run(BASE))
        # The headset moved its panel to 72 Hz mid-session (RE9 10-06): the target follows the live stream rate, so a
        # host claiming every frame at 72 fps passes.
        at72 = [out(f'10:00:{5 * i:02}', 450 * i, 0, 90 * i) for i in range(1, 6)]
        session = report.parse(BASE + at72)
        session.stream_fps = 72.0
        session.stream_fps_timeline = [(30000.0, 90.0), (36004.0, 72.0)]
        check = [c for c in report.evaluate(session) if c.name == 'Stream delivery'][0]
        self.assertEqual(check.status, 'PASS')
        self.assertIn('target 72.0/s', check.detail)
        self.assertIn('stream 72-90 fps, followed live', check.detail)
        live = report.HOST_LIVE_FPS.search('Virtual display resized live to 3840x2160 @ 72 Hz.')
        self.assertEqual(live.group(2), '72')
        self.assertEqual(report.HOST_LIVE_FPS.search('Capture pacing updated to 90fps for a live video-mode change')
                         .group(1), '90')

    def test_ui_layer_copies_judge_the_time_skipped_copies_left_the_layer_stale(self):
        def timing(t, copies, skipped, gap=None, presents=950, window_ms=10000, mean=3.6, peak=6):
            return line(t, '[Sunshine 3D] Sunshine Game 3D timing: presents={} cpu_ms={{mean=0.125 max=0.300}} '
                           'cpu_worst_ms={{setup=0.004 depth=0.200 ui=0.005 render=0.050 export=0.040}} gpu_frames=0 '
                           'gpu_ms mean/max={{total=0.000/0.000 inputs=0.000/0.000}} gpu_profile=disabled '
                           'dropped_fence_pending=0 dropped_unresolved=0 incomplete=0 ui_layer={{copies={} skipped={} '
                           'offers={} presents_since_copy={{mean={:.2f} max={}}}{}}}{}'.format(
                               presents, copies, skipped, presents, mean, peak,
                               '' if gap is None else f' skip_gap_ms={gap:.1f}',
                               '' if window_ms is None else f' window_ms={window_ms}'))

        def judged(lines, fps=90.0, timeline=()):
            session = report.parse(BASE + lines)
            session.stream_fps = fps
            session.stream_fps_timeline = list(timeline)
            return {c.name: c for c in report.evaluate(session)}
        # FG on (The Witcher 3 10-07: ~95 Presents/s, a copy per real frame, ~47/s): one skipped copy leaves the layer
        # unrefreshed for two real frames, ~43 ms, so streamed frames read a layer a real frame older than designed.
        fg = judged([timing('10:00:12', 470, 30, gap=42.6)])['UI layer copies']
        self.assertEqual(fg.status, 'WARN')
        self.assertEqual(fg.times, ['10:00:02-10:00:12 (10.0 s): skipped 30 of 500 copies, longest refresh gap 42.6 ms '
                                    'of a 11.1 ms stream frame (95 Presents/s, presents_since_copy mean 3.60 max 6)'])
        # An uncapped burst (~600 clears/s) skipping three of four copies costs ~3 ms: INFO, alone in its window or
        # sharing one with 8 s of FG-on play at 47 copies/s, so the window's placement and averages change nothing.
        burst = judged([timing('10:00:04', 300, 900, gap=3.4, presents=1200, window_ms=2000)])['UI layer copies']
        shared = judged([timing('10:00:12', 676, 900, gap=3.4, presents=1960)])['UI layer copies']
        for check in (burst, shared):
            self.assertEqual((check.status, check.detail),
                             ('INFO', '1 timing window skipped layer copies (a full ring): 1 every refresh gap within '
                                      'a stream frame'))
        self.assertIn('longest refresh gap 3.4 ms of a 11.1 ms stream frame', shared.times[0])
        # A high rate is no excuse for a long gap (140 fps without FG, isolated skips of 30% of copies: ~14 ms).
        self.assertEqual(judged([timing('10:00:12', 980, 420, gap=14.3, presents=1400)])['UI layer copies'].status,
                         'WARN')
        # The stream frame is the one at the window's start: 12.5 ms is within a 72 fps frame, not a 90 fps one.
        self.assertEqual(judged([timing('10:00:12', 800, 100, gap=12.5)], 72.0, [(36000.0, 72.0)])
                         ['UI layer copies'].status, 'INFO')
        self.assertEqual(judged([timing('10:00:12', 800, 100, gap=12.5)])['UI layer copies'].status, 'WARN')
        # A ring that stayed full past the layer's 250 ms recency gate warns in any window, with or without the host
        # log: the layer was then no longer offered at all.
        stuck = {c.name: c for c in report.evaluate(report.parse(BASE + [timing('10:00:12', 0, 900, gap=9950.0)]))}
        self.assertEqual(stuck['UI layer copies'].status, 'WARN')
        self.assertIn('unrefreshed for 250 ms or more, after which it is no longer offered',
                      stuck['UI layer copies'].detail)
        # Skips before the export streamed are INFO, and so are skips without the host log's stream rate.
        early = judged([timing('10:00:01', 40, 10, gap=60.0, window_ms=1000)])['UI layer copies']
        self.assertEqual(early.status, 'INFO')
        self.assertTrue(early.times[0].endswith(', not streamed'), early.times[0])
        unknown = run(BASE + [timing('10:00:12', 470, 30, gap=42.6)])['UI layer copies']
        self.assertEqual((unknown.status, unknown.detail),
                         ('INFO', '1 timing window skipped layer copies (a full ring): 1 stream rate unknown (pass '
                                  '--host-log to judge them)'))
        self.assertIn('longest refresh gap 42.6 ms (95 Presents/s', unknown.times[0])
        # A line before 10-07 logs no gap: its skips are named, not judged.
        old = judged([timing('10:00:12', 470, 30)])['UI layer copies']
        self.assertEqual((old.status, old.detail),
                         ('INFO', '1 timing window skipped layer copies (a full ring): 1 refresh gap not logged (an '
                                  'add-on before 10-07)'))
        # No skip passes; logs whose timing lines predate the group, or whose layer never copied, have no check.
        clean = judged([timing('10:00:12', 500, 0, gap=0.0), timing('10:00:22', 480, 0)])['UI layer copies']
        self.assertEqual((clean.status, clean.detail), ('PASS', 'no layer copy skipped in 2 timing windows'))
        self.assertNotIn('UI layer copies', judged([timing('10:00:12', 0, 0)]))
        older = line('10:00:12', '[Sunshine 3D] Sunshine Game 3D timing: presents=600 cpu_ms={mean=0.4 max=1.2} '
                                 'gpu_frames=0 gpu_ms mean/max={total=0.00/0.00}')
        self.assertNotIn('UI layer copies', judged([older]))
        # The line's own window length (a runtime reset's last window) and gap; older lines cover 10 s, without a gap.
        session = report.parse(BASE + [timing('10:00:12', 500, 0, window_ms=None),
                                       timing('10:00:18', 1000, 350, gap=3.4, window_ms=5700)])
        self.assertEqual([(w[1], w[8]) for w in session.layer_windows], [(10.0, None), (5.7, 3.4)])

    def test_witcher3_exit_full_ring_is_judged_by_its_refresh_gap(self):
        # The Witcher 3 10-07: the ring first filled at 00:29:19.747 as the quitting game jumped from ~66 to ~590
        # Presents/s; the runtime was destroyed at 00:29:22 before another timing line, so the skipped count was never
        # logged, nor what the skips cost. The once-only full-ring line (a WARN before 10-07) is the UI layer copies
        # check's, not a log warning, and it names those skips as uncounted rather than calling them harmless.
        lines = [
            '00:26:58:777 [54104] | INFO  | Registered add-on "Sunshine 3D" v0.0.0.0 using ReShade API version 20.',
            '00:27:11:511 [50588] | INFO  | [Sunshine 3D] Sunshine Game 3D: add-on GPU renderer ready (no FX file '
            'required)',
            '00:27:11:932 [69628] | INFO  | [Sunshine 3D] Sunshine SBS: generation 1, 7680x2160 full SBS, DXGI 24, '
            'D3D12, PQ HDR10, protocol 4 (source color 3)',
            '00:29:16:357 [69628] | INFO  | [Sunshine 3D] Sunshine Game 3D timing: presents=912 cpu_ms={mean=0.128 '
            'max=1.329} cpu_worst_ms={setup=0.004 depth=1.220 ui=0.005 render=0.052 export=0.048} gpu_frames=0 gpu_ms '
            'mean/max={total=0.000/0.000 inputs=0.000/0.000 source=0.000/0.000 detection=0.000/0.000 '
            'linearize=0.000/0.000 candidate=0.000/0.000 vertical=0.000/0.000 horizontal=0.000/0.000 '
            'eyes=0.000/0.000 pack=0.000/0.000} gpu_profile=disabled dropped_fence_pending=0 dropped_unresolved=0 '
            'incomplete=0 ui_layer={copies=547 skipped=0 offers=912 presents_since_copy={mean=3.51 max=5}}',
            '00:29:19:747 [69628] | WARN  | [Sunshine 3D] Sunshine UI layer: every live copy is offered, held or still '
            'being read; skipping a layer copy (logged once; the timing line counts them)',
            '00:29:22:048 [69628] | INFO  | [Sunshine 3D] Sunshine SBS: export inactive (runtime_reset); waiting for a '
            'valid focused technique',
        ]
        session = report.parse(lines)
        self.assertFalse(session.warnings)
        session.stream_fps = 90.0
        checks = {c.name: c for c in report.evaluate(session)}
        self.assertNotIn('Log warnings', checks)
        self.assertEqual((checks['UI layer copies'].status, checks['UI layer copies'].detail),
                         ('INFO', 'no layer copy skipped in 1 timing window; copies skipped from 00:29:19, after the '
                                  'last timing line, were not counted in any logged window, so their refresh gap is '
                                  "unknown (an add-on before 10-07 dropped a runtime reset's partial window, or the "
                                  'process ended first)'))

        # Since 10-07 the line is INFO and the reset logs its partial window, gap included. The counts below are
        # illustrative, not from this log (~3.6 s of FG-off play at ~66/s, then the ~590/s exit): whatever the
        # window's rates, its verdict follows the gap the skips left.
        def since(gap):
            return lines[:4] + [
                lines[4].replace('WARN ', 'INFO ').replace(
                    'skipping a layer copy (logged once; the timing line counts them)',
                    'skipping a layer copy, the offered copy stays offered (logged once; the timing line counts '
                    'skipped copies and their longest refresh gap)'),
                lines[3].replace('00:29:16:357', '00:29:22:047').replace('presents=912', 'presents=1530').replace(
                    'copies=547 skipped=0 offers=912', 'copies=780 skipped=750 offers=1530').replace(
                    'max=5}}', f'max=9}} skip_gap_ms={gap}}}') + ' window_ms=5690',
                lines[5]]
        for gap, status in (('3.4', 'INFO'), ('15.3', 'WARN')):
            session = report.parse(since(gap))
            session.stream_fps = 90.0
            checks = {c.name: c for c in report.evaluate(session)}
            self.assertNotIn('Log warnings', checks)
            self.assertEqual(checks['UI layer copies'].status, status, checks['UI layer copies'].detail)
            self.assertNotIn('not counted', checks['UI layer copies'].detail)

    def test_ui_layer_targets_dropped_are_parsed_and_reported(self):
        # C12: a timing line's layer group counts the qualifying clears the tracker dropped with every entry taken;
        # its window and gap still parse, and the drops are an INFO check of their own.
        def timing(t, dropped):
            return line(t, '[Sunshine 3D] Sunshine Game 3D timing: presents=600 cpu_ms={mean=0.125 max=0.300} '
                           'cpu_worst_ms={setup=0.004 depth=0.200 ui=0.005 render=0.050 export=0.040} gpu_frames=0 '
                           'gpu_ms mean/max={total=0.000/0.000 inputs=0.000/0.000} gpu_profile=disabled '
                           'dropped_fence_pending=0 dropped_unresolved=0 incomplete=0 ui_layer={copies=600 skipped=0 '
                           'offers=600 presents_since_copy={mean=1.00 max=1} skip_gap_ms=0.0 '
                           f'targets_dropped={dropped}}} window_ms=8000')
        session = report.parse(BASE + [timing('10:00:12', 40), timing('10:00:22', 0)])
        self.assertEqual([w[1] for w in session.layer_windows], [8.0, 8.0])
        self.assertEqual(session.layer_targets_dropped, [(session.layer_targets_dropped[0][0], 40)])
        checks = {c.name: c for c in report.evaluate(session)}
        self.assertEqual(checks['UI layer targets'].status, 'INFO')
        self.assertIn('40 qualifying clears in 1 timing windows', checks['UI layer targets'].detail)
        quiet = report.evaluate(report.parse(BASE + [timing('10:00:12', 0)]))
        self.assertNotIn('UI layer targets', {c.name for c in quiet})

    def test_logs_of_the_removed_shadow_features_still_report(self):
        # The first-run shadow, texel 11's dark pre-UI statistics, rule H2's still screens and the S3 identity shadow
        # were removed (selection revision 9). Their builds' lines still parse; none adds a check of its own, and H2's
        # counted frames are named as removed.
        dark = ui3('10:00:12', 0, 0, (0, 0, 0, 1000), 0x48, 0x0, layer=(0, 30), reason='ambiguous',
                   refused='current', scene=(-0.04, 'hidden'), pre_ui=('layer', 0.18, 1), guard=(1, 0, 0, 1),
                   shadow=1, hidden_ms=300, pre_ui_pixels=(990, 2, 30, 28),
                   still=(1, 0, 'pending', 1200, 900, 920, 325))
        old = [
            line('10:00:01', '[Sunshine 3D] Sunshine UI protection: first-run shadow measures this session '
                             '(UISceneShadow=absent)'),
            line('10:00:01', '[Sunshine 3D] Sunshine UI protection: still screens with no UI source are only logged '
                             '(UIFlattenStillScreens=0)'),
            dark,
            line('10:00:13', '[Sunshine 3D] Sunshine UI still screen: would flatten (UIFlattenStillScreens=0) after '
                             'run_ms=2000 samples=3 d=[0.010,0.040] still_min=0.970'),
            line('10:00:14', '[Sunshine 3D] Sunshine UI still screen: episode ended reason=moved duration_ms=900 '
                             'samples=2 d=[0.010,0.040] still_min=0.960 flattened=0'),
            line('10:00:14', '[Sunshine 3D] Sunshine FG interposers: runtime=0000000000000001 bits=3 '
                             'names=sl_interposer,sl_dlss_g fg_known=1 fg_enabled=1'),
            line('10:00:14', '[Sunshine 3D] Sunshine UI identity: runtime=0000000000000001 renders=9 tagged=8 '
                             'frames={total=4 once=2 missed=1 repeated=1 extra=2} gpu={exact=2 mismatch=1} '
                             'interposers=3'),
        ]
        h2 = {**CLEAN_COUNTERS, 'decided.0': 10, 'decided.11': 10, 'none.ambiguous': 10, 'still.entered': 1,
              'still.released': 1}
        session = report.parse(BASE + old)
        self.assertEqual([u.t for u in session.ui], [36012.0])
        self.assertEqual((session.ui[0].scene.pre_ui_pixels, session.ui[0].scene.hidden_ms), ((990, 2), 300))
        checks = run(BASE + old + [counters('10:00:15', h2, groups=FIX2_COUNTER_GROUPS)])
        for name in ('Dark pre-UI image (shadow)', 'UI still screen', 'UI first-run shadow', 'UI identity (S3)'):
            self.assertNotIn(name, checks)
        self.assertIn('still screen without a UI source (H2, removed) (11) 12%', checks['UI protection'].detail)
        self.assertEqual(checks['UI counters'].status, 'PASS')

    def test_logs_of_the_removed_fix3_and_fix4_builds_still_report(self):
        # Fix 3 and fix 4 (the pre-UI change set and pin only UI) were removed by user decision. Their builds' counter
        # text (decided source 12, refined and the change_set and darkening groups), switch, change-set and darkening
        # lines and the UI line's change_set group still parse, decide nothing and add no check of their own.
        text = ('auto_frames=15 detection_frames=11 held={generated=2 none=1} reused=1 '
                'inactive={no_candidates=1 size=0 unprepared=0} decided={0=2 1=0 2=0 3=0 4=0 5=3 6=1 8=0 10=4 11=5 '
                '12=0} refined=0 none={layer_aside=0 trusted_invalid=0 presented_blocked=0 ambiguous=0 '
                'difference_failed=1 gate_no_hold=1 no_candidate=0 other=0 unaccepted=2} full={6=1 8=0 '
                'depth_not_current=1} full_d={hidden=1 ambiguous=0 visible=0 invalid=0} scene={entered=1 released=1 '
                'refuted=2} still={entered=2 released=1 short=7} change_set={samples=9 retained=5 late=3 '
                'unavailable=1 valid=4 would_refine=3 would_decide=1 pair_verified=4 pair_contradicted=1} '
                'darkening={samples=2 unpinned_samples=1 unpinned_px=40 kept_px=7} '
                'untrusted_inferred=0 inexact_difference=3 contradicted=2 presented_over_dedicated=0 full_alpha=2 '
                'full_alpha_d={hidden=0 ambiguous=0 visible=1 invalid=0} trust={earned=1 revoked_exact=1 '
                'revoked_declared=0 lapsed=0 restored=0 discarded=2 forgotten=3} samples=4 through_ms=12345')
        fields = report.counter_fields('runtime=0000000000000001 ' + text)
        self.assertEqual((fields['decided.12'], fields['refined'], fields['change_set.samples'], fields['still.short']),
                         (0, 0, 9, 7))
        old = line('10:00:12', '[Sunshine 3D] Sunshine UI counters: runtime=0000000000000001 ' + text)
        status = (ui3('10:00:06', 4, 1000, (0, 0, 0, 1000), 0x48, 0x8) +
                  ' change_set={pairing=retained offset=1 valid=1 would_refine=1 refined=0 enabled=0}')
        later = ui3('10:00:07', 4, 1000, (0, 0, 0, 1000), 0x48, 0x8)
        pin = line('10:00:01', '[Sunshine 3D] Sunshine UI protection: dims and unchanged pixels are only logged '
                               '(UIPinOnlyUI=0)')
        session = report.parse([pin, status, later, old])
        self.assertEqual([sample.t for sample in session.ui], [36006.0, 36007.0])
        checks = run(BASE + [pin, status, later, old])
        self.assertNotIn('Pre-UI change set', checks)
        self.assertNotIn('Pin only UI', checks)

    def test_counted_whole_frame_alpha_over_a_visible_scene_is_reported(self):
        # The Witcher 3 sign wheel: a trusted layer covers the frame without an exact pair, so contradicted stays 0;
        # its samples after the first measure the visible scene beneath it. Pinning it flat is intended (P1, the
        # opacity ruling), so it is reported, not warned.
        wheel = {**CLEAN_COUNTERS, 'full_alpha': 40, 'full_alpha_d.visible': 3, 'full_alpha_d.invalid': 1}
        checks = run(BASE + [counters('10:00:12', wheel)])
        self.assertEqual((checks['UI full alpha'].status, checks['UI protection'].status), ('INFO', 'PASS'))
        self.assertIn('40 frames (50% of detection frames) decided a whole-frame alpha', checks['UI full alpha'].detail)
        self.assertIn('visible 3, invalid or unmeasured 1', checks['UI full alpha'].detail)
        self.assertIn('3 samples pinned an accepted whole-frame decision flat over a visible scene, as intended (P1)',
                      checks['UI full alpha'].detail)
        # Counted one-way contradictions without a sampled run reaching the A2 revocation condition are reported.
        unresolved = {**wheel, 'contradicted': 40}
        self.assertEqual(run(BASE + [counters('10:00:12', unresolved)])['UI protection'].status, 'PASS')
        # Clair Obscur Load Game: a whole-frame alpha over a hidden scene is reported, not warned.
        load = {**CLEAN_COUNTERS, 'full_alpha': 40, 'full_alpha_d.hidden': 3, 'full_alpha_d.invalid': 1}
        self.assertEqual(run(BASE + [counters('10:00:12', load)])['UI full alpha'].status, 'INFO')
        self.assertEqual(run(BASE + [counters('10:00:12', CLEAN_COUNTERS)])['UI full alpha'].status, 'PASS')

    def test_counted_known_wrong_cells_warn_with_their_stage(self):
        known = {**CLEAN_COUNTERS, 'untrusted_inferred': 8, 'inexact_difference': 4}
        checks = run(BASE + [counters('10:00:12', known)])
        # Since S1 only accepted candidates decide: unaccepted inferred alpha deciding is a defect.
        self.assertEqual(checks['UI inferred alpha'].status, 'FAIL')
        self.assertIn('8 frames (10% of detection frames)', checks['UI inferred alpha'].detail)
        self.assertIn('only accepted candidates decide since S1', checks['UI inferred alpha'].detail)
        # A Present-counted pair's partial difference is validated by its own pixels (V2): informative.
        self.assertEqual(checks['UI inexact difference'].status, 'INFO')
        self.assertIn('validated by its own pixels (V2)', checks['UI inexact difference'].detail)
        # Counters logged before S1 let the untrusted pass decide.
        before = run(BASE + [counters('10:00:12', {**S1_CLEAN_COUNTERS, 'untrusted_inferred': 8},
                                      groups=S0_COUNTER_GROUPS)])
        self.assertEqual(before['UI inferred alpha'].status, 'WARN')
        self.assertIn('expected before S1', before['UI inferred alpha'].detail)
        self.assertIn('opaque proof set 0', before['UI trust events'].detail)
        self.assertIn('legacy entries discarded 0', checks['UI trust events'].detail)
        clean = run(BASE + [counters('10:00:12', CLEAN_COUNTERS)])
        self.assertEqual((clean['UI inferred alpha'].status, clean['UI inexact difference'].status), ('PASS', 'PASS'))

    @staticmethod
    def gate(t, batch, real, generated=0):
        """A 'Sunshine UI capture gate' line's HUD-less pairing counts (game3d_ui_input_provider.cpp)."""
        return line(t, '[Sunshine 3D] Sunshine UI capture gate: runtime=0000000000000001 request_generation=1 '
                       'wanted={epoch=4 revision=4 viewport=0 device=5 size=3840x2160 kinds=0xf} hook={state=admitted '
                       'epoch=4 revision=4 viewport=0 sequence=1 tick=1 kinds=0x8 matches=1} boundary=1 attempted=1 '
                       f'recorded=1 hudless_presents={{batch={batch} real={real} late=0 generated={generated} stale=0 '
                       'other=0 none=0}; gate metadata does not authorize pixels')

    def test_hogwarts_hudless_only_session(self):
        # Hogwarts Legacy 10-05 tagged only HUDLessColor: no same-batch Backbuffer pair, so its exact full change-sets
        # (6) came from Present-counted pairs, and whole gameplay seconds were flat. Only a same-batch pair is exact.
        flat = {**CLEAN_COUNTERS, 'decided.10': 50, 'decided.6': 10}
        counted = run(BASE + [self.gate('10:00:06', 0, 551), counters('10:00:12', flat, groups=S2B_COUNTER_GROUPS)])
        self.assertEqual(counted['UI full frame pairing'].status, 'FAIL')
        self.assertIn('no same-batch Backbuffer pair (551 Present-counted HUD-less pairings, 0 same-batch)',
                      counted['UI full frame pairing'].detail)
        batch = run(BASE + [self.gate('10:00:06', 300, 39), counters('10:00:12', flat, groups=S2B_COUNTER_GROUPS)])
        self.assertNotIn('UI full frame pairing', batch)
        # FG on at 4x: every Present held without a real-frame decision. Counters advance only when a sample commits,
        # so the window runs from the counter line that last advanced.
        fg_on = line('10:00:05', '[Sunshine 3D] Sunshine Streamline frame generation: viewport=0 mode=1 '
                                 'generated_frames=3 supported=1 success=1')
        start = {'auto_frames': 100, 'detection_frames': 100, 'samples': 3}
        held = {'auto_frames': 1100, 'detection_frames': 140, 'held.none': 960, 'samples': 4}
        checks = run(BASE + [fg_on, counters('10:00:06', start), counters('10:00:10', start),
                             counters('10:00:20', held)])
        check = checks['UI holds without a decision']
        self.assertEqual(check.status, 'FAIL')
        self.assertEqual(check.times,
                         ['10:00:06-10:00:20 (14 s) FG on: 96% of 1000 Auto frames held without a decision'])
        brief = {'auto_frames': 200, 'detection_frames': 140, 'held.none': 60, 'samples': 4}
        warned = run(BASE + [counters('10:00:06', start), counters('10:00:09', brief)])['UI holds without a decision']
        self.assertEqual((warned.status, warned.times[0][:26]), ('WARN', '10:00:06-10:00:09 (3 s) FG'))
        self.assertEqual(run(BASE + [counters('10:00:12', CLEAN_COUNTERS)])['UI holds without a decision'].status,
                         'PASS')
        # The gap's pending lines carry their own FG state and the last sample's reason.
        unprotected = ui2('10:00:06', 0, 0, (0, 0, 0, 1000), 0x48, 0x0, layer=(1000, 0), reason='gate_no_hold',
                          refused='ui_layer')
        searching = [ui2(t, 0, 0, (0, 0, 0, 0), 0x0, 0x0, detection='searching', fg=1)
                     for t in ('10:00:08', '10:00:14')]
        gaps = run(BASE + [unprotected] + searching)['UI protection gaps']
        self.assertTrue(gaps.times[0].endswith('FG on: searching; last sample: UI layer covers 100% (not accepted); '
                                               'current alpha covers 100% (not accepted) (reason gate no hold: UI '
                                               'layer)'), gaps.times)

    def test_session_end_cost_and_settling_status(self):
        # An Unreal game ends its process after ReShade tore its runtimes down but before ReShade's own exit line.
        teardown = line('10:00:16', 'Destroyed runtime environment on runtime 00000001F5873510 (ReShade.ini).')
        self.assertEqual(report.parse(BASE + [teardown]).teardown, 36016.0)
        # Every timing window counts: Present-weighted CPU mean and the maximum over all windows.
        timing = [line(t, f'[Sunshine 3D] Sunshine Game 3D timing: presents={n} cpu_ms={{mean={mean} max={peak}}} '
                          'gpu_frames=0 gpu_ms mean/max={total=0.00/0.00} gpu_profile=disabled')
                  for t, n, mean, peak in (('10:00:06', 100, '0.40', '900.0'), ('10:00:16', 300, '0.20', '0.5'))]
        self.assertEqual(run(BASE + timing)['Game 3D cost'].detail,
                         'CPU 0.25 ms mean (900.0 max) over 400 Presents; GPU timing off (Diagnostics=0)')
        # A capture status while the export starts is the source settling; later ones warn.
        settling = line('10:00:03', '[Sunshine 3D] Sunshine Streamline depth: conflicting_state; viewport=0')
        later = line('10:00:12', '[Sunshine 3D] Sunshine Streamline depth: conflicting_state; viewport=0')
        self.assertEqual(run(BASE + [settling])['Capture status'].status, 'INFO')
        status = run(BASE + [settling, later])['Capture status']
        self.assertEqual((status.status, status.detail, status.times),
                         ('WARN', 'conflicting_state 1; 1 more while settling',
                          ['10:00:12 Streamline conflicting_state']))
        # Every API depth provider's statuses count (Stellar Blade's depth comes through NGX); the NGX periodic
        # summary line is no status.
        ngx = line('10:00:13', '[Sunshine 3D] Sunshine NGX depth: incomplete_state; viewport=0')
        api = line('10:00:14', '[Sunshine 3D] Sunshine API depth: conflicting_state; capture=1; using generic depth '
                               'fallback')
        summary = line('10:00:15', '[Sunshine 3D] Sunshine NGX depth: confirmed_features=1 capture_eligible=1 '
                                   'recovered_features=0 evaluations=4 nominations=4 copy_recorded=4 metadata_only=0')
        status = run(BASE + [ngx, api, summary])['Capture status']
        self.assertEqual((status.status, status.times),
                         ('WARN', ['10:00:13 NGX incomplete_state', '10:00:14 API conflicting_state']))

    def test_host_lines_from_another_session_are_not_counted(self):
        with TemporaryDirectory() as folder:
            host = Path(folder) / 'sunshine.log'
            host.write_text('[2026-10-01 09:00:00.000]: Info: ReShade SBS connected: process 1, 7680x2160\n'
                            '[2026-10-01 10:00:05.000]: Warning: NvEnc: picture waited 120 ms; '
                            'input_producer=done_by_99ms (encoder-side delay)\n', encoding='utf-8')
            checks = {c.name: c for c in report.host_checks(host, datetime(2026, 10, 1, 10, 0, 0),
                                                            datetime(2026, 10, 1, 10, 0, 20))}
            self.assertEqual(checks['Host link'].status, 'WARN')
            self.assertEqual(checks['Encoder stalls'].times, ['10:00:05 encoder-side'])
            other = report.host_checks(host, datetime(2026, 9, 30, 23, 0), datetime(2026, 9, 30, 23, 5))
            self.assertEqual([c.name for c in other], ['Host log'])

    def test_scaled_and_fitted_host_sizes_are_info_and_a_refused_one_fails(self):
        with TemporaryDirectory() as folder:
            host = Path(folder) / 'sunshine.log'
            host.write_text(
                '[2026-10-01 10:00:01.000]: Info: ReShade SBS connected: process 1, 3840x1080\n'
                "[2026-10-01 10:00:02.000]: Info: ReShade SBS: game eyes 1920x1200 are fitted into the stream's "
                '1920x1080 eyes at 1728x1080+96+0 (pillarboxed, different aspect ratio). Run the game at 1920x1080 to '
                'fill the stream.\n'
                "[2026-10-01 10:00:03.000]: Info: ReShade SBS: game eyes 2560x1440 are scaled to the stream's "
                '1920x1080 eyes. Run the game at 1920x1080 for native sharpness.\n'
                "[2026-10-01 10:00:04.000]: Warning: ReShade SBS: game eyes 1920x1080 cannot fill the stream's "
                '1920x3240 eyes: an undistorted fit would cover less than half of each eye (or the raster is too '
                'small or odd); staying in 2D.\n', encoding='utf-8')
            sizes = [c for c in report.host_checks(host, datetime(2026, 10, 1, 10, 0, 0),
                                                   datetime(2026, 10, 1, 10, 0, 20)) if c.name == 'Host size']
            self.assertEqual([c.status for c in sizes], ['INFO', 'INFO', 'FAIL'])
            self.assertIn('fitted into', sizes[0].detail)

    def test_an_nvenc_stall_counts_once_by_its_last_line(self):
        # E33 10-06: each stalled frame logged its warning and then its completion line, and the report counted 6
        # stalls for 3 frames. A stall is one per frame, classified by its last line; frame numbers far apart in time
        # (a new encoder session) are separate stalls.
        with TemporaryDirectory() as folder:
            host = Path(folder) / 'sunshine.log'
            host.write_text(
                '[2026-10-01 10:00:01.000]: Info: ReShade SBS connected: process 1, 7680x2160\n'
                '[2026-10-01 10:00:05.000]: Warning: NvEnc: frame 120 exceeded the 100 ms completion wait; '
                'its input is still being produced; input_producer=pending_at_101ms (upstream GPU delay)\n'
                '[2026-10-01 10:00:05.200]: Info: NvEnc: frame 120 completed after soft timeout in 180 ms; '
                'input_producer=done_by_99ms (finished before the first check; encoder or upstream)\n'
                '[2026-10-01 10:00:09.000]: Warning: NvEnc: frame 340 exceeded the 100 ms completion wait; '
                'allowing up to 250 ms total; input_producer=done_at_120ms (upstream GPU delay)\n'
                '[2026-10-01 10:00:09.100]: Error: NvEnc: frame 340 encode wait timeout after 260 ms; x; '
                'input_producer=done_at_120ms (upstream GPU delay)\n'
                '[2026-10-01 10:00:18.000]: Info: NvEnc: frame 120 completed after soft timeout in 140 ms; '
                'input_producer=done_by_99ms (finished before the first check; encoder or upstream)\n',
                encoding='utf-8')
            checks = {c.name: c for c in report.host_checks(host, datetime(2026, 10, 1, 10, 0, 0),
                                                            datetime(2026, 10, 1, 10, 0, 20))}
            stalls = checks['Encoder stalls']
            self.assertEqual((stalls.status, stalls.detail), ('WARN', '3'))
            # Each lists its length; frame 340's encoder ran 140 ms past its input before timing out.
            self.assertEqual(stalls.times, ['10:00:05 encoder-side 180 ms',
                                            '10:00:09 upstream GPU + encoder-side 260 ms timed out',
                                            '10:00:18 encoder-side 140 ms'])

    def test_an_upstream_stall_at_the_games_fg_switch_or_reset_is_info(self):
        # W3 10-07: the game lost focus and switched frame generation off; the GPU then ran the host's conversion
        # 165 ms late, and the encoder finished as soon as its input did. The game's own GPU reconfiguration delayed
        # one picture by design. Only a completed upstream stall near such a line is explained: one with no
        # transition near it (E33 10-06), an encoder-side one or a timeout still warns.
        w3 = ('[2026-10-07 20:57:32.474]: Warning: NvEnc: frame 12195 exceeded the 100 ms completion wait; its input '
              'is still being produced, so the encoder\'s 250 ms budget starts when the input completes (2000 ms at '
              'most); input_producer=pending_at_100ms (upstream GPU delay)\n'
              '[2026-10-07 20:57:32.539]: Info: NvEnc: frame 12195 completed after soft timeout in 165 ms; '
              'input_producer=done_at_165ms (upstream GPU delay)\n')
        lines = [
            '20:57:32:276 [43624] | INFO  | [Sunshine 3D] Sunshine SBS: export inactive (not_foreground); waiting '
            'for a valid focused technique',
            '20:57:32:302 [43624] | INFO  | [Sunshine 3D] Sunshine Streamline frame generation: viewport=348957 '
            'mode=0 generated_frames=1 supported=1 success=1',
        ]
        transitions = report.gpu_transitions(report.parse(lines))
        self.assertEqual([(round(t, 3), label) for t, label in transitions], [(75452.302, 'FG off')])
        start, end = datetime(2026, 10, 7, 20, 55, 21), datetime(2026, 10, 7, 20, 57, 40)
        with TemporaryDirectory() as folder:
            host = Path(folder) / 'sunshine.log'
            host.write_text(w3, encoding='utf-8')
            stalls = {c.name: c for c in report.host_checks(host, start, end, transitions)}['Encoder stalls']
            self.assertEqual((stalls.status, stalls.times),
                             ('INFO', ['20:57:32 upstream GPU 165 ms at FG off 20:57:32']))
            self.assertIn('of at most 250 ms', stalls.detail)
            # Focus loss alone explains nothing: the host's own switch to the desktop is its work.
            focus = report.gpu_transitions(report.parse(lines[:1]))
            stalls = {c.name: c for c in report.host_checks(host, start, end, focus)}['Encoder stalls']
            self.assertEqual((stalls.status, stalls.times), ('WARN', ['20:57:32 upstream GPU 165 ms']))
            # A runtime reset 1.9 s after the stall explains it (the add-on's line can follow the event); 2.1 s
            # before it does not.
            reset = [(75454.4, 'runtime reset')]
            stalls = {c.name: c for c in report.host_checks(host, start, end, reset)}['Encoder stalls']
            self.assertEqual(stalls.times, ['20:57:32 upstream GPU 165 ms at runtime reset 20:57:34'])
            stalls = {c.name: c for c in report.host_checks(host, start, end, [(75450.3, 'FG on')])}['Encoder stalls']
            self.assertEqual(stalls.status, 'WARN')
            # E33 10-06: upstream stalls with no FG switch or reset near them still warn, beside an explained one; so
            # do an encoder-side stall and a timeout at the switch itself.
            host.write_text(
                w3
                + '[2026-10-07 20:57:36.000]: Warning: NvEnc: frame 12500 exceeded the 100 ms completion wait; '
                'input_producer=pending_at_100ms (upstream GPU delay)\n'
                '[2026-10-07 20:57:36.103]: Info: NvEnc: frame 12500 completed after soft timeout in 203 ms; '
                'input_producer=done_at_203ms (upstream GPU delay)\n'
                '[2026-10-07 20:57:38.000]: Warning: NvEnc: frame 12700 exceeded the 100 ms completion wait; '
                'allowing up to 250 ms total; input_producer=done_by_99ms (finished before the first check; encoder '
                'or upstream)\n', encoding='utf-8')
            transitions = [(75452.302, 'FG off'), (75459.0, 'FG on')]
            stalls = {c.name: c for c in report.host_checks(host, start, end, transitions)}['Encoder stalls']
            self.assertEqual((stalls.status, stalls.detail, stalls.times),
                             ('WARN', "2; 1 more upstream GPU at the game's FG switches or resets",
                              ['20:57:36 upstream GPU 203 ms', '20:57:38 encoder-side at FG on 20:57:39']))
            host.write_text(
                '[2026-10-07 20:57:32.474]: Warning: NvEnc: frame 12195 exceeded the 100 ms completion wait; '
                'input_producer=pending_at_100ms (upstream GPU delay)\n'
                '[2026-10-07 20:57:34.474]: Error: NvEnc: frame 12195 encode wait timeout after 2000 ms; x; '
                'input_producer=pending_at_2000ms (upstream GPU delay)\n', encoding='utf-8')
            stalls = {c.name: c for c in report.host_checks(host, start, end, transitions)}['Encoder stalls']
            self.assertEqual((stalls.status, stalls.times),
                             ('WARN', ['20:57:32 upstream GPU 2000 ms timed out at FG off 20:57:32']))

    def test_a_transition_explains_only_a_short_completed_upstream_stall(self):
        # Review of the 10-07 rule: a transition explains a picture frozen no longer than the encoder's 250 ms
        # budget, and only when the encoder finished within two 25 ms polls of its input. Stellar Blade 10-06 froze
        # 690 ms at FG on; an encoder that ran 245 ms past its input, a removed device or a failed wait at the switch,
        # and a stall the log never ends all still warn, each with its length and the transition named.
        start, end = datetime(2026, 10, 7, 20, 55, 21), datetime(2026, 10, 7, 20, 57, 59)
        transitions = [(75452.302, 'FG off')]
        warn = ('[2026-10-07 20:57:32.474]: Warning: NvEnc: frame 12195 exceeded the 100 ms completion wait; its input '
                'is still being produced; input_producer=pending_at_100ms (upstream GPU delay)\n')
        cases = {
            'Info: NvEnc: frame 12195 completed after soft timeout in 250 ms; input_producer=done_at_210ms '
            '(upstream GPU delay)': ('INFO', '20:57:32 upstream GPU 250 ms at FG off 20:57:32'),
            'Info: NvEnc: frame 12195 completed after soft timeout in 690 ms; input_producer=done_at_690ms '
            '(upstream GPU delay)': ('WARN', '20:57:32 upstream GPU 690 ms at FG off 20:57:32'),
            'Info: NvEnc: frame 12195 completed after soft timeout in 350 ms; input_producer=done_at_105ms '
            '(upstream GPU delay)': ('WARN', '20:57:32 upstream GPU + encoder-side 350 ms at FG off 20:57:32'),
            'Info: NvEnc: frame 12195 completed after soft timeout in 160 ms; input_producer=done_at_105ms '
            '(upstream GPU delay)': ('WARN', '20:57:32 upstream GPU + encoder-side 160 ms at FG off 20:57:32'),
            'Error: NvEnc: frame 12195 device removed during encode wait after 520 ms; native_wait=0x00000000, '
            'native_error=0x00000000, device_removed_reason=0x887a0006': (
                'WARN', '20:57:32 upstream GPU 520 ms failed at FG off 20:57:32'),
            'Error: NvEnc: frame 12195 completion event wait failed after 130 ms; native_wait=0xffffffff': (
                'WARN', '20:57:32 upstream GPU 130 ms failed at FG off 20:57:32'),
            None: ('WARN', '20:57:32 upstream GPU at FG off 20:57:32'),
        }
        with TemporaryDirectory() as folder:
            host = Path(folder) / 'sunshine.log'
            for last, expected in cases.items():
                host.write_text(warn + (f'[2026-10-07 20:57:33.000]: {last}\n' if last else ''), encoding='utf-8')
                stalls = {c.name: c for c in report.host_checks(host, start, end, transitions)}['Encoder stalls']
                self.assertEqual((stalls.status, stalls.times[0]), expected, last)
            # A failure with no warning before it is a stall of its own.
            host.write_text('[2026-10-07 20:57:32.600]: Error: NvEnc: frame 12195 device removed during encode wait '
                            'after 90 ms; native_wait=0x00000000\n', encoding='utf-8')
            stalls = {c.name: c for c in report.host_checks(host, start, end, transitions)}['Encoder stalls']
            self.assertEqual((stalls.status, stalls.times),
                             ('WARN', ['20:57:32 unknown 90 ms failed at FG off 20:57:32']))

    def test_a_stall_names_its_nearest_transition(self):
        # Stellar Blade 10-06: the stall at 13:45:56.980 came 1.49 s after a runtime reset and 0.59 s after FG on.
        start, end = datetime(2026, 10, 6, 13, 45, 0), datetime(2026, 10, 6, 13, 47, 0)
        transitions = [(13 * 3600 + 45 * 60 + 55.417, 'runtime reset'), (13 * 3600 + 45 * 60 + 56.390, 'FG on')]
        with TemporaryDirectory() as folder:
            host = Path(folder) / 'sunshine.log'
            host.write_text(
                '[2026-10-06 13:45:56.980]: Warning: NvEnc: frame 4211 exceeded the 100 ms completion wait; '
                'input_producer=pending_at_99ms (upstream GPU delay)\n'
                '[2026-10-06 13:45:56.993]: Info: NvEnc: frame 4211 completed after soft timeout in 112 ms; '
                'input_producer=done_at_112ms (upstream GPU delay)\n', encoding='utf-8')
            stalls = {c.name: c for c in report.host_checks(host, start, end, transitions)}['Encoder stalls']
            self.assertEqual((stalls.status, stalls.times),
                             ('INFO', ['13:45:56 upstream GPU 112 ms at FG on 13:45:56']))

    def test_e33_title_depth_pause_and_scene_wait_are_the_games_own(self):
        # E33 10-22 session B, the title screen (real lines): NGX evaluated DLSS while the scene behind the title
        # rendered, so its depth was fresh, but the raw controller accepted no sample from it for 20 s. Leaving the
        # title, the game ran no DLSS for 828 ms (no active depth view) and resumed with its own history reset
        # (current_reset=1); the controller then had its reference within a second. The pause and the wait are the
        # game's own: INFO, each listed.
        checks = run(E33B_TITLE)
        self.assertEqual(checks['Depth'].status, 'PASS')
        self.assertEqual((checks['Depth paused by the game'].status, checks['Depth paused by the game'].times),
                         ('INFO', ['14:35:30.766-14:35:31.598 (828 ms) retained_depth_expired, NGX '
                                   'selection=inactive_views (the game supplied no depth)']))
        self.assertEqual(checks['Placement flat'].status, 'PASS')
        waited = checks['Placement waited for the scene']
        self.assertEqual((waited.status, waited.times),
                         ('INFO', ['14:35:10.016-14:35:32.770 raw calibrating (21.9 s with depth), game scene reset '
                                   '14:35:31.598']))
        # Without the game's reset neither is the game's own: the loss is a depth gap and the run a flat window.
        no_reset = [x.replace('current_reset=1', 'current_reset=0') for x in E33B_TITLE]
        checks = run(no_reset)
        self.assertEqual(checks['Depth'].status, 'WARN')
        self.assertNotIn('Depth paused by the game', checks)
        self.assertEqual(checks['Placement flat'].status, 'WARN')
        self.assertNotIn('Placement waited for the scene', checks)
        # A loss the add-on had a depth view for (its own capture failing) is not the game pausing DLSS.
        own = [x.replace('selection=inactive_views', 'selection=completed_already_consumed') for x in E33B_TITLE]
        self.assertEqual(run(own)['Depth'].status, 'WARN')
        # Placement more than the settle time (and the log gate) after the reset did not come from the new scene.
        late = [x.replace('14:35:32:770', '14:35:35:700') if 'raw automation: ready' in x else x for x in E33B_TITLE]
        self.assertEqual(run(late)['Placement flat'].status, 'WARN')
        # A controller in an error state was not waiting for a reference.
        error = E33B_TITLE[:4] + [E33B_TITLE[3].replace('14:35:10:016', '14:35:20:000').replace(
            'automation: calibrating;', 'automation: invalid_source;')] + E33B_TITLE[4:]
        checks = run(error)
        self.assertEqual(checks['Placement flat'].status, 'WARN')
        self.assertNotIn('Placement waited for the scene', checks)

    def test_frame_generation_sessions_count_reused_depth_as_depth(self):
        # E33 10-22 session A at 4x frame generation (real lines): a quarter of the publications carry fresh depth
        # and the rest bounded reuse of it. The depth checks began only once more than half were fresh, so a session
        # that started with FG on had none, and its 13:47:15 loss went unreported.
        checks = run(E33_DEPTH_LOSS)
        self.assertEqual((checks['Depth'].status, checks['Depth'].times),
                         ('WARN', ['13:47:15.502-13:47:17.373 (1859 ms) retained_depth_expired, Streamline '
                                   'selection=inactive_views (the game supplied no depth)']))
        self.assertIn('fresh from 13:47:14', checks['Depth'].detail)
        # Reused depth alone, with no fresh depth at all, has not started.
        reused = [x.replace('published_fresh_depth=476', 'published_fresh_depth=336') for x in E33_DEPTH_LOSS]
        self.assertEqual(run(reused)['Depth'].status, 'FAIL')

    def test_a_slow_present_without_a_named_step_is_a_hitch(self):
        # E33 10-22 session B (real lines): the window to 14:36:27 had a 47 ms Present in its render section, which
        # had no named step, and the report said 'Present hitches: none'. An FG switch lies in that window, so it is
        # expected there, and listed.
        checks = run(BASE + E33B_RENDER_HITCH)
        self.assertEqual((checks['Present hitches'].status, checks['Present hitches'].times),
                         ('PASS', ['14:36:27 render section (no step named; the slowest Present of 14:36:17-14:36:27) '
                                   '47.3 ms']))
        # Without a transition in the window it warns.
        checks = run(BASE + E33B_RENDER_HITCH[1:])
        self.assertEqual(checks['Present hitches'].status, 'WARN')
        self.assertEqual(checks['Present hitches'].times,
                         ['14:36:27 render section (no step named; the slowest Present of 14:36:17-14:36:27) 47.3 ms'])
        # A named step in the window accounts for its slowest Present.
        named = line('14:36:20', '[Sunshine 3D] Sunshine Game 3D hitch: stereo render took 47.0 ms on the present '
                                 'thread vram_local_mib=9000/14000 nonlocal_mib=200', 'WARN')
        checks = run(BASE + E33B_RENDER_HITCH[1:] + [named])
        self.assertEqual(checks['Present hitches'].times, ['14:36:20 stereo render 47.0 ms'])
        # A window whose slowest Present stayed under the hitch threshold is none.
        quick = [E33B_RENDER_HITCH[1].replace('max=47.306', 'max=7.990')]
        self.assertEqual(run(BASE + quick)['Present hitches'].detail, 'none')

    def test_vram_samples_warn_only_at_events_under_pressure(self):
        # The add-on's 10 s VRAM line (game3d_vram.h, its unit test's exact format) and the latest sample on a depth
        # loss: a loss at 15350 of a 15400 MiB budget came under pressure; the window alone (over budget for 250 ms,
        # with nothing visible) is INFO. Logs before 10-22 have neither check.
        window = line('10:00:12', '[Sunshine 3D] Sunshine Game 3D VRAM: local_mib={usage=15100 budget=15000} '
                                  'nonlocal_mib={usage=400 budget=60000} window={samples=2 max_usage_mib=15100 '
                                  'min_budget_mib=15000 min_headroom_mib=-100 over_budget_ms=250 budget_changes=0} '
                                  'window_ms=10000')
        lost = line('10:00:13', '[Sunshine 3D] Sunshine depth readiness: lost reason=retained_depth_expired '
                                'runtime=0x1 present=10 unavailable_ms=0 suppressed_episodes=0; provider=NGX '
                                'selection=inactive_views status=stale consumer=x consumer_loss=none cookie=0 '
                                'slot_invalid=0 vram_local_mib=15350/15400 nonlocal_mib=410')
        checks = run(BASE + [window])
        self.assertNotIn('VRAM pressure', checks)
        self.assertEqual(checks['Video memory'].status, 'INFO')
        self.assertIn('peaked at 15100 MiB', checks['Video memory'].detail)
        self.assertIn('as low as -100 MiB; 250 ms over budget', checks['Video memory'].detail)
        checks = run(BASE + [window, lost])
        self.assertEqual((checks['VRAM pressure'].status, checks['VRAM pressure'].times),
                         ('WARN', ['10:00:13.000 depth loss at 15350/15400 MiB']))
        roomy = lost.replace('15350/15400', '14000/15400')
        self.assertNotIn('VRAM pressure', run(BASE + [window, roomy]))
        # A recovery line's sample is not an event.
        recovered = lost.replace('lost reason=retained_depth_expired', 'recovered reason=fresh_copy')
        self.assertNotIn('VRAM pressure', run(BASE + [window, recovered]))
        self.assertNotIn('Video memory', run(BASE))
        gap = report.parse([line('10:00:14', '[Sunshine 3D] Sunshine Game 3D present gap: 343 ms since the previous '
                                             'Present; fg=1 foreground=0 suppressed=0 vram_local_mib=9000/14000 '
                                             'nonlocal_mib=200')])
        self.assertEqual(gap.present_gaps, [(36014.0, 343.0, True, False)])
        self.assertEqual(gap.vram_points, [(36014.0, 9000, 14000, 'present gap')])

    def test_an_upstream_stall_inside_the_games_own_freeze_is_info(self):
        # E33 10-22 session B (real host lines): frames 5931 and 8615 waited 229 and 341 ms for the host's own
        # conversion; the game presented nothing over either stall (240 and 343 ms Present gaps, the second as it lost
        # focus with FG on). The add-on now logs such gaps; a stall a game gap covers is the game holding the whole
        # GPU, unless video memory was under pressure at it. Before the gap lines both warned, as they still do here.
        start, end = datetime(2026, 10, 22, 14, 35, 2), datetime(2026, 10, 22, 14, 37, 4)
        fg_off = [(14 * 3600 + 36 * 60 + 30.880, 'FG off')]

        def gap(t, ms, foreground=1, vram='9618/14000'):
            return (f'{t} [41592] | INFO  | [Sunshine 3D] Sunshine Game 3D present gap: {ms} ms since the previous '
                    f'Present; fg=1 foreground={foreground} suppressed=0 vram_local_mib={vram} nonlocal_mib=208')
        gaps = [gap('14:35:58:940', 240), gap('14:36:30:861', 343, foreground=0)]

        def stalls(host_lines, game_lines):
            with TemporaryDirectory() as folder:
                host = Path(folder) / 'sunshine.log'
                host.write_text('\n'.join(host_lines) + '\n', encoding='utf-8')
                memory = report.game_memory(report.parse(game_lines))
                return {c.name: c for c in report.host_checks(host, start, end, fg_off, memory)}
        before = stalls(E33B_HOST_STALLS, [])['Encoder stalls']
        self.assertEqual((before.status, before.times),
                         ('WARN', ['14:35:58 upstream GPU 229 ms', '14:36:30 upstream GPU 341 ms at FG off 14:36:30']))
        froze = stalls(E33B_HOST_STALLS, gaps)['Encoder stalls']
        self.assertEqual((froze.status, froze.times),
                         ('INFO', ['14:35:58 upstream GPU 229 ms during a 240 ms game freeze (FG on before it)',
                                   '14:36:30 upstream GPU 341 ms at FG off 14:36:30 during a 343 ms game freeze (FG on '
                                   'before it, not foreground)']))
        self.assertIn('the game itself presented nothing', froze.detail)
        # A gap shorter than the stall, or one that ended before it did, does not cover it.
        for partial in (gap('14:35:58:940', 120), gap('14:35:58:800', 240)):
            checks = stalls(E33B_HOST_STALLS, [partial, gaps[1]])['Encoder stalls']
            self.assertEqual((checks.status, checks.times[0]), ('WARN', '14:35:58 upstream GPU 229 ms'))
        # Video memory under pressure at the stall, the game's or the host's, keeps it a warning.
        squeezed = stalls(E33B_HOST_STALLS, [gap('14:35:58:940', 240, vram='15350/15400'), gaps[1]])
        squeezed = squeezed['Encoder stalls']
        self.assertEqual((squeezed.status, squeezed.times[0]),
                         ('WARN', '14:35:58 upstream GPU 229 ms during a 240 ms game freeze (FG on before it) '
                                  'game VRAM 15350/15400 MiB'))
        host_vram = [x + ' host_vram_mib={local=5990/6000 nonlocal=300}' if 'frame 8615' in x else x
                     for x in E33B_HOST_STALLS]
        checks = stalls(host_vram, gaps)
        self.assertEqual((checks['Encoder stalls'].status, checks['Encoder stalls'].times),
                         ('WARN', ['14:36:30 upstream GPU 341 ms at FG off 14:36:30 during a 343 ms game freeze (FG on '
                                   'before it, not foreground) host VRAM 5990/6000 MiB']))
        self.assertEqual(checks['Host video memory'].detail, 'host local usage peaked at 5990 MiB; its headroom was as '
                                                             'low as 10 MiB (5990/6000 MiB at 14:36:30)')
        # An encoder that ran past its input is not explained by the game's freeze.
        slow = [x.replace('done_at_225ms', 'done_at_100ms') for x in E33B_HOST_STALLS]
        checks = stalls(slow, gaps)['Encoder stalls']
        self.assertEqual((checks.status, checks.times),
                         ('WARN', ['14:35:58 upstream GPU + encoder-side 229 ms']))


def source(name):
    """An add-on source file beside this test."""
    return (Path(__file__).resolve().parent / name).read_text(encoding='utf-8')


def enum_names(text, enum):
    """The enumerators of `enum class <enum> { ... }`, in order."""
    body = re.search(rf'enum class {enum} \{{([^}}]*)\}}', text).group(1)
    return tuple(name.strip() for name in body.split(',') if name.strip())


class SchemaMirrors(unittest.TestCase):
    """The report's copies of the add-on's names, checked against the C++ that owns them: a name added, renamed or
    removed there fails here until it is classified (in the report, or for a failure in the lists below)."""

    # The report fails every capture status outside CAPTURE_OK and warns on every export pause outside
    # ROUTINE_INACTIVE, a new name included. These lists name today's failures so that a status or pause reason
    # added in the C++ fails this test until someone decides its class.
    CAPTURE_FAIL = frozenset((
        'malformed', 'unsupported_state', 'missing_state', 'conflicting_state', 'incomplete_state',
        'unsupported_lifetime', 'unsupported_resource', 'unsupported_queue', 'exhausted', 'failed'))
    FAULT_PAUSES = frozenset((
        'consumer_protocol', 'consumer_undeclared', 'deferred_command_list', 'device_removed', 'fence_signal_failed',
        'generation_limit', 'native_depth_unprepared', 'output_unavailable', 'overlay_capture_failed',
        'overlay_prepare_failed', 'present_not_finished', 'renderer_unavailable', 'ring_creation_failed',
        'source_unavailable', 'unsupported_api'))

    def test_export_pauses(self):
        text = source('exporter.cpp')
        routine = re.search(r'routine_export_pauses\{([^}]*)\}', text).group(1)
        self.assertEqual(report.ROUTINE_INACTIVE, frozenset(re.findall(r'"(\w+)"', routine)))
        # Every reason the exporter pauses with is routine or a listed fault, never both, and every call names its
        # reason as a literal.
        calls = re.findall(r'\bdeactivate\([^;]*\);', text)
        self.assertTrue(calls)
        self.assertEqual([call for call in calls if '"' not in call], [])
        paused = set(re.findall(r'"(\w+)"', ' '.join(calls)))
        self.assertEqual(sorted(report.ROUTINE_INACTIVE | self.FAULT_PAUSES), sorted(paused))
        self.assertFalse(report.ROUTINE_INACTIVE & self.FAULT_PAUSES)
        # The pause line's format, filled in, is the line the report reads.
        line_format = re.search(r'"(Sunshine SBS: export inactive [^"]*)"', text).group(1)
        found = report.INACTIVE.search(line_format.replace('%s', 'ring_replaced', 1).replace('%s', 'routine', 1))
        self.assertEqual(found.groups(), ('ring_replaced', 'routine'))

    def test_capture_statuses(self):
        names = enum_names(source('streamline_depth_capture.h'), 'status')
        self.assertEqual(sorted(report.CAPTURE_OK | self.CAPTURE_FAIL), sorted(names))
        self.assertFalse(report.CAPTURE_OK & self.CAPTURE_FAIL)
        self.assertEqual(len(set(names)), len(names))

    def test_observation_loss_causes(self):
        text = source('observation_loss_journal.h')
        names = enum_names(text, 'reason')
        classified = report.LOSS_CAPACITY + report.LOSS_ROUTINE + report.LOSS_FAULT
        self.assertEqual(sorted(classified), sorted(names))
        self.assertEqual(len(set(classified)), len(classified))
        self.assertEqual(sorted(re.findall(r'SUNSHINE_LOSS_NAME\((\w+)\);', text)), sorted(names))

    def test_ui_counter_line(self):
        # format_ui_counters (game3d_ui_counters.h) field by field: the fixture of today's counter line, and the
        # report's reasons, hold kinds and decided sources.
        text = source('game3d_ui_counters.h')
        body = text[text.index('inline std::string format_ui_counters'):]
        body = body[:body.index('return text;')]
        no_mask = tuple(re.findall(r'"(\w+)"', re.search(r'names\{([^}]*)\}', text).group(1)))
        decided = tuple(re.search(r'decided_sources\{([^}]*)\}', text).group(1).replace(' ', '').split(','))
        groups = []
        for found in re.finditer(r'group\("(\w+)", \[&\] \{(.*?)\}\);|field\("(\w+)", ', body, re.S):
            key, inner, plain = found.groups()
            if plain:
                groups.append((plain, None))
            else:
                groups.append((key, {'decided': decided, 'none': no_mask}.get(
                    key, tuple(re.findall(r'field\("(\w+)", ', inner)))))
        self.assertEqual(tuple(groups), UNCOVERED_COUNTER_GROUPS)
        self.assertEqual(dict(groups)['inactive'], report.INACTIVE_REASONS)
        self.assertEqual(dict(groups)['held'], report.HOLD_KINDS)
        self.assertEqual(no_mask, report.NO_MASK_REASONS)
        self.assertLessEqual({int(v) for v in decided}, set(report.S2B_SOURCE_NAMES))


if __name__ == '__main__':
    unittest.main()
