# SPDX-License-Identifier: GPL-3.0-only
"""Unit tests for game3d_log_report: python -m unittest tools/reshade/test_game3d_log_report.py"""

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


def accepted_now(t, value):
    return line(t, f'[Sunshine 3D] Sunshine UI protection: accepted UI sources are now {value}; remembered for later '
                   'sessions of this game')


# The groups and keys of format_ui_counters (game3d_ui_counters.h), in its order.
COUNTER_GROUPS = (
    ('auto_frames', None), ('detection_frames', None),
    ('held', ('generated', 'inexact_after_exact', 'trusted_missing', 'cap')),
    ('inactive', ('no_candidates', 'size', 'unprepared')),
    ('decided', ('0', '1', '2', '3', '4', '5', '6', '8', '9', '10')),
    ('none', ('layer_aside', 'trusted_invalid', 'presented_blocked', 'ambiguous', 'difference_failed',
              'gate_no_hold', 'no_candidate', 'other', 'unaccepted')),
    ('full', ('6', '8', '9', 'depth_not_current')),
    ('full_d', ('hidden', 'ambiguous', 'visible', 'invalid')),
    ('untrusted_inferred', None), ('inexact_difference', None), ('trusted_full', None),
    ('presented_over_dedicated', None), ('full_alpha', None),
    ('full_alpha_d', ('hidden', 'ambiguous', 'visible', 'invalid')),
    ('trust', ('earned', 'revoked_full', 'revoked_presented', 'lapsed', 'restored', 'discarded')),
    ('samples', None), ('through_ms', None),
)
# The same before S1: sources 0-9, no unaccepted reason, and the tagged UI colour's opaque proof.
S0_COUNTER_GROUPS = tuple(
    (key, tuple(k for k in inner if k not in ('10', 'unaccepted', 'discarded'))
     + (('opaque_set', 'opaque_cleared') if key == 'trust' else ()) if inner else None)
    for key, inner in COUNTER_GROUPS)


def counters(t, values, runtime='0000000000000001', groups=COUNTER_GROUPS):
    """A 'Sunshine UI counters' line; values maps 'key' or 'group.key' to a count, every other field is 0."""
    fields = []
    for key, inner in groups:
        if inner is None:
            fields.append(f'{key}={values.get(key, 0)}')
        else:
            fields.append(f'{key}={{' + ' '.join(f'{k}={values.get(f"{key}.{k}", 0)}' for k in inner) + '}')
    return line(t, f'[Sunshine 3D] Sunshine UI counters: runtime={runtime} ' + ' '.join(fields))


# 100 Auto frames: 80 detected (60 by the UI layer, 20 without a mask, all for an ambiguous channel), 15 held and 5
# without a candidate, over 30 committed samples.
CLEAN_COUNTERS = {'auto_frames': 100, 'detection_frames': 80, 'held.generated': 10, 'held.inexact_after_exact': 3,
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


class ReadinessReport(unittest.TestCase):
    def test_clean_session_passes(self):
        checks = run(BASE + [ui('10:00:12', 2, 5, (0, 5, 0, 0), 0x2, 0x2), line('10:00:20', 'Finished exiting.')])
        self.assertTrue(all(c.status in ('PASS', 'INFO') for c in checks.values()), checks)

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
        rejected = ui('10:00:12', 0, 0, (0, 18, 0, 1000), 0xa, 0x0, layer=1, invalid=(0, 5, 0, 0))
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
        # The first-run shadow: the presented frame read hidden for 600 ms while no UI source decided.
        uncovered = ui('10:00:14', 0, 0, (0, 0, 0, 1000), 0x8, 0x0, scene=(0.01, 'hidden', 0, 1, 600))
        checks = run(BASE + [uncovered])
        self.assertEqual(checks['Hidden scene'].status, 'WARN')
        self.assertEqual(checks['Hidden scene'].times, ['10:00:14 600 ms (first-run shadow)'])
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

    def test_blocked_captures_and_copyless_ngx_are_named(self):
        checks = run(BASE + [
            line('10:00:06', '[Sunshine 3D] Sunshine Streamline depth: incomplete_state; source_selected=1'),
            line('10:00:07', '[Sunshine 3D] Sunshine NGX depth: confirmed_features=1 capture_eligible=1 '
                             'recovered_features=0 evaluations=40 nominations=40 copy_recorded=0 metadata_only=0 '
                             'unknown_feature=0')])
        self.assertIn('blocked', checks['Capture status'].detail)
        self.assertEqual(checks['NGX depth'].status, 'WARN')

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
        # The text test_game3d_alpha_auto pins for format_ui_counters (since S1).
        text = ('auto_frames=10 detection_frames=6 held={generated=2 inexact_after_exact=1 trusted_missing=0 cap=1} '
                'inactive={no_candidates=1 size=0 unprepared=0} decided={0=2 1=0 2=0 3=0 4=0 5=3 6=1 8=0 9=0 10=4} '
                'none={layer_aside=0 trusted_invalid=0 presented_blocked=0 ambiguous=0 difference_failed=1 '
                'gate_no_hold=1 no_candidate=0 other=0 unaccepted=2} full={6=1 8=0 9=0 depth_not_current=1} '
                'full_d={hidden=1 ambiguous=0 visible=0 invalid=0} untrusted_inferred=0 inexact_difference=3 '
                'trusted_full=0 presented_over_dedicated=0 full_alpha=2 '
                'full_alpha_d={hidden=0 ambiguous=0 visible=1 invalid=0} trust={earned=1 revoked_full=0 '
                'revoked_presented=0 lapsed=0 restored=0 discarded=2} samples=4 through_ms=12345')
        fields = report.counter_fields('runtime=0000000000000001 ' + text)
        self.assertEqual((fields['auto_frames'], fields['held.cap'], fields['decided.10'], fields['none.unaccepted'],
                          fields['full.depth_not_current'], fields['full_alpha'], fields['full_alpha_d.visible'],
                          fields['trust.discarded'], fields['through_ms']),
                         (10, 1, 4, 2, 1, 2, 1, 2, 12345))
        self.assertNotIn('runtime', fields)
        # The helper writes the same grammar.
        self.assertEqual(report.counter_fields(report.COUNTERS.search(counters('10:00:00', fields)).group(1)), fields)
        # A line before S1 (sources 0-9, the opaque proof) parses the same way.
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
        self.assertEqual(checks['UI holds'].detail, 'generated 10, inexact after exact 3, trusted missing 2 '
                                                    '(15% of Auto frames); 0 frames wanted a hold past the cap')
        self.assertEqual(checks['UI no mask'].detail,
                         '25% of Auto frames had no mask; decided none: ambiguous 20%; '
                         'without detection: no candidates 5.0%')
        self.assertIn('earned 1,', checks['UI trust events'].detail)
        # Without counters (older logs) none of these checks exists and the sampled ones decide.
        sampled = run(BASE + [sample])
        for name in ('UI counters', 'UI full frame', 'UI inferred alpha', 'UI inexact difference', 'UI holds',
                     'UI no mask', 'UI trust events'):
            self.assertNotIn(name, sampled)
        self.assertEqual(sampled['UI protection'].detail, 'detected 1')

    def test_counters_that_do_not_reconcile_fail(self):
        checks = run(BASE + [counters('10:00:12', {**CLEAN_COUNTERS, 'auto_frames': 101})])
        self.assertEqual(checks['UI counters'].status, 'FAIL')
        self.assertIn('does not reconcile', checks['UI counters'].detail)
        # A frame that wanted a hold past the cap detected instead: it is a detection frame, not a held one.
        capped = {**CLEAN_COUNTERS, 'held.cap': 4}
        self.assertEqual(run(BASE + [counters('10:00:12', capped)])['UI counters'].status, 'PASS')

    def test_counted_invariant_breaks_fail_without_a_sample(self):
        clean = ui('10:00:12', 2, 5, (0, 5, 0, 0), 0x2, 0x2)
        for key in ('presented_over_dedicated', 'trusted_full'):
            checks = run(BASE + [clean, counters('10:00:12', {**CLEAN_COUNTERS, key: 1})])
            self.assertEqual(checks['UI protection'].status, 'FAIL', key)
        # The counters are exact: a sampled flattening the counted frames do not confirm is only an example.
        flat = ui('10:00:12', 2, 1000, (0, 1000, 1000, 1000), 0x36, 0x2, hudless=(400, 600))
        self.assertEqual(run(BASE + [flat])['UI protection'].status, 'FAIL')
        checks = run(BASE + [flat, counters('10:00:12', {**CLEAN_COUNTERS, 'trusted_full': 2})])
        self.assertEqual((checks['UI protection'].status, len(checks['UI protection'].times)), ('FAIL', 1))
        self.assertIn('in 2 frames', checks['UI protection'].detail)
        self.assertEqual(run(BASE + [flat, counters('10:00:12', CLEAN_COUNTERS)])['UI protection'].status, 'PASS')

    def test_counted_full_claim_resolved_by_revocation_passes(self):
        # The designed safety net: a trusted channel covering the frame over an exact pair that shows the scene
        # keeps deciding until repeated contradictions revoke its trust, so its frames count as trusted_full first.
        layer = ui('10:00:12', 2, 1000, (0, 1000, 1000, 1000), 0x36, 0x2, hudless=(400, 600), layer=1)
        revoke = line('10:00:14', '[Sunshine 3D] Sunshine UI protection: alpha trust is now 0x2; '
                                  'remembered for later sessions of this game')
        resolved = {**CLEAN_COUNTERS, 'trusted_full': 120, 'trust.revoked_full': 1}
        checks = run(BASE + [layer, revoke, counters('10:00:15', resolved)])
        self.assertEqual(checks['UI protection'].status, 'PASS')
        self.assertIn('in 120 frames before 1 revocations of such a source', checks['UI protection'].detail)
        self.assertIn('a contradicted source lost its acceptance', checks['UI protection'].detail)
        # Without a sample, a revocation of a full claim also resolves the counted frames.
        self.assertEqual(run(BASE + [counters('10:00:15', resolved)])['UI protection'].status, 'PASS')
        # A sample no revocation resolved still fails, whatever else was revoked.
        flat = ui('10:00:12', 2, 1000, (0, 1000, 1000, 1000), 0x36, 0x2, hudless=(400, 600))
        self.assertEqual(run(BASE + [flat, revoke, counters('10:00:15', resolved)])['UI protection'].status, 'FAIL')

    def test_counted_full_frame_over_a_visible_scene_warns(self):
        full = {**CLEAN_COUNTERS, 'decided.10': 50, 'decided.8': 10, 'full_d.hidden': 3}
        self.assertEqual(run(BASE + [counters('10:00:12', full)])['UI full frame'].status, 'PASS')
        # The sample releasing a held route decided under the hold, so a visible one is not by itself a fault.
        checks = run(BASE + [counters('10:00:12', {**full, 'full_d.visible': 1})])
        self.assertEqual(checks['UI full frame'].status, 'WARN')
        self.assertIn('H1', checks['UI full frame'].detail)
        self.assertIn('each release of a held hidden-scene route (8, 9) shows one', checks['UI full frame'].detail)
        self.assertIn('S2b', checks['UI full frame'].detail)
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

    def test_counted_whole_frame_alpha_over_a_visible_scene_is_reported(self):
        # The Witcher 3 sign wheel: a trusted layer covers the frame without an exact pair, so trusted_full stays 0;
        # its samples after the first measure the visible scene beneath it. Pinning it flat is intended (P1, the
        # opacity ruling), so it is reported, not warned.
        wheel = {**CLEAN_COUNTERS, 'full_alpha': 40, 'full_alpha_d.visible': 3, 'full_alpha_d.invalid': 1}
        checks = run(BASE + [counters('10:00:12', wheel)])
        self.assertEqual((checks['UI full alpha'].status, checks['UI protection'].status), ('INFO', 'PASS'))
        self.assertIn('40 frames (50% of detection frames) decided a whole-frame alpha', checks['UI full alpha'].detail)
        self.assertIn('visible 3, invalid or unmeasured 1', checks['UI full alpha'].detail)
        self.assertIn('3 samples pinned an accepted whole-frame alpha flat over a visible scene, as intended (P1)',
                      checks['UI full alpha'].detail)
        # A trusted full claim over an exact pair that shows the scene, unresolved by any revocation, still fails.
        unresolved = {**wheel, 'trusted_full': 40}
        self.assertEqual(run(BASE + [counters('10:00:12', unresolved)])['UI protection'].status, 'FAIL')
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
        self.assertEqual(checks['UI inexact difference'].status, 'WARN')
        self.assertIn('until S3', checks['UI inexact difference'].detail)
        # Counters logged before S1 let the untrusted pass decide.
        before = run(BASE + [counters('10:00:12', known, groups=S0_COUNTER_GROUPS)])
        self.assertEqual(before['UI inferred alpha'].status, 'WARN')
        self.assertIn('expected before S1', before['UI inferred alpha'].detail)
        self.assertIn('opaque proof set 0', before['UI trust events'].detail)
        self.assertIn('legacy entries discarded 0', checks['UI trust events'].detail)
        clean = run(BASE + [counters('10:00:12', CLEAN_COUNTERS)])
        self.assertEqual((clean['UI inferred alpha'].status, clean['UI inexact difference'].status), ('PASS', 'PASS'))

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


if __name__ == '__main__':
    unittest.main()
