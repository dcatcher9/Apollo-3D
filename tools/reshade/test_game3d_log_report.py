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
        self.assertIn('(UI layer)', checks['UI channel admission'].times[0])
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

    def test_unprotected_streamed_time_warns_with_its_reason(self):
        protected = ui('10:00:12', 2, 5, (0, 5, 0, 0), 0x2, 0x2)
        # Stellar Blade SDR with FG off: the layer held the scene image and current alpha was opaque.
        colour_only = ui('10:00:06', 0, 0, (0, 0, 0, 1000), 0xa, 0x2, layer=1, invalid=(0, 600, 0, 0))
        checks = run(BASE + [colour_only, protected])
        self.assertEqual(checks['UI protection gaps'].status, 'WARN')
        self.assertEqual(checks['UI protection gaps'].times,
                         ['10:00:06-10:00:12 (6 s) FG off: UI layer has colour but no alpha (60% of pixels); '
                          'current alpha covers 100% (untrusted)'])
        # FG on: the trusted layer kept the selective Backbuffer alpha out.
        fg_on = ui('10:00:06', 0, 0, (0, 0, 3, 1000), 0x6, 0x6, layer=1, invalid=(0, 996, 0, 0), fg=1)
        self.assertEqual(run(BASE + [fg_on, protected])['UI protection gaps'].times,
                         ['10:00:06-10:00:12 (6 s) FG on: UI layer has colour but no alpha (100% of pixels); '
                          'Backbuffer alpha kept out beside a trusted UI channel'])
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
                         ['10:00:09-10:00:12 (3 s) FG off: current alpha covers 100% (untrusted)'])
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
                         ['10:00:06-10:00:12 (6 s) FG off: current alpha covers 100% (untrusted)'])
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
