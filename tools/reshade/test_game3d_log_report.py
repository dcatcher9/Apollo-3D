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
       scene=None):
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
    return line(t, f'[Sunshine 3D] Sunshine UI protection: runtime=0000000000000001 mode=auto rendered=1 mask_path=1 '
                   f'input=automatic_gpu_mask retained=1 fg=0 fg_known=1 fg_enabled=0 input_state=input_seen '
                   f'detection=detected selected=automatic source=automatic source_availability=detected '
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
