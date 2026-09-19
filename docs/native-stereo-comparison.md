# Native Game 3D comparison and decisions

The current direction starts from the existing SuperDepth3D pipeline and its Sunshine fixes.
The first milestone gives that complete pipeline the `SunshineGame3D.fx` filename, namespace
and technique, while retaining its rendering, depth preparation, profiles and controls.
Verified color-conversion bugs are an allowed, separately validated correction to this baseline;
the original's HDR output is not assumed correct merely because its stereo geometry is retained.
Sunshine's integrated depth selector, full-SBS export, HDR handling and stereo settings overlay
remain the surrounding integration. The runtime contract is in [ReShade SBS](reshade-sbs.md).

This original milestone establishes the comparison baseline. Manual retains its profiles and
controls; an opt-in Automatic candidate now supplies generic depth interpretation, gradual scale
refinement and adaptive screen-plane control through the same original-derived renderer. Neither candidate replaces that
renderer with the independent implementation. Automated evidence and remaining quality acceptance
are distinguished below.

The September 15 adaptive-controller simplification replaces fixed startup gain and the separate
source-recovery coordinator with one source-owned update. Actual source changes initialize once;
same-source pin/unpin preserves state. The scale and raw screen plane evolve independently, and
interrupted rendering requires fresh depth before resuming. The current equations and admission
rules are owned by the [raw integration contract](reshade-sbs.md#experimental-raw-depth-automation).
Earlier fixed-gain and cached-source-return evidence below remains historical. The
[adaptive-controller review](../cmake-build-relwithdebinfo/adaptive-raw-controller-20260915/review.md)
records the passing policy, actual 4K HDR and production tests, the rejected checkerboard fixture,
and installation into both local test games. Moving-game/headset comfort remains a live check.

## Current milestone acceptance

1. Compare the frozen original export with the renamed full pipeline using identical input
   color/depth, original controls, and actual ReShade execution. The identification-only change
   must preserve final stereo pixels; the comparison must not refit controls to conceal changes.
   Any color-conversion correction is measured separately against SDR/scRGB/PQ color oracles,
   with the expected pixel differences documented instead of requiring equality to a known bug.
2. Exercise the full original pipeline in D3D11/D3D12 and SDR/scRGB/PQ, including mono preservation,
   depth readiness, original control response, AA/sharpening and export metadata.
3. Verify installer migration, integrated selection, overlay and receiver ownership for the new
   effect identity. Then check an actual game and headset before claiming live acceptance.
4. Freeze that evidence before any trimming. Name subsequent treatments explicitly and compare
   their quality and full-effect GPU cost against the same baseline.

Before a changed preprocessing representation reaches a live installation, execute its declared
depth-domain and consumer contract through the actual ReShade renderer. Include simultaneous known
near/far planes, both eyes, strength zero/quarter/full, screen-plane changes, thin foreground and
matching/mixed color-depth edges, AA off/on, and the first frame after missing input or a diagnostic
mode. Verify the intended prepared-field dependencies separately from final pixel displacement.
Negative parallax is valid when its filters and warp support it; positivity alone is not a quality
gate. A representation change can preserve flat geometry while worsening edges or GPU cost, so all
three are required before promotion. Synthetic checks supplement native source/HDR/overlay tests;
they do not establish live-game acceptance.

The fixtures and installer path are described in the [add-on guide](../tools/reshade/README.md).
The renamed milestone passed automated runtime, color, selection, installer and transport tests
on September 13; its evidence is in `cmake-build-relwithdebinfo/game3d-milestone-20260913/review.md`.
Live game/headset acceptance remains pending.

## Configuration extraction candidate

The later `game3d-home-visibility-20260914/` cleanup makes the Home editor follow
the existing Automatic/Manual compile-time choice. It hides 32 inactive Manual
controls/help rows in Automatic, preserves their saved values, and leaves common
strength, AA and depth-view controls visible. The final 18 compiler configurations
produce 259 byte-identical GPU program pairs against the pre-cleanup shader, with
unchanged uniform defaults/layout. This is editor cleanup, not a new geometry policy.
Its add-on wording distinguishes source-buffer selection from stereo depth setup.
The associated installer no longer silently selects the old independent renderer
when current shader sources are missing; both 24-group filesystem suites pass.
These sources are staged for local testing; the game installations are unchanged.

The later DX9 cleanup removes 235 lines of unreachable alternative resources,
functions and pass definitions. SunshineGame3D already admits only D3D11/D3D12;
all retained supported branches and six included files are unchanged. Eighteen
fresh compiler pairs produce 259 identical GPU programs, complete binding metadata
and uniform defaults; 21 unsupported configurations reject on each side. This
reduces maintenance work, with no expected runtime speedup. The
[source and compiler review](../cmake-build-relwithdebinfo/game3d-remove-dx9-20260914/change-review.md)
records exact coverage and the unchanged original SuperDepth3D hash.

The subsequent permanently disabled `Filter_Image` component removal deletes
47 lines of unused color-filter functions and its pass. It retains the exact
unconditional zero definition because ReShade rejects conflicting predefined
macros. Eighteen fresh compiler pairs preserve all 259 programs, bindings and
defaults; all 46 rejection checks pass, including predefined `Filter_Image=0/1`.
The [component review](../cmake-build-relwithdebinfo/game3d-remove-filter-image-20260914/change-review.md)
records the source proof. This has no expected runtime speedup.

The Automatic-only spatial bypass skips reconstruction and smoothing
while mono or Normal Depth View is selected. Its corrected 4K PQ runtime pair
preserves eight complete SBS outputs and eight active/resumed spatial outputs
byte for byte; all eight inactive spatial outputs are zero. The first resumed
stereo frame is checked without extra settling. An initial R16F fixture readback
overread invalidated the earlier intermediate captures; the fixed fixture reran
both sides, retaining that failure. The [runtime review](../cmake-build-relwithdebinfo/game3d-source-recovery-20260914/runtime-review.md)
records source-replacement recovery evidence as well. A subsequent actual
4K PQ control/treatment/control timing sequence preserves all twelve complete
SBS outputs across two scenes and both AA settings. Mono saves approximately
0.045–0.052 ms (4.7–5.3%) of complete-effect GPU time. Narrow AA0 active stereo
costs an additional 0.003–0.005 ms; other active cases have wider run-to-run
variation, so neither an active speedup nor zero overhead is claimed. The
[GPU review](../cmake-build-relwithdebinfo/game3d-inactive-gpu-cost-20260914/review.md)
defines the timestamp interval and limitations. The combined latest FEAD add-on
and DX9-cleaned 02D2 shader also pass actual 4K PQ source-replacement recovery
and inactive/resumed stereo tests; all eight complete outputs and active spatial
surfaces match the earlier control. These files are still staged locally;
installation and live-game/headset acceptance remain outstanding.

The `game3d-remove-spread-strategy-20260914/` cleanup removes the rejected
whole-scene spread option from the production raw/camera policy, including its
duplicate accumulators and strategy state. The active central reference and
adaptive screen plane are unchanged. Lifecycle tests now exercise that actual
default; research-only spread assertions remain in frozen evidence and the
standalone statistics utility remains separate. Five focused tests pass, and
4,778 complete public-output trace lines match the old default exactly. A fresh
4K PQ normal-depth ReShade pair preserves all eight SBS and sixteen spatial
outputs byte for byte. The final add-on also recovers a lasting replacement
buffer with fresh H=63.0219536, matching the native-depth oracle. The
[cleanup review](../cmake-build-relwithdebinfo/game3d-remove-spread-strategy-20260914/runtime-review.md)
records exact source and executable identities. This is maintenance and test
coverage improvement; it does not change gain policy or establish live acceptance.

The later raw-wrapper cleanup removes a duplicate startup timestamp and stale
capture check. The stricter attempt-start bound already rejects those packets
with the same status before any subsequent state change. All 4,778 public trace
lines remain byte-identical, both frozen policy test suites pass, and the three
worktree lifecycle tests pass. The
[watermark review](../cmake-build-relwithdebinfo/game3d-remove-raw-watermark-20260914/review.md)
records the invariant and exact source boundaries. Calibration and zero-plane
behavior are unchanged; this removal has no measured performance claim.

The subsequent `game3d-conv-camera-fastpath-20260914/` experiment skips original
world-convergence calculations whose result Automatic replaces. Its first
candidate failed: the default deterministic comparison preserves all twenty full
stereo images and all spatial depth components, but changes two bottom-left
history components in every P preparation buffer. The stricter test now captures
both complete preparation base levels and rejects that difference. Reproducing
the actual ReShade SM5.1/O1 bytecode confirms that the earlier SM5.0/O3 compiler
check did not inspect the actual runtime programs; it does not establish a
compiler fault. The [runtime review](../cmake-build-relwithdebinfo/game3d-conv-camera-fastpath-20260914/runtime-review.md)
tracks the original history mismatch and separate repeatability/sampling
diagnostics. That failed candidate remains preserved separately.

Those diagnostics reproduce the original history mismatch and isolate a fix:
request mip zero when reading the stored weapon-history row. The resulting
compiled Mod_Z changes in exactly one sampling instruction; thirteen other
programs and all twenty final images remain identical. With that same lookup on
both sides, the corrected fastpath comparison now passes exact equality for all
twenty stereo images and forty complete P/L preparation buffers. This new default
Automatic scRGB result preserves the failed original comparison. The stronger
fixed-control pair now also passes: forty stereo images and eighty preparation
buffers are exact across both comparisons. The corrected normal-mode compiler
matrix passes all eighteen pairs and forty-six expected rejection executions;
all Manual programs, non-Mod_Z Automatic programs, bindings and defaults are
unchanged. Both default Direct3D 12 sides match their actual runtime bytecode.
The corrected candidate also passes all six native 4K PQ readiness/recovery runs.
The inactive pair preserves all eight complete SBS and sixteen spatial outputs;
startup and exact-source return pass the native assertions without injected
camera values. Startup/return do not save paired pixel outputs, so these checks
supplement the deterministic preparation gate rather than replacing it.

A native 4K PQ control/treatment/control timing sequence preserves all fifteen
source/output artifacts across three runs and verifies 2,304 GPU timestamp
samples. All twelve treatment medians improve against both controls; active
stereo saves 0.085–0.092 ms, about 5–7% of complete-effect GPU time. This excludes
readback, encoding, streaming and headset latency and is not a game-FPS claim.
The [promotion review](../cmake-build-relwithdebinfo/game3d-conv-camera-fastpath-20260914/promotion-review.md)
records the completed gates and their limits. Exact candidate `1D1F61B5` is now
the external working `SunshineGame3D.fx`, including the explicit history-mip fix;
its six includes and original `SuperDepth3D.fx` remain unchanged. Game files were
not updated and the add-on was not repackaged. Broad live-game/headset acceptance
and initial-scene scale robustness remain open; this change preserves calibration,
strength, zero-plane policy, filtering and the existing warp.

The `game3d-demand-calibration-20260914/` add-on cleanup makes the earlier
independent renderer's percentile calibration run only while that renderer
requests it. Game3D retains the same selector histogram and raw policy. The
instrumented control demonstrates two unsolicited cache entries; the treatment
has none, including after the actual Recalibrate action. Both production runs
preserve all 24 complete 4K PQ SBS/reconstruction/smoothing artifacts exactly,
and the reference renderer's content-selection calibration tests pass on both
sides. The test query is absent from the production DLL. This removes unnecessary
CPU work; no timing improvement was measured. The
[acceptance review](../cmake-build-relwithdebinfo/game3d-demand-calibration-20260914/acceptance-review.md)
retains the separate failed statistics-only control: its old small-viewport rule
skips the fixture's before-clear copy. That mode has no treatment acceptance.
Request-pause continuity retains existing eligibility invalidation rules and is
source-reviewed, not newly runtime-tested.

The user requested a live test after these checks. Exact production add-on
`B57A015F` and shader `1D1F61B5` with its six tested includes are now installed in
Dead Space and Expedition 33; `live-installation.json` records every installed
hash and backup. Both original comparison presets, Automatic presets, ReShade
settings and original SuperDepth3D files remain unchanged. Select
`B-Sunshine-Automatic.ini` for this build's Automatic test. Live halo, color,
strength and recovery acceptance remain pending.

The first live session with that installation is captured in
`game3d-live-20260914-1616/`. Host and client logs confirm 7680x2160 SBS,
scRGB export and BT.2020/PQ decoder output. Startup calibration was repeatedly
interrupted while source 110 was inactive on alternating frames; automatic
selection of source 121 recovered calibration at 16:14:04.611. The
[session review](../cmake-build-relwithdebinfo/game3d-live-20260914-1616/session-review.md)
separates these observations from unverified visual quality and frame-generation
assumptions. No readiness policy or running installation was changed.

The user's subsequent visual feedback reports opposite Stereo Depth brightness
ordering between Manual and Automatic, and weak strength-slider response in the
normal game picture, using the Sunshine add-on's strength control. The bounded
`game3d-stereo-diagnostic-20260914/v4/` actual-ReShade check includes the Dead Space
Manual application profile (`0x4C5965E5`). Both modes preserve the same near/far
ordering on simultaneous known-depth bands. Automatic directly exports negative
convergence as diagnostic grayscale, which the host's PQ conversion clips to
black; that display problem does not reproduce or explain a complete polarity
reversal. Automatic's small response does reproduce with supplied camera values:
strength25/50/100 gives .5/1/2.5 pixels of binocular shift, versus18.5/37.5/75.5
in Manual on one flat1080p scene. Different screen planes prevent treating that
ratio as a general depth-strength ratio. The
[rendered diagnostic review](../cmake-build-relwithdebinfo/game3d-live-20260914-1616/reported-stereo-response.md)
records complete artifacts, independent numeric checks and limitations. No
production or installed files changed. Live polarity and matched-strength quality
remain unresolved and take priority over promoting the next removal candidate.

The subsequent fixed-coefficient experiment tests `g=.5` against the installed
`g=.05`, retaining the reference and screen-plane policy. This is an explicit
presentation-gain choice, not recovery of metric camera scale or proof of an
incorrect depth sign. CPU lifecycle and finite-value checks pass. Six real
D3D12 scRGB captures compare the current shader at both AA settings: control
strength 100, increased gain at analytically matched strength 9.090909, and
increased gain at strength 100. Three flat-depth landmarks match exactly in
the matched pair, but one-pixel foreground signal falls by about 21–25%.
At strength 100 the increased gain produces approximately ten times the relief,
with more displaced fringe signal and thin-object loss. The gain candidate is
therefore **not promoted or installed**. The
[coefficient review](../cmake-build-relwithdebinfo/game3d-fixed-convergence-gain-20260914/gpu/review.md)
retains full floating-point intermediates, outputs and provenance; it does not
exercise native add-on UI events or establish live-game acceptance.

Source review identifies two separate contributors to investigate. Automatic's
signed foreground can lie below the original ray search's initial layer zero;
flat negative surfaces may shift by extrapolation even though the search never
visits their full displaced interval. Separately, reconstruction radius depends
on the strength slider itself, so matching gain times strength does not match
that filter's footprint. A search-domain experiment must keep gain and slider
identical between its control and treatment. Neither finding justifies flipping
the depth sign or accepting the coefficient experiment's quality loss.

The isolated negative-ray-domain test confirms missing foreground coverage.
Only admitted Automatic changes its starting layer to the existing mixed-depth
floor `-1.5`, with the source coordinate shifted to the same point on the ray.
Four D3D12 scRGB captures, at unchanged gain/slider within each pair, preserve
all prewarp bytes and flat-depth geometry within 0.0019 pixels. One-, two- and
four-pixel foreground strips recover approximately their source signal areas.
At the stronger experimental gain, the control also misses part of the expected
main foreground silhouette; a separately labelled post-capture coverage check
finds that deficit eliminated by the treatment. Lower outer leakage in a control
that loses foreground is not by itself better edge reconstruction.

Plain silhouettes with matching color/depth edges retain zero measured outer
leakage. The deliberately mixed color-fringe rows still expose more displaced
edge color at the stronger gain, so this is not a complete halo correction.
The [ray-domain review](../cmake-build-relwithdebinfo/game3d-negative-ray-domain-20260914/review.md)
retains that distinction and the original scoring setup failure: its revised
scorer records differing pre-stereo initialization wait counts separately while
requiring identical active controls, deterministic stereo ticks and prewarp
bytes. Two further Manual compiler pairs preserve all 29 programs, bindings and
defaults exactly. A subsequent 1080p control/treatment/control timing sequence
preserves all prewarp bytes, control-repeat outputs and three-plane geometry
with AA off/on. Its 384 hardware timestamp samples measure roughly 0.492 ms
additional full-effect cost without AA (0.441 to 0.933 ms), and 0.496 ms with AA
(0.530 to 1.026 ms). AA-on captures retain zero outer leakage in this weak-gain
scene, but thin-strip areas are not exactly unchanged. These are renderer-only
measurements; neither game frame rate nor headset latency was measured.

The expanded-search treatment and gain/radius experiments remain **unpromoted and
uninstalled**. The prepared 4K timing runs were not started. The user clarified
that simplified preprocessing is acceptable when its complete downstream contract
is supported, and requested a source-boundary fix. Original depth conversion and
convergence remain the comparison reference, not a requirement to copy every
equation into Automatic. The simple original conversion already has the same
rational form as the raw-reference candidate when optional shaping is disabled;
bypassing the rest of preparation and choosing an unmatched convergence gain were
separate changes. Original convergence can also be negative: the defect is an
incompatible consumer assumption, not the sign alone. Export, HDR fixes, overlay
and automatic depth-buffer selection remain required integration features.

The subsequent `game3d-positive-conditioning-20260914/` source-boundary experiment
keeps positive prepared D through conditioning, makes its active field independent
of presentation strength/zero plane, and converts to signed displacement only at
the ray samples. It uses the same complete negative-domain search as its reference;
it does not change the gain or reference policy. Four actual D3D12 scRGB runs cover
both gain inputs, both AA states and twelve first-frame parameter/admission cases.
The active field stays in the admitted positive domain and is exactly invariant
under the tested strength, zero-plane and reentry changes. Flat-plane geometry
remains within 0.008 pixels per eye and 0.016 pixels binocularly; mono copies and
first-frame recovery checks pass. Two Manual compiler pairs preserve all 29
programs, bindings and defaults.

These mechanical checks do **not** approve the candidate. At stronger gain, outer
edge-fringe signal in the composite scene increases with AA both off and on.
Complete-technique medians also increase from roughly 0.93–1.06 ms to 2.06–2.42 ms
over the already-corrected ray reference. All 512 raw timestamp pairs are retained;
one run per side/gain is not a timing-stability study. Positive-depth interpolation
followed by reciprocal conversion changes edge conditioning even when flat
geometry matches. Candidate `B9255FEB` is therefore **rejected for live promotion**,
with original and installed sources unchanged. The staged runtime report retains
the field, geometry, fringe and cost evidence separately. No 4K runtime, native
selector, actual overlay interaction, or headset acceptance is claimed for it.

The subsequent affine-conditioning experiment (`04171F0A`,
`game3d-affine-conditioning-20260914/`) retains an unreferenced inverse-depth
coordinate through conditioning and applies placement/strength at the ray
boundary. Eight actual D3D12 scRGB runs cover 1080p and 4K, two injected gains,
AA off/on and first-frame transitions. The declared field-domain, mono recovery,
diagnostic and independently measured strength checks pass. Mixed-edge leakage
is lower than the corrected Automatic reference in the stronger-depth cases,
while 4K AA-on one-pixel signal is slightly lower too; this remains an open
quality finding. Complete-technique cost increases roughly 5–9% over that
reference in these single runs. Four additional paired native 4K PQ scenarios
pass with the unchanged production add-on: diagnostic/inactive reentry, startup
recovery, same-source return and lasting source replacement. Their saved-field
and internal lifecycle assertions have different scopes, retained in the report.
These results do not establish quality parity with original Manual or every
game's source association. The
[bounded runtime review](../cmake-build-relwithdebinfo/game3d-affine-conditioning-20260914/runtime-review.md)
retains the evidence and remaining gates. The experiment is not promoted or
installed; original and production sources remain unchanged.

The fresh original-derived comparison then caught a matching assumption:
Compatibility_Power0 adds a depth-dependent compatibility translation, so
matching the main affine depth scale does not match final relief. That pair's
failed admission is retained. A separately frozen pair using fixed compatibility
passes the unchanged relative-relief and third-depth gates across16 static/moving
cases. It still finds greater mixed-edge fringe in the candidate, and unstable
one-pixel foreground-position coverage at particular movement phases. These
quality findings prevent promotion despite the successful geometry and native
recovery tests. The
[matched comparison review](../cmake-build-relwithdebinfo/game3d-affine-conditioning-20260914/matched-original-fixed-compat/runtime-review.md)
records the operating point, measured regressions and preserved evidence.

The user explicitly prefers full depth resolution. Automatic's target is now
100% of the game-frame grid; do not restore the75% reduction merely to improve a
timing score. A saved-field audit locates the moving1px failure in the first
75%-resolution point sample, before reconstruction or warp. The isolated
`game3d-full-resolution-20260914` experiment removes that reduction while
preserving camera/strength and the original Manual reference. Native game depth
may itself be lower resolution; a full processing grid cannot invent missing
game data. Further savings should preserve the supplied detail and be measured
in the remaining pipeline. This preference does not waive the halo/edge gates.

The full-resolution pair passes the unchanged input, finite-field and geometry
checks. All three flat-plane landmarks remain exactly equal, and every tested
one-pixel movement phase retains foreground-position signal. Complete-technique
medians increase by about 0.07 ms at 1080p in this one pair. However, mixed-edge
fringe increases before final AA, so full resolution remains the intended grid
while edge handling still needs correction. The
[full-resolution review](../cmake-build-relwithdebinfo/game3d-full-resolution-20260914/runtime-review.md)
records the captured detail benefit, fringe regression, immediate-outline checks
and the 58-program Manual compiler proof with its narrower SM5.0/O3 scope.
This candidate has not replaced the installed shader, and its 4K/native runtime
checks remain outstanding.

The isolated `game3d-fullres-gap-correction-20260914` pair then restores only the
original gap-dependent compatibility correction in Automatic. Both grids remain
full resolution; all 96 complete preparation-file pairs are byte-identical and
all flat-plane landmarks remain exactly equal. Static mixed fringe outside the
four-pixel boundary band drops to zero with AA off and on, while raw immediate
foreground retention is unchanged on both revealing and occluding broad edges.
However, the one-pixel foreground-position area falls from 1.030899 to 0.925075
with AA off and from 1.139994 to 0.950341 with AA on; two-pixel signal also falls.
This is a useful halo diagnosis, not acceptance of that correction. Its shifted
coordinate also feeds the later hole-mask taps. The full-resolution grid stays;
the next change must account for the lost thin-object contribution rather than
trading it for a smaller fringe metric. Production and installed sources remain
unchanged. The
[gap-correction review](../cmake-build-relwithdebinfo/game3d-fullres-gap-correction-20260914/runtime-review.md)
retains all 1,792 individual strip observations, including motion/eye asymmetry
after AA, and separates the benefit over the matched original reference from
the regression against full-resolution Automatic.

The next actual-renderer trace (`game3d-fullres-gap-uv-trace-20260914`) records
FP32 coordinates from the same ray calls used for color. All 84 ordinary file
pairs equal their uninstrumented counterparts; eight pre-gap coordinate/flag
array pairs and the complete deterministic clock prefixes also match. The first
checker stopped because its whole controls-file comparison included a natural
helper startup wait count. A separate, retained revision records that count and
compares every actual setting exactly, while preserving all image, runtime and
clock gates. The
[trace runtime review](../cmake-build-relwithdebinfo/game3d-fullres-gap-uv-trace-20260914/runtime-review.md)
documents this distinction; no instrumented timing is accepted as performance
evidence.

The trace shows that the gap correction moves valid partial thin-object color
support onto background texels. Pre-gap coordinates and preparation are unchanged,
and raw-eye confidence is identical over the complete captured frames. In the
bounded inspected rows, all five removed thin-color samples touch near-depth
source texels; all 38 removed mixed-fringe samples instead touch mixed-color
far-depth texels, with no near-depth support. The
[footprint analysis](../cmake-build-relwithdebinfo/game3d-fullres-gap-uv-trace-20260914/analysis/support-review.md)
records these observations and their limited coverage. This motivates testing
protection of foreground support within the color-sampling footprint. A direction
or hole-mask check alone cannot distinguish the two observed cases. This is not
yet a validated production rule or a change to the installed shader.

The isolated foreground-support guard (`3DD217F8`,
`game3d-fullres-foreground-support-20260914`) now passes the 1080p comparison.
It uses two existing point-depth taps within an interior normal-color footprint,
without new textures, passes, user controls, game profiles or depth clipping.
Every measured individual thin-strip area is unchanged or improved versus the
gap-only candidate; one-pixel AA-off area returns to 1.0306-1.0311 over all phases.
Measured plain/mixed outer fringe remains zero with AA off and on. All 96
preparation pairs, flat landmarks and measured immediate-outline statistics
remain unchanged. The
[guard review](../cmake-build-relwithdebinfo/game3d-fullres-foreground-support-20260914/runtime-review.md)
retains AA overshoot/phase variation and one-pair timing limits. This candidate
advances to 4K original-relative and native HDR/lifecycle checks. The accepted
quality floor remains the established original workflow, not the best individual
metric from every rejected intermediate; the latter comparisons help diagnose
tradeoffs. The candidate is not installed.

The actual 4K triple (`game3d-fullres-original-4k-20260914`) passes its frozen
mechanical gates but fails original-relative edge quality. Both full-resolution
Automatic candidates preserve moving thin objects and agree exactly on all
3,712 measured strip areas. All 96 Automatic preparation pairs are identical;
relative landmark error versus Manual is at most 0.04123 px per eye and
0.08039 px binocular. However, mixed outer-fringe peaks reach 0.500002 with AA
off and 0.166913 with AA on, versus zero for the reference. Thin-object leakage
beyond the existing four-pixel margin also appears. These results preserve the
failed quality evidence; mechanical eligibility is not quality acceptance.
Full resolution remains required while this failure is diagnosed. The complete
technique takes approximately 6.98/7.36 ms with AA off/on in this one synthetic
guard run, versus 3.50/3.84 ms for the original 75%-grid Manual reference. This
is descriptive cost, not a game-FPS claim or permission to reduce depth detail.

The [4K localization review](../cmake-build-relwithdebinfo/game3d-fullres-original-4k-20260914/halo-audit/review.md)
narrows that wide-edge excess to one measured pixel in movement phase f6; it is
not a continuous broad halo in this fixture. A separate right-eye extra lobe
affects the two-pixel strip windows across phases. The
[same-call UV trace](../cmake-build-relwithdebinfo/game3d-gap-uv-trace-4k-20260914/analysis/review.md)
preserves all 98 ordinary output files and the 686-tick/7-capture prefix exactly.
It establishes distinct causes: the f6 sample has no gap correction despite a
fully open existing hole mask, while the thin-window extra color is introduced
when correction moves a background-only footprint onto a different one-pixel
foreground object. Protecting foreground present before correction cannot
prevent foreground introduced afterward. Pixel admission and donor handling
therefore require separate tests; widening admission alone is not a complete fix.

The isolated pixel-admission candidate `C841FC4F` now reuses the existing
four-pixel full-hole threshold in Automatic's correction admission, while
preserving original admissions and correction magnitude. Its actual 4K
[runtime review](../cmake-build-relwithdebinfo/game3d-gap-pixel-admission-4k-20260914/runtime-review.md)
confirms zero measured outer fringe with both AA settings, all 96 preparation
pairs identical, exact paired landmarks and all 3,712 individual thin records
unchanged. The separate wrong-donor lobes remain. Raw immediate-edge retention
is unchanged; six AA-on moving cases lose 0.00090–0.00110 in the revealing
mixed-edge first-pixel mean. The review also retains the small AA retention
tradeoff versus the original reference. This is support for the isolated
admission change, not completed shader acceptance. A subsequent donor-veto
candidate preserves the pre-correction coordinate when the proposed footprint
would introduce a newly nearer source texel. The combined `D6373EC7`
[4K runtime review](../cmake-build-relwithdebinfo/game3d-gap-donor-veto-4k-20260914/runtime-review.md)
now confirms zero measured outer fringe and wrong-donor lobes, unchanged
preparation and geometry, and preserved one-, two- and four-pixel objects.
Every raw/export RGBA pixel in all 1,216 two-pixel main neighborhoods is exact
to the admission-only control: the removed area was the false outer lobe.
Tiny AA-on one-pixel changes remain documented. Complete-technique median GPU
time was 7.96/8.56 ms with AA off/on, versus 6.97/7.34 ms for that control;
the single ordered run has high spread and does not establish a stable cost.
The subsequent 1080p comparison also passes its geometry, whole preparation,
source, mono and layout gates. Its independent review retains small AA1 coverage
redistribution (worst four-pixel-strip area decrease 0.05165), with no disappearing
thin objects or measured outer fringe; mixed revealing-edge retention improves.
These checks are local warp corrections for Automatic's affine-U representation,
not changes to original SuperDepth3D or evidence that every live-game halo is resolved.

The first comparison against installed Manual bytecode caught a common-code
ordering change: hoisting the existing hole-width multiplication extended its
register lifetime and changed the compiled eye shader. Bindings, defaults and
the other 13 programs remain exact. The branch-isolated correction `A1BFC48E`
restores Manual's original computation order. All four Manual and six Automatic
compiler pairs pass: 144 whole programs and 20 bindings/default files are exact
to their respective controls. All 86 Automatic generated HLSL programs are also
exact after removing only source `#line` directives. The validator is SM5.0/O3;
it does not masquerade as an actual SM5.1/O1 runtime comparison. The source and
compiled-program identity support retaining the existing D637 Automatic runtime
evidence under its original identity. The first failed compiler output is preserved.

The current strong synthetic comparisons use Compatibility Power 1 and g=0.5.
Production retains depth-dependent Compatibility Power 0, native g=0.05 and
initial strength 50. The unchanged native lifecycle fixture explicitly forces
Compatibility Power 1 and active strength 100, while validating actual source-owned
g=0.05. All four full-resolution native 4K PQ pairs now pass: inactive/diagnostic
reentry, startup, same-source return and lasting depth-source replacement. The
separate production-control comparison preserves all 20 complete final RGBA
images, 40 whole P/L pairs and all three-plane geometry exactly against 185.
Its existing AA1 mixed-fringe residual remains (outer mean 0.000028018,
maximum 0.12498); default settings are not claimed halo-free. The
[default-settings review](../cmake-build-relwithdebinfo/game3d-donor-veto-defaults-20260914/runtime-review.md)
records this bounded regression judgment. No gain or production default was changed
to match the stronger synthetic test controls.

The user requested the next live test. After the above checks, `A1BFC48E` was
installed in the external working tree, Dead Space and Expedition 33 at
2026-09-14 21:15:57 local time. The
[installation record](../cmake-build-relwithdebinfo/game3d-donor-veto-manual-order-20260914/live-installation.json)
authenticates all three copies and exact previous-file backups. Six includes,
original SuperDepth3D files, the production B57 add-on, all presets and ReShade
settings remain unchanged. Select `B-Sunshine-Automatic.ini` to test the complete
signed-domain, affine-conditioning, full-resolution and edge-correction candidate.
Live quality and performance remain unverified. Weak default Automatic depth
relative to Manual remains unresolved; the negative search floor adds work and
a true full-field GPU minimum is a separate proposed optimization, not installed code.
The [source-only ray-bound plan](../cmake-build-relwithdebinfo/game3d-tight-ray-start-plan-20260914/README.md)
reduces the actual final `texSmooth` field without dropping any texel, then applies
the existing placement and strength arithmetic. It preserves a conservative
search margin but requires new sampling-phase and total-cost validation before use.

The subsequent 21:17–21:24 Dead Space live run reported a correct stereo map and
cleaner outlines, but intermittent lag and a small visible strength-slider range.
The preserved [live snapshot](../cmake-build-relwithdebinfo/game3d-live-20260914-2116/snapshot.json)
contains 24 source changes after five startup changes: 13 histogram promotions
and 11 resolution promotions, with seven transitions less than one second apart.
They caused 18 basis-change reports and 11 fresh-reference acquisitions. Most
startup frame-mismatch reports instead concern the same selected source becoming
ready/unready; they must not all be counted as source switches. Host processing
averaged approximately 7 ms during gameplay. The 709 ms maximum at 21:23:58
followed game exit and is not evidence of a steady-gameplay stall.

The add-on strength control reaches Depth_Adjustment directly. In Automatic,
the retained gain g=.05 and curve .01+.0099*S yield only about 5.05 to 10 pixels
of binocular separation for a reference-to-infinity surface at 4K when moving
S from 50 to 100. The reference surface remains at zero separation. Increasing
the visible range is a presentation change, not a broken-uniform repair; a
late strength multiplier can preserve the current conditioned depth and calibration.
The user additionally requested stable preference for a valid high-resolution
scene buffer. Selector changes must distinguish materially fuller scene support
from ordinary histogram movement, retain high resolution among comparable
sources, and still recover from invalid or destroyed resources.

The [paired-selection fix](../cmake-build-relwithdebinfo/game3d-depth-selection-paired-wins-20260914/change-review.md)
implements that preference without changing sampling, histogram analysis, calibration or
rendering. It uses the existing material completeness margin for globally ordered support
classes, prefers native coverage within comparable support, and requires three fresh paired
wins for healthy replacements. Six related CTests pass. Identical targeted regressions fail
against the frozen old header and pass against the new one. Production add-on `5D606AD5`
was installed in both games, with the B57 DLLs backed up and all shaders/presets/settings
verified unchanged; the [installation record](../cmake-build-relwithdebinfo/game3d-depth-selection-paired-wins-20260914/installation.json)
records exact files. Native live-game switching stability remains to be tested.

The isolated full-field minimum candidate `ADFD70AF` passed actual 1080p/4K preparation,
min-reduction and geometry gates, plus nine supported compiler comparisons (125 unchanged
existing CSOs, four intentionally changed eye programs, nine unchanged default buffers).
The initially requested odd-size compiler case correctly failed the existing even-dimension
admission gate; that failure is preserved. Supported partial-tile and oversize fallback cases
passed separately, and both odd-size inputs were rejected consistently on both sides.
Single ordered renderer pairs measured technique medians of 1.163/1.268 to .719/.823 ms
at 1080p and 7.826/8.408 to 3.269/3.699 ms at 4K, for AA off/on respectively.
However, the [quality review](../cmake-build-relwithdebinfo/game3d-tight-ray-start-20260914/quality-review.md)
found increased average missing one-pixel detail and temporal area variation with 4K AA on.
The optimization is therefore **not promoted or installed**. The combined late strength
multiplier candidate remains unrun and held on this baseline-quality failure; the installed
A1 shader and its strength range remain unchanged.

The subsequent [sample-reuse optimization](../cmake-build-relwithdebinfo/game3d-ray-sample-reuse-20260914/change-review.md)
keeps A1's complete ray-step history and reuses the unblended depth already sampled at the
previous step's end. F02AF0C7 introduces no new resources, passes, controls, precision reduction
or calibration. Nine compiler pairs preserve all bindings/defaults and 124 of 129 programs;
only the five Automatic eye programs change. Disassembly confirms that each ray-AA iteration
uses two depth reads instead of three. Fresh actual 1080p/4K scRGB comparisons preserve all
192 complete preparation pairs and 160 complete raw-eye/sharp-eye/SBS output pairs byte for
byte, including static and seven moving fine-detail phases with final AA off/on. Median
full-effect GPU time falls from 1.164/1.276 to .975/1.081 ms at 1080p and 7.936/8.415 to
6.158/6.605 ms at 4K, respectively. This is one ordered pair per resolution; timing tails remain
variable and do not prove end-to-end FPS or elimination of every hitch. The strong controlled
scene and exact local sample equivalence support this bounded promotion, without changing
production strength or claiming that existing halos are removed. The separate promotion record
determines installed status; live-game performance and the broader objective remain open.
F02 was subsequently installed in the external Depth3D tree, Dead Space and Expedition 33 at
2026-09-14 22:19:01 local time. The [installation record](../cmake-build-relwithdebinfo/game3d-ray-sample-reuse-20260914/promotion/installation.json)
records all three exact copies and A1 backups; 34 includes/original shaders/add-ons/presets/settings
and runtime files were verified unchanged. The stable selector remains 5D606AD5. Restart the game
to test this installed performance change; the Automatic strength-range follow-up is not installed.

The subsequent [linear-strength review](../cmake-build-relwithdebinfo/game3d-linear-strength-20260914/root-review.md)
tests a one-line Automatic factor `clamp(S,0,100)*.05*blend`, preserving preparation, camera state,
ray search, full resolution, precision and Manual behavior. Nine compiler pairs pass. F02/S100
versus F491/S20 preserves all 160 complete rendered outputs at 1080p/4K, including moving inputs.
Ordinary S0/1/25/50/75/100 sweeps with AA off/on pass independent geometry, exact mono at zero
and complete preparation invariance. The candidate's S25/50/75/100 gives approximately
10/20/30/40px synthetic near/far relief at 1080p and 20/40/60/80px at 4K.
**The candidate is held, not installed:** the wider S100 range exposes a halo outside the
deliberately mixed-color/background-depth fringe, while clean edges and outside-4px thin lobes
remain clear. Mechanical success and same-gain parity do not approve the larger range's quality.
Maximum-strength motion and the prepared clamp GPU fixture are deferred on this existing failure.
Current installed F02 strength and FP32 geometry remain unchanged; the previous sample-reuse
performance improvement remains installed. A preserved Python-cache setup failure was resolved
by authenticated byte restoration and disabled cache writes, without changing numeric runtime,
frozen manifests, scores or acceptance thresholds.

The subsequent [edge trace](../cmake-build-relwithdebinfo/game3d-edge-trace-20260914/conditioning-localization.md)
localizes one expanded-range halo through actual FP32 diagnostic outputs. All fifteen complete
ordinary color/depth artifact pairs remain byte-identical with instrumentation. At the inspected
1080p mixed-color edge, Reconstruction extends foreground depth by one pixel, and the default
horizontal-min smoothing operation extends that full foreground plateau by another pixel. The
ray therefore consumes near depth where the raw-depth donor guards still see background. The
existing 0.703125px correction helps, leaving 21.47% measured foreground chroma at the failing
pixel; disabling it would increase contamination there. This identifies a filter mismatch at
that edge, not a universal halo diagnosis. The next isolated candidate omits only Automatic's
extra horizontal-min assignment, with Reconstruction and other smoothing retained.
That [isolated removal was rejected](../cmake-build-relwithdebinfo/game3d-remove-horizontal-min-20260915/outcome.md):
the matched1080p actual-renderer pair preserves geometry and upstream preparation but worsens
both halo and thin-foreground coverage. At S100 AA off, mixed-edge outer contamination peaks
at0.500002 versus0.214734, while average one-pixel foreground area falls from0.906139 to0.369218.
S20 also loses foreground area. The expansion therefore provides useful ray support despite the
observed depth/color disagreement. Existing smoothing is retained;4K and motion runs stop on
this known regression. No new performance or quality acceptance is claimed, and installed F02
remains unchanged.

A subsequent [actual ray-bracket trace](../cmake-build-relwithdebinfo/game3d-ray-bracket-trace-20260915/outcome.md)
preserves all fifteen ordinary artifacts and the four prior diagnostic surfaces exactly. At the
same halo peak, the last ray step spans one layer and its unblended and blended exit depths
agree. A skipped-step predecessor or AA-created crossing therefore does not explain that peak.
The source footprint includes already mixed foreground color carrying far raw depth. A
supplemental depth-mismatch cue also cannot identify that contamination uniquely: clean
background samples share the same raw/filtered depth tuple. The frame's multi-step exits all
enter weighted POM, mostly in gradient rows; actual-endpoint interpolation is a separate
candidate, with no demonstrated halo fix or cost saving. This remains a single static 1080p
left-eye diagnostic, with no production change or expanded-strength quality acceptance.

The later [actual-endpoint candidate](../cmake-build-relwithdebinfo/game3d-actual-endpoint-pom-20260915/outcome.md)
reuses the recorded pre-step coordinate, layer and unblended depth. Source review caught and
corrected a units error before execution: residuals and the half-step interpolation bias must
remain in nominal-step units when an adaptive step spans multiple layers. The corrected
candidate preserves all112 complete preparation pairs across actual1080p/4K runs, independent
flat geometry, individual thin-strip measurements and the full gradient RGB region. It changes
some texture pixels, with final AA sometimes amplifying small raw differences. These static
results do not accept texture quality or a speedup; the candidate remains uninstalled while
motion and production AA-on/off benefit are evaluated. Internal ray/depth-edge filtering is
distinct from the final-AA option and retains its earlier quality justification.

A separately prepared candidate, `game3d-retire-alpha-state-20260914/`, removes the
unused alpha components of Automatic's CN and BlurEx targets while retaining
luminance, depth, all passes, and Manual behavior. MiniReconstruction remains
live and is not removed. This work remains paused while the preprocessing and
warp contract is validated. The candidate is not compiled, rendered, promoted or
installed; complete retained-field and final-output comparisons are required
before acceptance.

The configuration extraction candidate keeps the original rendering stages and replaces only the
Overwatch include when `SUNSHINE_GAME3D_GENERIC_CONFIG=1` is explicitly selected. The default
remains the installed milestone. `SunshineGenericConfig.fxh` retains the header-present generic
defaults and excludes application detection and numbered weapon profiles. Missing-include fallback values are not
equivalent and are not used as the generic baseline. Off/Custom weapon controls remain; an
unsupported saved profile value renders as Off without rewriting the preset.

The extraction is checked against a frozen source, then executed with the official ReShade
runtime. Current evidence is in `cmake-build-relwithdebinfo/game3d-simplification-20260913/`:
SDR/scRGB paired scenes with AA off/on match final RGBA exactly; 72 reflected numeric defaults
match the unmatched-application control. Running the same fixture binary under the Dead Space
and Expedition executable names also preserves the generic pixels and defaults. Full D3D11 SDR
and D3D12 4K scRGB/PQ checks cover depth preparation, controls, diagnostics, postprocessing and
missing-depth fallback. Stale numbered/negative weapon values match Off; Custom remains active.

This proves the extraction boundary on those inputs. It does not prove that generic defaults
interpret every game's projection correctly. The subsequent Automatic candidate adds raw-depth
initialization and convergence policy; camera/depth association and broad game quality remain
limitations. The selector's content histogram remains independent of rendering calibration;
no percentile gain was added.

## Implemented Automatic candidate

The compact Game 3D panel now offers Manual/Automatic, strength, AA and an advanced depth view.
Manual retains Smart Convergence; Automatic uses a fixed raw-depth reference and adaptive screen
plane, with explicit Recalibrate and current-source mono fallback. It requires neither a game-name
profile nor percentile normalization. The runtime contract and supported-input limits are in
[ReShade SBS](reshade-sbs.md#experimental-raw-depth-automation).

The September 14 candidate passed native, official-runtime 4K PQ, scRGB Performance Mode,
recalibration, controls, overlay and transport checks. Its report and tested/installed manifests
are in `cmake-build-relwithdebinfo/game3d-automatic-controls-20260914/`. Matched artifacts were
installed in Dead Space and Expedition 33 with existing presets preserved and Manual selected.
These checks establish the tested behaviors, not broad visual parity: paired moving-edge quality
and full-effect GPU-cost comparison against the original baseline remain outstanding, as does
live game/headset acceptance of halos, moving silhouettes and colors.

The September 14 Dead Space live test reported lag, weak pop-out, halos and repeated manual
recalibration. Its logs show automatic selector excursions away from the established 4K source,
followed by permanent raw-basis suspension even after that exact source returned. A subsequent
fix retains the original reference and permits exact-source recovery only after fresh captures;
it does not transfer calibration to a different buffer. The unused Streamline diagnostic probe
is now disabled by default. The old DLL reproduces the source-return failure in the actual
runtime; the treatment passes scRGB and 4K PQ recovery tests. Those tests do not establish a
lag or halo improvement. Evidence, preserved live logs and explicit original/Automatic presets
are in `cmake-build-relwithdebinfo/game3d-live-review-20260914/`. Original Manual is selected for
the next visual comparison; the selector's histogram promotions and Automatic reference-dependent
strength remain unchanged. Live quality parity is not yet established.

## First component removal: delayed color frames

The September 14 follow-up reported that lag was gone and that both original Manual and
Automatic still showed halos, with stronger depth and a larger halo in Manual. That establishes
the live comparison to preserve while removing components; it does not establish matched
Manual/Automatic disparity or eliminate the remaining edge-quality work.

The first removal deletes only SunshineGame3D's delayed-color-frame textures, samplers,
functions and copy passes. These were already excluded from supported full-SBS export. The
live-color branch is byte-identical; original SuperDepth3D, depth temporal filtering, profiles,
Manual/Automatic depth scale, ray search, HDR, AA, export and overlay remain unchanged. Legacy
delayed-frame opt-ins now fail explicitly even with export disabled. No FPS gain is claimed.

Evidence is in `cmake-build-relwithdebinfo/game3d-simplify-20260914/`. The frozen control and final
source run through the same official ReShade runtime and fixture binaries. Manual scRGB step,
detail and fringe scenes with AA off/on preserve every final RGBA pixel. The production
Automatic selector/sampler/policy/shader pair at 4K PQ preserves source/depth, H/t0, mono and
both full SBS images exactly. Two original corner-history G cells differ in each prepared-depth
snapshot; all convergence and non-history scene-depth bits match, and no final pixels are
excluded. Final-source D3D11 SDR correctness and delayed-frame rejection checks also pass.
The report identifies the final shader hash and retained captures. These are same-mode
regression results, not a claim that Automatic already matches Manual's strength or halo quality.

## Dedicated export presentation removal

The `game3d-presentation-removal-20260914/` stage removes 1,667 net lines of unsupported
display presentation from SunshineGame3D. Original SuperDepth3D and the six helper files
remain unchanged. Eighteen paired compiler/reflection variants preserve the supported
render graph and retained defaults; seventeen unsupported configurations reject explicitly.
The first default Manual comparison passed, while the first stronger Manual comparison
failed at detail-AA0. A same-old-shader repeat reproduced that failure, exposing uncontrolled
shader clocks and a temporary settling plateau. Those original failures remain recorded.

The subsequent `game3d-deterministic-presentation-20260914/` stage controls only the explicit
renderer test's frametime, timer and framecount. It starts stereo history after known inputs
are ready, preserves the original information helper and Manual adaptation, and uses a fixed
600-tick capture schedule with unchanged final stability checks. Strong old-old, strong
old-new and default old-new comparisons all pass: each preserves 20 complete RGBA captures,
36 source/depth artifacts, 5,486 verified clock rows and 23 capture records. No control fitting,
pixel exclusion or threshold relaxation was used. An earlier 300-tick stability failure and
180-second watchdog exit remain in the stage; redundant test-only readbacks were removed
without changing rendered ticks or final checks.

Finally, `game3d-minimal-export-20260914/` runs the pruned shader with the restored-center
production add-on together in the real 4K PQ acquisition path. The independent raw-depth
oracle and source recovery checks pass. Complete source/depth, mono and final AA0/AA1 buffers
match the installed baseline exactly. Two auxiliary corner-G cells differ in each prepared
texture under natural timing, so complete intermediate equality remains explicitly false.
The paired result preserves baseline output; it does not prove better halos, faster gameplay,
or physical headset/overlay behavior. No game installation was changed by these stages.

## Final image AA removal

On 2026-09-15 the user chose to remove final AA after reviewing its measured tradeoff.
The F02 Automatic motion comparison found approximately 10–20% less registered temporal
variation with AA, but 25–31% less fine-texture contrast and worse thin-object coverage.
Background RGB error also improved by 3–12%; this was a smoothing tradeoff, not an inert
control. Static 1080p/4K and older original-derived Manual observations supported the
detail-loss concern. These percentages describe the synthetic fixtures, not every game.

Current SunshineGame3D removes the `USE_AA` declaration, AXAA include, and Game3D-only
HDR-AA repair. The compact add-on panel also removes its binding and option. Shared packing
retains exact AA-off texel fetches. Internal ray de-artifacting, depth filtering, sharpening,
source transfer and mono fallback remain. Original SuperDepth3D keeps its own AXAA branch;
42 original shader programs compiled identically across SDR/scRGB/PQ. Current Game3D no longer
requires the AXAA file at installation; explicit older sources remain supported.

Evidence is under `game3d-remove-final-aa-20260915`. Source `301FFD73…` and post helper
`EF6E9732…` are compared against F02 with final AA explicitly off. The fixed native fixture
uses three static and seven moving cases without a geometry fit. This checks preservation
of AA-off output, not quality equality to AA-on. The completed bounded runs cover Manual at
1080p and Automatic at 4K: all 159 captured pairs agree exactly, including 20 final static/moving
images and their source/depth/preparation evidence. The additional mode/resolution combinations
in the reusable runner were not executed for this deletion. Native compact controls, preset
roundtrip, overlay and reload checks pass, as do scRGB/PQ interpolation/color endpoints,
corrected SDR interpolation, and D3D11 SDR depth/strength/sharpening/mono/readiness checks.

An SDR test correction accompanies this change: the existing shader intentionally filters
SDR in encoded sRGB space. Its old universal linear-light segment oracle failed identically
before and after AA removal. The oracle now measures SDR in the declared encoded domain;
HDR remains linear, thresholds remain unchanged, and captured outputs retain their previous
format. Fresh corrected SDR control and treatment runs pass, with all three captured outputs
byte-identical. This correction changes no production color processing.

The tested two shader files and unified add-on were installed into Dead Space and Expedition 33
with verified backups. Main shader hash is `301FFD73…`, post helper `EF6E9732…`, and add-on
`B3E75628…`. Presets and other support includes remain intact. Physical headset acceptance
still requires a fresh game launch; no performance gain or halo fix is claimed.

## Postprocessing merged into SBS packing

The next isolated removal folds the existing sharpening/infill calculation into the packing
pass. Game3D no longer allocates `SunshineSharpLeft`/`SunshineSharpRight` or executes a separate
`SunshineSharpen` pass. Raw per-eye textures, the sharpening calculation, HDR handling, native
mono endpoint and full-resolution SBS export remain. SDR explicitly retains the former FP16
intermediate rounding before RGB10A2 packing. The original SuperDepth3D path keeps its existing
pass through the shared include's default branch.

[Fusion evidence](../cmake-build-relwithdebinfo/game3d-fuse-post-20260915/status.md) records
72 exact native capture pairs: Manual/Automatic in SDR/scRGB/PQ at 640x360, plus Automatic
scRGB at 3840x2160. These cover active sharpening/infill settings and current-source,
zero-strength, missing-depth and recovery endpoints. All 48 corrected-source direct 4K
post cases also match exactly. Eighteen paired compilation/reflection configurations and
17 explicit rejection cases pass; the original reference retains identical bytecode,
defaults and bindings in all three color permutations. The two updated native fixtures build.

The tested main `1A1089A5…` and post include `A3682EF6…` are installed in the external Depth3D
checkout and both test games. Nine source/test files were copied with verified backups;
50 existing presets, add-ons, runtime libraries and other shader files were verified unchanged.
The installation record is under the evidence stage's
`promotion-20260915-151531-60a5578d3a7144418fc9f0e3ce4c105a` directory.
Removing two RGBA16F per-eye intermediates saves 126.56 MiB of nominal texture storage at 4K.
These correctness checks do not establish a measured FPS increase or physical headset acceptance.
The separate expanded-strength and gap-correction candidates remain uninstalled.

## AA-off comparison at equal relative strength

The actual ReShade D3D12 pair in
[game3d-strength-aaoff-20260915](../cmake-build-relwithdebinfo/game3d-strength-aaoff-20260915/outcome.md)
compares original-derived integrated Manual 66F at strength 83.2117919921875 with the current
AA-removed 301FFD build plus only the proposed late strength factor, candidate 6D5BAD97 at 100.
Both use AA off and the same 1920x1080 scRGB source/depth across one static and seven moving captures.
The controls were derived before rendering. All three relative plane pairs and independent
Automatic absolute predictions pass; maximum relative binocular mismatch is 0.075816 px against
the unchanged 0.25 px gate. Near/far relief is 39.93719 px Manual versus 39.88884 px Automatic.

Automatic retains substantially more of the 1/2/4 px foreground strips, but its deliberately mixed
color/background-depth edge has outer 4–32 px contamination peaking at 0.22266 versus zero in this
Manual reference. Clean-edge and thin-object outer regions remain clear in both. These are
unchanged synthetic metrics, not arbitrary-game visual acceptance. Manual retains its original
75% depth grid, Automatic uses full resolution, and their common zero planes/compatibility
processing differ. This compares complete workflows at equal relative depth; it does not isolate
one warp, match physical pop-out placement, or test automatic initialization.

The pair is eligible, but the stronger range remains uninstalled because of the edge-quality
tradeoff. The installed AA removal is unchanged. The outcome records complete provenance,
337 authenticated artifacts, exact source/depth/mono/clocks, review scope and descriptive timing.

### Isolated compatibility correction follow-up

The fresh [CP0-to-CP1 pair](../cmake-build-relwithdebinfo/game3d-compatibility-control-20260915/outcome.md)
holds the same Automatic shader, camera, strength, screen plane, source and preparation fixed.
Only the existing compatibility setting changes. The CP0 run reproduces all 112 ordinary
captured artifacts from the matched-strength treatment. All three absolute and relative plane
placements remain exactly unchanged between CP0 and CP1, as do all 896 individual thin-strip
area measurements across the eight captures.

With CP1, measured mixed-edge contamination 4–32 px outside the silhouette becomes zero in
every capture, down from a CP0 peak of 0.22266. Clean-edge measurements remain unchanged.
The inner 2–4 px mixed fringe improves on average but retains a higher worst peak than the
Manual reference, so this does not establish uniformly superior edge quality. This is a
1080p scRGB setting discriminator with fixed camera controls, not a production default change,
a calibration test or physical headset acceptance. The expanded strength range and this
correction remain uninstalled pending broader rendering checks.

The subsequent fixed-distance candidate `599F9D3F` passes exact 1080p reproduction of both
112-artifact CP1 collections, the existing 4K/S25 geometry and foreground-quality measurements,
and 4K geometry with planes beyond both displacement clamps. Clamp quality was excluded before
collection because its original strip measurement windows extend offscreen. However, complete
frame inspection of the ordinary 4K pair finds substantial background texture changes that the
foreground-chroma metric does not measure. Many changed pixels are analytically visible
background, rather than newly exposed gaps; individual pixels both improve and regress.
The existing gap classification is too broad for treating a fixed donor offset as a proven
quality fix. Keep the candidate held and add explicit observed-background texture evidence
before changing the live correction. See
[fixed-gap investigation](../cmake-build-relwithdebinfo/game3d-fixed-gap-20260915/outcome.md).

The subsequent native trace reproduces all 28 original complete static captures and identifies
one premature background-ray termination: internal de-artifacting blends a farther sample with
a nearer conditioned neighbor before the ray comparison. Both gap offsets then act on that
premature hit. A follow-up removes only this internal ray blend in an isolated Automatic shader
and adds a nonperiodic background with an independent visible-background RGB oracle. Across
paired static plus seven-frame motion runs, background error improves about 11.1% at 4K/S25 and
3.7% at 1080p/S100, with unchanged marker depth separation. However, at the stronger setting,
mixed-edge outer leakage peaks rise from 0.222659 to 0.663975 and clean edges acquire leakage.
The deletion is rejected; retain the ray's foreground protection. This is separate from final AA,
which remains removed. No experimental shader was installed. The
[ray-blend isolation report](../cmake-build-relwithdebinfo/game3d-ray-aa-isolation-20260915/outcome.md)
records exact input identities, the fixed metric, timing limits and the next source-consistency
problem to resolve.

On 2026-09-15 the user explicitly requested a live test after reviewing the matched-strength
halo tradeoff. The experimental `3FF257A6` shader is now installed in the external repository,
Dead Space and Expedition 33. It combines the previously tested wider strength mapping with
the existing post-pass fusion, preserving the internal ray blend and original gap correction.
The actual 4K scRGB integration pair matches all 12 full-frame captures against the tested
wider-range control, including zero/current-source/missing-depth/recovery endpoints. All 18
compile/reflection configurations and 17 explicit rejection checks pass. Both B Automatic
presets start at strength 20, approximately the former maximum with the same camera reference;
Expedition now selects B. The Manual A presets and add-ons remain preserved. This is an
authorized experimental installation, not a change to the recorded quality verdict. See the
[live-test record](../cmake-build-relwithdebinfo/game3d-expanded-strength-live-20260915/README.md)
for exact identities, six-file backups and restoration conditions.

On 2026-09-16, after successful Streamline depth capture made the stronger range unnecessary,
the user requested restoring the original strength range. Automatic presentation strength again
uses the original maximum, retaining linear slider response and exact zero. The active mapping
is owned by the [Streamline depth contract](reshade-sbs.md#streamline-depth-selection); the
historical expanded-range comparisons above remain evidence for that earlier experiment.

## Adaptive reference and matched-strength investigation

The initial spread experiment addressed the raw model's dependence on its first central reference.
Its standalone prototype measured spatial variance over the complete sampled scene and pooled
coherent frame variances before establishing one fixed gain. It included finite depth endpoints,
removed no samples, and rejected flat, endpoint-only, sparse-contribution or unrepresentable
references instead of clipping their depths or the requested gain. The screen plane remained a
separate responsibility. At that stage the prototype was not connected to the production add-on;
the subsequent integration and its negative findings are described below.

Its artistic spread target is a shared presentation choice, not inferred camera geometry. Raw
affine scale/offset invariance of its ideal convergence equation does not prove invariance of
SunshineGame3D's prepared-depth textures, filtering or ray search. Different reference scenes
can still produce different gains; persistent spatial noise and depth/color registration require
evidence outside these statistics. The installed center-reference policy remains the control.

The actual ReShade comparison now has a separate test-only Automatic fitting path. It matches
the original's measured near/far positions in each eye, holds out a third depth, and retains full
source/depth/stereo captures. This isolates rendering quality from weaker stereo strength; its
injected camera controls do not validate automatic initialization. The comparison must remain
ineligible when later scenes fail the endpoint match, even if their edge diagnostics look better.
Quality acceptance also requires the stronger recorded Manual controls and moving silhouettes,
not only the simpler static fixture. Evidence is collected under
`cmake-build-relwithdebinfo/game3d-adaptive-reference-20260914/`.

The native comparison reproduces a fourfold gain difference in the current policy when the
same complete depth distribution is rearranged around the center; the spread candidate keeps
the same gain. It also exposes the candidate's remaining limitation: halving a scene's genuine
spatial depth spread doubles its initial gain. No shipping artistic target has been selected.
At that evidence stage the candidate and its admission/reset tests passed, but no production
initialization used it yet.

The actual default-control renderer pair matches endpoints across six static and fourteen
moving captures. The stronger recorded Manual controls change their endpoint separation with
scene content and fail the unchanged matching gate after calibration. Those captures remain
ineligible for a quality claim. A separate no-refit experiment equalizes Manual's neutral depth
preparation while preserving Automatic's ideal convergence field; it does not improve the
measured narrow halo. The detailed evidence and reproduction commands are in the stage's
`matched-disparity-review.md`. These results do not justify installing either a new calibration
policy or the isolated preparation-scale change.

The later `game3d-deterministic-strength-20260914/` pair removes the natural-clock
confound from the stronger-control comparison. The same prior Automatic H/t0 is
held without refitting, with 5,486 byte-identical shader-clock rows, 23 capture
records and 36 source/depth pairs. Manual still changes its near/far landmark
separations with scene content; the step/fringe near endpoint mismatch remains
0.28115045 px, exceeding the unchanged 0.25 px limit. Eighteen of 20 captures are
therefore ineligible for a matched-quality claim. The
[controlled-clock review](../cmake-build-relwithdebinfo/game3d-deterministic-strength-20260914/review.md)
records this as scene-response evidence, not an explanation of every game halo.

The subsequent `game3d-smoothing-isolation-20260914/` experiment compares Automatic
against itself with identical fixed camera controls and clocks at default and
stronger settings. Both pairs pass all 20 endpoint comparisons and 36 source/depth
identity checks. Replacing directional foreground smoothing with a center sample
increases detached mixed-edge color in both pairs; the stronger pair also worsens
thin-strip area error. The existing smoother is retained. The
[smoothing review](../cmake-build-relwithdebinfo/game3d-smoothing-isolation-20260914/review.md)
distinguishes relocated source fringe from invented color and records the limits
of the static-fringe and row-averaged thin-strip diagnostics. These renderer-only
runs do not validate automatic calibration or claim real-game halo elimination.

The subsequent `game3d-spread-integration-20260914/` stage integrates the all-sample spread
initializer into the raw policy's existing timing and lifecycle machinery. It removes the
prototype's duplicate calibration state machine and retains the native-camera center strategy.
Four focused native tests pass. Paired actual production-path 4K PQ runs also pass the independent
depth-readback oracle and existing source/color recovery checks. However, a narrow startup scene
chooses H=387.569458 versus the former 21.7372704. On a subsequent broader scene, 4.49018% of
prepared convergence falls below -1.5, before the renderer's later filtering and limits; this is
not a measurement of final clamped pixels. That exposes the remaining scene-dependence problem.
In a separate paired 1080p scRGB native-depth permutation, the exact same depth distribution with
a different center keeps H=4.16604519 across explicit recalibration, whereas the old center
initializer changes 2.66666675 to 1.60000002. The candidate therefore fixes the center-placement
dependence in that fixture without establishing good strength across different scenes.
Prior renderer-only parameter fitting is not evidence that the new automatic initializer chooses
good strength. The installed game files remain the earlier center-mean build.
The 16-effective-cell admission heuristic can reject legitimate sparse foreground and cannot
distinguish stable bad depth from correct geometry. No 2nd–98th-percentile trimming was added;
trimming only the estimator would still change the gain of all rendered pixels, potentially
increasing the displacement of the excluded near object.

The later `game3d-center-default-20260914/` stage restores the installed central reference
as the production default and keeps spread selection explicit in research tests. Five focused
tests pass, and the actual 4K PQ pair against the installed add-on preserves complete source,
depth, mono and final AA0/AA1 output bytes. Auxiliary preparation history still varies with
natural frame timing; its strict intermediate comparison remains failed in the retained report.
This restores the baseline while leaving scene-independent gain unresolved.

The isolated `game3d-ratio-response-20260914/` stage then tests a logarithmic
depth-ratio response through the actual original-derived renderer. It removes the startup
gain from that experimental response only. Across 316 scRGB/PQ captures, source/mono bytes
remain exact and all rendered values remain finite. Two selected finite depths match the
linear control's measured separation, but this does not establish agreement at other depths
or comparable strength to Manual. The experiment is rejected as a general replacement:
positive FP32 subnormals collapse to its infinity code on the GPU path, and exact-zero
backgrounds cause excessive separation with wide blank borders under both AA modes.
Filtering its infinity sentinel also has an analytically demonstrated false finite-depth
interpretation. The stage retains every endpoint and edge counterexample; no epsilon or
percentile clamp repairs the result. Production hooks and calibration remain unchanged.

The stage-only `game3d-online-gain-20260914/` candidate tests slow gain corrections
from full-scene spread and range, retaining all sampled depth values. Its CPU
contracts pass, but it does not solve initial-scene bias: two different startup
scenes followed by byte-identical later inputs settle at H=8.333333 and H=33.333332.
The fourfold hold band permits both equilibria. Walking toward fixed surfaces
reduces H by 85.6%, and a persistent six-cell near object leaves the original scene
with half its earlier gain after disappearing. These are scalar controller
counterexamples, not rendered-pixel or comfort measurements. The candidate stays
outside production; its [independent review](../cmake-build-relwithdebinfo/game3d-online-gain-20260914/review-analysis.md)
also records that the purported cross-boundary transient did not actually cross
an evidence-block boundary. Slower updates alone neither recover absolute scale
nor distinguish an unrepresentative startup from legitimate scene changes.

The next stage, `game3d-refining-gain-20260914/`, tests a common target with a
30-second evidence half-life and bounded gain motion. It retains all grid values
and reuses the existing policy's admission, clock and basis ownership. Its
default-policy and mechanical CPU checks pass, including a corrected ordering
case that discards an incomplete evidence block before the first observation
after a long tick gap. Identical later inputs bring startup gains 4 and 200 to
16.666643 and 16.666706 after 600 simulated seconds, eliminating the previous
fourfold terminal bias. The ratio is still 1.2843 at 120 seconds.

That temporal improvement does not make variance a robust strength estimator.
Six near samples out of 576 raise sigma from 0.0100 to 0.091916 and move its
instantaneous target H from 20 to 2.1759. A sustained object lowers the filtered
gain to 5.2354 before it eventually returns to the event-free trajectory. Ordinary
walking also changes the learned strength substantially. The
[retained CPU summary](../cmake-build-relwithdebinfo/game3d-refining-gain-20260914/trajectories-initial/summary.txt)
separates these tradeoffs from mechanical success. This remains a stage-only
experiment; neither rendering quality nor the target's match to preferred Manual
strength has been established, and production still uses the central reference.

The stateless `game3d-robust-scale-screen-20260914/` screen compares sigma,
population mean absolute pairwise difference and median absolute deviation
(MAD), preserving all input depths. One shared two-plane reference matches the
estimators' response; no per-case fit or depth clipping is used. Pair difference
reduces rare-object influence on a varied background, but a single near cell on
a flat background produces an ideal convergence field below the existing -1.5
limit. MAD is exactly zero for ordinary unequal two-plane populations and
discontinuous at their equal split. Both are rejected as unconditional scale
replacements. Its [numerical review](../cmake-build-relwithdebinfo/game3d-robust-scale-screen-20260914/review.md)
retains 48 oriented grids, 144 estimator cases and all 112 nonzero-estimator
requested fields. These are scalar counterexamples, not rendered quality tests.

## Prior independent-renderer investigation

The analysis below compares the earlier independent `SunshineDepth3D.fx` with SuperDepth3D v5.4.0.
It is retained as reference material, not the current implementation plan. Its earlier objective
was automatic setup without a game-profile database, a small interface, and comparable stereo
quality at a practical GPU cost. The proposed replacements below must not be read as changes
already made to SunshineGame3D or as demonstrated quality parity.

## Reference and evidence boundaries

The inspected Depth3D checkout is `E:/Git/Repo/Depth3D`, upstream commit
`be494f78e1fcbe1f14752a0035ba5d38955571f0`. Its local Sunshine integration changes export, readiness,
color handling and offscreen postprocessing. Original rendering equations must be distinguished
from those integration changes. In particular, a pristine upstream intermediate is not equivalent
to the final postprocessed on-screen image. Comparisons for our actual workflow use the original
SuperDepth3D renderer with the local full-resolution Sunshine export integration, identified by hash.
The separately supplied reference shader is not part of the Sunshine distribution.

Current evidence is under `cmake-build-relwithdebinfo/native-parity-review-20260913/`:

- `source-identities.json` and `baseline.json` identify the inspected inputs.
- `uniform-inventory.csv` inventories 148 SuperDepth3D uniform declarations and 10 independent
  shader declarations. These counts include hidden inputs, duplicates and compile-time alternatives;
  they are not counts of simultaneously visible controls.
- `uniform-review-coverage.csv` assigns every SuperDepth3D declaration to one of the review groups.
- Detailed depth, rendering and comparison-test reviews are kept with this evidence.

The previous backward-search timing compared two versions of our own renderer. It is useful
regression evidence but cannot establish performance relative to SuperDepth3D. Existing native
shader tests establish specific geometry, color and lifecycle properties. They do not prove
game-wide quality parity or correct depth/color alignment in an upscaled or generated frame.

## Archived proposals by responsibility

These proposals belong to the independent-renderer investigation. The current milestone retains
the original components while their baseline behavior is validated.

### Depth acquisition and interpretation

| Difference | Decision | Required evidence or remaining limitation |
| --- | --- | --- |
| Scene-buffer choice versus ReShade's generic draw-statistics selection | Keep our content-aware automatic selection and manual recovery. | Existing low-draw, full-scene and resource-recreation tests apply. A histogram is evidence of useful contents, not proof of correct scene/color association. |
| Reduced-resolution and padded depth | Keep our explicit captured rectangle; use verified SDK rectangles when available. | Validate offsets, dynamic resolution and source lifetime. Equal image dimensions are insufficient. |
| Normal/reversed hardware depth | Keep automatic clear-direction evidence and a troubleshooting override; prefer verified camera metadata when available. | Ambiguous clears must remain ambiguous. Clear value alone does not prove the projection. |
| Nonlinear conversion before convergence | Keep raw FP32 as the geometry coordinate for ordinary perspective depth. | In the basic original path, conversion followed by convergence reduces algebraically to an affine raw-depth mapping. Optional original shaping/filtering changes that conclusion and is reviewed separately. |
| Game-specific near-plane, offsets and depth-range profiles | Replace with shared, versioned metadata adapters where verified; do not introduce a new game-name database. | The installed Streamline probe currently observes only. It has not established live camera/frame/depth association. |
| q10–q90 startup gain | Replace as the final calibration policy. Keep the current implementation only as the frozen comparison control while alternatives are evaluated. | Startup scene composition changes its gain. Replacing percentile gain requires a specified fallback and actual rendered transition tests. |
| Absolute distance and engine units | Do not infer meters or physical IPD from an unlabelled raw texture or matrix. | Perspective coefficients reconstruct camera-space units only. Relative stereo initialization can cancel unit scale; that does not establish physical scene scale. |
| Orthographic, logarithmic or nonstandard depth | Admit only with a supported, validated interpretation. | A successful percentile fit does not make these inputs perspective depth. Broad support remains incomplete. |
| Near-one and reversed-Z precision | Keep FP32 raw sampling/subtraction and finite checks. | Existing near-one regression is relevant. Reduced-precision intermediates require an explicit error budget before adoption. |
| Clear, invalid and extreme samples | Reject invalid data; distinguish clear/background semantics from statistical outliers. | Do not discard a valid near object because it occupies a small image fraction. A clear value can represent sky/background and needs an explicit rendering policy. |
| Mismatched or missing current depth | Keep current-frame readiness and mono fallback. | Retain camera preferences for recovery but never turn stale geometry into a new valid frame. Generated frames require a separate matched-frame policy. |

### Stereo camera and adaptation

| Difference | Decision | Required evidence or remaining limitation |
| --- | --- | --- |
| Depth Adjustment, ZPD, balance and many comfort parameters | Keep two primary controls: 3D strength and Screen plane. | Their effects must be independent and measurable before any limiter. Names should describe the visible result. |
| Hard plateau at extreme calibrated depths | Prefer a monotonic, bounded transition to saturation over the current abrupt plateau, if measured quality supports it. | Preserve central pairwise relief and thin-object ownership. Any nonlinear limiter still compresses geometry; it must be evaluated explicitly. |
| Automatic depth/range adjustment and Smart Convergence | Use adaptation to acquire valid interpretation and recover from verified changes. Do not make ordinary walking continuously rescale the scene. | Test approaching a wall, looking down/at sky and returning, and foreground objects entering/leaving. Smoothing alone does not remove unwanted convergence drift. |
| Camera-aware gain initialization | Evaluate a representative reference depth with globally defined relative camera strength; hold the resulting camera separation thereafter. | This avoids calibrating gain from a depth-range percentile span, but initial reference selection still depends on scene evidence. It must not be advertised as metric reconstruction. |
| Zero-plane adjustment after initialization | Apply a common disparity offset while keeping the chosen separation. | Recomputing gain whenever the screen plane moves changes relief as well as placement. |
| FOV, near/far and resolution changes | Update projection interpretation while preserving declared camera preferences when continuity is proven. | Define the zoom policy explicitly. Resource identity is not camera identity, and SDK history reset is not a request to move the screen plane. |
| Temporal smoothing and scene resets | Preserve stable camera state; discard only incompatible evidence on a qualified transition. | A depth histogram alone lacks spatial layout and is insufficient scene-cut proof. Match queued samples to their original source/camera epoch. |
| Screen-edge avoidance and fades | Evaluate as bounded presentation assistance after correct geometry, with clear ownership. | Do not hide geometry defects with global fades or add a family of game-specific edge masks. |

### Rendering, sampling and postprocessing

| Difference | Decision | Required evidence or remaining limitation |
| --- | --- | --- |
| Original parallax search versus our inverse search | The prior investigation retained the independent inverse search as its control. | Both use backward reconstruction. The relevant comparison is intersection accuracy, ownership, sampling and cost, not the forward/backward label. SunshineGame3D retains the original search for the current milestone. |
| Depth dilation before the original search | Evaluate a third approach that conditions only unsupported or misaligned boundaries. | Blanket foreground expansion can remove fine background detail or thicken thin objects. Measure fringes, silhouettes and thin-object survival separately. |
| Original depth reconstruction and directional smoothing | Use edge-aware reconstruction only when it improves the source-depth footprint. | Current smooth-neighbor interpolation and cliff footprints need comparison against sloped surfaces and low-resolution depth. Neither arbitrary blur nor all-point sampling is a universal winner. |
| Search steps and quality/performance presets | Prefer a bounded adaptive workload driven by disparity and actual depth resolution. | Preserve every narrow surface the quality contract promises. A smaller step count is not an improvement if it misses thin geometry. |
| Gap filling and de-artifacting | Keep our background-only donor protection as a baseline; compare alternative edge-aware fill. | Unknown background cannot be recovered from a single image. Report boundary-fringe suppression separately from invented-background accuracy. |
| Horizontal geometric AA versus output AXAA | Compare both and a selective two-dimensional edge treatment. | Our current AA is horizontal and geometric; original AXAA is output-image filtering. Compare new stereo-edge aliasing and preserved texture contrast at matched disparity. |
| Color resampling | Keep linear-light interpolation for SDR/PQ and signed scRGB handling. | Verify color and texture contrast independently. A gamma-space result can look sharper without being radiometrically correct. |
| Sharpening | Do not make whole-image sharpening a substitute for correct sampling. Evaluate a small confidence-aware option only if needed. | Current saved original game presets had sharpening disabled; its mere availability cannot explain the observed difference. Measure ringing as well as contrast. |
| Debanding and saturation | Keep native game appearance by default. | These are optional color effects, not prerequisites for stereo reconstruction. |
| Temporal color/depth history | Use only with validated motion, jitter and frame association. | Unqualified history risks trails, ghosting and stale outlines; the current independent effect has no temporal reprojection. |
| Foveation and reduced-quality regions | Defer until a supplied gaze/display contract makes them useful. | Do not lose peripheral thin-object quality merely to match an original performance control. |
| Confidence and diagnostics | Keep validity/confidence separate from pixel color and add evaluation diagnostics outside the normal workflow. | Filled pixels must not be scored as observed geometry. The same gray preview is not proof of matching disparity. |

### Layers, output and interface

| Difference | Decision | Required evidence or remaining limitation |
| --- | --- | --- |
| Weapon/hand depth adjustments and masks | Prefer verified separate layer/projection metadata. | Generic depth thresholds cannot reliably distinguish a weapon from nearby world geometry. Automatic handling across games remains incomplete. |
| HUD/menu detection and alpha isolation | Prefer an actual game-supplied layer when available; keep the normal flow simple. | A final composited image and scene depth do not uniquely identify HUD pixels. Do not replace profile rules with unsupported visual guesses presented as certainty. |
| Native game display and Sunshine export | Keep normal-resolution mono on the game display and separate full-resolution SBS. | Existing production exporter/receiver and exact mono-preservation tests apply. |
| HDR compatibility switches | Keep source-transfer detection and explicit SDR/scRGB/PQ export semantics. | Windows HDR state or texture bit depth alone does not establish the captured transfer. |
| Original image scaling, stereo layouts, anaglyph, interlace and frame packing | Keep these out of the native reconstruction shader. | Presentation/layout belongs to the consumer. The current contract is full SBS with a declared eye order. |
| Eye swap and screen optics | Place display-specific corrections with the presenter/client. | Avoid duplicating correction in both producer and consumer. |
| Barrel distortion, chromatic correction, theater mode and vignette | Leave to the display/presenter contract. | They are not missing depth-reconstruction stages. |
| Stereo cursor and ReShade controls | Keep the working stereo overlay composition; treat the game cursor as a distinct layer problem. | Current overlay tests cover native controls in both eyes, live changes and reload. They do not establish every game's cursor policy. |
| Automatic selection and manual repair controls | Keep automatic selection enabled; make advanced buffer heuristics and diagnostics troubleshooting controls. | The main user workflow should not require shader source edits or a table of per-game numbers. |
| Source/UI terminology | Keep 3D strength and Screen plane prominent; expose technical status only when useful for diagnosis. | The declaration inventory is a completeness aid, not a mandate to reproduce 148 original controls. |

## Deferred independent-renderer follow-up

These gates remain relevant if the independent approach is revisited; they do not authorize
replacing original components before the current baseline is established.

1. Compare the two actual shaders with identical source/depth data and measured matching endpoint
   disparities. Report an independent third-depth holdout so a transfer-curve difference cannot
   be mistaken for better edge reconstruction.
2. Compare AA off/on, texture contrast, silhouettes, diagonal boundaries, thin objects, sloped
   surfaces and uncovered gaps. Preserve raw floating-point output for HDR analysis.
3. Time the same GPU scope with the same dimensions, input data and completed warmup. Include
   the full original preparation/post chain required for the compared output.
4. Validate a replacement calibration/limiter on actual rendered trajectories, near/far changes,
   view changes and source recovery. Keep strength and placement behavior explicit.
5. Confirm camera/depth association and moving-image appearance in actual games, including an
   upscaler case. The newly installed Streamline observer has no gameplay evidence yet.
6. Measure UI/setup burden and compatibility on a broader game sample before claiming "most games."

The earlier percentile-calibration implementation remains a reference, not the final automatic
camera policy for the original-derived milestone.

## September 17 normalization candidate review

The September 17 depth-patch calibration experiment added real depth-map observations and a
background encoding estimator, but did not meet its rendered-pair accuracy gate. That experiment
and its later motion-vector replacement have been removed from maintained source; archived
evidence remains historical. Production uses an associated projection or the explicit infinite-far
raw-depth assumption described in the [current contract](reshade-sbs.md#experimental-raw-depth-automation).

The [offline normalization comparison](../cmake-build-reshade/scale-normalization-20260917/review.md)
evaluates 6,263 synthetic raw-depth grids with the actual shared gain smoother and
30 controlled original-derived shader cases. It compares the current raw-center
reference, variance, three range-limited variance budgets, effective span
`2*variance/mean(abs(t-mean(t)))`, and whole-image mean at matched reference gain.
The [opt-in experiment tools](../tools/reshade/experiments/README.md) retain
reproduction instructions and distinguish scalar evidence from renderer checks.

Effective span fixes the constructed two-plane object-coverage dependency, but
admitted shallow walls still select excessive gain that persists on returning
to a room. Range-limited variance shares that failure and suppresses normal
geometry strongly when an isolated extreme controls the range. Whole-image mean
avoids the wall case but over-amplifies centered foreground against near-zero
background depth. None earns adoption as a complete adaptive normalization rule.
Those experiments did not change the installed production policy. Finite HDR output and correct
prepared-depth values in the renderer checks do not establish comfortable
disparity, better halos, or a replacement for independent depth/encoding evidence.

The subsequent [known-motion depth-calibration experiment](../cmake-build-reshade/motion-depth-identification-20260917/review.md)
fitted the physical observation equation `1/Z=alpha*raw+beta` using known relative
camera pose and matched image points. Across eleven informative static-scene
fixtures, both bounded online QR and windowed reprojection fits recover the
coefficients within 1%, with final slope errors below .04%. Room-to-wall scene
changes do not force stronger gain. Flat/rotation-only inputs are unobservable;
shallow or tiny-motion inputs fail the windowed precision gate. Explicit unit
and coherent-motion counterexamples retain good residuals with wrong scale,
demonstrating the need for trusted pose and static-scene evidence. This calibrates
depth in pose units, not artistic stereo strength. The game logs at that time lacked the
matched motion/depth/pose dataset needed for live validation. The research implementation
has since been removed; its frozen source and results remain with this archive.

The [native online-calibration follow-up](../cmake-build-reshade/motion-depth-confidence-20260917/review.md)
implemented bounded QR accumulation with initial-publication confidence, then
information-weighted refinement without repeated readiness timers. Its frozen
native replay passes 26 checks across the same 840 synthetic frame pairs. Weak
scenes retain calibration, while flat-only, shallow-only, pure-rotation and
tiny-motion startup cases remain unpublished. An independent review exposed
history poisoning by a high-information contradictory pair; the fixed policy
checks consistency with prior uncertainty before merging. The native regression
also admits a statistically supported correction whose prior pixel error exceeds
a fixed residual gate, and verifies raw-encoding invariance.

This was an offline calibration component, not a shipping K/H replacement. Its proposed
shared result was inverse-depth scale and offset, computed from valid SL projection or
estimated from trusted geometry, with strength applied afterward. Live calibration would
have required matched pose, motion and depth evidence. Coherent object motion, omitted jitter
and unknown translation units were explicit limitations of the conditional fit. This proposal
is historical; current production uses the simpler projection/relative-depth contract above.

## September 17 stable stereo reference experiment (superseded)

The first implementation shared one fixed reference policy between projection and relative raw depth.
For decoded inverse depth `q`, the signed stereo field is proportional to
`strength * Zref * (q0-q)`. `Zref=1/qref` is established from four admitted center
captures and remains fixed during ordinary scene changes; the screen plane `q0`
continues to move independently. The initial reference and zero plane coincide,
but serve different purposes. Scaling every game-space distance by the same factor
leaves the rendered ratios unchanged. **Reset reference** replaces both values from
fresh evidence on either path. The UI labels the conversion **Projection** or
**Relative (uncalibrated)**, instead of presenting raw normalization as recovered geometry.

The [actual-shader evidence](../cmake-build-reshade/unit-independent-reference-20260917/stereo-reference-README.md)
passes 1080p scRGB and 4K scRGB/PQ. Known unit/encoding variants produce byte-identical
exports. At 4K, 100% and 50% strength produce measured pairwise separation of 15 and
7.5 pixels, and room → wall → room keeps that separation while the zero plane moves.
The archived fixture also tested a synthetic known-motion fit against known projection
coefficients and produced the same exported pixels. That estimator consumer has been removed
from the current fixture, which retains the finite-far matrix oracle cases. These were functional render checks, not live game or
performance results. The unchanged shader exposes an existing zero-plane offset
at 640×360; that failing result is retained in the evidence and is not waived.

At that milestone the geometric estimator lacked a trusted live observation producer.
Without valid projection, the fallback used oriented raw depth and could not
recover an unknown additive inverse-depth offset. A fixed reference also retained the
artistic scale selected by its startup view; it is not a universal preferred baseline.

## September 17 current-zero reference implementation

The revised design uses the current smoothed zero plane as its reference. One shared
controller owns initialization, target freshness and zero-plane smoothing. Rendering derives
`K=1/q0` or `H=1/t0` from the same published float zero, giving the signed field
`0.05*(1-q/q0)` before user strength. There is no independent gain or permanently retained
startup reference. Different starting scenes should converge to the same current geometry.

This intentionally couples placement and normalization: moving the screen plane changes
separation between two fixed scene depths. It is not a constant-baseline camera model.
Current projection A/B remain exact; Generic/NGX without projection retain the explicit
relative-raw limitation. Numerical tracking continues during a shader-domain rejection so
valid evidence can recover without manual reset, while unsupported rendered geometry stays mono.

The UI separates matrix depth-conversion scale/offset, zero-plane current/target, stereo
normalization and **Recenter**. It explicitly identifies normalization's current coupling to the
plane. Raw zero is labeled relative depth, not distance or geometric calibration. The initial four captures
initialize the plane only; later center evidence is smoothed with the existing 0.5-second
response. User strength remains a separate 0–100 multiplier after normalization.

The [current-zero validation](../cmake-build-reshade/zero-plane-reference-20260917/IMPLEMENTATION.md)
passes all 30 native groups and actual Generic/NGX/SL capture fixtures, including current
projection changes at 4K. Full shader checks at 1080p scRGB and 4K scRGB/PQ preserve unit and
encoding invariance, zero disparity at the moving zero plane, and convergence from different
initial scenes. At 4K, the independently predicted room/wall/return span is 15/7.5/15 pixels;
the rendered output matches. No deployment or live-game quality result is claimed.
