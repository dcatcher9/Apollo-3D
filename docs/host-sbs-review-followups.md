# Host SBS subtitle review follow-ups

This tracks the static review supplied in `CODEX_HANDOFF.md` on
`claude/exciting-meitner-iarw5l`, commit `166d1018f7b64bba004d59999abfaf3476b990db`.
The original confidence labels are retained: V = independently verified in that review,
R = reviewer-confirmed, U = unverified, X = rejected. They are not claims of a local GPU
reproduction. This is a work tracker, not an alternative algorithm contract; policy remains in
[Host SBS](host-sbs.md).

## Priority repairs

| ID | Original | Disposition and remaining evidence |
|---|---|---|
| HSR-01 | U | Confirmed and repaired missing-OCR failure: allocate resolve identity constants independently of the detector and require working probability only for optional OCR. The real missing-asset GPU test passes two depth observations with authenticated abstention, exact Base output and zero subtitle masks. Oversized full-source captures are an intentional separately enforced source boundary, not a newly established OCR regression. |
| HSR-02 | R | Repaired geometry-sensitive semantic identity and modal identity withdrawal. A freshly revalidated matching root/process can retain an already admitted native-player or browser-video context for suppression; native registration or the matching fresh browser heartbeat remains mandatory, and no new context is admitted during movement. Positioned ROI/render authority stays separate. Protected packed fullscreen pixels now bind current foreground/region/browser/modal authority, preventing retained semantic identity from replaying an old treated image; full-source Base-only reuse stays independent. Modal draws still use fresh Base, and real ROI/full-source or resized-domain transitions retain their required resets. Galaxy XR acceptance remains pending. |
| HSR-03 | R | Confirmed and repaired native-player reference mismatch, explicitly encoded in the previous policy. Foreground-client analysis ROIs now supply the caption reference, with matching dump validation and generated identity. The production classifier regression retains the same caption at three translated native-window positions and rejects it under the former capture reference. Full-source fallback has no player rectangle and remains screen-relative. |
| HSR-04 | confirmed | Repaired generator ownership of all five contract HLSL copies and three OCR assertion copies, plus automatic check/refresh of the runtime closure pins. Refresh now writes generated UTF-8 outputs only when their exact bytes change, preventing no-op regeneration from triggering broad rebuilds. Eight additional no-op, changed/missing-output and read-only-check cases pass; generated identities and check semantics are preserved. |

## Remaining bugs and stability work

| ID | Original | Status and next action |
|---|---|---|
| HSR-05 | V | One genuine empty OCR observation immediately returns to Base. This is documented current policy. A debounce would require a coordinated cover/controller/matte policy change and appearance/disappearance evidence; holding only the controller is insufficient. |
| HSR-08 | R | Left-aligned multiline captions are intentionally outside the confirmed supported scope; their rejection is not a bug or fix priority. The per-line centering rule stays unchanged. Actual production OCR-box coverage characterizes this boundary: rejected lines do not teach the prior, and independently centered rows retain all three pairs. EMPTY still clears current covers. Ribbon-prior effects remain open; empty-observation policy is tracked separately in HSR-05. |
| HSR-09 | R | Repaired the queued-renderer dependency and encode-thread preparation: completed live configuration omits queued probe assets and a tracked background future prepares programs for one-time adoption. Shared bytecode flights now settle attached callers with the original exception and evict only the failed entry, permitting retry without an old failure erasing a replacement; setup, observer and persistence failures cannot poison a successful flight. Three new deterministic flight tests and twelve real compile/cache cases pass. Cold completed-PS compilation remains; earlier diagnostic loads are not production startup or performance comparisons. |

## Performance investigations

All timing claims below are review estimates. Measure serially on an otherwise idle GPU before
calling them production costs or improvements.

| ID | Original | Status and next action |
|---|---|---|
| HSR-11 | V | Repeated lane-zero authorization across the contour passes. Investigate a shared immutable verdict without weakening frame/owner validation. |
| HSR-12 | R | Rejected the unused-P010-MRT claim: authenticated HDR/P010 Base fallback can supply luma and skip later conversion. Exact repeated images already reuse packed output without issuing a new draw. Remaining draws use changed mutable RGB or a new publication; pointer equality and the old draw lease cannot prove immutable C24 or b3 contents. No unsafe cache was added. Profile actual issued draws before introducing stronger material/publication generations. |
| HSR-14 | V | Repaired native-band allocation and redundant feature copying. Hard disable uses one-pixel native placeholders; suppression preserves existing backing/pixels. CPU-completed force-infer roots publish SCT5 through an authenticated identity-only entry and exchange complete feature owners. Receipt-gated/queued roots retain the old conditional copy, so zero-group reuse never swaps owners. Real GPU copy-versus-swap parity across observation boundaries and poisoned-scratch reuse tests pass; count and duplicate semantics remain unchanged. Generated producer identity is schema 127. |
| HSR-15 | R | Quadratic lane-zero grouping of OCR pieces on dense text. Add representative dense-text measurements before replacing the algorithm. |

## Maintenance and coverage

| ID | Original | Status and next action |
|---|---|---|
| HSR-18 | R | Corrected two active C++ ownership headers that incorrectly claimed DRAFT/non-runtime status. Probe-only queued, pending-color and body-witness paths still share application translation units; broader separation/naming remains maintenance work and does not justify removing their explicit diagnostic callers. The owner later retired these probes; they were deleted on 2026-11-25. |
| HSR-19 | R | Centralized selected-receipt ABI1 C++ storage and its C24 byte-copy boundary: 64 words/256 bytes, words 4--27 copied as the exact 96-byte constants. The header `src/host_sbs_selected_matte_receipt.h` owns these types/constants. Consumers still independently authenticate freshness and rendering authority. Wider HLSL/Python word maps and SCT/SCP validation remain separate work. |
| HSR-21 | U | Ordinary `run_eval` images use only Base and pass no subtitle flag, so they do not measure the live subtitle route. The file-fed [Direct Glyph qualification](../tools/sbsbench/README.md#direct-glyph-qualification) is the only harness qualifier of that route; it has no Python launcher or independent reader, and it does not establish live capture scheduling or headset quality. Evaluator/live metric parity remains open. |

## Rejected claims retained for traceability

| ID | Original | Disposition |
|---|---|---|
| HSR-23 | X | First cue after an empty observation publishing target zero was rejected as unreachable because geometry readiness enforces the observation clock. Do not treat it as a confirmed defect. |

## Validation record

Local Windows qualification on November 2, 2026:

- The default RelWithDebInfo build passes, as do generated contract/closure checks.
- The real missing-OCR-asset GPU regression passes two joined observations with
  authenticated abstention, exact Base output and zero subtitle masks.
- Native context/resolution, OCR shader and authenticated pixel-contract coverage has 107 passing
  cases and one skipped external saved-capture diagnostic. The new native-window test initially
  exposed a test helper's fixed frame-ID assumption; its corrected explicit identity assertion
  passes on rerun. Production code needed no change for that assertion failure.
- The evaluator suite ran 902 tests with 13 skips. Its only failure read the canonical identity
  paragraph before the documentation update; both changed modules subsequently pass all 201 tests,
  including the identity assertion and new runtime-pin drift checks.
- The matched three-clip, 72-frame control/treatment and authenticated report pass with a
  `screen_neutral` decision. Every per-frame metric and every aggregate metric is identical,
  and both runs report no issues. The validated interpreter/package fingerprint matches before
  control, treatment and report verification. These are functional, not GPU timing comparisons.

Continuation qualification uses `handoff-pass2-control` and `handoff-pass2-treatment`:

- The default RelWithDebInfo build and final native rebuild pass.
- All 18 source/lifecycle boundary tests pass on rerun. Two stale assertions also failed on
  `HEAD`'s unchanged product code: a reset-call count predating three safety additions and a
  completed-only literal predating queued/completed domain binding. They now check each required
  ownership boundary within its function region instead of obsolete counts or literals.
- All 201 evaluator contract/dump tests pass. The fresh 72-frame matched report again returns
  `screen_neutral`, with identical per-frame/aggregate metrics and no issues on either side.
  Interpreter/package fingerprints match exactly; no numeric-runtime exception was used.

A further continuation qualifies modal/cache and allocation repairs:

- The default optimized build and final native rebuild pass. All 62 focused native checks pass,
  including real joined inference for initial suppression, hard-disable placeholders, preserved
  same-context history, causal modal identity and packed-presentation revocation.
- Full evaluator discovery passes 906 tests with 13 skips.
- `handoff-pass3-treatment` against authenticated `handoff-pass2-treatment` passes with a
  `screen_neutral` report. All aggregate, per-frame and worst-frame metrics are identical across
  72 frames, with no issues on either side. The exact numeric-runtime fingerprint matches before
  treatment/report and after verification. This continuation changes no active shader bytes,
  geometry or scoring policy. Two attempted replacement controls were rejected for provenance
  drift/setup before qualification and were discarded; neither supplies report evidence.

A further headless investigation studies the OCR line classifier before considering a detection
change:

- Working OCR diagnostics capture all 96 observations across the four authored subtitle clips.
  Hashes, current inference/optional-OCR transactions, probability/cell equivalence, source crops,
  clocks and completed OCR records are checked. This is file-fed diagnostic evidence, not live
  scheduling or performance qualification.
- A road-marking OCR false positive already passes the current midpoint and center-span tests.
  Its native core is about 186 pixels tall. The learned position prior cannot veto its independent
  horizontal classification. Curvature in its eight probability-envelope bins averages to
  horizontal fitted slopes; it is not a stale-state or first-frame initialization diagnosis.
- Straight-envelope hypotheses are rejected before adoption. Requiring both envelopes to fit
  hypothetical half-detector-row intervals demotes independent-horizontal evidence on 59 of 145
  retained pairs with authored glyph overlap. A common six-bin variant demotes nine. Even the
  weakest tested either-envelope/six-bin variant demotes a single-island caption candidate with
  4,191 authored glyph pixels inside its core. These are saved-input geometric counterexamples,
  not measurements of final GPU mask withdrawals. Ink quantiles vary with glyph shape, so the
  interval width is not a proven text-baseline error bound. No detection threshold or shader changes
  are adopted from this study.
- A genuine empty OCR observation still clears treatment under the current contract. Neither a
  controller-only hold nor replaying old masks supplies current caption authority.

The reproducible local envelope analysis and authenticated packet bindings are retained in
`C:\Users\DCatc\.codex\tmp\apollo-subtitle-review-pass5-20261102\ocr-envelope-diagnosis.json`.

A subsequent headless continuation repairs two infrastructure failures:

- Refresh preserves the modification time of every generated output whose bytes are unchanged; a
  second full refresh changes none.
- Shared bytecode flights settle the original failure for attached callers, evict only that exact
  entry and permit a deduplicated retry. Three deterministic failure/clear/retry cases and twelve
  real compile/cache cases pass. Optional persistence failure cannot revoke shared success.

Use the local runtime and build requirements in
[AGENTS.md](../AGENTS.md), the [repository workflow](../CLAUDE.md), and the
[evaluation procedure](../tools/sbsbench/README.md#required-evaluation-loop).
Ordinary paired evaluation covers shared depth and Base only; HSR-21 limits what that evidence
establishes about the live subtitle route.
Galaxy XR visual acceptance has not been performed. Modal suppression and native full-source
fallback still limit visible caption treatment; preserving semantic identity does not eliminate
fresh Base frames or legitimate analysis-domain resets while a window is dragged or resized.

## Live packed-presentation clock follow-up

The subsequent live session exposed a separate presentation-clock defect: an exactly unchanged
packed image could remain valid indefinitely, but the first changed delivery inherited its entire
static age and could exhaust the `250 ms` hold immediately. The repair uses the smaller of cached
source age and time since the first observed known-source divergence, without restarting that
interval on retries or failed admission. Matched-completion age, route/context/transfer guards
and geometry/subtitle authority remain separate. Timeout/recovery diagnostics now follow the
successfully recorded output, including queued-to-flat fallback.

Eight focused clock/gate cases and the existing presentation/source-boundary suite pass. The
opt-in actual-converter regression also passes with real inference, the ordinary completed RGB
converter and a private `350 ms` joined-event gate with an independent `500 ms` safety bound.
After about `300 ms` idle, the first
changed B delivery repeats actual packed A whose raw age is `377.240 ms`. With B still genuinely
NOT_READY, a later delivery beyond the `250 ms` hold records current flat B; after the gate
releases, the original B source genuinely completes and recovers without resubmission. Packed
bytes equal the retained actual A output, and expired flat bytes equal a separate GPU reference
using the production flat shader and literal current B source. All five output-file hashes were
rechecked. The original failed fixture trace is preserved: its assertion incorrectly compared
model publication 1 with actual RGB draw 2; the corrected test binds those distinct identities.

Evidence is retained under
`E:\ApolloDev\host-subtitle-review-pass7-20261102\continuity-probe-result-v2\actual-converter-packed-continuity-v1`.
The result JSON SHA-256 is
`928d1a0412aa5683b89e785ed24379b58bfbaebb11c634f6c50ff0b7044ac0e1`.
The final C2 build invocation also passes the same functional probe, with its result retained in
`continuity-probe-result-candidate2\actual-converter-packed-continuity-v1` under the same pass7
directory. Its result SHA-256 is
`06ba536ebbfb125dbe1e673f4a955e07e9aaf41aae8341bdafdb382430388f01`.
Independent re-reading verifies all five output sizes/hashes, unchanged A/packed-A and
flat-B/reference-B pairs, actual NOT_READY before expiry and after the cap, and a genuine ready
B publication with no pending slot after release. All target readbacks follow that release.
This is a functional scheduling regression in an explicit synthetic eligibility context. It does
not qualify live capture classification, encoder/headset behavior, FPS, or the broader observed
stalls.

## Shared UI warp and later review rounds

### P14 shared Game 3D pin-band comparison

This comparison is the measured basis for the adoption in P15. The Host compositor it compares
against was removed afterwards (P18), and the masks it used came from the mask producer of that
time.

The native-mask comparison tests the then-current completed Host compositor against Game 3D's
single conditioned-field warp. Both routes receive the same original colour, Base, source
rectangle, render geometry, current native M, source snapshot, SCP target, SCT and selected
receipt. Neither route gets additional alpha or background evidence. The existing eighteen
saved-renderer routes remain independently checked against the authenticated G1 control.

The pin-band scan and arithmetic now have one owner,
`src_assets/windows/assets/shaders/directx/include/ui_pin_band_shared.hlsl`. Game 3D retains a
generated, configure-checked copy for its self-contained embedded compiler. All 258 optimized
SM5 binaries across eighteen Game 3D configurations are byte-identical before and after this
extraction. The private Host adapter includes that same owner; it is not selected by production
streaming or conversion.

The adapter constructs a native-resolution field over the subtitle band, applies the shared
pin pass, and samples original source RGB through one inverse. It performs no background RGB
reconstruction. Empty rows preserve the original Base inverse; both positive bilinear row
contributors are considered for fractional vertical sampling. Current RGB revocation and the
literal legacy fallback remain separate from field construction. A fresh GPU-owned header
records selected-receipt validity for that immutable draw only.

An initial native-field construction failed the horizontal slope check: hardware-filtered
coarse-to-native sampling produced approximately 0.50386 native pixels per adjacent texel in
both the unpinned and pinned fields. That attempt is preserved as a failed diagnostic. Explicit
FP32 interpolation in the native-field initializer restores the bound within the predeclared
FP32 roundoff allowance. The accepted comparison retains the original checks and an unpinned
resampling control.
The main production warp is unchanged.

The finite quality result does not justify replacing the current Host default:

- In ten authored stress cases, both methods reproduce opaque inner body exactly in nine.
  Body antialiasing is exact wherever that cohort exists in those cases. The shared warp better
  preserves the deliberately omitted stroke patch in the tenth.
  Its local clean-background error is worse in nine cases and tied in one. A thin-diagonal case
  rises from about 2.97 to 7.10 RGB8 mean error; a checkerboard rises from 13.19 to 26.43.
  A fractional-vertical outline case also worsens. Native proof images show the scene deformation
  and horizontal row transitions rather than treating fewer changed pixels as better quality.
- Four independent known-layer cases give the complementary result: both methods have identical
  inner-body errors, while the shared warp reduces decoration error in all four, substantially
  on the white, colour and black cases. Local background results are mixed. These fixed older
  masks deliberately test renderer robustness; they do not measure current G1 detection recall.
  Their old captured state fails current authentication and is never relabelled. A separate,
  explicitly authored current fixture context preserves their original pixel/Base/M/geometry
  bytes and numeric plane target for both routes.
- Three real HDR tuples use the actual current G1 producer masks. Full marked UI footprints
  preserve the same rigid source sample in both methods. The shared warp avoids some fractional
  repair fringes, but can deform nearby scene edges. These dumps have no clean-background truth.

The shared adapter uses about 15.0 MiB of logical native-band working storage on the saved
3840-wide cases, versus about 66.45 MiB for the push/pull owner of the compared compositor, excluding common inputs,
diagnostic outputs and driver overhead. Balanced warm preparation-plus-draw timestamps are a
private component comparison. They include submission gaps and fresh fixture resources; the
test executable is coverage-instrumented. They establish no live FPS or headset-latency gain.

The optimized native test build passes the separately selected saved-input comparisons and the
48 colour/ROI/signed-displacement/fractional-sampling combinations. The renderer regression
group runs 33 tests: 25 pass and eight external-input tests explicitly skip; the relevant saved
input paths run separately. Six generator/drift tests pass. All Python evaluation and scoring
uses the single machine interpreter and numeric fingerprint from AGENTS.md.

Evidence is outside Git under `E:\ApolloDev\host-game-ui-warp-comparison-20261104`.
`final-comparison-summary-v1.json` binds the final strict quality reports, proof images,
performance samples and frozen analysis code; `final-validation-v1.json` binds the source,
native executable, regression checks and unchanged live launcher. Initial compilation/setup
failures and the rejected field construction remain distinct from accepted evidence.

Keep the current Host renderer as the default under the preserve-or-improve requirement. The
common pin implementation and private adapter make a shared solution testable without separate
copies of its algorithm. A future two-axis pin band could bound the visible row transitions,
but would spread deformation further and needs independent false-positive, ROI and scene-edge
qualification before either product adopts it.

### P15 production shared UI warp

The subsequent explicit product decision selects the shared warp despite P14's mixed quality
result, superseding its recommendation to retain the Host default. This is an adoption decision,
not a reinterpretation of the measured background regressions as improvements.

Host selected rendering uses the same `include/ui_pin_band_shared.hlsl` owner as Game 3D. The
production adapter builds a native full-width subtitle band from authorised Base, including its
ROI embedding, applies current-RGB-revoked native mask contributors, pins the horizontal band,
and samples original SDR/HDR colour once through the existing bounded inverse. It prepares no
subtitle RGB repair background on the successful shared route. Mask detection, original outline
style, SCP plane policy and retained inference ownership remain unchanged. The queued wrapper
keeps its independent exact-source guard. Empty/revoked subtitle inputs preserve Base. The original
adoption retained a literal legacy fallback; the subsequent shared-only cleanup below removes it.

The actual shared draw is identified in qualification and dump evidence with its draw-owned
field, weights, rows/header and geometry, separately from the retained legacy analytic diagnostic
map.

The established limitation remains: horizontal pinning deforms nearby scene geometry and does
not preserve a vertical shear bound at UI-row transitions. No two-axis redesign or synthetic
outline is included. The P14 source-matched quality and performance evidence remains unchanged;
finite timing results establish no live FPS or headset-latency gain.
Eligible draws with an inactive plane or empty mask still incur native-band preparation; this
adoption does not establish a speedup for every frame without captions.

Production integration evidence is kept outside Git under
`E:\ApolloDev\host-shared-warp-production-20261104`, including the authenticated pre-change
control, build and regression results, selected-renderer qualification and final launcher seal.

The optimized production build passes the 53-test renderer/ownership group (45 pass, eight
external-input skips) and all 58 native dump tests. The enabled shared matrix compares production
and independent preparation across 48 colour, ROI, displacement and fractional-sampling cases,
including current-colour and stale-receipt rejection, empty-row reuse, state restoration, and
queued guards revoked after preparation. The three saved HDR tuples run through a fresh producer
and matching draw binary; all 24 colour crops and 36 pin diagnostics match accepted P14 payloads
exactly. An initial draw attempt correctly rejects a producer from the preceding binary; it is
preserved separately rather than waived.

The schema-4 actual selected qualification passes all 24 frames, with 17 active pin frames and
zero repair backgrounds. The actual `display_vram` HDR smoke observes the shared program and
exact pre-Draw bindings on both captioned sources (64,278 and 64,134 selected pixels), followed
by zero selected pixels on the authored empty source. Its native pin owner advances per draw and
no repair background is queued or bound. These are wiring and ownership checks, not headset or
clean-background quality acceptance.

All 955 Python evaluator/reader tests pass with 13 declared skips; six shared-owner generator
tests and contract drift checks pass. The ordinary three-clip paired evaluation is screen-neutral
and preserves all 576 PNG/F32 payloads byte-for-byte. Every evaluation stage uses the exact
AGENTS.md Python runtime and package fingerprint. The final receipt binds the merged commit,
source files, optimized binary, runtime assets, validation outputs and live launcher.

### P16 shared-warp review and cleanup

The review preserves the shared pin shader, ordered source pins, calibration, G1 mask policy
and original RGB sampling. It fixes three diagnostic/resource edge cases:

- Reject a native band whose extra row-header texel exceeds the D3D11 texture limit before
  allocating either large plane. Such a band previously failed later during row allocation.
- Keep cached diagnostic shaders on the production programs' device. Null, foreign and
  unprepared devices are rejected, and a successful cached load clears an earlier error.
- Reject duplicate/nonobject dump evidence, malformed artifact maps, boolean source-identity
  aliases and noncanonical invalid-content map sentinels with controlled validation errors.

Primary shader handles now belong to explicit completed/queued records. Live conversion removes
retired repair ownership and discarded background-observer queries, builds its draw request in
one place, batches diagnostic binding reads, and shares the map/boundary draw loop. Historical
dump fields retain their explicit unavailable values. The reader returns its authenticated map
snapshot rather than reading and hashing the full packed map a second time. Qualification builds
its padded native shape directly. These changes add no new image rule or temporal state.

The optimized build passes 106 native tests, including the 48-case controlled renderer matrix,
with eight optional/input skips in the broad group. The saved producer/draw paths run separately
with the same fresh test executable. All 960 Python tests complete with 13 declared skips.
Regression coverage includes oversized rows, foreign/null diagnostic devices, cached error
clearing, and restoration of nonempty retired PS bindings. The fixture now compares exact prior
binding state rather than assuming those slots start empty; its own b4 cleanup remains separate
from the production scope's restoration check.

Against validated master `ef353481`, all 24 authored-clock selected stereo outputs and literal
masks are byte-identical. The three saved HDR replays preserve all 24 colour crops and 36 pin
diagnostics. The ordinary three-clip evaluation is screen-neutral with all 576 PNG/F32 payloads
byte-identical. Every numeric stage uses the same AGENTS.md interpreter and fingerprint.

The actual converter smoke again observes correct shared bindings on its two captioned sources
and zero selected pixels on the authored empty source, without background repair. Its source
clocks come from native steady time, so it is wiring evidence rather than a matched pixel A/B.
The empty source's temporal Base and RGB differ between runs with different clocks; those
diagnostics remain recorded. An initial cross-clock pixel comparison is preserved separately.
The deterministic authored-clock qualification supplies strict pixel parity instead.

Evidence is outside Git under `E:\ApolloDev\shared-warp-cleanup-20261104`. Earlier fixture/setup
attempts remain distinct from the final checks. Existing scene deformation, row transitions,
inactive-mask preparation and occasional expensive cold fallback-shader compilation remain
limitations; this cleanup establishes no live FPS or additional physical-headset acceptance.

### P18 shared-only subtitle renderer cleanup

The November 4 live session waited from 14:51:50.844 to 14:54:23.269 for its per-stream pipeline,
although the depth estimator initialized in 309 ms. Publication was ordered after loading the
older selected-subtitle pixel shader; its new 441,264-byte cache artifact appeared at 14:54:18.
The log and cache timing identify that dependency as the observed preparation delay, without a
per-shader compile-duration measurement. The stream subsequently rendered normally.

The requested cleanup removes that renderer rather than retaining a separate deferred loader.
Its completed/queued shader copies, generator registrations, configuration fields, runtime pixel
loader and dedicated old-renderer tests are removed. Shared rendering imports the canonical Base
geometry and a current-mask authority helper, without an analytic subtitle inverse or RGB repair.
Invalid GPU selected receipts use ordinary Base. A CPU shared-preparation failure records ordinary
Base with actual b3 subtitle permission disabled and selected authority cleared, so it cannot
reactivate the old subtitle map. Qualification schema5 names only the active shared renderer;
retired evidence remains versioned and does not acquire current rendering authority.

The normal ordinary pixel-shader preprocessed token stream is unchanged and both generators
check. The full optimized build succeeds; the Python suite runs 962 tests with thirteen optional
skips. Twenty-seven distinct native tests pass, covering ordinary Base, current source/receipt
rejections, queued guards and shared rendering. Ten authored valid-output comparisons preserve
all 140 shared RGB/field/weight/row/header/map/coverage payloads byte-for-byte; their Base outputs
and unchanged authority operands also match. The production executable also passes schema5
qualification for all six ordered observations with current depth/OCR publication and producer
stages: the four caption observations activate the shared warp, and the two blank observations
retain empty masks. Its first isolated attempt failed a TensorRT DLL lookup; a fresh attempt
with verified adjacent runtime DLLs passes, and both attempts remain preserved. These functional
results do not measure live startup latency or FPS. Live XR startup evidence remains pending.
The superseded deferred-loader checkpoint and its thirteen passing mechanism tests remain
external historical evidence, not validation of this final shared-only cleanup. The
mask-recovery goal remains open separately.

### P19 integer clearance goals

The UI-plane controller searches the same cover population and
clearance budgets on a one-source-pixel grid, while retaining the public four-pixel diagnostic
index, cap, clock rules and symmetric continuation filter. This reduces unnecessary background
deformation.

The integer-goal study retains all fourteen authored geometry cases, all source-only budget
checks and twelve actual GPU route/reference positions. Every foreground-cohort mean error,
both source/layer references and left/right eyes improve or tie; both local and full-frame
background means improve or tie. Kai moves from eight to seven pixels: foreground MAE improves
from 8.252 to 6.791 RGB8 and local background from 1.308 to 1.011. The checker case moves from
twelve to eleven: foreground improves from 15.630 to 14.432 and local background from 1.469 to
0.020. The other twelve cases tie these primary means. All fourteen integer candidates pass
the original 20% cover/2% content conflict budgets and native field invariants.

These are finite mean/percentile improvements, not pointwise or XR-comfort guarantees. Kai has
927 background pixels with worse error and 4,525 with better error; its worst channel error
increases from 190 to 202, while p99 decreases from 50 to 38. The checker maximum decreases
from 209 to 7. Continuous fractional goals and preservation of an unrelated supplied phase were
rejected after foreground/background regressions.

The external evidence root is `E:\ApolloDev\subtitle-mask-background-20261104`. Integer source-only
feasibility is `integer-plane-v6/integer-validation-v6.json`; the complete measurement is
`integer-plane-quality-v6/report.json` (SHA256
`2ea5465a9728bee0305cc87ff856a2f3397cefa29c10b0bcd7ef8fccee8c44e6`).
All stages use the same recorded machine evaluation interpreter and package fingerprint. No
captured state was retagged and no CPU mask candidate or renderer supplied these measurements.

All 55 controller GPU tests pass, including
four new regressions covering integer goals, coarse-tail compatibility, cap edges, ROI union,
invalid input, clocks and empty reset.

Live headset acceptance remains pending.

### P20 ordinary Base rendering and render-binding cleanup

Ordinary live, queued and diagnostic shaders now call the canonical Base inverse directly.
The analytic subtitle shader include is removed. Shared current-glyph pinning and current-RGB
revocation retain their existing equations. Missing shared preparation or invalid selected
authority preserves ordinary Base with subtitle permission disabled; unavailable or invalid depth
retains the independent flat renderer. Render-only t8 leases, receipt backup and dump evidence are
removed. The original dense matte still used by maintained diagnostics remains.

Follow conversion now records the actual shared draw, with ordinary Base fallback. Its normal
path records one SBS draw and performs no qualification observer or diagnostic readback. Ordinary
evaluation and retained scene-cache replay use Base. Coordinate contract 128, selected-map 2,
Dump 3D 51, actual-draw evidence 2, qualification 7 and scene-cache contract 5 explicitly describe
these roles; the six-payload tuple format remains 1. Historical captures are not retagged.

On the same nine source sequences and clocks, all 54 actual selected RGBA images and selected
masks are byte-identical to the preceding e3495a54 build. The six-frame follow conversion also
matches the new selected qualification output exactly. The evidence is in
`E:\ApolloDev\cln1\runs\cleanup-main54-v1\comparison.json` (SHA256
`013a2427b7a13bfdcb3908a04a3f8f0d29bd7ba524614925b98efb15fa68997b`) and
`follow-latin-v1/comparison.json` (SHA256
`226769c105ec6889933c0ab970c2480b331a5f0cd6bcf97cffaf1c17eb5cf297`).
The evaluator suite passes 975 tests with 13 platform skips. Native gates pass 305 mandatory
tests; six external replay diagnostics have no supplied inputs and are explicitly skipped.
These are functional and finite output-parity checks, not live FPS or headset-comfort evidence.

The first live package omitted `video-dom-info.exe`, leaving browser subtitle admission disabled
despite resident OCR. The corrected immutable package and subsequent publication checks require
that helper; bounded helper failures are logged independently of context transitions. A separate
physical-display fix preserves the last verified recovery target when a retained virtual display
returns beside only part of the physical desktop. Three simulated CCD regressions cover partial
wake, delayed availability and deliberate user disable. The driver's virtual-display reactivation
and physical settling still require live validation.

### P21 static-review fix series

This series works through the numbered static-review list on `claude/intelligent-feynman-d45yhh`.
It runs in a dedicated worktree with a fresh RelWithDebInfo build. Model and TensorRT engine
assets were copied from the main development build; the exact AGENTS.md interpreter is used for
every evaluator stage. Earlier dumps are replay inputs only, since the producer closure changed.

The OCR fix merged without compilation in PR #1 builds and passes on the local RTX 5080:
`HostSbsOcrGpuTest.AssistedEntryKeepsItsEpisodeForTheFirstTrustedObservation`.

The series predates the removal of Current Outline. Where an item below cites the 24-frame
selected qualification or `q-control`, that was the qualifier of the route in use at the time;
it no longer exists. Items of the series that changed only Current Outline code are not listed.

Fix 1, diagnostic-only conditioning passes, is kept. Live streaming no longer runs the field-pin
projection, the two native distance passes, the proposal writer, the compatibility vertical copy
or the subtitle horizontal guard unless the completing observation is snapshotted for Dump 3D.
Other observations publish exact Base through the distinct Final texture. The four renderer
presence gates test Base. The harness, conversion worker and native tests keep the always-on
default, so their artifacts are unchanged. The current policy is described in
[Host SBS](host-sbs.md#native-subtitle-contours-and-ui-plane).

- A fresh `control` covers all 19 core clips and 480 frames on the unmodified binary. It reports
  one existing named issue, `subtitle_cjk_highres_transitions.subtitle_glyph_soft_plane_inlier_pct`
  at 62.678, which every later run reproduces unchanged.
- `fix1-treatment`, bound with `--report-control`, is `screen_neutral`. All 3,973 SBS, depth,
  raw, model-input, warp-map, structure, mask, cut-state, contract and receipt files are
  byte-identical; only timing JSON and the aggregate result file differ.
- `fix1-livepolicy` repeats the run with `--subtitle-conditioning-live-policy`, which really
  skips the passes. The same 3,973 files are byte-identical to the control.
- `TensorRtOcrAvailabilityGpuTest` now also checks that the live policy publishes Final equal to
  Base, keeps the result authenticated, clears the diagnostics-current flag, and sets it again
  when the completion is snapshotted for a dump. It passes.

The ordinary evaluator renders Base only, so this is parity of the
shared depth, Base and matte path rather than of the selected shared-warp draw. The skipped
passes write only the field mask, distance, Gaussian tuple and Final, none of which that draw
binds. No shader source changed, so no closure pin or contract identity moved. No timing claim
is made, and a live Dump 3D capture with the new policy has not been taken.

Later fixes in this series use a reduced gate at the owner's request: the subtitle GPU suites,
the four subtitle clips of the ordinary evaluator compared byte-for-byte with the same
`control`, and the 24-frame actual selected-compositor qualification on
`subtitle_cjk_highres_transitions`, compared byte-for-byte with a `q-control` captured from the
tree before fix 2. No full native suite and no full 19-clip paired report was run for them.
`q-control` reproduces the recorded mask counts: 153,796 true-positive, 23,625 false-positive
and zero false-negative pixels.

Fix 2, RGB history and probability-change cue, is kept as a removal. The review's premise holds
for genuine observations: every eligible one has a positive count, so the two-pixel neighbour
correspondence and the per-core probability-change reductions could not change support. They
did change which counts exceed one, and a count above one is what lets a duplicate delivery keep
a pixel whose RGB is close but not bit-identical. That duplicate rule is unchanged; only pixels
that had reached a higher count through a neighbour, and then met a non-identical duplicate,
can differ. Word 21 stays in the layout as zero, and the contract keeps the two retired
constants. Seven contour tests and seven locator tests that specified the retired behaviour
are removed and three are updated; the affected suites pass 280 tests. All 24 selected masks
and 48 stereo images are byte-identical to `q-control`, as are the
792 artifacts of the four subtitle clips.

Fix 6, detector dead code, is kept in part. The unused `score` parameter and two duplicated
helpers are removed with byte-identical output on the four subtitle clips and 66 passing OCR
GPU tests. The second reference-centre test is not dead, because ribbons skip the first one.
Raw slots 8--15 are record layout rather than code and stay. The publish-time evidence check
cannot fail but supplies the score, so it stays with a comment.

Fix 7, centring union, is kept. When the full group union fails the two-percent midpoint rule,
the union of only its independently horizontal members is tested, and only those members are
published. `UnknownPieceCannotPushACenteredHorizontalLineOffCenter` fails on the previous
shader, where the blob removes the caption, and passes now. The existing negatives pass
unchanged, including `SeparatedStrongFragmentsRequireTheirGroupToBeCentered`. The four subtitle
clips and the 24-frame selected qualification are byte-identical to their controls, so the
authored clips show no regression and no gain; the road-marking false positive is unchanged.
The remaining risk is a scene line whose horizontal part is centred on its own while a weaker
part is not; that part alone was already admissible before whenever the weaker part was absent.

Fix 8, short-line slope gate, is kept. A line whose 10th and 90th percentile envelopes each
move by at most one detector row across its populated bins is independently horizontal. The
tolerance is tested on the actual envelope range rather than the fitted rise, because a
least-squares fit of a one-row step reports about 1.33 rows. `ShortLineWithOneRowEnvelopeStepIsHorizontal`
fails on the previous shader for its one-row case and passes now; its two-row twin stays
rejected. The 68-test OCR suite passes, and the four subtitle clips and the selected
qualification are byte-identical to their controls. The authored clips contain no such short
line, so they show no regression and no gain. A short scene mark that is centred within two
percent and one row flat can now be independent evidence; the centring and span rules are the
remaining guard.

Fix 9, orientation-fit foreign pixels, is confirmed and not changed. The second classification
runs after ordinary merging, so `ActivePixelsInRow` does count every active pixel in the merged
bounding rectangle, including pieces too small to queue and pieces whose class kept them out of
the merge. Those pixels are confined to the union's own rows, and a least-squares fit gives
central bins little leverage. Several authored attempts to flip a merged line's class with
foreign pixels stayed within the two-degree bound, so there is no failing case to fix against.
Restricting the count to members needs per-member rectangles in a single-lane, already
quadratic path (HSR-15). It stays open as a low-priority accuracy item.

Fix 10, prior learning, is kept: ribbons no longer teach. A ribbon's core is its whole broad
bottom run, it is exempt from the centre-span rule, and its height is not a caption height, yet
an independently horizontal ribbon set the position votes and the single trusted height.
`BottomRibbonTopologyIsPublishedWithPairedCappedCover` now also requires an untouched prior
after a published ribbon; that assertion fails on the previous shader. The ribbon is still
published with its paired cover. The OCR suite passes 68 tests, and the four subtitle clips and
the selected qualification are byte-identical to their controls.

Fix 15, plane controller, is evaluated and reverted. The time step is capped at 50 ms, so at a
100 ms observation cadence the filter and the 30 pixel-per-second slew act at half their
nominal rate. Raising the cap to 150 ms was tried on the selected qualification replayed at
`--fps 10/1` (`q10-control` and `q10-fix15`). All 24 selected masks and stage buffers are
identical and 14 stereo frames change. The applied plane does converge faster, but the
per-observation goal on this clip moves by about one source pixel in alternating directions,
and the faster controller follows it: mean plane motion between consecutive captioned
observations rises from about 0.88 to 1.40 source pixels, with single steps above two pixels
where the control stays near one. That is more visible caption depth motion for no demonstrated
benefit, so the 50 ms cap stays; in practice it is part of the smoothing. A cap change should
follow a steadier goal, not precede it.

The second half of the item is not attempted. One clearance histogram spans the union of
current covers, so a false cover over near foreground can raise the shared plane. Separating
it needs a per-cover goal and a rule for combining them inside the fixed 40-word UIP state,
which is a controller redesign rather than a bounded fix, and the authored clips have no
false-cover case to measure it against.

Items 12 to 14 and 16 are not attempted in this series:

- Item 12, single empty detection (HSR-05). An exit hold has to be coordinated across cover,
  plane and mask, and needs appearance and disappearance evidence that the authored clips do
  not supply. HSR-05's earlier disposition stands.
- Item 13, vertical row behaviour. A cover-bounded vertical close of row authority needs the
  clean-background stress cases from the P14 comparison to judge scene deformation; those are
  external fixtures bound to an older contract and were not rebuilt here.
- Item 14, revocation flicker. Hysteresis or per-component revocation adds temporal state to a
  per-draw rule, and there is no captured sequence showing the toggle to measure against.
- Item 16, completion-stage ablation, depends on item 13.

Closing state of the series: both generators pass `--check` and evaluator discovery passes 979
tests with 13 skips.

Six stale source-text tests searched for strings that `master` did not contain either:
`DirectxShaderSourceTest.SubtitleResolveBindsExactOcrRecordAndConditionParams`,
two `OfflineSbsWorker` cases, two `ParallaxV2ContractTest` cases and
`SbsDebugDumpAsyncTest.Schema49PackagesCompleteContourTupleAndNoScalarPreviews`. They were
deleted at the owner's direction. The deletion removed the other, still-valid source-text
assertions those tests carried, including the 287-line production contract constant check.

`V2_SUBTITLE_CONTOUR_HISTORY_JITTER_RADIUS_SOURCE_PIXELS` is no longer read by any shader
after fix 2. It is left in the contract deliberately: removing a contract constant moves the
closure pin of every shader group that includes the contract, for no behavioural gain.

Fix 1's live dump path is confirmed on the host. The series build ran as the streaming host
and a client-requested Dump 3D during subtitled video playback was written complete
(`dump_20261114_093139_372_18e7f9ad46a62f28_0`, frame 4035, 54 files, no rejection or warning
in the host log). The OCR record carries the same frame, the conditioned Final field differs
from Base on 37,471 field samples, and that count equals the non-zero Gaussian weight samples,
so the diagnostic conditioner ran for the dumped observation under the live policy. The owner
watched the same session in the headset and reported that subtitles looked fine.
