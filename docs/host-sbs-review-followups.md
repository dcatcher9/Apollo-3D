# Host SBS subtitle review follow-ups

This tracks the static review supplied in `CODEX_HANDOFF.md` on
`claude/exciting-meitner-iarw5l`, commit `166d1018f7b64bba004d59999abfaf3476b990db`.
The original confidence labels are retained: V = independently verified in that review,
R = reviewer-confirmed, U = unverified, X = rejected. They are not claims of a local GPU
reproduction. This is a work tracker, not an alternative algorithm contract; policy remains in
[Host SBS](host-sbs.md) and [scene cuts](host-sbs-scene-cuts.md).

## Priority repairs

| ID | Original | Disposition and remaining evidence |
|---|---|---|
| HSR-01 | U | Confirmed and repaired missing-OCR failure: allocate resolve identity constants independently of the detector and require working probability only for optional OCR. The real missing-asset GPU test passes two depth observations with authenticated abstention, exact Base output and zero subtitle/current-outline masks. Oversized full-source captures are an intentional separately enforced source boundary, not a newly established OCR regression. |
| HSR-02 | R | Repaired geometry-sensitive semantic identity and modal identity withdrawal. A freshly revalidated matching root/process can retain an already admitted native-player or browser-video context for suppression; native registration or the matching fresh browser heartbeat remains mandatory, and no new context is admitted during movement. Positioned ROI/render authority stays separate. Protected packed fullscreen pixels now bind current foreground/region/browser/modal authority, preventing retained semantic identity from replaying an old treated image; full-source Base-only reuse stays independent. Modal draws still use fresh Base, and real ROI/full-source or resized-domain transitions retain their required resets. Galaxy XR acceptance remains pending. |
| HSR-03 | R | Confirmed and repaired native-player reference mismatch, explicitly encoded in the previous policy. Foreground-client analysis ROIs now supply the caption reference, with matching dump validation and generated identity. The production classifier regression retains the same caption at three translated native-window positions and rejects it under the former capture reference. Full-source fallback has no player rectangle and remains screen-relative. |
| HSR-04 | confirmed | Repaired generator ownership of all five contract HLSL copies and three OCR assertion copies, plus automatic check/refresh of all twelve selected-route runtime closure pins. Refresh now writes generated UTF-8 outputs only when their exact bytes change, preventing no-op regeneration from triggering broad rebuilds. Eight additional no-op, changed/missing-output and read-only-check cases pass; generated identities and check semantics are preserved. |

## Remaining bugs and stability work

| ID | Original | Status and next action |
|---|---|---|
| HSR-05 | V | One genuine empty OCR observation immediately returns to Base. This is documented current policy. A debounce would require a coordinated cover/controller/matte policy change and appearance/disappearance evidence; holding only the controller is insufficient. |
| HSR-06 | U | Confirmed and repaired history-authority leak with narrow owner invalidation at the ordered subtitle-context boundary. Real GPU coverage passes immediate revocation without a subtitle dispatch, exact replay, newer-frame/equal-clock calls, fresh reseeding and stale-reset high-water preservation. Strong-cache/AA plans and duplicate matching are revoked while allocations, pixels and owner lifetime survive. This is a history-authority repair, not a measured glyph-quality improvement. |
| HSR-07 | R | Three authored production enclosure GPU cases pass: ordinary/restore-bit normalization, component-local support recovery, and strong glyphs with one- and five-pixel open outline gaps beside a closed weak peer. Those cases do not establish the broad whole-glyph-loss claim. Historical restored components and new seedless recovery offers follow distinct code paths; counter differences alone do not prove an adoption-policy bug. Euclidean/inner-only cache offers versus Chebyshev seed support, and empty-unit cache revocation, remain open for captured mask reproduction. |
| HSR-08 | R | Left-aligned multiline captions are intentionally outside the confirmed supported scope; their rejection is not a bug or fix priority. The per-line centering rule stays unchanged. Actual production OCR-box coverage characterizes this boundary: rejected lines do not teach the prior, and independently centered rows retain all three pairs. EMPTY still clears current covers. Ribbon-prior effects remain open; empty-observation policy is tracked separately in HSR-05. |
| HSR-09 | R | Repaired the queued-renderer dependency and encode-thread preparation: completed live configuration omits queued probe assets and a tracked background future prepares programs for one-time adoption. Current-outline compute bytecode authenticates every mandatory closure before resolution, then uses at most four scoped CPU workers, all joined before sequential device shader creation. Transactional replacement and independent optional stability failures are preserved. Shared bytecode flights now settle attached callers with the original exception and evict only the failed entry, permitting retry without an old failure erasing a replacement; setup, observer and persistence failures cannot poison a successful flight. Three new deterministic flight tests and twelve real compile/cache cases pass. Cold completed-PS compilation remains; earlier diagnostic loads are not production startup or performance comparisons. |

## Performance investigations

All timing claims below are review estimates. Measure serially on an otherwise idle GPU before
calling them production costs or improvements.

| ID | Original | Status and next action |
|---|---|---|
| HSR-10 | R | Many serialized per-slot/per-mode indirect dispatches run even without captions; color passes span the band. Profile actual issued work, then consider a GPU-compacted active list and bounded regions. |
| HSR-11 | V | Repeated lane-zero authorization across contour/color/selector/band passes. Investigate a shared immutable verdict without weakening frame/owner validation. |
| HSR-12 | R | Rejected the unused-P010-MRT claim: authenticated HDR/P010 Base fallback can supply luma and skip later conversion. Exact repeated images already reuse packed output without issuing selected/background draws. Remaining draws use changed mutable RGB or a new publication; pointer equality and the old draw lease cannot prove immutable background inputs, C24 or b3 contents. No unsafe cache was added. Profile actual issued draws before introducing stronger material/publication generations. |
| HSR-13 | R | Confirmed and removed a duplicate Base inverse on six missing/stale/invalid-background fallback exits only; the successful BG path and ordinary root renderer already solved once. Completed and queued helpers now forward the original unsaturated Base UV, preserving authorization, both footprints, analytic UI, RGB and alpha. Six actual/source checks pass, including 48 combinations compared with a frozen authenticated prior PS. All 24 actual stereo PNGs and selected-mask hashes also match exactly. Uniform authorization work remains a profiling topic; no measured GPU-speed claim is made. |
| HSR-14 | V | Repaired native-band allocation and redundant feature copying. Hard disable uses one-pixel native placeholders; suppression preserves existing backing/pixels. CPU-completed force-infer roots publish SCT5 through an authenticated identity-only entry and exchange complete feature owners. Receipt-gated/queued roots retain the old conditional copy, so zero-group reuse never swaps owners. Real GPU copy-versus-swap parity across observation boundaries and poisoned-scratch reuse tests pass; count and duplicate semantics remain unchanged. Generated producer identity is schema 127. |
| HSR-15 | R | Single-thread calibration and quadratic lane-zero grouping on dense text. Add representative dense-text measurements before replacing either algorithm. |

## Mask accuracy investigations

| ID | Original | Status and next action |
|---|---|---|
| HSR-16 | R | The one-pixel cover-boundary GPU twin rejects the black attachment while preserving all 240 white BODY core pixels, so that case does not establish whole-glyph loss. Authenticated controls isolated all 92,092 missing opaque dark-outline pixels to uniform nonliteral `(7,8,11)` UNKNOWN/status5, with all 61,704 white BODY pixels retained. A narrowly qualified final-CBC private version4 now admits only current own-unit opaque Raw4CC support after full BODY enclosure/disjointness and boundary checks; it grants no AA, collar, Inside or history authority. The actual 24-frame compositor recovers all 92,092 pixels with no glyph withdrawal or added background protection. A broader candidate that added 27,920 scene pixels was rejected. The existing 23,625 false-positive pixels remain five current road markings through BODY fallback. Subsequent [known-model C2/C3/C4 follow-ups](#current-known-black-candidate-follow-up) have finite HDR/precision evidence; C4 and C5 finite native and ordinary regression gates pass; live and complete-alpha limits remain open. Broader palette/bin/diagonal/slope/HDR and live recall claims remain open. |

## Maintenance and coverage

| ID | Original | Status and next action |
|---|---|---|
| HSR-17 | confirmed | Removed the unused outline ROI clone; production uses canonical ROI, while enclosure-local compatibility stays in authenticated fixtures. Enclosure owns the physical-black helper and completed rendering owns the common selected-render helper; required outline/queued mirrors are byte-preserved and checked/materialized before closure refresh. Drift/missing-mirror tests pass. This preserves confined include roots and distinct caller macros; ordinary legacy and selected renderer roles remain separate. |
| HSR-18 | R | Corrected two active C++ ownership headers that incorrectly claimed DRAFT/non-runtime status. Probe-only queued, pending-color and body-witness paths still share application translation units; broader separation/naming remains maintenance work and does not justify removing their explicit diagnostic callers. |
| HSR-19 | R | Centralized selected-receipt ABI1 C++ storage and its C24 byte-copy boundary: 64 words/256 bytes, words 4--27 copied as the exact 96-byte constants. Both selector owners, transport, BG descriptor checks and qualification use the owned types/constants; five actual WARP shape/range/role/lifetime cases pass. Consumers still independently authenticate freshness and rendering authority. Wider HLSL/Python word maps and SCT/SCP validation remain separate work. |
| HSR-20 | R | Corrected the center-tolerance prose in HSR-03 and rewrote the selector overview for the actual completed current-outline PS, selected receipt/legacy fallback, optional GPU repair background and mixed-coordinate fractional RGB. The original legacy recipe is explicitly scoped to fallback/evaluator behavior. Four actual completed-PS draw cases pass; ordinary evaluator parity remains HSR-21. |
| HSR-21 | U | Added an explicit [actual selected compositor qualification](../tools/sbsbench/README.md#actual-selected-compositor-qualification): real current-outline estimator, completed PS, selected receipt, GPU background, immutable draw-role lease and observed pre-Draw bindings. Separate stereo sidecars bind source snapshots, clocks, current completion receipts, GPU evidence words and executable/source pins; the independent reader rejects tampering and stale ownership. The 24-frame high-resolution transition clip passes with 24 current selected receipts, 21 valid-background and three authenticated empty-background dispositions. Ordinary `run_eval` retains its legacy compositor/metric artifacts, so broader evaluator/live metric parity remains open; file-fed qualification does not establish live capture scheduling or headset quality. |
| HSR-22 | R | Actual completed-PS WARP coverage now includes current/legacy choice, signed/fractional UI, stale-owner/RGB revocation, ROI/SDR-linear-HDR, real BG fallback and authenticated old/new pixel parity. The opt-in qualifier exports literal native selected masks plus exact completed CBC/Raw/F2 stage buffers. Independent diagnostics bind source/GT/metadata/timeline/runtime and report precision, recall, IoU, empty false protection and mask toggles on unchanged RGB+GT pixels, with null missing denominators. Schema1/2 remain supported. The authored 24-frame clip is qualified; real-video distributions, final-RGB retention and live capture remain separate evidence. |

## Rejected claims retained for traceability

| ID | Original | Disposition |
|---|---|---|
| HSR-23 | X | First cue after an empty observation publishing target zero was rejected as unreachable because geometry readiness enforces the observation clock. Do not treat it as a confirmed defect. |
| HSR-24 | X | Palette hue-versus-RGB and fixed-radius complaints were rejected as descriptions of the live mask: live draws CurrentOutlineM. Any new claim must identify the actual active renderer. |

## Validation record

Local Windows qualification on November 2, 2026:

- The default RelWithDebInfo build passes, as do generated contract/closure checks.
- The real missing-OCR-asset GPU regression passes two joined observations with current-outline
  enabled, authenticated abstention, exact Base output and zero subtitle masks.
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
- Ten real GPU/device/runtime checks pass: the existing TensorRT context/publication lifecycle,
  four actual completed-PS draws, current-outline owner invalidation, and four renderer-role
  authentication/ownership cases. Three additional authored enclosure GPU cases pass.
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
- Full evaluator discovery passes 906 tests with 13 skips. The selected-compositor reader adds
  ten passing tests for source/receipt/clock/closure tampering, legitimate empty/fallback
  dispositions, canonical geometry and the exact FP32 plane boundary.
- The actual completed-compositor transition capture and its independently revalidated 24-frame
  manifest pass. These are functional provenance/renderer checks; compilation and capture times
  are not production performance evidence.
- `handoff-pass3-treatment` against authenticated `handoff-pass2-treatment` passes with a
  `screen_neutral` report. All aggregate, per-frame and worst-frame metrics are identical across
  72 frames, with no issues on either side. The exact numeric-runtime fingerprint matches before
  treatment/report and after verification. This continuation changes no active shader bytes,
  geometry or scoring policy. Two attempted replacement controls were rejected for provenance
  drift/setup before qualification and were discarded; neither supplies report evidence.

The headless-only continuation also repairs redundant feature copies, bounded bytecode preparation
and the repeated Base inverse on background fallback:

- The optimized default and final native builds pass. Across the focused native runs, 126 unique
  tests pass and two external saved-replay diagnostics skip. This includes actual GPU copy/swap
  parity, poisoned zero-group reuse, shared C24 transport and descriptor rejection, multiline and
  boundary-component checks, and the completed PS compared with its authenticated prior source
  across 48 fallback cases.
- Final evaluator discovery passes 930 tests with 13 skips. The affected contract, dump, closure,
  reader and mask diagnostics pass all 236 cases; generated checks and diff checks pass.
- `handoff-pass4-control` and the paired treatments pass with `screen_neutral`; all 72 ordinary
  aggregate/per-frame/worst-frame metrics are identical, with no issues. Every numeric-runtime
  fingerprint matches. Actual selected-compositor controls additionally authenticate all 24
  source snapshots, PNGs and literal masks; the final renderer produces identical hashes.
- Schema3 records current CBC/Raw/F2 buffers in addition to stereo and native-mask artifacts.
  The authored clip retains all 61,704 white BODY pixels but omits 92,092 opaque dark-outline
  pixels under the documented uniform-nonliteral UNKNOWN rule. Its 23,625 empty-frame false
  positives are current road markings, rather than stale subtitle ownership. These are bounded
  algorithm limitations, not an assertion of final RGB quality or a release gate.
- A separate frozen local Review127 snapshot was prepared for requested live testing. The number
  denotes contract schema 127, while Build154 is the prior build-series label. Its actual 24-frame
  packaged compositor qualification and launcher preflight pass; physical headset acceptance is
  still separate.

A further headless investigation captures fresh `handoff-pass5-control` and actual selected
compositor controls before considering a detection change:

- The ordinary 72-frame control passes. The actual 24-frame control reproduces exactly the
  prior mask counts: 61,704 true-positive, 23,625 false-positive and 92,092 false-negative pixels.
  All selected receipts are current. The validated Python/package fingerprint is unchanged.
- Working OCR diagnostics capture all 96 observations across the four authored subtitle clips.
  Hashes, current inference/optional-OCR transactions, probability/cell equivalence, source crops,
  clocks and completed OCR records are checked. The high-resolution cores also match the actual
  selected-compositor CBC records. This is file-fed diagnostic evidence, not live scheduling or
  performance qualification.
- The road-marking false positive already passes the current midpoint and center-span tests.
  Its native core is about 186 pixels tall, supplying a 557-pixel seed tangent limit for the
  225-by-21-pixel road components. The learned position prior cannot veto its independent
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
- Uniform nonliteral dark-outline recovery still requires a separately qualified physical-black
  policy extension; existing tests intentionally preserve UNKNOWN and white BODY. A genuine empty
  OCR observation still clears treatment under the current contract. Neither a controller-only
  hold nor replaying old masks supplies current caption authority.

The reproducible local envelope analysis and authenticated packet bindings are retained in
`C:\Users\DCatc\.codex\tmp\apollo-subtitle-review-pass5-20261102\ocr-envelope-diagnosis.json`.
Review127 remains frozen for the user's live test; these investigations do not alter that package.

A subsequent headless continuation qualifies the opaque-outline extension and repairs two
infrastructure failures:

- The optimized default build passes. Full evaluator discovery runs 938 tests with 13 skips.
  Generated contract/closure checks and whitespace checks pass. Refresh preserves the modification
  time of every generated output whose bytes are unchanged; a second full refresh changes none.
- Shared bytecode flights settle the original failure for attached callers, evict only that exact
  entry and permit a deduplicated retry. Three deterministic failure/clear/retry cases and twelve
  real compile/cache cases pass. Optional persistence failure cannot revoke shared success.
- The final current native suite passes 64 checks, with one external saved-28 replay skipped.
  Coverage includes ten uniform-nonliteral cases, malformed publisher phases, open and boundary
  components, BODY/black overlap, mixed known-owner overlap, opaque hole/exterior preservation,
  optional cache/AA lifecycle, missing OCR and the actual completed compositor. The standalone
  BODY/black-overlap negative also passes against frozen pre-extension shaders: all twelve input,
  full-mask, inherited-certificate and stage properties match the current run exactly.
- The final actual selected-compositor pair independently authenticates and remeasures all 24
  frames. Private version 4 recovers all 92,092 authored opaque-outline pixels, retains all 61,704
  original white BODY pixels and adds no background protection. Native aggregate counts are
  153,796 true-positive, 23,625 false-positive and zero false-negative pixels. All 24 receipts are
  current; 21 background dispositions are valid and three are authenticated empty. Sixteen stereo
  frames change. These mask results do not establish final RGB fidelity or live typography/HDR
  coverage. The road-marking false positive remains unchanged.
- The broader first candidate is rejected: its extra 27,920 nonblack scene pixels are preserved
  in a separately authenticated negative experiment. The accepted tier grants only original LOW
  BODY and current own-unit opaque Raw4CC support; it supplies no AA, nonblack collar/closure,
  Inside, BAND/N8 or cache/history authority. Existing normal-owner overlap and invalid Raw fallback
  remain intact. The compiler-rejected barrier shortcut produced no qualified frames and was
  removed before the final run.
- Fresh `handoff-pass6-control` and `handoff-pass6-treatment` produce a `screen_neutral` report.
  All 72 ordinary aggregate, per-frame, worst-frame and label values match exactly, with no issues
  on either side. The numeric-runtime fingerprint matches throughout, and the metric contract
  remains `fe465345dfcacba2`. This ordinary evaluator still uses the legacy compositor.

The final selected comparison, native XML, frozen-reference parity, numeric-runtime fingerprints
and ordinary report bindings are retained in
`E:\ApolloDev\host-subtitle-review-pass6-20261102`. Review127 remains unchanged; the newer live
launcher uses the current optimized development build with executable and runtime-shader hashes
that must pass preflight before launch. No host or headset was operated by the agent.

Use the local runtime and build requirements in
[AGENTS.md](../AGENTS.md), the [repository workflow](../CLAUDE.md), and the
[evaluation procedure](../tools/sbsbench/README.md#required-evaluation-loop).
Ordinary paired evaluation covers shared depth/Base and the legacy harness renderer; HSR-21
and HSR-22 limit what that evidence establishes about the production selected composite.
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
stalls. The separate C1 opaque-offer experiment remains unaccepted: its fixed authored-28 cohort
adds six glyph pixels and fourteen background pixels with no withdrawals. Its evidence remains
preserved; the clock result supplies no caption-mask precision or subsequent candidate acceptance.

## Current known-black candidate follow-up

C2 is a finite improvement candidate under the
[canonical geometry policy](host-sbs.md#subtitle-classification-before-protection), separate from
the previously qualified version-4 extension. The shader source SHA-256 is
`43b33d636c8f6151ae66d9b4066486b12e7a656a8346b5f02efbd0de7b998930`.
Its evidence remains scoped as follows:

- The three authenticated cold HDR producer replays add respectively `0`, `25` and `3,048`
  selected pixels against C0, with no withdrawals. BODY, CBC, Raw and model metadata remain
  unchanged. Actual completed pixel-shader replay passes and preserves the original five output
  variants.
- All three selected nonblack fringe witnesses now have native M=1 and match the independent UI
  reference in both GPU-rendered eyes. This is point evidence, not whole-caption alpha recovery:
  visible thin AA fringe remains, and the first dump's selected mask is unchanged. Per-pixel
  private AA tags were not exported.
- Both exact-shape authored subcohorts cover all original 28 sources and fixed independent truth,
  with zero mask additions or withdrawals on glyph and background pixels. The hardened report
  `precision28-comparison-candidate2.json` has SHA-256
  `76320befd5a36fe48304dbdab3b21956814a71f82aafa9d4e109e0d9884141e8`.
  C1 remains rejected as recorded above; this comparison does not waive its new background pixels.
- The ordinary 72-frame gate is complete: all 24 observations in each of `scene_cut`, `flat_page`
  and `subtitle_cjk_highres_transitions` are included. All 504 corresponding SBS, depth, raw,
  warp-map, structure, warp-mask and model-input artifacts are byte-exact. Metrics, labels,
  source/model/engine/configuration/timeline identities and numeric runtime match, yielding
  `screen_neutral`; executable and runtime-shader identities deliberately differ. Independent
  review rechecks all 1,744 receipt-bound files and every actual artifact pair. Fresh full
  evaluation and scoring pass; the independent audit authenticates that completed report without
  scoring it again.
  `ordinary-candidate2-parity.json` has SHA-256
  `f484690460622f36cf3974f103dbe7a02c7c8d25dbb41698bbd104fb50bfd9e4`.
  The incomplete, unscoped `handoff-pass7-treatment` attempt is preserved and excluded; the exact
  treatment is `handoff-pass7-treatment-core`. This ordinary gate still uses the legacy matte
  metric renderer and does not qualify live selected-compositor behavior or speed.

These files and `candidate2-residual-fringe-diagnosis.json` remain under
`E:\ApolloDev\host-subtitle-review-pass7-20261102`. The final native suite completes 65 tests with
63 passes and two explicit optional saved-replay skips. Both new authored regressions pass:
`OutwardRawLobeUsesIndependentBackgroundWitness` and `ClosedBlackCounterDoesNotCreateOpaqueOffer`.
Authentic pre-fix shaders fail the positive lobe/AA test at exactly four expected protection
assertions. Rejected C1 shaders fail the closed-counter test at its final center and all 384
private opaque-offer checks. These retained negative controls establish that the regressions
exercise the changed behavior. The final test executable SHA-256 is
`ecd4c1af09d83a223e09bd6d35e338bd81205b7fb61d4426d38e80981e2068fc`;
the main executable remains
`9c6eed72005f32b1c1eaef9bf21caf357f8fac833b5a98d1f65bf59aaed85aae`.
Python discovery passes 938 tests with 13 skips, and the fresh 72-frame ordinary screening above
passes. Final sealing/publication remains separate from live Galaxy XR acceptance; these headless
checks do not establish that all three captured caption issues are fixed or improve FPS.


C3 is a subsequent component-only opaque extension under that same canonical policy. Its band
source SHA-256 is `d052493a9dc7d72ca4c7f7ce74d77ea5cf0673fc100f2f86231e8914f15e6bb1`.
The three authenticated cold HDR producer replays add respectively 0, 0 and 14 protected pixels
against C2, with no withdrawals or BODY/CBC/Raw changes. All 14 new pixels match the independent
UI reference in both actual GPU-rendered eyes with regenerated candidate background. The authored
thick-junction/counter/scene regression and the two prior lobe/counter regressions pass. The
24-frame selected-compositor counts match C2.

All original 28 authored cases remain included. The strict C3 comparison adds two independently
labeled glyph pixels, one each in `black6-00004` and `black6-00005`, with zero added background
protection and zero glyph or background withdrawals. `precision28-comparison-candidate3.json`
has SHA-256 `f6f74b6616037ebf5c4e04cad8cc8e20b44f0ef4395877853b662f3050dfc30a` under
`E:\ApolloDev\host-subtitle-review-pass8-20261102`. Its explicit configuration supplement records
the byte-identical C2 benchmark reference without modifying or relabeling the original source
freeze; the failed setup log and original comparer are retained. This is finite C3 headless
qualification. The C2 72-frame and 65-test checkpoints above retain their original identities;
they are not fresh C3 runs. The first dump remains unchanged and visible thin AA fringe remains.
Actual saved-source graph audits find no independent current first-graph seeded-interior
authority at any of the 418 nonblack residual-neighbour candidates. They remain candidates,
not confirmed text-alpha omissions. No complete caption-alpha, live Galaxy XR or FPS improvement
is established.


The separate C4 one-pixel ordinary-AA experiment has passed the fixed authored negative-precision
gate. Its band source SHA-256 is
`37760ff5e23034a317e58f9ca317f025cf38d3085ef12ae11b962b0160973720`.
`precision28-comparison-candidate4.json` has SHA-256
`a238a57c839e770d2c3e325adb4ebc6c68c0c684ab3199fe3d352246c34a36de` in the pass8 directory.
All 28 cases are included against C2: two added glyph pixels, zero added background protection
and zero glyph or background withdrawals. These two glyph additions were already present in C3;
they do not establish additional C4 gap recall. All 28 actual C4 selected-mask payloads are
byte-exact against C3.

The three actual saved-source producer and GPU pixel-shader replays pass with regenerated
candidate background. Their selected-mask counts are 163,583, 73,749 and 19,227, respectively;
relative to C3 the changes are +0, +1 and +285 protected pixels. These counts are geometric
protection changes, not independently labeled glyph counts. The 24-frame selected-compositor
counts match C2. The preserved C4 fallback passes its final 67-test native suite (65 passes, two explicit optional skips) and fresh ordinary 72-frame/504-pair screening with `screen_neutral`.

Actual C3 graph/LOW8 diagnostics narrow the fixed 277 opposite-black candidates to 274 eligible
geometric candidates. Every one remains outside independent current seeded-interior authority;
all 418 broader residual candidates lack that authority too. No semantic alpha is inferred, and
C3's accepted opaque evidence is not itself a C4 AA-precision result. The first dump remains
unchanged and visible thin AA residual remains. Live XR, FPS and complete-alpha acceptance remain
open. Original C2, C3 and failed-attempt artifacts retain their separate identities.


C5 completes the finite headless gates: native 67 tests (65 passes, two explicit optional skips),
Python 938 tests with 13 skips, selected-compositor 24 frames, strict authored precision 28 cases,
three actual producer/completed-pixel-shader replays, fresh ordinary 72 frames/504 byte-exact
artifact pairs with `screen_neutral`, and the actual-converter packed-continuity regression.
The band source SHA-256 is `8c1bb992d0c78968af42db02855692274bb02ffcce742d30d3e79deb92b9c8e0`.
The strict28 receipt is `b507695b08814dbd72a398a4d4d22a174ff69b423917cefe6d464cdbf11536a2`;
it preserves all 28 C4 mask payloads byte-exact, retaining the two C3 glyph gains against C2 with
zero new background protection or withdrawals. Independent review rehashes all 1,013 bindings
and recounts every literal mask against original authored alpha. The fresh ordinary receipt is
`17532a96d3eac3d490c35267e256b537eb6876d495bd15ecd16ae01ae66e8e62`; its renderer keeps the
ordinary legacy-matte scope. Actual saved-source M gains against C4 are 1, 0 and 284, with no
withdrawals and only new ordinary bit-512 evidence. The first point is UI-reference exact in both
eyes; 277/284 newest-case points are exact per eye, while seven retain small linear-HDR RGB
differences (maxima about 0.000374 left/0.000755 right). Six prior side-fringe candidates remain
outside the certificate; neither those cohorts nor the original 418 points are semantic alpha
truth. C2/C3/C4 evidence keeps its historical identity. Cold live history, full-alpha coverage,
live Galaxy XR and FPS acceptance remain open; the long first cold compile is not a latency gain.

The subsequent rule review narrows the meaning of those passes. Across all 28 authenticated
C4/C5 F2 buffers, all eight units' candidate/accepted/ordinary-AA counters and the global unique
SUPPORT_AA count are unchanged. The new C5 helper increments those counters on every successful
short-gap offer, so this cohort exercises no successful C5 addition. Its independent authored
alpha remains valid truth, but the repeatedly consulted cases are a development regression set,
not an untouched holdout or active C5 precision qualification. The seven synthetic geometry
variants establish implementation mechanics. The three HDR dumps are development witnesses
without independent alpha truth; the ordinary 72-frame gate retains its legacy-matte scope.

The cap and width coupling remain empirical completion policy. A real open background notch can
satisfy the same Raw/background-root conditions; an attached scene branch can challenge whole
black-component permission. These were source-level falsifiers at that checkpoint; the subsequent
GPU experiment below reproduces the open-notch failure. C3/C4/C5 protection remains experimental. The next work is
the [shared-width original-outline detection proposal](host-sbs.md#proposed-shared-width-original-outline-detection-unimplemented),
including the upstream BODY admission problem and separate fill/style evidence, rather than
another permissive gap exception. Its clarified scope pools original-outline width from easy
same-style regions to detect the original outline in hard regions, preserving source RGB and
style. It neither generates a replacement outline nor adds original-style removal. Production
code and the published C5 package are unchanged
by this documentation review. Exact pre-review documentation bytes are archived under
`E:\ApolloDev\host-subtitle-rule-rethink-20261103` to preserve the earlier validation identities.

### Shared-width admission experiment

The 2026-11-03 experiment reuses the existing per-unit F2 width measurement to constrain
original-outline admission. It does not implement cross-unit style matching or independently
qualified inner-text support. Original RGB and the published C5 product remain unchanged; no new
outline is generated and no original style is removed.

The new native fixtures author scene RGB, caption RGB and caption alpha separately, then run the
real GPU pipeline. An open scene notch and a faint black outline can satisfy the same geometric
gap certificate. C5 protects all five alpha-zero target points across widths 3 and 4. Experimental
A adds conservative source/background compositing checks, but loses a prior one-pixel mechanism
case. B repairs that mechanism case and passes both new flat-background tests. Neither is promoted.

A fixed 24-case development challenge then varies flat, quantized gray/RGB gradients and checker
backgrounds, widths 3/4 and alpha 0/8/128. All cases, prerequisites and failures are retained.
The two faint-checker cases fail the intended pointwise-donor prerequisite because Code1 already
marks the donor. Their exported masks remain valid full-pipeline observations, but cannot isolate
the new helper. The literal counts below include those cases rather than silently filtering them.
Across the 20 background and 40 caption target points:

| Executed rule | Background points protected | Caption points missed |
|---|---:|---:|
| Published C5 | 20 | 3 |
| Experimental B | 2 | 6 |

B loses three additional faint-gradient caption points. The remaining two false protections use
the pre-existing measured-band AA path; they are not evidence that B's new helper admitted them.
The clean opposite pairs at these gradient gaps do not span both spatial axes, even when every
integer offset on the width-derived ring is checked. A locally constant background or an
unconstrained gradient fit would add an unsupported assumption. Width alone does not resolve
these ambiguous pixels. This challenge is a consulted development set, not an untouched holdout.

The correctly pinned B replay also withdraws 1/0/556 selected pixels from the three saved C5
captions, with no gains. In the newest capture it retains only 12 of C5's 284 additions over C4,
and withdraws another 284 previously protected C4 points. Source, upstream maps, Raw and all
non-bit-512 packed evidence stay exact. Those dumps lack alpha truth; the counts describe mask
withdrawals, not 556 independently labeled caption errors. The authored regression and these
withdrawals reject B as a replacement. There is no new live-test candidate from this experiment.

Replay identity matters: the first folders named `native-candidate-a`, `native-candidate-b` and
`native-c4-bg-probe` actually executed compiled C5 shaders. Their labels do not establish candidate
parity. The test loader now supports an explicit, authenticated fixture-F2 override, used by both
the producer and its diagnostic mirror. Reports distinguish effective closures from compiled
production closures, and the completed-draw reader checks the same source/closure joins. New
`*-effective` captures provide the B and C4 evidence above; old mislabeled attempts are preserved.

A separate actual-GPU probe holds C4's mask, source bytes, geometry and completed pixel shader
fixed, with separately authenticated draw leases, and changes only background erasure from the
current 3x3 neighbourhood to marked pixels. All six
ordinary draw routes match archived C4 exactly. All 284 newest C5-added points are C4 M0 pixels
being erased by the ordinary background path; the probe stops erasure at all of them. Displayed
RGB changes at 193 left-eye and 142 right-eye target points. Bright tears shrink, but fringes and
ghosts remain. The UI reference contains original caption colour, not clean-background or alpha
truth. This establishes an erasure contribution independently of mask changes; own-only erasure
is not qualified as a production fix.

Evidence lives under `E:\ApolloDev\host-subtitle-shared-width-pass1-20261103`:

- `effective-f2-probe-test-freeze.json` freezes the exact executable and both fixture sources.
- `candidate-b-vs-c4-c5-native.json` binds the literal real-dump mask comparison
  (SHA-256 `44ac326c4a9c7cb2545004aa9fd2380db3dcdebd1a57023995c65f6c29ac0fa4`).
- `true-c4-fixed-mask-bg-erasure-probe.json` and its PNG bind the fixed-mask GPU comparison;
  `true-c4-bg-probe-root-review.json` records the independent review.
- `authored-scene-challenge-matrix.json`, `challenge-c5-control` and `challenge-candidate-b`
  retain all 24 authored input/output bundles and the failing native XML/logs.
- `candidate-b-background-rank-audit.json` binds the source/rank audit and independently recounts
  every challenge target without filtering failed attempts.
- `saved-replay-loader-audit.json` records the earlier replay-loader mistake.

The truth assertions remain intact. The known-red native cases are explicitly named
`DISABLED_AuthoredOpenBackgroundNotchIsNotGlyphCoverage` and
`DISABLED_AuthoredScenePatternsChallengePixelAdmission`; opt in with
`--gtest_also_run_disabled_tests` and those exact filters after configuring the fixture's required
source roots and pins. They are counterexamples, not passing regression gates. The faint-outline
positive stays enabled. No skip, exclusion or old finite C5 gate resolves the recorded failures.

After retaining those two known-red tests explicitly, the focused native suite runs 68 tests:
66 pass and two saved-input tests skip without their optional inputs. The two opt-in
counterexamples are also rerun and remain failing; they are not included among the 66 passes.
Generated-contract checks and the published C5 launcher check pass. A fresh isolated ordinary
control covers 72 frames with the approved numeric runtime; its receipt
`ordinary-host-control-ready-receipt.json` qualifies only that control, not an A/B or either rejected
candidate. The failed initial engine-manifest preflight is preserved; the supported untimed warmup
was used for the successful new control. No new headset or FPS acceptance is claimed.

### Enclosed gaps in the current C5 mask

The subsequent full-mask census corrects the earlier narrow two-crop interpretation. C5 fills
the old row333/upper311 gaps, but it still has enclosed M0 islands elsewhere. Within the newest
saved caption's actual unit0 cover, four-connectivity gives 15 enclosed components containing
30 pixels and one open component containing 118,091 pixels. Under eight-connectivity, 11 enclosed
components contain 29 pixels; one point connects diagonally to the exterior. These are literal
mask topology counts, not caption-alpha labels.

All 30 enclosed points fail the learned RGB8 black box `(0,0,0)..(10,10,17)` and have zero
BODY, Inner and packed support. Twenty-five are already within the current known width4.
For example, `(1555,422)` is an enclosed singleton with source RGB8 `(1,5,19)`: its blue19 is
outside the learned blue17 ceiling, so grow writes no measured BAND. Its two usable adjacent
original-black AA donors see opposite-source gaps14 and11, below the existing21 guard, and
cannot use the inherited low-gap exception. This is an early mask-admission omission, independent
of later renderer repair. Its first source-passable root is exterior even though its final-mask
zero is enclosed; source-graph closure and final-mask topology are different.

The earlier six-point cohort is not a complete census and must not be called exterior fringe
only. All six are four-connected enclosed islands. The larger remaining examples include a
nine-pixel row at x1617..1625/y313 and a six-pixel component in `[1402,375,1404,379]`.

The new evidence root is `E:\ApolloDev\host-subtitle-mask-holes-pass2-20261103`:

- `current-c5-mask-hole-audit.json` (SHA-256
  `a8e24f1f3cb00633dc4457e859744c136aceb9d5dd056eb7a9a6d2a1e8891eb2`) binds every island,
  source value, width distance and adjacent donor decision.
- `native-source-and-selected-masks-v2.json` and its PNG show the authenticated native source
  beside literal C4, C5 and genuine rejected-B masks, with all 15 C5 islands marked. There is
  no CPU renderer, generated mask repair or assumed alpha truth.
- `experiment-plan.json` fixes the proposed interior-support checks and counter/cutout gates
  before new GPU experiments. External candidate A2 keeps physical-black and Raw classification
  unchanged, but deliberately introduces a new use of completed C3 component evidence for
  ordinary AA support.

The P10 A2 GPU experiment is rejected. All 30 independently authored attempts satisfy their
recorded prerequisites. It recovers all 84 primary ink misses in 12 opaque-tint cases, but adds
42 newly protected background pixels in six blue-cutout cases. The 12 genuine O-counter cases
already expose 108 false protections in C5, and A2 retains all of them. Outside-background tests
remain zero, and no case is filtered. In the three correctly pinned saved replays, A2 adds
60/7/6 selected pixels without withdrawals; six of the newest mask's 30 observed island pixels
are covered. Those real pixels lack independent alpha truth, so this is not a precision claim.
`authored30-and-saved-a2-comparison-v3.json` (SHA-256
`74457be7bdbb7bf0cc037df345a5e8406ffeeb31063e17f09e3d8f123b72d032`) binds the complete result.
Earlier reader setup failures and unchanged source captures are preserved.

The counter controls show why filling every enclosed mask gap is insufficient. Some transparent
counter pixels receive CLOSURE and geometric AA despite having no current BODY/LOW/Inner support.
A per-pixel width cutoff is also unsafe: the actual C5 source-risk audit identifies 99 protected
pixels at `t+1`, including 38 C5-over-C4 additions, that it could remove. The uncompiled B1
experiment is held. A whole-source-root envelope-overlap guard flags none of those protected
pixels in this saved cohort; this is a source-predicate risk audit, not a GPU result or alpha truth.
`current-c5-closed-recipient-risk.json` records all points and root groups. B2/B3 isolate that
conservative counter guard and preserve the existing black detector. B4 adds a separate
exterior-background exception for existing AA offers, as described below.

### P10 B4 counter-recipient correction

B4 is integrated into the working source as a narrowly bounded counter correction. The frozen
C5 live package remains preserved. This is distinct from the rejected P9 candidate B and from
P10 A2: B4 does not add A2's interior-support rule or widen the physical-black model.

For a current issued Raw graph with a known width, a closed-unseeded first-source component
needs at least one member within the exact original LOW-or-Inside width envelope before its
unsupported recipients may receive closure. Existing current BODY, own LOW/HIGH and Inside
recipients are preserved. The entire component's witness is reduced during aggregate and
consumed only in the following ordered dispatch. Original AA offers use the same recipient
restriction, with one exception: their actual original physical-black Raw donor can separately
prove current exterior background within the existing pointwise width bound. This exception
does not grant closure. New M/AA and the component-wide opaque witness cannot qualify donors.
Exterior recipients and unavailable-width/unissued-Raw fallback policies are retained.

All 30 predeclared authored attempts complete their recorded prerequisites. Across the 12
genuine O-counter cases, false protection falls from 108 pixels to zero. This is an active
negative: C5's closure and geometric-AA paths had protected those transparent pixels despite
their being beyond the original width envelope. The new counter test is enabled as
`AuthoredWhiteOCounterBeyondMeasuredStrokeRemainsScene`.

B4 retains the other C5 outcomes: 84 primary ink misses remain in the 12 opaque-tint cases;
the six blue-cutout cases still have 12 primary false protections and nine corridor false
protections. Positive ink corridors retain all 18 points, and the 270 outside-background
points remain unprotected. No case is filtered. The opaque-tint and blue-cutout tests remain
explicit disabled experiments with their original truth assertions intact, not passing gates.

The three authenticated cold saved replays have zero selected-mask additions or withdrawals:
their C5/B4 protected counts are respectively 163,584, 73,749 and 19,511. Thus all 30 original
observed mask-hole pixels remain unprotected. These saved masks have no independent alpha
truth; equality is a compatibility observation, not a precision or hole-recovery result.
The regular external-B4 native group runs 68 tests: 66 pass and two optional saved-input tests
skip. Those skips are separate from the explicitly configured three-case saved replay.

`authored30-regular-and-saved-b4-comparison-v1.json` (SHA-256
`6d5e237cf2e72ce64c93c3ec81fe2f56b6c702b95c8ad5572f0da7fb9887da19`) binds these finite results.
The subsequent integrated B4 build passes 69 runnable native checks: 67 pass and two optional
saved-input checks skip. The three-case production replay and completed-render replay are run
separately and both pass. All 938 Python tests pass with 13 skips. The 28 independently authored
precision masks and all 24 selected-sequence masks remain byte-identical to C5; all four
glyph/background addition/withdrawal counts are zero. Existing authored misses and false
protections remain in their unfiltered scores. The bound result is
`integrated-b4-qualification-comparison-v1.json` (SHA-256
`2787e07efbc110ef8cd514698e177d9be7f26e30023e30d5eab22a6c953c2fcd`).

The fresh 72-frame ordinary capture and scoring complete. Its original report command exits2
because it receives a results-file path where a directory is required; that attempt and its
outputs remain preserved. A separate authenticated CPU report from the completed control and
treatment exits0 with `screen_neutral` and passing canonical gates, using the same approved
Python and numeric runtime throughout. No numeric-runtime or metric mismatch is waived.
This is functional evidence, not a performance or headset result.

`integrated-b4-functional-comparison.json` (SHA-256
`a7bb0578145c14a429d16be976ce32f4fdc643382ff19b570afd0ff8b28693a1`) joins the frozen
native binary and fixture sources to these captures. It verifies three default-production B4
masks and receipts against frozen C5, all 18 actual GPU-rendered FP16 crop payloads, and all
504 ordinary artifact pairs with matching functional metric JSON. Complete draw outputs are
joined through native-recorded hashes because only the crops were exported. The original
report-path error and a reader's initial test-count scope error are retained separately from
the successful comparisons; no capture or provenance is rewritten.

After adding the independent low-contrast scene regression below, the final native group runs
70 tests: 68 pass and the same two optional saved-input checks skip. Five explicitly disabled
experiments retain their unresolved truth assertions. The tested host binary and B4 shader
assets are unchanged by these test-only additions. `integrated-b4-final-tests/receipt.json`
(SHA-256 `ca744d82fe5f087efa954bff8948eddd08c37088ce312c0c96f691028d2c74a8`)
binds the final XML and archived executable, fixture source and generated helper. The separate manual launcher
`E:\ApolloDev\run-outline-detection.cmd --check` verifies the B4 executable, 124 frozen shaders,
models, owned configuration and 61 warmed shader keys without starting a host. The older C5
launcher also still passes its check. Running the B4 host requires the user to quit the installed
tray host and start the launcher as Administrator; no headset result is claimed here.

A correction of the remaining ink holes still needs independently authored ink and genuine
transparent-counter/cutout negatives; neither a generic fill-holes operation nor a dump-specific
RGB threshold follows from this evidence. Live acceptance remains open.

### P10 A3 low-contrast AA experiment rejected

A3 changes only reachability of the existing RGB interval tests after the early contrast21
guard. Its additional current certificate requires an original LOW-or-Inside width envelope,
strict eight-ray surrounding original black support in the actual donor's Raw root, at least
one measured BAND point, and only the existing pointwise C2 background witness. It introduces
no C3 component AA permission, widened black model, generated outline or new RGB fit. B4's
counter-recipient restriction and ordinary observer/history admission remain intact.

The eight-case matrix is frozen before GPU execution: original black `(3,4,8)`, widths5/6,
independent flat/affine scene layers and original ink alpha32 or0. Every attempted case and all
12 O-counter controls complete their required model, graph, width and donor checks. B4 misses
all four original AA targets and correctly leaves all four transparent targets unprotected.
A3 recovers the four AA targets but newly protects both affine alpha0 targets. Whole-frame
comparison finds exactly four glyph additions, two background additions and no withdrawals;
no case or inherited error is filtered. The 12 O-counter controls remain unchanged.

The flat positive and affine transparent negative have identical local source values:
`q=(13,18,25)`, original donor `p=(3,4,8)`, reference `r=(15,20,27)`, contrast19 and distance4
from original inner support. Both satisfy the unchanged RGB fit and the new surround certificate.
The scene gradient therefore supplies a genuine false positive, rather than a negative rejected
before the new rule can activate. Width and this local RGB certificate do not establish alpha.

`lowgap-b4-a3-comparison-v1.json` (SHA-256
`2f60a8450f42e24cdd81308267e57128ea00c27b857ba9691076b44c911ffd26`) binds all inputs,
independently composed alpha truth, actual GPU maps, source closures and failure classification.
Both diagnostic test commands exit1: B4 for the four known AA misses, A3 for its two new
background protections. These are semantic failures with valid prerequisites. A3 remains
external and is not included in the product build. The transparent-scene regression is enabled;
the original-AA positive remains explicitly disabled and tracked as an unresolved miss.

`lowgap-a3-alpha-counterexample-v3.png` and its bound JSON receipt show the actual source pixels,
independently authored alpha, B4 mask and rejected A3 mask for the width5 twins. All crops are
literal native arrays enlarged with nearest-neighbour sampling. The width6 twins are included
in the complete eight-case result. This is a diagnostic source/mask figure, not a reconstructed
scene, renderer replica or generated outline.

### P11 C7v2 current mask completion candidate

C7v2 is the current manual candidate for the latest saved dump's missing mask pixels. Its
[current policy](host-sbs.md#current-mask-completion) is separate from the unimplemented shared-style
redesign. It preserves original RGB/style and creates no outline. C5 and B4 packages remain
preserved fallbacks; no new headset or FPS result is claimed.

The first compiled C7 attempt is rejected and preserved: the full authored/selected screen adds
36 declared ink pixels and 26 declared background pixels, with no withdrawals. The background
regression occurs in two same-source observations despite compatible width profiles. The
tightening removes the member-width exemption from the existing counter test; it changes no
colour threshold or truth. `c7-qualification52-comparison-v1.json` (SHA-256
`460fd970fee786158a7b58f14fd64e33ece44ad0c7dc81e56a3e6b23e1235e2f`) and the complete
`c7-precision18-new-regression-v2.json` retain that failed result, every changed coordinate and
the independent original alpha. The failed experiment is not the final candidate's proof.

Final C7v2 passes the fixed 28-case authored development screen and all 24 selected-sequence
cases: one added declared ink pixel, zero added background protection and zero withdrawals.
The smaller ten-case partition and all selected-sequence masks are byte-exact B4. The bound
`c7-v2-qualification52-comparison-v1.json` has SHA-256
`e95f1e5ebd33af087255008fe1411df15b7a169ecc1f857161fff8bd569cf679`.
This repeatedly consulted development cohort is not an untouched generality test.

The final compiled three-source cold replay changes masks by 0 / 0 / 30 pixels against B4,
with no withdrawals. The newest mask grows from 19,511 to 19,541 and selects exactly the original
30 recorded missing points; the first two remain unchanged. Actual frozen proof, fault state,
pending offers, all accepted component roles and unique counter changes are authenticated.
Final-binary completed-PS replay also passes with regenerated candidate background and unchanged
source/draw inputs. `c7-v2-saved-comparison-v1.json` (SHA-256
`6778f0e5f693b05c9e1f278ce90fd25787b27690b828b324a01815c15d5577e3`) and
`c7-v2-newest-mask-and-gpu-v1.json` (SHA-256
`9f147ee5b10382a8984ead06e46ef1b5c5ae0e169e7da2d0a817ded9026c0e2c`) bind the native
coordinates and actual GPU crops. The corresponding PNG shows literal source/B4/C7v2 masks and
actual draw crops, using only nearest-neighbour enlargement. These real dumps have no independent
alpha ground truth; this is finite mask/execution evidence, not complete visual-quality proof.

All 38 existing authored full masks remain byte-exact B4. The twelve genuine-counter cases
actively reach the completion counter veto and retain all 108 declared background targets.
Inherited errors remain: 84 primary opaque-ink misses, twelve primary blue-cutout false
protections and four low-contrast AA misses. `c7-v2-known38-comparison-v1.json` (SHA-256
`94328b27863fc44f8a7ccc1cceb05d458d13fd5004e738bcdace7a53b6f79828`) retains all truth,
failed diagnostic assertions and cases. Separately, all sixteen new predeclared challenge cases
have unqualified controls: the eight positives are already M1; four bright counters already
protect 516 declared background pixels; four dark counters lack the required boundary witness.
The final treatment changes none of their full masks. Their zero-delta result supplies no
qualified positive acceptance, and the old errors remain explicit in
`c7-v2-authored16-diagnostic-supplement-v1.json` (SHA-256
`bee55e5699315c45aa429f58bbaf86cd86d40bcaec1de496248ed5d09cb1037f`).

The focused final native group runs 79 checks: 77 pass and two optional saved-input checks skip;
the separate configured saved/PS replays pass. Python discovery runs 940 tests with thirteen
skips. The fresh ordinary 72-frame comparison passes `screen_neutral` with all 504 actual artifact
pairs byte-exact; `ordinary-c7-v2-paired-complete-v1.json` has SHA-256
`c5fbe319da639465291ada1d8cd5bee0596e960eabc18febd1b7c218d3bae704`. This is ordinary legacy-matte
screening, not selected-compositor or performance acceptance. The separate manual launcher
`E:\ApolloDev\run-last-dump-repair.cmd` is prepared with its own `review-last-dump-repair`
configuration. Its artifact check passes for the executable, frozen shaders, models, configuration
and warmed keys. `manual-c7-v2-stage-v1.json` has SHA-256
`fe8149b33197bd6ffcd4a2cef7101482c53d5d5a0e57fc8b735fef9ef4be14d2`.
The preserved C5 and B4 launcher checks also pass. Packaging did not launch a host.
All P11 receipts are under
`E:\ApolloDev\host-subtitle-last-dump-pass3-20261103`; source, main and test identities are frozen
separately from the rejected attempt. These finite checks do not establish broad caption alpha,
unseen styles, live Galaxy XR behavior or FPS.

### P12 generic original stroke coverage

The [optional coverage rule](host-sbs.md#original-stroke-coverage) preserves original source
RGB and uses the same recipient model for open edges and enclosed gaps. The adjacent-null
variant is packaged as the separate Generic Outline G1 manual candidate after the finite gates
below. Published C7v2 and its preserved launchers remain separate fallbacks.

Two earlier models are rejected and retained. The compiler-fixed `4bce` prototype recovers
four flat authored AA targets but adds two independently declared background pixels in the
existing 38-case cohort. A missed original-AA reference had passed the original-M0 reference
gates; affine feasibility did not establish clean scene. The `b599` model adds the direct
reference-colour null test and removes those two errors, yet its production large18 screen
still adds two declared ink pixels and one declared background pixel against C7v2, with no
withdrawals. Its alpha-zero target lies in exterior-connected scene just outside the authored
stroke. Neither failed result is acceptance evidence for the later model.

The external `1831` adjacent-null development comparison retains all 70 attempted cases:
24 new broad cases, eight existing-geometry AA cases, and the existing 38 cases. It adds twelve
declared ink pixels in the broad set and four in the eight-case set, with zero new background
protection or withdrawals in either. All 38 existing final masks remain exact. The six dark
broad cases change the learned K upper bound to `(11,13,15)` and fail the predeclared constant-K
prerequisite. They remain fully scored and unqualified, supplying no active AA acceptance.
Inherited misses and failed diagnostic assertions
also remain. `adjacent-null-paired-comparison-v2.json` records an overall blocked/review status,
not a full native or general-alpha pass. This repeatedly consulted development set is not an
untouched holdout.

The SDR trace has a per-case production-parity gate. Frame8 matches production CBC, Raw, F2
and final M, then matches all eight mirror outputs. Actual `1831` retains its declared alpha23
recipient with Offer1; all 28 exported source, proof, upstream, mask and receipt artifacts are
byte-identical to `b599`. The scalar interval bound of `16/255` is thus checked against actual
GPU admission at that point. Frame5's cold replay matches CBC/Raw but differs in F2/M, so it
supplies no authorized graph or reference-eligibility trace. Missing prior weak-AA state is a
source-consistent explanation, not established per-pixel attribution or a parity waiver.
`sdr-adjacent-trace-gpu-confirmation-v1.json` preserves both failed native attempts and this
limited frame8 result; it is not production qualification of the external shader.

- Final 52-case production screen: **passes** against C7v2, with two added declared ink
  pixels, zero added background protection and zero withdrawals. The two gains are the same
  alpha23 source point in observations eight and nine. Small10 and Selected24 masks are
  byte-identical to C7v2. The rejected `b599` alpha-zero point in frame five is unmasked.
  `canonical-adjacent-null-qualification52-comparison-v1.json` has SHA-256
  `71e8fed669110549deeff3c4ed1dcd1ee7b6ef15c4d41b2bcee1971d01958ec3`.
- Final default-production saved3 and completed-render comparison: **passes** the bounded
  ownership/execution checks. Masks change by 1 / 0 / 0 against C7v2's cold replay, with no
  withdrawals; all thirty repaired points in the newest dump remain selected. The first added
  pixel is native `(723,428)`, analysis-ROI `(723,1634)`, parent source `(1104,1882)`.
  Its original source RGB is preserved. Real dumps have no independent alpha truth, so this
  is an unverified added mask pixel. Both the original live captures and their differing cold
  replays remain distinct. `canonical-adjacent-null-saved3-comparison-v2.json` has SHA-256
  `a40bb6014982f4580d29cb4f7ec1e7a66f52e7d6463cb8263dcf38223428d8cc`.
  The completed-PS check binds all eighteen actual GPU routes; the regenerated-background
  route can change pixels beyond the single new mask point. Its bound comparison
  `canonical-adjacent-null-saved3-and-ps-comparison-v2.json` has SHA-256
  `332f5e15bafdebe358710d13e159a6d86f1c9991669c193d917911b6a5dd663d`.
- Final ordinary72 functional comparison: **passes** `screen_neutral`, with all 504 actual
  artifact pairs byte-identical to C7v2. The same approved numeric runtime, source clips and
  model/engine identities are authenticated. `ordinary-adjacent-null-paired-complete-v1.json`
  has SHA-256 `d22a576c77d458e134ef19ff994978410e39eb6b910df992acb78031272ea4fa`.

The optimized default build and 73 focused contract/evaluator tests pass. The final native
group runs 80 tests: 77 pass and three optional saved-input tests skip. The separately configured
three-source producer and completed-PS tests both pass. The compiled source/runtime archive
`candidate-adjacent-null-source-runtime-freeze-v1.json` has SHA-256
`d72a17de17930a0699f938383bc9844ed0c478ee9120ce9a5b34862da91abca3`.

`E:\ApolloDev\run-generic-outline.cmd` passes its no-launch artifact check for the exact
executable, 128 frozen shaders, seven model/runtime assets, owned configuration, immutable
source-key authority and 67 current warmed keys. It uses its own `review-generic-outline`
configuration and preserves C7v2/B4/C5; their checks also pass. The stage receipt
`manual-generic-g1-stage-v3.json` has SHA-256
`e77d616daaf8f7052ba3e865553f8e6f5c4512fb6328d8ddc59f81aa2e046240`.
No host was launched by packaging or checks. Manual headset, latency and FPS validation remain open.

All P12 evidence is under `E:\ApolloDev\host-subtitle-generic-outline-pass4-20261103`.
Source-identical ink/scene ambiguity and inherited alpha errors remain unresolved. These finite
checks establish no new live Galaxy XR, broad-style or performance acceptance.


### P13 subtitle execution and ownership cleanup

The performance review keeps G1's admission, original RGB, counter vetoes and history boundaries
unchanged. Coverage consumes the captured black-like bit, checks necessary donor availability
before the exact width search, reuses shared stencil references lazily, and shares exact colour
and interval arithmetic through one generated/checkable owner. C7 completes cheap component
validation before width searches without cropping its original-M0 graph. Rejected-root private
width flags may consequently differ; pending permission, final M, counters and source proof retain
their existing meanings.

Production hook constants are immutable, shader/device validation reuses retained exact pointer
tuples, and Source5 capture/envelope submission shares one owned transaction. Current resources,
source/receipt joins, predication and public consumption checks remain current on every call.
Own-offer storage is explicit diagnostic opt-in; normal protection does not require that plane.
At the saved3056x510 band this removes6,234,240bytes of diagnostic storage and its full-plane clear.
New bounded, nonblocking timings separate legacy work, Source5, CBC and F2, with actual owner
record/replay/allocation diagnostics and CPU queue time. They measure command-stream intervals,
including submission bubbles, rather than isolated kernels or live headset FPS.

The background static-constant/dynamic80-byte experiment is rejected. It preserved every literal
pixel but added about0.0046/0.0048ms of CPU draw time in two interleaved selected24 comparisons.
The final source restores G1's background shader/layout/updates, retaining only immutable shader
validation caching. The shared width-envelope plane was also not added: completed rejection now
avoids those queries, and an additional full-plane allocation/pass has no demonstrated benefit.
Legacy dense-M fallback and per-draw current-source background reconstruction remain required.

Evidence is kept outside Git under `E:\ApolloDev\host-subtitle-perf-pass5-20261103`:

- `candidate-source-runtime-v1.json` freezes the initially parity-qualified experiment; it is not
  the final launcher package after the background rollback.
- `performance-parity-v1.json` (SHA-256
  `b68f5482a0c10af132c91ddb7c8e6567d2fc92b282774099a466bcd12549fe27`) checks all70 native
  attempts, Saved3, all52 actual production frames and eighteen saved completed-renderer routes.
  Final masks, own offers, counters and rendered pixels are exact. Explicit graph-root equivalence
  and omitted width bits on already-rejected components are independently audited.
- `ordinary-byte-parity-v1.json` (SHA-256
  `0a87dc4b1e2638a805371d8883e0af67ed18f7d0559d110c27670ddaa33b025a`) checks all576
  PNG/F32 payloads from72 ordinary frames, including model-input tensors, byte-exactly against G1.
  The official evaluation also passes `screen_neutral`; ordinary rendering does not substitute
  for the selected-compositor qualification.
- The exact Python708 runtime is used throughout. The942 evaluator tests pass with13 skips;
  the86-test focused native run passes83 with3 optional input skips. Five enabled G1 cases prove
  actual positive/negative own offers, diagnostic-mode parity, Root0 hold, stale-source rejection
  and reserved-proof sticky-fault rollback. The broader257-test run passes236 with21 opt-in/input
  skips; explicit Source5/selector opt-ins and final-build results are bound by the final receipt.
- `interleaved-performance-summary-v1.json` retains both controlled timing pairs and unchanged
  inference controls. Their variation does not establish an FPS or subtitle-latency improvement.
  Reduced repeated work and diagnostic storage are implementation facts; no speedup is claimed.

The final master binary, source/asset identities, post-merge checks and launcher seal are recorded
in `final-master-validation-v1.json`. `E:\ApolloDev\run-dev.cmd` selects that validated
RelWithDebInfo package. The earlier direct G1 launcher remains a preserved fallback.
Finite parity preserves the existing limitations and failed diagnostic cases; it establishes no
new dark-scene, tint, left-aligned multiline, semantic-alpha or physical-headset acceptance.


### P14 shared Game 3D pin-band comparison

The native-mask comparison tests the existing completed Host compositor against Game 3D's
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
3840-wide cases, versus about 66.45 MiB for the current push/pull owner, excluding common inputs,
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
map. The prior background implementation remains available to historical comparison fixtures;
its storage and proofs do not grant shared-warp authority.

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

### P19 foreground edges and integer clearance goals

Two bounded changes address missing text edges and unnecessary background deformation. The
foreground pass uses an immutable post-Inside/C7/G1 mask, current qualified foreground donors and
one shared RGB blend interval with the current learned outline colour. It preserves source RGB
and generates no outlines. The UI-plane controller searches the same cover population and
clearance budgets on a one-source-pixel grid, while retaining the public four-pixel diagnostic
index, cap, clock rules and symmetric continuation filter.

The foreground hypothesis was measured on nine previously exposed authored development first
cues, using actual D3D production-mask readbacks and independent original RGBA alpha. It recovers
392 visible inner-edge pixels and 16 visible outer-edge pixels; no new zero-alpha background or
original-ink withdrawal occurs. One of the additional three cases has valid EMPTY OCR and remains
a full missed-ink case in the score. The six-case fixed-plane stereo replay improves 291
foreground pixels with none worse; five cases are byte-identical. Background error improves on
142 pixels and worsens on 48, with at most one RGB8 level of mean-channel error increase. This
does not solve detached unseeded punctuation, broad glow, or the existing false foreground
families. Native-scale OCR crops also fail to recover the 74 recorded punctuation misses.

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
rejected after foreground/background regressions. A separate exact vertical-exterior colour
veto stayed inactive on all six cases and is not integrated.

The external evidence root is `E:\ApolloDev\subtitle-mask-background-20261104`. The nine-case
alpha receipts are `private-resume-v1/fg-edge-all6-quality-v3.json` and
`private-resume-v1/fg-family-def-quality-v2.json`; the two fixed-plane RGB receipts are
`private-resume-v1/fg-fixed-{original,supplemental}-RGB-quality-v1.json`. Integer source-only
feasibility is `integer-plane-v6/integer-validation-v6.json`; the complete measurement is
`integer-plane-quality-v6/report.json` (SHA256
`2ea5465a9728bee0305cc87ff856a2f3397cefa29c10b0bcd7ef8fccee8c44e6`).
All stages use the same recorded machine evaluation interpreter and package fingerprint. No
captured state was retagged and no CPU mask candidate or renderer supplied these measurements.

The production port passes all nine saved composition replays against the actual GPU hypothesis
masks, including the 392-pixel mixed-width recovery. All 55 controller GPU tests pass, including
four new regressions covering integer goals, coarse-tail compatibility, cap edges, ROI union,
invalid input, clocks and empty reset. The five optional-program runtime tests and five shared-warp
ownership tests also pass.

Five foreground-owner GPU regressions pass. Two use explicitly authored current records to isolate
overlap accounting, immutable donors and disabled/zero-dispatch/stale-context behavior; the other
three exercise the production composition snapshot, binding restoration and foreign-device
rejection. The nine saved replays above supply the separate real-producer mask-parity evidence.

The actual main executable passes schema-6 qualification for 54 observations across nine source
sequences. Applied planes use the finer targets and every final blank publishes canonical zero.
The Latin control has identical source/timeline bytes; its finer continuation moves below the old
twelve-pixel target. This proves integration, not isolated stereo quality: the RGB evidence above
comes from the fourteen native renderer cases. The external actual-main integration audit is
`actual-main-plane-integration-v1.json` (SHA256
`1ca386738ad03eae0e168aa4da951b4aec6f0c9aa6140a5de1275ab6fdbfeeab`).

Packaged qualification and the requested combined live build are tracked separately from these
development measurements. Live headset acceptance remains pending; shader availability by itself
proves neither execution nor a visible gain.

### P20 ordinary Base rendering and render-binding cleanup

Ordinary live, queued and diagnostic shaders now call the canonical Base inverse directly.
The analytic subtitle shader include is removed. Shared current-glyph pinning and current-RGB
revocation retain their existing equations. Missing shared preparation or invalid selected
authority preserves ordinary Base with subtitle permission disabled; unavailable or invalid depth
retains the independent flat renderer. Render-only t8 leases, receipt backup and dump evidence are
removed. Raw and original masks still used by construction and maintained diagnostics remain.

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
