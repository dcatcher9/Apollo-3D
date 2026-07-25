# SBS 3D — current status and roadmap

Apollo converts captured mono frames into host-rendered SBS with TensorRT depth estimation and one
production geometry implementation: Apollo's occlusion-aware backward probe. The retired VD3D
forward/backward hybrid and its selectable profile were removed after headset testing found worse
rim/halo behavior and a serious thin-structure artifact for only about 0.12 ms of warp-time saving.
Historical experiment evidence remains in Git history and `sbs-feature-decision-revisit.md`.

Approved AR glasses connected as a Windows monitor also use an automatic local presenter; see
`docs/sbs-local-ar-glasses.md`. That path reuses the production depth and warp without NVENC.

## Shipping pipeline

1. Preserve source aspect while selecting a patch-aligned TensorRT input grid.
2. Convert SDR or HDR capture into the color domain expected by the depth model.
3. Transform model output into Apollo's high-is-near convention.
4. Normalize with permanent P2/P98 bounds and temporal range EMA.
5. Apply per-pixel EMA, accepted edge/change-aware EMA, Bestv2-derived subject estimation,
   and P5/P95 stretch/recenter.
6. Render Apollo's occlusion-aware backward probe.
7. Convert the packed SBS raster directly to the encoder format. If doubled width exceeds
   `sbs_3d_max_encode_width` or the selected codec's runtime `NV_ENC_CAPS_WIDTH_MAX`, preserve each
   eye's aspect while scaling to the lower cap. The configured production ceiling is 8192 packed
   pixels. The production RTX 5080 reports 4096 for H.264 and 8192 for HEVC/AV1, so a 3840x2160
   source becomes 4096x1152 in H.264 but remains 7680x2160 in HEVC/AV1. Apollo records the
   authoritative per-codec capability during NVENC probing and refreshes it at encoder creation.

Profiles remain configuration-only parameter sets over this single geometry. Define fields with
`sbs_3d_profile_<name>_<parameter>`; `sbs_3d_profile` selects the startup preset. Explicit top-level
`sbs_3d_*` keys override the corresponding selected-profile value. Artemis switches only between
Normal and Host SBS AI; changing the host profile requires restarting Apollo.

Bestv2 disparity is calibrated at the evaluator's 854-pixel source width and normalized to the
5120x2160 Artemis reference aspect. `sbs_3d_pop_strength` scales the final parallax (`0.25`–`2`,
default floor `1.25`) without changing that resolution correction. The accepted scene latch may
select up to `1.30` from depth-edge risk and holds the selection until a hard cut.

## Frozen processor decisions

- Bestv2-derived subject estimation and P2/P98 normalization are mandatory.
- Range-to-pixel temporal ordering and the Apollo probe are permanent.
- Edge/change-aware EMA is accepted for the Apollo profile at the headset-validated settings.
- TensorRT CUDA Graph replay is accepted and inherited by every profile. It validates the mapped
  D3D tensor addresses and model shape before every replay, warms and recaptures a changed
  signature, and falls back to ordinary enqueue when graph APIs/capture/launch are unavailable.
  It kept all 456 raw tensors, processed depth maps, and SBS frames byte-identical while reducing
  mean inference from 2.11 to 1.53 ms on core (-27.6%) and 2.38 to 1.59 ms on extended (-33.2%).
  Evidence: `cuda-graph-core-treatment` and `cuda-graph-extended-treatment` under
  `cmake-build-relwithdebinfo/sbs_eval/`.
- The production shader contains only the validated no-plane-lock warp. Its specialized loop
  removed the rejected plane-lock work from every search probe. Matched offline A/B reduced warp
  time by 8.37% on core and 8.24% on extended with no primary-axis regression or hard failure.
  Evidence:
  `plane-specialize-core` and `plane-specialize-extended`.
- The specialized loop precomputes its subject shift, parallax scale, convergence bias, output
  scale and safety bound once per output pixel. This removes repeated invariant arithmetic from
  every probe and reduced warp time by a further 3.38% on core and 3.19% on extended, again with
  no primary-axis regression or hard failure. Evidence: `parallax-invariants-core` and
  `parallax-invariants-extended`.
- The initialized-subject test is performed by the existing search-radius early return rather than
  repeated inside the specialized loop. All rendered, depth and coverage artifacts remained
  byte-identical while warp time fell another 1.43% on core and 1.25% on extended. Evidence:
  `shaped-branch-core` and `shaped-branch-extended`.
- The default subject-stretch-on and configurable no-stretch paths are selected once per output
  pixel rather than with a select at every probe. All non-performance artifacts remained
  byte-identical while warp time fell another 1.07% on core and 1.33% on extended. Evidence:
  `stretch-specialize-core` and `stretch-specialize-extended`.
- The standalone evaluator-only confidence audit separates predicted-boundary model risk from
  visible warp hazard and changes neither output nor the frozen metric contract. Warp-risk median
  AUC against independent final-eye artifacts was 0.69 on core and 0.74 on extended. Model-risk
  AUC against misplaced predicted boundaries was 0.59 on extended; that diagnostic cannot detect
  a GT boundary omitted by the model, and core lacks sufficient two-class model evidence. Warp
  risk is accepted for experiment screening only; model risk remains diagnostic, and neither is a
  renderer input. Evidence: `depth-confidence-core` and `depth-confidence-extended`.
- Horizontal silhouette-band snapping was screened at radii 1/2/4 with threshold and vertical-
  consensus variants, then rejected and removed. It reduced positional error on c747 and improved
  mean extended-suite halo by 0.09 px, but produced no validated primary-axis win, left GT edge F1
  essentially flat (+0.06), and increased c747 halo/rim. Evidence: `snap-core-r2-consensus` and
  `snap-extended-r2-d08-c08` under `cmake-build-relwithdebinfo/sbs_eval/`.
- Edge-aware warp-depth sampling was screened as continuous transition sharpening at strengths
  0.5/1.0 and normalized-depth thresholds 0.08/0.04, then rejected and removed after targeted core
  screening. Neither bound changed edge position or any validated primary metric; the stronger
  bound slightly worsened exact-hole fill and rim diagnostics. The existing 2x2 read remains the
  chosen anti-staircase/sharpness balance. Evidence: `edge-read-target-s05-t08` and
  `edge-read-target-s10-t04` under `cmake-build-relwithdebinfo/sbs_eval/`.
- NVIDIA RTX Video Super Resolution 1.1 was screened as an isolated same-resolution post-process
  on each rendered eye at quality levels 1 and 4, then rejected before extended-suite promotion.
  Still frames looked sharper, especially rain and thin highlights, but the independent enhancement
  changed corresponding pixels differently between eyes. At Q1 the flat-page disparity tails rose
  to +7.58%/-8.26% and flat-transition to +4.92%/-7.73%, all beyond the 3% comfort limit; Q4 also
  failed both flat clips and regressed static jitter in nine decision clips. Serial evaluation plus
  readback cost about 1.4-1.65 ms per stereo pair at Q1 and 2.5-3.0 ms at Q4 for 720-854 pixel-wide
  eyes. Do not place an independent learned image enhancer after stereo generation. Evidence:
  `vsr-per-eye-1x-q1-core` and `vsr-per-eye-1x-core` under
  `cmake-build-relwithdebinfo/sbs_eval/`.
- Applying Q1 VSR once to the shared mono source before warping was also rejected. Holding original
  inference depth fixed eliminated the catastrophic per-eye disparity mismatch, but visual
  inspection found new bright/dark ringing around flat-page lines, c747 rim-over worsened by 1.37,
  and static jitter increased. The apparent flat-transition stereo-volume win came from changed
  color correspondence despite byte-identical depth, not better geometry. Evidence:
  `mono-vsr-q1-target` under `cmake-build-relwithdebinfo/sbs_eval/`.
- Confidence-weighted local disparity compression was screened at strengths 0.10 and 0.20 on the
  five artifact-heavy clips plus both flat controls, then rejected and removed before extended
  promotion. At 0.10, c747 source-relative halo regressed from 11.41 to 14.69 with no primary-axis
  improvement. At 0.20, c747 worsened further to 16.09 and anime halo regressed from 5.80 to 7.39;
  pop and broad source residual barely moved. Reducing parallax only inside a silhouette hazard
  strip creates a new disparity discontinuity at the strip boundary instead of repairing the
  disocclusion. The evaluator-only confidence audit remains useful for diagnosis, but it must not
  directly scale local parallax. Evidence: `local-compress-target-s10` and
  `local-compress-target-s20` under `cmake-build-relwithdebinfo/sbs_eval/`.
  **The strip boundary was the symptom; the root cause is deeper, and it generalizes.** The
  backward search in `sbs_reprojection_ps.hlsl` solves `(x - uv) - eyeSign*P = 0` and selects the
  root of greatest depth. That selection is only well-posed because `P` is a monotone function of
  depth ALONE. Any gain that varies with image position makes `P` a function of `(x, d)`, so the
  gain enters the root equation directly: it can create spurious crossings and erase real ones, and
  "frontmost depth" stops implying "frontmost in parallax". Blurring the mask does not fix this —
  the field would have to vary on a scale much larger than the search radius to be locally
  constant, which destroys the silhouette localization that motivated it. **Rule: reshaping
  expressed as a function of depth is safe; reshaping expressed as a function of image position is
  not.** That single rule explains the whole ledger — the Bestv2 curve, `subject_stretch` and
  `subject_recenter` all reshape in the depth domain and all shipped. Depth-domain reshaping
  additionally requires `S'(d) > 0` everywhere, since a non-monotone `P(d)` inverts the occlusion
  ordering the frontmost-root rule depends on.
- DA-V2 Base FP16 was re-screened as a model-only replacement for Small and rejected as the
  production default. On the extended suite it improved GT boundary F1 from 45.07% to 50.27%, but
  left GT depth RMSE flat (11.28% to 11.26%), worsened mean source halo from 2.68 to 2.99, and
  nearly doubled inference time from 2.10 ms to 4.05 ms. The core suite also lost five stability
  votes and four warp votes; visual inspection confirmed that the large c747 edge-distance gain
  came with a different doorway/person interpretation and more rim. Keep the registry entry only
  for explicit future quantized/performance experiments. Evidence: `dav2-rescreen-core-base` and
  `dav2-rescreen-extended-base` under `cmake-build-relwithdebinfo/sbs_eval/`.
- A causal dual-rate Small+Base correction was rejected before extended promotion. Base ran every
  fourth frame; each positive-affine-aligned residual was bounded and held over current Small
  depth. A 25% blend was primary-axis neutral despite improving c747 edge distance (22.2 to 18.2
  px) and halo (8.31 to 7.80). At 50%, edge distance improved further to 16.4 px, but c747 halo and
  stretch worsened and fast-motion static jitter rose 26.9%. The marginal boundary signal does not
  justify a second TensorRT context/model, extra VRAM, scheduling complexity, and roughly 1 ms
  average inference cost. Evidence: `dualrate-i4-b25-core` and `dualrate-i4-b50-target`.
- Calibrated DA-V2 Small FP8 was screened with explicit TensorRT 11 Q/DQ models generated from 64
  representative core and extended frames, then rejected before extended promotion. Broad W8A8
  quantization increased mean core inference from 1.45 to 2.61 ms and produced five primary-axis
  regressions for one win. Restricting FP8 to transformer MatMul regions still increased inference
  from 1.45 to 2.52 ms and added a c747 stability regression with no primary win. At batch one,
  this already-small model pays more Q/DQ and scale-conversion overhead than Blackwell's FP8
  kernels save. Keep the production FP16 contract. Evidence: `fp8-calibrated-core-treatment` and
  `fp8-matmul-core-treatment` under `cmake-build-relwithdebinfo/sbs_eval/`.
- Exact-flow depth EMA was rejected on the two TartanAir sequences. Recursive history accumulated
  visible salt-and-pepper depth fragments. A constrained one-frame/full-edge-snap variant removed
  accumulation and reduced the ghost-edge diagnostic (easy 75.6% to 2.0%, motion 50.7% to 25.6%),
  but regressed hard-motion GT boundary F1 from 60.7% to 49.8% and worsened its flow-depth residual.
  Keep optical flow out of the production depth filter; retain it for validation and diagnostics.
  Evidence: `flow-ema-exact-1f-treatment` under `cmake-build-relwithdebinfo/sbs_eval/`.
- Warp-depth prefilter reshaping was rejected after core screening. Replacing the legacy separable
  `[.375, .25, .375]` kernel with Gaussian `[.25, .5, .25]` weights changed no primary axis and
  worsened mean halo. Removing only the vertical taps was likewise quality-neutral and saved less
  than 0.001 ms of measured warp time. Keep the validated legacy-equivalent prefilter. Evidence:
  `prefilter-gaussian-core` and `prefilter-horizontal-core` under
  `cmake-build-relwithdebinfo/sbs_eval/`.
- A safeguarded subpixel secant refinement of each reprojection crossing was rejected after the
  artifact-heavy core screen. It changed no primary axis, produced mixed halo deltas below 0.1 px,
  and increased warp time consistently by about 5-6%. Keep the existing endpoint interpolation.
  Evidence: `root-refine-target` under `cmake-build-relwithdebinfo/sbs_eval/`.
- Guided upsample, curvature, scene snap, range/depth floors, border fade, legacy shift, VD3D
  hybrid warp, and CPU warpsim were rejected and removed.
- Subject-plane lock, Bestv2 sharpen, and EMA-mask dilation were rejected and removed.
- A nearest/coherent multi-hypothesis replacement for the fallback background sample was a
  byte-exact no-op across all 275 core frames (`multi-fill-core`). Exact forward coverage still
  identifies holes, but the backward equation finds a mathematical root there and therefore never
  reaches its no-root fallback. Future disocclusion work must change root selection or explicitly
  consume forward coverage rather than elaborate that unreachable fallback.
- Explicitly consuming exact forward coverage was then screened and rejected. Forcing uncovered
  destinations to the tracked far-background candidate inserted thin black/white background
  slivers at subject boundaries. Across the six artifact-heavy clips, disocclusion flicker rose
  17.5-164%, the rim proxy rose 22.7-2795%, c841 and scene_cut acquired 5.59% and 6.14% positive
  disparity tails, and warp time rose about 17%. Fresh matched control/treatment imagery confirmed
  the measured slivers. Exact coverage remains diagnostic only; a future learned refiner would
  need color- and time-coherent synthesis rather than forced far-background substitution. Evidence:
  `coverage-hole-fill-control` and `coverage-hole-fill-target`.
- Symmetric horizontal edge-band supersampling was rejected after the full core screen. It nudged
  mean halo from 4.57 to 4.52 and the rim proxy from 4.41 to 4.31, but produced no validated
  primary-axis win, was visually imperceptible at the strongest frame, and increased mean warp
  time from 0.0553 to 0.0679 ms (+22.8%). Evidence: `edge-supersample-core`.
- The original comfort-only adaptive-pop proposal was rejected at a 2.0 ceiling: the worst signed
  disparity tail remained inside 3% (2.27%), but warp and stability accumulated 19 regressions.
  The accepted replacement is scene-latched and risk-aware. It chooses once from normalized-depth
  edge density, holds the multiplier bit-stable until a hard cut, and uses the independently gated
  1.25-1.30 band. Core gained c747 halo (11.41 to 8.31) with no primary regression; extended was
  neutral with no regression, and both suites passed hard limits. A 1.35 endpoint was separately
  rejected as an anime halo/stretch tradeoff. Evidence: `adaptive-pop-ceiling-core`,
  `pop-grid-130-core`, `pop-grid-130-extended`, `adaptive-pop-latched-core`, and
  `adaptive-pop-latched-extended`. The later `spring-adaptive-vs-fixed130` comparison added four
  independent cinematic true-stereo sequences. It was neutral: the controller was effectively at
  1.30 for the cave/daylight clips and backed off only to approximately 1.27/1.25 for the
  close-up/forest clips. Fixed 1.30 added at most 0.045 percentage points of pop spread and traded
  mixed artistic-reference deltas, so there is not yet evidence for a more complex controller.
  **Caveat for anyone re-testing this (added 2026-07-24): do not re-run these experiments and read
  the exit code.** The 1.35 and 2.0 rejections were measured against a metric schema in which
  `source_halo_p95`, `source_stretch_pct`, `source_residual_p95` and `flow_temporal_p95` were
  primary and baseline-gated. `source_halo_p95` — the metric that actually rejected 1.35 — no
  longer exists, and the others are no longer primary. Under the current schema three of the four
  baseline-gated metrics are `depth_gt_*`, computed on the pre-warp depth map and therefore
  structurally invariant to pop; only `static_jitter_p95` responds. A pop change can regress halo,
  stretch, fold, shear and disocclusion while `run_eval.py` still exits 0. Read the diagnostics by
  hand. Two further confounds: (1) the adaptive band is a ratio of only 1.04, below the `rel_tol`
  noise floor of every metric that could judge it, so the neutral `spring-adaptive-vs-fixed130`
  result is uninformative rather than reassuring; (2) `Bestv2SearchRadius` scales with
  `adaptive_pop_max` while `Bestv2ProbeSteps` is fixed, so raising the ceiling coarsens the probe
  for every scene — including scenes the controller declined to boost, which pay the cost and
  receive none of the gain. Decouple probe geometry from strength before attributing any ceiling
  regression to the gain itself.
  **SUPERSEDED 2026-07-24: the band is now 1.20-1.50.** With the probe geometry decoupled and the
  zero plane on `median`, a ceiling ladder on the core suite shows the historical rejection does not
  survive the new baseline. Against the configuration that shipped before 2026-07-24
  (`legacy` zero plane, 1.25-1.30 band), the adopted 1.20-1.50 band is **strictly dominant on every
  measured axis**: pop spread 1.0081 -> 1.0455 (+3.7%), mapping stretch 0.6325 -> 0.2750 (-56.5%),
  cross-row shear 4.1665 -> 4.1013 (-1.6%), static jitter p95 1.8159 -> 1.3708 (-24.5%), fold 0.000
  throughout, zero hard failures. More stereo volume AND fewer artifacts than the previous default.
  1.25-1.50 was also measured and buys more volume (+6.0%) but costs +2.9% shear; the wider 1.20
  floor was preferred because it lets the controller back off far enough on edge-dense scenes to
  keep shear below the old baseline, and the floor can be raised later if more volume is wanted.
  This also repairs the controller's authority: the ratio goes from 1.04 to 1.25, i.e. from below
  the noise floor of every judging metric to something that can actually be measured working.
  Evidence: `ceil-control`, `ceil-140`, `ceil-150`, `band-120-150` under
  `cmake-build-relwithdebinfo/sbs_eval/`.
  **SUPERSEDED again 2026-07-24: the shipped band is 1.20-2.00, paired with a magnitude-weighted
  risk statistic.** The extended suite was run and it did NOT reproduce the core-suite claim: at
  1.20-1.50 extended volume was -1.1% (a loss, which the no-compensation policy forbids), and the
  1.20 floor was the cause -- 1.25-1.50 recovered it to +2.3%. Ceilings were then laddered to 2.00.
  Fold stayed 0.000 at every rung on both suites, no hard bound was approached, and warp time rose
  only ~27% at 2.00 (inside the 0.30 perf tolerance), so neither comfort nor topology nor perf is
  what limits the ceiling.
  What limits it is `warp_cross_row_shear_severity_pct`, and the extended per-clip data showed the
  cost is **not diffuse**: at 1.20-2.00, nine of twelve clips were flat or improved and the entire
  +31.4% mean came from `tartanair_house_easy` (10.7 -> 23.5) and `spring_skeleton_cave`
  (1.2 -> 3.3). A wide band did not create a new problem; it made an existing classifier weakness
  visible, because at the old 1.04 ratio a misclassification cost 4% and at 1.67 it costs 67%.
  **The classifier now weights each edge texel by gradient magnitude instead of counting it.** Warp
  stress scales with the disparity step a silhouette produces, so a few violent discontinuities must
  outrank many gentle ones -- a distinction a threshold count cannot make. The weight is
  `grad / 0.02` capped at 8x, accumulated in fixed point, so a frame whose edges all sit exactly at
  the threshold reproduces the old count exactly and the 0.007/0.016 endpoints keep their
  calibration. No threshold was retuned.
  Result: strictly selective. Four core clips and nine extended clips are bit-identical; where it
  acts, artifacts fall 2-3x faster than the pop it gives up (c339 -29% pop for -52% shear and -65%
  stretch; c747 -19% pop for -54% shear). Nothing regressed on any axis. `spring_skeleton_cave`
  drops to 1.12, below its pre-today 1.22. **This statistic is what makes a 2.00 ceiling defensible**
  -- with the count-based classifier, 2.00 hands full pop to scenes that demonstrably cannot take it.
  Evidence: `band-120-200-*`, `riskstat-core`, `riskstat-ext`, `final-120-200-weighted`.
  **Open residual, do not lose this:** `tartanair_house_easy` did not move at all under the weighted
  statistic (23.54 -> 23.54) and remains ~2.2x its pre-today shear. Magnitude weighting was
  therefore falsified as an explanation for that clip. Since `warp_cross_row_shear_severity_pct`
  measures row-to-row displacement inconsistency, the likely cause is depth noise between adjacent
  scanlines rather than silhouette structure -- which neither an edge count nor an edge magnitude
  can observe. Diagnose that clip before tuning the classifier further.
  **Classifier thresholds recalibrated 2026-07-24, and this was the real defect.** Instrumenting the
  resolved multiplier (rather than inferring it) showed the controller was emitting its FLOOR on 19
  of 23 clips; the only clips reaching the ceiling were the three synthetic ones. Measuring the
  weighted edge fraction across the whole suite explains why: the synthetic clips sit at
  0.0001-0.0087 while real footage spans 0.038-0.245 (median ~0.10), so the old 0.007/0.016
  endpoints saturated on everything real and the band was inert. Endpoints are now 0.04/0.20,
  roughly the 10th-90th percentile of real content, and the band is genuinely exercised (clean
  depth -> near ceiling, busy -> floor, the rest spread between). Classification is also deferred
  until the depth field settles, reusing the cut detector's 8-update window, because an unsettled
  field reads smoother than the shot really is. Headset-validated. Note `bonn_person_close` barely
  moves: a close-up real person has the densest depth edges in the suite (0.24) and still sits at
  the floor, so the configuration already judged good in the headset is preserved.
  **`subject_stretch` band smoothing ACCEPTED 2026-07-24 at weight 0.18 (`STRETCH_BAND_EMA`).**
  It was the last per-frame adaptive gain in the depth domain with no temporal smoothing, while
  subject depth, convergence and the normalization min/max are all EMA'd. 0.18 matches
  `sbs_3d_minmax_ema`, which smooths the same kind of quantity -- a depth-domain range -- rather
  than the convergence EMA, which smooths an anchor.
  Consistently positive on REAL content: core-real jitter -2.0% (7 clips), extended -4.3%
  (12 clips), and -6.6% on a purpose-built 240-frame native clip. Stereo volume is flat throughout,
  so this is a pure stabilizer, not a changed depth mapping.
  **It regresses the SYNTHETIC clips (+16.7% jitter, driven by `fast_motion` 1.72 -> 3.15), and
  that was initially mistaken for a reason to reject it.** `fast_motion` is generated by
  `make_synth_clips.py` as an async-depth-ghost probe -- a textured block crossing a background at a
  known 30 px/frame -- so its depth field is a single block and its edge fraction is 0.0001. A
  synthetic single-failure-mode probe should not veto a change that helps every category of real
  content. The likely mechanism is nonetheless worth keeping: band smoothing adds temporal lag on
  top of an existing depth/color misalignment, and async depth is a PRODUCTION condition (the live
  pipeline drops frames when inference is busy), so revisit this if async-depth ghosting is ever
  chased directly.
  **Motion-adaptive weighting was properly tested and FALSIFIED -- do not retry it without a
  different signal.** `change_fraction` was instrumented and its real distribution measured across
  19 real clips: near-static content sits at 0.001-0.008 (c747, spring_daylight_path, c841, anime,
  spring_character_close, c525), mild motion at 0.03-0.10, and genuinely dynamic content at
  0.18-0.35 with per-frame peaks to 0.76. So the signal has plenty of dynamic range and the earlier
  thresholds were not obviously wrong.
  The hypothesis still fails, because **the benefit of smoothing does not decline with motion**.
  Correlating the shear improvement against measured motion: low-motion clips (<0.10) mean -5.5%,
  high-motion clips (>=0.10) mean -3.1%. If anything low-motion content benefits MORE, which is
  mechanically sensible -- a wobbling band is most visible when nothing else in the frame moves,
  and motion masks it. Blending toward unsmoothed at high motion would therefore surrender benefit
  rather than recover it. Two clips with nearly identical motion land on opposite sides
  (`tartanair_house_easy` 0.338 improves 18.3%; `sintel_ambush` 0.350 regresses 19.0%), so whatever
  separates them is not motion. `sintel_ambush` is the one real clip that regresses on shear and is
  the right place to look for a better signal.
  **Correction to the accepted-entry claim above:** "consistently positive" holds for SHEAR (8 of 12
  extended clips improved) but overstates JITTER, where the extended mean was driven almost entirely
  by `sintel_market` (-25.3%) with eight of twelve clips at exactly 0.0%. `static_jitter_p95` is
  quantized enough not to register small changes, so treat small jitter deltas on this suite as
  no-signal rather than as evidence.
  Superseded detail from the first attempt, kept because the falsifications are reusable:
  The band is still the only per-frame adaptive gain in the depth domain with no temporal smoothing
  (subject depth, convergence and the normalization min/max are all EMA'd), and the argument for
  latching it still stands. But smoothing it is not a free win. Three variants were measured:
  fixed 0.10, fixed 0.18 (matching `minmax_ema`, the analogous depth-range smoother), and a
  motion-adaptive weight blending to 1.0 on `change_fraction`. All three improve real footage
  (extended: jitter -4%, shear -7%, stretch -3%; a 240-frame native bbb clip: jitter -6.6%) and all
  three regress `fast_motion` jitter catastrophically, 1.715 -> ~3.15 (+87%). Stereo volume stays
  flat throughout, so these are pure stabilizer effects, not a changed depth mapping.
  Two hypotheses were falsified. It is NOT simple lag: nearly doubling the weight recovered almost
  nothing (3.211 -> 3.146). And it is NOT the motion signal: the motion-adaptive variant is
  bit-identical to fixed 0.18 on that clip, i.e. `change_fraction` never rises enough to engage,
  so `fast_motion`'s DEPTH field is not actually changing fast despite the clip's name. The
  mechanism is not understood; understand it before retrying.
  **Methodology note, because it nearly caused a wrong conclusion:** the 24-frame clips are shorter
  than a 0.10 EMA's convergence time (95% needs ~28 frames) and are stride-subsampled, which looks
  like it should invalidate temporal measurements. It does not. A controlled A/B on the same source
  and start -- 240 frames at native 30fps versus 24 frames at 10fps -- produced essentially the same
  verdict (jitter -6.6% vs -5.7%). Clip length and stride are not what flips temporal results here;
  content is. Do not dismiss a temporal measurement on clip-length grounds without running that
  control.
- Art3D-style shot-level zero-plane placement was screened as three scene-latched treatments:
  tracked subject, depth median, and far/mid-background (P25). Each resolves its anchor through
  the final Bestv2 curve and stores the source-pixel shift, so percentile motion cannot make
  convergence breathe between cuts. A legacy smoke run was byte-identical. No fixed anchor is a
  production win: median improved 6 primary Spring metrics with 1 cost, including character
  volume and daylight halo/stretch, but core lost fast-motion volume and added c747/c841 warp
  costs. Subject improved Spring artifact/reference fidelity but reduced character volume;
  background was the least harmful core treatment but weaker than median on Spring. Visual
  inspection confirmed convergence redistribution rather than universal detail recovery. Depth
  histograms alone are not a safe selector: scenes with similar percentiles and edge density chose
  different winners. Evidence:
  `zero-plane-{legacy,subject,median,background}-{core-screen,spring}`.
- **SUPERSEDED 2026-07-24: `sbs_3d_zero_plane` now defaults to `median`.** The headset preference
  label this decision was waiting on was collected and is decisively positive. Re-measured on the
  current metric schema (`pop-A-control` vs `zp-subject` vs `zp-median`, core suite), the earlier
  costs do not reproduce and the earlier framing understated the win:
  - **Stereo volume is unchanged on all 11 clips** (pop spread 1.008 -> 1.008). The "core lost
    fast-motion volume" cost does not reproduce: fast_motion is 1.044 -> 1.044.
  - **Mapping stretch improved on all 11 clips**, -15% to -80% (mean 0.633 -> 0.254).
  - **Static jitter improved on 10 of 11**, mean 1.816 -> 1.344. The one movement against is c747
    (+0.137), well inside its 0.5 gate threshold. c841 *improved* 31.8%, so that earlier cost does
    not reproduce either.
  - `experimental_stereo_window_crossed_burden_pct` left 0.000 for the first time (-> 0.0128),
    i.e. content finally renders in front of the screen plane at all.
  - Zero hard failures in all three arms; `exact_mapping_fold_pct` stayed 0.000.
  **Why it improves the warp, which the "anchor is a constant offset" model does not predict:** the
  explicit modes do not merely add a constant. They replace legacy's *per-frame EMA-tracked* anchor
  with a *shot-latched* one and zero the legacy convergence bias. Legacy's anchor therefore wobbles
  every frame, translating the whole disparity field slightly, and that wobble was feeding both
  stretch and jitter. **The legacy per-frame adaptive anchor is itself a source of temporal
  instability.** `subject` is a close second and remains available.
  **Note for pop re-tuning:** the ~58% stretch reduction is real headroom. Any future pop-ceiling
  experiment should be run against this baseline, not against legacy.

Do not reintroduce a removed processor without a current core and extended comparison, visual
evidence, and a headset-motivated hypothesis.

## Evaluation workflow

Use `tools/sbsbench/run_eval.py`. Every decision run must use matching model, clips, metric schema,
and profile provenance; cover the 11-clip core and public extended suites; generate `report.html`
and `decision.json`; inspect primary-axis examples; and treat comfort/image-integrity limits as hard
constraints. Headset evidence resolves coequal-axis tradeoffs.

The harness uses contract 15 and eval schema 24. It exports raw depth, pre-warp depth, exact forward
coverage diagnostics, and final SBS artifacts by numeric frame identity. Ground-truth depth scoring
is scale/shift invariant but polarity preserving. MPI Sintel true-right references additionally
score global-horizontal-registered PSNR/SSIM, local epipolar residual/coverage, and Art3D-inspired
positive-affine depth-budget/zero-plane alignment plus DDC-IoU structure. All remain diagnostic
pending headset correlation. Spring adds four 1920x1080 cinematic true-stereo sequences and complete
artistic-style evidence. Repetitive-texture phase-correlation candidates must also improve
non-wrapping photometric alignment before they may contribute to signed-disparity comfort gates.
Style summaries require complete evidence on every frame so failed fits cannot improve an average;
comfort and integrity remain hard gates.

## Current priorities

1. Correlate the new previous-only GT ghost-edge diagnostic with additional known-motion scenes
   and headset evidence before allowing it to become a primary gate.
2. Re-tune the adaptive-pop band, in this order. The 1.25-1.30 band is not defensible as a
   validated result in its current form: the shader receives a RATIO of
   `adaptive_pop_max / pop_strength` = **1.04**, which is below the `rel_tol` noise floor (0.15-0.20)
   of every metric that could judge it. A controller with 4% authority can neither be validated nor
   falsified, which is why `spring-adaptive-vs-fixed130` came back neutral. Either widen it enough
   to matter or delete it.
   1. **Decouple probe geometry from strength first.** `Bestv2SearchRadius` scales with
      `adaptive_pop_max`, but `Bestv2ProbeSteps` is a fixed `24 * aspect_scale`, so probe spacing
      `2R/steps` is proportional to strength. Raising the ceiling therefore coarsens the probe for
      **every** scene, including scenes the controller declined to boost -- they pay the cost and
      receive none of the gain. Any ceiling A/B run before this fix is partly measuring its own
      probe degradation, which is a strong candidate for why 1.35 and 2.0 both failed.
   2. Only then run the ceiling ladder, against the `median` zero-plane baseline rather than
      legacy, since that change bought roughly 58% of mapping-stretch headroom.
   3. Read the diagnostics by hand. The gate cannot answer this question -- see the caveat on the
      adaptive-pop entry above.
   Separately, a second-order finding worth acting on: `Bestv2SearchRadius` is bounded for a worst
   case of one extreme band minus an oppositely signed subject band, but with the shipped
   `subject_lock = 0.5` anchor the reachable displacement is well under that bound, so a large
   fraction of every probe falls where no root can exist. Reclaiming it would raise probe density
   at identical cost. That changes behaviour today and needs its own measured A/B, unlike the
   decoupling in (i) which is a no-op at the shipped configuration.
3. ~~Collect scene-level headset labels for explicit zero-plane placement.~~ **Done 2026-07-24:
   `median` was labelled decisively better in the headset and is now the default** (see the
   superseding entry above). The remaining open work is the *per-scene selector*, not the global
   default: a learned/semantic selector may use image content and depth together, but do not ship
   a percentile-only rule from the current small suite.

## References

- `tools/sbsbench/README.md` — build, evaluation, report, and dataset commands.
- `docs/sbs-feature-decision-revisit.md` — historical accepted/rejected evidence.
- `docs/sbs-resolution-robustness.md` — coordinate-space and encoder-resolution audit.
- `src/video_depth_estimator.cpp` — depth normalization and subject state.
- `src_assets/windows/assets/shaders/directx/sbs_reprojection_ps.hlsl` — production geometry.
