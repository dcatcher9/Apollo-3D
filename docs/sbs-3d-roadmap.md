# SBS 3D — current status and roadmap

Apollo converts captured mono frames into host-rendered SBS with TensorRT depth estimation and one
production live geometry implementation: Host SBS Depth Coordinate V2. The V1 occlusion-aware
backward-probe renderer and its live selector have been removed. V2 consumes the least anisotropic
2D near-preserving majorant of the signed candidate field: a vertical shear-2 column majorant
followed by the slope-0.5 row majorant. Neither pass lowers requested disparity; the final field
bounds crown shear and horizontal slope and is inverted with a unique 12-step contractive fixed
point. Candidate, vertical intermediate, and canonical coordinate remain diagnostics only; V2 has
no forward-owner, visibility-selection, or synthetic-fill path.

Production contract schema 14 authenticates DAV2 Small, its HDR/SDR preprocessing closure, the
seven-pass producer, the standalone live renderer, and six standard tensor shapes (`770x434`,
`1022x434`, `1036x434`, `434x770`, `434x1022`, and `434x1036`). DAV2 Base, custom models, and custom
tensor shapes fail flat. There is no V1 fallback. See
`docs/host-sbs-depth-coordinate-v2.md` for the complete fail-closed contract.

The legacy analysis stack remains live only as the temporary producer of the confirmed scene-cut
`{generation, pulse}` bridge consumed by V2. Its normalized depth, subject, adaptive-pop,
zero-plane, and warp outputs have no live geometry authority. Legacy renderer/shader sources remain
in the tree for offline conversion and evaluation harnesses; their presence does not make V1
selectable in Host SBS.

Approved AR glasses connected as a Windows monitor also use an automatic local presenter; see
`docs/sbs-local-ar-glasses.md`. That path reuses the production depth and warp without NVENC.

## Shipping pipeline

1. Preserve source aspect while selecting one of the six authenticated patch-aligned tensor grids.
2. Convert SDR or HDR capture into the authenticated DAV2 Small input domain.
3. Infer raw depth and obtain only the confirmed cut epoch from the temporary legacy-analysis
   bridge.
4. Acquire or retain the scene center and near-tail shoulder, then apply the fixed raw scale,
   monotone asymmetric curve, requested pop, and exact frame-local 4% source-U container.
5. Produce the vertical shear-2 majorant and then the slope-0.5 row majorant entirely on the GPU.
6. Render each eye with the unique 12-step contractive inverse. Invalid or unauthenticated current
   geometry renders flat rather than reusing stale depth or falling back to V1.
7. Convert the packed SBS raster directly to the encoder format. If doubled width exceeds
   `sbs_3d_max_encode_width` or the selected codec's runtime `NV_ENC_CAPS_WIDTH_MAX`, preserve each
   eye's aspect while scaling to the lower cap. The configured production ceiling is 8192 packed
   pixels. The production RTX 5080 reports 4096 for H.264 and 8192 for HEVC/AV1, so a 3840x2160
   source becomes 4096x1152 in H.264 but remains 7680x2160 in HEVC/AV1. Apollo records the
   authoritative per-codec capability during NVENC probing and refreshes it at encoder creation.

`sbs_3d_pop_strength` is the live user geometry control and V2 takes it literally. Legacy adaptive
pop, subject stretch/recenter, min/max normalization, and zero-plane controls do not modify live
V2 geometry. Standard stream resolutions choose among the authenticated landscape, ultrawide, and
portrait tensor shapes automatically; a configuration override cannot authorize an unlisted shape
or different model calibration.

## Archived V1 processor decisions

The entries below record the pre-cutover V1 evaluation history. They remain useful evidence for
offline/evaluator compatibility and for avoiding repeated experiments, but they do not describe
the shipped live renderer or live configuration authority.

Bestv2 disparity is calibrated at the evaluator's 854-pixel source width and normalized to the
5120x2160 Artemis reference aspect. `sbs_3d_pop_strength` scales the final parallax (`0.25`–`2`,
default floor `1.20`) without changing that resolution correction. The accepted scene latch may
select up to `sbs_3d_adaptive_pop_max` (default `2.00`) from depth-edge risk and holds the
selection until a hard cut.

### Frozen processor decisions

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
  scale and (then-present) safety bound once per output pixel. This removes repeated invariant
  arithmetic from every probe and reduced warp time by a further 3.38% on core and 3.19% on
  extended, again with no primary-axis regression or hard failure. Evidence: `parallax-invariants-core` and
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
- The later default-off Depth Coordinate V2 experiment exposed a different failure in the
  fullscreen hair dumps. An unbounded candidate cliff produced a broad translucent ramp; rejecting
  interpolation and filling uncovered pixels produced obvious horizontal comb/teeth. Full-
  resolution forward ownership did not solve it: exact attribution put every visible difference
  inside synthetic-fill pixels. Those owner/fill variants are rejected and removed from live V2.
  The first targeted treatment used a greatest row-wise minorant and removed the bands by lowering
  foreground disparity, but the fullscreen hair/body contour visibly separated. It is superseded.
  The first no-fill treatment computed the least row-wise majorant `q >= p` with
  `|dq/dx| <= 0.5`, then used the same 12-step contractive inverse. Exact pop-1.5 replay removes the
  duplicate and hair/body step without lowering any candidate value; 3.837% of fullscreen texels
  and 2.152% of windowed texels are raised. Its residual background bending around the fullscreen
  crown selected a pure vertical shear-2 majorant before the row pass. The final is the least
  anisotropic 2D majorant and preserves `q >= vertical >= candidate`. The source captures are
  historical schema-7 input witnesses and cannot authenticate the shipped schema-14 path. Current
  qualification therefore uses fresh schema-14/manifest-schema-7 live dumps and timing. Evidence:
  `E:\ApolloDev\majorant-row-both-confirm-20260802`.
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
  weighted edge fraction across the whole suite explains why: the three stable-shot synthetic
  probes used for calibration sit at 0.0001-0.0087, while the remaining non-probe measurements
  span 0.038-0.245 (median ~0.10). Those measurements cover the declared `real-capture`,
  `animation`, `simulation`, `ai-generated`, `anime` and `unclassified` categories; they are not
  one uniform source class. The old 0.007/0.016 endpoints saturated across that set and left the
  band inert.
  Endpoints are now 0.04/0.20, roughly its 10th-90th percentile, and the band is genuinely
  exercised (clean depth -> near ceiling, busy -> floor, the rest spread between). Classification
  is also deferred
  until the depth field settles, reusing the cut detector's 8-update window, because an unsettled
  field reads smoother than the shot really is. Headset-validated. Note `bonn_person_close` barely
  moves: this declared real-capture close-up has the densest depth edges in the suite (0.24) and
  still sits at the floor, so the configuration already judged good in the headset is preserved.
  **`subject_stretch` band smoothing ACCEPTED 2026-07-24 at weight 0.18 (`STRETCH_BAND_EMA`).**
  It was the last per-frame adaptive gain in the depth domain with no temporal smoothing, while
  subject depth, convergence and the normalization min/max are all EMA'd. 0.18 matches
  `sbs_3d_minmax_ema`, which smooths the same kind of quantity -- a depth-domain range -- rather
  than the convergence EMA, which smooths an anchor.
  Measured aggregate jitter changes were -2.0% on the historical seven-clip non-synthetic core
  grouping, -4.3% on the mixed-content public extended suite, and -6.6% on a purpose-built
  240-frame native clip. Stereo volume is flat throughout, so this is a pure stabilizer, not a
  changed depth mapping.
  **It regresses the SYNTHETIC clips (+16.7% jitter, driven by `fast_motion` 1.72 -> 3.15), and
  that was initially mistaken for a reason to reject it.** `fast_motion` is generated by
  `make_synth_clips.py` as an async-depth-ghost probe -- a textured block crossing a background at a
  known 30 px/frame -- so its depth field is a single block and its edge fraction is 0.0001. A
  synthetic single-failure-mode probe should not by itself veto a change that improves the tested
  non-probe aggregates. The likely mechanism is nonetheless worth keeping: band smoothing adds
  temporal lag on
  top of an existing depth/color misalignment, and async depth is a PRODUCTION condition (the live
  pipeline drops frames when inference is busy), so revisit this if async-depth ghosting is ever
  chased directly.
  **Motion-adaptive weighting was properly tested and FALSIFIED -- do not retry it without a
  different signal.** `change_fraction` was instrumented and its real distribution measured across
  19 non-probe clips: near-static content sits at 0.001-0.008 (c747, spring_daylight_path, c841, anime,
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
  separates them is not motion. The `animation` clip `sintel_ambush` regresses on shear, but visual
  inspection says there is nothing there to find -- see the metric caveat below.
  **`warp_cross_row_shear_severity_pct` moves double digits for sub-perceptual differences --
  weigh it accordingly.** Inspected twice, both times the same result. On `sintel_ambush` frame 14
  (the worst shear frame of the worst-regressing clip) the band-EMA and no-band-EMA renders differ
  by 149 pixels out of ~819,000 (0.018%), mean difference 0.08 gray levels, and are
  indistinguishable at 5x magnification on the exact region of maximum disagreement -- yet the
  metric reports +19.0%. On `tartanair_house_easy` a 23.5 reading (2.2x its pre-change value)
  showed no visible tear at 8x magnification either. The metric scores the displacement MAP, not
  pixels, so it responds strongly to coordinate differences that produce no visible change,
  particularly on smooth-ramp or low-texture content. It is `role: diagnostic` and
  `label_status: experimental` for good reason. Do not let a shear delta of a few percent decide a
  tradeoff, and inspect before treating a large one as a defect.
  **Correction to the aggregate summary above:** SHEAR improved on 8 of 12 extended clips, but the
  JITTER mean was driven almost entirely by `sintel_market` (-25.3%) with eight of twelve clips at
  exactly 0.0%. `static_jitter_p95` is
  quantized enough not to register small changes, so treat small jitter deltas on this suite as
  no-signal rather than as evidence.
  Superseded detail from the first attempt, kept because the falsifications are reusable:
  The band is still the only per-frame adaptive gain in the depth domain with no temporal smoothing
  (subject depth, convergence and the normalization min/max are all EMA'd), and the argument for
  latching it still stands. But smoothing it is not a free win. Three variants were measured:
  fixed 0.10, fixed 0.18 (matching `minmax_ema`, the analogous depth-range smoother), and a
  motion-adaptive weight blending to 1.0 on `change_fraction`. All three improve the mixed-content
  public extended aggregate (jitter -4%, shear -7%, stretch -3%) and a 240-frame native bbb clip
  (jitter -6.6%), while all three regress `fast_motion` jitter catastrophically, 1.715 -> ~3.15
  (+87%). Stereo volume stays flat throughout, so these are pure stabilizer effects, not a changed
  depth mapping.
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
- **The shot-latched zero-plane anchor is now resolved TWICE per shot (2026-07-25).** It was
  latched on the cut frame itself, which is the same defect fixed earlier for the pop classifier
  thirty lines away in the same shader: normalization settling perturbs 50-60% of depth texels on
  the first frames, and `lo_val`/`inv_range`/`delta` feeding the anchor are raw cut-frame values
  too (the stretch band's EMA resets on a cut by design). It matters more here than for pop,
  because the point of a shot-latched plane is that it does not move, so a bad latch is
  unrecoverable until the next cut.
  The fix resolves once immediately -- so a new shot never renders on the PREVIOUS shot's plane --
  and once more at `scene_age == POP_CLASSIFY_SETTLE_FRAMES`. Extended jitter -4.2%, extended
  stretch -0.7%, core unchanged to four significant figures, `scene_cut` unchanged at 4.900, and
  `experimental_stereo_window_crossed_burden_pct` +12% (more content in front of the screen plane,
  which is the axis this line of work exists to move). `exact_visible_pop_spread_pct` is
  offset-invariant and reads 0.0000% as expected.
  **Continuously re-resolving through the settle window was tried first and is WORSE -- do not
  retry it.** The intent was to avoid a visible convergence step by letting the plane drift and
  then lock. It converts one correction into ~8 frames of motion, and `scene_cut` -- the clip built
  to probe normalization swim across cuts -- regressed 4.900 -> 8.190 on `static_jitter_p95` (+67%),
  driving a +17.8% core mean. Resolve twice; do not track.
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

- Pre-fusing the depth model's expanded exact-GELU pattern into a single `Gelu` node was screened
  against the Artemis client's equivalent rewrite and rejected as a no-op for the host. The client
  gained 15.7% (22.20 -> 18.72 ms) by collapsing the per-block `Div/Erf/Add/Mul/Mul` DAG, because
  its runtime dispatches one kernel per op. TensorRT does not: verbose build logs show the parser
  reading all 12 `Erf` chains, then `After Myelin optimization: 1 layers` — the whole ViT becomes a
  single `ForeignNode[graph_input_cast0...graph_output_cast0]` with generated fused kernels, so the
  expanded form never costs extra dispatches. A fused ONNX (838 -> 790 nodes, 12 `Gelu`, 0 `Erf`;
  TensorRT's parser accepts `Gelu` at opset 14 even though the ONNX checker does not) reached the
  *same* single Myelin layer and byte-identical 49,324,544 B activation memory. `trtexec` at the
  production `--builderOptimizationLevel=5`, 400 iterations at 1x3x434x770, measured median GPU
  compute 1.54077 ms baseline vs 1.53125 ms fused: **0.62%**, against a p50-p90 spread of 0.003 ms.
  Do not port client model-graph fusions to the host without first checking whether Myelin already
  subsumes them.

- `depth_short_side` stays at 434, decided on **stereo volume**, not on GT edge agreement. Matched
  ladders were run on both suites at 57abea70 (identical executable and shader shas, only
  `--depth-short-side` differing): core `dss-434/392/336/280`, extended `dssx-434/392/336`. Note a
  matched same-sha control did not previously exist on either suite; comparing against an older
  baseline is what produced the earlier misleading read.
  - **The two suites agree on every metric they share.** `exact_visible_pop_spread_pct` at 392 is
    -1.58% on core and -1.74% on extended, and degrades monotonically on core: -1.58% / -3.58% /
    -7.88% at 392 / 336 / 280. Restricted to the five c*-named core clips in the legacy grouping
    it is -0.46% / -3.17% / -5.00%. That subset is now classified as two AI-generated and three
    unclassified clips, not real captures; the separately classified extended suite corroborates
    the same volume-loss direction. Everything else is flat and `exact_mapping_fold_pct` stays
    0.000. 434 is kept because that loss buys only 6.83% of `depth_infer` (1.512 -> 1.409 ms) —
    0.10 ms, which changes no frame budget.
  - **Lower depth resolution does not cost temporal stability on the historical non-probe
    evidence.** `static_jitter_p95`
    rises +7.61% / +22.26% / +47.33% across the whole core suite, but that is entirely synthetic:
    the seven clips historically grouped as non-synthetic move **-3.35% at 280** (i.e. jitter
    IMPROVES) while the four constructed probes move +70.98%, driven by `fast_motion` alone going
    3.145 -> 11.058. Quote the split, never the suite mean. The historical -3.35% subset includes
    three clips now marked unclassified, so it is not silently reused as the current aggregate.
    `tools/sbsbench/compare_runs.py` now prints every content type, marks both synthetic probes and
    unclassified content NOT DECISIVE, and reports `ALL CLASSIFIED NON-PROBE`. The four probe clips
    carry `content_type: synthetic`; c339 and c647 are `ai-generated`; c525/c747/c841 are explicitly
    `unclassified` rather than have provenance invented for them.
  - **`depth_gt_edge_f1` is not a usable discriminator at this scale and must be read as
    diagnostic.** It does not track actual depth change: between 434 and 392, tartanair_house_motion
    changed *more* (3-frame-mean affine-aligned depth NRMSE 4.12%) yet scored -0.08%, while
    tartanair_house_easy changed *less* (3.20%) yet scored -49.8%. Both vkitti clips changed ~0.04%
    and scored ~0.00%. The suite mean of -8.26% is produced entirely by that one clip; the other
    seven average +3.4%. Visual inspection of the depth maps at all three resolutions confirms
    neither the -49.8% collapse nor the +18.8% best case is apparent. house_easy is a low-texture
    indoor scene whose edge content swings across frames (0.34% of pixels at f8 vs 3.83% at f20),
    and `depth_gt_edge_f1` is the only metric in schema 32 exporting no support count, so nothing
    gates it on edge population. `depth_gt_polarity_ok` held at 100.0 everywhere.
- Depth input shapes must avoid the TensorRT tactic cliff just above a 128-token boundary. Measured
  with `trtexec` at `--builderOptimizationLevel=5`, 300 iterations: 1675 tokens 1.499 ms, 1706
  (production 16:9 770x434) 1.509, 1737 1.521, 1761 1.510, then **1793 tokens 1.988 ms and 1825
  1.982** — a +31% step, not a token-count effect, since 1761 -> 1793 is only +1.8% tokens. 1792 is
  14x128, so the DINOv2 CLS token is what pushes the sequence into an extra, badly-utilized tile.
  Consequence: do not raise `depth_short_side` to 448, which `aspect_aligned_dims` turns into
  798x448 = 1825 tokens for 16:9 and lands past the cliff. Current production shapes are all clear
  of it: 16:9 770x434 = 1706, 5K2K 994x420 = 2131, 21:9 1008x420 = 2161. Cost is also sub-linear in
  tokens below the cliff (-1.8% tokens bought only -0.67% latency), so shaving tokens is a poor
  perf lever regardless.

- **Cardboarding from hard band clamping — measured, fixed offline, awaiting headset arbitration.**
  `Bestv2WarpDepth` applied `saturate()` twice, and since `DepthParallax` is a pure function of
  shaped depth plus frame-uniform scalars, every clipped pixel received an *identical* disparity:
  a flat plateau with no relief. Measured from the exported backward maps (mass sitting exactly at
  a displacement extreme), the plateau covered **15.84% of pixels on extended and 22.57% on core**,
  reaching 68.24% on `c525` — the previously recorded "~10%" was an underestimate. It is not
  legitimate far-plane background: every plateau swallowed 4-46% of the frame's depth range.
  `clamp_abs` was ruled out as the cause — at `0.071 * aspect_scale` it is ~72 px on a 1024-wide
  eye while measured displacements peak at 3.7-8.3 px, so it never binds.
  - **Shipped fix: fold the two clamps into one and WIDEN the band, `STRETCH_BAND_TAIL` 0.05 ->
    0.02 (P5/P95 -> P2/P98).** Folding is free and strictly better (`saturate(saturate(x) + a)` is
    not `saturate(x + a)`; the inner clamp collapses every `x < 0` onto the constant `a` while the
    folded form still ramps over `[-a, 0]`) but on its own only relocates the clipped window, whose
    width is 1 either way. Reducing the clipped FRACTION is what removes cardboarding.
  - **A soft-knee variant was implemented, measured and REJECTED — do not retry it.** Softening the
    band edge keeps exactly the same over-clipping and charges the band interior for it. Measured
    from the backward maps against identical percentiles: the knee cost **6.3-6.4% of p10-p90
    mid-scene relief** and 4.08%/3.65% of gated stereo volume, versus ~0 for widening. The volume
    loss was the knee moving the reachable disparity endpoints inward, NOT a consequence of
    de-saturating. If cardboarding needs more work, widen further or attack the EMA lag — never
    reintroduce a knee in `Bestv2WarpDepth`.
  - Result (`wideband-{core,ext}` vs `dss-434`/`dssx-434`): plateau **22.57% -> 16.50% core and
    15.84% -> 9.21% extended**, `exact_mapping_stretch_pct` -8.11%/-4.57%, `static_jitter_p95`
    -1.80%/-4.03%, `vmisalign_p99_pct` -0.72%/-6.39%, fold 0.000 and image integrity 100.0, with
    `exact_visible_pop_spread_pct` **-1.05% core / +0.00% extended** — no stereo-volume cost.
    (That metric is `role: diagnostic`, not primary; it is quoted here because it is the axis
    the soft-knee variant lost, not because it gates.)
  - The residual plateau is NOT percentile-driven: P1/P99 (`wide99-*`) moved it only 9.21 -> 8.71
    extended and 16.50 -> 16.23 core for no further gain.
  - **Both EMA'd ranges are now attack-fast/release-slow envelopes** (`envfollow-*`). There were two
    lagging ranges, not one, and both saturate downstream: the normalization P2/P98 in
    `depth_minmax_ema_cs` (clipped by `buffer_to_tex_cs`) and the stretch band in
    `depth_subject_resolve_cs` (clipped by `Bestv2WarpDepth`). A symmetric EMA lags the live
    percentiles, and any frame whose smoothed range is narrower than the live one clips the
    difference — lag becomes clipped depth. Each now expands immediately to cover the live
    percentiles and contracts at its original alpha. Expansion is also the stability-safe direction:
    the range is a multiplicative gain, so growing it LOWERS the gain, and fast shrinking is what
    makes the mapping breathe. The band is additionally smoothed in (lo, hi) space rather than on
    the reciprocal `inv_range`. Plateau 9.21 -> 8.11 extended and 16.50 -> 15.77 core, with core
    `static_jitter_p95` -2.75% (it did not regress, contrary to the stability concern), pop
    +0.08%/-0.49%, fold 0.000 and integrity 100.0.
  - **Percentile bounds now take the crossing bin's OUTER EDGE, not its center** (both stages,
    `binedge-*`). A percentile only excludes its nominal fraction when the distribution is smooth
    across the crossing bin; when a large atom sits there a centered bound cuts through it. c525
    puts 66.1% of its pixels in histogram bin 0. An outer edge can only widen the range, costing at
    most one bin (~0.4%) of precision. Plateau 8.11 -> 7.62 extended and 15.77 -> 15.15 core, with
    stretch -0.25%/-1.00%, jitter -0.05%/-0.63%, shear -1.68%/-0.83%, pop -0.02%/-0.10%, fold 0.000.
  - **Correction: the plateau figure overstates cardboarding, and removing a normalization stage is
    NOT warranted.** Separating the near and far plateaus and testing each against `raw_*.f32`
    (pre-normalization model output) shows the two behave completely differently. The LARGE near
    plateau is legitimate: it swallows ~0% of the raw depth spread (c525 50.78% of pixels / 0.0%
    spread, vkitti_drive_rain 19.81% / 0.0%, c339 17.74% / 0.0%), i.e. the model genuinely assigns
    those pixels one depth and rendering them on one plane is correct. Only the FAR plateau destroys
    real range, and it is small — mean ~4% of pixels, though it reaches 76.9% of the raw range on
    c525 and 73.4% on spring_character_close. Earlier entries here quoted the pooled near+far
    figure as if all of it were destroyed relief; it is not. Any future measurement MUST separate
    the two plateaus and validate against raw depth, or it will chase legitimate flat geometry.
  - **The far plateau is closed as a warp concern.** Measured with `exact_disparity_plateau_far_*`
    after the band, envelope and bin-edge fixes, no clip on either suite shows destroyed relief:
    the largest fractions are entirely flat content (c525 53.27% at span 0.0%, vkitti_drive_rain
    19.89% at 0.0%, c339 18.60% at 0.0%) and the widest spans are small (bonn_person_close 6.1%,
    c647 4.4%, sintel_ambush 3.8%). The warp is faithfully rendering what it receives.
    **Caveat, and it bounds the claim:** that metric samples PROCESSED depth, which the stage-1
    normalization has already clipped, so it cannot see relief flattened before the warp. The
    earlier raw-referenced measurement put c525's far plateau at 76.9% of the RAW range. So the
    remaining question belongs to the depth normalization, not the warp, and needs a
    `raw_*.f32`-referenced measure to answer.
  - The two normalization stages serve different purposes (stage 1 gives temporal scale stability
    against DA-V2's per-frame scale drift; stage 2 allocates the parallax budget to the mid-range)
    and the evidence does not support collapsing them.
  - `STRETCH_BAND_EMA`'s `fast_motion` regression (jitter 1.72 -> 3.15) is NOT chased. That clip is
    now annotated `content_type: synthetic` and `compare_runs.py` marks it NOT DECISIVE, so it can
    no longer contaminate a suite mean. The original hypothesis stands — band smoothing compounds
    an existing depth/color temporal misalignment that the clip exists to expose — and it should be
    revisited only if async-depth ghosting is chased with direct live-stream evidence rather than
    inferred from this synthetic probe.
  - `depth_subject_resolve_cs` duplicates the shaping for the zero anchor rather than calling
    `Bestv2WarpDepth`. Shaping it differently makes the anchor describe a different plane than the
    warp renders — during this work an inconsistent version cost core `vmisalign_p99_pct` +8.76%.
    **Any change to the shaping function must be mirrored there.**
  - **Both remaining clamps on the per-pixel path were proved unreachable and removed.**
    (a) `Bestv2RawShiftPxFast`'s input `saturate(d)`: every call site already delivers [0,1] —
    `Bestv2WarpDepth` returns `saturate(...)` on its shaped path, `depth_subject_resolve_cs`
    saturates before calling, and the one unshaped passthrough (`sbs_forward_coverage_cs`) never
    reaches it because `DepthParallax` sits inside `if (shaped)` and the raw value feeds
    `saturate(shaped_depth)` instead. (b) `DepthParallax`'s `clamp_abs`: reach is
    `9.979 * (0.35/854) * strength * aspect_scale` against a bound of `0.071 * aspect_scale`, so
    **aspect cancels** and binding needs `strength > 17.36`, while `config.cpp:665-667` validates
    both `pop_strength` and `adaptive_pop_max` into [0.25, 2.0]. The 8.7x margin is therefore
    enforced, not incidental. Its coupled `min(reach, clamp_abs)` in `Bestv2SearchRadius` went with
    it. Note the fit domain is unaffected — shaped depth is still exactly [0,1].
  - Removal is metric-identical but **not** byte-identical (`noclamp-*` vs `wideband-*`): all nine
    reported metrics match to five decimals on all 23 clips, worst clip 0.000%, yet every artifact
    hash differs. Dropping a clamp lets the compiler fuse and schedule the multiply differently, so
    the warp map moves at float LSB level — the same class of trap as the probe-lattice
    accumulator. Warp time fell 1.27%/1.25%, consistent across both suites but inside the ~3% noise
    envelope that the untouched `depth_infer` stage showed in the same runs (+0.49%/+2.89%), so
    treat the perf gain as unproven and the simplification as the actual justification.
  - **`exact_visible_pop_spread_pct` rewards clipping and must not be the sole gate here.** It is a
    p0.5..p99.5 spread, so pinning 16-23% of pixels onto the two extreme disparities places those
    percentiles inside the plateaus and reports the clip bounds rather than scene relief. Read it
    alongside a plateau-excluded relief measure when judging any band change.

- **The depth profile bound is 1036, not 1008, so ultrawide reaches the configured short side.**
  `aspect_aligned_dims` is capped by the TensorRT profile kMAX, and at 1008 both production
  ultrawide cases lost a patch row: 5K2K resolved to 994x420 and 21:9 to 1008x420 rather than a
  434 short side. `models::depth_engine_max_dim` now carries the bound (1036 = 74 patches) and
  the recipe tag moved to `trt-opt770x434-max1036-level5-v3` — the tag MUST change with the
  profile or the cached engine is silently reused, since the filename encodes only opt shape
  and builder level.
  - Controlled on the extended suite (`uncap-ext` vs `rebase2-ext`): exactly the four ultrawide
    clips changed shape — sintel 980x420 -> 1022x434 (2101 -> 2264 tokens) and vkitti
    980x294 -> 1022x308 (1471 -> 1607) — while the other eight kept their tensors.
  - On those four, `depth_gt_affine_nrmse_pct` improved **-4.74%**, and that is trustworthy:
    the eight unchanged-shape clips drifted only 0.003% on the same metric, so the engine
    rebuild contributes essentially nothing to it. Stereo volume was neutral (+0.05%).
    **Gated run passes** against the committed baselines.
  - Two apparent regressions need the control to read correctly. `vmisalign_p99_pct` +12.09% is
    not credible — the unchanged-shape clips drifted 9.14% worst-case from the engine rebuild
    alone, on absolute values of ~0.02%. `static_jitter_p95` +5.88% is larger than its control
    drift (0.40% mean, 1.77% worst) and is probably real; it is the same accuracy-versus-
    temporal-stability tension the depth-resolution ladder found, and it stays inside the
    metric's 0.20 rel_tol. Worth headset attention on ultrawide content.
  - Cost is confined to ultrawide: `depth_infer` +9.92% on those clips (1.853 -> 2.037 ms).
    16:9 tensors are unchanged and pay nothing.

- **The `legacy` zero plane is removed, and with it every processor only it used.** `median` has
  been the validated default since 2026-07-24 and `legacy`'s per-frame anchor wobble was already
  documented as a source of stretch and jitter. Removing the mode deletes: the `convergence_bias`
  (identically 0 under every explicit plane), the convergence EMA that fed it, the
  `explicit_zero_plane` and `zero_anchor_shift_px` params, and `sbs_3d_subject_lock`
  — which only ever scaled the legacy anchor and had therefore been **inert under every shipped
  configuration**, while `--subject-lock` was still advertised as an A/B lever that did nothing.
  - The per-output-pixel cost went with it. `MakeBestv2Params` used to run `Bestv2WarpDepth` plus
    the degree-7 shift polynomial unconditionally to build the legacy anchor, then discard it on
    the explicit path — about 16.6M wasted evaluations per frame on a 7680x2160 target, and an HLSL
    ternary would not have avoided it because both sides evaluate.
  - The zero anchor is now resolved unconditionally. It was previously skipped for `legacy` and for
    a degenerate histogram, with the warp falling back to the subject-anchor path; the percentile
    defaults (median 0.5, background 0.25, lo 0 / range 1) give a sane plane with no histogram, so
    nothing is left to branch on and the fallback is unreachable rather than merely unused.
  - **Behaviourally a no-op, as predicted**: every reported metric is identical to five decimals on
    both suites with worst per-clip drift 0.0000%, and both gated runs pass. Artifacts are not
    byte-identical — removing code lets the compiler reschedule floats, the same effect seen when
    the dead clamps went. Warp time -2.15% core / -2.39% extended. Evidence: `nolegacy-{core,ext}`.
  - `zero_plane_mode` keeps its numbering (1 subject, 2 median, 3 background); 0 is simply never
    emitted. Do not renumber — the shader selects on `< 1.5` / `< 2.5` thresholds.

Do not reintroduce a removed processor without a current core and extended comparison, visual
evidence, and a headset-motivated hypothesis.

## Evaluation workflow

Use `tools/sbsbench/run_eval.py`. Every decision run must use matching model, clips, metric schema,
and profile provenance; cover the 11-clip core and public extended suites; generate `report.html`
and `decision.json`; inspect primary-axis examples; and treat comfort/image-integrity limits as hard
constraints. Headset evidence resolves coequal-axis tradeoffs.

The harness uses contract 18 and eval schema 36. It exports raw depth, pre-warp depth, exact forward
coverage diagnostics, and final SBS artifacts by numeric frame identity. Ground-truth depth scoring
is scale/shift invariant but polarity preserving. MPI Sintel true-right references additionally
score global-horizontal-registered PSNR/SSIM, local epipolar residual/coverage, and Art3D-inspired
positive-affine depth-budget/zero-plane alignment plus DDC-IoU structure. All remain diagnostic
pending headset correlation. Spring adds four 1920x1080 cinematic true-stereo sequences and complete
artistic-style evidence. Repetitive-texture phase-correlation candidates must also improve
non-wrapping photometric alignment before they may contribute to signed-disparity comfort gates.
Style summaries require complete evidence on every frame so failed fits cannot improve an average;
comfort and integrity remain hard gates.

## Archived V1 priorities and completed investigations

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
   **DONE 2026-07-24, and the outcome inverted the expected benefit.** The radius bound was
   re-derived from the curve (`Bestv2RawShiftPxFast` spans [-1.3964, +8.5823] over a domain both
   the stretch and recenter stages saturate, so with `subject_lock` configurable in [0,1] the
   reachable |shift - anchor| maxes at 9.979 px) and tightened 12.51 -> 10.1. The previous value's
   stated derivation cited a curve minimum of -2.52 that the polynomial does not have; it evidently
   predated a curve change.
   Note the roadmap's original premise here was also wrong: it assumed the slack came from
   `subject_lock = 0.5`, but under the `median` zero plane -- the default since that change -- the
   anchor is `s2.x` and is NOT scaled by `subject_lock`, so its worst case is the full 9.979 px. The
   number coincides, so the bound is safe, but the reasoning did not survive the zero-plane change.
   **Tightening the radius bought nothing** (pop/jitter/fold unchanged, stretch +0.5%): 20% finer
   probe spacing made no difference, i.e. the useful search interval was already over-resolved.
   That null result is what made the real win visible -- spend the reclaimed radius on FEWER probes
   at the ORIGINAL spacing instead of denser ones. `Bestv2ProbeSteps` 24 -> 19.4 (= 24 x 10.1/12.51)
   reproduces the historical spacing at ~20% less loop work: **warp time -16.0% core / -14.5%
   extended, with stretch -1.7%/-0.4%, stereo volume bit-identical, fold 0.000 and jitter
   unchanged.** The calibration pinned probe SPACING; the step count was only ever the number
   needed to achieve it across an oversized radius.
   **DONE 2026-07-25, byte-exact on both suites.** All three predicted slacks were real:
   (a) `convergence_bias` is identically 0 under the `median` default yet was ~49% of the radius,
   (b) the shift bound was the global worst case rather than the frame's, and (c) the radius used
   `adaptive_pop_max` rather than the resolved ratio -- the claim that the ratio was "unavailable
   to this loop-bound helper" was simply false. `Bestv2SearchRadius` now takes the `Bestv2Params`
   the mapping already built and returns
   `1.10 * min(clamp_abs, max|(S(d) - anchor) * parallax_scale + convergence_bias| * output_scale)`
   (the `clamp_abs` term was later proved unreachable and removed; see the cardboarding entry),
   which is exact rather than conservative because `S`'s extrema over its saturated [0,1] domain
   are its endpoints and everything else is frame-uniform.
   **Mean probes per output pixel 42.7 -> 16.0 (core) and 45.0 -> 15.1 (extended); warp time
   0.0668 -> 0.0361 ms (-45.9%) and 0.1460 -> 0.0742 ms (-49.2%).** Stereo volume, jitter and the
   disparity tails are bit-identical, fold stayed 0.000, mapping stretch moved +0.66%/+0.13% and
   cross-row shear -0.14%/-0.77%.
   **What made it validatable, and this is the reusable part.** Probes used to sit at
   `uv.x +- i*step`, i.e. relative to the output pixel, so narrowing the window MOVED every probe
   and no subset argument existed. They now sit on a global lattice `k * spacing` with `spacing`
   an explicit calibrated constant (`2 * 1.30 / 19.4 * <legacy radius geometry>`), which is what
   `BESTV2_CALIBRATED_STRENGTH` and the step count's strength renormalization were secretly
   maintaining -- both collapsed into it. A narrowed window is then a strict subset at bit-identical
   positions, every root is provably inside both windows, and a bracket outside the root region
   cannot change sign, so the two runs MUST agree bit-for-bit. Gate: `sbs_*`, `depth_*`,
   `warp_map_*.f32`, `warp_mask_*` and `raw_*` byte-compared, 1320 core + 1440 extended files,
   zero differences. That gate is the whole point -- three of the four baseline-gated metrics are
   `depth_gt_*` and are structurally blind to a warp change.
   **The float trap that nearly invalidated it.** The first Step B attempt differed from Step A on
   exactly one clip (`aigen_cogvideox_rain`, the only core clip narrower than 854 px), by ~2e-5 of
   normalized source U across ~15% of pixels. It was not a lost root -- tripling the margin changed
   nothing. The compiler strength-reduces `(float)(probeStart + i) * spacing` into an accumulator
   **seeded at each run's own starting index**, so the "shared" lattice was not shared. Two fixes
   work: marking the position `precise` (blocks the transform, costs ~18% more warp time) or
   quantizing `spacing` to a power-of-two multiple so every partial sum is exactly representable
   and accumulation equals multiplication (shipped; moves spacing by <2^-11). **Any future
   byte-compare across two shaders that index the same grid differently is exposed to this.**
   Two smaller findings: the loop trip count is frame-uniform because at the narrow radius that
   measured faster than a per-pixel count (0.0366 vs 0.0381 ms), though at the old oversized radius
   the ranking inverted; and the `clamp(..., 12, 72)` step floor is gone, since a 12-probe floor
   would have bound at the new counts. The no-root `bgX` fallback remains unreachable, now by
   proof rather than by observation: the window's outermost probes lie beyond `max|parallax|`, so
   `g` has opposite signs at the two ends and the IVT guarantees a crossing. The byte-compare
   confirms it -- `bgX`'s search span shrank ~3x with no output change anywhere.
   Evidence: `probe-control-{core,ext}`, `lattice2-{core,extended}`, `tight2-{core,extended}`.
3. ~~Collect scene-level headset labels for explicit zero-plane placement.~~ **Done 2026-07-24:
   `median` was labelled decisively better in the headset and is now the default** (see the
   superseding entry above). The remaining open work is the *per-scene selector*, not the global
   default: a learned/semantic selector may use image content and depth together, but do not ship
   a percentile-only rule from the current small suite.

## References

- `tools/sbsbench/README.md` — build, evaluation, report, and dataset commands.
- `docs/sbs-feature-decision-revisit.md` — historical accepted/rejected evidence.
- `docs/sbs-resolution-robustness.md` — coordinate-space and encoder-resolution audit.
- `src/video_depth_estimator.cpp` — V2 producer plus the temporary legacy scene-cut bridge.
- `src_assets/windows/assets/shaders/directx/sbs_reprojection_v2_live_ps.hlsl` — production live geometry.
- `src_assets/windows/assets/shaders/directx/sbs_reprojection_ps.hlsl` — legacy offline/evaluator geometry only.
