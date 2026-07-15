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
   `sbs_3d_max_encode_width`, preserve each eye's aspect while scaling to the cap.

Profiles remain configuration-only parameter sets over this single geometry. Define fields with
`sbs_3d_profile_<name>_<parameter>`; `sbs_3d_profile` selects the startup preset. Explicit top-level
`sbs_3d_*` keys override the corresponding selected-profile value. Artemis switches only between
Normal and Host SBS AI; changing the host profile requires restarting Apollo.

Bestv2 disparity is calibrated at the evaluator's 854-pixel source width and normalized to the
5120x2160 Artemis reference aspect. `sbs_3d_pop_strength` scales the final parallax (`0.25`–`2`,
default floor `1.25`) without changing that resolution correction. The accepted scene latch may
select up to `1.30` from depth-edge risk and holds the selection until a hard cut.

An experimental multi-output DA-V2 model may additionally emit a confidence-gated safe scale
ceiling. `sbs_3d_artistic_style = clean|balanced|immersive` consumes zero, half, or all of that
validated headroom and is inert for the shipping depth-only model. No learned checkpoint is
approved by default; see `docs/sbs-artistic-policy-plan.md` for the frozen promotion gates.

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
- Art3D-style shot-level zero-plane placement was screened as three scene-latched treatments:
  tracked subject, depth median, and far/mid-background (P25). Each resolves its anchor through
  the final Bestv2 curve and stores the source-pixel shift, so percentile motion cannot make
  convergence breathe between cuts. A legacy smoke run was byte-identical. No fixed anchor is a
  production win: median improved 6 primary Spring metrics with 1 cost, including character
  volume and daylight halo/stretch, but core lost fast-motion volume and added c747/c841 warp
  costs. Subject improved Spring artifact/reference fidelity but reduced character volume;
  background was the least harmful core treatment but weaker than median on Spring. Visual
  inspection confirmed convergence redistribution rather than universal detail recovery. Keep
  `sbs_3d_zero_plane = legacy`; retain the explicit modes for controlled headset labels and a
  future semantic controller. Depth histograms alone are not a safe selector: scenes with similar
  percentiles and edge density chose different winners. Evidence:
  `zero-plane-{legacy,subject,median,background}-{core-screen,spring}`.

Do not reintroduce a removed processor without a current core and extended comparison, visual
evidence, and a headset-motivated hypothesis.

## Evaluation workflow

Use `tools/sbsbench/run_eval.py`. Every decision run must use matching model, clips, metric schema,
and profile provenance; cover the 11-clip core and public extended suites; generate `report.html`
and `decision.json`; inspect primary-axis examples; and treat comfort/image-integrity limits as hard
constraints. Headset evidence resolves coequal-axis tradeoffs.

The harness uses contract 24 and eval schema 29. It exports raw depth, pre-warp depth, exact forward
coverage diagnostics, clamped and unclamped HLSL full-binocular disparity at the rendered
output-eye raster (with bars excluded by the exact pixel-center validity rule), and final SBS
artifacts by numeric frame identity. The primary stereo-volume metric,
`exact_pop_spread_pct`, is p95-p5 of that exact clamped field; the older image
phase-correlation `pop_spread_pct` estimate is diagnostic only. Ground-truth depth scoring
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
2. Collect headset preference labels for the current scene-latched 1.25-1.30 adaptive-pop band.
   The Spring true-stereo A/B is neutral and does not support raising the ceiling or replacing the
   current edge-risk controller without perceptual evidence.
3. Collect scene-level headset labels for explicit zero-plane placement. A learned/semantic
   selector may use image content and depth together; do not ship a percentile-only rule from the
   current small suite.

## References

- `tools/sbsbench/README.md` — build, evaluation, report, and dataset commands.
- `docs/sbs-feature-decision-revisit.md` — historical accepted/rejected evidence.
- `docs/sbs-resolution-robustness.md` — coordinate-space and encoder-resolution audit.
- `src/video_depth_estimator.cpp` — depth normalization and subject state.
- `src_assets/windows/assets/shaders/directx/sbs_reprojection_ps.hlsl` — production geometry.
