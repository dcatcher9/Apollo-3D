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
default `1.25`) without changing that resolution correction.

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
- Guided upsample, curvature, scene snap, range/depth floors, border fade, legacy shift, VD3D
  hybrid warp, and CPU warpsim were rejected and removed.
- Subject-plane lock, Bestv2 sharpen, and EMA-mask dilation were rejected and removed.

Do not reintroduce a removed processor without a current core and extended comparison, visual
evidence, and a headset-motivated hypothesis.

## Evaluation workflow

Use `tools/sbsbench/run_eval.py`. Every decision run must use matching model, clips, metric schema,
and profile provenance; cover the 11-clip core and public extended suites; generate `report.html`
and `decision.json`; inspect primary-axis examples; and treat comfort/image-integrity limits as hard
constraints. Headset evidence resolves coequal-axis tradeoffs.

The harness uses contract 12 and eval schema 19. It exports raw depth, pre-warp depth, exact forward
coverage diagnostics, and final SBS artifacts by numeric frame identity. Ground-truth depth scoring
is scale/shift invariant but polarity preserving.

## Current priorities

1. Correlate the new previous-only GT ghost-edge diagnostic with additional known-motion scenes
   and headset evidence before allowing it to become a primary gate.

## References

- `tools/sbsbench/README.md` — build, evaluation, report, and dataset commands.
- `docs/sbs-feature-decision-revisit.md` — historical accepted/rejected evidence.
- `docs/sbs-resolution-robustness.md` — coordinate-space and encoder-resolution audit.
- `src/video_depth_estimator.cpp` — depth normalization and subject state.
- `src_assets/windows/assets/shaders/directx/sbs_reprojection_ps.hlsl` — production geometry.
