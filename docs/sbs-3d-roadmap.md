# SBS 3D — current status and roadmap

Apollo converts captured mono frames into host-rendered SBS with TensorRT depth estimation and one
production geometry implementation: Apollo's occlusion-aware backward probe. The retired VD3D
forward/backward hybrid and its selectable profile were removed after headset testing found worse
rim/halo behavior and a serious thin-structure artifact for only about 0.12 ms of warp-time saving.
Historical experiment evidence remains in Git history and `sbs-feature-decision-revisit.md`.

## Shipping pipeline

1. Preserve source aspect while selecting a patch-aligned TensorRT input grid.
2. Convert SDR or HDR capture into the color domain expected by the depth model.
3. Transform model output into Apollo's high-is-near convention.
4. Normalize with permanent P2/P98 bounds and temporal range EMA.
5. Apply per-pixel EMA, accepted edge/change-aware EMA, Bestv2-derived subject estimation,
   P5/P95 stretch/recenter, and optional exact subject-plane lock.
6. Render Apollo's occlusion-aware backward probe.
7. Convert the packed SBS raster directly to the encoder format. If doubled width exceeds
   `sbs_3d_max_encode_width`, preserve each eye's aspect while scaling to the cap.

Profiles remain configuration-only parameter sets over this single geometry. Define fields with
`sbs_3d_profile_<name>_<parameter>`; `sbs_3d_profile` selects the default. Explicit top-level
`sbs_3d_*` keys override the corresponding value in every profile. Apollo advertises the complete
profile list and Artemis can switch profiles atomically during a stream.

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
- DA-V2 Base FP16 was re-screened as a model-only replacement for Small and rejected as the
  production default. On the extended suite it improved GT boundary F1 from 45.07% to 50.27%, but
  left GT depth RMSE flat (11.28% to 11.26%), worsened mean source halo from 2.68 to 2.99, and
  nearly doubled inference time from 2.10 ms to 4.05 ms. The core suite also lost five stability
  votes and four warp votes; visual inspection confirmed that the large c747 edge-distance gain
  came with a different doorway/person interpretation and more rim. Keep the registry entry only
  for explicit future quantized/performance experiments. Evidence: `dav2-rescreen-core-base` and
  `dav2-rescreen-extended-base` under `cmake-build-relwithdebinfo/sbs_eval/`.
- Guided upsample, curvature, scene snap, range/depth floors, border fade, legacy shift, VD3D
  hybrid warp, and CPU warpsim were rejected and removed.
- Subject-plane lock and Bestv2 sharpen remain fidelity ablations but are disabled by default.

Do not reintroduce a removed processor without a current core and extended comparison, visual
evidence, and a headset-motivated hypothesis.

## Evaluation workflow

Use `tools/sbsbench/run_eval.py`. Every decision run must use matching model, clips, metric schema,
and profile provenance; cover the 11-clip core and public extended suites; generate `report.html`
and `decision.json`; inspect primary-axis examples; and treat comfort/image-integrity limits as hard
constraints. Headset evidence resolves coequal-axis tradeoffs.

The harness uses contract 11 and eval schema 19. It exports raw depth, pre-warp depth, exact forward
coverage diagnostics, and final SBS artifacts by numeric frame identity. Ground-truth depth scoring
is scale/shift invariant but polarity preserving.

## Current priorities

1. Evaluate calibrated FP8 inference as a separate model contract; never replace FP16 artifacts
   or baselines in place.
2. Use optical flow only for EMA gating, cut detection, history validation, and ghost diagnostics.

## References

- `tools/sbsbench/README.md` — build, evaluation, report, and dataset commands.
- `docs/sbs-feature-decision-revisit.md` — historical accepted/rejected evidence.
- `docs/sbs-resolution-robustness.md` — coordinate-space and encoder-resolution audit.
- `src/video_depth_estimator.cpp` — depth normalization and subject state.
- `src_assets/windows/assets/shaders/directx/sbs_reprojection_ps.hlsl` — production geometry.
