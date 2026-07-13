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

The harness uses contract 10 and eval schema 18. It exports raw depth, pre-warp depth, exact forward
coverage diagnostics, and final SBS artifacts by numeric frame identity. Ground-truth depth scoring
is scale/shift invariant but polarity preserving.

## Current priorities

1. Test silhouette-band alignment on top of the accepted edge-aware EMA.
2. Re-screen DA-V2 base only if its quality gain fits the live frame budget.
3. Test edge-aware depth sampling at discontinuities.
4. Measure CUDA Graph/FP8 opportunities under real game contention before changing inference.
5. Use optical flow only for EMA gating, cut detection, history validation, and ghost diagnostics.

## References

- `tools/sbsbench/README.md` — build, evaluation, report, and dataset commands.
- `docs/sbs-feature-decision-revisit.md` — historical accepted/rejected evidence.
- `docs/sbs-resolution-robustness.md` — coordinate-space and encoder-resolution audit.
- `src/video_depth_estimator.cpp` — depth normalization and subject state.
- `src_assets/windows/assets/shaders/directx/sbs_reprojection_ps.hlsl` — production geometry.
