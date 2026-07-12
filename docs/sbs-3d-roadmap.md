# SBS 3D — current status and roadmap

Apollo converts a captured mono frame into host-rendered SBS using TensorRT depth estimation and
one of two production warp profiles. This document describes the current stack; historical feature
experiments and their evidence are recorded in `sbs-feature-decision-revisit.md` and
`sbs-vd3d-parity.md`.

## Shipping pipeline

1. Preserve source aspect while selecting a patch-aligned TensorRT input grid.
2. Convert SDR or HDR capture into the color domain expected by the depth model.
3. Transform model output into Apollo's high-is-near disparity convention.
4. Normalize with permanent P2/P98 bounds and temporal range EMA.
5. Apply per-pixel depth EMA, permanent Bestv2 subject estimation, P5/P95 stretch/recenter, and
   optional exact subject-plane lock.
6. Render either:
   - `apollo`: 24-step occlusion-aware backward probe;
   - `vd3d`: depth-ordered forward splat blended with the VD3D backward warp.
7. Convert the final packed SBS raster directly to the encoder format. When doubled width exceeds
   `sbs_3d_max_encode_width`, the packed raster is downscaled while preserving each eye's aspect.

The default `vd3d` profile and alternate `apollo` profile share the same model, normalization,
subject shaping, aspect mapping, and encoder path. They differ only in warp geometry and the VD3D
forward/backward blend. Individual `sbs_3d_*` values explicitly present in configuration override
the corresponding value in every profile. Additional profiles are configuration-only: define their
fields with `sbs_3d_profile_<name>_<parameter>` keys; `sbs_3d_profile` chooses the default. Apollo
advertises the complete list and Artemis can switch the whole profile per stream. The client does
not select Game/Movie behavior, a model, or individual processing parameters.
Bestv2 disparity is calibrated at the evaluator's 854px source width and normalized to the
5120x2160 Artemis stream reference, so other client-requested aspects retain the same perceived
stereo volume.
The optional `sbs_3d_pop_strength` control scales the shared final parallax field for both warps
(`0.25`-`2`, default `1.25`) without altering that resolution correction. The Apollo and VD3D
profiles deliberately inherit the same strength so their warp comparison remains controlled.

## Frozen processor decisions

- Bestv2 subject estimation and P2/P98 normalization are mandatory pipeline stages.
- Range-to-pixel temporal ordering and 24 Apollo probes are permanent.
- Guided upsample, curvature, scene snap, range/depth floors, border fade, legacy shift, and the
  CPU warpsim were rejected and removed.
- Exact subject-plane lock and Bestv2 sharpen remain available fidelity ablations but are disabled
  in both quality profiles.
- The old MLBW learned-warp experiment was rejected because it bypassed the shared Bestv2 field.

Do not reintroduce a removed processor without a new implementation, current core and extended
reports, visual evidence, and a headset-motivated hypothesis.

## Evaluation workflow

Use the real D3D11 harness through `tools/sbsbench/run_eval.py`. Every decision run must:

- use the same model, clip identities, metric schema, and profile provenance as its baseline;
- run the 11-clip core suite for the committed gate and the public extended suite for
  generalization;
- generate `report.html` together with the machine-readable conclusion;
- inspect primary-axis examples and any large diagnostic movement;
- treat comfort and image-integrity limits as hard constraints;
- use headset evidence to resolve coequal-axis tradeoffs rather than the scalar score.

Ground-truth depth comparison is scale/shift invariant but polarity preserving. Expected-flat clips
diagnose false stereo and are exempt only from the stereo-volume axis. Source-relative evidence is
computed on a bounded working raster so full-resolution report generation remains memory safe.

## Current priorities

1. Validate Apollo versus VD3D on core and extended suites after every correctness or performance
   change, then select the production warp from both metrics and headset evidence.
2. Improve disocclusion fill only if a reproducible primary warp-axis failure remains. A learned
   inpaint may be evaluated on top of the selected warp; it must not replace the shared depth and
   subject field.
3. Continue model evaluation for movie content where a larger model or lower depth cadence may be
   affordable.
4. Add real-device performance telemetry when a live-only bottleneck cannot be represented by the
   offline harness. The harness reports `warp_infer`; the live path currently reports
   `depth_infer` and `sbs_convert_cpu`.

## References

- `tools/sbsbench/README.md` — build, eval, report, and dataset commands.
- `docs/sbs-vd3d-parity.md` — Apollo/VD3D implementation parity and feature ledger.
- `docs/sbs-feature-decision-revisit.md` — accepted/rejected processor evidence.
- `docs/sbs-resolution-robustness.md` — coordinate-space and encoder-resolution audit.
- `src/video_depth_estimator.cpp` — depth normalization and subject-state implementation.
- `src_assets/windows/assets/shaders/directx/` — production warp shaders.
