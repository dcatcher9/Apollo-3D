# SBS host benchmarks — visual + perf

> Historical notes mentioning VD3D describe the completed warp-selection investigation. VD3D is
> no longer a production or evaluator option; current commands live in `tools/sbsbench/README.md`.

Design for two reproducible, host-side benchmarks so every SBS change ships with numbers:
**how much did visual quality change** and **how much did perf change**. Written 2026-07-07.

> Goal: replace "guess → deploy → headset eyeball" and the drifting offline warpsim with a
> deterministic, real-pipeline measurement. Each improvement produces a scorecard diff
> (`pop +6%, stretch-band −22%, flicker −15%`, `movie warp p95 6.8 → 2.1 ms`).

> **STATUS (2026-07-07): both benchmarks implemented + validated.**
> - **Perf** — in-app `sbs_3d_perf_stats` (src/sbs_perf.*): per-stage p50/p95/max + sbs_perf.json.
>   Live-validated both modes (game depth 2.0/warp 1.6 ms, movie 11.0/12.9 ms).
> - **Visual** — `tools/sbsbench/` (no-reference pop / vmisalign / stretch-band + `--seq` flicker),
>   `split_video.py`, and the **Tier-1 headless harness** `sunshine --sbs-bench` (src/sbs_bench_harness.cpp:
>   real estimator + real composite over a frames dir → SBS PNGs). Full chain validated:
>   video → frames → harness → sbsbench --seq.
> - **Eval v4** — hard signed-disparity/coverage/integrity limits, source-relative halo/stretch,
>   scale/shift-invariant GT-depth accuracy on deterministic synthetic clips, and exact-or-classical
>   optical-flow-compensated output/depth temporal validation. Native public metric-depth and
>   exact optical-flow sidecars are supported. The separate `extended-v2` suite adds visually
>   inspected Bonn RGB-D Dynamic, TartanAir V2, MPI Sintel Stereo, and Virtual KITTI 2 clips; URLs, SHA-256
>   hashes, frame windows and preparation logic are committed while media remains external.
>   The first eight-clip run exposed an Apollo bug: Bestv2 source-pixel shifts were normalized by
>   inference-depth width instead of source-color width, creating resolution-dependent amplification.
>   After the fix, median VD3D volume delta is −6.0%; seven clips are within noise, including two
>   where VD3D is slightly higher. Only Bonn walking retains a just-over-one-pixel geometry difference.
> Remaining: reference warp PSNR/SSIM (needs GT stereo) and a dedicated ghost/double-image metric.

## Why the offline warpsim was removed

The former CPU simulator (`tools/warpsim/`) could not predict the live result for five reasons.
Its code was removed after the real-pipeline evaluator became authoritative.

1. **It eats post-normalization depth.** warpsim consumes `depth.png` = the already-upsampled,
   min/max-EMA-normalized *guided* depth. Everything upstream — depth model, shifted-reciprocal
   transform, min/max EMA, A1 scene-cut snap, A3 range floor, guided-vs-MLBW — is baked in and
   untunable. (This is exactly why A1/A3 were "UNVALIDATED offline".)
2. **Single frame, no time axis.** Ghost (async depth lag), inpaint flicker (¼-res
   re-hallucination), depth swim after cuts, mask jitter are all temporal. A one-frame replica
   cannot see them — and they are the dominant live complaints now.
3. **Replica drift.** warpsim is a hand-maintained CPU-fp32 reimplementation of the shaders.
   fp16 rounding, HW-bilinear-vs-torch-antialias, and D3D sampler semantics differ subtly, and
   the warp amplifies tiny deltas (the measured 6.85% pre-smoothed-depth finding).
4. **Missing downstream stages.** NVENC HDR-8K compression, client decode, XR optics/resampling
   are never simulated; some "artifacts" are encode/scaling, not warp.
5. **Config skew.** Hardcoded region constants + a knob subset vs. whatever was live.

**Design decision that removes 1–3:** run the *real production code* — the actual
`video_depth_estimator` + the actual DirectX shaders — on the host, against fixed **raw** input
frame sequences. A Tier-2 in-stream capture removes 4.

## Shared linchpin: the deterministic clip harness

Both benchmarks are meaningless without identical input every run. Foundation = a fixed clip
library fed through the real pipeline.

There are two levels: the committed **core** library for rapid iteration, and the reproducibly
prepared **extended** public-data suite for feature acceptance and the final warp decision. Run
`prepare_public_datasets.py` once, then select it with `run_eval.py --suite extended`. Extended
baselines live separately in `baselines_extended/`, so they cannot be mixed with core clips.

- **Clip library:** `E:\ApolloDev\sbs_bench\clips\<name>\frame_%04d.png` (RGB or HDR PNG/EXR),
  each a short sequence (e.g. 60–120 frames). Chosen to span the catalogued failure modes:
  | clip | targets |
  |------|---------|
  | `game_arm` | silhouette staircase / stretch band (fullscreen game) |
  | `movie_fullscreen` | stretch band at high parallax (movie) |
  | `video_windowed` | window-edge halo disocclusion (roadmap #4) |
  | `hair_sky` | fold-over corrugation (left eye) |
  | `fast_motion` | ghost / async lag band |
  | `scene_cut` | depth swim after cuts (A1) |
  | `flat_page` | V2 hallucination + range amplification (A3) |

  Seed these from existing `E:\ApolloDev\sbs_dump\dump_*` frames (single frames are fine for
  no-reference spatial metrics; multi-frame sequences are required for temporal metrics — see
  Tier-2 burst dump to capture them).

- **Two run tiers:**
  - **Tier 1 — headless algorithm bench (workhorse).** Standalone exe that instantiates the real
    estimator + real shaders (no game, headset, or NVENC) and pushes each clip frame-by-frame at
    a pinned cadence, dumping the real SBS output + per-stage timings. Reproducible, scriptable,
    CI-able. Reuses production code paths → consistent with the live run for stages 1–3.
  - **Tier 2 — in-stream instrumentation + burst dump.** Perf counters and an N-frame consecutive
    dump (extend the 0x3004 path) toggled from the XR bar during a real stream, capturing
    *post-NVENC* frames. Validates Tier 1 against the headset and catches encode/client effects.

## Benchmark A — visual

Computed on the **real host SBS output** per clip. Ship no-reference first; add reference later.

### A.1 No-reference artifact metrics (phase 1, no new content)
- **Pop** — mean & p95 horizontal L↔R disparity (the "flat pop" axis, roadmap #2). Higher = more
  depth. Derive from the known warp/MLBW delta field where available, else L/R optical flow.
- **Stretch-band / edge fidelity** — detect depth-discontinuity silhouettes; in the disocclusion
  strip measure horizontal smear energy and band-vs-source SSIM (roadmap #1 target).
- **Temporal flicker** — the metric the sim can't produce: frame-to-frame residual energy in
  inpaint bands and static regions (catches ¼-res re-hallucination + depth swim). Requires a
  multi-frame clip.
- **Ghost** — lag-band width / double-image energy on `fast_motion`.
- **Geometry sanity** — L/R object-width consistency (a known correctness check).

### A.2 Reference metrics
Implemented for **depth**: clips may carry identity-matched 16-bit `gt_depth/frame_*.png`; prediction
is evaluated with scale/shift-invariant RMSE and boundary F1. `flat_page` and `fast_motion` provide
deterministic references. Missing GT remains absent.

Pending for **warp imagery** (needs GT stereo content):
For synthetic/known-stereo clips: feed one eye through the pipeline, reconstruct the other,
compare to the true eye. `PSNR / SSIM / LPIPS`, global and disocclusion-band-restricted. Gold
standard for warp+inpaint correctness. Requires rendered (Blender second-eye camera) or dataset
stereo pairs. Deferred until phase 1 is proven.

### Output
Scorecard: JSON + rendered HTML, one row per clip × metric, diffed vs. a stored baseline
(`E:\ApolloDev\sbs_bench\baseline\*.json`). Emits per-metric deltas.

## Benchmark B — perf

Wire the A2 per-stage timing (`sbs_3d_perf_stats`), p50/p95/max over each clip.

- **D3D11 timestamp-disjoint queries** around each D3D pass: `rgb_to_nchw_cs`,
  `depth_minmax_cs`, `depth_minmax_ema_cs`, `buffer_to_tex_cs`, guided-upsample CS /
  `mlbw_input_cs`, `mlbw_field_cs`, `sbs_mlbw_composite_ps`, jacobian/`mlbw_mask_cs`,
  `inpaint_input_cs`, `inpaint_writeback_ps`, `sbs_reprojection_ps` (fallback), Tier-2 NVENC.
- **CUDA events** around the three TRT `enqueueV3` calls (depth / warp / inpaint) — they run on
  CUDA streams and are invisible to D3D queries. Use the existing dynamically-loaded
  `cuEventRecord`/`cuEventElapsedTime` (`src/cuda_driver_api.h`).
- **Frame-level:** end-to-end `convert()` wall time, GPU busy %, VRAM, effective depth cadence
  (interval), dropped/repeated frames.
- Same output shape: JSON/CSV + baseline diff, same clip set, pinned config.

## Integration points (real code hooks)
- Pipeline entry / dump site: `src/platform/windows/display_vram.cpp` `convert()` (~:509, the
  `sbs_dumper.maybe_dump` call) — Tier-2 counters + burst dump hook here.
- Estimator (all TRT + compute passes): `src/video_depth_estimator.cpp/.h` — CUDA-event
  instrumentation around `enqueueV3`; Tier-1 harness constructs this directly.
- Single-frame dump to extend into N-frame burst: `src/platform/windows/sbs_debug_dump.{h,cpp}`
  + `video::sbs_debug_dump_pending` (0x3004).
- Shaders: `src_assets/windows/assets/shaders/directx/` (runtime-compiled → no exe rebuild).
- Config knobs to add: `sbs_3d_perf_stats` (already scoped as A2), `sbs_3d_bench_clips`
  (Tier-1 input dir), `sbs_3d_burst_frames` (Tier-2 N).

## Build order
1. **Deterministic clip harness + Tier-1 headless runner** (reuses estimator/shaders). — the
   foundation; nothing else is comparable without it.
2. **Perf counters** (mostly the scoped A2, now with a report sink).
3. **No-reference visual metrics + scorecard** (phase-1 A.1).
4. **Reference metrics** (phase-2 A.2 — pending GT-content decision).
5. **Tier-2 in-stream burst dump + post-NVENC capture** (cross-validates Tier 1 to the headset).

## Open decisions
- **Visual GT content:** no-reference only vs. add synthetic GT stereo (phased). Default: phase-1
  no-reference first, GT deferred to step 4.
- **Primary tier:** Tier-1 headless first (default) vs. Tier-2 in-stream first.

## Key references
- `tools/warpsim/README.md` — removal notice for the obsolete CPU replica.
- `docs/sbs-3d-roadmap.md` — the artifacts these metrics quantify (#1 stretch band, #2 pop).
- `sbs-3d-depth-feature` memory — A2 (perf timing) + A5 (jacobian mask) design context.
