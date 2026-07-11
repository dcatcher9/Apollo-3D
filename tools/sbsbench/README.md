# sbsbench — visual metrics for host SBS 3D output

No-reference visual metrics computed on **real "Dump 3D" output** (the actual `sbs.png` the
client receives). Unlike `tools/warpsim/` (a CPU replica of the reprojection shader), this
re-runs nothing — it measures the live pipeline's final frame — so its numbers match what you
see on the headset. This is the visual half of the host benchmark; see
[docs/sbs-benchmark-plan.md](../../docs/sbs-benchmark-plan.md). The perf half is the in-app
`sbs_3d_perf_stats` timing.

## Workflow
1. Capture frames: tap **Dump 3D** in the XR bar → `E:\ApolloDev\sbs_dump\dump_*/` (needs
   `APOLLO_SBS_DUMP` set; `run-dev.cmd` already sets it). Grab several across a scene / both
   modes; each folder has `source.png`, `depth.png`, `sbs.png`, `meta.txt`.
2. Score a set and save a baseline **before** your change:
   ```
   python tools/sbsbench/sbsbench.py --glob "E:/ApolloDev/sbs_dump/dump_2026*" --json base.json
   ```
3. Make the change, re-dump the same kind of scene, and diff:
   ```
   python tools/sbsbench/sbsbench.py --glob "E:/ApolloDev/sbs_dump/dump_NEW*" --baseline base.json
   ```
   → prints per-metric deltas (`pop_px_p50 12.0 -> 15.3  +3.3 (+27%)`).

Dependencies: `numpy` + `Pillow` only (system Python 3 is fine).

## One-command eval loop (start here)

```
python tools/sbsbench/run_eval.py                     # all committed clips vs committed baselines
python tools/sbsbench/run_eval.py --update-baselines  # after an INTENDED change: re-baseline + commit
python tools/sbsbench/run_eval.py --extra --divergence 0.027   # pass A/B levers to the harness
python tools/sbsbench/run_eval.py --label treat --report-control cmake-build-relwithdebinfo/sbs_eval/control --extra --warp vd3d
python tools/sbsbench/run_eval.py --label profile-b --conf profile-b.conf --report-control cmake-build-relwithdebinfo/sbs_eval/profile-a --report-allow-config-diff
python tools/sbsbench/run_eval.py --comparison-only --label ab-control  # fresh A/B; no committed gate
python tools/sbsbench/run_eval.py --suite extended --label public-control # prepared public suite
python tools/sbsbench/rescore_run.py cmake-build-relwithdebinfo/sbs_eval/<run> --in-place  # metrics only
```

**Mode / model (important):** `--mode` defaults to **`game`** (depth model **DA-V2 small**,
`depth_anything_v2_fp16`) — the committed baselines are game/DA-V2. `--mode movie` swaps in
**DA3MONO** (`da3mono_large_fp16`, a different, more aggressive model) and would NOT match the
game baselines. Always eval in game mode unless you deliberately want DA3MONO, and never compare
numbers across modes (they're different models). Default was `movie` until 2026-07-10; it was
flipped to `game` because DA-V2 is the model we test against (matches VisionDepth3D's DA-V2 at
r≈0.996) and implicitly selecting DA3MONO had caused an entire wrong-model
comparison.

Harness A/B levers (after `--extra`):
- `--warp apollo|vd3d` — choose Apollo's occlusion-aware probe or VD3D's Bestv2 hybrid
  (`35%` backward grid warp + `65%` depth-ordered forward splat and directional hole fill).
- `--vd3d-forward-blend F` — override the VD3D forward weight (`0.65` in Bestv2; `0` isolates
  its classic backward warp and `1` isolates the forward splat).
- `--divergence F` — parallax gain.
- `--shift-profile apollo|bestv2` — choose the disparity field independently of geometry.
  `bestv2` uses the preset's source-pixel FG/MG/BG shifts (`-9/-3/+2.4`), `.35` parallax
  balance, `1.11/1.05` multipliers, `.008` zero-parallax trim, dynamic convergence `.006`,
  `.071` safety cap, and the exact center-weighted/morphologically filled subject-plane lock.
  The completed SDR frame then receives Bestv2's exact per-eye sharpen 0.2 convolution.
  This is resolution-calibrated; `--divergence` remains the Apollo profile and uninitialized-depth
  fallback. Bestv2 cinematic-window sculpt was evaluated and rejected; see
  `docs/sbs-vd3d-parity.md`.
- `--depth-short-side N` — depth inference short side (default 432; VD3D parity). 336 to A/B
  back to the old under-resolved default.
- `--pct-lo F --pct-hi F` — robust percentile normalization bounds, e.g. `1 99` (default off =
  raw min/max).
- `--ema F` — per-pixel depth EMA override (`1.0` = off).
- `--subject-track` — VD3D-style shaped disparity (subject-anchored band curve). The pipeline
  is probe-reprojection-only, so the shaping is always live when this is on.
- `--subject-lock F` — subject anchor strength (e.g. `0.95`).
- `--subject-stretch` — shape_depth_for_pop 5/95 percentile stretch (default on within the
  subject path).
- `--subject-plane-lock F` — local subject-band flatten (e.g. `0.28`; default off).
- `--curvature F` — foreground-curvature bulge strength (e.g. `0.07`; reshapes the depth
  texture, both warp paths; default off).

Exit code is the verdict (0 pass / 1 regression / 2 setup error), so the eval→fix→eval loop is
scriptable. `results.json` carries provenance (git sha+dirty, models, clip hashes, gpu-contention
flag) and, for every triggered/regressed metric, the **worst frame index** to look at. The gate
thresholds live in [thresholds.json](thresholds.json); the pinned SBS config in
[bench.conf](bench.conf); baselines in `baselines/` (regenerate in the same commit whenever
bench.conf, the clip set, or a metric definition changes). Guards: fails fast if TRT engines
aren't prebuilt; warns + skips the perf gate if another sunshine.exe is running.

Hard comfort/integrity bounds apply even in comparison-only runs. Baseline updates are staged in
memory and written atomically only after every clip passes those bounds, so a broken render cannot
become the new normal.

The harness records `warp_infer` with D3D11 GPU timestamps around the selected warp. Use this for
the dual-warp performance comparison; `sbs_composite_cpu` measures submission overhead only.
For A/B runs, pass `--report-control <control-run-dir>` and the evaluator writes `report.html`
before printing its conclusion, including when the treatment exits with regressions.

Metric notes: silhouette detection runs at the native depth resolution with an absolute
depth-step floor (flat scenes legitimately read 0), and all pixel windows scale with the eye
width — but absolute values are still not comparable across clip resolutions; baselines are
per-clip-set. The harness writes 16-bit depth PNGs so `swim` resolves below 1/255.

**Eval schema 4 correctness contract (2026-07-10):** `run_eval.py` pins the model explicitly. The
harness submits and consumes exactly one inference per source frame, so EMA and normalization
update once.
Source, raw-model (`raw_*.f32`), pre-warp depth (`depth_*.png`), and SBS artifacts are joined by
numeric frame identity, never list position. Baselines are rejected with setup exit 2 if mode,
model, schema, stepping semantics, config hash, metric hash, or clip hash differs. Output folders
are cleared before reuse. `--output-every N` reduces saved artifacts while still processing every
input frame, so sampling cannot change temporal state.

Schema 3 added optional `gt_depth/frame_*.png` clip sidecars and includes them plus their semantic
`gt_depth_kind` metadata in the clip hash. Missing GT means reference metrics are absent—not zero
and not inferred. `flat_page` and `fast_motion` carry deterministic 16-bit disparity ground truth
generated by `make_synth_clips.py`; recorded movie clips remain no-reference.

Schema 4 also accepts native float32 `gt_depth/frame_*.npy` and exact forward-flow
`gt_flow/frame_*.npz` sidecars. Each flow sidecar belongs to its current frame and contains
`flow` (previous→current, H×W×2 pixels) plus an optional `valid` mask. The evaluator forward-splats
that source-grid flow to current coordinates and still requires photometric agreement; clips
without exact flow keep the classical phase-correlation fallback. Both reference folders are part
of the clip identity hash.

Clips may declare `"expected_flat": true` in `meta.json`. Their score rewards near-zero false
stereo instead of rewarding volume, while still measuring depth hallucination and temporal swim.
Disocclusion ratios require at least 0.1% eye-area support; below that, smear and disocclusion
flicker are omitted rather than turning a handful of pixels into a maximum penalty. Reports name
both run directories and show control/treatment provenance separately, even when their CLI args
are identical because the treatment is a code change.

## Deterministic clips via the headless harness (recommended)
Single dumps are sporadic and headset-bound. For repeatable A/B and **temporal** metrics, drive
the real pipeline over a fixed frame sequence with the built-in `--sbs-bench` subcommand (Tier-1
harness — runs the real estimator + real composite shaders, no game/client).

### The committed clip set (quick eval)
A small pre-resized clip set lives in **`tools/sbsbench/clips/<name>/frame_*.jpg`** (854 px wide,
24 frames, JPEG) so eval is fast and reproducible with no per-run preprocessing. Five recorded
movie clips (c339/c525/c647/c747/c841, each fingerprinting different artifacts) plus three
generated failure-mode clips from [make_synth_clips.py](make_synth_clips.py):

| clip | targets | validated fingerprint |
|------|---------|----------------------|
| `scene_cut` | depth-normalization response across a hard cut | schema-2 baseline flags the cut's stretch/rim behavior without duplicate EMA updates |
| `flat_page` | flat-content depth hallucination + amplification | static-input noise floor; disocc_smear flags hallucinated text-edge silhouettes |
| `fast_motion` | known 30 px/frame motion | current-frame depth separates warp/edge behavior from live async lag |

### Public extended suite (decision eval)

The quick suite remains useful for iteration, but the final warp/profile decision uses a separate,
reproducibly prepared public suite. Media is cached outside Git (default
`E:\ApolloDev\sbs_bench\datasets`); the repository commits the exact URLs, SHA-256 checksums,
frame windows, adapters, and baselines:

```
python tools/sbsbench/prepare_public_datasets.py
python tools/sbsbench/run_eval.py --suite extended --comparison-only --label public-apollo
python tools/sbsbench/run_eval.py --suite extended --comparison-only --label public-vd3d \
  --report-control cmake-build-relwithdebinfo/sbs_eval/public-apollo --extra --warp vd3d
```

`extended-v2` contains eight visually inspected 24-frame clips. The first four remain:

| clip | source | coverage / reference |
|------|--------|----------------------|
| `bonn_person_walk` | Bonn RGB-D Dynamic | real walking person, silhouette motion, registered depth |
| `bonn_person_close` | Bonn RGB-D Dynamic | close person/robot occlusion, registered depth |
| `tartanair_house_easy` | TartanAir V2 | attic/bed transition, exact metric depth and flow |
| `tartanair_house_motion` | TartanAir V2 | indoor/outdoor rotation, exact metric depth and flow |

The v2 expansion adds independent cinematic and outdoor-driving content:

| clip | source | coverage / reference |
|------|--------|----------------------|
| `sintel_ambush` | MPI Sintel Stereo | cinematic non-rigid action and rendered right-eye reference |
| `sintel_market` | MPI Sintel Stereo | people, clutter, depth layering and rendered right-eye reference |
| `vkitti_drive_clone` | Virtual KITTI 2 | clear outdoor driving and exact metric depth |
| `vkitti_drive_rain` | Virtual KITTI 2 | rainy low-contrast driving and exact metric depth |

Initial matched-profile validation (2026-07-10) found lower VD3D stereo volume on all eight clips:
median `-22.3%`, with four clips crossing the primary gate (`bonn_person_walk`, both TartanAir
clips, and `vkitti_drive_clone`). Close action, close person, and rainy driving were only `-6%` to
`-8%`; the effect is scene-dependent rather than a universal gain scale. VD3D improved one
validated stretch event, so the overall verdict remains a geometry tradeoff until artifact quality
is compared along a matched-volume curve.

The manifest is [datasets/manifest.json](datasets/manifest.json). Bonn derivatives remain local;
its official page requests citation but does not provide a redistribution grant. TartanAir V2 is
CC BY 4.0. `prepare_public_datasets.py` associates Bonn RGB/depth by nearest timestamp, preserves
TartanAir float depth without quantization, decodes its 16-bit flow PNG losslessly without OpenCV,
and writes source frames only at the clip root so reference images can never be mistaken for input.
Set `APOLLO_SBS_DATASETS` or pass `--cache` to relocate the cache.

The harness sizes the SBS output to the **input** resolution, so these small clips make a full 5-clip A/B take
seconds (≈8 s harness + 3 s scoring per clip) instead of a minute:

```
cd cmake-build-relwithdebinfo
./sunshine.exe E:/ApolloDev/config/sunshine.conf --sbs-bench \
    --frames ../tools/sbsbench/clips/c525 --out out/c525 \
    --model depth_anything_v2_fp16
python tools/sbsbench/sbsbench.py --seq out/c525 --frames tools/sbsbench/clips/c525 --json base.json
python tools/sbsbench/sbsbench.py --seq out/NEW  --frames tools/sbsbench/clips/c525 --baseline base.json
```

**Eval never resizes the input** — the SBS output tracks the input size, so to run a *full-res*
(slower, more sensitive) eval, just point `--frames` at a full-resolution frame dir; use `--eye-h`
only to pin a specific output height. Metric values are resolution-dependent, so keep baselines
per clip-set (a small-clip baseline isn't comparable to a full-res one; A/B deltas within one are).

### Adding / regenerating clips
```
python tools/sbsbench/split_video.py clip.mp4 -o tools/sbsbench/clips/mine --width 854 --jpg --max 24
```
Drop `--width/--jpg` for a full-resolution PNG clip. The gated runner uses current-frame depth
with one update per source frame.

## Local VD3D Bestv2 reference (media stays local)

`bestv2-phase-a.conf` pins the reproduction candidate. `vd3d_reference.py prepare` extracts and
hashes the source/Bestv2 render, restores original frame identities, verifies alignment, and keeps
all source frames so sampling cannot change history. The harness exports exact raw model floats and
the finalized depth texture immediately before reprojection. `export_vd3d_depth_reference.py`
produces matching VD3D checkpoints; `vd3d_reference.py score` gates the two depth stages separately
from final-warp reproduction. Pixel similarity is never treated as a warp-quality verdict.

Phase-B quality tuning uses `bestv2-apollo-warp.conf` and `bestv2-vd3d-warp.conf`. A processor may
be retained for one geometry and rejected for the other. Use `--report-allow-config-diff` when
generating their comparison report: it permits the intentional config-hash difference while still
requiring identical clips, mode, model, eval schema, depth step and metric definition.

The metrics split cleanly by subsystem: **warp**-side changes move pop / disocc / flicker_disocc;
**depth**-side changes move edge_acc / swim / depth_spread. So a delta tells you *where* the change
landed. (Validated: a 2× `--divergence` warp change moved pop +90% and left edge_acc/swim flat;
swapping da3mono→v2 moved edge_acc −96% and swim −100% and left the warp lever untouched.)

## Artifact score (0–100) and feature decision
`sbs_score(agg)` reports `score = q_clean` = 100 − weighted artifact penalties (each
`weight × min(value/scale, 1)`, saturating). `q_depth` remains a separate diagnostic of delivered
stereo volume (`pop_spread_pct` versus its target); it is not blended into score, because losing
depth must not buy artifact-quality points or cancel an artifact regression. Weights/scales live
in [thresholds.json](thresholds.json) `"score"`.

The score is a summary, never the feature verdict. Each metric declares a decision role in
`thresholds.json`: `hard` comfort/integrity constraints cannot be traded; validated `primary`
metrics vote inside named coequal stereo, warp, stability, and depth axes; unvalidated `diagnostic`
proxies remain visible but cannot accept or reject a feature. Cross-axis movement is reported as a
tradeoff rather than cancelled inside a scalar. Expected-flat clips remain visible as false-stereo
diagnostics but do not vote on general-content features.

`source_residual_p95` is a validated broad warp-axis metric. For each eye it finds the closest
source patch within the allowed horizontal disparity radius, then reports the worse eye's p95
residual. Intended stereo displacement is free; holes, blur, ringing, duplication and stretched
texture rise. Its visual card shows source/control/treatment plus a signed residual-delta mask.

`source_halo_p95` and `source_stretch_pct` specialize that alignment at depth silhouettes. Halo
subtracts thin-ridge energy already present in the source, so genuine bright outlines are free.
Stretch counts source-textured silhouette-near pixels whose horizontal detail collapses below 35%
of the aligned source. They replace the output-only rim/stretch proxies in decisions and score;
the old detectors remain diagnostics for exceptionally large changes.

Hard constraints cover both signed p99 disparity tails (3% of eye width), vertical mismatch,
source coverage (>=90%), and source-texture integrity (>=80%). Signed tails are deliberately not
called crossed/uncrossed: converting host pixels to angular vergence requires headset FOV and
display calibration that the harness does not possess.

`static_jitter_p95` is the validated stability-axis metric. It excludes every source pixel that
moved, expands that exclusion horizontally by the allowed disparity radius, and measures the
worse eye's p95 output change only on the remaining static support. Scene cuts/camera moves with
less than 10% support do not vote. Its evidence card shows the evaluated mask, each run's temporal
change and a signed red/blue treatment delta.

`flow_temporal_p95` extends stability validation to moving content. Overlapping source tiles are
phase-correlated into a dense classical optical-flow field; the previous output is warped into the
current frame and only photometrically reliable support votes. `flow_depth_p95` applies the same
compensation to pre-warp depth as a diagnostic. This avoids counting intended motion as flicker
without requiring an AI flow model.

On GT clips, `depth_gt_si_rmse` affine-aligns predicted relative disparity to ground-truth inverse
depth (shift-only for constant GT) and `depth_gt_edge_f1` validates boundaries with one-pixel
tolerance. Both are primary depth-axis metrics. Non-GT clips are reported as `n/a`.

`rescore_run.py` refreshes a comparison-only run directly from its preserved source/depth/SBS
artifacts after metric-code changes. It refuses committed-baseline verdicts, updates the metric
contract hash and writes atomically; use `run_eval.py` for any committed gate.

Each clip directory carries a `meta.json` (`{"name", "description"}`): the report labels clips by
that scene name and run_eval copies it into results.json. The clip identity hash covers source and
GT pixels plus scoring semantics, but excludes the human-readable name and description.

## Metrics — spatial (per frame)
| metric | meaning | direction |
|--------|---------|-----------|
| `pop_px_p50` / `p95` | L↔R horizontal disparity (tile phase-correlation), median & p95 of \|dx\|. REPORTED but NOT gated — subject anchoring legitimately lowers median \|dx\| | higher = more 3D pop |
| `pop_pct_p50` | same as % of eye width | higher = more pop |
| `pop_spread_px` / `pop_spread_pct` | near-to-far disparity RANGE = weighted p95−p5 of **signed** dx (px / % of eye width). The stereo VOLUME, invariant to where the zero-parallax plane sits — the **gated** pop metric and the `q_depth` driver (subject-mode-fair; also ~0 for a flat scene shifted bodily forward) | higher = more volume |
| `vmisalign_px` | median vertical L↔R offset — must be ~0 | nonzero = geometry fault |
| `positive_disparity_pct` / `negative_disparity_pct` | signed weighted p99 disparity tails as % eye width | each must remain ≤3% |
| `source_coverage_pct` | output patches explainable by horizontally displaced source content | ≥90% hard integrity limit |
| `image_integrity_pct` | retention of real source texture after alignment | ≥80% hard integrity limit |
| `depth_spread` | p95−p5 of the normalized depth = pop available at the source | separates flat-model from flat-warp |
| `disocc_frac` | fraction of the eye in a band beside a real depth silhouette | context for smear (how much was invented) |
| `disocc_smear` | horizontal-detail deficit in the narrow band: 1 − \|dI/dx\|<sub>band</sub>/\|dI/dx\|<sub>clean</sub> | 0 = clean fill · →1 = smeared (small-scale) |
| `stretch_area` | the LARGE horizontal disocclusion **stretch band** (bg rubber-banded to fill the gap; eye-asymmetric): area of wide low-horizontal-gradient / vertically-streaked runs anchored to silhouettes, per-mille of the eye | higher = more/bigger smeared patches. Ignores smooth (textureless) stretches |
| `rim_over_p50` / `p95` | silhouette **halo / white line**: a thin bright ridge hugging the silhouette (horizontal white top-hat of the eye, sampled in the silhouette band, ×255) — the residual bright sliver where the fill doesn't reach the fg edge | ~0 = no fringe · higher = brighter white line. Ignores broad bright regions (top-hat is thin-ridge specific) |
| `source_halo_p95` | excess silhouette ridge after subtracting the aligned source ridge | lower = less warp-created halo; **primary warp axis** |
| `source_stretch_pct` | source-textured silhouette-near pixels with >65% horizontal-detail loss | lower = less warp-created stretch; **primary warp axis** |
| `edge_acc_p50` / `p95` | depth-px distance from each depth silhouette to the nearest **source** color edge (needs `--frames`) | small = silhouette on the real edge · large = soft/bent/floating |

## Metrics — temporal (`--seq` on a harness clip)
| metric | meaning | direction |
|--------|---------|-----------|
| `static_jitter_p50` / `p95` | worse-eye output change over source-static support after disparity-radius motion exclusion | lower = steadier; **primary stability axis** |
| `flow_temporal_p50` / `p95` | source-flow-compensated output residual on photometrically reliable support | lower = steadier moving content; **primary stability axis** |
| `flow_depth_p50` / `p95` | source-flow-compensated pre-warp depth residual | lower = steadier depth; diagnostic |
| `flicker` | frame-to-frame mean\|Δ\| of the SBS luma (×255) | diagnostic only; includes normal motion |
| `flicker_disocc` | unregistered frame difference restricted to the current depth-silhouette band | diagnostic only; motion-confounded |
| `swim` | frame-to-frame \|depth change\| where the **source** is static (needs `--frames`) | diagnostic until support/locality handling is upgraded |

Notes:
- `vmisalign_px == 0` is the built-in correctness check (parallax must be horizontal-only).
- Flat/paused frames legitimately score `pop=0` — curate scenes with real depth.
- A matched clip does not make raw flicker perceptual: a processor can move the sampling mask or
  disparity and change how legitimate motion is counted. Only motion-excluded `static_jitter` votes.
- Temporal metrics need a multi-frame clip; the offline sim (single-frame) cannot produce them.
- `--frames <input dir>` unlocks `edge_acc` and `swim` (they compare against the source frames).
- Harness depth is dumped as 16-bit grayscale so sub-1/255 temporal changes remain measurable.

## Metrics — ground-truth depth (clips with `gt_depth/frame_*.png`)
| metric | meaning | direction |
|--------|---------|-----------|
| `depth_gt_si_rmse` | relative-disparity RMSE after monocular scale/shift alignment; shift-only on flat GT | lower = more accurate; **primary depth axis** |
| `depth_gt_edge_f1` | boundary F1 with one-pixel tolerance | higher = better boundaries; **primary depth axis** |

## Not yet (roadmap)
- **Ghost** — a lag-band metric on a known-motion clip (double-image energy).
- **Reference warp PSNR/SSIM** — still needs ground-truth stereo content (rendered second eye).
