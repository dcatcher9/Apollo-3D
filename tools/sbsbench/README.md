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
```

Harness A/B levers (after `--extra`):
- `--divergence F` — parallax gain.
- `--depth-short-side N` — depth inference short side (default 432; VD3D parity). 336 to A/B
  back to the old under-resolved default.
- `--pct-lo F --pct-hi F` — robust percentile normalization bounds, e.g. `1 99` (default off =
  raw min/max).
- `--ema F` — per-pixel depth EMA override (`1.0` = off).
- `--lock-frames N` — scene-locked normalization bounds (bench-falsified for DA-V2; see the
  config.h note on `sbs_3d_norm_lock_frames`).
- `--subject-track` — VD3D-style shaped disparity (subject-anchored band curve). The pipeline
  is probe-reprojection-only, so the shaping is always live when this is on.
- `--subject-lock F` — subject anchor strength (e.g. `0.95`).
- `--subject-stretch` — shape_depth_for_pop 5/95 percentile stretch (default on within the
  subject path).
- `--subject-plane-lock F` — local subject-band flatten (e.g. `0.28`; default off).
- `--curvature F` — foreground-curvature bulge strength (e.g. `0.07`; reshapes the depth
  texture, both warp paths; default off).
- `--dof F` — depth-of-field blur radius as a fraction of width (e.g. `0.008`; post-warp,
  probe path only; default off).

Exit code is the verdict (0 pass / 1 regression / 2 setup error), so the eval→fix→eval loop is
scriptable. `results.json` carries provenance (git sha+dirty, models, clip hashes, gpu-contention
flag) and, for every triggered/regressed metric, the **worst frame index** to look at. The gate
thresholds live in [thresholds.json](thresholds.json); the pinned SBS config in
[bench.conf](bench.conf); baselines in `baselines/` (regenerate in the same commit whenever
bench.conf, the clip set, or a metric definition changes). Guards: fails fast if TRT engines
aren't prebuilt; warns + skips the perf gate if another sunshine.exe is running.

Metric notes: silhouette detection runs at the native depth resolution with an absolute
depth-step floor (flat scenes legitimately read 0), and all pixel windows scale with the eye
width — but absolute values are still not comparable across clip resolutions; baselines are
per-clip-set. The harness writes 16-bit depth PNGs so `swim` resolves below 1/255.

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
| `scene_cut` | depth-normalization swim across a hard cut (A1) | swim/flicker worst frame lands exactly on the cut (frame 12; swim 3.2 → 41.6 there) |
| `flat_page` | flat-content depth hallucination + amplification (A3) | pop ≈ 0, flicker = 0 (static input = pipeline noise floor); disocc_smear 1.0 flags hallucinated text-edge silhouettes |
| `fast_motion` | async-depth ghost (known 30 px/frame motion) | highest swim of the set (9.0) + trailing-disocclusion smear | The harness sizes
the SBS output to the **input** resolution, so these small clips make a full 5-clip A/B take
seconds (≈8 s harness + 3 s scoring per clip) instead of a minute:

```
cd cmake-build-relwithdebinfo
./sunshine.exe E:/ApolloDev/config/sunshine.conf --sbs-bench \
    --frames ../tools/sbsbench/clips/c525 --out out/c525 --movie
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
Drop `--width/--jpg` for a full-resolution PNG clip. The harness settles the async depth per frame
(`--settle`, default 3) so each output uses depth caught up to its own frame.

The metrics split cleanly by subsystem: **warp**-side changes move pop / disocc / flicker_disocc;
**depth**-side changes move edge_acc / swim / depth_spread. So a delta tells you *where* the change
landed. (Validated: a 2× `--divergence` warp change moved pop +90% and left edge_acc/swim flat;
swapping da3mono→v2 moved edge_acc −96% and swim −100% and left the warp lever untouched.)

## Overall score (0–100)
`sbs_score(agg)` collapses the metric vector into one number per clip: `q_clean` = 100 − weighted
artifact penalties (each `weight × min(value/scale, 1)`, saturating), `q_depth` = realized stereo
(`pop_spread_pct` vs a target — the near-to-far VOLUME, not median \|dx\|, so subject anchoring is
scored on delivered depth; content-dependent, so a flat scene scores low), blended
`score = (1−w)·q_clean + w·q_depth`. Weights/scales live in [thresholds.json](thresholds.json)
`"score"` (retuning them also reorders the report, which sorts metrics by quality impact). The
score is gated by run_eval like any metric (a >1.5-point drop = regression) and best used to rank
configs on the SAME clips; the per-metric numbers stay the source of truth. Set
`score.depth.weight` to 0 for a purely artifact-based (content-independent) score.

Each clip directory carries a `meta.json` (`{"name", "description"}`): the report labels clips by
that scene name and run_eval copies it into results.json. The clip identity hash covers only the
`frame_*` files, so renaming a scene never invalidates its baseline.

## Metrics — spatial (per frame)
| metric | meaning | direction |
|--------|---------|-----------|
| `pop_px_p50` / `p95` | L↔R horizontal disparity (tile phase-correlation), median & p95 of \|dx\|. REPORTED but NOT gated — subject anchoring legitimately lowers median \|dx\| | higher = more 3D pop |
| `pop_pct_p50` | same as % of eye width | higher = more pop |
| `pop_spread_px` / `pop_spread_pct` | near-to-far disparity RANGE = weighted p95−p5 of **signed** dx (px / % of eye width). The stereo VOLUME, invariant to where the zero-parallax plane sits — the **gated** pop metric and the `q_depth` driver (subject-mode-fair; also ~0 for a flat scene shifted bodily forward) | higher = more volume |
| `vmisalign_px` | median vertical L↔R offset — must be ~0 | nonzero = geometry fault |
| `depth_spread` | p95−p5 of the normalized depth = pop available at the source | separates flat-model from flat-warp |
| `disocc_frac` | fraction of the eye in a band beside a real depth silhouette | context for smear (how much was invented) |
| `disocc_smear` | horizontal-detail deficit in the narrow band: 1 − \|dI/dx\|<sub>band</sub>/\|dI/dx\|<sub>clean</sub> | 0 = clean fill · →1 = smeared (small-scale) |
| `stretch_area` | the LARGE horizontal disocclusion **stretch band** (bg rubber-banded to fill the gap; eye-asymmetric): area of wide low-horizontal-gradient / vertically-streaked runs anchored to silhouettes, per-mille of the eye | higher = more/bigger smeared patches. Ignores smooth (textureless) stretches |
| `rim_over_p50` / `p95` | silhouette **halo / white line**: a thin bright ridge hugging the silhouette (horizontal white top-hat of the eye, sampled in the silhouette band, ×255) — the residual bright sliver where the fill doesn't reach the fg edge | ~0 = no fringe · higher = brighter white line. Ignores broad bright regions (top-hat is thin-ridge specific) |
| `edge_acc_p50` / `p95` | depth-px distance from each depth silhouette to the nearest **source** color edge (needs `--frames`) | small = silhouette on the real edge · large = soft/bent/floating |

## Metrics — temporal (`--seq` on a harness clip)
| metric | meaning | direction |
|--------|---------|-----------|
| `flicker` | frame-to-frame mean\|Δ\| of the SBS luma (×255) | lower = steadier |
| `flicker_disocc` | flicker restricted to the disocclusion bands — isolates inpaint/stretch re-hallucination from ordinary motion | lower = less shimmer where it matters (runs ~2–3× frame flicker) |
| `swim` | frame-to-frame \|depth change\| where the **source** is static (needs `--frames`) — scene-cut / flat-content depth instability, separated from real motion | lower = steadier depth |

Notes:
- `vmisalign_px == 0` is the built-in correctness check (parallax must be horizontal-only).
- Flat/paused frames legitimately score `pop=0` — curate scenes with real depth.
- On the SAME clip real motion cancels in the `--baseline` diff, so the temporal deltas are the
  change alone. Temporal metrics need a multi-frame clip; the offline sim (single-frame) can't produce them.
- `--frames <input dir>` unlocks `edge_acc` and `swim` (they compare against the source frames).
- The dumped depth is 8-bit, so `swim` has a ~1/255 quantization floor; dump 16-bit depth if you need finer.

## Not yet (roadmap)
- **Ghost** — a lag-band metric on a known-motion clip (double-image energy).
- **Reference PSNR/SSIM** — needs ground-truth stereo content (rendered second eye).
- **Depth-based metrics in `--seq`** — the harness could also dump `depth_%05d.png` to enable
  `depth_spread` / `stretch_band` per sequence frame (currently spatial-only in seq mode).
