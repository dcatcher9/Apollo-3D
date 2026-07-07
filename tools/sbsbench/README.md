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

## Deterministic clips via the headless harness (recommended)
Single dumps are sporadic and headset-bound. For repeatable A/B and **temporal** metrics, drive
the real pipeline over a fixed frame sequence with the built-in `--sbs-bench` subcommand (Tier-1
harness — runs the real estimator + real composite shaders, no game/client):

```
# 1. video -> frames (bundled ffmpeg via imageio-ffmpeg)
python tools/sbsbench/split_video.py clip.mp4 -o E:/ApolloDev/sbs_bench/clips/movie --fps 24

# 2. frames -> real SBS frames (run from the build dir so assets/ resolves; conf supplies the
#    warp/divergence config; --movie applies the movie warp+depth overrides)
cd cmake-build-relwithdebinfo
./sunshine.exe E:/ApolloDev/config/sunshine.conf --sbs-bench \
    --frames E:/ApolloDev/sbs_bench/clips/movie --out E:/ApolloDev/sbs_bench/out/movie --movie

# 3. score the sequence (adds temporal flicker) + baseline diff
python tools/sbsbench/sbsbench.py --seq E:/ApolloDev/sbs_bench/out/movie --json base.json
python tools/sbsbench/sbsbench.py --seq NEW_OUT --baseline base.json
```

The harness settles the async depth per frame (`--settle`, default 3) so each output uses depth
caught up to its own frame. Same input frames every run → a metric delta is your change alone.

## Metrics (single frame; spatial)
| metric | meaning | direction |
|--------|---------|-----------|
| `pop_px_p50` / `p95` | L↔R horizontal disparity (tile phase-correlation), median & p95 | higher = more 3D pop |
| `pop_pct_p50` | same as % of eye width | higher = more pop |
| `vmisalign_px` | median vertical L↔R offset — must be ~0 | nonzero = geometry fault |
| `depth_spread` | p95−p5 of the normalized depth map = pop available at the source | separates flat-model from flat-warp |
| `stretch_band` | **experimental** proxy: excess horizontal smoothness in the band beside strong depth edges (the disocclusion smear) | higher = more visible band |

Notes:
- `vmisalign_px == 0` across frames is the built-in correctness check (parallax must be
  horizontal-only); if it ever goes nonzero, a geometry regression slipped in.
- Flat/paused frames legitimately score `pop=0` (no parallax) — curate scenes with real depth.
- `stretch_band` is a heuristic; trust it once you've confirmed it moves on a case you can see.
  Refine it against a known-bad silhouette before leaning on the absolute value.

## Metric (temporal; `--seq` on a harness clip)
| metric | meaning | direction |
|--------|---------|-----------|
| `flicker` | frame-to-frame mean\|Δ\| of the SBS luma (×255) | lower = steadier; on the SAME clip the shared motion cancels in the baseline diff, isolating added shimmer (e.g. ¼-res inpaint re-hallucination) |

This is the metric the offline sim structurally cannot produce (it is single-frame). Validated:
a static clip (one frame repeated) converges toward flicker≈0 as the async depth EMA settles.

## Not yet (roadmap)
- **Ghost** — a lag-band metric on a known-motion clip (double-image energy).
- **Reference PSNR/SSIM** — needs ground-truth stereo content (rendered second eye).
- **Depth-based metrics in `--seq`** — the harness could also dump `depth_%05d.png` to enable
  `depth_spread` / `stretch_band` per sequence frame (currently spatial-only in seq mode).
