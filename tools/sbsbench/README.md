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

The metrics split cleanly by subsystem: **warp**-side changes move pop / disocc / flicker_disocc;
**depth**-side changes move edge_acc / swim / depth_spread. So a delta tells you *where* the change
landed. (Validated: a 2× `--divergence` warp change moved pop +90% and left edge_acc/swim flat;
swapping da3mono→v2 moved edge_acc −96% and swim −100% and left the warp lever untouched.)

## Metrics — spatial (per frame)
| metric | meaning | direction |
|--------|---------|-----------|
| `pop_px_p50` / `p95` | L↔R horizontal disparity (tile phase-correlation), median & p95 | higher = more 3D pop |
| `pop_pct_p50` | same as % of eye width | higher = more pop |
| `vmisalign_px` | median vertical L↔R offset — must be ~0 | nonzero = geometry fault |
| `depth_spread` | p95−p5 of the normalized depth = pop available at the source | separates flat-model from flat-warp |
| `disocc_frac` | fraction of the eye in a band beside a real depth silhouette | context for smear (how much was invented) |
| `disocc_smear` | horizontal-detail deficit in those bands: 1 − \|dI/dx\|<sub>band</sub>/\|dI/dx\|<sub>clean</sub> | 0 = clean fill · →1 = stretched/smeared |
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
