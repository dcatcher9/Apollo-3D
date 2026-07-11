# Apollo ↔ VisionDepth3D stage-by-stage parity

Working doc for the `sbs-vd3d-pipeline` branch. Goal, in the user's words: **reproduce
VisionDepth3D's result faithfully first, then integrate Apollo's improvements one at a time,
each gated by the eval harness.** VD3D (`E:\Git\Repo\VisionDepth3D`, offline converter, DA-V2
small, preset **Bestv2**) is the reference; Apollo is the thing we tune to match, then beat.

Companion to the validated findings in the sbs-vd3d-port memory and the roadmap in
[sbs-3d-roadmap.md](sbs-3d-roadmap.md). VD3D is proprietary (No-Derivatives) — we reimplement its
**math**, never copy its code.

## Method: two phases, eval-gated

1. **Phase A — faithful VD3D baseline.** Configure/port Apollo so each step matches Bestv2 as
   closely as its architecture allows; turn *off* every Apollo-specific addition VD3D lacks
   (guided upsample, range-floor, …). Commit as a named bench config + baseline.
2. **Phase B — integrate Apollo improvements.** Re-enable Apollo's additions one lever per eval
   run (`run_eval.py --extra …`), keep only what the numbers say beats Phase A, diff every step.

## Phase-B per-warp quality profiles

Phase B starts from two independent profiles rather than forcing the same processors onto both
geometry implementations:

- `tools/sbsbench/bestv2-apollo-warp.conf`
- `tools/sbsbench/bestv2-vd3d-warp.conf`

Both currently retain the accepted Bestv2 shift, subject tracking, recentering and stretch. The
quality profile intentionally disables exact sharpen on both warps: despite being essential to
Phase-A pixel reproduction, it reduced the independent score by `6.20` / `6.46` points and caused
large bright-rim regressions. Window sculpt, curvature, conceal/repair and DOF were also rejected
and remain off. Exact plane lock is rejected for both profiles by a final-stack off/on ablation:
Apollo lost `0.59` score and VD3D lost `1.06`; pop spread fell `0.63` / `0.66 px` and bright-rim
p95 worsened `0.90` / `2.01`. Visual reports are under `sbs_eval/bestv2-profile-plane-apollo`
and `sbs_eval/bestv2-profile-plane-vd3d`.

Guided upsample is rejected for both warps. Its localized rim/stretch wins originally raised the
old blended mean score for VD3D, but that conclusion was an eval error: the gain was dominated by
one scene while edge alignment regressed broadly. Under the corrected artifact-only score and
per-clip source-metric gate, Apollo has 15 regressions across all 7 ordinary clips and VD3D has 7
across 6 clips. Visual reports are under `sbs_eval/bestv2-profile-guided-apollo` and
`sbs_eval/bestv2-profile-guided-vd3d`. Min/max snap and range floor remain pending and must be
evaluated independently for each warp.

The starting profile comparison (flat-page clip excluded) is Apollo `74.13` versus VD3D `73.46`.
Apollo has `+0.67 px` pop spread and `0.84` lower bright-rim p95; VD3D has `0.23` lower stretch and
is about `8x` faster in the warp (`0.018 ms` versus `0.143 ms`). This is a starting report, not a
final warp decision: `cmake-build-relwithdebinfo/sbs_eval/bestv2-profile-comparison/report.html`.

## Validated status — Bestv2 shift calibration (2026-07-10)

The original Phase-A candidate used fixed normalized divergence `0.00285`. That was only fitted to
one resolution and was not the Bestv2 formula. `sbs_3d_shift_profile = bestv2` now evaluates the
preset in source-pixel units for both warp implementations:

- FG/MG/BG shifts `-9/-3/+2.4`, multipliers `1.11/1/1.05`, parallax balance `.35`;
- subject anchor `.95`, zero-parallax trim `.008`;
- dynamic convergence `.006` with its `.90` EMA;
- maximum shift cap `.071`.

Fresh comparison-only runs used the identical `bestv2-phase-a.conf` and verified every raw-model
and pre-warp depth artifact byte-identical across control/treatment and Apollo/VD3D geometry. Mean
movement across the committed clip set:

| geometry | score | pop p50 | pop spread | bright rim | disocc flicker |
|----------|------:|--------:|-----------:|-----------:|---------------:|
| Apollo, provisional → Bestv2 | `+2.88` | `+205%` | `+89%` | `-43%` | `+50%` |
| VD3D, provisional → Bestv2 | `+2.21` | `+226%` | `+67%` | `-38%` | `+46%` |

Against the aligned real Bestv2 render, pixel reproduction improved from MAE `0.0961` / PSNR
`16.1 dB` to MAE `0.0614` / PSNR `20.9 dB`. The remaining disparity-spread mismatch (`19.90 px`
Apollo versus `10.57 px` reference) is not hidden: the core pixel formula is now calibrated, but
Bestv2's curvature, cinematic window sculpt, silhouette-aware plane lock, smear suppression and
Fast repair are still absent or approximate. Those stages can change the measured final field.

With the exact field held constant, Apollo geometry scores `74.23` versus VD3D `73.74`; VD3D is
about `6.1×` faster in the warp (`0.0171 ms` versus `0.1036 ms`) and reduces stretch slightly,
while Apollo has lower bright rims. This remains a tradeoff, not a warp winner.

## Timeline of the pipeline

The pipeline is two cadence groups, not a flat list. Steps ordered by execution:

| # | step | cadence | shader / site | Apollo-only? |
|---|------|---------|---------------|--------------|
| **A. Depth production** | | *runs when a new async inference lands (`depth_fps`)* | | |
| A1 | preprocess (rgb→nchw, resize to depth res) | depth-tick | `rgb_to_nchw` | — |
| A2 | depth inference | depth-tick | TensorRT DA-V2 | — |
| A3 | per-model output transform | depth-tick | `buffer_to_tex_cs` / `depth_minmax_cs` | no |
| A4 | range normalize (min/max reduce + EMA + map) | depth-tick | `depth_minmax*_cs` → `buffer_to_tex_cs` | mechanism differs |
| A5 | range floor (flat-scene contrast guard) | depth-tick | `buffer_to_tex_cs:31` | ✅ |
| A6 | temporal per-pixel EMA | depth-tick | `buffer_to_tex_cs:37` | no |
| A7 | foreground curvature *(guided-off path)* | depth-tick | `depth_curvature_cs` | no |
| A8 | subject tracking (hist → resolve → `SubjectState`) | depth-tick | `depth_subject_*_cs` | no |
| **B. Frame synthesis** | | *runs every output frame* | | |
| B1 | guided upsample (+ curvature, guided-on path) | per-frame | `depth_guided_upsample_cs` + `depth_curvature_cs` | ✅ |
| B2 | warp incl. shaping (depth-floor, subject recenter/stretch/band-curve/plane-lock, probe search) | per-frame | `sbs_reprojection_ps.hlsl` | mechanism differs |
| B3 | disocclusion concealment | per-frame | evaluated, rejected, removed | VD3D-only; no measured gain |
| B4 | post (DOF, heal, sharpen) | per-frame | exact SDR sharpen retained for parity | DOF rejected; heal helper is not called |

Two timeline facts that drive everything:
- **Async ordering.** In async mode A-group consumes the *previous* frame's inference, so B-group
  warps current color against slightly stale depth (the lag). B1 re-snaps that stale depth to the
  *current* frame's color edges every frame, which is why silhouettes track motion between ticks.
- **Cadence split.** A runs at `depth_fps` (≈ every Nth frame); B runs every frame. VD3D offline
  runs its whole chain per frame — so Apollo's per-frame vs depth-tick split is itself a divergence
  from VD3D, and it's exactly why the guided upsample (B1) and shaping (B2) are per-frame while
  normalization (A4) is not.

Per the user we lock **Group A (the depth half)** first — it feeds everything downstream and is
where Apollo transforms depth in more places than "min/max EMA."

---

## Group A — Depth production

### A1–A2 · Preprocess & inference

| aspect | VD3D (Bestv2) | Apollo | faithful? |
|--------|---------------|--------|-----------|
| model | DA-V2 small fp16 (PyTorch) | DA-V2 small fp16 (TensorRT) | ✅ same weights |
| resolution | UI request 768×432; adapter snaps to patch-14 **770×434** | short-side 432 resolves to **770×434** at this aspect | ✅ matched. The old 336 claim confused DA-V2 with DA3MONO's `dynamic_width/fixed_h=336` registry entry. Runtime logs and raw-shape artifacts verify 770×434. |
| cadence | per-frame | **async** at `depth_fps` (45 game) — depth lags color | ⚠️ **the** depth-half gap |

**Live sync is gone.** `sbs_3d_sync_depth` and the forked synchronous `estimate()` were removed
(commit 5e7530fe). **Both GAME and MOVIE run async today** (MOVIE maps onto GAME's async path;
true synchronous MOVIE is roadmap #3, unbuilt). Async is the only behavior; the lag is a permanent
architectural property, not a toggle. The evaluator now uses a benchmark-only one-submit,
one-synchronize, one-normalize operation without duplicate enqueue. This is an evaluation
primitive, not a live mode. Real per-frame sync only pays off for
MOVIE's heavy DA3MONO, where it stalls the encode thread (a wash).

**Faithful-baseline decision:** evaluate current-frame depth with exactly one state update per
source frame. Live async cadence is a later Apollo-specific A/B; mixing it into Phase A caused the
former duplicate-settle conclusions.

### A3–A8 · The six depth transforms

My first pass wrongly listed only "min/max EMA." Apollo transforms the depth map in **six** places
before the warp, and **the guided upsample (B1) is a seventh** — a joint-bilateral filter *changes
depth values*, it is not a resize. Group-A transforms, in order:

| # | site | what it does | VD3D counterpart | Apollo-only? |
|---|------|--------------|------------------|--------------|
| A3 | per-model transform (`buffer_to_tex_cs:18`, `depth_minmax_cs`) | DA-V2: identity. DA-V3: `1/(raw+depth_shift)` → disparity | `depth_to_tensor(invert_depth=True)` | no |
| A4 | range normalize (`buffer_to_tex_cs:25`) | `saturate((raw−min)/(max−min))`, min/max **EMA-smoothed** (`minmax_ema=0.1`) | **per-frame** `DepthPercentileEMA(p2/p98, α=0.82)` (live-portable analog) — see the two-normalization note below | percentile vs raw min/max |
| A5 | range floor (`buffer_to_tex_cs:31`) | `0.5+(mapped−0.5)·range_scale`: compress contrast toward 0.5 on near-flat scenes | **none** | ✅ (A3 anti-hallucination guard) |
| A6 | temporal EMA (`buffer_to_tex_cs:37`) | `lerp(old, mapped, ema_alpha)`, `ema=0.6` | `TemporalDepthFilter(α=0.5)` | no |
| A7 | foreground curvature (`depth_curvature_cs`) | elliptical `pow(1.35)` bulge on `d>0.60`; guided-off path runs it here, guided-on runs it in B1 (fixed this session) | `enhance_foreground_curvature(0.07)` | no (ported; default-off) |
| A8 | subject tracking (`depth_subject_hist/resolve_cs`) | weighted-hist 35%-from-near → `SubjectState` (EMA α=0.80, lock 0.95); consumed at warp time (B2) | subject estimate + `SubjectDepthEMA(0.80)` | no (number-match validated: 0.408 vs 0.391) |

### A4 detail — VD3D normalizes depth *twice*

VD3D's offline flow composes two percentile normalizations; only one is live-portable:

1. **Depth-gen** (`render_depth.py:4104`, `FixedPercentileNormalizer(2,98)`): a **sparse
   bootstrap** — samples only `min(5, max(3, N/300))` = **3–5 frames** (`np.linspace(0, N−1, …)`,
   spread across the clip), `learn()` accumulates their min-lo/max-hi, then `lock()` **freezes**
   that range and applies it to every frame. Logged once as `🔒 Depth normalizer locked with
   range: lo=… hi=…` (source of the `[0, 5.3789]` in the port memory). It is **not** whole-clip
   statistics — just a frozen range from a handful of frames. Offline as written (linspace needs
   the last frame), but the idea (a few frames → freeze) is live-approximable.
2. **Render** (`render_3d.py:1479`, `DepthPercentileEMA(2,98, α=0.82)`): recomputes p2/p98
   **per frame** and EMA-smooths them (new-weight 0.18). Runs on every frame — this is the
   "frame-by-frame" you see in the render log.

The per-frame inference in the depth-gen log is *inference*, not per-frame normalization; the
range there is the frozen bootstrap value. So "generated frame by frame" and "frozen range" are
both true.

**For Apollo (single-stage, live) the analog is the render stage's per-frame percentile EMA.**
The reference exporter reproduces VD3D's non-causal bootstrap exactly for checkpoint comparison.
The live Apollo path uses **percentile p2/p98 + per-frame `minmax_ema`**; it does not expose a
causal first-N approximation as though it were the same algorithm.

### Faithful-baseline depth config (Group A)

| knob | Apollo default | Phase-A faithful | rationale |
|------|----------------|------------------|-----------|
| `sbs_3d_depth_short_side` | 432 | **432** | already parity |
| `sbs_3d_norm_pct_lo`/`_hi` | **2 / 98** (was 0/100 raw) | **2 / 98** | Direct `DepthPercentileEMA(p2/p98)` match. The min/max reduction remains because it defines the histogram range. Earlier score deltas used the superseded eval contract and are not evidence here. |
| `sbs_3d_minmax_ema` | 0.1 (α≈0.9) | **0.18** | Direct VD3D render-stage match. The earlier temporal conclusion used duplicate settle updates and is invalid for Phase A. |
| `sbs_3d_ema` (per-pixel) | 0.6 | **0.5** | VD3D `TemporalDepthFilter(α=0.5)` |
| `sbs_3d_ema_pixel_first` | false | **true** | Matches temporal-before-render-percentile ordering. The config parser now exposes this key. |
| `sbs_3d_range_floor` | 0 (off) | **off** | Apollo-only; VD3D lacks it. Was pinned 0.5 in bench.conf (so the eval ran it); **removed 2026-07-10** so the eval matches the config.h default |
| `sbs_3d_guided_upsample` | **off** (was on) | **off** | VD3D has no equivalent — see B1. Re-enable later as an Apollo-only eval lever. |

Earlier A/B numbers in this section were produced by repeated `--settle` submissions and are not
valid temporal evidence. Eval schema 2 replaces them: one inference and one state update per source
frame, explicit DA-V2 pinning, identity-keyed artifacts, and full-history processing before output
sampling. Apollo's single-stage transform is not algebraically identical to VD3D's two-stage
bootstrap/video/render chain, but the direct checkpoint measurement below shows their resulting
physical depth maps are already within the Phase-A gate.

**Measured Phase-A depth result:** `tools/sbsbench/bestv2-phase-a.conf` processes all 198 frames
before sampling 25. Raw DA-V2 correlation mean/min = 0.99843/0.99808. Physical pre-warp depth
correlation mean/min = 0.99848/0.99818 and mean MAE = 0.02696 after accounting for polarity
(Apollo high-near; VD3D low-near). All depth-stage gates pass. Curvature stays off at this common
checkpoint because VD3D applies it inside `pixel_shift_cuda`; each warp variant applies 0.07 in
its own shaping stage.

---

## Group B — Frame synthesis

### B1 · Guided upsample *(Apollo-only — why it exists, and why it's off in the faithful baseline)*

The joint-bilateral upsample (`depth_guided_upsample_cs`) averages low-res depth weighted by
color-edge match, ×`(1+2·d)` **FG_BIAS** (pull ties toward the nearer surface) + a **bimodal cliff
snap** (×3 toward the leaning side across depth cliffs). It upscales 2× **and** relocates/steepens
silhouettes onto the full-res color edges — a depth transform, not a resize. I originally mis-filed
it as "resolution"; that was the error you caught.

**Why Apollo needs it (and VD3D doesn't):** Apollo runs depth at low res with soft, patch-scale
silhouettes and has **no learned warp and no concealment stack** — so the guided pass *is* its
edge-crispness mechanism, snapping depth silhouettes to color edges every frame (even on
cadence-skipped frames, re-snapping stale depth). VD3D runs native 768×432 and leans on its
forward-warp + concealment (B2/B3) for edges, so it never needs an upsample.

**Needed for a faithful VD3D reproduction? No** — VD3D has no equivalent, so Phase A sets
`guided_upsample=OFF`, and it becomes **Phase-B improvement #1**. With guided off and no
concealment yet, Apollo's silhouettes will be softer (expected) — which makes the
guided-vs-concealment question concrete and measurable instead of hand-waved.

> Entanglement: B1's FG_BIAS fights A7 curvature (why curvature now runs post-guided) and feeds the
> subject-depth **saturation** (subject pinned ~1.0, no headroom to sculpt). Isolating Group A lets
> us measure how much saturation is A4 (per-frame min/max pinning the nearest object) vs B1 (FG
> bias) — which the combined pipeline can't separate.

### B2 · Warp + shaping *(dual geometry implemented)*

Shaping is **not** a separate pre-warp stage; it runs inside the reprojection shader at warp time,
reading `SubjectState` (A8): depth-floor compression, subject recenter/stretch/band-curve/plane-lock
(all number-match validated), then the warp itself.

The warp is where Apollo and VD3D fundamentally differ:
- **VD3D:** `grid_sample` bilinear ×0.35 **+** depth-order forward warp ×0.65 + hole-fill
  (`render_3d.py:3335`).
- **Apollo:** occlusion-aware backward **probe** (frontmost-wins, nearest-bg fill).

Both paths are implemented behind `sbs_3d_warp` / evaluator `--warp`: (A) VD3D's Bestv2 35%
backward grid warp + 65% depth-ordered forward splat with 96-pixel directional hole fill, and
(B) Apollo's occlusion-aware probe. They receive the same saved pre-warp depth and disparity field.
The first A/B deliberately stops before concealment. Final-SBS similarity measures reproduction
only; artifact, temporal, comfort, and performance gates decide which warp is better.

**Bare-warp result (2026-07-10):** the Apollo baseline remains metric-stable. VD3D hybrid reduces
`stretch_area` slightly on the affected clips, but raises `rim_over_p95` on 6/8 clips and crosses
the regression gate on 5/8 because Bestv2's later repair is intentionally absent. At 3840×1080 on RTX 5080,
harness GPU timestamps measure Apollo-probe at 0.478 ms p50 versus VD3D hybrid at 0.051 ms p50.
The hybrid is therefore the performance leader. The shared concealment experiment below does not
change the quality ordering: Apollo remains slightly ahead on the aggregate exact-field score,
while the per-metric tradeoffs do not establish a decisive visual-quality winner.

**Rejected Bestv2 cinematic-window result (2026-07-10):** the treatment applied
VD3D's active window sculpt (`strength=.06`, subject hold `.406`, near pull `.0666`, width `.12`)
and computes the subject anchor from raw tracked-subject depth rather than the transformed pixel
depth. Raw-model and pre-warp artifacts are byte-identical to the control. Against the aligned
real Bestv2 output, MAE improves 0.061375 → 0.061302 and PSNR 20.878 → 20.902 dB. On the generic
corpus, stereo spread rises on both geometries and mean score is essentially flat/slightly higher
(Apollo +0.0875, VD3D +0.1000 under the expected-flat contract), while rim-over-p95 worsens
(+0.517/+0.845), and median pop falls on nearly every meaningful clip. The reference change is
too small to justify the consistent pop/rim cost. The treatment is rejected and removed; its
local HTML reports remain as evidence.

**Rejected exact fitted-curvature result (2026-07-10):** VD3D's actual three-pass operation was
ported for evaluation: separable 31×31 mask feathering, mask-weighted centroid/variance ellipse
fit, gamma 1.35, and strength 0.07. Because VD3D's render tensor is low-near despite the helper's
white-near docstring, the port mirrored the code's actual polarity rather than its semantic name.
Raw-model and saved pre-warp artifacts were byte-identical. Ordinary-scene score changed only
about +0.09 (Apollo) / +0.04 (VD3D); mean rim improved by only 0.050/0.074 and volume movement was
small. The aligned reference changed MAE 0.061374603 → 0.061373916 and PSNR 20.878422 →
20.878621 dB—effectively zero. The apparent +0.79 VD3D mean-score headline came almost entirely
from `flat_page`, where false stereo fell 0.25 px. Three compute passes and multiple GPU resources
are not justified by that isolated effect, so the fitted port was removed. Reports remain under
`sbs_eval/bestv2-curvature-{apollo,vd3d}/`.

**Accepted exact subject-plane-lock reproduction (2026-07-10):** Bestv2 now uses its actual
center-weighted depth band, separable equivalents of the 21×21 dilation and 15×15 closing, the
70% silhouette-fill blend, 13×13 smoothing, mask-weighted mean of the current shift field, and
the `weight^0.75` local correction. One shared GPU mask/mean feeds both warp geometries; no new
configuration switch was added. Raw-model and saved pre-warp depth artifacts are byte-identical
to the control. Against the aligned real Bestv2 output, MAE improves 0.061375 → 0.056834 (-7.4%)
and PSNR improves 20.878 → 22.193 dB; median pop moves from 15.48 px to 11.60 px versus VD3D's
12.56 px. This is a meaningful Phase-A reproduction gain and is retained.

It is not an Apollo quality improvement: excluding `flat_page`, generic-corpus score changes
-0.56 (Apollo) / -1.03 (VD3D), pop spread falls 0.58/0.62 px, and `rim_over_p95` rises 0.94/1.87.
That regression is explicit evidence for the later Apollo-improvement phase, not hidden by the
reference-fidelity verdict. Reports are under `sbs_eval/bestv2-plane-{apollo,vd3d}/`; fresh
controls are `sbs_eval/bestv2-plane-control-{apollo,vd3d}/`.

### B3 · Disocclusion concealment *(evaluated and rejected)*

VD3D: smear-blend-back (shift-grad>0.006 → 5×5 dilate → 60% blend back to flat 2D) + one-sided
directional repair ("Fast" for Bestv2) (`render_3d.py:3372+`). Apollo: none (probe fill only).

**Bestv2 concealment result (2026-07-10):** a shared GPU implementation was applied after each
geometry and evaluated as separate smear-only, repair-only, and combined treatments. Raw and
pre-warp depth remained byte-identical. Fast repair produced byte-identical SBS frames on the
evaluation corpus because its validity/gradient mask never activated. Smear-only was therefore
also the combined result: Apollo exact-field score fell by 0.1125, rim-over-p95 rose
5.9818 → 6.3212, and disocclusion flicker rose 5.0052 → 5.0308; VD3D hybrid score fell
by 0.0125 with the same rim/flicker direction. Both paths cost about 0.059 ms more warp
time. Against the aligned 1920×1080 real Bestv2 reference, combined output was byte-identical to
shift-only output, so MAE/PSNR did not improve. The treatment is rejected and its runtime code and
switches were removed; local HTML reports remain under `cmake-build-relwithdebinfo/sbs_eval/`.

### B4 · Post *(partial)*

VD3D: DOF (0.3, five-level Gaussian pyramid) + sharpen 0.2. Its `heal_missing_pixels` helper is
defined but never called by either active render path. Apollo retains only exact SDR sharpen as a
Phase-A parity operation.

**Rejected exact Bestv2 DOF result (2026-07-10):** a shared post-warp pass reproduced the five
Gaussian levels from sigma 0 through 0.3, per-pixel level interpolation from
`abs(depth-focus)/0.35`, and reflect padding without bleeding across the SBS eye boundary. This
also exposed and replaced Apollo's old incorrect interpretation of 0.3 as a normalized image
radius, which had only run on the probe geometry. Raw-model and pre-warp depth artifacts were
byte-identical. Excluding `flat_page`, mean score changed only +0.014 on both geometries, rim p95
improved only 0.023/0.034, and stereo volume was unchanged; GPU cost was about 0.011 ms. Against
the aligned VD3D render, MAE slightly worsened 0.05683363 → 0.05683677 and PSNR changed
22.19312 → 22.19263 dB. The effect is below meaningful visual/metric resolution and does not
improve fidelity, so the treatment and the stale DOF configuration/evaluator switches were
removed. Reports remain under `sbs_eval/bestv2-dof-{apollo,vd3d}/`.

Testing DOF again in its intended position immediately before sharpen did not rescue it: versus
sharpen-only, ordinary-scene score changed -0.014/0.000, rim p95 only -0.027/-0.029, GPU time
rose another 0.011 ms, and aligned-reference MAE changed just 0.01604888 → 0.01604809. The
combination reports remain under `sbs_eval/bestv2-dof-sharpen-{apollo,vd3d}/`; DOF stays removed.

**Accepted exact Bestv2 SDR sharpen reproduction (2026-07-10):** a shared post-warp, per-eye
3×3 convolution matches `tensor_sharpen(.2)`: center coefficient 5.2, four axial neighbors -1,
zero padding, then clamp. Raw-model and pre-warp depth artifacts are byte-identical. Against the
aligned VD3D render, MAE improves 0.056834 → 0.016049 and PSNR 22.193 → 30.152 dB; median pop
12.59 px is nearly the 12.56 px reference and spread 9.96 px approaches the 10.57 px reference.
This is the largest remaining Phase-A fidelity gain and is retained.

It is deliberately not called an Apollo quality improvement. Excluding `flat_page`, score falls
6.20/6.46, rim p95 rises 25.78/37.73, and pop spread falls 0.35/0.59 px on Apollo/VD3D geometry.
The generic reports at `sbs_eval/bestv2-sharpen-{apollo,vd3d}/` visualize those regressions. The
Phase-B Apollo experiments start from this faithful-but-harsh parity checkpoint and must beat it.

---

## Next controlled experiment

Phase A is complete. Begin Phase B with Apollo's color-guided depth upsample as the first isolated
improvement on top of the faithful Bestv2 checkpoint. Keep the raw-model checkpoint fixed; the
saved pre-warp depth is expected to change and must be reported explicitly because guided
upsampling snaps depth silhouettes to current-frame color edges. Evaluate both warp geometries and
retain it only if the independent quality/temporal metrics justify the depth-stage divergence.
