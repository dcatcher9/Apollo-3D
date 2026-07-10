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
| B3 | disocclusion concealment | per-frame | — (not ported) | ✅ VD3D-only |
| B4 | post (DOF, heal, sharpen) | per-frame | `sbs_reprojection_ps.hlsl` (DOF only) | partial |

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
| resolution | 768×432 (short 432 ≈ 1.24× DA-V2's 518² training) | **runs at baked height 336** (short side 336, ~602×336 @16:9) — the DA-V2 engine is `dynamic_width` with `fixed_h=336` baked into the ONNX, so `depth_short_side=432` is **inert** for it (it only drives non-dynamic-width engines). To actually hit 432, DA-V2 needs re-export at height 432. | ⚠️ **NOT matched** — but depth still correlates r≈0.96 with VD3D (verified 2026-07-10), so resolution isn't the bottleneck |
| cadence | per-frame | **async** at `depth_fps` (45 game) — depth lags color | ⚠️ **the** depth-half gap |

**Sync is gone.** `sbs_3d_sync_depth` and the forked synchronous `estimate()` were removed
(commit 5e7530fe). **Both GAME and MOVIE run async today** (MOVIE maps onto GAME's async path;
true synchronous MOVIE is roadmap #3, unbuilt). Async is the only behavior; the lag is a permanent
architectural property, not a toggle. Kept because the harness `--settle` erases the lag before
scoring (so eval can't justify a global sync default) and real per-frame sync only pays off for
MOVIE's heavy DA3MONO, where it stalls the encode thread (a wash).

**Faithful-baseline decision:** keep async (matching VD3D's *offline result* is about depth
*values*, not delivery cadence; the harness settles the lag anyway). Resolution already at parity.
Nothing to change.

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

### A4 detail — VD3D normalizes depth *twice* (don't chase the lock)

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
Apollo could also reproduce the bootstrap-freeze (a few frames → freeze; that's what
`norm_lock_frames` does) but it was bench-falsified for DA-V2 — the raw output drifts affinely so a
frozen range goes stale, and the tracking EMA is what compensates. VD3D tolerates its frozen range
only because the render per-frame EMA corrects that drift on top. Apollo gets the same stability
more simply from **percentile p2/p98 + a slow per-frame `minmax_ema`** — no freeze. The faithful
config below encodes exactly that.

### Faithful-baseline depth config (Group A)

| knob | Apollo default | Phase-A faithful | rationale |
|------|----------------|------------------|-----------|
| `sbs_3d_depth_short_side` | 432 | **432** | already parity |
| `sbs_3d_norm_pct_lo`/`_hi` | **2 / 98** (was 0/100 raw) | **2 / 98** | VD3D `DepthPercentileEMA(p2/p98)`. **Flipped to default + re-baselined 2026-07-10**: +5.7 total score, 7/8 clips up (recovered depth range from DA-V2 outlier rejection). Cost: `flat_page −1.9` (percentile stretches a flat scene's tiny range and amplifies hallucinated structure — the exact thing range floor guarded, now removed; acceptable for real content, revisit for DESKTOP mode). The min/max reduction still runs (it's the histogram's bin range). |
| `sbs_3d_minmax_ema` | 0.1 (α≈0.9) | **0.1 (keep)** | VD3D's 0.18 (α=0.82) is only safe on its pre-locked input; on Apollo's un-locked drift it flickers (c339 `flicker_disocc` 2.7→47) — **eval-proven**, keep the slow range EMA |
| `sbs_3d_ema` (per-pixel) | 0.6 | **0.5** | VD3D `TemporalDepthFilter(α=0.5)` |
| `sbs_3d_range_floor` | 0 (off) | **off** | Apollo-only; VD3D lacks it. Was pinned 0.5 in bench.conf (so the eval ran it); **removed 2026-07-10** so the eval matches the config.h default |
| `sbs_3d_norm_lock_frames` | 0 | **0** | VD3D's bootstrap-freeze (3–5 frames) is live-approximable, but Apollo already bench-tested it (norm_lock_frames) and DA-V2's affine drift makes a frozen range go stale (swim worse); VD3D survives it via its per-frame render EMA on top |
| `sbs_3d_guided_upsample` | **off** (was on) | **off** | VD3D has no upsample — see B1. **Default flipped off 2026-07-10** + eval re-baselined; re-enabling it is now Phase-B #1 (edge crispness pending the concealment port) |

The one real gap Apollo closes here is **raw min/max → per-frame percentile** (`pct 2/98`), plus
the per-pixel `ema=0.5`. Apollo does **not** copy VD3D's α=0.82 range-bounds EMA: it keeps the
slower `minmax_ema=0.1`, because VD3D's α is calibrated for input its bootstrap-freeze has already
coarse-stabilized, whereas Apollo feeds raw drifting DA-V2 output straight in — using 0.18 there
just reintroduces range breathing as disocclusion flicker (eval-proven, below). VD3D's depth-gen
**bootstrap-freeze** is likewise not reproduced: it's live-approximable (a few frames → freeze),
but Apollo already bench-tested it (`norm_lock_frames`) and DA-V2's affine drift makes the frozen
range go stale (swim worse). VD3D only tolerates the frozen range because its per-frame render EMA
corrects the drift on top; Apollo gets the same stability more simply from the per-frame percentile
+ slow `minmax_ema` alone.

**Eval findings (2026-07-10, decomposed A/B vs the current default baseline):**
- *Percentile norm alone* (`pct 2/98`, `ema 0.5`) with `minmax_ema=0.18`: severe temporal
  regression — c339 `flicker_disocc` 2.7→47. Culprit isolated to `minmax_ema`: rerun with 0.1 and
  the flicker regression **vanishes** (only c747 `rim_over` remains, an edge metric).
- *Disable guided upsample alone*: edge/silhouette regressions on sharp content
  (fast_motion `edge_acc` 0→24, `disocc_smear` 0→0.36, score 81→70; flat_page `stretch`) — the
  expected cost of removing Apollo's edge crutch with no concealment ported yet.
- **Corrected faithful-norm config: `pct 2/98` + `minmax_ema 0.1` + `ema 0.5`** — nearly clean
  (only c747 `rim_over`, an edge metric concealment will address). Guided-off stays as the
  deliberate Phase-A edge sacrifice, remedied by porting Stage B3 concealment.

**Phase-A action:** commit `tools/sbsbench/bestv2_faithful.conf` with these values, capture its
baseline, then Phase B re-enables each Apollo depth addition (`guided_upsample`, raw min/max,
range-floor) as a gated A/B against it.

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

### B2 · Warp + shaping *(the irreducible divergence — deferred)*

Shaping is **not** a separate pre-warp stage; it runs inside the reprojection shader at warp time,
reading `SubjectState` (A8): depth-floor compression, subject recenter/stretch/band-curve/plane-lock
(all number-match validated), then the warp itself.

The warp is where Apollo and VD3D fundamentally differ:
- **VD3D:** `grid_sample` bilinear ×0.35 **+** depth-order forward warp ×0.65 + hole-fill
  (`render_3d.py:3335`).
- **Apollo:** occlusion-aware backward **probe** (frontmost-wins, nearest-bg fill).

Whether Phase A ports VD3D's forward warp (true stage-for-stage reproduction) or matches its *look*
on Apollo's probe (arguably the better warp) is the open decision, taken after Group A is locked.

### B3 · Disocclusion concealment *(not ported — likely the real edge-look gap)*

VD3D: smear-blend-back (shift-grad>0.006 → 5×5 dilate → 60% blend back to flat 2D) + one-sided
directional repair ("Fast" for Bestv2) (`render_3d.py:3372+`). Apollo: none (probe fill only). The
smear-blend-back trades pop for cleanliness in disocclusion zones — the most probable single source
of VD3D's "clean" look, and roadmap stage 3.

### B4 · Post *(partial)*

VD3D: DOF (0.3, pyramid) + `heal_missing_pixels` + sharpen 0.2. Apollo: DOF ported (13-tap; focal
fixed this session to the near plane when no subject). Heal/sharpen not ported.

---

## Open decisions

1. **Warp fidelity (B2):** port VD3D's forward warp for true stage-for-stage reproduction, or match
   its look on Apollo's probe? *(Deferred until Group A is locked.)*
2. **Concealment (B3):** port smear-blend-back + directional repair now (cheap, roadmap stage 3,
   probable edge-look driver) or after the warp decision?
