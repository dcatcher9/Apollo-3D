# SBS 3D — next-architecture direction (design notes)

Working notes from the 2026-07-05 design discussion, written so it can be resumed in a
new session. This is the **forward-looking design** for how to do the 2D→SBS pipeline
"correctly", distinct from [sbs-3d-roadmap.md](sbs-3d-roadmap.md) (the prioritized plan)
and [depth-model-switching-plan.md](depth-model-switching-plan.md) (the shipped switching
feature). Read the roadmap TL;DR first for the one-paragraph summary.

## The reframe: it's a 3-stage pipeline, and each stage's problem is separable

Every modern 2D→stereo system (iw3, StereoCrafter, M2SVid) — and Apollo — is the same
three stages. "End-to-end" models (e.g. M2SVid) still have all three; only stage 3's
*training/inference style* is end-to-end (single pass, image-space loss), not the whole
system. Truly monolithic frame-in/SBS-out nets (Deep3D 2016) exist but lost, because the
warp is exact free geometry and you want explicit control (baseline/convergence) + a
swappable depth model.

```
2D frame ──▶ [1] DEPTH ──▶ [2] WARP (geometry) ──▶ [3] FILL disocclusions ──▶ SBS eye
```

Apollo's two visible problems map cleanly onto two stages:
- **Flat pop** = a **stage 1 (depth model)** problem — SOLVED in principle (DA3MONO-LARGE).
- **Stretch band / jagged edges** = a **stage 3 (fill)** problem — the current fill is naive.
  (The warp geometry itself is fine; it's the hole-filling that's the artifact.)

Depth temporal-flicker and the async motion-ghost are separate axes and NOT the current
blockers — don't chase them yet.

---

## Stage 1 — DEPTH

**Current:** selectable model; DA-V2 small default (game). DA-V3 (onnx-community small/base/
large) integrated but **all give flat monocular pop** (compressed depth, pop ~0.55–0.66 vs
V2 ~0.82–0.92). Transform is now iw3's **shifted reciprocal** `1/(depth + sbs_3d_depth_shift)`
(default 0.2) with plain min/max — the `1/depth`+sigma-clip/norm_sigma band-aids were removed.

**Target for pop:** **DA3MONO-LARGE** (depth-anything/DA3MONO-LARGE) — the monocular-
*specialized* DA-V3 (0.35B, DINO ViT-L). Measured pop **0.74–0.90 ≈ V2**, with V3's geometric
accuracy + richer background. Only exists in LARGE (no small/base monocular variant).

**Decision needed / open:**
- Export DA3MONO-LARGE torch→ONNX (no ONNX exists; iw3 runs it in PyTorch). Weights (1.3 GB)
  + code are local at `E:\Git\Repo\nunif` (venv has torch-cpu; `iw3/pretrained_models/hub/`).
  Wrap `{depth, sky}` → single `predicted_depth` (rank-5 input), fold sky→far, fp16 via the
  ORT converter recipe (`tools/depth_models/convert_da3.py` pattern; RoPE backbone).
- Cost: 0.35B = movie-mode budget (~60–80 ms/inf est). Game mode stays DA-V2 small OR
  DA3MONO at reduced depth_fps (more ghost). Confirm on the 5080 after export.
- **DepthCrafter is NOT a live option** (video diffusion, needs future frames, seconds/chunk).
  It is the right depth for the OFFLINE M2SVid route only. Fixes flicker, not ghost.

---

## Stage 2 — WARP (geometry)

Three warp options; "best" depends on the stage-3 fill budget (see table). All use the same
disparity = f(depth, divergence, focal) geometry.

| warp | disocclusion output | artifact | cost | pairs with |
|---|---|---|---|---|
| **backward probe** (current, `sbs_reprojection_ps.hlsl`) | implicit naive stretch = THE BAND | winner-flip jaggies | ~24 taps/px | nothing (baseline) |
| **MLBW** learned backward (STASHED) | implicit *learned* coherent stretch | slight softness | 0.53 ms/eye | standalone, or +mask→inpaint |
| **forward/geometric** splat (iw3 `forward_warp.py`, M2SVid) | **explicit holes + mask** | cracks, collisions | cheap+atomics | REQUIRES an inpaint stage |

- Without an inpainter → **MLBW** is best (learned stretch beats naive stretch; already built).
- With an inpainter → a warp that emits an **explicit hole mask** (forward, or MLBW with
  iw3's `return_mask`) so the fill model gets an honest "no data here" signal, not smeared data.

---

## Stage 3 — FILL disocclusions (the missing piece)

**Current "step 3" = one line:** in the probe warp, if the search finds no surface crossing
(a disoccluded pixel), it samples the color at `bgX` — the nearest *background* column in the
search window (`return (bestDepth >= 0) ? bestX : bgX;`). Per-row independent → horizontal
smear + row-incoherent tearing = the stretch band on the face. `depth_floor`, the bimodal
snap, `border_fade` only *shrink/clean the edge of* the band; none fills it.

**Target:** a learned **inpaint** model conditioned on {warped eye, hole mask} (optionally the
left frame) that synthesizes hole content — this is M2SVid/StereoCrafter's stage 3 and iw3's
`mlbw_inpaint.py` / `light_inpaint_v1` (small ¼-res gMLP). Once holes are *filled* (not
stretched), divergence/pop can be increased safely — so **stage 3 also unblocks stage-1 pop**.

**Options for the inpaint model, cheapest→best:**
1. Port iw3's `light_inpaint_v1` as a 3rd TRT engine (small, real-time-plausible; movie
   comfortable, game tight — measure).
2. Train/distill our own tiny inpaint head on Apollo's content using **M2SVid as the teacher**
   (its 4.6 GB SVD net → a few-hundred-k student). M2SVid conditioning recipe = left + warped-
   right + hole mask, single pass, image-space loss — copy verbatim.

**M2SVid's three uses (it is NOT real-time — ~1.5B SVD, 512², needs frame chunks + DepthCrafter):**
- (a) **Offline pre-conversion** → stream via **Host SBS Raw** (works TODAY, zero code). The
  quality ceiling + a perceptual reference.
- (b) **Teacher/GT** for benchmarking our warp+inpaint in warpsim and for distillation.
- (c) **Architecture confirmation** — its stage 3 IS our planned inpaint stage.

---

## Concrete next steps (pick up here)

1. **Merge the MLBW stash** onto master (immediate edge win for V2 AND V3; V3's soft depth
   actually suits MLBW). Stash: `git stash` "MLBW learned warp WIP + SBS reprime fix" (cut
   from 683f3c0d = pre-everything; conflicts in video_depth_estimator/display_vram/main —
   unify its twin-cache with the shipped `g_engines` map; restore the profile-only-for-dynamic-
   input guard). Also extract the **SBS reprime fix** (black frame on rebuild over idle desktop
   — affects model switching too).
2. **Export DA3MONO-LARGE** → ONNX → fp16 → registry entry (pop fix, movie mode).
3. **Add the hole mask + inpaint** (MLBW `return_mask` → `light_inpaint_v1` TRT engine, or a
   distilled M2SVid student). This is what dissolves the "stretch vs pop" tension.
4. Warpsim-gate everything (CLAUDE.md mandate); benchmark against offline M2SVid outputs.

## Key measured facts (do not re-derive)

- V3 disparity pop (p90-p10 of min/max-normalized): onnx-community small/base 0.55, large
  0.57–0.66; **DA3MONO-LARGE 0.74–0.90**; V2 0.82–0.92. (da3mono at 336 res already 0.74–0.78.)
- V3 depth is inherently softer at silhouettes than V2 (~5×); guided upsample can relocate
  but not manufacture sharpness → NOT the lever for V3 edges. Warp is.
- Stretch band through the probe warp: V3 depth reduces it only ~6–10% vs V2 (warpsim); V2+MLBW
  eliminates it (eval sheets). ⇒ edges are warp-side.
- Shifted reciprocal `1/(depth+0.2)` bounds the near spike (no blowout) but does NOT add pop
  for the compressed onnx exports — pop must come from the model (DA3MONO) or more divergence.

## References

- iw3 (E:\Git\Repo\nunif\iw3): `depth_anything_v3_model.py` (shifted reciprocal + sky),
  `forward_warp.py`, `backward_warp.py`, `mlbw_inpaint.py`, `forward_inpaint.py`,
  `models/{mlbw,light_inpaint_v1,light_video_inpaint_v1,row_flow}.py`. Runs depth in PyTorch
  (no ONNX export in iw3).
- M2SVid: github.com/google-research/m2svid (Apache-2.0, weights released, offline).
- DA3MONO-LARGE: huggingface.co/depth-anything/DA3MONO-LARGE.
- Apollo warp: `src_assets/.../sbs_reprojection_ps.hlsl`; replica + workflow `tools/warpsim/`.
