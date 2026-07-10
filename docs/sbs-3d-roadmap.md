# SBS 3D — status and roadmap

The living plan for Apollo's real-time 2D→3D side-by-side feature (host-side depth
estimation + reprojection for the Galaxy XR / Artemis client). Updated 2026-07-05.

> **TL;DR of the 2026-07-05 depth investigation (read before more depth work):** the two
> problems are (1) the **stretch band / jagged edges** at silhouettes and (2) **flat pop**
> on DA-V3. Both were chased hard:
> - **Edges are a WARP problem, not a depth problem** — proven both directions: DA-V3 depth
>   through the probe warp still shows the band (warpsim, ~6–10% only), while DA-V2 depth
>   through a learned warp made it vanish (eval sheets). **UPDATE 2026-07-10: the MLBW learned
>   warp was REMOVED** — it's iw3's lineage (a learned *backward* warp) and ignored the whole
>   VD3D subject-tracked shaping pipeline this project committed to (subject anchoring / band
>   curve / stretch / DOF all run on the probe path; MLBW composited its own fields and consumed
>   none of them). The edge fix is now a learned **inpaint** for the disocclusion holes *on top
>   of the probe warp* (probe emits a hole mask → light_inpaint fills it), keeping the shaping.
> - **DA-V3 flat pop was a MODEL/EXPORT problem, not inherent.** The onnx-community DA-V3
>   exports (small/base/**large**) all output *compressed* monocular depth (pop ~0.55–0.66).
>   The monocular-specialized **DA3MONO-LARGE** (depth-anything/DA3MONO-LARGE, 0.35B) gives
>   V2-level pop (0.74–0.90) — but needs a torch→ONNX export and is movie-mode cost.
> - **Depth normalization is now iw3's shifted reciprocal** `1/(depth + sbs_3d_depth_shift)`
>   (bounds the near spike) — the whole `1/depth` + sigma-clip / norm_sigma saga was a
>   band-aid and has been **removed**. Config: `sbs_3d_depth_shift` (default 0.2).

## Current state (what ships today)

Pipeline: selectable depth model (default DA-V2 small fp16, TensorRT) at ~798×336 → per-model
transform (DA-V2 identity disparity / DA-V3 shifted reciprocal `1/(depth+shift)`) → GPU
min/max-EMA normalization → **color-guided joint-bilateral upsample to 2× with foreground
bias and bimodal edge snap** → occlusion-aware backward reprojection (`sbs_reprojection_ps.hlsl`)
with far-depth floor and smoothed depth reads. Client switches host modes mid-stream via the
`0x3003` control message (OFF / GAME async / MOVIE reserved) and the depth **model** via
`0x3005`; `0x3004` triggers a one-frame debug dump (source/depth/SBS PNGs) from the XR bar.

Note (2026-07-05): DA-V3's guided upsample is measured to NOT be tunable — its silhouettes
are inherently softer than V2's and the guided pass can relocate but not manufacture sharpness;
edge crispness for V3 (as for V2) comes from a learned inpaint on the probe warp (roadmap #1), not the guided pass.

Verified quality (vs. pre-guided): thin objects straight (no bent sword handles), facial
silhouettes real (nose/brow/lips exist in depth), per-eye widths consistent, narrower
disocclusion bands.

Known residuals:
- Faint stretch-band fringe on hard silhouettes over textured backgrounds (the object's
  contact-shadow columns carry background depth and detach from the object).
- Fold-over corrugation on wispy hair against bright sky (left eye), content-dependent.
- No true small-feature relief (e.g. a nose bump reads flat) — model prior/resolution limit.
- Motion ghost in GAME mode from async depth (depth lags color by the inference cadence).

## Constraints learned the hard way (do not relearn)

- **Use the offline simulator first** for any warp/shader change: `tools/warpsim/`
  (CLAUDE.md has the mandate). Validate the replica reproduces the artifact, instrument
  with numbers, test BOTH eyes and MULTIPLE scenes, byte-compare "equivalent" changes.
- **Hand-tuned shader special-casing does not generalize** (2026-07-03 warp rewrite:
  made one arm edge perfect, destabilized everything else; fully reverted). Edge and
  disocclusion quality needs a learned/multi-layer approach, not more branches.
- `sbs_3d_parallax_steps >= 22` is a **correctness requirement** with the guided
  (texel-sharp) depth: probe spacing must stay under the ~8px smoothed depth transition.
  24 is the baked default since 2026-07-03. Note `depth_scale` was folded into
  `divergence` the same day (default 0.0135 = old 0.015*0.9; sim-verified equivalent),
  and the legacy `normalize`/`depth_gamma`/`edge_dilation`/`depth_interval` knobs were
  removed — defaults now live in config.h (`sbs_t` member initializers), not config.cpp.
- DA-V2 is trained at 518²; pushing inference resolution far past ~1.5× training area
  degrades global depth coherence. A big "1080p engine" rebuild is NOT the free win it
  looks like; prefer model-capacity upgrades and modest resolution changes, measured.
- iw3's temporal tricks (look-ahead EMA, Video-Depth-Anything) need future frames —
  incompatible with a live stream. Its synchronous mode = the design basis for MOVIE mode.

## Model switching — SHIPPED (2026-07-04)

On-the-fly depth-model switching is done and headset-verified: model registry +
per-name TRT engine slots (`g_engines`) + `0x3005 Set Depth Model` control message + an
XR-bar "Model" cycle tile. Startup engine prebuild (`sbs_3d_prebuild_models`), engine-file
versioning by build recipe, and DA-V3 (rank-5 input, output pruning, shifted-reciprocal
transform) all landed. Client pushed to the dcatcher9 forks (host **not committed**). Full
detail + the verified per-model contracts: [depth-model-switching-plan.md](depth-model-switching-plan.md).
The registry has DA-V2 small/base and DA-V3 small/base (fp16 + fp32); the DA-V3 exports are
onnx-community's, which turned out to be flat-pop (see TL;DR).

## Roadmap (priority order)

1. **Learned inpaint on the probe warp (THE edge fix)** — the probe reprojection fills
   disocclusions by *stretching* → the stretch band / jagged edges. The fix: have the probe
   emit a disocclusion **hole mask**, then a small learned **inpaint** model
   (`light_inpaint_v1`, ¼-res gMLP, iw3) fills the holes — clean edges WITHOUT replacing the
   warp, so the VD3D subject-tracked shaping keeps working. (The MLBW *learned warp* — a full
   warp replacement from iw3's lineage — was REMOVED 2026-07-10 because it consumed none of the
   shaping; do not re-add it. The inpaint is the salvageable half.) Needs: (a) probe emits the
   hole mask, (b) port `light_inpaint_v1` as a second TRT engine, (c) composite the inpaint over
   the probe output. Unlocks pop too: once holes are inpainted, divergence can be cranked
   without the stretch band. Warpsim-gate every change.
2. **DA3MONO-LARGE depth (THE pop fix)** — the onnx-community DA-V3 exports give flat
   monocular pop; the monocular-specialized **DA3MONO-LARGE** (depth-anything/DA3MONO-LARGE,
   0.35B, DINO ViT-L) gives V2-level pop (0.74–0.90) *and* DA-V3 geometric accuracy.
   No ONNX exists (iw3 runs it in PyTorch; only the general DA3 is on onnx-community) —
   needs a torch→ONNX export: wrap `{depth,sky}` → single `predicted_depth`, rank-5 input,
   fp16 via the ORT converter recipe. Weights + code are local (`E:\Git\Repo\nunif` venv,
   1.3 GB checkpoint). Drops into the shifted-reciprocal pipeline. Cost: 0.35B = movie-mode;
   game mode needs a lower depth_fps or stays on v2-small (no smaller monocular variant exists).
3. **MOVIE mode host implementation** — synchronous depth (enqueue current frame,
   `cuStreamSynchronize`, warp current-with-current): zero motion ghost, full framerate
   for ≤30fps video content. Wire (`SBS_MODE_MOVIE`) and client tile already exist;
   host currently maps MOVIE→GAME. Fold the config surface into presets
   (`game`/`movie`/`desktop`) instead of 19 individual keys.
4. **DESKTOP mode** — movie mode on a region-of-interest: depth on a crop (the video
   window), flat composite outside. Manual/config rect first; ddup dirty-rect
   auto-detection later.
5. **3D Style assistant (intuitive preset system, ported from VisionDepth3D)** — VD3D's
   "3D Assistant" is a two-tier front-end over its ~19 backend knobs, and it's a clean fit
   for Apollo now that the subject-tracked shaped-disparity pipeline exists (sync depth +
   `subject_track`/`subject_stretch`/`subject_lock`/`stretch_lo·hi`/`subject_plane_lock` +
   `divergence`/`focal_plane`/`border_fade`). Two layers to port
   (`ui/pages/stereo_generator_page.py`: `SIMPLE_3D_ADVANCED_PRESETS`, `SIMPLE_3D_PRESETS`,
   `_apply_simple_3d_controls`):
   - **Named style bundles** — a handful of tuned Apollo `sbs_3d_*` param sets
     (Comfortable Cinema / Deep Background / Close-Up Safe / VR Comfortable / Showcase …),
     selectable as one unit.
   - **~6 intuitive sliders → params via curves** — Strength→`divergence` (+ max shift),
     Pop-Out→`subject_stretch` on + fg-weighted shaping/convergence bias,
     Comfort→caps `divergence` + `border_fade`, Stability→`subject_lock`/`subject_plane_lock`
     (+ depth EMA), Screen Depth→`focal_plane`, Subject Zero-Lock→`subject_lock`. Mirror
     VD3D's lerp/curve mapping (`strength**0.85`, `comfort_limit = 1 - 0.1·comfort`, …).
   - **Apollo-specific divergence**: the 2026-07-09 finding is that Apollo's edge quality is
     NOT parallax-coupled (the probe fill doesn't stretch, unlike VD3D's forward warp),
     so Apollo's "Showcase"/"Strong Pop-Out" styles can push `divergence` HIGHER than VD3D's
     without the stretch-band cost that makes VD3D's Showcase the worst-edge style. Retune,
     don't copy, the aggressive styles.
   - **Surfaces**: the Vue config UI (best for setup — a Style dropdown + the sliders writing
     `sbs_3d_*`), and/or client XR-bar Style tiles for on-headset cycling (like the existing
     mode/model tiles; would need a control message to push style/param overrides, à la
     `0x3003`/`0x3005`). Style overrides config last-writer-wins, exactly like VD3D's
     assistant overrides a loaded preset.

## Key references

- `tools/warpsim/README.md` — simulator workflow + warp math.
- `src/video_depth_estimator.cpp` — estimator/guided-upsample pipeline.
- `src_assets/windows/assets/shaders/directx/` — `sbs_reprojection_ps.hlsl`,
  `depth_guided_upsample_cs.hlsl` and friends.
- Client (Artemis, separate repo `moonlight-android`, branch `moonlight-noir`):
  `XrStreamPresenter.java` (mode tiles, Dump 3D), `moonlight-common-c` `0x3003`/`0x3004`.
