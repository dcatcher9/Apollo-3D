# SBS 3D — status and roadmap

The living plan for Apollo's real-time 2D→3D side-by-side feature (host-side depth
estimation + reprojection for the Galaxy XR / Artemis client). Updated 2026-07-03.

## Current state (what ships today)

Pipeline: Depth Anything V2 small (fp16, TensorRT) at ~798×336 → GPU normalization
(min/max EMA) → **color-guided joint-bilateral upsample to 2× with foreground bias and
bimodal edge snap** → occlusion-aware backward reprojection (`sbs_reprojection_ps.hlsl`)
with far-depth floor and smoothed depth reads. Client switches host modes mid-stream via
the `0x3003` control message (OFF / GAME async / MOVIE reserved); `0x3004` triggers a
one-frame debug dump (source/depth/SBS PNGs) from the XR bar's "Dump 3D" button.

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

## Roadmap (priority order)

1. **Learned warp** — port iw3's `mlbw` (multi-layer backward warp, optionally with
   inpainting) as a second small TensorRT engine replacing the hand-written reprojection
   at silhouettes. This is the principled fix for the stretch-band fringe and hair
   fold-over. Validate offline in warpsim against the dump library first.
2. **Depth model upgrade** — swap to DA-V2 **base** (config-only: `sbs_3d_depth_model` +
   URL; same ONNX I/O contract via onnx-community fp16 exports). Targets missing
   small-feature relief via stronger priors. ~3–4× inference cost: movie-mode budget,
   not 90fps game mode. A/B on dumps (nose/sword depth crops) before headset time.
   DA-V3 only after base validates the direction.
3. **MOVIE mode host implementation** — synchronous depth (enqueue current frame,
   `cuStreamSynchronize`, warp current-with-current): zero motion ghost, full framerate
   for ≤30fps video content. Wire (`SBS_MODE_MOVIE`) and client tile already exist;
   host currently maps MOVIE→GAME. Fold the config surface into presets
   (`game`/`movie`/`desktop`) instead of 19 individual keys.
4. **DESKTOP mode** — movie mode on a region-of-interest: depth on a crop (the video
   window), flat composite outside. Manual/config rect first; ddup dirty-rect
   auto-detection later.

## Key references

- `tools/warpsim/README.md` — simulator workflow + warp math.
- `src/video_depth_estimator.cpp` — estimator/guided-upsample pipeline.
- `src_assets/windows/assets/shaders/directx/` — `sbs_reprojection_ps.hlsl`,
  `depth_guided_upsample_cs.hlsl` and friends.
- Client (Artemis, separate repo `moonlight-android`, branch `moonlight-noir`):
  `XrStreamPresenter.java` (mode tiles, Dump 3D), `moonlight-common-c` `0x3003`/`0x3004`.
