# warpsim — offline simulator for Apollo's SBS reprojection shader

CPU replica of `sbs_reprojection_ps.hlsl`, fed by real "Dump 3D" data. Reproduces warp
artifacts pixel-exactly and lets warp formulations be iterated and FALSIFIED in seconds
instead of headset round-trips. This workflow — simulate on real data, test theories with
numbers — is the established method for all SBS quality work (2026-07-03).

## Files
- `warpsim.cpp`       arm scene (dump_20260702_234719_03); modes: 0=shader replica,
                      1=no depth smoothing, 2=cliff bisection, 3=splat, 4=two-layer blend,
                      5=pre-smoothed single-tap depth (equivalence test — FAILED, see below)
- `warpsim_head.cpp`  head/hair scene (dump_20260703_011224_03); eye sign = last arg
                      (**-1 = left eye** — hair corrugation only reproduces in the LEFT eye)
- `warpdiag.cpp`      per-pixel path visualization (red=bg-fill, green=winner depth, blue=#crossings)
- `warptrace.cpp`     numeric scanline of bestDepth / displacement along one output row
- `shcheck2.cpp`      offline HLSL validation: `shcheck2 <file> ps_5_0 main_ps` (cs_5_0 main for CS)
- `export_dump_region.ps1`  one-command raw export from a dump folder (depth/source/both-eye GT
                      bins + prints the region constants to paste into a sim copy)

Build (msys2 ucrt64): `g++ -O2 <file>.cpp -o <file>.exe` (shcheck2: add `-ld3dcompiler`).
These are manual-build dev tools; they are NOT part of the CMake build.
NOTE: warpsim replicates sbs_reprojection_ps.hlsl -- when that shader changes, update the
replica (Reproject/SampleDepth/DepthParallax) or mode-0 validation will drift.

## The workflow (follow in order)
1. **Data**: tap "Dump 3D" in the XR bar → `E:\ApolloDev\sbs_dump\dump_*/{source,depth,sbs}.png`
   (same frame: 2D source 5120x2160, guided depth 1596x672, packed SBS 8192x1728).
2. **Export raws** (PowerShell System.Drawing LockBits):
   - depth: R channel bytes of the full depth.png → `depth*.bin`
   - source: RGB24 crop around the region + >=40px margins (search radius ~35 src px)
   - ground truth: RGB24 crop of the sbs at the eye region
   Coordinates: eye_x = 0.8 * src_x (4096/5120), right eye +4096; y likewise * 0.8.
   Sim pixel → uv: ux = (EOX+px+0.5)/4096, vy = (EOY+py+0.5)/1728.
   Region constants (DW/DH, SW/SH, SOX/SOY, OW/OH, EOX/EOY) are hardcoded at the top of
   each sim file — copy the file per scene and edit them + the input filenames.
3. **Validate the replica**: render mode 0, compare against the GT crop. The sim MUST
   reproduce the dump's artifact before any conclusion is trusted.
4. **Instrument before theorizing**: warpdiag + warptrace. Numbers killed two plausible
   theories in minutes (no missed crossings; winner selection perfectly smooth) and located
   the real cause (the silhouette stretch band re-rendering row-varying edge content).
5. **Iterate offline**; judge with 4x nearest-neighbor crops, byte-diffs for regressions,
   and diff masks to see WHERE a change lands.
6. **Both eyes**: run with EYE_SIGN = +1 AND -1. Artifacts are eye-asymmetric.
7. **Multiple scenes** before shipping anything — see the overfit lesson below.
8. **Verify "equivalent" changes numerically**: the pre-smoothed-depth perf swap (mode 5),
   claimed mathematically identical, measured 6.85% differing pixels / max delta 197 —
   the warp amplifies tiny depth deltas at silhouettes into content jumps.
9. Port the winner to HLSL → shcheck2 → shaders are hardlinked into build assets → live on
   stream reconnect (no rebuild). Then headset confirmation.

## Key numbers (current tuned pipeline)
- searchRadius = divergence*depth_scale*max(focal,1-focal) in source UV
  (0.015*0.9*0.5 = 0.00675 ≈ 34.6 src px); probe spacing = 2*radius*5120/steps.
- Stability: probe spacing must stay below the smoothed depth transition (~8 px from the
  2x2 SampleDepth) → **parallax_steps >= 22 is a correctness requirement** with guided
  depth. Default 8 predates the guided upsample; sunshine.conf overrides to 24.

## 2026-07-03 conclusions (why the warp rewrite was reverted)
- The visible edge fringe = the stretch band at silhouettes: contact-shadow/outline source
  columns carry BACKGROUND depth, detach from the object, and render row-incoherently.
- A stack of hand-tuned fixes (two-layer blend, parallax-continued clamps, contact-shadow
  exclusion, TV gate) made ONE scene's arm pixel-perfect but destabilized generic content:
  constants overfit one edge's structure, and the single/multi-edge gate creates spatial
  switching seams everywhere. REVERTED to the plain warp (commit 0c467e86 state).
- The principled fix for edge/disocclusion quality is a learned warp (iw3 row_flow/mlbw as
  a second TensorRT engine) or true multi-layer rendering — not more shader special cases.
- Hair-on-sky fold-over corrugation exists in the committed state too (left eye), was never
  noticed before; content-dependent; same learned-warp fix applies.
