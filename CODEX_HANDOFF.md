# Handoff: Host 3D subtitle handling code review

Branch: `claude/exciting-meitner-iarw5l` (no source changes; this file only).
Scope: Host AI 3D (Host SBS) subtitle pipeline, mostly landed in commit `c6ef70e`
(~63k lines). Review was static only (Linux container, no Windows/GPU build).
User focus: bugs, perf, mask accuracy, stability, refactoring.

Pipeline map: OCR preprocess/boxes (`host_sbs_ocr_*_cs.hlsl`) -> classifier OCR8 ->
locator/UIP4+SCP4 controller (`host_sbs_subtitle_locator_cs.hlsl`) -> legacy contour matte
(`host_sbs_subtitle_contour*_cs.hlsl`) AND current-outline mask
(`shaders/directx/current_outline/**`, driven by `src/host_sbs_source5_cbc5_f2_composition_gpu.h`)
-> live selector PS `current_outline/completed/sbs_reprojection_v2_live_ps.hlsl`.
Live always enables current-outline (`display_vram.cpp:3649-3663`); the legacy contour matte is
fallback-only but still computed every observation. `queued_live_candidate_enabled` is
constexpr false (`display_vram.cpp:7336`), so queued/pending-colour paths are probe-only.

Status legend: **V** = confirmed by two independent verifiers; **R** = reviewer-confirmed,
verification not finished; **U** = found by gap reviewer, not verified; **X** = refuted.

## Start here (highest value, verify first)
1. **U - Whole Host 3D can fail flat.** Eligible context + OCR unavailable or source >5120 px:
   admission still requests subtitle work (`video_depth_estimator.cpp:896-899`), finalizer fails
   (`5136-5150`; `update_pending_ocr_constants` returns false on null line reference `7710-7716`;
   `host_sbs_resolution.h:384-395`) -> `fail_gpu_conditional_bridge_once` (~9190-9197).
2. **R - Window move/resize wipes subtitle state + flat frames.** Context key includes
   `foreground.generation` (`display_vram.cpp:5353-5363`, `subtitle_context_policy.h:40-48`,
   `foreground_window_region.h:45-47`); test `test_host_subtitle_context.cpp:271-273` asserts it.
   Fix: key on kind/HWND/PID/browser video identity only.
3. **R (found twice) - Windowed native players lose captions.** Foreground-client authority uses
   full-screen centre/lower-quarter reference (`host_sbs_resolution.h:391-405`).

## Bugs / stability
- **V (med, documented policy)** One empty OCR observation drops glyphs to Base then snaps back
  (locator `:536-552`, `:570-572`; matte `contour:1042`; SCP4 `locator:428`). Fix must hold
  covers/SCP4/matte across one empty, not just controller state.
- **U** `reset_subtitle_state` (`video_depth_estimator.cpp:6993-7030`) does not reset
  `subtitle_current_outline_composition/selected`; black-band cache survives context change.
- **R** Enclosure/outline flicker: one non-black outline gap flips whole glyph (`enclosure ~686-759`);
  adoption uses `Meta[u+3]` (incl. restored) vs `normalRefine` `Meta[u+30]` (`enclosure:732-745`,
  `support_band:1042-1047`) contradicting host-sbs.md:804-805; cache offer uses Euclidean/inner-only
  vs measured Chebyshev/SeedAt (`band:305-311,847-878`); one empty unit revokes whole cache
  (`band:377,740,760`).
- **R** Classifier gates are binary per frame (no hysteresis); per-line 2% centring drops
  left-aligned multi-line captions (`host_sbs_ocr_boxes_cs.hlsl:824-832,1147`); ribbons teach the
  layout prior (`boxes:1156`).
- **R** ~60 production shaders not prewarmed, compiled on encode thread; unused queued PS closure
  mismatch disables the whole route (`display_vram.cpp:4549-4568`).
- **X** "first cue after empty publishes target 0" - unreachable (geometry readiness enforces clock).

## Performance
- **R** ~600-800 serialized `DispatchIndirect` per observation regardless of captions
  (`host_sbs_subtitle_colour_commands.h:51-80`, `host_sbs_current_black_enclosure_gpu.h:713-757`,
  `host_sbs_current_outline_support_closure_gpu.h:756-846`); per-mode passes span full band
  (colour `:558`). Fix: GPU-compacted active (slot,mode) list + ROI dispatch.
- **V (med)** Lane-0 per-group revalidation in 4-6 passes (contour `494-521`, `705-815`; also
  colour/selector/band). Fix: one per-frame verdict record.
- **R** P010 luma MRT never used in Host 3D (`display_vram.cpp:2667-2691`); push-pull background
  rebuilt every draw on pow2 FP32 pyramid (`host_sbs_selected_glyph_push_pull_bg_gpu.h:153-179`).
- **R** Live PS: ~290-450 uniform auth loads + second inverse solve per band pixel
  (`current_outline/completed/include/host_sbs_subtitle_render.hlsl:255-317`).
- **V (low)** Diagnostic-only passes run live (note: `history_main` is the sole SCT5 writer - only
  its feature loop is gateable); native-band textures (~137-245 MB) allocated when ineligible.
- **R** `calibrate_main` single-thread (`band:236-258`); O(N^2) lane-0 OCR grouping on dense text.

## Mask accuracy (R unless noted)
Component touching domain edge drops whole glyph (`components:95-99,173`); brightest-anchor mode
representative (colour `:467`); hard 4-level ink bins (`:460-545`); axial-only witnesses lose
diagonals (`:337-343`); slope test ignores 1-row quantization (`boxes:703-723`); HDR halves 0.08
contrast margin. **X**: palette hue-vs-RGB metric and fixed radii (live draws CurrentOutlineM).

## Refactoring
- 4 copies of `depth_coordinate_v2_contract.generated.hlsl` under `current_outline/*/include/`
  not written/checked by `tools/sbsbench/generate_depth_coordinate_v2_contract.py` (confirmed).
  3 byte-identical ROI shaders; 2 `physical_black_model_shared.hlsl`; 2 diverging subtitle renderers.
- Production code named `research_*`/"DRAFT"; probe-only code (queued renderer, pending colour lane,
  body-witness temporal bridge whose shaders are not in tree) compiled into production TUs.
- Hand-copied word maps/flags across HLSL/C++/tests; SCT5/SCP4 validated ~4x -> generated typed contract.
- Docs: host-sbs.md live selector section (`:334-360`) outdated; says 10% centre tolerance (`:601-603`)
  vs 2% in contract. Tests: production selected PS untested; GPU fixtures skip by default.
- **U** Offline/`run_eval` render with legacy matte (`sbs_bench_harness.cpp:4520-4547,6743-6780`),
  so evaluator does not gate the live composite; no mask-accuracy or temporal-flicker metric.

## Constraints for whoever continues
Follow `CLAUDE.md`: Windows/MSYS2 RelWithDebInfo build, `test_sbsbench`/`test_sunshine`, and the
sbsbench control/treatment loop before any geometry/shader change; update the owning doc
(`docs/host-sbs.md`) with contract changes. Delete this file once its items are tracked elsewhere.
