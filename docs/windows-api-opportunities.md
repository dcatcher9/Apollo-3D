# Windows platform opportunities: overlays, window geometry, and ROI 3D

## Purpose and audience

This is an implementation handoff document. It collects a set of proposals for Sunshine 3D that
share one theme: using Windows platform APIs and desktop compositor knowledge to produce better
stereo geometry than a monocular depth model can infer on its own.

It is written to be self-contained for an implementer who has not seen the discussion that produced
it. Every claim about current behaviour is anchored to a verified `file:line` in this repository.
Claims I did **not** verify are collected in [Open questions](#open-questions-not-yet-verified) and
must be checked before relying on them.

This document does not own any contract. Where a threshold, schema, or geometry constant matters,
it links to the owning document instead of restating it, per the convention in
[CLAUDE.md](../CLAUDE.md).

Owning documents an implementer should read first:

- [Host SBS pipeline](host-sbs.md) — live/offline geometry and the HDR contract.
- [Host SBS scene cuts](host-sbs-scene-cuts.md) — cut evidence, state, acceptance.
- [Offline Host 3D conversion](whole-clip-sbs-pipeline.md) — the offline media/cache/HDR contract.
- [SBS 3D status and roadmap](sbs-3d-roadmap.md) — current limitations and active work.
- [sbsbench](../tools/sbsbench/README.md) — the evaluation workflow and its metric/dump contracts.

## Review outcome — August 6, 2026

This document is a roadmap, not an instruction to enable every proposal. The review found several
important corrections:

- **W1 is not a small cross-backend move.** Desktop Duplication keeps a separate pointer shape, but
  Windows.Graphics.Capture currently asks Windows to burn the pointer into the captured texture.
  A shared post-warp cursor needs a new backend-independent cursor layer. For ordinary desktop UI,
  screen-plane placement is the safe default; following inferred video depth should be an optional
  experiment, not the default.
- **W5 knows occlusion order, not physical depth.** HWND Z-order can author stable window planes, but
  transparent/layered windows, regions, child content, popups, the shell, and full-screen video make
  a rectangular window stack an approximation. It must never classify a maximised video player as
  a flat desktop merely because it occupies one HWND.
- **W11's host half already exists.** Windows touch injection is implemented with
  `CreateSyntheticPointerDevice` / `InjectSyntheticPointerInput`; the missing work is client
  hand-ray/gesture mapping into the existing touch protocol.
- **W12's claimed movie/game parameter sets no longer exist.** V2 intentionally exposes literal
  strength rather than hidden profiles. Foreground-process information is useful as classification
  evidence, but it must not silently change geometry strength.
- **Offline subtitle analysis must remain scene-bounded.** The offline converter is designed around
  scene-level buffered look-ahead. A new whole-clip pre-pass would duplicate work and cross the
  current ownership boundary; stable masks should instead be learned and revised inside the scene
  buffer.
- W8, W10, W11, and W9 are not host-only changes. They require protocol and Moonlight 3D work, so
  they cannot be treated as local edits to one Windows source file.

The first completed investigation is the diagnostics-only Desktop Duplication dirty-rectangle probe
in Phase 0.1. It changes no rendering or depth behavior. Later items remain proposals until their
own evidence gates pass.

## Context an implementer needs

### The pipeline as it stands

Host SBS has exactly one geometry implementation: Depth Coordinate V2. The legacy V1 renderer and
its configuration surface were deleted in commit `cf478187`. An invalid or unauthenticated V2 frame
renders **flat** SBS; there is no alternate renderer and no fallback geometry.

The live path is, in order: desktop capture (DXGI Desktop Duplication, with a Windows.Graphics.Capture
path alongside it) → cursor blend → `convert()` → DAV2 depth inference → V2 coordinate mapping →
signed final parallax → an 11-iteration contractive inverse warp → SBS packing → NVENC.

Only three configuration keys survive on this surface: `sbs_3d_pop_strength`,
`sbs_3d_max_encode_width`, `sbs_3d_cuda_graph`. Do not reintroduce removed keys; the deletion was
deliberate.

### A resolution constraint that will bite ROI work

`host_sbs_v2_source_resolution_rejection_reason()` gates a source resolution by **fitting a depth
tensor shape and then authenticating that shape**, not by checking the resolution directly
([src/host_sbs_resolution.h:151](../src/host_sbs_resolution.h:151)). The fitter takes the source
aspect ratio plus a fixed short side and snaps to a patch-14 grid.

The practical consequences:

- The gate is effectively **on aspect ratio, not absolute size**. A 640×360 region and a 1920×1080
  region fit the same tensor and both pass.
- The only size limits are upper bounds: long side ≤ 5120 and area ≤ 5120×2160
  ([src/host_sbs_resolution.h:19](../src/host_sbs_resolution.h:19)). There is no lower bound beyond
  non-empty.
- An **arbitrary** aspect ratio (an ROI of, say, 1123×649) will usually fit a shape that is not in
  the shipped calibration and will **fail closed to flat SBS**.

Therefore any ROI feature must snap its detected rectangle to an authenticated aspect before
submitting it, rather than warping the exact detected pixel rect. In practice this is cheap, because
video content is almost always 16:9 or 2.39:1.

### Non-negotiable repo process

An implementer who ignores this section will produce changes that cannot be accepted.

1. **Build**: MSYS2 UCRT64, build directory name starting `cmake-build-`, official Node.js on
   `PATH` before CMake. Use `RelWithDebInfo`. See [Building](building.md).
2. **Never validate a geometry or shader change by looking at the headset.** The mandated loop is
   `python tools/sbsbench/run_eval.py --comparison-only --label control`, then the treatment run,
   then bind them with `--report-control`. Exit codes: `0` pass, `1` named regression, `2` setup or
   provenance failure.
3. **Committed baselines are intentionally empty.** `tools/sbsbench/baselines/` holds only a README;
   the default gated mode fails closed with exit 2 by design. Use `--comparison-only` for ordinary
   development. Publishing a baseline is a deliberate, justified act committed together with the
   source, contract, and clip changes.
4. **Tests**: `ninja -C cmake-build-relwithdebinfo test_sunshine` then run
   `cmake-build-relwithdebinfo/tests/test_sunshine.exe`. Python:
   `python -m unittest discover -s tools/sbsbench -p "test_*.py"`. The native suite passes after the Phase 0.1 probe
   (533 passed, 8 expected skips); the previously verified Python suite remains 570 passed, 14
   expected skips. Keep both suites passing.
5. **Conventions**: `.clang-format` for C/C++, `.flake8` for Python, `.prettierrc.json` for web
   assets. Keep configuration, Web UI defaults, localization, generated contracts, and tests
   consistent. Preserve unrelated working-tree changes. GPL-3.0-only.
6. **Generated contracts are generated.** Do not hand-edit `src/generated/*.h` or the
   `*.generated.hlsl` files; regenerate them from their manifests under `tools/sbsbench/contracts/`
   and commit the manifest together with the regenerated output.

## The unifying principle: overlays are not scene content

Three of the work items below are the same bug wearing different clothes.

The mouse cursor, burned-in subtitles, and any future HUD are **overlays**: they are composited on
top of the scene by something that already knows they sit in front of everything. They are not
surfaces in the depicted world. When such an overlay is burned into the frame *before* depth
inference, three things go wrong at once:

1. **The depth model hallucinates a depth for it.** DAV2 sees high-contrast marks and treats them as
   part of the surface behind them, so a single line of text can receive several different depths
   across its own extent, and a cursor receives whatever the pixels beneath it suggest.
2. **The warp resamples it.** Thin, high-contrast strokes are the worst possible input to a
   resampling warp — this is precisely the artifact class the `rim_over` metric was built to detect
   (a thin bright ridge hugging a silhouette). White text with a black outline *is* a thin bright
   ridge by construction.
3. **It lands at the wrong depth.** An overlay that should be at a fixed, comfortable, authored
   depth instead floats at an inferred depth that disagrees with what it annotates or points at.
   In stereo this is actively uncomfortable, and for the cursor it destroys pointing accuracy.

The conservative treatment is: **keep a known overlay out of scene inference, then composite it
after the warp into both eyes at an authored disparity.** Everything in
[W1](#w1-composite-the-cursor-after-the-warp-candidate-defect) through
[W3](#w3-exclude-overlay-masks-from-depth-statistics) is an application of that one rule.

---

## Work items

### W1. Composite the cursor after the warp (candidate defect)

**Problem.** In the Desktop Duplication backend, Sunshine 3D alpha/XOR blends the separate cursor
shape into the captured frame before DAV2 and the stereo warp. Cursor-only updates can therefore
trigger full-frame depth work and the pointer can be resampled or assigned scene depth. The WGC
backend is different: Windows currently burns the cursor into the capture surface itself.

**Correct architecture.** Preserve the current flat-stream behavior, but give Host SBS a
backend-independent cursor layer:

1. capture scene color without a cursor;
2. carry cursor bitmap, hotspot, position, visibility, and update identity alongside the frame;
3. render stereo geometry from cursor-free color; and
4. composite one rigid cursor sprite into both packed eyes after the warp.

Default the cursor to **zero parallax / screen plane**, which preserves desktop pointing semantics.
A surface-following mode may sample a robust local median from the final parallax field, but it must
be opt-in and temporally smoothed because a hotspot on a depth edge otherwise flickers.

**Dependencies and gotchas.**

- DDup already supplies pointer shape and position. WGC needs independent Win32 pointer acquisition
  before its cursor capture can be disabled; merely moving the existing DDup draw does not cover it.
- Alpha and monochrome XOR cursors must retain their current visual semantics.
- Cursor-only acquisitions must not enqueue DAV2. They should reuse the last scene geometry and
  update only the post-warp overlay.
- The cursor metadata must be tied to the same captured-frame identity as the color layer; a stale
  cursor must never be applied to a later window position.
- Non-SBS streaming stays on the existing path until the shared overlay layer has equivalent tests.

**Acceptance.** Cursor pixels are absent from DAV2 input and V2 statistics; cursor-only movement
causes no inference enqueue; the sprite remains sharp and binocularly stable; both capture backends
and alpha/XOR cursors pass. Add pure tests for metadata identity and optional hotspot smoothing, plus
one GPU integration test for packed-eye placement.

**Effort.** Medium. It is still the first user-visible overlay fix, but it must follow the evidence
probe rather than being treated as a one-shader edit.

---

### W2. Subtitles (burned into the picture)

**Scope decision: soft-subtitle support is explicitly out of scope.** Subtitles are burned into the
picture in the content this project targets, both for hardsubbed offline sources and for anything
captured live from a browser. Do not build a soft-subtitle presentation path, and do not schedule
client-side per-eye subtitle rendering — the project has decided against it.

Two consequences for an implementer:

- **The burned-in treatment below is the whole subtitle story**, not one branch of three. It applies
  to the offline conversion path and the live capture path alike.
- **A latent footgun to close.** The offline mux pass still copies any subtitle stream that happens to
  exist in the source, with `-map 1:s?` and `-c copy`
  ([offline_sbs_worker.cpp:5018](../src/offline_sbs_worker.cpp:5018)). Since nothing will render such
  a stream correctly over an SBS-packed frame — a player drawing it once across the packed image puts
  half a sentence in each eye — a source carrying both hardsubs and a soft track can produce an
  output whose soft track auto-renders and looks broken. Clear the default disposition on copied
  subtitle streams (ffmpeg `-disposition:s 0`) so no player auto-selects one, while still preserving
  the data for anyone who wants it. Dropping the streams entirely is the alternative; preserving them
  disabled is the friendlier choice and costs nothing.

Note that the render pass already excludes subtitles with `-sn`
([offline_sbs_worker.cpp:3991](../src/offline_sbs_worker.cpp:3991), alongside `-an` and `-dn`). That
is correct and should stay — it just has no effect on subtitles that are already pixels.

#### W2a. Burned-in subtitles — zero-parallax mask

**Problem.** Hardsubbed sources and live browser capture both deliver subtitles already burned into
the pixels. `-sn` cannot help. Every failure mode in
[the unifying principle](#the-unifying-principle-overlays-are-not-scene-content) applies at full
strength. Additionally, subtitles sit near the bottom edge of the frame, so if they acquire negative
parallax while being cut by that edge, the result is a frame violation — occlusion and disparity
cues contradict each other, which is uncomfortable.

This is the primary subtitle case for this project, not an edge case. Hardsubbed releases dominate
Chinese-language content, and CJK glyphs carry far denser strokes than Latin ones, so both the
resampling damage and the rim artifact are more visible than they would be on Latin text. Tune and
test against CJK, not against English.

**Approach.** Do not try to warp text well. Exclude it.

Inside the detected subtitle region, force the parallax to a constant, and choose that constant to be
**zero**. Zero parallax means the region is identical in both eyes, which in turn means no
resampling, no disocclusion, and bit-exact glyphs, at essentially zero runtime cost — the region is
simply not warped. Screen-plane placement is also the conservative standard in stereoscopic
authoring.

Make the constant a configuration knob rather than a hard zero, so a small negative parallax (text
slightly in front of the screen plane) can be selected later if content that pops out is found to
conflict with the text.

**Gotchas — these determine whether the result is good or bad.**

- **Feather the mask edge.** A hard discontinuity in the parallax field will tear under the
  contractive inverse warp. Feathered over a few tens of pixels it becomes a gentle depth ramp
  instead.
- **Use a loose feathered rectangle, not a glyph-tight mask.** A tight mask is the intuitive choice
  and it is wrong: it places a parallax cliff at every single stroke edge. The loose rectangle costs
  only a small patch of background near the subtitles being flattened toward the screen plane, which
  is acceptable cardboarding. This is the single most important design decision in this item.
- Do not attempt inpainting behind the text. It is not needed: a region held at constant parallax
  translates rigidly, so the text stays sharp without any recovery of what is behind it.

**Detection.** Subtitles have a strong joint signature: high contrast (white or yellow, usually with
a dark outline), **piecewise static in time** (a line persists for one to five seconds then changes
abruptly), and dense in stroke gradients. The discriminating combination is *temporally
piecewise-static* **and** *high gradient*; contrast alone will also select station logos, watermarks,
and HUD elements — though since those deserve the same treatment, that particular false positive is
cheap.

Both the temporal-difference and gradient machinery already exist on the GPU in the depth pipeline;
prefer reusing them over adding a new pass.

Three properties of real hardsubbed content that a naive detector gets wrong:

- **Frame position is a weak prior, not a gate.** Do not restrict the search to a lower band. Fansub
  releases routinely place translated signs and notes at the *top* of the frame while dialogue runs
  along the bottom, and both need the same treatment.
- **Support several disjoint regions at once.** Top notes plus bottom dialogue is a common
  simultaneous case, so the mask must be a set of rectangles rather than a single one.
- **Bilingual stacking makes the region taller than one line.** Chinese-plus-English subtitle pairs
  are the norm in this content, so a height prior calibrated on single-line English will clip the
  second line and leave half the text being warped.

**The offline path can use stronger look-ahead without adding a whole-clip pre-pass.** Keep detection
inside the existing scene buffer: revise provisional cut boundaries with look-ahead, accumulate
piecewise-static high-gradient regions across that confirmed scene, then render the scene with its
stable mask. Carry a conservative band prior across adjacent scenes only as a hint, never as an
unverified mask. The live path remains causal and needs hysteresis. Build the shared mask consumer
first, then give the scene-buffered and live paths separate producers.

**Acceptance and measurement.** The repository already contains evidence of this failure mode:
`flat_page` is a synthetic static document page whose stated purpose is
"False stereo and depth hallucination on flat content"
([DATASETS.md:18](../tools/sbsbench/DATASETS.md:18),
[make_synth_clips.py:8](../tools/sbsbench/make_synth_clips.py:8)). That clip is full-page text, not
overlaid subtitles, so it is evidence but not a gate.

Add a subtitled clip to `tools/sbsbench/make_synth_clips.py` — an existing movie clip with white,
dark-outlined text composited over it — so the change has a gate. Include a CJK variant, since it is
the harder case.

Two new metrics are worth adding, and belong in the metric contract
([METRICS.md](../tools/sbsbench/METRICS.md)):

- **Disparity variance within the text region**, which should be approximately zero.
- **Text sharpness preservation**: the ratio of horizontal gradient energy in the text region after
  the warp to before it. Resampled glyphs lose high-frequency energy.

The existing `swim` metric already covers subtitle depth instability (its definition — frame-to-frame
depth change where the source is static — is exactly this case), and `rim_over` already covers the
bright fringe.

---

### W3. Exclude overlay masks from depth statistics

**This is the least obvious item here and possibly the highest value per line changed.**

**Problem.** Subtitles and the cursor are extreme outliers in luminance and in inferred depth. The V2
pipeline computes scene-wide statistics — a raw min/max EMA, histograms, and a scene-latched raw
centre — and those outliers contaminate them.

The visible consequence is not local. Every time a subtitle appears or disappears, the scene
statistics shift, and with them the depth mapping of the **entire frame**. The image appears to
breathe in and out in sync with the subtitles.

**Approach.** Carry the overlay mask produced in W1/W2 into the reduction shaders and exclude masked
texels from the min/max EMA, the histograms, and the scene-centre latch. The cost is one mask read in
each reduction.

**Gotcha.** The scene-cut detector is designed to resist brightness-only changes — [Host SBS scene
cuts](host-sbs-scene-cuts.md) states that brightness alone must not reset stereo geometry — so it
probably does not false-trigger on a subtitle appearing. **Statistical contamination is a separate
path and is not protected by that design.** Do not assume the cut detector's robustness covers this.

**Acceptance.** With a subtitled clip, global depth statistics should remain stable across subtitle
on/off transitions. This is directly measurable: the `swim` metric on frames where only the subtitle
changed should approach the noise floor.

---

### W4. A window model from Win32 and DWM

This is the enabling infrastructure for W5, W6, and W7. It has no user-visible effect on its own.

**Approach.** Maintain a live model of the desktop's window layout in the host:

- Enumerate with `EnumWindows`; establish Z-order by walking `GetWindow(hwnd, GW_HWNDNEXT)`.
- Take bounds from `DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS)`, **not** `GetWindowRect`.
  On Windows 11 the plain rect includes the drop shadow and does not describe the visible surface.
- Filter with `IsWindowVisible`, and skip cloaked windows via `DwmGetWindowAttribute(DWMWA_CLOAKED)`
  — cloaked windows are invisible but still enumerated, and including them corrupts the Z-order.
- Keep the model current with `SetWinEventHook` rather than polling. The relevant events are
  `EVENT_OBJECT_LOCATIONCHANGE`, `EVENT_SYSTEM_MOVESIZESTART` / `EVENT_SYSTEM_MOVESIZEEND`,
  `EVENT_SYSTEM_FOREGROUND`, and window create/destroy. Use an out-of-context hook so the host is
  not injected into other processes.

**Gotchas.**

- **DPI.** The host must be per-monitor DPI aware and window coordinates must be translated into the
  captured output's coordinate space. On multi-monitor systems the virtual desktop origin can be
  negative; the per-output offset is available from `DXGI_OUTPUT_DESC::DesktopCoordinates`, which
  the capture code already handles.
- Windows 11 rounded corners: `DWMWA_EXTENDED_FRAME_BOUNDS` accounts for the shadow but the visible
  window is not exactly its rectangle. For depth purposes this is a sub-pixel-scale error at the
  corners and can be ignored initially.
- Layered and tool windows, and always-on-top overlays, need deliberate policy rather than being
  silently included.
- During a window drag, geometry changes every frame. Consumers should degrade gracefully — see the
  latency note in W6.

---

### W5. Window Z-order as authored desktop planes

**Value.** Windows provides reliable top-level occlusion order and stable frame bounds. That is
excellent evidence for an authored XR desktop, but it is not metric scene depth and rectangular HWND
bounds are not an exact visible-region mask.

**Approach.** After W4, build a conservative window-plane compositor for regions that are positively
classified as desktop UI. Assign a shallow, capped set of comfortable planes from back to front. The
front-most ordinary application surface should remain near screen plane. Keep DAV2 for detected
video/game/natural-image content inside a window, and fall back to full-frame V2 whenever the
classification or visible-region mask is ambiguous.

**Required exclusions.** Do not treat these as ordinary opaque planes without additional evidence:
layered/transparent windows, non-rectangular regions, cloaked windows, menus/tooltips, shell surfaces,
DRM-black regions, and windows whose client content is moving at video cadence. A maximised or
full-screen video player is still video and must not collapse to one flat plane.

**What this can buy.** Static desktop regions become temporally stable, text is not fed through a
monocular natural-scene prior, and known overlap boundaries can be exact where the windows are
opaque rectangles. DAV2 cost can be skipped only for the positively classified desktop portion; the
composition and warp are not free.

**Acceptance.** Add a synthetic opaque rectangular window stack with exact authored disparity, then
separate adversarial cases for transparency, popups, taskbar overlap, and full-screen video. The
classifier must prefer full-frame V2 over a false desktop classification. Window-plane mode remains
off until those precision tests pass.

---

### W6. Video ROI detection

**Goal.** When a video plays in a window that is not full-screen, apply the stereo conversion to just
that rectangle, leaving the rest of the desktop flat.

**Detection approaches, in recommended order.**

1. **Desktop Duplication dirty rectangles (probe implemented).** A playing video region may be
   dirty on essentially every frame while the rest of the page changes intermittently. Phase 0.1 now
   samples `GetFrameDirtyRects` immediately after acquire in
   [`duplication_t::next_frame()`](../src/platform/windows/display_base.cpp:522), then reports
   two-second dirty-area and persistent-grid summaries. This is evidence, not an ROI implementation:
   high `RectsCoalesced` rates, diffuse damage, or unstable bounds invalidate the simple detector.
   **Caveat, and the reason this needs a measurement probe before it is designed against:** when
   video is presented through a hardware overlay (multi-plane overlay), that plane is composited by
   the display controller rather than by DWM, and the consistency of damage metadata across drivers
   is unverified.
2. **Content-based temporal activity.** The robust fallback, and it reuses machinery the depth
   pipeline already owns. Maintain a downsampled temporal activity map; video regions carry
   persistent energy while static page content is near zero. Take the largest connected active
   region with a plausible aspect ratio, then snap its edges to strong luminance edges — letterbox
   bars are easy to find. Weakness: a paused video produces no signal, so the ROI must latch with
   hysteresis rather than being recomputed from scratch.
3. **`GlobalSystemMediaTransportControlsSessionManager` (WinRT).** Gives no rectangle, but answers
   "is media actually playing, and in which process" almost for free. Excellent as a gate and to
   constrain the search to one window.
4. **UI Automation.** Can read a `<video>` element's `BoundingRectangle` directly, which is the most
   precise option, but querying it activates the browser's accessibility tree — a real cost for a
   host streaming at 90 fps — and cross-process COM traversal takes tens of milliseconds. If used,
   query once on latch, never per frame.
5. **Browser extension or DevTools protocol.** The accuracy ceiling: exact rects, `object-fit`
   letterboxing, and play state. Rejected as a primary path because of installation friction and
   because it covers only browsers.

**The hard part is not detection.**

- **Stereo window and frame violation.** Warping a sub-rectangle makes its border a stereo window.
  Content inside with negative parallax that is cut by that border produces contradictory occlusion
  and disparity cues. The floating-window treatment that is currently deferred for the full-screen
  case becomes **mandatory** here. The cheaper alternative is biasing the ROI's zero plane so all its
  content sits at or behind the screen plane.
- **Disocclusion at the ROI edge is better than full-screen, not worse.** Full-screen warping can
  only clamp at the frame edge. Inside an ROI, the pixels that should be revealed are actually
  present — they are the surrounding desktop. Sampling them is geometrically imperfect (that content
  is at screen depth, not video depth) but far better than smearing.
- **Aspect snapping is mandatory.** See
  [the resolution constraint](#a-resolution-constraint-that-will-bite-roi-work). Submit an
  authenticated aspect, not the raw detected rect, or the frame fails closed to flat.
- **Latency during window motion.** If the ROI lags a window drag, the warp is applied to the wrong
  pixels for several frames, which is a visible glitch. Shrink fast and grow slow, and use the
  `EVENT_SYSTEM_MOVESIZESTART` / `MOVESIZEEND` hooks from W4 to drop to flat during an active drag.

**A reality that will dominate user reports.** Hardware-DRM content — Netflix, Disney+, Prime — is
black in Desktop Duplication. The pixels are not obtainable, so no amount of detection makes ROI
conversion work there. YouTube, local files, and most non-DRM streams are fine. Say so in the UI
before a user discovers it.

**Probe status.** Phase 0.1 now instruments `GetFrameDirtyRects` without changing rendering. It
reports two-second dirty-area summaries and a persistent 32x18 activity-grid bounding box. The next
step is the live static/windowed/full-screen capture described in the evaluation section; no ROI
design is authorized until those logs show stable spatial concentration.

---

### W7. Local AR transparency from window geometry

Rides on W4 and W5 at almost no additional API cost, and serves the local AR glasses mode described
in [docs/sbs-local-ar-glasses.md](sbs-local-ar-glasses.md).

AR glasses of this class are additive displays: black is transparent. Given the window model, the
host can render only window pixels and leave everything else black, so windows appear to float in the
user's real room. Combined with W5's Z-order depth, each window sits at its own distance.

This is the shortest path to a genuine "AR desktop" and it needs no API surface beyond what W4
already establishes.

**Gotcha.** Additive displays cannot render black content — a window with a dark theme will appear
semi-transparent. That is a property of the hardware, not a bug to fix, but it should inform the
default theme recommendation.

---

### W8. Foveated encoding

**Approach.** The Galaxy XR client has eye tracking. Send gaze to the host and drive a per-region QP
map so the foveal region is encoded at higher quality than the periphery.

**Current state.** No QP map exists. The encoder configures spatial AQ only — `enableAQ` at
[nvenc_base.cpp:369](../src/nvenc/nvenc_base.cpp:369) — and lookahead is explicitly disabled at
[nvenc_base.cpp:363](../src/nvenc/nvenc_base.cpp:363). NVENC exposes a QP delta map / emphasis map
for this purpose; wiring it is new but localised work in a file the project already owns.

**Value.** At 3840×1080 SBS and roughly 200 Mbps, foveation can either recover substantial bitrate or
raise perceived central quality at the same bitrate.

**Gotchas.**

- The gaze round trip is client → host → encode → display. Saccades reach around 500°/s, so the
  high-quality region must be generously sized; do not fit it tightly to the reported gaze point.
- Losing gaze data must degrade to uniform quality, not to a stale foveal region parked in the wrong
  place.
- Verify the interaction with SBS packing: the foveal region exists in **both** eye images and the QP
  map must mark both.

---

### W9. Per-window capture and multi-display XR workspaces

Two routes to the same product goal — moving from "one flat desktop rectangle in the headset" to
multiple independently placed surfaces.

**Route A: per-window capture.** The infrastructure largely exists. `display_wgc.cpp` already uses
Windows.Graphics.Capture through `IGraphicsCaptureItemInterop`, but only ever calls
`CreateForMonitor` ([display_wgc.cpp:140](../src/platform/windows/display_wgc.cpp:140), also
[:78](../src/platform/windows/display_wgc.cpp:78)). `CreateForWindow` on that same factory is
single-window capture, and it works even when the window is occluded.

**Route B: multiple virtual displays.** The project already ships a virtual display driver. Letting
Windows do window management natively and streaming several clean rectangular displays is
substantially simpler than compositing arbitrary windows, and delivers the same productivity payoff.

**Honest cost.** Either route means a multi-stream protocol, multiple encoder instances, and
multiplied bandwidth. This is a large change and should not be attempted before the items above have
landed. Route B is the lower-risk of the two.

---

### W10. Per-process audio capture and spatialisation

`AUDIOCLIENT_PROCESS_LOOPBACK` (Windows 10 build 20348 and later, via
`ActivateAudioInterfaceAsync` with activation parameters) captures a single process's audio. It is
rarely used and it is exactly what a spatial desktop needs: each window's audio can be spatialised to
that window's position, so the video window's sound arrives from where the video is.

Without this, the spatial illusion is incomplete no matter how good the visual geometry becomes.
Depends on W4 for window positions and on W9 for the multi-surface case to be meaningful.

---

### W11. Hand tracking through the existing native touch path

The Windows host already injects native touch with `CreateSyntheticPointerDevice` and
`InjectSyntheticPointerInput` in `src/platform/windows/input.cpp`, and the streaming protocol
already carries normalized multi-touch contacts. Do not add the brokered
`Windows.UI.Input.Preview.Injection` API; it would duplicate working host functionality and carries
a restricted-capability requirement.

The remaining product work is client-side: convert Galaxy XR hand rays/pinch state into stable touch
contacts, map them through the displayed virtual-screen transform, and send the existing touch
messages. Required tests are contact-ID lifetime, cancellation when tracking is lost, multi-touch
ordering, and display-mode changes. Host changes should be limited to any protocol validation gaps
found by those tests.

---

### W12. Foreground-process and media-state classifier

`GetForegroundWindow`, process identity, window geometry, and optionally
`GlobalSystemMediaTransportControlsSessionManager` are useful **evidence**, not permission to
silently change pop strength. V2 no longer has movie/game profiles.

Use this signal to choose only among explicit rendering routes whose contracts are independently
validated: ordinary full-frame V2, conservative desktop planes, or a latched video ROI. Ambiguous or
stale evidence returns to full-frame V2. Any future automatic strength change needs a separate UI
contract and evaluation gate.

---

### W13. Damage-guided depth reuse (new review addition)

Desktop Duplication already distinguishes new desktop presentation from cursor-only acquisition and
can report dirty rectangles. After W1 separates the cursor, cursor-only frames can update the overlay
without enqueueing DAV2. A later conservative extension can reuse authenticated depth when damage is
empty or wholly confined to a known post-warp overlay/window-chrome mask.

Do **not** reuse depth merely because dirty area is small: a small foreground entering the frame can
be the most important geometry in the image. Move rectangles also need their own treatment because a
scroll moves valid old depth rather than proving it unchanged. The safe progression is:

1. measure dirty rectangles (Phase 0.1);
2. eliminate cursor-only inference through W1;
3. add move-rectangle instrumentation;
4. permit reuse only for independently classified non-scene damage; and
5. fail open to normal full-frame inference on metadata loss, WGC, DRM, or uncertainty.

This can reduce GPU work on interactive desktops without introducing another content classifier into
the critical path, but only the cursor-only case is initially strong enough to ship.

---

## Recommended implementation order

The ordering is driven by three things: fix defects before adding features, build shared
infrastructure before its consumers, and put cheap de-risking probes before expensive design.

### Phase 0 — Probes and gates (do these first; both are cheap)

| # | Item | Why first |
|---|---|---|
| 0.1 | **Completed:** dirty-rect instrumentation probe (part of [W6](#w6-video-roi-detection)) | Diagnostics-only, no rendering change. Two-second windows report dirty area and a 32x18 persistent-activity bounding box. Live evidence now decides whether DDup metadata is useful on the target driver. |
| 0.2 | Subtitled synthetic clips and the two new metrics ([W2a](#w2a-burned-in-subtitles--zero-parallax-mask)) | Repo process requires measuring before changing. Without a gate there is no way to demonstrate the subtitle mask works. Build the CJK and bilingual-stacked variants here, since they are the real content. |

#### Evaluate the completed dirty-rectangle probe

The probe is deliberately inactive in normal production. In `sunshine.conf`, use:

```ini
diagnostics = enabled
min_log_level = debug
```

Use the Desktop Duplication capture backend, start Host SBS, and collect at least three 10–20 second
segments: a static browser page, a windowed playing video, and the same video full-screen. Search
`sunshine.log` for `Desktop Duplication dirty-rect probe`.

Important fields:

- `window_frames=0` is the expected static-desktop result; it is not a missing sample.
- `failures=0` indicates that no probe operation failed in that interval.
- `coalesced_pct` shows how often Windows merged dirty regions; high values can include unchanged
  pixels and weaken spatial conclusions.
- `dirty_area_mean_pct` and `dirty_area_max_pct` show how much of the surface Windows marks.
- `persistent_tiles_pct` is the fraction of a 32x18 grid dirtied in at least 75% of acquired
  presentation frames.
- `persistent_bbox=L:T-R:B` is the quantized bounding rectangle of those persistent tiles, in the
  unrotated Desktop Duplication surface coordinates.
- `protected_seen=1` means at least one presentation in the interval reported DRM masking, so the
  interval cannot support content conclusions.

A static interval should report `window_frames=0`; cursor-only acquisitions and capture timeouts close
the two-second window without being treated as desktop damage.
The dirty-rect route is promising only if the windowed-video bounding box stays close to the video
rectangle and occupies substantially less than the full surface, while full-screen video expands
predictably. A full-surface, empty, or rapidly jumping box means dirty metadata is not a sufficient
ROI detector on that driver. This first probe intentionally does not aggregate `GetFrameMoveRects`;
scrolling is therefore diagnostic only and is not a pass/fail case yet.

### Phase 1 — The overlay compositing framework

| # | Item | Why here |
|---|---|---|
| 1.1 | [W1 — cursor after the warp](#w1-composite-the-cursor-after-the-warp-candidate-defect) | Medium cross-backend change: first create an independent WGC cursor layer, then build the post-warp compositor. |
| 1.2 | [W2a — subtitle zero-parallax mask](#w2a-burned-in-subtitles--zero-parallax-mask) | Reuses 1.1's compositing stage and its mask plumbing. Gated by 0.2. Ship the offline scene-buffered detector first; the live causal detector can follow. |
| 1.3 | [W3 — exclude masks from depth statistics](#w3-exclude-overlay-masks-from-depth-statistics) | Needs the masks that 1.1 and 1.2 produce. Small change, removes a whole-frame artifact. |
| 1.4 | [W12 — foreground/media classifier](#w12-foreground-process-and-media-state-classifier) | Evidence source only; it may choose a validated route but may not silently change V2 strength. |
| 1.5 | [W13 — damage-guided depth reuse](#w13-damage-guided-depth-reuse-new-review-addition) | Start only with cursor-only reuse after W1. Broader reuse remains gated on move metadata and independent damage classification. |

### Phase 2 — Window geometry

| # | Item | Why here |
|---|---|---|
| 2.1 | [W4 — window model](#w4-a-window-model-from-win32-and-dwm) | Enabling infrastructure for everything below. No user-visible effect alone. |
| 2.2 | [W5 — authored window planes](#w5-window-z-order-as-authored-desktop-planes) | Potentially high value, but only behind a high-precision desktop/content classifier. Needs 2.1. |
| 2.3 | [W6 — video ROI](#w6-video-roi-detection) | Needs 2.1 for window bounds and drag events, 0.1 for its detection route, and Phase 1's feathering discipline. Combines with 2.2 into the hybrid desktop mode. |
| 2.4 | [W7 — AR transparency](#w7-local-ar-transparency-from-window-geometry) | Nearly free once 2.1 and 2.2 exist. |

### Phase 3 — Larger bets, in decreasing confidence

| # | Item | Note |
|---|---|---|
| 3.1 | [W8 — foveated encoding](#w8-foveated-encoding) | Self-contained, in a file already owned. Needs a client-side gaze channel. |
| 3.2 | [W10 — per-process audio spatialisation](#w10-per-process-audio-capture-and-spatialisation) | Needs W4; becomes compelling with W9. |
| 3.3 | [W11 — hand tracking through existing touch](#w11-hand-tracking-through-the-existing-native-touch-path) | Host injection already exists; remaining work is primarily in Moonlight 3D. |
| 3.4 | [W9 — multi-surface capture](#w9-per-window-capture-and-multi-display-xr-workspaces) | Largest change here. Prefer the virtual-display route. Do not start before Phase 2 has landed. |

Soft-subtitle presentation and client-side per-eye subtitle rendering appear nowhere in this plan by
design — see the scope decision in [W2](#w2-subtitles-burned-into-the-picture). The only subtitle work
scheduled is the burned-in mask at 1.2, plus the one-line disposition guard described in W2.

---

## Open questions not yet verified

These were identified during analysis but not confirmed against code. Verify before relying on them.

1. **Does the offline conversion output declare container-level stereo metadata** (for example
   Matroska `StereoMode`)? Note the mux pass strips metadata with `-map_metadata -1`
   ([offline_sbs_worker.cpp:5028](../src/offline_sbs_worker.cpp:5028)), so any stereo signalling must
   be added deliberately. This is worth settling independently of subtitles: it is what lets any
   player recognise the file as side-by-side rather than as a very wide 2D video.
2. **Do sources in practice carry a soft subtitle stream alongside burned-in subtitles?** This sets
   the priority of the disposition guard in [W2](#w2-subtitles-burned-into-the-picture). If such
   sources are rare, the guard is a one-line precaution; if they are common, a user will hit the
   broken auto-rendered overlay early.
3. **Do Desktop Duplication dirty rectangles behave usefully when video is in a hardware overlay** on
   the target RTX 5080 driver? Phase 0.1 now emits the required two-second summaries; the live
   windowed/full-screen measurement is the remaining answer.
4. **What is the exact SBS packing order and coordinate convention in the current V2 renderer?**
   Any per-eye compositing work (W1, W2, W8) needs this exactly right, and it must be read from the
   current shaders rather than assumed from older documentation.
5. **How does the flat (non-SBS) path composite the cursor** once W1 moves compositing later? The
   flat path must keep working.

---

## Verified code locations referenced here

Every location below was confirmed against the current working tree on August 6, 2026.

| Location | What it shows |
|---|---|
| [display_vram.cpp:513](../src/platform/windows/display_vram.cpp:513) | `convert()` — where the SBS depth pipeline runs |
| [display_vram.cpp:4241](../src/platform/windows/display_vram.cpp:4241) | `display_ddup_vram_t::snapshot()` — capture, and where the cursor is blended |
| [display_vram.cpp:4294](../src/platform/windows/display_vram.cpp:4294) | `blend_mouse_cursor_flag` |
| [display_vram.cpp:4539](../src/platform/windows/display_vram.cpp:4539) | cursor VS/PS bind and `blend_alpha` — the pre-warp blend (W1) |
| [display_vram.cpp:4749](../src/platform/windows/display_vram.cpp:4749) | `display_wgc_vram_t::snapshot()` — the second capture backend |
| [display_base.cpp:542](../src/platform/windows/display_base.cpp:542) | `duplication_t::next_frame()` reads and summarizes `GetFrameDirtyRects` when diagnostics/debug logging are enabled (Phase 0.1/W6) |
| [display_wgc.cpp:140](../src/platform/windows/display_wgc.cpp:140) | WGC uses `CreateForMonitor` only; `CreateForWindow` unused (W9) |
| [nvenc_base.cpp:363](../src/nvenc/nvenc_base.cpp:363) | lookahead explicitly disabled |
| [nvenc_base.cpp:369](../src/nvenc/nvenc_base.cpp:369) | `enableAQ` is the only adaptive quantisation; no QP map (W8) |
| [offline_sbs_worker.cpp:3991](../src/offline_sbs_worker.cpp:3991) | render pass drops subtitle streams with `-sn`; no effect on burned-in text (W2) |
| [offline_sbs_worker.cpp:5018](../src/offline_sbs_worker.cpp:5018) | mux pass copies subtitle streams back with `-map 1:s?` — the disposition footgun (W2) |
| [host_sbs_resolution.h:19](../src/host_sbs_resolution.h:19) | source upper bounds: 5120 long side, 5120×2160 area |
| [host_sbs_resolution.h:151](../src/host_sbs_resolution.h:151) | the gate authenticates the **fitted tensor shape**, so it is aspect-driven |
| [DATASETS.md:18](../tools/sbsbench/DATASETS.md:18) | `flat_page` targets flat-content depth hallucination |
| [make_synth_clips.py:8](../tools/sbsbench/make_synth_clips.py:8) | `flat_page` expects near-flat depth from the model |
