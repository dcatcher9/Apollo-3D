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
  current ownership boundary. Per-frame tight masks must sanitize input before that frame reaches
  DAV2; the scene buffer may then revise candidates and finalize stable plan metadata without
  replacing the tight glyph mask with a flattened rectangle.
- W8, W10, W11, and W9 are not host-only changes. They require protocol and Moonlight 3D work, so
  they cannot be treated as local edits to one Windows source file.

The Desktop Duplication dirty-rectangle investigation in Phase 0.1 is complete and rejected as a
video-boundary authority. W6 now uses a supervised Chromium IAccessible2 helper to bind a fresh,
causally-prior `<video>` element rectangle observation to a matched Host-SBS frame. The production
Desktop-Duplication route now uses that result for a strict foreground-browser ROI when its aspect
can reach the current authenticated tensor with at most 2% inward area trimming. It never pads, expands,
or stretches the detected video. True fullscreen is a separate semantic route: an available video
may cover the complete foreground browser client despite element overscan, document-rectangle
clipping, or multiple full-cover clones, but the host must still map that client exactly to the
capture. A maximized/full-source browser alone is not authority. Observer, rectangle, fit, and
crop-resource eligibility failures leave the route on ordinary full-frame V2; internal base-V2
authentication failures still render flat. The optional `window_video_border.json` artifact is
diagnostic only.

## Context an implementer needs

### The pipeline as it stands

Host SBS has exactly one geometry implementation: Depth Coordinate V2. The legacy V1 renderer and
its configuration surface were deleted in commit `cf478187`. An invalid or unauthenticated V2 frame
renders **flat** SBS; there is no alternate renderer and no fallback geometry.

The live path is, in order: desktop capture (DXGI Desktop Duplication, with a Windows.Graphics.Capture
path alongside it) → cursor blend → exact foreground-video attribution or full-frame selection →
`convert()` → DAV2 depth inference → V2 coordinate mapping → signed post-limit field → bounded
bottom-band OCR8 detection and compact SLR8 final-field conditioning → an 11-iteration contractive
inverse warp → SBS packing → NVENC. The retired generic overlay sanitizer/exclusion graph is not in
this path.

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
  the shipped calibration and cannot be submitted directly.

The implemented Chromium ROI keeps the full-frame session's already authenticated tensor shape. If
the exact detected rectangle misses that aspect slightly, it may trim inward by at most 2% of its
area while retaining the largest centered portion. It never expands into a player frame, pads the tensor, or
stretches pixels. A larger mismatch or invalid rectangle selects ordinary full-frame V2; it does not
turn an otherwise supported Host SBS stream flat.

Separately authenticated true fullscreen is canonical full-frame V2: an available semantic video
must cover the complete foreground browser client and the host must map that client exactly to the
capture. It bypasses crop, trim, ROI-domain reset, and ROI embedding. Element overscan,
document-rectangle clipping, and multiple full-cover clones are allowed at helper selection; a
maximized/full-source browser without the semantic/client proof is not authority.

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
3. **It lands at inconsistent depth.** An overlay that should remain rigid on one comfortable plane
   instead inherits varying scene depth across its strokes. For burned-in text the implementation
   samples a nearby post-limiter value to reduce the surrounding cliff; a separate cursor can use an
   authored plane. Inconsistent stereo depth is uncomfortable and destroys cursor pointing accuracy.

The conservative treatment has two forms. A separately available overlay such as the DDup cursor
stays out of inference and is composited after the warp at an authored disparity. A burned-in
overlay cannot be separated from the delivered color: keep that original color untouched for final
rendering, make a sanitized **inference-only** copy, exclude its overlay cells from scene evidence,
and hold each pooled glyph component to its own stable plane in the final parallax field. Each plane
is sampled from ordinary post-limiter geometry rather than authored as zero, and a slope-safe collar
preserves a contractive inverse. [W1](#w1-composite-the-cursor-after-the-warp-candidate-defect)
uses the first form and [W2](#w2-subtitles-burned-into-the-picture) uses the second; W2 does not
depend on W1's sprite compositor.

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
- **The soft-track safeguard is implemented.** The offline mux pass still preserves any subtitle
  stream present in the source with `-map 1:s?` and `-c copy`, but now appends FFmpeg
  `-disposition:s 0`. Nothing can render one soft-subtitle plane correctly over an SBS-packed frame,
  so clearing every copied subtitle disposition prevents default/forced flags from driving
  auto-selection while keeping the stream data available to anyone who explicitly wants it. A
  player's language or explicit user policy may still select the stream; this is a best-effort
  safeguard, not a presentation feature.

Note that the render pass already excludes subtitles with `-sn`
([offline_sbs_worker.cpp:3991](../src/offline_sbs_worker.cpp:3991), alongside `-an` and `-dn`). That
is correct and should stay — it just has no effect on subtitles that are already pixels.

#### W2a. Burned-in subtitles — adaptive constant-plane mask

**Current implementation direction.** Burned-in subtitles remain part of the captured picture.
The current production path consumes bounded OCR8 subtitle boxes and conditions only the final V2
field through compact SLR8 rectangle state. It does not sanitize DAV2 input, classify arbitrary
overlays, retain pixel histories, or run the retired GST/OGR/ORS detector graph. At most four
same-frame authority rectangles receive the exact anisotropic slope-safe distance treatment; absent,
stale, mismatched, or unsupported evidence copies ordinary V2 bit-for-bit.

Acceptance still emphasizes CJK and bilingual material, authored tight-mask scoring, sharpness,
plane residual, transitions, cuts, and false-positive real-video evidence. Synthetic clips remain
deterministic contract tests rather than a substitute for a real-video distribution.

---

### W3. Overlay-mask reuse in scene evidence (deferred)

The current OCR8/SLR8 route does not sanitize DAV2 input and does not feed subtitle rectangles into
scene-cut evidence, V2 moments, or scene-center latching. It changes only the authenticated final
field. A confirmed cut is handled by the SLR8 transaction itself; no generic exclusion mask is
published.

Do not feed a subtitle box, cursor mask, HWND rectangle, browser chrome, or broad UI classifier into
cut/geometry reductions without authenticated traces proving that specific defect. Broad masks can
hide a genuine localized player cut, and text-box evidence alone does not establish that the
underlying scene evidence is invalid.

**Acceptance for any future reuse.** Show on real and deterministic conformance traces that the
proposed evidence source prevents the targeted false transition without hiding a localized player
cut or changing ordinary V2 outside its authenticated scope. Until then this item has no production
authority.

---

### W4. A window model from Win32 and DWM

This is the enabling infrastructure for W5, W7, and future generic or occlusion-aware extensions to
W6. The strict foreground-Chromium ROI does not depend on this top-level window model.

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

**Goal and current boundary.** The implemented windowed route finds the unique largest fully
contained Chromium `<video>` in the foreground browser, performs one crop-local V2 analysis, and
leaves the surrounding desktop at zero parallax beyond the required edge collar. A separate
true-fullscreen selection recognizes an available semantic video covering the whole foreground
browser client, which the host must then map exactly to the capture. Generic player windows,
background browsers, windowed games, occlusion reasoning, and multiple simultaneous 3D regions
remain later work.

**Rejected damage-metadata route.** Target-machine traces initially looked promising: a windowed
video repeatedly produced `820:510-2471:1439`, and maximise produced approximately the full capture.
Longer observation disproved dirty metadata as geometry authority. The same video alternated among
its full `278:416-2812:1841` envelope, a lower-only `278:896-2812:1841` rectangle, and small internal
updates; animated page furniture could also dominate a persistent-activity grid. These are all valid
answers to “which pixels changed,” but none answers “where is the video element.” Paused video has no
damage at all. The experimental exact-dirty-region tracker and its heuristics were therefore removed
rather than expanded. Sunshine no longer calls `GetFrameDirtyRects` for video discovery.

**Zero-extension Chromium semantic source.** `video-dom-info` walks the foreground Chrome or Edge
MSAA tree in a supervised process, obtains IAccessible2 through `IServiceProvider`, and accepts only
the exact escaped object attribute `tag:video`. Unlike dirty metadata, that identity and its
`accLocation` rectangle remain present when a video is paused or only a small part changes. Standard
UI Automation is not sufficient because Chromium exposes `<video>` as a generic `Group`; the HTML
tag survives in the extended IA2 attributes.

The helper scans only the foreground Chrome or Edge root. Background windows and background tabs
are deliberately ignored even if media continues playing there. For a windowed ROI it selects the
unique largest credible available video fully contained by its document and browser client; an exact
largest-area tie is ambiguous. Fullscreen is evaluated independently: any available semantic video
covering the complete client is eligible even if its element overscans, its available document
rectangle is clipped, or Chromium exposes multiple full-cover clones. The helper retains a stable
clone identity and publishes the client rectangle as authority. Both are semantic-content rules
rather than playback-state rules, which is why pause does not discard the selected border.

Build and run the opt-in probe from the configured RelWithDebInfo tree:

```powershell
ninja -C cmake-build-relwithdebinfo video-dom-info
cmake-build-relwithdebinfo/tools/video-dom-info.exe --all-scans --interval-ms 1000
```

Keep Chrome or Edge in the foreground during the measurement. The human diagnostic mode lists every
exact video node and applies a two-census display filter. Production machine mode instead stages a
candidate after one complete census, retains its document/video COM nodes without publishing
authority, and promotes it only when the next 100 ms tick independently confirms it. This avoids
requiring a second expensive whole-tree walk while preserving a separate observation boundary. For
windowed ROI, the available, fully contained videos must cover at least 5% of the foreground browser
client; the unique largest visible rectangle wins and equal-area leaders remain ambiguous. For
fullscreen, the retained video identity and available state must persist while its element continues
to cover the unchanged client. Document-rectangle clipping and harmless element overscan changes do
not revoke that fullscreen cache. Paused video needs no stale-identity exception.

Windowed partial clipping, an incomplete or changing tree walk, a foreground change, and unexpected
COM failures expose no initial ROI candidate. Fullscreen element overscan and document-rectangle
clipping are the explicit exception when the available video still covers the whole client. An
unselected incomplete census retries within one second; confirmed accessibility unavailability
retains the 15-second backoff so Chromium is not continuously walked. After promotion, a transient
incomplete full census cannot erase an independently revalidated cache. A foreground change or
failed cache check invalidates immediately; unrelated object events request a coalesced audit without
erasing a still-valid cached video, so dynamic controls and ads do not create a permanent rescan
loop. A complete fallback audit runs every 15 seconds. Because audits are coalesced to at most one
per three seconds, a newly appearing larger windowed video can replace a still-valid old selection
only after that bounded audit and semantic confirmation. Until replacement is confirmed, only the
independently revalidated current identity is eligible; a foreground change or stale/failed
validation immediately returns production to full-frame V2. Sunshine supervises the helper out of
process, accepts only a strict versioned record with a fresh heartbeat, and kills/restarts a silent
helper. The streaming path performs one atomic snapshot read and no COM, IPC wait, or GPU-to-CPU
image readback.

While Host SBS is active, it clips at most one physical pixel of Chromium/DPI endpoint overflow,
converts the helper's authority rectangle to an identity-oriented single-output capture, and binds a
causally-prior geometry run to one private matched color slot. The latest helper heartbeat may be
newer than paused source content, but the exact HWND/process/document/video/authority-rect tuple must
have remained continuously valid since no later than that content timestamp. For windowed ROI that
authority is the selected visible video rectangle. For fullscreen it is the browser client, so a
harmless change to element overscan or document geometry does not create a false domain change. DDup
uses `LastPresentTime`, never a later cursor-only timestamp. The host also rechecks foreground HWND
and PID immediately before binding.

The helper's fullscreen client authority is first mapped by the host and is admitted only when it
equals the full capture. It then canonicalizes to ordinary full-frame V2, with no crop, trim,
ROI-domain reset, or active-ROI dump rejection. A maximized browser or estimator full-source domain
without this semantic/client proof is not fullscreen authority. Otherwise the strict windowed slot
may trim the video rectangle inward by at most 2% of its area so its aspect maps to the current
full-frame authenticated tensor. It never trims more, grows into browser chrome, pads, or stretches.
A same-format D3D11 crop becomes the only DAV2 and ownership source for that frame; scene cuts,
center, and temporal histories are crop-local. The full captured texture remains the color source.
After V2, local parallax is multiplied by `ROI_width / source_width` to preserve pixel shift, and an
outside-only signed collar decays at the production slope limits to exact zero beyond the collar
without changing any ROI-interior value.

The versioned machine record distinguishes strict `ok` geometry from relaxed `ok-fullscreen`
client authority. The host accepts the latter only when its mapped rectangle equals the complete
capture; it cannot silently become a windowed ROI. That authority class is part of the live-overlay
domain key, so `ok`/`ok-fullscreen` transitions break detector history without treating repeated
heartbeats as domain changes.

This live route currently authorizes only Desktop Duplication. WGC has no `LastPresentTime`
equivalent for separating content from cursor-only compositor frames and therefore uses full-frame
V2. Diagnostics may record the frame-bound semantic border and the observer/mapping failure reason,
but that observation is not the live renderer authority by itself. Current Dump 3D schema 26 records
the crop-local analysis field, authoritative full-source placement/collar contract, and exact
full-source inverse map as distinct evidence. The strict reader accepts only the current schema and
identity; historical dump schemas are intentionally unsupported.

A paused page captured before the first valid geometry run still has no matched-frame border. The
host waits for a real `LastPresentTime`; once the geometry predates a presented content frame,
later identical heartbeats retain the border through pause without letting cursor updates or a new
rectangle retroactively authorize older desktop pixels.

Chromium's IA2 rectangle is still a semantic element box, not proof that every pixel is visible:
CSS clipping/occlusion, canvas or WebGL players, protected content, and disabled accessibility can
all defeat it. Reading extended IA2 attributes enables Chromium's richer accessibility
serialization process-wide, so browser CPU and memory overhead remain part of live qualification.
If an overlay is captured inside an otherwise authorized semantic rectangle, it remains ordinary
crop content unless a separately admitted burned-in-overlay treatment supplies the tight mask. The
ROI route itself does not recover unoccluded video or construct a compositor-visible mask.

NVIDIA RTX Video does not provide an alternate public discovery API. Chromium already owns the
decoded texture and content rectangle and explicitly enables NVIDIA processing on that D3D11 video
stream. The driver is therefore receiving identified application-owned video, not discovering an
external DOM rectangle that Sunshine can query.

**The rendering policy is deliberately minimal.**

- **Stereo window and frame violation remains a content limitation.** The renderer does not pull the
  ROI boundary to zero from inside, because that would compress legitimate video depth. It preserves
  every interior value and spends only the minimum slope-limited collar outside the fitted analysis
  rectangle before reaching the desktop's exact zero plane. A foreground object visibly cut by the
  fitted or DOM edge can still produce contradictory occlusion cues; monocular input cannot recover
  the hidden part of that object.
- **Full-source color remains available at the edge.** The renderer does not clamp sampling to the
  crop or stretch the cropped image. The outside collar can sample the captured browser/desktop
  pixels that are actually present, then ordinary zero-parallax surroundings remain unchanged.
- **Aspect adaptation is inward-only and bounded.** See
  [the resolution constraint](#a-resolution-constraint-that-will-bite-roi-work). At most 2% of the
  detected area may be removed to reach the current authenticated aspect. A larger mismatch uses
  full-frame V2.
- **Motion is exact-frame attributed.** Rendering consumes only the rectangle attached to the same
  private color/depth slot. Analysis-domain changes reset temporal/camera state, while a pure
  position change of the same-sized ROI retains it. Stale or partially off-monitor geometry uses
  full-frame V2 rather than warping the old rectangle.
- **Fullscreen authority is separate from windowed ROI.** When an available semantic video covers
  the complete foreground client and the host maps that client exactly to the capture, the ordinary
  full-frame domain continues without crop, temporal reset, or dump rejection. A maximized or
  otherwise full-source frame alone cannot take this route.

**A reality that will dominate user reports.** Hardware-DRM content — Netflix, Disney+, Prime — is
black in Desktop Duplication. The pixels are not obtainable, so no amount of detection makes ROI
conversion work there. YouTube, local files, and most non-DRM streams are fine. Say so in the UI
before a user discovers it.

**Status.** The dirty-region detector is retired. IA2 helper supervision, strict unique-largest
windowed selection, separate full-client semantic fullscreen selection, one-pixel endpoint tolerance,
screen-to-capture validation, matched-frame attribution, transition logging, bounded inward aspect
trim, same-format DAV2/ownership crop, crop-local state, full-source rendering, and the outside-only
zero-plane collar are implemented. Active ROI frames are represented by current Dump 3D schema 26;
ordinary replay remains fail-closed until it consumes both coordinate domains.

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
validated: ordinary full-frame V2, conservative desktop planes, or the exact matched-frame Chromium
video ROI. Ambiguous or stale evidence returns to full-frame V2. Any future automatic strength
change needs a separate UI contract and evaluation gate.

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

### Phase 0 — Probes and gates

| # | Item | Why first |
|---|---|---|
| 0.1 | **Completed and retired:** dirty-rectangle investigation (part of [W6](#w6-video-roi-detection)) | It proved that DDup damage can resemble a video rectangle, then disproved it as semantic authority on pause, partial updates, and dynamic pages. The probe/tracker code was removed. |
| 0.2 | **Implemented:** Chromium IA2 video-border attribution | The isolated helper keeps strict unique-largest selection for windowed ROI and separately accepts an available video covering the whole foreground client for fullscreen, including overscan/document clipping/duplicate-clone cases. It survives pause and publishes no COM work onto streaming threads. Host SBS validates causal ordering and requires the fullscreen client to map exactly to the capture; the optional dump artifact remains diagnostic only. |
| 0.3 | **Implemented:** subtitled synthetic clips and tight-mask metrics ([W2a](#w2a-burned-in-subtitles--adaptive-constant-plane-mask)) | Dense-CJK, tall-bilingual, disjoint top-plus-bottom, and high-resolution transition/cut fixtures cover consumer geometry and temporal evidence. Authored-region support and fitted constant-plane RMS remain evaluation evidence; absolute fitted plane has only the target-free 8.0% binocular representation/comfort bound. |

#### Evaluate W6 border attribution

Use the Desktop Duplication capture backend and start Host SBS with Edge or Chrome in the
foreground. Test one main video plus smaller previews, pause/resume, local-only pixel updates,
dynamic sidebars, maximise/restore, browser zoom, and a negative-coordinate monitor. The host log
should publish only transitions for a validated half-open capture rectangle. Fullscreen endpoint
rounding of at most one physical pixel is clipped. For windowed ROI, larger overflow, partial
clipping, stale helper heartbeat, unsupported aspect, or equal-largest videos must select ordinary
full-frame V2. Separately test fullscreen element overscan, a clipped-but-available document
rectangle, and multiple full-cover clones: these remain eligible only while one retained available
semantic video covers the unchanged foreground client. An incomplete walk cannot create or replace a
border; it may only preserve a previously selected object that still passes the independent cached-object
check for its route.

An authenticated fullscreen client rectangle must map exactly to the capture and then canonicalize
to ordinary full-frame V2 with no crop, trim, analysis-domain reset, or ROI embedding. A maximized
browser or full-source analysis domain without that semantic/client authority must not claim an ROI.
This route-level fallback is distinct from the base fail-closed contract: a model, provenance, state,
field, or renderer authentication failure must render flat rather than retrying through full-frame
geometry.

For accepted ROIs, verify that aspect fitting removes at most 2% of the detected area from inside the
video, uses the current authenticated tensor, and never pads or stretches. Inspect both disparity
signs at all four edges: the ROI interior must remain unchanged, the outside-only collar must respect
the production slope limits, and the farther surround must be exactly zero. A pure window
translation must retain scene state; ROI/full, identity, size, and transfer-domain changes must
  reset it. Capture a schema-26 ROI Dump 3D and require its full-source inverse map to be identity
beyond the conservative collar; the crop-local final field alone is not sufficient evidence.

### Phase 1 — Overlay treatment

| # | Item | Why here |
|---|---|---|
| 1.1 | [W1 — cursor after the warp](#w1-composite-the-cursor-after-the-warp-candidate-defect) | Medium cross-backend change: first create an independent WGC cursor layer, then build the post-warp compositor. |
| 1.2 | [W2a — burned-in subtitles](#w2a-burned-in-subtitles--adaptive-constant-plane-mask) | Current work is the bounded OCR-box/final-field route described above. The retired general overlay detector, inference sanitizer, and component-local mask graph are no longer production or replay dependencies. Semantic, cross-GPU, real-video, and headset qualification remain required. |
| 1.3 | [W3 — overlay-mask reuse (deferred)](#w3-overlay-mask-reuse-in-scene-evidence-deferred) | No OCR8/SLR8 mask currently enters cut evidence or scene-center moments; any future reuse remains evidence-gated. |
| 1.4 | [W12 — foreground/media classifier](#w12-foreground-process-and-media-state-classifier) | Evidence source only; it may choose a validated route but may not silently change V2 strength. |
| 1.5 | [W13 — damage-guided depth reuse](#w13-damage-guided-depth-reuse-new-review-addition) | Start only with cursor-only reuse after W1. Broader reuse remains gated on move metadata and independent damage classification. |

### Phase 2 — Window geometry

| # | Item | Why here |
|---|---|---|
| 2.1 | [W4 — window model](#w4-a-window-model-from-win32-and-dwm) | Enabling infrastructure for W5, W7, and broader W6 extensions. No user-visible effect alone. |
| 2.2 | [W5 — authored window planes](#w5-window-z-order-as-authored-desktop-planes) | Potentially high value, but only behind a high-precision desktop/content classifier. Needs 2.1. |
| 2.3 | [W6 — video ROI](#w6-video-roi-detection) | The strict foreground-Chromium/DDup ROI is implemented without a generic window model. Broader players, background windows, games, occlusion-aware geometry, or multi-window composition still depend on later W4/W5 work. |
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
2. **How often do sources carry a soft subtitle stream alongside burned-in subtitles?** The
   disposition safeguard in [W2](#w2-subtitles-burned-into-the-picture) is unconditional either way;
   this question now measures how often it prevents disposition-driven broken auto-rendering.
3. **How broadly does Chromium IA2 video geometry work in practice?** The target Edge pages produced
   stable playing, paused, multi-video, maximise, and restore rectangles, but canvas/WebGL players,
   CSS occlusion, accessibility policy, iframe structure, and browser overhead still need a larger
   qualification set. Every miss must remain a no-border/full-frame result.
4. **What is the exact SBS packing order and coordinate convention in the current V2 renderer?**
   Any per-eye compositing work (W1, W2, W8) needs this exactly right, and it must be read from the
   current shaders rather than assumed from older documentation.
5. **How does the flat (non-SBS) path composite the cursor** once W1 moves compositing later? The
   flat path must keep working.

---

## Verified code locations referenced here

Every location below was confirmed against the current working tree on August 7, 2026.

| Location | What it shows |
|---|---|
| [display_vram.cpp:514](../src/platform/windows/display_vram.cpp:514) | `convert()` — where the SBS depth pipeline runs |
| [display_vram.cpp:4456](../src/platform/windows/display_vram.cpp:4456) | `display_ddup_vram_t::snapshot()` — capture, and where the cursor is blended |
| [display_vram.cpp:4518](../src/platform/windows/display_vram.cpp:4518) | `blend_mouse_cursor_flag` |
| [display_vram.cpp:4813](../src/platform/windows/display_vram.cpp:4813) | cursor blend state — the pre-warp blend (W1) |
| [display_vram.cpp:4975](../src/platform/windows/display_vram.cpp:4975) | `display_wgc_vram_t::snapshot()` — the second capture backend |
| [video_dom_probe/main.cpp:425](../tools/video_dom_probe/main.cpp:425) | bounded Chromium IA2 tree scan and exact `tag:video` discovery |
| [video_dom_probe/main.cpp:1249](../tools/video_dom_probe/main.cpp:1249) | cached machine-mode validation, coalesced WinEvent audits, and strict heartbeat protocol |
| [video_dom_client.cpp:317](../src/platform/windows/video_dom_client.cpp:317) | supervised unelevated helper lifecycle and atomic host snapshots |
| [video_dom_client.cpp:668](../src/platform/windows/video_dom_client.cpp:668) | one-pixel-tolerant physical-screen to capture mapping |
| [display_vram.cpp:1810](../src/platform/windows/display_vram.cpp:1810) | matched color slot owns its optional video border identity |
| [display_vram.cpp:1910](../src/platform/windows/display_vram.cpp:1910) | causally-prior helper geometry is validated and bound to the matched frame |
| [sbs_debug_dump.cpp:2107](../src/platform/windows/sbs_debug_dump.cpp:2107) | Dump 3D fails closed on stale or mismatched optional border evidence |
| [display_wgc.cpp:140](../src/platform/windows/display_wgc.cpp:140) | WGC uses `CreateForMonitor` only; `CreateForWindow` unused (W9) |
| [nvenc_base.cpp:363](../src/nvenc/nvenc_base.cpp:363) | lookahead explicitly disabled |
| [nvenc_base.cpp:369](../src/nvenc/nvenc_base.cpp:369) | `enableAQ` is the only adaptive quantisation; no QP map (W8) |
| [offline_sbs_worker.cpp:3991](../src/offline_sbs_worker.cpp:3991) | render pass drops subtitle streams with `-sn`; no effect on burned-in text (W2) |
| [offline_sbs_worker.cpp:5018](../src/offline_sbs_worker.cpp:5018) | mux pass copies subtitle streams with `-map 1:s?` and clears their dispositions (W2) |
| [host_sbs_resolution.h:19](../src/host_sbs_resolution.h:19) | source upper bounds: 5120 long side, 5120×2160 area |
| [host_sbs_resolution.h:151](../src/host_sbs_resolution.h:151) | the gate authenticates the **fitted tensor shape**, so it is aspect-driven |
| [DATASETS.md:18](../tools/sbsbench/DATASETS.md:18) | `flat_page` targets flat-content depth hallucination |
| [make_synth_clips.py:8](../tools/sbsbench/make_synth_clips.py:8) | `flat_page` expects near-flat depth from the model |
