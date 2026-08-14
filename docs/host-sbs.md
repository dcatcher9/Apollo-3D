# Host SBS pipeline

This document is the canonical design and runtime contract for Sunshine 3D's host-side mono-to-SBS
conversion. Host SBS has one geometry implementation: Depth Coordinate V2. An invalid or
unauthenticated V2 frame renders flat SBS; there is no alternate renderer.

Client SBS is a separate Moonlight 3D pipeline and does not share these constants. Offline
conversion uses the same V2 geometry and is described in
[Offline Host 3D conversion](whole-clip-sbs-pipeline.md).

## Pipeline

```mermaid
flowchart LR
    CAPTURE["Matched source frame"]
    REGION["Chromium video or foreground-client observation"]
    MATCH["Priority and exact matched-frame region selection"]
    DOMAIN["Full frame or exact whole-client/video ROI"]
    PREPROCESS["HDR/SDR model preprocessing"]
    DAV2["Authenticated DAV2 Small inference"]
    CUT["Cut-only evidence and scene epoch"]
    OCR["PP-OCRv6 tiny detector (bottom 960x160)"]
    LOCATOR["OCR8 boxes and compact SLR12 tracker"]
    CAMERA["Scene-latched raw center"]
    CURVE["Fixed raw coordinate and curve"]
    CONTAINER["Pointwise soft source-U container"]
    OWNER["Conservative ownership correction"]
    VERTICAL["75/25 vertical envelope share"]
    ROW["Horizontal majorant"]
    PLANE["Post-limit subtitle plane and analytic collar"]
    INVERSE["11-step contractive inverse"]
    COLOR["One native linear-color sample"]
    ENCODE["Packed SBS and NVENC"]

    CAPTURE --> MATCH --> DOMAIN --> PREPROCESS --> DAV2
    REGION -. optional authority .-> MATCH
    DAV2 --> CUT --> CAMERA
    DAV2 --> CAMERA --> CURVE --> CONTAINER --> OWNER --> VERTICAL --> ROW --> PLANE --> INVERSE
    CAPTURE -. completed calibrated DAV2 observation .-> OCR --> LOCATOR
    CUT --> LOCATOR --> PLANE
    CAPTURE --> COLOR
    INVERSE --> COLOR --> ENCODE
```

Only the final authenticated parallax field, including optional adaptive overlay-plane
conditioning, has rendering authority. Raw depth, canonical coordinate, candidate parallax,
ownership-refined parallax, vertical-envelope output, and the unconditioned row majorant are
diagnostics that explain how the final field was produced.

## Authenticated production contract

The generated Depth Coordinate contract is the machine-readable authority. The current identity is
schema 50/tag `0x84C0CB62`, canonical SHA-256
`c452c4211f3b7bdfe566bdb2117adf15e3f663aeb1361a91514d5ed1130009a0`. It binds the
complete policy below, including all subtitle field/ROI semantics, and producer source-closure
SHA-256 `42d2ee8d50520f370d2052c2e21ade7b9b0eb9ba985e936f907d966fe3bc786f`.
The live-renderer source-closure SHA-256 is
`5850bb757becd0c4d359812298974de72b073a4be0279d3ba41c6c1a5c270af1`, and dump-only diagnostic
renderer source-closure SHA-256
`077eefb9830c6a9322210fe1059cc765aa52a7b369f8414249795b0f43e96aaa`. It admits the following
production calibration:

| Property | Production value |
|---|---|
| Model | Depth Anything V2 Small FP16 |
| Model key | `depth_anything_v2_fp16` |
| Tensor shapes | `770x434`, `1022x434`, `1036x434`, and their portrait transposes |
| Raw coordinate scale | `2.25` DAV2 units |
| Gain per pop unit | `0.00375` source U |
| Direct parallax container | `0.04` source U per eye |
| Far exponential tau | `0.75` |
| Near logarithmic tau | `0.5` |
| Maximum horizontal slope | `0.5` |
| Maximum vertical shear | `2.0` |
| Vertical upper-envelope share | `0.75` |
| Convergence translation | exactly `0` |

The model bytes, model URL, preprocessing profile and shader closure, tensor shape, ordered
producer closure, state checksum, and renderer closure must also authenticate. Live Host SBS,
production Web UI conversion, and the maintained benchmark harness all use this one pinned model.
There is no supported Host SBS model selector. Unsupported model identities and tensor shapes fail
closed.

Fixed-shape shader bytecode is cached across restarts under the executable's trusted configuration
directory at `shader-cache/host-sbs-v1`. Each artifact filename is keyed by the authenticated source
closure, ordered entrypoint/target, and compile flags. Runtime reflection validates the cached stage
and Shader Model 5.0 bytecode before use; missing, stale, truncated, or invalid artifacts are
compiled from the immutable source snapshot and replaced atomically. This is only a startup
optimization and cannot weaken source-closure or shader-creation fail-flat behavior. Successful
prewarm retains the 128 most recently used artifacts so superseded closures cannot grow without
bound.

### Authenticated resolution fitting

Moonlight 3D's 12 standard XR resolutions all fit one of the six authenticated tensors:
`770x434`, `1022x434`, `1036x434`, and their portrait transposes. A custom source resolution is
also valid when the production fitter maps it exactly to one of those shapes; for example, a
same-aspect `1280x720` source still maps to `770x434`. A custom aspect whose fitted tensor is not
allowlisted is rejected before live Host SBS starts. The host never substitutes a nearby tensor or
silently emits a flat stream for an unsupported setup.

Offline V2 conversion uses the same fitter and allowlist. It aborts an unsupported job rather than
publishing a flat converted video. Other documents link to this section instead of maintaining a
second resolution list.

Portrait is an explicit `width < height` display mode. Host SBS does not rotate captured content;
non-identity Windows display rotation is rejected before pipeline setup.

The window-region route does not introduce another model shape. It keeps the current full-frame
authenticated tensor shape and copies the selected Chromium-video or foreground-client rectangle
exactly, regardless of its aspect ratio. The preprocessor fits that whole source rectangle into a
deterministic centered integer content rectangle. It never crops or stretches source pixels.
Synthetic tensor pixels outside the content rectangle replicate the nearest content edge and are
excluded from depth statistics, scene-cut evidence, ownership, history, and OCR authority. The
published parallax field extends its content boundary through this padding so renderer filtering
cannot turn padding into a false depth shelf.

Chromium true fullscreen is selected separately from the strict semantic-video ROI candidate. The
helper may recognize an available semantic `<video>` that covers the complete foreground browser
client even when the element overscans that client, its owning document rectangle is clipped, or
Chromium exposes multiple full-cover clones. The helper publishes the client rectangle as the
fullscreen authority, and the host accepts that authority only when the client maps exactly to the
capture extent. This semantic proof is distinct from the lower-priority generic foreground-client
route. Any selected foreground client that independently equals the complete capture also
canonicalizes to ordinary full-frame V2 rather than creating a redundant crop. Once admitted, either
exact-full-capture case is canonical ordinary full-frame V2:
it is not cropped or trimmed, does not enter a new ROI analysis domain, and does not require the ROI
embedding branch. Dump 3D records it as the canonical full-source analysis domain.

## Color and HDR

The model and the rendered color have different color requirements:

- BGRA8 SDR capture is interpreted as display-referred sRGB for model preprocessing.
- FP16 scRGB HDR capture remains linear Rec.709/scRGB for rendering. The inference copy applies a
  luminance-preserving absolute tone map and the sRGB transfer needed by DAV2; it does not clamp
  highlights to `1.0` before inference.
- The stereo warp applies no transfer function, gamut conversion, sharpening, or post-warp blur.
  Each eye takes one sample from the original source texture in its native linear or encoded domain.
- The existing encoder conversion stage produces SDR YUV or Rec.2100 PQ and carries the validated
  HDR metadata. Host SBS does not independently reinterpret the stream's color metadata.

Tone-mapped debug PNGs are viewing aids, not numeric HDR evidence. Use the floating-point dump
artifacts and manifest color fields when auditing the pipeline.

## Scene camera and raw coordinate

For each finite raw DAV2 field, the producer calculates exact extrema, arithmetic mean, and
population standard deviation. Standard deviation is validity evidence only. A field with
`sigma <= 1e-6` is collapsed and cannot produce current-frame geometry.

At startup or after a confirmed cut, the first usable field acquires its arithmetic mean as the
scene center. This gives occupancy-weighted behavior without a discrete scene classifier: a small
near object barely moves the center and retains relief, while a large near region naturally pulls
the zero plane toward itself and is not boosted as an isolated object.

For an authorized window-region ROI, extrema, mean, standard deviation, cut evidence, and all depth
histories are computed from the cropped analysis domain only. Pixels outside the selected Chromium
video or foreground client do not pull its scene center. Entering or leaving ROI analysis, changing
its authority kind, identity or dimensions, or changing its input transfer domain resets the
temporal and scene-camera state before the new domain is used. Chromium-video and foreground-client
authority are distinct even if their rectangles happen to match. Translating the same ROI without
changing its dimensions is not a new analysis domain, so an ordinary window move does not by itself
reacquire the camera.

### OCR-box subtitle conditioner

The production subtitle path is the single current detector-only PP-OCRv6 tiny/OCR8/SLR12 route.
It does not run text recognition, language classification, or logo recognition. For a completed
DAV2 observation in any of the six calibrated landscape/portrait fields, the GPU takes the exact
bottom `6:1` analysis-source crop, resizes it to
`960x160`, converts BGR through the pinned ImageNet normalization, and runs the authenticated
`ppocrv6_tiny_det_modelopt_fp16` TensorRT 11 strong-typed mixed-FP16 engine with FP32 I/O
(`trt-strong-modelopt045-fp16-iofp32-tf32-fixed960x160-level5-v2`). The bundled artifact is
`models/ppocrv6_tiny_det_modelopt045_mixed_fp16_fp32io.onnx`, SHA-256
`169a233ba0ff7cac27f8ec7dccb6a406e614b25b21fe6a5638c423bf2118bb44`. It is derived by
NVIDIA ModelOpt `0.45.0` using
`nvidia-modelopt-autocast-fp16-keep-io-fp32-v1` and calibration profile
`apollo-live8-bottom960x160-v1` from the pinned upstream FP32 ONNX, SHA-256
`193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8`. The contract
authenticates the bundled artifact and upstream source independently. The detector's
`1x1x160x960` FP32 probability map is reduced on the GPU to bounded line boxes; no probability
texture or model input becomes rendering authority.

When the authenticated detector is available, OCR runs once for every depth observation accepted
by the existing nonblocking DAV2 stream gate. It does not run for source frames dropped while the
previous inference is busy, and it does not reuse stale boxes or add a separate host cadence. This
keeps OCR8 and SLR12 on the same exact-frame authority path without extra readback, drain, probe, or
subtitle-onset delay.

OCR8 is a fixed 208-word current record: a 16-word header followed by paired tight/core and cover
slots, with at most eight authoritative pairs. Each eight-word slot stores half-open coordinates,
score bits, a ribbon-kind bit, island count, and structural-gap count. Tight cores lie in the active
calibrated DAV2 field's dynamic bottom
ROI obtained by projecting detector rows `[24,155)` through the exact bottom-6:1 source crop. For a
16:9 source and `770x434` field this remains `[325,430)`; ultrawide and portrait fields have different
row intervals. Schema/tag,
authoritative flag, exact matched-frame and analysis-generation identity, source dimensions, field,
ROI, finite score, paired containment and metadata, topology, counts, and all unused words must
validate together. `flags == 1` with
zero boxes is an authoritative empty observation. Abstaining, incomplete, overflowing, stale, or
malformed output has no geometry authority.

OCR8 first discovers broad horizontal runs with a 12-cell inactive-gap allowance solely to decide
whether their unfiltered topology is a bottom ribbon. It does not prefilter small or weak islands
from this broad pass: a ribbon is labelled only when the broad run is bottom-attached within two
detector pixels, spans at least half the detector width, and contains at least three inactive gaps
of three or more eight-pixel cells. A broad run that is not a ribbon is always rescanned with the
ordinary four-cell join allowance, even when its broad aggregate would fail the ordinary evidence
gate. In that rescan each island must independently have width and height of at least three detector
pixels and mean score at least `0.4` before it can contribute. Each resulting ordinary subrun
recomputes its tight bounds, score, island count, and structural-gap count from only those retained
islands. Thus a weak speck cannot borrow a subtitle's confidence or bridge its cover across a
foreground object, while a segmented bottom index remains one ribbon.

A ribbon's expansion pad is capped at eight detector pixels; ordinary subtitle expansion is
unchanged. A ribbon's final cover is the canonical bottom strip
`[0, corrected_top, field_width, field_height)`, while its paired raw box remains tight evidence.
The detector tolerance is preserved after coordinate conversion: the inclusive minimum raw-core
bottom is the exact rational ceil projection of detector row `155 - 2 = 153` through the same
bottom-6:1 source crop into the active DAV2 field. A core at that projected row is valid; one field
row below it is malformed. The upper bound remains the projected safe ROI bottom.

SLR12 is a fixed 80-word compact state with schema `12` and little-endian tag bytes `SL12`,
containing at most four owner/core rectangles, four
pending/core rectangles, and four same-frame current cover rectangles. Header word 31 carries
owner, pending, and current ribbon masks in bits `0..3`, `4..7`, and `8..11`; higher bits are zero.
Generic ordinary-line geometry requires
width at least 48 cells, height at least 6 cells, width at least twice height, and width at most
`floor(9 * field_width / 10)`; there is no hard-coded logo position. DAV2 field cells remain square
and every calibrated shape has short side 434, so the lower cell thresholds and sampling offsets
stay fixed while only the maximum line width follows the long side. Vertically adjacent boxes form
one coherent centered, left-aligned, or right-aligned
stack, chosen deterministically by area. A detected ribbon joins that stack as another tracked
member, and an observation exceeding four total members abstains rather than dropping one.
Individual line rectangles and their gaps are preserved.
Ordinary, horizontally disjoint cores may nevertheless join the same transitive component when
they share a strong baseline: vertical overlap is at least three quarters of the shorter height,
the taller height is at most twice the shorter, doubled vertical-center separation is at most the
shorter height, horizontal separation is at most eight times the taller height, and the pair's
combined span is at most `floor(9 * field_width / 10)`. Transitive closure is accepted only when
the complete component also stays within that same maximum span. This relation affects only component
selection and the shared plane target; it never unions geometry. Every core and paired cover
remains independent, canonical core order is top then left, and paired current covers retain that
core-defined order. A selected set needing more than four rectangles fails flat.
The first exact observation is pending; only a compatible observation with a distinct exact
frame/domain identity confirms an owner. Compatibility requires the same member count and kind in
canonical order and IoU at least `0.6` for every corresponding member; high aggregate overlap from
other unchanged lines cannot confirm a disjoint replacement. A compatible subset can remove a line immediately, while
an appended or materially changed stack remains pending and conditions only lines still matched to
the old owner until its second observation confirms the handoff.

Only rectangles copied from the current OCR8 record can condition the current frame. Cached owner,
pending, target, and six-observation death grace cannot manufacture geometry. Authoritative empty
OCR removes all current rectangles immediately. Missing, stale, abstaining, or malformed OCR in an
otherwise valid unchanged domain also clears current and pending immediately, starts or advances
the same six-distinct-observation cached-target grace, and conditions exact Base. Redispatching the
same observation does not consume another grace step or reapply a same-frame cut pulse. A hard cut clears pending/grace, retains only
current rectangles that still overlap an old owner, and samples a new plane; disjoint boxes
begin a new two-observation transaction. An input-domain reset clears the owner as well and treats
any current boxes as the first pending observation. This lets a seek/reset landing on an
already-visible static subtitle acquire on the following distinct observation without an onset
edge.

The observed plane comes from two independent 16-sample rows above the combined lower-text owner
stack. Each row is sorted and is coherent only when its Tukey interquartile range—the average of
elements 11/12 minus the average of elements 3/4—is at most `8` binocular source pixels. A row with
invalid or out-of-container samples is independently unusable and does not invalidate the other
row. No coherent row makes the observation unreliable. If only one row is valid and coherent, its
median is the desired plane. If both rows are valid and at least one is coherent, median separation
of at most `4` pixels selects their mean; larger separation selects the larger-U median, even when
that nearer row crossed the IQR gate. This avoids placing text behind the nearer supporting surface
while still requiring one stable row as evidence. A target of `k` pixels is
stored as `k / (2 * analysis_source_width)` signed one-eye source U; positive and negative targets
are both permitted up to the signed direct-parallax container. There is no absolute screen-plane
or near-screen clamp. These units are exact binocular output pixels for a native 1:1 SBS encode;
packed-output downscaling reduces visible disparity by the eye-content-width to full-source-width
ratio.

Every distinct authoritative continuing, handoff, or grace-rebirth observation resamples that
local plane. A fresh birth has no inherited target and starts directly on the reliable observation.
A confirmed same-scene handoff inherits the previous owner target, and a grace rebirth inherits its
valid cached target; a difference of at most `1` binocular source pixel then preserves those bits
exactly. A reliable residual above `8` pixels reacquires the new supporting plane at half strength
instead of slowly dragging the old plane through the scene. Otherwise the inherited or continuing
target takes an exact `1/8` EMA step limited to `0.25` pixel per distinct observation. Hard cuts
never inherit either seed. Duplicate identities do no target arithmetic. A continuing same-scene owner with current geometry may hold its previous
target, covers, and fade through at most two distinct unreliable measurements; header word 25 stores
this count while an owner exists and still stores death grace when no owner exists. Duplicates and
observations without current OCR authority do not age the hold; the latter preserve an existing
counter and valid target but clear current covers and condition exact Base. The third distinct unreliable
measurement resets target authority and publishes exact Base; the next reliable observation
reacquires at half strength. Fresh owners and handoffs without a valid target never use this hold.
Grace caches only a valid filtered target. A hard-cut survivor likewise restarts from the reliable
new local plane at half strength, or publishes Base when neither row is reliable; it never
conditions the new scene with the old full-strength target.
A domain reset clears the owner and target so present geometry starts a new pending transaction.
New owners fade from half to full strength over two observations. The conditioner then evaluates
distance directly to
each current half-open cover. For integer cell `(x,y)`, `dx`/`dy` are zero inside a rectangle
and count cells to its nearest included edge outside it:

```text
d      = min(dx * 0.5 / field_width + dy * 2.0 / field_width)
budget = 0.5 / source_width + d
```

Base values already within `budget` of the target are copied bit-for-bit. Values outside it move
only to `target +/- budget`, with the half-strength birth/handoff fade when applicable. Thus each
ordinary line has a dense cover and a V2-slope-safe analytic collar, nearby line collars may meet
naturally, and the gap is never converted into one merged rectangle. Because a ribbon cover spans
the complete field width and reaches the field bottom, its only exterior boundary—and therefore its
only collar—is the corrected top edge. Missing current authority, invalid target
state, unsupported tensor shape, or any identity failure copies ordinary post-limit V2 exactly.
The conditioner validates the complete 80-word state once per `16x16` dispatch group, requires its
scene epoch to equal the authenticated CutBridge hard-cut count bound at `t1`, and shares that
verdict within the group; no thread can partially accept malformed or cross-scene state.
There is no pixel history, row lease, onset accumulator, signature, horizontal-distance texture,
full-resolution overlay detector, or GST/OGR/ORS dependency.

## Pop and the pointwise soft container

`sbs_3d_pop_strength` is the only live geometry control. V2 uses it literally:

```text
requested_gain = pop_strength * 0.00375
requested      = requested_gain * F(u)
candidate      = requested / fourth_root(1 + (requested / 0.04)^4)
```

The configured default is owned by [Configuration](configuration.md#sbs_3d_pop_strength). The
container is odd, monotone, has unit slope at zero, and approaches the signed `0.04` source-U
representation limit without a hard endpoint clamp. It is applied independently to every depth
texel. An extreme object therefore cannot shrink the relief of unrelated geometry, and the
container cannot make a local depth cliff steeper. It has no history and does not modify the scene
camera or configured pop.

The 12-word state layout retains `container_scale` for ABI compatibility. Its only valid value is
exactly `1.0`; all attenuation belongs to the pointwise map above. Dumps, replay traces, and the
live state validator fail closed on any other value.

## Foreground ownership

DAV2 runs at a lower spatial resolution than its analysis source. A depth boundary can therefore
land in a mixed model texel whose center belongs to the far side even when most of the source
footprint belongs to the foreground. Before slope conditioning, the ownership pass checks the
exact matched full-resolution analysis source along the candidate cliff normal: the full captured
frame for ordinary V2, or the same-format cropped video texture for an authorized ROI.

It changes a cell only when all of the following evidence agrees:

- the candidate contains one strong, stable near/far cliff;
- neighboring model cells establish an existing near plateau;
- the full-resolution source contains one unique, monotone contour at the same boundary; and
- the correction can only pull the mixed far-side cell toward that existing near plateau.

The pass never lowers parallax, creates a new edge, paints color, or guesses between competing
contours. Ambiguous evidence is an exact no-op. This precision-first policy intentionally abstains
on many transparent, reflective, very thin, or highly textured boundaries.

## Cliff conditioning

A large depth cliff cannot be inverted safely as a single-valued backward warp without either
compressing foreground disparity, deforming visible background, or synthesizing missing pixels.
Sunshine 3D does not synthesize hidden background. It uses a bounded compromise:

1. Compute exact vertical upper and lower Lipschitz envelopes with a maximum shear of
   `2 / depth content width` per adjacent row.
2. Combine them as `0.75 * upper + 0.25 * lower`.
3. Compute the least horizontal majorant with a maximum slope of
   `0.5 / depth content width` per adjacent column.

The horizontal majorant guarantees a unique contractive inverse and preserves lateral foreground
continuity. The vertical share limits crown bending without fully flattening the foreground. The
trade-off remains visible on severe hair, glass-rim, and small near-object crowns: a large raw cliff
becomes a wider safe ramp, which can bend real source samples differently in the two eyes.

A previously tested post-warp collar blur was removed because it introduced a translucent halo at
a hand boundary. More inverse iterations do not solve the crown trade-off; they only solve the
already selected field more precisely. Any replacement must be qualified on hair, shoulders,
transparent rims, thin structures, and ordinary non-edge content rather than one witness frame.

## Inverse warp and source sampling

Each eye solves the same final signed parallax field with opposite signs using 11 fixed-point
iterations. The horizontal slope bound keeps the mapping contractive and gives one unique source
coordinate. The renderer then takes one linear-filtered sample from the original source texture.

There is no forward owner, multi-root visibility choice, inpaint pass, or synthetic internal fill.
Only samples outside the finite source rectangle clamp to its nearest edge; the diagnostic mask
reports that source-boundary condition rather than internal disocclusion.

An ROI final-parallax texture remains local to the selected window-region analysis rectangle. Its
integer tensor-content rectangle maps back to the complete uncropped source ROI. Before
the full-source inverse, the renderer converts that field to source-U units by multiplying it by
`ROI_width / source_width`; this preserves the intended pixel displacement rather than shrinking or
amplifying it with the crop. The ROI interior is unchanged. Outside the rectangle, the renderer
continues the signed boundary value only through the minimum collar permitted by the horizontal and
vertical slope limits, then reaches exact zero parallax. The surrounding desktop beyond that collar
therefore stays on the screen plane, while the inverse remains continuous and contractive. Color is
always sampled from the original full captured frame; the crop is never stretched back over it.

## Frame attribution and failure behavior

Color, raw depth, scene state, and parallax are bound to an exact completed source-frame identity.
An unusable current field renders the current color flat. It may retain the small scene camera so a
later usable frame resumes the same coordinate, but it never pairs old per-pixel geometry with new
color. A confirmed cut invalidates the old camera; the next usable field acquires the new one.

Model preparation, shader compilation, and the live renderer are fail-closed. Live shaders are
compiled and cached at process startup. Dump-only resources are created lazily and cannot prevent a
stream from starting. A failure in optional diagnostics has no rendering authority. An armed
schema-31 dump preserves the selected path's same-frame authority resources with ordered D3D11
`CopyResource` operations and one terminal event. Submission performs no GPU-to-CPU wait or
synchronous Map. Later render-thread calls poll with `DONOTFLUSH`, collect staging resources with
`DO_NOT_WAIT`, then hand the CPU snapshot to the existing publication worker.

ROI observer, rectangle, planner, or crop-resource eligibility failure selects ordinary
full-frame V2. That route selection does not weaken the base contract: an internal V2 model,
provenance, state, field, or renderer authentication failure renders the affected frame flat.

TensorRT inference and all coordinate passes remain on the GPU. The live path does not add a
per-frame GPU-to-CPU readback. When inference is still busy, the capture loop must not enqueue an
unbounded backlog; it continues with flat/current output according to the matched-frame contract.
Telemetry readback is nonblocking and may drop samples under GPU load, while offline evaluation
may intentionally block to obtain a complete trace.

## Foreground window-region ROI

Windows Host SBS selects at most one optional analysis region for a matched Desktop Duplication
frame. A causally authenticated Chromium `<video>` has first priority. When no Chromium video route
is eligible, the root client rectangle of `GetForegroundWindow()` may authorize the same ROI-local
V2 path. This is geometric routing only: the generic route has no executable allowlist, media/game
classifier, playback test, or occlusion reconstruction.

The foreground observer resolves the top-level root, uses the client area rather than the title bar
or DWM frame as the analysis rectangle, and validates its physical screen geometry under per-monitor
DPI awareness. Screen rectangles are compared against raw
`DXGI_OUTPUT_DESC::DesktopCoordinates`, including negative virtual-desktop origins, rather than a
pointer-normalized offset. Distinct `HMONITOR` values from a duplicated output are equivalent only
when `GetMonitorInfo` reports an `rcMonitor` exactly equal to those selected-output coordinates.
No rectangle is published for a null foreground, the shell or desktop, Sunshine's own window, an
invalid/hidden/minimized/cloaked root, an excluded tool/no-activate style, or invalid geometry. A
layered root is admitted only when `GetLayeredWindowAttributes` positively proves uniform alpha
255 with no color key; unknown, partially transparent, color-keyed, and per-pixel-alpha layered
windows select ordinary full-frame V2. A
client must be wholly contained by the current identity-oriented capture output. A disjoint client
on another monitor and a client that partially intersects or spans outputs both select full-frame
V2; Host SBS never switches the capture monitor or crops the intersection. A client that exactly
covers the capture canonicalizes to the ordinary full-source domain.

### Chromium semantic source

Windows Host SBS can bind a Chromium `<video>` observation to one matched color/depth frame. A
supervised helper obtains semantic element and foreground-client rectangles through Chromium's
IAccessible2 tree; Sunshine never performs accessibility traversal or waits for the helper on the
capture, inference, encode, or render thread. The host accepts only a fresh, identity-bearing result
from the foreground Chrome or Edge root, maps its half-open authority rectangle to an
identity-oriented single-output capture, and tolerates at most one physical pixel of browser
endpoint rounding before clipping.

Windowed ROI and true fullscreen use deliberately separate selection rules. A windowed ROI still
requires the unique largest credible IA2-available `<video>` to be fully contained by both its
owning document and the foreground browser client; equal-largest candidates remain ambiguous. For
true fullscreen, an available semantic `<video>` need only cover the complete foreground client.
The element may overscan that client, the available owning document's rectangle may be clipped, and
multiple full-cover Chromium clones do not make fullscreen ambiguous; the helper retains a stable
semantic identity and publishes the client rectangle. Only the foreground browser root is scanned,
so background browser windows and tabs cannot authorize either route. Playback state is not
required, and a paused retained identity remains eligible.

The machine protocol preserves that provenance: strict windowed selection publishes `ok`, while
the relaxed full-client selection publishes `ok-fullscreen`. The host never accepts an
`ok-fullscreen` subrectangle as a windowed ROI; it is eligible only after mapping exactly to the
capture. A transition between the two positive status classes starts a new live-detector source
generation even when the semantic IDs and rectangle are otherwise identical.

One complete census can only stage a new machine-mode selection as provisional; it remains
unpublished and Host SBS continues on full-frame V2. On the next 100 ms helper tick, a windowed
selection's retained document and video objects must independently pass the same identity, state,
tag, containment, size, and exact-rectangle checks before the helper emits `ok`. A fullscreen
refresh instead requires the same available retained video identity to keep covering the unchanged
foreground client; document-rectangle clipping and harmless element overscan changes are not
fullscreen revocations. A failed refresh emits `changed` and requests an immediate new census. A
complete no-video or windowed ambiguity revokes an existing selection immediately. Foreground
changes are hard vetoes, while unrelated descendant-object churn only requests a coalesced
three-second audit. An uncached incomplete traversal retries within one second; confirmed
accessibility unavailability retains the 15-second backoff.

Both authority sources are currently Desktop-Duplication-only. WGC does not expose an equivalent of
`LastPresentTime` that separates desktop content from cursor-only compositor frames, so a missing
content timestamp selects ordinary full-frame V2 instead of claiming a false match.

Immediately before attribution, the host also rechecks that the helper HWND still exists, remains
the foreground root window, and still belongs to the reported process. This cheap Win32 guard
closes the helper-heartbeat gap on Alt-Tab, close, and HWND reuse without putting COM on the stream.

After exact frame attribution, a same-format D3D11 copy preserves the selected rectangle exactly for
both DAV2 preprocessing and the full-resolution ownership pass; all cut, center, and history state
is ROI-local. Arbitrary source aspect ratios use the centered integer contain-fit described above,
with edge-replicated padding excluded from analysis. The original full color texture remains the
renderer source. The final ROI field is
mapped back at its physical pixel scale, with the outside-only slope collar described above, and the
desktop beyond that collar is exactly at zero parallax. Separately, the host grants the helper's
true-fullscreen authority only when its foreground-client rectangle maps exactly to the capture;
that case is canonical full-frame V2. The generic foreground-client route follows the same
canonicalization whenever its client rectangle equals the capture, without claiming Chromium
semantic authority. Missing, stale or ambiguous selection, non-identity rotation, spanning,
partially off-monitor, mismatched geometry, unauthenticated tensor geometry, or crop allocation failure
selects ordinary full-frame V2 rather than a guessed ROI; an internal V2 authentication failure
remains fail-closed flat.

The host separately tracks the newest helper heartbeat and the beginning of the current exact
`{HWND, process, document, video, rectangle}` run. The heartbeat must be fresh when the private
matched color slot is created, while the uninterrupted geometry run must have begun no later than
the source frame's content timestamp. Identical later heartbeats therefore preserve a video border
through pause, but a changed identity or rectangle cannot be attached retroactively to older
pixels. The run-begin timestamp is part of both the live authority epoch and the matched slot, so
an A-to-B-to-A helper sequence cannot revive authority captured under the first A. Desktop
Duplication derives the content time from `LastPresentTime` only; a later
cursor-only update cannot make old desktop pixels appear new. This proves causal ordering and
bounded liveness, not simultaneous compositor geometry: WinEvent delivery can still lag the pixels
by a short interval. Rendering must never combine the latest browser rectangle with an older
completed depth frame. If the helper becomes unavailable, the ordinary full-frame V2 path
continues unchanged.

The foreground-client observer applies the same causal rule without a helper heartbeat. It records
the start of each uninterrupted exact `{HWND, process, client rectangle}` run. A newly focused,
moved, or resized client cannot authorize content presented before that run began; the Desktop
Duplication content timestamp must be at least as new as the run. The root and geometry must still
be current when bound. This one-observation continuity gate prevents an Alt-Tab or window move from
attaching new geometry to old pixels while avoiding a blocking drain or second capture. The current
root/client is also sampled while inference is busy. Every ROI matched slot carries that exact live
observation generation: focus, move, resize, desktop focus, and monitor changes revoke older pending
completions and cached ROI output before warp or dump. A pure translation may reuse the DAV2 analysis
history after a newly copied frame is authorized, but never the pre-move color/depth pair.

During the native USER32 interactive move/size loop, Host SBS withdraws window-region authority and
continues ordinary full-source 3D for the current captured display. DAV2/OCR, matched completion,
telemetry, and Dump 3D therefore retain their normal full-frame behavior instead of chasing a moving
client rectangle. When the loop ends, ordinary foreground causality must authorize a newly copied
frame before the matched-domain transition can submit a new ROI; output may briefly be identity
while that exact ROI completion is pending. This covers standard caption dragging, border resizing,
and borderless applications that delegate native hit-testing to USER32. For fully custom
applications that animate themselves with `SetWindowPos`, a copied desktop image that still
predates a newly observed same-size position is submitted through full-source V2 rather than
waiting indefinitely for another desktop content presentation. A later causally matched image may
re-enter the ROI while preserving position-independent analysis history.

## Dump 3D and evaluation

Dump 3D records one matched current-contract frame: source/model/raw DAV2 evidence, authenticated
analysis-region placement, the V2 geometry chain and inverse map, scene/cut attribution, packed SBS,
and—when selected—the exact OCR8 record and compact SLR12 state used by conditioning. The reader
accepts only the current schema and identities. Retired SLR3--SLR9 and GST/OGR/ORS packages are not
replayed or reinterpreted.

An explicit dump submits every GPU artifact to a single ordered D3D11 staging batch and terminates
that batch with one event query. The render thread never flushes or waits for it: later conversions
poll with `DONOTFLUSH`, map only after completion with `DO_NOT_WAIT`, and then hand CPU-owned bytes
to the process publication worker. Because every copy precedes the event and all later live writes
follow it on the same immediate context, the package remains one exact matched frame. Only one GPU
batch or CPU publication may be pending per session. Once the event is ready, CPU collection is
resumable: each conversion copies at most 64 MiB, uses at most eight maps, and observes a 2 ms turn
budget (with a one-row progress exception for a texture row wider than the byte budget). Every map
is released in the same callback and publication begins only after all slices are complete. Dump 3D
is an explicit diagnostic action, so the larger chunks intentionally trade capture cadence for much
fewer collection polls and simpler observation. Logs report GPU-ready age, cumulative CPU collection
time, poll count, and wall time. While Host SBS remains active, the pending batch owns the retained-
source conversion poll so even a completely static desktop receives every later callback. If the
session ends or changes mode first, its unpublished diagnostic batch is cancelled at teardown
instead of making teardown wait for the GPU.

Use `.f32` artifacts for quantitative comparisons. Independently stretched PNG previews can hide
scale differences and are diagnostic only. The supported commands, metrics, and baseline policy live
in [the sbsbench guide](../tools/sbsbench/README.md).

## Known limitations

- DAV2 is a relative monocular model. Framing and surrounding context can change its raw depth;
  windowed and fullscreen versions of the same content are not guaranteed to be affine-equivalent.
- Single-frame warping has no observation of newly exposed background. Strong foreground cliffs
  therefore require a visible geometry compromise unless temporal or synthesized fill is added.
- Transparent, reflective, and sub-grid objects often lack a unique ownership contour. The
  precision-first ownership pass abstains instead of risking a wrong snap.
- Scene detection is inferred, not ground truth. Exposure, structureless frames, persistent motion,
  and rapid consecutive cuts require the separate guarded state machine.
- The ROI route uses at most one current-output region. Chromium accessibility exposes a semantic
  element box, not exact composited visibility; when that route is unavailable, the generic
  foreground-client route analyzes the whole client, including any captured application UI or
  overlay. It does not classify applications or reconstruct pixels hidden by other windows. Shell,
  desktop, invalid, stale, spanning, partially off-monitor, and background-window rectangles do not
  authorize an ROI.
- The automatic subtitle path accepts all six authenticated DAV2 fields (`770x434`, `1022x434`,
  `1036x434`, and their portrait transposes) and one coherent bottom stack of at most four line
  boxes. Any other fitted tensor copies ordinary V2. The
  detector does not know whether a box contains a subtitle, UI text, a lower third, or a logo;
  bottom crop, generic horizontal-line geometry, coherent stacking, exact identity, and
  two-observation overlap are the only false-positive controls. A horizontal persistent scene-text
  box can therefore acquire, while a square badge is normally rejected by geometry. More than one
  simultaneous subtitle stack is unsupported and only the deterministic winning stack is tracked.
- Detection, not recognition, is pinned. Very small, very low-contrast, stylized, vertical, curved,
  or partially clipped text can be missed or boxed inaccurately. The two-observation transaction
  adds one completed-observation latency to births and material handoffs. Typewriter and karaoke
  additions condition only already-matched lines until the updated stack confirms. An authoritative
  one-observation detector miss removes current geometry immediately; grace retains only a target
  seed, never a mask.
- Current subtitle qualification comes from deterministic synthetic clip fixtures and WARP
  sequences; it does not measure the real-video distribution or a production false-positive rate.
  HDR, fades, stylized subtitles, false positives, cross-GPU timing, and Galaxy XR visual acceptance
  still require real-video qualification. Host SBS does not reconstruct unoccluded video behind the
  subtitle.
- Unsupported model or fitted-tensor setup is rejected before Host SBS starts. A bad per-frame
  shader/state/field identity renders that current frame flat rather than attempting a best-effort
  geometry fallback.
