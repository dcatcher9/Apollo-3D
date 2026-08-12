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
    IA2["Foreground Chromium IA2 observation"]
    MATCH["Exact matched-frame ROI selection"]
    DOMAIN["Full frame or authenticated inward-trimmed ROI"]
    PREPROCESS["HDR/SDR model preprocessing"]
    DAV2["Authenticated DAV2 Small inference"]
    CUT["Cut-only evidence and scene epoch"]
    OCR["PP-OCRv6 tiny detector (bottom 960x160)"]
    LOCATOR["OCR8 boxes and compact SLR9 tracker"]
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
    IA2 -. optional authority .-> MATCH
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
schema 46/tag `0xD18FF0F3`, canonical SHA-256
`8ab387f9bcda29e90455ce9e5b8677cef3cd7744fe03ee03202a4699fa7e4ead`. It binds the
complete policy below, including all subtitle field/ROI semantics, and producer source-closure
SHA-256 `f299ce49f332458a1d97634d4ced6d7fc802d2be8d583ee04f0e32d851ed1a22`.
The live-renderer source-closure SHA-256 is
`5a1ab7175b97b8ca89397da8b3f86812a3faedf5e15938abf42f64e9716f2c5b`, and dump-only diagnostic
renderer source-closure SHA-256
`12dbf80c10110b21d59d01b9120c4efa0b9147bed420f134053cfef1276d00d3`. It admits the following
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

The foreground-Chromium video route does not introduce another model shape. It keeps the current
full-frame authenticated tensor shape and, only when necessary, trims the detected video rectangle
inward by at most 2% of its area to match that shape's aspect ratio. It never expands the rectangle
into browser chrome, pads the model input, or stretches video pixels. If that small inward fit cannot
be proven, the frame uses ordinary full-frame V2.

Authenticated true fullscreen is selected separately from the strict windowed-ROI candidate. The
helper may recognize an available semantic `<video>` that covers the complete foreground browser
client even when the element overscans that client, its owning document rectangle is clipped, or
Chromium exposes multiple full-cover clones. The helper publishes the client rectangle as the
fullscreen authority, and the host accepts that authority only when the client maps exactly to the
capture extent. A maximized browser, an estimator full-source domain, or a failed windowed ROI does
not prove fullscreen by itself. Once admitted, true fullscreen is canonical ordinary full-frame V2:
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

For an authorized window-video ROI, extrema, mean, standard deviation, cut evidence, and all depth
histories are computed from the cropped video analysis domain only. Browser chrome and the desktop
surround do not pull the video's scene center. Entering or leaving ROI analysis, changing its
identity or dimensions, or changing its input transfer domain resets the temporal and scene-camera
state before the new domain is used. Translating the same ROI without changing its dimensions is
not a new analysis domain, so an ordinary window move does not by itself reacquire the camera.

### OCR-box subtitle conditioner

The production subtitle path is the single current detector-only PP-OCRv6 tiny/OCR8/SLR9 route.
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
keeps OCR8 and SLR9 on the same exact-frame authority path without extra readback, drain, probe, or
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

SLR9 is a fixed 80-word compact state containing at most four owner/core rectangles, four
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
same observation does not consume another grace step. A hard cut clears pending/grace, retains only
current rectangles that still overlap an old owner, and resamples the plane target; disjoint boxes
begin a new two-observation transaction. An input-domain reset clears the owner as well and treats
any current boxes as the first pending observation. This lets a seek/reset landing on an
already-visible static subtitle acquire on the following distinct observation without an onset
edge.

The plane target is the median of 32 samples from the same post-limit BaseField above the combined
lower-text owner stack. It is sampled once for a fresh confirmed birth and then latched bit-exact
for the continuing owner; ordinary Base changes, core jitter, and subset removal do not update it.
Any confirmed handoff or rebirth during the six-observation death grace inherits the cached target
bits, independent of overlap. A hard-cut
survivor advances owner generation, resamples immediately at full strength, and a domain reset
clears the owner and latch so present geometry starts a new pending transaction. New owners fade
from half to full strength over two observations. The conditioner then evaluates distance directly to
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
The conditioner validates the complete 80-word state once per `16x16` dispatch group and shares
that verdict within the group; no thread can partially accept a malformed state.
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
   `2 / depth_width` per adjacent row.
2. Combine them as `0.75 * upper + 0.25 * lower`.
3. Compute the least horizontal majorant with a maximum slope of
   `0.5 / depth_width` per adjacent column.

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

An ROI final-parallax texture remains local to the video analysis rectangle. Before the full-source
inverse, the renderer converts that field to source-U units by multiplying it by
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
schema-27 dump preserves the selected path's same-frame authority resources with ordered D3D11
`CopyResource` operations;
it adds no production GPU-to-CPU readback, flush, query wait, or synchronous Map. The explicit dump
then uses the existing diagnostic readback path to publish files.

ROI observer, rectangle, aspect-fit, or crop-resource eligibility failure selects ordinary
full-frame V2. That route selection does not weaken the base contract: an internal V2 model,
provenance, state, field, or renderer authentication failure renders the affected frame flat.

TensorRT inference and all coordinate passes remain on the GPU. The live path does not add a
per-frame GPU-to-CPU readback. When inference is still busy, the capture loop must not enqueue an
unbounded backlog; it continues with flat/current output according to the matched-frame contract.
Telemetry readback is nonblocking and may drop samples under GPU load, while offline evaluation
may intentionally block to obtain a complete trace.

## Foreground Chromium window-video ROI

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

This causal attribution is currently Desktop-Duplication-only. WGC does not expose an equivalent
of `LastPresentTime` that separates desktop content from cursor-only compositor frames, so WGC uses
ordinary full-frame V2 instead of claiming a false match.

Immediately before attribution, the host also rechecks that the helper HWND still exists, remains
the foreground root window, and still belongs to the reported process. This cheap Win32 guard
closes the helper-heartbeat gap on Alt-Tab, close, and HWND reuse without putting COM on the stream.

After exact frame attribution, the host optionally trims the rectangle inward by at most 2% of its
area to the current authenticated tensor aspect. A same-format D3D11 crop supplies both DAV2
preprocessing and the full-resolution ownership pass; all cut, center, and history state is
crop-local. The original full color texture remains the renderer source. The final ROI field is
mapped back at its physical pixel scale, with the outside-only slope collar described above, and the
desktop beyond that collar is exactly at zero parallax. Separately, the host grants the helper's
true-fullscreen authority only when its foreground-client rectangle maps exactly to the capture;
that case is canonical full-frame V2. A maximized window or full-source fallback without this
semantic/client proof is not fullscreen authority. Missing, stale, ambiguous windowed selection,
rotated, spanning, partially off-monitor, mismatched geometry, unsupported aspect fitting, or crop
allocation failure selects ordinary full-frame V2 rather than a guessed ROI; an internal V2
authentication failure remains fail-closed flat.

The host separately tracks the newest helper heartbeat and the beginning of the current exact
`{HWND, process, document, video, rectangle}` run. The heartbeat must be fresh when the private
matched color slot is created, while the uninterrupted geometry run must have begun no later than
the source frame's content timestamp. Identical later heartbeats therefore preserve a video border
through pause, but a changed identity or rectangle cannot be attached retroactively to older
pixels. Desktop Duplication derives the content time from `LastPresentTime` only; a later
cursor-only update cannot make old desktop pixels appear new. This proves causal ordering and
bounded liveness, not simultaneous compositor geometry: WinEvent delivery can still lag the pixels
by a short interval. Rendering must never combine the latest browser rectangle with an older
completed depth frame. If the helper becomes unavailable, the ordinary full-frame V2 path
continues unchanged.

## Dump 3D and evaluation

Dump 3D records one matched current-contract frame: source/model/raw DAV2 evidence, authenticated
analysis-region placement, the V2 geometry chain and inverse map, scene/cut attribution, packed SBS,
and—when selected—the exact OCR8 record and compact SLR9 state used by conditioning. The reader
accepts only the current schema and identities. Retired SLR3--SLR8 and GST/OGR/ORS packages are not
replayed or reinterpreted.

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
- The ROI route is foreground-Chromium-only. Chromium accessibility exposes a semantic element box,
  not exact composited visibility. Canvas or WebGL players, protected video, disabled accessibility,
  background tabs/windows, non-Chromium applications, and partially off-monitor video use
  full-frame V2 when they provide no valid rectangle.
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
