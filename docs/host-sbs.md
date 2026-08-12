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
    LOCATOR["OCR6 boxes and compact SLR6 tracker"]
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
    CAPTURE -. completed 770x434 observation .-> OCR --> LOCATOR
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
schema 40/tag `0xF7C853C2`, canonical SHA-256
`5c7116be0004e33e24f150430d85e06b9eb66782b4fc3b059501e04093835d9e`, with authenticated
producer source-closure SHA-256
`861f800db1cc06a6d25b80d18bb1b7bf4bc469ed1ccfacfd96a0b384bfb2a7a1`, live-renderer
source-closure SHA-256
`914fc624955c5e3b41f867429acb17a03c34ba71e41f7cbfab00fd939aafff9b`, and dump-only diagnostic
renderer source-closure SHA-256
`d2dada0490e727d88fdda802a9eb6759b34e9501c91835363ebdb0196e3d5c4b`. It admits the following
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

The production subtitle path is the single current detector-only PP-OCRv6 tiny/OCR6/SLR6 route.
It does not run text recognition, language classification, or logo recognition. For a completed
canonical `770x434` DAV2 observation, the GPU takes the exact bottom `6:1` source crop, resizes it to
`960x160`, converts BGR through the pinned ImageNet normalization, and runs the authenticated
`ppocrv6_tiny_det` TensorRT 11 strong-typed FP32 engine
(`trt-strong-fp32-tf32-fixed960x160-level5-v1`). The detector's `1x1x160x960` probability map is reduced
on the GPU to bounded line boxes; no probability texture or model input becomes rendering
authority.

OCR6 is a fixed 208-word current record: a 16-word header, at most 16 raw boxes, and at most eight
final boxes. Every half-open box lies in the `770x434` field's `[325,430)` bottom ROI. Schema/tag,
authoritative flag, exact matched-frame and analysis-generation identity, source dimensions, field,
ROI, finite score, counts, and all reserved/unused words must validate together. `flags == 1` with
zero boxes is an authoritative empty observation. Abstaining, incomplete, overflowing, stale, or
malformed output has no geometry authority.

SLR6 is a fixed 80-word compact state containing at most four owner rectangles, four pending
rectangles, and four same-frame current-authority rectangles. Generic line geometry rejects boxes
that are too small, too wide, too short, or not sufficiently horizontal; there is no hard-coded
logo position. Vertically adjacent boxes form one coherent centered, left-aligned, or right-aligned
stack, chosen deterministically by area. Individual line rectangles and their gaps are preserved.
The first exact observation is pending; only a compatible observation with a distinct exact
frame/domain identity confirms an owner. A compatible subset can remove a line immediately, while
an appended or materially changed stack remains pending and conditions only lines still matched to
the old owner until its second observation confirms the handoff.

Only rectangles copied from the current OCR6 record can condition the current frame. Cached owner,
pending, target, and six-observation death grace cannot manufacture geometry. Authoritative empty
OCR removes all current rectangles immediately. A hard cut clears pending/grace, retains only
current rectangles that still overlap an old owner, and resamples the plane target; disjoint boxes
begin a new two-observation transaction. An input-domain reset clears the owner as well and treats
any current boxes as the first pending observation. This lets a seek/reset landing on an
already-visible static subtitle acquire on the following distinct observation without an onset
edge.

The plane target is sampled from the same post-limit BaseField as the subtitle, above the owner
stack: the median of 32 samples is filtered with a `0.15` EMA and a maximum per-observation step of
`0.25 / source_width`. New owners fade from half to full strength over two observations; a
cut-surviving owner resamples at full strength. The conditioner then evaluates distance directly to
each current half-open rectangle. For integer cell `(x,y)`, `dx`/`dy` are zero inside a rectangle
and count cells to its nearest included edge outside it:

```text
d      = min(dx * 0.5 / field_width + dy * 2.0 / field_width)
budget = 0.5 / source_width + d
```

Base values already within `budget` of the target are copied bit-for-bit. Values outside it move
only to `target +/- budget`, with the half-strength birth/handoff fade when applicable. Thus each
line has a dense core and a V2-slope-safe analytic collar, nearby line collars may meet naturally,
and the gap is never converted into one merged rectangle. Missing current authority, invalid target
state, unsupported tensor shape, or any identity failure copies ordinary post-limit V2 exactly.
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
schema-21 dump preserves the selected path's same-frame authority resources with ordered D3D11
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
and—when selected—the exact OCR6 record and compact SLR6 state used by conditioning. The reader
accepts only the current schema and identities. Retired SLR3--SLR5 and GST/OGR/ORS packages are not
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
- The automatic subtitle path is qualified only for an exact `770x434` landscape model tensor and
  one coherent bottom stack of at most four line boxes. Other tensor shapes copy ordinary V2. The
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
