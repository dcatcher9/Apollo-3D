# SBS 3D status and roadmap

Sunshine 3D ships one host-side geometry pipeline: Depth Coordinate V2. This document describes
only the current product, its known limitations, and work that may still be justified.

## Current product boundary

| Path | Geometry owner | Status |
|---|---|---|
| Host SBS live stream | Sunshine 3D on Windows/NVIDIA | V2 production pipeline, including a causally attributed foreground window-region ROI route with Chromium-video priority |
| Offline Host 3D conversion | Sunshine 3D isolated worker | Same V2 geometry with bounded cut lookahead and compressed H.265/AV1 output |
| Client SBS | Moonlight 3D on Android XR | Separate client pipeline; unchanged by Host V2 |
| Local AR glasses | Sunshine 3D local presenter | Reuses Host V2 without network encode/decode |

Host V2 uses authenticated DAV2 Small inference, one scene-latched raw center, literal configured
pop, a frame-local parallax container, conservative foreground ownership, bounded vertical and
horizontal cliff conditioning, and a unique contractive inverse. Invalid or unauthenticated
geometry renders flat; there is no older fallback.

On Desktop Duplication, a fresh exact matched-frame rectangle may replace full-frame analysis with
one same-format crop. The unique largest fully contained `<video>` in the foreground Chrome or Edge
document has priority; otherwise the foreground top-level root client may authorize the crop. The
rectangle is copied exactly and fitted without cropping or stretching into a centered integer
content rectangle in the current authenticated tensor. Edge-replicated synthetic padding is
excluded from analysis. A selected client that exactly covers the capture
canonicalizes to ordinary full-frame V2 with no crop or domain reset. Null/shell/desktop/self,
hidden/minimized/cloaked/excluded-style windows, missing content timestamps, invalid or stale
geometry, non-identity rotation, unauthenticated tensor geometry, and other-monitor/partial/spanning
rectangles all retain ordinary full-frame analysis. The host never changes the capture monitor for
this route; an internal base-V2 authentication failure still renders flat.

The live DDup path may also reuse authenticated geometry for an already-authorized, exactly
positioned ROI when a complete contiguous dirty/move sequence proves that every intervening desktop
change was outside that crop. This bounded optimization changes neither ROI selection nor analysis
semantics: uncertainty returns to normal inference, and 16 skipped deliveries or 250 ms forces a
refresh. WGC has no damage-reuse authority.

The complete implementation contract is in [Host SBS pipeline](host-sbs.md). Scene-cut behavior is
owned by [Host SBS scene cuts](host-sbs-scene-cuts.md).

## What is intentionally absent

- No endpoint normalization, min/max range EMA, subject stretch/recenter, adaptive pop, or
  configurable zero-plane translation.
- No damage-driven ROI selection, image-tracked ROI, background-window/tab ROI, or second-inference
  ROI. The foreground window-region route replaces that frame's full analysis with at most one
  causally attributed crop. DDup damage can only retain an independently authorized exact route;
  it cannot create or alter one.
- No application/media classifier, allowlist, Z-order compositor, occlusion reconstruction, or
  automatic capture-monitor switch in the foreground-client route.
- No forward-owner render, multi-root visibility selector, post-warp blur, or synthetic hidden-pixel
  fill.
- No CPU depth fallback or best-effort use of an unauthenticated model/shape.
- No age-based replacement of a valid scene camera between confirmed cuts.

These are architectural boundaries, not dormant feature flags.

## Known limitations

### Monocular framing dependence

DAV2 is a relative monocular model. Fullscreen, windowed, or differently surrounded versions of
the same image can produce materially different raw depth. More tensor pixels on the main subject
do not guarantee a better boundary. Synthetic context must not influence monocular analysis. The
window-region route therefore uses only the exact client/video pixels: its edge-replicated tensor
padding is excluded from depth statistics, scene cuts, ownership, history, and OCR authority.

### Foreground window-region ROI boundary

The ROI is intentionally narrow in geometry and timing: Desktop Duplication, one current foreground
root, full containment on the currently captured identity-oriented output, and an exact causally
matched frame. A valid Chromium semantic `<video>` wins; otherwise the whole foreground client is
eligible without application classification. Paused Chromium video is supported because selection
follows DOM identity rather than playback activity. Desktop or shell focus, hidden/minimized/cloaked
or excluded windows, stale geometry, WGC, unauthenticated tensor geometry, and partially off-monitor,
spanning, or other-monitor rectangles use full-frame V2. The route does not crop an intersection or
switch outputs.

Independently, a semantic Chromium video covering the complete foreground client can authorize true
fullscreen even with overscan, document clipping, or duplicate full-cover clones. The host accepts
it only when that client maps exactly to the capture. The generic foreground route also
canonicalizes an exactly full-capture client to full-source V2. Chromium-video and foreground-client
ROIs are distinct analysis authorities, so switching between them resets temporal lineage even if
the rectangles match.

The helper reports strict windowed evidence as `ok` and relaxed full-client evidence as
`ok-fullscreen`. The latter can never authorize a mapped subrectangle as an ROI and canonicalizes to
the full-source domain only after exact capture mapping.

The route has no compositor-visible-region oracle. The current subtitle treatment therefore uses
OCR8 bounded boxes and compact SLR13 owner/pending/current-authority rectangles at the authenticated
DAV2 field shape. It does not retain the retired row-history or general overlay-detector pipelines.
Unsupported identities and shapes preserve ordinary V2 exactly. Dump and replay accept only the
current SLR13/OCR8 schema. The host does not reconstruct hidden video.

Damage reuse is deliberately DDup-only and fail-open. Dirty and move metadata is semantic only as a
proof that the exact current ROI pixels did not change: damage outside the crop may save inference,
whereas an intersecting move source or destination, a sequence gap, metadata failure, protected
content, route movement, authority change, or WGC forces the wrapper's inference path. Reuse freezes the
depth/cut/camera/OCR/SLR observation state and always warps the current capture color rather than
repeating a packed SBS frame. This exact-current DDup proof is distinct from the presentation-only
fallback below, which is eligible only when no matched completion reaches the render deadline.

GPU adaptive reuse is always part of the Host SBS depth pipeline. Exact cache reuse remains higher
priority; otherwise host-only route, cadence and complete DDup history classify a changed frame as
either force-infer or GPU-undecided. No host motion-size/shape heuristic stands between an
authority-valid frame and the dense atomics-free model-space detector, which compares every admitted
current NCHW texel with the authenticated state-1 history and applies finite, global medium/strong,
and supported-tile local bounds. The GPU selects the DAV2 infer/reuse branch without staging
readback, a decision poll, or CPU branch knowledge. Subtitle work is a separate authenticated
request domain inside the same joined completion. Ordinary current-ready OCR work `1`, or ordinary
ineligible abstention work `2`, publishes only on infer. Cadence-due current-ready OCR work `8`
publishes on infer or reuse; cadence-due ineligible work `16` publishes an exact-current abstention
on either branch with the optional handle off. Native USER32 suppression advances neither OCR nor
locator state.

There is no adaptive OCR-band, localized subtitle-transition, or SLR-state veto. The published
expected work and transaction cookie authenticate the exact subtitle disposition. Ordinary reuse
holds the prior OCR8/SLR13/final tuple. Due reuse retains depth, cut, camera, and immutable V2 Base
but advances the current OCR or abstention, SLR13, and conditioned final field. No mode relabels
retained boxes as a new OCR observation.

Up to four GPU-undecided deliveries may render private current color through GPU-selected
authenticated V2 geometry. A reuse freezes depth/cut/camera/V2 Base and the complete
OCR8/SLR13/conditioned-final tuple for ordinary work, while due subtitle work may advance that
tuple independently. Reprojection consumes the complete atomic final field directly, with no
schedule-coupled display recurrence. The CPU keeps the depth branch opaque. An ordinary opaque
result cannot label subtitle evidence exact-current, whereas a due result is exact-current on both
branches by request contract. The observation barrier continues blocking depth cache, telemetry,
and host-owned approximate holds.
A completed opaque draw may retain only its already-rendered packed SBS pixels for bounded
presentation continuity. If the next root misses the same-frame completion budget, that image can
be redelivered on the unchanged route/domain instead of flashing flat; it carries no DAV2, OCR,
SLR, damage, cache-lineage, or follow-up-submission authority.
A metadata-only anchor may nevertheless send the immediately following complete-history candidate
back to the GPU when it is strictly under `100 ms`, on the identical route, and
bound to the prior opaque frame and damage history. The device compares cumulatively against the
last actual infer input; owner age through four frame steps is allowed only while source observation
age is also strictly under `100 ms`. The fifth step or age boundary forces infer. Initial candidate
and opaque-follow-up admission share that strict `100 ms` host-age ceiling. Expiry, route/proof
failure, or interop-signature drift still
forces CPU-known inference and restores ordinary depth lineage.
Diagnostics count initial and follow-up roots plus host expiry/rejection/fallback, never the private
branch outcome.

Coherent GPU temporal state is not synthetically reset. Every post-bootstrap DAV2 submission,
including force-infer, uses the same conditional wrapper; wrapper/capability/interop failure is
terminal flat rather than a hidden raw-TensorRT fallback. The canonical thresholds, ownership,
four-delivery/`100 ms` depth risk and reset matrix are in
[Host SBS frame attribution](host-sbs.md#frame-attribution-and-failure-behavior).

OCR instead has a narrower independent cadence. Every accepted ordinary opaque root is
conservatively a dirty hold because the host cannot know whether its infer-coupled OCR ran. After two
such holds, or `33 ms` of source observation time since the last guaranteed subtitle observation,
the next accepted root is due. Current-ready due work uses `8`; an ineligible OCR input/interop/child
uses `16` and publishes abstention. There is no OCR-only DDup retained-box proof because desktop
damage does not cover separately composed hardware-cursor changes. Dump 3D always takes ordinary
subtitle work on a force-infer root.

### Adaptive follow-up status

The presentation-only packed-SBS fallback is deliberately narrower than semantic cache reuse:
already-rendered packed pixels may provide bounded continuity after a missed render deadline without
becoming DAV2, OCR, SLR, damage, telemetry, or adaptive-submission authority. Same-frame polling is
therefore a freshness/latency optimization rather than a geometry-correctness boundary.

Completed in schema 67: the schedule-coupled three-phase display recurrence was removed and the
complete atomic conditioned final field is rendered directly. The recurrence was introduced for the
earlier infer/reuse clocking hypothesis; removing it eliminates a persistent full-field resource and
pass, effective field lag, and rendering semantics coupled to reuse-owner age. Keep the GPU history
owner, four-frame/`100 ms` bound, diagnostic trace, mature
single-line provisional bridge, and compatible-handoff fade preservation until separate evidence
justifies changing them.

Phase B is implemented as independently authenticated depth and subtitle choices inside the existing
joined CUDA graph. Ordinary OCR is infer-coupled; due current-ready OCR (`8`) and due ineligible
abstention (`16`) publish on either depth branch. The host cadence permits two conservative dirty
holds or `33 ms`, while the depth history owner permits four frame steps with strict `<100 ms`
observation age. Live and offline replay share the request/chaining policy and one joined completion
proof. Do not restore the retired host motion-size, OCR-band, localized-tile, or SLR-state inference
vetoes.

For latency, keep DAV2 and OCR as siblings of the same CUDA root; do not add another stream or graph.
First add diagnostic-only timestamps for both branches. If profiling shows OCR slack, the first
candidate is its buffer-only probability-cell reduction and OCR8 candidate resolve moved into an OCR
child tail after TensorRT. Keep shared RGB-to-NCHW and similarity work before the root, and keep
SLR/conditioning/warp after the join. Port OCR crop preprocessing only if its measured benefit
exceeds the source-texture interop cost. This is expected to offer only a modest critical-path saving,
not a utilization claim, and must pass a weak-GPU contention gate. Remove each superseded HLSL
implementation after corpus parity rather than maintaining duplicate live/offline algorithms.

Any expansion beyond four consecutive reuses or the strict `<100 ms` observation-age limit requires
new shared-replay, exact-trace, scene-cut, subtitle-transition, weak-GPU-tail, and live-XR evidence.
The acceptance metric is final SBS readiness and presentation continuity, not the raw CUDA-root
event time or a promised reuse percentage.

### Foreground crowns and disocclusion

A single source image contains no observation of background hidden behind a shifted foreground.
V2 preserves foreground continuity while limiting the slope required by its inverse warp. On a
large raw cliff this converts one discontinuity into a wider safe ramp. Hair crowns, transparent
glass rims, and small near-object tops can therefore bend or stretch differently in the two eyes.

The full-resolution ownership pass corrects only a unique, corroborated mixed boundary cell.
Transparent, reflective, thin, or competing contours deliberately abstain. Loosening that policy
globally would exchange a visible residual for incorrect foreground ownership elsewhere.

### Scene-cut uncertainty

Scene cuts are inferred from depth, raw appearance, ordinal structure, and history. Exposure and
persistent-motion guards reduce false relatches but cannot create ground truth for an ambiguous
sequence. Rapid consecutive cuts and structureless transitions remain important headset tests.

### Unsupported identities fail flat

Live Host SBS, production conversion, and the maintained evaluator accept only the authenticated
DAV2 Small model, preprocessing closure, supported
[resolution fit](host-sbs.md#authenticated-resolution-fitting), producer/state contract, and
renderer closure. There is no Host SBS model fallback or selector.

### Live telemetry is sampled

Production readback is nonblocking. Samples may be missing under GPU load, while offline traces are
complete. A live/offline discrepancy must first be aligned by exact source-frame identity before it
is interpreted as a controller difference.

## Qualification priorities

Before changing V2 geometry:

1. Reproduce the issue with a current Dump 3D package and identify whether it begins in raw DAV2,
   ownership, cliff conditioning, or the inverse warp.
2. Run an exact same-input offline A/B. Do not compare independently captured frames as if they
   were pixel-identical.
3. Run the canonical core and extended evaluator suites and inspect both hard gates and diagnostic
   stereo/artifact movement.
4. Check hair, shoulders, transparent rims, thin structures, flat pages, HDR, scene cuts, invalid
   depth, and all authenticated aspect families. For the window-region route, also check Chromium
   priority, pause, foreground translation and resize, authority changes, desktop/minimized focus,
   other-monitor and spanning fallbacks, exact full-capture canonicalization, multiple videos,
   portrait/square/ultrawide contain-fit padding, and both signs at every ROI edge. For DDup damage
   reuse, separately check dirty and move rectangles inside, outside, and touching each half-open ROI
   edge; repeated and missing sequences; metadata/protected-content failure; WGC fallback; and both
   the 16-delivery and 250-ms forced-refresh boundaries.
5. Confirm the result in Galaxy XR at the intended pop strength before changing the production
   contract or baselines.

The current high-value investigation is a general, precision-first reduction of ambiguous
small-object crown distortion. A candidate must improve glass-rim and small-near-object witnesses
without reintroducing the hair/shoulder discontinuity or the hand-boundary halo. If it cannot name
that safe evidence boundary, retaining the current compromise is preferable.

## References

- [Host SBS pipeline](host-sbs.md)
- [Host SBS scene cuts](host-sbs-scene-cuts.md)
- [Offline Host 3D conversion](whole-clip-sbs-pipeline.md)
- [Local AR glasses](sbs-local-ar-glasses.md)
- [sbsbench evaluator](../tools/sbsbench/README.md)
