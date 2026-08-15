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
OCR8 bounded boxes and compact SLR12 owner/pending/current-authority rectangles at the authenticated
DAV2 field shape. It does not retain the retired row-history or general overlay-detector pipelines.
Unsupported identities and shapes preserve ordinary V2 exactly. Dump and replay accept only the
current SLR12/OCR8 schema. The host does not reconstruct hidden video.

Damage reuse is deliberately DDup-only and fail-open. Dirty and move metadata is semantic only as a
proof that the exact current ROI pixels did not change: damage outside the crop may save inference,
whereas an intersecting move source or destination, a sequence gap, metadata failure, protected
content, route movement, authority change, or WGC forces the ordinary path. Reuse freezes the
depth/cut/camera/OCR/SLR observation state and always warps the current capture color rather than
repeating a packed SBS frame.

The next quality-surface probe is implemented only as a default-off process A/B lever,
`APOLLO_SBS_LOW_MOTION_GATE=1`; it is not a persisted SBS setting. Starting from an authenticated
completed cache with no inference pending, it permits one hold within 50 ms when the conservatively
summed DDup overlap is no more than 0.25% of the exact DAV2 region and the bottom OCR crop has no
desktop damage. The baseline stays at the last real enqueue so drift accumulates instead of rolling
forward. Diagnostics report candidate, skip, and successful reuse rates with the lever off or on.
This is intentionally non-bit-exact and cursor-insensitive and must remain opt-in until clip-level
quality evaluation and Nsight GPU-load evidence justify a production policy.

OCR also has a narrower independent DDup optimization on frames where DAV2 does run. If damage is
complete and wholly outside the exact bottom `6:1` detector crop, the host retains deterministic
OCR8 boxes, restamps them to the new exact frame, and runs current SLR12 normally. This skips OCR
preprocess, interop, TensorRT, cells, and resolve without freezing depth, cut, camera, V2, or locator
state. Subtitle-band damage fails open immediately, so onset/removal has no extra readback frame of
latency. Hardware-cursor composition is deliberately ignored by this desktop-content proof; WGC and
Dump 3D always take ordinary OCR.

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
