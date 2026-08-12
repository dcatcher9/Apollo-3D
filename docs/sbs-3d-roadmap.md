# SBS 3D status and roadmap

Sunshine 3D ships one host-side geometry pipeline: Depth Coordinate V2. This document describes
only the current product, its known limitations, and work that may still be justified.

## Current product boundary

| Path | Geometry owner | Status |
|---|---|---|
| Host SBS live stream | Sunshine 3D on Windows/NVIDIA | V2 production pipeline, including a strict foreground-Chromium window-video ROI route |
| Offline Host 3D conversion | Sunshine 3D isolated worker | Same V2 geometry with bounded cut lookahead and compressed H.265/AV1 output |
| Client SBS | Moonlight 3D on Android XR | Separate client pipeline; unchanged by Host V2 |
| Local AR glasses | Sunshine 3D local presenter | Reuses Host V2 without network encode/decode |

Host V2 uses authenticated DAV2 Small inference, one scene-latched raw center, literal configured
pop, a frame-local parallax container, conservative foreground ownership, bounded vertical and
horizontal cliff conditioning, and a unique contractive inverse. Invalid or unauthenticated
geometry renders flat; there is no older fallback.

On Desktop Duplication, a fresh exact matched-frame rectangle for the unique largest fully contained
`<video>` in the foreground Chrome or Edge document may replace full-frame analysis with one
same-format crop. The detected rectangle can be trimmed inward by at most 2% of its area to the
current authenticated tensor aspect; it is never padded or stretched. True fullscreen has separate
semantic authority: an available video may cover the complete foreground browser client despite
element overscan, document-rectangle clipping, or multiple full-cover clones, but the host still
requires that client rectangle to map exactly to the capture. That admitted case canonicalizes to
ordinary full-frame V2 with no crop or domain reset. A maximized/full-source browser without this
proof and every other ROI-eligibility failure retain unauthoritative full-frame analysis; an internal
base-V2 authentication failure still renders flat.

The complete implementation contract is in [Host SBS pipeline](host-sbs.md). Scene-cut behavior is
owned by [Host SBS scene cuts](host-sbs-scene-cuts.md).

## What is intentionally absent

- No endpoint normalization, min/max range EMA, subject stretch/recenter, adaptive pop, or
  configurable zero-plane translation.
- No damage-driven, image-tracked, generic-window, background-tab, or second-inference ROI. The
  foreground-Chromium route replaces that frame's full analysis with one causally attributed crop.
- No forward-owner render, multi-root visibility selector, post-warp blur, or synthetic hidden-pixel
  fill.
- No CPU depth fallback or best-effort use of an unauthenticated model/shape.
- No age-based replacement of a valid scene camera between confirmed cuts.

These are architectural boundaries, not dormant feature flags.

## Known limitations

### Monocular framing dependence

DAV2 is a relative monocular model. Fullscreen, windowed, or differently surrounded versions of
the same image can produce materially different raw depth. More tensor pixels on the main subject
do not guarantee a better boundary. Neutral padding and synthetic browser surrounds have not shown
enough cross-scene reliability to ship. The Chromium ROI route avoids synthetic context: it uses an
inward-only crop and falls back to the full frame when the current authenticated aspect cannot be
reached with at most 2% area trimming.

### Foreground browser ROI boundary

The windowed ROI is intentionally narrow in scope: foreground Chrome/Edge, Desktop Duplication, one
uniquely largest accessible `<video>` fully contained by its document and browser client, fully on
one monitor, and an exact causally matched frame. Paused video is supported because selection
follows DOM identity rather than playback activity. Background windows/tabs, equal-largest windowed
videos, stale or ambiguous accessibility data, WGC, unsupported aspect fits, and partially
off-monitor rectangles use full-frame V2. Independently, a semantic video covering the complete
foreground client can authorize true fullscreen even with overscan, document clipping, or duplicate
full-cover clones; the host accepts it only when that client maps exactly to the capture.

The helper reports strict windowed evidence as `ok` and relaxed full-client evidence as
`ok-fullscreen`. The latter can never authorize a mapped subrectangle as an ROI. Switching between
the two authority classes clears live-detector lineage even when the semantic IDs and extents match.

The route has no compositor-visible-region oracle. The current subtitle treatment therefore uses
OCR6 bounded boxes and compact SLR6 owner/pending/current-authority rectangles at the authenticated
DAV2 field shape. It does not retain the retired row-history or general overlay-detector pipelines.
Unsupported identities and shapes preserve ordinary V2 exactly. Dump and replay accept only the
current SLR6/OCR6 schema. The host does not reconstruct hidden video.

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
   depth, and all authenticated aspect families. For the Chromium ROI route, also check pause,
   window translation and resize, exact full-capture canonicalization, multiple videos, inward
   aspect trim, fallback, and both signs at every video edge.
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
