# Host SBS single-ROI content focus

## Goal

Improve Host SBS depth for the two common browser cases where full-frame Depth Anything V2
underuses its model grid:

1. one video playing inside a browser page; and
2. a static collage or grid of video thumbnails surrounded by low-information browser UI.

Apollo will detect and track one content region, run the existing depth model on that region, and
map the resulting depth back into its original screen position. Browser chrome and sidebars stay
at zero disparity.

The runtime implementation is entirely native C++/HLSL/D3D11/TensorRT. Python may be used only by
the offline evaluator and scoring tools.

## Scope and non-goals

The production scope is deliberately narrow:

- one ROI at a time;
- one Depth Anything inference per accepted source frame;
- `full_frame`, `video`, `content`, and `scroll_hold` output states;
- image-based detection with no browser, DOM, extension, or accessibility dependency;
- no adaptive subdivision, tiled inference, per-thumbnail segmentation, retraining, second model,
  or second TensorRT execution context;
- no attempt to give browser chrome or sidebar advertisements depth; and
- conservative full-frame or flat fallback whenever the detector cannot identify a unique target.

A wrong focused ROI is worse than full-frame inference. Candidate exploration must not affect the
committed render state until a unique winner has remained stable for the acquisition interval.

## Detector inputs

Run a low-resolution full-frame analysis pass, initially `128x72`, at 10-15 Hz. It must use
nonblocking GPU readback and retain the last valid detector state if a staging slot is not ready.
The analysis produces:

- an exposure-invariant temporal-activity map;
- a photographic-density map from local color variance and dense multidirectional edges;
- a coherent vertical-scroll estimate;
- persistent low-evidence gutters; and
- candidate rectangles, content columns, and confidence evidence.

Temporal activity is occupancy-based rather than magnitude-based. A small flashing advertisement
must not beat a large video merely because its pixel values change more strongly.

### Primary video

A clear primary-video candidate initially satisfies this applicability contract:

- its visible rectangle covers at least approximately 12% of the viewport;
- motion persistently fills a meaningful part of its interior;
- it lies in the widest content column and is separated from sidebars by a stable gutter; and
- it has at least approximately 1.6 times the area/evidence of every other moving rectangle, or
  receives a recent interaction tie-breaker.

Common video aspect ratios are a bonus, not a requirement. A recent click or tap may break an
otherwise close tie, but interaction is never sufficient evidence by itself.

Once committed, a video ROI remains locked through quiet shots, pauses, subtitles, control
overlays, and sidebar activity. A challenger cannot steal the lock unless the incumbent
disappears or the challenger remains decisively stronger for a sustained interval. If two
unclicked videos are genuinely equal, the correct result is `content` or `full_frame`, not a guess.

Fullscreen video is not a special semantic case. A dominant candidate that covers nearly the
entire frame resolves to the identity/full-frame ROI.

### Content collage

When there is no unique playing-video candidate, combine photographic density with accumulated
scroll-history evidence:

1. bridge the small gaps between nearby dense-image blocks;
2. reject thin browser, toolbar, taskbar, and sidebar components;
3. select the largest two-dimensional content cluster;
4. take weighted bounds containing roughly 90-95% of its evidence;
5. exclude weak viewport-clipped outliers where doing so improves coverage density; and
6. add a small quiet halo for disparity feathering.

For the 3840x2160 diagnostic dump
`dump_20260728_121337_57928_18c689fbeeefc81c_0`, the expected complete-thumbnail envelope is
approximately `(846,525)-(3090,1761)`. A suitable model-aspect ROI is approximately
`(832,504)-(3104,1784)`, or `(800,485)-(3136,1801)` with a safer feathering halo.

That crop covers about 35% of the source area and gives approximately 1.69 times the linear model
resolution. It recovers more non-planar structure than the full frame, but one model view of
thirteen unrelated thumbnails can still look partially collage-flat. Independent depth for every
thumbnail is explicitly not guaranteed by this scope.

## Detector state machine

The committed states are:

| State | Behavior |
| --- | --- |
| `full_frame` | Existing Host SBS estimator and safe fallback. |
| `video` | Infer the single committed primary-video ROI. |
| `content` | Infer the single committed content envelope. |
| `scroll_hold` | Continue color streaming, fade disparity to zero, pause ROI history, and reacquire after the layout stabilizes. |

Uncommitted search/candidate states continue rendering the previous committed state. ROI
coordinates are quantized and hysteretic so small detector jitter cannot continuously move the
model input.

Coherent page translation enters `scroll_hold` within two detector updates. Scrolling frames do
not update ROI depth, appearance, or scene-cut history. After the page rests, the detector commits
one new generation. Tab changes, disappeared content, display geometry changes, HDR mode changes,
and fullscreen transitions likewise invalidate the old generation.

## Crop and depth mapping

Represent every committed selection with a value containing:

- normalized source rectangle;
- kind (`full_frame`, `video`, or `content`);
- confidence and acquisition age;
- monotonically increasing ROI generation; and
- capture frame identity.

The ROI must preserve its source aspect. Prefer a patch-aligned TensorRT tensor whose aspect
matches the ROI and whose short side follows `sbs_3d_depth_short_side`. If a requested shape cannot
be used, fit/letterbox it without anisotropic stretch. Never expand a crop across a stable gutter
merely to satisfy an aspect ratio.

Extend the existing exact-area preprocessing shader so each model texel integrates the
corresponding source-texel footprint inside the ROI. Preserve the current SDR/HDR color
conversion, ImageNet normalization, exposure-ordinal producer, and anti-aliasing contract.

The output renderer continues drawing the complete color frame. Reprojection maps full-frame
source UVs into ROI-local depth UVs. Outside the ROI and its feather band, disparity is exactly
zero. A raised-cosine confidence ramp crosses only the quiet outer halo; it must not introduce
mapping folds, eye asymmetry, or a fixed visible boundary.

## Matched-frame ownership

The asynchronous matched-frame contract is non-negotiable. Each private color slot must retain:

- source frame ID;
- ROI rectangle and kind;
- ROI generation;
- crop-to-depth affine; and
- color space.

The completed depth result is rendered only with metadata from its original matched color slot.
The hard invariant is:

```text
source_frame_id == completed_depth_frame_id == roi_frame_id
```

The renderer must never sample the tracker's newest rectangle when an older inference completes.
Repeating output repeats the entire previously matched SBS result rather than combining new color
with old depth or ROI coordinates.

## Scene-cut interaction

Content cuts and ROI geometry transitions are separate events:

- A real cut inside a stable ROI runs the existing exposure-invariant, geometry-corroborated cut
  detector. It resets adaptive pop, zero plane, subject state, normalization, and temporal depth
  state without unlocking the ROI.
- A cut in a sidebar advertisement lies outside the committed ROI and cannot relatch the video's
  zero plane.
- A changed ROI increments `roi_generation`, bypasses content-cut classification, and performs an
  explicit geometry reset without incrementing hard-cut counters.
- A pending result from an obsolete ROI generation is discarded before normalization so it cannot
  contaminate the new generation.
- Scrolling frames are excluded from content-cut history.
- A one-frame video cut cannot make the ROI tracker abandon its lock because ROI geometry uses
  independent multi-frame hysteresis.
- Telemetry reports `content_cut` and `roi_reset` separately.

The existing cut thresholds are fractional and therefore structurally compatible with a crop, but
they were calibrated on full frames. Cut-heavy embedded-video fixtures must demonstrate that fast
motion does not become a false cut when the video fills most of the tensor.

## Production integration

The implementation is expected to touch these areas:

- a pure C++ ROI scoring/state module that accepts low-resolution feature grids;
- a Windows D3D11 feature-extraction shader and bounded staging ring;
- `display_vram` for detector scheduling, matched-slot ROI ownership, state transitions, and
  render constants;
- `video_depth_estimator` for ROI-aware preprocessing, patch-aligned shape selection, generation
  reset, and obsolete-result discard;
- the common SBS warp/reprojection/coverage shaders for ROI-local depth lookup and the exact-flat
  exterior;
- the live debug dumper and telemetry contracts; and
- the native SBS benchmark harness so offline evaluation exercises the production path.

No second inference, execution context, full-resolution source copy, or blocking synchronization is
allowed. The detector must obey the same bounded/drop-on-busy philosophy as the existing depth and
telemetry paths.

## Diagnostics

Logs, telemetry, and Dump 3D should expose:

- detector and committed state;
- ROI rectangle, kind, confidence, age, and generation;
- all candidates, scores, winner margin, and rejection reason;
- main/sidebar column bounds and stable gutters;
- scroll vector and hold state;
- content-cut and ROI-reset counters/reasons;
- source/depth/ROI frame IDs;
- detector CPU/GPU time and readback drops;
- a source overlay showing candidates and the committed ROI; and
- temporal-activity, photographic-density, and gutter masks.

## Evaluation contract

Add deterministic browser-layout fixtures that composite existing real/synthetic clips into a
main player, sidebars, and surrounding UI. Cover:

- embedded video with a faster, brighter, earlier-starting, or 16:9 sidebar advertisement;
- several animated advertisements;
- dark video, fades, subtitles, controls, and letterboxing;
- a 30-60 second pause while advertisements continue;
- embedded/fullscreen transitions;
- slow scroll, fling, sticky sidebar, and partially clipped players;
- tab/page replacement;
- static light/dark collages with 2-6 columns, mixed card sizes, text, clipped rows, and animated
  tiles;
- equal split videos, PiP, video walls, and other genuinely ambiguous layouts;
- windowed browser beside another moving application; and
- matched SDR/scRGB cases, including a dark video beside a bright HDR advertisement.

Hard correctness gates:

- zero committed sidebar-ad ROIs;
- `source_frame_id == depth_frame_id == roi_frame_id` on every rendered result;
- model-input crop agreement with an independent reference within `2/255` for 99.9% of SDR
  samples;
- no aspect anisotropy above 0.1%;
- disparity outside the ROI and feather band no greater than `0.0001` pixel;
- no mapping fold or regression in existing eye symmetry, stretch, stereo-window, or cut-state
  contracts; and
- no old-generation depth on post-transition color.

Detector-quality targets:

- correct video on at least 99.5% of stable clear-target frames;
- video acquisition P95 no greater than 500 ms and every clear case no greater than 750 ms;
- content acquisition P95 no greater than 800 ms;
- stable-video median IoU at least 0.90 and P05 at least 0.82;
- content-envelope median IoU at least 0.80 with at least 95% annotated content coverage;
- no sidebar incursion beyond one analysis cell;
- steady ROI edge jitter P95 no greater than one analysis cell; and
- no committed target change during a retained pause.

Performance targets, measured over at least three warm 512-sample windows:

- tracker CPU P95 no greater than 0.20 ms;
- detector plus crop/map GPU P95 no greater than 0.25 ms;
- depth-inference P95 regression no greater than the larger of 0.10 ms or 5%;
- total Host SBS GPU and `sbs_convert_cpu` P95 regression no greater than the larger of 0.25 ms or
  10%;
- matched-frame-age P95 regression no greater than 2 ms;
- busy-drop/repeat regression no greater than one percentage point;
- per-stream VRAM increase no greater than 16 MiB; and
- no blocking readback, `Flush`, wait, TDR, device loss, DWM starvation, or allocation growth in a
  30-minute transition/scroll/pause soak.

## Implementation and rollout order

1. Add the pure detector state/scoring module and deterministic unit tests.
2. Add the bounded low-resolution D3D11 feature extractor.
3. Add ROI-aware preprocessing and patch-aligned ROI shapes.
4. Carry ROI metadata through matched slots and implement generation reset/discard.
5. Add ROI-aware warp mapping, flat exterior, and feathering.
6. Add scene-cut separation, telemetry, and Dump 3D evidence.
7. Extend the native harness and add the adversarial fixtures/contracts.
8. Run the real core/extended SBS evaluator and strict ROI performance gates.
9. Perform live Galaxy XR inspection and the 30-minute stability soak.

Develop behind an `off|auto` host setting. Keep it off while the detector and mapping contracts are
under construction. Make `auto` the production default only after every correctness, quality, HDR,
and performance gate above passes.
