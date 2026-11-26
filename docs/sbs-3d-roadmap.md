# SBS 3D status and roadmap

Current Host rendering uses the [shared Game 3D pin-band warp](host-sbs-review-followups.md#p15-production-shared-ui-warp).
Original RGB moves through one conditioned field instead of subtitle background reconstruction.
The shipped subtitle route is [Direct Glyph](host-sbs.md#direct-glyph-subtitle-mask): OCR finds the
text line boxes, and the Direct Glyph model decides which pixels inside them are subtitle. Live
Host 3D and Convert Video use the same configuration. There is no rule-based fallback; when
Direct Glyph is unavailable or fails, the picture is ordinary Base stereo.
Browser admission distinguishes an ordinary full-client video from a full-monitor client and maps
semantic video independently of a spanning browser's nearest monitor. These fixes do not establish complete decoration recovery or semantic rejection
of every scene object. Asymmetric video clipping still needs separately authenticated full-element
reference geometry to preserve the original caption-centering prior.

The only manual development launcher is `E:\ApolloDev\run-dev.cmd`. Physical monitor
recovery, live XR acceptance and live FPS remain unqualified by the headless checks. Shared pinning
still accepts nearby scene deformation and possible UI-row transitions; the preceding renderer
comparison has no universal quality winner.

Sunshine 3D ships one host-side geometry pipeline: Depth Coordinate V2. This document describes
only the current product, its known limitations, and work that may still be justified.

## Current product boundary

| Path | Geometry owner | Status |
|---|---|---|
| Host SBS live stream using Sunshine depth | Sunshine 3D on Windows/NVIDIA | V2 production pipeline, including a causally attributed foreground window-region ROI route with Chromium-video priority |
| Streamed Game 3D | Depth3D in the game; Sunshine receives final SBS | Negotiated discovery starts mono, then enters exact full SBS through an acknowledged transition; D3D11/D3D12 SDR/HDR handoff has controlled-runtime coverage; live game/device acceptance remains |
| Offline Host 3D conversion | Sunshine 3D isolated worker | Same causal V2 estimator/renderer, unpaced source-order processing, and compressed H.265/AV1 output |
| Client SBS | Moonlight 3D on Android XR | Separate client pipeline; unchanged by Host V2 |
| Local AR glasses | Sunshine 3D local presenter | Uses Host V2 or the local-only `sbs_reshade` provider setting without network encode/decode |

The [ReShade source](reshade-sbs.md) keeps the game and virtual display at normal resolution and
shares a separate full SBS texture. A compatible headset selects Game 3D without a host provider
toggle or restart. Resolution, FPS and Bandwidth belong to its own client pane. The stream remains
mono until source readiness and a confirmed transition; later source loss uses duplicate-eye
fallback while retaining packed geometry. Failed automatic widening stays latched until explicit
re-entry or a deliberate quality change. Host AI 3D remains the Sunshine depth provider.

Setup, HDR handling and outstanding device checks are documented in the ReShade guide. Exporting
original color plus depth for warping in Sunshine remains future work. Automatic Movie SBS image
detection is not part of Game provider discovery. The geometry and ROI behavior below describe
Sunshine's V2 provider.

Host V2 uses an authenticated DAV2 Small backbone with frozen ZipDepth convex-2x reconstruction,
one high-resolution model input/output and one high-resolution downstream grid. Its sole pipeline
continuously observes a Host-relative zero and scale, maps depth linearly, and protects current
native glyphs at an adaptive UI plane. It retains literal configured pop and bounded vertical and
horizontal scene-Base conditioning. Each Base eye has a unique contractive inverse; glyph source
selection can be discontinuous. Invalid or unauthenticated scene geometry renders flat.

On Desktop Duplication, a fresh exact matched-frame rectangle may replace full-frame analysis with
one source rectangle. The unique largest fully contained `<video>` in the foreground Chrome or Edge
document has priority; otherwise the foreground top-level root client may authorize the crop. The
rectangle is addressed directly inside the complete matched capture and fitted without cropping
or stretching into a centered integer content rectangle in the current authenticated tensor.
Edge-replicated synthetic padding is
excluded from analysis. A selected client that exactly covers the capture
canonicalizes to ordinary full-frame V2 with no crop or domain reset. Null/shell/desktop/self,
hidden/minimized/cloaked/excluded-style windows, missing content timestamps, invalid or stale
geometry, non-identity rotation, unauthenticated tensor geometry, and other-monitor/partial/spanning
rectangles all retain ordinary full-frame analysis. The host never changes the capture monitor for
this route; an internal base-V2 authentication failure still renders flat.

The live DDup path may also reuse authenticated geometry for an already-authorized, exactly
positioned ROI when a complete contiguous dirty/move sequence proves that every intervening desktop
change was outside that crop. This optimization changes neither ROI selection nor analysis
semantics: uncertainty returns to normal inference, while continuously proven unchanged input
does not require periodic refresh. WGC has no damage-reuse authority.

The complete implementation contract is in [Host SBS pipeline](host-sbs.md).

## What is intentionally absent

- No geometry endpoint normalization into `[0,1]`, subject stretch/recenter, adaptive pop gain, or
  configurable zero-plane translation. Private normalized depth/range EMA remains history
  evidence, separate from the adaptive geometry zero/scale controller.
- No damage-driven ROI selection, image-tracked ROI, background-window/tab ROI, or second-inference
  ROI. The foreground window-region route replaces that frame's full analysis with at most one
  causally attributed crop. DDup damage can only retain an independently authorized exact route;
  it cannot create or alter one.
- No application/media classifier, allowlist, Z-order compositor, occlusion reconstruction, or
  automatic capture-monitor switch in the foreground-client route.
- No forward-owner render, multi-root visibility selector, post-warp blur, or synthetic hidden-pixel
  fill.
- No CPU depth fallback or best-effort use of an unauthenticated model/shape.
- No timer-forced refresh of a valid near-identical depth/subtitle tuple. Geometry and UI adaptation
  consume genuine observations. Exact DDup/redelivery, or authorized changed-frame reuse with OCR
  unavailable or ineligible, can freeze unfinished adaptation indefinitely. A genuinely admitted
  ordinary source with eligible current OCR work uses the existing joined force-infer path.

The extra model, routing and renderer exclusions are architectural boundaries. The sole adaptive
geometry and subtitle-plane policy is owned by [Host SBS](host-sbs.md#adaptive-geometry-and-subtitle-plane).

## Known limitations

### Monocular framing dependence

DAV2 is a relative monocular model. Frozen convex reconstruction improves its spatial realization
but does not add metric depth or a second independent depth estimate. Fullscreen, windowed, or
differently surrounded versions of the same image can produce materially different raw depth. More
tensor pixels on the main subject do not guarantee a better boundary. Synthetic context must not
influence monocular analysis. The
window-region route therefore uses only the exact client/video pixels: its edge-replicated tensor
padding is excluded from depth statistics, history comparison/authority, and
OCR authority, although the fused map copies the complete raw padded tuple with exclusion `1`.

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
every producer-admitted paired current OCR8 cover and compact UIP4 current-publication state
at the authenticated active field shape. It does not retain the retired row-history or general overlay-detector pipelines.
Unsupported identities and shapes preserve ordinary V2 exactly. Dump and replay accept only the
current UIP4/OCR8 schema. The host does not reconstruct hidden video.

The historical candidate115 subtitle pipeline passed native regression checks,
same-input real-clip comparisons, authenticated production-dump inspection, and the user's Galaxy XR
live check, which reported normal character edges. Later live testing exposed delayed protection
at subtitle changes and protection after disappearance. Historical schema 118/OCR 30 retained immediate
current-evidence support in place of RGB-confirmation gating and provisional whole-cover protection.
It additionally makes the current authorized histogram's entry-budget UI-plane candidate an
immediate forward floor, removing approach dwell and the forward rate ramp that delayed placement
on the observed transition. Delayed backward retreat remains. The
candidate115 acceptance does not qualify this policy or establish complete glyph recall, a
production false-positive rate, or performance across devices and workloads. OCR30 additionally
classifies local text lines before protection using native line orientation and independent-episode
layout history. This targets false current detections such as printed badges and document text;
current empty evidence still clears protection immediately. The classifier's experimental thresholds
and first-entry unknown/angled-caption recall require qualification; it adds no text recognizer or
motion model. Canonical parameters and history ownership live in the subtitle section linked below.
Historical recipe 25 bounded ordinary native support vertically around the classified raw core and used a
quartic radial falloff to reduce nearby scene deformation while retaining near-glyph weight. This
is a presentation hypothesis requiring native field, full-glyph recall and paired quality checks;
a continuous slope-limited collar can still affect immediately adjacent content.

The first OCR30/recipe25 comparison is screening evidence, not release acceptance. The two CJK
probes retain 100% glyph-plane inliers; the tall bilingual stack remains below its recall target
and loses some incidental falloff coverage above the admitted crop/search region. Its full-frame
audit also exposed a new in-search middle-line rejection: strict envelope orientation confused
uneven character ink with line tilt. Exact native replay and a fresh paired comparison verify that
bounded current stack assistance restores that line without changing learned history or admitting
the outlying false fragments. The separate upper-crop and falloff limitation remains. The disjoint
layout improves but also remains below target. Exact native replays remove the observed badge
protection and substantially reduce the original paper-region effect, while a narrow collar near
the caption still affects adjacent price text. A separate cold replay retains nearly horizontal
photographed-page text as well as its real caption. Orientation and learned position are therefore
useful evidence rather than a complete scene-text classifier. Actual first-entry short-caption
recall and live onset/release still require qualification.

Historical schema 119/OCR 31 additionally constrained ambiguous prior-assisted candidates by their current line
geometry. Captured device/hand false positives had qualified position/height memory but diverging
or excessive envelope slopes; location history must not override that current evidence. Recipe26
replaces the Gaussian plateau/tail with the nearest slope-feasible interval around completed current
native footprints. It preserved immediate support/empty-mask release and shared numerical
guards. Native replays and its fresh full-estimator twelve-observation sequence preserved all
457,079 authored body, outline and antialiased glyph pixels, including first entry and replacement;
all four empty observations copied Base exactly. Both captured device/hand fixture cases produced zero
support and exact Base. Same-input native fixture background displacement above the captured subtitle fell from
6.45 to 0.97 pixels per eye at the 95th percentile. Some low-resolution collars widen, residual
boundary warping remains, and captions outside current covers/crop remain unsupported. The paired
screening reports retain those limitations. These are Build 119 results and do not qualify Build 120.
The user's subsequent Build 119 live test rejected its halo behavior and exposed a missed leading
Chinese stroke and visibly warped lower gray strokes. The captured UI plane remained at 36 pixels
per eye while its current release candidate was much lower. Passing the
authored sequence did not establish complete real glyph recall or live halo/comfort acceptance.

Schema 120/OCR 32/recipe 27 addresses those observed failures with separately testable changes.
Ordinary OCR search covers gain horizontal and vertical search margin without changing their
tight cores into masks. The UI plane retains the immediate current entry floor, resets invisible
history on genuinely newer valid empty evidence, and adapts backward toward the current release
candidate with a 100 ms source-time exponential response rather than repeated dwell/staircase goals.
Pixel selection learns current core color modes and admits matching short strokes or locally
witnessed dark outlines within center-guided bounded ordinary search and full ribbon search;
high probability is palette-anchor
evidence and cannot directly authorize a filled word rectangle. Native field, full-glyph/AA,
negative-region, paired full-estimator and live checks remain required. Earlier acceptance or
fixture results are not silently promoted to this policy.

The 2026-10-25 native fragment replay passes 21 current contour/presentation cases after 20
genuine current OCR replays. All 457,079 authored nonzero-alpha glyph pixels, including 63,851
antialiased pixels, retain zero budget, unit influence and exact UI-plane words at all four
sampling contributors. Nine empty cases preserve complete Base bits. At identical UI targets,
the authored alpha-zero background displacement sum falls 34.98% from Build 119, and the area
changing more than one native pixel falls 26.51%; every individual AA case is nonincreasing.
The current live leading-stroke witness and lower gray-stroke witnesses also retain all four
protected contributors. Those live labels are partial witnesses, not complete semantic ground truth.

These native replays freeze source/probability/Base and fixture UI authority. Holding the captured
36-pixel plane still leaves horizontal influence near the gold background; the separately authored
4-pixel alternative is not a real controller run. The paper close-up was not present in the latest
six live captures.

The subsequent full-estimator comparison passes four formal clips and the 12-frame AA sequence,
with neutral screening decisions and no hard, evidence or conformance failures. All 457,079 AA
sequence glyph pixels retain exact zero budget, unit influence and UI proposal. Final matches the
published horizontal guard exactly; two frames use the required half-Q30 upward rounding rather
than literal raw UI words (about 0.0000012 native pixels). All seven valid-empty observations clear
hidden controller state and copy complete Base bits. The full AA sequence's alpha-zero background
displacement sum falls 39.72% and its area over one pixel falls 29.60%. The subset with identical
actual UI targets falls 29.80% and 21.32%, respectively. The complete sequence includes legitimate
controller target changes, so its improvement is an end-to-end result. Formal upper/off-crop and
off-cover warnings remain recorded; these results do not establish global subtitle recall.

A fresh serial 24-frame comparison uses private frozen Build 119 assets and the same engine,
source frames and clocks under the shared GPU lock. OCR median remains about 0.776 ms. Subtitle
median increases from 0.604 to 0.694 ms, while mean falls from 1.057 to 0.936 ms and p95 from
2.731 to 2.085 ms. Earlier timing outliers are retained as diagnostic data, rather than used as a
matched speed reference. This supports the bounded user test; live halo/comfort acceptance remains
pending.

OCR supplies current search covers and geometric subtitle eligibility. Nearby boxes can form a
line for the eligibility check, while distant columns cannot borrow each other's center. The
reference comes from a matched video ROI when available and otherwise from the captured screen.
An ordinary cover includes four producer cells of horizontal guard and a vertical half-tight-height
guard added to DB expansion for omitted end or lower strokes. None of these
operations replaces pixel protection with a line-union rectangle. The authoritative center,
lower-region, grouping, capacity and guard rules remain in
[native subtitle contours and UI plane](host-sbs.md#native-subtitle-contours-and-ui-plane).

Within that current search, native pixels require valid current RGB, all four finite
unit-range probability contributors, and current palette/stroke or local dark-outline evidence.
Strong detector probability plus native contrast selects palette anchors; it is not a direct
per-pixel protection bypass. Mode matching, bounded color-coherent tangents and fixed outline
neighbors reject some unrelated scene edges without adding a frame confirmation wait. Positive
diagnostic count preserves duplicate admission. Supported pixels seed the bounded same-frame native
matte, which is a diagnostic output. The rendered subtitle pixels come from Direct Glyph, which
reads the source colour inside the OCR boxes. Actual rendering uses one shared pin field over that
selected matte and the surrounding
scene; ordinary Base remains the fallback when subtitle preparation or authority is unavailable.
The analytic subtitle selector is removed. The former interval extension and horizontal guard
remain independently checked diagnostics, not the rendered subtitle mask.
Ordinary scene depth retains both axis guards.
The plane's current clearance goal uses the existing cover-histogram conflict budgets; it is not
an instantaneous continuation floor, the maximum scene depth, a guarantee of clearance for every
glyph, or a comfort qualification. Its entry, continuation and reset policy is owned by the
same canonical subtitle section.
The fixed-RGB four-confirmation history and current-core change cues remain diagnostic; they do not
delay support or extend protection. There is no provisional whole-cover term. Whole-tuple reuse
holds the completed observation. The current route retains neither the former cover-owner tracker
nor a second subtitle pipeline. Its current support, diagnostic history and interval semantics are
owned by the same canonical subtitle section.

The boxes and current evidence do not identify semantic glyphs. Scene texture that matches a current
palette/stroke or outline witness inside a false cover can still receive support; probability,
RGB stability and cover membership alone cannot protect it. This bounded overlay-color/stroke assumption is not a calibrated semantic
guarantee. A busy text region can discard
smaller candidates; detector omissions beyond the guard and off-center captions without a matched
video ROI can still be missed. Low-probability, low-contrast antialiasing, fades and thin strokes can
fail the current glyph-evidence tests. Broader font, color, HDR, transition and false-positive qualification remains
required for the current policy.

The private subtitle research that preceded Direct Glyph has been removed from the host: the
inner-fill, cohort and matte-guard masks, the temporal source census, the current-RGB cue and
current-source snapshot helpers, the joined-completion, pending-current-colour and queued
same-source presentation probes, and the disabled queued converter route with its shared UI warp
shader. None of them reached
live streaming or Convert Video. Their recorded outcomes remain in the repository history. Every
rule-based mask variant either left background inside the fixed margin or lost required thin-font
and startup inner fill, and presenting current colour under an old mask lingered after text changed
because it cannot discover new glyphs. Current-frame glyph detection is therefore a prerequisite for
any later presentation route of that kind.
The CUDA branch receipt commits only after both conditional children complete.

Damage reuse is deliberately DDup-only and fail-open. Dirty and move metadata is semantic only as a
proof that the exact current ROI pixels did not change: damage outside the crop may save inference,
whereas an intersecting move source or destination, a sequence gap, metadata failure, protected
content, route movement, authority change, or WGC forces the wrapper's inference path. Reuse freezes the
depth/camera/OCR/UIP4 observation state and always warps the current capture color rather than
repeating a packed SBS frame. This exact-current DDup proof is distinct from the presentation-only
fallback below, which is eligible only when no matched completion reaches the render deadline.

When changed B reuses the complete tuple owned by A through the OCR-unavailable or ineligible
GPU-undecided route and then stays static, exact continuity from
B's authenticated publication may retain A indefinitely. There is no idle-tail correction or
periodic refresh. B remains the completed color and A remains the depth/subtitle owner; the next
changed candidate compares against A. Local and remote display conversion share this behavior.

GPU adaptive reuse is always part of the Host SBS depth pipeline. Exact cache reuse remains higher
priority; otherwise host-only route, identity and complete DDup history classify a changed frame as
either force-infer or GPU-undecided. Eligible ordinary OCR work uses the existing force-infer request;
only the remaining OCR-unavailable or ineligible changed candidate may be GPU-undecided. Eligibility
uses host-owned resource, work-mode and source-geometry facts, without reading image-derived OCR
evidence. No host motion-size/shape heuristic applies. The dense atomics-free model-space detector
compares every admitted current NCHW texel with the authenticated state-1 history and applies finite,
global medium/strong, and supported-tile local bounds. For GPU-undecided work, the GPU selects the
DAV2 infer/reuse branch without a CPU decision poll or wait. Only after final D3D publication may a
nonblocking receipt report the completed outcome to the CPU. Depth and subtitle use one joint
observation policy. Current-ready OCR (`1`) or current
ineligible abstention (`2`) publishes only on valid infer; native USER32 suppression (`0`) advances
neither OCR nor plane state. Every reuse preserves depth/camera/V2 Base and the complete
OCR8/UIP4/conditioned-final tuple. No mode relabels
retained boxes as a new OCR observation, and there is no adaptive OCR-band, localized subtitle or
SLR-state veto.

Reprojection consumes the complete atomic final field directly with current capture color. A
validated completed-publication receipt distinguishes completed color from the actual depth and
subtitle owner. Only that exact current proof may establish reuse authority; missing, stale or
invalid proof requires ordinary force-infer. Pending work and known completed lineage replace the
former persistent opaque barrier and separate follow-up anchor. The device compares cumulatively
against the actual infer input even when a newer color publication reused it. Reuse never moves
that fixed comparison baseline, and neither elapsed time nor hold count forces inference.

Without receipt proof, a completed draw may retain its already-rendered packed SBS pixels only for
bounded presentation continuity. If the next root misses the same-frame completion budget, that
image can be redelivered on the unchanged route/domain instead of flashing flat; packed pixels alone
carry no depth, OCR, UIP4, damage or adaptive-submission authority. Invalid ownership, route/proof
failure, or interop-signature drift still forces CPU-known inference. Diagnostic traces and aggregate
outcome counters remain separate from production receipt authority.

Coherent GPU temporal state is not synthetically reset. Every post-bootstrap DAV2 submission,
including force-infer, uses the same conditional wrapper; wrapper/capability/interop failure is
terminal flat rather than a hidden raw-TensorRT fallback. The canonical thresholds, ownership,
sub-threshold depth-reuse risk and reset matrix are in
[Host SBS frame attribution](host-sbs.md#frame-attribution-and-failure-behavior).

Depth and subtitle share one joined disposition, with no time/count expiration. A genuinely admitted
ordinary source with eligible OCR work uses force-infer; the remaining GPU-undecided work shares the
device detector decision. Current-ready OCR uses `1`; an ineligible OCR input/interop/child uses `2`
and publishes current abstention on valid infer. A suppressed completion cannot authorize ordinary
joint reuse without a complete ordinary owner under the current route. Exact unchanged DDup proof
bypasses admission and submits no root.
Dump 3D always forces a complete ordinary observation.

### Adaptive completion status

The presentation-only packed-SBS fallback is deliberately narrower than semantic cache reuse:
already-rendered packed pixels may provide bounded continuity after a missed render deadline without
becoming DAV2, OCR, UIP4, damage, telemetry, or adaptive-submission authority. Same-frame polling is
therefore a freshness/latency optimization rather than a geometry-correctness boundary.

Completed in schema 67: the schedule-coupled three-phase display recurrence was removed and the
complete atomic conditioned final field is rendered directly. The recurrence was introduced for the
earlier infer/reuse clocking hypothesis; removing it eliminates a persistent full-field resource and
pass, effective field lag, and rendering semantics coupled to reuse-owner age. Keep the GPU history
owner and diagnostic trace. The subtitle route uses exact current OCR8 covers without historical
cover ownership. It completes immediate current glyph-support footprints toward the adaptive UI
plane only within the minimum slope-budget interval, followed by the shared numerical guards.
Fixed-RGB confirmation remains diagnostic apart from duplicate admission. Historical compatible-handoff
fading from an old depth field and provisional whole-cover protection remain retired.

The current policy is schema 6: detector-only joint depth/subtitle reuse, legal work `0`/`1`/`2`,
and whole-tuple holds. The earlier independent due-OCR work `8`/`16`, host age/count budget and
stable-tail forced observation have been removed. Live and offline replay share the same detector
admission and joined completion proof. Every candidate compares cumulatively against the real last
infer owner; reuse does not advance it. The offline replay authenticates long changed-source holds
and their exact retained fields, but supplies no live DDup idle-tail evidence. A small semantic
change may remain unobserved indefinitely while the detector approves it.
Do not restore the retired host motion-size, OCR-band, localized-tile or SLR-state inference vetoes.

The completed-publication receipt reuses existing finalized GPU records through bounded nonblocking
copies, with no new shader pass, model call or calibration change. Local presentation, remote
streaming and the offline estimator share its wire authentication. Exact publication generations
prevent late receipts from authenticating overwritten borrowed resources. The design removes
post-completion branch uncertainty from host ownership bookkeeping; it does not promise a faster
GPU inference or higher FPS. Receipt readiness, saturated-slot fallback, final SBS latency and
weak-GPU contention still require measurement under the shared qualification workflow. The full
transport contract is owned by [Host SBS](host-sbs.md#gpu-owned-near-identical-transaction).

For latency, keep DAV2 and OCR as siblings of the same CUDA root; do not add another stream or graph.
Diagnostic-only parent event nodes now measure both sibling conditional completions from one common
post-setter start. They include scheduling and skipped bodies, are consumed only after exact current
root completion, and do not establish isolated model time or OCR slack. Their semantics are owned
by [Host SBS performance observations](host-sbs.md#performance-observations). If profiling shows OCR
slack, the first candidate is its buffer-only probability-cell reduction and OCR8 candidate resolve moved into an OCR
child tail after TensorRT. Keep shared RGB-to-NCHW and similarity work before the root, and keep
UIP4/conditioning/warp after the join. Port OCR crop preprocessing only if its measured benefit
exceeds the source-texture interop cost. This is expected to offer only a modest critical-path saving,
not a utilization claim, and must pass a weak-GPU contention gate. Remove each superseded HLSL
implementation after corpus parity rather than maintaining duplicate live/offline algorithms.
Saved common-start observations show earlier OCR conditional completion, but a source audit finds
that both OCR interop release and the shared transaction proof remain behind the full graph on
the existing host stream. An earlier diagnostic event alone therefore cannot publish a D3D-readable
current OCR result. The observations do not qualify independent release, current-mask readiness or
an end-to-end latency saving.

The policy no longer forces an inference after four deliveries or `100 ms`. Qualify prolonged
comparison against the fixed real-inference owner with shared-replay, exact-trace, cumulative-drift,
subtitle-transition, weak-GPU-tail, and live-XR evidence.
The acceptance metric is final SBS readiness and presentation continuity, not the raw CUDA-root
event time or a promised reuse percentage.

### Subtitle maintenance and player controls

The November 2 subtitle review is tracked in
[Host SBS review follow-ups](host-sbs-review-followups.md), including original confidence labels,
priority repairs, rejected findings, and the ordinary evaluator's live-composite coverage gap.
Geometry-sensitive subtitle identity, missing-detector failure isolation and native-player caption
references are qualified there separately from interactive move/size and glyph-quality work.

Subtitle pixels are selected by [Direct Glyph](host-sbs.md#direct-glyph-subtitle-mask). The
earlier rule-based pixel stage and its maintenance experiments were removed with it. Complete glyph
recall, temporal stability, live latency and GPU speed of the current route remain unqualified by
headless checks; the file-fed
[Direct Glyph qualification](../tools/sbsbench/README.md#direct-glyph-qualification) is the
only harness evidence for it.

Normal subtitle quality takes priority over player-control overlap. A simple local compatibility
fallback may cover controls as well as text, but must leave ordinary no-controls output unchanged;
the [pipeline owner](host-sbs.md) records its evidence and regression requirements. This lower-priority
case remains deferred while no reliable current-controls visibility signal is available.

### Foreground crowns and disocclusion

A single source image contains no observation of background hidden behind a shifted foreground.
V2 preserves foreground continuity while limiting the slope required by its inverse warp. On a
large raw cliff this converts one discontinuity into a wider safe ramp. Hair crowns, transparent
glass rims, and small near-object tops can therefore bend or stretch differently in the two eyes.

### Unsupported identities fail flat

Live Host SBS, production conversion, and the maintained evaluator accept only the authenticated
DAV2 Small calibration, required fused DAV2/ZipDepth composite identity, preprocessing closure,
supported [resolution fit](host-sbs.md#authenticated-resolution-fitting), producer/state contract,
and renderer closure. There is no Host SBS model selector or unauthenticated fallback.

### Live telemetry is sampled

Routine depth-health and cumulative outcome samples share the completion receipt's nonblocking
copy/event/map transport. Their validity remains separate from rendering proof and from each other;
capture-time age and publication ordering prevent delayed samples from appearing fresh. Samples may
be missing under GPU load, while offline traces are complete. A live/offline discrepancy must first
be aligned by exact source-frame identity before it is interpreted as a controller difference.
Dump 3D, performance timing and GPU safety queries retain their own lifecycles.

## Qualification priorities

Before changing V2 geometry:

1. Reproduce the issue with a current Dump 3D package and identify whether it begins in the fused
   raw depth, candidate mapping, cliff conditioning, or the inverse warp.
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
   edge; repeated and missing sequences; metadata/protected-content failure; WGC fallback; and
   continuous clean proof beyond the retained history ring versus an evicted unchecked interval.
5. Confirm the result in Galaxy XR at the intended pop strength before changing the production
   contract or baselines.

Galaxy XR window-region and subtitle scenarios, each exercised for at least 20 seconds and finished
with five seconds of a static image; the [pipeline contract](host-sbs.md) owns the rules they test:

- Enter and leave both window-region ROI authorities, including Chromium priority, and acquire and
  release a structural content child.
- Move a same-sized foreground window without changing its content, then resize it across an
  analysis-domain boundary and switch between Chromium-video and foreground-client authority.
  Pure translation must not reacquire the camera; an analysis-domain change must reset state
  instead of comparing incompatible history.
- Enter both the semantic Chromium fullscreen and the generic exact-full-capture foreground-client
  routes; each must keep ordinary full-frame continuity instead of creating an ROI transition.
- Land an input-domain reset on visible static subtitles. Valid current pixel evidence must give
  immediate support without a four-count wait or pre-reset support reuse.
- Keep a subtitle on screen across an editorial cut. Its UI-plane filter memory and pixel
  confirmation history must persist.
- Let a caption disappear. A genuine newer valid-empty observation must clear hidden controller
  state; repeated clocks and reuse must hold rather than spend time or manufacture a reset.
- Introduce moving scene text inside admitted OCR covers, and static non-subtitle text. Moving RGB
  must not inherit mature diagnostic counts; false current probabilities and textured scenery
  inside false current covers remain declared false-positive boundaries.
- Append and remove a second line, reorder or split OCR boxes, and repeat an observation.
  Still-covered qualifying same-pixel fixed RGB references keep their history, newly admitted
  diagnostic pixels start at one, empty or invalid current authority returns exact Base, and
  duplicate identities add no diagnostic confirmations.
- Replace one real subtitle cue with another, including similarly colored glyphs and a genuine
  disappearance. Pixels must lose support when current evidence fails, and a false cover alone
  must not retain protection after disappearance.

Repeat representative cases in SDR and HDR and across authenticated landscape, ultrawide, and
portrait tensor shapes.

The immediate priority is to qualify the single-high fused adaptive pipeline
across paired dumps and Galaxy XR, including high-grid temporal/reuse behavior and the
small-source bilinear-upscale exception. The producer shares its raw histogram traversal with private
normalization while preserving both populations and quantile policies; current-cover UI probes
also retain their exact visits through cached group scratch and add/carry coordinates. These are
work reductions requiring paired output and timing evidence, not a new distortion or gain policy.

The frozen ZipDepth branch already uses guarded FP16 feature/mask weights with FP32 public I/O and
convex reconstruction, projection-before-resize, and exact block-diagonal dense replacements for
grouped pointwise convolutions. The accepted graph, measured evidence and rejected alternatives are
owned by [the production fused-model document](host-sbs-prod-zipdepth-convex2x.md#model-only-optimization-result).
Further model/kernel rewrites require fresh evidence of a remaining latency bottleneck; the earlier
attention-GEMM and generic precision experiments are not unfinished production work. Geometry work
remains a precision-first reduction of
ambiguous small-object crown distortion: a candidate must improve glass-rim and small-near-object
witnesses without reintroducing the hair/shoulder discontinuity or hand-boundary halo.

## References

- [Host SBS pipeline](host-sbs.md)
- [Offline Host 3D conversion](whole-clip-sbs-pipeline.md)
- [Local AR glasses](sbs-local-ar-glasses.md)
- [sbsbench evaluator](../tools/sbsbench/README.md)
