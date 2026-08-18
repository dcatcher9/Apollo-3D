# Host SBS scene cuts

This document is the canonical contract and acceptance plan for the Host SBS scene-cut detector.
The detector has one authority: emit a confirmed cut generation and one-frame pulse that invalidates
the scene camera described in [Host SBS pipeline](host-sbs.md). Its normalized depth, history,
telemetry slots, and classification flags never become per-pixel geometry.

The design goal is deliberately asymmetric:

- brightness alone must not reset stereo geometry;
- persistent motion must not periodically retrigger a cut;
- a later observable cut must remain detectable during sustained motion; and
- ambiguous evidence should preserve the current scene rather than guess.

## GPU evidence

Every valid matched model-input/depth pair contributes these frame fractions:

| Evidence | Definition |
|---|---|
| Depth change | Normalized-depth texels changing by at least `0.05` |
| Raw RGB change | Model-domain texels whose largest channel delta is at least `0.20` |
| Structural change | Supported local max-RGB ordinal comparisons that reverse ordering |
| Structural support | Current, previous, and common support for the ordinal comparison |
| Brightness direction | Changed RGB texels whose max-RGB rises or falls |

The ordinal descriptor compares the ten pairs in a center/left/right/up/down stencil. A pair is
usable in each frame only when its max-RGB contrast is at least the larger of `0.0001` and 4% of
the brighter endpoint. A texel votes after at least four usable pairs, two reversals, and a
reversal majority.

The point-sampled max-RGB ordinal is normalized to scene-linear units: encoded SDR is decoded,
while linear SDR and scRGB are already in that domain. It is recorded before the HDR model tone
map and before spatial resizing. Under one global monotone per-channel exposure transform, a
supported ordering may stay the same or collapse into an abstaining tie, but it cannot reverse.
This does not claim invariance to local tone mapping, codec noise, color-matrix changes, or
independently transformed channels.

### Analysis-domain ownership

Ordinary V2 computes every item above from the full captured frame. When a window-region ROI is
authorized for an exact matched frame, its exact logical rectangle is the entire analysis domain
even though shaders address it directly inside the retained full-frame texture: DAV2 input,
normalized-depth comparison, appearance and ordinal evidence, scene center, baselines, and history
all exclude the surrounding desktop. A causally authenticated Chromium `<video>` has priority;
otherwise a causally continuous foreground root client may provide the rectangle. The retained full
captured color texture supplies those offset analysis loads and remains the final renderer source.

Evidence from different domains must never be compared. Entering or leaving ROI mode, changing the
authority kind (`chromium_video` versus `foreground_client`), authorized identity or crop
dimensions, or changing the input transfer domain clears cut history, baselines, pending
confirmation, and the scene camera before analysis resumes. Translation of the same authority and
same-sized crop retains the analysis domain and its histories; the exact matched-frame rectangle
still determines where that frame is rendered. Translation nevertheless revokes any older
positioned completion or cached ROI output; only a newly retained matched frame may reuse the
histories. DDup damage-guided reuse is not a translation exception: it requires the exact positioned
route and every live authority epoch to remain unchanged. A missing, stale, unsupported, spanning,
partially off-monitor, or other-monitor ROI selects full-frame V2. That resets state when leaving an
active ROI, but repeated ineligible observations while already full-frame are not domain changes.
Any selected client rectangle that equals the complete capture is already the canonical full-frame
domain: it creates no crop, ROI transition, history reset, or camera reacquisition. Chromium may
reach that result through separate semantic fullscreen evidence; the generic foreground-client
route reaches the same canonical result through exact geometry alone. The browser helper labels its
relaxed full-client evidence `ok-fullscreen`, distinct from strict `ok` video-ROI evidence, and the
host rejects an `ok-fullscreen` subrectangle. The bottom subtitle locator observes whichever domain
V2 actually used. Any real input-domain transition resets the subtitle locator together with the
ordinary cut and camera histories.

A bounded DDup damage reuse creates no scene observation. Once an authenticated ROI completion owns
the exact positioned route, a contiguous and complete sequence of dirty/move metadata may prove
that every later desktop change was wholly outside that analysis domain. Repeated delivery of one
immutable capture sequence is idempotent, while a missing or out-of-order sequence, metadata
failure, protected-content masking, or any dirty rectangle, move source, or move destination that
intersects the ROI forces the ordinary matched path. WGC supplies no such proof. On a reuse, raw
appearance, ordinal evidence, normalized-depth comparison, cut confirmation, baselines, scene
camera, OCR, and SLR do not advance, and an already-consumed cut pulse cannot be applied again. A
real enqueue is forced after 16 skipped deliveries or 250 ms even when the metadata chain remains
clean; only that successful enqueue establishes the next sequence baseline.

The always-on GPU adaptive path may omit up to four complete DAV2 observations for an authenticated
changed-frame candidate. The host uses only route, identity, cadence and complete retained DDup
history to classify that frame as GPU-undecided; it does not apply a motion-area or rectangle-shape
heuristic. A dense GPU comparison against the exact state-1 DAV2 model
input history then applies finite, global medium/strong, and supported-tile local bounds; there is
no CutBridge readback or CPU decision. The infer owner is eligible only through frame age `4` and a
strictly less than `100 ms` source-observation age. The host initial-candidate and opaque-follow-up
checks share that strict `100 ms` age ceiling.

Depth and OCR remain independent request domains inside one joined completion. Ordinary current-
ready OCR work `1`, or ordinary ineligible abstention work `2`, publishes only on infer. Cadence-due
current-ready OCR work `8` runs on infer or reuse. Cadence-due ineligible work `16` publishes an
exact-current abstention on either branch and can never authenticate an optional `OOCR` marker. An
ordinary reuse freezes the complete OCR8/SLR13/conditioned-final tuple along with depth, cut,
camera, and V2 Base. A due reuse freezes the depth tuple but advances the subtitle observation and
conditions retained immutable Base. Native USER32 suppression publishes neither OCR nor locator
state and publishes Base as the atomic final field. The private depth branch still cannot become CPU
depth cache, damage lineage, or telemetry.

The adaptive depth decision deliberately has no OCR-band, localized subtitle-transition, or
SLR-state veto. Ordinary work consumes OCR cells only with complete proposal/request/optional-
receipt proof and otherwise publishes a current abstention on infer; ordinary reuse holds. Due work
is branch-independent: value `8` consumes current-ready OCR only with the optional receipt, while
value `16` records the ineligible abstention with the optional handle off. Either due disposition
then runs the complete conditioner from immutable V2 Base to the atomic final field. A malformed
force proposal fails depth open to infer without authorizing stale OCR input. A CPU-known force with
malformed receipt proof publishes an exact-current OCR abstention and complete Base-derived output;
retained boxes are never relabeled. Invalid opaque receipt evidence instead holds the coherent prior
tuple. Failure to publish the complete joined tuple is terminal.

Every GPU-undecided transaction arms a depth observation barrier regardless of its opaque branch.
Exact depth-cache reuse and new CutBridge telemetry cannot bypass that barrier. The barrier may
admit a fresh same-route DDup-authenticated follow-up bound to the
immediately preceding opaque transaction; that scalar anchor is not host cache or branch knowledge.
The device-owned history remains the last actual infer input. It may be older than the immediate
host anchor, but the cumulative current-to-owner comparison is allowed only through frame age `4`
and while its source-observation timestamp is strictly less than `100 ms` old. Ordinary reuse
freezes the subtitle tuple; due reuse publishes current OCR or abstention independently. A sequence
that genuinely needs DAV2 may keep submitting within the existing bounded route/time authority.
Expiry, route or
damage discontinuity, signature drift, or any final
attribution failure returns to a CPU-known force-infer transaction.

An infer-gated owner record supplies the exact last accepted-infer frame to the next scene seed, so
the next observation uses C-B after an opaque infer and C-A after an opaque reuse without readback.
Thus a sub-threshold non-exact change can be absent from CutBridge for at most four deliveries and
strictly less than `100 ms`; the next frame step or age boundary forces infer and cannot reconsume
an old pulse. Subtitle observation has the narrower independent bound: every accepted ordinary
opaque root is conservatively one dirty hold, and two such holds or `33 ms` since the last
guaranteed observation makes the next accepted root due. An eventual CPU-known force completion
restores host depth-cache lineage without resetting coherent GPU temporal state.
Missing detector, conditional-graph, interop, or wrapper authority is terminal and renders flat; it
never silently submits DAV2 outside the wrapper. The detailed thresholds and ownership contract are
owned by [Host SBS frame attribution and failure behavior](host-sbs.md#frame-attribution-and-failure-behavior).

There is no OCR-only DDup retained-box proof. Each accepted root requests ordinary infer-coupled
subtitle work until the conservative cadence is due. If the due root has current-ready OCR input,
interop, and a captured child, work `8` runs OCR on either depth branch. If any of that current
authority is ineligible, work `16` publishes the exact-current abstention on either branch with no
optional child. Neither path relabels an older OCR8 payload. An armed Dump 3D root remains a complete
ordinary force-infer observation.

The production subtitle path applies no overlay exclusion to cut evidence or DAV2. Subtitle
appearance and disappearance remain ordinary scene evidence; the locator consumes the
already-resolved cut result and can neither suppress nor retroactively change it.

The production detector-only PP-OCRv6/OCR8/SLR13 subtitle path has no private cut classifier. It
consumes the authenticated durable CutBridge hard-cut epoch; processing the same frame/domain
identity cannot advance state twice. Ordinary DAV2 reuse dispatches no locator observation. A due
subtitle observation may publish on reuse, but it sees the retained CutBridge epoch and cannot
invent or reconsume a raw one-frame pulse: only an authenticated epoch transition consumed by a
distinct subtitle observation starts SLR cut handling. The saturating maximum epoch is the terminal
unique-generation lifetime
for one uninterrupted domain. The conditioner also binds that
CutBridge resource and copies Base unless the locator scene epoch equals its authenticated hard-cut
count. A confirmed cut clears pending and death-grace
state, preserves only current OCR rectangles that still match an old owner, and samples the new
local supporting plane. At the aggregate primary, two complete finite in-container rows use their
robust medians without an interquartile-range gate: close medians are averaged and separated
medians choose the larger-U, nearer support. If only one primary row is valid, that row must pass
the generated IQR gate. Otherwise the same strict two-row coherent fallback policy applies.
A reliable survivor restarts on the selected plane at half fade strength;
an unreliable one conditions exact Base. The old full-strength target never crosses the cut.
Additions or disjoint boxes
start a new two-observation transaction. An
input-domain reset clears the owner too and records current boxes only as the first pending
observation. Missing, stale, abstaining, malformed, or mismatched OCR8 evidence in an otherwise
valid unchanged domain clears current and pending subtitle authority, starts or advances the
generated bounded cached-target grace defined by the Host SBS subtitle contract, and copies the
ordinary post-limit field
exactly; processing the same identity does not age grace twice. Invalid OCR on a hard-cut/reset
boundary cannot carry that cache into the new scene. There is no preprocessing exclusion,
recognizer, subtitle-driven cut veto, or detector-private pause branch. The host's native USER32
interactive move/size policy may suppress the entire optional OCR/SLR observation while DAV2 and
this scene-cut pipeline continue normally; because no OCR8 record or locator dispatch is produced,
that transport-level suppression neither ages grace nor changes cut authority.

The SLR13 same-scene provisional single-line bridge is never cut or reset authority. A hard-cut
epoch change, input-domain reset, fresh onset, half-faded/transitional owner, unreliable local-plane
sample, multiline/ribbon stack, or geometry outside its generated one-baseline bounds remains the
ordinary first pending observation with exact Base. Only a distinct non-cut observation replacing
a mature full-fade single ordinary owner may render its exact same-frame OCR8 cover provisionally;
the old durable owner/generation/target/fade remain unchanged and the following distinct
observation must still confirm the ordinary handoff.

## Appearance proposals

A broad appearance proposal requires:

```text
raw_rgb_change >= 0.70
structural_change >= 0.03
```

A localized player or collage can replace much less than 70% of a desktop. The localized route is
admitted only when both endpoints have reliable structural support and all of these tests pass:

```text
raw_rgb_change >= 0.18
structural_change >= 0.03
raw_rgb_change >= appearance_baseline + 0.12
raw_rgb_change >= 3 * appearance_baseline
```

The appearance baseline uses new-value weight `0.25`. These surprise tests keep ordinary busy
motion from looking like a localized cut.

An appearance proposal is evidence, not authority. It must still be corroborated by depth before a
cut can pulse.

## Exposure veto

A broad RGB transition may be classified as exposure-like only when:

- current and previous structural support are each at least `0.01`;
- common support covers at least half of the smaller endpoint's support;
- ordinal structure is quiet, or its flips are no more than 5% of the broad RGB replacement; and
- at least 80% of changed RGB texels move in one brightness direction for the ratio-based case.

An exposure-like transition vetoes ordinary absolute and relative neural-depth authority. This is
necessary because model tone mapping or normalization can move most depth texels during a flash
even when scene geometry is unchanged.

The band between the quiet threshold and proposal threshold remains ambiguous. It does not claim
an appearance cut, but it also does not veto independently corroborated geometry.

## Structureless bridge

A flash, slate, or clipped frame may remove almost all ordinal support. On the first such update,
the detector preserves the previous reliable endpoint and withholds an ordinary cut. A continuing
structureless interval is allowed to use strong preserved depth evidence because ordinal
corroboration is impossible by construction.

When structure returns, a near-exact appearance return to the preserved endpoint is suppressed.
A photometrically or geometrically different return remains observable. This bridge is narrowly
scoped; it does not turn every low-texture frame into a cut.

## Cut state machine

The detector maintains independent geometry and appearance arms, a latched state, an optional
geometry-confirmation state, depth and appearance baselines, and scene age.
Its 32-word GPU bridge begins with the schema-6 cut-contract tag. The resolver restores generated
defaults on a tag mismatch, while telemetry, traces, and the V2 camera reject an unauthenticated
bridge instead of interpreting a same-sized foreign layout.

### Startup and rearm

- Startup has no armed branch. Both arms become ready at the first completed depth observation at
  least eight source-stream frames after initialization; that observation itself cannot pulse.
- An accepted cut clears both arms and enters the latched state.
- Geometry rearms after two consecutive updates below `0.10` depth change.
- Appearance rearms after two consecutive updates without a complete appearance proposal.
- Rearming affects the following update.

The branches rearm independently. Persistent depth motion therefore cannot permanently block a
later appearance-correlated cut, and persistent appearance activity cannot block later qualified
geometry.

### Acceptance routes

An armed appearance route accepts immediately when its complete proposal is not vetoed and depth
change is at least `0.25`.

An ordinary geometry candidate requires:

- no appearance veto;
- structural change of at least `0.005`, except for the explicitly tracked structureless bridge;
  a reference already known to be structureless also waives this otherwise impossible test; and
- either an armed absolute depth change of at least `0.60`, the same `0.60` recovery bar while
  returning from tracked low-structure history, or a qualified relative edge.

After a previous cut, the relative edge is available once scene age reaches eight source-stream
frames. Scene age uses the stream frame-id delta when depth observations are throttled or dropped;
quiet/low rearm evidence and both baseline EMAs still advance once per completed observation. It
requires depth change of at least `0.30` and either:

```text
depth_change >= depth_baseline + 0.20
```

or:

```text
depth_change >= 2 * depth_baseline
```

The depth baseline uses new-value weight `0.125` and resets on an accepted cut. A steady high value
therefore converges into its baseline and cannot pulse on a timer.

Ordinary geometry candidates use a two-update confirmation. The first update records a pending
endpoint; the next update must independently clear the geometry bar before the detector emits one
pulse. A continuing structureless bridge is already a preserved-endpoint comparison and may use
its separately guarded immediate route.

This confirmation is the protection against treating one cut's model-normalization settling as a
second cut. It also means a one-frame geometry spike without appearance support intentionally
abstains.

## State and telemetry

The live implementation reuses a fixed compatibility buffer so existing telemetry and Dump 3D
readers can observe cut evidence. Only the following values have production authority:

- confirmed cut count and one-frame pulse;
- cut flags and scene age;
- depth and appearance baselines;
- current evidence fractions and structural-support fractions; and
- analysis reason flags.

Reserved compatibility slots remain at their generated defaults and have no renderer authority.
The scene camera receives only the confirmed cut count/pulse.

The detector adds no CPU classification or synchronous GPU readback. Production telemetry uses a
nonblocking staging ring and may omit samples under load. Offline evaluation may block for a
complete trace; missing live samples must not be compared one-for-one with that trace.

## Deterministic qualification

The committed conformance clips must prove at least these contracts:

- gain, offset, gamma, clipped exposure, flashes, and strobe produce no cut;
- a broad appearance replacement without `0.25` depth corroboration does not cut;
- absolute and relative geometry values immediately below their thresholds do not cut;
- startup arming cannot pulse on the same update;
- persistent high evidence produces one cut rather than rhythmic relatching;
- quiet evidence independently rearms both branches without itself producing a cut;
- a relative geometry spike during sustained motion uses the pending-confirmation transition and
  produces exactly one pulse on its confirmed endpoint;
- a settling jump after an accepted cut cannot immediately repulse;
- a structureless flash returning to the same endpoint is suppressed, while a real different
  return remains detectable;
- localized video/player replacement passes only when it is a sharp surprise above its ordinary
  motion baseline;
- full-frame-to-ROI, ROI-to-full-frame, identity, crop-size, and transfer-domain changes clear old
  evidence instead of emitting a synthetic cut;
- a pure on-screen translation of the same-sized ROI retains the camera and cut histories while
  binding geometry to the new exact matched-frame rectangle;
- any selected client that maps exactly to the capture remains in the ordinary full-frame domain
  without a reset or camera reacquisition;
- changing an ROI between Chromium-video and foreground-client authority clears detector lineage
  even when its rectangle and dimensions match;
- a stable lower OCR line stack acquires only after two compatible observations with distinct exact
  frame/domain identities; processing one record again cannot self-confirm;
- a hard cut clears pending/grace, preserves only same-frame rectangles that still overlap an old
  owner, then either restarts from a reliable new local-plane sample at half strength or publishes
  exact Base when sampling is unreliable; disjoint or
  appended lines remain pending; and
- a no-owner cut or input reset landing on an already-visible static subtitle records the first box
  stack as pending and can acquire it on the next compatible distinct observation.

Run the production evaluator and the scene-cut unit tests after changing the evidence shader,
thresholds, state resolver, preprocessing order, or history layout:

```powershell
python tools/sbsbench/run_eval.py --comparison-only --label cut-control
python -m unittest discover -s tools/sbsbench -p "test_*.py"
```

## Galaxy XR acceptance

Exercise each sequence for at least 20 seconds and finish with five seconds of a static image:

1. Slow and fast pans, including a continuous turn.
2. Forward/backward corridor motion.
3. A moving foreground subject against a static background.
4. Fast browser scrolling and full-screen text.
5. Animation, particles, and rapid motion without an editorial cut.
6. Exposure ramps, flashes, and alternating bright/dark frames.
7. A texture-only cut with similar depth.
8. A true different-depth cut.
9. A true cut during uninterrupted persistent motion.
10. Motion that stops, rearms, then cuts again.
11. A rapid A-to-B-to-C sweep with the second cut placed one through eight valid depth updates
    after the first.
12. A structureless flash/slate that returns both to the original scene and to a different scene.
13. A localized video/player replacement inside an otherwise stable desktop.
14. Enter and leave both window-region ROI authorities, including Chromium priority and an
    unsupported-aspect fallback.
15. Move a same-sized foreground window without changing its content, then resize it across an
    analysis-domain boundary and switch between Chromium-video and foreground-client authority.
16. Enter both semantic Chromium fullscreen and generic exact-full-capture foreground-client routes;
    verify that each preserves ordinary full-frame continuity rather than creating an ROI transition.
    Repeat with desktop focus, a minimized window, a window on another monitor, and a spanning window;
    each must remain ordinary full-frame V2 without switching the capture output.
17. With no subtitle owner, land a cut and then an input-domain reset on an already-visible static
    subtitle; verify first-observation pending, second-distinct-observation acquisition, and exact
    Base on the pending frame.
18. Introduce subtitle-like lower-frame scene text during a textured pan. Verify that moving boxes
    fail the two-observation overlap transaction, then explicitly check the known false-positive
    boundary where a detector publishes a stable horizontal box for two observations.
19. Acquire one line, append a delayed translated line, and remove it again. Verify that the first
    line retains same-frame authority, the appended stack needs two observations, each line remains
    a separate dense rectangle, and empty/missed current OCR returns exact Base immediately.
20. From a mature full-fade single ordinary owner, replace it with each captured single-line
    geometry pair. Verify first-observation exact-current provisional conditioning, residual-above-8
    half fade versus residual-at-or-below-8 full fade, unchanged durable owner/generation/target,
    ordinary confirmation on the next distinct observation, and exact Base for cut/reset, half-fade,
    unreliable, malformed, IoU-equality, changed-cover duplicate, and out-of-gate cases.

Repeat representative cases in SDR and HDR and across authenticated landscape, ultrawide, and
portrait tensor shapes. Record frame identity, all evidence fractions, reason flags, arm/latch
state, pending confirmation, scene age, baselines, cut generation, camera generation, and rendered
disparity percentiles.

The live contract passes when exposure-only changes never reacquire the camera, sustained motion
never pumps it, each qualified editorial cut creates exactly one camera acquisition, a cut during
persistent motion remains observable, ROI-domain transitions cannot compare incompatible history,
pure ROI translation does not reacquire the camera, and diagnostics add no GPU queue stall.
The subtitle portion additionally requires two distinct exact-frame observations for birth and
material handoff, static cut/reset recovery on the second observation, cut survival only for
same-frame rectangles still matching the old owner, exact Base whenever neither normal nor bounded
exact-current provisional authority exists, and no geometry synthesized from owner, pending,
target, or grace state.
