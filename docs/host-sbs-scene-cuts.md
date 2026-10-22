# Host SBS scene cuts

This document is the canonical contract and acceptance plan for the Host SBS scene-cut detector.
The detector has one authority: emit a confirmed cut generation and one-frame pulse for history
and subtitle ownership. The adaptive geometry described in [Host SBS pipeline](host-sbs.md) retains
its controller across cuts. Its mean zero and robust
relative-depth amplitude continuously follow authenticated source-time observations. Confirmed
cuts retain that geometry controller and remain depth/subtitle-history reset authority. The adaptive
UI-plane filter resets on a cut or input-domain change. A true analysis-domain reset clears the
geometry controller too. Private normalized depth, history, evidence and cut state remain separate
from the curve-free geometry map.

## GPU evidence

Every valid matched active-model-input/depth pair contributes these frame fractions on the fused
composite's single high grid:

The fused migration initially retains the established numeric thresholds. Global fractions remain
dimensionless, but fixed-cell stencils and `16x16` local tiles cover half the source-space extent on
the exact-2x grid, and the refined field can expose edge motion that the coarse field did not
represent. Equivalent scene-cut and reuse behavior is therefore an evaluation requirement, not an
assumption; threshold retuning requires paired corpus and live evidence rather than compensating by
silently downsampling the active history.

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
normalized-depth comparison, appearance and ordinal evidence, adaptive zero, baselines, and history
all exclude the surrounding desktop. A causally authenticated Chromium `<video>` has priority;
otherwise a causally continuous foreground root client may provide the rectangle. The retained full
captured color texture supplies those offset analysis loads and remains the final renderer source.

Evidence from different domains must never be compared. Entering or leaving ROI mode, changing the
authority kind (`chromium_video` versus `foreground_client`), authorized identity or crop
dimensions, or changing the input transfer domain clears cut history, baselines, pending
confirmation, and the adaptive camera before analysis resumes. Translation of the same authority and
same-sized crop retains the analysis domain and its histories; the exact matched-frame rectangle
still determines where that frame is rendered. Translation nevertheless revokes any older
positioned completion or cached ROI output; only a newly retained matched frame may reuse the
histories. DDup damage-guided reuse is not a translation exception: it requires the exact positioned
route and every live authority epoch to remain unchanged. A missing, stale, unsupported, spanning,
partially off-monitor, or other-monitor ROI selects full-frame V2. That resets state when leaving an
active ROI, but repeated ineligible observations while already full-frame are not domain changes.
Any selected analysis-source rectangle that equals the complete capture is already the canonical
full-frame domain: it creates no crop, ROI transition, history reset, or camera reacquisition.
Chromium may reach that result through separate semantic fullscreen evidence; the generic
foreground-client route reaches the same canonical result through exact geometry alone. The browser helper labels its
relaxed full-client evidence `ok-fullscreen`, distinct from strict `ok` video-ROI evidence, and the
host rejects an `ok-fullscreen` subrectangle. The subtitle path observes whichever domain
V2 actually used. Any real input-domain transition resets subtitle history together with the
ordinary cut and camera histories.

A proven unchanged DDup reuse of authenticated completed content creates no scene observation. Once an
authenticated ROI completion owns the exact positioned route, a contiguous and complete sequence of dirty/move metadata may prove
that every later desktop change was wholly outside that analysis domain. Repeated delivery of one
immutable capture sequence is idempotent, while a missing or out-of-order sequence, metadata
failure, protected-content masking, or any dirty rectangle, move source, or move destination that
intersects the ROI forces the ordinary matched path. WGC supplies no such proof. On a reuse, raw
appearance, ordinal evidence, normalized-depth comparison, cut confirmation, baselines, scene
camera, OCR, and SLR do not advance, and an already-consumed cut pulse cannot be applied again.
Continuously proven unchanged content has no age or delivery-count expiry. A separate rolling clean
damage anchor preserves the transitive proof across unrelated desktop updates without moving the
real inference owner. Each unchecked interval still requires complete retained metadata, and only
a new authenticated completion can establish source continuity after dirty or unknown evidence. The exact proof and
its invalidation rules are owned by [the Host SBS pipeline](host-sbs.md).

If changed B reused analysis A and then remains static, its authenticated completion and exact
source continuity may retain A indefinitely. No final observation is scheduled merely because
capture becomes idle. B remains the completed-color identity and A remains the real depth/subtitle
owner; a later changed C still compares against A. Neither redelivery nor a later receipt creates a
new cut observation or replays A's consumed pulse.

The GPU adaptive path may omit repeated depth observations for an authenticated changed-frame
candidate when current OCR work is unavailable or ineligible. After exact unchanged reuse has been
ruled out, a genuinely admitted ordinary source with an eligible current OCR path submits the
existing force-infer request through the same joined depth/OCR root. Eligibility uses host-owned
resource, work-mode and source-geometry facts. It does not inspect OCR-band image content, boxes,
probabilities or SLR state. Otherwise the host uses route, identity and complete retained DDup
history to classify that frame as GPU-undecided; it does not apply a motion-area or rectangle-shape
heuristic. A dense GPU comparison against the exact state-1 active model-input history then applies
finite, global medium/strong, and supported-tile local bounds. That input is the fused composite's
high RGB tensor; there is no CutBridge readback or CPU decision for the current branch. The CPU may
learn a completed publication's outcome through the shared receipt described below. Reuse has no
time or hold-count expiry. Every comparison remains relative to the last real inference, with nonzero
ordered source timestamps and the same authenticated domain and dimensions. Reuse cannot replace
that owner and conceal cumulative drift.

Depth and subtitle advance or hold together inside one joined completion. Current-ready OCR work
`1`, or current ineligible abstention work `2`, publishes only on valid inference. Work `0` suppresses
subtitle observation during native USER32 move/size. Every reuse freezes the complete
OCR8/UIP4/contour/conditioned-final tuple along with depth, cut, camera and V2 Base. Suppression advances
neither OCR nor locator state and publishes Base as the atomic final field. Only a validated
completed-publication receipt can establish CPU reuse lineage; an external health snapshot cannot.

The device comparison has no OCR-band, localized subtitle-transition or SLR-state veto. The
host-owned OCR eligibility choice occurs before that comparison and changes no cut evidence.
Ordinary work consumes OCR cells only with complete proposal/request/optional-receipt proof and
otherwise publishes current abstention on valid infer. One direct post-CUDA detector finalizer
validates depth, geometry and requested subtitle authority before enabling the joint postprocess
shapes. Malformed conditional authority holds both tuples. A malformed force proposal fails depth
open to infer without authorizing stale OCR input. A CPU-known force with malformed receipt proof
publishes an exact-current OCR abstention and complete Base-derived output; retained boxes are never
relabeled. Failure to publish the complete joined tuple is terminal.

The CPU learns an outcome only from an ordered snapshot after the complete D3D history, depth and
subtitle publication. It polls that receipt without a flush or wait; missing, busy, stale or invalid
proof cannot authorize reuse. The transport and authentication contract is owned by
[the Host SBS pipeline](host-sbs.md#gpu-owned-near-identical-transaction). The completed color
publication, actual depth owner and subtitle observation remain separate identities. Reuse may
publish color B while retaining depth and subtitle owner A; the next GPU comparison still names A.
Ordinary pending work and known completed lineage replace the former persistent opaque barrier and
separate follow-up anchor. External CutBridge health snapshots remain diagnostic-only and cannot
substitute for receipt proof or feed production state.

An exact captured image is not a new observation: its matching completion or self-contained packed
pixels may be delivered again under the current route without submitting
another root. This moves neither completed-color continuity nor inference ownership and advances
no subtitle or cut state; changed cursor/presentation identity cannot use that shortcut. The
complete redelivery gates are owned by [the Host SBS pipeline](host-sbs.md). A later receipt does
not create another observation or replay a consumed cut pulse. Regressed ownership, route or damage
discontinuity, signature drift, or final attribution failure returns to a CPU-known force-infer
transaction.

An infer-gated owner record supplies the exact last accepted-infer frame to the next scene seed, so
the next observation uses C-B after B advances valid inference history and C-A after B reuses A.
That scene seed remains GPU-owned and never waits for CPU receipt consumption.
Thus a GPU-undecided sub-threshold non-exact change while OCR is unavailable or ineligible can
remain absent from CutBridge while cumulative input change against the fixed real owner stays
within the similarity thresholds. Eligible ordinary sources instead produce current joined
inference observations. Reuse never advances
scene state or reconsumes an old pulse. Depth and subtitle hold together for as long as the detector
authorizes reuse; elapsed source time and hold count never force an observation. Source timestamps
authenticate ordering and retain the actual infer interval for scene evidence. Learning the actual
completed outcome does not reset coherent GPU temporal state.
Missing detector, conditional-graph, interop, or wrapper authority is terminal and renders flat; it
never silently submits DAV2 outside the wrapper. The detailed thresholds and ownership contract are
owned by [Host SBS frame attribution and failure behavior](host-sbs.md#frame-attribution-and-failure-behavior).

There is no OCR-only DDup retained-box proof. Each accepted ordinary root requests current-ready
OCR (`1`) or current ineligible abstention (`2`). Neither path relabels an older OCR8 payload. An
armed Dump 3D root remains a complete ordinary force-infer observation. Exact unchanged DDup
deliveries submit no root, whether the current color was inferred or reused an older analysis owner.
The temporal pixel mask's four confirmations count genuine completed eligible inference
observations with distinct source frame and time, not display frames. Duplicate delivery, exact
DDup reuse, whole-tuple reuse and single-flight busy drops do not add confirmation observations.

The production subtitle path applies no overlay exclusion to cut evidence or DAV2. Subtitle
appearance and disappearance remain ordinary scene evidence; the locator consumes the
already-resolved cut result and can neither suppress nor retroactively change it.

Adaptive geometry consumes the same authenticated cut generation. It retains
continuously filtered mean zero and its model-prior-bounded P05/P95 amplitude across confirmed
cuts, without invalidating its targets or disarming timing. It initializes on the first valid
source-time observation. Its first valid observation after an interruption or gap over 250 ms
rearms without spending the missing interval; an input-domain reset reinitializes the complete
28-word geometry state to its sealed initial values.
Reuse dispatches no geometry resolver and freezes controller state with the complete published
tuple. Exact unchanged reuse, or GPU-undecided reuse while OCR is unavailable or ineligible, can
pause an unfinished transition. A newly admitted eligible ordinary source uses the current joined
inference path instead.
It is the Host relative-depth adapter described in the
[joint-plane contract](host-sbs.md#adaptive-geometry-and-subtitle-plane), not a physical calibration
of model depth or a new cut detector. Confirmed cuts continue to reset subtitle UI filter memory
and clear subtitle pixel history independently of geometry adaptation.

The detector-only PP-OCRv6/OCR8/UIP4 path consumes the authenticated durable CutBridge hard-cut
epoch. It has no private cut classifier or subtitle ownership transaction. A distinct infer
observation joins the current OCR8 record, source/domain identity and geometry; every valid
paired cover participates. A confirmed cut or input-domain change resets the independent UI
controller and native pixel history. The current observation starts pixel confirmation again;
four genuine eligible observations are required. Old support and an old UI target cannot cross
the cut. The saturating maximum epoch remains the terminal unique-generation lifetime for an
uninterrupted domain.

Duplicate observations do not add confirmations or spend controller dwell time. DAV2 reuse
dispatches no subtitle observation and preserves the entire published tuple byte-for-byte.
Current off-cover, empty or invalid authority clears positive pixel counts; raw pixel history has
no pending owner, lease or historical rectangle cache. Ordinary cover count/order/split/merge changes
do not reset still-covered pixel history by themselves. Recipe 23 compares current and prior
authenticated OCR probability over each current raw/core. A strict unit-mass spatial-distribution
change ratio above `1/8` sets a current probability cue. A qualifying same-pixel fixed RGB reference
retains its anchor and advances despite that cue. When the reference is missing or mismatched inside
a cued current raw/core, neighbor fallback is blocked and the pixel starts at one. Expanded covers
authorize search without enlarging this local fallback veto; four genuine RGB confirmations still
supply raw support. Each genuine nonzero cue independently rearms provisional protection of the
current cover union; four consecutive cue-free observations hand over to confirmed-pixel Gaussians.
This stateless witness uses the previous completed probability snapshot before overwrite, creates
no new flags on duplicates/reuse/reset or invalid/incompatible history, and leaves the independent
UI controller and durable CutBridge epoch unchanged. It creates no historical box ownership or cut.
The exact ratio, sampling, countdown, header and limitations are owned by the
[canonical subtitle policy](host-sbs.md#adaptive-geometry-and-subtitle-plane).
Missing or stale geometry clears granted UIP4 slots and copies
Base while preserving the applied controller plane/goal with timing disarmed. Invalid Base
probes preserve the current OCR union but publish inactive SCP4 parameters. These are separate
from valid empty OCR.

There is no preprocessing exclusion, recognizer, subtitle-driven cut veto or detector-private
pause branch. Native USER32 interactive move/size can suppress the entire optional subtitle
observation while DAV2 and cut detection continue normally. That suppression advances neither
pixel confirmation nor the UI controller and does not change cut authority.

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

The detector adds no CPU classification or synchronous GPU readback. Routine CutBridge telemetry
uses the optional diagnostic section of the shared nonblocking completion receipt, with independent
validity and capture-time age. It may omit samples under load. Dump 3D keeps its separate capture
lifecycle; offline evaluation may block for a complete trace. Missing live samples must not be
compared one-for-one with that trace.

## Deterministic qualification

In `flat_transition`, the source changes at frame 13 and takes the ordinary geometry route:
frame 13 records a pending endpoint, and frame 14 emits the single confirmed pulse. The
appearance-backed `scene_cut` fixture still pulses immediately at its frame-13 source change.
Fixture pulse schedules name accepted cuts, not just source edit boundaries. The generator and
committed metadata must agree with these schedules; correcting an expectation requires production
trace evidence and does not relax exact-frame scoring.

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
- any selected analysis source that maps exactly to the capture remains in the ordinary full-frame
  domain without a reset or camera reacquisition;
- changing an ROI between Chromium-video and foreground-client authority clears detector lineage
  even when its rectangle and dimensions match;
- all current admitted OCR8 covers participate in native pixel confirmation; repeating a record
  cannot supply one of the four genuine observations;
- a hard cut resets UI memory and pixel history, with provisional current-cover protection during fresh confirmation;
- a cut or input reset landing on an already-visible subtitle starts its first pixel observation;
  cover membership changes alone do not clear otherwise compatible, still-covered history.

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
    analysis-domain boundary, acquire/release a structural content child, and switch between
    Chromium-video and foreground-client authority.
16. Enter both semantic Chromium fullscreen and generic exact-full-capture foreground-client routes;
    verify that each preserves ordinary full-frame continuity rather than creating an ROI transition.
    Repeat with desktop focus, a minimized window, a window on another monitor, and a spanning window;
    each must remain ordinary full-frame V2 without switching the capture output.
17. Land a cut and an input-domain reset on visible static subtitles. Verify that support waits
    for four genuine eligible pixel observations and cannot reuse pre-reset confirmation.
18. Introduce moving scene text inside admitted OCR covers and test static non-subtitle text too.
    Moving RGB must not inherit mature counts; stable non-subtitle pixels inside valid covers
    remain a declared false-positive boundary of this generic method.
19. Append and remove a second line, reorder or split OCR boxes, and repeat an observation.
    Still-covered qualifying same-pixel fixed RGB references retain history even with a probability cue;
    newly admitted pixels start at1, and empty/invalid
    current authority returns exact Base. Duplicate identities do not discover new cover pixels.
20. Replace one real subtitle cue with another, including similarly colored glyphs and a genuine
    disappearance. Inspect native support, Gaussian coverage and the actual stereo glyph edges;
    partial-color matches and static backgrounds must not be presented as full semantic accuracy.
21. Exercise the current-core probability witness on same-size, expanded, shrunk and multiline
    changes, harmless detector noise and genuinely changed background/context. Verify strict
    `T > 1/8`, no cue when both probability masses are zero, a cue when exactly one is
    zero and local current slot flags. Verify that a qualifying same-pixel fixed RGB reference
    advances or holds despite a cue. When that reference is missing or mismatched, a cued
    current raw/core must restart at one without borrowing from the two-pixel neighborhood.
    Verify retained expanded-margin history and later uncued fallback borrowing too; a cue
    does not impose four new observations on every retained reference. Invalid current/prior P, stale/incompatible SCT5 identity and duplicate observations
    cannot manufacture flags. The cue leaves pixel history outside cued current raw/cores and
    independent UI state unchanged, but can rearm the shared current-cover provisional countdown.
    Measure missed changes, false protection and RGB restart delays on genuine sources instead of fitting the fixed
    experimental threshold to selected output frames.

Repeat representative cases in SDR and HDR and across authenticated landscape, ultrawide, and
portrait tensor shapes. Record frame identity, all evidence fractions, reason flags, arm/latch
state, pending confirmation, scene age, baselines, cut generation, camera generation, and rendered
disparity percentiles.

The live contract passes when exposure-only changes never reacquire the camera, sustained motion
never pumps it, each qualified editorial cut creates exactly one authenticated epoch transition, a cut during
persistent motion remains observable, ROI-domain transitions cannot compare incompatible history,
pure ROI translation does not reacquire the camera, and diagnostics add no GPU queue stall.
The subtitle portion requires four genuine eligible observations for native pixel confirmation,
strict cut/domain resets, all admitted current OCR8 covers, independent source-time UI filtering,
and exact Base whenever conditioning is unavailable. No cover geometry is synthesized from
historical ownership or controller state. Real cue changes, disappearance, edge quality and
static-background ambiguity remain necessary quality evidence beyond contract tests.

The current provisional OCR warmup is presentation state, independent of the four RGB confirmations. Every eligible strictly newer frame/time observation with a nonzero current probability-reset mask rearms protection of only current covers to four, even during an active countdown. Each genuine zero-mask observation decrements it; the fourth consecutive cue-free observation hands over to confirmed-pixel Gaussian support. Persistent or repeated cues can prolong provisional protection indefinitely; exact duplicates and full-tuple reuse hold it. Empty/invalid authority clears it, and cut/domain resets start a new window only with valid current nonempty evidence. Raw support still requires four confirmations. A genuine absent-caption frame with a current false OCR box can receive cue-held provisional protection, which must be measured separately from stale-history leakage.
