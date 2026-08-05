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
  return remains detectable; and
- localized video/player replacement passes only when it is a sharp surprise above its ordinary
  motion baseline.

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

Repeat representative cases in SDR and HDR and across authenticated landscape, ultrawide, and
portrait tensor shapes. Record frame identity, all evidence fractions, reason flags, arm/latch
state, pending confirmation, scene age, baselines, cut generation, camera generation, and rendered
disparity percentiles.

The live contract passes when exposure-only changes never reacquire the camera, sustained motion
never pumps it, each qualified editorial cut creates exactly one camera acquisition, a cut during
persistent motion remains observable, and diagnostics add no GPU queue stall.
