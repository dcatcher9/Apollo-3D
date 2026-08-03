# Host SBS V2 Crown Artifact Investigation Handoff

> **Superseded historical record.** The schema-14/minorant continuation instructions in this
> document are no longer actionable. Schema 15 implements the reviewed 75/25 vertical share plus
> row majorant and is tracked in
> [host-sbs-v2-crown-compromise-trial.md](host-sbs-v2-crown-compromise-trial.md). Use that document
> and the current source as authority; do not restore the dirty A/B or the old commit named below.

Status: investigation record, not a production design decision. The direct vertical-minorant
replacement tested here is a disposable A/B and is **not ready to commit or ship**.

Date: 2026-08-03

Scope: live Windows Host SBS V2 only. Do not change Client SBS or offline conversion as part of
this investigation.

## Executive summary

The remaining artifact is a translucent, bent, or duplicated-looking band around the top of a
strong near-foreground silhouette, most visibly the crown of dark hair in the fullscreen Bilibili
witness. The committed V2 renderer does not synthesize hidden pixels and does not use an owner or
hole-fill pass. The artifact is produced by the geometry field that makes a large foreground cliff
safe for the single-valued inverse warp.

The shipped policy uses two one-sided spatial envelopes:

1. a vertical least Lipschitz **majorant**, which never lowers requested near disparity; then
2. a horizontal least Lipschitz **majorant**, which keeps both eye maps contractive.

At the crown, preserving the high-disparity hair while limiting the vertical step forces the
vertical majorant to raise several rows of background above the hair. The inverse warp then bends
those real background samples toward the foreground. This is the visible crown collar.

The experiment replaced only the vertical majorant with a vertical greatest Lipschitz
**minorant**, leaving the horizontal majorant and 12-step inverse unchanged. On the supplied dump,
this reduced the average measured collar by 66.5%, but moved the transition into the top of the
hair. The top eight foreground rows lost 6.72 one-eye source pixels on average and as much as
35.67 pixels locally. The visual result has less translucent background collar but a more
stepped/slanted or compressed crown. The worst collar tail improved only slightly.

This establishes the core tradeoff; it does not solve it:

```text
preserve foreground disparity  -> deform visible background above the crown
preserve the background        -> compress/deform the top of the foreground
preserve both rigidly          -> expose missing background, requiring fill/temporal data/
                                  hallucination or a different representation
```

Recommendation: do not ship either a blind vertical majorant-to-minorant swap or a semantic
crown detector. Continue with exact direct-field experiments that seek a simple, bounded
compromise, and qualify any candidate on more than this one scene.

## Important working-tree state

The repository base at the time of this handoff is:

```text
branch: master
HEAD:   b60a43de refactor(host-sbs): simplify live v2 analysis
```

There is an uncommitted three-file live A/B:

```text
 M src/generated/depth_coordinate_v2_contract.h
 M src_assets/windows/assets/shaders/directx/depth_coordinate_v2_vertical_limit_cs.hlsl
 M tools/sbsbench/contracts/depth-coordinate-v2-v1.json
```

The shader changes the vertical recurrence from `max(previous - step, candidate)` to
`min(previous + step, candidate)`. The other two files contain only regenerated producer-closure
and derived canonical-contract hashes. A RelWithDebInfo `sunshine.exe` was built successfully and
the modified shader passed the direct shader-compilation unit test.

This dirty state is intentionally **not contract-complete**. The production schema/tag, C++ names,
dump schema, evaluator fields, tests, and documentation still declare a vertical majorant and the
ordering invariant `final >= vertical >= candidate`. After the trial the true relations are only:

```text
vertical_minorant <= candidate
final >= vertical_minorant
final has no fixed pointwise ordering relative to candidate
```

Do not commit the three-file A/B as-is. If the minorant is rejected, restore these three files. If
some descendant is accepted, perform a real semantic schema migration rather than merely updating
the shader-closure hash.

## Symptom and user observations

The relevant observations accumulated across the Host SBS V2 work are:

- Fullscreen video can show more hair-edge and crown artifacts than the same video inside a
  browser window, even though fullscreen gives DAV2 more pixels for the video.
- Earlier owner/fill warp variants duplicated hair or separated the hair from the body.
- The horizontal row majorant substantially improved lateral hair continuity and removed that
  explicit duplication/step.
- A transparent or bent band remained around the top of the head. It can also appear at the top of
  other strong foreground objects.
- Reducing pop to 0.5 visibly weakens the effect but also weakens the desired 3D. The supplied
  witness was captured at pop 1.0 and still contains the crown problem, so this is not only a
  pop-2.0 overload.
- More inverse iterations are not a plausible fix: the current 12-step contraction has a
  conservative residual below 0.04 source pixel at 3840 width.

The upstream framing/model investigation is documented separately in
[dav2-framing-depth-audit-handoff.md](dav2-framing-depth-audit-handoff.md). It established that the
fullscreen DAV2 raw field already has a stronger, more cutout-like hair/background relationship.
The spatial limiter does not create that raw cliff, but it decides where the unavoidable spatial
transition is paid in the rendered image.

## Exact supplied witness

Input dump:

```text
E:\ApolloDev\sbs_dump\dump_20260802_213620_45424_18c83197c0a08c2c_1
```

Key captured properties:

| Property | Value |
|---|---:|
| Source / eye | 3840 x 2160 |
| Packed SBS | 7680 x 2160 |
| Color | linear scRGB HDR capture |
| DAV2 field | 770 x 434 |
| Requested pop | 1.0 |
| Effective one-eye gain | 0.00375 source-U |
| Container scale | 1.0 |
| Vertical limit | `2.0 / 770` source-U per depth row |
| Horizontal limit | `0.5 / 770` source-U per depth column |
| Candidate range | -0.0005624813 to 0.0146859121 source-U |

The raw model provenance and captured raw hash validate. However, this dump predates the
then-current schema-14 producer closure and 12-word state, which has since been superseded by the
schema-15 C75 follow-up in [host-sbs-v2-crown-compromise-trial.md](host-sbs-v2-crown-compromise-trial.md).
Its captured producer used the older contract/tag and 8-word state. Treat it as an authenticated
historical candidate/final-field witness for exact direct replay, not as proof that either later
production schema emits identical fields.

This distinction does not weaken the limiter-only A/B: the test starts from the exact captured
`shadow_candidate_parallax.f32`, so it preserves the captured model output, scene calibration,
curve, pop, and container while changing only the two-dimensional spatial projection.

## Current production geometry

The authoritative shipped design is described in
[host-sbs-depth-coordinate-v2.md](host-sbs-depth-coordinate-v2.md) and
[host-sbs-depth-system-rethink.md](host-sbs-depth-system-rethink.md). In condensed form:

```mermaid
flowchart LR
    R["Authenticated DAV2 raw depth"] --> C["Fixed coordinate and monotone curve"]
    C --> P["Candidate parallax p"]
    P --> V["Vertical shear-2 majorant v"]
    V --> Q["Horizontal slope-0.5 majorant q"]
    Q --> I["12-step unique inverse"]
    I --> S["SBS eyes sampled from real source color"]
```

The candidate `p` is signed one-eye displacement in normalized source-U. The committed vertical
pass computes:

```text
v+(x,y) = max_s [p(x,s) - Lv * abs(y-s)]
Lv       = 2.0 / depth_width
```

It is the least column-wise `Lv`-Lipschitz field satisfying `v+ >= p`.

The horizontal pass then computes:

```text
q+(x,y) = max_s [v+(s,y) - Lh * abs(x-s)]
Lh       = 0.5 / depth_width
```

The horizontal slope is strictly below one, so each eye has one inverse:

```text
source_x = destination_x + eye_sign * q+(source_x, y)
```

There is no internal visibility owner, multi-root choice, synthetic hole fill, or copied hidden
background in live V2. The renderer samples a real source coordinate after 12 fixed-point
iterations. Therefore the present crown is not the old synthetic-fill duplication bug.

## Why the committed vertical majorant creates the crown collar

At depth-grid column `x = 381` (approximately source `x = 1900`), the supplied witness contains a
large positive near-foreground onset. Values below are one-eye horizontal displacement in source
pixels:

| Depth row | Approx. source y | Candidate | Committed final majorant | Trial final minorant |
|---:|---:|---:|---:|---:|
| 138 | 686.82 | -2.12 | -2.12 | -2.12 |
| 139 | 691.80 | -2.12 | 3.63 | -2.12 |
| 140 | 696.77 | -2.13 | 13.60 | -2.13 |
| 141 | 701.75 | -2.12 | 23.58 | -2.12 |
| 142 | 706.73 | -2.12 | 33.55 | -2.12 |
| 143 | 711.71 | 43.52 | 43.52 | 7.85 |
| 144 | 716.68 | 44.49 | 44.49 | 17.82 |
| 145 | 721.66 | 44.94 | 44.94 | 27.80 |
| 146 | 726.64 | 45.35 | 45.35 | 37.77 |
| 147 | 731.61 | 45.65 | 45.65 | 45.65 |

The candidate attempts to jump by roughly 45.6 one-eye pixels in one depth row. The selected
vertical bound permits about 9.97 source pixels per depth row. Because a majorant is forbidden to
lower the hair, it creates the required four-row ramp in the background above it. That raised
background is the geometric collar.

The horizontal majorant is still needed for the contractive eye maps. It preserves the vertical
bound but can spread raised support sideways within the same rows. This is one reason the collar is
not purely a one-column effect and why changing only the vertical pass cannot be judged from a
single scalar maximum.

The curved crown makes that interaction especially important. At one row, neighboring columns can
already be high/foreground while the current column is still low/background. Even if the vertical
minorant keeps every column's own pre-onset rows low, the subsequent horizontal majorant spreads
the nearby high values sideways to satisfy the slope-0.5 contract. That explains the measured
combination of a large improvement in average/support but only a small improvement in the worst
collar tail. This pure vertical-minorant change did not remove this curved crown collar while the
row pass remained unchanged; a vertical-only change cannot guarantee removal in that situation.

Only the horizontal bound is required for the unique contractive inverse. The vertical bound is an
artifact-shaping policy. That makes a no-vertical-pass replay a valid and important diagnostic,
provided the unchanged horizontal projection remains in place.

The apparent translucency is consistent with adjacent background rows being pulled toward the
foreground by different amounts, then linearly sampled at different coordinates in the two eyes.
For this captured source and exact renderer, the field-to-image causal path is established; the
perceptual labels "transparent" and "duplicated" remain visual descriptions rather than an alpha-
compositing mechanism. Exact reproduction of the original HDR color path is not claimed.

### Why it is most obvious at the top

The displacement is horizontal, but the strongest unsupported transition here is across vertical
rows. The column majorant must extend the near support upward across the nearly horizontal crown
edge. At lateral hair edges, the row majorant shapes the transition along the same axis as the
stereo displacement and previously improved the obvious hair/body discontinuity. Bottom edges can
also be conditioned, but their contrast, occlusion, and neighboring subject structure make this
particular upward background ramp much easier to see. This orientation explanation is supported by
the field, but it should be tested on other top, bottom, and diagonal silhouettes before being
generalized.

## Vertical-minorant experiment

The trial replaces only the vertical envelope:

```text
v-(x,y) = min_s [p(x,s) + Lv * abs(y-s)]
```

This is the greatest column-wise `Lv`-Lipschitz field satisfying `v- <= p`. The exact two-pass
float32 recurrence is:

```python
vertical = candidate.copy()
for y in range(1, height):
    vertical[y, :] = minimum(candidate[y, :], vertical[y - 1, :] + 2.0 / width)
for y in range(height - 2, -1, -1):
    vertical[y, :] = minimum(vertical[y, :], vertical[y + 1, :] + 2.0 / width)
```

The existing horizontal **majorant** is then applied unchanged:

```python
final = vertical.copy()
for x in range(1, width):
    final[:, x] = maximum(final[:, x], final[:, x - 1] - 0.5 / width)
for x in range(width - 2, -1, -1):
    final[:, x] = maximum(final[:, x], final[:, x + 1] - 0.5 / width)
```

This moves the vertical ramp from the low/background side into the high/foreground side. It does
not remove the need for a ramp. The final row majorant may raise some values after the vertical
minorant, so the final field is neither globally a majorant nor globally a minorant of the
candidate.

The max/min swap has the same scan count, memory traffic, dispatch count, and asymptotic cost as
production. A material performance difference is not expected, but it was not isolated or timed;
the one-frame harness is dominated by TensorRT startup and is not valid performance evidence.

## Exact replay method

The ordinary `tools/sbsbench/replay_depth_mapping_v2.py` command is **not** the right tool for this
question. It recomputes shot state from raw depth and its Python path still applies the committed
vertical majorant. The native whole-sequence replay also requires a complete authenticated clip
contract and would reconstruct state.

The clean limiter-only method used here was:

1. Baseline field: captured `shadow_final_parallax.f32`.
2. Trial input: captured `shadow_candidate_parallax.f32`.
3. Apply the float32 vertical minorant and unchanged horizontal majorant above.
4. Encode both through `tools.sbsbench.depth_mapping_v2.encode_direct_parallax()`.
5. Preserve captured `shadow_coordinate.f32` as diagnostic `order_00001.f32`.
6. Render each through Apollo's real D3D direct-parallax harness and 12-step inverse.

The harness command shape was:

```powershell
$env:SUNSHINE_SBS_BENCH = "1"
./sunshine.exe E:/Git/Repo/Apollo-3D/tools/sbsbench/bench.conf --sbs-bench `
  --frames <case>/frames `
  --out <case>/harness `
  --model depth_anything_v2_fp16 `
  --direct-parallax-root <case>/direct
```

Both resulting direct-geometry contracts validated. Maximum encode/decode disagreement with the
intended field was `0.0000092` one-eye source pixel. Replaying the captured baseline against the
dump's SDR diagnostic source preview achieved 51.79 dB PSNR versus the dump's HDR-derived SBS
preview. Geometry is exact for the captured fields; exact HDR color fidelity is not claimed because
tone mapping and linear filtering do not commute perfectly.

Exact float32 field hashes:

| Field | SHA-256 |
|---|---|
| Captured candidate | `21e874829da5f705b0c5bce5da442d62cf1b7d4c87a9499a16ffc57793290203` |
| Captured majorant final | `a247d30adf6ab55f4e5764879fcd228c5b2e7b29e04b7e93f20bf6ffdc53f99c` |
| Vertical minorant intermediate | `c790b00efa660e22015472e00631a952e9e2ce8ca16584273098a0046f16b0d6` |
| Minorant then horizontal-majorant final | `4fc469b6e65cf5c266263834b25f65447d0f6719b6fe6bc51fce16be757dddbc` |

## Quantitative result

Global field comparison against the captured final majorant:

| Measurement | Result |
|---|---:|
| Changed texels (`abs(delta) > 1e-8`) | 1.7347% |
| Lowered texels | 1.7347% |
| Raised texels | 0% |
| Mean absolute change | 0.097 one-eye source pixel |
| Maximum local lowering | 35.67 one-eye source pixels |
| Maximum-change location | depth `(381,143)`, source approximately `(1900,712)` |
| Final global peak retention | 100% in this witness |
| Horizontal scaled adjacent bound | 0.50000004 |
| Vertical scaled adjacent bound | 2.00000014 |

Global peak preservation is empirical for this frame, not a property of a minorant. An isolated
near peak can be lowered.

Relative to the original candidate, rather than relative to the captured final, the trial final
raises about 3.20% of texels and lowers about 0.25%. The captured all-majorant final raises about
3.45% and lowers none. Thus the unchanged horizontal pass still reconstructs broad raised support
after the vertical minorant. Describing the trial as globally "lowering only" is correct only for
its delta from this captured majorant baseline, not for its relation to the candidate.

The scene-specific crown diagnostic selected 60 central crown columns with a stable low side and a
persistent candidate onset. It is deliberately not a general detector:

| Crown measurement | Majorant baseline | Minorant trial |
|---|---:|---:|
| Pre-crown collar mean | 16.63 px | 5.56 px |
| Pre-crown collar P95 | 40.35 px | 31.89 px |
| Pre-crown collar maximum | 43.19 px | 39.78 px |
| Pre-crown positive support | 65.63% | 29.17% |
| First 8 foreground rows mean loss | 0 px | 6.72 px |
| First 8 foreground rows P95 loss | 0 px | 29.30 px |
| First 8 foreground rows maximum loss | 0 px | 35.67 px |
| First 30 foreground rows mean retention | about 100% | 96.22% |
| First 30 foreground rows minimum retention | 100% | 93.87% |
| First 30 foreground rows peak retention | 100% | 100% |

The average collar fell 66.5% and its positive support fell 55.6%, but the worst tail fell only
7.9%. The minorant therefore makes a visible average improvement without eliminating the hardest
part of the crown.

## Visual result

Local A/B image:

![Captured majorant versus vertical-minorant trial](E:/ApolloDev/crown-minorant-dump-trial-20260803/dump_20260802_213620/crown_baseline_vs_minorant.png)

The trial reduces the soft raised band above much of the hair. It also makes the top silhouette
more angular and moves the transition inside the foreground. The left and right eye silhouettes
do not become a clean rigid match; they express the tapered top differently. The result is a useful
diagnostic and a partial improvement, not a finished correction.

The blinking A/B is at:

```text
E:\ApolloDev\crown-minorant-dump-trial-20260803\dump_20260802_213620\
  crown_baseline_minorant_blink.gif
```

## Earlier supporting evidence

### Raw DAV2 framing effect

[dav2-framing-depth-audit-handoff.md](dav2-framing-depth-audit-handoff.md) independently proved that
the fullscreen raw DAV2 output is more cutout-like than the windowed output in this content. The
raw hair/background step is stronger and within-hair relief is relatively weaker. Synthetic
surround and simple preprocessing did not safely reproduce the browser-context benefit. Do not
train or preprocess against one browser surround based on this witness.

### Row-majorant work

Historical exact row-majorant evidence is under:

```text
E:\ApolloDev\majorant-row-both-confirm-20260802
```

It removed the explicit duplicated hair and hair/body step while preserving foreground candidate
disparity. The residual crown bending motivated the committed vertical shear-2 pass. This current
experiment is a **column-wise vertical minorant**, not the older rejected row-wise minorant
described in [sbs-3d-roadmap.md](sbs-3d-roadmap.md).

### Archived support-taper trial

The same vertical-minorant idea was previously replayed in Python under:

```text
E:\ApolloDev\crown-fix-trials-20260802\support-taper
```

At shear 2, both historical fullscreen and windowed witnesses retained the global peak and stable
interior, while the fullscreen maximum adjacent output-raster-row change in inverse source-x
displacement fell from 14.24 to 4.05 pixels. The contemporaneous visual conclusion was nevertheless
that the translucent crown became a stepped/slanted silhouette that differed between the eyes. The
new exact D3D replay corroborates that trade instead of overturning it.

### Other alternatives already explored

- Increasing the horizontal slope limit from 0.5 toward 0.65/0.75/0.85 narrows the lateral crown
  support, but makes the current 12-step fixed-point residual grow rapidly. At the largest tested
  value it reached roughly 9 source pixels. That is not a safe small calibration change.
- Peak-anchored/global range compression reduces the cliff but shifts far and median geometry
  toward positive disparity, damages convergence/zero-plane meaning, and can flatten the useful
  binocular range. It treats a local silhouette problem with a global geometry change.
- Stronger vertical limiting makes the transition smoother but wider; weaker limiting makes it
  narrower but steeper. Neither removes the missing-background trade.
- Extra inverse iterations solve the selected field more accurately but do not change where the
  spatial projection placed its ramp.

Auditable archived evidence for those experiments is under:

```text
E:\ApolloDev\crown-fix-trials-20260802\short-collar\REPORT.md
E:\ApolloDev\crown-fix-trials-20260802\anchor-compress\metrics.json
E:\ApolloDev\crown-fix-trials-20260802\vertical-envelope\metrics.json
```

## What is established

1. The strong silhouette relationship is already present in the DAV2 raw/candidate geometry; it
   is not created by the spatial limiter or by a synthetic fill stage. This dump does not by
   itself exclude authenticated HDR preprocessing from influencing the model prediction.
2. The committed vertical majorant deterministically raises background rows above this crown to
   satisfy the vertical bound without lowering the foreground.
3. Those raised rows reach the exact live inverse-warp position field.
4. A vertical minorant deterministically removes much of that raised background by lowering the
   first foreground rows instead.
5. The minorant preserves the configured slope bounds and the unique inverse in this witness.
6. The horizontal majorant remains a source of residual spatial spreading after the vertical
   minorant.
7. At pop 1.0, the trade remains large enough to be visible. It is not solely a pop-2.0 problem.
8. Twelve inverse iterations already solve the chosen field far more accurately than the visible
   error. Extra iterations will not repair the field's spatial policy.

## What is not established

1. This one historical frame is not schema-14/15, whole-clip, temporal, or release-gate evidence.
2. The scene-specific crown metric is not a detector and its manually bounded search window must
   not become production logic.
3. It is not known whether headset perception prefers the reduced collar over the compressed top
   across other scenes, pop values, displays, or motion.
4. It is not known how much of the residual minorant result is contributed by the unchanged
   horizontal majorant versus bilinear field/color sampling.
5. It is not known whether a symmetric bounded projection, a different separable pass order, or a
   genuinely two-dimensional objective provides a better simple compromise.
6. Runtime cost has not been isolated. The min/max swap should be comparable, but no performance
   claim is supported by the one-frame harness.
7. Peak preservation, subject-volume preservation, and temporal stability are not guaranteed by a
   vertical minorant.

## Fundamental design constraint

For a single monocular color image and a single-valued horizontal inverse warp, a large foreground
translation reveals background that the source frame does not contain. The implementation must
choose at least one compromise:

- bend visible background into a collar;
- reduce/deform foreground disparity near the silhouette;
- copy or hallucinate hidden background;
- use temporal information from other frames;
- or change the representation/rendering model.

V2 deliberately rejected hidden-background synthesis and temporal ownership in order to remain
deterministic, low-latency, and attributable. A deeper fix should state explicitly which cost it
accepts. No threshold or semantic classifier can make the missing source pixels exist.

## Recommended next investigation

Keep the next pass small and falsifiable.

1. **Stage isolation on this exact dump.** Render the vertical-minorant intermediate before the
   horizontal pass as a diagnostic only, then compare it with the final minorant-plus-horizontal-
   majorant output. This quantifies how much residual crown the horizontal pass recreates. Do not
   ship a field that violates the horizontal contraction bound.
2. **Test a symmetric bounded compromise.** Construct upper and lower Lipschitz envelopes and test
   a small number of mathematically bounded blends or a least-change projection. Avoid RGB masks,
   ROI tracking, crown classifiers, and per-scene thresholds. The question is whether sharing the
   ramp between foreground and background looks better than assigning all cost to either side.
3. **Use the exact D3D direct-field harness.** Do not recompute DAV2 or scene calibration while
   comparing spatial operators. Preserve the same candidate and source bytes.
4. **Separate projection from sampling.** Replay the same final fields with point versus current
   linear parallax sampling, and use an analytic textured foreground/background silhouette with
   exact depth. This distinguishes projection policy, low-resolution interpolation, and upstream
   DAV2 segmentation.
5. **Expand the witness set before live integration.** Capture a fresh committed-baseline
   schema-15 / manifest-schema-9 dump. Then include top, bottom, vertical, diagonal, thin, and
   multiple-object silhouettes; fullscreen and windowed video; HDR and SDR; and pop
   0.5/1.0/1.5/2.0. Do not make production conclusions from only the older 8-word-state witness.
6. **Then test temporal behavior.** A spatial result that looks good on one frame can crawl,
   breathe, or switch sides as the candidate edge moves by one depth texel.
7. **Only then build a live A/B.** Preserve flat failure, authenticated producer/renderer closures,
   the 4% container, and the fixed pop semantics.

Do not start by changing inverse iterations, adding another global adaptive-pop controller,
raising depth resolution, training against the browser surround, or adding a semantic crown/ROI
detector. Existing evidence does not identify those as the limiting mechanism.

## Qualification criteria for any replacement

At minimum, report all of the following separately:

- background collar mean, P95, maximum, and spatial support;
- first-foreground-row disparity loss and 30-row subject-volume retention;
- per-eye crown alignment and focus in the headset;
- horizontal slope and vertical shear bounds;
- fixed-point residual and source-boundary behavior;
- percentage of the full field raised and lowered;
- global and isolated near-object peak retention;
- temporal stability through sub-texel edge motion and scene cuts;
- HDR and SDR output;
- GPU producer and renderer timing;
- failure/authentication behavior;
- confirmation that Client SBS and offline conversion remain unchanged.

A scalar average improvement is insufficient. This experiment improved the average collar while
leaving a large worst-case tail and introducing foreground onset loss.

## Local evidence index

Exact requested-dump replay:

```text
E:\ApolloDev\crown-minorant-dump-trial-20260803\dump_20260802_213620
```

| Artifact | Purpose |
|---|---|
| `crown_baseline_vs_minorant.png` | labeled per-eye visual A/B |
| `crown_baseline_minorant_blink.gif` | animated visual A/B |
| `crown_difference_x8.png` | amplified rendered RGB difference |
| `metrics.json` | exact field hashes, bounds, global delta, and crown-column profile |
| `crown_metrics.json` | explicitly scene-specific collar/foreground trade metric |
| `image_metrics.json` | image delta and baseline replay fidelity |
| `replay_validation.json` | direct-geometry contract and encode/decode validation |
| `baseline/harness/sbs_00001.png` | captured-final-majorant direct replay |
| `minorant/harness/sbs_00001.png` | vertical-minorant-plus-row-majorant direct replay |

Other evidence:

```text
E:\ApolloDev\crown-fix-trials-20260802\support-taper
E:\ApolloDev\majorant-row-both-confirm-20260802
E:\ApolloDev\sbs_dump\dump_20260802_213620_45424_18c83197c0a08c2c_1
```

These are developer-machine artifacts, not committed baselines.

## Relevant source files

| File | Role |
|---|---|
| `src_assets/windows/assets/shaders/directx/depth_coordinate_v2_map_cs.hlsl` | candidate parallax generation |
| `src_assets/windows/assets/shaders/directx/depth_coordinate_v2_vertical_limit_cs.hlsl` | historical majorant/minorant experiment; current schema-15 upper/lower share producer |
| `src_assets/windows/assets/shaders/directx/depth_coordinate_v2_limit_cs.hlsl` | unchanged horizontal row majorant |
| `src_assets/windows/assets/shaders/directx/sbs_reprojection_v2_live_ps.hlsl` | live 12-step unique inverse |
| `src/sbs_bench_depth_coordinate_v2.cpp` | native exact V2 sequence replay and majorant metrics |
| `src/sbs_bench_harness.cpp` | direct final-parallax D3D replay |
| `tools/sbsbench/depth_mapping_v2.py` | NumPy reference; now follows the schema-15 conditioned share |
| `tools/sbsbench/direct_geometry_contract.py` | authenticated direct-field replay contract |
| `tools/sbsbench/depth_coordinate_v2_dump_contract.py` | strict schema-15 dump semantic and renderer-source validation |

## Migration debt if a descendant is accepted

A production adoption is substantially wider than the three-file live trial:

1. Choose a neutral semantic name such as `vertical_conditioned` unless the selected operator is
   provably a majorant or minorant.
2. Bump or explicitly revise the algorithm schema/tag; do not retain a tag whose documented
   invariant is false.
3. Update runtime resource/member names and comments in `video_depth_estimator.*`,
   `sbs_debug_dump.*`, and the horizontal shader.
4. Replace native replay's majorant-only `raised_fraction` and no-lowering checks with metrics and
   invariants appropriate to the chosen operator.
5. Update the NumPy reference, temporal replay, direct GPU tests, dump validator, and dump schema.
   The current Python dump validator checks stale descriptive strings but does not independently
   prove the array ordering, so merely rehashing the trial can produce semantically false metadata.
6. Regenerate the authenticated contract/header and all affected baselines only after semantics
   and tests agree.
7. Update the two authoritative V2 design documents; until then they correctly describe committed
   production, not this experiment.

## Related documentation

- [Host SBS Depth Coordinate V2](host-sbs-depth-coordinate-v2.md): authoritative shipped
  coordinate, majorant, inverse, dump, and hair-evidence contract.
- [Host SBS Depth-System Rethink](host-sbs-depth-system-rethink.md): why V2 deliberately assigns
  the missing-background cost to a visible background collar.
- [DAV2 Framing/Depth Audit Handoff](dav2-framing-depth-audit-handoff.md): upstream fullscreen versus
  windowed raw-model investigation.
- [SBS 3D Roadmap](sbs-3d-roadmap.md): earlier owner/fill, row-minorant, and row-majorant decisions.
- [SBS benchmark README](../tools/sbsbench/README.md): exact replay and provenance requirements.
- [Warp simulator README](../tools/warpsim/README.md): the old CPU replica is removed and must not
  replace the exact D3D direct-field replay or `run_eval.py` suite qualification.

## Continuation status

This investigation was continued and superseded by
[host-sbs-v2-crown-compromise-trial.md](host-sbs-v2-crown-compromise-trial.md). Do not start from
the historical commit or preserve the discarded dirty minorant A/B. Keep this file only as the
rejected-endpoint measurement record.
