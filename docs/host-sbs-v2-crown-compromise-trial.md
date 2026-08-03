# Host SBS V2 Crown Artifact: Bounded-Compromise Trial Results

Status: investigation record plus a live headset A/B implementation. The original trial was
offline-only; the follow-up at the end of this document records the subsequent schema-15 C75
geometry migration, live collar filter, and regression tests. The result is still awaiting visual
and worst-case 4K90 qualification before it becomes a locked production decision.

Date: 2026-08-03. Continues [host-sbs-v2-crown-artifact-handoff.md](host-sbs-v2-crown-artifact-handoff.md).

## Executive summary

This trial executed the handoff's recommended next investigation on the exact supplied witness
(`dump_20260802_213620_45424_18c83197c0a08c2c_1`) using bitwise-exact float32 replicas of the
production envelope recurrences, validated by hash against the captured GPU intermediates, and
rendered candidates through the real D3D direct-parallax harness.

Two root-cause refinements and one verified candidate family came out of it:

1. **The horizontal majorant recreates the vertical-minorant trial's residual collar on this
   witness.**
   The vertical-minorant intermediate has exactly zero collar in the crown window. The entire
   residual (mean 5.56 px, worst tail 39.78 px) is added back by the horizontal majorant
   spreading laterally from neighboring columns whose crown onset is higher up. Consequence:
   the tested constant vertical-envelope shares have limited leverage over the worst tail while
   the horizontal pass remains a pure majorant. This is an observation about this operator family,
   not a proof covering every possible vertical policy. In the shipped baseline the horizontal
   majorant also nearly doubles the collar (vertical-majorant-only mean 9.35 px -> final 16.63 px).
2. **The lateral cliff is the same cost paid on either side.** The columns where the horizontal
   majorant creates the worst collar (crown shoulders, late onset) are exactly the columns where
   a horizontal minorant component costs the most subject volume. The curved-crown geometry, not
   the vertical policy, owns the worst case.
3. **Verified candidate family: fixed convex blend of the two 2D Lipschitz envelopes.**
   `final = w * U + (1 - w) * L` with `U = h_majorant(v_majorant(p))` (the shipped field) and
   `L = h_minorant(v_minorant(p))`. For every `w` in [0, 1] the blend provably satisfies both
   production bounds (each envelope is Lipschitz in both axes; a convex combination preserves
   the bounds), keeps the unique contractive inverse, is deterministic and content-blind, is
   temporally non-expansive (each envelope is an L-infinity non-expansive, translation-equivariant
   map, and the weight is constant, so there is no detector-style switching mechanism), and costs
   one extra scan-pass pair plus a lerp. Within this fixed convex-envelope family, `w = 0.5`
   minimizes the maximum distance to the two endpoints: its deviation from either endpoint is
   exactly half the envelope gap. It is not claimed to be the global optimum over every bounded
   spatial conditioner.
   **Same-day revision:** visual comparison at the hair/shoulder junction showed the global
   blend's lateral foreground lowering produces a silhouette notch on lateral hair edges; the
   recommended operator narrowed to the orientation-selective variant (vertical-pass sharing
   only, horizontal pass kept a pure majorant) — see the follow-up section and revised
   recommendation below.

## Quantitative result on the exact witness (crown window of the handoff)

Same 60-column crown metric as the handoff (selection rule reverse-engineered and reproduced to
the last digit: onset = first row with candidate >= 20 px whose preceding 8 rows are all
<= 8 px, searched in depth x [300,469], y [130,159]).

| Operator | Collar mean | Collar P95 | Collar max | FG-8 loss mean | FG-8 loss max | FG-30 retention min | Global peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| A shipped `h+(v+)` | 16.63 | 40.35 | 43.19 | ~0 | 0 | 100% | 100% |
| B minorant trial `h+(v-)` | 5.56 | 31.89 | 39.78 | 6.72 | 35.67 | 93.9% | 100% |
| C `h+((v+ + v-)/2)` | 9.54 | 31.89 | 39.78 | 4.00 | 18.18 | 96.9% | 100% |
| **D w=0.65 blend of U,L** | 10.79 | 26.23 | 28.07 | 8.72 | 15.12 | 80.9% | 100% |
| **D w=0.5 midpoint of U,L** | 8.29 | 20.18 | 21.60 | 12.55 | 21.60 | 72.7% | 100% |

(px = one-eye source pixels at 3840 width; full table incl. w = 0.55/0.75/0.85 and diagnostics in
`E:\ApolloDev\crown-symmetric-trial-20260803\candidate_metrics.json` / `dweight_metrics.json`.)

Notable: the pure vertical minorant (B, the dirty working-tree A/B) is **dominated** by the
midpoint blend on both worst-case extremes (collar max 39.78 vs 21.60; FG loss max 35.67 vs
21.60). Its only win is collar mean. The trade knob `w` moves monotonically between the
shipped baseline (w=1) and the full lower envelope (w=0): raising `w` trades collar tail for
subject volume, with the FG-30 retention cost concentrated at the crown's lateral shoulders
(w=0.5 keeps ~97% at the crown center but only ~73% at late-onset shoulder columns).

## Visual result (real D3D harness renders, exact captured candidate)

All three rendered candidates applied their intended fields exactly (manifest SHA match, decode
error 1e-5 px). On the witness's thin vertical window-frame line above the crown:

- A (shipped): the line kinks strongly around the crown and differently per eye — the
  inter-ocular disagreement that reads as the translucent collar.
- D w=0.65 and D w=0.5: the line is nearly straight and the two eyes agree; the crown silhouette
  is slightly rounded/compressed at the apex, far milder than the minorant's stepped crown.
- C: roughly half of A's bend, silhouette essentially intact — the conservative option, but its
  worst collar tail is unchanged from B (horizontal-majorant recreation).

Evidence: `E:\ApolloDev\crown-symmetric-trial-20260803\` — crown region: `crown_all_variants.png`,
`crown_zoom_all.png`, `crown_zoom_v2.png` (all seven operators incl. alpha=0.25/0.75);
shoulder region: `shoulder_junction_4x.png`, `shoulder_left_zoom.png`, `shoulder_right_zoom.png`;
blink A/Bs: `blink_*.gif` (crown, shoulderL/R, junction); per-case harness outputs and logs,
`render_prep.json`, lowering heat maps `dmid_vs_shipped_*_heat.png`.

## Cross-scene qualification (field space, 10 captured witnesses)

Applied to every schema-compatible captured candidate under `E:\ApolloDev\sbs_dump\dump_202608*`
(4 capture sessions, cliffs from 29 to 100 px): both scaled bounds hold within recorded float32
tolerance for the blends
on all witnesses; the w=0.5 exact-halving of the worst raise is universal; global peak retention
is 100% on 8/10 and 97.4–97.7% on two (the isolated-near-peak lowering the handoff predicted for
any minorant component, bounded at `(1-w)` strength). Numbers in `cross_dump_qualification.json`.

## Follow-up same day: shoulder-junction comparison demotes the global blend

User headset observation: the global blend produces a visible discontinuity on the hair edge
around the shoulder. Reproduced offline and root-caused. Spatial footprint of each operator's
deviation from the shipped field on the witness:

| Operator | Texels changed vs shipped | Texels lowered vs candidate | Where the lowering lives |
|---|---:|---:|---|
| C `h+(v_shared)` | 1.7% | 0.25% | only the first rows at the top of vertical cliffs |
| D global blend w=0.5 | 7.0% | 3.6% | a band along **every** silhouette, including lateral hair/body edges |

The 4x renders at the left hair/shoulder junction (`shoulder_junction_4x.png`,
`shoulder_left_zoom.png`, `shoulder_right_zoom.png`, `blink_junction_*.gif`) show:

- A (shipped) and **C are visually indistinguishable at this junction** — smooth continuous hair
  edge in both eyes. C removes the specific global-blend mechanism that produced the notch: its
  horizontal pass never lowers the vertically conditioned field. The vertical share can still
  change diagonal or lateral junctions, so this witness is evidence rather than a universal proof.
- D w=0.65 shows an angular notch in the hair silhouette just above the shoulder; D w=0.5 shows a
  stronger notch plus smearing of the upper hair edge. Cause: the horizontal-minorant term lowers
  hair-edge disparity laterally (slope-bounded, ~2.5 px per depth column), so the edge band
  renders at a different offset than the hair interior — a step inside the most salient content.

Orientation analysis (why the crown is the hard case and lateral edges are not):

- At a lateral (vertically running) edge, the disparity cliff lies along the displacement axis; a
  monotone slope-<1 horizontal map renders it as background compressing behind the edge — the
  appearance of real occlusion. The ramp hides inside a perceptually legitimate phenomenon.
- At a top (horizontally running) edge, the cliff is orthogonal to the displacement, so adjacent
  rows shear horizontally. There is no occlusion interpretation for shear: straight background
  lines bend, oppositely per eye, producing rivalry — the translucent collar. The warp has no
  degree of freedom that can hide a vertical-direction cliff; it can only shear background
  (collar) or foreground (stepped crown).
- The configured Lipschitz ramp makes collar and foreground compression a coupled trade. The
  precise lower bound also depends on the allowed slope and the distance available for the ramp;
  the shorthand `collar + foreground_loss >= cliff gap` is not a general per-texel identity.

A bounded-shave composite `max(U - B, blend)` was also tested (provably bound-safe, caps all
lowering at B): it caps total deviation from the shipped field, which undoes the vertical sharing
beyond B — collar mean regresses (9.5 -> 13.6 at B=5 px). Rejected.

## Creative extensions probed (same day)

**Diagonal cliffs.** A diagonal silhouette splits its cliff into an x-component (hides as
occlusion — legitimate) and a y-component (must become shear). The crown "shoulders" are the
diagonal case and own the irreducible tail. Two structural facts close off the field-space
avenues: (1) the shipped `U` is the least majorant for the configured slope cone, so adding
diagonal scan passes to a majorant can only raise it (more collar), and (2) the slope-0.5 row
contract ties every collar texel to its same-row foreground neighbor, so any tail reduction
mathematically forces lateral foreground lowering. Measured confirmation via a two-axis
asymmetric share `h_blend_beta(v_blend_alpha)` (alpha=0.5): beta=0.92 improves the worst collar
only 39.8 -> 37.4 px while doubling lateral-lowering support; meaningful tail reduction needs
beta values that re-enter shoulder-notch territory. C is the selected Pareto point among the
constant envelope-share policies tested here, not a field-space optimum over untested operators.

**Historical offline collar-defocus prototype.** The positive deviation map
`max(final - candidate, 0)` is a deterministic, detector-free byproduct of the conditioner. It
marks raised geometry, not semantic background, and can therefore overlap foreground at some
boundaries. Modulating the sampling blur radius by delta (zero below ~4 px deviation, saturating ~6 px
sigma by ~20 px) renders the residual bent band as slightly out-of-focus background instead of a
sharp rivalrous line. Offline prototype on the C render (`crown_defocus_demo.png`,
`sbs_C_collar_defocus.png`, `blink_crown_C_vs_Cdefocus.gif`): the residual kink in the
window-frame line dissolves into the scene's existing bokeh; the hair silhouette stays sharp;
blur support is 3.3% of pixels. Live cost would be a mip-bias/multi-tap in the reprojection PS
plus binding the candidate alongside the final field. Its original generator was not persisted,
so the later live A/B uses the explicitly specified fixed-sigma/opacity interpretation documented
below rather than claiming this variable-radius wording is exact. Caveats to qualify: slight interocular
blur mismatch (far weaker rivalry than position mismatch), sharp-textured backgrounds, temporal
behavior of the blur mask (the final field is Lipschitz-bounded but the subtracted candidate is
not temporally or spatially bounded), HDR.

**True removal (out of current constraints).** Only new information removes the artifact:
temporal background accumulation behind moving subjects or an occlusion-layer representation
(Philips Declipse precedent), or learned inpainting. Documented as the exit ramp if bounded
compromises prove insufficient in the headset.

## Design context (not validation evidence)

A non-exhaustive survey of DIBR/2D-to-3D practice (Fehn; Zhang & Tam asymmetric depth smoothing; Vazquez/Tam
background-only extrapolation; Jantet joint projection filling; Lang et al. saliency-steered
warping; MSU VQMT3D defect taxonomy; iw3 `--edge-dilation`; VisionDepth3D edge-band shift
suppression; Google Cinematic Photos boundary pull-forward) supports:

- background-side ramps are a common default because they preserve salient foreground geometry;
- foreground compression is normally kept narrow and bounded when used;
- the *translucent* (vs merely bent) look of a collar is characteristically a foreground
  edge-ownership problem (monocular depth edges under-reaching into hair). Shipped systems fix it
  by dilating foreground disparity slightly before ramping. This is an orthogonal, deterministic,
  content-blind follow-up not tested here.

## Recommendation (revised after the shoulder-junction comparison)

Adopt the **orientation-selective share**: `final = h_majorant(alpha * v_majorant(p) +
(1 - alpha) * v_minorant(p))` — the C family. Sharing is confined to the vertical pass (which
shapes top edges, where the collar lives and where foreground compression proved visually mild);
the horizontal pass stays a pure majorant, avoiding the measured global-horizontal-minorant notch
mechanism. This does not prove that every diagonal or lateral junction is unchanged. `alpha` is
the single policy knob:

| alpha (majorant share) | Collar mean | Collar P95 | FG top-rows loss max | FG-30 retention min |
|---:|---:|---:|---:|---:|
| 1.0 (shipped) | 16.63 | 40.35 | 0 | 100% |
| 0.75 | 12.50 | 34.16 | 9.1 | 98.4% |
| 0.5 | 9.54 | 31.89 | 18.2 | 96.9% |
| 0.25 | 7.34 | 31.89 | 27.2 | 95.4% |

All three non-shipped alphas are rendered and harness-verified. **User preference (2026-08-03):
foreground preservation ranks above background — alpha = 0.75 is the leading candidate** (max
foreground compression 9.1 px, retention 98.4%), ideally paired with the collar-defocus
extension below, which hides the larger residual bend that alpha = 0.75 leaves; confirm 0.75 vs
0.5 in the headset before locking in. The accepted cost, stated per the
handoff's fundamental-constraint requirement: collar mean falls from 16.63 to 12.50 px and P95
from 40.35 to 34.16 px, while most of the lateral-recreation tail is knowingly retained and the top
few foreground rows are compressed by up to the table's bound. The global blend of 2D envelopes
(D family) remains documented as the least-max-change reference point, but is **not recommended
for production**: its lateral foreground lowering creates a visible hair/shoulder silhouette
notch (confirmed in the 4x renders at w=0.65 and w=0.5), spending error on exactly the salient
edge the MSU defect taxonomy and JND literature say is most protected.

The semantic schema migration described by the original handoff has now been implemented. The
broader witness expansion (top/bottom/diagonal/thin silhouettes, windowed + fullscreen, HDR + SDR,
pop 0.5–2.0) and temporal replay through sub-texel edge motion and cuts remain qualification work;
they were not prerequisites completed before this explicitly experimental live trial.

## Reproduction

All scripts are in `E:\ApolloDev\crown-symmetric-trial-20260803\` (`fields.py` validates the
float32 reimplementation bitwise against the captured GPU fields; `evaluate.py` scores the
operator family; `prep_render.py` + the `--sbs-bench --direct-parallax-root` harness render
candidates; `make_visuals.py` builds the A/Bs). The crown-metric selection rule reproduces the
handoff's published numbers exactly, so all tables are directly comparable.

## Live A/B implementation follow-up

The subsequent Host SBS working tree implements the recommended `alpha = 0.75` orientation-
selective field as depth-coordinate contract schema 15 and records its vertical upper/share
intermediates in Dump 3D manifest schema 8. It also implements a deliberately explicit live
interpretation of the prototype's underspecified collar defocus:

```text
source    = inverse(final, destination)
raised_px = max(final(source) - candidate(source), 0) * source_color_width
amount    = smoothstep(4, 20, raised_px)
color     = lerp(one_tap_color, fixed_sigma6_3x3_binomial_color, amount)
```

This is not claimed to be pixel-identical to `sbs_C_collar_defocus.png`: the prototype generator
was not persisted, that image used `alpha = 0.5`, and its exact kernel/mask coordinates cannot be
recovered from an 8-bit PNG. The live choice is positive-only to protect the foreground rows that
C75 lowers, deterministic between eyes, and color-only; it cannot alter final parallax, inverse
geometry, or comfort hashes. Its 4/20/6 values are calibrated in pixels of the current source-color
raster. That makes the mask independent of depth-grid resolution, but not invariant to changing
the stream resolution. Headset appearance and worst-case 4K90 pixel-stage cost remain qualification
items.

## Current implementation status (2026-08-03)

The live trial implements `alpha = 0.75` as depth-coordinate contract schema 15. The column pass
publishes the upper envelope and the conditioned fixed share; the lower envelope is transient and
overwritten. The row pass consumes only the conditioned share and remains a pure horizontal
majorant. Candidate is bound separately at `t3` for the color-only collar mask. The plain dump
metadata, JSON manifest, Python validator, source tests, and authenticated shader closures describe
the same policy.

### Geometry evidence

- The captured float32 C75 field is
  `E:\ApolloDev\crown-symmetric-trial-20260803\field_C_v75_hmaj.f32`, SHA-256
  `5a77102833ddd8db2a3ec9b31d3ffcc3fca1f2fd9febc8db170ac7905e95d376`.
  The rejected global `D w=0.65` field is the different `f793a218...` artifact.
- The float64 NumPy semantic oracle differs from the captured float32 field by at most
  `5.8115e-6` one-eye source pixel. Bit-exact evaluation order is owned by the executable float32
  CPU/GPU oracle, not by NumPy.
- The pure row majorant prevents the specific horizontal-minorant shoulder-notch mechanism. It
  does not prove that the vertical share leaves every diagonal or lateral junction unchanged.
- The vertical minorant recurrence is required to construct the 75/25 share. Only obsolete paths
  that select the pure-minorant endpoint may be removed.

### Collar filter policy

The active branch uses a separable 3x3 `[1,2,1]` binomial kernel. Taps are placed at
`sqrt(2) * 6` source pixels, giving a per-axis second moment of `sigma^2`; it performs nine total
color samples. A denser 5x5 experiment raised the active branch to 25 color samples; the user
reported little visible difference in `defocus_kernel_ab.png`, so it was removed. That subjective
one-crop observation does not establish general equivalence, and the unpersisted post-process image
is not treated as production-shader validation.

Every warped output pixel reads final and candidate once to form the mask. Pixels at or below the
four-pixel onset then take the original one-color-sample path; active pixels perform eight extra
color reads. The mask and filter operate on native source values, including linear scRGB HDR,
without clamping, tone mapping, or gamma conversion.

### Precision and authentication

Contract shares may be ordinary decimal values. The NumPy oracle explicitly rounds the majorant
coefficient and its complement to float32, matching the coefficients the shader consumes; the GPU
oracle still owns bit-exact arithmetic. Both the live-renderer and dump-diagnostic transitive
source closures are pinned and independently recomputed by native tests, so changing the included
live shader cannot silently leave Dump 3D unauthenticated.

## What the current tests do and do not prove

The native geometry tests execute production column/row shaders, compare exact float32 fields,
check the envelope sandwich and both slope bounds, and exercise every authenticated tensor shape.
The live-renderer test uses a prime-sized non-periodic source, an independent CPU implementation of
the exact 3x3 linear-clamp kernel, negative and greater-than-one color values, onset/intermediate/
saturated masks, negative-correction bypass, image boundaries, both eyes, and a nonzero inverse
warp. This avoids the former six-pixel-period/short-texture blind spot.

The default `tools/sbsbench/run_eval.py` gate intentionally sets `parallax_v2_render = false`; it
is the legacy comparison pipeline. Updating its baselines would not validate C75 or the live collar
filter. The manifest-driven direct-parallax harness validates V2 geometry but compiles a different
pixel shader and does not bind the live candidate mask. Therefore neither path should be cited as
live collar-filter evidence. A dedicated authenticated live-V2 clip path is still needed before
clip baselines can cover this renderer.

## Remaining qualification

1. **Headset trial.** Inspect the crown, shoulder junction, thin high-contrast lines, text, HDR,
   and scene transitions. Alpha and collar enablement are authenticated constants, not runtime
   toggles; a controlled A/B requires separately built and recorded shader-closure hashes.
2. **4K90 cost.** Compare otherwise identical warmed runs using `sbs_warp_gpu`. Record p50, p95,
   maximum, rendered/received frame counts, and DWM responsiveness. The provisional acceptance
   target is no more than 0.10 ms added p95 and no measurable increase in dropped-frame rate.
3. **Clip coverage.** Add an authenticated live-V2 renderer mode plus a crown/edge witness before
   publishing new V2 baselines. Do not rebaseline the default legacy evaluator as evidence for
   this work.
4. **Generalization.** Expand field and visual evidence to top/bottom/diagonal/thin silhouettes,
   windowed and fullscreen framing, HDR and SDR, pop 0.5–2.0, sub-texel motion, and cuts.

Evidence should be reproduced with commands and captured closure hashes. Session scratch logs are
not durable evidence and are intentionally not cited here.

## Proposed next experiments (2026-08-03, unimplemented — offline-first)

Four candidate directions, ranked. All must go through the exact-replay offline discipline of
this document (Python reference on captured dumps, crown metrics, direct-parallax harness render)
BEFORE any production or schema change, and none may bypass the qualification gates above.

### P1. Scene-latched empty-gap compression (attacks the cliff magnitude itself)

Measured motivation: the crown witness's canonical histogram is bimodal with a nearly empty
middle. With `u = (raw - mean(raw)) / 0.5` and a 400-bin histogram over the full range, the
interval `u in [0.41, 3.49]` (span 3.09) holds 0.075% of pixels on
`dump_20260802_213620_45424_18c83197c0a08c2c_1`; on `dump_20260802_110510_..._1` (the ~100 px
cliff scene) the empty span is 4.57 with 0.198% occupancy. In curve units the empty interval is
roughly 85% of the crown cliff. Every downstream cost — collar, lateral-recreation tail,
foreground compression — is proportional to the cliff, so a monotone remap that compresses only
the empty interval (e.g., retaining 30% of its span, hard-capped) would cut the witness cliff
from ~45.6 to ~19 one-eye px before any spatial conditioning. No pixel changes depth order and
intra-object relief is untouched; only the unoccupied subject/background separation compresses.
Secondary effect to measure: at pop 2.0 the empty gap consumes 4%-container budget, so
compressing it should RELAX the container and restore relief in occupied depth.

Design constraints for a trial: piecewise-linear monotone C0 remap of `u` inserted between the
curve and the gain stages; parameters latched once per shot from the same acquisition frame as
the near shoulder (the `depth_hist` evidence already exists on GPU); reset on confirmed cut;
compress only intervals with occupancy below a margin-guarded threshold; bounded maximum
compression so a mid-shot object entering the gap is displaced by a bounded, recoverable amount.
This is the same evidence class and lifecycle as the shipped near-tail shoulder — depth
occupancy, not RGB or semantics — but it IS a new content adaptation and must be justified
against the handoff's no-new-controllers guidance by exactly this offline evidence.

Falsification plan: regenerate candidates from all 10 schema-compatible dumps' `raw_depth.f32`
through the NumPy reference with the extra stage; recompute crown metrics (this document's
tables); verify container relaxation; render the witness through the direct-parallax harness;
check the multi-modal and near-empty-gap dumps for misplacement; then temporal replay across
cuts. Kill criteria: any depth-order inversion, container interaction that raises far/mid
geometry, or visible subject "flattening toward background" in the render.

### P2. Re-tune the vertical shear bound under the collar filter

The handoff rejected weaker vertical limiting as "narrower but steeper". The collar filter
changes that trade: defocus hides steepness but cannot shrink the corrupted area. Test
`max_vertical_shear` 2.0 -> 3.0 / 4.0 combined with C75 and the filter: the collar band drops
from ~4-5 to ~2-3 depth rows while the blur absorbs the added sharpness. One-constant offline
A/B in the existing exact-replay setup (the vertical step is a parameter of the Python
reference). Interaction to watch: the collar-defocus onset/full thresholds are calibrated in
source pixels against today's ramp heights.

### P3. Vertically anisotropic collar defocus kernel

The collar artifact is differential horizontal shift between adjacent rows — a
vertical-frequency error. An isotropic kernel spends half its taps on the harmless axis. A
vertical-dominant kernel (e.g., 1-D 5-tap vertical binomial) attacks inter-row disagreement
directly, preserves horizontal background texture, and uses fewer color reads than the current
3x3. This matches the classical subjective preference for vertically-dominant smoothing at
depth discontinuities (Zhang & Tam). Any kernel change must update the live-renderer closure
pin, the `collar_defocus.kernel` dump-contract strings, and their tests together (see the
Collar filter policy section).

### P4. Bounded foreground dilation before the envelopes (edge ownership)

The literature-standard fix for the translucent (vs merely bent) collar component: dilate
high-disparity texels by 1-2 depth texels (morphological max) before the envelope passes, so
mixed hair/background pixels ride with the head and the ramp starts in clean background. Shipped
precedent: iw3 `--edge-dilation`, Google Cinematic Photos boundary pull-forward. Deterministic
and content-blind; composes with C75 and with P1. Cost: subject silhouette carries ~5-10 source
px of background at near depth (classic halo). Offline test identical in shape to the C-family
trials.

Considered and NOT recommended: cross-eye collar color averaging (flattens collar background to
screen depth), per-eye asymmetric conditioned fields (breaks the mirrored-field invariant,
rivalry risk), and any RGB/saliency-guided ramp placement (detector class, rejected by the
handoff). P1 and P2 can share one offline experiment since both act upstream of the envelopes.
