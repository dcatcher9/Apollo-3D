# Host SBS Depth-System Rethink

Status: design record for the shipped Host SBS V2 cutover and schema-23 ownership-refined C75
geometry. V2 is the
sole live Host SBS renderer. The experimental collar color filter was rejected after a live
fullscreen hand-boundary halo regression; Client SBS and offline conversion remain unchanged.

The implementation details and exact equations are in
[host-sbs-depth-coordinate-v2.md](host-sbs-depth-coordinate-v2.md).

## Goal

The legacy path accumulated several overlapping geometry owners: endpoint normalization, temporal
range state, subject stretch/recenter, zero-plane choices, adaptive pop, and a complex inverse
warp. Those mechanisms can each be locally reasonable while their composition makes a depth cliff
or pop change hard to attribute.

V2 starts from a smaller rule set:

```text
fixed authenticated raw scale
  + one monotone asymmetric curve
  + one fixed logarithmic near branch
  + one conservative stage boundary acquired once per confirmed shot
  + one scene-latched selected center and zero plane
  + one requested pop
  + one frame-local hard container
  + one conservative full-resolution foreground-ownership correction
  + one orientation-selective anisotropic slope conditioner
  + one unique contractive inverse
  + one native-color sample at the inverse-warped coordinate
```

Every adaptation must name the constraint it owns. Ambiguous content is allowed to remain
unadapted rather than introducing another classifier or feedback loop.

## Ownership decisions

| Mechanism | V2 disposition | Reason |
|---|---|---|
| P2/P98 stretch and min/max EMA | Remove | Endpoint occupancy changes scale and can flatten near regions. |
| First-frame standard-deviation scale | Remove | It is content normalization and retains framing dependence. Standard deviation remains collapse evidence only. |
| Near saturation/clamp | Remove | It discards the closest relief with the strongest visual impact. |
| Subject stretch/recenter | Remove | It combines semantic inference, scale, and convergence. |
| Per-frame adaptive zero | Remove | It translates the whole stereo field and can make the screen plane wobble. |
| Scene convergence | Fix at exactly zero | The selected center is the zero plane; a second translation is redundant and cannot improve relative relief. |
| Three-class stage boundary | Keep with abstention | A deep upper histogram valley can separate the dominant subject from a nearer hand; shallow or downward candidates preserve the mean/zero camera. |
| Edge-risk adaptive-pop boost | Remove | A difficult local edge should be repaired locally, not flatten the frame. |
| Shot-latched near-occupancy shoulder | Remove | Browser surround changed the counted area and made matching windowed/fullscreen content select different geometry. Use fixed `tau_near = 2`. |
| User pop | Keep | V2 takes `cfg.pop_strength` literally. |
| Permanent within-shot safety minimum | Remove | A transient prediction must not weaken a long scene indefinitely. |
| Frame-local 4% container | Keep | This owns a precise representation bound and recovers immediately. |
| Temporal depth EMA | Do not port initially | Add temporal filtering only after raw-coordinate flicker is measured. |
| Cut pulse/generation | Keep behind a cut-only path | V2 consumes only one scene-reset input. Dedicated evidence/resolve shaders publish it without compiling or dispatching legacy subject, adaptive-pop, zero-plane, or warp analysis. |
| Old depth with current color | Remove | It breaks frame attribution. |
| Immutable candidate field | Keep as evidence only | It records requested geometry before local safety intervention; it is not bound to the live renderer and never selects geometry. |
| Conservative foreground ownership | Keep before slope conditioning | A unique, stable candidate cliff may pull one mixed far-side cell fractionally toward its existing near plateau when the matched full-resolution color has one corroborating contour. It never lowers, creates an edge, paints color, or guesses when evidence is ambiguous. |
| Canonical coordinate | Keep as diagnostic | It helps explain model output but does not select visibility. |
| Orientation-selective conditioned field | Keep as live position authority | A fixed 75/25 share of vertical upper/lower shear-2 envelopes reduces top-edge collar while preserving foreground; a pure slope-0.5 row majorant avoids lateral silhouette notches and keeps both eye maps invertible. |
| Positive-correction collar defocus | Removed | The 4/20/6 source-pixel filter introduced a visible translucent/soft halo around a reaching hand. Live rendering returned to one native-color sample while retaining C75 geometry. |
| Forward owner, multi-root visibility, synthetic fill | Remove | The slope-limited final field has a unique inverse; these mechanisms caused the measured bands. |
| Adaptive ROI | Keep removed | It adds inference, tracking, shape churn, and boundary policy. |
| Area/HDR-aware model preprocessing | Keep | It preserves input signal and belongs to authenticated model provenance. |

## Coordinate and convergence

The selected Host live calibration uses a fixed model/preprocess/shape scale:

```text
u           = (raw - gauge_center_raw) / coordinate_scale
curve       = F(u)
candidate p = requested_gain * container * curve
```

`gauge_center_raw` is acquired from the first usable depth field at startup or after a confirmed
cut. That center remains fixed through all later valid, invalid, and fast-motion frames until the
next confirmed cut; there is no pending or age-based replacement.
`coordinate_scale` is fixed by the authenticated calibration. Serialized convergence is exactly
zero and is an integrity invariant, not another controller.
Acquisition reuses the existing raw histogram and accepts an upper three-class Otsu valley only
when its smoothed valley is at most 75% of the weaker adjacent peak and `T - scale` is above the
arithmetic mean. The unchanged `T - scale > mean` expression remains only the conservative
acceptance guard. Accepted scenes use `center = T`; all other scenes use the arithmetic mean. In
both cases convergence is zero, so the selected center is the exact raw zero-disparity plane.
Acceptance changes the center without a separate translation, rescaling, or curve-shape change.

Schema 23 authenticates the standard tensor family: `770x434`, `1022x434`, `1036x434`,
`434x770`, `434x1022`, and `434x1036`. DAV2 Base, custom models, and custom tensor shapes have no
live calibration and fail flat rather than borrowing the Small-model scale.

There is no live gauge tracker yet. Per-frame mean changes cannot distinguish model gauge drift
from camera motion or changed occupancy. Any tracker would need controlled registered evidence and
source-time semantics before it could own screen-plane motion.

## Local cliff policy

After the exact source-U container, the candidate can still under-own a mixed foreground boundary
cell and contain a steep silhouette cliff. A precision-first full-resolution source-contour pass
first produces raise-only `o = ownership(p, source)`, with every ambiguous case returning `p`
exactly. The live field then uses an orientation-selective vertical share followed by a pure row
majorant:

```text
o  = conservative_ownership(p, matched_source)
V+ = vertical_majorant(o)
V- = vertical_minorant(o)
c  = 0.75 V+ + 0.25 V-
q  = horizontal_majorant(c)
|q(x+1,y) - q(x,y)| <= 0.5 / depth_width
|q(x,y+1) - q(x,y)| <= 2.0 / depth_width
```

Ownership relocates only corroborated foreground coverage; it does not synthesize hidden
background, eliminate the no-fill ramp, or replace the slope contract. Top/bottom scans compute
exact shear-2 upper and lower envelopes; their constant convex share keeps
the vertical bound and may raise background or lower only the top foreground rows. Left/right
scans then compute the exact row majorant `q` of that share. Keeping the horizontal pass one-sided
avoids the lateral hair/shoulder notch measured with a global 2D envelope blend. The passes keep
both eye maps one-to-one while accepting a bounded background/foreground compromise at top edges.
The rejected forward-owner experiments instead copied hair into synthetic fill, producing the
duplicated silhouette.

The final slope is below one, so each eye has a unique inverse. Live and exact dump mapping use 12
fixed-point iterations. Canonical coordinate, multiple-root selection, forward coverage, owner
textures, and internal hole filling have no live V2 role. The V2 dump mask attributes only inverse
samples beyond the finite source boundary that live rendering clamps to the nearest edge column.

## Adaptive responsibilities

### Pop

- Input: one user-requested `cfg.pop_strength`.
- Global reduction: only the exact current-frame 4% source-U container.
- Local response: the orientation-selective conditioner enforces the slope contract.
- Forbidden: content boost, edge-density feedback, permanent shot minimum, or stacked scale
  normalizers.

Ordinary safe frames therefore use the requested pop. Limiter burden is diagnostic evidence; it
does not feed another controller.

### Near shoulder

- Use fixed `tau_near = 2` for every authenticated frame.
- Keep the fixed raw scale, zero convergence, and configured pop unchanged by content occupancy.
- Size the hard container from the same curve that is rendered.
- Do not add ROI/content classification, RGB semantics, occupancy reduction, or hard near clamps.

### Zero plane

- Acquire the scene camera on the first usable frame with the conservative three-class selector.
- On a confirmed cut, invalidate the old camera and let the first usable new-shot depth field
  establish its replacement immediately.
- Hold that center through every later valid, invalid, and fast-motion frame until the next cut.
- Do not maintain a pending camera or apply a delayed correction.
- Make the selected center the exact raw zero-disparity plane.

Fallback and accepted selections both use exactly zero convergence. The Otsu branch can move the
selected center; an accepted valley is zero and `u = 1` sits one fixed raw scale above it. It
cannot add a whole-scene translation. No semantic subject detector or continuous zero-plane
tracker is claimed.

### Scene cuts

- V2 consumes only `{generation, pulse}`.
- Pulse handles the exact update; generation recovers a pulse missed while inference was skipped.
- Persistent-motion behavior belongs to the detector; V2 adds no second cooldown or timeout.
- The live estimator obtains this pair from dedicated cut-only evidence and resolve shaders. Their
  normalized depth is private detector evidence; no subject, anchor, adaptive-pop, or legacy warp
  output is produced for live V2.

### Invalid depth

- Unusable current depth publishes no new per-pixel geometry.
- Without a cut, retain only the small scene camera.
- On an unusable confirmed-cut frame, clear the camera.
- Never pair stale depth with current color.

A short hold of the entire already-matched SBS output may be considered separately, bounded by
source time and exposed as stale. That is presentation policy, not coordinate reuse.

## State machine

```text
NO_CAMERA -- usable field -----------------------> CAMERA + LIVE
CAMERA   -- usable, no confirmed cut ------------> CAMERA + LIVE
CAMERA   -- unusable, no confirmed cut ----------> CAMERA + FLAT
ANY      -- confirmed cut + usable --------------> CAMERA + LIVE (new camera)
ANY      -- confirmed cut + unusable ------------> NO_CAMERA + FLAT
```

Persistent state contains scene center, fixed scale, zero convergence, cut generation, calibration
revision, camera integrity, and reserved zero words. Frame validity and container scale are
current-dispatch values, not controller history.

## Runtime boundaries

| Runtime | Shared concept | Runtime-specific owner |
|---|---|---|
| Host live V2 | sign, curve, selected-center zero plane, container, final conditioned row-majorant | first-usable acquisition, cut invalidation, no late correction or lookahead |
| Host offline | potential future coordinate semantics | finalized scene boundary, lookahead, encoder and HDR policy |
| Client DAV2 | semantic interface only | Android model bytes, preprocessing, tensor shape, renderer and display calibration |
| Client MiDaS | semantic interface only | independent sign, scale, curve, validity, slope, and performance calibration |

Similar labels do not make legacy Client SBS values wire-compatible with V2. Client and offline
remain unchanged by the Host live cutover.

## Performance shape

The producer uses moments reduction, frame resolve, state resolve, coordinate/candidate mapping,
one matched-source ownership dispatch, one column-wise upper/lower/share dispatch, and one row-wise
majorant dispatch. Acquisition reuses
the existing raw min/max and 256-bin histogram by taking a GPU snapshot and merging adjacent bins;
it adds no second tensor scan or CPU readback. The source SRV accepted for inference is retained
until the exact asynchronous completion runs ownership, without a color copy. Columns and rows
are serial only inside their owning GPU threads; independent columns/rows execute in parallel and
no CPU readback or GPU/CPU synchronization is introduced.

The renderer samples the final field through a 12-step fixed-point inverse, then takes one native
linear color sample. Candidate evidence is no longer bound to the live pixel shader. This removes
the former mask-depth reads and up to eight extra color taps without changing geometry. Exact dump
mapping remains diagnostics-only and uses the same authenticated state and inverse.

The production parallax interval measures exactly the seven V2 coordinate compute passes. The dedicated
cut-only analysis runs before that interval and must be reported separately when measuring live
detector overhead. Historical shadow-mode or legacy-analysis totals are not production V2 cost.

## Provenance and evidence

Depth-coordinate contract schema 23 binds algorithm constants, model identity, preprocessing
closure, the six standard tensor shapes, the reused raw min/max/histogram roots, and the ordered
seven-pass coordinate producer inside a nine-root authenticated closure. The live renderer has its own
authenticated main/mapping/mask closure. New Dump 3D manifest schema 10 records final parallax as
position authority, the ownership result, vertical upper/share fields, candidate/canonical diagnostics, and the
disabled collar-defocus policy.

The original fullscreen and windowed hair dumps use depth-coordinate schema 7. They remain useful
historical input witnesses for exact replay but cannot authenticate schema-23 output. The prior
slope-0.5 row-majorant replay removes the duplicated hair and hair/body step while never lowering
the candidate. It raises 3.837% of fullscreen texels and 2.152% of windowed texels; its residual
fullscreen crown bending motivated the selected orientation-selective pass. Fresh schema-23,
state-schema-13/manifest-schema-10 live dumps remain required.

## Production cutover and continuing gates

The live cutover removes V1 renderer selection, prewarms the production V2 shaders, and fails flat
when production authentication or execution fails. It does not authorize unsupported models or
tensor shapes and it does not migrate Client SBS or offline conversion.

Continuing release evidence should:

1. Generate fresh schema-23 native and state-schema-13/manifest-schema-10 live evidence with exact producer and renderer closures.
2. Replay core and extended clips at the fixed scale and the current reference pop of `1.0`.
3. Inspect near-hair cliffs, thin structures, flat pages, HDR, cuts, flashes, fast motion, invalid
   inference, and long browser/video sessions.
4. Report container attenuation, candidate-to-final raising and lowering, near-core retention,
   final slope, fixed-point residual, disparity bounds, boundary integrity, and timing independently.
5. Prove V2 takes `pop_strength` literally, ignores legacy adaptive geometry controls, uses the
   same fixed near curve, and latches the stage selector only at acquisition/cuts.
6. Keep the extracted cut-only epoch contract and its persistent-motion tests authenticated as the detector evolves.
7. Calibrate every additional model, tensor shape, client backend, and offline path independently.
