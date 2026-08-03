# Host SBS Depth-System Rethink

Status: design record for the shipped Host SBS V2 cutover. V2 is the sole live Host SBS renderer;
Client SBS and offline conversion remain unchanged.

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
  + one scene-latched near-occupancy shoulder
  + one scene-latched convergence value
  + one requested pop
  + one frame-local hard container
  + one near-preserving anisotropic 2D slope constraint
  + one unique contractive inverse
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
| Scene convergence | Keep as a separate scalar | A stable screen plane must not also own scale or cliff shaping. |
| Edge-risk adaptive-pop boost | Remove | A difficult local edge should be repaired locally, not flatten the frame. |
| Shot-latched near-occupancy shoulder | Keep | A large nearest layer should not receive the same tail emphasis as a sparse foreground; it changes only `u > 1`, once per shot. |
| User pop | Keep | V2 takes `cfg.pop_strength` literally. |
| Permanent within-shot safety minimum | Remove | A transient prediction must not weaken a long scene indefinitely. |
| Frame-local 4% container | Keep | This owns a precise representation bound and recovers immediately. |
| Temporal depth EMA | Do not port initially | Add temporal filtering only after raw-coordinate flicker is measured. |
| Existing cut pulse/generation | Keep temporarily behind a narrow bridge | V2 consumes only one scene-reset input. The legacy analysis stack currently produces it, but none of its geometry state reaches V2. |
| Old depth with current color | Remove | It breaks frame attribution. |
| Immutable candidate field | Keep as diagnostic | It records requested geometry before local safety intervention. |
| Canonical coordinate | Keep as diagnostic | It helps explain model output but does not select visibility. |
| Least near-preserving anisotropic 2D majorant | Keep as live position authority | A vertical shear-2 majorant conditions one-row silhouette onsets, then a slope-0.5 row majorant retains requested near disparity, keeps both eye maps invertible, and assigns the unavoidable collar to less-important background. |
| Forward owner, multi-root visibility, synthetic fill | Remove | The slope-limited final field has a unique inverse; these mechanisms caused the measured bands. |
| Adaptive ROI | Keep removed | It adds inference, tracking, shape churn, and boundary policy. |
| Area/HDR-aware model preprocessing | Keep | It preserves input signal and belongs to authenticated model provenance. |

## Coordinate and convergence

The selected Host live calibration uses a fixed model/preprocess/shape scale:

```text
u           = (raw - gauge_center_raw) / coordinate_scale
curve       = F(u)
candidate p = requested_gain * container * (curve - convergence_curve)
```

`gauge_center_raw` is acquired on the first usable frame and replaced on a confirmed cut.
`coordinate_scale` is fixed by the authenticated calibration. `convergence_curve` is separate and
currently zero. This separation is structural: a future zero-plane allocator may translate the
curve once per scene, but it may not rescale or reshape depth.

Schema 14 authenticates the standard tensor family: `770x434`, `1022x434`, `1036x434`,
`434x770`, `434x1022`, and `434x1036`. DAV2 Base, custom models, and custom tensor shapes have no
live calibration and fail flat rather than borrowing the Small-model scale.

There is no live gauge tracker yet. Per-frame mean changes cannot distinguish model gauge drift
from camera motion or changed occupancy. Any tracker would need controlled registered evidence and
source-time semantics before it could own screen-plane motion.

## Local cliff policy

After the exact source-U container, the candidate can still contain a steep silhouette cliff. The
live field is a vertical-first separable majorant:

```text
v >= p
|v(x,y+1) - v(x,y)| <= 2.0 / depth_width
q >= v
|q(x+1,y) - q(x,y)| <= 0.5 / depth_width
```

Top/bottom scans compute the exact shear-2 column majorant `v`; left/right scans then compute the
exact row majorant `q`. Both are one-sided, so `q >= v >= p` and requested near disparity is never
lowered. The row scan preserves the vertical bound, making `q` the least anisotropic 2D majorant
under the selected separable metric. The passes raise only the minimum low/background collar
needed to bound crown shear and keep both eye maps one-to-one. That visible-background strain is
the deliberate no-hallucination trade: a rigid foreground shift exposes source pixels that do not
exist in a monocular frame. The rejected forward-owner experiments kept that rigid cliff and then
copied hair into their synthetic fill, producing the duplicated silhouette.

The final slope is below one, so each eye has a unique inverse. Live and exact dump mapping use 12
fixed-point iterations. Canonical coordinate, multiple-root selection, forward coverage, owner
textures, and internal hole filling have no live V2 role. The V2 dump mask attributes only inverse
samples beyond the finite source boundary that live rendering clamps to the nearest edge column.

## Adaptive responsibilities

### Pop

- Input: one user-requested `cfg.pop_strength`.
- Global reduction: only the exact current-frame 4% source-U container.
- Local response: the near-preserving majorant enforces the slope contract.
- Forbidden: content boost, edge-density feedback, permanent shot minimum, or stacked scale
  normalizers.

Ordinary safe frames therefore use the requested pop. Limiter burden is diagnostic evidence; it
does not feed another controller.

### Near shoulder

- Count the acquisition frame's canonical fraction `u > 1` entirely on the GPU.
- Hold `tau_near = 2` through 15% occupancy, blend to `tau_near = 1` at 22%, and latch until a
  confirmed cut.
- Keep `u <= 1`, the fixed raw scale, convergence, and configured pop unchanged.
- Size the hard container with the original tau-2 envelope, so compression cannot release global
  gain elsewhere.
- Forbidden: per-frame relatching, ROI/content classification, RGB semantics, or hard near clamps.

### Zero plane

- Acquire the scene camera on the first usable frame.
- Replace it on a confirmed cut.
- Hold it on ordinary frames.
- Keep convergence separate from scale and curve shape.

The current convergence value is zero. Intelligent zero-plane allocation is not claimed by V2 yet.

### Scene cuts

- V2 consumes only `{generation, pulse}`.
- Pulse handles the exact update; generation recovers a pulse missed while inference was skipped.
- Persistent-motion behavior belongs to the detector; V2 adds no second cooldown or timeout.
- The live estimator currently obtains this pair from a narrow bridge over the legacy analysis
  stack. That bridge is the only legacy live dependency; its depth, anchor, pop, and warp outputs
  have no V2 authority.

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

Persistent state contains scene center, fixed scale, convergence, cut generation, calibration
revision, and the acquisition frame's near-tail coverage/count/effective tau. Frame validity and
container scale are current-dispatch values, not controller history.

## Runtime boundaries

| Runtime | Shared concept | Runtime-specific owner |
|---|---|---|
| Host live V2 | sign, curve, separate convergence, container, final majorant | first usable acquisition, cut reset, no lookahead |
| Host offline | potential future coordinate semantics | finalized scene boundary, lookahead, encoder and HDR policy |
| Client DAV2 | semantic interface only | Android model bytes, preprocessing, tensor shape, renderer and display calibration |
| Client MiDaS | semantic interface only | independent sign, scale, curve, validity, slope, and performance calibration |

Similar labels do not make legacy Client SBS values wire-compatible with V2. Client and offline
remain unchanged by the Host live cutover.

## Performance shape

The producer uses moments reduction, frame resolve, an acquisition/cut-gated group-reduced
near-tail count, state resolve, coordinate/candidate mapping, one column-wise vertical-majorant
dispatch, and one row-wise majorant dispatch. On ordinary frames the count shader exits from a
GPU-uniform state/cut gate before tensor reads or barriers. When needed, the counter adds once per
threadgroup and is consumed/cleared by the same-frame state resolve. Columns and rows
are serial only inside their owning GPU threads; independent columns/rows execute in parallel and
no CPU readback or GPU/CPU synchronization is introduced.

The renderer samples the final field through a 12-step fixed-point inverse. It does not allocate,
clear, dispatch, or bind a full-resolution owner texture, and it performs
no bounded hole search. Exact dump mapping compiles from the same renderer closure; the mask
entrypoint writes zero.

The production parallax interval measures exactly the seven V2 compute passes. The retained
scene-cut bridge runs before that interval and must be reported separately when measuring the
remaining legacy-analysis overhead. Historical shadow-mode totals are not production V2 cost.

## Provenance and evidence

Depth-coordinate contract schema 14 binds algorithm constants, model identity, preprocessing
closure, the six standard tensor shapes, and the ordered seven-shader producer closure. The live renderer has its own
authenticated main/mapping/mask closure. New Dump 3D manifest schema 7 records final parallax as
position authority and candidate/canonical/vertical-majorant fields as diagnostics.

The original fullscreen and windowed hair dumps use depth-coordinate schema 7. They remain useful
historical input witnesses for exact replay but cannot authenticate schema-14 output. The prior
slope-0.5 row-majorant replay removes the duplicated hair and hair/body step while never lowering
the candidate. It raises 3.837% of fullscreen texels and 2.152% of windowed texels; its residual
fullscreen crown bending motivated the selected vertical shear-2 pass. Fresh schema-14,
manifest-schema-7 live dumps remain required.

## Production cutover and continuing gates

The live cutover removes V1 renderer selection, prewarms the production V2 shaders, and fails flat
when production authentication or execution fails. It does not authorize unsupported models or
tensor shapes and it does not migrate Client SBS or offline conversion.

Continuing release evidence should:

1. Generate fresh schema-14 native and manifest-schema-7 live evidence with exact producer and renderer closures.
2. Replay core and extended clips at the chosen fixed scale and pop values.
3. Inspect near-hair cliffs, thin structures, flat pages, HDR, cuts, flashes, fast motion, invalid
   inference, and long browser/video sessions.
4. Report container attenuation, candidate-to-final raising, near-core retention, final slope,
   fixed-point residual, disparity bounds, and timing independently.
5. Prove V2 takes `pop_strength` literally, ignores legacy adaptive geometry controls, and changes
   its near shoulder only on acquisition or a confirmed cut.
6. Extract and authenticate the narrow cut epoch before retiring the remaining legacy scene-cut bridge.
7. Calibrate every additional model, tensor shape, client backend, and offline path independently.
