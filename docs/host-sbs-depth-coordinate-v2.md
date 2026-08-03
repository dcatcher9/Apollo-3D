# Host SBS Depth Coordinate V2

Status: the V2 cutover is shipped as the sole live Host SBS renderer, with no V1 selection or
fallback. This working tree adds the unqualified schema-15 C75/collar trial described below; that
overlay still requires headset and 4K90 qualification. An unauthenticated model, tensor shape,
shader closure, state, or current frame renders flat SBS. Client SBS and offline conversion are
unchanged.

## Live working-tree behavior

```ini
sbs_3d_pop_strength = 2.0
```

Host SBS always starts the V2 producer and renderer. The producer, model identity, preprocess
identity, tensor shape, shader closure, resources, state, and renderer closure must all authenticate
before stereo geometry becomes live. Failure stays flat for that encode-device lifetime; it never
falls back to V1. The live shaders are compiled and cached at process start. Dump 3D's
full-frame diagnostic resources and mapping/mask shaders are created lazily; its optional canonical-
coordinate shader has no live authority, and no diagnostic failure can prevent a stream starting.

V2 takes `sbs_3d_pop_strength` literally. Legacy adaptive-pop, zero-plane, subject-stretch,
subject-recenter, min/max normalization, and adaptive-state controls do not change V2 geometry.
V2 has one separate, contract-owned adaptation: it chooses the logarithmic near-tail shoulder once
per shot from the acquisition frame's near occupancy. It never changes the configured pop, the
middle branch, the raw scale, or convergence.

## Design goal

V2 uses one authenticated raw DAV2 coordinate and one owner for each geometric decision. It avoids
endpoint normalization, near clamping, and stacked adaptive controllers while retaining foreground
relief and bounding the horizontal cliff that the stereo warp must invert.

```text
authenticated raw model field
  -> finite moments and collapse check
  -> scene-latched raw center + fixed model/shape scale
  -> same-frame canonical near-tail occupancy count
  -> scene-latched asymmetric far/near shoulder
  -> separate scene convergence scalar
  -> requested pop gain
  -> exact frame-local 4% source-U container
  -> immutable pre-limiter candidate p
  -> exact column upper/lower envelopes V+ and V- (shear 2)
  -> orientation-selective vertical share c = 0.75 V+ + 0.25 V-
  -> least row-wise majorant q of c
  -> 12-step contractive inverse using q
  -> positive (q - p) collar mask and color-only defocus
```

## Authenticated contract

Algorithm contract schema 15 admits one authenticated working-tree calibration:

| Property | Required value |
|---|---|
| Model key | `depth_anything_v2_fp16` |
| ONNX SHA-256 | `2df6223f206b5164e21f664ace61dabeb9bb6a49b8b5a3e00510b4807d0f5b04` |
| Preprocess profile | `apollo-dav2-area-hdr-srgb-imagenet-v1` |
| Input/output tensor shapes | 770 x 434; 1022 x 434; 1036 x 434; 434 x 770; 434 x 1022; 434 x 1036 |
| Raw coordinate scale | 0.5 DAV2 units |
| Direct source-U container | 0.04 |
| Maximum horizontal slope | 0.5 |
| Maximum vertical shear | 2.0 |
| Vertical upper-envelope share | 0.75 |
| Near-tail probe | `u > 1.0` |
| Dense-occupancy transition | 15% to 22% |
| Sparse / dense near-tail tau | 2.0 / 1.0 |

The exact model URL and bytes, preprocessing shader closure, ordered seven-shader producer closure,
and tensor shape are also authenticated. The six allowlisted shapes cover the standard landscape,
ultrawide, and portrait UI resolutions. Live Host SBS always selects the authenticated DAV2 Small
production identity; configured Base/custom model choices are read only by offline conversion and
the evaluator. An unauthenticated identity or tensor shape reaching the producer still fails flat.
Client SBS and MiDaS must not borrow this Host calibration.

The persistent GPU state also carries a deterministic checksum of the latched center, fixed
inverse scale, and calibration revision. A same-tag state with a corrupted center fails closed and
must reacquire from fresh same-frame evidence; a plausible but unauthenticated center cannot reach
the map or live renderer.

The contract decision is that model/preprocess/shape identity owns the scale; frame occupancy does
not. Occupancy selects only the near-tail compression described below.

## Coordinate and curve

For finite raw depth `r`, the frame producer calculates population mean, population standard
deviation, and exact extrema. Standard deviation is validity evidence only. A field with
`sigma <= 1e-6` is collapsed.

At the first usable frame, or after a confirmed cut, the scene camera latches:

```text
center = mean(r)
scale  = 0.5
u      = (r - center) / scale
```

There is no percentile stretch, min/max EMA, standard-deviation normalization, subject stretch,
or endpoint saturation. The canonical coordinate `u` is retained as a diagnostic; it is not a
visibility or rendering authority.

The curve is monotone and continuous:

```text
             tau_far * (exp(u / tau_far) - 1)                  u < 0
F(u) =       u                                                  0 <= u <= 1
             1 + tau_near * log(1 + (u - 1) / tau_near)        u > 1
```

with `tau_far = 0.15`. At acquisition, the GPU counts `coverage = fraction(u > 1)` and latches:

```text
dense_weight = smoothstep(0.15, 0.22, coverage)
tau_near     = lerp(2.0, 1.0, dense_weight)
```

The count shader is dispatched in sequence, but its GPU-uniform state/cut gate scans the tensor
only when acquiring a camera or accepting a confirmed cut. The result is held bit-stable within a
shot, so motion cannot pump depth. Sparse foregrounds
retain the original `tau_near = 2.0`; a large nearest layer such as browser surround receives no
extra near-tail emphasis and approaches `tau_near = 1.0`. The entire `u <= 1` branch is identical,
and both shoulder choices are monotone, C1 at `u = 1`, and unbounded—there is no near clamp or flat
slab.

The contract-pinned thresholds separate the paired 2026-08-02 hair inputs: the latest windowed frame
has 57.6% `u > 1` coverage while fullscreen has 12.3%. Across the current 360-frame core raw set,
coverage spans about 8% to 50% with a 24% median. This exercises both endpoints; extended
whole-clip evidence remains useful for future recalibration, but cannot silently change the
schema-15 constants.

## Pop, convergence, and the hard container

```text
requested_gain = requested_pop * 0.00375
p_requested    = requested_gain * (F(u) - convergence_curve)
p_base_envelope = requested_gain * (F_tau=2(u) - convergence_curve)
container      = min(1, 0.04 / max(abs(extrema(p_base_envelope))))
p              = requested_gain * container * (F(u) - convergence_curve)
```

`convergence_curve` is a separate scene-camera scalar and is currently zero. A future scene-level
zero-plane allocator may translate the curve once per scene; it must not change raw scale, curve
shape, or local cliffs.

The 4% container is frame-local representation safety. Its envelope always uses the original
`tau_near = 2.0`, even when the rendered shoulder is denser/compressed. Therefore occupancy
compression cannot relax the container and amplify middle or far geometry. It attacks and
recovers on the same frame, so one transient cannot weaken the rest of a shot.

## Orientation-selective cliff conditioning

The immutable candidate `p` records the requested signed one-eye source-U field. It is useful for
measuring how much intervention the renderer needed. It is also sampled by the live renderer only
to derive the color-only collar-defocus mask; it never replaces final geometry.

The first spatial pass computes both exact column-wise Lipschitz envelopes:

```text
V+(x, y) = max_s p(x,s) - step * abs(y-s)
V-(x, y) = min_s p(x,s) + step * abs(y-s)
c(x, y)  = 0.75 * V+(x,y) + 0.25 * V-(x,y)
step      = 2.0 / depth_width
```

A top-to-bottom scan followed by a bottom-to-top scan computes both envelopes and their fixed
share exactly. The selected shear-2
bound is expressed in source-width-normalized disparity: on an aspect-matched model/source grid it
is approximately two source-horizontal disparity pixels of change per source-image vertical pixel.
The 75/25 share reduces top-edge background shear while retaining substantially more foreground
volume than the rejected vertical-minorant endpoint. It may raise or lower the candidate locally.

The second pass computes the least row-wise majorant `q` of `c`:

```text
q(x, y) >= c(x, y)
abs(q(x + 1, y) - q(x, y)) <= 0.5 / depth_width
abs(q(x, y + 1) - q(x, y)) <= 2.0 / depth_width
```

The pure horizontal majorant preserves the vertical bound without introducing horizontal
foreground lowering, which avoids the measured hair/shoulder notch of a global two-dimensional
upper/lower blend. The final may still be below `p` where the vertical share compressed the first
foreground rows, so diagnostics report raising and lowering separately. The horizontal bound keeps
both eye maps one-to-one: every output pixel pulls a real source sample and no hidden-background
fill can copy the foreground.

`shadow_final_parallax` is the live position authority. `shadow_candidate_parallax` is immutable
pre-conditioner evidence. `shadow_vertical_majorant` is the exact `V+` diagnostic;
`shadow_vertical_conditioned` is the fixed 75/25 vertical share consumed by the row pass.
`shadow_coordinate` is coordinate evidence only: its full-size texture and alternate map
entrypoint are allocated/dispatched solely for an explicit Dump 3D snapshot.

## Collar defocus trial

After the inverse converges, the renderer evaluates the positive geometry correction at that same
source coordinate:

```text
raised_px = max(q - p, 0) * source_color_width
amount    = smoothstep(4, 20, raised_px)
color     = lerp(one_tap_color, fixed_sigma6_blur, amount)
```

The fixed blur is a one-pass 3x3 binomial approximation. A `[1,2,1]/4` axis kernel has variance
`radius^2/2`, so taps are placed at `sqrt(2) * 6` source pixels to preserve the configured
sigma-6 second moment. A 5x5 trial increased active color sampling from 9 to 25 reads; the user
reported little visible difference in that one crown A/B, so the denser kernel was removed. This
subjective observation is not a general equivalence claim.
The exact one-tap path is retained for `raised_px <= 4`; negative correction belongs to compressed
foreground rows and is never blurred. Both eyes use the same source-space rule. This is a
color-sampling experiment only: it does not change candidate, final parallax, the inverse, comfort
bounds, or scene state. Native color values are averaged without clamping, tone mapping, or gamma
conversion, preserving linear scRGB HDR values outside `[0,1]`. The live renderer closure and Dump
3D manifest pin these choices; the ordinary geometry evaluator does not qualify their appearance
or worst-case pixel cost. The 4/20/6 calibration is in pixels of the current source-color raster:
it is depth-grid independent, but it is not claimed to be stream-resolution invariant.

## Unique inverse

For either eye, the source coordinate is the fixed point:

```text
source = destination + eye_sign * q(source)
```

The slope bound is strictly below one, so this mapping is contractive and each destination has one
inverse. The production shader performs 12 iterations beginning at the destination. With slope
0.5 and the 0.04 source-U container, the remaining worst-case bound is
`0.5^12 * 0.04`, about 0.0375 source pixel at width 3840.

There is no multi-root visibility choice, canonical-order selection, forward-owner texture, or
synthetic hole fill in V2. The exact V2 mapping shader uses the same fixed-point function as live
rendering. The V2 mask shader marks only finite-source boundary extrapolation: red pixels are
inverse samples outside `[0, 1]` that live rendering clamps to the nearest source column. It does
not report internal holes, because the slope-limited map has none.

## GPU producer and state

An authenticated completion runs seven GPU passes:

1. Welford partial moments;
2. frame-stat merge;
3. acquisition/cut-gated canonical `u > 1` near-tail count with one atomic add per reduction group;
4. scene-camera/current-frame/near-shoulder resolve;
5. immutable candidate map;
6. exact column upper/lower shear-2 envelopes and their fixed 75/25 share;
7. least row-wise majorant of that vertically conditioned field.

All per-pixel data stays on the GPU. Live Host SBS binds the final conditioned field for geometry
and the matched immutable candidate for the collar-defocus mask; the removed V1 selector and
legacy warp prefilter have no live rendering role.

Dump 3D adds one diagnostic dispatch after production timing has ended to materialize the
canonical coordinate for that exact completed frame. Ordinary live frames neither allocate nor
write that full-size texture, and live-result authentication does not require it.

Camera availability and current-frame validity are distinct:

```text
NO_CAMERA + usable                    -> CAMERA + LIVE (acquire center + shoulder)
CAMERA + usable, no confirmed cut     -> CAMERA + LIVE (hold center + shoulder)
CAMERA + unusable, no confirmed cut   -> CAMERA + FLAT (retain camera only)
ANY + confirmed cut + usable          -> CAMERA + LIVE (replace center + shoulder)
ANY + confirmed cut + unusable        -> NO_CAMERA + FLAT
```

An unusable frame never pairs old per-pixel geometry with new color. The small scene camera may be
retained across an in-scene failure. Cut pulse/generation remains the sole scene-identity input.
The live pixel shader treats an invalid frame, invalid camera checksum, or unknown contract tag as
current-frame flat identity; it never uses `discard` to retain stale render-target pixels. A valid
matched pair may be repeated only while its source frame is at most 250 ms old. After that bound,
current color stays live and flat while the nonblocking depth query continues; a fresh completion
automatically resumes stereo.

Live telemetry follows the selected renderer: V2 reports fixed configured pop, no legacy adaptive
pop or zero-plane authority, and leaves legacy subject/anchor/range readiness fields invalid.

## Dump 3D semantics

New live dump manifest schema 8 describes the selected renderer without the rejected owner path.
Its `shadow_state.json` schema 8 records both generated spatial bounds and the complete 12-word
scene state. The added shoulder words expose the shot-latched near-tail coverage, the effective
near logarithmic tau, the exact tail count, and the camera-center integrity checksum. The same document
binds `near_tail_probe_u = 1`, coverage thresholds `0.15/0.22`, and dense-tail tau `1`, so a dump
cannot silently reinterpret how the near shoulder was selected.
Its `warp_map_shape.json` uses schema 2 so the unique-inverse/zero-mask validity contract cannot be
mistaken for the older coverage-based record:

| Artifact | V2 meaning |
|---|---|
| `warp_depth.f32` | exact final conditioned field sampled by live V2 geometry |
| `shadow_final_parallax.f32` | same final row-majorant of the vertical share; live position authority |
| `shadow_vertical_conditioned.f32` | exact 75/25 column-envelope share consumed by the row pass |
| `shadow_vertical_majorant.f32` | exact column-wise upper envelope `V+`; diagnostic |
| `shadow_candidate_parallax.f32` | immutable requested field before conditioning; diagnostic and color-mask input |
| `shadow_coordinate.f32` | dump-only canonical raw coordinate; diagnostic |
| `warp_map.f32` | exact 12-step fixed-point inverse source-U map |
| `warp_mask.png` | red = finite-source boundary extrapolation; no internal owner/fill path |

The manifest records the algorithm schema/tag, all seven producer shaders including the coverage
pass, the independent live renderer closure, and the fixed 4/20/6 collar-defocus policy.

The two 2026-08-02 hair captures were produced under depth-coordinate contract schema 7. They are
valuable historical input witnesses for replay, but their manifests and rendered artifacts do not
describe the schema-15 producer or the final live renderer. Fresh dumps are required for current
evidence.

## Hair-cliff evidence

Exact pop-1.5 replay of the two historical input witnesses selected slope 0.5. Evidence is outside
the repository under:

- `E:\ApolloDev\majorant-row-both-confirm-20260802`

The full-resolution eye crops show no duplicated hair or hair/body step. On fullscreen, 3.837% of
the field is raised, no candidate value is lowered, and the maximum raise is 0.0162933 source-U.
The central dark-subject proxy is 99.881% byte-for-byte unchanged and 100% preserved-or-raised. On
windowed input, 2.152% of the field is raised and the central subject is entirely unchanged. The
remaining fullscreen compromise was visible curtain/background bending around the crown. That
horizontal-only result motivated the selected vertical shear-2 pass; it is historical input
evidence, not proof of the new seven-pass output.

This is targeted historical evidence, not a current release baseline. Ongoing qualification should
use fresh schema-15/manifest-schema-8 dumps, whole-clip quality evaluation, isolated GPU timing,
hard-container telemetry, cut coverage, invalid-depth behavior, HDR, and long browser/video
sessions.

## Legacy boundary

V2 is the only live position authority and its standalone pixel shader does not include the V1
reprojection shader. Live Host SBS now runs a cut-only evidence and resolve path that publishes the
confirmed `{generation, pulse}` consumed by V2. It does not compile or dispatch the legacy subject,
stretch, adaptive-pop, or zero-plane analysis shaders. The cut detector's normalized depth is
private evidence and never becomes live geometry.

Legacy reprojection and normalization shader sources remain in the repository because the offline
converter and evaluation harness still compile their established pipelines. Their presence on disk
does not make V1 selectable in a live Host SBS stream. The corresponding `sbs_3d_*` configuration
keys remain accepted for offline/evaluator compatibility but cannot override live V2 geometry,
model identity, tensor shapes, or private cut calibration.

## Boundaries

This production cutover changes neither Client SBS nor offline conversion. Any future client backend must
authenticate and calibrate its own model bytes, preprocessing, tensor shape, direction, raw scale,
curve, gain, container, and slope. Offline conversion may reuse the coordinate and conditioned-field
semantics, but scene lookahead and encoder policy remain separate owners.
