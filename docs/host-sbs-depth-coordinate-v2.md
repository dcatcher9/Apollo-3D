# Host SBS Depth Coordinate V2

Status: the V2 cutover is shipped as the sole live Host SBS renderer, with no V1 selection or
fallback. Schema 23 adds a conservative, full-resolution foreground-ownership pass before the
C75 orientation-selective geometry, while the experimental collar
color filter was removed after it produced a visible hand-boundary halo in live fullscreen video.
An unauthenticated model, tensor shape,
shader closure, state, or current frame renders flat SBS. Client SBS and offline conversion are
unchanged.

## Live working-tree behavior

```ini
sbs_3d_pop_strength = 1.0
```

Host SBS always starts the V2 producer and renderer. The producer, model identity, preprocess
identity, tensor shape, shader closure, resources, state, and renderer closure must all authenticate
before stereo geometry becomes live. Failure stays flat for that encode-device lifetime; it never
falls back to V1. The live shaders are compiled and cached at process start. Dump 3D's
full-frame diagnostic resources and mapping/mask shaders are created lazily; its optional canonical-
coordinate shader has no live authority, and no diagnostic failure can prevent a stream starting.

V2 takes `sbs_3d_pop_strength` literally. Legacy adaptive-pop, zero-plane, subject-stretch,
subject-recenter, min/max normalization, and adaptive-state controls do not change V2 geometry.
The logarithmic near branch is fixed at `tau_near = 2.0`; frame occupancy does not select another
curve. At scene acquisition, a conservative histogram valley may become a higher raw zero plane;
ambiguous evidence retains the arithmetic mean. At startup or after a confirmed cut, the first
usable depth field establishes that center immediately. It then remains fixed through every later
valid, invalid, or fast-motion frame until the next confirmed cut. Both outcomes use exactly zero
convergence: the selected center is the raw
zero-disparity plane, and the selector adds no separate translation. V2 never changes the
configured pop, curve, or raw scale.

## Design goal

V2 uses one authenticated raw DAV2 coordinate and one owner for each geometric decision. It avoids
endpoint normalization, near clamping, and stacked adaptive controllers while retaining foreground
relief and bounding the horizontal cliff that the stereo warp must invert.

```text
authenticated raw model field
  -> finite moments and collapse check
  -> three-class upper-valley selection, else mean fallback
  -> immediate camera held unchanged until the next confirmed cut
  -> scene-latched raw center (the zero plane) + fixed model/shape scale
  -> fixed asymmetric far/near curve
  -> requested pop gain
  -> exact frame-local 4% source-U container
  -> immutable pre-limiter candidate p
  -> conservative full-resolution source-contour ownership o (raise-only; ambiguous = p)
  -> exact column upper/lower envelopes V+ and V- of o (shear 2)
  -> orientation-selective vertical share c = 0.75 V+ + 0.25 V-
  -> least row-wise majorant q of c
  -> 12-step contractive inverse using q
  -> one linear source-color sample at the inverse-warped coordinate
```

## Authenticated contract

Algorithm contract schema 23 admits one authenticated working-tree calibration:

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
| Near logarithmic tau | 2.0 |
| Stage valley maximum | 0.75 of the weaker adjacent peak |
| Zero-disparity raw plane | selected scene center; no separate convergence translation |

The exact model URL and bytes, preprocessing shader closure, the raw min/max and histogram roots,
the ordered seven-pass coordinate producer and complete nine-root producer closure,
and tensor shape are also authenticated. The six allowlisted shapes cover the standard landscape,
ultrawide, and portrait UI resolutions. Live Host SBS always selects the authenticated DAV2 Small
production identity; configured Base/custom model choices are read only by offline conversion and
the evaluator. An unauthenticated identity or tensor shape reaching the producer still fails flat.
Client SBS and MiDaS must not borrow this Host calibration.

The persistent GPU state also carries a deterministic checksum of the latched center, fixed
inverse scale, and calibration revision. A same-tag state with a corrupted center fails closed and
must reacquire from fresh same-frame evidence; a plausible but unauthenticated center cannot reach
the map or live renderer.

The contract decision is that model/preprocess/shape identity owns the scale and the curve is fixed.
The acquisition histogram may choose only the scene's selected center. It cannot change
convergence, pop, scale, or curve shape.

## Coordinate and curve

For finite raw depth `r`, the frame producer calculates population mean, population standard
deviation, and exact extrema. Standard deviation is validity evidence only. A field with
`sigma <= 1e-6` is collapsed.

At startup, or after a confirmed cut, the first usable depth field establishes the scene camera.
That one acquisition computes the exact arithmetic mean, reuses the existing 256-bin raw histogram,
merges adjacent pairs into 128 bins, and evaluates an ascending, strict-tie three-class Otsu split.
Every class must contain at
least `N/128` texels. The original counts are smoothed once with the zero-padded kernel
`[0.25, 0.5, 0.25]` only for the upper-valley test:

```text
valley = max(s[j], s[j+1]) / min(peak_middle, peak_near)
T      = center_of_histogram_bin(j)
acceptance_center = T - 0.5

if valley <= 0.75 and acceptance_center > mean(r):
    center = T
else:
    center = mean(r)

convergence_curve = 0
scale = 0.5
u     = (r - center) / scale
```

Once acquired, the selected center is held through all later usable, unusable, and fast-motion
frames until the next confirmed cut. There is no per-frame tracking, pending replacement, age-based
correction, or late relatch. When accepted, `T` is the zero plane and the upper knee `u = 1` is
`T + 0.5`.
The older `T - 0.5 > mean` guard deliberately remains the adoption test; replacing it with the
weaker `T > mean` test incorrectly adopts browser-surround histograms. This rule is intentionally
precision-first: a shallow valley, incomplete histogram, invalid/collapsed depth, or an
acceptance center that does not move upward from the mean abstains.

There is no percentile stretch, min/max EMA, standard-deviation normalization, subject stretch,
or endpoint saturation. The canonical coordinate `u` is retained as a diagnostic; it is not a
visibility or rendering authority.

The curve is monotone and continuous:

```text
             tau_far * (exp(u / tau_far) - 1)                  u < 0
F(u) =       u                                                  0 <= u <= 1
             1 + 2 * log(1 + (u - 1) / 2)                      u > 1
```

with `tau_far = 0.15` and fixed `tau_near = 2.0`. The curve is monotone, C1 at `u = 1`, and
unbounded: there is no near clamp or flat slab. The removed occupancy classifier compressed large
near regions toward `tau_near = 1.0`; that made otherwise similar windowed and fullscreen content
choose different geometry solely because browser surround changed the counted area. Schema 16
removes that content-dependent branch and its GPU reduction pass.

## Pop, zero plane, and the hard container

```text
requested_gain = requested_pop * 0.00375
p_requested    = requested_gain * F_tau=2(u)
container      = min(1, 0.04 / max(abs(extrema(p_requested))))
p              = requested_gain * container * F(u)
```

Convergence is exactly zero for both selector outcomes. It is retained in serialized state and its
integrity checksum as a fail-closed invariant, not as a calibrated controller or user parameter.
Because `F(0) = 0`, the exact zero-disparity raw coordinate is always `raw = center`. Accepting the
Otsu evidence puts its upper valley at zero and the fixed linear/log knee one raw scale above it;
it does not apply an additional whole-scene translation. The current live and qualification
reference is `requested_pop = 1.0`.

The 4% container is frame-local representation safety. It evaluates the same fixed curve that is
rendered, attacks and recovers on the same frame, and cannot weaken the rest of a shot.

## Orientation-selective cliff conditioning

The immutable candidate `p` records the requested signed one-eye source-U field. It is useful for
measuring how much intervention the renderer needed, but it is not sampled by the live renderer
and never replaces final geometry.

Before slope conditioning, the ownership pass compares each far-side candidate boundary texel with
its four cardinal neighbors. It proceeds only for one unique, sufficiently large nearward cliff
with stable near and far plateaus. Against the exact matched full-resolution source frame, it then
requires one strong, monotone color transition along the same normal; raw reversals, detected
competing contours, low contrast, source dimensions smaller than the model grid, invalid data, and
every other detected ambiguous case are exact no-ops. A fixed normalized profile, guaranteed-pure
endpoint references, and five bounded
raw-crossing refinements recover the same 10%-99% foreground coverage across 720p, 1080p, 1440p,
and 4K source raster phases. Accepted
coverage fractionally pulls only that mixed boundary texel toward its existing near plateau:

```text
o(x, y) = max(p(x, y), lerp(p(x, y), p_near(x, y), foreground_coverage))
```

The pass never lowers parallax, creates a depth edge, paints source color, or fills hidden
background. A missing matched source also produces `o = p`. It corrects conservative foreground
ownership only; the unavoidable no-fill background ramp and residual rim/crown compromise remain
limitations. Its fixed finite profile also cannot prove that source-pixel-scale contours which
alias into one monotone sampled transition are unique; those rare cases may remain unresolved.
The downstream envelopes, not ownership, remain the slope authorities.

The next spatial pass computes both exact column-wise Lipschitz envelopes of `o`:

```text
V+(x, y) = max_s o(x,s) - step * abs(y-s)
V-(x, y) = min_s o(x,s) + step * abs(y-s)
c(x, y)  = 0.75 * V+(x,y) + 0.25 * V-(x,y)
step      = 2.0 / depth_width
```

A top-to-bottom scan followed by a bottom-to-top scan computes both envelopes and their fixed
share exactly. The selected shear-2
bound is expressed in source-width-normalized disparity: on an aspect-matched model/source grid it
is approximately two source-horizontal disparity pixels of change per source-image vertical pixel.
The 75/25 share reduces top-edge background shear while retaining substantially more foreground
volume than the rejected vertical-minorant endpoint. It may raise or lower the ownership-refined
field and the original candidate locally.

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
pre-ownership evidence; `shadow_ownership_refined_parallax` is the exact field consumed by the
vertical envelopes. `shadow_vertical_majorant` is the exact `V+` diagnostic;
`shadow_vertical_conditioned` is the fixed 75/25 vertical share consumed by the row pass.
`shadow_coordinate` is coordinate evidence only: its full-size texture and alternate map
entrypoint are allocated/dispatched solely for an explicit Dump 3D snapshot.

## Rejected collar-defocus trial

Schema 15 briefly used positive `q - p` as a mask for a 3x3 color blur after inverse reprojection.
Matched earlier/current hand-scene dumps showed that the filter introduced a new translucent/soft
halo at fingertips and the hand silhouette, especially fullscreen. The older no-filter renderer
had comparable positive geometry corrections without that halo. The trial was therefore removed:
candidate is no longer bound to the live pixel shader, every output uses one native linear color
sample, and C75 geometry is unchanged. Manifest schema 9 records this disabled policy explicitly.

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

An authenticated completion runs seven coordinate GPU passes:

1. Welford partial moments;
2. frame-stat merge;
3. scene-camera/current-frame resolve;
4. immutable candidate map;
5. conservative matched-source foreground ownership refinement;
6. exact column upper/lower shear-2 envelopes and their fixed 75/25 share;
7. least row-wise majorant of that vertically conditioned field.

The acquisition resolver reuses a snapshot of the already-produced raw 256-bin histogram; it does
not add another full-tensor histogram dispatch or a CPU readback. The authenticated nine-root
source closure therefore includes `depth_minmax_cs` and `depth_hist_cs` in addition to these seven
coordinate roots. The accepted source SRV is retained until its asynchronous raw-depth completion
runs ownership; this adds no color copy or GPU/CPU synchronization. All per-pixel data stays on the
GPU. Live Host SBS binds only the final conditioned field and its
authenticated state for geometry; the removed V1 selector and
legacy warp prefilter have no live rendering role.

Dump 3D adds one diagnostic dispatch after production timing has ended to materialize the
canonical coordinate for that exact completed frame. Ordinary live frames neither allocate nor
write that full-size texture, and live-result authentication does not require it.

Camera availability and current-frame validity are distinct:

```text
NO_CAMERA + usable                    -> CAMERA + LIVE (acquire center)
CAMERA + usable, no confirmed cut     -> CAMERA + LIVE (hold center)
CAMERA + unusable, no confirmed cut   -> CAMERA + FLAT (retain camera only)
ANY + confirmed cut + usable          -> CAMERA + LIVE (replace center)
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

New live dump manifest schema 10 describes the selected renderer without a forward-owner/fill path
and separately records the conservative pre-conditioner ownership field.
Its `shadow_state.json` schema 13 records both generated spatial bounds and the complete 12-word
scene state: center, inverse scale, convergence, generation/revision, validity, current-frame
container state, camera-center integrity checksum, and three zeroed reserved words. The fixed
near-curve and stage-selection constants are authenticated by the algorithm contract. The camera
checksum includes convergence, so a stale or corrupted selector result fails closed.
Its `warp_map_shape.json` uses schema 2 so the unique-inverse/zero-mask validity contract cannot be
mistaken for the older coverage-based record:

| Artifact | V2 meaning |
|---|---|
| `warp_depth.f32` | exact final conditioned field sampled by live V2 geometry |
| `shadow_final_parallax.f32` | same final row-majorant of the vertical share; live position authority |
| `shadow_vertical_conditioned.f32` | exact 75/25 column-envelope share consumed by the row pass |
| `shadow_vertical_majorant.f32` | exact column-wise upper envelope `V+`; diagnostic |
| `shadow_ownership_refined_parallax.f32` | raise-only matched-source ownership result consumed by the vertical envelopes |
| `shadow_candidate_parallax.f32` | immutable requested field before ownership and conditioning; diagnostic |
| `shadow_coordinate.f32` | dump-only canonical raw coordinate; diagnostic |
| `warp_map.f32` | exact 12-step fixed-point inverse source-U map |
| `warp_mask.png` | red = finite-source boundary extrapolation; no internal owner/fill path |

The manifest records the algorithm schema/tag, all seven coordinate passes plus the reused raw
min/max and histogram roots (nine authenticated roots total), the independent live renderer
closure, and that collar defocus is
disabled in favor of a
single native-color sample.

The two 2026-08-02 hair captures were produced under depth-coordinate contract schema 7. They are
valuable historical input witnesses for replay, but their manifests and rendered artifacts do not
describe the schema-23 producer or the final live renderer. Fresh dumps are required for current
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
evidence, not proof of the current seven-pass ownership-refined output.

This is targeted historical evidence, not a current release baseline. Ongoing qualification should
use fresh schema-23/manifest-schema-10/state-schema-13 dumps, whole-clip quality evaluation, isolated GPU timing,
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
