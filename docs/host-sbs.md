# Host SBS pipeline

This document is the canonical design and runtime contract for Sunshine 3D's host-side mono-to-SBS
conversion. Host SBS has one geometry implementation: Depth Coordinate V2. An invalid or
unauthenticated V2 frame renders flat SBS; there is no alternate renderer.

Client SBS is a separate Moonlight 3D pipeline and does not share these constants. Offline
conversion uses the same V2 geometry and is described in
[Offline Host 3D conversion](whole-clip-sbs-pipeline.md).

## Pipeline

```mermaid
flowchart LR
    CAPTURE["Matched source frame"]
    PREPROCESS["HDR/SDR model preprocessing"]
    DAV2["Authenticated DAV2 Small inference"]
    CUT["Cut-only evidence and scene epoch"]
    CAMERA["Scene-latched raw center"]
    CURVE["Fixed raw coordinate and curve"]
    CONTAINER["Pointwise soft source-U container"]
    OWNER["Conservative ownership correction"]
    VERTICAL["75/25 vertical envelope share"]
    ROW["Horizontal majorant"]
    INVERSE["11-step contractive inverse"]
    COLOR["One native linear-color sample"]
    ENCODE["Packed SBS and NVENC"]

    CAPTURE --> PREPROCESS --> DAV2
    DAV2 --> CUT --> CAMERA
    DAV2 --> CAMERA --> CURVE --> CONTAINER --> OWNER --> VERTICAL --> ROW --> INVERSE
    CAPTURE --> COLOR
    INVERSE --> COLOR --> ENCODE
```

Only the final row-conditioned parallax field has rendering authority. Raw depth, canonical
coordinate, candidate parallax, ownership-refined parallax, and vertical-envelope outputs are
diagnostics that explain how the final field was produced.

## Authenticated production contract

The generated Depth Coordinate contract is the machine-readable authority. At schema 26, live
Host SBS admits the following production calibration:

| Property | Production value |
|---|---|
| Model | Depth Anything V2 Small FP16 |
| Model key | `depth_anything_v2_fp16` |
| Tensor shapes | `770x434`, `1022x434`, `1036x434`, and their portrait transposes |
| Raw coordinate scale | `2.25` DAV2 units |
| Gain per pop unit | `0.00375` source U |
| Direct parallax container | `0.04` source U per eye |
| Far exponential tau | `0.75` |
| Near logarithmic tau | `0.5` |
| Maximum horizontal slope | `0.5` |
| Maximum vertical shear | `2.0` |
| Vertical upper-envelope share | `0.75` |
| Convergence translation | exactly `0` |

The model bytes, model URL, preprocessing profile and shader closure, tensor shape, ordered
producer closure, state checksum, and renderer closure must also authenticate. Live Host SBS,
production Web UI conversion, and the maintained benchmark harness all use this one pinned model.
There is no supported Host SBS model selector. Unsupported model identities and tensor shapes fail
closed.

### Authenticated resolution fitting

Moonlight 3D's 12 standard XR resolutions all fit one of the six authenticated tensors:
`770x434`, `1022x434`, `1036x434`, and their portrait transposes. A custom source resolution is
also valid when the production fitter maps it exactly to one of those shapes; for example, a
same-aspect `1280x720` source still maps to `770x434`. A custom aspect whose fitted tensor is not
allowlisted is rejected before live Host SBS starts. The host never substitutes a nearby tensor or
silently emits a flat stream for an unsupported setup.

Offline V2 conversion uses the same fitter and allowlist. It aborts an unsupported job rather than
publishing a flat converted video. Other documents link to this section instead of maintaining a
second resolution list.

Portrait is an explicit `width < height` display mode. Host SBS does not rotate captured content;
non-identity Windows display rotation is rejected before pipeline setup.

## Color and HDR

The model and the rendered color have different color requirements:

- BGRA8 SDR capture is interpreted as display-referred sRGB for model preprocessing.
- FP16 scRGB HDR capture remains linear Rec.709/scRGB for rendering. The inference copy applies a
  luminance-preserving absolute tone map and the sRGB transfer needed by DAV2; it does not clamp
  highlights to `1.0` before inference.
- The stereo warp applies no transfer function, gamut conversion, sharpening, or post-warp blur.
  Each eye takes one sample from the original source texture in its native linear or encoded domain.
- The existing encoder conversion stage produces SDR YUV or Rec.2100 PQ and carries the validated
  HDR metadata. Host SBS does not independently reinterpret the stream's color metadata.

Tone-mapped debug PNGs are viewing aids, not numeric HDR evidence. Use the floating-point dump
artifacts and manifest color fields when auditing the pipeline.

## Scene camera and raw coordinate

For each finite raw DAV2 field, the producer calculates exact extrema, arithmetic mean, and
population standard deviation. Standard deviation is validity evidence only. A field with
`sigma <= 1e-6` is collapsed and cannot produce current-frame geometry.

At startup or after a confirmed cut, the first usable field acquires its arithmetic mean as the
scene center. This gives occupancy-weighted behavior without a discrete scene classifier: a small
near object barely moves the center and retains relief, while a large near region naturally pulls
the zero plane toward itself and is not boosted as an isolated object.

```text
u = (raw_depth - scene_center) / 2.25
```

The selected center is the exact zero-disparity plane. There is no additional convergence offset.
It remains fixed through later valid, invalid, and fast-motion frames until the next confirmed
cut. There is no age-based camera replacement, per-frame recentering, subject tracking, endpoint
normalization, or adaptive pop multiplier.

The fixed monotone curve is:

```text
             0.75 * (exp(u / 0.75) - 1)          u < 0
F(u) =       u                                    0 <= u <= 1
             1 + 0.5 * log(1 + (u - 1) / 0.5)    u > 1
```

It is continuous, monotone, and unbounded on the near branch. The wider linear coordinate region
preserves middle-layer ordering; the far shelf is deliberately lower-value than the middle and
near zones, and the near logarithm compresses extreme foreground without clamping it into a flat
slab.

## Pop and the pointwise soft container

`sbs_3d_pop_strength` is the only live geometry control. V2 uses it literally:

```text
requested_gain = pop_strength * 0.00375
requested      = requested_gain * F(u)
candidate      = requested / fourth_root(1 + (requested / 0.04)^4)
```

The configured default is owned by [Configuration](configuration.md#sbs_3d_pop_strength). The
container is odd, monotone, has unit slope at zero, and approaches the signed `0.04` source-U
representation limit without a hard endpoint clamp. It is applied independently to every depth
texel. An extreme object therefore cannot shrink the relief of unrelated geometry, and the
container cannot make a local depth cliff steeper. It has no history and does not modify the scene
camera or configured pop.

The 12-word state layout retains `container_scale` for ABI compatibility. Its only valid value is
exactly `1.0`; all attenuation belongs to the pointwise map above. Dumps, replay traces, and the
live state validator fail closed on any other value.

## Foreground ownership

DAV2 runs at a lower spatial resolution than the source. A depth boundary can therefore land in a
mixed model texel whose center belongs to the far side even when most of the source footprint
belongs to the foreground. Before slope conditioning, the ownership pass checks the exact matched
full-resolution source along the candidate cliff normal.

It changes a cell only when all of the following evidence agrees:

- the candidate contains one strong, stable near/far cliff;
- neighboring model cells establish an existing near plateau;
- the full-resolution source contains one unique, monotone contour at the same boundary; and
- the correction can only pull the mixed far-side cell toward that existing near plateau.

The pass never lowers parallax, creates a new edge, paints color, or guesses between competing
contours. Ambiguous evidence is an exact no-op. This precision-first policy intentionally abstains
on many transparent, reflective, very thin, or highly textured boundaries.

## Cliff conditioning

A large depth cliff cannot be inverted safely as a single-valued backward warp without either
compressing foreground disparity, deforming visible background, or synthesizing missing pixels.
Sunshine 3D does not synthesize hidden background. It uses a bounded compromise:

1. Compute exact vertical upper and lower Lipschitz envelopes with a maximum shear of
   `2 / depth_width` per adjacent row.
2. Combine them as `0.75 * upper + 0.25 * lower`.
3. Compute the least horizontal majorant with a maximum slope of
   `0.5 / depth_width` per adjacent column.

The horizontal majorant guarantees a unique contractive inverse and preserves lateral foreground
continuity. The vertical share limits crown bending without fully flattening the foreground. The
trade-off remains visible on severe hair, glass-rim, and small near-object crowns: a large raw cliff
becomes a wider safe ramp, which can bend real source samples differently in the two eyes.

A previously tested post-warp collar blur was removed because it introduced a translucent halo at
a hand boundary. More inverse iterations do not solve the crown trade-off; they only solve the
already selected field more precisely. Any replacement must be qualified on hair, shoulders,
transparent rims, thin structures, and ordinary non-edge content rather than one witness frame.

## Inverse warp and source sampling

Each eye solves the same final signed parallax field with opposite signs using 11 fixed-point
iterations. The horizontal slope bound keeps the mapping contractive and gives one unique source
coordinate. The renderer then takes one linear-filtered sample from the original source texture.

There is no forward owner, multi-root visibility choice, inpaint pass, or synthetic internal fill.
Only samples outside the finite source rectangle clamp to its nearest edge; the diagnostic mask
reports that source-boundary condition rather than internal disocclusion.

## Frame attribution and failure behavior

Color, raw depth, scene state, and parallax are bound to an exact completed source-frame identity.
An unusable current field renders the current color flat. It may retain the small scene camera so a
later usable frame resumes the same coordinate, but it never pairs old per-pixel geometry with new
color. A confirmed cut invalidates the old camera; the next usable field acquires the new one.

Model preparation, shader compilation, and the live renderer are fail-closed. Live shaders are
compiled and cached at process startup. Dump-only resources are created lazily and cannot prevent a
stream from starting. A failure in optional diagnostics has no rendering authority.

TensorRT inference and all coordinate passes remain on the GPU. The live path does not add a
per-frame GPU-to-CPU readback. When inference is still busy, the capture loop must not enqueue an
unbounded backlog; it continues with flat/current output according to the matched-frame contract.
Telemetry readback is nonblocking and may drop samples under GPU load, while offline evaluation
may intentionally block to obtain a complete trace.

## Dump 3D and evaluation

A current Dump 3D package records:

- the matched source, model input, and direct raw DAV2 output;
- candidate, ownership-refined, vertical-envelope, and final parallax fields;
- scene camera and cut attribution;
- the exact inverse-warp map and finite-source mask; and
- the rendered packed SBS image plus model, shader, state, color, and source provenance.

Use `.f32` artifacts for quantitative comparisons. Independently stretched PNG previews can hide
scale differences and must not be compared as absolute depth values. Older dumps remain useful as
input witnesses only; they cannot authenticate a newer producer or renderer schema.

The supported commands, dump contract, metrics, and baseline policy live in
[the sbsbench guide](../tools/sbsbench/README.md). Scene-cut behavior and its headset acceptance
plan live in [Host SBS scene cuts](host-sbs-scene-cuts.md).

## Known limitations

- DAV2 is a relative monocular model. Framing and surrounding context can change its raw depth;
  windowed and fullscreen versions of the same content are not guaranteed to be affine-equivalent.
- Single-frame warping has no observation of newly exposed background. Strong foreground cliffs
  therefore require a visible geometry compromise unless temporal or synthesized fill is added.
- Transparent, reflective, and sub-grid objects often lack a unique ownership contour. The
  precision-first ownership pass abstains instead of risking a wrong snap.
- Scene detection is inferred, not ground truth. Exposure, structureless frames, persistent motion,
  and rapid consecutive cuts require the separate guarded state machine.
- Unsupported model or fitted-tensor setup is rejected before Host SBS starts. A bad per-frame
  shader/state/field identity renders that current frame flat rather than attempting a best-effort
  geometry fallback.
