# Production DAV2 + frozen ZipDepth convex 2x

Status: single-high-I/O implementation candidate. The model graph, TensorRT plan, all-high runtime,
and six-shape limiter path are implemented. Full native/Python regression, authenticated dump, and
an initial landscape Galaxy XR execution are clean. A paired evaluator report and explicit visual
acceptance remain required before this candidate becomes the published production baseline.

This note records the implementation based on production commit
`1a434f379b2a6d39ec9915deefa2277043cbce72`. The live geometry authority remains the generated
Depth Coordinate V2 contract in [`host-sbs.md`](host-sbs.md); this note owns the composite model
artifact, its internal operator, and its promotion evidence.

## Decision and non-goals

Depth Anything V2 Small remains the depth estimator and the owner of the calibrated raw coordinate.
A frozen ZipDepth RGB branch predicts only the local convex reconstruction weights used to realize
DAV2 depth at exactly twice its calibrated width and height. ZipDepth's own depth head is not used.

This integration has:

- no training or fine-tuning;
- no graph cut or adaptive-J;
- no hard foreground mask or per-pixel model switch;
- no synthetic hidden-background fill; and
- no change to configured pop or the calibrated DAV2 raw scale.

The frozen ZipDepth source is commit `91f3fd21e131641f51e8d35736d1958350180e3a` with
`zipdepth_base.pth`, SHA-256
`a55910bb0b99c8c5e641cb9206e810b269690ad94e8a2ef08c827c4679391a65`. The embedded production
DAV2 ONNX SHA-256 is
`2df6223f206b5164e21f664ace61dabeb9bb6a49b8b5a3e00510b4807d0f5b04`.

## Single-high model boundary

One TensorRT engine and one enqueue expose only the high-resolution tensors:

| Tensor | Direction | FP32 shape | Meaning |
| --- | --- | --- | --- |
| `pixel_values` | input | `[1,3,2H,2W]` | Authenticated native-source RGB on the high grid |
| `refined_depth` | output | `[1,2H,2W]` | DAV2 depth reconstructed by frozen ZipDepth convex weights |

FP32 is the public I/O contract, not a claim that every internal node uses FP32. Inside the graph:

```text
high RGB [1,3,2H,2W]
  +-> FP32 AveragePool(2x2, stride 2, no padding) -> DAV2 -> coarse depth [1,H,W]
  +-> frozen ZipDepth feature/mask branch -----------------------------+
                                                                      |
coarse depth + 36 logits/cell -> 3x3 convex reconstruction -> high depth [1,2H,2W]
```

The coarse RGB and coarse DAV2 output are internal edges, not public buffers or diagnostic outputs.
The convex operator retains the released `FastConvexUpsample(scale=2, use_unfold=true)` behavior:
3x3 replicate padding, nine-neighbor softmax weights for each of four subpixels, convex sum,
row-major 2x pixel shuffle, then ReLU. Every output value therefore comes from DAV2's local 3x3
convex hull; ZipDepth selects weights but cannot invent an unrelated depth value.

The host preprocesses the retained native source directly onto the high grid with the authenticated
transfer conversion, source-cell-overlap resize or bilinear-upscale rule, ImageNet normalization,
ROI rectangle, and edge padding. For a source at least as large as the high grid, every coarse
footprint is exactly partitioned by four high cells, so FP32 2x2 averaging is mathematically the
same direct area sample up to floating-point summation order. If either native/ROI dimension is
smaller than the high target, the high preprocess uses its bilinear-upscale path; averaging that
result is not generally identical to the retired independently sampled coarse input. Small-source
and ROI cases are therefore explicit evaluation gates rather than assumed parity.

## Frozen identity and profiles

| Property | Value |
| --- | --- |
| Logical model | `prod_dav2_zipdepth_c2x_high_opset18` |
| Engine recipe | `trt-6high-point-l5-v2` |
| ONNX bytes | `74,279,879` |
| ONNX SHA-256 | `0547dd046dead55057bb34a356d987559b2d93248e84600245f02df828d8bbb7` |
| Development plan bytes | `104,277,276` |
| Development plan SHA-256 | `99165873cc27f6e4457a36a9199b4fe85afac166e3c7ef0691fa9edba95922fc` |

The plan hash is machine/runtime evidence, not a portable model identity. The ONNX hash, profile
recipe, TensorRT/GPU compatibility tag, active-engine manifest, and selected plan remain distinct.

TensorRT 11.2 cannot compile the dynamic convex tail under one ranged H/W profile. The one engine
therefore contains six exact `min=opt=max` point profiles:

| Internal DAV2 | Public input/output |
| --- | --- |
| `770x434` | `1540x868` |
| `1022x434` | `2044x868` |
| `1036x434` | `2072x868` |
| `434x770` | `868x1540` |
| `434x1022` | `868x2044` |
| `434x1036` | `868x2072` |

ONNX checker, ONNX Runtime, and TensorRT execute all six profiles with finite, exact-sized output.
Runtime profile selection is source-derived: the source is fitted once to the calibrated internal
DAV2 shape and that fit is doubled exactly. Another member of the six-profile allowlist, including
the opposite transpose, is not a valid substitute.

The runtime selects the local composite only for the exact production DAV2 identity and frozen
composite hash. Absence preserves the coherent legacy DAV2 path; a present but invalid composite
fails closed. A failed composite enqueue cannot fall back to a separately running DAV2 engine.

## One high-resolution runtime grid

For the fused path, the public output grid is also the only analysis, history, and live geometry
grid. The following all consume the same high field:

- raw moments, percentile range, normalization, and edge-aware temporal EMA;
- normalized-depth history, scene-cut evidence, camera acquisition, and near-identical ownership;
- pointwise depth-to-parallax mapping and soft container;
- source-RGB ownership correction;
- vertical and horizontal slope conditioning;
- subtitle projection/conditioning; and
- the final parallax texture sampled by inverse warp.

This prevents a high-resolution boundary from being quantized back through coarse temporal or
spatial state. Global statistics do not intrinsically require four times as many samples, but
retaining a second coarse authority would reintroduce duplicate dimensions, resources, history,
and reset rules for little measured benefit. Legacy DAV2 keeps its established coarse single grid.

The calibrated raw coordinate scale remains `2.25` DAV2 units. Runtime authentication validates
the embedded DAV2 identity against the internal `H x W` calibration while publishing and processing
the composite result at `2H x 2W`.

Near-identical reuse compares the current high-grid RGB tensor with the last authenticated infer
input. An authorized reuse skips the complete fused inference and holds the previous complete
high-grid depth/cut/camera/history tuple. The existing four-frame and strict `<100 ms` bounds remain
unchanged. Higher spatial sensitivity may reduce reuse frequency; it must never authorize a depth
result from an incompatible grid or domain.

The evidence buffer is allocated from the exact high grid as `ceil(width/16) * ceil(height/16)`
16-byte records. The six profiles require `5,335`, `7,040`, or `7,150` records, so `7,150` records /
`114,400` bytes is the current derived maximum rather than a fixed cap. The resolve shader uses
`GetDimensions` and accepts only the exact active-grid record count and 16-byte stride; any mismatch
reads no record and forces inference. Host allocation/layout failures log once with the actual
dimensions, tile grid, count, and bytes.

## ROI phase and portrait support

The high half-open content rectangle is derived exactly from the calibrated coarse fit:

```text
high_shape   = (2 * coarse_width, 2 * coarse_height)
high_content = (2 * left, 2 * top, 2 * right, 2 * bottom)
```

It is never fitted a second time. Source rectangle, authority kind/identity, and physical ROI
dimensions own the analysis domain; changing only its deterministic coarse/high raster realization
does not create a new scene.

All six landscape and portrait shapes are supported, including OCR8/SLR13 projection and Dump 3D
authentication on the three portrait high fields. The vertical limiter no longer stores four
full-height arrays. It retains the two forward-envelope arrays plus bounded carry state and
deterministically rereads immutable ownership input during replay/backward scans. The exact Q30
recurrence therefore fits the D3D11 shared-memory limit at height `2072` without approximation or
tile seams.

## Current evidence and remaining promotion gates

Completed evidence includes:

- deterministic ONNX export and frozen hash;
- all six ONNX Runtime and TensorRT profiles;
- graph parity when the retired coarse input equals exact 2x2 average of the high input;
- the single-high benchmark, live-display, and Dump 3D consumer migrations;
- complete Python evaluator regression: `769` tests passed with `12` intentional skips;
- complete native regression: `965` of `976` tests passed with `11` intentional skips, plus focused
  engine-contract, near-identical, runtime, ROI, subtitle, limiter, and live-warp coverage;
- exact `868x2072` vertical-limiter coverage;
- authenticated three-scene 4K, two-scene ROI, and 40-frame temporal dump runs;
- a 40-frame predecessor/current A/B whose high inputs are byte-identical, whose refined-depth
  mean correlation is `0.999994`, and whose final SBS mean absolute pixel delta is `0.040/255`;
- authenticated maximum-profile reuse replays at `2072x868` and `868x2072`: each exact 18-frame
  still sequence selected `6` inference frames and `12` reuse frames, with bit-exact raw ownership
  and zero control/treatment visual residual;
- an authenticated moving 40-frame `1540x868` replay that selected `30` inference frames and `10`
  reuse frames, passed raw/final-field closure, and reduced mean conditional-transaction time from
  `5.11 ms` to `3.91 ms` (`-23.6%`);
- measured steady high-grid postprocessing of about `0.48-0.58 ms`; and
- a cache-hit Galaxy XR session on the `1540x868` single grid at 30 fps with zero busy drops,
  repeats, flat frames, or ready failures in its steady intervals.

Promotion still requires representative small-source dumps and Galaxy XR inspection for foreground
edges, temporal stability, pop, scene cuts, and subtitles. The portrait computational path and
maximum portrait reuse allocation are authenticated; a representative physical portrait capture
remains useful presentation evidence rather than a missing tensor-contract test.

## Efficiency follow-up

The measured two-input/two-output precursor took approximately `4.93-5.01 ms` for profile 0 versus
about `1.61 ms` for legacy DAV2. Layer profiling attributes about 59.5% of fused latency to the
ZipDepth encoder/decoder and only 1.9% to convex reconstruction. TensorRT tiling levels 0 through 3
and auxiliary streams produced no latency improvement.

After this functional baseline is accepted, the ordered optimization experiments are:

1. selective FP16 for the ZipDepth feature/mask branch while preserving FP32 public I/O, DAV2
   depth, convex softmax, weighted sum, and final output;
2. algebraic replacement of group-4 1x1 projections by equivalent block-diagonal dense Conv/GEMM,
   plus the 1x1 spatial attention Conv by GEMM; and
3. only if those graph-level changes are insufficient, a CUDA 13.3 Tile C++ TensorRT plugin tuned
   with CompileIQ.

Every optimization remains subordinate to paired dump, convex-bound, edge, temporal, and live-XR
gates. CUDA 13.3 does not automatically accelerate the current ONNX plan.
