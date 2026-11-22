# Production DAV2 + ZipDepth convex 2x

Since 2026-11-21 the composite carries fine-tuned weights (`bt7`) and feeds DAV2 a larger input;
[Fine-tuned weights](#fine-tuned-weights-bt7) owns that change. Sections below that describe the
stock composite's export, timing and promotion evidence are kept as the record of the topology,
which is unchanged.

Status: single-high-I/O implementation candidate with the model-only optimization recipe integrated.
The original deterministic ONNX export, six-profile TensorRT build, authenticated model parity,
native build, all-high runtime, and six-shape limiter path are clean. The phone/tablet extension
retains those ONNX bytes and expands the fixed profile recipe to the shapes owned by
[Authenticated resolution fitting](host-sbs.md#authenticated-resolution-fitting).
A paired evaluator report and explicit Galaxy
XR visual acceptance remain required before this optimized artifact becomes the published production
baseline.

This note records the implementation based on production commit
`1a434f379b2a6d39ec9915deefa2277043cbce72`. The live geometry authority remains the generated
Depth Coordinate V2 contract in [`host-sbs.md`](host-sbs.md); this note owns the composite model
artifact, its internal operator, and its promotion evidence.

## Decision and non-goals

Depth Anything V2 Small remains the depth estimator and the owner of the calibrated raw coordinate.
A frozen ZipDepth RGB branch predicts only the local convex reconstruction weights used to realize
DAV2 depth at exactly twice its calibrated width and height. ZipDepth's own depth head is not used.

This integration has:

- no new layers: the fine-tune changes weights, the ViT input size and one output gain only;
- no graph cut or adaptive-J;
- no hard foreground mask or per-pixel model switch;
- no synthetic hidden-background fill; and
- no change to configured pop or the calibrated DAV2 raw scale.

The frozen ZipDepth source is commit `91f3fd21e131641f51e8d35736d1958350180e3a` with
`zipdepth_base.pth`, SHA-256
`a55910bb0b99c8c5e641cb9206e810b269690ad94e8a2ef08c827c4679391a65`. The embedded production
DAV2 ONNX SHA-256 is
`2df6223f206b5164e21f664ace61dabeb9bb6a49b8b5a3e00510b4807d0f5b04`.

The serialized ZipDepth branch is the released base encoder/decoder, not a generic depth model
substitute. Its encoder stage widths are `[48,96,192,384]`, stage depths are `[2,2,6,2]`, decoder
widths are `[288,192,144,96,32]`, and the mask head emits 36 logits. The two 1x1 operators after
global mean pooling form channel attention on a `[1,192,1,1]` tensor; they are not a spatial
attention convolution. ZipDepth's depth head remains absent.

## Fine-tuned weights (`bt7`)

The stock composite flattens thin objects: on unseen synthetic scenes it recovers about 8% of the
true relief of a 5-8 px wide object. The cause is DAV2's input resolution, not the reconstruction
step. The tuned composite differs from the stock one in four ways, all applied by
`tools/sbsbench/prod_zipdepth_tuning.py` during the export:

| Change | Detail |
| --- | --- |
| DAV2 input | bilinear resize of the high grid to `round(extent * 0.6875 / 14) * 14` per axis (1064x602 for 1540x868) instead of the 2x2 average; computed inside the graph from the bound shape, so one graph serves all 24 profiles |
| DAV2 final feature resize | targets the half grid, so `predicted_depth` and everything after it keep their shapes |
| Weights | 60 DAV2 decoder-head tensors written by name, 48 ViT linear weights (folded rank-8 adapters) written over the stock weight they replace, and a tuned ZipDepth branch |
| Output gain | `refined_depth` is multiplied by `1.0258`, which keeps the stock composite's raw range and therefore the depth-coordinate calibration |

The exported composite is checked in as
`src_assets/windows/assets/prod_dav2_zipdepth_c2x_high_opset18.onnx`, with
`prod_dav2_zipdepth_c2x_high_opset18.provenance.json` beside it and the Depth Anything V2 Small
(Apache-2.0) and ZipDepth (MIT) licence texts under `assets/licenses`. The build copies it into its
assets and the installer packages it, so a host needs no separate model step.
`tools/sbsbench/test_prod_zipdepth_composite_asset.py` holds the file, the contract, the C++
identity and the provenance record together. The provenance record also states what the weights
were tuned with: the teacher was Depth Anything V2 Base, a CC-BY-NC-4.0 model whose depth output
served as a training target; none of its weights are in the file.

The export's inputs stay outside git: the tuned checkpoint `ckpt_bt7.pt` (126,531,328 bytes, SHA-256
`a08834f278e516f4ecfe179e9901d1633a318b97781a8d8f5aae7499d88f5f95`) and the stock DAV2 PyTorch
weights `depth_anything_v2_vits.pth` (SHA-256
`715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`), next to the stock DAV2 ONNX and
ZipDepth repository named above. The contract records them under `sources.tuning`, and the exporter
refuses any other file:

```text
python tools/sbsbench/export_prod_zipdepth_convex2x.py --dav2-onnx <depth_anything_v2_fp16.onnx>
  --zipdepth-repo <ZipDepth> --zipdepth-checkpoint <zipdepth_base.pth>
  --tuned-checkpoint <ckpt_bt7.pt> --dav2-torch-checkpoint <depth_anything_v2_vits.pth>
  --output <dir> --skip-engine
```

The export is deterministic: it reproduces the fused file of the identity table byte for byte.

Measured against the stock composite (evidence and method in
`reports/thin-object-depth-2026-11-20.md` of the development checkout):

- thin-object relief on 160 unseen frames: 5-8 px objects `0.078` to `0.180`, under 5 px `0.009`
  to `0.039`; depth-edge F1 `0.41` to `0.52`;
- black bars: a pillarboxed or letterboxed picture is as deep as the same picture without bars
  (the stock composite read a bar as a near wall and halved the relief);
- TensorRT time at 1540x868: `2.5 ms` to `4.1 ms` (live session p50 `4.1 ms`, p95 `5.1-5.5 ms`);
- `run_eval.py` against the stock composite with the same warp: core suite `screen_candidate`
  (static jitter lower on seven clips, nothing regressed); extended suite `tradeoff` (depth-edge F1
  `58.8` to `76.8`, `54.1` to `64.6` and `47.5` to `54.9` on three clips, error on
  `tartanair_house_motion` `10.6%` to `8.8%`, error on `sintel_ambush` `13.1%` to `15.3%`); no
  conformance failures. `vkitti_drive_clone` measures `3.94%` in `exact_negative_disparity_pct`,
  for which the bound was relaxed from `3.5` to `4.25`;
- native tests with `APOLLO_RUN_TENSORRT_TESTS=1`: all pass, including
  `TensorRtConditionalWrapperGpuTest.AllAuthenticatedShapesBuildAndExecute` for the 24 profiles;
- the released file has the same initializer payloads and operations as the research export that
  was viewed live on a Galaxy XR on 2026-11-21, and its TensorRT output differs from that export's
  by less than two engine builds of one file differ.

Known limits: bright far regions seen through an opening (windows, sky) are judged too near, a
bias learned from the fine-tune's teacher; relief of objects under 5 px stays low in absolute
terms; training and scoring data are rendered 3D and film animation, not 2D art.

## Single-high model boundary

One TensorRT engine and one enqueue expose only the high-resolution tensors:

| Tensor | Direction | FP32 shape | Meaning |
| --- | --- | --- | --- |
| `pixel_values` | input | `[1,3,2H,2W]` | Authenticated native-source RGB on the high grid |
| `refined_depth` | output | `[1,2H,2W]` | DAV2 depth reconstructed by ZipDepth convex weights, times the output gain |

FP32 is the public I/O contract, not a claim that every internal node uses FP32. Inside the graph:

```text
high RGB [1,3,2H,2W]
  +-> FP32 bilinear Resize to the ViT grid [1,3,Hv,Wv] -> DAV2 -> coarse depth [1,H,W]
  +-> ZipDepth feature/mask branch ------------------------------------+
                                                                      |
coarse depth + 36 logits/cell -> 3x3 convex reconstruction -> high depth [1,2H,2W]
```

The production public boundary remains FP32. DAV2 was already mixed precision in its frozen source:
FP32 input, an entry Cast to FP16, an FP16 DINOv2/DPT body, and an exit Cast restoring FP32
`predicted_depth`. The postpass does not alter any DAV2 node or initializer. It adds an independent
FP32-to-FP16 Cast at the ZipDepth RGB seam, converts exactly 88 ZipDepth feature/mask initializers
(`6,143,503` elements) to FP16, and restores the 36 mask logits to FP32 before reshape, softmax, and
the convex reconstruction tail. `predicted_depth`, all eleven convex-tail node outputs, and
`refined_depth` therefore remain FP32.

Two exact algebraic rewrites follow that precision boundary. Four bias-free decoder low-path 1x1
projections execute before bilinear resize, reducing their projection sites by about 3.97x without
changing linear real-number semantics. TensorRT otherwise lowers each of ten group-4 pointwise
convolutions into four loop layers, so their FP16 weights are expanded to exact block-diagonal dense
weights and `group` is changed from 4 to 1. Live blocks are byte-identical and every off-block value
is positive zero. The cross-scale pool-before-project rewrite, attention-GEMM, FP8 Q/DQ, and custom
plugins are not part of the release graph.

The coarse RGB and coarse DAV2 output are internal edges, not public buffers or diagnostic outputs.
The convex operator retains the released `FastConvexUpsample(scale=2, use_unfold=true)` behavior:
3x3 replicate padding, nine-neighbor softmax weights for each of four subpixels, convex sum,
row-major 2x pixel shuffle, then ReLU. Every output value therefore comes from DAV2's local 3x3
convex hull; ZipDepth selects weights but cannot invent an unrelated depth value.

The host preprocesses the retained native source directly onto the high grid with the authenticated
transfer conversion, source-cell-overlap resize or bilinear-upscale rule, ImageNet normalization,
ROI rectangle, and edge padding. In the stock composite, for a source at least as large as the
high grid, every coarse footprint was exactly partitioned by four high cells, so FP32 2x2 averaging
was mathematically the same direct area sample up to floating-point summation order. The tuned
composite instead resizes the high grid bilinearly to the ViT grid, without an antialiasing
prefilter; that is the input the weights were tuned with, and it is not an area sample of the
source. Small-source and ROI cases remain explicit evaluation gates.

## Frozen identity and profiles

| Property | Value |
| --- | --- |
| Logical model | `prod_dav2_zipdepth_c2x_high_opset18` |
| Engine recipe | `trt-24high-point-l5-v3` |
| Raw ZipDepth branch bytes / SHA-256 | `24,637,932` / `2c257b721fc4c6f13cd83be7cd03a7d012b1b77cb50fc82599b5938592ceb252` |
| Optimized ZipDepth branch bytes / SHA-256 | `12,796,840` / `c4278ba6f5b9a82a5e5a4d6f644b9eed7e2350e62e7e22136113c95313886c4c` |
| Fused ONNX bytes | `62,344,826` |
| Fused ONNX SHA-256 | `134d92fa67ca746c4e03a42cc85666a7802bf459ae2245edc70d16d398d6ea05` |
| Stock composite before the fine-tune (fused bytes / SHA-256) | `62,438,471` / `26684c5da8fdd4bdc5f1c9cf919cec8d1e2d027fbe95705a454f85d31eee2c23` |
| Six-profile development plan bytes | `108,190,812` |
| Historical six-profile development plan SHA-256 | `a16421b2972165efc88bf12f5740826f5769abcf832ed8fba2def6efc64b4fba` |

The plan hash is machine/runtime evidence, not a portable model identity. The ONNX hash, profile
recipe, TensorRT/GPU compatibility tag, active-engine manifest, and selected plan remain distinct.

TensorRT 11.2 cannot compile the dynamic convex tail under one ranged H/W profile. The one engine
therefore contains 24 exact `min=opt=max` point profiles. The original six retain their profile
indices, followed by the nine added landscape shapes and then their portrait transposes. The
complete source/coarse/high mapping is owned by
[Authenticated resolution fitting](host-sbs.md#authenticated-resolution-fitting).
The recipe change invalidates the old six-profile plan; the host builds and caches the expanded
engine on first use. Model bytes, precision, optimization level, and the exact 2x operator remain
unchanged.

Historical ONNX checker, ONNX Runtime, and TensorRT evidence below covers the original six profiles.
The native `TensorRtConditionalWrapperGpuTest.AllAuthenticatedShapesBuildAndExecute` gate covers
every current profile through the real estimator, joined completion, and authenticated publication.
Runtime profile selection is source-derived: the source is fitted once to the calibrated internal
DAV2 shape and that fit is doubled exactly. Another member of the profile allowlist, including
the opposite transpose, is not a valid substitute.

The runtime requires the local composite for the exact production DAV2 identity and frozen
composite hash. Absence, a non-regular file, or any identity mismatch fails flat. A failed composite
enqueue cannot fall back to a separately running DAV2 engine.

## One high-resolution runtime grid

The public fused output grid is also the only analysis, history, and live geometry
grid. The following all consume the same high field:

- raw moments, percentile range, normalization, and edge-aware temporal EMA;
- normalized-depth history, camera state, and near-identical history ownership;
- pointwise depth-to-parallax mapping and soft container;
- vertical and horizontal slope conditioning;
- subtitle projection/conditioning; and
- the final parallax texture sampled by inverse warp.

This prevents a high-resolution boundary from being quantized back through coarse temporal or
spatial state. Global statistics do not intrinsically require four times as many samples, but
retaining a second coarse authority would reintroduce duplicate dimensions, resources, history,
and reset rules for little measured benefit.

The calibrated raw coordinate scale remains `2.25` DAV2 units. Runtime authentication validates
the embedded DAV2 identity against the internal `H x W` calibration while publishing and processing
the composite result at `2H x 2W`.

Near-identical reuse compares the current high-grid RGB tensor with the last authenticated infer
input. An authorized reuse skips the complete fused inference and holds the previous complete
high-grid depth/camera/history tuple. Reuse has no age or frame-count expiry and never promotes
the candidate to the comparison owner. Higher spatial sensitivity may reduce reuse frequency; it
must never authorize a depth result from an incompatible grid or domain.

The evidence buffer is allocated from the exact high grid as `ceil(width/16) * ceil(height/16)`
16-byte records. The largest supported profile still requires `7,150` records, so `7,150` records /
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

All authenticated landscape and portrait shapes are supported, including OCR8/SLR13 projection
and Dump 3D authentication. The vertical limiter no longer stores four
full-height arrays. It retains the two forward-envelope arrays plus bounded carry state and
deterministically rereads the immutable candidate during replay/backward scans. The exact Q30
recurrence therefore fits the D3D11 shared-memory limit at height `2072` without approximation or
tile seams.

## Current evidence and remaining promotion gates

The list below is the evidence of the stock composite and its topology. Evidence for the tuned
weights is in [Fine-tuned weights](#fine-tuned-weights-bt7); the parity items about the 2x2 average
do not apply to the tuned input path.

Completed evidence includes:

- deterministic ONNX export and frozen hash;
- an authenticated architecture audit proving all `888` converted DAV2 nodes and `512` DAV2
  initializers remain unchanged after the expected input rename;
- canonical exporter output bit-exact to the selected experimental graph on three authenticated
  inputs, with CPU correlation `0.9999999874`, mean absolute error `0.0000390`, relative L2
  `0.0000907`, and maximum absolute error `0.02859` against the prior FP32-ZipDepth graph;
- a maximum convex-bound violation of `1.43e-6`, with zero values above `1e-5`, and bit-exact repeat
  inference;
- strict ONNX checker/type/shape inference and all six TensorRT profiles;
- graph parity when the retired coarse input equals exact 2x2 average of the high input;
- the single-high benchmark, live-display, and Dump 3D consumer migrations;
- complete Python evaluator regression: `775` tests passed with `12` intentional skips;
- focused native fused-model/runtime regression: `17` tests passed;
- complete native regression: `965` of `976` tests passed with `11` intentional skips, plus focused
  engine-contract, near-identical, runtime, ROI, subtitle, limiter, and live-warp coverage;
- exact `868x2072` vertical-limiter coverage;
- authenticated three-scene 4K, two-scene ROI, and 40-frame temporal dump runs;
- a 40-frame predecessor/current A/B whose high inputs are byte-identical, whose refined-depth
  mean correlation is `0.999994`, and whose final SBS mean absolute pixel delta is `0.040/255`;
- authenticated maximum-profile reuse replays at `2072x868` and `868x2072`: each exact 18-frame
  still sequence selected `6` inference frames and `12` reuse frames, with bit-exact raw output
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

## Model-only optimization result

The frozen FP32-ZipDepth six-profile control averaged `4.93048 ms` at profile 0. Three fresh
interleaved control/treatment pairs put the canonical optimized six-profile engine at `2.70093 ms`,
a `45.22%` reduction and `1.825x` speedup; all three pairs won. The fused ONNX shrank by
`11,841,408` bytes (`15.94%`). The block-diagonal dense weights make the portable six-profile plan
`3,913,536` bytes larger (`3.75%`), an accepted trade for removing TensorRT's grouped-convolution
loop lowering. One inference at every landscape and portrait profile produced the exact output
shape with finite values. Three authenticated TensorRT outputs retained minimum correlation
`0.999999765`, maximum per-frame mean absolute error `0.000930`, and maximum absolute error
`0.03869` against the frozen control engine.

Rejected experiments remain excluded from the deterministic exporter:

- moving the cross-scale average pool before projection regressed matched TensorRT latency by
  `1.97%`;
- replacing the channel-attention 1x1 convolutions with GEMM won only two of three pairs, with a
  noise-scale `0.18%` nominal mean;
- generic Model Optimizer autocast rewrote DAV2 and raised mean error by `7.37x` versus the guarded
  FP16 conversion;
- surgical FP8 Q/DQ required opset 19 and reached holdout relative L2 `0.00477` plus mask-logit
  relative L2 `0.154`; and
- a CUDA Tile TensorRT plugin is a runtime/kernel extension and therefore outside this session's
  serialized-model-only boundary.

The optimization remains subordinate to paired dump, edge, temporal, subtitle, and
live-XR gates. CUDA 13.3 does not automatically accelerate the ONNX plan.
