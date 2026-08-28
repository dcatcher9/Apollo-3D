# Production DAV2 + frozen ZipDepth convex 2x experiment

Status: Stage 1 implemented and GPU-smoke-tested; refined depth has no live rendering authority.

This note records the implementation and evidence gates for adding the released ZipDepth convex
upsampler to the production Host SBS depth path based on commit
`1a434f379b2a6d39ec9915deefa2277043cbce72`. It is intentionally separate from
[`host-sbs.md`](host-sbs.md). The canonical contract must not change until the refined live field is
implemented, authenticated, and accepted by paired evaluation.

## Decision and non-goals

The experiment keeps production Depth Anything V2 (DAV2) as the only depth estimator. A frozen
ZipDepth RGB branch predicts the standard local convex weights used to reconstruct DAV2 at exactly
2x its width and height.

This iteration has:

- no training or fine-tuning;
- no graph cut;
- no adaptive-J;
- no per-pixel hard confidence gate or foreground mask;
- no ZipDepth depth-head output; and
- no replacement of current scene-cut, camera, range, temporal-history, or pop authority.

The frozen ZipDepth source is commit `91f3fd21e131641f51e8d35736d1958350180e3a` with
`zipdepth_base.pth`, SHA-256
`a55910bb0b99c8c5e641cb9206e810b269690ad94e8a2ef08c827c4679391a65`. The frozen production DAV2
ONNX SHA-256 is `2df6223f206b5164e21f664ace61dabeb9bb6a49b8b5a3e00510b4807d0f5b04`.

## Implemented Stage 1 baseline

The branch `codex/prod-zipdepth-convex2x` now selects the local fused graph only for the exact
production DAV2 identity and frozen fused-ONNX hash. Absence of that local asset preserves the
legacy production DAV2 path; a present but invalid asset fails closed. The composite ONNX/engine
identity is published separately from the unchanged raw-DAV2 calibration identity.

The model boundary tensors are FP32. The exported graph itself remains strongly typed mixed
precision and retains the released FP16 model parameters; the TensorRT build does not globally
force every layer to FP16.

The runtime owns four atomic D3D11/CUDA tensors, independently preprocesses the retained native RGB
source at the coarse and exact 2x grids, selects the frozen point profile, binds both inputs and
outputs, and enqueues one TensorRT graph. Output allocations use checked 512-byte capacity padding
required by TensorRT while their SRVs, snapshots, and files expose only the logical float count.
TensorRT 11.2 requires a distinct optimization profile for each concurrently live execution
context, so the six shape-distinct profiles are served by one physical fused context per process.
A quarantined fused context therefore requires a process restart; legacy DAV2 and OCR retain their
existing context limits.

The current executable completed both a one-frame end-to-end smoke and a three-frame run over the
selected laundry, crystal, and additional captured scenes. Every coarse, guidance, and refined
tensor was finite. Across all three frames, every refined sample stayed within the corresponding
replicate-padded 3x3 coarse-DAV2 convex hull (zero violations at `2e-6` tolerance). The final native
focused suite passed 30/30 tests. These results establish Stage 1 runtime integrity, not visual
promotion: a Galaxy XR session still renders the unchanged coarse production geometry.

## Target one-engine boundary

One TensorRT engine and one inference enqueue own both branches:

| Tensor | Direction | FP32 shape | Meaning |
| --- | --- | --- | --- |
| `pixel_values` | input | `[1,3,H,W]` | Existing production DAV2 input |
| `zip_pixel_values` | input | `[1,3,2H,2W]` | Same-frame, native-source RGB guidance |
| `predicted_depth` | output | `[1,H,W]` | Production DAV2 coarse observation |
| `refined_depth` | output | `[1,2H,2W]` | Frozen ZipDepth-mask convex reconstruction of DAV2 |

FP32 in this table is the engine I/O contract, not a claim that every internal node executes in
FP32.

`zip_pixel_values` must be produced independently from the retained native source rectangle. It
uses the same exact source-cell-overlap area resize (or authenticated bilinear upscale when the
target exceeds the source), transfer conversion, ImageNet normalization, ROI source rectangle, and
edge padding as the production preprocessing contract, but targets the 2x grid. It must not be
obtained by enlarging `pixel_values`; that would permanently discard the high-resolution RGB
evidence the mask branch is meant to use.

The convex operator is the released `FastConvexUpsample(scale=2, use_unfold=true)` behavior:
3x3 replicate padding, nine-neighbor softmax weights for each of four subpixels, convex sum,
row-major 2x pixel shuffle, then ReLU. The weights come only from the frozen ZipDepth RGB feature
path. DAV2 supplies every depth value.

### TensorRT profile strategy

The dynamic opset-18 graph containing the full ZipDepth convex tail exports successfully and runs
all six shapes in ONNX Runtime. TensorRT 11.2 does not build it with one ranged H/W profile: Myelin
fails a constant-shape assertion in the dynamic convex tail (`value.h:872 CHECK(is_const())`).

The selected solution is still one fused graph and one engine. It contains six exact
`min=opt=max` optimization profiles in the table order below. The complete DAV2 + ZipDepth +
convex-2x plan has built without plugins, and profile indices 0 through 5 have each executed with
finite, correctly shaped coarse and refined outputs. The host must call
`setOptimizationProfileAsync(index, stream)` before setting shapes and addresses whenever a context
changes shape. The host-side shape allowlist prevents any unlisted shape from reaching TensorRT.

This point-profile strategy avoids both the Myelin limitation and the memory cost of exposing a
36-channel mask tensor. A standalone logits-to-convex shader remains a diagnostic fallback only;
it is not part of the selected engine I/O, live producer closure, or cache recipe.

The deterministic fused ONNX is 74,279,734 bytes with SHA-256
`959fc90097d7055b9c56cb140f432e0f5aed533476e8cedd6ec2baae097b287f`.
The export strips non-executable PyTorch source-path/line annotations and deduplicates the identical
default-domain opset import before deterministic serialization. Re-exporting from the frozen
sources reproduces the same bytes and hash.

On the development RTX 5080, TensorRT 11.2.1 builder level 5 produced a 104,710,692-byte plan and
executed every profile. Direct paired execution of the current legacy and fused plans on the exact
same captured `model_input.f32` produced the following `770x434` coarse-DAV2 differences:

| Captured scene | MAE | p95 | p99 | maximum | correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| laundry/person ROI | `0.000989889` | `0.00390625` | `0.00976563` | `0.0415039` | `0.999999293` |
| crystal | `0.000899170` | `0.00390625` | `0.00695801` | `0.0317383` | `0.999999494` |
| second person ROI | `0.000871194` | `0.00390625` | `0.00781250` | `0.0371094` | `0.999999362` |

These measurements avoid a false comparison between a live dump's ROI-bound model input and a
new full-screen PNG preprocessing pass. They are useful profile-0 evidence, but they do not replace
the required paired legacy/fused validator across all six profiles. Until that gate and a tolerance
policy are executable, this branch remains experimental rather than production-promoted. The plan
SHA-256 was
`63b1c7a7200600b7a8c7c5ab8821e096e3d4034043edcc4a65e87367364071a5`. The plan hash is
machine/runtime evidence, not a portable model identity; the ONNX,
profile recipe, TensorRT/GPU compatibility tag, and resulting plan hash must all remain recorded.

## Stage 1: real fused inference, diagnostic refined output

Stage 1 integrates the complete two-input/two-output TensorRT, D3D11/CUDA interop, preprocessing,
convex reconstruction, and dump path. Live Host SBS continues to consume `predicted_depth` exactly
where production does today. The reconstructed 2x depth has no renderer authority in this stage.
The reconstructed tensor is the engine's `refined_depth` output.

The existing coarse path remains responsible for:

- normalization and depth range;
- moments, scene cuts, EMA, and near-identical reuse;
- camera/convergence state and requested pop mapping;
- history ownership; and
- every current V2 geometry texture and renderer input.

An ordinary `sbsbench` evaluation records the stable coarse input/output and the new 2x
guidance/refined output as separate artifacts. `raw_<frame>.f32` retains its existing coarse-DAV2
meaning; the refined tensor must not be relabeled as raw DAV2. The live Dump 3D package does not yet
publish these two new diagnostic files.

Adding the Zip branch can change TensorRT tactic selection even when the DAV2 graph branch is
unchanged. Production equivalence is therefore a measured gate, not an assumed bit-identity claim.
The fused `predicted_depth` must pass the agreed production parity test for every authenticated
shape before Stage 1 is accepted.

### Stage 1 gates

The operator and fused engine must pass all six current production shapes:

| Coarse DAV2 | Refined output |
| --- | --- |
| `770x434` | `1540x868` |
| `1022x434` | `2044x868` |
| `1036x434` | `2072x868` |
| `434x770` | `868x1540` |
| `434x1022` | `868x2044` |
| `434x1036` | `868x2072` |

Required evidence includes:

- exact four-tensor names, directions, FP32 types, and 1x/2x shape relation;
- ONNX Runtime and TensorRT parity with the official PyTorch convex operator;
- fused coarse-output parity with the production DAV2 engine;
- full-frame and ROI cases, including odd content extents;
- finite output and local convex-bound checks;
- stable dump/replay of both new tensors; and
- interop registration, pointer-change graph recapture, teardown, and fail-flat tests.

These six shapes describe operator and engine support. They do **not** imply that all six are ready
to become refined live-rendering fields; the portrait limiter blocker below is independent.

## Exact ROI phase

Convex 2x pixel shuffle gives every coarse cell one fixed 2x2 phase. The refined shape and half-open
content rectangle are therefore derived only by multiplying every coordinate by two:

```text
field_shape   = (2 * coarse_width, 2 * coarse_height)
field_content = (2 * left, 2 * top, 2 * right, 2 * bottom)
```

For example, authenticated content `770x433` becomes `1540x866` inside a `1540x868` field. The
refined ROI must not be run through the aspect-ratio fitter again: integer flooring can move an odd
edge and change the pixel-shuffle phase.

This deterministic field realization is not a new analysis domain. Source rectangle, coarse tensor
content, authority, analysis generation, scene state, and camera state remain coarse-domain
identity. Merely enabling the refined field must not trigger a scene/domain reset.

## Stage 2: coarse analysis, refined geometry

Only after Stage 1 passes does `refined_depth` enter the live spatial path. The split occurs before
the nonlinear V2 coordinate curve:

```text
coarse predicted_depth
  -> moments / range / cut / EMA / camera-state / history
  -> retained coarse fallback

refined_depth + coarse camera-state
  -> raw-depth coordinate mapping and V2Curve at 2x
  -> existing RGB ownership refinement at 2x
  -> vertical limiter at 2x
  -> horizontal limiter at 2x
  -> final 2x parallax field
```

Mapping or upsampling after `V2Curve` is not equivalent because the mapping is nonlinear. Likewise,
upsampling the already limited parallax field is unsafe: spatially varying convex weights do not
preserve the authenticated horizontal-slope bound used by the inverse warp.

The current production RGB ownership pass remains enabled after refined mapping. It is a bounded
source-contour ownership refinement, not graph cut or adaptive-J. An A/B diagnostic may isolate its
effect, but the planned production treatment is existing ownership plus convex 2x. The existing
ownership shader safely no-ops when the captured source is smaller than the refined field; thus a
1280x720 source feeding the 1540x868 live field receives convex refinement but no additional RGB
ownership snap. Supporting that case requires an explicit source-to-field sampling contract, not
an unchecked relaxation of the current bounds guard.

Stage 2 requires explicit coarse-analysis and refined-field dimensions throughout the estimator,
GPU executor, textures, OCR/subtitle projection, indirect dispatch records, result authentication,
dump metadata, and renderer ROI constants. `raw_width/raw_height` continue to mean coarse DAV2;
new field dimensions describe the final parallax textures.

OCR remains source/detector-domain analysis and its projected boxes naturally land on the refined
grid. SLR13 uses the generated field-cell scale (`1` for coarse, `2` for exact convex-2x): fixed
minimum line sizes, bottom-corner proximity, and target-probe offsets double on a live field, while
relative width/aspect rules and detector-grid topology thresholds stay unchanged. This preserves
the same physical subtitle and support-plane footprint instead of silently halving it at 2x.

## Portrait limiter blocker

The current vertical limiter allocates several full-height `groupshared` arrays bounded by the
generated maximum dimension `1036`. A refined portrait field can be `868x2072`; simply raising that
constant would require roughly 50 KiB of group-shared storage and exceed the D3D11 compute-shader
limit. The horizontal limiter uses fewer arrays and can support a refined width of 2072 with a
separate bound.

Consequently:

- Stage 1 engine/operator diagnostics cover all six shapes.
- Initial Stage 2 live geometry may cover the three landscape shapes, whose refined height is 868.
- Portrait refined live geometry remains unsupported until the vertical limiter is redesigned as
  an exact tiled or multi-pass scan with authenticated scratch/publication behavior.
- Portrait sessions remain on the complete coarse production geometry path; no limiter is bypassed.

## Fallback and publication

Mode is selected and latched for an estimator/session before renderer publication:

- `convex2x` is selected only when the fused engine, refined resources, shader closure, shape, ROI,
  orientation, and limiter contract all authenticate.
- Otherwise the entire session uses the coherent coarse production geometry path.
- A successful fused completion may fall back atomically to its own same-frame coarse output if the
  refined result fails validation.
- If the single fused engine, enqueue, CUDA mapping, or completion fails, both outputs are
  untrusted and existing fail-flat behavior applies. A separate standby DAV2 engine is deliberately
  outside this one-engine plan.
- There is no per-pixel mixture of refined and coarse depth, no stale-mask/current-depth pairing,
  and no per-frame field-resolution switching.
- Existing bounded submission and reuse behavior remains in force; busy inference holds/drops work
  according to the production transaction rather than queuing TensorRT indefinitely.

## Promotion rule

Do not modify the canonical Host SBS contract merely because Stage 1 produces a plausible refined
dump. Promotion requires the fused parity gates, paired sbsbench control/treatment from the same
pinned numeric runtime, slope and inverse-warp tests, scene-cut/pop continuity evidence, and Galaxy
XR live validation. The accepted live shape/orientation set and composite DAV2/ZipDepth/export
provenance must then be generated into the canonical C++/HLSL/Python contracts together.
