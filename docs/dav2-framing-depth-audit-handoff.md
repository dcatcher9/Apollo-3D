# DAV2 fullscreen-versus-windowed depth investigation

## Audit handoff

**Date:** 2026-08-01  
**Repository:** `E:\Git\Repo\Apollo-3D`  
**Branch / revision:** `master` at `9f90809a` (`revert(sbs): remove adaptive ROI pipeline`)  
**Scope:** One Bilibili scene captured once in a browser player and once in fullscreen, followed by raw-model, normalization, and warp replays. No production code was changed during this investigation.

## Question

The windowed Bilibili player looked better in Host SBS than fullscreen. The fullscreen result had a flatter, more cutout-like subject and more jagged/disocclusion artifacts around the hair, even though fullscreen gives the video more of the fixed 770x434 depth tensor.

The surprising questions were:

1. Is `raw_depth.f32` truly the direct Depth Anything V2 output, or is it already normalized or otherwise postprocessed?
2. Is the fullscreen difference caused by the source pixels, preprocessing, normalization, or DAV2's response to framing?
3. Which conclusions are established, and which remain hypotheses?

## Executive conclusion

1. **Yes: `raw_depth.f32` is a stable copy of TensorRT's exported FP32 `predicted_depth` output.** It is captured before Apollo's P2/P98 normalization, temporal EMA, subject recentering, Bestv2 curve, warp prefilter, or reprojection. An exact-input CPU ONNX replay correlates with the saved TensorRT data above 0.99998 for both dumps.
2. **The raw model geometry already differs materially.** In fullscreen, hair becomes much more uniform relative to the person/background step. The postprocessing pipeline amplifies this, but does not originate it.
3. **This scene shows a combined scale-and-context effect.** Making the video smaller restores more within-hair variation; the real browser UI also materially softens the hair/background relationship. Enlarging the windowed player crop back to fullscreen makes its raw result fullscreen-like.
4. **This does not prove that neutral padding is a general fix.** A neutral surround helped some affine-invariant ratios but did not reproduce the real browser result. The browser context is part of the measured effect.
5. **The ONNX bisection identifies the internal path.** Identical player tokens first diverge in block 0 self-attention, and the useful cross-region attention concentrates around block 8 head 2 and block 7 head 3. Activation splicing is more decisive: the deepest DPT tap controls the result, with outside-player tap features carrying about 77% of the visible softening. Pixel ablation attributes most source evidence to the horizontal UI below the player.
6. **The evidence covers one scene.** It proves DAV2 is non-invariant to this manipulation, not that every fullscreen video will fail or every browser frame will help.

## Evidence package

### Original matched Dump 3D packages

- Windowed: `E:\ApolloDev\sbs_dump\dump_20260801_001127_2972_18c79ce5b301c190_2`
- Fullscreen: `E:\ApolloDev\sbs_dump\dump_20260801_001132_2972_18c79ce6ede9ef5c_3`

Both manifests report:

- model: `depth_anything_v2_fp16`
- source: 3840x2160 FP16 scRGB HDR
- exact model input: 770x434 FP32 NCHW
- raw model output: 770x434 FP32
- CUDA graph active
- atomic matched-frame publication

The frames are **not identical captures**. Their matched IDs are 10539 and 10636, 97 source-frame identities apart. The scene is nearly static and aligns closely, but the original pair alone cannot establish causation. The controlled counterfactual tests below are required for that.

### Reproducibility hashes

| Artifact | SHA-256 |
|---|---|
| `cmake-build-relwithdebinfo/assets/depth_anything_v2_fp16.onnx` | `2DF6223F206B5164E21F664ACE61DABEB9BB6A49B8B5A3E00510B4807D0F5B04` |
| window `model_input.f32` | `85009165E7D053B68DEC1369E873EA47D3D87E3D2C63ED4B3B4DC7339CA03870` |
| window `raw_depth.f32` | `8F2CCD5F4ABA92C11478ED12B193786C6AC740B76F4570215D11C0E3BC6D053C` |
| fullscreen `model_input.f32` | `10ABD8F4633981DA63A3440243C346328C24CFD79EE5B402D992B30EC39F222E` |
| fullscreen `raw_depth.f32` | `655E0C37AB9F22D644DC0A8420529F6B905FE7F75DEA1870ADA05F16DD47A590` |

## Is `raw_depth.f32` direct model output?

### Answer: yes, with one terminology caveat

Here, **raw model output** means the exported ONNX/TensorRT `predicted_depth` tensor after all layers, decoder, and upsampling inside the DAV2 model graph. It does not mean an internal encoder feature or pre-decoder logit.

It receives no Apollo depth transformation before capture.

### Source trace

The relevant data path is:

```text
captured scRGB frame
  -> rgb_to_nchw_cs.hlsl
  -> tensor_in_buf / TensorRT "pixel_values"
  -> TensorRT enqueueV3
  -> tensor_out_buf / TensorRT "predicted_depth"
       |-> immutable raw_snapshot_buf -> raw_depth.f32
       `-> read-only normalization input -> depth_tex -> warp prefilter -> SBS warp
```

Evidence in the current source:

1. `src/video_depth_estimator.cpp:394-424` validates that both `pixel_values` and `predicted_depth` use the expected FP32 I/O contract.
2. `src/video_depth_estimator.cpp:2691-2717` obtains the CUDA output pointer and binds `predicted_depth` directly to it before `enqueueV3`.
3. `src/video_depth_estimator.cpp:2098-2189` binds `tensor_out_buf` only as a shader resource while normalization writes separate min/max, histogram, and `depth_tex` UAV resources. The raw buffer is not modified.
4. `src/video_depth_estimator.cpp:2614-2641` snapshots the completed `tensor_out_buf` and its paired `tensor_in_buf` before those allocations are reused for the next frame.
5. `src/video_depth_estimator.cpp:1977-2022` implements that snapshot as a D3D11 `CopyResource`, not a transform.
6. `src/platform/windows/display_vram.cpp:660-671` associates a completed inference ID with its buffered color. Lines 829-895 require the matched render slot and both immutable snapshots before passing them to Dump 3D.
7. `src/platform/windows/sbs_debug_dump.cpp:361-421` stages and copies the structured buffer unchanged. Lines 847-914 write those float values directly to `raw_depth.f32`.
8. `src/platform/windows/sbs_debug_dump.cpp:1725-1747` declares `raw_depth.f32` as `exact model output` and distinguishes the preview files.

The CUDA graph does not change this provenance. It captures/replays TensorRT `enqueueV3`, not Apollo's normalization kernels (`src/video_depth_estimator.cpp:1379-1442`).

### Important preview warning

`raw_depth.png` and `raw_depth_heat.png` are **not** numeric contracts. Each file is independently stretched using that dump's finite P2/P98 bounds. Two equally colored pixels across two heatmaps need not have the same raw value, and the previews can visually hide affine scale/offset differences.

Use `raw_depth.f32` for quantitative work. The current files contain 334,180 finite float32 values each:

| Dump | Raw min | Raw max | Preview P2 | Preview P98 |
|---|---:|---:|---:|---:|
| Windowed | 0.777344 | 4.671875 | 1.079102 | 4.507813 |
| Fullscreen | 0.426514 | 4.031250 | 0.616211 | 3.873047 |

The files are also plainly distinct from `depth.f32`: raw values extend outside `[0,1]`, while normalized and warp depth are in `[0,1]`.

### Runtime shader-path trap

Do not audit `assets/shaders/directx/rgb_to_nchw_cs.hlsl` at the repository root. That copy is stale and is not the Windows runtime source.

The active Windows shader is:

`src_assets/windows/assets/shaders/directx/rgb_to_nchw_cs.hlsl`

It performs exact area integration for 3840x2160 -> 770x434, followed by HDR tone mapping, sRGB conversion, and ImageNet normalization. Build/packaging selects `src_assets/windows/assets`; see `cmake/prep/constants.cmake` and `cmake/packaging/windows.cmake`.

## Independent exact-input model replay

To avoid trusting source comments or manifest labels, each exact `model_input.f32` was sent through the same ONNX file using ONNX Runtime CPU, and the output was compared with the saved TensorRT `raw_depth.f32`.

Because monocular relative depth and different execution backends can differ by a small affine transform, the comparison reports both correlation and residual after fitting one global `scale * prediction + offset`.

| Dump | Correlation | Fitted scale | Fitted offset | Affine MAE / saved P2-P98 span | Affine p95 / span |
|---|---:|---:|---:|---:|---:|
| Windowed | 0.999997664 | 1.002309 | -0.004850 | 0.0591% | 0.1559% |
| Fullscreen | 0.999987454 | 1.000154 | -0.005901 | 0.0812% | 0.2821% |

This independently rules out accidentally dumping `depth.f32`, applying an orientation change, or comparing only normalized previews.

Minimal reproduction outline:

```python
x = np.fromfile(dump / "model_input.f32", dtype="<f4").reshape(1, 3, 434, 770)
saved = np.fromfile(dump / "raw_depth.f32", dtype="<f4").reshape(434, 770)
session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
pred = session.run(None, {"pixel_values": x})[0].squeeze()

# Fit only for backend/relative-depth comparison.
A = np.c_[pred.ravel(), np.ones(pred.size)]
scale, offset = np.linalg.lstsq(A, saved.ravel(), rcond=None)[0]
residual = scale * pred + offset - saved
```

## What differs between the two original model inputs?

An RGB alignment fitted the windowed player onto the fullscreen tensor at approximately:

- origin: `(55.56, 83.42)` in the 770x434 tensor
- scale: `0.6596 x 0.6599`
- player extent: about 508x286 pixels, approximately 43.5% of the tensor area
- approximate patch coverage: 36x21 DAV2 patches versus the fullscreen 55x31 patch grid

After alignment, the underlying scene pixels are very close:

- median absolute RGB difference: about 0.18% in sampled background and 0.29% in hair
- fullscreen RGB gradients are only roughly 11-18% stronger in the sampled scene regions

The raw geometry changes much more than those pixel differences:

- after a robust affine match, fullscreen hair/background median separation is 2.44 versus 1.12 windowed: **2.18x stronger**
- the raw hair-boundary gradient p90 is 0.868 fullscreen versus 0.259 windowed: **3.35x stronger**

These original-capture measurements are descriptive, not causal, because the frame IDs differ. They motivated the controlled tests.

## Controlled framing experiments

Two independent paths were used:

1. **CPU ONNX counterfactuals** built from the dumped model-input previews. This is fast and produces diagnostic heatmaps, but the 8-bit `model_input.png` introduces small quantization. Replaying those previews versus exact-float ONNX output changes affine-aligned depth by only about 0.10% of the robust span, which is negligible relative to the observed effect.
2. **Native TensorRT harness counterfactuals** re-fed six generated 770x434 images through Apollo's real model path. This is the stronger cross-check for raw geometry.

### Native TensorRT result

The safest exploratory statistic was:

```text
hair raw P5-P95 span / (hair median - sampled background median)
```

This ratio measures within-hair variation relative to the silhouette step and is less misleading than independently P2/P98-normalizing every variant.

| Variant | Ratio | Interpretation |
|---|---:|---|
| Fullscreen original | 0.2606 | Strong step, relatively uniform hair |
| Window player crop enlarged to full | 0.2719 | Becomes fullscreen-like despite retaining window-player pixels |
| Fullscreen blurred with sigma 0.7 | 0.2981 | Blur alone does not restore window behavior |
| Fullscreen shrunk into neutral surround | 0.3644 | Partial change, not equivalent to browser context |
| Native windowed input | 0.4964 | About 1.90x the fullscreen relative internal variation |
| Exact fullscreen image shrunk into real browser UI | 0.5667 | About 2.17x fullscreen and in the window-like direction |

This establishes, for this image, that the raw DAV2 result depends strongly on video scale and surrounding context. It also rejects the simple claim that fullscreen merely contains poorer or blurrier Bilibili pixels:

- enlarging the windowed video makes it full-like
- mild fullscreen blur remains full-like
- shrinking the same fullscreen content into browser UI makes it window-like

### Separating the two effects

The experiments support two related but distinct effects:

1. **Smaller semantic scale restores more variation inside the hair/person.** The large fullscreen person is represented more like a coherent foreground cutout.
2. **The real browser context materially changes the person/background relationship.** A neutral surround did not reproduce the real browser result, so `zoom out onto a neutral border` is not yet a validated fix.

Apollo's later whole-frame normalization adds a third amplification: in the fullscreen dump, 28.5% of the sampled hair is above the frame's raw P98 bound, compared with 0% in the real windowed dump. This concentrates a spatially coherent part of the subject at the normalized endpoint.

### Architecture facts that make the result plausible

Inspection of the exact ONNX file found:

- a non-overlapping 14x14 patch convolution with weight shape `[384, 3, 14, 14]` and stride 14
- 12 transformer attention Softmax nodes, one in each `/blocks.0` through `/blocks.11`
- global token mixing through the transformer stack
- resized positional embeddings
- a DPT-style convolution/refinement depth head using intermediate transformer features

This architecture is not expected to be crop- or scale-equivariant: changing how much of the token grid a person occupies, and changing all surrounding tokens, can change every output patch.

That is a plausible mechanism, not by itself proof of the cause. The controlled outputs are the evidence.

## ONNX bisection: where browser context changes the prediction

### Controlled input contract

The bisection used two synthetic 770x434 inputs built from the same fullscreen frame:

- the frame reduced into the captured browser surround
- the same reduced frame at the same `(55,83)-(563,371)` coordinates in a uniform neutral surround

Every RGB value inside the player rectangle is bit-identical between the two inputs. The player therefore has the same scale, sampling, patch coordinates, and positional embeddings. Only the pixels outside the player change.

At 770x434, this DAV2-S model has a 55x31 grid of 14x14 patch tokens, plus one CLS token: 1,706 tokens in total. The transformer has 12 global-self-attention blocks with six heads. Its DPT depth decoder consumes normalized features after blocks 2, 5, 8, and 11.

### Activation bisection

The original ONNX graph was temporarily instrumented to publish its patch embedding, every attention and MLP residual, all four DPT taps/projects, refinement stages, and final depth. ONNX Runtime CPU inference used graph rewrites disabled. Publishing the extra tensors did not change `predicted_depth` (`max_abs_delta = 0`).

The table reports symmetric relative RMS difference inside a two-patch-inset video core. A value of zero means the browser and neutral activations are identical there.

| Stage | Core relative RMS |
|---|---:|
| patch convolution | 0.000 |
| tokens immediately before block 0 | 0.000 |
| block 0, after self-attention residual | 0.155 |
| block 0, after MLP residual | 0.107 |
| block 2 output / first tap neighborhood | 0.117 |
| block 5 output / second tap neighborhood | 0.164 |
| block 8 output / third tap neighborhood | 0.218 |
| block 11 output | 0.260 |
| DPT project 0 / 1 / 2 / 3 | 0.067 / 0.138 / 0.206 / 0.325 |
| DPT refinenet 4 / 3 / 2 / 1 | 0.301 / 0.279 / 0.260 / 0.283 |
| final predicted depth | 0.239 |

This locates the **first context ingress into the player tokens unambiguously in block 0 global self-attention**. The patch encoder produces exactly the same player features, so neither local preprocessing nor the patch convolution originates their difference. Later transformer blocks accumulate and reshape it, and the deepest DPT projection has the largest feature divergence. This trace identifies where values differ, but not whether the visible depth change is carried mainly inside player tokens or in surrounding spatial features; the activation splice below separates those routes.

The final fields still look broadly correlated because both are positive depth maps. That hides the geometry change: over the whole player, their affine-aligned correlation is 0.847 and the P95 residual is 31.9% of the browser result's P2-P98 span. In the inset core, correlation is 0.889 and P95 residual is 23.1%.

### Attention trace

Video queries assign more attention to outside patches with browser UI than with the neutral surround in every transformer block. Mean outside-patch attention for representative blocks was:

| Block | Browser UI | Neutral |
|---:|---:|---:|
| 0 | 0.384 | 0.242 |
| 3 | 0.109 | 0.061 |
| 6 | 0.130 | 0.057 |
| 7 | 0.130 | 0.063 |
| 8 | 0.116 | 0.061 |
| 9 | 0.146 | 0.080 |
| 11 | 0.175 | 0.126 |

Block 0 is highly head-specific. Its head 5 sends 99.8% of video-query attention to outside browser patches versus 18.3% with the neutral surround; most of the browser mass lands on two patches in the right-side UI. This is strong evidence that DAV2 reads the surround as global scene context, but it is **not** evidence that head 5 alone causes the useful depth change. Masking the largest-attention head, or the blocks ranked highest only by attention mass, did not reproduce the causal result. Attention magnitude is descriptive; values, projections, residual paths, and block interactions determine the output.

### Causal attention intervention

A second temporary ONNX variant set direct video-patch-to-outside-patch and outside-patch-to-video-patch attention logits to `-10000` before Softmax for selected block groups. Post-mask readback verified exactly zero direct cross-region attention. CLS remained unmasked, so an indirect relay was still possible.

The most useful shape diagnostic here is the difference in normalized hair/background separation between the browser and neutral inputs. In the untouched model it is `-0.316`: browser context substantially softens the otherwise harsh foreground/background step.

| Direct cross-region attention suppressed in | Browser-minus-neutral separation | Change from baseline |
|---|---:|---:|
| none | -0.316 | baseline |
| blocks 0-2 | -0.312 | 1.2% smaller |
| blocks 3-5 | -0.280 | 11.6% smaller |
| blocks 6-8 | -0.043 | **86.4% smaller** |
| blocks 9-11 | -0.147 | 53.6% smaller |
| blocks 6-11 | -0.040 | 87.4% smaller |

Blocks 6-8 are the cleanest block-level attention mediator of the browser UI's beneficial silhouette softening in this frame. Suppressing that path moves the browser prediction toward the harsher neutral prediction: browser hair/background separation rises from 0.608 to 0.898, while neutral is 0.941. This is an explanation of the observed effect, **not a proposed fix**.

Head-level masking refines that result. Block 8 head 2 mediates 81.1% of the separation change by itself; block 7 head 3 is secondary at 27.8%; suppressing both mediates 89.8%. These percentages are non-additive because residual and attention paths interact. Conversely, block 0 head 5 has the most dramatic raw outside-attention mass but is nearly irrelevant to the final separation. Suppressing the three blocks ranked highest only by attention-mass difference actually makes the geometry gap worse. Raw attention magnitude is not causal importance.

Directional masks resolve how the dominant heads communicate. For block 8 head 2, suppressing outside-query-to-video-key attention mediates 59.4% of the effect, while suppressing video-query-to-outside-key attention mediates 31.7%. For block 7 head 3 plus block 8 head 2, the corresponding figures are 76.4% and 41.5%. The main route is therefore **outside tokens consuming video values, becoming video-conditioned, and later influencing player depth**, not simply outside values being copied into player tokens. A real direct surround-to-player route exists, but is secondary.

The causally selected block 8 head 2 and block 7 head 3 send much of their browser-case video-query attention toward right-side recommendation cards. That describes their destination, not necessarily the pixel source of the useful prior; the spatial input ablation below reaches a different and stronger conclusion about source pixels.

### Activation splice: the dominant carrier is the deepest outside feature

A third graph-surgery experiment ran the unchanged decoder suffix on hybrid UI/neutral activations. At each transformer/tap boundary, it independently replaced player tokens or outside-player tokens. All four neutral decoder taps reproduce the neutral prediction exactly (`max_abs_delta = 0`), validating the suffix split.

Baseline normalized hair/background separation is 0.609 with browser UI and 0.924 with the neutral surround. Treating the neutral value as 100% recovery:

| Hybrid activation | Separation | Recovery toward neutral |
|---|---:|---:|
| UI baseline | 0.609 | 0.0% |
| neutral baseline | 0.924 | 100.0% |
| UI player features + neutral outside features at all four DPT taps | 0.851 | **76.9%** |
| neutral player features + UI outside features at all four DPT taps | 0.610 | **0.5%** |
| replace the entire deepest block-11/DPT tap only | 0.928 | **101.3%** |

Moving the player/outside mask by one token still recovers 67.5-76.7%, so this is not a one-cell boundary artifact. Replacing outside features earlier recovers 100.4% after block 2, 97.1% after block 5, 90.4% after block 8, and 76.8% after block 11. Context progressively enters player tokens, but most of the final silhouette effect remains carried by outside spatial features.

The strongest causal statement is therefore more specific than "global attention changes the player": **the deepest DPT feature is decisive, and the decoder's use of its outside-player tokens carries roughly three quarters of the visible browser softening in this frame.** Block 8 head 2, with block 7 head 3 secondary, makes those outside features video-conditioned; the DPT refinement path then uses the surrounding feature map to set depth inside the player.

### Spatial input ablation: most source evidence is below the player

A complete four-region factorial ablation restored captured UI pixels to a neutral surround while keeping every player pixel fixed. The regions were top, bottom, left, and right of the player.

- neutral separation: 0.924
- full browser separation: 0.608
- bottom strip alone: 0.654, reproducing about 85% of the full raw change
- Shapley allocation after interactions: bottom 72%, top 16%, left 10%, right 3%

The bottom strip was then split into five token-aligned horizontal bands. Its effect was spread across the two metadata/action rows and the Windows taskbar; the immediate player-boundary row slightly opposed the change. This rules out a simple edge or exact-border explanation. The strongest single sub-band was the taskbar, but no one band reproduced the whole result.

The pixel source and attention destination should not be conflated. The dominant source pixels are the horizontal page/desktop bands below the player, while the causally selected heads attend heavily to right-side context. Together they indicate an interacting whole-screen layout prior rather than one magic border, color transform, or sidebar patch.

### Mechanistic conclusion and limit

```mermaid
flowchart LR
    UI["Horizontal page/taskbar context"] --> OUT["Outside patch tokens"]
    VID["Identical video patch tokens"] --> ATTN["Block 8 head 2<br/>block 7 head 3"]
    OUT --> ATTN
    ATTN --> DEEP["Video-conditioned outside features<br/>deepest DPT tap"]
    DEEP --> DEC["DPT spatial refinement decoder"]
    DEC --> DEPTH["Softer player hair/background step"]
    OUT -. "secondary direct surround path" .-> VID
```

For this controlled frame, the browser surround acts as a learned global scene/scale prior. Player tokens first become context-dependent in block 0. The useful cross-region attention is concentrated around block 8 head 2 and block 7 head 3, predominantly by updating outside tokens from video values. Most importantly, the browser layout remains encoded in those video-conditioned outside features at the deepest DPT tap, and the convolutional/refinement decoder uses them to soften the subject/background step inside the player. The horizontal UI below the player supplies most of the causal source-pixel effect.

The bisection still does **not** establish the learned semantic label: embedded media, browser chrome, horizontal screen planes, repeated rectangular structure, or some combination. Attention masking and activation splicing are deliberately out of distribution and cannot be shipped as model modifications. They cover one PNG-quantized frame and no temporal behavior. Repeating the same component and activation ablations across matched clips is required before synthesizing a production context prior.

## How Apollo postprocessing amplifies the raw difference

The measured sequence is:

```text
fullscreen framing
  -> raw DAV2 person becomes more step-like and internally uniform
  -> whole-frame P2/P98 normalization places much of hair in the near tail
  -> additive subject recenter increases the endpoint plateau
  -> a larger silhouette disparity step creates more disocclusion/edge burden
```

Offline replay of the two dumps found:

- current fullscreen hair-collar forward holes: about 13.00% left / 11.23% right
- current windowed hair-collar forward holes: about 1.73% left / 2.43% right
- widening normalization and replacing additive recentering recovered internal fullscreen relief, but left the fullscreen hole burden essentially unchanged

Therefore normalization/recenter deserves its own correction, but it is not sufficient to fix the fullscreen hair artifact. The problematic silhouette relationship already exists in the direct model output.

## What is established

- `raw_depth.f32` is the direct exported TensorRT model output for the matched model input.
- The two analyzed runtime inputs used identical tensor dimensions and preprocessing contracts.
- The fullscreen raw result is more cutout-like in the sampled hair/person region.
- Enlarge/shrink counterfactuals change raw model geometry in the corresponding direction.
- Browser context is material; a neutral surround is not equivalent.
- Browser context first enters otherwise identical video tokens through transformer block 0 self-attention.
- In this frame, block 8 head 2 and block 7 head 3 are the dominant direct cross-region attention paths for the browser-induced separation change.
- Replacing only the deepest DPT tap with its neutral counterpart recovers the full neutral separation; outside-player features across the taps carry about 77% of the visible effect.
- Most causal source-pixel evidence is in the page/desktop bands below the player, not the immediate player boundary.
- Apollo's normalization and additive recentering amplify the fullscreen near-tail plateau.
- Normalization-only variants do not remove the observed silhouette/disocclusion burden.

## What is not established

- That all fullscreen video behaves this way.
- That Bilibili, browser video, HDR, or 4K capture is uniquely responsible.
- That browser UI always improves depth.
- That neutral, blurred, reflected, or synthetic padding is a production-ready fix.
- Which semantic part of the browser surround supplies the useful prior.
- That one block or attention head can be modified safely to recreate the effect.
- That the current manual hair/background masks are ground truth.
- That one still-frame treatment will remain temporally stable across motion and cuts.

## Methodological limitations

1. **Single scene:** the causal tests cover one Bilibili scene.
2. **Original frames differ:** windowed and fullscreen live captures are 97 matched IDs apart.
3. **Manual exploratory masks:** player rectangle, hair box, luma threshold, and background boxes were hand-selected. Hair/background claims are stronger than generic subject-mask claims; the subject mask partially used raw depth and should not be treated as independent ground truth.
4. **Relative-depth ambiguity:** raw values should be compared with rank, ratios, or a single fitted affine transform over corresponding regions. Independent per-image P2/P98 plots are visual aids, not absolute-detail measurements.
5. **Preview quantization in one test path:** CPU counterfactual inputs came from 8-bit `model_input.png`; the exact-input parity test and native TensorRT harness bound this error.
6. **No temporal validation:** the current counterfactual is frame-level and does not establish flicker or scene-cut behavior.

## Can fullscreen preprocessing reproduce the browser effect?

Two exploratory sweeps tested whether the useful context response could be induced without displaying a smaller video or adding a visible border.

### Photometric-only sweep

The exact fullscreen tensor was transformed and replayed through the same ONNX model without changing framing. For the affine-invariant ratio `hair P5-P95 span / hair-background separation`, the fullscreen baseline was 0.258 and the windowed target was 0.496.

- gamma 2.2: 0.269
- strong contrast: 0.339, with about 7% clipping
- unsharp luminance: 0.284
- CLAHE-like luminance: 0.288
- ordinary RGB blur: 0.200 or worse
- moderate spatial luminance blend: 0.363, but with a material global-geometry change
- aggressive spatial luminance replacement: 0.486, but with background gradients and halos

No safe global gamma, contrast, sharpening, blur, or local-contrast transform reproduced the windowed result. The one near-match changed the scene representation enough to be an invalid general solution.

### Borderless spatial-warp sweep

An endpoint-fixed center-compression warp made the center occupy about 84% of the input area, inferred once, then inverse-warped the raw depth. A strength near 0.085-0.09 recovered the fullscreen hair span to approximately the windowed level and softened peak edge gradients. However, it made the median hair/background separation 5.5% harsher, remained materially worse than windowed at the silhouette, and responded non-monotonically to strength.

The sweep proves a spatial input prior can influence the desired feature without literal padding, but not that this particular warp reproduces the browser benefit or is temporally safe. It should not be shipped from this frame.

### Full-resolution expanded-bottom-context experiment

This controlled test preserved the exact 770x434 fullscreen tensor bit-for-bit, appended 14, 28, 42, 56, or 70 rows of actual below-player context, neutral pixels, or repeated edge pixels, inferred at 770x448 through 770x504, then cropped the prediction back to the top 434 rows. For hair P5-P95 span / hair-background separation / near-tail saturation, fullscreen measured 0.17542 / 0.76156 / 28.22%, while the windowed reference measured 0.30457 / 0.60769. The best variant, `actual_below_player_5`, reached only 0.21127 / 0.71073 / 15.45%: 27.8% of the desired span recovery and 33.0% of the desired separation recovery.

Actual versus neutral rows produced a content-specific effect, but positional-embedding/canvas-size effects were comparable in scale and nonmonotonic with added height. Bottom-only expansion is therefore insufficient to reproduce the browser benefit and should not ship. Artifacts: `C:\Users\DCatc\AppData\Local\Temp\dav2_expanded_context_audit\report.json` and `C:\Users\DCatc\AppData\Local\Temp\dav2_expanded_context_audit\depth_comparison.png`.

### Full-resolution 2D-context experiment

A second test preserved every 770x434 fullscreen-video tensor value and appended both right-side and bottom context. Real captured UI improved the hair span to 0.22064 and softened hair/background separation to 0.69133, versus 0.17542 / 0.76156 fullscreen and 0.30457 / 0.60769 windowed. That recovers only 35.0% of the desired span and 45.6% of the desired separation change. More importantly, an edge-repeat control was as good or slightly better on those two values (0.22446 / 0.68404), so the improvement is not clean semantic-context evidence; canvas shape, positional embeddings, and boundary conditions are comparable causes.

Recreating the full browser layout at the largest practical model canvas (1036x588) was not stable. It overshot hair span to 0.36178 while making separation worse at 0.78318. The two expanded canvases have 2.18x and 3.32x the baseline attention matrix, respectively. Neither is a defensible live preprocessing path. Artifacts: `C:\Users\DCatc\AppData\Local\Temp\dav2_full_2d_context_audit\report.json`, `...\depth_comparison.png`, and `...\input_comparison.png`.

### Fixed-shape synthetic-surround experiment

The final sweep kept the existing 770x434 model shape and placed a bit-identical 508x288 copy of the fullscreen video at the controlled browser player rectangle. Only the hidden surround changed: replicate, reflect, wrapped/tiled, heavy blur, 16x9 low-frequency color, and neutral. The real captured UI target measured hair span 0.36928, separation 0.60757, zero near-end saturation, left/right edge steps 0.5268 / 0.5017, and peaks 0.2300 / 0.2173.

- Low-frequency context was the closest primary-shape variant at 0.31730 / 0.71978, but 42.35% of the hair stayed pinned at the global P98 endpoint and edge peaks were over-smoothed to 0.1160 / 0.1588.
- Heavy blur had the same failure: 0.28432 / 0.71007, 42.35% saturation, and over-smoothed edge peaks.
- Replication preserved more edge energy, but separation remained too harsh at 0.77783. Reflection, tiling, and neutral surrounds were worse.
- A corrected centered scale sweep from 0.66 to 0.88 of canvas width never approached the real-UI separation and retained 41-42% endpoint saturation. Moving the same 0.66-scale player from center to the browser's off-center rectangle changed separation from 0.94564 to 0.71978, confirming that layout/position dominates scale nonlinearly.

No tested synthetic surround safely reproduces the browser result. Shipping one would exchange the current fullscreen artifact for endpoint clipping, over-smoothed boundaries, or another framing-dependent failure. Artifacts: `C:\Users\DCatc\AppData\Local\Temp\dav2_synthetic_surround_sweep\report.json`, `...\depth_comparison.png`, and `...\input_comparison.png`.

## Decision after the depth-coordinate rethink

Do not add hidden padding, a larger TensorRT canvas, a fullscreen detector, another ROI state
machine, or a learned adapter for this issue. The evidence identifies a real context sensitivity in
DAV2, but the available browser-derived target is not ground truth and does not establish a
generalizable model correction. Training against it risks learning the browser layout rather than
better scene geometry.

Depth Coordinate V2 therefore treats the model output as relative evidence in an authenticated,
fixed coordinate system. It controls visible failure at the geometry boundary with an exact
frame-local disparity container and a one-sided horizontal cliff limiter that preserves near
relief. Model-level context invariance remains an independent research question and is not part of
the current implementation.

## Recommended independent audit

1. Verify the hashes above and inspect both `dump_manifest.json` files.
2. Re-run exact `model_input.f32 -> ONNX -> predicted_depth` parity before interpreting any downstream artifact.
3. Use the native harness to replay **the exact same decoded frame** under isolated variants:
   - fullscreen
   - shrink into captured browser context
   - shrink into several neutral colors
   - shrink into blurred/reflected image-derived context
   - enlarge the embedded player crop
4. Reset the estimator between variants for downstream tests. Raw output itself has no temporal state, but normalization, recenter, cut, and zero-plane state do.
5. Prefer affine-invariant statistics:
   - within-region depth span divided by subject/background separation
   - rank correlation and edge-step distributions
   - one affine alignment fitted over common scene support
6. Replace manual masks with a fixed externally generated mask or audited color-space mask, and report sensitivity to mask erosion/dilation.
7. Repeat over a temporally matched clip and multiple content classes before proposing a production rule.
8. Carry promising variants through the exact warp and forward-coverage diagnostics; more raw variation is not useful if silhouette holes or temporal instability worsen.

## Candidate directions, not accepted fixes

- **Context-conditioned single inference (rejected from current evidence):** real browser context worked in this scene, but expanded, neutral, replicated, reflected, blurred, tiled, and low-frequency surrounds all failed in different ways. Do not add this live path unless substantially broader evidence overturns those controls.
- **Multi-scale inference:** compare full and context-scaled predictions and combine confidence-aware regions. This is more expensive and has unresolved seam/temporal behavior.
- **Warp-side silhouette risk control:** limit locally unsupported disparity steps or improve disocclusion handling. This may reduce artifacts without pretending the raw prediction is correct, but it does not restore missing internal geometry.

None of these should be implemented from this one-scene result alone.

## Local analysis artifacts

These are developer-machine artifacts and are not committed source contracts:

- Counterfactual script: `C:\Users\DCatc\AppData\Local\Temp\dav2_framing_counterfactual.py`
- Counterfactual report: `C:\Users\DCatc\AppData\Local\Temp\dav2_framing_counterfactual\report.json`
- Counterfactual raw-depth montage: `C:\Users\DCatc\AppData\Local\Temp\dav2_framing_counterfactual\raw_depth_comparison.png`
- RGB/raw alignment report: `C:\Users\DCatc\AppData\Local\Temp\window_full_context\report.json`
- RGB/raw alignment montage: `C:\Users\DCatc\AppData\Local\Temp\window_full_context\comparison.png`
- Native harness report: `C:\Users\DCatc\AppData\Local\Temp\context_harness_out\context_report.json`
- Native harness raw files: `C:\Users\DCatc\AppData\Local\Temp\context_harness_out\raw_*.f32`
- Activation-bisection script and report: `C:\Users\DCatc\AppData\Local\Temp\dav2_activation_bisection\activation_bisection.py` and `...\report.json`
- Attention/causal-bisection script and report: `C:\Users\DCatc\AppData\Local\Temp\dav2_attention_context_audit.py` and `...\dav2_attention_context_audit\report.json`
- Attention layer/head table and mask validation: `C:\Users\DCatc\AppData\Local\Temp\dav2_attention_context_audit\attention_by_layer_head.csv` and `...\mask_validation.json`
- Head and directional ablations: `C:\Users\DCatc\AppData\Local\Temp\dav2_attention_context_audit\head_ablation_report.json` and `...\directional_ablation_report.json`
- Causal-head attention panel: `C:\Users\DCatc\AppData\Local\Temp\dav2_attention_context_audit\causal_head_attention_destinations.png`
- Activation/tap-splice report and montage: `C:\Users\DCatc\AppData\Local\Temp\dav2_context_causal_bisection\report.json` and `...\causal_key_montage.png`
- Spatial source-component ablation: `C:\Users\DCatc\AppData\Local\Temp\dav2_spatial_context_ablation.py` and `...\dav2_spatial_context_ablation\report.json`
- Photometric preprocessing sweep: `C:\Users\DCatc\AppData\Local\Temp\dav2_photometric_fullframe\report.json`
- Borderless spatial-warp sweep: `C:\Users\DCatc\AppData\Local\Temp\dav2_center_compress_ab\report.json`
- Normalization/controller replay: `C:\Users\DCatc\AppData\Local\Temp\normalization_controller_warp_ab\report.json`
- Exact warp/coverage replays: `C:\Users\DCatc\AppData\Local\Temp\sbs_depth_shaping_full\report.json` and `...\sbs_depth_shaping_window\report.json`
- Expanded-bottom context: `C:\Users\DCatc\AppData\Local\Temp\dav2_expanded_context_audit\report.json`
- Full 2D context: `C:\Users\DCatc\AppData\Local\Temp\dav2_full_2d_context_audit\report.json`
- Fixed-shape synthetic surrounds: `C:\Users\DCatc\AppData\Local\Temp\dav2_synthetic_surround_sweep\report.json`

## Short handoff statement

The counterintuitive observation is real for this scene: giving the video more DAV2 pixels did not improve the useful raw geometry. Fullscreen changed the model's relative prediction toward a coherent foreground cutout with a strong silhouette and less internal hair variation. The direct-output provenance is strongly verified, including an independent exact-input ONNX replay. The ONNX bisection identifies a more precise browser-context path: player tokens first diverge at block 0, block 8 head 2 chiefly makes outside tokens video-conditioned, the deepest DPT tap carries the decisive context, and the decoder's use of outside-player features accounts for roughly 77% of the visible softening. Most causal source pixels lie in the horizontal UI below the player. Expanded and synthetic-surround controls do not safely reproduce that response: layout/position effects are nonlinear, padding leaves endpoint saturation or damages edges, and larger canvases add disproportionate attention cost. Do not ship a preprocessing rule or train against the browser-derived response from this result. Use the fixed-coordinate V2 geometry boundary to contain cliffs while preserving near relief; treat model context invariance as separate research.
