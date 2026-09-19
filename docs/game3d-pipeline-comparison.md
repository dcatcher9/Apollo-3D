# Game3D pipeline comparison

The sole Sunshine warp replaces Game3D's ray renderer (2026-09-18).
The [integration contract](reshade-sbs.md) owns setup/capture; the
[Host SBS contract](host-sbs.md) owns the conditioning and inverse algorithm.

**Original** means the pristine `Shaders/SuperDepth3D.fx` blob at Depth3D commit
`be494f78e1fcbe1f14752a0035ba5d38955571f0`, inspected with `git show`. The working
`SuperDepth3D.fx` is an adapted export/HDR reference, not pristine upstream. Neither is a current
Game3D branch. The prior Sunshine experiment bypassed rays but still ran unused preparation.

The current [GPU program](../tools/reshade/game3d_native.hlsl) is embedded in the add-on.
[The renderer](../tools/reshade/game3d_renderer.cpp) owns its passes and resources; there is no
installed FX dependency. The former three-file Game3D effect is retained only as a comparison
oracle. The native port preserves its conversion, conditioning, inverse and color operations.

## Stage audit

| Stage | Original behavior | Sole Sunshine behavior and rationale |
|---|---|---|
| Acquisition | ReShade depth selection. | **Retained:** SL/NGX or Generic nomination, shared capture and manual pins. Selection, availability and calibration stay separate. |
| Coordinates/profiles | Overwatch near/range/weapon profiles, flips and guessed letterboxing. | **Replaced:** source-owned projection/orientation, active rectangle and captured jitter. Keep registration; remove manual profile implementations. |
| Conversion | `Depth`/`PrepDepth`: near-plane, power/offset, range boost and rational shaping. | **Replaced:** FP32 `q=(raw-A)*inverseB` feeds geometry; rational `1/(1+H*q)` remains only for Normal Depth display. |
| Scene information | `Average_Info`, `zBuffer_Blur`, `DepthMap`: averages, mips and masks. | **Removed:** unused by Sunshine; add-on owns reference tracking. |
| Prepared depth | `MiniReconstruction`, `DB_Comb`/`Mod_Z`, `Mix_Z`/`GetDB`: signed intermediates and dilation. | **Removed:** direct native-depth displacement replaces these textures and their ineffective quality controls. |
| Temporal depth | `TAA_Buffer`/`Acc_Buffer` history blending/clamping. | **Removed:** histories never fed Sunshine. Captured jitter and bounded FG reuse remain; neither is this temporal filter. |
| Edge reconstruction | `Min3x3`, `Disocclusion`, `DepthSmoothPS`: silhouette/donor preparation. | **Removed:** built for ray disocclusions. Do not transplant representation-specific filtering without evidence. Sunshine conditions displacement instead. |
| Eye geometry | Backward rays, refinement and hole donors. | **Replaced:** vertical envelopes → horizontal majorant → bounded inverse → color sample. Can bend outlines and flatten nearby background. |
| Color/HUD | `CSB`, borders, vignette, cursor and profile masks. | **Simplified:** native SDR/scRGB or decoded PQ. Baked-in HUD remains in the warped image; no separate weapon/HUD source is supplied. |
| Sharpen/AA | SmartSharp; optional AXAA/SDAA. | **Removed:** sharpening and final AA, including their controls, by user choice. Game TAA/depth filtering remain independent. |
| Presentation/UI | Multiple layouts and shader help/font pass. | **Removed:** alternate outputs and startup help. **Retained:** native mono game, SBS and actual ReShade/add-on overlay in both eyes. |
| Packing/transport | Original backbuffer; adapted reference adds export. | **Retained:** `DoubleTex`, per-eye texel centers, HDR/metadata/shared handoff. Zero strength gives current mono eyes, except that a ready Normal Depth View remains visible. Missing depth or unsupported sizes always give current mono eyes. |

HDR10 adds PQ decode before color filtering. Old ray preparation passes/textures are gone;
measure total GPU cost rather than infer it from code size.

## Coordinates and constants

For camera depth, `q=1/Z=(raw-A)/B`; raw fallback uses `q=raw` or `q=1-raw` under its explicit
infinite-far assumption. The shared controller tracks the central inverse-depth reference `q0`
and publishes `H=1/q0` (`K` on the camera path). This is not recovered meters or a physical baseline.
Zero-plane movement changes normalization; multiplicative game units cancel in `q/q0`.

With source width `W`, height `T`, transition blend `b` and slider `a`:

```text
s = clamp(a, 0, 100) / 100 * b
d = g * H * (q0 - q) * s
r = (100 * T / 2160) / W
p = clamp(-clamp(d, -1.5, 2.5) * r, -0.04, 0.04)
```

`p` is per-eye **source U**, not packed-SBS U; positive means near. There is no percentile clipping
of raw depth. Displacement clamping is an output/comfort decision, not depth-outlier removal.

| Value | Meaning and status |
|---|---|
| `g=0.05` | Source-owned dimensionless reference convergence; strength convention, not measured camera data. |
| Strength `0–100`, default `50` | User multiplier; zero is exact mono in the game picture and Stereo Depth View. A ready Normal Depth View remains visible at zero strength. Transition blend is source-owned in `[0,1]`. |
| `100*T/2160` | 100 source px at 2160 rows; inherited strength conversion, not physical headset geometry. |
| `d∈[-1.5,2.5]` | Inherited asymmetric output-strength cap, **retained deliberately**. Host inversion does not require it. Removing it moved saturated foreground landmarks by 3.60 px/eye at 4K and demonstrated no speed gain. It is not a raw-depth clamp. At 16:9 the positive-`d` limit is redundant with the `.04` container; at other aspect ratios it need not be. |
| `|p|≤0.04` | Chosen per-eye source-width bound. Game3D hard-clamps; it does not import Host AI's DAV2 curve/soft container. |
| Horizontal slope `0.5` | `0.5/W` source U per column. This chosen bound below 1 guarantees contractive, single-valued inversion; other bounds below 1 are possible. |
| Vertical shear `2.0` | `2/W` source U per row: vertical continuity policy, using source **width**, not height. |
| Upper/lower share `0.75/0.25` | Empirical vertical envelope blend; foreground/background tradeoff, not depth-accuracy proof. |
| Q30, scale `2^30` | Outward rounding and safe decay preserve bounded chunk composition. |
| Maximum `11` inverse steps | Bounded work with exact-bit early settlement, relying on the contractive field. |
| Maximum `3840` per source axis | SM5 implementation limit: vertical 31,744 bytes, horizontal 31,232 bytes. Larger supported exports stay mono with a warning. |
| Sharpening `0–5`, default `0` | Artistic postprocess; nonzero settings need separate detail/ringing assessment. |

[`scene_gain.h`](../tools/reshade/scene_gain.h) uses four captures spanning 750 ms, then a 0.5 s
response and bounded rate. These are adaptation heuristics, not inversion constraints. Finite-domain
checks remain; old reciprocal-FP16 limits were removed for the current R32 displacement field.

## Measured decisions

Tests ran serially on the RTX 5080, NVIDIA 616.92, with Sunshine and both games closed.
ReShade 6.8.0 and each test executable were frozen; paired runs used identical source/depth bytes,
dimensions, camera, strength and color settings. All numeric analysis used
`E:/ApolloDev/modelopt-py312/Scripts/python.exe` (Python 3.12.14, NumPy 2.5.1,
Pillow 12.3.0, ONNX 1.21.0). Fingerprints, shader hashes, raw timestamps and captures are in
`cmake-build-reshade/sole-sunshine-warp-20260918`.

### Removing work that does not feed the output

At **3840×2160 per eye, scRGB, strength 49.303135, sharpening off**, the ABBA comparison used
60 warmup and 120 measured frames per scene, twice per variant. The following values average the
two run medians; they measure the complete ReShade effect, including its COLOR capture. They do
not include Sunshine encoding, transport, decoding or headset display latency.

| Scene | Previous Sunshine option, with unused preparation | Sole Sunshine pipeline | Saved |
|---|---:|---:|---:|
| Normal scale, step | 2.211 ms | 1.399 ms | 0.812 ms / 37% |
| Normal scale, detail | 2.347 ms | 1.878 ms | 0.469 ms / 20% |
| Normal scale, source fringe | 2.399 ms | 1.552 ms | 0.847 ms / 35% |
| Deep scale, step | 2.281 ms | 1.329 ms | 0.952 ms / 42% |
| Deep scale, detail | 2.578 ms | 1.717 ms | 0.861 ms / 33% |
| Deep scale, source fringe | 2.239 ms | 1.423 ms | 0.816 ms / 36% |

The removed targets account for **195.92 MiB of logical texel storage** at 4K, including their
mips. This excludes allocation padding and driver/transient storage; it is not a measured reduction
in total process VRAM. A prior serial comparison of the old Game3D ray branch versus the still
untrimmed Sunshine option measured 5.92–7.55 versus 2.28–2.69 ms in normal scenes, and
3.16–3.59 versus 2.25–2.66 ms in deep scenes. Those historical numbers use that earlier snapshot;
they are not a new direct timing of pristine upstream SuperDepth3D against today's shader.

At 1280×720, the old Sunshine option and the trimmed path have **byte-identical candidate,
vertical-conditioned and final displacement fields**, with zero near/far/middle landmark error.
The production rewrite derives both eyes directly from pixel centers. Small color differences
remain from coordinate rounding and compiler contraction (detail RGB RMS 0.000654, maximum
0.009766 in linear scRGB). A separate proof variant restoring the prior coordinate evaluation and
`precise` inverse arithmetic reproduced **all RGBA and all fields byte for byte**. That variant is
test evidence, not production scaffolding. This isolates the removed preparation as unnecessary;
the final production output is not claimed byte-identical to the old coordinate arithmetic.

### Quality/cost ablations

These are actual 4K shader runs using the same deep-scene inputs, 120 warmup and 240 measured
frames. They are single paired ablations, so timing differences are observations rather than
precise universal stage costs.

| Change tested | Observation | Decision |
|---|---|---|
| Bypass vertical conditioning, retaining a copy pass | Saves 0.168–0.440 ms. Same interior disparity, but maximum adjacent-row change in the detail field rises from about 2 to **103.16 source pixels**. | **Keep.** It limits row-to-row outline discontinuities. It can also flatten nearby background; no claim that it recovers hidden geometry. |
| Remove `[-1.5,2.5]`, keep symmetric `.04` | Foreground saturation changes from 150 to about 153.60 px/eye. No observed speed improvement. It changes actual strength, so this is ineligible as an equal-strength halo comparison. | **Keep the tested output-strength cap.** It is legacy-derived artistic policy, not an inverse-solver requirement. Future consolidation must account for aspect ratio and saturation. |
| Sharpening 1 instead of 0 | Adds 0.265–0.585 ms. Detail ROI mean absolute horizontal RGB variation rises 2.6%, while RGB extrema widen from −0.160/3.199 to −0.189/3.563. Endpoint change stays below 0.011 px. | **Keep optional, default off.** Some detail emphasis, with overshoot and meaningful GPU cost. This metric is descriptive, not proof of recovered detail. |
| Old near/range/profile/depth smoothing and ray reconstruction | Exact-output proof shows no contribution to the previous Sunshine branch. | **Remove.** Zero quality return cannot justify their collective GPU/memory cost. |
| Source rect, jitter and correct color conversion | Actual GPU coordinate/color oracles pass, including half-resolution depth, padding, signed jitter, negative scRGB and PQ decode before filtering. | **Keep.** These register and preserve the input; removing them creates alignment/color errors. |

### Constants and controls that should not be collapsed blindly

- Horizontal slope `.5` and the maximum 11 inverse iterations form one numerical contract.
  With a `.04` initial source-U error and contraction `.5`, the ideal bound after 11 iterations
  is about **0.075 source pixel at 3840 width**, before floating-point error. Shortening the loop
  or widening the slope needs a new accuracy/performance test.
- The vertical bound `2`, envelope share `.75`, and `.04` output container are quality/comfort
  policy. They are useful, but not uniquely optimal physical constants. The new tests justify
  retaining vertical conditioning, not claiming that `.75` is optimal for every game.
- Q30 outward rounding, overflow-safe decay and chunk carries preserve the same bound across
  parallel chunks. Replacing them with a shorter float scan would change the numerical contract.
- The **16384 reciprocal-depth limit and 65504 half-float limit were removed** from the native
  shader and active CPU camera/raw/projection admission. Those belonged to deleted FP16 depth
  intermediates. Finite FP32 displacement checks, projection coefficient checks and current-depth
  readiness remain; actual GPU cases accept scales 32767 and 1e8 and reject invalid/overflowing input.
- `Min_Divergence()` and its fixed helper chain became the explicit `100*T/2160` conversion.
  The old warp selector, ray quality, compatibility, reconstruction, mask and extended-smoothing
  controls have no remaining consumer and are removed, including their saved Game3D preset keys.
  Strength and depth view are the public image settings in the add-on. GPU compile definitions
  are private source-size/transfer specializations; old display/debug options cannot select alternate paths.
- The source-owned projection, rect, jitter, readiness, transition, scale and zero-plane fields
  are integration inputs. Although `g=.05` could algebraically be absorbed into strength, doing so
  would change their shared producer/consumer convention for no demonstrated performance gain.
- The add-on's four startup samples/750 ms span and 0.5 s smoother are independent adaptation
  policy. This warp replacement leaves their behavior intact; it does not present a new validation
  or optimization of gameplay calibration.

### What can still transfer from SuperDepth3D

Its useful retained component is optional SmartSharp, with the existing HDR-safe adaptation.
The ray-specific masks, dilation, reconstruction and depth history cannot simply be inserted into
the affine displacement pipeline: they use a different representation and can alter real disparity.
No additional original preparation stage earned inclusion in this change. Separate weapon/HUD
handling could help if a provider supplies an associated layer; per-game guessed masks are not a
replacement for that missing input. Temporal filtering would need a motion test demonstrating
less shimmer without stale edges or blur before adding history cost.

### Direct comparison with the adapted original

The `original-pair-1280-v5` comparison uses the adapted original SuperDepth3D and
the sole Sunshine shader with identical 1280×720 color/depth inputs and final AA
off. An isolated GPU landmark fit sets the treatment's camera; both quality runs
then start fresh with matching deterministic timelines. Near/far eye positions
match within 0.004 px; the held-out middle-plane binocular difference is 0.071 px.

The synthetic four-pixel source fringe shows a tradeoff. The mean foreground-color
coefficient in the known background is:

| Distance from the independently shifted foreground edge | Original | Sole Sunshine |
|---|---:|---:|
| 4–8 pixels | 0.03208 | 0.15303 |
| 8–32 pixels | 0.01065 | 0.000321 |
| Combined 4–32 pixels | 0.01371 | 0.02214 |

Sunshine greatly reduces the distant color tail, but concentrates more color near
the silhouette. The combined 4–32-pixel metric is worse, so this is not a uniform
halo reduction. Its 0–4-pixel band is also stronger; that band includes deliberate
source fringe and normal edge antialiasing and cannot all be classified as a bug.
These coefficients describe a known synthetic background, not overall game quality.

The single 1280×720 pair measured original versus sole medians of 0.491/0.377 ms
for the step, 1.266/1.392 ms for detail, and 0.502/0.596 ms for source fringe.
Each used 60 warmup and 120 measured samples. A single pair establishes neither a
universal speedup nor a reliable per-stage cost. In particular, Sunshine was
slower for two of those small-resolution cases.

The separate **3840×2160 per-eye** pair (`original-pair-4k`) uses the same controls,
60 warmup and 120 measured frames, matching near/far endpoints within **0.038 px
per eye / 0.073 px binocularly**. At the held-out middle depth the binocular
difference is **3.21 px**: this is an endpoint-matched comparison, not a claim that
the two preprocessing curves are identical at every depth.

| 4K matched-endpoint scene | Adapted original SuperDepth3D | Sole Sunshine | Saved |
|---|---:|---:|---:|
| Step | 2.244 ms | 1.300 ms | 0.945 ms / 42% |
| Detail | 2.264 ms | 1.521 ms | 0.743 ms / 33% |
| Source fringe | 2.275 ms | 1.373 ms | 0.902 ms / 40% |

These are complete-effect median GPU times from one controlled pair. For context,
the detail p95 is 3.362 ms original versus 2.244 ms Sunshine; they are not headset
latency measurements. The repeated cleanup timings earlier use a different control
(the old Sunshine option with unused preparation), and must not be conflated with
this direct original comparison.

At 4K the fringe's 8–32-pixel mean coefficient falls from **0.01251 to 0.00323**,
while the 4–8-pixel band rises from **0.01380 to 0.21028**, and the combined
4–32-pixel coefficient rises from **0.01269 to 0.03281**. The two endpoint planes
are matched in this fixture, so a weaker global strength does not explain the
shorter distant tail. The stronger near-edge band remains a real tradeoff.
Original-style donor/color-fringe handling is therefore a sensible future
**isolated sampling experiment**, but this comparison does not establish which
original reconstruction stage causes its cleaner immediate edge. Porting its
entire preprocessing chain would add cost and change the new displacement field.

## Validation and limits

The production add-on builds and **all 30 configured CPU test groups pass**. All 24 installer
migration groups pass. The native embedded HLSL passes 105 entry-point compile/reflection checks
across 18 size/transfer configurations. There is no installed Game3D FX or configurable
Game3D preprocessor definition.

Actual D3D12 native output is byte-identical to the frozen prior Game3D FX for all 13 scenarios
in 640×360 SDR and 4K scRGB/PQ. These cover ready, zero/full strength, missing depth/camera,
transition blending, Normal/Stereo diagnostics, raw/projection depth, active rect/jitter and
full-resolution recovery. D3D11 output matches exactly in six scenarios for SDR/scRGB/PQ.
The game backbuffer remains unchanged. Native phases have zero loaded FX techniques.

The actual add-on passes no-FX export, source identity, both-eye overlay, collapsed/expanded panel,
settings/reset, enable/disable, effects-toggle independence, reload, focus and reconnect checks
in SDR/scRGB/PQ. A real 4K DSV fixture also passes automatic capture and positive scale, stable
pin/unpin, FX-reload continuity, stale-depth rejection and same-source recovery without injected
depth or calibration. A second no-FX 4K fixture supplies SL-nominated lower-resolution UAV depth
over a full-resolution Generic decoy and validates projection scale, failed-evaluation rejection,
reload and recovery. Its metadata-only SLFG adapter exercises eight real/generated pairs with
retained completed depth, followed by FG-off rejection and fresh-depth recovery. The adapter
does not synthesize image frames or inject capture readiness; these are lifecycle checks, not
live-game FG image-quality acceptance.
The later Expedition 33 regression uses its observed `2228×1256` allocation with a `2228×1253`
active rectangle. The installed control captures depth but never enables native stereo; the
fixed build establishes scale/readiness for that padding and a nonzero crop origin, then retains
ready depth across eight real/generated pairs. CPU resolver tests cover the same padded layout
with both raw and projection calibration and retain the missing-uniform rejection for old FX.
The cause was an FX-uniform capability guard left in native active-rectangle handling, not failed
SL/NGX capture. Evidence: `cmake-build-reshade/native-crop-{control,fixed}-20260918*` and
`native-crop-tests-20260918.log`.
Evidence is under `cmake-build-reshade/addon-native-*20260918*`.
Actual D3D12 tests pass camera-domain admission, ready/lost/recovered depth, zero strength,
Normal/Stereo diagnostics, jitter/active-rect mapping, SDR, 4K scRGB/PQ and
4800×2700 explicit mono fallback. Original SuperDepth3D remains available as an independent
reference; it was not rewritten by this cleanup.

The current SM5 scan supports at most **3840 pixels per source axis**. Above that, the shader
allocates no Host fields, the add-on warns, and export stays current mono. A tiled scan is a
separate future improvement, not a hidden fallback to the deleted renderer.

Synthetic tests do not prove absence of artifacts in moving games. FG reuse and SDK depth capture
were not changed, so their temporal limitations remain. Nor does a single color/depth image reveal
occluded background: Sunshine's conditioning trades local geometry fidelity for a stable inverse.

The earlier AA motion comparison measured approximately **10–20% less registered temporal
variation**, but **25–31% less fine-texture contrast** and worse thin-object coverage; the user chose
removal. These [historical measurements](native-stereo-comparison.md) do not evaluate the new warp.

The September 18 live test reported reduced halos, not artifact-free output across games or FG modes.

Host conversion-to-send latency excludes ReShade warp. Full evidence and the static original-source
audit are retained in the dated workspace above; no CPU renderer was substituted for shader output.
