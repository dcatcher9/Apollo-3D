# Dump 3D and exact replay

This document owns the developer artifact and replay contract for Host SBS V2. The renderer itself
is specified in [Host SBS pipeline](../../docs/host-sbs.md).

## Atomic matched-frame package

While **Host SBS** is active, **Dump 3D** writes one directory below `APOLLO_SBS_DUMP`, or below
the host's default `sbs_dump` directory when that environment variable is unset. A completed
directory represents one matched source/depth/render identity. Partial directories are never
published as successful dumps.

### Source and model artifacts

| Artifact | Meaning |
|---|---|
| `source.png` | Source color frame paired with this depth result |
| `depth_input_source.png` | Spatial preview of the full source or exact inward crop submitted to DAV2; diagnostic only |
| `depth_input_region.json` | Hashed authority for analysis placement, tensor extent, ROI authorization, embedding scale, and exterior collar |
| `model_input.f32` | Exact ImageNet-normalized NCHW model tensor |
| `model_input.png` | Inverse-normalized, viewable model-input preview |
| `model_input_shape.json` | Tensor shape and layout |
| `raw_depth.f32` | Exact direct model output; numeric raw-depth authority |
| `raw_depth.png`, `raw_depth_heat.png` | Finite-P2/P98 visualization only |
| `raw_shape.json` | Raw tensor dimensions and format |

`depth.f32`, `depth.png`, `depth_heat.png`, and `depth_shape.json` are optional scoring or
compatibility views. Production V2 may omit them, and the renderer never consumes them.

### V2 geometry artifacts

| Artifact | Meaning |
|---|---|
| `shadow_coordinate.f32` | Canonical coordinate diagnostic before the fixed curve |
| `shadow_candidate_parallax.f32` | Immutable pre-conditioner candidate |
| `shadow_ownership_refined_parallax.f32` | Conservative full-resolution contour-ownership result |
| `shadow_vertical_majorant.f32` | Exact column-wise upper envelope |
| `shadow_vertical_conditioned.f32` | Fixed upper/lower vertical share |
| `shadow_final_parallax.f32` | Final row-majorant position field used by live rendering |
| `warp_depth.f32` | Compatibility name for the actual scalar field sampled by reprojection |

Each floating-point field has a shape document and, where generated, a stretched PNG preview.
In full-source mode the final parallax field is the rendering position authority. In video-region
mode it is crop-local ROI-U, so rendering authority is the pair
`{shadow_final_parallax.f32, depth_input_region.json}`. Independently normalized previews cannot
prove absolute scale or compare two dumps quantitatively; use the hashed numeric artifacts.

The package deliberately carries two spatial domains:

| Domain | Artifacts |
|---|---|
| Full matched source/output | `source.png`, `sbs.png`, `warp_map.f32`, `warp_mask.png` |
| Analysis source | `depth_input_source.png`; full source in ordinary mode, inward ROI crop in video-region mode |
| Crop-local tensor | `model_input.*`, `raw_depth.*`, every `shadow_*` field, and `warp_depth.*` |

`dimensions.source` remains the full capture. `dimensions.analysis_source` is the exact same-format
texture submitted to preprocessing and ownership. Every tensor field must match the authenticated
model extent recorded by `depth_input_region.json`; a consumer must never stretch crop-local depth
over the full source.

### Warp and rendered artifacts

| Artifact | Meaning |
|---|---|
| `warp_map.f32` | Exact inverse-warp source-U coordinates for both eyes |
| `warp_map_shape.json` | Shape, eye layout, and renderer-specific validity schema |
| `warp_displacement_heat.png` | Derived eye-pixel displacement visualization |
| `warp_mask.png` | Finite-source-boundary mask, not an internal visibility mask |
| `sbs.png` | Packed SBS output shown to the encoder/client |
| `window_video_border.json` | Matched-source browser-video rectangle and identity; optional for full-source dumps, required and hashed for ROI dumps |
| `dump_manifest.json` | Effective model, producer, renderer, color, state, and artifact authority |
| `adaptive_state.json` | Compatibility-shaped cut/depth-health telemetry |
| `meta.txt` | Human-readable summary; never the machine authority |

The V2 warp is an 11-step contractive inverse. Its mask marks requests beyond the finite source
rectangle, which the renderer clamps to an edge column. V2 has no internal forward owner,
multi-root selection, or synthesized fill.

### Optional window-video border evidence

When the Windows video-DOM observer has a fresh, uniquely identified rectangle stamped onto the
same captured frame, Dump 3D writes `window_video_border.json` (schema 2). The rectangle uses
half-open `[left, top, right, bottom)` coordinates in the exact `source.png` pixel space and binds
the source extent, matched frame ID, HWND/process/document/video identity, observer generation,
latest-heartbeat age, uninterrupted geometry age, and source-content age at capture. Validation
requires a fresh heartbeat and proves that the exact geometry run began no later than the source
content. A stale, noncausal, out-of-bounds, identity-incomplete, or frame-mismatched snapshot is
omitted from an ordinary full-source package. An ROI-active package instead requires and hashes the
artifact because `depth_input_region.json` cross-checks its semantic rectangle and exact
HWND/process/document/video identity.
The manifest summary always records the declared observer and screen-to-capture mapping status so
an omitted optional artifact has a machine-readable reason.
Package verification opens and validates the file whenever the manifest advertises it as
available, and cross-checks its frame ID and source extent against `dump_manifest.json`. A missing,
malformed, or mismatched advertised artifact fails verification. An unavailable artifact remains
non-authoritative and is not consulted.

The border artifact has no renderer authority by itself. `depth_input_region.json` (schema 1) is
the authority that distinguishes `full-source` from `video-region`.

For ROI it records the semantic rectangle, deterministic inward inference rectangle, independent
analysis generation/domain-reset state, tensor extent, observer authorization, exact float renderer
scale, and signed outside-only slope collar. The inference rectangle must be the production
planner's centered inward fit, never padding, stretching, or outward expansion, and may trim at
most 2% of semantic area.

ROI verification requires hashes for the region, semantic border, and full-source inverse map. It
recomputes the deterministic float32 planner and renderer constants, validates crop/full dimensions,
and examines the selected `warp_map.f32`. On full-source samples beyond maximum conservative collar
support, the raw inverse map must equal the unwarped source coordinate within 0.005 output-eye pixel.
The report records maximum embedded parallax, horizontal/vertical collar burden in source pixels,
exterior sample count, beyond-collar count/fraction, and maximum identity error. A near-full ROI may
honestly report zero beyond-collar raster samples when its entire visible surround lies inside the
required slope collar.

The existing `sbs_stereo_window_metrics.py` metric scores the fitted display's outer lateral
border. It cannot be applied unchanged to this inner rectangle: a video-frame metric must restrict
support to the rectangle and measure subject disparity relative to the visible frame's own
binocular disparity before reusing the existing component and perceptual weighting machinery.

## Current schema boundaries

Several independent schemas coexist. A schema number is meaningful only with its
named contract:

| Contract | Current schema | Scope |
|---|---:|---|
| Depth Coordinate V2 | 27 | Calibrated producer, optional known-mask overlay analysis/conditioning, authorization state, constants, and complete shader closure |
| Direct-geometry harness | 25 | Independent comparison-only replay input |
| Dump manifest | 13 | One Dump 3D package, analysis domain, and artifact authority |
| Depth-input region | 1 | Full-source or crop-local analysis placement and live renderer embedding |
| V2 state dump | 16 | Serialized scene-camera and renderer-authorization state |
| V2 frame statistics | 2 | Serialized frame-local statistics |
| Warp-map shape | 2 | Renderer-specific exact source-map layout and mask meaning |
| Window-video border | 2 | Optional matched-source rectangle, identity, freshness, and causal-continuity evidence |
| Raw-model provenance | 3 | Model/input/preprocess-to-raw binding |
| Evaluator | 36 | `run_eval.py` gating and result semantics |
| Production harness | 20 | Ordinary V2 clip artifact contract |
| Direct sequence replay | 18 | Whole-shot exact direct-geometry replay |

The direct-geometry harness uses schema 25. It is not an older copy of the Depth
Coordinate contract and must not be renumbered when that production contract changes. In this
mode, the harness runs the fixed production estimator only to obtain matched cut-state and resolved
tensor-shape evidence, then renders the supplied external final-parallax field. That estimator work
does not make the external geometry model-produced.

## Raw-model provenance

Exact geometry and authoritative model provenance are separate claims:

- **Exact dump geometry** hashes and consumes captured `raw_depth.f32`, publishes separately
  attributable V2 intermediate fields, and authenticates the rendered final field through the
  direct-geometry harness.
- **Authoritative raw-model provenance** additionally binds the model key and URL, ONNX SHA-256,
  preprocessing profile and source closure, exact input tensor and shape, and the raw-depth hash.

Malformed, contradictory, or hash-mismatched proof always aborts. The unverified option cannot
downgrade corrupt proof into valid evidence. “Authoritative” authenticates model-to-raw identity;
it does not certify perceptual quality, HDR preview fidelity, or metric ground truth.

Legacy captures without raw-model attestation fail closed by default. They may be used only for an
explicitly unverified geometry experiment:

```powershell
python tools/sbsbench/replay_depth_mapping_v2.py `
  --dump E:/ApolloDev/sbs_dump/dump_... `
  --out E:/ApolloDev/mapping-v2/replay `
  --allow-unverified-model-provenance `
  --experimental-raw-coordinate-scale 0.5
```

Both options are required because an old model name or URL does not prove the calibration scale.
The report records `experiment-unverified`, the reason, and the exact raw hash. The replay
environment still initializes the pinned production model, but its generated geometry is replaced
by the authenticated direct fields and has no provenance or geometry role in this experiment.

Production accepts only the model identities and fitted tensor shapes bound by coordinate schema
27. The canonical allowlist and custom-resolution behavior are documented once under
[authenticated resolution fitting](../../docs/host-sbs.md#authenticated-resolution-fitting).

## Exact sequence replay

`replay_depth_mapping_v2_sequence.py` uploads an authenticated raw sequence and cut generations to
one persistent GPU state, dispatches the same seven base V2 coordinate compute passes as production
(overlay-active live dumps are rejected until this package carries their mask/conditioner authority),
and renders the final field through the exact D3D harness:

```powershell
python tools/sbsbench/replay_depth_mapping_v2_sequence.py `
  --raw-seq cmake-build-relwithdebinfo/sbs_eval/<run>/<clip> `
  --frames tools/sbsbench/clips/<clip> `
  --out E:/ApolloDev/depth-coordinate-v2-exact/<clip>
```

Replay input frames must be lossless `frame_<id>.png`. Pillow and WIC can round JPEG differently
at hard ownership gates, so identical JPEG bytes do not guarantee identical decoded pixels.

The wrapper requires consecutive identities and complete matching raw, source, state, SBS, and
warp-map sets. It binds the evaluator result, model URL, ONNX and engine hashes, preprocessing
closure, V2 calibration, tensor shape, ordered raw hashes, cut pulses, and camera generation. The
native harness rejects every V2 constant that differs from the generated contract; only the
explicit pop treatment may vary.

This native GPU sequence replay consumes the authenticated raw tensors directly and does not
execute TensorRT. Its timing therefore covers coordinate-state production and rendering, not model
inference.

Schema-13 dumps capture both ordinary full-frame and foreground-Chromium ROI domains. The standalone
`replay_depth_mapping_v2.py` command still supports only full-source dumps: it fails closed on an ROI
package before creating output because the direct replay harness does not yet consume
`depth_input_region.json`, crop-local temporal state, full-source scale, or the outside-only collar.
This restriction applies to replay, not to capture or package verification.

The state trace records field validity, retained scene-camera validity, confirmed-cut attribution,
the ABI-retained `container_scale = 1.0`, effective requested gain, and source/render identity. The
pointwise soft container is stateless map-local geometry, so no frame extreme can rescale the
remaining field. An unusable field renders its
current color flat. Without a cut it retains only the scene camera; after a confirmed cut the next
usable field acquires a new camera. Old per-pixel depth is never paired with current color.

NumPy recomputes the coordinate after native execution as a comparison oracle. Its tolerances must
pass, but its fields never drive the renderer or authoritative state trace. Diagnostic mean,
spread, extrema, center drift, pointwise-container burden, and limiter burden are non-controlling
evidence.

The native sequence replay bypasses model inference and consumes authenticated raw fields. Its
compact cut-state buffer has cut evidence only. The authoritative geometry/controller evidence is
`depth_coordinate_v2_state_trace.json`; compact cut-state scores stay isolated from
renderer-quality decisions.

## Direct dump scoring

Score matched dump groups with:

```powershell
python tools/sbsbench/sbsbench.py --glob "E:/ApolloDev/sbs_dump/dump_2026*" --json base.json
python tools/sbsbench/sbsbench.py --glob "E:/ApolloDev/sbs_dump/dump_NEW*" --baseline base.json
```

The scorer reports per-metric deltas. It does not make an older manifest authoritative, and a PNG
preview never substitutes for a missing numeric artifact.
