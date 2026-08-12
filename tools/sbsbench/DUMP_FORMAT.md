# Dump 3D format

Dump 3D is one atomic, matched-frame diagnostic package for the authenticated Host SBS V2 path.
The writer and strict reader accept one current manifest schema only. Retired SLR3--SLR5,
GST/OGR/ORS, and offline overlay-detector packages are intentionally unsupported.

## Current package

`dump_manifest.json` binds the exact source frame, analysis domain, model/preprocess provenance,
Depth Coordinate V2 contract and shader closures, renderer selection, dimensions, artifact
availability, and SHA-256 hashes of numeric authority files. `capture_status` is `complete`;
a later frame never impersonates an unavailable requested frame.

The core package contains:

| Artifact | Authority |
|---|---|
| `source.png` | Matched full-source color preview |
| `depth_input_source.png` | Exact full-source or inward-cropped analysis-color preview |
| `depth_input_region.json` | Required half-open analysis-domain and ROI/full-source placement |
| `model_input.f32`, `model_input_shape.json` | Exact calibrated DAV2 input |
| `raw_depth.f32`, `raw_shape.json` | Exact DAV2 output |
| `shadow_candidate_parallax.f32` | Pre-limiter V2 candidate |
| `shadow_ownership_refined_parallax.f32` | Full-resolution contour ownership result |
| `shadow_vertical_majorant.f32` | Vertical upper-envelope diagnostic |
| `shadow_vertical_conditioned.f32` | Fixed vertical-share result |
| `shadow_base_final_parallax.f32` | Required in active SLR8 packages; ordinary post-limiter field before subtitle conditioning |
| `shadow_final_parallax.f32` | Selected final model-domain field, after SLR8 when active |
| `warp_depth.f32` | Exact field consumed by reprojection |
| `shadow_state.json`, `shadow_frame_stats.json` | Typed V2 state and current-frame statistics |
| `warp_map.f32`, `warp_map_shape.json` | Exact inverse map; required for ROI packages |
| `warp_mask.png` | Boundary-extrapolation diagnostic |
| `window_video_border.json` | Required semantic observation for ROI packages |
| `sbs.png` | Packed stereo preview |
| `subtitle_conditioning.json` | Required current subtitle-authority descriptor; canonical `none` or `subtitle-slr8` |
| `subtitle_ocr_record.u32` | Active-only exact-frame OCR8 record: 208 little-endian uint32 words |
| `subtitle_locator_state.u32` | Active-only compact SLR8 state: 80 little-endian uint32 words |

All `.f32` files are little-endian float32. Schema 26 accepts the canonical inactive descriptor or
the one current `subtitle-slr8` package. It binds Depth Coordinate V2 schema `45`, tag
`0xfbd3cdb1`, canonical SHA-256
`8515cf7bc352c2e9e56e6a5fd9dad9802e1e7cd02f705fd8a957617c7ba94e9a`, and producer
source closure `11bd8c0ab14d22caab83044e5f0d38cca10f5eef5df5d5e671cbedf64e256a1f`.
The active descriptor
also binds the authenticated PP-OCRv6 tiny detector
`ppocrv6_tiny_det_modelopt_fp16`: bundled artifact
`models/ppocrv6_tiny_det_modelopt045_mixed_fp16_fp32io.onnx` with SHA-256
`169a233ba0ff7cac27f8ec7dccb6a406e614b25b21fe6a5638c423bf2118bb44`, independently pinned
upstream FP32 ONNX SHA-256
`193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8`, NVIDIA ModelOpt
`0.45.0` conversion recipe `nvidia-modelopt-autocast-fp16-keep-io-fp32-v1`, and calibration
profile `apollo-live8-bottom960x160-v1`. Its fixed strongly typed TensorRT recipe is
`trt-strong-modelopt045-fp16-iofp32-tf32-fixed960x160-level5-v2`. Its authenticated
preprocess profile takes BGR/ImageNet `x` as FP32 NCHW
`[1,3,160,960]` from the bottom 6:1 crop and emits FP32 `fetch_name_0` as
`[1,1,160,960]`; `120x160` is only the later eight-pixel aggregation grid. The descriptor also
binds the current Depth Coordinate V2 shader closure, exact OCR producer and locator entrypoints,
and four artifact roles (OCR record, locator state, Base field, selected field). No retired layout
is preserved or reinterpreted.

## OCR8 and SLR8 records

OCR8 has schema `3`, tag `OCR8` (`0x3852434f`), and exactly 208 words. Its 16-word header contains
authority flags, raw/final counts, frame and analysis-generation identities, exact analysis-input
and field extents, the source/aspect-derived OCR field-row interval, and a zero reserved word.
The interval projects detector rows `[24,155)` through the exact bottom-6:1 source crop into the
active calibrated DAV2 field. Sixteen raw and eight final slots follow, but authoritative records
contain at most eight strictly paired slots with equal counts. Each slot is half-open
`left,top,right,bottom`, float32 score bits, kind flags (`1` means ribbon), island count, and
structural-gap count. A final cover contains its paired raw core and repeats identical metadata.
Ordinary pairs remain ROI-bound. A ribbon core is bottom-attached, wide, and segmented. Its bottom
edge lies inclusively between the exact rational ceil projection of detector row `153`
(`SAFE_ROW_BOTTOM 155 - tolerance 2`) and the projected ROI bottom; the projection uses the same
bottom-6:1 source crop and active DAV2 field. Its final cover is exactly
`[0, corrected_top, field_width, field_height)`. Unused slots are canonical zero.
Flag `1` is authoritative,
including an authoritative empty observation; flag `0` abstains and cannot publish boxes.

Ribbon topology is classified from an unfiltered broad run joined across at most 12 inactive
eight-pixel cells. If that run is not a ribbon, it is always rescanned into ordinary subruns with a
four-cell join allowance, regardless of whether the broad aggregate itself passes ordinary
evidence. An island contributes to an ordinary subrun only when its own tight width and height are
both at least three detector pixels and its own mean score is at least `0.4`; bounds, score, island
count, and structural-gap count are then recomputed per retained subrun. The broad pass deliberately
does not apply this island filter, so segmented bottom indexes keep their complete topology.

SLR8 has schema `8`, tag `SLR8` (`0x38524c53`), and exactly 80 words: a 32-word header, four owner
rectangles, four pending rectangles, and four same-frame current-authority rectangles. The header
binds owner/pending aggregates, target and generation state, exact frame/analysis identities,
event/fade/grace state, scene epoch, and field extent. Rectangles are half-open field coordinates;
unused slots are zero. Header word 31 packs owner/pending/current ribbon masks at shifts 0/4/8;
bits 12..31 and mask bits beyond each corresponding count are zero. Only the current block
authorizes conditioning.
Ordinary rectangles form one transitive component through either the vertically adjacent
coherent-line relation or the same-baseline relation. Same-baseline cores are horizontally
disjoint, overlap vertically by at least three quarters of the shorter height, have height ratio at
most two, doubled vertical-center separation at most the shorter height, horizontal gap at most
eight times the taller height, and pair span at most `floor(9 * field_width / 10)`. The complete
transitive component must also fit that maximum span. Grouping affects selection and the shared
target only: cores and covers remain independent, canonical core order is top then left, and
current covers retain the order of their paired cores. More than four selected rectangles abstain
rather than being silently dropped.
The target is sampled once on a fresh confirmed birth and then retained bit-exact for a continuing
owner. Any confirmed handoff or rebirth during death grace inherits the cached target independent
of overlap. A hard-cut survivor increments owner generation
and resamples at full strength, while an input-domain reset clears owner and target before starting
present geometry as pending. There is no target EMA or per-observation rate limiter.
Nonempty current authority requires a live owner, matching target generation, and fade step `1` or
`2`; each current rectangle must be an exact final box from the same-frame OCR8 record.
An empty current block requires `shadow_final_parallax.f32` to equal
`shadow_base_final_parallax.f32` bit for bit. For a nonempty current block, each integer field
sample uses its minimum distance to the half-open rectangles: horizontal distance is weighted by
`max_horizontal_slope/field_width`, vertical distance by `max_vertical_shear/field_width`, and
`0.5/analysis_source_width` is added as the core budget. Base values already within that budget of
the owner target remain bit-exact; other values clamp to the target plus signed budget. Fade step
`1` publishes the float32 halfway value from Base to that clamp, and step `2` publishes the clamp.
Ordinary text rectangles use the four-sided horizontal-plus-vertical distance. A ribbon rectangle
instead replays `dx = 0` everywhere and `dy = max(top - y, 0)`. Together with its authenticated
full-width/to-bottom current cover, this gives the constant ribbon plane a collar only above its
corrected top edge; never at a nominal side or bottom edge.

## Analysis domains

A full-source package uses analysis generation zero. A video-region package binds a nonzero
analysis generation, the exact inward crop, tensor extent, unit conversion, and outside-only collar
in `depth_input_region.json`. Crop-local depth must never be interpreted as a full-source field.
The required full-source inverse map proves that samples beyond the conservative collar return to
identity. `window_video_border.json` is semantic planning evidence, not independent geometry
authority.

## Strict verification

The maintained reader:

1. rejects every manifest schema except the current one;
2. checks exact DVC2 contract/tag/source-closure bindings;
3. verifies every required content hash and numeric extent;
4. replays the ownership, vertical-share, and horizontal-majorant recurrences in float32 order;
5. requires `warp_depth.f32` to equal the selected final field;
6. validates ROI placement, semantic-border identity, inverse-map geometry, and the exterior
   zero-plane evidence; and
7. validates canonical inactive metadata, or the exact current OCR8/SLR8 model, shader, record,
   state, and artifact identities;
8. replays the ordinary V2 chain into `shadow_base_final_parallax.f32` when SLR8 is active, then
   replays the exact analytic rectangle budget and fade into `shadow_final_parallax.f32` (including
   bit-exact Base passthrough when current authority is empty).

Use `.f32` artifacts for quantitative work. Independently stretched PNG previews are diagnostic and
must not be compared as absolute depth values.

## Commands

Validate the Python contract and geometry reader from the repository root:

```powershell
python -m unittest tools.sbsbench.test_depth_coordinate_v2_dump_contract
```

Score a capture set with `tools/sbsbench/sbsbench.py`; see [README.md](README.md) for the maintained
evaluation commands and [METRICS.md](METRICS.md) for metric authority.
