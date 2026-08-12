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
| `shadow_base_final_parallax.f32` | Required in active SLR6 packages; ordinary post-limiter field before subtitle conditioning |
| `shadow_final_parallax.f32` | Selected final model-domain field, after SLR6 when active |
| `warp_depth.f32` | Exact field consumed by reprojection |
| `shadow_state.json`, `shadow_frame_stats.json` | Typed V2 state and current-frame statistics |
| `warp_map.f32`, `warp_map_shape.json` | Exact inverse map; required for ROI packages |
| `warp_mask.png` | Boundary-extrapolation diagnostic |
| `window_video_border.json` | Required semantic observation for ROI packages |
| `sbs.png` | Packed stereo preview |
| `subtitle_conditioning.json` | Required current subtitle-authority descriptor; canonical `none` or `subtitle-slr6` |
| `subtitle_ocr_record.u32` | Active-only exact-frame OCR6 record: 208 little-endian uint32 words |
| `subtitle_locator_state.u32` | Active-only compact SLR6 state: 80 little-endian uint32 words |

All `.f32` files are little-endian float32. Schema 21 accepts the canonical inactive descriptor or
the one current `subtitle-slr6` package. It binds Depth Coordinate V2 schema `40`, tag
`0xf7c853c2`, and producer source closure
`861f800db1cc06a6d25b80d18bb1b7bf4bc469ed1ccfacfd96a0b384bfb2a7a1`. The active descriptor
also binds the authenticated PP-OCRv6
tiny detector (`ppocrv6_tiny_det`, official FP32 ONNX SHA-256
`193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8`) and fixed
strongly typed FP32+TF32 TensorRT recipe `trt-strong-fp32-tf32-fixed960x160-level5-v1`. Its authenticated
profile takes BGR/ImageNet `x` as FP32 NCHW
`[1,3,160,960]` from the bottom 6:1 crop and emits FP32 `fetch_name_0` as
`[1,1,160,960]`; `120x160` is only the later eight-pixel aggregation grid. The descriptor also
binds the current Depth Coordinate V2 shader closure, exact OCR producer and locator entrypoints,
and four artifact roles (OCR record, locator state, Base field, selected field). No retired layout
is preserved or reinterpreted.

## OCR6 and SLR6 records

OCR6 has schema `1`, tag `OCR6` (`0x3652434f`), and exactly 208 words. Its 16-word header contains
authority flags, raw/final counts, frame and analysis-generation identities, exact analysis-input
and field extents,
the `[325,430)` OCR field-row interval, and a zero reserved word. Sixteen raw and eight final box
slots follow; each eight-word slot is half-open `left,top,right,bottom`, float32 score bits, zero box
flags, and two zero reserved words. Unused slots are canonical zero. Flag `1` is authoritative,
including an authoritative empty observation; flag `0` abstains and cannot publish boxes.

SLR6 has schema `6`, tag `SLR6` (`0x36524c53`), and exactly 80 words: a 32-word header, four owner
rectangles, four pending rectangles, and four same-frame current-authority rectangles. The header
binds owner/pending aggregates, target and generation state, exact frame/analysis identities,
event/fade/grace state, scene epoch, and field extent. Rectangles are half-open field coordinates;
unused slots and the reserved header word are zero. Only the current block authorizes conditioning.
Nonempty current authority requires a live owner, matching target generation, and fade step `1` or
`2`; each current rectangle must be an exact final box from the same-frame OCR6 record.
An empty current block requires `shadow_final_parallax.f32` to equal
`shadow_base_final_parallax.f32` bit for bit. For a nonempty current block, each integer field
sample uses its minimum distance to the half-open rectangles: horizontal distance is weighted by
`max_horizontal_slope/770`, vertical distance by `max_vertical_shear/770`, and
`0.5/analysis_source_width` is added as the core budget. Base values already within that budget of
the owner target remain bit-exact; other values clamp to the target plus signed budget. Fade step
`1` publishes the float32 halfway value from Base to that clamp, and step `2` publishes the clamp.

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
7. validates canonical inactive metadata, or the exact current OCR6/SLR6 model, shader, record,
   state, and artifact identities;
8. replays the ordinary V2 chain into `shadow_base_final_parallax.f32` when SLR6 is active, then
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
