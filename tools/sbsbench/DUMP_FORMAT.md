# Dump 3D format

Dump 3D is one atomic, matched-frame diagnostic package for the authenticated Host SBS V2 path.
The writer and strict reader accept one current manifest schema only. Retired SLR3--SLR9,
GST/OGR/ORS, and offline overlay-detector packages are intentionally unsupported.

## Current package

`dump_manifest.json` binds the exact source frame, analysis domain, model/preprocess provenance,
Depth Coordinate V2 contract and shader closures, renderer selection, dimensions, artifact
availability, and SHA-256 hashes of numeric and state authority files. `capture_status` is
`complete` and `published_atomically` is `true`; a later frame never impersonates an unavailable
requested frame. The manifest's V2 state summary is an exact authenticated mirror of the required
`shadow_state.json`, whose frame-valid bit is cross-checked against the required same-frame
`shadow_frame_stats.json`.

The core package contains:

| Artifact | Authority |
|---|---|
| `source.png` | Matched full-source color preview |
| `depth_input_source.png` | Exact full-source or full window-region analysis-color preview |
| `depth_input_region.json` | Required half-open analysis-domain and ROI/full-source placement |
| `model_input.f32`, `model_input_shape.json` | Exact calibrated DAV2 input |
| `raw_depth.f32`, `raw_shape.json` | Exact DAV2 output |
| `shadow_candidate_parallax.f32` | Pre-limiter V2 candidate |
| `shadow_ownership_refined_parallax.f32` | Full-resolution contour ownership result |
| `shadow_vertical_majorant.f32` | Vertical upper-envelope diagnostic |
| `shadow_vertical_conditioned.f32` | Fixed vertical-share result |
| `shadow_base_final_parallax.f32` | Required in active SLR12 packages; ordinary post-limiter field before subtitle conditioning |
| `shadow_final_parallax.f32` | Selected final model-domain field, after SLR12 when active |
| `warp_depth.f32` | Exact field consumed by reprojection |
| `shadow_state.json`, `shadow_frame_stats.json` | Typed V2 state and current-frame statistics |
| `warp_map.f32`, `warp_map_shape.json` | Exact inverse map; required for ROI packages |
| `warp_mask.png` | Boundary-extrapolation diagnostic |
| `window_region.json` | Required semantic observation for ROI packages |
| `sbs.png` | Packed stereo preview |
| `subtitle_conditioning.json` | Required current subtitle-authority descriptor; canonical `none` or `subtitle-slr12` |
| `subtitle_ocr_record.u32` | Active-only exact-frame OCR8 record in generated-contract word order |
| `subtitle_locator_state.u32` | Active-only compact SLR12 state in generated-contract word order |

All `.f32` files are little-endian float32. Schema 32 accepts the canonical inactive descriptor or
the one current `subtitle-slr12` package. It binds the exact current generated Depth Coordinate V2
identity, producer and renderer closures, OCR model provenance, entrypoints, and four artifact
roles (OCR record, locator state, Base field, selected field). The generated contract is the sole
owner of those live identities and numeric policy values. The
[Host SBS OCR-box subtitle conditioner](../../docs/host-sbs.md#ocr-box-subtitle-conditioner) is the
sole owner of their runtime and state-machine semantics; this dump-format document copies neither.
No retired layout is preserved or reinterpreted.

## OCR8 and SLR12 records

The generated `subtitle_ocr.ocr_record` contract owns OCR8's schema, tag, word counts, offsets,
capacities, flags, and numeric detector/box policy values. The dump serializes exactly that fixed
word array as little-endian uint32 values. Its frame, analysis generation, analysis-source extent,
tensor-content geometry, projected ROI, paired core/cover records, topology metadata, and canonical
zero tail must all agree with the manifest's selected exact frame. See the
[live OCR8 contract](../../docs/host-sbs.md#ocr-box-subtitle-conditioner) for producer semantics.

The generated `subtitle_ocr.locator_state` contract owns SLR12's schema, the unambiguous
little-endian `SL12` tag bytes, word layout,
rectangle capacity, kind packing, numeric qualification/death-grace limits, and the complete
local-supporting-plane target policy. The resolver descriptor serializes the aggregate-center
primary policy exactly as `binocular-source-pixels`: two independent 16-sample rows; median indices
`7/8`; Tukey-IQR lower indices `3/4` and upper indices `11/12`; at least one coherent row with IQR
at most `8`; and a row-median difference of `4` as the both-valid mean-versus-maximum-median
selection boundary. Schema 32 also authenticates its primary-failure fallback: ordinary-core span
step `W/16`, maximum radius two, negative then positive order, ordinary-over-ribbon placement,
unclamped 61-cell strips, two coherent rows and at most `4` pixels of intra-probe median separation.
At one radius, two qualifying mean targets must themselves agree within `4` pixels; conflict stops
the search and makes the observation unreliable. It also binds deadband `1`, EMA alpha `0.125`,
maximum slew `0.25`, maximum continuing residual `8`,
and at most two distinct continuing same-scene unreliable-measurement holds in owner state word
25. Only current authority increments that counter; duplicates do not age it, absent current
authority may preserve it without conditioning, and hard cuts cannot hold it. The signed
direct-parallax container is the target's only representation limit. The dump
serializes exactly that fixed word array as little-endian uint32 values. The reader authenticates
every identity, flag, aggregate, rectangle, target, event, fade, contextual hold/grace counter,
kind mask, and canonical
zero slot, and requires current covers to come from the same-frame OCR8 record. See the
[live SLR12 contract](../../docs/host-sbs.md#ocr-box-subtitle-conditioner) for overlap, cut-survival,
target, fade, death-grace, and conditioning semantics.

The Python conditioner replay below is deliberately an independent dump-integrity verifier. It
recomputes the frozen SM5 operation order, including the bounded one-ULP division alternatives, to
reject corrupt or internally inconsistent packages. It is not a second production policy and is
not the shader regression gate; live behavior remains qualified by the generated contract and the
native WARP/GPU tests.

## Analysis domains and window provenance

A full-source package uses analysis generation zero. A window-region package binds a nonzero
analysis generation, the exact uncropped source rectangle, tensor extent, centered integer
contain-fit `tensor_content_rect_px`, edge-replicated excluded padding fraction, unit conversion,
and outside-only collar in `depth_input_region.json` schema `3`. The source rectangle is never
stretched or trimmed: a wider region pads above/below and a taller region pads left/right. Padding
does not participate in model statistics, scene-cut evidence, ownership, OCR, or subtitle state;
published fields extend the nearest content boundary through it. Crop-local depth must never be
interpreted as a full-source field. The required full-source inverse map proves that samples beyond
the conservative collar return to identity.

`window_region.json` schema `1` is the matched-frame provenance behind that planner input, not an
independent geometry or renderer authority. Its `authority_kind` is decisive:

- `chromium-video` requires HWND, PID, observer generation, and nonzero Chromium document/video
  identities. Its freshness is the age of the latest semantic observation.
- `foreground-client` requires HWND, PID, and observer generation, while document/video identities
  are exactly zero because a native foreground window has no DOM identity. Its freshness is the age
  of the latest native foreground-window observation.

Both kinds bind the same half-open matched-source rectangle, source extent, causality checks, and
exact deterministic integer contain-fit planner validation. An authoritative window-region package requires
observer status `ok` and mapping status `ok`; diagnostic full-source packages may additionally
carry available Chromium `ok-fullscreen` provenance. Non-authoritative/unavailable statuses remain
diagnostic strings and never grant ROI authority.

## Strict verification

The maintained reader:

1. rejects every manifest schema except the current one;
2. checks exact DVC2 contract/tag/source-closure bindings;
3. verifies every required content hash and numeric extent;
4. replays ownership and the schema-selected limiter over the full tensor: serial float32 for
   lines up to 32 elements, otherwise conservative Q30 upper/lower envelopes and horizontal
   majorant with the authenticated 75/25 float32 share, using content width for both steps;
5. requires `warp_depth.f32` to equal the selected final field;
6. validates ROI placement, authority-specific window identity, inverse-map geometry, and the exterior
   zero-plane evidence; and
7. validates canonical inactive metadata, or the exact current OCR8/SLR12 model, shader, record,
   state, and artifact identities;
8. replays the ordinary V2 chain into `shadow_base_final_parallax.f32` when SLR12 is active, then
   replays the exact content-clamped analytic rectangle budget and fade into
   `shadow_final_parallax.f32` (including exact nearest-content Base extension when current
   authority is empty). SM5 division is checked against the finite globally consistent one-ULP
   result set permitted for the three power-of-two divisions; this is not an epsilon comparison.

Use `.f32` artifacts for quantitative work. Independently stretched PNG previews are diagnostic and
must not be compared as absolute depth values.

## Commands

Validate the Python contract and geometry reader from the repository root:

```powershell
python -m unittest tools.sbsbench.test_depth_coordinate_v2_dump_contract
```

Score a capture set with `tools/sbsbench/sbsbench.py`; see [README.md](README.md) for the maintained
evaluation commands and [METRICS.md](METRICS.md) for metric authority.
