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
| `depth_input_source.png` | Exact full-source preview, or post-completion reconstruction of the logical window-region analysis source; diagnostic only |
| `depth_input_region.json` | Required half-open analysis-domain and ROI/full-source placement |
| `model_input.f32`, `model_input_shape.json` | Exact calibrated DAV2 input |
| `raw_depth.f32`, `raw_shape.json` | Exact DAV2 output |
| `shadow_candidate_parallax.f32` | Pre-limiter V2 candidate |
| `shadow_ownership_refined_parallax.f32` | Full-resolution contour ownership result |
| `shadow_vertical_majorant.f32` | Vertical upper-envelope diagnostic |
| `shadow_vertical_conditioned.f32` | Fixed vertical-share result |
| `shadow_base_final_parallax.f32` | Required in active SLR13 packages; ordinary post-limiter field before subtitle conditioning |
| `shadow_final_parallax.f32` | Complete atomic conditioned model-domain field; sole live render authority |
| `warp_depth.f32` | Exact final field consumed by reprojection; bit-identical to `shadow_final_parallax.f32` |
| `shadow_state.json`, `shadow_frame_stats.json` | Typed V2 state and current-frame statistics |
| `warp_map.f32`, `warp_map_shape.json` | Exact inverse map; required for ROI packages |
| `warp_mask.png` | Boundary-extrapolation diagnostic |
| `window_region.json` | Required semantic observation for ROI packages |
| `sbs.png` | Packed stereo preview |
| `subtitle_conditioning.json` | Required current subtitle-authority descriptor; canonical `none` or `subtitle-slr13` |
| `subtitle_ocr_record.u32` | Active-only OCR8 record for the atomic target's authenticated subtitle-publication frame: current on ordinary infer or due work, older on held ordinary reuse |
| `subtitle_locator_state.u32` | Active-only compact SLR13 state in generated-contract word order |
| `gpu_trace_ring.u32` | Optional raw diagnostic history of the last 300 accepted-root completions; available only when diagnostics was enabled before reproduction |
| `gpu_trace.json` | Optional chronological decode of the authenticated raw GPU trace |
| `gpu_trace_contract.json` | Optional exact trace offsets, enums, receipt ABI, and shader provenance |

All `.f32` files are little-endian float32. Schema 37 accepts the canonical inactive descriptor or
the one current `subtitle-slr13` package. It binds the exact current generated Depth Coordinate V2
identity, producer and renderer closures, OCR model provenance, entrypoints, and four artifact
roles (OCR record, locator state, Base field, atomic final field). The generated contract is the sole
owner of those live identities and numeric policy values. The
[Host SBS OCR-box subtitle conditioner](../../docs/host-sbs.md#ocr-box-subtitle-conditioner) is the
sole owner of their runtime and state-machine semantics; this dump-format document copies neither.
No retired layout is preserved or reinterpreted.

## Atomic final field

The manifest's required `final_parallax` object binds one complete atomic field and the warp input.
Depth infer publishes a new V2 Base, while authenticated depth reuse retains Base. Subtitle work is
independent inside that joined publication: ordinary work publishes only on infer and holds on
reuse; cadence-due work publishes current OCR or current abstention on either depth branch. The
conditioned final field follows that authenticated combination; invalid publication fails closed.
There is no second persistent display resource or temporal display recurrence. In every package,
`warp_depth.f32` must equal `shadow_final_parallax.f32` exactly, while the verifier replays the
ordinary limiter and, when active, SLR13 directly into that final field.

## Diagnostic GPU completion trace

Schema 37 may carry a diagnostic-only 300-slot GPU completion ring. It records completed accepted
DAV2 roots, not source frames, presentation frames, busy drops, or every captured desktop update.
Each 176-word (704-byte) record binds an exact trace ordinal, matched frame, analysis generation,
analysis-domain tag, transaction token, analysis-source and DAV2-field extents, the immutable
64-word postprocessed transaction snapshot, all 80 SLR13 words, and all six condition-parameter
words. The authenticated RQST/CBRG token, cookies, work disposition, optional OOCR marker, and
submission class determine `infer`, `reuse`, or `invalid`; a force-class reuse receipt is invalid,
never inferred as reuse. Work `1` is ordinary current-ready OCR, `2` ordinary abstention, `8`
cadence-due current-ready OCR, and `16` cadence-due ineligible abstention. Ordinary work is
infer-coupled; due work is branch-independent. `OOCR` is valid only for work `1` on infer or work
`8` on either branch, and is always absent for work `2`/`16`. Subtitle disposition is cross-checked
against expected work, host outcome, raw flags, and the authenticated device receipt. An ordinary
reuse is `held_with_depth`: OCR8, SLR80, condition6, and the atomic conditioned target remain the
prior coherent tuple. A due reuse is instead `optional_ocr` for work `8` or `abstention` for work
`16`; it advances the subtitle tuple and reconditions retained immutable Base.

The shared live/offline policy makes subtitle work due after two accepted ordinary opaque dirty
holds or `33 ms` of source observation time since the last guaranteed subtitle observation. The
device infer owner is reusable through at most four frame steps and only with a nonregressed source
observation age strictly below `100 ms`; the host's initial-candidate and opaque-follow-up freshness
checks use the same strict `100 ms` ceiling. Each trace record carries the source-observation
timestamp; its frame identity and authenticated decision history expose the device half of those
bounds without a production readback.

The ring writer invalidates the header tag before overwriting any slot, commits payload before the
record tag, then updates the cursor and republishes the header tag last. The reader reconstructs
wrap from `next_sequence`, `next_slot`, and `committed_count`, rejects torn/non-consecutive records,
and requires a record matching the dump frame, generation, full analysis-domain hash, analysis
extent, field extent, and domain-reset bit. `dump_forced_at_enqueue` is perturbation provenance:
it is false when a dump request harvests a root that was already pending, and is therefore not a
requirement for the matched record.

For non-suppressed authenticated subtitle publications, SLR80 and condition6 are captured after SLR
resolve and condition-parameter publication. These are ordinary infer publications or due
publications on either depth branch. A held ordinary reuse captures the same bytes with the
locator's older frame identity. When its predecessor remains in the ring, both stored tuples must be
byte-identical to that immediately prior record. The raw ring is schema 3; the decoded
`gpu_trace.json` shape is schema 4. A nonzero
SLR `current_count` requires the exact six-word condition tuple; normal authority uses
durable target/fade words 18/24, while provisional-current flag bit 4 uses ephemeral words 29/30.
A zero `current_count` requires the conditioner's canonical zero6 Base verdict. Suppression freezes
SLR, skips conditioning, and publishes immutable Base as the atomic final field; its stored SLR/condition
sections may consequently be frozen or stale and are explicitly marked unused. The raw ring,
decoded JSON, and trace contract
are each SHA-256 bound by the manifest. The contract authenticates
`host_sbs_gpu_trace_cs.hlsl:main:cs_5_0`, its source-closure schema/flags/macro count/hash, every
header/record/constant offset, receipt cookies/tags, enum value, and embedded state schema.

Trace setup, staging, validation, or publication failure makes all three artifacts canonically
unavailable and never rejects or weakens the core Dump 3D package. With diagnostics disabled the
trace shader/resources are not created and no append, copy, or dispatch runs; enabling Dump 3D
afterward cannot recover prior history. With diagnostics enabled, the trace still adds no per-frame
CPU map, readback, query, or synchronization and has no rendering authority.

## OCR8 and SLR13 records

The generated `subtitle_ocr.ocr_record` contract owns OCR8's schema, tag, word counts, offsets,
capacities, flags, and numeric detector/box policy values. The dump serializes exactly that fixed
word array as little-endian uint32 values. Its frame, analysis generation, analysis-source extent,
tensor-content geometry, projected ROI, paired core/cover records, topology metadata, and canonical
zero tail must all agree with the atomic target's publication frame. That is the matched frame for
ordinary infer and either depth branch of a due publication, and the trace-authenticated older tuple
frame for held ordinary reuse. See the
[live OCR8 contract](../../docs/host-sbs.md#ocr-box-subtitle-conditioner) for producer semantics.

The generated `subtitle_ocr.locator_state` contract owns SLR13's schema, the unambiguous
little-endian `SL13` tag bytes, word layout,
rectangle capacity, kind packing, numeric qualification/death-grace limits, and the complete
local-supporting-plane target policy. Its resolver descriptor explicitly binds the symmetric
ordinary-core corner rejection: edge clearance is strictly below `floor(content_width / 32)`, the
core reaches `dynamic_ROI_bottom - 16`, threshold equality is accepted, and ribbons are exempt.
The resolver descriptor serializes the aggregate-center
primary policy exactly as `binocular-source-pixels`: two independent 16-sample rows; median indices
`7/8`; both complete finite in-container rows bypass the IQR gate; a row-median difference of `4`
is the both-valid mean-versus-maximum-median selection boundary; and a sole valid row is accepted
only when its Tukey IQR at indices `3/4` and `11/12` is at most `8`. Schema 37 also authenticates
the strict primary-failure fallback: ordinary-core span
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
kind mask, and canonical zero slot, and requires current covers to come from the OCR8 record for
the same target-publication frame. The provisional-current descriptor also owns flag bit 4, target/fade words 29/30, the exact
single-ordinary mature-owner geometry bounds, and the requirement that the dump's matched record
replay the exact selected OCR8 raw-core/final-cover pair. Historical trace rows have no OCR8 payload,
so they validate the same state geometry, cover containment, and condition tuple structurally; they
do not claim exact historical OCR replay. See the
[live SLR13 contract](../../docs/host-sbs.md#ocr-box-subtitle-conditioner) for overlap, cut-survival,
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
and outside-only collar in `depth_input_region.json` schema `4`. The source rectangle is never
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
5. requires `warp_depth.f32` to equal the atomic final field bit-for-bit;
6. validates ROI placement, authority-specific window identity, inverse-map geometry, and the exterior
   zero-plane evidence; and
7. validates canonical inactive metadata, or the exact authenticated current-subtitle-publication/
   held-ordinary-reuse OCR8/SLR13 model, shader, record, state, and artifact identities;
8. replays the ordinary V2 chain into `shadow_base_final_parallax.f32` when SLR13 is active, then
   replays the exact content-clamped analytic rectangle budget and fade into
   `shadow_final_parallax.f32` (including exact nearest-content Base extension when current
   authority is empty). SM5 division is checked against the finite globally consistent one-ULP
   result set permitted for the three power-of-two divisions; this is not an epsilon comparison; and
9. when the optional GPU trace is available, authenticates all three hashes, the exact trace shader
   closure and ABI document, chronological wrap/commit structure, raw branch/OCR proof, matched
   analysis domain, and bit-for-bit decoded JSON. An unavailable trace remains valid optional state.

Use `.f32` artifacts for quantitative work. Every generated scalar-field shape sidecar and PNG
preview is described by `dump_manifest.json`; the pre-SLR13 Base previews remain optional,
non-authoritative schema-37 diagnostics for compatibility with already captured packages.
Independently stretched PNG previews are diagnostic and must not be compared as absolute depth
values.

## Commands

Validate the Python contract and geometry reader from the repository root:

```powershell
E:\ApolloDev\modelopt-py312\Scripts\python.exe -m unittest tools.sbsbench.test_depth_coordinate_v2_dump_contract
```

Score a capture set with `tools/sbsbench/sbsbench.py`; see [README.md](README.md) for the maintained
evaluation commands and [METRICS.md](METRICS.md) for metric authority.
