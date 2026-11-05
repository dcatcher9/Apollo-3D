# Dump 3D format

Dump 3D is one atomic, matched-frame diagnostic package for the authenticated Host SBS V2 path.
The canonical geometry/metric reader accepts schema 51 only. Earlier schemas 44/45/46/47/48/49/50 and presentation-experiment
schemas 41/42/43 remain historical artifacts and are rejected by the current reader. Retired SLR3--SLR9,
GST/OGR/ORS, and offline overlay-detector packages are intentionally unsupported.

## Current package

`dump_manifest.json` binds the exact source frame, analysis domain, model/preprocess provenance,
Depth Coordinate V2 contract and shader closures, renderer selection, dimensions, artifact
availability, and SHA-256 hashes of numeric and state authority files. `capture_status` is
`complete` and `published_atomically` is `true`; a later frame never impersonates an unavailable
requested frame. The manifest's V2 state summary is an exact authenticated mirror of the required
`shadow_state.json`, whose frame-valid bit is cross-checked against the required same-frame
`shadow_frame_stats.json`, including mandatory P05/P95 histogram authority. The root
`matched_observation_timestamp_us` is a positive uint64 source-transaction clock from the completed
publication; it must match the optional ring's matched record. It is independent of GPU state and
never inferred from the camera or subtitle accepted clock.

The core package contains:

| Artifact | Authority |
|---|---|
| `source.png` | Matched full-source color preview |
| `depth_input_source.png` | Exact full-source preview, or post-completion reconstruction of the logical window-region analysis source; diagnostic only |
| `depth_input_region.json` | Required half-open analysis-domain and ROI/full-source placement |
| `model_input.f32`, `model_input_shape.json` | Exact fused public model input and calibrated high-resolution preprocess authority |
| `raw_depth.f32` | Exact fused `refined_depth` output on the same high grid as `model_input` |
| `shadow_coordinate.f32` | Required exact unbounded canonical-coordinate diagnostic; never renderer authority |
| `shadow_candidate_parallax.f32` | Pre-limiter V2 candidate |
| `shadow_vertical_majorant.f32` | Vertical upper-envelope diagnostic |
| `shadow_vertical_conditioned.f32` | Fixed vertical-share result |
| `shadow_base_final_parallax.f32` | Required in active UIP4 packages; ordinary post-limiter field before subtitle conditioning |
| `shadow_final_parallax.f32` | Protected conditioned model-domain diagnostic; no live renderer authority |
| `shadow_state.json`, `shadow_frame_stats.json` | Typed V2 state and current-frame statistics |
| `warp_map.f32` | Legacy analytic selected pre-saturation source-U diagnostic; dimensions, layout, validity, clamp, and derivation semantics live in manifest `warp_map_contract`; required for ROI packages. Shared-warp packages explicitly deny it actual-render authority. |
| `warp_mask.png` | Authenticated finite-source boundary evidence; required with `warp_map.f32` for ROI packages |
| `window_region.json` | Required semantic observation for ROI packages |
| `sbs.png` | Packed stereo preview |
| `subtitle_conditioning.json` | Required current subtitle-authority descriptor; canonical `none` or `subtitle-contours-v1` |
| `subtitle_ocr_record.u32` | Active-only OCR8 record for the atomic target's authenticated subtitle-publication frame: current on infer, older on whole-tuple reuse |
| `subtitle_locator_state.u32` | Active-only compact UIP4 state in generated-contract word order |
| `contour_support.u32`, `contour_history.rgba32f` | Active-only binary current-evidence support and diagnostic fixed-reference RGB/valid-bit/count/legacy count-four-bit history |
| `subtitle_glyph_matte.u32`, `subtitle_glyph_matte.json` | Native literal 0/1 matte and generated recipe/source/identity binding |
| `subtitle_condition_params.u32`, `subtitle_condition_params.json` | Six raw SCP4 words and tuple binding; required independently of optional GPU trace |
| `render_source.bin` | Exact tightly packed rendered native color; manifest format/dimensions preserve FP16 HDR revocation inputs |
| `contour_field_pin.u32`, `contour_distance.f32` | Active-only projected field pins and native five-center rectangular-footprint slope budget D in parallax pixels |
| `gaussian_weight.f32`, `gaussian_blend.f32` | Actual equivalent depth influence M and precise out-of-place minimum-slope-budget UI/Base proposal; names retained for compatibility |
| `gaussian_vertical_conditioned.f32`, `gaussian_control.u32` | Bit-exact subtitle proposal copy under the compatibility name, and one-word conditioning decision |
| `contour_identity.u32` | Active-only exact 32-word SCT5 source/crop/domain/clock identity with reserved words24..31 zero, with independent current-pin and RGB-snapshot validity |
| `contour_source.rgba16f`, `ocr_probability.f32` | Active-only joined completed source display crop and verbatim detector probability snapshot |
| `gpu_trace_ring.u32` | Optional raw diagnostic history of the last 300 accepted-root completions; available only when diagnostics was enabled before reproduction |
| `gpu_trace.json` | Optional chronological decode of the authenticated raw GPU trace |
| `gpu_trace_contract.json` | Optional exact trace offsets, enums, receipt ABI, and shader provenance |

Scalar `.f32` files are little-endian float32; `.u32`, `.rgba32f`, and `.rgba16f` contain raw
little-endian uint32, float32 RGBA, and float16 RGBA respectively. Schema 51 accepts the canonical
inactive descriptor or the one current `subtitle-contours-v1` package. It binds the exact current generated Depth Coordinate V2
identity, producer and renderer closures, OCR model provenance, entrypoints, the four existing artifact
roles (OCR record, locator state, Base field, atomic final field), all eleven diagnostic contour roles, and both analytic-render tuple roles with
typed dimensions, exact byte counts and SHA-256 hashes. The generated contract is the sole
owner of those live identities and numeric policy values. The
[Host SBS native subtitle contours and UI plane](../../docs/host-sbs.md#native-subtitle-contours-and-ui-plane) is the
sole owner of their runtime and state-machine semantics; this dump-format document copies neither.
No retired layout is preserved or reinterpreted.

The current typed `shadow_frame_stats.json` schema `3` binds twelve words: the existing eight
statistic words and a four-word P05/P95/validity/bin-width tail. The generated geometry contract
owns their exact names, order and meaning. The tail is the GPU histogram result, not a percentile
recomputed by the dump writer; unavailable tail authority is canonical zero. This frame-statistics
schema is separate from the shared `warp_map_contract` schema described below.

Schema 51 authenticates one capture grid across `model_input`, `raw_depth`, and every V2 field. A
production package uses one exact supported convex-2x high shape; halving both dimensions must
recover an exact calibrated embedded DAV2 shape. The capture-time model provenance continues to authenticate that embedded DAV2
identity and derived coarse calibration, while the public numeric artifacts remain high-resolution.
A split coarse/high package, an arbitrary enlarged grid, or a high package without this exact
half-shape relation is rejected rather than reinterpreted.

## Selected map and independent Final diagnostic

The required `selected_map` object binds schema 2's ordinary authenticated Base inverse.
It describes `warp_map.f32`, which is an independent Base diagnostic when the actual draw uses
the shared subtitle field. Its required boolean `render_enabled` records the observed shared
subtitle permission independently of retained OCR/SCP authority; it does not add a selector to
ordinary rendering. `warp_map_contract` schema 3 retains the normalized pre-clamp U encoding,
and `warp_mask.png` classifies the boundary from that same ordinary Base U. Registration metrics
retain their existing geometry and image-quality checks.

The `final_parallax` object binds diagnostic schema 4. Its `warp_input_artifact` is null and its
relation explicitly denies render authority. The verifier still replays the ordinary limiter and
all existing native support/projection/distance/M/proposal/shared-horizontal-limit checks into
`shadow_final_parallax.f32`. Active packages additionally authenticate native dense matte against
the generated same-observation recipe, and SCP4 against current UIP4/OCR8 even if GPU trace is
absent. Matte and condition sidecars reject unknown recipes or tuple bindings.

Infer atomically publishes Base/state/matte/SCT5/SCP4/joined-source plus diagnostics; reuse retains
the actual older owner. Source and probability snapshots belong to that infer owner and do not
become current through a newer presentation. Exact native current source is preserved separately
from `source.png`; only same-texel display conversion and FP16 roundtrip can revoke old matte during
a draw. Preview equality cannot authenticate that decision. New empty OCR publishes zero matte
and inactive SCP4 immediately. No extra observation wait is added.

### Actual shared UI warp observation

Schema 51 requires version 2 of `renderer.actual_draw_renderer`. Its `ordinary-base-v1`
kind names the authenticated ordinary Base diagnostic. A `shared-ui-pin-v1` kind binds the
production shared shader closure and schema-2 `actual_draw_evidence.json`;
`mapping_artifacts_match_selected_renderer` is false because `warp_map.f32` remains the independent
Base diagnostic. Its geometry-chain and ROI exterior checks cannot score or explain the actual
shared render. Historical schema-50 exports and their unversioned analytic attribution remain
unchanged; the standalone shared reader can inspect that format without relabelling it as current.
The current canonical reader rejects older dump schemas.

Schema-2 draw evidence removes the unused `t8` role and `actual_original_legacy_M.u32` artifact.
Raw and legacy masks required by mask construction or independent diagnostics remain available.
Failure to prepare shared subtitles, or invalid selected authority, preserves usable ordinary Base
with subtitle permission disabled. Invalid or unavailable depth retains the independent flat path.

The actual recorder observes PS bindings immediately before Draw. Requested dumps stage these
additional immutable operands on that same immediate context, before the next draw can overwrite
them. The existing completion event and bounded CPU collection apply; normal streaming performs
no diagnostic readback or additional queue wait.

| Artifact | Observed role |
|---|---|
| `actual_shared_pin_weights.f32` | t12, authorized current native binary pin weights |
| `actual_shared_pin_field.f32` | t13, actual native band after the shared pin scan |
| `actual_shared_pin_rows.u32` | t14, one row flag per native band row, then one selected-receipt header word |
| `actual_shared_pin_geometry.u32` | b6, four uint32 words: full-source width/height, native band top/height |
| `actual_shared_pin_map.rgba32f` | Dump-only invocation of the actual shared inverse, packed-output RGBA: pre-clamp source U, source V, contributing pin row, content-valid bit |
| `actual_shared_pin_boundary.rgba32f` | Same inverse's finite-source boundary, RGBA: boundary bit, zero, zero, one |

Fields and weights are `R32_FLOAT` at full-source width by native band height. Rows are
`R32_UINT`, width one by band-height-plus-one. Map and boundary are `RGBA32F` at the exact packed
output dimensions. Invalid content map pixels carry `(-1,-1,0,0)`; boundary pixels always have
alpha one. The row header records selected-receipt validity for that immutable draw; it does not
replace state/SCT/SCP/source/colour authorization. Every diagnostic has its own availability,
byte count, layout and SHA-256. Optional staging failure is recorded as unavailable and never
authorizes substitution of the legacy map.

`shared_ui_pin_dump.verify` joins the manifest, observed draw identity/clock/closure, all readback
hashes, shapes, binary weights and row flags, and actual map/boundary channels. Its
`load_actual_map` returns the authenticated shared source-U plane for numeric inspection and
rejects incomplete readbacks. It does not implement a CPU pin-scan replica or claim missing
background truth. `replay_depth_mapping_v2.py` remains a fresh raw-depth experiment, not an exact
replay of this captured compositor; its capture summary retains the shared-renderer attribution.

## Diagnostic GPU completion trace

Schema 51 may carry a diagnostic-only 300-slot GPU completion ring. It records completed accepted
depth roots, not source frames, presentation frames, busy drops, or every captured desktop update.
Each 136-word (544-byte) record binds an exact trace ordinal, matched frame, analysis generation,
analysis-domain tag, transaction token, analysis-source and live-field extents, the immutable
64-word postprocessed transaction snapshot, all 40 UIP4 words, and all six condition-parameter
words. The authenticated RQST/CBRG token, cookies, work disposition, optional OOCR marker, and
submission class determine `infer`, `reuse`, or `invalid`; a force-class reuse receipt is invalid,
never inferred as reuse. Legal work is `0` for native subtitle suppression, `1` for current-ready OCR,
or `2` for current abstention. `OOCR` is valid only for work `1` on infer and is absent for work `0`/`2`.
Retired independent due values `8`/`16` are invalid. Subtitle disposition is cross-checked against
expected work, host outcome, raw flags and the authenticated device receipt. Every ordinary reuse is
`held_with_depth`: OCR8, 40-word UIP4, condition6, the complete contour tuple and the atomic conditioned target remain the prior coherent
tuple. Authenticated reuse cannot claim a direct current subtitle publication.

The shared live/offline policy lets the near-identical detector decide joint reuse without a time
or hold-count expiry. Every changed candidate names the true infer owner, never the newer completed
color publication. Known completed lineage requires an exact current completion receipt and valid
route/time ordering. Exact unchanged DDup deliveries submit no root; if B reused A and remains
static, its authenticated tuple may retain A indefinitely without a final refresh. Offline
changed-source replay does not cover that idle behavior. Replay request policy schema `6`
authenticates this policy; the dump and trace record layouts are unchanged. Long reuse chains must
still preserve the whole depth/subtitle/final tuple exactly.

The ring writer invalidates the header tag before overwriting any slot, commits payload before the
record tag, then updates the cursor and republishes the header tag last. The reader reconstructs
wrap from `next_sequence`, `next_slot`, and `committed_count`, rejects torn/non-consecutive records,
and requires a record matching the dump frame, generation, full analysis-domain hash, analysis
extent, field extent, and domain-reset bit. `dump_forced_at_enqueue` is perturbation provenance:
it is false when a dump request harvests a root that was already pending, and is therefore not a
requirement for the matched record.

For non-suppressed subtitle publications, UIP4 and SCP4 are captured after the ordered
resolver and conditioner. A held reuse records the identical prior tuple, including the
original publication frame and source clock. When the predecessor is still in the ring, both
tuples must be byte-identical. The raw ring is schema5, its contract schema6, and the decoded
JSON schema7. The 40 UIP4 words contain current source/domain identity and the independent
16-word adaptive controller; they contain no historical OCR rectangles. Historical trace rows
authenticate controller structure and condition parameters, not missing OCR geometry or camera
validity. The matched package separately joins the actual OCR8 paired covers, geometry, contour
snapshots, original source clock and cut epoch.

Dispatched SCP4 always has its schema/tag. Its payload is either the exact current count/kind
mask, full-strength value2 and bit-identical applied UI plane, or positive-zero words2..5 when
conditioning is unavailable. Invalid Base probes retain the current UIP4 cover union while
disarming conditioning; unavailable geometry clears granted UIP4 slots. The controller's
accepted clock is distinct from the current source clock. Suppression freezes the subtitle
tuple, skips conditioning and publishes immutable Base; trace subtitle sections are marked
unused. The raw ring, decoded JSON and trace contract are SHA-256-bound, including the shader
closure, offsets, enum values, receipt cookies and embedded state schema.

Trace setup, staging, validation, or publication failure makes all three artifacts canonically
unavailable and never rejects or weakens the core Dump 3D package. With diagnostics disabled the
trace shader/resources are not created and no append, copy, or dispatch runs; enabling Dump 3D
afterward cannot recover prior history. With diagnostics enabled, the trace still adds no per-frame
CPU map, readback, query, or synchronization and has no rendering authority.

## OCR8 and UIP4 records

Coordinate schema 121/OCR 33/contour recipe 27 publishes only classified current subtitle pairs.
Its schema-4 OCR8 buffer is 256 words:
the original 208-word header/core/cover prefix plus a 48-word classifier history tail. Both producer
and resolver descriptors bind the complete generated classifier policy. The tail's SCL1 identity
binds its current source clock/frame/generation and source/crop/field/content/reference geometry,
independent-entry counter, blank clock, trusted height/span, and 16 bounded position votes. A zero
tail means no retained prior; abstention requires zero. Current empty OCR can retain a valid prior
but publishes no covers and no protection. Prior bytes are never mask or geometry authority.
Validators check serialized history identity and bounds; a single dump cannot replay the historical
classification decisions that produced its vote counters. The authenticated producer owns those
decisions. The local-line scratch buffer is temporary and is not a dump artifact.

The generated `subtitle_ocr.ocr_record` contract owns OCR8's schema, tag, word counts, offsets,
capacities, flags, and numeric detector/box policy values. The dump serializes exactly that fixed
word array as little-endian uint32 values. Its frame, analysis generation, analysis-source extent,
tensor-content geometry, projected ROI, paired core/cover records, topology metadata, canonical
unused slots and valid classifier history must all agree with the atomic target's publication frame. That is the matched frame for
infer, and the trace-authenticated older tuple frame for whole-tuple reuse. See the
[live OCR8 contract](../../docs/host-sbs.md#native-subtitle-contours-and-ui-plane) for producer semantics.

The current OCR policy binds lower-quarter, approximately centered line
eligibility with native-source horizontal adjacency before the existing eight-pair capacity selection. A line's tight union is only an
eligibility witness; the record retains independent original core/cover pairs. The current reader
authenticates this producer policy through the contract and source closure. It does not claim to
independently reconstruct preselection geometry from retained pairs, because capacity selection
can omit members of the original line.

Ordinary paired covers bind the canonical four-producer-cell horizontal line-end guard and a
vertical guard of half the tight detector height, added to DB expansion before coordinate mapping.
The UI-plane probe, diagnostic RGB history and recipe 27 native glyph search use the whole
producer-published cover. There is no second core-relative vertical clipping step. An expanded
search cover grants no blanket pixel protection. These policies are authenticated with the
coordinate contract and producer source closure, as defined by the live owner document.

Both box counts describe the retained paired candidates, not the number discovered. The generated
`subtitle_ocr.candidate_selection` policy retains at most eight pairs by tight-core area, with
earlier raster discovery winning ties, and publishes them in discovery order. Capacity pressure
does not change the fixed record layout or introduce an overflow flag or dropped-count field.

The generated `subtitle_ocr.plane_state` contract owns the UIP4 schema/tag, 40-word layout
and eight current slots. Every admitted valid OCR8 paired cover participates directly; there
is no secondary component winner, ownership tracker, corner filter, pending state or box-count
history reset. Covers remain in the exact current OCR8 record. The fixed-RGB history counts
four genuine eligible observations for diagnostics; current palette/stroke or local dark-outline
evidence admits support immediately. Positive count preserves duplicate admission without a
four-count wait. This history is not a new stable temporal subtitle classifier. SCT5 preserves source/domain history without
encoding OCR group identity; its reserved words24..31 are canonical zero.

The independent UI controller probes the current cover union using its existing conflict
histogram. Under the authenticated `current-authorized-clearance-floor-immediate-forward-current-release-exponential-source-time-empty-reset-v2`
policy, that current histogram's entry-budget candidate is an immediate forward floor. Backward
adaptation follows the current release-budget candidate with a 100 ms source-time exponential
time constant and at most 250 ms integrated per observation, without a dwell or historical maximum.
The applied plane is only in its 16-word tail. A genuine newer valid-empty observation clears
invisible plane/requirement/probe state while retaining its accepted clock; invalid or repeated
clocks cannot authorize that reset. The floor is budgeted cover evidence, not maximum-scene or
all-glyph clearance. The adaptive known flag mask is `12`; retired flags `1`/`2` and relative
words 6..11 are canonical zero. Relative word 2 and decoded `current_entry_requirement_level`
describe the current entry requirement. Compatibility delay/dwell/rate metadata is zero in both
directions, with `retreat_time_constant_ms=100`. Invalid source clocks or interrupted authority
cannot spend missing time; duplicates and reuse hold the controller. Empty or unconditioned output remains exact
Base, including original padding bits. Contour recipe 27 uses current glyph evidence within
center-guided bounded ordinary search, retaining the full published ribbon search. Its horizontal
minimum slope-budget interval and rectangular field-footprint completion are bound by the
generated contour policy. Compatibility Gaussian names
remain; M records equivalent depth influence and is not a color alpha. The legacy sigma divisor
does not select a falloff in this recipe.
The independent diagnostic minimum slope-budget interval projects immutable Base toward that plane only where required,
with an explicit closed Base/UI endpoint clamp for FP32 roundoff, followed by the shared horizontal
guard. The compatibility vertical field is an exact proposal copy. The live native matte uses current-evidence-supported pixels without a per-pixel
fade or rectangular fallback. UIP4 word22 is canonical zero. Missing,
collapsed or stale geometry preserves the controller plane/goal but clears granted current
slots, timing authority and conditioning. A valid OCR8 record may still contain covers;
those are not usable protection authority without the current geometry join.

The reader validates exact source/frame/generation/time/cut/ROI/content identity, current slot
mask, reserved words, complete controller and SCP4 parameters. Current covers come from the
actual captured OCR8 record. Historical trace rows have no OCR8 payload and therefore cannot
prove exact historical cover geometry. See the
[live subtitle contract](../../docs/host-sbs.md#native-subtitle-contours-and-ui-plane).

The Python conditioner replay below is deliberately an independent dump-integrity verifier. It
recomputes the frozen SM5 operation order, including the bounded one-ULP division alternatives, to
reject corrupt or internally inconsistent packages. It is not a second production policy and is
not the shader regression gate; live behavior remains qualified by the generated contract and the
native WARP/GPU tests.

## Analysis domains and window provenance

A full-source package uses analysis generation zero. A window-region package binds a nonzero
analysis generation, the exact uncropped source rectangle, tensor extent, centered integer
contain-fit `tensor_content_rect_px`, edge-replicated excluded padding fraction, unit conversion,
and outside-only video-inverse collar in `depth_input_region.json` schema `4`. The source rectangle is never
stretched or trimmed: a wider region pads above/below and a taller region pads left/right. Padding
does not participate in model statistics, scene-cut evidence, OCR, or subtitle state;
ordinary published fields extend the nearest content boundary through it. The renderer clamps
sampling to centers of the real-content rectangle, so synthetic padding cannot change edge-video
geometry. Crop-local depth must never be interpreted as a full-source field. Ordinary required
full-source inverse maps prove that samples beyond the conservative collar return to identity.

Parallax fields are signed one-eye U offsets in the analysis source's coordinate system. A window
ROI field value becomes source pixels by multiplying by the inference/source-ROI width, not the
padded tensor width or full captured width. Its embedding into full-source U uses the authenticated
`full_source_parallax_scale`. In contrast, `warp_map.f32` and the native ownership diagnostic's U
channels are normalized full-captured-source coordinates; multiply their signed difference from
destination identity by the full captured width to measure one-eye source pixels. A binocular
separation is twice a symmetric one-eye offset; these units must not be mixed.

The sole adaptive policy (diagnostic identity3) uses the authenticated linear Candidate
`clamp(requested_gain*(raw-zero)/D, -0.04, +0.04)`, with no depth curve or soft envelope.
Candidate, post-limiter Base, and conditioned Final remain separate actual numeric artifacts.
The signed hard endpoint is in analysis-source U; it does not bypass the native slope/shear guards.
Older with-curve experiment artifacts retain their original geometry identity and mapping and
cannot be relabeled as the linear candidate.

All current ROI packages use `warp_map_contract` schema `3`.
The map is the ordinary authenticated Base inverse embedded into the full capture.
Base retains its slope-limited outside-only collar. `warp_mask.png` red marks finite-source
boundary extrapolation; green and blue are zero. There are no source-kind or visibility flags,
static-rim samples, or synthetic interior fill paths. Nearby browser pixels may move within the
collar; only samples beyond its bounded support must be at zero displacement.

Both eyes sample one coordinate from the original full matched source, preserving source V. Positive
foreground can project outside the video rectangle without being flattened at the ROI boundary.
Continuous sampling can stretch browser background into a retracted edge; it cannot recover source
video pixels or a foreground outline cut off by the original image. Historical layered schema 3/4
packages retain their original identities and must be verified using their frozen original source;
they cannot be relabeled as the current selected-map renderer.

The deterministic ROI contain fit is planned on the embedded calibrated DAV2 grid. For a fused
package, each half-open content-rectangle coordinate is then multiplied by exactly two; padding
fraction is inherited from the coarse plan. Independently rounding a second fit on the high grid is
not an accepted equivalent.

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
3. verifies every required content hash and numeric extent, including one common calibrated coarse
   grid or exact supported convex-2x high grid and its derived embedded-DAV2 half shape;
4. replays the schema-selected limiter from the candidate over the full tensor: serial float32 for
   lines up to 32 elements, otherwise conservative Q30 upper/lower envelopes and horizontal
   majorant with the authenticated 75/25 float32 share, using content width for both steps;
5. authenticates `shadow_final_parallax.f32` as an independent protected-field diagnostic, and binds
   the selected-map schema and exact boolean draw permission;
6. validates ROI placement, authority-specific window identity, inverse-map geometry, and the exterior
   zero-plane evidence; and
7. validates canonical inactive metadata, or the exact authenticated current-subtitle-publication/
   held-reuse OCR8/UIP4 model, shader, record, state, and artifact identities;
8. replays the ordinary V2 chain into `shadow_base_final_parallax.f32` when UIP4 is active, then
   validates exact contour-support bilinear projection and the independent native five-center rectangular-footprint
   horizontal slope-budget oracle with exact rational Y-footprint membership and its FP32 X
   arithmetic envelope, authenticates equivalent M and direct two-sided interval proposal,
   verifies the exact compatibility copy, then replays shared horizontalQ30 into
   `shadow_final_parallax.f32`. Empty/invalid control and whole unchanged rows must copy Base
   exactly. Infinite local content budget publishes zero M and original Base, including padding;
   finite active padding extends its nearest-content values. The compatibility intermediate
   is authenticated independently of original Base diagnostics;
9. replays the same-observation native matte recipe, authenticates separate SCP4 and tuple sidecars,
   and validates exact native render-source layout and bytes without substituting preview equality; and
10. when the optional GPU trace is available, authenticates all three hashes, the exact trace shader
   closure and ABI document, chronological wrap/commit structure, raw branch/OCR proof, matched
   analysis domain, and bit-for-bit decoded JSON. An unavailable trace remains valid optional state.

Use `.f32` artifacts for quantitative work. Schema 51 does not package scalar/heat preview PNGs or
per-field shape sidecars. The sole retained shape sidecar, `model_input_shape.json`, is calibrated
preprocess authority rather than a preview description. On a fused capture it describes the sole
high input; the embedded DAV2 calibration is derived from its exact half shape. Float dimensions live in
`dump_manifest.json`; inverse-map semantics live in its `warp_map_contract`. The package retains
only the color/evidence PNGs `source.png`, `depth_input_source.png`, `sbs.png`, and, when available,
`warp_mask.png`.

A production package carries `composite_runtime_provenance` schema 2 and authenticates the frozen fused
ONNX, embedded DAV2, ZipDepth checkpoint, preprocess closure, engine recipe/artifact, and active
engine manifest as one record. Missing or stale composite evidence rejects a high-grid package.
The reader also recognizes the calibrated one-grid DAV2 diagnostic shape under the same current
schema and provenance checks; it is not a live or conversion runtime fallback. Historical schema-40
packages require their original evaluator snapshot and are rejected by the current reader.

Generate a non-authoritative diagnostic preview outside the atomic package when needed:

```powershell
E:\ApolloDev\modelopt-py312\Scripts\python.exe tools/sbsbench/generate_dump_previews.py `
  E:/ApolloDev/sbs_dump/dump_2026... shadow_final_parallax.f32 `
  --output-dir E:/ApolloDev/sbs_dump_previews/dump_2026...
```

The generator first verifies the complete required geometry/state package and the requested
artifact's manifest SHA-256. A model-input preview additionally authenticates
`model_input_shape.json` before using its normalization parameters. It then derives scalar
grayscale/heat views with finite p2--p98 normalization. Independently stretched previews are
diagnostic and must not be compared as absolute depth values.

## Commands

Validate the Python contract and geometry reader from the repository root:

```powershell
E:\ApolloDev\modelopt-py312\Scripts\python.exe -m unittest tools.sbsbench.test_depth_coordinate_v2_dump_contract
```

Score a capture set with `tools/sbsbench/sbsbench.py`; see [README.md](README.md) for the maintained
evaluation commands and [METRICS.md](METRICS.md) for metric authority.

UIP4 words22 and23 are canonical zero. The former provisional-cover countdown and cue latch grant
no authority. Native support and slope-budget distance describe current cover pixels with valid RGB,
all four finite unit-range bilinear probability contributors, and positive diagnostic count for
duplicate admission. Detector probability above `0.2` and radius-two current native contrast above
`0.08` select palette anchors on each paired core's exact 64-by-8 grid. Q16 color bins retain
their brightest actual anchor with earliest-index ties. They are ordered by representative
brightness, count and bin index; dimmer modes with matching normalized Q12 hue are pruned before the eight-mode
cap. Probability is never a direct per-pixel bypass. A foreground pixel must match a current
mode, have current contrast and pass the bounded color-coherent tangent test. A dark low-chroma
outline instead needs current contrast and a contrasting palette-colored neighbor at one of eight
fixed radius-six offsets; this is not mask dilation. Source and neighbor RGBA must be finite,
RGB unit-range, alpha exactly one and inside the exact native crop; invalid or outside neighbors
are ignored. Neighbors may lie outside the paired cover. The generated contract owns exact
integer matching, brightness pruning and stroke bounds. This overlay-color/stroke hypothesis
does not identify semantic dialogue or guarantee complete glyph alpha. The retained count-four
feature bit is diagnostic. Readers recompute the current support rule and verify its five-center
footprint, minimum slope budget, actual M/proposal and final guards without a whole-cover term.
