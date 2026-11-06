# sbsbench

`sbsbench` measures the real Host SBS V2 estimator and renderer. It scores exact matched source,
depth, warp-map, and SBS artifacts; it does not substitute a hand-maintained CPU renderer for the
production path.

The normative renderer contract lives in [Host SBS pipeline](../../docs/host-sbs.md). This folder
owns evaluation commands and evidence formats, organized as follows:

| Guide | Owns |
|---|---|
| [Dump and replay format](DUMP_FORMAT.md) | Dump 3D files, provenance, schemas, and exact V2 replay |
| [Datasets](DATASETS.md) | Core fixtures, public extended suite, metadata, and reference sidecars |
| [Metrics and gates](METRICS.md) | Metric meanings, baseline policy, validators, and evidence floors |
| [Optional oracles](ORACLES.md) | FLIP, RAFT-Stereo, SEA-RAFT, and iSQoE diagnostics |
| [Host SBS scene cuts](../../docs/host-sbs-scene-cuts.md) | Normative cut evidence and state machine |

The native harness and offline manager/worker share typed diagnostic and whole-clip wire codecs.
Compatibility names such as `scene-cache` and `scene-plan` remain evaluator/harness contracts.
Current production conversion is the causal one-pass
pipeline owned by [Offline Host 3D conversion](../../docs/whole-clip-sbs-pipeline.md). Fixed-schema
JSON is published atomically, duplicate keys and unknown fields fail closed on read, and raster
dimensions are checked before byte-count arithmetic or GPU allocation. The Python evaluators
consume those current documents without a second native-contract compatibility translator.
Evaluator file-follow and raster artifacts remain available for reproducible scoring. Production
conversion's raw frame transport and paged audits are owned by the linked conversion document.

## Required evaluation loop

Run commands from the repository root. The previous clean V2 core references predate the current
geometry contract and remain in Git history rather than serving as stale gate inputs. Start this
migration with an authenticated comparison run:

```powershell
$SbsbenchPython = "C:\absolute\path\to\the\validated\python.exe"
& $SbsbenchPython tools/sbsbench/run_eval.py --comparison-only --label control
```

Select that interpreter once and use the exact same executable for control, treatment, scoring,
report generation, comparisons, and related tests. Record its fingerprint before the control and
verify the exact same output before the treatment and report:

```powershell
& $SbsbenchPython -c "import sys, numpy, PIL, onnx; print(sys.executable); print(sys.version); print('numpy', numpy.__version__); print('Pillow', PIL.__version__); print('onnx', onnx.__version__)"
```

Never switch between bare `python`, `py`, a bundled tool runtime, or another virtual environment
inside one evidence chain. A numeric-runtime mismatch invalidates the pair: rerun both sides with
one interpreter. Do not edit provenance or use an allow flag to waive the mismatch.
`--report-allow-executable-diff` applies only to the Sunshine executable; it does not authorize a
Python or numeric-runtime difference.

Useful variants are:

```powershell
# Fresh matched A/B without the committed-baseline verdict
& $SbsbenchPython tools/sbsbench/run_eval.py --comparison-only --label control
& $SbsbenchPython tools/sbsbench/run_eval.py --comparison-only --label treatment `
  --report-control cmake-build-relwithdebinfo/sbs_eval/control `
  --report-allow-executable-diff

# One supported geometry or execution treatment
& $SbsbenchPython tools/sbsbench/run_eval.py --comparison-only --label pop-1p0 `
  --extra --pop-strength 1.0

# Prepared public suite
& $SbsbenchPython tools/sbsbench/run_eval.py --suite extended --comparison-only --label public-control

# Publish an inspected V2 baseline from accepted production defaults only
& $SbsbenchPython tools/sbsbench/run_eval.py --update-baselines
```

`--update-baselines` rejects `--extra`; first move an accepted setting into production defaults.
The process exit code is the verdict: `0` pass, `1` regression, and `2` invalid or incomplete
evidence.

Every estimator clip receives an authenticated source-observation timeline. An explicit
`--extra --observation-timeline <timeline>` overrides the authored prepared-source cadence;
its frame count must match every selected clip. Otherwise `meta.json`'s `fps` declares the
cadence, or the evaluator explicitly authors 30 Hz when it is absent. Neither path uses wall time
or inference speed. These authored stimuli do not recover original video timing or qualify
long-form temporal behavior. Native raw/field/cache replay uses its own existing timing contract.
Host SBS has one adaptive zero/scale and subtitle UI-plane pipeline; the former experiment/mode
selectors are rejected before launch.

## Actual selected compositor qualification

Ordinary `run_eval.py` metric images retain the legacy native matte compositor. They do not
qualify the live completed current-outline shared warp.
Use the separate bounded qualification command to exercise those actual production programs,
selected receipt, native pin-band owner and shared draw recorder:

```powershell
& $SbsbenchPython tools/sbsbench/qualify_current_outline.py `
  --frames tools/sbsbench/clips/subtitle_cjk_highres_transitions `
  --out cmake-build-relwithdebinfo/sbs_eval/current-outline-qualification

# Independently recheck the saved evidence against the current sources and native binary.
& $SbsbenchPython tools/sbsbench/qualify_current_outline.py --validate-only `
  --frames tools/sbsbench/clips/subtitle_cjk_highres_transitions `
  --out cmake-build-relwithdebinfo/sbs_eval/current-outline-qualification
```

Build first and retain the build/test DLL `PATH` documented in [AGENTS.md](../../AGENTS.md).
The launcher directly runs `sunshine.exe --sbs-bench --current-outline-qualification`; it does
not rebuild. `--build-dir` selects another existing native build. Inputs must be 1..64 immutable
SDR PNGs named `frame_<id>.png`, and output must be absent or empty. `--observation-timeline`
binds explicit source clocks; otherwise `--fps NUM/DEN` authors a 30/1 cadence by default.

Selected stereo images live separately in `current-outline/sbs_<id>.png`. The atomic
`current_outline_qualification.json` binds each exact source snapshot, observation clock,
publication receipt, selected/SCT/SCP/ParallaxState words, draw b3, native pin-band shape,
per-draw serial, literal GPU row/header flags, output digest, executable and current source-closure
pins. A pre-Draw observer verifies the actual shader, consumed t0..t7/t12..t14 and b2/b3/b6
bindings, including the absence of background RGB/proof bindings. The unused t8 alias no longer
participates in rendering or binding acceptance. The Python reader independently validates those
joins and rejects stale owners, corrupt hashes, unknown fields and incomplete frame coverage.
It reports selected-current, shared-warp and active-pin frame counts, including explicit unavailable
fallbacks. This file-fed full-source check establishes renderer wiring and provenance; it does
not establish live capture eligibility, window-region behavior, visual quality or live timing.
Ordinary metric PNGs, schemas and scores remain separate.

Qualification schema8 records `authenticated-ordinary-base-renderer-v1` for the ordinary metric
artifacts, which now use only authenticated Base without an analytic subtitle selector. Selected
shared output remains separate. Schema8 retains all thirteen composition closures and requires
foreground-edge and trusted-inner shader availability at the root and every frame. Availability is
not evidence of GPU execution. The shared renderer has no retired completed-pixel-shader field. Schemas1–7 remain
read-only formats and never qualify the current rendering policy. Each original ordinary metric
label is preserved; source, coordinate-contract and executable checks still apply, so older
captures require their authenticated frozen contract/source reader. No captured evidence is retagged.

Schema2 added a binding for
`current-outline/selected_matte_<id>.u32`, the exact post-draw selected t3 input in little-endian
binary R32_UINT form, with its native source rectangle and dimensions. The independent reader
still accepts schema1 captures; they provide no literal-mask score evidence. The launcher writes
`current_outline_mask_scores.json` automatically. To independently regenerate that diagnostic:

```powershell
& $SbsbenchPython tools/sbsbench/current_outline_mask_metrics.py `
  --frames tools/sbsbench/clips/subtitle_cjk_highres_transitions `
  --out cmake-build-relwithdebinfo/sbs_eval/current-outline-qualification
```

When optional `gt_subtitle_overlay_mask` exists, its exact frame set must match the source PNGs.
Each truth image must be a source-sized 8-bit L PNG containing only 0/255. Declared tight-mask or
sanitizer requirements in `meta.json` are enforced; required missing truth is rejected. The report
binds current `meta.json` and tight-mask hashes, source snapshots, native qualification manifest,
executable, authored observation timeline, reader/scorer implementations and numeric runtime.
The literal crop occupies its source rectangle directly; source pixels outside that crop have
zero selected protection. Full-source truth is never cropped to improve recall.

Diagnostic precision, recall and IoU come from source-pixel confusion counts; aggregate ratios
use pooled counts. Empty truth frames add their false-protected pixel count. Consecutive
matched-size frames also measure mask toggles
only where all three source RGB codes and the truth bit are unchanged; overall, glyph and
background denominators are reported separately. These are fractions of eligible pixels, not
rates per second. Missing masks or truth yield null scores, and every zero denominator yields
null rather than an invented perfect result. These diagnostics have no release quality gate and
do not establish final stereo RGB retention, recovered alpha, capture eligibility or live timing.

The current native manifest is schema4. Each observation retains schema3's three small producer
snapshots under `current-outline/producer-stages/frame_<id>/`: `cbc_meta.u32` (512 words),
`raw_meta.u32` (512 words), and `f2_data.u32` (68,608 words). These are exact existing GPU buffer
contents, about 272 KiB per observation, copied after the completed selected draw. The observer
must return that same selected-M/receipt pair and completed frame/clock, with `pending_queued=false`.
Native descriptor checks require full same-device buffers: structured CBC/Raw views have stride
four and unknown format; F2 is a stride-zero RAW BUFFEREX view with R32_TYPELESS format. Subranges,
oversized underlying buffers and role/type mismatches are rejected before the diagnostic copies.

Schema3 records symbolic buffer roles, source owner, whole-view shapes, file hashes and the exact
current composition source-closure pins. The reader verifies complete artifact coverage and the
positive kind3 selector's existing current headers, embedded C24, source envelope and transaction
token joins. It does not invent a physical-black model-bank proof or require a KNOWN model.
Fallback bytes remain diagnostic and do not become current producer authority. These mutable
aliases establish no rendering authority and do not alter mask scores. Schema1/2 reading remains
supported, with producer-stage evidence unavailable there. Schema1--3 preserve their historical
background proof/lease requirements. Schema4 replaces those fields and the background source pin
with the shared warp's current source closure and an explicit `shared-ui-pin-band` draw record.
Its native row flags may grant pin support only with current selected receipt, SCT and SCP authority;
they do not prove semantic glyph completeness or a clean hidden background.

## Subtitle diagnostics

For bounded subtitle research, ordinary evaluation can use
`--subtitle-plane-diagnostics --subtitle-plane-diagnostic-frames 61-78,111-121,143-146`
with an explicit observation timeline. The ordered, disjoint ranges select one-based input
observations independently of `--output-every`; every input still advances the real estimator.
The selected diagnostic manifest records processed observations, captured snapshots and rendered
outputs separately. The current manifest is schema2 and per-frame metadata schema3. These synchronous
internal snapshots are research evidence, not live timing
or a complete per-frame render capture. Without the range selector, subtitle diagnostics retain
their full-cadence output contract.

Add `--subtitle-plane-diagnostic-render-selected` to save actual stereo output for every
selected observation as well as the regular output cadence. This option requires the range
selector and records `diagnostic_render_selected` in the manifest. It increases stored outputs;
it does not change estimator processing or live rendering. Keep the selection bounded.

To diagnose an unavailable OCR record, add `--ocr-working-diagnostics`. This requires subtitle
diagnostics and an explicit observation timeline, and writes a separate `ocr-working` directory.
The probability, cell, native source-band, transaction and resolve-constant resources are mutable
working data copied synchronously after the exact joined completion. They are research inputs;
the raw transaction must prove current optional OCR execution before interpreting P/cells/RGB.
They do not replace the completed subtitle snapshots or authorize rendering.

`--subtitle-cohort-research` additionally runs the private cohort experiment on the estimator's
actual device inputs. It requires both diagnostic options above and the explicit timeline.
The separate `subtitle-cohort-research` sidecars contain private state, ownership, filtered
support, unchanged dense completion, and their GPU inputs. Production rendering continues to use
its ordinary subtitle tuple. `guarded_glyph_matte.u32` additionally records the private final
subset guard; `glyph_matte.u32` retains the unchanged dense result, with separate guard shader
provenance. The `published_next_*` files are the coherent previous tuple
published **for the next observation**, not copies of the old inputs consumed by the current
one. Command frame/time fields alone do not prove fresh GPU work: inspect the private state,
SCT, transaction and indirect arguments to distinguish updates from reuse. This option is
default off and its synchronous exports do not establish live speed or presentation quality.

`--subtitle-inner-colour-research` runs the complete private inner-fill experiment on the same
estimator device, including palette discovery, current colour/black-stroke witnesses, closed
four-connected components and a fixed three-native-pixel margin on each axis. It consumes the
production **baseline support**, not the cohort-filtered support. It requires the real estimator,
both diagnostic options and the explicit timeline; cache/direct-geometry modes are rejected.
`subtitle-inner-colour-research` contains `inner_body.u32`, `inner_matte.u32`, exact current inputs,
CB/transaction/dispatch bytes and separate colour/component source closures. Private schema 2 also
exports `numeric_validation_receipt.u32`, the helper source pin and command recording/replay counts.
It uses one parallel numeric validation per call and reuses its recorded fixed commands. All palette
decisions, mode dispatch and mask union run on the GPU. Reuse holds the accumulated masks through
conditional zero dispatches; command clocks alone are not evidence of a fresh observation. This
experiment is default off and does not change production drawing. Its synchronous diagnostic
capture is not a live latency or performance benchmark.

`--joined-completion-probe <plan.json>` selects a separate, default-off three-source experiment
for current-color presentation while a real joined CUDA transaction is pending. It requires three
immutable SDR PNGs, their pinned plan and an explicit observation timeline. It rejects ordinary
diagnostic options, replay, caches, follow mode and HDR. A genuine empty warmup completes before
the authored capture epoch; the plan keeps conversion IDs separate from estimator frame IDs.
The private gate follows both interop unmaps and precedes the real completion event. Both busy
draws require actual `CUDA_ERROR_NOT_READY` queries before and after their GPU readbacks.

The probe records matched A, busy B, busy empty C, late B with current empty color, and matched C.
Staging and owned CPU snapshot buffers are allocated before the epoch. GPU copies and Maps finish
before the after-query; file output and hashing use those immutable CPU bytes after release.
Inspect the invocation, completion, phase and raw artifact bindings together. Missing completion,
an expired presentation owner, or a ready event before a required busy boundary is a failed or
unavailable experiment. Preserved partial files do not establish a complete trajectory.
This is bounded approximate presentation with an unknown current cut verdict. It creates no
depth-reuse authorization or new OCR observation. Old-mask withdrawal cannot detect new glyphs,
and unchanged background in an imprecise old mask can retain protection after text disappears.
The live converter has not adopted this route; these captures do not qualify production latency.

An explicit `current_source_snapshot` section in the private probe plan adds the V5 transport-only
experiment. It preallocates and exports `current-display-source.rgba16f`, a source-only receipt and
the literal current-source constant-buffer readback for all five phases. The source uses its own
capture epoch, conversion serial and authored steady-clock timestamp; retained Base/SCT clocks stay
unchanged. All new Maps also finish before the busy after-query, with hashes and files deferred
until release. Missing or disabled sections keep the V4 GPU/render path. These pixels are not bound
to the renderer and do not authorize current glyphs or authenticated subtitle absence.

### Adaptive infer/reuse A/B without a headset

Use the maintained adaptive replay when a bug may depend on DAV2's private device decision. One
command decodes every video once to lossless PNGs, feeds that exact corpus first to a force-infer
control and then to the shared production conditional transaction policy, authenticates the full
GPU trace, and writes JSON plus HTML comparison reports:

```powershell
& $SbsbenchPython tools/sbsbench/run_adaptive_replay.py `
  "C:\captures\first.mp4" "C:\captures\second.mp4"
```

Prepared `frame_*.png` directories are also accepted. The numerically ordered selected subset is
staged once under canonical one-based IDs, then hashed before and after both runs. The runner
first requires and verifies the current `sunshine` target, then binds the executable, config,
runtime shader tree, production composite
engine/ONNX, OCR engine/contract ONNX, and generated coordinate contract across both serial legs.
The default gate requires at least one actual reuse and proves that every reuse retained the
preceding fused `refined_depth`, UIP4 source/controller state, condition parameters and
`final_parallax_<frame-id>.f32` bit-exactly. Native geometry replay authenticates the paired source
snapshot hash but receives no source texture, color mode or HDR scale. Every reuse holds the whole
depth/OCR/UIP4/final tuple; only valid infer publishes current OCR or abstention. The runner derives
GPU history-owner age from frame identity and the exact source-observation timeline, not processing
speed or a host baseline delta. Reuse preserves the actual infer owner. Every infer raw field must
match the same-frame force control.

Request policy schema `6` lets the near-identical detector decide depth and subtitle reuse together.
Elapsed source time and hold count never force inference; each candidate still compares against the
actual last infer owner. Legal work values are `0`/`1`/`2`; old policy `5` and independent due values
`8`/`16` are rejected. Exact unchanged DDup reuse submits no root, even after the current color reused
older analysis, but the offline replay supplies no DDup proof or live idle-tail coverage.
The default gate still requires actual joint reuse. `--allow-zero-reuse` remains an explicit option
for a comparison whose footage does not qualify; there is no frame-rate-based exemption. Missing,
extra, wrong-sized or misidentified artifacts fail closed. Long holds retain the same byte-exact
raw depth, subtitle state and final field checks. These invariants use no image-tuned thresholds or
headset judgment.

The report measures final-field step and jerk for both force control and adaptive treatment. Every
serial harness stage from `sbs_perf.json` is also aligned by name/sample count and reported with mean and
p95 control-to-treatment deltas; nested GPU stages are kept separate and are never summed.
Optional `--max-scene-residual-delta-p95` and `--max-subtitle-residual-delta-p95` bounds can turn
the report's downsampled image diagnostics into clip-specific CI gates.

This replay deliberately supplies no Desktop Duplication damage history, live window authority,
retained-source idle delivery, or encode-deadline scheduling. Those remain live admission inputs.
Request formation, bounded nonblocking completion receipts, known actual-owner lineage, estimator, conditioner, renderer and
GPU trace are the same
production code, so the offline path does not maintain a second infer/reuse state machine.
The runner uses private replay harness schemas 31 (force oracle) and 30 (conditional treatment),
plus metadata schema 3, leaving formal `run_eval.py` schema 22 and its baseline evidence untouched.
Formal old-code/new-code reports use `--report-allow-executable-diff` (or the report launcher's
`--allow-executable-diff`) for the binary and HLSL difference. Each run still authenticates its source,
model/preprocess identity, scored artifacts and numeric remeasurement under the same metric, label
and numeric-runtime contracts. This does not authorize reading old Dump 3D or adaptive-replay
schemas with current validators; retain their original evaluator snapshot for independent replay.

Host SBS V2 has no model/profile selector. Production and the maintained benchmark harness use the
single authenticated DAV2 Small + frozen ZipDepth convex 2x composite specified by
[the Host SBS production contract](../../docs/host-sbs.md#authenticated-production-contract).
The supported treatments are intentionally narrow and are listed by `run_eval.py --help`;
unrecognized historical options fail argument parsing.

### Current subtitle authority

The current dump reader accepts the authenticated OCR8 record, 40-word UIP4 source/controller state and
complete joined native contour tuple as live subtitle authority. Retired SLR3--SLR9, GST/OGR/ORS, and offline overlay-detector
paths are not accepted as live or replay authority.
The current coordinate contract is schema 129/OCR 38/contour recipe 27. UIP4 remains 40 words and
OCR8 remains 256 words. The UI plane keeps its immediate current entry floor, uses a 100 ms
source-time exponential response toward the current release candidate, and clears invisible
controller history on a genuinely newer valid-empty observation. Duplicate clocks and whole-tuple
reuse hold state. These rules do not by themselves qualify live timing, glyph recall or comfort.

## Reports and comparisons

When a treatment was not run with `--report-control`, generate a report separately:

```powershell
& $SbsbenchPython tools/sbsbench/generate_report.py <control-run> <treatment-run> <report.html>
& $SbsbenchPython tools/sbsbench/compare_runs.py <control-run> <treatment-run>
```

Do not average two `results.json` files manually. `compare_runs.py` first authenticates matching
clip and evaluator provenance, intersects applicable finite evidence, and reports results by
declared `content_type`. Synthetic probes and unclassified clips never masquerade as a decisive
general-content mean.

`run_eval.py` runs TensorRT/D3D11 harnesses serially so concurrent GPU work cannot contaminate
timings. It then scores clips with ordered CPU workers (`--jobs N`, default up to eight) under one
shared 24-megapixel image-working-set budget. Use `--jobs 1` for the serial reference. The HTML
entry point accepts `--scoring-jobs N`.

The runner records the executable, runtime shader tree, TensorRT engine, ONNX model, preprocessing
closure, clip evidence, configuration, and GPU-contention state. After rebuilding `sunshine.exe`
or editing a runtime HLSL file, the engine manifest can be intentionally stale; use `--allow-build`
once for the untimed preflight. Close other `sunshine.exe` instances before collecting performance
evidence or updating baselines.

## Dump scoring

While Host SBS is active, **Dump 3D** creates one atomic matched-frame directory. Score a group of
captures directly with:

```powershell
& $SbsbenchPython tools/sbsbench/sbsbench.py `
  --glob "E:/ApolloDev/sbs_dump/dump_2026*" `
  --json base.json

& $SbsbenchPython tools/sbsbench/sbsbench.py `
  --glob "E:/ApolloDev/sbs_dump/dump_NEW*" `
  --baseline base.json
```

See [Dump and replay format](DUMP_FORMAT.md) before generating or interpreting preview PNGs. The
reader accepts only the current UIP4/OCR8 dump schema; older experimental captures are
intentionally unsupported. An active schema-49 package authenticates the OCR8/UIP4/contour tuple for the
atomic final field's publication frame, ordinary Base, conditioned final field, and the resolver's
current-cover conflict histogram and source-time plane controller. It independently validates
current native glyph support, its projection and minimum slope-budget extension into `shadow_final_parallax.f32`, which is authenticated once as both the atomic final field and warp
input. An
active resolver authenticates all admitted current OCR8 paired covers without a second
geometry selection. Per-pixel RGB history survives cover count/order/split/merge changes. An inactive package uses the one canonical `none` descriptor.

The current OCR producer's centered-line eligibility and source-reference selection are defined in
the [Host SBS owner contract](../../docs/host-sbs.md#native-subtitle-contours-and-ui-plane).
Eligibility bounds current search; it does not pin a rectangle. Ordinary search uses current
line-center geometry to retain a wide guard only on the corrective side; ribbons retain their
published search extent. Exact geometry belongs to the linked owner contract.
Valid current RGB and all four finite unit-range bilinear probability contributors are required.
Strong current probability plus contrast samples select up to eight current actual ink
representatives from each paired tight core, merging dimmer antialiasing shades by normalized hue.
Current palette-matching short strokes or locally witnessed dark outlines can
then supply support, including model-missed matching strokes elsewhere inside that cover.
Probability alone and unrelated RGB edges grant no support. The canonical rule binds integer
palette pruning/matching, native contrast, the three-core-height tangent limit and fixed radius-six
outline probes. Fixed-RGB history remains diagnostic, with positive count retained only for
duplicate admission; it adds no four-observation wait or stable temporal text classification.
There is no provisional whole-cover protection. This overlay-color/stroke hypothesis remains
fallible on unrepresented colors, low-contrast strokes and matching scene texture; it is not
semantic text identification. Current quality requires actual full-glyph, negative and live checks.

Current schema-49 window-region packages preserve the complete authorized source rectangle at any
aspect ratio. `depth_input_region.json` schema 4 records the centered integer content rectangle in
the fixed DAV2 tensor and its edge-replicated excluded padding. Quantitative consumers must use
that content width for limiter and UI-plane steps and must project current OCR cover geometry only into that
content rectangle; treating the whole tensor as real source pixels is rejected.

Schema 49 packages no scalar/heat preview PNGs or redundant per-field shape sidecars. Generate a
verified diagnostic view from any retained authenticated `.f32` artifact outside the package:

```powershell
& $SbsbenchPython tools/sbsbench/generate_dump_previews.py `
  E:/ApolloDev/sbs_dump/dump_2026... shadow_final_parallax.f32 `
  --output-dir E:/ApolloDev/sbs_dump_previews/dump_2026...
```

## Whole-clip conversion boundary

Sunshine 3D's Web UI conversion is a native job manager and isolated native worker. `sbsbench`
does not implement a second scene-planning or media pipeline. The production workflow and its
opt-in native smoke test are documented in
[Offline Host 3D conversion](../../docs/whole-clip-sbs-pipeline.md).

## Tests

Run the maintained Python suite with:

```powershell
& $SbsbenchPython -m unittest discover -s tools/sbsbench -p "test_*.py"
```

Metric changes also require their applicable authenticated corruption validators and headset
inspection. The exact commands and qualification rules live in [Metrics and gates](METRICS.md).
