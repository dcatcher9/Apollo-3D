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
& $SbsbenchPython -c "import json, platform, sys, numpy, PIL; print(json.dumps({'executable': sys.executable, 'python': platform.python_version(), 'numpy': numpy.__version__, 'pillow': PIL.__version__}, sort_keys=True))"
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
runtime shader tree, DAV2
engine/ONNX, OCR engine/contract ONNX, and generated coordinate contract across both serial legs.
The default gate requires at least one actual reuse and proves that every reuse retained the
preceding raw DAV2 field bit-exactly. Ordinary subtitle work also holds SLR80, condition parameters,
and `final_parallax_<frame-id>.f32`; cadence-due work may instead publish a current subtitle
observation on the reused depth. The authenticated trace distinguishes those cases and rejects an
ordinary OCR marker on reuse. The runner derives GPU history-owner age from both the frame ID and
the exact source-observation timeline—not from processing speed or a host baseline delta. No run
may exceed the production four-frame cap or strict `<100 ms` observation-age bound, and every infer
raw field must match the same-frame force control.
Missing, extra, wrong-sized, or misidentified artifacts fail closed. These trace and artifact
invariants guard the adaptive subtitle-clock bug without image-tuned thresholds or headset
judgment.

The report measures final-field step and jerk for both force control and adaptive treatment. Every
serial harness stage from `sbs_perf.json` is also aligned by name/sample count and reported with mean and
p95 control-to-treatment deltas; nested GPU stages are kept separate and are never summed.
Optional `--max-scene-residual-delta-p95` and `--max-subtitle-residual-delta-p95` bounds can turn
the report's downsampled image diagnostics into clip-specific CI gates.

This replay deliberately supplies no Desktop Duplication damage history, live window authority,
or encode-deadline scheduling. Those remain live admission inputs. Request formation, opaque
chaining, force-completion release, estimator, conditioner, renderer, and GPU trace are the same
production code, so the offline path does not maintain a second infer/reuse state machine.
The runner uses private replay harness schemas 27 (force oracle) and 26 (conditional treatment),
plus metadata schema 3, leaving formal `run_eval.py` schema 22 and its baseline evidence untouched.

Host SBS V2 has no model/profile selector. Production and the maintained benchmark harness use the
authenticated DAV2 Small calibration. The supported treatments are intentionally narrow and are
listed by `run_eval.py --help`; unrecognized historical options fail argument parsing.

### Current subtitle authority

The current dump reader accepts the authenticated OCR8 record and compact SLR13 locator state as
the only live subtitle authority. Retired SLR3--SLR9, GST/OGR/ORS, and offline overlay-detector
paths are not accepted as live or replay authority.

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

See [Dump and replay format](DUMP_FORMAT.md) before interpreting preview PNGs. The reader accepts
only the current SLR13/OCR8 dump schema; older experimental captures are intentionally unsupported.
An active schema-36 package authenticates the OCR8/SLR13 tuple for the atomic final field's
publication frame, ordinary Base, conditioned final field, and the resolver's bounded strict
fallback placement policy. It replays SLR13 directly into that final field and requires
`warp_depth` to equal it bit-for-bit. An
active resolver also authenticates the strict symmetric bottom-corner ordinary-core qualification
and its ribbon exemption. An inactive package uses the one canonical `none` descriptor.

Current schema-36 window-region packages preserve the complete authorized source rectangle at any
aspect ratio. `depth_input_region.json` schema 3 records the centered integer content rectangle in
the fixed DAV2 tensor and its edge-replicated excluded padding. Quantitative consumers must use
that content width for limiter and SLR13 steps and must project OCR/SLR geometry only into that
content rectangle; treating the whole tensor as real source pixels is rejected.

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
