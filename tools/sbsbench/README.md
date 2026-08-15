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
python tools/sbsbench/run_eval.py --comparison-only --label control
```

Useful variants are:

```powershell
# Fresh matched A/B without the committed-baseline verdict
python tools/sbsbench/run_eval.py --comparison-only --label control
python tools/sbsbench/run_eval.py --comparison-only --label treatment `
  --report-control cmake-build-relwithdebinfo/sbs_eval/control `
  --report-allow-executable-diff

# One supported geometry or execution treatment
python tools/sbsbench/run_eval.py --comparison-only --label pop-1p0 `
  --extra --pop-strength 1.0
python tools/sbsbench/run_eval.py --comparison-only --label no-graph `
  --extra --cuda-graph off

# Prepared public suite
python tools/sbsbench/run_eval.py --suite extended --comparison-only --label public-control

# Publish an inspected V2 baseline from accepted production defaults only
python tools/sbsbench/run_eval.py --update-baselines
```

`--update-baselines` rejects `--extra`; first move an accepted setting into production defaults.
The process exit code is the verdict: `0` pass, `1` regression, and `2` invalid or incomplete
evidence.

Host SBS V2 has no model/profile selector. Production and the maintained benchmark harness use the
authenticated DAV2 Small calibration. The supported treatments are intentionally narrow and are
listed by `run_eval.py --help`; unrecognized historical options fail argument parsing.

### Current subtitle authority

The current dump reader accepts the authenticated OCR8 record and compact SLR12 locator state as
the only live subtitle authority. Retired SLR3--SLR9, GST/OGR/ORS, and offline overlay-detector
paths are not accepted as live or replay authority.

## Reports and comparisons

When a treatment was not run with `--report-control`, generate a report separately:

```powershell
python tools/sbsbench/generate_report.py <control-run> <treatment-run> <report.html>
python tools/sbsbench/compare_runs.py <control-run> <treatment-run>
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
python tools/sbsbench/sbsbench.py `
  --glob "E:/ApolloDev/sbs_dump/dump_2026*" `
  --json base.json

python tools/sbsbench/sbsbench.py `
  --glob "E:/ApolloDev/sbs_dump/dump_NEW*" `
  --baseline base.json
```

See [Dump and replay format](DUMP_FORMAT.md) before interpreting preview PNGs. The reader accepts
only the current SLR12/OCR8 dump schema; older experimental captures are intentionally unsupported.
An active schema-32 package authenticates the exact OCR8 record, compact SLR12 state, ordinary Base
field, selected conditioned field, and the resolver's bounded strict fallback placement policy. An
inactive package uses the one canonical `none` descriptor.

Current schema-32 window-region packages preserve the complete authorized source rectangle at any
aspect ratio. `depth_input_region.json` schema 3 records the centered integer content rectangle in
the fixed DAV2 tensor and its edge-replicated excluded padding. Quantitative consumers must use
that content width for limiter and SLR12 steps and must project OCR/SLR geometry only into that
content rectangle; treating the whole tensor as real source pixels is rejected.

## Whole-clip conversion boundary

Sunshine 3D's Web UI conversion is a native job manager and isolated native worker. `sbsbench`
does not implement a second scene-planning or media pipeline. The production workflow and its
opt-in native smoke test are documented in
[Offline Host 3D conversion](../../docs/whole-clip-sbs-pipeline.md).

## Tests

Run the maintained Python suite with:

```powershell
python -m unittest discover -s tools/sbsbench -p "test_*.py"
```

Metric changes also require their applicable authenticated corruption validators and headset
inspection. The exact commands and qualification rules live in [Metrics and gates](METRICS.md).
