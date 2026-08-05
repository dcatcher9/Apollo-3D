# Metrics, gates, and baselines

This document owns automatic metric semantics and qualification. Dataset provenance lives in
[Datasets](DATASETS.md); exact artifact authority lives in
[Dump and replay format](DUMP_FORMAT.md).

## Decision model

The evaluator deliberately has no scalar quality score. Correlated metrics must not cancel each
other, and more stereo volume cannot buy permission for rivalry, topology failure, or missing
content.

Every entry in [thresholds.json](thresholds.json) has one scope and one role:

| Field | Meaning |
|---|---|
| `scope: style` | Describes visible stereo volume; never an unbounded reward |
| `scope: perceptual` | Candidate viewing-risk evidence; experimental until headset-qualified |
| `scope: conformance` | Exact renderer, transport, and state contract evidence |
| `scope: gt-only` | Requires authenticated depth/disparity reference |
| `scope: temporal-only` | Requires a sequence and does not become a single-frame model label |
| `role: hard` | Non-tradeable fail-closed constraint |
| `role: primary` | Independent quality axis used in a decision |
| `role: diagnostic` | Supports interpretation but cannot decide alone |

The current compact policy exposes 37 metrics: 15 hard constraints, four primary axes, and 18
diagnostics. There are no qualified training labels. A metric must explicitly declare both a
training label and `label_status: qualified`; all current perceptual candidates remain
experimental, so label export abstains.

Unavailable or insufficient evidence is `n/a`, never zero. A hard constraint with evidence cannot
be averaged away.

## Exact correspondence basis

The exact source-U sidecar preserves requested coordinates beyond `[0,1]`. Per-eye clamp, fold,
and Jacobian diagnostics retain that raw demand. Stereo/comfort metrics invert both maps onto a
common source-U row and measure actual `x_right - x_left` only on unique, mutually visible,
in-range, forward-covered correspondences. Output-Jacobian weighting prevents a collapsed or
expanded interval from voting like ordinary area.

The existing ±3% image-disparity bound is an experimental engineering heuristic, not a retinal-
angle comfort guarantee. Headset FOV, display geometry, and user calibration are required before
it can become a perceptual claim.

## Spatial style and risk metrics

| Metric | Meaning | Preferred direction |
|---|---|---|
| `exact_visible_pop_spread_pct` | Source-structure-supported p0.5–p99.5 exact disparity spread | Higher, after all safety axes pass |
| `exact_positive_disparity_pct`, `exact_negative_disparity_pct` | Jacobian-weighted p99.9 signed disparity tails | Each at or below the configured bound |
| `exact_over_3pct_area_pct` | Mutually visible area beyond the current heuristic | Lower |
| `exact_mapping_stretch_pct` | Repeated/low-Jacobian source mapping | Lower |
| `exact_mapping_fold_pct` | Reversed coordinate steps | Lower |
| `warp_cross_row_shear_severity_pct` | Unsupported displacement change between adjacent rows | Lower |
| `experimental_stereo_window_crossed_burden_pct` | Crossed disparity actually cut by a lateral stereo window | Lower |
| `interocular_phase_orientation_burden_pct` | Registered equal-detail phase/orientation disagreement | Lower |
| `interocular_exposure_rivalry_burden_pct` | Source-relative exposure disagreement between eyes | Lower |
| `interocular_color_gain_rivalry_burden_pct` | Source-relative opponent-color disagreement | Lower |
| `exact_disparity_plateau_far_pct`, `exact_disparity_plateau_near_pct` | Rendered share pinned to an endpoint | Lower when real depth was lost |
| `exact_disparity_plateau_far_depth_span_pct`, `exact_disparity_plateau_near_depth_span_pct` | Processed-depth range swallowed by each plateau | Lower |

`exact_visible_pop_spread_pct` can increase when disparity clips into endpoint plateaus; always read
it with the plateau fraction and swallowed-depth-span metrics. The span uses the processed field
that drove the warp. It cannot recover relief already lost in raw model inference; compare
`raw_*.f32` when attributing upstream flattening.

The cross-row shear metric complements stretch/fold: stretch and fold inspect horizontal mapping
inside one row, while shear detects a whole row jumping sideways. It excludes source-supported
horizontal boundaries and remains diagnostic pending broader headset qualification.

Disocclusion bad-fill remains a standalone falsifier. Mono input cannot authenticate newly exposed
background, so an automatic score could punish plausible fill or reward copied foreground. It is
not in the compact gate or training-label export.

## Renderer and state conformance

The retained `container_scale` state word is ABI compatibility only and must equal `1.0` on every
frame. Source-U safety is measured on the emitted field after its stateless pointwise soft
container; a frame-global effective-gain reduction is a conformance failure, not adaptation.

| Metric | Contract |
|---|---|
| `exact_binocular_support_pct` | At least 80% mutually visible unique-source support |
| `source_coverage_pct` | At least 99% supported interior reproduction within 4/255 luma |
| `image_integrity_pct` | At least 80% exact mapped-source texture integrity |
| `source_coverage_worst_patch_bad_pct` | At most 20% bad area in the worst local patch |
| `image_integrity_worst_patch_bad_pct` | At most 25% bad texture in the worst patch |
| `exact_symmetry_residual_p95_pct` | At most 0.1% common-camera residual |
| `exact_polarity_ok` | Correct global warp ordering |
| `exact_local_polarity_component_pct` | Local polarity diagnostic |
| `vmisalign_p99_pct` | At most 0.1% texture-supported vertical mismatch |
| `shot_state_pulse_mismatch` | No missing or extra authenticated cut pulse |
| `shot_state_trace_inconsistent` | No invalid cut/camera state transition |
| `shot_state_initialized_ok` | Startup state initializes as declared |
| `shot_state_relative_escape_ok` | Sustained-motion escape behaves as declared |
| `shot_state_accepted_pulse`, `shot_state_expected_pulse` | Diagnostic accepted/expected pulse counts |

Conformance proves that the renderer reproduced its selected coordinate and that the scene-cut
state followed its declared schedule. It does not prove that the monocular depth or selected
disparity is perceptually ideal.

## Temporal metrics

| Metric | Meaning | Preferred direction |
|---|---|---|
| `static_jitter_p95` | Worse-eye source-conditioned change on source-static support | Lower |
| `flow_temporal_p95` | Flow-compensated SBS residual after signed source change is removed | Lower |
| `depth_gt_lag_f1_p95` | Advantage of previous-frame GT boundaries over current GT | Lower |

`static_jitter_p95` removes moving source support plus a disparity-radius margin, then subtracts
the signed mono-source change. Exposure/noise reproduced by both eyes cancels; equal-and-opposite
eye change is penalized twice. Scene cuts and camera moves with less than 10% support abstain.

`flow_temporal_p95` requires authenticated flow and photometric support. The retired classical
tile-flow fallback was unreliable on articulation and occlusion; SEA-RAFT is an optional,
separately versioned diagnostic rather than canonical truth.

## Ground-truth depth metrics

| Metric | Meaning | Preferred direction |
|---|---|---|
| `depth_gt_affine_nrmse_pct` | Robust positive-affine aligned error over valid GT | Lower |
| `depth_gt_edge_f1` | One-pixel-tolerance depth-boundary F1 | Higher |
| `depth_gt_edge_support_pct` | Share of valid pixels supporting the boundary F1 | Evidence context |
| `depth_gt_polarity_ok` | Sign of prediction-to-GT fit | Must pass |

The affine fit rejects negative scale and requires at least 5% valid support. Edge F1 gates only
when `depth_gt_edge_support_pct >= 1.0`; below that it remains diagnostic because tiny weak-gradient
changes can flip binary membership without meaningful geometry movement.

## Evidence floors

Evidence minima are part of the contract:

- signed disparity tails require 1,024 mutually visible samples;
- visible volume and local polarity require 256 samples;
- cross-row shear requires 512 qualified pixels;
- texture integrity requires at least 0.1% independently textured support; and
- vertical alignment requires at least 2% overlapping texture-tile support.

These are conservative screening floors, not statistical-independence proofs. Lower support
abstains.

## Baselines and controlled A/Bs

The retired V1 baseline set is not valid V2 evidence. Until a reviewed V2 baseline is published,
use `--comparison-only` for authenticated matched A/Bs. A future committed baseline represents the
canonical production configuration and must bind the clip hash, evaluator schema,
metric/threshold contract, V2 calibration, stepping semantics, model, and runtime provenance.
`conformance-only` fixtures remain exempt from numeric baselines because exact hard invariants
decide them directly.

Use `compare_runs.py` rather than hand-averaging. It intersects finite evidence from identical
clips and exposes category splits. A report permits a changed executable, model, or configuration
only with the corresponding explicit `--report-allow-*` flag.

`rescore_run.py` can refresh only an explicitly comparison-only run from preserved authenticated
artifacts:

```powershell
python tools/sbsbench/rescore_run.py `
  cmake-build-relwithdebinfo/sbs_eval/<comparison-run> `
  --in-place
```

It cannot update committed baseline verdicts. Use `run_eval.py` for baseline publication.

Performance evidence comes from D3D11 GPU timestamps around the real stages. GPU harnesses remain
serial; CPU scoring may run in parallel under the shared pixel budget. A running second host marks
GPU contention and disables the performance gate.

## Required validation

The maintained unit-test entry point is:

```powershell
python -m unittest discover -s tools/sbsbench -p "test_*.py"
```

Run each standalone validator affected by a metric change:

```powershell
python tools/sbsbench/validate_real_stereo_window_metric.py `
  --suite both --strict --output <window.json>

python tools/sbsbench/validate_disocclusion_topology_real_sources.py `
  --suite both --strict --output <topology.json>

python tools/sbsbench/validate_interocular_phase_chroma.py --output <phase.json>

python tools/sbsbench/validate_interocular_photometric_rivalry.py `
  --run <current-run> --output <photometric.json>
```

Controlled corruption ladders must include benign controls, preserve abstention/support counts,
and remain unable to auto-promote a label. Passing a synthetic unit test or validator is necessary,
not sufficient: source-content false positives and headset-correlated inspection still decide
qualification.

Optional learned diagnostics are governed separately by [Optional oracles](ORACLES.md) and never
enter automatic gates or labels.
