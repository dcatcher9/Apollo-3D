# sbsbench — visual metrics for host SBS 3D output

Offline visual metrics computed on **real "Dump 3D" output** (the actual `sbs.png` the client
receives), with authenticated depth/stereo/flow references where a dataset supplies them. It runs
and measures the real pipeline rather than a hand-maintained CPU replica, so its numbers track the
images shown on the headset. This is the visual half of the host benchmark; see
[docs/sbs-benchmark-plan.md](../../docs/sbs-benchmark-plan.md). The perf half is the in-app
`sbs_3d_perf_stats` timing.

## Workflow
1. Capture frames: tap **Dump 3D** in the XR bar → `E:\ApolloDev\sbs_dump\dump_*/` (needs
   `APOLLO_SBS_DUMP` set; `run-dev.cmd` already sets it). Grab several across a scene / both
   modes; each folder has `source.png`, `depth.png`, `sbs.png`, `meta.txt`.
2. Score a set and save a baseline **before** your change:
   ```
   python tools/sbsbench/sbsbench.py --glob "E:/ApolloDev/sbs_dump/dump_2026*" --json base.json
   ```
3. Make the change, re-dump the same kind of scene, and diff:
   ```
   python tools/sbsbench/sbsbench.py --glob "E:/ApolloDev/sbs_dump/dump_NEW*" --baseline base.json
   ```
   → prints per-metric deltas (`exact_visible_pop_spread_pct 2.4 -> 2.6  +0.2 (+8%)`).

Dependencies: `numpy` + `Pillow` only (system Python 3 is fine).

## One-command eval loop (start here)

Production configuration resolves one profile at host startup:

```
sbs_3d_profile = apollo   # default
```

The profile supplies the complete stack. A configuration-only preset can use
`sbs_3d_profile_<name>_<parameter>` keys; selecting another preset requires restarting Apollo.
For example:

```
sbs_3d_profile = cinema
sbs_3d_profile_cinema_depth_model = depth_anything_v2_base_fp16
sbs_3d_profile_cinema_pop_strength = 1.25
```

Every profile uses Apollo's occlusion-aware warp; profiles select depth-processing, model, pop,
and performance parameters only. Unspecified values inherit Apollo defaults. Ordinary
`sbs_3d_*` keys are applied last and explicitly override the corresponding selected-profile
parameter. Artemis selects only Normal or Host SBS AI; it does not select a host profile, model,
or individual parameter. Each encode device receives an immutable startup configuration snapshot.

Production always keeps a bounded two-frame color buffer and presents only exact color/depth pairs,
repeating the last completed SBS frame while inference is busy. The former wrong-frame async path,
live synchronous mode, `depth_frame_mode`, and `depth_fps` profile parameters were removed. The
offline evaluator retains a private synchronous finish primitive so every source frame can be
scored deterministically; it is not selectable by production configuration.

Wire extensions: `0x3003` byte 0 selects host SBS mode (`bytes 1..3` are reserved and ignored),
`0x3004` requests a debug dump, and `0x3006` reports depth-engine status. Apollo builds, loads,
creates a reusable execution context for, and warms the single selected model at startup.

```
python tools/sbsbench/run_eval.py                     # all committed clips vs committed baselines
python tools/sbsbench/run_eval.py --update-baselines  # accepted defaults only; --extra is rejected
python tools/sbsbench/run_eval.py --extra --subject-lock 0.6   # pass supported A/B levers
python tools/sbsbench/run_eval.py --label profile-b --conf profile-b.conf --report-control cmake-build-relwithdebinfo/sbs_eval/profile-a --report-allow-config-diff
python tools/sbsbench/run_eval.py --label model-b --conf model-b.conf --report-control cmake-build-relwithdebinfo/sbs_eval/model-a --report-allow-config-diff --report-allow-model-diff
python tools/sbsbench/run_eval.py --label cadence-b --report-control cmake-build-relwithdebinfo/sbs_eval/cadence-a --report-allow-depth-step-diff --extra --depth-every 2
python tools/sbsbench/run_eval.py --label code-b --report-control cmake-build-relwithdebinfo/sbs_eval/code-a --report-allow-executable-diff
python tools/sbsbench/run_eval.py --comparison-only --label ab-control  # fresh A/B; no committed gate
python tools/sbsbench/run_eval.py --suite extended --label public-control # prepared public suite
python tools/sbsbench/rescore_run.py cmake-build-relwithdebinfo/sbs_eval/<comparison-run> --in-place
```

Optional eval-only FLIP/RAFT-Stereo/SEA-RAFT/iSQoE diagnostics are documented in
[OFFLINE_LEARNED_ORACLES.md](OFFLINE_LEARNED_ORACLES.md); they never affect gates or labels.

Generate or regenerate an authenticated HTML report with the Windows-safe parallel launcher:

```
python tools/sbsbench/generate_report.py <control-run> <treatment-run> <report.html>
```

Metric scoring parallelizes independent frames with one reusable process pool. The documented
`generate_report.py` entry point safely provides the same process backend on Windows; only a direct
`build_report.py` invocation falls back to threads because that HTML module is intentionally
top-level. Numeric libraries stay single-threaded inside each worker to avoid nested CPU
oversubscription. The automatic worker cap is 24 and is reduced by a 24-megapixel in-flight image
budget. Developers can override these bounds with `SBSBENCH_SPATIAL_WORKERS` and
`SBSBENCH_SPATIAL_PIXEL_BUDGET_MPX`; ordinary gates should use the defaults. Profile preserved
artifacts without rerunning TensorRT or the harness:

```
python tools/sbsbench/profile_eval.py <run> --clips-root tools/sbsbench/clips --workers 1
python tools/sbsbench/profile_eval.py <run> --clips-root tools/sbsbench/clips --workers 24
```

Profiler JSON records the backend, pixel budget, metric hash/runtime, repository revision,
platform, and logical processor count. It is local timing evidence, never an evaluator gate.

The authoritative report still hashes and validates control and treatment independently. When a
performance-only A/B has byte-identical SBS/depth/map inputs, it transiently reuses the first
in-memory pixel remeasurement; no cached JSON metric authenticates itself.

Committed baselines are canonical production-profile runs. Move an accepted setting into the
profile/config before `--update-baselines`; the runner rejects harness `--extra` overrides so a
treatment baseline cannot later masquerade as the default. Missing hard, baseline-supported
primary, or configured performance evidence fails closed. `rescore_run.py` is metric-only: it
accepts only explicitly comparison-only runs produced by the current evaluator schema, preserves
that schema, verifies the recorded source hashes, and uses the run's recorded clip root.

**Profile / model (important):** the evaluator resolves the depth model from the selected profile
and then the explicit `sbs_3d_depth_model` override, exactly like production. There is no separate
`--mode` model selector. Never compare reports using different model/config hashes as if they were
a controlled feature A/B.

Harness A/B levers (after `--extra`):
- `--pop-strength F` — multiply the final shared stereo-parallax field (`0.25`-`2`; default
  `1.25`). This is the user-facing pop control. It is separate from the internal
  854-pixel Bestv2 calibration that keeps apparent depth stable across source resolutions. Below
  854 pixels, production preserves Bestv2's literal pixel shift and independently applies the
  reference-aspect correction; non-16:9 low-resolution inputs therefore receive both effects.
  This behavior requires a dedicated 4:3 A/B before changing.
- `--literal-bestv2` — comparison-only reference mode. It bypasses production resolution,
  aspect, and pop scaling and writes the fact to `contract.json`; never use it for quality gates or
  committed baselines.
- `--depth-override-root DIR` — comparison-only offline reference that replaces explicitly
  supplied processed-depth frames while retaining the real subject state and production warp.
  Generate classical-flow reuse-2 references with `prepare_depth_motion_reference.py`; this is
  an experiment boundary, not a production feature or a permitted committed baseline. The
  schema-3 manifest binds the treatment to the exact source hash, policy and override-frame IDs;
  missing, extra or stale override frames fail the run before scoring.
- `--depth-override-all` requires `--depth-override-root` and `--depth-every 1`; it replaces every
  inferred depth frame for a spatial/temporal processor oracle. `prepare_flow_ema_reference.py`
  builds the exact-flow EMA oracle used to reject recursive and one-frame flow history. It requires
  exact `gt_flow` sidecars and a source run made with `--ema 1 --ema-edge-change 0`.
- Bestv2 is the only disparity field. It uses the preset's source-pixel FG/MG/BG shifts
  (`-9/-3/+2.4`), `.35` parallax
  balance, `1.11/1.05` multipliers, `.008` zero-parallax trim, dynamic convergence `.006`,
  `.071` safety cap. Rejected subject-plane lock and per-eye sharpen paths have been removed.
  Before subject state initializes, output
  remains flat instead of using the removed legacy divergence/focal-plane fallback.
- `--depth-short-side N` — depth inference short side (default 432). Use 336 to A/B
  back to the old under-resolved default.
- `--simulate-hdr --hdr-scale F` — direct-harness color-path smoke: decode the PNG source into
  linear FP16 scRGB, scale its luminance (`4` = 320-nit diffuse white), run the HDR depth and warp
  paths, and write a tone-mapped PNG plus `hdr_output_stats.json`. This checks FP16 preservation,
  finite values through the pre-encode SBS stage. It is not a
  PQ/NVENC/headset colorimetric evaluation; do not compare its PNG metrics to SDR baselines.
- `--ema F` — per-pixel depth EMA override (`1.0` = off).
- `--ema-edge-change F --ema-edge-gradient F --ema-edge-strength F`
  — accepted flowless moving-edge EMA (`0.05`, `0.02`, `0.25`). It preserves ordinary EMA outside a deterministic
  depth-transition mask and blends masked pixels toward current depth. A 16-bit
  `ema_mask_<frame>.png` locality artifact is required whenever enabled.
- `--subject-lock F` — subject anchor strength (e.g. `0.95`).
- `--subject-recenter F` — subject depth-field recenter strength.
- `--subject-stretch` — shape_depth_for_pop 5/95 percentile stretch (default on in the permanent
  Bestv2 subject path).
- `--no-subject-stretch` — disable that stretch for an accepted-feature ablation.
- `--zero-plane legacy|subject|median|background` — choose a shot-latched screen-plane anchor.
  The three explicit modes are experimental camera-offset treatments; they preserve
  symmetric eye geometry and the disparity scale, and update only at startup or a hard scene cut.
  `legacy` remains the production default because fixed anchors produced scene-dependent
  tradeoffs rather than a suite-wide improvement.
- `--cuda-graph on|off` — capture and replay the TensorRT enqueue when the mapped D3D tensor
  addresses remain stable. The first enqueue for a new address/shape is an uncaptured warmup.

Production uses the equivalent `sbs_3d_pop_strength = F` key. Like every individual SBS key it
overrides every profile; omit it to retain each profile's configured/default value (`1.25`).
Scene-adaptive pop is enabled by default. It selects once per scene between the `1.25` floor and
the validated `sbs_3d_adaptive_pop_max = 1.30` ceiling using normalized-depth edge density, then
holds the result bit-stable until a hard cut. Set `sbs_3d_adaptive_pop = false` for the fixed floor.
The former experimental 2.0 ceiling was rejected: comfort remained within 3%, but temporal and
warp artifacts regressed. Symmetric left/right geometry is unchanged.
The equivalent production key is `sbs_3d_zero_plane`. Keep it at the default `legacy` outside a
controlled headset or evaluator A/B; `subject`, `median`, and `background` change convergence
placement without changing the configured pop multiplier.
CUDA Graph replay is enabled by default for every profile. Use `sbs_3d_cuda_graph = false` only
for driver diagnosis or a controlled performance A/B; unsupported/capture-failed systems already
fall back to ordinary TensorRT enqueue automatically.

Exit code is the verdict (0 pass / 1 regression / 2 setup error), so the eval→fix→eval loop is
scriptable. `results.json` carries provenance (git sha+dirty, executable, runtime-shader tree,
exact TensorRT engine/ONNX hashes, clip hashes, and gpu-contention flag) and, for every
triggered/regressed metric, the **worst frame index** to look at. The gate
thresholds live in [thresholds.json](thresholds.json); the pinned SBS config in
[bench.conf](bench.conf); baselines in `baselines/` (regenerate in the same commit whenever
bench.conf, the clip set, or a metric definition changes). Guards: the runtime must publish a
fresh `<model>.active-engine.json` naming the exact nonempty TensorRT artifact and current ONNX
SHA-256; an unrelated/stale engine never satisfies preflight. `--allow-build` performs one untimed
one-frame harness preflight and revalidates that manifest before any measured clip. The runner
warns and skips the perf gate if another sunshine.exe is running.

Configured disparity/integrity engineering bounds apply even in comparison-only runs. The image-
relative disparity bounds are not a calibrated retinal-angle comfort guarantee. Baseline updates are staged in
memory and written atomically only after every clip passes those bounds, so a broken render cannot
become the new normal.

The harness records `warp_infer` with D3D11 GPU timestamps around the Apollo warp;
`sbs_composite_cpu` measures submission overhead only.
For A/B runs, pass `--report-control <control-run-dir>` and the evaluator writes `report.html`
before printing its conclusion, including when the treatment exits with regressions.

Metric notes: exact-map topology is evaluated in normalized or reference-resolution coordinates.
Percent/normalized outputs are preferred; raw pixel diagnostics are never compared across clip
resolutions. Harness depth is 16-bit so
sub-1/255 changes remain measurable.

**Eval schema 32 / harness contract 16:** `run_eval.py` pins the profile, model, and zero-plane
mode explicitly, records the exact Sunshine executable, runtime HLSL tree, engine, and ONNX
hashes, and has no alternate warp selector. A normal report requires all four to match;
`--report-allow-executable-diff` explicitly permits a code/shader A/B, while
`--report-allow-model-diff` permits an engine/ONNX A/B.
By default the
harness submits and consumes exactly one inference per source frame, so EMA and normalization
update once. `--depth-every N` is an explicit comparison-only cadence treatment: color advances
while the last completed depth/subject geometry is reused, and the contract records `reuse-N`.
Source, raw-model (`raw_*.f32`), pre-warp depth (`depth_*.png`), exact scalar-R32 **raw**
`Reproject` source-U coordinates
(`warp_map_*.f32` plus `warp_map_shape.json`) are joined with SBS artifacts by
numeric frame identity, never list position. `warp_mask_*.png` remains a raw validity companion
for internal coverage audits; it is not a decision metric or report section. Baselines are rejected
with setup exit 2 if mode,
model, baseline-update provenance, schema, stepping semantics, config hash, metric
implementation/threshold hash, or clip hash differs. Runner/gating semantics are versioned by the
schema rather than the runner's comments or diagnostic wording. Schema 27 introduced the stricter
`label_contract_sha256`/`label_context_sha256` and full per-frame numeric evidence. Schema 28 makes
model-label export fail closed: a metric must declare both `label` and the exact
`label_status: "qualified"`; an omitted, unknown, or experimental status is excluded. Every frame
still retains its complete numeric metric vector for evaluation and later qualification, while its
`labels` object contains only qualified metrics plus validity/support and abstention state. Cached
model labels must match the hashes, `meta.training_labels` and each clip's `label_summary`; aggregate
scores are not labels. Schema 29 replaced the broad proxy list with explicitly scoped
perceptual/conformance/GT/temporal evidence. Schema 32 replaces inferred source-profile artifact
scores with exact-map topology, strict renderer conformance, authenticated GT, and registered
temporal evidence. Old results and baselines are intentionally stale.
Output folders
are cleared before reuse. `--output-every N` reduces saved artifacts while still processing every
input frame, so sampling cannot change temporal state.

There are deliberately **no qualified model labels in the current threshold contract**. Exact
comfort, clamp, topology, binocular-conflict, and mapping-stretch candidates are
marked `experimental` pending controlled-corruption and headset calibration, so schema-32 frames abstain with
`no_qualified_training_labels`. Exact mapped-source residual, color, coverage and integrity
remain active report/run-gate evidence but have `scope: "conformance"` and no training `label`:
they verify that the renderer reproduced the coordinate selected by its own warp, which cannot
independently prove that the selected correspondence was perceptually correct. Promotion requires
an explicit review changing `label_status` to `qualified`; merely adding `label` is insufficient.

Schema 3 added optional `gt_depth/frame_*.png` clip sidecars and includes them plus their semantic
`gt_depth_kind` metadata in the clip hash. A matching `gt_depth_valid/frame_*` sidecar authenticates
valid/visible pixels; disparity clips that require GT fail closed when it is missing or mismatched.
Invalid, occluded, non-finite, and resize-unsupported pixels never enter accuracy or boundary
statistics. Missing GT means reference metrics are absent—not zero and not inferred. `flat_page`
and `fast_motion` carry deterministic 16-bit disparity ground truth generated by
`make_synth_clips.py`; recorded movie clips remain no-reference.

Schema 4 also accepts native float32 `gt_depth/frame_*.npy` and exact forward-flow
`gt_flow/frame_*.npz` sidecars. Each flow sidecar belongs to its current frame and contains
`flow` (previous→current, H×W×2 pixels) plus an optional `valid` mask. The evaluator forward-splats
that source-grid flow to current coordinates and still requires photometric agreement; clips
without exact flow abstain in canonical evaluation and may use the separately versioned SEA-RAFT
diagnostic. Both reference folders are part
of the clip identity hash.

`gt_right/frame_*` is authenticated **diagnostic stereo reference material**, declared with
`"reference_stereo_available": true`; it is not a consumed ground-truth metric, training label,
or gate exemption. Public clips that only provide this pair use
`"evaluation_role": "reference-only"` and remain useful for source diversity and separately
versioned stereo oracles. Existing external caches using the retired `required_gt_stereo` spelling
are migrated in memory to this diagnostic declaration and never published or interpreted as GT;
new preparation writes only the new contract. Sintel preparation also preserves the
official occlusion and out-of-frame masks and writes explicit all-pixel and non-occluded validity
masks. Every semantic scoring sidecar participates in the clip identity hash.

Clips may declare `"expected_flat": true` in `meta.json`. They remain explicit false-stereo and
depth-hallucination diagnostics but do not vote on the general-content enhanced-pop objective.
Reports name
both run directories and show control/treatment provenance separately, even when their CLI args
are identical because the treatment is a code change.

## Deterministic clips via the headless harness (recommended)
Single dumps are sporadic and headset-bound. For repeatable A/B and **temporal** metrics, drive
the real pipeline over a fixed frame sequence with the built-in `--sbs-bench` subcommand (Tier-1
harness — runs the real estimator + real composite shaders, no game/client).

### The committed clip set (quick eval)
A small clip set lives in **`tools/sbsbench/clips/<name>/frame_*.jpg`** (24 JPEG frames) so eval is
fast and reproducible with no per-run preprocessing. It contains five recorded movie clips
(c339/c525/c647/c747/c841), one open cel-anime clip, one official AI-video-model gallery clip,
and three generated failure-mode clips from [make_synth_clips.py](make_synth_clips.py):

| clip | targets | validated fingerprint |
|------|---------|----------------------|
| `anime_morevna_closeup` | cel outlines, flat colors, face-depth hallucination | visually clean source silhouette; intentional ink/white clothing annotated separately from warp-created halo |
| `aigen_cogvideox_rain` | AI-video human motion, rain, splash, blur and low contrast | source already contains rain/splash rims and generative temporal inconsistency; tests whether the warp adds to them |
| `scene_cut` | depth-normalization response across a hard cut | exact geometry and temporal evidence verify one state update per source frame |
| `flat_page` | flat-content depth hallucination + amplification | static-input false-stereo and depth-boundary noise floor |
| `fast_motion` | known 30 px/frame motion | current-frame depth separates warp/edge behavior from live async lag |

### Public extended suite (decision eval)

The quick suite remains useful for iteration, but the final warp/profile decision uses a separate,
reproducibly prepared public suite. Media is cached outside Git (default
`E:\ApolloDev\sbs_bench\datasets`); the repository commits the exact URLs, source checksums,
frame windows, adapters, and baselines:

```
python tools/sbsbench/prepare_public_datasets.py
python tools/sbsbench/run_eval.py --suite extended --comparison-only --label public-apollo
```

`extended-v3` contains twelve visually inspected 24-frame clips. The first four remain:

| clip | source | coverage / reference |
|------|--------|----------------------|
| `bonn_person_walk` | Bonn RGB-D Dynamic | real walking person, silhouette motion, registered depth |
| `bonn_person_close` | Bonn RGB-D Dynamic | close person/robot occlusion, registered depth |
| `tartanair_house_easy` | TartanAir V2 | attic/bed transition, exact metric depth and flow |
| `tartanair_house_motion` | TartanAir V2 | indoor/outdoor rotation, exact metric depth and flow |

The v2 expansion adds independent cinematic and outdoor-driving content:

| clip | source | coverage / reference |
|------|--------|----------------------|
| `sintel_ambush` | MPI Sintel Stereo | cinematic non-rigid action and rendered right-eye reference |
| `sintel_market` | MPI Sintel Stereo | people, clutter, depth layering and rendered right-eye reference |
| `vkitti_drive_clone` | Virtual KITTI 2 | clear outdoor driving and exact metric depth |
| `vkitti_drive_rain` | Virtual KITTI 2 | rainy low-contrast driving and exact metric depth |

The v3 expansion adds high-resolution cinematic stereo from the CC BY 4.0 Spring benchmark:

| clip | source | coverage / reference |
|------|--------|----------------------|
| `spring_skeleton_cave` | Spring Stereo | deep cave layering, character motion, bones and volumetric light |
| `spring_character_close` | Spring Stereo | face/hair close-up, thin prop and shallow depth of field |
| `spring_daylight_path` | Spring Stereo | bright outdoor path, rocks, vegetation and fine ground texture |
| `spring_forest_seeds` | Spring Stereo | thin airborne structures, defocus and layered forest background |

The suite caught and now guards the former source/depth resolution-normalization bug. Apollo's
source-pixel shifts are normalized by the source/eye geometry, not the smaller inference texture.

The manifest is [datasets/manifest.json](datasets/manifest.json). Bonn derivatives remain local;
its official page requests citation but does not provide a redistribution grant. TartanAir V2 is
CC BY 4.0. `prepare_public_datasets.py` associates Bonn RGB/depth by nearest timestamp, preserves
TartanAir float depth without quantization, decodes its 16-bit flow PNG losslessly without OpenCV,
and writes source frames only at the clip root so reference images can never be mistaken for input.
Spring's official left/right archives are approximately 3 GB each, and the Sintel stereo archive
is approximately 2.6 GB. Their adapters use validated HTTP byte ranges to extract only the pinned
frame windows. Each range-backed clip additionally pins a SHA-256 over the decoded frame,
depth, and mask evidence that evaluation actually consumes, so a changed remote payload fails
before the prepared directory is published.
Set `APOLLO_SBS_DATASETS` or pass `--cache` to relocate the cache.

The harness sizes the SBS output to the **input** resolution, so short clips keep each A/B sample
bounded (approximately 8 s harness + 3 s scoring per clip on the reference machine):

```
cd cmake-build-relwithdebinfo
./sunshine.exe E:/ApolloDev/config/sunshine.conf --sbs-bench \
    --frames ../tools/sbsbench/clips/c525 --out out/c525 \
    --model depth_anything_v2_fp16
python tools/sbsbench/sbsbench.py --seq out/c525 --frames tools/sbsbench/clips/c525 --json base.json
python tools/sbsbench/sbsbench.py --seq out/NEW  --frames tools/sbsbench/clips/c525 --baseline base.json
```

**Eval never resizes the input** — the SBS output tracks the input size, so to run a *full-res*
(slower, more sensitive) eval, just point `--frames` at a full-resolution frame dir; use `--eye-h`
only to pin a specific output height. Metric values are resolution-dependent, so keep baselines
per clip-set (a small-clip baseline isn't comparable to a full-res one; A/B deltas within one are).

### Adding / regenerating clips
```
python tools/sbsbench/split_video.py clip.mp4 -o tools/sbsbench/clips/mine --width 854 --jpg --max 24
```
Drop `--width/--jpg` for a full-resolution PNG clip. The gated runner uses current-frame depth
with one update per source frame.

Every A/B HTML report now writes a sibling `decision.json`. Both are generated from the same
already-unwrapped per-clip aggregate dictionaries, so automation should consume that sidecar rather
than reimplementing decision parsing. Sidecar schema 3 includes `metric_sha256`,
`report_sha256`, and both inputs' executable/shader/engine/ONNX hashes, plus both canonical run
gates. Missing evidence and performance or
baseline failures cannot be relabelled as a candidate by the report's metric-only A/B view. To
check whether a depth processor compresses or clips depth:

```
python tools/sbsbench/audit_depth_transform.py <control-run> <treatment-run>
```

This compares every native 16-bit pre-warp depth frame, endpoint saturation, p95-p5 spread, final
stereo spread, and available GT depth metrics, writing `depth_transform_audit.json`.

Before letting an edge-confidence heuristic control a processor, calibrate it against an existing
real-pipeline run without changing the metric/gate contract:

```
python tools/sbsbench/audit_depth_confidence.py <run> --out <report-dir>
```

The standalone audit writes `report.html` and `summary.json`. It reports two independent verdicts:
warp risk is validated against final-eye source-relative artifacts, while model risk is validated
against the placement of predicted boundaries where exact GT exists. The latter does not measure a
GT boundary omitted by the model, and the report exposes those missing-prediction frames. A passing
warp verdict only authorizes a controlled warp-processor A/B; that processor must still pass the
normal core and extended decision gates. Model-boundary validation remains diagnostic and does not
authorize depth or EMA changes by itself.

## Compact metric policy and feature decision
The evaluator deliberately has no scalar quality score. Correlated metrics must not cancel each
other, and a loss of stereo volume cannot buy an artifact improvement. Legacy `score`, `q_clean`,
and `q_depth` fields in schema-28-or-older artifacts are ignored; current runs do not emit them.

Every metric in [thresholds.json](thresholds.json) has one explicit scope:

- `style`: visible stereo volume used to choose the strongest already-safe result, never an
  unbounded training reward;
- `perceptual`: a potential DA-V2 risk/safety target, but still `experimental` until controlled
  corruptions, benign transforms, diverse clips, and headset inspection qualify it;
- `conformance`: exact renderer/transport contract evidence, never a perceptual model label;
- `gt-only`: authenticated depth/disparity-reference validation;
- `temporal-only`: sequence evaluation, not a label for the single-frame policy head.

`hard` metrics can reject an automated screen, `primary` metrics vote on coequal axes, and
`diagnostic` metrics only support a conclusion. A screen candidate is not a perceptually validated
candidate. There are currently no qualified training labels, so export
abstains instead of teaching the model an unvalidated number. Stereo volume and perceptual defects
are coequal, non-compensating axes: stronger depth cannot buy permission for rivalry, topology
failure, or missing content. The evaluator intentionally has no scalar score that could average
one such failure away.

Schema 32 currently exposes 26 report metrics: 11 fail-closed hard constraints, four primary
quality axes, and 11 supporting diagnostics. The report must present those roles separately. In
particular, all 11 hard constraints must appear in its top constraint view, while unavailable GT
or temporal evidence is shown as not applicable rather than as a zero-valued pass.

The exact source-U sidecar preserves requested coordinates outside [0,1]. Per-eye clamp, fold,
and Jacobian-stretch diagnostics retain that raw demand. Stereo/comfort instead invert the two
maps onto common source-U rows and measure actual `x_right - x_left` only on unique, mutually
visible, in-range, forward-covered correspondences. Output-Jacobian area weights prevent either
an expanded or collapsed source interval from voting like an ordinary pixel. Independently,
image-supported visible-pop checks whether real structure carries that geometry. Experimental
window and interocular-conflict detectors use exact source registration and explicit validity
masks; they remain separate rather than being fused into one artifact score. Disocclusion
bad-fill remains a standalone falsifier because mono input cannot authenticate revealed content.

Metric implementation changes must pass controlled-corruption unit tests and each detector's
authenticated-real/source-content validator before an A/B result is trusted:

```
python -m unittest discover -s tools/sbsbench -p "test_sbs_*.py"
python tools/sbsbench/validate_real_stereo_window_metric.py --suite both --strict --output <window.json>
python tools/sbsbench/validate_disocclusion_topology_real_sources.py --suite both --strict --output <topology.json>
python tools/sbsbench/validate_interocular_phase_chroma.py --output <phase.json>
python tools/sbsbench/validate_interocular_photometric_rivalry.py --run <schema32-run> --output <photometric.json>
python tools/sbsbench/validate_actual_sbs_metric_corruptions.py --run <schema32-run> --max-clips 4 --output <actual-output.json>
```

The unit tests provide precisely controlled corruption truth. The standalone validators apply
deterministic ladders and benign controls to real source content, retain every scenario's response
and abstention, and are permanently unable to auto-promote labels. The stereo-window and topology
validators additionally enforce the shared authenticated clip-provenance contract across core and
extended suites. Passing synthetic tests alone is insufficient: real-image content-dependent
support floors and false positives are exactly what the standalone validators are intended to
expose. A validator pass remains necessary rather than sufficient; label qualification still
requires headset-correlated human inspection.
Evidence minima are part of the metric contract, not report decoration: actual binocular tails
need 1,024 mutually visible samples, visible volume and local polarity need 256, and cross-row
shear needs 512 qualified pixels. New experimental detectors enforce their own native-equivalent
support floors and report the counts beside every value. Texture integrity needs at least 0.1%
independently textured support and
vertical alignment at least 2% of overlapping texture tiles. These are conservative screening
floors, not proof of statistical independence; label qualification will additionally require
spatial effective-sample counts and confidence intervals. Lower support is `n/a`, not perfect zero
or a missing-metric failure.

Exact mapped-source coverage, texture integrity, binocular support, camera symmetry, and vertical
alignment remain conformance checks. Coverage uses a strict 4/255 luma residual internally and
must retain at least 99% of supported interior content. These checks detect a renderer or
color-contract failure, but cannot prove that the chosen disparity looks good. The 11 hard rows
are the positive and negative signed `x_right - x_left` p99.9 disparity tails; binocular support;
high-near warp polarity; camera symmetry; P99 vertical mismatch; global and worst-patch source
coverage; global and worst-patch texture integrity; and, on authenticated GT clips, prediction-to-
GT polarity. Only the disparity tails are perceptual-risk candidates; the rest prove renderer or
depth-sign contracts.
The +/-3% disparity limit is an experimental image-relative heuristic, not an angular comfort
guarantee; headset FOV and user calibration are required before it can become a qualified label.
`static_jitter_p95` is the validated stability-axis metric. It excludes every source pixel that
moved, expands that exclusion horizontally by the allowed disparity radius, and measures the
worse eye's p95 source-conditioned change only on the remaining static support. The signed mono
source change is subtracted before taking the absolute residual, so a reproduced exposure/noise
change cancels but an equal-and-opposite eye change is penalized twice. Scene cuts/camera moves
with less than 10% support do not vote. Its evidence card shows the evaluated mask, each run's
temporal change and a signed red/blue treatment delta.

`flow_temporal_p95` extends stability inspection to moving content only when authenticated dataset
flow exists. The previous output is warped into the current frame and the matched mono-source flow
change is subtracted with its sign intact before the absolute residual is pooled over
photometrically reliable support. The former classical
tile-flow fallback is no longer part of the compact evaluator because it is not reliable on
articulated motion or occlusion; SEA-RAFT remains a separately versioned optional diagnostic.
Every sequence records expected, source-available, evidence-qualified, and actually measured
transition counts. An evidence-qualified middle transition that silently loses its metric fails
closed; the first frame and measured low-support scene cuts remain legitimate `n/a` cases.

On GT clips, `depth_gt_affine_nrmse_pct` robustly positive-affine-aligns predicted relative disparity to
ground-truth inverse depth with IRLS (negative scale is rejected; flat GT is shift-only), requires
at least 5% valid support, and `depth_gt_edge_f1` uses a strict one-pixel boundary tolerance so a
coarse match cannot hide a thin-edge regression. Both are primary depth-axis metrics. Non-GT
clips are reported as `n/a`.

`depth_gt_lag_f1_p95` detects a prediction that matches the previous GT boundary better than the
current one. The former GT ghost-edge percentage was removed from the compact policy: on clean
extended clips it was dominated by the monocular model's ordinary edge disagreement and correlated
poorly with actual one-frame lag. Its standalone helper remains available for research falsifiers.

True-stereo references remain valuable dataset evidence, but global similarity, permissive patch
matching, SIoU, and camera-style fitting are intentionally absent from the compact policy. Those
methods can reward a copied input eye, hide wrong correspondence, or optimize another renderer's
look instead of Apollo's enhanced-pop objective. Reference stereo is retained for separately
versioned offline correspondence oracles, not as an automatic camera-style target.

`rescore_run.py` refreshes a comparison-only run directly from its preserved source/depth/SBS
artifacts after metric-code changes. It refuses committed-baseline verdicts, updates the metric
contract hash and writes atomically; use `run_eval.py` for any committed gate.

Each clip directory carries a `meta.json` with a name and description. It may also record
`content_type`, provenance/license fields, the exact extraction window, and `source_artifacts`
found by visual inspection. Reports show the latter beside an original frame so baked bloom,
outlines, rain or generative inconsistency are not misidentified as warp regressions. The clip
identity hash covers source and GT pixels plus scoring semantics, but excludes these human-readable
annotations.

## Metrics - spatial (per frame)

### Style and perceptual candidates

| metric | meaning | direction |
|--------|---------|-----------|
| `exact_visible_pop_spread_pct` | source-structure-supported exact disparity spread; abstains without horizontal evidence | higher = more visible stereo volume |
| `exact_positive_disparity_pct` / `exact_negative_disparity_pct` | output-Jacobian-weighted p99.9 tails of actual mutually visible `x_right - x_left` disparity | each <=3% under the current experimental hard limit |
| `exact_over_3pct_area_pct` | mutually visible rendered area outside that current limit | lower = less over-limit burden |
| `exact_mapping_stretch_pct` / `exact_mapping_fold_pct` | low-Jacobian repeated columns and reversed coordinate steps | lower = less stretch/fold |
| `warp_cross_row_shear_severity_pct` | unsupported horizontal displacement change between adjacent rows, excluding real source boundaries and other topology failures | lower = less scanline-like tearing; diagnostic pending qualification |
| `experimental_stereo_window_crossed_burden_pct` | contrast/frequency/orientation-weighted crossed disparity that is actually cut by a lateral stereo window | lower = less perceptible window conflict; experimental |
| `interocular_phase_orientation_burden_pct` | coherent exact-source-registered equal-detail phase/orientation disagreement | lower = less binocular structural conflict; experimental |
| `interocular_exposure_rivalry_burden_pct` | coherent source-relative linear-light exposure disagreement between eyes; shared binocular transforms cancel | lower = less binocular exposure rivalry; experimental |
| `interocular_color_gain_rivalry_burden_pct` | coherent source-relative opponent-colour disagreement from unilateral white balance, RGB gain, or hue changes | lower = less binocular colour rivalry; experimental |

The experimental disocclusion bad-fill detector remains available only in its standalone
corruption validator. A mono source cannot authenticate the newly revealed background: the score
can punish plausible inpainting or reward a copied/smeared foreground. It is therefore excluded
from the compact manifest, decisions, reports, and model labels until right-eye ground truth can
validate it.

Visible volume is a style descriptor. The perceptual risk entries above remain experimental and
are excluded from model-label export until their corruption and headset qualification is complete.
The former p99.5 local-relief score was removed: a single disparity spike could improve it without
improving a viewer's overall stereo experience.
The three interocular axes are intentionally separate: phase/orientation, exposure, and colour-gain
rivalry are perceptually different and must not cancel. Shared binocular photometric transforms
cancel, while unilateral global or localized changes remain evidence. Folded, clamped, disjoint, flat, or
unsupported correspondences abstain. They remain diagnostic until real one-eye-fault and headset
qualification passes. Former blur, ringing, double-edge, jagged, missing-edge, color-fringe,
median-normalized chroma, low-frequency luma-rivalry, detail-energy-rivalry, and raw
depth-edge-offset probes were removed
after expanded real-image ladders found weak sensitivity and/or benign-control false positives.
They are absent from implementation, the compact manifest, decisions, reports, and training
labels; rejected detectors are not kept as dormant alternatives that can silently return.

`exact_forward_coverage_pct` is still emitted as internal forward-warp context. Stronger stereo
legitimately exposes more background, so it is deliberately absent from the decision vector and
HTML metric axes.

The cross-row shear detector complements, rather than duplicates, stretch/fold: those inspect the
horizontal Jacobian inside each row, while shear detects a row that jumps sideways even though
every row remains individually monotonic. It is normalized through image coordinates, excludes
aspect-fit bars/clamps/folds, and suppresses changes supported by a real horizontal source
boundary. Its current report role is diagnostic because the first real qualification pair is
strong but not broad enough to set a universal training-label threshold.

### Exact renderer conformance

| metric | meaning | direction |
|--------|---------|-----------|
| `exact_binocular_support_pct` | common rendered area backed by unique source samples in both eyes, limited by the smaller eye Jacobian | >=80% hard evidence floor; prevents comfort-tail bypass by collapsed overlap |
| `source_coverage_pct` | supported interior pixels within 4/255 luma of the exact shader-selected source sample | >=99% hard contract |
| `image_integrity_pct` | retention of exact mapped-source texture without collapse or overshoot | >=80% hard contract on supported texture |
| `source_coverage_worst_patch_bad_pct` | worst resolution-scaled local patch fraction that does not reproduce the exact shader-selected source sample | <=20% hard localized-missing-content contract |
| `image_integrity_worst_patch_bad_pct` | worst resolution-scaled textured patch fraction with collapsed, amplified, or directionally corrupted gradients | <=25% hard localized-texture-damage contract |
| `exact_symmetry_residual_p95_pct` | P95 common-camera residual `abs((x_left+x_right)/2 - x_unwarped)` on unique mutual support | <=0.1% hard symmetric-camera contract |
| `exact_polarity_ok` / `exact_local_polarity_component_pct` | warp ordering relative to the processed depth that drove it | hard global sign audit / local diagnostic; conformance only |
| `vmisalign_p99_pct` | localized texture-supported P99 vertical offset as percent eye height | <=0.1% hard contract |

Conformance metrics catch implementation, color, and transport bugs. They are not perceptual
quality labels because the renderer can reproduce a geometrically poor source coordinate exactly.
Raw luma residual, clamp amount, and foreground-leak/bad-fill subtype maps remain available inside
their standalone falsifiers for diagnosis, but are intentionally absent from canonical aggregates,
reports, gates, and frame-label records. Exact coverage/integrity, mapping stretch/fold, and
binocular-support measurements are the stronger non-redundant policy axes.

## Metrics - temporal evaluation (`--seq`)

| metric | meaning | direction |
|--------|---------|-----------|
| `static_jitter_p95` | worse-eye signed-source-conditioned output change on source-static support after disparity-radius motion exclusion | lower = steadier static content |
| `flow_temporal_p95` | flow-compensated SBS residual after subtracting the registered signed mono-source change | lower = steadier moving content |
| `depth_gt_lag_f1_p95` | previous-frame GT boundary advantage over current GT | lower = less stale depth |

Temporal metrics require a multi-frame clip and measured support. They remain evaluation-only:
the DA-V2 augmentation policy consumes one image, so only registered, source-conditioned temporal
evidence is retained and none of it becomes a single-frame model label.

## Metrics - authenticated ground-truth depth

| metric | meaning | direction |
|--------|---------|-----------|
| `depth_gt_affine_nrmse_pct` | robust positive-affine aligned RMSE normalized by robust GT range, in percent | lower = better global relative depth |
| `depth_gt_edge_f1` | strict boundary F1 with one-pixel positional tolerance | higher = better boundaries |
| `depth_gt_polarity_ok` | explicit sign of prediction-to-GT affine fit | must remain 100% |

Per-pixel bad-disparity rates remain optional raw evidence for dataset inspection, but are absent
from the compact policy and report: after affine alignment they strongly duplicate affine NRMSE and
their one-pixel threshold is not comparable across arbitrary depth/disparity resolutions.

## Independent stereo oracles under qualification

Learned image models are optional offline diagnostics. They do not enter the compact manifest,
gates, or training labels until they pass the same corruption, benign-transform, real-clip, and
headset checks as deterministic metrics:

- the optional [NVIDIA FLIP appearance oracle](NVIDIA_FLIP_APPEARANCE_ORACLE.md) compares each
  final eye to its exact regenerated source sample and reports worst-eye perceptual tails; it
  abstains on HDR previews until raw linear/HDR evidence and display calibration are available;
- the optional [RAFT-Stereo oracle](RAFT_STEREO_ORACLE.md) estimates dense correspondence with
  texture and left/right-consistency masks, then reports signed residual and Middlebury-style bad
  pixels against Apollo's exact intended disparity where that comparison is valid;
- the optional [SEA-RAFT temporal oracle](SEA_RAFT_TEMPORAL_ORACLE.md), which uses learned
  uncertainty plus bidirectional flow to localize output-only edge ghosts, flicker, and static
  jitter while excluding cuts, occlusions, and legitimate source change;
- Apple's iSQoE can be run as an explicitly rejected, optional headset-preference diagnostic.
  Apollo's controlled ladders showed that its holistic score can reward repeated columns, blur,
  and missing content, so it never enters gates, conclusions, or training labels. The report binds
  every measured result to the official checkpoint ID, URL, SHA-256, and checkout revision; HDR
  preview PNGs abstain before the model or its dependencies are loaded.

No oracle is promoted merely because it exists or moves on one clip. Controlled double-edge,
jaggedness, missing-structure, polarity, fold, clamp, disparity, shear, benign transforms, and
manual image/headset confirmation remain the qualification evidence.
