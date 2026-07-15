# DA-V2 scene-aware stereo experiment

The authoritative design and approval contract is
[`docs/sbs-artistic-policy-plan.md`](../../docs/sbs-artistic-policy-plan.md).

This directory implements an opt-in, global-only experiment that reuses frozen DA-V2 Small
features. Apollo can consume the optional TensorRT output, but no learned checkpoint is enabled
in production until it passes the complete approval sequence.

## Model contract

```text
pixel_values -> predicted_depth, artistic_global
artistic_global = [safe_scale_ceiling, safe_ceiling_confidence]
```

An untrained head returns `[1.0, 0.02]`, preserving the current symmetric warp. The first channel
is a learned scene-safety cap, not an authored-camera target and not a style selection. At a shot
boundary the runtime forms `safe_cap = confidence >= 0.5 ? clamp(ceiling, 1.0, 1.5) : 1.0`.
The second channel is trained as the hard actionable `0/1` probability used by that threshold.
The selector's continuous safety-margin score is stored separately as
`safety_margin_reliability` and only weights ceiling/render supervision.
Clean uses `1.0`, balanced uses `1 + 0.5 * (safe_cap - 1)`, and immersive uses the complete safe
cap. Learned
zero-plane control was rejected on development because it regressed convergence MAE; Apollo keeps
its existing zero plane. There is no dense residual or inpainting output in this milestone.

The depth decoder is frozen and policy gradients cannot change it. Re-exporting the shared graph
is tolerance-neutral rather than bit-identical: approval requires normalized depth drift no larger
than one 10-bit code on average and two codes at p99 against the shipping ONNX. The global head
pools class and mean patch tokens from all four DINO stages and adds detached, scale-invariant depth and DPT
decoder risk statistics. This gives the small head spatial-complexity evidence without permitting
policy gradients to change depth or introducing a per-pixel policy output.

## Data pipeline

Prepare true left/right authored stereo with a whole-production split:

```powershell
python tools\depth_models\prepare_stereo_movie_training.py `
  --video D:\StereoMovies\example-full-sbs.mkv `
  --output E:\ApolloDev\artistic-policy\example-film-v2 `
  --name example-film --domain example_film --film-id example_film `
  --layout side-by-side --eye-order first-left --split training `
  --eye-aspect-ratio 1.7777777778 `
  --global-policy-weight 1.0 `
  --license "user-owned noncommercial research copy"
```

For half-SBS web video, `--eye-aspect-ratio` is mandatory: it restores each horizontally
squeezed stored eye to its intended display aspect. Omit it only when each stored eye already has
the correct shape, such as a full-SBS source.
Use `--global-policy-weight` to keep supplementary domains from overwhelming cinematic
productions; the weight is copied into every shot manifest and the dataset manifest.

The curated open-source catalog is `artistic_stereo_sources.json`. Fetch automatic sources into
external storage and write hash/probe/contact-sheet evidence with:

```powershell
& E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\depth_models\fetch_artistic_stereo_sources.py `
  --all-auto --output E:\ApolloDev\artistic-policy\authored-sources-v1 `
  --ffmpeg E:\ApolloDev\tools\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe `
  --ffprobe E:\ApolloDev\tools\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffprobe.exe
```

`source_record.json` is the admission record. A global-policy source cannot retain an unverified
eye order. Inspect the generated contact sheet and disparity polarity before changing a candidate
to `global_policy`; separate-eye derived files also record both input hashes.

Freeze the active complete-production split before label generation:

```powershell
python tools\depth_models\audit_artistic_dataset_splits.py `
  --dataset-manifest E:\ApolloDev\artistic-policy\film-a\dataset_manifest.json `
  --dataset-manifest E:\ApolloDev\artistic-policy\film-b\dataset_manifest.json `
  --output E:\ApolloDev\artistic-policy\active-artistic-split-v1.json
```

The audit cross-checks catalog split and weight, rejects duplicate productions, hashes each
prepared manifest, and requires training, development, and at least two sealed test productions.

Use `development` for epoch selection and `test` for at least two separately sourced sealed
productions. Never split shots from one production across these roles.

Generate production depth plus exact Apollo baseline disparity:

```powershell
python tools\depth_models\generate_artistic_depth_run.py `
  --suite E:\ApolloDev\artistic-policy\example-film-v2 `
  --output E:\ApolloDev\artistic-policy\example-film-depth-v2

python tools\depth_models\prepare_artistic_policy_labels.py `
  --run E:\ApolloDev\artistic-policy\example-film-depth-v2 `
  --clips E:\ApolloDev\artistic-policy\example-film-v2 `
  --output E:\ApolloDev\artistic-policy\example-film-labels-v2
```

This first label step is only correspondence evidence. Authored stereo camera strength is not the
product target: many otherwise useful films deliberately use less disparity than Apollo's desired
headset style. Such sources remain useful for polarity, geometry, occlusion and artifact evidence.

Each schema-8 single-geometry bundle is selected from a complete exact-scale render grid with
`select_render_feasible_labels.py`. Starting at identity, the selector keeps only the connected
candidate frontier that has exact pop evidence and passes hard comfort/integrity gates plus
protected warp and stability metrics. The highest connected candidate is `safe_scale_ceiling`.
Authored-right-eye PSNR is
diagnostic only. The bundle retains the complete candidate evidence, exact unclamped disparity,
clamp-aware render targets, and deterministic preset targets:

- clean: `min(1.0, safe ceiling)`;
- balanced: the connected rendered point nearest `1 + 0.5 * (safe ceiling - 1)`; and
- immersive: `min(1.5, safe ceiling)`.

Schema 8 is deliberately not trainable. The global head cannot observe the destination eye
raster, so run the same complete grid at every exact intended deployment geometry, then collapse
those bundles with `merge_artistic_geometry_labels.py`. Its required geometry manifest is a
versioned exact allow-list of source dimensions, production model-input dimensions, eye raster,
resolved `depth_short_side`/`depth_max_aspect`, content scale, disparity raster, and color mode.
The model-input dimensions are recomputed with the same ultra-wide/portrait aspect cap as
production. Every RGB must cover every tuple in its matching
source/model/color group. The merge intersects identity-connected safe frontiers, uses their
common safe maximum, takes the minimum reliability, and emits exactly one schema-9 row per unique
RGB; duplicate RGBs with ambiguous temporal context fail closed.

The trainer accepts schema 9 only. It supervises the common ceiling on every shot, including
identity shots, so a confidence false positive cannot expose an unconstrained multiplier. For the
rendered disparity loss it retains every geometry's exact unclamped disparity sample, applies that
geometry's HLSL clamp to both predicted and target fields, then uses the worst field and gradient
loss across geometries. The scalar
mean-absolute disparity remains diagnostic only. Frontier bounds add an independent safety loss.

Every candidate run must use full-cadence state processing plus `--output-gt-right-only`; only the
authored frame identities are scored. It must also pass `--no-artistic-policy`; the selector
rejects learned-policy-on-policy supervision, stale schema-7 source contracts, non-production
literal warps, reused depth, or a source-depth configuration that differs from the render grid.
Warp and stability feasibility uses each metric's directional worst frame recorded by `run_eval`,
not the clip average, so one severe artifact cannot be diluted by otherwise-clean frames.
The control is the scale-1 candidate itself and must be rendered with the explicit
`--artistic-scale-override 1.0`; override `0` is the legacy clamp path and is not policy identity.
A typical final selection is:

```powershell
python tools\depth_models\select_render_feasible_labels.py `
  --source-labels E:\ApolloDev\artistic-policy\example-film-labels-v2\labels.jsonl `
  --control E:\ApolloDev\artistic-policy\example-film-scale100\results.json `
  --candidate 0.90=E:\ApolloDev\artistic-policy\example-film-scale090\results.json `
  --candidate 1.00=E:\ApolloDev\artistic-policy\example-film-scale100\results.json `
  --candidate 1.10=E:\ApolloDev\artistic-policy\example-film-scale110\results.json `
  --candidate 1.20=E:\ApolloDev\artistic-policy\example-film-scale120\results.json `
  --candidate 1.30=E:\ApolloDev\artistic-policy\example-film-scale130\results.json `
  --candidate 1.40=E:\ApolloDev\artistic-policy\example-film-scale140\results.json `
  --candidate 1.50=E:\ApolloDev\artistic-policy\example-film-scale150\results.json `
  --output E:\ApolloDev\artistic-policy\example-film-safe-frontier-v3
```

Repeat that selection for every destination geometry. Then create one reviewed manifest and use
that same manifest for training, development, and sealed-test merges:

```powershell
python tools\depth_models\merge_artistic_geometry_labels.py `
  --geometry-labels E:\ApolloDev\artistic-policy\example-film-1080p-v3\labels.jsonl `
  --geometry-labels E:\ApolloDev\artistic-policy\example-film-4k-v3\labels.jsonl `
  --deployment-geometry-manifest E:\ApolloDev\artistic-policy\deployment-geometries-v1.json `
  --output E:\ApolloDev\artistic-policy\example-film-multigeometry-v4
```

The merger requires at least two distinct geometry variants per RGB. An unlisted runtime tuple is
not extrapolated: the exported policy contract requires Apollo to fail closed to scale 1.0. Every
merged training, development, and sealed-test bundle must exercise the complete manifest; prepare
sources at the manifest's canonical source sizes rather than adding untested source-size entries.

The upward grid must reach the model limit at `1.50`, with no adjacent step larger than `0.10`.
The connected frontier still stops at the first unsafe render; samples above that failure prove
that a reported ceiling was not merely the highest scale the evaluator happened to render.

All mutable inputs and code contracts are hash checked. The evaluator's exact
`metric_sha256` is recovered from the verified render-grid control and recomputed from the
recorded `sbsbench.py`, thresholds, and `run_eval.py` gating semantics before it is admitted to a dataset. Every label bundle in
one training run must carry that same identity; it is then copied into the checkpoint and export.

## Train, export, and screen

Use the external CUDA/PyTorch environment documented in repository history. Training accepts only
`training` and `development` labels. The sealed-test label files are not opened until the one-time
evaluation command:

```powershell
& E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\depth_models\train_artistic_policy.py `
  --labels E:\ApolloDev\artistic-policy\big-buck-bunny-safe-frontier-v3\labels.jsonl `
    E:\ApolloDev\artistic-policy\nasa-evolution-moon-safe-frontier-v3\labels.jsonl `
    E:\ApolloDev\artistic-policy\nasa-vlbi-safe-frontier-v3\labels.jsonl `
    E:\ApolloDev\artistic-policy\nasa-lidar-safe-frontier-v3\labels.jsonl `
  --split-manifest E:\ApolloDev\artistic-policy\active-artistic-split-v1.json `
  --depth-anything-root E:\ApolloDev\third_party\Depth-Anything-V2 `
  --depth-weights E:\ApolloDev\models\depth-anything-v2-small\depth_anything_v2_vits.pth `
  --output E:\ApolloDev\artistic-policy\global-v1

& E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\depth_models\evaluate_artistic_policy.py `
  --labels E:\ApolloDev\artistic-policy\elephants-dream-safe-frontier-v3\labels.jsonl `
    E:\ApolloDev\artistic-policy\nasa-goddard-safe-frontier-v3\labels.jsonl `
  --split-manifest E:\ApolloDev\artistic-policy\active-artistic-split-v1.json `
  --depth-anything-root E:\ApolloDev\third_party\Depth-Anything-V2 `
  --depth-weights E:\ApolloDev\models\depth-anything-v2-small\depth_anything_v2_vits.pth `
  --policy E:\ApolloDev\artistic-policy\global-v1\artistic_policy_best.pt `
  --output E:\ApolloDev\artistic-policy\global-v1\test-report

& E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\depth_models\export_artistic_policy.py `
  --depth-anything-root E:\ApolloDev\third_party\Depth-Anything-V2 `
  --depth-weights E:\ApolloDev\models\depth-anything-v2-small\depth_anything_v2_vits.pth `
  --policy E:\ApolloDev\artistic-policy\global-v1\artistic_policy_best.pt `
  --evaluation E:\ApolloDev\artistic-policy\global-v1\test-report\evaluation.json `
  --output E:\ApolloDev\artistic-policy\global-v1\depth_anything_v2_artistic.onnx

& E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\depth_models\validate_artistic_depth_neutrality.py `
  --reference cmake-build-relwithdebinfo\assets\depth_anything_v2_fp16.onnx `
  --candidate E:\ApolloDev\artistic-policy\global-v1\depth_anything_v2_artistic.onnx `
  --depth-short-side 432 --depth-max-aspect 4.0 `
  --image tools\sbsbench\clips\aigen_cogvideox_rain\frame_00001.jpg `
  --image tools\sbsbench\clips\anime_morevna_closeup\frame_00001.jpg `
  --image tools\sbsbench\clips\c339\frame_00001.jpg `
  --image tools\sbsbench\clips\c525\frame_00001.jpg `
  --image tools\sbsbench\clips\c647\frame_00001.jpg `
  --image tools\sbsbench\clips\c747\frame_00001.jpg `
  --image tools\sbsbench\clips\c841\frame_00001.jpg `
  --image tools\sbsbench\clips\fast_motion\frame_00001.jpg `
  --image tools\sbsbench\clips\flat_page\frame_00001.jpg `
  --image tools\sbsbench\clips\flat_transition\frame_00001.jpg `
  --image tools\sbsbench\clips\scene_cut\frame_00001.jpg `
  --output E:\ApolloDev\artistic-policy\global-v1\depth-neutrality.json

# Prepare two pinned copies of tools\sbsbench\bench.conf. They must differ only in the explicit
# sbs_3d_artistic_style line: immersive in one and balanced in the other.
python tools\sbsbench\run_eval.py --suite core `
  --conf E:\ApolloDev\artistic-policy\artistic-immersive.conf `
  --label artistic-policy-immersive-core `
  --extra --model depth_anything_v2_artistic
python tools\sbsbench\run_eval.py --suite extended `
  --conf E:\ApolloDev\artistic-policy\artistic-immersive.conf `
  --label artistic-policy-immersive-extended `
  --extra --model depth_anything_v2_artistic
python tools\sbsbench\run_eval.py --suite core `
  --conf E:\ApolloDev\artistic-policy\artistic-balanced.conf `
  --label artistic-policy-balanced-core `
  --extra --model depth_anything_v2_artistic
python tools\sbsbench\run_eval.py --suite extended `
  --conf E:\ApolloDev\artistic-policy\artistic-balanced.conf `
  --label artistic-policy-balanced-extended `
  --extra --model depth_anything_v2_artistic

# First create a non-approved deployment manifest for the live headset-review stage.
python tools\depth_models\promote_artistic_policy.py `
  --onnx E:\ApolloDev\artistic-policy\global-v1\depth_anything_v2_artistic.onnx `
  --metadata E:\ApolloDev\artistic-policy\global-v1\depth_anything_v2_artistic.json `
  --checkpoint E:\ApolloDev\artistic-policy\global-v1\artistic_policy_best.pt `
  --evaluation E:\ApolloDev\artistic-policy\global-v1\test-report\evaluation.json `
  --reference-depth-onnx cmake-build-relwithdebinfo\assets\depth_anything_v2_fp16.onnx `
  --neutrality-report E:\ApolloDev\artistic-policy\global-v1\depth-neutrality.json `
  --core-results cmake-build-relwithdebinfo\sbs_eval\artistic-policy-immersive-core\results.json `
  --extended-results cmake-build-relwithdebinfo\sbs_eval\artistic-policy-immersive-extended\results.json `
  --balanced-core-results cmake-build-relwithdebinfo\sbs_eval\artistic-policy-balanced-core\results.json `
  --balanced-extended-results cmake-build-relwithdebinfo\sbs_eval\artistic-policy-balanced-extended\results.json `
  --stage-headset-review `
  --output E:\ApolloDev\artistic-policy\global-v1\depth_anything_v2_artistic.deployment.json
```

Copy that staged manifest beside the exact deployed ONNX/engine under the same model stem, set
`sbs_3d_artistic_live_review = true`, restart Apollo, and review immersive style at one exact
deployment-geometry tuple listed in the sidecar. After the review, rerun the same promotion command
and replace its `--stage-headset-review` line with:

```powershell
  --approve-headset-review --headset-reviewer dcatcher `
  --headset-device "Galaxy XR" --headset-resolution 3840x1080 `
  --headset-refresh-hz 90 --headset-color-mode sdr-srgb-8bit `
  --headset-geometry-index 0 `
  --headset-notes "Immersive style passed comfort, cuts, and visible-artifact review" `
```

That second invocation atomically replaces the staged file with an approved production manifest.
Clear the explicit `sbs_3d_artistic_live_review` override for normal production use.

Training balances ceiling-action class, domain and clip; samples adjacent same-shot pairs; uses
exact post-clamp rendered-disparity and gradient supervision from the full raw fields; learns an
actionable probability from hard `0/1` targets; uses the separate safety-margin reliability only
for loss weighting;
and constrains every prediction to the labelled safe frontier. Checkpoint selection uses
the earliest labelled frame of
each complete shot, matching the runtime latch, and prioritizes effective ceiling error, balanced
action error, raw actionable ceiling error, calibration and within-shot variation. Evaluation
separately guards identity false actions and rejects a sealed checkpoint if any shot-latched
prediction exceeds its labelled safe ceiling by more than 0.05 scale, or if the film-balanced
mean positive overshoot exceeds 0.01 scale. The report also records the film-balanced overshoot
rate. Passing this label screen is necessary but insufficient: the
checkpoint must then pass depth neutrality, all four immersive/balanced core/extended SBS render
gates, and live headset review before production activation.

The schema-11 evaluation records the exact checkpoint, active-split, metric, geometry allow-list,
and canonical label-
fitter identities and one-sided unsafe-ceiling evidence. The v2 sealed-test approval contract
requires that artifact, hashes it, and refuses development results,
rejected decisions, mismatched identities, or a different set of sealed productions. This binds
the label-screen approval to the checkpoint being exported. Export alone is not deployable.

`promote_artistic_policy.py` is the only manifest-authoring step. Both invocations authenticate the
exact ONNX, schema-4 sidecar, checkpoint and sealed evaluation bytes; the fixed-limit schema-4
neutrality report and every canonical core first-frame image; four fresh, unrescored, passing
immersive/balanced core/extended render results; and current metric/warp hashes. The first invocation
uses `--stage-headset-review` and writes a non-approved schema-1
`apollo-artistic-policy-deployment-v1` manifest that Apollo accepts only under the explicit
`sbs_3d_artistic_live_review` authorization. The second invocation replaces it with an approved
production manifest after binding an explicit named immersive headset review.

Every render result must prove that the optional output was actually consumed and record the exact
ONNX and sidecar hashes. The headset review must select an exact tuple index from the sidecar's
canonical deployment-geometry allow-list; its stated full-SBS resolution and color mode must match
that tuple, proving the optional policy was eligible rather than silently falling back to identity.
Every clip geometry in all four fresh gates must be allow-listed, and their union must exercise
every tuple in that allow-list, including the headset-reviewed tuple.
A promotion also recomputes the metric and warp identities directly from the canonical source lists
in `cmake/prep/artistic_warp_contract.cmake`, and requires harness schema 24 plus eval schema 29.
It validates the same policy/feature, tensor, output-semantics, bounds, and runtime contracts that
Apollo consumes, so a manifest cannot be successfully promoted and then silently rejected by the
runtime for a stale sidecar contract.
A path string without the matching content hash is never approval evidence.

Export writes a schema-4 JSON sidecar beside the ONNX. It includes `deployed_model`, the source
`base_depth_model`, exact ONNX and evaluation hashes, the sealed-test approval contract, output
semantics, `metric_sha256`, the exact deployment-geometry allow-list, and the frozen policy
baseline. Export first hashes the supplied DA-V2
weights and refuses to build when they differ from the checkpoint's `depth_weights_sha256`. Apollo also
requires the TensorRT engine's generated source marker and compares every policy-affecting resolved
setting before consuming the optional head. The marker binds both the ONNX and serialized-engine
hashes to the TensorRT recipe. Production activation additionally requires the deployment manifest
beside the model. A mismatch disables only the policy; DA-V2 depth still runs normally.
