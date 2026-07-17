# DA-V2 scene-aware stereo experiment

The authoritative design and approval contract is
[`docs/sbs-artistic-policy-plan.md`](../../docs/sbs-artistic-policy-plan.md).

Status: the scalar checkpoint and the later monotone 26-bin safety-frontier experiment were both
rejected. The ordinal implementation is preserved for reproducibility, but its first-failure
frontier is not valid training truth: rendered halo/stretch evidence can pass again after an
intermediate metric failure, and near-identical DA-V2 evidence produced incompatible targets.
Future work should collect headset preference/acceptability outcomes for explicit absolute pop
settings. Do not regenerate or train the ordinal corpus for production selection.

## Simple-controller falsification screen

Before paying for a complete learned-policy label grid, use
`evaluate_simple_artistic_controller.py` to test whether a shallow rule over the eight existing
scale-invariant DA-V2 depth/DPT statistics can safely identify scale-1.1 shots. The fit uses the
completed training scale-1.0/1.1 pair; evaluation uses only the first monocular frame from the
current held-out schema-10 development bundle. Candidate SBS images and evaluator metrics are
targets only and are never controller inputs.

```powershell
& E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\depth_models\evaluate_simple_artistic_controller.py `
  --training-control cmake-build-relwithdebinfo\sbs_eval\artv1-training-g1280-s10\results.json `
  --training-candidate cmake-build-relwithdebinfo\sbs_eval\artv1-training-g1280-s11\results.json `
  --development-labels E:\ApolloDev\artistic-policy\nasa-lidar-multigeometry-schema9-v1\labels.jsonl `
  --depth-anything-root E:\ApolloDev\third_party\Depth-Anything-V2 `
  --depth-weights E:\ApolloDev\models\depth-anything-v2-small\depth_anything_v2_vits.pth `
  --output E:\ApolloDev\artistic-policy\simple-controller-dav2-stats-v1 `
  --overwrite
```

This is a binary falsification screen, not a production safe-ceiling controller or sealed-test
approval. The statistics are internal to the shared DA-V2 graph today; deploying the rule would
still require baking it into an `artistic_global` output or adding an authenticated statistics
output. Identity is the safety baseline, so a controller that avoids false positives by selecting
no safe shots has demonstrated no useful gain.

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

### Rejected ordinal V2 experiment (historical)

This experiment was deliberately separate from the scalar contract above. It predicted 26
monotone connected-safety probabilities for scales `1.00..1.50` in `0.02` steps. Its rendered
metric grid is retained as diagnostic evidence, not as a perceptual or safety source of truth.
The archived model experiment was target-only: train, development, and the intended sealed test
contained only authenticated target images. Nearby context frames and causal-controller replay were
deliberately excluded while single-image safety prediction was the priority. Safe-through-`1.50`
examples are right-censored and identity hard failures are explicit left-censored examples.
The 26 outputs are calibrated point probabilities unless a caller explicitly supplies a separately
validated lower-bound construction. Identity below threshold means abstention, not an implicit
safe fallback. Report triggers stay diagnostic; ordinal absolute limits use only
`ordinal_hard_min`/`ordinal_hard_max`.

The isolated foundation lives in:

- `artistic_policy_ordinal_contract.py` for the canonical interval/censoring evidence;
- `artistic_policy_ordinal_model.py` for the frozen-DA-V2 monotone head;
- `artistic_policy_ordinal_loss.py` for interval likelihood and known-bin asymmetric calibration;
- `select_ordinal_render_frontiers.py` for per-geometry connected-frontier selection;
- `merge_ordinal_geometry_frontiers.py` for the exact two-geometry safety intersection.

`run_eval.py --publish-selected-frame-gates` is required for this corpus's sparse ordinal safety
runs and is bound to the authenticated `label_frames.json`; a caller cannot choose an arbitrary
sparse output list. It publishes contract `apollo-target-frame-gate-evidence-v2` with output
selection `authenticated-label-targets-only-v2`. The existing mutually
exclusive `--publish-frame-gates` mode remains the all-source-frame diagnostic path; it is not used
by this corpus's sparse safety renderer.
Do not train or deploy this model from the old unauthenticated `--output-label-frames` render grid.
The shipping scalar ONNX/runtime remained unchanged; the ordinal experiment was rejected before
runtime promotion.
Temporal control and adjacent-frame validation are deferred. They remain required before eventual
runtime promotion, but do not participate in dataset cardinality, development scoring, checkpoint
selection, or the present report.
The trainer also requires a reviewed positive `--minimum-development-pop-gain-pct` plus a
nonempty `--minimum-development-pop-gain-rationale`; both are frozen before predictions. The gate
applies independently to native SDR, the HDR aggregate, and every HDR condition. This keeps an
all-abstain zero-gain head from being published merely because it has zero overshoot. Development
output is a checkpoint candidate only and always records `production_policy_accepted=false`.
Checkpoint eligibility also requires zero non-boundary unproven frames and zero measured identity
hard failures in SDR, the HDR aggregate, and every HDR condition; abstention cannot hide an unsafe
identity fallback. The offline evaluator fallback is not a runtime model input. Known-bin ECE is
reported beside NLL and Brier. A safe action beyond the first scale attaining the same maximum pop
is a diagnostic/checkpoint tie-breaker only: it blocks an optimal-utility claim but never rewrites
the causal replay or changes a safety label.

`orchestrate_artistic_ordinal_labels.py` owns the fail-closed implementation. The user-facing,
unattended entry point is `regenerate_artistic_ordinal_corpus.py`; use it instead of manually
invoking or monitoring orchestration subprocesses. It opens only the active split's `training` and
`development` manifests. Each authenticated multiscale safety batch retains sparse visual examples
at `1.00`, `1.30`, and `1.50`; those copies are inspection evidence, not a separate render or a
training row. The safety path uses the exact 26-bin lattice and both deployment geometries. The
published model corpus contains only authenticated targets. After a batch is authenticated and
scored, bulky rendered images/fields are deleted; results, gate evidence, sparse visual evidence,
multiscale provenance, and runtime-scene evidence remain for audit. The `bundle` phase creates one
canonical safety lattice per clip and condition. The `sources` phase publishes matching production
model inputs only for target frames, including authenticated linear scRGB FP16 sources for native
PQ. The catalog fails closed unless target rows and safety labels join exactly one-to-one and no
context row is present.

### Authenticated cache layers

Three independent content-addressed caches accelerate different boundaries. A cache hit never
changes dataset, label, or acceptance semantics; every hit is revalidated against its immutable
identity and payload hashes before use.

1. `E:\ApolloDev\artistic-policy\cache\v1\preprocessing` is populated only by dataset preparers
   that receive `--preprocess-cache`. It reuses expensive source decode/color conversion across
   repeated dataset publications. The ordinal `plan`/`run` command consumes an already prepared
   active split and therefore neither reads nor seeds this raw cache.
2. `E:\ApolloDev\artistic-policy\cache\v1\depth-state` stores the completed production depth state
   before warp prefiltering. On a cold ordinal run, the first deployment geometry performs model
   inference and the second geometry replays the same authenticated state. Exact later workspaces
   can reuse it without loading or invoking the estimator.
3. `E:\ApolloDev\artistic-policy\cache\v1\ordinal-scores` stores compacted, authenticated
   render/metric results. An exact later workspace can materialize a complete safety batch without
   running either the render harness or the 26 scalar scorers.

Source bytes, preprocessing/color contracts, native runtimes, model/config/code identities, and
selected frame identities invalidate the layers that depend on them. A metric or threshold change
invalidates scored results while still permitting an exact depth-state hit. Dataset-only metadata,
shot thresholds, and sparse-label cadence do not invalidate the raw movie/CHUG frame cache. The
frozen plan records the depth and score roots explicitly; changing a root requires a new workspace
and plan digest.

Cache deletion is correctness-neutral: after stopping every preparer and regeneration process,
delete a complete cache root to force a cold rebuild. Do not hand-edit or selectively replace files
inside a hashed entry. Published datasets and completed workspace outputs remain independently
authenticated. There is no automatic eviction, so inspect free space before a large cold run.

### Historical ordinal corpus regeneration

The commands below reproduce the rejected experiment for audit only. They are not an endorsed
training workflow.

First freeze a plan. This validates the current RelWithDebInfo build and writes
`regeneration_request.json` plus `ordinal_orchestration_plan.json`; it does not render or train:

```powershell
$python = 'E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe'
$regen = 'tools\depth_models\regenerate_artistic_ordinal_corpus.py'
$workspace = 'E:\ApolloDev\artistic-policy\ordinal-v2-target-only-v9'

& $python $regen plan `
  --workspace $workspace `
  --active-split E:\ApolloDev\artistic-policy\public-mono-hdr-bootstrap-v3\datasets\active_artistic_split_sdr_native_pq_full_cadence.json `
  --build-dir cmake-build-relwithdebinfo `
  --conf tools\sbsbench\bench.conf `
  --python $python `
  --run-prefix ordv2-target-only-v9 `
  --score-workers 8 `
  --depth-state-cache-root E:\ApolloDev\artistic-policy\cache\v1\depth-state `
  --scored-result-cache-root E:\ApolloDev\artistic-policy\cache\v1\ordinal-scores
```

Do not resume `ordinal-v2-selected-full-v6` or `ordinal-v2-selected-full-v7`: their frozen requests
predate the final regeneration/orchestration and cache-identity contracts. The v9 workspace above
is the last reproducible target-only experiment and remains historical evidence only.

`--score-workers` parallelizes independent exact-scale scorers; every child still uses one metric
worker and the complete 26-scale result grid must succeed before anything is published or deleted.
On the prescribed 24-logical-CPU development host, `8` is the measured throughput knee: the
1280x720 native-PQ smoke batch fell from 211.9 seconds on the former serial-scale path to 50.7
seconds. `12` workers reached 47.6 seconds but adds substantially more contention for only a 6%
gain, so the unattended default is `8`.

Review the printed scope, eligibility, cardinality estimates, and `plan_sha256`. Then acknowledge
that exact immutable plan and let the script run unattended. The orchestrator streams subprocess
output to the terminal and also writes per-step logs under `orchestration_logs`:

```powershell
$planSha = '<paste the plan_sha256 printed by the plan command>'
& $python $regen run `
  --workspace $workspace `
  --accept-plan-sha256 $planSha
```

Interruption, terminal closure, or reboot does not require a new plan. Run the same `run` command
with the same digest; authenticated completed steps are validated and resumed, never rerendered.
Read progress from another terminal without changing state:

```powershell
& $python $regen status --workspace $workspace
& $python $regen status --workspace $workspace --verify-completed
```

`status` reports the lifecycle (`running`, `failed`, `interrupted`, or `completed`), active step,
last failure, and a complete `resume_command`. Its percentage is explicitly unweighted plan-step
progress; safety steps dominate runtime. In a fresh PowerShell after reboot, no old variables are
needed:

```powershell
$workspace = 'E:\ApolloDev\artistic-policy\ordinal-v2-target-only-v9'
$request = Get-Content -Raw (Join-Path $workspace 'regeneration_request.json') |
  ConvertFrom-Json
$regen = 'tools\depth_models\regenerate_artistic_ordinal_corpus.py'
& $request.python $regen run --workspace $workspace `
  --accept-plan-sha256 $request.plan_sha256
```

Only one `run` process may own a workspace. A second invocation fails immediately and `status`
shows the active PID/host instead of allowing two processes to race on render trees or state.

An interrupted subprocess can leave an unrecorded, nonempty partial output. The normal resume fails
closed instead of trusting or overwriting it. After inspecting its log, rerun with
`--repair-partials`; this removes only invalid unrecorded outputs inside the orchestrator's managed
roots and then regenerates them. It never deletes an authenticated completed step:

```powershell
& $python $regen run `
  --workspace $workspace `
  --accept-plan-sha256 $planSha `
  --repair-partials
```

After completion, revalidate the entire corpus before using it. Verification always rereads and
hashes every source byte; a receipt is progress evidence, never permission to substitute a stat-only
check. `--completed-only` is useful after an interruption:

```powershell
& $python $regen verify --workspace $workspace
```

For a one-clip native-PQ smoke, add
`--production chug_native_pq_full_cadence_v3_development --clip-limit 1` to the `plan` command and
use a new workspace. The frozen plan and resulting catalog are explicitly not training-eligible.

Regeneration stops at the authenticated catalog. It never launches training, selects a checkpoint,
or touches the sealed test split. Training is a separate reviewed action using the verified catalog,
as shown below.

The trainer accepts only the exact bundle/source publications named by the training-eligible
catalog. It authenticates the active split, thresholds, metric contract, `sbsbench.py`,
`run_eval.py`, all catalog code identities, and every publication hash before loading media or
caching DA-V2 features. Each sequence carries its own authenticated source frame rate; there is no
trainer-wide 30 fps fallback. The four mono render conditions keep independent safety targets even
when their frozen DA-V2 features are nearly identical. Their paired feature/depth/target comparison
is a calibration-risk diagnostic and never merges or rejects a condition label.

```powershell
$workspace = 'E:\ApolloDev\artistic-policy\ordinal-v2-target-only-v9'
$catalogPath = Join-Path $workspace 'ordinal_frame_label_catalog.json'
$catalog = Get-Content -Raw $catalogPath | ConvertFrom-Json
$frontiers = @($catalog.bundles | ForEach-Object { $_.labels })
$sources = @($catalog.sources | ForEach-Object { $_.labels })
& E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\depth_models\train_artistic_policy_ordinal.py `
  --catalog $catalogPath --frontiers $frontiers --source-rows $sources `
  --split-manifest E:\ApolloDev\artistic-policy\public-mono-hdr-bootstrap-v3\datasets\active_artistic_split_sdr_native_pq_full_cadence.json `
  --depth-anything-root E:\ApolloDev\third_party\Depth-Anything-V2 `
  --depth-weights E:\ApolloDev\models\depth-anything-v2-small\depth_anything_v2_vits.pth `
  --minimum-development-pop-gain-pct 0.10 `
  --minimum-development-pop-gain-rationale '0.10 percentage points is 2.5x the frozen exact_pop_spread_pct absolute floor of 0.04; selected before development predictions.' `
  --output E:\ApolloDev\artistic-policy\ordinal-v2-run-v4

& E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\depth_models\build_artistic_policy_ordinal_report.py `
  --training-output E:\ApolloDev\artistic-policy\ordinal-v2-run-v4 `
  --output E:\ApolloDev\artistic-policy\ordinal-v2-run-v4\report
```

The standalone `plan` output replaces the old ad hoc `--dry-run` workflow: its frozen `estimates`
object is the authoritative cardinality review for that exact split, code, config, and build.
`--production`/`--clip-limit` remain smoke controls and cannot produce a training-eligible catalog
unless the complete active train/development set is selected. The current active split contains
400 supervised train/development targets across native SDR, HDR80/200/480, and native-PQ runtime
conditions. The model corpus contains exactly those 400 rows: 300 training and 100 development,
all tagged `target`, with no nearby context rows. Treat
any unexpected plan estimate as a split or contract change that must be reviewed before accepting
the digest.

For the current full split, expect 259 plan steps: 160 safety batches, 80 label bundles, 18 source
publications, and one catalog publication. That is 400 target source rows, zero context rows, 400
selected output frames, and 20,800 geometry/scale visits for 400
supervised targets. A completely cold cache
requires 80 estimator sequences (the second geometry reuses depth state) and 4,160 scalar scorers.
An exact fully warm scored-result cache requires neither estimator, harness, nor scorer work for the
safety batches. A partial hit lies between those bounds; these are work counts, not a wall-clock
promise. `plan` does not currently estimate cache bytes or elapsed time.

## Data pipeline

Use ordinary monocular movie shots as the primary scale-safety dataset. They match the model's
actual RGB input and do not teach Apollo to imitate the often-conservative stereo volume of an
authored movie. Preserve each shot at full cadence: isolated stills cannot validate temporal
warp/stability regressions. Authored SBS remains useful as an auxiliary diagnostic/validation
source for polarity, correspondence, occlusion, and stereo-reference metrics; it is not required
for a generic source row.

Prepare a monocular production into full-cadence shots with sparse target manifests:

```powershell
python tools\depth_models\prepare_monocular_movie_training.py `
  --video D:\Movies\example-mono.mkv `
  --output E:\ApolloDev\artistic-policy\example-mono-v1 `
  --name example-mono --domain example_mono --production-id example_mono `
  --sample-fps 2 --split training `
  --ffprobe E:\ApolloDev\tools\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffprobe.exe `
  --preprocess-cache E:\ApolloDev\artistic-policy\cache\v1\preprocessing `
  --write-workers 4 `
  --global-policy-weight 1.0 `
  --license "user-owned noncommercial research copy"
```

Every shot keeps all RGB frames so depth/EMA state advances exactly as production. Its strict
`label_frames.json` contains only `schema` and sparse target `frame_ids`. The harness emits
artifacts for each target plus one consecutive neighbor, resets temporal comparisons across the
remaining gaps, and records both the target IDs and exact emitted IDs. This substantially reduces
artifact I/O and metric work without fabricating temporal evidence.
Input color is probed before OpenCV decoding. PQ, HLG, wide-gamut, and unreviewed
high-bit-depth sources are rejected rather than silently clipped into an SDR label. Convert such
material through a versioned HDR-to-SDR preprocessing step first. `--input-color sdr` is only a
recorded override for ambiguous metadata; it cannot override explicit HDR signaling.

`--preprocess-cache` stores the expensive source-bound decode once as a flat immutable raw-frame
object. Rebuilding with different shot thresholds, sparse-label cadence, publication names,
licenses, policy weights, context length, or writer parallelism reuses those decoded/resized PNGs
and only reruns the cheap shot/metadata derivation. The persistent cache contains one PNG per
decoded source frame plus a small signature manifest; the prepared output contains one further copy
of each retained shot frame. While both exist, expect at most two persistent frame payloads (raw
cache plus prepared output), not three; there is deliberately no second cache copy of the finished
dataset. Cache and output roots must be disjoint.

### Pinned REDS/Spring monocular pool

`prepare_public_monocular_training.py` authenticates the archives pinned by
`public_monocular_sources.json`, then publishes six full-cadence image-sequence splits outside the
repository. It uses only REDS Sharp and Spring left RGB as model input: blurred/compressed/LR REDS
variants and the Spring right eye are not duplicate training examples. Spring disparity remains a
sparse auxiliary sidecar on the five label frames of each training/development sequence.

```powershell
python tools\depth_models\prepare_public_monocular_training.py `
  --download-root E:\ApolloDev\artistic-datasets\downloads `
  --prepared-root E:\ApolloDev\artistic-datasets\prepared `
  --catalog-output E:\ApolloDev\artistic-datasets\artistic_sources_public_mono.json `
  --active-split-output E:\ApolloDev\artistic-datasets\active_public_mono_split.json `
  --workers 4 --download-workers 2
```

When repeatedly deriving a bounded REDS/Spring bootstrap from those prepared sequences, give the
exact normalized-subset preparer the same cache root and always publish into a new output root:

```powershell
python tools\depth_models\prepare_artistic_bootstrap_subset.py `
  --public-prepared-root E:\ApolloDev\artistic-datasets\prepared `
  --output-root E:\ApolloDev\artistic-policy\public-mono-hdr-bootstrap-v4\datasets `
  --workers 8 `
  --preprocess-cache E:\ApolloDev\artistic-policy\cache\v1\preprocessing
```

This exact subset cache is intentionally selection/code-bound. The movie and CHUG raw caches are
the broader reuse boundary when publication metadata, label cadence, or cut thresholds change.

The frozen split is REDS 216/24/30 and Spring 32/5/10 for training/development/test. Official
validation/test archives provide the sealed tests; development sequences are selected from each
training archive by salted source identity. RGB PNG bytes are copied unchanged at full cadence,
five evenly spaced frames per sequence are marked for expensive label generation, and every
aggregate split hashes its source-sequence manifest. REDS has policy weight 1.0 and Spring 1/3, so
balanced source sampling is approximately 75% real dynamic footage and 25% cinematic animation.
Use `--source spring --prepare-only` to stage one already-downloaded source without freezing an
incomplete catalog. Publication is per-split atomic and completed sequences resume safely.

### CHUG native-PQ reference pool

Use `fetch_chug_hdr_references.py` to build a reproducible non-commercial HDR reference pool from
the pristine rows of CHUG. The pinned catalog contains 856 references and 5,136 bitrate-ladder
degradations; the tool admits only `ref == 1` and verifies the complete row counts before selecting
anything. The actual pinned `license.txt` is **CC BY-NC-SA 4.0**, which is stricter than the
repository README's abbreviated CC BY-NC label. ShareAlike applies if an adapted dataset is shared.

```powershell
python tools\depth_models\fetch_chug_hdr_references.py `
  --output E:\ApolloDev\artistic-datasets\chug-hdr-reference-v1 `
  --ffprobe E:\ApolloDev\tools\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffprobe.exe `
  --jobs 4 --probe-only

& E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\depth_models\build_chug_capture_groups.py `
  --dataset E:\ApolloDev\artistic-datasets\chug-hdr-reference-v1 `
  --ffmpeg E:\ApolloDev\tools\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe `
  --depth-anything-root E:\ApolloDev\third_party\Depth-Anything-V2 `
  --depth-weights E:\ApolloDev\models\depth-anything-v2-small\depth_anything_v2_vits.pth

python tools\depth_models\fetch_chug_hdr_references.py `
  --output E:\ApolloDev\artistic-datasets\chug-hdr-reference-v1 `
  --ffprobe E:\ApolloDev\tools\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffprobe.exe `
  --limit 96 --jobs 4
```

Selection is deliberately two-stage. The first command authenticates all candidate probes without
selecting or downloading. The grouper then combines bounded filename/timestamp evidence, five
tone-mapped temporal anchors, pinned DA-V2 Small multiscale identities, exact-recut checks, and
recorded human boundary decisions. It publishes a source-bound `capture_group_manifest.json` only
after all 801 valid references have identities. Selection rejects a missing, stale,
`review_required`, or semantically invalid group manifest. `--cache-only` makes regrouping fail
instead of extracting a missing visual identity. The fingerprint tone map is only a versioned
deduplication transform; it is not production color parity or training preprocessing.

The default fresh selection chooses one deterministic representative per capture group, then
stratifies the 96 representatives by probed orientation/frame-rate and assigns exactly 72 training,
12 development, and 12 test clips. No capture group may cross a split. `--limit 0` selects every
group representative; `--jobs` is hard-bounded to 1-8. The CSV frame-rate column is not trusted.
Interrupted downloads remain as URL-bound `.part` files and resume with a validated HTTP range
response. Candidate ffprobe JSON is reusable by default and bound to the video ID, pinned source
commit and URL, probe schema, and producing ffprobe version. Use `--refresh-probes` only when the
remote objects must be queried again.

For a one-time, network-free migration of an already downloaded 96-clip pool, use
`--audit-only --reuse-existing-selection`. It re-hashes and re-probes only existing masters and
reassigns complete capture groups to exact 72/12/12 counts. It may retain multiple members of one
group within the same split, and records `capture_group_representative_only: false`; a fresh build
remains the preferred fully deduplicated form.

Every pristine URL must have a remote probe before selection; later runs may reuse that cached JSON
as described above. Every selected local MP4 is hashed and probed again after download. Admission
requires exactly one HEVC
`yuv420p10+` video stream, BT.2020 primaries, BT.2020 non-constant matrix, SMPTE ST 2084/PQ, the
catalog geometry, and a duration from 5 through 15 seconds. `selection_manifest.json` freezes the
pre-frame split and grouping identity; `download_receipt.json` records byte counts, SHA-256
identities, ffprobe version, accepted clips, and explicit rejections. The untouched PQ masters
remain under `videos/`.

The fetch command performs **no frame decode or color conversion**. Do not feed these masters
directly to either OpenCV movie preparer: that path intentionally rejects native PQ. The separate
CHUG preparers use the authenticated production color contract: limited-range BT.2020 NCL decode,
explicit PQ EOTF, linear BT.2020-to-Rec.709, 80-nit Windows scRGB normalization, an FP16 capture
boundary, and the exact production HDR depth-input preview. `prepare_chug_native_hdr_training.py`
retains five three-frame diagnostic windows per source video and is not temporal safety evidence.

For ordinal safety, publish the same frozen 12-train/4-development source-video split as one
contiguous full-cadence clip per source video:

```powershell
& E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\depth_models\prepare_chug_native_hdr_full_cadence.py `
  --chug-root E:\ApolloDev\artistic-datasets\chug-hdr-reference-v1 `
  --source-bootstrap-manifest E:\ApolloDev\artistic-policy\public-mono-hdr-bootstrap-v2\datasets\chug-native-pq-v1\native_hdr_bootstrap_manifest.json `
  --output-root E:\ApolloDev\artistic-policy\public-mono-hdr-bootstrap-v3\datasets\chug-native-pq-full-cadence-v3 `
  --ffmpeg E:\ApolloDev\tools\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe `
  --preprocess-cache E:\ApolloDev\artistic-policy\cache\v1\preprocessing `
  --workers 4 --dry-run
```

Remove `--dry-run` only after reviewing the exact 3,088 training and 930 development source-frame
cardinality. Each clip records FFprobe PTS ticks, stream time base, rational source frame rate,
per-frame timestamps, native source/capture identity, and diagnostic preview-delta cut candidates.
The cut candidates never split a source video and are not the hard-cut authority; ordinal replay
uses the harness's completed-depth `SubjectState` evidence. The preparer derives selected IDs only
from the already frozen train/development manifests before resolving or hashing media, and never
opens CHUG test masters. It refuses to replace a stale existing output instead of mutating it.

The CHUG preprocessing cache is below label/cut publication: it stores each authenticated PQ
source's converted FP16 scRGB frame, production depth-input preview PNG, normalized timing, and
threshold-independent preview delta once. A later generation may change curated label IDs, the cut
threshold, capture-group/publication metadata, or worker count without decoding or color-converting
the video again. Each FP16 frame is exactly `width * height * 8` bytes. At the default 1280x720 and
4,018 train+development frames, the raw FP16 portion is about 27.6 GiB, plus preview PNGs and a small
manifest. While the prepared output also exists, expect approximately two frame payloads (cache +
output), not three; no full finished-directory object is cached. The small bootstrap-subset exact
artifact cache remains a separate contract and is unchanged.

After publication, replace the sparse CHUG catalog rows in a new active split without changing the
old catalog or sealed-test assignments:

```powershell
& E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\depth_models\build_full_cadence_artistic_split.py `
  --base-active-split E:\ApolloDev\artistic-policy\public-mono-hdr-bootstrap-v2\datasets\active_artistic_split_sdr_native_pq.json `
  --full-cadence-bootstrap-manifest E:\ApolloDev\artistic-policy\public-mono-hdr-bootstrap-v3\datasets\chug-native-pq-full-cadence-v3\native_hdr_full_cadence_manifest.json `
  --output-catalog E:\ApolloDev\artistic-policy\public-mono-hdr-bootstrap-v3\datasets\artistic_sources_sdr_native_pq_full_cadence.json `
  --output-active-split E:\ApolloDev\artistic-policy\public-mono-hdr-bootstrap-v3\datasets\active_artistic_split_sdr_native_pq_full_cadence.json
```

`hdr_validation_sources.json` pins three complementary non-training fixture sets: three five-second
Sol Levante 4K24 12-bit BT.2020/PQ excerpts for stylized animation, a deterministic eight-second
Nocturne 4K60 12-bit P3/PQ segment for live-action silhouettes and bright spotlights, and CTA-WAVE
CLG1 t53 for HLG signaling. The Sol Levante excerpts retain dark cyan detail, extreme cyan/magenta
highlights, and saturated red/orange action without storing the 37.4 GB complete master. Sol Levante
and Nocturne are bit-exact video-copy remuxes; independent FFmpeg 8.1.2 derivations produced the
recorded byte-identical Matroska files. A direct Nocturne MXF remux was rejected because regenerated
MXF wrapper identifiers made its whole-file hash nondeterministic.
CTA-WAVE HLG validation must inspect decoded-frame metadata: its stream VUI reports transfer
characteristic 14 while preferred-transfer signaling resolves the decoded frame to ARIB STD-B67.
These fixtures test distinct color/scene contracts and do not replace CHUG's scene diversity.

Prepare true left/right authored stereo with a whole-production split when that evidence is
available:

```powershell
python tools\depth_models\prepare_stereo_movie_training.py `
  --video D:\StereoMovies\example-full-sbs.mkv `
  --output E:\ApolloDev\artistic-policy\example-film-v2 `
  --name example-film --domain example_film --film-id example_film `
  --layout side-by-side --eye-order first-left --split training `
  --eye-aspect-ratio 1.7777777778 `
  --ffprobe E:\ApolloDev\tools\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffprobe.exe `
  --write-workers 4 `
  --global-policy-weight 1.0 `
  --license "user-owned noncommercial research copy"
```

For half-SBS web video, `--eye-aspect-ratio` is mandatory: it restores each horizontally
squeezed stored eye to its intended display aspect. Omit it only when each stored eye already has
the correct shape, such as a full-SBS source.
Use `--global-policy-weight` to keep supplementary domains from overwhelming cinematic
productions; the weight is copied into every shot manifest and the dataset manifest.

Both video preparers encode PNGs with a bounded writer pool. The default four threads keep at
most eight writes in flight, so decoding cannot create an unbounded memory backlog. Use
`--write-workers 1` for synchronous diagnosis. A shot remains under a hidden partial name until
every queued write succeeds; any write failure removes the incomplete dataset so the same output
path can be retried safely. Both mono and stereo ingestion probe and record the SDR color contract
before OpenCV decoding, and reject explicit HDR signaling instead of silently clipping it into
BGR8.

`artistic_sources.json` is the schema-2 active admission catalog used by
`audit_artistic_dataset_splits.py` by default. Register each real complete-production mono source
there before auditing it. A global-policy row pins its production ID, whole-production split,
source group, license, and weight; the catalog must contain training, development, and two sealed
test productions from independent groups before it can become an active split. For example, one
real prepared production contributes a row shaped like this (repeat for the complete split; do not
commit placeholder or unavailable media):

```json
{
  "id": "<your-source-id>",
  "production_id": "<the exact --production-id>",
  "source_kind": "mono-video",
  "source_group": "<independent origin>",
  "split": "training",
  "admission": "global_policy",
  "complete_production": true,
  "global_policy_weight": 1.0,
  "license": "<verified license>",
  "license_url": "<license provenance>"
}
```

The current generic catalog intentionally invents no mono paths or productions. Its known authored
SBS sources use `stereo_auxiliary`: they remain available for polarity, correspondence, occlusion,
and stereo-reference validation, but are not camera-strength targets and cannot satisfy an active
global-policy split by themselves.

`artistic_stereo_sources.json` remains the legacy retrieval/reproduction catalog. Fetch its
automatic authored-SBS sources into external storage and write hash/probe/contact-sheet evidence
with:

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

The omitted `--catalog` intentionally selects `artistic_sources.json`, so registered mono
productions use the generic workflow without silently falling back to the stereo-only catalog. The
audit cross-checks catalog kind, split, and weight, rejects duplicate productions, hashes each
prepared manifest, and requires training, development, and at least two sealed test productions.
To reproduce an already frozen legacy stereo-only split, pass
`--catalog tools\depth_models\artistic_stereo_sources.json` explicitly; do not use that compatibility
path for a new mono-first training split.

Use `development` for epoch selection and `test` for at least two separately sourced sealed
productions. Never split shots from one production across these roles.

Generate production depth plus exact Apollo baseline disparity:

```powershell
python tools\depth_models\generate_artistic_depth_run.py `
  --suite E:\ApolloDev\artistic-policy\example-film-v2 `
  --output E:\ApolloDev\artistic-policy\example-film-depth-v2
```

New monocular and stereo preparations contain `label_frames.json`; the generator detects that
manifest automatically and emits each sparse target plus its adjacent temporal evidence frame.
For a frozen **legacy authored-stereo** suite that has sparse `gt_right\frame_*` files but no
`label_frames.json`, opt into the exact GT-right frame-ID selection explicitly:

```powershell
python tools\depth_models\generate_artistic_depth_run.py `
  --suite E:\ApolloDev\artistic-policy\legacy-authored-stereo-v1 `
  --output E:\ApolloDev\artistic-policy\legacy-authored-stereo-depth-v1 `
  --output-gt-right-only
```

That option is auxiliary SBS compatibility, not the mono-first training default. It requires an
authenticated authored-stereo dataset manifest, rejects a simultaneous `label_frames.json`, and
requires every numeric GT-right identity to have the same numeric source RGB identity. The exact
selection is stored in the generation identity, so adding, removing, or renumbering GT-right frames
invalidates depth reuse.

For suites that will be regenerated repeatedly, freeze the expensive source identity once:

```powershell
python tools\sbsbench\build_clip_hash_manifest.py `
  --clips-root E:\ApolloDev\artistic-policy\example-film-v2 `
  --workers 8 --verify
```

When `<suite>\clip_hash_manifest.json` exists, `generate_artistic_depth_run.py` automatically
verifies its exact root, clip/file set, resolved paths, sizes, timestamps, device and inode, then
uses its authenticated clip identities without rereading every source frame. A present but stale
manifest is a hard error; it never silently falls back. Pass `--verify-clip-hashes` to additionally
reread every file, validate SHA-256, and recompute the legacy clip identity. If no frozen manifest
exists, generation preserves the previous full `source_fingerprint` content-hash path. Manifest
schema 2 separates its stable semantic-content digest from the raw manifest-file hash, so rebuilding
an unchanged manifest does not invalidate reusable depth artifacts merely because `created_utc`
changed. The depth-run manifest records both values, the verification level, and every resolved
source identity; it also revalidates the complete source snapshot after generation before publishing
the run manifest.

Sequence `clip` values must be unique single path components. The generator resolves every source
and destination under its declared root, rejects overlap/traversal before any old output is removed,
and accepts exact numeric `frame_*.png`, `.jpg`, or `.jpeg` inputs. Resume validation likewise
requires depth PNGs and baseline-disparity F32 files with their exact artifact extensions.

For a generic mono-first suite, authenticate target RGB/depth rows without requiring a right eye:

```powershell
python tools\depth_models\prepare_artistic_source_rows.py `
  --run E:\ApolloDev\artistic-policy\example-film-depth-v2 `
  --clips E:\ApolloDev\artistic-policy\example-film-v2 `
  --output E:\ApolloDev\artistic-policy\example-film-sources-v1
```

Legacy authored-stereo suites can instead run their deterministic correspondence fitter:

```powershell
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
source/model/color group. Schema 10 requires exactly native SDR plus Windows-HDR raw-white
1000/2500/6000. It emits one row per unique RGB with four authenticated condition targets; each is
the safe-frontier intersection of that condition's two same-color geometries. The eight exact
geometry/input evidence variants remain attached. The all-condition intersection is diagnostic
only, and duplicate RGBs with ambiguous temporal context fail closed.

The trainer accepts schema 10 only and expands each RGB into four image-only samples. No mode or
white-level metadata enters the head: preprocessing changes the pixels/features, while the sample
target comes from the matching authenticated condition target. For rendered disparity loss it
applies each geometry's HLSL clamp to prediction and target, then uses the worse of that condition's
two field/gradient losses. Near-equivalent two-geometry disparity evidence with condition targets
differing by at least 0.1 is rejected as inconsistent rather than learned. The scalar
mean-absolute disparity remains diagnostic only. Frontier bounds add an independent safety loss.

Every candidate run must process full-cadence state. Generic suites use
`--output-label-frames`, which emits authenticated sparse targets plus their consecutive temporal
evidence frames; legacy authored-stereo suites use `--output-gt-right-only`. Every run must also
pass `--no-artistic-policy`; the selector
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

For the bounded REDS/Spring public-mono SDR/HDR bootstrap, the complete repeatable sequence is
wrapped by `orchestrate_artistic_hdr_bootstrap.py`. It uses the production-equivalent same-aspect
uncapped and packed-width-capped eye rasters (1280x720 and 960x540), native
`sdr-srgb-8bit`, Windows HDR SDR-white anchors 1000, 2500, and 6000, and the full 0.9..1.5 grid.
Native SDR is an independent production input contract, not an alias for HDR raw-white 1000. The
plan is self-contained: before source-row preparation it runs 16 resumable
`generate_artistic_depth_run.py` publications bound to the exact schema-28 executable, TensorRT
model assets, config, input variant, preview encoding, dataset, and clip manifest. For each dataset
the SDR publication fully rereads the clip hashes; all three HDR variants use the same
authenticated manifest's stat verification after that full step, avoiding redundant passes over
identical source bytes without claiming that their preprocessing is equivalent. Each RGB is merged
into four policy samples (one SDR and three HDR), and each sample retains both deployment
geometries for safety loss: 240 training samples and 80 development samples. Run the identity phase
first; the full command resumes authenticated results rather than repeating them:

```powershell
& E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\depth_models\orchestrate_artistic_hdr_bootstrap.py `
  --workspace E:\ApolloDev\artistic-policy\public-mono-hdr-bootstrap-v1 `
  --build-dir cmake-build-relwithdebinfo `
  --conf tools\sbsbench\bench.conf `
  --depth-anything-root E:\ApolloDev\third_party\Depth-Anything-V2 `
  --depth-weights E:\ApolloDev\models\depth-anything-v2-small\depth_anything_v2_vits.pth `
  --restart `
  --stop-after identity

# Resume through selection, schema-10 merge, 10-epoch seed-7 training, and development evaluation.
& E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\depth_models\orchestrate_artistic_hdr_bootstrap.py `
  --workspace E:\ApolloDev\artistic-policy\public-mono-hdr-bootstrap-v1 `
  --build-dir cmake-build-relwithdebinfo `
  --conf tools\sbsbench\bench.conf `
  --depth-anything-root E:\ApolloDev\third_party\Depth-Anything-V2 `
  --depth-weights E:\ApolloDev\models\depth-anything-v2-small\depth_anything_v2_vits.pth `
  --restart
```

`--dry-run` emits every exact command and storage estimate without publishing manifests or running
the GPU. `--restart` deletes only stale or partial outputs owned by this plan; authenticated
schema-28 steps remain resumable, so it is safe to keep the flag on the continuation command. The
default compaction retains candidate `results.json` files and only the identity
contracts/disparity rasters needed to replay selection and training. A fully measured identity hard
failure is retained as an explicit confidence-zero no-op negative with all multiplier candidates
disconnected; missing evidence still fails closed. The active split is passed to training and
development evaluation, but the orchestrator never supplies or opens sealed-test labels or frames.

Harness schema 28 deliberately invalidates every committed schema-25 baseline and old HDR identity
render because those artifacts do not authenticate the corrected source-relative preview encoding.
After bootstrap iteration, run fresh core and extended production baseline evaluations and reports
before using regression gates again. Never rescore or relabel the old preview artifacts in place.

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

Training balances ceiling-action class and gives native SDR and aggregate HDR equal runtime-regime
weight. Domains and clips are balanced inside each regime, while the three HDR white anchors split
the HDR half equally; samples adjacent same-shot pairs; uses
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

The schema-13 evaluation records the exact checkpoint, active-split, metric, geometry allow-list,
and canonical label-
fitter identities and one-sided unsafe-ceiling evidence. It reports native SDR, conservative
coherent worst-risk HDR, and each HDR white anchor separately, including sample/shot and
action/identity-class counts, scale,
confidence calibration, exact rendered disparity, and overshoot. Acceptance requires both SDR and
HDR plus every expected HDR white to pass against its own condition target, and fails closed if any
source frame lacks the complete authenticated condition set. The v3 sealed-test approval contract
requires that artifact, hashes it, and refuses development results,
rejected decisions, mismatched identities, or a different set of sealed productions. This binds
the label-screen approval to the checkpoint being exported. Export alone is not deployable.

`promote_artistic_policy.py` is the only manifest-authoring step. Both invocations authenticate the
exact ONNX, schema-5 sidecar, checkpoint and sealed evaluation bytes; the fixed-limit schema-4
neutrality report and every canonical core first-frame image; four fresh, unrescored, passing
immersive/balanced core/extended render results; and current metric/warp hashes. The first invocation
uses `--stage-headset-review` and writes a non-approved schema-2
`apollo-artistic-policy-deployment-v2` manifest that Apollo accepts only under the explicit
`sbs_3d_artistic_live_review` authorization. The second invocation replaces it with an approved
production manifest after binding an explicit named immersive headset review.

Every render result must prove that the optional output was actually consumed and record the exact
ONNX and sidecar hashes. The headset review must select an exact tuple index from the sidecar's
canonical deployment-geometry allow-list; its stated full-SBS resolution and color mode must match
that tuple, proving the optional policy was eligible rather than silently falling back to identity.
Every clip geometry in all four fresh gates must be allow-listed, and their union must exercise
every tuple in that allow-list, including the headset-reviewed tuple.
A promotion also recomputes the metric and warp identities directly from the canonical source lists
in `cmake/prep/artistic_warp_contract.cmake`, and requires harness schema 28 plus eval schema 31.
Every admitted render also binds its per-clip metric preview: `native-srgb-v1` for SDR and
`source-relative-srgb-from-scrgb-white-normalized-v1` for HDR scRGB. This makes pre-fix
schema-25 HDR identity grids stale instead of allowing their incorrectly encoded previews to
become hard-negative training evidence.
It validates the same policy/feature, tensor, output-semantics, bounds, and runtime contracts that
Apollo consumes, so a manifest cannot be successfully promoted and then silently rejected by the
runtime for a stale sidecar contract.
A path string without the matching content hash is never approval evidence.

Export writes a schema-5 JSON sidecar beside the ONNX. It includes `deployed_model`, the source
`base_depth_model`, exact ONNX and evaluation hashes, the sealed-test approval contract, output
semantics, `metric_sha256`, the exact deployment-geometry and input-condition allow-lists, the
`per-input-condition-two-geometry-safe-frontier-v1` target contract, and the frozen policy
baseline. Export first hashes the supplied DA-V2
weights and refuses to build when they differ from the checkpoint's `depth_weights_sha256`. Apollo also
requires the TensorRT engine's generated source marker and compares every policy-affecting resolved
setting before consuming the optional head. The marker binds both the ONNX and serialized-engine
hashes to the TensorRT recipe. Production activation additionally requires the deployment manifest
beside the model. A mismatch disables only the policy; DA-V2 depth still runs normally.
