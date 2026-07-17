# Scene-aware stereo control plan

Status: the first scale-only global checkpoint passed label and depth-neutrality screens but was
rejected by the rendered core suite; no learned checkpoint is approved for live rendering. The
subsequent 26-bin ordinal experiment is also paused and rejected as a source of training truth.
Its connected-prefix assumption was contradicted by the completed render grid: 451 of 800
geometry/frame sequences returned to a passing state after their first metric failure, and nearly
identical DA-V2 evidence sometimes received widely separated frontier labels. The implementation
and evidence are retained on the training-research branch for audit only. Future scene-aware pop
work must learn user-perceived outcomes for explicit absolute pop settings, with automatic metrics
kept as validated constraints or diagnostics rather than an assumed perceptual oracle.

## First checkpoint result (2026-07-15)

The frozen scale-only checkpoint (`global-scale-schema6-v1`) passed its development and sealed
authored-stereo label screens. Its exported depth also passed the production P2/P98-normalized
neutrality gate on all 11 core first frames (mean <= one 10-bit depth code, p99 <= two codes).

The final rendered core A/B rejected it. Against a fresh official DA-V2 control, the learned
controller reduced average stereo spread by 5.6% and regressed `c747` primary warp evidence:
`source_halo_p95` 8.26 -> 12.18 and `source_stretch_pct` 7.36 -> 8.95. There were no hard comfort
failures. A model-depth ablation with policy consumption disabled returned `c747` to 8.34 halo and
7.31 stretch, isolating the failure to the camera multiplier rather than the multi-output depth.
Fixed-scale screening confirmed that 0.95 reproduced the regression, 0.98 still crossed the
primary halo tolerance, and 0.99 was within tolerance but too small to justify learned control.

Development-only confidence screening selected threshold 0.10 and retained every development
frame. Thresholds high enough to suppress the failing core scene materially worsened development
label error, so no test-set-tuned confidence exception is accepted. The expanded suite and live
review were deliberately not run after the core rejection.

## Goal

Reuse DA-V2 Small's already-computed DINOv2 and DPT features to produce the strongest stable
artistic stereo that Apollo can render for a scene without violating comfort, image-integrity,
warp-artifact, or temporal-stability constraints. The target is deliberately **not** physical
camera truth. Authored stereo is valuable geometry/artifact evidence and an optional reference
look, but a conservative theatrical master must not teach the default product to reduce pop.

The runtime exposes three realizable looks from one learned safety prediction:

- `clean`: preserve the current Apollo camera (`1.0` multiplier);
- `balanced`: consume half of the scene's validated extra-pop headroom; and
- `immersive`: consume all validated headroom.

This separation lets later profiles choose a look without retraining or confusing style preference
with safety. Approval is based on the final Apollo SBS render and headset-oriented comfort/artifact
metrics, not policy-label loss or similarity to an authored right eye alone.

This is direct supervised training. The deterministic render-feasibility selector is data
preparation, not a second neural model and not part of Apollo's runtime. Authored-stereo
correspondence fitting remains an optional diagnostic path only.

## Milestone 1: global controls only

The network contract is:

```text
pixel_values -> predicted_depth, artistic_global
artistic_global = [safe_scale_ceiling, ceiling_confidence]
```

- `safe_scale_ceiling` (the first tensor element; the old `baseline_multiplier` name is retired)
  is the highest identity-connected scale at or above `1.0` that passed the complete frozen render
  grid. It is a safety capacity, not an authored-camera estimate.
- `ceiling_confidence` is trained as the hard actionable `0/1` probability used by runtime. Below
  `0.5`, runtime falls back to identity. Protected-metric margins and grid support are retained
  separately as `safety_margin_reliability`; that value weights uncertain labels but is not the
  model's probability target. Neither value is derived from authored PSNR.
- Runtime applies `1 + style_mix * (safe_scale_ceiling - 1)`, where style mix is `0`, `0.5`, or
  `1`, then independently applies the exact production warp clamp. The selected value is
  latched at the first resolved frame of a shot and changes only at a validated hard cut.
- Apollo continues to shift the eyes symmetrically by `+D/2` and `-D/2`.

The current production warp remains unchanged until a checkpoint passes all gates. No dense local
output is included in this milestone: the previous residual mixed authored intent, DA-V2 error,
stereo-matcher error, and warp/disocclusion error into one unidentifiable target.

## Rejected V2 experiment: ordinal safety frontier

The following historical offline experiment did not replace the deployed two-channel scalar
contract and must not be resumed as authoritative supervision. It reused the same frozen DA-V2
Small multiscale/DPT/depth features and predicted:

```text
artistic_safety_frontier[k] = P(the identity-connected scale 1.00 + 0.02*k is safe)
k = 0..25
```

The head constructs non-increasing probabilities by subtracting cumulative positive logit
decrements. Labels store the highest proven safe scale and the first proven unsafe scale. A render
that stays safe through `1.50` is right-censored, not an exact `1.50` ceiling; an identity hard
failure is an explicit left-censored/non-actionable example. Unknown bins beyond the first failure
are never invented as unsafe supervision. Training uses interval-censored likelihood plus a masked
known-bin calibration loss with a larger penalty for predicting a measured unsafe bin as safe.

Offline selection uses the highest contiguous calibrated point probability. A separately validated
lower confidence bound may replace the point estimate only when its construction and coverage are
recorded explicitly. If identity misses the authorization threshold, selection abstains; identity
is never silently declared safe. `clean`,
`balanced`, and `immersive` style mixing happens only after that safety cap is selected. Runtime
integration is forbidden until a replay of the same causal controller passes rendered gates:

- lower the cap on the same completed depth update when risk increases;
- raise only after a development-frozen sustained-safe window;
- apply an upward cooldown to prevent pumping; and
- reset to identity on the authoritative production hard-cut event.

That controller work is deferred during the current target-only model experiment. Training,
development, and the eventual sealed test contain only authenticated target images; nearby frames
do not enter the model corpus or checkpoint score. The requirements above return only at the
runtime-promotion stage.

Existing evaluator `trigger`/`trigger_min` values remain report diagnostics; only reviewed
`ordinal_hard_min`/`ordinal_hard_max` fields add ordinal absolute safety bounds. Label headers bind
portable hashes of the builder, ordinal contract/selector, geometry merge, runtime-scene adapter,
current metric contract, and `thresholds.json` so code or acceptance drift invalidates stale
bundles. Training additionally requires the exact training-eligible orchestration catalog and
authenticates its active split, executable/config publication, `sbsbench.py`, `run_eval.py`, code
identities, source rows, and frontier bundles before DA-V2 feature caching starts.

The first V2 experiment remains a globally coherent scale action. A dense head, if justified later,
first supplies internal renderer-risk evidence to the ordinal head; it does not directly multiply
per-pixel disparity.

## Label construction

Ordinary full-cadence monocular movie shots are the primary source. They match the deployed model's
RGB input and do not import a theatrical master's often-conservative stereo volume. Label
generation has two authenticated passes: a sparse identity-artifact pass for depth/raw-disparity
training artifacts, and a full-frame safety pass for every scale used to prove a frontier. Every
source frame advances DA-V2 normalization, EMA, subject state, and scene state in both. Standalone
stills may supplement spatial/edge coverage but cannot provide temporal safety labels.

For every selected target/evidence frame, the offline harness exports:

1. the production processed depth map;
2. `baseline_disparity_*.f32`, Apollo's exact signed full-binocular disparity after its current
   depth shaping, convergence, pop, aspect compensation, adaptive control, and clamp; and
3. `baseline_unclamped_disparity_*.f32`, the same HLSL field before the artistic multiplier and
   final production warp clamp, at artistic scale `1.0`.

When authored SBS is available as auxiliary evidence, the privileged stereo source fitter estimates
authored left-to-right disparity, filters
left/right inconsistencies, photometric mismatches, occlusions, weak texture, and vertical
parallax, and robustly fits an authored-reference scale. That value is diagnostic; it is not the
immersive training target. A direct rendered-right PSNR is also only diagnostic because the input
eye is an authored left camera while Apollo interprets its mono input as cyclopean and renders
symmetric `+D/2` and `-D/2` eyes.

```text
authored_full_disparity ~= authored_fit_scale * apollo_full_disparity
```

The fit must have positive polarity and pass through the neutral origin. A learned convergence
offset was prototyped but rejected because it regressed development MAE; Apollo retains its
existing zero plane. Controls are latched over a complete shot. Every label stores
hashes of the source image, baseline-disparity artifacts, executable, configuration, suite manifest,
and exact evaluator metric implementation. Optional right-eye/reference hashes are retained when
that auxiliary evidence exists. Resume is allowed only when those applicable identities still match.

The schema-8 label selector adds single-geometry render feasibility without becoming a neural
teacher. For
each training/development shot, render a frozen grid of symmetric scale candidates through Apollo,
score exact HLSL pop, comfort, integrity, halo/stretch, and temporal stability, and fail closed when
any required evidence is missing. Starting at identity, walk the sampled grid in each direction and
stop at the first unsafe point; farther disconnected candidates cannot become a target even if they
pass individually. The highest safe scale at or above identity is `safe_scale_ceiling`. Clean,
balanced, and immersive targets are deterministic views of that ceiling; authored-fit scale/PSNR
remain diagnostics. Training and development bundles are frozen
before fitting. Sealed-test productions remain unavailable until the checkpoint is frozen; their
schema-8 targets are generated only for the one-time held-out evaluation. Before either training
or evaluation, schema-8 variants are collapsed into one schema-10 row per unique RGB. Native SDR
and HDR raw-white 1000/2500/6000 each receive their own authenticated target: the intersection of
that input condition's two same-color deployment geometries. The row retains all eight raw
disparity artifacts for condition-local worst-geometry field/gradient loss. The former
all-condition intersection remains a diagnostic and is never a training target. This directly supervises
the camera decision that Apollo can safely realize, instead of fitting authored disparity while
ignoring the gather warp's nonlinear artifact response.

Every grid render must use current-frame depth, the production Apollo warp, and
`--no-artistic-policy`. The control and scale-1 candidate are the same explicit
`--artistic-scale-override 1.0` render; override `0` uses the legacy clamp and is rejected.
Otherwise the selector rejects the grid as circular or mismatched supervision. The schema-8
contract also freezes the complete policy-affecting production baseline (depth model, pop/adaptive
pop, EMA/range tracking, subject shaping, depth resolution/aspect, zero plane, profile, and
harness/evaluator versions). The source-depth run's schema-7 harness contracts and configuration
hash must match that same baseline. Existing pre-schema-8 bundles are intentionally not reusable.

The global head does not observe destination geometry or explicit SDR/HDR metadata. Therefore the
schema-10 merge requires a
single reviewed `exact-artistic-policy-render-tuples-v1` allow-list shared by training,
development, and sealed test. Each tuple records exact source dimensions, production DA-V2 model
input dimensions, output-eye dimensions, content scales, disparity-raster dimensions, and color
mode, plus the resolved `depth_short_side` and `depth_max_aspect` used to derive that model input.
Each RGB must contain all four image-derived input conditions and both geometries for each.
Within a condition, the action ceiling is the minimum of its two connected maxima, reliability is
the minimum evidence grade, and artifact supervision uses the maximum loss. Every condition owns
its safety target even when frozen DA-V2 features or depth happen to be nearly identical: the final
warp evidence and safety constraints are condition-local. A paired feature/depth/target audit
reports such ambiguity as calibration risk, but it never merges targets, blocks label admission,
or adds SDR/HDR metadata to the model input. Identical RGB bytes may appear only once within one
condition's merged labels. Every split must exercise the complete allow-list, so source preparation
normalizes to the reviewed canonical source sizes. Runtime must ignore the policy for an unlisted
tuple.

Target-only sampling is intentional for the current single-image safety experiment. Source media
and label provenance remain authenticated, but only the frozen target frames are published to
train/development/test. This phase makes no temporal-stability or controller claim; full-cadence
processing and adjacent-frame gates return as a separate promotion experiment after the image
model demonstrates useful safety/pop discrimination.

True-HDR temporal evidence uses the separate
`apollo-chug-native-pq-full-cadence-v1` preparation. It preserves the frozen CHUG source-video
split while retaining all 3,088 training and 930 development frames: one source video is one
contiguous clip. Every sequence binds its rational source frame rate, FFprobe stream time base, and
exact per-frame PTS. The 24, 29.97, and 30 fps sources therefore replay at their authenticated
cadence rather than a trainer-wide 30 fps assumption. Preview-delta cut candidates are stored for
audit only; the authoritative controller reset remains the completed-depth `SubjectState` hard-cut
event emitted by the exact render harness. Sparse three-frame CHUG windows remain visual/metric
artifacts and cannot enter ordinal safety labeling. The full-cadence publication is a new dataset
and active split; it never overwrites the existing sparse production or opens sealed CHUG test
masters.

## Training and splits

- Freeze DA-V2 and train only the separate ordinal global head first. The production depth output
  and the shipping scalar head remain unchanged during this experiment.
- Use the same dynamic positional interpolation in training, evaluation, and ONNX export.
- Resize each image with Apollo's production aspect-aligned patch-grid rule and linear sampling;
  never stretch every source to 770x434.
- Pool class and patch tokens from every DINO stage and append detached, scale-invariant summaries
  of the unchanged depth and DPT decoder field. This gives the global head semantic, geometry, and
  edge-risk evidence while keeping the cache O(frames), not O(pixels).
- Split by complete production into `training`, `development`, and sealed `test`.
- Development selects the epoch. Test is evaluated once for a candidate and must contain at least
  two separately sourced complete productions from independent groups. Authored stereo and geometry
  references may validate polarity/artifacts, but do not supervise camera strength.
- The trainer receives only training/development label bundles plus the frozen split manifest.
  Sealed-test label files are neither opened nor hashed into a checkpoint; the one-time evaluator
  verifies their production identities and fitter contract against the frozen split.
- Give native SDR and aggregate HDR equal total sampling mass. Within each regime, balance the
  exact censor/highest-safe ceiling bin crossed with the first-unsafe failure-family set, then
  condition, domain, clip, and frame. This prevents common easy ceilings or one artifact family
  from dominating while the three simulated-HDR anchors still share one HDR half.
- Train the 26-bin survival frontier with finite-interval, right-censored, and identity-failure
  likelihoods. Apply calibration only through one shared positive affine logit transform so scale
  monotonicity is preserved. Do not penalize unconditional same-scene prediction disagreement: a
  real new hazard must be allowed to lower the frontier immediately. Temporal stability is outside
  the current target-only checkpoint and report contract.
- Reconstruct selected and target fields with the shipping nonlinear equation
  `clamp(raw_disparity * scale, +/- production_clamp)` for auxiliary rendered-disparity/gradient
  evidence. The independent `+/-3%` perceived-disparity comfort limits remain evaluator hard gates;
  they are not a second renderer clamp. Checkpoint eligibility first requires zero unproven
  selected/applied overshoot and zero measured left-censored identity hard failures per runtime
  regime. Identity failures remain hard failures even when the learned selector abstains. Among
  eligible checkpoints, maximize film-balanced exact realized-pop gain, then naturally prefer
  checkpoints that do not select above the first safe scale attaining the same maximum realized
  pop, followed by interval likelihood, Brier, known-bin ECE, underreach, and pumping. Plateau
  evidence is diagnostic only: oracle target pop never edits a selected scale or causal replay,
  and excess blocks an optimal-utility claim rather than changing a safety bin.
- Training requires an explicit, positive `--minimum-development-pop-gain-pct` and a reviewed
  `--minimum-development-pop-gain-rationale`, both frozen into the contract before predictions.
  SDR, the HDR aggregate, and every HDR variant must each meet that film-balanced
  realized-pop floor. Zero-overshoot/all-abstain is diagnostic, never checkpoint-eligible, and a
  development-selected checkpoint is never described as a production-accepted policy. Because a
  live model has no evaluator-side `unproven` input, checkpoint eligibility also requires zero
  non-boundary unproven frames in SDR, HDR aggregate, and every HDR variant.
- Every global-policy source must be registered in the schema-2 `artistic_sources.json` catalog with
  its source kind, complete-production identity, license, retrieval provenance, independent source
  group, weight, and fixed split. Mono productions need no eye metadata. Authored SBS entries are
  retained there as `stereo_auxiliary` evidence with verified packing, eye order, and intended eye
  aspect; half-SBS storage is anamorphic and must be restored before validation.
- Freeze the active split by hashing each prepared dataset manifest and checking it against the
  generic source catalog. The committed generic catalog deliberately contains no fabricated mono
  productions; real training, development, and two independent sealed-test productions must be
  prepared and registered before a new split can pass audit. The old schema-1
  `artistic_stereo_sources.json` catalog and its Big Buck Bunny/NASA/Elephants Dream assignment are
  preserved only to retrieve auxiliary SBS evidence and reproduce the frozen stereo-only v1 work,
  never as the implicit default for new training. Deleted/unavailable media and an anaglyph-only
  Vimeo presentation remain explicitly excluded rather than substituted.

## Approval sequence

1. Unit tests: polarity, units, symmetric eye convention, provenance, split leakage, preprocessing.
2. Label screen on sealed films: the calibrated ordinal selection must have zero unproven overshoot,
   zero identity hard failures, and material exact realized-pop gain over scale `1.0`, without
   penalizing an exact identity tie. Report interval-censored NLL, known-bin Brier,
   measured-bin example-averaged ECE, underreach, right-censored utilization,
   and identity-failure abstention. Aggregate scene first,
   then complete production. The schema-13 scalar result is not sufficient evidence for V2; the
   new ordinal result must report native SDR and a
   coherent worst-risk-white HDR aggregate independently, plus every authenticated HDR white
   anchor with its own two-geometry target. SDR, the HDR aggregate, and every white must pass;
   missing or incomplete condition evidence fails closed. The sealed-test approval v3 records the
   exact checkpoint, active-split, evaluator-metric, canonical fitter identities, condition-target
   contract, and
   one-sided unsafe-ceiling evidence. ONNX export requires an accepted sealed-test result whose
   identities and production set match the checkpoint; development results and rejected or
   edited-to-another-checkpoint results fail closed.
3. Export neutrality: the decoder is frozen, but the shared multi-output ONNX is not promised to
   be bit-identical. Its normalized depth must stay within one 10-bit code on average and two codes
   at p99 against the shipping DA-V2 ONNX.
4. After the target-only model passes, restore the deferred promotion phase: render the actual
   calibrated model selection after the causal controller (not merely static oracle scales), and
   pass its adjacent-frame temporal gates. Replay the live completed-depth
   cadence including busy drops; the synchronous harness scene sequence alone is not promotion
   evidence.
5. Run four fresh, unrescored offline Apollo render gates: immersive core, immersive extended,
   balanced core, and balanced extended. These include the temporal sequences and together must
   cover every deployable geometry tuple.
6. Hard gates: comfort disparity, image integrity, coverage, halo/stretch, temporal ghosting, and
   ground-truth depth must all be present and pass. Missing metrics fail closed.
7. Use `promote_artistic_policy.py --stage-headset-review` with those same four gates to create a
   non-approved `stage=headset-review` deployment manifest. Place it beside the exact model and
   explicitly set `sbs_3d_artistic_live_review = true`; without that authorization Apollo fails
   closed to the ordinary depth-only camera.
8. Review that staged model on Galaxy XR at immersive style for scene cuts, shot stability,
   comfort, and visible artifacts. The review must exercise one exact sidecar-listed
   source/model/eye/disparity/color geometry tuple; an inactive identity fallback cannot count.
9. Rerun `promote_artistic_policy.py` without `--stage-headset-review`, supplying the same four
   render gates plus the named approval, exact geometry index, full-SBS resolution, refresh rate,
   color mode, and notes. This replaces the staged file with an approved `stage=production`
   `apollo-artistic-policy-deployment-v2` manifest. Clear the explicit live-review override for
   normal production use.

Apollo already consumes the optional output through a confidence-gated shot latch, but only under
an explicit authorization: an offline candidate evaluation, a fully gated staged headset review,
or an approved production deployment. Any identity, geometry, contract, or approval
mismatch disables only the optional artistic action and preserves ordinary DA-V2 depth.

The exported schema-5 JSON sidecar binds the deployed model stem, source DA-V2 identity, exact
DA-V2 weight hash, evaluator `metric_sha256`, ONNX SHA-256, exact sealed-test evaluation SHA-256,
approval identities, output channel meanings, runtime preset equations, frozen policy baseline,
the canonical exact deployment-geometry/input-condition allow-lists, and the authenticated
per-condition two-geometry target contract.
Export fails before ONNX generation if the supplied depth weights do not match the checkpoint or
the supplied evaluation is not its accepted sealed-test result. TensorRT
compilation writes a matching ONNX/recipe/engine-hash source marker beside the engine. Apollo consumes
`artistic_global` only when all of those identities and the current resolved configuration match;
on any mismatch it ignores the optional output while continuing normal depth inference.

## Later milestones

A dense model output changes the model architecture and is considered only after the ordinal global
controller proves useful. Its first role is a multiscale risk map for halo/stretch,
disocclusion/coverage, foldover/Jacobian, and temporal-boundary evidence, aggregated conservatively
into the ordinal head. It must consume
the exact depth output and DPT multiscale decoder features rather than bilinearly expanding only
last-layer patch tokens, and it requires explicit target-gradient/boundary losses; unconditional
TV smoothing is not acceptable because it rewards blurred silhouettes. A valid dense
target must be explicitly defined. Only after the risk-only head and a scale-by-zero-plane oracle
reach their limit may a bounded cyclopean full-disparity correction be tested. That correction must
be zero-mean and decorrelated from base disparity so it cannot secretly reproduce global scale or
convergence. Training must evaluate the corrected field through the same clamp and warp, with
disparity, gradient/edge-placement, temporal, and rendered artifact losses. It cannot be called an
inpainting target.

RGB disocclusion inpainting remains a separate post-warp process because the hole mask exists only
after reprojection and a monocular DA-V2 input cannot know hidden background color. If revisited,
it needs its own model, supervision, latency budget, and rendered-image evaluation.
