# Scene-aware stereo control plan

Status: the first scale-only global checkpoint passed label and depth-neutrality screens but was
rejected by the rendered core suite; no learned checkpoint is approved for live rendering.

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

This is direct supervised training. The offline stereo label fitter is a deterministic data
preparation algorithm, not a second neural model and not part of Apollo's runtime.

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

## Label construction

For every source frame, the offline harness exports:

1. the production processed depth map;
2. `baseline_disparity_*.f32`, Apollo's exact signed full-binocular disparity after its current
   depth shaping, convergence, pop, aspect compensation, adaptive control, and clamp; and
3. `baseline_unclamped_disparity_*.f32`, the same HLSL field before the artistic multiplier and
   final production warp clamp, at artistic scale `1.0`.

The privileged stereo source fitter estimates authored left-to-right disparity, filters
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
  hashes of the source eye, right eye, baseline-disparity artifact, reference disparity, executable,
  configuration, suite manifest, and exact evaluator metric implementation. Resume is allowed only
  when those identities still match.

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
or evaluation, schema-8 variants are collapsed into one schema-9 row per unique RGB. The merge
intersects connected safe frontiers across every exact approved deployment tuple and retains every
geometry's raw disparity for worst-geometry field/gradient loss. This directly supervises
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

The global head does not observe destination geometry. Therefore the schema-9 merge requires a
single reviewed `exact-artistic-policy-render-tuples-v1` allow-list shared by training,
development, and sealed test. Each tuple records exact source dimensions, production DA-V2 model
input dimensions, output-eye dimensions, content scales, disparity-raster dimensions, and color
mode, plus the resolved `depth_short_side` and `depth_max_aspect` used to derive that model input.
Each RGB must contain every tuple for its source/model/color group; identity must be safe in
all variants. The action ceiling is the minimum of their connected maxima, reliability is the
minimum evidence grade, and artifact supervision uses the maximum loss. Identical RGB bytes may
appear only once in the merged labels. Every split must exercise the complete allow-list, so source
preparation normalizes to the reviewed canonical source sizes. Runtime must ignore the policy for
an unlisted tuple.

Sparse sampling is not allowed to masquerade as production temporal processing. Movie preparation
must ultimately feed full-cadence frames through Apollo and select label frames only after the
stateful depth pass. Existing 2 fps depth exports are bootstrap/debug data, not approval data.

## Training and splits

- Freeze DA-V2 and train only the global head first.
- Use the same dynamic positional interpolation in training, evaluation, and ONNX export.
- Resize each image with Apollo's production aspect-aligned patch-grid rule and linear sampling;
  never stretch every source to 770x434.
- Pool class and patch tokens from every DINO stage and append detached, scale-invariant summaries
  of the unchanged depth and DPT decoder field. This gives the global head semantic, geometry, and
  edge-risk evidence while keeping the cache O(frames), not O(pixels).
- Split by complete production into `training`, `development`, and sealed `test`.
- Development selects the epoch. Test is evaluated once for a candidate and must contain at least
  two separately sourced authored productions. Geometry datasets do not supervise camera intent.
- The trainer receives only training/development label bundles plus the frozen split manifest.
  Sealed-test label files are neither opened nor hashed into a checkpoint; the one-time evaluator
  verifies their production identities and fitter contract against the frozen split.
- Balance domains, then clips within domains, so frame count cannot dominate the decision.
- Sample same-shot frame pairs and penalize prediction disagreement. Reconstruct both predicted
  and target fields with the shipping nonlinear equation
  `clamp(raw_disparity * scale, +/- production_clamp)` before the rendered-disparity loss; the old
  `mean(abs(D)) * abs(scale_error)` factorization is invalid once pixels saturate the clamp. Train
  both the resulting field and its horizontal/vertical gradients so boundary saturation remains
  visible to the objective. The independent `+/-3%` perceived-disparity comfort limits remain
  evaluator hard gates; they are not a second renderer clamp. Train confidence as a binary
  safe-ceiling action probability, report
  Brier/ECE calibration, and select checkpoints by clip-then-film macro effective-scale error,
  calibration, and within-shot variation rather than batch loss.
- Every online source must be registered in `artistic_stereo_sources.json` with its production
  identity, license, retrieval provenance, packing, eye order, intended eye aspect, and fixed split.
  Half-SBS storage is anamorphic and must be restored before Apollo preprocessing.
- Freeze the active split by hashing each prepared dataset manifest and checking it against the
  source catalog. The current v1 split uses Big Buck Bunny plus two 0.25-weight NASA productions
  for training, NASA Intro to LIDAR for development, and Elephants Dream plus NASA Goddard as
  sealed tests from two independent source groups. Deleted/unavailable media and an anaglyph-only
  Vimeo presentation are explicitly excluded rather than substituted.

## Approval sequence

1. Unit tests: polarity, units, symmetric eye convention, provenance, split leakage, preprocessing.
2. Label screen on sealed films: confidence-gated ceiling error and action Brier score must beat
   identity where a non-identity safe action exists, without penalizing an exact identity tie;
   aggregate clip first, then film, and require strict majorities of actionable films/domains/clips.
   Report ECE at the fixed runtime threshold. The schema-11 result and sealed-test approval v2
   record the exact checkpoint, active-split, evaluator-metric, canonical fitter identities, and
   one-sided unsafe-ceiling evidence. ONNX export requires an accepted sealed-test result whose
   identities and production set match the checkpoint; development results and rejected or
   edited-to-another-checkpoint results fail closed.
3. Export neutrality: the decoder is frozen, but the shared multi-output ONNX is not promised to
   be bit-identical. Its normalized depth must stay within one 10-bit code on average and two codes
   at p99 against the shipping DA-V2 ONNX.
4. Run four fresh, unrescored offline Apollo render gates: immersive core, immersive extended,
   balanced core, and balanced extended. These include the temporal sequences and together must
   cover every deployable geometry tuple.
5. Hard gates: comfort disparity, image integrity, coverage, halo/stretch, temporal ghosting, and
   ground-truth depth must all be present and pass. Missing metrics fail closed.
6. Use `promote_artistic_policy.py --stage-headset-review` with those same four gates to create a
   non-approved `stage=headset-review` deployment manifest. Place it beside the exact model and
   explicitly set `sbs_3d_artistic_live_review = true`; without that authorization Apollo fails
   closed to the ordinary depth-only camera.
7. Review that staged model on Galaxy XR at immersive style for scene cuts, shot stability,
   comfort, and visible artifacts. The review must exercise one exact sidecar-listed
   source/model/eye/disparity/color geometry tuple; an inactive identity fallback cannot count.
8. Rerun `promote_artistic_policy.py` without `--stage-headset-review`, supplying the same four
   render gates plus the named approval, exact geometry index, full-SBS resolution, refresh rate,
   color mode, and notes. This replaces the staged file with an approved `stage=production`
   `apollo-artistic-policy-deployment-v1` manifest. Clear the explicit live-review override for
   normal production use.

Apollo already consumes the optional output through a confidence-gated shot latch, but only under
an explicit authorization: an offline candidate evaluation, a fully gated staged headset review,
or an approved production deployment. Any identity, geometry, contract, or approval
mismatch disables only the optional artistic action and preserves ordinary DA-V2 depth.

The exported schema-4 JSON sidecar binds the deployed model stem, source DA-V2 identity, exact
DA-V2 weight hash, evaluator `metric_sha256`, ONNX SHA-256, exact sealed-test evaluation SHA-256,
approval identities, output channel meanings, runtime preset equations, frozen policy baseline,
and the canonical exact deployment-geometry allow-list plus its SHA-256.
Export fails before ONNX generation if the supplied depth weights do not match the checkpoint or
the supplied evaluation is not its accepted sealed-test result. TensorRT
compilation writes a matching ONNX/recipe/engine-hash source marker beside the engine. Apollo consumes
`artistic_global` only when all of those identities and the current resolved configuration match;
on any mismatch it ignores the optional output while continuing normal depth inference.

## Later milestones

A dense model output changes the model architecture and is considered only after the global
controller proves useful. It must consume
the exact depth output and DPT multiscale decoder features rather than bilinearly expanding only
last-layer patch tokens, and it requires explicit target-gradient/boundary losses; unconditional
TV smoothing is not acceptable because it rewards blurred silhouettes. A valid dense
target must be explicitly defined as a bounded cyclopean full-disparity correction plus per-pixel
confidence/occlusion-boundary risk. Training must evaluate the corrected field through the same
clamp and warp, with disparity, gradient/edge-placement, temporal, and rendered artifact losses.
It cannot be called an inpainting target.

RGB disocclusion inpainting remains a separate post-warp process because the hole mask exists only
after reprojection and a monocular DA-V2 input cannot know hidden background color. If revisited,
it needs its own model, supervision, latency budget, and rendered-image evaluation.
