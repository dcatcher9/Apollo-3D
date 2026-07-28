# Persistent-motion scene-cut contract and test plan

## Production contract

Shot-latched stereo state follows geometry, not brightness. Apollo evaluates three GPU-resident
fractions from each valid, matched model-input/depth pair:

- `depth_change_fraction`: normalized-depth texels whose value changes by at least `0.05`;
- `raw_rgb_change_fraction`: model-input texels whose largest absolute RGB-channel change is at
  least `0.20`;
- `structural_change_fraction`: texels whose local max-RGB ordinal structure changes.

The structural descriptor compares all ten pairwise orderings in a center/left/right/up/down
stencil. A pair is usable only when its max-RGB difference is at least `0.01` in both frames; a
texel votes after at least four usable pairs, two reversals, and a reversal majority.
`max(R,G,B)` commutes with an identical global monotone curve on every channel. Gain, offset,
gamma, and clipping can therefore preserve an ordering or collapse it into an abstaining tie, but
cannot reverse it. Apollo records this scalar from fixed point-sampled capture texels before the
HDR tonemapper and before area-filtered model resizing; both operations can otherwise reverse
ranks.
The guarantee is scoped to matched capture samples under a global per-channel transform. Codec
noise, color-matrix changes, and local tone mapping are not global exposure.

An appearance proposal requires both:

- raw RGB replacement at or above `0.70`; and
- ordinal structural replacement at or above `0.03`.

The opposite classification is deliberately narrower: raw RGB replacement at or above `0.70`
with ordinal structural replacement strictly below `0.01` is exposure-like. On that transition
alone it vetoes standalone and relative neural-depth authority, because HDR tone mapping or model
normalization can make depth change exceed `0.60` without scene geometry moving. The `0.01`-
to-`0.03` structural band is intentionally ambiguous and does not veto geometry; this preserves
low-structure editorial cuts. A qualified appearance proposal remains mutually exclusive with
the veto and keeps its normal depth-corroborated route.

Appearance is never sufficient by itself. One `shot_cut` pulse is accepted through one of three
geometry-authoritative routes:

1. the geometry arm is ready, the transition is not exposure-like, and depth change is at least
   `0.60`;
2. the appearance arm is ready, the complete appearance proposal is present, and depth change is
   at least `0.25`;
3. after an earlier cut has latched and the eight-update settling window has elapsed, the
   transition is not exposure-like, and depth change is at least `0.30` and rises either `0.20`
   above or to `2x` a depth-motion EMA.

The EMA uses new-value weight `0.125` and is reset to the current depth fraction on every accepted
cut. The third route is an evidence edge, not a timer-triggered reset: a steady high value converges
into its baseline and cannot periodically pulse. The settling guard prevents a cut-frame
normalization jump from appearing as a second cut.

Startup has neither arm. Both arms become ready only after eight settled updates, and arming cannot
fire on that same update. An accepted cut clears both arms. Thereafter:

- geometry independently rearms after two consecutive updates below `0.10` depth change;
- appearance independently rearms after two consecutive updates without the complete appearance
  proposal.

Rearming affects the following update. Persistent depth motion therefore cannot starve a later
appearance+depth cut, and persistent appearance activity cannot starve a later absolute or
relative geometry event. No detector can distinguish a true cut from a sequence that presents
exactly the same steady evidence before and after it; the relative geometry edge is the explicit
escape for the observable case.

The implementation adds no reduction pass, CPU readback, or GPU synchronization point. The
existing preprocess pass writes one capture-ordinal scalar per model texel and the existing valid
history pass copies it. Current/history storage costs
`2 * target_w * target_h * sizeof(float)` (about 2.55 MiB at `770x434`, or 3.43 MiB at
`1036x434`). The existing three-`float4` `SubjectState` layout stores the depth-motion EMA in
`[1].z` and a small integer-valued arm/latch bitmask in `[2].z`; the zero-plane anchor remains in
`[2].x`.

## Calibration evidence

Run the read-only measurement against an existing evaluator artifact tree:

```text
python tools/sbsbench/measure_scene_cut_evidence.py ^
  --artifacts-root cmake-build-relwithdebinfo/sbs_eval/<run> --summary
```

The committed core clips measured at their exact dumped model grids with the footprint-sampling
treatment (`sampling-area-treatment-v1`, runtime shader SHA-256
`598b56a241c243022b186ebee7d3ebbeb3396c2c885711d24dc47a3a56c098fb`):

| Clip / pair class | max raw RGB | max ordinal | max depth |
|---|---:|---:|---:|
| `flat_transition` known cut | 0.958651 | 0.033413 | 0.632135 |
| `scene_cut` known cut | 0.825872 | 0.129041 | 0.631064 |
| `sustained_motion_scene_cut` setup cut | 0.827473 | 0.121931 | 0.657628 |
| `sustained_motion_scene_cut` settled alternating roll | 0.730929 | 0.244880 | 0.289329 |
| `sustained_motion_scene_cut` later true cut | 0.822383 | 0.123050 | 0.687480 |
| `exposure_flash_strobe` largest SDR flash | 1.000000 | 0.000000 | 0.570274 |
| `c525` exposure-heavy sequence | 0.795745 | 0.025154 | 0.016793 |
| largest ordinary raw RGB (`c647`) | 0.127401 | 0.078500 | 0.072658 |
| largest ordinary depth (`aigen_cogvideox_rain`) | 0.037917 | 0.050352 | 0.194175 |

Across the ordinary core clips, `raw >= 0.70 && ordinal >= 0.03` selects the two annotated cut
pairs. The sustained-motion conformance probe deliberately keeps that proposal high after its
setup cut to prove latching and the relative escape. The broad RGB branch rejects ordinary
detailed motion, while the ordinal branch rejects exposure replacement. The `0.25` depth
corroboration floor remains above the measured ordinary-motion maximum of `0.194175`; the `0.60`
absolute geometry floor remains below the known different-depth cuts. Recalibrate all three
frame-level thresholds together if model preprocessing, the depth normalizer, or the descriptor
changes.

The measurement script emits every adjacent-frame record by default so a clip average cannot hide
a periodic reset. It reads source clips and depth artifacts without changing either.

The deterministic `sustained_motion_scene_cut` probe starts monitoring at frame 2. A setup scene
replacement pulses at frame 11, then an exact alternating 0/96-pixel horizontal-roll schedule
holds the production flag word at `16` (latched, neither proposal arm ready) through frame 26.
That frame has scene age 15 and depth-motion EMA `0.3443`. The real replacement at frame 27 changes
`0.6864` of depth texels, clears the relative `baseline + 0.20` test, and produces one accepted
`16 -> 16` latched escape with scene age reset to zero. The evaluator authenticates the source
roll/replacement construction and hard-gates both the exact pulse schedule and those route
preconditions. The default baseline-gated command evaluates this probe without a numeric baseline;
`conformance-only` authority comes from the authenticated invariant itself.

Final production-harness validation covered both conformance probes with the documented default
runner: zero hard/evidence failures, zero exposure pulses or settled anchor/pop drift, and exactly
the authenticated sustained-motion pulses at frames 11 and 27. A separate FP16 scRGB run at
`--simulate-hdr --hdr-scale 4` also produced zero exposure pulses and zero anchor/pop drift. The
complete schema-34 comparison-only suites passed 13 core and 12 extended clips with zero
hard/evidence failures or issues and no GPU-contention flag.

Across the 12 clips shared with the pre-ordinal production run, the measured GPU-stage sum changed
by +0.0410 ms on average (+0.0398 ms median). The added scalar write accounted for +0.0448 ms in
preprocess and ordinal comparison/history for +0.00224 ms in postprocess; inference timing moved
-0.00589 ms, which is run-to-run jitter rather than work performed by this change. The client
mailbox exposure classification adds bit tests only and no dispatch, buffer, readback, or
synchronization.

## Offline deterministic checks

Shader-equivalent numerical and state-sequence tests cover:

- additive, positive-gain, gamma, and clipped exposure changes produce no ordinal vote;
- a gain that crosses the contrast floor and opposing sub-floor noise produce no vote;
- a horizontal-to-vertical ramp is detected by the full ten-pair census;
- the `c525`-like `raw=.80 / ordinal=.019` exposure case cannot use the appearance arm;
- a quiet-structure `raw=.70 / ordinal=.009` exposure transition cannot use either absolute or
  relative depth authority even when neural depth change is `0.95`;
- the exact `ordinal=.01` quiet-veto boundary preserves standalone depth authority;
- raw, ordinal, depth-corroboration, and absolute-depth values immediately below their thresholds
  do not pulse, while exact threshold values do;
- `raw=.95 / ordinal=.10 / depth=.60` cannot pulse on the startup arming update;
- persistent high evidence produces one pulse, not periodic relatches;
- two `raw=.05 / ordinal=.04 / depth=.07` updates independently rearm both branches and cannot
  pulse while doing so;
- a latched `0.35` depth baseline followed by a `0.62` spike pulses once, while steady `0.62`
  evidence never repeats;
- a production trace recognizes the relative route's canonical `cut_flags 16 -> 16` plus
  scene-age reset and rejects a pre-cut trace where either proposal arm has reopened;
- a cut accepted at `0.25` followed by a `0.55` settling jump cannot immediately repulse;
- touching the `0.10` geometry-low boundary does not count, and the two low updates must be
  consecutive.

## Runtime sequences

Run each sequence for at least 20 seconds, followed by five seconds of a static image:

1. Slow and fast camera pans, including a continuous 360-degree turn.
2. Forward/backward corridor motion.
3. A moving foreground subject against a static background.
4. Full-screen scrolling text or a web page.
5. Rapid animation and particle effects without an editorial cut.
6. Exposure ramps, full-screen flashes, and alternating bright/dark frames.
7. A texture-only cut between scenes with deliberately similar depth.
8. A true cut between different-depth scenes.
9. A cut during uninterrupted persistent motion (the deterministic host fixture covers the
   offline path; repeat it on Galaxy XR for end-to-end behavior).
10. Sustained motion that stops, independently rearms the branches, then cuts again.
11. A structured A -> B -> C editorial-cut sweep with the second cut placed one through eight
    valid depth updates after the first. Run each spacing once with a stable interstitial shot and
    once with uninterrupted motion. This is the explicit decision test for the retained
    anti-double-pulse refractory: record both missed second cuts and false relatches from the first
    cut before changing the relative-geometry settle boundary.

Repeat the pan, corridor, flash, and true-cut cases with SDR/HDR, 16:9/21:9/32:9, each supported
depth model, fixed/adaptive pop, and every zero-plane mode.

Record one row per depth update with the three evidence fractions, accepted-shot pulse/reason,
scene age, arm/latch flags, depth-motion baseline, subject-depth raw/EMA, stretch-band bounds,
adaptive-pop ratio, zero-plane source-pixel shift, and rendered disparity percentiles. Diagnostics
may use the exact blocking offline readback. Production telemetry may use only its existing
nonblocking query/staging ring: a `DONOTFLUSH`/`DO_NOT_WAIT` miss must drop that sample rather than
introduce a CPU/GPU synchronization point in the capture loop. Missing live samples under load
are expected and must not be compared one-for-one with the complete offline trace.

## Galaxy XR acceptance

Build and launch the RelWithDebInfo dev build, connect Artemis, and capture synchronized host
telemetry and headset video. The contract passes when:

- exposure-only changes do not move the zero plane or restart adaptive-pop classification;
- sustained motion produces no repeated shot pulses, rhythmic depth pumping, or convergence jumps;
- subject and stretch tracking continue between accepted cuts;
- each different-depth hard cut produces exactly one shot pulse; the zero plane resolves
  immediately and performs its existing settled-age correction without another cut pulse;
- a true cut during persistent motion remains observable through an independently ready
  appearance arm or a relative geometry edge;
- the rapid-cut sweep identifies the shortest spacing that accepts the second cut without ever
  turning one cut's normalization settling into a second pulse; keep the current refractory until
  that false-positive/false-negative boundary has headset evidence;
- fixed HUD elements remain stable relative to the zero plane;
- diagnostics introduce no GPU queue stall; production readback remains nonblocking and any
  skipped live samples are attributed separately from controller behavior.
