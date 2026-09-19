# Offline Game 3D depth experiments

These opt-in research tools do not change the installed add-on, source selection
or shader. Current production normalizes by its current smoothed screen plane.
Four fresh setup captures initialize that plane only; later center measurements
move it with a 0.5-second smoothing time constant and a bounded rate of change.
For projection-derived inverse distance `q=1/Z`, the current zero is `q0` and the
reference multiplier is always `K=1/q0`; relative raw depth similarly uses
`H=1/t0`. The signed field is `0.05*(1-q/q0)` before the user's strength multiplier.
Screen-plane placement and normalization therefore move together; the initial
scene does not permanently determine strength. Both paths share **Recenter** and
the zero-plane **current | Target** display. The UI separately shows matrix-derived depth
conversion scale/offset, zero-plane distance (or relative depth), and stereo normalization.
It explicitly identifies the coupling between zero plane and normalization. No scene-derived
reference is labeled as an estimated geometric conversion scale.
See the [production contract](../../../docs/reshade-sbs.md#streamline-depth-selection).

The geometric-estimation experiment has been retired. No live motion-vector/pose
capture feeds the native estimator, and missing projection data uses the relative
raw-depth approximation rather than an estimated geometric conversion.

The [raw/projection convergence check](../../../cmake-build-reshade/raw-projection-convergence-20260917/README.md)
runs the actual production controllers on identical synthetic depth sequences. Their stereo
fields agree when oriented raw depth is proportional to inverse distance. An unknown additive
offset remains a steady error: the live fallback has no parameter that estimates it. Both paths
can forget their different starting scenes without converging to each other's output. This is
a controller/formula check, not a validation of game motion correspondence or rendered artifacts.

## Historical scalar-normalization comparison

`scale_normalization.cpp` compared scalar normalization candidates using the then-current
statistics, depth-content classifier and `scene_gain.h`; Python independently
checked the formulas. The retained September 17 experiment used the earlier adaptive
gain policy. Both that policy and the later fixed-reference experiment are superseded
by current-zero normalization. The historical C++ driver uses the old controller API;
do not build it against the current header or present its results as current-policy
acceptance. Reproduction requires the authenticated frozen source tree or executable
and matching numeric runtime. Restore the original include layout when rebuilding
the archived source snapshots; do not substitute current production headers.
The renderer experiment is separate, through the opt-in
`SUNSHINE_GAME3D_SCALE_CANDIDATES_TEST` fixture.

Use the machine's evaluation interpreter from `AGENTS.md` throughout any reproduction,
put UCRT64 on the actual compiler process's PATH, and write fresh output directories.
Do not bypass the archive's source or numeric-runtime identity checks. Current production
normalization is tested by the controller tests and the actual-shader
`SUNSHINE_GAME3D_STEREO_REFERENCE_TEST` fixture described in the
[add-on README](../README.md#full-pipeline-shader-tests).

The generator requires the earlier synthetic geometry traces identified in its
manifest. They are not game recordings. It retains every input float, source
snapshot, runtime fingerprint and SHA256. Analysis rejects changed inputs,
sources, or numeric runtime. The retained pre-change 2026-09-17 final run contains 6,263 grids and
87,682 formula/admission rows. The initial `cpu/` run is exploratory; `cpu-final/`
adds the whole-image mean and additional adversarial cases.

All methods match gain 4 on equal-area raw depths .125/.375. Static targets and
temporally accepted gains are distinct columns. The common zero plane cancels
from pair differences. Scalar output is an ideal normalized depth-difference
field, not actual pixel disparity, headset comfort, or halo quality.

Both finite-input and existing-content admission are reported. The latter uses
the actual generic depth classifier; it is not asserted to reproduce every SL
or camera-controller admission rule. Source identities, readback association,
camera conversion, and zero-plane movement are deliberately outside this scalar
experiment. Historical traces replay geometry and capture times only, not their
obsolete gains or source-lifecycle state.

See the retained [decision and measurements](../../../cmake-build-reshade/scale-normalization-20260917/review.md)
and the canonical [comparison record](../../../docs/native-stereo-comparison.md).

The retired geometric calibration and motion-capture experiments were removed from the maintained source. Their frozen inputs, source and results remain in the local September 17–18 evidence archives. Production uses an associated projection, otherwise the explicit infinite-far relative-depth assumption. See the production contract linked above.
