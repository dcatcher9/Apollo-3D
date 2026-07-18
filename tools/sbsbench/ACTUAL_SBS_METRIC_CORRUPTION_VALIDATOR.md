# Actual SBS metric corruption validator

`validate_actual_sbs_metric_corruptions.py` falsifies the retained deterministic image metrics
against copies of real SBS frames produced by the D3D11 harness. It is deliberately separate from
`run_eval.py`: the validator cannot alter thresholds, rescore a run, or qualify a training label.

The default input is the completed schema-32 core run:

```powershell
E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\sbsbench\validate_actual_sbs_metric_corruptions.py `
  --run cmake-build-relwithdebinfo\sbs_eval\metric-schema32-core-control `
  --max-clips 2
```

The tool validates at least two independent real clips. For each selected interior frame it
authenticates the committed source pixels against the clip hash recorded by the run, requires the
harness schema-16 exact-map contract, checks the packed SBS/map/source geometry, and records SHA-256
hashes for the source, SBS output, warp map, warp mask, and contracts.

## Controlled falsification matrix

Each corruption is applied only to a copy of the actual right-eye output. Nested high-detail
patches occupy approximately 0.08%, 0.5%, 1%, 2%, and 5% of the eye:

| injected defect | detector that must respond |
|---|---|
| missing/black content | exact mapped-source coverage |
| local blur | exact mapped-source texture integrity |
| ringing/oversharpen | exact mapped-source texture integrity |
| local one-eye vertical shift | exact source-relative vertical disparity |
| local one-eye phase shift | interocular phase/orientation burden |
| local one-eye linear-light exposure gain | interocular exposure-rivalry burden |
| local one-eye RGB/white-balance gain | interocular colour-gain-rivalry burden |

The validator checks response direction, approximate monotonicity, a material response at the 5%
endpoint, and 1% localization wherever a detector exposes a spatial map. It also checks exact
no-op reproducibility, global unilateral exposure/colour faults, and a common binocular
photometric transform that should create neither phase nor photometric rivalry. Evidence support
is explicit: a target detector that abstains is a failed
validation, not a zero artifact score.

Image integrity is not reimplemented here. Both its scalar and localization evidence call
`sbsbench.exact_image_integrity_maps`: a 5x5 local RMS gradient-energy ratio detects blur and
ringing, while local gradient-vector error detects phase/orientation corruption with similar
energy. Fixed code-value texture support and exact-map validity make this independent of unrelated
frame contrast. Keeping one shared map contract prevents the validator from passing a detector
different from the one used for evaluation and future labels.

The binocular name contract is also fail closed. Only coherent localized-energy phase plus
source-relative exposure and colour-gain rivalry names are accepted. The retired percentile and
median-normalized chroma names are rejected because they had a footprint cliff and erased a severe
whole-eye unilateral white-balance change.

## Interpretation

A pass is necessary, not sufficient. Synthetic injection proves that a detector responds to a
known defect under actual production geometry; it does not establish human visibility thresholds
or make an experimental metric a safe model label. A failure is intentionally actionable: inspect
the per-clip ladder and localization evidence before changing the detector. Do not loosen this
validator to make a weak metric pass.

Every report states:

```json
{
  "training_label_qualification": "blocked",
  "eligible_training_labels": [],
  "auto_promotes_thresholds": false
}
```

The CLI exits `0` only when every check passes, `1` when falsification checks fail, and `2` when
provenance, contracts, files, or evidence are incomplete.
