# Optional offline oracles

Learned image models are expensive, optional diagnostics for an existing `run_eval.py` result.
They are intentionally separate from the deterministic evaluator so the normal gate remains
reproducible without third-party repositories, checkpoints, PyTorch, internet access, or another
model's failure modes.

Every oracle payload declares `training_label_eligible: false`. No oracle is an automatic gate,
training target, or replacement for headset inspection.

## Available diagnostics

| Oracle | Useful evidence | Important limitation |
|---|---|---|
| [NVIDIA FLIP](NVIDIA_FLIP_APPEARANCE_ORACLE.md) | Registered source-relative appearance defects and interocular imbalance | Native SDR only; requires exact warp maps and calibrated PPD |
| [RAFT-Stereo](RAFT_STEREO_ORACLE.md) | Independent dense correspondence, consistency, and vertical alignment | Optional checkpoint; not ground truth |
| [SEA-RAFT](SEA_RAFT_TEMPORAL_ORACLE.md) | Flow-compensated temporal residual, uncertainty, ghosts, and jitter | Optional checkpoint; cuts/occlusions must abstain |
| Apple iSQoE | Headset-preference-trained holistic diagnostic | Controlled tests found non-monotonic responses to repeated columns, blur, and missing content |

FLIP regenerates the exact source sample selected by each eye's warp map and reports worst-eye
tails. It never substitutes image registration for a missing exact map. RAFT-Stereo and SEA-RAFT
load once and are reused across the run rather than once per frame or clip.

ColorVideoVDP is excluded from current 8-bit preview artifacts because its physical display model
requires the real EOTF, pixels-per-degree, and absolute luminance. Generic PSNR, SSIM, VMAF, and
LPIPS whole-image scalars are also excluded: they are not stereo-aware, can reward copied/flat eyes,
and can hide a severe local artifact inside an image average.

FoundationStereo was reviewed but is not a replacement for RAFT-Stereo. Its frozen Depth Anything
V2 prior is not independent of Apollo's DA-V2 depth source, its official assets are research-only,
and it is materially heavier. If added later, agreement between independent experts may increase
confidence; disagreement must abstain rather than be averaged.

## Run

Use a Python environment containing PyTorch and each selected model's official dependencies. FLIP
uses the official package:

```powershell
python -m pip install flip-evaluator==1.7
```

Then run any desired subset:

```powershell
python tools/sbsbench/run_offline_oracles.py `
  --run-dir cmake-build-relwithdebinfo/sbs_eval/<run-name> `
  --oracles raft-stereo sea-raft nvidia-flip apple-isqoe `
  --raft-repo E:/ApolloDev/third_party/RAFT-Stereo `
  --raft-checkpoint E:/ApolloDev/models/raftstereo-middlebury.pth `
  --sea-repo E:/ApolloDev/third_party/SEA-RAFT `
  --sea-checkpoint E:/ApolloDev/models/sea-raft `
  --sea-config E:/ApolloDev/third_party/SEA-RAFT/config/eval/spring-M.json `
  --isqoe-repo E:/ApolloDev/third_party/ml-isqoe `
  --isqoe-checkpoint E:/ApolloDev/third_party/ml-isqoe/checkpoint/isqoe_1_1.ckpt `
  --device cuda
```

Equivalent environment variables are:

- `APOLLO_RAFT_STEREO_REPO`
- `APOLLO_RAFT_STEREO_CHECKPOINT`
- `APOLLO_SEA_RAFT_REPO`
- `APOLLO_SEA_RAFT_CHECKPOINT`
- `APOLLO_SEA_RAFT_CONFIG`
- `APOLLO_ISQOE_REPO`
- `APOLLO_ISQOE_CHECKPOINT`

Use `--allow-unavailable` to turn a missing checkout/checkpoint into a successful machine-readable
skip. It does not hide validation, load, or inference failures. Optional evidence maps can be saved
with `--save-raft-fields`, `--save-sea-evidence`, and `--save-flip-evidence`.

`--sea-source-only-flow` avoids per-eye flow when only source-compensated residuals are needed.
`--flip-ppd` must describe the target viewing density; the default 67 pixels/degree remains
diagnostic until calibrated for the actual headset.

FLIP abstains before loading its package when `hdr_output_stats.json` identifies an HDR/scRGB
preview. An 8-bit tone-mapped PNG is never valid raw HDR evidence.

## Results

The root index is:

```text
<run>/offline_oracles.json
```

Per-clip results are written below:

```text
<run>/<clip>/offline_oracles/raft_stereo.json
<run>/<clip>/offline_oracles/sea_raft_temporal.json
<run>/<clip>/offline_oracles/nvidia_flip_appearance.json
<run>/<clip>/offline_oracles/apple_isqoe.json
```

Only oracles named in the current root manifest belong to that invocation; files left by an older
selection do not silently re-enter the result.

Status meanings are:

- `complete`: every selected oracle ran; individual clips may still abstain;
- `partial`: at least one ran and at least one dependency was allowed to be unavailable;
- `unavailable`: no selected oracle ran because dependencies were absent; and
- `failed`: configured model/input validation, loading, or inference failed.

Never replace `abstained`, `unavailable`, or `failed` with zero. They are evidence states, not
quality scores.

## Qualification rule

No oracle is promoted because it moved in the expected direction on one clip. Qualification needs
controlled double-edge, jaggedness, missing-structure, polarity, fold, clamp, disparity, shear, and
benign-transform ladders; authenticated diverse source clips; explicit support/abstention behavior;
and manual image plus headset confirmation. Deterministic gate semantics remain in
[Metrics and gates](METRICS.md).
