# Optional offline-oracle pass

`run_offline_oracles.py` adds expensive optional diagnostics to an existing `run_eval.py` output.
It is intentionally a separate command: the deterministic evaluator remains reproducible without
third-party repositories, checkpoints, PyTorch, or internet access.

The pass currently supports:

- **RAFT-Stereo** for independent dense stereo correspondence, left/right consistency, vertical
  alignment, and comparison with Apollo's exact warp map when that sidecar exists.
- **SEA-RAFT** for flow-compensated temporal residuals, learned flow uncertainty, scene-cut
  abstention, edge ghosts, and eye/source motion mismatch.
- **NVIDIA FLIP** for localized appearance defects after regenerating and registering each eye
  with the exact production source-U map. It reports worst-eye and interocular-imbalance tails,
  not a global quality score. See [NVIDIA_FLIP_APPEARANCE_ORACLE.md](NVIDIA_FLIP_APPEARANCE_ORACLE.md).
- **Apple iSQoE** as a headset-preference-trained holistic diagnostic. Apollo evaluates both eye
  orders and records the official repository revision and checkpoint identity. Controlled
  localized-corruption screens found non-monotonic responses, so its score is never a gate,
  conclusion, or training target.

All are experimental, optional evidence. They are not gates and are not training targets. Every
root and per-clip payload contains `training_label_eligible: false`; downstream label generation
must fail closed if it is asked to ingest these files.

ColorVideoVDP is intentionally not run on the current 8-bit preview artifacts. Its physical
display model requires the actual EOTF, pixels-per-degree, and absolute luminance; treating a
tone-mapped PNG as HDR evidence would produce a precise-looking but invalid score. Add it only
after the harness preserves raw linear/HDR eyes and the target headset/display is calibrated.
Generic PSNR/SSIM/VMAF/LPIPS-style whole-image scalars are also excluded: they are not
stereo-aware, can reward a copied/flattened eye, and let a small severe artifact disappear inside
an image average.

FoundationStereo was also reviewed as a possible replacement for RAFT-Stereo. It is substantially
stronger on public zero-shot stereo benchmarks, but its frozen Depth Anything V2 prior is not an
independent teacher for Apollo's DA-V2 augmentation. Its official code and weights are also
research-only and materially heavier. If added later, keep it as a separately versioned expert:
RAFT and FoundationStereo agreement may increase confidence, while disagreement must abstain and
request inspection. Never average their outputs or promote FoundationStereo alone.

## Run

Use the Python environment that contains PyTorch plus the dependencies required by the official
model repositories and the official FLIP package:

```powershell
python -m pip install flip-evaluator==1.7
```

Then run:

```powershell
python tools/sbsbench/run_offline_oracles.py `
  --run-dir cmake-build-relwithdebinfo/sbs_eval/<run-name> `
  --raft-repo E:/ApolloDev/third_party/RAFT-Stereo `
  --raft-checkpoint E:/ApolloDev/models/raftstereo-middlebury.pth `
  --sea-repo E:/ApolloDev/third_party/SEA-RAFT `
  --sea-checkpoint E:/ApolloDev/models/sea-raft `
  --sea-config E:/ApolloDev/third_party/SEA-RAFT/config/eval/spring-M.json `
  --isqoe-repo E:/ApolloDev/third_party/ml-isqoe `
  --isqoe-checkpoint E:/ApolloDev/third_party/ml-isqoe/checkpoint/isqoe_1_1.ckpt `
  --device cuda
```

The equivalent environment variables are:

- `APOLLO_RAFT_STEREO_REPO`
- `APOLLO_RAFT_STEREO_CHECKPOINT`
- `APOLLO_SEA_RAFT_REPO`
- `APOLLO_SEA_RAFT_CHECKPOINT`
- `APOLLO_SEA_RAFT_CONFIG` (optional; defaults to `config/eval/spring-M.json`)
- `APOLLO_ISQOE_REPO`
- `APOLLO_ISQOE_CHECKPOINT`

Select any subset after `--oracles`, for example `--oracles raft-stereo sea-raft`, or run just FLIP
with `--oracles nvidia-flip`. Use `--oracles apple-isqoe` for the Apple diagnostic. RAFT-Stereo loads once and receives every SBS frame in the run;
SEA-RAFT loads once and is reused across every clip. FLIP requires a same-ID `frame_<id>.*`, `sbs_<id>.png`,
and exact `warp_map_<id>.f32` plus `warp_map_shape.json`; it never falls back to image alignment.
This avoids model initialization per frame or per clip and prevents intended disparity from being
misclassified as an appearance defect.

By default, a missing checkout/checkpoint exits with status 2 after writing an explicit
`unavailable` manifest. `--allow-unavailable` changes missing dependencies into a successful,
machine-readable skip. It does not hide model load/inference failures, which always exit nonzero.

Optional expensive evidence can be enabled with `--save-raft-fields` and
`--save-sea-evidence`. `--save-flip-evidence` stores registered reference/actual eyes, support,
and FLIP maps as compressed NPZ files. `--sea-source-only-flow` skips the per-eye flow estimates
when only the source-compensated residual diagnostics are wanted. `--flip-ppd` must describe the
target viewing density; the default is 67 pixels/degree and remains diagnostic until calibrated
against the target headset.

FLIP currently supports only native-SDR run artifacts. An `hdr_output_stats.json` identifying an
HDR/scRGB preview causes an explicit `abstained` result before the optional package is loaded.
The 8-bit PNG must never be treated as a valid FLIP-HDR buffer; raw linear HDR plus calibrated
display conditions are required first.

## Results contract

The authoritative root index is:

```text
<run>/offline_oracles.json
```

Per-clip payloads are:

```text
<run>/<clip>/offline_oracles/raft_stereo.json
<run>/<clip>/offline_oracles/sea_raft_temporal.json
<run>/<clip>/offline_oracles/nvidia_flip_appearance.json
<run>/<clip>/offline_oracles/apple_isqoe.json
```

The root manifest records selected oracles, dependency/run status, sample counts, and relative
paths to the per-clip payloads. A rerun may leave files from an oracle that was not selected; only
the oracles listed in the current root manifest belong to that invocation.

Status semantics:

- `complete`: every selected oracle ran; individual clips may still legitimately abstain.
- `partial`: at least one oracle ran and at least one dependency was allowed to be unavailable.
- `unavailable`: none ran because dependencies were absent.
- `failed`: a configured model failed validation/loading/inference, or an exact input contract was
  incomplete or invalid.

Do not silently replace `abstained`, `unavailable`, or `failed` values with zero. They are validity
states, not quality scores.
