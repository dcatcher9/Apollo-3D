# Optional SEA-RAFT temporal oracle

`sea_raft_temporal_oracle.py` is an offline, model-backed validator for sequence-only SBS
artifacts. It is not part of the deterministic core gate, and its outputs are never eligible as
single-frame DA-V2 augmentation labels.

Schema 2 preserves the sign of registered temporal changes until after source conditioning.
Schema 1 results are stale because their magnitude-before-subtraction formula could hide an
equal-magnitude change in the opposite direction.

## Why this oracle exists

The deterministic evaluator can use authenticated dataset flow, but its fallback flow estimator
is classical tile matching. SEA-RAFT provides substantially stronger dense motion and a learned
mixture-Laplace uncertainty estimate. The oracle uses both flow directions and rejects:

- scene cuts;
- out-of-bounds matches and forward/backward-inconsistent disocclusions;
- the highest-uncertainty matches;
- pixels whose flow-compensated source appearance changed too much;
- clips with insufficient remaining support.

On the surviving support it subtracts the **signed**, registered mono-source change from the
signed change in each eye, then takes the residual magnitude. Luma, gradient-vector, and
Laplacian changes all follow that order. This is important: subtracting change magnitudes would
mistake an equal-size change in the opposite direction for a perfect match.

| output | interpretation |
|---|---|
| `sea_flow_edge_ghost_p95` | compact worst-eye gradient/LoG trail severity near edges |
| `sea_flow_gradient_ghost_p95` | source-conditioned gradient-vector change residual, in 8-bit code values |
| `sea_flow_log_ghost_p95` | source-conditioned Laplacian-change residual for fine double edges/ringing, in code values |
| `sea_flow_flicker_p95` | signed output-luma change not reproduced by the signed source change |
| `sea_static_jitter_p95` | the same luma residual on source motion <= 0.25 px only |
| `sea_*_motion_mismatch_p95_px` | optional per-eye motion disagreement with source motion |

The source-flow residual, support, cycle error, uncertainty, cut state, and local evidence images
are always retained so a plausible scalar cannot hide bad registration. An `abstained` or `cut`
pair contributes no quality value.

## External setup

Do not vendor the model or its implementation. The official repository is BSD-3-Clause:

```powershell
git clone --depth 1 https://github.com/princeton-vl/SEA-RAFT.git `
  E:\ApolloDev\third_party\SEA-RAFT
```

Download the official Spring-M Hugging Face snapshot outside the Apollo repository:

```powershell
E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe -c `
  "from huggingface_hub import snapshot_download; snapshot_download(`
  'MemorySlices/Tartan-C-T-TSKH-spring540x960-M', `
  local_dir=r'E:\ApolloDev\third_party\SEA-RAFT\models\Tartan-C-T-TSKH-spring540x960-M')"
```

The validated local installation used:

- SEA-RAFT revision `9137517ba24e628442aec097d3afe71d03503b75`;
- `model.safetensors` SHA-256
  `cb8cfbf14c5e0f6734b64add383708b7ff68cc6089a0007c67165d4761346102`;
- official `config/eval/spring-M.json` SHA-256
  `0b2c3f778ea606cc9b093787aa9bfd3d05c831e0765c6cc0572d928d7234e021`.

The wrapper prevents the official constructor from downloading an unnecessary ImageNet ResNet
checkpoint because the complete SEA-RAFT state immediately replaces those initialization
weights. It accepts the official checkpoint's known missing identity BatchNorm downsample state,
but fails on any other model/checkpoint mismatch.

## Run

Source and SBS directories must contain the same numeric frame IDs. Source frames can differ in
resolution; they are resized once to the per-eye evaluation geometry.

```powershell
E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\sbsbench\sea_raft_temporal_oracle.py `
  --source-dir tools\sbsbench\clips\aigen_cogvideox_rain `
  --source-pattern "frame_*.jpg" `
  --sbs-dir cmake-build-relwithdebinfo\sbs_eval\RUN\aigen_cogvideox_rain `
  --repo E:\ApolloDev\third_party\SEA-RAFT `
  --checkpoint E:\ApolloDev\third_party\SEA-RAFT\models\Tartan-C-T-TSKH-spring540x960-M `
  --evidence-dir cmake-build-relwithdebinfo\sbs_eval\RUN\sea-raft-evidence `
  --output cmake-build-relwithdebinfo\sbs_eval\RUN\sea-raft-temporal.json
```

Each evidence PNG is, left to right: current source, reliable-flow mask, localized edge-ghost
heatmap, localized flicker heatmap. Heatmaps are independently scaled for localization only; use
the JSON values for comparisons.

`--source-only-flow` skips four optional bidirectional per-eye flow calls. It preserves the
primary artifact metrics and reduces model inference from six calls to two per pair, but omits
the motion-mismatch diagnostics.

## Qualification boundary

Synthetic tests cover clean translation, held and EMA ghost trails, output-only flicker,
disocclusion masking, scene cuts, static jitter, legitimate source exposure changes, and
equal-magnitude opposite-sign luma/gradient/Laplacian changes that must not cancel. Before
promoting any SEA-RAFT result into a decision gate, run it on authenticated GT-flow clips and
corruption ladders, then correlate the localized worst cases with visual/headset review. A model
oracle must never qualify itself.

References: [official SEA-RAFT repository](https://github.com/princeton-vl/SEA-RAFT) and
[paper](https://arxiv.org/abs/2405.14793).
