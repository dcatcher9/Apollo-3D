# Optional NVIDIA FLIP appearance oracle

`flip_appearance_oracle.py` is an offline appearance diagnostic based on NVIDIA's official
[FLIP](https://github.com/NVlabs/flip) implementation. It is available standalone and through
`run_offline_oracles.py`; it is intentionally **not** wired into `run_eval.py`, evaluator
thresholds, acceptance decisions, conclusions, or DA-V2 training labels.

## Dependency

Install the official optional Python package in the environment used for offline analysis:

```powershell
python -m pip install flip-evaluator==1.7
```

The module imports FLIP lazily. A missing package produces `status=unavailable`; it never falls
back to SSIM, PSNR, a reimplementation, or a zero score. The package version is recorded in every
measured payload.

## Why exact-map registration comes first

Raw left/right images intentionally differ from the mono source because of stereo disparity.
Running FLIP directly would score intended camera geometry as an error. The oracle instead:

1. regenerates a clean reference for each eye from the mono source and that eye's exact
   production source-U map;
2. inverts actual and clean eye samples onto the source raster;
3. keeps only samples with a unique inverse in both eyes, excluding bars, clamps, folds, holes,
   and out-of-frame mappings;
4. replaces all invalid test pixels with their clean reference before FLIP;
5. measures the installed FLIP filter footprint with achromatic and opponent-colour impulses,
   then erodes support by that radius plus two pixels before pooling.

This makes clean fractional sampling and legitimate disparity exactly benign. It also prevents a
masked defect from leaking into valid evidence through FLIP's spatial filters.

Only localized reductions are reported:

- worst-eye FLIP p99;
- worst-eye area above FLIP 0.05 (this catches defects occupying less than one percent of a frame);
- p99 absolute left/right FLIP-map imbalance; and
- left/right threshold-area imbalance.

No FLIP mean, fused quality score, or automatic accept/reject threshold is computed.

## HDR limitation

The current harness writes linear-scRGB statistics but saves an 8-bit display-preview PNG. That
PNG is not the lossless linear input required by FLIP-HDR. Passing an `hdr_output_stats.json` with
an HDR/scRGB/FP16 contract therefore returns `status=abstained`. Do not remove this abstention or
silently run LDR-FLIP on the preview. A future HDR oracle needs the raw FP16 eye buffers plus
calibrated display/viewing conditions.

## One-frame command

```powershell
python tools/sbsbench/flip_appearance_oracle.py `
  --source E:/path/to/source/frame_00001.png `
  --sbs E:/path/to/run/clip/sbs_00001.png `
  --warp-map E:/path/to/run/clip/warp_map_00001.f32 `
  --warp-shape E:/path/to/run/clip/warp_map_shape.json `
  --hdr-stats E:/path/to/run/clip/hdr_output_stats.json `
  --output E:/path/to/flip_00001.json `
  --evidence-npz E:/path/to/flip_00001_maps.npz
```

`--hdr-stats` is optional only for a known native-SDR run. `--evidence-npz` stores registered
references, actual eyes, support, and error maps for visual audit.

## Existing-run command

```powershell
python tools/sbsbench/run_offline_oracles.py `
  --run-dir cmake-build-relwithdebinfo/sbs_eval/<run-name> `
  --oracles nvidia-flip `
  --flip-ppd 67 `
  --save-flip-evidence
```

The runner pairs frames strictly by numeric ID and requires the exact production map for every
SBS frame. Missing sources, maps, or a shape contract produce `failed`, not approximate image
registration. A missing official package produces `unavailable` and a nonzero exit unless
`--allow-unavailable` is explicit. HDR preview input remains a legitimate `abstained` result.
Per-clip output is written to `offline_oracles/nvidia_flip_appearance.json`; the normal report
builder renders it only in the collapsed, non-decisive offline-oracle appendix.

## Qualification

Run the synthetic suite with the same Python environment:

```powershell
python -m unittest tools.sbsbench.test_flip_appearance_oracle -v
```

It checks blur, thin-line deletion, ringing, halo, jagged edges, and double edges; monotonic
severity ladders; one-eye versus symmetric defects; clean fractional geometry; hostile bars;
invalid-map masking; flat content; HDR abstention; and missing-dependency behavior.

The authenticated real-source falsification pass uses only clips accepted by the shared dataset
provenance contract:

```powershell
python tools/sbsbench/validate_flip_appearance_real_sources.py `
  --suite core --max-clips 4 --frames-per-clip 1 --max-width 512 `
  --output cmake-build-relwithdebinfo/sbs_eval/flip-appearance-validator-core.json
```

Passing these controlled tests only establishes detector behavior for known corruptions. The
oracle remains `qualification=experimental_diagnostic_only` and
`training_label_eligible=false` until real Apollo regressions are visually annotated and its
viewing-condition sensitivity is calibrated against the target headset.
