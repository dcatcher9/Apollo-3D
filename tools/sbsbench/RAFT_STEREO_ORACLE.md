# Optional RAFT-Stereo correspondence oracle

`raft_stereo_oracle.py` is an offline, experimental cross-check of the rendered SBS eyes. It is
not part of Apollo's runtime, committed evaluator gate, or model-training label set.

The wrapper deliberately does not vendor third-party code or weights. Use the official
[Princeton-VL RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) checkout and its
recommended `raftstereo-middlebury.pth` checkpoint for in-the-wild images. That repository is
MIT-licensed. Its README documents the checkpoint download and recommends the memory-efficient
`alt` correlation implementation for Middlebury; this wrapper follows that inference contract.

Example external setup:

```powershell
git clone https://github.com/princeton-vl/RAFT-Stereo.git `
  E:\ApolloDev\third_party\RAFT-Stereo
# Run the checkout's download_models.sh or download its official models.zip link.
E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe -m pip install opt_einsum
```

Native-resolution evaluation:

```powershell
E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe `
  tools\sbsbench\raft_stereo_oracle.py `
  cmake-build-relwithdebinfo\sbs_eval\RUN\CLIP `
  --repo E:\ApolloDev\third_party\RAFT-Stereo `
  --checkpoint E:\ApolloDev\third_party\RAFT-Stereo\models\raftstereo-middlebury.pth `
  --output cmake-build-relwithdebinfo\sbs_eval\RUN\CLIP\raft_stereo.json
```

`--max-eye-width` is an optional speed lever, but native eye resolution is the qualification
default because downscaling can remove the thin structures this oracle is meant to audit.

## Evidence contract

RAFT-Stereo emits only horizontal displacement and has no confidence head. The wrapper therefore:

- evaluates both possible packed-eye orientations without consulting Apollo's exact warp map;
- runs a flipped/swapped reverse inference and keeps only finite, in-bounds, left-right-consistent
  matches with independent horizontal texture support;
- abstains when textured support is too small or the two eye-orientation hypotheses are ambiguous;
- retains signed left-to-right displacement, so polarity is never hidden by an absolute value;
- measures vertical alignment with a separate local image-cost search, because the model itself
  structurally forces its vertical flow to zero;
- compares the selected inferred field with the exact renderer map only where that map has a
  unique monotonic correspondence, reporting signed bias, MAE, p95, bad-0.5/1/2/4-pixel rates,
  polarity agreement, and coverage.

Exact folds, repeated columns, boundary clamps, and holes remain the responsibility of Apollo's
deterministic map/topology metrics. A learned correspondence model cannot make those regions
uniquely matchable. The oracle also remains a diagnostic rather than a label until it passes
controlled corruptions, benign transforms, real artifact examples, and held-out headset review.
