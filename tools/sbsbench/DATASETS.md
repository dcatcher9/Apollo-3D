# SBS benchmark datasets

This document owns clip preparation, provenance, semantic metadata, and reference sidecars. Metric
interpretation belongs in [Metrics and gates](METRICS.md).

## Core suite

The committed quick suite lives under `tools/sbsbench/clips/<clip>/`. It is small enough for rapid
iteration and contains real/curated content plus deterministic failure-mode probes.

| Clip | Declared role or target |
|---|---|
| `c339`, `c647` | Legacy AI-generated witnesses |
| `c525`, `c747`, `c841` | Legacy unclassified witnesses |
| `anime_morevna_closeup` | Cel outlines, flat colors, and face/silhouette behavior |
| `aigen_cogvideox_rain` | Human motion, rain, splash, blur, and generative source artifacts |
| `scene_cut` | One authenticated hard-cut update |
| `flat_page` | False stereo and depth hallucination on flat content |
| `flat_transition` | Recovery from textured content to a flat page |
| `fast_motion` | Known 30-pixel-per-frame motion |
| `exposure_flash_strobe` | Exact global exposure changes that must not reset V2 geometry |
| `sustained_motion_scene_cut` | A true replacement during persistent motion |
| `structureless_history_bridge` | Black flash, persistent black, and supported scene return |
| `structureless_white_history_bridge` | Saturated-white version of the structureless bridge |

The normative pulse schedule and detector expectations are maintained in
[Host SBS scene cuts](../../docs/host-sbs-scene-cuts.md), not duplicated here.

Generate the deterministic fixtures with:

```powershell
python tools/sbsbench/make_synth_clips.py
```

## Public extended suite

The decision suite is prepared reproducibly from public datasets and cached outside Git. The
repository-owned [manifest](datasets/manifest.json) pins URLs, checksums, source windows, adapters,
semantic types, and reference evidence.

```powershell
python tools/sbsbench/prepare_public_datasets.py
python tools/sbsbench/run_eval.py --suite extended --comparison-only --label public-control
```

The default cache is `E:\ApolloDev\sbs_bench\datasets`. Set `APOLLO_SBS_DATASETS` or pass
`--cache` to relocate it. When only manifest-owned metadata changes, use
`--refresh-metadata-only`; the preparer verifies the existing frame/reference layout and pinned
decoded-evidence hashes before publishing updated metadata.

`extended-v3` contains twelve inspected 24-frame clips:

| Clip | Dataset | Evidence and coverage |
|---|---|---|
| `bonn_person_walk` | Bonn RGB-D Dynamic | Real capture; walking person and registered depth |
| `bonn_person_close` | Bonn RGB-D Dynamic | Real capture; close person/robot occlusion and depth |
| `tartanair_house_easy` | TartanAir V2 | Simulation; indoor transition, exact depth and flow |
| `tartanair_house_motion` | TartanAir V2 | Simulation; rotation, exact depth and flow |
| `sintel_ambush` | MPI Sintel Stereo | Animation; non-rigid action and right-eye reference |
| `sintel_market` | MPI Sintel Stereo | Animation; people, clutter, and right-eye reference |
| `vkitti_drive_clone` | Virtual KITTI 2 | Simulation; outdoor driving and exact depth |
| `vkitti_drive_rain` | Virtual KITTI 2 | Simulation; low-contrast rain and exact depth |
| `spring_skeleton_cave` | Spring Stereo | Animation; deep layers and thin structures |
| `spring_character_close` | Spring Stereo | Animation; face, hair, and shallow depth of field |
| `spring_daylight_path` | Spring Stereo | Animation; rocks, vegetation, and fine texture |
| `spring_forest_seeds` | Spring Stereo | Animation; thin airborne structure and defocus |

“Public” describes provenance and external storage, not captured reality. Bonn clips are declared
`real-capture`; TartanAir and Virtual KITTI are `simulation`; Sintel and Spring are `animation`.
The evaluator never folds those categories into one unlabeled “real” average.

Bonn derivatives remain local because its official page requests citation without granting
redistribution. TartanAir V2 and Spring are CC BY 4.0. Large Spring and Sintel archives use pinned
HTTP byte ranges and decoded-evidence hashes so changed remote content fails before a prepared
directory is published.

## Clip layout and identity

Input frames are named `frame_<id>.png` or `frame_<id>.jpg` at the clip root. Reference material
is kept in dedicated sidecar directories and can never be discovered as source input by filename
sorting.

Each `meta.json` declares a human-readable name and description and should declare one controlled
`content_type`:

- `real-capture`
- `animation`
- `simulation`
- `ai-generated`
- `anime`
- `synthetic`
- `unclassified`

It may also carry source URL, license/citation, extraction window, `expected_flat`, evaluation
role, and `source_artifacts`. Reports show known source artifacts beside the original frame so
baked bloom, outlines, rain, and generative inconsistency are not blamed on the stereo warp.

The clip identity covers source and consumed GT pixels plus scoring semantics. Human-readable name,
description, and source-artifact notes do not change the identity.

## Authenticated reference sidecars

### Depth or disparity

`gt_depth/frame_*.png` stores deterministic 16-bit depth/disparity evidence. Native float32
`gt_depth/frame_*.npy` is also supported. `gt_depth_valid/frame_*` marks valid and visible pixels;
invalid, occluded, non-finite, and unsupported resize pixels never enter accuracy statistics.

`meta.json` must declare `gt_depth_kind`. Missing GT means the metric is not applicable—it is never
inferred and never replaced by zero. `flat_page` and `fast_motion` contain deterministic synthetic
disparity truth; the five legacy `c*` clips remain no-reference.

### Flow

`gt_flow/frame_*.npz` belongs to its current frame and contains previous-to-current `flow` in source
pixels plus an optional `valid` mask. The evaluator forward-splats the source-grid flow and still
requires photometric agreement. Clips without exact flow abstain from the canonical flow metric;
SEA-RAFT remains a separately versioned optional diagnostic.

### Right-eye reference

`gt_right/frame_*` is diagnostic stereo reference material declared with
`reference_stereo_available: true`. It is not consumed as automatic ground truth, a training label,
or a gate exemption. Clips with only this evidence use `evaluation_role: reference-only` and feed
separately versioned offline correspondence oracles.

Deterministic state/renderer fixtures use `evaluation_role: conformance-only`. They cannot declare
consumed depth/flow GT merely because their source scene was generated by code.

## Add a clip

Extract a short bounded clip with:

```powershell
python tools/sbsbench/split_video.py clip.mp4 `
  -o tools/sbsbench/clips/mine `
  --width 854 --jpg --max 24
```

Drop `--width` and `--jpg` for full-resolution PNG evidence. Evaluation never resizes source frames;
the harness output follows the input size. Results from a small prepared clip are not numerically
comparable with a full-resolution run, although a matched A/B at either resolution is valid.

Before accepting a new clip:

1. classify `content_type` explicitly;
2. document provenance, license, and extraction range;
3. record visible source artifacts;
4. add only reference sidecars whose geometry and validity are authenticated;
5. verify deterministic IDs and hashes; and
6. regenerate baselines only after the current gate and headset review pass.
