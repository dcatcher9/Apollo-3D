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
| `subtitle_cjk_dense` | Dense, outlined CJK burned-in subtitles over moving imagery |
| `subtitle_bilingual_tall_stack` | Unusually tall four-line CJK/English subtitle stacks |
| `subtitle_top_bottom_disjoint` | Simultaneous disjoint top-note and bottom-dialogue regions |
| `subtitle_cjk_highres_transitions` | 1440p thin CJK strokes plus subtitle transitions and a true scene cut |

The normative pulse schedule and detector expectations are maintained in
[Host SBS scene cuts](../../docs/host-sbs-scene-cuts.md), not duplicated here.

Generate the deterministic fixtures with:

```powershell
python tools/sbsbench/make_synth_clips.py
```

The four subtitle fixtures can be regenerated without rewriting any older fixture:

```powershell
python tools/sbsbench/make_synth_clips.py subtitle_cjk_dense `
  subtitle_bilingual_tall_stack subtitle_top_bottom_disjoint `
  subtitle_cjk_highres_transitions
```

`subtitle_cjk_highres_transitions` is a lossless `2560x1440` probe with authored three-pixel CJK
strokes and a two-pixel dark outline. Scaling it to the `770x434` detector canvas is approximately
3.3x in each dimension, making the strokes genuinely sub-pixel there. Its exact metadata schedule
is frames 1–4 empty, 5–8 cue A, 9–12 subtitle-only cue B replacement, 13–16 empty after a
disappearance, 17–20 cue C, and 21–24 a broad scene replacement with cue C changing to cue D.
The scene-cut contract expects the single genuine cut at frame 21, so excluding subtitle pixels
cannot excuse missing the broad replacement outside the loose overlay region.

The same high-resolution fixture also carries the Phase 1.2 authored sanitizer oracle. Its tight
mask is exactly the visible glyph-plus-outline support, while its subtitle-free RGB frames are the
pre-composite moving movie plates. This is reference evidence for testing a future sanitized
inference copy; it is not a synthetic sanitizer result.

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

`extended-v4` contains twelve inspected 24-frame clips:

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

The production V2 tensor contract supports the standard XR landscape, ultrawide, and portrait
families. Six public sources use different native canvases: Bonn is `640x480`, TartanAir is
`640x640`, and Virtual KITTI is `1242x375`. Preparation centers those original pixels without
resizing or cropping on deterministic black production-compatible canvases (`854x480`,
`1138x640`, and `1242x520`, respectively). Every depth, flow, stereo, and validity sidecar receives
the identical translation. Added metric-depth samples are invalid zero, added flow is zero with
`valid=false`, and added out-of-frame masks are asserted. `meta.json` records the source shape,
canvas, offsets, padding, and fitted authenticated tensor for each clip.

Canvas pixels remain part of ordinary stereo and hard-safety scoring because the renderer really
sees them. They are excluded only from reference metrics through the associated validity contract.
This keeps the benchmark on the production fitter without asserting that dataset-native tensor
shapes are calibrated product resolutions.

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

### Subtitle region

`gt_subtitle_region/frame_*.png` stores an 8-bit single-channel binary mask aligned exactly to its
source frame: background is `0` and an authored subtitle region is `255`. Every source frame in a
clip that declares `required_gt_subtitle_region: true` must have one same-sized mask with the same
frame ID; missing, extra, differently sized, or non-`L`/non-binary sidecars are a dataset error.
The same metadata must explicitly declare `subtitle_target_disparity_pct`. It is signed
`x_right - x_left` disparity in the evaluator's reference-aspect image percentage and must be finite
within `[-3.5, 3.0]`; all current subtitle fixtures author the screen plane as `0.0`.

The mask is the union of one or more **loose rectangles around complete visual subtitle blocks**.
It is deliberately not a glyph segmentation: nearby background padding belongs to the region so a
future zero-parallax consumer does not put a disparity boundary around every stroke. Simultaneous
top and bottom subtitles remain separate rectangles in the same binary mask. This sidecar records
the unfeathered authored support; any feathering policy belongs to the consuming renderer rather
than dataset truth. An empty mask represents a frame without visible subtitles. Masks retain full
source-frame resolution, matching the runtime plan to detect burned-in pixels before inference
downscaling. Their pixels, requirement flag, authored target disparity, and any strict
`subtitle_transition_contract` schedule are part of clip identity.

### Subtitle sanitizer oracle

`required_gt_subtitle_sanitizer_oracle: true` requires two additional full-resolution sidecars for
every source frame:

- `gt_subtitle_overlay_mask/frame_*.png` is an 8-bit single-channel `0`/`255` mask tightly covering
  the authored glyph and outline pixels;
- `gt_subtitle_free/frame_*.png` is the same-sized RGB PNG movie plate before those pixels were
  composited.

The tight mask must be wholly contained by `gt_subtitle_region`, and it must equal the exact
per-pixel RGB difference support between the rendered source and subtitle-free oracle. Empty
subtitle states therefore have an all-zero tight mask and a source frame identical to the oracle;
non-empty states have non-empty tight support. The requirement currently applies only to
`subtitle_cjk_highres_transitions` and requires its strict transition contract.

Every declared appearance, subtitle-only replacement, and disappearance is rechecked on the
subtitle-free oracle after excluding the union of the previous and current tight masks. At most 5%
of the remaining pixels may change by more than 40 in one or more 8-bit RGB channels. The generated
fixture's worst subtitle-only boundary is approximately 1.10%; the 5% contract leaves deterministic
plate-motion margin without allowing an undeclared broad scene replacement.

Every declared broad scene cut receives the complementary check over the same unmasked support:
at least 90% of pixels must exceed that RGB-delta threshold. This authenticates the frame-21 scene
cut independently of the simultaneous subtitle replacement. The threshold and both bounds live in
the strict transition contract. Both sidecar directories and the requirement flag are part of clip
identity.

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

Drop `--width` and `--jpg` for full-resolution PNG evidence. The harness never resizes source
frames; its output follows the prepared input size. A manifest-owned public-data adapter may add a
recorded black canvas as described above, but it preserves every source/reference pixel bit-exact
and performs no interpolation. Results from a small prepared clip are not numerically comparable
with a full-resolution run, although a matched A/B at either resolution is valid.

Before accepting a new clip:

1. classify `content_type` explicitly;
2. document provenance, license, and extraction range;
3. record visible source artifacts;
4. add only reference sidecars whose geometry and validity are authenticated;
5. verify deterministic IDs and hashes; and
6. regenerate baselines only after the current gate and headset review pass.
