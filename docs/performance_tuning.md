# Performance Tuning
Apollo uses the native NVIDIA NVENC path. In addition to the options in
[Configuration](configuration.md), keep the NVIDIA driver current and avoid forcing a capture GPU
that is not attached to the streamed display.

## NVIDIA

Enabling *Fast Sync* in NVIDIA settings may reduce latency. For movie-oriented Artemis profiles,
NVENC quality features can intentionally trade latency for image quality.

<div class="section_buttons">

| Previous            |          Next |
|:--------------------|--------------:|
| [Guides](guides.md) | [API](api.md) |

</div>

<details style="display: none;">
  <summary></summary>
  [TOC]
</details>
