# Getting Started

Sunshine 3D is a Windows 11 streaming host for modern Moonlight 3D Android XR clients. The supported
video path is native NVIDIA NVENC on a compatible NVIDIA GPU.

## Install and pair

1. Install Sunshine 3D and the SudoVDA virtual-display driver.
2. Start Sunshine 3D as administrator.
3. Open `https://localhost:47990` and leave the pairing page open. No sign-in is required on this
   PC or an allowed trusted local network; configure credentials only if you explicitly enable
   WAN Web UI access.
4. In Moonlight 3D, discover the host on the local network or add its IP address manually.
5. Enter the PIN shown by Moonlight 3D in Sunshine 3D's pairing page.

## Start a stream

Moonlight 3D supplies the resolution, refresh rate, HDR state, virtual-display request, scale factor,
and initial SBS mode for each launch. Sunshine 3D accepts one active remote session at a time and keeps
the launched app and virtual display warm for the configured reconnect grace period.

Sunshine 3D supports H.264, HEVC, and AV1 in 4:2:0, including 10-bit HDR where the selected codec,
NVIDIA GPU, Windows display path, and Moonlight 3D decoder all support it. Live Host 3D uses the
authenticated Depth Anything V2 Small TensorRT model and takes the configured Host 3D strength
literally before native NVENC encoding. Model, adaptive-pop, subject-shaping, and zero-plane
overrides remain available only to offline conversion and evaluation.

For the developer build and Galaxy XR test loop, see [CLAUDE.md](../CLAUDE.md). For local DisplayPort
AR glasses, see [Local AR glasses](sbs-local-ar-glasses.md).

<div class="section_buttons">

| Previous | Next |
|:---------|-----:|
| [About](about.md) | [Configuration](configuration.md) |

</div>
