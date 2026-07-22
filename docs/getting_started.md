# Getting Started

Apollo is a Windows 11 streaming host for modern Artemis Android XR clients. The supported
video path is native NVIDIA NVENC on a compatible NVIDIA GPU.

## Install and pair

1. Install Apollo and the SudoVDA virtual-display driver.
2. Start Apollo as administrator.
3. Open `https://localhost:47990`, create the Web UI credentials, and leave the pairing page open.
4. In Artemis, discover the host on the local network or add its IP address manually.
5. Enter the PIN shown by Artemis in Apollo's pairing page.

## Start a stream

Artemis supplies the resolution, refresh rate, HDR state, virtual-display request, scale factor,
and initial SBS mode for each launch. Apollo accepts one active remote session at a time and keeps
the launched app and virtual display warm for the configured reconnect grace period.

Apollo supports H.264, HEVC, and AV1 in 4:2:0, including 10-bit HDR where the selected codec,
NVIDIA GPU, Windows display path, and Artemis decoder all support it. Host SBS AI uses the configured
Depth Anything V2/TensorRT profile before native NVENC encoding.

For the developer build and Galaxy XR test loop, see [CLAUDE.md](../CLAUDE.md). For local DisplayPort
AR glasses, see [Local AR glasses](sbs-local-ar-glasses.md).

<div class="section_buttons">

| Previous | Next |
|:---------|-----:|
| [About](about.md) | [Configuration](configuration.md) |

</div>
