# Building Sunshine 3D

Sunshine 3D builds and runs on **Windows 11 with NVIDIA NVENC**. CMake rejects non-Windows
targets. Historical platform source remains only for upstream comparison.

## Requirements

- An NVIDIA GPU and current Windows display driver
- The NVIDIA TensorRT C++ Windows ZIP, extracted locally
- MSYS2 UCRT64 with GCC, CMake, Ninja, cppwinrt, curl-winssl, MinHook, OpenSSL, Opus,
  nlohmann-json, NSIS, and the UCRT64 toolchain
- Official Windows Node.js LTS for the Vue/Vite Web UI
- A recursive checkout, including `third-party` submodules

Set `TENSORRT_DIR` to the extracted directory containing `include/NvInfer.h` and the TensorRT
libraries. A CUDA Toolkit is **not** required: the runtime uses the NVIDIA driver API, while CMake
links the packaged TensorRT libraries and copies their DLLs beside the build output.

Install the MSYS2 packages from an **MSYS2 UCRT64** shell:

```bash
pacman -Syu
pacman -S git mingw-w64-ucrt-x86_64-boost mingw-w64-ucrt-x86_64-cmake \
  mingw-w64-ucrt-x86_64-cppwinrt mingw-w64-ucrt-x86_64-curl-winssl \
  mingw-w64-ucrt-x86_64-MinHook mingw-w64-ucrt-x86_64-nsis \
  mingw-w64-ucrt-x86_64-openssl mingw-w64-ucrt-x86_64-opus \
  mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-nlohmann_json
```

Install Node.js from [nodejs.org](https://nodejs.org/) or nvm-windows. Do not install MSYS2's
Node package; its runtime is incompatible with the current toolchain. `node.exe` and `npm` must be
on `PATH` when CMake configures the Web UI.

## Configure and build

From an MSYS2 UCRT64 shell:

```bash
export PATH="/c/Program Files/nodejs:$PATH"
export TENSORRT_DIR="/c/path/to/TensorRT"
cmake -B cmake-build-relwithdebinfo -G Ninja -S . \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja -C cmake-build-relwithdebinfo
```

From PowerShell, invoke the same shell explicitly:

```powershell
$env:MSYSTEM = "UCRT64"
$env:MSYS2_PATH_TYPE = "inherit"
C:\msys64\usr\bin\bash.exe -lc `
  'export PATH="/c/Program Files/nodejs:$PATH"; export TENSORRT_DIR="/c/path/to/TensorRT"; cd /e/Git/Repo/Apollo-3D; cmake -B cmake-build-relwithdebinfo -G Ninja -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo; /ucrt64/bin/ninja -C cmake-build-relwithdebinfo'
```

Use `RelWithDebInfo` for live XR testing. It preserves symbols without the capture, AI, and encode
stalls of an unoptimized Debug build. Build directories must use the `cmake-build-` prefix because
repository tooling uses it to distinguish generated artifacts.

## Test

```powershell
C:\msys64\ucrt64\bin\ninja.exe -C cmake-build-relwithdebinfo test_sunshine
cmake-build-relwithdebinfo\tests\test_sunshine.exe
python -m unittest discover -s tools/sbsbench -p "test_*.py"
```

GPU-dependent Host SBS and offline-worker checks require a compatible NVIDIA host and are not
substitutes for the ordinary unit suite. Follow [CLAUDE.md](../CLAUDE.md) and
[sbsbench](../tools/sbsbench/README.md) for their controlled evaluation loop.

## Offline media tools

Production offline conversion needs an approved `ffmpeg.exe` and `ffprobe.exe` pair. Place the
pair beside `sunshine.exe`, in a sibling `tools` directory, or supply both trusted absolute paths at
configure time:

```powershell
cmake -B cmake-build-relwithdebinfo -G Ninja -S . `
  -DCMAKE_BUILD_TYPE=RelWithDebInfo `
  -DSUNSHINE_OFFLINE_FFMPEG=C:\approved-media-tools\ffmpeg.exe `
  -DSUNSHINE_OFFLINE_FFPROBE=C:\approved-media-tools\ffprobe.exe
```

Sunshine 3D deliberately does not search the user's `PATH` and the Web UI cannot choose executable
paths. The FFmpeg build must provide the required decoders/demuxers, concat support, static-HDR
filters, Matroska/MP4 muxers, and `hevc_nvenc`/`av1_nvenc`. The job performs its real NVENC
preflight only after acquiring the exclusive offline GPU lease.

## Package

```powershell
cpack -G NSIS --config cmake-build-relwithdebinfo\CPackConfig.cmake
# or
cpack -G ZIP --config cmake-build-relwithdebinfo\CPackConfig.cmake
```

Packagers are responsible for the TensorRT, model, FFmpeg, codec, and other third-party terms of
the exact artifacts they redistribute. See [Legal and redistribution](legal.md).
