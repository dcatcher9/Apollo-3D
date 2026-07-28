# Building Sunshine 3D

Sunshine 3D supports Windows with native NVIDIA NVENC. CMake deliberately rejects non-Windows
targets; historical platform files remain only to make upstream comparisons easier.

## Requirements

- Windows 11
- NVIDIA GPU and current display driver
- MSYS2 UCRT64 with GCC, CMake, Ninja, cppwinrt, curl-winssl, MinHook, OpenSSL, Opus,
  nlohmann-json, NSIS, and the standard UCRT64 toolchain
- Official Node.js for the Vue Web UI

Offline Host 3D conversion additionally requires an approved, compatible `ffmpeg.exe` and
`ffprobe.exe` pair. These are runtime media tools for the native C++ worker, not a Python
dependency.

Install the MSYS2 packages from an **MSYS2 UCRT64** shell:

```bash
pacman -Syu
pacman -S git mingw-w64-ucrt-x86_64-boost mingw-w64-ucrt-x86_64-cmake \
  mingw-w64-ucrt-x86_64-cppwinrt mingw-w64-ucrt-x86_64-curl-winssl \
  mingw-w64-ucrt-x86_64-MinHook mingw-w64-ucrt-x86_64-nsis \
  mingw-w64-ucrt-x86_64-openssl mingw-w64-ucrt-x86_64-opus \
  mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-nlohmann_json
```

Install Node.js from [nodejs.org](https://nodejs.org/) or nvm-windows. Do not use MSYS2's
Node package: its GCC 16 runtime currently fails during startup. `node.exe` and `npm` must be
on `PATH` when CMake configures the Web UI.

## Configure and build

From PowerShell:

```powershell
$env:MSYSTEM = "UCRT64"
$env:MSYS2_PATH_TYPE = "inherit"
C:\msys64\usr\bin\bash.exe -lc 'export PATH="/c/Program Files/nodejs:$PATH"; cd /e/Git/Repo/Apollo-3D; cmake -B cmake-build-relwithdebinfo -G Ninja -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo; ninja -C cmake-build-relwithdebinfo'
```

Use `RelWithDebInfo` for live XR testing. It retains symbols without the capture/encode stalls
of an unoptimized Debug build.

## Test

```powershell
C:\msys64\ucrt64\bin\ninja.exe -C cmake-build-relwithdebinfo test_sunshine
cmake-build-relwithdebinfo\tests\test_sunshine.exe
```

## Package

To enable offline conversion, install `ffmpeg.exe` and `ffprobe.exe` either beside
`sunshine.exe` or under a sibling `tools` directory. The application deliberately does not search
the user's `PATH`; arbitrary tool paths are not accepted from the Web UI. The FFmpeg build must
include the required demuxers/decoders, static-HDR filters, Matroska/MP4 muxers, and
NVIDIA `hevc_nvenc`/`av1_nvenc` encoders. Packagers are responsible for the selected FFmpeg
build's licensing and redistribution terms.

```powershell
cmake -B cmake-build-relwithdebinfo -G Ninja -S . `
  -DCMAKE_BUILD_TYPE=RelWithDebInfo `
  -DSUNSHINE_OFFLINE_FFMPEG=C:\approved-media-tools\ffmpeg.exe `
  -DSUNSHINE_OFFLINE_FFPROBE=C:\approved-media-tools\ffprobe.exe
ninja -C cmake-build-relwithdebinfo

cpack -G NSIS --config cmake-build-relwithdebinfo\CPackConfig.cmake
# or
cpack -G ZIP --config cmake-build-relwithdebinfo\CPackConfig.cmake
```

The two CMake paths must be absolute and supplied together. The build copies that exact pair
into `tools/`, and the package installs the same files. At startup, Sunshine inspects the packaged
FFmpeg encoder and filter listings without running NVENC. After a conversion job acquires the
exclusive offline GPU lease, Sunshine performs a one-frame 10-bit NVENC hardware preflight for
that job's selected codec.

See [../AGENTS.md](../AGENTS.md) for the development launch, Galaxy XR deployment, pairing,
and log-debugging loop.
