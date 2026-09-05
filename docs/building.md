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

The NVIDIA Windows headers use the MSVC interface-vtable layout. After extracting a fresh package,
apply the repository's idempotent MinGW compatibility patch once before configuring:

```powershell
$BuildPython = "C:\absolute\path\to\python.exe"
& $BuildPython .\patch_trt.py C:\path\to\TensorRT
```

Re-running the command is safe and should report zero changed files. Apply it again whenever the
TensorRT package is replaced; do not reuse headers patched for a different TensorRT release.

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
export PATH="/ucrt64/bin:/c/Program Files/nodejs:$PATH"
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
  'export PATH="/ucrt64/bin:/c/Program Files/nodejs:$PATH"; export TENSORRT_DIR="/c/path/to/TensorRT"; cd /e/Git/Repo/Apollo-3D; cmake -B cmake-build-relwithdebinfo -G Ninja -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo; /ucrt64/bin/ninja -C cmake-build-relwithdebinfo'
```

The `/ucrt64/bin` entry must be present in the environment inherited by Ninja, compiler, and test
child processes. A login shell may rebuild `PATH`, so setting it only in the parent PowerShell is
not sufficient. A missing entry commonly appears as a compiler failure with no diagnostic or a
`cc1plus.exe` exit code of `0xC0000135`.

Use `RelWithDebInfo` for live XR testing. It preserves symbols without the capture, AI, and encode
stalls of an unoptimized Debug build. Build directories must use the `cmake-build-` prefix because
repository tooling uses it to distinguish generated artifacts.

## Test

Select the validated evaluator interpreter as `$SbsbenchPython` using the
[sbsbench runtime procedure](../tools/sbsbench/README.md#required-evaluation-loop).
The machine-local interpreter and DLL paths belong to [AGENTS.md](../AGENTS.md).

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;C:\path\to\TensorRT\bin;C:\path\to\TensorRT\lib;C:\Program Files\nodejs;$env:PATH"
C:\msys64\ucrt64\bin\ninja.exe -C cmake-build-relwithdebinfo test_sunshine
Push-Location cmake-build-relwithdebinfo
& .\tests\test_sunshine.exe
Pop-Location
& $SbsbenchPython -m unittest discover -s tools/sbsbench -p "test_*.py"
```

GPU-dependent Host SBS and offline-worker checks require a compatible NVIDIA host and are not
substitutes for the ordinary unit suite. Follow [CLAUDE.md](../CLAUDE.md) and
[sbsbench](../tools/sbsbench/README.md) for their controlled evaluation loop.
The [joint workflow gate](joint-workflow-tests.md) runs the focused host/client boundary suites
and documents the portable CI subset separately from GPU and device checks.

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
paths. The FFmpeg build must provide the codecs, rawvideo pipe formats, filters and muxers required
by [Offline Host 3D conversion](whole-clip-sbs-pipeline.md). The job performs its real NVENC
preflight only after acquiring the exclusive offline GPU lease.

## Package

```powershell
cpack -G NSIS --config cmake-build-relwithdebinfo\CPackConfig.cmake
# or
cpack -G ZIP --config cmake-build-relwithdebinfo\CPackConfig.cmake
```

Packagers are responsible for the TensorRT, model, FFmpeg, codec, and other third-party terms of
the exact artifacts they redistribute. See [Legal and redistribution](legal.md).
