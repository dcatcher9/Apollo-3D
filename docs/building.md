# Building Apollo XR Host

Apollo supports Windows with native NVIDIA NVENC. CMake deliberately rejects non-Windows
targets; historical platform files remain only to make upstream comparisons easier.

## Requirements

- Windows 11
- NVIDIA GPU and current display driver
- MSYS2 UCRT64 with GCC, CMake, Ninja, cppwinrt, curl-winssl, MinHook, OpenSSL, Opus,
  nlohmann-json, NSIS, and the standard UCRT64 toolchain
- Official Node.js for the Vue Web UI

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

```powershell
cpack -G NSIS --config cmake-build-relwithdebinfo\CPackConfig.cmake
# or
cpack -G ZIP --config cmake-build-relwithdebinfo\CPackConfig.cmake
```

See [../AGENTS.md](../AGENTS.md) for the development launch, Galaxy XR deployment, pairing,
and log-debugging loop.
