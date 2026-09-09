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

The optional virtual DualSense backend has a separate .NET 10 helper build and
pinned HIDMaestro runtime. Follow the [helper build and runtime setup](../tools/sunshine-ds5-sidecar/README.md)
to build its local component and inspect driver readiness. The native host builds
without that component, and ordinary controller support does not require .NET.

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

TensorRT engine caches bind the model and build recipe to the TensorRT version, CUDA driver API
version, GPU name, and checked hardware properties such as compute capability and shared-memory
limits. Reported total/free GPU memory and the Windows driver package version are excluded:
changes in memory reservations do not by themselves require an engine rebuild. A change to a keyed
property selects a new cache automatically at the next process start.
The device identity is captured once per process so display changes cannot trigger a rebuild in a
running stream. A failed identity query prevents cache selection rather than falling back to an
ambiguous filename. Engine filenames use bounded SHA-256 identities; active-engine manifests retain
the model, recipe, and authenticated source hashes. Old cache files are not overwritten or edited
to suppress compatibility warnings. TensorRT's own device-property warnings remain visible; a
warning alone does not establish a functional incompatibility. See NVIDIA's
[engine compatibility checks](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html#compatibility-checks).

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

The Windows installer defaults to `C:\Program Files\Sunshine3D`. It retains Apollo's
upgrade registration and service identity so existing installations are replaced through the
normal upgrade prompt. When changing directories, setup copies the previous installed
configuration into an unused destination config directory before starting the service; existing
destination settings are retained. The original configuration is kept in its old directory.
Development settings outside the installation, such as `E:\ApolloDev\config`, are not imported.

TensorRT remains fully bundled, including all GPU architecture resources; setup does not download
it. The default package installs the SDK DLLs directly with ordinary NSIS LZMA compression. To
share repeated data across the complete set of 14 TensorRT 11 DLLs, enable the local archive mode:

```powershell
cmake -B cmake-build-relwithdebinfo -G Ninja -S . `
  -DSUNSHINE_PACKAGE_TENSORRT_ARCHIVE=ON `
  '-DSUNSHINE_7ZIP_EXECUTABLE=C:/Program Files/7-Zip/7z.exe'
```

Archive mode requires an existing Windows x64 7-Zip installation with `7z.dll` and `License.txt`
beside the executable. CMake compresses all 14 DLLs with solid LZMA2, a 512 MiB dictionary and four
threads, caching the result under the build directory's `tensorrt-package`. This can take several
minutes on first configuration. `SUNSHINE_TENSORRT_ARCHIVE_FILE` can instead point to an existing
local archive; CMake verifies its exact file membership and every extracted hash against the SDK.
Subsequent configurations reuse a valid cache, and packaging rejects inputs changed after
configuration. No SDK files are omitted or modified, and development runtime copies are unchanged.

The outer NSIS package uses non-solid zlib in archive mode to avoid NSIS's 2 GiB uncompressed solid
block limit. Setup expands the local archive and verifies every DLL before starting the host;
decompression needs approximately 512 MiB of additional memory. The unmodified 7-Zip executable,
DLL, license and attribution are bundled under `tools/tensorrt/7zip`. Its license includes LGPL,
BSD terms and the unRAR restriction; source code is available from [7-Zip](https://www.7-zip.org/).
`SUNSHINE_PACKAGE_TENSORRT_ARCHIVE` defaults to `OFF` and never downloads build tools or TensorRT.

To include the optional microphone and DualSense setup components, first build the helper in a
fresh directory with a .NET 10 SDK, then supply it to CMake. These are build-time dependencies;
the installer includes a self-contained .NET runtime for end users.

```powershell
.\scripts\build-ds5-sidecar.ps1 `
  -Dotnet C:\path\to\dotnet-10-sdk\dotnet.exe `
  -OutputDirectory C:\package-inputs\ds5-component
cmake -B cmake-build-relwithdebinfo -G Ninja -S . `
  -DSUNSHINE_PACKAGE_CLIENT_DRIVERS=ON `
  -DSUNSHINE_DS5_COMPONENT_DIR=C:\package-inputs\ds5-component
cmake --build cmake-build-relwithdebinfo
```

`SUNSHINE_PACKAGE_CLIENT_DRIVERS` defaults to `OFF` for native-only development builds. Enabling
it requires a complete helper manifest with `installer_driver_setup = 1`; CMake verifies all files
at configuration and packaging time. Rebuild the helper after managed source changes. CMake
downloads the hash-pinned VB-CABLE Pack45 ZIP, or accepts its exact local copy through
`SUNSHINE_VBCABLE_ARCHIVE`. The original package and attribution are included intact.

The DualSense payload deliberately omits HIDMaestro Core. When the user selects that component,
setup downloads the pinned upstream archive directly and verifies it before installation. The
complete upstream DLL embeds Microsoft SDK/WDK tools whose redistribution rights have not been
established. Do not add the local complete component or upstream archive to release assets.

```powershell
cpack -G NSIS --config cmake-build-relwithdebinfo\CPackConfig.cmake
# For a portable ZIP with TensorRT DLLs already beside sunshine.exe:
cmake -B cmake-build-relwithdebinfo -DSUNSHINE_PACKAGE_TENSORRT_ARCHIVE=OFF
cpack -G ZIP --config cmake-build-relwithdebinfo\CPackConfig.cmake
```

NSIS runs optional setup only for selected components, before starting the host service. Both
components are unchecked on a fresh install. For silent installation from an elevated terminal:

```powershell
$installer = Start-Process .\Sunshine3D.exe -ArgumentList '/S', '/CLIENTDRIVERS' -WindowStyle Hidden -Wait -PassThru
$installer.ExitCode
```

`/CLIENTDRIVERS` selects both optional components; `/MICROPHONE` and `/DUALSENSE` select them
independently. `/S` alone retains ordinary component selection. VB-CABLE's pinned `-i -h` mode was
verified by static inspection of its signed binary; details are shipped in `VB-CABLE-SETUP-CONTRACT.md`.
Driver failures return `1` while allowing the main application to finish installing. A successful new
VB-CABLE installation sets the reboot flag and returns `3010`; other success returns `0`. Setup does
not reboot automatically. See [feature setup](host-client-features.md)
for permissions, host settings, logs, and physical qualification.

ZIP packaging includes the same scripts and incomplete DualSense payload but does not execute
them. Disable `SUNSHINE_PACKAGE_TENSORRT_ARCHIVE` before creating a portable ZIP; the archive mode
depends on NSIS running the extraction script before Sunshine starts. For manual optional-driver
setup, stop Sunshine and run the scripts as Administrator, passing
`-PackageDirectory <root>/drivers/vbcable` to `scripts/install-vbcable.ps1`, or
`-PayloadDirectory <root>/tools/sunshine-ds5-component/payload -ComponentRoot <root>/tools/sunshine-ds5-component`
to `scripts/install-dualsense.ps1`; both accept `-LogDirectory <root>/config/driver-setup`.
The DualSense destination must be protected against writes by unprivileged users.

Installer regression checks use Windows PowerShell 5.1 with an already installed Pester 3/4:

```powershell
ctest --test-dir cmake-build-relwithdebinfo/tests -R '^driver_installer_unit_tests$' --no-tests=error --output-on-failure
```

These tests mock driver installation and downloads. They cover readiness, silent invocation,
integrity/path rejection, partial failures and component rollback; they do not replace an elevated
Windows installation test.

Packagers are responsible for the TensorRT, model, FFmpeg, codec, and other third-party terms of
the exact artifacts they redistribute. See [Legal and redistribution](legal.md).
