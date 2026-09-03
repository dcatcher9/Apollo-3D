# Repository development guide

This file contains the repository-wide workflow for coding agents and contributors. It deliberately
does not duplicate algorithm constants, evaluator metric definitions, or user configuration; those
belong to the linked canonical documents.

## Product and platform

Sunshine 3D is a Windows XR-streaming host paired with modern Moonlight 3D/Artemis clients on
Android XR. It is a hard fork of Apollo and LizardByte Sunshine. The supported product contract is
intentionally narrow:

- Windows 11;
- native NVIDIA NVENC;
- encrypted Artemis protocol 13;
- one active remote session per host; and
- host, client, authored-SBS, and local-AR presentation modes documented in [README.md](README.md).

Internal names such as `Apollo`, `sunshine.exe`, `test_sunshine`, the CMake project FQDN, service
identifiers, and configuration paths remain for upstream and protocol compatibility.

## Build on Windows

Use MSYS2 UCRT64 and a build directory whose name starts with `cmake-build-`. Official Node.js must
be on `PATH` before CMake runs because the `web-ui` target invokes npm. Do not install MSYS2's
UCRT64 Node package; it is incompatible with the current GCC/libstdc++ combination.

From a UCRT64 shell:

```bash
export PATH="/c/Program Files/nodejs:$PATH"
cmake -B cmake-build-relwithdebinfo -G Ninja -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja -C cmake-build-relwithdebinfo
```

Use `RelWithDebInfo` for live XR testing. An unoptimized Debug build is too slow for a meaningful
streaming test. The complete dependency and packaging instructions are in
[Building Sunshine 3D](docs/building.md).

## Tests

Build and run the native test executable:

```bash
ninja -C cmake-build-relwithdebinfo test_sunshine
cmake-build-relwithdebinfo/tests/test_sunshine.exe
```

Run a focused group with the standard GoogleTest filter:

```bash
cmake-build-relwithdebinfo/tests/test_sunshine.exe --gtest_filter=SuiteName.TestName
```

Unit tests are under `tests/unit`, integration tests under `tests/integration`, and shared fixtures
under `tests/fixtures`. Python evaluator tests run with:

```powershell
python -m unittest discover -s tools/sbsbench -p "test_*.py"
```

## Host SBS changes

Read these owners before changing Host SBS:

- [Host SBS pipeline](docs/host-sbs.md) owns the live/offline geometry and HDR contract.
- [Host SBS scene cuts](docs/host-sbs-scene-cuts.md) owns cut evidence, state, and acceptance.
- [SBS 3D status and roadmap](docs/sbs-3d-roadmap.md) owns current limitations and active work.
- [sbsbench](tools/sbsbench/README.md) owns the evaluation workflow and links its non-overlapping
  [dump](tools/sbsbench/DUMP_FORMAT.md), [dataset](tools/sbsbench/DATASETS.md),
  [metric](tools/sbsbench/METRICS.md), and [oracle](tools/sbsbench/ORACLES.md) contracts.

The CPU warpsim and pre-V2 renderer were removed. Do not validate a shader change with a
hand-maintained replica or treat headset inspection as the regression gate.

### Required evaluation loop

Build the current binary, then capture an authenticated control with the real estimator and D3D
shaders. The V1 baseline set was retired with its renderer, so ordinary development starts with a
matched comparison rather than treating an old baseline as a verdict:

```powershell
python tools/sbsbench/run_eval.py --comparison-only --label control
```

Exit codes are `0` for pass, `1` for a named regression, and `2` for setup/provenance failure.
Results are written below `cmake-build-relwithdebinfo/sbs_eval/`. Capture a control before a
geometry or metric change, then bind the treatment to it with `--report-control`. Publish a new V2
baseline only when its output and scoring semantics have been inspected and justified; commit the
source, contract, clip, and baseline changes together.

Useful controlled levers are passed after `--extra`, for example:

```powershell
python tools/sbsbench/run_eval.py --comparison-only --label pop-1p0 --extra --pop-strength 1.0
```

The GPU harness remains serial so performance evidence is uncontended. CPU scoring can use
`--jobs`; use `--jobs 1` only when reproducing the serial reference. Run commands from the
RelWithDebInfo build working directory when a direct harness invocation needs relative `assets/`.

For a live-only failure, capture a current **Dump 3D** package and align evidence by exact source
frame identity. Use floating-point artifacts for numeric comparisons; stretched preview PNGs are
not an absolute-depth contract. Production telemetry is nonblocking and may skip samples under
load, while offline traces may be complete.

Performance evidence comes from `diagnostics = enabled`. Live diagnostics remain bounded and do
not write per-frame JSON; the offline harness writes its explicit `sbs_perf.json` snapshot.

### Offline conversion

The Web UI conversion path uses an in-process job manager and an isolated `sunshine.exe` worker,
not Python or a separate Windows service. Its causal one-pass media, HDR, cancellation, and packaging
contract is in [Offline Host 3D conversion](docs/whole-clip-sbs-pipeline.md). There is no maintained
second whole-clip policy in `tools/sbsbench`; evaluator utilities remain diagnostics only.

## Web UI

The Vue 3/Vite sources are under `src_assets/common/assets/web`. The normal CMake build drives the
`web-ui` target. For direct iteration, use the npm scripts in `package.json`:

```bash
npm run build
npm run dev
npm run serve
```

Use a scoped localhost HTTP server for rendered testing. Do not ask browser automation to open
`file://` URLs.

## Code layout

- Streaming data planes: `src/stream.cpp`, `src/rtsp.cpp`, `src/video.cpp`, `src/audio.cpp`, and
  `src/input.cpp`.
- HTTP, pairing, and configuration: `src/nvhttp.cpp`, `src/confighttp.cpp`,
  `src/httpcommon.cpp`, `src/config.cpp`, and `src/crypto.cpp`.
- NVIDIA encoding: `src/nvenc/`.
- App and display lifecycle: `src/process.cpp`, `src/display_device.cpp`, and
  `src/platform/windows/virtual_display.cpp`.
- Windows capture, input, audio, local AR, and virtual display: `src/platform/windows/`.
- Host SBS model, state, shaders, and harness: `src/video_depth_estimator.*`,
  `src_assets/windows/assets/shaders/directx/`, and `src/sbs_bench_harness.*`.

Historical non-Windows source remains only for upstream comparison; CMake rejects non-Windows
builds. Vendored dependencies are submodules under `third-party/`; clone recursively.

## Conventions

- Follow `.clang-format` for C/C++, `.flake8` for Python, and `.prettierrc.json` for web assets.
- Preserve unrelated working-tree changes. Do not reset or rewrite user-owned work.
- Keep configuration, Web UI defaults, localization, generated contracts, and tests consistent.
- Update the single owning document when a contract changes; link to it instead of copying its
  thresholds or schema into another Markdown file.
- This repository is GPL-3.0-only.
