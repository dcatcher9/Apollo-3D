# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Apollo is a self-hosted game-stream host — a hard fork of LizardByte's **Sunshine** that
pairs with the **Artemis** (Moonlight Noir) client. It is a C++ application (the streaming
server) with a Vue 3 web UI for configuration and client pairing. The fork has diverged from
upstream Sunshine and is not intended to stay compatible with OG Sunshine/Moonlight.

Many internal names, the CMake project FQDN (`dev.lizardbyte.app.Sunshine`), the test binary
(`test_sunshine`), the npm package (`sunshine`), and most docs still say "Sunshine" — that is
expected, it is the upstream lineage. Apollo-specific additions include the built-in virtual
display (SudoVDA, Windows-only), a per-client permission system, and clipboard sync.

## Build (Windows / MSYS2 UCRT64)

This repo is developed on Windows with MSYS2 + UCRT64. Prefix every build command with the
MSYS2 shell wrapper so it runs in the ucrt64 environment:

```
C:\msys64\msys2_shell.cmd -defterm -here -no-start -ucrt64 -c "<command>"
```

Name build directories with a `cmake-build-` prefix (e.g. `cmake-build-release`).

Configure and build:

```
cmake -B cmake-build-release -G Ninja -S .
ninja -C cmake-build-release
```

`node.exe` must be on `PATH` before running CMake — the `web-ui` target invokes `npm install`
via `find_program(NPM npm)`. **Do not** use MSYS2's `mingw-w64-ucrt-x86_64-nodejs`; it crashes
with `std::bad_weak_ptr` under the gcc-16 libstdc++. Install official Node.js from nodejs.org or
via nvm-windows instead. See [docs/building.md](docs/building.md) for the full dependency list
(boost, cppwinrt, curl-winssl, MinHook, miniupnpc, onevpl, openssl, opus, nlohmann_json, …) and
for Linux/macOS instructions.

CMake logic lives under [cmake/](cmake) and is orchestrated from [CMakeLists.txt](CMakeLists.txt):
`prep/` (options, version, package config), `dependencies/`, `compile_definitions/`,
`targets/` (per-platform: `windows.cmake`, `linux.cmake`, `macos.cmake`, `unix.cmake`),
`packaging/`. Build options are defined in `cmake/prep/options.cmake`.

Package with CPack (`-G NSIS`/`ZIP` on Windows, `DEB`/`RPM` on Linux, `DragNDrop` on macOS).

## Tests

GoogleTest. The test executable is `test_sunshine`, built into the `tests/` directory of the
build dir. Tests are configured in [tests/CMakeLists.txt](tests/CMakeLists.txt) and compiled with
coverage flags (`-fprofile-arcs -ftest-coverage -O0`). Source is globbed from `tests/**`, and
`main.cpp` is excluded from the app sources so the test main ([tests/tests_main.cpp](tests/tests_main.cpp))
can take over.

Run all tests:

```
ninja -C cmake-build-release test_sunshine
cmake-build-release/tests/test_sunshine.exe
```

Run a single test or group via the standard gtest filter:

```
cmake-build-release/tests/test_sunshine.exe --gtest_filter=SuiteName.TestName
```

Unit tests are in [tests/unit/](tests/unit) (one `test_<module>.cpp` per source module, e.g.
`test_video.cpp`, `test_stream.cpp`, `test_http_pairing.cpp`), with integration tests in
[tests/integration/](tests/integration) and shared HTTP fixtures in [tests/fixtures/](tests/fixtures).

## SBS 3D quality / shader work — use the offline simulator FIRST

The SBS 3D feature's status, known residual artifacts, hard-won constraints, and the
prioritized roadmap live in [docs/sbs-3d-roadmap.md](docs/sbs-3d-roadmap.md) — read it
before planning any SBS work.

Before changing the SBS reprojection shader or debugging visual artifacts in the 2D→3D
pipeline, read [tools/warpsim/README.md](tools/warpsim/README.md) and follow its workflow:
capture a frame with the client's "Dump 3D" button, reproduce the artifact in the CPU
simulator (it replicates `sbs_reprojection_ps.hlsl` pixel-exactly), test theories with the
instrumentation tools and numbers, and validate candidate changes offline — on BOTH eyes and
MULTIPLE scenes — before deploying anything to the headset. Do NOT iterate guess→deploy→
headset-check; that loop has repeatedly failed here. When `sbs_reprojection_ps.hlsl` changes,
update the simulator's replica functions to match.

## Web UI (Vue 3 + Vite)

The config web UI is a separate Vite build. Sources are in
[src_assets/common/assets/web/](src_assets/common/assets/web) (multi-page: `index`, `config`,
`apps`, `pin`, `password`, `welcome`, `login`, `troubleshooting` — each an `.html` entry in
[vite.config.js](vite.config.js)) and output to `build/assets/web`. During a normal CMake build
the `web-ui` target drives this; CMake passes the source/dest dirs via the
`SUNSHINE_SOURCE_ASSETS_DIR` / `SUNSHINE_ASSETS_DIR` env vars.

npm scripts (from [package.json](package.json)):

- `npm run build` — one-off Vite build
- `npm run dev` — `vite build --watch` for iterating on the UI
- `npm run serve` — serve the HTTP fixtures for local UI testing

## Code layout

Core C++ server lives flat in [src/](src). Major subsystems:

- **Streaming pipeline**: [src/stream.cpp](src/stream.cpp), [src/rtsp.cpp](src/rtsp.cpp),
  [src/video.cpp](src/video.cpp), [src/audio.cpp](src/audio.cpp), [src/input.cpp](src/input.cpp) —
  RTSP control plus the video/audio/input data planes.
- **HTTP / pairing / config**: [src/nvhttp.cpp](src/nvhttp.cpp) (Moonlight protocol HTTP),
  [src/confighttp.cpp](src/confighttp.cpp) (web-UI config API), [src/httpcommon.cpp](src/httpcommon.cpp),
  [src/config.cpp](src/config.cpp), [src/crypto.cpp](src/crypto.cpp).
- **Encoding**: [src/nvenc/](src/nvenc) (NVENC, D3D11 / CUDA paths), plus CBS bitstream helpers
  ([src/cbs.cpp](src/cbs.cpp)) and colorspace ([src/video_colorspace.cpp](src/video_colorspace.cpp)).
- **Process/app launching & display**: [src/process.cpp](src/process.cpp),
  [src/display_device.cpp](src/display_device.cpp).
- **Platform abstraction**: [src/platform/](src/platform) with `common.h` and per-OS implementations
  under `windows/`, `linux/`, `macos/`. Capture backends (e.g. `display_wgc`, `display_vram`,
  `kmsgrab`, `x11grab`, `wlgrab`, `vaapi`), audio, input, and the Windows-only virtual display
  ([src/platform/windows/virtual_display.cpp](src/platform/windows/virtual_display.cpp)) all live here.

Vendored dependencies are git submodules under [third-party/](third-party) (Simple-Web-Server,
moonlight-common-c, googletest, inputtino, ViGEmClient, sudovda, nvapi/nvfbc/nv-codec-headers,
etc.). Clone with `--recurse-submodules`.

## Conventions

- C/C++ must follow [.clang-format](.clang-format) (LLVM-based, 2-space access offset, block-indent
  brackets). Format before committing.
- Python tooling is linted with flake8 ([.flake8](.flake8)); JS/web with prettier
  ([.prettierrc.json](.prettierrc.json)).
- This is GPLv3. Upstream user docs live under [docs/](docs) and on Read the Docs; Apollo-specific
  behavior (permission system, virtual display, multi-instance) is documented in the project Wiki
  referenced from [README.md](README.md).
