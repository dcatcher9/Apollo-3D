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

## SBS 3D quality / shader work — use the real pipeline evaluator

The SBS 3D feature's status, known residual artifacts, hard-won constraints, and the
prioritized roadmap live in [docs/sbs-3d-roadmap.md](docs/sbs-3d-roadmap.md) — read it
before planning any SBS work.

The old CPU warpsim was removed because it modeled deleted legacy branches and drifted from the
real D3D shaders. Validate every change with `tools/sbsbench/run_eval.py`: it runs the production
estimator and shaders on both eyes over multiple temporal scenes and generates the report.

### Quantify every SBS change with the two host benchmarks (offline, real pipeline)

To evaluate what a change actually does to the shipped pipeline, ALWAYS use the two host
benchmarks — they run the
real estimator + real shaders and produce numbers to diff against a baseline. Do not judge an
SBS change by eyeballing the headset; produce the before/after numbers. See
[docs/sbs-benchmark-plan.md](docs/sbs-benchmark-plan.md) and
[tools/sbsbench/README.md](tools/sbsbench/README.md).

- **One command runs the whole loop**: `python tools/sbsbench/run_eval.py` — harnesses every
  committed clip (`tools/sbsbench/clips/`) through the real pipeline with the pinned
  `tools/sbsbench/bench.conf`, scores all metrics, and gates against the committed baselines
  (`tools/sbsbench/baselines/` + `thresholds.json`). **Exit 0 = pass, 1 = regression (named, with
  worst frame), 2 = setup error.** Results + provenance (git sha, models, clip hashes) land in
  `<build-dir>/sbs_eval/<label>/results.json`. After an INTENDED metric change, re-baseline with
  `--update-baselines` and commit the baselines together with the change. Supported A/B levers
  pass through, for example `--extra --pop-strength 1.25` or `--extra --subject-lock 0.6`.
  Changing bench.conf or the clip set
  invalidates baselines.
- **Adding a clip to the eval set**: a clip is just a directory of same-size `frame_%05d.jpg`
  frames under `tools/sbsbench/clips/<name>/` — `run_eval.py` auto-discovers it, no registration.
  From a video: `python tools/sbsbench/split_video.py video.mp4 -o tools/sbsbench/clips/<name>
  --width 854 --jpg --max 24` (854/24 matches the committed set's speed; full-res frames are also
  valid — the eval never resizes input, so a big clip is simply a slower, more sensitive eval).
  Synthetic/spliced clips: add a generator to `tools/sbsbench/make_synth_clips.py` (keeps the
  clip deterministic and licensing-free). Add a `clips/<id>/meta.json` = `{"name": "<scene-name>",
  "description": "..."}` — the report labels clips by that name and run_eval copies it into
  results.json (the frame content, not meta.json, defines the clip's identity hash, so editing the
  name never invalidates a baseline). Pick content that isolates ONE failure mode (see the
  clip table in the README). Then run `run_eval.py --update-baselines` and **commit the frames,
  the generator change (if any), and the new `baselines/<name>.json` together**; sanity-check the
  new clip's baseline numbers and worst frames before committing (a mis-sized or mis-ordered clip
  shows up as nonsense metrics, not an error).
- **Visual** — the headless frame-fed harness `sunshine --sbs-bench` (implemented in
  [src/sbs_bench_harness.cpp](src/sbs_bench_harness.cpp)): runs the real depth estimator + real
  composite shaders over a fixed directory of frames (split a short video with
  `tools/sbsbench/split_video.py`), writing `sbs_%05d.png` + `depth_%05d.png`, deterministically
  and with no game/client. Score with `python tools/sbsbench/sbsbench.py --seq <out> --baseline
  base.json` → pop_spread (near-to-far stereo VOLUME, the gated pop metric — subject-anchoring-fair,
  unlike median-|dx| pop_px which is reported-only), vmisalign (geometry, must stay ~0),
  disocc_frac/disocc_smear (disocclusion severity), flicker (temporal shimmer, which the offline
  sim can't measure). Capture a baseline before the change; `--pop-strength`,
  `--depth-short-side`, `--ema`, `--minmax-ema`, and the subject
  lock/recenter/stretch/plane-lock controls are the supported A/B levers (see the harness
  README). Run
  it from `cmake-build-relwithdebinfo` so `assets/` resolves.
- **Perf** — the in-app `sbs_3d_perf_stats = enabled` config knob ([src/sbs_perf.cpp](src/sbs_perf.cpp))
  logs live `depth_infer` (CUDA events) and `sbs_convert_cpu` every 300 SBS frames and writes
  `sbs_perf.json`. The offline harness additionally records `warp_infer` with correctly ordered
  D3D11 timestamp queries, so a `--sbs-bench` run yields the visual scorecard and warp cost.

When these tools change (new metric, harness contract, or a shader they replicate), update their
READMEs and this section so the workflow stays discoverable.

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
