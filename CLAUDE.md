# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Apollo is a Windows XR-streaming host — a hard fork of LizardByte's **Sunshine** that pairs
with modern **Artemis** clients on Android XR. It is a C++ application with a Vue 3 web UI for
configuration and pairing. The supported contract is deliberately narrow: Windows 11, native
NVIDIA NVENC, encrypted Artemis protocol 13, and one active remote session per host. It is not
compatible with legacy Sunshine/Moonlight clients or non-NVIDIA/non-Windows hosts.

Many internal names, the CMake project FQDN (`dev.lizardbyte.app.Sunshine`), the test binary
(`test_sunshine`), the npm package (`sunshine`), and most docs still say "Sunshine" — that is
expected, it is the upstream lineage. Apollo-specific additions include SudoVDA virtual
displays, host-side AI SBS conversion, local AR-glasses presentation, a per-client permission
system, clipboard sync, and warm reconnect of the single session.

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
via nvm-windows instead. See [docs/building.md](docs/building.md) for the Windows dependency
list (boost, cppwinrt, curl-winssl, MinHook, openssl, opus, nlohmann_json, …).

CMake logic lives under [cmake/](cmake) and is orchestrated from [CMakeLists.txt](CMakeLists.txt).
Historical non-Windows files remain in the tree for upstream comparison, but CMake rejects
non-Windows builds. Build options are defined in `cmake/prep/options.cmake`.

Package with CPack (`-G NSIS` or `ZIP`).

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
  `tools/sbsbench/bench.conf`, scores all metrics, and gates baseline-bearing clips against the
  committed baselines (`tools/sbsbench/baselines/` + `thresholds.json`). Deterministic core probes
  explicitly marked `evaluation_role: conformance-only` are decided by authenticated hard
  invariants and neither require nor write a numeric baseline. Every other role remains
  fail-closed on a missing baseline. **Exit 0 = pass, 1 = regression (named, with
  worst frame), 2 = setup error.** Results + provenance (git sha, models, clip hashes) land in
  `<build-dir>/sbs_eval/<label>/results.json`. After an INTENDED metric change, re-baseline with
  `--update-baselines` and commit the baselines together with the change. Supported A/B levers
  pass through, for example `--extra --pop-strength 1.25` or `--extra --depth-short-side 392`.
  GPU harnesses remain serial so their timing evidence is uncontended; authenticated CPU scoring
  is ordered but parallel across clips (`--jobs`, default up to eight, `--jobs 1` for the serial
  reference), and report remeasurement inherits the same job count.
  Changing `bench.conf` invalidates baselines; changing a baseline-bearing clip's source or
  scoring semantics invalidates that clip's baseline. Adding a conformance-only probe does not.
- **Adding a clip to the eval set**: a clip is just a directory of same-size numbered
  `frame_%05d.jpg` or lossless `frame_%05d.png` files under
  `tools/sbsbench/clips/<name>/` — `run_eval.py` auto-discovers it, no registration.
  From a video: `python tools/sbsbench/split_video.py video.mp4 -o tools/sbsbench/clips/<name>
  --width 854 --jpg --max 24` (854/24 matches the committed set's speed; full-res frames are also
  valid — the eval never resizes input, so a big clip is simply a slower, more sensitive eval).
  Synthetic/spliced clips: add a generator to `tools/sbsbench/make_synth_clips.py` (keeps the
  clip deterministic and licensing-free). Add `clips/<id>/meta.json` with `name`, `description`,
  an explicit `content_type`, and the applicable evidence role/flags (`evaluation_role`,
  `required_gt_*`, `expected_flat`, or `shot_state_contract`) — the report labels clips by name
  and `run_eval.py` copies the published fields into `results.json`. Clip identity hashes source
  pixels, authenticated GT/reference sidecars, and scoring semantics (`content_type`,
  `evaluation_role`, evidence requirements, dataset/depth kind, flat expectation, and shot-state
  contract). Human-readable `name`, `description`, and provenance wording remain deliberately
  excluded, so editing only those labels does not invalidate a baseline. Pick content that
  isolates ONE failure mode (see the
  clip table in the README). Then run `run_eval.py --update-baselines`. For a baseline-bearing
  clip, **commit the frames, the generator change (if any), and the new
  `baselines/<name>.json` together** and sanity-check its numbers/worst frames. For a
  `conformance-only` probe, commit the frames, generator, source-authentication contract, and hard
  invariant instead; `--update-baselines` intentionally writes no baseline for it. A mis-sized or
  mis-ordered ordinary clip shows up as nonsense metrics, not an error.
- **Visual** — the headless frame-fed harness `sunshine --sbs-bench` (implemented in
  [src/sbs_bench_harness.cpp](src/sbs_bench_harness.cpp)): runs the real depth estimator + real
  composite shaders over a fixed directory of frames (split a short video with
  `tools/sbsbench/split_video.py`), writing `sbs_%05d.png` + `depth_%05d.png`, deterministically
  and with no game/client. Score with `python tools/sbsbench/sbsbench.py --seq <out> --baseline
  base.json` → `exact_visible_pop_spread_pct` (near-to-far stereo VOLUME, subject-anchoring-fair),
  vmisalign (geometry, must stay ~0), disocc_frac/disocc_smear (disocclusion severity), flicker
  (temporal shimmer). **A green exit code does NOT mean pop is safe.** Baseline regressions are
  gated only for `role: primary` metrics (`run_eval.py`), plus the non-tradeable `role: hard`
  bounds. Stereo-volume and stretch/fold/shear metrics are `role: diagnostic` and are reported,
  not gated — see `tools/sbsbench/thresholds.json` for each metric's role. Two traps:
  `exact_visible_pop_spread_pct` is a p0.5..p99.5 spread and therefore REWARDS clipping, so read
  it beside `exact_disparity_plateau_*` (fraction at an extreme, and the depth span each plateau
  swallowed — span ~0 is legitimate flat geometry, a wide span is destroyed relief); and
  `depth_gt_edge_f1` only gates where `depth_gt_edge_support_pct` >= 1.0, because it scores
  thresholded edge sets and swings wildly on weak-gradient content. Raising pop typically
  keeps the ±3% comfort tails inside their hard bound while degrading the ungated artifact
  diagnostics, so read those by hand. Capture a baseline before the change; `--pop-strength`,
  `--depth-short-side`, `--ema`, `--minmax-ema`, and the subject
  recenter/stretch controls are the supported A/B levers (see the harness
  README). Run
  it from `cmake-build-relwithdebinfo` so `assets/` resolves.
- **Whole-clip analysis/conversion** — `tools/sbsbench/run_whole_clip.py` keeps one native Host SBS
  estimator alive across a complete video or numbered frame sequence, exports cut, depth-health,
  per-frame, and per-shot evidence, and can encode a packed-SBS H.265/HEVC or AV1 video. It runs
  the production Depth Coordinate V2 pipeline (`--parallax-v2-live`: `host_sbs_v2` estimator,
  signed final-parallax scene cache schema 2, shared contractive-inverse renderer); legacy
  geometry levers are rejected and remain evaluator-only in `run_eval.py`, which stays the
  authenticated regression gate. The offline path synchronously completes each requested
  inference and is therefore a deterministic current-frame quality oracle, not an exact replay of
  live Host SBS's bounded asynchronous drop/repeat timing. See
  [docs/whole-clip-sbs-pipeline.md](docs/whole-clip-sbs-pipeline.md).
- **Perf** — the global `diagnostics = enabled` config knob ([src/sbs_perf.cpp](src/sbs_perf.cpp));
  shipped config defaults it off and `E:\ApolloDev\config\sunshine.conf` enables it. It logs a rolling
  p50/p95/max summary for inference, depth pre/postprocess, matched-copy, warp, output-pass,
  frame-age, and CPU submission costs every 300 SBS frames. Live diagnostics perform no JSON I/O;
  the offline harness explicitly writes `sbs_perf.json` and records its additional harness stages.

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
- **HTTP / pairing / config**: [src/nvhttp.cpp](src/nvhttp.cpp) (Artemis GameStream HTTP),
  [src/confighttp.cpp](src/confighttp.cpp) (web-UI config API), [src/httpcommon.cpp](src/httpcommon.cpp),
  [src/config.cpp](src/config.cpp), [src/crypto.cpp](src/crypto.cpp).
- **Encoding**: [src/nvenc/](src/nvenc) contains the native D3D11 NVENC path. H.264, HEVC,
  and AV1 parameter sets and color metadata are produced directly by NVENC; Apollo no longer
  links the generic FFmpeg/CBS encoder stack.
- **Process/app launching & display**: [src/process.cpp](src/process.cpp),
  [src/display_device.cpp](src/display_device.cpp).
- **Windows platform**: [src/platform/windows/](src/platform/windows) contains DXGI/WGC capture,
  audio, input, local AR presentation, and the SudoVDA virtual display
  ([src/platform/windows/virtual_display.cpp](src/platform/windows/virtual_display.cpp)). The
  historical Linux/macOS source directories are retained only for upstream comparison and are
  not part of the supported build.

Vendored dependencies are git submodules under [third-party/](third-party) (Simple-Web-Server,
moonlight-common-c, googletest, inputtino, ViGEmClient, sudovda, nvapi/nvfbc/nv-codec-headers,
etc.). Clone with `--recurse-submodules`.

## Conventions

- C/C++ must follow [.clang-format](.clang-format) (LLVM-based, 2-space access offset, block-indent
  brackets). Format before committing.
- Python tooling is linted with flake8 ([.flake8](.flake8)); JS/web with prettier
  ([.prettierrc.json](.prettierrc.json)).
- This is GPLv3. Upstream user docs live under [docs/](docs) and on Read the Docs; Apollo-specific
  behavior (client permissions, sole-session virtual display, and local AR presentation) is documented in the project Wiki
  referenced from [README.md](README.md).
