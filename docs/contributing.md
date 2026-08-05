# Contributing to Sunshine 3D

Sunshine 3D is a Windows/NVIDIA XR-streaming fork with a deliberately narrower product contract
than upstream Sunshine. A useful change should preserve that boundary and include evidence
appropriate to its risk.

## Before editing

1. Read [CLAUDE.md](../CLAUDE.md) for the repository workflow and build commands.
2. Read the one document that owns the contract being changed. In particular:
   - [Host SBS pipeline](host-sbs.md) owns live/offline V2 geometry and HDR;
   - [Host SBS scene cuts](host-sbs-scene-cuts.md) owns cut evidence and state;
   - [Offline Host 3D conversion](whole-clip-sbs-pipeline.md) owns the native job workflow; and
   - [sbsbench](../tools/sbsbench/README.md) owns evaluation artifacts and gates.
3. Preserve unrelated working-tree changes. Do not reset or reformat files outside the change.

Do not reintroduce Linux/macOS host behavior, AMD/Intel/software encoders, the Host SBS V1
renderer, a CPU depth fallback, or duplicate algorithm policy in a second document.

## Build and test

Use the Windows MSYS2 UCRT64 `RelWithDebInfo` build described in [Building](building.md):

```bash
export PATH="/c/Program Files/nodejs:$PATH"
cmake -B cmake-build-relwithdebinfo -G Ninja -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja -C cmake-build-relwithdebinfo
```

Run the relevant native tests:

```powershell
cmake-build-relwithdebinfo\tests\test_sunshine.exe
```

Run evaluator tests when `tools/sbsbench` changes:

```powershell
python -m unittest discover -s tools/sbsbench -p "test_*.py"
```

Host SBS changes also require a matched GPU A/B through the current evaluator and headset review.
The exact qualification workflow is in [sbsbench](../tools/sbsbench/README.md). A visual change is
not accepted from a stretched debug PNG alone.

## Change discipline

- Follow `.clang-format`, `.flake8`, and `.prettierrc.json` for their respective languages.
- Keep generated HLSL/C++ contracts synchronized with their canonical JSON manifest and generator.
- Update configuration parsing, Web UI controls, English localization, generated/reference docs,
  and tests together when a public option changes.
- Add tests for observable behavior and contract boundaries. Avoid source-text tombstones whose
  only assertion is that deleted V1 code remains absent.
- Use explicit provenance for benchmark evidence. Do not regenerate baselines from a dirty or
  unauthenticated build.
- Update only the owning Markdown document and link to it from other guides.

## Web UI

The Vue/Vite Web UI lives in `src_assets/common/assets/web`. The CMake build runs its production
build. For focused UI work:

```bash
npm run build
npm run dev
```

Add English text to `public/assets/locale/en.json`; do not hand-edit generated translations. Test
rendered pages through a scoped `http://localhost` development server rather than a `file://` URL.

## Pull-request evidence

Describe:

- the user-visible problem and supported mode;
- the root cause;
- the files and contracts intentionally changed;
- native/Python/Web UI tests run;
- GPU evaluator control and treatment identifiers for Host SBS work; and
- remaining limitations or headset checks.

Sunshine 3D is licensed under GPL-3.0-only. Contributions must be compatible with
[the repository license](../LICENSE).
