# Depth model roster + on-the-fly switching — implementation plan

> **Historical implementation plan.** Model switching shipped, while the warpsim and guided
> processor workflow referenced below was subsequently removed. Use `sbs-3d-roadmap.md` and
> `tools/sbsbench/README.md` for current validation.
>
> **STATUS (2026-07-05): A + B SHIPPED and headset-verified. C partly done but its
> conclusions changed the direction — read this first:**
> - **Model switching (B) is live**: registry, `g_engines` per-name TRT slots, `0x3005`
>   control message, XR "Model" tile, startup prebuild, recipe-versioned engine files.
>   Client pushed to dcatcher9 forks; host uncommitted.
> - **DA-V2 base (A): rejected** — only ~11% more relief than small, headset-confirmed
>   marginal, not worth the cost.
> - **DA-V3 (C): the onnx-community small/base/large exports all give FLAT monocular pop**
>   (compressed depth, pop ~0.55–0.66). The confidence-guided-upsample idea (C4) is moot
>   (guided upsampling is measured un-tunable for V3, and is bypassed under MLBW anyway).
>   The depth transform is now iw3's **shifted reciprocal** `1/(depth+sbs_3d_depth_shift)`;
>   all the `1/depth` + sigma-clip/norm_sigma band-aids were **removed**.
> - **The real pop fix is DA3MONO-LARGE** (monocular-specialized DA-V3, 0.35B, pop
>   0.74–0.90 ≈ V2), which needs a torch→ONNX export — new roadmap item #2.
> - **The real edge fix is the learned warp + inpaint (MLBW stash)** — roadmap item #1.
>
> So this doc's original C4/C5 (confidence/focal-anchored) are superseded; the live model
> registry + switching machinery it describes is correct and in use.

Execution plan for: (A) validating **DA-V2-base** as a drop-in upgrade, (B) building
**runtime model switching** (no process restart), (C) integrating **DA-V3 small/base**
including local fp16 conversion and using DA-V3's extra outputs (confidence, intrinsics)
to improve the SBS warp. Expands roadmap item #2 in [sbs-3d-roadmap.md](sbs-3d-roadmap.md).

All model-contract facts below were **verified empirically on 2026-07-03** (graphs
downloaded and inspected with the `onnx` package; inference cross-checked with
onnxruntime against the production DA-V2 model on real `E:\ApolloDev\sbs_dump` frames).
Do not re-derive them; do trust them.

---

## Hard constraints (read before writing any code)

1. **TensorRT 11 is strongly-typed, always.** There is NO `kFP16` builder flag
   (deprecated/ignored, see NvInfer.h ~11937). Engine precision comes from the ONNX
   weight dtypes. fp32 ONNX → fp32 engine → ~2× slower. **fp16 speed requires fp16
   weights in the ONNX file.**
2. **MinGW/MSVC ABI rule:** never call TRT member functions that return `Dims` BY VALUE
   (`getDimensions`, `getTensorShape`, `getProfileShape`, …) — they scramble arguments
   and SIGSEGV. Only use functions returning `const char*`/enum/int/pointer. Passing
   `Dims` INTO TRT (`setDimensions`, `setInputShape`) is fine. (AGENTS.md rule #6.)
3. **Never delete `IExecutionContext`s** — they're pooled (`g_context_pool`) because
   deletion across the ABI boundary is unsafe and each holds ~1.3 GB scratch. Extend the
   pooling pattern; do not "fix" it.
4. **Warpsim-first mandate** (CLAUDE.md): any change to warp/shader behavior — including
   the Phase C confidence/intrinsics work — must be prototyped and A/B'd offline in
   `tools/warpsim/` on the dump library (both eyes, multiple scenes, numeric evidence)
   BEFORE headset time. When `sbs_reprojection_ps.hlsl` changes, update the simulator
   replicas to match.
5. Build: MSYS2 UCRT64, `ninja -C cmake-build-relwithdebinfo sunshine` (skips the web-ui
   step). Shaders are hardlinked into the build assets dir — shader-only change = restart
   Apollo, no rebuild. Exe locks while Apollo runs. `TENSORRT_DIR=E:/TensorRT-11.1.0.106`.
6. Push only to the `dcatcher9` forks, never upstream (`ClassicOldSong/*`).
7. Dev rig: config at `E:\ApolloDev\config\sunshine.conf` (sbs_3d_* keys are conf-only,
   not in the web UI); log at `E:\ApolloDev\config\sunshine.log`; client = Galaxy XR.

## Verified model contracts (2026-07-03)

| | DA-V2-small (current) | DA-V2-base | DA-V3-small | DA-V3-base |
|---|---|---|---|---|
| Source | onnx-community `.../depth-anything-v2-small/resolve/main/onnx/model_fp16.onnx` | same repo pattern, `-v2-base`, `model_fp16.onnx` (195 MB) | onnx-community `-v3-small` — **fp32 only**: `onnx/model.onnx` (640 KB graph) + `model.onnx_data` (105 MB) | `-v3-base`, same two-file fp32 (412 MB data) |
| fp16 single file | yes (50 MB) | **yes** | **NO** — PR #2's "model_fp16.onnx" is mislabeled (all-fp32 initializers). Convert locally (Phase C1) | NO — convert locally |
| Input | `pixel_values` FP32 rank-4 `[1,3,H,W]` | same | `pixel_values` FP32 **rank-5** `[batch, num_images, 3, H, W]` | same |
| Outputs | `predicted_depth` FP32 `[1,H,W]` | same | `predicted_depth` `[b,n,H,W]` + `confidence` `[b,n,H,W]` + `extrinsics` `[b,n,3,4]` + `intrinsics` `[b,n,3,3]`, all FP32 | same |
| Patch multiple | 14 | 14 | **14** (first Conv strides [14,14] — verified) | 14 |
| Output convention | relative **disparity** (larger = closer) | same | **scale-shift-invariant DEPTH (larger = farther)** — needs reciprocal. Verified: corr(DA2, 1/DA3) = 0.78–0.93 on 4 fullscreen dumps; raw corr negative. Output strictly positive (0.59–1.9 observed) | same |
| Normalization | ImageNet mean/std (baked in `rgb_to_nchw_cs.hlsl`) | same | same (verified working) | same |
| Params / cost class | ~25 M, game-mode | ~97 M, ~3–4×, movie budget | ~26 M, ≈ V2-small cost | ~103 M, ≈ V2-base cost |

Known DA-V3 behavior note: on a browser/desktop dump it returned near-flat depth
(range 0.81..1.02) — it *correctly* recognizes flat screen content that DA-V2
hallucinates parallax for. Good for future DESKTOP mode, but include a desktop scene
in every A/B so this is tracked deliberately.

---

## Phase A — DA-V2-base A/B (config only, no code)

1. In `E:\ApolloDev\config\sunshine.conf`:
   ```
   sbs_3d_depth_model = depth_anything_v2_base_fp16
   sbs_3d_depth_model_url = https://huggingface.co/onnx-community/depth-anything-v2-base/resolve/main/onnx/model_fp16.onnx
   ```
2. Restart Apollo. Background download + engine build runs (~6.5 min+ for small; base
   will be longer). Stream shows flat SBS until the engine exists — expected.
3. Verify in the log: `Depth engine tensor 'pixel_values' (input) dtype=FP32` and the
   matching `predicted_depth` line (I/O validation passes), plus the cadence lines.
4. A/B protocol (same scenes as the existing dump library — include the nose/sword
   small-feature-relief crops the roadmap calls out, plus one desktop/browser scene):
   - Dump 3D per scene with base, compare against the v2-small dumps in warpsim.
   - Record sustained inference cost: `Depth cadence` log lines + GPU % — what
     `sbs_3d_depth_fps` does base actually hold at depth_short_side 336?
5. Outcome feeds Phase B registry defaults (per-model `depth_fps` override if base
   can't hold 45) and the roadmap's V2-base go/no-go.

Restart-per-switch is acceptable for this phase; removing it is Phase B.

### Phase A results — offline quality half (2026-07-03)

Ran DA-V2-base vs the production DA-V2-small (both fp16 ONNX, onnxruntime CPU, ImageNet
norm, 798×336) on **all 55 `E:\ApolloDev\sbs_dump` source frames**. Metrics: global
base-vs-small correlation, and a foreground "relief" metric = mean |Laplacian| of the
min/max-normalized depth over the closer-than-median region (proxy for small-feature
depth structure, the roadmap's stated V2-base target).

- **No global regression:** correlation mean **0.989** (min 0.833, max 0.997). Base and
  small agree on overall scene layout everywhere.
- **More relief, modestly:** relief ratio base/small mean **1.11**, median 1.12; base
  has more foreground depth structure in **78%** of scenes. Biggest gains on
  portraits/faces (up to ~1.38×).
- **Visual (panels in scratchpad `ab_out/`):** on the fullscreen warrior and the washing
  portrait, base resolves robe/arm/torso relief that small flattens, and recovers real
  background architecture small crushes to black.
- **Two caveats to watch on the headset:**
  1. Base reads overlaid **text/logo watermarks as having depth** (the portrait's "豆云"
     watermark got its own parallax blob) — could make overlays float. Watch UI/subtitle
     overlays in movie content.
  2. The one low-correlation scene (0.833) is a backlit symmetric hall with sheer
     curtains — a genuinely ambiguous scene where the models diverge (both plausible),
     not a base defect. This is the exact low-confidence case DA-V3 confidence (Phase C4)
     would flag.

**Quality verdict: GO** — base is a real if modest small-feature-relief improvement with
no coherence regression. Remaining Phase A gate is the **live perf half** (below): does
base sustain a usable `sbs_3d_depth_fps` at depth_short_side 336 on the rig's GPU? That
number can only come from the running host and decides GAME-mode viability vs
movie-mode-only. Config + model are pre-staged for it.

### Phase A — live perf half (hand-off; needs the running host + headset)

Everything is staged: `E:\ApolloDev\config\sunshine.conf` now sets
`sbs_3d_depth_model = depth_anything_v2_base_fp16` (+ URL), and the fp16 ONNX is already
in `cmake-build-relwithdebinfo/assets/` so no download is needed — only the engine build.
To run it:
1. Quit the tray Apollo; launch `E:\ApolloDev\run-dev.cmd` as admin.
2. First start builds `depth_anything_v2_base_fp16.engine` (several min, longer than
   small). Watch `E:\ApolloDev\config\sunshine.log` for `Saved built engine`. Stream
   shows flat SBS until then (expected).
3. Connect the Galaxy XR, enable Host SBS Game, and read the log:
   - `Depth engine tensor 'pixel_values' (input) dtype=FP32` + matching `predicted_depth`
     → I/O validation passed (confirms the drop-in).
   - `Depth cadence` lines → the sustained effective depth fps and inference interval.
   - GPU % (Task Manager / nvidia-smi) during an active game scene.
4. Decision: if base holds ~45 fps depth like small, it's GAME-mode viable; if it drops
   to ~20–30, it's movie-mode-only → set a per-model `depth_fps_override` in the Phase B
   registry. Either way, record the number here.
5. Revert: comment out the two `sbs_3d_depth_model*` lines to fall back to small.

**Live perf result (2026-07-03): GO / GAME-viable.** Base runs well on the rig,
"marginally better than small" perceptually, holds usable depth cadence — **no
`depth_fps_override` needed**. Phase A complete; proceeding to Phase B.

## Phase B — dynamic switching foundation (host + client)

### B1. Model registry + per-model params

`config.h` / `config.cpp` / `video_depth_estimator.h`:

- Add a `depth_model_info` struct: `{ std::string name, url; int input_rank = 4;
  int patch = 14; int output_transform = 0 /*0=identity, 1=reciprocal*/;
  bool keep_confidence = false; std::string input_tensor = "pixel_values",
  output_tensor = "predicted_depth"; double depth_fps_override = 0; }`.
- Built-in preset table (index = wire id):

  | id | name | notes |
  |---|---|---|
  | 0 | `depth_anything_v2_fp16` | current default, rank-4, identity |
  | 1 | `depth_anything_v2_base_fp16` | rank-4, identity, fp16 URL above |
  | 2 | `depth_anything_v3_small_fp16` | rank-5, reciprocal, keep_confidence, **no URL** (local file, Phase C1) |
  | 3 | `depth_anything_v3_base_fp16` | ditto |

- Existing `sbs_3d_depth_model`/`_url` keys stay as the *startup default* and the custom
  escape hatch (a name not in the table = a registry entry synthesized from the two keys
  with default params).

### B2. Multi-engine TRT state

`video_depth_estimator.cpp` — replace the single `g_engine`/`g_context_pool`/
`g_engine_model`/`io_validated` statics with, guarded by the existing `g_trt_mutex`:

```cpp
struct engine_slot {
  nvinfer1::ICudaEngine* engine = nullptr;
  std::vector<nvinfer1::IExecutionContext*> context_pool;
  bool io_validated = false;
};
static std::map<std::string, engine_slot> g_engines;  // key: model name
```

- Estimator ctor looks up/creates its model's slot; contexts return to *their* slot's
  pool on teardown. Delete the "restart Apollo to apply" error path.
- Engines stay resident once loaded (no eviction — constraint #3). With sequential
  testing this is 2–3 engines resident; acceptable. If VRAM becomes a problem, the lever
  is a smaller per-model optimization profile MAX (currently 1008²) — if you change it,
  encode the profile in the engine filename (e.g. `<name>.p700.engine`) so cached engines
  never mismatch.
- I/O validation becomes per-slot and uses the registry's tensor names.

### B3. Runtime switch (host)

- **Active-model state:** `video::active_depth_model()` getter/setter (mutex'd string,
  initialized from config). `ensure_depth_estimator()`
  ([display_vram.cpp:535](../src/platform/windows/display_vram.cpp)) polls
  `<active>.engine` instead of the config value; the estimator ctor receives the active
  registry entry.
- **Switch path:** reuse the encoder-rebuild machinery that `mail::sbs_mode` already
  drives ([video.cpp:2403](../src/video.cpp) `capture_async`, drain-to-latest). On a
  model-switch request: set active model, kick `precompile_tensorrt_engine` on a detached
  thread (it's already mutex-serialized) if `<name>.engine` is absent, then raise the
  sbs_mode event with the *current* mode to force the rebuild. If the engine isn't ready
  the existing flat-stream-and-poll behavior covers the gap (log already warns every ~20 s).
- **Wire protocol:** new control message `0x3006` "Set Depth Model" (**0x3004 = debug
  dump, and 0x3005 may exist in other forks — check `packetTypes` in both repos and take
  the next free value; update this doc if not 0x3006**), payload `[model_id, 0, 0, 0]`,
  mirroring the `0x3003` handler in [stream.cpp:1060](../src/stream.cpp): validate
  `model_id < registry.size()`, log the model name, do the switch. Reject + warn on
  unknown ids.
- **Dump attribution:** include the active model name in the Dump-3D output directory
  name (or a `meta.txt` in it) so A/B crops are attributable.

### B4. Client trigger (Artemis — `E:\Git\Repo\moonlight-android`, branch `moonlight-noir`)

Mirror the shipped 0x3003 work (see memory / commit `dd1d52e7` for the exact pattern):

- `moonlight-common-c` (the **dcatcher9 fork** submodule): next free `IDX_*`, new packet
  type appended to ALL 5 `packetTypes` tables (Gen7Enc carries it, −1 elsewhere),
  `LiSendSetDepthModel(uint8_t id)` in ControlStream.c, decl + `DEPTH_MODEL_*` in
  Limelight.h.
- JNI: `simplejni.c` (name-based) + `MoonBridge.sendSetDepthModel(int)`.
- UI: a "Model" **cycle tile** on the XR control bar (`XrStreamPresenter.selectMode`
  area): v2-S → v2-B → v3-S → v3-B, hardcoded 4 ids, toast with the model name and a
  note that a first-time switch may build for minutes (host streams flat meanwhile).
  Only visible/enabled in Host SBS presentations.

### Phase B status — DONE + HEADSET-VERIFIED (2026-07-03)

Implemented, both build (host `ninja sunshine` links; client
`assembleNonRoot_gameDebug` BUILD SUCCESSFUL), and **verified on the Galaxy XR**: 21
mid-stream DA-V2 small↔base switches over ~5 min. Log evidence — **2 TensorRT execution
contexts created total, 20 reused** (one context per engine, then pure reuse → no per-switch
1.3 GB allocation, no VRAM growth: the core B2/B5 guarantee), switches instant (both engines
cached, no multi-minute stall), no errors/mismatches/device-removed (only unrelated benign
"Audio capture buffer overflow" warnings). All B5 acceptance criteria met. Not yet committed.

- **B1 registry** — `config::depth_model_info` + `config::depth_model_registry()` (4 entries;
  ids 0/1 = DA-V2 small/base with URLs, 2/3 = DA-V3 small/base, no URL, Phase C params).
- **B2 multi-engine** — `g_engine`/`g_context_pool`/`g_engine_model`/`io_validated` statics
  replaced by `std::map<std::string, engine_slot> g_engines` (per-model engine + context pool
  + io_validated). Dtor returns the context to its model's pool. "Restart Apollo" path deleted.
- **B3 host switch** — `video::active_depth_model()`/`set_active_depth_model()` (mutex'd,
  lazily resolved from config); `mail::depth_model_reload` peeked in `encode_run` (both break
  sites) and drained in `capture_async` (rebuild reuses the proven sbs_mode teardown path);
  `ensure_depth_estimator` loads the active model; `0x3005 IDX_SET_DEPTH_MODEL` handler in
  stream.cpp (rejects unknown ids and — Phase B — rank-5/DA-V3 with a clear log; kicks a
  background `precompile_tensorrt_engine` under `#ifdef _WIN32`; raises the reload event).
  Dump attribution: `sbs_debug_dump` writes `meta.txt` with the active model name.
- **B4 client** — moonlight-common-c (dcatcher9 fork submodule): `IDX_SET_DEPTH_MODEL=18`,
  `0x3005` appended to all 5 packetTypes tables (Gen7Enc only), `LiSendSetDepthModel(uint8_t)`,
  `DEPTH_MODEL_*` + decl in Limelight.h. JNI `sendSetDepthModel` in simplejni.c +
  `MoonBridge.sendSetDepthModel`/`DEPTH_MODEL_*`. XR bar "Model" cycle tile
  (`ic_xr_model.xml`, `xr_bar_model`/`xr_toast_depth_model` strings) cycling **DA-V2
  small<->base only** (v3 joins in Phase C), toast shows the selected name.

### B5. Acceptance for Phase B

- Switch v2-small ↔ v2-base mid-stream from the headset: brief IDR/rebuild blip, depth
  visibly changes, no VRAM growth over repeated toggles (contexts pooled per slot —
  watch GPU memory across ≥10 toggles), no 6.5-min stall when both engines are cached.
- Unknown model id from client → warn + ignore, stream unaffected.
- `test_sunshine` still builds/passes; add a unit test for registry lookup/id mapping.

## Phase C — DA-V3 integration

> **Status (2026-07-04): C1–C3 DONE for DA-V3 small (fp32 + fp16), host builds, offline-validated;
> awaiting headset test. C4/C5 not started.** Both ONNX variants staged in assets (registry ids
> 2=v3_small_fp16, 4=v3_small_fp32; rank-5, reciprocal, keep_confidence=false until C4). Conversion
> recipe lives in `tools/depth_models/convert_da3.py` — note the fp16 path had to use ORT's
> transformers converter (+ duplicate-node cleanup), NOT onnxconverter-common, which produces an
> invalid graph on the DINOv3/RoPE backbone (the flagged risk). Reciprocal orientation verified
> offline (corr +0.78 vs v2 with it, −0.71 without). To test: uncomment the v3 line in the dev
> conf, restart, Host SBS Game on the headset. Watch for reciprocal outliers (near-zero depth ->
> huge disparity; only a 1e-6 floor today).

### C1. Local fp16 conversion (one-time offline tool)

`tools/depth_models/convert_da3_fp16.py` (new; Python, flake8-clean):

1. Download `onnx/model.onnx` + `onnx/model.onnx_data` from the HF repo (main branch)
   into a temp dir — the graph references the literal filename `model.onnx_data`, so the
   pair must sit together.
2. `onnx.load(...)` (resolves external data) →
   `onnxconverter_common.float16.convert_float_to_float16(model, keep_io_types=True)`
   → save **single-file** (no external data) as `depth_anything_v3_small_fp16.onnx` /
   `..._v3_base_fp16.onnx`.
3. Validate in-script with onnxruntime: run fp32 vs fp16 on a real dump frame
   (798×336, ImageNet norm; input shaped `[1,1,3,H,W]`), assert
   `corr(fp32_depth, fp16_depth) > 0.99` and output finite/positive. If the converter
   trips on LayerNorm/GELU precision, use its `op_block_list` to keep those fp32 —
   TRT forces LayerNorm-adjacent ops to fp32 anyway (known benign warning).
4. Drop the result into the assets dir (`cmake-build-relwithdebinfo/assets/` on the dev
   rig). `ensure_model_available` picks up a local `<name>.onnx` before trying URLs —
   **no downloader changes needed**. Registry entries 2/3 have no URL; if the file is
   missing, log exactly where to put it and which script produces it.

### C2. Engine build for rank-5 / multi-output models

`precompile_tensorrt_engine` ([video_depth_estimator.cpp:68](../src/video_depth_estimator.cpp)):

- Optimization profile from the registry's `input_rank`: rank-5 uses `nvinfer1::Dims`
  with `nbDims=5`, `{1,1,3,H,W}` for MIN/OPT/MAX (keep the current 14/518/1008 spatial
  points). Rank-4 path unchanged.
- After parsing, **prune outputs**: iterate `network->getNbOutputs()` /
  `network->getOutput(i)` (pointer — ABI-safe) and `unmarkOutput()` every tensor whose
  name is neither `output_tensor` nor (if `keep_confidence`) `confidence`. This drops
  the extrinsics/intrinsics heads from the engine entirely.

### C3. Runtime rank-5 + reciprocal transform

`video_depth_estimator.cpp` impl:

- `setInputShape` with rank-5 dims when `input_rank == 5` (same H×W; the extra leading
  `1,1` doesn't change buffer layout — the existing D3D NCHW buffer binds as-is).
- Bind a `confidence` output buffer when `keep_confidence` (H×W floats, created next to
  `tensor_out_buf`). All engine I/O tensors must have addresses before `enqueueV3`.
- **Transform:** add `uint output_transform` to the constant buffers (they're immutable,
  built once — fine, it's per-session). Apply `t(x) = transform==1 ? 1.0/max(x,1e-6) : x`
  in **both** `depth_minmax_cs` (reduce over transformed values) and `buffer_to_tex_cs`
  (transform before normalize). Transformed values stay positive → the asuint min/max
  ordering trick still holds.
- **Numeric gate before any quality work:** first V3 run, dump depth and compare against
  a DA-V2 dump of the same scene — normalized depth must positively correlate (this
  catches a missed/incorrectly-placed reciprocal immediately).
- A/B V3-small vs V2-small and V3-base vs V2-base per the Phase A protocol. V3-small at
  V2-small cost with richer background gradation is the hoped-for result.

### C4. Confidence-guided SBS (quality work — warpsim FIRST)

Target: the roadmap's residual #1 (stretch-band fringe on silhouettes) and hair
fold-over. Confidence is low exactly where those depths are wrong.

1. **Offline prototype (before any HLSL):** extend the warpsim Python side to run the
   converted V3 ONNX via onnxruntime and emit depth + confidence for the dump library.
   Implement candidate uses in the simulator and A/B numerically on the known
   stretch-band/hair scenes, both eyes:
   - (a) confidence-weighted guided upsample — down-weight low-confidence depth samples
     in the joint-bilateral so the color-edge snap stops inheriting contact-shadow
     columns (the diagnosed fringe cause);
   - (b) confidence-modulated parallax — in the reprojection, blend disparity toward the
     focal plane as confidence drops (wrong pixels get *less* warp);
   - (c) confidence-weighted temporal EMA — low-confidence pixels lean on depth history
     (hair flicker).
2. Ship only what wins numerically. Likely plumbing: copy confidence into an `R32F`
   texture in the same pass as `buffer_to_tex_cs` (second UAV), new SRV into
   `depth_guided_upsample_cs` and/or `sbs_reprojection_ps`; all gated on
   `keep_confidence` so V2 models run the unchanged path (byte-identical output for V2 —
   verify by byte-compare in warpsim).
3. Update the simulator replica functions for any shader change (constraint #4).

### C5. Focal-anchored disparity (optional, after C4)

DA-V3's intrinsics output estimates focal length (sanity-checked: ~640 px on a 798-wide
frame ≈ 64° HFOV, plausible). Replaces min/max range-stretching with physical mapping
`disparity_px = f_px × baseline / Z` (per DA3's metric recipe `depth = focal × out / 300`):
scene- and time-stable parallax, no range pumping, and flat content correctly gets ~zero
parallax (future DESKTOP mode win). Requires keeping the `intrinsics` output (12 floats —
cheap) and a small CPU readback or GPU-side constant; EMA the scale. Prototype in warpsim
against the minmax pipeline on close-up + landscape + desktop scenes before deciding.
This is exploratory — do not start it until C4 has shipped or been rejected on numbers.

---

## Execution notes for the implementing agent

- Read first: `CLAUDE.md`, `docs/sbs-3d-roadmap.md`, `tools/warpsim/README.md`, and the
  auto-memory (`sbs-3d-depth-feature` has the pipeline mental model, tuned defaults, and
  the 0x3003 protocol history you'll mirror in B3/B4).
- Order: A (no code, do it immediately) → B1→B2→B3 (one host PR-sized change, keep
  commits per sub-phase) → B4 (client commit on `moonlight-noir`) → C1→C2→C3 (V3 runs
  correctly) → C4 (quality, sim-gated) → C5 (optional).
- Each phase compiles and streams on its own; never leave the tree in a state where
  V2-small (id 0) doesn't work exactly as today.
- Out of scope: engine eviction/LRU, multi-file model downloads, host→client model-list
  negotiation, INT8 models, MOVIE-mode sync depth (separate roadmap item), the `mlbw`
  learned warp (roadmap item #1 — but keep C4's confidence texture in mind as its future
  input channel).
