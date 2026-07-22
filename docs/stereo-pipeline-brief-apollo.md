# Adapting the "CUDA-native 2D→SBS" brief to Apollo-3D

> **Archived design snapshot (2026-07-05).** Claims below about MLBW, guided upsample,
> legacy disparity, and warpsim describe removed experiments. Use `sbs-3d-roadmap.md` and
> `tools/sbsbench/README.md` for the current implementation and workflow.

Source: `stereo-pipeline-implementation-brief.md` (external, 2026-07). That brief assumes a
**greenfield** project (standalone `libsbs` + CLI apps, Apollo integration as a distant
Milestone 3). Apollo-3D is already at and past its target state: the pipeline lives inside
the host between capture and NVENC, with a learned warp and an implemented (stashed) inpaint
stage. This document maps the brief onto our reality: what already exists, what is worth
adopting, and what we explicitly reject.

Read first: [sbs-3d-roadmap.md](sbs-3d-roadmap.md) (status + constraints),
[sbs-inpaint-plan.md](sbs-inpaint-plan.md) (the inpaint stage, currently in
`git stash` "SBS inpaint stage WIP").

## Status map: brief section → Apollo-3D

| Brief | Proposal | Apollo-3D status |
|---|---|---|
| §3.1 capture | DXGI duplication, HDR tonemap for inference copy only, cursor, register-once interop | **SHIPPED** (ddup/wgc; `rgb_to_nchw_cs.hlsl` Reinhard+sRGB on the inference copy only; interop registered once, lazily on the encode thread — hard-won, see estimator notes) |
| §3.2 depth | DA-V2-small TRT fp16, 518² static, model-agnostic wrapper | **SHIPPED, BEYOND**: startup-profile-owned DA-V2 registry, startup engine/context warmup, aspect-aware 770×434 at 16:9 (no square-letterbox waste) |
| §3.3 temporal | P2/P98 percentile range + EMA + scene-cut reset | **PARTIAL**: min/max + EMA exists (`depth_minmax_cs` → `depth_minmax_ema_cs`, α=0.1). Percentiles + scene-cut reset are **new — adopt/experiment below** |
| §3.4 JBU | joint bilateral upsample | **SHIPPED** (`depth_guided_upsample_cs`, foreground bias + bimodal edge snap) — and **deliberately bypassed under MLBW** (texel-sharp depth is out of the warp's training distribution). Do not re-add under MLBW. |
| §3.5 disparity | divergence/convergence, ±40 px clamp | **SHIPPED** (`divergence`, `focal_plane`, `depth_floor`, border fade) |
| §3.6 warp + Jacobian mask | backward gather `d(x')≈d(x)`, ∂d/∂x threshold mask | **SHIPPED, TWO GENERATIONS BEYOND**: the brief's gather is what our probe-search replaced, and MLBW replaced the probe. The mask exists as a **learned head** (`mask_mlbw_l2`) matched to the trained inpaint — strictly better than a Jacobian threshold. |
| §3.7 composite | full/half SBS, anaglyph, debug views | full SBS **SHIPPED**; half-SBS + debug views = ideas below |
| §4 perf | CUDA graphs, per-stage `cudaEvent` HUD, no per-frame allocs | timing = **adopt below**; graphs = reject (hybrid D3D/TRT pipeline); zero per-frame allocation already holds (immutable cbuffers, preallocated buffers) |
| §6 inpaint seam | interface stub, no ML | **SURPASSED**: real `light_inpaint_v1` stage implemented, offline-validated in-hole exact vs iw3, pipelined, downscale-grid perf knob — in `stash@{0}` pending headset acceptance |
| §7 testing | PyTorch golden reference, synthetic cards, perf gates | **SHIPPED as workflow**: `tools/warpsim` replica (byte-compare vs iw3 mandated by CLAUDE.md), Dump-3D captures, offline eval harnesses in `E:/ApolloDev/mlbw_eval` |
| M0 file-mode CLI | `sbs_convert` offline converter | reject: warpsim + dumps cover validation; iw3 itself is the offline converter for headset A/B files |
| M1 standalone app + virtual display | `sbs_live` presenting to a display Apollo captures | reject: adds a capture→present→capture round trip + latency; we ARE the M3 integration already |

## Adopt (prioritized)

### A1 — Scene-cut range snap (small, do first)
The brief is right that EMA'd range normalization "swims" after a hard cut: with
`minmax_ema = 0.1` per depth update, a cut needs ~10 updates to re-converge (0.2–0.7 s of
visibly drifting depth scale at movie rates). Cheapest correct fix, fully GPU-resident, no
new reduction pass: in `depth_minmax_ema_cs.hlsl`, when the fresh raw range disagrees with
the EMA'd range by a large factor (e.g. raw min/max outside the EMA bounds by
> 25 % of the EMA range, both ends, or range ratio > ~1.6× either way), **snap** the EMA to
the raw values instead of blending. This detects exactly the failure (range jumps at cuts)
without needing a frame-difference pass. This experiment was neutral and the processor/configuration
were later removed; it is retained here only as historical design context.
Validate offline on a dump pair straddling a cut before headset (workflow mandate).

### A2 — Per-stage GPU timing (the brief's best process idea)
"A stage that can't report its cost doesn't exist" — the 2026-07-06 inpaint perf surprise
(63 ms/frame discovered by reading frame-latency logs) proves the point. Add:
- D3D11 timestamp-query pairs (`ID3D11Query` DISJOINT + TIMESTAMP) around the stages of
  `convert()`: depth-input CS / composite PS / (inpaint stages when restashed), and CUDA
  events around `enqueueV3` on the depth/warp streams inside the estimator.
- Rolling 300-frame p50/p95/max, logged at info level in one concise line. Live summaries are
  process-wide windows for the sole active SBS pipeline; harness runs are isolated.
- Gate with the global `diagnostics = enabled` switch. The shipped default is disabled, while the
  ApolloDev config enables it; timestamp queries are not allocated while diagnostics are off.
This subsumes the CPU-side telemetry added in the inpaint stash and should land with it.

### A3 — Percentile normalization + range floor (experiment, offline-gated)
The brief's P2/P98 histogram normalization is a robustness idea worth ONE bounded
experiment, with a twist that targets a **measured** Apollo problem: on windowed desktop
content in GAME mode, min/max normalization **amplifies DA-V2's hallucinated page depth to
full parallax** (diagnosed 2026-07-06: game windowed < fullscreen). Two candidate knobs,
tested together in the offline harness (`E:/ApolloDev/mlbw_eval` + warpsim, multi-scene,
both eyes):
- `P2/P98` instead of min/max (histogram CS at depth res; FP32 accumulation) — kills hot
  pixels stretching the range (V2 raw disparity can spike).
- **Range floor**: if `(hi − lo)` is below a threshold (near-flat scene, e.g. a desktop
  page), don't stretch to full 0–1 — scale parallax down proportionally. This, not
  percentiles, is what actually fixes the windowed-game amplification; percentiles alone
  still stretch a flat page.
CAUTION from history: the sigma-clip/`norm_sigma` saga (removed 2026-07-05) was a
normalization band-aid; anything here must prove itself against the DA-V2 baseline offline
before shipping. If results are marginal, drop it.

### A4 — Debug composite views (small, quality-of-life)
The brief's debug views map cleanly onto the existing dump path rather than live modes:
extend Dump-3D (`0x3004`) with optional depth-heatmap and mask-overlay PNGs (host-side
colormap in the dumper). Live anaglyph/debug rendering: skip — the dump + offline viewers
already fill that role in our workflow. The Jacobian FOLD mask (`J < 1−τ`, A5) is a good
extra channel here.

### A5 — Jacobian hole mask over the MLBW effective delta (offline-gated experiment)
(a.k.a. the GAPW "Gradient-Aware Parallax Warping" mask idea — its Jacobian-threshold
occlusion mask is the reusable part; its single-field backward WARP is a step back from
MLBW and rejected, see §Rejected.) The brief's §3.6 Jacobian mask, re-aimed: we can't use
it on a raw disparity field (we don't warp that way), but the MLBW fields give a per-texel
**effective displacement**
`Σ wᵢ·dᵢ`, and a tiny field-res CS (same shape as `mlbw_mask_cs`) can compute
`J = 1 + ∂(Σw·d)/∂x` by central differences and threshold `J > 1+τ` into a hole mask.
Why it matters: the learned mask head exists **only as an l2 checkpoint** — it's the sole
reason movie mode dropped from the preferred l4 warp to `mask_mlbw_l2` when the inpaint
landed. A synthesized Jacobian mask decouples inpaint from warp choice (l4 + inpaint
becomes possible; any future warp too). Risks: τ is hand-tuned vs trained, and
`light_inpaint_v1` was trained on iw3's mask distribution — the Jacobian mask must prove
close enough after closing/threshold/dilation. Gate: offline harness, arm + head dumps —
Jacobian-mask-from-l4-fields → inpaint, compared against the l2 learned-mask result
(mask IoU + fill quality), before any host wiring. Only relevant once the stashed inpaint
is restored.

## Roadmap-level ideas (note, don't build now)

- **Half-SBS anamorphic output** (§3.7): halves encode pixels (8192×1728 → 4096×1728 for
  the same stereo content) → real NVENC/bitrate headroom. Requires Artemis-side unpacking
  (client change + a capability flag on `0x3003`). Worth a line on the roadmap.
- **Single-synthetic-eye mode** (`right_only`, §3.6): iw3's `synthetic_view=right` with
  divergence×2 — one eye stays bit-exact source, host does HALF the warp+inpaint work.
  Tradeoffs: doubled band width in the synthetic eye, possible binocular rivalry
  (sharp-vs-warped eye). Cheap config experiment once inpaint has landed; offline sheets
  first.
- **Streaming video-depth hook** (§3.3 note): nothing to do until a suitable model has a
  usable low-latency TensorRT export.

## Explicitly rejected (with reasons — do not relitigate)

1. **Pure-CUDA rewrite / "no graphics-API compute shaders" / `libsbs` / standalone apps
   (M0–M2)**: our hybrid D3D11-CS + TRT pipeline is shipped, validated, and its interop
   hazards are already paid for (lazy encode-thread registration, etc.). A rewrite buys
   fewer interop syncs and CUDA-graph eligibility at the cost of re-fighting every battle
   in the roadmap's "constraints learned the hard way". The standalone-app-plus-virtual-
   display route *adds* a capture hop and latency vs. our in-process integration.
2. **CUDA graphs**: inapplicable across the D3D/TRT boundary as built; TRT's own enqueue
   path is already optimized. Launch count is not our bottleneck (TRT inference and NVENC
   are, measured).
3. **Jacobian disocclusion mask as a REPLACEMENT for the learned mask on l2**: the
   `mask_mlbw` head is trained jointly with the warp and matches the inpaint's training
   distribution — keep it where it exists. But the idea is NOT rejected wholesale: see A5
   for its real use (synthesizing a mask for warp variants with NO mask checkpoint, e.g.
   l4).
4. **JBU under the learned warp**: guided upsample exists and is intentionally bypassed —
   texel-sharp depth is out of MLBW's training distribution (renders as staircase fringe).
5. **518² static depth input**: aspect-aware DA-V2 short-side sizing is equal-cost and strictly
   better. No going back to square letterboxing.
6. **NullInpainter seam**: superseded by the real, implemented inpaint stage.
7. **The brief's warp itself** (`d(x')≈d(x)` gather): two generations behind master.

## Design rules from the brief worth keeping as norms

Already true here: fail-soft everywhere (flat-SBS/probe/no-inpaint fallbacks), zero
per-frame allocations, golden-reference testing (warpsim byte-compare), fp16 storage with
fp32 accumulation in reductions. Newly adopted: **every stage reports its cost** (A2), and
its corollary — perf claims come from the counters, not from feel.
