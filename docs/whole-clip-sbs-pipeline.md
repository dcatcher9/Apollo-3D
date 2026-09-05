# Offline Host 3D conversion

Sunshine 3D's offline path runs the production Host SBS V2 pipeline causally in source order. It
does not have a second geometry policy, scene lookahead, a depth/state scene cache, or a render
replay. The authoritative geometry, camera, scene-cut, OCR, conditioning, and warp contracts remain
[Host SBS](host-sbs.md) and [Host SBS scene cuts](host-sbs-scene-cuts.md).

`sbs_3d_pop_strength` has the same literal meaning online and offline. Offline jobs do not search
for stronger scene-wide parameters or revise a camera reset after later frames arrive.

## Pipeline

```text
selected video
  -> one primary ordered decoder (two-source bounded window)
  -> production causal estimator + V2 renderer (strictly sequential)
  -> one-slot ordered CPU serializer
  -> one-slot ordered handoff to one persistent NVENC encoder
  -> stream-preserving mux
  -> timeline/color/stream verification
  -> atomic output publication
```

The native harness is launched once with follow protocol schema 2. For conversion it uses
`--artifacts conversion`; Analyze Only uses `--artifacts adaptive`. Both modes consume the same
decoded full frames and the same source-derived observation timestamps. Every admitted source frame
runs one authenticated TensorRT update. Conversion additionally publishes the exact SBS raster drawn
from that estimator state.

The harness copies each completed GPU readback into one CPU-owned slot. PNG/PFM serialization for
frame `N` can therefore overlap the production inference and render of `N+1`; no D3D device or
context crosses threads. The worker then transfers the atomic SBS artifact to a one-slot ordered
handoff whose consumer serves it to the persistent FFmpeg/NVENC process and removes it only after a
complete authenticated GET. NVENC consumption of `N` can likewise overlap production of `N+1`.
The worker never launches a render decoder, scene replay, or second TensorRT pass.

## Causal scene evidence

Scene records are diagnostic epochs of the online state machine:

- the first epoch begins at source sequence 1;
- an authenticated `hard_cut_pulse` and matching `hard_cut_count` transition at sequence `S`
  closes `[start, S)` and starts the next epoch at `S`;
- EOF closes the final epoch; and
- no future frame may move, reject, merge, or otherwise revise a boundary.

The worker rejects a pulse/count disagreement. Reports identify this contract as
`causal-production-exact`, with `lookahead: false`. Scene records do not control rendering: each SBS
frame has already been committed from the production causal state before the diagnostic epoch is
reported.

## Throughput and storage

Offline processing is not paced to source playback time. Decoder, GPU, CPU serialization, and
encoder advance through three bounded ordered stages. At most two decoded source artifacts are
published, one completed SBS snapshot is being serialized, and one SBS artifact is owned by the
encoder handoff. The next follow acknowledgement is not published until the worker has removed the
previous source artifact, proving the previous latest-state snapshot and ACK were consumed before
they can be replaced. Windows directory-change notifications provide this backpressure without a
fixed per-frame polling delay.

The production estimator and renderer remain strictly causal and execute one frame at a time;
throughput comes only from overlapping independent decoder, CPU serialization, and NVENC work, not
from reordering inference or looking ahead. After the first source reveals its exact transport size
and the shared production geometry resolver establishes the packed output, the harness checks one
source reservation plus one conservative SBS reservation before inference, rendering, or the first
SBS output snapshot. A conversion whose minimum single-slot reservation exceeds the configured
transient-raster hard cap fails closed. Before overlap is enabled, the worker separately checks two
source reservations plus two conservative SBS reservations. Each SBS reservation includes a tightly
packed CPU snapshot (BGRA8 for SDR or
R16G16B16A16_FLOAT for HDR), conservative serializer scratch (a BGRA surface plus codec allowance
for WIC, or the explicit RGB32F row used by PFM), and its worst-case serialized PNG/PFM artifact.
This covers the active serializer retaining one snapshot while the main thread maps the next, and
the encoder retaining one serialized artifact while the serializer publishes the next. The bound
is for the bounded pipeline handoff residency enumerated above; it is not a claim about total
process RSS, source-decoder/WIC internals, D3D driver allocations, or NVENC-internal surfaces. If
the two-slot bound does not fit but the mandatory single-slot bound does, conversion retains one
source at a time and a synchronous encoder handoff; CPU serialization still runs on its bounded
worker thread. Peak-raster attestation reports the applicable conservative pipeline bound. There is
no whole-clip image sequence, unresolved-scene depth cache,
or per-scene compressed file. Legacy cache-limit request fields remain accepted for persisted/stale
clients and remain only a transient-raster safety bound; they cannot split scenes or change
geometry.

Per-frame rasters, adaptive-state snapshots, and follow acknowledgements use same-volume atomic
publication but do not force stable-storage flushes. They are runtime handoffs, never restart
inputs; interrupted jobs are not resumed. A consumer that is woken while the producer's rename or
a filesystem filter still holds a transient name retries that same snapshot for a short, bounded
interval, with no delay on the normal first-read path and no advance to a future frame. Job state,
retained evidence, result contracts, and final output publication keep their existing durability
guarantees.

SDR encoding does not reopen the source video. Static PQ/HLG encoding keeps a second, aligned source
decode only as an HDR side-data donor because PFM carries pixels but not per-frame mastering-display
and content-light side data. Every encoded pixel still comes from the direct SBS PFM stream. A
single-frame donor must not be looped: FFmpeg 8.1.2 does not retain those side-data records on looped
clones. This correctness-only HDR exception can be optimized only after HEVC and AV1 random-access
tests prove the metadata survives at every keyframe.

## Job types

### Analyze Only

Analyze Only runs the causal estimator and state machine without publishing SBS video. It emits the
source contract, causal scene audit, progress, and terminal whole-clip attestation. This is the dry
run for conversion state, not a future-aware optimizer.

### Convert Video

Convert Video runs the same causal pass and publishes every atomic SBS frame to one persistent
H.265/HEVC or AV1 NVENC encoder. It then preserves supported auxiliary streams, verifies the encoded
timeline and color contract, muxes the final container, and publishes the output only after all
checks pass.

## Media contract

- Input is one non-empty regular video file selected by absolute path.
- The first video stream is decoded in presentation order with square pixels, progressive scan, a
  fixed raster, and a stable color description.
- Source presentation timestamps and durations are represented exactly. MP4 output must match the
  rational source timeline exactly; Matroska may differ by at most one output tick per frame and may
  not accumulate drift.
- SDR remains SDR. Static BT.2020 PQ and HLG use linear-scRGB PFM interchange and 10-bit output.
- Dolby Vision, dynamic HDR10+, changing HDR metadata, rotation, interlace, alpha, changing raster,
  and unsupported semantic side data fail closed.
- Supported audio, subtitle, data, attachment, chapter, language, and title metadata are preserved
  under the stream-inventory contract. Subtitle dispositions are cleared where container rules
  require it.
- Output is `.mkv` or `.mp4`, beside the input, and an existing destination is never overwritten.

## Native ownership and isolation

The Web UI uses the in-process job manager and launches an authenticated isolated `sunshine.exe`
child. Python and `tools/sbsbench` do not own production conversion. `ffmpeg.exe` and `ffprobe.exe`
must be the approved installation-local pair.

Only one offline job may own the GPU. Admission fails while a live Moonlight stream or local AR
presentation owns it. Codec preflight runs only after the offline job receives the exclusive GPU
lease. Ordinary streaming remains available when the offline prerequisites are absent.

The child receives a hashed worker specification, writes only inside its identity-pinned job root
and claimed staging output, and publishes typed progress/result contracts. Cancellation terminates
the worker process tree. Startup recovery marks interrupted jobs; version 1 does not resume them.

## Terminal attestation

A successful causal pass must attest:

- `artifact_mode: adaptive` for Analyze Only or `conversion` for Convert Video;
- `inference_mode: single-pass-tensorrt`;
- inference enabled with scheduled/enqueued counts equal to source frame count;
- the selected-input full-frame source scope;
- the production live V2 signed-parallax renderer;
- no scene-cache write or replay configuration;
- atomic SBS publication, complete frame count, and resolved output geometry for conversion; and
- an empty replay-contract list and zero retained cache bytes.

Conversion success additionally requires encoded-video, final-container, timeline, color/HDR,
auxiliary-stream, staging-identity, and output-file verification. Any mismatch fails the job before
publication.

## Web UI workflow

1. Open **Offline 3D Conversion**.
2. Choose **Convert video** or **Analyze only**.
3. Select the input file.
4. For conversion, choose the output name and H.265 or AV1.
5. Stop any live stream/local presenter and start the job.
6. Monitor source-frame progress for the current job.
7. Use the atomically published video after completion.
8. Clear the finished job to remove its retained temporary files and history. Published video is
   preserved. If staging cleanup was interrupted, Clear retires only the identity-attested staging
   link after confirming the published video still matches; an absent staging link is already clean.
   A replaced staging path or a staging file that may be the last intact copy is preserved.

There are no lookahead, cache-size, or administrative-split controls.

## Verification

Native regression coverage must include:

- causal pulse/count epoch tracking and fail-closed divergence;
- direct conversion command wiring with no cache, replay, or render decoder;
- one-pass whole-clip/result contract validation;
- follow protocol schema 2 geometry and event-driven wakeup;
- two-source/one-serializer/one-encoder-slot bounds and conservative byte-cap fallback;
- serializer and encoder overlap without ACK overwrite or D3D cross-thread access;
- persistent encoder frame ordering and exact timeline checks;
- SDR, PQ, and HLG color/metadata preservation;
- HEVC and AV1 admission/output validation; and
- UI assertions that no lookahead/cache/split controls are exposed.

The production integration smoke must run only when the installed host has released its ports/GPU
ownership. It creates disposable media and job directories and must never publish into an existing
user destination.
