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
  -> one primary ordered decoder
  -> production causal estimator + V2 renderer
  -> one atomic SBS raster
  -> one persistent NVENC encoder
  -> stream-preserving mux
  -> timeline/color/stream verification
  -> atomic output publication
```

The native harness is launched once with follow protocol schema 2. For conversion it uses
`--artifacts conversion`; Analyze Only uses `--artifacts adaptive`. Both modes consume the same
decoded full frames and the same source-derived observation timestamps. Every admitted source frame
runs one authenticated TensorRT update. Conversion additionally publishes the exact SBS raster drawn
from that estimator state.

The worker hands each completed SBS raster directly to a persistent FFmpeg/NVENC process, waits
until the local authenticated frame bridge has served it, and removes both transient rasters. The
worker never launches a render decoder, scene replay, or second TensorRT pass.

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

Offline processing is not paced to source playback time. Decoder, GPU, and encoder run as quickly as
their completion and backpressure allow. Windows directory-change notifications wake both sides of
the frame protocol; there is no fixed per-frame polling delay. Frames remain ordered because the
online temporal state is causal, so throughput does not come from reordering dependent frames.

At most the current source raster and current SBS raster are live between the harness and worker.
There is no whole-clip image sequence, unresolved-scene depth cache, or per-scene compressed file.
The persistent encoder retains only its normal compressed-video state. Legacy cache-limit request
fields remain accepted for persisted/stale clients and act only as a transient-raster safety bound;
they cannot split scenes or change geometry.

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
- persistent encoder frame ordering and exact timeline checks;
- SDR, PQ, and HLG color/metadata preservation;
- HEVC and AV1 admission/output validation; and
- UI assertions that no lookahead/cache/split controls are exposed.

The production integration smoke must run only when the installed host has released its ports/GPU
ownership. It creates disposable media and job directories and must never publish into an existing
user destination.
