# Offline Host 3D conversion

This document owns production offline frame transport, job lifecycle, resource bounds, diagnostics,
encoding, and publication. [Host SBS](host-sbs.md) owns model, geometry, and color policy;
[Host SBS scene cuts](host-sbs-scene-cuts.md) owns causal cut decisions. The
[evaluator guide](../tools/sbsbench/README.md) owns experiments and their evidence requirements.
[AGENTS.md](../AGENTS.md) owns machine-specific execution instructions.

Offline conversion runs the production Host SBS V2 pipeline in source order. Pop strength,
camera, OCR, conditioning, and warp rules have the same meaning as live conversion. Diagnostic
scenes cannot revise a rendered frame or choose new geometry parameters.

## Pipeline and ownership

```text
selected video
  -> one primary FFmpeg decoder, raw stdout pipe
  -> bounded source-memory channel
  -> production estimator + V2 renderer on one GPU owner thread
  -> immutable packed CPU readback, bounded memory handoff
  -> raw stdin pipe to one persistent FFmpeg/NVENC encoder
  -> stream-preserving mux and verification
  -> atomic output publication
```

The manager launches an authenticated isolated worker process. That worker calls the real native
harness once on its GPU thread. Conversion uses `--artifacts conversion`; Analyze Only uses
`--artifacts adaptive`. Every source frame runs one authenticated TensorRT update and consumes its
source-derived observation timestamp. All D3D/TensorRT work stays on this single owner.

Production frame transport is `bounded-raw-memory-v1`, defined in `src/offline_frame_transport.h`.
Each immutable frame carries schema, sequence, exact PTS and rational time base, width, height,
and pixel format. Payload size, ordering, and geometry are checked before use. Moving a frame
transfers ownership of its allocation.

| Boundary | SDR | Static PQ/HLG |
|---|---|---|
| FFmpeg to worker/harness | Top-down BGRA8, sRGB | Top-down planar GBR float32, linear scRGB |
| Harness upload | BGRA8 | RGBA16F, using the existing finite FP16 conversion |
| Packed CPU handoff | Top-down BGRA8 | Top-down RGBA16F |
| Worker to FFmpeg | Raw BGRA8 | Planar GBR float32, expanded in one 64 KiB buffer |

HDR reference white and primaries remain owned by Host SBS. Production conversion does not write
source BMP/PFM or rendered PNG/PFM files, run WIC codecs, or serve pixels over HTTP. The standalone
evaluator keeps its established file-based harness inputs and outputs.

The source channel has capacity two and the packed-output channel has capacity one. A separate
source acknowledgement proves the worker consumed that frame's exact adaptive-state snapshot.
Receiving pixels alone cannot release that acknowledgement. The next snapshot/progress record
waits for the previous acknowledgement. Analyze Only uses the same evidence barrier. Small state
and progress files retain atomic snapshots and directory-change notifications; raw pixel transfer
uses condition-variable backpressure.

The decoder pipe uses bounded overlapped reads, so each available chunk wakes its owner without
a polling delay. The persistent encoder has one outstanding handoff. Its private inherited pipe
uses bounded overlapped writes, a reusable event, and explicit cancellation. EOF follows the final admitted
frame. Premature exit, truncated data, unexpected frame count, header mismatch, non-finite HDR
values, or a timed-out handoff fails the job before output publication.

## Throughput and raster bounds

Offline execution is unpaced. Independent decode, rendering, and encoding may overlap while GPU
work remains causal. Before reading pixels, the worker reserves the source, decoder pipe, and
HDR upload conversion for both Analyze Only and Convert Video. The harness
authenticates output geometry and the minimum raster reservation before the first SBS readback.
`raw_source_byte_bound()` and `raw_raster_byte_bound()` define these shared production reservations.

The serial reservation includes one decoded source, the HDR upload conversion when applicable,
one packed readback, bounded decoder/encoder pipes, and one HDR conversion chunk. Overlap reserves
two source leases, the HDR upload conversion, and three packed snapshots: encoder-owned, in
handoff, and newly read back. If overlap exceeds the configured hard cap, the worker uses one
source and synchronous encoder handoff. If the minimum reservation exceeds the cap, conversion
fails. The attested peak is this conservative handoff bound, not total process RSS, codec
internals, TensorRT/D3D allocations, or driver-owned NVENC surfaces.

The retained `sbs_perf.json` snapshot includes decoder-read and encoder-write timings after the
encoder drains. These measure transfer and backpressure as well as copying; GPU stage timings
remain separate. Compare matched source/output evidence before attributing a throughput change.

Legacy cache-limit request fields remain accepted for existing clients and persisted jobs. They
mean only the transient-raster cap. There is no whole-clip image sequence, depth/state cache,
scene replay, per-scene encoder, or second inference pass.

SDR encoding does not decode the source again. Static HDR keeps a second aligned source decode
solely to donate per-frame mastering-display and content-light metadata. Every encoded pixel
comes from the packed raw stream. Replacing this donor requires equivalent HEVC/AV1 evidence at
every random-access keyframe; a looped single donor frame does not establish that equivalence.

## Causal diagnostic pages

The first epoch begins at sequence 1. An authenticated hard-cut pulse with its matching count
transition at sequence S closes [start, S) and opens the next epoch at S. EOF closes the final
epoch. Pulse/count disagreement fails closed. Reports identify this as `causal-production-exact`,
with `lookahead: false`.

Conversion duration is independent of the old 1,920-scene inline-document limit. The worker keeps
at most 32 recent scenes and one pending page. Immutable pages contain at most 128 scenes and
2 MiB, with matching boundary records. Finalized pages are never rewritten. The version 4
`scene-audit.json` manifest records page SHA-256 digests, exact byte counts, contiguous scene/frame
coverage, and totals. Worker-result schema 3 authenticates that manifest and carries the recent
tail. Previous inline schemas remain readable for retained jobs.

Storage still has explicit limits: 1 GiB of audit pages, 16,384 page descriptors, a 32 MiB manifest,
and a 16 MiB worker result. Exhaustion produces a storage error. These limits do not control cuts,
camera resets, or scene splitting. Checkpoints flush a small page and publish the compact
manifest; completion publishes the final manifest after output verification.

The manager validates pages one at a time, including hashes, byte bounds, sequence coverage, and
the result's final scene tail. Downloads revalidate the selected page. The UI explicitly offers
the manifest and numbered page downloads, with pagination of the page list. A manifest download
does not claim to include all scene records.

## Media and publication

- Input is one selected non-empty regular file. Its first video stream must be progressive,
  square-pixel, fixed-raster, and stable in color description.
- PTS and durations remain exact rational values. MP4 must match the source exactly; Matroska may
  differ by at most one output tick per frame, without cumulative drift.
- SDR remains SDR. Static BT.2020 PQ/HLG retain color and metadata in 10-bit output.
- Dynamic HDR10+, Dolby Vision, changing HDR metadata/raster, rotation, interlace, alpha, and
  unsupported semantic side data fail closed.
- Supported audio, subtitles, data, attachments, chapters, language, and titles are preserved
  under the stream-inventory contract. Subtitle dispositions are cleared where required.
- HEVC and AV1 use NVENC. Output is `.mkv` or `.mp4`; existing destinations are never overwritten.

Analyze Only emits source, scene, progress, and terminal attestations without encoding. Convert
Video additionally verifies compressed video, muxed streams, timing, HDR/color, staging identity,
and complete output before publication. Success requires exactly one scheduled/enqueued TensorRT
update per source frame, selected-input/full-frame scope, the live V2 renderer, complete immutable
SBS publication, and zero cache/replay use.

## Job ownership and recovery

The native manager owns production conversion; Python and tools/sbsbench own evaluation only.
FFmpeg/FFprobe must be the approved installation-local pair. One offline job may hold the GPU lease.
Admission refuses work while streaming or local AR presentation owns it.

The exact SHA-256 authenticates the worker specification. Job-root and staging-file identity pins
constrain writes and cleanup. Cancellation terminates the entire worker process tree, including
an unresponsive GPU owner and media children. Memory handoffs also wake on cancellation. Restart
marks interrupted jobs; raw frames and transient snapshots are never resumed. Durable job state,
audit evidence, and output publication retain their existing guarantees.

The Web UI selects a file, operation, output name, and codec; monitors frame progress; downloads
diagnostic pages; and clears retained job artifacts. Clear preserves published video. Interrupted
staging cleanup removes only the identity-attested staging link after confirming the published
file. Replaced paths and a possible last intact copy are preserved.

## Verification

The joint local gate covers raw handoff ordering/cancellation/bounds, negotiated FEC sizing,
retained transport generations, permission release, capture clock units/retry scheduling,
authenticated pages, and client HTTP/presentation transactions.

The opt-in `tests/integration/Invoke-OfflineSbsWorkerSmoke.ps1` runs the actual worker on a small
deterministic fixture. It checks raw transport, causal state, page digests, frame order,
timestamps, auxiliary streams, attachment bytes, output identity, and failed-job publication.
Run SDR, PQ, and HLG with HEVC/AV1 as appropriate after the installed host releases its ports/GPU.
GPU smoke, throughput comparisons, and physical XR presentation are separate evidence; portable
CI does not claim to validate the headset compositor or display hardware.
