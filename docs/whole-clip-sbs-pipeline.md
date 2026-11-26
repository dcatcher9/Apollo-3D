# Offline Host 3D conversion

This document owns production offline frame transport, job lifecycle, resource bounds, encoding,
and publication. [Host SBS](host-sbs.md) owns model, geometry, and color policy. The
[evaluator guide](../tools/sbsbench/README.md) owns experiments and their evidence requirements.
[AGENTS.md](../AGENTS.md) owns machine-specific execution instructions.

Offline conversion runs the production Host SBS V2 pipeline in source order. Pop strength,
camera, OCR, conditioning, and warp rules have the same meaning as live conversion. Each frame is
rendered once, causally; nothing later in the file can revise it or choose new geometry
parameters.

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

The manager launches an authenticated isolated worker process. That worker calls the typed native
production entry once on its GPU thread, selecting Convert Video or Analyze Only. The evaluator
CLI adapts to the same causal estimator and rendering engine. Production does not construct or
parse evaluator arguments and cannot select its replay, research, or diagnostic modes. Every
source frame runs one authenticated TensorRT update and consumes its source-derived observation
timestamp. All D3D/TensorRT work stays on this single owner.

Follow conversion prepares the same current-glyph shared subtitle warp used by live rendering.
Each valid shared draw samples the original source once; unavailable subtitle programs or
authority retain authenticated ordinary Base. Optional subtitles never disable usable depth.
Production follow draws add no qualification observer or diagnostic readback.

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

The source channel has capacity two and the packed-output channel has capacity one. The separate
`bounded-completion-memory-v1` handoff owns one resolved header, one frame-evidence slot, and a
sticky terminal result. The header authenticates the generated adaptive-state identity and layout
and the resolved model, pop strength, and depth interval. Each completion owns its exact source
header and adaptive-state words, plus packed-output geometry when converting. Native evidence
preserves finite float32 bits; the evaluator's decimal JSON representation remains independent.

A separate source acknowledgement proves the worker validated that frame's exact evidence and
handed its packed output to the encoder when converting. The worker then discards the state words;
no conversion decision or retained document is derived from their values. Receiving pixels or
dequeueing evidence alone cannot release the acknowledgement. The next evidence publication waits
for the previous acknowledgement, and successful terminal publication waits for the final one.
Analyze Only uses the same barrier without packed output. Header, frame, pixel, and acknowledgement
waits use condition-variable backpressure and wake on failure or cancellation.

Production's GPU thread and worker exchange neither per-frame progress/state files nor evaluator
CLI messages. Within a phase, the worker atomically publishes manager/UI progress at most once
every 200 ms, with immediate first, phase-change, and final-source-frame updates. Only successful
publication advances the deadline. Suppressed snapshots perform no JSON or file work; filesystem
errors are reported when publication is due. The manager coalesces durable job updates. Source,
runtime, timeline, result, and publication contracts remain on disk.
Evaluator JSONL and file-follow adapters retain their existing atomic snapshots and
directory-change notifications.

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

With `diagnostics = enabled`, the worker log's periodic performance summary includes decoder-read
and encoder-write timings. These measure transfer and backpressure as well as copying; GPU stage
timings remain separate. Compare matched source/output evidence before attributing a throughput
change. With diagnostics off, a conversion collects no timings.

The manager sets the 4 GiB transient-raster cap; a create request names only the input,
operation, output name, and codec, and any other field is rejected. There is no whole-clip image
sequence, depth/state cache, scene replay, per-scene encoder, or second inference pass.

SDR encoding does not decode the source again. Static HDR keeps a second aligned source decode
solely to donate per-frame mastering-display and content-light metadata, because the raw pipe
carries no frame side data. Every encoded pixel comes from the packed raw stream: the donor is
only resized, nearest-neighbour, to the packed raster and float format that FFmpeg's `blend`
requires of both inputs, and the zero-opacity blend copies every SBS plane unchanged while its
output frame keeps the donor's side data. The donor was chosen against the bundled FFmpeg build,
whose frame-loop filters do not keep that side data on cloned frames; it is a cost to revisit,
not a format requirement. Replacing it requires equivalent HEVC/AV1 metadata at every
random-access keyframe; a looped single donor frame does not establish that equivalence.

## Retained job documents

A job retains its worker specification (schema 3), its worker result (schema 4, at most 16 MiB),
and the contracts in its result directory: source, native capabilities, and, for a conversion,
output, timeline, and publication. The specification and result are exact-key documents
exchanged only between the manager and the worker of the same build; the manager reads a worker
result once, when that worker exits, and accepts only the raw memory transport and completion
evidence that worker produces. No scene list or scene-audit file is produced.

New production contracts identify the typed runner, its condition-variable evidence wakeup, and
its memory transport without snapshot filenames. Retained file-transport contracts remain
readable under their original labels; they are never retagged as native memory evidence.

The durable job record `job.json` is schema 1. A retained record may still carry the retired
`scene_decisions`, `progress.scene_count`, and `progress.current_scene` fields and scene-audit
references in its compact worker result. They are accepted and discarded when the record is
read and are never written, so such a job stays listed, readable, and clearable. Its
`scene-audit.json` and `scene-audit-page-*.json` files are not read or served; they stay in the
job's result directory, counted against the retained-artifact quota, until the job is cleared
or pruned. A record from a release that still admitted the retired scene split may say
`"cache_budget_policy": "split"`; it is read as `fail`, the only value written.

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

Analyze Only emits source, progress, and terminal attestations without encoding. Convert
Video additionally verifies compressed video, muxed streams, timing, HDR/color, staging identity,
and complete output before publication. The decoded probe of the stream-copied final output is
that gate; the intermediate encoder stream is not decoded. Because Matroska's millisecond
timestamps allow one output tick, a Matroska job also checks the intermediate's packet
timestamps and durations exactly against the source. Success requires exactly one
scheduled/enqueued TensorRT update per source frame, selected-input/full-frame scope, the live V2
renderer, complete immutable SBS publication, and zero cache/replay use.

## Job ownership and recovery

The native manager owns production conversion; Python and tools/sbsbench own evaluation only.
FFmpeg/FFprobe must be the approved installation-local pair. One offline job may hold the GPU lease.
Admission refuses work while streaming or local AR presentation owns it.

The process facade captures its account and configuration at host startup, but starts the offline
service only after an authorized, validated offline API request. Other host features do not create
its job directories, run its media-tool checks, or start its supervisor. That first demand also
runs retained-job recovery: interrupted-state marking and publication cleanup are deferred until
then, and finish before any retained record is served or a new job is admitted. The offline
overview reports initialization while this runs. Failed initialization stays unavailable until
the host restarts; repeated status polling does not retry it. Shutdown closes new initialization
requests, stops the service, and joins its initializer before process globals are released.

The exact SHA-256 authenticates the worker specification. Job-root and staging-file identity pins
constrain writes and cleanup. Cancellation terminates the entire worker process tree, including
an unresponsive GPU owner and media children. Memory handoffs also wake on cancellation. Restart
marks interrupted jobs; raw frames and transient snapshots are never resumed. Durable job state
and output publication retain their existing guarantees.

The Web UI selects a file, operation, output name, and codec; monitors frame progress; and clears
retained job artifacts. Clear preserves published video. Interrupted
staging cleanup removes only the identity-attested staging link after confirming the published
file. Replaced paths and a possible last intact copy are preserved.

## Verification

The joint local gate covers raw handoff ordering/cancellation/bounds, negotiated FEC sizing,
retained transport generations, permission release, capture clock units/retry scheduling, and
client HTTP/presentation transactions.

The opt-in `tests/integration/Invoke-OfflineSbsWorkerSmoke.ps1` runs the actual worker on a small
deterministic fixture. It checks raw transport, causal state, frame order, timestamps, auxiliary
streams, attachment bytes, output identity, and failed-job publication. Its PQ fixture carries
HDR10 mastering-display and content-light metadata, which every output frame must keep.
Run SDR, PQ, and HLG with HEVC/AV1 as appropriate after the installed host releases its ports/GPU.
GPU smoke, throughput comparisons, and physical XR presentation are separate evidence; unit
tests do not validate the headset compositor or display hardware.
