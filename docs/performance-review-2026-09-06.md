# Joint performance review — 2026-09-06

This is a review and measurement record, not a new rendering contract. Host behavior remains owned
by [Host SBS](host-sbs.md) and [scene cuts](host-sbs-scene-cuts.md); client rendering and measurement
contracts remain in the companion Moonlight repository's `docs/android-xr-sbs.md` and
`docs/client-sbs-evaluation.md`.

## Profile and priorities

The 08:54–08:55 host session used HEVC Main10, P1, quarter-resolution multipass and two-engine split
encoding for a 7680×2160 Host SBS output requested at 90 Hz. Nearby diagnostic windows measured:

| Boundary | Mean |
| --- | ---: |
| New captured content age at handoff | 1.62 ms |
| Capture handoff to first conversion | 1.88 ms |
| Conversion CPU call | 2.29 ms |
| NVENC encode/retrieve CPU call | 6.05 ms |
| Nested NVENC completion wait | 5.79 ms |
| SBS warp GPU region | about 0.22 ms |
| Encoded packet queue residence | about 0.01 ms |

These windows are not matched per-frame samples and nested intervals must not be added. The NVENC
wait includes unfinished GPU input work and thread wakeup as well as encoding. The serial owner
completes one output before reusing the input surface. The much smaller warp timer does not support
prioritizing shader arithmetic over encoding. The current logs do not establish the actual capture
FPS or an encoder hardware ceiling below 90 FPS.

GPU decision counters recorded 2,597 inferences and 4,466 reuses (63.23%), with no invalid decisions.
The session contained no warning/error/fatal lines. Historical processing windows overlap the
current range; the available logs do not demonstrate a new encoding regression.

## Changes applied

- **Client GPU setup:** private fixed-shape depth and disparity programs initialize immutable
  uniforms once. This removes 21 Java-to-GL uploads from each normal real-inference/postprocess
  cycle. Dynamic source offsets, timing coefficients, dispatches, barriers and ownership remain
  unchanged. No shader, model, calibration or Near threshold changed.
- **Client hidden Stats:** completed decoder windows update cumulative accounting directly without
  allocating a snapshot and its histogram. The pruning clock is read only when visible telemetry
  has pending records. Reopening retains a fresh visible window.
- **Client visible Stats:** unchanged labels/values retain their text, and panel sizing shares one
  retained callback cancelled on hide or destruction. Decode scheduling and surface transitions
  are unchanged.
- **Host encoder cleanup:** unregister an async event only after its registration succeeded.
  Partial initialization no longer calls the driver with an unowned registration. Fix the older
  accepted-frame timeout path to retain its input through completion and successful lock/unlock/
  unmap. Complete failed owners retire on tracked workers, including startup probes. Replacement
  creation stays blocked until safe destruction finishes; picture and EOS completion use separate
  events. The [Host SBS owner](host-sbs.md) specifies the failure and shutdown contract. Normal
  streaming keeps its existing completion deadline and gains no extra waits or diagnostic work.
- **Host measurement clarity:** replace the misleading `source ~fps` estimator-call EWMA with
  elapsed-window `admission attempts ~/s`. These attempts include retained-source retries and are
  neither capture FPS nor actual model invocation rate. The unnecessary EWMA state/arithmetic is
  removed. Diagnostics-disabled execution still bypasses the accounting before clock reads.

Stats, operational logs, Dump 3D and indefinite valid Near reuse are retained. No additional live
probe, readback, queue or background diagnostic control was added.

## Native encoder experiment

An ignored standalone probe calls the optimized production native D3D11 NVENC implementation with
diagnostics disabled. Only multipass differs. Both paths use the same eight-frame synthetic P010
sequence, 7680×2160, 90 Hz rate control and 166,170 kbit/s target. Input upload is completed before
the encode timer. Six legs run in quarter/single/single/quarter/quarter/single order, each with 24
warmup and 120 measured frames. Submission is unpaced; this is not live stream throughput.

| Measured result | Quarter pass | Single pass |
| --- | ---: | ---: |
| Mean encode/retrieve wall time across three legs | 5.221 ms | 4.166 ms |
| Encoded payload rate at nominal 90 FPS | 104.912 Mbit/s | 163.983 Mbit/s |
| Mean per-frame ten-bit code-value PSNR | 31.050 dB | 31.018 dB |
| Mean SSIM | 0.964276 | 0.965051 |

Single pass saves about 1.055 ms (20.2%) but uses 56.3% more encoded bits for similar scores on this
synthetic sequence. **Keep the quarter-pass default and development configuration.** The existing
single-pass option is a latency/bandwidth tradeoff to qualify with representative moving video;
these code-value metrics are not perceptual HDR or visual-quality acceptance. All outputs were
nonempty and ordered, and the three streams within each mode were byte-identical. Quality scoring
aligns both decoded and reference streams by frame index at a common 1/90 time base; warmups are
excluded. An earlier timestamp-misaligned scoring attempt is not valid evidence.

The probe, input, hashes, commands, CSV, bitstreams and scores are local ignored artifacts under
`cmake-build-relwithdebinfo/nvenc-multipass-2026-09-06/`. The production pipeline does not contain
the probe or its deliberate input-upload synchronization.

## Completion-wait experiment

A second ignored probe compares the existing asynchronous event wait with the synchronous fallback
at the same quarter-pass settings. Six legs run in async/sync/sync/async/async/sync order, each with
24 warmup and 120 measured frames. The 144-frame streams and every frame's encoded size are identical
in all six runs, including to the earlier quarter-pass output. Pooled encode/retrieve means were
5.271 ms asynchronous and 5.236 ms synchronous. Paired mean differences shrank from 0.097 ms to
0.009 ms and 0.0001 ms, so there is no consistent meaningful improvement to adopt.

The first asynchronous run also had a longer input-upload wait outside the encode timer; its cause
is unknown and it did not recur. No samples are excluded. This short synthetic probe does not
establish live-stream throughput or isolated hardware encode time. Keep asynchronous completion and
its failure deadline. Frozen sources, objects, runtime identity, hashes, commands and per-frame
results are under `cmake-build-relwithdebinfo/nvenc-wait-mode-2026-09-06/`.

## Validation and remaining limits

- The later 12:03 live-session follow-up built the full optimized host and Web UI and passed 40
  focused native admission, DDup, route, poll and source-contract tests. It fixes the duplicate
  completed-source admission described below. This is separate from the earlier NVENC test gate.
- Optimized host and test executable build passed; 44 focused native tests passed, including
  partial NVENC initialization cleanup, retained-source pacing, diagnostics-disabled behavior,
  delayed picture/EOS completion, consumed-event handling, accepted `NEED_MORE_INPUT`, unlock/unmap
  retries, and replacement blocking through driver destruction.
- Client application and instrumentation APKs built on JDK 25; 258 focused JVM tests passed.
- All nine depth/disparity GLES tests passed on the Galaxy XR, including initialization, distinct
  processor instances, temporal resets, reliable-history output and production disparity shapes.
  A new test initially used a float getter for a sampler; the type-correct integer query passed
  with the production APK unchanged. The corrected test retains exact expected bindings.
- The client was update-installed, preserving app data and pairing. Only the instrumentation
  package was subsequently removed. The host executable is built for the next normal launch.

Whole-stream client FPS, device GPU utilization and end-to-end live latency improvement have not
been measured for these edits. The removed client calls/allocations are concrete work reductions,
not evidence that they explain the headset's system-wide idle GPU load.

A separate real-GPU cleanup probe runs the final native encoder normally, then forces the first
accepted picture's completion wait to report a timeout in an ignored source copy. The failed call
returns empty and another submission is rejected. All subsequent picture waits, EOS submission,
flush waits and driver destruction are real. Cleanup returns, a fresh native owner is admitted, and
both normal and replacement encoders produce the exact earlier 144-frame quarter-pass stream.
The injected timeout and resource-retention errors are expected. Probe sources and results are under
`cmake-build-relwithdebinfo/nvenc-cleanup-final-2026-09-06/`; no injection hook enters production.

This demonstrates the ownership/recovery path on the local GPU, not recovery from an actually hung
driver. Unit tests cover delayed and failed API results. Retry counts are bounded, but driver calls
can still stall; the failure contract retains their resources and requires restart if recovery
fails. The reviewed live session did not exhibit this older timeout bug. No multiframe encode queue
or longer streaming wait was added.

## Follow-up to the 12:03 live session

The full-frame path could retire a late GPU-opaque completion, reject the identical retained source
as an invalid changed-source successor, and immediately force inference again for the same captured
pixels. The host now consumes that existing completion once, or redelivers its already-packed image.
It reuses the existing slots, presentation cache and opaque anchor; no second resource cache or queue
was added. The [Host SBS contract](host-sbs.md) owns the exact source/route/epoch and invalidation gates.
The opaque observation barrier, actual inference owner, OCR/cut state and changing-source fallback
remain intact. Four new deterministic tests cover late completion, long repeated delivery, cursor
changes and source/transfer/authority changes. Live cadence improvement still needs qualification.

On the client, a reported audio recovery previously reset accumulated drop totals before its next
PCM write succeeded. Two new regression tests failed on that boundary, then passed when reporting
was moved after a complete write on the same active generation. No extra queue reads, waits, larger
buffers, threshold changes or additional audio drops were introduced. Three legacy database readers
also skip absent migration files, avoiding SQLite error stacks on already-migrated installs while
retaining real migrations. All 31 focused audio, priority and migration tests passed; the APK built
and was update-installed with app data preserved. Client build/install evidence is under the companion
repository's `app/build/audio-cadence-2026-09-06-*`.

This accounting fix does not eliminate the observed audio gaps. Five bursts align with system Wi-Fi
scan completion roughly 121.7 seconds apart. The scan requester remains unidentified; an
`AllSingleScanListener` record is not proof that connectivity selection initiated it. Current link
evidence is Wi-Fi 7 at 6295 MHz. Samsung documents Galaxy XR PC-streaming stutter on Wi-Fi 7/6 GHz and
recommends a temporary [Wi-Fi 6/5 GHz workaround](https://www.samsung.com/us/support/troubleshoot/TSG10007646/).
That is a relevant controlled test, not proof of this exact scan defect or an app-level fix. Detailed
wireless evidence and research belong to the client evaluation record. No device settings changed.

The startup pose and mode-switch subview warnings arise at the SceneCore/runtime boundary. The JNI
warning's exact native library remains unattributed without a stack; no safe app-owned fix or newer
official SceneCore release was established. The warnings were not suppressed.

The changed diagnostics and packed-width labels/descriptions are translated in all 21 other host
locales (84 additions). English regional variants match `en`; the other 19 use translations. All
22 JSON files parse and each added key appears once. Android's two removed performance-logging
strings were already absent from every translation. The full Web UI build passed.

## Host/client telemetry and source-identity follow-up

Host Stats could stop updating during the normal GPU-opaque inference/reuse path: the observation
barrier blocked new diagnostic snapshots, and only CPU-known inference completions updated their
sample identity. Health sampling now follows any authenticated completed transaction while keeping
the production ownership barrier intact. Held cut-pulse bits no longer repeatedly announce the same
cut. The client subscribes only while Host Stats is visible and rejects late packets after hiding.

The standard processing-latency field previously measured old pixel content age. It now measures the
current conversion start through send-side packetization admission; a repeated encoder input with
no conversion reports no new sample. Content-age diagnostics retain their separate measurement.
The [Host SBS contract](host-sbs.md#performance-observations) defines the exact timing boundary.

A negotiated token in existing reserved frame-header bytes lets Client SBS retain a proven static
encoder input before model input rendering, classification, color copying and packed presentation.
The [wire contract](host-sbs.md#client-exact-repeat-transport) is owned here; decoded timestamp
attribution, GPU-owner confirmation and initial lossy-decoder settling are owned by the client
architecture document. Legacy sessions follow ordinary local Near reuse. No shader, model, depth
math, encoder policy or recurring Near expiry changed.

The optimized host and native tests built successfully; all 123 focused tests across 26 suites
passed. Protocol/FEC CPU tests and 156 client JVM tests also passed, and both client debug ABI APKs
assembled. Local host evidence is under `cmake-build-relwithdebinfo/source-identity-review-2026-09-06/`;
client qualification is recorded in its `docs/client-sbs-evaluation.md`. A live reconnect remains
necessary to verify Host Stats cadence and source-repeat work avoidance on the headset. These tests
do not establish a change in whole-device GPU utilization.
