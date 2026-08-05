# Troubleshooting Sunshine 3D

Sunshine 3D supports Windows 11 with NVIDIA NVENC. Advice for Linux, macOS, AMD, Intel, and
software encoding does not apply to this fork.

## Start with the log

Open **Troubleshooting** in the Web UI and download the current log. Keep the timestamp of the
failure and the client mode that was active: **2D**, **Client 3D**, **Raw SBS**, or **Host 3D**.
Startup encoder probes intentionally try unsupported combinations and can print errors; diagnose
the later selected encoder and stream, not an isolated probe line.

For a repeatable Host 3D visual defect, switch to Host 3D and use **Dump 3D**. A completed dump is
an atomic source/depth/render package. Its numeric contract and replay commands are documented in
[Dump 3D and exact replay](../tools/sbsbench/DUMP_FORMAT.md).

## Web UI or pairing

### The Web UI does not open

1. Open `https://localhost:47990` on the host.
2. Confirm that only one `sunshine.exe` process is running; the installed copy and a development
   build cannot own the same ports simultaneously.
3. Check Windows Defender Firewall and the Web UI address policy.
4. If a reverse proxy is used, preserve the original HTTPS `Host` and `Origin` values.

Local trusted access does not require a sign-in. WAN access does when it is explicitly enabled.

### PIN pairing fails

Use the four-digit PIN shown by Moonlight 3D in Sunshine 3D's **Enter PIN** card. PIN is the
primary flow; QR/OTP is secondary. Verify that the PC and headset can reach one another and that
the client is using the modern encrypted protocol. The first successfully paired client receives
full permissions; later clients may need launch/input permissions granted in the Web UI.

### Reset WAN credentials

Stop the running host and run the packaged binary from an elevated Windows terminal:

```powershell
sunshine.exe --creds <new-user> <new-password>
```

This changes Web UI credentials; it does not pair or unpair Moonlight 3D clients.

## Connection, latency, and stutter

### Separate network loss from host load

The client overlay and host log should be read together:

- rising network latency, packet loss, or decode queueing points to the link or client;
- rising capture, depth-inference, warp, or encode time points to the host GPU;
- a stream that throttles from 90 to 72 FPS should report the new delivered cadence on both ends;
  the host must follow the client rather than continue producing unused 90-FPS work.

Enable `diagnostics = enabled` only while measuring. It adds bounded GPU timing and system
sampling; turn it off for ordinary use. See [Configuration](configuration.md#diagnostics).

### Network checks

Prefer Ethernet for the PC and strong Wi-Fi for the headset. A stable link with low jitter matters
more than peak bandwidth. To test the reverse PC-to-headset path with a client that provides
iPerf3:

```powershell
# PC
iperf3.exe -s

# client
iperf3 -c <PC-address> -t 60 -u -R -b 50M
```

Increase the test bitrate toward the intended stream rate. Packet loss should remain near zero;
large jitter or loss warrants lowering bitrate before changing encoder or 3D settings.

### Host GPU is saturated

- Keep the NVIDIA driver current.
- Do not force an adapter that is different from the display's capture adapter.
- Leave `sbs_3d_cuda_graph = true`; disable it only for a controlled driver diagnosis.
- Reduce stream resolution or bitrate before selecting slower NVENC quality options.
- Close other GPU-heavy applications while collecting Host 3D performance evidence.
- Do not use an unoptimized Debug build for streaming tests.

NVIDIA Fast Sync can help some 2D workloads but is not a universal stutter fix. Test changes one at
a time and keep before/after logs.

## Virtual display

### Virtual Display is unavailable

Confirm that the bundled SudoVDA driver is installed and that Sunshine 3D is running with the
privileges required to create and remove a monitor. Launch the built-in **Virtual Display** app;
Raw SBS and negotiated portrait/landscape modes depend on that managed session.

### Taskbar running indicators disappear after disconnect

Windows Explorer can fail to rebuild taskbar buttons after a virtual monitor is removed. Enable
**Restart Explorer after virtual-display removal** in **Configuration > Essentials**, or set:

```ini
virtual_display_restart_explorer = on
```

The option is off by default because restarting Explorer closes the taskbar briefly, may close File
Explorer windows, and can interrupt file operations hosted by Explorer. It runs only after a
confirmed final virtual-display removal, not during warm reconnect or display replacement.

## Host 3D

### Host 3D is flat

Look for model/engine, authenticated tensor-shape, D3D11/CUDA interop, or shader-authorization
errors. V2 fails flat when current geometry is invalid; it does not silently fall back to the
deleted V1 renderer or CPU depth. Rebuild the TensorRT engine after changing the model, TensorRT
version, GPU, or authenticated shape contract.

### Depth or pop changes after a cut

V2 holds one scene camera and literal configured pop until an accepted scene cut. Capture the cut
state and exact frame identity rather than tuning unrelated geometry constants. The evidence and
rearm rules are owned by [Host SBS scene cuts](host-sbs-scene-cuts.md).

### Transparent crowns, duplicated edges, or stretched object rims

These are generally occlusion/ownership or raw monocular-depth boundary failures, not encoder
blur. Capture a Dump 3D package and compare `raw_depth.f32`, ownership-refined parallax,
vertical-conditioned parallax, final parallax, and the two-eye warp map. PNG heat maps stretch
their own ranges and cannot prove the size of a depth cliff.

The current limitations and qualification order are tracked in
[SBS 3D status and roadmap](sbs-3d-roadmap.md).

### HDR looks washed out or depth fails in highlights

Verify Windows HDR state, a 10-bit-capable codec, NVIDIA encode support, and client HDR support.
The Host 3D neural input uses the production HDR-to-SDR preprocessing contract while the encoded
color remains HDR. Do not compare a tone-mapped debug PNG with the original HDR color numerically.

## Offline conversion

### Offline job manager is unavailable

Open the **Convert** page and read its named prerequisite failure. Production conversion needs:

- the native isolated worker in the same Sunshine 3D build;
- approved `ffmpeg.exe` and `ffprobe.exe` beside `sunshine.exe` or in its sibling `tools` folder;
- compatible input media;
- TensorRT/D3D11 Host 3D readiness; and
- NVENC support for the selected H.265 or AV1 output.

The job manager does not search `PATH`, need Python, or require a separate Windows service. Path,
HDR, cache, and cancellation behavior are documented in
[Offline Host 3D conversion](whole-clip-sbs-pipeline.md).

### Conversion waits or refuses to start

Offline conversion owns an exclusive GPU lease. Stop live streaming and any local AR presentation,
then retry. A codec preflight can reject AV1 or 10-bit output even when ordinary H.264 streaming is
available.

## Input and audio

### No gamepad

Verify that the bundled ViGEmBus driver is installed. If several physical controllers are already
connected, the virtual controller may not be the device a game selects; disconnect unused devices
or choose the controller inside the game.

### Application launch is denied

Check the application's command, working directory, and permissions. Enable elevation only for an
application that actually requires it. Network and removable drives may not be visible to the
account or context that starts Sunshine 3D.

### No streamed audio

Inspect the selected `audio_sink` and `virtual_sink`. Use the packaged `audio-info.exe` to list
Windows endpoint names and IDs. A virtual sink is required when the host speakers should remain
muted while audio is captured.
