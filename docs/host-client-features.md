# Client microphone and DualSense haptics

Sunshine 3D supports optional client microphone forwarding and game-authored DualSense PCM
haptics on Windows. Both require a compatible client and a working host backend. The existing
encrypted Artemis streaming and SBS contracts remain in effect; adding these features does not
broaden the host's supported client protocols.

## Optional installer components

Windows NSIS packages built with optional client-driver support offer two unchecked components:

- **Microphone forwarding (VB-CABLE)** bundles the original VB-Audio Pack45 ZIP. Setup opens the
  vendor installer; complete that dialog and reboot before using the cable. The component identifies
  VB-Audio and its donationware model. Silent setup uses the pinned installer's verified install-only,
  hidden mode; see the unattended command below.
- **DualSense haptics (Internet required)** includes the self-contained Sunshine helper and downloads
  the pinned HIDMaestro 1.6.2 runtime directly from its upstream release when selected. Both archive
  and DLL hashes are checked before execution. Setup installs missing HIDMaestro/USB-IP backends,
  verifies readiness, and activates the component only after success. It adds a local signing
  certificate and USB devices may briefly reconnect. No .NET or Windows SDK installation is needed
  on the end user's PC.

The installed Sunshine service is stopped for driver setup; close any manually launched host first.
For unattended setup, run `Sunshine3D.exe /S /CLIENTDRIVERS` from an elevated terminal. Select only
one feature with `/MICROPHONE` or `/DUALSENSE`. Exit `0` means success, `3010` means success with a
Windows restart required, and `1` means optional setup failed. No automatic reboot is requested.
Setup failures are shown and logged in `<install directory>/config/driver-setup`; ordinary streaming
remains available and setup can be rerun. A failed driver installation can leave partial system
changes even when the previous helper remains active. The optional drivers are retained when
Sunshine is uninstalled because other applications may use them. Host feature settings and paired
client permissions are still configured below; installation does not grant microphone access.

Build/package instructions are in [Building Sunshine 3D](building.md#package). These components do
not supply missing client microphone capture or PCM rendering implementations.

## Microphone forwarding

1. Select the installer component above, or install [VB-CABLE](https://vb-audio.com/Cable/) using the vendor's administrator instructions,
   including its reboot requirement. The driver is a separate dependency.
2. Run `tools\audio-info.exe` and find the cable's playback endpoint, usually
   `CABLE Input (VB-Audio Virtual Cable)`.
3. In **Configuration > Audio/Video**, enable **Allow Client Microphone Forwarding** and enter
   that exact endpoint name or ID in **Microphone Virtual Cable Endpoint**. Save and restart the host.
4. In the paired client's permissions, grant **Microphone Input**. Existing clients retain their
   previous permissions until explicitly edited.
5. Select the cable's matching recording endpoint, usually `CABLE Output`, in the PC game or voice
   application. Enable microphone capture in a client that implements the shared microphone API.

The receiver uses encrypted Opus, 48 kHz mono, with 20 ms frames. It writes only to the explicitly
selected, active VB-Audio cable pair. It does not select speakers, replace the default microphone,
or forward the desktop audio mix. A missing cable or denied permission disables the feature during
negotiation. Optional microphone setup failure leaves ordinary streaming available.

Microphone traffic uses UDP `base port + 12`, **48001** with the default base port. The Windows
Sunshine firewall rule already permits the application's UDP traffic. Add this port to any manual
router or restrictive external firewall rules when forwarding a microphone across that network.

The RTSP launch owns a dormant receiver until a validated stream starts. Reservation cancellation,
disconnect, or microphone permission revocation stops and joins it before a later session reuses
the port. A new session uses its own key, nonce, client address, and UDP source port. Granting the
permission again requires reconnecting to negotiate a new receiver.

The compatible legacy microphone wire format uses AES-CBC, **not authenticated encryption**.
The session nonce ping binds the expected IP and source port; sequence checks reject duplicate and
old packets. Header and padding behavior intentionally match the shared client's existing sender.

## DualSense controller and authored haptics

Build the optional helper with [the component build instructions](../tools/sunshine-ds5-sidecar/README.md).
With Sunshine stopped, place the complete verified component at:

```text
<directory containing sunshine.exe>/tools/sunshine-ds5-component/active/
```

Select the optional installer component above, or install the pinned HIDMaestro 1.6.2 runtime
drivers using its administrator setup procedure.
The standard controller profile needs the UMDF2 backend; the composite audio profile also needs
usbip-win2. The helper runs with the host's existing elevated token. Sunshine does not install or
elevate drivers during a connection.

Select **DualSense (PS5, optional driver)** in **Configuration > Input** and leave **Forward
DualSense PCM Haptics** enabled. The corresponding settings are:

```ini
gamepad = ds5
ds5_audio_haptics = enabled
```

The virtual controller accepts input, motion, touchpad and battery state. Its output preserves
ordinary rumble, adaptive triggers and lightbar updates. The composite profile exposes a
four-channel controller audio endpoint: only authored haptic channels 3/4 are forwarded as
48 kHz, stereo, signed 16-bit PCM. Speaker channels 1/2 are excluded. This requires a PC game with
native DualSense audio haptics and a client that consumes the PCM callback.

`gamepad = auto` keeps the existing ViGEm selection. An unavailable optional backend falls back
to ViGEm. If a running helper fails and its one restart does not recover, the next controller event
switches to ViGEm while retaining the controller's host/client indices and feedback queue. Failed
ViGEm attachment is retried on input at most once per second; controller removal cancels pending
fallback. Missing audio support or client opt-in withholds PCM while standard controller feedback
remains available. PCM uses a separate bounded queue and unreliable sequenced control delivery;
late waveforms cannot accumulate through reliable retransmission or erase queued rumble stops.

## Shared client contract

The shared `moonlight-common-c` migration is published at
`3a235790931e8092500e215f4864e696147bf3f6`. Client applications must still capture microphone audio
and register/render authored PCM using that API. A shared-library update alone does not enable
the client UI or its platform audio/controller devices. This host work includes no iOS changes.

Negotiation preserves Sunshine 3D's existing SBS feature bits:

| Feature | Negotiation |
| --- | --- |
| Microphone encryption | Encryption namespace `0x08`; optional RTSP `SETUP .../mic` |
| Authored PCM | Host `0x80` plus capability-profile marker `0x08000000`; client `0x20` |
| Controller PCM opt-in | Controller capability `0x200` |
| PCM control message | `0x550a`, version 1, 28-byte header, at most 240 stereo frames |

The optional authored IR format is not advertised. Advertising both client PCM and IR formats
is rejected. Clients that do not negotiate these optional features continue through the existing
media and controller paths.

## Qualification

Automated coverage exercises the real microphone UDP/Opus/CBC receiver, activation and teardown,
malformed packets, bounded buffering, native helper IPC, owned PCM copies, channel isolation,
feedback routing, cancellation, and the managed helper's protocol/runtime self-checks.

After the normal host build, run the feature suites with the toolchain runtime on `PATH`:

```powershell
cmake-build-relwithdebinfo\tests\test_sunshine.exe --gtest_filter="ClientFeatures.*:Microphone.*:RtspMicrophoneLifecycleTest.*"
ctest --test-dir cmake-build-relwithdebinfo/tests -R "^ds5_(component|sidecar_client)_unit_tests$" --no-tests=error --output-on-failure
```

The [joint workflow checks](joint-workflow-tests.md)
also compile the production host encoder directly with the shared C client's PCM parser.

End-to-end device qualification additionally requires the installed virtual cable and controller
drivers, a supporting client, and a game with authored DualSense output. Check microphone
pause/resume and reconnect, permission revocation, separate left/right haptics, rumble stops,
adaptive triggers, and recovery after helper failure. A successful build or readiness probe does
not establish those physical results.
