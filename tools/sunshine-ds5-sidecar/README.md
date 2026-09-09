# Sunshine 3D virtual DualSense helper

This optional Windows helper owns a virtual DualSense and returns the game's
controller output to Sunshine 3D over the versioned `SDS5` named-pipe protocol.
The standard profile supports input, motion, touchpad, rumble, adaptive triggers,
and lightbar output. The composite profile also presents a four-channel USB audio
endpoint: channels 1/2 are the controller speaker and channels 3/4 carry authored
haptic PCM. Only channels 3/4 enter the haptics transport.

The helper source is adapted from
[Sunshine-Foundation 3e142f7d](https://github.com/AlkaidLab/foundation-sunshine/tree/3e142f7d192ea26a4283ca8c9ab7c7d30db62023/tools/sunshine-ds5-sidecar)
under GPL-3.0-only. It uses the public
[HIDMaestro v1.6.2 API](https://github.com/hifihedgehog/HIDMaestro/tree/v1.6.2),
whose Core assembly is an external dependency. No HIDMaestro binaries are committed
to this repository.

## Build a local component

Use a .NET 10 SDK, including a portable SDK if the existing machine toolchain is
older. HIDMaestro Core itself targets .NET 10; changing only this project's target
to .NET 8 does not make the dependency compatible. The native Sunshine build still
uses its existing MSYS2 UCRT64 toolchain.

```powershell
.\scripts\build-ds5-sidecar.ps1 `
  -Dotnet C:\path\to\dotnet-10-sdk\dotnet.exe `
  -HIDMaestroArchive C:\path\to\HIDMaestro-v1.6.2.zip `
  -OutputDirectory C:\path\to\new-empty-component-directory
```

Omitting `-HIDMaestroArchive` downloads the pinned official asset to the repository's
ignored `build/ds5-sidecar-cache` directory. The script verifies the release archive
and Core assembly, publishes a self-contained x64 helper, runs `--self-check`, and
writes a `component.json` SHA-256 manifest for every component file. It does not
install drivers or certificates, elevate a process, modify Sunshine configuration,
or activate the component. It requires a fresh output directory and never deletes
an existing component. Restore uses the official NuGet service for Microsoft .NET
and Windows SDK reference/runtime packages.

Pinned dependency:

| Property | Value |
| --- | --- |
| Asset | [HIDMaestro-v1.6.2.zip](https://github.com/hifihedgehog/HIDMaestro/releases/download/v1.6.2/HIDMaestro-v1.6.2.zip) |
| Archive size | 118,881,819 bytes |
| Archive SHA-256 | `6ae8df0cf317baf7e65777e2929f618916a67831b5ff1162205310f2c08b80ff` |
| Core file version | `1.6.2.0` |
| Core SHA-256 | `df59cd5b6031311f03031c32b794901fc1f45503410f4f6ad55eaa854cee8746` |

The HIDMaestro source is [MIT licensed](https://github.com/hifihedgehog/HIDMaestro/blob/v1.6.2/LICENSE).
Its composite transport bundles unmodified usbip-win2 0.9.7.7 under BSD-2-Clause;
the [upstream notices](https://github.com/hifihedgehog/HIDMaestro/blob/v1.6.2/sdk/HIDMaestro.Core/THIRD-PARTY-NOTICES.txt)
identify that binary and its digest. The build keeps those notices with the local
component. The full Core also embeds Windows driver signing/catalog tools. This
local development recipe does not add Core or its upstream package to the Sunshine
installer or a Sunshine release asset. Review all embedded payload redistribution
terms before publishing a binary component containing them.

## Runtime setup

With Sunshine stopped, place the complete locally built directory at:

```text
<directory containing sunshine.exe>/tools/sunshine-ds5-component/active/
```

Keep `component.json` and every hashed file together. The host checks that fixed
directory, verifies the manifest, and rejects modified, incomplete, or additional
executable/runtime files. The location is relative to `sunshine.exe`, even when
the host uses a configuration file in a different directory.

HIDMaestro driver setup remains a separate administrator action performed through
the pinned upstream runtime. The standard profile requires its UMDF2 driver. The
composite profile requires usbip-win2 0.9.7.7. Follow
[the pinned upstream setup documentation](https://github.com/hifihedgehog/HIDMaestro/tree/v1.6.2#quick-start)
for that system change. Sunshine does not install a driver during controller attach.

Select `gamepad = ds5` to opt in. `ds5_audio_haptics = enabled` permits authored PCM
when both the runtime and client support it. `gamepad = auto` retains the existing
ViGEm controller selection. The host runs the helper with its existing elevated
token, owns it through a Job Object, and does not request elevation itself. The
named pipe accepts only its owning elevated user and every controller is disposed
when the owner disconnects.

Read-only readiness inspection, safe before driver setup:

```powershell
& C:\path\to\component\Sunshine.Ds5Sidecar.exe --probe
```

The JSON reports `runtime_version`, `elevated`, `standard`, `composite`,
`driver_installed`, and `usbip_available`. `standard` and `composite` indicate
profile availability; the host also requires elevation and the corresponding
installed backend before advertising the feature. The internal
`--probe-host` option returns a readiness bitmask for native code; use the ordinary
`--probe` option for human inspection.

## Verification

The build's deterministic check can be repeated without elevation or devices:

```powershell
& C:\path\to\component\Sunshine.Ds5Sidecar.exe --self-check
```

It checks the real protocol encoding, channel isolation, output validity flags,
controller state handling, and the pinned public runtime contract. It does not
prove a driver can create a device on a particular Windows installation or that a
game and remote physical controller deliver authored haptics end to end.

After the administrator has installed the required runtime drivers, run the
elevated device checks separately:

```powershell
& C:\path\to\component\Sunshine.Ds5Sidecar.exe --self-test standard
& C:\path\to\component\Sunshine.Ds5Sidecar.exe --self-test composite
```

Then stream with a client that registers the authored PCM callback, launch a game
with native DualSense support, and verify independent adaptive triggers, ordinary
rumble, lightbar, and authored haptic output. Verify reconnect and helper failure
remove the virtual controller and do not affect the next session. The helper sets
the virtual audio interfaces' never-default policy and exits on a detected default
audio endpoint violation; it does not replace the user's selected default endpoint.

## Microphone boundary

Sunshine 3D's microphone path uses the explicitly configured WASAPI render endpoint
of an active VB-Audio virtual cable pair. Select its capture side in the receiving
application. A physical speaker endpoint does not provide a microphone to a game.
[VB-CABLE](https://vb-audio.com/Cable/) supplies that external virtual cable;
its [donationware terms](https://vb-audio.com/Services/licensing.htm) allow testing
before payment and specify payment for professional use. Install it separately
using the vendor's administrator instructions and reboot requirement. The host
does not download or install it, and does not change the system default microphone.

The inherited helper's composite microphone prototype remains explicitly opt-in
for developer tests. Its presence is not a claim that Sunshine's production
microphone sink uses the helper. It is not advertised during ordinary helper startup.
