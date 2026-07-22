# Apollo XR Host

Apollo XR is a Windows desktop-streaming host for
[Artemis](https://github.com/ClassicOldSong/moonlight-android) on Android XR. It is a hard
fork of Sunshine, but its runtime contract is intentionally narrower: one active XR session,
modern encrypted Artemis protocol, and native NVIDIA NVENC encoding.

## Supported host

- Windows 11
- NVIDIA GPU with native NVENC support
- H.264, HEVC Main/Main10, and AV1 Main/Main10 in 4:2:0
- Artemis protocol 13 client on Android XR

AMD, Intel, software encoding, Linux/macOS hosts, legacy Moonlight protocol variants, multiple
simultaneous sessions, input-only sessions, UPnP, and remote file/server-command features are
not supported.

## XR features

- Per-client SudoVDA virtual displays with resolution, refresh-rate, and HDR negotiation
- Host-side AI 2D-to-SBS conversion using Depth Anything V2 and TensorRT
- Local AR-glasses presentation for directly connected displays
- HDR10 and full/limited-range video metadata
- Desktop audio, stereo/surround virtual sinks, keyboard, mouse, touch, pen, gamepad, and
  clipboard synchronization
- A warm reconnect window that retains the single session's app and virtual desktop
- Web UI for configuration, pairing, applications, and client permissions

Only the first authenticated stream owns the host session. A second launch is rejected until
that session exits. A short disconnect can resume the same virtual desktop during the configured
`session_resume_grace`; quitting the session or expiry performs full teardown.

## Development build

The supported development toolchain is MSYS2 UCRT64 on Windows. Official Node.js must precede
MSYS2 on `PATH` for the Vue Web UI build.

```powershell
$env:MSYSTEM = "UCRT64"
$env:MSYS2_PATH_TYPE = "inherit"
C:\msys64\usr\bin\bash.exe -lc 'export PATH="/c/Program Files/nodejs:$PATH"; cd /e/Git/Repo/Apollo-3D; cmake -B cmake-build-relwithdebinfo -G Ninja -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo; ninja -C cmake-build-relwithdebinfo'
```

See [AGENTS.md](AGENTS.md) for the local Galaxy XR build/deploy/debug loop,
[CLAUDE.md](CLAUDE.md) for architecture and validation guidance, and
[docs/configuration.md](docs/configuration.md) for the active configuration surface.

## Runtime notes

- The installed Apollo XR instance and a development instance cannot bind the streaming ports at
  the same time.
- Apollo XR needs elevation for virtual display and input support.
- The first paired client receives full permissions. Later clients must be granted the required
  launch, input, and clipboard permissions in the Web UI.
- Steam Streaming Speakers is installed best-effort when Steam is available and is used for
  muted-host and surround-audio streaming.

Apollo XR retains upstream internal names such as `sunshine.exe`, `test_sunshine`, and some
configuration/state paths. Those names are implementation lineage, not compatibility promises.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
