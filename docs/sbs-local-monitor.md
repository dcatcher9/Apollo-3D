# Local monitor with shutter glasses

Sunshine 3D can present directly to a selected physical Windows monitor using an Open3DOLED USB
emitter and compatible active shutter glasses. This local display surface sits alongside remote
VR-headset streaming and [local AR glasses](sbs-local-ar-glasses.md). It uses D3D11 directly and does
not require Moonlight, network streaming, or NVENC.

The selected monitor shows alternating full-screen left and right images. Sunshine's shared 3D
pipeline produces a side-by-side stereo pair internally; the physical monitor receives each eye
sequentially. Host 3D uses the production [Host SBS converter](host-sbs.md). Game 3D receives the
game's stereo source through the existing [Game 3D setup](reshade-sbs.md). There is no separate
monitor depth or geometry implementation.

## Setup and daily use

1. Connect the physical monitor and Open3DOLED emitter to the PC. Choose a stable, fixed high
   refresh rate for that monitor in Windows Display settings. Disable G-SYNC, FreeSync, and other
   variable-refresh features for the selected output.
2. Open **Settings → Local monitor** in Sunshine 3D. Enable the capability, choose the physical
   monitor, and select its emitter COM port. The monitor is stored by its exact Windows device
   path rather than its current `DISPLAY` number.
3. Select **Host 3D** for AI conversion of captured content or **Game 3D** for the game's stereo
   export. Game 3D uses the existing game setup and source-readiness behavior.
4. Select the glasses' IR protocol and calibrate the shutter delay and duration for the panel.
   The default protocol is Panasonic custom firmware. Timing values are microseconds; a longer
   shutter window can increase brightness and also expose scanout transitions as crosstalk.
5. Save and restart Sunshine 3D. Enabling the capability does not start a session automatically.
6. Choose **Start 3D** on the home page or in Local monitor settings when you want to use the
   glasses. Choose **Return to 2D** to restore the ordinary Windows desktop.

The Sunshine 3D tray menu offers the same session controls. **Ctrl+Alt+Shift+3** toggles a local
monitor session; **Ctrl+Alt+Shift+0** returns to 2D. The shortcuts are registered while the monitor
capability is enabled. If another application has reserved one, use the tray or web controls.

Start creates a private virtual desktop on the monitor's GPU at its native per-eye dimensions.
The private source replaces the selected monitor at that monitor's original desktop rectangle and
keeps its original primary-display role. The physical sink is parked outside the interactive row
for stereo scanout. Other monitors remain usable in their existing positions. Sunshine transfers
the selected monitor's windows and cursor to the private source automatically; you do not need to
drag applications to a separate virtual screen.

Minimized placements are retained, and newly opened application windows that remember the parked
physical output are transferred automatically. Windows that the user moves onto another monitor
remain there. Native input goes directly to the application on the virtual source; the presenter
does not activate itself or reinject keyboard or mouse events.

Returning to 2D stops local capture, restores the selected physical monitor, and hands its windows
and cursor back through the shared display lifecycle. A connecting or active remote session takes
priority over local presentation. Use borderless fullscreen for games so their windows follow the
desktop handoff. A game tied to an exclusive physical output may need a switch to borderless or
windowed mode before entering local 3D.

Unlock Windows and dismiss its lock or sign-in screen before starting 3D. A lock, sign-in, or
permission-screen transition stops the monitor session and restores 2D. Start 3D again after
returning to the ordinary desktop.

**Refreshes per eye** controls how many physical refreshes repeat one eye before switching.
For example, a 330 Hz monitor with two refreshes per eye switches eyes at 165 Hz and presents
82.5 complete stereo pairs per second. Repeated refreshes provide a longer stable image for
calibration. This setting does not add black frames or change the monitor's Windows mode.

The correct delay and duration depend on the monitor's actual scanout and the glasses. Verify
left/right orientation, brightness, and crosstalk with the physical kit. A USB sync command does
not prove that a particular panel or timing calibration is correct.

## Emitter and presentation contract

The native Windows emitter implementation uses a 115200 baud, 8N1 COM connection. It first reads
the firmware settings and stops optical triggering and averaging while preserving the old protocol
and timings. After verified readback and about 100 ms for outstanding timer work to drain, it applies
and verifies the selected profile, then waits another drain interval before sending eye commands.
No settings are saved to EEPROM. Supported protocols have separate open/close commands: NVIDIA 3D
Vision (`2`), Sony (`4`), Panasonic (`5`), and Panasonic custom firmware (`6`, the default kit).
Averaging remains disabled, so stopping presentation leaves no autonomous trigger sequence running.
The authoritative native `timing_limits` helper accounts for each protocol's IR tokens and a 100 µs
scheduling margin. Delay must provide at least 100 µs of comparator lead (150 µs for NVIDIA); duration
must accommodate the complete opening token and spacing before the close command. Delay plus duration
must stay below the 32768 µs Timer1 wrap and fit the selected eye interval. The close-token queue tail
must also drain within that interval. Timer4's 1023 µs capacity includes both the IR token and spacing,
so the maximum spacing is 882 µs for NVIDIA, 483 µs for Sony, and 803 µs for Panasonic. These bounds
are current-eye presentation policy and firmware interoperability limits, not physical panel
calibration. The host eye interval is independent of the unused firmware averaging filter, so slow
refresh/repetition combinations do not wrap a 16-bit filter value. Serial work is bounded. A failed
emitter connection or sync write stops the presentation attempt rather than continuing without
synchronization.

Before sending the first eye command, the presenter holds one left-eye image while acquiring the
selected output's clock. It requires 13 consecutive exact present identities, 12 uninterrupted
refresh intervals, and QPC intervals within 100 µs of the selected fixed refresh and repeat count.
Cold timeouts and counter-domain changes restart acquisition within a two-second budget; the
per-present wait scales with the eye interval. Waiting for the first stereo packet has a separate
five-second limit. No eye commands are sent during this startup phase.

Once armed, a missing frame, counter discontinuity, or invalid timestamp stops presentation. Each
confirmed frame must also be recent enough for the configured delay, duration, spacing, and
protocol close margin to fit the remaining eye interval. Freshness is checked again immediately
before the USB command. These checks bound known host lateness; USB delivery time and panel
response still require calibration with the physical kit.

The presenter retains one stereo tuple for its complete left/right cycle. Host AI 3D and Game 3D
keep their existing source, frame-attribution, fallback, HDR, and geometry rules. Monitor mode,
device identity, and emitter configuration belong to this output surface; they do not alter those
shared pipeline contracts.

## Configuration

| Option | Default | Meaning |
|---|---|---|
| `local_monitor_enabled` | `disabled` | Allow explicitly requested monitor sessions |
| `local_monitor_display_id` | empty | Exact physical Windows monitor device path |
| `local_monitor_mode` | `host` | `host` or `game` |
| `local_monitor_emitter_port` | empty | Open3DOLED COM port, such as `COM3` |
| `local_monitor_ir_protocol` | `6` | Finite-window protocol: NVIDIA 3D Vision (`2`), Sony (`4`), Panasonic (`5`), or Panasonic custom firmware (`6`) |
| `local_monitor_frame_delay_us` | `500` | Delay from sync to shutter window; minimum `150` µs for protocol `2`, `100` µs for `4`/`5`/`6`; maximum `32767` µs |
| `local_monitor_frame_duration_us` | `2000` | Shutter-open duration; minimum configured spacing plus `300` µs for protocol `2`, `640` µs for `4`, or `320` µs for `5`/`6`; maximum `32767` µs |
| `local_monitor_signal_spacing_us` | `30` | IR signal spacing; `30`–`882` µs for protocol `2`, `30`–`483` µs for `4`, or `30`–`803` µs for `5`/`6` |
| `local_monitor_flip_eyes` | `disabled` | Swap left/right shutter assignment |
| `local_monitor_eye_repeats` | `1` | Consecutive refreshes per eye, `1`–`4` |

Invalid mode, boolean, timing, protocol, or repeat values disable the monitor capability during
configuration loading. An enabled configuration also requires both a physical monitor selection
and an emitter port. Configuration loading never starts a 3D session by itself.

Delay plus duration must not exceed `32767` µs and must fit the assigned eye interval: the monitor's
physical refresh period multiplied by **Refreshes per eye**. Duration plus the protocol's close-pulse
margin must also fit that interval. These limits leave room for the emitter's IR pulse and scheduling.
The UI receives the native protocol limits and checks the interval when the selected monitor is
connected; the emitter driver validates the actual target before starting.
The `30` µs minimum spacing is a conservative host floor for the firmware's greater-than-`5` µs
gap-arm lead and observed `17` µs service overhead.

## Live qualification

Native tests exercise the shared converter, immutable stereo packets, delayed presentation
feedback, firmware command validation, and display transactions with injected Windows I/O. They
do not establish the physical panel's scanout phase or confirm application interaction after a
real topology change. Qualify a monitor/emitter profile with the optimized development host:

1. Open a normal browser or video player on the selected monitor before starting Host 3D. Check
   typing, clicking, seeking, the cursor, and movement between that monitor and an unchanged 2D
   monitor. No manual move to a virtual screen should be required.
2. Minimize and restore an application, then launch another application that remembers this
   monitor. Verify both remain reachable, and that windows deliberately moved to another monitor
   stay there. Return to 2D and verify the windows, monitor roles, and layout are restored.
3. Repeat with a configured Game 3D source in borderless fullscreen. Check both stereo content
   and game input; a converter test alone does not qualify the game's window/display binding.
4. Use a left/right test pattern with the actual glasses to verify eye assignment, shutter phase,
   brightness, and crosstalk. Record the Windows refresh, repeat count, glasses protocol, and
   delay/duration/spacing that worked. Software presentation feedback is not optical measurement.
5. Verify Return to 2D and the emergency shortcut, then verify USB loss restores the desktop.
   Check remote-stream priority and local restart separately using the shared display lifecycle.

Follow the machine's local development-launch instructions. Keep the installed and development
hosts mutually exclusive. A profile remains unqualified until these live observations are made;
mock API, offscreen GPU, and fake topology results cannot substitute for them.
