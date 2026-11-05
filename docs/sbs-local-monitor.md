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

A Windows sign-in or permission-screen transition stops the monitor session, because the normal
presenter and its shortcuts cannot operate on the secure desktop. Start 3D again after returning
to the ordinary desktop.

**Refreshes per eye** controls how many physical refreshes repeat one eye before switching.
For example, a 330 Hz monitor with two refreshes per eye switches eyes at 165 Hz and presents
82.5 complete stereo pairs per second. Repeated refreshes provide a longer stable image for
calibration. This setting does not add black frames or change the monitor's Windows mode.

The correct delay and duration depend on the monitor's actual scanout and the glasses. Verify
left/right orientation, brightness, and crosstalk with the physical kit. A USB sync command does
not prove that a particular panel or timing calibration is correct.

## Emitter and presentation contract

The native Windows emitter implementation uses a 115200 baud, 8N1 COM connection. It reads the
firmware settings, applies PC serial drive mode and the selected timings without saving EEPROM,
then sends left/right sync commands from the presentation path. Serial work is bounded. A failed
emitter connection or sync write stops the presentation attempt rather than continuing without
synchronization.

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
| `local_monitor_ir_protocol` | `6` | Glasses protocol, `0` through `6` |
| `local_monitor_frame_delay_us` | `500` | Delay from sync to shutter window, `0`–`65000` µs |
| `local_monitor_frame_duration_us` | `2000` | Shutter-open duration, `1`–`65000` µs |
| `local_monitor_signal_spacing_us` | `30` | IR signal spacing, `0`–`65000` µs |
| `local_monitor_flip_eyes` | `disabled` | Swap left/right shutter assignment |
| `local_monitor_eye_repeats` | `1` | Consecutive refreshes per eye, `1`–`4` |

Invalid mode, boolean, timing, protocol, or repeat values disable the monitor capability during
configuration loading. An enabled configuration also requires both a physical monitor selection
and an emitter port. Configuration loading never starts a 3D session by itself.
