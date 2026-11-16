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

**Refreshes per eye** controls how many physical refreshes repeat one eye before switching. The
presenter shows an eye by presenting it again on every one of those refreshes, each for exactly
one refresh; it does not ask Windows to hold a single present for several refreshes, because
Windows was measured to hold a finished image one refresh too long about every ten seconds at
330 Hz. The first present of an eye identifies it and marks its boundary.
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
Cold timeouts and counter-domain changes restart acquisition within a six-second budget; the
per-present wait scales with the eye interval. Waiting for the first stereo packet has a separate
five-second limit. No eye commands are sent during this startup phase.

Only unsignalled clock acquisition may use `DwmFlush` to request feedback publication. Armed
presentation polls exact metadata with cancellable 200 µs waits; it never falls back to this
blocking call, which can consume more than a shutter interval under load. If a driver requires
a flush for every feedback update, the live wait times out and follows quiet recovery instead.

Once armed, each confirmed image may have been held for between the configured eye interval and
four times that interval. Its raw refresh-counter difference must be exact, and its QPC elapsed
time must agree with that difference within 750 µs or a quarter of one refresh period, whichever
is smaller; the timestamp Windows attaches to a refresh can be off by several hundred microseconds
under load without the image being on a different refresh. Windows can also publish a present
slightly before the refresh timestamp it carries, so an armed boundary up to 250 µs ahead of the
query clock counts as current. Its exact next present identity and current-frame freshness remain
mandatory. Cold acquisition accepts neither allowance.
The emitter has averaging disabled and receives explicit finite left/right commands: no command
is replayed for held refreshes, and eye phase advances once per confirmed image, independent of
the number or parity of refreshes. This tolerates an occasional held image without rebuilding
the desktop; it does not recover a stale frame. A late image does not move the schedule of the
images after it, so the sequence can run behind for several images until one image is shown for
fewer refreshes and the following boundary arrives early. The presenter tracks the refreshes an
armed sequence is behind: an early boundary is accepted only up to that amount, and the amount
expires after eight on-time images. Any other shorter interval is a fault. The shortened image
keeps its eye command, so its shutter window can overlap the start of the next scanout for that
one opening. Cold acquisition still requires 13 consecutive samples at the exact configured
minimum cadence.

Windows can report a present as displayed a moment before it fills in the refresh counters and
timestamp. The presenter polls up to eight more 200 µs samples for the completed statistics
before handing the sample to the same validation as any other.

A missing/invalid present, counter-domain discontinuity, excessive hold, or invalid timestamp
immediately stops eye commands. For a timing-only interruption, the presenter freezes image
submissions and retains its stereo pair while the emitter finishes its previous USB write.
A read-only settings query on the
same serial connection acts as a firmware processing barrier; the complete verified profile must
match before the finite-pulse drain can begin. It then reacquires 13 strict samples with no eye
commands and resumes a fresh left/right cycle on that pair. This preserves the desktop during a
short interruption. Recovery allows at most six attempts in ten seconds, with a shared 1.5-second
USB barrier deadline, a finite-pulse drain, and a six-second clock-acquisition budget per attempt.
A failed barrier, persistent timing fault, display/device change, or Windows security transition
returns to 2D. Recovery never infers eye parity or adjusts raw present IDs or refresh counters.

Each confirmed frame must also be recent enough for the configured delay, duration, spacing, and
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
physical refresh period multiplied by **Refreshes per eye**. Delay, duration, signal spacing, and the
protocol's close-pulse margin together must leave positive time for host presentation feedback in
that interval. These limits leave room for the emitter's IR pulse and scheduling.
The UI receives the native protocol limits and checks the interval when the selected monitor is
connected; Start 3D and the presenter share the native feedback-budget check before preparing a
display session or sending eye commands.
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

For development calibration, launch the optimized monitor host with
`SUNSHINE_LOCAL_MONITOR_TEST_PATTERN=1`, then use the usual Start/Stop controls. The physical
output shows a red **L** for the logical left image and a blue **R** for the logical right image,
with large white labels. Check one lens at a time: the left lens should show L and the right
lens R. Compare the top, middle and bottom of the panel for the other eye's colour or label.
The fixture keeps the production capture, stereo-packet ownership, timing checks and USB
commands active, including unsignalled acquisition and quiet recovery. It identifies eye
assignment and crosstalk; judge brightness with normal content. The environment flag is sampled
once per presenter initialization and only affects shutter monitors. Stop the session and
launch the usual host without the flag to return to normal content.

For per-slot host timing measurements, additionally set
`SUNSHINE_LOCAL_MONITOR_SLOT_TIMING=1` before launching the host. This diagnostic leaves the
profile, image order, presentation checks, and emitter commands unchanged. It collects bounded
in-memory measurements and writes separate logical-left and logical-right summaries only after
presentation stops. The summaries identify the configured physical shutter channel after eye
swapping; that channel name does not establish which image the panel actually emitted.

Compare reconstructed display-boundary-to-feedback and boundary-to-USB-`WriteFile` timing,
submission timing, write-call duration, and the next slot's submission. Inspect recovery segments
and held-refresh counts alongside the distributions. A host delay in one slot can postpone the
next slot's submission, so unchanged settings for the other eye do not establish unchanged
presentation. Host write timestamps measure submission to Windows, not USB arrival, IR decoding,
the panel's light output, or the glasses' transmission window. Visual calibration remains required.

Independent host-phase calibration is experimental. The optional environment variables
`SUNSHINE_LOCAL_MONITOR_L_EMIT_PHASE_US` and `SUNSHINE_LOCAL_MONITOR_R_EMIT_PHASE_US` select a
logical slot's host command phase after its confirmed display boundary. Both default to zero,
which preserves immediate emission. A positive phase and its 250 µs write-lateness allowance
must fit the existing host-feedback budget. These values do not replace the emitter's shared
delay or duration, and cannot move a command earlier than the received feedback.

For repeated calibration without relaunching the elevated host, set
`SUNSHINE_LOCAL_MONITOR_PHASE_CONFIG` to a UTF-8 JSON file containing exactly these two integer
fields:

```json
{"logical_L_us": 0, "logical_R_us": 0}
```

The file is limited to 2 KiB and is read once per presenter initialization. Stop 3D and wait for
the inactive state before editing it; the next Start reads the new values. Nonzero direct phase
environment variables cannot be combined with a phase file. Invalid files or values fail before
emitter configuration. No phase file is read during armed presentation.

An already-missed target, cancellation, excessive wake lateness, or expiry inside the transport
checks sends no scheduled command. Timing failures use the existing bounded quiet recovery;
transport failures retain their existing stop behavior. The final deadline is checked again
immediately before `WriteFile`. A positive phase uses timer waits followed by bounded processor
polling within the last 750 µs of the target; zero phases bypass this additional work. This avoids
relying solely on repeated short kernel waits, while preserving cancellation and the write cutoff.
It does not guarantee operating-system dispatch deadlines. Calibrate one logical slot at a time, then verify both physical
lenses, held-refresh counts and neighboring-slot submission timing. A passing host deadline does
not establish a clean optical window or unchanged behavior of the other lens.

The internal `SUNSHINE_LOCAL_MONITOR_PHASE_LOOKAHEAD` flag is disabled by default. Set it to
exactly `1` to test one-image lookahead with a positive logical-slot phase; `0` or an absent flag
preserves the serial presentation policy. Both zero phases keep immediate emission and the
ordinary submission path even when the flag is enabled. The flag is read once per presenter
initialization and malformed values fail before emitter configuration.

For a phased current image, lookahead admits and submits at most one next image before waiting
for the current command's phase. Admission retries only the zero-time readiness event, for at
most 500 µs after feedback receipt and only before the phase target minus 250 µs. There is one
GPU attempt, using the existing positive repeat interval and `DO_NOT_WAIT`. The next iteration
consumes its saved present identity and timestamps once; it does not reacquire a stereo packet,
wait for another admission, or render and submit that image again. A queued R retains its L's
tuple; a queued new L may acquire the next tuple only after the old R's copy was actually
displayed, and retains that new tuple through R and any cold recovery.

Immediately before the phased command, a non-consuming query must still identify the accepted
current image and the same reconstructed boundary. Freshness and the actual `WriteFile` deadline
remain mandatory. A known queued image appearing early, an unavailable admission, a nonaccepted
busy Present, or a missed phase stops commands and uses bounded quiet recovery. There is no
serial emission fallback for a failed lookahead attempt. An unexpected submitted/displayed
identity or ownership mismatch stops after the emitter drain and returns to 2D.

Recovery inhibits the pending ledger before serial work, stops new image submissions, verifies
the same-handle FIFO barrier and finite-pulse drain, then starts a new metadata epoch and 13
unsignalled left samples. One already-submitted image can appear during the drain; clearing a
CPU descriptor cannot cancel it. A readiness signal consumed without an accepted Present stays
owned through recovery and is used once for the first cold submission. Timing summaries report
negative write-end-to-next-submission intervals as valid signed overlaps. This experiment can
remove a host submission dependency; it does not establish unchanged panel phase, USB arrival,
or a clean optical window for either lens.

To separate glasses behaviour from host timing, the internal
`SUNSHINE_LOCAL_MONITOR_MUTE_SLOT` flag accepts exactly `L` or `R` and is absent by default. The
named logical slot keeps its image, presentation checks and timing gates but sends no USB eye
command; the other slot is unchanged. The log names the physical lens that receives no command
after eye swapping and reports skipped and sent command counts when presentation stops. A lens
that still shutters while its slot is muted is reacting to something other than its own command.
The flag is read once per presenter initialization, malformed values fail before emitter
configuration, and write-timing metadata for the muted slot is empty. It is a diagnostic only and
never a usable viewing mode.

Follow the machine's local development-launch instructions. Keep the installed and development
hosts mutually exclusive. A profile remains unqualified until these live observations are made;
mock API, offscreen GPU, and fake topology results cannot substitute for them.
