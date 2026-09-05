# Local AR-glasses presentation

Sunshine 3D can automatically own a local presentation session for an AR-glasses monitor connected as a
Windows display. This is a local D3D11 path: it does not use RTSP, NVENC, Moonlight, or Artemis.

Monitor decisions are stored by stable EDID/PnP model ID in the managed `ar_glass_devices` option.
Names specific to common AR-glasses families are approved automatically. A newly seen monitor that
cannot be identified safely remains inactive and appears in the web UI's **AR Displays** tab, where
the user can approve or reject it. Clicking the Windows notification for an unknown monitor opens
that tab. Both decisions persist across reconnects and connector changes.

## Mode contract

The active physical-monitor resolution selects the presentation mode:

| AR display mode | Private virtual source | Presentation |
|---|---|---|
| 1920x1080 | 1920x1080 | Normal 2D |
| 3840x1080 | 1920x1080 | Sunshine 3D full-SBS AI |

Other resolutions are rejected. Resolution alone never identifies a monitor as AR glasses.

## Lifecycle

The Windows display topology is polled every 250 ms. A mode or connection change stops the old
presenter immediately, then must remain stable for 750 ms before Sunshine 3D applies it. A supported
1920x1080/3840x1080 transition on the same physical output keeps the existing SudoVDA desktop and
its windows attached; Sunshine 3D updates its refresh in place, rebases the recovery journal to the new
physical width, re-isolates the source/sink row, and rebuilds only capture/conversion/presentation
resources. Position, neighboring-monitor, and HDR-state changes likewise reuse the virtual source
whenever the physical PnP target, GPU adapter, and supported mode remain compatible. If virtual HDR
cannot activate for an HDR sink, Sunshine 3D retains the same color-managed SDR fallback used at startup.
Moving the glasses to another GPU/adapter also forces a complete rebuild: the adapter LUID is part
of both the detected target contract and the presenter's expected output identity.

### RayNeo Air 4 Pro wear state

Wear detection is intentionally model-specific. Sunshine 3D enables it only when the selected
physical display is exactly `DISPLAY:TCL03D4` and the matching RayNeo Taurus HID device
(`VID_1BBB&PID_AF50`, vendor usage page `0xFF00`, usage `0x0001`) authenticates as board `0x3A`
(RayNeo Air 4 Pro). Other approved displays, other RayNeo boards, a missing HID interface, or an
ambiguous match retain the legacy connected-display behavior.

After board authentication, the host accepts only the exact 65-byte Windows HID sensor report
shape (`00 99 65`). The little-endian float at vendor-payload byte offset 44 is the wear proximity
observation. Repeated live worn/off-head A/B captures on the
authenticated Air 4 Pro established its polarity: finite values in the validated range at or above
`15000` propose **worn**, and finite nonnegative values at or below `10000` propose **off-head**.
The lower worn boundary includes the `22835`-to-`23188` band observed during a sustained live 3D
wear test, while retaining substantial separation from the observed `1545`-to-`3046` off-head band.
Intermediate, non-finite, or out-of-range proximity proposes **unknown**. The device tick is decoded
for diagnostics but is not used as the Windows transport-freshness gate; host receipt time of the
complete report supplies that authority. The observed `65535.0` worn saturation is hardware evidence
for this unit, not a claim that `65535` is a vendor-documented or universal RayNeo sentinel. The
hardware 1920x1080 2D / 3840x1080 3D mode does not change this wear signal.

A worn, off-head, or unknown proposal must remain continuous for at least 150 ms before it changes
the effective state. Each correctly framed report refreshes transport liveness, regardless of its
device tick. Ten seconds without any valid report also makes the state **unknown**; an explicit HID
read failure or physical unplug remains immediate. This liveness deadline bounds a silent stream
failure without treating the Air 4 Pro's observed multi-second quiet intervals as wear-state
changes. A silent subscription is closed and re-authenticated with bounded `1/2/5/10/30`-second
recovery backoff instead of repeatedly canceling and reissuing reads on the same possibly stale
handle. A healthy connection sends its board query and sensor-start command once, then keeps one
outstanding read and only consumes pushed HID reports; it never polls wear status. Unknown state
fails open: it cancels sensor-only suppression and permits the same local presentation behavior used
before wear sensing, rather than guessing that the glasses are worn or leaving a user with a
permanently blank display.

HID-interface arrival/removal and Windows suspend/resume callbacks interrupt every outstanding
command or read. Suspend/removal immediately closes the old handle; resume/arrival performs a fresh
exact-identity discovery, board authentication, and sensor subscription after a short device-settle
window. When the HID is absent, Windows interface notifications replace the old one-second SetupAPI
scan; a 30-second SetupAPI enumeration is retained only as a missed-notification safety net. If
notification registration itself fails, the one-second compatibility fallback remains. SetupAPI
enumeration is not a hardware/status query. Sunshine 3D cannot synthesize sensor data or reset the
USB bus when Windows restores only DisplayPort video and never enumerates `VID_1BBB&PID_AF50`; it
stays fail-open and recovers automatically if that interface later appears.

Confirmed off-head state stops and joins the local presenter, including capture and Host SBS 3D
conversion, but retains the session's SudoVDA virtual monitor, desktop and windows, physical-output
isolation, cursor isolation, and live GPU-ownership lease. Confirmed worn state rebuilds the
presenter resources against that same retained desktop. Wear state is a presenter pause reason
separate from the display-topology transition pause, so neither reason may resume presentation
while the other remains active. This gate adds no Host SBS inference, reuse, scene-cut, geometry,
or temporal rule; resumed resources continue to follow the canonical Host SBS contract.

Off-head is not a disconnect. A true Windows display unplug remains authoritative and follows the
ordinary full session teardown below: capture stops, the private virtual source is removed, and the
physical-display topology is restored. Remote virtual-display ownership likewise keeps its existing
priority over a local session whether that local presenter is running or paused off-head.

Only one presentation path owns an interactive virtual desktop at a time. A connecting or active
remote virtual-display stream takes priority without being terminated: Sunshine 3D synchronously stops
local AR before the remote display is created. Sunshine 3D admits only one remote stream. When it
disconnects, Sunshine 3D retains its app and virtual display for `session_resume_grace`; local AR waits for that reconnect
window to expire. After expiry, local AR waits for the remote SudoVDA identity to leave the Windows
topology and only then creates its own source. Resuming the remote client performs the inverse handoff. This arbitration
does not depend on transient `DISPLAYn` names and does not affect a remote session that captures a
physical display without creating a virtual desktop.
Both handoff directions retain ownership until the retired SudoVDA adapter/target identity has
disappeared from Windows topology; driver acknowledgement alone is not treated as completed removal.
The remote ownership reservation uses the configured `ping_timeout` connection window (with a
small scheduling grace), including values longer than one minute; it is not governed by a separate
fixed lease.

On connect Sunshine 3D:

1. Computes the deterministic source/sink row from a complete topology snapshot and durably records
   the exact expected physical rectangle before attaching SudoVDA. It rechecks the plan after the
   journal write and stops if the desktop changed concurrently.
2. Binds SudoVDA to the physical glasses' GPU and creates a private 1920x1080 display at the
   glasses refresh rate. Sunshine 3D retains the driver-returned adapter/target identity, so Windows GDI
   renumbering cannot redirect presentation to a remote client's same-resolution virtual display.
3. Builds one deterministic row after the rightmost interactive monitor: interactive desktop,
   private virtual source, then the physical glasses. The source and sink share a full vertical edge;
   no absolute coordinates, empty gap, or one-pixel corner contact are used. While this topology is
   active Sunshine 3D asks Windows to confine the cursor through the virtual source's right edge. If
   another application owns cursor confinement, Sunshine 3D yields to it and the presenter window uses
   an event-driven edge clamp as a fallback. Physical-sink input is never reinjected synthetically.
   If Windows cannot produce a unique non-primary, non-cloned row, local presentation fails closed
   before attaching the virtual display.
4. After the first isolated row is durably committed, matches the private source to the physical
   output's active color mode. HDR uses the same proven delayed off-then-on Advanced Color workaround
   as Sunshine 3D's remote virtual-display sessions, followed by another journaled isolation pass.
5. Captures the virtual display on the configured GPU.
6. Re-queries the selected monitor's exact device-instance path after topology changes, matches its actual
   `IDXGIOutput`, and uses the coordinates Windows applied rather than the requested position.
7. Presents a non-activating, borderless, topmost swapchain restricted to that physical output.
8. Uses passthrough in 1920x1080 or the production matched-frame depth and warp in 3840x1080.

Disconnect, an unsupported resolution, a graphics-adapter change, or Sunshine 3D shutdown
stops capture before removing the private virtual display. A supported 2D/SBS resolution change and
ordinary swapchain/capture reinitialization retain it. Unexpected presenter failures retry after a
delay; repeated setup failures use bounded exponential backoff while the same stable glasses mode
remains active.

On clean shutdown, disconnect, or an incompatible transition, Sunshine 3D removes the private virtual
source and restores the physical glasses to the desktop position they occupied before the session.
The physical output must remain active (and therefore visible in Windows Display Settings) because
disabling its display path would also stop DP scanout.

Before attaching a virtual source or moving a physical output, Sunshine 3D atomically updates a small
recovery journal beside the active `sunshine.conf`. The journal keeps one entry per exact PnP target.
A pending transaction owns only its exact requested rectangle; confirmed prior Sunshine 3D rectangles
remain owned independently. Any other position is preserved as a user/Windows change rather than
being guessed to belong to Sunshine 3D. Immediately before applying a complete `SetDisplayConfig`
snapshot Sunshine 3D re-queries every active path and aborts if anything changed after validation.

On teardown, Sunshine 3D classifies the physical rectangle before removing SudoVDA, then restores that
decision after the virtual output has fully retired. This prevents removal-time Windows normalization
from disguising a Sunshine 3D-owned row as a user move. If SudoVDA attachment produces a rectangle other
than the pre-journaled expected one, Sunshine 3D cannot prove ownership; it preserves the unexpected
position and aborts the session. If Sunshine 3D, the GPU driver, or Windows exits without normal teardown,
the next launch restores connected targets only from exact pending or confirmed evidence. A
disconnected target's entry remains pending until it reconnects, but it does not block presentation
on another approved pair of glasses. Starting a session waits only for recovery of that same PnP
target. The former single-target journal format is migrated atomically without discarding its pending
recovery.

If Sunshine 3D enables HDR on the physical glasses, it leaves that per-display Windows preference in
place. Reverting it during an internal 2D/SBS session rebuild would trigger another topology change
and can make the display oscillate between modes. A policy restriction is logged and stays SDR.
Displays that advertise HDR but do not actually enter HDR (for example, until an on-device HDR10
setting is enabled) also continue in SDR instead of entering a create/remove retry loop.

While a captured source has pending pipeline construction, depth completion, conversion, or a busy final presentation,
the local owner limits capture/pacing waits to 5 ms and yields the capture-device lock for 1 ms
after a timeout. It then retries the retained work even if the desktop never changes again.
Completion checks remain nonblocking on that same owner thread; retrying a busy Present does not
reconvert its pixels. Once the work is presented, ordinary idle capture resumes. Remote capture
keeps its existing idle timeout. These are scheduling bounds, not a GPU completion-time guarantee.

The local path avoids an RGB-to-YUV encode/decode round trip. SDR uses a BGRA8 Rec.709 swapchain.
When both outputs have stably entered HDR, Sunshine 3D captures linear FP16 scRGB and presents it through
an FP16 scRGB swapchain; DWM performs the final device-specific HDR10/PQ conversion. Presentation
does not start while either source or target is still changing color mode.

## Audio

Local AR presentation is currently video-only. Windows continues to route sound through its default
audio endpoint; Sunshine 3D does not select the glasses' DisplayPort/HDMI audio endpoint or restore a
previous endpoint when presentation stops.
