# Local AR-glasses presentation

Apollo can automatically own a local presentation session for an AR-glasses monitor connected as a
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
| 3840x1080 | 1920x1080 | Apollo full-SBS AI |

Other resolutions are rejected. Resolution alone never identifies a monitor as AR glasses.

## Lifecycle

The Windows display topology is polled every 250 ms. A mode or connection change stops the old
presenter immediately, then must remain stable for 750 ms before Apollo applies it. A supported
1920x1080/3840x1080 transition on the same physical output keeps the existing SudoVDA desktop and
its windows attached; Apollo updates its refresh in place, rebases the recovery journal to the new
physical width, re-isolates the source/sink row, and rebuilds only capture/conversion/presentation
resources. Position, neighboring-monitor, and HDR-state changes likewise reuse the virtual source
whenever the physical PnP target, GPU adapter, and supported mode remain compatible. If virtual HDR
cannot activate for an HDR sink, Apollo retains the same color-managed SDR fallback used at startup.
Moving the glasses to another GPU/adapter also forces a complete rebuild: the adapter LUID is part
of both the detected target contract and the presenter's expected output identity.

Only one presentation path owns an interactive virtual desktop at a time. A connecting or active
remote virtual-display stream takes priority without being terminated: Apollo synchronously stops
local AR before the remote display is created. Apollo admits only one remote stream. When it
disconnects, Apollo retains its app and virtual display for `session_resume_grace`; local AR waits for that reconnect
window to expire. After expiry, local AR waits for the remote SudoVDA identity to leave the Windows
topology and only then creates its own source. Resuming the remote client performs the inverse handoff. This arbitration
does not depend on transient `DISPLAYn` names and does not affect a remote session that captures a
physical display without creating a virtual desktop.
Both handoff directions retain ownership until the retired SudoVDA adapter/target identity has
disappeared from Windows topology; driver acknowledgement alone is not treated as completed removal.
The remote ownership reservation uses the configured `ping_timeout` connection window (with a
small scheduling grace), including values longer than one minute; it is not governed by a separate
fixed lease.

On connect Apollo:

1. Computes the deterministic source/sink row from a complete topology snapshot and durably records
   the exact expected physical rectangle before attaching SudoVDA. It rechecks the plan after the
   journal write and stops if the desktop changed concurrently.
2. Binds SudoVDA to the physical glasses' GPU and creates a private 1920x1080 display at the
   glasses refresh rate. Apollo retains the driver-returned adapter/target identity, so Windows GDI
   renumbering cannot redirect presentation to a remote client's same-resolution virtual display.
3. Builds one deterministic row after the rightmost interactive monitor: interactive desktop,
   private virtual source, then the physical glasses. The source and sink share a full vertical edge;
   no absolute coordinates, empty gap, or one-pixel corner contact are used. While this topology is
   active Apollo asks Windows to confine the cursor through the virtual source's right edge. If
   another application owns cursor confinement, Apollo yields to it and the presenter window uses
   an event-driven edge clamp as a fallback. Physical-sink input is never reinjected synthetically.
   If Windows cannot produce a unique non-primary, non-cloned row, local presentation fails closed
   before attaching the virtual display.
4. After the first isolated row is durably committed, matches the private source to the physical
   output's active color mode. HDR uses the same proven delayed off-then-on Advanced Color workaround
   as Apollo's remote virtual-display sessions, followed by another journaled isolation pass.
5. Captures the virtual display on the configured GPU.
6. Re-queries the selected monitor's exact device-instance path after topology changes, matches its actual
   `IDXGIOutput`, and uses the coordinates Windows applied rather than the requested position.
7. Presents a non-activating, borderless, topmost swapchain restricted to that physical output.
8. Uses passthrough in 1920x1080 or the production matched-frame depth and warp in 3840x1080.

Disconnect, an unsupported resolution, a graphics-adapter change, or Apollo shutdown
stops capture before removing the private virtual display. A supported 2D/SBS resolution change and
ordinary swapchain/capture reinitialization retain it. Unexpected presenter failures retry after a
delay; repeated setup failures use bounded exponential backoff while the same stable glasses mode
remains active.

On clean shutdown, disconnect, or an incompatible transition, Apollo removes the private virtual
source and restores the physical glasses to the desktop position they occupied before the session.
The physical output must remain active (and therefore visible in Windows Display Settings) because
disabling its display path would also stop DP scanout.

Before attaching a virtual source or moving a physical output, Apollo atomically updates a small
recovery journal beside the active `sunshine.conf`. The journal keeps one entry per exact PnP target.
A pending transaction owns only its exact requested rectangle; confirmed prior Apollo rectangles
remain owned independently. Any other position is preserved as a user/Windows change rather than
being guessed to belong to Apollo. Immediately before applying a complete `SetDisplayConfig`
snapshot Apollo re-queries every active path and aborts if anything changed after validation.

On teardown, Apollo classifies the physical rectangle before removing SudoVDA, then restores that
decision after the virtual output has fully retired. This prevents removal-time Windows normalization
from disguising an Apollo-owned row as a user move. If SudoVDA attachment produces a rectangle other
than the pre-journaled expected one, Apollo cannot prove ownership; it preserves the unexpected
position and aborts the session. If Apollo, the GPU driver, or Windows exits without normal teardown,
the next launch restores connected targets only from exact pending or confirmed evidence. A
disconnected target's entry remains pending until it reconnects, but it does not block presentation
on another approved pair of glasses. Starting a session waits only for recovery of that same PnP
target. The former single-target journal format is migrated atomically without discarding its pending
recovery.

If Apollo enables HDR on the physical glasses, it leaves that per-display Windows preference in
place. Reverting it during an internal 2D/SBS session rebuild would trigger another topology change
and can make the display oscillate between modes. A policy restriction is logged and stays SDR.
Displays that advertise HDR but do not actually enter HDR (for example, until an on-device HDR10
setting is enabled) also continue in SDR instead of entering a create/remove retry loop.

The local path avoids an RGB-to-YUV encode/decode round trip. SDR uses a BGRA8 Rec.709 swapchain.
When both outputs have stably entered HDR, Apollo captures linear FP16 scRGB and presents it through
an FP16 scRGB swapchain; DWM performs the final device-specific HDR10/PQ conversion. Presentation
does not start while either source or target is still changing color mode.

## Audio

Local AR presentation is currently video-only. Windows continues to route sound through its default
audio endpoint; Apollo does not select the glasses' DisplayPort/HDMI audio endpoint or restore a
previous endpoint when presentation stops.
