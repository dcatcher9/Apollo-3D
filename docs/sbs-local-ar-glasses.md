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
presenter immediately, then must remain stable for 750 ms before Apollo creates the replacement
session. A position-only change moves the existing presenter without recreating the virtual display.
Changing the primary monitor or a neighboring monitor's layout recreates the session so both the
virtual desktop and physical sink remain attached to the current topology.

Only one presentation path owns an interactive virtual desktop at a time. A connecting or active
remote virtual-display stream takes priority without being terminated: Apollo synchronously stops
local AR before the remote display is created. When the last remote stream disconnects, its app is
normally paused; if approved glasses are connected, local AR terminates that inactive session,
waits for its stable SudoVDA adapter/target identity to leave the Windows topology, and only then
creates the local source. Resuming the remote client performs the inverse handoff. This arbitration
does not depend on transient `DISPLAYn` names and does not affect remote sessions that capture a
physical display without creating a virtual desktop.

On connect Apollo:

1. Binds SudoVDA to the physical glasses' GPU and creates a private 1920x1080 display at the
   glasses refresh rate. Apollo retains the driver-returned adapter/target identity, so Windows GDI
   renumbering cannot redirect presentation to a remote client's same-resolution virtual display.
2. Matches the private source to the physical output's active color mode. HDR uses the same proven
   delayed off-then-on Advanced Color workaround as Apollo's remote virtual-display sessions.
3. Places the virtual source beside the interactive desktop, identifies the current Windows primary
   monitor, and positions the physical glasses from that monitor's live bottom-right corner. A
   one-pixel-wide boundary segment keeps the topology valid. If the pointer crosses that segment,
   the presenter redirects it to the corresponding location on the private virtual source; a
   zero-length point contact is disconnected and Windows otherwise normalizes it back to a full
   shared edge. No absolute desktop position is assumed.
4. Captures the virtual display on the configured GPU.
5. Re-queries the selected monitor's exact device-instance path after topology changes, matches its actual
   `IDXGIOutput`, and uses the coordinates Windows applied rather than the requested position.
6. Presents an input-transparent, borderless, topmost swapchain restricted to that physical output.
7. Uses passthrough in 1920x1080 or the production matched-frame depth and warp in 3840x1080.

Disconnect, resolution change, swapchain loss, or Apollo shutdown stops capture before removing the
private virtual display. Unexpected presenter failures retry after two seconds while the same stable
glasses mode remains active.

On clean shutdown or a mode transition, Apollo removes the private virtual source and restores the
physical glasses to the desktop position they occupied before the session. The physical output must
remain active (and therefore visible in Windows Display Settings) because disabling its display path
would also stop DP scanout.

If Apollo enables HDR on the physical glasses, it leaves that per-display Windows preference in
place. Reverting it during an internal 2D/SBS session rebuild would trigger another topology change
and can make the display oscillate between modes. A policy restriction is logged and stays SDR.

The local path avoids an RGB-to-YUV encode/decode round trip. SDR uses a BGRA8 Rec.709 swapchain.
When both outputs have stably entered HDR, Apollo captures linear FP16 scRGB and presents it through
an FP16 scRGB swapchain; DWM performs the final device-specific HDR10/PQ conversion. Presentation
does not start while either source or target is still changing color mode.

## Audio

Local AR presentation is currently video-only. Windows continues to route sound through its default
audio endpoint; Apollo does not select the glasses' DisplayPort/HDMI audio endpoint or restore a
previous endpoint when presentation stops.
