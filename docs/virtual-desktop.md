# Virtual desktop interaction

Sunshine 3D's virtual desktop is an additional monitor in the active Windows desktop. It shares
the signed-in user's applications, keyboard focus, and cursor with the physical monitors. It is
not a separate Windows login or an isolated Windows session.

## Opening applications normally

By default, while a virtual-display stream is active, Sunshine 3D temporarily makes that monitor the
Windows primary display. Use Windows Start, Search, the taskbar, desktop shortcuts, Explorer, or
Run normally. Both the mouse/keyboard connected directly to the PC and client input use the same
Windows launch behavior. There is no custom launcher or background window-routing helper.

Applications that default to the primary monitor now open on the virtual display. This changes
the default for the whole Windows desktop: a launch initiated from a physical monitor may also
open there. It is not a per-launch or per-client placement rule. Sunshine 3D supports one active
remote session per host.

In this default mode, physical displays remain enabled. The host preserves their relative
arrangement and does not deliberately move existing application windows. Changing the primary
display changes the desktop coordinate origin, however, and Windows or individual applications may react to the topology
change. Existing windows are not guaranteed to remain visually unchanged.

Primary-monitor placement is a default, not a guarantee for every application:

- Applications can restore a saved monitor or use explicit window coordinates.
- A launch that reuses an existing application window can leave that window on its current
  monitor. Sunshine does not move it automatically.
- Games with a built-in monitor selector may continue to use their selected display.
- Owned dialogs and other application windows may follow their parent window.

## Using only the virtual display

For applications that keep opening on a physical monitor, enable **Use virtual display only while
streaming** in the host Web UI under **Essentials → Virtual display**, or set
`virtual_display_only = on`. The default is `off`, so existing configurations keep physical displays
enabled.

The host captures this choice when a new application session starts. A reconnect keeps the retained
session's choice even if the host setting changes; stop that session and launch a new one to switch
between display modes.

With this option on, Sunshine 3D temporarily disables every other active display during a
virtual-display stream. Physical screens go blank and the virtual monitor becomes the only active
Windows display. This affects the entire desktop, including applications that were already open on
physical monitors: Windows or the application may move or minimize those windows. It does not
create an isolated desktop or guarantee how an application positions its windows.

Sunshine restores the previous displays on disconnect, including during the reconnect grace period.
Applications may move again when physical displays return, and their exact positions or minimized
states are not guaranteed to return. Streams using a physical desktop are unaffected. The client
cursor-confinement setting remains a separate choice.

## Display restoration

The host records the original display setup before changing it. On disconnect it restores the
previous primary and, when virtual-display-only mode was used, the other active displays. This also
applies during the reconnect grace period while the application and virtual monitor remain
available. Reconnecting applies the selected mode again before capture resumes. Full session
teardown restores the display setup before the virtual monitor is removed.

Recovery information is written to disk before the display change. Virtual-display-only mode also
records the original active displays, their modes, and their advanced-color settings. A failed
restore keeps that record and prevents the host from removing the virtual monitor before the
pending restoration is reconciled. If an original physical monitor is unplugged, recovery lights
the available original monitors and retains the record until the missing monitor returns and the
remaining settings can be restored. The virtual monitor stays active during that partial recovery.

Before disabling physical displays, the host starts a hidden, independent recovery process and
waits for its readiness handshake. If it cannot establish that process, the virtual-display-only
launch fails before physical displays are disabled. The recovery process waits for the actual host
process to exit; it never restores displays while that host is alive. After a crash it can restore
the saved physical displays without restarting the host, retrying every five seconds when recovery
is incomplete. A shared ownership lock prevents it from changing a restarted host's live display
setup. Each recovery process writes a separate log beside the host configuration file, named
`<config-file>.display-recovery-<PID>.log`, for example `sunshine.conf.display-recovery-12345.log`.

If a terminal, service wrapper, or development runner keeps the recovery process inside a Windows
job, the process refuses readiness and records the reason in that log. Virtual-display-only mode
then fails before disabling physical displays; the host must be started in an environment that
allows its recovery process to run independently.

The host also checks for unfinished restoration during startup. While a disconnected session
remains in its reconnect grace period, it retries a failed restore once per second without
extending that deadline. A new accepted launch or active session cancels those retries. Changed
physical display arrangements can prevent exact restoration until the saved arrangement is
available again. Windows can also move windows when their monitor is removed.

Virtual-display-only mode still needs live Galaxy XR verification with applications that remember
a monitor, existing windows, reconnects, and physical display recovery after a host crash. Automated
lifecycle and geometry tests do not establish every application's placement behavior or replace a
real multi-monitor test.

## Keeping the remote cursor visible

By default, remote cursor movement is bounded to the streamed virtual display so the pointer stops
at its edges. This applies to relative remote mouse movement as well as absolute pointer placement.
When other monitors are active, if the physical mouse moves the shared cursor onto one of them,
the next remote pointer movement brings it back to the streamed display.

Sunshine does not apply a global Windows cursor lock. Someone at the PC can still move the
physical mouse normally. The remote bounds apply only while input belongs to the virtual-display
stream; disconnecting or switching away ends that input scope.

For desktop-style relative movement, bounding the resulting pointer position changes the native
relative-input path and may bypass Windows mouse acceleration. A foreground game that hides its
cursor and confines it within the virtual display retains its native relative movement. If another
application needs unmodified raw-relative input or Windows acceleration, turn off **Keep cursor
on virtual display** in Moonlight 3D's **Global Settings → Audio & input**. During a stream, use
the same control in **Session settings**, then **Apply & reconnect**. This is a client choice;
there is no host configuration toggle.

## Client protocol

Authenticated host `/serverinfo` advertises `CursorConfinementSupported=1`. A client must check
this specific capability before presenting confinement as supported; the host-session-token or
Host SBS capabilities alone do not imply it.

Supporting clients send the exact query value `confineCursor=1` (enabled) or `confineCursor=0`
(disabled) on both `/launch` and `/resume`. The host defaults to enabled when an older client
omits the flag. Other values reject the request before display or application side effects.
Every accepted resume supplies the new stream's choice rather than inheriting its predecessor's
setting. Confinement applies only to a host-created virtual monitor; physical desktop streams
retain ordinary input behavior.

Clients apply a changed choice through reconnect. Moonlight 3D uses its existing **Apply &
reconnect** action. There is no custom control packet or live setter for this option.

The cursor is shared Windows state. These controls prevent remote movement from losing the
pointer on another monitor; they do not provide independent pointers or simultaneous independent
keyboard focus for local and remote users.
