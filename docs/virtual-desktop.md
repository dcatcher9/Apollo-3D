# Virtual desktop interaction

Sunshine 3D's virtual desktop is an additional monitor in the active Windows desktop. It shares
the signed-in user's applications, keyboard focus, and cursor with the physical monitors. It is
not a separate Windows login or an isolated Windows session.

## Opening applications normally

While a virtual-display stream is active, Sunshine 3D temporarily makes that virtual monitor the
Windows primary display. Use Windows Start, Search, the taskbar, desktop shortcuts, Explorer, or
Run normally. Both the mouse/keyboard connected directly to the PC and client input use the same
Windows launch behavior. There is no custom launcher or background window-routing helper.

Applications that default to the primary monitor now open on the virtual display. This changes
the default for the whole Windows desktop: a launch initiated from a physical monitor may also
open there. It is not a per-launch or per-client placement rule. Sunshine 3D supports one active
remote session per host.

Physical displays remain enabled. The host preserves their relative arrangement and does not
deliberately move existing application windows. Changing the primary display changes the desktop
coordinate origin, however, and Windows or individual applications may react to the topology
change. Existing windows are not guaranteed to remain visually unchanged.

Primary-monitor placement is a default, not a guarantee for every application:

- Applications can restore a saved monitor or use explicit window coordinates.
- A launch that reuses an existing application window can leave that window on its current
  monitor. Sunshine does not move it automatically.
- Games with a built-in monitor selector may continue to use their selected display.
- Owned dialogs and other application windows may follow their parent window.

The host records the original primary display and arrangement before changing them. On disconnect
it restores the previous primary display, including during the reconnect grace period
while the application and virtual monitor remain available. Reconnecting makes the virtual
display primary again before capture resumes. Full session teardown restores the original primary
display before the virtual monitor is removed.

Recovery information is written to disk before the primary-display change. The host checks for
unfinished restoration during startup; a crash does not require the previous process to remain
alive for recovery. A failed restore retains that information for a retry and prevents the host
from removing the virtual monitor before the pending restoration is reconciled. Disconnected or
changed physical displays can prevent exact restoration until the display arrangement is available
again. While the disconnected session remains in its reconnect grace period, the host retries a
failed restore once per second without extending that deadline. A new pending or active session
cancels those retries. Windows can also move windows when their monitor is removed.

The change needs live Galaxy XR verification with real Start/shortcut launches, applications that
remember a monitor, reconnects, and display restoration. Automated lifecycle and geometry tests
do not establish every application's placement behavior or replace a real multi-monitor test.

## Keeping the remote cursor visible

By default, remote cursor movement is bounded to the streamed virtual display so the pointer stops
at its edges. This applies to relative remote mouse movement as well as absolute pointer placement.
If the physical mouse moves the shared cursor onto another monitor, the next remote pointer
movement brings it back to the streamed display.

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
