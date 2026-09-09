# Virtual desktop interaction

Sunshine 3D's virtual desktop is an additional monitor in the active Windows desktop. It shares
the signed-in user's applications, keyboard focus, and cursor with the physical monitors. It is
not a separate Windows login or an isolated Windows session.

The app launcher is enabled by default under the host's **Configuration → Essentials → Virtual
display**. Cursor confinement is enabled by default in Moonlight 3D's client settings. This
implementation still needs live Galaxy XR verification,
including application compatibility and reconnect behavior.

## Launching applications

While a virtual-display stream is active, a launcher appears on that monitor. Browse to an `.exe`,
enter optional arguments, and launch it from there. The helper tracks the process it starts and
attributable child processes, placing their new windows on the streamed monitor. New dialogs and
additional windows can follow the same placement rule when their origin can be established.

Existing windows are excluded from automatic placement. In particular, launching another copy of
an app never authorizes moving its existing main-monitor window. Changing focus alone does not
move a window. The launcher does not change the primary monitor or the physical monitors' display
configuration.

Use the launcher for this behavior. Starting an application through the Windows Start menu, a
desktop shortcut, or an existing application does not reliably identify it as a virtual-display
launch.

Some applications require additional settings:

- **Apps with a single running instance:** A new launch may activate an existing window. Use the
  application's separate-window or separate-instance argument when it supports one. The helper
  leaves that existing window in place and reports when it cannot find a new attributable window.
- **Launchers, brokers, and detached processes:** A launch may be handed to an existing service or
  another process whose origin cannot be established. Those windows are left in place; the
  launcher reports the compatibility limitation.
- **Games:** Windowed and borderless modes can accept normal window placement. An
  exclusive-fullscreen game may select its own output; use the game's monitor setting. Some games
  also restore their saved window placement after launch.

The launcher and its application tracking survive a reconnect while the same virtual monitor is
retained. Closing the launcher, replacing the monitor, or ending the retained session stops its
window management and keeps launched applications running. A new helper does not take ownership
of an earlier helper's applications. After final virtual-display removal, Windows may move windows
from the removed monitor according to its normal display-removal behavior.

Set [`virtual_display_launcher = off`](configuration.md#virtual_display_launcher) to disable the
helper. This setting does not close applications already launched through it.

## Keeping the remote cursor visible

Remote cursor movement is bounded to the streamed virtual display so the pointer stops at its
edges. This applies to relative remote mouse movement as well as absolute pointer placement.
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
