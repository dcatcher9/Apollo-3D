# Virtual desktop interaction

Sunshine 3D's virtual desktop is an additional monitor in the active Windows desktop. It shares
the signed-in user's applications, keyboard focus, and cursor with the physical monitors. It is
not a separate Windows login or an isolated Windows session.

## Opening applications normally

Use Windows Start, Search, the taskbar, desktop shortcuts, Explorer, or Run from the streamed
virtual monitor. There is no custom application launcher. The host automatically starts a hidden,
standard-user window router for its virtual display. Both the mouse/keyboard connected to the PC
and client input can invoke native Windows launches. Standard actionable application controls
and Enter from an application/terminal on the virtual monitor can also identify an invocation.
The source monitor decides placement: a qualifying invocation on the virtual display admits the
next eligible foreground application window there; a gesture on a physical monitor does not.
Client input carries the host session's tag so it can be recognized alongside ordinary PC input.
Other injected input is ignored.

A launch that activates an existing app may move that selected window from a physical monitor.
Other document windows in the same process are not moved. The router does not change the primary
monitor or rearrange existing windows when streaming starts. It does not run applications itself;
Windows continues to handle shortcuts, arguments, file associations, and application reuse.

Activation on a Windows shell surface or an eligible application control starts a short, one-use
routing opportunity. Pointer clicks must hit a recent, asynchronously inspected actionable item. Desktop and
Explorer icons also honor Windows' single/double-click setting. Keyboard Enter and native Windows
shortcuts can invoke the same routing. Empty-space clicks do not authorize a move.
Unrecognized injected input, pointer input outside the target, a secure-desktop switch,
or a missing/changed target cancels it. Pointer motion does not create a launch request. Background
window creation alone does not authorize placement. After the selected foreground root has been
handled, another launch gesture is required to move another root.

Windows does not expose a universal causal link between a shell click and an application window.
Routing therefore has compatibility limits:

- A launch must present an eligible foreground window within five seconds. Background-only apps,
  delayed launches, and apps that keep the same foreground window without an activation event
  can remain where Windows places them. An unrelated application stealing focus during that
  brief launch interval can be mistaken for the launch result.
- Mouse routing needs fresh shell-item evidence. If a shell provider is unavailable or too slow,
  or an item is clicked before its first hover query completes, that launch is left to Windows.
- The PC's physical mouse/keyboard and the XR mouse/ray path supply routing provenance. Direct
  native pen/touch injection is not identified by the mouse/keyboard tag.
- Elevated applications, protected surfaces, and apps that override their own placement may
  reject the standard-user helper's move. Games with exclusive display selection may need their
  built-in monitor setting; ordinary windowed/borderless windows use Windows placement.
- Custom application controls without standard actionable UI Automation metadata, background
  launches, and other activation paths without an observable invocation cannot be attributed
  reliably. Owned dialogs can follow a specifically adopted root while interaction remains on
  the virtual monitor. Moving input to a physical monitor clears that tracking. Other document
  windows in the same application process are not adopted.

The helper stops on disconnect and starts again for the next active stream, including a warm
reconnect to a retained monitor. Its old routing intent and dialog tracking are discarded.
Replacing/removing the monitor or stopping the host also ends its work. It never closes user
applications. Windows may move windows when their monitor is removed, according to its normal
display-removal behavior.

This feature needs live Galaxy XR verification with real Start/shortcut launches, existing app
reuse, reconnects, and a simultaneous local user. Isolated tests do not establish application
compatibility on the headset.

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
