# Virtual desktop interaction

Sunshine 3D's virtual desktop is an additional monitor in the active Windows desktop. It shares
the signed-in user's applications, keyboard focus, and cursor with the physical monitors. It is
not a separate Windows login or an isolated Windows session.

## Opening applications normally

When **Use virtual display only while streaming** is disabled in Moonlight 3D, Sunshine 3D
temporarily makes the virtual monitor the Windows primary display. Use Windows Start, Search, the
taskbar, desktop shortcuts, Explorer, or Run normally. Both the mouse/keyboard connected directly
to the PC and client input use the same Windows launch behavior. There is no custom launcher or
background window-routing helper.

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
streaming** in Moonlight 3D's **Global Settings → Streaming defaults**. Supporting clients enable
it by default.
The client sends its current choice for every launch and resume. The session owner, or another
client with launch permission, can apply a changed choice by reconnecting. A different view-only
client inherits the retained session's choice.

With this option on, Sunshine 3D temporarily disables ordinary physical displays during a
virtual-display stream and makes the virtual monitor primary. An approved AR-glasses output remains
active because disabling its Windows display path would stop its scanout. This affects the entire
desktop, including applications that were already open on disabled monitors: Windows or the
application may move or minimize those windows. It does not create an isolated desktop or guarantee
how an application positions its windows.

Windows does not provide a general active-but-hidden state for a normal monitor: an active display
path participates in the desktop. A black overlay or a powered-off panel would still leave that
desktop region available to windows and the cursor. Sunshine therefore disables ordinary paths and
keeps only approved AR outputs active when their glasses require a live scanout.

Sunshine restores the previous displays on disconnect, including during the reconnect grace period.
Applications may move again when physical displays return, and their exact positions or minimized
states are not guaranteed to return. Streams using a physical desktop are unaffected.

Exclusive mode also applies a Windows cursor boundary around the virtual display. Both the mouse
connected to the PC and client input are restricted by it, keeping normal pointer movement off the
active AR display. This boundary applies throughout exclusive mode, including before glasses
connect, and takes precedence over the client's **Keep cursor on virtual display** setting. Windows
has one shared cursor and no exclusive owner for its `ClipCursor` boundary. The host repairs the
boundary after virtual-display resizing, display hotplug, and foreground-window changes and checks
it periodically. An application that continuously replaces the shared boundary can briefly win
until the next repair. The host releases its ownership before restoring the desktop on disconnect
or teardown.

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

The host restores unfinished display transactions early during normal startup, before GPU and
platform initialization. The existing packaged Windows service restarts the host after an
unexpected exit, with a three-second retry interval while the service remains running. Recovery
then uses the saved display record; no additional recovery process or launcher is required.

A standalone development host has no automatic restart. After a hard crash, physical displays may
remain disabled until that host is relaunched with the **same configuration file**. Starting an
installed copy with a different configuration does not recover the development host's record.
Normal disconnect and graceful shutdown restore displays without requiring a restart.

A shared ownership lock prevents another host from restoring a live transaction. While a
disconnected session remains in its reconnect grace period, the host retries a failed restore once
per second without extending that deadline. A new accepted launch or active session cancels those
retries. Changed physical display arrangements can prevent exact restoration until the saved
arrangement is available again. Windows can also move windows when their monitor is removed.

The ordinary virtual-display-only stream and disconnect path has been verified on Galaxy XR.
Recovery after a hard host crash still requires a live multi-monitor test; automated lifecycle and
geometry tests cannot establish that Windows and every display driver recover identically.

## Keeping the remote cursor visible

By default, remote cursor movement is bounded to the streamed virtual display so the pointer stops
at its edges. This applies to relative remote mouse movement as well as absolute pointer placement.
When other monitors are active, if the physical mouse moves the shared cursor onto one of them,
the next remote pointer movement brings it back to the streamed display.

With **Use virtual display only while streaming** disabled, this setting constrains client input;
someone at the PC can still move the physical mouse normally. With exclusive mode enabled, its
Windows cursor boundary constrains the shared cursor regardless of this client setting. Existing
application cursor restrictions are preserved where they overlap the virtual display. Cleanup
restores the previous restriction only while Sunshine still owns the applied boundary, without
overwriting another application's replacement.

While confinement is enabled, every relative movement is converted to a bounded absolute position,
including movement with a hidden game cursor. An application's own cursor lock may change before
Windows processes queued input, so it cannot provide the remote confinement guarantee. If an
application needs unmodified raw-relative input or Windows acceleration, turn off **Keep cursor
on virtual display** in Moonlight 3D's **Global Settings → Audio & input**. During a stream, use
the same control in **Session settings**, then **Apply & reconnect**. This is a client choice;
there is no host configuration toggle. Turning it off preserves raw-relative client input, but
exclusive mode's Windows cursor boundary still excludes the AR display.

## Client protocol

An authenticated Windows host `/serverinfo` advertises `VirtualDisplayOnlySupported=1`. Supporting
clients send the exact query value `virtualDisplayOnly=1` (enabled) or `virtualDisplayOnly=0`
(disabled) on both `/launch` and `/resume`. The host defaults to disabled when an older client omits
the parameter. On resume, the retained session owner or a client with launch permission may change
the choice. A different client with view-only permission can resume the stream but inherits the
retained choice. A retryable reconfiguration failure keeps the last accepted choice; an
unrecoverable topology failure terminates the retained session.

Authenticated host `/serverinfo` also advertises `CursorConfinementSupported=1`. A client must
check this specific capability before presenting confinement as supported; the host-session-token
or Host SBS capabilities alone do not imply it.

Supporting clients send the exact query value `confineCursor=1` (enabled) or `confineCursor=0`
(disabled) on both `/launch` and `/resume`. The host defaults to enabled when an older client
omits the flag. Other values reject the request before display or application side effects.
Every accepted resume supplies the new stream's choice rather than inheriting its predecessor's
setting. Confinement applies only to a host-created virtual monitor; physical desktop streams
retain ordinary input behavior.

Clients apply a changed choice through reconnect. Moonlight 3D uses its existing **Apply &
reconnect** action. There is no custom control packet or live setter for this option.

The cursor is shared Windows state. These controls do not provide independent pointers or
simultaneous independent keyboard focus for local and remote users.
