# Windows application examples

Applications appear in Moonlight 3D after they are configured on Sunshine 3D's **Applications**
page. This host supports Windows only. Leave fields not shown below empty unless the application
needs them.

> [!TIP]
> Use the built-in **Virtual Display** entry for the complete XR desktop workflow. It lets the
> client negotiate landscape or portrait resolution, refresh rate, HDR, and Raw SBS state without
> application-specific display scripts.

## Desktop

Use this entry when the existing Windows desktop is the intended capture source.

| Field | Value |
|---|---|
| Application name | `Desktop` |
| Image | `desktop.png` |

No command is required. **Virtual Display** is generally preferable for an isolated XR desktop;
**Desktop** is useful when the physical desktop itself should be mirrored.

## Steam Big Picture

Steam replaces its launcher process during startup, so use a detached command.

| Field | Value |
|---|---|
| Application name | `Steam Big Picture` |
| Detached command | `steam://open/bigpicture` |
| Undo command | `steam://close/bigpicture` |
| Image | `steam.png` |

## One Steam game

The Steam URI is more resilient than a path to a versioned executable. Replace `464920` with the
game's Steam application ID.

| Field | Value |
|---|---|
| Application name | `Example Steam game` |
| Detached command | `steam://rungameid/464920` |

## One Epic Games title

Copy the game's launch URI from the Epic Games Launcher shortcut and use it as the command.

| Field | Value |
|---|---|
| Application name | `Example Epic game` |
| Command | `com.epicgames.launcher://apps/<catalog-id>?action=launch&silent=true` |

Launcher URIs are normally more stable than direct executable paths because the launcher owns
updates, authentication, and prerequisites.

## Direct executable

Use a working directory when the program loads assets relative to itself.

| Field | Value |
|---|---|
| Application name | `Example game` |
| Command | `Game.exe` |
| Working directory | `"C:\Games\Example"` |

Alternatively, put the absolute executable path in **Command**:

```text
"C:\Games\Example\Game.exe"
```

## Elevated application

Enable **Run as administrator** only when the target itself requires elevation. Do not elevate a
normal game as a generic troubleshooting step. An application entry can express the same setting
in JSON:

```json
{
  "name": "Administrator-only tool",
  "cmd": "\"C:\\Tools\\Example.exe\"",
  "elevated": true,
  "prep-cmd": []
}
```

## Preparation commands

Preparation commands run before the application, and their paired undo commands run during normal
teardown. Keep them idempotent: reconnect, cancellation, or a failed launch must not leave Windows
in a different state.

Sunshine 3D already owns virtual-display resolution and refresh-rate negotiation. Do not add QRes
or another display switcher to the **Virtual Display** entry; two independent display owners can
race and corrupt the retained-monitor lifecycle.

## Choosing a 3D mode

Application entries launch content; the client chooses how it is presented:

| Client mode | Use it for |
|---|---|
| **2D** | Ordinary mono desktop, text, or the lowest processing cost |
| **Client 3D** | Mono content converted on the Android XR device |
| **Raw SBS** | Authored packed stereo inside a Virtual Display-backed session |
| **Host 3D** | Mono content converted by the Windows NVIDIA GPU |

See [Configuration](configuration.md) for host settings and [Host SBS pipeline](host-sbs.md) for
the Host 3D geometry contract.
