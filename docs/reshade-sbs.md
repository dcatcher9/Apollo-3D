# ReShade stereo input

Sunshine can receive the final full side-by-side texture from a ReShade game. The game renders
on Sunshine's virtual display at its normal resolution. The Sunshine 3D add-on uses the game's depth buffer
to render both eyes into a separate double-width texture. It shares that
texture on the GPU, and Sunshine presents it on local AR glasses or encodes it for a headset.

| Destination | Game and virtual display | Shared stereo texture |
| --- | --- | --- |
| RayNeo glasses with 1920 × 1080 per eye | 1920 × 1080 | 3840 × 1080 |
| Headset with 4K per eye | 3840 × 2160 | 7680 × 2160 |

Sunshine continues to own the virtual display, local glasses presenter and stream lifecycle.
This mode replaces the **live stereo source**. It does not pause Sunshine or turn off local
glasses takeover. Offline conversion continues to use Sunshine's existing Host SBS renderer.

On a streamed headset, **Game 3D** starts with an ordinary W × H stream showing the same view
to both eyes. Sunshine discovers the game export while that stream remains 2D. Only a valid
export and a confirmed stream transition enable the exact 2W × H stereo raster. Selecting
Game 3D does not double the virtual desktop or immediately widen the stream.
The exported stereo raster must match that negotiated resolution exactly. A fullscreen game's
render resolution can differ from the Windows desktop resolution: fullscreen ownership is checked
against the display rectangle, while texture dimensions are checked against the stream. During
mode switches, unavailable exports show the current desktop identically to both eyes and resume
stereo when a valid export returns. Local AR retains its display-sized stereo requirement.

Remote Game 3D polls the independent game export on the stream's presentation deadlines, including
time spent converting and encoding in each frame interval. A static desktop must not add another
whole-frame wait after encoding. Valid packed exports use the receiver's GPU synchronization;
they do not open the unrelated desktop texture or acquire its keyed mutex. Ordinary desktop output and the
duplicate-eye fallback retain capture synchronization.

With Desktop Duplication capture, Sunshine also composites the visible Windows cursor over each
authored eye at the protected game UI's rendered plane. The add-on publishes that frame's applied
per-eye UI displacement with its stereo texture; the host uses the published value without
recomputing scene depth or advancing the UI policy. A cached export retains its matching scalar.
Protocol-1 exports retain the screen-plane cursor behavior. Opening ReShade's overlay selects
screen depth for the system cursor to match its controls; unavailable or disabled game UI
protection and depth diagnostic views also publish zero cursor displacement.
ReShade exports do not contain that separate Windows cursor, even when it appears on the PC
monitor. The capture frame carries an immutable cursor shape/position snapshot; the encoder
uploads the small shape only when it changes and composites into its own scratch texture,
leaving the retained game export untouched. Hidden Windows cursors skip this pass; cursors
already rendered by the game remain in the export. Fullscreen display scaling applies to cursor
placement, each eye clips its own cursor, and HDR cursor white follows the display's SDR white
level. This cursor metadata is currently supplied by Desktop Duplication; the WGC fallback does
not expose it. Install the matching protocol-2 host and add-on together for cursor/UI alignment.

## Native renderer ownership

Game input has a provider boundary shared by depth, camera and UI:

```text
ReShade / Streamline / NGX adapters
       -> bounded capture and source selection
       -> depth_input (paired depth + camera + completed statistics)
          ui_input   (available mask + semantics + observation identity)
       -> stereo calibration and one render_frame_input
       -> native renderer -> stereo export
```

`game3d_depth_input` owns the presentation-facing depth/camera queries and passive
observer calls. Camera conversion and jitter remain attached to the exact captured depth;
there is no separate camera selection that could mix providers or frames. Current readiness,
completed calibration evidence and permission to reuse prior pixels remain distinct.
`game3d_ui_input_provider` owns feature observation, capture requests, mask selection and
upload, channel interpretation, automatic candidate admission and source diagnostics. SDK versions, feature IDs,
generated-frame rules and resource lifetimes stay below these providers. Feature information
may still be reported in diagnostics and controls; it does not select a stereo formula.

The renderer accepts one `render_frame_input`: current color, borrowed depth, resolved
depth/camera/stereo parameters, and eligible UI candidates. UI input distinguishes current color
alpha, captured color alpha, a dedicated UI mask and paired HUD-less scene color. The GPU
validates and selects a usable candidate on each current frame. Missing or rejected inputs
allow the next eligible candidate; no user review or approval is required. Captured pixels
still require independent scope, lifetime, completion and presentation-pairing checks.

The native presentation follows this order: observe presentation requirements and
request future UI captures; open the shared depth lease; acquire depth with its
paired camera/encoding and resolve stereo controls; acquire the completed UI input;
assemble one `render_frame_input`; render and publish its consumed state; close the
lease. SDK callbacks capture inputs at their declared lifetimes before presentation;
the presentation path consumes completed snapshots rather than fetching every SDK anew.

| Resource | Automatic source order and fallback |
| --- | --- |
| Displayed color | Current game swapchain backbuffer, before unrelated ReShade effects. Retained input color never replaces displayed RGB. |
| Depth | Prefer usable SL captures, then usable NGX captures. Generic ReShade depth is the fallback when the API provider does not own the pass. Within an SL batch, try HighResDepth (48), Depth (0), then LinearDepth (49). |
| Camera / encoding / jitter | Use metadata paired with the selected depth capture. Linear-distance depth can work without camera matrices; device depth without a paired projection uses the existing relative-depth calibration when eligible. Never borrow another provider's camera. |
| UI with FG off or no observed FG | Automatically validate captured SL UIAlpha (R; tag 68 in 2.11.x, 69 from 2.12.0, see [buffer contract](#streamline-buffer-type-contract)), UIColorAndAlpha (23, A), Backbuffer (53, A), current color alpha, then paired HUDLessColor (2) difference. |
| UI with FG on | Use the same captured-source order and HUD-less fallback; presented alpha is excluded. HUD-less pairs with its same-batch tagged Backbuffer, else with the real frame after the generated Presents, which reuse its mask. UIAlpha stays manual-only when the loaded interposer is outside the surveyed header range. |

Depth order preserves source continuity: a brief pending copy can retain an admitted
source and its authorized prior pixels. Known FG requirements restrict selection to
the matching SL FG source; missing pixels then use only permitted bounded reuse or
remain unavailable. An established API source's temporary gap does not trigger Generic
recalibration. Explicit manual depth pins bypass automatic API selection. These rules
belong to the capture/provider layer; stereo consumes only readiness, provenance and
the resulting depth/camera values. Manual UI Off always wins. Discovery and capture availability
do not by themselves authorize automatic UI protection.

The maintained GPU code is `tools/reshade/game3d_native.hlsl`, embedded into the add-on at build time.
`game3d_renderer.cpp` owns compilation, pipelines, textures and pass submission for D3D11/D3D12.
The prior external Game3D effect is only a frozen comparison oracle. GPU math, FP32 depth fields,
FP16 eye intermediates, PQ-before-filtering conversion and packed export formats are unchanged.

The native driver runs from the swapchain present callback, after the shared depth owner updates
its source observations and before ReShade draws its overlay. It explicitly opens and closes the
same depth-capture lease, resolves calibration once, renders and submits export. No loaded FX
technique or effect callback is required. Other enabled depth effects get a state-only read lease
around their effects pass without repeating selection, capture or calibration sampling.
The native constant buffer always supports the capture's active rectangle, including allocation
padding and nonzero crop origins. Reference FX must separately expose their rectangle uniform;
the absence of an FX handle must never reject native cropped depth.
Its color input is the game backbuffer before unrelated ReShade effects. Those effects can still
run on the desktop view; they are not part of the native SBS pass graph.

Every Game 3D pass runs inside the game's Present on its graphics queue, so its GPU time is
display latency. A `Sunshine Game 3D timing` log line every 10 seconds reports the present hook's
CPU time and GPU stage times (mean/max ms: inputs, source copy, UI detection, depth conditioning,
eyes, pack, total) from timestamp queries read after each frame's completion fence, without a
wait. Inputs covers the depth and UI captures recorded before rendering. Eyes is the one pass
that renders both eyes into the side-by-side target; pack is used only by older embedded replay
shaders. Both are zero on frames nothing consumed. The CPU entry also splits the slowest present into setup, depth, UI, render and
export, and a rate-limited `Sunshine Game 3D hitch` warning names any present-thread step that
takes more than 8 ms.

ReShade writes every log line through to disk synchronously. In Hogwarts Legacy single lines took
up to ~50 ms while the game streamed assets, and the periodic depth, camera and UI diagnostics were
written from the game's Present (present-hook spikes of 23-170 ms). The add-on therefore queues its
log lines in order and writes them from one thread-pool drain; the present thread only formats and
enqueues. At most 512 lines wait, later ones are counted and reported as dropped, and pending
lines are discarded once DLL detach begins. Unit tests that intercept ReShade's log use a
synchronous build.

The renderer compiles its HLSL for each swapchain size and color space. D3DCompile of the full
source costs 100-300 ms per entry point, and doing it on the Present froze Hogwarts Legacy for
about 2 s at start and 1.4 s after an HDR switch. The add-on now loads or compiles every entry point a
configuration can use (including lazily created UI passes) in parallel on the thread pool; the
Present only looks up bytecode already in memory and never touches the disk. Until
they finish the game stays 2D and the overlay reports "2D: preparing Game 3D shaders". Bytecode is
kept for the process and written to `%LOCALAPPDATA%\Sunshine3D\game3d-shaders`, keyed by the
shader's build SHA-256, the loaded `d3dcompiler_47.dll`, the defines and the entry point. A later
launch loads it without compiling, and a damaged file is recompiled. Once per process the first
pool task prunes files untouched for 30 days and stale temporary files. A failed compile is
remembered rather than retried every Present. Replay and test renderers with explicit sources
still compile synchronously and never use the cache. With the cache in place the slowest renderer
setup Present went from 1,969 ms to about 10 ms in the 4K provider fixture.

The vertical and horizontal limit passes are mechanical translations of the host shaders, whose
tiling is described in [Host SBS pipeline](host-sbs.md). Each group owns eight adjacent columns or
rows, so the passes no longer hold whole lines in group memory. UI pinning always runs as its own
pass (`SunshineApplyUICS`) after the horizontal pass; the adaptive probe observes the unpinned
field between them, as before. That pass is tiled the same way (`SUNSHINE_UI_PIN_LINE_GROUPS`):
each chunk finds its first and last UI texel, the carries give every chunk the nearest UI texel on
each side, and a look-ahead cursor yields each texel's distance. Rows without UI are only scanned
once, and no row is held in group memory. With UI active the pass fell from about 0.1 to 0.02 ms in
the 4K provider fixture; distances, and therefore the pinned field, are unchanged. Results are byte-identical. At 4K on an idle GPU the vertical pass
went from 0.44 to 0.17 ms and the horizontal pass, including the UI pass, from 0.35 to 0.24 ms; the
whole renderer went from 1.38 to 0.91 ms. The shader declares the lines per group as
`SUNSHINE_LIMITER_LINE_GROUPS` and `SUNSHINE_UI_PIN_LINE_GROUPS`; the renderer sizes its dispatches
from those values, and the shader test requires them to match the reflected thread-group shapes.
A dump whose embedded shader predates them replays with one group per line and in-limiter UI
pinning. Replaying two Hogwarts Legacy dumps with either shader reproduced every captured
artifact byte-exactly.

All rendering stays on the runtime's graphics queue. Native passes add no CPU fence wait or full
frame CPU readback. A completion fence protects replacement of the renderer's working set;
in-flight reconfiguration skips output instead of waiting. A Present that never reaches `finish_present`
leaves its recorded work unsignaled rather than assumed complete: the next presentation renders
again, and the next completion signal retires both, so working-set replacement waits until then. ReShade drains the GPU before runtime
destruction, allowing normal resource disposal. Exceptional hot-unload retains an unfinished
working set until process exit rather than freeing commands' resources prematurely.
D3D11 context state is restored across capture and rendering; D3D12 uses ReShade's immediate list.
The native game target stays mono, and the existing overlay/export fence includes the controls
composited into both eyes. Only the existing small calibration readback reaches the CPU.
Multisampled and RGBX/BGRX swapchains are not admitted by this native driver: ReShade uses an
internal resolve target for their GUI, which this overlay path does not own. The add-on reports
an unavailable display mode rather than redirecting the GUI to the wrong texture.

## Setup

1. Build and install the [Sunshine SBS ReShade add-on](../tools/reshade/README.md) for the game.
   Use the ReShade version with full add-on support and a supported DirectX API.
2. Run the package's `install.ps1` with `-GameExecutable <game.exe>`.
   `SunshineSBS.addon64` contains the GPU renderer, controls, depth capture and export; no FX shader
   or external Depth3D checkout is required. The installer migrates saved strength and depth-view
   settings into `[SUNSHINE_GAME3D]` in ReShade.ini, preserving existing native preferences.
   Obsolete owned Game3D shader/include files are backed up before removal, and old stereo
   techniques are disabled. Original SuperDepth3D reference files/settings remain available.
   An explicit `-ShaderDirectory` installs a reference renderer and disables native Game 3D;
   this is for comparison testing, not ordinary setup.
3. Put the game in fullscreen or borderless fullscreen on Sunshine's virtual display at the normal resolution.
   Its client area must exactly cover that display, and ReShade and Sunshine must use the same GPU.
4. For a streamed headset, select **Game 3D** in Moonlight 3D. Its own options pane controls
   **Resolution**, **FPS** and **Bandwidth**. Choose a normal source resolution such as
   1920 × 1080 or 3840 × 2160. No Sunshine provider toggle or host restart is required.
   Both host and client must support the negotiated Game provider v1 extension. Select HEVC
   or AV1 for 4K per eye and use a client capable of decoding the resulting 7680 × 2160 stream.
5. For locally connected glasses, retain the usual Sunshine local AR setup and the glasses'
   hardware 3D display mode. In Sunshine's Web UI, open **Configuration → Essentials**, enable
   **Use ReShade for local AR 3D**, save and restart Sunshine. The equivalent configuration is
   `sbs_reshade = enabled`; it defaults to disabled. This setting selects only the local AR
   provider. Streamed **Host AI 3D** always uses Sunshine's AI conversion.
6. In **ReShade → Add-ons → Sunshine 3D**, the **Game 3D** panel keeps **Enable Game 3D**,
   **3D strength**, and **UI protection** in fixed rows above the live diagnostics.
   Strength defaults to **50%**. **Reset** buttons keep their space and are disabled at the
   default value; opening the UI preserves existing saved values.
   **Status** shows output state, UI detection, Frame Generation, depth source/resolution and
   depth freshness. **Troubleshooting** contains **Depth view** (default game image),
   **UI mask source**, and **Use automatic depth**, followed by **Advanced capture settings**
   and **Manual depth selection**. **Calibration** contains conversion and current/target gain.
   Live messages and word wrapping cannot push the primary controls or tab bar down.
   Troubleshooting controls precede live buffer details. Sharpening is not part of the native controls.
   Depth setup is always automatic. There is no Manual shader mode, per-game geometry/weapon
   profile branch or warp selector. Image controls are owned by the add-on and automatically saved
   independently of ReShade shader presets, Performance Mode and the global effects toggle.
   Manual **depth-buffer pinning** remains a separate source-selection override.
   **Depth path** identifies the provider actually selected: **Streamline (game provided)**, **NGX (game provided)** or
   **Generic (fallback/manual override)**. Active generic capture shows a warning that game-provided
   depth and camera data are not being used. Loading a Streamline DLL alone does not establish an
   active Streamline depth provider. AMD depth integration is not implemented yet.
   Current depth-source status is in **Status**; raw buffers are under **Manual depth selection**.
   Streamline's current depth has its own **ACTIVE** row because its native resource need not appear
   in the generic list. **Last valid** describes history only; unavailable current depth remains 2D.
   Sunshine's host depth-strength control applies to Sunshine-generated stereo and offline conversion.

When the game requests Frame Generation, the **Status** tab shows guidance below the controls.
For **3× or higher**, select **Off** or **2×** in the game's settings. **2× may work**; turn it off
if artifacts or stutter appear. **DLSS Super Resolution can stay on.** The message reflects verified
game settings, including Auto mode, rather than measured generated output. Ambiguous settings are
not guessed, and Sunshine does not change the game's Frame Generation options.
When Automatic has identified the relative-depth fallback without associated camera data, it
suggests trying **Frame Generation 2×**, if available, because some games supply Streamline camera
data through FG. Availability is not guaranteed. An already observed enabled FG setting instead
shows that camera data is still unavailable; the existing 3×/higher guidance remains above it.
The hint is not shown during the initial unknown-depth state. A busy or ambiguous FG observation
does not produce a suggestion to enable a mode that may already be enabled.

**UI protection** offers **On / Off / Auto**, saved per game in ReShade.ini as
`SourceAlphaUIMode` (0 Auto, 1 On, 2 Off). All runtimes in the game share this choice and its
per-game configuration file. Auto is the default. On and Off are persistent manual
overrides, including across game restarts. The previous `SourceAlphaUI=false` migrates to Off;
the old enabled heuristic migrates to Auto. An explicit new mode takes precedence over the old key.
No game-specific detection profile is used. When protection is enabled, white and gray pin already-composited
UI to one global plane selected by the adaptive display-fraction policy below, while
black keeps scene depth. This is a placement policy, not recovered UI geometry or physical distance.
Auto detects eligible inputs continuously and validates their current pixels on the GPU. It
requires no human review, approval, game profile, startup scan or confirmation timer. The
candidate order is authenticated UI opacity, UI color alpha, captured real-input alpha,
current color alpha unless FG is known enabled, then a paired HUD-less/final-color difference. A missing or
rejected candidate allows the next candidate in that same render. If none passes, UI protection
is off for that frame; it can recover automatically on a later frame.

The **Troubleshooting → UI mask source** selector offers automatic discovery and explicit source overrides,
including HUD-less difference. It is useful for diagnosis; normal use needs no source choice.
UIAlpha is admitted to Auto when the hooked interposer's version lies in the surveyed header
range that assigns it (2.11.x tag 68; 2.12.0 through 2.14.x tag 69); see the
[buffer contract](#streamline-buffer-type-contract). Outside that range it stays manual-only, and
its presence never blocks tag 23, Backbuffer, current alpha or HUD-less comparison. The same GPU
content checks apply as for every candidate. Manual On can use eligible alpha as an explicit override, including UIAlpha and
an all-one alpha channel. An all-one manual mask places the whole frame at the UI plane.
Off always disables protection. Neither manual mode overrides GPU capture ordering,
source lifetime, scope, format or pairing requirements, and changing mode does not recalibrate depth.

For automatic alpha selection, the GPU rejects nonfinite or out-of-range samples, an empty
mask and a mask covering at least 90% of the image. These are conservative content checks,
not a claim to recover UI semantics in every game. An opaque channel is ambiguous: it is
either full-screen UI or no UI information at all (Hogwarts Legacy's UIColorAndAlpha is the
opaque final image during play). Selection is recomputed from the current frame instead of using
an earlier CPU decision or latching a startup result.

One exception uses session history. Once completed GPU samples have accepted the same alpha
candidate as a partial UI mask at least three times over at least 2 s, that channel is proven to
carry UI coverage for the rest of the game session. A later frame in which the proven channel is
exactly 1.0 on at least 98% of pixels is full-screen UI and stays flat (sampled source 7). HUD-less
checks still decide first: an accepted HUD-less mask or full-screen result wins, and the proven
channel applies only when they accept nothing. Expedition 33's pause menu with FG on showed why:
the menu tints the live scene, so 76% of pixels differ from HUD-less, which is neither a HUD mask
nor full-screen UI, while the proven alpha is opaque. Exact opacity matters: a channel that is merely above zero almost everywhere can carry other data, such
as luma. Proof is per candidate, so a presented-color alpha proven with frame generation off does
not prove the tagged Backbuffer's alpha, and a title screen shown before any gameplay has no proof
yet. Clair Obscur: Expedition 33 needed it: its alpha covers 2-23% of pixels during play and is
exactly opaque in full-screen and pause menus, and with frame generation off, or before the first
level has loaded, it tags no HUD-less image.

The D3D12 adapter captures Streamline UIAlpha (69, red), UIColorAndAlpha (23, alpha),
Backbuffer (53, alpha) and HUDLessColor (2, RGB) independently of FG being enabled. A fresh
successful tag establishes its epoch, observation revision and viewport even when no FG
options were observed. Each source has at most two snapshot reservations: latest ready and
pending. The shared native owner also enforces its 32-slot, 256 MiB auxiliary limit. A full or
invalid high-priority candidate cannot consume another kind's reservations. Null tags, failed
SDK calls, incompatible extents/formats, observation loss and expiry revoke the affected
available pixels. Recreated layouts and runtimes cannot inherit an earlier capture scope.

HUD-less comparison additionally requires matching format and full extents. When the game also
tags its Backbuffer (53) and that capture was made in the same Present interval as HUDLessColor
(equal Presents since each tag), the two images belong to one tag batch and HUD-less is compared
with that tagged Backbuffer. This exact pair does not depend on which Present Game 3D renders, on
frame generation or on queue timing; the mask then protects the current frame. Hogwarts Legacy
needed it: with DLSS-G on, differencing HUD-less with the presented color changed nearly every
pixel, while its same-batch HUD-less and Backbuffer differed only in the UI (5.7-12.8% of pixels,
over 200 of 256 tiles unchanged). A manual HUD-less choice captures the Backbuffer only for this
pairing. Without a same-batch Backbuffer, HUD-less must pair with the real frame of its tag: a current source presentation generation
one beyond the generation captured at the tag, or `generated_frames + 1` beyond while FG is known
enabled (at most three generated frames). Streamline presents the generated frames first, so the
Present that matches HUDLessColor comes last. The integrated depth Present callback advances that
generation before the Game 3D render. A merely recent snapshot from an earlier presentation does
not pass. Each generated Present in between shows interpolated scene color, which must not be
differenced. It copies nothing and reuses the mask the GPU resolved on the preceding real frame,
for at most three consecutive Presents, and only when that frame compared a HUD-less candidate.
A real frame whose HUD-less image pairs only by Present counting while FG is on is not exact.
Directly after an exact decision it keeps that decision and mask the same way, within the same
three-Present bound. Detecting afresh from the inexact pair would lose full-screen UI on that frame.
In Expedition 33, 2-10% of menu frames paired this way and flipped the menu between flat and 3D.
A capture from another queue can still be incomplete or unretired at its own real frame. When
the newest ready capture belongs to a real frame one or two Presents ago, it is compared with that
frame's retained color, and the resulting mask protects the current frame. The renderer starts
retaining the last two presented colors (two source-size copies, one extra copy per Present) only
after the first such late capture, and stops copying 120 Presents after the last one; games whose
captures pair on time never allocate them. A late
capture older than the retained history does not pair.
Hogwarts Legacy (SL 2.6.10, DLSS-G on) showed the pattern: its real frame matched HUDLessColor
on 93.5% of pixels two Presents after the tag, and its UIColorAndAlpha was the opaque final image.
It also presents on a different queue from the one that records its tags, so pairing does not
require the producer's queue. The capture rules below already admit another queue's pixels only
after the producer completes and its recording retires.
Local-only snapshots use the existing depth owner's queue-ordering contract: after the SDK
call succeeds and the producer is actually submitted on the presentation queue, a local GPU
copy may follow it without waiting for CPU-visible completion or producer Reset. The consumer
copy rechecks this admission and binds one queue and destination. Foreign queues and shared
dump acquisition still require producer completion and recording retirement. No reverse queue
wait is inserted, and all producer/consumer fence and recording leases remain required before
storage can be reused. Unsubmitted, failed or replay-invalidated captures remain unavailable.

The GPU compares native RGB in matching color space. A changed pixel exceeds 4/1023 for
R10 color or 2/255 for other SDR/PQ color; floating-point scRGB uses 0.005 times the larger
of one and the current pixel's largest absolute RGB component. Both RGB inputs must be finite.
Acceptance requires a nonempty difference covering fewer than 25% of pixels, at least 75% of pixels within half the
threshold, and at least 128 of 256 tiles with 99% matching pixels. These guards reject broad
scene mismatch before the difference becomes a UI mask. One exception covers full-screen UI:
when the pair is exact (a same-batch Backbuffer, or Present counting with frame generation off),
at least 98% of pixels changed and at least half of the HUD-less pixels are lit (above eight times
the threshold), the whole frame is UI and stays flat. Hogwarts Legacy's title screen needed it:
the game keeps rendering its next 3D scene behind the menu and tags that scene's depth and
HUD-less image, so 99.9% of pixels differed while the depth matched the hidden room. A mismatched
pair can also change every pixel, so an unverified pair is still rejected (after an exact
decision it is held, as described above), and a black HUD-less image never counts as a scene. HUD-less comparison remains guarded in manual modes. Its binary mask identifies changed pixels, not exact compositing opacity, and cannot detect
UI whose color matches the underlying scene. Local motion or other postprocessing differences
can still resemble UI; live game/headset validation remains necessary.

The native path uses GPU statistics, reduction/selection and mask passes. Each of the 16x16
statistics tiles is one 256-thread group, the reduction sums the 256 tiles in parallel, and the
mask pass loads only the selected candidate; the integer counts and the mask are unchanged. At 4K
this took UI detection from about 0.148 to 0.12 ms averaged over the provider fixture, which
is mostly bound by reading the candidate textures. A bounded asynchronous
96-byte summary may be read at 100 ms intervals for diagnostics; it never authorizes protection
and there is no full-frame CPU readback. Besides the decision, it carries the candidate bits the
shader was offered, each alpha candidate's covered, invalid and exactly opaque pixels, the
proven-alpha bits, and the HUD-less changed, unchanged, non-finite and lit pixel counts with
matching tiles. Recording accepted partial alpha from this summary is the only way it feeds back:
it proves a channel for later frames and never authorizes the current one. The
`Sunshine UI protection` log (`sampled_candidates`, `sampled_alpha_covered`,
`sampled_alpha_opaque`, `proven_alpha`, `sampled_hudless`) and the dump's
`source_alpha_auto.sampled_evidence` report it, so a rejection names the failing check. Source availability, GPU validation and actual applied
protection remain separate diagnostic facts. Older startup fields describe a retired heuristic.
NGX UI resources remain discovery/dump observations; live NGX, FSR and XeSS UI adapters are not
implemented. Current presented alpha is excluded while the retained FG mode is known enabled;
an initial unknown mode alone does not establish that the presented color is generated.

Live `record_local_texture` and dump `record_diagnostic_texture` captures use the same
`auxiliary_state_policy` and preserve the declared SL v2 lifetime. `OnlyValidNow` inputs use
`texture_state_policy::prefer_observed_recording`; `ValidUntilPresent` and `ValidUntilEvaluate`
use the strict `source_contract` policy. A synchronous copy does not rewrite a longer-lived tag
into `OnlyValidNow` or authorize overriding a state declaration intended for later consumption.
Unknown SL v2 lifetimes are rejected; SL v1, which has no lifecycle field, keeps its existing
synchronous capture. This applies to HUDLessColor, UIAlpha, UIColorAndAlpha, Backbuffer and other
optional masks/hints. Resource content and local/shared storage do not change GPU-state proof,
and the dump does not require or fabricate FG.
For the at-call policy, a supported, known, unblocked, nonzero state observed on the same
command recording takes precedence over the provider's declaration. This permits a stale declared
UAV hint when the recording independently proves the resource is currently a render target.
If that recording has no state entry for the resource at all, an explicit nonzero supported
provider declaration may supply the state, but only with declared-state proof and compatible
render-target, UAV or depth resource flags. This restores the original trust in a declared source
when no current barrier was observed; it does not independently validate the declaration.
There is no declaration fallback over a blocked or unknown entry, COMMON/zero, a split barrier,
lost or invalid recording evidence, an active render pass, or a source requiring
`observed_nonzero` proof.
The [matching Streamline 2.7.30 DLSS-G contract](https://github.com/NVIDIA-RTX/Streamline/blob/v2.7.30/docs/ProgrammingGuideDLSS_G.md#tagging-recommendations)
requires only type and extent for the special Backbuffer tag; its other tag inputs are optional.
The fallback therefore requires an explicitly supplied state rather than assuming one from the tag.
The private copy transitions from and restores exactly the selected state, preserving whether
that state came from an observation or the provider declaration.
Depth copies, NGX and default local texture copies retain
their existing declaration checks. Shared dump allocations remain immutable after host
acknowledgement; local retired storage may be reused. This storage distinction does not select
the resource-state policy. The declaration fallback still needs live game/headset acceptance.
Every proof/policy combination above, including NGX's SDK input contract, resolves through the one
table in [`capture_state_policy.h`](../tools/reshade/capture_state_policy.h): declared (must match an
observation), observed only, observed-else-declared and observed-else-contract. Blocked or
incomplete observations reject every rule. Its CPU test enumerates every proof, policy, claimed state,
observation and resource-flag combination.

The FG decision is independent of depth-provider selection, including a manually pinned Generic
buffer. The last confirmed mode survives transient observation loss and effects reloads; a new
confirmed mode or observer/runtime shutdown updates it. Renderer, panel and dump share that
presentation decision. An enabled FG mode does not identify an individual generated frame.
UI placement is independent of shape protection: explicitly supplied UI-layer geometry may define
its placement, but the current SL/NGX adapters expose no authoritative target UI depth. This path
therefore uses mode 5, `display_fraction`, while the scene retains its contrast-midpoint zero.
The shader consumes `pUI = f * C`, where `C` is the authoritative current positive display
bound, including strength and stereo blend. Live policy controls absolute per-eye parallax
`pUI` and derives the consumed fraction `f = pUI / C`. Its five targets are
`min(L, 0.50 * C)` for `L = {0, 0.001, 0.002, 0.003, 0.0035}` in source UV. At source width
3840 these absolute levels are 0, 3.84, 7.68, 11.52 and 13.44 pixels per eye before the relative
cap. Increasing scene strength does not increase the 0.0035 independent ceiling. This is a
trial value, not a general headset comfort guarantee. The policy starts at screen depth;
unavailable stereo resolves to zero rendered UI parallax. Scene zero, gain and display limits
remain unchanged.

The observational GPU pass divides the screen into a 16-by-16 grid but scans only the 144
tiles with both zero-based coordinates in `2..13`. This is the central 75% of width and height:
`[floor(W/8), floor(7W/8))` by `[floor(H/8), floor(7H/8))`. Outer tiles write empty coverage
records without loading the mask or scene. Their geometric pixel count remains initialized
for readback validation. This omits 43.75% of image pixels at dimensions divisible by 16;
integer bounds define the exact saving otherwise. Border UI remains protected by the normal
UI conditioner at the same global plane and moves with the system cursor.

Inside the central rectangle, each pixel first loads the selected alpha. Only finite positive
alpha causes a scene-field load and five comparisons. These measure the conditioned scene
before UI pinning, so they cannot measure the previous UI correction. A covered pixel conflicts
at candidate `i` when `scene_parallax + 0.05 * C > min(levels_uv[i], 0.50 * C)`.
The CPU sums the central counters once. A candidate conflicts when its bad-pixel count is
strictly greater than 20% of central covered UI **or** strictly greater than 2% of all pixels
in the central rectangle. The second condition prevents a large clear panel from diluting
substantial foreground overlap. Both use the same conflict numerator; central image area is
`(floor(7W/8)-floor(W/8)) * (floor(7H/8)-floor(H/8))`, independent of UI coverage. Integer
cross-products preserve the exact thresholds without rounded percentages. The first candidate
that exceeds neither bound is the required target; if none qualifies, it is the cap and
`capped_conflict` is set. Exact threshold equality and empty coverage do not trigger entry.
There is no per-region grouping, pixel floor, image subsampling or histogram. Each GPU tile
still keeps five counters; the extra area comparison adds no GPU work or readback data. Dump
records `center_pixels`, both area thresholds and `observed_front_cap_uv` alongside coverage
and conflicts so the decision remains interpretable after a bound change.

A nearer requirement must persist across distinct observations for at least 100 ms before
changing the target. The controller chooses the least forward requirement sustained through
that confirmation. After reaching its current target, retreat requires a shallower candidate
with conflicts strictly below both 15% of covered UI and 1.5% of central image pixels for
1500 ms of fresh observations. Equality at either release threshold blocks retreat. The nearest candidate needed during that dwell becomes the retreat target, so a
briefly clearer observation cannot pull it too far back. Empty coverage permits return to zero.
Applied movement is limited to 0.03 UV per second forward and 0.005 UV per second backward.
Targets and movement use absolute UV. A changed scene bound immediately enforces its current
half-bound cap, invalidates old-cap observations and disarms movement until fresh evidence
arrives. Fraction serialization rounds conservatively so its shader product cannot exceed
the controlled absolute position or current cap.

Probes run at most once per 100 ms with one bounded asynchronous readback outstanding. Source
and routing changes invalidate evidence without restarting this renderer-wide work interval. Each
tile contains eight uint32 words; the 16-by-32 RGBA32_UINT record texture occupies 8 KiB.
No source image is read back. Source age uses the older consumed depth/retained-alpha timestamp
and must be at most 250 ms. Duplicate depth or retained-mask identities do not renew evidence.
Missing, expired, invalid or failed observations hold the absolute applied position, subject
to the current cap, and disarm ramps and continuity. Gaps over 250 ms break continuity. A fresh
eligible source scope starts at screen depth. Generic depth uses runtime, raw-basis and routing
epochs as that logical scope; normal A/B/C allocation rotation and different per-allocation layout
epochs preserve placement and pending display-space observations. Every observation still retains
its physical source/layout, original capture time and global frame sequence. Calibration admission
must validate the currently selected depth; a changed routing group, basis or runtime revokes the
old scope. Provider sources keep their existing feature/viewport/observation-revision identity.
Alpha On / Off / Auto admission remains separate
from placement. Live probes require `SUNSHINE_UI_ABSOLUTE_LEVEL_PROBE` so historical relative
count shaders cannot be misinterpreted as absolute statistics.

A cap can leave scene geometry in front of UI; fresh geometry can also arrive before the next
probe or before a ramp finishes. The existing local UI conditioner continues to adjust adjacent
scene pixels; no additional scene-wide compression is added. These correctness and work bounds
do not establish headset comfort or a measured frame-time improvement.
Historical nearest-depth mode 2 creates its reduction textures and pipelines only when first
requested. Normal adaptive mode 5 does not allocate or compile those legacy resources; mode-2
preparation failure occurs before command recording and leaves ordinary rendering usable.
The switch can disable automatic protection because arbitrary
game backbuffer alpha is not universally a UI mask and selective coverage can still be a false positive.

All candidate paths feed the same UI protection shader. Displayed RGB still comes from current
color, never retained input RGB. Captures share the existing depth transport's producer/consumer
fence retirement and maximum source age. Automatic detection adds the three GPU passes described
above and a bounded diagnostic summary, without a new queue, GPU wait or CPU image readback.
Adaptive placement separately reads only its bounded conflict records described above.
Captures without the required submission/queue ordering are skipped rather than waiting. Scope, viewport,
resolution or observation-revision changes revoke old masks. Alpha and depth have independently
recorded source identities: this is bounded previous-input reuse, not an exact generated-frame
color/depth/mask pairing. Fast-changing UI can therefore briefly lag behind the current color.
Local FG snapshots reuse their allocations only after all producer and consumer GPU work and
recordings retire. They expose no shared handle; optional diagnostic snapshots shared with the
host remain immutable even after acknowledgement. Both use the same bounded capture owner.
The renderer copies each admitted capture into its private UI texture once, then reuses those
bytes for later presentations while that capture remains valid. Renderer replacement clears this
copy identity; failed replacement copies never authorize stale pixels.
Live adaptive placement uses conflict counts, not a nearest-covered-depth maximum. Historical mode 2
remains available for replay: a 16-by-16 tile pass and a 256-thread reduction resolve
`max(submitted midpoint floor, nearest valid decoded q under finite positive selected alpha)`
into one R32 float on the rendering queue. Its crop/jitter mapping matches the scene candidate;
every covered valid pixel contributes, and empty coverage retains the submitted floor. These
resources are overwritten on each mode-2 render, with no CPU readback or additional queue wait.
After the existing horizontal conditioning, each row computes the exact distance `d` in pixels
to positive finite source alpha and clips its signed displacement to
`pUI +/- 0.5 * max(d - 1, 0) / source_width`. Live mode 5 supplies the applied display-fraction displacement
described above. This existing local conditioner can compress nearby foreground or bring nearby
background forward toward the UI plane; the trial adds no separate scene-wide compression. The one-pixel
horizontal collar protects the bilinear color footprint. This rigidly shifts UI in each eye and
preserves the horizontal invertibility bound without overlaying a second copy of already-composited
text. Empty rows retain their original field exactly. The distance scan itself reuses the horizontal
pass and its shared memory. It does not retain the
vertical shear bound across UI-row boundaries. Half-transparent UI retains its original color but
locally flattens the background beneath it; exact independent stereo background requires a separate
HUDless image and UI color/alpha layer. Source alpha interpretation is not inferred from NGX or SL
provider identity. Dump/replay records the effective switch as `replay.source_alpha_ui`, the chosen
request as `source_alpha_ui_requested`, and the reason as `source_alpha_ui_status`. `source_alpha_auto`
records automatic/manual state separately from the current render. New `ui_source_detection`
metadata replaces session qualification and carries no approval flag. Its asynchronous summary
may describe an earlier frame and is diagnostic only. Historical monitoring, timer and coverage
fields remain readable in older packages but do not authorize live UI. Offline replay uses the
frozen effective value and mask without reclassification; explicit replay
overrides still honor entirely white masks. It uses the separate 16-byte `b1` UI constants; the 80-byte geometry `b0` ABI
is unchanged. `source_alpha_ui_fg_mode` records the retained mode and its observation provenance.
The UI constant buffer stores four 32-bit words: enabled, mode, a mode-dependent float word, and
mask channel (`0` alpha, `1` red). New dumps record `sunshine_game3d.ui_parameters.v7` and
`ui_constant_binding.mask_channel`; older v2–v6 packages require zero in the formerly reserved word.
Live mode 5 requires
`#define SUNSHINE_UI_DISPLAY_FRACTION_PLANE 1`. Its third word is the exact applied display fraction,
not inverse depth. The binding labels it `word2_role: front_limit_fraction` and records
`front_limit_fraction` plus `front_limit_fraction_bits`. Finite fractions in `[0,0.75]` are valid
for frozen replay compatibility; live adaptation is capped at 0.50.
Out-of-range and nonfinite words retain their raw bits for replay but produce screen disparity.
Nonfinite values have a null JSON description. Optional `replay.ui_adaptive` records levels,
target and required indices, applied fraction, capped conflict, probe age, accepted depth/mask
identities and coverage/conflict counts. Current policy counts include only UI inside the central
rectangle; full-screen alpha-enable detection retains its separate counts. These describe accepted observations rather than a new
measurement of the dumped frame. Replay freezes the exact applied fraction, including intermediate
ramp values, without rerunning the classifier, its history or its observational probe. No new
GPU artifact is required for replay.
Historical mode 4, `shallow_front`, retains v5 and requires `#define SUNSHINE_UI_SHALLOW_FRONT_PLANE 1`.
It uses one quarter of `C`; its inverse-depth word is unused and its binding retains the required
`front_limit_fraction: 0.25` metadata without adding another constant-buffer word.
Historical mode 1 retains the applied inverse-depth plane in
`sunshine_game3d.ui_parameters.v2`; the shader must consume `Sunshine_UIPlaneMode` and the supplied
inverse depth. Historical mode 0 retains the screen plane and the same v2 ABI. Historical mode 2
uses the submitted depth as a floor for the GPU maximum and retains the v3 ABI; its shader requires
`#define SUNSHINE_UI_NEAREST_PLANE 1` and the reduction entry points. Historical mode 3,
`front_limit`, retains v4 and requires `#define SUNSHINE_UI_FRONT_LIMIT_PLANE 1`. It places UI at
the current positive `b0` display budget times strength and stereo blend, subject to warp
readiness; its inverse-depth word is unused, including malformed bits preserved for replay.
Mode 2 requires v3 or newer, mode 3 requires v4 or newer, mode 4 requires v5 or newer, and mode 5 requires v6 or newer.
Unknown modes,
invalid depth in modes 1/2, and unavailable stereo resolve to zero UI parallax.
Dump 3D records the exact submitted UI constants, separate from diagnostic targets. Old packages
without these constants retain mode 0. Replay's `--ui-inverse-depth` selects mode 1;
`--ui-nearest-floor` selects mode 2; `--ui-plane screen` selects mode 0;
`--ui-plane front-limit` selects mode 3; `--ui-plane shallow-front` selects mode 4; and
`--ui-front-fraction 0..0.75` selects mode 5 with an explicit frozen fraction.
`--ui-plane adaptive` is invalid because a single-frame package has no policy history to replay.
Depth-mode geometry overrides recompute UI displacement on the GPU. Front-limit and shallow-front
placement respond to the applied display budget, strength, blend and readiness, without deriving
a displacement from scene gain or zero. Mode 5 keeps the consumed fraction fixed while applying
the current display bound; geometry overrides do not rerun its classifier.
Mode-3 and mode-4 dumps mark the inverse-depth word unused
and the reduction inactive. Historical mode-2 dumps distinguish the
submitted floor from the GPU-resolved plane; replay additionally saves that resolved scalar.
Live rendering never waits to read it back.
`replay.ui_alpha_source` identifies `none`, `source_color`, or `ui_source_color`. When a retained
input was consumed, required artifact 33 (`ui_source_color.bin`) preserves its exact typed pixels.
For automatic detection this artifact is the resolved full-resolution `R32_FLOAT` mask, consumed
through its red channel. It includes an all-zero result when every candidate was rejected;
the enabled mask path does not by itself imply any protected pixels. Replay uses these frozen
values and does not run candidate selection again. Optional candidate metadata remains separate.
RGBA sources retain their native color format; red-channel masks additionally accept R8_UNORM,
R16_UNORM, R16_FLOAT and R32_FLOAT. No conversion pass or color-space interpretation is applied.
`ui_source_alpha.png` shows an alpha-channel input and `ui_source_mask.png` shows a red-channel input.
Its provenance records the
input observation, age and completed capture; it never substitutes the optional latest tag-53
diagnostic snapshot. Live dump transport is version 3 with 40 descriptor slots; offline replay
also accepts earlier packages without an external alpha input.
`replay.source_alpha_capture_attempt` freezes the latest scope- and lifecycle-qualified live
alpha capture attempt observed at that render, including its outcome and available capture-state
evidence. It describes an attempt, not the consumed mask or an exact color/depth/alpha pairing;
the consumed-alpha artifact and its provenance remain authoritative when a mask was used.
Its `hook_gate` records the latest public tag callback's scope, observed UI kinds, matching
request count and admission result; `request_generation` identifies the retained live request.
A bounded `Sunshine UI capture gate` log reports the same evidence. An admitted hook with a
zero capture boundary and a changing request generation indicates that the request was replaced
or invalidated after admission, rather than that the game supplied no UI tag. Its
`hudless_presents` counts, since the previous line, Presents whose newest ready HUD-less capture
paired with its `batch` Backbuffer, was the current `real` frame, a `late` real frame paired with retained color, a `generated`
Present that held the previous mask, `stale` (older than the retained history), `other`, or
`none` (no ready HUD-less capture). Unchanged failed
or successful FG Off options preserve independent UI tags; an actual FG knowledge/mode loss
still revokes the old scope once and requires fresh input.
Capture diagnostics preserve the raw declared hint and independent observed state.
`copy_state`, `copy_state_known` and `used_observed_state` identify the selected copy/restore state
and whether state selection used observed recording evidence (including the shared UI capture policy).
A declared-state selection reports `copy_state_known=true` and `used_observed_state=false`;
"known" here identifies the selected copy state, not independent validation of the provider hint.

The panel controls native `Strength`, `DepthView`, `Enabled` and `SourceAlphaUIMode` settings in ReShade.ini,
and provides the optional UI source selector and automatic-detection status described above.
Opening the panel does not rewrite values. Edits apply immediately and are saved automatically,
independently of shader preset Auto Save. Each reset changes only its own parameter.
These controls remain independent of shader enablement, Performance Mode and preset transitions.
Source-owned camera/raw inputs determine depth decoding, coordinate transforms and screen plane;
no `SUNSHINE_GAME3D_AUTOMATIC`, `SUNSHINE_GAME3D_CAMERA_DEPTH` or
`SUNSHINE_GAME3D_GENERIC_CONFIG` mode definitions are needed. The installer removes those
obsolete definitions while preserving unrelated definitions. The **Calibration** tab displays the
current zero plane and independent gain, or the last applied values while paused, in a labeled table.
**Conversion scale** and **Conversion offset** identify validated camera decoding;
**relative depth (assumed infinite far plane)** identifies the normal raw-depth fallback.
The gain's Target column shows `L/Q`, where Q is the largest decoded inverse depth in the active
depth region and L is the normalization for the output shape and parallax limit. The panel also
shows that limit at the current strength. A status message identifies gain below its target while
adapting; it does not imply missing depth. Gain adapts automatically in either direction;
the panel has no recenter or manual gain control. The zero has independent current and target values.
The numeric reference uses game units for projection depth and relative units for raw
fallback, so numbers from different representations must not be compared directly. A depth gap
shows the last applied gain and plane as paused. FG depth reuse preserves those controls with
the sampled real depth frame; user strength reductions apply immediately, while increases wait
for fresh real depth.
Native Game 3D has one renderer: the current Sunshine Host V2 vertical/horizontal displacement
conditioner and inverse, implemented in the embedded `tools/reshade/game3d_native.hlsl`.
It consumes native game depth with the existing source-owned camera/raw scale, screen plane,
strength, capture crop and jitter correction. It does not run AI depth estimation or the removed
SuperDepth3D ray search, reconstruction, depth-history and per-game geometry branches.
The host's canonical [warp contract](host-sbs.md) owns the conditioning and inverse algorithm;
Game3D supplies its own native-depth-to-displacement adapter.
The [Game3D pipeline comparison](game3d-pipeline-comparison.md) records the stage/constant audit
and measured differences from the retired renderer.

The renderer preserves full source resolution through 3840×2160 (also portrait), HDR conversion,
export, overlay and mono fallback. Both source dimensions must be at most
3840; larger sources show an explicit unsupported-resolution message and export duplicate mono
eyes. There is no hidden alternative warp. Per-eye displacement is bounded to 4% of source width
before conditioning. Conditioning can shorten stretched mixed-color strips, but can bend outlines
and flatten nearby background depth; it cannot recover unseen background or repair unaligned depth.
The September 18 live test reported reduced halos. That report does not establish artifact-free
output in every game or quantify latency; compare actual GPU timing separately.

The installer migrates the old `Depth_Adjustment` and `Depth_Map_View` preset values, including
explicit zero strength, into native `Strength` and `DepthView` settings when no saved native value
exists. It backs up and removes the owned Game3D FX/support files and disables competing stereo
techniques. Native Game 3D requires no user preprocessor definitions and has no Home-tab shader
entry. Original SuperDepth3D files, profiles and settings remain in their separate reference
installation. The earlier independent `SunshineDepth3D.fx` experiment also remains a reference,
not a Game3D rendering branch.

The following component-removal results are historical evidence for the previous ray renderer.

The first component removal was delayed-color-frame rendering in SunshineGame3D. Its export
already required the current color frame; the unused history textures and copy passes are now
removed. That version rejected legacy `Delay_Frame_Mode=1` requests, including profile defaults,
even with export disabled. Its depth temporal filtering was separate. The original
SuperDepth3D reference retains its own output modes. This removes source complexity, not work
that previously ran in supported full-SBS export.

The external SunshineGame3D milestone became dedicated to full-SBS export and mono game presentation. Legacy anaglyph,
interlaced, checkerboard, frame-sequential, frame-packed, compressed display, VR-distortion and
REST output paths have been removed. That milestone rejected unsupported display opt-ins and
`SUNSHINE_SBS_EXPORT=0`, and overrode `DoubleBuffer_Mode=0`. A later cleanup removed those
compatibility definitions entirely. Current native Game 3D is enabled or disabled in the add-on panel.
At that milestone original ray search,
internal depth filtering, sharpening, HDR and export annotations remained. The separately supplied
original SuperDepth3D reference retains its own final-AA option.

SunshineGame3D no longer exposes or runs final image AA.
The add-on and Home editor both omit this option; old `USE_AA` preset entries are inert.
Packing retains the exact AA-off texel fetch. At that milestone internal ray de-artifacting and
depth-edge filtering remained; the sole Sunshine warp has since replaced them. The AA removal favors fine-detail contrast over the modest shimmer reduction
observed with the final filter; it does not claim to fix disocclusion halos. The installer
no longer requires AXAA for current Game3D sources; explicit original SuperDepth3D reference
installations retain their matching dependency.

The earlier final-AA removal had exact default and stronger Manual renderer comparisons covering static and moving
inputs. Their test-only fixed clock preserves the original adaptation while giving both versions
the same startup and capture sequence. A combined production add-on plus shader 4K PQ comparison
also preserves complete source/depth, mono and final AA0/AA1 output bytes. These bounded regressions
are historical evidence for that removal, not acceptance of later shader changes. They do not
establish halo improvement or a frame-rate gain; most removed modes were already inactive
in full-SBS export. See the [comparison evidence](native-stereo-comparison.md).

Use **Normal Depth View** to check that the selected buffer contains recognizable scene depth.
The exported diagnostic shows prepared scene depth before artistic stereo controls;
it is not a calibrated measurement in meters. Missing depth returns duplicate source eyes.
The integrated output preserves the normal mono game image while exporting a separate stereo texture.

Opening ReShade's overlay keeps the game stereoscopic. The add-on captures the actual controls,
tooltips and cursor and composes the same panel at the same coordinates in both eyes. The panel
has no stereo disparity; the game behind transparent controls retains its own depth. Changes to
add-on settings are visible on subsequent frames, so depth can be adjusted while watching 3D.
Closing the overlay removes the panel without changing the source generation or stream size.
Unrelated shader recompilation and the global effects toggle do not disable native Game 3D.
Disabling Game 3D itself uses the flat desktop fallback. If overlay capture cannot be prepared, the add-on selects that fallback
instead of exporting invisible controls.

The add-on logs concise depth-selection, readiness and export events. The temporary shader-control
polling, per-sample traces and GPU control-echo probe are removed. Histogram sampling continues as
part of automatic depth selection. Readiness means a binding is available; it does not prove that
the buffer contains the game's scene depth.

The source desktop stays at the normal game resolution, including its mouse-coordinate space.
Do not select Raw SBS or manually double the game display width for this setup. The separate
texture already contains the full-resolution left and right eyes.

Normal installation omits `-ShaderDirectory` and uses the GPU renderer embedded in the add-on.
For comparison testing only, the exporter also accepts annotated SuperDepth3D and independent
SunshineDepth3D reference exports. These require an explicit `-ShaderDirectory` with their matching
sources; the installer disables native Game 3D and enables only the selected reference technique.
To return to native rendering, rerun the installer without `-ShaderDirectory`, then enable Game 3D
in the add-on panel if it was previously disabled. Existing native preferences are preserved.

### Dump 3D diagnostics

In Moonlight 3D, **Dump 3D** is available in a stable **Game 3D** session, including its
mono/waiting state. Use it while the game remains foreground on the virtual display. The host,
client and native add-on must all include Game dump support; an older or missing add-on produces
an explicit unavailable report rather than an AI-depth package. Movie 3D does not enable this action.

The add-on snapshots the inputs actually consumed by one native stereo render: original source
color, full-resolution raw depth when available, displacement preparation/conditioning fields,
and packed SBS before the ReShade overlay and host cursor. Unwritten or unavailable fields are
omitted and explained. Frame Generation can pair current color with retained real-frame depth;
the dump records that reuse and original depth identity instead of claiming an exact real-frame pair.
Each capture ID associates the color and depth actually consumed together. A swapchain presentation
or export sequence is not a shared game-frame ID: when the middleware and presented color do not
provide a common frame identity, their temporal correspondence is explicitly unverified. Timestamps
or a matching resolution do not establish that correspondence.

Sunshine publishes a separate `game3d_*` package below its existing `sbs_dump` directory next to
the host log (or `APOLLO_SBS_DUMP` when set). Native texture bytes and JSON descriptors preserve
FP16 HDR and R32 depth without preview normalization. Descriptors identify dimensions, DXGI
format and packed row size. Render metadata includes the applied constants, strength, depth
rectangle, jitter, camera conversion, screen-plane state and capture/export identities.
The package embeds the exact HLSL used by the renderer, the complete 80-byte constant buffer
(including padding), its ABI/layout, compiler flags/defines, and pass/sampler bindings. Depth
is saved as unfiltered R32 samples from the full consumed allocation at mip 0/layer 0, before
crop, jitter compensation, normalization or clamping. Its original allocation/SRV descriptors
remain recorded. The captured SBS and conditioning fields provide the production comparison.

The middleware observations are separate from that render snapshot:

- **Streamline:** observed resource tags, viewport/frame scope, constants and camera/motion
  data, supported ABI/module identity and Frame Generation observations.
- **NGX:** feature identity and available named resource/scalar parameters, including depth,
  motion vectors, jitter and reset evidence. NGX parameters are not Streamline tags.
- Missing values, failed getters, bounded-cache truncation and observation timestamps remain
  explicit. Latest observed metadata does not by itself establish association with selected depth.

UI-related observations require separate interpretation. Streamline's `UIColorAndAlpha` (23)
can provide a UI layer and opacity, while `HUDLessColor` (2) is a scene image without UI, not
a mask. `NoWarpMask` (54) marks pixels to skip middleware warping; its presence does not prove
that those pixels represent UI. Consult the game's actual SDK version, for example the
[Streamline 2.7.30 tag definitions](https://github.com/NVIDIA-RTX/Streamline/blob/v2.7.30/include/sl_core_types.h).
NGX's `TransparencyMask` likewise is not a UI classification. A successful getter returning
null means no resource was supplied by that observation.

The NGX catalog also queries the official DLSS-G inputs `DLSSG.HUDLess`, `DLSSG.UI`,
`DLSSG.UIAlpha` and `DLSSG.Backbuffer` through the owning SDK's D3D12 resource getter.
These are separate from Super Resolution's `Color`/`Output` and transparency hints.
Their [SDK definitions](https://github.com/NVIDIA/DLSS/blob/374959484e79a640feaba44c93ac8cfb0a03f5b5/include/nvsdk_ngx_defs_dlssg.h)
describe HUDless scene color, premultiplied UI color/alpha and single-channel UI opacity;
the three optional UI inputs need not be supplied. Successful capture produces
`ngx_hudless_color`, `ngx_ui_color_alpha`, `ngx_ui_alpha` and `ngx_backbuffer` artifacts.
Each input's own complete subrectangle is retained; all-zero or entirely unavailable
subrectangle fields use the full allocation, while partial or malformed rectangles reject
that optional copy. `DLSSG.ColorBuffersHDR` is recorded but does not alone identify a transfer
function. Copies require observed native resource-state evidence and successful evaluation.
The existing SL-first depth ownership gate remains in force: a successfully admitted SL
capture suppresses nested NGX resource copies. During an armed dump, the outer supported
NGX evaluation still records bounded named-parameter observations, including UI inputs,
without nominating depth, advancing its reset/sequence state or issuing resource-copy callbacks.
These observations report `resource_capture_status = not_attempted_depth_owned` and
`sequence_domain = diagnostic-depth-owned`; known feature creation metadata remains available,
while capture metadata and scene association remain unavailable. Nested NGX wrappers share one
diagnostic observation owner, independently of whether a rejected outer depth input permits an
inner depth fallback. Direct/fallback NGX evaluations query
these keys only during an armed dump, without restricting the feature ID. These resources
are diagnostic inputs; they do not automatically replace the selected live UI mask or prove
pairing with the final rendered color.

NGX query failures retain the numeric result and a `result_detail` containing its hexadecimal
code and known SDK name. `FAIL_UnsupportedParameter` does not distinguish an unset key from
an unsupported parameter or a getter-type mismatch. The dump records the original typed
getter result without retrying optional UI resources through other pointer types.

These inputs are feature-specific. NVIDIA's [Super Resolution helper](https://github.com/NVIDIA/DLSS/blob/374959484e79a640feaba44c93ac8cfb0a03f5b5/include/nvsdk_ngx_helpers_d3d.h#L277-L340)
sets the transparency, current-color-bias, particle and animated-texture entries, including
null values, but does not set the transparency-layer, responsivity or disocclusion entries
queried by this catalog. A failed optional query therefore does not establish an obsolete SDK
or a missing HUD mask. The dump records what the current game evaluation actually exposes.

An explicit dump also requests the catalogued UI-related SL/NGX resources, including HUDless
and UI color, alpha, no-warp, transparency, particle, history-rejection and related hints.
Every catalog entry remains visible in metadata when unobserved, null or unavailable.
SL v2 descriptors accept `eTex2d` and `eUnknown` identities, matching live capture; SL v1 accepts
only `eTex2d`. `eUnknown` still requires native COM, device and texture-description validation;
known buffer and invalid resource types remain unsupported.
Optional copies preserve the full allocation's native bytes and its own rectangle; they do not apply
the selected depth crop or jitter. RGBA8/BGRA8 typeless color allocations are copied into compatible
UNORM snapshots without changing their bytes, including alpha. This normalization is shared by live
UI capture and diagnostics; source diagnostics retain the original allocation format while captured
artifacts report their actual typed format. Other typeless families remain unsupported here because
their numeric interpretation cannot be inferred from storage alone. Unsupported formats, expired resources, readback failures and
capture-budget limits are reported per resource without discarding the main replay inputs.
`ui_resources` contains middleware observations; `optional_captures` describes frozen optional
copy decisions, and host-side failures appear in `optional_capture_errors`. These scopes must not
be treated as interchangeable frame evidence.
Each optional copy's `capture_diagnostic` records its available declared and observed resource
states, state flags, resource cookie, device and command-recording identity. These are existing
capture-admission observations; recording them does not relax capture checks or change GPU work,
geometry or zero placement. Optional Streamline copies and live UI captures select the same
state policy from the unchanged declared lifecycle. For `OnlyValidNow`, a stale declared UAV
hint does not reject the dump when that recording proves a supported shader-resource or
render-target state. Longer-lived tags keep the strict source contract.
The original declaration, observed state and selected copy state remain separately recorded.
A rejected optional copy does not by itself establish that live alpha failed for the same reason:
the optional capture and live snapshot still have separate request timing, source selection and
storage ownership. Compare their provenance rather than assuming an identical frame.
In Expedition 33's earlier 2026-09-19 FG-on diagnostic windows, tags 2 and 54 were nonnull,
and tag 23 was explicitly null. This establishes potential HUDless/no-warp inputs, not an
available UI mask or a match to the exported frame. Before using either input to protect UI,
capture it within its declared resource lifetime and verify frame, extent, color-space and
pixel semantics. Comparing unsynchronized final and HUDless images is not reliable UI separation.

The request uses diagnostic wire v3 with capacity for 40 textures, independent of streaming SBS v2.
Host and add-on must agree on this mapping version; incompatible versions fail explicitly.
The additive on-disk artifact schema remains `sunshine.game3d.dump.v1`.
Optional metadata publication is transactional. If its serialization or mailbox budget fails,
the producer keeps the primary response unchanged and sets the wire `optional_metadata_omitted`
flag; the host records that failure without requiring another byte in the producer JSON.
Only an explicit dump request allocates diagnostic snapshot textures or arms detailed middleware observation.
Main replay snapshots use the rendering queue inside its depth lease. Optional UI resources
are copied at their authenticated SDK call on that call's command list, with the original resource
state restored. They reuse the native capture owner's state, submission and retirement checks,
in a separate 32-slot/256-MiB pool that cannot consume the depth pool. A request keeps the first
recorded observation of each resource kind; later API observations remain independently labeled.
New optional copies stop at the main snapshot boundary. Publication polls producer completion
and recording retirement for at most 250 ms, never inserting a GPU dependency or CPU wait.
An unfinished or rejected optional resource is explained without dropping the main replay inputs.
Each successful copy retains its own frame/viewport/feature identity and rectangle; no match to
the main color frame or transfer function is inferred. The producer publishes immutable
shared resources only after nonblocking GPU completion; the host opens and retains them before
acknowledging, stages readback without waiting for the GPU, and writes files on its publication
worker. The same worker automatically generates human-readable PNGs and an `index.html` guide
before publishing the directory. There is no additional client action, GPU pass or Python runtime
requirement for previews. Game 3D CPU publication is limited to one package process-wide,
including across converter replacement and reconnects; those transitions cannot accumulate
large queued packages. Diagnostic work is not performance evidence.

The human views include full-size `color.png`, `raw_depth.png` and `final_sbs.png` when those inputs
are available, plus a cropped/jitter-aligned depth view, a color/depth alignment overlay, signed
parallax, conditioning changes and a left/right difference view. The HTML guide labels each
mapping and missing input. An eye difference is expected from stereo displacement; it is not an
image-error metric. Alignment overlays help inspect outlines but cannot prove temporal pairing.
The manifest's `visualizations` entry records preview results independently from raw capture
status, so a preview failure cannot discard a valid lossless package.

When the captured `source_color` format has a real alpha component, `source_alpha.png` shows
that existing channel with a fixed 0–1 range, without another GPU copy. Constant zero/one remain
black/white and nonfinite values are magenta. This is raw source alpha, whose UI meaning is not
guaranteed; it may carry interface coverage, constant data or another game-defined value. It
never substitutes SBS output alpha; generating this preview does not enable UI protection. The Python inspector's
`--previews` also writes `previews/source_alpha.png`.

Optional PNGs use their catalog file stems, such as `sl_hudless_color.png`,
`sl_no_warp_mask.png` and `sl_ui_color_alpha.png`. Color resources with a real A channel also
receive an `_alpha.png` view; absent channels are never synthesized. Masks and alpha use a fixed
display range: 0 is black and 1 is white, including constant-zero and constant-one textures.
Additional channels receive separate labeled views. Motion components use a signed full finite
range. NaN/Infinity are magenta. These are full-allocation previews, with no thresholding,
depth-crop alignment or inferred UI classification. The HTML metadata includes empty/unavailable
catalog entries as well as captured resources.
Streamline Backbuffer tag 53 is optional artifact 32 (`sl_backbuffer.bin`, with RGB and alpha
previews). It is copied at the tag-call boundary using the existing resource-lifetime and command
completion tracking. This is candidate evidence for game color before FG, not an authenticated UI
mask or a proven match to the exported color/depth frame. Its provenance is retained independently;
production rendering does not consume this optional dump snapshot. The independently retained
input actually consumed by live UI protection is the required artifact 33 described above.
Optional color uses RGB code values when its transfer is unknown; it does not silently inherit
the final backbuffer's transfer. A declared or explicitly assumed per-capture transfer is labeled,
and any HDR tone mapping remains display-only. Alpha is neither unpremultiplied nor applied to
color without a verified compositing contract.

Color PNGs are 8-bit display previews. SDR retains its encoded color; HDR uses a labeled diagnostic
tone map rather than reproducing headset color processing. Depth and field PNGs label their
display range, direction and invalid pixels; they are not calibrated absolute-depth values. The
native `.bin` files and exact shader constants remain authoritative for numeric evaluation and
replay. A 4K HDR raw snapshot can occupy roughly 380 MiB, with previews adding disk space, so dumps
are explicit one-shot actions, not a continuous recording.

`preview_game3d_dump.exe <package>` generates the same human views for an older package, using
the host's preview implementation. `tools/reshade/inspect_game3d_dump.py <package>` additionally
reports numeric statistics offline (NumPy required); its optional `--previews` writes
artifact PNGs in a `previews` subdirectory (Pillow required).

### UI source discovery and snapshot qualification

The inspector's `ui_discovery` report separates interface observation, native capture,
pixel content, optional offline review and the input actually consumed by the renderer. It reads the
production catalog from `producer_metadata.latest_observations.ui_resources`; older flat
catalogs remain readable. An empty current catalog never inherits a stale flat catalog.
The catalog supplies resource semantics; this report does not infer SDK tag meanings from
numeric IDs or load a game's middleware.

Live discovery follows the automatic order above. For troubleshooting a new game, the offline
inspector investigates these sources without changing live protection:

| Priority | Candidate | Current observed interfaces |
| --- | --- | --- |
| 1 | Explicit UI opacity | Streamline UIAlpha; NGX DLSSG.UIAlpha |
| 2 | Explicit UI color with alpha | Streamline UIColorAndAlpha; NGX DLSSG.UI |
| 3 | Game color alpha | Streamline/NGX pre-FG Backbuffer; presented source alpha |
| 4 | HUDless/final-color comparison for a derived mask | Streamline HUDLessColor; NGX DLSSG.HUDLess |
| Diagnostic only | No-warp, transparency, particle and other hints | Catalogued SL/NGX inputs |

The offline order applies to captured snapshots, not merely nonnull pointers. AMD FSR and
Intel XeSS UI adapters are not implemented by this workflow. An unobserved interface, an
explicit null, a failed getter, a rejected copy, a pending copy and a missing/unreadable artifact
remain distinct outcomes. Depth-owned NGX observations explicitly explain why pixels were not
copied. Optional snapshots retain their own provenance and never replace the required consumed
UI artifact. Neither matching dimensions nor nearby sequence numbers establish correspondence
with the presented color.

When dedicated masks and color alpha are unsuitable, compare native HUDless RGB with final
RGB as the next candidate investigation. Decode both with the same transfer and color-space
handling; independently rendered preview PNGs can create a false scene-wide difference.
Inspect scene residuals, camera motion and HUD alignment before choosing a noise threshold.
A selective difference can locate changed HUD pixels, but does not recover exact compositing
opacity or identify HUD pixels whose color happens to match the scene. Live use additionally
needs a paired capture/presentation identity and checks for scene-wide mismatch. The live
Streamline provider now supplies this guarded difference fallback automatically. Offline
comparison remains useful for investigating its acceptance or rejection.

When automatic discovery fails in a new game, establish hook/feature coverage with the call-route diagnostic
below. The producer translates each raw Streamline tag through the hooked interposer's
[buffer contract](#streamline-buffer-type-contract) before cataloguing it, so catalog rows and
`tag_type` fields use canonical 2.12+ numbers (UIAlpha is 69 even when a 2.11.x game sent 68).
Dump tag rows keep the raw SDK value as `type` beside `canonical_type`. Explicitly
ambiguous/unsupported catalog bindings cannot be qualified.

Capture ordinary gameplay with visible UI, camera movement, hidden UI when available, a menu,
and a return to gameplay. Use the validated evaluation interpreter selected by the repository's
runtime procedure to inspect each existing Dump 3D package:

```powershell
& $SbsbenchPython tools/reshade/inspect_game3d_dump.py '<dump-directory>' --previews `
  --ui-report '<new-report.md>' --write-ui-review '<new-review.json>'
```

The report and review paths must be new files; the inspector never overwrites an existing review.
Without `--previews`, numeric channel facts and discovery still run. The generated review is
intentionally incomplete. Review the fixed-range mask alongside the game image and record
scene context, mask alignment, exclusion of scene content, an explicit verdict and explanatory
notes before supplying it with `--ui-review`:

```powershell
& $SbsbenchPython tools/reshade/inspect_game3d_dump.py '<dump-directory>' `
  --ui-review '<review.json>' --ui-report '<new-reviewed-report.md>'
```

Review binds the manifest, candidate native bytes and source-color evidence by SHA-256.
Changed evidence, missing required channels, failed captures, nonfinite/out-of-range opacity
or contradictory review context prevent qualification. Numeric coverage alone never establishes
UI meaning: empty coverage can be correct without visible UI, and full positive coverage can be
correct for a full-screen menu or can incorrectly cover a 3D scene. All-positive gray values
also cover the entire frame under the protection predicate. A selective pattern is only a
candidate, not a semantic verdict. HUDless color and generic hints cannot become opacity masks
through this review mechanism.

`human_reviewed_snapshot` qualifies only the specific artifact and observed scene. The report's
`selected_snapshot` is an offline recommendation; it neither changes live protection nor proves
frame pairing, continuous availability, moving-UI alignment or generated-frame correctness.
The required consumed artifact and recorded enable state remain separate render facts. Reviewing
several game states can explain automatic decisions and remaining failure cases. Loading or
editing this JSON does not affect live behavior; no review file or in-game approval is required.

The inspector/qualification regression tests run without a game or GPU:

```powershell
& $SbsbenchPython tools/reshade/test_game3d_ui_discovery.py
& $SbsbenchPython tools/reshade/test_inspect_game3d_dump.py
```

### Native frame replay

The add-on build also produces `replay_game3d_dump.exe`. It uses an official ReShade runtime
and the production native renderer, with the package's embedded shader and exact constants:

```powershell
.\replay_game3d_dump.exe '<dump-directory>' '<official-ReShade64.dll>' '<new-output-directory>' --verify
```

It uploads the full source color and depth, then reapplies the saved crop/jitter. It compares
the reconstructed conditioning fields and SBS against the captured production artifacts,
reporting byte equality and numeric errors in `replay.json`. `--verify` returns failure when
errors exceed the reported format tolerance; source color copying must remain byte-exact.
An optional `--strength 30`, `--shader '<experimental.hlsl>'`,
`--depth-gain 500`, `--source-alpha-ui on|off`, `--zero-inverse-depth 0.001`,
`--ui-inverse-depth 0.002`, `--ui-nearest-floor 0.002`, `--ui-front-fraction 0.25`, or
`--ui-plane screen|front-limit|shallow-front` creates a clearly labeled experiment using
the same pass/constant ABI. Gain and zero overrides are independent: moving the zero changes
only `convergence[1] = q0`, preserving the captured gain unless separately overridden.
Historical midpoint UI maps the captured applied UI depth through the overridden geometry; its
target is not recomputed by a single-frame replay. Mode 5 freezes the consumed display fraction
without rerunning adaptive observations or history. Historical shallow-front UI uses one quarter
of the applied positive display limit; historical front-limit UI uses that full limit. All include
strength, blend and warp readiness; gain and zero overrides do not directly position them. A legacy
captured shader needs a compatible `--shader` override to select a mode it does not support.
Gain must be positive and representable; zero must be finite and nonnegative.
Its argument and resulting values are recorded alongside the unchanged captured parameter bytes
in `replay.json`. It does not change the live zero-plane policy. Inputs remain unchanged and each
run needs a fresh output directory. Alternate source must provide the recorded entry points and bindings.
Source-alpha protection defaults to the captured selection (off in older packages); enabling it
requires a shader that consumes `Sunshine_SourceAlphaUI`. Optional API UI snapshots are validated
against the shared resource catalog and limits, then listed as unused by this renderer in the report.
They do not substitute for the original source-color alpha or the required consumed-alpha artifact.

This reproduces one native Game 3D frame before overlay/cursor, host scaling, HDR encoding and
video compression. Saved constants freeze the scene policy at capture time. A single snapshot
does not evaluate temporal adaptation, FG interpolation or flicker; those need a frame sequence.

### Explicit linear-distance depth

Production Game3D carries a depth encoding with each immutable capture and asynchronous sample.
Device depth keeps its existing camera-affine or relative-raw path. Explicit linear distance
uses `z = raw * raw_scale + raw_bias`, then `q = 1/z`; it needs no guessed near plane or camera
matrix. Streamline tag 49 declares this encoding, including its resource precision transform.
NGX's explicit `DLSS.Use.HW.Depth = 0` selects linear distance and `1` selects device depth.

The native shader uses `coordinate_basis = 2`, retaining the 80-byte b0 layout. Its existing
projection pair stores `(-raw_bias/raw_scale, raw_scale)` for the reciprocal decoder.
`#define SUNSHINE_LINEAR_DISTANCE_DEPTH 1` identifies shaders that implement this basis.
The same inverse-distance controller sets scene gain and zero; provider, logical source,
encoding and observation domain isolate calibration across source changes. A linear input
does not imply that camera projection metadata was recovered.

Nonfinite or nonpositive decoded distances produce neutral disparity and do not contribute to
calibration. The existing tiled GPU reduction counts valid samples, and an all-invalid capture
cannot establish or replace a reference. Linear depth has no supplied near-clip bound, so each
pixel's reciprocal and scaled displacement must remain finite. Equivalent world-unit changes
cancel through the shared gain. Device-depth decoding and its endpoint policy remain unchanged.

### Native depth calibration and rendering

The remainder of this section documents the **independent SunshineDepth3D reference renderer**.
Its percentile calibration and rendering equation do not replace SunshineGame3D's source-owned
camera/raw depth preparation. See the [comparison decision record](native-stereo-comparison.md)
for historical milestones and their evidence scope.

For ordinary perspective hardware depth, `d = A + B/z`. Relative stereo disparity is affine
in the same coordinate, so the independent effect needs no near/far plane guesses. The add-on
maintains a raw median anchor and the reciprocal of a robust 10–90% raw span, using three fresh
useful sample captures. Those values stay fixed for that resource lifetime and capture layout;
they do not stretch every newly visible scene to a constant depth range. A source or capture
rectangle change discards its calibration. This initial version does not recover projection
matrices or detect a changed projection within an otherwise unchanged buffer.

Normal/reversed orientation requires three distinct frames of unambiguous exact clear-one or
clear-zero evidence respectively. Mixed and non-endpoint clears remain unknown. This is an
inference from clear behavior, not proof of the camera projection or depth-test function.
The **Depth direction** override handles ambiguous cases. Unknown direction keeps stereo flat.
The histogram log coordinate used to select buffers never becomes the rendering coordinate.
Raw depth stays FP32 through subtraction; standard-Z values near one must not collapse in FP16.

The add-on updates `Sunshine_DepthReady`, `Sunshine_Calibrated`, `Sunshine_RawAnchor`,
`Sunshine_RawGain` and `Sunshine_DepthRect` together after its depth-capture callback and before
the effect runs. A separate per-present proof prevents a missed update from exporting stale
calibration. Current-frame capture readiness and older calibration evidence are distinct.
The raw gain is signed so `nearness = (raw - anchor) * gain` increases toward the viewer.
The rectangle stores normalized `scale.xy, offset.zw` into the captured depth texture.

The shader subtracts Screen plane, bounds this nearness to [-1,1], then multiplies by
`0.005 * clamp(Strength,0,2)` of source width per eye. Strength defaults to 1 and acts after
automatic calibration; zero bypasses geometry exactly. Left-eye foreground moves right;
right-eye foreground moves left. The independently authored backward search traces each destination
ray from near disparity to far disparity. For disparity `q` in source pixels, its source coordinate
is `destination - eye_sign*q`; a hit satisfies `q = disparity(depth(source))`. The first validated
intersection owns the pixel. It does not forward-project source segments or scatter source pixels.

Smooth neighbors whose disparities differ by at most one source pixel are interpolated after
the disparity clamp. Larger discontinuities keep their native pixel footprints. The ray visits every
knot and edge of that reconstructed field, including one-pixel objects. The three subpixel rays
share a traversal of at most 168 source-center pairs at the supported maximum dimensions/strength.
Smooth pairs use one affine interval; cliffs split into two constant half-pixel intervals. Each
new source depth is fetched once and shared by the rays. Within an interval, a residual
bracket is refined with up to six secant landings, each checked against a fresh depth sample with
a 0.002-source-pixel residual tolerance. Depth-edge sign jumps between intervals are recorded as
uncovered gaps rather than accepted as surface intersections. This is the general inverse
ray-search approach used by parallax-occlusion stereo, with our own depth coordinate and code.

Gaps use the farther surface with
zero confidence. To avoid stretching a TAA/upscaling color fringe into a horizontal spike, the
fill donor moves into that background by one active depth-texel footprint plus one source-color
pixel, rounded up and capped at eight source pixels. Each intervening sample must be valid,
no nearer than the original background and at most one pixel farther in disparity; otherwise the search
stops at the last supported sample. The footprint uses the actual depth texture and captured
viewport. Fractional ray samples are snapped to a validated background color-texel center before
the donor search. This changes only uncovered-pixel donors, preserving observed surfaces and thin
foreground. It remains an approximation of unseen background, with no guarantee for wider
temporal-filter fringes or missing depth/color alignment. A background sloping nearer to the
camera can retain the boundary donor rather than borrowing a closer surface.
Color interpolation decodes SDR/PQ before filtering and preserves signed linear scRGB.

The first implementation assumes a perspective depth buffer that aligns with the displayed
color through its viewport rectangle. It does not infer DLSS jitter, frame-generated depth,
custom logarithmic depth, orthographic projection, HUD layers or separate weapon projections.
Matching resolution alone does not establish alignment. Calibration currently requires a sampled
backup; keep the installed depth-preservation setting enabled. Physical game/headset acceptance
is separate from synthetic buffer, shader and transport validation.

### Streamline ABI compatibility

Streamline discovery selects a public ABI family from consistent DLL version metadata, then checks
the exports actually present. It does not maintain a game-name or individual patch-version allowlist.
The public-header survey covers all 28 published tags from **1.0.0 through 2.14.1**. The relevant
Windows x64 boundaries are:

| Public ABI family | Consumed layout and entry points | Pinned official boundary headers |
| --- | --- | --- |
| 1.0.0–1.0.2 | 432-byte Constants; 40-byte Resource without a state field. 1.0.0/1.0.1 use the original `sl::setConstants`, `sl::setTag` and `sl::evaluateFeature` namespace exports; 1.0.2 introduces the `slSet*`/`slEvaluateFeature` C names. | [1.0.0](https://github.com/NVIDIA-RTX/Streamline/blob/999b2087d0858bcec3d9d0113c99cf0d1c8b932e/include/sl.h), [1.0.2](https://github.com/NVIDIA-RTX/Streamline/blob/1da84fff73a02df9d7f7a732b83005e833c21a34/include/sl.h) |
| 1.x from 1.0.3 | Same Constants; 48-byte Resource adds state before the extension pointer. | [1.0.3](https://github.com/NVIDIA-RTX/Streamline/blob/8a9e67be0279f248d04391be7a4dc40546be40b9/include/sl.h), [1.1.1](https://github.com/NVIDIA-RTX/Streamline/blob/5bac43f464f53bc0583bab8df506b788d8d14c3c/include/sl.h) |
| 2.x | Typed/versioned structures; the consumed core function signatures and field offsets remain compatible across the surveyed releases. Constants are 456 bytes, Resource 112, ResourceTag 64 and ViewportHandle 40. | [2.0.0](https://github.com/NVIDIA-RTX/Streamline/blob/f36f47cffef53c59e7da54288689f1525049aa7d/include/sl.h), [2.14.1](https://github.com/NVIDIA-RTX/Streamline/blob/2122257e0fce486f91b385aa63b9a09b0a34b363/include/sl_core_types.h) |

NVIDIA's [versioned-structure contract](https://github.com/NVIDIA-RTX/Streamline/blob/2122257e0fce486f91b385aa63b9a09b0a34b363/include/sl_struct.h#L49-L106)
requires appended fields and a version increment. The observer still validates the structure GUID,
supported structure version, readable prefix and extension interpretation; a compatible DLL major
does not make arbitrary structures or missing resources usable. Missing or contradictory version
metadata and unknown ABI majors remain unsupported. Official 1.0.2–1.1.1 binaries have a resource
script quirk: fixed file/product version 1.0.0.0 but matching strings containing the actual version.
Official 2.0.1 and 2.1.0 have the equivalent fixed 2.0.0.0 quirk. These exceptions are limited to
the surveyed legacy release range and those two exact 2.x releases, with complete, agreeing strings.

Constants, global tags and evaluation exports are required. `slSetTagForFrame`,
`slGetNewFrameToken` and `slGetFeatureFunction` are optional capabilities. Frame tags first appear
in the surveyed 2.7.30 headers; earlier 2.x releases retain global/evaluation-local tag observation.
Without token creation observations the adapter cannot invent numeric frame evidence, and without
feature-function lookup it cannot discover the optional PCL/FG observers. These omissions do not
disable the core hooks. Hogwarts Legacy's **2.6.10** build is not a public SDK tag: its admission is
based on the compatible 2.x family, with game behavior still requiring live validation.

An ABI match is separate from a capturable depth source. Early 1.x resources have no state word;
they require state established by the native command-recording observer. Later legacy resources
can declare a nonzero state. The private resource-state encoding is still enabled only for an
exactly identified **sl.common 1.1.1.0**, independently of the public interposer family.
Unknown, incomplete or conflicting state remains unavailable.

ABI changes are checked with the `reshade_streamline_camera_data`,
`reshade_streamline_camera_probe` and `reshade_streamline_v1_resource_state` CTest cases. The probe
test executable also accepts `--inspect-version <path-to-sl.interposer.dll>` for read-only identity
checks. The opt-in [SDK ABI checker](../tools/reshade/test_streamline_sdk_abi.cpp) compiles external
official headers against the production sizes, offsets and function signatures; supply the SDK
include path and `SUNSHINE_SL_MAJOR`, `SUNSHINE_SL_MINOR`, `SUNSHINE_SL_PATCH` definitions as shown
in that file. Keep original headers unchanged and document any compiler-only portability copies.
Run this comparison at the pinned boundaries when extending a family; it does not load SDK DLLs
or replace a game capture test.

#### Streamline buffer-type contract

Buffer-type numbers are header constants compiled into the game, and they are not stable across
releases. [`streamline_buffer_contract.h`](../tools/reshade/streamline_buffer_contract.h) translates
each raw tag once, at the hook boundary, using the hooked interposer's validated version:

| Interposer header | Declared range | Consumed differences |
| --- | --- | --- |
| 1.0.x / 1.1.x | 0–32 / 0–33 | Tag 23 is `UIHint`, not UIColorAndAlpha; it is not consumed. |
| 2.0–2.2 | 0–37 | No HighResDepth, LinearDepth or Backbuffer. |
| 2.4–2.6 | 0–53 | Adds HighResDepth 48, LinearDepth 49, Backbuffer 53. |
| 2.7–2.10 | 0–66 (0–67 from 2.7.30) | No UIAlpha. |
| 2.11.x | 0–68 | UIAlpha is **68**. |
| 2.12.0 and later | 0–72 | ResponsivityMask is 68; UIAlpha is **69**. |

Raw values above the governing header's range are unknown and ignored. Consumers compare only
canonical 2.12+ numbers, so the depth order, UI candidate order and dump catalog need no
per-version branches. The pinned boundaries are the public NVIDIA-RTX/Streamline headers
`include/sl.h` (1.0.0–2.4.15) and `include/sl_core_types.h` (2.7.2–2.14.1); types were append-only
between them except for the two differences above. A release newer than 2.14.x keeps the 2.12
numbering but is not treated as surveyed, so its UIAlpha stays manual-only until the table is
extended. The version identifies the header the loaded interposer interprets; a game whose DLLs
were downgraded below its compile-time headers can still send a number with a different meaning.
The UI candidate's GPU content checks remain in force for that case. The log line
`Sunshine Streamline: buffer contract` records the selected contract at hook installation.

### Streamline depth selection

Supported D3D12 Streamline integrations identify depth at successful evaluations or supported
enabled-FG tag boundaries. This path defaults on (`[SUNSHINE_DEPTH] StreamlineDepthSource=1`).
Admission is based on the resource contract, not a feature-ID allowlist. Typed 2.x evaluation
inputs can identify the viewport and depth for any feature; opaque or unrelated inputs pass through
without replacing valid evidence. Legacy features without a documented renderer contract require
their exact numeric frame/view tuple and validated depth/camera association. Legacy Reflex remains
a timing-marker contract, not a viewport.
Known SR/RR malformed contracts still revoke their own evidence instead of being ignored.

Capture prefers tag48 (explicit high-resolution device depth), then tag0 (scene/render-resolution
device depth), then tag49 (positive linear view distance), even when the heuristic inventory contains
a larger texture. Unsupported capture of an earlier candidate does not block a capturable later
candidate from the same evaluation. If no tagged texture can be copied, a metadata-only nomination
cannot displace usable NGX or authorize stereo pixels; without another usable API source it remains
explicitly unavailable. Linear depth carries its encoding and precision scale/bias with the resource;
it reconstructs inverse distance directly and does not require a camera matrix.
A successful batch may clear an old depth tag and supply an alternate in either order.
The first clear revokes prior leases; only explicitly supplied entries in that same batch can
advance across its own revision change. Repeated successful null clears are idempotent.
Cached tags/cameras, failed SDK calls and intervening observation losses are never revalidated.
The ordered API tags share one pending-evaluation lifetime. An intermediate failed tag cannot
interrupt that lifetime; after it resolves, a known unsupported capture is not pending GPU work
and cannot authorize holding historical stereo. Metadata nominations prefer their reserved slots
so they do not take the last slot available for a usable tag's texture copy.
A valid manual pin retains priority. The panel
identifies this source as **Provided by game (Streamline)**, and selection logs name the tag type.
The add-on panel shows current readiness and resolution. Its expandable details identify the
source resource and tag. When a current frame is unavailable it shows 2D status and labels retained
metadata **Last valid**, rather than presenting it as current depth. Manual overrides are named
explicitly; a directly supplied resource is not represented by a checked generic buffer row.
**Manual depth selection** is collapsed by default. Its list marks the current ready source
**ACTIVE** and places it first, matching SL/NGX and ReShade by native lifetime identity rather than
address or resolution. API-only textures appear as a read-only active row; historical sources are
never marked active. Checkboxes continue to mean manual override, independently of the active label.
No game middleware is loaded or replaced; the supported public interfaces are described in
[Streamline ABI compatibility](#streamline-abi-compatibility).

Source selection, capture and scale preparation are separate responsibilities. SL/NGX adapters
identify the exact game resource and provide a logical source generation, extent and optional camera
metadata. They do not select an unrelated candidate when pixels are unavailable. Generic ranking
supplies fallback candidates only when no valid API source owns selection; explicit manual pins
retain priority. Resource creation, drawing and queue observation remain common bookkeeping even
while Generic ranking and qualification are suspended.
Optional candidate accounting is separate from those capture facts. Numeric draw/vertex counters
pause on native API capture unless the manual list is visible or activity diagnostics request them.
Generic selection and shared preservation retain counters needed for their actual capture decisions.
Per-clear history is collected only for the visible list. UI demand expires after the list stops
being drawn, including when the overlay closes or changes tabs. Hidden lists do not copy or sort
the inventory. Work-presence, viewport, clear direction, preservation boundaries, resource lifetime
and queue synchronization remain live so new/rotating sources can still use the same capture path.

ReShade's D3D12 inventory and the adapters use one native-resource identity attached to the live
COM object. One allocator also supplies fallback and presentation-assignment IDs. Device cookies
handle ReShade's device proxy; resolution and draw counts never establish object identity. Strong
source references and the current registry entry prevent address reuse from matching a destroyed
source. Identity retention does not require that the capture owner support the resource's format.

A validated, successfully evaluated and actually submitted source nomination establishes API
ownership on the consuming queue independently of display readiness. Between API providers,
readable SL takes priority over NGX; an SL DLL, metadata-only nomination or pending first copy cannot
replace working NGX. Once SL has supplied pixels, a valid successor pending for less than 250 ms
from the last copied source timestamp retains that preference. Failed, unsupported, missing,
expired or revoked SL evidence permits a ready NGX source to take over. The current SL observation
revision is checked before arbitration, so an obsolete native copy cannot hide valid NGX.
Explicit enabled FG retains its separate mandatory SL scope described below. Provider choice is
frozen within one presentation, and each packet retains its own encoding, camera and logical source.

When no alternative API source is usable, unsupported format/state or failed display creation
produces mono while retaining the established API selection. A pending snapshot can use the newest
unconsumed completed snapshot from the same source under the ordering, freshness and interruption
rules below; otherwise it also produces mono. Neither case starts Generic ranking or scene-derived
recalibration. Invalid pointers,
device/extent/lifetime mismatches and an unsuccessful or unsubmitted first evaluation cannot
establish authority. A loaded DLL alone cannot establish it either. Once established, temporary
missing or failed observations retain ownership. Explicit source release, disable or lifecycle
teardown ends it; a silence timeout does not. Disabling DLSS without an observed teardown can
therefore require disabling its source option and restarting to use Generic fallback.

One native D3D12 capture owner serves both API-selected and Generic-selected sources. It owns the
snapshot pool, recording/submission identities, source and snapshot leases, consumer registration
and GPU retirement. The two recording opportunities are a legal ReShade preservation boundary
(before clear/unbind/fullscreen reuse, or an admissible end-of-frame capture) and the API evaluation
entry within the supplied resource lifetime. Both record the full depth plane through the same
allocation/copy code. Generic's former backup allocations are stable presentation textures, filled
from an admitted snapshot; they no longer constitute a separate D3D12 capture-pixel lifetime.
The Direct3D 11 path retains its existing Generic preservation backend; this native owner does not
claim to unify D3D11 and D3D12 synchronization.

API nominations always capture at the middleware boundary and leave Generic capture dormant,
including for tracked depth-stencil resources. Manual/fallback selection resumes Generic capture
before selection. Stable API frames do not toggle that demand or repeatedly clear
Generic capture state. All depth-readiness writers share one per-runtime uniform cache, reflected
again after effects reload and published only when readiness changes.

For each effects pass, the exporter resolves calibration into a value-only frame decision before
publishing shader parameters and Automatic UI status. Export and FG retention consume that same
decision. This separates calibration from publication without combining source selection, capture
readiness and calibration readiness. The camera branch is disabled before resolution and enabled
only after a complete ready decision is applied.

An API nomination snapshots the nominated texture at the middleware call, even when the same
resource is in ReShade's depth-stencil inventory. A generic before-clear copy can contain an earlier
or later partial render pass from that allocation; matching its resource identity does not prove
that it contains the depth evaluated by SL or NGX. API capture therefore does not depend on Generic
draw counts, clear indices or preservation mode. It still requires valid resource lifetime and
observed or supplied native state; missing evidence never authorizes an assumed transition.

Generic capture retains its preservation opportunities. In mode 2 it requires an observed
meaningful-work/clear-or-unbind-or-fullscreen boundary. An end-of-frame copy cannot recover an
observed boundary that was missed: the original may already be cleared or reused. That presentation
remains mono until a fresh valid snapshot exists.

Preservation statistics retain a strong snapshot ticket with its allocation identity. Publishing
it requires the current, active, unambiguous presentation record and a matching live ticket.
Before-clear snapshots require actual producer submission. An end-of-frame snapshot can instead
be consumed later in the same still-open recording, where command order is explicit. Preserved
presentations use the game's existing rendering/presentation ordering contract, as Generic Depth
does; they do not require producer CPU completion or insert queue waits. This is not proof of an
arbitrary independent queue graph or of which intra-frame use of a reused depth buffer matches
final color. Resource identity alone provides neither guarantee.

An independent API-entry snapshot retains the stricter native ordering checks. Its same-queue
consumer uses queue order. A different consumer queue on the same device requires successful
producer submission and Reset/destruction to retire the producer recording. A still-replayable
recording could overwrite the copy again; completion alone does not make it immutable. If the
immutable copy is still in flight, source metadata remains selected but its pixels are unavailable
until its private producer fence completes. The add-on never inserts a foreign-queue Wait: the
producer can already depend on future consumer work, so a reverse wait could deadlock the game.
Flushing an immediate command list cannot prove that the application's queue graph is acyclic.
A failed ordering check declines the copy before any read is recorded. There is no CPU depth-fence
wait or pending-copy flush. Device removal is not completion, and actual GPU
completion is still required for allocation retirement. The selected capture stays frozen across
the pass. While a same-source successor is pending, SL and NGX may advance to the newest unconsumed
completed snapshot. Its original sequence, timestamp, projection, jitter and feedback travel with
its pixels. Provider, epoch, logical source, viewport, FG role, observation/reset revision and known
depth layout must match; reset, failed/missing/ambiguous observations and expiry revoke historical
eligibility. The pool retains one eligible completed snapshot while recording its successor, so
CPU nomination cannot continually hide or overwrite the newest usable pixels. This selection does
not repeatedly reuse an already consumed snapshot or prove exact color/depth frame correspondence.
These ordering proofs enter the same capture owner; choosing SL, NGX or Generic does not create a
separate allocation or retirement implementation.

Both opportunities use the same consumer-lease and depth-plane copy operation to fill stable
presentation storage. A snapshot binds to one consuming queue because reads temporarily transition
its state. Producer and consumer retirement retain their own queue fences; values from different
queues are never compared. Strong CPU tickets prevent reuse while frame/command statistics still
refer to a snapshot, and GPU retirement independently prevents reuse while commands can access it.
Repeated unconsumed preservation copies of one source within one open recording may reuse an
allocation with a new ticket ID; the replaced ID is no longer admissible. No raw snapshot handle
alone authorizes a read. Sampling uses the stable display allocation on its consumer queue.
Submission and evaluation completion remain separate facts and may arrive in either order;
neither an unfinished evaluation nor a replay is a fresh successful source.
Every native boundary queries the exact command-list, queue or resource interface and uses the
returned interface before reading method slots or calling native operations. An opaque middleware
pointer or a valid base COM object does not prove the expected interface. Hook discovery must
reject a device before inspecting command-list slots: its slot 9 creates command allocators and
has a different signature from command-list Close. Unknown submission cookies never match
retired recording slots.
Command recording state belongs to the actual native command list through COM private data,
not a fixed global table. Native-only lists discovered through shared method hooks therefore
cannot exhaust a 128-entry registry or leave entries behind when ReShade misses destruction.
Reset changes the recording identity and clears its state; closed lists stay closed until Reset.
A generation counter invalidates earlier recordings when observations are lost. Per-list lifetime
tokens retire producer/consumer identities on native destruction without taking a callback lock;
shared native snapshot slots still require actual GPU fence completion before reuse. Metadata-only
nominations have no GPU allocation to retire and add no per-submission fence.
A submitted source frame is consumed only after its snapshot is successfully copied for effects.
Acquisition of metadata or an unsuccessful copy does not consume it; a subsequent presentation can
retry the same still-current snapshot. Repeated effects within one presentation may reuse it.
Streamline frame numbers are never interpreted as ReShade present numbers.

The game owns the original textures. Adapter records retain source leases; the shared capture
owner owns copied textures and their GPU retirement obligations. Runtime presentation textures and
the exporter SBS ring have separate owners. Dropping one lease never substitutes for GPU completion.
One add-on session owns exporter/UI state and explicit teardown. It relinquishes active ownership
before cleanup so duplicate or reentrant unload cannot dispose the publisher twice. Reinitialization
remains blocked until that cleanup and publisher destruction finish. Adapter source
leases are detached under their metadata lock and released after unlocking. GPU snapshot leases
remain subject to the existing completion/recording-retirement rules.
Retained middleware hooks can keep the module loaded until process exit. The session storage is
process-persistent, preventing a second implicit CRT cleanup owner. A shared detach fence stops
callback observation before other C++ owners are destroyed; vendor calls still forward normally.
Ordinary add-on unload continues to unregister callbacks and release its active session resources.

Streamline 2.x has a synchronous depth tag adapter independent of Frame Generation. Ordinary
`OnlyValidNow` depth can be copied on the command list supplied to a successful tag call even if
FG options and `slEvaluateFeature` have never been observed. That call's declared resource state,
resource identity, active rectangle and lifetime remain the copy contract; it does not provide
state evidence for a different NGX command list. Constants and depth tags are associated within
their own viewport/frame scope. Observed successful enabled FG options assign the separate FG
source role; a tag boundary alone never implies FG.
The consumed `DLSSGOptions` mode/count prefix is validated for structure versions 1–5. Off, On,
Auto and Dynamic modes are recognized; Auto/Dynamic describe an enabled policy, not proof that
a particular presentation was generated. The requested count is diagnostic metadata, not an
observed fixed frame-generation multiplier.
The observer intercepts both newly returned and previously cached FG options function pointers.
Resolving the public function or loading middleware does not enable FG; a successful game options
call is still required. Global `OnlyValidNow` tags may pair with the latest successful constants
call for that viewport when a frame-token creation was not observed. This bounded association
uses the constants-call identity, ordering, observation generation and age, never an invented
numeric frame index. Newer constants, invalid/reset camera data or lost observations invalidate it.
An ordinary synchronous tag without a current camera identity still supplies raw depth, with no
projection, direction, jitter or SDK-frame claim borrowed from older constants or another API.
The existing raw-depth calibration and orientation requirements continue to apply. This fallback
is limited to the explicit `OnlyValidNow` resource in that call; neighboring longer-lived tags
retain their original evaluation/presentation lifetimes. Known enabled FG retains its stricter
frame/camera association and mandatory source priority.
Within the compatible 2.x Resource layout, the observed all-zero nested Resource header is also supported.
Unknown nonzero headers and extensions remain rejected; actual native resource/device/description
validation still applies. A zero state in this uninitialized header form requires observed native
state rather than being assumed to declare COMMON.
For volatile `OnlyValidNow` depth, a copy is recorded on the supplied command list during the tag
call, through the same snapshot owner as all other sources. Normal GPU ordering completes it;
there is no CPU wait. It cannot use a later Generic preservation opportunity.
The shared snapshot owner also orders its injected source read before restoring a shader-read or
COMMON state: it transitions through COPY_DEST and then restores the exact original state,
without writing the source. Full-image asynchronous copy/overwrite tests exposed partial later
writes with the direct read-only restoration, including in the frozen previous add-on. Original
write states already provide that ordering boundary; original COPY_SOURCE stays COPY_SOURCE.
This is a bounded GPU ordering measure, not a claim that a particular game or driver is at fault.
An independent FG Present cannot expire this synchronous tag-call lifetime. Other lifetime rules,
recording retirement and the shared consumer-ordering checks above still apply.
Known affine `PrecisionInfo` extensions are decoded. For device depth their scale/bias are reflected
in the texture-space projection coefficients and valid stored-depth range. For tag49 they recover
positive view distance before reciprocal-depth reconstruction. Changing only the storage encoding
cannot change the reconstructed geometry or its adaptive stereo gain.
Unknown or malformed extensions remain unavailable with a
bounded rejection diagnostic. A supported enabled FG source has priority over NGX,
including while its current capture is pending. Its depth and source-associated camera metadata
stay together; missing usable FG pixels do not authorize an NGX substitution or cross-API
projection borrowing. An observed FG Off retires that source so independently captured ordinary
SL tag/evaluation or NGX depth can resume with newly submitted pixels. A failed or ambiguous FG-options observation also revokes the old FG scope
and its retained display pixels, but does not confirm that FG is Off. Selection excludes those
unconfirmed FG inputs and permits independently validated ordinary SL or NGX depth, with the
selected capture's own encoding and camera. A busy query preserves a confirmed policy or this
revoked-scope fallback policy; it cannot restore the old FG scope. A subsequent successful enabled
observation restores mandatory FG selection. Older captures from the
previous ownership interval cannot be resurrected. While FG is enabled for a viewport, ordinary
evaluation adapters do not also nominate that same viewport; FG's first attempt supersedes only
its old ordinary evaluation observation. Other viewports remain independently ambiguous. No cross-API
projection borrowing is needed.

When an explicitly enabled FG viewport presents again without a new source frame, or a valid
new capture from the same logical FG source is awaiting registration/tag-call result/submission/immutability,
the provider may reuse the last real depth already copied into its private display texture.
The completed-snapshot selection described above applies to both SR and FG. Reusing the already
consumed depth in the private display texture is separate. NGX can also hold that display copy
for exactly one subsequent native presentation when its newer successful snapshot is submitted,
and its producer recording retirement or GPU completion remains pending. This reads only the
previous private display copy; copying the new snapshot still requires both conditions to finish.
The capture selector must confirm that its newest completed snapshot is precisely the consumed capture; source,
feedback/reset, projection, layout and consumer identity must still match. A second missing
native presentation stays mono. Both paths obey the same age bound below.
Explicit source interruptions advance the existing admission watermark,
preventing pre-interruption captures from returning.
Reused real depth retains its applied gain, zero plane, crop and jitter and does not advance
range learning or gain adaptation. Reducing user strength, including setting zero, applies
immediately; increasing strength waits for fresh real depth so reused geometry cannot acquire
more disparity without a fresh placement decision.
Every pass renders the new effects-input color and publishes a new SBS image; it no longer holds
the previous complete SBS image. This is deliberately approximate temporal matching, not generated
depth or proof of real/generated presentation identity. Fast object/camera motion can produce edge
distortion. No extra game-resource copy, shader, GPU queue or wait is introduced for reuse.
Interpolating depth for FG 2x from motion vectors and two real depth frames is future work.
The current implementation does not synthesize depth for generated frames.

Nomination, admission of completed snapshots and display-depth reuse share one freshness limit:
less than 250 ms from the original depth capture timestamp, defined by
`sunshine_scene_depth::maximum_source_age_ms`. FG display-depth reuse requires the same
logical source/epoch/viewport, runtime queue and display allocation, unchanged color dimensions
and any known depth dimensions/crop, and enabled FG. Failed/missing-depth attempts, failed consumer
copies, observed camera reset/lost observations, FG Off, focus/technique/lifecycle changes and
expiry invalidate reusable history until a fresh copy succeeds. A nomination gap cannot prove an
as-yet unknown depth layout; its same-source reuse remains subject to the short bound and color
dimensions. Source pointers may rotate normally without invalidating the private copied depth.
An SL camera reset breaks temporal continuity, not the validity of the new frame's depth or
projection. Fresh tags associated with that reset frame may provide depth and valid camera
coefficients after the old observation revision is revoked. Pre-reset tags and captures remain
ineligible. The reset feedback discards old scene measurements while retaining established gain
and zero placement; it does not force a valid current depth frame to render mono.
Reused frames preserve the real capture's sequence/timestamp/projection, skip new depth readback
and calibration updates, and keep the last matching resolved scene parameters. The strength slider
still applies. Neither reuse nor a new pending nomination extends the age limit. The overlay marks
`Previous real frame (FG)` instead of presenting reused depth as freshly captured.

The capture owner publishes an in-progress nomination with its evaluation watermark. A Present
between this publication and snapshot registration can identify the same established FG source
without acquiring pending pixels. Missing/invalid attempts withdraw that intent; an older call
returning cannot clear a newer intent. The explicit bounded reuse policy above may use its private
previous depth during this gap; GPU resource retirement is unchanged.

The shared capture pool also retains the current valid nominated record after its GPU work has
retired. Pixel-backed and metadata-only records carry the same source authority: an unrelated
NGX/Streamline capture or Generic preservation must not recycle the current record and fabricate
a missing nomination. The same pool retains the existing eligible completed snapshot beneath a
valid current nomination while its successor is pending, including against allocations by another
provider. Supersession, failure or invalidation releases the current record's authority hold;
completed-fallback eligibility and all GPU retirement requirements still apply before storage can
be reused. This uses the existing bounded pool and does not extend capture freshness or authorize
reading expired depth.

Capture acquisition returns source authority separately from the packet's pixel readiness.
Its typed decision carries the selected identity, valid repeated/pending FG continuity, and
the eligible consumed predecessor for an NGX pending successor. The capture owner alone
classifies SDK success, invalidation, recording retirement and producer completion.
`depth_cache_update.h` turns those facts and the last successful copy's value description into
one source decision: `copy_fresh`, `hold`, or `invalidate`. It owns source identity, FG mode,
NGX predecessor/presentation limits, depth layout, observation revision and source-age policy.
One sampled SL observation revision is used coherently for the current and retained checks.

`display_depth_cache.h` owns the sole reusable private depth frame. `copy_fresh` permits a copy
attempt; only successful copying commits that frame and its capture identity, original timestamp,
projection and crop. A failed attempt clears reuse eligibility and cannot fall back to a hold.
`hold` verifies the cache's local color size, immediate command list, queue, texture/view and age,
then copies the saved description for this presentation without changing the saved frame or age.
`invalidate` clears the cache; restoring a source or receiving another pending nomination cannot
revive it. A new successful copy is required. Lifecycle invalidations use that same cache operation.
Calibration/scene placement still belongs to the renderer and retains the existing real-capture
pairing; cache loss does not itself reset learned geometry. Historical UI status is not authority
to reuse pixels. The provider dispatches these actions rather than recombining continuity flags.
Capture diagnostics remain observations, never inputs to reuse authorization.

Pool reclamation composes two independent checks: whether the record is still logically
required by source selection, and whether its GPU storage has retired. Metadata-only records
retain their existing no-GPU-storage semantics. Neither a GPU fence completing nor logical
supersession by itself permits recycling a pixel-backed slot.

While FG is active, `Sunshine SBS FG output` reports cumulative per-runtime publication counters
at most once every five seconds. `published_fresh_depth`, `published_reused_depth` and
`published_depth_missing` distinguish actual shared-ring publications using fresh, reused or
unavailable depth. Their disposition is retained with the pending export copy. These classify
depth availability, not stereo pixels: diagnostics, zero strength and calibration can change the
rendered image. Runtime reload/destruction resets these counters.

`Sunshine depth readiness: lost/recovered` records the first availability transition of a
bounded diagnostic episode independently of the one-second status-log gate. At most four loss
episodes per second are admitted, each with a paired recovery; suppressed episodes are counted.
The record freezes source and newest-view ages at selection, the prior retained depth identity
before invalidation, FG scope, reset/observation revisions, copy outcome, source action/reason and
the cache's final reason. A zero timestamp is unavailable (reported age `UINT64_MAX`), not fresh
data. An untested current observation is `-1`; a zero check revision means no such check ran.
Explicit lifecycle/reuse invalidation clears
are also identified. This is read-only evidence: it adds no GPU pass, readback, wait, or extension
of depth freshness, and does not change source selection or stereo placement.

Each admitted readiness episode also logs `Sunshine depth observation evidence`. Its
`current_check` and `retained_check` query the exact Streamline observation revisions read by
those decisions. `sampled_only` is an additional contemporaneous revision sample, useful when
no decision check ran; it is not proof of what caused the loss. Recovery repeats the frozen
first-loss evidence, rather than attributing the interruption to a later callback.
The observation owner keeps a fixed 64-entry diagnostic journal of revision changes, including
the reason, call site, tick, thread, call sequence, and known viewport, feature, SDK result and
camera reset. It operates with the lightweight source observer even when `StreamlineCameraProbe`
is disabled. Publication and lookup each try one independent slot lock without waiting;
contention, overwritten entries and an in-flight publication report `found=0` / `unavailable`.
They never substitute another revision, alter the production loss counter or authorize reuse.
Unknown viewport, feature and reset values are `UINT32_MAX`; `sdk_known=0` means no SDK result
was observed at that point. Interpret the raw result using the observed SDK ABI: v1 returns a
Boolean, while v2 returns a result code. A recorded reason explains that revision increment,
not necessarily the whole availability episode when several increments or other rejection gates intervened.

The add-on panel separately reports **Fresh / Previous / Unavailable** percentages for the last
five completed one-second buckets of observed provider presentations. It updates once per second,
deduplicates multiple effects passes within a presentation and excludes the current incomplete
bucket. Startup reports the shorter completed interval while collecting the full window. Fresh
means newly copied depth, Previous means reused real depth, and Unavailable means no usable depth;
these are not classifications of rendered versus generated color frames or counts of SBS
publications. Source/FG-mode, focus and lifecycle changes reset the window. The source's stable
ACTIVE label identifies ownership; these percentages describe how often its pixels were usable.

Current capture validation uses the production-owner `reshade_depth_queue_cycle_test` default,
`--pipeline`, `--foreign-reclaim` and `--content` cases plus `reshade_game3d_native_provider_runtime_test` with its
optional SL FG 2x metadata interposer and zero installed FX techniques. Their roles and commands
are documented in [source and submission validation](../tools/reshade/README.md#source-and-submission-validation).
They check capture timing and native lifecycle independently; neither proves a real game's
generated-color/depth correspondence.

Historical effect-based FG validation uses
`SUNSHINE_NGX_FRAME_GENERATION_TEST=1`, `SUNSHINE_FG_LIVE_COMPAT_TEST=1`,
`SUNSHINE_NGX_CROSS_QUEUE_TEST=1`, `SUNSHINE_NGX_CROSS_QUEUE_COMPLETED_TEST=1`, and
`SUNSHINE_FG_PENDING_PRODUCER_TEST=1`. The completed-producer option applies to warmup; eight
gated cases deliberately keep the producer unfinished until effect observation. They require
explicit previous-depth reuse and fresh current-color HDR output within the age bound, then mono
after expiry, without CPU blocking. Completed recovery checks byte-exact depth against an
independent producer copy, whose every pixel is also checked against the generated scene. The test includes
unsubmitted/replayable recordings, overlapping original SDK calls, success/failure/missing tags,
FG Off/On and expiry. Run at normal and 4K output resolution. The normal CPU/CTest suite or a GPU
run that only completes producers before presentation does not cover this contract. These legacy
FX fixtures do not observe the current native renderer lifecycle reliably and are not current
acceptance gates; retain them for historical comparisons. Diagnostic
oracle-only, fresh-allocator and contiguous-copy variants isolate faults and are not substitutes
for the default source-state regression. `SUNSHINE_FG_SUSTAINED_PRODUCER_TEST=1` adds a separate
continuous pipeline regression: sixteen consecutive pending newest captures while each previous
real capture has completed before its successor is recorded, without intervening settled warmups.
It checks exact previous-depth pixels, advancing real capture identity, current-color stereo,
stable calibration, and continued readiness beyond a single reuse lifetime. None of these fixtures establishes a real game's FG
intermediate-color correspondence.

Also enable `SUNSHINE_FG_PUBLICATION_GAP_TEST=1` to pause nomination before a capture slot exists.
This checks actual exported Normal Depth pixels, both-eye equality and previous-depth/current-color
publication through the gap, then fresh-depth recovery, failure, expiry and FG Off. A control binary
needs the identical test-only pause seam; a missing test export is a setup failure, not evidence
of the production bug. `SUNSHINE_FG_SCALE_UI_TEST=1` additionally compares the panel's scale with
the shader's applied coefficient, checks its stability through FG holds, and verifies the change
from camera-depth adaptive gain to raw-depth adaptive gain after FG is disabled.

Version-specific adapters translate to one plain scene-depth contract: provider and logical source
identity, optional validated projection, depth direction, resource extent, optional state hints and
capture lifetime. The capture owner resolves missing hints from its actual observed command state;
an adapter need not claim that a usable state was observed. A future AMD adapter should emit this
same contract; no AMD integration is implemented. The shared scale controller consumes admitted
frames and immutable samples, independently of source-selection and allocation decisions.
Camera projection is optional: invalid matrices never block an otherwise valid source or become
scale coefficients. Without projection, a known depth direction allows the shared adaptive raw
controller to follow the provider's logical generation/viewport. Direction comes from the API when
supplied, otherwise from shared preservation's established clear-value evidence; unknown stays mono.

The direct provider supports D3D12 full-resource, single-sample
device depth and positive linear view distance with UntilEvaluate/UntilPresent lifetimes; it accepts bounded active rectangles while
copying the full allocation's depth plane. Evaluation-bound native capture declines OnlyValidNow;
the synchronous v2 tag boundary supports it with or without FG. Native capture still declines
unknown depth encodings, enhanced/unknown resource states and ambiguous viewports. For V1.1.1, an
explicit nonzero tag state takes priority. If omitted, its adapter reads the provider's resource
state at evaluation entry, just as that version's DLSS implementation does. This private,
version-pinned encoding is enabled only after validating the loaded `sl.common` version too.
It supports depth-only and packed depth/stencil textures: the pinned DLSS implementation asserts
its cached input state for all subresources, and capture uses that same provider contract while
transitioning, copying and restoring only depth plane 0. Stencil is neither copied nor modified;
malformed or missing metadata still requires observed nonzero state on the actual command
recording. An observed conflicting or incomplete transition always rejects capture. A named alias
or split barrier blocks the affected resource for that recording until Reset; unrelated resources
remain eligible. Per-resource state storage grows with the actual command recording and reuses its
allocation on Reset; unrelated resources cannot invalidate depth by exceeding a fixed entry count.
Wildcard aliases or an actual state-storage allocation failure conservatively block the whole
recording. Already-owned copies retain their normal lifetime checks. No state is guessed from texture type.
The [public ABI families](#streamline-abi-compatibility) share these capture checks; broad version
admission does not broaden the private 1.1.1 resource-state encoding. Resource identity remains
attached to the native resource when temporary metadata expires, so rotating buffers retain their
observed state. Failed acquisition logs describe the rejection without presenting an empty output
packet as evidence of absent depth or camera metadata.
Failed V1 state lookups additionally report the actual resource format/shape and the private
property result, size and encoded value at most once per five seconds. This distinguishes a
supported module from a usable state lookup without enabling per-draw diagnostic tracking.
Provider handoff logs distinguish native capture readiness from shader-facing readiness. They
report display-texture/view creation and consumer-command admission separately, and report
projection availability from the captured metadata. A ready native packet does not imply that
ReShade accepted its pixels for the current effect pass.
Failed capture logs retain the first invalidation cause and a coherent snapshot of its evaluation,
submission and fence facts. Observer counters separately identify barrier/submission snapshot
failures and discovery contention. These bounded diagnostics distinguish observation loss, actual
evaluation failure and recording/queue lifetime failures without changing rendering admission.
Native barrier and submission batches have no fixed-count admission cutoff. Small snapshots use
inline storage; larger snapshots allocate invocation-owned storage before calling the original.
The application still receives its original single call, count and arguments. Post-call observation
uses every captured barrier/command identity in order, and nested forwarding reports only the
innermost observed call. Unreadable input or failed snapshot allocation still invalidates evidence;
the overflow counters describe allocation/size failure rather than exceeding inline capacity.

Native command observation is scoped to the authenticated COM interface's vtable slot.
Implementation addresses are not unique method identities: Streamline can share one forwarding
function between command-list and device methods with incompatible signatures. Discovery validates
the exact interface and retains module-owned table storage; installation replaces only that slot
and readiness requires that it still contains our hook. The same slot mechanism serves optional
discard diagnostics. Disabled hooks retain a callable original and pass through for process life.

In **Automatic**, stored depth `r` is decoded as `d=r*raw_scale+raw_bias`, then the frame-bound
projection supplies `q=(d-A)/B`. The identity transform is used without `PrecisionInfo`.
Camera reconstruction and scene placement have separate owners. The shared
`tools/reshade/scene_gain.h` policy initializes an independent gain `K` from the nearest depth
and zero `q0` from the contrast-midpoint trial described below, then tracks that target with a
parallax-based speed limit.
The raw fallback shares this policy on its oriented coordinate, with independent `H`
and `t0`. The [gain and zero-plane contract](#experimental-raw-depth-automation) below owns
initialization, gradual gain tracking, rendered parallax limits, and sample timing.
There is no fixed near-plane multiplier or percentile clipping.
The projection and inverse must pass the perspective, inverse, direction and finite-domain checks.
Disagreement with redundant scalar near/far/FOV/aspect metadata is reported as
`valid-matrix-scalar-mismatch`; it does not rewrite or reject an internally consistent matrix pair.
Scalar distance units and horizontal/vertical FOV conventions are not guessed.
The native-depth signed input field is `0.05*K*(q0-q)`.
The 0–100 strength slider multiplies that field once with
`clamp(Depth_Adjustment,0,100)/100`. This restores the original maximum of 1 while retaining
linear response and exact zero; the experimental 5x presentation-strength expansion is removed.
The `0.05` reference convergence in the field above is unchanged.
`K` and `H` are retained internal names for the independent stereo gain; neither is derived
from the current zero. Eye direction is supplied by the Sunshine inverse warp.
The adapter bounds the strength-scaled field to `[-1.5,2.5]`,
negates it and converts it to source-UV displacement using `(source_height/2160*100)/source_width`,
then applies the Host displacement container and conditioning described above. A separate final
Game3D limit bounds per-eye parallax to 1% of source-image width at full strength, scaled by
the strength fraction and stereo reentry. It limits displacement, not decoded scene depth.
Multiplying every distance by the same unit factor preserves the result when the same gain
reference and zero are represented in those units. Moving the zero preserves separation between
fixed depths in the unclamped field while gain remains unchanged; the full-image maximum updates
the gain target with bounded adaptation in both directions. Per-pixel saturation bounds transient
parallax before that target is reached. Current `A/B` always reconstruct current pixels exactly
and are never interpolated. The whole decoded
domain and current zero plane must fit shader storage; otherwise the frame stays mono without
clamping scene depth to force a supported encoding.
One controller follows the logical viewport across rotating physical resources. Its range
readbacks retain their own projection. A genuine depth-domain reset automatically collects a
fresh reference and screen plane. Queued old-domain observations cannot reinstall the previous plane.
Missing range measurements hold established controls. Missing current depth normally returns
current-color mono, except for bounded FG reuse and the one-presentation confirmed-pending NGX hold below. Without a valid source-associated
matrix, the relative raw path used by Generic also supplies gain and zero placement for NGX.

For matched non-flat ranges, a positive multiplicative change of depth coordinate changes gain and zero
together so that the signed field remains unchanged. This does not recover physical distance
from raw fallback, and separate initialization histories or an unknown nonlinear encoding need
not match. The normal UI labels the fallback **relative depth (assumed infinite
far plane)**; it does not imply recovered distance. Automatic uses no per-game presets or hidden
percentile calibration. Camera and raw API controllers also retain separate
reference histories. They share one reference policy, but switching to a branch with a different
history is not guaranteed to be jump-free. Generic physical buffers retain separate references
unless their encoding equivalence is established; matching resolution alone is not such proof.
Switching numeric bases restarts the existing 0.5-second stereo reentry; it never blends old
pixels or unrelated camera coefficients.

Scene gain is an artistic choice bounded by renderer limits, not recovered meters, a physical
stereo baseline or a validated comfort standard.
Changing only world units or resource storage preserves normalized depth and disparity when
the same reference is represented in those units. Streamline does not supply a
game-units-to-meters conversion. Associated jitter corrects the reported render-sample offset;
direct capture still cannot prove alignment after arbitrary later color transforms. HDR, overlay
and mono/export behavior are independent of the sole Sunshine warp. Manual per-game depth interpretation is removed.
The explicitly enabled FG repeat case above reuses bounded real depth with current color instead of publishing mono.
Set both `StreamlineDepthSource=0` and `NGXDepthSource=0`, then restart to use the generic detector
and its raw-scene policy.

### Depth jitter registration

Valid Streamline or NGX TAA jitter travels with the immutable real-depth capture, independently
of whether that frame also has usable camera coefficients. The offset is measured in active
render pixels, not output pixels or allocation pixels. For render jitter `(jx,jy)`, render extent
`(Wr,Hr)`, depth crop `(Wc,Hc)` and depth allocation `(Wa,Ha)`, the shader's depth-texture UV
correction is `(jx/Wr * Wc/Wa, jy/Hr * Hc/Ha)`. It is added after crop mapping and before the
existing active-crop half-texel clamp. High-resolution SL depth uses the associated same-frame
render extent when supplied, rather than assuming its larger allocation is the jitter domain.

FG reuse retains the jitter belonging to the reused real-depth pixels. A newer pending source
cannot replace that metadata. Missing, non-finite or unsupported jitter/layout metadata produces
zero correction without rejecting otherwise usable depth. A subsequent frame with no jitter
clears the correction; it does not retain an earlier frame's offset.

This adds arithmetic to the existing depth fetch, with no new texture copy, shader pass or GPU
wait. It does not interpolate generated-frame depth or reconstruct background hidden behind an
object. Wide foreground halos in deep scenes remain a live quality issue; jitter correction is
not evidence that those halos are fixed. Halo comparisons need matched actual eye separation,
since equal strength-slider values can still yield different separation after gain initialization
or a required gain reduction.

### Direct NGX depth selection

`[SUNSHINE_DEPTH] NGXDepthSource=1` defaults on for compatible D3D12 NGX integrations, including
Super Resolution and Ray Reconstruction. Exact exported Create/Evaluate/Release and typed parameter getter functions
are discovered in already loaded modules, including a linked game executable or plugin, without
a module-filename allowlist. The NGX SDK declares its API `dllexport`, so a game or engine plugin
that links it directly exports these symbols. The driver's `_nvngx.dll` core and DLSS snippets
export only Create/Evaluate/Release, so calls Streamline makes through them stay with the
Streamline adapter. Discovery also runs
at device initialization, before ordinary feature creation. After that it rescans only when the
loader reports a newly loaded module (at most every 250 ms), not on a fixed timer. Without loader
notifications the one-second rescan remains. Discovery runs on a thread-pool thread, and each scan
enables all of its new hooks in one queued MinHook apply. Every MinHook enable/apply first takes
a system-wide thread snapshot: about 50 ms with ~10,000 system threads, before any thread is
suspended. Hogwarts Legacy exposes 38 NGX entry points, and enabling them one by one on the present
thread froze the game for 2.06 s at startup. The export scan itself is about 2 ms. Reports wait
for a pending scan, so coverage is never reported from its pre-scan state. Streamline's own
hooks (the interposer entry points in one queued apply, and each Frame Generation options and PCL
marker target as it is registered) are installed the same way: a pool thread holding the probe's
poll lock runs them, while the present-path poll skips for those few frames. In Hogwarts Legacy this
removed the last start-up stalls of 160 ms and 106 ms. The per-frame vtable-slot observers patch
no code and stay on the present path. No game DLL is replaced, no unknown
C++ parameter vtable is interpreted, and no feature is inferred merely from installed files.
There is no DLL patch-version or feature-ID allowlist. Every observed successful feature creation
establishes its handle generation; capture additionally requires valid render dimensions, create
flags and depth encoding. `DLSS.Use.HW.Depth` explicitly selects device depth or positive linear
view distance. Standard Super Resolution's documented device-depth default is accepted when that
parameter is absent; other features require the explicit encoding. Unknown ABI contracts, missing
getters or opaque input are not interpreted speculatively.
If creation was missed (late hook discovery, or an add-on reinitialization that cleared the
registry), the first evaluation registers the live handle from its own parameter map. NGX's create
helpers write `Width`, `Height` and `DLSS.Feature.Create.Flags` into the map that later
evaluations reuse. Recovery requires those values plus a valid explicit encoding or, when absent,
applies the SR device-depth default. Ray Reconstruction's helpers always write the explicit key,
and Frame Generation has no `Depth` input. A map without creation values stays unknown, and a
handle released in the current epoch is never revived without a new successful creation. The
five-second `Sunshine NGX depth` line reports `recovered_features`.
D3D11 NGX interception remains diagnostic-only.

At evaluation entry the adapter reads the exact `Depth` resource and explicit render/depth subrect.
NGX supplies no per-call resource state, but its SDK fixes one: NVIDIA's DLSS Programming Guide,
section 3.4 "Resource States", requires every D3D12 input to be
`D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE` at the evaluation call and states that DLSS
restores it afterwards. The adapter therefore nominates with `state_proof::sdk_contract`. A known,
unblocked state observed on the same recording takes precedence. An observed COMMON/zero,
blocked, split or unknown state still rejects capture. Only a recording with no entry for the
resource uses the contract state, and a resource created with `DENY_SHADER_RESOURCE` is rejected.
Engines commonly record that transition on an earlier command list. Hogwarts Legacy logged
`missing_state` for all 27,290 NGX nominations under the former observed-only rule. Capture
preserves the resource before the original call; successful evaluation and actual queue
submission are separate requirements. Capture diagnostics report `used_contract_state`. Capture always copies the full depth subresource (plane zero for packed formats),
as D3D12 requires for depth/stencil textures. A validated active rectangle, including offsets and
allocation padding, travels with that capture. The shader and immutable depth sampler use the
same rectangle; no extra GPU crop pass or illegal partial depth/stencil copy is introduced.
An older shader without the rectangle uniform must remain mono for cropped input.

A newly created native command list can record transitions before its first Reset or API
evaluation. The observer associates its first recording at native operation entry, before
forwarding the operation, so those transitions and any Close, render-pass or opaque-command
restrictions reach the capture owner. The callback retains that recording identity across the
original call; a later Reset cannot relabel old evidence as belonging to the new recording.
Creating the identity supplies no resource state: capture still requires an actual observed
nonzero transition, and a submission with no observed recording cannot establish ownership.

Nested NGX capture is suppressed only when an enclosing SL or NGX evaluation has actually recorded
a valid native copy, or when SL has confirmed enabled FG authority for that viewport. A retained
resource, metadata-only ticket or rejected native attempt does not block a usable inner NGX input.
Pending recorded copies retain deduplication authority; all nonzero tickets still receive completion.
SL and NGX observations remain separate, so malformed input from one cannot invalidate the other's
usable frame. The shared selector prefers readable SL and uses ready NGX when ordinary SL is
unusable, with the bounded established-SL pending interval and explicit FG scope described above.
An explicit successful release of its feature ends NGX ownership and permits automatic fallback;
silence alone does not. Manual pins continue to override the automatic provider.
Concurrent ambiguous NGX feature/view evaluations remain mono; the add-on does not infer which
unrelated viewport is the final game camera.

Ordinary NGX Super Resolution does not supply a camera projection. Its confirmed device depth uses
the shared raw-scene controller with the infinite-far assumption above. Explicit linear view distance
uses reciprocal depth directly without that assumption or an invented camera matrix.
Calibration is keyed to the provider, logical feature generation and depth convention, rather than physical texture addresses or
dynamic render sizes. Rotating textures retain the reference; a new feature/convention starts a
fresh one. Frame and sample ordering is scoped to that logical encoding: switching between SL and
NGX resets their independent sequence watermarks, while capture-time floors and exact identity
continue rejecting delayed packets from the previous source. Immutable exact-range readbacks drive
the same independent gain and contrast-midpoint zero trial as Generic; this is
not percentile clipping or a separate NGX strength formula. Rendering still requires current depth.

The panel identifies **Depth path: NGX (game provided)** and displays the current active resource
and active resolution. Generic fallback retains its warning. Logs distinguish capture readiness,
projection availability and gain/placement state. A successful NGX evaluation and valid parameter
getters do not prove that a depth copy was recorded. Bounded producer diagnostics retain the
exact capture call's rejection stage and recording invalidation, independently of later consumer
acquisition status. Initial fallback also reports the capture identity and producer/consumer queues,
so a rejected handoff is distinguishable from an absent API. Diagnostics do not relax resource-state
or lifetime validation.
A lightweight Streamline camera-availability summary
reports whether the game also sends matrices, without enabling per-draw diagnostics. This does not
authorize combining an unrelated or merely recent Streamline matrix with NGX depth; a shared frame
and view association must be established first. AMD support remains unimplemented.

The NGX adapter queries depth, source identity and reset parameters. Motion-based calibration,
its auxiliary capture and fitting worker have been removed; there is no calibration build flag.
Optional camera diagnostic metadata is queried only by the separate opt-in diagnostic below.
The capture adapter reads typed `Reset`
every evaluation for the shared scene-history control described under
[zero-plane tracking](#experimental-raw-depth-automation).
Streamline Frame Generation is a separate possible source of per-frame camera constants, even
when Super Resolution uses NGX. Its camera observations still require frame/view association;
loading its DLL or enabling frame generation does not automatically authorize a projection for
an NGX capture.

### NGX direct-calibration diagnostic

`[SUNSHINE_DEPTH] NGXCalibrationProbe=1` enables a default-off availability check after restart.
At most once per five seconds per observed capture-eligible D3D12 feature, it queries the optional
research `Position.ViewSpace` texture with the SDK's typed resource getter and the public
`WorldToViewMatrix`, `ViewToClipMatrix`, `InvViewProjectionMatrix`, and `ClipToPrevClipMatrix`
names with its optional void-pointer getter. Failed queries, successful null values, non-null values
and a missing getter are reported separately. A failed getter's output is discarded.

The probe records bounded POD metadata in the evaluation callback and logs from the existing poll.
It never dereferences or retains these optional pointers, copies their resources, or changes rendering.
A non-null position texture would justify a separate capture experiment: paired raw depth and valid
view-space Z could fit `raw = A + B/Z` within one frame. Presence alone does not establish contents,
encoding, frame alignment or that the feature successfully evaluated. Likewise, matching an NGX depth
address with an SL tag does not establish that stale SL projection constants apply after FG changes.
No matrix or position data from this diagnostic is used to calibrate depth. Set the option to `0`
and restart after the investigation.

### Upscaler call-route diagnostic

`[SUNSHINE_DEPTH] UpscalerCallTrace=1` enables a separate, default-off runtime trace after a game
restart. It observes supported Streamline evaluation hooks and the public NGX D3D11/D3D12
CreateFeature, EvaluateFeature and ReleaseFeature exports in already-loaded modules, including
the game executable when it exports the NGX SDK. This switch adds diagnostics only. The shared
hooks separately serve the default-on NGX depth adapter described above; the trace does not enable
or disable production capture, choose depth, or change stereo/scale.

The trace logs installed-hook coverage, the observation window, API/feature call counts,
success/failure, caller module and offset, and a bounded first-call stack. NGX feature identity
remains unknown when its creation was not observed. NGX calls nested on the same thread inside
an observed Streamline evaluation are identified separately from calls outside that scope.
Outside-scope calls alone do not prove direct integration: unobserved hooks or another thread
can break that association. Zero calls mean none observed in the reported window, not API absence.
Use positive call/caller evidence to decide which integration adapter a game needs; DLL presence
or version alone is insufficient. This diagnostic is independent of the active depth-path label.

NGX depth telemetry distinguishes `nominations`, `copy_recorded`, and `metadata_only`.
A successful metadata nomination can survive a rejected pixel copy and does not establish
readable depth. Rejection details are retained even when that nomination has a nonzero ticket;
`observed` and `observed_state` distinguish absent state evidence from observed COMMON/zero.
Recording a copy still does not establish GPU completion or a successful consumer handoff.

Diagnostic callbacks only record bounded in-memory observations. Discovery, module-name resolution and
throttled logging run from the existing serialized effect callback. Process-pinned hooks become
pass-through when disabled. Set the option back to `0` and restart after the diagnostic session.
Full `StreamlineCameraProbe` command/content tracking stays disabled unless separately requested.

### Streamline camera metadata experiment

The add-on includes an optional passive camera-metadata probe using the same
[Streamline ABI families](#streamline-abi-compatibility) as the production depth adapter. It observes
only an already-loaded `sl.interposer.dll`; it does not load or replace game middleware. One shared
adapter handles each supported interface, without game-name profiles. Unknown ABI majors remain
unobserved. Camera constants and tagged depth resources are copied for validation; original call
arguments and results are passed through unchanged.

It is **disabled by default**. To opt into a diagnostic session, set `StreamlineCameraProbe=1`
under `[SUNSHINE_DEPTH]` in the game's `ReShade.ini`, then restart the game. Omit the setting or
set it to `0` for ordinary play. With it disabled, no probe-only barrier, end-render-pass or command-close callbacks are
registered. Shared raster/mesh/copy/resolve depth-activity callbacks remain active, but skip detailed
metadata tracking and native discard discovery. The lightweight depth-source mode above can still
observe middleware constants, tags, evaluations and resource lifetimes. If both modes and the
independent call-route diagnostic and NGX capture are disabled, middleware discovery and polling are inactive.
This removes unused diagnostic work; it is not a
measured frame-rate claim.

This is an acquisition experiment. Diagnostic camera metadata does not independently authorize
shader inputs or change the independent reference calibration. Production
direct capture and projection-based scale are the separately bounded use described above.
ReShade's log records projection/depth-encoding checks, viewport
and source-resource matches, freshness evidence, and reasons for unavailable or rejected data.
The bounded five-second report includes the immutable evaluation's A/B coefficients, frame identity,
selected source/layout, depth tag and association status. The separate latest-camera line must not
be joined to selected-depth evidence by timestamp. Internal camera snapshots omit unused pose
metadata but retain optional jitter for capture-bound registration. The SDK wire structures retain
their original fields and binary layout. Missing or invalid optional jitter does not invalidate
otherwise valid projection coefficients.
Compare valid A/B coefficients, near/far/FOV and reset flags within each viewport, alongside
the existing source/frame association reports. Five-second snapshots can establish sampled
stability, not rule out shorter projection changes or provide dense point correspondences.
Matching a camera matrix and resource does not prove that a captured depth copy contains the same
scene or that game-space units represent meters. Same-observation evidence is distinguished from
an explicit frame identity; an opaque Streamline frame-token address alone is not a frame number.

Camera, tag, frame history, evaluation and FG-option metadata are published as coherent immutable
versions. Every callback, renderer, panel and diagnostic reader pins a completed version without
owning the mutation flag. The synchronous tag camera selection and evaluation capture use one pin,
so they cannot splice different metadata versions. Presentation markers retain their separate
owner. The v2 token table uses a separate instance of the same publication mechanism, so neither
unrelated metadata reads nor token-generation reads can lock out SDK token registration.

Each owner has eight fixed slots. A nonblocking writer reserves an unpinned slot, copies the
latest completed state, mutates its private candidate and publishes only after lifecycle checks.
Publication tickets include a monotonically increasing generation to reject address-reuse ABA.
Readers can retain old values; they do not grant old observation generations authority. Retired
source leases are destroyed outside mutation ownership, including when the final reader releases
an old version. Actual writer collisions and exhaustion of pinned slots remain explicit losses;
they do not silently publish partial state or authorize retained depth. This is bounded storage
and acquisition, not a guarantee that observations can never be lost under arbitrary contention.
Lifecycle reset closes observation admission before clearing each owner and opening a fresh epoch.
Recycled token addresses still get a new generation even if the optional numeric index is unchanged.
Camera reset, SDK failures, token-generation checks, resource lifetimes and depth expiry remain
authoritative. The command/content ledger retains its independent observation contract.

When the diagnostic is enabled, the actual depth-sampling submission freezes a compact camera
observation in its existing immutable sample envelope. The coefficients, evidence level and
evaluation identity must agree with that submission's source, lifetime, crop and preserved-copy
facts. The capture owner's frame and Streamline's frame remain separate. Readback cannot attach
a newer camera to older pixels, and rejected observations clear the coefficients. All observation
levels retain `proof_admitted=false`; none authorizes rendering. Throttled diagnostics may inspect
up to three tagged addresses in the selector's existing resource map with a nonblocking lock.
Unknown or busy lookups remain explicit and never override selection. These queries, lookups and
reports are skipped when the probe is disabled.

The probe compares tags with the **original game depth resource**, not the selector's backup
texture. The diagnostic branch of hook callbacks performs bounded observation only; its validation and throttled logging run
from the effect callback. Once hooked, the observer module and interposer remain resident until
process exit so in-flight return paths remain valid. Disabling/unloading the add-on stops observation
and leaves safe pass-through calls; updating the DLL therefore requires a game restart.

On D3D12, evaluation entry and the selector's actual preserved-copy sites also record command
object lifetimes and recording generations. The log distinguishes source-address matches from
two markers in the same recording with one observed close and submission on the runtime's queue.
Repeated submissions, retired objects, missing callbacks, unknown native identities, secondary
execution and incompatible queues do not earn this association. These are pre-call API
observations, not proof of native API success or GPU completion. Different recordings on the
same queue remain explicitly order-unverified: concurrent native submissions can occur in a
different order from their ReShade callbacks.

The tracking store is bounded and nonblocking. A missed lifecycle event revokes old evidence;
independent per-ReShade-object lifetime cookies allow an ordinary reset and submission to
reestablish tracking without restarting the game. A busy marker capture simply drops that
marker. D3D11 has different deferred-context semantics and does not use this D3D12 model.
For an opted-in diagnostic, while no supported Streamline observer is active, content write/invalidation callbacks return
before taking the tracker lock or scanning the ledger. Lifecycle discovery and command-only copy
markers remain available. Activation starts a fresh content-observation epoch, preserving current
resource lifetimes but rejecting proof from the inactive interval.

The content observer adds immutable use/copy snapshots to that same recording. Observed draws,
depth clears and copies advance the source's content version; resource lifetime, backup reuse and
later backup writes revoke mismatched snapshots. A copy followed by a source clear and then an
evaluation is rejected, while an evaluation followed by a matching preserved copy and then a clear
can retain their observed correspondence. Partial writes and ambiguous alias/barrier events revoke
evidence rather than infer which pixels survived. Recording destruction retires its uncertainty
entry so short-lived command lists do not exhaust the bounded store permanently.

The add-on separately observes native D3D12 `DiscardResource`, which ReShade 6.8 does not expose
as an add-on event. It discovers the standard COM entry from live native command lists, retains
at most eight interface slots, and installs hooks later during polling. Partial discards
revoke whole-resource evidence without reading the optional rectangle array. Hook targets and the
add-on stay loaded for process lifetime; shutdown leaves pass-through hooks in place. Activating
new observation coverage starts a fresh content epoch, so earlier snapshots cannot inherit it.

Evaluations with validated scene-depth contracts publish diagnostic camera/depth evidence,
independently of their feature ID. Typed 2.x inputs and exact legacy frame/view contracts use the
same production admission described above. Unsupported opaque inputs execute unchanged without
invalidating another source. The separate v2 tag adapter can capture `OnlyValidNow` depth at a successful
tag boundary independently of FG. Legacy Reflex uses the apparent viewport argument for a timing marker; treating marker
0 as viewport 0 would overwrite the rendering evidence.

The observer separately retains HUD-less, scaling-input, scaling-output, backbuffer, UIColorAndAlpha
and UIAlpha tags. Live UI capture at tag boundaries independent of FG is described under the UI protection
contract; merely retaining diagnostic color metadata does not authorize a mask. Actual effects-input
identity comes from ReShade's begin-effects render-target view,
which may differ from the swapchain after a resolve/copy. V1 Reflex and V2 PCL PresentStart/End
markers supply bounded, current-thread presentation brackets. V2 PCL discovery passively observes
`slGetFeatureFunction`; a marker function cached before the hook was installed remains unavailable.
Failed, nested, repeated, stale, untracked or out-of-order brackets do not establish a current frame.
Thread lifetimes are retained explicitly so a reused OS thread ID cannot inherit an open bracket.
These are frame, identity and declared-extent observations; none establishes final spatial alignment.

An accepted result is explicitly **observed-content-match, coverage-incomplete**. Other native
mutations and some enhanced-barrier and render-pass resolve information remain unobserved.
These diagnostic observations alone do not authorize rendering across aliasing or generated frames,
or establish final-color/depth registration. Production capture and jitter admission are described above.
Different command recordings remain unverified even on the same queue. Direct sampling and copies
recorded inside the current effects pass may have no submitted copy marker. None of these results
changes existing depth selection, readiness or stereo output.

To test, enable the diagnostic setting, start a supported game, enter actual gameplay, and inspect `ReShade.log` for the camera
probe report. A present but unused Streamline library may supply no camera constants. That outcome
is recorded rather than treated as support. The separate production Streamline path above consumes
supported capture-bound projection data; the passive diagnostic alone cannot authorize rendering.

### Controlled camera-depth candidate

This section records the historical controlled candidate and its archived fixtures. Its
`SUNSHINE_GAME3D_CAMERA_DEPTH` opt-in and independent fixed scale are not current setup instructions.
Current Game3D embeds `game3d_native.hlsl` in the add-on; source-owned readiness starts false, and
Streamline projection and raw automation publish their explicit bases through its constant buffer.
Unsupported current frames stay mono. The historical fixture below supplied its own known inputs.

For an explicitly validated projection `d = A + B/z`, the candidate computes inverse distance
`q = (d-A)/B` and prepared depth `D = 1/(1+K*q)`, where `K` is supplied by the independent
scene-gain controller. It does not
select or clamp to scene percentiles. It does not forcibly map finite far to one; true infinity
maps to one. Ordinary floating-point storage still limits distinguishable distances.
Metadata must keep the prepared depth and convergence representable in the actual RG16F textures;
rejecting an unrepresentable configuration is separate from discarding scene depth extremes.

The convergence field is `referenceZPD*K*(q0-q)`. Holding `K` and `referenceZPD` fixed lets `q0`
move the screen plane without changing the inverse-depth slope. The active candidate bypasses
the competing legacy scene gain/convergence adjustments while preserving their saved controls.
The active candidate also omits the legacy compatibility translation added after ray search;
otherwise a zero convergence field would still have nonzero binocular disparity. Its stored
compatibility setting is preserved for the original path. Ray search, reconstruction, anti-aliasing,
sharpening, export and HDR remain the original-derived pipeline. Full rendered disparity still has
the renderer's existing occlusion and range limits.

The production Streamline path retains the inverse-distance camera decoding above; the sole
Sunshine renderer no longer needs this candidate's rational preparation texture. The historical
controlled fixtures remain separate from real game validation; broad game quality and final-color
registration are still live acceptance work.

### Experimental automatic scale and screen plane

This fixed-scale policy and the mode gates below are historical. They do not describe current
Game3D's current independent-gain/zero-plane controller or an available Manual shader mode.

`tools/reshade/camera_scene_policy.h` retains a historical, separate CPU policy for the controlled
camera fixture. It is not the production `projection_depth_controller.h` path: its scene-derived
fixed-reference `K=1/q_ref` and admission logic must not replace the shared reference policy and
production source admission described above. Its input
requires independently admitted depth/color/frame registration, with immutable projection and
source/extent identities attached to the actual sampled frame. The existing sampler's resource
token and the latest camera matrix alone do not establish that correspondence.

The default native-camera strategy observes the fixed central 4x4 cells of the 32x18 depth grid. It requires at least four
distinct captures spanning 750 ms with coherent mean inverse distance, including every accepted
sample's temporal variation. An unstable window restarts without extending the five-second
startup deadline. Exact depth endpoints make the whole reference ambiguous; they are not trimmed
out or reweighted, and the rendering shader still retains its endpoint depths. These timing and
coherence values are prototype policy constants, not game-specific settings.

For reference inverse distance `q_ref`, the candidate chooses `K=1/q_ref` and a fixed
`referenceZPD=0.05`, initially placing `q_ref` at the screen plane. Both remain fixed after startup.
Later samples move only the screen plane `q0`, using elapsed-time smoothing. The policy stores
`q0` and its target directly; normalized `K*q0` is derived only for diagnostics. Dividing the
normalized motion-rate bound by `K` preserves the existing screen-plane movement in these
coordinates. Target freshness comes from capture time, not readback arrival. Missing proof, stale
targets, cuts and unsupported FP16 ranges suspend readiness while retaining the established scale;
resumption does not accumulate movement credit from the gap. Only an explicit calibration/unit epoch resets K.

This reference is scene-relative, not meters or a universal stereo strength. Starting against a
wall versus a vista, or including a close foreground object in the central patch, can choose a
different scale. Freezing K avoids continuing scene gain changes but does not solve that initial
choice. Likewise, preserving K during missing evidence does not establish seamless rendered
transitions: the current shader readiness fallback restores the saved original rendering path.
Live transition behavior and representative silhouette/AA comparisons remain acceptance work.

The opt-in `SUNSHINE_GAME3D_CAMERA_SCENE_TEST=1` fixture runs this CPU policy through the actual
external shader with synthetic, known registration; it also requires
`SUNSHINE_GAME3D_CAMERA_DEPTH=1`. It tests automatically chosen scale, equivalent world units,
clipping-plane changes, walking disparity, the measured screen plane and cut/gap continuity.
It does not replace a real game's camera/depth association or a headset quality test.

### Experimental raw-depth automation

Game3D acquisition has three owners. Capture publishes immutable per-present records and
authenticated asynchronous samples. Selection chooses a usable current copy from those records,
using independently maintained source preferences. Calibration consumes completed samples for
their exact sources independently of which source can render this present. A missing current copy
returns mono without resetting unrelated numerical evidence.

Capture produces any permitted end-of-frame backup before current-source routing.
In preservation mode 1, a game may draw depth without a trailing clear, so waiting
for a selected current copy before making that backup would deadlock discovery.
The capture members, legacy selection and histogram probe reuse the same producer;
it checks the live source lifetime, present, activity and owned backup assignment.
An in-progress copy is reserved and unavailable to other runtimes. Mode 2 still
forbids end-of-frame reads on D3D12/Vulkan because those resources may be aliased.
Lower-resolution depth remains eligible when its active region matches the output
shape; output-resolution preference does not require a native-resolution buffer.

The existing full-grid content classifier runs once per completed readback. Selection and
calibration reuse that same result. Only authenticated captures classified as useful enter the
numeric sample stream; flat or unreliable captures cannot initialize or refine H or refresh its
target. This is a property of that capture, independent of the current preference or accumulated
selection confidence. Rejected content leaves the prior numeric history intact and its target
expires normally. A flat loading frame with interior depth values therefore cannot seed the
reference later used by gameplay.

`raw_scene_policy.h` owns one controller for one exact physical depth source. `raw_scene_pool.h`
retains up to sixteen exact-source histories independently of the four-slot capture set. The
capture owner validates their lifetime, layout, dimensions, crop and depth direction against
the live resource inventory in one bounded lookup. Temporary loss of a capture slot does not
erase calibration. Destroyed or changed sources are invalidated; inactive histories use bounded
least-recently-admitted eviction when the cache fills. Only current capture-set members can
ingest observations or render. Its `synchronize`,
`observe` and `evaluate` operations respectively apply authoritative source membership, ingest
completed measurements, and evaluate current-frame geometry. The capture owner also supplies a
runtime/frame watermark; a packet cannot establish membership or authorize its own frame. Neither
controller wraps the physical-camera policy or fabricates camera registration. Shared identity
types and numerical constants do not
establish recovered camera distance. Orient raw depth toward the viewer as `t=d` for reversed
depth and `t=1-d` for normal depth. The preview remains `D=1/(1+H*t)`, while geometry uses
`0.05*H*(t0-t)`. Gain `H` and zero `t0` are independent state.

Both raw and camera controllers reuse `tools/reshade/scene_gain.h`; source admission remains
with their respective owners. In the formulas below, `q` means reconstructed inverse distance
for the camera path or oriented raw depth for the fallback, and `K` means the corresponding
gain `K` or `H`. The reference convergence factor remains `0.05`.
For a non-flat current range, the independent nearest reference `Q` is the largest converted
inverse depth of every pixel in the active depth rectangle. The gain target is `Ktarget=L/Q`,
where L is the normalization derived from the output shape and full-strength parallax limit below.
The target signed field is `0.05*L*s*(q0-q)/Q`, with `s` the user strength fraction. Applied gain
can lag the target in either direction; the renderer separately caps each pixel's parallax.
Scaling the raw-depth or game-distance coordinate scales `q`, `q0` and `Q` together, so units
cancel from the ratio. Moving `q0` does not redefine `Q` or change pairwise separation in the
unclamped field by itself. The current scene supplies Q; no startup instruction screen or scene permanently fixes it.
This is an artistic reference, not a recovered game-unit scale. Camera movement can still change
it; temporal tracking reduces, but cannot eliminate, that dependence. In particular, a nearly
flat wall is not amplified to fill the entire permitted disparity range.

One shared `depth_moments.h` grid generator partitions the active depth crop according to its
aspect ratio, targeting approximately 576 nearly square tiles: 32x18 at 16:9, 28x21 at 4:3,
and 18x32 at 9:16. Tiny crops cap each axis to its pixel extent; a bounded capacity handles
extreme aspect ratios. Resolution changes at a fixed aspect otherwise preserve the grid.
Generic selection takes one representative texel per tile; a geometry request additionally
reduces every texel in those same tiles to raw extrema and decoded inverse-depth statistics.
They share layout generation, not the decision about which buffer to use. Known API providers
bypass Generic candidate selection. The contrast-midpoint trial first obtains each tile's
extrema, then traverses that tile again to accumulate first and second moments centered on its
decoded minimum. Both scans run in the same GPU dispatch. They retain the existing resources,
bounded asynchronous readback and queue submission, with no additional CPU/GPU wait. The extra
texture traversal and reduction add GPU work; unchanged dispatch count is not a performance claim.
Projection coefficients are frozen with the capture and applied before accumulation. The CPU
merges centered tile sums in double precision, shifting each origin to the full-image minimum
using nonnegative offsets. This avoids deriving small depth contrasts by subtracting nearly equal
raw moments. Totals follow pixel counts, never equal weighting of unequal-sized tiles.
The full-image maximum still supplies Q, including a nearest pixel between the diagnostic grid
points. The full-image minimum and centered moments supply the trial zero target below; the
previous uncentered mean and squared moments remain diagnostics. Full-image sampling removes
point-grid aliasing without discarding sparse geometry.
Every finite value contributes, including hardware endpoints, sky and thin objects between
point samples. Allocation padding outside the crop does not contribute. Any NaN or infinity
inside the crop invalidates the range; no percentile selection, trimming, winsorization or
epsilon clipping repairs it. Conversion uses the capture's frozen affine coefficients and
reorders transformed extrema for reversed encodings. The range remains evidence for that
captured frame, not a guarantee about newly appearing geometry on every later present.

For source dimensions `W,H`, the source-UV conversion factor is `c=(H/2160*100)/W`.
The existing field clamp is `[-1.5,+2.5]` and the Host source-U container is `[-0.04,+0.04]`.
Their combined foreground and background magnitudes are `F=min(1.5,0.04/c)` and
`B=min(2.5,0.04/c)`. The shared `game3d_stereo_contract.h` supplies the full-strength per-eye
source-UV limit `U=0.01`. `render_limits` derives `L=min(F,B,U/c)/0.05`; at 16:9, L is about
7.68, independent of source resolution. This is an artistic display budget, not a recovered
camera baseline or validated human comfort standard. At user strength
`s=clamp(strength,0,100)/100`, the signed input is `0.05*K*s*(q0-q)`. The renderer limits final
per-eye horizontal parallax to `[-U*s*b,+U*s*b]`, with b the stereo reentry blend. For example,
50% strength allows at most 0.5% of source-image width per eye before any further reduction by
reentry or reused-frame strength protection. The same clamp is reapplied after horizontal Q30
conditioning so fixed-point rounding cannot exceed the final budget. The limit does not clip or
replace decoded depth.
At zero strength, established gain and zero hold and rendered parallax is zero.

Initialization collects four valid non-flat observations at least 250 ms apart, spanning at
least 750 ms. The average of their nearest references initializes gain to `L/Q`; the average
of their contrast-midpoint targets initializes the zero. An exact flat range cannot initialize scale.
After initialization, a positive flat range updates the zero target while holding gain and
clearing its gain target; an all-zero range holds both controls and clears both targets.
No epsilon span is invented. Incomplete
evidence expires after 1500 ms rather than accumulating an indefinitely old startup window.
The scene may move during initialization; there is no stationary-scene variance test.
The diagnostic startup count is not a lifetime observation count.

Each fresh valid non-flat measurement sets `Ktarget=L/Q`, independently of the
actual strength slider. Dividing this target by user strength would cancel the slider after
convergence, so user strength is applied only afterward. The initialization reference is a seed,
not a permanent target. Gain follows the reference with exponential smoothing in log gain,
using the existing 0.5-second time constant and a maximum doubling/halving rate of once per
second. A newly appearing near point does not immediately reduce gain across the whole image;
its rendered parallax is capped while gain adapts. Exact flat depth suspends gain adaptation.
Missing or expired evidence, all-zero depth, zero strength, clock rollback and presentation gaps
over 250 ms disarm temporal adaptation; resumed frames earn no catch-up time.

The current zero-plane trial uses a contrast midpoint. For every decoded pixel in the active
depth rectangle, let `b=qmin` and `d=q-b`. For a non-flat range, the target is
`m=b+0.5*sum(d*d)/sum(d)`. The ratio is the mean contrast weighted by contrast itself, so pixels
at the farthest depth contribute zero while nearer pixels contribute by their depth contrast
and area. Every finite pixel remains represented; there is no percentile trimming or semantic
main-object detection. For two exact depth layers this equals their midpoint independently of
their relative areas. With additional layers it lies between `b` and the former extrema midpoint
`(b+Q)/2`. Gain remains `Ktarget=L/Q`. The UI midpoint independently tracks the extrema midpoint
for historical depth-mode placement and diagnostics, as described below; live mode 5 does not consume it.
The scene-zero formula is unchanged by UI protection.
After initialization, an exact positive flat range targets its single depth `b` while holding
gain; an all-zero range holds both controls and clears their targets.

All live geometry samples supply the centered payload described above. Legacy point/range
fixtures and old uncentered-moment fixtures that lack it retain the extrema midpoint solely for
compatibility. Invalid supplied centered statistics invalidate the target instead of silently
falling back to that midpoint. The trial is a production policy experiment, with headset
acceptance still pending; historical midpoint and fixed-reference evidence does not validate it.

The applied zero follows this target with the same 0.5-second exponential time constant:
`alpha=1-exp(-dt/0.5)`, followed by
`delta_q0=clamp(alpha*(m-q0), -L*dt/Knew, L*dt/Knew)`.
The normalized bound `abs(Knew*delta_q0/L)<=dt` permits at most one full parallax budget per
second from zero movement alone. It does not divide by slider strength and is invariant to a
positive multiplicative change of inverse-depth units. There is no immediate clamp of the
applied zero to the new range; it can remain outside that range during a transition. The renderer's
final per-pixel limit remains active. The zero and gain targets share accepted evidence and timing;
a positive flat range can move the zero toward its single depth without changing gain.

The independent UI midpoint targets `qm_target=(qmin+qmax)/2`. Its initial value is the arithmetic mean
of the same four accepted startup samples' extrema midpoints. It then follows its own target
with the identical exponential update and `L*dt/Knew` step limit above, sharing the scene
controller's accepted evidence, clock, reset, hold and expiry rules. It adds no independent gain
or sampling pass. Positive flat depth updates both plane targets to that depth while holding gain;
all-zero depth clears both targets and holds both tracked values. The applied UI base travels with the
resolved real-depth scene, including FG reuse, instead of being recomputed from newer statistics.
This tracking remains available for diagnostics and historical depth-mode replay. Live mode 5
instead applies a fraction of the current positive display-parallax limit, including strength and
stereo blend, subject to warp readiness. Its independent temporal policy observes scene/UI overlap
as described in the setup contract above; it does not map this historical midpoint to parallax.
It dispatches no covered-depth maximum. The scene candidate and vertical fields are unchanged;
the existing source-alpha conditioner moves UI and its horizontal support to one rigid plane,
locally changing adjacent geometry as described above. The scene's positive and negative display
limits stay unchanged, so foreground can be nearer than UI even at the 50% live cap. Fullscreen UI may
encounter the existing edge-sampling limits. Depth/alpha registration
and bounded FG reuse limitations still apply. Historical modes 0 through 4 retain their replay
behavior but are not selected for live placement.
A change of nearest depth can change gain and hence separation between otherwise unchanged
depths. The exact maximum deliberately includes even a single nearest particle or surface:
if it persists, its smaller gain target can flatten the distant background. Large depth contrasts
also receive more zero-target weight, so the trial remains sensitive to near outliers. Together
with nearest-depth gain adaptation, zero tracking can reduce an approaching object's apparent
pop-out in a three-layer scene. Gradual tracking limits abrupt global changes but cannot
remove these steady-state tradeoffs. During adaptation,
per-pixel saturation can reduce depth separation locally. There is no percentile rejection or
menu/content detector in this policy.
No manual reset is needed to recover depth strength or leave startup imagery behind. A real
depth-domain reset initializes a new scale and zero automatically.
The renderer applies the final parallax limit every frame because range evidence is asynchronous.
Game3D does not force the final parallax to zero at the screen edges. Source sampling retains its
basic coordinate and sampler clamps; an inverse lookup outside the finite image repeats its
nearest edge color because no offscreen color is available. These sampling bounds do not alter
the final parallax field or flatten an edge collar.

SL, NGX and Generic share this gain/placement policy. K/H are artistic gains, not projection
coefficients or a recovered viewer baseline. A matched positive multiplicative coordinate change
preserves non-flat range initialization and the resulting signed field when gain and zero are
transformed together. Raw fallback still cannot report physical distance from an unknown
depth offset; arbitrary nonlinear encodings or different initialization histories need not
agree. There is no camera-motion estimation or game-name lookup in this calculation.
Unsupported finite-storage geometry remains mono rather than modifying input depth to force
publication. The shared immutable `scene_feedback.h` input carries SDK reset evidence through
the same capture/readback path.

NGX reads its typed `Reset` parameter on every evaluation. SL's observation revision and NGX's
per-feature reset revision reject older readbacks without
changing logical source identity. Established gain and the applied zero plane remain; old
zero-plane targets are discarded while valid current depth can continue rendering. Startup samples
from different revisions cannot mix. The always-visible UI table displays **Nearest reference Q**,
**Farthest (q min)**, **Zero plane q0**, **Stereo gain K**, **Reference midpoint q**, and **Normalization L**, with
**Current/Target** columns. Q and the farthest bound come directly from the same accepted policy
measurement, without new sampling, and cover the full active depth rectangle. The depth
rows use the same converted inverse-depth coordinate, where
larger values are nearer; the bounds are scene extrema, not camera clipping planes. Unavailable
measurements display `--`; retained values during a missing-depth hold are labeled as last applied.
Dump 3D records the captured scene decision's accepted measurements in
`render_scene_policy.depth_statistics`, rather than statistics newly computed for the dumped frame.
Its `centered_moments_supplied`, `mean_q_minus_min` and `mean_square_q_minus_min` fields identify
the centered evidence; `zero_plane_policy` records the trial formula and the legacy fixture path.
A missing-depth decision may record `null` while the UI retains older values labeled as held;
the dump never borrows UI history to fill missing frame evidence.
The gain target follows the latest valid nearest reference as `L/Q`; the applied gain can lag on
either side. L is captured for the render decision's output shape. `1/K` is not the measured Q.
The panel separately shows the per-eye source-width parallax limit at the current slider strength.
The zero shows its applied value and contrast-midpoint target. A positive flat scene can show a zero target
while its gain target is absent. **Reference midpoint q** shows the independently tracked historical UI
depth and its extrema-midpoint target. These are diagnostics, not the live mode-5 placement input.
The panel identifies adaptive UI targets as **0 / 10 / 25 / 50% of the front limit**.
The exact consumed fraction and optional accepted-observation telemetry are recorded separately;
the bounded conflict readback is not a covered-depth plane value.
Camera mode additionally shows current and target **Zero-plane distance** as
the reciprocal inverse depth in game-distance units, including infinity at zero. The table retains **Conversion scale**
and **Conversion offset**. A visible status reports when gain is below target while adapting.
Moving the zero alone does not change gain. The matrix line
describes `q=scale*raw+offset`, using the same coefficients as the shader. Logs retain
`stereo_scale`, `target_scale`, `q0`/`t0`, `target_q0`/`target_t0` and `plane_state`, plus `reference_Q`, `normalization_L`
and `reference_valid`; `target_scale` is the nearest-reference-derived gain, not a reciprocal zero
target. Dump 3D records the captured decision's `target_zero_inverse` and converted
`target_zero_plane`, or null when no active accepted target exists. Its `zero_tracking` object
records the production time constant and normalized zero-speed bound. Held UI values likewise clear
their target columns. Projection logs also include `conversion_scale` and
`conversion_offset`. There is no requirement for gain times zero to equal one.

The API provider normally requests a sample about every 125 ms, plus GPU completion delay.
Generic selected sources normally sample every 250 ms when they render, subject to frame
scheduling and intervening challenger probes. Neither cadence guarantees a range for every present.
Sampling still permits only one in-flight GPU readback. A valid completed measurement may update
its source while another source is selected or current depth is unavailable. Its numeric evidence
does not depend on GPU backup storage remaining allocated after authenticated readback.
Invalid input or missing presentation suspends placement until current evidence is usable.
Repeated ingestion of the latest packet is equivalent to no new packet: it never refreshes the
range/statistics evidence or its age. Gain and zero can continue following already valid targets on fresh presentations;
authorized depth reuse holds its captured controls as described above. Output dimensions can update
L and the gain target for a still-valid measured range; strength scales rendered parallax and its
limit without changing Q or the contrast-midpoint target. Ranges expire 1500 ms after capture, rather than after readback
arrival. Ordinary expiry pauses placement and reports `holding_reference`: initialized H and t0 remain
unchanged while the same exact source supplies a valid current depth image. It does not turn
3D off or reuse old depth pixels. A fresh range resumes placement. Missing or
ambiguous current depth, explicit cuts, malformed matching evidence and genuine basis changes
still fail closed; an uninitialized source still needs its complete startup window.

Generic candidate selection continues to use the point-grid classifier. Once a source is
selected, its authenticated full-image statistics reach geometry independently of that sparse
classifier: thin geometry between representative points must not be discarded. Capture, source
and layout checks still apply; grid-only challenger results cannot supply geometry statistics,
and malformed full-image measurements invalidate prior evidence. Central pixels have no special
gain or zero authority. The standalone `raw_reference_statistics.h` utility and the
[comparison record](native-stereo-comparison.md) describe earlier research candidates, not
the current bounded range policy. A menu with useful depth can still initialize a reference;
the policy does not identify menus. Representative moving-game comfort remains acceptance work.

The GPU constant buffer's coordinate-basis input explicitly distinguishes this raw model from a native
camera model. Both now feed the sole Sunshine warp directly; HDR and export retain
their separate responsibilities. Game3D sharpening has been removed.

Game3D embeds its GPU program in the add-on. Calibration passes values directly to the renderer;
there is no production FX uniform discovery or user preprocessor definition.
Effect reload preserves native rendering and calibration. Backend recalibration hooks remain for
diagnostics and test fixtures; they start a new raw basis epoch and invalidate queued old-epoch
samples. There is no recenter button in the normal panel. Ordinary source changes and gain
recovery are owned by the controller. The earlier independent test gates
(`[SUNSHINE_DEPTH] RawSceneAutomation=1` plus `SUNSHINE_GAME3D_GENERIC_CONFIG=1` and
`SUNSHINE_GAME3D_CAMERA_DEPTH=1`) describe archived controlled candidates only and are retired
from current production setup.
One initialization window requires four distinct useful captures at the spacing above. It begins
with useful input and has no five-second timeout/retry wrapper. An unusable new matching-source
observation invalidates that source's incomplete evidence. A missing current copy does not break
the window or move its asynchronous capture-admission floor. Authoritative source facts are
applied first, all available completed packets are observed, and current geometry then ticks once.
Sample and current-frame order are checked separately.

A source outside the retained capture set supplies its own fresh initialization window. Lifetime,
crop/layout, orientation or reported convention changes invalidate that member's numeric state.
Membership does not prove a shared depth encoding: each allocation independently initializes and
maintains H and t0. A → B → A within the retained set resumes A's controller. Its completed measurements
can arrive while B renders and still update A. Leaving the set evicts the controller rather than keeping an
unbounded historical calibration cache. Pinning or unpinning the same source preserves its state.
A genuinely missing current capture suspends rendering and screen-plane motion; it never makes a
previous frame's depth current. Valid immutable measurements captured before a brief output gap
remain admissible for their exact source, subject to age and identity checks. Numeric readiness
alone cannot revive stereo without a new usable current depth copy. Source changes retain
replay/order watermarks, and packets cannot adopt a source.
Old-source, duplicate, stale and future packets do not refresh a target. A malformed new packet
for the current source invalidates its target while retaining the established numeric values.

A menu with stable useful depth can establish a reference; this
policy does not identify menus or assume continuity from a menu to gameplay. In Automatic,
unavailable current depth or a suspended
reference returns the current source image to both eyes, bypassing final sharpening/AA as well
as stereo geometry. Recovery ramps stereo gain over 500 ms of valid-frame exposure. A missing
frame itself has zero stereo gain but retains ramp progress through gaps no longer than 250 ms.
Time spent displaying missing-depth frames adds no progress; clock rollback or a longer gap
restarts the ramp. Ordinary generic capture does not reuse old color/depth; the explicit bounded
API FG depth-retention path described above is separate.
The zero-strength endpoint also returns current mono; an explicitly selected Normal Depth View
still shows valid prepared depth. Gain attenuation and restoration require no user intervention.
Broad game quality and moving-scene comfort remain acceptance work.

Startup text and loading screens may have no current scene depth; this is reported as depth
unavailable and keeps current-color mono. Transient depth-binding and raw-controller log changes
are limited to one per second. Ordinary rotation follows this limit; it must not produce one log
entry per frame. Manual-selection changes and binding failures are reported immediately.
Ready raw-controller summaries include the source, applied gain H and zero plane t0 every
ten seconds so independent gain and placement can be inspected without per-frame telemetry. These limits
affect logging only; depth selection, readiness, UI and shader state still update every frame.
Camera/readiness/blend uniforms have no explicit constant initializers: ReShade would otherwise
specialize them in Performance Mode despite their source annotations. Their runtime storage starts
at zero and remains writable; artistic controls retain ReShade's normal specialization behavior.

Every sample has an independent capture ID, source/backup lifetimes, runtime/device epochs, frame,
orientation and crop captured at submission. Readback uses the retained native resource's actual
format, size and crop. This association is separate from the selector's existing source-ID token;
later camera settings or source selections cannot be retrofitted onto older pixels. Submitted GPU
work still retains resources until completion when its consumer is canceled. No sample association
claims exhaustive native-write or final-color proof.

This raw integration uses the baseline's standard aligned-view assumption and accepts only
full depth views whose aspect matches the current image. The obsolete per-game coordinate,
weapon, perspective and scene-strength branches are removed. The current Sunshine conditioner
and inverse consume the resulting coordinates; HDR processing remains separate.
In the historical controlled raw mode, saved incompatible transforms retained the original path. It required
`Auto_Scaler_Adjust=0` for ordinary aspect ratios and `Disable_CO=true` at 16:10; external depth
auto-fit and nonidentity letterbox/side scaling were not admitted. Those legacy settings and test
gates no longer select a production rendering branch.
Color processing such as tone mapping
does not by itself change spatial alignment; a UV distortion or generated frame can. Neither equal
RGB nor equal dimensions alone proves registration. Direction inferred from clear values does not
prove perspective encoding: logarithmic depth, orthographic views, mixed projections and unnoticed
near/far changes remain limitations. In exact arithmetic, additive finite-far offsets cancel from
the numerator's depth difference if the corresponding zero transforms with them, but they do
not cancel from the nearest reference Q in the denominator. Therefore this ratio policy is invariant
to positive multiplicative units, not unknown additive offsets. It does not recover the camera or guarantee
identical FP32 behavior near the far plane. A screen-plane reference cannot compensate for an unknown
change of camera encoding. Validated camera metadata supports continuity across known projection
changes; without that metadata the raw fallback cannot establish physical equivalence.

### Selecting scene depth

If **Normal Depth View** shows a white field, stripes or dots during gameplay, first check
**ReShade → Add-ons → Sunshine 3D**. Try the active depth-buffer rows while watching the
depth view, and select the buffer that shows recognizable scene geometry. If necessary,
inspect that buffer's recorded **CLEAR** entries to preserve the useful contents. A bound
resource, a large resolution, or a high draw count alone does not establish correct scene depth.
SunshineGame3D's status describes source-associated projection decoding when available, or
the explicit relative-depth assumption otherwise. Both maintain an independent stereo gain and
place the screen plane within the measured disparity budget. Manually pinning a buffer changes
source selection, not that interpretation.
The independent reference effect has separate calibration. The selected buffer and current-frame
depth readiness remain shared acquisition responsibilities.

In D3D12 preservation mode 2, a direct switch from depth target A to B can also
preserve A. This additional path requires an explicit depth clear on the same
command list, single-mip/layer/sample geometry and no intervening resource-state
uncertainty. It copies only depth subresource 0, without assuming the stencil
plane's state. Touching or unknown barriers, render-pass entry and secondary
execution revoke that proof. Missing copies still suspend Automatic setup and
produce mono; this does not enable end-of-frame access to potentially aliased depth.

This resolved the reported Dead Space depth problem during manual testing on 2026-09-13:
changing the Generic Depth selection replaced flat or saturated shader inputs with spatially
varying scene depth. The previous extreme signed depths had saturated the shared preparation,
making positive Depth Adjustment settings produce the same separation while zero bypassed
stereo. Changing either warp's strength formula would not correct the selected source.

ReShade 6.8's default selection ranks eligible buffers using recorded vertices or draw calls;
it does not identify DLSS/TAA or inspect depth contents. The correct buffer can have fewer
recorded calls than another candidate. DLSS also separates render and output resolution, so
do not require scene depth to match the final color size or encode a universal low/high
draw-count rule. See the [Generic Depth implementation](https://github.com/crosire/reshade/blob/v6.8.0/examples/09-depth/generic_depth_addon.cpp)
and [NVIDIA's DLSS integration guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS.md).

The unified `SunshineSBS.addon64` includes the scene-depth selector and replaces Generic Depth for this game. Its persistent
**Auto-select scene depth (Sunshine)** option allows native and reduced-resolution candidates,
then samples raw depth contents asynchronously. Repeated useful samples can promote a scene
buffer above a busier flat buffer; repeated bad samples allow recovery without switching on
one menu or transition frame. A currently accepted source no longer blocks all other accepted
candidates. Among coherent candidates, sustained broader scene coverage can replace a
foreground-only buffer, including when their dimensions match or the fuller scene has a lower
resolution. The estimate combines the fraction of raw samples strictly between depth endpoints
with their spread across coarse screen regions. It does not identify people or classify scenery;
exact endpoint samples may be legitimate sky, and arbitrarily small positive reversed-Z depths
remain supported. Smoothness is measured within supported depth so a mostly cleared buffer
cannot earn confidence merely from its empty background.

The sampler retains one immutable native-device program: a D3D11 compute shader or a D3D12
root signature and compute pipeline. It reuses that program across samples, including different
depth formats and capture rectangles. One completed scratch bundle is reused for the next sample:
output/readback resources, command storage, descriptors and completion objects. A sample retains
its source and immutable capture facts until GPU completion; only then may the source binding
be replaced. D3D12 fence values increase across reuse, and each executed list restores the scratch
output's starting state. Runtime retirement or a device change discards idle resources; submitted
work retains its resources until completion. Failure to prove completion still quarantines one
bundle and disables further sampling. Reuse introduces no wait or additional in-flight sample.

Selection takes one owned, contiguous snapshot of resource identity, dimensions, activity and
capture-region facts under the tracking lock, then releases that lock before device calls. It
does not copy per-clear histories or camera-observation records. One eligibility-pruning pass
removes destroyed and filtered candidates while retaining evidence for live eligible sources
during inactivity, including manually pinned rows.

`depth_capture_policy.h` separates the bounded source budget, immutable current capture records,
and pure current-copy selection. Independently useful sources with matching dimensions, format
and capture region can retain capture slots; membership does not require mutually exclusive
rendering. A live unchanged member can survive temporary inactivity or inconclusive content.
The existing slow quality policy chooses a preferred anchor. Current selection uses that anchor's
usable copy when qualified, otherwise another qualified retained source's current copy. Multiple
depth resources rendering in the same present is not itself a rejection. Ambiguous provenance of
the actual copy, wrong runtime/present, missing copy, or a newly assigned backup without its own
capture still rejects that record. Transfer-only activity does not manufacture a current scene
copy, and no relationship to the swapchain buffer index is assumed.

Capture layout follows the actual preserved crop. Aggregate activity and off-turn writes cannot
change that identity without a new usable capture; a real captured-crop change advances the
layout and invalidates the corresponding numeric basis. Binding, sampling and the public current
depth accessor use the same capture record and currentness checks. The view/backup cache is
bounded and retains retiring resources until their existing retirement period ends; ordinary
rotation reuses views rather than allocating or waiting for the GPU each frame. A manual pin
continues to select exactly the requested allocation.

Challengers need three distinct fresh positive captures, using conservative recent coverage
against the incumbent's protected history. Small content differences retain the current source;
resolution remains a preference when completeness is comparable. Active captured viewport
dimensions determine resolution coverage, so padding and oversized allocations do not earn an
artificial preference. A full-resolution buffer still has to pass the content checks.

At startup, or while the preferred source has less than full output-resolution coverage,
general discovery gives active output-matching candidates earlier sampling opportunities.
These visits rotate through eligible full-resolution sources and alternate with the ordinary
all-candidate scan, whose cursor advances independently. Empty or uncapturable full-resolution
sources therefore cannot monopolize discovery or prevent a useful lower-resolution fallback.
This uses the same active-viewport coverage calculation as selection; there is no fixed 4K
threshold. A full-resolution incumbent keeps ordinary discovery order. Scheduling priority
does not grant content confidence, change fresh paired comparisons, or override a manual pin.
The existing rotating-peer hints, current-source refresh and in-flight probe deadlines remain
separate; priority improves visit order rather than promising a universal recovery time.

New candidate probes respect the same short activity grace as selection. A present that renders
another depth target pauses an already tracked, live eligible probe instead of releasing its
backup and restarting qualification. Only the candidate's next current rendered frame may supply
another sample; inactivity never makes an old backup current. Destruction, filtering and the
original two-second probe deadline still cancel it. The official-runtime `--reload-only` regression exercises native
promotion and recovery through interleaved depth-active presents, resource recreation, and
D32S8 capture before clear.

A probe chosen during backup retirement owns its pending visit before allocation. An unrelated
raw-sample completion cannot cancel that visit. Queue waiting and the allocated measurement
phase each have a two-second limit; successful allocation starts the measurement clock once,
and individual samples or paired comparisons never extend it. Releasing a probe that already
transferred its backup to a nonretiring capture slot adds no retirement wait. Actual retiring
allocations keep the existing delay.
When the current capture set leaves a present uncovered, repeatedly raster-active eligible
sources with matching geometry receive discovery priority. Recent activity can queue a probe
between that source's rendering turns; it never authorizes sampling or displaying an old copy.
Priority visits alternate with normal
round-robin visits so an uncapturable candidate cannot monopolize discovery. This grants a probe
opportunity only: current-copy authentication and useful-content qualification remain separate.
The crowded rotating-source regression exercises fresh rotating scene buffers among unrelated
active resources, including off-turn copy activity.

During automatic buffer selection, a qualified challenger already winning a pending comparison keeps its
existing backup to finish the remaining fresh paired wins. It does not release it after qualification and
wait for another full candidate rotation. The original two-second probe deadline still bounds
the visit; retention never restarts that deadline or adds wins without new captures from both
sources. Automatic raw sampling continues refreshing the incumbent between challenger samples.
Losing the advantage, completing selection or timing out returns to ordinary probing. This is
buffer selection policy; it does not switch shader setup modes.

Among similarly complete and coherent candidates with equal active-resolution coverage, a
consistently broader histogram is an additional preference. Samples at exact zero/one are counted separately so a cleared background
cannot create artificial near-to-far breadth. The histogram uses fixed bins of
`log2(d / (1 - d))`, computed with double precision from raw depth strictly between the endpoints.
Reversing the depth convention reflects this coordinate; candidates are never independently
stretched to fill the histogram. The coordinate is a dimensionless distribution comparison,
not recovered camera distance or a guarantee that projections from different passes match.

Both a wider robust percentile span and additional substantially populated bins are required
for a histogram-driven switch. Three fresh captures and conservative incumbent history still
apply. Histogram preference cannot compensate for materially missing scene coverage or poor
coherence, or demote useful higher-resolution depth when completeness is comparable. A qualified
higher-resolution source can replace a broader lower-resolution source after fresh paired
confirmation. Globally anchored completeness classes prevent repeated small support losses from
buying resolution; a promotion cannot fall below the incumbent's protected completeness class.
A narrow-range scene remains eligible; no rule requires all valid scenes to contain
both nearby and distant objects. Spatial checks remain necessary because noise can also have
a broad histogram.

This is content-based selection, not a hook into DLSS's tagged inputs. The selected buffer can
match the game resolution, and neither low draw counts nor a smaller size alone earn preference.
See the [selector installation and validation instructions](../tools/reshade/README.md#automatic-scene-depth-selection).

The original buffer list and CLEAR controls remain available. **Depth buffer** identifies
the actual shader binding and whether it was selected automatically or manually; row checkboxes
are manual overrides, so unchecked rows do not mean no buffer is active. **Use automatic buffer selection**
clears that override and enables automatic selection. Unchecking a pinned row also releases
the override, using the current automatic-selection setting. Both actions preserve the current
binding, captured-content evidence and per-source calibration history; they do not restart from
draw-count fallback. Selecting a different manual row also preserves other candidates' history;
the ordinary source-change path updates the binding. Automatic selection still checks current
activity, source lifetime and content, so releasing an invalid source can select a useful
alternative immediately. A different source or raw basis still requires its own calibration.
Buffer selection is independent of the Game 3D **Depth setup** mode.
A manual override takes priority while its native resource renders. With automatic selection
enabled, two seconds of continuous observed inactivity permit replacement probing while the
manual pin remains selected. Recovery clears the pin only when another eligible source renders
in the current frame and has repeated useful captures from after this inactivity began. Cached
positive evidence, flat loading screens and time alone cannot revoke the pin. Resumed rendering
of the original source cancels recovery immediately. Clears alone do not count as scene
rendering: the manual pin remains, but readiness and content probing pause until scene work
resumes or a freshly confirmed replacement qualifies. Raster and mesh draws, plus supported
copies/resolves into the sampled depth subresource, count as activity without inventing vertex
counts. Known compute/ray dispatches, zero-count calls and explicitly empty transfer regions do
not imply writes to the bound DSV. ReShade 6.8 reports D3D12 ExecuteIndirect commands as unknown,
so those retain their existing bound-DSV activity treatment; their command signature is not observed.
Activity is not proof of useful contents or full image coverage; the histogram checks still
apply, and real drawn flat scenes retain their existing handling. If automatic selection is disabled, an
allocated manual source remains pinned. Resource destruction/recreation clears the override
without reusing a saved pointer. Restarting the game or recreating its effect runtime also clears the temporary
override. Saved format/resolution filters and forced CLEAR indices still apply,
so a restrictive filter can exclude the correct source. Sunshine's exporter continues to use
the resulting ReShade `DEPTH` binding. Sparse content samples cannot prove semantic correctness
for every game: different cameras or partial passes can have similar spatial coverage, and a
spatially varying non-scene buffer may still require manual selection. A clear endpoint is not
proof that depth is missing; the selector does not know the semantic role of every sample.
Capture evidence records the rectangle actually copied. Changes to that rectangle invalidate
content confidence without canceling an in-progress candidate merely because tracking started.

## Activation and recovery

The native add-on publishes its completed stereo texture, with the settings overlay composed
into both eyes when it is open. Explicit reference renderers publish their annotated export
texture only after the selected technique has run.
Sunshine accepts it only from the foreground fullscreen game on the captured virtual display.
It validates the process creation time, window identity, GPU adapter, output dimensions, eye
layout and color transfer. A fullscreen image alone does not prove that it is stereo; there is
no visual SBS guessing heuristic.

Before the first compatible export, streamed Game 3D remains an ordinary mono W × H stream.
After stereo activation, Sunshine duplicates the current desktop into both eyes whenever the
publisher becomes unavailable. Alt-Tab, game exit, disabling Game 3D and resolution changes
invalidate the old publication. Unrelated shader reloads and ReShade's global effects toggle do
not disable native Game 3D; reference renderers follow their own effect lifecycle.
These temporary losses keep the packed 2W × H stream;
they do not repeatedly resize the encoder or desktop. Local AR uses the same duplicate-eye
fallback in the glasses' hardware 3D display mode. The add-on and receiver negotiate a new
resource generation when needed, including after Sunshine restarts. Sunshine never applies
its AI warp to an already-warped ReShade frame.

The headset requests widening only after receiving a valid source status for its confirmed
presentation generation and source resolution. The host acknowledges the exact applied geometry
after encoder initialization; the client waits for a fresh decoded frame before changing its eye
interpretation. A failed automatic widening attempt is not retried for every new export or
receiver generation. Explicitly re-enter Game 3D or deliberately change its quality to allow a
new attempt. Automatic FPS following does not clear that failure state.

The first add-on supports a top-level game swapchain window. A game that renders exclusively
into a child window requires further integration. Reloading the entire add-on DLL while a host
holds its mapping also requires restarting the game; ordinary effect/preset reload is supported.

The output must be exactly twice the source width at the same height. If a codec, client or
configured encoder cap cannot carry that raster, choose a lower normal game/source resolution
or a suitable codec. Sunshine rejects the incompatible mode instead of silently reducing the
authored eyes. Native sharing support is required; there is no CPU readback transport fallback.

## GPU handoff contract

`src/reshade_bridge_protocol.h` is the versioned ABI shared by the add-on and Windows receiver.
The per-process `Local\\Sunshine3D.ReShade.SBS.<PID>` mapping contains metadata and three slots.
Streaming protocol 2 adds `cursor_plane_flags` and `ui_parallax_uv` in eight bytes of each slot's
former padding, preserving the 64-byte slot, 120-byte metadata and 384-byte shared-state layouts.
The signed scalar is one eye's displacement in source-eye UV, with finite values in `[-0.04,0.04]`.
Only zero flags or `cursor_plane_present` are valid; zero flags require a zero scalar. The current
host accepts protocol 1 as well, ignores its unspecified padding and uses zero cursor displacement.
An older protocol-1 host rejects protocol-2 exports, so this feature requires a paired host/add-on
update; it does not change the separate streamed Game provider negotiation version.
The mapping uses Windows' default access control. Sunshine verifies the process creation time
before duplicating its NT texture and fence handles; resources are never looked up by a global
texture name.

The producer publishes immutable metadata with a seqlock. A new consumer nonce requests a fresh
generation of textures and a producer-ready fence. Each slot's atomic 64-bit control word binds
its generation and ownership state, so an old consumer cannot unlock a replacement ring.
Sunshine detaches whenever the game stops being the covering foreground window and then clears
its own nonce (compare-exchange, so a replacement receiver's nonce survives). Returning to the
game therefore publishes one generation, for the reattached receiver's new nonce. Previously the
stale nonce was answered first, so each return built two 7680x2160 generations, each a 24-120 ms
present-thread step in Hogwarts Legacy, and discarded the first.

A new generation normally allocates a new ring. When the previous ring has the same source
(size, format, color, adapter, window), belongs to the same runtime and swapchain, has no
unfinished producer work and has been inactive for at least 500 ms, it is reused: the generation
number and slot states are new, and the receiver reopens the same shared texture and fence
handles. Any read an abandoned receiver left in flight finished long before that, so this only
removes the allocation from a return to the game or a reconnecting stream. A receiver that
restarts while the game keeps exporting still gets a fresh ring, as does any size, format or HDR
change.

For each exported frame, the producer claims a slot, writes the final stereo image into it,
signals its GPU fence, writes the frame sequence, QPC timestamp and matching UI displacement,
then marks the slot ready. Native Game 3D renders both eyes in one pass
(`SunshineRenderPackedPS`) directly into the claimed slot, so the common path writes no eye
intermediate and has no full-frame copy. Each half branches on its eye so the compiler folds the
pixel-center scale into the warp exactly as the former per-eye pass did; the output is
byte-identical to the former two passes and to the frozen FX reference. At 4K this drops two FP16
eye textures (~133 MB) and their write/read-back. While the ReShade overlay is visible or a
diagnostic dump is armed, the pass writes the renderer's own SBS image instead, which is then
copied into the slot with any overlay composition; reference FX exports always use that copy.
Without a consumer, a free slot or an armed dump, no eye is rendered at all. The shader declares
`SUNSHINE_PACKED_EYES`; a dump whose embedded shader predates it replays with the former eye and
pack passes.
Slot render-target views are cached for one export generation. The slot rests in the shared
common state between owners, and the pass reads only the renderer's working set, which the
renderer's completion fence already retains. Up to three submitted GPU writes may remain unfinished, bounded by
the existing three-slot ring. A recorded slot write must receive its submission fence before the next
is admitted; there is no global requirement to wait for the previous GPU write to complete.
Admission checks each slot's own fence sequence, including slots already marked free by a
consumer that discarded them. Only free or unconsumed-ready ownership can be claimed, and only
after that slot's previous GPU write completes. Reading slots remain unavailable. If every slot
is busy, the add-on drops the publication and returns to the game without waiting.

Each slot retains its native source resource and a separate overlay compositor. They cannot be
replaced while that slot's copy/composition remains in flight. Reload, deactivation and runtime
destruction withdraw publication but retain source, destination, overlay and fence lifetimes
through GPU retirement; an unfinished callback does not make recorded work safe to destroy.
The opt-in `reshade_exporter_tests.exe --async-ring` regression delays GPU completion across
three publications, checks fourth-frame backpressure and ownership transitions, and verifies
exact copied pixels and source lifetimes.

Sunshine claims the newest completed slot and copies it into a private texture. All subsequent
conversion and presentation read that private texture on the same D3D11 context. A GPU query
releases the shared slot only after the copy completes. There is at most one pending receiver
copy, no cross-process GPU wait, and no texture overwrite while either side uses the slot.
Receiver teardown abandons an unfinished read. A new consumer requests fresh resources rather
than reusing the abandoned generation. The receiver validates and copies the scalar while it owns
the same reading slot as the texture. Invalid cursor metadata rejects that publication.
Repeated presentation retains the original frame timestamp and matching UI displacement.

## Streamed Game provider contract

Game provider v1 is negotiated alongside atomic presentation v2. Wire mode `2` arms discovery
with ordinary W × H encoding; mode `3` uses the exact 2W × H final SBS texture or duplicate-eye
fallback. Wire mode `1` remains Sunshine AI. Game sessions start in mode `2`, including reconnects;
mode `3` is entered only through the acknowledged live transition. Source resolution and absolute
input coordinates remain W × H in both modes.

Source status carries the applied presentation generation, a source revision, availability,
provider identity and exact source/packed dimensions. Readiness revisions follow validated
publisher identity and availability changes, not every frame sequence. The client accepts only
status for its confirmed generation and source geometry, with ordered revisions. Old converter
status cannot authorize stereo in a replacement session or after a source-size change. The
receiver polls on the existing D3D owner while the captured desktop is static, with one pending
private GPU copy and no producer-fence wait.

## Color and HDR

Windows HDR being enabled does not determine the game's rendering color space. SDR games can
run on an HDR desktop, native HDR games can render scRGB or HDR10/PQ, and Windows Auto HDR is a
separate conversion feature. See Microsoft's [Windows HDR settings](https://support.microsoft.com/en-us/windows/hardware/display-graphics/hdr-settings-in-windows)
and [Auto HDR explanation](https://support.microsoft.com/en-us/windows/hardware/display-graphics/use-auto-hdr-for-better-gaming-in-windows).

The native add-on declares and validates the actual game swapchain color space:

| Game backbuffer color | Shared stereo texture | Transfer supplied to Sunshine |
| --- | --- | --- |
| SDR sRGB | RGB10A2 | Encoded sRGB |
| Native HDR scRGB | RGBA16F | Linear Rec.709/scRGB, 1.0 = 80 nits |
| Native HDR10/PQ, Rec.2020 | RGBA16F | PQ decoded and converted to linear Rec.709/scRGB, 1.0 = 80 nits |

The HDR export preserves values above 1.0 and negative scRGB components. It applies no tone map
to the shared texture. The normal game window retains its original color encoding. A float or
10-bit format alone is not evidence of HDR: the native renderer uses the swapchain's declared
color space and recreates its programs/resources when it changes. Reference FX must supply
matching color-space annotations; a stale reference shader permutation is rejected.

SunshineGame3D decodes HDR10 texels into one full-resolution
FP16 surface before stereo color sampling. Interpolating PQ codes and decoding afterward
darkens mixtures even when constant-color tests pass. Native scRGB requires no decode surface.
The earlier HDR AXAA wrapper corrected contrast normalization and applied the low-pass weight
consistently to all three linear channels; it is historical because current Game3D has removed
final image AA. Decode-before-filtering remains active. The original SuperDepth3D remains a
separate comparison baseline.

Display gamma preference is a separate operation. For example, Gloam 1.9.1 reconstructs
the SDR code from Windows' assigned luminance and substitutes a selected power curve:
`L_out = W * sRGBEncode(L_in / W)^gamma` within SDR white `W`. With gamma alone it leaves
HDR values above that white unchanged. See its [versioned implementation](https://github.com/halideworks/gloam/blob/v1.9.1/src/Gloam.Core/LutGenerator.cs).
The correction acts on display channel intensity, not app identity, so native HDR shadows below
that white can also change; it does not classify the desktop's source content as SDR or HDR.
Sunshine does not apply this preference automatically: the raw game export precedes the
monitor's final correction, and native HDR must not receive a blanket gamma-2.4 transform.
An HDR desktop can contain SDR UI and HDR game surfaces together; its output mode does not
identify which transfer an individual source intended.

Overlay composition uses a separate FP16 layer with reliable alpha. It blends in linear Rec.709,
then encodes SDR output back to sRGB; HDR output remains scRGB. The overlay follows ReShade's
HDR UI brightness while leaving the underlying game's highlights and negative components intact.
Opening the panel does not tone-map the game or alter either eye's geometry.

Sunshine converts the declared input to the destination's color space at final presentation or
encoding. HDR output preserves HDR; SDR output uses tone mapping for HDR input. SDR input on
an HDR output uses the Windows SDR white level when available. This does not invent HDR
highlights in an SDR game. For streamed HDR, enable HDR for the client session and use HEVC
or AV1 with a compatible decoder and display. Local glasses must themselves support HDR in
their current display mode; the Windows HDR setting on another monitor does not establish that.

Native Game 3D reads the game backbuffer before unrelated ReShade effects; reference FX exports
are read after their technique. Auto HDR or driver processing applied after the corresponding
capture point is not part of the export, so an HDR desktop does not prove that the
shared texture includes those enhancements. Native scRGB and PQ input are the supported HDR
paths; actual Auto HDR integration requires separate validation.

## Validation and remaining device check

The dated milestone evidence below is historical and does not establish acceptance of later
automatic-only cleanup or jitter changes. Current live testing still needs to assess deep-scene
foreground halos at matched actual stereo separation, with FG both off and on.

On 2026-09-13, SunshineGame3D passed the actual-runtime milestone checks. The rename-only
version produced byte-identical final pixels against the original on the tested 1080p scRGB
scenes with AA on and off. After the HDR fixes, AA-off pixels and measured disparity endpoints
remained identical; the intended AA color changes passed independent checker and exposure
oracles. HDR10 interpolation failed on the original and passed after decode-before-filtering.
Full D3D11/D3D12 rendering, 4K scRGB/PQ, the production automatic selector, and the named
producer's SDR/scRGB/PQ receiver/overlay tests passed. The installer preserved the existing
game values and installed the new effect into both test games with backups.

The local evidence is `cmake-build-relwithdebinfo/game3d-milestone-20260913/review.md`.
Physical game/headset acceptance of halos, moving silhouettes and colors is still pending.
The remaining evidence below also describes earlier independent shader and transport checks;
it is not a substitute for that live acceptance.

Native tests exercise protocol rejection and the real D3D11 shared-resource/fence receiver with
separate devices. Independent shader fixtures load the packaged SunshineDepth3D shader
in the official ReShade 6.8 D3D11 and D3D12 runtimes. They inspect the actual full-SBS
texture and native mono presentation. The standalone add-on build and interop smoke tests
are documented alongside its source.

HDR checks exercise the independent renderer with native scRGB and PQ input, including negative
scRGB components, 100/1000/10000-nit reference colors and Rec.2020 primaries. Receiver tests
verify bit-preserving FP16 handoff and color changes across resource generations. Final-output
GPU tests check SDR white in PQ/scRGB, tone mapping for SDR output and per-eye chroma filtering.

Controlled D3D11 and D3D12 applications also exercise the official ReShade 6.8 runtime, exporter
and production Sunshine receiver together in SDR, scRGB and PQ. They verify shared and private
texture pixels, source timestamps, effect/reload/focus invalidation, recovery and receiver
restart. Their separate test add-on supplies foreground observations for a hidden fixture window;
the shipping add-on keeps the real Windows foreground check. Additional runs use a distinct
receiver process and exercise the real process-handle duplication, texture/fence import and
private GPU copy. All six SDR/scRGB/PQ and D3D11/D3D12 cases passed, including restart into a
new receiver process. This establishes sharing under the same user/session/integrity level;
the installed game's interaction with the elevated host still requires live validation.
See the [add-on test instructions](../tools/reshade/README.md#diagnostics-and-validation).

The overlay update passed all six Direct3D 11/12 and SDR/scRGB/PQ cases with that separate-process
receiver. The official runtime drew its native controls plus deterministic opaque/translucent
test patches. Both eyes retained distinct game pixels, matched UI positions and correct linear
alpha blending, including HDR highlights and negative components. Live uniform changes advanced
source timestamps without replacing the generation. Close cleanup, shader reload, focus loss
while open and receiver restart were also checked. These are controlled runtime tests, not a
physical headset or installed-game acceptance result.

The independent fixtures bind synthetic raw native depth and compare measured eye shifts with
an analytical constant-plane projection. They require distinct positive Strength responses,
Screen plane sign reversal, equivalent normal/reversed-Z output, FP32 precision near one,
an unwarped diagnostic invariant to artistic controls, and full-resolution one-column foreground
coverage. A deliberately fractional boundary must exercise reversible stereo-edge antialiasing.
Missing capture and missing calibration return duplicate source eyes. Native mono stays byte-exact.
Invalid anchors, gains and capture rectangles also return duplicate source eyes, then recover
when valid inputs return. Poisoned padding checks verify the active viewport's exact mapping;
two-color occlusion tests require foreground ownership and background-only, zero-confidence fill.
Mixed-color boundary tests require that foreground contamination in background-edge texels is
not replicated across a gap, while a separate one-column foreground retains its projected
position and contrast. Both mirrored directions, active depth footprints and stereo-edge AA
are exercised independently of the exact donor-selection implementation.
Tilted-depth planes are checked against a closed-form inverse projection and native color
samples filtered in linear light. Three overlapping depth layers must select the nearest surface,
including a deliberately incorrect zero-disparity fixed point at the destination coordinate.
The actual PerformanceMode1 check confirms calibration/readiness remain dynamic even when
ReShade specializes the user's artistic controls. Automatic uniforms omit explicit initializers
so they cannot be converted into fixed specialization constants.

The automatic-calibration fixture uses the production DLL, real DSV drawing, a busy flat decoy,
and a lower-resolution scene in an offset viewport. It does not inject depth or calibration.
It verifies source contents, direction, fresh calibration, the exact rectangle and exported
diagnostic pixels; a color-only frame, resource recreation, and an offset-only layout change
exercise invalidation and recovery. These checks complement the existing broader selector tests.
Its statistics mode disables content-based Auto-select and verifies that the eligible buffer
chosen by draw statistics still receives calibration, without a manual override.
The separately supplied full-pipeline fixtures can select SuperDepth3D or SunshineGame3D.
Their source is not packaged with the add-on. The [add-on guide](../tools/reshade/README.md)
describes the explicit effect selector and baseline comparison workflow.

The physical acceptance check is a game on the virtual display: verify depth-buffer selection,
stereo comfort, frame pacing, overlay use, Alt-Tab, effect reload, resolution change and host
reconnect on the glasses/headset. Automated shader and IPC checks do not establish game
compatibility or headset smoothness.

Original color plus a full-resolution depth texture for warping inside Sunshine is future work.
This version transfers only the final stereo color texture. Movie SBS image detection is separate
work; the Game provider protocol does not infer stereo from captured images.
