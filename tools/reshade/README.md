# Sunshine 3D ReShade add-on

One add-on selects scene depth, runs Sunshine's displacement conditioning and inverse warp,
and shares its stereo texture with Sunshine. Game 3D requires no external ReShade FX files.
The game stays at its normal resolution. Sunshine consumes the separate full-resolution
left/right image for its local glasses presenter or Game 3D stream. Follow the
[host setup and handoff contract](../../docs/reshade-sbs.md) for the complete workflow.

The Game 3D panel keeps **Enable Game 3D**, **3D strength**, and **UI protection** above its tabs.
Strength defaults to **50%**; **Reset** stays in place and is disabled at the default value.
The UI protection row's **Forget** clears the UI sources Auto accepted for this game.
Existing saved values, including zero strength, are preserved until an explicit edit/reset.
**Status**, **Troubleshooting**, and **Calibration** separate live information from source/preview
controls and detailed measurements. Wrapped status text stays in its own scroll area and cannot
move the controls. See the [panel guide](../../docs/reshade-sbs.md) for the workflow.
Gain and zero placement are fully automatic. Sharpening is removed.
Depth setup is automatic: source-associated camera data or the explicit raw-depth
assumption controls preparation. Manual shader setup, per-game geometry/weapon profiles and their
legacy processing branch have been removed. Manual **depth-buffer pinning** remains available;
it selects the source and does not select another renderer. Image controls live in the add-on panel
and save automatically in ReShade.ini. The old ray-search renderer, its quality settings and the experimental warp
selector have been removed. Sources larger than 3840 in either dimension show an explicit warning
and remain 2D; there is no hidden legacy renderer fallback.
Automatic UI discovery considers authenticated UI opacity, tag 23 alpha, the game's offscreen UI
layer, tag 53 backbuffer alpha, current color alpha unless FG is known enabled, then paired tag 2
HUD-less color difference.
It validates candidates on the GPU every frame and requires no review or approval. Raw Streamline
tags are first translated through the hooked interposer's version-specific
[buffer contract](../../docs/reshade-sbs.md#streamline-buffer-type-contract): UIAlpha is tag 68 in
2.11.x and 69 from 2.12.0. It is automatic inside the surveyed header range and manual-only
outside it; lower candidates remain available either way. Manual On remains an eligible-source override
and Off is absolute. Typed private copies preserve the original channel and format through dump/replay.
With Frame Generation, source-alpha UI protection distinguishes missing real-input alpha from
an observed input whose capture was rejected. It requires a valid submitted input mask with
authenticated queue ordering; generated
output alpha is never substituted. Live UI snapshots and Streamline optional dump resources
preserve the declared SL v2 lifetime and choose the same state policy. `OnlyValidNow` uses the
at-call policy, which prefers a clean,
supported nonzero state observed on the same command recording. Only a completely absent state
entry permits fallback to an explicit supported nonzero provider declaration with compatible
resource flags and declared-state proof. Blocked, unknown, COMMON/zero, split, lost, invalid or
render-pass state still rejects capture; a source requiring observed-state proof cannot fall back.
`ValidUntilPresent` and `ValidUntilEvaluate` keep strict declaration/proof checks; copying at
the tag call does not rewrite their lifetime. Unknown SL v2 lifetimes reject capture, while
SL v1 without a lifecycle keeps its existing synchronous behavior.
The copy restores the selected state; shared dump storage stays immutable and local retired
storage may be reused. Depth captures retain their existing declaration/proof checks. NGX depth
uses the SDK's documented `NON_PIXEL_SHADER_RESOURCE` input contract only when the evaluating
recording has no state entry for the resource; an observed state still takes precedence. UI protection offers per-game, persistent On / Off / Auto choices. Auto rejects invalid
alpha, and empty or nearly full alpha unless the game's channel has earned trust (remembered per
game), and tests the next candidate in the same render. HUD-less comparison
requires a matching queue/presentation generation and spatial scene-agreement checks. Capture
works with FG off or unknown; there is no startup deadline or confirmation timer. The panel
separates source availability, automatic validation and actual application;
RGB remains current and real-input reuse remains bounded. The declaration fallback restores the
original declared-source trust, without independent validation, and its live game/headset
acceptance is pending. See the [UI protection contract](../../docs/reshade-sbs.md#setup).

Game3D maintains independent stereo gain and screen-plane placement, as described below, with the
Sunshine renderer, export, HDR handling and stereo settings overlay. Changes are checked
against the actual shader and independent color/geometry oracles; controlled runtime tests do not
replace live game/headset acceptance.

Automatic uses an associated camera projection when available and otherwise assumes an
infinite far plane for relative depth. The experimental motion-based calibration has been
removed, including its extra capture, fitting worker, controls and build option. The ordinary
depth sampler and shared gain/zero-plane policy serve both remaining paths.

The implementation supports **64-bit Windows games, Direct3D 11 or Direct3D 12, SDR and HDR**, and
ReShade **6.8.0 with full add-on support**. It requires native NT texture sharing and shared fences
on the same GPU as Sunshine. OpenGL, Vulkan, 32-bit games and legacy VR exporter modes
are not supported. The game must present to the foreground window, or to a child window of it
that covers the foreground window's client area exactly (see "Child swapchain windows" in
[the Game 3D design](../../docs/reshade-sbs.md)).

## Build

The build requires the header-only `nlohmann_json` CMake package (3.11 or newer) for on-demand
diagnostic metadata. Its license is included in the add-on package.

Build this directory separately from the host. CMake downloads the official ReShade source,
pinned to `v6.8.0`, commit `18deaa52de0c425a78b329e9cb3c497281cd00ec`, and its exact ImGui headers
at `3912b3d9a9c1b3f17431aebafd86d2f40ee6e59c` (19250). Both are used only through ReShade's
public SDK/function table; no second ImGui runtime is linked. An offline build may pass
`-DRESHADE_SDK_DIR=<checkout-of-that-commit>` and `-DRESHADE_IMGUI_DIR=<pinned-imgui-headers>`.
The build checks the ImGui header hashes because callback structures are part of the ABI.
MinGW UI code must also avoid ImGui helpers returning `ImVec2` by value across that
MSVC function table, including `CalcTextSize` and `GetContentRegionAvail`. Use scalar
returns/style references and ImGui's negative item widths instead. Matching header
versions alone does not make aggregate-return calling conventions compatible.
The camera-metadata probe also uses the existing static MinHook development library and Windows
Version APIs. CMake locates `MinHook.h` and `MinHook`/`minhook`; no extra runtime DLL is installed.

From the repository root with the existing MSYS2 UCRT64 toolchain:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
cmake -S tools/reshade -B cmake-build-reshade -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build cmake-build-reshade
ctest --test-dir cmake-build-reshade --output-on-failure
```

The add-on output is `cmake-build-reshade/SunshineSBS.addon64`, with `install.ps1` and
`SunshineSBS-LICENSES.txt`. Game 3D GPU programs and depth selection are included in that DLL.
The earlier independent reference shader at `Shaders/SunshineDepth3D.fx` is for explicit
reference testing only; normal installation does not copy or enable it.
The MinGW build links its C++ and thread
runtimes statically; no MSYS2 runtime DLL needs to be copied into a game directory. The build also
supports an existing MSVC x64 development environment (`-G Ninja`, or an x64 Visual Studio
generator); that compiler has not been exercised in the local validation.
The default build stages the shipping files in `cmake-build-reshade/package`; test DLLs
and obsolete files left in an existing build directory are not part of that package.

## Install and enable

The host bundles its matching add-on. While Game 3D is active it detects the fullscreen game and
offers setup on its Game 3D setup card; nothing is written into a game folder until the user
confirms that executable, and Sunshine never installs ReShade itself for a game with kernel
anti-cheat. Where ReShade is already installed only the add-on is offered, and an installed
add-on is kept matched to the host automatically. ReShade downloads are verified and cached for
reuse, fresh installations take effect on the next game launch, and existing add-on updates wait
for the game to exit. See [setup, consent and its discovery limits](../../docs/reshade-sbs.md#setup).
The instructions below also cover manual setup and standalone add-on development.

For a troubleshooting capture, use the client's **Dump 3D** action while in Game 3D. See the
[Game dump contract](../../docs/reshade-sbs.md#dump-3d-diagnostics) for the lossless artifacts,
SL/NGX observations, missing-data reporting and output location.
Dump wire v3 also carries optional UI-related resources, including
the NGX DLSS-G HUDLess, UI, UIAlpha and Backbuffer inputs on the existing direct/fallback NGX
path. Successful SL capture keeps its existing priority; nested NGX textures are not captured again.
An armed dump still queries bounded NGX UI parameters under SL depth ownership and reports
`not_attempted_depth_owned` separately from resource availability. Nested NGX wrappers do not
repeat these optional queries.
These new inputs remain diagnostics until their pixel semantics and pairing are validated.
Every catalog entry is reported, including null or unavailable inputs. Native bytes stay unchanged; full-allocation mask and alpha previews
use fixed black=0/white=1, with separate channels and visible nonfinite values. Optional color
transfer is labeled unknown unless captured metadata declares or explicitly assumes it. These
resources are diagnostics and do not establish UI classification or frame matching. Host and
add-on must both support the diagnostic mapping version.
The required consumed UI artifact is separate: automatic detection exports its resolved
full-resolution `R32_FLOAT` mask as artifact 33 (`ui_source_color.bin`), interpreted through
the red channel. An empty mask remains an enabled but ineffective mask path. Replay consumes
the exact frozen mask without running detection again; `ui_source_detection` and its earlier
asynchronous summary are diagnostic metadata, not permission or replacement pixels.
NGX failures include readable result codes. Successful null and failed query remain separate states.
The dump records the latest scoped live-alpha capture attempt observed at the render in
`replay.source_alpha_capture_attempt`, and optional copies carry `capture_diagnostic` state and
recording provenance. The latest attempt is not proof of a consumed-alpha pairing, and an optional
copy's rejection does not prove the same live failure: request timing and source selection can
differ even though optional SL resources and live UI copies share the same resource-state policy.
Diagnostics retain both input states and identify the selected copy state with `copy_state`,
`copy_state_known` and `used_observed_state`. A declaration fallback has a known copy state and
`used_observed_state=false`; that is provider provenance, not independent state validation.
The linked dump contract defines these separate
evidence scopes; recording diagnostics does not change capture admission or rendering.

The inspector also produces an ordered UI discovery report and a review worksheet bound to
the captured bytes. See [UI source discovery and snapshot qualification](../../docs/reshade-sbs.md#ui-source-discovery-and-snapshot-qualification)
for the new-game workflow, `--ui-report`, `--write-ui-review` and `--ui-review` options.
Snapshot review JSON records optional offline evidence and never changes live behavior.
Normal automatic UI detection requires neither a review file nor a panel approval.

1. Install ReShade **6.8 with full add-on support** for the game's actual executable, then close it.
2. Run the installer from the add-on build/package directory:

   ```powershell
   .\cmake-build-reshade\install.ps1 -GameExecutable 'D:\Games\My Game\Game.exe'
   ```

   The installer installs one DLL, backs up replaced files, keeps Generic Depth disabled,
   enables native Game 3D for a new setup, and removes the old separate depth/probe add-ons.
   Saved SunshineGame3D strength/depth-view values migrate into `[SUNSHINE_GAME3D]` in ReShade.ini;
   existing native settings win, including zero strength or an explicitly disabled renderer.
   If no prior SunshineGame3D section exists, matching SuperDepth3D settings may seed the migration.
   The obsolete SunshineGame3D FX and support files are backed up and removed only from their
   owned `reshade-shaders\Shaders\SunshineGame3D` directory. Original SuperDepth3D files/settings,
   other effects and add-on preferences remain available. Competing stereo techniques are disabled.

   `-ShaderDirectory` is reserved for explicit reference testing with the original SuperDepth3D
   export and matching includes, or the independent SunshineDepth3D reference. Such an installation
   disables native Game 3D to prevent two renderers from publishing together. To return to native
   Game 3D, rerun the installer without that argument, then enable Game 3D in the add-on panel if
   it was previously disabled. Keep reference source files separate from the game's install directory.
3. Start the game and open **ReShade → Add-ons → Sunshine 3D**. Automatic scene-depth selection
   is enabled for a new setup. Manual depth selection remains available if a game needs it.
4. Finish the [Sunshine setup](../../docs/reshade-sbs.md#setup): choose Game 3D on the streamed
   client, or enable the local AR provider for directly attached glasses, and use fullscreen
   at the normal source resolution. Keep the game's HDR setting as intended;
   the exported image declares its own color transfer. The game display itself stays mono.

Game 3D enable/disable, focus changes and a Sunshine reconnect are supported.
Updating or unloading/reloading the add-on DLL requires a **game restart**. The exporter rejects
an already-existing per-process mapping to prevent two writers; Sunshine may retain that mapping
until it disconnects. Opening the panel does not change ReShade presets. Explicit control edits
save automatically in ReShade.ini; legacy VR output is not enabled.
If the DLL is copied manually while Generic Depth is still enabled, it saves the required disabled
entry and asks for one game restart; it cannot remove ReShade's already-registered built-in callbacks.
Use the installer to configure this before the first launch and migrate old separate DLLs safely.

Native Game 3D stores `Strength` (percent, default 50), `DepthView` (0 game, 1 stereo depth,
2 normal depth), `Enabled` (default 1), and `SourceAlphaUIMode` (0 Auto, 1 On, 2 Off;
default Auto) under `[SUNSHINE_GAME3D]` in ReShade.ini. On/Off persists across restarts;
Auto discovers and validates candidates continuously without approval. **UI mask source**
provides optional diagnostic source overrides, including HUD-less difference. Missing or rejected
automatic candidates allow the next eligible source, while Off always disables protection. See
the [UI protection contract](../../docs/reshade-sbs.md#setup) for automatic checks and source restrictions.
Edits save independently of ReShade's shader-preset Auto Save option; no Save button is needed.
Reset affects only its parameter.
The opt-in `reshade_game_present_d3d12_test` draws this production panel in the real
ReShade GUI with collapsed and expanded sections, in addition to checking control
edits and persistence. Model/API tests or Home-tab overlay tests alone cannot catch
cross-compiler ABI failures in the panel's layout calls.
The retired `SUNSHINE_GAME3D_ALPHA_DELAYED_STARTUP_TEST=1` flag now exits with code 2 before
graphics initialization. Current source-alpha and native renderer runtime tests exercise
automatic validation, source revocation, manual modes and exact field/SBS pixels. The native
renderer runtime tests (D3D12 and D3D11) also print the present-thread cost of a 4K renderer
start (`MEASURE <api> startup`) and check every path of its background build: cancellation,
park, device teardown, failure without retry and the present-thread executor
([renderer start](../../docs/reshade-sbs.md#native-renderer-ownership)).
No editable shader definitions are required. Obsolete Game3D/debug definitions are removed from
their saved scopes; shared definitions belonging to other effects remain unchanged. Original
SuperDepth3D controls and includes remain in their separate reference installation.
The integrated selector supplies a borrowed shader-readable depth view and readiness directly to
the renderer; Game3D does not use the independent renderer's raw median/span calibration.
The game stays mono at normal resolution while the add-on produces a
separate full-resolution SBS export. Use **Normal Depth View** when checking buffer selection.
See the [current renderer contract](../../docs/reshade-sbs.md#setup) for full-resolution conditioning,
supported sizes and limitations, and the [decision record](../../docs/native-stereo-comparison.md)
for historical comparisons. The [independent rendering contract](../../docs/reshade-sbs.md#native-depth-calibration-and-rendering)
is retained for reference tests and is not SunshineGame3D's geometry equation.
The add-on updates that older percentile-calibration cache only when the independent
reference renderer requests it. Selector histogram analysis and Game3D's raw samples
remain active independently. Existing cached references follow the same source, layout
and eligibility invalidation rules; first reference activation needs its own three useful
sample completions instead of unsolicited background calibration.

The sole depth setup uses source-associated Streamline
projection data when available. Otherwise it uses oriented raw depth and labels the conversion
**relative depth (assumed infinite far plane)**. Both use the same 0–100 **3D strength** slider.
The fallback does not claim recovered meters or game units.
When this fallback is active, the panel suggests trying **Frame Generation**, if supported:
some games supply usable Streamline camera data through FG. This is not guaranteed. If FG is
already observed enabled, the panel reports the missing camera data without asking to enable it
again. Initial unknown depth status
does not show this hint, and Sunshine does not change the game's FG setting.

The current production policy trials a contrast midpoint for the scene's zero plane, using the
farthest depth and moments of each pixel's contrast from it. The independent gain still follows
the nearest depth. Protected UI uses mode 5, `display_fraction`: one rigid display-space plane
with five absolute-disparity targets selected from central UI overlap. Placement uses
absolute per-eye disparity with an independent trial ceiling, also bounded by half the current
scene front limit. It prefers screen depth, confirms forward changes, and returns after sustained
clearance. The system cursor follows the final consumed position. The policy observes the
conditioned scene before UI pinning and does not change scene zero, gain or display limits.
The caps can leave foreground nearer than UI; `capped_conflict` records significant unresolved
overlap. The existing local UI conditioner and one-pixel support collar remain unchanged.
Gain and zero retain their existing
initialization, bounded temporal tracking and source lifecycle. Exact flat depth cannot initialize
gain. After initialization, positive flat depth can move the zero while holding gain; all-zero
depth holds both controls. The first instruction screen or scene does not establish a permanent
scale. Moving zero preserves separation between fixed depths in the unclamped field when gain
remains unchanged. No reset or manual gain adjustment is needed.

The [Game3D policy contract](../../docs/reshade-sbs.md#experimental-raw-depth-automation) owns the
trial formula, centered statistics, initialization, temporal limits, compatibility behavior and
final per-eye parallax limit for both camera and raw-depth paths. The trial does not detect the
main subject and remains sensitive to near outliers. With nearest-depth gain adaptation, zero
tracking can reduce an approaching object's apparent pop-out in a scene with several depth layers. Headset
acceptance is pending. Historical midpoint and fixed-reference tests do not establish acceptance
of this trial.

Live adaptive placement reads five bounded conflict counters per tile, with one asynchronous
probe outstanding. Only the central rectangle is scanned; scene disparity is read only for
covered UI. The CPU checks the summed conflicts against both covered UI and central image area,
so a large clear panel cannot indefinitely dilute substantial foreground overlap. Border UI
remains protected at the same global plane. Fixed absolute targets,
confirmed entry and delayed release keep placement independent of scene strength. Missing or
invalid observations hold the position; new eligible scopes start at screen depth. The
[UI protection contract](../../docs/reshade-sbs.md#setup) owns the record layout, trial levels and caps,
thresholds, timing, source admission and remaining limitations. This observation path is
separate from automatic source validation and adds no source-image readback or GPU wait.

Mode 5 uses the exact 16-byte `sunshine_game3d.ui_parameters.v7` contract and requires
`#define SUNSHINE_UI_DISPLAY_FRACTION_PLANE 1`. Its third word is the consumed display fraction,
not inverse depth. Word four selects alpha (`0`) or red (`1`); older v2–v6 dumps retain alpha.
Dump records `word2_role: front_limit_fraction`, the finite fraction or null,
and its raw bits. Replay freezes that word, including intermediate ramp values; it does not rerun
the temporal classifier or its observational probe. Optional `ui_adaptive` telemetry records the
absolute levels, target/applied/required UV, central image area, UI coverage, conflict counts,
both area thresholds, capped conflict, accepted source identities and age.
Current policy coverage/conflict counts refer to the central rectangle and affect placement only.
Separate GPU source-validation passes evaluate alpha and paired HUD-less candidates each frame.
Their bounded asynchronous diagnostic readback (the decision texels) cannot authorize protection; selection
and mask generation remain on the GPU with no full-frame CPU readback or startup deadline.
No new probe artifact is required. `--ui-front-fraction 0..0.75` supplies an explicit offline
fraction and preserves historical 75% renders despite the lower live ceiling;
`--ui-plane adaptive` is invalid without history. Invalid captured fraction words retain
their bits and exercise the shader's screen-disparity fallback.

Replay's `--ui-inverse-depth` selects historical mode 1 with an explicit depth. Modes 0 (screen/v2),
1 (inverse-depth plane/v2), 2 (nearest covered depth with a submitted floor/v3), 3 (maximum front
display limit/v4) and 4 (fixed quarter-limit/v5) retain their replay behavior. Mode 2 alone
dispatches the nearest-depth tile/reduction passes. Mode 3 requires
`#define SUNSHINE_UI_FRONT_LIMIT_PLANE 1` and ignores its inverse-depth word; it is available via
`--ui-plane front-limit` with a compatible shader. Mode 4 requires
`#define SUNSHINE_UI_SHALLOW_FRONT_PLANE 1`, ignores its inverse-depth word, records the required
`front_limit_fraction: 0.25` metadata, and is available via `--ui-plane shallow-front`.
Modes 0 through 4 are historical replay choices rather than the live adaptive policy.
The 80-byte geometry `b0` ABI and default numeric parallax limit are unchanged.

The sampler covers every texel in the active depth crop, including hardware endpoints, without
percentile trimming. It obtains tile extrema, then scans each tile again for centered moments
within the same dispatch. Resources, readback and queue submission remain unchanged; the extra
texture traversal and reduction add GPU work, so this is not a performance claim. Old uncentered
moments remain diagnostic. All live geometry samples provide the centered statistics; the
historical midpoint fallback exists only for fixtures lacking that payload, and an invalid
supplied payload cannot use it.
The native-depth adapter converts its field to bounded source-UV displacement before Sunshine's
vertical/horizontal conditioning and inverse. It does not run the removed ray reconstruction or
depth-history preparation passes. The [canonical Host contract](../../docs/host-sbs.md) owns the
conditioning and inverse.
Game3D does not force screen-edge parallax to zero. Basic source-coordinate and
sampler clamps remain; missing offscreen color can therefore repeat the nearest image edge.
Readbacks are asynchronous, so newly appearing geometry can precede a range update. The renderer's
per-pixel parallax limit remains active and scales with user strength and stereo reentry.
These are artistic renderer limits, not a validated comfort standard or recovered camera baseline.
There are no per-game presets or hidden percentile calibration in this Automatic conversion.
The raw fallback still does not recover an unknown depth offset or physical distance. Matched
multiplicative inverse-depth representations preserve ratio geometry; separate histories or custom
encodings need not match. No percentile clipping is applied to rendered depth.

The always-visible calibration table shows **Nearest reference Q**, **Farthest (q min)**,
**Zero plane q0**, **Stereo gain K**, **Reference midpoint q**, and **Normalization L**, with Current/Target columns.
Q and the farthest value are the same accepted measurement's full active-depth extrema, not camera
clipping planes. These depth values share one inverse-depth coordinate: larger is nearer.
Gain shows the applied value and nearest-depth-derived target; q0 shows the applied zero and
contrast-midpoint target. **Reference midpoint q** shows the independently tracked historical UI depth and
its extrema-midpoint target. This tracking continues for diagnostics and historical depth-mode
replay; live adaptive placement consumes neither value.
The [UI protection contract](../../docs/reshade-sbs.md#setup) owns its displacement, timing,
occlusion limitations and replay constants. UI uses an independently resolved fraction of the
current display budget, not recovered physical distance; strength, stereo blend and readiness still apply.
A positive flat scene has a zero target but no gain target. L is captured for
the render decision's output shape, and `1/K` is not Q. The panel also shows the per-eye parallax
limit as a percentage of source-image width at the current slider strength.
Camera mode additionally shows current and target **Zero-plane distance** (reciprocal inverse depth
in game units, infinity at zero),
and the table retains **Conversion scale** and **Conversion offset**. No additional GPU work is
requested for these rows. Dump 3D includes its captured scene decision's accepted measurements with
their units, diagnostic moments, centered statistics, pixel count, tile layout and zero target;
a missing measurement stays null even if
the UI displays older values labeled as held. The UI also shows the accepted pixel count and layout.
Without a matrix, the conversion is labeled **relative depth (assumed infinite far plane)**
and the zero row displays `t0` in relative raw units. A visible status identifies gain below
its target. The matrix line shows the shader's affine conversion `q=scale*raw+offset`,
including raw packing. It is derived from the checked projection matrix, not estimated from
scene content. Held values are marked as last applied and paused, with target columns cleared. The panel exposes no recenter
or manual gain control; zero placement and gain adaptation are automatic on both paths.
SDK history resets discard stale measurements while retaining established gain and plane until
fresh observations resume.
Same-source pin/unpin retains generic state; a real generic source change initializes from fresh
captures. Retained depth allocations have independent bounded controller
state; unrelated or destroyed sources do not inherit an old calibration. Camera and raw branches
also keep separate screen-plane histories. Switching representations is not guaranteed to be
jump-free, and unknown raw encoding offsets prevent exact projection/raw equivalence.
Capture availability,
current source selection and numerical sample ingestion have separate interfaces. Brief missing
captures return mono without restarting every controller, and initialization tolerates a moving
scene.
Missing depth shows the last applied screen plane as held. Source-associated depth and projection
are required independently of that numerical state.
The [raw integration contract](../../docs/reshade-sbs.md#experimental-raw-depth-automation) describes
admission, waiting behavior and current limits. There is no Manual shader mode or mode reload.

[Streamline depth selection](../../docs/reshade-sbs.md#streamline-depth-selection) is enabled by
default for supported D3D12 games. Any feature can provide depth through validated resource,
frame/view, encoding and lifetime contracts; feature IDs are not an admission allowlist. Automatic
selection tries tagged high-resolution device depth (48), scene device depth (0), then positive
linear view distance (49), ahead of unrelated larger heuristic candidates. Manual pins take
precedence. SL/NGX select the resource; Generic selection is the fallback when no valid API source
owns selection. One D3D12 capture owner handles
both Generic preservation boundaries and API evaluation opportunities, including snapshot storage,
submission, consumer leases and retirement. API capture snapshots the nominated resource at the
middleware call, including tracked depth-stencil resources; matching a Generic backup's resource
identity cannot authenticate its contents. While a live source's successor is pending, acquisition
may advance to that source's unconsumed completed snapshot with its own sequence, metadata,
allocation size and active rectangle under the contract's ordering, freshness and interruption
checks, whether or not it already owns selection. A failed, invalid or refused newest nomination
(also one refused before its capture existed) is a missing frame of its display lineage, taken
the same way and marking no interruption, and so is a nomination still in progress at the Present
(it ends one way or the other, and both take the same copy); only identity-level interruptions revoke earlier
snapshots ([source interruptions](../../docs/reshade-sbs.md#streamline-depth-selection)). A new render size of the same source (dynamic
resolution) neither refuses that snapshot nor ends the hold of the last copied depth; only another
storage format, view or resource kind does (`format_changed`, `depth_format_changed`).
When neither API supplies readable pixels, an established API selection can remain unavailable and
return mono. A live owner that copied a frame within 250 ms keeps its source while readable or while
a successor of that frame is pending. Otherwise readable SL takes priority over NGX, also for an
established SL owner whose frames fail display preparation or copy; an unsupported or pending first
SL copy cannot displace working NGX. When neither provider is readable, that delivering owner also
keeps selection through a failed, rejected (even before its capture existed), replaced or ambiguous
attempt, so another provider's pixel-less capture takes no authority. Expired or revoked owner
evidence, or an owner that has not delivered for 250 ms, permits the other live source; an expired
owner, or a failing one that stopped delivering, never hides a pending live one.
Selection stays fixed within a presentation and never pairs another provider's camera with its depth.
Automatic reconstructs inverse depth from the exact current projection, without smoothing its
coefficients. The independent gain and screen plane follow the measured-range policy above. Current depth and
zero-plane values must fit the shader domain. Unsupported encodings stay mono without clamping depth.
Device depth without a matrix uses the shared raw-depth screen-plane controller with the explicit
infinite-far assumption. Explicit positive linear view distance uses reciprocal depth directly and
does not require a matrix.
The linked contract describes the reference convention and
the limits of comparing unknown raw encodings with projection depth.
Successful evaluation and actual submission establish API source ownership independently of
snapshot readiness; an explicit release, or one second without any evaluation of that source (a
game that switches DLSS to TAA), returns the queue to Generic selection. The linked contract owns
the capture-ordering requirements and limitations;
D3D11 retains its existing Generic preservation backend. AMD support is deferred. Set
`[SUNSHINE_DEPTH] StreamlineDepthSource=0` in `ReShade.ini` and restart to disable this path.
[Queue order](../../docs/reshade-sbs.md#queue-order) observes the game's queue Signal and Wait calls
(and CPU fence signals) so that a copy on another queue can be read before the CPU sees it
complete; `QueueFenceOrder=0` turns it off and keeps the completion rule.

[Direct NGX depth](../../docs/reshade-sbs.md#direct-ngx-depth-selection) is also enabled by default
(`NGXDepthSource=1`) for compatible D3D12 integrations, including Super Resolution and Ray
Reconstruction. Public exports and typed depth parameters determine admission, without a DLL
patch-version or feature-ID allowlist. Unknown contracts remain unavailable. The adapter shares native
capture, retirement, display binding and rendering with Streamline. Device depth without a supplied
projection uses the shared raw-depth screen plane and infinite-far assumption. Explicit linear depth
uses reciprocal view distance. Both preserve their logical source history across rotating textures
and dynamic resolution changes; provider/encoding changes start an independent calibration domain.
The panel identifies the actual NGX/Streamline/Generic path. Read the
linked contract for supported crops, feature lifetime and early-discovery limitations.
Nested SDK calls are deduplicated only after a valid native copy is recorded, or under confirmed
and live same-viewport SL FG authority. Rejected native copies and metadata-only nominations allow a usable
nested NGX source to be captured.
When a confirmed newer NGX capture is waiting only for GPU completion, the owned completed depth
can bridge one native presentation. This holds its exact previous scene controls and never advances
calibration. Unknown, failed, changed or stale input cannot authorize this hold; a second missing
presentation returns to mono. No completion wait or extra copy is added.

When a supported Streamline Frame Generation source is enabled, its depth and associated camera
data take priority over NGX, including while its current snapshot is pending. A missing usable FG
copy stays with that source and uses the documented bounded real-depth reuse or mono behavior;
it does not borrow an NGX depth texture and attach unrelated SL constants. A depth untag (a null
resource, which ends the tag) is not a capture attempt and never withdraws depth already captured
at the frame's tag. FG keeps that priority only for the frames it supplied: while its newest frame
outcome is an accepted depth copy, made since FG On and within the last second. A frame without
one (its FG tag refused, its deferred copy cancelled, refused or lapsed, or DLSS-G's own evaluation
of it refused) lets the same viewport's SR/RR evaluations nominate again, and their depth (or NGX)
replaces FG's once FG's last copy can no longer be held (the source age), until the next accepted
FG copy takes it back; FG On alone, or a single copy, grants nothing more.
Without such a source enabled FG still keeps the game in mono; it never falls back to generic depth.
Explicit FG Off retires that source and lets a fresh NGX submission resume. Motion-vector interpolation of two real depth
frames for FG 2x is future work; the current implementation does not generate intermediate depth.
Reused depth holds its applied gain and zero without learning. User strength reductions apply
immediately, including zero; increases wait for fresh real depth and a new placement decision.

Valid SL/NGX TAA jitter is stored with the captured real depth. The shader converts render-pixel
offsets through the active render extent and depth crop before its existing clamped depth fetch.
Reused FG depth keeps its own jitter rather than adopting the newest pending frame's offset.
Missing or invalid jitter applies zero correction and does not reject otherwise valid depth.
This adds no texture copy, rendering pass or GPU wait. It corrects subpixel registration; it does
not reconstruct hidden background or generated-frame depth, and wide foreground halos remain a
live quality issue. See the [jitter contract](../../docs/reshade-sbs.md#depth-jitter-registration).

## First run of a new game

Every game so far broke on an assumption that held in the games before it: Resident Evil
Requiem aliases heap memory every frame, sends a camera without frame identity and offers a
presented alpha that is not UI; The Witcher 3 moves command lists to per-object method tables and
requests each frame token about 30 times. A first run therefore covers the paths those differences
reach, and the report below names each known failure signature:

1. Play about 2 minutes with FG off and 2 minutes with FG on (if the game has it).
2. Open a full-screen menu and a pause menu.
3. Take one Dump 3D during play and one in a menu.
4. Run the report on the game's log, with the host log when Sunshine streamed the session:

```powershell
python tools\reshade\game3d_log_report.py "D:\Games\<game>" --host-log E:\ApolloDev\config\sunshine.log
```

It reads existing log lines only and prints PASS, WARN or FAIL per check with the times to look
at; the exit code is 1 when a check failed. The checks cover: add-on and renderer readiness;
fresh depth and gaps outside the settle time after FG switches and runtime resets (depth starts at the
first counter interval in which fresh depth and the bounded reuse frame generation shows between real
frames carry most publications: at 4x only a quarter are fresh, and before 10-22 a session that started
with FG on was never checked); publications
that showed the colour frame with depth; capture coverage by the list lifecycle; whether the
camera projection or only the raw controller places the scene (the latter while a valid camera
exists is a warning); UI acceptance changes and Forget; inferred alpha
deciding beside an accepted UIAlpha or UI color tag; an accepted alpha (inferred, and since
selection revision 10 UIAlpha or the UI color tag captured in the pair's tag batch) that a valid
exact HUD-less pair contradicted one way three times within 2 s without a revocation (in logs
before S2a, an accepted source covering the whole frame while an exact pair shows the scene; shorter
contradictions, which A2 never revokes, are noted); and accepted sources that disagree as often,
with the revocation that resolved them named by its kind (by the selection and A2 revocation rules of
[UI protection](../../docs/reshade-sbs.md#setup)); selective UI channels rejected for invalid pixels
on more than 1% of the frame (V1) while no mask protected it (`UI channel admission`);
`UI protection gaps`, the streamed time
outside settle times in which Auto rendered frames without a UI mask (no UI source offered, or no
usable mask while no UI channel was offered clean and empty, which means no UI on screen), so HUD
and menus took the scene's depth, with each window's time and the reason each offered channel gave
no mask, and in logs since S2a the add-on's own reason and refused candidate (gaps shorter than the overlay warning's minimum, defined in the
[UI protection panel](../../docs/reshade-sbs.md#setup), are counted, not listed; a runtime's
`Sunshine depth readiness` loss is left out, from its line to its recovery, and its length named only
when the `Sunshine SBS output` counters around it count as many publications without depth as it had
Presents, so every frame there was flat and no HUD took depth; a loss that never recovered still
counts); how many samples were
full-frame UI over a hidden scene (H1, source 8; in logs before S2b source 8 or 9), by the pre-UI
scene image whose claim acted (HUD-less image or UI layer), with the hidden-scene guard's entered,
released and refuted counts from the last counter line, and how many hidden samples had a pre-UI
layer whose signature was not yet proven the pre-UI scene image (since fix 1 by matching pixels, in
S2b logs by D; in logs before S2b, how many were a
route's exit, whose own evidence read the presented frame visible: each hidden scene that ends
shows one, so they are counted rather than judged); in logs before selection revision 9, a
warning when the removed first-run shadow measured the presented frame hidden, not blank, for at
least 500 ms while no UI source decided (`shadow_hidden_ms`, listed once per run by its start); in
logs since fix 1, `Pre-UI proof` for each layer proof key (`pre_ui:<format>:<space>`) restored,
earned, lapsed or forgotten ([hidden-scene evidence](../../docs/reshade-sbs.md#setup) defines it);
observation losses, capture statuses and export pauses, each judged by its origin (below);
present-thread hitches, named by their step, and, from each timing line, a window whose slowest Present took the hitch threshold
(8 ms) or more with no hitch line in it, named by its slowest section (no step named; before 10-22 the
render section had none): its time within the window is unknown, so it is expected, and listed with
the PASS, when a reset, FG switch, export start or overlay opening lies in that window or within the
settle time before it; Game 3D
CPU and GPU cost over every timing window (means weighted by each window's Presents or GPU frames,
and the largest maximum); `Stream delivery`, the new frames per second the host took (published
minus overwritten-unconsumed slots) against what the game offered (published plus dropped), capped
at the stream rate, which the host log supplies and which follows live 90/72 Hz changes; windows
that start in the settle time are skipped, and one below 85% of that target for 10 s or longer
warns. Without `--host-log` the stream rate is unknown, so the check only reports both rates as INFO
(the `Sunshine SBS output` slot counters are defined in
[Streamline depth selection](../../docs/reshade-sbs.md#streamline-depth-selection)); `UI layer copies`,
per timing window, the offscreen UI layer copies a full ring skipped (`ui_layer={copies skipped ...}`)
judged by what they cost in time, the group's `skip_gap_ms`: the longest time the layer went
unrefreshed across skipped copies, while every Present meanwhile read the older offered copy and was
published like any other. A gap longer than one stream frame interval in a window that overlapped a
streamed span warns (streamed frames could read a UI layer that missed a whole stream frame of UI changes,
however fast the game presented), and so does a gap of 250 ms or more in any window (the layer was no
longer offered at all: a ring that stayed full). Shorter gaps (one skip at an uncapped ~590 clears a
second costs about 3 ms), skips that were not streamed or without `--host-log`, and skips on lines
before 10-07, which log no gap, are INFO. The gap is a maximum, so neither a window's placement nor its
average rates can hide a burst of skips. The once-per-ring full-ring line (a WARN before 10-07) belongs
to this check, not to `Log warnings`, and one after the last timing line is named as uncounted, its
cost unknown; `UI layer targets` (INFO, since C12) counts the qualifying clears the layer tracker
dropped with all 32 of its entries taken (the timing group's `targets_dropped`: the targets cleared
first in a frame, beyond the 32 cleared last); and, from the host log, the Game 3D link,
size fit and encoder stalls (one per stalled frame, classified by its last line and listed with its length and
the nearest of the game's frame-generation switches or runtime resets within 2 s). An upstream stall that
completed within 2 s of such a transition, froze its picture no longer than the encoder's own 250 ms budget, and
whose encoder finished within 50 ms (two 25 ms producer polls) of its input is INFO: the game's own GPU
reconfiguration kept the host's conversion off the GPU, and the encoder's budget starting at its input's
completion lets that delay one picture by design
([GPU handoff contract](../../docs/reshade-sbs.md#gpu-handoff-contract)). Live 10-04 to 10-07 every completed
upstream stall but E33's two came within 0.8 s of one, and all but Stellar Blade's 690 ms at its 10-06 FG enable
lasted 112-209 ms; the add-on's log drains asynchronously, so its line can follow the event. Since 10-22 a
completed upstream stall (encoder finished with its input) that a game `Sunshine Game 3D present gap` covers, both
ends within 50 ms, is INFO however long: the game presented nothing for the whole stall, so it held the whole GPU
(E33 10-22: 229 and 341 ms). The stall names the gap, the FG state before it and a focus loss at its end. Video
memory under pressure at the stall keeps it a warning: the host's own sample on the stall's line, or a game sample
within 1 s, over its local budget or within 128 MiB of it, each named (`host VRAM` or `game VRAM` usage/budget). A
longer stall at a transition without such a gap, one
whose encoder also ran past its input (`upstream GPU + encoder-side`), an encoder-side stall, a timeout or a
removed device or failed wait (`timed out`, `failed`), and an upstream stall with no such transition or gap near it
warn; a focus loss or export start alone explains none, since the host's switch to or from the desktop is its own
work. Host logs since 10-22 add `Host video memory`, INFO: the host process's largest local usage and smallest
headroom on its stall and 20 s encode-loop lines. A log whose game process ended right
after ReShade tore its runtimes down (Unreal games often end before ReShade logs its exit) counts
as a normal exit.

Since 10-23 these checks judge each event by its origin rather than warning at any count:

- `Export`: each `export inactive (reason, class)` line names its class (`export_pause_class` in
  `exporter.cpp`). A `routine` pause comes from the host, the game or the player (`no_consumer`,
  `ring_replaced`, `consumer_transfer_changed`, `consumer_without_pq`, `not_foreground`,
  `runtime_reset`, `runtime_gone`, `present_without_render`, `color_space_changed`,
  `swapchain_unknown`, `frame_not_prepared`) and warns only as a storm: 7 or more within 60 s, more
  than one per 10 s for a minute. Every other reason, a new one included, is a `fault` and warns;
  `device_removed` fails. Lines from before the class tag are classed by the report's copy of that
  list.
- `Capture status`: every status of a depth provider fails except `inactive`, `unavailable`,
  `recorded`, `submitted`, `ready`, `stale` and `ambiguous`, so `unsupported_queue`, `malformed`,
  `exhausted`, `unknown` and a status added later fail by default. A failing status within the
  settle time after an export start, FG switch or reset is INFO, as before.
- `Observation losses` (sampled on `Sunshine depth observation evidence` lines): a capacity loss
  (`tokens_busy`, `records_busy`, `commands_busy`, `resource_table_full`, `metadata_storage_busy`: a
  full bounded table, which a burst of game work explains) warns only by rate: all of them more than
  once per 2 minutes of the session (counted as at least 2 minutes), or one cause more than 3 times
  within 60 s. `lifecycle` and `token_replaced` are the game's own and never warn. `invalid_input`,
  `sdk_failure`, `test_injected` and a cause the report does not know warn at any count. The depth a
  loss cost is the `Depth` check's to judge.
- `Log warnings`: only WARN and ERROR lines the add-on owns warn: its own `[Sunshine 3D] ` (or
  `[Sunshine SBS TEST ONLY] `) lines, except its hitch, display-scaling and full-ring lines that
  their own checks report, and ReShade's lines about it: one naming the add-on or its module, a
  failed add-on load, or a failed effect-runtime creation. Every other WARN or ERROR line (shader
  compile warnings, reference-count notes, DirectInput, another add-on) is counted as INFO
  `Other log warnings`, with one example per kind of line.
- `Host size`: game eyes scaled to the stream's (the same aspect) or fitted into it (letterboxed or
  pillarboxed) are INFO; eyes the host refuses, which stay 2D, fail.

The test's `SchemaMirrors` reads these names from the C++ that owns them (`routine_export_pauses`
and every `deactivate` reason, `depth_capture::status`, `loss_diagnostics::reason` and
`format_ui_counters`), so a reason, status, cause or counter added there fails the test until it is
classified: the report's routine pauses and OK statuses beside the test's own lists of fault pauses
and failing statuses (the report itself still fails an unknown status and warns on an unknown pause
by default). A `Sunshine UI
counters` reason for frames without detection that the report does not know still counts in the
accounting and is listed after the known ones.

Logs that contain `Sunshine UI counters` lines carry the add-on's exact per-frame
[UI counters](../../docs/reshade-sbs.md#setup). For these logs the report decides the UI
invariants from the last counter line (the holds check from every counter line), not from the
100 ms samples. It reports these checks:

- `UI counters`: the add-on's own accounting (Auto frames are detection frames, generated Presents
  that held a real frame's decision or had none, and frames without detection).
- `UI protection`: inferred alpha beside an accepted declared UI channel (`presented_over_dedicated`,
  counter lines through dc7e3c77 only), and the frames in which a
  valid exact pair contradicted a deciding accepted alpha one way (`contradicted`; inferred alpha per frame, since selection revision 10 an accepted UIAlpha, UI color tag, Backbuffer or current alpha (never the one-frame-late layer copy), per sample), which
  fail only with a sampled run of three such contradictions within 2 s that no revocation followed:
  A2 never revokes shorter ones, and the counter cannot tell them apart. Counter lines logged
  before S2a keep the older rule: accepted full coverage of a visible scene (`trusted_full`) that no
  revocation of that source resolved.
- `UI full frame`: H1 samples (8) whose scene read visible, against the hidden-scene guard's
  releases (`scene={entered released refuted}`, since S2b). The sample that releases a held hidden
  verdict decided 8 and read visible, and counts as a release, so this passes while
  `full_d.visible` is at most `scene.released` and warns that an H1 hold acted over a visible
  scene otherwise. Counter lines logged before S2b keep the older rule: each release of a held
  route (8 or 9) counted one, so it warns when such a route decided, and an exact full change-set
  (6) alone is INFO.
- `UI full alpha`: accepted whole-frame decisions, a whole-frame mask from alpha or (since S2b) an
  exact full change-set (6), and, in logs before selection revision 9, whether their samples read
  the scene visible. It is INFO: an accepted source pins its alpha at any coverage, so a full one is
  flat even over a visible scene, as the
  [opacity ruling](../../docs/reshade-sbs.md#ui-decision-framework) intends.
- `UI full frame pairing`: a failure when the whole frame was decided flat (6) in a session whose UI
  capture gate offered HUD-less pairs but never a same-batch Backbuffer pair. Only a same-batch pair
  is exact (rule E2); a Present-counted pair can belong to another frame and then differs everywhere
  (Hogwarts Legacy 10-05, which tagged only `HUDLessColor`).
- `UI inferred alpha`: a failure when an unaccepted inferred alpha decided, which only accepted
  candidates do since S1 (a warning on counter lines logged before S1). Only counter lines through
  dc7e3c77 carry the count, so later logs have no such check.
- `UI inexact difference`: INFO, the frames whose HUD-less difference came from a Present-counted
  (inexact) pair, validated by its own pixels (V2); expected wherever a game offers no same-batch
  pair.
- `UI holds without a decision`: per counter window (from one counter line that advanced to the
  next; windows less than 1 s apart merge), a warning when at least half of the Auto frames were
  held without a real-frame decision to show (T1) for 2 s or longer, each window listed with its FG
  state, and a failure when at least 90% were for 10 s or longer: UI detection effectively never
  ran (Hogwarts Legacy 10-05 at 4x frame generation, before the HUD-less pairing fix).
- `UI holds`, `UI no mask` and `UI trust events`: exact totals. Holds are the generated Presents
  that showed a real frame's decision or had none, and the real frames that reused the held
  decision (T1: once, or across a HUD-less snapshot's re-offers); frames without a mask also list
  the sampled reasons with the
  candidate each refused; trust events include one-way and declared-coverage revocations and
  Forget.

Logs without counter lines use the sampled checks above. The report reads the UI and counter lines
of logs written before S2b, before S2a and before S1 too, each with the checks of its time; a UI
layer logged before S1 is read as the layer candidate. UI lines since S2b carry the pre-UI scene
image's evidence (`sampled_pre_ui_scene`), the informative claims and the H1 word
(`sampled_claims`, `sampled_h1`) and the guard's pushed verdicts and layer proof (`scene_guard`) in place of
`sampled_hudless_scene` and `scene_hold`; UI lines since fix 1 add the layer's pixels against the
presented frame (`sampled_pre_ui_pixels`), and lines without them read as before;
[UI protection](../../docs/reshade-sbs.md#setup) defines them. Lines of removed features still parse
and add no check of their own: the first-run shadow's session line, which is ignored (the `shadow`
and `shadow_hidden_ms` fields of UI lines written before selection revision 9 still feed the
hidden-scene warning above), rule H2's still screens (`still` groups, `decided.11`, `Sunshine
UI still screen` lines), the S3 identity shadow (`Sunshine UI identity`, `Sunshine FG interposers`),
and the dark pre-UI statistics. Fields the add-on logged as 0 are gone from later lines: on counter
lines through dc7e3c77 `full_alpha_d` (0 from selection revision 9) and the S1 invariants
`untrusted_inferred` and `presented_over_dedicated` (0 from selection revision 10); on UI lines
through be7788bf `shadow`, `shadow_hidden_ms`, `presented_lit` and `presented_lit_differs` (0 from
selection revision 9) and the one-way layer column (0 on every line since S2a, which added it: every
layer was the one-frame-late copy, which no judge reads). UI lines from S2a through 180f1842 also
carry `sampled_late_layer`, which only repeated that a layer was offered and is skipped. The
[UI counters](../../docs/reshade-sbs.md#setup) table and the UI line description there own these
ranges.
Logs since 10-22 carry the game process's video memory
([Diagnostics switch and per-Present cost](../../docs/reshade-sbs.md#diagnostics-switch-and-per-present-cost)).
`VRAM pressure` warns for each depth loss, hitch or Present gap line whose own sample put the game's local usage
over its budget or within 128 MiB of it (the OS demotes allocations above the budget, which can stall the whole
GPU), listed with its usage and budget; `Video memory` is INFO with the windows' largest usage, smallest budget and
headroom, and the time over budget, which with nothing visible at it warns of nothing. Older logs have neither.

Each `UI protection gaps` window lists its pieces with each line's own FG state; a line whose
status sample was still pending (`checking` or `searching`), or an unrendered line, continues a run,
labelled with that state after the last sample's reason.

The depth and flat checks decide from the periodic `Sunshine SBS output` counters, then name
the add-on's own evidence for each window. A depth gap lists each `Sunshine depth readiness`
loss inside it, paired with the same runtime's recovery, with its exact start, `unavailable_ms`,
reason, provider and selection (`inactive_views` means the game supplied no depth view). A gap
whose every loss had `inactive_views` and recovered with the game's own declared history reset
(`current_reset=1` on the recovered line: a new scene) is INFO `Depth paused by the game`: the game
ran no DLSS (a menu or loading screen; E33 10-22 leaving its title, 828 and 1265 ms) and every frame
was flat. A loss without that reset, or one the add-on had a depth view for, stays a gap. A loss
that a `Sunshine Game 3D present gap` falls inside ends with `the game froze N ms` (the longest), so a
game hitch reads apart from a loss the add-on caused; it still counts (E33 10-22 15:43:35, 773 ms
while the game streamed in assets). A loss that ends within an FG switch's or reset's settle time
and starts at most 0.25 s before the switch line is the switch's own handover. Render-size changes
of one depth source (dynamic resolution) keep depth since 10-23 and are no gap: INFO
`Depth render size` sums the `Sunshine depth render size` lines (changes, Presents held across a
pending size, the smallest and largest active extent). Before, each change was a
`depth_shape_changed` loss (Assassin's Creed Shadows with FG on: 125-438 ms, six in 6 s). DLSS-G's
own inputs (10-23) are judged from the `Sunshine NGX depth` line's `dlssg_depth`/`dlssg_ui` fields:
INFO `DLSS-G inputs` counts `DLSSG.Depth` FG-scope copies per real-frame DLSS-G evaluation and the
UI inputs captured, or says that Streamline FG tags supplied the depth (`tag_copied`) and how many
UI inputs the game's own Streamline UI tags copied first (`dlssg_ui` `tag_copied`, per kind); WARN names
the refusals when DLSS-G evaluated with no copy (`missing_state`: DLSS-G's command list observed no
state for its depth; `observer_coverage`: ReShade does not cover that list), and when it copied but
refused more evaluations than it copied (other than for a tag's copy): each refused frame turns the
FG scope silent, so FG depth came and went. INFO `Depth losses` counts every loss episode while the
export streamed, whatever the 5 s interval gate above: the logged ones plus those the add-on
counted without logging them (`suppressed_episodes`; it logs at most four a second), their rate
per streamed minute, the logged ones' time without depth, and their reasons. Compare sessions by
these counts: a fix can bring `Depth` to PASS while short losses remain (Assassin's Creed Shadows
10-24: 66 episodes, about 29 a minute). `Placement flat` counts the re-entry fade with the flat
output (`scene_fading`, 500 ms after every placement interruption of more than 250 ms; Shadows
10-24: 15% flat and 19% fading): a window is an interval of which a quarter was flat or fading, and
the summary gives both shares. A flat
window is reported as INFO `Placement calibration` only when every placement-controller run
without placement inside it (raw automation other than `ready`/`holding_reference`, or
Streamline scale other than `ready`) followed a cause the log names (a depth-provider or frame
generation switch, a depth loss, or the export starting) and had depth for no longer in total
than the report's settle time, or, as INFO `Placement waited for the scene`, when the run only waited
for a reference (its states calibrating, waiting for depth or depth unavailable, never an error
state) and placed within the settle time and the log gate of the game's own scene reset inside it:
the depth before the reset gave no reference and the next scene did at once (E33 10-22 title
screens: 20 and 23 s, ready 0.9 and 1.2 s after the game's reset); its length is how long the
player stayed there. A longer or unexplained run, or flat output while placement was
ready, stays a warning that names the controller's state. The add-on's diagnostic log gate can
hold a controller line, so run ends are approximate and a cause may lead its run by that gate.
From `Sunshine SBS: export inactive` until the next `Sunshine SBS: generation` nothing is
streamed, and depth losses there are not counted. The stop's own `explicit_reuse_invalidation`
follows its pause line; add-ons before 10-23 logged it just before, in the same millisecond, and
the report counts that one with the pause too.

A failing check is a bug report: fix the rule at its root (see
[the generalization rules](../../docs/reshade-sbs.md#rules-for-new-game-behaviour)), then add a
check here when a new signature appears. Its tests run with the add-on's other tests when CMake
finds a Python 3.9 interpreter: `reshade_game3d_log_report` checks each signature on synthetic
lines, and `reshade_game3d_log_report_golden` compares the report's exact output on saved
first-launch logs of real games with its golden in `log_report_goldens/`. Both also run directly,
as `python tools/reshade/test_game3d_log_report.py` and `test_game3d_log_report_golden.py`. When
a report change moves a golden on purpose, rewrite the goldens with
`python tools/reshade/test_game3d_log_report_golden.py --update` and review their diff in the
commit. The repository is public, so a new game's log enters only through
`scrub_game3d_log.py <name> "<game>" <ReShade.log> [--host-log sunshine.log]`, run on the saved
log with its original modification time. It replaces install, user and other drive paths, names
and addresses with placeholders and keeps only the session's host lines. The machine's own user and
computer names are replaced only in that scrub, so the golden test and `--rescrub` give the same
result on any machine. The golden test fails on anything left to scrub; the raw logs are not
kept, so a new scrub rule reaches the committed fixtures through `scrub_game3d_log.py --rescrub`,
and their goldens should not move.

UI decision changes have two offline gates; [UI protection](../../docs/reshade-sbs.md#setup)
owns their contracts. The first is the single-frame replay of the labelled dumps. Run it from
`cmake-build-relwithdebinfo/reshade-addon`:

```powershell
.\ui_detection_replay.exe ..\..\tools\reshade\game3d_native.hlsl E:\ApolloDev\sbs_dump\ui_detection_cases.json
```

It ends with
`PASS|FAIL UI detection replay: N passed, X xfailed, Y xpassed, F failed, S skipped`. A known-wrong
cell is XFAIL while it keeps today's recorded outcome, and XPASS once it meets its target.
`--strict` also fails on an XPASS, so that its xfail gets removed.

The second gate is the sequence test of the temporal rules (trust, holds, the hidden-scene guard).
It runs with the other tests:

```bash
ctest --test-dir cmake-build-relwithdebinfo/reshade-addon -R reshade_game3d_ui_sequence --output-on-failure
```

Its `KNOWN_TODAY <stage> <rule>` lines name today's known-wrong outcomes by
[UI decision framework](../../docs/reshade-sbs.md#ui-decision-framework) stage and rule ID, and
its `KNOWN_LIMIT <rule>` lines name known limits that no stage changes. Neither fails the test. It
ends with `PASS UI sequence replay: <groups> groups, 0 KNOWN_TODAY, 2 KNOWN_LIMIT`, both limits of
Present counting under a misreported frame-generation multiplier (T1/E2); the
[UI protection](../../docs/reshade-sbs.md#setup) contract gives the current group count. A
replay `xfail` reason names the same rule IDs.

## Additional diagnostics

`[SUNSHINE_GAME3D] Diagnostics=1` (the **Diagnostics** row under Troubleshooting) turns on the
add-on's diagnostic-only per-frame work: per-pass GPU timing and the Streamline presentation
brackets with their PCL marker hooks
(see [Diagnostics switch](../../docs/reshade-sbs.md#diagnostics-switch-and-per-present-cost)). It
is off by default and changes no decision, candidate binding or exported pixel.

The separate [Streamline camera probe](../../docs/reshade-sbs.md#streamline-camera-metadata-experiment)
is disabled by default. `StreamlineCameraProbe=1` enables detailed diagnostics after restart;
leave it off for ordinary play. Production source selection does not enable its per-draw content
tracking. The passive diagnostic does not control rendering; production Streamline capture and
scale are described above. Generic fallback retains raw-scene calibration. The shared
[ABI compatibility](../../docs/reshade-sbs.md#streamline-abi-compatibility) uses three public families:
early 1.x resources without state, later 1.x resources with state, and typed 2.x structures. The
official-header survey covers all 28 public tags from 1.0.0 through 2.14.1. Optional frame-tag,
token and feature-function exports do not gate the core observer. Unknown ABI majors and
contradictory identities remain unsupported; native capture and private resource-state checks
remain separate from public ABI admission.

For call-route investigation only, `[SUNSHINE_DEPTH] UpscalerCallTrace=1` enables the
[separate runtime trace](../../docs/reshade-sbs.md#upscaler-call-route-diagnostic) after restart.
It reports actual Streamline/NGX calls and their callers without enabling the full camera probe
or changing depth selection. A loaded DLL or zero observed calls does not prove the rendering
route; inspect the positive call evidence and reported hook coverage. Disable it after testing.

Opening ReShade's overlay keeps the game in stereo, with its real controls composed at the same
position in both eyes. Normal mouse coordinates are retained and add-on control changes appear
live. Closing the overlay removes the controls without replacing the stereo resource generation.
With Desktop Duplication, the host composites the separate Windows cursor at the protected game
UI's applied plane, using the same frame's exported per-eye displacement. Cached stereo frames
retain their matching scalar; the host does not recalculate scene depth or the adaptive target.
While ReShade's overlay is open, the system cursor returns to screen depth to match its controls.
Disabled or unavailable UI protection and depth diagnostic views also use zero cursor displacement.
Game-rendered cursors remain part of the exported image. Install the host and add-on of the same
release together: both speak export protocol 4. This host names an add-on of another protocol in
sunshine.log. Hosts of protocols 1-3 silently ignore this add-on's mapping, so the stream stays 2D
with no log line on either side ([handoff contract](../../docs/reshade-sbs.md#gpu-handoff-contract)).
WGC does not supply the separate cursor metadata.
If another add-on cancels that transition, the observed state may differ from the visible overlay;
toggle the overlay again after resolving that add-on conflict. Native Game 3D does not depend on
ReShade FX compilation or the Home tab's effect toggle. The explicit reference renderer retains
its own technique/reload lifecycle.

## Source and submission validation

Some runtime fixtures retain `SUNSHINE_GAME3D_AUTOMATIC=1` as a test-process selector and to
load frozen comparison sources. It is not a current Game3D shader-mode definition. Current
installation and ordinary play require no such switch; historical Manual comparisons below
apply only to their archived shader/add-on controls.

Current API-boundary capture validation uses `reshade_depth_queue_cycle_test`, which links the
production capture owner and native observer directly:

```text
reshade_depth_queue_cycle_test.exe
reshade_depth_queue_cycle_test.exe --pipeline
reshade_depth_queue_cycle_test.exe --foreign-reclaim
reshade_depth_queue_cycle_test.exe --content 3840 2160
```

The default case checks that a pending foreign producer cannot introduce a reverse GPU queue
dependency, then verifies natural recovery and same-queue admission. `--pipeline` checks admission
of an unconsumed completed NGX snapshot while its successor is CPU-recorded or GPU-pending or
carries new observation/feedback revisions, a reset flag, a new active crop or a new
dynamic-resolution allocation (the completed snapshot keeps its own 64x64 allocation), or failed or
was refused before its copy existed (a missing frame: shown once, then consumed), plus rejection
across epoch and storage format changes (`format_changed`).
`--foreign-reclaim` retires producer and consumer recordings and
releases private snapshot leases before allocating an unrelated NGX capture or a Generic
preservation copy from a separate resource. Its fourteen cases check that SL FG nomination authority,
current depth and completed depth beneath a pending, failed or missing successor survive
reclamation, and that the completed depth that stood in for a failed or missing one is never shown
again; and that the viewport's completed SR depth beneath an FG input pending or refused before its
copy existed survives reclamation and the Present's idle sweep and stands in for it
(`lineage_snapshot`). Pixel cases use actual GPU readback.
`--content [width height]` nominates a full patterned packed D32S8 scene through NGX, which
copies it at the API call, then clears that same source after the API capture. It checks the
snapshot's scene depth byte-for-byte and verifies that depth-plane
copies preserve both the original and consumer stencil. This is one post-capture-clear scenario;
it does not cover a separate no-clear case.
Any fixture completion waits serve its deterministic pixel oracle and cleanup; they are not
production capture behavior. These owner-level tests do not establish SDK-hook coverage or
color/depth correspondence in a real game.

Use `reshade_game3d_native_provider_runtime_test` for the add-on's current native lifecycle with
zero installed FX techniques:

```text
reshade_game3d_native_provider_runtime_test.exe <ReShade64.dll> <frozenShaders> <SunshineSBSTest.addon64> <fresh-output-directory> 3840 2160 <frame_generation_interposer.dll>
```

It exercises real GPU depth production, capture, sampling and matrix preparation, including
padded/nonzero-offset rectangles, failed evaluation (the newest completed copy held within its
age bound, then mono with the native scale held) and recovery. The optional metadata-only
interposer enables the SL FG 2x case: retained real depth, generated presentations, a hitch of the
source age and more before a real Present and one between it and its generated Present (the
generated Present holds that frame's own copy and every Present keeps full stereo strength,
`strength_blend` 1; until the 10-27 review that generated Present was mono at strength 0), a tagged
extent changing every second real frame (dynamic resolution: fresh depth of each extent, held on
the generated frame, the matrix scale active without recalibration, and a
`Sunshine depth render size` line), FG Off holding the finished copy within its source age and
fresh SL recovery. It also checks repeated holds against the original capture's
expiry, camera reset, and FG off/on without a fresh capture; none may revive expired depth.
Fresh copies must recover each interval. A `ValidUntilPresent` FG depth tag sent without a command
list is copied before its frame's Present, and every Present shows that frame's fresh FG depth (R2;
the bounded `Sunshine Streamline deferred tag copies` line counts the copies). A `ValidUntilPresent`
or `ValidUntilEvaluate` FG depth tag that declares `NON_PIXEL_SHADER_RESOURCE` (its state at the
later use) while the call's own list records `PIXEL_SHADER_RESOURCE` is copied at the call from the
recorded state, nothing is refused, and every Present shows fresh FG depth (S11; the bounded
`Sunshine Streamline FG tag-call copies` line counts `recorded_over_declared`). Where the list
records `UNORDERED_ACCESS`, a write state, a `ValidUntilPresent` tag is copied before its frame's
Present instead (counted on the deferred line) and a `ValidUntilEvaluate` one still at its call;
every Present shows fresh FG depth. With FG options on
and every FG depth tag's copy cancelled by the same frame's untag (R7), the FG scope owns no depth
for those frames: their Presents hold the last FG copy while it is younger than the source age,
from the second one on over a fresh SR copy, and then show the same viewport's fresh SR depth, with
no readiness loss logged (until 10-24 the scope stayed live, with neither FG nor SR depth, for a
second after its last copy). Each generated presentation, at
2x and 3x, holds its real frame's SR copy, also after the fixture's window leaves the foreground
(the export stop invalidates reuse once) and back, with no readiness loss logged. A single accepted
FG copy amid such frames (Assassin's Creed Shadows, 10-24) shows at its Present and is held while it
is younger than the source age, and SR depth returns then, four times, with no readiness loss; a
`MEASURE native-provider SLFG-stray-FG-copy` line counts the Presents not rendered while the placement
controller settles on the other logical source, as at any FG/SR switch. Then the test add-on's
`SunshineStreamlineTestPresentBeforeNextCopy` makes a Present of an earlier frame land inside the SR
evaluation, between the read of its own local depth input and the copy (as DLSS-G's Presents on its
own thread do in Assassin's Creed Shadows); four such real frames keep their fresh SR depth with no
readiness loss. Once accepted FG copies own the viewport again, six rounds of two FG copies and two
frames without one (the second one's SR copy fresh, its evaluation having come while the scope was
silent) hold the last FG copy at every frame without one, with every Present rendered at the same
scale and no readiness loss (`PASS native SLFG occasional miss`).
The test executable also exports the NGX stand-in's entry points (below); its DLSS-G case creates
feature 11 and sets only the named `DLSSG.*` inputs (no capture). With its FG depth tag command-less
and untagged in the same frame (its deferred copy cancelled, R7) and its HUD-less and Backbuffer
tags command-less, as in Assassin's Creed Shadows, a DLSS-G evaluation whose list observes no state
for `DLSSG.Depth`, `DLSSG.HUDLess` and `DLSSG.Backbuffer` is refused and counted
(`dlssg_depth={copied=0 ... refused=missing_state:<n>}`) with the SR depth unchanged, and the test
add-on's refusal query (API contract matrix) reports it as a `state_unprovable` capture refusal;
meanwhile the game's command-less HUD-less and Backbuffer tags, copied before their frame's Present
(R2), reach UI detection as a same-batch pair (`hudless_presents={batch=<n>`). Once its list
transitions them, `DLSSG.Depth` with its subrect and `DLSSG.DepthInverted` owns the FG scope,
generated Presents hold it, the command-less FG tag withdraws nothing, and once the same frame also
untags the HUD-less and Backbuffer tags (cancelling their deferred copies) DLSS-G's own
HUD-less/Backbuffer pair reaches UI detection as a same-batch pair. HUD-less and Backbuffer tags sent
with the game's command list keep precedence per kind: across two NGX lines `dlssg_ui` offers and
captures nothing more while its `tag_copied` grows, and a later UI capture gate window still counts
the tags' own same-batch pairs; untagged in their frame again, DLSS-G's inputs are captured again. An FG tag sent with a
command list copies the frame first and keeps precedence (`tag_copied`). DLSS-G evaluating on its
own COMPUTE list on its own compute queue (as Streamline does in Assassin's Creed Shadows, after the
game's work ran) owns the FG scope the same way from depth observed there, with no `command_type`
refusal (N7: the compute queue takes its fence at the copy's submission). DLSS-G cycles three
recordings and resets each only when it records into it again, so a frame's copy is readable three
frames later. The case starts with that depth unobserved (refused, SR depth owns the viewport), and
the handover to DLSS-G's first readable copy, the real/generated pairs after it and two NGX lines
show no readiness loss; a scope that turned live at the copy's submission lost depth until the
reset. Refused again, SR depth
returns. Last, with the FG depth tag command-less and not untagged, a frame whose Present comes
after the game's work but before DLSS-G evaluated it (an earlier frame's paced or generated Present)
keeps its copy deferred (the Present holds the previous depth, `waited_for_frame` counts it); the
next Present, after that evaluation, shows the frame's own copy.
It injects no captured pixels, readiness or GPU completion.
Each FG case is a frame script: a warm-up that presents until the case's starting state holds, the
stimulus whose frames it judges, and a recovery to the fixture's control, with the per-frame SDK calls
as fixture members. Every frame of a script begins before the script's own deadline, so a case that
stops making progress fails by name, and a `MEASURE native-provider case=<name> elapsed_ms=` line
reports each case's time. Every Present of the run, in its scripts and between them, is also
judged by the continuity invariant ([Session-level tests](#session-level-tests)); its last two
scripts move SR and Generic depth onto the game's own direct queue for the invariant's known gaps.

Both runtime tests also check the
[frame record](../../docs/reshade-sbs.md#frame-record). Each queue-cycle acquisition's
`depth_choice` must repeat what `acquire` returned. In the provider test every provider status
line must name, in order, a Present whose record states that selection (D1 or D2 with fresh or
held pixels, H in mono, the handoff hold, or Generic), and the record's rows are checked at the
settled, failed-evaluation, FG pair, silent-scope, SR-through-a-silent-scope and DLSS-G exact-pair
points. The contract matrix's runtime layer, which includes this fixture, ends each family with
the same line-by-line check. `reshade_exporter_lifecycle` maps the provider's display and FG-policy
timelines to the record on the CPU through the provider's own mapping
(`provider::describe_acquisition` and `describe_display`, which `begin` calls), and the capture
owner's provider admission regression (run by the Streamline probe tests) checks `depth_choice`
against its selection timeline.

The interposer DLLs are one synthetic Streamline release each, built from one source by a CMake list
(`sunshine_sl_fixture_interposers`): the release its version resource names, which production
discovery reads, and whether it exports `slSetTagForFrame`. `frame_generation_interposer.dll` is
2.7.30 with frame tags; `legacy_v2_interposer.dll` reports 2.6.10.0 and omits `slSetTagForFrame`, so
the test submits global tags through the production hooks and requires the same capture, hold,
failure and recovery behavior (pass it as the final argument in the same command, with a fresh output
directory); `sl_v2_14_1_interposer.dll` is 2.14.1, the first of them whose header declares Dynamic
mode. Each accepts the frame generation options its release's header declares (the `DLSSGMode`
values, and `numFramesToGenerate` of 1, or up to 5 where the release documents multi frame
generation, as the [survey tables](#api-contract-matrix) record), reads a newer options version by its
known prefix, and refuses the rest. `SunshineFixtureSetResults(constants, tag, evaluate, options,
feature)` sets the `sl::Result` each boundary returns for a call it accepts.
`reshade_sl_fixture_interposers` (CTest, no device) checks every listed DLL's version resource,
exports, accepted options and results. They are synthetic metadata providers, not copies of a game SDK
or proof of Hogwarts Legacy's live behavior.
With `frame_generation_interposer.dll`, set `SUNSHINE_GAME3D_UI_PUBLIC_TAG_TEST=1` to exercise
automatic depth and HUD-less UI capture through the public hooks without SL evaluation or
confirmed FG. The viewport-zero mixed depth/motion/HUD-less batch uses zero resource headers
and omitted resource dimensions. Two production dumps must retain one request generation,
contain the exact expected GPU mask and produce distinct stereo eyes. Also set
`SUNSHINE_GAME3D_UI_FAILED_OFF_TEST=1` to follow every tag batch with a failed FG Off call,
checking that this unrelated repeated failure cannot erase the independent UI input. The
fixture's foreground override applies only to dump ownership; it does not change desktop focus.
See the [canonical capture diagnostics](../../docs/reshade-sbs.md) for the gate and request fields.
Run these functional cases serially when they share resources or output directories; their timing
is not performance evidence. The NGX, direct Streamline, preservation-mode-2, probe-round,
rotating-allocation and adaptive raw-depth runtime fixtures below use the same zero-FX native
lifecycle. They select their frozen `SunshineGame3D.fx` boot effect and the test add-on themselves,
so they need no environment settings. The command-association, native-selection and
native-present fixtures exercise the reference effects with native Game 3D off.

The optional `reshade_streamline_direct_runtime_test` is a component integration fixture.
Its metadata seam bypasses SDK discovery; the real SDK-entry NGX/Streamline cases below remain
necessary to validate that entry path. Like the NGX fixture it renders through native Game 3D
with no FX and needs no environment settings; each Present's consumed depth and constants come
from the test add-on's last-render query and Dump 3D captures verify the depth pixels and
projection a native render consumed. Its gain and zero oracle is the owning contract applied to
the inverse distance of the known draw: gain `L/Q` from the nearest depth and the zero at the
contrast midpoint.

This fixture exercises independent capture and
projection-based rendering. It writes real R32_FLOAT UAV textures that never enter the generic
depth-stencil inventory, preserves them through native command submission, and checks their
spatial depth pattern against the consumed depth. It also checks rotating resources, manual
pin/release, a changed center that moves only the zero, explicit recentering, proportional strength,
failed, missing and refused evaluations (the newest completed copy held within its age bound, then
current-color mono; one second of silence or of rejected evaluations releases the association),
a hitch longer than the source age before a frame's evaluation and one between the evaluation and
its Present (the first Present after each shows that frame's own copy at the full stereo strength
it had; until the 10-27 ruling such a pause restarted the fade from flat), Enable Game 3D off for
a second at full strength and on again (its first ready Present restarts the fade from flat, then
fades in to full strength: Presents shown without Game 3D are a loss of depth, not a pause; until
the 10-27 review stereo came back at full strength), an SR evaluation without its own camera right
after such a pause (it keeps the projection placement the Present before the pause showed, at full
strength, and so does the next frame with its camera; until the 10-27 review it went to the raw
basis, mono, and faded in), and zero-strength HDR color.
It uses the same arguments as the NGX fixture below, with
`reshade_streamline_direct_runtime_test.exe` and a fresh output directory. Its test-only
seam supplies middleware metadata, never shader readiness, copied pixels or GPU completion.
`SUNSHINE_STREAMLINE_MATRIX_CHANGE_TEST=1` checks changed near/far projection metadata:
the render must use current reconstruction coefficients while the gain and zero follow the
new decoded depth. It also checks rotation, gaps, FOV invariance and actual HDR pixels.
Like the modes below, it is exclusive. A source nomination ticket is not a pixel-readiness
assertion: negative capture cases verify held API ownership and current-color mono separately.
Set `SUNSHINE_STREAMLINE_COLD_V1_TEST=1` for V1-only rotation without successful V2 warmup.
Alternatively, `SUNSHINE_STREAMLINE_PRIVATE_STATE_TEST=1` checks the pinned V1 provider's
resource-state metadata on a fresh evaluation recording, including the rejection of its Present
state (COMMON) on that colour source (D3b) and of missing, malformed, conflicting or
split-transition metadata. These modes are mutually exclusive and
use the same real GPU capture and shader path.
`SUNSHINE_STREAMLINE_PACKED_V1_TEST=1` is a third exclusive mode for the packed
`R32G8X24_TYPELESS` depth/stencil format observed in Dead Space. It checks rotating sources,
prior-recording provider state, plane-aware depth readback, unchanged source stencil, the
consumed depth and projection, mono fallback and recovery.
`SUNSHINE_STREAMLINE_COMMAND_CAPACITY_TEST=1` is another exclusive mode. It retains
160 real native-only command lists on the observed vtable, then destroys and recreates them
while checking packed-depth rotation, shader output and recovery. These lists bypass ReShade's
device wrapper, so their cleanup must follow native COM lifetime rather than ReShade events.
`SUNSHINE_STREAMLINE_PENDING_EVALUATION_TEST=1` is another exclusive mode. After proving
ordinary packed-depth rendering, it submits the real producer recording before completing the
evaluation. Successful completion must still reach the shader; failed completion and replaying
the unchanged producer must remain mono. It checks both rotating sources, preserved source
stencil and consumed depth pixels, then verifies recovery with a new recording.
`SUNSHINE_STREAMLINE_LARGE_BATCH_TEST=1` is another exclusive mode. It submits 97 real native
command lists together with the depth producer at the start, middle and end; a separate idle-only
96-list batch must not invalidate captured depth. A 257-barrier call puts the source transition
last. Each case checks current shader pixels and source stencil, then failure/recovery. Set
`SUNSHINE_STREAMLINE_LARGE_BATCH_CASE=barriers` to isolate the barrier case; omit it for all cases.
`SUNSHINE_STREAMLINE_REQUIRE_PROVIDER_STATUS=1` additionally requires the test-only read-only UI
snapshot and checks current, missing/last-valid and manual-override source reporting. Without that
flag an older control DLL lacking the getter can still run the GPU capture comparison.

`reshade_ngx_depth_runtime_test` exercises named NGX entry points through the native lifecycle.
Its game-side SDK is the NGX stand-in in [`test_ngx_stand_in.h`](test_ngx_stand_in.h), which the native
provider fixture and the contract matrix's CPU layer also export: it creates any feature ID, returns a
settable result from each entry point, and answers a parameter map only for the keys a case set, each
through the getter of the type it was set with (`Depth` set as a void pointer is absent to
`GetD3d12Resource`). Evaluation accepts any command list; the fixture's separate producer can use a
COMPUTE queue (`producer_type`), which records only UAV depth and the evaluation.
Like the native provider fixture, it loads the supplied shaders only to initialize the runtime and
then removes every FX technique. Each present is observed through the test add-on's provider
status and Automatic/scale queries. Current output is read from the production export ring.
Production Dump 3D captures supply consumed depth, render constants and SBS at checkpoints. Like the
native provider fixture, it selects its frozen `SunshineGame3D.fx` boot effect, compile gate and
test add-on itself; the shader directory must contain that frozen effect:

```text
reshade_ngx_depth_runtime_test.exe <ReShade64.dll> <frozenShaders> <SunshineSBSTest.addon64> <fresh-output-directory> 3840 2160
```

Failed and invalid-extent evaluations must not take ownership from Generic. A valid evaluation
whose recording observed no source state is admitted through the DLSS input-state contract. The
default case checks exact full-allocation depth, poisoned padding and active-rectangle render
constants through rotation, recreation and a packed D32S8 crop with unchanged stencil.
Failed, missing and silent frames hold the newest completed copy until its source age bound,
then render current-color mono. Screen-plane refinement keeps readiness. Recalibration twice
gives an identical fresh reference, and 0/50/100 strength is proportional. A released feature
returns ownership to Generic; a normal-depth feature gets a new logical source.
`ngx-depth-trajectory.csv` records every present and `native-renders.jsonl` every captured render.

Set `SUNSHINE_NGX_CROSS_QUEUE_TEST=1` and `SUNSHINE_NGX_CROSS_QUEUE_COMPLETED_TEST=1`
for the deterministic separate-queue regression. The fixture executes depth production on
another real queue, retires its submitted recording, and waits for its actual completion before presenting. This test-only wait
is not add-on behavior. The fixture checks changing depth and HDR output. A still-replayable
recording's copy is never displayed; its predecessor may be admitted once the next reset retires it.
An unfinished producer is never displayed and adds no CPU wait. The case also covers
missing-frame hold and recovery and four producer queue lifetimes. Omit the completed flag for
asynchronous coverage, where admitted copies are verified and pending work may hold or remain
mono. These are functional checks, not performance evidence; an idle GPU is not required.
Run cases serially with separate output directories. `SUNSHINE_NGX_STATE_PRESSURE_TEST=1`
is a separate case and must not be combined with the cross-queue flags.

The former effect-callback NGX cases are retired and exit with code 2. These cover
`SUNSHINE_NGX_FRAME_GENERATION_TEST`, `SUNSHINE_SL_TAG_FALLBACK_TEST`, the `SUNSHINE_FG_*` variants,
`SUNSHINE_NGX_TRACKED_SOURCE_TEST`, `SUNSHINE_CAPTURE_DEMAND_TEST`, `SUNSHINE_NGX_PENDING_CONTINUITY_TEST`
and the NGX `SUNSHINE_DEPTH_BIND_SWITCH_TEST` variant. Native SL FG coverage is the native provider
fixture above. CTest `reshade_streamline_camera_probe` covers tag-48/tag-0 selection; no current
GPU fixture repeats the unsupported-capture tag fallback.

The retired generic-nomination adapter and its metadata-injection fixture are removed. Current
validation combines the native owner/provider gates above with the actual-hook CPU tests.
The generic selector retains its separate policy and runtime tests. No synthetic fixture alone
establishes a particular game's hook order or final depth/color jitter alignment.

Native Game 3D records its GPU passes and export on the runtime's immediate queue. It borrows
the selected depth only within the capture owner's open pass and supplies frame-associated
camera/jitter values directly. Its renderer owns the output texture and its color contract;
no FX reflection or export annotation is required for this path. Explicit `SuperDepth3D.fx` and
independent `SunshineDepth3D.fx` remain reference sources: their annotated texture is accepted
only after the owning technique executes, with native Game 3D disabled. Only the independent
reference requires its separate current-frame percentile calibration update.
The exporter verifies the normal runtime resolution, actual texture dimensions/format and
current game swapchain color space. Native Game 3D packs both eyes straight into the claimed slot;
a reference FX texture is copied there without resampling or color conversion. An HDR10/PQ
swapchain exports PQ code values to a consumer that accepts them, and a native scRGB
swapchain does so for a consumer that encodes HDR10 PQ; otherwise HDR exports linear Rec.709 scRGB
(1.0 = 80 cd/m²), PQ input decoded and its Rec.2020 primaries transformed first
([Color and HDR](../../docs/reshade-sbs.md#color-and-hdr)).

For reference FX only, the required `sunshine_sbs_source_color_space` annotation records the shader's compiled
`BUFFER_COLOR_SPACE`. It must match the game swapchain exactly: SDR, scRGB or HDR10/PQ.
The separate `sunshine_sbs_color_space` annotation declares the exported `srgb` or `scrgb`
texture. This prevents a stale PQ decoder from being accepted after a switch to native scRGB,
even though both exports use FP16. Unknown/HLG declarations and contradictory combinations
are rejected. A floating-point or 10-bit format alone never establishes HDR.

Windows HDR can stay enabled for an SDR game. The exporter describes the game's ReShade input;
Sunshine separately converts that SDR or HDR texture for its selected output. Windows Auto HDR
is not itself evidence of native HDR at this hook: if an OS color expansion occurs after ReShade,
it is outside this exported texture. The exporter follows the observed swapchain and renderer
color contract rather than inferring an Auto HDR result from the desktop setting. Actual Auto HDR
integration and per-game hook ordering require a running-game check.

The active renderer must execute in each presented frame. Disable, failed source proof and
focus loss invalidate metadata immediately; reference FX reload also invalidates its export. A game that stops presenting may retain its
last valid image; the exporter does not assign a false new timestamp to that retained image.

The Direct3D 11 path uses `SHARED | SHARED_NTHANDLE` textures and native shared fences. Both APIs
publish in ReShade's `reshade_present` callback, after the pack and any overlay composition and
before the game's Present: the slot is marked ready, then the fence is signalled. D3D11 signals on
ReShade's immediate context and flushes submission without waiting for it. D3D12 records into
ReShade's immediate command list, restores the resource states it changed, submits that list and
signals its native queue fence on the presenting queue. ReShade 6.8 fires `finish_present` only
after the real Present returns, so it fences only a D3D12 export whose `reshade_present` did not
run. A different swapchain or queue cannot publish the export.

While settings are open, a public ImGui draw callback adds an FP16 target to ReShade's native
overlay pass. ReShade still draws its actual controls, textures, tooltips and cursor and handles
input. A compatible GUI pipeline retains the game's native target and captures linear color
with full alpha in the additional target. A final GPU pass composes that layer into both eyes,
blending in linear light and encoding SDR back to sRGB. An scRGB export stays linear scRGB,
including negative values and highlights above SDR white; a PQ export is decoded, blended in
linear scRGB and encoded back as PQ ([Color and HDR](../../docs/reshade-sbs.md#color-and-hdr)).
D3D11 composition restores the application
state it touches; D3D12 uses ReShade's own immediate command list. Composition shares the
existing slot and fence lifetime; there is no additional queue wait or CPU image readback.

Up to four slot writes may be outstanding, one per ring slot. A Present that finds no reusable
slot (a free one, else the oldest ready one whose write completed, never the newest unconsumed
frame) drops its export without waiting and counts it as `dropped` (see the
[handoff contract](../../docs/reshade-sbs.md#gpu-handoff-contract)).
Source and destination resources stay alive across a pending copy. If runtime destruction prevents
submission confirmation, one retired generation is retained so a successor runtime can proceed.
The allocation cap is one retired ring plus one active ring; another unconfirmed destruction stops
export and logs that the game must restart. During DLL unload, unresolved native resources are
preserved until process exit instead of waiting on the game's GPU.

Lifecycle callbacks briefly serialize CPU state changes; render callbacks skip on contention.
Neither callback path waits for a producer or consumer GPU fence. See the
[versioned host handoff](../../docs/reshade-sbs.md#gpu-handoff-contract) for slot ownership,
generation replacement and process validation.

ReShade's SDK uses a C++ ABI. MSVC and MinGW disagree on aggregate returns, including the
eight-byte `resource` handle returned by `device::get_resource_from_view` and the description
returned by `device::get_resource_desc`. The MinGW build checks the exact pinned device-header
checksum and generates an isolated SDK copy with these two virtual slots expressed as explicit
result-pointer methods plus nonvirtual wrappers. It preserves the
slot order and leaves the downloaded SDK unchanged. This follows the explicit aggregate-return
pattern in MinGW's D3D12 COM headers. Other aggregate-return SDK methods must not be called;
native COM output parameters supply texture descriptions. An SDK API-version match alone does
not establish ABI compatibility.

## Automatic scene depth selection

The integrated scene-depth module is derived from ReShade 6.8's BSD-licensed Generic Depth example.
It owns depth tracking and the `DEPTH` binding inside `SunshineSBS.addon64`. The installer handles
the mutually exclusive built-in Generic Depth setting. In **Add-ons → Sunshine 3D**, leave automatic
scene-depth selection enabled. Existing depth-copy settings, explicit size/format filters and manual
selection remain available; restrictive filters still exclude buffers.

Read **Depth buffer** to see what the shader actually uses. A row checkbox forces a manual
override; automatic selection does not check one. Use **Use automatic buffer selection** after trying a
manual buffer. This selects the source buffer; Game3D's depth interpretation remains automatic.
The active row stays identifiable even when its draw activity pauses.

The selector refuses to register duplicate depth callbacks if Generic Depth or the old separate
selector is active. Keep only the unified DLL. Backups from the installer include a manifest of
original paths; they live in a subdirectory outside ReShade's nonrecursive add-on scan.

The [scene-depth contract](../../docs/reshade-sbs.md#selecting-scene-depth) describes selection,
manual precedence and limitations. D3D11/D3D12 candidate contents are sampled with a small raw
depth grid through private command lists and nonblocking readback. Only one readback and one
additional candidate backup can be active across runtimes. The inherited selected-buffer switch
path can wait for the command queue; asynchronous sampling itself adds no CPU fence wait.
The add-on keeps concise selection/readiness and failure messages. Per-sample histogram traces,
shader control polling and the temporary GPU control-echo add-on have been removed. Content
sampling remains part of automatic selection; it is not a debug feature. The
[scene-depth contract](../../docs/reshade-sbs.md#selecting-scene-depth) owns the histogram's meaning.

Resource lifetimes are tracked separately from capture layouts. Changes to rendering resolution
can reacquire depth after the viewport settles. Continuously changing dynamic-resolution
viewports may prevent confirmation because readbacks from an older layout are discarded;
the current implementation does not infer equivalence between those layouts. These content
checks also cannot conclusively identify the semantic role of every plausible depth texture.

The CPU policy test runs under CTest. The opt-in runtime build also retains a D3D12 selector
reference fixture using the separately supplied legacy shader. It draws real depth targets
through the official runtime, without injecting `DEPTH`:

```powershell
.\cmake-build-reshade\reshade_depth_selection_runtime_test.exe C:\test\ReShade64.dll `
  E:\Git\Repo\Depth3D\Shaders .\cmake-build-reshade\SunshineSBS.addon64 `
  .\cmake-build-reshade\scene-depth-dx12 1280 720
```

`--reload-only` checks recovery through interleaved depth-active presents and native-resource
recreation with the production add-on. Its interleaved phase checks binding dimensions and
periodic pixels; it does not establish current-frame capture or Automatic readiness on every
present. The current reference-policy fixture below covers continuous sources and isolated replacements, not
alternating scene allocations. `--manual-reload-only` additionally keeps the manually
selected original allocation alive after rendering moves elsewhere. This action test requires
`SUNSHINE_DEPTH_MANUAL_RECOVERY_TEST=1` and the separately named `SunshineSBSTest.addon64`,
which exposes a fixture adapter to the same override handover used by the UI. Never install that
test add-on into a game. Production recovery has no fixture action exports. The native test
checks active manual priority, short gaps, flat loading, resumption and confirmed replacement;
the CPU tests cover rejection of historical and duplicate confirmation samples.

`--manual-handover-only` uses the same test-only setup to pin useful native depth and release
it through the automatic-selection action. It checks uninterrupted shader binding, backup and
readiness against a busier reduced-resolution source, later recovery when the native contents
become invalid, and preservation of useful native history while trying another manual buffer.
The source dimensions come from the fixture arguments. A negative control uses the frozen old
selector with only its test adapter exposing the old automatic-selection action; the same
updated fixture must fail on lost binding/readiness, not on a missing export. Production DLLs
never expose these actions.
The existing Automatic test-only action run also pins and unpins its already calibrated
source while reading the real `Sunshine_CameraDepthScale` uniform. Readiness, binding, H and
the full-strength blend must remain unchanged across the handover; no calibration or shader
uniforms are injected.

It checks low-draw reduced-resolution scene depth against high-draw flat and square buffers,
replacement of an already qualified smaller source by a similarly credible native source,
stable comparison when both remain useful, fallback when the native source becomes flat,
format and resolution changes, capture before a clear, same-size competitors, content changes,
multi-viewport capture within the same resource, and a single active one-draw buffer. It also
checks a useful foreground-only incumbent against a full scene at the same resolution, a
lower-resolution full scene against native foreground depth, and scenes containing clear sky.
Histogram cases compare equally complete, equal-size buffers with narrow and broad depth
distributions, retain useful native-resolution depth against an equally complete broader scaled
source, and verify narrow-range validity and both normal/reversed depth conventions. The earlier resolution
comparison uses mirrored gradients with identical histograms to isolate the resolution preference.
It reads the actual shader depth texture as well as the
chosen binding, comparing linear depth with the independently projected input gradient.
The opt-in `reshade_depth_sampler_d3d11_test.exe` needs no arguments and checks the production
native sampler across D16, D24S8, D32 and D32S8 backups, full and padded viewports, and preservation
of the caller's compute state. It also checks immutable program reuse, shared-runtime retirement,
device replacement and an actual D3D12 dispatch whose program survives retirement until GPU
completion. Completed scratch resources are reused across changing source formats, dimensions
and capture rectangles; the native tests check increasing fence values, unchanged capture metadata
and release of completed source references. `--benchmark` reports preparation/submission/readback
CPU costs separately from GPU execution. CPU timings are descriptive, without a noisy speed
threshold. It does not exercise full D3D11 automatic selection. These
synthetic checks do not replace a game's DLSS/TAA acceptance test.

With runtime tests enabled, the CPU-only `reshade_depth_selection_snapshot` CTest loads the test
add-on without initializing ReShade or a graphics device. It compares the former deep copy with
the production compact snapshot, including capture-region parity, independent row ownership,
resource-address reuse and inactive manual history. Its synthetic workload timings include
construction, consumption and destruction; reported value bytes exclude allocator overhead.

## Diagnostics and validation

For an investigation of recurring missing depth, the temporary opt-in
`[SUNSHINE_DEPTH] FrameActivityTrace=1` in `ReShade.ini` records bounded metadata
bursts in the normal ReShade log. Restart the game after changing it. The trace
records observed present/queue identities, buffer activity and copy flags, and
selection/access readiness. Its counters are add-on callback ordinals, not game
frame IDs. It adds no depth readback or GPU wait and does not change selection or
calibration. Leave it disabled outside a diagnostic session.

The raw screen-plane controller has unit coverage in `reshade_raw_scene_policy_tests`.
`reshade_adaptive_raw_runtime_test` drives real game DSV draws through the integrated
selector/sampler and the raw-depth scale controller into native Game 3D, with no FX. Each Present
is observed through the test add-on's last-render query; Dump 3D captures verify the consumed
depth pixels and constants and the mono/stereo SBS. Its oracle is the owning contract's formula
applied to the independently known draw: gain `L/Q` from the full-image maximum and the zero at the
contrast midpoint ([raw-depth automation](../../docs/reshade-sbs.md#experimental-raw-depth-automation)).
Its scenarios check fresh single-window initialization, same-source pin/unpin continuity, a changed
center patch that keeps the gain and moves only the zero, held gain and zero through depth gaps
with current-color mono and through pauses in Presents before and after a frame's depth draws
(that frame's own depth at the full strength it had; that a pause earns no catch-up credit is
`reshade_scene_gain`'s, as the settled gain and zero here could not show it), and a pause during the
zero's adaptation judged the same way (its own frame's depth, never a reused copy) and left out of
its deadline (until the 10-27 review one followed by a Present without ready depth was
inconclusive, exit 3), a retained exact source resuming its own history, a
destroyed source and an explicit recalibration starting fresh, zero-strength mono and finite HDR
stereo, in both depth conventions. No camera/gain values or synthetic readback results are injected.
Like the bind-switch fixture it needs no environment settings:

```text
reshade_adaptive_raw_runtime_test.exe <ReShade64.dll> <frozenShaders> <SunshineSBSTest.addon64> <fresh-output-directory> <scrgb|pq> <normal|reversed> <width> <height>
```

The former `SUNSHINE_DEPTH_RELOAD_TEST` case required the retired effect's scale to restart after
an FX reload; native capture and scale deliberately survive an FX reload
(`reshade_game3d_native_depth_runtime_test`), and the case exits as a setup failure.
`reshade_depth_ready_uniform_cache_tests` covers reflection reuse and change-only publication by
the shared Generic/API readiness owner. The fixture requires the separate test add-on for the
real pin/unpin and **Recalibrate** actions. The production package excludes these test exports.

`reshade_depth_bind_switch_runtime_test` exercises preservation mode 2 with real
D3D12 D32S8 draws through native Game 3D. Like the NGX fixture it needs no environment settings
and selects preservation mode 2 itself; its arguments are only the runtime DLL, frozen
shader directory, test add-on and fresh output directory. Like the NGX fixture, it removes every
FX technique after initialization. It reads readiness from the Automatic/scale queries on every present.
Production Dump 3D captures supply the consumed depth, render constants and SBS. It runs 4K scRGB with
2228×1256 scene depth, first unbinding each buffer to null, then switching directly
from scene depth to another depth target. Changing depth patterns reject stale
copies. Same-binding, clear-only, missing-depth, transition-away and unknown-alias cases verify
capture admission: a refused case must stay unready on every present and render current-color
mono. Fresh recalibration and rendered stereo verify recovery. Each refused case differs from
the passing direct-switch control only by its clear, transition, alias barrier or missing draw;
a build without the generic alias rule fails at `unknown-alias-before-B`. Like the other native GPU fixtures, execution is opt-in and serial,
outside CTest. `bind-switch-trajectory.csv` and `native-renders.jsonl` record the evidence.
Add `SUNSHINE_DEPTH_GENERIC_ONLY_TEST=1` to disable both API sources, camera diagnostics and
call tracing before initialization. The fixture verifies those settings and runs the same
preservation and HDR checks, proving that shared capture does not depend on API discovery.

`reshade_depth_probe_round_runtime_test` uses the same four arguments and, like
the bind-switch fixture, renders through native Game 3D with no FX and no environment
settings to check bounded challenger sampling. The test add-on's last-render query
(`SunshineGame3DTestLastRender`) reports the depth and constants each native render
consumed; Dump 3D captures verify the consumed depth pixels.
A real 4K source is created before six active flat peers; it must replace the
calibrated lower-resolution source within three seconds, with its actual depth
pixels verified. The unchanged old add-on must fail that promotion bound. The
fixture also checks that an uncapturable candidate yields within its two bounded
probe visits and an active manual pin remains authoritative. These are controlled
regression bounds, not promised recovery times for every game. No selection, depth
or camera values are injected. `--native-priority` runs the late-native case instead.

`reshade_depth_alternating_runtime_test` uses those same four arguments and, like the
probe-round fixture, renders through native Game 3D with no FX and no environment settings
for an actual 4K rotating-allocation regression. Each allocation
first passes a continuous capture/calibration control. ABC and AABB cadences must
then reach continuous current capture, readiness and full stereo within a bounded
warmup; each allocation keeps its own gain and zero, those of its own drawn range and
contrast midpoint, which differ between the members. Every checked
present verifies the physical depth source and actual copied depth its native render
consumed, a changing color marker in both eyes of the exported SBS, and current mono when
depth is unavailable. Additional
phases cover off-turn transfer writes, exact manual pin/release, lifetime replacement,
unsupported partial crops, full-layout return, real depth gaps and a moving-center
trajectory. Run the identical executable against the frozen old and candidate add-ons;
the old single-source implementation must fail the rotating-readiness assertion after
passing its continuous controls. Compact GPU readbacks are correctness evidence,
not an uncontended frame-time benchmark. Execution is opt-in and serial, outside CTest.

The same fixture accepts one optional isolated case after its four normal arguments:
`--overlap`, `--interrupted-startup`, `--moving-startup`, `--rotating-startup`,
`--crowded-rotating-startup`, `--flat-startup`, or `--layout`. Each case needs a
fresh output directory. They cover complementary rotation becoming concurrent rendering,
real asynchronous startup samples across missing presents, changing central depth from the
first rendered scene, fresh ABC rotation without any prior continuous-source calibration,
and a stable preserved crop despite unrelated viewport activity. The layout case also requires
an actual captured-crop change to establish a fresh numeric basis.
The flat-startup case draws constant interior depth on the real source before useful scene
depth appears. It checks fresh initialization on that same lifetime, then a return to flat
content that must preserve established numerical state without refreshing readiness.
Every checked present inspects current depth and source-color pixels, including mono during
missing/unsupported depth. Run one frozen executable against both control and treatment;
these checks deliberately exercise the capture/selection/calibration boundaries.

`reshade_addon_unload_runtime_test` checks the add-on across ReShade's add-on unload and reload
([its session lifetime](../../docs/reshade-sbs.md#streamline-depth-selection)), from a startup
device with no queue, swapchain or frame (as Assassin's Creed Shadows creates) to game devices
destroyed with frames in flight. It is the lifecycle test of
[Session-level tests](#session-level-tests).

Shared native-depth, uniform and HDR mono observations of the depth fixtures live in
`test_raw_runtime_fixture.h`. The shared shader/transport fixtures retain their own contracts.

For full-pipeline rendering checks, use the D3D11/D3D12 fixtures described under
[Full-pipeline shader tests](#full-pipeline-shader-tests). Set
`SUNSHINE_DEPTH3D_EFFECT=SunshineGame3D` to select the renamed effect; unset it or use
`SuperDepth3D` for the frozen original. Each loads the entire separately supplied pipeline with
its own supported controls. A comparison must match the remaining shared controls explicitly.
These rendering checks complement the current controller fixtures and
installer/receiver workflow; they do not establish physical game/headset acceptance.

For paired DX12 collection, set `SUNSHINE_STEREO_PARITY=1` and run the original fixture in a fresh
directory. It records source/depth bytes, linear floating-point stereo output, actual uniforms,
texture layouts, measured landmark disparities and GPU timestamps under `parity`. Set
`SUNSHINE_STEREO_PARITY_REFERENCE=<baseline-directory>\parity` for the second run and use
`SUNSHINE_STEREO_PARITY_VARIANT` to label it. `SUNSHINE_STEREO_PARITY_EXACT=1` requires identical
final RGBA pixels for the identification-only check. A verified color-conversion correction needs
its own color-oracle result and documented expected pixel differences; exact baseline equality
cannot validate a correction to a baseline bug. Clear these variables before the full correctness
suite. Run GPU fixtures serially, with matching dimensions, color mode and inputs.

For an Automatic-to-Automatic shader-component comparison, enable
`SUNSHINE_GAME3D_AUTOMATIC=1` and `SUNSHINE_STEREO_PARITY_MATCH_AUTOMATIC=1`, then
provide `SUNSHINE_STEREO_PARITY_FIXED_CAMERA=<file>` containing `H`, `referenceZPD`
and `t0` values. This explicit test-only mode can record the first reference without
fitting to Manual. A comparison against that reference must supply byte-identical
fixed-camera and artistic-control files and use the same shader-clock contract.
Source/depth identity and measured endpoint matching remain required. These runs
inject camera controls to isolate rendering; they do not validate production
depth selection or calibration. Preserve the separate native-depth acquisition
tests for those claims.

`SUNSHINE_STEREO_PARITY_CONTROLS=<file>` accepts the shared Game3D quality controls
as `name=value` lines, including `De_Artifacting=x,y` and `Extended_Smoothing=0|1`.
Explicit controls must exist with the expected reflected type; removed Manual
defaults are optional only during fixture initialization. To vary the synthetic
depth distribution, `SUNSHINE_STEREO_PARITY_SCENE=<file>` accepts `background_raw`
and `middle_raw` while retaining the diagnostic foreground value of 0.02. Paired
runs require identical control/scene files and actual source/depth bytes. These
test overrides do not alter the installed game preset or the live controller.

`SUNSHINE_STEREO_PARITY_CAPTURE_PREPARATION=1` additionally records the complete
RG16F base level of both `Mod_Z` outputs at every static and moving endpoint. It
requires the full SunshineGame3D effect, deterministic time, and
`SUNSHINE_STEREO_PARITY_MOTION=1`. Both sides must use the same capture contract.
All channels and corner metadata are retained and checked for finite values;
readbacks restore resource state and must not advance the shader clock. Compare
all 40 texture files and 20 layout records as well as the final images when
claiming an exact preparation-preserving change.

### Test builds

[`CMakeLists.txt`](CMakeLists.txt) lists the add-on's production sources once
(`sunshine_game3d_core_sources`). Each set of test seam defines compiles them once, as a view: an
OBJECT library that an add-on DLL links whole, and `<view>_archive`, the same objects as a static
library from which a test executable links only what it uses. A view's seam defines are public, so
every file of a binary sees the same declarations; a binary links one view.

| View | Seam defines | Linked by |
|---|---|---|
| `sunshine_game3d_core` | none | `SunshineSBS`, `replay_game3d_dump`, and the tests without a seam (UI mask, UI capture, UI layer, camera data, queue-cycle and source-alpha) |
| `sunshine_game3d_core_runtime_test` | `SUNSHINE_SBS_RUNTIME_TEST_ADDON`, `SUNSHINE_GAME3D_RENDERER_TEST` (the renderer's build hooks, inert until the lifecycle test sets them) | `SunshineSBSTest` |
| `sunshine_game3d_core_sbs_test` | `SUNSHINE_SBS_TEST` | `reshade_exporter_tests`, whose test file compiles `exporter.cpp` itself |
| `sunshine_game3d_core_probe_test` | Streamline probe, NGX call trace, native observers, synchronous log | the probe, contract matrix (pure, hook and device layers), seam characterization, call trace, native observer and diagnostic-texture tests |
| `sunshine_game3d_core_renderer_test` | `SUNSHINE_GAME3D_RENDERER_TEST` | the native runtime tests |

Production files hold no tests. A white-box test reaches a file's internals through that file's
`*_test_access.h`: the capture owner's regressions are in
[`test_streamline_depth_capture.cpp`](test_streamline_depth_capture.cpp), and the test add-on's
exports of the depth add-on and the Streamline probe are in
[`test_depth_addon_exports.cpp`](test_depth_addon_exports.cpp) and
[`test_streamline_camera_probe_exports.cpp`](test_streamline_camera_probe_exports.cpp). In the
shipping build such a header changes nothing: the internals keep internal linkage. In the view that
builds the tests they get external linkage (an inline namespace, or `inline` for the depth add-on's
shared state) and the header declares the ones the tests use. The test add-on's refusal record for
the [API contract matrix](#api-contract-matrix) follows the same split: where an adapter decides a
refusal, its production file only calls a recorder, under `SUNSHINE_SBS_RUNTIME_TEST_ADDON`. The
probe's recorders are defined in its exports file and declared in its test-access header; the NGX
adapter, which has no test-access header, keeps its one recorder under the same define; and the
query, `SunshineDepthTestLastRefusal`, is one of the depth add-on's exports.

Two test binaries compile a production file outside every view: `reshade_exporter_tests`
compiles `exporter.cpp` inside [`test_exporter.cpp`](test_exporter.cpp) (its view excludes that
file), and the opt-in D3D11 sampler fixture `reshade_depth_sampler_d3d11_test`, which no `ctest`
runs, compiles `depth_content_sampler.cpp` inside
[`test_depth_content_sampler_d3d11.cpp`](test_depth_content_sampler_d3d11.cpp) to reach the
sampler's internal pipeline cache and private command list. Neither links a view.

### API contract matrix

The [rules for new game behaviour](../../docs/reshade-sbs.md#rules-for-new-game-behaviour) gate
admission on what Streamline, D3D12 and NGX require. The contract matrix checks that rule without a
game. [`test_game3d_api_contract_dimensions.h`](test_game3d_api_contract_dimensions.h) lists what each
API permits and filters illegal combinations;
[`test_game3d_api_contract_oracle.h`](test_game3d_api_contract_oracle.h) derives each required
outcome from cited contract text and the owner rulings R1 to R7 and N4 (its header lists them); and
[`test_game3d_api_contract_known_gaps.h`](test_game3d_api_contract_known_gaps.h) records where the
add-on falls short. A recorded gap that starts passing fails its test, so the fix that closes a gap
also removes its row; in the hook families a recorded gap must also keep failing under its row's rule
and the way the row records. Never change the oracle to match the add-on: a disagreement is either a
gap row or a fix, and an open owner question is EITHER or a pending row.

`ctest -L contract` runs the matrix. `reshade_contract_matrix_cpu` links the Streamline probe
tests' view of the production sources ([Test builds](#test-builds)) and holds two layers. The pure layer identifies every surveyed
Streamline release's interposer DLL as the probe's discovery does (the matrix mirrors that contract
selection) and reads raw buffer types 0 to 80 through the selected contract; it judges the capture
state table over every legal state cell and a set of deliberately illegal rows (a layout set by an
enhanced barrier on an earlier list reaches the table as the capture owner's latch and that list's
proof, D6 and N8; a depth or UI declaration of a later use, which a tag call copies from a state
its own recording names, as `tag_call_proof`'s `declared_for_use`, S11 and U5); and it reads every
pair of DXGI allocation and declared formats through
[`depth_view_format.h`](depth_view_format.h).

The hook layer drives the real adapters without a device. Streamline cases go through the probe's
own hooks on fixture entry points (`testing::install` takes the release's version number and selects
the ABI and buffer contract as discovery does for that version resource). They cover the full
product of tag entry point, lifecycle, command list, copy point and generated Presents; every
other value alone; a pairwise covering array over all dimensions, seeded with rows from the games
the add-on was tuned on; and one row per owner question the known-gap table held pending (S14 and
S24, since decided from the SDK headers). A changing extent runs two render
sizes and must keep one logical source. NGX cases go through the evaluation adapter over every
combination of its dimensions, DLSS-G's own `DLSSG.*` inputs included. The admission checks the
layer cannot run without a device (the copy's extent within the case's allocation, whatever the
Resource's declared size, as `source_region` checks it with `extent_in_allocation`, S16d; its command
list and its lifetime: inside the call that supplied the tag, a tag call or the evaluate a local
input came with, at an evaluation of the tag's own frame as the probe reports it, or with no
Present since; where an
FG tag call copies its tag, `deferred_copy` in [`api_contract.h`](api_contract.h) over the case's
recorded state with `tag_call_proof` for the copy made there, and the frame's untag that cancels a
copy deferred to its Present, R2, S11 and R7; and DLSS-G's observed-state nomination) are repeated
from the capture owner; the device and runtime layers replace them. A known gap that such a repeat
decides closes only when its fix changes the repeat in the same commit, as S7's (the lifetime at
the tag's own frame's Present), S11's (the proof at the tag call) and S4's (an `OnlyValidNow` local
input inside its own evaluate) did.
[`test_contract_matrix_generator.h`](test_contract_matrix_generator.h) builds the cases: a case id
hashes its non-default assignment, so adding a value renames no case. The executable takes
`--list`, `--case <id>` or `--case <sl|ngx>:<name=value,...>` (one case, described; an illegal
Streamline assignment must be refused, as in the full run), and `--shard <i>/<N>`. Shard 0 also runs
the pure layer; only an unsharded run reports a hook family's stale known-gap row.
`reshade_contract_oracle_independence` refuses any add-on include or name in the oracle's headers and
the generator, and compiles each on its own with the configured compiler, whatever the caller's
`PATH`.

The device layer, `reshade_contract_matrix_device`
([`test_game3d_api_contract_device.cpp`](test_game3d_api_contract_device.cpp), built with
`SUNSHINE_SBS_RUNTIME_TESTS`), records the capture owner's copies on real D3D12 lists of the first
hardware adapter, through the diagnostic texture fixture (included under
`SUNSHINE_DIAGNOSTIC_TEXTURE_FIXTURE_ONLY`). Each cell puts a fresh source in its state on an
earlier, executed list. Its copy's recording then holds the game's barrier: a transition, a split
ended there (begun on an earlier list of the same `ExecuteCommandLists` call, the only place D3D12
completes a split) or still open, a wildcard alias, an enhanced barrier, or nothing after an
enhanced layout set on the earlier list (each claim's copy from that layout, whichever it is,
where the claim names it, D6 and N8; the latch's refusal of a claim an unexecuted earlier list's
move makes stale is reshade_diagnostic_texture_capture_test's); on subresource 0, all subresources, another mip or the stencil plane. Each source is
retained before its earlier list executes, so the capture owner watches it for the enhanced-layout
latch as an adapter's retention does. Then comes the copy at its point: an adapter's copy with the
proof its claim selects,
a Generic boundary (`record_preserved` with the contract `depth_addon.cpp` passes: a clear's
`DEPTH_WRITE`, a read-only DSV's unbind, which names no state (`generic::bound_view`, D11c), and at
the end of the frame the source's last proven state on the copy's queue, which the earlier list
proved; each depth-stencil
source is tracked from its creation, as ReShade's `init_resource` reports it), or the UI clear
boundary (the barriers `game3d_ui_layer.cpp` records through ReShade where the capture owner's
`legacy_render_target_copy` admits them, as the layer asks; the runtime layer runs the layer
itself). The cells cover every legal state of every
allocation, each claim, each direct-list enhanced layout, every depth family typed and typeless with
mips and planes, the shapes no input names, COMPUTE lists (each state and claim a compute list may
use, and the compute-queue layouts, copied on that list and executed on a compute queue; S22, N7)
and UI tags. A cell meets the oracle by
its kind, refusal class and copy basis; the basis must be the state the test put the source in, so
the copy restores it exactly. Each copy executes alone: subresource 0's plane-0 bytes are read back
through the capture owner's consumer copies, and a D3D12 debug-layer error fails the cell. A copy
that a known-gap row says starts from an unproven state is recorded but never executed. The test
exits 77 (skipped) without a hardware adapter, and also without the D3D12 debug layer (Windows
Graphics Tools), though then only after every other check has run; `--gpu-validation` adds GPU-based
validation, `--block <name>` runs one block and `--batch <cells>` sets how many sources are prepared
together (16; at most 26, which the capture owner's pool and the debug layer were validated with).
It holds ctest's `gpu` resource lock; outside ctest, run it as any other GPU fixture.

The runtime layer drives the native provider fixture through the synthetic SDKs its documentation
above describes: an interposer DLL per Streamline release with settable results and the options its
header declares, the NGX stand-in the executable exports, and cases as frame scripts with their own
deadlines. `reshade_contract_matrix_runtime_<family>`
([`test_game3d_api_contract_runtime.cpp`](test_game3d_api_contract_runtime.cpp), which includes the
native provider fixture under `SUNSHINE_NATIVE_PROVIDER_FIXTURE_ONLY`) runs one family in its own
process: the add-on's own discovery hooks the release's interposer and the NGX exports, never the
`SunshineStreamlineTestCapture` bypass. The NGX runtime fixture is not included: both fixtures
build on the direct runtime fixture, and the native provider fixture already carries the stand-in.

The Streamline families put the tagged R32 allocation, on the game's own list, in one of five state
classes (a read state its recording proves, `GENERIC_READ` likewise, the producer's UAV write state,
an undeclared state, an enhanced `SHADER_RESOURCE` layout) and vary entry point, lifecycle, command
list, copy point, generated Presents (real Presents while the game records), FG mode (Off, On, Auto,
Dynamic) and frame count (2x, 4x), extent (a changing one is dynamic resolution, R3), untag, a
two-tag array and the three SDK results. A local evaluate input has no copy point after its
frame's Present here, since its own evaluate's copy would show at the Present that judges that
point; the hook layer runs it as a later evaluate without the input. `sl27` (2.7.30) runs every
cell: the product of state class, lifecycle and command list (at a same-frame evaluate and at an FG
tag call), seed rows, each value alone and a pairwise array. `sl26` (2.6.10, no frame tags) and
`sl212_214` (2.14.1, the newest surveyed release) run their seed rows and a pairwise array;
`sl215_30` runs the seed rows of 2.15.0 (newer than the survey: its arrays of version-1 tags are
read as a surveyed release's, S24) and of 3.0.0 (a
major no header defines, which discovery refuses, so nothing it tags may be shown) in a process
each. Each case is judged at its
Presents: a capture is the case's allocation shown fresh, with its tagged extent and one identity,
at six judged Presents from the first, and every case (its warm-up, stimulus and isolation) meets
the continuity invariant ([Session-level tests](#session-level-tests)); a case that shows nothing
refuses or holds, and a refusal
class the test add-on records must be one its copy points give. Where the oracle requires a
refusal alone, the test add-on must record its class, unless discovery refused the release, an SDK
call of the case failed (S21b discards that copy unread) or the class is `fg_scope` or
`no_copy_point`, which name a call that is no copy point, where no gate decides anything. A frame's
tag call and evaluate are
both copy points that a Present cannot tell apart, so a case is judged at the earliest one whose
outcome requires a capture. A capture at an FG tag call is deferred (R2) when the add-on's
nomination gate saw no nomination inside that call. Where the tag call permits a copy that the
case's own point does not (an open question there names a refusal first and the copy as its
alternative), the expectation keeps that copy, judged as at the tag call. A generated Present may show the case's source
only as the capture the last real Present showed (a hold). Every SDK call must return the
interposer's result. Each case has its own viewport and allocation; afterwards FG is turned off,
the depth untagged, the case's viewport retired (constants on four scratch viewports evict its
record: a viewport whose FG is Off still counts toward FG authority) and the fixture presents past
the source age ([`freshness.h`](freshness.h)) with no SDK call (the depth must be gone) and
recalibrates; a known-good control, an OnlyValidNow tag at its FG tag call, runs first and after
every ten cases. Their gaps are the known-gap table's `sl_runtime_rows`, whose cells pair the case
with the copy point that decides; a gap at a copy point the frame reaches before the deciding check
also shows its refusal class there. Only `sl27` can show a row stale: the others sample their
release's cells.

Family `ngx` drives Super Resolution, Ray Reconstruction and an unknown feature through `Depth`
and the DLSS keys, and DLSS-G's own evaluation through `DLSSG.*` on a viewport whose FG the 2.7.30
interposer's options enabled, over feature, `DLSS.Use.HW.Depth`, how the depth is set, subrect,
result, a recorded or unrecorded state and a recreated feature (seed rows and a pairwise array).
Family `generic` turns the Streamline and NGX sources off and draws a depth-stencil view of the
swapchain's size: a clear after or before the draws (depth or stencil only), a render-pass depth
clear, an unbind and the end of frame, D32 and D32S8, judged by the depth the last native render
consumed. The view is created in `DEPTH_WRITE` and never transitioned, so its creation proves that
state at the end of the frame and at a stencil-only clear (D10b). The states an earlier executed
list proves at the end of the frame are the device layer's, and so are a read-only DSV's unbind
(D11c) and the UI layer's clear boundary on an enhanced or simultaneous-access target (D12). Family
`ui` (2.14.1) tags a
captured depth and a full-size UI input (UI colour and alpha, HUD-less colour, Backbuffer, UI alpha)
at a tag call with the game's list, without a command list or as a local evaluate input (at a tag
call also `ValidUntilPresent` and `ValidUntilEvaluate`, with its list recording a read state, a write
state or none, U5), or draws
an offscreen UI layer cleared every frame (a render target in `RENDER_TARGET`, or a
simultaneous-access one whose copy the layer refuses, D12); a case is offered when Dump3D lists its input as
available to UI detection. Their gaps are `ngx_runtime_rows`, `generic_runtime_rows` and
`ui_runtime_rows`.

The executable takes `--family` and the generator's `--list`, `--case` and `--shard`, and appends
one JSON line per case (with the family's scope) to `contract-<family>.jsonl` in its output
directory. It is registered only where `SUNSHINE_GAME3D_RESHADE_RUNTIME` (the official
`ReShade64.dll`) and `SUNSHINE_GAME3D_FROZEN_SHADERS` (a directory with the frozen
`SunshineGame3D.fx`) are configured: one test per family, labelled `contract;gpu`, `RUN_SERIAL` with
the `gpu` resource lock and a 240 s timeout, after `reshade_contract_matrix_runtime_output` removes
the families' output directory. At 1280x720 the seven families run 318 cells in about 5.5 minutes:
`sl27` 123 cells and 14 controls in 107 s, `sl26` 49 in 61 s, `sl212_214` 48 in 58 s, `sl215_30` 38
in 37 s, `ngx` 34 in 32 s, `generic` 12 in 17 s and `ui` 14 in 19 s. A capture takes 0.4 s, a
refusal or hold 0.9 s (its 600 ms window and the isolation), an FG case copied at an evaluate up to
1.9 s (FG owns its viewport's depth for a second after it is enabled), and a UI case its Dump3D.
Repeated runs give the same result for every cell.

[`Invoke-Game3DContractMatrix.ps1`](Invoke-Game3DContractMatrix.ps1) configures and builds this
directory on its own (the Sunshine build forces the add-on's tests off), runs `ctest -L contract`
and writes `contract-matrix.jsonl` and `summary.json` under `<build>/contract-matrix-<time>`. The
JSON lines hold each hook and runtime case's id, dimensions, rule, expected and observed outcome,
verdict, gap rows and evidence ([`test_game3d_api_contract_jsonl.h`](test_game3d_api_contract_jsonl.h);
`reshade_contract_matrix_cpu` writes its lines where `SUNSHINE_CONTRACT_MATRIX_JSONL` names), and one
line per contract test with its ctest result. The runner checks those cases against the known-gap
table itself: a case may only meet the oracle, fail as a recorded gap or sit in a pending row, and
each expected-failure row of a family that ran whole (and, at runtime, whose scope is whole) must be
the only row of a failing case. `summary.json` names the commit the run started from and lists under
`uncommitted` the tracked files that differed from it (the console says how many), so results from
a merge in progress or a local edit do not pass as evidence for that commit: run it after the commit.
`-SkipGpu` leaves out the device and runtime layers; run them only with no game running. A contract
test that ctest reports as skipped (the device layer without a hardware adapter or the D3D12 debug
layer) is listed as `skipped` in `summary.json` and fails the run unless `-AllowSkip` is given.

The runtime layer reads refusal classes from the test add-on instead of waiting for rate-limited log
lines. `SunshineDepthTestLastRefusal(runtime, out)` returns the newest depth refusal of the
Streamline, NGX and DLSS-G adapters and the capture owner: the adapter, the gate that refused (the
call's source, its depth input or the capture of its copy), the production reason, the capture
stage and status, the call's epoch, sequence and viewport, and a serial that tells a new refusal
from an earlier one. It is recorded where the refusal is decided, before the 5 s limit on its log
line. [`depth_refusal.h`](depth_refusal.h) maps each reason to the oracle's REFUSE class for the
input that produces it, or outside the contract to `hold`, `absent` or `internal`; the pure layer
requires every reason to classify and each one an oracle case produces to carry that case's class.
Only `SunshineSBSTest.addon64` records or exports refusals: `reshade_addon_exports` reads both
export tables without running either add-on and requires the shipping add-on to export exactly
ReShade's add-on entry points.

Buffer-type meanings, and each release's `DLSSGMode` values, multi frame generation and
`ResourceType` values, come from
[`test_sl_survey_tables.generated.h`](test_sl_survey_tables.generated.h), never from the add-on's
contract table. [`contract_sl_survey_tables.cmake`](contract_sl_survey_tables.cmake) generates it
from the Streamline ABI survey: each release's own headers and its interposer DLL's version
resource. The survey is a research artifact kept outside the repository, which cannot recreate it.
`reshade_contract_sl_survey_tables` compares the checked-in copy with the survey. It exists only
where `SUNSHINE_SL_ABI_SURVEY_DIR` names a survey (configure reports when it is skipped); the default
is the main checkout's `cmake-build-relwithdebinfo/sl-abi-survey`, also from a worktree. After the
survey gains a release, regenerate the copy, review its diff and commit it with the matrix changes it
causes:

```bash
cmake -DSURVEY_DIR=<sl-abi-survey> -DOUTPUT=tools/reshade/test_sl_survey_tables.generated.h -P tools/reshade/contract_sl_survey_tables.cmake
```

### Session-level tests

The contract matrix judges single decisions. Assassin's Creed Shadows showed defects that only a
session shows: depth flicker from correct decisions composed over frames (fixed by 9146a468,
88b45edf, 64d6d06c, 46a4510e, ad66b1c8, 77d48255 and d8c52e3b). The continuity invariant, defined
once in [`test_game3d_continuity.h`](test_game3d_continuity.h) for the native provider fixture and
the runtime layer that includes it, judges every Present:

> After warm-up, a Present whose frame had at least one legal source shows depth from some source,
> or holds earlier depth within the source age ([`freshness.h`](freshness.h)). A Present whose
> frame had no legal source may hold within the source age, then go mono. A generated Present
> presents an earlier frame: after a Present that showed depth it holds that depth while the copy
> is younger than the source age.

Next to it, the freshness rule. Continuity alone accepts a Present that keeps showing an old copy
within the source age while newer ones are readable, as Assassin's Creed Shadows' SR copies were
shown 140-220 ms old before the copy-out (24312303):

> A Present that shows API depth shows the newest copy of that source the presenting queue could
> legally read for the frame it presents, within two of the fixture's frame intervals of it.

The source is the shown rung's: D1, an FG scope's own depth (an FG tag's copy or DLSS-G's input),
or D2, scene depth (an SR or RR evaluation's input, or a tag's copy outside FG); never the other,
since FG keeps its viewport while it can supply (`frame_generation_holds`) and a switch between
them restarts the placement calibration. The record names the shown copy by its capture tick
(`depth_tick_ms`). The supply log says which copies the presenting queue could certainly read at
the Present: its own legal source (one that meets no limitation and that no live FG scope keeps
out) of a frame before the one the Present presents (a copy deferred to its frame's Present, R2,
is read at the Present after it), or of that frame when recorded at its call on a list the
presenting queue executes before it (`supply::immediate`: an SR evaluation's input, a tag copied
at its call, DLSS-G's input on the game's list; until 10-27 none counted before the Present after
it, so a Present that held SR depth up to the source age old beside its frame's own fresh SR copy
passed, as at FG On with DLSS-G's first copy in flight), and a legal copy recorded on another queue
(`supply::elsewhere`) that the contract copies out ([Copy-out on the producer
queue](../../docs/reshade-sbs.md)) and that the judge saw complete at an earlier Present or right
before this Present's call (`judge::before_present`; its fence, `supply::completions`). The
fixture models the contract, never the add-on's report: a copy
recorded on another queue is copied out when a copy of its source was submitted on such a queue
within the source age before it was recorded; a source's first copy elsewhere keeps the Reset rule,
past the source age in every fixture, and never counts. A Present is `stale_shown` when the
newest such copy's stamp is more than `freshness_bound_ms` after the shown copy's capture tick:
`freshness_frames` (2) of the fixture's frame intervals (its own pacing, the mean of its last two
real Present intervals: the presented frame's own copy and the one before it, which the add-on may
show instead), plus `tick_slack_ms` for two steps of `GetTickCount64`, as the stamp and the tick
are separate reads. No row excuses it: no known limitation makes a readable copy unreadable. It
passes on every provider script, L3 case and session timeline with the copy-out, and fails without
it (10-27, the copy-out disabled in a temporary build): `sr-other-queue` holds its last
presenting-queue copy 188 ms behind three completed game-queue copies (bound 132 ms at its 46 ms
frames) before that copy expires, `dlssg-evaluation` holds an FG copy behind DLSS-G's completed
compute copies, and 12 of the 43 fixed timelines fail it.

`unordered_read` (Step 4, U-C1): a read of a copy still in flight must be ordered after its
producer by the game's own fences. Each fixture logs every `ExecuteCommandLists`, `Signal` and
`Wait` it makes on the game's queues and fences, its CPU signals and each Present, and an
INDEPENDENT oracle ([`test_queue_order_script.h`](test_queue_order_script.h)) computes vector
clocks from that script alone: a wait for w orders its queue only after the pointwise earliest of
every signal of that fence reaching w anywhere in the script, and after nothing when the fence's
initial value, a CPU signal or no signal reaches w. Every Present whose record says
`depth_read_order=ordered` names the producer position the add-on read after; the fixture learned
each Execute's position from the test add-on right after the call returned
(`SunshineQueueOrderTestLastExecute`), so it finds that Execute in its script and the oracle must
order the Present after it (and the copy-out the add-on made in that call's post-call). A script
that knows a read was in flight (its producer queue held behind a gate the CPU opens after the
Present) adds that read whatever the record says. No row excuses a violation (the Present showed
depth its copy had not written yet) but one: a read that the script's calls up to its Present
order, and only a signal first submitted after it (the CPU's, or another queue's, of a fence its
wait relied on) unorders, is `late` (`judge_reads`), the residual the owner accepted as O1 (10-27),
which no observation at the decision can exclude. The session timelines count it as the known gap
`R1-late-signaller`'s, the native provider fixture as `late=`. The add-on's side of O1: the model
counts each relied read whose fence such a signal poisons within the source age as `late_conflict`
and marks the read's capture conflicted, and the provider no longer holds that capture's
presentation copy from the next Present (`read_conflicted`; reason `late_conflict`): mono rather
than pixels the producer may not have written. `sr-other-queue-in-flight-late-signaller` waits
on a handoff fence only the game queue signals, then after its last Present the CPU signals that
fence to the waited value while the game queue is still gated: its reads are `late`, and the next
Present (no new copy) must be mono; with the provider's check removed it held that copy
(`retained_copy`, 10-27, temporary build). The native provider fixture judges each script
(`MEASURE native-provider unordered_read judged=<n> late=<n>`), the session timelines each run
(`ordered_reads=` and `late_reads=` on its line; 342 reads over the 43 fixed timelines on 10-27,
none unordered). With `readable_elsewhere` mutated to admit every positioned copy,
`sr-other-queue-in-flight-no-wait` fails both the in-flight check and `unordered_read` (12 reads);
with the model's CPU-signal poison removed, `sr-other-queue-in-flight-cpu-signal` fails it from
the records' own `ordered` claims, and so does the session timelines' regression
`O3-cpu-signal-ahead`, whose failure shrinks to `sr=other_queue,order=cpu; sr=other_queue`
(10-27, temporary builds).

The session timelines' fence events (`order=`, drawn from a stream of their own so that every seed
keeps its other events): each frame's work on another queue (its SR on the game's queue, DLSS-G's
compute copy) is followed by that queue's signal of an order fence, one per ring, which the
presenting queue waits for after the signal (`wait`) or before it (`wait_first`, a legal wait
before signal), or which the CPU waits for (`cpu_wait`). `cpu` and `second` signal it one value
ahead from the CPU or a third queue of the game's before the frame's Present, so the next frame's
wait is reached before that frame's work (the add-on must refuse that fence for good, O3); `late`
signals the waited value again after the frame's Present (O1). Both fences are created anew for
each run and at each FG toggle. The fixture models the contract's fail-closed rules from its own
script: a frame's copy elsewhere is readable at its own Present only when it is complete (the CPU
waited) or ordered by a settled fence that no CPU or second signal reached (a relied read, which
names that fence). Either way it stays legal with its completion for the freshness rule. A frame
whose Present can read none of its legal sources (its copies elsewhere not readable there, and its
SR input, if any, kept out by a live FG scope) is `unordered`: its Present may go mono only once no
copy it could read is younger than the source age (`judge::holdable`, read as of the add-on's
decision: a copy whose work completed between that decision and the judge's record neither
excuses nor demands anything there; until 10-27 it made the mono a gap, so seed
c28965dca8d7c76c's frame 38 failed or passed by when DLSS-G's compute copy completed), and any
other loss there is a gap. The first CPU, second or late signal of a fence makes the copies read through it
unholdable from then on (`judge::late_signal`), as O1 stops holding such a read: each copy names
its own fence (`supply::completion::relied`), the fixture's token of that fence object, new at
each FG toggle (until the 10-27 review every copy of a frame took the frame's fences, keyed by a
struct reused across toggles). A dropped hold on a generated Present after a frame whose fence a
late signal reaches after its Present is `R1-late-signaller`'s only where the copy its real
Present showed was read through such a fence (until the 10-27 review any was, also of a copy the
presenting queue made). Once a late signal reaches the fence the scope's held DLSS-G copy was read
through, the fixture no longer keeps the viewport's SR out for that copy (the add-on drops it and
SR stands in). `reshade_game3d_continuity` pins the judge's rules on synthetic records; the
regressions `O1-silent-scope-late-conflict-unordered-sr` and `O1-late-signal-presenting-copy-held`
run them on the add-on. Until 10-27 such copies were permitted
with no completion, which disarmed the judge until FG toggled (a mono run after a CPU signal
passed), and a late window tagged every legal source of every real frame within the source age,
which excused any loss there. The regressions `O3-cpu-signal-ahead` (depth on every frame after
the CPU signal) and `O1-silent-scope-late-conflict` (a silent FG scope's held DLSS-G compute copy
conflicted while the frame's own SR copy stands in; with the provider's O1 check after the FG
hold, as at 1a7efdc1, frame 32 lost depth, 10-27, temporary build) pin both; a regression's
`lossless` check counts losses only, not late reads.

The session timelines' slow intervals (10-27), modelled rather than left to the machine's load,
which made the full check's two failures of that day depend on timing: `hitch=` sleeps `brief`
(half the source age, which every hold outlasts) or `before_sources` (the source age and 150 ms)
right after the frame's stamp, before its calls, `after_sources` (as long) after them, before its
Present (the fixture's bookkeeping after it, such as whether a live FG scope still holds its copy,
reads the clock after the hitch, as the add-on decides then), `between` (as long) before the
frame's first call and its stamp, or `after_present` (as long) after its real Present, before its
generated Presents; `dlssg_gpu=after_present` holds DLSS-G's compute work on the GPU until
the frame's Present returned (its queue waits on a gate fence the CPU signals then, a GPU-bound
game's copy in flight at its own Present; the gate is no order fence, so no read relies on it, and
a GPU signal after the Present would deadlock against the presenting queue's wait before it). Seeds
draw them at a low weight (about one frame in 80 hitches; DLSS-G's compute work is held in about
one compute frame in eight) from a stream of their own, after the fence events, so every seed keeps
its other frames (until the 10-27 review no seed drew them). With FG on the game's FG options declare
each frame's generated count as `numFramesToGenerate` (and DLSS-G's `MultiFrameCount`), repeated
when it changes, as the add-on reads it (always 1 until the 10-27 review).

The judge reads each Present's [frame record](../../docs/reshade-sbs.md#frame-record)
(`SunshineGame3DTestFrameRecord`, copied in ReShade's `reshade_present` event) and the fixture's
own supply log. Each frame function names, at the call that supplies it, what the frame supplies as
the fixture knows it, never as the add-on reports it: an SR or RR evaluation whose input state is
provable (`sr_evaluate`), a depth tag the contract lets the add-on copy at its call or before its
frame's Present (`tag_copy`), DLSS-G's `DLSSG.Depth` in a state its recording names
(`dlssg_input`), or a Generic boundary with a proof (`generic_boundary`). A source the add-on may
show but need not is permitted, not legal: the oracle's EITHER outcome, and the full-size decoy
view each provider frame draws, which API ownership leaves aside (H). A frame whose later call
withdraws its source supplies nothing (a command-less tag's frame presented before DLSS-G evaluated
it, whose copy belongs to the next Present), nor does a generated Present, which the game makes
while it records a later frame. A refused later tag of a frame (the untag script's malformed one)
withdraws only the observation: the copy the frame's own tag made before it stays supplied, and
stands in as a missing frame's (until 10-28 the script withdrew it from the supply log and required
no fresh depth there). Each source's copy is
stamped when it was made (`supply::made`, its deciding call after any hitch before it, or its
recording's submission for a copy from another queue), and holds and the freshness rule age copies
by it, not by the frame's first call (until the 10-27 review a copy made after a hitch inside its
frame looked as old as the hitch, and a dropped hold of it passed).

A Present shows depth when the rung that decided it has pixels: D1 or D2 ready or held, or
Generic's binding ready; H in mono shows none. The judge is armed by a Present that shows depth. A
Present that shows none disarms it when its frame's own Present reads none of its legal sources and
nothing it could hold is younger than the source age (an unordered frame, a copy deferred to the
next Present, a pipelined source; frames whose sources meet a known limitation keep their loss for
its row), when its frame's events may end its source (an untag, a run's isolation:
`supply::drop_permitted`), when its frame had no legal source and nothing is holdable, or when it is
a violation; a generated Present neither arms nor disarms it. In the session timelines
(`judge::strict_holds`) a frame with no legal source must hold the last depth while it is younger
than the source age (a mono Present there is a `dropped_hold`); the native provider fixture's frame
scripts end sources without marking them and keep the older rule. Until the 10-27 review any frame
without a legal source could go mono, and its disarm gave the next legal frames a new source's
warm-up. While
disarmed, a frame's legal sources have one source age to show depth, counted from the first such
Present without depth, and a loss that lasts longer is reported once per source age. Only a warm-up
gives them `establish_ms`, `fg_silent_ms` plus `association_ms` (2 s): the session's start, a stall
(a Present at least the source age after the previous one, which expires every hold, while a
pipelined source such as DLSS-G's ring has nothing readable until it refills; never for a frame whose
own Present reads one of its legal sources, `supply::immediate`, that a live FG scope does not keep
out, wherever the pause came, since that source stays readable: until 10-27 a hitch inside such a
frame disarmed the judge, which excused a mono Present after a hitch between a frame's SR evaluation
and its Present, `hitch-after-sources`, and until the 10-27 review one before its first call still
did, `hitch-between-*`; nor for a generated Present after its frame's real Present; a silence of
`association_ms` is one for every frame, as the API association ends there) or a new source (a
source-age gap in legal sources). Frames whose legal sources all meet a known limitation and frames
with a source that meets none are timed apart, so a loss a known gap excuses never extends the time
the others have. A generated Present after a real one that showed a fresh copy of a frame whose
legal sources the presenting queue reads must hold it while that copy is certainly younger than the
source age (from its made time when it is that frame's own, else from the real Present two before:
a copy deferred to the next Present was made after it; less 40 ms), unless a copy recorded on
another queue may have become readable meanwhile. A generated Present that presents its frame as
the real Present did (after it, or inside the next frame's recording before that frame's first
depth call: `between`) shows that frame's own copy, or the one its real Present showed, at age 0, and
when that frame supplied an FG copy its live scope accepts (`supply::fg_frame`, so the add-on counts
its generated Presents, `frame_presenting`), dropping the frame's own fresh copy there is a dropped
hold whatever the clock (until the 10-27 review only within the source age of the clock, so the
add-on's mono generated and real Presents after a hitch with FG on, open item 1, passed). A violation is a gap (armed, a legal source, no depth), an unestablished
source (no depth within its window), a dropped hold (that generated Present without depth), a
stale one (a rung's pixels at least the source age, plus 40 ms for two steps of `GetTickCount64`,
after the newest legal or permitted source of its kind: API for D1 and D2, Generic for D3; a real
Present's API copy captured inside the frame it presents, its capture tick no earlier than the
frame's stamp, is that frame's own depth and age 0 however long the frame took: until 10-27 its age
ran from the frame's first call, so the 10-27 full check's `C8-fg-off-color-only` failed at frame 39
with age 672 ms after a 683 ms hitch inside that frame, before its SR evaluation, and
`hitch-before-sources` pins it; a real Present that holds an older frame's copy is also stale once
that copy itself is the source age old, whatever newer supply its frame made: until the 10-27 review
the frame after a hitch could hold the copy before it, 418 ms old, while its own copy was deferred
to the next Present, `hitch-after-sources-then-deferred-tag`) or `stale_shown` (the freshness rule
above).

The native provider fixture judges every Present of its run in segments, each frame script and
the session's frames between scripts, with a fresh judge each, so that no segment's state or loss
stands in for another's: each segment reports its violations
(`MEASURE native-provider continuity <script> violations=<n> unknown=<n>`), any one no known gap
records fails the run with its frame and record, and so does a known gap supplied in a segment
without a violation there (the gap closed). The run ends with its counts
(`continuity-session presents= legal= shown= generic= unrecorded= stalls=` and each known gap's
supplied frames and violations). The runtime layer judges each case's frame script with the
case's supply: the outcome the oracle names at the deciding copy point, legal where it requires a
capture, permitted where it only allows one and nothing for illegal input, supplied at the tag call
or the evaluate that decides. A violation fails the case as an oracle disagreement does; its JSON
line counts `observed.continuity_violations` and `evidence.continuity` describes the first. No case
violates it (until C5 closed on 10-26 the `S1-evaluate` cells never established depth inside their
window, and their oracle row recorded them).

The known gaps are the header's `known_gaps` rows, under the matrix's known-gap discipline with
their own key: each names a limitation that the fixture tags on the frames that meet it. A
violation of a tagged frame is its row's only once no copy the presenting queue could read (a
frame's legal source that meets no limitation, and that no live FG scope kept out of a frame it
supplies) is younger than the source age, as no limitation
explains an earlier loss: a hold dropped within that age is a violation. Any other violation fails
the run, and so does a row whose limitation was supplied in a run without a violation (the gap
closed, and the fix removes the row). Each provider run with an interposer exercises the open one
(C2) and the closed one's scripts:

- C1, SR depth on another queue, closed on 10-27 by U-C1b
  ([docs](../../docs/reshade-sbs.md), Copy-out on the producer queue, and
  [Queue order](../../docs/reshade-sbs.md#queue-order)). `sr-other-queue` warms up with the
  viewport's SR evaluation on the presenting queue (FG off), then records it on the game's own
  direct queue, a ring of 16 recordings each reset only when recorded into again and paced 20 ms
  apart, which the presenting queue waits for. Until 10-27 such a copy was readable only once its
  recording retired, after the source age here, so the Present lost depth once the presenting
  queue's last copy aged out (in Assassin's Creed Shadows 110-235 ms, about 30
  `no_completed_snapshot` losses a session). The copy-out made it readable once its fence
  completed, Step 4's in-flight reads (U-C1) once the game's fence waits order the presenting queue
  after it: `sr-other-queue-in-flight` holds the game queue's GPU behind a gate the CPU opens only
  after each Present, so every copy is still in flight at its Present, and each of its 12 Presents
  must show that frame's copy (with the completion rule alone none did: `in-flight copies shown=0 of
  12`). Its three negative forms must show none: the presenting queue does not wait (`-no-wait`), or
  it waits on a handoff fence that the CPU (`-cpu-signal`) or a third queue of the game's
  (`-second-signaller`) signals to the waited value before the game queue's own signal can run;
  `unordered_read` judges every form. U-C1b gives a source's first copies copy-out storage too: a
  copy on a `COMPUTE` list (DLSS-G's, N7) always, and a copy on a list not known to execute on the
  queue that reads depth (last executed on another queue, as the render queue's lists are once FG
  On moves Present to DLSS-G's queue, or never executed while the newest `DIRECT`-list copy ran on
  another queue: with FG Off none, so a game that pools its lists allocates nothing for them). A
  copy whose copy-out storage cannot be allocated is still recorded, with the Reset rule
  (`copy_out={... failed=}`). A copy recorded for the reader's queue and
  submitted after Present moved takes an idle slot's cached storage of its shape at its submission,
  with no allocation there; with none it keeps the Reset rule (`copy_out={... unstored=}`). For the
  source age after FG options turn on, the reader's own queue copies out too, since FG's first
  Present moves Present after that frame's copies were submitted there (until 10-27 those kept the
  Reset rule, so that first FG Present was flat). The
  capture test pins each case (`copy_out_replay`, `copy_out_reader_moves`, the compute-list
  cases); with U-C1b reverted they fail, and the fixed timelines' seed `6e5b9c154ddc34d8` shows its
  frame 10 (its first SR copy on the game's queue) mono instead of fresh, which the judge's warm-up
  excuses. No row remains: every copy elsewhere is copied out, so the fixtures' freshness supply
  names each one's completion, and the timelines' DLSS-G compute frames and the SR inputs a live FG
  scope keeps out (`supply::owned`) are judged as any. A copy whose queue's copy-out ring was all in
  flight (`ring_busy`, eight lists per queue, at most one per game submission) is a counted resource
  bound with no row: no fixture fills the ring, so a loss it caused fails its run.
- C2, Generic on another queue (until Step 4's fence hooks): `generic-other-queue` lets the API
  association lapse, then draws one view of the swapchain's size on the game's queue with no
  boundary but the end of the frame; each frame's list moves it through `DEPTH_READ` back to
  `DEPTH_WRITE`, so its end-of-frame state is that queue's proof, and the presenting queue's copy
  is refused (`end_of_frame_refused` grows with `last=missing_state`). A view that only its creation
  proves (D10b) is copied at the end of the frame from either queue. The same view on the
  presenting queue settles as ready Generic depth.

#### Found defects

Only C2 (until Step 4; C1 closed on 10-27) and R1-late-signaller (O1, 10-27) are limitations the owner approved; a matrix row's recorded gap
over frames would be one too (C5, the last, closed on 10-26). Any other row is a defect these tests
found, its `found` naming when and which test; it is excused only until the owner decides it, so that
the gate stays usable meanwhile, and the decision removes its row and witness. None is open: the
last, C8, closed on 10-26. Each run prints found defects apart
(`MEASURE native-provider continuity-found-defects` and `MEASURE session-timelines found-defects`,
each row's violations, or `none`), and a strict run fails their violations as any other:
`SUNSHINE_GAME3D_CONTINUITY_STRICT=1`, or the contract runner's `-StrictContinuity`, which also
records `strict_continuity` in `summary.json`. With no found row open a strict run judges as any
(until C8 closed it failed at 5 of the fixed set's 35 timelines, C8's witness among them).

C3, a command queue created mid-session (10-25, T1), is closed (10-26): installing the native
observer's hook for the new queue is an observation loss that refuses the next Present's fresh copy
(`consumer=recording_invalid loss=global_observation_loss`), and a failed fresh copy dropped the
retained display copy, so that Present was mono while a copy younger than the source age was
retained. A failed fresh copy now holds the retained copy as a pending successor does
([Streamline depth selection](../../docs/reshade-sbs.md#streamline-depth-selection)); the provider fixture judges the frame after its first
queue created mid-session (the UI hook's second queue) as any, and the seam characterization pins
`decide_failed_copy`.

C4, an FG scope that outlived the copy it can show (10-25, T2), is closed (10-26): a scope stayed
live for `fg_silent_ms` after its newest accepted copy unless an FG call marked a miss, and kept the
viewport's SR/RR evaluations out meanwhile, while the provider holds that copy only for the source
age. A frame with no FG call marked none, nor did a frame whose FG tag left it without a copy within
`fg_silent_ms` of DLSS-G's own last copy (the miss was left to a DLSS-G evaluation that need not
come), so after a single FG copy the Presents were mono from about 250 ms to a second although each
frame's SR input was copyable. Any frame that records no FG copy now misses the scope: each frame of
the viewport begins with its Streamline constants, a live scope expects the frame to record an FG
copy (in flight or accepted), and when the next frame begins with none recorded since, the scope
turns silent, so the viewport's SR/RR evaluations supply it, ready when the held FG copy expires
(`depth_capture::frame_generation_frame`). A frame whose only FG input is DLSS-G's compute copy
still supplies the scope: the SR input it keeps out while that copy is in flight was C1's, which
Step 4's fence observation makes readable at its Present; since C1 closed (10-27) the timelines
judge that SR input as the scope's (`supply::owned`). That held
for the frame after it too, whose SR input the still-live scope keeps out while its Present can
show only the scope's newest readable copy, three DLSS-G evaluations old: the first frame without a
compute copy loses depth when that copy expires before its Present (real frames of about a quarter
of the source age or longer), which the round 3 review found (soak seed `76a7d0a6922d20ea`, frame
104; `20*sr=local; fg=on,sr=local,dlssg=compute,generated=3; 11*sr=local,dlssg=compute,generated=3;
10*sr=local,generated=3` loses frame 32). With fence observation that compute copy is readable at
its own Present, so the timelines tagged that SR input with C1 as well until C1 closed (the
regression timeline `C4-compute-copy-stops` keeps the shape, now judged as any). The
timelines also model the precedence itself: the SR input of a frame after one that supplied an FG
copy is the scope's (`continuity::supply::owned`), so that frame's Present is judged as any but its
SR input leaves no readable copy for later Presents. And C8's frames (until it closed) included FG Off after an FG copy
that met a limitation (DLSS-G's compute copy, or until C6 closed one inside its window) with no readable source
since, which the display may hold as well. The witness became the regression timelines
`C4-no-fg-call` and `C4-cancelled-fg-tags`, the seam characterization pins the rule, and C6's
witness, whose SR evaluation now supplied, became a frame shape with DLSS-G's compute copy and no SR;
C6 then also recorded, as a dropped hold, the SR copies that compute copy supersedes, until it closed.

C6, DLSS-G deferring to an earlier frame's FG tag copy (10-25, T2), is closed (10-26): DLSS-G's
evaluation added no depth copy while any FG tag copy of its scope was younger than the source age
(`tag_copy_recent`), a precedence meant for the same frame's tag, so a frame without an FG tag of
its own had DLSS-G's fresher depth copied only once that copy's hold expired, and on DLSS-G's compute
list (readable three evaluations later) those Presents were mono. The precedence now belongs to
the frame DLSS-G evaluates, the viewport's newest, begun by its Streamline constants: a tag call is
that frame's when both name the same frame (its token), or, where either names none, when it came
after those constants (its call sequence). DLSS-G stands aside when that frame's tag copy was
recorded, or waits for the frame's Present (R2), while the scope's newest tag copy is younger than
the source age (`frame_generation::depth_tag_precedence`); without the viewport's constants the
source-age rule applies as before. The witness became the regression timeline
`C6-dlssg-after-tag-copy`, which is lossless: with the source-age rule restored its three lost
Presents meet C1's limitation (DLSS-G's compute copy), so any loss a known gap records fails it
too. The seam characterization pins the rule. C6's tagging (a frame without an FG tag, with
DLSS-G's evaluation, within the source age of an FG tag copy) also covered losses of DLSS-G's
compute copy that keying the precedence does not change, which the add-on loses the same way on
the build before this fix and which no row records now, so their soak seeds fail as the other
unrowed ones do until the owner decides them: a silent FG scope's first compute copy, made as the
FG copy that the display holds over the viewport's SR copies (`frame_generation_holds`) expires,
supersedes those SR copies before it is readable (soak seed `8227e2e85dc62560`, frame 123; no
fixed-set seed loses depth that way), and FG turned on with DLSS-G's compute copy its only FG input
drops a held copy younger than the source age (seed `f44999a9be60c936`, frame 70, which fails at
frame 115 anyway). `b76037565ceead51`, failing before, loses the same way at frame 40.

C7, an FG tag copy deferred to its frame's Present that waited for DLSS-G (10-25, T2), is closed
(10-26): under FG such a copy (R2) was made at the first Present after DLSS-G evaluated a frame
following the tag whenever DLSS-G had evaluated anything in its scope within the source age, refused
evaluations included. When DLSS-G stopped evaluating while the game kept tagging (generation paused,
its evaluations dropped, or DLSS-G evaluating some frames only), each frame's copy lapsed unmade
until that age passed while the held copy aged out first, so those Presents were mono although each
frame's tag was copyable at its Present. A deferral now waits for DLSS-G only while DLSS-G evaluates
every frame: an evaluation in its scope within the source age, and one since the viewport's previous
deferred frame's newest tag call, which is that frame's own (with no earlier deferred frame of the
viewport in the epoch it waits as before). Otherwise its frame's Present is the first after a game
list executed on the presenting queue since the tag call, the rule without DLSS-G's evaluations, or
the first after an evaluation that followed the tag call (`frame_generation::deferral_evaluations`).
So the frame after DLSS-G's last evaluation still waits, the held copy standing in for it, and each
later frame is copied at its own Present; when DLSS-G evaluates every other frame, a frame it
evaluates is copied after that evaluation and the next one waits. Counting from the previous frame's
Present instead would end the wait at an earlier frame's Present that DLSS-G makes after a frame's
tag call and before its evaluation, which the provider fixture's command-less FG tag case pins. The
session timelines model the one frame that still waits: a frame with a deferred FG tag copy and no
DLSS-G evaluation of its own, after DLSS-G evaluated a frame since the previous frame whose tag copy
was deferred (or with none before it), within the source age of that evaluation, supplies that copy
as permitted only, since its Present holds the earlier copy and leaves no readable copy for a later
frame's loss. The witness became the regression timeline `C7-dlssg-stops`, and the seam
characterization pins the rule. Two soak seeds that lost depth at frames C7's tagging covered, within
the source age of a readable copy (`3ff9ab504ef2c9f2` at frame 51, `e1bfe1504e0dd344` at frame 59),
now pass; `3962e7a36baf19cf`, named with the row, fails at frame 71 by a dropped hold that the
build with the old rule shows too.

C8, FG turned off while the display held an FG copy (10-25, T4), is closed (10-26): with the
viewport's SR depth on another queue (C1), FG Off dropped the FG copy as a changed source
(`source_changed`, the SR snapshot pending), so the Presents until that SR copy was readable were
mono although the display held a copy younger than the source age; 88b45edf's hold
(`frame_generation_holds`) kept an FG copy over a pending SR copy only while FG was on. In
Assassin's Creed Shadows each FG Off would lose depth until SR is readable (110-235 ms). The owner
chose holding: across FG Off the FG copy is held within the source age, as any source's missing
frame is. Two rules make it so. FG Off retires the FG source, which released the queue's API
association at once, so a color-only Present after it dropped the copy (`source_not_associated`);
the queue now stays the API's while the display holds an FG copy younger than the source age and FG
is not revoked (`depth_capture::frame_generation_retained`), and only then may it hand over to
Generic. And the viewport's ordinary source that follows an FG copy (the same viewport's SR/RR
source of the copy's epoch, once FG is off or its scope silent) is no changed source until its first
readable frame: `decide_display` holds the FG copy meanwhile (`FG_successor_pending`, or
`retained_after_failed_copy` after a failed copy of that source; the display lineage,
`depth_capture::detail::lineage_successor`, since 10-27 one rule for both directions), and that source's storage and encoding, which
compare frames of one source, do not end the hold. A readable frame of any source replaces the copy,
and another viewport's, epoch's or provider's source still drops it. A silent scope selects as
without FG, so FG turned on again within the hold keeps the copy until the scope supplies, while
`frame_generation_holds` alone keeps a copy made since FG On over a readable SR copy. The provider
fixture's `fg-off` script now holds the finished copy at a color-only Present after FG Off and is
mono once the seed's source age passed, and `mode-off-on` holds it through FG Off and On and revives
nothing past that age (both required the copy dropped at FG Off, as the row's decision noted). The
row, its limitation, its tagging, its witness and the dropped-hold exception (`drops_hold`, C8's
alone since C6 closed) are removed. The witness became the regression timeline
`C8-fg-off-sr-other-queue`, extended to FG on again within the hold, and `C8-fg-off-color-only`
puts frames that supply nothing between FG Off and the SR copies. With the fix reverted the fixed
set (then 35 timelines) fails at 5, each a gap at FG Off (`source_changed`) within the source age of
the FG copy: the regression at frame 21 and the four seeds the strict run failed (shrinking to
`fg=on,sr=other_queue; tag=present_bare,sr=other_queue; fg=off,tag=present_bare,sr=other_queue`).
With the successor rule alone, soak seeds `357bb706de968043`, `ecae326826f15f93` and
`cc3476634cec651f` (C8's tagging had excused them) failed: color-only Presents after FG Off dropped
the copy and Generic's decoy view took the queue, so the SR copy on another queue that followed
within the copy's source age found nothing to hold; `C8-fg-off-color-only` then fails at frame 32
(it needs Generic ready before the SR copies, so a much faster run may not reach that loss). Soaks
over seeds 24 to 311 also no longer fail where FG Off dropped a copy that C5's frames would have held
(`8796b81ac6f85eb1`, `df5b8bef81f93ab5`, `9b04fb316a7df676`, `b71fd435cfcf4171`: `source_handoff`
at FG Off). The seam characterization pins the successor rule (twelve rows), and the display cache
test `frame_generation_retained`. The fixed set's seed `30618337c4062c47` meets the unrowed pattern
C6's note names (frame 56 or 57) whenever its frames 36 to 54 take less than the source age: the FG
copy that `frame_generation_holds` keeps over the viewport's readable SR copies then expires only as
DLSS-G's compute copy supersedes them. That depends on the run's pace alone (both builds take the same
decisions there); it failed 4 of 29 runs during this fix, each when the machine ran fast, and none of
12 runs of either build once those frames took 297 ms. Since the round 3 review (10-27) the timelines
model that hold: while a silent scope's display holds its own FG copy made since FG On, younger than
the source age, the viewport's SR inputs are the scope's (`supply::owned`), as `frame_generation_holds`
keeps them off-screen, so they leave no readable copy for later Presents, and the loss when
DLSS-G's compute copy supersedes them is C1's (that copy was readable only frames later; since
Step 4 it is readable at the next Present ordered after its copy-out). Whether `frame_generation_holds` should keep an FG
copy over a readable SR copy at all (it spares the placement calibration a switch, and costs these
losses until Step 4) awaits the owner.

C5, the API contract matrix's `S1-evaluate` row over frames (the runtime layer found the row on
10-23, the session timelines carried it from 10-25), is closed (10-26): an `OnlyValidNow` global or
frame tag outside FG scope is copied at its call (S1), and the same frame's SR evaluation of that
tag refuses it for its lifetime (S3, `not-valid-until-present`). That refusal began a new evaluation
of the viewport with no copy (`withdraw_invalid_attempt`), which superseded the tag call's, so each
such frame's Present was mono once the depth before aged out. The evaluate now keeps its tag call's
outcome when every depth tag it refuses is an `OnlyValidNow` tag of the frame it was set for,
refused for its lifetime ([`api_contract.h`](api_contract.h) `keeps_tag_call_copy`; the frame is the
evaluate's when it names the tag's frame or no Present came since the tag call, as the capture
judges a later use of a tag). The evaluate still records its refusal. A tag of an earlier frame and
a tag refused for anything else still withdraw. The matrix's row, the continuity row, its
limitation, its tagging and its witness are removed. The runtime family's gap seeds `now-evaluate`
and `frame-now-evaluate` keep the row's cell (through `slSetTag` and, from 2.7.30,
`slSetTagForFrame`) in every release, and `test_api_contract.cpp` pins the rule. The witness became
the regression timelines `C5-frame-tag-then-evaluate` (the witness itself) and
`C5-global-tag-then-evaluate`. With the fix reverted both fail at frame 35 (a gap,
`provider_associated`), and so do the row's four `sl27` cells and `frame-now-evaluate`. The round 3
review's soak then found the same loss where the frame's untag ends the tag between its call and the
evaluate (seed `089154be9db5f403`, frame 113): the evaluate reads no depth tag (`missing-depth`), a
source refusal that withdrew before the rule was asked. It now judges the tag call the untag ended
in the untag's place (the viewport keeps its newest tag call per depth kind for that). The regression
timelines `C5-frame-untag-then-evaluate` and `C5-global-untag-then-evaluate` fail at frame 35 on the
build before (a gap, `provider_associated`, shown 35 of 50 frames).

FG On before the scope's first FG copy is readable, the most common failure of the round 3
review's soak (10-26, seeds 400 to 599: 13 of 20 failures, none rowed), is fixed (10-27): a live FG
scope dropped every copy but its own, so a generated Present between FG On and its first readable FG
copy (inside the SR evaluation, inside the recording, or before DLSS-G's compute copy is readable)
was mono although the real Present before showed an SR copy younger than the source age
(`dropped_hold`, `FG_scope_required` with `fg=refused/no_completed_snapshot`; shrunk to
`2*defaults; sr=local; fg=on,tag=now_list,sr=local,generated=1,at=evaluate`), once per FG On
(Assassin's Creed Shadows toggled FG 27 times in its 10-25 session). The owner's C8 ruling (hold
within the source age, as for any source's missing frame) now holds in both directions: a
viewport's Streamline FG input and its ordinary SR/RR source of one epoch are one display lineage
(`decide_display`, `depth_capture::detail::lineage_successor`), whose newcomer takes the viewport at
its first readable frame while the predecessor's copy is held. That one rule replaced
`frame_generation_successor` (FG to SR) and `frame_generation_pending` (SR to FG in a silent scope
only); `frame_generation_holds` still keeps a silent scope's FG copy over a readable SR copy (its
loss with DLSS-G's compute copy, in C6's note above, awaits the owner). The regression timeline
`fg-on-holds-sr-copy` toggles FG three times so; with the live scope's old drop restored it fails
at each FG On (frames 20, 37 and 54, `dropped_hold`). The seam characterization pins the lineage
(eight rows), and the display cache test the SR-to-FG direction under a silent and a live scope.

The same soak's gaps at a frame-generation tag of lifecycle `ValidUntilEvaluate` (seeds
`bf06bc26f9b88f1c`, frame 153, and `6b3b06cd02a48739`, frame 140, which the review counted with the
FG On holds) are a different defect, fixed with them (10-27): such a tag whose list records no state
of it is refused at its call (S11c, `missing_state`), and the frame's SR evaluate of the tag, the use
its declaration names, copies it. The probe ended a `ValidUntilEvaluate` tag's lifetime at the
return of any frame-generation tag call (`finish_evaluation`, a rule from before S11c that ordinary
tag calls were already exempt from), so that evaluate read the tag unsupported (`missing-depth`) and
every such frame was mono once the depth before aged out. A tag call now ends no `ValidUntilEvaluate`
tag. The regression timelines `S11c-fg-until-evaluate-tag-then-evaluate` (through
`slSetTagForFrame`) and `S11c-fg-global-until-evaluate-tag-then-evaluate` (through `slSetTag`) fail
on the build before (gap at frame 34, and at frame 16 of the shorter form without the warm-up).

`reshade_native_provider_continuity` runs the provider fixture at 1280x720 with the 2.7.30
interposer, every frame script under the invariant, in about 115 s. Like
`reshade_addon_unload_runtime` it is labelled `contract;gpu`, runs serially with the `gpu` resource
lock and is registered only where `SUNSHINE_GAME3D_RESHADE_RUNTIME` and
`SUNSHINE_GAME3D_FROZEN_SHADERS` are configured, so `ctest -L contract` and the contract runner run
it. The two other-queue scripts add about 6 s to each provider run. It runs the whole fixture, so a
failure may also be one of the fixture's own UI-detection checks (the offscreen UI layer's
acceptance, H1's flattened claim), which have failed about once in 60 runs; its log names the
script. It is the provider fixture's run at 1280x720 with the 2.7.30 interposer, so a check that
runs the contract tests need not run that configuration again.

The session timelines
([`test_game3d_session_timeline_runtime.cpp`](test_game3d_session_timeline_runtime.cpp)) compose
the matrix's dimensions over frames at random. A timeline is 60 to 200 real frames, each a row of
one family of frame events: a Streamline depth tag in each lifecycle with or without a command
list, through `slSetTagForFrame` or `slSetTag`; a null untag after the tag (by the frame's token or
by none) or before it (the frame before's), or none; FG turned on or off, and 1 to 3 generated
Presents after the real one, inside the recording, or landing inside the viewport's SR evaluation of
its own input, between that input's read and its copy (2x to 4x: the test add-on's
`SunshineStreamlineTestPresentBeforeNextCopy`, withdrawn when the evaluation nominated no copy, then
the Presents after the evaluation; DLSS-G's present thread does this in Assassin's Creed Shadows,
fixed by 64d6d06c); a dynamic-resolution step; the
viewport's SR evaluation with its own input on the presenting queue or the game's own queue, or of
the frame's tag; DLSS-G's own evaluation on a direct or compute list, its depth in a state recorded
there or not; the game's HUD-less and Backbuffer tags; and the state choice of the frame's depth
inputs (nothing recorded, a read state recorded before the tag call, undeclared, or a state recorded
at the tag call that a longer-lived declaration does not name). Each declaration names the state at
the use its lifecycle names, as the frame keeps it there: an incorrect declaration of a later use is
a game's bug the oracle cannot describe at a deferred copy (the add-on refuses it as
`conflicting_state`). The oracle names an untag through the tag's own entry (R7), so a token-less
untag of a frame tag (`slSetTag` after `slSetTagForFrame`) is judged EITHER: the outcome with the
untag or without it. A seed drives `generator::sample` in
[`test_contract_matrix_generator.h`](test_contract_matrix_generator.h): each dimension keeps the
previous frame's value with a chance or is drawn by weight under the family's predicate, so a
timeline has the stretches a game has, and FG-only events are dropped where FG is off.

Each timeline runs on its own viewport through the native provider fixture (the official runtime,
the test add-on, a real D3D12 device, the 2.7.30 interposer, the NGX stand-in), then isolation
retires the viewport, presents past the association and drains the other queues' rings. It is
judged on four counts:

- the continuity invariant at every Present, each frame supplying what the oracle names at its copy
  decisions: the tag call (or the copy it defers before the frame's Present), an evaluation of the
  tag (the earliest point whose outcome requires a capture decides, as the runtime layer's
  `at_present`), the SR input, and DLSS-G's input, each legal where the oracle requires a capture,
  permitted where it only allows one and nothing for illegal input; a frame whose legal sources all
  meet a known limitation is tagged with it;
- each copy decision by the oracle through the refusals its gate records
  (`SunshineDepthTestLastRefusal`, polled after every call and Present): a contract class recorded
  for a decision whose required refusal names another, or for a decision the oracle requires to
  capture, fails it. In a session the FG scope may decide a call first (it owns its viewport's SR
  inputs, its tags' deferred copies wait for DLSS-G, and DLSS-G's evaluation defers to its frame's
  tag copy), so a required refusal its gate never reached under the oracle's FG scope, or for which it
  recorded a class outside the contract's (`fg_scope`, `hold`), is counted (`preempted=`). Outside
  the FG scope no rule preempts a gate, so a required refusal recorded nowhere fails
  (`oracle:refusal_unrecorded`);
- each untag the oracle names as R7 (the frame's own untag cancels the copy its tag deferred to the
  frame's Present, which the oracle holds by S9): when the tag call deferred a copy, the untag must
  cancel it, judged by the add-on's deferral totals (`SunshineStreamlineTestDeferrals`; `cancels=`
  counts them, `oracle:deferral_kept` fails one);
- a Present, real or generated, shows a fresh copy only while a frame within the source age
  supplied an API source the oracle allows (copies recorded on another queue, which become readable
  later, aside), and a fresh copy of an FG scope's own input only while one supplied an FG source (an
  FG tag or DLSS-G's input): a refused or held decision never shows new depth. A deferred copy (R2)
  is read at the Present after the one it precedes, and a copy its FG scope kept out shows once the
  scope falls silent, so a fresh copy may be an earlier frame's. The frame a Present presents supplies
  it whatever a hitch since its calls: its real Present, its generated Presents after it, and those
  inside the next frame's recording before that frame's first depth call (until the 10-27 review only
  the clock counted, so a frame's own FG copy at its own Present after a hitch failed as unsupplied);
- every provider status line names a Present's frame record (`require_records_match_log`).

A violation of a tagged frame is that row's (as above); any other failure prints the timeline's
seed, its event list and the frames before the first failure. Once every timeline ran, the first
failing one is shrunk by delta debugging to the smallest list that still fails with the same
signature (the failure's kind and its cause: the record's owner and reason, or the decision and its
oracle rule), printed as `REPRODUCER ... reproduced=yes|no: --events "..."`: the frames after the
failure, each event kind back to its default in every frame, chunks of frames, then each event kind
of each frame of a short list, every kept candidate cut after the frame where it failed, then the
minimal list once more (`reproduced=`). Shrinking re-runs each candidate in the same process, after
the timelines before it, so `--shrink-runs` (default 40) and, in the fixed set, one minute
(`shrink_ms`, within its 460 s bound) limit it; a failure that depends on timing may shrink less, and
a fresh process's `--events` replay may still differ. The fixed set also runs one regression timeline per
Assassin's Creed Shadows fix (and one for the generated Presents' hold, `repeated-frame-hold`,
two for C4's fix, `C4-no-fg-call` and `C4-cancelled-fg-tags` (and `C4-compute-copy-stops`, which pins
the C1 tagging of the frame after DLSS-G's last compute copy), one for C6's,
`C6-dlssg-after-tag-copy`, one for C7's, `C7-dlssg-stops`, two for C8's,
`C8-fg-off-sr-other-queue` and `C8-fg-off-color-only`, and four for C5's,
`C5-frame-tag-then-evaluate`, `C5-global-tag-then-evaluate` and their untag forms
`C5-frame-untag-then-evaluate` and `C5-global-untag-then-evaluate`, one for the FG On hold,
`fg-on-holds-sr-copy`, and two for S11c's evaluate, `S11c-fg-until-evaluate-tag-then-evaluate` and
`S11c-fg-global-until-evaluate-tag-then-evaluate`, and one each for the stale rule's own-frame age and
the frame's own depth after a hitch, `hitch-before-sources` and `hitch-after-sources`, and four
for FG On with DLSS-G's first compute copy in flight, `fg-on-dlssg-in-flight-sr-after-hold-expired`,
`fg-on-dlssg-in-flight-sr-stands-in`, `fg-on-dlssg-in-flight-fresh-sr-over-held` (the frame's own
SR copy stands in: with `lineage_snapshot` removed the first two lose depth at FG On's first
Present and the third holds SR depth 188 ms old beside the frame's own, 10-27, temporary build) and
`fg-on-only-dlssg-in-flight-mono` (seed c28965dca8d7c76c's frame 38 made deterministic: its SR input
unprovable, mono is the contract there) and its form after a hitch before frame 21's calls
(`fg-on-only-dlssg-in-flight-held-after-hitch-before-sources`: frame 21's SR copy, made after the
hitch, is held), and seven for FG On with a hitch between a frame's evaluation and one of its
Presents (open items 1 and 2 of the 10-27 review): `fg-on-hitch-generated-before-real` (DLSS-G's
order), `fg-on-hitch-real-before-generated` and its 4x form, `fg-on-hitch-after-present`,
`fg-on-dlssg-in-flight-hitch-after-sources`, `fg-on-live-compute-hitch-after-sources` (each fails
with the add-on's `frame_presenting` reverted, 10-27, temporary build: a gap or a dropped hold at
the hitch frame) and `fg-on-owned-sr-hitch-after-sources` (a live scope refuses the frame's SR copy,
so mono is the contract there after the hitch), four with a hitch between frames
(`hitch-between-sr`, `hitch-between-fg-tag`, `hitch-between-sr-tag`, `hitch-between-dlssg-direct`)
and `hitch-after-sources-then-deferred-tag` (the judge's hold-age rule), and two for a missing
frame (a failed or refused newest nomination, whose completed predecessor stands in:
`refused-fg-attempt-keeps-completed-copy`, a live scope's refused DLSS-G attempt between brief
hitches, and `refused-sr-tags-keep-completed-copies`, every other SR tag unprovable while SR runs
on another queue; with the rule reverted the first drops a holdable copy at frame 28 and the second
shows stale depth from frame 34 and then mono, five of five runs each, 10-28, temporary build)),
an event list shaped like the
session it fixed and judged as any
timeline (random seeds meet these shapes only by chance, and a change of the event language changes
every seed), and one witness per open row the timelines found or carry (none is open since C5 closed
on 10-26): an event list whose frames meet the limitation alone, long enough to lose depth on every
run; a witness that loses none fails the set, as the matrix's gap seeds keep a recorded gap's own
failing cell. `--trace` prints every Present's
frame record, the supply it was judged with and each recorded refusal.

`reshade_session_timelines` runs the fixed set (24 seeds and 44 regression timelines, about
5330 frames and 13700 Presents, with the seeds' hitches) in about 240 s, registered like
`reshade_native_provider_continuity` but with a 480 s timeout, about twice a full run (its own
watchdog, 10 s after its 460 s bound `fixed_budget_ms`, ends a hung run first). Locally:

    reshade_game3d_session_timeline_runtime_test [--list] [--timeline <seed> | --events "<list>" |
      --seeds N [--from M]] [--shard i/N] [--shrink-runs N] [--trace]
      official.dll frozenShaders test.addon64 freshOutput frame_generation_interposer.dll

`--list` prints each selected timeline's seed (or regression or witness) and event list;
`--timeline` replays one seed and `--events` one list (frames separated by `;`, each `defaults` or
`name=value,...`, `N*` repeating one); `--seeds N --from M` soaks N further seeds (with the
regressions and witnesses). Soaks over the 350 seeds after the fixed
set (`--from 24`) found the shapes of C4, C6 and C7 beyond their witnesses; seeds 279 to 373 then
passed with no failure (in the event language before Presents inside an SR evaluation, which
changed every seed).

The lifecycle test
([`test_addon_unload_runtime.cpp`](test_addon_unload_runtime.cpp)) covers the add-on across
ReShade's unload, which comes whenever a game destroys its last device, and the reload for the next
one (the crash fixed by 5148a6da: Assassin's Creed Shadows' startup device). It includes the
timelines' fixture and drives one process, one hidden window and these devices:

1. a startup device with no queue, swapchain or frame, destroyed while its initialization's module
   discovery and content-sampler compile run on the pool. The discovery is held before it scans
   (`SunshineSbsTestHoldDiscovery`, armed by the fixture's own `init_device` callback, which ReShade
   calls before the test add-on's): this process exports the NGX stand-in, so a discovery that ran
   would hook it within the startup device's lifetime, and a hook pins the add-on as its owner,
   leaving AddonInit's own pin untested. Held, it is in flight at the unload as in Assassin's Creed
   Shadows, and the module stays mapped only through AddonInit's pin;
2. the game's first device, through the FX boot and Streamline discovery until Game 3D renders;
3. a device of another swapchain size, destroyed while its shader entries compile on the pool (the
   add-on's shader store is isolated in the output directory, so each size compiles);
4. a device destroyed while its renderer builds on the pool (the test add-on carries the renderer's
   build hooks, inert until set, and delays each step by 40 ms);
5. four game devices in turn, on one Streamline viewport, each verified by a session timeline,
   then destroyed with frames in flight: the add-on's copies of the last frames not yet observed,
   DLSS-G's compute copies whose ring entries were never reset, and a last frame recorded and
   submitted but never presented, whose command-less FG depth tag (the oracle names its copy
   deferred to that Present) and UI tags leave copies deferred to it.

Each device also leaves one held pool callback of the add-on's code (`SunshineSbsTestHoldPoolWork`),
and as ReShade destroys it a `destroy_device` probe queues 128 lines in the add-on's asynchronous
log writer (`SunshineSbsTestLog`). After each unload the module must still be mapped, the held
callback must compile and return, the 128 lines must be written in order, the shader compiles must
finish (`SunshineGame3DTestShaderEntriesPending`) and the renderer builds must be released
(`SunshineGame3DTestRendererBuilds`), and the next native device must carry no private data the test
set on the one before (D3D12 returns an adapter's existing device while a reference keeps it alive).
When the game releases a device, nothing else may reference ReShade's proxy (the test counts its
references besides the device's own, its presenting queue's and its swapchain's), ReShade must
destroy it and unload the add-on whatever its session captured, its native object must go with the
proxy (ReShade logs `Reference count for ID3D12Device` as inconsistent otherwise), and the capture
owner must log that it released everything of it (`Sunshine capture: released device`, never `kept
device`). Game 3D must render on each game device, reloaded after the unloads before it, and each
game device's timeline (30 frames of SR depth with the game's HUD-less and Backbuffer tags, 31 of FG
depth tags, 10 of DLSS-G compute evaluations, then 31 of SR depth with the tags on the game's list)
must pass the continuity invariant and the oracle and render Game 3D from API depth, show an FG
scope's depth and offer the UI pair to detection. Each game device of item 5 after its second must
leave the process's handles (64), threads (8), GDI and USER objects (4) and private bytes (32 MB)
within those bounds of the one before.

The game devices of item 5 are the witnesses of the former known gap L1, found by this test and
closed on 10-26: a device whose session captured depth outlived the game's release of it. ReShade's
`ID3D12Resource::GetDevice` hook returns its device proxy with a reference, and the add-on's capture
storage and source references kept those (8 in this test) until a later Present reused or retired
the storage. When the game stopped presenting and released the device, no `destroy_device` event
came: ReShade neither destroyed the device nor unloaded the add-on, and the game's next
`D3D12CreateDevice` returned the same device. The capture owner now names a device without owning it
and, at `destroy_device`, waits for its queues' GPU work and releases every source and snapshot of
it wherever they are still held (`depth_capture::release_device`, [Streamline depth
selection](../../docs/reshade-sbs.md#streamline-depth-selection)). The capture test's `device_release`
([`test_diagnostic_texture_capture.cpp`](test_diagnostic_texture_capture.cpp), the cheapest layer)
checks it on a device without ReShade: a retained source adds no reference to its device;
`release_device` waits for a copy behind a gate on its queue (debug-layer clean), then releases a
snapshot a packet still holds, one a ticket still holds, the sources an adapter still holds and the
queue places, and clears the device's identity; held objects are refused afterwards; and a copy that
does not finish within the wait bound keeps all of them. A retired queue whose copies are still in
flight keeps its place until they retire, so ReShade still logs its presenting queue's `Reference
count for ID3D12CommandQueue` as inconsistent once per cycle; the place goes with the device at the
latest. While the device was reused, its fourth session (the fifth game device) lost FG depth
(`FG_scope_required`, mono) while the first 8 queues to execute a game list kept their entries in the
game-execution table (`depth_capture::game_executions_on`) for good: its presenting queue had none
(fixed by ad1f15a6, the Assassin's Creed Shadows hotfix of 10-26).
Device removal is not simulated; `release_device` reads a removed device's fences as complete.

Changing the Streamline viewport is outside the lifecycle: in one session, a new viewport's FG scope
showed no FG depth (`source_handoff`, mono) for a whole timeline while an earlier viewport that used
FG had not been retired by the timelines' isolation (`retired_viewports`, the probe's four viewport slots, scratch viewports'
constants), whether or
not the device was recreated. The lifecycle keeps one viewport, as games do.

`reshade_addon_unload_runtime` runs it with the 2.7.30 interposer in about 32 s, registered like
`reshade_native_provider_continuity` (200 s timeout). Locally:

    reshade_addon_unload_runtime_test official.dll frozenShaders test.addon64 freshOutput
      frame_generation_interposer.dll

These tests were proven against the Assassin's Creed Shadows fixes (10-25): the test add-on built
from batch A (18e5d922, before the hotfix) and from master with each fix's production change
reverted (5148a6da reverts cleanly; the others by hand, as later commits changed their lines) fails
them, while master's passes. The timelines' first failing seed shrinks to:

- 5148a6da, AddonInit's pin: `reshade_addon_unload_runtime` at the startup device ("unmapped the
  add-on; AddonInit must pin the module"); batch A also crashed there in the pool work it held.
- 9146a468, FG owning its viewport only for the frames it supplied: continuity gaps,
  `defaults; fg=on,sr=local` (mono, `FG_scope_required`).
- 88b45edf, FG's copy held over SR: continuity gaps,
  `fg=on,tag=present_bare; tag=present_bare,untag=after_frame,sr=other_queue` (`source_changed`).
- 64d6d06c, an evaluation's local input current through other Presents: the oracle at the SR
  input's copy (`source_lifetime`),
  `fg=on,sr=local; sr=local,state=unprovable,generated=3,at=evaluate`.
- 46a4510e (with S22, which also copies on compute lists), DLSS-G's compute-list copies: the oracle
  at DLSS-G's input (`command_type`), `fg=on; dlssg=compute_unobserved`.
- ad66b1c8, a compute-list copy taking its viewport only once readable: continuity gaps,
  `tag=now_list; fg=on; dlssg=compute`.

The proof found three blind spots, closed in the tests: the NGX stand-in's hook pinned the add-on
before the startup device's unload, so the lifecycle passed without 5148a6da (now the discovery
gate); no timeline put a Present inside an SR evaluation's copy (now `at=evaluate`); and C1 excused
any loss of a frame tagged with it, which hid 88b45edf's and ad66b1c8's (now only once no readable
copy is younger than the source age, which also found C8). Each fix also has its regression
timeline, since a change of the event language moves which seeds meet it.

A review of these tests (10-25) mutated the add-on and found three more, closed the same way. With
`decide_display` holding nothing at a repeated frame, every generated Present was mono and one seed
of 24 failed, by chance: a generated Present carried no obligation and disarmed the judge, and a
disarmed judge gave legal sources 2 s, longer than most timelines (now the generated Presents' hold,
a source age outside warm-ups, and `repeated-frame-hold`; 23 seeds and the regression fail it,
shrinking to `defaults; fg=on; dlssg=direct,generated=2`). With an untag that never cancels its
frame's deferred copy (R7), one seed failed through a stale symptom: R7's outcome is a HOLD, which no
decision check judged, and the refusals it displaced counted as preempted (now the deferral check,
and no preemption outside the FG scope; 11 seeds and `88b45edf-stray-fg-copy` fail it, shrinking to
`fg=on,tag=present_bare,entry=global,untag=after_global`). And rows C4 to C7 excused losses within
the source age of a readable copy (now the readable-age rule of every row).

### Characterization tests

The contract matrix judges what an API permits. Characterization tests instead pin what the add-on
does today at the seams the matrix does not reach, so that a structural move which must not change
behaviour is checked row by row. They are not an oracle: a row changes only in the commit that
changes that behaviour on purpose, and a disagreement with an API contract belongs in the matrix as
a known gap. Each table names the behaviour of every row and the refactor phase it protects.
`ctest -L characterization` runs them; each also runs in the default `ctest`.

- [`test_game3d_seam_characterization.cpp`](test_game3d_seam_characterization.cpp)
  (`reshade_seam_characterization`) links the Streamline probe tests' view
  ([Test builds](#test-builds)) and drives the capture owner without a device: the frame
  generation policy's query timelines, the FG scope's liveness (accepted copies from a Streamline
  FG tag or from DLSS-G's own evaluation) and the selection the provider derives from it
  (`provider::select_frame_generation_scope`), NGX's copy rule under FG
  (`sunshine_ngx::frame_generation_owns_depth`), `select_provider` with a protected owner,
  readable and pending sources and required or excluded FG, per-queue ownership and
  `establish_provider`'s watermark (also a handover to a copy recorded before the previous owner's), a single FG copy's handover to SR snapshots that are readable
  only frames later (the held FG copy, `frame_generation_holds`, and the shared watermark), which
  Presents present a stamped frame (`frame_presenting`: its first, and with FG the declared count
  after it for a frame that made a frame-generation call, until the next evaluation; a copy's
  submission on a presenting queue as its frame, `presented_by`), the frame's own SR stand-in for
  its DLSS-G copy in flight at those Presents after a hitch, never at a later one nor an earlier
  frame's, a missing frame's stand-in (each kind of failed or refused head, SR and FG, against
  every per-copy check and every identity interruption, which still revoke) and a nomination in
  progress's (the same copy, at once), the source a head refused before its capture existed names to
  the display (another viewport's or NGX feature's drops the retained copy), and every reason of
  `decide_display`. Its freshness table
  pins each age limit one millisecond either side against an explicit clock: the source age at
  every selection site, the association timeout of `expire_association` and `evaluation_live`, an
  FG scope's enabled and accepted ticks, which also admit a tick ahead of the clock (the capture
  owner's `scope_clock` fixes `frame_generation_scope_live`'s clock in this view only), and the
  provider's handoff limit in `provider::handoff_hold` and the `evaluation_live` window it passes.
  Each limit is named once in [`freshness.h`](freshness.h). Its
  UI table pins the UI adapter's admission of each ready UI snapshot (`ui_mask::admit_candidate`)
  and of the Present's own color (`ui_input::current_color_refusal`). Their contracts are in
  [Streamline depth selection](../../docs/reshade-sbs.md#streamline-depth-selection) and the
  [UI decision framework](../../docs/reshade-sbs.md#ui-decision-framework). It shares the capture
  owner's slot and evaluation fixtures with that owner's regressions
  ([`test_streamline_depth_capture_fixture.h`](test_streamline_depth_capture_fixture.h)).
- `reshade_upscaler_call_trace` holds the NGX nesting table
  ([`test_upscaler_call_trace.cpp`](test_upscaler_call_trace.cpp)): each outcome of an outer
  evaluation against a nested SR or DLSS-G call, including an outer copy the capture owner
  rejected (the inner call supplies depth) and an outer recorded copy (the inner call is
  deduplicated), a live FG scope and a claimed Streamline evaluation
  ([Direct NGX depth selection](../../docs/reshade-sbs.md#direct-ngx-depth-selection)). The test
  states whether a live FG scope owns depth through a callback that only the call-trace build has
  (`sunshine_ngx::testing::callbacks`), in place of the Streamline FG query; the seam
  characterization pins the production rule. It also pins each nomination's state proof: Super
  Resolution's contract state for SR and for a registration recovered from a map without
  `DLSS.Use.HW.Depth`, and only an observed state for Ray Reconstruction and for a recovered
  registration with that key (N1b).
- [`test_game3d_hook_set.cpp`](test_game3d_hook_set.cpp) (`reshade_game3d_hook_set`) pins the
  shipping add-on's installed hook set, which hook host moves must keep: the binary is its own
  process's ReShade, runs `AddonInit` and `AddonUninit`, and compares the ReShade events each
  registers and unregisters, in order, with its tables, checks the one overlay, and requires that no
  MinHook hook is created without Streamline or NGX. It also pins the native observer's vtable
  methods (`native_observer::hooked_methods`): `ResourceBarrier` (26), `ExecuteCommandLists` (10),
  CL7 `Barrier` (80) and, since Step 4, queue `Signal` (14) and `Wait` (15) and fence `Signal` (10). Each MinHook family's hooks on fixture exports
  are pinned where they are installed, through
  [`test_minhook_ledger.h`](test_minhook_ledger.h) (MinGW links those tests with
  `-Wl,--wrap=MH_CreateHook`): the Streamline probe test (v1, v2 with and without its optional
  exports, the FG options entry and the PCL marker) and the call trace (the four NGX exports).
- [`test_provider_log_lines.cpp`](test_provider_log_lines.cpp) (`reshade_provider_log_lines`)
  renders the API depth provider's status lines from fixed values for representative Presents
  (a settled fresh copy, a held pending successor, the release to Generic and the handoff hold, an
  FG scope turning silent and live, a readiness loss and recovery with their observation evidence,
  and dynamic resolution) and compares them byte for byte with
  [`provider_log_goldens/status_lines.txt`](provider_log_goldens/status_lines.txt). The provider
  gathers each line's values where it decides and renders them through
  [`streamline_depth_provider_log.h`](streamline_depth_provider_log.h), so a move of how the values
  are gathered keeps every line. A commit that changes a line on purpose regenerates the goldens
  with `reshade_provider_log_lines_tests <goldens> --update`.
- `reshade_game3d_alpha_auto` pins the acceptance ledger's edges that its policy groups leave
  unexercised: a sample over no pixels, a contradiction older than the newest one, and a
  testable sample older than the previous one
  ([UI decision framework](../../docs/reshade-sbs.md#ui-decision-framework)).
  `ui_selection::decide` itself is covered in full by the selection contract and sequence tests.
- [`test_scene_depth_source.cpp`](test_scene_depth_source.cpp) (`reshade_scene_depth_source`) pins
  the legacy `source_id` that logs and Dump 3D packages carry for each `source_key` of
  [`scene_depth_source.h`](scene_depth_source.h): FG depth `(1 << 63) | viewport` from
  `fg_depth_key`, Streamline scene depth 0 and NGX its feature generation. Through its own inverse
  it checks the round trip and that no two keys a frame carries share a provider, viewport and
  `source_id`.
- [`test_freshness.cpp`](test_freshness.cpp) (`reshade_freshness`) pins each named time limit of
  [`freshness.h`](freshness.h): its value and, for each edge its call sites keep (below or at most
  the limit, with or without a tick ahead of the clock, a zero tick unavailable), the outcome one
  millisecond either side of it. It is the only pin of the Streamline probe's camera, query and
  tag-precedence limits, whose sites read their own clock.
- [`test_api_contract.cpp`](test_api_contract.cpp) (`reshade_api_contract`) pins each API's mapper
  in [`api_contract.h`](api_contract.h): a Streamline tag's lifetime by version, lifecycle and copy
  point (its tag call or a later evaluate, the one difference between the depth and UI readings),
  its state proof, the lifecycles the probe reads and `match()` judges, the NGX, DLSS-G and Generic
  contracts (a depth clear's contract state, the end of the frame's last proven state, a bound
  view's `DEPTH_WRITE` or, read-only, no state, and the other boundaries' declaration), Dump 3D's
  stricter contract for its optional UI copies and an absent extent as the whole allocation. The
  matrix's L1 hook cells drive the same mappers through the
  probe.
- [`test_queue_order.cpp`](test_queue_order.cpp) (`reshade_queue_order`) first pins the fixtures'
  script oracle ([`test_queue_order_script.h`](test_queue_order_script.h), `unordered_read`): a
  wait after the producer's signal orders, and a wait for an earlier signal, a CPU signal or a
  second signaller reaching the value (before or after the read), the initial value, an unknown
  producer and an in-flight read without a wait do not; a two-hop chain does. It pins the queue-order model
  of [`queue_order.h`](queue_order.h) ([Queue order](../../docs/reshade-sbs.md#queue-order)): one
  cell per fail-closed rule (a signalling queue never the least recently used record, baseline, unsettled and the settle marker above every seen value, CPU
  signal, second signaller and its cookie across a reset, decreasing and `UINT64_MAX` values, a
  completed value no logged signal set, a signal in flight that is not unobserved, a queue without
  a record, device removal, foreign and cross device, ring eviction, a reset forgetting positions
  while a re-sighted fence keeps its poison, late conflicts and their expiry, and each way a
  relied read fails closed: the decisive fence after four other waited ones, a chain longer than a
  verdict holds, a model reset, an evicted fence record, a fence poisoned between the verdict and
  the read, and a live read evicted by newer ones), the positive chains
  (after-signal, wait-before-signal, two and three hops) and the negative ones (a wait satisfied
  before the producer, a non-quiet producer or hop wait, a hop wait after the hop's signal). An
  INDEPENDENT brute-force oracle then explores every GPU interleaving of 20000 generated scripts
  (two or three queues, one or two fences, four to ten Execute/Signal/Wait/CPU-signal calls) and
  requires that every verdict the model gives as ordered is ordered in every interleaving
  (soundness; at least 200 such verdicts), and fixed positive scripts must be ordered by both
  (completeness). Dropping the first-signal resolution, the hop quiet rule, the settle, baseline,
  second-signaller or overlap check fails a cell.
- `reshade_queue_order_device` ([`test_queue_order_device.cpp`](test_queue_order_device.cpp), label
  `device`) runs the queue-order model on real queues of the first hardware adapter (else WARP)
  under the D3D12 debug layer, through the native observer's hooks as the capture owner installs
  them. The producer queue renders behind a gate the CPU opens only after the verdict, so every
  read is still in flight when the model decides: a reader that waits after the producer's signal,
  and one whose wait was submitted before that signal, are ordered and read back exactly what the
  render wrote; a wait satisfied by a signal before the producer, a CPU signal, a second signaller
  and a `D3D12_FENCE_FLAG_SHARED` fence each refuse. The debug layer stays clean (a refused cell's
  reader reads nothing, as its read would race the render). It skips (77) without the debug layer.
- `reshade_streamline_native_observer` ([`test_streamline_native_observer.cpp`](test_streamline_native_observer.cpp))
  also runs the order family's hooks on COM-shaped fixtures: the method table; Signal and Wait
  discovered by `observe_queue`; a fence recorded only once its CPU Signal slot is installed and
  every verdict refused while it is pending (with no capture evidence revoked); a real edge; a
  suppressed `ExecuteCommandLists` taking its own ticket; the quiet bit for two threads on one
  queue; a failed Wait not logged; a shared fence; a failing CPU Signal poisoning its fence, which
  stays poisoned after its records were dropped; a third party's Signal slot resetting the order
  model only; and `QueueFenceOrder=0`. Its review cells (10-27): a Signal whose queue record a full
  table evicted between its ticket and its log is logged on the re-sighted record (dropped, the
  reader's pending wait resolved to the next signal, after a copy-out it precedes); a CPU Signal
  poisoning a fence on another thread while a queue Signal writes its first signaller keeps its
  poison in the private data and through a model reset (the write ran after the lock was released);
  Signals and Waits on a fence whose CPU Signal slot was rejected read no slot address (each read
  two and repeated the discovery lookup); a CPU Signal slot a third party replaced after a proven
  edge refuses the next verdict as `uncovered` before any poll (it stayed ordered until the poll).
- [`test_proven_state.cpp`](test_proven_state.cpp) (`reshade_proven_state`) pins the per-resource
  last-proven state of [`proven_state.h`](proven_state.h), which admits the Generic end-of-frame
  copy: a creation state holds on every queue until a list uses the texture and not once enhanced
  barriers were seen, an executed list's proof holds only on its own queue, every update takes a
  new version, unseen changes leave a texture unproven, a full table forgets its least recently
  used texture (one forgotten latched stays latched when held again while its record keeps it; an
  overflowing record latches every texture not held; a destroyed one is forgotten with its latch),
  and a depth clear keeps the regime of the texture's proof (an enhanced layout stays enhanced; a
  latched texture without a proof has none). The capture owner's feeding of it is
  proven on real lists by the device layer's end-of-frame cells and by
  `reshade_diagnostic_texture_capture_test`'s `proven_state_end_of_frame` (executed barriers and
  depth clears per queue, a copy discarded when its proof changed or another queue ran it, unproven
  after an unobserved list or a wildcard alias, an untracked depth-stencil source tracked again by
  Generic's copy) and `enhanced_layout_latch` (a depth clear after an enhanced barrier, and a later
  clear-only list, prove the `DEPTH_STENCIL_WRITE` layout, so the end-of-frame copy uses enhanced
  barriers under the debug layer; a destroyed latched texture is forgotten with its latch).
  Its `dlssg_compute_list_capture` checks DLSS-G's copy on its own COMPUTE list under the debug
  layer (N7): a depth-stencil depth observed in `NON_PIXEL_SHADER_RESOURCE` there is copied there
  with exact bytes into storage resting in COMMON, unobserved it is refused as `missing_state`, a
  Streamline FG tag input there is copied as well (S22, never refused as `command_type`), the
  compute queue takes a fence at its first submission of the copy, and the direct consumer reads it
  only once the compute recording was reset.
- [`test_read_only_dsv.cpp`](test_read_only_dsv.cpp) (`reshade_read_only_dsv`) pins the read-only
  depth-stencil view table of [`read_only_dsv.h`](read_only_dsv.h), which decides the Generic
  path's unbind and clear-like draw of a bound view (D11c): a view created read-only stays so until
  a writable view is created at its handle, a writable view is writable, and a handle the table
  does not hold (never reported, or with no slot left) counts as read-only. The capture owner's
  refusal of such a copy is the device layer's `unbind_read_only_dsv` cells, and the direct
  Streamline fixture checks the wiring under ReShade: a DSV created with `READ_ONLY_DEPTH` reads as
  read-only in the table and where a list binds it, a writable one does not, a read-only one copied
  with `CopyDescriptorsSimple` into a handle no view was created at reads as read-only, and a
  writable view created again at the read-only view's handle replaces it
  (`SunshineDepthTestReadOnlyView`).
- Compute lists (S22, N7): [`test_capture_state_policy.cpp`](test_capture_state_policy.cpp) pins
  `copy_state::list_basis` (the states and layouts a COMPUTE list may name, none of a compute
  queue's own layouts on a DIRECT list) and the copy's compute-queue source layout;
  `reshade_diagnostic_texture_capture_test`'s `compute_list_capture`, under the debug layer, copies
  NGX's contract state on an evaluation's own compute list and reads it back exact on a direct
  consumer queue only after the compute recording was reset (the compute queue takes its fence at
  that submission), lets an observed state win there and refuses a direct-only declared state for
  its state; and the capture-owner self-test (`reshade_streamline_camera_probe`) accepts a
  compute-list FG copy for its scope only once readable (submitted with the fence, reset, fence
  passed), never an auxiliary one, and a direct-list copy at its call; an accepted compute-list copy
  whose producer fails later leaves its frame missed, a consumer's failure does not.

### Independent renderer reference tests

The independent shader's opt-in runtime tests remain regression references. They load the official
ReShade DLL and execute the bundled reference effect. Fresh output directories keep their configuration,
copied shader and logs attributable to the tested build:

```powershell
.\cmake-build-reshade\reshade_native_stereo_runtime_test.exe C:\test\ReShade64.dll `
  .\tools\reshade\Shaders .\cmake-build-reshade\native-dx12-scrgb scrgb 0
.\cmake-build-reshade\reshade_native_stereo_runtime_d3d11_test.exe C:\test\ReShade64.dll `
  .\tools\reshade\Shaders .\cmake-build-reshade\native-dx11-scrgb scrgb 0
.\cmake-build-reshade\reshade_native_selection_runtime_test.exe C:\test\ReShade64.dll `
  .\tools\reshade\Shaders .\cmake-build-reshade\native-auto `
  .\cmake-build-reshade\SunshineSBS.addon64 scrgb
```

Repeat the renderer fixtures for `srgb` and `pq`. D3D12's final `0` is ReShade PerformanceMode;
use `1` for the dedicated dynamic-calibration/readiness specialization check. Optional width and
height arguments enable 4K source testing. D3D11 additionally accepts a final source-bit-depth
argument: `srgb 0 640 360 10` exercises SDR10, whose texture has no native sRGB view.
Append `--statistics` to the automatic-selection fixture to verify calibration with content-based
Auto-select disabled and a buffer chosen by draw statistics, without a manual override. That
fixture writes `[SUNSHINE_GAME3D] Enabled=0` itself, as reference effects require, and also
reloads the effect while it keeps drawing the same depth: the rebuilt `Sunshine_DepthReady` must
read ready, then unready on a color-only Present, then calibrate again, which checks that the
shared change-only readiness cache republishes into a reloaded effect.
The [owning validation section](../../docs/reshade-sbs.md#validation-and-remaining-device-check)
describes the geometry, color, source-precision, AA and acquisition assertions. CTest includes
the pure calibration policy alongside the existing selector and publisher lifecycle tests.

Set `SUNSHINE_NATIVE_GPU_TIMING=1` for the D3D12 renderer fixture to append a bounded GPU timing
comparison. It uses four warmup frames and eight samples with identical source/depth fingerprints,
Strength 1 and stereo-edge AA enabled. Native GPU timestamps bracket ReShade's effect setup and
COLOR capture copy through completion of the Sunshine technique; they exclude the game's initial
copy into its swapchain, overlay and mono readback.
Compare fresh control/treatment directories with the same fixture executable, dimensions and color.
Use `SUNSHINE_NATIVE_GPU_TIMING=stress` for dense foreground slats, Strength 2 and a padded,
reduced-resolution depth viewport. `SUNSHINE_NATIVE_GPU_TIMING_ONLY=1` runs only the benchmark
and its readiness/export preflight; run the complete correctness suite separately. Neither setting
changes the shipping shader or add-on.
This measures a synthetic effect workload, not the complete game's frame time.

`test_native_shader_compile.ps1 -ValidatorPath <validate_shaders.exe> -OutputDirectory <fresh-dir>`
checks the production FX compiler output/ABI across backend, color and size variants plus explicit
unsupported-input rejections. Its performance-macro variant is a compile check; only the official
runtime fixture exercises real specialization. The validator is a supplied development tool,
not a dependency of the shipping shader or installer.

`reshade_native_present_d3d12_test` takes the same arguments as the legacy presentation fixture
below. It uses a controlled paint effect with the independent effect's filename, technique and
dynamic calibration ABI to isolate transport from stereo geometry. With the separate-process
receiver proxy it checks real shared textures/fences, source timestamps, overlay pixels, reload,
focus loss/recovery and receiver restart. It also poisons the calibration inputs and requires the
publisher to rewrite them on the next prepared frame; readiness, which the shared cache writes only
when it changes, is not poisoned and must read unready; because an unpublished uniform also
defaults to false, this only rejects stale readiness, and `reshade_native_selection_runtime_test`
(including its effect-reload case) is the check that readiness is actually published. This complements the
production shader and automatic-selection fixtures; controlled paint does not test depth warping.

Look in the game's `ReShade.log` for `Sunshine SBS:` messages:

- `exporter ready (export protocol 4, 4 export slots)` means the mapping exists and the add-on is
  waiting for the technique/consumer. If no generation follows while streaming, check that the host
  is of the same release: hosts of protocols 1-3 ignore this mapping without a log line.
- `no compatible export annotations` means the current shader definition or color annotations are absent.
- `generation ..., ... full SBS, DXGI ..., D3D...` means resources were created for a consumer.
- `export inactive (reason, class)` means focus, technique execution or source proof stopped qualifying;
  the reason names the gate, for example `not_foreground`, `present_without_render`,
  `frame_not_prepared`, `no_consumer` or `overlay_capture_failed`. The class (since 10-23) is
  `routine` for a pause the host, the game or the player caused and `fault` for every other reason
  (the [readiness report](#first-run-of-a-new-game) lists them).
- `swapchain window W presents under the foreground window F: owner=V changes=N` names the verdict
  for a game that presents to a child window W of the foreground window F (at most every 5 s, when
  it changes): `owner`, or why not (`partial`, `overlapped`, `not_visible`, `other_process`,
  `not_foreground`, `incomplete`), or `ambiguous` when another swapchain window of the game
  qualifies too (then a top-level window reads `is the foreground window F: owner=ambiguous`, once
  for each episode: the ambiguity clearing is not logged, but a recurrence is a change again). Any
  verdict but `owner` keeps the export inactive with `not_foreground`. `owner`, `not_foreground`,
  `not_visible` and `partial` (focus loss, a hidden or a resizing window) are info lines, as routine
  as a top-level window's `not_foreground` pause; the others are warnings.
- HRESULT messages identify native sharing/fence failures. Repeated allocation failure retries
  are throttled; publication never falls back to CPU readback.

The standalone CTest executable exercises the actual publisher's per-present invalidation,
contended reload, overlay open/reload/close, bounded retirement, queue/swapchain identity and
deferred publication state. It also rejects contradictory/unknown color declarations and stale
PQ-to-scRGB source switches. Its GPU checks import all three RGB10A2 and FP16 textures and the
ready fence from production D3D11/D3D12 allocation code into an isolated D3D11 device and verify
signal visibility.
These checks require D3D11/D3D12 shared-fence support.

An additional opt-in fixture loads the **actual official ReShade 6.8 runtime and production
add-on** into a dedicated synthetic D3D11 application. Supply `ReShade64.dll` extracted from the
[official full add-on support installer](https://reshade.me/downloads/ReShade_Setup_6.8.0_Addon.exe);
the installer need not run. The official DLL is not bundled in this repository. Use a dedicated
output directory: the fixture writes its own ReShade configuration, preset and synthetic shader
there. It does not install anything into a game, start Sunshine, or change display settings.

```powershell
.\cmake-build-reshade\reshade_runtime_test.exe C:\test\ReShade64.dll `
  .\cmake-build-reshade\SunshineSBS.addon64 .\cmake-build-reshade\runtime-srgb srgb
```

Repeat with `scrgb` and `pq` and separate output directories. The test creates a small window and
tries to activate only that window, then closes it automatically. It drives the runtime's real
`D3D11CreateDeviceAndSwapChain` proxy and natural `Present` calls. It verifies the actual
present/technique/finish event order, game color-space reporting, compiled export annotations,
stereo texture dimensions and pixels, effect disable/enable, overlay transitions and shader
reload. It compares the adapted SDK getter with the resource returned through native D3D11 COM.
The synthetic HDR output includes values above SDR white and negative scRGB values; this is a
handoff fixture, not a replacement for the companion shader's PQ/depth conversion tests.

Exit **0** additionally means the production exporter published matching metadata, shared fence
and left/right pixels to an independent D3D11 receiver device, with invalidation and recovery for
effects/overlay/reload. Exit **77** means the fixture could not become foreground: the runtime,
pixels, ABI and events were checked, and the production foreground gate correctly rejected the
background window, but shared publication was skipped. Other nonzero exits fail the test. The
fixture never bypasses the production foreground gate. It has a 45-second process watchdog.

A separate D3D12 ABI probe creates a hidden window and an embedded official effect runtime:

```powershell
.\cmake-build-reshade\reshade_runtime_d3d12_test.exe C:\test\ReShade64.dll
```

It creates a native FP16 texture and an SDK resource view through an output parameter, then
requires the adapted SDK getter to return that exact native resource pointer and verifies its
native `GetDesc`. It uses isolated temporary configuration and empty effect/add-on directories,
never calls `Present`, and never activates its window. A 30-second watchdog bounds its process,
including teardown. This checks the ABI against the real D3D12 implementation; it does not test
D3D12 effect execution or final-SBS publication.

For automated positive publication tests on a desktop that cannot grant foreground focus, build
the separate **test-only** add-on and a C facade around Sunshine's unchanged production receiver:

```powershell
cmake -S tools/reshade -B cmake-build-reshade -G Ninja `
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSUNSHINE_SBS_RUNTIME_TESTS=ON
cmake --build cmake-build-reshade
cmake --build cmake-build-relwithdebinfo --target reshade_receiver_test

.\cmake-build-reshade\reshade_runtime_test.exe C:\test\ReShade64.dll `
  .\cmake-build-reshade\SunshineSBSTest.addon64 .\cmake-build-reshade\controlled-dx11-srgb srgb `
  .\cmake-build-relwithdebinfo\tests\reshade_receiver_test.dll
.\cmake-build-reshade\reshade_runtime_present_d3d12_test.exe C:\test\ReShade64.dll `
  .\cmake-build-reshade\SunshineSBSTest.addon64 .\cmake-build-reshade\controlled-dx12-srgb srgb `
  .\cmake-build-relwithdebinfo\tests\reshade_receiver_test.dll
```

The host build directory must already be configured with its normal dependencies. Repeat each
runtime test with `scrgb` and `pq`, using separate output directories. These controlled tests keep
their windows hidden and supply foreground observations for only their own window. They drive
natural game `Present` calls through the official D3D11 or D3D12 proxy. The real receiver performs
nonce negotiation, process checks, handle duplication, shared-fence checks, slot ownership and
direct reads of the claimed shared slot; the fixture does not modify the receiver's slot state. Pixels, declared
transfer, retained timestamps, effect/overlay/reload/focus invalidation, recovery and receiver
restart are checked. Producer and receiver run in one process on independent native devices;
these tests do not establish access across a live game's process or host-service boundary.

Set `SUNSHINE_GAME3D_NATIVE_ONLY=1` for the controlled D3D11 fixture to run native Game 3D with
no FX file or technique installed. It requires the production receiver facade and checks the
add-on's direct slot export, the overlay's copy path, effects toggle, focus recovery and
receiver restart with exact SDR/scRGB/PQ pixels, then a colour toggle through ResizeBuffers and back
that must reuse the first transfer's renderer and export ring. `reshade_game_present_d3d12_test` accepts the same
flag.

To exercise the same checks with the production receiver in a separate process, build its
test proxy and helper in the host build directory:

```powershell
cmake --build cmake-build-relwithdebinfo --target reshade_receiver_process_test
.\cmake-build-reshade\reshade_runtime_test.exe C:\test\ReShade64.dll `
  .\cmake-build-reshade\SunshineSBSTest.addon64 .\cmake-build-reshade\process-dx11-srgb srgb `
  .\cmake-build-relwithdebinfo\tests\reshade_receiver_process_test.dll
```

Use the D3D12 presentation executable for that backend, and repeat with `scrgb` and `pq`.
The proxy starts its own hidden helper with a restricted set of inherited IPC handles and
verifies its distinct process identity. The helper selects the same GPU and alone runs the
unchanged production receiver, including the real cross-process `OpenProcess`, `DuplicateHandle`,
texture/fence import and direct shared-slot read. While the receiver holds the claimed slot
`reading`, the test copies that slot to a staging texture and sends its pixels to the parent for
inspection; this CPU path exists only in the fixture. Production texture transport remains
entirely on the GPU. Receiver restart starts a new helper process.
The logs record both PIDs and helper exit status. These checks cover separate processes under
the same user/session/integrity level, not a live installed game or the elevated host-service
boundary. All six D3D11/D3D12 SDR/scRGB/PQ cases passed locally again on 2026-10-05 with the
direct-read receiver, including receiver restart in a new process and clean shutdown of all
twelve helper instances.

`SUNSHINE_SBS_RUNTIME_TESTS` defaults to **OFF**. Its `SunshineSBSTest.addon64` is for these isolated
fixtures only. The shipping `SunshineSBS.addon64` retains the real Windows foreground check and
does not export the test observation setter; `reshade_addon_exports` checks that it exports only
ReShade's add-on entry points. Do not install the test add-on into a game.

### Isolated comparison with the current Host warp

`test_host_warp_comparison.cpp` is an opt-in hardware D3D11 experiment. It consumes
the exact FP16 linear scRGB color and R32 raw-depth inputs retained by the Game3D
parity fixture. It runs the current production Host V2 vertical/horizontal
conditioning and live inverse shaders, using a small native-depth adapter to
retain Game3D's scale, zero plane and strength. It does not run DAV2, test camera
calibration, or establish live producer authorization. Its explicit renderer
authorization is confined to the test state; no production gate is relaxed.

Build with the existing Windows UCRT64 toolchain:

```powershell
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -static tools/reshade/test_host_warp_comparison.cpp -o cmake-build-reshade/host_warp_comparison.exe -ld3d11 -ld3dcompiler -ldxgi -lbcrypt
cmake-build-reshade/host_warp_comparison.exe <host-directx-shader-directory> <parity-input-directory> <fresh-output-directory> <width> <height> <H> <q0> <referenceZPD> <strength-0-to-100>
```

The adapter is restricted to oriented raw depth (`A=0`, `inverseB=1`), full-frame
depth, zero jitter and blend=1. Verify the supplied controls against the retained
parity uniforms. Compare measured per-eye displacement before making quality
claims. The production limiters' unchanged maximum dimension is enforced; this
fixture is not a full-resolution 4K integration. Inputs outside the Host parallax
container fail instead of being silently clipped. GPU intermediates, compiled
shaders, complete shader-source closure, output images and hashes are retained.
Verify that closure against the production sources when reporting a run.

`SUNSHINE_STEREO_PARITY_FRINGE_WIDTH=1..8` optionally widens the parity fixture's
50% mixed-color source fringe (default remains two source pixels). The explicit
width contract must match both sides. The existing edge diagnostic's four-pixel
exclusion margin does not cover every wider-fringe case; measure the actual mixed
strip separately rather than treating that metric as hidden-background truth.

The [2026-09-18 experiment](../../cmake-build-reshade/sunshine-warp-game3d-20260918/RESULTS.md)
records matched native-depth GPU comparisons and the distortion tradeoff. It is
synthetic evidence, not game/headset or performance acceptance.

### Full-pipeline shader tests

For the historical current-zero ratio fixture, set `SUNSHINE_GAME3D_AUTOMATIC=1`,
`SUNSHINE_DEPTH3D_EFFECT=SunshineGame3D` and
`SUNSHINE_GAME3D_STEREO_REFERENCE_TEST=1`, then run:

```powershell
reshade_depth3d_runtime_d3d12_test.exe <ReShade64.dll> <supplied-shader-directory> <fresh-output-directory> scrgb 0 3840 2160
```

Repeat with `pq` and with `1920 1080` only against the matching frozen implementation. The fixture
measures exported eye shifts, unit/encoding invariance, zero disparity and strength 0/50/100.
Its historical independent oracle is
`0.05*(1-q/q0)` before strength. Room → wall → return changes normalization with the
screen plane; different starting scenes must converge to the same current geometry.
Projection inputs are synthetic and explicitly trusted; these tests do not validate a live
source association. The earlier [fixed-reference evidence](../../cmake-build-reshade/unit-independent-reference-20260917/stereo-reference-README.md)
is historical. It retains complete-export hashes and a 640×360 failure in the old shader's
low-resolution ray interpolation. These records predate the current independent-gain and
contrast-midpoint zero trial and do not validate it.
The [current-zero shader evidence](../../cmake-build-reshade/zero-plane-reference-20260917/stereo-reference-README.md)
passed 1080p scRGB and 4K scRGB/PQ, including the independent ratio oracle, complete-export
unit/encoding parity and different-initial-scene convergence. It is shader/controller evidence;
the current policy requires its own source-integration and live-game evidence.

The same opt-in build includes a full-effect D3D11 fixture for the separately obtained original
Depth3D reference. It needs no exporter or foreground observation override:

```powershell
.\cmake-build-reshade\reshade_depth3d_runtime_test.exe C:\test\ReShade64.dll `
  E:\Git\Repo\Depth3D\Shaders .\cmake-build-reshade\depth3d-scrgb-compat0 scrgb 0
```

The default/unset `SUNSHINE_DEPTH3D_EFFECT` selects `SuperDepth3D`. The original-reference assertions
below exercise its controls, including ZPD; they are not the automatic-only Game3D control contract.
Use the current Game3D camera/raw fixtures above for source-owned depth inputs.
Replace the external shader path with your checkout.
Repeat for `srgb`, `scrgb`, `pq` and final
argument `0` or `1` (`HDR_Compatible_Mode`), using separate output directories. The fixture copies
the user-supplied shader and its dependencies into that isolated directory, then loads the full
effect in the actual runtime. It checks annotations, native full-SBS texture dimensions/format,
the original renderer, reference color pixels and byte-exact mono presentation. A native R32
depth texture is bound through ReShade's `DEPTH` semantic, with `bufready_depth` supplied before
the actual shader passes. The export checks require:

- Equivalent normal/reversed depth inputs produce matching prepared depth.
- Retired renderer controls and resources are absent from the compiled effect.
- Changing Depth Adjustment between nonzero strengths changes measured opposite-eye source
  shifts; ZPD changes the direction/amount of separation. Checks account for the original
  POM compatibility offset and inspect final stereo pixels, so a zero/nonzero-only response
  cannot pass as a working strength control.
- The shared Normal Depth view matches linear scene depth before convergence and repeats
  a complete, finite, monotonic grayscale map in both eyes without isolated bright columns.
  Its pixels must remain identical across Depth Adjustment 0/100 and ZPD
  0/0.8 with unchanged input. A separate oracle checks the known reversed-Z ramp's linear depth.
- Sharpening alters the final export, and disabling it restores the unprocessed result.
  Original SuperDepth3D reference checks also retain its AA response and reversal checks;
  current Game3D explicitly requires that the retired final-AA uniform is absent.
- Losing depth readiness returns flat duplicate eyes.
  The full native mono raster stays byte-exact throughout.

Run the original reference's D3D11/D3D12 × SDR/scRGB/PQ × HDR-compatibility permutations. These checks exercise synthetic
depth through the real full effect; game-specific appearance, depth selection and performance
remain physical acceptance work. No Depth3D source is bundled in this build. Prior original-effect
results are not acceptance evidence for the renamed effect or a subsequent color correction.

Current Game3D fixtures require `USE_AA` to be absent. For a frozen older Game3D control only,
set `SUNSHINE_GAME3D_LEGACY_FINAL_AA=1`; that explicitly requires the old uniform. Do not use
this opt-in with Performance Mode, which may specialize uniforms away. Original SuperDepth3D
reference checks keep their own AA contract without this flag. Set
`SUNSHINE_STEREO_PARITY_AA0_ONLY=1` on both sides of a final-AA removal comparison, with the
legacy flag only on the old side. This keeps controls, clocks and captures matched to AA off;
it does not relax image comparison. Current camera/acquisition checks run the one supported
final-AA state, while explicit older-source runs retain their AA0/AA1 coverage.

The optional `reshade_depth3d_runtime_d3d12_test` runs the full supplied effect through
natural D3D12 swapchain `Present` calls in the official runtime, also in a hidden window:

```powershell
.\cmake-build-reshade\reshade_depth3d_runtime_d3d12_test.exe C:\test\ReShade64.dll `
  E:\Git\Repo\Depth3D\Shaders .\cmake-build-reshade\depth3d-dx12-scrgb-compat0 scrgb 0
```

It takes the same arguments and six SDR/scRGB/PQ × `HDR_Compatible_Mode=0/1` permutations,
and exercises the same shared-preparation, control-response, diagnostic, exported postprocessing
and depth-loss checks on D3D12. Its zero-strength reference must contain a nonconstant source
pattern and two complete matching eyes, so blank output cannot satisfy later comparisons.
Both full-effect fixtures default to 640×360 and use isolated user-supplied shader copies,
with 70-second D3D11 and 180-second D3D12 process watchdogs. The D3D12 fixture also accepts
optional final source-width and source-height arguments up to 3840×2160. The full 4K scRGB
case passed on 2026-09-12, including measured displacement, identical diagnostic eyes without
isolated columns, exported AA/sharpening and byte-exact mono. This is not a game performance
measurement; its correlation oracle samples every pixel on three selected rows.

The strengthened Normal Depth regression rejected the pre-fix shader on ZPD invariance,
then passed the corrected shader's full 4K D3D12 scRGB run on 2026-09-12. The
former add-on depth logging also passed actual-runtime D3D12 scRGB and D3D11 SDR
checks, along with lifecycle tests. This verifies the diagnostic and logging behavior,
not the cause of the reported Dead Space depth-input failure.

Local validation with the official 6.8.0.2155 DLL passed all six controlled D3D11/D3D12 tests in
SDR, native scRGB and HDR10/PQ with exit **0**. The production-add-on D3D11 runs returned **77**
because this execution desktop could not grant foreground focus; its real background rejection
was verified. Standalone sharing tests and the actual-runtime D3D12 ABI probe also passed.
The stereo-overlay update passed the six separate-process SDR/scRGB/PQ and DX11/DX12 cases
again on 2026-09-12. The actual official GUI is checked in both eyes, with opaque and translucent
foreground markers, live uniform changes, stable generation across open/close, fresh source
timestamps, close cleanup, reload and focus recovery while open. The independent pixel oracle
checks linear-light blending, SDR re-encoding and HDR highlights/negative values. The test-only
markers do not appear in the shipping add-on. Standalone lifecycle/sharing CTest also passed.

These checks do not establish an installed game's hook order, real foreground transitions or
headset frame pacing. The physical acceptance steps remain in the
[host guide](../../docs/reshade-sbs.md#validation-and-remaining-device-check).

Official references used for this original implementation:

- [ReShade SDK and runtime at v6.8.0](https://github.com/crosire/reshade/tree/v6.8.0).
- [Technique and presentation event definitions](https://github.com/crosire/reshade/blob/v6.8.0/include/reshade_events.hpp).
- [DXGI submission and finish-present ordering](https://github.com/crosire/reshade/blob/v6.8.0/source/dxgi/dxgi_swapchain.cpp).
- [Microsoft x64 calling convention and aggregate returns](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention?view=msvc-170).

The add-on source is GPL-3.0-only, as is the host. ReShade's SDK headers retain their upstream
BSD-3-Clause OR MIT licensing; its adapted GUI shaders retain their BSD notice. The generated
`SunshineSBS-LICENSES.txt` also includes the ImGui license and accompanies binary distributions.
No third-party VR exporter implementation or Depth3D shader is copied into this add-on.

## Runtime installer fixture tests

Run these checks from the repository root in **Windows PowerShell 5.1** with the already
available Pester 3 or 4. Supply an existing copy of the official ReShade 6.8.0 full-add-on
setup; the suite verifies its pinned SHA-256 before using it. The tests use temporary game
directories and mocked downloads, and never execute a game, installer or runtime DLL.

```powershell
Import-Module Pester
$suite = @{
    Path = '.\tools\reshade\test_install_runtime.ps1'
    Parameters = @{ SetupArchive = '.\cmake-build-relwithdebinfo\ReShade_Setup_6.8.0_Addon.exe' }
}
$result = Invoke-Pester -Script $suite -PassThru
if ($result.FailedCount -gt 0 -or $result.TotalCount -eq 0) { throw 'Runtime installer tests failed.' }
```

The suite covers fresh installation, persistent download-cache reuse, corrupt-cache recovery,
archive substitution, existing-file preservation, official compatibility restrictions, anti-cheat
refusal and rollback.
