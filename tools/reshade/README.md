# Sunshine 3D ReShade add-on

One add-on selects scene depth, runs Sunshine's displacement conditioning and inverse warp,
and shares its stereo texture with Sunshine. Game 3D requires no external ReShade FX files.
The game stays at its normal resolution. Sunshine consumes the separate full-resolution
left/right image for its local glasses presenter or Game 3D stream. Follow the
[host setup and handoff contract](../../docs/reshade-sbs.md) for the complete workflow.

The Game 3D panel keeps **3D strength** and depth-source status visible. Strength defaults to **50%**;
**Reset** appears beside each editable image parameter only when it differs from its default.
Existing saved values, including zero strength, are preserved until an explicit edit/reset.
Output state, depth source/resolution, camera/relative conversion, current/target gain and zero values
and depth-freshness statistics are visible without expanding a section. Sharpening is removed.
**Depth & troubleshooting** contains the depth view (default game image) and manual buffer
selection. Gain and zero placement are fully automatic. Capture heuristics are under **Advanced capture settings**.
Depth setup is automatic: source-associated camera data or the explicit raw-depth
assumption controls preparation. Manual shader setup, per-game geometry/weapon profiles and their
legacy processing branch have been removed. Manual **depth-buffer pinning** remains available;
it selects the source and does not select another renderer. Image controls live in the add-on panel
and save automatically in ReShade.ini. The old ray-search renderer, its quality settings and the experimental warp
selector have been removed. Sources larger than 3840 in either dimension show an explicit warning
and remain 2D; there is no hidden legacy renderer fallback.
With Frame Generation, source-alpha UI protection distinguishes missing real-input alpha from
an observed input whose capture was rejected. It requires a valid completed input mask; generated
output alpha is never substituted. Its synchronous live Streamline input snapshot prefers a clean,
supported nonzero state observed on the same command recording. Only a completely absent state
entry permits fallback to an explicit supported nonzero provider declaration with compatible
resource flags and declared-state proof. Blocked, unknown, COMMON/zero, split, lost, invalid or
render-pass state still rejects capture; a source requiring observed-state proof cannot fall back.
The private copy restores the selected state. Other captures retain their existing declaration
checks. UI protection offers per-game, persistent On / Off / Auto choices. Auto starts detection
when alpha sampling can begin, enables selective alpha provisionally, and holds its result after
confirmation or timeout. Game/device loading does not consume the scan. The panel shows the
current detection status and result;
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
are not supported. A top-level game window must be the foreground window; child-window rendering
is not supported by this exporter.

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

For a troubleshooting capture, use the client's **Dump 3D** action while in Game 3D. See the
[Game dump contract](../../docs/reshade-sbs.md#dump-3d-diagnostics) for the lossless artifacts,
SL/NGX observations, missing-data reporting and output location.
Dump wire v3 also carries optional UI-related resources and reports every catalog entry, including
null or unavailable inputs. Native bytes stay unchanged; full-allocation mask and alpha previews
use fixed black=0/white=1, with separate channels and visible nonfinite values. Optional color
transfer is labeled unknown unless captured metadata declares or explicitly assumes it. These
resources are diagnostics and do not establish UI classification or frame matching. Host and
add-on must both support the diagnostic mapping version.
NGX failures include readable result codes. Successful null and failed query remain separate states.
The dump records the latest scoped live-alpha capture attempt observed at the render in
`replay.source_alpha_capture_attempt`, and optional copies carry `capture_diagnostic` state and
recording provenance. The latest attempt is not proof of a consumed-alpha pairing, and an optional
copy's rejection does not prove the same live failure: optional tag-53 copies can still reject a
stale declaration while the guarded live-alpha copy succeeds from observed recording state.
Diagnostics retain both input states and identify the selected copy state with `copy_state`,
`copy_state_known` and `used_observed_state`. A declaration fallback has a known copy state and
`used_observed_state=false`; that is provider provenance, not independent state validation.
The linked dump contract defines these separate
evidence scopes; recording diagnostics does not change capture admission or rendering.

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
Auto performs one observation window each game launch, beginning with the first eligible alpha
probe. It uses an initial frequent scan followed by sparse checks for a later mask.
The panel displays Waiting, Detecting, Off (checking), On (confirming), then the final On or Off.
See the [UI protection contract](../../docs/reshade-sbs.md#setup) for timing and coverage rules.
Edits save independently of ReShade's shader-preset Auto Save option; no Save button is needed.
Reset affects only its parameter.
The opt-in `reshade_game_present_d3d12_test` draws this production panel in the real
ReShade GUI with collapsed and expanded sections, in addition to checking control
edits and persistence. Model/API tests or Home-tab overlay tests alone cannot catch
cross-compiler ABI failures in the panel's layout calls.
With `SUNSHINE_GAME3D_NATIVE_ONLY=1`, setting
`SUNSHINE_GAME3D_ALPHA_DELAYED_STARTUP_TEST=1` also delays device initialization
and checks the production alpha detection window and final result in ReShade.log.
Use a fresh output directory for this regression.
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
When this fallback is active, the panel suggests trying **Frame Generation 2×**, if supported:
some games supply usable Streamline camera data through FG. This is not guaranteed. If FG is
already observed enabled, the panel reports the missing camera data without asking to enable it
again; the existing 3×/higher and artifact guidance still applies. Initial unknown depth status
does not show this hint, and Sunshine does not change the game's FG setting.

The current production policy trials a contrast midpoint for the scene's zero plane, using the
farthest depth and moments of each pixel's contrast from it. The independent gain still follows
the nearest depth. Protected UI uses mode 1, `depth_midpoint`, with the independently tracked,
smoothed inverse-depth midpoint. The shader maps this applied UI depth through the current gain,
scene zero, strength, stereo blend and per-eye clamp, subject to warp readiness. All protected UI
shares one plane, but its disparity can vary as the scene and controls change. It is not forced
ahead of the nearest depth beneath UI coverage. Gain and zero retain their existing
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

Live midpoint UI dispatches no nearest-covered-depth reduction and records its applied depth in
the exact 16-byte `sunshine_game3d.ui_parameters.v2` contract. Replay's `--ui-inverse-depth` selects
mode 1 with an explicit depth. Historical modes 0 (screen/v2), 2 (nearest covered depth with a
submitted floor/v3) and 3 (maximum front display limit/v4) retain their replay behavior. Mode 2
alone dispatches the tile/reduction passes. Mode 3 requires
`#define SUNSHINE_UI_FRONT_LIMIT_PLANE 1` and ignores its inverse-depth word; it is available via
`--ui-plane front-limit` with a compatible shader. Neither mode 2 nor mode 3 is selected live.
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
**Zero plane q0**, **Stereo gain K**, **UI midpoint q**, and **Normalization L**, with Current/Target columns.
Q and the farthest value are the same accepted measurement's full active-depth extrema, not camera
clipping planes. These depth values share one inverse-depth coordinate: larger is nearer.
Gain shows the applied value and nearest-depth-derived target; q0 shows the applied zero and
contrast-midpoint target. **UI midpoint q** shows the independently applied UI depth and its
extrema-midpoint target; live rendering consumes the applied value, not the latest target.
The [UI protection contract](../../docs/reshade-sbs.md#setup) owns its displacement, timing,
occlusion limitations and replay constants. UI uses the same current geometry mapping and
per-eye bound as the scene; it is not a fixed display disparity or recovered physical distance.
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
default for supported D3D12 games. Automatic buffer selection prefers the game's explicitly tagged
high-resolution depth, otherwise its tagged DLSS render-resolution depth, ahead of unrelated
larger heuristic candidates. Manual pins take precedence. SL/NGX select the resource; Generic
selection is the fallback when no valid API source owns selection. One D3D12 capture owner handles
both Generic preservation boundaries and API evaluation opportunities, including snapshot storage,
submission, consumer leases and retirement. API capture snapshots the nominated resource at the
middleware call, including tracked depth-stencil resources; matching a Generic backup's resource
identity cannot authenticate its contents. While a same-source successor is pending, acquisition
may advance to an unconsumed completed snapshot with its own sequence and metadata under the
contract's ordering, freshness and interruption checks. A valid API source with no admitted pixels
stays selected and returns mono. Automatic reconstructs inverse
depth from the exact current projection, without smoothing its coefficients. The independent gain
and screen plane follow the measured-range policy above. Current depth and
zero-plane values must fit the shader domain. Unsupported encodings stay mono without clamping depth. Missing matrices use
the shared raw-depth screen-plane controller with the explicit infinite-far assumption.
The linked contract describes the reference convention and
the limits of comparing unknown raw encodings with projection depth.
Successful evaluation and actual submission establish API source ownership independently of
snapshot readiness. The linked contract owns the capture-ordering requirements and limitations;
D3D11 retains its existing Generic preservation backend. AMD support is deferred. Set
`[SUNSHINE_DEPTH] StreamlineDepthSource=0` in `ReShade.ini` and restart to disable this path.

[Direct NGX depth](../../docs/reshade-sbs.md#direct-ngx-depth-selection) is also enabled by default
(`NGXDepthSource=1`) for supported D3D12 DLSS Super Resolution integrations. It shares native
capture, retirement, display binding and rendering with Streamline. Without a supplied projection,
it uses the shared raw-depth screen plane, preserving its history across rotating textures and
dynamic resolution changes. NGX and Generic fallback use the same explicit infinite-far assumption.
The panel identifies the actual NGX/Streamline/Generic path. Read the
linked contract for supported crops, feature lifetime and early-discovery limitations.
When a confirmed newer NGX capture is waiting only for GPU completion, the owned completed depth
can bridge one native presentation. This holds its exact previous scene controls and never advances
calibration. Unknown, failed, changed or stale input cannot authorize this hold; a second missing
presentation returns to mono. No completion wait or extra copy is added.

When a supported Streamline Frame Generation source is enabled, its depth and associated camera
data take priority over NGX, including while its current snapshot is pending. A missing usable FG
copy stays with that source and uses the documented bounded real-depth reuse or mono behavior;
it does not borrow an NGX depth texture and attach unrelated SL constants. Explicit FG Off retires
that source and lets a fresh NGX submission resume. Motion-vector interpolation of two real depth
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

## Additional diagnostics

The separate [Streamline camera probe](../../docs/reshade-sbs.md#streamline-camera-metadata-experiment)
is disabled by default. `StreamlineCameraProbe=1` enables detailed diagnostics after restart;
leave it off for ordinary play. Production source selection does not enable its per-draw content
tracking. The passive diagnostic does not control rendering; production Streamline capture and
scale are described above. Generic fallback retains raw-scene calibration. The shared
ABI declarations are pinned to Streamline 1.1.1 and 2.7.30; unsupported versions are left alone.

For call-route investigation only, `[SUNSHINE_DEPTH] UpscalerCallTrace=1` enables the
[separate runtime trace](../../docs/reshade-sbs.md#upscaler-call-route-diagnostic) after restart.
It reports actual Streamline/NGX calls and their callers without enabling the full camera probe
or changing depth selection. A loaded DLL or zero observed calls does not prove the rendering
route; inspect the positive call evidence and reported hook coverage. Disable it after testing.

Opening ReShade's overlay keeps the game in stereo, with its real controls composed at the same
position in both eyes. Normal mouse coordinates are retained and add-on control changes appear
live. Closing the overlay removes the controls without replacing the stereo resource generation.
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
of an unconsumed completed NGX snapshot while its successor is CPU-recorded or GPU-pending, plus
rejection across failed/missing input, observation/reset revisions, a reset flag, epoch and layout
changes. `--foreign-reclaim` retires producer and consumer recordings and releases private
snapshot leases before allocating an unrelated NGX capture or a Generic preservation copy from
a separate resource. Its ten cases check that SL FG nomination authority, current depth and
completed depth beneath a pending successor survive reclamation, while failed/missing input
still revokes old depth and prevents pre-gap resurrection. Pixel cases use actual GPU readback.
`--content [width height]` nominates a full patterned packed D32S8 scene through NGX
while the Generic preservation callback reports availability, then clears that same source after
the API capture. It checks the snapshot's scene depth byte-for-byte and verifies that depth-plane
copies preserve both the original and consumer stencil. This is one post-capture-clear scenario;
it does not cover a separate missing-callback or no-clear case.
Any fixture completion waits serve its deterministic pixel oracle and cleanup; they are not
production capture behavior. These owner-level tests do not establish SDK-hook coverage or
color/depth correspondence in a real game.

Use `reshade_game3d_native_provider_runtime_test` for the add-on's current native lifecycle with
zero installed FX techniques:

```text
reshade_game3d_native_provider_runtime_test.exe <ReShade64.dll> <frozenShaders> <SunshineSBSTest.addon64> <fresh-output-directory> 3840 2160 <frame_generation_interposer.dll>
```

It exercises real GPU depth production, capture, sampling and matrix preparation, including
padded/nonzero-offset rectangles, failed evaluation and recovery. The optional metadata-only
interposer enables the SL FG 2x case: retained real depth, generated presentations, FG-off
rejection and fresh SL recovery. It also checks repeated holds against the original capture's
expiry, camera reset, and FG off/on without a fresh capture; none may revive invalidated depth.
Fresh copies must recover each interval. It injects no captured pixels, readiness or GPU completion.
Run these functional cases serially when they share resources or output directories; their timing
is not performance evidence. The effect-based fixtures documented below remain in the repository,
but their shader-uniform and shared-preservation expectations are historical and do not replace
these native capture gates.

The optional `reshade_streamline_direct_runtime_test` is a component integration fixture.
Its metadata seam bypasses SDK discovery; the real SDK-entry NGX/Streamline cases below remain
necessary to validate that entry path. Its historical rendering assertions use the former
current-zero ratio and independently derived inverse-depth values; those assertions are not
acceptance evidence for the current gain policy.

This fixture exercises independent capture and
projection-based rendering. It writes real R32_FLOAT UAV textures that never enter the generic
depth-stencil inventory, preserves them through native command submission, and checks their
spatial depth pattern against the shader input. It also checks rotating resources, manual
pin/release, the former coupled screen-plane/normalization changes, explicit recentering, proportional strength,
failed evaluation recovery, and zero-strength HDR color. It uses the same arguments/environment
below, with `reshade_streamline_direct_runtime_test.exe` and a fresh output directory. Its test-only
seam supplies middleware metadata, never shader readiness, copied pixels or GPU completion.
`SUNSHINE_STREAMLINE_MATRIX_CHANGE_TEST=1` checks changed near/far projection metadata:
the shader must use current reconstruction coefficients. Its old zero-tracking assertions belong
to the coupled ratio policy. It also checks rotation, gaps, FOV invariance
and actual HDR pixels.
Like the modes below, it is exclusive. A source nomination ticket is not a pixel-readiness
assertion: negative capture cases verify held API ownership and current-color mono separately.
Set `SUNSHINE_STREAMLINE_COLD_V1_TEST=1` for V1-only rotation without successful V2 warmup.
Alternatively, `SUNSHINE_STREAMLINE_PRIVATE_STATE_TEST=1` checks the pinned V1 provider's
resource-state metadata on a fresh evaluation recording, including COMMON and missing,
malformed, conflicting or split-transition rejection. These modes are mutually exclusive and
use the same real GPU capture and shader path.
`SUNSHINE_STREAMLINE_PACKED_V1_TEST=1` is a third exclusive mode for the packed
`R32G8X24_TYPELESS` depth/stencil format observed in Dead Space. It checks rotating sources,
prior-recording provider state, plane-aware depth readback, unchanged source stencil, actual
shader preparation, mono fallback and recovery.
`SUNSHINE_STREAMLINE_COMMAND_CAPACITY_TEST=1` is another exclusive mode. It retains
160 real native-only command lists on the observed vtable, then destroys and recreates them
while checking packed-depth rotation, shader output and recovery. These lists bypass ReShade's
device wrapper, so their cleanup must follow native COM lifetime rather than ReShade events.
`SUNSHINE_STREAMLINE_PENDING_EVALUATION_TEST=1` is another exclusive mode. After proving
ordinary packed-depth rendering, it submits the real producer recording before completing the
evaluation. Successful completion must still reach the shader; failed completion and replaying
the unchanged producer must remain mono. It checks both rotating sources, preserved source
stencil and actual prepared-depth pixels, then verifies recovery with a new recording.
`SUNSHINE_STREAMLINE_LARGE_BATCH_TEST=1` is another exclusive mode. It submits 97 real native
command lists together with the depth producer at the start, middle and end; a separate idle-only
96-list batch must not invalidate captured depth. A 257-barrier call puts the source transition
last. Each case checks current shader pixels and source stencil, then failure/recovery. Set
`SUNSHINE_STREAMLINE_LARGE_BATCH_CASE=barriers` to isolate the barrier case; omit it for all cases.
`SUNSHINE_STREAMLINE_REQUIRE_PROVIDER_STATUS=1` additionally requires the test-only read-only UI
snapshot and checks current, missing/last-valid and manual-override source reporting. Without that
flag an older control DLL lacking the getter can still run the GPU capture comparison.

`reshade_ngx_depth_runtime_test` is the historical effect-based fixture for named NGX entry points,
depth controls and HDR shader output. It does not observe the current no-FX native lifecycle
reliably, including when supplied frozen FX sources. Its archived invocation uses `SUNSHINE_GAME3D_AUTOMATIC=1`,
`SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST=1`, and `SUNSHINE_DEPTH3D_EFFECT=SunshineGame3D`:

```text
reshade_ngx_depth_runtime_test.exe <ReShade64.dll> <Depth3D/Shaders> <SunshineSBSTest.addon64> <fresh-output-directory> 3840 2160
```

Set `SUNSHINE_NGX_CROSS_QUEUE_TEST=1` and `SUNSHINE_NGX_CROSS_QUEUE_COMPLETED_TEST=1`
for the deterministic separate-queue regression. The fixture executes depth production on
another real queue, retires its submitted recording, and waits for its actual completion before presenting. This test-only wait
is not add-on behavior. The fixture checks changing depth and HDR pixels, unfinished-producer
mono without blocking, missing-frame recovery, and producer queue recreation. The frozen old
add-on must fail the named `NGX-completed-separate-producer` admission check. Omit the completed
flag for asynchronous coverage, where current depth is verified when admitted and pending work
may remain mono. These are functional checks, not performance evidence; an idle GPU is not required.
Run cases serially with separate output directories. `SUNSHINE_NGX_STATE_PRESSURE_TEST=1`
is a separate case and must not be combined with the cross-queue flags.

The legacy `SUNSHINE_NGX_TRACKED_SOURCE_TEST=1` case assumes API nominations reuse Generic
preservation and that DSV/UAV transport changes toggle Generic capture demand. Those assumptions
conflict with the current API-boundary snapshot contract, so this case is not a current regression
gate. It remains available for historical binaries; its mode-2 variant uses
`SUNSHINE_DEPTH_BIND_SWITCH_TEST=1`. Use the current `--content` owner test and native provider
runtime above for this change, not a passing legacy effect-based result.

The retired generic-nomination adapter and its metadata-injection fixture are removed. Current
validation combines the native owner/provider gates above with the actual-hook CPU tests.
The generic selector retains its separate policy and runtime tests. No synthetic fixture alone
establishes a particular game's hook order or final depth/color jitter alignment.

For SL tag priority and capture-failure recovery, run `reshade_ngx_depth_runtime_test` with
`SUNSHINE_NGX_FRAME_GENERATION_TEST=1` and `SUNSHINE_SL_TAG_FALLBACK_TEST=1`. The real SL
entry points nominate both tag48 and tag0: an unsupported high-tag texture must not block the
usable ordinary depth, a usable high tag must retain priority, and two unsupported captures must
retain SL ownership while publishing current-color mono. Recovery checks the actual shader
binding and copied depth pixels. Add `SUNSHINE_FG_LIVE_COMPAT_TEST=1` for global tags with
zero-initialized resource wrappers; omit it for typed resources and explicit frame tokens.
Use a fresh output directory and keep these functional GPU runs serial.

Native Game 3D records its GPU passes and export on the runtime's immediate queue. It borrows
the selected depth only within the capture owner's open pass and supplies frame-associated
camera/jitter values directly. Its renderer owns the output texture and its color contract;
no FX reflection or export annotation is required for this path. Explicit `SuperDepth3D.fx` and
independent `SunshineDepth3D.fx` remain reference sources: their annotated texture is accepted
only after the owning technique executes, with native Game 3D disabled. Only the independent
reference requires its separate current-frame percentile calibration update.
The exporter verifies the normal runtime resolution, actual texture dimensions/format and
current game swapchain color space. With the overlay closed, it copies the texture without
resampling or color conversion. For HDR10/PQ input, the renderer decodes PQ and transforms Rec.2020 primaries to
linear Rec.709; native scRGB input keeps that representation. scRGB uses 1.0 = 80 cd/m².

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

The Direct3D 11 path uses `SHARED | SHARED_NTHANDLE` textures and native shared fences. It copies
on ReShade's immediate context, finishes any overlay composition, then signals the fence and
flushes submission without waiting for it.
The Direct3D 12 path appends copy barriers to ReShade's immediate command list and restores both
resource states. It signals its native queue fence only in the matching `finish_present` callback,
after ReShade submits that list. It never forces a ReShade command-list flush, which could wait for
an allocator and reset the game's draw state. A different swapchain or queue cannot publish the copy.

While settings are open, a public ImGui draw callback adds an FP16 target to ReShade's native
overlay pass. ReShade still draws its actual controls, textures, tooltips and cursor and handles
input. A compatible GUI pipeline retains the game's native target and captures linear color
with full alpha in the additional target. A final GPU pass composes that layer into both eyes,
blending in linear light and encoding SDR back to sRGB. HDR remains linear scRGB, including
negative values and highlights above SDR white. D3D11 composition restores the application
state it touches; D3D12 uses ReShade's own immediate command list. Composition shares the
existing slot and fence lifetime; there is no additional queue wait or CPU image readback.

Only one producer copy may be outstanding. Busy slots or an incomplete fence drop the next export.
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
The historical `reshade_adaptive_raw_runtime_test` actual-ReShade fixture uses real game DSV draws,
the integrated selector/sampler, and published scale/zero-plane/readiness uniforms. Its scenarios
check the former reciprocal normalization of the smoothed zero through scene changes, pin/unpin
continuity, source initialization, gaps, stereo and mono
in both HDR depth conventions. No camera/gain uniforms or synthetic readback results are injected.
Use a separate control add-on and fresh output directories for a matched run. When comparing
the old ratio and fixed-reference implementations, its normalization assertion distinguishes
those historical policies. It does not establish acceptance of the current range-based policy.

Build `reshade_adaptive_raw_runtime_test` with runtime tests enabled, then run it
with `SUNSHINE_GAME3D_AUTOMATIC=1`, `SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST=1` and
`SUNSHINE_DEPTH3D_EFFECT=SunshineGame3D` in the environment:

```text
reshade_adaptive_raw_runtime_test.exe <ReShade64.dll> <Depth3D/Shaders> <SunshineSBSTest.addon64> <fresh-output-directory> <scrgb|pq> <normal|reversed> <width> <height>
```

Set `SUNSHINE_DEPTH_RELOAD_TEST=1` to also reload the real effect while depth is absent,
rediscover its new resources, and require fresh readiness, calibration and HDR stereo after
depth returns. `reshade_depth_ready_uniform_cache_tests` covers reflection reuse and change-only
publication by the shared Generic/API readiness owner. The legacy NGX runtime fixture's
`SUNSHINE_CAPTURE_DEMAND_TEST=1` and `SUNSHINE_NGX_TRACKED_SOURCE_TEST=1` combination expects
shared-preservation bootstrap and DSV/UAV demand handovers. Those are historical expectations,
not current acceptance criteria: API snapshots keep Generic capture dormant for both resource
types, while manual pins still enable Generic capture.

The fixture requires the separate test add-on for the real pin/unpin/**Recenter**
actions. It checks native depth pixels, shader preparation and exported HDR pixels
as well as the control trajectory. The production package excludes these test exports.

`reshade_depth_bind_switch_runtime_test` exercises preservation mode 2 with real
D3D12 D32S8 draws. Use the three environment settings above plus
`SUNSHINE_DEPTH_BIND_SWITCH_TEST=1`; its arguments are only the runtime DLL,
shader directory, test add-on and fresh output directory. It runs 4K scRGB with
2228×1256 scene depth, first unbinding each buffer to null, then switching directly
from scene depth to another depth target. Changing depth patterns reject stale
copies. Same-binding, clear-only, missing-depth and transition-away cases verify
capture admission; fresh recalibration and rendered stereo verify recovery. Run
the identical fixture against the old and candidate add-ons: the old direct-switch
path must fail after passing the null-unbind control. Like the other native GPU
fixtures, execution is opt-in and serial, outside CTest.
Add `SUNSHINE_DEPTH_GENERIC_ONLY_TEST=1` to disable both API sources, camera diagnostics and
call tracing before initialization. The fixture verifies those settings and runs the same
preservation and HDR checks, proving that shared capture does not depend on API discovery.

`reshade_depth_probe_round_runtime_test` uses the same four arguments and
environment settings to check bounded challenger sampling in Automatic mode.
A real 4K source is created before six active flat peers; it must replace the
calibrated lower-resolution source within three seconds, with its actual depth
pixels verified. The unchanged old add-on must fail that promotion bound. The
fixture also checks that an uncapturable candidate yields and an active manual
pin remains authoritative. These are controlled regression bounds, not promised
recovery times for every game. No selection, depth or camera values are injected.

`reshade_depth_alternating_runtime_test` uses those same four arguments and
environment settings for an actual 4K rotating-allocation regression. Each allocation
first passes a continuous capture/calibration control. ABC and AABB cadences must
then reach continuous current capture, readiness and full stereo within a bounded
warmup; each allocation keeps its own deliberately different H/t0. Every checked
present verifies the physical depth source, actual copied depth, a changing color
marker in both exported eyes, and current mono when depth is unavailable. Additional
phases cover off-turn transfer writes, exact manual pin/release, lifetime replacement,
unsupported partial crops, full-layout return, real depth gaps and a moving-center
trajectory. Run the identical executable against the frozen old and candidate add-ons;
the old single-source implementation must fail the rotating-readiness assertion after
passing its continuous controls. Compact GPU readbacks are correctness evidence,
not an uncontended frame-time benchmark. Execution is opt-in and serial, outside CTest.

The same fixture accepts one optional isolated case after its four normal arguments:
`--overlap`, `--interrupted-startup`, `--moving-startup`, `--rotating-startup`,
`--flat-startup`, or `--layout`. Each case needs a
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

The older `reshade_raw_scene_runtime_test` scenarios below remain fixed-gain baseline and
component-research fixtures. Build that target explicitly when testing a historical control;
it is excluded from the default build. Their startup, source and cached A → B → A assertions
describe a previous controller; a frozen H alone does not make them the current reference-policy
regression gate. Shared native-depth, uniform and HDR mono observations live in
`test_raw_runtime_fixture.h`; the current fixture
does not include or rename the old program's entrypoint. The shared shader/transport and
independent physical-camera fixtures retain their own contracts.

For full-pipeline rendering checks, use the D3D11/D3D12 fixtures described under
[Full-pipeline shader tests](#full-pipeline-shader-tests). Set
`SUNSHINE_DEPTH3D_EFFECT=SunshineGame3D` to select the renamed effect; unset it or use
`SuperDepth3D` for the frozen original. Each loads the entire separately supplied pipeline with
its own supported controls. A comparison must match the remaining shared controls explicitly.
These rendering checks complement the current controller fixtures and
installer/receiver workflow; they do not establish physical game/headset acceptance.

For historical fixed-gain comparisons, the `reshade_raw_scene_runtime_test` fixture can record
actual Automatic source color, game DSV depth, prepared depth and full-SBS outputs with AA off/on.
Set `SUNSHINE_GAME3D_AUTOMATIC=1`, `SUNSHINE_DEPTH3D_EFFECT=SunshineGame3D` and
`SUNSHINE_GAME3D_AUTOMATIC_EVIDENCE=1`; use the same fixture, shader source, dimensions, color and
orientation with separate control/treatment add-on binaries and fresh output directories.
Require identical `automatic-evidence` source color, game depth, final mono/stereo pixels and
geometry metadata. The original prepared RG16F surface also stores temporal payloads in its
corner G cells: report their differences separately and require every convergence R value and
all remaining scene-depth G values to match. Retain the complete intermediates and never mask
final stereo pixels. The fixture lets
continuous rendering recover after large readbacks before capturing full-strength stereo.
These snapshots exercise the actual selector, sampler, policy and shader without injected camera
values; they are regression evidence for unchanged geometry, not a quality comparison between
Manual and Automatic or a substitute for moving-game/headset acceptance.

For the test-only action run, `SUNSHINE_GAME3D_LEGACY_CALIBRATION_TEST=1` adds
read-only checks that a fresh Game3D session requests no legacy calibration and
has no legacy cache entries, both after startup and after the existing source-change
and Recalibrate checks. It requires `SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST=1`
and the separate `SunshineSBSTest.addon64`; it rejects the distribution-test combination.
The state-query export is not present in the production add-on and does not change
the renderer's request. Use the same fixture/query on both control and treatment.

`SUNSHINE_GAME3D_AUTOMATIC_GPU_COST=1` selects an isolated actual-runtime timing
scenario instead. It uses the real selector and policy on two native-depth
scenes, both AA settings, and stereo/mono/Normal Depth View. Each case waits for
the real zero plane and reentry, then brackets 64 complete effect executions
after 16 warmup frames. `gpu-cost/samples.csv` retains GPU ticks, frequency and
actual H/t0/blend; complete source/depth/SBS files accompany the summary. Large
readbacks and disk writes are outside the timestamp interval. This measures
effects-begin through the selected technique, including COLOR capture, not game
FPS or end-to-end streaming. Use fresh directories and no other scenario flags.
The separate ReShade Performance Mode flag stays off so controls remain editable.

`SUNSHINE_GAME3D_AUTOMATIC_STARTUP_TEST=1` instead exercises unsuccessful startup recovery through
that production runtime fixture. After the first real calibration sample, a scissored game draw
makes the center unsuitable for more than five seconds while preserving the surrounding scene.
The fixture verifies the actual selected depth, current mono, a fresh stabilization interval and
automatic recovery when useful depth returns. Add `SUNSHINE_GAME3D_AUTOMATIC_STARTUP_REPLACE=1`
to replace the source before calibration commits. Use fresh directories and run these separately
from snapshot, Performance Mode and test-only action scenarios. No shader/camera/depth values are
injected into the integration. An already established reference cannot be reused by a different
source. An exact return to the original source can recover as described in the owning contract.

`SUNSHINE_GAME3D_AUTOMATIC_RETURN_TEST=1` exercises a transient selected A → B → A transition
after calibration. The original DSV allocation stays alive; B has a different allocation/lifetime
and is held for only 150 ms after selection, before persistent-replacement qualification can
finish. B must remain mono. Returning A must recover with a fresh capture and the same H,
followed by full-strength stereo pixels, without a Recalibrate action. Run with
`SUNSHINE_GAME3D_AUTOMATIC=1` in a fresh directory, separately from startup, replacement,
Performance Mode and test-only action scenarios.

`SUNSHINE_GAME3D_AUTOMATIC_REPLACE_TEST=1` instead leaves B selected with a materially different
depth distribution. For the frozen control add-on, leave
`SUNSHINE_GAME3D_AUTOMATIC_EXPECT_RECOVERY` unset: the fixture requires persistent mono and the
unchanged old H after 3.5 seconds. For the treatment, also set
`SUNSHINE_GAME3D_AUTOMATIC_EXPECT_RECOVERY=1`: the fixture requires automatic recovery within its
12-second bound, a fresh H/reference matching the independently read native replacement DSV,
and full-strength stereo pixels. It writes selected depth, recovered SBS and reference/timing
metadata under `source-recovery-evidence`. Both runs must use identical shader source, fixture,
dimensions, color and orientation with separate add-on binaries and fresh directories. Run
separately from startup, transient return, snapshots, distribution tests, Performance Mode and
test-only actions; neither depth nor camera/calibration uniforms are injected.

Those historical source-recovery scenarios used two successive qualification windows and kept
H fixed within a source. The current controller uses one fresh initialization window for
independent gain and zero, and resumes A's history when A → B → A stays within the retained
capture set. Eviction or a change to A's exact source basis requires fresh initialization. Its
[source and timing contract](../../docs/reshade-sbs.md#experimental-raw-depth-automation) owns
the current behavior; the older frozen experiments do not establish current-policy acceptance.

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
Auto-select disabled and a buffer chosen by draw statistics, without a manual override.
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
focus loss/recovery and receiver restart. It also poisons every calibration input and requires
the publisher to replace the complete set on the next published frame. This complements the
production shader and automatic-selection fixtures; controlled paint does not test depth warping.

Look in the game's `ReShade.log` for `Sunshine SBS:` messages:

- `exporter ready` means the mapping exists and the add-on is waiting for the technique/consumer.
- `no compatible export annotations` means the current shader definition or color annotations are absent.
- `generation ..., ... full SBS, DXGI ..., D3D...` means resources were created for a consumer.
- `export inactive` means focus, technique execution or source proof stopped qualifying.
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
private GPU copies; the fixture does not modify the receiver's slot state. Pixels, declared
transfer, retained timestamps, effect/overlay/reload/focus invalidation, recovery and receiver
restart are checked. Producer and receiver run in one process on independent native devices;
these tests do not establish access across a live game's process or host-service boundary.

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
texture/fence import and private GPU copy. The test then reads back that private copy and sends
its pixels to the parent for inspection; this CPU path exists only in the fixture. Production
texture transport remains entirely on the GPU. Receiver restart starts a new helper process.
The logs record both PIDs and helper exit status. These checks cover separate processes under
the same user/session/integrity level, not a live installed game or the elevated host-service
boundary. All six D3D11/D3D12 SDR/scRGB/PQ cases passed locally, including receiver restart in a
new process and clean shutdown of all twelve helper instances.

`SUNSHINE_SBS_RUNTIME_TESTS` defaults to **OFF**. Its `SunshineSBSTest.addon64` is for these isolated
fixtures only. The shipping `SunshineSBS.addon64` retains the real Windows foreground check and
does not export the test observation setter. Do not install the test add-on into a game.

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
