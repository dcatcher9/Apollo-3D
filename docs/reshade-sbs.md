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
authored eye at the screen plane. ReShade exports do not contain that separate cursor, even when it
appears on the PC monitor. The capture frame carries an immutable cursor shape/position snapshot;
the encoder uploads the small shape only when it changes and composites into its own scratch
texture, leaving the retained game export untouched. Hidden Windows cursors skip this pass;
cursors already rendered by the game remain in the export. Fullscreen display scaling applies to
cursor placement, each eye clips its own cursor, and HDR cursor white follows the display's SDR white level. This cursor metadata
is currently supplied by Desktop Duplication; the WGC fallback does not expose it.

## Native renderer ownership

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

All rendering stays on the runtime's graphics queue. Native passes add no CPU fence wait or full
frame CPU readback. A completion fence protects replacement of the renderer's working set;
in-flight reconfiguration skips output instead of waiting. ReShade drains the GPU before runtime
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
6. In **ReShade → Add-ons → Sunshine 3D**, the **Game 3D** panel keeps **3D strength** and source
   status visible. Strength defaults to **50%**. Each editable image parameter shows **Reset**
   only when its value differs from its default; opening the UI preserves existing saved values.
   Output state, depth source/resolution, conversion, current/target gain and depth freshness
   are always visible. Sharpening is not part of the native renderer or its controls.
   **Depth & troubleshooting** contains **Depth view** (default game image)
   and manual selection. Generic capture heuristics are under **Advanced capture settings**.
   Depth setup is always automatic. There is no Manual shader mode, per-game geometry/weapon
   profile branch or warp selector. Image controls are owned by the add-on and automatically saved
   independently of ReShade shader presets, Performance Mode and the global effects toggle.
   Manual **depth-buffer pinning** remains a separate source-selection override.
   **Depth path** identifies the provider actually selected: **Streamline (game provided)**, **NGX (game provided)** or
   **Generic (fallback/manual override)**. Active generic capture shows a warning that game-provided
   depth and camera data are not being used. Loading a Streamline DLL alone does not establish an
   active Streamline depth provider. AMD depth integration is not implemented yet.
   Current depth-source status stays visible; raw buffers are under **Manual depth selection**.
   Streamline's current depth has its own **ACTIVE** row because its native resource need not appear
   in the generic list. **Last valid** describes history only; unavailable current depth remains 2D.
   Sunshine's host depth-strength control applies to Sunshine-generated stereo and offline conversion.

When the game requests Frame Generation, the panel shows guidance near the **Game 3D** heading.
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

**Keep UI on a separate plane (source alpha)** is off by default and saved per game as `SourceAlphaUI`.
Enable it only when the game's source alpha represents UI: white and gray pin already-composited
UI to one global plane at the independently tracked, smoothed inverse-depth midpoint, while
black keeps scene depth. This is a placement policy, not recovered UI geometry or physical distance.
An entirely white real-input alpha channel makes the entire frame flat at the UI plane; an entirely black
one adds no UI pins. No content heuristic overrides this interpretation. With FG off, alpha comes from the current color frame,
independently of depth-provider selection. With FG on, the D3D12 adapter captures the real input's
Streamline Backbuffer tag 53 at its declared call boundary, keeps a bounded private GPU snapshot,
and reuses the latest completed input alpha across presentations. Generated output alpha is never
used as the UI mask. This path requires that the game expose that input tag. The panel distinguishes
no observed real-input alpha from an observed input whose capture was rejected; neither authorizes
generated output alpha. Protection stays off when no valid completed input mask is available.
Changing the switch does not reload shaders or recalibrate depth.

Only the synchronous live pre-FG alpha snapshot uses
`record_local_texture` with `local_texture_state_policy::prefer_observed_recording`. At the guarded
Streamline FG tag-53 call, a supported, known, unblocked, nonzero state observed on the same
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
Depth copies, optional diagnostic captures and default local texture copies retain their existing
declaration checks. The declaration fallback still needs live game/headset acceptance.

The FG decision is independent of depth-provider selection, including a manually pinned Generic
buffer. The last confirmed mode survives transient observation loss and effects reloads; a new
confirmed mode or observer/runtime shutdown updates it. Renderer, panel and dump share that
presentation decision. An enabled FG mode does not identify an individual generated frame.
UI placement is independent of shape protection: explicitly supplied UI-layer geometry may define
its placement, but the current SL/NGX adapters expose no authoritative target UI depth. This path
therefore uses mode 1, `depth_midpoint`, while the scene retains its contrast-midpoint zero.
The submitted UI inverse depth is the applied, independently smoothed extrema midpoint described
below, not the instantaneous midpoint or the scene zero. The shader maps it through the same
current gain, scene zero, strength, stereo blend and per-eye clamp as scene depth. Every protected
UI pixel shares this plane, but its displayed disparity can move as those controls or the tracked
midpoint change during walking and camera rotation. Unavailable stereo resolves to zero UI
parallax. The plane does not follow the nearest depth beneath UI coverage and does not guarantee
placement in front of that foreground. The source-selection switch is retained because arbitrary game backbuffer
alpha is not universally a UI mask; after selection, constant alpha is honored without heuristics.

Both paths use the same UI protection shader. For FG, only its alpha input changes; all RGB passes
still use the current color, never the retained input's RGB. Captures share the existing depth
transport's producer/consumer fence retirement and maximum source age. No new queue, GPU wait or
CPU pixel readback is added. Unfinished captures are skipped rather than waiting. Scope, viewport,
resolution or observation-revision changes revoke old masks. Alpha and depth have independently
recorded source identities: this is bounded previous-input reuse, not an exact generated-frame
color/depth/mask pairing. Fast-changing UI can therefore briefly lag behind the current color.
Local FG snapshots reuse their allocations only after all producer and consumer GPU work and
recordings retire. They expose no shared handle; optional diagnostic snapshots shared with the
host remain immutable even after acknowledgement. Both use the same bounded capture owner.
The renderer copies each admitted capture into its private UI texture once, then reuses those
bytes for later presentations while that capture remains valid. Renderer replacement clears this
copy identity; failed replacement copies never authorize stale pixels.
Live midpoint placement dispatches no nearest-covered-depth reduction. Historical mode 2
remains available for replay: a 16-by-16 tile pass and a 256-thread reduction resolve
`max(submitted midpoint floor, nearest valid decoded q under finite positive selected alpha)`
into one R32 float on the rendering queue. Its crop/jitter mapping matches the scene candidate;
every covered valid pixel contributes, and empty coverage retains the submitted floor. These
resources are overwritten on each mode-2 render, with no CPU readback or additional queue wait.
After the existing horizontal conditioning, each row computes the exact distance `d` in pixels
to positive finite source alpha and clips its signed displacement to
`pUI +/- 0.5 * max(d - 1, 0) / source_width`. Live mode 1 supplies the displacement derived from
the applied UI midpoint and current geometry. The one-pixel
horizontal collar protects the bilinear color footprint. This rigidly shifts UI in each eye and
preserves the horizontal invertibility bound without overlaying a second copy of already-composited
text. Empty rows retain their original field exactly. The distance scan itself reuses the horizontal
pass and its shared memory. It does not retain the
vertical shear bound across UI-row boundaries. Half-transparent UI retains its original color but
locally flattens the background beneath it; exact independent stereo background requires a separate
HUDless image and UI color/alpha layer. Source alpha interpretation is not inferred from NGX or SL
provider identity. Dump/replay records the effective switch as `replay.source_alpha_ui`, the saved
request as `source_alpha_ui_requested`, and the reason as `source_alpha_ui_status`. Offline replay
uses the effective value and the separate 16-byte `b1` UI constants; the 80-byte geometry `b0` ABI
is unchanged. `source_alpha_ui_fg_mode` records the retained mode and its observation provenance.
The UI constant buffer stores four 32-bit words: enabled, mode, inverse-depth float bits, and
zero reserved padding. Live mode 1 records the applied UI midpoint in
`sunshine_game3d.ui_parameters.v2`; the shader must consume `Sunshine_UIPlaneMode` and the supplied
inverse depth. Historical mode 0 retains the screen plane and the same v2 ABI. Historical mode 2
uses the submitted depth as a floor for the GPU maximum and retains the v3 ABI; its shader requires
`#define SUNSHINE_UI_NEAREST_PLANE 1` and the reduction entry points. Historical mode 3,
`front_limit`, retains v4 and requires `#define SUNSHINE_UI_FRONT_LIMIT_PLANE 1`. It places UI at
the current positive `b0` display budget times strength and stereo blend, subject to warp
readiness; its inverse-depth word is unused, including malformed bits preserved for replay.
Mode 2 is rejected in v2 packages, and mode 3 is rejected in v2/v3 packages. Unknown modes,
invalid depth in modes 1/2, and unavailable stereo resolve to zero UI parallax.
Dump 3D records the exact submitted UI constants, separate from diagnostic targets. Old packages
without these constants retain mode 0. Replay's `--ui-inverse-depth` selects mode 1;
`--ui-nearest-floor` selects mode 2; `--ui-plane screen` selects mode 0; and
`--ui-plane front-limit` selects mode 3. Depth-mode geometry overrides recompute UI displacement
on the GPU. Front-limit placement responds to the applied display budget, strength, blend and
readiness, without deriving a displacement from scene gain or zero. Mode-3 dumps mark the
inverse-depth word unused and the reduction inactive. Historical mode-2 dumps distinguish the
submitted floor from the GPU-resolved plane; replay additionally saves that resolved scalar.
Live rendering never waits to read it back.
`replay.ui_alpha_source` identifies `none`, `source_color`, or `ui_source_color`. When a retained
input was consumed, required artifact 33 (`ui_source_color.bin`) preserves its exact full RGBA
pixels, and `ui_source_alpha.png` shows the alpha used by that render. Its provenance records the
input observation, age and completed capture; it never substitutes the optional latest tag-53
diagnostic snapshot. Live dump transport is version 3 with 40 descriptor slots; offline replay
also accepts earlier packages without an external alpha input.
`replay.source_alpha_capture_attempt` freezes the latest scope- and lifecycle-qualified live
alpha capture attempt observed at that render, including its outcome and available capture-state
evidence. It describes an attempt, not the consumed mask or an exact color/depth/alpha pairing;
the consumed-alpha artifact and its provenance remain authoritative when a mask was used.
Capture diagnostics preserve the raw declared hint and independent observed state.
`copy_state`, `copy_state_known` and `used_observed_state` identify the selected copy/restore state
and whether state selection used observed recording evidence (including the narrow live-alpha policy).
A declared-state selection reports `copy_state_known=true` and `used_observed_state=false`;
"known" here identifies the selected copy state, not independent validation of the provider hint.

The panel controls native `Strength`, `DepthView`, `Enabled` and `SourceAlphaUI` settings in ReShade.ini.
Opening the panel does not rewrite values. Edits apply immediately and are saved automatically,
independently of shader preset Auto Save. Each reset changes only its own parameter.
These controls remain independent of shader enablement, Performance Mode and preset transitions.
Source-owned camera/raw inputs determine depth decoding, coordinate transforms and screen plane;
no `SUNSHINE_GAME3D_AUTOMATIC`, `SUNSHINE_GAME3D_CAMERA_DEPTH` or
`SUNSHINE_GAME3D_GENERIC_CONFIG` mode definitions are needed. The installer removes those
obsolete definitions while preserving unrelated definitions. **Calibration details** displays the
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
Every catalog entry remains visible in metadata when unobserved, null or unavailable. Optional
copies preserve the full allocation's native typed bytes and its own rectangle; they do not apply
the selected depth crop or jitter. Unsupported formats, expired resources, readback failures and
capture-budget limits are reported per resource without discarding the main replay inputs.
`ui_resources` contains middleware observations; `optional_captures` describes frozen optional
copy decisions, and host-side failures appear in `optional_capture_errors`. These scopes must not
be treated as interchangeable frame evidence.
Each optional copy's `capture_diagnostic` records its available declared and observed resource
states, state flags, resource cookie, device and command-recording identity. These are existing
capture-admission observations; recording them does not relax capture checks or change GPU work,
geometry or zero placement. A rejected optional Backbuffer tag-53 copy does not by itself establish
that live alpha capture failed for the same reason. Compare its provenance with the separately
recorded live attempt. Optional tag-53 copies retain declaration checks and can report a declaration
mismatch even while live alpha succeeds using the guarded observed-recording policy. The optional
copy and live snapshot have separate roles; a single dump's optional rejection cannot prove the
live failure reason.
In Expedition 33's earlier 2026-09-19 FG-on diagnostic windows, tags 2 and 54 were nonnull,
and tag 23 was explicitly null. This establishes potential HUDless/no-warp inputs, not an
available UI mask or a match to the exported frame. Before using either input to protect UI,
capture it within its declared resource lifetime and verify frame, extent, color-space and
pixel semantics. Comparing unsynchronized final and HUDless images is not reliable UI separation.

The request uses diagnostic wire v3 with capacity for 40 textures, independent of streaming SBS v1.
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
`--ui-inverse-depth 0.002`, `--ui-nearest-floor 0.002`, or `--ui-plane screen|front-limit` creates a clearly labeled experiment using
the same pass/constant ABI. Gain and zero overrides are independent: moving the zero changes
only `convergence[1] = q0`, preserving the captured gain unless separately overridden.
Live midpoint UI maps the captured applied UI depth through the overridden geometry; its target
is not recomputed by a single-frame replay. Historical front-limit UI instead uses the applied
display budget, strength and blend, subject to warp readiness; gain and zero overrides do not
position it. A legacy captured shader needs a compatible `--shader` override to select that mode.
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

### Streamline depth selection

Supported D3D12 Streamline integrations identify depth at successful DLSS
evaluations or supported enabled-FG tag boundaries. This path defaults on
(`[SUNSHINE_DEPTH] StreamlineDepthSource=1`). It prefers
tag48 (explicit high-resolution depth), then tag0 (scene/render-resolution depth), even when the
heuristic inventory contains a larger texture. A valid high-tag nomination whose capture is
unsupported does not block a capturable tag0 from the same evaluation. If neither tag can be
captured, the preferred valid nomination retains SL ownership without authorizing stereo pixels.
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
No game middleware is loaded or replaced; supported ABI versions remain 1.1.1 and 2.7.30.

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
ownership on the consuming queue independently of pixel readiness. Unsupported capture format or
state or failed display creation therefore produces mono while retaining the API source. A pending
snapshot can use the newest unconsumed completed snapshot from the same source under the ordering,
freshness and interruption rules below; otherwise it also produces mono. Neither case starts
Generic ranking or scene-derived recalibration. Invalid pointers,
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

Streamline 2.7.30 Frame Generation has a separate tag-boundary adapter. Observed successful FG
options enable it; constants and depth tags are associated within their own viewport/frame scope.
The observer intercepts both newly returned and previously cached FG options function pointers.
Resolving the public function or loading middleware does not enable FG; a successful game options
call is still required. Global `OnlyValidNow` tags may pair with the latest successful constants
call for that viewport when a frame-token creation was not observed. This bounded association
uses the constants-call identity, ordering, observation generation and age, never an invented
numeric frame index. Newer constants, invalid/reset camera data or lost observations invalidate it.
Within the validated 2.7.30 layout, the observed all-zero nested Resource header is also supported.
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
Known linear `PrecisionInfo` extensions are decoded, and their scale/bias are reflected in the
texture-space projection coefficients and valid stored-depth range. The original camera matrix
still determines inverse-distance reconstruction; changing only the storage encoding cannot change
the reconstructed center or its adaptive stereo gain.
Unknown or malformed extensions remain unavailable with a
bounded rejection diagnostic. A supported enabled FG source has priority over NGX Super Resolution,
including while its current capture is pending. Its depth and source-associated camera metadata
stay together; missing usable FG pixels do not authorize an NGX substitution or cross-API
projection borrowing. An observed FG Off retires that source so NGX can resume with a newly
submitted frame. Older captures from the
previous ownership interval cannot be resurrected. While FG is enabled for a viewport, its SR
adapter does not also nominate that same viewport; FG's first attempt supersedes only the old
same-viewport SR observation. Other viewports remain independently ambiguous. No cross-API
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

The initial direct provider supports D3D12 full-resource, single-sample
hardware depth and UntilEvaluate/UntilPresent lifetimes; it accepts bounded active rectangles while
copying the full allocation's depth plane. Evaluation-bound native capture declines OnlyValidNow;
the synchronous FG tag boundary supports it. Native capture still declines
linear-depth tags, enhanced/unknown resource states and ambiguous viewports. For V1.1.1, an
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
The current adapters validate versions 1.1.1.0 and 2.7.30.0; other versions
require explicit ABI validation, not a wider unchecked version match. Resource identity remains
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

`[SUNSHINE_DEPTH] NGXDepthSource=1` defaults on for D3D12 games that evaluate DLSS Super Resolution
directly through NGX. Exact exported Create/Evaluate/Release and typed parameter getter functions
are discovered in already loaded SDK modules or the linked game executable. Discovery also runs
at device initialization, before ordinary feature creation. No game DLL is replaced, no unknown
C++ parameter vtable is interpreted, and no feature is inferred merely from installed files.
Successful creation of feature 1 establishes its handle generation, render size and depth direction.
If discovery misses creation, the source remains unknown until the game recreates the feature.
D3D11 NGX interception remains diagnostic-only.

At evaluation entry the adapter reads the exact `Depth` resource and explicit render/depth subrect.
Native capture requires observed nonzero state on the actual command recording and preserves the
resource before the original call; successful evaluation and actual queue submission are separate
requirements. Capture always copies the full depth subresource (plane zero for packed formats),
as D3D12 requires for depth/stencil textures. A validated active rectangle, including offsets and
allocation padding, travels with that capture. The shader and immutable depth sampler use the
same rectangle; no extra GPU crop pass or illegal partial depth/stencil copy is introduced.
An older shader without the rectangle uniform must remain mono for cropped input.

NGX nested within an observed Streamline evaluation is suppressed, as are nested SDK/core calls,
so one evaluation cannot create competing copies. Before ownership is established, Streamline and
NGX observations are kept separate; a malformed attempt from one cannot invalidate the other's
usable frame. Outside the explicitly enabled FG priority above, the first usable provider on
the queue retains ownership through temporary gaps.
An explicit successful release of its feature ends NGX ownership and permits automatic fallback;
silence alone does not. Manual pins continue to override the automatic provider.
Concurrent ambiguous NGX feature/view evaluations remain mono; the add-on does not infer which
unrelated viewport is the final game camera.

Ordinary NGX Super Resolution does not supply a camera projection. Its confirmed depth uses the
shared raw-scene controller with the infinite-far assumption above.
The same signed native-depth input feeds the Sunshine warp. Calibration is keyed
to the logical feature generation and depth convention, rather than physical texture addresses or
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
At most once per five seconds per observed D3D12 Super Resolution feature, it queries the optional
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

Diagnostic callbacks only record bounded in-memory observations. Discovery, module-name resolution and
throttled logging run from the existing serialized effect callback. Process-pinned hooks become
pass-through when disabled. Set the option back to `0` and restart after the diagnostic session.
Full `StreamlineCameraProbe` command/content tracking stays disabled unless separately requested.

### Streamline camera metadata experiment

The add-on includes an optional passive camera-metadata probe for Streamline **1.1.1.0** and **2.7.30.0**,
the installed interfaces in the current Dead Space and Expedition 33 test setups. It observes
only an already-loaded `sl.interposer.dll`; it does not load or replace game middleware. One shared
adapter handles each supported interface, without game-name profiles. Unknown versions remain
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
owning the mutation flag. The synchronous FG camera selection and evaluation capture use one pin,
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

Supported DLSS rendering evaluations (feature 0) publish diagnostic camera/depth evidence. The
separate enabled-FG source adapter can capture depth at a successful tag boundary. Reflex and
other features still execute unchanged. In Streamline 1.1.1, Reflex uses the apparent viewport
argument for a timing marker; treating marker 0 as viewport 0 would overwrite the DLSS evidence.

The observer separately retains HUD-less, scaling-input, scaling-output and supported backbuffer
color tags. Actual effects-input identity comes from ReShade's begin-effects render-target view,
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
for live mode-1 placement, as described below.
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
Live mode 1 consumes the applied UI midpoint directly and dispatches no covered-depth reduction.
Its disparity follows the current native-depth adapter, including gain, scene zero, strength,
stereo blend and the final per-eye bound. Smoothing the midpoint does not make its displayed
disparity constant while the scene or controls change. The scene candidate and vertical fields
are unchanged; source-alpha conditioning moves UI and its horizontal support to one rigid plane.
The midpoint can be behind nearer scene objects. Fullscreen UI may encounter the existing
edge-sampling limits. Depth/alpha registration and bounded FG reuse limitations still apply.
Historical modes 2 and 3 retain their replay behavior but are not selected for live placement.
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
**Farthest (q min)**, **Zero plane q0**, **Stereo gain K**, **UI midpoint q**, and **Normalization L**, with
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
while its gain target is absent. **UI midpoint q** shows the independently applied UI depth and
its extrema-midpoint target. The applied value is consumed by live mode 1 and recorded separately
from its diagnostic target; there is no covered-depth reduction value to read back.
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
The mapping uses Windows' default access control. Sunshine verifies the process creation time
before duplicating its NT texture and fence handles; resources are never looked up by a global
texture name.

The producer publishes immutable metadata with a seqlock. A new consumer nonce requests a fresh
generation of textures and a producer-ready fence. Each slot's atomic 64-bit control word binds
its generation and ownership state, so an old consumer cannot unlock a replacement ring.

For each exported frame, the producer claims a slot, copies the final stereo texture, includes
any overlay composition, signals its GPU fence, writes the frame sequence and QPC timestamp,
then marks the slot ready. Up to three submitted GPU writes may remain unfinished, bounded by
the existing three-slot ring. A recorded copy must receive its submission fence before the next
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
than reusing the abandoned generation. Repeated presentation retains the original frame timestamp.

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
