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

Remote Game 3D converts each game export when its ready fence completes, within the stream's
presentation cadence; between exports it encodes only cursor changes, recovery frames and the
minimum-FPS keepalive, never a repeat at stream cadence ([GPU handoff contract](#gpu-handoff-contract)). Valid packed
exports use the receiver's GPU synchronization; they do not open the unrelated desktop texture or
acquire its keyed mutex, and while one is converted desktop capture forwards only timestamps and
cursor metadata. Ordinary desktop output and the duplicate-eye fallback retain capture
synchronization.

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
not expose it. Capture restarts explained by display-mode changes do not trigger that fallback;
see [Display mode changes](virtual-desktop.md#display-mode-changes). Install the matching
protocol-2 host and add-on together for cursor/UI alignment.

## Rules for new game behaviour

Each new game has so far broken an assumption that held in the games before it. A rule added for
one game is held to these four tests before it ships, so that the next game starts in a safe state
instead of a wrong one:

1. **Gate on the API contract, not on observed behaviour.** Admission follows what D3D12,
   Streamline or NGX require; a rule stricter than the contract fails on the next engine.
   Treating every aliasing barrier as invalidating the recording blocked all of Resident Evil
   Requiem's DLSS depth, although an aliasing barrier changes no resource state.
2. **Treat optional SDK features as optional.** Frame tokens, HUD-less images and UI buffers may be
   absent; every path has a fallback that does not need them. Resident Evil Requiem's camera has no
   frame identity, so depth takes the viewport's steady camera constants instead.
3. **Fail safe, not sideways.** When the best evidence is missing this frame, do nothing (or hold
   the previous result within its bound) rather than take the next candidate; a fallback must not
   be able to make the result wrong in a new way. A presented alpha deciding when a trusted UI
   channel failed flattened most of a scene.
4. **Anything learned or remembered can be unlearned in every game.** Trust, calibration and
   remembered state need a contradiction that every game can produce, not only games with a
   particular buffer.

The hidden-scene check of UI protection (**Hidden-scene evidence** under Setup) follows rules 2
to 4: its evidence needs only the presented colour, the consumed depth and, when the game offers
one, a pre-UI scene image (a HUD-less image or a cleared target holding the scene); an informative
full claim, not a source-specific route, gates it; without fresh evidence its bounded hold lapses
and nothing changes; and a visible verdict, which every game can produce, refutes the claim of the
source signature that made it. It never selects depth, changes calibration or earns trust.

The [first-run report](../tools/reshade/README.md#first-run-of-a-new-game) checks a new game's
first session for the failure signatures these rules came from.

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
| UI with FG off or no observed FG | Automatically validate captured SL UIAlpha (R; tag 68 in 2.11.x, 69 from 2.12.0, see [buffer contract](#streamline-buffer-type-contract)), UIColorAndAlpha (23, A), the game's offscreen UI layer (A), Backbuffer (53, A), current color alpha, then paired HUDLessColor (2) difference. |
| UI with FG on | Use the same captured-source order and HUD-less fallback; presented alpha is excluded. HUD-less pairs exactly with its same-batch tagged Backbuffer, else inexactly: a snapshot's first offer on the Present after its tag with that Present's colour, and a later first offer and every later offer of the same snapshot with its own Present's colour, so the real frame's own Present is compared wherever DLSS-G's generated-first order places it; only V2's pixel test validates these pairs, and a mispaired re-offer keeps the held decision (T1). UIAlpha stays manual-only when the loaded interposer is outside the surveyed header range. |

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
shaders. Both are zero on frames nothing consumed. `gpu_profile` says whether the timestamp
queries started (`no_timestamp_frequency` and `no_query_heap` name a failed start), and frames
that did not count are split into `dropped_fence_pending` (the slot was reused before the frame's
completion fence passed), `dropped_unresolved` (the fence passed but the results were not
readable) and `incomplete` (render or conditioning marks missing). Where the device can copy query
results (D3D12), the renderer resolves each timestamp into its own readback buffer on the command
list that recorded it, so its completion fence alone decides readiness. ReShade 6.8 reads D3D12
results through `ID3D12Device15::ResolveQueryData` whenever the game's D3D12 runtime offers it
(The Witcher 3 ships Agility SDK 1.619), but creates its query heaps without
`D3D12_QUERY_HEAP_FLAG_CPU_RESOLVE`, so that read fails and every frame there was
`dropped_unresolved`. D3D11 reads ReShade's results. The line ends with the offscreen UI layer's
live copies over the same 10 s (`ui_layer={copies skipped offers presents_since_copy={mean max}
skip_gap_ms}`): copies recorded, copies skipped because every ring entry was offered, held or still
read, Presents offered a copy, the offered copy's Present count over those Presents (1 when every
Present reads the copy of the frame before it; see the layer cross-queue fence under
[Diagnostics switch and per-Present cost](#diagnostics-switch-and-per-present-cost)), and, since
10-07, the longest time the layer went without a recorded copy across skipped copies
(`ui_layer::skip_gap`, 0 without skips; time between clears more than 250 ms apart is left out). All
are zero while UI detection does not ask for the layer. Since 10-07 the line ends with `window_ms`, the
window's length: 10 s for a periodic line, and shorter for the last window of a runtime reset or
destruction, which logs its partial window instead of dropping it (before, a game's exit window, and
any UI layer copies skipped in it, was never logged). The CPU entry also splits the slowest present into setup, depth, UI, render and
export, and a rate-limited `Sunshine Game 3D hitch` warning names any present-thread step that
takes more than 8 ms. Each step name has its own once-per-second throttle, so a nested step and the
step around it both log. The GPU stage times exist only while the add-on's Diagnostics switch is on
([Diagnostics switch and per-Present cost](#diagnostics-switch-and-per-present-cost)); off, no
timestamp is recorded, the line keeps its CPU fields and reports `gpu_profile=disabled`, and the
query heap is created on the first enable. The conditioning stages (`linearize` through
`horizontal`) time the depth conditioning wherever it was recorded: with the pack it was owed to,
or in the render for a probe frame; a render whose pack never comes counts with zero conditioning
and pack, and only a missing render mark counts as `incomplete`.

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
field between them, as before. That pass is tiled the same way (`SUNSHINE_UI_PIN_LINE_GROUPS`).
Each chunk summarizes, as an anchor texel and a slack, the bound of the **UI pin band** (below)
that reaches furthest past each of its ends, and lane 0 combines the summaries into carries. Each
chunk then applies the band in place in two scans: backward with the anchors at or right of each
texel, then forward with those at or left of it. The forward scan reads values the backward scan
already pinned, which never tightens a bound beyond the band, and is skipped where the backward
scan left the whole chunk on the plane. Rows without UI are only scanned once, and no row is held
in group memory. Timed alone at 3840x2160 on dumped inputs (with Sunshine running), the band pass
took 0.26 ms on a Stellar Blade menu and 0.25 ms on its HUD, where the distance scan it replaced
took 0.31 and 0.35 ms, and 0.20 ms against 0.15 ms with an all-ones mask, its worst case. The
mask pass, one load of the selected alpha per texel, took 0.063 ms on those layers (0.030 ms at
2560x1440 on The Witcher 3's notice board). When the limiters were tiled,
the 4K vertical pass went from 0.44 to 0.17 ms on an idle GPU and the horizontal pass, including
the UI pass, from 0.35 to 0.24 ms; the whole renderer went from 1.38 to 0.91 ms. The shader declares the lines per group as
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

### Diagnostics switch and per-Present cost

**Diagnostics switch.** `Diagnostics` under `[SUNSHINE_GAME3D]` in the game's `ReShade.ini` is
read once at add-on start; when it is absent the
add-on writes 0 so the key can be found, 1 turns it on, and any other value is off. It is one
process-wide switch (`game3d_diagnostics.h`), and the Troubleshooting panel's **Diagnostics** row
below **UI mask source** edits it at runtime in both directions. Loading and every edit log
`Sunshine Game 3D: diagnostics on (Diagnostics=1)` or `... off (Diagnostics=0)`. Off (the
default), the add-on records none of its diagnostic-only per-frame work: per-pass GPU timestamps
and their resolves (the timing line keeps its CPU fields and hitch lines stay), and the Streamline
V1 Reflex / V2 PCL presentation brackets, which only the probe's presentation trace reads (the PCL
marker hooks are installed only while it is on; see
[Upscaler call-route diagnostic](#upscaler-call-route-diagnostic)). The A2 one-way
statistics (A2 revocation and the one-way judgment read them) run with the switch on or off, on
status samples only (per-frame bit `0x4000`, at most one every 100 ms); the exact UI counters
always run. Decisions, masks, candidate binding and exported pixels
are the same with the switch on or off. The capture owner's queue watch, the layer's carrier
events and its submission listener stay registered either way, since direct binding reads them,
the layer's live-copy ring offers a copy only once a list carrying it executed, and its
cross-queue fence is signalled after that submission. The switch's own cost was not measured separately: Verify
found no renderer-level GPU difference with it on or off. Before selection revision 9 the switch
also gated S3's frame-identity shadow (its clocks, stamps, proposals and log lines) and H2's own
evidence passes, both removed, and with it on candidates were copied rather than bound directly.

**Owed conditioning.** `render()` records the source copy, UI detection, retained colour and every
CPU decision. The depth conditioning that only the pack, a dump or the adaptive probe read (depth
candidate, vertical, nearest-UI tiles and reduce, horizontal, UI pinning) is owed with the pack,
and `pack()` records it first (into an export slot, `output()` or a Dump 3D, inside the pack's
isolated D3D11 state). A render records it at once for an adaptive-probe frame, an older two-pass
replay shader or an immediate pack; a probe frame records it only while a consumer is attached or
Dump 3D is armed (`render_frame_input::adaptive_probe`): without one nothing reads that
conditioning, so no probe is submitted (until the 10-05 review a monitor-only session spent about
0.6-0.9 ms of 4K conditioning ten times a second on it) and placement holds until a consumer
attaches, which probes at once; a Present whose pack is never recorded records none, and the
next render or `begin_present` drops what it owed. An owed pack that shows the source mono (no
depth or camera, zero strength or blend, the depth preview, or a source larger than 3840) records
none unless Dump 3D is armed (`render_frame_input::diagnostic_armed`); the shader declares
`SUNSHINE_MONO_SKIPS_CONDITIONING`. Diagnostic textures hold the last recorded conditioning.
`render_frame_input::depth_identity` is nonzero only when the depth view's content is immutable
for that value: the exporter derives it from the API capture's epoch, source, sequence and
viewport, and Generic depth gives 0. An exact repeat of {view, nonzero identity, the 80 `b0`
bytes} (a generated Present reusing its real frame's depth) keeps the last depth candidate and
vertical field. When that repeat also has the same UI words (`b1`: UI on, plane mode and depth or
fraction, channel) and, with UI applied, the same detected mask, which no detection rewrote since
(a held generated Present), it also keeps the final field: neither the horizontal nor the UI pin
pass is recorded (`conditioning_activity().field_memo`). A probe frame, an armed dump and the
nearest-UI plane always record both. Until the 10-05 review every held generated Present
re-recorded them on unchanged inputs (about 0.25-0.4 ms at 3840x2160). Packed and dumped outputs
are byte-identical to always conditioning. An owed pack
without an armed dump runs the live vertical pass (`SunshineHostVerticalLiveCS`, declared by
`SUNSHINE_VERTICAL_LIVE_ENTRY`): the same field without the final majorant write-back, which only
Dump 3D reads, so the majorant texture keeps its intermediate values. `render()` (replay,
fixtures, probe frames) and an armed pack record the full `SunshineHostVerticalCS`, and an armed
pack never keeps the memo, so a dump's majorant is its own render's. At 3840x2160 the vertical
pass went from 0.230 to 0.200 ms.

**Detection and pinning shape.** The detection tiles pass splits each statistics tile over
`SUNSHINE_UI_DETECTION_TILE_PARTS` groups (4; group z is the part, taking the tile's 16-row runs
k modulo 4). Part k of tile (x,y) is stored at column x+16k of a 64-column statistics texture, and
the reduce sums the parts of every per-tile row before it matches tiles; integer sums make the
result independent of the split. Unoffered candidates are unbound and their loads are skipped by
uniform branches. UI pinning walks its three serial scans in 16-texel preloaded blocks with the
same arithmetic and order. A per-row occupancy prepass that skipped rows without UI was measured
and dropped: the pass time is the slowest group's serial chain, which empty groups already run
beside. The detected mask stays R32F: R16_UNORM cannot hold FP16 or 10-bit alpha candidates
exactly and R16_FLOAT cannot hold 8-bit alpha, so a 16-bit mask would change masks. At 3840x2160
detection went from 0.20 to 0.14 ms and UI pinning from 0.28 to 0.22 ms on a HUD-like frame;
owed conditioning removes about 0.5 ms from every Present whose pack is never recorded.

**Direct candidate binding.** With no dump armed (with the Diagnostics switch on or off since
selection revision 9), a UI candidate is read where it lies instead of being copied into the
renderer's slot first. A Streamline tag
snapshot captured on the presenting queue is leased (`depth_capture::lease_local_view`): its
storage moves to the shader-resource state on ReShade's immediate list, the lease registers that
recording as a consumer exactly like a copy, so the storage is neither reused nor released until
the submission's fence passes, and `end_local_views` returns it to its resting state after the
Present's last read (its render, owed pack and dump). A snapshot leased on the Present's recording
is never also copied there, even when the renderer cannot bind it (it is then not offered that
Present), and `end_local_views` runs whenever a lease was taken. The renderer describes the leased storage
in its own original-device descriptor (`renderer::bind_ui_snapshot`); the capture owner's own
descriptor belongs to a heap ReShade wraps, which `push_descriptors` cannot copy. The offscreen UI
layer keeps a ring of live copies (`ui_layer::ring_capacity`, 6): each
before-clear copy goes to an entry other than the newest and the one a Present is reading whose
last reader, a renderer submission, completed (`ui_layer::bound` with the renderer's completion
fence and value). The entry `ui_layer::latest` offers stays pinned until `bound` registers that
Present as its reader: in between it has no reader yet, and a game thread may promote a newer copy
and clear the layer again, which before the 10-05 review could record the next copy into the entry
the presenting queue was about to read. A Present that renders nothing still binds (a copy into
the renderer's slot recorded at acquisition may read the entry), and every bound reader's value is
the completion signal its Present claims (`renderer::claim_completion_value`), which that Present's
`finish_present` sends whether or not it rendered, and which a renderer released after its
runtime's drain signals from the CPU on D3D12 when that `finish_present` never came. Before 10-07 only
a rendered Present signalled: a run of Presents that acquired without rendering (depth lost) kept
each read entry busy until the next render and could fill the ring, and a renderer released before
it rendered again (a swapchain recreated at the same size on the same device, or a cached renderer
dropped) left its entry busy until the ring was retired. With
every entry busy the ring grows, and a full ring skips that copy and keeps the offered copy offered,
counts it in the timing line (`ui_layer={... skipped=...}`) and logs once per ring (INFO since 10-07,
a WARN before). Presents keep reading the older offered copy until a copy is recorded again, and the
host publishes those Presents like any other, so a skip costs time, however fast the game presents:
the timing line's `skip_gap_ms` is the longest time the layer went unrefreshed across skipped
copies, about 3 ms for one skip at 590 clears a second and 43 ms at 47. The readiness report's `UI
layer copies` check warns on a streamed window whose gap exceeded one stream frame interval, where
streamed frames could read a UI layer that missed a whole stream frame of UI changes, and on any gap of
250 ms or more, after which the layer is no longer offered at all. The Witcher 3's exit screen (10-07)
presented about 590 times a second when the ring first filled; its skips were never counted, so
their cost is unknown. A clear needs the offered entry, one entry per copy
held for its cross-queue fence (one per frame the copy's queue runs behind), one per earlier
offered copy an unfinished Present still reads (one per frame the presenting queue runs behind) and
the target: 2L + 2 with both queues L frames behind, so six entries hold L = 2. Four sufficed while a
cross-queue copy was offered at its signal (the GPU wait below); with copies held until their fence
completed, four would leave a game whose queues run two frames behind (Stellar Blade's copy queue
with frame generation waits for presenting-queue work queued after the Present) skipping every other
copy. The ring only grows when every entry is busy, so a game that needs three entries still
allocates three. An entry whose
allocation failed is taken last, so it is allocated again only when no entry with a copy is free,
as a growth would be, and the ring keeps rotating its existing copies meanwhile. A recorded copy is
pending until a list carrying it executes (on D3D12, until the submission hook ran after the native
call, and when it owes the layer cross-queue fence a signal, until the CPU saw the fence reach it);
only then is it offered, so detection reads the newest such copy as
the single live texture did. A carrying list reset unexecuted returns the entry
(never offered), and a copy pending for 2 s is abandoned. Dump 3D census copies follow the same
pending rule (never abandoned) and are read only once proven complete
([census ordering](#dump-3d-diagnostics)). Each entry keeps the queue that
executed its carrying list (one executed on two queues has none). A copy that ran on the
presenting queue alone (a D3D11 immediate or deferred-context copy always does) is in queue
order before the Present's reads once its write reached that queue: on D3D12 it is offered only
after its native submission returned (below). The verdict is per copy: before WP1b one scope-wide queue watch
remembered at most four carrying lists, so a fifth dropped the oldest list's queue while its copy
was still promoted, and one foreign execution kept the whole scope from direct binding. A copy
in queue or fence order (below) is bound directly; one no fence orders is copied. Streamline
snapshots on D3D11, from foreign or mixed queues and sRGB-stored snapshots read through UNORM keep
the copy, and so does every candidate while a dump is armed. A leased snapshot lives in the
capture owner's 32-slot colour/mask pool, where live snapshots and Dump 3D copies keep separate
byte budgets (live storage is sized from the request, dumps have a fixed budget; see
[Streamline depth selection](#streamline-depth-selection)), so an armed dump never evicts or crowds
out the live storage a lease binds.
The pixels detection reads are the same either way. The log
names the first direct read of each kind once (`Sunshine UI input: reading ... directly (no copy)`).

**Layer cross-queue fence.** On D3D12 a copy whose carrying list ran on one other queue is ordered
by a fence of the add-on that only the CPU reads (`ui_layer::read_order`); the presenting queue
never waits for a layer copy. ReShade reports an execution before the native
`ExecuteCommandLists`, so the execution only marks the copy as owed to that native list; the
add-on's own submission hook (the capture owner's native observer, which already watches every
queue's `ExecuteCommandLists`, through `native_observer::set_submission_listener`) runs right after
the native call returns. Until then every D3D12 copy executed on one queue stays pending
(`ui_layer::held_for_submission`), the presenting queue's own included: a Present on another thread
(a frame-generation pacer, an engine's separate submission thread) between ReShade's event and the
native call would otherwise bind the copy in queue order and submit its reads before the copy's
write reached the queue, reading the entry's previous content or, in a new entry, an empty layer.
The hook then offers a copy that ran on the presenting queue in queue order, without a fence, and
for a copy that ran on another queue signals that queue's fence, so the value follows the copy in
that queue's order, and records the value with the ring entry. That entry stays pending until the
CPU sees the fence reach it (`ui_layer::fence_pending`: `GetCompletedValue`, never a wait), checked
whenever a Present asks for the layer (`ui_layer::latest`) and before a clear picks a ring entry.
A pending entry is neither offered nor rewritten, and the copy offered before it stays offered, so
while the hook is heard a Present never reads a copy whose write is not yet submitted or may still
run (offering a cross-queue copy at the execution event let a concurrent Present copy it unordered
while the game's queue wrote it). Without the hook heard (below) a same-queue copy is offered at
its execution event and relies on the game submitting and presenting from one thread, as before. The
layer is a one-frame-late input by design (E2); a cross-queue copy is offered a frame or more later
still while its queue runs behind the CPU, and the dumps' `presents_since_copy`, `age_ms` and
`capture_id` all describe the offered copy (`age_ms` from its own recording, not the newest
recorded copy's, which only gates recency). Once the fence reached the value the entry is offered unless a newer copy is
offered already, in fence order (`fence_passed`: it completed before anything the Present submits)
and bound directly like a same-queue one. Only the copy's own queue signals its fence, in
increasing values, and nothing signals it from the CPU, so a reached value stays reached (a removed
device reads `UINT64_MAX`, which reaches it). A reset of the carrying list after its submission
(legal while its work runs) keeps the entry held for its fence. A submission the hook never reports
cannot hold a copy forever: a reset of its list before the hook offers it (a cross-queue copy
unordered, logged as not observed; a same-queue copy in queue order), a copy held for the hook for
2 s is abandoned like an unexecuted one (one whose fence signal is recorded is never abandoned,
since its queue may still write it), and while the hook was not heard in the last second (the
capture owner's observer inactive) an execution offers its copy at once (a cross-queue copy
unordered). The fence is created on first use, one per device and queue (at most four),
from the queue's own device; a record is reused only once no live or census copy names it and its
last signal ran.

From 10-05 to 10-06 (5b83f8a5, fcb47ad0, 19fa56a6) the presenting queue instead waited on the GPU
for that value before reading the copy, bracketed by marker signals of its own, with a 1 s CPU
watchdog that released a stalled wait from the CPU and revoked the queue. Live in Stellar Blade
with frame generation on (10-06 13:45:44) the presenting queue was blocked at that wait for 1078 ms
with no fence progress and the game froze for a second: the copy's queue may be paced by, or wait
for, presenting-queue work queued after the Present, so any GPU wait of the game's queue for
evidence can close a cycle. The wait, its reuse for re-offered copies, the marker fences, the
watchdog, the queue revocation, the teardown drain and the timing line's `layer_fence_waits` were
removed. The D3D12 provider fixture (`reshade_game3d_native_provider_runtime_test` with the
frame-generation interposer) reproduces that schedule, each frame's copy run on a second queue
behind a signal the presenting queue queues after its Present and the next frame waiting for the
previous copy: two frames now complete in 16 ms, where the wait closed them into a cycle that only
the 1 s rescue left. A Present while the second queue is gated for up to 1 s reads the previous
copy without waiting (the whole Present step measured 0 ms at the tick counter's resolution), and
the gated copy is offered in fence order once its fence reached it.

Where no fence orders a copy (its submission not observed, the fence not created or not signalled,
or a copy run on two queues), the copy is still offered at once and copied into the renderer's
slot unordered, as before the fence; each reason logs once (`Sunshine UI layer: the live layer copy
executed on queue ..., not on the presenting queue ..., and no fence orders it (...)`), and the
first fenced copy logs `... such a copy is offered once the CPU sees the fence value signalled after
it reached, never waited for`. Dumps record the layer's `read_order` (`queue`, `fence_passed` or
`unordered`) and `fence_value` beside `executed_queue` and `presenting_queue_order` (queue order
only). Census rows record their own `read_order` and `fence_value`; a census copy without proof is
omitted, never read unordered ([census ordering](#dump-3d-diagnostics)). Refusing cross-queue
copies was considered and rejected: the Stellar Blade (SDR, FG off) and
The Witcher 3 (FG on) dumps of 10-04 both record their layer copies as run on another queue than
the presenting one, so a refusal would remove their layer protection; before this fence such a copy
was read with no ordering at all (a torn or next-frame mask, both queues touching the texture).

**Colour toggles and the overlay.** The renderer keeps the working set of the previous colour
transfer or extent (one cached entry) and makes it current again on a toggle back, without a wait
or a recompilation; it starts every per-scene decision state as a new renderer would, and an
entry cached for 60 s is released once its fence completed. The exporter likewise keeps the
previous finished export ring (one entry, 60 s) and renews it under a new generation when its
source matches again after 500 ms inactive; the current ring, once it stopped exporting, is
released after 10 s idle ([GPU handoff contract](#gpu-handoff-contract)). A game toggles by ResizeBuffers, around which ReShade
resets its effect runtime (`destroy_effect_runtime`, `init_effect_runtime`): the renderer is parked
over the reset (its back-buffer views dropped) and taken back by the reinitialised runtime, and the
finished export rings stay, without any back-buffer reference (submitted sources and overlay
compositors are released). Both are released only when ReShade destroys the swapchain itself
(`destroy_swapchain` without resize) or its device. The overlay compositor's shaders for the current
export transfer are compiled on the thread pool once a renderer is ready, so opening the overlay
neither compiles nor stalls the Present.

**Present-thread bookkeeping.** `ui_layer::observe_output` caches the swapchain's back buffers and
re-enumerates them only when the swapchain, its size or buffer count, or its first buffer
changes, or ReShade reports it created, resized or destroyed. It does any of this only while the
layer is wanted (UI detection asked for it within the last second, or a dump is armed: one atomic
flag, which every transparent clear also reads before it resolves its view or takes the layer's
lock); otherwise a Present only retires the live copies once the layer was not asked for in 10 s
(a shorter lapse, such as a stall or a Game 3D toggle, reuses them instead of allocating them again
on the game's recording thread) and destroys retired copies once due. The layer allocates new
copies outside its lock (an entry is reserved under it, then allocated and published under it
again, or destroyed when its scope moved on meanwhile) and destroys the presenting device's retired
copies outside it, so a slow D3D12 allocation never stalls the Present thread or another recording
thread. A retired copy of another device (the ring of a replaced device) is destroyed under the
lock that device's `destroy_device` takes, so the device cannot go first. A retired copy is due
2 s after it was retired, never earlier, even when another thread retired it after the Present
read its clock. The depth owner observes the effects
list, the immediate list and the queue once per Present (`observe_present`), and later copies on
the registered immediate list skip their repeated interface checks.

**Depth handoff.** When the API source is no longer associated (Streamline FG released while NGX
evaluates, or the reverse), API ownership is held in mono (depth not ready) for at most 250 ms
after the last associated Present, with the same runtime and swapchain size and a live API
evaluation, instead of starting the Generic selector, its challenger and two DEPTH rebinds. Any
real evaluation within those 250 ms is live, Streamline SR's (logical source 0) included; only the
malformed-input marker (epoch 0) is not, and it never keeps an established Streamline SR source
associated either. Until the final review a live evaluation needed a nonzero source, so FG Off in
a game that runs SR and FG through Streamline fell back to Generic for a Present while SR's first
capture was pending. An explicit release with no other evaluation falls back at once. The log line ends `holding API
ownership (mono) for a provider handoff, at most 250 ms`, and the readiness reason is
`source_handoff`. Display storage keeps one entry per shape and format (at most two), so a flip
switches entries rather than reallocating; a replaced entry is retired by a fence signalled on a
later Present, with no `wait_idle`. New snapshot storage is reserved under the capture lock,
allocated outside it, then re-validated and published under it; a reservation is only a
preference, each byte budget (live snapshots or Dump 3D copies) counts the outstanding
reservations of its own kind, and a lost race releases the new objects after the lock. The flip's
present-thread steps each log their own hitch line: `native depth frame`, `depth display
retirement`, `depth present observation`, `depth list coverage report`, `frame generation query`,
`depth readiness trace`, `API depth binding`, `generic capture switch`, `generic challenger
release`, `generic depth rebinding`, `depth capture acquisition` and `depth display preparation`.

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
   Windows display scaling above 100% on that display makes a game that is not DPI aware see the
   display at its logical size (3840 × 2160 at 150% looks like 2560 × 1440), so a borderless game
   renders there whatever its own resolution setting says. When the game's back buffer equals that
   logical size below the display's real mode, the panel shows an orange warning, ReShade.log gets one
   `Sunshine Game 3D: display scaling limits the game` line per swapchain initialization, and the
   readiness report adds a **Display scaling** WARN (`game3d_display_scale.h`). Set scaling to 100%
   for that display, or set the game's .exe high-DPI override to **Application**, and restart the game.
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
   default value; opening the UI preserves existing saved values. The UI protection row's
   **Forget** button, laid out like them, clears the UI sources Auto accepted for this game (A3,
   below) and is disabled while none is accepted.
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

When the game requests Frame Generation, the **Status** tab shows its multiplier (for example
**Frame Generation: 4x**) without a hint: every multiplier is supported, and the stream takes each new
frame the game presents up to the stream rate. **DLSS Super Resolution can stay on.** The line reflects
verified game settings, including Auto mode, rather than measured generated output. Ambiguous settings
are not guessed, and Sunshine does not change the game's Frame Generation options.
When Automatic has identified the relative-depth fallback without associated camera data, it
suggests trying **Frame Generation**, if available, because some games supply Streamline camera
data through FG. Availability is not guaranteed. An already observed enabled FG setting instead
shows that camera data is still unavailable.
The hint is not shown during the initial unknown-depth state. A busy or ambiguous FG observation
does not produce a suggestion to enable a mode that may already be enabled.

**UI protection** offers **On / Off / Auto**, saved per game in ReShade.ini as
`SourceAlphaUIMode` (0 Auto, 1 On, 2 Off). All runtimes in the game share this choice and its
per-game configuration file. Auto is the default. On and Off are persistent manual
overrides, including across game restarts. The previous `SourceAlphaUI=false` migrates to Off;
the old enabled heuristic migrates to Auto. An explicit new mode takes precedence over the old key.
No game-specific detection profile is used. When protection is enabled, mask alpha at or above
1/8 (32/255 in 8 bits) pins already-composited UI exactly to one global plane selected by the adaptive
display-fraction policy below. Fainter alpha pins in proportion, so faint halo and vignette tails
blend into scene depth without a step, and zero alpha keeps scene depth (**UI pin band**
below). This is a placement policy, not recovered UI geometry or physical distance.
Auto detects eligible inputs continuously and validates their current pixels on the GPU. It
requires no human review, approval, game profile, startup scan or confirmation timer. The
candidate order, the [UI decision framework](#ui-decision-framework)'s draw order (S1), is
authenticated UI opacity, the tagged UI color alpha, the game's own offscreen UI layer (described
below), captured real-input (Backbuffer) alpha, current color alpha unless FG is known enabled, then
a paired HUD-less/final-color difference. Only an accepted candidate (below) that is valid this
frame decides, the first such in that order. A missing, unaccepted or invalid candidate allows the
next candidate in that same render, except that an offered, accepted UI alpha or UI color tag keeps
the inferred candidates (the layer, Backbuffer and current alpha) out even while it is invalid
itself. If none passes, UI protection is off for that frame; it can recover automatically on a later
frame.

The Status tab warns, in amber under the UI protection line, when Auto has rendered without a UI
mask for at least 2 s (`alpha_trust_span_ms`, the span over which detection earns or loses
trust): "No UI mask in this game mode: HUD and menus take the scene's depth." A rendered frame is
unprotected when no source decided and either no UI source was offered or no dedicated UI channel
(UIAlpha, UI color or the UI layer) was offered clean and empty. A clean empty dedicated channel
means no UI on screen, and an accepted channel deciding an empty mask protects, so neither warns.
`source_alpha_ui_decision::unprotected()` owns this definition and the log report's
`UI protection gaps` check mirrors it with the same 2 s minimum. The exporter keeps the start of
the current unprotected run: a decided rendered frame (a mask, or a clean empty UI channel), a
manual mode, protection Off or Game 3D off clears it. A Present that rendered nothing, or whose
status sample is still being collected after a change of the winning accepted candidate or of the
epoch, revision or viewport (F1, below), neither starts nor ends it. The warning is hidden in On
and Off, in 2D, in depth preview, at zero strength and while an FG capture block has its own
text. Hints follow the evidence: "Try HDR output" only for an SDR swapchain whose latest sample
read an offscreen UI layer with color but no alpha (`source_alpha_ui_decision::layer_without_alpha`, below; a layer that V1
rejects while it has some alpha, such as a scene buffer, gives no hint), and "Try Frame Generation"
only when the game reports FG off. Neither names a game, and Sunshine changes no game setting.
Stellar Blade in SDR with FG off shows both: its layer then holds the scene image, current alpha
covers every pixel and its same-frame HUD-less differs from the final frame on about 2% of pixels
along edges and rain, so no mask is possible. Its full-screen SDR menus with FG on also stay 3D
when the Backbuffer alpha covers nearly the whole frame (below).

The **Troubleshooting → UI mask source** selector offers automatic discovery and explicit source overrides,
including HUD-less difference. It is useful for diagnosis; normal use needs no source choice.
UIAlpha is admitted to Auto when the hooked interposer's version lies in the surveyed header
range that assigns it (2.11.x tag 68; 2.12.0 through 2.14.x tag 69); see the
[buffer contract](#streamline-buffer-type-contract). Outside that range it stays manual-only, and
its presence never blocks tag 23, Backbuffer, current alpha or HUD-less comparison. The same GPU
content checks apply as for every candidate. Manual On (S2) offers the filtered candidates
through the same detection and accepts every one of them for this session only: it never
persists, earns or revokes acceptance, and UIAlpha is eligible outside the surveyed range. An
all-one alpha channel then places the whole frame at the UI plane. Where detection cannot run (a
frame larger than 3840, or detection resources that could not be prepared), Manual On applies
its explicit first filtered capture in draw order, and adaptive placement matches that
capture's own epoch, revision and viewport.
Off always disables protection. Neither manual mode overrides GPU capture ordering,
source lifetime, scope, format or pairing requirements, and changing mode does not recalibrate depth.

Automatic selection separates two questions. Whether a source carries UI coverage at all is a
property of the game, so it is learned once and remembered for that game (acceptance, A1). How much
UI the frame shows is read from that source's current pixels on the GPU, whatever the answer: none,
a HUD, or a whole menu. Validity (V1) does not depend on acceptance: alpha that is non-finite or out
of range on more than 1% of pixels is invalid for that frame, and so is an offscreen UI layer whose
pixels beyond the premultiplied bound (below) lie on more than 1% of the frame and either on more
than 5% of it or on more pixels than its opaque pixels within the bound. Presented and Backbuffer
alpha are checked for range only, at any bit depth, so Expedition 33's 2-bit alpha stays eligible.

Acceptance is one bit per game and source signature: the candidate kind (UIAlpha, UI color tag,
offscreen UI layer, Backbuffer, current color or HUD-less pair), its typed DXGI format and the
swapchain colour space. Frame generation is not part of it, so turning FG on or off keeps it, while
HDR and SDR output are different signatures that each earn their own. A declared source, the game's
own UI contract (the UIAlpha and UIColorAndAlpha tags, and a HUD-less pair, exact or not), is
accepted by its first valid selective sample, one covering some but less than 90% of the frame; for
a HUD-less pair that is a partial change set that passes V2's tile test, which proves its pixels.
A game that tags only HUDLessColor pairs every image by Present counting, inexactly (below), and
Hogwarts Legacy did on 10-05: while only exact pairs earned, its pair was never accepted. An
inferred source (the offscreen UI layer, Backbuffer and current alpha) is a guess: it needs
consecutive completed GPU samples covering some but less than 90% of the frame at least three times
over at least 2 s, each within a factor of two of the others. A full-frame sample restarts that run,
and so does, since selection revision 10, a selective sample more than 2 s after the run's last one
(empty samples within that bound stay neutral), so a run is bounded: sparse selective samples spread
over minutes never add up to acceptance. Coverage that swings more (scene effects such as particles
in a transparent target) never earns acceptance. A source that is never selective is never accepted: an empty or nearly full-scene source
is ambiguous, because an opaque channel can carry no UI at all (Hogwarts Legacy's UIColorAndAlpha,
and Stellar Blade's in HDR, are the opaque final image during play). A sample in which an offered
UIAlpha or UI color tag is invalid is void for Backbuffer and current alpha, the sources whose
declared-coverage judge (below) it lacks: their runs neither advance nor restart, and it is not
testable for them (Resident Evil Requiem's rejected-tag frames). The other declared tag, the layer
and the HUD-less pair meet the same judges as on any sample and earn as usual; until the final
review the void stopped them too, so a declared tag invalid on every frame (an uncleared FP16 target)
left a restored HUD-less pair to lapse with nothing able to earn it back. Holds and manual inputs never earn.

Once accepted, a source is the mask on every frame in which it is offered and valid, ahead of the
sources after it in draw order, at any coverage (P1). A full-screen menu is then simply a source
covering everything, and a pause menu that tints the live scene needs no special case. Clair Obscur:
Expedition 33 motivated the model: its alpha covers 2-23% of pixels during play and every pixel in
full-screen and pause menus, and with FG off, or before its first level loads, it tags no HUD-less
image. Its pause menu with FG on differs from HUD-less on 76% of pixels, which is neither a HUD mask
nor full-screen UI. Acceptance belongs to a signature, never to a detection slot: a presented-color
alpha accepted with FG off does not make the tagged Backbuffer's alpha accepted, and the offscreen
UI layer has a candidate slot and bit of its own beside the tagged UIColorAndAlpha (E1). Stellar
Blade in HDR offers both with FG on. When the layer and the opaque tag shared one slot and one trust
bit, turning FG on handed the layer's trust to the tag, which flattened the whole frame for 15 s
until a contradiction revoked it under the full-claim rule of that time.

Acceptance is remembered per game. Each change is written to `TrustedUISources` under
`[SUNSHINE_GAME3D]` in that game's `ReShade.ini`, as a ReShade array of signature keys `<kind>:<DXGI
format>:<colour space>` such as `ui_layer:28:srgb` or `backbuffer:24:pq` (kinds `ui_alpha`,
`ui_color`, `ui_layer`, `backbuffer`, `current` and `hudless`; spaces `srgb`, `scrgb`, `pq` and
`hlg`); at most 32 are restored. Since fix 1 the same list also holds the ledger-only kind `pre_ui`:
`pre_ui:<layer format>:<colour space>` such as `pre_ui:87:srgb` records that an offscreen UI layer of
that signature was proven the pre-UI scene image (H1 (d), **The layer's pre-UI proof** below). It is
never a UI source: it neither decides, blocks nor judges. The next session starts with those sources accepted. Entries in any
other form, the integer bitmask written before S1 included, are discarded on load with the log line
`discarded N legacy UI trust entries`, and the older `TrustedUIAlpha` key is no longer read: neither
can say which signature earned it, so each source is accepted again once by its own evidence.
Without remembered acceptance, a launch's title screen and menus would stay 3D until a selective
frame had been on screen for 2 s; Expedition 33's first screen covers every pixel, and leaving its
main menu within 2 s left the Load Game menu unprotected until play. Only a game's first session,
and its first session after that discard, still has to earn acceptance. **Forget** on the UI
protection row (A3) clears this game's acceptance in the session (accepted and provisional entries
with their earning runs and contradiction windows, pre-UI proofs included) and in `TrustedUISources`, which becomes empty; it logs
`accepted UI sources are now none` followed by `forgot learned UI sources <keys> for this game`,
counts `trust.forgotten`, and touches no hold or hidden-scene guard state. Each source is then accepted
again by its own evidence. Deleting the key forgets it for later sessions.

Acceptance is revoked only by evidence of stronger provenance in the same sample (A2), whatever
the drawing rank. There are two judges: a valid exact change set (an exact HUD-less pair whose
change set passes V2, partial, empty or full, accepted or not), and each offered, accepted, valid (V1)
UIAlpha or UI color tag. The exact change set judges every alpha but the one-frame-late layer copy
(below): UIAlpha, the UI color tag, Backbuffer and current alpha, when offered and valid, accepted or
not. Since selection revision 10 that includes the declared alphas, so that a declared alpha wrongly
accepted from one selective sample (an opaque final image tagged as UI color, read selective once
during a fade) no longer decides the frame flat until Forget. A declared tag is judged one way only
when it was captured in the exact pair's tag batch (the same Present interval as the pair's
Backbuffer, `ui_mask::same_tag_interval`): each kind is the newest ready snapshot of its own, and a
tag of another frame (UIAlpha tagged in a list that completes a frame later) is not same-sample
evidence (E2), since UI that moved, appeared or disappeared between the two frames would read as
contradicted where the pair shows the unchanged scene. The renderer pushes such a tag unaligned
(per-frame `0x1000` UIAlpha, `0x2000` UI color), and the tiles pass counts no strong pixel of it, so
it is not judged on that sample, its contradictions are not counted, and it keeps deciding. The
declared alphas' coverage judges the inferred Backbuffer and current alpha only; the layer copy is
never judged.
The one-way counts are counted on status samples only (the frames whose decision texels the CPU
reads, per-frame bit `0x4000`), in the passes' second phase, since no other frame's counts are read.

- **One-way lit-pixel disagreement.** An exact pair contradicts a judged alpha when at least a
  tenth of its strong pixels (finite alpha of at least 1/2) lie where the HUD-less image is lit
  (above eight times the difference threshold, as in the full-frame rule below) and unchanged
  (within half the threshold) against both the pair's tagged Backbuffer and the presented frame:
  the scene shown without UI on screen. One rule for every judged kind. The GPU counts both per
  tile (statistics rows 80-111, below). Real UI changes the final image under it, and a dim or tint
  either changes the pixels it covers or covers dark ones, so neither meets the test: Expedition
  33's pause tint and The Witcher 3's sign-wheel backdrop stay accepted, and so does a correct
  UIAlpha or UI color tag. UI drawn after the tagged Backbuffer (a game that tags its Backbuffer, or
  its final colour as HUDLessColor, before compositing the UI, or late subtitles and a cursor)
  changes the presented frame, so it never contradicts its own tag; until the 10-05 review only the
  tagged Backbuffer was compared, and such a pair (an empty change set) contradicted every correct
  UIAlpha or UI color tag and revoked a restored one within about 0.2 s. An opaque final image read
  as UI alpha marks the scene unchanged in both images and still meets it; under frame generation
  the presented frame may be another frame, which only makes a contradiction rarer. The test is
  one way: alpha missing where the pair changed contradicts nothing. It needs a basis: a source
  without strong pixels in the sample is not judged by it.
- **Declared-versus-inferred coverage disagreement.** An accepted declared alpha contradicts an
  inferred one whose covered pixels differ from those of each accepted, valid declared alpha in the
  sample by at least 10% of the frame. Resident Evil Requiem's UI color covers 0.2% during play
  while its presented alpha covers 35-100%, and in menus the UI color covers everything while the
  presented alpha covers 0.05%. The layer copy is neither judged (below) nor a judge.

The one-frame-late layer copy (every layer until S4 replaces the copy) is not
same-sample evidence (E2): it is no judged kind (`ui_selection::judged_kinds`), so the GPU counts
no strong or contradicted pixel for it and neither judge reads it (before selection revision 10 the
GPU kept a layer column that was always zero and the sample recorded `late_layer`; through
180f1842 every layer copy also carried stored flag `0x4`). Comparing it with this frame's UIAlpha would otherwise contradict it on every UI
transition, and a moving marker would contradict it one way. An accepted source is revoked by three
contradicting judged samples within 2 s (`alpha_trust_samples` and `alpha_trust_span_ms` in
`game3d_alpha_auto.h`), counted `trust.revoked_exact` when the one-way test contradicted it in the
revoking sample and `trust.revoked_declared` otherwise. Older contradictions drop out of the 2 s
window; agreeing samples and samples without a judge change nothing, so a source contradicted on
most samples is revoked even when some agree, and ambiguous or invalid samples never revoke. One or
two contradictions within 2 s never revoke. The one-way judge reads same-sample evidence (a
declared tag only from the exact pair's tag batch), so a cut or animated UI does not contradict a
correct source through it. The coverage judge is not batch-aligned: it compares the newest offered
declared tag, which may be another frame's (UIAlpha recorded in a list that completes a frame
later), with this frame's inferred alpha, and only coverage that differs by a tenth of the frame on
three samples within 2 s revokes, so a UI fading or flashing over most of the frame for that long
could revoke a correct inferred source, which then earns again by its run. At the 100 ms sample cadence a persistent
contradiction revokes about 0.2 s after it starts. A contradicted sample also earns nothing and
restarts an unaccepted source's earning run, unless the sample is void for it. A revoked source must
earn acceptance again; remembered acceptance is revoked the same way, and the revocation is
remembered too. A declared alpha that the one-way test revoked earns again, for the rest of the
session (until Forget), only from a selective sample that the exact pair judged without
contradicting it, never from one unjudged selective sample: the opaque final image above reads
selective again at the next fade, when the dark HUD-less image fails V2, and until the final review
that sample accepted it again, so after every fade it showed the frame flat until three more
contradictions and rewrote `TrustedUISources` twice. Manual mode edits leave it unchanged. In a context without a declared or exact judge
(The Witcher 3, Stellar Blade in HDR and Expedition 33, all with FG off) a wrongly accepted inferred
source is cleared only by Forget or by the provisional lapse below; that is an open question of the
[framework](#ui-decision-framework).

Acceptance remembered from an earlier session protects from the first frame but is provisional
(A3): unless the session earns it again within 60 s (`alpha_trust_reconfirm_ms`) of testable
samples, it lapses and is forgotten, so a wrong remembered claim cannot outlive every session. A
sample is testable for a source when it could earn or refute it: the source is offered, valid (V1,
or V2 for a HUD-less pair) and not full (alpha below 90% of the frame; a partial or empty change
set). The clock counts only the time between consecutive testable samples of the source, each gap
capped at 250 ms (`alpha_trust_reconfirm_gap_ms`, two and a half sample intervals), and nothing
else touches it. Samples that are not testable for every kind (alpha that fails V1, a HUD-less pair
whose change set is not V2-valid, such as the middle band of a pair mispaired with interpolated
colour or an inexact pair changed everywhere, a full sample, and for Backbuffer and current alpha a
void sample), samples that do not offer the
source or offer another signature of its kind (frame generation that stops offering current alpha,
an HDR toggle), a manual mode and a time without samples (a loading screen, alt-tab) therefore
never count beyond one capped gap. A source never lapses during an invalid run, such as Resident
Evil Requiem's rejected-tag frames, nor while it is not offered, and an interleaved invalid sample
never restarts its clock, so a source that is testable now and then still lapses after 60 s of
testable time in total unless it earns acceptance. A full menu can neither earn nor refute a
source (only a selective sample earns), so counting it lapsed Stellar Blade's presented alpha during
long SDR menu visits, which left the menus unprotected until gameplay with frame generation off
earned it again; a full change set is not testable for a HUD-less pair the same way. Until 10-05 a
remembered inferred source kept a plain clock through its invalid samples, and a HUD-less pair's
clock ran through full change sets; until the 10-05 review the clock was wall time from a testable
sample to the next sample that offered the source, so a source absent for a minute (Stellar Blade
SDR with frame generation on, an HDR toggle) lapsed at its first offer back.

An accepted alpha candidate that is valid in the frame decides by itself (S1). While an accepted
UIAlpha or UI color tag is offered, inferred alpha never decides (the declared-alpha block): when the
declared alpha is invalid, the frame has no decision of its own rather than falling back to a source
that may cover the scene, and the temporal hold (T1, below) reuses the previous real frame's decision
for that one frame.

Some games expose no UI buffer through any vendor API but draw UI into their own offscreen layer.
Frostbite's HDR pipeline draws UI into an RGBA8 target at backbuffer resolution with premultiplied
alpha and composites it in the final pass; Unreal does the same with its HDR UI composite mode.
The Witcher 3 Remastered tags Streamline's UI buffers only while FG is on, and its presented alpha
is opaque, but it draws all UI into such a layer (RGBA8, premultiplied, alpha exactly the UI's
coverage). A layer is cleared to transparent black every frame, so the add-on observes each clear
of a single-sample 2D color target that matches the swapchain size, has an alpha channel of at
least 8 bits, is not a back buffer and is cleared whole to exactly (0, 0, 0, 0): without
rectangles, or with a rectangle that covers the whole target. A partial clear (a viewport strip,
one half of a split target) leaves the rest of the target's previous content, so it is not a
layer clear; before WP1b any rectangle qualified. A 2-bit alpha
(R10G10B10A2) cannot hold blended coverage; with FG on The Witcher 3 also clears such a scene
target, which otherwise became the active layer and claimed the whole frame. Clearing requires the render-target state on
every API, so a target can be copied just before its clear: the copy shows the previous frame.

A target cleared in at least three frames with gaps under 250 ms is a confirmed layer; of several,
the one cleared last in the frame is active, since UI draws last. Transient targets never displace
a confirmed one. Only the foreground swapchain's Presents drive the tracker. The tracker runs only
while the layer is wanted: UI detection asked for it within the last second, or a dump is armed
(its census, which can defer the dump, and so keep the layer wanted, for up to 1000 ms while its
copies are not yet proven complete; see [census ordering](#dump-3d-diagnostics)). Otherwise every
clear returns before it resolves its view or takes a lock, and
Presents skip the output bookkeeping (**Present-thread bookkeeping** above); once wanted again the
layer is confirmed anew after three cleared frames. Before WP1b clears and Presents were tracked
with no reader. While UI detection
asks for the layer (within the last second), the active layer is copied once per Present, at its
first clear, into a persistent add-on texture; otherwise no copy is recorded into the game's
frame. Auto offers the newest executed copy while the newest recorded one is under 250 ms old and
only when the offered copy was recorded at most 250 ms before it (`ui_layer::recent_copy`): after
a gap in clears or in demand (a Game 3D or UI-source toggle keeps the ring for 10 s) the first new
copy is still held for its fence, and until the final review the copy from before the gap was
offered as this frame's layer meanwhile. It is offered as a candidate of its own (candidate
bit `0x40`, deciding as source 10) beside any tagged UIColorAndAlpha; the two never share a slot
or an acceptance (E1). Until S4 one layer, the tracker's active one, is offered. Copies are released through their own device, at the latest when it is
destroyed. It is one frame late, and its UI may have moved since, but its
consumed mask is its raw alpha, exactly like every other source; the **UI pin band** (below) pins
only finite positive alpha. UI that moves in the layer can therefore show a one-frame edge or tear.
A motion margin, the alpha dilated by 6 texels in rows and in columns, was removed on 2026-10-02
after a live Stellar Blade test: its 13x13 window flattened a band of scene about 13 texels wide
around thin UI strokes in both axes, most visibly around horizontal lines. Replaying Stellar
Blade's equipment menu, its HUD and The Witcher 3's notice board with both shaders
(`--pin-metrics` below, glyphs moved against the unmoved layer), the raw alpha flattened 17%, 47%
and 29% less scene than the margin and less than the binary pin on all three. Its scene tear
stayed below the binary pin's (286, 273 and 392 against 467, 576 and 540), though on the menu it
rose from the margin's 206. The cost is torn glyphs. Moves of 1 to 4 rows or 4 columns, which the
margin tore not at all, tore about 1.2 to 1.4 times the binary pin on the menu, up to about 600
times on the HUD, whose faint halos the binary pin had pinned fully, and up to 429 pixels on the
notice board, where the binary pin tore 0 to 95. Moves of 8 to 16 texels tore about 1.1 to 11
times the binary pin. These are rigid shifts of the whole layer, so real exposure is limited to
UI that animates, such as scrolling lists, slide-ins and compasses; how often the tear is visible
in play is unmeasured. Tagged sources belong to the presented frame and never had a margin. Detection,
coverage and acceptance read the same raw alpha. V1 admits it only while premultiplied: a pixel whose
color exceeds twice its alpha by more than 4/255 (for a float layer, 125 times its alpha, up to 10000
nits) lies beyond the bound. Such pixels are not UI: they are neither covered nor opaque, and when
they lie on more than 1% of the frame they invalidate the whole layer for the frame, as a scene image
drawn into the target does, unless they lie on at most 5% of the frame and the layer's opaque
pixels (alpha of at least 254/255 within the bound) outnumber them, as a HUD or menu outnumbers an
additive glow strip beside it. Before selection revision 10 more than 1% of the frame beyond the
bound invalidated the layer outright, so such a glow strip would have disabled layer protection
entirely; the GPU now counts them per tile (statistics rows 208-223) and the reduce reports them in
the layer's invalid pixels (decision texel 7 `.y`) only when they invalidate it. They are weighed
against opaque pixels rather than covered ones because a scene buffer's luma-like alpha covers much
of a dark frame faintly within the bound: weighed against covered pixels (as WP1a first did), two
of the replay's synthetic Dead Space layers (below) stayed valid, the dark corridor with 2.7% of the
frame beyond the bound beside 95.8% covered and the pause menu with 20% beside 30%, and neither has
an opaque pixel. The 5% cap bounds the tolerance whatever the opaque count, so a bright scene image
whose near-white pixels count as opaque still fails once its saturated pixels pass 5% of the frame.
UI blended over transparent black is premultiplied by construction; the factor two
admits UI tinted brighter than white. Stellar Blade's real UI layer
failed the former limit (alpha plus 4/255) in most samples: its notification dots sat 5-9/255 above
their nearly opaque alpha, and GPU samples during pulsing markers counted 1,306 and 5,377 such
pixels, so the layer could not earn acceptance during play. Every UI layer copy in the replay dumps
(Stellar Blade HDR, The Witcher 3, Resident Evil Requiem) has at most 0.02% of its pixels beyond the
bound. A scene buffer is not premultiplied: Dead Space's only qualifying target is a post-upscale
scene buffer whose luma-like alpha (82% of pixels nonzero, mean 11/255) lies below its saturated
colors. That buffer was seen in a census dump that is not among the replay dumps, and its rejection
has not been observed live. The replay's synthetic Dead Space cases stand in for it: BGRA8 layers built from two
presented Dead Space frames (a dark corridor and the pause menu), the 8-bit sRGB encoding of the
color at a 203- or 80-nit white with BT.709 luma of the encoded or linear color as alpha, scaled to a
mean of 11/255, which V1 rejects with 2.7-81% of the frame beyond the bound (synthetic dumps
`synthetic_deadspace_*_luma_layer`). A mostly desaturated scene buffer stays within the bound
whatever the rule: the pause menu's encoded luma at a 203-nit white has 0.45% of its pixels beyond
it and 50% covered, so V1 cannot tell it from UI, and only its acceptance (A1) stands between such
a buffer and a decision, a limit of V1 that the rule before selection revision 10 had too (its
saturated pixels lie on less than 1% of the frame). The glow tolerance has a synthetic case too: The
Witcher 3's notice-board layer with an additive strip (alpha 0) on 2.0% of the frame beside 2.7%
opaque UI stays valid and decides its own alpha (`synthetic_w3_notice_board_glow_strip`).

A layer without alpha is no layer. Premultiplied UI over transparent black cannot have color without
alpha, so a layer copy with no alpha anywhere but color beyond its alpha on more than 1% of pixels
(then outnumbering its opaque pixels, of which it has none) fails V1's premultiplied bound like any
other invalid layer: it neither decides nor blocks, and
makes no full claim of its own; since fix 1 a layer without coverage whose signature gameplay proved
equal to the presented frame is the pre-UI scene image of the hidden-scene guard (claim (d) under
**Hidden-scene evidence**). Since S1 this needs no rule of
its own: an invalid candidate never blocks another, and the layer, an inferred source, never blocks
declared alpha (the replay's Stellar Blade SDR cases decide the tag's 0.26% beside it). Stellar
Blade (Unreal) clears the same BGRA8 output-size target to (0, 0, 0, 0) every frame in both output
modes. In HDR it holds the UI (HUD icons, alpha on 0.18-0.71% of pixels); in SDR it holds the scene
image without UI: alpha 0 everywhere and color on 56-99% of pixels, in gameplay and in the settings
menu. Before S1 a trusted layer there blocked presented alpha on every frame, so SDR with FG on had
no mask although the tagged Backbuffer's alpha marked exactly the UI (0.26% of pixels, the same
pixels as the tagged UIColorAndAlpha, which in SDR is real UI). Now the tag decides there once its
first selective sample accepts it, and the Backbuffer, accepted under its own SDR signature, decides
on Presents without the tag. An HDR Backbuffer is opaque and never selective, so it is never
accepted and cannot flatten the scene beside a color-only HDR layer. Resident Evil Requiem with FG
off offers an accepted tag at coverage 0 with more than 1% invalid pixels beside presented alpha
covering 64-80%, which must not decide: the declared-alpha block keeps it out. The layer copy is
still taken in SDR, at the same cost as in HDR, and the rule needs no state: the first copy with
valid alpha is a layer again. A real layer whose only UI is additive color without alpha on more
than 1% of pixels would be invalid for that frame; this breaks the premultiplied contract and has
not been observed. Additive color on at most 5% of the frame beside opaque UI that outnumbers it
keeps the layer valid, with that color uncovered. A layer earns acceptance like any inferred source: selective coverage during
play, after which a sign wheel that dims the whole scene or a full-screen menu covers everything and
stays flat. That holds for any uniform dim at or above 1/8; a fainter dim flattens the scene only in
proportion. Games that draw UI straight onto the back buffer have no layer and are unchanged.

The D3D12 adapter captures Streamline UIAlpha (69, red), UIColorAndAlpha (23, alpha),
Backbuffer (53, alpha) and HUDLessColor (2, RGB) independently of FG being enabled. A fresh
successful tag establishes its epoch, observation revision and viewport even when no FG
options were observed. Each source has at most two snapshot reservations: latest ready and
pending. The shared native owner also bounds its 32-slot auxiliary pool, whose live snapshots and
Dump 3D copies keep separate byte budgets ([Streamline depth selection](#streamline-depth-selection)). A full or
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
pairing. Without a same-batch Backbuffer, counting proposes the presented color of the Present
after the tag for a snapshot's first offer (`ui_mask::pair_hudless_present`): a current source
presentation generation one beyond the generation captured at the tag. The integrated depth Present
callback advances that generation before the Game 3D render. Without frame generation a merely
recent snapshot first seen later does not pass unless it is a late capture (below). The capture owner offers the newest ready
snapshot of its kind for as long as it is recent (250 ms), so one snapshot is offered on every
Present until the next tag replaces it. The provider remembers, per runtime, the snapshot (ticket,
epoch and viewport) it offered last. Under frame generation, since 10-05, every later offer of the
same snapshot pairs with its own Present's color (`ui_mask::pair_hudless_offer`, `reoffered`,
presents ago 0), and so does a first offer that arrives after the Present following its tag: DLSS-G
presents its generated frames first, so the Present after the tag is a generated one and the real
frame comes last in the snapshot's window, and only the current color can be the real frame's.
Before, every later offer compared the color retained from the Present after the tag, the
interpolated one, so the real frame was never compared, and at 4x the last offers were too old to
pair at all; and until the WP1a review a late first offer was compared with retained color too, so
at 2x a capture one Present late reached only the real Present and compared it with the generated
one, never a valid pair. This assumes the generated-first order: where the real frame is presented
first, a late capture never compares it. Without frame generation every offer counts from its tag,
a re-offer included: the Present after the tag, the snapshot's own frame, now or through retained
color (below), and a re-offer more than two Presents after its tag is unpaired and not offered.
Until the review an FG-off re-offer was compared with the current, newer frame, a pair of two frames
that V2 can pass at low motion (a still camera over animated foliage), so one frame's mask could pin
scene patches. Such pairs are never exact, with frame generation on or off (E2): a game may tag its
next frame before the Present counted for it, and frame generation presents interpolated color
between real frames. Only V2's test of the pixels validates them: the real frame's own Present compares equal but for the
UI, and every other Present of the window is mispaired, which the temporal hold covers
(**Temporal hold**, below, the re-offer bit). A full change set from them is never full-screen UI,
and they never judge (A2). The HUD-less image is offered on every Present it pairs, so such a
Present is real to the temporal hold. The capture gate line counts the pairings per interval
(`hudless_presents={batch real late generated=0 reoffered stale other none}`: same batch, first
offer on time, first offer late, always 0, later offers, too old, unpaired, none). The pairing reads no
frame generation multiplier, only whether frame generation is active. Until 10-05 it expected the real frame `generated_frames + 1` Presents
after the tag, held the Presents before it as generated and counted pairs exact with FG off.
Hogwarts Legacy on 10-05 (Epic launch, DLSS-G 4x multi-frame generation) tagged only HUDLessColor,
with no Backbuffer and no UI color: with FG off, counted pairs that belonged to another frame changed
about every pixel, decided an exact full change set and showed whole seconds of gameplay flat
(source 6) while D read the scene visible (0.45-0.65); with FG on the game tagged its next frame
before the counted real Present, so every Present read as generated and held a decision no real
frame had made, 49 s without a UI mask.
A capture from another queue can still be incomplete or unretired when its Present comes. Without
frame generation, when the newest ready capture belongs to a Present one or two Presents ago, it is
compared with that Present's retained color, as a re-offer is, and the resulting mask protects the
current frame. The renderer starts
retaining the last two presented colors (two source-size copies, one extra copy per Present) only
after the first such late capture or re-offer, and stops copying 120 Presents after the last one;
games whose captures pair on time never allocate them. A late
capture or re-offer older than the retained history does not pair. Under frame generation nothing
pairs with retained color: a late first offer and every later offer pair with their own Presents.
Hogwarts Legacy on 09-30 (SL 2.6.10, DLSS-G on) showed the pattern: its real frame matched
HUDLessColor on 93.5% of pixels two Presents after the tag, so with frame generation the counted
Present shows another frame as often as not, and its UIColorAndAlpha was the opaque final image.
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

**Temporal hold (T1).** A Present without its own fresh decision uses the last real decision,
and a real frame without a decision of its own reuses the previous real frame's decision once;
after that there is no mask. There is no multiplier constant and no time bound. Present counting
tells real from generated Presents: under FG, a Present that offers no input within the reported
generated count (one while it is unknown) of the last Present that offered a Streamline UI tag is
generated (`ui_mask::generated_without_input`). The capture owner, however, offers each kind's
newest ready snapshot on every Present while it is recent (250 ms), and the offscreen UI layer its
newest copy, so a generated Present normally re-offers its real frame's inputs instead of offering
nothing: Expedition 33's Backbuffer, The Witcher 3's UIAlpha and Stellar Blade's layer and tags at
2x-4x. Under FG a Present that offers exactly the snapshot identities the last detecting Present
offered (the UIAlpha, UI color and Backbuffer tickets, an exact HUD-less pair's ticket and the
layer copy's id, at least one) is therefore generated too (`ui_mask::same_snapshot_inputs`) and
does not advance Present counting; current colour and an inexact HUD-less image are a Present's
own and never match. Any other Present that offers an input of its own is real, a HUD-less image
(offered on every Present it pairs) or a new layer copy included. Until the 10-05 review only
Present counting held, which fires only once snapshots expire, so every re-offering generated
Present reran the full detection (tiles, reduce, mask and samples) on its real frame's inputs.
A Present carries no real-frame id. Until 10-05 the HUD-less tag's present generation was one, and the HUD-less
pairing classified Presents as generated, which held every Present of Hogwarts Legacy's 4x frame
generation (above). `tools/reshade/game3d_ui_temporal.h` owns the arbiter and the call order that
the renderer and the sequence replay follow:

- A generated Present never detects and takes no sample. It applies the mask of the last real
  decision unchanged (`held.generated`) when that decision was made in the same identity scope
  (epoch and viewport: a recreated swapchain or device, or another viewport). Otherwise it has no
  mask (`held.none`) and the chain ends. An observation revision alone (a depth observation loss)
  is a missing input, not another identity, so it keeps the chain. A misreported multiplier
  therefore fails safe: a count reported too high holds only Presents that offer nothing, one
  reported too low leaves the later generated Presents to the grace below, and 6x frame generation
  holds every generated Present.
- A real Present always detects from its own offered candidates, accepted bits and layer flags.
  The CPU pushes `0x200000` when this chain has no previous real decision (the first detection, an
  identity scope change, an inactive frame or a `held.none` Present since), else `0x100000` when an accepted
  candidate that the last adopting real frame offered is missing now. A frame flagged `0x100000`
  adopts nothing: the status key and the reference for missing candidates keep the last adopting
  real frame's inputs. While an accepted candidate stays missing,
  every later real frame is flagged too, so those keys stay on the old inputs until it returns or
  the chain resets, and each frame without its own decision after one with it reuses once.
- The GPU applies the grace exactly per frame. The reduce computes the frame's own decision (S1)
  unchanged; the frame has none of its own when it decided no source while `0x100000` is pushed or
  an offered, accepted candidate is invalid (V1, or V2 for the pair). A three-texel `R32_UINT` hold
  store, `SunshineUIHoldStore` at `u5`, bound for the reduce only, keeps {state, applied source,
  applied covered}, the state 0 none, 1 own or 2 spent. A frame without its own decision directly
  after one with its own (state own, `0x200000` not pushed) applies the stored decision and marks
  the store spent: decision texel 9 bit 16 (`reused`), and the mask pass writes nothing, so the
  previous real frame's resolved mask stays. Any other frame without its own decision applies none
  and marks the store spent; a frame with its own decision applies and stores it.
- HUD-less re-offers (selection revision 10). The CPU pushes `0x8000` when the offered inexact
  HUD-less snapshot is the one the previous render offered, whatever tags come with it: an
  accepted, valid UIAlpha, UI color or Backbuffer tag decides the frame on its own, and an
  unaccepted or invalid one cannot, so it must not cost the held decision (before the final review
  any offered tag cleared the bit, so an unaccepted tag offered on every Present, such as an opaque
  final image tagged as UI color, brought the window flicker back). Such a re-offer without a decision of its own (a generated Present's
  interpolated pair, V2-invalid; with FG off a re-offer pairs with its own retained frame, and is not
  offered once that is too old) leaves
  the store as it is and applies the stored decision while one is held (state own, or spent with a
  stored source: the decision an earlier frame of the snapshot reused); `0x200000` still blocks all
  reuse. So every Present of a snapshot's window shows the last real decision; before, at 3x or 4x
  the first generated Present spent the grace and the rest of the window had no mask, a HUD flicker in a
  game that tags only HUDLessColor (Hogwarts Legacy) and in one that tags UIAlpha on real Presents
  only beside a HUD-less image offered on all of them (The Witcher 3). The held decision lasts only
  while the same snapshot is re-offered: the next snapshot's first offer without a decision of its
  own reuses it at most once more, as any frame.
- A real Present that offers no candidate but carries `0x100000` runs the reduce and mask passes
  without the tiles pass and never submits a sample, so the grace can apply: a real frame whose tag
  is missing, or, with a multiplier reported too low, a generated Present beyond the count. Any
  other Present without candidates stays inactive.

This replaces S1's three hold kinds (a generated Present, an inexact pair after an exact one, an
accepted candidate missing) and their cap of three consecutive Presents. After an exact full
change-set (6) the first real frame that pairs inexactly reuses it and the next has no mask
(`difference_failed`: the inexact HUD-less image that changed nearly everywhere is a pre-UI claim,
which acts only under the hidden-scene guard's pre-UI hold); in Expedition 33, 2-10% of menu frames paired inexactly and flipped the menu between flat and 3D. An
observation loss refuses the previous revision's captures until the game tags again: generated
Presents keep holding, and the grace covers the first real frame; Resident Evil Requiem's single
rejected-tag frames keep the previous mask. With 3x or 4x FG and no HUD-less pairing, as in
Expedition 33, every generated Present within the reported count holds, where S1 held at most
three; a count reported too low leaves the later generated Presents to the grace (the first reuses
once, the rest have no mask), a known limit of Present counting since S3's real-frame identity was
removed (selection revision 9). A generated Present that re-offers the last completed snapshot
detects from it as a real Present: its own pair is compared with its own color (above), and when
that pair is mispaired the re-offer bit keeps the held decision.

The GPU compares native RGB in matching color space. A changed pixel exceeds 4/1023 for
R10 color or 2/255 for other SDR/PQ color; floating-point scRGB uses 0.005 times the larger
of one and the current pixel's largest absolute RGB component. Both RGB inputs must be finite.
Acceptance requires a nonempty difference covering fewer than 25% of pixels, at least 75% of pixels within half the
threshold, and at least 128 of 256 tiles with 99% matching pixels. These guards reject broad
scene mismatch before the difference becomes a UI mask. Since selection revision 8 a lit pair (at
least half of the HUD-less pixels above eight times the threshold) without any changed pixel, in
at least 128 such tiles, exact or not, is a valid empty change set: no UI is on screen, so an
accepted pair decides an empty mask of its own (source 5, covered 0), as an accepted alpha deciding
coverage 0 does, instead of being invalid and reusing the previous real frame's mask once under T1.
It earns nothing, runs a restored pair's reconfirm clock and, exact, judges (A2). One exception
covers full-screen UI: when the pair is exact (a same-batch Backbuffer), at least 98% of pixels
changed and at least half of the HUD-less pixels are lit (above eight times the threshold), the
whole frame is UI and stays flat. Hogwarts Legacy's title screen needed it:
the game keeps rendering its next 3D scene behind the menu and tags that scene's depth and
HUD-less image, so 99.9% of pixels differed while the depth matched the hidden room. A mismatched
pair can also change every pixel, so an unverified pair is still rejected (directly after an exact
decision the T1 grace reuses it once), and a black HUD-less image never counts as a scene. A pair
by Present counting is never exact, with frame generation off too: on 10-05 Hogwarts Legacy's
FG-off counted pairs decided gameplay as source 6 while they were counted exact. The full-screen menu of a game
that tags only HUDLessColor is flat through H1 (d) instead: its HUD-less image, changed on at least
90% of pixels, is the pre-UI scene image, which acts once the hidden-scene guard holds the presented
frame hidden and that image visible (**Hidden-scene evidence**, below). HUD-less comparison remains guarded in manual modes. Its binary mask identifies changed pixels, not exact compositing opacity, and cannot detect
UI whose color matches the underlying scene. Local motion or other postprocessing differences
can still resemble UI; live game/headset validation remains necessary.

**Hidden-scene evidence.** Some full-screen UI hides a 3D scene that the game keeps rendering and
whose depth it keeps tagging, without any input the rules above accept as full-frame UI. Stellar
Blade draws its notice and SHIFT UP splashes into its offscreen UI layer, fully opaque over the next
scene; in a first session the layer is not yet accepted, and an unaccepted source covering the
frame is ambiguous. Hogwarts Legacy's title screen pairs its HUD-less image only by Present
counting when it tags no same-batch Backbuffer (with FG on on 09-30, and in every mode on 10-05), so
the exact-pair rule cannot apply. Stellar Blade in SDR suspends frame generation while
a menu is open, so its settings page has no Streamline tag at all: its only candidates are the
cleared output target, which holds the scene without UI (a layer without alpha, below), and opaque
presented alpha. In all three the presented frame lacks the consumed depth's edges, which is what
this check measures: the depth describes another image than the one shown, so the frame is UI over
a hidden scene and is shown flat (rule H1 of the [UI decision framework](#ui-decision-framework)).

The statistic D compares an image with the consumed depth on a frame-relative grid of 256 x 144
cells: output pixel x lies in cell column
`floor(x * 256 / width)` and row y in cell row `floor(y * 144 / height)` (15 pixels at 4K, 10 at
1440p, 7.5 at 1080p). Each cell holds the area mean of perceptual-code luma (PQ code for HDR10, read
before PQ linearization; sRGB code for SDR; PQ of luminance for scRGB) and of the strength-1
parallax, decoded as the scene candidate pass decodes it but independent of the strength and blend
controls, clamped at the disparity limit, from depth texel `floor(uv * size)`. A cell is a depth
edge when its mean parallax steps by at least 4 pixels per 2160 output rows to the cell on its
right or below; its activity is the same right or lower step of its mean luma. Each edge cell's
activity is compared with the activity at four null cells, offset by (5, 3), (-7, 4), (8, -2) and
(-4, -6) cells and wrapped around the grid: D = (wins - losses) / (4 n) over n edge cells, ties
counting 0, so a black or flat image reads 0. D is 0 in expectation for unrelated colour and depth
and near 1 when the colour shows the depth's silhouettes. Evidence is valid with at least 128 edge
cells, depth and camera ready, depth captured for this frame (not reused or behind a generated
Present) and a frame of at least 256 x 144 pixels, so that no cell is empty. Valid evidence reads
hidden below 0.15, visible from 0.25 and ambiguous between. Sums are fixed point (2^-20 luma,
2^-12 pixel) and cell means round half away from zero, so the GPU result does not depend on
summation order and `inspect_game3d_dump.py --scene-evidence` reproduces it. The shader measures
frames up to 3840 x 3840, the renderer's detection bound, within which every cell sum fits 32 bits,
and compares parallax steps with a precomputed integer bound. The evidence also counts the
presented image's decided comparisons, wins plus losses: a black or flat frame decides none.

D is measured on two images. The presented colour gives decision texel 5 with its verdict. The
pre-UI scene image gives texel 6 with the image it measured: the HUD-less image (`t14`) when one is
offered, else the offered offscreen UI layer's RGB (`t7`), else nothing. The evidence passes need no
binding of their own for it, and the CPU derives that image's verdict from its D with the same
bounds. A UI layer with coverage (Stellar Blade in HDR, The Witcher 3) is measured but never acts
through this reading (claim (d) below). A float layer's luma is not encoded as the presented
frame's, which can only make it read flat or hidden, so it fails safe. The layer is the
one-frame-late copy; this coarse statistic on a static menu tolerates it, and the two-sample entry
below absorbs the first menu sample after a cut. On Stellar Blade's SDR dumps the presented frame
and the layer read 0.031 and 0.588 on the settings page with FG suspended, -0.005 and 0.574 on it
with FG off, and within 0.004 of each other in gameplay (0.515 and 0.518, 0.504 and 0.508, 0.376
and 0.375). Hogwarts Legacy's title reads 0.039 presented and 0.556 on its HUD-less image.

The bounds come from the labelled dumps and an offline stress study. Real hidden scenes read at most
0.039 (Stellar Blade splashes -0.044 and -0.056, The Witcher 3 graphics settings -0.022 and -0.006,
Hogwarts Legacy title 0.039, Expedition 33 settings -0.024 and -0.002); matched frames at least
0.260 (Dead Space's dimmed pause; then 0.278 to 0.87). Blur, depth of field, motion blur, grain,
darkening, 8-pixel shifts and scaling of 23 matched frames left all of them at or above 0.161;
unrelated colour and depth pairs across games read at most 0.211 (99th percentile 0.140). The
weakest case is a dark, low-contrast scene under heavy grain, which drifts toward hidden; only an
informative claim below lets that act, entry needs two hidden samples, and a signature that a
visible verdict refuted cannot act. The margin between the hidden bound and the stress minimum
(0.15 against 0.161) is thin; it stays an open question of the
[framework](#ui-decision-framework). Claim (d) needs a second, independent reading to contradict
the presented one: fog, blackout or dark grain lower both images' D together when the pre-UI image
is the presented frame without its UI, so a dark scene reads hidden on both and never enters the
pre-UI hold. The declared HUD-less image is that by contract. An offscreen layer holding colour
without alpha is only inferred to be: a cleared scene buffer from before fog, grade, grain or
vignette (Dead Space's only qualifying target is a post-upscale scene buffer) would keep reading
visible while the presented frame drifts hidden, which would look like a menu. Such a layer acts only
once its signature is proven, below: gameplay must show it equal to the presented frame, pixel for
pixel at a coarse threshold, on at least 90% of the pixels. Claim (b), such as The Witcher 3's 2-bit scene target
with FG on at the start of a dark session, still depends on the presented reading alone until its
first visible verdict refutes it.
All labelled dumps are 16:9. Cropped to their centre 21:9, 32:9 or 4:3, where the fixed grid's
cells are 1.31, 1.78 and 0.75 times as wide as tall, hidden scenes read at most 0.109 and matched
frames at least 0.193, except Dead Space's dimmed pause cropped to 4:3 (0.112): with square cells
the same crop reads 0.111, so the crop removed its scene rather than the cells' shape changing the
verdict. Real output in other aspect ratios is unvalidated.

**Informative full claims.** The detection reduce computes from each frame's own counts which valid
candidates claim the whole frame. Opaque-full means alpha of at least 254/255 on at least 99% of
pixels: 254/255 is one 8-bit code below opaque, so an 8-bit layer must be fully opaque and a float
layer blended to within that of 1 still counts, and 99% is the share at which the UI counters treat
alpha as covering the whole frame (`full_alpha`). A claim is informative when it is:

- (a) an offered, V1-valid, accepted alpha candidate that is opaque-full and that S1 may select
  (any of UIAlpha, the UI color tag, the layer, Backbuffer and current alpha, except an inferred alpha
  that an offered, accepted declared alpha keeps out of S1): Resident Evil Requiem's presented alpha,
  opaque everywhere beside its accepted UI color tag, never claims, so its dark rooms never depend on D
  alone;
- (b) the offered offscreen UI layer when V1-valid and opaque-full, accepted or not: every offered
  layer is a target the tracker confirmed cleared to transparent black every frame, which stands for
  proof that it was cleared this frame until S4;
- (c) an exact full change-set (exact, at least 98% changed, lit; the full-frame exception above),
  accepted or not;
- (d) a pre-UI scene image, claim bit `0x80`: the HUD-less image when it changed on at least 90% of
  pixels, exact or not, or, without a HUD-less image, an offered layer without coverage whose
  signature is proven the pre-UI scene image (per-frame bit `0x80000000`, pushed from the ledger's
  `pre_ui` key, below: the scene image Stellar Blade's SDR target holds). 90% is the share from which
  a sample is no longer selective (A1), and leaves room for parts of a full-screen menu that do not
  change. Since fix 1 the layer's claim no longer depends on V1: an unproven layer, or one with any
  covered pixel, makes no claim (d).

An unaccepted UIAlpha, UI color tag, Backbuffer or current alpha is never informative, whatever its
opacity: Hogwarts Legacy and The Witcher 3 present always-opaque alpha, and the tagged
UIColorAndAlpha of Hogwarts Legacy is the opaque final image. Before S2b an unaccepted opaque UIAlpha
opened the layer route; a first-session full-screen menu drawn only into such a UIAlpha now stays 3D
until the UIAlpha earns acceptance from one selective sample (no recorded case depends on it). The
layer's claim (b) uses V1 validity (at most 1% invalid pixels), where the old layer route needed
none. Claims (a)-(c) act unless their signature is refuted (below); claim (d) acts only while the
guard's held samples read the pre-UI image visible, and for the layer only while it is proven.

**H1.** When the CPU holds a hidden verdict (per-frame bit `0x400000`) and some claim acts, the
frame is full-frame UI over a hidden scene:
source 8, covered pixels all, and the mask pass writes 1.0, as for source 6. It applies whatever S1
selected, an accepted partial winner included: an accepted HUD tag deciding 0.25% while a full menu
covers the scene gives way to the flat frame for as long as the hold lasts. An opaque-full winner
with transparent pixels would leave them to P1, which warps them with the hidden scene's depth, so
H1 takes it over: Expedition 33's accepted Load Game Backbuffer (99.9998% opaque) decides 3 until
the guard holds its hidden verdict and 8 from then on, flat throughout (replay case `E33 load game
FG on, trusted backbuffer, hold`). Since selection revision 10 a winner already flat on every pixel
(source 6, or an alpha winner opaque on every pixel) is relabelled 8 too: P1 pins it at weight 1
either way, so only its label and counters change (`decided.8` and `full_d` instead of `decided.6`
or the alpha source and `full_alpha`), and H1 needs no winner test (replay case `E33 settings FG on,
exact full-frame pair under a hold, relabelled 8 by H1`, which decided 6 before). A render without
depth has no parallax and is flat anyway, so H1 does not test for depth. Reused or generated depth
(`0x40000`) keeps H1 active: the held verdict comes only from samples with current depth and lasts
at most 500 ms, reused depth comes from the same source, and flat is the fail-safe direction. Until
the 10-05 review reused depth disabled H1, so under frame generation every generated Present that
re-offered the real frame's inputs fell back to its own S1 decision (often none) and a full menu
over a hidden scene alternated flat and warped at the Present rate. The A2 and T1 counts, the hold store and the counters follow
the final decision, so an overridden winner counts as decided 8 and never as contradicted. The
acceptance ledger reads the candidates' own counts, so H1 neither earns nor revokes acceptance, and
the status names a detected UI source while it decides 8. A claim that acts while H1 does not apply
gives the frame's reason `gate_no_hold` and refuses its first claimant in draw order, or for the
pre-UI claim alone the HUD-less image when offered, else the layer. A pre-UI image without the
pre-UI hold does not act, so Stellar Blade's SDR gameplay keeps `layer_aside`, and an inexact
HUD-less image that changed nearly everywhere reads `difference_failed`.

**The hidden-scene guard.** The CPU half of H1 (M5) is `tools/reshade/game3d_scene_guard.h`, a pure
header that reads only decision words and signatures, never acceptance, slots, layer flags or the
T1 hold, so the renderer, `ui_detection_replay` and the sequence replay drive it alike. The one
ledger fact it uses, whether the offered layer's signature is proven (below), its caller passes in
as an argument on every frame; the guard keeps no proof. It holds two verdicts of 500 ms each:

- **Hidden.** Two valid hidden samples of the presented frame within 500 ms enter it, each further
  one renews it to its tick plus 500 ms, and one valid visible sample releases it. An ambiguous
  sample breaks an entry run without releasing a held verdict; invalid evidence changes nothing.
  Once held, every valid hidden sample renews it, with or without a claim, so it lasts as long as the
  presented frame reads hidden; nothing flattens without an acting claim.
- **Pre-UI.** On the samples that hit the hidden verdict and carry claim (d), the pre-UI image's
  valid D hits this hold when it reads visible and releases it otherwise, with the same two-sample
  entry, and only for the image the claim names; the layer claims only while it is proven. A dark
  scene reads hidden in both images and never enters it. The renderer pushes the held pre-UI bit
  only while the offer's pre-UI image is the HUD-less image or a proven layer of the offered
  signature.

**The layer's pre-UI proof.** Since fix 1 an offscreen layer is proven the pre-UI scene image by
its pixels, and the proof is a ledger entry, not guard state: the acceptance ledger
(`game3d_alpha_auto.h`) keeps it under the key `pre_ui:<layer format>:<colour space>`, the
`ui_layer` signature with the ledger-only kind `pre_ui` (`ui_selection::pre_ui_key`). The detection
tiles pass compares the offered layer's RGB (`t7`) with the presented colour (`t6`) at eight times
their pair threshold, the coarse bound of the lit tests (`b2` word 4, `ui_selection::comparable` of
the layer signature's encoding and the presented one; 0 without a layer or a comparable pair, which
counts nothing), with scRGB differences relative above one as for the HUD-less difference. Decision
texel 11 sums the pixels that match and the lit layer pixels (above the same bound). A sample
matches when the layer is offered without coverage, matches on at least 90% of the pixels
(`ui_selection::full`) and is lit on at least half (`ui_selection::pre_ui_match`): the half-frame
rule of a lit HUD-less image (`change_set_full`), without which a transparent black UI layer over a
black loading screen, equal everywhere, would be proven. Three matching samples spanning 2 s
(`alpha_trust_samples`, `alpha_trust_span_ms`, as an inferred source earns acceptance) prove the
signature. They need not be consecutive, and a mismatch neither withdraws the proof nor restarts
the run: UI over the scene is exactly what a mismatch looks like. The proof needs no scene evidence
and no evidence pass, and in FG-on gameplay, where the HUD-less image is texel 6's image, texel 11
still compares the layer.

The proof survives what clears the guard. An identity change (epoch or viewport), an FG toggle and an
acceptance change leave it; a different layer format or colour space is another key. It is stored
in `TrustedUISources` with the accepted sources, restored provisionally at start and confirmed by
three matching samples over 2 s; it lapses after 60 s (`alpha_trust_reconfirm_ms`) of testable
time without them. Testable samples offer the layer without coverage while the presented frame's
evidence is valid and reads visible, and the clock counts the capped gaps between them as above;
every other sample (a menu, invalid evidence, no layer, or another signature) never counts beyond
one capped gap, so a menu opened at start keeps the restored key. Forget
clears it. Manual modes neither earn nor lapse it but honour it. The renderer asks the ledger
(`alpha_auto_policy::pre_ui_proven` of the offered layer's signature) on every frame, after the
poll on a detecting frame so that a sample that just earned the proof counts at once, pushes
`0x80000000` while it is proven and passes the answer to the guard's `per_frame` and `measure`.
Acceptance lines list the key (`accepted UI sources are now ...,pre_ui:87:srgb`), the `trust.*`
counters count it with the other keys, and the log reports the answer as `proven` in
`scene_guard`.

The numbers that fix the rule come from Stellar Blade's dumps, over all 8,294,400 pixels at eight
times each pair threshold (16/255 for the SDR `R10G10B10A2` presented frame and `B8G8R8A8` layer),
and the GPU counts equal a numpy oracle: gameplay matches on 99.79%, 99.82% and 99.32% of pixels with the layer lit on 88.9%, 88.8%
and 90.4% (the layer is the one-frame-late copy, so camera motion costs a little), the settings page
on 32.79% (FG off) and 35.28% (FG suspended) with the layer lit on 25.9% and 26.0%, and the HDR
layer, a real UI layer with 14,519 covered pixels, on 1.57%. The rule is SDR-specific for Stellar
Blade: in HDR the same cleared target is a premultiplied UI layer (`ui_layer:87:pq`) whose own alpha
decides its menus, and its coverage and its mismatch keep it from ever being proven. The HDR side
rests on those dumps and a session log from before fix 1, not a live run of it.

The proof replaced S2b's D similarity, which never held where it was needed. S2b proved a layer when
a valid sample read the presented frame visible or ambiguous and the two D within 0.03 of each
other, withdrew it when they differed by more, and cleared it with the guard on an identity change.
In a Stellar Blade SDR session with FG at 2x, FG-on gameplay proved the layer (layer D 0.358 and
0.384 against presented 0.364 and 0.405), but opening a menu suspends FG, which moves depth from
Streamline (viewport 1) to NGX (viewport 0): that identity change cleared the proof, and a fade-in
sample (presented D 0.453, visible, against the layer's 0.538) withdrew it again. The settings page
then read hidden (presented D -0.001 and -0.002) with the layer visible (0.544 and 0.569) and claim
`0x80` on every sample, but unproven, so H1 never applied. Per-frame bit `0x20000000`, which
measured the layer as texel 6's image beside a HUD-less image for that proof, is retired.

Cases that stay 3D or unprotected by design: a first session that opens a menu before 2 s of lit
gameplay (from the second session the restored key covers it); gameplay in which the layer stays lit
on less than half of the pixels, which cannot earn; a cut straight from proven gameplay into a scene
where effects after the layer push the presented frame hidden without an ambiguous or visible sample
in between (indistinguishable by D from a menu opening; it holds until the first visible sample);
and a target that is not the final pre-UI image but equals the presented frame on 90% of the pixels
at the coarse bound (a late post-process input), which would be proven. The two-sample entry, release
on one visible sample, the provisional lapse and Forget bound them.

**Shadow statistics for a future dark pre-UI image rule: removed.** Selection revisions 4 to 8
also counted in texel 11 the lit presented pixels and those of them that differ from the layer at
the same bound, which nothing acted on. They measured screens that no claim covers, such as Stellar
Blade's SDR loading screen: the layer without coverage is an almost black scene image (invalid on
only 0.2-0.4% of pixels, so V1-valid), the presented frame reads hidden and the layer only weakly
(D 0.15-0.22), so H1 (d) does not apply and it stays 3D. Selection revision 9 writes texel 11's
`.z` and `.w` as reserved zeros, and the UI log line keeps `presented_lit=0` and
`presented_lit_differs=0` in place.

A visible sample also refutes the signature (`<kind>:<format>:<colour space>`, as for acceptance) of
every candidate whose full claim it carried, since the presented frame then shows the depth's edges
through it. A refuted signature does not act until an in-scope sample offers that candidate valid
and below 99% opaque (an alpha), or as an exact pair without a full change-set (HUD-less), which
shows it to be an overlay. At most eight refutations are kept, the oldest dropping first, and the
renderer pushes the offered refuted candidates as bits 24-30 (candidate bits shifted left by 24). A
source that never reads visible is never refuted, so a dark scene buffer at the start of a session
remains possible. Requiring transparency before a claim may act was rejected: Stellar Blade's splash
layer covered every pixel from its first logged sample of a first session, so its splashes would
have stayed 3D.

Only an identity change clears the guard: another epoch or viewport (a recreated swapchain or
device, or another viewport). An observation revision, an inactive frame, an acceptance change and
Forget clear nothing; a sample of another revision is still discarded unread. The layer's pre-UI
proof is ledger state, so an identity change never clears it; Forget does. After a depth
observation loss a held verdict lasts up to 500 ms; it keeps acting while the depth is reused,
which never renews or releases it, and a new current depth within that window inherits it, since
the screen is still the same menu. The renderer
calls the guard in a fixed order: `enter_scope(epoch, viewport)` on every detecting frame,
`per_frame` (the held bits and refuted candidates) ORed into the pushed flags and `measure` before
detection, and at poll `observe` on the completed sample, with the signatures it was submitted
with, before the acceptance ledger observes it. A generated Present reports the guard's bits but
pushes none.

The evidence passes only measure: cell sums into their own 256 x 144 texture, a 16 x 16-cell
comparison per group into nine statistics rows, and their sum into decision texels 5 and 6. They
run after detection on sample frames only (the 100 ms readback cadence), only over depth that is
this frame's (no `0x40000`: reused depth is no evidence, so the sample stays not actionable), and
only while the latest sample had a claim that could act (a claim not refuted, or claim (d)), a
verdict is held or a proven layer is the offer's pre-UI image (H1 (d)); otherwise nothing is
dispatched. Everything they
measure is therefore actionable. Until selection revision 9 they also ran, never actionably, for
the first-run shadow, as a diagnostic after an accepted whole-frame decision (for the
`full_alpha_d` UI counters) and, in selection revisions 5 to 8, for fix 2's still-screen rule H2;
all three were removed (**The first-run shadow: removed** and **Still screens without a UI source
(H2, fix 2): removed** below). That also ends a leak: non-actionable evidence of a visible scene
reached the acceptance ledger and could run a restored pre-UI proof's reconfirm clock. A proven layer is actionable from its first
sample, so a menu that opens just after an identity change (Stellar Blade suspending FG) enters at
its second hidden sample. Stellar Blade in SDR measures every sample once its layer is proven, since
its target always holds the scene, and so does a game while its presented frame keeps reading
hidden under a held verdict. Before fix 1 a set-aside layer (offered, V1-invalid) ran them, not
actionably, for the D proof. The proof's own pixel counts come from the detection tiles pass, which
then also reads the presented colour wherever a comparable layer is offered; that added read has
not been timed. At 3840 x 2160 on an idle RTX 5080 an evaluated frame cost
0.060-0.077 ms (cells 0.054-0.070 ms, comparison and sum 0.009-0.010 ms; HDR10 and scRGB, with and
without a HUD-less image, the most for an SDR frame with one), once per 100 ms; the detection tiles
and reduce kept their cost.

Entry takes three samples where the claim appears with the menu (one shows the claim, the next two
measure hidden), about 300 ms, and two where the claim is always present, as for Stellar Blade's
proven SDR target with FG off or suspended, about 200 ms: with FG on its gameplay offers the
HUD-less image as the pre-UI image, but the proven layer claims from the first menu sample after FG
is suspended. Release takes one visible sample. Claim (d) on Stellar Blade's SDR target
stays true after its menu closes, so up to one sample interval plus the readback of gameplay
(the sequence replay bounds it at 132 ms) is shown flat before the first gameplay sample releases
the verdict; a per-frame pre-UI difference would close that window and belongs to S5.

**The first-run shadow: removed.** Until selection revision 9 a session could run the evidence
passes on every sample frame whatever the claims and log the result without changing a decision
(the first-run shadow, the per-game `UISceneShadow` key, absent meaning the game's first session),
reporting `shadow_hidden_ms`, the run of consecutive samples reading the presented frame hidden while
no source decided. It found Stellar Blade's SDR settings page with FG suspended hidden for about
9 s per visit, which is flat through claim (d) once its layer is proven since fix 1. It was removed
with its key, its `first-run shadow measures this session` line and its dump fields; the UI log
line keeps `shadow=0 shadow_hidden_ms=0` in place, and a `ReShade.ini` that still carries
`UISceneShadow` loads unchanged with the key ignored. Known first-session gaps stay open by design,
because none has an informative claim: Expedition 33 settings with FG off and The Witcher 3
settings without a UI layer offer only presented alpha, and Expedition 33's Load Game with FG on
only its tagged Backbuffer alpha, which flatten from the second session through remembered
acceptance.
Stellar Blade in HDR with FG on offers its opaque tagged UI colour, never accepted and never
informative, beside its layer, so in a first session, until the layer earns acceptance, only its
layer's claim (b) and its HUD-less image's claim (d) can act.

**Still screens without a UI source (H2, fix 2): removed.** Fix 2 (selection revision 5) added
rule H2: in SDR Auto, a screen on which no UI source decided, that read hidden (D at most 0.05) and
kept 95% of the D grid's cells still for 2 s was shown flat as source 11, behind the per-game
`UIFlattenStillScreens` switch (a shadow that only logged by default). Selection revision 9 removed
it together with S3's frame identity, which shared decision texel 12: the stillness counts and
their statistics rows, the previous-luma texture, the flag in `b2` word 5, the `Sunshine UI still
screen` lines, the `still` counter and log groups, the panel's checkbox and its status texts are
gone. Its identifiers stay reserved and are never reused: source 11 (its counter word stayed zero
through dc7e3c77 and was then dropped), `b2` word 5 (pushed as zero), decision texel 12 (words 48-51) and statistics rows 144-159. A
`ReShade.ini` that still carries `UIFlattenStillScreens` loads unchanged; the add-on ignores the
key. A screen on which no UI source decides and no informative claim acts, such as Stellar Blade's
SDR loading screen, stays 3D.

**Pre-UI change sets (fix 3) and pin only UI (fix 4): removed.** Fix 3 offered the offscreen
layer proven the pre-UI scene image as a change-set provider (candidate `0x100`, source 12, the
`UIPinChangedPixels`/`UIPinOnlyUI` switch, the refine rule and its change-set shadow), and fix 4
added rule P2's darkening passes under the same switch. The user rejected both in SDR and HDR, and
selection revision 7 removed them, so detection decides exactly as revision 5 did. Their
identifiers stay reserved and are never reused: candidate bit `0x100`, source 12, the h1 word's bit
`0x200`, `b2` word 5 bits `0x2`-`0x200`, decision texels 13-15, statistics rows 160-207 and Dump 3D
artifacts 43-45. Source 12's counter word and counter word 30 stayed zero through dc7e3c77 and were
then dropped. A `ReShade.ini` that still
carries `UIPinOnlyUI` or `UIPinChangedPixels` loads unchanged; the add-on ignores the key. Fix 1's
pre-UI proof (the ledger key `pre_ui:<format>:<space>`, texel 11 and claim (d)) is unchanged, so
a remembered proof keeps working. The D3D12 native runtime fixture keeps its `D3D12 pre-UI proof`
section (fix 1's texel 11 proof by 22 gameplay frames, then the Equipment, settings and black
loading pages flat by the accepted presented alpha, under the D3D12 debug layer when
`d3d12SDKLayers` is installed).
The native path uses GPU statistics, reduction/selection and mask passes. Each of the 16x16
statistics tiles is one 256-thread group, the reduction sums the 256 tiles in parallel, and the
mask pass loads only the selected candidate; the integer counts and the mask are unchanged. At 4K
this took UI detection from about 0.148 to 0.12 ms averaged over the provider fixture, which
is mostly bound by reading the candidate textures. A bounded asynchronous 272-byte summary (192
bytes in selection revision 9, 208 in revisions 5 to 8) may be
read at 100 ms intervals for diagnostics; it never authorizes protection and there is no full-frame
CPU readback. Besides the decision, it carries the candidate bits the shader was offered, the
accepted candidates pushed with the detection, the offered candidates that passed V1 or V2, the
covered and invalid pixels of UIAlpha, the UI color tag, Backbuffer and current alpha, the offscreen
UI layer's covered, invalid and nearly opaque pixels, the HUD-less changed, unchanged, non-finite
and lit pixel counts with matching tiles, the nearly opaque pixels of all five alpha kinds,
the one-way judgment counts of UIAlpha, the UI color tag, Backbuffer and current alpha (A2, on
status samples), the frame's own-decision
reason, the candidate it refused and whether the T1 grace reused a decision (F1), the informative
full claims, the S1 winner and whether H1 applied, the offscreen UI layer's pixels against the
presented frame (matching and lit; since fix 1), and, on samples that
measured it, the hidden-scene evidence of the presented frame and the pre-UI scene image. Updating
acceptance and the layer's pre-UI proof from this summary is the only way it feeds back; it never authorizes the frame it describes. Every sample read in the
same scope feeds the hidden-scene guard, the ledger and the counters; only a sample from another
epoch, revision or viewport, or one already stale when it arrives, is discarded unread. The status
shows a sample only while it is fresh (at most 500 ms old), in the current scope, and was taken
under the current winner: the first offered, accepted candidate in draw order (UIAlpha, the UI
color tag, the layer, Backbuffer, current alpha, then the HUD-less pair) of the last adopting real
frame (F1). Otherwise it shows "Checking source quality". Candidates after the winner, unaccepted
ones and a HUD-less pairing may therefore come and go without a new status. Resident Evil Requiem
pairs its HUD-less image on only some Presents with frame generation on; discarding the sample on
each such change reported "Checking source quality" about half the time while its accepted UI color
alpha protected the HUD, and an exact pair accepted with FG off stays accepted with FG on. Stellar
Blade with FG tags its Backbuffer on real Presents only; a key that changed with it discarded every
sample, so generated Presents never saw the latest sample and lost the Backbuffer mask at the FG
rate. Before S2a the status key also held acceptance bits and stored flags, and a change of it
discarded the pending sample. The candidate bits
in the summary are those offered, invalid ones included. The `Sunshine UI protection` log
(`sampled_candidates`, `sampled_alpha_covered` and `sampled_alpha_invalid` for UIAlpha, the UI color
tag, Backbuffer and current, `accepted` as candidate bits, `sampled_layer` with the offscreen UI
layer's `covered`, `invalid` and `opaque` pixels, `sampled_one_way` with the `strong` and
`contradicted` pixels of the layer (0 since selection revision 10, which judges no layer copy),
Backbuffer and current alpha, `sampled_reason` (the own
decision's `ui_no_mask` reason, or `decided`), `sampled_refused` (the refused candidate's kind,
`ui_alpha`, `ui_color`, `ui_layer`, `backbuffer`, `current` or `hudless`, or `none`),
`sampled_reused`, `sampled_hudless`, `sampled_alpha_opaque` for
UIAlpha and the UI color tag, `sampled_inferred_opaque` for Backbuffer and current alpha,
`sampled_claims` (the informative full claims before refutation: candidate bits, `0x80` the pre-UI
scene image), `sampled_h1` with `applied` (H1 overrode the S1 winner with 8) and `winner` (the S1
winner's source), `sampled_scene` with the presented image's `n`, `d`, `valid`, `ran`
and `verdict`, `sampled_pre_ui_scene` with the pre-UI scene image's `image` (`none`, `hudless` or
`layer`), `n`, `d` and `valid`, then
`scene_guard` with `hidden` and `pre_ui` (the verdicts the hidden-scene guard pushed with that
render), `refuted` (the signatures it holds refuted) and `proven` (the offered layer's signature is
proven the pre-UI scene image, the ledger's `pre_ui` key, so that its pre-UI image may act),
`shadow` and `shadow_hidden_ms` (the removed first-run shadow's fields, 0 since selection revision
9), then `sampled_pre_ui_pixels` with decision texel 11's `match` and `image_lit` and the reserved
`presented_lit` and `presented_lit_differs` (0 since selection revision 9; the layer against the presented frame,
**The layer's pre-UI proof**), then (since selection revision 10) `sampled_declared_one_way` with
the `strong` and `contradicted` pixels of UIAlpha and the UI color tag (both 0 for a tag outside
the exact pair's tag batch), and the dump's `source_alpha_auto.sampled_evidence` (`alpha_opaque`, `inferred_opaque`, `claims`, `h1`
with `applied` and `winner`, `layer`, `accepted`, `valid_bits`,
`scene` with the presented image's `decided` comparisons, `pre_ui_scene` with its `image` and
`verdict`, `pre_ui_pixels` with the same four counts, the current run's
`one_way` with `strong` and `contradicted` by judged kind (`ui_alpha`, `ui_color`, `backbuffer`,
`current` since selection revision 10; before it `ui_layer`, `backbuffer`, `current` and
`late_layer`), `reason`, `refused` and
`reused`, with the render's `scene_guard` beside it; dumps before selection revision 9 also have
`shadow_hidden_ms`, `scene_shadow` and the last two `pre_ui_pixels`) report it, so a
rejection names the failing check. `sampled_source` and `sampled_covered` are the applied decision,
which the T1 grace may have reused. Logs before S1 wrote `trusted_alpha` (slot bits) and
`sampled_ui_layer` (the layer in the UI color slot) instead of `accepted` and `sampled_layer`; logs
before S2a have no `sampled_one_way`, `sampled_reason`, `sampled_refused` or `sampled_reused`; logs
from S2a through 180f1842 also have `sampled_late_layer` after `sampled_reused` (a layer was offered:
every layer was the one-frame-late copy, which no A2 judge reads, so the report never judges the
layer whether or not the field is there); logs
before S2b have `sampled_hudless_scene` (the HUD-less image's evidence) and `scene_hold` (1 layer
route, 2 HUD-less route) instead of `sampled_pre_ui_scene` and `scene_guard`, and no
`sampled_inferred_opaque`, `sampled_claims` or `sampled_h1`; logs before fix 1 have no
`sampled_pre_ui_pixels`, their `proven` is the guard's D proof, and in them a V1-invalid layer
claims `0x80` without a proof (since fix 1 an unproven layer adds nothing to `sampled_claims`);
logs of fix 2 to selection revision 8 end with H2's `still` group (**Still screens without a UI
source (H2, fix 2): removed**), and their dumps carry `still_short_ms` and `still_screen`. A
presented verdict entering or leaving hidden logs within a second, like a change of source; turns
between ambiguous and visible, frequent near the visible bound, wait for the next periodic line.
Without a mask the panel's status names the reason and the refused candidate: "No usable UI mask
(learning the real-input alpha)" for an unaccepted Backbuffer (`ui_protection_reason_text` in
`game3d_controls_model.h`).
Source availability, GPU validation and actual applied protection remain separate diagnostic facts.
Older startup fields describe a retired heuristic.
Sampled source 10 is the offscreen UI layer, 8 full-frame UI over a hidden scene (H1,
**Hidden-scene evidence** above). 7, 9 and 11 are retired and never reused: Dump 59540_032 still
records 7, logs and dumps before S2b record 9 for the HUD-less route that H1 replaced, and those of
fix 2 to selection revision 8 may record 11 for H2's still screen.

**UI detection flags and decision texels.** `tools/reshade/game3d_ui_detection_contract.h` names
these values and the decision words for the renderer, Dump 3D and the replay tools;
`game3d_native.hlsl` mirrors the flags as `SUNSHINE_UI_STORED_*` and `SUNSHINE_UI_PER_FRAME_*`
defines, and the `reshade_game3d_ui_layer` unit test fails when the two disagree or the size markers
below leave the supported range. `SUNSHINE_UI_CANDIDATE_LAYOUT` 2 (S1, E1) gives every candidate
its own slot and bit in `Sunshine_UICandidates` (`b2` word 0, offered) and
`Sunshine_UIAcceptedCandidates` (`b2` word 2, accepted): `0x1` UIAlpha (`t11`), `0x2` the UI color
tag (`t12`), `0x4` Backbuffer (`t13`), `0x8` current color (the presented colour's alpha, `t6`;
`t0`'s alpha until the 10-05 review, which read a retained or tagged frame beside a HUD-less pair),
`0x10` HUD-less (`t14`, compared with its paired colour at `t0`, which the tiles pass reads only
with an offered HUD-less image, so the difference and lit words are zero without one), `0x20`
an exact pair (not a candidate) and `0x40` the offscreen UI layer (`t7`). The frame's decision is
`decide()` in `tools/reshade/game3d_ui_selection.h`, which the reduce ports line for line;
`SUNSHINE_UI_SELECTION_REVISION` names the revision of that port (2 since S2a: the T1 grace with the
hold store, the one-way counts, the refused candidate and the frame reason; 3 since S2b: the
informative full claims and the H1 override, texel 10 and the pre-UI scene image in texel 6; 4 since
fix 1: claim (d) for the layer by its proven signature, and texel 11; 5 since fix 2: the H2 override
with `b2` word 5, and texel 12; 7 removed fix 3 and fix 4 and decided as 5; 8 since 10-05: the empty
change set, V2 above; 9: H2 and S3 removed, so texel 12 is reserved and 9 otherwise decides as 8;
10 since WP1a: T1 keeps the held decision across HUD-less re-offers (per-frame bit `0x8000`), the
one-way test judges the declared alphas too (texel 16) and no longer the layer copy (words 32 and 36
reserved), counted on status samples only (per-frame bit `0x4000`) and only of declared tags in the
exact pair's tag batch (per-frame bits `0x1000` and `0x2000`), the layer's pixels beyond the
premultiplied bound invalidate it only when they lie on more than 1% of the frame and on more than
5% of it or more than its opaque pixels (statistics rows 208-223), H1 overrides every S1 winner,
and the shader no longer counts S1's invariants).
`reshade_game3d_ui_selection_contract` runs the reduce on crafted and random counts, previous hold
states and per-frame bits, and compares every decision word, the written hold store and the counter
adds with `decide()` and `counter_adds()`; it asserts S1's invariants (no unaccepted inferred alpha
decides, none beside an accepted declared alpha) on every case, which the shader no longer counts
since revision 10. The renderer runs automatic detection only with that layout, selection revision
10, its 17 decision texels and both scene-evidence images; otherwise its frames count as
`inactive.unprepared`. The live decoders (`ui_selection::counts_from_words`,
`ui_temporal::decode_detection_sample`, which the hidden-scene guard and the acceptance ledger read
once decoded) read only that revision's texture. `ui_detection_replay` alone still replays a shader
without the marker (layout 1, the layer in the UI color slot) or of an older revision, sizing its
statistics rows and padding its decision words itself, without the mirror check.
`Sunshine_UIDifferenceThreshold` (`b2` word 1) is the HUD-less pair's difference threshold. Since
fix 1 `b2` has five words (a 32-byte constant buffer): `Sunshine_UIPreUIThreshold` (word 4, float) is the
pair threshold of the offscreen UI layer and the presented colour (`ui_selection::comparable` of the
layer signature's format and colour space against the presented encoding), 0 without a layer or
when they are not comparable, at eight times which the tiles pass counts texel 11. The renderer
pushes it only on a detection sample frame (at most one every 100 ms, the frames whose decision
texels the CPU reads); every other frame pushes 0, so the tiles pass skips the presented-colour
loads and both passes skip the second sum and write texel 11 as zero. The decision never reads
texel 11. Since selection revision 9 `b2` has six words (`ui_detection::b2_words`, a 32-byte
constant buffer): word 5, `Sunshine_UIReserved`, is reserved and pushed as zero. H2's flag `0x1`
(fix 2) and fix 3 and fix 4's bits `0x2`-`0x200` used it, all removed; S3's words 6-9 (the proposed
labels and `Sunshine_UIIdentity`, 48 bytes in all) were removed with S3. A layer seen
through an `*_SRGB` view is not comparable with a UNORM presented frame, so it is never proven.
`Sunshine_UIDetectionFlags` (`b2` word 3) has stored and per-frame bits:

| Value | Kind | Meaning |
| --- | --- | --- |
| `0x1` | stored | The offscreen UI layer must pass the premultiplied bound (V1). |
| `0x2` | stored | With a float layer's HDR headroom. |
| `0x4` | stored | Reserved and never reused; the shader defines nothing for it. Through 180f1842 every layer copy carried it, marking the one-frame-late copy, which is never judged (A2, E2); selection revisions 2-9 read it, and `ui_detection_replay` pushes it to a shader that still defines `SUNSHINE_UI_STORED_LATE_LAYER`, so those revisions replay as before. |
| `0x8` | stored | Reserved and never reused; the shader defines nothing for it. Since selection revision 10 stored bits end at `0x800` (`0x10`-`0x800` never used) and per-frame bits start at `0x1000`. |
| `0x1000`, `0x2000` | per-frame | Since selection revision 10 (A2): the offered UIAlpha (`0x1000`) or UI color tag (`0x2000`), its candidate bit shifted by `SUNSHINE_UI_PER_FRAME_UNALIGNED_SHIFT` (12), was not captured in the exact HUD-less pair's tag batch. The tiles pass counts no strong or contradicted pixel of it, so the one-way test does not judge it on that frame. Pushed only beside an exact pair. |
| `0x4000` | per-frame | Since selection revision 10: a status sample, a detection whose decision texels the CPU reads (at most one every 100 ms). Only it counts the one-way judgment (A2) and the pre-UI pixels (texel 11) in the passes' second phase; every other frame's one-way and pre-UI counts are zero. Every `ui_detection_replay` case pushes it. |
| `0x8000` | per-frame | Since selection revision 10 (T1): the offered inexact HUD-less snapshot is the one the previous render offered, whatever tags come with it. Without a decision of its own the frame leaves the hold store as it is and applies the stored decision while one is held. |
| `0x10000` | per-frame | Reserved and never reused; before S2b the layer route's hidden-scene hold (source 8). The shader defines nothing for it. |
| `0x20000` | per-frame | Reserved and never reused; no render pushes it and the shader defines nothing for it. |
| `0x40000` | per-frame | The consumed depth is not this frame's (reused, or behind a generated Present). It only counts (`full.depth_not_current`) and makes the hidden-scene evidence invalid; the renderer then dispatches no evidence passes. H1 still acts on a held verdict. |
| `0x80000` | per-frame | Reserved and never reused; before S2b the HUD-less route's hidden-scene hold (source 9). The shader defines nothing for it. |
| `0x100000` | per-frame | An accepted candidate that the last adopting real frame offered is missing (T1). |
| `0x200000` | per-frame | No previous real decision in this chain: the T1 grace cannot reuse one. |
| `0x400000` | per-frame | The hidden-scene guard holds a hidden verdict of D on the presented frame (H1). |
| `0x800000` | per-frame | The guard's held samples read the pre-UI scene image visible, so claim (d) may act (H1). |
| `0x5f000000` | per-frame | Bits 24-30, candidate bits shifted left by 24 (`SUNSHINE_UI_PER_FRAME_REFUTED_SHIFT`): offered candidates whose signature a visible verdict refuted, so their full claims do not act (H1). |
| `0x20000000` | per-frame | Reserved and never reused; bit 29, where the exact bit would shift to (never a refuted candidate). From S2b to fix 1 the evidence passes measured the offered layer as the pre-UI scene image beside an offered HUD-less image, for the guard's D proof of the layer. The shader defines nothing for it. |
| `0x80000000` | per-frame | Bit 31 (`SUNSHINE_UI_PER_FRAME_PRE_UI_PROVEN`, since fix 1): the offered layer's signature is proven the pre-UI scene image (the ledger's `pre_ui` key), so a layer without coverage makes claim (d) (H1). |

An 8-bit layer stores 1 and a layer in the `R16G16B16A16` or `R32G32B32A32` typeless family (UNORM
included) 3 (5 and 7 through 180f1842). Stored bits describe the offscreen UI layer slot: the renderer keeps them between
frames. They key no decision or status, and since S2b nothing on the CPU reads them.
Per-frame bits ride only in one render's pushed word and are never stored.
The tiles pass reads `0x4000`, the detection reduce the guard's bits, `0x4000`, `0x40000` and the
three T1 bits, and the evidence sum reads `0x40000`; a dump's `replay.ui_detection.flags` records the pushed word. The informative
claims add claim bit `0x80` (`SUNSHINE_UI_CLAIM_PRE_UI`, the pre-UI scene image), outside the
candidate bits, and the h1 word marks an applied override with `0x100` (`SUNSHINE_UI_H1_APPLIED`).

The decision texture has `SUNSHINE_UI_DECISION_TEXELS` `R32G32B32A32_UINT` texels (5 without the
marker, 7 with scene evidence, 8 with candidate layout 2, 10 with selection revision 2, 11 with
selection revision 3, 12 with selection revision 4, 13 with selection revisions 5 to 8, 12 with
selection revision 9 and 17 with selection revision 10), and the bounded summary above is that whole
texture, 272 bytes:

| Texel | x | y | z | w |
| --- | --- | --- | --- | --- |
| 0 | Applied source (0-10) | Applied covered pixels | Pixels | Matching tiles |
| 1 | Candidate bits | HUD-less changed | HUD-less unchanged | HUD-less non-finite |
| 2 | Covered pixels of UIAlpha | Of the UI color tag | Of Backbuffer alpha | Of current alpha |
| 3 | Invalid pixels of UIAlpha | Of the UI color tag | Of Backbuffer alpha | Of current alpha |
| 4 | Lit HUD-less pixels | Accepted candidate bits | Pixels of UIAlpha at least 254/255 | Of the UI color tag |
| 5 | Presented `n` | `asuint(D)` | `valid \| ran << 1 \| verdict << 2` | Presented decided comparisons |
| 6 | Pre-UI scene image `n` | `asuint(D)` | `valid \| ran << 1` | The image: 0 none, 1 HUD-less, 2 the offscreen UI layer |
| 7 | Covered pixels of the offscreen UI layer (within the premultiplied bound since selection revision 10) | Its invalid pixels: out of range, plus those beyond the premultiplied bound when these lie on more than 1% of the frame and on more than 5% of it or more than its opaque ones (before selection revision 10 every pixel beyond the bound) | Its pixels at least 254/255 (within the bound) | Offered candidate bits that passed V1 or V2 |
| 8 | Reserved zero (the layer's strong pixels before selection revision 10, always 0 for the late copy) | Strong pixels (finite alpha of at least 1/2) of Backbuffer alpha | Of current alpha | The refused candidate's bit (0 when the frame decided) |
| 9 | Reserved zero (the layer's contradicted pixels before selection revision 10) | Strong pixels of Backbuffer alpha where an offered exact pair's HUD-less image is lit and unchanged against both the pair's colour and the presented colour | The same of current alpha | Bits 0-7: the own decision's `ui_no_mask` index, 0xFF when it decided a source; bit 16: the T1 grace reused the previous decision |
| 10 | Pixels of Backbuffer alpha at least 254/255 | Of current alpha | The informative full claims before refutation (candidate bits, `0x80` the pre-UI scene image) | The h1 word: bits 0-7 the S1 winner's source, `0x100` H1 applied |
| 11 | Pixels where the offscreen UI layer's RGB equals the presented colour (words 44-47; since fix 1) | Lit layer pixels | Reserved zero (lit presented pixels in selection revisions 4 to 8) | Reserved zero (lit presented pixels that differ from the layer in selection revisions 4 to 8) |
| 16 | Strong pixels of UIAlpha (words 64-67; since selection revision 10; zero when pushed unaligned, `0x1000`) | Of the UI color tag (zero when pushed unaligned, `0x2000`) | Strong pixels of UIAlpha where an offered exact pair's HUD-less image is lit and unchanged against both the pair's colour and the presented colour | The same of the UI color tag |

Texel 12 (words 48-51) is reserved: selection revisions 5 to 8 wrote H2's still and compared cells
there and, from a shader with `SUNSHINE_UI_IDENTITY`, S3's identity verdicts and deltas, both
removed. Texels 13-15 (words 52-63) are reserved too: selection revision 6 (fix 3's change-set
shadow and fix 4's darkening words) used them, and both were removed by user decision. Selection
revision 10 writes texels 12-15 as zeros and the declared alphas' one-way counts in texel 16. The
one-way counts of texels 8, 9 and 16 and texel 11 are zero on a frame that is not a status sample
(per-frame bit `0x4000`).

Texel 0 is the applied decision: the frame's own, or under the T1 grace the stored one of the
previous real frame. The other texels describe the frame's own counts. The refused candidate is the
first candidate in draw order that the reason names: for `gate_no_hold` the first claimant whose
claim acts, or for the pre-UI claim alone the HUD-less image when offered, else the layer; for `presented_blocked` the first blocked accepted
valid inferred alpha; for `trusted_invalid` the first accepted invalid alpha; for `layer_aside` the
layer; for `unaccepted` the first unaccepted valid selective candidate; for `difference_failed` the
HUD-less image; for `ambiguous` the first unaccepted valid alpha that is not selective; none for
`no_candidate` and `other`. A frame without candidates that runs only for the grace keeps the
previous frame's statistics, so its texels 1-10 are stale; it never submits a sample. Texel 11
compares at eight times `b2` word 4 (match: the largest channel difference at most that, relative
above one in scRGB; lit: the largest channel above it) and is zero without a layer or a comparable
pair; the reduce writes it as zero when no layer is offered, so a grace frame never carries an
earlier layer's counts. Its first two words feed the layer's pre-UI proof
(`ui_selection::pre_ui_match`); the last two are reserved zeros.
The verdict is 0 none, 1 hidden, 2 ambiguous or 3 visible. The detection reduce writes texels 5
and 6 as zero on every frame and the evidence passes overwrite them when they run, so a sample
never carries an earlier frame's evidence; texel 6 stays zero without a pre-UI scene image. The
pre-UI image's verdict is not stored: the CPU reads it from D with the bounds above.

`SUNSHINE_UI_SCENE_EVIDENCE_IMAGES` (0 without the marker) is 2 when the shader measures the
presented and pre-UI scene images (before selection revision 3 the HUD-less image). The statistics texture has 16 columns and 112 rows, plus 9 with
scene evidence: rows 0-15 hold each tile's alpha coverage, 16-31 its invalid alpha, 32-47 its
HUD-less difference counts, 48-63 its lit HUD-less pixels with the nearly opaque pixels of UIAlpha,
the UI color tag and (`.w`, since selection revision 3) Backbuffer alpha, 64-79 the offscreen UI
layer's covered, invalid and nearly opaque pixels with (`.w`) the nearly opaque pixels of current
alpha,
80-95 (`SUNSHINE_UI_JUDGMENT_ROW` onward) the strong pixels of UIAlpha, the UI color tag,
Backbuffer and current alpha (since selection revision 10; the layer, Backbuffer and current alpha
before), and 96-111 those of them that an offered exact pair contradicts (the texel 8, 9 and 16
counts per tile, on status samples only); rows 112-120 (`SUNSHINE_UI_SCENE_PARTIAL_ROW` onward) hold each 16x16-cell comparison
group's {n, wins - losses of each image, the presented image's decided comparisons}. Since selection
revision 4 the texture has 144 rows: rows 128-143 (`SUNSHINE_UI_PRE_UI_ROW` onward) hold each
tile's texel 11 counts {match, lit layer, 0, 0}, and rows
121-127 are unused. Selection revisions 5 to 8 had 160 rows, rows 144-152 holding each comparison
group's H2 counts {still cells, compared cells, 0, 0}; rows 144-207 are reserved. Since selection
revision 10 the texture has 224 rows: rows 208-223 (`SUNSHINE_UI_LAYER_BOUND_ROW` onward) hold each
tile's layer pixels beyond the premultiplied bound {beyond, 0, 0, 0}, which the reduce weighs
against 5% of the frame and the layer's opaque pixels (V1). On a status sample the tiles pass and the reduce sum the
one-way and pre-UI counts in a second phase that reuses the coverage, invalid and difference
group-shared arrays, which keeps both within the 32 KiB `cs_5_0` limit; every other frame skips it
and writes those rows as zero. The 256 x 144
cell sums have their own texture, because a pass cannot read the texture it writes.
`SUNSHINE_UI_SCENE_*` defines mirror the statistic's constants. The renderer sizes the textures,
the readback and its parse for the current revision, and makes automatic detection unavailable for
a shader whose markers differ from it. The T1 hold store (`SunshineUIHoldStore`, three `R32_UINT` texels at
`u5`, `SUNSHINE_UI_HOLD_*`) is bound for the reduce only, never beside the resolved-plane store
that shares the register. It needs no initial clear: every chain's first detection pushes
`0x200000`, so the reduce ignores the store and writes it whole, and later detections of the chain
read what it wrote. The offline replay binds
nothing there, so its reduce reads state none and never reuses a decision.

**UI counters.** Each `Sunshine UI protection` line describes the latest 100 ms status sample;
the UI counters count every frame. `tools/reshade/game3d_ui_counters.h` owns their names, the GPU
word indices and the log text. On every detection frame thread 0 of `SunshineUIDetectionReduceCS` adds
the frame's outcome to a 26-word `R32_UINT` texture, `SunshineUICountersStore` at `u7`, which is
bound for the reduce only. `SUNSHINE_UI_COUNTER_*` defines mirror the word indices, and
`reshade_game3d_ui_layer` fails when they disagree. The renderer zero-clears the texture once and
copies it beside the decision texels on sample frames, under the same fence. It then adds the
uint32 wrap-safe change since its previous read to the game session's totals, together with its own
CPU counts as they stood when that sample was submitted. That is one locked call per committed
sample. The renderer and the sequence replay decode the sample and assemble its counts with the
same functions in `tools/reshade/game3d_ui_temporal.h` (`decode_detection_sample`,
`sample_counters`). Counting adds no per-frame CPU wait, lock or GPU synchronization. The renderer counts only
with a shader whose `SUNSHINE_UI_COUNTER_WORDS` is 26, so a shader of another word layout leaves
counting off instead of misattributing its words, and the offline replay binds nothing at `u7`, so
its adds are dropped. `decided` has eleven words, sources 0-10 by index (the retired 7 and 9 stay
zero), followed by `inexact_difference`, `depth_not_current`, `contradicted`, the nine no-mask
reasons, `full_alpha` and `reused`. Tools use the named indices; no log, dump or replay tool reads
the words themselves. Shaders through dc7e3c77 wrote 31 words (29 before fix 3, 28 before fix 2):
`decided` had thirteen, sources 0-12 with the removed H2's 11 and fix 3's 12 kept as zero words,
words 14 and 18 were S1's invariants (zero by construction, no longer written since selection
revision 10) and word 30 fix 3's `refined`; S3's identity verdict words 31-35 (a 36-word texture
from a shader with `SUNSHINE_UI_IDENTITY`) were removed with S3.

| Field | Counts |
| --- | --- |
| `auto_frames` | Renders that requested GPU detection: Auto, or an explicit HUD-less difference input outside manual Off. |
| `detection_frames` | Frames on which the detection passes ran: every real Present that detected, including a frame without candidates that ran for the T1 grace. |
| `held.generated` | Generated Presents that applied the decision of the real frame they show (T1); they run no detection. |
| `held.none` | Generated Presents (offering nothing within the reported generated count of the last Present that offered a UI tag, or under FG re-offering exactly the snapshots of the last detecting Present) without such a decision in their identity scope: no mask, and the chain ends. A Present that offers a HUD-less image, a re-offered snapshot included, is real and never counts here: it detects, and when mispaired it reuses the held decision (`reused`). Counter lines before the 10-05 pairing change also count Presents the old HUD-less pairing held as generated. |
| `reused` | Detection frames without a decision of their own that applied the held decision and mask (decision texel 9 bit 16): the T1 grace once after a real decision, and since selection revision 10 every HUD-less re-offer (`0x8000`) while a decision is held. |
| `inactive.no_candidates`, `inactive.size`, `inactive.unprepared` | Requested renders without detection: no usable candidate, a frame larger than 3840, or detection resources that could not be prepared. |
| `decided.N` | Detection frames by applied source 0-10, a reused decision included. 7 and 9 are retired and not logged; their GPU words stay zero. Counter lines of fix 2 to selection revision 8 also have 11 (H2's still screen). |
| `none.R` | Frames whose own decision was source 0 and that applied no mask, each by the first reason that applies: `gate_no_hold` (an informative full claim acted but H1 did not apply: no held hidden verdict; the name is kept for log compatibility), `presented_blocked` (an offered, accepted UIAlpha or UI color tag, invalid itself, kept an accepted, valid inferred alpha out), `trusted_invalid` (an offered, accepted alpha had more than 1% invalid pixels), `layer_aside` (the offered layer was V1-invalid, such as a layer without alpha), `unaccepted` (an offered, unaccepted candidate was valid and selective: its acceptance is still being earned), `difference_failed` (a HUD-less image was offered, including an inexact one that changed nearly everywhere without the pre-UI hold), `ambiguous` (an unaccepted valid alpha empty or nearly full), `no_candidate` (no alpha offered), then `other`. |
| `full.6`, `full.8` | Repeat `decided.6` and `.8`. Counter lines before S2b also have `full.9`. |
| `full.depth_not_current` | Detection frames pushed with `0x40000`, whatever they decided. |
| `full_d.hidden`, `.ambiguous`, `.visible`, `.invalid` | Committed samples that decided H1 (source 8), by the hidden-scene verdict measured on that same sample. `invalid` includes evidence that was not measured. These count samples, not frames. Counter lines before S2b count sources 6, 8 and 9 here. |
| `scene.entered`, `.released`, `.refuted` | The hidden-scene guard's observations of committed samples (since S2b): its hidden verdict entered; a held verdict released by a visible sample, every visible sample that decided 8 included; source signatures newly refuted. |
| `untrusted_inferred` | Counter lines through dc7e3c77 only. Frames decided from an inferred source (Backbuffer 3, current alpha 4 or the offscreen UI layer 10) that was not accepted: zero by construction since S1, where only accepted candidates decide, and logged as 0 from selection revision 10, when the shader stopped counting it. The selection contract and sequence tests assert the invariant on every case. |
| `inexact_difference` | Source 5 decided from an inexact HUD-less pair (before S2b also 9). |
| `contradicted` | An accepted alpha (since selection revision 10 UIAlpha 1, the UI color tag 2, Backbuffer 3 or current alpha 4; before it Backbuffer, current alpha or the layer 10) decided by the frame's own decision, not overridden by H1, while the same frame's valid exact pair contradicted it one way (A2). It keeps deciding until the ledger revokes it. Since selection revision 10 the one-way counts exist on status samples only, so it counts samples rather than frames. Counter lines before S2a have `trusted_full` instead: an accepted alpha source (1-4, 10) covering at least 99% of pixels while an exact pair without invalid pixels left at least half of the frame unchanged. |
| `presented_over_dedicated` | Counter lines through dc7e3c77 only. Inferred alpha (3, 4, 10) decided while an accepted UIAlpha or UI color tag was offered: the declared-alpha block keeps it 0, and it was logged as 0 from selection revision 10, when the shader stopped counting it. The selection contract and sequence tests assert the invariant on every case. |
| `full_alpha` | An applied alpha source (1-4, 10) covering at least 99% of pixels: a whole-frame mask from alpha, with or without a HUD-less pair. Only accepted sources decide. |
| `full_alpha_d.hidden`, `.ambiguous`, `.visible`, `.invalid` | Counter lines through dc7e3c77 only, logged as 0 from selection revision 9. Before it: committed samples that decided an accepted whole-frame decision, such a whole-frame alpha or (since S2b) an exact full change-set (6), by the hidden-scene verdict that a diagnostic evidence run, now removed, measured on that same sample from the sample after one that decided it. |
| `trust.earned`, `.revoked_exact`, `.revoked_declared`, `.lapsed`, `.restored`, `.discarded`, `.forgotten` | Session acceptance events. `earned`: a signature accepted, or a restored one confirmed, by this session's samples. `revoked_exact`: revoked when the one-way test of an exact pair contradicted it in the revoking sample (A2). `revoked_declared`: revoked by an accepted declared alpha's coverage. `lapsed`: a provisional restore lapsed. `restored`: restored entries. `discarded`: legacy `TrustedUISources` entries discarded on load. `forgotten`: accepted signatures that Forget cleared. Since fix 1 the layers' pre-UI proof keys (`pre_ui`) count with the accepted signatures: earned by matching samples, restored, lapsed and forgotten. Counter lines of S1 have `revoked_full` (a full claim over a visible exact pair) and `revoked_presented` (presented alpha disagreeing with an accepted UI channel) instead of `revoked_exact` and `revoked_declared`, and no `forgotten`; counter lines before S1 have `opaque_set` and `opaque_cleared` (the retired opaque-tag proof) instead of `discarded`. |
| `samples`, `through_ms` | Committed samples, and the sample tick through which the totals are exact. |

The totals are exact per frame through the last committed sample. A frame rendered after it (at most
one sample interval plus the pending copy) is counted by a later commit, or lost when its renderer
is destroyed first. A discarded or unreadable sample commits nothing; the next commit catches up.
At every commit `auto_frames` equals `detection_frames` plus `held.generated` and `held.none` plus
the three inactive reasons; a reused frame is a detection frame. Counter lines of S1 instead hold
`held.generated`, `held.inexact_after_exact` and `held.trusted_missing` (the three S1 hold kinds,
which entered the identity) and `held.cap` (frames that wanted a hold past three Presents and
detected instead, part of `detection_frames`), and no `reused`. The totals sum all runtimes of the
game session. While every Present reads as generated, as with a misreported multiplier, nothing
detects and no sample commits, so the `held.none` frames appear only once a real Present detects
again.

The add-on writes `Sunshine UI counters: runtime=<runtime> <fields>` after each `Sunshine UI
protection` status line once a render of that runtime has reached UI detection, whether the line
was written for a change or periodically. It writes one more when a runtime that wrote one resets
or is destroyed, so the session's totals end the log. Fields are space-separated `key=value` in
the order of the table above, and a group is written `key={key=value ...}`. The values are
cumulative, so the last line holds the session's totals.

**Readiness-report invariants.** When a log has counter lines,
`tools/reshade/game3d_log_report.py` reads the last one instead of the sampled lines; the holds
check below reads every counter line, and the pairing check also sums the `hudless_presents` of the
`Sunshine UI capture gate` lines:

| Check | Status |
| --- | --- |
| The accounting identity above does not hold | FAIL `UI counters` |
| `presented_over_dedicated` is above zero (counter lines through dc7e3c77) | FAIL `UI protection` |
| `contradicted` is above zero | `UI protection`: PASS with the count: A2 revokes on the third contradiction within 2 s and never on shorter ones, which the counter cannot tell apart, so a handled case and a short one both count `contradicted`. The sampled lines decide: three sampled lines within 2 s in which a valid exact pair contradicts the same accepted alpha one way (an inferred alpha, or since selection revision 10 UIAlpha or the UI color tag from `sampled_declared_one_way`; decided or not, the T1 reuse included, never the one-frame-late layer), with no revocation of that kind within 10 s of the first, FAIL; the same run of declared-coverage contradictions alone WARNs; shorter sampled contradictions are noted. Counter lines of S1 keep the S1 rule: `trusted_full` with `trust.revoked_full`, and sampled lines of a full claim over an exact pair that shows the scene. |
| `full_alpha` is above zero, or since S2b `decided.6` | INFO `UI full alpha` (accepted whole-frame decisions: alpha, or an exact full change-set 6), with the hidden-scene verdicts of their samples on counter lines before selection revision 9 (`full_alpha_d`, logged as 0 from it through dc7e3c77). An accepted source's whole-frame alpha pins flat even over a visible scene, as the [opacity ruling](#ui-decision-framework) intends (P1); its wrong case, an accepted source a valid exact pair contradicts, is counted by `contradicted`. |
| `decided.6` is above zero in a session whose capture gate lines counted Present-counted HUD-less pairings (`real`, `late` or `reoffered`) but never a same-batch one (`batch`) | FAIL `UI full frame pairing`: only a same-batch Backbuffer pair is exact (E2). A Present-counted pair can belong to another frame and then differs everywhere, so these frames were flattened from pairs that need not be their own (Hogwarts Legacy 10-05, which tagged only HUDLessColor, showed whole seconds of gameplay flat). |
| `full_d.visible` exceeds `scene.released` (counter lines since S2b) | `UI full frame`: WARN: an H1 hold acted over a visible scene. The sample that releases a held hidden verdict decided 8 before its own evidence read the scene visible, and the guard counts it as a release, so `full_d.visible` never exceeds `scene.released` otherwise; PASS with the H1 samples' verdicts and the guard's entered, released and refuted counts. |
| `full_d.visible` is above zero (counter lines before S2b) | `UI full frame`: WARN when the layer or HUD-less route (8 or 9) decided: each release of a held route counted one, and those lines cannot tell a release from a held route over a visible scene. INFO when only source 6 decided: an exact full change-set decided without a hold, as intended for an accepted exact pair (P1). |
| `untrusted_inferred` is above zero (counter lines through dc7e3c77) | `UI inferred alpha`: FAIL: zero by construction since S1. WARN on counter lines before S1 (with `trust.opaque_set` rather than `trust.discarded`), whose untrusted pass could decide. PASS when it is zero; a later line, which no longer measures it, has no such check. |
| `inexact_difference` is above zero | INFO `UI inexact difference`: source 5 from a Present-counted pair, which only proposes the frame; its own pixels validated the difference (V2), so it is expected wherever a game offers no same-batch pair. Before the 10-05 pairing change this warned pending S3's frame identity. |
| A counter window (from one counter line whose `auto_frames` advanced to the next; windows less than 1 s apart merge) in which `held.none` was at least half of the Auto frames for at least 2 s | WARN `UI holds without a decision`, each window with its time, length, FG state and share: those Presents had no real-frame decision to show (T1) and so no UI mask. FAIL when a window held at least 90% for at least 10 s: UI detection effectively never ran (Hogwarts Legacy 10-05 at 4x frame generation, before the pairing change). PASS without such a window. |
| Holds by kind (`held.generated`, `held.none`) and `reused`; on S1 lines the three hold kinds and the cap | INFO `UI holds` |
| Frames without a mask by reason, as a share of `auto_frames` (`held.none` included), with the sampled lines' reasons and refused candidates | INFO `UI no mask` |
| Acceptance events, `forgotten` included | INFO `UI trust events` |
| A pre-UI proof key (`pre_ui:<format>:<space>`, since fix 1) restored, added or removed by the acceptance lines | INFO `Pre-UI proof` with the key: restored (provisional), earned, lapsed, or forgotten when a Forget line beside the change names it. The key is never counted as a UI source of its kind in the dispute checks. |
| A `Sunshine Game 3D: diagnostics on/off` line | INFO `Diagnostics` with the switch; with it off the per-pass GPU stage times are absent by design, and `Game 3D cost` reads `GPU timing off (Diagnostics=0)` (INFO, not WARN) when the timing line says `gpu_profile=disabled`. Unexpected hitches name the depth-source flip steps listed under [Diagnostics switch and per-Present cost](#diagnostics-switch-and-per-present-cost). |

The checks of removed features are gone, and their lines still parse without a check: the
first-run shadow (`UI first-run shadow`), the dark pre-UI image statistics (`Dark pre-UI image
(shadow)`), H2's still screens (`UI still screen`, from `Sunshine UI still screen` lines and the
`still` groups) and S3's identity shadow (`UI identity (S3)`, from `Sunshine UI identity` and
`Sunshine FG interposers` lines). Since selection revision 9 the add-on logs their remaining fields
(`shadow`, `shadow_hidden_ms`, `presented_lit` and `presented_lit_differs`) as 0; it logged
`full_alpha_d` as 0 through dc7e3c77, and later counter lines omit it.

The acceptance-dispute and time-based checks (`UI protection gaps`, hidden scene) still read the
sampled lines; on lines since S2b the hidden-scene check lists the H1 samples by the pre-UI image
whose claim acted with the guard's counts from the last counter line, and source 8 is protected. It
also counts hidden samples whose layer could not act for want of a proof: on lines since fix 1 an
offered layer without coverage, the pre-UI image, while `proven=0` (no proof yet: three samples over
2 s in which the layer equals the presented frame on at least 90% of pixels and is lit on at least
half); on S2b lines a layer claim (`0x80`) without the guard's D proof. In logs before selection
revision 9 it also warns where the removed first-run shadow measured the presented frame hidden for
at least 500 ms while no UI source decided (`shadow_hidden_ms`). A `UI protection gaps` window lists
its pieces, each labelled with its own line's FG state; a line whose status sample was still
pending (`checking` or `searching`), or one that did not render, continues a run, labelled with that
state after the last sample's reason, as the panel's warning run does. `UI channel admission` warns
on samples that decided no source while a selective UIAlpha, UI color tag or layer was rejected for
invalid pixels on more than 1% of the frame (V1). Logs without counter lines keep the sampled checks unchanged. A dispute is judged as
the ledger judges it: on lines since S2a an accepted, valid inferred alpha (a layer that is not the
one-frame-late copy included) whose coverage differs by at least 10% of the frame from each
accepted, valid UIAlpha or UI color tag, as a run of three within 2 s (shorter ones are noted), and
a resolving revocation is named by its A2 kind (declared coverage, or one way by an exact pair); on
older lines presented alpha against any accepted UI channel, the layer included. The
report derives V1 and V2 validity from the logged counts, since the line carries no validity bits. A
gap on a line since S2a also names the add-on's own reason and refused candidate. The report reads
`Sunshine UI protection` lines of every candidate layout: a line before S1 (`trusted_alpha`,
`sampled_ui_layer`) is read with its UI layer moved to the layer candidate (bit `0x40`, source 10),
and acceptance changes from both `accepted UI sources are now <keys>` and the older `alpha trust is
now <bits>`. It lists the Forget line as INFO; the first-run shadow's session line is ignored. UI
lines since fix 1 end the hidden-scene fields with `sampled_pre_ui_pixels`; lines of fix 2 to
selection revision 8 add H2's `still` group after it, which is ignored, and a counter line's
`decided.11` is named as H2's removed still screen. Since selection revision 10 the UI line's
`sampled_declared_one_way` feeds the A2 run check above, and capture gate lines count `reoffered`
pairings with the Present-counted ones.

Besides these, the report checks the session as a whole (the
[first-run report](../tools/reshade/README.md#first-run-of-a-new-game) lists them): add-on
readiness, depth gaps and flat placement outside the settle time after an export start, FG switch
or runtime reset, capture coverage and status, observation losses, export pauses, hitches, cost,
stream delivery, UI layer copies and the host log. A failing Streamline capture status inside that settle time is INFO (the source
settling) and outside it WARN. `Game 3D cost` covers every timing line's window, each mean weighted
by its window's Presents (CPU) or GPU frames, with the largest maximum. A log that ends within
seconds of ReShade tearing its runtimes down (`Destroyed runtime environment on runtime`) counts as
a normal exit: some games (Unreal) end the process before ReShade logs its own exit.

Stage (S0-S6) and rule IDs (E1, E2, V1, V2, A1-A3, S1, S2, H1, H2, P1, T1, F1) in report output,
replay labels and sequence cases are those of the [UI decision framework](#ui-decision-framework). Each stage that changes a decision
updates the affected replay labels and sequence cases.

Detection rule changes are checked offline before a live test. `ui_detection_replay` (built with
the add-on) compiles the three detection passes from a shader file, and its three scene-evidence
passes when the shader has them, binds each Dump 3D package's
captured candidates by artifact kind (presented color, Backbuffer, UIColorAndAlpha, UIAlpha,
HUD-less, or a census `ui_layer_candidate_N` in the layer slot `t7` with the layer's stored flags;
in the UI color slot for a layout 1 shader; a census with `ordering` carries such an artifact only
for a copy proven complete, and a label naming an omitted one fails its case, while an older
census's artifacts were read unordered ([census ordering](#dump-3d-diagnostics))), sets the candidate
and exact-pair bits and the
accepted candidates a label names (`accepted`, a list of those artifact kinds; the retired
`trusted` key fails its case), and compares the decision
and the resulting mask (empty, partial HUD, or flat) with that label:

```powershell
.\ui_detection_replay.exe ..\..\tools\reshade\game3d_native.hlsl E:\ApolloDev\sbs_dump\ui_detection_cases.json
```

It also pushes the exact 80-byte `b0` from `replay.parameter_hex`, as every renderer compute pass
receives it, and binds the raw depth artifact at `t1` and the presented color at `t6` (`t0` stays
the color HUD-less is paired with; current alpha reads `t6`) for the scene-evidence passes, which it runs after the decision
as on a sample frame and which read the pre-UI scene image from `t14` or the layer slot `t7`; they
write only decision texels 5 and 6. Its `b2` has the renderer's six words (32 bytes): word 4 is
`ui_selection::comparable` of the layer artifact's encoding and the presented one, so texel 11
counts the captured layer against the captured presented colour, and word 5 is reserved and zero.
A package of selection revisions 5 to 8 also records H2's flag (`still_bits`) and S3's stamps and
proposals; the replay ignores them, and the `--identity` option is gone.
Without a raw depth artifact
`t1` is a 1x1 placeholder and depth and camera readiness are cleared, as in a render without depth.
A label's hidden-scene guard keys set the per-frame bits of H1: `scene_hold: true` pushes the held
hidden verdict (`0x400000`), `pre_ui_visible: true` the held visible pre-UI image (`0x800000`), and
`refuted`, a list of artifact kinds, those candidates' refuted bits (shifted left by 24);
`depth_not_current` pushes `0x40000`. `pre_ui_proven: true`, only with an offered layer, stands
for the ledger's pre-UI proof of the layer's signature: it pushes `0x80000000` on every pass, with
any form of `scene_hold`, and with `"measured"` it is the guard's `layer_proven`, so a proven layer
that is the pre-UI image makes the evidence actionable from the tick-900 sample. Without it the
layer makes no claim (d) and its pre-UI image never acts. Before fix 1 it had the guard observe a
synthetic gameplay sample at tick 800 that proved the layer by D. `scene_hold: "measured"` instead has
`game3d_scene_guard.h` observe the frame's own sample three times, at ticks 900 (which only opens
the gate, as the first live sample showing a claim does), 1000 and 1100, with each artifact's
signature (its typed format and the manifest colour space), then reruns the reduce, mask and
evidence passes with the bits the guard pushes at 1100; the line shows them as
`measured(per_frame=...)`. It proves the relation within one sample, not temporal behaviour, which
the sequence replay owns, and fails its case with a shader without scene evidence. The keys
`scene_hold_hudless` and `hudless_scene` (in `expect` or `xfail.today`) are retired since S2b and
fail their case, as do a malformed `scene_hold` and an unknown `refuted` kind. A `sample` key is
accepted and ignored, because `0x20000` is reserved. It sizes the
statistics and decision textures from the shader's markers, but
never below 80 rows and 6 texels, so retired shader revisions still replay. `expect.mask_exact`
compares the resolved R32 mask bit for bit with a CPU reference: the selected candidate's raw alpha
(red for UIAlpha, the offscreen UI layer included), all zeros without a source, or all ones for a
full-frame decision (sources 6 and 8); a HUD-less difference has no reference and fails the
check. The brief shader revision with the late-layer margin (2026-10-02) dilates an offscreen UI
layer's mask, so such a case reports `differs` with that shader. `expect.scene` checks the presented image's
hidden-scene evidence in decision texel 5 and `expect.pre_ui_scene` the pre-UI scene image's in
texel 6, with `image` (`hudless` or `layer`) the image it measured: `verdict` is one name or a list
(none, hidden, ambiguous or visible; the pre-UI verdict is read from its valid D with the same
bounds, else none) and `d_min` and `d_max` bound D inclusively. Evidence that did not run, as with a shader without scene evidence, fails the check.
`expect.pre_ui_match` (`true` or `false`, since fix 1) checks whether the sample would count toward
the layer's pre-UI proof: an offered layer without coverage and `ui_selection::pre_ui_match` on
texel 11. Stellar Blade's SDR gameplay dumps expect `true`, its settings pages and its HDR layer
`false`; their cases with `pre_ui_proven` are flat on the settings pages and never flat in gameplay,
where the presented frame reads visible.
On the labelled dumps the GPU's D equals the CPU oracle `inspect_game3d_dump.py --scene-evidence`
within 0.0012.
`--write-mask <new-dir>` copies each labelled package that consumed an automatic R32 mask of the
detection extent to `<new-dir>/<NN>_<dump>`, where `NN` is the case's position in the cases file,
with its `ui_source_color` replaced by the mask this replay resolved, for
`replay_game3d_dump --shader`; any other package fails its case, and the captured package is never
changed. The copy's `ui_detection_replay_mask` records the decision and the detection `flags` this
replay pushed, which describe the replaced mask. `--verbose`
prints every decision word (48 with selection revisions 4 and 9, 52 with revisions 5 to 8, 68 with
revision 10). Each line also shows the own decision's
reason (`(reused)` when the grace applied), the refused candidate, the one-way counts (since
selection revision 10 the layer's as reserved zeros, and UIAlpha's and the UI color tag's as
`declared_one_way`; every case pushes the status-sample bit `0x4000`), the
informative claims (`claims=`), the h1 word (`h1={applied=,winner=}`) and, since selection
revision 4, texel 11 (`pre_ui_pixels={match= image_lit=}`).
With a shader of the current layout and selection revision (10, **UI detection flags and decision
texels** above), every decision, its claims and its h1 word
are checked against `decide()` on the GPU's counts with the pushed bits and no previous decision
(`mirror=match`; a mirror that differs fails its case). A single-frame replay binds nothing at the hold store, so the T1 grace never
fires: a dump taken on a frame whose decision the grace reused replays as its own decision (no
mask, with its reason), and its label must expect that. A label whose dump directory no longer exists is
reported as SKIP and does not fail the run; a directory without its manifest fails, a `dump_root`
that is not a directory stops the run, and a run in which every case was skipped fails.

A label can record a known-wrong cell. Its `expect` then holds the target outcome, and its `xfail`
holds three fields:

- `stage`: the [roadmap stage](#ui-decision-framework) expected to fix the cell, one of S1, S2a,
  S2b, S3, S4, S5 or S6.
- `reason`: text that names the [rule](#ui-decision-framework) that fixes it, as a standalone ID
  (E1, E2, V1, V2, A1, A2, A3, S1, S2, H1, P1, T1 or F1).
- `today`: the outcome the current shader gives, with the fields of `expect`. `mask` and a
  non-empty `source` list are required.

Such a case reports XPASS when the target is met (remove its xfail), XFAIL when today's outcome is
met, and FAIL otherwise, so an unexpected change is never green. A malformed xfail fails its case.
A label with `needs_dump: "<what is missing>"` is skipped with that reason, for a dump that lacks a
needed artifact. The run ends with

```text
PASS|FAIL UI detection replay: N passed, X xfailed, Y xpassed, F failed, S skipped
```

It fails on any FAIL and when no case ran. `--strict` also fails on an XPASS.

Temporal rules are checked by `reshade_game3d_ui_sequence`, a ctest executable. The renderer's hold
arbiter, scope clears, sample decode and counter commit are pure functions in
`tools/reshade/game3d_ui_temporal.h` (`ui_temporal::detection_state`, `decode_detection_sample`,
`sample_counters`), and its hidden-scene guard is `scene_guard::state` in
`tools/reshade/game3d_scene_guard.h`, all called by `render()`, `detect_ui()` and
`poll_detection()` in a fixed order. The test drives per-frame decision streams through those
functions, `alpha_auto_policy` trust and the Present counting in exactly that order. The GPU
decisions are inputs: synthetic, or taken from replay output of labelled dumps, checked against
`decide()` when loaded; the test carries the T1 hold store from one detection to the next through
`decide()` as the GPU does, and a frame without candidates that runs for the grace reads the
previous statistics and submits no sample. The test asserts
today's behaviour, including synthetic adversaries: RE9-like presented alpha, a premultiplied
bloom-like layer, a dark grainy scene under a full claim, dark gameplay in which the presented
frame and the pre-UI image both read hidden (also with their D correlated under grain, and a Dead
Space-like stream without claims near the bounds), fog or grain after a scene buffer that stays
visible while the presented frame drifts hidden (never flat: such a target never equals the
presented frame on 90% of pixels, so it is never proven), black frames over a transparent real UI
layer (equal everywhere but unlit, never proven),
Resident Evil Requiem-like dark gameplay with an accepted UI color tag beside accepted presented
alpha that is opaque everywhere (never flat, no claim), and frame generation whose presented
cadence differs from the multiplier the provider reports (lagging, leading, or changed in the middle
of a real frame). Frame generation streams identify generated Presents the production way: a game
tags its UI inputs on real frames only, a HUD-less image is offered on every Present as an inexact
pair, and a Present that offers nothing within the reported count of the last tag is generated; a
stream whose generated Presents re-offer their real frame's snapshot identities holds them by
identity.
Recorded streams from the labelled dumps cover Stellar Blade's SDR menu visits
from FG-off gameplay and with FG suspended after FG-on gameplay that proves the layer by its pixels
(H1 (d)), the same menu booted into before any gameplay (unproven in a first session, never flat), and an
accepted partial winner that H1 overrides, refutes and re-arms. Since fix 1 they also replay the
session that motivated it (FG-on gameplay proving the layer, the FG suspension's viewport change, a
mismatching fade-in sample, a menu flat from its second hidden sample and released by the first
visible gameplay sample), a second session with the restored `pre_ui` key (a menu before any
gameplay flat, the provisional key lapsing after 60 s of testable time and not while paused,
confirmation and Forget), and keys that never prove (a covered layer) or that are independent (an
HDR and an SDR signature).
Each stream also checks that the committed UI counters reconcile with its own
per-frame tally, that the guard's `scene` counts equal its observations and that `full_d.visible`
never exceeds `scene.released`. An outcome a roadmap stage will change is marked
`KNOWN_TODAY <stage> <rule>: <text>`, which prints and does not fail; that stage turns it into a
strict assertion. A known limit that no stage changes is marked `KNOWN_LIMIT <rule>: <text>`, which
also prints and does not fail. An outcome the rules already call correct, such as an accepted
source pinning a whole-frame alpha flat over a visible scene, is asserted strictly. The run ends
with `PASS UI sequence replay: <groups> groups, <n> KNOWN_TODAY, <m> KNOWN_LIMIT`. Since S2b that
was 28 groups and two S3 (T1/E2) lines, and fix 1 added four groups. Fix 2 added one for H2, and
the S3 shadow two (its GPU identity verdicts and its identity shadow); fix 3 and fix 4 added groups
for the pre-UI change set and pin only UI. All of them were removed with their rules. The 10-05
pairing change adds `E2/V2/A1/H1 Hogwarts HUD-less pairs exact only by batch`: a game that tags
only HUDLessColor under 4x frame generation offers an inexact pair on every Present, earns
acceptance from its first V2-valid partial set and decides 5, while a full change set over a visible
scene decides no mask (never 6, never 8 without the hold); with FG off a counted full change set
never decides 6; such a game's title over a hidden scene is flat through H1 (d) (8) from the hold's
entry; and a same-batch exact full change set still decides 6. Selection revision 10 derives that
group's change sets from the pair the provider makes for each Present under frame generation (a
snapshot's first offer on the Present after its tag with that Present, a late first offer and every
re-offer with its own Present), with the real Present last or first in 2x and 4x windows, and with
the real Present last and a capture one Present late, so every mispaired gameplay Present shows the
held decision, and adds `T1 HUD-less re-offers keep the held decision`: UIAlpha on real Presents
only beside a HUD-less image re-offered on all of them at 3x and 4x, both orders, shows the last
real UIAlpha on every generated Present; with FG off a re-offer pairs with its own retained frame
and decides, and once it is too old to pair the accepted pair is missing, so the grace applies once;
no re-offer bit is pushed beside an exact pair, while a tag beside an inexact re-offer leaves it pushed;
and an unaccepted, opaque UI color tag offered on every Present beside the re-offered image at 3x and
4x, both orders, leaves every generated Present the real frame's change set. The A2 group also checks that a
declared tag pushed unaligned with the exact pair is never judged. The 10-05 review adds `T1
generated Presents re-offering their real frame's snapshots hold`: at 2x-4x generated Presents
re-offer the real frame's Backbuffer, UIAlpha and layer identities and hold by identity. That makes
35 groups. With S3
removed (selection revision 9), its two T1/E2 lines are known limits of Present counting rather
than outcomes a stage will change: under a frame-generation multiplier reported lower than the
presented one, or changed in the middle of a real frame and reported late, the generated Presents
beyond the reported count fall to the T1 grace (the first reuses the decision once, the rest have
no mask), so the run ends with 0 KNOWN_TODAY and 2 KNOWN_LIMIT.
The single-frame replay remains the gate for decisions. `--log <ReShade.log>`, outside ctest,
replays logged samples through the trust policy and prints its predicted transitions beside the
logged ones; logged samples are sparse, so this is informational only.

The labels live next to the dumps, which stay outside the repository. A rule change is accepted
only when every labelled screen of every game still passes; add a label whenever a new screen
exposes a wrong decision. A dump's HUD-less capture can be later than the frame the live GPU used,
so HUD-less labels accept either a partial or an empty mask when that pairing is uncertain.
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

Inside the central rectangle, each pixel first loads the selected alpha. Only pixels the UI pin
band pins exactly (alpha at or above 1/8) cause a scene-field load and five comparisons; fainter
UI blends toward scene depth and never places the plane. These measure the conditioned scene
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
Coverage at weight 1 of at least 99% of the central region (a full-frame decision: H1's source 8,
a full change set, an opaque menu) is not placement evidence: no scene is visible for the UI to
conflict with, and under H1 the depth does not describe the image. Such a sample is rejected as
`full_frame` and holds the applied position, so a menu over a hidden scene neither ramps forward
toward that scene's near geometry nor forces a retreat when gameplay resumes.
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
old scope. Provider sources keep their feature/viewport identity; an observation loss (revision)
keeps placement, because it changes neither the scene nor the UI. Resetting on it snapped the UI to
the screen at every loss (Hogwarts Legacy: about 20 in 2.5 minutes).
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
**UI pin band.** After the existing horizontal conditioning, `SunshineApplyUICS` pins the
conditioned field `h` toward the UI plane `pUI`. A texel `u` with finite positive mask alpha `a`
pins with weight `w(u) = saturate(8 * a)` (`SUNSHINE_UI_SOFT_PIN_GAIN`), exactly for alpha at or
above the knee 1/8 (32/255 for 8-bit masks), and leaves the slack
`r(u) = (1 - w(u)) * |h(u) - pUI|`. Each row becomes `field = clamp(h, pUI - b, pUI + b)` with
`b(x) = min over u of [r(u) + 0.5 * max(|x - u| - 1, 0) / source_width]`. A binary mask, and any
mask whose finite positive alpha is at or above 1/8, reproduces the previous distance rule
`pUI +/- 0.5 * max(d - 1, 0) / source_width` bit for bit; the zero-slack bound keeps that rule's
non-precise multiply-adds, which `reshade_game3d_native_shader_test` checks in the compiled pass.
An alpha ramp from 0 to 1/8 spanning at least `2 * |h - pUI|` rows (in pixels) adds no vertical
step above 0.5 px, where the binary rule switched a whole row at once. An 8-bit ramp that ends at
32/255 reaches weight 256/255, so its steps can exceed 0.5 px by that factor. Live mode 5
supplies the applied display-fraction displacement
described above. This existing local conditioner can compress nearby foreground or bring nearby
background forward toward the UI plane; the trial adds no separate scene-wide compression. The one-pixel
horizontal collar protects the bilinear color footprint. Exactly pinned UI shifts rigidly in each
eye without overlaying a second copy of already-composited text, and `b` grows by at most
0.5/source_width per pixel, which preserves the horizontal invertibility bound. Empty rows retain
their original field exactly. It does not retain the
vertical shear bound across UI-row boundaries. Half-transparent UI at or above 1/8 retains its
original color but locally flattens the background beneath it; exact independent stereo background
requires a separate HUDless image and UI color/alpha layer. Source alpha interpretation is not inferred from NGX or SL
provider identity. Dump/replay records the effective switch as `replay.source_alpha_ui`, the chosen
request as `source_alpha_ui_requested`, and the reason as `source_alpha_ui_status`. `source_alpha_auto`
records automatic/manual state separately from the current render. New `ui_source_detection`
metadata replaces session qualification and carries no approval flag. Its asynchronous summary
may describe an earlier frame and is diagnostic only. Older packages also carry monitoring,
window and probe-interval fields; current packages omit them, and none authorizes live UI. Offline replay uses the
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
through its red channel: the selected source's raw alpha, the offscreen UI layer's included, and
for a full-frame decision (sources 6 and 8; 9 in packages before S2b and 11 in packages of fix 2 to selection revision 8) all 1.0. It includes an all-zero
result when every candidate was rejected;
the enabled mask path does not by itself imply any protected pixels. Replay uses these frozen
values and does not run candidate selection again. Optional candidate metadata remains separate.
`replay.ui_detection` records the detection constants behind that mask: `candidates`,
`threshold_bits` (the float32 difference threshold), `pre_ui_threshold_bits` (since fix 1, the float32 pair threshold of the offscreen UI layer and the presented colour, `b2` word 4), `accepted` (candidate bits), `candidate_layout`, `flags` (the full pushed word,
per-frame bits included), `ran_or_held` (`ran` this render from this real frame's own offered
candidates, `held` by a generated Present showing a real frame's decision (T1), or `inactive`) and
`held_presents` (the generated Presents held in a row). `replay.ui_pin` records the captured shader's
`soft_pin_gain`, `decision_texels` (10 with selection revision 2: texels 8 and 9 and an applied
decision in texel 0; 11 with selection revision 3: texel 10 and the pre-UI scene image in texel 6;
12 with selection revision 4: texel 11, the layer against the presented frame, and again with
selection revision 9; 13 with selection revisions 5 to 8: texel 12, H2's still and compared cells
and, from a shader with `SUNSHINE_UI_IDENTITY`, S3's identity verdicts, both removed; selection
revision 6's texels 13-15, fix 3's change-set shadow and fix 4's darkening counts, were removed;
texels 12-15 stay reserved)
and `evidence_images` markers; 0 means absent
(binary pinning and the 5-texel decision of older packages, which replay
that way with their embedded shader unless `--shader` is given). Packages of fix 2 to selection
revision 8 also record `replay.ui_detection.still_bits` (H2's `b2` word 5) and, since S3,
`expected_layer_present`, `expected_layer_token`, `expected_hudless_present`, `identity_bits` (`b2`
words 6-9) and `stamps`; `ui_detection_replay` ignores them. Packages captured while the shader
had the late-layer margin, briefly on 2026-10-02, also carry `late_margin`, the texels by which
that shader dilated the offscreen UI layer's mask in rows and columns; `--pin-metrics` counts those
margin texels as UI. In the pass
table, the horizontal limiter of a shader with limiter line groups reads no UI input;
`SunshineApplyUICS` pins the selected mask into `final_field` after it, and on mode-5 probe frames
`SunshineUIConflictCS` observes the unpinned field in between.
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
A request matches a tag only while its runtime refreshed it (on every Present that wants UI
captures) within 250 ms (`maximum_source_age_ms`) of the tag's entry, so a swapchain that stopped
presenting (a launcher, an old swapchain beside its replacement) no longer makes every tag of the
live one `ambiguous_request`, and a new runtime takes the entry of the one refreshed longest ago
once it expired. A tag is copied only while its runtime read its UI captures within 1 s
(`ui_mask::reader_window_ms`): a Present that requests captures but returns before its render
(a renderer still compiling or unavailable) reads none, and its tags are refused rather than
copied for no reader. The capture owner is read once per Present (`ui_mask::acquire_all`: every
recent snapshot polled once, each filtered kind's newest ready one returned), and a tag scans the
owner, its own kind only, only when both of its kind's reservations are taken. The gate line's
`begin_refused` counts, per stage and since start, the tags whose `begin()` stopped without a
capture attempt: `no_request`, `ambiguous_request`, `kind_filtered`, `not_newer`, `shape`,
`unsupported_lifetime`, `no_reservation`, `no_reader` (the reader rule above) and `format` (the
tagged resource's format, stored as its snapshot would be, cannot be read as its kind:
`ui_mask::supported_format`, checked before any copy; such a tag revokes the kind's older
snapshots like any invalid tag); `begin_last` names the request's latest. `no_reader` and `format`
are new in WP1b: before it such tags were copied and then never read or never usable.
A bounded `Sunshine UI capture gate` log reports the same evidence. An admitted hook with a
zero capture boundary and a changing request generation indicates that the request was replaced
or invalidated after admission, rather than that the game supplied no UI tag. Its
`hudless_presents` counts, since the previous line, Presents whose newest ready HUD-less capture
paired with its `batch` Backbuffer (exact), with the current Present after its tag (`real`), as a
first offer after that Present (`late`: with a Present one or two Presents ago through retained
color, or under frame generation with the current one), as a later offer of the same snapshot
(`reoffered`; all three inexact), `stale` (older than
the retained history), `other`, or `none` (no ready HUD-less capture); `generated` is always 0
since 10-05, when HUD-less pairing stopped holding Presents, and stays so that old and new lines
parse alike. Unchanged failed
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
The frame prepared just before the close has no controls to capture; it is composed under an
empty panel and published. It previously withdrew the export, and the host showed the captured
desktop for two seconds.
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
If the game renders its eyes at another size with the same aspect ratio as the stream (within
0.5%), the host resamples both eyes to the stream before YUV conversion and logs once per
generation `ReShade SBS: game eyes WxH are scaled to the stream's W'xH' eyes`. Each output eye
samples only its own source half, so linear filtering never mixes the eyes, and the exact-texel
chroma conversion always reads an output-sized raster. Running the game at the stream resolution
avoids that extra pass and its softening. Eyes with a different aspect ratio would distort
disparity, so the host keeps the stream in 2D and logs a warning naming both sizes. The host
decides that from the producer's published source, with or without a consumer, before it
requests a ring: it writes no consumer nonce for such eyes, and withdraws its own nonce from a
live generation that stops fitting (the producer then deactivates with `no_consumer` and frees
its idle ring), so the game neither allocates the shared ring nor packs stereo for a stream that
stays 2D. A fitting size requests a fresh ring under a new nonce. Until the 10-05 review the
nonce was written first, and an aspect-mismatched stream kept the game packing and fencing
every Present into a ring nobody read.

Normal installation omits `-ShaderDirectory` and uses the GPU renderer embedded in the add-on.
For comparison testing only, the exporter also accepts annotated SuperDepth3D and independent
SunshineDepth3D reference exports. These require an explicit `-ShaderDirectory` with their matching
sources; the installer disables native Game 3D and enables only the selected reference technique.
To return to native rendering, rerun the installer without `-ShaderDirectory`, then enable Game 3D
in the add-on panel if it was previously disabled. Existing native preferences are preserved.

### UI decision framework

The target model for automatic UI protection, and the staged roadmap toward it. Replay `xfail`
labels, `KNOWN_TODAY` sequence cases and readiness-report checks name the stage that changes an
outcome and the rule ID it applies; thresholds and validation evidence stay in the sections above.

UI is what is drawn on top of the finished scene. **Opacity ruling.** One rule serves every game:
for the best accepted source, each pixel's pin weight is `saturate(8 * a)` at any coverage. A
semi-transparent backdrop (the Witcher 3 sign wheel's backdrop under its solid wheel) and an opaque
full alpha from an accepted source (Expedition 33 menus) therefore pin flat, whatever D reads: the
UI is not distorted and the scene behind it is flat. A source that has only ever been opaque is
never accepted. UI distortion is the priority.

Eight modules each own one question, and data flows one way: capture, per-frame validity,
acceptance and selection, hidden-scene guard, hold, pin weight. Diagnostics only read.

| Module | Owns |
| --- | --- |
| M1 Evidence capture and identity | Snapshots with proof, real-frame identity, exactness, encoding and source signature; one slot per signature. Reads no trust. |
| M2 Producers | Each candidate's coverage image, per-frame validity and class; pair comparability; pairwise judgments. Reads no trust or history. |
| M3 Acceptance ledger | One accepted bit per game and source signature: earning, revocation, provisional persistence, Forget and the manual override; since fix 1 also, per offscreen layer signature, the proof that the layer is the pre-UI scene image (H1 (d)). |
| M4 Selector | Which accepted, valid candidate draws the mask this frame. |
| M5 Hidden-scene guard | The depth path's held D verdict: whether this frame's depth describes the image. |
| M6 Temporal hold | The decision a Present without its own reuses. |
| M7 Pin weighting | The pin weight of the selected coverage, or flat. |
| M8 Diagnostics | Status, warning, counters, report and replays; feeds nothing back. |

| Rule | Requires |
| --- | --- |
| E1 Proof and identity (M1) | A candidate exists only as a snapshot with proven state, boundary, real-frame identity, encoding (typed format and swapchain colour space) and signature; otherwise there is no candidate and a recorded reason. Each cleared target or declared resource is its own signature in its own slot. Over budget, a spare slot rotates across unseen signatures and records a budget refusal. |
| E2 Exactness (M1) | Only same-batch Streamline tags, with the same-batch Backbuffer tag as the final-image reference, with frame generation on or off; the one-way test also requires the presented frame to show the HUD-less scene unchanged, so UI composited after the tagged Backbuffer never contradicts its tag. Present counting only proposes a pair (a snapshot's first offer with the Present after the tag; under frame generation a late first offer and every later offer of it with its own Present, without it with the retained Present after the tag): it is inexact, validated by V2's pixel test alone, never full-frame UI and never a judge. Frame tokens and FG interposers do not make a pair exact (S3's frame identity was removed). The one-frame-late layer copy is inexact opacity evidence: it may draw, but never judges or pairs. |
| V1 Opacity validity (M2) | Finite, in-range alpha with at most 1% invalid pixels, at any bit depth; a cleared layer must also pass the premultiplied bound: its pixels beyond it are uncovered, and invalidate the layer when they lie on more than 1% of the frame unless they lie on at most 5% of it and its opaque pixels outnumber them. Presented and Backbuffer alpha are checked for range only. The tolerance does not depend on trust. |
| V2 Change-set validity (M2) | A difference is evidence only with the threshold from the two snapshots' own encodings, and only a valid change set is: the changed set passes the tile test above when partial, changes no pixel of a lit HUD-less image in clean tiles (empty, since revision 8, which an accepted pair decides as an empty mask of its own), or, from an exact pair whose HUD-less image is lit, is nearly the whole frame. The middle band (a pair mispaired with interpolated or another frame's colour) and noisy pairs are invalid. |
| A1 Earning (M3) | Acceptance is keyed by game and source signature (with the swapchain colour space), not by FG mode. A declared source (UI alpha or color tag, or a HUD-less pair, exact or not, by its partial change set) is accepted by its first valid selective sample; an inferred source needs the steady selective run above, consecutive and bounded (a selective sample more than 2 s after the run's last one restarts it). A source that is never selective is never accepted. A sample does not count while a declared alpha is offered but invalid. Holds and manual inputs never earn. An offered layer without coverage earns its signature's pre-UI proof (ledger key `pre_ui:<format>:<space>`) like an inferred source: three samples over 2 s in which it equals the presented frame at eight times their pair threshold on at least 90% of pixels and is lit on at least half; a mismatch neither withdraws it nor restarts the run, and no judge revokes it. |
| A2 Revocation (M3) | Three contradictions within 2 s by valid same-sample evidence of stronger provenance (an accepted declared alpha or an exact change-set), whatever the drawing rank: one-way disagreement on lit pixels unchanged against both the paired Backbuffer and the presented frame, or declared-versus-inferred coverage disagreement. Agreeing samples do not reset the count. Every accepted alpha is judged by exact one-way contradictions, declared ones included (selection revision 10) when captured in the exact pair's tag batch; the declared alphas' coverage judges inferred alpha only; the one-frame-late layer copy is not same-sample evidence (E2), so neither judge reads it. Forget also revokes. Ambiguous or invalid samples never revoke. |
| A3 Persistence (M3) | Restored acceptance is provisional and lapses unless earned again within 60 s of testable time: for every kind the clock counts only the gaps, capped at 250 ms, between consecutive samples that could earn or refute the source (offered, valid and not full: alpha below 90% of the frame, a partial or empty change set), so other samples, absence, manual periods and gaps without samples never count beyond one capped gap (a full menu can neither earn nor refute it, so Stellar Blade's presented alpha no longer lapses during long menu visits or FG-on play, nor a HUD-less pair through mispaired samples). A restored pre-UI proof's clock counts the same way on its testable samples (its layer offered without coverage while the presented frame's evidence is valid and visible), so it lapses after 60 s of testable time without a match. Legacy per-kind entries are discarded; Forget clears the game's entries, pre-UI proofs included. |
| S1 Selection (M4) | Among accepted, valid candidates the first in draw order wins: opacity before change-set, then declared before inferred. An unaccepted or invalid candidate never blocks another, except that an offered, accepted declared alpha blocks inferred alpha (when it is invalid, T1 applies). Nothing qualifies: no mask, with the reason of the highest-ranked refused candidate. |
| S2 Manual (M3) | Off offers nothing; the filter restricts offers; On accepts the filtered valid candidates for this session only, without persisting, earning or revoking. |
| H1 Hidden scene (M5) | With valid depth, a held hidden D verdict (two hidden samples enter, renewals extend, a visible sample releases) and an informative full claim (from an accepted source, a layer proven cleared transparent this frame, an exact full change-set, or a pre-UI scene image on which D reads visible while D on the presented frame reads hidden: the declared HUD-less image, or an offscreen layer without coverage whose signature the ledger holds proven, A1), the frame is flat whatever M4 selected (a winner already flat is relabelled 8 too). A visible verdict refutes that signature's full claim until it shows below 99% opaque. Invalid D acts on nothing; only a scope change clears D state. |
| H2 Still screen (M5, removed) | Fix 2's flattening of still SDR screens without a UI source, removed in selection revision 9 (**Still screens without a UI source (H2, fix 2): removed**); the ID is not reused. |
| P1 Pin weight (M7) | `saturate(8 * c)` of the selected coverage at any coverage (binary for change-sets); 1 everywhere when H1 says flat. |
| T1 Hold (M6) | A Present without its own fresh decision uses the last real decision; a real frame without one reuses the previous real frame's decision once, then has no mask, except that a mispaired inexact re-offer of the previous render's HUD-less snapshot keeps the held decision for as long as that snapshot is re-offered, whatever tags come with it. No multiplier constant and no time bound; real frames are identified by Present counting (S3's real-frame identity was removed): a Present that offers nothing within the reported generated count of the last UI tag is generated, and so, under FG, is one that re-offers exactly the snapshot identities of the last detecting Present (no current colour or inexact HUD-less image); one that offers any other input is real. |
| F1 Fail safe and diagnostics (M8) | No qualifying source gives no mask with a named reason and refused candidate, the panel warning and exact counters. Status freshness is keyed on the scope and the winning accepted candidate. Diagnostics feed nothing back. |

Scope (runtime, device, epoch, viewport, size, colour mode and encoding, but not the FG multiplier)
is an M1 identity property: changing it drops snapshots, pairs, holds and D state. An observation
revision (a depth observation loss) is not identity: it drops the previous revision's captures and
samples, but the T1 chain keeps its last real decision and the hidden-scene guard its D state.
Acceptance changes, Forget and inactive frames never clear D state either. The layers' pre-UI
proofs are ledger state, not D state: scope changes and FG toggles keep them, and Forget clears
them. Cost is a
constraint, not a rule: given the ledger state a frame's decision does not depend on the sampling
cadence, snapshot memory and copies have bounded budgets, and a new capture boundary is enabled only
after a Present-interval A/B shows no frame-time cost.

Open questions that may move labels: whether D's hidden bound (0.15) leaves enough margin to the
gameplay stress minimum (0.161), where validation covers 16:9 only: claim (d) needs the pre-UI
image to contradict the presented reading, which fog, blackout and dark grain do not do, but claim
(b) on an opaque cleared target still depends on the presented reading until its first visible
verdict, so fog, blackout and flashlight scenes (Hogwarts Legacy's forest, Dead Space) and non-16:9
output need dumps of the uncovered hidden screens before the informative-claim gate is relaxed in any way;
whether the flat gameplay after a Stellar Blade SDR menu closes (up to one sample interval plus the
readback, because claim (d) on its scene target stays true) needs a per-frame pre-UI difference
(S5) or a shorter sample cadence while a pre-UI hold is active; whether the entry of about 300 ms
where a claim appears with the menu (200 ms before S2b) is acceptable live; whether a first-session
menu drawn only into an unaccepted, all-opaque UIAlpha, which no longer claims, needs another
claim; whether a first session's menu opened before 2 s of lit gameplay, whose scene layer is
still unproven, needs another proof (later sessions restore it); whether the remaining layer-proof
gaps (a cut straight from proven gameplay into a scene that effects after the layer push hidden,
with no ambiguous sample between, and a late post-process input equal to the presented frame on 90%
of pixels at the coarse bound) need a check beyond D; whether dark pre-UI images such as Stellar
Blade's SDR loading screen (a proven layer lit on less than half of the pixels under a hidden
presented frame) need a rule of their own (texel 11's shadow statistics that measured them were removed);
whether a wrongly
accepted inferred source with no declared or exact judge needs more than Forget and the provisional
lapse (until S4 this includes a wrongly accepted layer, which the one-way test does not judge);
whether a wrong declared alpha that once read selective needs a judge where no exact pair is
offered (since selection revision 10 an exact pair of its tag batch judges it one way, as every
alpha but the layer copy); whether the premultiplied bound also applies to declared UI color tags; and whether a generated Present over untagged UI should
use the masks of both real frames it interpolates (it needs a moving-HUD FG dump).

**S3 snapshot ticket: removed.** S3 gave every UI candidate snapshot an immutable ticket with
GPU frame stamps from two per-device clocks (a present clock advanced at every foreground Present
and a token clock written after each Backbuffer tag snapshot), paired an image with its final-image
reference only by equal labels, had the CPU propose labels (`b2` words 6-9) that the GPU verified
against the stamps at `t9` (decision texel 12 `.z`/`.w`, five identity counter words), and would
have held T1 by token. It ran only as a diagnostic shadow behind the Diagnostics switch and never
decided. Selection revision 9 removed it together with H2, which shared decision texel 12: the
tickets, clocks, stamps, proposals, the `Sunshine UI identity` and `Sunshine FG interposers` lines,
the token-batch pairing and the test-only identity override are gone. Its identifiers stay
reserved and are never reused: `b2` words 6-9, decision words 50-51 (with H2's 48-49, all of texel
12) and counter words 31-35. HUD-less pairs are
exact only by a same-batch Backbuffer (E2) and real frames are identified by Present counting
(T1). The layer's queue watch, which S3 introduced, remained until WP1b replaced it by each live
copy's own executing queue: the live layer copy is in queue order when it ran on the presenting
queue, and since the layer cross-queue fence one run on another queue is ordered by that queue's
fence. Its dumps record `executed_queue`, `presenting_queue_order`, `read_order` and `fence_value`
(before WP1b `foreign_present` and `queue_mixed`, the scope-wide watch's sticky verdicts).

S0, S1, S2a and S2b are done; S3 was removed. Fix 1 after S2b (selection revision 4) replaced the
guard's D proof of the scene layer, which an FG suspension's viewport change cleared and a fade-in
sample withdrew, by the ledger's pixel proof, so Stellar Blade's SDR settings and full menus are
flat with FG suspended. Fix 2 (selection revision 5) added H2 for still SDR screens without any UI
source, such as Stellar Blade's SDR loading screen, as a shadow by default. Fix 3 and fix 4
(selection revision 6: the pre-UI change set and pin only UI) were removed by user decision, and
selection revision 7 decides as revision 5. Selection revision 8 added the empty change set, and
revision 9 removed H2 and S3, so it decides as revision 8 without source 11. Selection revision 10
(WP1a, the current one) keeps the held decision across HUD-less re-offers (T1), judges the declared
alphas one way within the exact pair's tag batch and no longer the layer copy (A2), counts both on
status samples only, bounds the layer's glow tolerance (V1), restarts an inferred earning run after
more than 2 s between selective samples (A1) and has H1 relabel every winner, flat ones included. S2b answered whether an exact HUD-less pair decides a
full change-set before its first selective sample: it decides 6 only once accepted (P1), and before
that its full claim acts only through H1 (c), under a held hidden verdict.

| Stage | Scope | Validation |
| --- | --- | --- |
| S0 | Behaviour-neutral: strict replay labels with xfail cells, the sequence replay, exact UI counters with the report reading them, dead-code removal. | Replay identical, sequence replay passing on today's code, counters reconciling. |
| S1 | Decision structure: separate candidate slots for the tagged UI color and the offscreen UI layer (E1), the S1 selection predicate with the declared-alpha block, V1 validity independent of acceptance, A1 acceptance keyed by signature for every deciding source (declared sources by one sample, inferred ones by the steady run, the earning void) with the legacy discard, the manual override (S2), and comparable pairs from the pair's own encodings (V2). | No cross-signature or cross-colour-space acceptance in the sequence replay; acceptance stable across HDR/SDR and FG switches; `untrusted_inferred` and `presented_over_dedicated` at 0. |
| S2a | Provenance revocation and Forget (A2), provisional persistence (A3) beyond the S1 key and discard, the identity hold (T1) with the GPU grace, refusal reasons, the winner-keyed status and the shadow toggle (F1; the toggle went with the first-run shadow in selection revision 9), selection revision 2. | Revocation, persistence, Forget and hold cases in the sequence replay, the S2a T1 cases strict; single-frame replay outcomes unchanged; no hold across a scope change. |
| S2b | The hidden-scene guard H1 in the depth path (M5, `game3d_scene_guard.h`) replaces the layer and HUD-less routes (8 and 9, now 8 alone), with informative claims including the pre-UI scene image, per-signature refutation, two-sample entry, identity-only clears and selection revision 3. | No H1 hold over a scene D reads visible (`full_d.visible` counts releases only, at most `scene.released`); accepted whole-frame alpha and accepted exact full change-sets unchanged (P1), except that H1 takes over an accepted winner with transparent pixels under a held hidden verdict; Stellar Blade's SDR settings flat with FG suspended once gameplay proved its scene layer; the S2b sequence cases strict. |
| S2b fix 1 | The layer's pre-UI proof by pixels in the acceptance ledger (`pre_ui` keys, earned by matching samples, persisted, provisional with a testable-time clock, cleared by Forget) instead of the guard's D similarity; claim (d) for a layer without coverage and a proven signature; texel 11 with shadow statistics; selection revision 4. | Stellar Blade's SDR settings flat with FG suspended across the FG suspension's identity change and a mismatching fade-in sample; a menu before any gameplay flat from the second session; a target that never matches never proven; the SDR gameplay dumps match (99.3-99.8%), the settings and HDR dumps do not; every other replay case unchanged; loading screens unchanged (shadow statistics only). |
| S2b fix 2 (removed) | H2, still screens without a UI source in SDR Auto (M5): stillness counts per D cell (texel 12, the previous-luma texture), source 11 through `b2` word 5, counters `decided.11` and `still`, the `UIFlattenStillScreens` switch defaulting to a shadow that only logs, and selection revision 5. Removed in selection revision 9 with S3; every identifier stays reserved (**Still screens without a UI source (H2, fix 2): removed**). | Every replay case unchanged; its sequence group was removed with it. |
| S2b fix 3 (removed) | The pre-UI change set (candidate `0x100`, source 12), the refine rule, the lit rule for partial change sets, the change-set shadow and its bit-plane passes, the ring's per-requester slots, `UIPinChangedPixels`, the `Sunshine UI change set` line, the report's `Pre-UI change set` check and Dump 3D artifacts 43-45 (selection revision 6). Removed by user decision (rejected in SDR and HDR); selection revision 7 decides exactly as revision 5, and every identifier stays reserved (**Pre-UI change sets (fix 3) and pin only UI (fix 4): removed**). | Every replay case and sequence group that existed before fix 3 gives the outcome of fix 2 plus S3, A3 and the Dump 3D overflow fix; the 24 fix 3 and fix 4 replay cases and their groups were removed with them. |
| S2b fix 4 (removed) | Rule P2, pin only UI: the darkening passes, planes 6-8, statistics rows 192-207, decision words 62-63, `b2` word 5 `0x100` and `0x200`, `UIPinOnlyUI` and its panel row, the `Sunshine UI darkening` line and the report's `Pin only UI` check. Removed by user decision with fix 3. | As fix 3's row. |
| S3 (removed) | Snapshot identity and exactness (E2, T1) by snapshot tickets with GPU frame stamps, verified on the GPU and shipped as a shadow behind the Diagnostics switch; removed in selection revision 9 with H2 (**S3 snapshot ticket: removed**). HUD-less exactness stays by the same-batch Backbuffer and T1 by Present counting; the layer's queue watch remained until WP1b replaced it by each copy's own executing queue. | Every replay case unchanged; its two sequence groups were removed, and its two KNOWN_TODAY T1/E2 cases are now `KNOWN_LIMIT` lines of Present counting. |
| S4 | Same-frame cleared target: the offscreen UI layer at its write end (D3D12) or at Present (D3D11), replacing the one-frame-late copy; one slot per cleared-target signature with a budgeted census rotation (E1, moved from S3). | No one-frame tear on moving HUD; Present-interval A/B. |
| S5 | Back-buffer pre-UI snapshot, evidence-gated. | Census and shadow logs per affected title; bounded added cost. |
| S6 | Evidence sweep, no code: every game in each colour mode and FG mode, scRGB, non-16:9, FMV and a first boot into a menu. | Every cell has a labelled dump and a report without FAIL counters. |

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

While a dump request is armed, the first clear of up to three
[offscreen UI layer](#setup) candidates is also copied as
optional artifacts `ui_layer_candidate_0` to `_2` (IDs 40-42) with RGB and alpha previews, and
`ui_layer_census` lists each target's size, format, clears seen while armed and capture status.
The census shows every qualifying target; `active` marks the one the live tracker chose, which the
dump's automatic candidate set names as `ui_layer`.

**Census ordering.** A census copy is recorded into the game's list at the clear, like a live copy:
created in the shader-resource state with the same leading barrier, so a list executed again
records valid transitions. It follows the live copies' queue order (`ui_layer::copy_order`: the
same carriers, submission hook and
[layer cross-queue fence](#diagnostics-switch-and-per-present-cost)). The dump reads it only with
CPU proof that its write completed before the dump's own reads on the presenting queue, never with
a GPU wait (`ui_layer::census_verdict`). On D3D11 it ran on that queue (the immediate context, or a
deferred list the immediate context executed). On D3D12 the submission hook ran after its latest
execution, it ran on the dump's queue or the CPU saw its queue's fence reach
the value signalled after it, and no game list still carries it: a list executed again would write
it while the dump reads it, and games reset their lists within a few frames as their allocators
rotate. The capture is deferred while any copy is undecided (`ui_layer::census_pending`: still being
allocated, not executed, held for the hook or its fence, or still carried by a list), for at most
`ui_layer::census_wait_ms` (1000 ms) after the dump was armed; Presents keep presenting and the
presenting queue never waits meanwhile. Then the dump is taken, and each copy without proof is
omitted (no artifact, `captured: false`) with a status that says why: `not_executed`,
`reset_unexecuted` (every carrying list reset before running it), `awaiting_submission`,
`awaiting_fence`, `awaiting_list_reset`, or `unordered` when nothing can prove it, with
`unordered_reason` `mixed_queue` (run on two queues), `submission_not_observed` (the hook did not see
it, or its list was reset before the hook ran) or `no_fence` (run on another queue than the dump's
with no fence value, or its fence record was released). Each omission logs once per status and
reason (`Sunshine UI layer: a Dump 3D census copy was not proven complete before the dump read it
(...)`). An omitted copy that a list still carries or may still write is destroyed after 10 s
instead of 2 s. A proven copy keeps the status `captured_before_clear`. Every row records
`read_order` (`queue` or `fence_passed`; null without proof), `fence_value`, `executed_queue` and
`presents_since_copy` (Presents since the copy, as for the live layer); the census records
`ordering: proven`, `settled` (false when a copy was still undecided as the dump was taken: the
deadline passed, or a target was first cleared just before the capture) and `wait_ms` (from arming
to the capture). The other statuses are unchanged: `copy_allocation_failed`, `allocating` (the copy,
allocated outside the layer's lock since WP1b, was not yet published), `transport_budget_exceeded`
and `other_device`. A census without `ordering` comes from an older build, which handed every copy
to the dump as `captured_before_clear` whether or not its list had run: a copy run on another queue
than the presenting one (Stellar Blade's layer, The Witcher 3 with FG on) could be read before or
while it was written, so check such an artifact's alpha preview before using it as replay evidence
(`ui_detection_replay` binds it at `t7`). Cross-queue census copies stay supported: the D3D12
provider fixture (`reshade_game3d_native_provider_runtime_test` with the frame-generation
interposer) reads a copy gated on a second queue in fence order, with the layer's content, once its
fence reached it and its list was reset, with no Present waiting meanwhile, and omits one still gated
at the deadline (`awaiting_fence`) and one whose list never executed (`not_executed`).

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
in the separate 32-slot colour/mask pool that cannot consume the depth pool. There they have a
fixed byte budget of their own and take only slots without live UI storage, so a dump never
evicts or crowds out live UI snapshots ([Streamline depth selection](#streamline-depth-selection)). A request keeps the first
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
artifact PNGs in a `previews` subdirectory (Pillow required). `--pin-metrics` measures UI pinning
in the package, or in a replayed `final_field.bin` given with `--pin-field`, against the unpinned
field (the horizontal limiter applied to `vertical_field`): for each `--pin-band FIRST:LAST`, the
most columns in one row pair with a new vertical step above 0.5 px; the scene tear (new steps
outside UI rows, weighted by horizontal luma texture and scene visibility); the flattening, mean
`(1 - alpha)|F - h|` in pixels; and the torn glyph pixels when the solid light bodies of the
one-frame-late UI layer move against the unmoved pin by 1, 2, 4 or 8 rows up, 4 or 8 rows down, 4,
8 or 16 columns right, or 8 rows up and 8 columns right. Fields are compared in double precision, and a 1e-4 px tolerance keeps float32
rounding of the pin collar's exact half-pixel bound from counting, so figures computed in float32
with a strict 0.5 px threshold can differ slightly. It supports the fixed and display-fraction UI
planes, which sit on the screen plane whenever the shader's camera admission rejects the frame's
constants. Visibility, UI rows and flattening weigh by the consumed mask's alpha, which the
report's `alpha` names.

With the Diagnostics switch off a dump records the same artifacts and fields. A dump armed on a
mono Present still records the full depth
conditioning. In an HDR10 session exporting the PQ transfer, the `sbs` artifact is R10G10B10A2 PQ
code values (`sbs_transfer` 3) and there is no `linear_color` artifact.

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

The add-on updates `Sunshine_Calibrated`, `Sunshine_RawAnchor`, `Sunshine_RawGain` and
`Sunshine_DepthRect` together after its depth-capture callback and before the effect runs.
`Sunshine_DepthReady` (`bufready_depth`) comes from the shared readiness cache
([Streamline depth selection](#streamline-depth-selection)), written only when readiness changes
and again into an effect rebuilt by a reload. A separate per-present proof prevents a missed update
from exporting stale calibration. Current-frame capture readiness and older calibration evidence
are distinct.
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
A clear or replacement is an observed fact: later evaluations see the new tag set, while earlier
evaluations and their copies keep the tags they were given. Repeated successful null clears are
idempotent. Tags and cameras recorded before a failed SDK call or an observation loss are never
revalidated for later evaluations.
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
Generic selection retains the counters needed for its actual capture decisions.
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
ownership on the consuming queue independently of display readiness. A source is identified by
provider, epoch, logical source and viewport. It is live while its newest nomination is younger
than the source age below (250 ms). Ownership arbitrates only between live sources, and only a
delivering owner, one whose last copied frame is younger than 250 ms, keeps its own live source:
while that source is readable, or while a valid successor of its last copied frame is pending. A
short pipeline delay, or a stray readable capture from the other provider, therefore does not
alternate selection; the owner's already copied frame is not a successor. Otherwise readable SL is
preferred over readable NGX, and the selected source becomes the owner; an SL DLL, metadata-only
nomination or pending first copy cannot replace working NGX. That preference also covers an
established SL owner whose readable frames are never copied (failed display preparation or copy):
it is not delivering, yet keeps selection over readable NGX until its source is revoked or stops
being readable. When neither provider is readable, a delivering owner keeps selection unless its own
newest evidence has expired, even when its current attempt failed, was rejected (also before a
capture existed), names a replaced source or is ambiguous: the display may hold its last copied
frame, and the other provider's capture without readable pixels, such as an inner NGX evaluation
still in flight, takes no authority. Once the owner has not delivered for 250 ms, or when no owner
is established, a live source whose newest snapshot is pending is reported in front of a failing
(failed, unsupported or ambiguous) one, and any live source in front of an expired one; an expired
owner is never reported in front of a live source. A failing owner that has stopped delivering, or
an expired one, would otherwise mark an interruption at every Present and so keep the pending
source's completed snapshots unreadable. Expired or undelivered owner evidence therefore holds
nothing; failed, rejected, missing, replaced (a new logical source of the same provider) or
ambiguous evidence holds only a delivering owner's selection, and only while the other provider is
not readable. Metadata observed after a copy (camera reset, tag change, observation loss) does not
make that finished copy obsolete.
Explicit enabled FG retains its separate mandatory SL scope described below. Provider choice is
frozen within one presentation, and each packet retains its own encoding, camera and logical source.

When no alternative API source is usable, or while the SL preference above retains a non-delivering
SL owner, unsupported format/state or failed display creation produces mono while retaining the
established API selection. A live source whose newest snapshot is pending can use its own newest
unconsumed completed snapshot under the ordering, freshness and interruption rules below; otherwise
it also produces mono. Neither case starts Generic ranking or scene-derived recalibration.
Ownership is not a precondition for that completed snapshot: in a pipelined game the newest capture
is still in flight at every Present, so the completed snapshot is the only way such a source becomes
readable and is established. Requiring an established owner first would lock such a source out
while another provider's expired capture holds ownership, when no owner is established (after a
reset or a new generation), and when the same provider changes source identity, for example when a
DLSS quality change re-creates the NGX feature at a new render resolution. Stellar Blade showed the
first case: one late SL capture took ownership after FG Off, expired, and left its pipelined NGX
depth unavailable for minutes. Invalid pointers,
device/extent/lifetime mismatches and an unsuccessful or unsubmitted first evaluation cannot
establish authority. A loaded DLL alone cannot establish it either. Once established, temporary
missing or failed observations retain ownership. Explicit source release, disable or lifecycle
teardown ends it, and so does one second without any evaluation of the established source (a
game that switches DLSS to TAA, or a cutscene without DLSS, without releasing the feature); the
queue then returns to Generic through its usual selection. A silent owner never blocks another
live API source.

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
again after effects reload and published only when readiness changes. Readiness never rebinds
`DEPTH`: ReShade 6.8 waits for the whole queue on every binding update once an effect declares
`DEPTH`, so Generic rebinds only when its view changes. A logical Generic source rotating through
several physical members binds one stable copy per format and size instead of each member's view.
The add-on's renderer reads the selected capture directly, so the copy exists only while effects
are enabled and some technique is enabled (any enabled technique counts as a possible `DEPTH`
reader); otherwise a
rotating source binds nothing. One depth copy is recorded per new ready capture.

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
until its private producer fence completes. The add-on never inserts a foreign-queue Wait for a
depth capture: the producer can already depend on future consumer work, so a reverse wait could
deadlock the game. Flushing an immediate command list cannot prove that the application's queue
graph is acyclic. (The offscreen UI layer's cross-queue copy follows the same rule since 10-06: it
is offered only once the CPU sees its own fence completed, after a GPU wait for it froze a game; see
the layer cross-queue fence under
[Diagnostics switch and per-Present cost](#diagnostics-switch-and-per-present-cost).)
A failed ordering check declines the copy before any read is recorded. There is no CPU depth-fence
wait or pending-copy flush. Device removal is not completion, and actual GPU
completion is still required for allocation retirement. The selected capture stays frozen across
the pass. While a same-source successor is pending, SL and NGX may advance to the newest unconsumed
completed snapshot: on the consumer's own queue one already submitted there (queue order, as for a
current capture), from another queue one whose recording retired and whose producer fence
completed. Its original sequence, timestamp, projection, jitter and feedback travel with
its pixels. Provider, epoch, logical source, viewport, FG role and known
depth layout must match, and the snapshot must be within the source age. Later camera resets and
observation bookkeeping do not revoke its eligibility. The pool retains one eligible completed snapshot while recording its successor, so
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
not a fixed global table. Lists seen through the shared barrier hooks therefore cannot exhaust a
128-entry registry or leave entries behind when ReShade misses destruction.
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
An independent FG Present cannot expire this synchronous tag-call lifetime. The same holds for an
FG input `ValidUntilPresent` depth tag copied inside its own tag call: Frame Generation presents
earlier frames on another thread, and a Present landing between reading the tag and recording its
copy used to refuse that real frame's depth as `unsupported_lifetime`. The Witcher 3 hit this on a
fraction of real frames with multi frame generation. A tag copied later, at an evaluation or as a
preservation copy, still expires at the next Present. Other lifetime rules, recording retirement
and the shared consumer-ordering checks above still apply.
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

When a presentation has no newly readable depth (a generated FG frame, a pending, failed or
missing successor, or an evaluation without depth), the provider holds the last real depth already
copied into its private display texture. One rule serves SR, FG and NGX. The completed-snapshot
selection described above supplies newer pixels first; the hold reads only the previous private
display copy, and copying a new snapshot still requires its ordering conditions. The hold ends when
the provider names a different source (provider, epoch, logical source, viewport, or an older
sequence), when a selected snapshot has a different layout or encoding, when the enabled FG scope
changes, or at the age bound below.
Explicit source interruptions advance the existing admission watermark,
preventing pre-interruption captures from returning.
Reused real depth retains its applied gain, zero plane, crop and jitter and does not advance
range learning or gain adaptation. It takes the scene exactly as the real frame resolved it,
placed or still calibrating, so a generated frame shows depth whenever its real frame does.
Requiring a placed scene used to flatten every generated frame while the real frames kept their
depth during range calibration (the first second after FG starts) or after a camera cut; with
multi frame generation that flickered up to four flat frames per real frame. Reducing user strength, including setting zero, applies
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
`sunshine_scene_depth::maximum_source_age_ms`. Display-depth reuse also requires the same runtime
queue and display allocation, unchanged color dimensions and any known depth dimensions/crop; under
enabled FG the retained copy must belong to that FG scope. Failed consumer copies, loss of the
source association (for example FG Off before an ordinary source exists), focus/technique/lifecycle
changes and expiry invalidate reusable history until a fresh copy succeeds. A nomination gap cannot
prove an as-yet unknown depth layout; its same-source reuse remains subject to the short bound and
color dimensions. Under required FG a present without a nomination still names the FG scope but no
sequence; that gap holds the retained copy instead of reading as a source change. Before this, The
Witcher 3 with FG on (one to four generated frames) dropped depth for 94-172 ms whenever one real
frame's FG tag capture was refused (`status=unsupported_lifetime`, reported as
`source_changed`). Source pointers may rotate normally without invalidating the private copied depth.

Metadata bookkeeping never discards finished pixels. An SL camera reset, invalid or missing camera
constants, a null or replaced tag and an observation loss affect only later evaluations: a reset
starts a new temporal history for its own viewport, an invalid camera removes metric scale, a
cleared tag removes that kind, and a loss makes metadata recorded before it unusable for new
captures. None of them revokes an evaluation already captured, its completed snapshot or the
private display copy. Only a real observation loss (contended or failed hook bookkeeping, a failed
SDK call or lifecycle) advances the Streamline observation revision. A frame-token request that
finds another thread writing the token table retries briefly (a few microseconds, still
non-blocking) before counting as `tokens_busy`: The Witcher 3 requests about 30 tokens a frame
from several threads, and each collision dropped depth for about five presents and reset UI
placement, most visibly in its settings menu. Each viewport's feedback revision counts only what
can hide or end its temporal history: lost constants observations, observation restarts and its
own resets. Other losses (a busy lock, a failed SDK call) leave scene gain history intact. A Frame
Generation options call publishes its mode with the same brief retry, since a game may set its
options only once. The reset feedback discards old scene measurements
while retaining established gain and zero placement; it does not force a valid current depth frame
to render mono. The Witcher 3 exercises both cases: another viewport sends reset constants every
frame, and the depth tag is cleared after each evaluation.
Depth direction is a fixed convention of a viewport, unlike its matrices. When a capture has no
frame-correlated camera (The Witcher 3's FG depth arrives this way while its camera is valid), a
valid camera of the same viewport seen within the source age still supplies the
direction. Projection, metric scale and jitter continue to require the frame's own camera; an
invalid or older camera supplies nothing. Without any direction, relative calibration waits for
clear-value evidence and Game 3D stays mono.
A capture without its own camera between captures that have one does not switch placement to
the raw controller. When the projection controller placed the preceding presentation from the same
source (epoch, source and viewport) within the source age, the capture renders with that scene's
coefficients, gain and zero plane, advances no calibration and carries no jitter. The Witcher 3
misses the camera of a few SR evaluations each second: each such frame previously rendered mono,
because raw calibration never completed there, and restarted the 500 ms strength ramp on return.
The `Sunshine 3D Streamline scale` line counts these frames as `camera_missing_holds`. A capture
older than the source age, after a depth gap or from another source still takes the raw controller.
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
reading expired depth. Once per Present, every other reusable record drops its game resource,
device and recording references and keeps only its private allocation; idle allocations beyond
256 MB are released. A hitch therefore cannot leave the pool holding old game depth buffers.
While API depth owns the pass, Generic's unselected rotation members retire on their usual delay.
In the 32-slot colour/mask pool, live snapshots (`record_local_texture`) and Dump 3D copies keep
separate byte budgets: dumps 256 MB, live snapshots eight of the requested size and at least
256 MB. A dump never counts against or evicts live storage: it takes only a slot without live
storage and is otherwise exhausted. Idle live storage is released 2 s after its last snapshot.

Capture acquisition returns source authority separately from the packet's pixel readiness.
Its typed decision carries the selected identity and repeated/pending continuity. The capture
owner alone classifies SDK success, invalidation, recording retirement and producer completion.
`depth_cache_update.h` turns those facts and the last successful copy's value description into
one source decision: `copy_fresh`, `hold`, or `invalidate`. It owns source identity, FG scope,
depth layout/encoding and source-age policy; it does not consult observation revisions.

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

`Sunshine SBS output` reports cumulative per-runtime Game 3D publication counters at most once
every five seconds, with or without FG: `published` and, of those, `fg` with frame generation
active. `scene_flat` counts publications that had depth but showed the colour frame because the
scene was not placed or its strength blend was zero; `scene_fading` counts those still ramping in.
`published_fresh_depth`, `published_reused_depth` and
`published_depth_missing` distinguish actual shared-ring publications using fresh, reused or
unavailable depth. Their disposition is retained with the pending export copy. Missing depth is
split by reason: `unavailable` (the provider had none this Present, for example a source switch,
an FG toggle or expiry; `Sunshine depth readiness` names the provider's reason), `reuse_after_gap`
(a generated or pending frame whose previous presentation had no depth) and `reuse_other_source`
(its depth belongs to another source than the previous presentation's scene). These classify
depth availability; `scene_flat` and `scene_fading` classify placement. `dropped` counts
Presents with an attached consumer that found no reusable export slot (the line is also written,
at the same rate limit, from a dropped Present, so a ring that drops every Present still reports
it with `published` frozen), and
`overwritten_unconsumed` packs written over a ready slot the consumer never claimed (it took a
newer frame, or none yet); see the [GPU handoff contract](#gpu-handoff-contract) for the slot
policy. Runtime reload/destruction resets these counters.

Game 3D captures color at ReShade's Present event, so it sees exactly the Presents that pass
through ReShade's swapchain wrapper. Whether frame-generated images are among them depends on how
the game's Streamline proxy and ReShade are stacked, not on Game 3D. A census of ReShade, DXGI and
Streamline frame counts (since removed) showed generated frames passing through ReShade in The
Witcher 3 once its DLSS-G modules were left unpatched. Dump 3D's `render_identity.presentation_ordinal` and Streamline frame numbers advanced one-to-one with FG
on in The Witcher 3, Expedition 33 and Hogwarts Legacy dumps.

`Sunshine depth readiness: lost/recovered` records the first availability transition of a
bounded diagnostic episode independently of the one-second status-log gate. At most four loss
episodes per second are admitted, each with a paired recovery; suppressed episodes are counted.
The record freezes source and newest-view ages at selection, the prior retained depth identity
before invalidation, FG scope, reset/observation revisions, copy outcome, source action/reason and
the cache's final reason. A zero timestamp is unavailable (reported age `UINT64_MAX`), not fresh
data. Explicit lifecycle/reuse invalidation clears
are also identified. This is read-only evidence: it adds no GPU pass, readback, wait, or extension
of depth freshness, and does not change source selection or stereo placement.

Each admitted readiness episode also logs `Sunshine depth observation evidence`. Its
`sampled_only` entry is a contemporaneous observation-revision sample; depth decisions never
consult it, and it is not proof of what caused the loss. Recovery repeats the frozen
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

The effect-based FG cases of `reshade_ngx_depth_runtime_test` are retired; their
`SUNSHINE_NGX_FRAME_GENERATION_TEST` and `SUNSHINE_FG_*` selectors now exit as setup failures.
That fixture and its separate-queue NGX regression run on native rendering with no installed FX;
see [source and submission validation](../tools/reshade/README.md#source-and-submission-validation).
None of these fixtures establishes a real game's FG intermediate-color correspondence.

Version-specific adapters translate to one plain scene-depth contract: provider and logical source
identity, optional validated projection, depth direction, resource extent, optional state hints and
capture lifetime. The capture owner resolves missing hints from its actual observed command state;
an adapter need not claim that a usable state was observed. A future AMD adapter should emit this
same contract; no AMD integration is implemented. The shared scale controller consumes admitted
frames and immutable samples, independently of source-selection and allocation decisions.
Camera projection is optional: invalid matrices never block an otherwise valid source or become
scale coefficients. Without projection, a known depth direction allows the shared adaptive raw
controller to follow the provider's logical generation/viewport. Direction comes from the API;
unknown stays mono. (API sources always copy at the middleware call; the former shared ReShade
preservation route, which could also supply clear-value direction, had no production caller and
was removed.)

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
recording. An observed conflicting or incomplete transition always rejects capture. An enhanced
(CommandList7) texture barrier that includes subresource 0 sets that texture's observed state to
its new layout's legacy equivalent (COMMON, GENERIC_READ, RENDER_TARGET, UNORDERED_ACCESS,
DEPTH_STENCIL_WRITE/READ, SHADER_RESOURCE as pixel and non-pixel shader resource, COPY_SOURCE/DEST,
RESOLVE_SOURCE/DEST), exactly as a legacy transition does, so the capture's own legacy copy
barriers start from it; other subresources keep subresource 0's state. A split barrier, or an
enhanced barrier into a layout without a legacy equivalent (queue-specific and video layouts),
blocks the affected resource for that recording until Reset (`incomplete_state`); unrelated
resources remain eligible, and enhanced global or buffer barriers change no texture state. Per-resource state
storage grows with the actual command recording and reuses its allocation on Reset; unrelated
resources cannot invalidate depth by exceeding a fixed entry count. An actual state-storage
allocation failure conservatively blocks the whole recording.

Admission blocks only what can change the copied resource's state, because every capture copies
where the SDK consumes its source (the tag call or the evaluation), where the game must keep that
source active and initialized. Aliasing barriers, named or wildcard (NULL meaning any placed or
reserved resource), change no resource state; they only move heap memory between overlapping
resources, so they do not affect admission. Resident Evil Requiem (RE Engine) aliases transient
memory every frame, and treating a wildcard alias as fatal had refused every DLSS depth capture
(`loss=wildcard_alias`). Bundles cannot record barriers, clears or copies, so `ExecuteBundle`
changes no state either. The generic bind-switch preservation copies at no SDK point and keeps its
own alias rule. Already-owned copies retain their normal lifetime checks. No state is guessed from texture type.
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
A slot whose discovery is refused is remembered: its table or original function lies outside a
loaded module image (for example another tool's trampoline), or its method already has eight
slots. Every later object sharing that slot would be refused identically, so the refusal
invalidates evidence once instead of on every call. QueryInterface refusals are per object and are
not remembered. D3D12Core 1.619 (the Agility SDK bundled with The Witcher 3) moves each command
list to a per-object method table inside the object (offset 0x5c8) at its first Reset, so no
method of such a list can be hooked through a shared slot; the Windows 10.0.26100 D3D12Core keeps
module tables, and lists created through Streamline's proxies keep the proxy's module table.

Command-list work is split by job. ReShade reports every command list it wraps (create, Reset,
Close, submission, render pass, destroy) from its own proxy, whose dispatch no runtime can
move. That lifecycle is the only source of a list's recording and of capture coverage: Reset starts
a recording, Close ends it, and a list is covered while it is open outside a render pass. A submitted list stays closed until ReShade reports its
next Reset, so a native Reset that bypassed ReShade never readmits an old recording. Lists ReShade
never reported are not covered, except the runtime's own list described below. Wrappers resolve to the native lifecycle through a private-data
tag, which proxies forward. Command lists hook only ResourceBarrier and, where supported,
CommandList7 Barrier: they supply observed resource states when they see the list. Otherwise
(refused or moved tables) the capture uses the state the game declared on its tag. ReShade's own
`barrier` event cannot replace these hooks: it reports a transition for the whole resource without
its subresource (Dead Space transitions the stencil plane of its R32G8X24 depth separately), drops
split flags, and reports an aliasing barrier as a completed transition of its "after" resource.
The queue's ExecuteCommandLists hook stays and reports submissions. A list's observed states live
in its own recording state and are read and written only by the thread recording it (its lifecycle
events, its captures and these hooks), so a barrier takes no capture lock; a list ReShade never
reported is not tracked at all, since it cannot admit a capture. A consumer copy is recorded after
its read lease is registered, outside the capture lock.

A ReShade runtime's own immediate list is never reported by those events. ReShade records it only
during present/effects events, open outside any render pass, and resets it right after each
submission without replay. The add-on registers that list before Generic depth's end-of-frame copy
and before the provider's reads; each submission the queue hook observes then ends its recording
and begins the next, so it is covered like a game list. A ReShade event for the same object proves
a game list reused the address and ends the registration. Copies into any covered list use the
recording-based lease, whose checks also prove the list is open and outside a render pass. An
unregistered runtime list falls back to the runtime's contract for consumer reads: the read lease
is released at the list's next observed submission, followed by the private queue fence; until
then the capture storage and destination stay retained. Other lists outside the lifecycle stay
unavailable.

`Sunshine list lifecycle` (at most every 5 s) counts capture admissions: `covered` (open outside a
render pass), split into `states observed` by the barrier hooks and `declared`, and `not_open` by
what the lifecycle showed instead (`unknown`, `closed`, `pass`). The lifecycle lives in
each list's own recording state (COM private data, which proxies forward), beside its observed
resource states; there is no global list table. A submission is not a lifecycle event (Close
already ended the recording), and a destroy event retires the recording. Before this split, a comparison of ReShade's barrier event
with the hooked states agreed in The Witcher 3, Expedition 33 and Hogwarts Legacy; every Dead Space
mismatch was the separately transitioned stencil plane.

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
Missing range measurements hold established controls. Missing current depth returns
current-color mono once the bounded hold of the last real copy described above ends. Without a valid source-associated
matrix, the relative raw path used by Generic also supplies gain and zero placement for NGX.

For matched non-flat ranges, a positive multiplicative change of depth coordinate changes gain and zero
together so that the signed field remains unchanged. This does not recover physical distance
from raw fallback, and separate initialization histories or an unknown nonlinear encoding need
not match. The normal UI labels the fallback **relative depth (assumed infinite
far plane)**; it does not imply recovered distance. Automatic uses no per-game presets or hidden
percentile calibration. Camera and raw API controllers also retain separate
reference histories. They share one reference policy, but switching to a branch with a different
history is not guaranteed to be jump-free. Raw API encodings retain theirs per logical encoding
across provider and frame generation switches, as described under
[direct NGX depth selection](#direct-ngx-depth-selection). Generic physical buffers retain separate references
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
D3D11 NGX interception remains diagnostic-only: its exports are hooked only while the upscaler
call trace is on.

At evaluation entry the adapter reads the exact `Depth` resource and explicit render/depth subrect.
A render subrect reported as 0x0 means the whole creation-size input, exactly like absent keys.
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
evaluation. ReShade's create event associates its first recording; if a hooked barrier arrives
first, the observer associates it at native operation entry, before forwarding the operation, so
those transitions reach the capture owner. Close and render-pass restrictions come
from ReShade's lifecycle. The callback retains that recording identity across the original call; a
later Reset cannot relabel old evidence as belonging to the new recording.
Creating the identity supplies no resource state: capture still requires an actual observed
nonzero transition, and a submission with no observed recording cannot establish ownership.

Nested NGX capture is suppressed only when an enclosing SL or NGX evaluation has actually recorded
a valid native copy, or when SL has confirmed enabled FG authority for that viewport. A direct
(not nested) NGX evaluation applies the same FG rule: while SL reports FG enabled on its one
viewport it begins its evaluation (live handoff evidence) but records no depth copy, since
selection never considers NGX while FG is required and FG Off makes every snapshot taken meanwhile
inadmissible; a busy or ambiguous FG read copies as before. The `Sunshine NGX depth` line counts
these evaluations as `fg_owned`. Before the final review a UE title with NVIDIA's DLSS plugin and
the Streamline DLSS-G plugin copied its full depth plane on every real frame with FG on, for no
consumer. A retained
resource, metadata-only ticket or rejected native attempt does not block a usable inner NGX input.
Pending recorded copies retain deduplication authority; all nonzero tickets still receive completion.
SL and NGX observations remain separate, so malformed input from one cannot invalidate the other's
usable frame. The shared selector applies the delivering-owner rule above, so an inner NGX capture
still in flight does not displace a delivering SL owner after a rejected copy, including a native
attempt rejected before its capture existed. Otherwise it prefers readable SL over ready NGX, with
the bounded pending interval and explicit FG scope described above.
An explicit successful release of its feature ends NGX ownership and permits automatic fallback,
and so does one second without any evaluation of the established source, as for Streamline
(above); an expired owner never blocks another live API source.
Manual pins continue to override the automatic provider.
Concurrent ambiguous NGX feature/view evaluations remain mono; the add-on does not infer which
unrelated viewport is the final game camera.

Ordinary NGX Super Resolution does not supply a camera projection. Its confirmed device depth uses
the shared raw-scene controller with the infinite-far assumption above. Explicit linear view distance
uses reciprocal depth directly without that assumption or an invented camera matrix.
Calibration is keyed to the provider, logical feature generation and depth convention, rather than physical texture addresses or
dynamic render sizes. Rotating textures retain the reference; a new feature/convention starts a
fresh one. API-provided raw sources keep a bounded least-recently-used set of exact logical
encodings (provider, logical source generation, viewport, layout and convention epochs, and
direction). Each encoding has its own reference and its own frame and sample watermarks, so SL
and NGX sequences never constrain each other, while capture-time floors and exact identity
continue rejecting delayed packets from another encoding or an earlier visit. A returning
encoding therefore does not repeat its startup window after a frame generation switch. Inactive
encodings receive no packets, so re-entry, whether after another raw encoding was bound or after
the projection path took the present, keeps gain and zero but renders only after a fresh target
captured after the return; until then a held reference reports `no_target`, not `holding_reference`. A
returning encoding's DLSS reset or feedback revision revokes that evidence, not the reference,
as it does for a continuously bound encoding. A provider identity change (for example a
recreated NGX feature, a new Streamline observation epoch, viewport or direction), eviction from
the set, a new raw basis epoch and a runtime reset discard it. There is no age cap; one short
enough to bound the reference would also expire it across ordinary frame generation intervals.
Production has no recenter; the test-fixture recalibration hook also discards retained
encodings on the projection path. A Generic interlude or a present without a known identity is
not an absence from the provided path, so the encoding keeps the current hold, and NGX/Generic
selection flicker or brief depth loss cannot starve it of fresh targets. Immutable exact-range readbacks drive
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
Discovery never patches a DLSS Frame Generation module (`dlssg` in its path: `nvngx_dlssg.dll` or
a driver model under `NGX\models\dlssg`); `skipped_frame_generation` counts them. Frame
Generation refuses to create its feature once its entry points are patched: The Witcher 3
Remastered got `FAIL_PlatformError` (`0xBAD00002`) on every attempt while hooked and generated
frames with the NGX hooks off. It supplies no depth through NGX; Streamline tags carry FG depth.

The trace logs installed-hook coverage, the observation window, API/feature call counts,
success/failure with the newest NGX failure code (`last_failure`, an `NVSDK_NGX_Result`), caller
module and offset, and a bounded first-call stack. NGX feature identity
remains unknown when its creation was not observed. NGX calls nested on the same thread inside
an observed Streamline evaluation are identified separately from calls outside that scope.
Outside-scope calls alone do not prove direct integration: unobserved hooks or another thread
can break that association. Zero calls mean none observed in the reported window, not API absence.
Use positive call/caller evidence to decide which integration adapter a game needs; DLL presence
or version alone is insufficient. This diagnostic is independent of the active depth-path label.

The Witcher 3 Remastered with FG off tags 1485x835 depth for every Ray Reconstruction evaluation
(SL feature 1001), yet no SR/RR evaluation found its depth tag until repeated token requests
stopped orphaning the frame's tags (see frame-token identity below).

A token address returned for a different numeric index is a recycled token and gets a new
generation. Requesting the same index again returns that frame's token, so it keeps its identity
and the constants and frame tags already recorded for it; treating such a repeat between tagging
and evaluation as a new frame orphans that frame's tags. The Witcher 3 Remastered with FG off
matched this: Dump 3D showed its framed depth tag and Ray Reconstruction evaluation on one frame,
yet no SR/RR evaluation found its framed tag. A repeated request for the same index changes
nothing and takes no writer, so the about 30 requests a frame from several threads in The Witcher
3 no longer contend for the token store. Without a numeric index, every request is a new frame.
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
different order from their ReShade callbacks. A D3D12 Generic backup is published from its
before-clear snapshot on the runtime's own immediate list, which the tracker never sees reset, so
the frame's capture marker is the snapshot's own recording in the game's list, and the published
copy itself reads as an unknown recording that never inherits content correspondence.

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
markers supply bounded, current-thread presentation brackets. Only the diagnostic presentation
trace reads them, so they are tracked (and the PCL marker hooks installed) only while the
Diagnostics switch is on. V2 PCL discovery passively observes
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

The historical fixed-scale CPU policy (`K=1/q_ref` from a central reference patch, then a moving
screen plane) and its opt-in `SUNSHINE_GAME3D_CAMERA_SCENE_TEST=1` fixture had no production
consumer and were removed. `tools/reshade/camera_scene_policy.h` keeps only the registration
types and constants shared with the camera sample binding and the production
`projection_depth_controller.h`, scene gain and raw controllers.

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
controller fabricates camera registration. Shared identity
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
work retains its resources until completion. Failure to prove completion (device removal, a
failed fence signal or query) quarantines one bundle for the process lifetime and pauses sampling
until that bundle's runtime is retired; a replacement device or runtime samples again. Only a
shader compilation failure disables sampling permanently. Reuse introduces no wait or additional
in-flight sample.

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
The per-process `Local\\Sunshine3D.ReShade.SBS.<PID>` mapping contains metadata and four slots:
export protocol 4, a 448-byte shared state with 128-byte metadata (four texture handles) and
64-byte slots from offset 192. Each slot carries `cursor_plane_flags` and `ui_parallax_uv` with
the same ownership as its texture: the signed scalar is one eye's displacement in source-eye UV,
with finite values in `[-0.04,0.04]`; only zero flags or `cursor_plane_present` are valid, and
zero flags require a zero scalar. The streamed Game provider negotiation has its own version.

**One protocol, refused by name otherwise.** Protocols 1-3 had three slots (a 384-byte shared
state with 120-byte metadata; 1 without the cursor plane, 3 for the PQ transfer). Every protocol
starts with the same header: `shared_bytes` at offset 4, then the metadata's `signature`,
`protocol_version` and `metadata_bytes` at offsets 16, 24 and 28 (`same_protocol`). The add-on
and the host are installed together and each accepts exactly protocol 4. The receiver maps the
whole section and reads that header before anything else; a mapping still being initialised
(zero) is retried quietly, and any other published header is refused, never read or written, with
`ReShade SBS: the Game 3D add-on in process P speaks export protocol N (B-byte mapping); this host
speaks protocol 4 (448 bytes, 4 export slots). Install the add-on of this Sunshine 3D release and
restart the game; the stream stays 2D.`, once per receiver and producer process. A consumer
declares its protocol with its capabilities (`consumer_protocol`, offset 156, below), and the
producer answers only a request that declared protocol 4: another is refused with `Sunshine SBS:
refusing a host that speaks export protocol N; this add-on speaks protocol 4 (4 export slots)...`
once per request and gets no ring, and a request that declared nothing gets none either
(`export inactive (consumer_undeclared)`). Hosts of protocols 1-3 already refused a mapping whose
`shared_bytes` differs from their 384 bytes, so they never write into a protocol 4 mapping, and
neither side logs that pairing: the add-on's `exporter ready (export protocol 4, 4 export slots)`
line is then followed by no generation. The add-on's named refusal guards a later host that
declares another protocol.

**PQ wire transfer.** `transfer::pq` (3) is Rec.2020 primaries and SMPTE ST 2084 code values,
1.0 = 10000 cd/m2, valid only as R10G10B10A2_UNORM (DXGI 24); its slots carry the cursor plane
like every other transfer. A consumer advertises what it accepts and the protocol it speaks: it
writes `consumer_capabilities` (offset 152; `consumer_accepts_pq` = 1) and `consumer_protocol`
(offset 156), then `capability_nonce` (offset 144) = its nonce, then `consumer_nonce`.
`consumer_stream_pq` = 2 means the consumer encodes HDR10 PQ: a producer may pack any HDR source,
scRGB included, as R10G10B10A2 PQ. Sunshine sets it only for a streaming encoder whose output is
HDR, never for Local AR. A consumer whose capabilities change requests the connection again under
a new nonce. A producer honours the declaration only when `capability_nonce` equals the nonce it
answers (`answered_capabilities`, `answered_protocol`), so a replaced consumer's declaration never
applies to another request. Two receivers' requests can interleave so that `consumer_nonce` names
one and `capability_nonce` the other; the receiver that `consumer_nonce` names then finds its
request unanswered with another's declaration and sends the whole request again under a new nonce
(the other backs off, as a replaced receiver does). A replaced receiver that finds the connection
without any receiver (`consumer_nonce` zero, its replacement detached) requests again too. An HDR10 swapchain exports PQ to a consumer that accepts it, and a
native scRGB swapchain exports PQ to a consumer that encodes HDR10 PQ (`exports_pq`; the bit
implies acceptance). Any other consumer gets FP16 scRGB. The native render chooses the transfer
from the consumer it reads before rendering, so a consumer attached, replaced or detached between
a Present's render and its export whose answer differs, in either direction, gets the next
Present's export, rendered for it (`transfer_current`, export reason
`consumer_transfer_changed`); a reference export cannot render again and only refuses PQ output to
a consumer without PQ (`consumer_without_pq`). Until the final review only that PQ direction was
checked, so a PQ host attaching during an HDR10 render made with no consumer got an FP16 ring that
the next Present replaced, two generations and up to two ring allocations on the Present thread
for one attach. For an HDR stream the host would
itself encode an FP16 scRGB export with `scRGBTo2100PQ`, so packing it in the game is the same
encoding at half the shared bytes per frame. An SDR stream and Local AR keep FP16 scRGB from a
native scRGB swapchain: wide-gamut and over-10000-nit values clip differently in PQ (a negative
Rec.2020 clip) than in the SDR tone map (a Rec.709 clip). SDR stays sRGB. A changed transfer is a
new export ring. The generation line names `PQ HDR10, protocol 4`.
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
change. The finished ring of the previous transfer or extent stays cached (one entry, 60 s) and is
renewed the same way on a toggle back. A ring that stopped exporting (focus loss, or no consumer,
as in Game mono) and has no unfinished producer work is released after 10 s idle
(`export_ring_idle`), so leaving stereo frees its slots: about 530 MB for a 4K FP16 ring and
265 MB for an RGB10A2 one (four 7680x2160 slots of 133 or 66 MB), plus up to four 66 MB FP16
overlay layers if the overlay composed into that ring. The cached ring of the other transfer holds
the same until its 60 s expiry, so a transfer or size toggle briefly holds two four-slot rings. A
later export allocates a new ring, the present-thread step above.
Every released ring (idle, cached, or with its runtime or swapchain) first has each renderer drop
its cached render-target views of the slots: a D3D11 view holds its texture, so the slots would
otherwise stay allocated until the renderer's next pack.

For each exported frame, the producer claims a slot, writes the final stereo image into it,
writes the frame sequence, QPC timestamp and matching UI displacement, marks the slot ready, then
signals its GPU fence (the host's fence wake then always finds the slot ready). The host never
reads a ready slot before the fence reaches its sequence, so the early mark exposes nothing. The
producer publishes in ReShade's `reshade_present` callback, right after the pack and any overlay
composition and before the game's Present, on both APIs: D3D11 signals on the immediate context
and flushes; D3D12 first submits ReShade's immediate command list (`flush_immediate_command_list`),
then signals on the presenting queue. ReShade 6.8 fires `finish_present` only after the real
Present returns, so a D3D12 signal there sat behind the Present packet and waited for the CPU's
Present call to return. `finish_present` now fences only a D3D12 export whose `reshade_present`
did not run, and publishes it only if its overlay composite was recorded. A failed signal withdraws
the slot and deactivates the export. Native Game 3D renders both eyes in one pass
(`SunshineRenderPackedPS`, or `SunshineRenderPackedPQPS` for the PQ transfer) directly into the claimed slot, so the common path writes no eye
intermediate and has no full-frame copy. Each half branches on its eye so the compiler folds the
pixel-center scale into the warp exactly as the former per-eye pass did; the output is
byte-identical to the former two passes and to the frozen FX reference. At 4K this drops two FP16
eye textures (~133 MB) and their write/read-back. While a diagnostic dump is armed the pass
writes the renderer's own SBS image instead, which is then copied into the slot; reference FX
exports use that copy too. While the ReShade overlay is visible the pass also writes the
renderer's own image, and the overlay's composite then writes every slot pixel from it (the image
itself wherever the controls are transparent), so nothing is copied into the slot first.
Without a consumer, a free slot or an armed dump, no eye is rendered at all. The shader declares
`SUNSHINE_PACKED_EYES`; a dump whose embedded shader predates it replays with the former eye and
pack passes.
Slot render-target views are cached for one export generation and dropped when their ring is
released. The slot rests in the shared
common state between owners, and the pass reads only the renderer's working set, which the
renderer's completion fence already retains. Up to four submitted GPU writes may remain unfinished, bounded by
the four-slot ring. A recorded slot write must receive its submission fence before the next
is admitted; there is no global requirement to wait for the previous GPU write to complete.
Admission (`acquire_slot`) takes a free slot first, else the oldest ready slot that a newer
completed frame supersedes. The host claims the newest ready frame whose fence passed, so the
newest completed frame of the ready and reading slots is the one it claims next (or holds), and it
is never written over, even while a newer write is still queued. Writing over it as soon as a
newer write was queued replaced about 70 finished frames/s that the host never claimed in Stellar
Blade at 4K with FG 4x: a host that came a few milliseconds late found nothing claimable until the
queued write landed. A slot is reusable once its previous GPU write completed, judged by that
slot's own fence sequence, including slots already marked free by a consumer that discarded them.
A ready slot whose write is still queued is never reused either: when the GPU runs two or more
Presents behind, writing over it every Present would leave the host no claimable frame at all.
A copy or overlay write waits for completion. Reading slots remain unavailable. If no slot is
reusable, the add-on drops the publication and returns to the game without waiting.
`Sunshine SBS output` counts both (`dropped`, and `overwritten_unconsumed` for a completed ready
slot written over, which a newer completed frame had superseded). `reshade_exporter_tests.exe`
runs the real admission against a host that claims at 90 fps (`ring_throughput`): a busy host
without any fence wake whose replaced slot returns at the claim, and the live host, woken at each
completion, with two pictures in flight completing 5.5-22 ms after submission and a replaced slot
returned only once its reads completed. It must never write over the host's next frame, and with
four slots the host must receive 85.5 or more new frames/s at FG off, 2x and 4x Present rates.

**Why four slots.** On 10-06 (15:39:44-15:40:00) Stellar Blade at 4K ran FG 4x GPU-bound: 24 NGX
evaluations/s and about 95 Presents/s. The host claimed nearly every published frame
(`overwritten_unconsumed` about 0), yet the add-on dropped 27-35 Presents/s and the stream had
57-65 new frames/s: of three slots one was held, one replaced and still being read, and the last
carried the host's next frame or a write still queued on the busy GPU. The live host against that
pattern with each fence completing 15-35 ms after its Present reproduces it with three slots
(published 61/s, dropped 33/s, overwritten 3/s, 58.0 new frames/s) and delivers 83 published, 11
dropped and 70.8 new frames/s with four. Fences at 5-15 ms drop 3/s with three slots and none
with four (88.8 and 89.5 new frames/s). In the same model a fifth slot would reach 77 new
frames/s and a sixth no more: queued writes complete in clumps, and the host takes the newest of a
clump. `RemoteEncodeProviderPacingTest.AFourthExportSlotTakesThePresentsAGpuBoundGameDroppedWithThree`
runs the host's own encode loop on the same pattern (three slots: 34 dropped, 56.2 new frames/s;
four: 12 and 71.8). The fourth slot costs 66 MB at 7680x2160 RGB10A2 (sRGB or PQ) and 133 MB in
FP16 scRGB, for the whole time a ring exists. Once the ReShade overlay has composed into a slot,
that slot also keeps its own back-buffer-sized R16G16B16A16_FLOAT overlay layer (about 66 MB at
3840x2160) until the ring is released, so with the overlay used the fourth slot costs another
66 MB.

Each slot retains its native source resource and a separate overlay compositor. They cannot be
replaced while that slot's copy/composition remains in flight. Reload, deactivation and runtime
destruction withdraw publication but retain source, destination, overlay and fence lifetimes
through GPU retirement; an unfinished callback does not make recorded work safe to destroy.
The opt-in `reshade_exporter_tests.exe --async-ring` regression delays GPU completion across
one publication per slot, checks backpressure on the next and ownership transitions, and verifies
exact copied pixels and source lifetimes.

Sunshine claims the newest completed slot and converts straight from it through a shader view
created per slot when the generation opens; there is no private copy. The newest frame's slot stays
`reading` while it is the newest, so repeat conversions and stream-gamma reconversion keep reading
it. Every conversion that reads it then signals the receiver's own read fence after all of its
reads (Y, UV, Local AR, resample and the cursor patch) and flushes (`receiver_t::reads_recorded`).
When a newer frame is claimed and that value has completed, the replaced slot returns to free at the
claim itself, after the generation, nonce and sequence checks. Otherwise (its reads are still
running, or a poll returned the frame again after the signal, which then signals a new value) a
thread-pool wait on that value returns the slot as the GPU completes it, whatever the encode loop is
doing; the loop's polls and submissions check the same only as a fallback for a wait that could not
be armed. With two pictures in flight a claim can come before the previous conversion's GPU work
(queued behind the game's) completed, and the loop then sleeps to its next poll target: returning
the slot only at the loop's next check left the producer one slot for part of each such interval
(`AReplacedSlotReturnsAsItsReadsCompleteWithPicturesInFlight`: on the three-slot ring 85.5 and 86.5
against 88.8 and 89.5 new frames/s with FG 2x and 4x and the live FG-on encode times; with four
slots both reach the stream rate and the read-fence wait drops 4 and 48 Presents/s against 18 and
68). Nothing blocks. A generation change
or detach abandons held and retiring slots rather than releasing them early, and cancels the wait. While the host encodes a frame it therefore holds only that frame's slot
and leaves the producer three. Before, the replaced slot stayed `reading` until that encode returned,
so for about 9 of every 11 ms at 90 fps the producer of the three-slot ring had one slot. There is no cross-process GPU
wait and no texture overwrite while either side uses the slot. A new consumer requests fresh
resources rather than reusing the abandoned generation.

Sunshine converts a live export when its fence completes, not at its next poll. The receiver
arms the generation's ready fence (`SetEventOnCompletion` for the next value) on a thread-pool
wait that wakes the encode loop; a woken frame converts once it is within the variation threshold
of its target, so the stream cadence still caps the encode rate. Every slot is gated on being
ready for the current generation with a sequence at most the fence's completed value. A frame
that has completed before its poll target (its presentation target minus the variation
threshold) is held exactly to that target on a high-resolution timer (`provider_hold`); the loop
then takes the newest completed frame without waiting. A hold lasts at most one frame interval
and a control wake does not end it, so a control request that arrives during a hold waits at most
the rest of it. The loop then re-checks for an IDR or reference invalidation: one that arrived
restarts the iteration, and the newest completed frame is encoded at once as the recovery frame
rather than first as a P-frame the client cannot use. Shutdown, reinit and video-mode changes are
re-checked before the conversion, and a stream-gamma request applies from the next one. Every
other wait ends at the first of a fence
wake, a desktop capture, a control request and its bound (the pending frame's poll target or the
keepalive). With this toolchain (winpthreads; libstdc++ without a clock-based condition-variable
wait) a timed condition-variable wait that nothing notifies ends only at the next 15.625 ms Windows
scheduler tick, even for a zero timeout and whatever the timer resolution. Every bounded image wait
of the encode loop (Game 3D, desktop, desktop Host SBS, the keepalive) therefore waits on a
deadline waiter (`platf::create_deadline_waiter`, `detail::pop_encode_image`): the encode thread
itself waits for an event, which every raise, wake and stop of the image event sets while it waits,
or for a high-resolution waitable timer armed at the bound, and then re-checks the bound against the
steady clock, so a stale notification never ends a wait early and no callback outlives the wait.
A wait that reaches its bound returns no image, as a timeout always did. Such waits ran 4.4-15.3 ms
past their bound on average and up to 24 ms in `RemoteEncodeImageWaitTest` (live 10-06 desktop
Host SBS: 48-133 of 1100-1400 waits per 20 s ran to their bound, 5.6 ms late on average and 12-13
ms at most); they now end 0.2-0.45 ms after it on average and at most 0.8 ms (1.6 ms for a bare
3 ms pop) over 25 repeated runs. Until 10-06 every loop iteration also
began with a zero-timeout pop of the empty stream-gamma queue, so the loop ran at most once per
tick: in Stellar Blade at 4K the host claimed 56-62 new frames/s of a 90 fps stream while its fence
wakes arrived on time and the add-on replaced 25-109 finished frames/s that it never claimed.
Zero-timeout pops now only check (`safe::no_wait`), which also removes that cap from the desktop
and Host SBS loops. The one-millisecond re-check of a held export (`export_recheck_interval`),
added for a late or missing wake that the live counters then refuted, is gone. With
`diagnostics = enabled` the host logs every 20 s `Game 3D export: N new frames claimed and W fence
wakes in 20 s; K claims found their frame complete before its wake (late or missing wake); fence
wake to claim avg A ms, max M ms over C claims`; K close to N means the wake arrives late or not
at all. Only these diagnostics time wakes: the receiver records wake times and the wake-to-claim
delay only after the first such report starts its window, so without diagnostics a claim never
takes the fence callback's lock. The encode loop logs its own account at the same interval,
`Video encode loop: I iterations in 20.0 s; N new-content and R repeated-content encodes; holding
... in H exact holds (requested ..., overshoot avg/max); waiting ... in W image waits (requested
...; T ran to their bound on the deadline timer, overshoot avg/max); converting ...; encoding ... (NVENC submit ...,
completion wait ...; up to D pictures in flight, B submitted behind another, submission to packet
avg/max); loop work ...`. Encodes are told apart by the encoded content's identity,
as the packets' content-age loggers do, so a keepalive that re-renders an unchanged export counts
as repeated content although it converted. On the deadline timer an image wait that runs to its
bound overshoots it by a fraction of a millisecond; `at a scheduler tick` instead means the timer
could not be created and such waits end up to 15.6 ms late. While an
export is live the loop neither polls nor repeats frames at stream cadence: new exports, cursor
changes and the minimum-FPS keepalive (which re-checks the connection) produce frames, and a due
stream-gamma white-level query runs in the next of them rather than forcing a repeat.
`RemoteEncodeProviderPacingTest.TickBoundWaitsReproduceTheLiveSixtyFrameCap` runs this loop's
wait and schedule helpers against the ring's rules with the live FG off, 2x and 4x arrival
patterns and waits as long as they really are, and reproduces the live 56-62 new frames/s with
the old rules on the three-slot ring of protocols 1-3 those runs used (the calibration runs keep
three slots; the production expectations run on four). `LiveExportDeliversNewFramesAtStreamRate` requires 87.5 or more new frames/s with
no repeat and no skipped stream frame with prompt wakes, and 80 or more with late or lost wakes
(nothing re-checks for them: a frame that completed during the previous encode is found by the
next iteration, others by the next desktop capture). `SlowerGameIsClaimedWhenItsFenceWakes`
claims every frame of a 40 fps game within 1 ms of its fence completion. Game mono observes the
producer's published source without attaching (a status-only READY): the game then creates no
ring and packs no stereo.

A picture whose input the GPU is still producing is waiting on upstream work, not on the encoder.
The encoder's 250 ms completion budget then starts when that input completes (2 s at most, the
default GPU timeout). On 10-06 the game's first frame-generation enable kept the host's own
conversion off the GPU for about 300 ms: the fixed budget failed the encoder at 251 ms with
`input_producer=pending_at_251ms`, blocked new encoders, rebuilt the session, detached the receiver
and so made the game build a second export ring. Such a stall now delays one picture.

The encoder of an independent provider, and of desktop Host SBS, keeps up to two pictures in flight
(`video::nvenc_pipeline_depth()`, fixed per encoder as `nvenc_base::pipeline_depth()`; plain
desktop keeps one, and Local AR presents without NVENC). Each picture has its own
NVENC input, bitstream and completion event: at submission the converted surface is copied into
the picture's input on the immediate context (about 50 MB each way at 7680x2160 10-bit; the two
inputs cost about 100 MB of video memory), so the next frame converts and starts encoding while the
previous picture still encodes. An encoder that cannot create a completion event or an input for
every picture (video memory short, as with Host SBS beside a GPU-bound game) encodes one picture at a
time from the conversion target instead of failing the stream
(`NvencPipelineTest.HostSbsEncodesOnePictureAtATimeWhenASecondInputDoesNotFit`). A thread of its own (`encode_pipeline_t`) waits for the pictures in
submission order and publishes each packet as soon as its picture completes, whatever the loop is
doing; the loop waits only while two are in flight, and then for the oldest. Before, on 10-06 in
Stellar Blade at 4K with FG on, `encode_frame()` waited for each picture: NVENC's completion wait
averaged 8.5-12.2 ms (about 5 ms uncontended; the rest is the conversion's GPU work queued behind
the game's) and the host took 55-71 new frames/s, against 80-90 with FG off. A bitrate change, and a
reference invalidation that calls `NvEncInvalidateRefFrames`, first wait for the pictures in flight,
as a returning `encode_frame()` did; one already handled, malformed or too large for the DPB (its
range extends to the last submitted picture) becomes done or an IDR at once. An IDR applies to the
next picture submitted. When the full encoded queue drops a delta frame and requests the recovery
IDR, the deltas already in flight behind it are discarded until that IDR, so the first packet after
the drop is the IDR. That thread then wakes the loop's image wait, and the loop reads an IDR
requested during its wait (by that thread or by the client) before converting, as after a provider
hold: the recovery frame is the capture that ended the wait, or the retained input, not a delta the
gate discards followed by the IDR. A recovery that skips the wait discards a wake that came while
nothing waited, so a desktop source does not encode a repeat for it. A picture that fails on that
thread fails the encoder; a loop exit first
delivers the pictures still in flight, and is clean only if none of them failed, so a failure then
still keeps new encoders waiting for its teardown. A picture's 250 ms budget starts when the wait for
it does, after the previous one completed. The loop claims at its poll target and then waits for
the oldest picture when both are in flight: waiting first would claim a newer frame but rebase the
schedule after every slow picture (`ClaimingAtThePollTargetKeepsTheCadenceWhenEveryPictureIsInFlight`:
82.5 against 87.5-88.2 new frames/s and gaps of 34-35 ms with a 14-45 ms tail every eighth picture
on the three-slot ring; with four slots 83.0-83.8 against 88.8-89.0 and gaps of 34 against 23 ms,
where waiting first would also claim frames 1.3-1.7 ms fresher: smoothness first).
`PicturesInFlightTakeEveryStreamFrameAtFrameGenerationEncodeTimes` runs the loop model with the
live FG-on encode times (5.5-22 ms per picture): 68 new frames/s one picture at a time on the
three-slot ring the live 55-71/s came from (its calibration band) and 69 on four, 90 with two in
flight, Present to packet within 0.5 ms of one picture at a time on the same ring.

Desktop Host SBS (AI depth on desktop capture, a 7680x2160 picture at 4K) keeps its capture
cadence: it is not an independent provider, so it holds nothing and an image wait that runs out
still encodes the input again; only its picture now encodes while the next capture converts. A
conversion writes only the conversion target, which each picture copied at its submission, so the
next capture, a retained-source poll of pending inference (busy, it keeps the target and the picture
repeats it; complete, it redraws the frame with its own depth), a depth-ready install and a
stream-gamma reconversion never change a picture in flight
(`NvencPipelineTest.HostSbsPicturesKeepTheConversionTheyWereSubmittedWith`). The same-frame poll
reserves 3 ms after its inference wait for postprocess, output and NVENC submission
([Host SBS](host-sbs.md)); one picture at a time the loop then also waited for the picture, so a
capture taken behind a slow picture reached its conversion with too little budget left and was drawn
with older depth. `RemoteEncodeHostSbsPipelineTest` runs the loop's wait, schedule and same-frame
budget helpers with conversions of 3-6 ms (1.5-4.5 ms of it the awaited inference) and 6-12 ms from
submission to packet. At 90 fps one picture at a time takes 87.2 new frames/s of 90, about 79 of them
a second drawn with older depth, Present to packet 19.6 ms on average and 26.2 ms at most; two in
flight take 90.0, 0.5/s with older depth, 15.7 and 19.7 ms. A conversion that always costs the loop
3-6 ms gives 72.8 against 90.0 new frames/s and 20.9 against 15.7 ms. At 60 fps (a 60 or 90 fps
stream), and at the 09-06 live times (conversion call 2.3 ms, NVENC 6 ms on average), no picture is
ever submitted behind another: both depths take every capture and two in flight cost the copy into
the picture's input (0.3 ms modelled for its 50 MB each way). Polls of a pending
inference on a static desktop never come faster than the stream interval; with their bound on the
deadline timer, two in flight poll at that interval and one at a time each poll also waits for its
picture, so the frame's own depth leaves 19.7 ms after its inference completes on average with two
in flight against 24.7 ms one at a time. A packet dropped by the full encoded queue while the loop waits needs the
retrieving thread's wake, as desktop capture wakes nothing else: without it the IDR waited for the
next capture, which became a discarded delta (the IDR 5.5 ms after the drop at 90 fps), or on a
static desktop for the idle bound behind a discarded repeat (55 ms); with it the IDR is submitted
0.4 ms after the drop, as one picture at a time. A client's IDR during the wait makes the capture
that ends it the IDR, one picture earlier at either depth (10.9 ms after the request one at a time
before, 4.5 ms now; `RemoteEncodeHostSbsPipelineTest.AQueueDrop*` and
`AClientIdrDuringTheCaptureWaitMakesTheCaptureTheIdr`).

While a streaming encoder converts a live packed export it holds the display's capture-pixel
claim, and Desktop Duplication and WGC forward only timestamps and cursor metadata. When the
claim is released (the export ends, or the encode loop exits or rebuilds), capture recovers the
dropped desktop. Desktop Duplication reads the next acquisition's surface as a whole-desktop
present even when only the pointer moved (damage unknown); if none arrives within an idle source
wait (200 ms; a pacing probe does not count), it re-duplicates the output once in place, and the
new duplication's first frame is the whole desktop. WGC keeps the newest frame it forwarded
without a copy and copies it on the first timeout after the release. An exiting encode loop
releases the claim before handing its retained source to a replacement and never hands over an
image without pixels. Converting such an image after the export is gone keeps the encoder input
and is encoded only as the minimum-FPS keepalive or a recovery frame. An encoder rebuilt while an
export owned the output encodes nothing until capture delivers the desktop, at most 210 ms, then
falls back to its startup frame; an IDR requested meanwhile is applied to the first real frame.
IDR, reference-invalidation, stream-gamma, video-mode and shutdown requests, and a capture
reinit, wake an independent provider's encode loop at once; a reference invalidation then encodes
a recovery frame immediately, like an IDR.

The cursor no longer needs a full-frame scratch copy: only its changed rectangles, grown by
`ceil(5 + 3/s)` texels where s is the most minified pass's scale, are copied (decoded to FP16 scRGB
for a PQ export) into a frame-sized patch, the existing blend draws run on the patch, and each
conversion pass re-runs scissored over the cursor's output region from the patch. The patch has
the frame's size, so sampling coordinates are identical and the output equals converting a fully
composited copy (byte-for-byte for SDR, R10G10B10A2 and FP16). The receiver validates and copies the scalar while it owns
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
receiver runs on the existing D3D owner. In mode `2` it observes the producer's status without
attaching, refreshed on captures and keepalives, so the game builds no export ring. In mode `3`,
until a mapped producer publishes a generation, it is refreshed on captures and keepalives and
polled only when the mapping's metadata, nonce or producer changed (a static screen is then
encoded at the minimum-FPS keepalive, not at stream cadence; until the 10-05 review every stream
deadline converted and encoded the flat fallback); a live export whose ready-fence wake is not
armed is polled at stream cadence, and Local AR always polls. A live export converts only on that
wake or a changed connection, reading the slot directly with no private copy and no CPU wait for
the producer's fence ([GPU handoff contract](#gpu-handoff-contract)).

## Color and HDR

Windows HDR being enabled does not determine the game's rendering color space. SDR games can
run on an HDR desktop, native HDR games can render scRGB or HDR10/PQ, and Windows Auto HDR is a
separate conversion feature. See Microsoft's [Windows HDR settings](https://support.microsoft.com/en-us/windows/hardware/display-graphics/hdr-settings-in-windows)
and [Auto HDR explanation](https://support.microsoft.com/en-us/windows/hardware/display-graphics/use-auto-hdr-for-better-gaming-in-windows).

The native add-on declares and validates the actual game swapchain color space:

| Game backbuffer color | Shared stereo texture | Transfer supplied to Sunshine |
| --- | --- | --- |
| SDR sRGB | RGB10A2 | Encoded sRGB |
| Native HDR scRGB | RGB10A2 to a consumer that encodes HDR10 PQ (`consumer_stream_pq`), else RGBA16F | PQ code values (Rec.2020, ST 2084, encoded as the host's `scRGBTo2100PQ` of the scRGB eye), else linear Rec.709/scRGB, 1.0 = 80 nits |
| Native HDR10/PQ, Rec.2020 | RGB10A2 to a consumer that accepts the PQ transfer, else RGBA16F | PQ code values (Rec.2020, ST 2084, encoded as the host's `scRGBTo2100PQ` of the scRGB eye), else PQ decoded to linear Rec.709/scRGB, 1.0 = 80 nits |

The HDR export preserves values above 1.0 and negative scRGB components. It applies no tone map
to the shared texture. The normal game window retains its original color encoding. A float or
10-bit format alone is not evidence of HDR: the native renderer uses the swapchain's declared
color space and recreates its programs/resources when it changes. Reference FX must supply
matching color-space annotations; a stale reference shader permutation is rejected.

SunshineGame3D decodes HDR10 texels before interpolating them. Interpolating PQ codes and
decoding afterward darkens mixtures even when constant-color tests pass. The pack decodes the
warp's two horizontal taps itself (`SUNSHINE_PQ_PER_TAP`; rows are texel centres, so the vertical
weight is zero), each held in FP16 as the former full-resolution FP16 surface held it, and
interpolates them with the sampler's own weights (the coordinate truncated to 21 fractional bits,
then rounded to 1/256 texel, measured exactly on NVIDIA), so neither the linearization pass nor
its 7680x2160 FP16 texture remains. The FP16 export differs from the former surface's by at most
one FP16 unit of the brighter tap (the hardware's filter precision). The PQ export holds each eye's
scRGB value in FP16, as the FP16 export holds it, and encodes it with the host's own matrix, 80-nit
scale and `NitsToPQ` constants in their order; the mono pack writes the source's own codes, and
alpha keeps its 0/1 coverage, which no host pass reads. Against the former FP16 export pushed
through the host's `scRGBTo2100PQ`, it differs by at most one 10-bit code (one rounding). A dump's
`sbs_transfer` (1 sRGB, 2 scRGB, 3 PQ) declares the SBS artifact's encoding, which the inspect and
preview tools decode and replay reproduces. At 3840x2160 the per-tap pack took 0.33-0.37 ms
against 0.37-0.43 ms for the former linearization pass and FP16 pack together (P1, GPU otherwise
idle). Native scRGB requires no decoding. For a consumer that encodes HDR10 PQ, a native scRGB
source packs the same 10-bit PQ export (`SUNSHINE_SCRGB_PQ_PACK`): each eye's FP16 export value,
mono included, encoded with the same constants, within one 10-bit code of the host's encoding of
the FP16 export (mono exactly). At 3840x2160 that pack took 0.237 ms against 0.210 ms for the
FP16 pack (the source-alpha fixture's `--timing`), while the shared image halves to 66 MB per
frame and the host passes its codes through instead of encoding FP16.
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
then encodes SDR output back to sRGB; HDR output remains scRGB, and a PQ export decodes, blends in
linear scRGB and encodes back as the host does. The overlay follows ReShade's
HDR UI brightness while leaving the underlying game's highlights and negative components intact.
Opening the panel does not tone-map the game or alter either eye's geometry.

Sunshine converts the declared input to the destination's color space at final presentation or
encoding. HDR output preserves HDR; SDR output uses tone mapping for HDR input. A PQ export
streamed as HDR with the Windows default stream gamma passes its codes through; stream gamma 1
or 2 decodes it to linear Rec.2020, applies the channel correction and re-encodes; an SDR stream
decodes it to scRGB and uses the FP16 tone map; Local AR decodes per tap (an scRGB swapchain) or
tone maps (an sRGB swapchain) and never takes the exact-copy path; scaled eyes decode per tap into
the FP16 scRGB intermediate; the cursor over a PQ export blends in linear light on a decoded patch. SDR input on
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
and production Sunshine receiver together in SDR, scRGB and PQ. They verify the claimed
shared-slot pixels, source timestamps, effect/reload/focus invalidation, recovery and receiver
restart. Their separate test add-on supplies foreground observations for a hidden fixture window;
the shipping add-on keeps the real Windows foreground check. Additional runs use a distinct
receiver process and exercise the real process-handle duplication, texture/fence import and
direct read of the claimed shared slot, including slot hold and retirement. All six
SDR/scRGB/PQ and D3D11/D3D12 cases passed again on 2026-10-05 with the direct-read receiver,
including restart into a new receiver process. This establishes sharing under the same user/session/integrity level;
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
