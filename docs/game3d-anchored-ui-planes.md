# World-anchored UI planes for Game 3D

Status: design proposal, not implemented. [Game 3D](reshade-sbs.md) owns the UI protection
contract. When this ships, its rules move there as a new module M9 and new rules E3, P3 and T2 of
the [UI decision framework](reshade-sbs.md#ui-decision-framework) (P2 stays reserved), and this file
is retired. Thresholds owned by that document are named here, not restated.

## In brief

- Today every UI pixel sits on one plane. An enemy's health bar should sit at the enemy instead.
- No game, engine SDK or Streamline/NGX input says which UI pixels belong to which enemy, or how
  deep that enemy is. The depth under a floating bar is the background behind it.
- What does carry the answer is motion. A bar the game projects from a point on the enemy moves on
  screen exactly as the enemy does, and the game's motion vectors say how every scene pixel
  moved. Every DLSS game supplies them, because DLSS requires them. If the scene pixels near a bar
  move as the bar does, and keep doing so through two *different* motions, the bar belongs to them
  and takes their displayed depth. Screen HUD never moves with the world, so it never qualifies.
- Every case without that proof leaves the element on today's plane: no motion vectors, no exact
  frame pairing, a still or only-rotating camera, an occluded owner or an unproven route. The
  worst case is today's look, and the first stages are offline only.
- Reach is limited at first. Anchoring runs only on D3D12 routes that pair the UI mask, depth and
  motion vectors from one real frame. These are Streamline tag batches and DLSS-G evaluations;
  [Frame identity](#frame-identity-and-frame-generation) lists the games. Presented alpha, the
  one-frame-late layer and the Generic path keep today's plane.

## Problem

UI protection pins every texel of the selected UI mask toward one plane, `pUI`, chosen by the
adaptive display-fraction policy (mode 5; **UI protection** and placement under
[Setup](reshade-sbs.md#setup)). `SunshineApplyUICS` reads that scalar from
`SunshineUIPlaneParallax()` ([game3d_native.hlsl](../tools/reshade/game3d_native.hlsl), lines 2005
and 2077).

In the owner's words: "some health bar or enemy indicator should be above the actual enemy." Enemy
health bars, nameplates, lock-on and target markers, quest markers and interaction prompts all sit
at the HUD plane. In the Witcher 3 "Roach" dump (`game3d_38380_164034700566678`, 2560 px wide, FG
on, UIAlpha winner) the nameplate is pinned at `+0.00200` UV per eye, the horse directly below it
is at `-0.00002`, the trees behind it at `-0.00098`: the label floats about 5 px per eye in front of
its horse. Also, the conflict probe counts every exactly pinned texel, so a bar over a near enemy
pushes the whole HUD toward its cap. Today's dumps reach the cap only through HUD (subtitles, notice
text, the W3 tracker and hints over foliage, wheels, menus); none shows combat yet.

**Goal.** Give each world-anchored element its own pin target at the displayed depth of its owner,
automatically in every game whose inputs can prove it, and leave every other frame, row and element
exactly as today. **Out of scope:** aim-point depth; UI the game draws into the scene with depth
(Hogwarts Legacy's court rings, Dead Space's diegetic UI); interpolating element positions on
generated Presents; the system cursor, which stays on `pUI`.

## What the inputs can and cannot tell us

**Coverage, not elements.** Every UI source gives coverage per pixel; S1 picks one winner per frame
and H1 can turn the frame flat. None says which element a pixel belongs to or how deep it is, and
the Streamline and NGX adapters expose no UI depth.

**The depth under an element is not its owner's depth:**

| Dump | Element | Scene under it | Owner |
| --- | --- | --- | --- |
| `game3d_38380_164034700566678` (W3) | "Roach" nameplate | trees, `-0.00098` | horse below, `-0.00002` |
| `game3d_12632_54160766941827` (W3; read from colour, no mask captured) | "Talk / Vesemir" prompt | Geralt, near | Vesemir, far, hidden |
| `game3d_69460_296226962143480` (SB layer route; illustration only) | ground dot | ground, `+0.00156` | the same ground |

Historical mode 2 resolved one global plane from the nearest depth under all coverage, floored at
the midpoint ([game3d_ui_plane.h](../tools/reshade/game3d_ui_plane.h)); per element it would put
each element on its nearest occluder.

**One frame cannot tell world UI from HUD.** HUD comes in the same sizes and places: RE9's centre
crosshair (`game3d_52696_225539427440975`, 84 px, five pixel components that merge into one cell
element), subtitles, the SB and ACS compasses, an HL marker clamped 125 px from the top
(`game3d_40404_173061335442888`), the W3 tracker (`game3d_46312_198402901901357`). A size or edge
rule anchors all of them. Single-frame statistics fail both ways (on `vertical_field`, margin at
C = 0.005): around Roach "majority" picks the trees (51%) and "maximum" `+0.00038`; a floor from
each footprint moves the W3 tracker from `+0.00200` to `-0.00033` and the RE9 crosshair from `0` to
`-0.00138`. HUD text swimming in depth is the worst outcome.

**Motion separates them, within limits.** A world-anchored element moves on screen with the scene
point it labels; HUD does not. That needs the masks of two consecutive real frames and the scene
motion between them. Streamline type 1 appears in every dump with Streamline tags, NGX SR
`MotionVectors` in the E33, SB, HL, ACS and RE9 dumps; `DLSSG.MVecs` was never queried. Game 3D
consumes none ([streamline_buffer_types.h](../tools/reshade/streamline_buffer_types.h)). Camera
matrices are not enough: `clip_to_prev_clip` (Streamline, `DLSSG.ClipToPrevClip`) predicts only
static-world motion, is frame-correlated only through Streamline frame identity or one DLSS-G
evaluation, and is identity in every E33 and SB sample. Each limit of motion becomes "no evidence":

- **One-parameter camera paths** (pure rotation, a pure orbit, a straight walk at varying speed)
  scale every image velocity by one number, so HUD moving with the camera rate (sway, head-bob)
  matches one depth layer at every instant.
- **No relative motion** (still camera and owner, sub-tolerance motion, a nearly still third-person
  player) and **no owner** (occluded or off-screen) say nothing; zero motion never counts.
- **Wrong semantics** (a misdeclared or misdecoded sign, scale or subrect) give wrong but
  self-consistent matches that pass any agreement test.

**Timing.** The mask, depth and motion vectors must come from one real frame by contract: E2's
exactness, extended to motion. That holds for Streamline tags of one Present interval and for the
inputs of one DLSS-G evaluation. It does not hold for presented alpha (every dump records
`same_game_frame: "unverified"`; the owning document calls it bounded previous-input reuse), the
one-frame-late layer copy, HUD-less pairs by Present counting, or tags pushed as unaligned.
Some E33 captures (`game3d_59540_*`) record the UI colour and Backbuffer tags on a different
command list from depth and motion. E2's Present-interval rule still pairs them. Stage A2 measures
whether that pairing holds within one frame in practice; if it does not, such frames must abstain.

## Prior art

No shipped tool or published method anchors already-composited UI to its owner's depth
automatically:

- **Exact owner depth has only ever come from inside the renderer.** 3D Vision Automatic
  stereoized each UI draw from its own clip-space `w`. NVIDIA's hybrid patent
  ([US10935788](https://patents.google.com/patent/US10935788)) re-issues UI draws per eye. Engine
  VR ports and UEVR re-project markers using engine knowledge. Each needs draw or shader
  instrumentation, or engine-specific code.
- **Community fixes were per game.** 3Dmigoto and HelixMod fixes
  ([Auto-Crosshair](https://github.com/bo3b/3Dmigoto/wiki/Auto-Crosshair)) found HUD shaders by
  hash and sampled depth in the UI vertex shader. Even those flickered. The Control fix added a key
  that disables depth for target marks and health bars.
- **Generic converters take the depth under or around the UI.** vorpX Z3D, SuperDepth3D's UI modes
  and TriDef all do this, which puts a floating bar on the background.
- **Overlay standards agree on one rigid disparity per element**, kept in front of the nearest
  depth it covers. Examples are DVB subtitles
  ([EN 300 743](https://www.etsi.org/deliver/etsi_EN/300700_300799/300743/01.05.01_60/en_300743v010501p.pdf))
  and Blu-ray plane-plus-offset. The floor below follows the same principle.
- **SDKs give no UI depth.** Streamline, DLSS-G, FSR 3 frame interpolation, XeSS-FG and DirectSR
  expose UI only as a 2-D composite or alpha
  ([DLSS-G guide](https://github.com/NVIDIAGameWorks/Streamline/blob/main/docs/ProgrammingGuideDLSS_G.md)).
  Their only 3-D signals are depth, motion vectors (mandatory) and camera matrices.
- **Most engines project world widgets on the CPU and draw flat quads.** Unreal Screen-space
  widget components, Unity overlay canvases, Scaleform/Coherent and REDengine all do this, so draw
  observation cannot recover the anchor. Depth-tested world UI in the UI pass (RE Engine World GUI)
  is a minority. Under upscaling, an output-size UI target cannot bind the render-resolution depth
  as its depth-stencil view.

Anchoring by motion therefore appears to be new. Nothing in the field proves it either way, which is
why the stages below start with offline replay.

## Design

### Overview

Notation: `h` is the unpinned conditioned field; `C` the scene's display bound,
`SunshineBoundFinalParallax(SunshineHostContainer)`; `w` the pin weight (P1); `W`, `H` the output
size; `W_mv` the motion-vector rect width; `ρ` the consumed depth decoded to inverse depth as the
candidate pass decodes it, before scene gain and zero.

On every armed real frame, after the horizontal pass, the anchor passes run: **cells**, **labels**,
**association**, **motion check and evidence** (valid pairs only), **follow and floor**, **state
machine and ramps**, and **feasibility resolve**, which writes the cell target map `T` and one flag
per 8-row pin group. Then come the anchored probe, a copy of the flagged rows of `h`, the unchanged
`SunshineApplyUICS`, and a correction pass over flagged rows only. Without a world element the
correction writes nothing, so the output is today's byte for byte.

Against the binding rules: (1) evidence pairs only inside one Streamline tag interval or DLSS-G
evaluation, with copies under [api_contract.h](../tools/reshade/api_contract.h); (2) motion vectors,
their constants, frame indices and DLSS-G keys are optional, and without them an element keeps
today's plane; (3) motion the live check cannot confirm arms nothing, and without fresh evidence a
track holds within a bound, only where current depth confirms it; (4) contradiction, the support
bound and the check unlearn in every game with motion, and nothing persists across sessions. Pin
weight is unchanged and no margin or dilation enters the pin; freshness is judged by identity and
age, and every refusal, exit, step and churn is counted.

### Arming

Anchors run only with UI protection in Auto (automatic, partial mask), mode 5 eligible, D3D12, a
Streamline or DLSS-G motion route for the consumed depth, the shader's anchor marker and an open
motion check. Armed frames use plane mode 6 (`display_fraction_anchored`), which `b1`,
`SunshineUIPlaneParallax` and `prepare_adaptive_frame` treat as mode 5; the exporter still requests
mode 5, so the wire, the cursor and the adaptive policy see no change. While a route is eligible but
unarmed, a **census** (cells, labels, motion check) runs on the UI detection status-sample frames
(the 100 ms rule in [game3d_renderer.cpp](../tools/reshade/game3d_renderer.cpp)), read back with the
existing one-outstanding pattern. It arms when the check is open and a candidate's box changed
between two consecutive censuses (nothing enters without motion, so a static crosshair never arms),
and disarms after two censuses without a change and with no pending or world track. Armed, every
real frame runs the anchor passes; generated Presents never do.

### Motion-vector check

Rule 3 forbids trusting a motion declaration, and a wrong one errs consistently, so every pair is
checked against the depth its route already pairs with the motion. On a lattice over the depth rect
(at most 4096 samples, a cost cap), where the decoded motion is at least `2ε`, the check compares
the residual of `log ρ(N)` against `log ρ(N-1)` warped by the decoded motion (L1 about its median,
so a depth-scale change does not enter; far-plane samples skipped) with the residual of each
alternative that real misdeclarations produce: zero; either axis negated; x2 and x1/2; and the
render-to-output ratio and its inverse.

It **passes** when the decoded residual is at most half of every alternative's (factor 2, tuned),
**fails** when an alternative's is lower, and gives no verdict otherwise or with fewer than `n_min`
informative samples. It runs on every census and armed real frame from a two-entry ring of
quarter-resolution `log ρ`. A route signature's gate (provider, format, extent, declared semantics)
opens on a pass and closes on a fail, which returns every track to global and refuses the pair: a
wrong declaration never arms, and a cut without a reset flag fails its pair. The same comparison of
decoded against zero motion on one element's reach samples marks the pair *unexplained* for that
element, which then gets neither evidence nor contradiction.

### Segmentation and tracks

**Cells.** Square cells, the smallest side giving at most 7168 cells: derived, since 32-bit labels
(whole words, for `InterlockedMin`) then take 28 KiB of a Shader Model 5.0 group's 32 KiB, leaving
scratch. That is 35 px at 3840x2160 and 23 px at 2560x1440, and at most 7168 cells at any aspect
ratio inside the 3840x3840 detection bound. `SunshineUIAnchorCellsCS` loads the mask once and
records per cell the texels with `w > 0` (occupancy: one target must cover an element's soft halo,
because the band's collar takes a neighbour's bound), those with `w >= 1` (core: the P1 knee, which
the probe counts), and the outline bounds. `SunshineUIAnchorLabelCS` labels 8-connected components
of occupied cells by minimum cell index (atomic-min union, pointer jumping), independent of thread
order.

**Candidates.** Only candidates gather evidence or enter: the outline box lies inside the central 75%
rectangle the placement probe uses (no compass strips, minimaps, edge-clamped markers or bottom
subtitles), the longer side is at most `H/4` (cost cap; height, because UI scales with height), it
has a core texel, and it is among the first 64 elements, tracked ones first. Size and position never
make an element world; only motion does. A tracked element that stops qualifying keeps its track and
ramps back to global, so leaving the rectangle or growing as an enemy charges never snaps it.

**Tracks.** A GPU table of up to 64 rows (class, box, motion, goal, applied value, owner box, the
pending record with `(d_E, a)`, support tick, exit timer), ping-ponged because association reads the
previous table while writing the next (unlike the T1 hold store, overwritten in place by one reduce);
a header names the writing decision (real sequence, winner kind and signature, scope). Each track
predicts its box: the last box shifted by the last motion, grown by one cell. A component under one
predicted box goes to that track (largest overlap, ties to the lower index). A component under
several (touching bars in a crowd) is split: cells inside exactly one predicted box form that
track's sub-element, and cells inside several or none join the member with the nearest applied
value. Sub-elements keep their own targets; a merged component gives no motion evidence, but follow,
floor and resolve run per sub-element. On a split, the track keeps the part with the best
intersection over union with its predicted box, and other parts start global tracks that inherit the
applied value. An unmatched track waits up to the support bound for its element, then is dropped.

### Motion evidence

**Element motion.** `d_E` is the change of the element's outline box over a valid pair. It is rigid
only if left and right edges moved alike within 1 px, and top and bottom too (derived from integer
bounds). The outline sees a bar's short ends, so a bar sliding along its long axis is measured, and
an HP fill draining inside its frame does not move it; a fill-only drain, a glyph-width change or an
edge cut gives no evidence. With `ε = 1 + 0.5 * W / W_mv` px (box quantization plus half a motion
texel), the motion is **still** at `|d_E|∞ <= 1` px, **moving** at `|d_E|∞ >= 2ε`, otherwise no
evidence.

**Scene motion.** The reach `R_E` is the outline box grown on every side by its longer side (tuned:
the owner lies within one element size of its label). The paired motion vectors over `R_E` are read
once per motion texel, evenly subsampled above 1024 texels, converted with that frame's declared
semantics; `h` and `ρ` are read at each sample's output pixel through the crop and [jitter
registration](reshade-sbs.md#depth-jitter-registration). `M` holds samples within `ε` of `d_E`, `R`
those differing by more than `2ε`. Splitting `R` at its median `ρ`, the median flows and depths of
its nearer and farther halves give a line `f(ρ) = a + b ρ`. Its value `a` at `ρ = 0` is the
depth-independent (rotational) flow for a calibrated decode, and in any case the flow at one fixed
depth, which is all distinctness needs. The flow **discriminates depth** when the halves' median
flows differ by at least `2ε`. `n_min` is 4 distinct motion texels (tuned: the smallest 2x2 patch).

On an explained valid pair:

- **World evidence:** rigid and moving; `|M| >= n_min`, `|R| >= n_min`; the flow discriminates
  depth; p90 minus p10 of `h` over `M` at most `δ = 2/W` (one pixel of per-eye shift either side).
  The goal is the median of `h` over `M`, the owner box the box of `M`; the record keeps `(d_E, a)`.
- **Consistency** (pending or world, goal `t`): rigid, still or moving, and
  `A_t = {p in M : |h(p) - t| <= δ}` holds at least `n_min`; the goal moves to the median over
  `A_t`. Only *discriminating* consistency (also `|R| >= n_min` and depth discriminated) refreshes
  support: a lock-on target does (bar and enemy still, background orbiting); a still scene does not.
- **Contradiction** (pending or world): rigid; at least `n_min` samples at the track's depth refute
  `d_E` under both this pair's flow and the previous pair's flow at the previous position, and fewer
  than `n_min` match. Testing both tolerates a widget projected one frame behind the render.

### Classification

**Entry.** Global becomes pending on world evidence. Pending becomes world on a later world evidence
at least `ui_adaptive::approach_dwell_ms` after the first, with no contradiction between, whose goal
agrees with the followed goal within `δ`, and which is **distinct**: no scalar `s` maps the first
`(d_E, a)` onto this one within `2ε` per component. A disagreeing goal restarts pending.

**Why distinctness.** A one-parameter path scales the element's motion and the camera's flow alike,
and HUD swaying with the camera scales with them, so agreement along it proves nothing. A change in
the mix of rotation and translation, or in the owner's own motion, changes `(d_E, a)`
non-proportionally, and only a world-anchored element still lands on the same depth. A walking enemy
under a steady camera, or a static owner under a pure orbit, stays pending until the mix changes.

**Exit.** A contradiction starts the exit timer and any consistency cancels it; a contradiction at
least `approach_dwell_ms` after the first returns the track to global, as does the support bound. A
one-frame hit-shake or a camera start/stop lag never exits.

### Depth assignment

**Goal and follow.** The goal is the median of `h` over `M` or `A_t`, from integer histograms over
`[-C, C]` with quarter-pixel bins (finer than `δ`), so it is deterministic. Because `h` is the
displayed field, the target is where the owner is shown, with no new depth adapter or camera. On
every armed real frame the goal moves to the median `h` of owner-box samples within `δ` of it, if
there are `n_min`. The goal is only a search key: the target is the goal while follow succeeds and
`pUI` while it fails, so no depth is shown that the current frame does not confirm.

**Applied value**, on the GPU each armed real frame: (1) the target is the goal for a world track
whose follow succeeded, else `pUI`; (2) the value ramps toward it at the global plane's retreat rate
in both directions, `dt` capped at `ui_adaptive::max_age_ms`, because class changes are preferences
and the fast rate exists to clear conflicts; (3) a confirmed floor `F_E` above the value moves it
forward at the global approach rate, and the ramp never takes it below one; (4) it is clamped to
`[-C, C]` and resolved for feasibility. A new world track starts from `pUI`; a returning one becomes
the sentinel within one step of `pUI`. On Roach the goal is the horse, `-0.00002` against `+0.00200`
today, reached in about 0.4 s; its floor (`-0.00041`) does not lift it.

### Conflict, feasibility and comfort

**Floor.** `F_E` is the 80th percentile of `h` over the element's core plus the probe's conflict
margin: the coverage-share half of the global conflict rule (under [Setup](reshade-sbs.md#setup))
applied to one element; the area half has no per-element meaning. As for the global plane, a floor
above the applied value acts only after persisting across distinct observations for
`approach_dwell_ms`, and releases through the ramp. A nearer object over more than a fifth of a bar
brings it forward; briefer or smaller crossings stay local conflicts, as on the HUD plane today.

**Feasibility.** Two rigid targets in shared rows must fit the band: across a texel gap `g` between
occupied outlines it joins at most `D(g) = 0.5 * g / W` (derived from the band's slope and collar).
The single-group resolve fixes global UI at `pUI`, then places world elements in order of their
previous applied value, nearest first, each projected into `[F_E, C]` intersected with `v ± D(g)`
for every placed or global neighbour in shared cell rows. Projection only raises a farther element
next to a nearer one and slides an element continuously toward `pUI` next to HUD. On an empty
interval the floor is dropped first (a local conflict, as today); if still empty, the element shows
`pUI` this frame and restarts its ramp, and the resolve repeats (demotions only accumulate, so at
most 64 rounds). The resolved value becomes the applied value, so a released constraint ramps. A
demotion is a one-frame step only when a constraint appears abruptly; it is counted.

**Comfort and probe.** World targets stay inside `±C`; the live `0.5*C` cap and comfort ceiling
remain HUD-plane policies (an owner question). Vertical steps at an element's edges become `|t - h|`
instead of `|pUI - h|`, usually smaller and unbounded today too. An anchored probe variant skips the
core of every world-class track, at its target or demoted, and with none counts exactly what
`SunshineUIConflictCS` counts: bars over near enemies stop pushing `pUI`, and a demoted bar cannot
drive `pUI` into a limit cycle with its own feasibility. The adaptive policy and cursor are unchanged.

### Temporal holds and unlearning

- **Support bound:** class, goal and owner box last at most `ui_adaptive::release_dwell_ms` after
  the last world evidence or discriminating consistency, the time the global plane holds a forward
  position without fresh need. A fill-only bar or a briefly hidden owner keeps its class within it.
- **Reused decision** (T1 grace): no segmentation or evidence; follow, floor and resolve read the
  current `h`.
- **Drop all tracks:** scope change; full-frame decision (sources 6 or 8, 99% central coverage); held
  hidden verdict (H1).
- **Return all to global**, refusing motion across the boundary: change of winner kind or signature;
  `per_frame_hold_reset`, held.none or an inactive frame; camera or NGX reset; failed motion check.
- **Observation revision:** keeps tracks, as placement does; refuses that pair.

### Frame identity and frame generation

A pair is real frame N and the real frame N-1 whose decision wrote the track table. The CPU sets a
pair-valid bit only when all four conditions hold, else counts the first that failed:

1. **Route.** *Streamline:* the winning UI tag (UIAlpha, UI colour, Backbuffer, or an exact HUD-less
   pair) lies in the Present interval of the consumed depth tag and the type-1 tag, by the interval
   rule of `ui_mask::same_batch`; depth and motion boundaries carry the Present generation UI
   snapshots carry. *DLSS-G:* the UI input, `DLSSG.Depth` as the consumed FG depth, and
   `DLSSG.MVecs` of one evaluation. Never: presented alpha, the layer copy, Present-counted HUD-less
   pairs, unaligned tags, a mask older than its depth, a DLSS-G input with a Streamline tag. NGX SR
   `MotionVectors` has no live route; its only UI partner would be presented alpha.
2. **Consecutive.** Both frames made a fresh decision (decision texel 9 not reused; the GPU checks the
   header's sequence) in one T1 chain with the same winner; the winner's snapshot ids differ; N's
   depth is the next published capture of N-1's source. A Streamline frame index, where present,
   must advance by one.
3. **Semantics**, from the same frame; missing or non-finite values refused. *Streamline:*
   `motion_vector_scale`, `camera_motion_included = 1`, not 3D, not jittered, no reset, invalid
   value excluded. *DLSS-G:* `DLSSG.MvecScaleX`/`Y`, `CameraMotionIncluded = 1`, `MvecJittered = 0`,
   `MvecInvalidValue` excluded, extent from `DLSSG.MVecsSubrect*`. Dilated vectors are accepted:
   dilation spreads foreground motion about one render texel onto the background, which the
   depth-spread test refuses rather than averages. RE9, without frame identity, uses its motion
   vectors' own Present-interval constants, as its depth does.
4. **Fresh.** Depth current (not reused under FG), aged within `freshness::source_age_ms` (API depth
   evidence, the motion capture included), scope unchanged.

**Generated Presents** never run the anchor passes and keep the real frame's field through the field
memo ([game3d_renderer.cpp](../tools/reshade/game3d_renderer.cpp), around line 853), whose key gains
an anchor epoch. Whenever one re-records conditioning anyway (a probe, an armed dump, a `b0`/`b1`
change such as a `pUI` ramp step), the pin and correction reuse the real frame's `T`. A moving bar's
one-frame misregistration remains, now near its owner's depth.

Expected routes, which A2 measures: **eligible** are Hogwarts Legacy sessions that tag a same-batch
Backbuffer (exact HUD-less pair; the 09-30 FG-on gameplay dumps), E33 FG on (Backbuffer tag), SB SDR
FG on and RE9 (UI colour tag), and W3 FG on (UIAlpha). **None** are Hogwarts Legacy's 10-05 Epic
build in every mode, FG off and its title screen (HUD-less by Present counting); E33, SB and W3 with
FG off, AC Shadows and SB HDR (presented alpha or the layer: an owner ruling, roadmap S4); Dead Space
(no mask); and every D3D11 or Generic-depth game.

### Pin pass

`SunshineApplyUICS` is unchanged byte for byte, and so is every row without a world element. The
correction `SunshineApplyUIAnchoredCS` keeps the band's group shape (8 rows by 8 chunks, 16-texel
blocks, summaries and carries). A texel with `w(u) > 0` has target `t(u) = T[cell(u)]` (or `pUI`)
and slack `r(u) = (1 - w(u)) * |h(u) - t(u)|`, with `h` from the unpinned copy;
`U(x) = min_u [t(u) + r(u) + 0.5*max(|x-u|-1, 0)/W]`, `L(x) = max_u [t(u) - r(u) - ramp]`, and the
field becomes `min(max(h, L), U)`. With every target at `pUI` this is today's **UI pin band**. One
best anchor per side and envelope stays exact, so group state doubles to `int4`/`float4` (about
4 KiB). Where the resolve guarantees `L <= U`, the backward and forward scans' nested clamps equal
the combined clamp; elsewhere the result is still a min/max of `0.5/W`-Lipschitz functions, so the
eye pass's fixed-point inversion holds as a defence.

`SunshineUIAnchorUnpinCS` copies the flagged rows into a lazily allocated R32 texture before the
classic pin; the correction rewrites only those rows. Barriers sit at top level, flagged work in
barrier-free branches (a barrier under a groupshared-flag branch does not compile). The renderer and
replay record only direct dispatches, so both passes run on every armed real frame, idle when
nothing is flagged; anchored frames force `pin_apart`. The anchor entries get their own layout
(`b0`, `b1`, an anchor block at `b3` because `b2` bits 12-31 are taken, SRVs above `t14`, UAVs
aliasing `u4`-`u7` under new names); the conditioning layout does not change. The shader test checks
only two named alias pairs, so it must be generalized to compare every entry's reflected bind points.

### CPU, readback, replay and diagnostics

**Capture** lands dump-only in A2, then live. `streamline_buffer_types.h` gains `motion_vectors = 1`;
type-1 tags become an auxiliary kind on the local snapshot path (`record_local_texture` /
`lease_local_view`) in the shared auxiliary budget, copied only while a route is eligible or a dump
armed, under the per-declaration rules; their constants decode into a `motion_data` struct beside
`camera_data` ([streamline_camera_data.h](../tools/reshade/streamline_camera_data.h)).
`DLSSG.MVecs` is copied under an observed state on the evaluation's own list, like its other inputs.
The colour families gain R16G16 and R32G32; a typeless allocation without a declared view is refused.
The [API contract matrix](../tools/reshade/README.md#api-contract-matrix) gains the rows. A pure
header, `game3d_ui_anchor.h`, mirrors every anchor pass in double precision for the oracle, the
sequence ctest and the fixtures.

**Counters and report.** Census, verdicts and counters are read at the status cadence (one
outstanding read). Counters live in the track-table header, leaving the 24-word UI layout untouched:
armed frames, check verdicts, refusals by reason, evidence outcomes, entries and exits by reason,
demotions, floor lifts, **churn** (class transitions and applied-target travel per
world-element-second) and the oldest support age. A `Sunshine UI anchors` log line runs at that
cadence. The [first-run report](../tools/reshade/README.md#first-run-of-a-new-game) gains a
*World-anchored UI* check: FAIL on an entry without the route bit or with the check closed; WARN when
exits of any reason reach half of ten or more entries, or on a failed check; INFO naming each route
that could not anchor. The [frame record](reshade-sbs.md#frame-record) gains route, check, pair
status and world count.

**Dump 3D and replay.** New artifacts are optional on the host, IDs 43-45 stay retired, and a
top-level `ui_anchor` section (at most 64 rows) records route, pairing proofs, semantics, check
verdict, consecutiveness, the previous decision's identity and the `b3` bytes.

| ID | Artifact | Purpose |
| --- | --- | --- |
| 38, 39, 46 | Streamline type 1, NGX `MotionVectors`, `DLSSG.MVecs` (catalog, unpaired) | formats, extents, scales at no live risk |
| 47-49 | `scene_motion` (or `no_same_batch_motion`), `previous_ui_mask`, `previous_depth` (`log ρ`) | single-pair oracle and motion check |
| 50-53 | tracks before and after, cells, targets (`T` and flags; required in mode 6) | replay from frozen state |
| 54 | trace of the last 32 armed real frames' decisions | flip-storm diagnosis |
| 55 | sequence: up to 2 s recorded forward, per frame the occupied cells, pair proofs, check verdict and every candidate's samples (motion, `ρ`, `h`); at most 16 MiB | offline classifier on real sequences |

`ui_parameters` moves to v8 and the shader gains a marker; mode 6 requires both, so an older parser
refuses an anchored dump. [Native replay](reshade-sbs.md#native-frame-replay) binds the frozen
tracks, `scene_motion`, `previous_depth` and `b3`, reruns the anchor passes, and requires tracks,
`T` and the final field to match byte for byte. History is never rerun.

**Constants.** Three tuned values and three cost caps; the rest is derived or reused.

| Constant | Value | Kind |
| --- | --- | --- |
| Reach; `n_min`; check pass factor | longer side; 4 motion texels; 2 | tuned |
| Candidate size, elements, samples | `H/4`; 64; 1024 per element, 4096 for the check | tuned cost caps |
| Core, occupancy; cell grid | P1 knee, `w > 0`; at most 7168 square cells | derived (P1 and collar; SM 5.0 group memory) |
| Candidate region | central 75% | derived (placement-evidence convention) |
| Check alternatives | zero, axis signs, x2, x1/2, render/output ratio | derived (misdeclaration families) |
| `ε`, moving, rigid; `δ`, bin | `1 + 0.5*W/W_mv`, `2ε`, 1 px; `2/W`, `0.25/W` | derived (sampling, integer bounds, display pixel) |
| Floor; proximity | p80 of core + probe margin; `0.5*g/W` | derived (global rule; band slope) |
| Dwell, support bound, rates, `dt` cap | `approach_dwell_ms`, `release_dwell_ms`, retreat and approach rates, `max_age_ms` | reused (`ui_adaptive`) |

## Fallbacks and failure modes

| Case | What the owner sees | Safe |
| --- | --- | --- |
| No route: D3D11, Generic, presented alpha, layer before S4, Present-counted HUD-less | Today's look; the report's INFO row names the route | yes |
| Misdeclared sign, axis, x2 or render/output scale; cut without reset | The check fails: nothing arms, or bars glide back to the HUD plane | yes |
| Scale error the alternatives do not separate (about x1.2) | Matches lean toward a neighbouring layer; entry still needs two distinct agreeing evidences | no, bounded |
| One-parameter path (pure rotation, orbit, straight walk) | Pending stays pending; HUD sway never anchors; a static owner's bar stays on the HUD plane | yes |
| Still camera and owner, or slow owner | HUD plane; after motion that changes its mix the bar glides to the enemy in about 0.4 s | yes |
| Lock-on orbit | An anchored bar holds; one locked before entry stays on `pUI` | yes |
| Walking enemy, steady camera | Pending until either motion changes | yes |
| Fill-only drain, damage number touching the bar, brief occlusion | Keeps its class up to the support bound; glides slowly toward `pUI` while the owner is hidden | yes; churn counted |
| Hit-shake; widget one frame behind the motion | No exit unless the contradiction persists for the dwell | yes |
| Crowd; bar beside HUD or another bar | Stacked bars keep their depths; side-by-side ones slide toward the nearer; floor dropped before a counted one-frame demotion to `pUI` | yes |
| Nearer object over a fifth of the bar for the dwell | Comes forward, then drifts back; briefer crossings stay local conflicts | yes |
| Owner occluded or off-screen (W3 Vesemir, ACS "54m"); HUD over a still player; bar on the player | `pUI`, as today | yes |
| Central marker co-varying with one layer through two distinct motions | Floats at that layer until a contradiction persists for the dwell | no, bounded |
| Moving aim-point reticle (vehicle, turret) | Anchors at the surface it points at; static crosshairs stay on `pUI` | owner question |
| Menu, scope change, H1; over 64, oversize or edge elements | Tracks drop or stay on `pUI`; a tracked element that stops qualifying ramps back | yes |
| Mask leaking onto scene pixels (SB menu character) | They take their own scene depth | yes |
| Generated Presents, FG multiplier changes | The real frame's positions and targets are held | yes |
| Enemy approaching fast | The ramp lags its owner; the confirmed floor keeps it in front of what it covers | yes |

## Cost

Estimates at 3840x2160 with 2228x1253 rendering, scaled from the current 0.22 ms pin and 0.063 ms
mask load (**Detection and pinning shape** under [Native renderer
ownership](reshade-sbs.md#native-renderer-ownership)). The framework requires a Present-interval A/B
with no frame-time cost before a new capture boundary is enabled; it takes the armed state as the
steady state in combat.

| State | Work | Estimate |
| --- | --- | --- |
| Not eligible | none | 0 |
| Eligible, unarmed | quarter-resolution `log ρ` per real frame; census and check per status sample | about 0.01 ms per frame plus 0.07 ms per sample (0.012 ms per frame at 60 fps) |
| Armed, no world element | anchor passes, motion copy (11 MB RG16F), idle correction groups | 0.09-0.16 ms per real frame |
| Armed, world elements on about 10% of rows | plus the flagged rows' copy and correction | 0.12-0.22 ms per real frame |
| Generated Presents | memo hit, or a pin re-record with the real frame's `T` | as today |

VRAM: about 0.5 MB (cells, tracks, trace, `log ρ` ring); while armed, the unpinned copy (31.6 MiB)
and a two-entry motion ring (2 x 11 MiB) in the auxiliary budget. Readback: at most 4 KB per status
sample. Dump 3D: up to 12 extra textures (45-80 MiB at 4K) plus at most 16 MiB of sequence. If the
A/B demands savings: fold the cells into `SunshineUIDetectionMaskCS` behind a selection revision, or
run evidence at the status cadence.

## Evidence plan and stages

Offline first; the headset check comes last.

| Stage | Deliverable | Acceptance |
| --- | --- | --- |
| A0 Combat census (no code) | Dump 3D captures of bars, lock-on, nameplates and prompts, camera moving and still, in E33, SB, HL, W3, ACS, RE9 and BF6, FG on and off; labels in `ui_anchor_cases.json` in the dump root (outside git) | Per game: world elements in the mask versus drawn in the scene, the winning route, and the share each route can anchor. **Go/no-go:** mostly depth-tested bars, rescope to nameplates and markers; a small anchorable share, the owner orders the presented-alpha ruling and roadmap S4 first |
| A1 Single-frame oracle | `inspect_game3d_dump.py --anchor-elements`; `--pin-metrics` with per-texel targets; `envelope()` in `test_game3d_ui_pin_band.h`; label-driven classes | All-sentinel envelopes equal `band()` within 1e-5 px on every dump; one element per labelled element; Roach's target within `δ` of the horse; randomized resolves never give `L > U`; 21:9 and 32:9 fixtures |
| A2 Motion capture and check, dump-only | IDs 38/39/46 and 47-49, `ui_anchor` metadata, decode, check verdicts recorded only, contract rows, formats | Pairing status or named refusal per game and route; the check passes on every eligible route and fails on injected sign, axis and scale errors; UI-to-motion lag per game within one frame; Present-interval A/B unchanged with nothing armed |
| A3 Sequence capture | GPU cells, labels and lattice sampling, running only while a sequence dump (ID 55) records | Equal to the mirror on recorded frames; real combat sequences on every eligible route |
| A4 Offline classifier | `game3d_ui_anchor.h`; a CPU `ui_anchor_replay` modelled on `ui_detection_replay` (PASS/XFAIL/XPASS/FAIL/SKIP, FAIL when nothing ran) over A3 sequences; a `reshade_game3d_ui_anchor_sequence` ctest with the adversaries below; reach and `n_min` sweeps | **No HUD-labelled element reaches world** in any sequence or adversary; world recall per game and camera class (below about 50% in moving combat means revisit); anchor error at most `δ`; churn at most one class transition per two world-element-seconds in moving combat (starting target) |
| A5 GPU passes | Anchor entries, correction, mode 6 with v8 and replay; D3D12 fixtures, D3D11 compilation | GPU equals mirror; tracks and `T` byte-exact on replay; frames without a world element byte-identical across the corpus; `SunshineApplyUICS` disassembly checks unchanged; correction within 1e-5 px, slope at most `0.5/W(1+1e-4)`; injected infeasible pair stays Lipschitz; per-pass cost |
| A6 Temporary live shadow | Behind the Diagnostics switch with `T` forced to the sentinel; counters, log, report, trace; removed at A7 | No FAIL; decisions match labels in same-session sequence dumps; churn within the A4 target; A/B within budget, armed and unarmed |
| A7 Live apply | Mode 6 on eligible routes | Offline replay of session dumps, then the owner's headset check in combat: bars at their enemies, HUD unchanged |
| A8 Coverage | Layer route after S4; presented alpha after an owner ruling; cells folded into the mask pass | The same strict negatives per new route; validated routes unchanged |

Adversaries: still crosshair, subtitles; compass under yaw; clamped marker under yaw then pitch; a
notification sliding during a strafe, with and without easing; HUD sway under a lock-on orbit;
head-bob with walking speed; fast wide-FOV yaw with an `H/4` element; lock-on with an anchored bar;
a steadily walking enemy; HP drains; "54m" to "55m"; stacked bars at different depths; a damage
number merging and separating; a bar crossing the central edge or outgrowing the cap; a bar beside
HUD with a near occluder; injected sign, axis and scale errors; a cut without reset; FG cadence and
multiplier drift; a skipped real frame; split, merge, blink; calibration ramps, scope changes and
observation revisions.

## Alternatives considered

- **Size and edge split (the first hypothesis), depth under each element, window maximum or
  majority.** The RE9 crosshair, W3 tracker, HL marker and subtitles pass the split; the depth under
  an element is farther (Roach), nearer (Talk / Vesemir) or equal (SB dot); window statistics put
  Roach on clutter or trees and make HUD swim.
- **Camera reprojection from `clip_to_prev_clip`.** Blind to an enemy's own motion, needs a
  frame-correlated camera, ambiguous when identity; possibly later, for static anchors.
- **Presented alpha paired by Present interval.** No contract ties it to a tag batch or evaluation,
  and E2 already refuses Present counting; left to an owner ruling.
- **A luma motion check.** Needs a colour paired with the motion by contract, which not every route
  offers, and particles and UI disturb it; depth is already paired and carries neither.
- **Distinctness by element speed, or a scalar fit over a fixed lattice; per-game UI lag.** Speed
  passes HUD sway on one-parameter paths, a moved lattice compares different scene content, and a
  per-game lag is per-game state.
- **Co-motion from depth residuals without motion vectors.** Picks the nearest object whose edges
  move alike, `b0` changes with [scene_gain.h](../tools/reshade/scene_gain.h) almost every pair, and
  it would accept inexact routes.
- **Overlap score for element motion; unified per-texel kernel; groupshared-flag branch; indirect
  dispatch; CPU-owned class policy.** A sliding bar reads as still; rounding changes on HUD-only
  frames; illegal under FXC; a primitive the renderer never records; a readback behind association.
- **The UI draw's own depth (depth-tested UI draws).** This is exact where it exists. But most
  engines draw bars as flat CPU-projected quads, and upscaled titles cannot depth-test an
  output-size UI target against render-resolution depth. Recovering it would also need pipeline
  interception (`init_pipeline` on D3D12) in every game. At most it could be a D3D11 diagnostic
  census.
- **A sequential hypothesis test instead of fixed tolerances.** Each pair would test screen-fixed
  (H0) against the window's depth layers, and decide on a cumulative likelihood ratio at one target
  misfire rate `alpha`, with sigma built from UI and motion-vector quantization. It would replace
  `n_min` and the check's pass factor with one knob, and adds an absolute-fit requirement (the
  winner must fit, not just be least bad). Its abstentions are the same as this design's
  (rotation, still scenes, constant-velocity coincidences). A4 should replay it beside the
  tolerance rules and keep whichever shows fewer HUD misfires at equal recall.
- **NoWarpMask actor IDs** (UE only, unpaired): a possible later tightening. **Regional HUD planes**
  (rejected earlier, `test_game3d_ui_adaptive.cpp`) cannot follow enemies. **Fixed dilation
  margins** are excluded by the removed late-layer margin's precedent.

## Open questions for the owner

1. After A0: if enemy bars are mostly drawn in the scene, continue for nameplates, quest markers and
   prompts?
2. Does any contract tie a Present's alpha to its Streamline tag batch or DLSS-G evaluation? Without
   one, games whose FG-off UI winner is presented alpha (E33, SB, W3, ACS) never anchor.
3. Bound world targets only by `±C`, or also by the HUD plane's `0.5*C` cap and comfort ceiling?
   A stricter alternative: when either bound would bind (nearer than today's UI ceiling, or behind
   something the element overlaps), revert that element to `pUI` instead of clamping it.
4. Floor: per-element p80 (proposed) or the strict maximum over the core (Roach at `+0.00020`
   instead of the horse)? Keep the global plane's 100 ms confirmation?
5. Infeasible neighbours: slide, drop the floor, then a one-frame demotion (proposed), or both to
   `pUI`?
6. Is it acceptable that nothing anchors without relative motion that changes the camera's mix
   (still camera, pure rotation or orbit, slow owners, static owners under an orbit)?
7. May an anchored bar hold `release_dwell_ms` (1.5 s, proposed) without discriminating motion?
   Longer steadies bars in still scenes; shorter clears a wrong anchor sooner.
8. Moving aim-point reticles (vehicle, turret) will anchor at their surface; static crosshairs stay
   on `pUI`. Acceptable, or must aim points stay global (a rule this design lacks)?
9. Should layer-route games wait for roadmap S4?
10. Should elements outside the central 75% never anchor in this version?
11. Evidence on every armed real frame (proposed, about 0.1 ms), or at the status cadence if the A/B
    fails?
12. Is a temporary live shadow (A6, removed at A7) acceptable?
13. Refuse typeless motion-vector allocations (proposed), or read an R16G16 typeless one as FLOAT?
14. Please capture the A0 combat dumps, and the A3 sequence dumps once they exist.
15. Later: should per-element floors also lift HUD behind near geometry (SB corner icons pinned at 0
    over ground at `+0.0025`)?
