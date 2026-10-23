// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The Game 3D API contract oracle: expect(case) returns the outcome that the
// Streamline, D3D12 and NGX contracts require for one assignment of the
// matrix dimensions (test_game3d_api_contract_dimensions.h). Every rule
// carries the header line or specification clause it comes from, so a
// reviewer checks the oracle against the contract text, never against what
// the add-on does today. Where the add-on disagrees, the row belongs in
// test_game3d_api_contract_known_gaps.h; the oracle is not bent to match.
//
// Owner rulings encoded here: R1 COMMON on a non-simultaneous depth-stencil
// allocation is a proven state; R2 a ValidUntilPresent tag with no command
// list, or no observed state at its tag call, is copied on the Present queue
// before that frame's Present; R3 dynamic-resolution extents keep one
// continuous source when format and projection are unchanged; R4 local UI
// inputs follow the tag contract and UI alpha on 2.15 or later stays
// manual-only; R5 the viewport bound to frame generation breaks a tie between
// two active viewports, otherwise mono; R6 an eError* refusal is global
// across viewports. With a2563fa2 the owner also decided N4: DLSS-G's own
// NGX evaluation supplies frame-generation depth only through a copy on its
// own command list in a state recorded there.
//
// Only <d3d12.h>, <cstdint>, the dimensions header and generated SDK
// constant tables may be included (reshade_contract_oracle_independence).
// Buffer-type meanings come from the generated Streamline survey tables.
#include "test_game3d_api_contract_dimensions.h"
#include "test_sl_survey_tables.generated.h"

#include <cstdint>
#include <d3d12.h>

namespace game3d_contract {
  struct rule {
    const char *id, *requirement, *citation;
  };

  // The rules by family: S = Streamline, D = D3D12, N = NGX, U = UI, X =
  // deliberately illegal input (a safe refusal).
  namespace rules {
    inline constexpr rule S1 {"S1", "Lifecycle 0 with a command list, copied at its own tag call: CAPTURE", "SLT:400 eOnlyValidNow may change after the call; SLT:423 a volatile tag needs a valid command buffer"};
    inline constexpr rule S2 {"S2", "A tag other than ValidUntilPresent with no command list: REFUSE no_copy_point (illegal input)", "SLA:145, 162: cmdBuffer 'can be null if ALL tags are null or have eValidUntilPresent life-cycle'"};
    inline constexpr rule S3 {"S3", "Lifecycle 0 global or frame tag used at a later evaluate or after Present: REFUSE lifetime", "SLT:400: 'Resource can change, get destroyed or reused ... after it is provided to SL'"};
    inline constexpr rule S4 {"S4", "Lifecycle 0 local evaluate input with the evaluate's command list: CAPTURE", "SLA:258-262: evaluate inputs may be tags, 'local' tags that do NOT interact with global tags; current at that evaluate"};
    inline constexpr rule S5 {"S5", "Lifecycle 1 at a frame-generation tag call with a command list, or lifecycle 1 or 2 at a same-frame evaluate: CAPTURE", "SLT:402, 404: unchanged until the frame is presented / until slEvaluateFeature has returned"};
    inline constexpr rule S6 {"S6", "Lifecycle 1 with no command list: CAPTURE at the same-frame evaluate, else on the Present queue before that frame's Present (R2)", "SLA:145, 162 (null cmdBuffer legal for eValidUntilPresent); SLT:393-396 (try eValidUntilPresent first); owner ruling R2"};
    inline constexpr rule S7 {"S7", "Lifecycle 1 with 1-3 generated Presents of earlier frames before the evaluate: CAPTURE", "SLT:402: valid 'until the frame is presented', that frame's own Present"};
    inline constexpr rule S8 {"S8", "Lifecycle 1 after its frame's Present, lifecycle 2 after its evaluate returned, or lifecycle 3 or more: REFUSE lifetime / malformed", "SLT:397-405: three lifecycles, each ending at its stated point"};
    inline constexpr rule S9 {"S9", "Untag (null Resource or null native), or a later copy point of a tag the game untagged after setting it: HOLD the earlier capture, then mono; the frame-generation source is not withdrawn", "SLA:143, 160: 'set to null to remove the specified tag'"};
    inline constexpr rule S10 {"S10", "Declared state UINT_MAX and no observation: REFUSE state_unprovable", "SLT:335, 357-359: state is MANDATORY; its UINT_MAX default declares nothing"};
    inline constexpr rule S10a {"S10a", "A declared state at the point it describes, with no observation: CAPTURE from the declaration", "SLT:357: 'State is MANDATORY and needs to be correct when tagged resources are actually used'"};
    inline constexpr rule S11 {"S11", "At a tag call before the declared use (lifecycle 1 or 2), an observed state: CAPTURE from the observation", "SLT:357: the declaration describes the state at use (Present or evaluate), not at the tag call"};
    inline constexpr rule S11a {"S11a", "A declaration that disagrees with the state recorded at the point it describes: EITHER CAPTURE from the observation or REFUSE state_unprovable", "SLT:357 (the declaration is incorrect input); MS Learn, Using Resource Barriers: 'At any given time, a subresource is in exactly one state'"};
    inline constexpr rule S11b {"S11b", "At a tag call before the declared use, no observation: CAPTURE from the declaration through a copy on the Present queue (R2)", "SLT:357; owner ruling R2"};
    inline constexpr rule S12 {"S12", "Resource type 0 or 8: normal; type 1 (buffer): REFUSE not_depth", "SLT:280-292 ResourceType: eTex2d 0, eBuffer 1, eUnknown 8"};
    inline constexpr rule S13 {"S13", "Constants, ViewportHandle, DLSSGOptions, Resource or a single ResourceTag above the newest known version: as at the newest known version", "sl_struct.h:43-106: new members go at the end with a version bump, so the known prefix keeps its meaning"};
    inline constexpr rule S14 {"S14", "Unknown chained struct on a tag or Resource, a ResourceTag above v1 in an array of more than one, a zero Resource header, PrecisionInfo formula 2+ or scale 0: REFUSE unknown_abi / malformed", "sl_struct.h:83-99 (optional data is a chained struct with its own GUID; the array stride is the known size); SLT:437-475 (formulas 0 and 1; scale 0 has no inverse)"};
    inline constexpr rule S15 {"S15", "A tag that is not last in an evaluate input chain: CAPTURE, as in an array", "SLA:250-262: inputs are an array or a chain of BaseStructures"};
    inline constexpr rule S16 {"S16", "Extent null or with a zero side: CAPTURE of the whole resource", "sl_consts.h:150 Extent::operator bool is width != 0 && height != 0; SLT:425 'if using the entire resource leave as null'"};
    inline constexpr rule S16b {"S16b", "Extent inside the allocation (padded or offset): CAPTURE of that region", "sl_consts.h:143-169 Extent {top, left, width, height}; SLT:425 the area of the tagged resource to use"};
    inline constexpr rule S16c {"S16c", "Extent outside the allocation: REFUSE malformed", "SLT:425: the extent is an area of the tagged resource"};
    inline constexpr rule S16d {"S16d", "Declared Resource size differs from the allocation, extent valid: CAPTURE", "SLT:334-336, 360-363: width and height are not among the mandatory Resource fields"};
    inline constexpr rule S16e {"S16e", "Extent changing every frame inside the allocation: CONTINUOUS (R3)", "sl_dlss_g.h:46, 82-90 eDynamicResolutionEnabled and per-frame resolution; owner ruling R3"};
    inline constexpr rule S17 {"S17", "Buffer type B at release V: a depth source exactly when V's own header declares B as depth", "each release's BufferType declarations (sl-abi-survey headers)"};
    inline constexpr rule S17b {"S17b", "An official release's interposer version resource selects that release's header (its exact version, or an inexact family that includes it) and the structure ABI its Resource defines", "sl-abi-survey all-release-dll-results.json (each release DLL's version resource); each release's sl.h Resource (a state word from 1.0.3) and sl_struct.h (versioned 2.x structures)"};
    inline constexpr rule S18 {"S18", "33 to 73 tags in one call with depth among them: CAPTURE; other slots are not wiped", "SLA:143-173: numTags is a uint32_t with no stated limit"};
    inline constexpr rule S19 {"S19", "Two SR viewports active together: the viewport bound to frame generation CAPTUREs, any other REFUSE ambiguous_view (R5)", "SLT:614-621 ViewportHandle (any number); owner ruling R5"};
    inline constexpr rule S19b {"S19b", "Viewports rotating, one active at a time: CAPTURE", "SLT:614-621 ViewportHandle"};
    inline constexpr rule S20 {"S20", "Frame generation Auto and Dynamic are enabled; mode 4+, failed options, Off, or On or Auto with 0 frames leave no frame-generation scope (Dynamic chooses its own count): a frame-generation tag-call copy REFUSE fg_scope, SR unaffected", "sl_dlss_g.h:34-40 DLSSGMode {eOff, eOn, eAuto, eDynamic}; sl_dlss_g.h:75-79 numFramesToGenerate (the multiplier); sl_dlss_g.h:116-118 dynamicTargetFrameRate (Dynamic adapts it)"};
    inline constexpr rule S21 {"S21", "Result 39 (eWarnOutOfVRAM): treated as success", "sl_result.h:74: eWarnOutOfVRAM is the only warning"};
    inline constexpr rule S21b {"S21b", "eError* on the tag or evaluate: that copy is discarded, HOLD, then mono (global across viewports, R6)", "sl_result.h:33-73; owner ruling R6"};
    inline constexpr rule S22 {"S22", "COMPUTE evaluate list with a compute-legal state: CAPTURE", "SLA:253: the command buffer must be created on a device where the feature is supported (no list type is required)"};
    inline constexpr rule S22b {"S22b", "COMPUTE list with a DEPTH_*, PIXEL_SHADER_RESOURCE or RENDER_TARGET state: REFUSE", "Enhanced Barriers, Command Queue Layout Compatibility (COMPUTE has no depth, render-target or pixel-shader layouts)"};
    inline constexpr rule S23 {"S23", "Render pass open at the copy point: HOLD", "D3D12 render passes (BeginRenderPass..EndRenderPass) admit no copy in the pass; fail-safe hold"};
    inline constexpr rule S24 {"S24", "Major version 3 or above, or a multi-tag array on a release newer than the survey: REFUSE unknown_abi; the game is unaffected", "sl-abi-survey REPORT.md (surveyed through 2.14.1); sl_struct.h:43-106 (array stride needs the known size)"};

    inline constexpr rule D0 {"D0", "No state evidence on the recording and no state claim: REFUSE state_unprovable", "MS Learn, Using Resource Barriers: a subresource's state is 'determined by the set of D3D12_RESOURCE_STATES flags supplied to ResourceBarrier'; nothing named it"};
    inline constexpr rule D1 {"D1", "A legal read-only combination, or DEPTH_WRITE, RENDER_TARGET, UNORDERED_ACCESS or COPY_DEST alone, proven on the recording: CAPTURE, restored exactly", "d3d12.h:1762-1788; MS Learn, Read/write resource state restrictions: one write bit alone, or any number of read bits"};
    inline constexpr rule D2 {"D2", "RESOLVE_DEST (legacy state or enhanced layout) proven: CAPTURE", "d3d12.h:1775 RESOLVE_DEST is a single write state; Enhanced Barriers, Command Queue Layout Compatibility lists RESOLVE_DEST for DIRECT"};
    inline constexpr rule D3 {"D3", "COMMON (observed, declared or the enhanced layout) on a non-simultaneous depth-stencil allocation: CAPTURE (R1)", "MS Learn, Using Resource Barriers, Common state promotion: 'Depth-stencil resources must be non-simultaneous-access textures and thus can never be implicitly promoted'; owner ruling R1"};
    inline constexpr rule D3b {"D3b", "Legacy COMMON on any other texture: REFUSE state_unprovable", "MS Learn, Common state promotion: non-simultaneous textures promote from COMMON to NON_PIXEL_SHADER_RESOURCE, PIXEL_SHADER_RESOURCE, COPY_DEST or COPY_SOURCE on first access; simultaneous-access textures to most states"};
    inline constexpr rule D4 {"D4", "A split barrier begun and not ended, BEGIN_ONLY|END_ONLY together, or legacy and enhanced barriers mixed in a split: REFUSE state_unprovable", "MS Learn, Split Barriers: 'While the barrier is pending the (sub)resource cannot be read or written by the GPU'; d3d12.h:1792-1799; Enhanced Barriers SYNC_SPLIT"};
    inline constexpr rule D4b {"D4b", "END_ONLY with its begin on an earlier list: CAPTURE from StateAfter", "MS Learn, Split Barriers: the END_ONLY barrier 'completes the pending transition'"};
    inline constexpr rule D4c {"D4c", "END_ONLY whose StateAfter differs from its begin's: REFUSE (illegal input)", "MS Learn, Split Barriers: the only legal barrier on a pending (sub)resource has 'the same before and after states'"};
    inline constexpr rule D5 {"D5", "An enhanced layout a direct list may use, with access: CAPTURE with enhanced barriers from exactly that layout and access", "d3d12.h:19273-19306; Enhanced Barriers: a texture in an enhanced layout takes a legacy barrier only from LAYOUT_COMMON; Command Queue Layout Compatibility (DIRECT)"};
    inline constexpr rule D5b {"D5b", "Enhanced NO_ACCESS, UNDEFINED, or a layout the recording's queue type cannot use: REFUSE", "Enhanced Barriers: 'Queue type specific layouts may only be used within a compatible command queue'; NO_ACCESS deactivates a subresource; an UNDEFINED layout holds no meaningful data"};
    inline constexpr rule D6 {"D6", "Layout set by an enhanced barrier on an earlier list, nothing on this recording: REFUSE, or an enhanced copy from the claim's equivalent layout", "Enhanced Barriers, legacy interop: an enhanced-layout texture must reach LAYOUT_COMMON before a legacy barrier references it"};
    inline constexpr rule D7 {"D7", "Subresource 0 transitioned separately from other mips or the stencil plane: CAPTURE", "d3d12.h D3D12_RESOURCE_TRANSITION_BARRIER.Subresource: each subresource has its own state; the copy reads subresource 0"};
    inline constexpr rule D8 {"D8", "MSAA, array or 3D resource: REFUSE", "SLT:412-429 and the NGX depth parameters name no slice or sample"};
    inline constexpr rule D9 {"D9", "Every depth family and typeless view: CAPTURE, plane 0 byte-exact", "D3D12_RESOURCE_FLAGS ALLOW_DEPTH_STENCIL: 'CopyTextureRegion must copy a whole subresource'"};
    inline constexpr rule D9b {"D9b", "R16_TYPELESS with no typed format of its family declared: linear depth REFUSE; hardware depth EITHER R16_UNORM (D16_UNORM is the family's only depth format) or REFUSE", "SLT:331-389 Resource.nativeFormat is optional; MS Learn, DXGI_FORMAT: the R16 family holds UNORM (D16_UNORM, R16_UNORM) or FLOAT (R16_FLOAT) data; docs/reshade-sbs.md, Rules for new game behaviour 3 (fail safe)"};
    inline constexpr rule D9c {"D9c", "A format outside the four depth families, or one of their stencil-plane, integer or SNORM formats: REFUSE", "MS Learn, DXGI_FORMAT: D32_FLOAT, D32_FLOAT_S8X24_UINT, D24_UNORM_S8_UINT and D16_UNORM are the depth formats; X32_TYPELESS_G8X24_UINT and X24_TYPELESS_G8_UINT read the stencil plane"};
    inline constexpr rule D10 {"D10", "Generic end of frame with no proven state: REFUSE, or HOLD the earlier frame", "docs/reshade-sbs.md, Rules for new game behaviour 3 (fail safe); nothing names the state at Present"};
    inline constexpr rule D11 {"D11", "ClearDepthStencilView, or BeginRenderPass with a depth CLEAR: CAPTURE from DEPTH_WRITE", "MS Learn D3D12_RESOURCE_STATES: DEPTH_WRITE is used for ClearDepthStencilView"};
    inline constexpr rule D11b {"D11b", "Unbinding a writable DSV after a depth write on the recording: CAPTURE from DEPTH_WRITE; without one: REFUSE", "MS Learn D3D12_RESOURCE_STATES: DEPTH_WRITE with a writable DSV 'when the PSO has depth write enabled', DEPTH_READ when it is disabled"};
    inline constexpr rule D11c {"D11c", "Unbinding a read-only DSV with no observation: REFUSE", "MS Learn D3D12_RESOURCE_STATES: DEPTH_READ 'can be combined with other read states'"};
    inline constexpr rule D11d {"D11d", "A copy before a transition barrier: CAPTURE from that barrier's StateBefore", "MS Learn, Using Resource Barriers: 'The application must ensure that the before and after states of consecutive calls to ResourceBarrier agree'"};
    inline constexpr rule D12 {"D12", "UI clear boundary on an enhanced RENDER_TARGET layout: REFUSE or an enhanced copy; on a SIMULTANEOUS target in COMMON: REFUSE; on legacy RENDER_TARGET: CAPTURE", "MS Learn D3D12_RESOURCE_STATES: ClearRenderTargetView requires RENDER_TARGET; Enhanced Barriers legacy interop; Common state promotion (simultaneous access)"};

    inline constexpr rule N1 {"N1", "SR with Depth via SetD3d12Resource, Use.HW.Depth 0 or 1, subrect valid: CAPTURE from the observed state, otherwise the SDK contract's NON_PIXEL_SHADER_RESOURCE", "dlssg-guide.txt:529, 2842 and DLSS Programming Guide 3.4: inputs 'must be in state D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE'"};
    inline constexpr rule N2 {"N2", "Use.HW.Depth other than 0 or 1 (or absent outside SR), subrect with one side only or out of bounds: REFUSE malformed", "nvsdk_ngx_defs.h:803-812 DLSS.Use.HW.Depth and the subrect keys"};
    inline constexpr rule N3 {"N3", "Evaluate failure: HOLD, then mono", "nvsdk_ngx_defs.h:99-177 NVSDK_NGX_Result_Fail (0xBAD00000)"};
    inline constexpr rule N4 {"N4", "DLSS-G's own evaluation (feature 11): DLSSG.Depth, set as a D3D12 resource or a void pointer the capture owner validates, CAPTUREs as the frame-generation depth only from a state recorded on the evaluation's own command list; no recorded state, or no DLSSG.Depth: REFUSE", "nvsdk_ngx_helpers_dlssg_d3d.h:73-78 (DLSSG.Depth, HUDLess, UI, UIAlpha and Backbuffer); dlssg-guide.txt:2321-2324 (the evaluation's command list); owner decision with a2563fa2 (docs/reshade-sbs.md, DLSS-G's own NGX evaluation): only a recorded state, not dlssg-guide.txt:529-530's NON_PIXEL_SHADER_RESOURCE, is a copy basis"};
    inline constexpr rule N5 {"N5", "Depth set other than through SetD3d12Resource: REFUSE malformed", "nvsdk_ngx_params.h:55-71; nvsdk_ngx_helpers_d3d.h:305 writes Depth with SetD3d12Resource"};
    inline constexpr rule N6 {"N6", "DENY_SHADER_RESOURCE allocation with no observation: REFUSE", "D3D12_RESOURCE_FLAGS: DENY_SHADER_RESOURCE 'disables the resource from transitioning into' NON_PIXEL_SHADER_RESOURCE, so the SDK contract cannot hold"};
    inline constexpr rule N7 {"N7", "COMPUTE list: CAPTURE (NON_PIXEL_SHADER_RESOURCE is legal on compute)", "dlssg-guide.txt:2321-2324: 'Executed on a non-copy command queue'"};
    inline constexpr rule N8 {"N8", "Enhanced layout from an earlier list: as D6", "Enhanced Barriers, legacy interop"};
    inline constexpr rule N9 {"N9", "Release and recreate: CAPTURE resumes", "NGX ReleaseFeature/CreateFeature: a new handle is a new feature"};
    inline constexpr rule N9b {"N9b", "NGX nested inside a Streamline evaluate: one source only", "the nested evaluation consumes the same frame's depth"};

    inline constexpr rule U1 {"U1", "Full-size UI tag at its tag call with a command list: offered; another size: EITHER", "SLT:64-215 BufferType (HUDLessColor, UIColorAndAlpha, Backbuffer); SLA:145"};
    inline constexpr rule U1b {"U1b", "Tag 23 on 1.x (UIHint): ignored", "v1.x sl.h: value 23 is UIHint"};
    inline constexpr rule U1c {"U1c", "68 on 2.11.x, or 69 on 2.12 and later: UI alpha", "v2.11.1 sl_core_types.h UIAlpha = 68; v2.12.0 UIAlpha = 69 and 68 is ResponsivityMask"};
    inline constexpr rule U2 {"U2", "UI tag with lifecycle 1 and no command list: offered through a copy before that frame's Present, as S6", "SLA:145, 162; owner ruling R2"};
    inline constexpr rule U3 {"U3", "Local UI inputs: offered under the same contract (R4)", "SLA:258-262 local tags; owner ruling R4"};
    inline constexpr rule U4 {"U4", "UI alpha on 2.15 or later: manual-only (ignored automatically, R4)", "release newer than the survey; owner ruling R4"};

    inline constexpr rule X1 {"X1", "A state or layout the allocation's flags do not allow: REFUSE state_unprovable (illegal input)", "D3D12_RESOURCE_FLAGS: ALLOW_RENDER_TARGET, ALLOW_DEPTH_STENCIL and ALLOW_UNORDERED_ACCESS each 'enables the resource to transition into' its state; DENY_SHADER_RESOURCE disables the shader-resource states"};
    inline constexpr rule X2 {"X2", "Not a texture state: a write bit with any other bit, a reserved bit, RAYTRACING_ACCELERATION_STRUCTURE, or SHADING_RATE_SOURCE on a depth or UI texture: REFUSE state_unprovable (illegal input)", "MS Learn, Read/write resource state restrictions ('If a write bit is set, then no read-only bit may be set'); D3D12_RESOURCE_STATES (acceleration structures are created in that state; SHADING_RATE_SOURCE marks a VRS image)"};
  }  // namespace rules

  inline constexpr const rule *catalogue[] {
    &rules::S1,
    &rules::S2,
    &rules::S3,
    &rules::S4,
    &rules::S5,
    &rules::S6,
    &rules::S7,
    &rules::S8,
    &rules::S9,
    &rules::S10,
    &rules::S10a,
    &rules::S11,
    &rules::S11a,
    &rules::S11b,
    &rules::S12,
    &rules::S13,
    &rules::S14,
    &rules::S15,
    &rules::S16,
    &rules::S16b,
    &rules::S16c,
    &rules::S16d,
    &rules::S16e,
    &rules::S17,
    &rules::S17b,
    &rules::S18,
    &rules::S19,
    &rules::S19b,
    &rules::S20,
    &rules::S21,
    &rules::S21b,
    &rules::S22,
    &rules::S22b,
    &rules::S23,
    &rules::S24,
    &rules::D0,
    &rules::D1,
    &rules::D2,
    &rules::D3,
    &rules::D3b,
    &rules::D4,
    &rules::D4b,
    &rules::D4c,
    &rules::D5,
    &rules::D5b,
    &rules::D6,
    &rules::D7,
    &rules::D8,
    &rules::D9,
    &rules::D9b,
    &rules::D9c,
    &rules::D10,
    &rules::D11,
    &rules::D11b,
    &rules::D11c,
    &rules::D11d,
    &rules::D12,
    &rules::N1,
    &rules::N2,
    &rules::N3,
    &rules::N4,
    &rules::N5,
    &rules::N6,
    &rules::N7,
    &rules::N8,
    &rules::N9,
    &rules::N9b,
    &rules::U1,
    &rules::U1b,
    &rules::U1c,
    &rules::U2,
    &rules::U3,
    &rules::U4,
    &rules::X1,
    &rules::X2
  };

  namespace detail {
    constexpr bool same_text(const char *a, const char *b) {
      while (*a && *a == *b) {
        ++a;
        ++b;
      }
      return *a == *b;
    }

    // Every rule has an id, a requirement and a citation; ids are unique.
    constexpr bool catalogued() {
      constexpr auto count = sizeof(catalogue) / sizeof(catalogue[0]);
      for (unsigned i = 0; i != count; ++i) {
        const auto &value = *catalogue[i];
        if (!value.id[0] || !value.requirement[0] || !value.citation[0]) {
          return false;
        }
        for (unsigned j = 0; j != i; ++j) {
          if (same_text(catalogue[j]->id, value.id)) {
            return false;
          }
        }
      }
      return true;
    }
  }  // namespace detail

  static_assert(detail::catalogued(), "a rule lacks its requirement or citation, or repeats an id");

  // Outcomes. Every cell also keeps the global invariants, which the layers
  // that run the add-on check: no copy from a state that was not proven, the
  // SDK's return value reaches the game unchanged, no hang, and a clean debug
  // layer on the device tier.
  enum class outcome_kind {
    capture,  // Fresh depth from the tagged resource and extent; bytes and state restored exactly.
    refuse,  // No Present uses this source; mono or a permitted hold; no barrier or copy recorded.
    hold,  // The same capture for at most maximum_source_age_ms (scene_depth_source.h), then mono.
    continuous,  // Every frame captured with the same logical source while the extent changes.
    offered,  // A UI input reaches UI selection.
    ignored  // Not a candidate for this role; nothing recorded.
  };

  enum class refusal {
    none,
    state_unprovable,
    no_copy_point,
    lifetime,
    not_depth,
    malformed,
    unknown_abi,
    ambiguous_view,
    fg_scope
  };

  // The state a copy leaves and restores: a legacy state, or an enhanced
  // layout and access.
  struct copy_basis {
    bool enhanced {};
    std::uint32_t state {}, layout {}, access {};
  };

  // alternative differs from kind only for EITHER, where the contract is
  // silent or permits both; why and basis describe the refuse and capture
  // branches.
  struct outcome {
    outcome_kind kind {outcome_kind::refuse};
    outcome_kind alternative {outcome_kind::refuse};
    refusal why {refusal::none};
    copy_basis basis;
    bool exact_basis {};  // The capture names its basis.
    bool deferred {};  // Captured by a copy before that frame's Present (R2).
    const rule *by {};  // The rule that decided.
    const rule *basis_by {};  // The rule that chose the basis (the state family).
  };

  constexpr bool either(const outcome &value) {
    return value.alternative != value.kind;
  }

  constexpr bool permits(const outcome &value, outcome_kind kind) {
    return value.kind == kind || value.alternative == kind;
  }

  // What the code under test did, in the oracle's terms.
  struct result {
    outcome_kind kind {outcome_kind::refuse};
    refusal why {refusal::none};
    copy_basis basis;
    bool basis_known {};
    bool deferred {};
  };

  constexpr bool same_basis(const copy_basis &a, const copy_basis &b) {
    return a.enhanced == b.enhanced && (a.enhanced ? a.layout == b.layout && a.access == b.access : a.state == b.state);
  }

  constexpr bool satisfies(const outcome &expected, const result &actual) {
    if (!permits(expected, actual.kind)) {
      return false;
    }
    if (actual.kind == outcome_kind::refuse) {
      return actual.why == expected.why;
    }
    if (actual.kind == outcome_kind::capture) {
      return actual.deferred == expected.deferred &&
             (!expected.exact_basis || !actual.basis_known || same_basis(expected.basis, actual.basis));
    }
    return true;
  }

  namespace detail {
    constexpr copy_basis legacy(std::uint32_t state) {
      return {false, state, 0, 0};
    }

    constexpr copy_basis enhanced(std::uint32_t layout, std::uint32_t access) {
      return {true, 0, layout, access};
    }

    constexpr outcome capture(const rule &by, copy_basis basis) {
      return {outcome_kind::capture, outcome_kind::capture, refusal::none, basis, true, false, &by, &by};
    }

    constexpr outcome plain(const rule &by, outcome_kind kind) {
      return {kind, kind, refusal::none, {}, false, false, &by, nullptr};
    }

    constexpr outcome refuse(const rule &by, refusal why) {
      return {outcome_kind::refuse, outcome_kind::refuse, why, {}, false, false, &by, &by};
    }

    // EITHER: first decides kind, rule and (for a capture) the basis; the
    // second outcome adds its kind and refusal class.
    constexpr outcome either(outcome first, const outcome &second) {
      first.alternative = second.kind;
      if (second.kind == outcome_kind::refuse) {
        first.why = second.why;
      }
      if (second.kind == outcome_kind::capture) {
        first.basis = second.basis;
        first.exact_basis = second.exact_basis;
      }
      return first;
    }

    // A legacy state named by an observation or a claim, through the
    // legality and COMMON rules. by names the rule for an ordinary state.
    constexpr outcome legacy_basis(std::uint32_t state, std::uint32_t flags, list_type list, const rule &by) {
      if (!texture_state(state)) {
        return refuse(rules::X2, refusal::state_unprovable);
      }
      if (!state_allowed(state, flags)) {
        return refuse(rules::X1, refusal::state_unprovable);
      }
      if (list == list_type::compute && !compute_list_state(state)) {
        return refuse(rules::S22b, refusal::state_unprovable);
      }
      if (state == D3D12_RESOURCE_STATE_COMMON) {
        return depth_stencil(flags) ? capture(rules::D3, legacy(state)) : refuse(rules::D3b, refusal::state_unprovable);
      }
      return capture(state == D3D12_RESOURCE_STATE_RESOLVE_DEST ? rules::D2 : by, legacy(state));
    }

    // An observed basis against a Streamline declaration (SLT:357).
    constexpr outcome against_declaration(const outcome &observed, const state_case &value, bool agrees) {
      if (observed.kind != outcome_kind::capture || value.source != claim::sl_declared || agrees) {
        return observed;
      }
      if (value.scope == claim_scope::before_use) {
        return capture(rules::S11, observed.basis);
      }
      return either(capture(rules::S11a, observed.basis), refuse(rules::S11a, refusal::state_unprovable));
    }
  }  // namespace detail

  // The state a copy may start from (D-rules, S10/S10a/S11*, N1/N6, X1/X2).
  constexpr outcome expect(const state_case &value) {
    using namespace detail;
    const auto &source = value.resource;
    const auto flags = source.flags;
    if (source.shape == resource_shape::msaa || source.shape == resource_shape::array || source.shape == resource_shape::texture3d) {
      return refuse(rules::D8, refusal::malformed);
    }
    // A barrier on another mip or the stencil plane names nothing about
    // subresource 0, which the copy reads.
    auto seen = source.seen;
    if (source.subresource == subresource_barrier::other_mip_only || source.subresource == subresource_barrier::stencil_plane_only) {
      seen = evidence::none;
    }
    switch (seen) {
      case evidence::split_open:
      case evidence::enhanced_split_open:
      case evidence::begin_and_end:
      case evidence::mixed_split:
        return refuse(rules::D4, refusal::state_unprovable);
      case evidence::end_mismatched:
        return refuse(rules::D4c, refusal::state_unprovable);
      case evidence::enhanced:
        {
          if (source.access == D3D12_BARRIER_ACCESS_NO_ACCESS || source.layout == std::uint32_t(D3D12_BARRIER_LAYOUT_UNDEFINED)) {
            return refuse(rules::D5b, refusal::state_unprovable);
          }
          if (!(value.list == list_type::compute ? compute_queue_layout(source.layout) : direct_queue_layout(source.layout))) {
            return refuse(rules::D5b, refusal::state_unprovable);
          }
          if (source.layout == D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE) {
            return refuse(rules::X2, refusal::state_unprovable);
          }
          if (!layout_allowed(source.layout, flags)) {
            return refuse(rules::X1, refusal::state_unprovable);
          }
          const auto &by = source.layout == D3D12_BARRIER_LAYOUT_RESOLVE_DEST ? rules::D2 : rules::D5;
          // A legacy declaration names no enhanced layout, so it never agrees.
          return against_declaration(capture(by, enhanced(source.layout, source.access)), value, false);
        }
      case evidence::transition:
      case evidence::end_without_begin:
        {
          const auto &by = seen == evidence::end_without_begin             ? rules::D4b :
                           source.subresource == subresource_barrier::zero ? rules::D7 :
                                                                             rules::D1;
          return against_declaration(legacy_basis(source.observed, flags, value.list, by), value, value.declared == source.observed);
        }
      case evidence::earlier_list_enhanced:
        {
          // A legacy barrier is illegal here; only a shader-read claim names
          // an enhanced equivalent (SHADER_RESOURCE layout and access).
          const auto claimed = value.source == claim::ngx_contract ? std::uint32_t(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) :
                               value.source == claim::sl_declared  ? value.declared :
                                                                     0u;
          const auto &by = value.source == claim::ngx_contract ? rules::N8 : rules::D6;
          if (claimed && !(claimed & ~shader_read_states) && !(flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)) {
            return either(refuse(by, refusal::state_unprovable), capture(by, enhanced(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_ACCESS_SHADER_RESOURCE)));
          }
          return refuse(by, refusal::state_unprovable);
        }
      default:  // Nothing, or an aliasing barrier (no state change): no evidence.
        break;
    }
    switch (value.source) {
      case claim::none:
        return refuse(rules::D0, refusal::state_unprovable);
      case claim::sl_undeclared:
        return refuse(rules::S10, refusal::state_unprovable);
      case claim::ngx_contract:
        if (flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) {
          return refuse(rules::N6, refusal::state_unprovable);
        }
        return capture(rules::N1, legacy(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
      default:
        {
          auto out = legacy_basis(value.declared, flags, value.list, rules::S10a);
          if (out.kind == outcome_kind::capture && value.scope == claim_scope::before_use) {
            out.by = out.basis_by = &rules::S11b;
            out.deferred = true;
          }
          return out;
        }
    }
  }

  namespace detail {
    constexpr bool frame_generation_scope(const sl_case &value) {
      const bool counted = value.fg == fg_mode::on || value.fg == fg_mode::automatic;
      return ((counted && value.frames_generated) || value.fg == fg_mode::dynamic) && value.options_result != sl_result::error;
    }

    constexpr bool local(tag_entry entry) {
      return entry == tag_entry::local_array || entry == tag_entry::local_chain_not_last;
    }

    // The rule a non-default dimension contributes to a capture it does not
    // change, in increasing precedence.
    constexpr const rule *attribution(const sl_case &value, const rule *by) {
      if (value.extent == extent_form::one_dimension_zero) {
        by = &rules::S16;
      }
      if (value.extent == extent_form::padded || value.extent == extent_form::offset) {
        by = &rules::S16b;
      }
      if (value.extent == extent_form::declared_size_differs) {
        by = &rules::S16d;
      }
      if (value.tag_header == struct_form::above_known || value.resource_header == struct_form::above_known || value.constants == struct_form::above_known || value.viewport_handle == struct_form::above_known || value.frame_generation_options == struct_form::above_known) {
        by = &rules::S13;
      }
      if (value.entry == tag_entry::local_chain_not_last) {
        by = &rules::S15;
      }
      if (value.tags_in_call > 32) {
        by = &rules::S18;
      }
      if (value.views == viewports::two_active) {
        by = &rules::S19;
      }
      if (value.views == viewports::five_rotating) {
        by = &rules::S19b;
      }
      if (value.tag_result == sl_result::warn_out_of_vram || value.evaluate_result == sl_result::warn_out_of_vram || value.options_result == sl_result::warn_out_of_vram) {
        by = &rules::S21;
      }
      if (value.list == list_type::compute) {
        by = &rules::S22;
      }
      return by;
    }
  }  // namespace detail

  // A Streamline depth tag (S-rules); its copy state comes from expect(state_case).
  constexpr outcome expect(const sl_case &value) {
    using namespace detail;
    if (value.version.major < 1 || value.version.major > 2 || (newest_surveyed < value.version && value.tags_in_call > 1)) {
      return refuse(rules::S24, refusal::unknown_abi);
    }
    if (value.resource == resource_form::null_resource || value.resource == resource_form::null_native) {
      return plain(rules::S9, outcome_kind::hold);
    }
    // An untag after the tag call removes the tag before any later copy point.
    if (value.untag == untag_pattern::null_after_each && value.at != copy_point::tag_call && !local(value.entry)) {
      return plain(rules::S9, outcome_kind::hold);
    }
    if (value.tag_header == struct_form::unknown_chained || value.resource_header == struct_form::unknown_chained || (value.tags_in_call > 1 && value.tag_header == struct_form::above_known)) {
      return refuse(rules::S14, refusal::unknown_abi);
    }
    if (value.precision == precision_form::zero_scale || value.precision == precision_form::unknown_formula) {
      return refuse(rules::S14, refusal::malformed);
    }
    if (value.resource == resource_form::type_buffer) {
      return refuse(rules::S12, refusal::not_depth);
    }
    if (value.meaning != buffer_meaning::depth && value.meaning != buffer_meaning::high_resolution_depth && value.meaning != buffer_meaning::linear_depth) {
      return plain(rules::S17, outcome_kind::ignored);
    }
    if (value.life == lifecycle::undeclared) {
      return refuse(rules::S8, refusal::malformed);
    }
    if (value.extent == extent_form::outside_allocation) {
      return refuse(rules::S16c, refusal::malformed);
    }
    if (value.views == viewports::two_active && !value.fg_bound_view) {
      return refuse(rules::S19, refusal::ambiguous_view);
    }
    if (value.tag_result == sl_result::error || (value.at == copy_point::same_frame_evaluate && value.evaluate_result == sl_result::error)) {
      return plain(rules::S21b, outcome_kind::hold);
    }
    // The copy point against the lifecycle (SLT:397-405) and the command
    // buffer rule (SLA:145, 162).
    const auto until_present = value.life == lifecycle::valid_until_present;
    const rule *point {};
    auto scope = claim_scope::at_use;
    bool deferred {};
    switch (value.at) {
      case copy_point::tag_call:
        if (value.life == lifecycle::only_valid_now) {
          if (value.commands == tag_commands::none) {
            return refuse(rules::S2, refusal::no_copy_point);
          }
          point = &rules::S1;
          break;
        }
        if (!frame_generation_scope(value)) {
          return refuse(rules::S20, refusal::fg_scope);
        }
        if (value.commands == tag_commands::none) {
          if (!until_present) {
            return refuse(rules::S2, refusal::no_copy_point);
          }
          // R2: copied on the Present queue, where the declaration is exact.
          point = &rules::S6;
          deferred = true;
          break;
        }
        point = &rules::S5;
        scope = claim_scope::before_use;
        break;
      case copy_point::same_frame_evaluate:
        if (value.commands == tag_commands::none && (!until_present || local(value.entry))) {
          return refuse(rules::S2, refusal::no_copy_point);
        }
        if (value.life == lifecycle::only_valid_now) {
          if (!local(value.entry)) {
            return refuse(rules::S3, refusal::lifetime);
          }
          point = &rules::S4;
          break;
        }
        // A local input is tagged at its own evaluate: no Present can come between.
        point = value.commands == tag_commands::none                             ? &rules::S6 :
                value.generated_presents && until_present && !local(value.entry) ? &rules::S7 :
                                                                                   &rules::S5;
        break;
      default:
        return refuse(value.life == lifecycle::only_valid_now ? rules::S3 : rules::S8, refusal::lifetime);
    }
    if (value.pass == render_pass::open) {
      return plain(rules::S23, outcome_kind::hold);
    }
    state_case state {value.depth, consumer::depth, value.declared_state == UINT32_MAX ? claim::sl_undeclared : claim::sl_declared, value.declared_state, scope, value.list};
    if (deferred) {
      // The Present-queue copy is outside the tag's (absent) recording.
      state.resource.seen = evidence::none;
    }
    if (value.resource == resource_form::zero_header) {
      // Not an SDK struct: only a state recorded on this list could prove it.
      state.source = claim::none;
      const auto proven = expect(state);
      return proven.kind == outcome_kind::capture && !either(proven) ? either(refuse(rules::S14, refusal::malformed), proven) :
                                                                       refuse(rules::S14, refusal::malformed);
    }
    auto out = expect(state);
    if (out.kind != outcome_kind::capture || either(out)) {
      return out;
    }
    const auto *by = attribution(value, point);
    // A tag-call copy with no observed state is the R2 deferral itself.
    out.by = by == point && out.basis_by == &rules::S11b ? &rules::S11b : by;
    out.deferred = out.deferred || deferred;
    if (value.extent == extent_form::changing) {
      out.kind = out.alternative = outcome_kind::continuous;
      out.by = &rules::S16e;
    }
    return out;
  }

  // An NGX evaluation (N-rules): super resolution, ray reconstruction or
  // another feature through Depth, or DLSS-G through its own DLSSG.* inputs
  // (the subrect dimension is then DLSSG.DepthSubrect*).
  constexpr outcome expect(const ngx_case &value) {
    using namespace detail;
    const bool frame_generation = value.feature == ngx_feature::frame_generation;
    if (frame_generation ? value.depth_set == ngx_depth_set::missing || value.depth_set == ngx_depth_set::null : value.depth_set != ngx_depth_set::d3d12_resource) {
      return refuse(frame_generation ? rules::N4 : rules::N5, refusal::malformed);
    }
    // SR defaults to device depth; other features must state the encoding.
    // DLSSG.Depth is device depth (DLSSG.DepthInverted names its direction).
    const bool encoding = frame_generation || value.hw_depth == ngx_hw_depth::linear || value.hw_depth == ngx_hw_depth::device ||
                          (value.hw_depth == ngx_hw_depth::absent && value.feature == ngx_feature::super_resolution);
    if (!encoding || value.subrect == ngx_subrect::one_side || value.subrect == ngx_subrect::outside) {
      return refuse(rules::N2, refusal::malformed);
    }
    if (value.result == ngx_result::failure) {
      return plain(rules::N3, outcome_kind::hold);
    }
    if (frame_generation) {
      // No contract state: only the evaluation's own recording proves one.
      auto out = expect(state_case {value.depth, consumer::depth, claim::none, 0, claim_scope::at_use, value.list});
      if (!either(out) && (out.kind == outcome_kind::capture || out.by == &rules::D0)) {
        out.by = &rules::N4;
      }
      return out;
    }
    auto out = expect(state_case {value.depth, consumer::depth, claim::ngx_contract, 0, claim_scope::at_use, value.list});
    if (out.kind != outcome_kind::capture || either(out)) {
      return out;
    }
    out.by = value.list == list_type::compute                     ? &rules::N7 :
             value.lifetime == ngx_lifetime::release_recreate     ? &rules::N9 :
             value.lifetime == ngx_lifetime::nested_in_streamline ? &rules::N9b :
                                                                    out.by;
    return out;
  }

  // The Generic path's depth-stencil boundaries (D10, D11*).
  constexpr outcome expect(const generic_case &value) {
    using namespace detail;
    switch (value.boundary) {
      case generic_boundary::clear_before_draws:
      case generic_boundary::clear_after_draws:
      case generic_boundary::render_pass_clear:
        return capture(rules::D11, legacy(D3D12_RESOURCE_STATE_DEPTH_WRITE));
      case generic_boundary::transition_away:
        return legacy_basis(value.transition_before, value.depth.flags, list_type::direct, rules::D11d);
      default:
        break;
    }
    // A state proven on the game's recording names the state exactly.
    if (value.depth.seen != evidence::none && value.depth.seen != evidence::alias_named && value.depth.seen != evidence::alias_wildcard) {
      return expect(state_case {value.depth, consumer::depth, claim::none, 0, claim_scope::at_use, list_type::direct});
    }
    switch (value.boundary) {
      case generic_boundary::unbind_writable_dsv:
        return value.depth_written ? capture(rules::D11b, legacy(D3D12_RESOURCE_STATE_DEPTH_WRITE)) :
                                     refuse(rules::D11b, refusal::state_unprovable);
      case generic_boundary::unbind_read_only_dsv:
        return refuse(rules::D11c, refusal::state_unprovable);
      default:
        return either(refuse(rules::D10, refusal::state_unprovable), plain(rules::D10, outcome_kind::hold));
    }
  }

  // A Streamline UI input (U-rules) or a full-frame UI clear boundary (D12).
  constexpr outcome expect(const ui_case &value) {
    using namespace detail;
    switch (value.clear) {
      case ui_clear_boundary::legacy_render_target:
        return plain(rules::D12, outcome_kind::offered);
      case ui_clear_boundary::enhanced_render_target:
        return either(refuse(rules::D12, refusal::state_unprovable), plain(rules::D12, outcome_kind::offered));
      case ui_clear_boundary::simultaneous_common:
        return refuse(rules::D12, refusal::state_unprovable);
      default:
        break;
    }
    if (value.life == lifecycle::undeclared) {
      return refuse(rules::S8, refusal::malformed);
    }
    switch (value.meaning) {
      case buffer_meaning::ui_hint:
        return plain(rules::U1b, outcome_kind::ignored);
      case buffer_meaning::ui_alpha:
        if (newest_surveyed < value.version) {
          return plain(rules::U4, outcome_kind::ignored);
        }
        break;
      case buffer_meaning::hudless_color:
      case buffer_meaning::ui_color_and_alpha:
      case buffer_meaning::backbuffer:
        break;
      default:
        return plain(rules::S17, outcome_kind::ignored);
    }
    auto out = plain(value.meaning == buffer_meaning::ui_alpha ? rules::U1c : rules::U1, outcome_kind::offered);
    if (value.how == ui_capture::local) {
      out.by = &rules::U3;
    }
    if (value.how == ui_capture::no_command_list) {
      out.by = &rules::U2;
      out.deferred = true;
    }
    if (!value.full_size) {
      out = either(out, plain(rules::U1, outcome_kind::ignored));
    }
    return out;
  }

  namespace detail {
    struct named_meaning {
      const char *name;
      buffer_meaning meaning;
    };

    // The buffer-type names Game 3D gives a role, as the survey headers spell
    // them. 1.0.0 and 1.0.1 named the scaling colours DLSSInputColor and
    // DLSSOutputColor; 1.x value 23 is UIHint (v1.x sl.h).
    inline constexpr named_meaning meanings[] {
      {"Depth", buffer_meaning::depth},
      {"HiResDepth", buffer_meaning::high_resolution_depth},
      {"LinearDepth", buffer_meaning::linear_depth},
      {"HUDLessColor", buffer_meaning::hudless_color},
      {"UIColorAndAlpha", buffer_meaning::ui_color_and_alpha},
      {"UIAlpha", buffer_meaning::ui_alpha},
      {"UIHint", buffer_meaning::ui_hint},
      {"Backbuffer", buffer_meaning::backbuffer},
      {"ScalingInputColor", buffer_meaning::scaling_input},
      {"DLSSInputColor", buffer_meaning::scaling_input},
      {"ScalingOutputColor", buffer_meaning::scaling_output},
      {"DLSSOutputColor", buffer_meaning::scaling_output},
    };

    constexpr sl_version version_of(const survey::release &release) {
      return {release.major, release.minor, release.patch};
    }

    // The surveyed release with this version, or null. Tags that share a
    // version (v2.0.0 and v2.0.0a) declare the same buffer types.
    constexpr const survey::release *surveyed(const sl_version &version) {
      for (const auto &release : survey::releases) {
        if (version_of(release) == version) {
          return &release;
        }
      }
      return nullptr;
    }

    constexpr bool same_types(const survey::release &a, const survey::release &b) {
      if (a.type_count != b.type_count) {
        return false;
      }
      for (std::uint32_t i = 0; i != a.type_count; ++i) {
        if (a.types[i].value != b.types[i].value || !same_text(a.types[i].name, b.types[i].name)) {
          return false;
        }
      }
      return true;
    }

    // One header per version, and newest_surveyed is the survey's newest.
    constexpr bool survey_agrees() {
      sl_version newest {};
      for (const auto &release : survey::releases) {
        if (!same_types(*surveyed(version_of(release)), release)) {
          return false;
        }
        if (newest < version_of(release)) {
          newest = version_of(release);
        }
      }
      return newest == newest_surveyed;
    }
  }  // namespace detail

  static_assert(detail::survey_agrees(), "a surveyed version names two headers, or newest_surveyed is not the survey's newest release");

  // The structure versions the newest surveyed release declares: the newest
  // a game can send that a reader knows (S13).
  constexpr const survey::release &newest_known() {
    return *detail::surveyed(newest_surveyed);
  }

  // The meaning a release's header gives a buffer-type name.
  constexpr buffer_meaning meaning_of(const char *name) {
    for (const auto &entry : detail::meanings) {
      if (detail::same_text(entry.name, name)) {
        return entry.meaning;
      }
    }
    return buffer_meaning::other;
  }

  // The meaning that value.version's own header gives value.raw (S17), or
  // undeclared where the survey holds no such release.
  constexpr buffer_meaning declared_meaning(const buffer_case &value) {
    if (const auto *release = detail::surveyed(value.version)) {
      for (std::uint32_t i = 0; i != release->type_count; ++i) {
        if (release->types[i].value == value.raw) {
          return meaning_of(release->types[i].name);
        }
      }
    }
    return buffer_meaning::undeclared;
  }

  // The role a buffer meaning gives a tag: a depth source (S17), a UI input
  // (U1, U1c), or none (S17, U1b; the scaling colours feed neither).
  constexpr outcome buffer_role(buffer_meaning meaning) {
    using namespace detail;
    switch (meaning) {
      case buffer_meaning::depth:
      case buffer_meaning::high_resolution_depth:
      case buffer_meaning::linear_depth:
        return plain(rules::S17, outcome_kind::capture);
      case buffer_meaning::hudless_color:
      case buffer_meaning::ui_color_and_alpha:
      case buffer_meaning::backbuffer:
        return plain(rules::U1, outcome_kind::offered);
      case buffer_meaning::ui_alpha:
        return plain(rules::U1c, outcome_kind::offered);
      case buffer_meaning::ui_hint:
        return plain(rules::U1b, outcome_kind::ignored);
      default:
        return plain(rules::S17, outcome_kind::ignored);
    }
  }

  // A raw buffer type at the release the interposer identifies (S17, S24):
  // the role that release's own header gives it. The survey holds no header
  // for an unpublished build (2.6.10) or a newer release, so its value may or
  // may not keep the newest surveyed meaning; UI alpha newer than the survey
  // stays manual (U4).
  constexpr outcome expect(const buffer_case &value) {
    using namespace detail;
    if (value.version.major < 1 || value.version.major > 2) {
      return refuse(rules::S24, refusal::unknown_abi);
    }
    if (!surveyed(value.version)) {
      const auto newest = buffer_role(declared_meaning({newest_surveyed, value.raw}));
      if (newest.by == &rules::U1c && newest_surveyed < value.version) {
        return plain(rules::U4, outcome_kind::ignored);
      }
      return newest.kind == outcome_kind::ignored ? newest : either(newest, plain(rules::S17, outcome_kind::ignored));
    }
    return buffer_role(declared_meaning(value));
  }

  // Whether a dedicated UI-alpha tag may be used automatically (U1c, U4):
  // only where a surveyed header declares UIAlpha.
  constexpr bool ui_alpha_automatic(const sl_version &version) {
    if (const auto *release = detail::surveyed(version)) {
      for (std::uint32_t i = 0; i != release->type_count; ++i) {
        if (meaning_of(release->types[i].name) == buffer_meaning::ui_alpha) {
          return true;
        }
      }
    }
    return false;
  }

  // The structure ABI a surveyed release's headers define (S17b).
  constexpr sl_abi declared_abi(const survey::release &release) {
    return release.resource_tag ? sl_abi::typed : release.resource_state ? sl_abi::legacy :
                                                                           sl_abi::legacy_no_state;
  }

  // The typeless storage a depth copy keeps and the view that reads the depth
  // plane (D9, D9b, D9c). kind is capture or refuse; alternative differs for
  // EITHER.
  struct format_reading {
    outcome_kind kind {outcome_kind::refuse};
    outcome_kind alternative {outcome_kind::refuse};
    std::uint32_t storage {}, view {};
    const rule *by {};
  };

  namespace detail {
    constexpr format_reading reading(std::uint32_t storage, std::uint32_t view) {
      return {outcome_kind::capture, outcome_kind::capture, storage, view, &rules::D9};
    }
  }  // namespace detail

  // A depth input's formats (D9, D9b, D9c), from DXGI's format families (MS
  // Learn, Format casting): every format of a depth family shares its
  // typeless storage, and the depth plane reads through one view.
  constexpr format_reading expect(const format_case &value) {
    using detail::reading;
    switch (value.allocation) {
      case DXGI_FORMAT_D32_FLOAT:
      case DXGI_FORMAT_R32_TYPELESS:
      case DXGI_FORMAT_R32_FLOAT:
        return reading(DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_FLOAT);
      case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
      case DXGI_FORMAT_R32G8X24_TYPELESS:
      case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
        return reading(DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS);
      case DXGI_FORMAT_D24_UNORM_S8_UINT:
      case DXGI_FORMAT_R24G8_TYPELESS:
      case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
        return reading(DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_R24_UNORM_X8_TYPELESS);
      case DXGI_FORMAT_D16_UNORM:
      case DXGI_FORMAT_R16_UNORM:
        return reading(DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM);
      case DXGI_FORMAT_R16_FLOAT:
        return reading(DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_FLOAT);
      case DXGI_FORMAT_R16_TYPELESS:
        // A typed allocation reads as itself; a typeless R16 one needs the
        // declared typed format of its family.
        if (value.declared == DXGI_FORMAT_R16_FLOAT) {
          return reading(DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_FLOAT);
        }
        if (value.declared == DXGI_FORMAT_R16_UNORM || value.declared == DXGI_FORMAT_D16_UNORM) {
          return reading(DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM);
        }
        if (value.device_depth) {
          return {outcome_kind::capture, outcome_kind::refuse, DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM, &rules::D9b};
        }
        return {outcome_kind::refuse, outcome_kind::refuse, 0, 0, &rules::D9b};
      default:
        return {outcome_kind::refuse, outcome_kind::refuse, 0, 0, &rules::D9c};
    }
  }

  // Pinned rows: one case per rule an expect() decides, written from the rule
  // texts above. They keep every such rule reachable and its outcome fixed
  // as the oracle grows.
  namespace pinned {
    constexpr resource_state recorded(std::uint32_t flags, evidence seen, std::uint32_t observed = 0, std::uint32_t layout = 0, std::uint32_t access = 0) {
      resource_state value;
      value.flags = flags;
      value.seen = seen;
      value.observed = observed;
      value.layout = layout;
      value.access = access;
      return value;
    }

    constexpr state_case state(resource_state resource, claim source = claim::none, std::uint32_t declared = 0, claim_scope scope = claim_scope::at_use) {
      state_case value;
      value.resource = resource;
      value.source = source;
      value.declared = declared;
      value.scope = scope;
      return value;
    }

    template<class Case, class Field, class Value>
    constexpr Case with(Case value, Field Case::*field, Value assigned) {
      value.*field = assigned;
      return value;
    }

    constexpr bool decides(const outcome &value, const rule &by, outcome_kind kind) {
      return value.by == &by && value.kind == kind;
    }

    constexpr bool refuses(const outcome &value, const rule &by, refusal why) {
      return decides(value, by, outcome_kind::refuse) && !either(value) && value.why == why;
    }

    constexpr bool captures(const outcome &value, const rule &by, std::uint32_t state) {
      return decides(value, by, outcome_kind::capture) && !either(value) && !value.basis.enhanced && value.basis.state == state;
    }

    constexpr bool undecided(const outcome &value, const rule &by) {
      return value.by == &by && either(value);
    }

    constexpr auto ds = std::uint32_t(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    constexpr auto depth_read = std::uint32_t(D3D12_RESOURCE_STATE_DEPTH_READ);
    constexpr auto depth_write = std::uint32_t(D3D12_RESOURCE_STATE_DEPTH_WRITE);
    constexpr auto nps = std::uint32_t(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    constexpr auto resolve_dest = std::uint32_t(D3D12_RESOURCE_STATE_RESOLVE_DEST);

    static_assert(refuses(expect(state(recorded(ds, evidence::none))), rules::D0, refusal::state_unprovable));
    static_assert(captures(expect(state(recorded(ds, evidence::transition, depth_read))), rules::D1, depth_read));
    static_assert(captures(expect(state(recorded(0, evidence::transition, resolve_dest))), rules::D2, resolve_dest));
    static_assert(captures(expect(state(recorded(ds, evidence::transition, 0))), rules::D3, 0));
    static_assert(captures(expect(state(recorded(ds, evidence::none), claim::sl_declared, 0)), rules::D3, 0));
    static_assert(refuses(expect(state(recorded(0, evidence::transition, 0))), rules::D3b, refusal::state_unprovable));
    static_assert(refuses(expect(state(recorded(D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS, evidence::none), claim::sl_declared, 0)), rules::D3b, refusal::state_unprovable));
    static_assert(refuses(expect(state(recorded(ds, evidence::split_open, depth_read))), rules::D4, refusal::state_unprovable));
    static_assert(captures(expect(state(recorded(ds, evidence::end_without_begin, depth_read))), rules::D4b, depth_read));
    static_assert(refuses(expect(state(recorded(ds, evidence::end_mismatched, depth_read))), rules::D4c, refusal::state_unprovable));
    static_assert(decides(expect(state(recorded(ds, evidence::enhanced, 0, D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ, D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ))), rules::D5, outcome_kind::capture));
    static_assert(refuses(expect(state(recorded(ds, evidence::enhanced, 0, D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE, D3D12_BARRIER_ACCESS_SHADER_RESOURCE))), rules::D5b, refusal::state_unprovable));
    static_assert(refuses(expect(state(recorded(ds, evidence::enhanced, 0, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_ACCESS_NO_ACCESS))), rules::D5b, refusal::state_unprovable));
    static_assert(undecided(expect(state(recorded(ds, evidence::earlier_list_enhanced), claim::sl_declared, nps)), rules::D6));
    static_assert(with(state(recorded(ds, evidence::transition, depth_read)), &state_case::list, list_type::compute).list == list_type::compute);
    static_assert(refuses(expect(with(state(recorded(ds, evidence::transition, depth_read)), &state_case::list, list_type::compute)), rules::S22b, refusal::state_unprovable));
    static_assert(refuses(expect(state(recorded(ds, evidence::none), claim::sl_undeclared, UINT32_MAX)), rules::S10, refusal::state_unprovable));
    static_assert(captures(expect(state(recorded(ds, evidence::none), claim::sl_declared, depth_write)), rules::S10a, depth_write));
    static_assert(captures(expect(state(recorded(ds, evidence::transition, depth_read), claim::sl_declared, nps, claim_scope::before_use)), rules::S11, depth_read));
    static_assert(undecided(expect(state(recorded(ds, evidence::transition, depth_read), claim::sl_declared, nps)), rules::S11a));
    static_assert(captures(expect(state(recorded(ds, evidence::none), claim::sl_declared, nps, claim_scope::before_use)), rules::S11b, nps) && expect(state(recorded(ds, evidence::none), claim::sl_declared, nps, claim_scope::before_use)).deferred);
    static_assert(captures(expect(state(recorded(ds, evidence::none), claim::ngx_contract)), rules::N1, nps));
    static_assert(refuses(expect(state(recorded(ds | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE, evidence::none), claim::ngx_contract)), rules::N6, refusal::state_unprovable));
    static_assert(undecided(expect(state(recorded(ds, evidence::earlier_list_enhanced), claim::ngx_contract)), rules::N8));
    static_assert(refuses(expect(state(recorded(0, evidence::transition, depth_write))), rules::X1, refusal::state_unprovable));
    static_assert(refuses(expect(state(recorded(ds, evidence::transition, depth_write | depth_read))), rules::X2, refusal::state_unprovable));

    constexpr state_case on_subresource(subresource_barrier subresource, resource_shape shape) {
      auto value = state(recorded(ds, evidence::transition, depth_read));
      value.resource.subresource = subresource;
      value.resource.shape = shape;
      return value;
    }

    static_assert(captures(expect(on_subresource(subresource_barrier::zero, resource_shape::texture2d_mips)), rules::D7, depth_read));
    static_assert(refuses(expect(on_subresource(subresource_barrier::stencil_plane_only, resource_shape::texture2d)), rules::D0, refusal::state_unprovable));
    static_assert(refuses(expect(on_subresource(subresource_barrier::all, resource_shape::msaa)), rules::D8, refusal::malformed));

    constexpr sl_case tag(lifecycle life, copy_point at, tag_commands commands = tag_commands::game_list, tag_entry entry = tag_entry::set_tag) {
      sl_case value;
      value.life = life;
      value.at = at;
      value.commands = commands;
      value.entry = entry;
      value.fg = fg_mode::on;
      return value;
    }

    constexpr auto now = lifecycle::only_valid_now;
    constexpr auto present = lifecycle::valid_until_present;
    constexpr auto evaluate = copy_point::same_frame_evaluate;

    static_assert(captures(expect(tag(now, copy_point::tag_call)), rules::S1, nps));
    static_assert(refuses(expect(tag(now, copy_point::tag_call, tag_commands::none)), rules::S2, refusal::no_copy_point));
    static_assert(refuses(expect(tag(now, evaluate)), rules::S3, refusal::lifetime));
    static_assert(captures(expect(tag(now, evaluate, tag_commands::game_list, tag_entry::local_array)), rules::S4, nps));
    static_assert(captures(expect(tag(present, evaluate)), rules::S5, nps));
    static_assert(captures(expect(with(tag(present, copy_point::tag_call), &sl_case::depth, recorded(ds, evidence::transition, depth_read))), rules::S5, depth_read));
    static_assert(captures(expect(tag(present, copy_point::tag_call)), rules::S11b, nps) && expect(tag(present, copy_point::tag_call)).deferred);
    static_assert(captures(expect(tag(present, copy_point::tag_call, tag_commands::none)), rules::S6, nps) && expect(tag(present, copy_point::tag_call, tag_commands::none)).deferred);
    static_assert(captures(expect(with(tag(present, evaluate), &sl_case::generated_presents, 3u)), rules::S7, nps));
    static_assert(refuses(expect(tag(present, copy_point::after_present)), rules::S8, refusal::lifetime));
    static_assert(refuses(expect(tag(lifecycle::undeclared, evaluate)), rules::S8, refusal::malformed));
    static_assert(decides(expect(with(tag(present, evaluate), &sl_case::resource, resource_form::null_native)), rules::S9, outcome_kind::hold));
    static_assert(decides(expect(with(tag(present, evaluate), &sl_case::untag, untag_pattern::null_after_each)), rules::S9, outcome_kind::hold));
    static_assert(captures(expect(with(tag(present, copy_point::tag_call, tag_commands::none), &sl_case::untag, untag_pattern::null_after_each)), rules::S6, nps));
    static_assert(captures(expect(with(tag(present, evaluate, tag_commands::game_list, tag_entry::local_array), &sl_case::generated_presents, 3u)), rules::S5, nps));
    static_assert(refuses(expect(with(tag(present, evaluate), &sl_case::resource, resource_form::type_buffer)), rules::S12, refusal::not_depth));
    static_assert(captures(expect(with(tag(present, evaluate), &sl_case::resource, resource_form::type_unknown)), rules::S5, nps));
    static_assert(captures(expect(with(tag(present, evaluate), &sl_case::constants, struct_form::above_known)), rules::S13, nps));
    static_assert(refuses(expect(with(tag(present, evaluate), &sl_case::tag_header, struct_form::unknown_chained)), rules::S14, refusal::unknown_abi));
    static_assert(refuses(expect(with(tag(present, evaluate), &sl_case::precision, precision_form::zero_scale)), rules::S14, refusal::malformed));
    static_assert(captures(expect(tag(now, evaluate, tag_commands::game_list, tag_entry::local_chain_not_last)), rules::S15, nps));
    static_assert(captures(expect(with(tag(present, evaluate), &sl_case::extent, extent_form::one_dimension_zero)), rules::S16, nps));
    static_assert(captures(expect(with(tag(present, evaluate), &sl_case::extent, extent_form::offset)), rules::S16b, nps));
    static_assert(refuses(expect(with(tag(present, evaluate), &sl_case::extent, extent_form::outside_allocation)), rules::S16c, refusal::malformed));
    static_assert(captures(expect(with(tag(present, evaluate), &sl_case::extent, extent_form::declared_size_differs)), rules::S16d, nps));
    static_assert(decides(expect(with(tag(present, evaluate), &sl_case::extent, extent_form::changing)), rules::S16e, outcome_kind::continuous));
    static_assert(decides(expect(with(tag(present, evaluate), &sl_case::meaning, buffer_meaning::undeclared)), rules::S17, outcome_kind::ignored));
    static_assert(captures(expect(with(tag(present, evaluate), &sl_case::tags_in_call, 73u)), rules::S18, nps));
    static_assert(refuses(expect(with(tag(present, evaluate), &sl_case::views, viewports::two_active)), rules::S19, refusal::ambiguous_view));
    static_assert(captures(expect(with(with(tag(present, evaluate), &sl_case::views, viewports::two_active), &sl_case::fg_bound_view, true)), rules::S19, nps));
    static_assert(captures(expect(with(tag(present, evaluate), &sl_case::views, viewports::five_rotating)), rules::S19b, nps));
    static_assert(refuses(expect(with(tag(present, copy_point::tag_call), &sl_case::fg, fg_mode::undeclared)), rules::S20, refusal::fg_scope));
    static_assert(refuses(expect(with(tag(present, copy_point::tag_call), &sl_case::frames_generated, 0u)), rules::S20, refusal::fg_scope));
    static_assert(captures(expect(with(tag(present, copy_point::tag_call), &sl_case::fg, fg_mode::dynamic)), rules::S11b, nps));
    static_assert(captures(expect(with(with(tag(present, copy_point::tag_call), &sl_case::fg, fg_mode::dynamic), &sl_case::frames_generated, 0u)), rules::S11b, nps));
    static_assert(refuses(expect(with(with(tag(present, copy_point::tag_call), &sl_case::fg, fg_mode::automatic), &sl_case::frames_generated, 0u)), rules::S20, refusal::fg_scope));
    static_assert(captures(expect(with(tag(present, evaluate), &sl_case::evaluate_result, sl_result::warn_out_of_vram)), rules::S21, nps));
    static_assert(decides(expect(with(tag(present, evaluate), &sl_case::evaluate_result, sl_result::error)), rules::S21b, outcome_kind::hold));
    static_assert(captures(expect(with(tag(present, evaluate), &sl_case::list, list_type::compute)), rules::S22, nps));
    static_assert(decides(expect(with(tag(present, evaluate), &sl_case::pass, render_pass::open)), rules::S23, outcome_kind::hold));
    static_assert(refuses(expect(with(tag(present, evaluate), &sl_case::version, sl_version {3, 0, 0})), rules::S24, refusal::unknown_abi));
    static_assert(refuses(expect(with(with(tag(present, evaluate), &sl_case::version, sl_version {2, 15, 0}), &sl_case::tags_in_call, 2u)), rules::S24, refusal::unknown_abi));

    static_assert(captures(expect(ngx_case {}), rules::N1, nps));
    static_assert(refuses(expect(with(ngx_case {}, &ngx_case::hw_depth, ngx_hw_depth::other)), rules::N2, refusal::malformed));
    static_assert(refuses(expect(with(ngx_case {}, &ngx_case::feature, ngx_feature::ray_reconstruction)), rules::N2, refusal::malformed));
    static_assert(decides(expect(with(ngx_case {}, &ngx_case::result, ngx_result::failure)), rules::N3, outcome_kind::hold));
    static_assert(refuses(expect(with(ngx_case {}, &ngx_case::feature, ngx_feature::frame_generation)), rules::N4, refusal::state_unprovable));
    static_assert(captures(expect(with(with(ngx_case {}, &ngx_case::feature, ngx_feature::frame_generation), &ngx_case::depth, recorded(ds, evidence::transition, depth_read))), rules::N4, depth_read));
    static_assert(captures(expect(with(with(with(ngx_case {}, &ngx_case::feature, ngx_feature::frame_generation), &ngx_case::depth_set, ngx_depth_set::void_pointer), &ngx_case::depth, recorded(ds, evidence::transition, nps))), rules::N4, nps));
    static_assert(refuses(expect(with(with(ngx_case {}, &ngx_case::feature, ngx_feature::frame_generation), &ngx_case::depth_set, ngx_depth_set::missing)), rules::N4, refusal::malformed));
    static_assert(refuses(expect(with(ngx_case {}, &ngx_case::depth_set, ngx_depth_set::void_pointer)), rules::N5, refusal::malformed));
    static_assert(captures(expect(with(ngx_case {}, &ngx_case::list, list_type::compute)), rules::N7, nps));
    static_assert(captures(expect(with(ngx_case {}, &ngx_case::lifetime, ngx_lifetime::release_recreate)), rules::N9, nps));
    static_assert(captures(expect(with(ngx_case {}, &ngx_case::lifetime, ngx_lifetime::nested_in_streamline)), rules::N9b, nps));

    static_assert(captures(expect(generic_case {}), rules::D11, depth_write));
    static_assert(captures(expect(with(with(generic_case {}, &generic_case::boundary, generic_boundary::unbind_writable_dsv), &generic_case::depth_written, true)), rules::D11b, depth_write));
    static_assert(refuses(expect(with(generic_case {}, &generic_case::boundary, generic_boundary::unbind_writable_dsv)), rules::D11b, refusal::state_unprovable));
    static_assert(refuses(expect(with(generic_case {}, &generic_case::boundary, generic_boundary::unbind_read_only_dsv)), rules::D11c, refusal::state_unprovable));
    static_assert(captures(expect(with(with(generic_case {}, &generic_case::boundary, generic_boundary::transition_away), &generic_case::transition_before, depth_read)), rules::D11d, depth_read));
    static_assert(undecided(expect(with(generic_case {}, &generic_case::boundary, generic_boundary::end_of_frame)), rules::D10));

    static_assert(decides(expect(ui_case {}), rules::U1, outcome_kind::offered));
    static_assert(decides(expect(with(ui_case {}, &ui_case::meaning, buffer_meaning::ui_hint)), rules::U1b, outcome_kind::ignored));
    static_assert(decides(expect(with(ui_case {}, &ui_case::meaning, buffer_meaning::ui_alpha)), rules::U1c, outcome_kind::offered));
    static_assert(decides(expect(with(with(ui_case {}, &ui_case::how, ui_capture::no_command_list), &ui_case::life, present)), rules::U2, outcome_kind::offered));
    static_assert(decides(expect(with(ui_case {}, &ui_case::how, ui_capture::local)), rules::U3, outcome_kind::offered));
    static_assert(decides(expect(with(with(ui_case {}, &ui_case::meaning, buffer_meaning::ui_alpha), &ui_case::version, sl_version {2, 15, 0})), rules::U4, outcome_kind::ignored));
    static_assert(decides(expect(with(ui_case {}, &ui_case::clear, ui_clear_boundary::legacy_render_target)), rules::D12, outcome_kind::offered));
    static_assert(undecided(expect(with(ui_case {}, &ui_case::clear, ui_clear_boundary::enhanced_render_target)), rules::D12));
    static_assert(refuses(expect(with(ui_case {}, &ui_case::clear, ui_clear_boundary::simultaneous_common)), rules::D12, refusal::state_unprovable));

    static_assert(decides(expect(buffer_case {{2, 2, 1}, 48}), rules::S17, outcome_kind::capture) && declared_meaning({{2, 2, 1}, 49}) == buffer_meaning::linear_depth);
    static_assert(decides(expect(buffer_case {{2, 2, 0}, 48}), rules::S17, outcome_kind::ignored) && declared_meaning({{2, 2, 0}, 48}) == buffer_meaning::undeclared);
    static_assert(decides(expect(buffer_case {{1, 1, 1}, 23}), rules::U1b, outcome_kind::ignored));
    static_assert(decides(expect(buffer_case {{1, 0, 0}, 3}), rules::S17, outcome_kind::ignored) && declared_meaning({{1, 0, 0}, 3}) == buffer_meaning::scaling_input);
    static_assert(decides(expect(buffer_case {{2, 11, 1}, 68}), rules::U1c, outcome_kind::offered));
    static_assert(decides(expect(buffer_case {{2, 12, 0}, 68}), rules::S17, outcome_kind::ignored) && declared_meaning({{2, 12, 0}, 68}) == buffer_meaning::other);
    static_assert(decides(expect(buffer_case {{2, 14, 1}, 69}), rules::U1c, outcome_kind::offered));
    static_assert(refuses(expect(buffer_case {{3, 0, 0}, 0}), rules::S24, refusal::unknown_abi));
    static_assert(decides(expect(buffer_case {{2, 15, 0}, 69}), rules::U4, outcome_kind::ignored));
    static_assert(undecided(expect(buffer_case {{2, 6, 10}, 0}), rules::S17));
    static_assert(ui_alpha_automatic({2, 11, 1}) && ui_alpha_automatic({2, 12, 0}) && !ui_alpha_automatic({2, 10, 3}) && !ui_alpha_automatic({2, 15, 0}));
    static_assert(declared_abi(*detail::surveyed({1, 0, 2})) == sl_abi::legacy_no_state && declared_abi(*detail::surveyed({1, 0, 3})) == sl_abi::legacy && declared_abi(*detail::surveyed({2, 0, 0})) == sl_abi::typed);

    constexpr bool reads(const format_reading &value, const rule &by, std::uint32_t storage, std::uint32_t view) {
      return value.by == &by && value.kind == outcome_kind::capture && value.alternative == outcome_kind::capture && value.storage == storage && value.view == view;
    }

    static_assert(reads(expect(format_case {DXGI_FORMAT_D24_UNORM_S8_UINT}), rules::D9, DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_R24_UNORM_X8_TYPELESS));
    static_assert(reads(expect(format_case {DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_FLOAT, false}), rules::D9, DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_FLOAT));
    static_assert(expect(format_case {DXGI_FORMAT_R16_TYPELESS}).by == &rules::D9b && expect(format_case {DXGI_FORMAT_R16_TYPELESS}).alternative == outcome_kind::refuse);
    static_assert(expect(format_case {DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_UNKNOWN, false}).kind == outcome_kind::refuse);
    static_assert(expect(format_case {DXGI_FORMAT_X24_TYPELESS_G8_UINT}).by == &rules::D9c && expect(format_case {DXGI_FORMAT_X24_TYPELESS_G8_UINT}).kind == outcome_kind::refuse);
  }  // namespace pinned
}  // namespace game3d_contract
