// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Where the add-on does not meet the API contract oracle yet. Each row names
// the oracle rule it misses, a predicate over the matrix dimensions that
// selects exactly the cells concerned, and an owner note.
//
// expected_failure: every matching cell must fail the oracle. A matching cell
// that passes fails the run, so the fix that closes a gap also removes its
// row (and updates any test that pinned the old behaviour). A row that
// matches no failing cell fails the run as stale (in a family judged with
// judge_each, no failing cell that it alone matches).
// pending: the cell's verdict is not enforced (EITHER) until owner work
// lands or an owner ruling decides it; the global invariants still apply.
//
// Rows describe the add-on's current behaviour in the oracle's dimension
// vocabulary; they never feed back into expect(). Like the oracle, this
// header includes nothing from the add-on.
#include "test_game3d_api_contract_dimensions.h"
#include "test_game3d_api_contract_oracle.h"

#include <cstdint>
#include <d3d12.h>

namespace game3d_contract::known_gaps {
  enum class kind {
    expected_failure,
    pending
  };

  template<class Case>
  struct row {
    const char *id;
    const rule *contract;  // The oracle rule the add-on misses.
    kind status;
    bool (*matches)(const Case &);
    const char *owner_note;
  };

  // The state a copy may start from (checked by test_game3d_api_contract_cpu.cpp).
  namespace state {
    // An observed legacy COMMON on a depth-stencil allocation is refused as
    // missing evidence, unless an equal COMMON declaration is checked against it.
    constexpr bool common_on_depth_stencil(const state_case &value) {
      return depth_stencil(value.resource.flags) && value.resource.seen == evidence::transition &&
             value.resource.observed == D3D12_RESOURCE_STATE_COMMON && value.source != claim::sl_declared;
    }

    // A depth declaration of COMMON on a promotable texture is copied from
    // COMMON when the recording names nothing else.
    constexpr bool declared_common_on_promotable(const state_case &value) {
      const auto &source = value.resource;
      return !depth_stencil(source.flags) && value.role == consumer::depth && value.source == claim::sl_declared &&
             value.declared == D3D12_RESOURCE_STATE_COMMON &&
             (source.seen == evidence::none || (source.seen == evidence::transition && source.observed == D3D12_RESOURCE_STATE_COMMON));
    }

    // RESOLVE_DEST, legacy or enhanced, is never a copy basis.
    constexpr bool resolve_destination(const state_case &value) {
      const auto &source = value.resource;
      const bool declared = value.source == claim::sl_declared && value.declared == D3D12_RESOURCE_STATE_RESOLVE_DEST;
      if (source.seen == evidence::transition && source.observed == D3D12_RESOURCE_STATE_RESOLVE_DEST) {
        return value.source != claim::sl_declared || declared;
      }
      if (source.seen == evidence::enhanced && source.layout == D3D12_BARRIER_LAYOUT_RESOLVE_DEST && source.access != D3D12_BARRIER_ACCESS_NO_ACCESS) {
        return value.source != claim::sl_declared;
      }
      return source.seen == evidence::none && declared;
    }

    // The resource flags are not checked against the copy basis: an observed
    // state or layout, or a depth declaration, that the allocation cannot be
    // in is copied from. (A UI declaration is checked for render-target, UAV
    // and depth flags; a depth declaration that names another observed state
    // is refused as conflicting first.)
    constexpr bool basis_flags_unchecked(const state_case &value) {
      const auto &source = value.resource;
      const bool depth_declaration = value.role == consumer::depth && value.source == claim::sl_declared;
      if (source.seen == evidence::transition) {
        return texture_state(source.observed) && !state_allowed(source.observed, source.flags) &&
               !(depth_declaration && value.declared != source.observed);
      }
      if (source.seen == evidence::enhanced) {
        return direct_queue_layout(source.layout) && source.layout != D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE &&
               !layout_allowed(source.layout, source.flags) && source.access != D3D12_BARRIER_ACCESS_NO_ACCESS && !depth_declaration;
      }
      return source.seen == evidence::none && depth_declaration && texture_state(value.declared) &&
             !state_allowed(value.declared, source.flags);
    }
  }  // namespace state

  inline constexpr row<state_case> state_rows[] {
    {"D3", &rules::D3, kind::expected_failure, &state::common_on_depth_stencil, "Owner ruling R1 (decided): fix copy_state so COMMON on a non-simultaneous depth-stencil allocation is a proof, whether declared or observed. Update test_capture_state_policy.cpp's 'undeclared state took an observed COMMON' check with it."},
    {"D3b", &rules::D3b, kind::expected_failure, &state::declared_common_on_promotable, "A COMMON declaration on a colour or linear-depth texture may already be promoted; fix together with D3 (one COMMON rule keyed by depth-stencil allocation). Update the 'typed declared COMMON was not retained' check with it."},
    {"D2", &rules::D2, kind::expected_failure, &state::resolve_destination, "Fail-safe today (refusal is harmless), lowest priority: admit RESOLVE_DEST in copy_state::supported and layouts::admissible when a game is seen to need it."},
    {"X1", &rules::X1, kind::expected_failure, &state::basis_flags_unchecked, "Illegal game input only: check the allocation flags for every copy basis in copy_state::resolve (today only the at-call UI declaration fallback does), so an impossible state never reaches a barrier."},
  };

  // Raw Streamline buffer types at each surveyed release (checked by
  // test_game3d_api_contract_cpu.cpp).
  namespace buffer_types {
    // 2.2.1 already declares HiResDepth 48 and LinearDepth 49 (v2.2.1 sl.h),
    // but the contract table gives every release below 2.4 the 2.0 range
    // (0-37), so a 2.2.1-2.3.x game's display or linear depth is ignored.
    constexpr bool depth_kinds_before_2_4(const buffer_case &value) {
      return !(value.version < sl_version {2, 2, 1}) && value.version < sl_version {2, 4, 0} && (value.raw == 48 || value.raw == 49);
    }
  }  // namespace buffer_types

  inline constexpr row<buffer_case> buffer_rows[] {
    {"S17", &rules::S17, kind::expected_failure, &buffer_types::depth_kinds_before_2_4, "Few games pair 2.2.1-2.3.x with these tags: add a 2.2.1 tier (last type 52) to buffers::for_version, keep 37 below it, and update test_streamline_camera_probe.cpp's '2.2 contract accepted post-2.4 tags' check (3187-3189 at c67e9cf9) and the docs/reshade-sbs.md buffer-type table's 2.0-2.2 row with it."},
  };

  // Streamline depth tags (checked by test_game3d_api_contract_cpu.cpp's hook
  // layer). Its cells combine several dimensions, so one cell may match more
  // than one row (judge_each). Each expected-failure predicate names the
  // add-on's behaviour and keeps only cells where the oracle requires a
  // capture, so a refusal the contract asks for never matches it.
  namespace streamline {
    constexpr bool local(tag_entry entry) {
      return entry == tag_entry::local_array || entry == tag_entry::local_chain_not_last;
    }

    // A legal cell where the oracle permits nothing but a capture (an EITHER
    // cell may also refuse, so no gap can match it).
    constexpr bool capture_required(const sl_case &value) {
      const auto expected = expect(value);
      return legal(value) && (expected.kind == outcome_kind::capture || expected.kind == outcome_kind::continuous) && !either(expected);
    }

    // No barrier on the copy's recording names the source's state.
    constexpr bool unobserved(const resource_state &value) {
      return value.seen == evidence::none || value.seen == evidence::alias_named || value.seen == evidence::alias_wildcard;
    }

    // A same-frame evaluate copy is refused once any Present of the device
    // follows the tag: present() advances every source's present generation
    // (source_lifetime_current), so a generated Present of an earlier frame
    // ends lifecycle 1 and 2 alike. The hook layer counts the case's generated
    // Presents (it has no device).
    constexpr bool generated_present_before_copy(const sl_case &value) {
      return value.generated_presents != 0 && value.at == copy_point::same_frame_evaluate && !local(value.entry) && capture_required(value);
    }

    // The probe promotes lifecycle 0 to its tag call only; an OnlyValidNow
    // evaluate input fails match() as not-valid-until-present.
    constexpr bool local_only_valid_now(const sl_case &value) {
      return value.entry == tag_entry::local_array && value.life == lifecycle::only_valid_now && value.at == copy_point::same_frame_evaluate &&
             capture_required(value);
    }

    // A frame-generation tag call without a command list is refused as
    // boundary_without_commands, with no later copy.
    constexpr bool commandless_frame_generation_tag(const sl_case &value) {
      return value.at == copy_point::tag_call && value.life == lifecycle::valid_until_present && value.commands == tag_commands::none &&
             capture_required(value);
    }

    // A longer-lived tag is copied at its frame-generation tag call from its
    // declaration, which describes the state at its use, instead of later.
    constexpr bool undeferred_frame_generation_tag(const sl_case &value) {
      return value.at == copy_point::tag_call && value.life != lifecycle::only_valid_now && value.commands == tag_commands::game_list &&
             unobserved(value.depth) && capture_required(value) && expect(value).deferred;
    }

    // At that tag call a declaration that disagrees with the recorded state
    // is refused as conflicting (declared_must_match).
    constexpr bool disagreeing_frame_generation_tag(const sl_case &value) {
      return value.at == copy_point::tag_call && value.life != lifecycle::only_valid_now && value.commands == tag_commands::game_list &&
             value.depth.seen == evidence::transition && value.declared_state != UINT32_MAX && value.depth.observed != value.declared_state &&
             capture_required(value);
    }

    // Constants above v2 lose the camera, a ViewportHandle above v1 makes the
    // whole call invalid, and DLSSGOptions above v5 leave frame generation
    // unknown (decode caps, not the known prefix).
    constexpr bool newer_struct(const sl_case &value) {
      const bool options = value.frame_generation_options == struct_form::above_known && value.at == copy_point::tag_call &&
                           detail::frame_generation_scope(value);
      return (value.constants == struct_form::above_known || value.viewport_handle == struct_form::above_known || options) &&
             capture_required(value);
    }

    // A tag whose next pointer continues the input chain is read as carrying
    // an unknown extension.
    constexpr bool chained_local(const sl_case &value) {
      return value.entry == tag_entry::local_chain_not_last && capture_required(value);
    }

    constexpr bool zero_sided_extent(const sl_case &value) {
      return value.extent == extent_form::one_dimension_zero && capture_required(value);
    }

    // match() checks the extent against the declared Resource size.
    constexpr bool declared_size(const sl_case &value) {
      return value.extent == extent_form::declared_size_differs && capture_required(value);
    }

    // More than 32 tags in a call, or more than 32 evaluate inputs or chain
    // nodes (a local tag shares the evaluate's inputs with its viewport),
    // make the whole call invalid.
    constexpr bool beyond_tag_limit(const sl_case &value) {
      return (value.tags_in_call > 32 || (local(value.entry) && value.tags_in_call + 1 > 32)) && capture_required(value);
    }

    // Illegal input: a global or frame UntilEvaluate tag without a command
    // list is copied at its evaluate, on that evaluate's own list, when no
    // Present came between (S7 refuses it otherwise).
    constexpr bool commandless_until_evaluate(const sl_case &value) {
      return !legal(value) && value.commands == tag_commands::none && value.life == lifecycle::valid_until_evaluate && !local(value.entry) &&
             value.at == copy_point::same_frame_evaluate && value.resource == resource_form::native && !value.generated_presents;
    }

    // A multi-tag array from a release newer than the survey whose tags
    // declare ResourceTag v1, which S24 refuses.
    constexpr bool unsurveyed_tag_array(const sl_case &value) {
      return expect(value).by == &rules::S24 && value.version.major == 2 && value.tags_in_call > 1 && value.tags_in_call <= 32 &&
             value.tag_header == struct_form::newest_known;
    }

    // The zero-header Resource layout (a typed ResourceTag whose Resource
    // base was zero-initialised) with a nonzero declaration and no recorded
    // state, which S14 refuses.
    constexpr bool zero_header_declaration(const sl_case &value) {
      return legal(value) && expect(value).by == &rules::S14 && value.resource == resource_form::zero_header && value.declared_state != 0 &&
             value.declared_state != UINT32_MAX && unobserved(value.depth);
    }
  }  // namespace streamline

  inline constexpr row<sl_case> sl_rows[] {
    {"S6", &rules::S6, kind::expected_failure, &streamline::commandless_frame_generation_tag, "Owner ruling R2 (decided): keep a commandless ValidUntilPresent tag pending and copy it at the same frame's evaluate, else on the Present queue before that frame's Present. Today source_rejection refuses it as boundary_without_commands (since a2563fa2 DLSS-G's own NGX evaluation supplies that frame's FG depth where it is seen, N4, and the refusal then withdraws nothing), and game3d_ui_mask.cpp refuses the UI equivalent (U2). Fix together with S7 and S11b; that commit inverts the pinned refusal, test_game3d_native_provider_runtime.cpp's 'native SLFG refusal' check (1795-1811 at c67e9cf9, 1925-1951 at a2563fa2), and revisits its boundary_without_commands log checks (2010 silent scope and 2162 live scope at a2563fa2)."},
    {"S7", &rules::S7, kind::expected_failure, &streamline::generated_present_before_copy, "End a tag's lifetime at its own frame's Present (its frame token, or a present ordinal recorded at the tag), not at any Present of the device: streamline_depth_capture.cpp present() advances every source's present generation and source_lifetime_current then refuses the evaluate copy. Multi frame generation puts generated Presents of earlier frames there. 24bf1f4b and a2563fa2 hold a captured copy across generated Presents; neither lets this copy be made. Fix together with S6, whose deferred copy meets the same check."},
    {"S11b", &rules::S11b, kind::expected_failure, &streamline::undeferred_frame_generation_tag, "Owner ruling R2 (decided): a ValidUntilPresent tag at its FG tag call with no observed state is copied on the Present queue, where its declaration is exact (an UntilEvaluate one at its same-frame evaluate), not at the call. Same deferral as S6."},
    {"S11", &rules::S11, kind::expected_failure, &streamline::disagreeing_frame_generation_tag, "At an FG tag call of a longer-lived tag, take the state observed on the recording: its declaration describes the use point (SLT:357). Today copy_state's declared_must_match refuses it as conflicting (streamline_depth_capture.cpp record_impl, source_contract policy)."},
    {"S4", &rules::S4, kind::expected_failure, &streamline::local_only_valid_now, "A local tag is current at its own evaluate (SLA:258-262): in tag_rejection (streamline_camera_probe.cpp) promote an OnlyValidNow evaluate input as the tag call's is, and copy it at that evaluate when it has a command list. Fix with S15 (both in read_local_inputs/tag_rejection)."},
    {"S15", &rules::S15, kind::expected_failure, &streamline::chained_local, "In read_local_inputs, decode a chained tag with its next pointer cleared (as the viewport already is), treating a PrecisionInfo node as the preceding tag's; fix with S4. Chains of more than 32 nodes also fail S18."},
    {"S13", &rules::S13, kind::expected_failure, &streamline::newer_struct, "Read the known prefix of any version >= 1 and keep refusing a chained next: decode_v2_constants (max 2) and decode_v2_viewport (max 1) in streamline_camera_data.cpp, and hook_fg_options (version <= 5). Update the pinned checks with it: test_streamline_camera_data.cpp:267-268 and 279-280 ('unknown version interpreted', 'unknown viewport version accepted') and test_streamline_camera_probe.cpp's 'unknown FG options version retained a known enabled state' (3891 at c67e9cf9)."},
    {"S16", &rules::S16, kind::expected_failure, &streamline::zero_sided_extent, "Use Streamline's whole-resource rule !(width && height) (sl_consts.h:150) in match() (streamline_camera_data.cpp) and source_region (streamline_depth_capture.cpp)."},
    {"S16d", &rules::S16d, kind::expected_failure, &streamline::declared_size, "Drop the declared Resource size check from match(); the capture already bounds the extent by the allocation (source_region)."},
    {"S18", &rules::S18, kind::expected_failure, &streamline::beyond_tag_limit, "Read the type and lifecycle of each element of a longer tag array or input list in chunks with no fixed cap (read_tags, read_local_inputs; tag_limit and chain_limit are 32); at minimum keep the depth slot of a call with more tags."},
    {"S2", &rules::S2, kind::expected_failure, &streamline::commandless_until_evaluate, "Illegal input and harmless, lowest priority: the copy happens inside the tag's lifetime on the evaluate's own list. Refuse it, or let the oracle accept it, when the commandless-tag work (S6) settles what a commandless tag may do."},
    {"S24-pending", &rules::S24, kind::pending, &streamline::unsurveyed_tag_array, "Owner ruling needed: the add-on reads a multi-tag array from a release newer than the survey when its ResourceTag declares version 1, whose stride is known (sl_struct.h appends members with a version bump; S14 still refuses a later version in an array). The design's S24 refuses any such array. Either relax S24 to major 3 and above, or make read_tags refuse these arrays."},
    {"S14-pending", &rules::S14, kind::pending, &streamline::zero_header_declaration, "Owner ruling needed: the zero-header Resource layout (some 2.x integrations zero-initialise Resource's base) is accepted with its nonzero declared state as proof (streamline_camera_probe.cpp append_tag; pinned by test_streamline_camera_probe.cpp's 'nonzero compatibility resource state was discarded'). S14 accepts only a state recorded on the copy's list for a Resource that is not an SDK structure. Either rule the declaration usable and relax S14, or require an observed state as for its declared COMMON."},
  };

  // NGX evaluations, DLSS-G's own (N4) included, are checked by the hook
  // layer and meet the oracle: no rows yet.

  // One family's verdicts against its rows.
  enum class verdict {
    meets,  // No row; the cell satisfies the oracle.
    violates,  // No row; the cell fails the oracle.
    known_failure,  // An expected_failure row; the cell fails as recorded.
    gap_closed,  // An expected_failure row; the cell now passes.
    misattributed,  // An expected_failure row whose rule the oracle did not apply.
    pending,  // A pending row; not enforced.
    overlapping  // More than one row matches.
  };

  template<class Case, unsigned N>
  constexpr int find(const row<Case> (&rows)[N], const Case &value) {
    int found = -1;
    for (unsigned i = 0; i != N; ++i) {
      if (rows[i].matches(value)) {
        if (found >= 0) {
          return -2;
        }
        found = int(i);
      }
    }
    return found;
  }

  template<class Case, unsigned N>
  constexpr verdict judge(const row<Case> (&rows)[N], const Case &value, const outcome &expected, bool satisfied, int *index = nullptr) {
    const auto found = find(rows, value);
    if (index) {
      *index = found;
    }
    if (found == -2) {
      return verdict::overlapping;
    }
    if (found < 0) {
      return satisfied ? verdict::meets : verdict::violates;
    }
    const auto &gap = rows[found];
    if (gap.status == kind::pending) {
      return verdict::pending;
    }
    if (satisfied) {
      return verdict::gap_closed;
    }
    return expected.by == gap.contract || expected.basis_by == gap.contract ? verdict::known_failure : verdict::misattributed;
  }

  // A family whose cells combine several dimensions (the hook layer): matched
  // receives every row that matches. Any pending row leaves the cell
  // unenforced; otherwise a failing cell is a known failure of each matching
  // row and a passing one closes them all. The caller also requires every
  // expected-failure row to be the only row of some failing cell, so the fix
  // that closes a gap turns its isolated cells into closed gaps.
  template<class Case, unsigned N>
  constexpr verdict judge_each(const row<Case> (&rows)[N], const Case &value, bool satisfied, bool (&matched)[N]) {
    bool any {}, pending {};
    for (unsigned i = 0; i != N; ++i) {
      matched[i] = rows[i].matches(value);
      any |= matched[i];
      pending |= matched[i] && rows[i].status == kind::pending;
    }
    if (!any) {
      return satisfied ? verdict::meets : verdict::violates;
    }
    if (pending) {
      return verdict::pending;
    }
    return satisfied ? verdict::gap_closed : verdict::known_failure;
  }
}  // namespace game3d_contract::known_gaps
