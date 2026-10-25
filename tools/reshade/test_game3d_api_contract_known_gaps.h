// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Where the add-on does not meet the API contract oracle yet. Each row names
// the oracle rule it misses, a predicate over the matrix dimensions that
// selects exactly the cells concerned, and an owner note.
//
// expected_failure: every matching cell must fail the oracle, under the rule
// the row names and (in a hook family) the way the row records. A matching
// cell that passes fails the run, so the fix that closes a gap also removes
// its row (and updates any test that pinned the old behaviour). A row that
// matches no failing cell fails the run as stale (in a family judged with
// judge_each, no failing cell that it alone matches; only a complete run
// shows that).
// pending: the cell's verdict is not enforced (EITHER) until owner work
// lands or an owner ruling decides it; the global invariants still apply. Any
// family may hold pending rows.
//
// Where a hook layer repeats a capture-owner check it cannot run without a
// device (test_game3d_api_contract_cpu.cpp, admission), the row's verdict
// comes from that repeat: the fix must change the repeat in the same commit,
// or the gap cannot close there.
//
// Rows describe the add-on's current behaviour in the oracle's dimension
// vocabulary; they never feed back into expect(). Like the oracle, this
// header includes nothing from the add-on.
//
// Stage 3 of the general pipeline plan fixes the gaps one commit each, in
// this order: 1 D10, 2 S6 + S7, 3 D6, 4 S11, 5 D3, 6 S13, 7 D11c, 8 S4/S15,
// 9 S22, 10 S18, 11 S17, 12 S16, 13 D4c. Each expected-failure note names
// the fix that closes its row ("Stage 3 fix n"), or says the row lies
// outside that order.
#include "test_game3d_api_contract_dimensions.h"
#include "test_game3d_api_contract_oracle.h"

#include <cstdint>
#include <d3d12.h>

namespace game3d_contract::known_gaps {
  enum class kind {
    expected_failure,
    pending
  };

  // What the add-on did in a hook cell: its result at the copy point, and
  // whether it agreed with that layer's cross-run checks (the newest-known
  // twin, S13; one logical source across render sizes, R3).
  struct behaviour {
    result actual;
    bool consistent {true};
  };

  template<class Case>
  struct row {
    const char *id;
    const rule *contract;  // The oracle rule the add-on misses.
    kind status;
    bool (*matches)(const Case &);
    const char *owner_note;
    // Hook families: what the add-on does in the row's cells. A failing cell
    // that no matching row records fails the run (misattributed).
    bool (*records)(const behaviour &) = nullptr;
  };

  // The ways a hook row records the add-on's failure.
  namespace behaviours {
    constexpr bool no_copy(const behaviour &value) {
      return value.actual.kind != outcome_kind::capture;
    }

    constexpr bool copy(const behaviour &value) {
      return value.actual.kind == outcome_kind::capture;
    }

    constexpr bool copy_at_call(const behaviour &value) {
      return value.actual.kind == outcome_kind::capture && !value.actual.deferred;
    }

    // No copy, or one unlike the newest-known version's.
    constexpr bool unlike_newest(const behaviour &value) {
      return value.actual.kind != outcome_kind::capture || !value.consistent;
    }
  }  // namespace behaviours

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
        return value.source != claim::sl_declared || declared;
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

    // copy_state::supported admits the buffer read bits (VERTEX_AND_CONSTANT_BUFFER,
    // INDEX_BUFFER, INDIRECT_ARGUMENT) with read-only texture bits, which the
    // D3D12 runtime refuses on a texture other than GENERIC_READ itself: a
    // completed transition into one, or a declaration of one with nothing
    // observed, becomes the copy's basis.
    constexpr bool buffer_read_bits(const state_case &value) {
      constexpr std::uint32_t buffer_bits = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_INDEX_BUFFER |
                                            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
      const auto buffer_state = [](std::uint32_t state) {
        return (state & buffer_bits) && state != D3D12_RESOURCE_STATE_GENERIC_READ &&
               !(state & ~std::uint32_t(D3D12_RESOURCE_STATE_GENERIC_READ | texture_read_states));
      };
      const auto &source = value.resource;
      if (source.seen == evidence::transition) {
        return buffer_state(source.observed) && !(value.source == claim::sl_declared && value.declared != source.observed);
      }
      return source.seen == evidence::none && value.source == claim::sl_declared && buffer_state(value.declared);
    }

    // An END_ONLY whose StateAfter differs from its BEGIN_ONLY's is taken as
    // a completed transition into the END's state (the recorder keeps no
    // begin state), so it is copied from unless a declaration disagrees.
    constexpr bool mismatched_split_end(const state_case &value) {
      const auto &source = value.resource;
      return source.seen == evidence::end_mismatched && (value.source != claim::sl_declared || value.declared == source.observed);
    }
  }  // namespace state

  inline constexpr row<state_case> state_rows[] {
    {"D3", &rules::D3, kind::expected_failure, &state::common_on_depth_stencil, "Stage 3 fix 5 (D3). Owner ruling R1 (decided): fix copy_state so COMMON on a non-simultaneous depth-stencil allocation is a proof, whether declared or observed. Update test_capture_state_policy.cpp's 'undeclared state took an observed COMMON' check with it."},
    {"D3b", &rules::D3b, kind::expected_failure, &state::declared_common_on_promotable, "Stage 3 fix 5 (D3). A COMMON declaration on a colour or linear-depth texture may already be promoted; fix together with D3 (one COMMON rule keyed by depth-stencil allocation). Update the 'typed declared COMMON was not retained' check with it."},
    {"D2", &rules::D2, kind::expected_failure, &state::resolve_destination, "Outside the Stage 3 order. Fail-safe today (refusal is harmless), lowest priority: admit RESOLVE_DEST in copy_state::supported and layouts::admissible when a game is seen to need it."},
    {"X1", &rules::X1, kind::expected_failure, &state::basis_flags_unchecked, "Outside the Stage 3 order. Illegal game input only: check the allocation flags for every copy basis in copy_state::resolve (today only the at-call UI declaration fallback does), so an impossible state never reaches a barrier."},
    {"X2", &rules::X2, kind::expected_failure, &state::buffer_read_bits, "Outside the Stage 3 order. Illegal game input, but not harmless: the copy's barrier from such a state makes the game's own ID3D12GraphicsCommandList::Close fail (E_INVALIDARG), as the device layer saw for GENERIC_READ | DEPTH_READ. Admit GENERIC_READ itself and no other state with a buffer read bit in copy_state::supported, and update test_capture_state_policy.cpp's supported() checks with it."},
    {"D4c", &rules::D4c, kind::expected_failure, &state::mismatched_split_end, "Stage 3 fix 13 (D4c). Illegal game input: streamline_depth_capture.cpp's barrier recorder ends a split at any END_ONLY and takes its StateAfter, never comparing it with the BEGIN_ONLY's. Keep the begin's StateAfter while the split is pending and block the source when its END_ONLY names another state. The device layer cannot run this row (the debug layer refuses the barrier), so this table is its only check."},
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
    {"S17", &rules::S17, kind::expected_failure, &buffer_types::depth_kinds_before_2_4, "Stage 3 fix 11 (S17). Few games pair 2.2.1-2.3.x with these tags: add a 2.2.1 tier (last type 52) to buffers::for_version, keep 37 below it, and update test_streamline_camera_probe.cpp's '2.2 contract accepted post-2.4 tags' check (3187-3189 at c67e9cf9) and the docs/reshade-sbs.md buffer-type table's 2.0-2.2 row with it."},
  };

  // Streamline depth tags (checked by test_game3d_api_contract_cpu.cpp's hook
  // layer). Its cells combine several dimensions, so one cell may match more
  // than one row (judge_each). Each expected-failure predicate names the
  // add-on's behaviour and keeps only cells where the oracle requires a
  // capture (or, for an add-on that copies, a refusal), so an outcome the
  // contract leaves open never matches it.
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

    // The probe promotes lifecycle 0 to its tag call only; an OnlyValidNow
    // evaluate input fails match() as not-valid-until-present.
    constexpr bool local_only_valid_now(const sl_case &value) {
      return value.entry == tag_entry::local_array && value.life == lifecycle::only_valid_now && value.at == copy_point::same_frame_evaluate &&
             capture_required(value);
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
    {"S11", &rules::S11, kind::expected_failure, &streamline::disagreeing_frame_generation_tag, "Stage 3 fix 4 (S11). At an FG tag call of a longer-lived tag, take the state observed on the recording: its declaration describes the use point (SLT:357). Today copy_state's declared_must_match refuses it as conflicting (streamline_depth_capture.cpp record_impl, source_contract policy).", &behaviours::no_copy},
    {"S4", &rules::S4, kind::expected_failure, &streamline::local_only_valid_now, "Stage 3 fix 8 (S4/S15). A local tag is current at its own evaluate (SLA:258-262): in tag_rejection (streamline_camera_probe.cpp) promote an OnlyValidNow evaluate input as the tag call's is, and copy it at that evaluate when it has a command list. Fix with S15 (both in read_local_inputs/tag_rejection).", &behaviours::no_copy},
    {"S15", &rules::S15, kind::expected_failure, &streamline::chained_local, "Stage 3 fix 8 (S4/S15). In read_local_inputs, decode a chained tag with its next pointer cleared (as the viewport already is), treating a PrecisionInfo node as the preceding tag's; fix with S4. Chains of more than 32 nodes also fail S18.", &behaviours::no_copy},
    {"S13", &rules::S13, kind::expected_failure, &streamline::newer_struct, "Stage 3 fix 6 (S13). Read the known prefix of any version >= 1 and keep refusing a chained next: decode_v2_constants (max 2) and decode_v2_viewport (max 1) in streamline_camera_data.cpp, and hook_fg_options (version <= 5). Update the pinned checks with it: test_streamline_camera_data.cpp:267-268 and 279-280 ('unknown version interpreted', 'unknown viewport version accepted') and test_streamline_camera_probe.cpp's 'unknown FG options version retained a known enabled state' (3891 at c67e9cf9).", &behaviours::unlike_newest},
    {"S16", &rules::S16, kind::expected_failure, &streamline::zero_sided_extent, "Stage 3 fix 12 (S16). Use Streamline's whole-resource rule !(width && height) (sl_consts.h:150) in match() (streamline_camera_data.cpp) and source_region (streamline_depth_capture.cpp).", &behaviours::no_copy},
    {"S16d", &rules::S16d, kind::expected_failure, &streamline::declared_size, "Stage 3 fix 12 (S16). Drop the declared Resource size check from match(); the capture already bounds the extent by the allocation (source_region).", &behaviours::no_copy},
    {"S18", &rules::S18, kind::expected_failure, &streamline::beyond_tag_limit, "Stage 3 fix 10 (S18). Read the type and lifecycle of each element of a longer tag array or input list in chunks with no fixed cap (read_tags, read_local_inputs; tag_limit and chain_limit are 32); at minimum keep the depth slot of a call with more tags.", &behaviours::no_copy},
    {"S24-pending", &rules::S24, kind::pending, &streamline::unsurveyed_tag_array, "Owner ruling needed: the add-on reads a multi-tag array from a release newer than the survey when its ResourceTag declares version 1, whose stride is known (sl_struct.h appends members with a version bump; S14 still refuses a later version in an array). The design's S24 refuses any such array. Either relax S24 to major 3 and above, or make read_tags refuse these arrays."},
    {"S14-pending", &rules::S14, kind::pending, &streamline::zero_header_declaration, "Owner ruling needed: the zero-header Resource layout (some 2.x integrations zero-initialise Resource's base) is accepted with its nonzero declared state as proof (streamline_camera_probe.cpp append_tag; pinned by test_streamline_camera_probe.cpp's 'nonzero compatibility resource state was discarded'). S14 accepts only a state recorded on the copy's list for a Resource that is not an SDK structure. Either rule the declaration usable and relax S14, or require an observed state as for its declared COMMON."},
  };

  // A Streamline frame judged at its Present by the runtime layer
  // (test_game3d_api_contract_runtime.cpp): the case, and the case at the
  // copy point that decides what the Present shows (the frame's tag call
  // where its outcome requires a capture, else the case's own point).
  struct present_case {
    sl_case value, decided;
  };

  // The lifecycle and command-list rule (S1-S7) of a copy point of the
  // case's frame, as if the checks the oracle makes before any copy point
  // (the extent, the SDK results, the untag) passed.
  constexpr const rule *point_rule(sl_case value, copy_point at) {
    value.at = at;
    value.extent = extent_form::null;
    value.tag_result = value.evaluate_result = sl_result::ok;
    value.untag = untag_pattern::none;
    if (at == copy_point::tag_call) {
      value.generated_presents = 0;
    }
    return expect(value).point_by;
  }

  // Whether the oracle applied by at the deciding copy point, or by is the
  // rule of a copy point the frame reaches (its tag call, or its own point):
  // a recorded gap there refuses at that point's gate, before the check that
  // decided the oracle's outcome, so the cell shows that gap's refusal class.
  constexpr bool applies(const present_case &value, const rule &by) {
    const auto &tag = value.value;
    const bool local = tag.entry == tag_entry::local_array || tag.entry == tag_entry::local_chain_not_last;
    return game3d_contract::applies(value.decided, by) || (!local && point_rule(tag, copy_point::tag_call) == &by) || point_rule(tag, tag.at) == &by;
  }

  namespace behaviours {
    // Refused with class why (the runtime layer reads it from the add-on).
    template<refusal why>
    constexpr bool refused_as(const behaviour &value) {
      return value.actual.kind == outcome_kind::refuse && value.actual.why == why;
    }
  }  // namespace behaviours

  // The runtime layer's cells (judged at Present).
  namespace present {
    // An OnlyValidNow evaluate input (S4) of a release the add-on hooks that
    // the oracle refuses (or may refuse) for another class is refused first
    // for its lifecycle.
    constexpr bool local_only_valid_now_refused(const present_case &value) {
      const auto &cell = value.decided;
      const auto expected = expect(cell);
      return legal(cell) && cell.version.major == 2 && cell.entry == tag_entry::local_array && cell.life == lifecycle::only_valid_now &&
             cell.at == copy_point::same_frame_evaluate && permits(expected, outcome_kind::refuse) && expected.why != refusal::lifetime;
    }

    constexpr bool local_only_valid_now(const present_case &value) {
      return streamline::local_only_valid_now(value.decided);
    }

    // A refusal the runtime layer requires the add-on to record: the oracle
    // refuses alone, for a class a reached copy point decides, and no SDK
    // call of the case failed (S21b discards that copy unread).
    constexpr bool recorded_refusal(const sl_case &cell) {
      const auto expected = expect(cell);
      return legal(cell) && cell.version.major == 2 && expected.kind == outcome_kind::refuse && !either(expected) && expected.why != refusal::fg_scope &&
             expected.why != refusal::no_copy_point && cell.tag_result != sl_result::error && cell.evaluate_result != sl_result::error;
    }

    // A chained input that is not last (S15) is not read: a cell the oracle
    // refuses for its state or extent records only missing-depth (absent).
    constexpr bool chained_local_refused(const present_case &value) {
      return value.decided.entry == tag_entry::local_chain_not_last && recorded_refusal(value.decided);
    }

    // An OnlyValidNow tag outside frame-generation scope that its call
    // copies (S1), followed by the same frame's evaluate, which refuses the
    // tag as not valid until present (S3): that Present shows no depth.
    constexpr bool only_valid_now_then_evaluate(const present_case &value) {
      return value.value.at == copy_point::same_frame_evaluate && !streamline::local(value.value.entry) && value.decided.at == copy_point::tag_call &&
             value.decided.life == lifecycle::only_valid_now && !detail::frame_generation_scope(value.value) && streamline::capture_required(value.decided);
    }

    constexpr bool chained_local(const present_case &value) {
      return streamline::chained_local(value.decided);
    }

    constexpr bool zero_sided_extent(const present_case &value) {
      return streamline::zero_sided_extent(value.decided);
    }

    // An extent the add-on calls invalid on a cell whose tag match() reads
    // (at its tag call or same-frame evaluate, a chained input aside: S15
    // refuses that first) and the oracle refuses, or may refuse, for its
    // state, or after the frame's Present for its lifetime (the tag call
    // reads it first, outside frame generation too), is refused first as an
    // invalid extent.
    constexpr bool extent_refused(const present_case &value, extent_form form) {
      const auto &cell = value.decided;
      const auto expected = expect(cell);
      const bool after_present = cell.at == copy_point::after_present && !streamline::local(cell.entry) && cell.untag == untag_pattern::none &&
                                 cell.tag_result != sl_result::error && expected.why == refusal::lifetime;
      return legal(cell) && cell.extent == form && cell.entry != tag_entry::local_chain_not_last && permits(expected, outcome_kind::refuse) &&
             ((cell.at != copy_point::after_present && expected.why == refusal::state_unprovable) || after_present);
    }

    // A zero-sided extent: the whole resource (S16).
    constexpr bool zero_sided_extent_refused(const present_case &value) {
      return extent_refused(value, extent_form::one_dimension_zero);
    }

    constexpr bool declared_size(const present_case &value) {
      return streamline::declared_size(value.decided);
    }

    // A declared Resource size other than the allocation's (S16d).
    constexpr bool declared_size_refused(const present_case &value) {
      return extent_refused(value, extent_form::declared_size_differs);
    }

    constexpr bool unsurveyed_tag_array(const present_case &value) {
      return streamline::unsurveyed_tag_array(value.decided);
    }
  }  // namespace present

  // Streamline cells run through the native provider fixture (judged with
  // judge_each). A row that shares an id with the hook table's names the
  // same gap; that row's note holds the fix.
  inline constexpr row<present_case> sl_runtime_rows[] {
    {"S1-evaluate", &rules::S1, kind::expected_failure, &present::only_valid_now_then_evaluate, "Outside the Stage 3 order, which predates this finding: it reworks the tag-to-evaluate path fix 2 (S6 + S7) changes, so it fits beside it. Found by the runtime layer: an OnlyValidNow global tag outside FG scope is copied at its call, but the same frame's evaluate refuses the tag (match(): not-valid-until-present, S3) and that frame's Present then shows no depth (mono), or only between generated Presents. S3 is right that the evaluate may not use the tag; it must not cost the frame the copy its tag call made (S1). Today the refused evaluate begins a new evaluation of the viewport with no copy (streamline_camera_probe.cpp nominate_depth_source, withdraw_invalid_attempt -> depth_capture::begin_evaluation), which supersedes the tag call's. Keep a tag-call copy when a later point of its frame refuses the same tag for its lifetime. The hook layer judges one copy point per cell, so only a Present shows this.", &behaviours::no_copy},
    {"S4-refusal", &rules::S4, kind::expected_failure, &present::local_only_valid_now_refused, "Stage 3 fix 8 (S4/S15). With the hook table's S4 row: an OnlyValidNow evaluate input that the oracle refuses, or may refuse, for another class (its state, its extent) is refused first as not-valid-until-present (lifetime). Promoting the input at its own evaluate closes this row.", &behaviours::refused_as<refusal::lifetime>},
    {"S4", &rules::S4, kind::expected_failure, &present::local_only_valid_now, "Stage 3 fix 8 (S4/S15). As the hook table's S4 row: the evaluate input is refused as not valid until present (lifetime).", &behaviours::refused_as<refusal::lifetime>},
    {"S15", &rules::S15, kind::expected_failure, &present::chained_local, "Stage 3 fix 8 (S4/S15). As the hook table's S15 row.", &behaviours::no_copy},
    {"S15-refusal", &rules::S15, kind::expected_failure, &present::chained_local_refused, "Stage 3 fix 8 (S4/S15). With S15: the chained input is not read, so a cell the oracle refuses for its state or extent records only missing-depth (absent), no contract class.", &behaviours::no_copy},
    {"S16", &rules::S16, kind::expected_failure, &present::zero_sided_extent, "Stage 3 fix 12 (S16). As the hook table's S16 row.", &behaviours::no_copy},
    {"S16-refusal", &rules::S16, kind::expected_failure, &present::zero_sided_extent_refused, "Stage 3 fix 12 (S16). With S16: a zero-sided extent on a cell the oracle refuses, or may refuse, for another class (its state) is refused as invalid-extent (malformed) before that is judged, so its refusal class differs. The S16 fix closes this row.", &behaviours::refused_as<refusal::malformed>},
    {"S16d", &rules::S16d, kind::expected_failure, &present::declared_size, "Stage 3 fix 12 (S16). As the hook table's S16d row.", &behaviours::no_copy},
    {"S16d-refusal", &rules::S16d, kind::expected_failure, &present::declared_size_refused, "Stage 3 fix 12 (S16). With S16d: a declared Resource size other than the allocation's, on a cell the oracle refuses or may refuse for another class (an enhanced layout's state), is refused as invalid-extent (malformed) by match() first. Dropping that check closes this row.", &behaviours::refused_as<refusal::malformed>},
    {"S24-pending", &rules::S24, kind::pending, &present::unsurveyed_tag_array, "As the hook table's S24-pending row: through 2.15.0's interposer the add-on reads and copies the array's depth tag."},
  };

  // NGX evaluations, DLSS-G's own (N4) included (checked by the hook layer).
  namespace ngx {
    // A Depth otherwise valid, with no state observed on the recording, of a
    // feature whose input state no cited guide states: the adapter nominates
    // every feature with the SR contract's NON_PIXEL_SHADER_RESOURCE.
    constexpr bool uncited_contract_state(const ngx_case &value, ngx_feature feature) {
      return value.feature == feature && value.depth_set == ngx_depth_set::d3d12_resource && value.lifetime != ngx_lifetime::nested_in_streamline &&
             value.depth.seen == evidence::none && !(value.depth.flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) && legal(value) &&
             expect(value).by == &rules::N1b;
    }

    constexpr bool ray_reconstruction(const ngx_case &value) {
      return uncited_contract_state(value, ngx_feature::ray_reconstruction);
    }

    constexpr bool unknown_feature(const ngx_case &value) {
      return uncited_contract_state(value, ngx_feature::unknown);
    }
  }  // namespace ngx

  inline constexpr row<ngx_case> ngx_rows[] {
    {"N1b", &rules::N1b, kind::expected_failure, &ngx::unknown_feature, "Outside the Stage 3 order. A feature no cited guide covers is copied from the SR contract state: in ngx_depth_source.cpp nominate features other than SR (and RR once N1 covers it) with state_proof::observed_nonzero instead of sdk_contract, so only a recorded state proves them. Fail-safe for no shipping game known today.", &behaviours::copy},
    {"N1b-pending", &rules::N1b, kind::pending, &ngx::ray_reconstruction, "Owner action needed: the DLSS-RR guide is not among the cited sources. If its resource-state section names NON_PIXEL_SHADER_RESOURCE as the SR guide does, cite it in N1 for feature 13 and drop this row; otherwise RR follows N1b like an unknown feature."},
  };

  // NGX evaluations judged at Present by the runtime layer
  // (test_game3d_api_contract_runtime.cpp, family ngx, judged with
  // judge_each). A row that shares an id with the hook table's names the same
  // gap; that row's note holds the fix.
  inline constexpr row<ngx_case> ngx_runtime_rows[] {
    {"N1b", &rules::N1b, kind::expected_failure, &ngx::unknown_feature, "Outside the Stage 3 order. As the hook table's N1b row: the Present shows the unknown feature's depth, copied from the SR contract state with nothing recorded on its list.", &behaviours::copy},
    {"N1b-pending", &rules::N1b, kind::pending, &ngx::ray_reconstruction, "As the hook table's N1b-pending row."},
  };

  namespace behaviours {
    // A UI input that UI detection was not offered.
    constexpr bool not_offered(const behaviour &value) {
      return value.actual.kind != outcome_kind::offered;
    }
  }  // namespace behaviours

  // Generic depth-stencil boundaries judged at Present by the runtime layer
  // (family generic). The add-on meets every one; C++ has no empty array, so
  // one pending row that matches no cell stands in for the table.
  namespace generic {
    constexpr bool no_cell(const generic_case &) {
      return false;
    }
  }  // namespace generic

  inline constexpr row<generic_case> generic_runtime_rows[] {
    {"none", &rules::D11, kind::pending, &generic::no_cell, "No Generic runtime gap remains: Stage 3 fix 1 closed D10 and the render-pass clear (D11). This row matches no cell."},
  };

  // UI inputs judged by the runtime layer (family ui): whether UI detection
  // is offered the input (Dump3D's UI candidates).
  namespace ui {
    // A UI input passed to slEvaluateFeature (U3, R4).
    constexpr bool local_input(const ui_case &value) {
      const auto expected = expect(value);
      return legal(value) && value.clear == ui_clear_boundary::none && value.how == ui_capture::local && expected.kind == outcome_kind::offered && !either(expected);
    }
  }  // namespace ui

  inline constexpr row<ui_case> ui_runtime_rows[] {
    {"U3", &rules::U3, kind::expected_failure, &ui::local_input, "Stage 3 fix 8 (S4/S15). Found by the runtime layer; owner ruling R4 (decided): a local UI input is offered under the tag contract. streamline_camera_probe.cpp offers UI inputs only from its tag hooks (capture_ui_inputs after slSetTag or slSetTagForFrame); its slEvaluateFeature hook reads local inputs (read_local_inputs) for depth alone, so a UI input passed there is never offered. Offer it there, on the evaluate's command list, in the commit that changes read_local_inputs for S4 and S15.", &behaviours::not_offered},
  };

  // Copies recorded on real lists (checked by
  // test_game3d_api_contract_device.cpp, judged with judge_each). A cell whose
  // row records a copy is recorded but never executed: that copy starts from
  // a state nothing proved.
  namespace device {
    constexpr bool api(const device_case &value) {
      return value.point == device_point::api;
    }

    // The copy's recording completes a legacy barrier into state on
    // subresource 0 (alone or with every subresource).
    constexpr bool completes(const resource_state &source, std::uint32_t state) {
      return (source.seen == evidence::transition || source.seen == evidence::end_without_begin) && source.observed == state &&
             (source.subresource == subresource_barrier::zero || source.subresource == subresource_barrier::all);
    }

    // The recording names nothing about subresource 0.
    constexpr bool unnamed(const resource_state &source) {
      return source.subresource == subresource_barrier::other_mip_only || source.subresource == subresource_barrier::stencil_plane_only ||
             source.seen == evidence::none || source.seen == evidence::alias_named || source.seen == evidence::alias_wildcard;
    }

    // A legacy COMMON completed on the recording of a depth-stencil
    // allocation is refused as missing evidence (copy_state::resolve), unless
    // a declaration is checked against it.
    constexpr bool common_on_depth_stencil(const device_case &value) {
      const auto &state = value.state;
      return api(value) && depth_stencil(state.resource.flags) && completes(state.resource, D3D12_RESOURCE_STATE_COMMON) &&
             state.source != claim::sl_declared;
    }

    // A depth declaration of COMMON on a promotable texture is copied from
    // COMMON when the recording names nothing else.
    constexpr bool declared_common_on_promotable(const device_case &value) {
      const auto &state = value.state;
      return api(value) && state.list == list_type::direct && !depth_stencil(state.resource.flags) && state.role == consumer::depth &&
             state.source == claim::sl_declared && state.declared == D3D12_RESOURCE_STATE_COMMON &&
             (unnamed(state.resource) || completes(state.resource, D3D12_RESOURCE_STATE_COMMON));
    }

    // RESOLVE_DEST, legacy or enhanced, is never a copy basis.
    constexpr bool resolve_destination(const device_case &value) {
      const auto &state = value.state;
      const auto &source = state.resource;
      if (!api(value) || state.list != list_type::direct) {
        return false;
      }
      const bool declared = state.source == claim::sl_declared && state.declared == D3D12_RESOURCE_STATE_RESOLVE_DEST;
      if (completes(source, D3D12_RESOURCE_STATE_RESOLVE_DEST)) {
        return state.source != claim::sl_declared || declared;
      }
      if (source.seen == evidence::enhanced && source.layout == D3D12_BARRIER_LAYOUT_RESOLVE_DEST && source.access != D3D12_BARRIER_ACCESS_NO_ACCESS) {
        return state.source != claim::sl_declared || declared;
      }
      return unnamed(source) && declared;
    }

    // Unbinding a read-only DSV declares DEPTH_WRITE (depth_addon.cpp
    // on_bind_depth_stencil -> on_clear_depth_impl) and record_preserved copies
    // from it when the recording names nothing.
    constexpr bool read_only_unbind(const device_case &value) {
      return value.point == device_point::unbind_read_only_dsv && unnamed(value.state.resource);
    }

    // The UI layer's clear-boundary copy records legacy RENDER_TARGET
    // barriers (game3d_ui_layer.cpp record_into) whatever the target's layout
    // or simultaneous access.
    constexpr bool legacy_ui_clear(const device_case &value) {
      return value.point == device_point::ui_clear && value.clear != ui_clear_boundary::legacy_render_target;
    }

    // The capture owner records copies on DIRECT lists only
    // (streamline_depth_capture.cpp record_impl, record_stage::command_type):
    // every COMPUTE cell is refused before its state is judged.
    constexpr bool compute_list(const device_case &value, bool ngx) {
      const auto &state = value.state;
      const bool by_ngx = state.role == consumer::depth && (state.source == claim::ngx_contract || state.source == claim::none);
      return api(value) && state.list == list_type::compute && by_ngx == ngx;
    }

    constexpr bool streamline_compute_list(const device_case &value) {
      return compute_list(value, false);
    }

    constexpr bool ngx_compute_list(const device_case &value) {
      return compute_list(value, true);
    }
  }  // namespace device

  inline constexpr row<device_case> device_rows[] {
    {"D3", &rules::D3, kind::expected_failure, &device::common_on_depth_stencil, "Stage 3 fix 5 (D3). Owner ruling R1 (decided): as the state table's D3 row; on the device a COMMON transition on a depth-stencil allocation, then the copy, is legal D3D12.", &behaviours::no_copy},
    {"D3b", &rules::D3b, kind::expected_failure, &device::declared_common_on_promotable, "Stage 3 fix 5 (D3). As the state table's D3b row: fix with D3 (one COMMON rule keyed by depth-stencil allocation).", &behaviours::copy},
    {"D2", &rules::D2, kind::expected_failure, &device::resolve_destination, "Outside the Stage 3 order. As the state table's D2 row: fail-safe, lowest priority.", &behaviours::no_copy},
    {"D11c", &rules::D11c, kind::expected_failure, &device::read_only_unbind, "Stage 3 fix 7 (D11c). depth_addon.cpp declares DEPTH_WRITE when a DSV is unbound (on_bind_depth_stencil -> on_clear_depth_impl). Read the bound DSV's read-only flags: a writable DSV keeps DEPTH_WRITE; a read-only one with no observation is refused (D10's per-resource proven state may supply it).", &behaviours::copy},
    {"D12", &rules::D12, kind::expected_failure, &device::legacy_ui_clear, "Stage 3 fix 3 (D6). game3d_ui_layer.cpp record_into records legacy RENDER_TARGET -> COPY_SOURCE barriers at every full-frame clear boundary: illegal on a target an enhanced barrier put in the RENDER_TARGET layout, and wrong for a simultaneous-access target still in COMMON. Refuse both (or copy an enhanced target with enhanced barriers) with the device-level layout latch D6 adds.", &behaviours::copy},
    {"S22", &rules::S22, kind::expected_failure, &device::streamline_compute_list, "Stage 3 fix 9 (S22). Allow COMPUTE in record_impl and the consumer (streamline_depth_capture.cpp command_type, consume_owned) when the starting state is one a compute list may use (NON_PIXEL_SHADER_RESOURCE, UNORDERED_ACCESS, COPY_SOURCE, COPY_DEST, COMMON on depth-stencil, or a COMPUTE_QUEUE_* layout), with a compute-legal resting state for the snapshot storage. Until then every compute cell is refused as no_copy_point, also where the oracle refuses for its state.", &behaviours::no_copy},
    {"N7", &rules::N7, kind::expected_failure, &device::ngx_compute_list, "Stage 3 fix 9 (S22). As S22 for NGX evaluations (dlssg-guide.txt:2321-2324: a non-copy queue); the same record_impl change fixes both.", &behaviours::no_copy},
  };

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

  // A family whose cells combine several dimensions (the hook layers):
  // matched receives every row that matches. Any pending row leaves the cell
  // unenforced, and a passing cell closes every matching row. A failing cell
  // is a known failure only when the oracle applied each matching row's rule
  // to it (applies: the deciding, basis, copy point or attributed rule) and
  // some matching row records what the add-on did there; otherwise the gap
  // changed shape and the cell is misattributed. The caller also requires
  // every expected-failure row to be the only row of some failing cell, so
  // the fix that closes a gap turns its isolated cells into closed gaps.
  template<class Case, unsigned N>
  constexpr verdict judge_each(const row<Case> (&rows)[N], const Case &value, bool satisfied, const behaviour &actual, bool (&matched)[N]) {
    bool any {}, pending {}, applied {true}, recorded {};
    for (unsigned i = 0; i != N; ++i) {
      matched[i] = rows[i].matches(value);
      if (!matched[i]) {
        continue;
      }
      any = true;
      pending |= rows[i].status == kind::pending;
      applied = applied && applies(value, *rows[i].contract);
      recorded = recorded || !rows[i].records || rows[i].records(actual);
    }
    if (!any) {
      return satisfied ? verdict::meets : verdict::violates;
    }
    if (pending) {
      return verdict::pending;
    }
    if (satisfied) {
      return verdict::gap_closed;
    }
    return applied && recorded ? verdict::known_failure : verdict::misattributed;
  }
}  // namespace game3d_contract::known_gaps
