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
  }  // namespace state

  inline constexpr row<state_case> state_rows[] {
    {"D3", &rules::D3, kind::expected_failure, &state::common_on_depth_stencil, "Owner ruling R1 (decided): fix copy_state so COMMON on a non-simultaneous depth-stencil allocation is a proof, whether declared or observed. Update test_capture_state_policy.cpp's 'undeclared state took an observed COMMON' check with it."},
    {"D3b", &rules::D3b, kind::expected_failure, &state::declared_common_on_promotable, "A COMMON declaration on a colour or linear-depth texture may already be promoted; fix together with D3 (one COMMON rule keyed by depth-stencil allocation). Update the 'typed declared COMMON was not retained' check with it."},
    {"D2", &rules::D2, kind::expected_failure, &state::resolve_destination, "Fail-safe today (refusal is harmless), lowest priority: admit RESOLVE_DEST in copy_state::supported and layouts::admissible when a game is seen to need it."},
    {"X1", &rules::X1, kind::expected_failure, &state::basis_flags_unchecked, "Illegal game input only: check the allocation flags for every copy basis in copy_state::resolve (today only the at-call UI declaration fallback does), so an impossible state never reaches a barrier."},
    {"X2", &rules::X2, kind::expected_failure, &state::buffer_read_bits, "Illegal game input, but not harmless: the copy's barrier from such a state makes the game's own ID3D12GraphicsCommandList::Close fail (E_INVALIDARG), as the device layer saw for GENERIC_READ | DEPTH_READ. Admit GENERIC_READ itself and no other state with a buffer read bit in copy_state::supported, and update test_capture_state_policy.cpp's supported() checks with it."},
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

    // A same-frame evaluate copy is refused once any Present of the device
    // follows the tag: present() advances every source's present generation
    // (source_lifetime_current), so a generated Present of an earlier frame
    // ends lifecycle 1 and 2 alike. The hook layer has no device: its
    // admission repeats source_lifetime_current over the case's generated
    // Presents, so this verdict comes from that repeat.
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

    // A ValidUntilPresent tag is copied at its frame-generation tag call from
    // its declaration, which describes the state at its Present, where the
    // oracle permits only the copy deferred to the Present queue. An untag of
    // the same frame after the call cancels that copy (R7, S9 HOLD); the copy
    // at the call honours neither.
    constexpr bool undeferred_frame_generation_tag(const sl_case &value) {
      auto kept = value;
      kept.untag = untag_pattern::none;
      const auto expected = expect(kept);
      return legal(value) && value.at == copy_point::tag_call && value.life == lifecycle::valid_until_present &&
             value.commands == tag_commands::game_list && unobserved(value.depth) && detail::frame_generation_scope(value) &&
             (permits(expected, outcome_kind::capture) || permits(expected, outcome_kind::continuous)) && expected.deferred;
    }

    // A ValidUntilEvaluate tag is copied at its frame-generation tag call
    // from its declaration, which describes the state at its evaluate.
    constexpr bool until_evaluate_frame_generation_tag(const sl_case &value) {
      return legal(value) && value.at == copy_point::tag_call && value.life == lifecycle::valid_until_evaluate &&
             value.commands == tag_commands::game_list && unobserved(value.depth) && expect(value).by == &rules::S11c;
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
    {"S6", &rules::S6, kind::expected_failure, &streamline::commandless_frame_generation_tag, "Owner ruling R2 (decided): keep a commandless ValidUntilPresent tag pending and copy it at the same frame's evaluate, else on the Present queue before that frame's Present. Today source_rejection refuses it as boundary_without_commands (since a2563fa2 DLSS-G's own NGX evaluation supplies that frame's FG depth where it is seen, N4, and the refusal then withdraws nothing), and game3d_ui_mask.cpp refuses the UI equivalent (U2). Fix together with S7 and S11b; that commit inverts the pinned refusal, test_game3d_native_provider_runtime.cpp's 'native SLFG refusal' check (1795-1811 at c67e9cf9, 1925-1951 at a2563fa2), and revisits its boundary_without_commands log checks (2010 silent scope and 2162 live scope at a2563fa2). Owner ruling R7 (decided): a null untag of the same frame (frame token and viewport), or one with no frame token, cancels that deferred copy (S9 HOLD); a later frame's untag leaves it.", &behaviours::no_copy},
    {"S7", &rules::S7, kind::expected_failure, &streamline::generated_present_before_copy, "End a tag's lifetime at its own frame's Present (its frame token, or a present ordinal recorded at the tag), not at any Present of the device: streamline_depth_capture.cpp present() advances every source's present generation and source_lifetime_current then refuses the evaluate copy. Multi frame generation puts generated Presents of earlier frames there. 24bf1f4b and a2563fa2 hold a captured copy across generated Presents; neither lets this copy be made. The hook layer's verdict comes from its repeat of source_lifetime_current (test_game3d_api_contract_cpu.cpp admission), so the fix changes that repeat in the same commit. Fix together with S6, whose deferred copy meets the same check.", &behaviours::no_copy},
    {"S11b", &rules::S11b, kind::expected_failure, &streamline::undeferred_frame_generation_tag, "Owner ruling R2 (decided): a ValidUntilPresent tag at its FG tag call with no observed state is copied on the Present queue, where its declaration is exact, not at the call. Same deferral as S6.", &behaviours::copy_at_call},
    {"S11c", &rules::S11c, kind::expected_failure, &streamline::until_evaluate_frame_generation_tag, "An UntilEvaluate tag at its FG tag call with no observed state is copied there from a declaration that describes its evaluate (R2 defers lifecycle 1 only). Refuse that copy and let the same-frame evaluate copy it (S5); fix with S11b, which changes the same tag-call admission in source_rejection/record_impl.", &behaviours::copy_at_call},
    {"S11", &rules::S11, kind::expected_failure, &streamline::disagreeing_frame_generation_tag, "At an FG tag call of a longer-lived tag, take the state observed on the recording: its declaration describes the use point (SLT:357). Today copy_state's declared_must_match refuses it as conflicting (streamline_depth_capture.cpp record_impl, source_contract policy).", &behaviours::no_copy},
    {"S4", &rules::S4, kind::expected_failure, &streamline::local_only_valid_now, "A local tag is current at its own evaluate (SLA:258-262): in tag_rejection (streamline_camera_probe.cpp) promote an OnlyValidNow evaluate input as the tag call's is, and copy it at that evaluate when it has a command list. Fix with S15 (both in read_local_inputs/tag_rejection).", &behaviours::no_copy},
    {"S15", &rules::S15, kind::expected_failure, &streamline::chained_local, "In read_local_inputs, decode a chained tag with its next pointer cleared (as the viewport already is), treating a PrecisionInfo node as the preceding tag's; fix with S4. Chains of more than 32 nodes also fail S18.", &behaviours::no_copy},
    {"S13", &rules::S13, kind::expected_failure, &streamline::newer_struct, "Read the known prefix of any version >= 1 and keep refusing a chained next: decode_v2_constants (max 2) and decode_v2_viewport (max 1) in streamline_camera_data.cpp, and hook_fg_options (version <= 5). Update the pinned checks with it: test_streamline_camera_data.cpp:267-268 and 279-280 ('unknown version interpreted', 'unknown viewport version accepted') and test_streamline_camera_probe.cpp's 'unknown FG options version retained a known enabled state' (3891 at c67e9cf9).", &behaviours::unlike_newest},
    {"S16", &rules::S16, kind::expected_failure, &streamline::zero_sided_extent, "Use Streamline's whole-resource rule !(width && height) (sl_consts.h:150) in match() (streamline_camera_data.cpp) and source_region (streamline_depth_capture.cpp).", &behaviours::no_copy},
    {"S16d", &rules::S16d, kind::expected_failure, &streamline::declared_size, "Drop the declared Resource size check from match(); the capture already bounds the extent by the allocation (source_region).", &behaviours::no_copy},
    {"S18", &rules::S18, kind::expected_failure, &streamline::beyond_tag_limit, "Read the type and lifecycle of each element of a longer tag array or input list in chunks with no fixed cap (read_tags, read_local_inputs; tag_limit and chain_limit are 32); at minimum keep the depth slot of a call with more tags.", &behaviours::no_copy},
    {"S2", &rules::S2, kind::expected_failure, &streamline::commandless_until_evaluate, "Illegal input and harmless, lowest priority: the copy happens inside the tag's lifetime on the evaluate's own list. Refuse it, or let the oracle accept it, when the commandless-tag work (S6) settles what a commandless tag may do.", &behaviours::copy},
    {"S24-pending", &rules::S24, kind::pending, &streamline::unsurveyed_tag_array, "Owner ruling needed: the add-on reads a multi-tag array from a release newer than the survey when its ResourceTag declares version 1, whose stride is known (sl_struct.h appends members with a version bump; S14 still refuses a later version in an array). The design's S24 refuses any such array. Either relax S24 to major 3 and above, or make read_tags refuse these arrays."},
    {"S14-pending", &rules::S14, kind::pending, &streamline::zero_header_declaration, "Owner ruling needed: the zero-header Resource layout (some 2.x integrations zero-initialise Resource's base) is accepted with its nonzero declared state as proof (streamline_camera_probe.cpp append_tag; pinned by test_streamline_camera_probe.cpp's 'nonzero compatibility resource state was discarded'). S14 accepts only a state recorded on the copy's list for a Resource that is not an SDK structure. Either rule the declaration usable and relax S14, or require an observed state as for its declared COMMON."},
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
    {"N1b", &rules::N1b, kind::expected_failure, &ngx::unknown_feature, "A feature no cited guide covers is copied from the SR contract state: in ngx_depth_source.cpp nominate features other than SR (and RR once N1 covers it) with state_proof::observed_nonzero instead of sdk_contract, so only a recorded state proves them. Fail-safe for no shipping game known today.", &behaviours::copy},
    {"N1b-pending", &rules::N1b, kind::pending, &ngx::ray_reconstruction, "Owner action needed: the DLSS-RR guide is not among the cited sources. If its resource-state section names NON_PIXEL_SHADER_RESOURCE as the SR guide does, cite it in N1 for feature 13 and drop this row; otherwise RR follows N1b like an unknown feature."},
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
        return state.source != claim::sl_declared;
      }
      return unnamed(source) && declared;
    }

    // A layout an enhanced barrier set on an earlier list, described by a
    // declaration (D6) or by NGX's contract state (N8): the copy uses a
    // legacy barrier from the claim (copy_state::resolve sees no
    // observation). A declared RESOLVE_DEST is refused (D2's table).
    constexpr bool legacy_after_earlier_layout(const device_case &value, claim source) {
      const auto &state = value.state;
      return api(value) && state.list == list_type::direct && state.resource.seen == evidence::earlier_list_enhanced && state.source == source &&
             !(source == claim::sl_declared && state.declared == D3D12_RESOURCE_STATE_RESOLVE_DEST);
    }

    constexpr bool declared_after_earlier_layout(const device_case &value) {
      return legacy_after_earlier_layout(value, claim::sl_declared);
    }

    constexpr bool contract_after_earlier_layout(const device_case &value) {
      return legacy_after_earlier_layout(value, claim::ngx_contract);
    }

    // The Generic path's end-of-frame copy declares DEPTH_WRITE for every
    // depth-stencil allocation (depth_addon.cpp preserve_d3d12_at_effects,
    // from the allocation's usage) and record_preserved copies from it.
    constexpr bool end_of_frame(const device_case &value) {
      return value.point == device_point::end_of_frame;
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
    {"D3", &rules::D3, kind::expected_failure, &device::common_on_depth_stencil, "Owner ruling R1 (decided): as the state table's D3 row; on the device a COMMON transition on a depth-stencil allocation, then the copy, is legal D3D12.", &behaviours::no_copy},
    {"D3b", &rules::D3b, kind::expected_failure, &device::declared_common_on_promotable, "As the state table's D3b row: fix with D3 (one COMMON rule keyed by depth-stencil allocation).", &behaviours::copy},
    {"D2", &rules::D2, kind::expected_failure, &device::resolve_destination, "As the state table's D2 row: fail-safe, lowest priority.", &behaviours::no_copy},
    {"D6", &rules::D6, kind::expected_failure, &device::declared_after_earlier_layout, "A texture whose layout an enhanced barrier set on an earlier list takes a legacy barrier from its declaration: illegal D3D12 (Enhanced Barriers, legacy interop). Latch 'this resource uses enhanced layouts' at device level from any texture barrier group naming it (streamline_depth_capture.cpp enhanced_textures); when latched and the recording names nothing, refuse, or copy with enhanced barriers from the layout the declaration describes (SHADER_RESOURCE for shader-resource states).", &behaviours::copy},
    {"N8", &rules::N8, kind::expected_failure, &device::contract_after_earlier_layout, "As D6 for NGX's contract state (ngx_depth_source.cpp nominates NON_PIXEL_SHADER_RESOURCE with state_proof::sdk_contract): the same device-level latch fixes both.", &behaviours::copy},
    {"D10", &rules::D10, kind::expected_failure, &device::end_of_frame, "Unsafe today, first in the fix order: depth_addon.cpp preserve_d3d12_at_effects declares DEPTH_WRITE from the allocation's usage and copies on ReShade's own list, where no game barrier is observed. Keep a per-resource fact for the last state proven on a game recording (set at ExecuteCommandLists) and copy only from it; otherwise refuse, or hold the earlier capture. Add a counter for how often an end-of-frame copy wins first, to measure the effect on generic-only games.", &behaviours::copy},
    {"D11c", &rules::D11c, kind::expected_failure, &device::read_only_unbind, "depth_addon.cpp declares DEPTH_WRITE when a DSV is unbound (on_bind_depth_stencil -> on_clear_depth_impl). Read the bound DSV's read-only flags: a writable DSV keeps DEPTH_WRITE; a read-only one with no observation is refused (D10's per-resource proven state may supply it).", &behaviours::copy},
    {"D12", &rules::D12, kind::expected_failure, &device::legacy_ui_clear, "game3d_ui_layer.cpp record_into records legacy RENDER_TARGET -> COPY_SOURCE barriers at every full-frame clear boundary: illegal on a target an enhanced barrier put in the RENDER_TARGET layout, and wrong for a simultaneous-access target still in COMMON. Refuse both (or copy an enhanced target with enhanced barriers) with the device-level layout latch D6 adds.", &behaviours::copy},
    {"S22", &rules::S22, kind::expected_failure, &device::streamline_compute_list, "Allow COMPUTE in record_impl and the consumer (streamline_depth_capture.cpp command_type, consume_owned) when the starting state is one a compute list may use (NON_PIXEL_SHADER_RESOURCE, UNORDERED_ACCESS, COPY_SOURCE, COPY_DEST, COMMON on depth-stencil, or a COMPUTE_QUEUE_* layout), with a compute-legal resting state for the snapshot storage. Until then every compute cell is refused as no_copy_point, also where the oracle refuses for its state.", &behaviours::no_copy},
    {"N7", &rules::N7, kind::expected_failure, &device::ngx_compute_list, "As S22 for NGX evaluations (dlssg-guide.txt:2321-2324: a non-copy queue); the same record_impl change fixes both.", &behaviours::no_copy},
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
