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
  }  // namespace behaviours

  // The state a copy may start from (checked by test_game3d_api_contract_cpu.cpp).
  // The add-on meets every cell; C++ has no empty array, so one pending row
  // that matches no cell stands in for the table.
  namespace state {
    constexpr bool no_cell(const state_case &) {
      return false;
    }
  }  // namespace state

  inline constexpr row<state_case> state_rows[] {
    {"none", &rules::D1, kind::pending, &state::no_cell, "No copy-state gap remains: Stage 3 batch B closed D3, D3b (with S11-common), D2 (with S11-resolve), X1, X2 and D4c. This row matches no cell."},
  };

  // Raw Streamline buffer types at each surveyed release (checked by
  // test_game3d_api_contract_cpu.cpp). The add-on meets every cell; one
  // pending row that matches no cell stands in for the table.
  namespace buffer_types {
    constexpr bool no_cell(const buffer_case &) {
      return false;
    }
  }  // namespace buffer_types

  inline constexpr row<buffer_case> buffer_rows[] {
    {"none", &rules::S17, kind::pending, &buffer_types::no_cell, "No buffer-type gap remains: Stage 3 batch B closed S17 (a 2.2.1 tier with last type 52). This row matches no cell."},
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

    constexpr bool no_cell(const sl_case &) {
      return false;
    }

    // A legal cell where the oracle permits nothing but a capture (an EITHER
    // cell may also refuse, so no gap can match it).
    constexpr bool capture_required(const sl_case &value) {
      const auto expected = expect(value);
      return legal(value) && (expected.kind == outcome_kind::capture || expected.kind == outcome_kind::continuous) && !either(expected);
    }

  }  // namespace streamline

  inline constexpr row<sl_case> sl_rows[] {
    {"none", &rules::S1, kind::pending, &streamline::no_cell, "No Streamline hook gap remains: Stage 3 batch B closed S4, S15, S16 and S16d (and with earlier fixes every other row). This row matches no cell."},
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

  // The runtime layer's cells (judged at Present).
  namespace present {
    // An OnlyValidNow tag outside frame-generation scope that its call
    // copies (S1), followed by the same frame's evaluate, which refuses the
    // tag as not valid until present (S3): that Present shows no depth.
    constexpr bool only_valid_now_then_evaluate(const present_case &value) {
      return value.value.at == copy_point::same_frame_evaluate && !streamline::local(value.value.entry) && value.decided.at == copy_point::tag_call &&
             value.decided.life == lifecycle::only_valid_now && !detail::frame_generation_scope(value.value) && streamline::capture_required(value.decided);
    }

  }  // namespace present

  // Streamline cells run through the native provider fixture (judged with
  // judge_each). A row that shares an id with the hook table's names the
  // same gap; that row's note holds the fix.
  inline constexpr row<present_case> sl_runtime_rows[] {
    {"S1-evaluate", &rules::S1, kind::expected_failure, &present::only_valid_now_then_evaluate, "Outside the Stage 3 order, which predates this finding: it reworks the tag-to-evaluate path fix 2 (S6 + S7) changes, so it fits beside it. Found by the runtime layer: an OnlyValidNow global tag outside FG scope is copied at its call, but the same frame's evaluate refuses the tag (match(): not-valid-until-present, S3) and that frame's Present then shows no depth (mono), or only between generated Presents. S3 is right that the evaluate may not use the tag; it must not cost the frame the copy its tag call made (S1). Today the refused evaluate begins a new evaluation of the viewport with no copy (streamline_camera_probe.cpp nominate_depth_source, withdraw_invalid_attempt -> depth_capture::begin_evaluation), which supersedes the tag call's. Keep a tag-call copy when a later point of its frame refuses the same tag for its lifetime. The hook layer judges one copy point per cell, so only a Present shows this.", &behaviours::no_copy},
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
  // is offered the input (Dump3D's UI candidates). The add-on meets every
  // cell; one pending row that matches no cell stands in for the table.
  namespace ui {
    constexpr bool no_cell(const ui_case &) {
      return false;
    }
  }  // namespace ui

  inline constexpr row<ui_case> ui_runtime_rows[] {
    {"none", &rules::U3, kind::pending, &ui::no_cell, "No UI runtime gap remains: Stage 3 batch B closed U3 (a local UI input is offered at its evaluate). This row matches no cell."},
  };

  // Copies recorded on real lists (checked by
  // test_game3d_api_contract_device.cpp, judged with judge_each). A cell whose
  // row records a copy is recorded but never executed: that copy starts from
  // a state nothing proved.
  namespace device {
    constexpr bool api(const device_case &value) {
      return value.point == device_point::api;
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
