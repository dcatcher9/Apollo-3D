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

  // Two SR viewports active together (S19, owner ruling R5), judged at the
  // capture owner's selection (test_game3d_api_contract_cpu.cpp, family
  // sl_views, judged with judge_each). Each viewport meets the oracle for
  // its own case; the add-on meets every cell.
  namespace views {
    constexpr bool no_cell(const sl_case &) {
      return false;
    }
  }  // namespace views

  inline constexpr row<sl_case> view_rows[] {
    {"none", &rules::S19, kind::pending, &views::no_cell, "No two-viewport gap remains: lane F (11-01) closed S19-dynamic-zero-frames. DLSSGMode eDynamic with numFramesToGenerate 0 (options accepted) is enabled FG to the probe, and its viewport broke a tie of two active viewports (R5), where the oracle reads a count of 0 as no frame generation (S20), so both viewports refuse. sl_dlss_g.h (2.11.1 to 2.14.1) does not say whether Dynamic then generates frames, so the add-on now takes the fail-safe reading: an FG scope without a declared generated count names no tie viewport (fg_view::update) and the tie stays. This row matches no cell."},
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

  // The runtime layer's cells (judged at Present). The add-on meets every
  // cell; one pending row that matches no cell stands in for the table.
  namespace present {
    constexpr bool no_cell(const present_case &) {
      return false;
    }
  }  // namespace present

  // Streamline cells run through the native provider fixture (judged with
  // judge_each). A row that shares an id with the hook table's names the
  // same gap; that row's note holds the fix.
  inline constexpr row<present_case> sl_runtime_rows[] {
    {"none", &rules::S1, kind::pending, &present::no_cell, "No Streamline runtime gap remains: the fix of 10-26 closed S1-evaluate, found by this layer (an OnlyValidNow global or frame tag outside FG scope, copied at its call (S1), whose same-frame evaluate refused it for its lifetime (S3) and so began an evaluation of the viewport with no copy: that frame's Present was mono). That evaluate now keeps the tag call's copy (api_contract.h keeps_tag_call_copy); the runtime family's gap seed now-evaluate keeps its cell in every release. This row matches no cell."},
  };

  // NGX evaluations, DLSS-G's own (N4) included (checked by the hook layer).
  // The add-on meets every cell; one pending row that matches no cell stands
  // in for the table.
  namespace ngx {
    constexpr bool no_cell(const ngx_case &) {
      return false;
    }
  }  // namespace ngx

  inline constexpr row<ngx_case> ngx_rows[] {
    {"none", &rules::N1, kind::pending, &ngx::no_cell, "No NGX hook gap remains: Stage 3 batch B closed N1b (Ray Reconstruction and any feature but Super Resolution are proven only by an observed state; the DLSS-RR Integration Guide states no input state). This row matches no cell."},
  };

  // NGX evaluations judged at Present by the runtime layer
  // (test_game3d_api_contract_runtime.cpp, family ngx, judged with
  // judge_each). The add-on meets every cell; one pending row that matches no
  // cell stands in for the table.
  inline constexpr row<ngx_case> ngx_runtime_rows[] {
    {"none", &rules::N1, kind::pending, &ngx::no_cell, "No NGX runtime gap remains: Stage 3 batch B closed N1b. This row matches no cell."},
  };

  // Generic end-of-frame copies judged at Present by the runtime layer
  // (family generic). The add-on meets every one; C++ has no empty array, so
  // one pending row that matches no cell stands in for the table.
  namespace generic {
    constexpr bool no_cell(const generic_case &) {
      return false;
    }
  }  // namespace generic

  inline constexpr row<generic_case> generic_runtime_rows[] {
    {"none", &rules::D10, kind::pending, &generic::no_cell, "No Generic runtime gap remains: Stage 3 fix 1 closed D10 (the end-of-frame copy). This row matches no cell."},
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
  // a state nothing proved. The add-on meets every cell; one pending row that
  // matches no cell stands in for the table.
  namespace device {
    constexpr bool no_cell(const device_case &) {
      return false;
    }
  }  // namespace device

  inline constexpr row<device_case> device_rows[] {
    {"none", &rules::S22, kind::pending, &device::no_cell, "No device gap remains: Stage 3 batch B closed S22 with N7 (copies on a COMPUTE list from a state or layout that list may name). This row matches no cell."},
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
