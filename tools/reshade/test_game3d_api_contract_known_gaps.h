// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Where the add-on does not meet the API contract oracle yet. Each row names
// the oracle rule it misses, a predicate over the matrix dimensions that
// selects exactly the cells concerned, and an owner note.
//
// expected_failure: every matching cell must fail the oracle. A matching cell
// that passes fails the run, so the fix that closes a gap also removes its
// row (and updates any test that pinned the old behaviour). A row that
// matches no failing cell fails the run as stale.
// pending: the cell's verdict is not enforced (EITHER) until owner work
// lands; the global invariants still apply.
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

  // The state a copy may start from (checked by test_capture_state_policy.cpp).
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

  // Streamline tags. No matrix runs these cells yet; rows that touch the
  // concurrent Assassin's Creed Shadows work stay pending until it lands, and
  // the hook matrix adds its expected failures here.
  namespace streamline {
    constexpr bool dynamic_resolution(const sl_case &value) {
      return value.extent == extent_form::changing;
    }

    constexpr bool generated_frame_reuse(const sl_case &value) {
      return value.generated_presents != 0;
    }
  }  // namespace streamline

  inline constexpr row<sl_case> sl_rows[] {
    {"S16e-pending", &rules::S16e, kind::pending, &streamline::dynamic_resolution, "Dynamic-resolution continuity (e31c9d2c) and R3 calibration carry-over: EITHER until the Assassin's Creed Shadows workflow lands."},
    {"S7-pending", &rules::S7, kind::pending, &streamline::generated_frame_reuse, "Generated Presents between a tag and its use touch the generated-frame reuse in progress in the Assassin's Creed Shadows workflow: EITHER until it lands, then the hook matrix records or closes S7."},
  };

  // NGX evaluations. No matrix runs these cells yet.
  namespace ngx {
    constexpr bool direct_frame_generation(const ngx_case &value) {
      return value.feature == ngx_feature::frame_generation;
    }
  }  // namespace ngx

  inline constexpr row<ngx_case> ngx_rows[] {
    {"N4-pending", &rules::N4, kind::pending, &ngx::direct_frame_generation, "Live DLSS-G NGX depth is in progress in the Assassin's Creed Shadows workflow: EITHER until it lands."},
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
}  // namespace game3d_contract::known_gaps
