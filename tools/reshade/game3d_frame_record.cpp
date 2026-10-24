// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_frame_record.h"

// ReShade's UUID templates require the Windows declarations first on MinGW.
// clang-format off
#include <Windows.h>
#include <reshade.hpp>
// clang-format on

namespace sunshine_game3d {
  struct __declspec(uuid("5b0d6c39-2f4e-4c55-9a51-7d3c0e8a4b12")) frame_record_holder {
    frame_record value;
  };
}  // namespace sunshine_game3d
#if defined(__MINGW32__)
__CRT_UUID_DECL(sunshine_game3d::frame_record_holder, 0x5b0d6c39, 0x2f4e, 0x4c55, 0x9a, 0x51, 0x7d, 0x3c, 0x0e, 0x8a, 0x4b, 0x12)
#endif

namespace sunshine_game3d {
  void update_row(source_status &row, source_status next, std::uint64_t now) {
    const bool same = row.key == next.key && row.epoch == next.epoch;
    const bool ready = next.state == source_state::ready || next.state == source_state::held;
    if (!same || !row.first_ready_ms) {
      next.first_ready_ms = ready ? now : 0;
    } else {
      next.first_ready_ms = row.first_ready_ms;
    }
    row = next;
  }

  void record_api_depth(frame_record &record, const api_depth_decision &decision, std::uint64_t present, std::uint64_t now) {
    record.present = present;
    record.tick_ms = now;
    auto &fg = record[source_slot::fg_depth];
    auto &scene = record[source_slot::scene_depth];
    auto &hold = record[source_slot::hold];
    if (!decision.allowed) {
      // The API path did not run; the Generic selector decides (D3).
      record.depth_owner = source_slot::generic;
      for (auto *row : {&fg, &scene, &hold}) {
        update_row(*row, {{}, source_state::inactive, decision.refusal}, now);
      }
      return;
    }
    const bool owned = decision.selected || decision.associated || decision.handoff_hold;
    const bool pixels = decision.copied || decision.reused;
    // This rung's source: the selected one, the held copy's or the reported
    // one. A live required FG scope owns the source choice (select_provider),
    // so it is D1 before it reports a source.
    const bool sourced = decision.selected || pixels || decision.reported;
    const bool fg_live = decision.fg_required && !decision.fg_silent;
    const bool fg_role = fg_live || (sourced && decision.frame_generation_input);
    const auto rung = fg_role ? source_slot::fg_depth : source_slot::scene_depth;
    if (!owned) {
      record.depth_owner = source_slot::generic;
    } else {
      record.depth_owner = pixels ? rung : source_slot::hold;
    }
    const auto fallback = [&](source_slot self) {
      return record.depth_owner == self ? source_slot::none : record.depth_owner;
    };
    // An FG scope is named by its row and viewport, not its source id.
    source_status source {{}, source_state::absent, decision.capture_reason};
    if (sourced) {
      source.key = decision.source;
      source.epoch = decision.epoch;
      if (fg_role) {
        source.key.id = 0;
      }
    } else if (fg_role) {
      source.key = {source_api::streamline, decision.fg_viewport};
      source.epoch = decision.fg_epoch;
    }
    if (pixels) {
      source.state = decision.copied ? source_state::ready : source_state::held;
      source.reason = decision.display_reason;
    } else {
      if (decision.selected) {
        source.state = source_state::pending;
      } else {
        source.state = decision.reported ? source_state::refused : source_state::absent;
      }
      source.fallback_to = fallback(rung);
    }
    update_row(record[rung], source, now);
    if (fg_role) {
      // A live required scope selects only its own FG input; scene depth is
      // not considered. An FG input selected without one (a silent scope's)
      // is the newest Streamline source.
      update_row(scene, {{}, source_state::inactive, fg_live ? "FG_scope_required" : "FG_input_selected"}, now);
    } else if (decision.fg_silent) {
      // No FG copy was accepted for a second: the viewport's scene depth may
      // supply depth until the next one.
      update_row(fg, {{source_api::streamline, decision.fg_viewport}, source_state::refused, "FG_scope_silent", decision.fg_epoch, fallback(source_slot::fg_depth)}, now);
    } else {
      update_row(fg, {{}, source_state::inactive, decision.fg_excluded ? "FG_scope_unconfirmed" : "FG_off"}, now);
    }
    // H: API ownership without pixels is mono, never another selector. Its
    // key is the source it holds the queue for (the owner without another).
    source_status held {decision.source, source_state::inactive, "API_depth_ready", decision.epoch};
    if (decision.handoff_hold) {
      held.state = source_state::mono;
      held.reason = "source_handoff";
    } else if (!owned) {
      held.state = source_state::absent;
      held.reason = "source_not_associated";
      held.fallback_to = source_slot::generic;
    } else if (!pixels) {
      held.state = source_state::mono;
      if (decision.selected) {
        held.reason = decision.display_reason;
      } else {
        held.reason = decision.fg_required ? "FG_scope_required" : "provider_associated";
      }
    }
    update_row(hold, held, now);
  }

  void record_generic_depth(frame_record &record, const generic_depth_decision &decision, std::uint64_t present, std::uint64_t now) {
    record.present = present;
    record.tick_ms = now;
    auto &row = record[source_slot::generic];
    if (decision.api_owner) {
      update_row(row, {{}, source_state::inactive, "API_depth_owner"}, now);
      return;
    }
    record.depth_owner = source_slot::generic;
    // The last rung: without a ready buffer the Present is mono.
    update_row(row, {{source_api::generic, 0, decision.identity}, decision.state, decision.reason}, now);
  }

  void record_ui(frame_record &record, const std::array<source_status, ui_slot_count> &rows, std::uint64_t sequence, std::uint64_t now) {
    record.ui_sequence = sequence;
    record.ui_tick_ms = now;
    for (std::size_t i = 0; i != rows.size(); ++i) {
      update_row(record.rows[static_cast<std::size_t>(source_slot::ui_alpha) + i], rows[i], now);
    }
  }

  namespace frame_record_store {
    void initialize(reshade::api::effect_runtime *runtime) {
      runtime->create_private_data<frame_record_holder>();
    }

    void destroy(reshade::api::effect_runtime *runtime) {
      runtime->destroy_private_data<frame_record_holder>();
    }

    frame_record *writable(reshade::api::effect_runtime *runtime) {
      auto *holder = runtime->get_private_data<frame_record_holder>();
      return holder ? &holder->value : nullptr;
    }
  }  // namespace frame_record_store
}  // namespace sunshine_game3d
