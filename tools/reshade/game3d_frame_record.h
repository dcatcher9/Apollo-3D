// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace reshade::api {
  struct effect_runtime;
}

// One typed record per Present of what each depth rung and UI candidate did,
// and why (docs/reshade-sbs.md, Frame record). The present thread writes it
// where the depth owners and the UI adapter decide; nothing reads it yet, so
// no log line, decision, wire byte or report changes with it. It is a fixed
// value: writing it never allocates and takes no lock.
namespace sunshine_game3d {
  // The rows: the depth ladder (D1 an FG scope's own depth, D2 scene depth,
  // H API ownership held in mono, D3 Generic), then the UI candidates in
  // capture-slot order (ui_mask::capture_order), the UI layer and the
  // presented color.
  enum class source_slot : std::uint8_t {
    fg_depth,
    scene_depth,
    hold,
    generic,
    ui_alpha,
    ui_color,
    backbuffer,
    hudless,
    ui_layer,
    current_color,
    none
  };
  inline constexpr std::size_t source_slot_count = static_cast<std::size_t>(source_slot::none);
  inline constexpr std::size_t ui_slot_count = source_slot_count - static_cast<std::size_t>(source_slot::ui_alpha);
  enum class source_api : std::uint8_t {
    none,
    streamline,
    ngx,
    generic,
    ui_layer,
    present
  };
  // inactive: out of scope this Present. absent: in scope, nothing reported.
  // refused: reported, not usable (the reason says why). pending: selected,
  // its pixels not ready. ready: current pixels, or a UI input offered to
  // detection. held: an earlier copy, or the previous real frame's UI
  // decision, stands in. mono: API ownership kept without pixels (H).
  enum class source_state : std::uint8_t {
    inactive,
    absent,
    refused,
    pending,
    ready,
    held,
    mono
  };

  struct record_key {
    source_api api {source_api::none};
    std::uint32_t viewport {};
    // The logical source within its row: NGX's feature generation, Generic's
    // resource identity; 0 for Streamline scene depth, an FG scope (its row
    // and viewport name it) and the UI kinds (one per row).
    std::uint64_t id {};
  };

  constexpr bool operator==(const record_key &a, const record_key &b) {
    return a.api == b.api && a.viewport == b.viewport && a.id == b.id;
  }

  constexpr bool operator!=(const record_key &a, const record_key &b) {
    return !(a == b);
  }

  struct source_status {
    record_key key;
    source_state state {source_state::inactive};
    // A static token (a matching log, cache or enum name), never owned.
    const char *reason {"not_recorded"};
    std::uint64_t epoch {};
    // The row that decided instead, when this row was in scope and could not
    // (absent, refused or pending); none otherwise.
    source_slot fallback_to {source_slot::none};
    // When this key and epoch first were ready or held (tick ms; 0 never).
    std::uint64_t first_ready_ms {};
  };

  struct frame_record {
    // The depth rows' Present (the native Present index) and tick.
    std::uint64_t present {}, tick_ms {};
    // The rung that decided this Present's depth: D1 or D2 with pixels, H
    // in mono under API ownership, or D3 when the Generic selector ran.
    source_slot depth_owner {source_slot::none};
    // The UI rows' render (the UI adapter's frame sequence) and tick.
    std::uint64_t ui_sequence {}, ui_tick_ms {};
    std::array<source_status, source_slot_count> rows {};

    source_status &operator[](source_slot slot) {
      return rows[static_cast<std::size_t>(slot)];
    }

    const source_status &operator[](source_slot slot) const {
      return rows[static_cast<std::size_t>(slot)];
    }
  };

  constexpr const char *name(source_slot value) {
    switch (value) {
      case source_slot::fg_depth:
        return "fg_depth";
      case source_slot::scene_depth:
        return "scene_depth";
      case source_slot::hold:
        return "hold";
      case source_slot::generic:
        return "generic";
      case source_slot::ui_alpha:
        return "ui_alpha";
      case source_slot::ui_color:
        return "ui_color";
      case source_slot::backbuffer:
        return "backbuffer";
      case source_slot::hudless:
        return "hudless";
      case source_slot::ui_layer:
        return "ui_layer";
      case source_slot::current_color:
        return "current_color";
      default:
        return "none";
    }
  }

  constexpr const char *name(source_state value) {
    switch (value) {
      case source_state::absent:
        return "absent";
      case source_state::refused:
        return "refused";
      case source_state::pending:
        return "pending";
      case source_state::ready:
        return "ready";
      case source_state::held:
        return "held";
      case source_state::mono:
        return "mono";
      default:
        return "inactive";
    }
  }

  constexpr const char *name(source_api value) {
    switch (value) {
      case source_api::streamline:
        return "streamline";
      case source_api::ngx:
        return "ngx";
      case source_api::generic:
        return "generic";
      case source_api::ui_layer:
        return "ui_layer";
      case source_api::present:
        return "present";
      default:
        return "none";
    }
  }

  // Writes next into row. first_ready_ms carries over while the key and epoch
  // are unchanged, and is set by the first ready or held state.
  void update_row(source_status &row, source_status next, std::uint64_t now);

  // The API provider's decision for one Present, as
  // streamline_depth_provider.cpp begin() made it.
  struct api_depth_decision {
    // false: the API path did not run (refusal says why); Generic decides.
    bool allowed {};
    const char *refusal {"not_recorded"};
    // The FG scope (provider::frame_generation_policy): required, required
    // but silent (frame_generation_scope_live), or revoked as unconfirmed.
    bool fg_required {}, fg_silent {}, fg_excluded {};
    std::uint64_t fg_epoch {};
    std::uint32_t fg_viewport {};
    // The acquisition selected a source. Without one, the queue stays API
    // owned while associated, or for a provider handoff (handoff_hold).
    bool selected {}, associated {}, handoff_hold {};
    // The acquisition reported a capture or nomination it did not select.
    bool reported {};
    // The source: the selected packet's, else the held copy's, else the
    // reported one; without any, the one H holds the queue for (its role is
    // then the live required FG scope's, if any).
    record_key source;
    std::uint64_t epoch {};
    bool frame_generation_input {};
    // The capture owner's selection reason, or its status without one.
    const char *capture_reason {"not_recorded"};
    // A fresh copy, or the retained copy reused; display_reason is the
    // provider's readiness reason (fresh_copy, retained_copy or the cache's).
    bool copied {}, reused {};
    const char *display_reason {"not_recorded"};
  };

  void record_api_depth(frame_record &record, const api_depth_decision &decision, std::uint64_t present, std::uint64_t now);

  // The Generic selector's outcome for one Present (depth_addon.cpp
  // begin_depth_frame): skipped while the API provider owns the Present.
  struct generic_depth_decision {
    bool api_owner {};
    source_state state {source_state::absent};
    const char *reason {"not_recorded"};
    std::uint64_t identity {};
  };

  void record_generic_depth(frame_record &record, const generic_depth_decision &decision, std::uint64_t present, std::uint64_t now);

  // One render's UI rows (game3d_ui_input_provider.cpp acquire), ui_alpha
  // to current_color in slot order.
  void record_ui(frame_record &record, const std::array<source_status, ui_slot_count> &rows, std::uint64_t sequence, std::uint64_t now);

  // The record of one effect runtime, written on its present thread only.
  namespace frame_record_store {
    void initialize(reshade::api::effect_runtime *runtime);
    void destroy(reshade::api::effect_runtime *runtime);
    // Null before initialize and after destroy.
    frame_record *writable(reshade::api::effect_runtime *runtime);
  }  // namespace frame_record_store
}  // namespace sunshine_game3d
