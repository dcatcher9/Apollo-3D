// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>
#include <limits>
#include <tuple>

namespace sunshine_game3d::ui_qualification {
  enum class choice : std::uint32_t { automatic, current_color, sl_ui_alpha, sl_ui_color_alpha, sl_backbuffer, sl_hudless };

  inline bool valid(choice value) {
    return value >= choice::automatic && value <= choice::sl_hudless;
  }

  inline const char *name(choice value) {
    switch (value) {
      case choice::automatic: return "automatic";
      case choice::current_color: return "current_color";
      case choice::sl_ui_alpha: return "sl_ui_alpha";
      case choice::sl_ui_color_alpha: return "sl_ui_color_alpha";
      case choice::sl_backbuffer: return "sl_backbuffer";
      case choice::sl_hudless: return "sl_hudless_difference";
      default: return "invalid";
    }
  }

  // Logical source identity supplied by the presentation adapter. Native resource
  // pointers, capture tickets and frame sequences are deliberately absent: they
  // rotate during normal use and remain the pixel owner's eligibility contract.
  struct scope {
    std::uint64_t runtime{}, device{}, source_id{}, epoch{}, revision{}, semantic_contract{};
    std::uint32_t provider{}, viewport{};
    choice source = choice::automatic; // Automatic denotes the current candidate set.
    bool fg_known{}, fg_enabled{}, fg_automatic{};
    std::uint32_t fg_generated_frames{};
    std::uint32_t output_width{}, output_height{}, output_format{}, color_space{};
    std::uint32_t mask_width{}, mask_height{}, mask_format{}, channel{};
    std::uint32_t left{}, top{}, width{}, height{};

    bool valid() const {
      return ui_qualification::valid(source) && runtime && device &&
        output_width && output_height && output_format && mask_width && mask_height &&
        mask_format && channel <= 1 && width && height && left <= mask_width && top <= mask_height &&
        width <= mask_width - left && height <= mask_height - top;
    }

    bool operator==(const scope &other) const {
      return std::tie(runtime, device, source_id, epoch, revision, semantic_contract, provider, viewport, source,
        fg_known, fg_enabled, fg_automatic, fg_generated_frames, output_width, output_height, output_format,
        color_space, mask_width, mask_height, mask_format, channel, left, top, width, height) ==
        std::tie(other.runtime, other.device, other.source_id, other.epoch, other.revision, other.semantic_contract,
          other.provider, other.viewport, other.source, other.fg_known, other.fg_enabled, other.fg_automatic,
          other.fg_generated_frames, other.output_width, other.output_height, other.output_format, other.color_space,
          other.mask_width, other.mask_height, other.mask_format, other.channel, other.left, other.top,
          other.width, other.height);
    }
    bool operator!=(const scope &other) const { return !(*this == other); }
  };

  struct status {
    choice selected = choice::automatic;
    scope candidate;
    std::uint64_t token{}, choice_revision{};
    bool available{};
  };

  // Read-only source availability and an optional troubleshooting selection.
  // Quality is checked from current pixels on the GPU, not approved by this
  // object. The diagnostic token changes with logical source identity only.
  class session {
  public:
    explicit session(std::uint64_t initial_token = 0) : token_(initial_token) {}

    choice selected_choice() const { return selected_; }

    bool set_choice(choice value) {
      if (!valid(value) || value == selected_) return false;
      selected_ = value;
      if (choice_revision_ != std::numeric_limits<std::uint64_t>::max()) ++choice_revision_;
      candidate_ = {};
      have_candidate_ = available_ = false;
      advance_token();
      return true;
    }

    status observe(const scope &candidate, bool available = true) {
      if (!have_candidate_ || candidate_ != candidate) {
        candidate_ = candidate;
        have_candidate_ = true;
        advance_token();
      }
      available_ = available && candidate.valid() &&
        (selected_ == choice::automatic || selected_ == candidate.source);
      return snapshot();
    }

    // A pending/absent capture preserves identity but never availability.
    void unavailable() { available_ = false; }

    void clear() {
      candidate_ = {};
      have_candidate_ = available_ = false;
      advance_token();
    }

    status snapshot() const {
      return {selected_, candidate_, token_, choice_revision_, available_};
    }

  private:
    void advance_token() {
      if (token_ != std::numeric_limits<std::uint64_t>::max()) ++token_;
    }

    choice selected_ = choice::automatic;
    scope candidate_;
    std::uint64_t token_{}, choice_revision_{};
    bool have_candidate_{}, available_{};
  };
}
