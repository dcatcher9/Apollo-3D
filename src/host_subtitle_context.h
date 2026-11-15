/**
 * @file src/host_subtitle_context.h
 * @brief Host subtitle eligibility and revocation independent of depth geometry.
 */
#pragma once

#include <cstdint>
#include <string_view>

namespace models::host_subtitle_context {

  enum class kind_e : std::uint8_t {
    disabled,
    browser_video,
    registered_video_player,
    foreground_app,
  };

  [[nodiscard]] constexpr std::string_view kind_name(const kind_e kind) noexcept {
    switch (kind) {
      case kind_e::browser_video:
        return "browser-video";
      case kind_e::registered_video_player:
        return "registered-video-player";
      case kind_e::foreground_app:
        return "foreground-app";
      default:
        return "disabled";
    }
  }

  /**
   * Native (non-browser) admission for a causal whole-client foreground candidate. A registered
   * video player needs no opt-in. Any other application, such as a 2D game, is admitted only by
   * the owner's Host 3D foreground-subtitle switch; it carries no video evidence of its own.
   */
  [[nodiscard]] constexpr kind_e native_kind(
    const bool registered_video_player,
    const bool foreground_app_enabled
  ) noexcept {
    return registered_video_player ? kind_e::registered_video_player :
           foreground_app_enabled ? kind_e::foreground_app :
                                    kind_e::disabled;
  }

  /**
   * Video captions are centered, so off-center text is rejected as scene text. An owner-enabled
   * foreground app, such as a 2D game, anchors its text elsewhere; only the bottom position holds.
   */
  [[nodiscard]] constexpr bool requires_centered_lines(const kind_e kind) noexcept {
    return kind != kind_e::foreground_app;
  }

  /** Native kinds identify a root window and process only; they share retention rules. */
  [[nodiscard]] constexpr bool is_native_kind(const kind_e kind) noexcept {
    return kind == kind_e::registered_video_player || kind == kind_e::foreground_app;
  }

  /** Semantic subtitle authority. Fullscreen and windowed content use the same identity rules. */
  struct key_t {
    kind_e kind = kind_e::disabled;
    std::uint64_t window = 0u;
    std::uint32_t process_id = 0u;
    // Geometry runs are authenticated separately by the matched-frame owner. Moving the same
    // video must not create a new semantic context merely because its rectangle changed.
    std::int32_t document_id = 0;
    std::int32_t video_id = 0;

    [[nodiscard]] constexpr bool operator==(const key_t &) const = default;
  };

  /** Positioned presentation authority is separate from the retained semantic video identity. */
  struct presentation_authority_t {
    std::uint64_t foreground_generation = 0u;
    std::uint64_t region_generation = 0u;
    std::uint64_t browser_epoch = 0u;
    bool interactive_move_size = false;

    [[nodiscard]] constexpr bool operator==(const presentation_authority_t &) const = default;

    [[nodiscard]] constexpr bool permits_packed_repeat(
      const presentation_authority_t &current,
      const bool positioned_output
    ) const noexcept {
      return !positioned_output || *this == current;
    }
  };

  /** Each context transition revokes previous submissions, including video/desktop/video ABA. */
  class tracker_t {
  public:
    constexpr bool update(key_t candidate) noexcept {
      candidate = canonicalize(candidate);
      if (candidate == key_) {
        return false;
      }
      key_ = candidate;
      if (++generation_ == 0u) {
        ++generation_;
      }
      return true;
    }

    [[nodiscard]] constexpr bool eligible() const noexcept {
      return key_.kind != kind_e::disabled;
    }

    [[nodiscard]] constexpr std::uint64_t generation() const noexcept {
      return generation_;
    }

    [[nodiscard]] constexpr kind_e kind() const noexcept {
      return key_.kind;
    }

    [[nodiscard]] constexpr key_t key() const noexcept {
      return key_;
    }

    /** Generation identity also applies to disabled publications; zero is never authenticated. */
    [[nodiscard]] constexpr bool matches(const std::uint64_t submitted_generation) const noexcept {
      return submitted_generation != 0u && submitted_generation == generation_;
    }

  private:
    [[nodiscard]] static constexpr key_t canonicalize(key_t candidate) noexcept {
      if (candidate.window == 0u || candidate.process_id == 0u) {
        return {};
      }
      switch (candidate.kind) {
        case kind_e::browser_video:
          return candidate.document_id != 0 && candidate.video_id != 0 ? candidate : key_t {};
        case kind_e::registered_video_player:
        case kind_e::foreground_app:
          candidate.document_id = 0;
          candidate.video_id = 0;
          return candidate;
        case kind_e::disabled:
        default:
          return {};
      }
    }

    key_t key_;
    std::uint64_t generation_ = 1u;
  };

}  // namespace models::host_subtitle_context
