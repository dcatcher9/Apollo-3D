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
  };

  [[nodiscard]] constexpr std::string_view kind_name(const kind_e kind) noexcept {
    switch (kind) {
      case kind_e::browser_video:
        return "browser-video";
      case kind_e::registered_video_player:
        return "registered-video-player";
      default:
        return "disabled";
    }
  }

  /** Semantic video authority. Fullscreen and windowed video use the same identity rules. */
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
