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
    std::uint64_t foreground_generation = 0u;
    std::uint64_t browser_epoch = 0u;

    [[nodiscard]] constexpr bool operator==(const key_t &) const = default;
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

    /** Generation identity also applies to disabled publications; zero is never authenticated. */
    [[nodiscard]] constexpr bool matches(const std::uint64_t submitted_generation) const noexcept {
      return submitted_generation != 0u && submitted_generation == generation_;
    }

  private:
    [[nodiscard]] static constexpr key_t canonicalize(key_t candidate) noexcept {
      if (candidate.window == 0u || candidate.process_id == 0u || candidate.foreground_generation == 0u) {
        return {};
      }
      switch (candidate.kind) {
        case kind_e::browser_video:
          return candidate.browser_epoch != 0u ? candidate : key_t {};
        case kind_e::registered_video_player:
          candidate.browser_epoch = 0u;
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
