// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>

namespace video {
  // These wire identities deliberately do not depend on video.h's pipeline types.
  [[nodiscard]] constexpr bool is_game_mode(int mode) noexcept {
    return mode == 2 || mode == 3;
  }

  [[nodiscard]] constexpr bool is_packed_mode(int mode) noexcept {
    return mode == 1 || mode == 3;
  }

  constexpr std::uint8_t GAME_SOURCE_WAITING = 0;
  constexpr std::uint8_t GAME_SOURCE_READY = 1;
  constexpr std::uint8_t GAME_SOURCE_UNSUPPORTED = 2;
  constexpr std::uint8_t GAME_PROVIDER_NONE = 0;
  constexpr std::uint8_t GAME_PROVIDER_RESHADE = 1;

  /** Replaceable provider observation tied to an encoder-proven presentation generation. */
  struct game_source_state_t {
    std::uint8_t state = GAME_SOURCE_WAITING;
    std::uint8_t provider = GAME_PROVIDER_NONE;
    std::uint32_t presentation_generation = 0;
    std::uint32_t source_revision = 0;
    std::uint16_t source_width = 0;
    std::uint16_t source_height = 0;
    std::uint16_t packed_width = 0;
    std::uint16_t packed_height = 0;
  };
}  // namespace video
