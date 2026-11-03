#pragma once

#include <cstdint>

// CPU diagnostic ownership only. Neither role proves GPU completion or UI presence.
namespace models::current_outline_observation_v1 {
  struct identity_t {
    std::uint64_t frame = 0, observation_us = 0, generation = 0, token = 0;
    std::uint32_t work = 0;
    constexpr bool operator==(const identity_t &) const = default;
  };

  enum class role_e { unavailable, queued, completed };

  [[nodiscard]] constexpr role_e classify(
    const std::uint64_t frame, const std::uint64_t observation,
    const identity_t &current, const identity_t &pending,
    const bool has_pending_frame, const bool postprocess_queued,
    const std::uint64_t ticket, const identity_t &published,
    const bool publication_current
  ) noexcept {
    if (!frame || !observation || current.frame != frame ||
        current.observation_us != observation || !current.token || current.work > 2u)
      return role_e::unavailable;
    if (has_pending_frame && postprocess_queued && ticket && current == pending)
      return role_e::queued;
    if (publication_current && current == published)
      return role_e::completed;
    return role_e::unavailable;
  }
}
