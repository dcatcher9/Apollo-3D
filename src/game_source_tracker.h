#pragma once

#include "game_stereo.h"

#include <chrono>
#include <cstdint>
#include <optional>

namespace video {
  struct game_source_identity_t {
    std::uint32_t process_id = 0;
    std::uint64_t creation_time = 0;
    std::uint64_t resource_generation = 0;

    bool operator==(const game_source_identity_t &) const = default;
  };

  /** Coalesces source observations without treating frame sequences as source generations. */
  class game_source_tracker_t {
  public:
    using clock_t = std::chrono::steady_clock;

    std::optional<game_source_state_t> observe(
      game_source_state_t current,
      game_source_identity_t identity,
      clock_t::time_point now
    ) {
      if (current.presentation_generation == 0 || current.source_width < 2 || current.source_height < 2 || static_cast<std::uint32_t>(current.source_width) * 2 != current.packed_width || current.source_height != current.packed_height) {
        return std::nullopt;
      }
      const bool changed = !last_ || identity != identity_ ||
                           current.presentation_generation != last_->presentation_generation ||
                           current.state != last_->state || current.provider != last_->provider ||
                           current.source_width != last_->source_width ||
                           current.source_height != last_->source_height;
      if (changed) {
        const auto previous_revision = last_ && current.presentation_generation == last_->presentation_generation ?
                                         last_->source_revision :
                                         0u;
        current.source_revision = previous_revision + 1;
        if (current.source_revision == 0) {
          current.source_revision = 1;
        }
        last_ = current;
        identity_ = identity;
      } else if (now - published_at_ < std::chrono::seconds(1)) {
        return std::nullopt;
      }
      // The heartbeat retains its revision. It repairs an early status arriving before the
      // matching atomic ACK, without inventing another ready/lost edge or automatic retry.
      published_at_ = now;
      return last_;
    }

  private:
    std::optional<game_source_state_t> last_;
    game_source_identity_t identity_;
    clock_t::time_point published_at_ {};
  };
}  // namespace video
