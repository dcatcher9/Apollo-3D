#pragma once
// Source-conversion ordering only. This is neither model publication nor a presentation clock.
#include <cstdint>

namespace platf::dxgi::detail {
  // One converter owns monotonically increasing conversion IDs. Reset only with that converter's
  // explicit source namespace reset. Repeating packed pixels does not advance this watermark.
  class host_sbs_drawn_source_order_t {
    std::uint64_t last_drawn_ = 0u;

  public:
    void reset() noexcept {
      last_drawn_ = 0u;
    }

    [[nodiscard]] bool permits(std::uint64_t source_conversion) const noexcept {
      return source_conversion != 0u && source_conversion >= last_drawn_;
    }

    bool note_drawn(std::uint64_t source_conversion) noexcept {
      if (!permits(source_conversion)) {
        return false;
      }
      last_drawn_ = source_conversion;
      return true;
    }

    [[nodiscard]] std::uint64_t last_drawn() const noexcept {
      return last_drawn_;
    }
  };
}  // namespace platf::dxgi::detail
