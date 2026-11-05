#pragma once

#include <cstdint>
#include <optional>

namespace platf::dxgi::detail {
  // Eye phase advances only when DXGI proves the requested present reached scanout. A late or
  // missing flip invalidates the session instead of guessing an eye from elapsed wall time.
  class shutter_presentation_clock_t {
  public:
    explicit shutter_presentation_clock_t(unsigned eye_repeats): repeats_(eye_repeats) {}

    [[nodiscard]] bool left_eye() const noexcept {
      return left_;
    }

    [[nodiscard]] bool starts_pair() const noexcept {
      return left_;
    }

    bool observe_displayed(std::uint32_t refresh_count) noexcept {
      if (repeats_ < 1 || repeats_ > 4 ||
          (last_refresh_ && static_cast<std::uint32_t>(refresh_count - *last_refresh_) != repeats_)) {
        return false;
      }
      last_refresh_ = refresh_count;
      left_ = !left_;
      return true;
    }

  private:
    unsigned repeats_;
    bool left_ = true;
    std::optional<std::uint32_t> last_refresh_;
  };
}  // namespace platf::dxgi::detail
