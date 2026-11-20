#pragma once

#include "platform/common.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <memory>

namespace video::detail {
  enum class capture_image_retention_e { keep, reclaim_idle };

  // The capture thread owns this pool. Each returned shared pointer keeps its exact image and
  // backing alive until all consumers release it.
  template<std::size_t Capacity, class Clock = std::chrono::steady_clock>
  class capture_image_pool_t {
  public:
    explicit capture_image_pool_t(capture_image_retention_e retention = capture_image_retention_e::keep):
        retention_(retention) {}

    capture_image_pool_t(const capture_image_pool_t &) = delete;
    capture_image_pool_t &operator=(const capture_image_pool_t &) = delete;

    template<class Allocate>
    std::shared_ptr<platf::img_t> try_acquire(Allocate &&allocate) {
      const auto slot = std::find_if(slots_.begin(), slots_.end(), [](const auto &entry) {
        return !entry.image || entry.image.use_count() == 1;
      });
      if (slot == slots_.end()) {
        return {};
      }
      if (!slot->image) {
        slot->image = allocate();
        if (!slot->image) {
          return {};
        }
      }

      // Allocated entries stay ahead of empty slots. Prefer the most recently used free backing
      // so Desktop Duplication can reuse its accumulated damage and cursor history.
      if (slot != slots_.begin()) {
        std::rotate(slots_.begin(), slot, slot + 1);
      }
      auto image = slots_.front().image;
      if (retention_ == capture_image_retention_e::reclaim_idle) {
        const auto now = Clock::now();
        slots_.front().last_checkout = now;
        if (now >= next_sweep_) {
          const auto unused = std::remove_if(slots_.begin(), slots_.end(), [now](const auto &entry) {
            return entry.image && entry.image.use_count() == 1 &&
                   now - entry.last_checkout >= std::chrono::seconds {3};
          });
          std::fill(unused, slots_.end(), slot_t {});
          next_sweep_ = now + std::chrono::seconds {1};
        }
      }

      image->frame_timestamp.reset();
      image->content_timestamp.reset();
      image->pixels_skipped = false;
      image->conversion_needed = true;
      return image;
    }

    void clear() {
      slots_ = {};
      next_sweep_ = Clock::time_point::min();
    }

  private:
    struct slot_t {
      std::shared_ptr<platf::img_t> image;
      typename Clock::time_point last_checkout {};
    };

    static_assert(Capacity > 0);
    const capture_image_retention_e retention_;
    std::array<slot_t, Capacity> slots_ {};
    typename Clock::time_point next_sweep_ = Clock::time_point::min();
  };
}  // namespace video::detail
