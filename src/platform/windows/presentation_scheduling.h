/**
 * @file src/platform/windows/presentation_scheduling.h
 * @brief Shared lifetime of process-wide local and remote presentation scheduling.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

namespace platf::detail {
  // A native request and the multimedia fallback have different release APIs. Record only
  // successful acquisition and retain a failed release for retry without acquiring it twice.
  class presentation_timer_request_t {
  public:
    template<class AcquireNative, class AcquireMultimedia>
    void start(AcquireNative acquire_native, AcquireMultimedia acquire_multimedia) {
      if (native_resolution_ || multimedia_requested_) {
        return;
      }
      native_resolution_ = acquire_native();
      if (!native_resolution_) {
        multimedia_requested_ = acquire_multimedia();
      }
    }

    template<class ReleaseNative, class ReleaseMultimedia>
    void stop(ReleaseNative release_native, ReleaseMultimedia release_multimedia) {
      if (native_resolution_ && release_native(*native_resolution_)) {
        native_resolution_.reset();
      }
      if (multimedia_requested_ && release_multimedia()) {
        multimedia_requested_ = false;
      }
    }

  private:
    std::optional<std::uint32_t> native_resolution_;
    bool multimedia_requested_ = false;
  };

  /**
   * Serialize first-owner setup and last-owner cleanup. The controller must outlive its leases.
   * Operations tracks which best-effort platform changes succeeded and undoes only those.
   */
  template<class Operations>
  class presentation_scheduling_t {
  public:
    class lease_t {
    public:
      lease_t() = default;
      lease_t(const lease_t &) = delete;
      lease_t &operator=(const lease_t &) = delete;

      lease_t(lease_t &&other) noexcept:
          owner_(std::exchange(other.owner_, nullptr)) {
      }

      lease_t &operator=(lease_t &&other) noexcept {
        if (this != &other) {
          reset();
          owner_ = std::exchange(other.owner_, nullptr);
        }
        return *this;
      }

      ~lease_t() {
        reset();
      }

      void reset() noexcept {
        if (auto *owner = std::exchange(owner_, nullptr)) {
          owner->release();
        }
      }

    private:
      friend class presentation_scheduling_t;

      explicit lease_t(presentation_scheduling_t *owner):
          owner_(owner) {
      }

      presentation_scheduling_t *owner_ = nullptr;
    };

    explicit presentation_scheduling_t(Operations operations = {}):
        operations_(std::move(operations)) {
    }

    [[nodiscard]] lease_t acquire() {
      std::lock_guard lock(mutex_);
      if (owners_ == 0) {
        try {
          operations_.start();
        } catch (...) {
          operations_.stop();
          throw;
        }
      }
      ++owners_;
      return lease_t {this};
    }

  private:
    void release() noexcept {
      std::lock_guard lock(mutex_);
      if (--owners_ == 0) {
        operations_.stop();
      }
    }

    std::mutex mutex_;
    std::size_t owners_ = 0;
    Operations operations_;
  };
}  // namespace platf::detail
