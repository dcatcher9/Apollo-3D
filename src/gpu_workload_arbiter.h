/**
 * @file src/gpu_workload_arbiter.h
 * @brief Atomic admission between latency-sensitive streaming and offline GPU conversion.
 */
#pragma once

#include <optional>

namespace gpu_workload {
  enum class kind_e {
    live_stream,
    offline_sbs,
  };

  class lease_t {
  public:
    lease_t() = default;
    ~lease_t();

    lease_t(const lease_t &) = delete;
    lease_t &operator=(const lease_t &) = delete;
    lease_t(lease_t &&other) noexcept;
    lease_t &operator=(lease_t &&other) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept {
      return held_;
    }

    void reset() noexcept;

  private:
    friend std::optional<lease_t> try_acquire(kind_e kind);
    explicit lease_t(kind_e kind):
        kind_(kind),
        held_(true) {
    }

    kind_e kind_ = kind_e::live_stream;
    bool held_ = false;
  };

  /**
   * Acquire one mutually exclusive GPU workload slot.
   *
   * Multiple live leases are counted defensively, but an offline lease excludes every live
   * lease and vice versa. The check and reservation are one operation, closing the
   * live-start/offline-create race.
   */
  std::optional<lease_t> try_acquire(kind_e kind);

  [[nodiscard]] bool live_stream_active();
  [[nodiscard]] bool offline_sbs_active();
}  // namespace gpu_workload
