/**
 * @file src/gpu_workload_arbiter.cpp
 * @brief Atomic GPU workload admission implementation.
 */

#include "gpu_workload_arbiter.h"

#include <cstddef>
#include <mutex>
#include <utility>

namespace gpu_workload {
  namespace {
    std::mutex mutex;
    std::size_t live_leases = 0;
    bool offline_lease = false;

    void release(const kind_e kind) noexcept {
      std::lock_guard lock {mutex};
      if (kind == kind_e::live_stream) {
        if (live_leases > 0) {
          --live_leases;
        }
      } else {
        offline_lease = false;
      }
    }
  }  // namespace

  lease_t::~lease_t() {
    reset();
  }

  lease_t::lease_t(lease_t &&other) noexcept:
      kind_(other.kind_),
      held_(std::exchange(other.held_, false)) {
  }

  lease_t &lease_t::operator=(lease_t &&other) noexcept {
    if (this != &other) {
      reset();
      kind_ = other.kind_;
      held_ = std::exchange(other.held_, false);
    }
    return *this;
  }

  void lease_t::reset() noexcept {
    if (std::exchange(held_, false)) {
      release(kind_);
    }
  }

  std::optional<lease_t> try_acquire(const kind_e kind) {
    std::lock_guard lock {mutex};
    if (kind == kind_e::live_stream) {
      if (offline_lease) {
        return std::nullopt;
      }
      ++live_leases;
    } else {
      if (offline_lease || live_leases != 0) {
        return std::nullopt;
      }
      offline_lease = true;
    }
    return lease_t {kind};
  }

  bool live_stream_active() {
    std::lock_guard lock {mutex};
    return live_leases != 0;
  }

  bool offline_sbs_active() {
    std::lock_guard lock {mutex};
    return offline_lease;
  }
}  // namespace gpu_workload
