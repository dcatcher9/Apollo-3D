/**
 * @file src/platform/windows/ar_glasses_arbiter.h
 * @brief The virtual-display arbiter: one SudoVDA owner at a time between local presentation and a
 * remote stream, including a retired local display that has not yet left Windows.
 */
#pragma once

#include "ar_glasses.h"
#include "virtual_display_session.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <stop_token>

namespace ar_glasses {
  namespace internal {
    /** Guards the ownership state below and the arbiter's own remote leases. */
    extern std::mutex ownership_mutex;
    extern std::condition_variable ownership_changed;
    extern bool local_session_present;
    extern std::optional<std::stop_source> local_session_construction_stop;
    extern std::optional<VDISPLAY::session_t> retired_local_virtual_display;

    bool remote_blocks_local_locked(std::chrono::steady_clock::time_point now);
    void release_local_virtual_display_claim_impl();
    bool begin_local_virtual_display_retirement(
      VDISPLAY::session_t &&session,
      VDISPLAY::session_retirement_callbacks_t callbacks
    );
    bool wait_for_local_virtual_display_retirement_impl(
      std::chrono::milliseconds timeout
    );
  }  // namespace internal
}  // namespace ar_glasses
