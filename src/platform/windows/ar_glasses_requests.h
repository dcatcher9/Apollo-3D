/**
 * @file src/platform/windows/ar_glasses_requests.h
 * @brief Local 3D start and stop requests, each surface's chosen 3D source and the last refusal,
 * shared by target discovery, the local session and its controller in ar_glasses.cpp.
 */
#pragma once

#include <atomic>
#include <mutex>
#include <string>

namespace ar_glasses {
  namespace internal {
    extern std::atomic<bool> physical_monitor_active;
    extern std::atomic<bool> physical_monitor_requested;
    extern std::atomic<bool> ar_session_requested;
    // -1 means not switched yet: the saved setting applies.
    extern std::atomic<int> ar_source_game;
    extern std::atomic<int> monitor_source_game;

    bool ar_uses_game_source();
    bool monitor_uses_game_source();
    bool ar_session_allowed();

    extern std::mutex physical_monitor_error_mutex;
    extern std::string physical_monitor_error;
  }  // namespace internal
}  // namespace ar_glasses
