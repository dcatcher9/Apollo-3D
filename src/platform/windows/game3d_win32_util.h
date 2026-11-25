/**
 * @file src/platform/windows/game3d_win32_util.h
 * @brief Small Win32 helpers the Game 3D host files share.
 */
#pragma once

#include <cstdint>
#include <windows.h>

namespace platf::game3d::win32 {
  /** Owns one kernel handle and closes it. Neither null nor INVALID_HANDLE_VALUE is closed. */
  class handle_t {
  public:
    explicit handle_t(HANDLE value = nullptr) noexcept:
        value_(value) {}

    ~handle_t() {
      reset();
    }

    handle_t(const handle_t &) = delete;
    handle_t &operator=(const handle_t &) = delete;

    [[nodiscard]] HANDLE get() const noexcept {
      return value_;
    }

    void reset(HANDLE value = nullptr) noexcept {
      if (value_ && value_ != INVALID_HANDLE_VALUE) {
        CloseHandle(value_);
      }
      value_ = value;
    }

  private:
    HANDLE value_;
  };

  /**
   * A process's creation time as Windows FILETIME ticks, or zero when it cannot be read. With
   * the process id it names one process: an id alone is reused after the process ends.
   */
  [[nodiscard]] inline std::uint64_t process_creation_time(HANDLE process) noexcept {
    FILETIME created {}, exited {}, kernel {}, user {};
    if (!GetProcessTimes(process, &created, &exited, &kernel, &user)) {
      return 0;
    }
    return (static_cast<std::uint64_t>(created.dwHighDateTime) << 32) | created.dwLowDateTime;
  }
}  // namespace platf::game3d::win32
