/**
 * @file src/platform/windows/game3d_win32_util.h
 * @brief Small Win32 helpers the Game 3D host files share.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
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

    /** Closes the held handle and returns the slot for an out parameter (CreatePipe). */
    [[nodiscard]] HANDLE *put() noexcept {
      reset();
      return &value_;
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

  /** The creation time of the process with this id, or zero when it cannot be opened or read. */
  [[nodiscard]] inline std::uint64_t process_id_creation_time(DWORD process_id) noexcept {
    const handle_t process {OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, process_id)};
    return process.get() ? process_creation_time(process.get()) : 0;
  }

  /** Windows' case-insensitive ordinal comparison (paths, executable and file names): equal. */
  [[nodiscard]] inline bool same_name(std::wstring_view left, std::wstring_view right) noexcept {
    return left.size() == right.size() &&
           CompareStringOrdinal(left.data(), static_cast<int>(left.size()), right.data(), static_cast<int>(right.size()), TRUE) == CSTR_EQUAL;
  }

  /** name begins with prefix, compared as same_name does. */
  [[nodiscard]] inline bool starts_with_name(std::wstring_view name, std::wstring_view prefix) noexcept {
    return name.size() >= prefix.size() && same_name(name.substr(0, prefix.size()), prefix);
  }

  /** name ends with suffix, compared as same_name does. */
  [[nodiscard]] inline bool ends_with_name(std::wstring_view name, std::wstring_view suffix) noexcept {
    return name.size() >= suffix.size() && same_name(name.substr(name.size() - suffix.size()), suffix);
  }

  /** A path as UTF-8 text, as JSON, logs and the Web UI carry it. */
  [[nodiscard]] inline std::string utf8(const std::filesystem::path &path) {
    const auto value = path.u8string();
    return {value.begin(), value.end()};
  }

  /** The path that UTF-8 text names. */
  [[nodiscard]] inline std::filesystem::path path_from_utf8(std::string_view value) {
    return std::filesystem::path {std::u8string(value.begin(), value.end())};
  }

  /**
   * A non-zero 64-bit id unique within this host run: a counter seeded from QueryPerformanceCounter
   * and the process id, so an id an earlier host run or another process wrote differs too. The
   * shared-memory requests (the ReShade bridge's consumer nonce, Dump 3D's nonce and request id)
   * name themselves with it.
   */
  [[nodiscard]] inline std::uint64_t unique_id() noexcept {
    static std::atomic<std::uint64_t> next {[] {
      LARGE_INTEGER now {};
      QueryPerformanceCounter(&now);
      return static_cast<std::uint64_t>(now.QuadPart) ^ (static_cast<std::uint64_t>(GetCurrentProcessId()) << 32);
    }()};
    const auto id = next.fetch_add(1, std::memory_order_relaxed);
    return id ? id : next.fetch_add(1, std::memory_order_relaxed);
  }
}  // namespace platf::game3d::win32
