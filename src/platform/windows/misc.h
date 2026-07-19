/**
 * @file src/platform/windows/misc.h
 * @brief Miscellaneous declarations for Windows.
 */
#pragma once

// standard includes
#include <chrono>
#include <functional>
#include <string_view>
#include <utility>

// platform includes
#include <Windows.h>
#include <winnt.h>

namespace platf {
  namespace detail {
    class mouse_keys_controller_t {
    public:
      template<class Getter, class Setter>
      bool refresh(bool mouse_present, Getter &&getter, Setter &&setter) {
        if (enabled_by_host_ || mouse_present) {
          return false;
        }

        MOUSEKEYS current {};
        current.cbSize = sizeof(current);
        if (!std::invoke(std::forward<Getter>(getter), current)) {
          return false;
        }

        constexpr DWORD required_flags = MKF_MOUSEKEYSON | MKF_AVAILABLE;
        if ((current.dwFlags & required_flags) == required_flags) {
          return false;
        }

        auto replacement = current;
        replacement.dwFlags |= required_flags;
        if (!std::invoke(std::forward<Setter>(setter), replacement)) {
          return false;
        }

        previous_state_ = current;
        enabled_by_host_ = true;
        return true;
      }

      template<class Setter>
      bool restore(Setter &&setter) {
        if (!enabled_by_host_) {
          return false;
        }

        if (!std::invoke(std::forward<Setter>(setter), previous_state_)) {
          return false;
        }

        enabled_by_host_ = false;
        previous_state_ = {};
        return true;
      }

      [[nodiscard]] bool enabled_by_host() const {
        return enabled_by_host_;
      }

    private:
      bool enabled_by_host_ = false;
      MOUSEKEYS previous_state_ {};
    };
  }  // namespace detail

  void print_status(const std::string_view &prefix, HRESULT status);
  HDESK syncThreadDesktop();

  int64_t qpc_counter();

  std::chrono::nanoseconds qpc_time_difference(int64_t performance_counter1, int64_t performance_counter2);

  /**
   * @brief Convert a UTF-8 string into a UTF-16 wide string.
   * @param string The UTF-8 string.
   * @return The converted UTF-16 wide string.
   */
  std::wstring from_utf8(const std::string_view &string);

  /**
   * @brief Convert a UTF-16 wide string into a UTF-8 string.
   * @param string The UTF-16 wide string.
   * @return The converted UTF-8 string.
   */
  std::string to_utf8(const std::wstring_view &string);
}  // namespace platf
