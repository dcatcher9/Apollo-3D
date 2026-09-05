/**
 * @file src/platform/windows/misc.h
 * @brief Miscellaneous declarations for Windows.
 */
#pragma once

// standard includes
#include <chrono>
#include <functional>
#include <optional>
#include <string_view>
#include <utility>

// platform includes
#include <Windows.h>
#include <winnt.h>

namespace platf {
  enum class explorer_restart_result_e {
    restarted_automatically,
    relaunched,
    skipped_no_shell,
    skipped_identity,
    preflight_failed,
    terminate_failed,
    exit_timeout,
    relaunch_failed,
    replacement_timeout,
  };

  namespace detail {
    /**
     * Run an input injection attempt with at most one retry after synchronizing the thread's
     * input desktop. This keeps a persistent injection failure from monopolizing the input worker.
     */
    template<class Attempt, class SynchronizeDesktop>
    bool run_with_desktop_retry(Attempt &&attempt, SynchronizeDesktop &&synchronize_desktop) {
      if (std::invoke(attempt)) {
        return true;
      }
      if (!std::invoke(synchronize_desktop)) {
        return false;
      }
      return std::invoke(attempt);
    }

    /**
     * Replace the handle that backs this thread's desktop association.
     *
     * A failed SetThreadDesktop leaves the previous association intact and closes only the
     * candidate. A successful switch retains the candidate (CloseDesktop must not be called on a
     * handle in use by a thread) and closes the superseded handle.
     */
    template<class Handle, class Operations>
    bool replace_thread_desktop_handle(Handle &active, Operations &operations) {
      const auto candidate = operations.open_input_desktop();
      if (!candidate) {
        return false;
      }
      if (!operations.set_thread_desktop(candidate)) {
        operations.close_desktop(candidate);
        return false;
      }
      if (active) {
        operations.close_desktop(active);
      }
      active = candidate;
      return true;
    }

    struct create_process_with_token_plan_t {
      DWORD creation_flags;
      DWORD startup_info_size;
      bool assign_job_after_create;
    };

    /**
     * CreateProcessWithTokenW does not reliably accept the process-thread job-list
     * attribute used by CreateProcessW/CreateProcessAsUserW, and it has no
     * bInheritHandles parameter required by PROC_THREAD_ATTRIBUTE_HANDLE_LIST. Launch with
     * plain STARTUPINFO instead, then assign a suspended process to the requested job before
     * any child code runs.
     */
    constexpr create_process_with_token_plan_t create_process_with_token_plan(
      const DWORD requested_flags,
      const bool has_job
    ) {
      return {
        .creation_flags =
          (requested_flags & ~EXTENDED_STARTUPINFO_PRESENT) |
          (has_job ? CREATE_SUSPENDED : 0),
        .startup_info_size = sizeof(STARTUPINFOW),
        .assign_job_after_create = has_job,
      };
    }

    /**
     * Finish a CreateProcessWithTokenW launch that was suspended for safe job assignment.
     * Operations is a small Win32 adapter in production and a deterministic fake in tests.
     */
    template<class Operations>
    DWORD finish_suspended_job_launch(Operations &operations) {
      DWORD failure = ERROR_SUCCESS;
      if (!operations.assign_to_job()) {
        failure = operations.last_error();
        if (failure == ERROR_SUCCESS) {
          failure = ERROR_PROCESS_ABORTED;
        }
      } else if (!operations.resume_initial_thread()) {
        failure = operations.last_error();
        if (failure == ERROR_SUCCESS) {
          failure = ERROR_PROCESS_ABORTED;
        }
      }
      if (failure == ERROR_SUCCESS) {
        return ERROR_SUCCESS;
      }

      // Assignment failure leaves the child suspended and uncontained. Resume failure leaves it
      // suspended inside the kill-on-close job. Terminate and release both handles in either case.
      operations.terminate_process(failure);
      operations.close_initial_thread();
      operations.close_process();
      return failure;
    }

    /**
     * Coordinate one bounded Explorer restart without embedding Win32 side effects in the policy.
     *
     * Production supplies exact-PID/token operations. Unit tests supply fakes, so no automated
     * test can accidentally terminate the developer's real shell.
     */
    template<class Operations>
    explorer_restart_result_e run_explorer_restart(Operations &operations) {
      if (const auto preflight_failure = operations.preflight()) {
        return *preflight_failure;
      }
      if (!operations.terminate_exact_shell()) {
        return explorer_restart_result_e::terminate_failed;
      }
      if (!operations.wait_for_old_shell_exit()) {
        return explorer_restart_result_e::exit_timeout;
      }
      if (operations.wait_for_replacement(false)) {
        return explorer_restart_result_e::restarted_automatically;
      }
      // AutoRestartShell can appear exactly as its grace period expires. Recheck immediately
      // before spawning so the fallback never becomes an unwanted File Explorer window.
      if (operations.replacement_present_now()) {
        return operations.wait_for_replacement(true) ?
                 explorer_restart_result_e::restarted_automatically :
                 explorer_restart_result_e::replacement_timeout;
      }
      if (!operations.launch_captured_shell()) {
        return explorer_restart_result_e::relaunch_failed;
      }
      return operations.wait_for_replacement(true) ?
               explorer_restart_result_e::relaunched :
               explorer_restart_result_e::replacement_timeout;
    }

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

  /**
   * Restart the exact Explorer shell belonging to the active interactive Windows user.
   *
   * This is a bounded, best-effort taskbar repair. It validates the shell's session, account,
   * and executable path before terminating only that PID, then gives AutoRestartShell a chance
   * to recover before launching one fallback with the captured shell token.
   */
  explorer_restart_result_e restart_active_user_explorer();

  void print_status(const std::string_view &prefix, HRESULT status);
  bool syncThreadDesktop();

  int64_t qpc_counter();

  int64_t qpc_frequency();

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
