/**
 * @file src/platform/windows/scheduling_win32.cpp
 * @brief Thread priority, presentation scheduling, streaming host settings, and timers on Windows.
 */
// standard includes
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

// prevent clang format from "optimizing" the header include order
// clang-format off
#include <Windows.h>
#include <dwmapi.h>
#include <timeapi.h>
#include <WinUser.h>
#include <wlanapi.h>
// clang-format on

// local includes
#include "misc.h"
#include "nvprefs/nvprefs_interface.h"
#include "presentation_scheduling.h"
#include "src/globals.h"
#include "src/logging.h"
#include "src/platform/common.h"

#ifndef WLAN_API_MAKE_VERSION
  #define WLAN_API_MAKE_VERSION(_major, _minor) (((DWORD) (_minor)) << 16 | (_major))
#endif

#include <winternl.h>
extern "C" {
  NTSTATUS NTAPI NtSetTimerResolution(ULONG DesiredResolution, BOOLEAN SetResolution, PULONG CurrentResolution);
}

namespace {

  std::mutex mouse_keys_mutex;
  platf::detail::mouse_keys_controller_t mouse_keys_controller;
  std::chrono::steady_clock::time_point mouse_keys_retry_after;

}  // namespace

using namespace std::literals;

namespace platf {
  HANDLE wlan_handle = nullptr;

  void adjust_thread_priority(thread_priority_e priority) {
    int win32_priority;

    switch (priority) {
      case thread_priority_e::low:
        win32_priority = THREAD_PRIORITY_BELOW_NORMAL;
        break;
      case thread_priority_e::normal:
        win32_priority = THREAD_PRIORITY_NORMAL;
        break;
      case thread_priority_e::high:
        win32_priority = THREAD_PRIORITY_ABOVE_NORMAL;
        break;
      case thread_priority_e::critical:
        win32_priority = THREAD_PRIORITY_HIGHEST;
        break;
      default:
        BOOST_LOG(error) << "Unknown thread priority: "sv << (int) priority;
        return;
    }

    if (!SetThreadPriority(GetCurrentThread(), win32_priority)) {
      auto winerr = GetLastError();
      BOOST_LOG(warning) << "Unable to set thread priority to "sv << win32_priority << ": "sv << winerr;
    }
  }

  namespace {
    // The driver profile object is shared by presentation start-up and the per-presenter
    // variable-refresh choice, which run on different threads.
    std::mutex driver_profile_mutex;

    struct presentation_cpu_scheduling_operations_t {
      detail::presentation_timer_request_t timer_request;
      DWORD previous_priority = 0;

      void start() {
        // Retain the exact successfully acquired timer request. A failed cleanup can be retried
        // after the next presentation lifetime without adding another outstanding request.
        timer_request.start(
          []() -> std::optional<std::uint32_t> {
            ULONG minimum = 0, maximum = 0, current = 0;
            if (NT_SUCCESS(NtQueryTimerResolution(&minimum, &maximum, &current)) && maximum != 0 && NT_SUCCESS(NtSetTimerResolution(maximum, TRUE, &current))) {
              return maximum;
            }
            return std::nullopt;
          },
          []() {
            BOOST_LOG(warning) << "NtSetTimerResolution() failed, falling back to timeBeginPeriod()."sv;
            const bool acquired = timeBeginPeriod(1) == TIMERR_NOERROR;
            if (!acquired) {
              BOOST_LOG(warning) << "Could not acquire a fine presentation timer period."sv;
            }
            return acquired;
          }
        );

        const DWORD priority = GetPriorityClass(GetCurrentProcess());
        // Preserve a caller-selected higher priority and only claim a change with a known undo.
        if (!previous_priority && priority != 0 && priority != HIGH_PRIORITY_CLASS && priority != REALTIME_PRIORITY_CLASS && SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
          previous_priority = priority;
        }
      }

      void stop() noexcept {
        if (previous_priority) {
          const DWORD current_priority = GetPriorityClass(GetCurrentProcess());
          if (current_priority != 0 && (current_priority != HIGH_PRIORITY_CLASS || SetPriorityClass(GetCurrentProcess(), previous_priority))) {
            // A later external priority change is not ours to overwrite.
            previous_priority = 0;
          }
        }

        timer_request.stop(
          [](const std::uint32_t native_resolution) {
            ULONG current = 0;
            // FALSE releases our request; requesting a slower period with TRUE does not release it.
            if (NT_SUCCESS(NtSetTimerResolution(native_resolution, FALSE, &current))) {
              return true;
            }
            BOOST_LOG(warning) << "Could not release the native presentation timer request."sv;
            return false;
          },
          []() {
            return timeEndPeriod(1) == TIMERR_NOERROR;
          }
        );
      }
    };

    using presentation_cpu_scheduling_controller_t =
      detail::presentation_scheduling_t<presentation_cpu_scheduling_operations_t>;
    // The full presentation controller below and test CPU leases share this exact owner.
    presentation_cpu_scheduling_controller_t presentation_cpu_scheduling;

    struct presentation_scheduling_operations_t {
      bool dwm_mmcss_enabled = false;
      presentation_cpu_scheduling_controller_t::lease_t cpu_scheduling;

      void start() {
        if (!dwm_mmcss_enabled) {
          dwm_mmcss_enabled = SUCCEEDED(DwmEnableMMCSS(TRUE));
        }
        cpu_scheduling = presentation_cpu_scheduling.acquire();

        // This first-owner transition is serialized with every local/remote acquire and release.
        // Startup/shutdown profile ownership surrounds both presentation lifetimes in main().
        std::lock_guard profile_lock(driver_profile_mutex);
        if (nvprefs_instance.load()) {
          if (!nvprefs_instance.owning_undo_file()) {
            nvprefs_instance.restore_from_and_delete_undo_file_if_exists();
          }
          nvprefs_instance.modify_application_profile();
          nvprefs_instance.modify_global_profile();
          nvprefs_instance.unload();
        }
      }

      void stop() noexcept {
        cpu_scheduling.reset();
        if (dwm_mmcss_enabled && SUCCEEDED(DwmEnableMMCSS(FALSE))) {
          dwm_mmcss_enabled = false;
        }
      }
    };

    using presentation_scheduling_controller_t =
      detail::presentation_scheduling_t<presentation_scheduling_operations_t>;
    // Declaration order keeps this controller alive through remote lease destruction.
    presentation_scheduling_controller_t presentation_scheduling;
    std::unique_ptr<deinit_t> remote_presentation_scheduling;

    class presentation_scheduling_lease_t final: public deinit_t {
    public:
      presentation_scheduling_lease_t():
          lease_(presentation_scheduling.acquire()) {
      }

    private:
      presentation_scheduling_controller_t::lease_t lease_;
    };
  }  // namespace

  std::unique_ptr<deinit_t> acquire_presentation_scheduling() {
    return std::make_unique<presentation_scheduling_lease_t>();
  }

  bool set_presentation_variable_refresh(bool allowed) {
    std::lock_guard profile_lock(driver_profile_mutex);
    nvprefs_instance.allow_variable_refresh(allowed);
    if (!nvprefs_instance.load()) {
      return false;
    }
    const bool written = nvprefs_instance.modify_application_profile();
    nvprefs_instance.unload();
    return written;
  }

  bool host_sbs_completion_wait_available() noexcept {
    try {
      return presentation_cpu_scheduling.active_satisfies([](const auto &cpu) {
        // This is eligibility for the measured notification transport, not a new timer policy.
        // Production still requests the OS-reported finest native period, or its existing
        // multimedia fallback. The fallback alone has no matching notification qualification.
        constexpr std::uint32_t maximum_period_100ns = 10'000u;
        if (!cpu.timer_request.owns_native_request_at_most(maximum_period_100ns)) {
          return false;
        }
        ULONG minimum = 0, maximum = 0, current = 0;
        if (!NT_SUCCESS(NtQueryTimerResolution(&minimum, &maximum, &current)) ||
            current == 0u || current > maximum_period_100ns) {
          return false;
        }
        const auto process_priority = GetPriorityClass(GetCurrentProcess());
        if (process_priority != HIGH_PRIORITY_CLASS && process_priority != REALTIME_PRIORITY_CLASS) {
          return false;
        }
        const auto owner_priority = GetThreadPriority(GetCurrentThread());
        return owner_priority != THREAD_PRIORITY_ERROR_RETURN && owner_priority >= THREAD_PRIORITY_ABOVE_NORMAL;
      });
    } catch (...) {
      return false;
    }
  }

#ifdef SUNSHINE_TESTS
  namespace testing {
    std::unique_ptr<deinit_t> acquire_presentation_cpu_scheduling_for_test() {
      class cpu_scheduling_lease_t final: public deinit_t {
      public:
        cpu_scheduling_lease_t():
            lease_(presentation_cpu_scheduling.acquire()) {
        }

      private:
        presentation_cpu_scheduling_controller_t::lease_t lease_;
      };
      return std::make_unique<cpu_scheduling_lease_t>();
    }
  }
#endif

  void streaming_will_start() {
    // stream.cpp serializes the remote hooks and retains this lease across reconnect grace.
    if (!remote_presentation_scheduling) {
      remote_presentation_scheduling = acquire_presentation_scheduling();
    }
    // Enable low latency mode on all connected WLAN NICs when the service is available.
    DWORD negotiated_version;
    if (WlanOpenHandle(WLAN_API_MAKE_VERSION(2, 0), nullptr, &negotiated_version, &wlan_handle) == ERROR_SUCCESS) {
      PWLAN_INTERFACE_INFO_LIST wlan_interface_list;

      if (WlanEnumInterfaces(wlan_handle, nullptr, &wlan_interface_list) == ERROR_SUCCESS) {
        for (DWORD i = 0; i < wlan_interface_list->dwNumberOfItems; i++) {
          if (wlan_interface_list->InterfaceInfo[i].isState == wlan_interface_state_connected) {
            // Enable media streaming mode for 802.11 wireless interfaces to reduce latency and
            // unnecessary background scanning operations that cause packet loss and jitter.
            //
            // https://docs.microsoft.com/en-us/windows-hardware/drivers/network/oid-wdi-set-connection-quality
            // https://docs.microsoft.com/en-us/previous-versions/windows/hardware/wireless/native-802-11-media-streaming
            BOOL value = TRUE;
            auto error = WlanSetInterface(wlan_handle, &wlan_interface_list->InterfaceInfo[i].InterfaceGuid, wlan_intf_opcode_media_streaming_mode, sizeof(value), &value, nullptr);
            if (error == ERROR_SUCCESS) {
              BOOST_LOG(info) << "WLAN interface "sv << i << " is now in low latency mode"sv;
            }
          }
        }

        WlanFreeMemory(wlan_interface_list);
      } else {
        WlanCloseHandle(wlan_handle, nullptr);
        wlan_handle = nullptr;
      }
    }

    {
      std::lock_guard lock(mouse_keys_mutex);
      mouse_keys_retry_after = {};
    }
    refresh_mouse_keys();
  }

  void refresh_mouse_keys() {
    std::lock_guard lock(mouse_keys_mutex);
    const auto now = std::chrono::steady_clock::now();
    if (now < mouse_keys_retry_after) {
      return;
    }

    bool operation_failed = false;

    const auto enabled = mouse_keys_controller.refresh(
      GetSystemMetrics(SM_MOUSEPRESENT) != 0,
      [&](MOUSEKEYS &state) {
        if (SystemParametersInfoW(SPI_GETMOUSEKEYS, sizeof(MOUSEKEYS), &state, 0)) {
          return true;
        }

        operation_failed = true;
        BOOST_LOG(warning) << "Unable to get current state of Mouse Keys: "sv << GetLastError();
        return false;
      },
      [&](MOUSEKEYS &state) {
        if (SystemParametersInfoW(SPI_SETMOUSEKEYS, sizeof(MOUSEKEYS), &state, 0)) {
          return true;
        }

        operation_failed = true;
        BOOST_LOG(warning) << "Unable to enable Mouse Keys: "sv << GetLastError();
        return false;
      }
    );

    mouse_keys_retry_after = operation_failed ? now + 30s : std::chrono::steady_clock::time_point {};

    if (enabled) {
      BOOST_LOG(info) << "A mouse was not detected. Sunshine 3D enabled Mouse Keys while streaming to keep the remote cursor visible."sv;
    }
  }

  void streaming_will_stop() {
    remote_presentation_scheduling.reset();

    // Closing our WLAN client handle will undo our optimizations
    if (wlan_handle != nullptr) {
      WlanCloseHandle(wlan_handle, nullptr);
      wlan_handle = nullptr;
    }

    // Restore Mouse Keys back to the previous settings if we turned it on. The controller keeps
    // ownership after a failure so a later stream stop can retry instead of losing the snapshot.
    std::lock_guard lock(mouse_keys_mutex);
    mouse_keys_controller.restore([](MOUSEKEYS &state) {
      if (SystemParametersInfoW(SPI_SETMOUSEKEYS, sizeof(MOUSEKEYS), &state, 0)) {
        return true;
      }

      BOOST_LOG(warning) << "Unable to restore original state of Mouse Keys: "sv << GetLastError();
      return false;
    });
  }

  int64_t qpc_counter() {
    LARGE_INTEGER performance_counter;
    if (QueryPerformanceCounter(&performance_counter)) {
      return performance_counter.QuadPart;
    }
    return 0;
  }

  int64_t qpc_frequency() {
    auto get_frequency = []() {
      LARGE_INTEGER frequency;
      frequency.QuadPart = 0;
      QueryPerformanceFrequency(&frequency);
      return frequency.QuadPart;
    };
    static const std::int64_t frequency = get_frequency();
    return frequency;
  }

  std::chrono::nanoseconds qpc_time_difference(int64_t performance_counter1, int64_t performance_counter2) {
    const auto frequency = qpc_frequency();
    if (frequency > 0) {
      const auto seconds = std::chrono::duration<long double> {
        static_cast<long double>(performance_counter1 - performance_counter2) /
        static_cast<long double>(frequency)
      };
      return std::chrono::duration_cast<std::chrono::nanoseconds>(seconds);
    }
    return {};
  }

  class win32_high_precision_timer: public high_precision_timer {
  public:
    win32_high_precision_timer() {
      timer = CreateWaitableTimerEx(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
      if (!timer) {
        BOOST_LOG(error) << "Unable to create high_precision_timer, CreateWaitableTimerEx() failed: " << GetLastError();
      }
    }

    ~win32_high_precision_timer() {
      if (timer) {
        CloseHandle(timer);
      }
    }

    void sleep_for(const std::chrono::nanoseconds &duration) override {
      if (!timer) {
        BOOST_LOG(error) << "Attempting high_precision_timer::sleep_for() with uninitialized timer";
        return;
      }
      if (duration < 0s) {
        BOOST_LOG(error) << "Attempting high_precision_timer::sleep_for() with negative duration";
        return;
      }
      if (duration > 5s) {
        BOOST_LOG(error) << "Attempting high_precision_timer::sleep_for() with unexpectedly large duration (>5s)";
        return;
      }

      LARGE_INTEGER due_time;
      due_time.QuadPart = duration.count() / -100;
      SetWaitableTimer(timer, &due_time, 0, nullptr, nullptr, false);
      WaitForSingleObject(timer, INFINITE);
    }

    operator bool() override {
      return timer != nullptr;
    }

  private:
    HANDLE timer = nullptr;
  };

  std::unique_ptr<high_precision_timer> create_high_precision_timer() {
    return std::make_unique<win32_high_precision_timer>();
  }

  namespace {
    /** safe::deadline_waiter_t: the waiting thread waits for its notification event or its
     * high-resolution timer, which each wait arms at its deadline. */
    class win32_deadline_waiter_t: public safe::deadline_waiter_t {
    public:
      win32_deadline_waiter_t(HANDLE timer, HANDLE notified) noexcept:
          timer_ {timer},
          notified_ {notified} {
      }

      ~win32_deadline_waiter_t() override {
        CloseHandle(timer_);
        CloseHandle(notified_);
      }

      win32_deadline_waiter_t(const win32_deadline_waiter_t &) = delete;
      win32_deadline_waiter_t &operator=(const win32_deadline_waiter_t &) = delete;

      void notify() noexcept override {
        SetEvent(notified_);
      }

      bool wait_until(std::chrono::steady_clock::time_point deadline) noexcept override {
        const auto left = deadline - std::chrono::steady_clock::now();
        if (left <= std::chrono::steady_clock::duration::zero()) {
          return true;
        }
        // Relative, in whole 100 ns units rounded up: the timer never ends before the deadline.
        // Setting the timer also clears a signal left from an earlier deadline.
        using timer_unit_t = std::chrono::duration<LONGLONG, std::ratio<1, 10'000'000>>;
        LARGE_INTEGER due;
        due.QuadPart = -std::chrono::ceil<timer_unit_t>(left).count();
        if (!SetWaitableTimer(timer_, &due, 0, nullptr, nullptr, FALSE)) {
          return false;
        }
        const HANDLE handles[] {notified_, timer_};
        const auto result = WaitForMultipleObjects(2, handles, FALSE, INFINITE);
        if (result == WAIT_OBJECT_0) {
          // Notified first: no expiry is left to interrupt anything after this wait.
          CancelWaitableTimer(timer_);
          return true;
        }
        return result == WAIT_OBJECT_0 + 1;
      }

    private:
      HANDLE timer_;
      HANDLE notified_;
    };
  }  // namespace

  std::unique_ptr<safe::deadline_waiter_t> create_deadline_waiter() {
    // Without a high-resolution timer a waitable timer ends at the timer resolution, not its deadline.
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_MODIFY_STATE | SYNCHRONIZE);
    if (!timer) {
      BOOST_LOG(error) << "Unable to create a high-resolution deadline timer, CreateWaitableTimerExW() failed: "sv << GetLastError();
      return nullptr;
    }
    HANDLE notified = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!notified) {
      BOOST_LOG(error) << "Unable to create a deadline waiter's event, CreateEventW() failed: "sv << GetLastError();
      CloseHandle(timer);
      return nullptr;
    }
    return std::make_unique<win32_deadline_waiter_t>(timer, notified);
  }
}  // namespace platf
