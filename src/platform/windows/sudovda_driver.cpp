/**
 * @file src/platform/windows/sudovda_driver.cpp
 * @brief SudoVDA driver handle, watchdog heartbeat, and virtual display add/remove IOCTLs.
 */
#include "virtual_display.h"

#include "display_config.h"
#include "misc.h"
#include "src/logging.h"
#include "src/platform/common.h"

#include <algorithm>
#include <chrono>
#include <combaseapi.h>
#include <condition_variable>
#include <cstdint>
#include <dxgi.h>
#include <dxgi1_6.h>
#include <initguid.h>
#include <mutex>
#include <optional>
#include <thread>
#include <wrl/client.h>

using namespace SUDOVDA;

namespace VDISPLAY {
  namespace {
    HANDLE sudovdaDriverHandle = INVALID_HANDLE_VALUE;
    std::mutex virtualDisplayMutationMutex;
    std::mutex driverLifecycleMutex;
    std::mutex watchdogOwnerMutex;
    std::mutex watchdogWaitMutex;
    std::condition_variable_any watchdogWake;
    std::jthread watchdogThread;

    std::chrono::milliseconds watchdogPingInterval(uint32_t timeoutSeconds) {
      const auto thirdOfTimeout = std::chrono::milliseconds(
        static_cast<uint64_t>(timeoutSeconds) * 1000 / 3
      );
      // A longer driver lease protects the monitor from debugger and GPU-startup stalls, but the
      // heartbeat itself should remain frequent. This also detects a dead driver promptly.
      return std::min(thirdOfTimeout, std::chrono::milliseconds(1000));
    }

    template<typename Query, typename Now, typename WaitUntil>
    display_identity_query_t waitForDisplayIdentity(
      const LUID &adapterLuid,
      uint32_t targetId,
      const Query &query,
      const Now &now,
      const WaitUntil &waitUntil
    ) {
      // Preserve the original 20+40+80+160+320+640 ms wait budget, but observe publication after
      // every wait, including the final one. The deadline bounds waits, not Windows API latency.
      const auto deadline = now() + std::chrono::milliseconds(1260);
      auto retryInterval = std::chrono::milliseconds(20);
      for (;;) {
        auto published = query(adapterLuid, targetId);
        if (published.state == display_identity_state_e::present) {
          return published;
        }
        const auto observedAt = now();
        if (observedAt >= deadline) {
          return published;
        }
        waitUntil(std::min(observedAt + retryInterval, deadline));
        retryInterval *= 2;
      }
    }

    void stopWatchdogThread() {
      std::jthread retiring;
      {
        std::lock_guard lock(watchdogOwnerMutex);
        watchdogThread.request_stop();
        watchdogWake.notify_all();
        retiring = std::move(watchdogThread);
      }
      if (retiring.joinable()) {
        retiring.join();
      }
    }

    void closeDriverHandleLocked() {
      if (sudovdaDriverHandle != INVALID_HANDLE_VALUE) {
        CloseHandle(sudovdaDriverHandle);
        sudovdaDriverHandle = INVALID_HANDLE_VALUE;
      }
    }

    std::optional<LUID> adapterLuidByName(const std::wstring &adapterName) {
      Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
      if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        return std::nullopt;
      }

      for (UINT index = 0;; ++index) {
        Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
        if (factory->EnumAdapters(index, &adapter) == DXGI_ERROR_NOT_FOUND) {
          break;
        }

        DXGI_ADAPTER_DESC desc {};
        if (SUCCEEDED(adapter->GetDesc(&desc)) && std::wstring_view(desc.Description) == adapterName) {
          return desc.AdapterLuid;
        }
      }
      return std::nullopt;
    }
  }  // namespace

void closeVDisplayDevice() {
  std::lock_guard lifecycle_lock(driverLifecycleMutex);
  stopWatchdogThread();
  std::lock_guard device_lock(virtualDisplayMutationMutex);
  closeDriverHandleLocked();
}

DRIVER_STATUS openVDisplayDevice() {
  std::lock_guard lifecycle_lock(driverLifecycleMutex);
  stopWatchdogThread();
  std::lock_guard device_lock(virtualDisplayMutationMutex);
  closeDriverHandleLocked();

  uint32_t retryInterval = 20;
  while (true) {
    sudovdaDriverHandle = OpenDevice(&SUVDA_INTERFACE_GUID);
    if (sudovdaDriverHandle == INVALID_HANDLE_VALUE) {
      if (retryInterval > 320) {
        BOOST_LOG(error) << "[SUDOVDA] Open device failed!";
        return DRIVER_STATUS::FAILED;
      }
      retryInterval *= 2;
      Sleep(retryInterval);
      continue;
    }
    break;
  }

  if (!CheckProtocolCompatible(sudovdaDriverHandle)) {
    BOOST_LOG(error) << "[SUDOVDA] SUDOVDA protocol not compatible with driver!";
    closeDriverHandleLocked();
    return DRIVER_STATUS::VERSION_INCOMPATIBLE;
  }
  return DRIVER_STATUS::OK;
}

bool startPingThread(std::function<void()> failCb) {
  std::lock_guard lifecycle_lock(driverLifecycleMutex);
  stopWatchdogThread();

  // Keep the lifetime heartbeat independent of Add/RemoveVirtualDisplay. A separate shared handle
  // avoids delaying a ping behind the process-wide mutation lock during topology transitions.
  const auto watchdogHandle = OpenDevice(&SUVDA_INTERFACE_GUID);
  if (watchdogHandle == INVALID_HANDLE_VALUE) {
    BOOST_LOG(error) << "[SUDOVDA] Failed to open dedicated watchdog heartbeat handle.";
    return false;
  }

  VIRTUAL_DISPLAY_GET_WATCHDOG_OUT watchdogOut {};
  if (!GetWatchdogTimeout(watchdogHandle, watchdogOut)) {
    BOOST_LOG(error) << "[SUDOVDA] Watchdog fetch failed!";
    CloseHandle(watchdogHandle);
    return false;
  }
  BOOST_LOG(info) << "[SUDOVDA] Watchdog: Timeout " << watchdogOut.Timeout << ", Countdown " << watchdogOut.Countdown;
  if (!watchdogOut.Timeout) {
    CloseHandle(watchdogHandle);
    return true;
  }
  if (watchdogOut.Timeout < 10) {
    BOOST_LOG(warning) << "[SUDOVDA] Watchdog timeout is only " << watchdogOut.Timeout
                       << " seconds. Install or configure Sunshine 3D's recommended 30-second timeout to prevent "
                          "the retained virtual display from being removed during GPU initialization.";
  }

  const auto sleepInterval = watchdogPingInterval(watchdogOut.Timeout);
  std::lock_guard owner_lock(watchdogOwnerMutex);
  try {
    watchdogThread = std::jthread([watchdogHandle, sleepInterval, failCb = std::move(failCb)](std::stop_token stop_token) {
      // This heartbeat is the lifetime lease for every SudoVDA monitor. Favor it over ordinary
      // application work so a busy encoder startup cannot let the driver remove an active desktop.
      if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST)) {
        BOOST_LOG(warning) << "[SUDOVDA] Failed to raise watchdog heartbeat thread priority: " << GetLastError();
      }

      uint8_t fail_count = 0;
      while (!stop_token.stop_requested()) {
        bool watchdog_failed = false;
        if (PingDriver(watchdogHandle)) {
          fail_count = 0;
        } else if (++fail_count > 3) {
          {
            std::lock_guard device_lock(virtualDisplayMutationMutex);
            closeDriverHandleLocked();
          }
          watchdog_failed = true;
        }
        if (watchdog_failed) {
          failCb();
          break;
        }

        std::unique_lock wait_lock(watchdogWaitMutex);
        watchdogWake.wait_for(wait_lock, stop_token, sleepInterval, []() {
          return false;
        });
      }
      CloseHandle(watchdogHandle);
    });
  } catch (...) {
    CloseHandle(watchdogHandle);
    BOOST_LOG(error) << "[SUDOVDA] Failed to start watchdog heartbeat thread.";
    return false;
  }
  return true;
}

#ifdef SUNSHINE_TESTS
uint32_t watchdogPingIntervalMsForTest(uint32_t timeoutSeconds) {
  return static_cast<uint32_t>(watchdogPingInterval(timeoutSeconds).count());
}

display_identity_query_t waitForDisplayIdentityForTest(
  const LUID &adapterLuid,
  uint32_t targetId,
  const std::function<display_identity_query_t(const LUID &, uint32_t)> &query,
  const std::function<std::chrono::steady_clock::time_point()> &now,
  const std::function<void(std::chrono::steady_clock::time_point)> &waitUntil
) {
  return waitForDisplayIdentity(adapterLuid, targetId, query, now, waitUntil);
}
#endif

namespace {
display_identity_query_t queryDisplayIdentity(const LUID &adapterLuid, uint32_t targetId) {
  platf::display_config::topology_t topology;
  if (!platf::display_config::query_topology(QDC_ONLY_ACTIVE_PATHS, topology)) {
    return {};
  }

  for (const auto &path : topology.paths) {
    if (path.targetInfo.id != targetId || !platf::display_config::same_luid(path.targetInfo.adapterId, adapterLuid)) {
      continue;
    }

    auto sourceName = platf::display_config::query_source_name(path.sourceInfo.adapterId, path.sourceInfo.id);
    auto targetName = platf::display_config::query_target_name(path.targetInfo.adapterId, path.targetInfo.id);
    if (sourceName && targetName && !sourceName->empty()) {
      return {
        display_identity_state_e::present,
        std::move(*sourceName),
        std::move(targetName.device_path),
        std::move(targetName.friendly_name),
      };
    }

    // The exact stable adapter/target identity is still in the active topology, but Windows could
    // not publish its GDI source name. Do not let callers mistake that transient query failure for
    // completed removal.
    return {};
  }

  // Absence is authoritative only after a complete active-topology snapshot contains no exact
  // adapter/target match.
  return {display_identity_state_e::absent, {}, {}, {}};
}
}  // namespace

namespace {
  std::optional<LUID> primaryDisplayAdapterLuid() {
    std::wstring primaryDisplay;
    for (DWORD index = 0;; ++index) {
      DISPLAY_DEVICEW device {};
      device.cb = sizeof(device);
      if (!EnumDisplayDevicesW(nullptr, index, &device, 0)) {
        break;
      }
      if ((device.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE) != 0) {
        primaryDisplay = device.DeviceName;
        break;
      }
    }
    if (primaryDisplay.empty()) {
      return std::nullopt;
    }

    platf::display_config::topology_t topology;
    if (!platf::display_config::query_topology(QDC_ONLY_ACTIVE_PATHS, topology)) {
      return std::nullopt;
    }
    for (const auto &path : topology.paths) {
      const auto sourceName = platf::display_config::query_source_name(path.sourceInfo.adapterId, path.sourceInfo.id);
      if (sourceName && platf::display_config::same_device_path(*sourceName, primaryDisplay)) {
        return path.sourceInfo.adapterId;
      }
    }
    return std::nullopt;
  }

  creation_result_t createVirtualDisplayImpl(
    const char *s_client_uid,
    const char *s_client_name,
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    const GUID &guid,
    const std::optional<LUID> &adapterLuid
  ) {
    creation_result_t result;
    result.render_adapter_luid = adapterLuid;
    VIRTUAL_DISPLAY_ADD_OUT output {};
    {
      // SudoVDA's render-adapter choice is process-global. Keep adapter selection and AddVirtualDisplay
      // in one critical section so local AR and remote streaming sessions cannot steal each other's GPU.
      std::lock_guard lock(virtualDisplayMutationMutex);
      if (sudovdaDriverHandle == INVALID_HANDLE_VALUE) {
        return {};
      }
      if (adapterLuid && !SetRenderAdapter(sudovdaDriverHandle, *adapterLuid)) {
        BOOST_LOG(error) << "[SUDOVDA] Failed to select render adapter for virtual display.";
        return {};
      }
      if (!AddVirtualDisplay(sudovdaDriverHandle, width, height, fps, guid, s_client_name, s_client_uid, output)) {
        BOOST_LOG(error) << "[SUDOVDA] Failed to add virtual display.";
        return {};
      }
    }
    // Driver creation and Windows display-name publication are separate operations. Record Add's
    // success immediately so callers can always retire the exact output even if name lookup times
    // out during topology churn.
    result.identity = output;

    auto published = waitForDisplayIdentity(
      output.AdapterLuid,
      output.TargetId,
      queryDisplayIdentity,
      std::chrono::steady_clock::now,
      [](std::chrono::steady_clock::time_point deadline) {
        std::this_thread::sleep_until(deadline);
      }
    );
    if (published.state != display_identity_state_e::present) {
      BOOST_LOG(error) << "[SUDOVDA] Cannot get name for newly added virtual display!";
      return result;
    }

    BOOST_LOG(info) << "[SUDOVDA] Virtual display added successfully: " << platf::to_utf8(published.display_name);
    BOOST_LOG(info) << "[SUDOVDA] Configuration: W: " << width << ", H: " << height << ", FPS: " << fps;

    result.display_name = std::move(published.display_name);
    result.device_path = std::move(published.device_path);
    result.friendly_name = std::move(published.friendly_name);
    return result;
  }
}  // namespace

creation_result_t createVirtualDisplay(
  const char *s_client_uid,
  const char *s_client_name,
  uint32_t width,
  uint32_t height,
  uint32_t fps,
  const GUID &guid,
  const std::optional<LUID> &adapterLuid,
  const std::wstring &adapterName
) {
  std::optional<LUID> renderAdapter = adapterLuid;
  if (!renderAdapter && !adapterName.empty()) {
    renderAdapter = adapterLuidByName(adapterName);
    if (!renderAdapter) {
      BOOST_LOG(error) << "[SUDOVDA] Cannot find requested render adapter.";
      return {};
    }
  }
  return createVirtualDisplayImpl(
    s_client_uid,
    s_client_name,
    width,
    height,
    fps,
    guid,
    renderAdapter ? renderAdapter : primaryDisplayAdapterLuid()
  );
}

bool removeVirtualDisplay(const GUID &guid) {
  std::lock_guard lock(virtualDisplayMutationMutex);
  if (sudovdaDriverHandle == INVALID_HANDLE_VALUE) {
    return false;
  }

  if (RemoveVirtualDisplay(sudovdaDriverHandle, guid)) {
    BOOST_LOG(info) << "[SUDOVDA] Virtual display removed successfully.";
    return true;
  } else {
    return false;
  }
}

}  // namespace VDISPLAY
