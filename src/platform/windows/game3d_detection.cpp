// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_detection.h"

#include <algorithm>
#include <array>
#include <string_view>
#include <tlhelp32.h>
#include <vector>

namespace platf::game3d {
  namespace {
    class handle_t {
    public:
      explicit handle_t(HANDLE value):
          value_(value) {}

      ~handle_t() {
        if (value_ && value_ != INVALID_HANDLE_VALUE) {
          CloseHandle(value_);
        }
      }

      handle_t(const handle_t &) = delete;
      handle_t &operator=(const handle_t &) = delete;

      [[nodiscard]] HANDLE get() const noexcept {
        return value_;
      }

    private:
      HANDLE value_;
    };

    [[nodiscard]] bool equal_name(std::wstring_view left, std::wstring_view right) noexcept {
      return left.size() == right.size() &&
             CompareStringOrdinal(left.data(), static_cast<int>(left.size()), right.data(), static_cast<int>(right.size()), TRUE) == CSTR_EQUAL;
    }

    [[nodiscard]] bool excluded_executable(const std::filesystem::path &executable) {
      // Browser and launcher UI can load both D3D and gaming SDKs. Editors can load the same
      // engine libraries as games. Their foreground windows must never prompt for game setup.
      constexpr std::array excluded {
        L"chrome.exe",
        L"msedge.exe",
        L"firefox.exe",
        L"brave.exe",
        L"opera.exe",
        L"vivaldi.exe",
        L"arc.exe",
        L"steam.exe",
        L"steamwebhelper.exe",
        L"EpicGamesLauncher.exe",
        L"EpicWebHelper.exe",
        L"GalaxyClient.exe",
        L"GalaxyClient Helper.exe",
        L"UbisoftConnect.exe",
        L"upc.exe",
        L"Uplay.exe",
        L"EADesktop.exe",
        L"EALauncher.exe",
        L"Origin.exe",
        L"Battle.net.exe",
        L"Battle.net Helper.exe",
        L"XboxPcApp.exe",
        L"GamingServicesUI.exe",
        L"Playnite.DesktopApp.exe",
        L"Playnite.FullscreenApp.exe",
        L"explorer.exe",
        L"ApplicationFrameHost.exe",
        L"ShellExperienceHost.exe",
        L"StartMenuExperienceHost.exe",
        L"SearchHost.exe",
        L"dwm.exe",
        L"LockApp.exe",
        L"SystemSettings.exe",
        L"Taskmgr.exe",
        L"sunshine.exe",
        L"moonlight.exe",
        L"Discord.exe",
        L"obs64.exe",
        L"vlc.exe",
        L"mpv.exe",
        L"Unity.exe",
        L"Unity Hub.exe",
        L"UnrealEditor.exe",
        L"UE4Editor.exe",
        L"blender.exe",
      };
      const auto name = executable.filename().wstring();
      return std::any_of(excluded.begin(), excluded.end(), [&](const auto candidate) {
        return equal_name(name, candidate);
      });
    }

    [[nodiscard]] bool unreal_shipping_layout(const std::filesystem::path &executable) {
      // The engine's Binaries/Win64 layout and Shipping target identify the actual game rather
      // than its launcher without maintaining a list of individual games.
      const auto parent = executable.parent_path();
      const auto stem = executable.stem().wstring();
      constexpr std::wstring_view suffix = L"-Win64-Shipping";
      return equal_name(parent.filename().wstring(), L"Win64") &&
             equal_name(parent.parent_path().filename().wstring(), L"Binaries") &&
             stem.size() > suffix.size() && equal_name(std::wstring_view {stem}.substr(stem.size() - suffix.size()), suffix);
    }

    [[nodiscard]] std::uint64_t process_creation_time(HANDLE process) noexcept {
      FILETIME created {}, exited {}, kernel {}, user {};
      if (!GetProcessTimes(process, &created, &exited, &kernel, &user)) {
        return 0;
      }
      return (static_cast<std::uint64_t>(created.dwHighDateTime) << 32) | created.dwLowDateTime;
    }

    [[nodiscard]] std::optional<std::filesystem::path> process_executable(HANDLE process) {
      std::array<wchar_t, 32768> buffer {};
      DWORD length = static_cast<DWORD>(buffer.size());
      if (!QueryFullProcessImageNameW(process, 0, buffer.data(), &length) || !length) {
        return std::nullopt;
      }
      return std::filesystem::path {std::wstring {buffer.data(), length}};
    }

    [[nodiscard]] bool in_windows_directory(const std::filesystem::path &executable) {
      std::array<wchar_t, MAX_PATH + 1> buffer {};
      const auto length = GetWindowsDirectoryW(buffer.data(), static_cast<UINT>(buffer.size()));
      if (!length || length >= buffer.size()) {
        return true;
      }
      const auto path = executable.wstring();
      return path.size() > length && (path[length] == L'\\' || path[length] == L'/') &&
             equal_name(std::wstring_view {path}.substr(0, length), std::wstring_view {buffer.data(), length});
    }

    [[nodiscard]] std::optional<std::vector<std::filesystem::path>> process_modules(DWORD process_id) {
      HANDLE snapshot = INVALID_HANDLE_VALUE;
      // The module list may change during game startup. ERROR_BAD_LENGTH explicitly permits a
      // retry; bound it so a busy loader cannot monopolize the setup worker.
      for (int attempt = 0; attempt < 3; ++attempt) {
        snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, process_id);
        if (snapshot != INVALID_HANDLE_VALUE || GetLastError() != ERROR_BAD_LENGTH) {
          break;
        }
      }
      if (snapshot == INVALID_HANDLE_VALUE) {
        return std::nullopt;
      }
      const handle_t snapshot_guard {snapshot};
      MODULEENTRY32W module {};
      module.dwSize = sizeof(module);
      if (!Module32FirstW(snapshot, &module)) {
        return std::nullopt;
      }
      std::vector<std::filesystem::path> result;
      do {
        result.emplace_back(module.szModule);
        if (result.size() > 4096) {
          return std::nullopt;
        }
      } while (Module32NextW(snapshot, &module));
      if (GetLastError() != ERROR_NO_MORE_FILES) {
        return std::nullopt;
      }
      return result;
    }

    [[nodiscard]] std::string display_name(const std::filesystem::path &executable) {
      const auto name = executable.stem().wstring();
      const auto length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, name.data(), static_cast<int>(name.size()), nullptr, 0, nullptr, nullptr);
      if (length <= 0) {
        return "Game";
      }
      std::string result(static_cast<std::size_t>(length), '\0');
      if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, name.data(), static_cast<int>(name.size()), result.data(), length, nullptr, nullptr)) {
        return "Game";
      }
      return result;
    }
  }  // namespace

  std::optional<detection_evidence_e> detail::classify_game_candidate(
    const std::filesystem::path &executable,
    const std::span<const std::filesystem::path> loaded_modules,
    const foreground_window::observation_t &foreground,
    const RECT capture_rect
  ) {
    if (!foreground_window::carries_geometry(foreground.status) || !foreground.window || !foreground.process_id || !executable.is_absolute() || !equal_name(executable.extension().wstring(), L".exe") || excluded_executable(executable)) {
      return std::nullopt;
    }
    const auto client = foreground.client_screen_rect;
    if (!client.valid() || capture_rect.right <= capture_rect.left || capture_rect.bottom <= capture_rect.top) {
      return std::nullopt;
    }
    const auto width = static_cast<std::int64_t>(client.right) - client.left;
    const auto height = static_cast<std::int64_t>(client.bottom) - client.top;
    const auto overlap_width = (std::min<std::int64_t>) (client.right, capture_rect.right) - (std::max<std::int64_t>) (client.left, capture_rect.left);
    const auto overlap_height = (std::min<std::int64_t>) (client.bottom, capture_rect.bottom) - (std::max<std::int64_t>) (client.top, capture_rect.top);
    // Admit ordinary windowed games, but not tiny helper windows or an application mostly on a
    // different display. This geometry is setup evidence only; stereo export keeps its own
    // stricter fullscreen ownership contract.
    if (width < 320 || height < 180 || overlap_width <= 0 || overlap_height <= 0 || static_cast<long double>(overlap_width) * overlap_height < static_cast<long double>(width) * height * 0.75L) {
      return std::nullopt;
    }
    bool direct3d = false;
    bool game_runtime = unreal_shipping_layout(executable);
    constexpr std::array game_modules {
      L"steam_api64.dll",
      L"UnityPlayer.dll",
      L"GameAssembly.dll",
      L"Galaxy64.dll",
      L"EOSSDK-Win64-Shipping.dll",
      L"uplay_r1_loader64.dll",
      L"upc_r1_loader64.dll",
    };
    for (const auto &module : loaded_modules) {
      const auto name = module.filename().wstring();
      direct3d = direct3d || equal_name(name, L"d3d11.dll") || equal_name(name, L"d3d12.dll");
      game_runtime = game_runtime || std::any_of(game_modules.begin(), game_modules.end(), [&](const auto candidate) {
                       return equal_name(name, candidate);
                     });
    }
    if (!direct3d) {
      return std::nullopt;
    }
    if (game_runtime) {
      return detection_evidence_e::known_runtime;
    }
    // Engines and storefront SDKs change independently of the host. A foreground Direct3D
    // client filling this exact output is general game evidence even without a known module.
    // Use the client, not the outer frame: ordinary maximized application chrome and windows
    // spanning multiple outputs must not become automatic installation targets this way.
    if (client.left == capture_rect.left && client.top == capture_rect.top && client.right == capture_rect.right && client.bottom == capture_rect.bottom) {
      return detection_evidence_e::fullscreen_direct3d;
    }
    return std::nullopt;
  }

  bool detail::is_game_candidate(
    const std::filesystem::path &executable,
    const std::span<const std::filesystem::path> loaded_modules,
    const foreground_window::observation_t &foreground,
    const RECT capture_rect
  ) {
    return classify_game_candidate(executable, loaded_modules, foreground, capture_rect).has_value();
  }

  std::optional<detected_game> detection_confirmation::observe(
    std::optional<detected_game> candidate,
    const RECT capture_rect,
    const std::chrono::steady_clock::time_point now
  ) {
    using namespace std::chrono_literals;
    if (!candidate) {
      reset();
      return std::nullopt;
    }
    if (candidate->evidence == detection_evidence_e::known_runtime) {
      reset();
      return candidate;
    }
    if (!candidate->process_id || !candidate->creation_time || !candidate->window || capture_rect.right <= capture_rect.left || capture_rect.bottom <= capture_rect.top) {
      reset();
      return std::nullopt;
    }
    const bool same_capture = capture_rect.left == capture_rect_.left && capture_rect.top == capture_rect_.top && capture_rect.right == capture_rect_.right && capture_rect.bottom == capture_rect_.bottom;
    const bool same_process = provisional_ && provisional_->process_id == candidate->process_id && provisional_->creation_time == candidate->creation_time && provisional_->window == candidate->window && equal_name(provisional_->executable.wstring(), candidate->executable.wstring());
    if (!same_capture || !same_process || now < last_seen_ || now - last_seen_ > 5s) {
      provisional_ = candidate;
      capture_rect_ = capture_rect;
      first_seen_ = now;
      last_seen_ = now;
      return std::nullopt;
    }
    last_seen_ = now;
    return now - first_seen_ >= 2s ? candidate : std::nullopt;
  }

  void detection_confirmation::reset() noexcept {
    provisional_.reset();
    capture_rect_ = {};
    first_seen_ = {};
    last_seen_ = {};
  }

  std::optional<detected_game> detect_game(const RECT capture_rect) noexcept {
    try {
      const auto foreground = foreground_window::sample();
      if (!foreground_window::carries_geometry(foreground.status) || !foreground.process_id) {
        return std::nullopt;
      }
      const handle_t process {OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, foreground.process_id)};
      if (!process.get() || WaitForSingleObject(process.get(), 0) != WAIT_TIMEOUT) {
        return std::nullopt;
      }
      USHORT process_machine {}, native_machine {};
      if (!IsWow64Process2(process.get(), &process_machine, &native_machine) || process_machine != IMAGE_FILE_MACHINE_UNKNOWN || native_machine != IMAGE_FILE_MACHINE_AMD64) {
        return std::nullopt;
      }
      const auto creation_time = process_creation_time(process.get());
      const auto executable = process_executable(process.get());
      if (!creation_time || !executable || in_windows_directory(*executable) || excluded_executable(*executable)) {
        return std::nullopt;
      }
      const auto modules = process_modules(foreground.process_id);
      if (!modules) {
        return std::nullopt;
      }
      const auto evidence = detail::classify_game_candidate(*executable, *modules, foreground, capture_rect);
      if (!evidence) {
        return std::nullopt;
      }
      const auto after = foreground_window::sample();
      const auto executable_after = process_executable(process.get());
      if (!foreground_window::carries_geometry(after.status) || after.window != foreground.window || after.process_id != foreground.process_id || after.client_screen_rect != foreground.client_screen_rect || !executable_after || !equal_name(executable->wstring(), executable_after->wstring()) || process_creation_time(process.get()) != creation_time || WaitForSingleObject(process.get(), 0) != WAIT_TIMEOUT) {
        return std::nullopt;
      }
      return detected_game {*executable, foreground.process_id, creation_time, display_name(*executable), *evidence, foreground.window};
    } catch (...) {
      // This advisory must not break an active stream on a path conversion/allocation failure.
      return std::nullopt;
    }
  }
}  // namespace platf::game3d
