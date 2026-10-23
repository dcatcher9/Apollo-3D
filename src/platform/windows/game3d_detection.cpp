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
      // Non-game applications that commonly present a fullscreen Direct3D client: browsers and
      // their embedded runtimes, store launchers, media players, presentation/office and creative
      // tools, editors, communication, capture/overlay and VR compositors, and the shell. Setup
      // never writes without confirmation; this list only keeps them from being offered.
      constexpr std::array excluded {
        // Browsers and embedded web runtimes.
        L"chrome.exe",
        L"msedge.exe",
        L"msedgewebview2.exe",
        L"firefox.exe",
        L"librewolf.exe",
        L"waterfox.exe",
        L"floorp.exe",
        L"zen.exe",
        L"brave.exe",
        L"opera.exe",
        L"vivaldi.exe",
        L"thorium.exe",
        L"arc.exe",
        // Store and game launchers.
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
        L"RiotClientServices.exe",
        L"RiotClientUx.exe",
        L"RockstarLauncher.exe",
        L"Amazon Games UI.exe",
        L"itch.exe",
        L"Heroic.exe",
        L"XboxPcApp.exe",
        L"GamingServicesUI.exe",
        L"Playnite.DesktopApp.exe",
        L"Playnite.FullscreenApp.exe",
        // Media players.
        L"vlc.exe",
        L"mpv.exe",
        L"mpvnet.exe",
        L"kodi.exe",
        L"mpc-hc64.exe",
        L"mpc-be64.exe",
        L"mpc-qt.exe",
        L"PotPlayerMini64.exe",
        L"PotPlayer64.exe",
        L"KMPlayer64.exe",
        L"smplayer.exe",
        L"wmplayer.exe",
        L"Microsoft.Media.Player.exe",
        L"Video.UI.exe",
        L"Plex.exe",
        L"Plex HTPC.exe",
        L"JellyfinMediaPlayer.exe",
        L"stremio.exe",
        L"Spotify.exe",
        // Office, presentation and creative tools.
        L"POWERPNT.EXE",
        L"WINWORD.EXE",
        L"EXCEL.EXE",
        L"OUTLOOK.EXE",
        L"olk.exe",
        L"ONENOTE.EXE",
        L"MSACCESS.EXE",
        L"VISIO.EXE",
        L"Photoshop.exe",
        L"AfterFX.exe",
        L"Adobe Premiere Pro.exe",
        L"Resolve.exe",
        L"blender.exe",
        // Engine editors and IDEs.
        L"Unity.exe",
        L"Unity Hub.exe",
        L"UnrealEditor.exe",
        L"UE4Editor.exe",
        L"Code.exe",
        L"devenv.exe",
        L"rider64.exe",
        L"idea64.exe",
        L"clion64.exe",
        // Communication, capture, overlays, streaming and VR compositors.
        L"Discord.exe",
        L"Teams.exe",
        L"ms-teams.exe",
        L"slack.exe",
        L"Zoom.exe",
        L"obs64.exe",
        L"Streamlabs OBS.exe",
        L"NVIDIA app.exe",
        L"NVIDIA Overlay.exe",
        L"NVIDIA Share.exe",
        L"sunshine.exe",
        L"moonlight.exe",
        L"vrcompositor.exe",
        L"vrmonitor.exe",
        L"OculusClient.exe",
        // Shell and system surfaces outside the Windows directory check.
        L"explorer.exe",
        L"ApplicationFrameHost.exe",
        L"ShellExperienceHost.exe",
        L"StartMenuExperienceHost.exe",
        L"SearchHost.exe",
        L"dwm.exe",
        L"LockApp.exe",
        L"SystemSettings.exe",
        L"Taskmgr.exe",
      };
      const auto name = executable.filename().wstring();
      return std::any_of(excluded.begin(), excluded.end(), [&](const auto candidate) {
        return equal_name(name, candidate);
      });
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

  bool detail::is_game_candidate(
    const std::filesystem::path &executable,
    const std::span<const std::filesystem::path> loaded_modules,
    const foreground_window::observation_t &foreground,
    const RECT capture_rect
  ) {
    if (!foreground_window::carries_geometry(foreground.status) || !foreground.window || !foreground.process_id || !executable.is_absolute() || !equal_name(executable.extension().wstring(), L".exe") || excluded_executable(executable)) {
      return false;
    }
    const auto client = foreground.client_screen_rect;
    if (!client.valid() || capture_rect.right <= capture_rect.left || capture_rect.bottom <= capture_rect.top) {
      return false;
    }
    // Use the client, not the outer frame, and require it to cover exactly this output: that is
    // the presentation Game 3D export itself requires. Maximized application chrome, windows
    // spanning several outputs and ordinary windows do not qualify. A windowed (e.g. >=75%
    // overlap) rule is deliberately not offered: almost every desktop application now loads
    // D3D11 (Chromium/Electron, Office, WinUI), so it would turn every focused window into a
    // setup offer, while a windowed game cannot export stereo until it covers the output anyway.
    if (client.left != capture_rect.left || client.top != capture_rect.top || client.right != capture_rect.right || client.bottom != capture_rect.bottom) {
      return false;
    }
    return std::any_of(loaded_modules.begin(), loaded_modules.end(), [](const auto &module) {
      const auto name = module.filename().wstring();
      return equal_name(name, L"d3d11.dll") || equal_name(name, L"d3d12.dll");
    });
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
      if (!detail::is_game_candidate(*executable, *modules, foreground, capture_rect)) {
        return std::nullopt;
      }
      const auto after = foreground_window::sample();
      const auto executable_after = process_executable(process.get());
      if (!foreground_window::carries_geometry(after.status) || after.window != foreground.window || after.process_id != foreground.process_id || after.client_screen_rect != foreground.client_screen_rect || !executable_after || !equal_name(executable->wstring(), executable_after->wstring()) || process_creation_time(process.get()) != creation_time || WaitForSingleObject(process.get(), 0) != WAIT_TIMEOUT) {
        return std::nullopt;
      }
      return detected_game {*executable, foreground.process_id, creation_time, display_name(*executable), foreground.window};
    } catch (...) {
      // This advisory must not break an active stream on a path conversion/allocation failure.
      return std::nullopt;
    }
  }
}  // namespace platf::game3d
