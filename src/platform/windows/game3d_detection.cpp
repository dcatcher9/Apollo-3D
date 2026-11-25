// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_detection.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string_view>
#include <mutex>
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
        // SNAPMODULE32 also lists the 32-bit modules of a 32-bit process.
        snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, process_id);
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

  namespace {
    // A verified executable whose whole client area is on the captured output. A text game is
    // converted by the host from the captured picture, which works for a window as well as for
    // fullscreen, so it need not cover the output the way an add-on game's own export must.
    [[nodiscard]] bool client_on_capture_output(
      const std::filesystem::path &executable,
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
      return client.left >= capture_rect.left && client.top >= capture_rect.top && client.right <= capture_rect.right && client.bottom <= capture_rect.bottom;
    }

    [[nodiscard]] bool module_loaded(const std::span<const std::filesystem::path> loaded_modules, const std::wstring_view name) {
      return std::any_of(loaded_modules.begin(), loaded_modules.end(), [&](const auto &module) {
        return equal_name(module.filename().wstring(), name);
      });
    }
  }  // namespace

  namespace {
    // The import table of an executable mapped as an image, every offset checked against the
    // image's size. The file is whatever the game's publisher shipped: nothing in it is trusted.
    class image_t {
    public:
      image_t(const std::uint8_t *base, const std::size_t size):
          base_(base),
          size_(size) {}

      template<class T>
      [[nodiscard]] const T *at(const std::uint64_t rva, const std::size_t count = 1) const {
        return rva <= size_ && count <= (size_ - rva) / sizeof(T) ? reinterpret_cast<const T *>(base_ + rva) : nullptr;
      }

      // A zero-terminated name of at most 255 characters, or an empty view.
      [[nodiscard]] std::string_view name(const std::uint64_t rva) const {
        if (rva >= size_) {
          return {};
        }
        const auto *text = reinterpret_cast<const char *>(base_ + rva);
        const auto length = strnlen(text, std::min<std::size_t>(size_ - rva, 256));
        return length < std::min<std::size_t>(size_ - rva, 256) ? std::string_view {text, length} : std::string_view {};
      }

    private:
      const std::uint8_t *base_;
      std::size_t size_;
    };

    [[nodiscard]] bool equal_ascii(const std::string_view left, const std::string_view right) noexcept {
      return left.size() == right.size() && _strnicmp(left.data(), right.data(), left.size()) == 0;
    }

    [[nodiscard]] detail::executable_imports_t image_imports(const image_t &image) {
      const auto *dos = image.at<IMAGE_DOS_HEADER>(0);
      if (!dos || dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew <= 0) {
        return {};
      }
      const auto header = static_cast<std::uint64_t>(dos->e_lfanew);
      const auto *signature = image.at<DWORD>(header);
      const auto *magic = image.at<WORD>(header + sizeof(DWORD) + sizeof(IMAGE_FILE_HEADER));
      if (!signature || *signature != IMAGE_NT_SIGNATURE || !magic) {
        return {};
      }
      const bool wide = *magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC;
      IMAGE_DATA_DIRECTORY directory {};
      if (wide) {
        const auto *nt = image.at<IMAGE_NT_HEADERS64>(header);
        if (!nt || nt->OptionalHeader.NumberOfRvaAndSizes <= IMAGE_DIRECTORY_ENTRY_IMPORT) {
          return {};
        }
        directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
      } else if (*magic == IMAGE_NT_OPTIONAL_HDR32_MAGIC) {
        const auto *nt = image.at<IMAGE_NT_HEADERS32>(header);
        if (!nt || nt->OptionalHeader.NumberOfRvaAndSizes <= IMAGE_DIRECTORY_ENTRY_IMPORT) {
          return {};
        }
        directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
      } else {
        return {};
      }
      constexpr std::array<std::string_view, 4> gdi_text {"TextOut", "ExtTextOut", "DrawText", "DrawTextEx"};
      bool direct2d = false, directwrite = false, gdi = false, glyphs = false;
      const std::size_t descriptors = std::min<std::size_t>(directory.Size / sizeof(IMAGE_IMPORT_DESCRIPTOR), 4096);
      for (std::size_t index = 0; index < descriptors; ++index) {
        const auto *descriptor = image.at<IMAGE_IMPORT_DESCRIPTOR>(directory.VirtualAddress + index * sizeof(IMAGE_IMPORT_DESCRIPTOR));
        if (!descriptor || descriptor->Name == 0) {
          break;
        }
        const auto library = image.name(descriptor->Name);
        direct2d = direct2d || equal_ascii(library, "d2d1.dll");
        directwrite = directwrite || equal_ascii(library, "dwrite.dll");
        if ((gdi && glyphs) || (!equal_ascii(library, "gdi32.dll") && !equal_ascii(library, "user32.dll"))) {
          continue;
        }
        // The names are listed where the loader has not replaced them with addresses.
        const std::uint64_t thunks = descriptor->OriginalFirstThunk ? descriptor->OriginalFirstThunk : descriptor->FirstThunk;
        for (std::size_t entry = 0; entry < 65536 && !(gdi && glyphs); ++entry) {
          std::uint64_t value = 0;
          if (wide) {
            const auto *thunk = image.at<std::uint64_t>(thunks + entry * 8);
            if (!thunk || *thunk == 0 || (*thunk >> 63) != 0) {
              if (thunk && *thunk != 0) {
                continue;  // imported by number
              }
              break;
            }
            value = *thunk;
          } else {
            const auto *thunk = image.at<std::uint32_t>(thunks + entry * 4);
            if (!thunk || *thunk == 0 || (*thunk >> 31) != 0) {
              if (thunk && *thunk != 0) {
                continue;
              }
              break;
            }
            value = *thunk;
          }
          // Past the two-byte hint; the A and W forms count alike.
          auto function = image.name(value + 2);
          if (function.ends_with('A') || function.ends_with('W')) {
            function.remove_suffix(1);
          }
          gdi = gdi || std::find(gdi_text.begin(), gdi_text.end(), function) != gdi_text.end();
          glyphs = glyphs || function == "GetGlyphOutline";
        }
      }
      return {direct2d && directwrite, gdi, glyphs};
    }

    [[nodiscard]] bool name_starts_with(const std::wstring_view name, const std::wstring_view prefix) noexcept {
      return name.size() >= prefix.size() && equal_name(name.substr(0, prefix.size()), prefix);
    }
  }  // namespace

  detail::executable_imports_t detail::inspect_imports(const std::filesystem::path &executable) noexcept {
    // Mapped as an image, so that the offsets in the file's headers apply as they are, and
    // without execute access: nothing of the file runs here.
    const handle_t file {CreateFileW(executable.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr)};
    if (file.get() == INVALID_HANDLE_VALUE) {
      return {};
    }
    const handle_t mapping {CreateFileMappingW(file.get(), nullptr, PAGE_READONLY | SEC_IMAGE_NO_EXECUTE, 0, 0, nullptr)};
    if (!mapping.get()) {
      return {};
    }
    const auto *base = static_cast<const std::uint8_t *>(MapViewOfFile(mapping.get(), FILE_MAP_READ, 0, 0, 0));
    if (!base) {
      return {};
    }
    executable_imports_t imports {};
    // The view is as long as the image the headers describe; no more than that is read.
    const image_t headers {base, 4096};
    const auto *dos = headers.at<IMAGE_DOS_HEADER>(0);
    const auto *nt = dos && dos->e_magic == IMAGE_DOS_SIGNATURE && dos->e_lfanew > 0 ? headers.at<IMAGE_NT_HEADERS32>(static_cast<std::uint64_t>(dos->e_lfanew)) : nullptr;
    if (nt && nt->Signature == IMAGE_NT_SIGNATURE) {
      // SizeOfImage sits at the same offset in the 32-bit and 64-bit optional headers.
      imports = image_imports({base, nt->OptionalHeader.SizeOfImage});
    }
    UnmapViewOfFile(base);
    return imports;
  }

  bool detail::game_library_loaded(const std::span<const std::filesystem::path> loaded_modules) {
    constexpr std::array exact {L"dsound.dll", L"dinput.dll", L"dinput8.dll", L"d3d8.dll", L"d3d9.dll", L"ddraw.dll", L"openal32.dll", L"soft_oal.dll", L"bass.dll"};
    constexpr std::array prefixes {L"xaudio2_", L"x3daudio", L"xinput", L"fmod"};
    return std::any_of(loaded_modules.begin(), loaded_modules.end(), [&](const auto &module) {
      const auto name = module.filename().wstring();
      return std::any_of(exact.begin(), exact.end(), [&](const auto candidate) {
        return equal_name(name, candidate);
      }) || std::any_of(prefixes.begin(), prefixes.end(), [&](const auto prefix) {
        return name_starts_with(name, prefix);
      });
    });
  }

  bool detail::is_text_game_candidate(
    const std::filesystem::path &executable,
    const std::span<const std::filesystem::path> loaded_modules,
    const foreground_window::observation_t &foreground,
    const RECT capture_rect,
    const bool native,
    const executable_imports_t &imports
  ) {
    if (!client_on_capture_output(executable, foreground, capture_rect)) {
      return false;
    }
    if (native) {
      return (imports.direct2d_text || imports.gdi_glyphs) && game_library_loaded(loaded_modules);
    }
    // Direct2D text drawing always goes through DirectWrite.
    return (module_loaded(loaded_modules, L"dwrite.dll") && module_loaded(loaded_modules, L"d2d1.dll")) ||
           ((imports.gdi_text || imports.gdi_glyphs) && game_library_loaded(loaded_modules));
  }

  bool detail::text_helper_serves(
    const std::span<const std::filesystem::path> loaded_modules,
    const bool native,
    const executable_imports_t &imports
  ) {
    return imports.direct2d_text || imports.gdi_text ||
           (!native && module_loaded(loaded_modules, L"dwrite.dll") && module_loaded(loaded_modules, L"d2d1.dll")) ||
           module_loaded(loaded_modules, L"d3d11.dll");
  }

  bool detail::is_choosable(
    const std::filesystem::path &executable,
    const foreground_window::observation_t &foreground,
    const RECT capture_rect
  ) {
    return client_on_capture_output(executable, foreground, capture_rect);
  }

  bool detail::is_game_candidate(
    const std::filesystem::path &executable,
    const std::span<const std::filesystem::path> loaded_modules,
    const foreground_window::observation_t &foreground,
    const RECT capture_rect
  ) {
    if (!client_on_capture_output(executable, foreground, capture_rect)) {
      return false;
    }
    const auto client = foreground.client_screen_rect;
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
    const bool same_process = provisional_ && provisional_->process_id == candidate->process_id && provisional_->creation_time == candidate->creation_time && provisional_->window == candidate->window && provisional_->kind == candidate->kind && equal_name(provisional_->executable.wstring(), candidate->executable.wstring());
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
    const auto application = observe_foreground(capture_rect);
    return application && application->recognised ? std::optional(application->game) : std::nullopt;
  }

  detected_game as_chosen(const foreground_application &application) {
    auto game = application.game;
    game.kind = game_kind_e::text_hook;
    game.text_helper = application.text_helper;
    game.chosen = true;
    return game;
  }

  std::optional<foreground_application> observe_foreground(const RECT capture_rect) noexcept {
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
      if (!IsWow64Process2(process.get(), &process_machine, &native_machine) || native_machine != IMAGE_FILE_MACHINE_AMD64) {
        return std::nullopt;
      }
      // Native x64 can load the add-on. A 32-bit process never can; it may be a text game.
      const bool native = process_machine == IMAGE_FILE_MACHINE_UNKNOWN;
      if (!native && process_machine != IMAGE_FILE_MACHINE_I386) {
        return std::nullopt;
      }
      const auto creation_time = process_creation_time(process.get());
      const auto executable = process_executable(process.get());
      if (!creation_time || !executable || in_windows_directory(*executable) || excluded_executable(*executable)) {
        return std::nullopt;
      }
      // A module snapshot reads the other process's loader data, so the running game is not
      // snapshotted again at every poll once its list shows the add-on. The add-on pins itself
      // for the life of its process, which makes that list final for every decision below; a
      // process without it is read afresh, because the libraries the rules look for load late.
      static std::mutex modules_mutex;
      static std::uint32_t modules_process = 0;
      static std::uint64_t modules_created = 0;
      static std::vector<std::filesystem::path> modules_with_addon;
      std::optional<std::vector<std::filesystem::path>> modules;
      {
        std::lock_guard lock(modules_mutex);
        if (modules_process == foreground.process_id && modules_created == creation_time) {
          modules = modules_with_addon;
        }
      }
      const bool remembered = modules.has_value();
      if (!remembered) {
        modules = process_modules(foreground.process_id);
      }
      if (!modules) {
        return std::nullopt;
      }
      // A game that already runs the add-on is an add-on game, whatever else it loads.
      const bool addon_loaded = native && module_loaded(*modules, L"SunshineSBS.addon64");
      if (addon_loaded && !remembered) {
        std::lock_guard lock(modules_mutex);
        modules_process = foreground.process_id;
        modules_created = creation_time;
        modules_with_addon = *modules;
      }
      // The executable's file is read once per process, not at every poll.
      static std::mutex imports_mutex;
      static std::uint32_t imports_process = 0;
      static std::uint64_t imports_created = 0;
      static detail::executable_imports_t imports;
      {
        std::lock_guard lock(imports_mutex);
        if (imports_process != foreground.process_id || imports_created != creation_time) {
          imports = detail::inspect_imports(*executable);
          imports_process = foreground.process_id;
          imports_created = creation_time;
        }
      }
      const bool text = !addon_loaded && detail::is_text_game_candidate(*executable, *modules, foreground, capture_rect, native, imports);
      const bool addon = !text && native && detail::is_game_candidate(*executable, *modules, foreground, capture_rect);
      // An application no rule takes for a text game is its owner's to choose for conversion,
      // unless it already supplies stereo of its own.
      const bool choosable = !text && !addon_loaded && detail::is_choosable(*executable, foreground, capture_rect);
      if (!text && !addon && !choosable) {
        return std::nullopt;
      }
      const auto after = foreground_window::sample();
      const auto executable_after = process_executable(process.get());
      if (!foreground_window::carries_geometry(after.status) || after.window != foreground.window || after.process_id != foreground.process_id || after.client_screen_rect != foreground.client_screen_rect || !executable_after || !equal_name(executable->wstring(), executable_after->wstring()) || process_creation_time(process.get()) != creation_time || WaitForSingleObject(process.get(), 0) != WAIT_TIMEOUT) {
        return std::nullopt;
      }
      const bool serves = detail::text_helper_serves(*modules, native, imports);
      return foreground_application {
        detected_game {*executable, foreground.process_id, creation_time, display_name(*executable), foreground.window, text ? game_kind_e::text_hook : game_kind_e::addon, text && serves},
        text || addon,
        choosable,
        serves,
      };
    } catch (...) {
      // This advisory must not break an active stream on a path conversion/allocation failure.
      return std::nullopt;
    }
  }
}  // namespace platf::game3d
