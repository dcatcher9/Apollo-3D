// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_win32_util.h"
#include "game3d_anti_cheat.h"

#include <string_view>

#include <Windows.h>

#include <ShlObj.h>

namespace platf::game3d {
  namespace {
    namespace fs = std::filesystem;

    using win32::ends_with_name;
    using win32::starts_with_name;
    using win32::utf8;

    // Files and folders that kernel-level anti-cheat systems install inside a game's folder.
    // tools/reshade/install-runtime.ps1 applies the same markers and walk independently.
    bool anti_cheat_marker(std::wstring_view name) {
      for (const std::wstring_view prefix : {L"EasyAntiCheat", L"BattlEye", L"EAAntiCheat", L"XIGNCODE", L"AntiCheatExpert", L"HoYoKProtect", L"mhyprot"}) {
        if (starts_with_name(name, prefix)) {
          return true;
        }
      }
      for (const std::wstring_view exact : {L"start_protected_game.exe", L"GameGuard", L"randgrid.sys"}) {
        if (name.size() == exact.size() && starts_with_name(name, exact)) {
          return true;
        }
      }
      return ends_with_name(name, L"_BE.exe");
    }

    // Program Files and the Windows directory hold the anti-cheat services' own system-wide
    // installations, and a volume root holds unrelated libraries; markers there do not
    // identify the game, so the walk stops before them.
    bool shared_root(const fs::path &directory) {
      if (!directory.has_relative_path()) {
        return true;
      }
      for (const auto &folder : {FOLDERID_ProgramFilesX64, FOLDERID_ProgramFilesX86, FOLDERID_Windows}) {
        PWSTR known = nullptr;
        const auto result = SHGetKnownFolderPath(folder, KF_FLAG_DEFAULT, nullptr, &known);
        const bool same = SUCCEEDED(result) && known && win32::same_name(fs::path(known).lexically_normal().native(), directory.native());
        CoTaskMemFree(known);
        if (same) {
          return true;
        }
      }
      return false;
    }
  }  // namespace

  // Checks the executable's folder and up to four parents: the game root of Unreal-style
  // <game>/<project>/Binaries/Win64 layouts, or at most its library folder. Sibling games'
  // folders are never entered. A folder that cannot be listed completely is unknown.
  anti_cheat_e find_anti_cheat(const fs::path &executable, fs::path &marker) {
    auto directory = executable.parent_path();
    for (int level = 0; level < 5 && !shared_root(directory); ++level) {
      WIN32_FIND_DATAW entry {};
      const auto pattern = directory / L"*";
      const auto find = FindFirstFileExW(pattern.c_str(), FindExInfoBasic, &entry, FindExSearchNameMatch, nullptr, FIND_FIRST_EX_LARGE_FETCH);
      if (find == INVALID_HANDLE_VALUE) {
        return anti_cheat_e::unknown;
      }
      bool complete = false;
      for (std::size_t entries = 0; entries < 65536; ++entries) {
        if (anti_cheat_marker(entry.cFileName)) {
          marker = directory / entry.cFileName;
          FindClose(find);
          return anti_cheat_e::present;
        }
        if (!FindNextFileW(find, &entry)) {
          complete = GetLastError() == ERROR_NO_MORE_FILES;
          break;
        }
      }
      FindClose(find);
      if (!complete) {
        return anti_cheat_e::unknown;
      }
      directory = directory.parent_path();
    }
    return anti_cheat_e::absent;
  }

  std::string injection_refusal(const fs::path &game_executable) {
    try {
      fs::path marker;
      switch (find_anti_cheat(game_executable, marker)) {
        case anti_cheat_e::present:
          return "Anti-cheat detected (" + utf8(marker.filename()) + " in " + utf8(marker.parent_path()) + "), so Sunshine 3D does not load anything into this game.";
        case anti_cheat_e::unknown:
          return "The game's folders could not be checked for anti-cheat, so Sunshine 3D does not load anything into this game.";
        case anti_cheat_e::absent:
          return {};
      }
    } catch (const std::exception &) {
    }
    return "The game's folders could not be checked for anti-cheat, so Sunshine 3D does not load anything into this game.";
  }
}  // namespace platf::game3d
