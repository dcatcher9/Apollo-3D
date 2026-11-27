// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <filesystem>
#include <string>

namespace platf::game3d {
  enum class anti_cheat_e {
    absent,
    present,
    unknown,
  };

  // Kernel anti-cheat markers (EasyAntiCheat, BattlEye, EA AntiCheat, ...) in the executable's
  // folder or up to four parents: the game root of Unreal-style <game>/<project>/Binaries/Win64
  // layouts, or at most its library folder. Sibling games' folders, Program Files, the Windows
  // directory and a volume root are never searched. `marker` names the marker found. A folder
  // that cannot be listed completely is unknown.
  [[nodiscard]] anti_cheat_e find_anti_cheat(const std::filesystem::path &executable, std::filesystem::path &marker);

  // Why nothing may be loaded into this game while it runs: anti-cheat software was found with
  // it, or its folders could not be checked for it. Empty when neither. Unlike an installation
  // this does not depend on what else is in the folder.
  [[nodiscard]] std::string injection_refusal(const std::filesystem::path &game_executable);
}  // namespace platf::game3d
