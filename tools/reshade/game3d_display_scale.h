#pragma once

#include <cstdint>

#ifdef _WIN32
  #include <Windows.h>
#endif

// A game that is not DPI aware sees its monitor at the logical size, the
// physical mode divided by the Windows display scaling: at 150% a 3840x2160
// display looks like 2560x1440, and a borderless game then renders at that
// size whatever its own resolution setting says. The warning names the cause
// when the game's back buffer equals that logical size while the physical mode
// is larger; a DPI-aware game sees the physical size and is never flagged.
namespace sunshine_game3d::display_scale {
  struct status {
    bool limited{};
    std::uint32_t game_width{}, game_height{}, display_width{}, display_height{}, scale_percent{100};
  };

  constexpr bool within_two(std::uint32_t a, std::uint32_t b) {
    return a > b ? a - b <= 2u : b - a <= 2u;
  }

  constexpr status classify(std::uint32_t game_width, std::uint32_t game_height, std::uint32_t logical_width,
      std::uint32_t logical_height, std::uint32_t physical_width, std::uint32_t physical_height) {
    status s;
    s.game_width = game_width;
    s.game_height = game_height;
    s.display_width = physical_width;
    s.display_height = physical_height;
    if (logical_width) s.scale_percent = (physical_width * 100u + logical_width / 2u) / logical_width;
    s.limited = logical_width && logical_height && physical_width > logical_width + 2u &&
      physical_height > logical_height + 2u && within_two(game_width, logical_width) && within_two(game_height, logical_height);
    return s;
  }

  static_assert(classify(2560, 1440, 2560, 1440, 3840, 2160).limited &&
    classify(2560, 1440, 2560, 1440, 3840, 2160).scale_percent == 150);
  static_assert(!classify(3840, 2160, 3840, 2160, 3840, 2160).limited); // 100% or a DPI-aware game.
  static_assert(!classify(1920, 1080, 2560, 1440, 3840, 2160).limited); // A smaller window or setting.
  static_assert(classify(1536, 864, 1536, 864, 1920, 1080).limited &&
    classify(1536, 864, 1536, 864, 1920, 1080).scale_percent == 125);

#ifdef _WIN32
  // Monitor geometry as this (game) thread sees it against the monitor's
  // physical mode. EnumDisplaySettings reports the real mode; GetMonitorInfo
  // reports logical coordinates to a DPI-unaware process.
  inline status query(void *window, std::uint32_t game_width, std::uint32_t game_height) {
    const HMONITOR monitor = MonitorFromWindow(static_cast<HWND>(window), MONITOR_DEFAULTTONEAREST);
    MONITORINFOEXW info {};
    info.cbSize = sizeof(info);
    if (!monitor || !GetMonitorInfoW(monitor, &info)) return {};
    DEVMODEW mode {};
    mode.dmSize = sizeof(mode);
    if (!EnumDisplaySettingsW(info.szDevice, ENUM_CURRENT_SETTINGS, &mode)) return {};
    const auto logical_width = static_cast<std::uint32_t>(info.rcMonitor.right - info.rcMonitor.left);
    const auto logical_height = static_cast<std::uint32_t>(info.rcMonitor.bottom - info.rcMonitor.top);
    return classify(game_width, game_height, logical_width, logical_height, mode.dmPelsWidth, mode.dmPelsHeight);
  }
#endif
}
