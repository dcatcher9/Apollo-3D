// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>
#include <future>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace platf::video_player {
  /** Registration-based application context, not evidence that media is loaded or playing. */
  class registry_t {
  public:
    // Census registry data once on a worker; an unavailable or incomplete result grants nothing.
    registry_t() noexcept;
    [[nodiscard]] bool ready() const noexcept;
    // The caller owns foreground HWND/PID validation. Never caches a PID or discovers a window.
    [[nodiscard]] bool contains_process(std::uint32_t process_id) const noexcept;

  private:
    std::shared_future<std::vector<std::wstring>> executables_;
  };

  namespace detail {
    struct association_t {
      std::wstring extension;
      std::wstring content_type;
      std::wstring perceived_type;
      std::wstring open_command;
    };

    struct registration_t {
      bool registered_media_client = false;
      std::wstring name;
      std::vector<association_t> associations;
    };

    [[nodiscard]] bool is_video_type(const association_t &association) noexcept;
    [[nodiscard]] std::optional<std::wstring> command_executable(std::wstring_view command);
    [[nodiscard]] bool same_executable_path(std::wstring_view left, std::wstring_view right) noexcept;
    // The explicit Media category is mandatory; an Open With/file-handler entry alone cannot pass.
    [[nodiscard]] std::vector<std::wstring> registered_video_executables(const registration_t &registration);
  }  // namespace detail
}  // namespace platf::video_player
