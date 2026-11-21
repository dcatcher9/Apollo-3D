// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <future>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace platf::video_player {
  /** Registration-based application context, not evidence that media is loaded or playing. */
  class registry_t {
  public:
    // Census registry data once on a worker; an unavailable or incomplete result grants nothing.
    registry_t() noexcept;
#ifdef SUNSHINE_TESTS
    // Finite census result for an owned CPU/GPU fixture; never starts native discovery.
    explicit registry_t(std::shared_future<std::vector<std::wstring>> executables) noexcept;
#endif
    [[nodiscard]] bool ready() const noexcept;
    // Borrow the immutable census while the registry is alive. An incomplete census grants nothing.
    [[nodiscard]] std::span<const std::wstring> executables_if_ready() const noexcept;

  private:
    std::shared_future<std::vector<std::wstring>> executables_;
  };

  [[nodiscard]] bool matches_registered_executable(
    std::span<const std::wstring> registered,
    std::wstring_view executable
  ) noexcept;

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
