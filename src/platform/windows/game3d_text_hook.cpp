// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_text_hook.h"

#include "tools/text_hook/text_hook_shared.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#include <windows.h>

namespace platf::game3d::text_hook {
  namespace {
    using namespace std::chrono_literals;
    constexpr wchar_t hook_library[] = L"SunshineTextHook32.dll";
    constexpr wchar_t injector[] = L"sunshine-text-hook-inject32.exe";
    constexpr DWORD injector_timeout_ms = 15000u;

    struct handle_closer {
      void operator()(HANDLE handle) const noexcept {
        if (handle) {
          CloseHandle(handle);
        }
      }
    };

    std::wstring section_name(const std::uint32_t process_id) {
      return std::wstring {sunshine_text_hook::section_prefix} + std::to_wstring(process_id);
    }

    /** Run the bundled injector and return its exit code, or a code it never uses on failure. */
    std::uint32_t run_injector(const std::filesystem::path &bundle_directory, const std::uint32_t process_id) {
      const auto executable = bundle_directory / injector;
      std::wstring command = L"\"" + executable.wstring() + L"\" " + std::to_wstring(process_id) + L" \"" + (bundle_directory / hook_library).wstring() + L"\"";
      STARTUPINFOW startup {};
      startup.cb = sizeof(startup);
      PROCESS_INFORMATION process {};
      if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, bundle_directory.c_str(), &startup, &process)) {
        return 0xfffffffeu;
      }
      CloseHandle(process.hThread);
      DWORD exit_code = 0xffffffffu;
      if (WaitForSingleObject(process.hProcess, injector_timeout_ms) != WAIT_OBJECT_0) {
        TerminateProcess(process.hProcess, 0xffffffffu);
      } else {
        GetExitCodeProcess(process.hProcess, &exit_code);
      }
      CloseHandle(process.hProcess);
      return exit_code;
    }
  }  // namespace

  std::filesystem::path bundled_directory() {
    std::wstring path(32768, L'\0');
    const auto length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (!length || length == path.size()) {
      return {};
    }
    path.resize(length);
    return std::filesystem::path(path).parent_path() / "tools" / "text-hook";
  }

  namespace {
    std::mutex consent_mutex;
    // One game is in front at a time; a short list of the processes approved in this host run.
    std::vector<std::pair<std::uint32_t, std::uint64_t>> consents;
  }  // namespace

  void set_consent(const std::uint32_t process_id, const std::uint64_t creation_time, const bool consent) {
    std::lock_guard lock(consent_mutex);
    std::erase_if(consents, [&](const auto &entry) {
      return entry.first == process_id;
    });
    if (consent && process_id != 0u && creation_time != 0u) {
      consents.emplace_back(process_id, creation_time);
    }
  }

  bool consented(const std::uint32_t process_id, const std::uint64_t creation_time) {
    std::lock_guard lock(consent_mutex);
    return std::find(consents.begin(), consents.end(), std::pair {process_id, creation_time}) != consents.end();
  }

  bool loaded(const std::uint32_t process_id) {
    if (process_id == 0u) {
      return false;
    }
    HANDLE section = OpenFileMappingW(FILE_MAP_READ, FALSE, section_name(process_id).c_str());
    if (!section) {
      return false;
    }
    bool current = false;
    if (const auto *view = static_cast<const sunshine_text_hook::snapshot_t *>(MapViewOfFile(section, FILE_MAP_READ, 0u, 0u, sizeof(sunshine_text_hook::snapshot_t)))) {
      current = view->magic == sunshine_text_hook::magic && view->version == sunshine_text_hook::version;
      UnmapViewOfFile(view);
    }
    CloseHandle(section);
    return current;
  }

  reader_t::~reader_t() {
    close();
  }

  void reader_t::close() {
    if (coverage_) {
      UnmapViewOfFile(coverage_);
    }
    if (coverage_section_) {
      CloseHandle(coverage_section_);
    }
    if (state_) {
      UnmapViewOfFile(state_);
    }
    if (state_section_) {
      CloseHandle(state_section_);
    }
    coverage_ = state_ = nullptr;
    coverage_section_ = state_section_ = nullptr;
    coverage_width_ = coverage_height_ = process_id_ = 0u;
    last_ = {};
  }

  text_bounds_t detail::text_bounds(const float *boxes, const std::uint32_t box_count, const std::uint32_t surface_width, const std::uint32_t surface_height) {
    const auto width = static_cast<float>(surface_width);
    const auto height = static_cast<float>(surface_height);
    text_bounds_t bounds {width, height, 0.0f, 0.0f, 0.0f};
    for (std::uint32_t i = 0u; i < box_count; ++i) {
      const float left = boxes[i * 4u], top = boxes[i * 4u + 1u], right = boxes[i * 4u + 2u], bottom = boxes[i * 4u + 3u];
      // A comparison with NaN is false, so a damaged box is skipped rather than widening the result.
      if (!(right > left) || !(bottom > top) || !(right > 0.0f) || !(bottom > 0.0f) || !(left < width) || !(top < height)) {
        continue;
      }
      bounds.left = std::min(bounds.left, std::max(left, 0.0f));
      bounds.top = std::min(bounds.top, std::max(top, 0.0f));
      bounds.right = std::max(bounds.right, std::min(right, width));
      bounds.bottom = std::max(bounds.bottom, std::min(bottom, height));
      bounds.line_height = bounds.line_height > 0.0f ? std::min(bounds.line_height, bottom - top) : bottom - top;
    }
    return bounds.empty() ? text_bounds_t {} : bounds;
  }

  bool reader_t::read(const std::uint32_t process_id, text_t &text) {
    namespace shared = sunshine_text_hook;
    text = {};
    if (process_id == 0u) {
      return false;
    }
    if (process_id != process_id_) {
      close();
      process_id_ = process_id;
    }
    if (!state_) {
      state_section_ = OpenFileMappingW(FILE_MAP_READ, FALSE, section_name(process_id).c_str());
      state_ = state_section_ ? MapViewOfFile(state_section_, FILE_MAP_READ, 0u, 0u, sizeof(shared::snapshot_t)) : nullptr;
      if (!state_) {
        close();
        return false;
      }
    }
    const auto *state = static_cast<const shared::snapshot_t *>(state_);
    const std::uint32_t sequence = state->sequence;
    const std::uint32_t width = state->surface_width;
    const std::uint32_t height = state->surface_height;
    const std::uint32_t count = state->box_count;
    if (state->magic != shared::magic || state->version != shared::version) {
      return false;
    }
    // Mid-update: the last complete state still describes the text on screen, one glyph behind
    // at most. Dropping the protection for such a frame would make it flicker while a line types.
    const auto last_complete = [&] {
      text = last_;
      return last_.pixels != nullptr && last_.pixels == coverage_;
    };
    if ((sequence & 1u) != 0u) {
      return last_complete();
    }
    if (!state->coverage_available || width == 0u || height == 0u || width > 16384u || height > 16384u || count > shared::max_boxes) {
      return false;
    }
    // The boxes are copied out before they are trusted: the game may rewrite them at any time.
    std::array<float, shared::max_boxes * 4u> boxes;
    std::memcpy(boxes.data(), state->boxes, static_cast<std::size_t>(count) * sizeof(shared::box_t));
    if (state->sequence != sequence) {
      return last_complete();
    }
    if (!coverage_ || coverage_width_ != width || coverage_height_ != height) {
      if (coverage_) {
        UnmapViewOfFile(coverage_);
      }
      if (coverage_section_) {
        CloseHandle(coverage_section_);
      }
      coverage_ = nullptr;
      const auto name = std::wstring {shared::coverage_prefix} + std::to_wstring(process_id) + L"-" + std::to_wstring(width) + L"x" + std::to_wstring(height);
      const std::size_t bytes = static_cast<std::size_t>(width) * height * 4u;
      coverage_section_ = OpenFileMappingW(FILE_MAP_READ, FALSE, name.c_str());
      coverage_ = coverage_section_ ? MapViewOfFile(coverage_section_, FILE_MAP_READ, 0u, 0u, bytes) : nullptr;
      MEMORY_BASIC_INFORMATION region {};
      // A section smaller than its name promises must never be read past its end.
      if (!coverage_ || VirtualQuery(coverage_, &region, sizeof(region)) == 0u || region.RegionSize < bytes) {
        if (coverage_) {
          UnmapViewOfFile(coverage_);
        }
        if (coverage_section_) {
          CloseHandle(coverage_section_);
        }
        coverage_ = nullptr;
        coverage_section_ = nullptr;
        return false;
      }
      coverage_width_ = width;
      coverage_height_ = height;
      last_ = {};
    }
    text.sequence = sequence;
    text.surface_width = width;
    text.surface_height = height;
    text.bounds = detail::text_bounds(boxes.data(), count, width, height);
    if (!text.bounds.empty()) {
      // Two rows of margin: an outline pass and antialiasing reach just past the measured text.
      const auto first = static_cast<std::uint32_t>(text.bounds.top);
      text.top = first > 2u ? first - 2u : 0u;
      text.bottom = std::min(height, static_cast<std::uint32_t>(std::ceil(text.bounds.bottom)) + 2u);
    }
    text.pixels = static_cast<const std::uint8_t *>(coverage_);
    last_ = text;
    return true;
  }

  bool reader_t::still_current(const text_t &text) const {
    const auto *state = static_cast<const sunshine_text_hook::snapshot_t *>(state_);
    return state && text.pixels == coverage_ && state->sequence == text.sequence;
  }

  std::string detail::injector_failure(const std::uint32_t exit_code) {
    switch (exit_code) {
      case 3u:
        return "The game is not a 32-bit application, so the text helper does not apply to it.";
      case 4u:
        return "The game process could not be opened, so the text helper was not loaded.";
      case 5u:
        return "The game refused to load the text helper.";
      case 0xfffffffeu:
        return "The bundled text helper could not be started.";
      case 0xffffffffu:
        return "Loading the text helper took too long and was stopped.";
      default:
        return "The text helper could not be loaded (code " + std::to_string(exit_code) + ").";
    }
  }

  load_result detail::load_with(
    const std::function<bool()> &is_loaded,
    const std::function<bool()> &bundle_present,
    const std::function<std::uint32_t()> &run,
    const std::function<bool()> &process_running
  ) {
    // The user can approve a game after closing it. That is not a failure: the approval stands
    // and the helper is loaded when the game is next detected.
    if (!process_running()) {
      return {load_state_e::not_running, "Approved. The text helper loads the next time this game starts."};
    }
    if (is_loaded()) {
      return {load_state_e::loaded, "The text helper is running in this game."};
    }
    if (!bundle_present()) {
      return {load_state_e::unavailable, "This Sunshine 3D build does not include the text helper for 32-bit games."};
    }
    if (const auto exit_code = run(); exit_code != 0u) {
      return {load_state_e::failed, injector_failure(exit_code)};
    }
    // The hook creates its section from its own start-up thread, shortly after it loads.
    for (int attempt = 0; attempt < 40; ++attempt) {
      if (is_loaded()) {
        return {load_state_e::loaded, "The text helper is running in this game."};
      }
      std::this_thread::sleep_for(50ms);
    }
    return {load_state_e::failed, "The text helper loaded but did not start."};
  }

  load_result load(const std::uint32_t process_id, const std::filesystem::path &bundle_directory) {
    return detail::load_with(
      [&] {
        return loaded(process_id);
      },
      [&] {
        std::error_code error;
        return !bundle_directory.empty() && std::filesystem::is_regular_file(bundle_directory / hook_library, error) &&
               std::filesystem::is_regular_file(bundle_directory / injector, error);
      },
      [&] {
        return run_injector(bundle_directory, process_id);
      },
      [&] {
        const std::unique_ptr<void, handle_closer> process {OpenProcess(SYNCHRONIZE, FALSE, process_id)};
        return process && WaitForSingleObject(process.get(), 0u) == WAIT_TIMEOUT;
      }
    );
  }
}  // namespace platf::game3d::text_hook
