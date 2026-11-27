// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_win32_util.h"
#include "game3d_text_hook.h"

#include "tools/text_hook/text_hook_shared.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <windows.h>

namespace platf::game3d::text_hook {
  namespace {
    using namespace std::chrono_literals;
    constexpr DWORD injector_timeout_ms = 15000u;

    using game3d::win32::handle_t;

    std::filesystem::path hook_library(const unsigned bits) {
      return L"SunshineTextHook" + std::to_wstring(bits) + L".dll";
    }

    std::filesystem::path injector(const unsigned bits) {
      return L"sunshine-text-hook-inject" + std::to_wstring(bits) + L".exe";
    }

    std::wstring section_name(const std::uint32_t process_id) {
      return std::wstring {sunshine_text_hook::section_prefix} + std::to_wstring(process_id);
    }

    /** Run the bundled loader and return its exit code, or a code it never uses on failure. */
    std::uint32_t run_injector(const std::filesystem::path &bundle_directory, const unsigned bits, const std::uint32_t process_id, const std::uint64_t creation_time) {
      const auto executable = bundle_directory / injector(bits);
      std::wstring command = L"\"" + executable.wstring() + L"\" " + std::to_wstring(process_id) + L" " + std::to_wstring(creation_time) +
                             L" \"" + (bundle_directory / hook_library(bits)).wstring() + L"\"";
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

    std::mutex consent_mutex;
    // One game is in front at a time; a short list of the processes approved in this host run.
    std::vector<std::pair<std::uint32_t, std::uint64_t>> consents;

    std::mutex protection_mutex;

    struct {
      std::uint32_t process_id {};
      std::uint64_t creation_time {};
      protection_t state;
      std::chrono::steady_clock::time_point at;
    } reported_protection;
  }  // namespace

  void report_protection(const std::uint32_t process_id, const std::uint64_t creation_time, const protection_t state) {
    std::lock_guard lock(protection_mutex);
    reported_protection = {process_id, creation_time, state, std::chrono::steady_clock::now()};
  }

  protection_t protection(const std::uint32_t process_id, const std::uint64_t creation_time) {
    std::lock_guard lock(protection_mutex);
    const auto &last = reported_protection;
    if (last.process_id != process_id || last.creation_time != creation_time || process_id == 0u || std::chrono::steady_clock::now() - last.at > 3s) {
      return {};
    }
    return last.state;
  }

  std::string describe(const protection_t state) {
    if (state.protection == protection_e::game_text) {
      return "Text protection: the game's own text, reported by its helper.";
    }
    if (state.protection != protection_e::picture) {
      return {};
    }
    switch (state.fallback) {
      case fallback_e::no_helper:
        return "Text protection: fallback. No text helper answers in this game, so its text is found in the picture.";
      case fallback_e::no_text:
        return "Text protection: fallback. The helper reports no text, so text is searched for in the picture.";
      case fallback_e::unsure:
        return "Text protection: fallback. The helper is not sure that what it follows is text, so the text is found in the picture.";
      case fallback_e::not_shown:
        return "Text protection: fallback. The picture does not show the text the helper reports, so the text is found in the picture.";
      case fallback_e::unavailable:
        return "Text protection: fallback. The helper's text cannot be used on this stream now, so the text is found in the picture.";
      case fallback_e::none:
        break;
    }
    return "Text protection: fallback. The text is found in the picture.";
  }

  std::filesystem::path bundled_directory() {
    std::wstring path(32768, L'\0');
    const auto length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (!length || length == path.size()) {
      return {};
    }
    path.resize(length);
    return std::filesystem::path(path).parent_path() / "tools" / "text-hook";
  }

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

  bool loaded(const std::uint32_t process_id, const std::uint64_t creation_time) {
    text_t text;
    return reader_t {}.read(process_id, creation_time, text);
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

  bool reader_t::read(const std::uint32_t process_id, const std::uint64_t creation_time, text_t &text) {
    namespace shared = sunshine_text_hook;
    text = {};
    if (process_id == 0u || creation_time == 0u) {
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
    // The game rewrites these fields at any time. Every one is read once, in this order, through
    // a volatile view, and trusted only if the sequence did not move meanwhile.
    const auto *state = static_cast<const volatile shared::snapshot_t *>(state_);
    const std::uint32_t sequence = state->sequence;
    // A section can outlive its process while this reader holds it open; the same name then
    // says nothing about a later process with the same id.
    if (state->magic != shared::magic || state->version != shared::version || state->process_created != creation_time) {
      return false;
    }
    // Mid-publication: the last complete state still describes the text on screen. Dropping the
    // protection for such a frame would make it flicker while a line types.
    const auto last_complete = [&] {
      text = last_;
      return last_.pixels != nullptr && last_.pixels == coverage_;
    };
    if ((sequence & 1u) != 0u) {
      return last_complete();
    }
    const std::uint32_t width = state->surface_width;
    const std::uint32_t height = state->surface_height;
    const std::uint32_t top = state->text_top;
    const std::uint32_t bottom = state->text_bottom;
    const std::uint64_t window = state->window;
    const std::uint32_t source = state->source;
    if (state->sequence != sequence) {
      return last_complete();
    }
    text.sequence = sequence;
    text.window = window;
    text.unsure = source == shared::source_unsure;
    // A helper that has not seen the game draw yet is running all the same.
    if (width == 0u || height == 0u) {
      last_ = {};
      return true;
    }
    if (width > shared::max_surface_extent || height > shared::max_surface_extent || top > bottom || bottom > height) {
      return false;
    }
    if (!coverage_ || coverage_width_ != width || coverage_height_ != height) {
      if (coverage_) {
        UnmapViewOfFile(coverage_);
      }
      if (coverage_section_) {
        CloseHandle(coverage_section_);
      }
      last_ = {};
      const auto name = std::wstring {shared::coverage_prefix} + std::to_wstring(process_id) + L"-" + std::to_wstring(width) + L"x" + std::to_wstring(height);
      coverage_section_ = OpenFileMappingW(FILE_MAP_READ, FALSE, name.c_str());
      // A view longer than its section is refused, so a section smaller than its name promises
      // is never read past its end.
      coverage_ = coverage_section_ ? MapViewOfFile(coverage_section_, FILE_MAP_READ, 0u, 0u, static_cast<std::size_t>(width) * height * 4u) : nullptr;
      if (!coverage_) {
        if (coverage_section_) {
          CloseHandle(coverage_section_);
        }
        coverage_section_ = nullptr;
        return false;
      }
      coverage_width_ = width;
      coverage_height_ = height;
    }
    text.surface_width = width;
    text.surface_height = height;
    text.top = top;
    text.bottom = bottom;
    text.pixels = static_cast<const std::uint8_t *>(coverage_);
    last_ = text;
    return true;
  }

  bool reader_t::still_current(const text_t &text) const {
    const auto *state = static_cast<const volatile sunshine_text_hook::snapshot_t *>(state_);
    return state && text.pixels == coverage_ && state->sequence == text.sequence;
  }

  std::string detail::injector_failure(const std::uint32_t exit_code) {
    switch (exit_code) {
      case 3u:
        return "The text helper of the other bitness was chosen for this game, so it was not loaded.";
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
    const unsigned bits,
    const std::function<bool()> &is_loaded,
    const std::function<bool()> &bundle_present,
    const std::function<std::uint32_t()> &run,
    const std::function<bool()> &process_running
  ) {
    // The user can approve a game after closing it. That is not a failure: the approval stands
    // and the helper is loaded when the game is next detected.
    if (!process_running()) {
      return {load_state_e::not_running, approved_for_next_start};
    }
    if (is_loaded()) {
      return {load_state_e::loaded, "The text helper is running in this game."};
    }
    if (!bundle_present()) {
      return {load_state_e::unavailable, "This Sunshine 3D build does not include the text helper for " + std::to_string(bits) + "-bit games."};
    }
    if (const auto exit_code = run(); exit_code != 0u) {
      return {load_state_e::failed, injector_failure(exit_code)};
    }
    // The ready marker follows graphics initialization, which can take seconds on a cold start.
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!is_loaded()) {
      if (!process_running()) {
        return {load_state_e::not_running, "The approved game closed while its text helper was starting."};
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        return {load_state_e::failed, "Timed out waiting for the text helper to report ready."};
      }
      std::this_thread::sleep_for(50ms);
    }
    return {load_state_e::loaded, "The text helper is running in this game."};
  }

  bool running(const std::uint32_t process_id, const std::uint64_t creation_time) {
    // A process id alone may by now name another program: the process must be the one created
    // at the time the owner's approval was given for.
    const handle_t process {OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, process_id)};
    return process.get() && creation_time != 0u && game3d::win32::process_creation_time(process.get()) == creation_time && WaitForSingleObject(process.get(), 0u) == WAIT_TIMEOUT;
  }

  load_result load(const std::uint32_t process_id, const std::uint64_t creation_time, const std::filesystem::path &bundle_directory) {
    const handle_t process {OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, process_id)};
    USHORT process_machine = IMAGE_FILE_MACHINE_UNKNOWN, native_machine = IMAGE_FILE_MACHINE_UNKNOWN;
    const bool identified = process.get() && creation_time != 0u && game3d::win32::process_creation_time(process.get()) == creation_time &&
                            IsWow64Process2(process.get(), &process_machine, &native_machine);
    const unsigned bits = process_machine == IMAGE_FILE_MACHINE_UNKNOWN ? 64u : 32u;
    return detail::load_with(
      bits,
      [&] {
        return loaded(process_id, creation_time);
      },
      [&] {
        std::error_code error;
        return !bundle_directory.empty() && std::filesystem::is_regular_file(bundle_directory / hook_library(bits), error) &&
               std::filesystem::is_regular_file(bundle_directory / injector(bits), error);
      },
      [&] {
        return run_injector(bundle_directory, bits, process_id, creation_time);
      },
      [&] {
        // Asked again while the helper starts, so a game that closes meanwhile is seen to.
        return identified && WaitForSingleObject(process.get(), 0u) == WAIT_TIMEOUT;
      }
    );
  }
}  // namespace platf::game3d::text_hook
