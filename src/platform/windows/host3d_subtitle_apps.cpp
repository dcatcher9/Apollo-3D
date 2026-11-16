// SPDX-License-Identifier: GPL-3.0-only
#include "host3d_subtitle_apps.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <windows.h>

#include <nlohmann/json.hpp>

namespace platf::host3d_subtitle_apps {
  namespace {
    constexpr std::size_t max_path_bytes = 4096u;

    std::optional<std::wstring> widen(const std::string_view text) {
      if (text.empty() || text.size() > max_path_bytes) {
        return std::nullopt;
      }
      const auto length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), nullptr, 0);
      if (length <= 0) {
        return std::nullopt;
      }
      std::wstring result(static_cast<std::size_t>(length), L'\0');
      if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), result.data(), length) != length) {
        return std::nullopt;
      }
      return result;
    }

    std::optional<std::string> narrow(const std::wstring_view text) {
      if (text.empty()) {
        return std::nullopt;
      }
      const auto length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
      if (length <= 0 || static_cast<std::size_t>(length) > max_path_bytes) {
        return std::nullopt;
      }
      std::string result(static_cast<std::size_t>(length), '\0');
      if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), result.data(), length, nullptr, nullptr) != length) {
        return std::nullopt;
      }
      return result;
    }

    bool equal_path(const std::string_view left, const std::string_view right) {
      const auto a = widen(left);
      const auto b = widen(right);
      return a && b && a->size() == b->size() &&
             CompareStringOrdinal(a->data(), static_cast<int>(a->size()), b->data(), static_cast<int>(b->size()), TRUE) == CSTR_EQUAL;
    }

    /** One spelling per executable: absolute, lexically normal, forward slashes, `.exe`. */
    std::optional<std::string> normalized(const std::string_view executable) {
      if (executable.find('\0') != std::string_view::npos) {
        return std::nullopt;
      }
      const auto wide = widen(executable);
      if (!wide) {
        return std::nullopt;
      }
      try {
        const std::filesystem::path path {*wide};
        if (!path.is_absolute() || !path.has_filename()) {
          return std::nullopt;
        }
        const auto extension = path.extension().native();
        if (CompareStringOrdinal(extension.data(), static_cast<int>(extension.size()), L".exe", 4, TRUE) != CSTR_EQUAL) {
          return std::nullopt;
        }
        return narrow(path.lexically_normal().generic_wstring());
      } catch (...) {
        return std::nullopt;
      }
    }

    std::string file_name(const std::string_view executable) {
      const auto slash = executable.find_last_of('/');
      return std::string {slash == std::string_view::npos ? executable : executable.substr(slash + 1u)};
    }

    auto find_path(const std::vector<std::string> &paths, const std::string_view executable) {
      return std::find_if(paths.begin(), paths.end(), [&](const auto &candidate) {
        return equal_path(candidate, executable);
      });
    }

    std::mutex selection_mutex;
    selection_t selection;
  }  // namespace

  void selection_t::load(const std::string_view json_array) {
    enabled_.clear();
    if (json_array.empty()) {
      return;
    }
    const auto parsed = nlohmann::json::parse(json_array, nullptr, false);
    if (!parsed.is_array()) {
      return;
    }
    for (const auto &entry : parsed) {
      if (entry.is_string()) {
        set_enabled(entry.get<std::string>(), true);
      }
    }
  }

  std::string selection_t::serialize() const {
    return nlohmann::json(enabled_).dump();
  }

  bool selection_t::set_enabled(const std::string_view executable, const bool enabled) {
    const auto path = normalized(executable);
    if (!path) {
      return false;
    }
    const auto existing = find_path(enabled_, *path);
    if (!enabled) {
      if (existing != enabled_.end()) {
        enabled_.erase(existing);
      }
      return true;
    }
    if (existing != enabled_.end()) {
      return true;
    }
    if (enabled_.size() >= max_enabled) {
      return false;
    }
    enabled_.push_back(*path);
    return true;
  }

  bool selection_t::enabled(const std::string_view executable) const {
    const auto path = normalized(executable);
    return path && find_path(enabled_, *path) != enabled_.end();
  }

  bool selection_t::observe(const std::string_view executable) {
    const auto path = normalized(executable);
    if (!path) {
      return false;
    }
    if (recent_.empty() || !equal_path(recent_.front(), *path)) {
      const auto existing = find_path(recent_, *path);
      if (existing != recent_.end()) {
        recent_.erase(existing);
      }
      recent_.insert(recent_.begin(), *path);
      if (recent_.size() > max_recent) {
        recent_.pop_back();
      }
    }
    return find_path(enabled_, *path) != enabled_.end();
  }

  std::vector<app_t> selection_t::list() const {
    std::vector<app_t> result;
    result.reserve(enabled_.size() + recent_.size());
    for (const auto &path : enabled_) {
      result.push_back({path, file_name(path), true, find_path(recent_, path) != recent_.end()});
    }
    for (const auto &path : recent_) {
      if (find_path(enabled_, path) == enabled_.end()) {
        result.push_back({path, file_name(path), false, true});
      }
    }
    return result;
  }

  void load(const std::string_view json_array) {
    std::lock_guard lock(selection_mutex);
    selection.load(json_array);
  }

  std::string serialize() {
    std::lock_guard lock(selection_mutex);
    return selection.serialize();
  }

  std::vector<app_t> list() {
    std::lock_guard lock(selection_mutex);
    return selection.list();
  }

  bool set_enabled(const std::string_view executable, const bool enabled) {
    std::lock_guard lock(selection_mutex);
    return selection.set_enabled(executable, enabled);
  }

  bool observe_process(const std::uint32_t process_id) noexcept {
    if (process_id == 0u) {
      return false;
    }
    try {
      const std::unique_ptr<void, decltype(&CloseHandle)> process(
        OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, process_id), CloseHandle
      );
      std::array<wchar_t, 32768> image {};
      DWORD length = static_cast<DWORD>(image.size());
      if (!process || !QueryFullProcessImageNameW(process.get(), 0u, image.data(), &length)) {
        return false;
      }
      const auto path = narrow(std::wstring_view {image.data(), length});
      if (!path) {
        return false;
      }
      std::lock_guard lock(selection_mutex);
      return selection.observe(*path);
    } catch (...) {
      return false;
    }
  }
}  // namespace platf::host3d_subtitle_apps
