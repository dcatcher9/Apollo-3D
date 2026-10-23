// SPDX-License-Identifier: GPL-3.0-only
#include "video_player_context.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <memory>
#include <windows.h>

#include <shellapi.h>

namespace platf::video_player {
  namespace {
    constexpr DWORD max_clients = 128u;
    constexpr DWORD max_associations = 256u;
    constexpr DWORD max_total_associations = 4096u;
    constexpr std::size_t max_executables = 128u;
    constexpr DWORD max_string = 32768u;

    struct key_t {
      HKEY value = nullptr;

      key_t(HKEY root, const std::wstring &path, const REGSAM view) {
        RegOpenKeyExW(root, path.c_str(), 0, KEY_READ | view, &value);
      }

      ~key_t() {
        if (value) {
          RegCloseKey(value);
        }
      }

      key_t(const key_t &) = delete;
      key_t &operator=(const key_t &) = delete;
    };

    std::wstring read_string(HKEY key, const wchar_t *name) {
      DWORD size = 0u;
      DWORD type = 0u;
      if (!key || RegQueryValueExW(key, name, nullptr, &type, nullptr, &size) != ERROR_SUCCESS ||
          (type != REG_SZ && type != REG_EXPAND_SZ) || size < sizeof(wchar_t) ||
          size > max_string * sizeof(wchar_t) || size % sizeof(wchar_t) != 0u) {
        return {};
      }
      std::wstring value(size / sizeof(wchar_t), L'\0');
      const auto expected_type = type;
      if (RegQueryValueExW(key, name, nullptr, &type, reinterpret_cast<BYTE *>(value.data()), &size) != ERROR_SUCCESS ||
          type != expected_type || size < sizeof(wchar_t) || size % sizeof(wchar_t) != 0u) {
        return {};
      }
      value.resize(size / sizeof(wchar_t));
      if (value.back() != L'\0') {
        return {};
      }
      value.pop_back();
      if (value.find(L'\0') != std::wstring::npos) {
        return {};
      }
      if (type == REG_EXPAND_SZ) {
        const DWORD needed = ExpandEnvironmentStringsW(value.c_str(), nullptr, 0u);
        if (needed == 0u || needed > max_string) {
          return {};
        }
        std::wstring expanded(needed, L'\0');
        if (ExpandEnvironmentStringsW(value.c_str(), expanded.data(), needed) != needed) {
          return {};
        }
        expanded.pop_back();
        return expanded;
      }
      return value;
    }

    bool equal_text(const std::wstring_view left, const std::wstring_view right) noexcept {
      return left.size() == right.size() && left.size() < max_string &&
             CompareStringOrdinal(left.data(), static_cast<int>(left.size()), right.data(), static_cast<int>(right.size()), TRUE) == CSTR_EQUAL;
    }

    std::optional<std::wstring> normalized_path(const std::wstring_view path) {
      if (path.empty() || path.size() >= max_string || path.find(L'\0') != std::wstring_view::npos) {
        return std::nullopt;
      }
      auto value = std::filesystem::path {path};
      if (!value.is_absolute()) {
        return std::nullopt;
      }
      return value.lexically_normal().make_preferred().native();
    }

    std::vector<std::wstring> census() {
      std::vector<std::wstring> result;
      DWORD total_associations = 0u;
      for (const auto root : {HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE}) {
        for (const auto view : {KEY_WOW64_64KEY, KEY_WOW64_32KEY}) {
          key_t media(root, L"Software\\Clients\\Media", view);
          if (!media.value) {
            continue;
          }
          for (DWORD index = 0u;; ++index) {
            std::array<wchar_t, 256> name {};
            DWORD length = static_cast<DWORD>(name.size());
            const auto status = RegEnumKeyExW(media.value, index, name.data(), &length, nullptr, nullptr, nullptr, nullptr);
            if (status == ERROR_NO_MORE_ITEMS) {
              break;
            }
            if (index == max_clients || status != ERROR_SUCCESS) {
              return {};
            }
            detail::registration_t registration {true, std::wstring(name.data(), length), {}};
            key_t associations(media.value, registration.name + L"\\Capabilities\\FileAssociations", view);
            if (!associations.value) {
              continue;
            }
            for (DWORD item = 0u;; ++item) {
              std::array<wchar_t, 256> extension {};
              length = static_cast<DWORD>(extension.size());
              const auto value_status = RegEnumValueW(associations.value, item, extension.data(), &length, nullptr, nullptr, nullptr, nullptr);
              if (value_status == ERROR_NO_MORE_ITEMS) {
                break;
              }
              if (item == max_associations || value_status != ERROR_SUCCESS || ++total_associations > max_total_associations) {
                return {};
              }
              const auto prog_id = read_string(associations.value, extension.data());
              if (prog_id.empty() || prog_id.find_first_of(L"\\/") != std::wstring::npos) {
                continue;
              }
              key_t type(HKEY_CLASSES_ROOT, extension.data(), view);
              detail::association_t association {
                std::wstring(extension.data(), length),
                read_string(type.value, L"Content Type"),
                read_string(type.value, L"PerceivedType"),
                {},
              };
              if (detail::is_video_type(association)) {
                key_t command(HKEY_CLASSES_ROOT, prog_id + L"\\shell\\open\\command", view);
                association.open_command = read_string(command.value, nullptr);
                registration.associations.push_back(std::move(association));
              }
            }
            for (auto &executable : detail::registered_video_executables(registration)) {
              if (std::none_of(result.begin(), result.end(), [&](const auto &current) {
                    return equal_text(current, executable);
                  })) {
                if (result.size() == max_executables) {
                  return {};
                }
                result.push_back(std::move(executable));
              }
            }
          }
        }
      }
      return result;
    }
  }  // namespace

  namespace detail {
    bool is_video_type(const association_t &association) noexcept {
      const auto &extension = association.extension;
      return extension.size() > 1u && extension.size() < 256u && extension.front() == L'.' &&
             extension.find(L'\0') == std::wstring::npos &&
             extension.find_first_of(L"\\/: \t\r\n") == std::wstring::npos &&
             (equal_text(association.perceived_type, L"video") ||
              (association.content_type.size() > 6u && equal_text(std::wstring_view {association.content_type}.substr(0u, 6u), L"video/")));
    }

    std::optional<std::wstring> command_executable(const std::wstring_view command) {
      if (command.empty() || command.size() >= max_string || command.find(L'\0') != std::wstring_view::npos) {
        return std::nullopt;
      }
      int argc = 0;
      const auto text = std::wstring(command);
      auto *argv = CommandLineToArgvW(text.c_str(), &argc);
      if (!argv) {
        return std::nullopt;
      }
      std::unique_ptr<void, decltype(&LocalFree)> cleanup(argv, LocalFree);
      if (argc == 0 || !equal_text(std::filesystem::path {argv[0]}.extension().native(), L".exe")) {
        return std::nullopt;
      }
      return normalized_path(argv[0]);
    }

    bool same_executable_path(const std::wstring_view left, const std::wstring_view right) noexcept {
      try {
        const auto a = normalized_path(left);
        const auto b = normalized_path(right);
        return a && b && equal_text(*a, *b);
      }
      catch (...) {
        return false;
      }
    }

    std::vector<std::wstring> registered_video_executables(const registration_t &registration) {
      std::vector<std::wstring> result;
      if (registration.registered_media_client && !registration.name.empty()) {
        for (const auto &association : registration.associations) {
          if (is_video_type(association)) {
            if (auto executable = command_executable(association.open_command)) {
              result.push_back(std::move(*executable));
            }
          }
        }
      }
      return result;
    }
  }  // namespace detail

  registry_t::registry_t() noexcept {
    try {
      executables_ = std::async(std::launch::async, [] {
        try {
          return census();
        }
        catch (...) {
          return std::vector<std::wstring> {};
        }
      }).share();
    }
    catch (...) {
    }
  }

  bool registry_t::ready() const noexcept {
    try {
      return executables_.valid() && executables_.wait_for(std::chrono::seconds {0}) == std::future_status::ready;
    }
    catch (...) {
      return false;
    }
  }

  bool registry_t::contains_process(const std::uint32_t process_id) const noexcept {
    if (process_id == 0u || !ready()) {
      return false;
    }
    try {
      const auto &registered = executables_.get();
      if (registered.empty()) {
        return false;
      }
      std::unique_ptr<void, decltype(&CloseHandle)> process(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, process_id), CloseHandle);
      std::array<wchar_t, max_string> image {};
      DWORD length = static_cast<DWORD>(image.size());
      if (!process || !QueryFullProcessImageNameW(process.get(), 0u, image.data(), &length)) {
        return false;
      }
      const auto path = normalized_path(std::wstring_view {image.data(), length});
      return path && std::any_of(registered.begin(), registered.end(), [&](const auto &candidate) {
               return equal_text(candidate, *path);
             });
    }
    catch (...) {
      return false;
    }
  }
}  // namespace platf::video_player
