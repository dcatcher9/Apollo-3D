/**
 * @file src/platform/windows/primary_display_journal.cpp
 * @brief The primary-display crash journal: its JSON codec and its file beside the configuration.
 */
#include "primary_display_journal.h"

#include "misc.h"
#include "src/config.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <type_traits>

namespace platf::primary_display::detail {
  namespace {
    constexpr size_t max_journal_bytes = 1024 * 1024;

    template<class T>
    std::string encode_ccd(const T &value) {
      static_assert(std::is_trivially_copyable_v<T>);
      constexpr char digits[] = "0123456789abcdef";
      std::string result(sizeof(T) * 2, '0');
      const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
      for (size_t i = 0; i < sizeof(T); ++i) {
        result[i * 2] = digits[bytes[i] >> 4];
        result[i * 2 + 1] = digits[bytes[i] & 15];
      }
      return result;
    }

    template<class T>
    T decode_ccd(const nlohmann::json &value) {
      const auto encoded = value.get<std::string>();
      if (encoded.size() != sizeof(T) * 2) {
        throw std::runtime_error("unsupported CCD structure size");
      }
      T result {};
      auto *bytes = reinterpret_cast<unsigned char *>(&result);
      auto nibble = [](char c) -> unsigned char {
        if (c >= '0' && c <= '9') {
          return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
          return c - 'a' + 10;
        }
        throw std::runtime_error("invalid CCD encoding");
      };
      for (size_t i = 0; i < sizeof(T); ++i) {
        bytes[i] = static_cast<unsigned char>((nibble(encoded[i * 2]) << 4) | nibble(encoded[i * 2 + 1]));
      }
      return result;
    }

    nlohmann::json encode_snapshot(const detail::snapshot_t &snapshot) {
      nlohmann::json result {{"encoding", "windows-ccd-v1"}, {"paths", nlohmann::json::array()}, {"modes", nlohmann::json::array()}, {"devices", nlohmann::json::array()}, {"colors", nlohmann::json::array()}};
      // CCD structures have a fixed Win32 ABI. Preserve all timing/scaling data, including
      // desktop-image modes absent from older MinGW headers. IDs/indexes are remapped on
      // recovery; this payload is never sent directly back to SetDisplayConfig.
      for (const auto &path : snapshot.paths) {
        result["paths"].push_back(encode_ccd(path));
      }
      for (const auto &mode : snapshot.modes) {
        result["modes"].push_back(encode_ccd(mode));
      }
      for (const auto &identity : snapshot.device_paths) {
        result["devices"].push_back(platf::to_utf8(identity));
      }
      for (const auto &color : snapshot.colors) {
        if (!color) {
          result["colors"].push_back(nullptr);
        } else {
          // "legacy" and "enabled" mean nothing to this build, which ignores them. Older host
          // builds, such as the installed host or a staged build run beside this one, read every
          // colour key with .at() and drop the whole recovery record when one is missing, so a
          // crash journal written here would not restore under them. Keep writing both as those
          // builds wrote them on Windows 11 24H2: "legacy" false, "enabled" the active state.
          result["colors"].push_back({{"legacy", false}, {"enabled", color->advanced_color_active}, {"hdr_supported", color->hdr_supported}, {"hdr", color->hdr_user_enabled}, {"wcg", color->wcg_user_enabled}, {"active", color->advanced_color_active}, {"limited", color->limited_by_policy}, {"bits", color->bits_per_color_channel}, {"mode", static_cast<int>(color->active_mode)}});
        }
      }
      return result;
    }

    detail::snapshot_t decode_snapshot(const nlohmann::json &value) {
      if (value.at("encoding") != "windows-ccd-v1" || !value.at("paths").is_array() || value.at("paths").size() > max_displays || !value.at("modes").is_array() || value.at("modes").size() > max_displays * 3 || !value.at("devices").is_array() || value.at("devices").size() != value.at("paths").size() || !value.at("colors").is_array() || value.at("colors").size() > value.at("paths").size()) {
        throw std::runtime_error("invalid saved display topology");
      }
      detail::snapshot_t result;
      for (const auto &path : value.at("paths")) {
        result.paths.push_back(decode_ccd<DISPLAYCONFIG_PATH_INFO>(path));
      }
      for (const auto &mode : value.at("modes")) {
        result.modes.push_back(decode_ccd<DISPLAYCONFIG_MODE_INFO>(mode));
      }
      for (const auto &identity : value.at("devices")) {
        result.device_paths.push_back(platf::from_utf8(identity.get<std::string>()));
      }
      for (const auto &entry : value.at("colors")) {
        if (entry.is_null()) {
          result.colors.push_back(std::nullopt);
          continue;
        }
        const int mode = entry.at("mode").get<int>();
        if (mode < DISPLAYCONFIG_ADVANCED_COLOR_MODE_SDR || mode > DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR) {
          throw std::runtime_error("invalid saved display color mode");
        }
        // "legacy" and "enabled" are written only for older builds (see encode_snapshot) and are
        // ignored here: the fields below hold the HDR/WCG state either way (a "legacy": true entry
        // from a build with the pre-24H2 query recorded HDR on or off with WCG off), so every
        // entry restores through those setters.
        result.colors.push_back(display_config::advanced_color_state_t {entry.at("hdr_supported").get<bool>(), entry.at("hdr").get<bool>(), entry.at("wcg").get<bool>(), entry.at("active").get<bool>(), entry.at("limited").get<bool>(), entry.at("bits").get<UINT32>(), static_cast<DISPLAYCONFIG_ADVANCED_COLOR_MODE>(mode)});
      }
      const bool empty = result.paths.empty() && result.modes.empty() && result.device_paths.empty() && result.colors.empty();
      if (!empty && !detail::inspect(result)) {
        throw std::runtime_error("invalid saved display paths or modes");
      }
      return result;
    }
  }  // namespace

  std::filesystem::path journal_path() {
    auto path = std::filesystem::path(platf::from_utf8(config::sunshine.config_file));
    path += L".apollo-primary-display.json";
    return path;
  }

  detail::load_result_t load_journal() {
    const auto path = journal_path();
    std::error_code error;
    if (!std::filesystem::exists(path, error)) {
      return {!error, std::nullopt};
    }
    const auto size = std::filesystem::file_size(path, error);
    if (error || size == 0 || size > max_journal_bytes) {
      return {};
    }
    std::ifstream input(path, std::ios::binary);
    std::string contents(static_cast<size_t>(size), '\0');
    if (!input.read(contents.data(), static_cast<std::streamsize>(contents.size())) || input.peek() != std::char_traits<char>::eof()) {
      return {};
    }
    auto journal = detail::deserialize(contents);
    return {journal.has_value(), std::move(journal)};
  }

  bool save_journal(const detail::journal_t &journal) {
    const auto contents = detail::serialize(journal);
    if (contents.empty() || contents.size() > max_journal_bytes) {
      return false;
    }
    const auto path = journal_path();
    auto temporary = path;
    temporary += L".tmp";
    const HANDLE file = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
      return false;
    }
    DWORD written = 0;
    const bool durable = WriteFile(file, contents.data(), static_cast<DWORD>(contents.size()), &written, nullptr) &&
                         written == contents.size() && FlushFileBuffers(file);
    const bool closed = CloseHandle(file);
    if (!durable || !closed || !MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
      DeleteFileW(temporary.c_str());
      return false;
    }
    return true;
  }

  bool clear_journal() {
    if (DeleteFileW(journal_path().c_str())) {
      return true;
    }
    return GetLastError() == ERROR_FILE_NOT_FOUND;
  }

  std::string serialize(const journal_t &journal) {
    if (!valid_journal(journal)) {
      return {};
    }
    auto encode = [](const layout_t &layout) {
      auto result = nlohmann::json::array();
      for (const auto &entry : layout) {
        result.push_back({{"device_path", platf::to_utf8(entry.device_path)}, {"x", entry.x}, {"y", entry.y}});
      }
      return result;
    };
    nlohmann::json result {{"version", journal.monitor_replacement ? 5 : !journal.local_sink.empty() ? 4 :
                                                                       journal.exclusive             ? 3 :
                                                                                                       1},
                           {"prepared", journal.prepared},
                           {"original_primary", platf::to_utf8(journal.original_primary)},
                           {"promoted_primary", platf::to_utf8(journal.promoted_primary)},
                           {"original", encode(journal.original)},
                           {"promoted", encode(journal.promoted)}};
    if (journal.exclusive) {
      if (journal.monitor_replacement) {
        result["monitor_replacement"] = true;
      }
      result["exclusive"] = true;
      result["exclusive_started"] = journal.exclusive_started;
      result["original_topology"] = encode_snapshot(*journal.original_topology);
      result["before_exclusive"] = journal.before_exclusive ? encode_snapshot(*journal.before_exclusive) : nlohmann::json(nullptr);
      result["pending_restore"] = journal.pending_restore ? encode_snapshot(*journal.pending_restore) : nlohmann::json(nullptr);
      result["exclusive_preserved"] = nlohmann::json::array();
      for (const auto &identity : journal.exclusive_preserved) {
        result["exclusive_preserved"].push_back(platf::to_utf8(identity));
      }
      result["exclusive_topology"] = journal.exclusive_topology ? encode_snapshot(*journal.exclusive_topology) : nlohmann::json(nullptr);
      if (!journal.local_sink.empty()) {
        result["local_sink"] = platf::to_utf8(journal.local_sink);
      }
    }
    return result.dump();
  }

  std::optional<journal_t> deserialize(std::string_view contents) {
    if (contents.size() > max_journal_bytes) {
      return std::nullopt;
    }
    try {
      const auto value = nlohmann::json::parse(contents);
      const int version = value.at("version").get<int>();
      if (version != 1 && version != 2 && version != 3 && version != 4 && version != 5) {
        return std::nullopt;
      }
      auto decode = [](const nlohmann::json &entries) {
        layout_t result;
        if (!entries.is_array() || entries.size() > max_displays) {
          throw std::runtime_error("invalid primary-display layout");
        }
        for (const auto &entry : entries) {
          auto coordinate = [&](const char *name) {
            const auto &number = entry.at(name);
            if (!number.is_number_integer()) {
              throw std::runtime_error("invalid display coordinate");
            }
            if (number.is_number_unsigned() && number.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<LONG>::max())) {
              throw std::runtime_error("display coordinate out of range");
            }
            const auto parsed = number.get<int64_t>();
            if (parsed < std::numeric_limits<LONG>::min() || parsed > std::numeric_limits<LONG>::max()) {
              throw std::runtime_error("display coordinate out of range");
            }
            return static_cast<LONG>(parsed);
          };
          result.push_back({platf::from_utf8(entry.at("device_path").get<std::string>()), coordinate("x"), coordinate("y")});
        }
        return result;
      };
      journal_t result {platf::from_utf8(value.at("original_primary").get<std::string>()), platf::from_utf8(value.at("promoted_primary").get<std::string>()), decode(value.at("original")), decode(value.at("promoted")), value.at("prepared").get<bool>()};
      if (version >= 2) {
        if (version >= 5) {
          result.monitor_replacement = value.at("monitor_replacement").get<bool>();
          if (!result.monitor_replacement) {
            return std::nullopt;
          }
        }
        result.exclusive = value.at("exclusive").get<bool>();
        result.exclusive_started = value.at("exclusive_started").get<bool>();
        result.original_topology = decode_snapshot(value.at("original_topology"));
        if (!value.at("before_exclusive").is_null()) {
          result.before_exclusive = decode_snapshot(value.at("before_exclusive"));
        }
        if (!value.at("pending_restore").is_null()) {
          result.pending_restore = decode_snapshot(value.at("pending_restore"));
        }
        if (version >= 3) {
          const auto &preserved = value.at("exclusive_preserved");
          if (!preserved.is_array() || preserved.size() > max_displays) {
            throw std::runtime_error("invalid preserved-display list");
          }
          for (const auto &identity : preserved) {
            result.exclusive_preserved.push_back(platf::from_utf8(identity.get<std::string>()));
          }
          if (!value.at("exclusive_topology").is_null()) {
            result.exclusive_topology = decode_snapshot(value.at("exclusive_topology"));
          }
        }
        if (version >= 4) {
          result.local_sink = platf::from_utf8(value.at("local_sink").get<std::string>());
          if (result.local_sink.empty()) {
            return std::nullopt;
          }
        }
      }
      return valid_journal(result) ? std::make_optional(std::move(result)) : std::nullopt;
    } catch (const std::exception &) {
      return std::nullopt;
    }
  }

}  // namespace platf::primary_display::detail
