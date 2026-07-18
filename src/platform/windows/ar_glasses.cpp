#include "ar_glasses.h"

#include "display.h"
#include "misc.h"
#include "src/config.h"
#include "src/file_handler.h"
#include "src/logging.h"
#include "src/process.h"
#include "src/system_tray.h"
#include "src/uuid.h"
#include "virtual_display.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace ar_glasses {
  using namespace std::literals;

  namespace {
    constexpr std::string_view managed_config_key = "ar_glass_devices";
    constexpr wchar_t virtual_display_driver_name[] = L"SudoMaker Virtual Display Adapter";
    constexpr char virtual_display_uuid[] = "F43872A4-9948-45DF-BD84-5B33BD71E4E8";
    constexpr char virtual_display_name[] = "Apollo AR Desktop";
    constexpr int source_width = 1920;
    constexpr int source_height = 1080;
    constexpr auto topology_poll_interval = 250ms;
    constexpr auto topology_debounce = 750ms;
    constexpr auto failed_session_retry = 2s;
    constexpr auto maximum_failed_session_retry = 30s;
    constexpr auto ownership_release_timeout = 10s;

    std::mutex ownership_mutex;
    std::condition_variable ownership_changed;
    bool local_session_present = false;
    bool remote_session_active = false;
    std::chrono::steady_clock::time_point remote_session_pending_until {};
    std::optional<std::stop_source> local_session_construction_stop;

    std::chrono::milliseconds remote_pending_duration(std::chrono::milliseconds connect_timeout) {
      return std::clamp(connect_timeout + 2000ms, 2000ms, 60000ms);
    }

    bool remote_blocks_local_locked(std::chrono::steady_clock::time_point now) {
      return remote_session_active || now < remote_session_pending_until;
    }

    struct hdr_state_t {
      bool known = false;
      bool supported = false;
      bool user_enabled = false;
      bool active = false;
      bool limited_by_policy = false;
      UINT32 bits_per_color = 0;
    };

    hdr_state_t query_hdr_state(const LUID &adapter_id, UINT32 target_id) {
      DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO_2 info2 {};
      info2.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO_2;
      info2.header.size = sizeof(info2);
      info2.header.adapterId = adapter_id;
      info2.header.id = target_id;
      if (DisplayConfigGetDeviceInfo(&info2.header) == ERROR_SUCCESS) {
        return {
          true,
          info2.highDynamicRangeSupported != 0,
          info2.highDynamicRangeUserEnabled != 0,
          info2.activeColorMode == DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR,
          info2.advancedColorLimitedByPolicy != 0,
          info2.bitsPerColorChannel,
        };
      }

      // Windows 10 exposes only the combined Advanced Color query. On an HDR display its
      // enabled state is also the active HDR state, so it remains a valid compatibility fallback.
      DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO info {};
      info.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO;
      info.header.size = sizeof(info);
      info.header.adapterId = adapter_id;
      info.header.id = target_id;
      if (DisplayConfigGetDeviceInfo(&info.header) == ERROR_SUCCESS) {
        return {
          true,
          info.advancedColorSupported != 0,
          info.advancedColorEnabled != 0,
          info.advancedColorEnabled != 0,
          info.advancedColorForceDisabled != 0,
          info.bitsPerColorChannel,
        };
      }
      return {};
    }

    bool set_hdr_state(const LUID &adapter_id, UINT32 target_id, bool enabled) {
      DISPLAYCONFIG_SET_HDR_STATE hdr {};
      hdr.header.type = DISPLAYCONFIG_DEVICE_INFO_SET_HDR_STATE;
      hdr.header.size = sizeof(hdr);
      hdr.header.adapterId = adapter_id;
      hdr.header.id = target_id;
      hdr.enableHdr = enabled;
      if (DisplayConfigSetDeviceInfo(&hdr.header) == ERROR_SUCCESS) {
        return true;
      }

      DISPLAYCONFIG_SET_ADVANCED_COLOR_STATE advanced_color {};
      advanced_color.header.type = DISPLAYCONFIG_DEVICE_INFO_SET_ADVANCED_COLOR_STATE;
      advanced_color.header.size = sizeof(advanced_color);
      advanced_color.header.adapterId = adapter_id;
      advanced_color.header.id = target_id;
      advanced_color.enableAdvancedColor = enabled;
      return DisplayConfigSetDeviceInfo(&advanced_color.header) == ERROR_SUCCESS;
    }

    std::optional<float> query_sdr_white_nits(const LUID &adapter_id, UINT32 target_id) {
      DISPLAYCONFIG_SDR_WHITE_LEVEL white_level {};
      white_level.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
      white_level.header.size = sizeof(white_level);
      white_level.header.adapterId = adapter_id;
      white_level.header.id = target_id;
      if (DisplayConfigGetDeviceInfo(&white_level.header) != ERROR_SUCCESS) {
        return std::nullopt;
      }
      // Windows encodes SDR reference white relative to 80 nits, with 1000 == 80 nits.
      return (float) white_level.SDRWhiteLevel * 80.0f / 1000.0f;
    }

    struct target_state_t {
      std::string device_id;
      std::wstring device_path;
      std::string friendly_name;
      std::string gdi_name;
      std::string desktop_topology;
      LUID adapter_id {};
      UINT32 target_id = 0;
      hdr_state_t hdr;
      RECT rect {};
      int refresh_millihz = 60000;
      presentation_mode_e mode = presentation_mode_e::unsupported;

      bool operator==(const target_state_t &other) const {
        return device_id == other.device_id && device_path == other.device_path &&
               gdi_name == other.gdi_name && desktop_topology == other.desktop_topology &&
               rect.left == other.rect.left && rect.top == other.rect.top &&
               rect.right == other.rect.right && rect.bottom == other.rect.bottom &&
               refresh_millihz == other.refresh_millihz && mode == other.mode &&
               hdr.known == other.hdr.known &&
               hdr.supported == other.hdr.supported &&
               hdr.active == other.hdr.active &&
               hdr.limited_by_policy == other.hdr.limited_by_policy;
      }
    };

    struct stored_device_t: device_info_t {
      bool notified = false;
    };

    std::mutex device_mutex;
    std::vector<stored_device_t> known_devices;
    std::mutex topology_recovery_mutex;

    struct topology_recovery_t {
      std::wstring device_path;
      RECT original_rect {};
      RECT applied_rect {};
    };

    bool same_rect(const RECT &left, const RECT &right) {
      return left.left == right.left && left.top == right.top &&
             left.right == right.right && left.bottom == right.bottom;
    }

    std::filesystem::path topology_recovery_path() {
      auto path = std::filesystem::path(platf::from_utf8(config::sunshine.config_file));
      path += L".apollo-ar-topology.json";
      return path;
    }

    topology_recovery_t parse_topology_recovery(const nlohmann::json &value) {
      topology_recovery_t recovery {};
      recovery.device_path = platf::from_utf8(value.at("device_path").get<std::string>());
      recovery.original_rect.left = value.at("original_left").get<LONG>();
      recovery.original_rect.top = value.at("original_top").get<LONG>();
      recovery.original_rect.right = value.at("original_right").get<LONG>();
      recovery.original_rect.bottom = value.at("original_bottom").get<LONG>();
      recovery.applied_rect.left = value.at("applied_left").get<LONG>();
      recovery.applied_rect.top = value.at("applied_top").get<LONG>();
      recovery.applied_rect.right = value.at("applied_right").get<LONG>();
      recovery.applied_rect.bottom = value.at("applied_bottom").get<LONG>();
      if (recovery.device_path.empty() ||
          recovery.original_rect.right <= recovery.original_rect.left ||
          recovery.original_rect.bottom <= recovery.original_rect.top ||
          recovery.applied_rect.right <= recovery.applied_rect.left ||
          recovery.applied_rect.bottom <= recovery.applied_rect.top) {
        throw std::runtime_error("invalid monitor identity or rectangle");
      }
      return recovery;
    }

    nlohmann::json serialize_topology_recovery(const topology_recovery_t &recovery) {
      return {
        {"device_path", platf::to_utf8(recovery.device_path)},
        {"original_left", recovery.original_rect.left},
        {"original_top", recovery.original_rect.top},
        {"original_right", recovery.original_rect.right},
        {"original_bottom", recovery.original_rect.bottom},
        {"applied_left", recovery.applied_rect.left},
        {"applied_top", recovery.applied_rect.top},
        {"applied_right", recovery.applied_rect.right},
        {"applied_bottom", recovery.applied_rect.bottom},
      };
    }

    std::optional<std::vector<topology_recovery_t>> load_topology_recoveries_locked(
      bool *legacy_format = nullptr
    ) {
      if (legacy_format) {
        *legacy_format = false;
      }
      const auto path = topology_recovery_path();
      std::ifstream input(path, std::ios::binary);
      if (!input) {
        std::error_code error;
        if (std::filesystem::exists(path, error) || error) {
          BOOST_LOG(warning) << "Could not read local-AR topology recovery state ["sv
                             << platf::to_utf8(path.wstring()) << "]."sv;
          return std::nullopt;
        }
        return std::vector<topology_recovery_t> {};
      }
      try {
        const auto value = nlohmann::json::parse(input);
        const int version {value.at("version").get<int>()};
        std::vector<topology_recovery_t> recoveries;
        if (version == 2) {
          // Version 2 stored one global record. Preserve it as the first per-device entry and
          // atomically migrate it on the next successful write.
          recoveries.emplace_back(parse_topology_recovery(value));
          if (legacy_format) {
            *legacy_format = true;
          }
        } else if (version == 3) {
          for (const auto &entry : value.at("recoveries")) {
            auto recovery = parse_topology_recovery(entry);
            if (std::ranges::any_of(recoveries, [&](const auto &existing) {
                  return existing.device_path == recovery.device_path;
                })) {
              throw std::runtime_error("duplicate monitor identity");
            }
            recoveries.emplace_back(std::move(recovery));
          }
        } else {
          throw std::runtime_error("unsupported recovery-record version");
        }
        return recoveries;
      } catch (const std::exception &error) {
        BOOST_LOG(warning) << "Ignoring invalid local-AR topology recovery record ["sv
                           << platf::to_utf8(path.wstring()) << "]: "sv << error.what();
        return std::nullopt;
      }
    }

    bool write_topology_recoveries_locked(const std::vector<topology_recovery_t> &recoveries) {
      const auto path = topology_recovery_path();
      if (recoveries.empty()) {
        std::error_code error;
        const bool removed = std::filesystem::remove(path, error);
        if (error || removed) {
          return !error;
        }
        const bool still_exists = std::filesystem::exists(path, error);
        return !error && !still_exists;
      }

      auto temporary_path = path;
      temporary_path += L".tmp";
      nlohmann::json entries {nlohmann::json::array()};
      for (const auto &recovery : recoveries) {
        entries.emplace_back(serialize_topology_recovery(recovery));
      }
      const nlohmann::json value {
        {"version", 3},
        {"recoveries", std::move(entries)},
      };
      {
        std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
        if (!output) {
          return false;
        }
        output << value.dump();
        output.flush();
        if (!output) {
          output.close();
          std::error_code ignored;
          std::filesystem::remove(temporary_path, ignored);
          return false;
        }
      }
      if (!MoveFileExW(
            temporary_path.c_str(),
            path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
          )) {
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        return false;
      }
      return true;
    }

    bool clear_topology_recovery(std::wstring_view device_path) {
      std::lock_guard lock(topology_recovery_mutex);
      auto recoveries {load_topology_recoveries_locked()};
      if (!recoveries) {
        return false;
      }
      const auto original_size {recoveries->size()};
      std::erase_if(*recoveries, [&](const auto &recovery) {
        return recovery.device_path == device_path;
      });
      return recoveries->size() == original_size || write_topology_recoveries_locked(*recoveries);
    }

    bool persist_topology_recovery(
      std::wstring_view device_path,
      const RECT &original_rect,
      const RECT &applied_rect,
      bool &created
    ) {
      created = false;
      std::lock_guard lock(topology_recovery_mutex);
      auto recoveries {load_topology_recoveries_locked()};
      if (!recoveries) {
        return false;
      }
      auto existing {std::ranges::find_if(*recoveries, [&](const auto &recovery) {
        return recovery.device_path == device_path;
      })};
      if (existing == recoveries->end()) {
        recoveries->push_back({std::wstring(device_path), original_rect, applied_rect});
        created = true;
      } else {
        // Preserve the first pre-session rectangle. Only the exact rectangle Windows actually
        // applied may change while this device's isolation session remains active.
        existing->applied_rect = applied_rect;
      }
      if (!write_topology_recoveries_locked(*recoveries)) {
        created = false;
        return false;
      }
      return true;
    }

    bool same_presentation_contract(
      const std::optional<target_state_t> &left,
      const std::optional<target_state_t> &right
    ) {
      if (!left || !right) {
        return !left && !right;
      }
      const auto left_width = left->rect.right - left->rect.left;
      const auto left_height = left->rect.bottom - left->rect.top;
      const auto right_width = right->rect.right - right->rect.left;
      const auto right_height = right->rect.bottom - right->rect.top;
      return left->device_path == right->device_path &&
             left_width == right_width && left_height == right_height &&
             left->refresh_millihz == right->refresh_millihz && left->mode == right->mode &&
             left->desktop_topology == right->desktop_topology &&
             left->hdr.known == right->hdr.known &&
             left->hdr.supported == right->hdr.supported &&
             left->hdr.active == right->hdr.active &&
             left->hdr.limited_by_policy == right->hdr.limited_by_policy;
    }

    bool contains_case_insensitive(std::wstring_view haystack, std::wstring_view needle) {
      return std::search(
               haystack.begin(),
               haystack.end(),
               needle.begin(),
               needle.end(),
               [](wchar_t left, wchar_t right) {
                 return std::towupper(left) == std::towupper(right);
               }
             ) != haystack.end();
    }

    std::string lowercase(std::string_view value) {
      std::string result(value);
      std::transform(result.begin(), result.end(), result.begin(), [](unsigned char ch) {
        return (char) std::tolower(ch);
      });
      return result;
    }

    bool is_internal_virtual_display(std::string_view device_id, std::string_view friendly_name) {
      const auto name = lowercase(friendly_name);
      return device_id == "DISPLAY:SMKD1CE"sv || name.starts_with("apollo ar des"sv);
    }

    const char *decision_name(device_decision_e decision) {
      switch (decision) {
        case device_decision_e::approved:
          return "approved";
        case device_decision_e::rejected:
          return "rejected";
        default:
          return "pending";
      }
    }

    device_decision_e decision_from_name(std::string_view value) {
      if (value == "approved") {
        return device_decision_e::approved;
      }
      if (value == "rejected") {
        return device_decision_e::rejected;
      }
      return device_decision_e::pending;
    }

    std::string serialize_devices_locked() {
      nlohmann::json value = nlohmann::json::array();
      for (const auto &device : known_devices) {
        value.push_back({
          {"id", device.id},
          {"name", device.name},
          {"decision", decision_name(device.decision)},
          {"auto_detected", device.auto_detected},
        });
      }
      return value.dump();
    }

    std::string replace_managed_config_value(std::string contents, std::string_view value) {
      const std::string replacement = std::string(managed_config_key) + " = " + std::string(value);
      size_t cursor = 0;
      while (cursor < contents.size()) {
        const auto newline = contents.find('\n', cursor);
        const auto line_end = newline == std::string::npos ? contents.size() : newline;
        std::string_view line(contents.data() + cursor, line_end - cursor);
        const bool had_carriage_return = !line.empty() && line.back() == '\r';
        if (!line.empty() && line.back() == '\r') {
          line.remove_suffix(1);
        }
        const auto first = line.find_first_not_of(" \t");
        const auto equals = line.find('=', first == std::string_view::npos ? 0 : first);
        const auto key_end = equals == std::string_view::npos ? equals : line.find_last_not_of(" \t", equals - 1);
        const bool managed_line = first != std::string_view::npos && equals != std::string_view::npos &&
                                  key_end != std::string_view::npos &&
                                  line.substr(first, key_end - first + 1) == managed_config_key;
        if (managed_line) {
          contents.replace(cursor, line_end - cursor, replacement + (had_carriage_return ? "\r" : ""));
          return contents;
        }
        if (newline == std::string::npos) {
          break;
        }
        cursor = newline + 1;
      }

      const std::string_view newline = contents.find("\r\n") != std::string::npos ? "\r\n" : "\n";
      if (!contents.empty() && contents.back() != '\n') {
        contents.append(newline);
      }
      contents.append(replacement).append(newline);
      return contents;
    }

    bool write_config_atomically(std::string_view contents) {
      const std::filesystem::path path = platf::from_utf8(config::sunshine.config_file);
      auto temporary_path = path;
      temporary_path += L".apollo-ar.tmp";
      {
        std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
        if (!output) {
          return false;
        }
        output.write(contents.data(), (std::streamsize) contents.size());
        output.flush();
        if (!output) {
          output.close();
          std::error_code ignored;
          std::filesystem::remove(temporary_path, ignored);
          return false;
        }
      }

      if (!MoveFileExW(
            temporary_path.c_str(),
            path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
          )) {
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        return false;
      }
      return true;
    }

    void persist_devices_locked() {
      auto contents = file_handler::read_file(config::sunshine.config_file.c_str());
      contents = replace_managed_config_value(std::move(contents), serialize_devices_locked());
      if (!write_config_atomically(contents)) {
        BOOST_LOG(error) << "Could not persist the AR display decision list to "sv
                         << config::sunshine.config_file << '.';
      }
    }

    void load_devices() {
      std::lock_guard lock(device_mutex);
      known_devices.clear();
      bool removed_internal_display = false;
      try {
        const auto vars = config::parse_config(file_handler::read_file(config::sunshine.config_file.c_str()));
        const auto option = vars.find(std::string(managed_config_key));
        if (option == vars.end() || option->second.empty()) {
          return;
        }
        for (const auto &entry : nlohmann::json::parse(option->second)) {
          stored_device_t device;
          device.id = entry.value("id", "");
          device.name = entry.value("name", device.id);
          device.decision = decision_from_name(entry.value("decision", "pending"));
          device.auto_detected = entry.value("auto_detected", false);
          if (is_internal_virtual_display(device.id, device.name)) {
            removed_internal_display = true;
            continue;
          }
          if (!device.id.empty() && std::none_of(known_devices.begin(), known_devices.end(), [&](const auto &existing) {
                return existing.id == device.id;
              })) {
            known_devices.emplace_back(std::move(device));
          }
        }
        if (removed_internal_display) {
          persist_devices_locked();
          BOOST_LOG(info) << "Removed Apollo's internal virtual desktop from the AR display decision list."sv;
        }
      } catch (const std::exception &error) {
        BOOST_LOG(warning) << "Ignoring invalid "sv << managed_config_key << " configuration: "sv
                           << error.what();
      }
    }

    std::string stable_model_id(const DISPLAYCONFIG_TARGET_DEVICE_NAME &target_name) {
      std::wstring path = target_name.monitorDevicePath;
      const auto display = std::search(
        path.begin(),
        path.end(),
        L"DISPLAY#",
        L"DISPLAY#" + 8,
        [](wchar_t left, wchar_t right) {
          return std::towupper(left) == std::towupper(right);
        }
      );
      if (display != path.end()) {
        const auto model_end = std::find(display + 8, path.end(), L'#');
        if (model_end != path.end()) {
          std::wstring model(display, model_end);
          std::replace(model.begin(), model.end(), L'#', L':');
          std::transform(model.begin(), model.end(), model.begin(), std::towupper);
          return platf::to_utf8(model);
        }
      }

      std::ostringstream fallback;
      fallback << "DISPLAY:" << std::hex << std::uppercase
               << target_name.edidManufactureId << ':' << target_name.edidProductCodeId;
      return fallback.str();
    }

    std::optional<std::vector<target_state_t>> enumerate_targets() {
      std::vector<DISPLAYCONFIG_PATH_INFO> paths;
      std::vector<DISPLAYCONFIG_MODE_INFO> modes;
      if (!VDISPLAY::queryActiveDisplayConfig(paths, modes)) {
        return std::nullopt;
      }

      const auto virtual_sources = VDISPLAY::matchDisplay(virtual_display_driver_name);
      std::vector<target_state_t> targets;
      for (const auto &path : paths) {
        DISPLAYCONFIG_SOURCE_DEVICE_NAME source_name {};
        source_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
        source_name.header.size = sizeof(source_name);
        source_name.header.adapterId = path.sourceInfo.adapterId;
        source_name.header.id = path.sourceInfo.id;
        if (DisplayConfigGetDeviceInfo(&source_name.header) != ERROR_SUCCESS) {
          continue;
        }
        const auto virtual_source = std::find_if(virtual_sources.begin(), virtual_sources.end(), [&](const auto &name) {
          return contains_case_insensitive(source_name.viewGdiDeviceName, name) &&
                 std::wstring_view(source_name.viewGdiDeviceName).size() == name.size();
        });
        if (virtual_source != virtual_sources.end()) {
          continue;
        }

        DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
        target_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
        target_name.header.size = sizeof(target_name);
        target_name.header.adapterId = path.targetInfo.adapterId;
        target_name.header.id = path.targetInfo.id;
        if (DisplayConfigGetDeviceInfo(&target_name.header) != ERROR_SUCCESS || contains_case_insensitive(target_name.monitorDevicePath, L"SUDOVDA")) {
          continue;
        }
        const auto device_id = stable_model_id(target_name);
        const auto friendly_name = platf::to_utf8(target_name.monitorFriendlyDeviceName);
        if (is_internal_virtual_display(device_id, friendly_name)) {
          continue;
        }

        if (path.sourceInfo.modeInfoIdx == DISPLAYCONFIG_PATH_MODE_IDX_INVALID || path.sourceInfo.modeInfoIdx >= modes.size()) {
          continue;
        }
        const auto &mode_info = modes[path.sourceInfo.modeInfoIdx];
        if (mode_info.infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE) {
          continue;
        }

        const auto &source_mode = mode_info.sourceMode;
        const auto &refresh = path.targetInfo.refreshRate;
        int refresh_millihz = 60000;
        if (refresh.Numerator && refresh.Denominator) {
          refresh_millihz = (int) std::lround(
            (double) refresh.Numerator * 1000.0 / (double) refresh.Denominator
          );
        }

        target_state_t state;
        state.device_id = device_id;
        state.device_path = target_name.monitorDevicePath;
        state.friendly_name = friendly_name;
        if (state.friendly_name.empty()) {
          state.friendly_name = state.device_id;
        }
        state.gdi_name = platf::to_utf8(source_name.viewGdiDeviceName);
        state.adapter_id = path.targetInfo.adapterId;
        state.target_id = path.targetInfo.id;
        state.hdr = query_hdr_state(state.adapter_id, state.target_id);
        state.rect.left = source_mode.position.x;
        state.rect.top = source_mode.position.y;
        state.rect.right = source_mode.position.x + (LONG) source_mode.width;
        state.rect.bottom = source_mode.position.y + (LONG) source_mode.height;
        state.refresh_millihz = refresh_millihz;
        state.mode = classify_mode(source_mode.width, source_mode.height);
        targets.emplace_back(std::move(state));
      }

      std::wstring primary_gdi_name;
      for (DWORD index = 0;; ++index) {
        DISPLAY_DEVICEW device {};
        device.cb = sizeof(device);
        if (!EnumDisplayDevicesW(nullptr, index, &device, 0)) {
          break;
        }
        if ((device.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE) != 0) {
          primary_gdi_name = device.DeviceName;
          break;
        }
      }
      std::string primary_device_path;
      for (const auto &target : targets) {
        if (platf::from_utf8(target.gdi_name) == primary_gdi_name) {
          primary_device_path = platf::to_utf8(target.device_path);
          break;
        }
      }
      for (auto &target : targets) {
        std::vector<std::string> neighbors;
        for (const auto &neighbor : targets) {
          if (neighbor.device_path == target.device_path) {
            continue;
          }
          std::ostringstream description;
          description << platf::to_utf8(neighbor.device_path) << '@'
                      << neighbor.rect.left << ',' << neighbor.rect.top << ':'
                      << (neighbor.rect.right - neighbor.rect.left) << 'x'
                      << (neighbor.rect.bottom - neighbor.rect.top) << '@'
                      << neighbor.refresh_millihz;
          neighbors.emplace_back(description.str());
        }
        std::sort(neighbors.begin(), neighbors.end());
        std::ostringstream fingerprint;
        fingerprint << "primary=" << primary_device_path;
        for (const auto &neighbor : neighbors) {
          fingerprint << ';' << neighbor;
        }
        target.desktop_topology = fingerprint.str();
      }
      std::sort(targets.begin(), targets.end(), [](const auto &left, const auto &right) {
        return std::tie(left.device_id, left.device_path) < std::tie(right.device_id, right.device_path);
      });

      return targets;
    }

    std::optional<target_state_t> find_target(
      std::wstring_view required_device_path = {},
      std::wstring_view preferred_device_path = {},
      bool *query_succeeded = nullptr
    ) {
      if (query_succeeded) {
        *query_succeeded = false;
      }
      auto targets_result = enumerate_targets();
      if (!targets_result) {
        return std::nullopt;
      }
      if (query_succeeded) {
        *query_succeeded = true;
      }
      auto &targets = *targets_result;
      if (!required_device_path.empty()) {
        const auto match = std::find_if(targets.begin(), targets.end(), [&](const auto &target) {
          return target.device_path == required_device_path;
        });
        return match == targets.end() ? std::nullopt : std::optional<target_state_t>(*match);
      }

      std::vector<std::pair<std::string, std::string>> notify_devices;
      std::optional<target_state_t> selected;
      bool changed = false;
      {
        std::lock_guard lock(device_mutex);
        for (auto &device : known_devices) {
          device.connected = false;
        }
        for (const auto &target : targets) {
          auto known = std::find_if(known_devices.begin(), known_devices.end(), [&](const auto &device) {
            return device.id == target.device_id;
          });
          if (known == known_devices.end()) {
            stored_device_t device;
            device.id = target.device_id;
            device.name = target.friendly_name;
            device.connected = true;
            device.auto_detected = is_recognized_ar_display(device.id, device.name);
            device.decision = device.auto_detected ? device_decision_e::approved : device_decision_e::pending;
            known_devices.emplace_back(std::move(device));
            known = std::prev(known_devices.end());
            changed = true;
            BOOST_LOG(info) << "Discovered monitor model ["sv << known->name << ", "sv << known->id
                            << "] classified as "sv << decision_name(known->decision) << '.';
          } else {
            known->connected = true;
            if (known->name != target.friendly_name) {
              known->name = target.friendly_name;
              changed = true;
            }
            if (known->decision == device_decision_e::pending && is_recognized_ar_display(known->id, known->name)) {
              known->decision = device_decision_e::approved;
              known->auto_detected = true;
              changed = true;
              BOOST_LOG(info) << "Pending monitor model ["sv << known->name << ", "sv << known->id
                              << "] is now recognized as an AR display."sv;
            }
          }

          if (known->decision == device_decision_e::pending && !known->notified) {
            notify_devices.emplace_back(known->id, known->name);
          }
          if (known->decision == device_decision_e::approved && (!selected || target.device_path == preferred_device_path)) {
            selected = target;
          }
        }
        if (changed) {
          persist_devices_locked();
        }
      }

      for (const auto &[id, name] : notify_devices) {
        if (system_tray::update_tray_ar_display_decision(name)) {
          std::lock_guard lock(device_mutex);
          const auto device = std::find_if(known_devices.begin(), known_devices.end(), [&](const auto &candidate) {
            return candidate.id == id;
          });
          if (device != known_devices.end()) {
            device->notified = true;
          }
        }
      }
      return selected;
    }

    const char *mode_name(presentation_mode_e mode) {
      switch (mode) {
        case presentation_mode_e::normal:
          return "normal";
        case presentation_mode_e::sbs_ai:
          return "SBS AI";
        default:
          return "unsupported";
      }
    }

    std::optional<std::wstring> refresh_virtual_display_name(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity
    ) {
      auto name = VDISPLAY::getDisplayName(identity.AdapterLuid, identity.TargetId);
      return name.empty() ? std::nullopt : std::optional<std::wstring>(std::move(name));
    }

    bool restore_physical_output_position(
      const RECT &original_rect,
      const std::wstring &target_device_path
    );

    std::optional<target_state_t> isolate_physical_output(
      const std::wstring &virtual_gdi_name,
      const std::wstring &target_device_path,
      const RECT &original_rect,
      bool *layout_changed = nullptr
    ) {
      std::wstring primary_gdi_name;
      for (DWORD index = 0;; ++index) {
        DISPLAY_DEVICEW device {};
        device.cb = sizeof(device);
        if (!EnumDisplayDevicesW(nullptr, index, &device, 0)) {
          break;
        }
        if ((device.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE) != 0) {
          primary_gdi_name = device.DeviceName;
          break;
        }
      }
      if (primary_gdi_name.empty()) {
        BOOST_LOG(warning) << "Could not identify the Windows primary display for AR-output placement."sv;
        return std::nullopt;
      }

      std::vector<DISPLAYCONFIG_PATH_INFO> paths;
      std::vector<DISPLAYCONFIG_MODE_INFO> modes;
      if (!VDISPLAY::queryActiveDisplayConfig(paths, modes)) {
        return std::nullopt;
      }

      size_t physical_mode_index = std::numeric_limits<size_t>::max();
      size_t virtual_mode_index = std::numeric_limits<size_t>::max();
      size_t primary_mode_index = std::numeric_limits<size_t>::max();
      for (const auto &path : paths) {
        if (path.sourceInfo.modeInfoIdx == DISPLAYCONFIG_PATH_MODE_IDX_INVALID || path.sourceInfo.modeInfoIdx >= modes.size() || modes[path.sourceInfo.modeInfoIdx].infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE) {
          continue;
        }

        DISPLAYCONFIG_SOURCE_DEVICE_NAME source_name {};
        source_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
        source_name.header.size = sizeof(source_name);
        source_name.header.adapterId = path.sourceInfo.adapterId;
        source_name.header.id = path.sourceInfo.id;
        if (DisplayConfigGetDeviceInfo(&source_name.header) == ERROR_SUCCESS) {
          const std::wstring_view gdi_name = source_name.viewGdiDeviceName;
          if (gdi_name == virtual_gdi_name) {
            virtual_mode_index = path.sourceInfo.modeInfoIdx;
          }
          if (gdi_name == primary_gdi_name) {
            primary_mode_index = path.sourceInfo.modeInfoIdx;
          }
        }

        DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
        target_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
        target_name.header.size = sizeof(target_name);
        target_name.header.adapterId = path.targetInfo.adapterId;
        target_name.header.id = path.targetInfo.id;
        if (DisplayConfigGetDeviceInfo(&target_name.header) == ERROR_SUCCESS && std::wstring_view(target_name.monitorDevicePath) == target_device_path) {
          physical_mode_index = path.sourceInfo.modeInfoIdx;
        }
      }

      if (physical_mode_index == std::numeric_limits<size_t>::max() || virtual_mode_index == std::numeric_limits<size_t>::max() || primary_mode_index == std::numeric_limits<size_t>::max() || physical_mode_index == virtual_mode_index) {
        return std::nullopt;
      }
      if (physical_mode_index == primary_mode_index) {
        BOOST_LOG(warning) << "The selected AR display is the Windows primary display; automatic pointer isolation is unavailable."sv;
        return std::nullopt;
      }

      bool have_anchor = false;
      LONG anchor_right = 0;
      LONG anchor_top = 0;
      for (size_t index = 0; index < modes.size(); ++index) {
        if (index == physical_mode_index || index == virtual_mode_index || modes[index].infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE) {
          continue;
        }
        const auto &mode = modes[index].sourceMode;
        const auto right = mode.position.x + (LONG) mode.width;
        if (!have_anchor || right > anchor_right) {
          have_anchor = true;
          anchor_right = right;
          anchor_top = mode.position.y;
        }
      }
      if (!have_anchor) {
        return std::nullopt;
      }

      // Put the virtual source directly beside the rightmost interactive monitor so ordinary
      // mouse movement reaches it. The physical sink is moved beyond a large empty gap.
      auto &virtual_mode = modes[virtual_mode_index].sourceMode;
      auto &physical_mode = modes[physical_mode_index].sourceMode;
      const auto &primary_mode = modes[primary_mode_index].sourceMode;
      // Derive the placement from the live primary-monitor rectangle. A zero-length point contact
      // is considered a disconnected display island, so retain a one-pixel-wide segment at the
      // primary monitor's bottom-right corner. This remains correct when the primary resolution or
      // the surrounding monitor layout changes and does not depend on absolute desktop coordinates.
      const LONG isolated_x = primary_mode.position.x + (LONG) primary_mode.width - 1;
      const LONG isolated_y = primary_mode.position.y + (LONG) primary_mode.height;
      const RECT isolated_rect {
        isolated_x,
        isolated_y,
        isolated_x + (LONG) physical_mode.width,
        isolated_y + (LONG) physical_mode.height,
      };

      const bool already_isolated = virtual_mode.position.x == anchor_right &&
                                    virtual_mode.position.y == anchor_top &&
                                    physical_mode.position.x == isolated_x &&
                                    physical_mode.position.y == isolated_y;
      if (already_isolated) {
        return find_target(target_device_path);
      }

      bool recovery_record_created = false;
      if (!persist_topology_recovery(
            target_device_path,
            original_rect,
            isolated_rect,
            recovery_record_created
          )) {
        BOOST_LOG(error) << "Refusing to move the AR output without a durable topology recovery record."sv;
        return std::nullopt;
      }

      virtual_mode.position.x = anchor_right;
      virtual_mode.position.y = anchor_top;
      physical_mode.position.x = isolated_x;
      physical_mode.position.y = isolated_y;

      const auto status = SetDisplayConfig(
        (UINT32) paths.size(),
        paths.data(),
        (UINT32) modes.size(),
        modes.data(),
        SDC_APPLY | SDC_USE_SUPPLIED_DISPLAY_CONFIG
      );
      if (status != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "Could not isolate the AR presentation output: "sv << status;
        if (recovery_record_created) {
          clear_topology_recovery(target_device_path);
        } else if (const auto current = find_target(target_device_path)) {
          // persist_topology_recovery() updated this existing device's applied rectangle before
          // SetDisplayConfig so a successful move was crash recoverable. The move failed, so put
          // the ownership evidence back to the rectangle that is actually still active.
          bool ignored_created = false;
          if (!persist_topology_recovery(
                target_device_path,
                original_rect,
                current->rect,
                ignored_created
              )) {
            BOOST_LOG(error) << "Could not restore the AR topology journal after the display move failed."sv;
          }
        } else {
          BOOST_LOG(error) << "Could not query the AR output after its display move failed; "sv
                              "retaining conservative topology recovery state."sv;
        }
        return std::nullopt;
      }
      if (layout_changed) {
        *layout_changed = true;
      }

      // Never trust the requested source position. DisplayConfig may normalize a valid request,
      // so query the exact PnP target again and use the coordinates Windows actually applied.
      for (int attempt = 0; attempt < 10; ++attempt) {
        if (const auto actual = find_target(target_device_path)) {
          bool ignored_created = false;
          if (!persist_topology_recovery(
                target_device_path,
                original_rect,
                actual->rect,
                ignored_created
              )) {
            BOOST_LOG(error) << "Could not update the durable AR topology record with Windows' "sv
                                "actual applied rectangle; refusing to retain the isolated layout."sv;
            if (restore_physical_output_position(original_rect, std::wstring(target_device_path))) {
              if (layout_changed) {
                *layout_changed = false;
              }
              if (!clear_topology_recovery(target_device_path)) {
                BOOST_LOG(warning) << "The AR output was restored, but its topology recovery record could not be cleared."sv;
              }
            } else {
              BOOST_LOG(error) << "Could not restore the AR output after its recovery record update failed; "sv
                                  "retaining recovery state for the next launch."sv;
            }
            return std::nullopt;
          }
          const bool applied_exactly = actual->rect.left == isolated_rect.left &&
                                       actual->rect.top == isolated_rect.top;
          if (applied_exactly) {
            BOOST_LOG(info) << "AR presentation output attached to the primary display "sv
                            << "through its bottom-right one-pixel corner segment at ["sv
                            << actual->rect.left << ',' << actual->rect.top << "]."sv;
          } else {
            BOOST_LOG(warning) << "Windows normalized the AR output's one-pixel placement from ["sv
                               << isolated_rect.left << ',' << isolated_rect.top << "] to ["sv
                               << actual->rect.left << ',' << actual->rect.top
                               << "]; pointer isolation is not guaranteed."sv;
          }
          return actual;
        }
        std::this_thread::sleep_for(50ms);
      }
      BOOST_LOG(warning) << "AR-output topology applied, but its actual rectangle could not be queried."sv;
      if (restore_physical_output_position(original_rect, std::wstring(target_device_path))) {
        if (layout_changed) {
          *layout_changed = false;
        }
        if (!clear_topology_recovery(target_device_path)) {
          BOOST_LOG(warning) << "The AR output was restored, but its topology recovery record could not be cleared."sv;
        }
      } else {
        BOOST_LOG(error) << "Could not restore the AR output after its applied rectangle became unobservable; "sv
                            "retaining recovery state for the next launch."sv;
      }
      return std::nullopt;
    }

    bool restore_physical_output_position(
      const RECT &original_rect,
      const std::wstring &target_device_path
    ) {
      std::vector<DISPLAYCONFIG_PATH_INFO> paths;
      std::vector<DISPLAYCONFIG_MODE_INFO> modes;
      if (!VDISPLAY::queryActiveDisplayConfig(paths, modes)) {
        return false;
      }

      bool found = false;
      for (const auto &path : paths) {
        if (path.sourceInfo.modeInfoIdx == DISPLAYCONFIG_PATH_MODE_IDX_INVALID || path.sourceInfo.modeInfoIdx >= modes.size() || modes[path.sourceInfo.modeInfoIdx].infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE) {
          continue;
        }
        DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
        target_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
        target_name.header.size = sizeof(target_name);
        target_name.header.adapterId = path.targetInfo.adapterId;
        target_name.header.id = path.targetInfo.id;
        if (DisplayConfigGetDeviceInfo(&target_name.header) != ERROR_SUCCESS || std::wstring_view(target_name.monitorDevicePath) != target_device_path) {
          continue;
        }
        auto &source_mode = modes[path.sourceInfo.modeInfoIdx].sourceMode;
        source_mode.position.x = original_rect.left;
        source_mode.position.y = original_rect.top;
        found = true;
        break;
      }
      if (!found) {
        return false;
      }

      const auto status = SetDisplayConfig(
        (UINT32) paths.size(),
        paths.data(),
        (UINT32) modes.size(),
        modes.data(),
        SDC_APPLY | SDC_USE_SUPPLIED_DISPLAY_CONFIG
      );
      if (status != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "Restoring the AR display position failed with status "sv << status << '.';
        return false;
      }
      for (int attempt = 0; attempt < 10; ++attempt) {
        if (const auto actual = find_target(target_device_path)) {
          if (same_rect(actual->rect, original_rect)) {
            return true;
          }
        }
        std::this_thread::sleep_for(50ms);
      }
      return false;
    }

    bool migrate_legacy_topology_recovery() {
      std::lock_guard lock(topology_recovery_mutex);
      bool legacy_format {false};
      const auto recoveries {load_topology_recoveries_locked(&legacy_format)};
      if (!recoveries) {
        return false;
      }
      if (!legacy_format) {
        return true;
      }
      if (!write_topology_recoveries_locked(*recoveries)) {
        BOOST_LOG(warning) << "Could not migrate the legacy local-AR topology recovery record; "sv
                              "the original record remains intact."sv;
        return false;
      }
      BOOST_LOG(info) << "Migrated the legacy local-AR topology recovery record to per-device state."sv;
      return true;
    }

    bool recover_saved_topology(std::wstring_view device_path) {
      std::optional<topology_recovery_t> recovery;
      {
        std::lock_guard lock(topology_recovery_mutex);
        const auto recoveries = load_topology_recoveries_locked();
        if (!recoveries) {
          return false;
        }
        const auto match = std::ranges::find_if(*recoveries, [&](const auto &candidate) {
          return candidate.device_path == device_path;
        });
        if (match != recoveries->end()) {
          recovery = *match;
        }
      }
      if (!recovery) {
        return true;
      }

      const auto current = find_target(recovery->device_path);
      if (!current) {
        // Preserve the record while the glasses are disconnected. It will be retried before a
        // future session for this exact PnP target.
        return false;
      }
      if (same_rect(current->rect, recovery->original_rect)) {
        clear_topology_recovery(recovery->device_path);
        return true;
      }
      if (!same_rect(current->rect, recovery->applied_rect)) {
        // The target no longer occupies the exact rectangle Apollo recorded after applying its
        // isolation. A user or Windows topology change now owns the layout; restoring the old
        // coordinates would overwrite that newer choice.
        BOOST_LOG(warning) << "Retiring stale local-AR topology recovery state because the current "sv
                              "display layout no longer matches Apollo's applied rectangle."sv;
        clear_topology_recovery(recovery->device_path);
        return true;
      }
      if (!restore_physical_output_position(recovery->original_rect, recovery->device_path)) {
        BOOST_LOG(warning) << "Could not recover the AR display topology left by an interrupted session."sv;
        return false;
      }

      clear_topology_recovery(recovery->device_path);
      BOOST_LOG(info) << "Recovered the AR display's pre-session desktop position from durable state."sv;
      return true;
    }

    std::size_t recover_connected_saved_topologies() {
      std::vector<topology_recovery_t> recoveries;
      {
        std::lock_guard lock(topology_recovery_mutex);
        const auto loaded = load_topology_recoveries_locked();
        if (!loaded) {
          return 1;
        }
        recoveries = *loaded;
      }

      std::size_t pending = 0;
      for (const auto &recovery : recoveries) {
        if (!find_target(recovery.device_path) || !recover_saved_topology(recovery.device_path)) {
          ++pending;
        }
      }
      return pending;
    }

    bool wait_for_virtual_display_mode(
      std::wstring &display_name,
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view physical_device_path,
      bool expected_hdr,
      std::chrono::milliseconds timeout,
      std::stop_token stop_token
    ) {
      const auto deadline = std::chrono::steady_clock::now() + timeout;
      int stable_observations = 0;
      while (!stop_token.stop_requested() && std::chrono::steady_clock::now() < deadline) {
        bool topology_query_succeeded = false;
        const auto physical_target = find_target(physical_device_path, {}, &topology_query_succeeded);
        if (!topology_query_succeeded) {
          stable_observations = 0;
          std::this_thread::sleep_for(50ms);
          continue;
        }
        if (!physical_target) {
          BOOST_LOG(info) << "AR display disconnected during virtual-display color setup."sv;
          return false;
        }

        const auto refreshed = refresh_virtual_display_name(identity);
        if (!refreshed) {
          stable_observations = 0;
          std::this_thread::sleep_for(50ms);
          continue;
        }
        display_name = *refreshed;

        DEVMODEW mode {};
        const bool geometry_ready = VDISPLAY::getDeviceSettings(display_name.c_str(), mode) &&
                                    mode.dmPelsWidth == source_width &&
                                    mode.dmPelsHeight == source_height;
        const bool hdr_matches = VDISPLAY::getDisplayHDRByName(display_name.c_str()) == expected_hdr;
        if (geometry_ready && hdr_matches) {
          if (++stable_observations >= 3) {
            return true;
          }
        } else {
          stable_observations = 0;
        }
        for (int sleep_step = 0; sleep_step < 4 && !stop_token.stop_requested(); ++sleep_step) {
          std::this_thread::sleep_for(50ms);
        }
      }
      return false;
    }

    bool configure_virtual_display_hdr(
      std::wstring &display_name,
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view physical_device_path,
      bool enable_hdr,
      bool &configured_hdr,
      std::stop_token stop_token
    ) {
      configured_hdr = false;
      // Match the remote-stream workaround: wait for the new mode, force Advanced Color off,
      // then enable it after the first transition has settled. Windows often ignores a direct
      // enable immediately after IddCx output creation or a resolution/topology change.
      if (!wait_for_virtual_display_mode(display_name, identity, physical_device_path, false, 5s, stop_token)) {
        if (stop_token.stop_requested()) {
          return false;
        }
        BOOST_LOG(warning) << "Local AR virtual display did not settle before HDR configuration."sv;
      }
      if (!VDISPLAY::setDisplayHDRByName(display_name.c_str(), false) && VDISPLAY::getDisplayHDRByName(display_name.c_str())) {
        BOOST_LOG(error) << "Could not reset the local AR virtual display to SDR before HDR setup."sv;
        return false;
      }
      if (!wait_for_virtual_display_mode(display_name, identity, physical_device_path, false, 5s, stop_token)) {
        if (stop_token.stop_requested()) {
          return false;
        }
        BOOST_LOG(error) << "Local AR virtual display did not reach a stable SDR state."sv;
        return false;
      }
      if (!enable_hdr) {
        return true;
      }

      if (!VDISPLAY::setDisplayHDRByName(display_name.c_str(), true)) {
        BOOST_LOG(warning) << "Windows rejected HDR for the local AR virtual display; using color-managed SDR presentation."sv;
        return true;
      }
      if (!wait_for_virtual_display_mode(display_name, identity, physical_device_path, true, 15s, stop_token)) {
        if (stop_token.stop_requested()) {
          return false;
        }
        BOOST_LOG(warning) << "Local AR virtual display did not reach a stable HDR state; using color-managed SDR presentation."sv;
        if (!VDISPLAY::setDisplayHDRByName(display_name.c_str(), false) || !wait_for_virtual_display_mode(display_name, identity, physical_device_path, false, 5s, stop_token)) {
          BOOST_LOG(error) << "Local AR virtual display could not recover to SDR after HDR setup failed."sv;
          return false;
        }
        return true;
      }
      configured_hdr = true;
      return true;
    }

    class local_session_t {
    public:
      explicit local_session_t(const target_state_t &target, std::stop_token controller_stop_token):
          original_target_rect_(target.rect),
          target_device_path_(target.device_path) {
        if (proc::vDisplayDriverStatus != VDISPLAY::DRIVER_STATUS::OK) {
          BOOST_LOG(error) << "AR glasses detected, but the SudoVDA driver is unavailable."sv;
          return;
        }
        auto active_target = target;

        if (target.hdr.known) {
          BOOST_LOG(info) << "AR display HDR: supported="sv << target.hdr.supported
                          << " enabled="sv << target.hdr.user_enabled
                          << " active="sv << target.hdr.active
                          << " bits_per_color="sv << target.hdr.bits_per_color << '.';
        } else {
          BOOST_LOG(warning) << "Could not query the AR display's HDR capability; keeping SDR."sv;
        }
        if (target.hdr.known && !target.hdr.supported) {
          BOOST_LOG(info) << "The AR display currently advertises SDR only. If the glasses have an "sv
                          << "internal HDR10 mode, enable it in the glasses menu; Apollo will detect "sv
                          << "the updated capability."sv;
        }

        auto physical_hdr = target.hdr;
        if (physical_hdr.supported && !physical_hdr.limited_by_policy && !physical_hdr.active) {
          if (set_hdr_state(target.adapter_id, target.target_id, true)) {
            // HDR activation can renumber the target or change its mode. Wait for the stable PnP
            // target rather than polling the now-stale source/target IDs from the original path.
            const auto deadline = std::chrono::steady_clock::now() + 10s;
            while (!controller_stop_token.stop_requested() &&
                   std::chrono::steady_clock::now() < deadline && !physical_hdr.active) {
              for (int sleep_step = 0; sleep_step < 4 && !controller_stop_token.stop_requested(); ++sleep_step) {
                std::this_thread::sleep_for(50ms);
              }
              bool topology_query_succeeded = false;
              const auto current = find_target(target_device_path_, {}, &topology_query_succeeded);
              if (!topology_query_succeeded) {
                continue;
              }
              if (!current) {
                BOOST_LOG(info) << "AR display disconnected while enabling HDR."sv;
                return;
              }
              if (current->mode != active_target.mode || current->rect.right - current->rect.left != active_target.rect.right - active_target.rect.left || current->rect.bottom - current->rect.top != active_target.rect.bottom - active_target.rect.top) {
                BOOST_LOG(info) << "AR output mode changed while enabling HDR; waiting for the topology controller to rebuild the session."sv;
                return;
              }
              active_target = *current;
              physical_hdr = active_target.hdr;
            }
            if (controller_stop_token.stop_requested()) {
              return;
            }
            if (!physical_hdr.active) {
              // Some glasses advertise HDR before their on-device HDR10 mode is active. Treat
              // that as a stable SDR presentation state rather than tearing down and recreating
              // the virtual desktop every retry interval.
              BOOST_LOG(warning) << "AR display did not enter HDR after Windows accepted the request; continuing in SDR."sv;
            }
          } else {
            BOOST_LOG(warning) << "The AR display reports HDR support, but Windows rejected HDR activation."sv;
          }
        }
        if (physical_hdr.supported && physical_hdr.limited_by_policy) {
          BOOST_LOG(warning) << "HDR on the AR display is disabled by Windows policy."sv;
        }
        target_hdr_active_ = physical_hdr.active;
        if (physical_hdr.supported && !target_hdr_active_) {
          BOOST_LOG(warning) << "The AR display supports HDR, but HDR is not active in its current mode."sv;
        } else if (target_hdr_active_) {
          BOOST_LOG(info) << "AR display HDR is active at "sv << physical_hdr.bits_per_color
                          << " bits per color."sv;
        }

        std::string uuid_string = virtual_display_uuid;
        auto uuid = uuid_util::uuid_t::parse(uuid_string);
        static_assert(sizeof(display_guid_) == sizeof(uuid));
        std::memcpy(&display_guid_, &uuid, sizeof(display_guid_));

        virtual_display_name_ = VDISPLAY::createVirtualDisplayOnAdapter(
          virtual_display_uuid,
          virtual_display_name,
          source_width,
          source_height,
          active_target.refresh_millihz,
          display_guid_,
          active_target.adapter_id,
          &virtual_display_identity_
        );
        virtual_display_created_ = true;
        if (virtual_display_name_.empty()) {
          BOOST_LOG(error) << "Failed to create the local AR virtual desktop."sv;
          return;
        }

        if (VDISPLAY::changeDisplaySettings(virtual_display_name_.c_str(), source_width, source_height, active_target.refresh_millihz) != DISP_CHANGE_SUCCESSFUL) {
          BOOST_LOG(warning) << "The local AR virtual desktop rejected its requested mode."sv;
        }
        auto presentation_target = active_target;
        if (const auto isolated_target = isolate_physical_output(
              virtual_display_name_,
              target_device_path_,
              original_target_rect_,
              &layout_repositioned_
            )) {
          presentation_target = *isolated_target;
          applied_target_rect_ = isolated_target->rect;
          pointer_isolated_ = true;
        } else {
          BOOST_LOG(warning) << "The AR display remains in the interactive desktop layout; keep the pointer on the Apollo AR virtual desktop."sv;
        }

        // Let DXGI observe the newly attached output before capture initializes.
        for (int sleep_step = 0; sleep_step < 6 && !controller_stop_token.stop_requested(); ++sleep_step) {
          std::this_thread::sleep_for(50ms);
        }
        if (controller_stop_token.stop_requested()) {
          return;
        }
        if (const auto refreshed = refresh_virtual_display_name(virtual_display_identity_)) {
          virtual_display_name_ = *refreshed;
        } else {
          BOOST_LOG(error) << "Could not resolve the newly created local AR virtual display by its driver identity."sv;
          return;
        }
        bool virtual_hdr_active = false;
        if (!configure_virtual_display_hdr(
              virtual_display_name_,
              virtual_display_identity_,
              target_device_path_,
              target_hdr_active_,
              virtual_hdr_active,
              controller_stop_token
            )) {
          if (controller_stop_token.stop_requested()) {
            return;
          }
          BOOST_LOG(error) << "Local AR virtual display color-mode configuration failed; rebuilding the session."sv;
          return;
        }
        BOOST_LOG(info) << "Local AR source color mode: "sv
                        << (virtual_hdr_active ? "HDR linear scRGB"sv : "SDR Rec.709"sv) << '.';
        if (virtual_hdr_active) {
          const auto source_white = query_sdr_white_nits(
            virtual_display_identity_.AdapterLuid,
            virtual_display_identity_.TargetId
          );
          const auto target_white = query_sdr_white_nits(active_target.adapter_id, active_target.target_id);
          BOOST_LOG(info) << "Local AR HDR SDR-reference white: source="sv
                          << (source_white ? std::to_string(*source_white) : "unknown"s)
                          << " nits target="sv
                          << (target_white ? std::to_string(*target_white) : "unknown"s) << " nits."sv;
        }

        // Advanced Color can renumber GDI sources. Refresh the stable physical PnP target before
        // publishing the live rectangle/name consumed by the DXGI presenter.
        if (const auto refreshed_target = find_target(target_device_path_)) {
          const int refreshed_width = refreshed_target->rect.right - refreshed_target->rect.left;
          const int refreshed_height = refreshed_target->rect.bottom - refreshed_target->rect.top;
          const int expected_width = active_target.rect.right - active_target.rect.left;
          const int expected_height = active_target.rect.bottom - active_target.rect.top;
          if (refreshed_width != expected_width || refreshed_height != expected_height || refreshed_target->mode != active_target.mode) {
            BOOST_LOG(info) << "AR output mode changed during virtual HDR setup; waiting for the topology controller."sv;
            return;
          }
          active_target = *refreshed_target;
          presentation_target = *refreshed_target;
        } else {
          BOOST_LOG(error) << "Could not refresh the physical AR output after virtual HDR setup."sv;
          return;
        }

        // Advanced Color frequently lets Windows normalize the physical output back into the
        // interactive row. Reapply isolation after every HDR transition instead of accepting the
        // new position as the presentation target.
        if (const auto isolated_target = isolate_physical_output(
              virtual_display_name_,
              target_device_path_,
              original_target_rect_,
              &layout_repositioned_
            )) {
          presentation_target = *isolated_target;
          applied_target_rect_ = isolated_target->rect;
          pointer_isolated_ = true;
        } else {
          // Isolation is a pointer-safety optimization, not a presentation prerequisite. Primary
          // displays and some clone topologies cannot be moved; keep one stable session alive.
          pointer_isolated_ = false;
          BOOST_LOG(warning) << "Could not isolate the AR output after color-mode setup; presenting without pointer isolation."sv;
        }

        platf::dxgi::local_presenter_config_t presenter_config;
        presenter_config.source_display_name = platf::to_utf8(virtual_display_name_);
        presenter_config.target_rect = presentation_target.rect;
        presenter_config.target_refresh_millihz = presentation_target.refresh_millihz;
        presenter_config.hdr = virtual_hdr_active;
        presenter_config.sbs_mode = presentation_target.mode == presentation_mode_e::sbs_ai ?
                                      ::video::SBS_AI :
                                      ::video::SBS_OFF;
        presenter_config.sbs_config = config::video.sbs;
        live_target_ = std::make_shared<platf::dxgi::local_presenter_config_t::target_t>();
        live_target_->rect = presentation_target.rect;
        live_target_->display_name = presentation_target.gdi_name;
        presenter_config.live_target = live_target_;
        presented_frames_ = std::make_shared<std::atomic<std::uint64_t>>(0);
        presenter_config.presented_frames = presented_frames_;

        running_.store(true);
        presenter_ = std::jthread([this, presenter_config, identity = virtual_display_identity_](std::stop_token stop_token) mutable {
          auto reinit_window_started = std::chrono::steady_clock::now();
          int consecutive_reinits = 0;
          while (!stop_token.stop_requested()) {
            const auto presented_before = presenter_config.presented_frames->load(std::memory_order_relaxed);
            const auto result = platf::dxgi::run_local_presenter(presenter_config, stop_token);
            if (result != platf::dxgi::local_presenter_result_e::reinit) {
              break;
            }

            const auto now = std::chrono::steady_clock::now();
            const auto presented_after = presenter_config.presented_frames->load(std::memory_order_relaxed);
            if (presented_after - presented_before >= 60) {
              // Reset only after the attempt demonstrably presented a stable run of frames. Slow
              // initialization failures no longer evade the retry breaker merely by taking time.
              reinit_window_started = now;
              consecutive_reinits = 0;
            }
            ++consecutive_reinits;
            if (consecutive_reinits >= 12 || now - reinit_window_started >= 10s) {
              BOOST_LOG(error) << "Local AR presenter did not recover after "sv
                               << consecutive_reinits << " reinitializations; rebuilding the session."sv;
              break;
            }

            // The newly attached virtual output can invalidate an already-created DXGI factory.
            // Recreate only the capture/presentation resources; removing the virtual display here
            // would cause another topology change and an endless reinitialization loop.
            std::this_thread::sleep_for(100ms);
            if (const auto refreshed = refresh_virtual_display_name(identity)) {
              const auto refreshed_utf8 = platf::to_utf8(*refreshed);
              if (refreshed_utf8 != presenter_config.source_display_name) {
                BOOST_LOG(info) << "Local AR virtual desktop was renumbered ["sv
                                << presenter_config.source_display_name << " -> "sv
                                << refreshed_utf8 << "]."sv;
                presenter_config.source_display_name = refreshed_utf8;
              }
            } else {
              BOOST_LOG(error) << "Local AR virtual desktop identity disappeared; rebuilding the session."sv;
              break;
            }
          }
          // Any exit not requested by the topology controller should be retried, including a
          // user-closed or driver-closed presenter window that otherwise exits cleanly.
          failed_.store(!stop_token.stop_requested());
          running_.store(false);
        });
        ready_ = true;
      }

      ~local_session_t() {
        presenter_.request_stop();
        if (presenter_.joinable()) {
          presenter_.join();
        }
        bool restore_layout = layout_repositioned_;
        if (restore_layout && applied_target_rect_) {
          const auto current = find_target(target_device_path_);
          if (!current) {
            // Keep the durable record for the next time this PnP target appears.
            restore_layout = false;
          } else if (!same_rect(current->rect, *applied_target_rect_)) {
            // A newer user/Windows topology no longer matches the exact rectangle Apollo owned.
            // Do not overwrite it during teardown, and retire the obsolete recovery contract.
            BOOST_LOG(warning) << "Skipping AR display-position restore because the current layout "sv
                                  "no longer matches Apollo's applied rectangle."sv;
            clear_topology_recovery(target_device_path_);
            restore_layout = false;
          }
        }
        if (virtual_display_created_) {
          if (!VDISPLAY::removeVirtualDisplay(display_guid_)) {
            BOOST_LOG(warning) << "Failed to remove the local AR virtual desktop."sv;
          }
        }
        if (restore_layout) {
          // SudoVDA removal changes the active topology asynchronously. Retry and verify the exact
          // applied rectangle instead of assuming the source has disappeared after a fixed delay.
          const auto deadline = std::chrono::steady_clock::now() + 3s;
          bool restored = false;
          while (std::chrono::steady_clock::now() < deadline && !restored) {
            std::this_thread::sleep_for(100ms);
            restored = restore_physical_output_position(original_target_rect_, target_device_path_);
          }
          if (!restored) {
            BOOST_LOG(warning) << "Could not restore the AR display's original desktop position."sv;
          } else {
            clear_topology_recovery(target_device_path_);
          }
        }
      }

      bool valid() const {
        return ready_;
      }

      bool running() const {
        return running_.load();
      }

      bool failed() const {
        return failed_.load();
      }

      bool stable() const {
        return presented_frames_ && presented_frames_->load(std::memory_order_relaxed) >= 60;
      }

      std::optional<target_state_t> re_isolate_target() {
        const auto refreshed_name = refresh_virtual_display_name(virtual_display_identity_);
        if (!refreshed_name) {
          return std::nullopt;
        }
        virtual_display_name_ = *refreshed_name;
        const auto target = pointer_isolated_ ?
                              isolate_physical_output(
                                virtual_display_name_,
                                target_device_path_,
                                original_target_rect_,
                                &layout_repositioned_
                              ) :
                              find_target(target_device_path_);
        if (!target) {
          return std::nullopt;
        }
        if (pointer_isolated_) {
          applied_target_rect_ = target->rect;
        }
        if (!live_target_) {
          return target;
        }
        std::lock_guard lock(live_target_->mutex);
        live_target_->rect = target->rect;
        live_target_->display_name = target->gdi_name;
        return target;
      }

    private:
      GUID display_guid_ {};
      SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT virtual_display_identity_ {};
      RECT original_target_rect_ {};
      std::wstring target_device_path_;
      std::wstring virtual_display_name_;
      bool virtual_display_created_ = false;
      bool layout_repositioned_ = false;
      bool pointer_isolated_ = false;
      bool target_hdr_active_ = false;
      bool ready_ = false;
      std::optional<RECT> applied_target_rect_;
      std::shared_ptr<platf::dxgi::local_presenter_config_t::target_t> live_target_;
      std::shared_ptr<std::atomic<std::uint64_t>> presented_frames_;
      std::jthread presenter_;
      std::atomic<bool> running_ {false};
      std::atomic<bool> failed_ {false};
    };

    class controller_t final: public platf::deinit_t {
    public:
      controller_t():
          worker_([this](std::stop_token stop_token) {
            run(stop_token);
          }) {
      }

      ~controller_t() override {
        worker_.request_stop();
        if (worker_.joinable()) {
          worker_.join();
        }
      }

    private:
      void reset_failure_backoff() {
        failure_retry_delay_ = failed_session_retry;
      }

      void schedule_failure_retry() {
        retry_after_ = std::chrono::steady_clock::now() + failure_retry_delay_;
        BOOST_LOG(warning) << "Local AR session retry deferred for "sv
                           << std::chrono::duration_cast<std::chrono::seconds>(failure_retry_delay_).count()
                           << " seconds."sv;
        failure_retry_delay_ = std::min(failure_retry_delay_ * 2, maximum_failed_session_retry);
      }

      void stop_session() {
        std::unique_ptr<local_session_t> retiring_session;
        {
          std::lock_guard lock(ownership_mutex);
          if (local_session_construction_stop) {
            local_session_construction_stop->request_stop();
          }
          retiring_session = std::move(session_);
          if (!retiring_session) {
            local_session_present = false;
            local_session_construction_stop.reset();
            ownership_changed.notify_all();
            return;
          }
        }

        // Presenter shutdown and SudoVDA topology removal can take seconds. Keep ownership marked
        // local until teardown is complete, but never hold ownership_mutex across those waits so a
        // remote launch can publish its reservation and wait on the condition variable.
        retiring_session.reset();
        {
          std::lock_guard lock(ownership_mutex);
          local_session_present = false;
          local_session_construction_stop.reset();
          ownership_changed.notify_all();
        }
      }

      void suspend_for_remote_if_needed() {
        bool should_stop = false;
        {
          std::lock_guard lock(ownership_mutex);
          if (!remote_blocks_local_locked(std::chrono::steady_clock::now())) {
            return;
          }
          if (local_session_construction_stop) {
            local_session_construction_stop->request_stop();
          }
          should_stop = session_ != nullptr;
          deferred_for_remote_ = true;
          retry_after_ = std::chrono::steady_clock::now() + failed_session_retry;
        }

        if (should_stop) {
          BOOST_LOG(info) << "Remote virtual-display session requested ownership; stopping local AR presentation."sv;
          stop_session();
        }
      }

      void start_session(const target_state_t &target, std::stop_token stop_token) {
        const auto handoff = proc::proc.prepare_local_ar_handoff();
        if (handoff != proc::local_ar_handoff_e::ready) {
          if (!deferred_for_remote_) {
            BOOST_LOG(info) << (handoff == proc::local_ar_handoff_e::remote_busy ? "Local AR presentation is waiting for the active/connecting remote virtual-display session."sv : "Local AR presentation is waiting for inactive remote-display cleanup."sv);
          }
          deferred_for_remote_ = true;
          retry_after_ = std::chrono::steady_clock::now() + failed_session_retry;
          return;
        }
        if (!recover_saved_topology(target.device_path)) {
          BOOST_LOG(warning) << "Deferring local AR startup until this display's previous desktop topology can be restored."sv;
          schedule_failure_retry();
          return;
        }

        std::stop_source construction_stop;
        std::stop_callback controller_stop_callback(stop_token, [&construction_stop]() {
          construction_stop.request_stop();
        });
        {
          std::lock_guard lock(ownership_mutex);
          if (remote_blocks_local_locked(std::chrono::steady_clock::now())) {
            deferred_for_remote_ = true;
            retry_after_ = std::chrono::steady_clock::now() + failed_session_retry;
            return;
          }

          // Publish local ownership before construction starts. Remote launches can acquire the
          // mutex immediately, request this stop source, and then wait for full teardown.
          local_session_present = true;
          local_session_construction_stop = construction_stop;
          ownership_changed.notify_all();
        }

        auto candidate = std::make_unique<local_session_t>(target, construction_stop.get_token());
        bool rejected_for_remote = false;
        {
          std::lock_guard lock(ownership_mutex);
          local_session_construction_stop.reset();
          rejected_for_remote = remote_blocks_local_locked(std::chrono::steady_clock::now());
          if (candidate->valid() && !construction_stop.stop_requested() && !rejected_for_remote) {
            session_ = std::move(candidate);
            local_session_present = true;
            ownership_changed.notify_all();
            if (deferred_for_remote_) {
              BOOST_LOG(info) << "Remote virtual-display ownership ended; starting deferred local AR presentation."sv;
            }
            deferred_for_remote_ = false;
            session_stability_confirmed_ = false;
            return;
          }
        }

        // Destruction removes any partially created virtual display. Keep local ownership true
        // until that removal and topology restoration finish, so the remote side cannot overlap.
        candidate.reset();
        {
          std::lock_guard lock(ownership_mutex);
          local_session_present = false;
          ownership_changed.notify_all();
          if (rejected_for_remote) {
            deferred_for_remote_ = true;
          }
        }
        if (!rejected_for_remote && !construction_stop.stop_requested()) {
          schedule_failure_retry();
        }
      }

      void apply(const std::optional<target_state_t> &target, std::stop_token stop_token) {
        stop_session();
        reset_failure_backoff();
        applied_ = target;
        if (!target) {
          deferred_for_remote_ = false;
          BOOST_LOG(info) << "Approved AR display disconnected; local presentation is off."sv;
          return;
        }

        const int width = target->rect.right - target->rect.left;
        const int height = target->rect.bottom - target->rect.top;
        if (target->mode == presentation_mode_e::unsupported) {
          BOOST_LOG(warning) << "AR display mode "sv << width << 'x' << height
                             << " is unsupported; expected 1920x1080 or 3840x1080."sv;
          return;
        }

        BOOST_LOG(info) << "Approved AR display ["sv << target->friendly_name << "] detected at "sv
                        << width << 'x' << height << '@'
                        << (target->refresh_millihz / 1000.0) << "; starting "sv
                        << mode_name(target->mode) << " local presentation."sv;
        start_session(*target, stop_token);
      }

      void run(std::stop_token stop_token) {
        std::optional<target_state_t> pending;
        auto pending_since = std::chrono::steady_clock::now();

        while (!stop_token.stop_requested()) {
          suspend_for_remote_if_needed();

          const std::wstring_view preferred_device_path = applied_ ?
                                                            std::wstring_view(applied_->device_path) :
                                                            std::wstring_view {};
          bool topology_query_succeeded = false;
          auto observed = find_target({}, preferred_device_path, &topology_query_succeeded);
          if (!topology_query_succeeded) {
            // A failed query is not a disconnect. Retain the previous observation and live session;
            // Windows commonly reports an undersized snapshot while applying display topology.
            std::this_thread::sleep_for(topology_poll_interval);
            continue;
          }
          const auto now = std::chrono::steady_clock::now();
          if (observed != pending) {
            pending = observed;
            pending_since = now;
            // Stop using old-size textures immediately. Recreate only after the new topology has
            // remained stable for the debounce interval.
            if (observed != applied_ && !same_presentation_contract(observed, applied_)) {
              stop_session();
            }
          }

          if (pending != applied_ && now - pending_since >= topology_debounce) {
            if (pending && applied_ && same_presentation_contract(pending, applied_)) {
              if (session_) {
                const auto isolated = session_->re_isolate_target();
                if (!isolated) {
                  BOOST_LOG(warning) << "AR display position drifted and could not be re-isolated; rebuilding the local session."sv;
                  apply(pending, stop_token);
                  continue;
                }
                pending = isolated;
                pending_since = now;
              }
              applied_ = pending;
              BOOST_LOG(info) << "AR display position changed; restored physical-output isolation without recreating its virtual desktop."sv;
            } else {
              apply(pending, stop_token);
            }
          } else if (pending == applied_ && session_ && !session_stability_confirmed_ &&
                     session_->stable()) {
            // Construction only proves that a presenter thread was spawned. Reset exponential
            // retry backoff after sustained scanout so permanent DXGI/swapchain failures cannot
            // recreate the whole topology every two seconds forever.
            session_stability_confirmed_ = true;
            reset_failure_backoff();
          } else if (pending == applied_ && session_ && !session_->running() && session_->failed() && now >= retry_after_) {
            BOOST_LOG(warning) << "Local AR presentation failed; scheduling a clean restart."sv;
            stop_session();
            schedule_failure_retry();
          } else if (pending == applied_ && applied_ && !session_ && applied_->mode != presentation_mode_e::unsupported && now >= retry_after_) {
            start_session(*applied_, stop_token);
          }

          std::this_thread::sleep_for(topology_poll_interval);
        }

        stop_session();
      }

      std::jthread worker_;
      std::optional<target_state_t> applied_;
      std::unique_ptr<local_session_t> session_;
      std::chrono::steady_clock::time_point retry_after_ {};
      std::chrono::seconds failure_retry_delay_ {failed_session_retry};
      bool deferred_for_remote_ = false;
      bool session_stability_confirmed_ = false;
    };
  }  // namespace

  bool is_recognized_ar_display(std::string_view model_id, std::string_view friendly_name) {
    if (is_internal_virtual_display(model_id, friendly_name)) {
      return false;
    }
    const auto model = lowercase(model_id);
    const auto name = lowercase(friendly_name);
    if (model == "display:tcl03d4") {
      return true;
    }

    static constexpr std::string_view specific_markers[] = {
      "smartglasses",
      "smart glasses",
      "ar glasses",
      "xr glasses",
      "xreal",
      "nreal air",
      "viture",
      "rokid",
      "rayneo",
      "nxtwear",
    };
    return std::any_of(std::begin(specific_markers), std::end(specific_markers), [&](const auto marker) {
      return name.find(marker) != std::string::npos;
    });
  }

  std::vector<device_info_t> devices() {
    std::lock_guard lock(device_mutex);
    return {known_devices.begin(), known_devices.end()};
  }

  bool set_device_decision(std::string_view id, device_decision_e decision) {
    if (decision == device_decision_e::pending) {
      return false;
    }
    std::lock_guard lock(device_mutex);
    const auto device = std::find_if(known_devices.begin(), known_devices.end(), [&](const auto &candidate) {
      return candidate.id == id;
    });
    if (device == known_devices.end()) {
      return false;
    }
    if (device->decision != decision || device->auto_detected) {
      device->decision = decision;
      device->auto_detected = false;
      persist_devices_locked();
      BOOST_LOG(info) << "AR display decision updated: ["sv << device->name << ", "sv << device->id
                      << "] is "sv << decision_name(decision) << '.';
    }
    return true;
  }

  bool write_config_with_devices(std::string_view contents) {
    std::lock_guard lock(device_mutex);
    auto merged = replace_managed_config_value(std::string(contents), serialize_devices_locked());
    return write_config_atomically(merged);
  }

  bool remote_virtual_display_starting(std::chrono::milliseconds connect_timeout) {
    std::unique_lock lock(ownership_mutex);
    const auto now = std::chrono::steady_clock::now();
    const auto release_deadline = now + ownership_release_timeout;
    const auto pending_duration = remote_pending_duration(connect_timeout);
    remote_session_pending_until = std::max(remote_session_pending_until, now + pending_duration);
    if (local_session_construction_stop) {
      local_session_construction_stop->request_stop();
    }
    ownership_changed.notify_all();

    if (ownership_changed.wait_until(lock, release_deadline, []() {
          return !local_session_present;
        })) {
      // Teardown time must not consume the connection reservation. Give the remote path its full
      // configured connection window after local ownership has actually been released.
      remote_session_pending_until = std::max(
        remote_session_pending_until,
        std::chrono::steady_clock::now() + pending_duration
      );
      return true;
    }

    if (!remote_session_active) {
      remote_session_pending_until = {};
    }
    ownership_changed.notify_all();
    BOOST_LOG(error) << "Local AR did not release virtual-display ownership within "sv
                     << ownership_release_timeout.count() << " seconds."sv;
    return false;
  }

  void remote_virtual_display_awaiting_client(std::chrono::milliseconds connect_timeout) {
    std::lock_guard lock(ownership_mutex);
    // Creating the virtual display, probing encoders, and running application preparation all
    // happen while proc_t owns its process lock and can legitimately outlive the initial lease.
    // Start a fresh connection window only after that work succeeds, before the process lock is
    // released and local AR is allowed to inspect ownership again.
    remote_session_pending_until = std::max(
      remote_session_pending_until,
      std::chrono::steady_clock::now() + remote_pending_duration(connect_timeout)
    );
    ownership_changed.notify_all();
  }

  void remote_virtual_display_active() {
    std::lock_guard lock(ownership_mutex);
    remote_session_active = true;
    remote_session_pending_until = {};
    ownership_changed.notify_all();
  }

  void remote_virtual_display_ended() {
    std::lock_guard lock(ownership_mutex);
    remote_session_active = false;
    remote_session_pending_until = {};
    ownership_changed.notify_all();
  }

  bool remote_virtual_display_blocks_local() {
    std::lock_guard lock(ownership_mutex);
    return remote_blocks_local_locked(std::chrono::steady_clock::now());
  }

  std::unique_ptr<platf::deinit_t> init() {
    load_devices();
    migrate_legacy_topology_recovery();
    const auto pending_recoveries = recover_connected_saved_topologies();
    if (pending_recoveries != 0) {
      BOOST_LOG(info) << pending_recoveries
                      << " saved AR topology recovery record(s) remain pending until their display reconnects."sv;
    }
    return std::make_unique<controller_t>();
  }
}  // namespace ar_glasses
