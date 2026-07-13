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

    struct target_state_t {
      std::string device_id;
      std::wstring device_path;
      std::string friendly_name;
      std::string gdi_name;
      std::string desktop_topology;
      RECT rect {};
      int refresh_millihz = 60000;
      presentation_mode_e mode = presentation_mode_e::unsupported;

      bool operator==(const target_state_t &other) const {
        return device_id == other.device_id && device_path == other.device_path &&
               gdi_name == other.gdi_name && desktop_topology == other.desktop_topology &&
               rect.left == other.rect.left && rect.top == other.rect.top &&
               rect.right == other.rect.right && rect.bottom == other.rect.bottom &&
               refresh_millihz == other.refresh_millihz && mode == other.mode;
      }
    };

    struct stored_device_t: device_info_t {
      bool notified = false;
    };

    std::mutex device_mutex;
    std::vector<stored_device_t> known_devices;

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
             left->desktop_topology == right->desktop_topology;
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

    std::vector<target_state_t> enumerate_targets() {
      UINT32 path_count = 0;
      UINT32 mode_count = 0;
      if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) != ERROR_SUCCESS) {
        return {};
      }

      std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
      std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
      if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count, modes.data(), nullptr) != ERROR_SUCCESS) {
        return {};
      }

      paths.resize(path_count);
      modes.resize(mode_count);
      const auto virtual_sources = VDISPLAY::matchDisplay(virtual_display_driver_name);
      std::vector<target_state_t> targets;
      for (const auto &path : paths) {
        // IddCx/SudoVDA outputs are virtual sources, never physical AR presentation sinks.
        if (path.targetInfo.outputTechnology == DISPLAYCONFIG_OUTPUT_TECHNOLOGY_INDIRECT_WIRED) {
          continue;
        }
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
      std::wstring_view preferred_device_path = {}
    ) {
      auto targets = enumerate_targets();
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

    std::optional<std::wstring> refresh_virtual_display_name(std::string_view preferred_name) {
      const auto matches = VDISPLAY::matchDisplay(virtual_display_driver_name);
      std::vector<std::wstring> matching_modes;
      for (const auto &candidate : matches) {
        DEVMODEW mode {};
        if (!VDISPLAY::getDeviceSettings(candidate.c_str(), mode) || mode.dmPelsWidth != source_width || mode.dmPelsHeight != source_height) {
          continue;
        }
        if (platf::to_utf8(candidate) == preferred_name) {
          return candidate;
        }
        matching_modes.emplace_back(candidate);
      }

      if (matching_modes.size() == 1) {
        return matching_modes.front();
      }
      return std::nullopt;
    }

    std::optional<target_state_t> isolate_physical_output(
      const std::wstring &virtual_gdi_name,
      const std::wstring &target_device_path
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

      UINT32 path_count = 0;
      UINT32 mode_count = 0;
      if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) != ERROR_SUCCESS) {
        return std::nullopt;
      }

      std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
      std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
      if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count, modes.data(), nullptr) != ERROR_SUCCESS) {
        return std::nullopt;
      }
      paths.resize(path_count);
      modes.resize(mode_count);

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
      virtual_mode.position.x = anchor_right;
      virtual_mode.position.y = anchor_top;

      auto &physical_mode = modes[physical_mode_index].sourceMode;
      const auto &primary_mode = modes[primary_mode_index].sourceMode;
      // Derive the placement from the live primary-monitor rectangle. A zero-length point contact
      // is considered a disconnected display island, so retain a one-pixel-wide segment at the
      // primary monitor's bottom-right corner. This remains correct when the primary resolution or
      // the surrounding monitor layout changes and does not depend on absolute desktop coordinates.
      physical_mode.position.x = primary_mode.position.x + (LONG) primary_mode.width - 1;
      physical_mode.position.y = primary_mode.position.y + (LONG) primary_mode.height;
      const RECT isolated_rect {
        physical_mode.position.x,
        physical_mode.position.y,
        physical_mode.position.x + (LONG) physical_mode.width,
        physical_mode.position.y + (LONG) physical_mode.height,
      };

      const auto status = SetDisplayConfig(
        (UINT32) paths.size(),
        paths.data(),
        (UINT32) modes.size(),
        modes.data(),
        SDC_APPLY | SDC_USE_SUPPLIED_DISPLAY_CONFIG
      );
      if (status != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "Could not isolate the AR presentation output: "sv << status;
        return std::nullopt;
      }

      // Never trust the requested source position. DisplayConfig may normalize a valid request,
      // so query the exact PnP target again and use the coordinates Windows actually applied.
      for (int attempt = 0; attempt < 10; ++attempt) {
        if (const auto actual = find_target(target_device_path)) {
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
      return std::nullopt;
    }

    bool restore_physical_output_position(
      const RECT &original_rect,
      const std::wstring &target_device_path
    ) {
      UINT32 path_count = 0;
      UINT32 mode_count = 0;
      if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) != ERROR_SUCCESS) {
        return false;
      }

      std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
      std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
      if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count, modes.data(), nullptr) != ERROR_SUCCESS) {
        return false;
      }
      paths.resize(path_count);
      modes.resize(mode_count);

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
      }
      return status == ERROR_SUCCESS;
    }

    class local_session_t {
    public:
      explicit local_session_t(const target_state_t &target):
          original_target_rect_(target.rect),
          target_device_path_(target.device_path) {
        if (proc::vDisplayDriverStatus != VDISPLAY::DRIVER_STATUS::OK) {
          BOOST_LOG(error) << "AR glasses detected, but the SudoVDA driver is unavailable."sv;
          return;
        }

        if (!config::video.adapter_name.empty()) {
          VDISPLAY::setRenderAdapterByName(platf::from_utf8(config::video.adapter_name));
        }

        std::string uuid_string = virtual_display_uuid;
        auto uuid = uuid_util::uuid_t::parse(uuid_string);
        static_assert(sizeof(display_guid_) == sizeof(uuid));
        std::memcpy(&display_guid_, &uuid, sizeof(display_guid_));

        virtual_display_name_ = VDISPLAY::createVirtualDisplay(
          virtual_display_uuid,
          virtual_display_name,
          source_width,
          source_height,
          target.refresh_millihz,
          display_guid_
        );
        virtual_display_created_ = true;
        if (virtual_display_name_.empty()) {
          BOOST_LOG(error) << "Failed to create the local AR virtual desktop."sv;
          return;
        }

        if (VDISPLAY::changeDisplaySettings(virtual_display_name_.c_str(), source_width, source_height, target.refresh_millihz) != DISP_CHANGE_SUCCESSFUL) {
          BOOST_LOG(warning) << "The local AR virtual desktop rejected its requested mode."sv;
        }
        if (VDISPLAY::getDisplayHDRByName(virtual_display_name_.c_str()) && !VDISPLAY::setDisplayHDRByName(virtual_display_name_.c_str(), false)) {
          BOOST_LOG(warning) << "The local AR virtual desktop could not be switched to SDR."sv;
        }

        auto presentation_target = target;
        if (const auto isolated_target = isolate_physical_output(virtual_display_name_, target_device_path_)) {
          presentation_target = *isolated_target;
          layout_repositioned_ = true;
        } else {
          BOOST_LOG(warning) << "The AR display remains in the interactive desktop layout; keep the pointer on the Apollo AR virtual desktop."sv;
        }

        // Let DXGI observe the newly attached output before capture initializes.
        std::this_thread::sleep_for(300ms);
        if (const auto refreshed = refresh_virtual_display_name(platf::to_utf8(virtual_display_name_))) {
          virtual_display_name_ = *refreshed;
        }

        platf::dxgi::local_presenter_config_t presenter_config;
        presenter_config.source_display_name = platf::to_utf8(virtual_display_name_);
        presenter_config.target_rect = presentation_target.rect;
        presenter_config.target_refresh_millihz = target.refresh_millihz;
        presenter_config.sbs_mode = target.mode == presentation_mode_e::sbs_ai ?
                                      ::video::SBS_AI :
                                      ::video::SBS_OFF;
        presenter_config.sbs_config = config::video.sbs;
        live_target_ = std::make_shared<platf::dxgi::local_presenter_config_t::target_t>();
        live_target_->rect = presentation_target.rect;
        live_target_->display_name = presentation_target.gdi_name;
        presenter_config.live_target = live_target_;

        running_.store(true);
        presenter_ = std::jthread([this, presenter_config](std::stop_token stop_token) mutable {
          while (!stop_token.stop_requested()) {
            const auto result = platf::dxgi::run_local_presenter(presenter_config, stop_token);
            if (result != platf::dxgi::local_presenter_result_e::reinit) {
              break;
            }

            // The newly attached virtual output can invalidate an already-created DXGI factory.
            // Recreate only the capture/presentation resources; removing the virtual display here
            // would cause another topology change and an endless reinitialization loop.
            std::this_thread::sleep_for(100ms);
            if (const auto refreshed = refresh_virtual_display_name(presenter_config.source_display_name)) {
              const auto refreshed_utf8 = platf::to_utf8(*refreshed);
              if (refreshed_utf8 != presenter_config.source_display_name) {
                BOOST_LOG(info) << "Local AR virtual desktop was renumbered ["sv
                                << presenter_config.source_display_name << " -> "sv
                                << refreshed_utf8 << "]."sv;
                presenter_config.source_display_name = refreshed_utf8;
              }
            }
          }
          // Any exit not requested by the topology controller should be retried, including a
          // user-closed or driver-closed presenter window that otherwise exits cleanly.
          failed_.store(!stop_token.stop_requested());
          running_.store(false);
        });
      }

      ~local_session_t() {
        presenter_.request_stop();
        if (presenter_.joinable()) {
          presenter_.join();
        }
        if (virtual_display_created_) {
          if (!VDISPLAY::removeVirtualDisplay(display_guid_)) {
            BOOST_LOG(warning) << "Failed to remove the local AR virtual desktop."sv;
          }
        }
        if (layout_repositioned_) {
          // SudoVDA removal changes the active topology asynchronously; wait until the virtual
          // source has vacated the interactive position before restoring the physical sink.
          std::this_thread::sleep_for(300ms);
          if (!restore_physical_output_position(original_target_rect_, target_device_path_)) {
            BOOST_LOG(warning) << "Could not restore the AR display's original desktop position."sv;
          }
        }
      }

      bool valid() const {
        return virtual_display_created_ && !virtual_display_name_.empty();
      }

      bool running() const {
        return running_.load();
      }

      bool failed() const {
        return failed_.load();
      }

      void update_target(const target_state_t &target) {
        if (!live_target_) {
          return;
        }
        std::lock_guard lock(live_target_->mutex);
        live_target_->rect = target.rect;
        live_target_->display_name = target.gdi_name;
      }

    private:
      GUID display_guid_ {};
      RECT original_target_rect_ {};
      std::wstring target_device_path_;
      std::wstring virtual_display_name_;
      bool virtual_display_created_ = false;
      bool layout_repositioned_ = false;
      std::shared_ptr<platf::dxgi::local_presenter_config_t::target_t> live_target_;
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
      void start_session(const target_state_t &target) {
        session_ = std::make_unique<local_session_t>(target);
        retry_after_ = std::chrono::steady_clock::now() + failed_session_retry;
        if (!session_->valid()) {
          session_.reset();
        }
      }

      void apply(const std::optional<target_state_t> &target) {
        session_.reset();
        applied_ = target;
        if (!target) {
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
        start_session(*target);
      }

      void run(std::stop_token stop_token) {
        std::optional<target_state_t> pending;
        auto pending_since = std::chrono::steady_clock::now();

        while (!stop_token.stop_requested()) {
          const std::wstring_view preferred_device_path = applied_ ?
                                                            std::wstring_view(applied_->device_path) :
                                                            std::wstring_view {};
          auto observed = find_target({}, preferred_device_path);
          const auto now = std::chrono::steady_clock::now();
          if (observed != pending) {
            pending = observed;
            pending_since = now;
            // Stop using old-size textures immediately. Recreate only after the new topology has
            // remained stable for the debounce interval.
            if (observed != applied_ && !same_presentation_contract(observed, applied_)) {
              session_.reset();
            }
          }

          if (pending != applied_ && now - pending_since >= topology_debounce) {
            if (pending && applied_ && same_presentation_contract(pending, applied_)) {
              applied_ = pending;
              if (session_) {
                session_->update_target(*pending);
              }
              BOOST_LOG(info) << "AR display position changed; moved the existing local presentation without recreating its virtual desktop."sv;
            } else {
              apply(pending);
            }
          } else if (pending == applied_ && session_ && !session_->running() && session_->failed() && now >= retry_after_) {
            BOOST_LOG(warning) << "Restarting failed local AR presentation."sv;
            auto target = applied_;
            session_.reset();
            retry_after_ = now + failed_session_retry;
            if (target) {
              start_session(*target);
            }
          } else if (pending == applied_ && applied_ && !session_ && applied_->mode != presentation_mode_e::unsupported && now >= retry_after_) {
            start_session(*applied_);
          }

          std::this_thread::sleep_for(topology_poll_interval);
        }

        session_.reset();
      }

      std::jthread worker_;
      std::optional<target_state_t> applied_;
      std::unique_ptr<local_session_t> session_;
      std::chrono::steady_clock::time_point retry_after_ {};
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

  std::unique_ptr<platf::deinit_t> init() {
    load_devices();
    return std::make_unique<controller_t>();
  }
}  // namespace ar_glasses
