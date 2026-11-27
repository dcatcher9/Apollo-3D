/**
 * @file src/platform/windows/ar_glasses_devices.cpp
 * @brief The AR device registry: saved monitor decisions, target discovery and the approved-target
 * selection.
 */
#include "ar_glasses_devices.h"

#ifdef SUNSHINE_TESTS
  #include "ar_glasses_discovery_test.h"

  #include <thread>
#endif

#include "ar_glasses_requests.h"
#include "misc.h"
#include "src/config.h"
#include "src/logging.h"
#include "src/system_tray.h"
#include "src/utility.h"
#include "virtual_display.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cwctype>
#include <iterator>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace ar_glasses {
  using namespace std::literals;
  using namespace internal;

  namespace {
#ifdef SUNSHINE_TESTS
    struct discovery_test_state_t {
      const testing::discovery_hooks_t &hooks;
      testing::discovery_costs_t &costs;
    };

    thread_local discovery_test_state_t *active_discovery_test = nullptr;

    LONG WINAPI discovery_get_info_for_test(DISPLAYCONFIG_DEVICE_INFO_HEADER *header) {
      if (!active_discovery_test) {
        throw std::logic_error("discovery device-info trampoline has no scoped owner");
      }
      return active_discovery_test->hooks.get_info(header);
    }

    platf::display_config::device_info_api_t discovery_info_api_for_test() {
      if (active_discovery_test) {
        return {.get = &discovery_get_info_for_test, .set = nullptr};
      }
      return {};
    }
#endif

    constexpr wchar_t virtual_display_driver_name[] = L"SudoMaker Virtual Display Adapter";

    hdr_state_t query_hdr_state(const LUID &adapter_id, UINT32 target_id) {
#ifdef SUNSHINE_TESTS
      if (active_discovery_test) {
        ++active_discovery_test->costs.hdr_queries;
      }
#endif
      const auto state = platf::display_config::query_advanced_color(
        adapter_id,
        target_id
#ifdef SUNSHINE_TESTS
        ,
        discovery_info_api_for_test()
#endif
      );
      if (!state) {
        return {};
      }
      return {
        true,
        state->hdr_supported,
        state->hdr_user_enabled,
        state->active_mode == DISPLAYCONFIG_ADVANCED_COLOR_MODE_HDR,
        state->limited_by_policy,
        state->bits_per_color_channel,
      };
    }

    bool device_persistence_dirty = false;
    std::chrono::steady_clock::time_point device_persistence_retry_after {};

    std::string lowercase(std::string_view value) {
      std::string result(value);
      std::transform(result.begin(), result.end(), result.begin(), [](unsigned char ch) {
        return (char) std::tolower(ch);
      });
      return result;
    }

    // Saved AR decisions name a monitor by its model ID, the SudoVDA hardware ID for the
    // driver's targets, and by its friendly name.
    bool is_internal_virtual_display(std::string_view device_id, std::string_view friendly_name) {
      return device_id == "DISPLAY:SMKD1CE"sv || is_apollo_ar_virtual_name(platf::from_utf8(friendly_name));
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

    bool persist_devices_locked() {
#ifdef SUNSHINE_TESTS
      if (active_discovery_test) {
        ++active_discovery_test->costs.persistence_attempts;
      }
#endif
      auto contents = read_config_file_strict();
      if (!contents) {
        BOOST_LOG(error) << "Could not read the complete configuration file; refusing to replace it while persisting the AR display decision list ["sv
                         << config::sunshine.config_file << "]."sv;
        return false;
      }
      auto updated = replace_managed_config_value(std::move(*contents), serialize_devices_locked());
      if (!write_config_atomically(updated)) {
        BOOST_LOG(error) << "Could not persist the AR display decision list to "sv
                         << config::sunshine.config_file << '.';
        return false;
      }
      return true;
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

    template<class Decisions>
    bool preserve_during_remote_virtual_display_policy(
      std::string_view model_id,
      std::string_view friendly_name,
      const Decisions &decisions
    ) {
      if (model_id.empty() || lowercase(model_id) == "display:0:0" || is_internal_virtual_display(model_id, friendly_name)) {
        return false;
      }

      const auto normalized_id = lowercase(model_id);
      bool approved = false;
      for (const auto &device : decisions) {
        if (lowercase(device.id) != normalized_id) {
          continue;
        }
        if (device.decision == device_decision_e::rejected) {
          return false;
        }
        approved = approved || device.decision == device_decision_e::approved;
      }
      return approved || is_recognized_ar_display(model_id, friendly_name);
    }

    bool primary_source_is_authoritative(
      std::wstring_view primary_gdi_name,
      const std::vector<std::wstring> &active_gdi_names
    ) {
      return !primary_gdi_name.empty() &&
             std::ranges::any_of(active_gdi_names, [&](const auto &active_name) {
               return equal_device_paths(active_name, primary_gdi_name);
             });
    }

    bool query_discovery_paths(
      std::vector<DISPLAYCONFIG_PATH_INFO> &paths,
      std::vector<DISPLAYCONFIG_MODE_INFO> &modes
    ) {
#ifdef SUNSHINE_TESTS
      if (active_discovery_test) {
        return active_discovery_test->hooks.query_active(paths, modes);
      }
#endif
      platf::display_config::topology_t topology;
      if (!platf::display_config::query_topology(QDC_ONLY_ACTIVE_PATHS, topology)) {
        return false;
      }
      paths = std::move(topology.paths);
      modes = std::move(topology.modes);
      return true;
    }

    std::optional<DISPLAYCONFIG_TARGET_DEVICE_NAME> discovery_target_name(const DISPLAYCONFIG_PATH_INFO &path) {
      DISPLAYCONFIG_TARGET_DEVICE_NAME target {};
      target.header = {DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME, sizeof(target), path.targetInfo.adapterId, path.targetInfo.id};
#ifdef SUNSHINE_TESTS
      const auto status = discovery_info_api_for_test().get(&target.header);
#else
      const auto status = DisplayConfigGetDeviceInfo(&target.header);
#endif
      if (status != ERROR_SUCCESS) {
        return std::nullopt;
      }
      return target;
    }

    std::optional<RECT> discovery_source_rect(
      const DISPLAYCONFIG_PATH_INFO &path,
      const std::vector<DISPLAYCONFIG_MODE_INFO> &modes
    ) {
      const auto index = path.sourceInfo.modeInfoIdx;
      if (index >= modes.size()) {
        return std::nullopt;
      }
      const auto &mode = modes[index];
      if (mode.infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE || mode.id != path.sourceInfo.id || !same_luid(mode.adapterId, path.sourceInfo.adapterId)) {
        return std::nullopt;
      }
      const auto &source = mode.sourceMode;
      const auto right = static_cast<std::int64_t>(source.position.x) + source.width;
      const auto bottom = static_cast<std::int64_t>(source.position.y) + source.height;
      if (!source.width || !source.height || !std::in_range<LONG>(source.width) || !std::in_range<LONG>(source.height) || !std::in_range<LONG>(right) || !std::in_range<LONG>(bottom)) {
        return std::nullopt;
      }
      return RECT {source.position.x, source.position.y, static_cast<LONG>(right), static_cast<LONG>(bottom)};
    }
  }  // namespace

  namespace internal {
    bool is_apollo_ar_virtual_name(std::wstring_view value) {
      constexpr std::wstring_view prefix = L"apollo ar des";
      return value.size() >= prefix.size() &&
             std::equal(
               prefix.begin(),
               prefix.end(),
               value.begin(),
               [](wchar_t left, wchar_t right) {
                 return std::towlower(left) == std::towlower(right);
               }
             );
    }

    int target_source_width(const target_state_t &target) {
      return is_monitor_mode(target.mode) ? target.rect.right - target.rect.left : source_width;
    }

    int target_source_height(const target_state_t &target) {
      return is_monitor_mode(target.mode) ? target.rect.bottom - target.rect.top : source_height;
    }

    bool target_exclusive(const target_state_t &target) {
      return is_monitor_mode(target.mode) || config::video.local_ar_virtual_display_only;
    }

    std::vector<stored_device_t> known_devices;

    void load_devices(bool persist_cleanup) {
      std::lock_guard lock(device_mutex);
      const auto contents = read_config_file_strict();
      if (!contents) {
        BOOST_LOG(warning) << "Could not read the complete configuration file; leaving the AR display decision list unchanged."sv;
        return;
      }
      try {
        std::vector<stored_device_t> loaded_devices;
        bool removed_internal_display = false;
        const auto vars = config::parse_config(*contents);
        const auto option = vars.find(std::string(managed_config_key));
        if (option != vars.end() && !option->second.empty()) {
          const auto value = nlohmann::json::parse(option->second);
          if (!value.is_array()) {
            throw std::runtime_error("AR display decision list is not an array");
          }
          for (const auto &entry : value) {
            if (!entry.is_object()) {
              throw std::runtime_error("AR display decision entry is not an object");
            }
            stored_device_t device;
            device.id = entry.value("id", "");
            device.name = entry.value("name", device.id);
            device.decision = decision_from_name(entry.value("decision", "pending"));
            device.auto_detected = entry.value("auto_detected", false);
            if (is_internal_virtual_display(device.id, device.name)) {
              removed_internal_display = true;
              continue;
            }
            if (!device.id.empty() && std::none_of(loaded_devices.begin(), loaded_devices.end(), [&](const auto &existing) {
                  return existing.id == device.id;
                })) {
              loaded_devices.emplace_back(std::move(device));
            }
          }
        }

        known_devices = std::move(loaded_devices);
        device_persistence_dirty = false;
        device_persistence_retry_after = {};
        if (removed_internal_display) {
          device_persistence_dirty = true;
          if (persist_cleanup) {
            if (persist_devices_locked()) {
              device_persistence_dirty = false;
              device_persistence_retry_after = {};
              BOOST_LOG(info) << "Removed Sunshine 3D's internal virtual desktop from the AR display decision list."sv;
            } else {
              device_persistence_retry_after = std::chrono::steady_clock::now() + 2s;
            }
          }
        }
      } catch (const std::exception &error) {
        BOOST_LOG(warning) << "Ignoring invalid "sv << managed_config_key << " configuration: "sv
                           << error.what();
      }
    }

    std::optional<target_discovery_snapshot_t> discover_targets(
      std::vector<DISPLAYCONFIG_PATH_INFO> paths,
      const std::vector<DISPLAYCONFIG_MODE_INFO> &modes
    ) {
#ifdef SUNSHINE_TESTS
      const auto virtual_sources = active_discovery_test ? active_discovery_test->hooks.match_display(virtual_display_driver_name) : VDISPLAY::matchDisplay(virtual_display_driver_name);
#else
      const auto virtual_sources = VDISPLAY::matchDisplay(virtual_display_driver_name);
#endif
      std::vector<discovered_target_t> targets;
      std::vector<std::wstring> active_gdi_names;
      active_gdi_names.reserve(paths.size());
      for (const auto &path : paths) {
        DISPLAYCONFIG_SOURCE_DEVICE_NAME source_name {};
        source_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
        source_name.header.size = sizeof(source_name);
        source_name.header.adapterId = path.sourceInfo.adapterId;
        source_name.header.id = path.sourceInfo.id;
#ifdef SUNSHINE_TESTS
        if ((active_discovery_test ? discovery_get_info_for_test(&source_name.header) : DisplayConfigGetDeviceInfo(&source_name.header)) != ERROR_SUCCESS) {
#else
        if (DisplayConfigGetDeviceInfo(&source_name.header) != ERROR_SUCCESS) {
#endif
          // An active path that cannot be described makes the whole observation indeterminate.
          // Treating a partial snapshot as authoritative can look like a hot-unplug and destroy a
          // healthy presentation session during HDR/topology transitions.
          return std::nullopt;
        }
        active_gdi_names.emplace_back(source_name.viewGdiDeviceName);
        const auto virtual_source = std::find_if(virtual_sources.begin(), virtual_sources.end(), [&](const auto &name) {
          return equal_device_paths(source_name.viewGdiDeviceName, name);
        });
        if (virtual_source != virtual_sources.end()) {
          continue;
        }

        const auto target = discovery_target_name(path);
        if (!target) {
          return std::nullopt;
        }
        const auto &target_name = *target;
        if (VDISPLAY::isSudoVirtualDisplayPath(target_name.monitorDevicePath)) {
          continue;
        }
        const auto device_id = stable_model_id(target_name);
        const auto friendly_name = platf::to_utf8(target_name.monitorFriendlyDeviceName);
        if (is_internal_virtual_display(device_id, friendly_name)) {
          continue;
        }

        const auto rect = discovery_source_rect(path, modes);
        if (!rect) {
          return std::nullopt;
        }

        const auto &source_mode = modes[path.sourceInfo.modeInfoIdx].sourceMode;
        const auto &refresh = path.targetInfo.refreshRate;
        int refresh_millihz = 60000;
        if (refresh.Numerator && refresh.Denominator) {
          refresh_millihz = (int) std::lround(
            (double) refresh.Numerator * 1000.0 / (double) refresh.Denominator
          );
        }

        discovered_target_t state;
        state.device_id = device_id;
        state.device_path = target_name.monitorDevicePath;
        state.friendly_name = friendly_name;
        if (state.friendly_name.empty()) {
          state.friendly_name = state.device_id;
        }
        state.gdi_name = platf::to_utf8(source_name.viewGdiDeviceName);
        state.source_adapter_id = path.sourceInfo.adapterId;
        state.source_id = path.sourceInfo.id;
        state.adapter_id = path.targetInfo.adapterId;
        state.target_id = path.targetInfo.id;
        state.rect = *rect;
        state.refresh_millihz = refresh_millihz;
        state.mode = classify_mode(source_mode.width, source_mode.height);
        state.game_source = ar_uses_game_source();
        if (config::video.local_monitor.enabled && physical_monitor_requested.load() && equal_device_paths(state.device_path, platf::from_utf8(config::video.local_monitor.display_id))) {
          state.mode = classify_monitor_mode(source_mode.width, source_mode.height, refresh_millihz,
            monitor_uses_game_source());
          state.game_source = false;
        }
        targets.emplace_back(std::move(state));
      }

      std::wstring primary_gdi_name;
      for (DWORD index = 0;; ++index) {
        DISPLAY_DEVICEW device {};
        device.cb = sizeof(device);
#ifdef SUNSHINE_TESTS
        if (!(active_discovery_test ? active_discovery_test->hooks.enum_devices(nullptr, index, &device, 0) : EnumDisplayDevicesW(nullptr, index, &device, 0))) {
#else
        if (!EnumDisplayDevicesW(nullptr, index, &device, 0)) {
#endif
          break;
        }
        if ((device.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE) != 0) {
          primary_gdi_name = device.DeviceName;
          break;
        }
      }
      if (!primary_source_is_authoritative(primary_gdi_name, active_gdi_names)) {
        // Primary/clone status is a destructive-transition safety gate. If Windows cannot provide
        // a primary source belonging to this complete active-path snapshot, the observation is
        // indeterminate rather than evidence that every selectable output is non-primary.
        return std::nullopt;
      }
      std::string primary_device_path;
      for (const auto &target : targets) {
        if (equal_device_paths(platf::from_utf8(target.gdi_name), primary_gdi_name)) {
          primary_device_path = platf::to_utf8(target.device_path);
          break;
        }
      }
      std::sort(targets.begin(), targets.end(), [](const auto &left, const auto &right) {
        return std::tie(left.device_id, left.device_path) < std::tie(right.device_id, right.device_path);
      });

      return target_discovery_snapshot_t {
        .targets = std::move(targets),
        .paths = std::move(paths),
        .primary_gdi_name = std::move(primary_gdi_name),
        .primary_device_path = std::move(primary_device_path),
      };
    }

    std::optional<target_discovery_snapshot_t> discover_targets() {
      std::vector<DISPLAYCONFIG_PATH_INFO> paths;
      std::vector<DISPLAYCONFIG_MODE_INFO> modes;
      if (!query_discovery_paths(paths, modes)) {
        return std::nullopt;
      }
      return discover_targets(std::move(paths), modes);
    }

    // The desktop safety check only needs the current screen rectangle of the already selected
    // physical instance. Full target discovery still owns startup and topology decisions.
    std::optional<RECT> monitor_watchdog_rect(std::wstring_view device_path) {
      std::vector<DISPLAYCONFIG_PATH_INFO> paths;
      std::vector<DISPLAYCONFIG_MODE_INFO> modes;
      if (device_path.empty() || !query_discovery_paths(paths, modes)) {
        return std::nullopt;
      }

      std::optional<RECT> rect;
      for (const auto &path : paths) {
        if (!(path.flags & DISPLAYCONFIG_PATH_ACTIVE)) {
          continue;
        }
        const auto target = discovery_target_name(path);
        if (!target) {
          // A virtual source may lack target metadata. Apply the existing source exclusion
          // and completeness rules to this same snapshot when a target name is unresolved.
          const auto snapshot = discover_targets(std::move(paths), modes);
          if (!snapshot) {
            return std::nullopt;
          }
          rect.reset();
          for (const auto &row : snapshot->targets) {
            if (equal_device_paths(row.device_path, device_path)) {
              if (rect) {
                return std::nullopt;
              }
              rect = row.rect;
            }
          }
          return rect;
        }
        if (!equal_device_paths(device_path, target->monitorDevicePath)) {
          continue;
        }
        if (rect) {
          return std::nullopt;
        }
        rect = discovery_source_rect(path, modes);
        if (!rect) {
          return std::nullopt;
        }
      }
      return rect;
    }

    // One call-local snapshot supplies identity, geometry, primary evidence and every clone path.
    target_state_t materialize_target(const target_discovery_snapshot_t &snapshot, std::size_t index) {
      target_state_t target;
      static_cast<discovered_target_t &>(target) = snapshot.targets[index];
      target.hdr = query_hdr_state(target.adapter_id, target.target_id);
      const auto &paths = snapshot.paths;
      const auto &targets = snapshot.targets;
      const auto &primary_gdi_name = snapshot.primary_gdi_name;
      const auto &primary_device_path = snapshot.primary_device_path;
      target.is_primary = equal_device_paths(platf::from_utf8(target.gdi_name), primary_gdi_name);
      // Inspect every active CCD path, including Apollo/Sudo sources excluded from selection.
      // A physical target cloned with the local virtual source is still unsafe to reconfigure.
      target.is_cloned = std::ranges::any_of(paths, [&](const auto &candidate) {
        return same_luid(candidate.sourceInfo.adapterId, target.source_adapter_id) &&
               candidate.sourceInfo.id == target.source_id &&
               (!same_luid(candidate.targetInfo.adapterId, target.adapter_id) ||
                candidate.targetInfo.id != target.target_id);
      });
      std::vector<std::string> neighbors;
      for (const auto &neighbor : targets) {
        if (equal_device_paths(neighbor.device_path, target.device_path)) {
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
#ifdef SUNSHINE_TESTS
      if (active_discovery_test) {
        ++active_discovery_test->costs.fingerprints;
      }
#endif
      target.desktop_topology = fingerprint.str();
      return target;
    }

    std::optional<target_state_t> find_target(
      std::wstring_view required_device_path,
      std::wstring_view preferred_device_path,
      bool *query_succeeded,
      std::vector<target_state_t> *approved_targets
    ) {
      if (query_succeeded) {
        *query_succeeded = false;
      }
      auto targets_result = discover_targets();
      if (!targets_result) {
        return std::nullopt;
      }
      if (query_succeeded) {
        *query_succeeded = true;
      }
      const auto &snapshot = *targets_result;
      const auto &targets = snapshot.targets;
      if (!required_device_path.empty()) {
        const auto match = std::find_if(targets.begin(), targets.end(), [&](const auto &target) {
          return equal_device_paths(target.device_path, required_device_path);
        });
        return match == targets.end() ? std::nullopt :
                                        std::make_optional(materialize_target(snapshot, static_cast<std::size_t>(std::distance(targets.begin(), match))));
      }

      // Explicit monitor selection chooses a surface by its exact instance. An unplug must not
      // silently transfer the emitter to another monitor or activate a nearby pair of glasses.
      if (config::video.local_monitor.enabled && physical_monitor_requested.load()) {
        const auto match = std::find_if(targets.begin(), targets.end(), [](const auto &target) {
          return equal_device_paths(target.device_path, platf::from_utf8(config::video.local_monitor.display_id));
        });
        if (match == targets.end()) {
          return std::nullopt;
        }
        auto target = materialize_target(snapshot, static_cast<std::size_t>(std::distance(targets.begin(), match)));
        if (approved_targets) {
          approved_targets->push_back(target);
        }
        return target;
      }

      std::vector<std::pair<std::string, std::string>> notify_devices;
      std::vector<std::size_t> approved_indices;
      std::optional<std::size_t> selected_index;
      bool changed = false;
      {
        std::lock_guard lock(device_mutex);
        for (auto &device : known_devices) {
          device.connected = false;
        }
        for (std::size_t index = 0; index < targets.size(); ++index) {
          const auto &target = targets[index];
          // A configured shutter monitor is an explicitly activated surface, even if its model
          // was approved as glasses in an older configuration. Stop must remain a 2D exit.
          if (config::video.local_monitor.enabled && equal_device_paths(target.device_path, platf::from_utf8(config::video.local_monitor.display_id))) {
            continue;
          }
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
          // Discovery and approval continue for the settings page; only a requested AR session
          // may select the glasses as a presentation target.
          if (known->decision == device_decision_e::approved && ar_session_allowed()) {
            // Capture this row's decision now: a later same-model row may promote its policy.
            approved_indices.push_back(index);
            if (!selected_index || equal_device_paths(target.device_path, preferred_device_path)) {
              selected_index = index;
            }
          }
        }
        device_persistence_dirty = device_persistence_dirty || changed;
        const auto persistence_now = std::chrono::steady_clock::now();
        if (device_persistence_dirty && persistence_now >= device_persistence_retry_after) {
          if (persist_devices_locked()) {
            device_persistence_dirty = false;
            device_persistence_retry_after = {};
          } else {
            device_persistence_retry_after = persistence_now + 2s;
          }
        }
      }

      std::optional<target_state_t> selected;
      for (const auto index : approved_indices) {
        // Keep native color queries outside the decision lock, before notification callbacks.
        auto target = materialize_target(snapshot, index);
        if (selected_index == index) {
          selected = target;
        }
        if (approved_targets) {
          approved_targets->push_back(std::move(target));
        }
      }

      for (const auto &[id, name] : notify_devices) {
#ifdef SUNSHINE_TESTS
        if (active_discovery_test ? active_discovery_test->hooks.notify(name) : system_tray::update_tray_ar_display_decision(name)) {
#else
        if (system_tray::update_tray_ar_display_decision(name)) {
#endif
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
  }  // namespace internal

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

  std::vector<monitor_info_t> monitor_devices() {
    std::vector<monitor_info_t> result;
    const auto snapshot = discover_targets();
    if (!snapshot) {
      return result;
    }
    for (const auto &target : snapshot->targets) {
      result.push_back({platf::to_utf8(target.device_path), target.friendly_name, static_cast<int>(target.rect.right - target.rect.left), static_cast<int>(target.rect.bottom - target.rect.top), target.refresh_millihz, equal_device_paths(platf::from_utf8(target.gdi_name), snapshot->primary_gdi_name)});
    }
    return result;
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
      const auto previous_decision = device->decision;
      const bool previous_auto_detected = device->auto_detected;
      const bool previous_persistence_dirty = device_persistence_dirty;
      device->decision = decision;
      device->auto_detected = false;
      if (!persist_devices_locked()) {
        device->decision = previous_decision;
        device->auto_detected = previous_auto_detected;
        device_persistence_dirty = previous_persistence_dirty;
        return false;
      }
      device_persistence_dirty = false;
      device_persistence_retry_after = {};
      BOOST_LOG(info) << "AR display decision updated: ["sv << device->name << ", "sv << device->id
                      << "] is "sv << decision_name(decision) << '.';
    }
    return true;
  }

  bool preserve_during_remote_virtual_display(const LUID &adapter_id, UINT32 target_id) {
    DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
    target_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
    target_name.header.size = sizeof(target_name);
    target_name.header.adapterId = adapter_id;
    target_name.header.id = target_id;
    if (DisplayConfigGetDeviceInfo(&target_name.header) != ERROR_SUCCESS) {
      return false;
    }

    const auto model_id = stable_model_id(target_name);
    const auto friendly_name = platf::to_utf8(target_name.monitorFriendlyDeviceName);
    std::lock_guard lock(device_mutex);
    return preserve_during_remote_virtual_display_policy(model_id, friendly_name, known_devices);
  }

  bool write_config_with_devices(std::string_view contents) {
    std::lock_guard lock(device_mutex);
    auto merged = replace_managed_config_value(std::string(contents), serialize_devices_locked());
    return write_config_atomically(merged);
  }

  void load_preservation_policy() {
    load_devices(false);
  }

#ifdef SUNSHINE_TESTS
  bool detail::primary_source_is_authoritative_for_test(
    std::wstring_view primary_gdi_name,
    const std::vector<std::wstring> &active_gdi_names
  ) {
    return primary_source_is_authoritative(primary_gdi_name, active_gdi_names);
  }

  bool detail::preserve_during_remote_virtual_display_for_test(
    std::string_view model_id,
    std::string_view friendly_name,
    const std::vector<device_info_t> &decisions
  ) {
    return preserve_during_remote_virtual_display_policy(model_id, friendly_name, decisions);
  }

  namespace testing {
    class discovery_session_t::impl_t {
    public:
      impl_t(const std::filesystem::path &path, discovery_hooks_t hooks, const discovery_request_t &request):
          hooks_(std::move(hooks)),
          previous_config_(platf::to_utf8(path.native())),
          previous_surface_(config::video.local_surface),
          previous_monitor_(config::video.local_monitor),
          previous_ar_requested_(ar_session_requested.load()),
          previous_monitor_requested_(physical_monitor_requested.load()) {
        if (!path.is_absolute() || !std::filesystem::is_regular_file(std::filesystem::symlink_status(path)) || !hooks_.query_active || !hooks_.match_display || !hooks_.get_info || !hooks_.enum_devices || !hooks_.notify) {
          throw std::invalid_argument("discovery fixture requires an owned absolute config file and every finite native hook");
        }
        config::video_t::local_monitor_t monitor;
        monitor.enabled = !request.monitor_device_path.empty();
        monitor.display_id = platf::to_utf8(request.monitor_device_path);
        {
          std::lock_guard lock(device_mutex);
          previous_devices_ = std::move(known_devices);
          previous_dirty_ = device_persistence_dirty;
          previous_retry_ = device_persistence_retry_after;
          known_devices.clear();
          device_persistence_dirty = false;
          device_persistence_retry_after = {};
          config::sunshine.config_file.swap(previous_config_);
        }
        auto restore_on_failure = util::fail_guard([this]() {
          restore();
        });
        // Own the real activation inputs; discovery must not bypass the production Start gate.
        config::video.local_surface = monitor.enabled ? config::video_t::local_surface_e::shutter_monitor : config::video_t::local_surface_e::ar_glasses;
        config::video.local_monitor = std::move(monitor);
        ar_session_requested.store(request.ar_requested);
        physical_monitor_requested.store(request.monitor_requested);
        reload_decisions();
        restore_on_failure.disable();
      }

      ~impl_t() {
        restore();
      }

      void reload_decisions() {
        scoped_call([&]() {
          load_devices(false);
        });
      }

      discovery_result_t find(std::wstring_view required, std::wstring_view preferred, bool collect_approved) {
        discovery_result_t result;
        scoped_call([&]() {
          // Every collecting call supplies an existing prefix to the actual append-only API.
          target_state_t prefix;
          prefix.device_id = "TEST_APPROVED_PREFIX";
          prefix.device_path = L"TEST_APPROVED_PREFIX_PATH";
          prefix.friendly_name = "Existing caller friendly name";
          prefix.gdi_name = "Existing caller GDI name";
          prefix.desktop_topology = "Existing caller topology";
          prefix.source_adapter_id = {0x12345, -31};
          prefix.source_id = 901;
          prefix.adapter_id = {0x6789A, -47};
          prefix.target_id = 902;
          prefix.hdr = {true, true, true, true, true, 13};
          prefix.rect = {-33, 22, 1444, 1199};
          prefix.refresh_millihz = 59123;
          prefix.mode = presentation_mode_e::sbs_ai;
          prefix.is_primary = true;
          prefix.is_cloned = true;
          // This owned tuple deliberately includes fields omitted by presentation operator==.
          const auto all_fields = [](const target_state_t &target) {
            return std::make_tuple(
              target.device_id,
              target.device_path,
              target.friendly_name,
              target.gdi_name,
              target.desktop_topology,
              target.source_adapter_id.LowPart,
              target.source_adapter_id.HighPart,
              target.source_id,
              target.adapter_id.LowPart,
              target.adapter_id.HighPart,
              target.target_id,
              target.hdr.known,
              target.hdr.supported,
              target.hdr.user_enabled,
              target.hdr.active,
              target.hdr.limited_by_policy,
              target.hdr.bits_per_color,
              target.rect.left,
              target.rect.top,
              target.rect.right,
              target.rect.bottom,
              target.refresh_millihz,
              target.mode,
              target.is_primary,
              target.is_cloned
            );
          };
          const auto original_prefix = all_fields(prefix);
          std::vector<target_state_t> approved {prefix};
          const auto selected = find_target(required, preferred, &result.query_succeeded, collect_approved ? &approved : nullptr);
          result.approved_prefix_preserved = !approved.empty() && all_fields(approved.front()) == original_prefix;
          if (selected) {
            result.selected = summarize(*selected);
          }
          for (std::size_t index = result.approved_prefix_preserved ? 1 : 0; index < approved.size(); ++index) {
            result.approved.push_back(summarize(approved[index]));
          }
        });
        return result;
      }

      discovery_costs_t costs() const {
        return costs_;
      }

      std::optional<RECT> watchdog_rect(std::wstring_view device_path) {
        std::optional<RECT> result;
        scoped_call([&]() {
          result = monitor_watchdog_rect(device_path);
        });
        return result;
      }

    private:
      template<class Call>
      void scoped_call(Call &&call) {
        discovery_test_state_t state {hooks_, costs_};
        auto *previous = std::exchange(active_discovery_test, &state);
        auto restore_hooks = util::fail_guard([previous]() {
          active_discovery_test = previous;
        });
        call();
      }

      static discovery_target_t summarize(const target_state_t &target) {
        return {
          .device_id = target.device_id,
          .device_path = target.device_path,
          .friendly_name = target.friendly_name,
          .gdi_name = target.gdi_name,
          .desktop_topology = target.desktop_topology,
          .source_adapter_id = target.source_adapter_id,
          .source_id = target.source_id,
          .adapter_id = target.adapter_id,
          .target_id = target.target_id,
          .hdr = {target.hdr.known, target.hdr.supported, target.hdr.user_enabled, target.hdr.active, target.hdr.limited_by_policy, target.hdr.bits_per_color},
          .rect = target.rect,
          .refresh_millihz = target.refresh_millihz,
          .mode = target.mode,
          .is_primary = target.is_primary,
          .is_cloned = target.is_cloned,
        };
      }

      void restore() {
        std::lock_guard lock(device_mutex);
        known_devices.swap(previous_devices_);
        device_persistence_dirty = previous_dirty_;
        device_persistence_retry_after = previous_retry_;
        config::sunshine.config_file.swap(previous_config_);
        config::video.local_surface = previous_surface_;
        std::swap(config::video.local_monitor, previous_monitor_);
        ar_session_requested.store(previous_ar_requested_);
        physical_monitor_requested.store(previous_monitor_requested_);
      }

      discovery_hooks_t hooks_;
      discovery_costs_t costs_;
      std::string previous_config_;
      config::video_t::local_surface_e previous_surface_;
      config::video_t::local_monitor_t previous_monitor_;
      bool previous_ar_requested_;
      bool previous_monitor_requested_;
      std::vector<stored_device_t> previous_devices_;
      bool previous_dirty_ = false;
      std::chrono::steady_clock::time_point previous_retry_ {};
    };

    discovery_session_t::discovery_session_t(const std::filesystem::path &path, discovery_hooks_t hooks, discovery_request_t request):
        impl_(std::make_unique<impl_t>(path, std::move(hooks), request)) {
    }

    discovery_session_t::~discovery_session_t() = default;

    void discovery_session_t::reload_decisions() {
      impl_->reload_decisions();
    }

    discovery_result_t discovery_session_t::find(std::wstring_view required, std::wstring_view preferred, bool collect_approved) {
      return impl_->find(required, preferred, collect_approved);
    }

    std::optional<RECT> discovery_session_t::watchdog_rect(std::wstring_view device_path) {
      return impl_->watchdog_rect(device_path);
    }

    discovery_costs_t discovery_session_t::costs() const {
      return impl_->costs();
    }

    bool discovery_mutex_available_for_test() {
      bool available = false;
      std::thread observer([&]() {
        std::unique_lock lock(device_mutex, std::try_to_lock);
        available = lock.owns_lock();
      });
      observer.join();
      return available;
    }
  }  // namespace testing
#endif
}  // namespace ar_glasses
