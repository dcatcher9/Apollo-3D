#include "ar_glasses.h"

#include "display.h"
#include "misc.h"
#include "src/config.h"
#include "src/logging.h"
#include "src/process.h"
#include "src/system_tray.h"
#include "src/utility.h"
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
#include <functional>
#include <iterator>
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
    constexpr auto incompatible_transition_grace = 5s;
    constexpr auto virtual_display_absence_grace = 5s;
    constexpr unsigned virtual_display_absence_confirmations = 3;
    constexpr auto failed_session_retry = 2s;
    constexpr auto maximum_failed_session_retry = 30s;
    constexpr auto ownership_release_timeout = 10s;

    std::mutex ownership_mutex;
    std::condition_variable ownership_changed;
    bool local_session_present = false;
    std::optional<remote_virtual_display_lease_t> remote_session_active_lease;
    struct pending_remote_session_t {
      remote_virtual_display_lease_t lease = 0;
      std::chrono::steady_clock::time_point until {};
      bool handoff_in_progress = false;
      bool setup_in_progress = false;
    };
    std::optional<pending_remote_session_t> remote_session_pending;
    std::optional<std::stop_source> local_session_construction_stop;

    struct retired_local_virtual_display_t {
      GUID guid {};
      SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT identity {};
      std::wstring device_path;
      std::wstring gdi_name;
      bool was_published = false;
      std::chrono::steady_clock::time_point retirement_started {};
      std::function<bool()> prepare_removal;
      std::function<bool()> finish_topology_cleanup;
    };

    std::optional<retired_local_virtual_display_t> retired_local_virtual_display;

    void expire_remote_pending_locked(std::chrono::steady_clock::time_point now) {
      if (remote_session_pending && !remote_session_pending->handoff_in_progress &&
          !remote_session_pending->setup_in_progress &&
          now >= remote_session_pending->until) {
        remote_session_pending.reset();
      }
    }

    bool remote_blocks_local_locked(std::chrono::steady_clock::time_point now) {
      expire_remote_pending_locked(now);
      return remote_session_active_lease.has_value() || remote_session_pending.has_value();
    }

    bool pending_remote_lease_matches_locked(remote_virtual_display_lease_t lease) {
      expire_remote_pending_locked(std::chrono::steady_clock::now());
      return remote_session_pending && remote_session_pending->lease == lease;
    }

    void clear_remote_lease_locked(remote_virtual_display_lease_t lease) {
      if (remote_session_active_lease == lease) {
        remote_session_active_lease.reset();
      }
      if (remote_session_pending && remote_session_pending->lease == lease) {
        remote_session_pending.reset();
      }
    }

    bool same_luid(const LUID &left, const LUID &right) {
      return left.HighPart == right.HighPart && left.LowPart == right.LowPart;
    }

    bool same_virtual_display_identity(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &left,
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &right
    ) {
      return same_luid(left.AdapterLuid, right.AdapterLuid) && left.TargetId == right.TargetId;
    }

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

    bool contains_case_insensitive(std::wstring_view haystack, std::wstring_view needle);

    bool matches_retiring_local_virtual_display(
      const retired_local_virtual_display_t &retiring,
      const LUID &adapter_id,
      UINT32 target_id,
      std::wstring_view device_path,
      std::wstring_view friendly_name
    ) {
      const bool local_friendly_name = is_apollo_ar_virtual_name(friendly_name);
      const bool sudo_hardware_path = contains_case_insensitive(device_path, L"SMKD1CE") ||
                                      contains_case_insensitive(device_path, L"SUDOVDA");
      const bool learned_path = !retiring.device_path.empty() && retiring.device_path == device_path;
      const bool exact_local_identity = same_luid(retiring.identity.AdapterLuid, adapter_id) &&
                                        retiring.identity.TargetId == target_id &&
                                        (local_friendly_name || sudo_hardware_path);
      // DISPLAY numbers and friendly names are recyclable. Only the exact learned device path, or
      // the driver's exact adapter/target identity plus Apollo/Sudo evidence, can keep this
      // retirement barrier attached to a candidate.
      return learned_path || exact_local_identity;
    }

    void release_local_virtual_display_claim_impl() {
      std::lock_guard lock(ownership_mutex);
      local_session_present = false;
      local_session_construction_stop.reset();
      ownership_changed.notify_all();
    }

    bool begin_local_virtual_display_retirement(
      const GUID &guid,
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring device_path,
      std::wstring gdi_name,
      bool was_published,
      std::function<bool()> prepare_removal,
      std::function<bool()> finish_topology_cleanup
    ) {
      std::lock_guard lock(ownership_mutex);
      if (retired_local_virtual_display) {
        if (same_virtual_display_identity(retired_local_virtual_display->identity, identity)) {
          return true;
        }
        BOOST_LOG(error) << "Refusing to overwrite a different local AR virtual-display retirement record."sv;
        return false;
      }
      retired_local_virtual_display = retired_local_virtual_display_t {
        guid,
        identity,
        std::move(device_path),
        std::move(gdi_name),
        was_published,
        std::chrono::steady_clock::now(),
        std::move(prepare_removal),
        std::move(finish_topology_cleanup),
      };
      ownership_changed.notify_all();
      return true;
    }

    VDISPLAY::display_identity_state_e query_retiring_local_virtual_display(
      const retired_local_virtual_display_t &retiring
    ) {
      std::vector<DISPLAYCONFIG_PATH_INFO> paths;
      std::vector<DISPLAYCONFIG_MODE_INFO> modes;
      if (!VDISPLAY::queryActiveDisplayConfig(paths, modes)) {
        return VDISPLAY::display_identity_state_e::indeterminate;
      }

      for (const auto &path : paths) {
        DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
        target_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
        target_name.header.size = sizeof(target_name);
        target_name.header.adapterId = path.targetInfo.adapterId;
        target_name.header.id = path.targetInfo.id;
        if (DisplayConfigGetDeviceInfo(&target_name.header) != ERROR_SUCCESS) {
          return VDISPLAY::display_identity_state_e::indeterminate;
        }

        if (matches_retiring_local_virtual_display(
              retiring,
              path.targetInfo.adapterId,
              path.targetInfo.id,
              target_name.monitorDevicePath,
              target_name.monitorFriendlyDeviceName
            )) {
          return VDISPLAY::display_identity_state_e::present;
        }
      }
      return VDISPLAY::display_identity_state_e::absent;
    }

    bool wait_for_local_virtual_display_retirement_impl(
      std::chrono::milliseconds timeout,
      bool retry_remove
    ) {
      std::optional<retired_local_virtual_display_t> retiring;
      {
        std::lock_guard lock(ownership_mutex);
        retiring = retired_local_virtual_display;
      }
      if (!retiring) {
        return true;
      }

      const auto deadline = std::chrono::steady_clock::now() + std::max(timeout, 0ms);
      int consecutive_absent_observations = 0;
      bool removal_requested = !retry_remove;
      bool first_observation = true;
      while (first_observation || std::chrono::steady_clock::now() < deadline) {
        first_observation = false;
        if (!removal_requested &&
            (!retiring->prepare_removal || retiring->prepare_removal())) {
          if (VDISPLAY::removeVirtualDisplay(retiring->guid)) {
            removal_requested = true;
          } else {
            BOOST_LOG(warning) << "Failed to request removal of the retiring local AR virtual desktop; retrying while its ownership barrier remains active."sv;
          }
        }
        const auto identity_state = query_retiring_local_virtual_display(*retiring);
        if (identity_state == VDISPLAY::display_identity_state_e::absent) {
          ++consecutive_absent_observations;
          const bool quarantine_complete = retiring->was_published ||
                                           std::chrono::steady_clock::now() - retiring->retirement_started >= topology_debounce;
          if (consecutive_absent_observations >= 3 && quarantine_complete) {
            // Give Windows one additional topology notification interval after stable absence so a
            // late removal event cannot renumber the replacement display created immediately next.
            if (std::chrono::steady_clock::now() + 100ms > deadline) {
              return false;
            }
            std::this_thread::sleep_for(100ms);
            const auto settled_state = query_retiring_local_virtual_display(*retiring);
            if (settled_state != VDISPLAY::display_identity_state_e::absent) {
              consecutive_absent_observations = 0;
              continue;
            }
            // Retirement and topology restoration are one ownership barrier. Keep this record
            // alive if cleanup is transiently blocked so a later local/remote handoff retries the
            // exact pre-removal decision instead of forgetting a user-owned rectangle.
            if (retiring->finish_topology_cleanup) {
              if (!retiring->finish_topology_cleanup()) {
                return false;
              }
              // Latch successful cleanup before another topology query. That query can be
              // transiently indeterminate because restoring the physical rectangle itself emits
              // display notifications; a later waiter must certify absence without replaying the
              // old rectangle over a newer user move.
              std::lock_guard lock(ownership_mutex);
              if (!retired_local_virtual_display) {
                return true;
              }
              if (!same_virtual_display_identity(
                    retired_local_virtual_display->identity,
                    retiring->identity
                  )) {
                return false;
              }
              retired_local_virtual_display->finish_topology_cleanup = {};
            }
            const auto cleaned_state = query_retiring_local_virtual_display(*retiring);
            if (cleaned_state != VDISPLAY::display_identity_state_e::absent) {
              return false;
            }
            std::lock_guard lock(ownership_mutex);
            if (!retired_local_virtual_display) {
              return true;
            }
            if (same_virtual_display_identity(
                  retired_local_virtual_display->identity,
                  retiring->identity
                )) {
              retired_local_virtual_display.reset();
              ownership_changed.notify_all();
              return true;
            }
            BOOST_LOG(error) << "A different local AR retirement record appeared while waiting; refusing to certify cleanup."sv;
            return false;
          }
        } else {
          // Both a confirmed presence and an indeterminate Windows topology query break the
          // authoritative-absence sequence. Query failure must never be treated as removal.
          consecutive_absent_observations = 0;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          break;
        }
        std::this_thread::sleep_for(std::min(
          50ms,
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now())
        ));
      }
      return false;
    }

    bool physical_adapter_contract_valid(const LUID &before, const LUID &after) {
      return same_luid(before, after);
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
      LUID source_adapter_id {};
      UINT32 source_id = 0;
      LUID adapter_id {};
      UINT32 target_id = 0;
      hdr_state_t hdr;
      RECT rect {};
      int refresh_millihz = 60000;
      presentation_mode_e mode = presentation_mode_e::unsupported;
      bool is_primary = false;
      bool is_cloned = false;

      bool operator==(const target_state_t &other) const {
        return device_id == other.device_id && device_path == other.device_path &&
               gdi_name == other.gdi_name && desktop_topology == other.desktop_topology &&
               same_luid(adapter_id, other.adapter_id) &&
               rect.left == other.rect.left && rect.top == other.rect.top &&
               rect.right == other.rect.right && rect.bottom == other.rect.bottom &&
               refresh_millihz == other.refresh_millihz && mode == other.mode &&
               is_primary == other.is_primary && is_cloned == other.is_cloned &&
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
    bool device_persistence_dirty = false;
    std::chrono::steady_clock::time_point device_persistence_retry_after {};
    std::mutex topology_recovery_mutex;

    struct topology_recovery_t {
      std::wstring device_path;
      RECT original_rect {};
      std::vector<RECT> owned_rects;
      std::optional<RECT> pending_rect;
    };

    struct topology_recovery_document_t {
      std::vector<topology_recovery_t> recoveries;
      bool rewrite_required = false;
    };

    bool same_rect(const RECT &left, const RECT &right) {
      return left.left == right.left && left.top == right.top &&
             left.right == right.right && left.bottom == right.bottom;
    }

    struct linear_layout_t {
      RECT virtual_rect {};
      RECT physical_rect {};
    };

    linear_layout_t compute_linear_layout(
      const RECT &anchor,
      LONG virtual_width,
      LONG virtual_height,
      LONG physical_width,
      LONG physical_height
    ) {
      linear_layout_t layout;
      layout.virtual_rect = {
        anchor.right,
        anchor.top,
        anchor.right + virtual_width,
        anchor.top + virtual_height,
      };
      layout.physical_rect = {
        layout.virtual_rect.right,
        anchor.top,
        layout.virtual_rect.right + physical_width,
        anchor.top + physical_height,
      };
      return layout;
    }

    std::filesystem::path topology_recovery_path() {
      auto path = std::filesystem::path(platf::from_utf8(config::sunshine.config_file));
      path += L".apollo-ar-topology.json";
      return path;
    }

    bool valid_rect(const RECT &rect) {
      return rect.right > rect.left && rect.bottom > rect.top;
    }

    RECT parse_rect(const nlohmann::json &value) {
      RECT rect {};
      rect.left = value.at("left").get<LONG>();
      rect.top = value.at("top").get<LONG>();
      rect.right = value.at("right").get<LONG>();
      rect.bottom = value.at("bottom").get<LONG>();
      if (!valid_rect(rect)) {
        throw std::runtime_error("invalid rectangle");
      }
      return rect;
    }

    nlohmann::json serialize_rect(const RECT &rect) {
      return {
        {"left", rect.left},
        {"top", rect.top},
        {"right", rect.right},
        {"bottom", rect.bottom},
      };
    }

    topology_recovery_t parse_legacy_topology_recovery(const nlohmann::json &value) {
      topology_recovery_t recovery {};
      recovery.device_path = platf::from_utf8(value.at("device_path").get<std::string>());
      recovery.original_rect.left = value.at("original_left").get<LONG>();
      recovery.original_rect.top = value.at("original_top").get<LONG>();
      recovery.original_rect.right = value.at("original_right").get<LONG>();
      recovery.original_rect.bottom = value.at("original_bottom").get<LONG>();
      RECT applied_rect {};
      applied_rect.left = value.at("applied_left").get<LONG>();
      applied_rect.top = value.at("applied_top").get<LONG>();
      applied_rect.right = value.at("applied_right").get<LONG>();
      applied_rect.bottom = value.at("applied_bottom").get<LONG>();
      if (recovery.device_path.empty() ||
          !valid_rect(recovery.original_rect) ||
          !valid_rect(applied_rect)) {
        throw std::runtime_error("invalid monitor identity or rectangle");
      }
      recovery.owned_rects.emplace_back(applied_rect);
      return recovery;
    }

    nlohmann::json serialize_topology_recovery(const topology_recovery_t &recovery) {
      nlohmann::json owned_rects = nlohmann::json::array();
      for (const auto &rect : recovery.owned_rects) {
        owned_rects.emplace_back(serialize_rect(rect));
      }
      return {
        {"device_path", platf::to_utf8(recovery.device_path)},
        {"original_rect", serialize_rect(recovery.original_rect)},
        {"owned_rects", std::move(owned_rects)},
        {"pending_rect", recovery.pending_rect ?
                           serialize_rect(*recovery.pending_rect) :
                           nlohmann::json(nullptr)},
      };
    }

    topology_recovery_t parse_topology_recovery(const nlohmann::json &value) {
      topology_recovery_t recovery {};
      recovery.device_path = platf::from_utf8(value.at("device_path").get<std::string>());
      recovery.original_rect = parse_rect(value.at("original_rect"));
      const auto &owned_rects = value.at("owned_rects");
      if (!owned_rects.is_array()) {
        throw std::runtime_error("owned rectangle list is not an array");
      }
      for (const auto &rect : owned_rects) {
        const auto parsed = parse_rect(rect);
        if (std::ranges::any_of(recovery.owned_rects, [&](const auto &existing) {
              return same_rect(existing, parsed);
            })) {
          throw std::runtime_error("duplicate owned rectangle");
        }
        recovery.owned_rects.emplace_back(parsed);
      }
      const auto &pending = value.at("pending_rect");
      if (!pending.is_null()) {
        recovery.pending_rect = parse_rect(pending);
      }
      if (recovery.device_path.empty() ||
          (recovery.owned_rects.empty() && !recovery.pending_rect)) {
        throw std::runtime_error("invalid monitor identity or empty recovery transaction");
      }
      return recovery;
    }

    topology_recovery_document_t parse_topology_recovery_document(const nlohmann::json &value) {
      topology_recovery_document_t document;
      const int version {value.at("version").get<int>()};
      if (version == 2) {
        // Version 2 stored one global record. Preserve it as the first per-device entry and
        // atomically migrate it on the next successful write.
        document.recoveries.emplace_back(parse_legacy_topology_recovery(value));
        document.rewrite_required = true;
        return document;
      }
      if (version != 3 && version != 4) {
        throw std::runtime_error("unsupported recovery-record version");
      }

      const auto &entries = value.at("recoveries");
      if (!entries.is_array()) {
        throw std::runtime_error("recovery list is not an array");
      }

      // A former brace-initialization bug wrote [[], record, ...] instead of [record, ...].
      // Accept only that exact, unambiguous signature. All other malformed entries remain a
      // fail-closed error so Apollo never discards topology state it cannot understand.
      const bool has_known_empty_sentinel = entries.size() >= 2 &&
                                            entries.front().is_array() &&
                                            entries.front().empty();
      document.rewrite_required = version != 4 || has_known_empty_sentinel;
      for (std::size_t index = has_known_empty_sentinel ? 1 : 0; index < entries.size(); ++index) {
        const auto &entry = entries[index];
        if (!entry.is_object()) {
          throw std::runtime_error("recovery entry is not an object");
        }
        auto recovery = version == 3 ?
                          parse_legacy_topology_recovery(entry) :
                          parse_topology_recovery(entry);
        if (std::ranges::any_of(document.recoveries, [&](const auto &existing) {
              return existing.device_path == recovery.device_path;
            })) {
          throw std::runtime_error("duplicate monitor identity");
        }
        document.recoveries.emplace_back(std::move(recovery));
      }
      return document;
    }

    nlohmann::json serialize_topology_recovery_document(
      const std::vector<topology_recovery_t> &recoveries
    ) {
      nlohmann::json entries = nlohmann::json::array();
      for (const auto &recovery : recoveries) {
        entries.emplace_back(serialize_topology_recovery(recovery));
      }
      return {
        {"version", 4},
        {"recoveries", std::move(entries)},
      };
    }

    std::optional<std::vector<topology_recovery_t>> load_topology_recoveries_locked(
      bool *rewrite_required = nullptr
    ) {
      if (rewrite_required) {
        *rewrite_required = false;
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
        auto document = parse_topology_recovery_document(value);
        if (rewrite_required) {
          *rewrite_required = document.rewrite_required;
        }
        return std::move(document.recoveries);
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
      const auto value = serialize_topology_recovery_document(recoveries);
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

    bool begin_topology_recovery_move(
      std::wstring_view device_path,
      const RECT &original_rect,
      const RECT &requested_rect
    ) {
      std::lock_guard lock(topology_recovery_mutex);
      auto recoveries {load_topology_recoveries_locked()};
      if (!recoveries) {
        return false;
      }
      auto existing {std::ranges::find_if(*recoveries, [&](const auto &recovery) {
        return recovery.device_path == device_path;
      })};
      if (existing == recoveries->end()) {
        topology_recovery_t recovery;
        recovery.device_path = std::wstring(device_path);
        recovery.original_rect = original_rect;
        recovery.pending_rect = requested_rect;
        recoveries->emplace_back(std::move(recovery));
      } else {
        if (!same_rect(existing->original_rect, original_rect)) {
          BOOST_LOG(error) << "Refusing to replace an AR topology transaction with a different original rectangle."sv;
          return false;
        }
        // Keep all confirmed positions. During recovery a pending marker means SetDisplayConfig
        // may have completed before Apollo could observe Windows' possibly normalized result.
        existing->pending_rect = requested_rect;
      }
      if (!write_topology_recoveries_locked(*recoveries)) {
        return false;
      }
      return true;
    }

    bool commit_topology_recovery_move(
      std::wstring_view device_path,
      const RECT &original_rect,
      const RECT &applied_rect
    ) {
      std::lock_guard lock(topology_recovery_mutex);
      auto recoveries {load_topology_recoveries_locked()};
      if (!recoveries) {
        return false;
      }
      auto existing {std::ranges::find_if(*recoveries, [&](const auto &recovery) {
        return recovery.device_path == device_path;
      })};
      if (existing == recoveries->end()) {
        topology_recovery_t recovery;
        recovery.device_path = std::wstring(device_path);
        recovery.original_rect = original_rect;
        recovery.owned_rects.emplace_back(applied_rect);
        recoveries->emplace_back(std::move(recovery));
      } else {
        if (!same_rect(existing->original_rect, original_rect)) {
          return false;
        }
        if (std::ranges::none_of(existing->owned_rects, [&](const auto &rect) {
              return same_rect(rect, applied_rect);
            })) {
          existing->owned_rects.emplace_back(applied_rect);
        }
        existing->pending_rect.reset();
      }
      return write_topology_recoveries_locked(*recoveries);
    }

    bool cancel_topology_recovery_move(std::wstring_view device_path) {
      std::lock_guard lock(topology_recovery_mutex);
      auto recoveries {load_topology_recoveries_locked()};
      if (!recoveries) {
        return false;
      }
      const auto existing {std::ranges::find_if(*recoveries, [&](const auto &recovery) {
        return recovery.device_path == device_path;
      })};
      if (existing == recoveries->end()) {
        return true;
      }
      existing->pending_rect.reset();
      if (existing->owned_rects.empty()) {
        recoveries->erase(existing);
      }
      return write_topology_recoveries_locked(*recoveries);
    }

    bool topology_recovery_should_restore(
      const topology_recovery_t &recovery,
      const RECT &current_rect
    ) {
      if (same_rect(current_rect, recovery.original_rect)) {
        return false;
      }
      const bool is_confirmed_owned = std::ranges::any_of(recovery.owned_rects, [&](const auto &rect) {
        return same_rect(current_rect, rect);
      });
      // A write-ahead marker proves ownership only of the exact rectangle Apollo was about to
      // request. Treating every non-original rectangle as owned would overwrite a user layout if
      // Apollo stopped after writing the marker but before SetDisplayConfig.
      return is_confirmed_owned ||
             (recovery.pending_rect && same_rect(current_rect, *recovery.pending_rect));
    }

    bool topology_rect_is_safe_isolation_baseline(
      std::wstring_view device_path,
      const RECT &original_rect,
      const RECT &current_rect
    ) {
      if (same_rect(current_rect, original_rect)) {
        return true;
      }

      std::lock_guard lock(topology_recovery_mutex);
      const auto recoveries = load_topology_recoveries_locked();
      if (!recoveries) {
        return false;
      }
      const auto recovery = std::ranges::find_if(*recoveries, [&](const auto &candidate) {
        return candidate.device_path == device_path;
      });
      return recovery != recoveries->end() &&
             same_rect(recovery->original_rect, original_rect) &&
             topology_recovery_should_restore(*recovery, current_rect);
    }

    RECT resize_rect_preserving_desktop_edge(const RECT &rect, LONG width, LONG height) {
      // Windows' primary desktop origin is (0,0). A monitor wholly to its left is attached by its
      // right edge; keeping its left edge fixed while widening would overlap the primary desktop.
      // Right-side and vertically separated layouts retain their top-left source-mode position.
      const bool attached_left_of_primary = rect.right <= 0 && rect.left < rect.right;
      const LONG left = attached_left_of_primary ? rect.right - width : rect.left;
      return {
        left,
        rect.top,
        left + width,
        rect.top + height,
      };
    }

    bool rebase_topology_recovery_for_mode_change(
      topology_recovery_t &recovery,
      const RECT &previous_original_rect,
      const RECT &current_rect
    ) {
      if (!same_rect(recovery.original_rect, previous_original_rect) || !valid_rect(current_rect)) {
        return false;
      }

      const LONG width = current_rect.right - current_rect.left;
      const LONG height = current_rect.bottom - current_rect.top;
      auto projected_matches = [&](const RECT &rect) {
        return same_rect(resize_rect_preserving_desktop_edge(rect, width, height), current_rect);
      };
      const bool current_position_was_owned = projected_matches(recovery.original_rect) ||
                                              std::ranges::any_of(recovery.owned_rects, projected_matches) ||
                                              (recovery.pending_rect && projected_matches(*recovery.pending_rect));
      if (!current_position_was_owned) {
        return false;
      }

      recovery.original_rect = resize_rect_preserving_desktop_edge(
        recovery.original_rect,
        width,
        height
      );
      std::vector<RECT> resized_owned;
      resized_owned.reserve(recovery.owned_rects.size());
      for (const auto &rect : recovery.owned_rects) {
        const auto resized = resize_rect_preserving_desktop_edge(rect, width, height);
        if (std::ranges::none_of(resized_owned, [&](const auto &existing) {
              return same_rect(existing, resized);
            })) {
          resized_owned.emplace_back(resized);
        }
      }
      recovery.owned_rects = std::move(resized_owned);
      if (recovery.pending_rect) {
        recovery.pending_rect = resize_rect_preserving_desktop_edge(
          *recovery.pending_rect,
          width,
          height
        );
      }
      return true;
    }

    std::optional<RECT> prepare_topology_recovery_for_mode_change(
      std::wstring_view device_path,
      const RECT &previous_original_rect,
      const RECT &current_rect
    ) {
      std::lock_guard lock(topology_recovery_mutex);
      auto recoveries = load_topology_recoveries_locked();
      if (!recoveries) {
        return std::nullopt;
      }

      const auto existing = std::ranges::find_if(*recoveries, [&](const auto &recovery) {
        return recovery.device_path == device_path;
      });
      if (existing == recoveries->end()) {
        // No Apollo move is outstanding. The current mode's exact rectangle is therefore the
        // user-owned baseline that a later isolation transaction must restore.
        return current_rect;
      }

      if (rebase_topology_recovery_for_mode_change(
            *existing,
            previous_original_rect,
            current_rect
          )) {
        const auto rebased_original = existing->original_rect;
        return write_topology_recoveries_locked(*recoveries) ?
                 std::optional<RECT>(rebased_original) :
                 std::nullopt;
      }

      // A different origin cannot be attributed to the resolution change. Preserve it as a user
      // move by retiring Apollo's old-size ownership evidence and using the observed rectangle as
      // the new baseline. This mirrors normal recovery's fail-closed treatment of unowned layouts.
      recoveries->erase(existing);
      if (!write_topology_recoveries_locked(*recoveries)) {
        return std::nullopt;
      }
      return current_rect;
    }

    bool same_physical_output(
      const std::optional<target_state_t> &left,
      const std::optional<target_state_t> &right
    ) {
      return left && right &&
             left->device_path == right->device_path &&
             same_luid(left->adapter_id, right->adapter_id);
    }

    bool can_reconfigure_local_session(
      const std::optional<target_state_t> &left,
      const std::optional<target_state_t> &right
    ) {
      return same_physical_output(left, right) &&
             left->mode != presentation_mode_e::unsupported &&
             right->mode != presentation_mode_e::unsupported &&
             !right->is_primary && !right->is_cloned;
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
             same_luid(left->adapter_id, right->adapter_id) &&
             left_width == right_width && left_height == right_height &&
             left->refresh_millihz == right->refresh_millihz && left->mode == right->mode &&
             left->is_primary == right->is_primary && left->is_cloned == right->is_cloned &&
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

    std::optional<std::string> read_config_file_strict() {
      const std::filesystem::path path = platf::from_utf8(config::sunshine.config_file);
      std::ifstream input(path, std::ios::binary);
      if (!input) {
        return std::nullopt;
      }
      std::string contents {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>(),
      };
      if (input.bad()) {
        return std::nullopt;
      }
      return contents;
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

    bool persist_devices_locked() {
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

    void load_devices() {
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
          if (persist_devices_locked()) {
            device_persistence_dirty = false;
            device_persistence_retry_after = {};
            BOOST_LOG(info) << "Removed Apollo's internal virtual desktop from the AR display decision list."sv;
          } else {
            device_persistence_retry_after = std::chrono::steady_clock::now() + 2s;
          }
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

    bool primary_source_is_authoritative(
      std::wstring_view primary_gdi_name,
      const std::vector<std::wstring> &active_gdi_names
    ) {
      return !primary_gdi_name.empty() &&
             std::ranges::any_of(active_gdi_names, [&](const auto &active_name) {
               return active_name == primary_gdi_name;
             });
    }

    std::optional<std::vector<target_state_t>> enumerate_targets() {
      std::vector<DISPLAYCONFIG_PATH_INFO> paths;
      std::vector<DISPLAYCONFIG_MODE_INFO> modes;
      if (!VDISPLAY::queryActiveDisplayConfig(paths, modes)) {
        return std::nullopt;
      }

      const auto virtual_sources = VDISPLAY::matchDisplay(virtual_display_driver_name);
      std::vector<target_state_t> targets;
      std::vector<std::wstring> active_gdi_names;
      active_gdi_names.reserve(paths.size());
      for (const auto &path : paths) {
        DISPLAYCONFIG_SOURCE_DEVICE_NAME source_name {};
        source_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
        source_name.header.size = sizeof(source_name);
        source_name.header.adapterId = path.sourceInfo.adapterId;
        source_name.header.id = path.sourceInfo.id;
        if (DisplayConfigGetDeviceInfo(&source_name.header) != ERROR_SUCCESS) {
          // An active path that cannot be described makes the whole observation indeterminate.
          // Treating a partial snapshot as authoritative can look like a hot-unplug and destroy a
          // healthy presentation session during HDR/topology transitions.
          return std::nullopt;
        }
        active_gdi_names.emplace_back(source_name.viewGdiDeviceName);
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
        if (DisplayConfigGetDeviceInfo(&target_name.header) != ERROR_SUCCESS) {
          return std::nullopt;
        }
        if (contains_case_insensitive(target_name.monitorDevicePath, L"SUDOVDA")) {
          continue;
        }
        const auto device_id = stable_model_id(target_name);
        const auto friendly_name = platf::to_utf8(target_name.monitorFriendlyDeviceName);
        if (is_internal_virtual_display(device_id, friendly_name)) {
          continue;
        }

        if (path.sourceInfo.modeInfoIdx == DISPLAYCONFIG_PATH_MODE_IDX_INVALID || path.sourceInfo.modeInfoIdx >= modes.size()) {
          return std::nullopt;
        }
        const auto &mode_info = modes[path.sourceInfo.modeInfoIdx];
        if (mode_info.infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE) {
          return std::nullopt;
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
        state.source_adapter_id = path.sourceInfo.adapterId;
        state.source_id = path.sourceInfo.id;
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
      if (!primary_source_is_authoritative(primary_gdi_name, active_gdi_names)) {
        // Primary/clone status is a destructive-transition safety gate. If Windows cannot provide
        // a primary source belonging to this complete active-path snapshot, the observation is
        // indeterminate rather than evidence that every selectable output is non-primary.
        return std::nullopt;
      }
      std::string primary_device_path;
      for (const auto &target : targets) {
        if (platf::from_utf8(target.gdi_name) == primary_gdi_name) {
          primary_device_path = platf::to_utf8(target.device_path);
          break;
        }
      }
      for (auto &target : targets) {
        target.is_primary = platf::from_utf8(target.gdi_name) == primary_gdi_name;
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

    bool restore_physical_output_position(
      const RECT &original_rect,
      const std::wstring &target_device_path
    );

    bool recover_saved_topology(std::wstring_view device_path);

    struct display_target_identity_t {
      LUID adapter_id {};
      UINT32 target_id = 0;
    };

    struct active_source_t {
      std::size_t path_index = 0;
      std::size_t mode_index = 0;
      LUID source_adapter_id {};
      UINT32 source_id = 0;
      display_target_identity_t target;
      std::wstring gdi_name;
      std::wstring device_path;
      std::wstring friendly_name;
      RECT rect {};
    };

    struct topology_snapshot_t {
      std::vector<DISPLAYCONFIG_PATH_INFO> paths;
      std::vector<DISPLAYCONFIG_MODE_INFO> modes;
      std::vector<active_source_t> sources;
    };

    bool same_target_identity(
      const display_target_identity_t &left,
      const display_target_identity_t &right
    ) {
      return same_luid(left.adapter_id, right.adapter_id) && left.target_id == right.target_id;
    }

    bool same_source_identity(const active_source_t &left, const active_source_t &right) {
      return same_luid(left.source_adapter_id, right.source_adapter_id) &&
             left.source_id == right.source_id;
    }

    std::optional<topology_snapshot_t> query_complete_topology_snapshot() {
      topology_snapshot_t snapshot;
      if (!VDISPLAY::queryActiveDisplayConfig(snapshot.paths, snapshot.modes)) {
        return std::nullopt;
      }

      snapshot.sources.reserve(snapshot.paths.size());
      for (std::size_t path_index = 0; path_index < snapshot.paths.size(); ++path_index) {
        const auto &path = snapshot.paths[path_index];
        if (path.sourceInfo.modeInfoIdx == DISPLAYCONFIG_PATH_MODE_IDX_INVALID ||
            path.sourceInfo.modeInfoIdx >= snapshot.modes.size() ||
            snapshot.modes[path.sourceInfo.modeInfoIdx].infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE) {
          return std::nullopt;
        }

        DISPLAYCONFIG_SOURCE_DEVICE_NAME source_name {};
        source_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
        source_name.header.size = sizeof(source_name);
        source_name.header.adapterId = path.sourceInfo.adapterId;
        source_name.header.id = path.sourceInfo.id;
        if (DisplayConfigGetDeviceInfo(&source_name.header) != ERROR_SUCCESS) {
          return std::nullopt;
        }

        DISPLAYCONFIG_TARGET_DEVICE_NAME target_name {};
        target_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
        target_name.header.size = sizeof(target_name);
        target_name.header.adapterId = path.targetInfo.adapterId;
        target_name.header.id = path.targetInfo.id;
        if (DisplayConfigGetDeviceInfo(&target_name.header) != ERROR_SUCCESS) {
          return std::nullopt;
        }

        const auto &mode = snapshot.modes[path.sourceInfo.modeInfoIdx].sourceMode;
        active_source_t source;
        source.path_index = path_index;
        source.mode_index = path.sourceInfo.modeInfoIdx;
        source.source_adapter_id = path.sourceInfo.adapterId;
        source.source_id = path.sourceInfo.id;
        source.target = {path.targetInfo.adapterId, path.targetInfo.id};
        source.gdi_name = source_name.viewGdiDeviceName;
        source.device_path = target_name.monitorDevicePath;
        source.friendly_name = target_name.monitorFriendlyDeviceName;
        source.rect = {
          mode.position.x,
          mode.position.y,
          mode.position.x + (LONG) mode.width,
          mode.position.y + (LONG) mode.height,
        };
        snapshot.sources.emplace_back(std::move(source));
      }
      return snapshot;
    }

    struct resolved_virtual_display_t {
      SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT identity {};
      std::wstring gdi_name;
      std::wstring device_path;
    };

    enum class virtual_display_presence_e {
      indeterminate,
      absent,
      present,
    };

    struct virtual_display_probe_t {
      virtual_display_presence_e presence {virtual_display_presence_e::indeterminate};
      std::optional<resolved_virtual_display_t> resolved;
    };

    virtual_display_probe_t probe_virtual_display(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view device_path,
      std::wstring_view gdi_name
    ) {
      const auto snapshot = query_complete_topology_snapshot();
      if (!snapshot) {
        return {};
      }

      auto is_local_virtual_source = [](const active_source_t &source) {
        return lowercase(platf::to_utf8(source.friendly_name)).starts_with("apollo ar des"sv);
      };
      auto select = [&](auto predicate) {
        std::vector<const active_source_t *> matches;
        for (const auto &source : snapshot->sources) {
          if (predicate(source)) {
            matches.push_back(&source);
          }
        }
        return matches;
      };

      auto matches = select([&](const auto &source) {
        return same_luid(source.target.adapter_id, identity.AdapterLuid) &&
               source.target.target_id == identity.TargetId &&
               is_local_virtual_source(source);
      });
      if (matches.empty() && !device_path.empty()) {
        matches = select([&](const auto &source) {
          // This path was learned from the driver-returned identity while Apollo exclusively
          // owned SudoVDA, so it remains valid when Windows renumbers the target during a mode
          // or Advanced Color transition.
          return source.device_path == device_path;
        });
      }
      if (matches.empty() && !gdi_name.empty()) {
        matches = select([&](const auto &source) {
          return source.gdi_name == gdi_name && is_local_virtual_source(source);
        });
      }

      // A complete active snapshot can temporarily renumber every published identifier. Any
      // Apollo AR source proves the driver-owned desktop is still present, but an ambiguous match
      // is deliberately not selected for capture or topology mutation.
      if (matches.empty()) {
        matches = select(is_local_virtual_source);
      }
      if (matches.empty()) {
        return {virtual_display_presence_e::absent, std::nullopt};
      }
      if (matches.size() != 1) {
        return {virtual_display_presence_e::present, std::nullopt};
      }

      const auto &source = *matches.front();
      const bool expected_geometry = source.rect.right - source.rect.left == source_width &&
                                     source.rect.bottom - source.rect.top == source_height;
      if (!expected_geometry) {
        // ChangeDisplaySettings can publish an intermediate mode for the retained source. That is
        // present-but-not-ready, never evidence that it is safe to destroy the desktop.
        return {virtual_display_presence_e::present, std::nullopt};
      }

      return {
        virtual_display_presence_e::present,
        resolved_virtual_display_t {
          .identity = {source.target.adapter_id, source.target.target_id},
          .gdi_name = source.gdi_name,
          .device_path = source.device_path,
        },
      };
    }

    std::optional<resolved_virtual_display_t> resolve_virtual_display(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view device_path,
      std::wstring_view gdi_name,
      bool *topology_query_succeeded = nullptr
    ) {
      if (topology_query_succeeded) {
        *topology_query_succeeded = false;
      }
      auto probe = probe_virtual_display(identity, device_path, gdi_name);
      if (topology_query_succeeded) {
        *topology_query_succeeded = probe.presence != virtual_display_presence_e::indeterminate;
      }
      return std::move(probe.resolved);
    }

    std::optional<std::wstring> refresh_virtual_display_name(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity
    ) {
      const auto resolved = resolve_virtual_display(identity, {}, {});
      return resolved ? std::optional<std::wstring>(resolved->gdi_name) : std::nullopt;
    }

    std::optional<std::vector<std::string>> topology_snapshot_fingerprint(
      const topology_snapshot_t &snapshot
    ) {
      if (snapshot.sources.size() != snapshot.paths.size()) {
        return std::nullopt;
      }

      std::vector<std::string> entries;
      entries.reserve(snapshot.sources.size());
      for (const auto &source : snapshot.sources) {
        if (source.path_index >= snapshot.paths.size() || source.mode_index >= snapshot.modes.size()) {
          return std::nullopt;
        }
        const auto &path = snapshot.paths[source.path_index];
        const auto &source_mode_info = snapshot.modes[source.mode_index];
        if (source_mode_info.infoType != DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE) {
          return std::nullopt;
        }
        const auto &source_mode = source_mode_info.sourceMode;

        std::ostringstream entry;
        entry << path.sourceInfo.adapterId.HighPart << ':' << path.sourceInfo.adapterId.LowPart
              << ':' << path.sourceInfo.id << ':' << path.sourceInfo.statusFlags
              << '>' << path.targetInfo.adapterId.HighPart << ':' << path.targetInfo.adapterId.LowPart
              << ':' << path.targetInfo.id << ':' << path.targetInfo.statusFlags
              << ':' << (unsigned int) path.targetInfo.outputTechnology
              << ':' << (unsigned int) path.targetInfo.rotation
              << ':' << (unsigned int) path.targetInfo.scaling
              << ':' << path.targetInfo.refreshRate.Numerator
              << '/' << path.targetInfo.refreshRate.Denominator
              << ':' << (unsigned int) path.targetInfo.scanLineOrdering
              << ':' << path.targetInfo.targetAvailable
              << ':' << path.flags
              << "|src=" << source_mode.position.x << ',' << source_mode.position.y
              << ':' << source_mode.width << 'x' << source_mode.height
              << ':' << (unsigned int) source_mode.pixelFormat
              << "|gdi=" << platf::to_utf8(source.gdi_name)
              << "|pnp=" << platf::to_utf8(source.device_path);

        const auto target_mode_index = path.targetInfo.modeInfoIdx;
        if (target_mode_index == DISPLAYCONFIG_PATH_MODE_IDX_INVALID) {
          entry << "|target=none";
        } else {
          if (target_mode_index >= snapshot.modes.size() ||
              snapshot.modes[target_mode_index].infoType != DISPLAYCONFIG_MODE_INFO_TYPE_TARGET) {
            return std::nullopt;
          }
          const auto &signal = snapshot.modes[target_mode_index].targetMode.targetVideoSignalInfo;
          entry << "|target=" << signal.pixelRate
                << ':' << signal.hSyncFreq.Numerator << '/' << signal.hSyncFreq.Denominator
                << ':' << signal.vSyncFreq.Numerator << '/' << signal.vSyncFreq.Denominator
                << ':' << signal.activeSize.cx << 'x' << signal.activeSize.cy
                << ':' << signal.totalSize.cx << 'x' << signal.totalSize.cy
                << ':' << signal.videoStandard
                << ':' << (unsigned int) signal.scanLineOrdering;
        }
        entries.emplace_back(entry.str());
      }
      std::sort(entries.begin(), entries.end());
      return entries;
    }

    bool same_topology_snapshot(
      const topology_snapshot_t &left,
      const topology_snapshot_t &right
    ) {
      const auto left_fingerprint = topology_snapshot_fingerprint(left);
      const auto right_fingerprint = topology_snapshot_fingerprint(right);
      return left_fingerprint && right_fingerprint && *left_fingerprint == *right_fingerprint;
    }

    bool source_is_cloned(
      const std::vector<active_source_t> &sources,
      const active_source_t &selected
    ) {
      return std::ranges::any_of(sources, [&](const auto &candidate) {
        return !same_target_identity(candidate.target, selected.target) &&
               same_source_identity(candidate, selected);
      });
    }

    struct anchor_candidate_t {
      RECT rect {};
      LUID source_adapter_id {};
      UINT32 source_id = 0;
    };

    bool anchor_is_better(const anchor_candidate_t &candidate, const anchor_candidate_t &current) {
      if (candidate.rect.right != current.rect.right) {
        return candidate.rect.right > current.rect.right;
      }
      if (candidate.rect.top != current.rect.top) {
        return candidate.rect.top < current.rect.top;
      }
      if (candidate.rect.left != current.rect.left) {
        return candidate.rect.left < current.rect.left;
      }
      if (candidate.source_adapter_id.HighPart != current.source_adapter_id.HighPart) {
        return candidate.source_adapter_id.HighPart < current.source_adapter_id.HighPart;
      }
      if (candidate.source_adapter_id.LowPart != current.source_adapter_id.LowPart) {
        return candidate.source_adapter_id.LowPart < current.source_adapter_id.LowPart;
      }
      return candidate.source_id < current.source_id;
    }

    std::optional<anchor_candidate_t> select_anchor(
      const std::vector<active_source_t> &sources,
      const active_source_t &physical,
      const active_source_t &virtual_source
    ) {
      std::optional<anchor_candidate_t> selected;
      for (const auto &source : sources) {
        if (same_source_identity(source, physical) || same_source_identity(source, virtual_source)) {
          continue;
        }
        anchor_candidate_t candidate {source.rect, source.source_adapter_id, source.source_id};
        if (!selected || anchor_is_better(candidate, *selected)) {
          selected = candidate;
        }
      }
      return selected;
    }

    std::optional<anchor_candidate_t> select_pre_add_anchor(
      const std::vector<active_source_t> &sources,
      const active_source_t &physical
    ) {
      std::optional<anchor_candidate_t> selected;
      for (const auto &source : sources) {
        if (same_source_identity(source, physical)) {
          continue;
        }
        anchor_candidate_t candidate {source.rect, source.source_adapter_id, source.source_id};
        if (!selected || anchor_is_better(candidate, *selected)) {
          selected = candidate;
        }
      }
      return selected;
    }

    struct pre_add_isolation_plan_t {
      RECT physical_rect {};
      linear_layout_t layout;
    };

    std::optional<pre_add_isolation_plan_t> build_pre_add_isolation_plan(
      std::wstring_view target_device_path
    ) {
      const auto snapshot = query_complete_topology_snapshot();
      if (!snapshot) {
        return std::nullopt;
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
      if (primary_gdi_name.empty()) {
        return std::nullopt;
      }

      const active_source_t *physical = nullptr;
      const active_source_t *primary = nullptr;
      for (const auto &source : snapshot->sources) {
        if (source.device_path == target_device_path) {
          if (physical) {
            return std::nullopt;
          }
          physical = &source;
        }
        if (source.gdi_name == primary_gdi_name) {
          primary = &source;
        }
      }
      if (!physical || !primary || same_source_identity(*physical, *primary) ||
          source_is_cloned(snapshot->sources, *physical)) {
        return std::nullopt;
      }

      const auto anchor = select_pre_add_anchor(snapshot->sources, *physical);
      if (!anchor) {
        return std::nullopt;
      }
      const auto physical_width = physical->rect.right - physical->rect.left;
      const auto physical_height = physical->rect.bottom - physical->rect.top;
      return pre_add_isolation_plan_t {
        .physical_rect = physical->rect,
        .layout = compute_linear_layout(
          anchor->rect,
          source_width,
          source_height,
          physical_width,
          physical_height
        ),
      };
    }

    struct isolation_plan_t {
      topology_snapshot_t snapshot;
      std::size_t physical_index = 0;
      std::size_t virtual_index = 0;
      display_target_identity_t virtual_identity;
      linear_layout_t layout;
    };

    std::optional<isolation_plan_t> build_isolation_plan(
      std::wstring_view virtual_gdi_name,
      const std::optional<display_target_identity_t> &virtual_identity,
      std::wstring_view target_device_path
    ) {
      auto snapshot = query_complete_topology_snapshot();
      if (!snapshot) {
        return std::nullopt;
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
      if (primary_gdi_name.empty()) {
        return std::nullopt;
      }

      std::optional<std::size_t> physical_index;
      std::optional<std::size_t> virtual_index;
      std::optional<std::size_t> primary_index;
      for (std::size_t index = 0; index < snapshot->sources.size(); ++index) {
        const auto &source = snapshot->sources[index];
        if (source.device_path == target_device_path) {
          if (physical_index) {
            return std::nullopt;
          }
          physical_index = index;
        }
        const bool is_virtual = virtual_identity ?
                                  same_target_identity(source.target, *virtual_identity) :
                                  source.gdi_name == virtual_gdi_name;
        if (is_virtual) {
          if (virtual_index) {
            return std::nullopt;
          }
          virtual_index = index;
        }
        if (source.gdi_name == primary_gdi_name) {
          primary_index = index;
        }
      }
      if (!physical_index || !virtual_index || !primary_index || *physical_index == *virtual_index) {
        return std::nullopt;
      }

      const auto &physical = snapshot->sources[*physical_index];
      const auto &virtual_source = snapshot->sources[*virtual_index];
      const auto &primary = snapshot->sources[*primary_index];
      if (same_source_identity(physical, primary)) {
        BOOST_LOG(warning) << "The selected AR display is the Windows primary display; automatic pointer isolation is unavailable."sv;
        return std::nullopt;
      }
      if (source_is_cloned(snapshot->sources, physical)) {
        BOOST_LOG(warning) << "The selected AR display shares a cloned desktop source; refusing to move the clone group for pointer isolation."sv;
        return std::nullopt;
      }
      if (source_is_cloned(snapshot->sources, virtual_source)) {
        BOOST_LOG(warning) << "The private AR virtual desktop unexpectedly shares a cloned source; refusing to move the clone group."sv;
        return std::nullopt;
      }
      const auto anchor = select_anchor(snapshot->sources, physical, virtual_source);
      if (!anchor) {
        return std::nullopt;
      }

      const auto virtual_width = virtual_source.rect.right - virtual_source.rect.left;
      const auto virtual_height = virtual_source.rect.bottom - virtual_source.rect.top;
      const auto physical_width = physical.rect.right - physical.rect.left;
      const auto physical_height = physical.rect.bottom - physical.rect.top;
      isolation_plan_t plan;
      plan.physical_index = *physical_index;
      plan.virtual_index = *virtual_index;
      plan.virtual_identity = virtual_source.target;
      plan.layout = compute_linear_layout(
        anchor->rect,
        virtual_width,
        virtual_height,
        physical_width,
        physical_height
      );
      plan.snapshot = std::move(*snapshot);
      return plan;
    }

    bool layout_is_exact_and_adjacent(const isolation_plan_t &plan) {
      const auto &physical = plan.snapshot.sources[plan.physical_index].rect;
      const auto &virtual_source = plan.snapshot.sources[plan.virtual_index].rect;
      return same_rect(virtual_source, plan.layout.virtual_rect) &&
             same_rect(physical, plan.layout.physical_rect) &&
             virtual_source.right == physical.left &&
             virtual_source.top == physical.top &&
             virtual_source.bottom == physical.bottom;
    }

    std::optional<isolation_plan_t> query_verified_isolation(
      const display_target_identity_t &virtual_identity,
      std::wstring_view target_device_path,
      const linear_layout_t &expected
    ) {
      auto plan = build_isolation_plan({}, virtual_identity, target_device_path);
      if (!plan) {
        return std::nullopt;
      }
      plan->layout = expected;
      return layout_is_exact_and_adjacent(*plan) ? std::move(plan) : std::nullopt;
    }

    std::optional<target_state_t> isolate_physical_output(
      const std::wstring &virtual_gdi_name,
      const std::wstring &target_device_path,
      const RECT &original_rect
    ) {
      auto initial_plan = build_isolation_plan(virtual_gdi_name, std::nullopt, target_device_path);
      if (!initial_plan) {
        return std::nullopt;
      }
      const auto initial_physical_rect = initial_plan->snapshot.sources[initial_plan->physical_index].rect;
      const auto initial_virtual_rect = initial_plan->snapshot.sources[initial_plan->virtual_index].rect;
      const auto virtual_identity = initial_plan->virtual_identity;
      if (!topology_rect_is_safe_isolation_baseline(
            target_device_path,
            original_rect,
            initial_physical_rect
          )) {
        // SudoVDA/HDR setup and a user can both move the physical target before this pass. Only
        // the pre-session rectangle or an exact journaled Apollo rectangle is safe to replace.
        // Preserve everything else and retire stale evidence instead of laundering it into a new
        // pending transaction.
        BOOST_LOG(warning) << "AR output occupies an unowned rectangle; preserving it and aborting isolation."sv;
        if (!clear_topology_recovery(target_device_path)) {
          BOOST_LOG(error) << "Could not durably retire stale AR topology evidence for the preserved rectangle."sv;
        }
        return std::nullopt;
      }
      if (layout_is_exact_and_adjacent(*initial_plan)) {
        // Creating the virtual source can make Windows move the physical target into the desired
        // row before Apollo reaches SetDisplayConfig. That is still an Apollo-owned topology
        // change and needs durable recovery evidence whenever it differs from the pre-session rect.
        if (!same_rect(original_rect, initial_physical_rect)) {
          if (!commit_topology_recovery_move(
                target_device_path,
                original_rect,
                initial_physical_rect
              )) {
            BOOST_LOG(error) << "Refusing to adopt the AR output row without a durable topology recovery record."sv;
            return std::nullopt;
          }
        }
        const auto actual = find_target(target_device_path);
        return actual && same_rect(actual->rect, initial_physical_rect) ? actual : std::nullopt;
      }

      auto requested_layout = initial_plan->layout;
      if (!begin_topology_recovery_move(
            target_device_path,
            original_rect,
            requested_layout.physical_rect
          )) {
        BOOST_LOG(error) << "Refusing to move the AR output without a durable topology recovery record."sv;
        return std::nullopt;
      }

      // The write-through recovery record can take long enough for another monitor/HDR transition
      // to complete. Re-query the complete CCD state after every journal write and apply only the
      // freshly reconciled snapshot. A changed anchor gets its own pending transaction first.
      std::optional<isolation_plan_t> plan;
      for (int attempt = 0; attempt < 4; ++attempt) {
        auto candidate = build_isolation_plan({}, virtual_identity, target_device_path);
        if (!candidate) {
          break;
        }
        const auto &current_physical = candidate->snapshot.sources[candidate->physical_index].rect;
        const auto &current_virtual = candidate->snapshot.sources[candidate->virtual_index].rect;
        if (!same_rect(current_physical, initial_physical_rect) ||
            !same_rect(current_virtual, initial_virtual_rect)) {
          BOOST_LOG(info) << "AR topology changed while its recovery transaction was being persisted; deferring isolation."sv;
          break;
        }
        if (!same_rect(candidate->layout.physical_rect, requested_layout.physical_rect) ||
            !same_rect(candidate->layout.virtual_rect, requested_layout.virtual_rect)) {
          requested_layout = candidate->layout;
          if (!begin_topology_recovery_move(
                target_device_path,
                original_rect,
                requested_layout.physical_rect
              )) {
            BOOST_LOG(error) << "Could not reconcile the pending AR topology transaction with the latest desktop layout."sv;
            break;
          }
          continue;
        }
        plan = std::move(candidate);
        break;
      }
      if (!plan) {
        if (!cancel_topology_recovery_move(target_device_path)) {
          BOOST_LOG(error) << "Could not cancel the unapplied AR topology transaction."sv;
        }
        return std::nullopt;
      }

      const auto validated_baseline = plan->snapshot;
      auto &virtual_mode = plan->snapshot.modes[plan->snapshot.sources[plan->virtual_index].mode_index].sourceMode;
      auto &physical_mode = plan->snapshot.modes[plan->snapshot.sources[plan->physical_index].mode_index].sourceMode;
      virtual_mode.position.x = requested_layout.virtual_rect.left;
      virtual_mode.position.y = requested_layout.virtual_rect.top;
      physical_mode.position.x = requested_layout.physical_rect.left;
      physical_mode.position.y = requested_layout.physical_rect.top;

      const auto validate_status = SetDisplayConfig(
        (UINT32) plan->snapshot.paths.size(),
        plan->snapshot.paths.data(),
        (UINT32) plan->snapshot.modes.size(),
        plan->snapshot.modes.data(),
        SDC_VALIDATE | SDC_USE_SUPPLIED_DISPLAY_CONFIG
      );
      if (validate_status != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "Could not validate AR presentation-output isolation: "sv << validate_status;
        if (!cancel_topology_recovery_move(target_device_path)) {
          BOOST_LOG(error) << "Could not cancel the failed AR topology transaction; retaining conservative recovery state."sv;
        }
        return std::nullopt;
      }

      // SDC_VALIDATE does not reserve the display topology. Re-query after validation and apply
      // only a semantically identical complete snapshot, so a user change to any active output is
      // never overwritten by the supplied full CCD image.
      auto fresh_plan = build_isolation_plan({}, virtual_identity, target_device_path);
      if (!fresh_plan ||
          !same_topology_snapshot(validated_baseline, fresh_plan->snapshot) ||
          !same_rect(fresh_plan->layout.virtual_rect, requested_layout.virtual_rect) ||
          !same_rect(fresh_plan->layout.physical_rect, requested_layout.physical_rect)) {
        BOOST_LOG(info) << "Desktop topology changed after AR isolation validation; deferring the move."sv;
        if (!cancel_topology_recovery_move(target_device_path)) {
          BOOST_LOG(error) << "Could not cancel the stale AR topology transaction."sv;
        }
        return std::nullopt;
      }
      auto &fresh_virtual_mode = fresh_plan->snapshot.modes[fresh_plan->snapshot.sources[fresh_plan->virtual_index].mode_index].sourceMode;
      auto &fresh_physical_mode = fresh_plan->snapshot.modes[fresh_plan->snapshot.sources[fresh_plan->physical_index].mode_index].sourceMode;
      fresh_virtual_mode.position.x = requested_layout.virtual_rect.left;
      fresh_virtual_mode.position.y = requested_layout.virtual_rect.top;
      fresh_physical_mode.position.x = requested_layout.physical_rect.left;
      fresh_physical_mode.position.y = requested_layout.physical_rect.top;
      const auto status = SetDisplayConfig(
        (UINT32) fresh_plan->snapshot.paths.size(),
        fresh_plan->snapshot.paths.data(),
        (UINT32) fresh_plan->snapshot.modes.size(),
        fresh_plan->snapshot.modes.data(),
        SDC_APPLY | SDC_USE_SUPPLIED_DISPLAY_CONFIG
      );
      if (status != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "Could not isolate the AR presentation output: "sv << status;
        if (!cancel_topology_recovery_move(target_device_path)) {
          BOOST_LOG(error) << "Could not cancel the failed AR topology transaction; retaining conservative recovery state."sv;
        }
        return std::nullopt;
      }

      // Never infer success from SetDisplayConfig alone. Resolve both the exact SudoVDA identity
      // and physical PnP target, then require the complete requested row and shared edge.
      std::optional<isolation_plan_t> verified_plan;
      for (int attempt = 0; attempt < 10; ++attempt) {
        if (auto verified = query_verified_isolation(
              virtual_identity,
              target_device_path,
              requested_layout
            )) {
          const auto &verified_physical = verified->snapshot.sources[verified->physical_index].rect;
          if (!commit_topology_recovery_move(
                target_device_path,
                original_rect,
                verified_physical
              )) {
            BOOST_LOG(error) << "Could not update the durable AR topology record with Windows' "sv
                                "actual applied rectangle; refusing to retain the isolated layout."sv;
            if (!recover_saved_topology(target_device_path)) {
              BOOST_LOG(error) << "Could not safely classify or restore the AR output after its "sv
                                  "recovery-record update failed; retaining durable recovery state."sv;
            }
            return std::nullopt;
          }
          verified_plan = std::move(verified);
          break;
        }
        std::this_thread::sleep_for(50ms);
      }

      if (verified_plan) {
        const auto verified_physical = verified_plan->snapshot.sources[verified_plan->physical_index].rect;
        const auto verified_virtual = verified_plan->snapshot.sources[verified_plan->virtual_index].rect;
        for (int attempt = 0; attempt < 10; ++attempt) {
          bool topology_query_succeeded = false;
          const auto actual = find_target(target_device_path, {}, &topology_query_succeeded);
          if (!topology_query_succeeded) {
            std::this_thread::sleep_for(50ms);
            continue;
          }
          if (!actual) {
            BOOST_LOG(info) << "AR output disconnected after isolation; retaining its recovery record."sv;
            return std::nullopt;
          }
          if (same_rect(actual->rect, verified_physical)) {
            BOOST_LOG(info) << "Verified AR presentation row: virtual source ["sv
                            << verified_virtual.left << ',' << verified_virtual.top << " -> "sv
                            << verified_virtual.right << ',' << verified_virtual.bottom
                            << "]; physical sink ["sv
                            << verified_physical.left << ',' << verified_physical.top << " -> "sv
                            << verified_physical.right << ',' << verified_physical.bottom << "]."sv;
            return actual;
          }

          // A different authoritative rectangle may be a user move. The durable classifier rolls
          // back only exact pending/confirmed Apollo rectangles and retires evidence for any other
          // position without moving it.
          BOOST_LOG(info) << "AR output moved after isolation verification; classifying the newer layout before rollback."sv;
          if (!recover_saved_topology(target_device_path)) {
            BOOST_LOG(warning) << "Could not finish AR topology classification; retaining recovery state."sv;
          }
          return std::nullopt;
        }
        BOOST_LOG(warning) << "AR-output isolation was verified, but its final target query remained indeterminate; "sv
                              "retaining recovery state without moving the display."sv;
        return std::nullopt;
      }

      BOOST_LOG(warning) << "AR-output topology applied, but the complete source/sink row could not be verified."sv;
      if (!recover_saved_topology(target_device_path)) {
        BOOST_LOG(error) << "Could not safely classify or restore the unverified AR topology; "sv
                            "retaining recovery state for the next stable observation."sv;
      }
      return std::nullopt;
    }

    bool restore_physical_output_position(
      const RECT &original_rect,
      const std::wstring &target_device_path
    ) {
      auto snapshot = query_complete_topology_snapshot();
      if (!snapshot) {
        return false;
      }

      const auto physical = std::ranges::find_if(snapshot->sources, [&](const auto &source) {
        return source.device_path == target_device_path;
      });
      if (physical == snapshot->sources.end() || source_is_cloned(snapshot->sources, *physical)) {
        return false;
      }
      if (same_rect(physical->rect, original_rect)) {
        return true;
      }

      // This snapshot is intentionally acquired after all recovery-file I/O. Validate and apply it
      // immediately so an old full CCD image cannot overwrite an unrelated monitor adjustment.
      const auto validated_baseline = *snapshot;
      auto &source_mode = snapshot->modes[physical->mode_index].sourceMode;
      source_mode.position.x = original_rect.left;
      source_mode.position.y = original_rect.top;

      const auto validate_status = SetDisplayConfig(
        (UINT32) snapshot->paths.size(),
        snapshot->paths.data(),
        (UINT32) snapshot->modes.size(),
        snapshot->modes.data(),
        SDC_VALIDATE | SDC_USE_SUPPLIED_DISPLAY_CONFIG
      );
      if (validate_status != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "Validating the AR display-position restore failed with status "sv
                           << validate_status << '.';
        return false;
      }

      auto fresh_snapshot = query_complete_topology_snapshot();
      if (!fresh_snapshot || !same_topology_snapshot(validated_baseline, *fresh_snapshot)) {
        BOOST_LOG(info) << "Desktop topology changed after AR restore validation; preserving the newer layout."sv;
        return false;
      }
      const auto fresh_physical = std::ranges::find_if(fresh_snapshot->sources, [&](const auto &source) {
        return source.device_path == target_device_path;
      });
      if (fresh_physical == fresh_snapshot->sources.end() ||
          source_is_cloned(fresh_snapshot->sources, *fresh_physical)) {
        return false;
      }
      auto &fresh_source_mode = fresh_snapshot->modes[fresh_physical->mode_index].sourceMode;
      fresh_source_mode.position.x = original_rect.left;
      fresh_source_mode.position.y = original_rect.top;
      const auto status = SetDisplayConfig(
        (UINT32) fresh_snapshot->paths.size(),
        fresh_snapshot->paths.data(),
        (UINT32) fresh_snapshot->modes.size(),
        fresh_snapshot->modes.data(),
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

    bool normalize_topology_recovery_state() {
      std::lock_guard lock(topology_recovery_mutex);
      bool rewrite_required {false};
      const auto recoveries {load_topology_recoveries_locked(&rewrite_required)};
      if (!recoveries) {
        return false;
      }
      if (!rewrite_required) {
        return true;
      }
      if (!write_topology_recoveries_locked(*recoveries)) {
        BOOST_LOG(warning) << "Could not normalize the local-AR topology recovery record; "sv
                              "the original record remains intact."sv;
        return false;
      }
      BOOST_LOG(info) << "Normalized the local-AR topology recovery record."sv;
      return true;
    }

    struct pre_removal_topology_decision_t {
      RECT desired_rect {};
      bool apollo_owned = false;
      bool recovery_record_present = false;
    };

    std::optional<pre_removal_topology_decision_t> classify_pre_removal_topology(
      std::wstring_view device_path
    ) {
      bool topology_query_succeeded = false;
      const auto current = find_target(device_path, {}, &topology_query_succeeded);
      if (!topology_query_succeeded || !current) {
        return std::nullopt;
      }

      std::optional<topology_recovery_t> recovery;
      {
        std::lock_guard lock(topology_recovery_mutex);
        const auto recoveries = load_topology_recoveries_locked();
        if (!recoveries) {
          return std::nullopt;
        }
        const auto match = std::ranges::find_if(*recoveries, [&](const auto &candidate) {
          return candidate.device_path == device_path;
        });
        if (match != recoveries->end()) {
          recovery = *match;
        }
      }

      if (!recovery || !topology_recovery_should_restore(*recovery, current->rect)) {
        return pre_removal_topology_decision_t {
          .desired_rect = current->rect,
          .apollo_owned = false,
          .recovery_record_present = recovery.has_value(),
        };
      }
      return pre_removal_topology_decision_t {
        .desired_rect = recovery->original_rect,
        .apollo_owned = true,
        .recovery_record_present = true,
      };
    }

    class pre_removal_topology_cleanup_t {
    public:
      pre_removal_topology_cleanup_t(
        std::optional<pre_removal_topology_decision_t> decision,
        std::wstring device_path
      ):
          decision_(std::move(decision)),
          decision_resolved_(decision_.has_value()),
          device_path_(std::move(device_path)) {
      }

      bool prepare_removal() {
        std::lock_guard lock(mutex_);
        bool topology_query_succeeded = false;
        const auto connected_target = find_target(
          device_path_,
          {},
          &topology_query_succeeded
        );
        if (!topology_query_succeeded) {
          return false;
        }
        if (!connected_target) {
          // Removing SudoVDA cannot normalize a disconnected physical target. Keep its durable
          // recovery evidence for reconnect and allow virtual-display retirement to proceed.
          return true;
        }

        const auto refreshed_decision = classify_pre_removal_topology(device_path_);
        if (!refreshed_decision) {
          return false;
        }
        // removeVirtualDisplay() runs immediately after this callback. If it fails, the next
        // attempt re-enters here and replaces this snapshot, so finish() can only consume the
        // classification adjacent to the accepted removal request.
        decision_ = refreshed_decision;
        decision_resolved_ = true;
        return true;
      }

      bool finish() {
        std::lock_guard lock(mutex_);
        bool topology_query_succeeded = false;
        const auto connected_target = find_target(
          device_path_,
          {},
          &topology_query_succeeded
        );
        if (!topology_query_succeeded) {
          return false;
        }
        if (!connected_target) {
          // Physical-display recovery is independent of SudoVDA ownership. Do not block every
          // later local/remote virtual display while the glasses are unplugged; the durable
          // journal remains untouched and reconnect recovery will classify the target then.
          return true;
        }
        if (!decision_resolved_) {
          decision_ = classify_pre_removal_topology(device_path_);
          if (!decision_) {
            return false;
          }
          decision_resolved_ = true;
        }

        if (!position_restored_) {
          if (!restore_physical_output_position(decision_->desired_rect, device_path_)) {
            return false;
          }
          // Latch this before file cleanup. A failed journal write may be retried, but the old
          // rectangle must never be applied again over a later user move.
          position_restored_ = true;
        }
        if (!recovery_evidence_cleared_) {
          if (decision_->recovery_record_present && !clear_topology_recovery(device_path_)) {
            return false;
          }
          recovery_evidence_cleared_ = true;
        }
        return true;
      }

    private:
      std::mutex mutex_;
      std::optional<pre_removal_topology_decision_t> decision_;
      bool decision_resolved_ = false;
      bool position_restored_ = false;
      bool recovery_evidence_cleared_ = false;
      std::wstring device_path_;
    };

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
        if (!clear_topology_recovery(recovery->device_path)) {
          BOOST_LOG(warning) << "The AR topology is already restored, but its durable recovery record could not be cleared."sv;
          return false;
        }
        return true;
      }
      if (!topology_recovery_should_restore(*recovery, current->rect)) {
        // No topology transaction was in flight and the target no longer occupies any rectangle
        // Apollo confirmed. A user or Windows change owns this layout; retire the stale evidence,
        // but do not start another session until that retirement is durable.
        BOOST_LOG(warning) << "Retiring stale local-AR topology recovery state because the current "sv
                              "display layout no longer matches an Apollo-owned rectangle."sv;
        if (!clear_topology_recovery(recovery->device_path)) {
          BOOST_LOG(error) << "Could not durably retire stale AR topology recovery state."sv;
          return false;
        }
        return true;
      }
      // Exact pending and confirmed prior Apollo positions remain independently recoverable across
      // later re-isolation moves. Unrecognized rectangles were handled above as user-owned.
      if (!restore_physical_output_position(recovery->original_rect, recovery->device_path)) {
        BOOST_LOG(warning) << "Could not recover the AR display topology left by an interrupted session."sv;
        return false;
      }

      if (!clear_topology_recovery(recovery->device_path)) {
        BOOST_LOG(error) << "Recovered the AR display position, but could not durably clear its recovery record."sv;
        return false;
      }
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

    bool recover_connected_saved_topologies_for_remote_handoff() {
      std::vector<topology_recovery_t> recoveries;
      {
        std::lock_guard lock(topology_recovery_mutex);
        const auto loaded = load_topology_recoveries_locked();
        if (!loaded) {
          return false;
        }
        recoveries = *loaded;
      }

      for (const auto &recovery : recoveries) {
        bool topology_query_succeeded = false;
        const auto current = find_target(
          recovery.device_path,
          {},
          &topology_query_succeeded
        );
        if (!topology_query_succeeded) {
          // A remote SudoVDA Add must not race an indeterminate local recovery generation.
          return false;
        }
        if (current && !recover_saved_topology(recovery.device_path)) {
          return false;
        }
        // A disconnected target keeps its record for reconnection but cannot overlap this remote
        // display creation, so it does not block the handoff.
      }
      return true;
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
        const auto hdr_state = VDISPLAY::queryDisplayHDRByName(display_name.c_str());
        const bool hdr_matches = hdr_state && *hdr_state == expected_hdr;
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
      if (!VDISPLAY::setDisplayHDRByName(display_name.c_str(), false)) {
        const auto current_hdr = VDISPLAY::queryDisplayHDRByName(display_name.c_str());
        if (!current_hdr || *current_hdr) {
          BOOST_LOG(error) << "Could not authoritatively reset the local AR virtual display to SDR before HDR setup."sv;
          return false;
        }
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
      local_session_t() = default;

      void initialize(const target_state_t &target, std::stop_token controller_stop_token) {
        original_target_rect_ = target.rect;
        target_device_path_ = target.device_path;
        target_adapter_id_ = target.adapter_id;
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

        // Establish the exact physical rectangle Apollo intends to own before attaching SudoVDA.
        // Adding an IddCx output can itself normalize the desktop, so the first possible topology
        // mutation must already have a durable, narrowly scoped recovery marker.
        const auto pre_add_plan = build_pre_add_isolation_plan(target_device_path_);
        if (!pre_add_plan || !same_rect(pre_add_plan->physical_rect, original_target_rect_)) {
          BOOST_LOG(info) << "AR topology changed before virtual-display creation, or no safe "sv
                             "non-primary/non-cloned isolation row exists; preserving the current layout."sv;
          return;
        }
        if (!begin_topology_recovery_move(
              target_device_path_,
              original_target_rect_,
              pre_add_plan->layout.physical_rect
            )) {
          BOOST_LOG(error) << "Refusing to attach the local AR virtual desktop without a durable pre-add topology transaction."sv;
          return;
        }
        const auto confirmed_pre_add_plan = build_pre_add_isolation_plan(target_device_path_);
        if (!confirmed_pre_add_plan ||
            !same_rect(confirmed_pre_add_plan->physical_rect, original_target_rect_) ||
            !same_rect(confirmed_pre_add_plan->layout.virtual_rect, pre_add_plan->layout.virtual_rect) ||
            !same_rect(confirmed_pre_add_plan->layout.physical_rect, pre_add_plan->layout.physical_rect)) {
          BOOST_LOG(info) << "Desktop topology changed while the pre-add AR recovery marker was being persisted; "sv
                             "cancelling before virtual-display creation."sv;
          if (!cancel_topology_recovery_move(target_device_path_)) {
            BOOST_LOG(error) << "Could not cancel the stale pre-add AR topology transaction."sv;
          }
          return;
        }

        std::string uuid_string = virtual_display_uuid;
        auto uuid = uuid_util::uuid_t::parse(uuid_string);
        static_assert(sizeof(display_guid_) == sizeof(uuid));
        std::memcpy(&display_guid_, &uuid, sizeof(display_guid_));

        const auto created_display = VDISPLAY::createVirtualDisplayOnAdapter(
          virtual_display_uuid,
          virtual_display_name,
          source_width,
          source_height,
          active_target.refresh_millihz,
          display_guid_,
          active_target.adapter_id
        );
        virtual_display_added_ = created_display.added();
        virtual_display_published_ = !created_display.display_name.empty();
        if (created_display.identity) {
          virtual_display_identity_ = *created_display.identity;
          BOOST_LOG(debug) << "Local AR virtual desktop identity: adapter="sv
                           << virtual_display_identity_.AdapterLuid.HighPart << ':'
                           << virtual_display_identity_.AdapterLuid.LowPart
                           << " target="sv << virtual_display_identity_.TargetId << '.';
        }
        virtual_display_name_ = created_display.display_name;
        if (virtual_display_name_.empty()) {
          BOOST_LOG(error) << (virtual_display_added_ ?
                                 "The local AR virtual desktop was added, but Windows did not publish its display name."sv :
                                 "Failed to create the local AR virtual desktop."sv);
          return;
        }

        bool post_add_query_succeeded = false;
        const auto post_add_target = find_target(target_device_path_, {}, &post_add_query_succeeded);
        if (!post_add_query_succeeded || !post_add_target) {
          BOOST_LOG(info) << "Physical AR topology became indeterminate while attaching SudoVDA; "sv
                             "stopping without claiming a new rectangle."sv;
          return;
        }
        if (!same_rect(post_add_target->rect, original_target_rect_) &&
            !same_rect(post_add_target->rect, pre_add_plan->layout.physical_rect)) {
          BOOST_LOG(info) << "SudoVDA attachment coincided with an unexpected physical-display move; "sv
                             "preserving that position as user-owned and rebuilding later."sv;
          return;
        }

        if (VDISPLAY::changeDisplaySettings(virtual_display_name_.c_str(), source_width, source_height, active_target.refresh_millihz) != DISP_CHANGE_SUCCESSFUL) {
          BOOST_LOG(warning) << "The local AR virtual desktop rejected its requested mode."sv;
        }
        auto presentation_target = active_target;
        if (const auto isolated_target = isolate_physical_output(
              virtual_display_name_,
              target_device_path_,
              original_target_rect_
            )) {
          presentation_target = *isolated_target;
          active_target = *isolated_target;
        } else {
          BOOST_LOG(warning) << "Could not establish the first journaled AR source/sink row; "sv
                                "stopping before any unowned presentation can begin."sv;
          return;
        }

        // Let DXGI observe the newly attached output before capture initializes.
        for (int sleep_step = 0; sleep_step < 6 && !controller_stop_token.stop_requested(); ++sleep_step) {
          std::this_thread::sleep_for(50ms);
        }
        if (controller_stop_token.stop_requested()) {
          return;
        }
        if (const auto resolved = resolve_virtual_display(
              virtual_display_identity_,
              {},
              virtual_display_name_
            )) {
          virtual_display_identity_ = resolved->identity;
          virtual_display_name_ = resolved->gdi_name;
          virtual_display_device_path_ = resolved->device_path;
        } else {
          BOOST_LOG(error) << "Could not resolve the newly created local AR virtual display by its driver identity."sv;
          return;
        }

        // Physical Advanced Color is deliberately changed only after the first isolated rectangle
        // is committed. If Windows normalizes the topology during HDR activation, the preexisting
        // v4 record still identifies every exact Apollo-owned position; an unexpected rectangle is
        // preserved as user-owned and causes a clean rebuild.
        auto physical_hdr = active_target.hdr;
        if (physical_hdr.supported && !physical_hdr.limited_by_policy && !physical_hdr.active) {
          if (set_hdr_state(active_target.adapter_id, active_target.target_id, true)) {
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
              if (current->mode != active_target.mode ||
                  current->rect.right - current->rect.left != active_target.rect.right - active_target.rect.left ||
                  current->rect.bottom - current->rect.top != active_target.rect.bottom - active_target.rect.top) {
                BOOST_LOG(info) << "AR output mode changed while enabling HDR; waiting for the topology controller to rebuild the session."sv;
                return;
              }
              active_target = *current;
              presentation_target = *current;
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
        const bool requested_target_hdr = physical_hdr.active;
        if (physical_hdr.supported && !requested_target_hdr) {
          BOOST_LOG(warning) << "The AR display supports HDR, but HDR is not active in its current mode."sv;
        } else if (requested_target_hdr) {
          BOOST_LOG(info) << "AR display HDR is active at "sv << physical_hdr.bits_per_color
                          << " bits per color."sv;
        }

        bool virtual_hdr_active = false;
        if (!configure_virtual_display_hdr(
              virtual_display_name_,
              virtual_display_identity_,
              target_device_path_,
              requested_target_hdr,
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
          if (refreshed_width != expected_width || refreshed_height != expected_height ||
              refreshed_target->refresh_millihz != active_target.refresh_millihz ||
              refreshed_target->mode != active_target.mode) {
            BOOST_LOG(info) << "AR output mode changed during virtual HDR setup; waiting for the topology controller."sv;
            return;
          }
          if (!physical_adapter_contract_valid(active_target.adapter_id, refreshed_target->adapter_id)) {
            BOOST_LOG(info) << "AR output migrated to another graphics adapter during setup; "sv
                               "waiting for the topology controller to rebuild the complete session."sv;
            return;
          }
          if (refreshed_target->hdr.active != physical_hdr.active) {
            BOOST_LOG(info) << "AR output HDR state changed during virtual-display setup; "sv
                               "waiting for one stable topology generation before presentation."sv;
            return;
          }
          if (virtual_hdr_active && !refreshed_target->hdr.active) {
            BOOST_LOG(error) << "Local AR virtual source is HDR while the final physical output is SDR; rebuilding instead of presenting with mismatched color state."sv;
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
              original_target_rect_
            )) {
          presentation_target = *isolated_target;
          active_target = *isolated_target;
        } else {
          BOOST_LOG(warning) << "Could not re-establish the journaled AR row after color-mode setup; "sv
                                "stopping instead of presenting through an unowned topology."sv;
          return;
        }

        // SetDisplayConfig can itself change Advanced Color state or renumber the physical path.
        // Reconcile the final PnP target once more after the last topology mutation and start only
        // from a coherent adapter/mode/color generation.
        if (const auto final_target = find_target(target_device_path_)) {
          const int final_width = final_target->rect.right - final_target->rect.left;
          const int final_height = final_target->rect.bottom - final_target->rect.top;
          const int expected_width = active_target.rect.right - active_target.rect.left;
          const int expected_height = active_target.rect.bottom - active_target.rect.top;
          if (!physical_adapter_contract_valid(active_target.adapter_id, final_target->adapter_id) ||
              final_width != expected_width || final_height != expected_height ||
              final_target->refresh_millihz != active_target.refresh_millihz ||
              final_target->mode != active_target.mode ||
              final_target->hdr.active != active_target.hdr.active ||
              (virtual_hdr_active && !final_target->hdr.active)) {
            BOOST_LOG(info) << "AR output contract changed during final placement; waiting for the topology controller to rebuild from the stable state."sv;
            return;
          }
          active_target = *final_target;
          presentation_target = *final_target;
        } else {
          BOOST_LOG(error) << "Could not resolve the final physical AR output after placement."sv;
          return;
        }

        virtual_hdr_active_ = virtual_hdr_active;
        start_presenter(presentation_target);
        ready_ = true;
      }

      void pause_presenter(const std::optional<target_state_t> &target) {
        stop_presenter();
        if (target && live_target_) {
          std::lock_guard lock(live_target_->mutex);
          live_target_->rect = target->rect;
          live_target_->display_name = target->gdi_name;
        }
        refresh_pointer_isolation();
      }

      void resume_presenter(const target_state_t &target) {
        start_presenter(target);
      }

      void refresh_pointer_isolation() {
        std::wstring source_name;
        {
          std::lock_guard lock(virtual_display_mutex_);
          source_name = virtual_display_name_;
        }
        if (!source_name.empty()) {
          platf::dxgi::refresh_local_presenter_pointer_isolation(
            platf::to_utf8(source_name),
            live_target_,
            cursor_clip_
          );
        }
      }

      void release_pointer_isolation() {
        if (!cursor_clip_) {
          return;
        }
        for (int attempt = 0; attempt < 8; ++attempt) {
          if (cursor_clip_->restore()) {
            return;
          }
          std::this_thread::sleep_for(5ms);
        }
      }

      bool virtual_display_authoritatively_absent() {
        std::lock_guard lock(virtual_display_mutex_);
        const auto probe = probe_virtual_display(
          virtual_display_identity_,
          virtual_display_device_path_,
          virtual_display_name_
        );
        if (probe.presence != virtual_display_presence_e::absent) {
          virtual_display_absence_started_.reset();
          virtual_display_absence_observations_ = 0;
          if (probe.resolved) {
            virtual_display_identity_ = probe.resolved->identity;
            virtual_display_name_ = probe.resolved->gdi_name;
            virtual_display_device_path_ = probe.resolved->device_path;
          }
          return false;
        }

        const auto now = std::chrono::steady_clock::now();
        if (!virtual_display_absence_started_) {
          virtual_display_absence_started_ = now;
          virtual_display_absence_observations_ = 1;
          return false;
        }
        ++virtual_display_absence_observations_;
        return virtual_display_absence_observations_ >= virtual_display_absence_confirmations &&
               now - *virtual_display_absence_started_ >= virtual_display_absence_grace;
      }

      std::optional<target_state_t> reconfigure_target(
        const target_state_t &target,
        std::stop_token controller_stop_token
      ) {
        stop_presenter();
        if (controller_stop_token.stop_requested() ||
            target.device_path != target_device_path_ ||
            !same_luid(target.adapter_id, target_adapter_id_) ||
            target.mode == presentation_mode_e::unsupported) {
          return std::nullopt;
        }

        const auto rebased_original = prepare_topology_recovery_for_mode_change(
          target_device_path_,
          original_target_rect_,
          target.rect
        );
        if (!rebased_original) {
          BOOST_LOG(warning) << "Could not durably rebase AR topology recovery for the new physical mode; preserving the virtual desktop and retrying later."sv;
          return std::nullopt;
        }
        original_target_rect_ = *rebased_original;

        if (!refresh_virtual_display_reference()) {
          BOOST_LOG(warning) << "The local AR virtual desktop could not be resolved after the physical mode change; preserving it for a later topology retry."sv;
          return std::nullopt;
        }

        const auto mode_status = VDISPLAY::changeDisplaySettings(
          virtual_display_name_.c_str(),
          source_width,
          source_height,
          target.refresh_millihz
        );
        if (mode_status != DISP_CHANGE_SUCCESSFUL) {
          // The source is still usable at its previous refresh. Capture/presentation pacing follows
          // the physical target, so a rejected refresh update must not destroy the desktop.
          BOOST_LOG(warning) << "The persistent local AR virtual desktop rejected refresh "sv
                             << (target.refresh_millihz / 1000.0)
                             << " Hz; continuing with its current source refresh."sv;
        }

        for (int sleep_step = 0; sleep_step < 6 && !controller_stop_token.stop_requested(); ++sleep_step) {
          std::this_thread::sleep_for(50ms);
        }
        if (controller_stop_token.stop_requested() || !refresh_virtual_display_reference()) {
          return std::nullopt;
        }

        // Unknown physical color state must not preserve an old HDR source: if the new mode is
        // actually SDR, that mismatch makes the presenter reject every target generation. SDR is
        // the safe common contract and remains color-managed when the sink later proves to be HDR.
        const bool desired_virtual_hdr = target.hdr.known && target.hdr.active;
        const auto current_virtual_hdr = VDISPLAY::queryDisplayHDRByName(virtual_display_name_.c_str());
        if (!current_virtual_hdr) {
          return std::nullopt;
        }
        auto wait_for_source_color = [&](bool expected_hdr) {
          for (int attempt = 0; attempt < 20 && !controller_stop_token.stop_requested(); ++attempt) {
            std::this_thread::sleep_for(50ms);
            if (!refresh_virtual_display_reference()) {
              continue;
            }
            const auto observed_hdr = VDISPLAY::queryDisplayHDRByName(virtual_display_name_.c_str());
            if (observed_hdr && *observed_hdr == expected_hdr) {
              return true;
            }
          }
          return false;
        };

        bool configured_virtual_hdr = *current_virtual_hdr;
        if (configured_virtual_hdr != desired_virtual_hdr) {
          const bool setting_accepted = VDISPLAY::setDisplayHDRByName(
            virtual_display_name_.c_str(),
            desired_virtual_hdr
          );
          if (setting_accepted && wait_for_source_color(desired_virtual_hdr)) {
            configured_virtual_hdr = desired_virtual_hdr;
          } else if (desired_virtual_hdr) {
            // Match initial startup: an HDR physical sink may consume a color-managed SDR
            // swapchain when Windows/SudoVDA rejects source HDR. Never remove the desktop merely
            // because its virtual output cannot activate Advanced Color in this mode.
            BOOST_LOG(warning) << "Persistent local AR source HDR was unavailable after the mode switch; using color-managed SDR presentation."sv;
            const auto fallback_state = VDISPLAY::queryDisplayHDRByName(virtual_display_name_.c_str());
            bool fallback_ready = fallback_state && !*fallback_state;
            if (!fallback_ready) {
              const bool fallback_accepted = VDISPLAY::setDisplayHDRByName(
                virtual_display_name_.c_str(),
                false
              );
              if (fallback_accepted) {
                fallback_ready = wait_for_source_color(false);
              } else {
                const auto observed_fallback = VDISPLAY::queryDisplayHDRByName(
                  virtual_display_name_.c_str()
                );
                fallback_ready = observed_fallback && !*observed_fallback;
              }
            }
            if (!fallback_ready) {
              BOOST_LOG(warning) << "The persistent local AR source could not settle to its SDR fallback; retaining the desktop for retry."sv;
              return std::nullopt;
            }
            configured_virtual_hdr = false;
          } else {
            // An HDR virtual source cannot be presented coherently to an SDR physical mode.
            BOOST_LOG(warning) << "The persistent local AR source could not return to SDR; retaining the desktop for retry."sv;
            return std::nullopt;
          }
        }
        virtual_hdr_active_ = configured_virtual_hdr;

        const auto presentation_target = isolate_physical_output(
          virtual_display_name_,
          target_device_path_,
          original_target_rect_
        );
        if (!presentation_target ||
            !same_luid(presentation_target->adapter_id, target_adapter_id_) ||
            presentation_target->mode != target.mode ||
            (target.hdr.known && presentation_target->hdr.active != target.hdr.active)) {
          BOOST_LOG(warning) << "The new AR physical mode could not be safely re-isolated; preserving the virtual desktop for retry."sv;
          return std::nullopt;
        }
        const auto final_virtual_hdr = VDISPLAY::queryDisplayHDRByName(virtual_display_name_.c_str());
        if (!final_virtual_hdr || *final_virtual_hdr != virtual_hdr_active_) {
          BOOST_LOG(warning) << "The persistent local AR source color state changed during final isolation; retaining the desktop for a coherent retry."sv;
          return std::nullopt;
        }
        start_presenter(*presentation_target);
        BOOST_LOG(info) << "Local AR presentation switched in place to "sv
                        << mode_name(presentation_target->mode)
                        << " without recreating its virtual desktop."sv;
        return presentation_target;
      }

      bool prepare_same_output_teardown(const target_state_t &target) {
        stop_presenter();
        if (target.device_path != target_device_path_ ||
            !same_luid(target.adapter_id, target_adapter_id_) ||
            !valid_rect(target.rect)) {
          return false;
        }
        const auto rebased_original = prepare_topology_recovery_for_mode_change(
          target_device_path_,
          original_target_rect_,
          target.rect
        );
        if (!rebased_original) {
          BOOST_LOG(warning) << "Could not durably rebase local-AR recovery evidence before an incompatible same-output transition; retaining the virtual desktop for retry."sv;
          return false;
        }
        original_target_rect_ = *rebased_original;
        return true;
      }

      ~local_session_t() {
        stop_presenter();
        // The shared clip deliberately survives presenter-only restarts, but whole-session
        // teardown restores it before Windows removes the source rectangle it was based on.
        release_pointer_isolation();
        cursor_clip_.reset();

        // Removing SudoVDA can normalize the physical sink before recovery inspects it. Classify
        // ownership while the source/sink row still exists and remember the exact rectangle that
        // must survive teardown. An unrecognized rectangle is user-owned and is preserved rather
        // than being reclassified after Windows removes the virtual source.
        const auto pre_removal_topology = target_device_path_.empty() ?
                                            std::nullopt :
                                            classify_pre_removal_topology(target_device_path_);
        const auto topology_cleanup = target_device_path_.empty() ?
                                        std::shared_ptr<pre_removal_topology_cleanup_t> {} :
                                        std::make_shared<pre_removal_topology_cleanup_t>(
                                          pre_removal_topology,
                                          target_device_path_
                                        );
        bool virtual_display_retired = !virtual_display_added_;
        if (virtual_display_added_) {
          const auto prepare_removal = [topology_cleanup]() {
            return topology_cleanup->prepare_removal();
          };
          const auto finish_topology_cleanup = [topology_cleanup]() {
            return topology_cleanup->finish();
          };
          const bool retirement_tracked = begin_local_virtual_display_retirement(
            display_guid_,
            virtual_display_identity_,
            virtual_display_device_path_,
            virtual_display_name_,
            virtual_display_published_,
            prepare_removal,
            finish_topology_cleanup
          );
          if (!retirement_tracked) {
            BOOST_LOG(error) << "Local AR teardown could not register its virtual-display retirement; retaining the existing cleanup barrier."sv;
          } else {
            virtual_display_retired = wait_for_local_virtual_display_retirement_impl(
              3s,
              true
            );
            if (!virtual_display_retired) {
              BOOST_LOG(warning) << "The local AR virtual desktop is still retiring; subsequent local or remote display creation will wait for its stable identity to disappear."sv;
            }
          }
        }
        if (!target_device_path_.empty()) {
          if (!virtual_display_retired) {
            // The source can still occupy the middle of the source/sink row. Moving the physical
            // sink while that source remains present can overlap displays or make Windows perform
            // another normalization. Leave the recovery evidence intact and finish only after the
            // retirement barrier has authoritatively observed the exact SudoVDA identity absent.
            BOOST_LOG(warning) << "Deferring AR topology restoration until the local virtual display is authoritatively retired."sv;
            return;
          }
          const auto deadline = std::chrono::steady_clock::now() + 3s;
          // A retired virtual display completed this callback as part of the authoritative
          // retirement barrier. Do not apply it a second time: the user may move the monitor as
          // soon as SudoVDA disappears. A failed Add has no retirement callback, so clean it here.
          bool recovery_complete = virtual_display_added_;
          while (!recovery_complete) {
            recovery_complete = topology_cleanup->finish();
            if (recovery_complete || std::chrono::steady_clock::now() >= deadline) {
              break;
            }
            std::this_thread::sleep_for(100ms);
          }
          if (!recovery_complete) {
            if (pre_removal_topology && !pre_removal_topology->apollo_owned) {
              BOOST_LOG(warning) << "Could not reapply the pre-removal user-owned AR rectangle after "sv
                                    "SudoVDA retirement; no Apollo ownership is claimed for that position."sv;
            } else {
              BOOST_LOG(warning) << "Could not complete local-AR topology recovery during teardown; "sv
                                    "the durable record is retained for the next stable observation."sv;
            }
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
        const auto refreshed_name = refresh_virtual_display_reference();
        if (!refreshed_name) {
          return std::nullopt;
        }
        const auto target = isolate_physical_output(
          *refreshed_name,
          target_device_path_,
          original_target_rect_
        );
        if (!target) {
          return std::nullopt;
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
      void stop_presenter() {
        presenter_.request_stop();
        if (presenter_.joinable()) {
          presenter_.join();
        }
        running_.store(false);
      }

      std::optional<std::wstring> refresh_virtual_display_reference() {
        std::lock_guard lock(virtual_display_mutex_);
        const auto resolved = resolve_virtual_display(
          virtual_display_identity_,
          virtual_display_device_path_,
          virtual_display_name_
        );
        if (!resolved) {
          return std::nullopt;
        }
        if (!same_virtual_display_identity(virtual_display_identity_, resolved->identity) ||
            virtual_display_name_ != resolved->gdi_name) {
          BOOST_LOG(info) << "Local AR virtual desktop was renumbered ["sv
                          << platf::to_utf8(virtual_display_name_) << " -> "sv
                          << platf::to_utf8(resolved->gdi_name) << ", target "sv
                          << virtual_display_identity_.TargetId << " -> "sv
                          << resolved->identity.TargetId << "]."sv;
        }
        virtual_display_identity_ = resolved->identity;
        virtual_display_name_ = resolved->gdi_name;
        virtual_display_device_path_ = resolved->device_path;
        virtual_display_absence_started_.reset();
        virtual_display_absence_observations_ = 0;
        return virtual_display_name_;
      }

      void start_presenter(const target_state_t &presentation_target) {
        stop_presenter();
        if (!live_target_) {
          live_target_ = std::make_shared<platf::dxgi::local_presenter_config_t::target_t>();
        }
        std::string source_display_name;
        std::wstring source_device_path;
        {
          std::lock_guard lock(virtual_display_mutex_);
          source_display_name = platf::to_utf8(virtual_display_name_);
          source_device_path = virtual_display_device_path_;
        }
        if (source_display_name.empty() || source_device_path.empty() ||
            presentation_target.device_path.empty()) {
          BOOST_LOG(error) << "Local AR presenter refused to open without stable source/sink identities."sv;
          failed_.store(true);
          running_.store(false);
          return;
        }
        {
          std::lock_guard lock(live_target_->mutex);
          live_target_->rect = presentation_target.rect;
          live_target_->display_name = presentation_target.gdi_name;
          // Seed the presenter with the stable paths established by the controller before it opens
          // either volatile \\.\DISPLAYn name. Later generations may verify these paths but must
          // never learn a replacement identity from a recycled GDI name.
          if (live_target_->source_device_path.empty()) {
            live_target_->source_device_path = source_device_path;
          }
          if (live_target_->target_device_path.empty()) {
            live_target_->target_device_path = presentation_target.device_path;
          }
        }
        if (!cursor_clip_) {
          cursor_clip_ = std::make_shared<platf::dxgi::local_presenter_cursor_clip_t>();
        }

        platf::dxgi::local_presenter_config_t presenter_config;
        presenter_config.source_display_name = std::move(source_display_name);
        presenter_config.target_rect = presentation_target.rect;
        presenter_config.target_adapter_id = presentation_target.adapter_id;
        presenter_config.target_refresh_millihz = presentation_target.refresh_millihz;
        presenter_config.hdr = virtual_hdr_active_;
        presenter_config.sbs_mode = presentation_target.mode == presentation_mode_e::sbs_ai ?
                                      ::video::SBS_AI :
                                      ::video::SBS_OFF;
        presenter_config.sbs_config = config::video.sbs;
        presenter_config.live_target = live_target_;
        presented_frames_ = std::make_shared<std::atomic<std::uint64_t>>(0);
        presenter_config.presented_frames = presented_frames_;
        presenter_config.cursor_clip = cursor_clip_;

        failed_.store(false);
        running_.store(true);
        presenter_ = std::jthread([this, presenter_config](std::stop_token stop_token) mutable {
          auto reinit_window_started = std::chrono::steady_clock::now();
          int consecutive_reinits = 0;
          while (!stop_token.stop_requested()) {
            // Stability is per continuous presenter attempt. Do not let several short failed
            // attempts accumulate enough frames to be mistaken for one stable session.
            presenter_config.presented_frames->store(0, std::memory_order_relaxed);
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
                               << consecutive_reinits << " reinitializations; pausing for a controller retry."sv;
              break;
            }

            // Recreate only capture/presentation resources. The SudoVDA source and its desktop
            // contents must survive DXGI/topology churn, including a physical 2D/SBS mode switch.
            std::this_thread::sleep_for(100ms);
            if (const auto refreshed = refresh_virtual_display_reference()) {
              presenter_config.source_display_name = platf::to_utf8(*refreshed);
            } else {
              BOOST_LOG(error) << "Local AR virtual desktop could not be resolved; pausing for a controller retry."sv;
              break;
            }
          }
          // Any exit not requested by the topology controller should be retried, including a
          // user-closed or driver-closed presenter window that otherwise exits cleanly.
          failed_.store(!stop_token.stop_requested());
          running_.store(false);
        });
      }

      GUID display_guid_ {};
      SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT virtual_display_identity_ {};
      RECT original_target_rect_ {};
      LUID target_adapter_id_ {};
      std::wstring target_device_path_;
      std::wstring virtual_display_name_;
      std::wstring virtual_display_device_path_;
      bool virtual_display_added_ = false;
      bool virtual_display_published_ = false;
      bool virtual_hdr_active_ = false;
      bool ready_ = false;
      std::mutex virtual_display_mutex_;
      std::optional<std::chrono::steady_clock::time_point> virtual_display_absence_started_;
      unsigned virtual_display_absence_observations_ = 0;
      std::shared_ptr<platf::dxgi::local_presenter_config_t::target_t> live_target_;
      std::shared_ptr<std::atomic<std::uint64_t>> presented_frames_;
      std::shared_ptr<platf::dxgi::local_presenter_cursor_clip_t> cursor_clip_;
      std::jthread presenter_;
      std::atomic<bool> running_ {false};
      std::atomic<bool> failed_ {false};
    };

    class controller_t final: public platf::deinit_t {
    public:
      controller_t() {
        // The thread must start only after every state member below has completed construction.
        // Starting a jthread from the member initializer list lets run() observe later members
        // before their lifetime begins.
        worker_ = std::jthread([this](std::stop_token stop_token) {
          run(stop_token);
        });
      }

      ~controller_t() override {
        worker_.request_stop();
        if (worker_.joinable()) {
          worker_.join();
        }
      }

    private:
      void reset_failure_backoff() {
        retry_after_ = {};
        presenter_retry_after_ = {};
        failure_retry_delay_ = failed_session_retry;
      }

      void arm_presenter_retry() {
        presenter_retry_after_ = std::chrono::steady_clock::now() + failure_retry_delay_;
        failure_retry_delay_ = std::min(failure_retry_delay_ * 2, maximum_failed_session_retry);
      }

      void schedule_failure_retry() {
        BOOST_LOG(warning) << "Local AR session retry deferred for "sv
                           << std::chrono::duration_cast<std::chrono::seconds>(failure_retry_delay_).count()
                           << " seconds."sv;
        retry_after_ = std::chrono::steady_clock::now() + failure_retry_delay_;
        failure_retry_delay_ = std::min(failure_retry_delay_ * 2, maximum_failed_session_retry);
      }

      void stop_session() {
        transition_presenter_paused_ = false;
        incompatible_transition_started_.reset();
        std::unique_ptr<local_session_t> retiring_session;
        {
          std::lock_guard lock(ownership_mutex);
          if (local_session_construction_stop) {
            local_session_construction_stop->request_stop();
          }
          retiring_session = std::move(session_);
        }

        if (!retiring_session) {
          release_local_virtual_display_claim_impl();
          return;
        }

        // Presenter shutdown and SudoVDA topology removal can take seconds. Keep ownership marked
        // local until teardown is complete, but never hold ownership_mutex across those waits so a
        // remote launch can publish its reservation and wait on the condition variable.
        retiring_session.reset();
        release_local_virtual_display_claim_impl();
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
        std::stop_source construction_stop;
        std::stop_callback controller_stop_callback(stop_token, [&construction_stop]() {
          construction_stop.request_stop();
        });
        const auto handoff = proc::proc.prepare_local_ar_handoff(construction_stop);
        if (handoff != proc::local_ar_handoff_e::ready) {
          if (!deferred_for_remote_) {
            BOOST_LOG(info) << (handoff == proc::local_ar_handoff_e::remote_busy ? "Local AR presentation is waiting for the active/connecting remote virtual-display session."sv : "Local AR presentation is waiting for inactive remote-display cleanup."sv);
          }
          deferred_for_remote_ = true;
          retry_after_ = std::chrono::steady_clock::now() + failed_session_retry;
          return;
        }
        // prepare_local_ar_handoff() atomically claims local ownership. Keep release exception-safe,
        // and declare the guard before the candidate so a partially constructed session retires its
        // display before remote creation can observe ownership as free.
        auto release_claim = util::fail_guard([]() {
          release_local_virtual_display_claim_impl();
        });
        if (construction_stop.stop_requested()) {
          return;
        }
        if (!recover_saved_topology(target.device_path) || construction_stop.stop_requested()) {
          BOOST_LOG(warning) << "Deferring local AR startup until this display's previous desktop topology can be restored."sv;
          schedule_failure_retry();
          return;
        }

        auto candidate = std::make_unique<local_session_t>();
        candidate->initialize(target, construction_stop.get_token());
        bool rejected_for_remote = false;
        {
          std::lock_guard lock(ownership_mutex);
          rejected_for_remote = remote_blocks_local_locked(std::chrono::steady_clock::now());
          if (candidate->valid() && !construction_stop.stop_requested() && !rejected_for_remote) {
            session_ = std::move(candidate);
            local_session_construction_stop.reset();
            local_session_present = true;
            ownership_changed.notify_all();
            if (deferred_for_remote_) {
              BOOST_LOG(info) << "Remote virtual-display ownership ended; starting deferred local AR presentation."sv;
            }
            deferred_for_remote_ = false;
            session_stability_confirmed_ = false;
            transition_presenter_paused_ = false;
            incompatible_transition_started_.reset();
            // A spawned thread is not yet a healthy presenter. Arm bounded backoff now and reset it
            // only after the controller observes a sustained run of presented frames.
            arm_presenter_retry();
            release_claim.disable();
            return;
          }
        }

        // Destruction removes any partially created virtual display. Keep local ownership true
        // until that removal and topology restoration finish, so the remote side cannot overlap.
        candidate.reset();
        if (rejected_for_remote) {
          deferred_for_remote_ = true;
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
        if (target->is_primary || target->is_cloned) {
          BOOST_LOG(warning) << "AR display cannot start local presentation while it is "sv
                             << (target->is_primary ? "the Windows primary display"sv : "part of a cloned display source"sv)
                             << "; waiting for a unique extended-desktop topology."sv;
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
            // Unknown time cannot count toward either stable debounce or destructive disconnect
            // grace: require a fresh continuous sequence of authoritative observations afterward.
            const auto indeterminate_at = std::chrono::steady_clock::now();
            // Reset unconditionally: a paused transition may already have observed applied_ again,
            // and unknown time must not count toward its return-to-stable resume debounce either.
            pending_since = indeterminate_at;
            if (incompatible_transition_started_) {
              incompatible_transition_started_ = indeterminate_at;
            }
            if (transition_presenter_paused_ && session_) {
              session_->refresh_pointer_isolation();
            }
            std::this_thread::sleep_for(topology_poll_interval);
            continue;
          }
          const auto now = std::chrono::steady_clock::now();
          if (observed != pending) {
            pending = observed;
            pending_since = now;
            // Stop using old-size textures immediately, but keep SudoVDA attached during the
            // debounce interval. A physical 2D/SBS switch is allowed to pass through transient
            // modes; removing the source here would make Windows migrate all of its windows.
            if (observed != applied_ && !same_presentation_contract(observed, applied_)) {
              if (session_) {
                session_->pause_presenter(observed);
                transition_presenter_paused_ = true;
              }
            }
            // Every distinct incompatible generation gets its own grace period. Churn through
            // several temporary modes must not accumulate enough time to make the newest 750 ms
            // observation look like a stable disconnect.
            if (session_ && observed != applied_ &&
                !can_reconfigure_local_session(applied_, observed)) {
              incompatible_transition_started_ = now;
            } else {
              incompatible_transition_started_.reset();
            }
          }
          if (transition_presenter_paused_ && session_) {
            session_->refresh_pointer_isolation();
          }

          if (pending != applied_ && now - pending_since >= topology_debounce) {
            if (pending && applied_ && same_presentation_contract(pending, applied_)) {
              if (session_) {
                if (now < retry_after_) {
                  std::this_thread::sleep_for(topology_poll_interval);
                  continue;
                }
                const auto isolated = session_->re_isolate_target();
                if (!isolated) {
                  session_->pause_presenter(pending);
                  transition_presenter_paused_ = true;
                  if (session_->virtual_display_authoritatively_absent()) {
                    BOOST_LOG(warning) << "The local AR virtual desktop is authoritatively absent; rebuilding the session."sv;
                    apply(pending, stop_token);
                  } else {
                    BOOST_LOG(warning) << "AR display position drifted and could not yet be re-isolated; preserving its virtual desktop for retry."sv;
                    schedule_failure_retry();
                    pending_since = now;
                  }
                } else {
                  pending = isolated;
                  pending_since = now;
                  applied_ = pending;
                  if (transition_presenter_paused_) {
                    session_->resume_presenter(*isolated);
                    session_stability_confirmed_ = false;
                    transition_presenter_paused_ = false;
                    incompatible_transition_started_.reset();
                    arm_presenter_retry();
                  }
                  BOOST_LOG(info) << "AR display position changed; restored physical-output isolation without recreating its virtual desktop."sv;
                }
              } else {
                applied_ = pending;
              }
            } else if (session_ && can_reconfigure_local_session(applied_, pending)) {
              if (now < retry_after_) {
                std::this_thread::sleep_for(topology_poll_interval);
                continue;
              }
              const auto reconfigured = session_->reconfigure_target(*pending, stop_token);
              if (!reconfigured) {
                if (session_->virtual_display_authoritatively_absent()) {
                  BOOST_LOG(warning) << "The local AR virtual desktop disappeared during its mode switch; rebuilding the session."sv;
                  apply(pending, stop_token);
                } else {
                  BOOST_LOG(warning) << "AR presentation mode change is waiting for a stable in-place reconfiguration; the virtual desktop remains attached."sv;
                  schedule_failure_retry();
                  pending_since = now;
                }
              } else {
                pending = reconfigured;
                applied_ = reconfigured;
                pending_since = now;
                session_stability_confirmed_ = false;
                transition_presenter_paused_ = false;
                incompatible_transition_started_.reset();
                arm_presenter_retry();
              }
            } else if (!session_ || !incompatible_transition_started_ ||
                       now - *incompatible_transition_started_ >= incompatible_transition_grace) {
              if (session_ && pending && applied_ && same_physical_output(applied_, pending)) {
                if (now < retry_after_) {
                  std::this_thread::sleep_for(topology_poll_interval);
                  continue;
                }
                if (!session_->prepare_same_output_teardown(*pending)) {
                  schedule_failure_retry();
                  pending_since = now;
                  incompatible_transition_started_ = now;
                  std::this_thread::sleep_for(topology_poll_interval);
                  continue;
                }
              }
              apply(pending, stop_token);
            }
          } else if (pending == applied_ && transition_presenter_paused_ && session_ && pending &&
                     now - pending_since >= topology_debounce && now >= retry_after_) {
            // A transient display generation can settle back to the original contract after the
            // old presenter was deliberately paused. Restart resources on the still-attached
            // desktop instead of leaving a valid session silently idle.
            const auto reconfigured = session_->reconfigure_target(*pending, stop_token);
            if (!reconfigured) {
              if (session_->virtual_display_authoritatively_absent()) {
                apply(pending, stop_token);
              } else {
                schedule_failure_retry();
                pending_since = now;
              }
            } else {
              pending = reconfigured;
              applied_ = reconfigured;
              pending_since = now;
              session_stability_confirmed_ = false;
              transition_presenter_paused_ = false;
              incompatible_transition_started_.reset();
              arm_presenter_retry();
            }
          } else if (pending == applied_ && session_ && !session_stability_confirmed_ &&
                     session_->stable()) {
            // Construction only proves that a presenter thread was spawned. Reset exponential
            // retry backoff after sustained scanout so permanent DXGI/swapchain failures cannot
            // recreate the whole topology every two seconds forever.
            session_stability_confirmed_ = true;
            reset_failure_backoff();
          } else if (pending == applied_ && session_ && !session_->running() && session_->failed() &&
                     now >= retry_after_ && now >= presenter_retry_after_) {
            transition_presenter_paused_ = true;
            session_->pause_presenter(pending);
            if (session_->virtual_display_authoritatively_absent()) {
              BOOST_LOG(warning) << "Local AR presentation lost its virtual desktop; scheduling a complete restart."sv;
              stop_session();
              schedule_failure_retry();
            } else if (pending) {
              BOOST_LOG(warning) << "Local AR presentation failed; restarting presenter resources while retaining its virtual desktop."sv;
              const auto reconfigured = session_->reconfigure_target(*pending, stop_token);
              if (reconfigured) {
                pending = reconfigured;
                applied_ = reconfigured;
                pending_since = now;
                session_stability_confirmed_ = false;
                transition_presenter_paused_ = false;
                arm_presenter_retry();
              } else {
                schedule_failure_retry();
              }
            }
          } else if (pending == applied_ && applied_ && !session_ &&
                     applied_->mode != presentation_mode_e::unsupported &&
                     !applied_->is_primary && !applied_->is_cloned && now >= retry_after_) {
            start_session(*applied_, stop_token);
          }

          std::this_thread::sleep_for(topology_poll_interval);
        }

        stop_session();
      }

      std::optional<target_state_t> applied_;
      std::unique_ptr<local_session_t> session_;
      std::chrono::steady_clock::time_point retry_after_ {};
      std::chrono::steady_clock::time_point presenter_retry_after_ {};
      std::chrono::seconds failure_retry_delay_ {failed_session_retry};
      bool deferred_for_remote_ = false;
      bool session_stability_confirmed_ = false;
      bool transition_presenter_paused_ = false;
      std::optional<std::chrono::steady_clock::time_point> incompatible_transition_started_;
      std::jthread worker_;
    };
  }  // namespace

#ifdef SUNSHINE_TESTS
  bool detail::primary_source_is_authoritative_for_test(
    std::wstring_view primary_gdi_name,
    const std::vector<std::wstring> &active_gdi_names
  ) {
    return primary_source_is_authoritative(primary_gdi_name, active_gdi_names);
  }

  bool detail::physical_adapter_contract_valid_for_test(
    const LUID &physical_before,
    const LUID &physical_after,
    const LUID &virtual_adapter
  ) {
    (void) virtual_adapter;
    return physical_adapter_contract_valid(physical_before, physical_after);
  }

  detail::linear_layout_t detail::compute_linear_layout_for_test(
    const RECT &anchor,
    LONG virtual_width,
    LONG virtual_height,
    LONG physical_width,
    LONG physical_height
  ) {
    const auto layout = compute_linear_layout(
      anchor,
      virtual_width,
      virtual_height,
      physical_width,
      physical_height
    );
    return {layout.virtual_rect, layout.physical_rect};
  }

  std::optional<detail::anchor_candidate_t> detail::select_anchor_for_test(
    const std::vector<detail::anchor_candidate_t> &candidates
  ) {
    std::optional<::ar_glasses::anchor_candidate_t> selected;
    for (const auto &candidate : candidates) {
      ::ar_glasses::anchor_candidate_t converted {
        candidate.rect,
        candidate.source_adapter_id,
        candidate.source_id,
      };
      if (!selected || anchor_is_better(converted, *selected)) {
        selected = converted;
      }
    }
    if (!selected) {
      return std::nullopt;
    }
    return detail::anchor_candidate_t {
      selected->rect,
      selected->source_adapter_id,
      selected->source_id,
    };
  }

  bool detail::source_is_cloned_for_test(
    const std::vector<detail::topology_path_identity_t> &paths,
    std::size_t selected_index
  ) {
    if (selected_index >= paths.size()) {
      return false;
    }
    std::vector<active_source_t> sources;
    sources.reserve(paths.size());
    for (const auto &path : paths) {
      active_source_t source;
      source.source_adapter_id = path.source_adapter_id;
      source.source_id = path.source_id;
      source.target = {path.target_adapter_id, path.target_id};
      sources.emplace_back(source);
    }
    return source_is_cloned(sources, sources[selected_index]);
  }

  bool detail::isolated_layout_matches_for_test(
    const detail::linear_layout_t &expected,
    const RECT &virtual_rect,
    const RECT &physical_rect
  ) {
    isolation_plan_t plan;
    plan.snapshot.sources.resize(2);
    plan.virtual_index = 0;
    plan.physical_index = 1;
    plan.snapshot.sources[0].rect = virtual_rect;
    plan.snapshot.sources[1].rect = physical_rect;
    plan.layout = {expected.virtual_rect, expected.physical_rect};
    return layout_is_exact_and_adjacent(plan);
  }

  detail::topology_recovery_parse_result_t detail::parse_topology_recovery_json_for_test(
    std::string_view contents
  ) {
    try {
      auto document = parse_topology_recovery_document(nlohmann::json::parse(contents));
      return {
        .valid = true,
        .rewrite_required = document.rewrite_required,
        .record_count = document.recoveries.size(),
        .normalized_json = serialize_topology_recovery_document(document.recoveries).dump(),
      };
    } catch (const std::exception &) {
      return {};
    }
  }

  bool detail::topology_recovery_should_restore_for_test(
    std::string_view contents,
    const RECT &current_rect
  ) {
    try {
      const auto document = parse_topology_recovery_document(nlohmann::json::parse(contents));
      return document.recoveries.size() == 1 &&
             topology_recovery_should_restore(document.recoveries.front(), current_rect);
    } catch (const std::exception &) {
      return false;
    }
  }

  bool detail::local_session_can_reconfigure_for_test(
    const local_session_contract_t &before,
    const local_session_contract_t &after
  ) {
    auto make_target = [](const local_session_contract_t &contract) {
      target_state_t target;
      target.device_path = contract.device_path;
      target.adapter_id = contract.adapter_id;
      target.mode = contract.mode;
      target.hdr.known = contract.hdr_known;
      target.hdr.supported = contract.hdr_supported;
      target.hdr.active = contract.hdr_active;
      target.hdr.limited_by_policy = contract.hdr_limited_by_policy;
      target.is_primary = contract.is_primary;
      target.is_cloned = contract.is_cloned;
      return target;
    };
    return can_reconfigure_local_session(make_target(before), make_target(after));
  }

  bool detail::retirement_identity_matches_for_test(
    const virtual_display_identity_contract_t &retiring,
    const virtual_display_identity_contract_t &observed
  ) {
    retired_local_virtual_display_t record;
    record.identity = {retiring.adapter_id, retiring.target_id};
    record.device_path = retiring.device_path;
    record.gdi_name = retiring.gdi_name;
    return matches_retiring_local_virtual_display(
      record,
      observed.adapter_id,
      observed.target_id,
      observed.device_path,
      observed.friendly_name
    );
  }

  std::optional<std::string> detail::rebase_topology_recovery_json_for_test(
    std::string_view contents,
    const RECT &previous_original_rect,
    const RECT &current_rect
  ) {
    try {
      auto document = parse_topology_recovery_document(nlohmann::json::parse(contents));
      if (document.recoveries.size() != 1 ||
          !rebase_topology_recovery_for_mode_change(
            document.recoveries.front(),
            previous_original_rect,
            current_rect
          )) {
        return std::nullopt;
      }
      return serialize_topology_recovery_document(document.recoveries).dump();
    } catch (const std::exception &) {
      return std::nullopt;
    }
  }
#endif

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

  bool write_config_with_devices(std::string_view contents) {
    std::lock_guard lock(device_mutex);
    auto merged = replace_managed_config_value(std::string(contents), serialize_devices_locked());
    return write_config_atomically(merged);
  }

  bool try_claim_local_virtual_display(const std::stop_source &construction_stop) {
    std::lock_guard lock(ownership_mutex);
    if (construction_stop.stop_requested() || local_session_present ||
        retired_local_virtual_display ||
        remote_blocks_local_locked(std::chrono::steady_clock::now())) {
      return false;
    }
    local_session_present = true;
    local_session_construction_stop = construction_stop;
    ownership_changed.notify_all();
    return true;
  }

  void release_local_virtual_display_claim() {
    release_local_virtual_display_claim_impl();
  }

  bool wait_for_local_virtual_display_retirement(std::chrono::milliseconds timeout) {
    return wait_for_local_virtual_display_retirement_impl(timeout, true);
  }

  bool remote_virtual_display_starting(
    remote_virtual_display_lease_t lease,
    std::chrono::milliseconds connect_timeout,
    bool setup_in_progress
  ) {
    if (lease == 0) {
      BOOST_LOG(error) << "Refusing a remote virtual-display reservation without a session lease."sv;
      return false;
    }
    std::unique_lock lock(ownership_mutex);
    const auto now = std::chrono::steady_clock::now();
    const auto release_deadline = now + ownership_release_timeout;
    const auto pending_duration = detail::remote_pending_duration(connect_timeout);
    // Cover whichever is longer: the bounded local teardown or one complete connection window.
    // Do not add the two durations; a quick handoff should not reserve the display for an extra
    // teardown timeout after a failed remote launch.
    const auto pending_until = std::max(release_deadline, now + pending_duration);
    if (remote_session_pending && remote_session_pending->lease == lease) {
      remote_session_pending->until = std::max(remote_session_pending->until, pending_until);
    } else {
      // A newer accepted HTTP launch supersedes an older handshake reservation. The session lease
      // prevents teardown from the older process lifecycle from clearing this replacement.
      remote_session_pending = pending_remote_session_t {
        lease,
        pending_until,
        true,
        setup_in_progress,
      };
    }
    remote_session_pending->handoff_in_progress = true;
    remote_session_pending->setup_in_progress =
      remote_session_pending->setup_in_progress || setup_in_progress;
    if (local_session_construction_stop) {
      local_session_construction_stop->request_stop();
    }
    ownership_changed.notify_all();

    if (ownership_changed.wait_until(lock, release_deadline, []() {
          return !local_session_present;
        })) {
      lock.unlock();
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        release_deadline - std::chrono::steady_clock::now()
      );
      const bool retirement_complete = wait_for_local_virtual_display_retirement(
        std::max(remaining, 0ms)
      );
      const bool topology_recovery_complete = retirement_complete &&
                                               recover_connected_saved_topologies_for_remote_handoff();
      lock.lock();
      if (!remote_session_pending || remote_session_pending->lease != lease) {
        BOOST_LOG(info) << "Remote virtual-display reservation was superseded during local handoff."sv;
        return false;
      }
      if (!topology_recovery_complete) {
        clear_remote_lease_locked(lease);
        ownership_changed.notify_all();
        BOOST_LOG(error) << (retirement_complete ?
                               "The retired local AR display topology could not be recovered; refusing remote virtual-display creation."sv :
                               "The retired local AR virtual display remained in Windows topology; refusing overlapping remote display creation."sv);
        return false;
      }
      // Teardown time must not consume the connection reservation. Give the remote path its full
      // configured connection window after local ownership has actually been released.
      remote_session_pending->until = std::chrono::steady_clock::now() + pending_duration;
      remote_session_pending->handoff_in_progress = false;
      return true;
    }

    clear_remote_lease_locked(lease);
    ownership_changed.notify_all();
    BOOST_LOG(error) << "Local AR did not release virtual-display ownership within "sv
                     << ownership_release_timeout.count() << " seconds."sv;
    return false;
  }

  void remote_virtual_display_awaiting_client(
    remote_virtual_display_lease_t lease,
    std::chrono::milliseconds connect_timeout
  ) {
    std::lock_guard lock(ownership_mutex);
    // Creating the virtual display, probing encoders, and running application preparation all
    // happen while proc_t owns its process lock and can legitimately outlive the initial lease.
    // Start a fresh connection window only after that work succeeds, before the process lock is
    // released and local AR is allowed to inspect ownership again.
    if (!pending_remote_lease_matches_locked(lease)) {
      BOOST_LOG(warning) << "Ignoring a stale remote virtual-display reservation renewal."sv;
      return;
    }
    remote_session_pending->until = std::max(
      remote_session_pending->until,
      std::chrono::steady_clock::now() + detail::remote_pending_duration(connect_timeout)
    );
    // Process setup deliberately pins this lease past its ordinary connect deadline. Once all
    // display creation, probing, and application work has completed, the normal bounded RTSP
    // connection window resumes and an abandoned launch can expire again.
    remote_session_pending->setup_in_progress = false;
    ownership_changed.notify_all();
  }

  bool remote_virtual_display_active(remote_virtual_display_lease_t lease) {
    std::lock_guard lock(ownership_mutex);
    if (remote_session_active_lease == lease) {
      return true;
    }
    if (!pending_remote_lease_matches_locked(lease)) {
      return false;
    }
    remote_session_active_lease = lease;
    remote_session_pending.reset();
    ownership_changed.notify_all();
    return true;
  }

  void remote_virtual_display_ended(remote_virtual_display_lease_t lease) {
    if (lease == 0) {
      return;
    }
    std::lock_guard lock(ownership_mutex);
    clear_remote_lease_locked(lease);
    ownership_changed.notify_all();
  }

  bool remote_virtual_display_blocks_local() {
    std::lock_guard lock(ownership_mutex);
    return remote_blocks_local_locked(std::chrono::steady_clock::now());
  }

  std::unique_ptr<platf::deinit_t> init() {
    load_devices();
    normalize_topology_recovery_state();
    const auto pending_recoveries = recover_connected_saved_topologies();
    if (pending_recoveries != 0) {
      BOOST_LOG(info) << pending_recoveries
                      << " saved AR topology recovery record(s) remain pending until their display reconnects."sv;
    }
    return std::make_unique<controller_t>();
  }
}  // namespace ar_glasses
