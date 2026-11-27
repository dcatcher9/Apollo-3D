#include "ar_glasses_topology_journal.h"

#include "ar_glasses.h"
#include "ar_glasses_ccd.h"
#include "misc.h"
#include "src/config.h"
#include "src/logging.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace ar_glasses {
  using namespace std::literals;
  using namespace internal;

  namespace {
    std::mutex topology_recovery_mutex;

    struct topology_recovery_t {
      std::wstring device_path;
      RECT original_rect {};
      std::vector<RECT> owned_rects;
      std::optional<RECT> pending_rect;
    };

    std::filesystem::path topology_recovery_path() {
      auto path = std::filesystem::path(platf::from_utf8(config::sunshine.config_file));
      path += L".apollo-ar-topology.json";
      return path;
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

    nlohmann::json serialize_topology_recovery(const topology_recovery_t &recovery) {
      nlohmann::json owned_rects = nlohmann::json::array();
      for (const auto &rect : recovery.owned_rects) {
        owned_rects.emplace_back(serialize_rect(rect));
      }
      return {
        {"device_path", platf::to_utf8(recovery.device_path)},
        {"original_rect", serialize_rect(recovery.original_rect)},
        {"owned_rects", std::move(owned_rects)},
        {"pending_rect", recovery.pending_rect ? serialize_rect(*recovery.pending_rect) : nlohmann::json(nullptr)},
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
      if (recovery.device_path.empty() || (recovery.owned_rects.empty() && !recovery.pending_rect)) {
        throw std::runtime_error("invalid monitor identity or empty recovery transaction");
      }
      return recovery;
    }

    std::vector<topology_recovery_t> parse_topology_recovery_document(
      const nlohmann::json &value
    ) {
      const int version {value.at("version").get<int>()};
      if (version != 4) {
        throw std::runtime_error("unsupported recovery-record version");
      }

      const auto &entries = value.at("recoveries");
      if (!entries.is_array()) {
        throw std::runtime_error("recovery list is not an array");
      }

      std::vector<topology_recovery_t> recoveries;
      recoveries.reserve(entries.size());
      for (const auto &entry : entries) {
        if (!entry.is_object()) {
          throw std::runtime_error("recovery entry is not an object");
        }
        auto recovery = parse_topology_recovery(entry);
        if (std::ranges::any_of(recoveries, [&](const auto &existing) {
              return equal_device_paths(existing.device_path, recovery.device_path);
            })) {
          throw std::runtime_error("duplicate monitor identity");
        }
        recoveries.emplace_back(std::move(recovery));
      }
      return recoveries;
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

    std::optional<std::vector<topology_recovery_t>> load_topology_recoveries_locked() {
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
        return parse_topology_recovery_document(value);
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
        return equal_device_paths(recovery.device_path, device_path);
      });
      return recoveries->size() == original_size || write_topology_recoveries_locked(*recoveries);
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
        return equal_device_paths(recovery.device_path, device_path);
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
        return equal_device_paths(candidate.device_path, device_path);
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
  }  // namespace

  namespace internal {
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
        return equal_device_paths(recovery.device_path, device_path);
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

    bool cancel_topology_recovery_move(std::wstring_view device_path) {
      std::lock_guard lock(topology_recovery_mutex);
      auto recoveries {load_topology_recoveries_locked()};
      if (!recoveries) {
        return false;
      }
      const auto existing {std::ranges::find_if(*recoveries, [&](const auto &recovery) {
        return equal_device_paths(recovery.device_path, device_path);
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
        return equal_device_paths(recovery.device_path, device_path);
      });
      if (existing == recoveries->end()) {
        // No Apollo move is outstanding. The current mode's exact rectangle is therefore the
        // user-owned baseline that a later isolation transaction must restore.
        return current_rect;
      }

      if (rebase_topology_recovery_for_mode_change(*existing, previous_original_rect, current_rect)) {
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
        if (!same_rect(current_physical, initial_physical_rect) || !same_rect(current_virtual, initial_virtual_rect)) {
          BOOST_LOG(info) << "AR topology changed while its recovery transaction was being persisted; deferring isolation."sv;
          break;
        }
        if (!same_rect(candidate->layout.physical_rect, requested_layout.physical_rect) || !same_rect(candidate->layout.virtual_rect, requested_layout.virtual_rect)) {
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
      const auto place_rows = [&](isolation_plan_t &placed) {
        auto &virtual_mode = placed.snapshot.modes[placed.snapshot.sources[placed.virtual_index].mode_index].sourceMode;
        auto &physical_mode = placed.snapshot.modes[placed.snapshot.sources[placed.physical_index].mode_index].sourceMode;
        virtual_mode.position.x = requested_layout.virtual_rect.left;
        virtual_mode.position.y = requested_layout.virtual_rect.top;
        physical_mode.position.x = requested_layout.physical_rect.left;
        physical_mode.position.y = requested_layout.physical_rect.top;
      };
      place_rows(*plan);

      // Apply only a semantically identical complete snapshot, re-queried after validation, so a
      // user change to any active output is never overwritten by the supplied full CCD image.
      const auto isolation = platf::display_config::validate_and_apply(plan->snapshot, 0, 0, [&](platf::display_config::topology_t &supplied) {
        auto fresh_plan = build_isolation_plan({}, virtual_identity, target_device_path);
        if (!fresh_plan || !same_topology_snapshot(validated_baseline, fresh_plan->snapshot) || !same_rect(fresh_plan->layout.virtual_rect, requested_layout.virtual_rect) || !same_rect(fresh_plan->layout.physical_rect, requested_layout.physical_rect)) {
          return false;
        }
        place_rows(*fresh_plan);
        supplied = std::move(fresh_plan->snapshot);
        return true;
      });
      if (isolation.state == platf::display_config::supplied_apply_e::stale) {
        BOOST_LOG(info) << "Desktop topology changed after AR isolation validation; deferring the move."sv;
        if (!cancel_topology_recovery_move(target_device_path)) {
          BOOST_LOG(error) << "Could not cancel the stale AR topology transaction."sv;
        }
        return std::nullopt;
      }
      if (isolation.state != platf::display_config::supplied_apply_e::applied) {
        if (isolation.state == platf::display_config::supplied_apply_e::invalid) {
          BOOST_LOG(warning) << "Could not validate AR presentation-output isolation: "sv << isolation.status;
        } else {
          BOOST_LOG(warning) << "Could not isolate the AR presentation output: "sv << isolation.status;
        }
        if (!cancel_topology_recovery_move(target_device_path)) {
          BOOST_LOG(error) << "Could not cancel the failed AR topology transaction; retaining conservative recovery state."sv;
        }
        return std::nullopt;
      }

      // Never infer success from SetDisplayConfig alone. Resolve both the exact SudoVDA identity
      // and physical PnP target, then require the complete requested row and shared edge.
      std::optional<isolation_plan_t> verified_plan;
      for (int attempt = 0; attempt < 10; ++attempt) {
        if (auto verified = query_verified_isolation(virtual_identity, target_device_path, requested_layout)) {
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
          return equal_device_paths(candidate.device_path, device_path);
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

    pre_removal_topology_cleanup_t::pre_removal_topology_cleanup_t(
      std::optional<pre_removal_topology_decision_t> decision,
      std::wstring device_path
    ):
        decision_(std::move(decision)),
        decision_resolved_(decision_.has_value()),
        device_path_(std::move(device_path)) {
    }

    bool pre_removal_topology_cleanup_t::prepare_removal() {
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

    bool pre_removal_topology_cleanup_t::finish() {
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

    bool recover_saved_topology(std::wstring_view device_path) {
      std::optional<topology_recovery_t> recovery;
      {
        std::lock_guard lock(topology_recovery_mutex);
        const auto recoveries = load_topology_recoveries_locked();
        if (!recoveries) {
          return false;
        }
        const auto match = std::ranges::find_if(*recoveries, [&](const auto &candidate) {
          return equal_device_paths(candidate.device_path, device_path);
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
                              "display layout no longer matches a Sunshine 3D-owned rectangle."sv;
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
  }  // namespace internal

#ifdef SUNSHINE_TESTS
  detail::topology_recovery_parse_result_t detail::parse_topology_recovery_json_for_test(
    std::string_view contents
  ) {
    try {
      auto recoveries = parse_topology_recovery_document(nlohmann::json::parse(contents));
      return {
        .valid = true,
        .record_count = recoveries.size(),
        .normalized_json = serialize_topology_recovery_document(recoveries).dump(),
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
      const auto recoveries = parse_topology_recovery_document(nlohmann::json::parse(contents));
      return recoveries.size() == 1 &&
             topology_recovery_should_restore(recoveries.front(), current_rect);
    } catch (const std::exception &) {
      return false;
    }
  }

  std::optional<std::string> detail::rebase_topology_recovery_json_for_test(
    std::string_view contents,
    const RECT &previous_original_rect,
    const RECT &current_rect
  ) {
    try {
      auto recoveries = parse_topology_recovery_document(nlohmann::json::parse(contents));
      if (recoveries.size() != 1 || !rebase_topology_recovery_for_mode_change(recoveries.front(), previous_original_rect, current_rect)) {
        return std::nullopt;
      }
      return serialize_topology_recovery_document(recoveries).dump();
    } catch (const std::exception &) {
      return std::nullopt;
    }
  }
#endif
}  // namespace ar_glasses
