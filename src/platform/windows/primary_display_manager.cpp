/**
 * @file src/platform/windows/primary_display_manager.cpp
 * @brief The primary-display transaction policy and the session state machines around it.
 */
#include "primary_display_manager.h"

#include "misc.h"
#include "src/config.h"
#include "src/logging.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string_view>
#include <utility>

namespace platf::primary_display::detail {
  namespace {
    using display_config::equal_device_paths;
    using display_config::same_device_path;
    using display_config::same_luid;

    // A verified user arrangement becomes the existing recovery baseline, not another journal
    // mode. Keep the exact pending apply separately so color failures and new hotplugs can retry
    // without rediscovering the superseded launch positions. Missing outputs start fresh later.
    bool rebase_recovery(detail::journal_t &journal, const detail::snapshot_t &current, const detail::snapshot_t &desired) {
      auto layout = detail::inspect(desired);
      if (!layout) {
        return false;
      }
      auto physical = desired;
      if (const auto index = find_path(physical, journal.promoted_primary)) {
        physical.paths.erase(physical.paths.begin() + static_cast<std::ptrdiff_t>(*index));
        physical.device_paths.erase(physical.device_paths.begin() + static_cast<std::ptrdiff_t>(*index));
        physical.colors.erase(physical.colors.begin() + static_cast<std::ptrdiff_t>(*index));
      }
      if (!detail::inspect(physical)) {
        return false;
      }
      if (!find_position(*layout, journal.promoted_primary)) {
        // A pause detaches the VD, but the existing journal still records its identity and a
        // secondary position for a retry that keeps it active. This does not activate anything.
        std::int64_t right = std::numeric_limits<LONG>::min();
        LONG top = 0;
        for (size_t i = 0; i < physical.paths.size(); ++i) {
          const auto spec = display_spec(physical, i);
          const auto edge = static_cast<std::int64_t>(spec.source.position.x) + spec.source.width;
          if (edge > right) {
            right = edge;
            top = spec.source.position.y;
          }
        }
        if (right > std::numeric_limits<LONG>::max()) {
          return false;
        }
        layout->push_back({journal.promoted_primary, static_cast<LONG>(right), top});
      }
      const auto *virtual_position = find_position(*layout, journal.promoted_primary);
      const auto promoted = translated(*layout, virtual_position->x, virtual_position->y);
      const auto primary = std::ranges::find_if(*layout, [](const auto &position) {
        return position.x == 0 && position.y == 0;
      });
      if (!promoted || primary == layout->end()) {
        return false;
      }
      journal.original_primary = primary->device_path;
      journal.original = std::move(*layout);
      journal.promoted = *promoted;
      journal.original_topology = std::move(physical);
      journal.before_exclusive = current;
      journal.exclusive_topology.reset();
      std::erase_if(journal.exclusive_preserved, [&](const auto &identity) {
        return !find_path(*journal.original_topology, identity);
      });
      if (!journal.local_sink.empty() && !find_path(*journal.original_topology, journal.local_sink)) {
        journal.local_sink.clear();
      }
      return true;
    }

    bool preserved_exclusive_identity(const detail::journal_t &journal, std::wstring_view identity) {
      return std::ranges::any_of(journal.exclusive_preserved, [&](const auto &preserved) {
        return same_device_path(preserved, identity);
      });
    }

    bool exclusive_continuation_with_recorded_outputs(
      const detail::snapshot_t &current,
      const detail::journal_t &journal,
      std::wstring_view virtual_identity,
      bool reconcile_active_source
    ) {
      if (!journal.exclusive_topology || !journal.original_topology || !find_path(current, virtual_identity)) {
        return false;
      }
      auto owned = current;
      for (size_t i = owned.paths.size(); i-- > 0;) {
        const auto &identity = owned.device_paths[i];
        if (find_path(*journal.exclusive_topology, identity)) {
          continue;
        }
        const bool original = find_path(*journal.original_topology, identity).has_value();
        const bool recorded = journal.pending_restore && find_path(*journal.pending_restore, identity).has_value();
        if (!original && !recorded) {
          return false;
        }
        owned.paths.erase(owned.paths.begin() + static_cast<std::ptrdiff_t>(i));
        owned.device_paths.erase(owned.device_paths.begin() + static_cast<std::ptrdiff_t>(i));
        if (i < owned.colors.size()) {
          owned.colors.erase(owned.colors.begin() + static_cast<std::ptrdiff_t>(i));
        }
      }
      if (!find_path(owned, virtual_identity)) {
        return false;
      }
      if (!journal.local_sink.empty()) {
        // A hardware mode switch can restore Windows' remembered extended layout, including
        // the ordinary monitor as primary and new positions for both local outputs. The full
        // snapshot must be valid; after removing only durably recorded ordinary outputs above,
        // the remaining identities must still be our exact source and selected sink. Promotion
        // reestablishes the source origin instead of requiring it before recovery can begin.
        if (detail::inspect(current) && std::ranges::all_of(owned.device_paths, [&](const auto &identity) {
              return same_device_path(identity, virtual_identity) || same_device_path(identity, journal.local_sink);
            })) {
          return true;
        }
      } else if (reconcile_active_source && journal.exclusive_preserved.empty() && owned.paths.size() == 1 && detail::inspect(current)) {
        // A game mode change or exit can restore a recorded ordinary monitor as primary
        // while translating and retiming our virtual source. With no preserved outputs,
        // the exact remaining source is still owned even when its previous mode no longer
        // matches. Promotion retains its current mode and restores the exclusive origin.
        // Unknown identities were rejected above; preserved physical modes use the stricter
        // continuation checks below. Restore/pause must instead preserve user arrangements.
        return true;
      }
      return owned_subset(owned, *journal.exclusive_topology) ||
             owned_exclusive_layout(owned, virtual_identity, *journal.exclusive_topology) ||
             owned_exclusive_state(
               owned,
               virtual_identity,
               journal.exclusive_preserved,
               *journal.original_topology
             );
    }

    // Windows can accept an Advanced Color request before CCD reports the new state. Give it the
    // time the virtual-display detach gives its surviving monitors: a look every 50 ms for 500 ms.
    constexpr display_config::color_settle_policy_t color_settle {std::chrono::milliseconds(500), std::chrono::milliseconds(50)};

    struct color_request_t {
      std::wstring_view identity;
      std::optional<display_config::advanced_color_state_t> color;
    };

    std::vector<color_request_t> color_requests(const detail::snapshot_t &desired) {
      std::vector<color_request_t> requests;
      for (size_t i = 0; i < desired.paths.size(); ++i) {
        requests.push_back({desired.device_paths[i], i < desired.colors.size() ? desired.colors[i] : std::nullopt});
      }
      return requests;
    }

    std::vector<color_request_t> color_requests(const std::vector<display_spec_t> &specs) {
      std::vector<color_request_t> requests;
      for (const auto &spec : specs) {
        if (spec.color) {
          requests.push_back({spec.identity, spec.color});
        }
      }
      return requests;
    }

    /**
     * Request every color that `observed` does not already show, then wait until a query in
     * `topology` shows all of them. A request without a color, or for an output whose color
     * `observed` lacks, is rejected after the other requests are made. On return `observed`
     * holds the last query.
     */
    display_config::color_change_result_e set_colors_and_settle(const detail::io_t &io, const detail::snapshot_t &topology, const std::vector<color_request_t> &requests, std::optional<detail::snapshot_t> &observed) {
      const auto color_index = [&](std::wstring_view identity) -> std::optional<size_t> {
        const auto index = find_path(*observed, identity);
        return index && *index < observed->colors.size() && observed->colors[*index] ? index : std::nullopt;
      };
      return display_config::change_color_and_settle(
        [&]() {
          bool accepted = true;
          for (const auto &request : requests) {
            const auto index = color_index(request.identity);
            if (!request.color || !index) {
              accepted = false;
            } else if (!color_matches(*observed->colors[*index], *request.color)) {
              accepted = io.set_color && io.set_color(observed->paths[*index], *request.color) && accepted;
            }
          }
          return accepted;
        },
        [&]() {
          observed = io.query();
          if (!observed || !same_topology(*observed, topology)) {
            return display_config::color_observation_e::failed;
          }
          for (const auto &request : requests) {
            const auto index = color_index(request.identity);
            if (!index) {
              return display_config::color_observation_e::failed;
            }
            if (!color_matches(*observed->colors[*index], *request.color)) {
              return display_config::color_observation_e::pending;
            }
          }
          return display_config::color_observation_e::matched;
        },
        color_settle,
        [&](std::chrono::milliseconds delay) {
          if (io.sleep) {
            io.sleep(delay);
          }
          return true;
        }
      );
    }

    void log_restore_snapshot(std::string_view phase, const detail::snapshot_t &snapshot) {
      if (!config::sunshine.diagnostics_enabled) {
        return;
      }
      std::ostringstream text;
      for (size_t i = 0; i < snapshot.paths.size(); ++i) {
        const auto &path = snapshot.paths[i];
        text << " [device=" << (i < snapshot.device_paths.size() ? platf::to_utf8(snapshot.device_paths[i]) : "unknown")
             << " available=" << path.targetInfo.targetAvailable
             << " status=" << path.targetInfo.statusFlags;
        const auto index = source_index(path);
        if (index < snapshot.modes.size() && snapshot.modes[index].infoType == DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE) {
          const auto &source = snapshot.modes[index].sourceMode;
          text << " mode=" << source.width << 'x' << source.height
               << " position=" << source.position.x << ',' << source.position.y;
        }
        text << " refresh=" << path.targetInfo.refreshRate.Numerator << '/' << path.targetInfo.refreshRate.Denominator;
        if (i < snapshot.colors.size() && snapshot.colors[i]) {
          const auto &color = *snapshot.colors[i];
          text << " color_mode=" << static_cast<int>(color.active_mode)
               << " color_enabled=" << color.advanced_color_active
               << " hdr_enabled=" << color.hdr_user_enabled << " bits=" << color.bits_per_color_channel;
        }
        text << ']';
      }
      BOOST_LOG(info) << "Display restore " << phase << ":" << text.str();
    }
  }  // namespace

  bool valid_journal(const journal_t &journal) {
    if (journal.monitor_replacement && (!journal.exclusive || journal.local_sink.empty())) {
      return false;
    }
    if (!valid_layout(journal.original)) {
      return false;
    }
    const auto *original = find_position(journal.original, journal.original_primary);
    if (!original || original->x != 0 || original->y != 0) {
      return false;
    }
    if (journal.exclusive) {
      const bool headless = journal.original_topology && journal.original_topology->paths.empty() &&
                            journal.original_topology->device_paths.empty() && journal.original_topology->modes.empty() &&
                            journal.original_topology->colors.empty() && journal.original.size() == 1 &&
                            same_device_path(journal.original_primary, journal.promoted_primary) && !journal.prepared && journal.exclusive_started;
      if (!journal.original_topology || (!headless && !detail::inspect(*journal.original_topology)) || journal.original_topology->colors.size() != journal.original_topology->paths.size() || !std::ranges::all_of(journal.original_topology->colors, [](const auto &color) {
            return color.has_value();
          })) {
        return false;
      }
      const auto saved = headless ? std::optional<detail::layout_t> {detail::layout_t {}} : detail::inspect(*journal.original_topology);
      for (const auto &entry : *saved) {
        const auto *expected = find_position(journal.original, entry.device_path);
        if (!expected || entry.x != expected->x || entry.y != expected->y) {
          return false;
        }
      }
      for (size_t i = 0; i < journal.exclusive_preserved.size(); ++i) {
        const auto &identity = journal.exclusive_preserved[i];
        bool duplicate = false;
        for (size_t j = 0; j < i; ++j) {
          duplicate = duplicate || same_device_path(identity, journal.exclusive_preserved[j]);
        }
        const bool recorded_hotplug = journal.pending_restore &&
                                      std::ranges::any_of(journal.pending_restore->device_paths, [&](const auto &pending) {
                                        return same_device_path(identity, pending);
                                      });
        if (identity.empty() || (!find_position(*saved, identity) && !recorded_hotplug) || duplicate) {
          return false;
        }
      }
      if (!journal.local_sink.empty() && (!find_position(*saved, journal.local_sink) || journal.exclusive_preserved.size() != 1 || !same_device_path(journal.exclusive_preserved.front(), journal.local_sink) || same_device_path(journal.local_sink, journal.promoted_primary))) {
        return false;
      }
      const bool empty_before = headless && journal.before_exclusive && journal.before_exclusive->paths.empty() &&
                                journal.before_exclusive->device_paths.empty() && journal.before_exclusive->modes.empty() &&
                                journal.before_exclusive->colors.empty();
      if (journal.exclusive_started && (!journal.before_exclusive || (!empty_before && !detail::inspect(*journal.before_exclusive)) || journal.prepared)) {
        return false;
      }
      if ((!journal.exclusive_started && journal.exclusive_topology) || (journal.exclusive_topology && (!detail::inspect(*journal.exclusive_topology) || journal.exclusive_topology->colors.size() != journal.exclusive_topology->paths.size()))) {
        return false;
      }
      if (journal.exclusive_topology) {
        const auto exclusive_layout = detail::inspect(*journal.exclusive_topology);
        const auto *exclusive_primary = find_position(*exclusive_layout, journal.promoted_primary);
        if (!exclusive_primary || (!journal.monitor_replacement && (exclusive_primary->x != 0 || exclusive_primary->y != 0)) || std::ranges::any_of(*exclusive_layout, [&](const auto &entry) {
              if (same_device_path(entry.device_path, journal.promoted_primary) || (journal.monitor_replacement && find_position(*saved, entry.device_path)) || std::ranges::any_of(journal.exclusive_preserved, [&](const auto &preserved) {
                    return same_device_path(entry.device_path, preserved);
                  })) {
                return false;
              }
              // A display absent from the launch baseline may join the owned transaction only
              // after restore has durably recorded its approved hotplug identity.
              return !journal.pending_restore || find_position(*saved, entry.device_path) ||
                     std::ranges::none_of(journal.pending_restore->device_paths, [&](const auto &pending) {
                       return same_device_path(entry.device_path, pending);
                     });
            })) {
          return false;
        }
      }
      if (journal.pending_restore && ((!journal.exclusive_started && !journal.prepared) || !detail::inspect(*journal.pending_restore))) {
        return false;
      }
    } else if (journal.exclusive_started || journal.original_topology || journal.before_exclusive || journal.pending_restore || !journal.exclusive_preserved.empty() || journal.exclusive_topology || !journal.local_sink.empty()) {
      return false;
    }
    if (journal.prepared) {
      return journal.promoted.empty() && journal.promoted_primary.empty();
    }
    if (!valid_layout(journal.promoted) || (!journal.exclusive && same_device_path(journal.original_primary, journal.promoted_primary))) {
      return false;
    }
    const auto *target = find_position(journal.original, journal.promoted_primary);
    if (!target) {
      return false;
    }
    const auto expected = translated(journal.original, target->x, target->y);
    return expected && same_layout(*expected, journal.promoted);
  }

  void exclusive_session_state_t::changed() {
    ++generation_;
  }

  void exclusive_session_state_t::prepared() {
    if (expected_) {
      return;
    }
    expected_ = true;
    changed();
  }

  bool exclusive_session_state_t::can_use_identity(std::wstring_view device_path) const {
    if (device_path.empty()) {
      return false;
    }
    return (!pending_identity_ || same_device_path(*pending_identity_, device_path)) &&
           (!active_identity_ || same_device_path(*active_identity_, device_path));
  }

  void exclusive_session_state_t::bound(std::wstring_view device_path) {
    if (!expected_ || active_identity_ || pending_identity_) {
      return;
    }
    pending_identity_ = device_path;
    changed();
  }

  bool exclusive_session_state_t::begin_promotion(std::wstring_view device_path) {
    if (!can_use_identity(device_path)) {
      return false;
    }
    bool state_changed = false;
    if (!expected_) {
      expected_ = true;
      state_changed = true;
    }
    if (!active_identity_ && !pending_identity_) {
      pending_identity_ = device_path;
      state_changed = true;
    }
    if (state_changed) {
      changed();
    }
    return true;
  }

  bool exclusive_session_state_t::promotion_succeeded(std::wstring_view device_path) {
    if (!can_use_identity(device_path)) {
      return false;
    }
    const bool already_active = active_identity_ && same_device_path(*active_identity_, device_path) && !pending_identity_;
    expected_ = true;
    pending_identity_.reset();
    active_identity_ = device_path;
    if (!already_active) {
      changed();
    }
    return true;
  }

  exclusive_restore_action_e exclusive_session_state_t::restore_action(std::wstring_view expected_device_path) const {
    if (expected_device_path.empty()) {
      return exclusive_restore_action_e::disarm_before;
    }
    if (active_identity_ || pending_identity_) {
      return can_use_identity(expected_device_path) ? exclusive_restore_action_e::disarm_before :
                                                      exclusive_restore_action_e::reject;
    }
    return expected_ ? exclusive_restore_action_e::disarm_after_success :
                       exclusive_restore_action_e::keep;
  }

  void exclusive_session_state_t::restore_finished(exclusive_restore_action_e action, bool restored) {
    if (restored && action == exclusive_restore_action_e::disarm_after_success) {
      disarm();
    }
  }

  void exclusive_session_state_t::disarm() {
    if (!expected_ && !pending_identity_ && !active_identity_) {
      return;
    }
    expected_ = false;
    pending_identity_.reset();
    active_identity_.reset();
    changed();
  }

  exclusive_reconcile_action_e exclusive_session_state_t::reconcile_action() const {
    if (active_identity_) {
      return exclusive_reconcile_action_e::reconcile;
    }
    return expected_ || pending_identity_ ? exclusive_reconcile_action_e::defer :
                                            exclusive_reconcile_action_e::recover;
  }

  bool exclusive_session_state_t::expected() const {
    return expected_;
  }

  const std::optional<std::wstring> &exclusive_session_state_t::pending_identity() const {
    return pending_identity_;
  }

  const std::optional<std::wstring> &exclusive_session_state_t::active_identity() const {
    return active_identity_;
  }

  std::uint64_t exclusive_session_state_t::generation() const {
    return generation_;
  }

  bool restore_observation_cache_t::unchanged_failure(
    const snapshot_t &current,
    const snapshot_t &desired,
    const snapshot_t &available
  ) const {
    if (!failure_ || !same_topology(current, failure_->observed) ||
        !same_colors(current, failure_->observed) || !same_topology(desired, failure_->desired) ||
        !same_colors(desired, failure_->desired) || available.paths.size() != failure_->available.paths.size() ||
        available.paths.size() != available.device_paths.size() ||
        failure_->available.paths.size() != failure_->available.device_paths.size()) {
      return false;
    }
    // QDC_ALL_PATHS may reorder alternative routes or update transient IN_USE flags after
    // our own apply. Compare the target/source route multiset, not mode indexes or status.
    const auto same_route = [](const snapshot_t &a, size_t ai, const snapshot_t &b, size_t bi) {
      const auto &ap = a.paths[ai];
      const auto &bp = b.paths[bi];
      return same_device_path(a.device_paths[ai], b.device_paths[bi]) &&
             same_luid(ap.sourceInfo.adapterId, bp.sourceInfo.adapterId) && ap.sourceInfo.id == bp.sourceInfo.id &&
             same_luid(ap.targetInfo.adapterId, bp.targetInfo.adapterId) && ap.targetInfo.id == bp.targetInfo.id &&
             ap.targetInfo.targetAvailable == bp.targetInfo.targetAvailable;
    };
    std::vector<bool> used(failure_->available.paths.size(), false);
    for (size_t i = 0; i < available.paths.size(); ++i) {
      bool matched = false;
      for (size_t j = 0; j < used.size(); ++j) {
        if (!used[j] && same_route(available, i, failure_->available, j)) {
          used[j] = true;
          matched = true;
          break;
        }
      }
      if (!matched) {
        return false;
      }
    }
    return true;
  }

  void restore_observation_cache_t::remember_failure(
    const snapshot_t &observed,
    const snapshot_t &desired,
    const snapshot_t &available
  ) {
    failure_ = failure_t {observed, desired, available};
  }

  void restore_observation_cache_t::clear() {
    failure_.reset();
  }

  manager_t::manager_t(io_t io, restore_observation_cache_t *restore_observations):
      io_(std::move(io)),
      restore_observations_(restore_observations) {}

  bool manager_t::prepare(bool exclusive, std::wstring_view local_sink, bool monitor_replacement) {
    if ((!local_sink.empty() && !exclusive) || (monitor_replacement && (!exclusive || local_sink.empty()))) {
      return false;
    }
    const auto loaded = io_.load();
    if (!loaded.success || loaded.journal) {
      return false;
    }
    const auto snapshot = io_.query();
    if (!snapshot) {
      return false;
    }
    if (snapshot->paths.empty() && snapshot->device_paths.empty()) {
      // A headless machine has no original primary display to restore.
      return local_sink.empty();
    }
    const auto layout = inspect(*snapshot);
    if (!layout) {
      return false;
    }
    const auto primary = std::ranges::find_if(*layout, [](const auto &entry) {
      return entry.x == 0 && entry.y == 0;
    });
    journal_t journal {primary->device_path, {}, *layout, {}, true};
    journal.monitor_replacement = monitor_replacement;
    if (exclusive) {
      journal.exclusive = true;
      journal.original_topology = *snapshot;
      if (io_.preserve_exclusive) {
        for (size_t i = 0; i < snapshot->paths.size(); ++i) {
          if ((local_sink.empty() || same_device_path(local_sink, snapshot->device_paths[i])) && io_.preserve_exclusive(snapshot->paths[i], snapshot->device_paths[i])) {
            journal.exclusive_preserved.push_back(snapshot->device_paths[i]);
          }
        }
        std::ranges::sort(journal.exclusive_preserved, [](const auto &left, const auto &right) {
          return CompareStringOrdinal(left.c_str(), -1, right.c_str(), -1, TRUE) == CSTR_LESS_THAN;
        });
      }
      journal.local_sink = local_sink;
      if (monitor_replacement) {
        journal.exclusive_preserved = {std::wstring(local_sink)};
      }
    }
    return valid_journal(journal) && io_.save(journal);
  }

  bool manager_t::is_local_exclusive(std::wstring_view source_device_path) {
    const auto loaded = io_.load();
    return loaded.success && loaded.journal && valid_journal(*loaded.journal) &&
           !loaded.journal->local_sink.empty() &&
           same_device_path(loaded.journal->promoted_primary, source_device_path);
  }

  bool manager_t::verify_local_monitor_replacement(std::wstring_view source_device_path, std::wstring_view sink_device_path) {
    const auto loaded = io_.load();
    if (!loaded.success || !loaded.journal || !valid_journal(*loaded.journal)) {
      return false;
    }
    const auto &journal = *loaded.journal;
    if (!journal.monitor_replacement || journal.prepared || !journal.exclusive_started || !journal.exclusive_topology || journal.pending_restore || !same_device_path(journal.promoted_primary, source_device_path) || !same_device_path(journal.local_sink, sink_device_path)) {
      return false;
    }
    const auto &committed = *journal.exclusive_topology;
    if (!find_path(committed, sink_device_path)) {
      return false;
    }

    const auto current = io_.query();
    // same_colors accepts an absent color in both snapshots. Reuse requires a fresh color
    // observation for every active path, just like the original replacement verification.
    return current && same_topology(*current, committed) && same_colors(*current, committed) &&
           std::ranges::all_of(current->colors, [](const auto &color) {
             return color.has_value();
           });
  }

  bool manager_t::bind_pending(std::wstring_view device_path) {
    const auto loaded = io_.load();
    if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
      return false;
    }
    if (!loaded.journal) {
      return true;
    }
    const auto &prepared = *loaded.journal;
    if (!prepared.prepared) {
      return same_device_path(prepared.promoted_primary, device_path);
    }
    const auto snapshot = io_.query();
    const auto current = snapshot ? inspect(*snapshot) : std::nullopt;
    if (!current) {
      return false;
    }
    const auto *target = find_position(*current, device_path);
    const auto *original_primary = find_position(*current, prepared.original_primary);
    const bool new_target = !find_position(prepared.original, device_path);
    if (prepared.exclusive && target && (!prepared.local_sink.empty() || !original_primary || current->size() != prepared.original.size() + (new_target ? 1 : 0))) {
      // Windows can restore a previously remembered virtual-only topology inside Add.
      // The exact Add identity is known here; preserve the pre-Add physical snapshot.
      for (const auto &entry : *current) {
        if (!same_device_path(entry.device_path, device_path) && !find_position(prepared.original, entry.device_path)) {
          return false;
        }
      }
      auto original = prepared.original;
      if (new_target) {
        int64_t right = std::numeric_limits<LONG>::min();
        LONG top = 0;
        for (const auto &path : prepared.original_topology->paths) {
          const auto &source = prepared.original_topology->modes[source_index(path)].sourceMode;
          const int64_t edge = static_cast<int64_t>(source.position.x) + source.width;
          if (edge > right) {
            right = edge;
            top = source.position.y;
          }
        }
        if (right > std::numeric_limits<LONG>::max()) {
          return false;
        }
        original.push_back({std::wstring(device_path), static_cast<LONG>(right), top});
      }
      const auto *baseline_target = find_position(original, device_path);
      const auto promoted = translated(original, baseline_target->x, baseline_target->y);
      if (!promoted) {
        return false;
      }
      auto bound = prepared;
      bound.prepared = false;
      bound.promoted_primary = std::wstring(device_path);
      bound.original = std::move(original);
      bound.promoted = *promoted;
      bound.exclusive_started = true;
      bound.before_exclusive = *snapshot;
      return valid_journal(bound) && io_.save(bound);
    }
    if (!target || !original_primary || current->size() != prepared.original.size() + (new_target ? 1 : 0)) {
      return false;
    }
    const auto original = translated(*current, original_primary->x, original_primary->y);
    if (!original || !std::ranges::all_of(prepared.original, [&](const auto &saved) {
          const auto *observed = find_position(*original, saved.device_path);
          return observed && observed->x == saved.x && observed->y == saved.y;
        })) {
      return false;
    }
    if (same_device_path(prepared.original_primary, device_path) && !prepared.exclusive) {
      return io_.clear();
    }
    const auto promoted = translated(*current, target->x, target->y);
    if (!promoted) {
      return false;
    }
    auto bound = prepared;
    bound.prepared = false;
    bound.promoted_primary = target->device_path;
    bound.original = *original;
    bound.promoted = *promoted;
    if (bound.exclusive && bound.pending_restore) {
      bound.exclusive_started = true;
      bound.before_exclusive = *snapshot;
    }
    return valid_journal(bound) && io_.save(bound);
  }

  bool manager_t::baseline_topology_active(std::wstring_view device_path) {
    const auto loaded = io_.load();
    if (!loaded.success) {
      return false;
    }
    const auto snapshot = io_.query();
    const auto current = snapshot ? inspect(*snapshot) : std::nullopt;
    const auto *target = current ? find_position(*current, device_path) : nullptr;
    if (!target) {
      return false;
    }
    if (!loaded.journal) {
      // Every host topology change is journaled before it is applied, so none is active. Windows
      // can still make an added display primary itself; accept that only for a sole output.
      return current->size() == 1 || target->x != 0 || target->y != 0;
    }
    const auto &journal = *loaded.journal;
    if (!valid_journal(journal) || journal.prepared || !same_device_path(journal.promoted_primary, device_path)) {
      return false;
    }
    // Every topology the host applies either moves the original primary off the origin or
    // detaches this display, so the original primary at the origin beside this active display
    // is not a host topology. Windows can still have switched a recorded output off by itself:
    // it applies its own remembered layout for the monitor set inside Add and after a
    // reactivation. That layout is the one Windows already holds, so it does not block the
    // save. An output the record does not know is not the user's recorded desktop.
    const auto *primary = find_position(*current, journal.original_primary);
    return primary && primary->x == 0 && primary->y == 0 &&
           std::ranges::all_of(*current, [&](const auto &entry) {
             return find_position(journal.original, entry.device_path) != nullptr;
           });
  }

  bool manager_t::adopt_virtual_color(std::wstring_view device_path) {
    auto loaded = io_.load();
    if (!loaded.success) {
      return false;
    }
    // Only a remote exclusive session records a color state for its virtual display. A local
    // presentation follows its physical sink, and anything else restores no recorded color.
    if (!loaded.journal || !valid_journal(*loaded.journal) || !loaded.journal->exclusive || !loaded.journal->exclusive_topology ||
        !loaded.journal->local_sink.empty() || !same_device_path(loaded.journal->promoted_primary, device_path)) {
      return true;
    }
    auto journal = *loaded.journal;
    const auto snapshot = io_.query();
    const auto current = snapshot ? find_path(*snapshot, device_path) : std::nullopt;
    const auto recorded = find_path(*journal.exclusive_topology, device_path);
    if (!current || !recorded || *current >= snapshot->colors.size() || !snapshot->colors[*current] ||
        *recorded >= journal.exclusive_topology->colors.size()) {
      return false;
    }
    auto &recorded_color = journal.exclusive_topology->colors[*recorded];
    if (recorded_color && color_matches(*snapshot->colors[*current], *recorded_color)) {
      return true;
    }
    recorded_color = snapshot->colors[*current];
    return valid_journal(journal) && io_.save(journal);
  }

  bool manager_t::apply_verified(const snapshot_t &before, const layout_t &desired) {
    const auto baseline = inspect(before);
    auto latest = io_.query();
    const auto current = latest ? inspect(*latest) : std::nullopt;
    if (!baseline || !current || !same_layout(*baseline, *current) || !same_modes(before, *latest)) {
      BOOST_LOG(warning) << "Display configuration changed during primary-display transaction; retaining recovery record.";
      return false;
    }
    for (size_t i = 0; i < latest->paths.size(); ++i) {
      const auto *position = find_position(desired, latest->device_paths[i]);
      if (!position) {
        return false;
      }
      latest->modes[source_index(latest->paths[i])].sourceMode.position = {position->x, position->y};
    }
    if (!inspect(*latest) || !io_.apply(*latest)) {
      return false;
    }
    const auto observed = io_.query();
    const auto layout = observed ? inspect(*observed) : std::nullopt;
    if (!layout || !same_layout(*layout, desired) || !same_modes(*latest, *observed)) {
      BOOST_LOG(warning) << "Could not verify primary-display change; retaining recovery record.";
      return false;
    }
    return true;
  }

  bool manager_t::apply_snapshot_verified(const snapshot_t &before, const snapshot_t &desired) {
    const auto latest = io_.query();
    if (!latest || !same_topology(*latest, before) || !same_colors(*latest, before)) {
      return false;
    }
    if (!same_topology(before, desired) && !io_.apply(desired)) {
      return false;
    }
    auto observed = io_.query();
    return observed && same_topology(*observed, desired) &&
           set_colors_and_settle(io_, desired, color_requests(desired), observed) == display_config::color_change_result_e::verified;
  }

  bool manager_t::promote_monitor(std::wstring_view device_path) {
    const auto reject = [](std::string_view stage) {
      BOOST_LOG(warning) << "Local monitor replacement failed at " << stage << '.';
      return false;
    };
    auto loaded = io_.load();
    if (!loaded.success || !loaded.journal || !valid_journal(*loaded.journal) || !loaded.journal->monitor_replacement) {
      return reject("recovery journal validation");
    }
    if (loaded.journal->prepared) {
      if (!bind_pending(device_path)) {
        return reject("exact virtual-source journal binding");
      }
      loaded = io_.load();
    }
    if (!loaded.success || !loaded.journal || !same_device_path(loaded.journal->promoted_primary, device_path)) {
      return reject("bound virtual-source identity verification");
    }
    auto journal = *loaded.journal;
    const auto current = io_.query();
    const auto available = io_.query_all ? io_.query_all() : std::nullopt;
    const auto source_index_now = current ? find_path(*current, device_path) : std::nullopt;
    const auto sink_index_now = current ? find_path(*current, journal.local_sink) : std::nullopt;
    const auto sink_saved = find_path(*journal.original_topology, journal.local_sink);
    if (!current || !inspect(*current) || !available || available->paths.size() != available->device_paths.size() || available->paths.size() > max_available_paths || !source_index_now || !sink_saved) {
      return reject("active source/available-route snapshot validation");
    }
    bool connected_sink_route = false;
    for (size_t i = 0; i < available->paths.size(); ++i) {
      connected_sink_route = connected_sink_route ||
                             (available->paths[i].targetInfo.targetAvailable && same_device_path(available->device_paths[i], journal.local_sink));
    }
    if (!sink_index_now && !connected_sink_route) {
      return reject("inactive selected sink has no exact connected target route");
    }
    auto source = display_spec(*current, *source_index_now);
    // AddVirtualDisplay can temporarily deactivate a connected physical target while restoring
    // Windows' remembered virtual mode. The pre-Add journal already owns this exact sink and its
    // native mode/color. QDC_ALL_PATHS must still prove it connected; build_topology below rejects
    // ambiguous routes and reconstructs its saved mode before any CCD mutation.
    auto sink = sink_index_now ? display_spec(*current, *sink_index_now) : display_spec(*journal.original_topology, *sink_saved);
    if (!sink_index_now) {
      BOOST_LOG(info) << "Local monitor replacement: selected sink is temporarily inactive; reconstructing its recorded native mode from an exact connected target route.";
    }
    if (source.source.width != sink.source.width || source.source.height != sink.source.height) {
      BOOST_LOG(warning) << "Local monitor replacement source/sink raster mismatch: source "
                        << source.source.width << 'x' << source.source.height << ", sink "
                        << sink.source.width << 'x' << sink.source.height << '.';
      return reject("native virtual-source raster verification");
    }
    // Only the source/sink placement proves that replacement reached Windows. A changed
    // neighbor mode, HDR toggle or hotplug cannot turn a saved-but-unapplied intent into proof.
    const bool continuing = journal.exclusive_topology && !journal.pending_restore &&
                            monitor_pair_placement_matches(*current, *journal.exclusive_topology, device_path, journal.local_sink);
    if (!continuing) {
      source.source.position = display_spec(*journal.original_topology, *sink_saved).source.position;
    }

    std::vector<display_spec_t> wanted {source};
    if (continuing) {
      // Unselected physical outputs remain owned by the user. Their latest layout, modes and
      // colors also become the recovery baseline; the parked sink never supplies that layout.
      for (size_t i = 0; i < current->paths.size(); ++i) {
        if (!same_device_path(current->device_paths[i], device_path) && !same_device_path(current->device_paths[i], journal.local_sink)) {
          wanted.push_back(display_spec(*current, i));
        }
      }
    } else {
      for (size_t i = 0; i < journal.original_topology->paths.size(); ++i) {
        if (!same_device_path(journal.original_topology->device_paths[i], device_path) && !same_device_path(journal.original_topology->device_paths[i], journal.local_sink)) {
          auto ordinary = display_spec(*journal.original_topology, i);
          if (journal.exclusive_topology) {
            const auto live = find_path(*current, ordinary.identity);
            if (live) {
              const auto position = ordinary.source.position;
              ordinary = display_spec(*current, *live);
              ordinary.source.position = position;
            }
          }
          wanted.push_back(std::move(ordinary));
        }
      }
      // Do not erase a physical hotplug that arrived while AddVirtualDisplay was publishing.
      for (size_t i = 0; i < current->paths.size(); ++i) {
        if (!same_device_path(current->device_paths[i], device_path) && !same_device_path(current->device_paths[i], journal.local_sink) && !find_path(*journal.original_topology, current->device_paths[i])) {
          wanted.push_back(display_spec(*current, i));
        }
      }
    }
    if (!normalize_monitor_desktop(wanted, same_device_path(journal.original_primary, journal.local_sink) ? device_path : std::wstring_view(journal.original_primary))) {
      return reject("interactive desktop origin normalization");
    }
    source = wanted.front();
    // This logical physical snapshot is the reverse operation, saved before parking the sink.
    auto logical = wanted;
    sink.source.position = source.source.position;
    logical.front() = sink;
    std::string_view reconstruction_failure;
    const auto physical_baseline = build_topology(*available, logical, &reconstruction_failure);
    if (!physical_baseline) {
      BOOST_LOG(warning) << "Local monitor physical-baseline reconstruction: " << reconstruction_failure << '.';
      return reject("physical recovery baseline reconstruction");
    }
    if (!park_sink_at_desktop_corner(sink, wanted)) {
      return reject("physical sink parking");
    }
    wanted.push_back(sink);
    const auto desired = build_topology(*available, wanted, &reconstruction_failure);
    if (!desired) {
      BOOST_LOG(warning) << "Local monitor replacement topology reconstruction: " << reconstruction_failure << '.';
      return reject("replacement topology reconstruction");
    }
    journal.original_topology = *physical_baseline;
    const auto baseline_layout = inspect(*physical_baseline);
    const auto primary = std::ranges::find_if(*baseline_layout, [](const auto &entry) {
      return entry.x == 0 && entry.y == 0;
    });
    if (primary == baseline_layout->end()) {
      return reject("preserved physical primary verification");
    }
    journal.original_primary = primary->device_path;
    journal.original = *baseline_layout;
    auto parked_source = source;
    if (!park_beside_desktop(parked_source, logical)) {
      return reject("retirement-source parking");
    }
    journal.original.push_back({std::wstring(device_path), parked_source.source.position.x, parked_source.source.position.y});
    const auto promoted = translated(journal.original, parked_source.source.position.x, parked_source.source.position.y);
    if (!promoted) {
      return reject("recovery-layout translation");
    }
    journal.promoted = *promoted;
    journal.exclusive_started = true;
    journal.before_exclusive = *current;
    journal.exclusive_topology = *desired;
    journal.pending_restore.reset();
    if (!valid_journal(journal)) {
      return reject("replacement recovery journal validation");
    }
    if (!io_.save(journal)) {
      return reject("durable replacement recovery save");
    }
    if (!apply_snapshot_verified(*current, *desired)) {
      return reject("CCD apply/readback/color verification");
    }
    return true;
  }

  bool manager_t::promote_exclusive(std::wstring_view device_path, bool reconcile_active_source) {
    auto loaded = io_.load();
    if (!loaded.success || (loaded.journal && (!valid_journal(*loaded.journal) || !loaded.journal->exclusive))) {
      return false;
    }
    if (loaded.journal && loaded.journal->monitor_replacement) {
      return promote_monitor(device_path);
    }
    auto snapshot = io_.query();
    auto layout = snapshot ? inspect(*snapshot) : std::nullopt;
    if (!layout || !find_position(*layout, device_path)) {
      return false;
    }
    if (!loaded.journal) {
      if (snapshot->paths.size() == 1) {
        return true;
      }
      // Warm reconnect has an active retained virtual display but no outstanding journal.
      if (!prepare(true) || !bind_pending(device_path)) {
        return false;
      }
      loaded = io_.load();
    } else if (loaded.journal->prepared) {
      if (!bind_pending(device_path)) {
        return false;
      }
      loaded = io_.load();
    }
    if (!loaded.success || !loaded.journal || !valid_journal(*loaded.journal)) {
      return false;
    }
    auto journal = *loaded.journal;
    if (!same_device_path(journal.promoted_primary, device_path)) {
      return false;
    }

    const auto available = io_.query_all ? io_.query_all() : std::nullopt;
    if (!available || available->paths.size() > max_available_paths || available->paths.size() != available->device_paths.size()) {
      return false;
    }
    const auto virtual_index = find_path(*snapshot, device_path);
    if (!virtual_index) {
      return false;
    }
    auto virtual_spec = display_spec(*snapshot, *virtual_index);
    if (journal.exclusive_topology && journal.local_sink.empty()) {
      const auto prior_virtual = find_path(*journal.exclusive_topology, device_path);
      if (prior_virtual && *prior_virtual < journal.exclusive_topology->colors.size()) {
        virtual_spec.color = journal.exclusive_topology->colors[*prior_virtual];
      }
    }
    virtual_spec.source.position = {0, 0};
    std::vector<display_spec_t> wanted;
    wanted.push_back(std::move(virtual_spec));

    auto target_available = [&](std::wstring_view identity) {
      for (size_t i = 0; i < available->paths.size(); ++i) {
        if (available->paths[i].targetInfo.targetAvailable && same_device_path(available->device_paths[i], identity)) {
          return true;
        }
      }
      return false;
    };
    std::vector<display_spec_t> preserved;
    for (const auto &identity : journal.exclusive_preserved) {
      const auto original_index = find_path(*journal.original_topology, identity);
      const auto pending_index = journal.pending_restore ? find_path(*journal.pending_restore, identity) : std::nullopt;
      if (!original_index && !pending_index) {
        return false;
      }
      auto spec = original_index ? display_spec(*journal.original_topology, *original_index) :
                                   display_spec(*journal.pending_restore, *pending_index);
      const auto current_index = find_path(*snapshot, identity);
      if (!journal.local_sink.empty() && same_device_path(identity, journal.local_sink) && current_index) {
        // The physical button controls the local presentation mode and color.
        spec = display_spec(*snapshot, *current_index);
      }
      const auto prior_index = journal.exclusive_topology ? find_path(*journal.exclusive_topology, identity) : std::nullopt;
      if (journal.local_sink.empty() && prior_index && *prior_index < journal.exclusive_topology->colors.size()) {
        spec.color = journal.exclusive_topology->colors[*prior_index];
      }
      preserved.push_back(std::move(spec));
    }
    std::ranges::sort(preserved, [](const auto &left, const auto &right) {
      return CompareStringOrdinal(left.identity.c_str(), -1, right.identity.c_str(), -1, TRUE) == CSTR_LESS_THAN;
    });
    std::erase_if(preserved, [&](const auto &spec) {
      return !target_available(spec.identity);
    });
    std::int64_t preserved_width = 0;
    for (const auto &spec : preserved) {
      preserved_width += spec.source.width;
      if (preserved_width > std::numeric_limits<LONG>::max()) {
        return false;
      }
    }
    LONG preserved_x = -static_cast<LONG>(preserved_width);
    for (auto &spec : preserved) {
      const auto width = spec.source.width;
      spec.source.position = {preserved_x, 0};
      preserved_x += static_cast<LONG>(width);
      wanted.push_back(std::move(spec));
    }

    const auto desired = build_topology(*available, wanted);
    const auto desired_layout = desired ? inspect(*desired) : std::nullopt;
    if (!desired || !desired_layout) {
      return false;
    }
    if (journal.exclusive_started && same_topology(*snapshot, *desired) && same_colors(*snapshot, *desired)) {
      if (!journal.exclusive_topology || !same_topology(*journal.exclusive_topology, *desired) || !same_colors(*journal.exclusive_topology, *desired)) {
        journal.exclusive_topology = *desired;
        return valid_journal(journal) && io_.save(journal);
      }
      return true;
    }
    const bool remembered_exclusive = journal.exclusive_started && !journal.pending_restore && same_topology(*snapshot, *journal.before_exclusive);
    const bool continuing_exclusive = journal.exclusive_started &&
                                      ((journal.exclusive_topology && owned_subset(*snapshot, *journal.exclusive_topology)) ||
                                       exclusive_continuation_with_recorded_outputs(*snapshot, journal, device_path, reconcile_active_source) ||
                                       owned_exclusive_state(
                                         *snapshot,
                                         device_path,
                                         journal.exclusive_preserved,
                                         *journal.original_topology
                                       ));
    const bool resuming_restore = journal.exclusive_started && journal.pending_restore &&
                                  (owned_subset(*snapshot, *journal.pending_restore) ||
                                   same_layout_and_physical_modes(*snapshot, *journal.pending_restore, device_path) ||
                                   resume_layout_replaced_by_windows(*snapshot, *journal.pending_restore, device_path));
    if (journal.exclusive_started && !remembered_exclusive && !continuing_exclusive && !resuming_restore) {
      return false;
    }
    if (!remembered_exclusive && !continuing_exclusive && !resuming_restore && !same_layout(*layout, journal.original) && !same_layout(*layout, journal.promoted)) {
      return false;
    }
    snapshot_t current_physical, saved_physical;
    current_physical.modes = snapshot->modes;
    saved_physical.modes = journal.original_topology->modes;
    for (size_t i = 0; i < journal.original_topology->paths.size(); ++i) {
      if (same_device_path(journal.original_topology->device_paths[i], device_path)) {
        continue;
      }
      const auto current_index = find_path(*snapshot, journal.original_topology->device_paths[i]);
      if ((remembered_exclusive || continuing_exclusive || resuming_restore) && !current_index) {
        continue;
      }
      if (!current_index || *current_index >= snapshot->colors.size() || !snapshot->colors[*current_index] || (!remembered_exclusive && !continuing_exclusive && !resuming_restore && !color_matches(*snapshot->colors[*current_index], *journal.original_topology->colors[i]))) {
        return false;
      }
      current_physical.paths.push_back(snapshot->paths[*current_index]);
      current_physical.device_paths.push_back(snapshot->device_paths[*current_index]);
      saved_physical.paths.push_back(journal.original_topology->paths[i]);
      saved_physical.device_paths.push_back(journal.original_topology->device_paths[i]);
    }
    if (saved_physical.paths.empty() && !remembered_exclusive && !continuing_exclusive && !resuming_restore) {
      return io_.clear();
    }
    if (!remembered_exclusive && !continuing_exclusive && !resuming_restore && !same_modes(current_physical, saved_physical)) {
      return false;
    }
    journal.exclusive_started = true;
    journal.before_exclusive = *snapshot;
    journal.exclusive_topology = *desired;
    if (!valid_journal(journal) || !io_.save(journal)) {
      return false;
    }
    const auto latest = io_.query();
    if (!latest || !same_topology(*latest, *snapshot)) {
      return false;
    }
    if (!same_topology(*snapshot, *desired) && !io_.apply(*desired)) {
      return false;
    }
    auto observed = io_.query();
    if (!observed || !same_topology(*observed, *desired) || set_colors_and_settle(io_, *desired, color_requests(wanted), observed) != display_config::color_change_result_e::verified) {
      return false;
    }
    if (wanted.size() == 1) {
      BOOST_LOG(info) << "Virtual display is temporarily the only active Windows display.";
    } else {
      BOOST_LOG(info) << "Virtual display is temporarily primary with ordinary monitors disabled; kept "
                      << wanted.size() - 1 << " approved AR display(s) active.";
    }
    return true;
  }

  bool manager_t::pause(std::wstring_view device_path, retained_display_ptr &retained, std::wstring_view local_sink) {
    if (device_path.empty() || (retained && !same_device_path(retained->display.identity, device_path)) || (!local_sink.empty() && (same_device_path(local_sink, device_path) || (retained && !same_device_path(retained->local_sink, local_sink))))) {
      return false;
    }
    const auto loaded = io_.load();
    if (!loaded.success || (loaded.journal && (!valid_journal(*loaded.journal) || (!loaded.journal->prepared && !same_device_path(loaded.journal->promoted_primary, device_path)) || (!local_sink.empty() && !loaded.journal->local_sink.empty() && !same_device_path(local_sink, loaded.journal->local_sink)) || (retained && !loaded.journal->local_sink.empty() && !same_device_path(retained->local_sink, loaded.journal->local_sink))))) {
      return false;
    }
    const auto current = io_.query();
    if (!current || (!current->paths.empty() && !inspect(*current))) {
      return false;
    }
    const auto index = find_path(*current, device_path);
    if (!retained) {
      // Windows can deactivate the source or lose its color readback before disconnect
      // reaches us. That must not block physical recovery. Retain only this exact identity's
      // saved exclusive mode; reactivation still resolves and verifies the actual target.
      const bool current_snapshot_complete = index && *index < current->colors.size() && current->colors[*index];
      const auto *retained_snapshot = &*current;
      auto retained_index = index;
      if (!current_snapshot_complete && loaded.journal && loaded.journal->exclusive_topology) {
        retained_snapshot = &*loaded.journal->exclusive_topology;
        retained_index = find_path(*retained_snapshot, device_path);
      }
      if (!retained_index || *retained_index >= retained_snapshot->colors.size() || !retained_snapshot->colors[*retained_index]) {
        return false;
      }
      std::optional<POINT> cursor_before_pause;
      if (current_snapshot_complete && io_.query_cursor) {
        try {
          cursor_before_pause = io_.query_cursor();
        } catch (...) {
          // A cursor bookmark is optional; its failure must not delay display restoration.
        }
      }
      retained = std::make_shared<retained_display_t>(retained_display_t {display_spec(*retained_snapshot, *retained_index)});
      if (cursor_before_pause) {
        const auto &source = retained->display.source;
        const auto x = static_cast<std::int64_t>(cursor_before_pause->x) - source.position.x;
        const auto y = static_cast<std::int64_t>(cursor_before_pause->y) - source.position.y;
        if (x >= 0 && y >= 0 && x < source.width && y < source.height && x <= std::numeric_limits<LONG>::max() && y <= std::numeric_limits<LONG>::max()) {
          retained->cursor_offset = POINT {static_cast<LONG>(x), static_cast<LONG>(y)};
        }
      }
      retained->display.source.position = {0, 0};
      retained->local_sink = loaded.journal && !loaded.journal->local_sink.empty() ?
                               loaded.journal->local_sink :
                               std::wstring(local_sink);
      retained->monitor_replacement = loaded.journal && loaded.journal->monitor_replacement;
    }
    // Persist the in-memory bookmark before a clip release can move the pointer. All work
    // above is read-only; failed topology validation leaves both topology and clip unchanged.
    if (io_.cursor_clip && !io_.cursor_clip(device_path, std::nullopt)) {
      return false;
    }
    if (loaded.journal) {
      if (loaded.journal->exclusive_started) {
        return restore_exclusive(*loaded.journal, false, (retained->local_sink.empty() || retained->monitor_replacement) ? retained : retained_display_ptr {});
      }
      // A failed reactivation may have saved its physical baseline before binding. Complete
      // that recovery first, then detach any target which Windows already made active.
      return restore(device_path) && pause(device_path, retained, retained->local_sink);
    }
    if (retained->monitor_replacement && retained->pause_recovery) {
      if (!index && !current->paths.empty()) {
        // The detached desktop is now entirely user-owned. A missing selected sink must not
        // erase the retained monitor policy or its last owned rollback snapshot.
        return true;
      }
      // Windows can reactivate a retained VD after its successful pause journal was cleared.
      // Reuse that exact rollback proof even if the selected sink is currently unplugged;
      // remaining ordinary monitors are still a complete physical destination for detachment.
      return restore_monitor(*retained->pause_recovery, false, retained);
    }
    if (retained->pause_recovery && retained->pause_recovery->original_topology &&
        (index || retained->partial_wake_recovery) &&
        !current->paths.empty() && (!index || current->paths.size() > 1)) {
      const auto &remembered = *retained->pause_recovery->original_topology;
      const auto missing_remembered = [&](std::wstring_view identity) {
        return !same_device_path(identity, device_path) && !find_path(*current, identity);
      };
      if (std::ranges::any_of(remembered.device_paths, missing_remembered)) {
        const auto available = io_.query_all ? io_.query_all() : std::nullopt;
        if (!available || available->paths.size() > max_available_paths || available->paths.size() != available->device_paths.size()) {
          return false;
        }
        const bool missing_output_available = std::ranges::any_of(remembered.device_paths, [&](const auto &identity) {
          if (!missing_remembered(identity)) {
            return false;
          }
          for (size_t i = 0; i < available->paths.size(); ++i) {
            if (available->paths[i].targetInfo.targetAvailable && same_device_path(available->device_paths[i], identity)) {
              return true;
            }
          }
          return false;
        });
        retained->partial_wake_recovery = true;
        if (index || missing_output_available) {
          // A wake can reattach our VD with only part of the restored physical desktop.
          // Recover available remembered outputs through the existing transaction instead
          // of replacing its baseline with that transient subset. Current physical edits
          // still use restore_exclusive's ordinary user-desktop preservation rules.
          return restore_exclusive(*retained->pause_recovery, false, retained);
        }
        // The VD is already detached and every absent remembered output is unavailable.
        // Keep the usable physical desktop and its prior recovery bookmark without a CCD
        // mutation; a later availability observation can recover a waking monitor.
        return true;
      }
    }
    if (!index && !current->paths.empty()) {
      // A healthy disconnected desktop is user-owned. Remember its latest arrangement, modes
      // and color, including deliberate changes made after the first successful restore.
      if (retained->pause_recovery) {
        auto recovery = *retained->pause_recovery;
        if (!rebase_recovery(recovery, *current, *current)) {
          return false;
        }
        recovery.pending_restore = *current;
        if (!valid_journal(recovery)) {
          return false;
        }
        retained->pause_recovery = std::move(recovery);
        retained->partial_wake_recovery = false;
      }
      return true;
    }
    if (current->paths.empty() || (index && current->paths.size() == 1)) {
      if (retained->pause_recovery) {
        retained->partial_wake_recovery = true;
        const auto available = io_.query_all ? io_.query_all() : std::nullopt;
        if (!available || available->paths.size() > max_available_paths || available->paths.size() != available->device_paths.size()) {
          return false;
        }
        for (size_t i = 0; i < available->paths.size(); ++i) {
          if (available->paths[i].targetInfo.targetAvailable && find_path(*retained->pause_recovery->original_topology, available->device_paths[i])) {
            // Reconstruct current routes from the last verified physical modes. The ordinary
            // restore transaction journals and verifies every mutation before detaching the VD.
            return restore_exclusive(*retained->pause_recovery, false, retained);
          }
        }
      }
      // No remembered physical output is usable yet. Preserve both the sole VD and recovery
      // intent; the grace poll retries, and an authorized headless reconnect remains possible.
      return true;
    }
    BOOST_LOG(info) << "Retained virtual display is active again; detaching it from "
                    << current->paths.size() << " active outputs while preserving the current physical desktop.";
    // Reconnect rollback can reach us after its activation journal was completed. Save the
    // current physical desktop before removing this still-active target from that desktop.
    const auto connected_sink = find_path(*current, retained->local_sink) ? std::wstring_view(retained->local_sink) : std::wstring_view {};
    if (!prepare(true, connected_sink, retained->monitor_replacement) || !bind_pending(device_path)) {
      return false;
    }
    const auto prepared = io_.load();
    if (!prepared.success || !prepared.journal) {
      return false;
    }
    auto journal = *prepared.journal;
    journal.exclusive_started = true;
    journal.before_exclusive = *current;
    if (!valid_journal(journal) || !io_.save(journal)) {
      return false;
    }
    return restore_exclusive(std::move(journal), false, (retained->local_sink.empty() || retained->monitor_replacement) ? retained : retained_display_ptr {});
  }

  void manager_t::restore_retained_cursor(const retained_display_ptr &retained) {
    if (!retained || !retained->cursor_offset || !io_.query || !io_.set_cursor) {
      return;
    }
    try {
      const auto current = io_.query();
      if (!current || !inspect(*current)) {
        return;
      }
      const auto index = find_path(*current, retained->display.identity);
      if (!index) {
        return;
      }
      const auto source = display_spec(*current, *index).source;
      if (!source.width || !source.height) {
        return;
      }
      const auto x = static_cast<std::int64_t>(source.position.x) +
                     std::min<std::int64_t>(retained->cursor_offset->x, static_cast<std::int64_t>(source.width) - 1);
      const auto y = static_cast<std::int64_t>(source.position.y) +
                     std::min<std::int64_t>(retained->cursor_offset->y, static_cast<std::int64_t>(source.height) - 1);
      if (x < std::numeric_limits<LONG>::min() || x > std::numeric_limits<LONG>::max() || y < std::numeric_limits<LONG>::min() || y > std::numeric_limits<LONG>::max()) {
        return;
      }
      (void) io_.set_cursor(POINT {static_cast<LONG>(x), static_cast<LONG>(y)});
    } catch (...) {
      // Cursor restoration is best effort and never changes display/session admission.
    }
  }

  bool manager_t::reactivate(const retained_display_ptr &retained, bool exclusive) {
    // Reasons are static literals. Local AR may retry the same retained state periodically;
    // report a changed failed proof once without monitor identities or altering rollback.
    const auto reject = [&](std::string_view reason) {
      if (retained && retained->last_resume_failure == reason) {
        return false;
      }
      if (retained) {
        retained->last_resume_failure = reason;
      }
      BOOST_LOG(warning) << "Retained virtual-display resume rejected: " << reason << '.';
      return false;
    };
    if (!retained || !io_.query_all || !io_.set_color) {
      return reject("retained state or required display callbacks are unavailable");
    }
    const auto &identity = retained->display.identity;
    auto loaded = io_.load();
    if (!loaded.success) {
      return reject("display recovery journal could not be loaded");
    }
    if (loaded.journal && (!valid_journal(*loaded.journal) || !loaded.journal->exclusive || !equal_device_paths(loaded.journal->local_sink, retained->local_sink) || loaded.journal->prepared || !same_device_path(loaded.journal->promoted_primary, identity))) {
      return reject("display recovery journal does not prove ownership of this retained output and sink");
    }
    const auto current = io_.query();
    if (!current) {
      return reject("active display snapshot could not be queried");
    }
    if (!current->paths.empty() && !inspect(*current)) {
      return reject("active display snapshot has invalid modes or layout");
    }
    if (current->colors.size() != current->paths.size() || std::ranges::any_of(current->colors, [](const auto &color) {
          return !color;
        })) {
      return reject("active display snapshot lacks complete color state");
    }
    if (!retained->local_sink.empty() && !find_path(*current, retained->local_sink)) {
      return reject("the retained local output is not active");
    }
    auto verify_colors = [&](const snapshot_t &desired) {
      auto observed = io_.query();
      if (!observed) {
        return reject("display snapshot could not be queried before resume color verification");
      }
      if (!same_topology(*observed, desired)) {
        return reject("observed topology differs from the requested resume topology before color verification");
      }
      const auto colors = set_colors_and_settle(io_, desired, color_requests(desired), observed);
      if (colors == display_config::color_change_result_e::verified) {
        return true;
      }
      if (colors == display_config::color_change_result_e::rejected) {
        return reject("a resumed output lacks or rejected its requested color state");
      }
      if (!observed) {
        return reject("display snapshot could not be queried after resume color setup");
      }
      if (!same_topology(*observed, desired)) {
        return reject("observed topology differs from the requested resume topology after color setup");
      }
      return reject("observed color state differs from the requested resume color state");
    };
    if (find_path(*current, identity)) {
      // Idempotence covers a previous CCD apply whose color verification had to retry, and
      // the sole-output pause which deliberately kept a headless virtual desktop active.
      if (!loaded.journal) {
        const auto index = *find_path(*current, identity);
        std::string_view route_failure;
        const auto wanted = build_topology(*current, {retained->display}, &route_failure);
        snapshot_t active_target = *current;
        active_target.paths = {current->paths[index]};
        active_target.device_paths = {current->device_paths[index]};
        if (!wanted) {
          return reject(route_failure);
        }
        if (!same_modes(active_target, *wanted)) {
          return reject("the already-active retained output mode differs from its saved pause mode");
        }
        // A local-only PC keeps its sole VD when the glasses are unplugged. Their return
        // needs a fresh local sink journal even though the source never needed activation.
        return !exclusive || retained->local_sink.empty() ||
               (prepare(true, retained->local_sink, retained->monitor_replacement) && bind_pending(identity)) ||
               reject("the already-active local source could not prepare and bind its recovery journal");
      }
      const auto &journal = *loaded.journal;
      if (!journal.pending_restore || !same_topology(*current, *journal.pending_restore)) {
        return reject("the already-active retained output topology differs from the pending recovery topology");
      }
      if (!verify_colors(*journal.pending_restore)) {
        return false;
      }
      return exclusive || restore(identity) || reject("primary-only resume could not complete its recovery journal");
    }
    if (loaded.journal && (!loaded.journal->pending_restore || !owned_subset(*current, *loaded.journal->pending_restore))) {
      return reject("active physical outputs are not an owned subset of the pending recovery topology");
    }
    const auto available = io_.query_all();
    if (!available) {
      return reject("available display routes could not be queried");
    }
    std::vector<display_spec_t> wanted;
    std::int64_t right = current->paths.empty() ? 0 : std::numeric_limits<LONG>::min();
    LONG top = 0;
    for (size_t i = 0; i < current->paths.size(); ++i) {
      auto spec = display_spec(*current, i);
      if (loaded.journal && loaded.journal->pending_restore) {
        const auto saved = find_path(*loaded.journal->pending_restore, spec.identity);
        if (saved && *saved < loaded.journal->pending_restore->colors.size()) {
          spec.color = loaded.journal->pending_restore->colors[*saved];
        }
      }
      const auto edge = static_cast<std::int64_t>(spec.source.position.x) + spec.source.width;
      if (edge > right) {
        right = edge;
        top = spec.source.position.y;
      }
      wanted.push_back(std::move(spec));
    }
    if (right > std::numeric_limits<LONG>::max() || right + retained->display.source.width > std::numeric_limits<LONG>::max()) {
      return reject("the retained output would overflow desktop coordinates");
    }
    auto virtual_spec = retained->display;
    virtual_spec.source.position = {static_cast<LONG>(right), top};
    wanted.push_back(std::move(virtual_spec));
    std::string_view route_failure;
    const auto desired = build_topology(*available, wanted, &route_failure);
    const auto layout = desired ? inspect(*desired) : std::nullopt;
    if (!desired) {
      return reject(route_failure);
    }
    if (!layout) {
      return reject("the reconstructed resume layout is invalid");
    }
    if (!loaded.journal) {
      // Full color recovery is required even if this reconnect switches to primary-only
      // mode. Its ordinary primary-only journal starts after activation completes below.
      if (current->paths.empty()) {
        // The added VD position and original physical topology are separate journal fields.
        // Record the actual empty baseline when physical outputs vanish during the pause.
        // A local presentation cannot resume without its physical sink.
        if (!retained->local_sink.empty()) {
          return reject("a local session cannot resume without an active physical output");
        }
        journal_t headless {identity, identity, *layout, *layout};
        headless.exclusive = true;
        headless.original_topology = *current;
        loaded.journal = std::move(headless);
      } else {
        if (!prepare(true, retained->local_sink, retained->monitor_replacement)) {
          return reject("the physical desktop recovery baseline could not be prepared");
        }
        loaded = io_.load();
        if (!loaded.success || !loaded.journal) {
          return reject("the prepared physical desktop recovery journal could not be loaded");
        }
      }
    }
    auto journal = *loaded.journal;
    if (journal.prepared) {
      journal.prepared = false;
      journal.promoted_primary = identity;
      journal.original = *layout;
      const auto promoted = translated(*layout, static_cast<LONG>(right), top);
      if (!promoted) {
        return reject("the promoted resume layout would overflow desktop coordinates");
      }
      journal.promoted = *promoted;
    }
    journal.exclusive_started = true;
    journal.before_exclusive = *current;
    journal.pending_restore = *desired;
    if (!valid_journal(journal)) {
      return reject("the proposed resume recovery journal is invalid");
    }
    if (!io_.save(journal)) {
      return reject("the proposed resume recovery journal could not be saved");
    }
    const auto latest = io_.query();
    if (!latest) {
      return reject("the physical desktop could not be queried before resume apply");
    }
    if (!same_topology(*latest, *current)) {
      return reject("the physical desktop topology changed before resume apply");
    }
    if (!same_colors(*latest, *current)) {
      return reject("the physical desktop color state changed before resume apply");
    }
    if (!io_.apply(*desired)) {
      return reject("Windows rejected the requested resume topology");
    }
    if (!verify_colors(*desired)) {
      return false;
    }
    if (!bind_pending(identity)) {
      return reject("the resumed virtual output could not bind its recovery journal");
    }
    BOOST_LOG(info) << "Reactivated the retained virtual display for session resume.";
    return exclusive || restore(identity) || reject("primary-only resume could not complete its recovery journal");
  }

  bool manager_t::restore_monitor(journal_t journal, bool keep_virtual_active, retained_display_ptr retained) {
    const auto current = io_.query();
    const auto available = io_.query_all ? io_.query_all() : std::nullopt;
    if (!current || (!current->paths.empty() && !inspect(*current)) || !available || available->paths.size() != available->device_paths.size() || available->paths.size() > max_available_paths) {
      return false;
    }
    const auto source_index_now = find_path(*current, journal.promoted_primary);
    const bool owned = journal.exclusive_topology && owned_subset(*current, *journal.exclusive_topology);
    bool source_replaced_sink = false;
    if (source_index_now && journal.exclusive_topology && journal.before_exclusive) {
      const auto before_source = find_path(*journal.before_exclusive, journal.promoted_primary);
      const auto actual_position = display_spec(*current, *source_index_now).source.position;
      const auto before_position = before_source ? display_spec(*journal.before_exclusive, *before_source).source.position : actual_position;
      // A saved desired topology proves intent, not that Windows applied it. In particular,
      // an Add-appended source is never authority for the physical sink's restore position.
      source_replaced_sink = owned || actual_position.x != before_position.x || actual_position.y != before_position.y;
    }
    const auto is_available = [&](std::wstring_view identity) {
      return std::ranges::any_of(available->paths, [&](const auto &path) {
        const auto i = static_cast<size_t>(&path - available->paths.data());
        return path.targetInfo.targetAvailable && same_device_path(available->device_paths[i], identity);
      });
    };
    std::vector<display_spec_t> wanted;
    for (size_t i = 0; i < journal.original_topology->paths.size(); ++i) {
      const auto &identity = journal.original_topology->device_paths[i];
      if (same_device_path(identity, journal.promoted_primary) || !is_available(identity)) {
        continue;
      }
      auto spec = display_spec(*journal.original_topology, i);
      const auto baseline_color = spec.color;
      const auto live = find_path(*current, identity);
      if (live) {
        const auto baseline_position = spec.source.position;
        spec = display_spec(*current, *live);
        if (owned || same_device_path(identity, journal.local_sink)) {
          spec.source.position = baseline_position;
        }
        if (owned) spec.color = baseline_color;
        if (same_device_path(identity, journal.local_sink) && source_index_now && source_replaced_sink && !journal.pending_restore) {
          spec.source.position = display_spec(*current, *source_index_now).source.position;
        }
      }
      if (journal.pending_restore) {
        const auto pending = find_path(*journal.pending_restore, identity);
        if (pending && *pending < journal.pending_restore->colors.size()) spec.color = journal.pending_restore->colors[*pending];
      }
      wanted.push_back(std::move(spec));
    }
    for (size_t i = 0; i < current->paths.size(); ++i) {
      if (!same_device_path(current->device_paths[i], journal.promoted_primary) && !find_path(*journal.original_topology, current->device_paths[i])) {
        wanted.push_back(display_spec(*current, i));
      }
    }
    if (wanted.empty()) {
      const bool physical_available = std::ranges::any_of(available->paths, [&](const auto &path) {
        const auto i = static_cast<size_t>(&path - available->paths.data());
        return path.targetInfo.targetAvailable && !same_device_path(available->device_paths[i], journal.promoted_primary);
      });
      if (physical_available) {
        // An output exists but is not yet part of a restorable desktop. Keep the durable
        // transaction until Windows can restore an actual physical desktop.
        return false;
      }
      // Every physical output is gone, as on a single-monitor PC whose 3D monitor was unplugged
      // or powered off. As with unplugged glasses, the PC is intentionally headless and there
      // is nothing to restore; waiting here would block the next remote session indefinitely.
      if (retained) {
        retained->pause_recovery.reset();
      }
      return io_.clear();
    }
    if (!normalize_monitor_desktop(wanted, journal.original_primary)) {
      return false;
    }
    if (!monitor_desktop_connected(wanted)) {
      const auto primary = std::ranges::find_if(wanted, [](const auto &spec) {
        return spec.source.position.x == 0 && spec.source.position.y == 0;
      });
      if (primary == wanted.end() || !pack_horizontally(wanted, primary->identity, 0, 0)) {
        return false;
      }
    }
    if (keep_virtual_active && source_index_now) {
      auto source = display_spec(*current, *source_index_now);
      if (!park_beside_desktop(source, wanted)) {
        return false;
      }
      wanted.push_back(std::move(source));
    }
    const auto desired = build_topology(*available, wanted);
    if (!desired) {
      return false;
    }
    journal.pending_restore = *desired;
    if (!valid_journal(journal) || !io_.save(journal) || !apply_snapshot_verified(*current, *desired)) {
      return false;
    }
    if (retained) {
      retained->monitor_replacement = true;
      retained->pause_recovery = journal;
    }
    return io_.clear();
  }

  bool manager_t::restore_exclusive(journal_t journal, bool keep_virtual_active, retained_display_ptr retained) {
    if (journal.monitor_replacement) {
      return restore_monitor(std::move(journal), keep_virtual_active, std::move(retained));
    }
    const auto restore_started = std::chrono::steady_clock::now();
    if (!io_.query_all || !io_.set_color) {
      return false;
    }
    const auto current = io_.query();
    if (!current || current->colors.size() != current->paths.size() || (!current->paths.empty() && !inspect(*current))) {
      return false;
    }
    const auto virtual_index = find_path(*current, journal.promoted_primary);
    snapshot_t owned_current = *current;
    std::vector<display_spec_t> added_outputs;
    bool newly_recorded_output = false;
    bool preservation_changed = false;
    for (size_t i = current->paths.size(); i-- > 0;) {
      const auto &identity = current->device_paths[i];
      if (same_device_path(identity, journal.promoted_primary) || find_path(*journal.original_topology, identity)) {
        continue;
      }
      const auto pending_index = journal.pending_restore ? find_path(*journal.pending_restore, identity) : std::nullopt;
      newly_recorded_output = newly_recorded_output || !pending_index;
      added_outputs.push_back(pending_index ? display_spec(*journal.pending_restore, *pending_index) : display_spec(*current, i));
      if (journal.local_sink.empty() && io_.preserve_exclusive && io_.preserve_exclusive(current->paths[i], identity) && !preserved_exclusive_identity(journal, identity)) {
        journal.exclusive_preserved.push_back(identity);
        preservation_changed = true;
      }
      owned_current.paths.erase(owned_current.paths.begin() + static_cast<std::ptrdiff_t>(i));
      owned_current.device_paths.erase(owned_current.device_paths.begin() + static_cast<std::ptrdiff_t>(i));
      owned_current.colors.erase(owned_current.colors.begin() + static_cast<std::ptrdiff_t>(i));
    }
    std::ranges::sort(journal.exclusive_preserved, [](const auto &left, const auto &right) {
      return CompareStringOrdinal(left.c_str(), -1, right.c_str(), -1, TRUE) == CSTR_LESS_THAN;
    });
    const auto owned_virtual_index = find_path(owned_current, journal.promoted_primary);
    const bool owned_exclusive = owned_virtual_index &&
                                 ((journal.exclusive_topology &&
                                   owned_exclusive_layout(owned_current, journal.promoted_primary, *journal.exclusive_topology)) ||
                                  owned_exclusive_state(
                                    owned_current,
                                    journal.promoted_primary,
                                    journal.exclusive_preserved,
                                    *journal.original_topology
                                  ));
    // SudoVDA may retire the source when an older host exits before it can recover a
    // hardware mode switch. The remaining exact recorded local outputs still need their
    // saved ordinary layout restored; no replacement or unrecorded output gains ownership.
    const bool retired_local_source = !journal.local_sink.empty() && !virtual_index &&
                                      std::ranges::all_of(current->device_paths, [&](const auto &identity) {
                                        return find_path(*journal.original_topology, identity) ||
                                               (journal.pending_restore && find_path(*journal.pending_restore, identity));
                                      });
    const bool paused_activation = !keep_virtual_active && virtual_index && journal.pending_restore &&
                                   same_layout_and_physical_modes(*current, *journal.pending_restore, journal.promoted_primary);
    const bool owned = owned_current.paths.empty() || owned_exclusive || retired_local_source || paused_activation || same_topology(owned_current, *journal.before_exclusive) ||
                       exclusive_continuation_with_recorded_outputs(*current, journal, journal.promoted_primary, false) ||
                       (journal.pending_restore && owned_subset(owned_current, *journal.pending_restore)) ||
                       (!owned_virtual_index && owned_subset(owned_current, *journal.original_topology));
    const auto available = io_.query_all();
    if (!available || available->paths.size() > max_available_paths || available->paths.size() != available->device_paths.size()) {
      return false;
    }
    auto available_identity = [&](std::wstring_view identity) {
      std::optional<std::pair<LUID, UINT32>> target;
      for (size_t i = 0; i < available->paths.size(); ++i) {
        const auto &path = available->paths[i];
        if (!path.targetInfo.targetAvailable || !same_device_path(identity, available->device_paths[i])) {
          continue;
        }
        if (target && (!same_luid(target->first, path.targetInfo.adapterId) || target->second != path.targetInfo.id)) {
          return false;
        }
        target = std::pair {path.targetInfo.adapterId, path.targetInfo.id};
      }
      return target.has_value();
    };
    if (journal.pending_restore) {
      for (size_t i = 0; i < journal.pending_restore->paths.size(); ++i) {
        const auto &identity = journal.pending_restore->device_paths[i];
        if (same_device_path(identity, journal.promoted_primary) || find_path(*journal.original_topology, identity) || !available_identity(identity) || std::ranges::any_of(added_outputs, [&](const auto &entry) {
              return same_device_path(entry.identity, identity);
            })) {
          continue;
        }
        added_outputs.push_back(display_spec(*journal.pending_restore, i));
      }
    }
    std::ranges::sort(added_outputs, [](const auto &left, const auto &right) {
      return CompareStringOrdinal(left.identity.c_str(), -1, right.identity.c_str(), -1, TRUE) == CSTR_LESS_THAN;
    });
    std::vector<std::wstring> retired_preserved;
    std::erase_if(journal.exclusive_preserved, [&](const auto &identity) {
      const bool retire = !find_path(*journal.original_topology, identity) &&
                          std::ranges::none_of(added_outputs, [&](const auto &added) {
                            return same_device_path(added.identity, identity);
                          });
      if (retire) {
        retired_preserved.push_back(identity);
      }
      return retire;
    });
    if (journal.exclusive_topology) {
      for (size_t i = journal.exclusive_topology->paths.size(); i-- > 0;) {
        if (std::ranges::none_of(retired_preserved, [&](const auto &identity) {
              return same_device_path(journal.exclusive_topology->device_paths[i], identity);
            })) {
          continue;
        }
        journal.exclusive_topology->paths.erase(journal.exclusive_topology->paths.begin() + static_cast<std::ptrdiff_t>(i));
        journal.exclusive_topology->device_paths.erase(journal.exclusive_topology->device_paths.begin() + static_cast<std::ptrdiff_t>(i));
        journal.exclusive_topology->colors.erase(journal.exclusive_topology->colors.begin() + static_cast<std::ptrdiff_t>(i));
      }
    }
    if ((newly_recorded_output || preservation_changed) && owned) {
      auto record = added_outputs;
      std::wstring anchor;
      if (virtual_index) {
        auto virtual_spec = display_spec(*current, *virtual_index);
        anchor = virtual_spec.identity;
        record.push_back(std::move(virtual_spec));
      } else if (!record.empty()) {
        anchor = record.front().identity;
      }
      if (anchor.empty() || !pack_horizontally(record, anchor, 0, 0)) {
        return false;
      }
      const auto provisional_restore = build_topology(*available, record);
      if (!provisional_restore) {
        return false;
      }
      journal.before_exclusive = *current;
      journal.pending_restore = *provisional_restore;
      if (!valid_journal(journal) || !io_.save(journal)) {
        return false;
      }
    }
    const bool had_added_outputs = !added_outputs.empty();
    std::vector<display_spec_t> wanted;
    bool missing = false;
    bool missing_local_sink = false;
    bool local_sink_resized = false;
    size_t physical_count = 0;
    for (size_t i = 0; i < journal.original_topology->paths.size(); ++i) {
      const auto &identity = journal.original_topology->device_paths[i];
      if (same_device_path(identity, journal.promoted_primary)) {
        continue;
      }
      ++physical_count;
      if (!available_identity(identity)) {
        if (!journal.local_sink.empty() && same_device_path(identity, journal.local_sink)) {
          missing_local_sink = true;
        } else {
          missing = true;
        }
        continue;
      }
      auto spec = display_spec(*journal.original_topology, i);
      const auto sink_index = find_path(*current, identity);
      if (!journal.local_sink.empty() && same_device_path(identity, journal.local_sink) && sink_index) {
        const auto saved_position = spec.source.position;
        const auto saved_width = spec.source.width;
        const auto saved_height = spec.source.height;
        spec = display_spec(*current, *sink_index);
        local_sink_resized = spec.source.width != saved_width || spec.source.height != saved_height;
        spec.source.position = saved_position;
        // A sink originally attached to the primary's left keeps its right edge when
        // its hardware switches width. Restoring the old left edge would overlap it.
        if (saved_position.x < 0 && static_cast<std::int64_t>(saved_position.x) + saved_width <= 0) {
          const auto rebased_x = static_cast<std::int64_t>(saved_position.x) + saved_width - spec.source.width;
          if (rebased_x < std::numeric_limits<LONG>::min()) {
            return false;
          }
          spec.source.position.x = static_cast<LONG>(rebased_x);
        }
      }
      wanted.push_back(std::move(spec));
    }
    if (physical_count == 0 && added_outputs.empty()) {
      if (!io_.clear()) {
        return false;
      }
      if (retained) {
        retained->pause_recovery.reset();
        retained->partial_wake_recovery = false;
      }
      return true;
    }
    if (missing_local_sink && !missing && physical_count == 1 && wanted.empty() && added_outputs.empty()) {
      // A PC whose only physical output was these glasses is intentionally headless
      // after they are unplugged. There is no remaining physical output to restore.
      return io_.clear();
    }
    if (physical_count != 0 && wanted.empty() && added_outputs.empty()) {
      return false;
    }
    if (wanted.empty()) {
      wanted.push_back(std::move(added_outputs.front()));
      added_outputs.erase(added_outputs.begin());
    }
    auto primary = std::ranges::find_if(wanted, [&](const auto &spec) {
      return same_device_path(spec.identity, journal.original_primary);
    });
    std::wstring restore_primary;
    POINTL origin {};
    if (primary != wanted.end()) {
      restore_primary = primary->identity;
      origin = primary->source.position;
      std::rotate(wanted.begin(), primary, primary + 1);
    } else if (keep_virtual_active && same_device_path(journal.original_primary, journal.promoted_primary) && virtual_index && find_path(*journal.original_topology, journal.original_primary)) {
      const auto original_primary_index = find_path(*journal.original_topology, journal.original_primary);
      if (!original_primary_index) {
        return false;
      }
      restore_primary = journal.original_primary;
      origin = journal.original_topology->modes[source_index(journal.original_topology->paths[*original_primary_index])].sourceMode.position;
    } else {
      primary = wanted.begin();
      restore_primary = primary->identity;
      origin = primary->source.position;
      std::rotate(wanted.begin(), primary, primary + 1);
    }
    if (!added_outputs.empty()) {
      std::int64_t rightmost = std::numeric_limits<LONG>::min();
      for (const auto &spec : wanted) {
        const auto edge = static_cast<std::int64_t>(spec.source.position.x) + spec.source.width;
        if (edge > std::numeric_limits<LONG>::max()) {
          return false;
        }
        rightmost = std::max(rightmost, edge);
      }
      for (auto &spec : added_outputs) {
        if (rightmost > std::numeric_limits<LONG>::max() || rightmost + spec.source.width > std::numeric_limits<LONG>::max()) {
          return false;
        }
        spec.source.position = {static_cast<LONG>(rightmost), origin.y};
        rightmost += spec.source.width;
        wanted.push_back(std::move(spec));
      }
    }
    if (virtual_index && keep_virtual_active) {
      if (!available_identity(journal.promoted_primary)) {
        return false;
      }
      auto virtual_spec = display_spec(*current, *virtual_index);
      if (journal.pending_restore) {
        const auto pending_index = find_path(*journal.pending_restore, journal.promoted_primary);
        if (pending_index && *pending_index < journal.pending_restore->colors.size() && journal.pending_restore->colors[*pending_index]) {
          // A prior CCD apply may have reset HDR before its color setter failed. Keep
          // that durable color intent across retries while preserving the live VD mode.
          virtual_spec.color = journal.pending_restore->colors[*pending_index];
        }
      }
      const auto *original = find_position(journal.original, journal.promoted_primary);
      virtual_spec.source.position = {original->x, original->y};
      const auto overlaps = [&](const display_spec_t &physical) {
        return sources_overlap(virtual_spec, physical);
      };
      if (std::ranges::any_of(wanted, overlaps)) {
        int64_t right = std::numeric_limits<LONG>::min();
        LONG top = 0;
        for (const auto &physical : wanted) {
          const int64_t edge = static_cast<int64_t>(physical.source.position.x) + physical.source.width;
          if (edge > right) {
            right = edge;
            top = physical.source.position.y;
          }
        }
        if (right > std::numeric_limits<LONG>::max()) {
          return false;
        }
        virtual_spec.source.position = {static_cast<LONG>(right), top};
      }
      // The virtual mode may legitimately change during the session. Preserve its current
      // mode and color; it must stay active for orderly retirement and warm reconnect.
      wanted.push_back(std::move(virtual_spec));
    }
    // User-owned active outputs override the saved plan before its geometry is validated.
    // Removing a VD that bridged two physical monitors can disconnect that arrangement.
    if (!owned) {
      if (!preserve_current_desktop(*current, wanted, journal.promoted_primary, keep_virtual_active)) {
        return false;
      }
      primary = std::ranges::find_if(wanted, [](const auto &spec) {
        return spec.source.position.x == 0 && spec.source.position.y == 0;
      });
      if (primary == wanted.end()) {
        return false;
      }
      restore_primary = primary->identity;
      origin = {};
      BOOST_LOG(info) << "Preserving the user's current desktop while restoring available disabled outputs.";
    }
    // The local sink comes back at its saved position but in its current mode. Overlap is checked
    // whenever there is a sink; a gap needs its size to have changed, so connectivity is checked
    // only then. A desktop this host did not arrange is checked both ways.
    const bool restore_overlap = (!journal.local_sink.empty() || !owned) && any_sources_overlap(wanted);
    const bool restore_disconnected =
      (local_sink_resized || !owned) && !wanted.empty() && !sources_edge_connected(wanted);
    if ((owned && (missing || missing_local_sink)) || restore_overlap || restore_disconnected) {
      // Removing an unavailable monitor from the saved arrangement can leave a hole between
      // survivors. Restore a contiguous usable desktop; absent originals do not retain
      // a blocking recovery transaction after the available outputs are verified.
      if (!pack_horizontally(wanted, restore_primary, 0, 0)) {
        return false;
      }
    } else {
      for (auto &spec : wanted) {
        const auto x = subtract(spec.source.position.x, origin.x);
        const auto y = subtract(spec.source.position.y, origin.y);
        if (!x || !y) {
          return false;
        }
        spec.source.position = {*x, *y};
      }
    }
    const auto desired = build_topology(*available, wanted);
    if (!desired) {
      return false;
    }
    if (!owned && !rebase_recovery(journal, *current, *desired)) {
      return false;
    }
    journal.pending_restore = *desired;
    if (had_added_outputs) {
      // These outputs were not present in the launch baseline. Persist the exact pre-restore
      // ownership point before SetDisplayConfig so a crash remains recoverable.
      journal.before_exclusive = *current;
    }
    if (!valid_journal(journal) || !io_.save(journal)) {
      return false;
    }
    const auto latest = io_.query();
    if (!latest || !same_topology(*current, *latest)) {
      return false;
    }
    const bool apply_needed = !same_topology(*current, *desired);
    if (apply_needed && restore_observations_ && restore_observations_->unchanged_failure(*current, *desired, *available)) {
      // One failed apply with a verified outcome is enough for this exact observation.
      // Repeating it cannot supply new evidence and can flash a healthy physical monitor.
      // Keep the journal and wake bookmark; raw WM_DISPLAYCHANGE alone grants no retry.
      return false;
    }
    log_restore_snapshot("before", *current);
    log_restore_snapshot("requested", *desired);
    const auto apply_started = std::chrono::steady_clock::now();
    bool applied = true;
    if (apply_needed) {
      const bool force_physical_restore = !keep_virtual_active && retained && retained->local_sink.empty() && io_.apply_physical_restore;
      if (force_physical_restore) {
        BOOST_LOG(info) << "Forcing the restored physical-display modes down to the driver while retaining the virtual-display device.";
      }
      applied = force_physical_restore ? io_.apply_physical_restore(*desired) : io_.apply(*desired);
    }
    const auto apply_finished = std::chrono::steady_clock::now();
    if (!applied && !restore_observations_) {
      return false;
    }
    auto observed = io_.query();
    if (!applied || !observed || !same_topology(*observed, *desired)) {
      if (apply_needed && restore_observations_) {
        const bool observed_valid = observed && observed->colors.size() == observed->paths.size() &&
                                    (observed->paths.empty() || inspect(*observed));
        // If the immediate query is unavailable, retain the last valid pre-apply view.
        // Suppression still requires a fresh next-call query to match that exact view.
        restore_observations_->remember_failure(observed_valid ? *observed : *current, *desired, *available);
        BOOST_LOG(warning) << "Display restore did not reach the requested topology; preserving recovery state and deferring identical mode sets until display observations change.";
      }
      return false;
    }
    const auto requests = color_requests(wanted);
    const auto color_changes = std::ranges::count_if(requests, [&](const auto &request) {
      const auto index = find_path(*observed, request.identity);
      return index && *index < observed->colors.size() && observed->colors[*index] && !color_matches(*observed->colors[*index], *request.color);
    });
    const auto colors = set_colors_and_settle(io_, *desired, requests, observed);
    if (config::sunshine.diagnostics_enabled) {
      const auto now = std::chrono::steady_clock::now();
      BOOST_LOG(info) << "Display restore timing: apply_needed=" << apply_needed
                      << " apply_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(apply_finished - apply_started).count()
                      << " color_changes=" << color_changes
                      << " verify_color_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(now - apply_finished).count()
                      << " total_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(now - restore_started).count();
      if (observed) {
        log_restore_snapshot("observed", *observed);
      }
    }
    if (colors != display_config::color_change_result_e::verified) {
      if (observed && same_topology(*observed, *desired)) {
        BOOST_LOG(warning) << "Available physical displays are active; retaining exclusive-display recovery until their color settings are restored.";
      }
      return false;
    }
    const bool preserve_verified_pause_recovery = retained && retained->partial_wake_recovery && retained->pause_recovery && missing;
    if (missing || missing_local_sink) {
      // All available outputs were restored and verified; an unplugged monitor must not
      // hold a durable transaction open. A prior verified pause bookmark still covers a
      // transient wake failure, while an initially absent output starts fresh on reconnect.
      BOOST_LOG(info) << (preserve_verified_pause_recovery ?
                           "Restored available physical displays; retaining the last verified desktop for monitor-wake recovery." :
                           "Restored available physical displays; absent monitors will use a fresh topology baseline on reconnect.");
    }
    if (retained && !preserve_verified_pause_recovery) {
      // The first completed recovery defines the pause baseline, and a later complete
      // physical desktop adopts accepted user edits. A partial wake must not replace a
      // previously verified desktop with its temporary subset.
      if (!rebase_recovery(journal, *observed, *observed)) {
        return false;
      }
      journal.pending_restore = *observed;
      if (!valid_journal(journal)) {
        return false;
      }
    }
    BOOST_LOG(info) << (keep_virtual_active ? "Restored physical displays, modes, color settings, and primary display after exclusive streaming." : "Verified physical-display topology, modes, and color; detached the retained virtual display from the desktop.");
    if (!io_.clear()) {
      return false;
    }
    if (retained && !preserve_verified_pause_recovery) {
      retained->pause_recovery = std::move(journal);
      retained->partial_wake_recovery = false;
    }
    if (restore_observations_) {
      restore_observations_->clear();
    }
    return true;
  }

  bool manager_t::recover_prepared_outputs(journal_t journal) {
    if (!io_.query_all || !io_.set_color) {
      return false;
    }
    const auto current = io_.query();
    if (!current || (!current->paths.empty() && !inspect(*current))) {
      return false;
    }
    std::vector<display_spec_t> extras;
    size_t active_originals = 0;
    for (size_t i = 0; i < current->paths.size(); ++i) {
      if (find_path(*journal.original_topology, current->device_paths[i])) {
        ++active_originals;
      } else {
        extras.push_back(display_spec(*current, i));
      }
    }
    const bool prior_recovery = journal.pending_restore && owned_subset(*current, *journal.pending_restore);
    if (active_originals && !prior_recovery && (!extras.empty() || !owned_subset(*current, *journal.original_topology))) {
      return false;
    }
    const auto available = io_.query_all();
    if (!available || available->paths.size() > max_available_paths || available->paths.size() != available->device_paths.size()) {
      return false;
    }
    std::vector<display_spec_t> physical;
    bool missing = false;
    bool missing_local_sink = false;
    for (size_t i = 0; i < journal.original_topology->paths.size(); ++i) {
      const auto &identity = journal.original_topology->device_paths[i];
      bool found = false;
      for (size_t j = 0; j < available->paths.size(); ++j) {
        found = found || (available->paths[j].targetInfo.targetAvailable && same_device_path(available->device_paths[j], identity));
      }
      if (found) {
        physical.push_back(display_spec(*journal.original_topology, i));
      } else {
        if (!journal.local_sink.empty() && same_device_path(identity, journal.local_sink)) {
          missing_local_sink = true;
        } else {
          missing = true;
        }
      }
    }
    if (physical.empty()) {
      return missing_local_sink && !missing && extras.empty() && io_.clear();
    }
    if (missing || missing_local_sink) {
      std::wstring anchor_identity;
      LONG anchor_x = 0;
      LONG anchor_y = 0;
      if (!extras.empty()) {
        // An unbound output is never identified as ours or moved/disabled. Place the compacted
        // known block directly beside it until the driver orphan is gone or exactly bound.
        const auto rightmost = std::ranges::max_element(extras, [](const auto &a, const auto &b) {
          return static_cast<int64_t>(a.source.position.x) + a.source.width < static_cast<int64_t>(b.source.position.x) + b.source.width;
        });
        const auto right_edge = static_cast<int64_t>(rightmost->source.position.x) + rightmost->source.width;
        if (right_edge < std::numeric_limits<LONG>::min() || right_edge > std::numeric_limits<LONG>::max()) {
          return false;
        }
        const auto leftmost = std::ranges::min_element(physical, [](const auto &left, const auto &right) {
          if (left.source.position.x != right.source.position.x) {
            return left.source.position.x < right.source.position.x;
          }
          return left.source.position.y < right.source.position.y;
        });
        anchor_identity = leftmost->identity;
        anchor_x = static_cast<LONG>(right_edge);
        anchor_y = rightmost->source.position.y;
      } else {
        auto primary = std::ranges::find_if(physical, [&](const auto &spec) {
          return same_device_path(spec.identity, journal.original_primary);
        });
        if (primary == physical.end()) {
          primary = physical.begin();
        }
        anchor_identity = primary->identity;
      }
      if (!pack_horizontally(physical, anchor_identity, anchor_x, anchor_y)) {
        return false;
      }
    } else {
      int64_t offset_x = 0, offset_y = 0;
      if (!extras.empty()) {
        // An unbound output is never identified as ours or moved/disabled. Re-enable only
        // the known physical block alongside it, leaving its primary, modes and positions
        // alone until the driver's orphan cleanup or exact Add identity resolves ownership.
        const auto rightmost = std::ranges::max_element(extras, [](const auto &a, const auto &b) {
          return static_cast<int64_t>(a.source.position.x) + a.source.width < static_cast<int64_t>(b.source.position.x) + b.source.width;
        });
        const auto leftmost = std::ranges::min_element(physical, {}, [](const auto &spec) {
          return spec.source.position.x;
        });
        offset_x = static_cast<int64_t>(rightmost->source.position.x) + rightmost->source.width - leftmost->source.position.x;
        offset_y = static_cast<int64_t>(rightmost->source.position.y) - leftmost->source.position.y;
      } else {
        auto primary = std::ranges::find_if(physical, [&](const auto &spec) {
          return same_device_path(spec.identity, journal.original_primary);
        });
        if (primary == physical.end()) {
          primary = physical.begin();
        }
        offset_x = -static_cast<int64_t>(primary->source.position.x);
        offset_y = -static_cast<int64_t>(primary->source.position.y);
      }
      for (auto &spec : physical) {
        const int64_t x = spec.source.position.x + offset_x;
        const int64_t y = spec.source.position.y + offset_y;
        if (x < std::numeric_limits<LONG>::min() || x > std::numeric_limits<LONG>::max() || y < std::numeric_limits<LONG>::min() || y > std::numeric_limits<LONG>::max()) {
          return false;
        }
        spec.source.position = {static_cast<LONG>(x), static_cast<LONG>(y)};
      }
    }
    auto wanted = extras;
    wanted.insert(wanted.end(), physical.begin(), physical.end());
    const auto desired = build_topology(*available, wanted);
    if (!desired) {
      return false;
    }
    journal.pending_restore = *desired;
    if (!io_.save(journal)) {
      return false;
    }
    const auto latest = io_.query();
    if (!latest || !same_topology(*latest, *current)) {
      return false;
    }
    if (!same_topology(*current, *desired) && !io_.apply(*desired)) {
      return false;
    }
    auto observed = io_.query();
    if (!observed || !same_topology(*observed, *desired)) {
      return false;
    }
    std::vector<color_request_t> requests;
    for (const auto &spec : physical) {
      // Every recovered physical output restores its recorded color; one without it fails.
      requests.push_back({spec.identity, spec.color});
    }
    const bool colors_ok = set_colors_and_settle(io_, *desired, requests, observed) == display_config::color_change_result_e::verified;
    if (!extras.empty() || !colors_ok) {
      return false;
    }
    return io_.clear();
  }

  bool manager_t::clip_cursor_to_display(const snapshot_t &snapshot, std::wstring_view device_path) {
    const auto layout = inspect(snapshot);
    if (!layout) {
      return false;
    }
    const auto index = find_path(snapshot, device_path);
    if (!index) {
      return false;
    }
    const auto &source = snapshot.modes[source_index(snapshot.paths[*index])].sourceMode;
    return io_.cursor_clip(device_path, cursor_bounds_t {source.position.x, source.position.y, static_cast<int>(source.width), static_cast<int>(source.height)});
  }

  bool manager_t::reconcile_active_exclusive(std::wstring_view device_path) {
    auto loaded = io_.load();
    if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
      return false;
    }
    if (loaded.journal && loaded.journal->monitor_replacement) {
      return promote_monitor(device_path) && refresh_exclusive_cursor_clip(device_path);
    }
    if (!loaded.journal) {
      // A session that began truly headless needs no recovery journal. If a display appears
      // later, promotion captures that new baseline before disabling or preserving it.
      return promote_exclusive(device_path) && refresh_exclusive_cursor_clip(device_path);
    }
    auto journal = *loaded.journal;
    if (!journal.exclusive || !journal.exclusive_started || !same_device_path(journal.promoted_primary, device_path) || !journal.exclusive_topology || !journal.original_topology || !io_.query_all) {
      return false;
    }
    const auto current = io_.query();
    const auto virtual_index = current ? find_path(*current, device_path) : std::nullopt;
    const auto available = io_.query_all();
    if (!current || !virtual_index || !inspect(*current) || !available || available->paths.size() > max_available_paths || available->paths.size() != available->device_paths.size()) {
      return false;
    }

    auto is_available = [&](std::wstring_view identity) {
      for (size_t i = 0; i < available->paths.size(); ++i) {
        if (available->paths[i].targetInfo.targetAvailable && same_device_path(available->device_paths[i], identity)) {
          return true;
        }
      }
      return false;
    };
    std::vector<display_spec_t> recorded_hotplugs;
    auto upsert_hotplug = [&](display_spec_t spec) {
      const auto existing = std::ranges::find_if(recorded_hotplugs, [&](const auto &entry) {
        return same_device_path(entry.identity, spec.identity);
      });
      if (existing == recorded_hotplugs.end()) {
        recorded_hotplugs.push_back(std::move(spec));
      } else {
        *existing = std::move(spec);
      }
    };
    if (journal.pending_restore) {
      for (size_t i = 0; i < journal.pending_restore->paths.size(); ++i) {
        const auto &identity = journal.pending_restore->device_paths[i];
        if (!same_device_path(identity, device_path) && !find_path(*journal.original_topology, identity) && is_available(identity)) {
          upsert_hotplug(display_spec(*journal.pending_restore, i));
        }
      }
    }

    bool discovered_hotplug = false;
    for (size_t i = 0; i < current->paths.size(); ++i) {
      const auto &identity = current->device_paths[i];
      if (same_device_path(identity, device_path) || find_path(*journal.original_topology, identity)) {
        continue;
      }
      discovered_hotplug = true;
      upsert_hotplug(display_spec(*current, i));
      if (journal.local_sink.empty() && io_.preserve_exclusive && io_.preserve_exclusive(current->paths[i], identity) && !preserved_exclusive_identity(journal, identity)) {
        journal.exclusive_preserved.push_back(identity);
      }
    }
    if (!recorded_hotplugs.empty()) {
      // pending_restore is the durable catalog for outputs absent from the launch baseline.
      // If one of those outputs is no longer available while another appears, retire the stale
      // allowlist entry before replacing the catalog. A later reconnect is classified again.
      std::vector<std::wstring> retired_preserved;
      std::erase_if(journal.exclusive_preserved, [&](const auto &identity) {
        const bool retire = !find_path(*journal.original_topology, identity) &&
                            std::ranges::none_of(recorded_hotplugs, [&](const auto &recorded) {
                              return same_device_path(recorded.identity, identity);
                            });
        if (retire) {
          retired_preserved.push_back(identity);
        }
        return retire;
      });
      if (journal.exclusive_topology) {
        for (size_t i = journal.exclusive_topology->paths.size(); i-- > 0;) {
          if (std::ranges::none_of(retired_preserved, [&](const auto &identity) {
                return same_device_path(journal.exclusive_topology->device_paths[i], identity);
              })) {
            continue;
          }
          journal.exclusive_topology->paths.erase(journal.exclusive_topology->paths.begin() + static_cast<std::ptrdiff_t>(i));
          journal.exclusive_topology->device_paths.erase(journal.exclusive_topology->device_paths.begin() + static_cast<std::ptrdiff_t>(i));
          journal.exclusive_topology->colors.erase(journal.exclusive_topology->colors.begin() + static_cast<std::ptrdiff_t>(i));
        }
      }
    }
    std::ranges::sort(journal.exclusive_preserved, [](const auto &left, const auto &right) {
      return CompareStringOrdinal(left.c_str(), -1, right.c_str(), -1, TRUE) == CSTR_LESS_THAN;
    });

    if (!recorded_hotplugs.empty()) {
      std::ranges::sort(recorded_hotplugs, [](const auto &left, const auto &right) {
        return CompareStringOrdinal(left.identity.c_str(), -1, right.identity.c_str(), -1, TRUE) == CSTR_LESS_THAN;
      });
      auto virtual_spec = display_spec(*current, *virtual_index);
      virtual_spec.source.position = {0, 0};
      std::vector<display_spec_t> record;
      record.push_back(std::move(virtual_spec));
      std::int64_t x = record.front().source.width;
      for (auto &spec : recorded_hotplugs) {
        if (x > std::numeric_limits<LONG>::max() || x + spec.source.width > std::numeric_limits<LONG>::max()) {
          return false;
        }
        spec.source.position = {static_cast<LONG>(x), 0};
        x += spec.source.width;
        record.push_back(std::move(spec));
      }
      const auto pending = build_topology(*available, record);
      if (!pending) {
        return false;
      }
      journal.pending_restore = *pending;
    }
    if (!exclusive_continuation_with_recorded_outputs(*current, journal, device_path, true)) {
      return false;
    }
    if (discovered_hotplug) {
      journal.before_exclusive = *current;
    }
    if (!valid_journal(journal) || !io_.save(journal)) {
      return false;
    }
    return promote_exclusive(device_path, true) && refresh_exclusive_cursor_clip(device_path);
  }

  bool manager_t::recover_inactive_exclusive() {
    const auto loaded = io_.load();
    if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
      return false;
    }
    if (!loaded.journal || !loaded.journal->exclusive) {
      return true;
    }
    return restore();
  }

  bool manager_t::refresh_exclusive_cursor_clip(std::wstring_view device_path) {
    if (!io_.cursor_clip) {
      return true;
    }
    const auto loaded = io_.load();
    if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
      return false;
    }
    const auto snapshot = io_.query();
    if (!snapshot) {
      return false;
    }
    if (loaded.journal) {
      const auto &journal = *loaded.journal;
      if (journal.monitor_replacement) {
        return same_device_path(journal.promoted_primary, device_path) && io_.cursor_clip(device_path, std::nullopt);
      }
      if (!journal.exclusive || !journal.exclusive_started || !same_device_path(journal.promoted_primary, device_path) || !journal.exclusive_topology || !same_topology(*snapshot, *journal.exclusive_topology)) {
        return false;
      }
    } else if (snapshot->paths.size() != 1) {
      return false;
    }
    // With one active output Windows has nowhere else to move the shared cursor. Avoid taking
    // ownership of the process-global ClipCursor state until a preserved AR output is active.
    if (snapshot->paths.size() == 1) {
      return io_.cursor_clip(device_path, std::nullopt);
    }
    return clip_cursor_to_display(*snapshot, device_path);
  }

  bool manager_t::promote(std::wstring_view device_path, bool exclusive) {
    if (exclusive) {
      const bool promoted = promote_exclusive(device_path) && refresh_exclusive_cursor_clip(device_path);
      if (!promoted && io_.cursor_clip) {
        io_.cursor_clip(device_path, std::nullopt);
      }
      return promoted;
    }
    auto loaded = io_.load();
    if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
      return false;
    }
    if (loaded.journal && loaded.journal->exclusive) {
      return false;
    }
    if (loaded.journal && loaded.journal->prepared) {
      if (!bind_pending(device_path)) {
        return false;
      }
      loaded = io_.load();
      if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
        return false;
      }
    }
    auto snapshot = io_.query();
    auto layout = snapshot ? inspect(*snapshot) : std::nullopt;
    if (!layout) {
      return false;
    }
    if (loaded.journal) {
      // Keep the first baseline on repeated promotion (including a resized display).
      if (!same_device_path(loaded.journal->promoted_primary, device_path)) {
        return false;
      }
      if (same_layout(*layout, loaded.journal->promoted)) {
        return true;
      }
      if (!same_layout(*layout, loaded.journal->original)) {
        return false;
      }
      // A prepared baseline has already been bound. Keep that durable first baseline
      // instead of clearing it and taking a later snapshot after mode changes.
      return apply_verified(*snapshot, loaded.journal->promoted);
    }
    const auto *target = find_position(*layout, device_path);
    if (!target) {
      return false;
    }
    if (target->x == 0 && target->y == 0) {
      return true;
    }
    const auto original_primary = std::ranges::find_if(*layout, [](const auto &entry) {
      return entry.x == 0 && entry.y == 0;
    });
    auto desired = translated(*layout, target->x, target->y);
    if (!desired) {
      return false;
    }
    const journal_t journal {original_primary->device_path, target->device_path, *layout, *desired};
    if (!io_.save(journal)) {
      BOOST_LOG(warning) << "Could not durably save primary-display recovery record; display configuration was not changed.";
      return false;
    }
    if (!apply_verified(*snapshot, *desired)) {
      return false;
    }
    BOOST_LOG(info) << "Virtual display is temporarily the Windows primary display.";
    return true;
  }

  bool manager_t::restore(std::wstring_view expected_device_path) {
    // Cursor ownership is independent of journal readability. Release it even if display
    // recovery must wait; an old monitor identity cannot release a newer session's clip.
    if (io_.cursor_clip && !io_.cursor_clip(expected_device_path, std::nullopt)) {
      return false;
    }
    const auto loaded = io_.load();
    if (!loaded.success || (loaded.journal && !valid_journal(*loaded.journal))) {
      BOOST_LOG(warning) << "Could not read primary-display recovery record; leaving displays unchanged.";
      return false;
    }
    if (!loaded.journal) {
      return true;
    }
    const auto &journal = *loaded.journal;
    if (journal.exclusive_started) {
      if (!expected_device_path.empty() && !same_device_path(expected_device_path, journal.promoted_primary)) {
        return false;
      }
      return restore_exclusive(journal);
    }
    if (journal.prepared) {
      const auto snapshot = io_.query();
      const auto current = snapshot ? inspect(*snapshot) : std::nullopt;
      if (!expected_device_path.empty() && current && find_position(*current, expected_device_path)) {
        return bind_pending(expected_device_path) && restore(expected_device_path);
      }
      if (journal.exclusive) {
        return recover_prepared_outputs(journal);
      }
      const auto *primary = current ? find_position(*current, journal.original_primary) : nullptr;
      if (!primary || current->size() != journal.original.size()) {
        return journal.exclusive && recover_prepared_outputs(journal);
      }
      const auto rebased = translated(*current, primary->x, primary->y);
      if (!rebased || !same_layout(*rebased, journal.original)) {
        return journal.exclusive && recover_prepared_outputs(journal);
      }
      // If a crash occurred before binding the target, wait for the driver's orphan
      // cleanup. Only the exact known physical set may be restored without that identity.
      if (!same_layout(*current, journal.original) && !apply_verified(*snapshot, journal.original)) {
        return false;
      }
      return io_.clear();
    }
    if (!expected_device_path.empty() && !same_device_path(expected_device_path, journal.promoted_primary)) {
      return false;
    }
    const auto snapshot = io_.query();
    const auto layout = snapshot ? inspect(*snapshot) : std::nullopt;
    if (!layout) {
      return false;
    }
    const auto original = restrict_layout(journal.original, *layout, journal.promoted_primary);
    const auto promoted = restrict_layout(journal.promoted, *layout, journal.promoted_primary);
    const auto preserve_current_desktop = [&]() {
      if (std::ranges::none_of(*layout, [&](const auto &position) {
            return !same_device_path(position.device_path, journal.promoted_primary);
          })) {
        // Keep recovery when the virtual source is the only usable desktop.
        return false;
      }
      const auto verified = io_.query();
      if (!verified || !same_topology(*snapshot, *verified)) {
        return false;
      }
      BOOST_LOG(info) << "Preserving the current physical desktop and retiring the superseded primary-display recovery record.";
      return io_.clear();
    };
    if (!original || !promoted) {
      return preserve_current_desktop();
    }
    if (same_layout(*layout, *original)) {
      return io_.clear();
    }
    if (!find_position(*layout, journal.promoted_primary)) {
      const auto *primary = find_position(*layout, journal.original_primary);
      const auto rebased = primary ? translated(*layout, primary->x, primary->y) : std::nullopt;
      if (rebased && same_layout(*rebased, *original)) {
        return apply_verified(*snapshot, *original) && io_.clear();
      }
    }
    if (!same_layout(*layout, *promoted)) {
      return preserve_current_desktop();
    }
    if (!apply_verified(*snapshot, *original)) {
      return false;
    }
    if (!io_.clear()) {
      return false;
    }
    BOOST_LOG(info) << "Restored the original Windows primary display and monitor positions.";
    return true;
  }
}  // namespace platf::primary_display::detail
