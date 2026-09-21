// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_mask.h"

#include <windows.h>
#include <array>
#include <algorithm>
#include <utility>

namespace sunshine_game3d::ui_mask {
  namespace {
    namespace capture = sunshine_streamline::depth_capture;
    constexpr unsigned runtime_capacity = 4, snapshot_capacity = 3;
    struct slot {
      std::uint64_t reservation{};
      boundary origin;
      capture::diagnostic_ticket ticket;
    };
    struct entry {
      request wanted;
      std::uint64_t generation{}, latest_sequence{};
      std::array<slot, snapshot_capacity> snapshots;
      diagnostic_snapshot diagnostic;
      std::uint64_t diagnostic_reservation{};
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
      boundary last_attempt;
#endif
    };
    struct state {
      SRWLOCK lock = SRWLOCK_INIT;
      std::uint64_t serial{};
      std::array<entry, runtime_capacity> entries;
    };
    state &owner() {
      // The add-on is pinned and its SDK detours can outlive AddonUninit. Keep
      // the lock alive; explicit invalidation releases all native pixel leases.
      static auto *value = new state;
      return *value;
    }
    using retired = std::array<capture::diagnostic_ticket, runtime_capacity * snapshot_capacity>;
    void release(retired &values) {
      // Native release/COM callbacks must never run under the owner lock.
      for (auto &value : values) if (value) capture::release_diagnostic_texture(value);
    }
    void retire(slot &value, retired &values, unsigned &count) {
      if (value.ticket) values[count++] = std::move(value.ticket);
      value = {};
    }
    void clear(entry &value, retired &values, unsigned &count) {
      for (auto &item : value.snapshots) retire(item, values, count);
      value.latest_sequence = 0;
      value.generation = ++owner().serial;
      value.diagnostic = {};
      value.diagnostic_reservation = 0;
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
      value.last_attempt = {};
#endif
    }
    bool same(const request &a, const request &b) {
      return a.runtime == b.runtime && a.device_identity == b.device_identity && a.epoch == b.epoch && a.revision == b.revision &&
        a.viewport == b.viewport && a.width == b.width && a.height == b.height && a.enabled == b.enabled;
    }
    bool matches(const request &a, std::uint64_t epoch, std::uint64_t revision, std::uint32_t viewport) {
      return a.enabled && a.epoch == epoch && a.revision == revision && a.viewport == viewport;
    }
    bool recent(std::uint64_t tick, std::uint64_t now) {
      return tick && now >= tick && now - tick < sunshine_scene_depth::maximum_source_age_ms;
    }
    bool alpha_format(std::uint32_t format) {
      // Typed DXGI color formats with a real alpha channel, not BGRX padding.
      return format == 2 || format == 10 || format == 24 || format == 28 || format == 29 || format == 87 || format == 91;
    }
    bool usable(const request &wanted, const capture::diagnostic_texture &value) {
      return value.device_identity == wanted.device_identity && value.width == wanted.width && value.height == wanted.height &&
        !value.area.left && !value.area.top && value.area.width == wanted.width && value.area.height == wanted.height &&
        alpha_format(value.format);
    }
    struct observed_slot {
      slot value;
      capture::diagnostic_texture texture;
      capture::status status{capture::status::unavailable};
    };
    // Poll outside our lock, then commit only against the identical generation
    // and reservation. Producers/consumers can never revive a revoked scope.
    bool collect(std::uint64_t runtime, std::uint64_t now, selection *out) {
      auto &state = owner();
      std::array<observed_slot, snapshot_capacity> observed;
      request wanted;
      std::uint64_t generation{};
      AcquireSRWLockShared(&state.lock);
      for (const auto &item : state.entries) if (item.wanted.runtime == runtime && item.wanted.enabled) {
        wanted = item.wanted; generation = item.generation;
        for (unsigned i = 0; i != snapshot_capacity; ++i) observed[i].value = item.snapshots[i];
        break;
      }
      ReleaseSRWLockShared(&state.lock);
      if (!wanted.enabled) return false;
      std::uint64_t newest_ready{};
      for (auto &item : observed) if (item.value.ticket && recent(item.value.origin.source.tick, now)) {
        item.status = capture::acquire_diagnostic_texture(item.value.ticket, item.texture);
        if (item.status == capture::status::ready && usable(wanted, item.texture))
          newest_ready = std::max(newest_ready, item.value.origin.source.sequence);
      }
      retired old; unsigned count = 0;
      bool found = false;
      AcquireSRWLockExclusive(&state.lock);
      for (auto &item : state.entries) if (item.wanted.runtime == runtime && item.generation == generation) {
        for (unsigned i = 0; i != snapshot_capacity; ++i) {
          auto &stored = item.snapshots[i];
          const auto &checked = observed[i];
          if (!stored.reservation || stored.reservation != checked.value.reservation) continue;
          // A record call is still reserving its slot. Never evict pending work
          // merely to admit another frame when all three slots are occupied.
          if (!stored.ticket || !checked.value.ticket) continue;
          // Older SDK callbacks can arrive after a newer one. Their entry tick
          // cannot expire a future snapshot; acquire likewise cannot expose it.
          if (now < stored.origin.source.tick) continue;
          const bool ready = checked.status == capture::status::ready && usable(wanted, checked.texture);
          const bool pending = checked.status == capture::status::recorded || checked.status == capture::status::submitted;
          if (!recent(stored.origin.source.tick, now) || (!ready && !pending) ||
              (ready && stored.origin.source.sequence < newest_ready)) {
            retire(stored, old, count);
          } else if (ready && stored.origin.source.sequence == newest_ready && out) {
            out->ticket = stored.ticket; out->texture = checked.texture; out->origin = stored.origin;
            found = true;
          }
        }
        break;
      }
      ReleaseSRWLockExclusive(&state.lock);
      release(old);
      return found;
    }
  }

  void set_request(const request &value) {
    if (!value.runtime) return;
    auto wanted = value;
    wanted.enabled = wanted.enabled && wanted.device_identity && wanted.epoch && wanted.width && wanted.height;
    auto &state = owner();
    retired old; unsigned count = 0;
    AcquireSRWLockExclusive(&state.lock);
    entry *selected = nullptr;
    for (auto &item : state.entries) {
      if (item.wanted.runtime == wanted.runtime) { selected = &item; break; }
      if (!selected && !item.wanted.runtime) selected = &item;
    }
    if (selected && !same(selected->wanted, wanted)) {
      clear(*selected, old, count);
      selected->wanted = wanted.enabled ? wanted : request{};
    }
    ReleaseSRWLockExclusive(&state.lock);
    release(old);
  }

  void invalidate(std::uint64_t runtime) {
    auto &state = owner();
    retired old; unsigned count = 0;
    AcquireSRWLockExclusive(&state.lock);
    for (auto &item : state.entries) if (item.wanted.runtime == runtime) {
      clear(item, old, count); item.wanted = {};
    }
    ReleaseSRWLockExclusive(&state.lock);
    release(old);
  }

  void invalidate_all() {
    auto &state = owner();
    retired old; unsigned count = 0;
    AcquireSRWLockExclusive(&state.lock);
    for (auto &item : state.entries) { clear(item, old, count); item.wanted = {}; }
    ReleaseSRWLockExclusive(&state.lock);
    release(old);
  }

  void invalidate_scope(std::uint64_t epoch, std::uint32_t viewport) {
    auto &state = owner();
    retired old; unsigned count = 0;
    AcquireSRWLockExclusive(&state.lock);
    for (auto &item : state.entries) if (item.wanted.epoch == epoch && item.wanted.viewport == viewport) {
      clear(item, old, count); item.wanted = {};
    }
    ReleaseSRWLockExclusive(&state.lock);
    release(old);
  }

  bool acquire(std::uint64_t runtime, selection &out, std::uint64_t now_ms) {
    out = {};
    return collect(runtime, now_ms, &out);
  }

  bool query_diagnostic(std::uint64_t runtime, diagnostic_snapshot &out) {
    out = {};
    auto &state = owner();
    bool found = false;
    AcquireSRWLockShared(&state.lock);
    for (const auto &item : state.entries) if (item.wanted.runtime == runtime && item.wanted.enabled) {
      out = item.diagnostic;
      out.wanted = item.wanted;
      found = true;
      break;
    }
    ReleaseSRWLockShared(&state.lock);
    return found;
  }

  bool interested(std::uint64_t epoch, std::uint64_t revision, std::uint32_t viewport) {
    auto &state = owner();
    unsigned matches_count = 0;
    AcquireSRWLockShared(&state.lock);
    for (const auto &item : state.entries) if (matches(item.wanted, epoch, revision, viewport)) ++matches_count;
    ReleaseSRWLockShared(&state.lock);
    // Two runtimes claiming the same viewport cannot safely share a ticket:
    // each ticket binds one destination/queue. Do not guess the active consumer.
    return matches_count == 1;
  }

  attempt begin(const boundary &where, const capture::input &input) {
    auto &state = owner();
    const auto &source = where.source;
    std::uint64_t runtime{};
    unsigned matches_count = 0;
    AcquireSRWLockShared(&state.lock);
    for (const auto &item : state.entries) if (matches(item.wanted, source.epoch, source.observation_revision, source.viewport)) {
      if (source.sequence <= item.latest_sequence) { ReleaseSRWLockShared(&state.lock); return {}; }
      runtime = item.wanted.runtime; ++matches_count;
    }
    ReleaseSRWLockShared(&state.lock);
    if (matches_count != 1 || !source.sequence) return {};
    collect(runtime, source.tick, nullptr);
    retired old; unsigned count = 0;
    attempt result;
    AcquireSRWLockExclusive(&state.lock);
    for (auto &item : state.entries) if (item.wanted.runtime == runtime &&
        matches(item.wanted, source.epoch, source.observation_revision, source.viewport) && source.sequence > item.latest_sequence) {
      item.latest_sequence = source.sequence;
      item.diagnostic = {};
      item.diagnostic.wanted = item.wanted;
      item.diagnostic.latest_boundary = where;
      item.diagnostic_reservation = 0;
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
      item.last_attempt = where;
#endif
      result.runtime = runtime; result.generation = item.generation; result.sequence = source.sequence;
      const auto &area = source.resource.area;
      const bool shape = (!source.resource.width || source.resource.width == item.wanted.width) &&
        (!source.resource.height || source.resource.height == item.wanted.height) && !area.left && !area.top &&
        (!area.width || area.width == item.wanted.width) && (!area.height || area.height == item.wanted.height);
      if (!where.command || !source.resource.native || !input.source || !shape ||
          source.valid_until == sunshine_scene_depth::lifetime::unsupported) {
        for (auto &snapshot : item.snapshots) retire(snapshot, old, count);
      } else {
        for (auto &snapshot : item.snapshots) if (!snapshot.reservation) {
          result.reservation = ++state.serial;
          snapshot.reservation = result.reservation; snapshot.origin = where;
          item.diagnostic_reservation = result.reservation;
          item.diagnostic.record_attempted = true;
          break;
        }
      }
      break;
    }
    ReleaseSRWLockExclusive(&state.lock);
    release(old);
    if (!result.reservation) return result;
    capture::record_diagnostic record;
    // A synchronous pre-FG snapshot runs at this command's current boundary.
    // Prefer current recording evidence over a stale tag hint. If this recording
    // contains no state observation, the capture owner validates the explicitly
    // provided state under the ordinary source contract; never invent a state.
    auto ticket = capture::record_local_texture(where.command, input, &record,
      capture::local_texture_state_policy::prefer_observed_recording);
    bool retained = false;
    AcquireSRWLockExclusive(&state.lock);
    for (auto &item : state.entries) if (item.wanted.runtime == runtime && item.generation == result.generation) {
      for (auto &snapshot : item.snapshots) if (snapshot.reservation == result.reservation) {
        if (item.latest_sequence == result.sequence && item.diagnostic_reservation == result.reservation) {
          item.diagnostic.record = record;
          item.diagnostic.record_completed = true;
        }
        snapshot.ticket = ticket;
        if (!ticket) snapshot = {};
        retained = true;
        break;
      }
    }
    ReleaseSRWLockExclusive(&state.lock);
    if (retained) result.ticket = std::move(ticket);
    else if (ticket) {
      capture::finish_diagnostic_texture(ticket, false);
      capture::release_diagnostic_texture(ticket);
    }
    return result;
  }

  void finish(const attempt &value, bool successful) {
    if (!value.runtime) return;
    auto &state = owner();
    retired old; unsigned count = 0;
    bool current = false;
    AcquireSRWLockExclusive(&state.lock);
    for (auto &item : state.entries) if (item.wanted.runtime == value.runtime && item.generation == value.generation) {
      if (item.latest_sequence == value.sequence && item.diagnostic_reservation == value.reservation) {
        item.diagnostic.sdk_result_known = true;
        item.diagnostic.sdk_successful = successful;
      }
      // A failed SDK call revokes this and older real inputs, but cannot erase
      // a newer successful call that completed on another application thread.
      if (!successful) {
        for (auto &snapshot : item.snapshots)
          if (snapshot.origin.source.sequence <= value.sequence) retire(snapshot, old, count);
      } else {
        for (const auto &snapshot : item.snapshots)
          if (snapshot.reservation == value.reservation && snapshot.ticket.id == value.ticket.id) current = true;
      }
      break;
    }
    ReleaseSRWLockExclusive(&state.lock);
    if (value.ticket) capture::finish_diagnostic_texture(value.ticket, successful && current);
    release(old);
  }
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing {
    bool last_attempt(std::uint64_t runtime, boundary &out) {
      out = {};
      auto &state = owner();
      AcquireSRWLockShared(&state.lock);
      for (const auto &item : state.entries) if (item.wanted.runtime == runtime) { out = item.last_attempt; break; }
      ReleaseSRWLockShared(&state.lock);
      return out.source.sequence != 0;
    }
  }
#endif
}
