// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_mask.h"

#include <windows.h>
#include <array>
#include <algorithm>
#include <utility>

namespace sunshine_game3d::ui_mask {
  namespace {
    namespace capture = sunshine_streamline::depth_capture;
    constexpr unsigned runtime_capacity = 4, source_count = 4, snapshots_per_source = 2;
    constexpr unsigned snapshot_capacity = source_count * snapshots_per_source;
    struct slot {
      std::uint64_t reservation{};
      boundary origin;
      capture::diagnostic_ticket ticket;
      capture::source_ref source;
    };
    struct entry {
      request wanted;
      std::uint64_t generation{}, latest_sequence{};
      unsigned latest_kinds{};
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
      capture_gate_observation hook_gate;
      std::array<entry, runtime_capacity> entries;
      std::array<std::uint64_t, std::size_t(begin_stage::count)> begin_refusals{};
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
      value.latest_kinds = 0;
      value.generation = ++owner().serial;
      value.diagnostic = {};
      value.diagnostic_reservation = 0;
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
      value.last_attempt = {};
#endif
    }
    bool same(const request &a, const request &b) {
      return a.runtime == b.runtime && a.device_identity == b.device_identity && a.epoch == b.epoch && a.revision == b.revision &&
        a.viewport == b.viewport && a.width == b.width && a.height == b.height && a.enabled == b.enabled &&
        a.allowed_kinds == b.allowed_kinds;
    }
    bool matches(const request &a, std::uint64_t epoch, std::uint64_t revision, std::uint32_t viewport) {
      return a.enabled && a.epoch == epoch && a.revision == revision && a.viewport == viewport;
    }
    bool recent(std::uint64_t tick, std::uint64_t now) {
      return tick && now >= tick && now - tick < sunshine_scene_depth::maximum_source_age_ms;
    }
    unsigned priority(source_kind kind) {
      return kind == source_kind::alpha ? 4u : kind == source_kind::color_and_alpha ? 3u :
        kind == source_kind::backbuffer ? 2u : kind == source_kind::hudless ? 1u : 0u;
    }
    bool usable(const request &wanted, source_kind kind, const capture::diagnostic_texture &value) {
      return value.device_identity == wanted.device_identity && value.width == wanted.width && value.height == wanted.height &&
        !value.area.left && !value.area.top && value.area.width == wanted.width && value.area.height == wanted.height &&
        supported_format(kind, value.format);
    }
    struct observed_slot {
      slot value;
      capture::diagnostic_texture texture;
      capture::status status{capture::status::unavailable};
      std::uint64_t current_present_generation{};
    };
    // Poll outside our lock, then commit only against the identical generation
    // and reservation. Producers/consumers can never revive a revoked scope.
    bool collect(std::uint64_t runtime, std::uint64_t now, selection *out, std::uint32_t allowed = all_sources,
        std::uint64_t consumer_queue = 0) {
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
      std::array<std::uint64_t, source_count> newest_ready{};
      unsigned best_priority{};
      for (auto &item : observed) if (item.value.ticket && recent(item.value.origin.source.tick, now)) {
        item.status = consumer_queue ? capture::acquire_local_texture(item.value.ticket, consumer_queue, item.texture) :
          capture::acquire_diagnostic_texture(item.value.ticket, item.texture);
        item.current_present_generation = capture::source_present_generation(item.value.source);
        if (item.status == capture::status::ready && usable(wanted, item.value.origin.kind, item.texture)) {
          const auto rank = priority(item.value.origin.kind);
          if (!rank || rank > source_count) continue;
          newest_ready[rank - 1] = std::max(newest_ready[rank - 1], item.value.origin.source.sequence);
          if (allowed & source_mask(item.value.origin.kind)) best_priority = std::max(best_priority, rank);
        }
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
          // merely to admit another frame when its two source slots are occupied.
          if (!stored.ticket || !checked.value.ticket) continue;
          // Older SDK callbacks can arrive after a newer one. Their entry tick
          // cannot expire a future snapshot; acquire likewise cannot expose it.
          if (now < stored.origin.source.tick) continue;
          const auto rank = priority(stored.origin.kind);
          if (!rank || rank > source_count) { retire(stored, old, count); continue; }
          const bool ready = checked.status == capture::status::ready && usable(wanted, stored.origin.kind, checked.texture);
          const bool pending = checked.status == capture::status::recorded || checked.status == capture::status::submitted;
          if (!recent(stored.origin.source.tick, now) || (!ready && !pending) ||
              (ready && stored.origin.source.sequence < newest_ready[rank - 1])) {
            retire(stored, old, count);
          } else if (ready && rank == best_priority && stored.origin.source.sequence == newest_ready[rank - 1] && out) {
            out->ticket = stored.ticket; out->texture = checked.texture; out->origin = stored.origin;
            out->current_source_present_generation = checked.current_present_generation;
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
    wanted.allowed_kinds &= all_sources;
    wanted.enabled = wanted.enabled && wanted.device_identity && wanted.epoch && wanted.width && wanted.height &&
      wanted.allowed_kinds;
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
    state.hook_gate = {};
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

  bool acquire_kind(std::uint64_t runtime, source_kind kind, selection &out, std::uint64_t now_ms, std::uint64_t consumer_queue) {
    out = {};
    return source_mask(kind) && collect(runtime, now_ms, &out, source_mask(kind), consumer_queue);
  }

  bool query_diagnostic(std::uint64_t runtime, diagnostic_snapshot &out) {
    out = {};
    auto &state = owner();
    bool found = false;
    AcquireSRWLockShared(&state.lock);
    for (const auto &item : state.entries) if (item.wanted.runtime == runtime && item.wanted.enabled) {
      out = item.diagnostic;
      out.wanted = item.wanted;
      out.hook_gate = state.hook_gate;
      out.request_generation = item.generation;
      out.begin_refusals = state.begin_refusals;
      found = true;
      break;
    }
    ReleaseSRWLockShared(&state.lock);
    return found;
  }

  bool interested(std::uint64_t epoch, std::uint64_t revision, std::uint32_t viewport,
      std::uint32_t *matching_requests) {
    auto &state = owner();
    unsigned matches_count = 0;
    AcquireSRWLockShared(&state.lock);
    for (const auto &item : state.entries) if (matches(item.wanted, epoch, revision, viewport)) ++matches_count;
    ReleaseSRWLockShared(&state.lock);
    if (matching_requests) *matching_requests = matches_count;
    // Two runtimes claiming the same viewport cannot safely share a ticket:
    // each ticket binds one destination/queue. Do not guess the active consumer.
    return matches_count == 1;
  }

  void observe_gate(const capture_gate_observation &value) {
    auto &state = owner();
    AcquireSRWLockExclusive(&state.lock);
    // SDK calls can finish out of order; telemetry must not move backwards.
    if (value.sequence >= state.hook_gate.sequence) state.hook_gate = value;
    ReleaseSRWLockExclusive(&state.lock);
  }

  attempt begin(const boundary &where, const capture::input &input) {
    auto &state = owner();
    const auto &source = where.source;
    // Every return without an attempt names its stage (the counts, and the
    // request's latest refusal when the request is known).
    const auto refuse = [&](begin_stage stage, std::uint64_t refused_runtime = 0) {
      AcquireSRWLockExclusive(&state.lock);
      ++state.begin_refusals[std::size_t(stage)];
      if (refused_runtime) for (auto &item : state.entries) if (item.wanted.runtime == refused_runtime) {
        item.diagnostic.begin_refusal = stage;
        break;
      }
      ReleaseSRWLockExclusive(&state.lock);
      return attempt{};
    };
    const auto rank = priority(where.kind);
    if (!rank) return refuse(begin_stage::kind_filtered);
    const auto kind_bit = 1u << rank;
    const auto newer = [&](const entry &item) {
      return source.sequence > item.latest_sequence ||
        (source.sequence == item.latest_sequence && !(item.latest_kinds & kind_bit));
    };
    std::uint64_t runtime{}, filtered_runtime{}, stale_runtime{};
    unsigned matches_count = 0;
    AcquireSRWLockShared(&state.lock);
    for (const auto &item : state.entries) if (matches(item.wanted, source.epoch, source.observation_revision, source.viewport)) {
      if (!(item.wanted.allowed_kinds & source_mask(where.kind))) { filtered_runtime = item.wanted.runtime; continue; }
      if (!newer(item)) { stale_runtime = item.wanted.runtime; break; }
      runtime = item.wanted.runtime; ++matches_count;
    }
    ReleaseSRWLockShared(&state.lock);
    if (stale_runtime) return refuse(begin_stage::not_newer, stale_runtime);
    if (!matches_count && filtered_runtime) return refuse(begin_stage::kind_filtered, filtered_runtime);
    if (matches_count != 1) return refuse(matches_count ? begin_stage::ambiguous_request : begin_stage::no_request);
    if (!source.sequence) return refuse(begin_stage::shape, runtime);
    collect(runtime, source.tick, nullptr);
    retired old; unsigned count = 0;
    attempt result;
    auto stage = begin_stage::no_request;
    AcquireSRWLockExclusive(&state.lock);
    for (auto &item : state.entries) if (item.wanted.runtime == runtime &&
        matches(item.wanted, source.epoch, source.observation_revision, source.viewport) && newer(item)) {
      if (!(item.wanted.allowed_kinds & source_mask(where.kind))) { stage = begin_stage::kind_filtered; break; }
      if (source.sequence != item.latest_sequence) item.latest_kinds = 0;
      item.latest_sequence = source.sequence;
      item.latest_kinds |= kind_bit;
      item.diagnostic = {};
      item.diagnostic.wanted = item.wanted;
      item.diagnostic.latest_boundary = where;
      item.diagnostic_reservation = 0;
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
      item.last_attempt = where;
#endif
      result.runtime = runtime; result.generation = item.generation; result.sequence = source.sequence;
      result.kind = where.kind;
      const auto &area = source.resource.area;
      const bool shape = (!source.resource.width || source.resource.width == item.wanted.width) &&
        (!source.resource.height || source.resource.height == item.wanted.height) && !area.left && !area.top &&
        (!area.width || area.width == item.wanted.width) && (!area.height || area.height == item.wanted.height);
      if (!where.command || !source.resource.native || !input.source || !shape ||
          source.valid_until == sunshine_scene_depth::lifetime::unsupported) {
        for (auto &snapshot : item.snapshots) if (snapshot.origin.kind == where.kind) retire(snapshot, old, count);
        stage = source.valid_until == sunshine_scene_depth::lifetime::unsupported ? begin_stage::unsupported_lifetime :
          begin_stage::shape;
      } else {
        stage = begin_stage::no_reservation;
        // Two reservations per semantic source prevent a full-scene alpha
        // candidate from starving HUD-less comparison. The shared native owner
        // additionally enforces its global snapshot and allocation-byte limits.
        const auto begin = (rank - 1) * snapshots_per_source;
        for (unsigned i = begin; i != begin + snapshots_per_source; ++i) if (!item.snapshots[i].reservation) {
          auto &snapshot = item.snapshots[i];
          result.reservation = ++state.serial;
          snapshot.reservation = result.reservation; snapshot.origin = where;
          snapshot.origin.source_present_generation = input.source_present_generation;
          snapshot.source = input.source;
          item.diagnostic_reservation = result.reservation;
          item.diagnostic.record_attempted = true;
          stage = begin_stage::none;
          break;
        }
      }
      if (stage != begin_stage::none) item.diagnostic.begin_refusal = stage;
      break;
    }
    if (stage != begin_stage::none) ++state.begin_refusals[std::size_t(stage)];
    ReleaseSRWLockExclusive(&state.lock);
    release(old);
    if (!result.reservation) return result;
    capture::record_diagnostic record;
    // OnlyValidNow can use current recording evidence over a stale hint.
    // Longer-lived tags retain their declared lifetime and strict state contract.
    auto ticket = capture::record_local_texture(where.command, input, &record, capture::auxiliary_state_policy(input));
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
      if (item.latest_sequence == value.sequence && item.diagnostic.latest_boundary.kind == value.kind &&
          item.diagnostic_reservation == value.reservation) {
        item.diagnostic.sdk_result_known = true;
        item.diagnostic.sdk_successful = successful;
      }
      // A failed SDK call revokes this and older real inputs, but cannot erase
      // a newer successful call that completed on another application thread.
      if (!successful) {
        for (auto &snapshot : item.snapshots)
          if (snapshot.origin.kind == value.kind && snapshot.origin.source.sequence <= value.sequence) retire(snapshot, old, count);
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
