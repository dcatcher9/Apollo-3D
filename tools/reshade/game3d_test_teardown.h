// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// TEST add-on only (SunshineGame3DTestTeardownOrder): the order in which the
// exporter's owners go, for the add-on lifecycle fixture
// (test_addon_unload_runtime.cpp), which writes each unload's sequence. The
// publisher's destructor (exporter.cpp ~publisher_t, run by every
// AddonUninit) records its body's steps; the destructors of the export rings
// (generation_t), the renderers, the diagnostic mailbox (debug_dump) and the
// publisher's mapping handle (handle_t) record themselves. A structure move
// of the exporter must leave the sequence unchanged. One fixed ring
// (game3d_test_event_ring.h); nothing allocates. The production add-on
// records and exports nothing.
#include "game3d_test_event_ring.h"

#include <cstdint>

namespace sunshine_game3d::test::teardown {
  // What went, or what the publisher's destructor did. The values are stable
  // record ids.
  enum class step : std::uint8_t {
    none = 0,
    publisher_begin = 1,  // ~publisher_t entered; detail 1: the add-on is stopping, and the body skips its cleanup.
    ui_cleared = 2,  // Its automatic UI entries cleared.
    identity_published = 3,  // Its identity metadata published to the shared state.
    preserved = 4,  // A ring's in-flight allocations preserved (preserve_inflight_on_unload); detail: the ring's owner.
    unmapped = 5,  // Its shared state unmapped.
    publisher_end = 6,  // Its body finished; its members go next, the last declared first.
    generation = 7,  // ~generation_t; detail: the ring's owner.
    parked_renderer = 8,  // ~renderer of a renderer parked across a runtime reset.
    runtime_renderer = 9,  // ~renderer of any other renderer (a runtime's).
    debug_dump = 10,  // ~debug_dump, the diagnostic mailbox.
    mapping = 11,  // ~handle_t of the publisher's mapping; detail 1: it closed a handle.
  };

  // The publisher member that held a ring when the publisher last named its
  // rings (before it releases or replaces one, and at the end of its
  // destructor's body); unowned for any other ring.
  enum class ring_owner : std::uint8_t {
    unowned = 0,
    active = 1,  // generation_
    retired = 2,  // retired_
    cached = 3,  // cached_ring_
  };

  // One record; detail is step's (a ring_owner for preserved and generation).
  struct event {
    std::uint64_t serial;  // 1 for the first event since the add-on loaded.
    step what;
    std::uint8_t detail;
    std::uint8_t reserved[6];
  };
  static_assert(sizeof(event) == 16);

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  namespace detail {
    inline constexpr unsigned capacity = 256;
    inline constexpr unsigned parked_capacity = 16;

    struct ring_note {
      const void *ring;
      ring_owner owner;
    };

    struct state_t {
      event_ring<event, capacity> ring;
      ring_note rings[3];  // The publisher's rings at its last note_rings.
      const void *mapping;  // The publisher's mapping handle, from the end of its destructor's body.
      const void *parked[parked_capacity];  // The renderers parked now.
    };

    inline state_t state;

    // Requires state.ring.lock, exclusive.
    inline void append_locked(step what, std::uint8_t value) {
      auto &event = state.ring.append_locked();
      event.what = what;
      event.detail = value;
    }
  }  // namespace detail

  inline void note(step what, std::uint8_t value = 0) {
    AcquireSRWLockExclusive(&detail::state.ring.lock);
    detail::append_locked(what, value);
    ReleaseSRWLockExclusive(&detail::state.ring.lock);
  }

  inline void note(step what, ring_owner owner) {
    note(what, static_cast<std::uint8_t>(owner));
  }

  // The publisher's rings now: generation_, retired_ and cached_ring_.
  inline void note_rings(const void *active, const void *retired, const void *cached) {
    AcquireSRWLockExclusive(&detail::state.ring.lock);
    detail::state.rings[0] = {active, ring_owner::active};
    detail::state.rings[1] = {retired, ring_owner::retired};
    detail::state.rings[2] = {cached, ring_owner::cached};
    ReleaseSRWLockExclusive(&detail::state.ring.lock);
  }

  inline void generation_destroyed(const void *ring) {
    AcquireSRWLockExclusive(&detail::state.ring.lock);
    auto owner = ring_owner::unowned;
    for (auto &entry : detail::state.rings) {
      if (entry.ring == ring) {
        owner = entry.owner;
        entry.ring = nullptr;
        break;
      }
    }
    detail::append_locked(step::generation, static_cast<std::uint8_t>(owner));
    ReleaseSRWLockExclusive(&detail::state.ring.lock);
  }

  // The publisher's mapping_, named at the end of its destructor's body.
  inline void note_mapping(const void *handle) {
    AcquireSRWLockExclusive(&detail::state.ring.lock);
    detail::state.mapping = handle;
    ReleaseSRWLockExclusive(&detail::state.ring.lock);
  }

  // Every handle_t; only the named mapping records.
  inline void handle_destroyed(const void *handle, bool closes) {
    AcquireSRWLockExclusive(&detail::state.ring.lock);
    if (handle == detail::state.mapping) {
      detail::state.mapping = nullptr;
      detail::append_locked(step::mapping, closes ? 1 : 0);
    }
    ReleaseSRWLockExclusive(&detail::state.ring.lock);
  }

  // A renderer parked across a runtime reset (parked true), or taken back
  // by its runtime (false).
  inline void note_parked(const void *renderer, bool parked) {
    AcquireSRWLockExclusive(&detail::state.ring.lock);
    for (auto &entry : detail::state.parked) {
      if (parked ? entry == nullptr : entry == renderer) {
        entry = parked ? renderer : nullptr;
        break;
      }
    }
    ReleaseSRWLockExclusive(&detail::state.ring.lock);
  }

  inline void renderer_destroyed(const void *renderer) {
    AcquireSRWLockExclusive(&detail::state.ring.lock);
    auto what = step::runtime_renderer;
    for (auto &entry : detail::state.parked) {
      if (entry == renderer) {
        entry = nullptr;
        what = step::parked_renderer;
        break;
      }
    }
    detail::append_locked(what, 0);
    ReleaseSRWLockExclusive(&detail::state.ring.lock);
  }

  // SunshineGame3DTestTeardownOrder (event_ring::read).
  inline unsigned read(std::uint64_t &cursor, event *out, unsigned count, std::uint64_t &skipped) {
    return detail::state.ring.read(cursor, out, count, skipped);
  }
#endif
}  // namespace sunshine_game3d::test::teardown
