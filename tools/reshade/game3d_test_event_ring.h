// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// TEST add-on only: the fixed event ring behind the test add-on's ordered
// event records (game3d_test_teardown.h). Events are POD records numbered
// from 1 since the add-on loaded; a reader passes the serial of the last
// event it read and gets the ones after it, oldest first. Nothing allocates:
// the ring is static storage, zero-initialized (SRWLOCK_INIT is zero), and
// its lock is taken by no other code. The production add-on has no ring.
#include <cstdint>
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  #include <Windows.h>
#endif

namespace sunshine_game3d::test {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  // Event is a POD record whose first member is std::uint64_t serial. A
  // namespace-scope ring is zero-initialized and needs no constructor.
  template<class Event, unsigned Capacity>
  struct event_ring {
    SRWLOCK lock;
    std::uint64_t newest;  // The serial of the newest event; 0 before the first.
    Event events[Capacity];

    // Requires lock, exclusive: the next event, cleared and numbered.
    Event &append_locked() {
      auto &event = events[newest % Capacity];
      event = Event {};
      event.serial = ++newest;
      return event;
    }

    // The events after cursor, at most count of them, oldest first; cursor
    // becomes the serial of the last one copied. skipped: the events after
    // the old cursor that the ring overwrote before this read (cursor then
    // moves past them).
    unsigned read(std::uint64_t &cursor, Event *out, unsigned count, std::uint64_t &skipped) {
      AcquireSRWLockShared(&lock);
      const auto last = newest;
      const auto oldest = last > Capacity ? last - Capacity + 1 : 1;
      auto next = cursor + 1;
      skipped = 0;
      if (next < oldest) {
        skipped = oldest - next;
        next = oldest;
        cursor = next - 1;
      }
      unsigned copied = 0;
      for (; next <= last && copied < count; ++next) {
        out[copied++] = events[(next - 1) % Capacity];
      }
      if (copied) {
        cursor = out[copied - 1].serial;
      }
      ReleaseSRWLockShared(&lock);
      return copied;
    }
  };
#endif
}  // namespace sunshine_game3d::test
