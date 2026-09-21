// SPDX-License-Identifier: GPL-3.0-only
#define SUNSHINE_OBSERVER_SNAPSHOT_TEST
#include "observer_snapshot.h"

#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace sunshine_streamline::observation {
  struct snapshot_test_access {
    template<class T, std::size_t N>
    static void interlock(snapshot_owner<T, N> &owner,
        void (*callback)(void *, std::uint64_t) noexcept, void *context) {
      owner.read_interlock_ = callback;
      owner.read_context_ = context;
    }
    template<class T, std::size_t N>
    static void commit_interlock(snapshot_owner<T, N> &owner,
        void (*callback)(void *) noexcept, void *context) {
      owner.commit_interlock_ = callback;
      owner.commit_context_ = context;
    }
  };
}

namespace {
  using namespace sunshine_streamline::observation;
  void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
  }
  template<class Predicate> bool wait_for(Predicate predicate) {
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
      if (std::chrono::steady_clock::now() >= limit) return false;
      std::this_thread::yield();
    }
    return true;
  }

  struct lease;
  struct value {
    std::uint64_t sequence{}, inverse{};
    std::array<std::uint64_t, 64> payload{};
    std::shared_ptr<lease> owned;
  };
  using bank = snapshot_owner<value>;
  void fill(value &out, std::uint64_t sequence) {
    out.sequence = sequence;
    out.inverse = ~sequence;
    for (std::size_t i = 0; i != out.payload.size(); ++i)
      out.payload[i] = (sequence * 1009 + i * 9176) ^ 0x5a1dc357b149ULL;
  }
  bool exact(const value &out, std::uint64_t sequence) {
    if (out.sequence != sequence || out.inverse != ~sequence) return false;
    for (std::size_t i = 0; i != out.payload.size(); ++i)
      if (out.payload[i] != ((sequence * 1009 + i * 9176) ^ 0x5a1dc357b149ULL)) return false;
    return true;
  }
  void publish(bank &owner, std::uint64_t sequence) {
    auto write = owner.try_write();
    require(bool(write), "Uncontended publication could not acquire a candidate");
    fill(*write, sequence);
    require(write.commit(), "Uncontended publication failed to commit");
  }

  struct reentry {
    bank *owner{};
    unsigned calls{}, succeeded{}, owner_busy{}, storage_busy{};
    std::uint64_t sequence{9000};
    bool clear_first{}, cleared{};
  };
  struct lease {
    reentry *target{};
    explicit lease(reentry &target) noexcept: target(&target) {}
    ~lease() {
      auto &result = *target;
      ++result.calls;
      if (result.clear_first) result.cleared = result.owner->clear();
      auto write = result.owner->try_write();
      if (!write) {
        if (write.failure() == write_failure::owner_busy) ++result.owner_busy;
        else ++result.storage_busy;
        return;
      }
      fill(*write, result.sequence);
      if (write.commit()) ++result.succeeded;
    }
  };
  void with_lease(bank &owner, reentry &result, std::uint64_t sequence) {
    auto supplied = std::make_shared<lease>(result);
    auto write = owner.try_write();
    require(bool(write), "Lease fixture could not acquire writer");
    fill(*write, sequence);
    write->owned = supplied;
    require(write.commit(), "Lease fixture could not publish");
  }

  void pinned_progress_and_capacity() {
    bank owner;
    require(!owner.read(), "Unpublished owner returned metadata");
    publish(owner, 1);
    auto held = owner.read();
    require(held && exact(*held, 1), "Initial pinned version was wrong");
    for (std::uint64_t sequence = 2; sequence != 200; ++sequence) {
      publish(owner, sequence);
      require(exact(*held, 1), "Held reader changed during later publications");
    }
    held.reset();
    require(owner.clear() && !owner.read(), "Clear failed without readers");

    std::array<bank::read_pin, 8> pins;
    for (std::size_t i = 0; i != pins.size(); ++i) {
      publish(owner, i + 1);
      pins[i] = owner.read();
      require(pins[i] && exact(*pins[i], i + 1), "Pressure fixture lost a pinned generation");
    }
    auto blocked = owner.try_write();
    require(!blocked && blocked.failure() == write_failure::storage_busy,
      "Eight pinned versions did not report bounded storage pressure");
    auto latest = owner.read();
    require(latest && exact(*latest, 8), "Storage pressure changed current publication");
    latest.reset();
    pins[2].reset();
    publish(owner, 9);
    for (std::size_t i = 0; i != pins.size(); ++i)
      if (pins[i]) require(exact(*pins[i], i + 1), "Recovery overwrote a still-pinned version");
    const auto old_ticket = pins[7].ticket();
    require(owner.clear() && !owner.read(), "Clear waited for or retained pinned publication");
    require(exact(*pins[7], 8), "Clear mutated a reader's old value");
    publish(owner, 10);
    latest = owner.read();
    require(latest && exact(*latest, 10) && latest.ticket() > old_ticket,
      "New lifecycle reused an old publication ticket");
  }

  void writer_abort_and_move() {
    bank owner;
    publish(owner, 11);
    auto prior = owner.read();
    const auto ticket = prior.ticket();
    auto write = owner.try_write();
    require(bool(write), "Writer fixture failed");
    fill(*write, 12);
    auto blocked = owner.try_write();
    require(!blocked && blocked.failure() == write_failure::owner_busy,
      "Concurrent mutation did not fail closed as owner_busy");
    auto during = owner.read();
    require(during && exact(*during, 11), "An uncommitted candidate leaked to readers");
    auto moved = std::move(write);
    require(!write && bool(moved), "Transaction move retained duplicate ownership");
    moved.abort();
    auto after = owner.read();
    require(after && after.ticket() == ticket && exact(*after, 11), "Abort replaced the committed publication");
    publish(owner, 13);
    auto reader = std::move(after);
    require(!after && reader && exact(*reader, 11), "Read-pin move lost its frozen value");
  }

  void lease_reentry() {
    // Commit retirement: overwritten inherited leases stay rooted by the base
    // pin until the writer flag is clear. Their destructor can publish anew.
    {
      bank owner;
      reentry result{&owner};
      with_lease(owner, result, 21);
      auto write = owner.try_write();
      require(bool(write), "Commit-reentry writer failed");
      write->owned.reset();
      fill(*write, 22);
      require(result.calls == 0, "Inherited lease died inside the writer transaction");
      require(write.commit(), "Commit-reentry publication failed");
      auto latest = owner.read();
      require(result.calls == 1 && result.succeeded == 1 && !result.owner_busy &&
          latest && exact(*latest, result.sequence), "Commit released a lease under writer ownership");
    }
    // Abort retirement: a lease held only by an unpublished candidate is also
    // released after the writer flag, without publishing that candidate.
    {
      bank owner;
      reentry result{&owner};
      publish(owner, 31);
      auto write = owner.try_write();
      require(bool(write), "Abort-reentry writer failed");
      write->owned = std::make_shared<lease>(result);
      write.abort();
      require(result.calls == 1 && result.succeeded == 1 && !result.owner_busy,
        "Abort released a lease under writer ownership");
    }
    // Lifecycle clear may reserve all slots while destructing them. Reentry
    // can report storage pressure, but must never find the writer held.
    {
      bank owner;
      reentry result{&owner};
      with_lease(owner, result, 41);
      require(owner.clear(), "Lease clear failed");
      require(result.calls == 1 && !result.owner_busy &&
          result.succeeded + result.storage_busy == 1, "Clear destroyed a lease under writer ownership");
    }
    // The final reader releases a retired lease after clear without acquiring
    // the writer flag, and can reenter even though no further poll is required.
    {
      bank owner;
      reentry result{&owner};
      with_lease(owner, result, 51);
      auto old = owner.read();
      require(owner.clear() && !owner.read() && result.calls == 0 && exact(*old, 51),
        "Clear destroyed or changed a still-pinned lease");
      publish(owner, 52);
      old.reset();
      require(result.calls == 1 && result.succeeded == 1 && !result.owner_busy,
        "Final retired reader held the writer during lease release");
    }
    // Reentrant clear meets the last-reader reclaimer while its slot still has
    // the writing bit. Adding retired must not strand that slot or let a new
    // publication reuse it before its destructor returns.
    {
      bank owner;
      reentry result{&owner};
      result.clear_first = true;
      with_lease(owner, result, 55);
      auto old = owner.read();
      auto write = owner.try_write();
      require(bool(write), "Reentrant-clear writer failed");
      write->owned.reset();
      fill(*write, 56);
      require(write.commit() && result.calls == 0, "Pinned lease retired before reentrant-clear fixture");
      old.reset();
      auto latest = owner.read();
      require(result.calls == 1 && result.cleared && result.succeeded == 1 && !result.owner_busy &&
          latest && exact(*latest, result.sequence), "Reentrant clear corrupted a writing/retired slot or newer publication");
      latest.reset();
      for (unsigned i = 0; i != 20; ++i) publish(owner, 100 + i);
    }
  }

  struct paused_read {
    std::atomic<bool> entered{}, resume{}, timed_out{};
    static void intercept(void *context, std::uint64_t) noexcept {
      auto &state = *static_cast<paused_read *>(context);
      if (state.entered.exchange(true, std::memory_order_acq_rel)) return;
      if (!wait_for([&] { return state.resume.load(std::memory_order_acquire); })) state.timed_out = true;
    }
  };
  void publication_aba() {
    for (bool lifecycle_clear : {false, true}) {
      bank owner;
      publish(owner, 61);
      auto initial = owner.read();
      const auto old_address = &*initial;
      const auto old_ticket = initial.ticket();
      initial.reset();
      paused_read pause;
      snapshot_test_access::interlock(owner, paused_read::intercept, &pause);
      std::atomic<bool> valid{};
      std::thread reader([&] {
        auto pin = owner.read();
        valid = pin && exact(*pin, 63) && pin.ticket() > old_ticket;
      });
      const bool entered = wait_for([&] { return pause.entered.load(std::memory_order_acquire); });
      bool reused = false, cleared = !lifecycle_clear;
      if (entered) {
        publish(owner, 62);
        if (lifecycle_clear) cleared = owner.clear();
        publish(owner, 63);
        auto current = owner.read();
        reused = current && &*current == old_address && current.ticket() != old_ticket;
      }
      pause.resume.store(true, std::memory_order_release);
      reader.join();
      snapshot_test_access::interlock(owner, nullptr, nullptr);
      require(entered && reused && cleared && !pause.timed_out && valid,
        "Reader paused before pin accepted an ABA-reused publication as its old version");
    }
  }

  struct paused_commit {
    std::atomic<bool> entered{}, resume{}, timed_out{};
    static void intercept(void *context) noexcept {
      auto &state = *static_cast<paused_commit *>(context);
      state.entered.store(true, std::memory_order_release);
      if (!wait_for([&] { return state.resume.load(std::memory_order_acquire); })) state.timed_out = true;
    }
  };
  void preempted_commit_keeps_readers_available() {
    bank owner;
    publish(owner, 71);
    auto old = owner.read();
    paused_commit pause;
    snapshot_test_access::commit_interlock(owner, paused_commit::intercept, &pause);
    std::atomic<bool> committed{};
    std::thread writer([&] {
      auto update = owner.try_write();
      if (update) {
        fill(*update, 72);
        committed = update.commit();
      }
    });
    const bool entered = wait_for([&] { return pause.entered.load(std::memory_order_acquire); });
    bool available = entered;
    if (entered) for (unsigned i = 0; i != 100; ++i) {
      auto pin = owner.read();
      available = available && pin && exact(*pin, 71) && pin.ticket() == old.ticket();
    }
    pause.resume.store(true, std::memory_order_release);
    writer.join();
    snapshot_test_access::commit_interlock(owner, nullptr, nullptr);
    auto current = owner.read();
    require(entered && available && committed && !pause.timed_out && current && exact(*current, 72) && exact(*old, 71),
      "A preempted commit made both old and new metadata unavailable to readers");
  }

  void concurrent_publication() {
    bank owner;
    publish(owner, 1);
    constexpr unsigned readers = 3, writes = 12000;
    std::atomic<bool> begin{}, done{}, failed{};
    std::atomic<unsigned> started{}, observations{};
    std::vector<std::thread> workers;
    for (unsigned i = 0; i != readers; ++i) workers.emplace_back([&] {
      started.fetch_add(1, std::memory_order_release);
      while (!begin.load(std::memory_order_acquire)) std::this_thread::yield();
      do {
        auto pin = owner.read();
        if (pin) {
          const auto sequence = pin->sequence;
          if (!exact(*pin, sequence)) failed = true;
          std::this_thread::yield();
          if (!exact(*pin, sequence)) failed = true;
          observations.fetch_add(1, std::memory_order_relaxed);
        }
      } while (!done.load(std::memory_order_acquire));
    });
    const bool ready = wait_for([&] { return started.load(std::memory_order_acquire) == readers; });
    begin.store(true, std::memory_order_release);
    unsigned published = 0;
    if (ready) for (unsigned sequence = 2; sequence != writes + 2; ++sequence) {
      auto write = owner.try_write();
      if (!write) { failed = true; break; }
      fill(*write, sequence);
      if (!write.commit()) { failed = true; break; }
      ++published;
    }
    done.store(true, std::memory_order_release);
    for (auto &worker : workers) worker.join();
    auto final = owner.read();
    require(ready && !failed && published == writes && observations && final && exact(*final, writes + 1),
      "Readers blocked a sole writer or observed a torn/reused immutable version");
  }
}

int main() {
  try {
    pinned_progress_and_capacity();
    std::cout << "PASS immutable pinned readers, eight-version pressure, release recovery and lifecycle clear\n";
    writer_abort_and_move();
    std::cout << "PASS genuine writer contention, unpublished abort and ownership moves\n";
    lease_reentry();
    std::cout << "PASS lease destructor reentry after commit, abort, clear and final retired reader\n";
    publication_aba();
    std::cout << "PASS paused reader rejects old ticket after exact slot-address ABA reuse\n";
    preempted_commit_keeps_readers_available();
    std::cout << "PASS preempted commit leaves complete old publication continuously readable\n";
    concurrent_publication();
    std::cout << "PASS 12000 full publications with three concurrent immutable readers\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
