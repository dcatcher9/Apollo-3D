// SPDX-License-Identifier: GPL-3.0-only
#define SUNSHINE_OBSERVATION_LOSS_JOURNAL_TEST
#include "observation_loss_journal.h"

#include <atomic>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace sunshine_streamline::loss_diagnostics {
  struct journal_test_access {
    static bool lock(journal &value, std::uint64_t revision) {
      return !value.slots_[revision % journal::capacity].lock.test_and_set(std::memory_order_acquire);
    }
    static void unlock(journal &value, std::uint64_t revision) {
      value.slots_[revision % journal::capacity].lock.clear(std::memory_order_release);
    }
  };
}

namespace {
  using namespace sunshine_streamline::loss_diagnostics;
  void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
  }
  event sample(std::uint64_t revision) {
    static constexpr char site[] = "journal-test";
    event out;
    out.revision = revision; out.tick = revision * 13 + 9;
    out.thread_id = static_cast<std::uint32_t>(revision ^ 0x5a5a5a5a);
    out.line = static_cast<std::uint32_t>(revision % 997 + 1);
    out.cause = static_cast<reason>(revision % 11);
    out.site = site;
    out.details.sequence = revision * 17;
    out.details.viewport = static_cast<std::uint32_t>(revision % 7);
    out.details.feature = static_cast<std::uint32_t>(revision % 19);
    out.details.sdk_result = -static_cast<std::int32_t>(revision % 1000);
    out.details.has_sdk_result = (revision & 1) != 0;
    out.details.reset = static_cast<std::uint32_t>(revision % 2);
    return out;
  }
  bool equal(const event &a, const event &b) {
    return a.revision == b.revision && a.tick == b.tick && a.thread_id == b.thread_id &&
      a.line == b.line && a.cause == b.cause && a.site == b.site &&
      a.details.sequence == b.details.sequence && a.details.viewport == b.details.viewport &&
      a.details.feature == b.details.feature && a.details.sdk_result == b.details.sdk_result &&
      a.details.has_sdk_result == b.details.has_sdk_result && a.details.reset == b.details.reset;
  }
  void exact_and_wrap() {
    journal value;
    event out = sample(999);
    require(!value.query(1, out) && equal(out, {}), "Absent revision did not clear output");
    require(!value.publish(sample(0)) && !value.query(0, out) && equal(out, {}),
      "Revision zero became a recorded observation");
    for (std::uint64_t revision = 1; revision <= journal::capacity; ++revision)
      require(value.publish(sample(revision)), "Uncontended publication failed");
    const auto &read_only = value;
    for (std::uint64_t revision = 1; revision <= journal::capacity; ++revision)
      require(read_only.query(revision, out) && equal(out, sample(revision)), "Exact revision lookup changed payload");
    const auto replacement = journal::capacity + 1;
    require(value.publish(sample(replacement)), "Newer modulo-slot revision was rejected");
    out = sample(999);
    require(!value.query(1, out) && equal(out, {}), "Overwritten revision returned replacement evidence");
    require(!value.publish(sample(1)), "Delayed older writer replaced newer revision");
    require(value.query(replacement, out) && equal(out, sample(replacement)), "Delayed writer damaged current evidence");
    require(value.query(2, out) && equal(out, sample(2)), "Wrap overwrote a different slot");
    require(value.publish(sample(UINT64_MAX)), "Large externally supplied revision was rejected");
    require(!value.publish(sample(UINT64_MAX - journal::capacity)), "Large delayed revision replaced newer evidence");
    require(value.query(UINT64_MAX, out) && equal(out, sample(UINT64_MAX)), "Large revision lookup was truncated");
  }
  void contention() {
    journal value;
    constexpr std::uint64_t revision = 13;
    require(value.publish(sample(revision)), "Contention fixture publication failed");
    event out = sample(999);
    {
      require(journal_test_access::lock(value, revision), "Contention fixture could not take its slot");
      struct release {
        journal &value;
        std::uint64_t revision;
        ~release() { journal_test_access::unlock(value, revision); }
      } held{value, revision};
      require(!value.publish(sample(revision + journal::capacity)), "Contended publication did not fail immediately");
      require(!value.query(revision, out) && equal(out, {}), "Contended lookup did not clear output");
      require(value.publish(sample(revision + 1)) && value.query(revision + 1, out) &&
          equal(out, sample(revision + 1)), "One busy slot blocked an unrelated slot");
    }
    require(value.query(revision, out) && equal(out, sample(revision)), "Contended publication changed old evidence");
    require(value.publish(sample(revision + journal::capacity)), "Released slot remained contended");
  }
  void concurrent_wrap() {
    journal value;
    constexpr unsigned writer_count = 4, reader_count = 2, writes_per_thread = 20000;
    std::atomic<std::uint64_t> next{1}, published{}, observed{};
    std::atomic<unsigned> done{};
    std::atomic<bool> start{}, failed{};
    const auto read = [&](std::uint64_t revision) {
      event out = sample(999);
      if (value.query(revision, out)) {
        if (!equal(out, sample(revision))) failed.store(true, std::memory_order_relaxed);
        observed.fetch_add(1, std::memory_order_relaxed);
      } else if (!equal(out, {})) failed.store(true, std::memory_order_relaxed);
    };
    std::vector<std::thread> workers;
    for (unsigned reader = 0; reader != reader_count; ++reader) workers.emplace_back([&, reader] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      std::uint64_t attempts = reader;
      while (done.load(std::memory_order_acquire) != writer_count) {
        const auto newest = next.load(std::memory_order_relaxed) - 1;
        const auto age = attempts++ % (journal::capacity * 2);
        read(newest > age ? newest - age : 0);
      }
    });
    for (unsigned writer = 0; writer != writer_count; ++writer) workers.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (unsigned i = 0; i != writes_per_thread; ++i) {
        const auto revision = next.fetch_add(1, std::memory_order_relaxed);
        if (value.publish(sample(revision))) published.fetch_add(1, std::memory_order_relaxed);
        read(revision);
      }
      done.fetch_add(1, std::memory_order_release);
    });
    start.store(true, std::memory_order_release);
    for (auto &worker : workers) worker.join();
    require(!failed.load() && published.load() && observed.load(), "Concurrent wrap exposed torn or wrong-revision evidence");
    // With contention ended, a complete final ring remains exactly readable.
    const auto final = next.load();
    for (std::uint64_t revision = final; revision != final + journal::capacity; ++revision)
      require(value.publish(sample(revision)), "Post-contention publication remained locked");
    for (std::uint64_t revision = final; revision != final + journal::capacity; ++revision) {
      event out;
      require(value.query(revision, out) && equal(out, sample(revision)), "Post-contention ring lost exact lookup");
    }
  }
}

int main() {
  try {
    exact_and_wrap();
    contention();
    concurrent_wrap();
    require(std::strcmp(name(reason::records_busy), "records_busy") == 0 &&
        std::strcmp(name(reason::test_injected), "test_injected") == 0 &&
        std::strcmp(name(static_cast<reason>(99)), "unknown") == 0, "Loss reason naming changed");
    std::cout << "Observation-loss journal exact lookup, retirement, contention and concurrent wrap passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
