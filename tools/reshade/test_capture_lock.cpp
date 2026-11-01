// SPDX-License-Identifier: GPL-3.0-only
// The capture lock's wait counter (capture_lock_stats.h, depth_capture/
// lock_stats.inl) on CPU threads: its numbers decide the owner's capture-lock
// split (docs/reshade-sbs.md, Capture lock waits; ruling Q5), and until 11-01
// no test read them (F91). A capture_mutex of the test's own, the capture
// owner's present() naming the present thread, and capture_lock_report
// driven with a clock of the test's own (10 s per window). Each row checks a
// rule that holds on any machine (no timing threshold): role attribution,
// the histogram against the waits, the holder sites and their overflow, the
// window baseline and maxima, the line's gate, bucket_of and quantile_us.
//   reshade_capture_lock_tests [row]   runs one row alone (a fresh process).
#include "capture_lock_stats.h"
#include "streamline_depth_capture.h"

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// The counter's holder accounting (lock_stats.inl), reached directly in the
// probe-test archive like the capture owner's other white-box regressions.
namespace sunshine_streamline::depth_capture {
  inline namespace test_access {
    void count_lock_holder(std::uint64_t site, std::uint64_t ns);
  }
}

namespace {
  namespace capture = sunshine_streamline::depth_capture;
  namespace stats = capture::lock_stats;

  void require(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
  }

  capture::capture_mutex mutex;
  std::uint64_t clock_ms = 1000;

  // The next window's report (10 s after the previous call): its line, or
  // empty when it printed none.
  std::string report(bool always) {
    char line[640]{};
    clock_ms += 10000;
    return capture::capture_lock_report(clock_ms, always, line, sizeof(line)) ? std::string(line) : std::string();
  }
  std::string field(const std::string &line, const char *name) {
    const auto key = std::string(" ") + name + "=";
    const auto at = line.find(key);
    require(at != std::string::npos, "the line has no " + std::string(name) + "=: " + line);
    const auto start = at + key.size();
    return line.substr(start, line.find(' ', start) - start);
  }
  std::uint64_t count(const std::string &line, const char *name) { return std::stoull(field(line, name)); }
  double number(const std::string &line, const char *name) { return std::stod(field(line, name)); }

  void lock_times(unsigned count) {
    for (unsigned i = 0; i != count; ++i) {
      mutex.lock();
      mutex.unlock();
    }
  }
  // One counted wait of the waiter's role: the present thread (this one)
  // waits while another thread holds the lock for hold_ms, or another thread
  // waits while this one holds it. Retried until the waiter's role counted
  // the wait (a waiter scheduled only after the hold ended takes the lock
  // uncontended).
  void contend(bool present_waits, unsigned hold_ms) {
    const auto waits = [present_waits] {
      const auto values = capture::capture_lock_values();
      return (present_waits ? values.present : values.game).waits;
    };
    for (unsigned attempt = 0; attempt != 20; ++attempt) {
      const auto before = waits();
      std::atomic<bool> held{}, waiting{};
      const auto hold = [&] {
        mutex.lock();
        held.store(true, std::memory_order_release);
        while (!waiting.load(std::memory_order_acquire)) std::this_thread::yield();
        Sleep(hold_ms);
        mutex.unlock();
      };
      const auto wait = [&] {
        while (!held.load(std::memory_order_acquire)) std::this_thread::yield();
        waiting.store(true, std::memory_order_release);
        mutex.lock();
        mutex.unlock();
      };
      if (present_waits) {
        std::thread other(hold);
        wait();
        other.join();
      } else {
        std::thread other(wait);
        hold();
        other.join();
      }
      if (waits() == before + 1) return;
    }
    require(false, "no contended acquisition was counted in 20 attempts");
  }
  std::uint64_t sum(const stats::role_values &value, std::size_t from = 0) {
    std::uint64_t out = 0;
    for (std::size_t i = from; i != stats::buckets; ++i) out += value.histogram[i];
    return out;
  }
  // The invariants of one role's cumulative counters, at rest (no thread
  // inside the mutex).
  void require_consistent(const stats::role_values &value, const char *role) {
    const std::string name = role;
    require(value.waits <= value.locks, name + ": more waits than acquisitions");
    require(sum(value) == value.waits, name + ": the histogram does not hold every wait");
    require(value.long_waits == sum(value, stats::bucket_of(stats::long_wait_ns)),
      name + ": the waits of a millisecond or more are not the histogram's buckets from 1000 us");
    require(!value.waits || value.max_ns, name + ": waits without a largest wait");
  }

  // F59: a wait before the first report call is no part of the first
  // window's maximum. The first call takes the baseline; until 11-01 the
  // window maxima were left in place there, so the first printed window's
  // present_max_ms was the largest wait since the add-on loaded.
  void first_window() {
    capture::present(0, 0);
    contend(true, 4);
    require(capture::capture_lock_values().present.max_ns >= stats::long_wait_ns, "the long wait before the first window was not counted");
    require(report(true).empty(), "the first report call printed a line instead of taking the baseline");
    const auto line = report(true);
    require(!line.empty(), "a Diagnostics window printed no line");
    require(count(line, "present_waits") == 0 && number(line, "present_max_ms") == 0.0,
      "the first window's present maximum held a wait from before the window: " + line);
    std::puts("PASS capture lock first window: a wait before the baseline is no part of the first window's maximum");
  }

  // F71: a reset of the counters (the fixtures' contention mode) takes a new
  // window baseline. Until 11-01 the report kept its own, and the next
  // window's counts were the reset counts minus the old baseline, wrapped.
  void reset_baseline() {
    capture::present(0, 0);
    report(true);
    lock_times(50);
    report(true);
    capture::reset_capture_lock_values();
    lock_times(2);
    auto line = report(true);
    if (!line.empty())
      require(count(line, "present_locks") <= 2 && count(line, "game_locks") <= 2,
        "the window after a reset counted from the old baseline: " + line);
    lock_times(3);
    line = report(true);
    require(!line.empty() && count(line, "present_locks") == 3 && count(line, "game_locks") == 0,
      "the second window after a reset did not count its own acquisitions: " + line);
    std::puts("PASS capture lock reset: a reset of the counters takes a new window baseline");
  }

  // F91: acquisitions of the present thread and of others count in their
  // own role, each wait once with its holder's site and role, and every
  // role's histogram holds its waits.
  void roles_and_waits() {
    capture::present(0, 0);
    capture::reset_capture_lock_values();
    lock_times(5);
    std::thread([] { lock_times(3); }).join();
    auto values = capture::capture_lock_values();
    require(values.present.locks == 5 && values.game.locks == 3, "acquisitions were not counted in their thread's role");
    require(!values.present.waits && !values.game.waits, "an uncontended acquisition counted a wait");
    contend(true, 4);
    contend(false, 4);
    values = capture::capture_lock_values();
    require_consistent(values.game, "game");
    require_consistent(values.present, "present");
    require(values.present.waits == 1 && values.present.long_waits == 1 && values.present.max_ns >= stats::long_wait_ns,
      "the present thread's wait of 4 ms was not counted as a wait of a millisecond or more");
    require(values.game.waits == 1 && values.game.long_waits == 1, "a game thread's wait of 4 ms was not counted");
    unsigned present_held = 0, game_held = 0;
    for (const auto &holder : values.holders) {
      if (!holder.waits) continue;
      require(holder.site && holder.waits == 1, "a wait was attributed to no site, or twice");
      ++(stats::site_present(holder.site) ? present_held : game_held);
    }
    require(present_held == 1 && game_held == 1, "the waits were not attributed to their holders' roles");
    std::puts("PASS capture lock roles: each acquisition counts in its thread's role, each wait once with its holder's site and role, "
      "and every histogram holds its role's waits");
  }

  // F56: a holder site past the named slots counts as other, never under
  // another site's name. Until 11-01 the last slot was named by the site
  // that claimed it, and every later site's waits were added to it.
  void holder_overflow() {
    capture::reset_capture_lock_values();
    report(true);  // The baseline after the reset.
    const unsigned sites = stats::holder_slots + 4;
    for (unsigned i = 0; i != sites; ++i)
      capture::count_lock_holder(stats::site(0x10000 + 64 * i, i % 2 != 0), 1000 + i);
    const auto values = capture::capture_lock_values();
    std::uint64_t named = 0, other = 0;
    for (const auto &holder : values.holders) {
      if (!holder.waits) continue;
      if (!holder.site) {
        other += holder.waits;
        continue;
      }
      require(holder.waits == 1, "a named holder site carries waits on other sites");
      ++named;
    }
    require(named == stats::holder_slots - 1 && other == sites - named,
      "the sites past the named slots were not counted as other: named " + std::to_string(named) + ", other " + std::to_string(other));
    const auto line = report(true);
    require(line.find(" holders=other:5/0.005,") != std::string::npos,
      "the line does not print the other slot first: " + line);
    std::puts("PASS capture lock holders: the sites past the named slots count as other, printed so");
  }

  // F39: the present thread's holds are timed only while the report's
  // always (Diagnostics) is on, and the line names them only for a window
  // that timed some; off, an acquisition costs a try_lock and a relaxed
  // count. Until 11-01 every present-thread hold read the clock twice.
  void hold_timing() {
    capture::present(0, 0);
    capture::reset_capture_lock_values();
    report(false);  // The baseline; holds untimed from here.
    lock_times(4);
    auto values = capture::capture_lock_values();
    require(values.present.locks == 4 && values.present_hold.locks == 0, "a present-thread hold was timed without Diagnostics");
    auto line = report(true);  // A Diagnostics window: holds timed from here.
    require(!line.empty() && line.find(" present_hold_") == std::string::npos,
      "a window without timed holds printed hold fields: " + line);
    lock_times(4);
    values = capture::capture_lock_values();
    require(values.present_hold.locks == 4 && sum(values.present_hold) == 4, "a present-thread hold was not timed with Diagnostics");
    line = report(true);
    require(line.find(" present_hold_p99_us=") != std::string::npos && line.find(" present_hold_max_ms=") != std::string::npos,
      "a window of timed holds printed no hold fields: " + line);
    // What timing costs an uncontended present-thread acquisition here (no
    // verdict: a machine's clock source decides it).
    const auto per_pair = [] {
      constexpr unsigned pairs = 200000;
      LARGE_INTEGER start{}, end{}, frequency{};
      QueryPerformanceFrequency(&frequency);
      QueryPerformanceCounter(&start);
      lock_times(pairs);
      QueryPerformanceCounter(&end);
      return double(end.QuadPart - start.QuadPart) * 1e9 / double(frequency.QuadPart) / pairs;
    };
    const auto timed = per_pair();
    report(false);
    const auto untimed = per_pair();
    std::printf("MEASURE capture lock present-thread lock+unlock ns: holds timed %.1f, untimed %.1f\n", timed, untimed);
    std::puts("PASS capture lock holds: the present thread's holds are timed only with Diagnostics, and printed only for a window "
      "that timed some");
  }

  // F91: bucket_of's edges, and quantile_us over every acquisition (an
  // uncontended one waits 0) as its bucket's upper bound, the open bucket
  // as the largest wait.
  void buckets_and_quantiles() {
    require(stats::bucket_of(0) == 0 && stats::bucket_of(9999) == 0 && stats::bucket_of(10000) == 1 &&
        stats::bucket_of(999999) == 6 && stats::bucket_of(1000000) == 7 && stats::bucket_of(15999999) == 10 &&
        stats::bucket_of(16000000) == 11 && stats::bucket_of(UINT64_MAX) == stats::buckets - 1,
      "bucket_of's edges moved");
    require(stats::bucket_of(stats::long_wait_ns) == 7 && stats::bounds_us[6] * 1000ull == stats::long_wait_ns,
      "a wait of a millisecond does not start the bucket after 1000 us");
    stats::role_values value{};
    require(stats::quantile_us(value, .99) == 0, "a role without acquisitions has a quantile");
    value.locks = 100;
    require(stats::quantile_us(value, .99) == 0, "uncontended acquisitions have a quantile");
    value.waits = 2;
    value.histogram[3] = 2;
    require(stats::quantile_us(value, .5) == 0 && stats::quantile_us(value, .99) == 100,
      "two waits of 50 to 100 us in 100 acquisitions are not p50 0 and p99 100 us");
    value = {};
    value.locks = value.waits = 1;
    value.histogram[stats::buckets - 1] = 1;
    value.max_ns = 20000001;
    require(stats::quantile_us(value, .99) == 20001, "the open bucket's quantile is not the largest wait");
    std::puts("PASS capture lock buckets: bucket_of's edges and quantile_us over every acquisition");
  }

  // F91: without Diagnostics a window prints only when a game thread waited
  // a millisecond or more; with Diagnostics every window prints.
  void gate() {
    capture::present(0, 0);
    capture::reset_capture_lock_values();
    report(false);  // The baseline after the reset.
    lock_times(3);
    auto line = report(false);
    require(line.empty(), "a window without waits printed a line without Diagnostics: " + line);
    contend(true, 4);
    line = report(false);
    require(line.empty(), "a window with only the present thread's wait printed a line without Diagnostics: " + line);
    contend(false, 4);
    line = report(false);
    require(!line.empty() && count(line, "game_waits") == 1 && count(line, "game_waits_1ms") == 1 &&
        number(line, "game_max_ms") >= 1.0, "a game thread's wait of 4 ms printed no line, or a wrong one: " + line);
    line = report(true);
    require(!line.empty() && count(line, "game_waits") == 0, "a Diagnostics window printed no line, or kept waits: " + line);
    std::puts("PASS capture lock gate: a window prints without Diagnostics only for a game thread's wait of a millisecond or more");
  }

  const std::vector<std::pair<const char *, void (*)()>> rows{
    {"first_window", first_window}, {"reset_baseline", reset_baseline}, {"roles_and_waits", roles_and_waits},
    {"holder_overflow", holder_overflow}, {"hold_timing", hold_timing}, {"buckets_and_quantiles", buckets_and_quantiles},
    {"gate", gate}};
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    bool ran = false;
    for (const auto &[name, row] : rows)
      if (argc < 2 || std::strcmp(argv[1], name) == 0) {
        row();
        ran = true;
      }
    require(ran, "no such row");
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
  return 0;
}
