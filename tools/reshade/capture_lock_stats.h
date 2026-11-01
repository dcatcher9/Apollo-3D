// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The capture owner's mutex and its wait counter (docs/reshade-sbs.md,
// Capture lock waits). Every capture-owner entry point (selection on the
// present thread; recording, barriers, submission and the Streamline/NGX
// hooks on game threads) serializes on one mutex. Whether game threads wait on
// it for long enough to split it (perf item 3) is decided by these counts:
// lock() first tries the native mutex, so an uncontended acquisition costs
// one try_lock and a relaxed count (on the present thread also a relaxed
// flag read); only a contended one is timed, and, while Diagnostics is on,
// the present thread's holds (two counter reads each; until 11-01 always,
// F39). No
// constructor or destructor beyond std::mutex's (its counters are atomics
// with constant initializers), so the object is initialized and destroyed
// where the plain mutex was.
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace sunshine_streamline::depth_capture {
  namespace lock_stats {
    // Wait histogram buckets: upper bounds in microseconds; the last bucket
    // is open.
    inline constexpr std::array<std::uint32_t, 11> bounds_us{10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000};
    inline constexpr std::size_t buckets = bounds_us.size() + 1;
    // The holder sites a waiter found holding the lock (the holder's lock()
    // return address): the first holder_slots - 1 sites each claim a slot;
    // the last slot names no site (site 0, printed as other) and takes the
    // waits on every later one. About 60 lock calls take the capture lock,
    // some on both thread roles. Until 11-01 the 16th site also claimed the
    // last slot, and every later site's waits were printed under it (F56).
    inline constexpr std::size_t holder_slots = 64;
    // A wait of this length counts toward waits of a millisecond or more.
    inline constexpr std::uint64_t long_wait_ns = 1000000;
    constexpr std::size_t bucket_of(std::uint64_t ns) {
      std::size_t i = 0;
      while (i != bounds_us.size() && ns >= std::uint64_t(bounds_us[i]) * 1000) ++i;
      return i;
    }
    // A holder site: valid bit 63, the lock call's return address shifted
    // left by one, and bit 0 set for the present thread.
    constexpr std::uint64_t site(std::uintptr_t address, bool present) {
      return (std::uint64_t(1) << 63) | (std::uint64_t(address) << 1) | (present ? 1u : 0u);
    }
    constexpr std::uintptr_t site_address(std::uint64_t value) {
      return std::uintptr_t((value & ~(std::uint64_t(1) << 63)) >> 1);
    }
    constexpr bool site_present(std::uint64_t value) { return (value & 1u) != 0; }
    // One role's acquisitions: game (every thread but the newest Present's)
    // or present. For the present thread's holds, locks counts the holds,
    // wait_ns their time and the histogram their lengths.
    struct role_counters {
      std::atomic<std::uint64_t> locks{}, waits{}, wait_ns{}, long_waits{}, max_ns{}, window_max_ns{};
      std::array<std::atomic<std::uint64_t>, buckets> histogram{};
    };
    struct holder_counters {
      std::atomic<std::uint64_t> site{}, waits{}, wait_ns{};
    };
    // A plain copy of the counters (capture_lock_values): cumulative since
    // the process began or the last reset; max_ns is the largest since then.
    // A holder whose site is 0 is the other slot (holder_slots).
    struct role_values {
      std::uint64_t locks{}, waits{}, wait_ns{}, long_waits{}, max_ns{};
      std::array<std::uint64_t, buckets> histogram{};
    };
    struct holder_values {
      std::uint64_t site{}, waits{}, wait_ns{};
    };
    struct values {
      role_values game, present, present_hold;
      std::array<holder_values, holder_slots> holders{};
      // The add-on module's base, for holder sites as module offsets.
      std::uint64_t module_base{};
    };
    // The q-quantile (0..1) of a role's waits over every acquisition (an
    // uncontended one waits 0), as its bucket's upper bound in microseconds;
    // the open bucket gives the largest wait seen. 0 without acquisitions.
    inline std::uint64_t quantile_us(const role_values &value, double q) {
      if (!value.locks) return 0;
      const auto rank = std::uint64_t(q * double(value.locks - 1)) + 1;
      std::uint64_t seen = value.locks > value.waits ? value.locks - value.waits : 0;
      if (seen >= rank) return 0;
      for (std::size_t i = 0; i != buckets; ++i) {
        seen += value.histogram[i];
        if (seen >= rank) return i < bounds_us.size() ? bounds_us[i] : (value.max_ns + 999) / 1000;
      }
      return (value.max_ns + 999) / 1000;
    }
  }

  // The capture owner's mutex: std::mutex with the wait counter. lock() and
  // unlock() are out of line (streamline_depth_capture.cpp), so a holder's
  // site is its caller's return address.
  class capture_mutex {
  public:
    capture_mutex() noexcept = default;
    capture_mutex(const capture_mutex &) = delete;
    capture_mutex &operator=(const capture_mutex &) = delete;
    void lock();
    bool try_lock();
    void unlock();

  private:
    std::mutex native_;
    // The holder's site (lock_stats::site), for a waiter's attribution; 0
    // once released.
    std::atomic<std::uint64_t> holder_{};
    // The performance counter at a present-thread acquisition while
    // Diagnostics is on (its hold is timed at unlock), else 0. Written and
    // read by the holder only.
    std::uint64_t held_at_{};
  };

  // The counters (cumulative), and a reset (fixtures only), which also makes
  // the next capture_lock_report call take a new window baseline.
  lock_stats::values capture_lock_values();
  void reset_capture_lock_values();
  // The bounded "Sunshine depth capture lock:" line: at most every 10 s,
  // only for a window in which a game thread waited a millisecond or more,
  // or every window when always is set (Diagnostics). The first call (and
  // the first after a reset) takes the window's baseline and prints
  // nothing; each window's maxima are its own. always also times the
  // present thread's holds from this call on (present_hold_*, printed for a
  // window that timed some).
  bool capture_lock_report(std::uint64_t now_ms, bool always, char *out, std::size_t size);
}
