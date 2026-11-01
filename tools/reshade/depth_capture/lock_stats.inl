// SPDX-License-Identifier: GPL-3.0-only
// The capture mutex's wait counter (capture_lock_stats.h), a part of
// streamline_depth_capture.cpp (included at its end with the other parts).
// Counts only: nothing here changes which thread gets the lock or when.
namespace sunshine_streamline::depth_capture {
  SUNSHINE_CAPTURE_INTERNAL {
    struct lock_counter_state {
      lock_stats::role_counters game, present, present_hold;
      std::array<lock_stats::holder_counters, lock_stats::holder_slots> holders{};
      // The newest Present's thread (present()): its acquisitions are the
      // present role, every other thread's the game role.
      std::atomic<unsigned long> present_thread{};
      // Whether the present thread's holds are timed: the last report's
      // always (Diagnostics).
      std::atomic<bool> time_holds{};
    };
    lock_counter_state lock_counters;
    // capture_lock_report's window, beside the counters it is taken from:
    // the counters at the window's start (before), its start, and its
    // earliest end (next; 0: no baseline yet, also after a reset). busy:
    // one caller at a time (try: a Present never waits; a reset waits). Until
    // 11-01 the baseline was the report function's own statics, which a
    // reset of the counters left in place, and the next window's counts
    // wrapped below zero (F71).
    struct lock_report_window {
      std::atomic<bool> busy{};
      std::atomic<std::uint64_t> next{};
      std::uint64_t start{};
      lock_stats::values before{};
    };
    lock_report_window report_window;

    std::uint64_t lock_clock() {
      LARGE_INTEGER value{};
      QueryPerformanceCounter(&value);
      return std::uint64_t(value.QuadPart);
    }
    std::uint64_t lock_clock_ns(std::uint64_t ticks) {
      static const std::uint64_t frequency = [] {
        LARGE_INTEGER value{};
        QueryPerformanceFrequency(&value);
        return value.QuadPart > 0 ? std::uint64_t(value.QuadPart) : std::uint64_t(1);
      }();
      return ticks / frequency * 1000000000ull + ticks % frequency * 1000000000ull / frequency;
    }
    void raise_lock_max(std::atomic<std::uint64_t> &max, std::uint64_t value) {
      auto seen = max.load(std::memory_order_relaxed);
      while (value > seen && !max.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {}
    }
    void count_lock_duration(lock_stats::role_counters &role, std::uint64_t ns) {
      role.waits.fetch_add(1, std::memory_order_relaxed);
      role.wait_ns.fetch_add(ns, std::memory_order_relaxed);
      if (ns >= lock_stats::long_wait_ns) role.long_waits.fetch_add(1, std::memory_order_relaxed);
      raise_lock_max(role.max_ns, ns);
      raise_lock_max(role.window_max_ns, ns);
      role.histogram[lock_stats::bucket_of(ns)].fetch_add(1, std::memory_order_relaxed);
    }
    // A wait of ns on a lock that site held when the wait began (0: it was
    // released in between, counted under the unknown site). A site without a
    // slot of its own counts in the last slot, other (holder_slots).
    void count_lock_holder(std::uint64_t site, std::uint64_t ns) {
      if (!site) site = lock_stats::site(0, false);
      auto &slots = lock_counters.holders;
      auto *slot = &slots.back();
      for (std::size_t i = 0; i + 1 != slots.size(); ++i) {
        auto seen = slots[i].site.load(std::memory_order_acquire);
        if (!seen && slots[i].site.compare_exchange_strong(seen, site, std::memory_order_acq_rel)) seen = site;
        if (seen == site) {
          slot = &slots[i];
          break;
        }
      }
      slot->waits.fetch_add(1, std::memory_order_relaxed);
      slot->wait_ns.fetch_add(ns, std::memory_order_relaxed);
    }
    lock_stats::role_values lock_role_values(const lock_stats::role_counters &role) {
      lock_stats::role_values out;
      out.locks = role.locks.load(std::memory_order_relaxed);
      out.waits = role.waits.load(std::memory_order_relaxed);
      out.wait_ns = role.wait_ns.load(std::memory_order_relaxed);
      out.long_waits = role.long_waits.load(std::memory_order_relaxed);
      out.max_ns = role.max_ns.load(std::memory_order_relaxed);
      for (std::size_t i = 0; i != lock_stats::buckets; ++i) out.histogram[i] = role.histogram[i].load(std::memory_order_relaxed);
      return out;
    }
    void reset_lock_role(lock_stats::role_counters &role) {
      for (auto *value : {&role.locks, &role.waits, &role.wait_ns, &role.long_waits, &role.max_ns, &role.window_max_ns})
        value->store(0, std::memory_order_relaxed);
      for (auto &value : role.histogram) value.store(0, std::memory_order_relaxed);
    }
    // A window's role values: counts since the previous window, and the
    // window's own maximum.
    lock_stats::role_values lock_window(const lock_stats::role_values &now, const lock_stats::role_values &before,
        std::atomic<std::uint64_t> &window_max) {
      lock_stats::role_values out;
      out.locks = now.locks - before.locks;
      out.waits = now.waits - before.waits;
      out.wait_ns = now.wait_ns - before.wait_ns;
      out.long_waits = now.long_waits - before.long_waits;
      out.max_ns = window_max.exchange(0, std::memory_order_relaxed);
      for (std::size_t i = 0; i != lock_stats::buckets; ++i) out.histogram[i] = now.histogram[i] - before.histogram[i];
      return out;
    }
  }

  __attribute__((noinline)) void capture_mutex::lock() {
    const auto address = reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));
    const bool present = GetCurrentThreadId() == lock_counters.present_thread.load(std::memory_order_relaxed);
    auto &role = present ? lock_counters.present : lock_counters.game;
    role.locks.fetch_add(1, std::memory_order_relaxed);
    if (!native_.try_lock()) {
      const auto holder = holder_.load(std::memory_order_relaxed);
      const auto start = lock_clock();
      native_.lock();
      const auto ns = lock_clock_ns(lock_clock() - start);
      count_lock_duration(role, ns);
      count_lock_holder(holder, ns);
    }
    holder_.store(lock_stats::site(address, present), std::memory_order_relaxed);
    held_at_ = present && lock_counters.time_holds.load(std::memory_order_relaxed) ? lock_clock() : 0;
  }

  __attribute__((noinline)) bool capture_mutex::try_lock() {
    const auto address = reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));
    const bool present = GetCurrentThreadId() == lock_counters.present_thread.load(std::memory_order_relaxed);
    if (!native_.try_lock()) return false;
    (present ? lock_counters.present : lock_counters.game).locks.fetch_add(1, std::memory_order_relaxed);
    holder_.store(lock_stats::site(address, present), std::memory_order_relaxed);
    held_at_ = present && lock_counters.time_holds.load(std::memory_order_relaxed) ? lock_clock() : 0;
    return true;
  }

  void capture_mutex::unlock() {
    const auto held_at = std::exchange(held_at_, 0);
    holder_.store(0, std::memory_order_relaxed);
    native_.unlock();
    if (held_at) {
      lock_counters.present_hold.locks.fetch_add(1, std::memory_order_relaxed);
      count_lock_duration(lock_counters.present_hold, lock_clock_ns(lock_clock() - held_at));
    }
  }

  lock_stats::values capture_lock_values() {
    lock_stats::values out;
    out.game = lock_role_values(lock_counters.game);
    out.present = lock_role_values(lock_counters.present);
    out.present_hold = lock_role_values(lock_counters.present_hold);
    for (std::size_t i = 0; i != lock_stats::holder_slots; ++i) {
      const auto &slot = lock_counters.holders[i];
      out.holders[i] = {slot.site.load(std::memory_order_acquire), slot.waits.load(std::memory_order_relaxed),
        slot.wait_ns.load(std::memory_order_relaxed)};
    }
    HMODULE module{};
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCWSTR>(&lock_counters), &module))
      out.module_base = reinterpret_cast<std::uint64_t>(module);
    return out;
  }

  void reset_capture_lock_values() {
    // Fixtures only: waits for a report in progress.
    while (report_window.busy.exchange(true, std::memory_order_acquire)) SwitchToThread();
    reset_lock_role(lock_counters.game);
    reset_lock_role(lock_counters.present);
    reset_lock_role(lock_counters.present_hold);
    for (auto &slot : lock_counters.holders) {
      slot.site.store(0, std::memory_order_relaxed);
      slot.waits.store(0, std::memory_order_relaxed);
      slot.wait_ns.store(0, std::memory_order_relaxed);
    }
    report_window.next.store(0, std::memory_order_relaxed);
    report_window.busy.store(false, std::memory_order_release);
  }

  bool capture_lock_report(std::uint64_t now_ms, bool always, char *out, std::size_t size) {
    // Diagnostics times the present thread's holds (a store only on a change).
    if (lock_counters.time_holds.load(std::memory_order_relaxed) != always)
      lock_counters.time_holds.store(always, std::memory_order_relaxed);
    auto &window = report_window;
    if (const auto next = window.next.load(std::memory_order_relaxed); next && now_ms < next) return false;
    // One caller takes a window (another runtime's Present skips it).
    if (window.busy.exchange(true, std::memory_order_acquire)) return false;
    struct release_window {
      lock_report_window &window;
      ~release_window() { window.busy.store(false, std::memory_order_release); }
    } release{window};
    const auto next = window.next.load(std::memory_order_relaxed);
    if (next && now_ms < next) return false;
    const auto now = capture_lock_values();
    // The window maxima are exchanged at every window, the baseline's too:
    // the first window's are its own, not the largest since the process
    // began (until 11-01, F59).
    const auto game = lock_window(now.game, window.before.game, lock_counters.game.window_max_ns);
    const auto present = lock_window(now.present, window.before.present, lock_counters.present.window_max_ns);
    const auto hold = lock_window(now.present_hold, window.before.present_hold, lock_counters.present_hold.window_max_ns);
    // A slot's site is claimed once (until a reset, which takes a new
    // baseline), so each slot's window is its own difference.
    std::array<lock_stats::holder_values, lock_stats::holder_slots> holders{};
    for (std::size_t i = 0; i != holders.size(); ++i) {
      holders[i] = now.holders[i];
      holders[i].waits -= window.before.holders[i].waits;
      holders[i].wait_ns -= window.before.holders[i].wait_ns;
    }
    const auto window_ms = now_ms > window.start ? now_ms - window.start : 1;
    window.before = now;
    window.start = now_ms;
    window.next.store(now_ms + 10000, std::memory_order_relaxed);
    if (!next) return false;  // The baseline.
    if (!always && !game.long_waits) return false;
    if (!out || !size) return false;
    std::sort(holders.begin(), holders.end(), [](const auto &a, const auto &b) { return a.wait_ns > b.wait_ns; });
    char sites[256]{};
    for (std::size_t i = 0, used = 0; i != 4 && holders[i].waits; ++i) {
      const auto address = lock_stats::site_address(holders[i].site);
      const auto offset = address && address >= now.module_base ? address - now.module_base : 0;
      char site[32];
      if (holders[i].site)
        std::snprintf(site, sizeof(site), "%c+0x%llx", lock_stats::site_present(holders[i].site) ? 'p' : 'g',
          static_cast<unsigned long long>(offset));
      else std::snprintf(site, sizeof(site), "other");
      const int written = std::snprintf(sites + used, sizeof(sites) - used, "%s%s:%llu/%.3f", i ? "," : "", site,
        static_cast<unsigned long long>(holders[i].waits), double(holders[i].wait_ns) / 1e6);
      if (written < 0 || used + std::size_t(written) >= sizeof(sites)) break;
      used += std::size_t(written);
    }
    // The present thread's holds, for a window that timed some.
    char holds[80]{};
    if (hold.locks)
      std::snprintf(holds, sizeof(holds), " present_hold_p99_us=%llu present_hold_max_ms=%.3f",
        static_cast<unsigned long long>(lock_stats::quantile_us(hold, .99)), double(hold.max_ns) / 1e6);
    std::snprintf(out, size,
      "Sunshine depth capture lock: window_ms=%llu game_locks=%llu game_waits=%llu game_wait_ms=%.3f "
      "game_wait_pct=%.3f game_p50_us=%llu game_p99_us=%llu game_max_ms=%.3f game_waits_1ms=%llu "
      "present_locks=%llu present_waits=%llu present_wait_ms=%.3f present_p99_us=%llu present_max_ms=%.3f"
      "%s holders=%s",
      static_cast<unsigned long long>(window_ms), static_cast<unsigned long long>(game.locks),
      static_cast<unsigned long long>(game.waits), double(game.wait_ns) / 1e6,
      double(game.wait_ns) / 1e4 / double(window_ms),
      static_cast<unsigned long long>(lock_stats::quantile_us(game, .5)),
      static_cast<unsigned long long>(lock_stats::quantile_us(game, .99)), double(game.max_ns) / 1e6,
      static_cast<unsigned long long>(game.long_waits), static_cast<unsigned long long>(present.locks),
      static_cast<unsigned long long>(present.waits), double(present.wait_ns) / 1e6,
      static_cast<unsigned long long>(lock_stats::quantile_us(present, .99)), double(present.max_ns) / 1e6, holds,
      sites[0] ? sites : "none");
    return true;
  }
}
