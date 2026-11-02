// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// The fixtures' side of the test add-on's game clock (game3d_clock.h, README,
// Test builds): a fixture that engages it plays game time on the add-on's
// virtual clock instead of the real tick.
//
// - bind(addon) finds the test add-on's clock exports (Engage, Advance, Now)
//   and its pool and log barrier (SunshineGame3DTestQuiesce). Until a bound
//   clock engages, nothing changes: now() is the real tick, hitch(ms) sleeps
//   ms and the shared fixture paces each Present with Sleep(3), as before.
// - engage() runs once the fixture's setup is done: every watched ring's work
//   completes, the add-on's pool work and log drain, and the clock starts
//   setup_lead_ms (120 s) ahead of the real tick, longer than every game-time
//   window, so nothing stamped on the real tick during setup survives into the
//   first timeline. Under --wall-clock (take_wall_clock_option) it never
//   engages and the fixture runs on the real clock exactly as before.
// - Engaged, every Present of the shared fixture ends at frame_boundary()
//   (fixture_t::step): watched rings complete, the add-on quiesces, and only
//   then does game time move frame_ms forward. A hitch is hitch(ms): the same
//   barrier, then ms of game time at once. Game time never moves while work
//   that has not been waited for may still read what the add-on would retire,
//   so there is no raw advance.
// - Wall time is wall_now(): watchdogs, deadlines and bounds on real hangs.
//
// Drive it from the fixture's own thread. now() alone may be read anywhere.
#include "game3d_pool_submit.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <d3d12.h>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <windows.h>

namespace sunshine_game3d_fixture {
  class game_clock {
  public:
    // One virtual frame: the real pace is one 15.6 ms timer tick per Present
    // (Sleep(3) ends at the next tick).
    static constexpr std::uint64_t frame_ms = 16;
    // The engage's lead: longer than the association window, the display's
    // 2 s idle release, the UI graveyard's 10 s and the cached renderer's 60 s.
    static constexpr std::uint64_t setup_lead_ms = 120000;

    // The real bound of each barrier: watched rings, then the add-on's
    // quiesce. Fixtures keep 3000; only the helper's own test shortens it.
    DWORD bound_ms = 3000;

    // What the barriers did, for a fixture's MEASURE lines (report).
    struct totals_t {
      std::uint64_t frames {}, hitches {}, hitch_ms {}, ring_waits {}, ring_skips {};
      double ring_wait_ms {}, ring_wait_max_ms {}, quiesce_ms {}, quiesce_max_ms {};
    };

    // A watched ring's registration; the ring is no longer waited for once
    // this is destroyed or reset (a ring that lives in one function).
    class ring_watch {
    public:
      ring_watch() = default;
      ring_watch(const ring_watch &) = delete;
      ring_watch &operator=(const ring_watch &) = delete;

      ring_watch(ring_watch &&other) noexcept:
          owner(std::exchange(other.owner, nullptr)), id(std::exchange(other.id, 0)) {}

      ring_watch &operator=(ring_watch &&other) noexcept {
        if (this != &other) {
          reset();
          owner = std::exchange(other.owner, nullptr);
          id = std::exchange(other.id, 0);
        }
        return *this;
      }

      ~ring_watch() {
        reset();
      }

      void reset() {
        if (owner) {
          owner->unwatch(id);
        }
        owner = nullptr;
        id = 0;
      }

    private:
      friend class game_clock;

      ring_watch(game_clock *clock, std::uint64_t value):
          owner(clock), id(value) {}

      game_clock *owner {};
      std::uint64_t id {};
    };

    game_clock() = default;
    game_clock(const game_clock &) = delete;
    game_clock &operator=(const game_clock &) = delete;

    ~game_clock() {
      if (event) {
        CloseHandle(event);
      }
    }

    // Binds the test add-on's clock exports and barrier; false (and unbound)
    // when the module lacks any of them, as the production add-on does.
    bool bind(HMODULE addon) {
      const auto find = [addon](const char *name) {
        return addon ? GetProcAddress(addon, name) : nullptr;
      };
      const auto engage_export = reinterpret_cast<engage_fn>(find("SunshineGame3DTestClockEngage"));
      const auto advance_export = reinterpret_cast<advance_fn>(find("SunshineGame3DTestClockAdvance"));
      const auto now_export = reinterpret_cast<now_fn>(find("SunshineGame3DTestClockNow"));
      const auto quiesce_export = reinterpret_cast<quiesce_fn>(find("SunshineGame3DTestQuiesce"));
      if (!engage_export || !advance_export || !now_export || !quiesce_export) {
        return false;
      }
      engage_call = engage_export;
      advance_call = advance_export;
      now_call = now_export;
      quiesce_call = quiesce_export;
      return true;
    }

    bool bound() const {
      return engage_call != nullptr;
    }

    // --wall-clock: never engage; the fixture keeps the real clock.
    void set_wall_clock(bool value) {
      wall_clock_mode = value;
    }

    bool wall_clock() const {
      return wall_clock_mode;
    }

    // Once the fixture's setup is done: watched rings complete, the add-on
    // quiesces, then the clock engages lead_ms ahead of the real tick. Returns
    // false, engaging nothing, under --wall-clock; true when engaged (a second
    // call changes nothing). Throws when unbound or when a barrier fails.
    bool engage(std::uint64_t lead_ms = setup_lead_ms) {
      if (wall_clock_mode) {
        std::printf("MEASURE game clock: wall clock (--wall-clock), not engaged\n");
        return false;
      }
      if (!bound()) {
        throw std::logic_error("Game clock: engage() needs bind() with the test add-on first");
      }
      if (engaged()) {
        return true;
      }
      barrier("engage");
      if (!engage_call(lead_ms)) {
        throw std::runtime_error("Game clock: the add-on did not engage its clock");
      }
      engaged_flag.store(true, std::memory_order_release);
      std::printf("MEASURE game clock engaged: lead_ms=%llu frame_ms=%llu\n", static_cast<unsigned long long>(lead_ms),
        static_cast<unsigned long long>(frame_ms));
      return true;
    }

    bool engaged() const {
      return engaged_flag.load(std::memory_order_acquire);
    }

    // Game time, as the add-on reads it: the virtual clock once engaged,
    // before that the real tick.
    std::uint64_t now() const {
      return engaged() ? now_call() : GetTickCount64();
    }

    // Wall time: watchdogs, deadlines and bounds on real hangs, never a
    // game-time stamp or judge.
    static std::uint64_t wall_now() {
      return GetTickCount64();
    }

    // The end of one Present (fixture_t::step, engaged only): watched rings
    // complete, the add-on quiesces, then game time moves frame_ms forward.
    void frame_boundary() {
      if (!engaged()) {
        throw std::logic_error("Game clock: frame_boundary() needs an engaged clock");
      }
      barrier("frame boundary");
      advance(frame_ms, "frame boundary");
      ++sums.frames;
    }

    // A hitch of ms: engaged, the frame boundary's barrier and then ms of game
    // time at once; otherwise the real Sleep(ms) it always was. Safe inside a
    // frame's recording before its Present: rings behind a closed gate are
    // not waited for.
    void hitch(std::uint64_t ms) {
      if (!engaged()) {
        Sleep(DWORD(ms < INFINITE ? ms : INFINITE - 1));
        return;
      }
      barrier("hitch");
      advance(ms, "hitch");
      ++sums.hitches;
      sums.hitch_ms += ms;
    }

    // Waits, at most bound_ms of real time in all, for each watched ring's
    // last signalled value, except a ring whose gate says a closed fixture
    // gate holds its work. Moves no time; the first part of every barrier.
    void wait_rings(const char *where = "ring wait") {
      const auto deadline = wall_now() + bound_ms;
      for (const auto &ring : rings) {
        ID3D12Fence *const fence = *ring.fence;
        const std::uint64_t value = *ring.value;
        if (!fence || !value || fence->GetCompletedValue() >= value) {
          continue;
        }
        if (ring.gated && ring.gated()) {
          ++sums.ring_skips;
          continue;
        }
        if (!event) {
          event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
          if (!event) {
            throw std::runtime_error("Game clock: could not create the ring wait event");
          }
        }
        const auto started = std::chrono::steady_clock::now();
        if (FAILED(fence->SetEventOnCompletion(value, event))) {
          fail(where, "could not wait for ring " + ring.name);
        }
        // Until the value completes; a wake by an earlier wait's late
        // completion waits again for what is left of the bound.
        while (fence->GetCompletedValue() < value) {
          const auto current = wall_now();
          const DWORD left = DWORD(deadline > current ? deadline - current : 0);
          if (WaitForSingleObject(event, left) != WAIT_OBJECT_0 && fence->GetCompletedValue() < value) {
            fail(where, "ring " + ring.name + " did not reach value " + std::to_string(value) + " within " +
              std::to_string(bound_ms) + " ms (completed " + std::to_string(fence->GetCompletedValue()) + ")");
          }
        }
        const double waited = elapsed_ms(started);
        ++sums.ring_waits;
        sums.ring_wait_ms += waited;
        sums.ring_wait_max_ms = waited > sums.ring_wait_max_ms ? waited : sums.ring_wait_max_ms;
      }
    }

    // Watches a ring of the game's queues: fence and value are the ring's
    // done fence and its last signalled value, read at every barrier (a
    // released ring's null fence is skipped). gated, when given, says a closed
    // fixture gate holds the ring's work (for example until after the
    // frame's Present), so a barrier inside the frame does not wait for it.
    [[nodiscard]] ring_watch watch(const char *name, ID3D12Fence *const *fence, const std::uint64_t *value,
        std::function<bool()> gated = {}) {
      if (!fence || !value) {
        throw std::logic_error("Game clock: a watched ring needs its fence and value");
      }
      rings.push_back({++next_id, name ? name : "ring", fence, value, std::move(gated)});
      return ring_watch(this, next_id);
    }

    // The same for a queue ring with a done fence (done.p) and its last
    // signalled value (value), as the native provider fixture's queue_ring.
    template<class Ring>
    [[nodiscard]] ring_watch watch(const char *name, const Ring &ring, std::function<bool()> gated = {}) {
      return watch(name, &ring.done.p, &ring.value, std::move(gated));
    }

    const totals_t &totals() const {
      return sums;
    }

    // One MEASURE line of the barriers' totals (times are wall time, for
    // information only).
    void report(const char *label) const {
      std::printf("MEASURE game clock %s: mode=%s frames=%llu hitches=%llu hitch_ms=%llu ring_waits=%llu ring_skips=%llu "
                  "ring_wait_ms=%.3f ring_wait_max_ms=%.3f quiesce_ms=%.3f quiesce_max_ms=%.3f\n",
        label, engaged() ? "virtual" : wall_clock_mode ? "wall" : "real", static_cast<unsigned long long>(sums.frames),
        static_cast<unsigned long long>(sums.hitches), static_cast<unsigned long long>(sums.hitch_ms),
        static_cast<unsigned long long>(sums.ring_waits), static_cast<unsigned long long>(sums.ring_skips),
        sums.ring_wait_ms, sums.ring_wait_max_ms, sums.quiesce_ms, sums.quiesce_max_ms);
    }

  private:
    using engage_fn = std::uint64_t (*)(std::uint64_t);
    using advance_fn = std::uint64_t (*)(std::uint64_t);
    using now_fn = std::uint64_t (*)();
    using quiesce_fn = BOOL (*)(unsigned, unsigned *);

    struct ring_entry {
      std::uint64_t id;
      std::string name;
      ID3D12Fence *const *fence;
      const std::uint64_t *value;
      std::function<bool()> gated;
    };

    static double elapsed_ms(std::chrono::steady_clock::time_point started) {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    }

    [[noreturn]] static void fail(const char *where, const std::string &what) {
      throw std::runtime_error(std::string("Game clock ") + where + ": " + what);
    }

    // Watched rings, then the add-on's pool work and log (bound_ms each).
    void barrier(const char *where) {
      wait_rings(where);
      unsigned missed = 0;
      const auto started = std::chrono::steady_clock::now();
      const BOOL quiet = quiesce_call(bound_ms, &missed);
      const double waited = elapsed_ms(started);
      sums.quiesce_ms += waited;
      sums.quiesce_max_ms = waited > sums.quiesce_max_ms ? waited : sums.quiesce_max_ms;
      if (!quiet) {
        std::string families;
        namespace pool = sunshine_game3d::pool;
        for (unsigned kind = 0; kind != unsigned(pool::family::count); ++kind) {
          if (missed & pool::family_bit(pool::family(kind))) {
            families += (families.empty() ? "" : ", ") + std::string(pool::family_name(pool::family(kind)));
          }
        }
        fail(where, "the add-on's pool work or log did not quiesce within " + std::to_string(bound_ms) +
          " ms (still out: " + (families.empty() ? std::string("none named") : families) + ")");
      }
    }

    // Moves game time ms forward; only after a barrier.
    void advance(std::uint64_t ms, const char *where) {
      if (!advance_call(ms)) {
        fail(where, "the add-on refused to advance its clock by " + std::to_string(ms) + " ms");
      }
    }

    void unwatch(std::uint64_t id) {
      for (auto ring = rings.begin(); ring != rings.end(); ++ring) {
        if (ring->id == id) {
          rings.erase(ring);
          return;
        }
      }
    }

    engage_fn engage_call {};
    advance_fn advance_call {};
    now_fn now_call {};
    quiesce_fn quiesce_call {};
    std::atomic<bool> engaged_flag {false};
    bool wall_clock_mode {};
    std::vector<ring_entry> rings;
    std::uint64_t next_id {};
    HANDLE event {};
    totals_t sums;
  };

  // The process's game clock: one test add-on per fixture process, one clock.
  inline game_clock &fixture_clock() {
    static game_clock clock;
    return clock;
  }

  // The common --wall-clock option: removes every --wall-clock from the
  // arguments (argc shrinks, argv keeps its order and its null end) before a
  // fixture parses its own, and puts the process's clock in wall-clock mode,
  // in which it never engages. Returns whether it was given.
  inline bool take_wall_clock_option(int &argc, char **argv) {
    int kept = argc > 0 ? 1 : 0;
    bool found = false;
    for (int i = kept; i < argc; ++i) {
      if (argv[i] && std::strcmp(argv[i], "--wall-clock") == 0) {
        found = true;
      } else {
        argv[kept++] = argv[i];
      }
    }
    if (found) {
      argc = kept;
      argv[argc] = nullptr;
      fixture_clock().set_wall_clock(true);
    }
    return found;
  }
}  // namespace sunshine_game3d_fixture
