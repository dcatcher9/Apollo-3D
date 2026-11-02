// SPDX-License-Identifier: GPL-3.0-only
// The fixtures' game clock (test_game3d_clock.h) against stand-ins for the
// test add-on: this executable exports the add-on's clock exports over the
// real test branch of game3d_clock.h (compiled here with
// SUNSHINE_GAME3D_TEST_CLOCK) and a scripted SunshineGame3DTestQuiesce, and
// its rings are CPU fences. Unbound or under --wall-clock nothing moves game
// time; engaged, each barrier waits for its rings first, then quiesces, and
// only then advances, and a failed barrier advances nothing.
#ifndef SUNSHINE_GAME3D_TEST_CLOCK
#error "Compiled with the test add-on's clock seam (SUNSHINE_GAME3D_TEST_CLOCK)"
#endif
#include "game3d_clock.h"
#include "test_game3d_clock.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
  void require(bool value, const std::string &message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  // Every call the clock made into the stand-in add-on, in order.
  std::vector<std::string> calls;
  // The scripted barrier: its result and the families it names when it fails,
  // and what runs inside it (the rings' state when the add-on was asked).
  BOOL quiesce_result = TRUE;
  unsigned quiesce_missed = 0;
  unsigned quiesce_bound = 0;
  std::function<void()> inside_quiesce;
  bool refuse_advance = false;

  // A fence the CPU signals, with ID3D12Fence's completion semantics: the
  // event is set once the completed value reaches the waited value, at once
  // when it has already.
  struct cpu_fence final : ID3D12Fence {
    std::mutex mutex;
    UINT64 completed = 0;
    std::vector<std::pair<UINT64, HANDLE>> waiters;

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void **out) override {
      if (out) *out = nullptr;
      return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT *, void *) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void *) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown *) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetName(const WCHAR *) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void **device) override {
      if (device) *device = nullptr;
      return E_NOTIMPL;
    }

    UINT64 STDMETHODCALLTYPE GetCompletedValue() override {
      std::lock_guard<std::mutex> lock(mutex);
      return completed;
    }

    HRESULT STDMETHODCALLTYPE SetEventOnCompletion(UINT64 value, HANDLE event) override {
      if (!event) return E_INVALIDARG;
      std::lock_guard<std::mutex> lock(mutex);
      if (completed >= value) {
        SetEvent(event);
      } else {
        waiters.emplace_back(value, event);
      }
      return S_OK;
    }

    HRESULT STDMETHODCALLTYPE Signal(UINT64 value) override {
      std::lock_guard<std::mutex> lock(mutex);
      completed = value;
      for (auto waiter = waiters.begin(); waiter != waiters.end();) {
        if (completed >= waiter->first) {
          SetEvent(waiter->second);
          waiter = waiters.erase(waiter);
        } else {
          ++waiter;
        }
      }
      return S_OK;
    }
  };

  // A queue ring as the native provider fixture names its parts.
  struct ring_t {
    struct {
      ID3D12Fence *p {};
    } done;
    std::uint64_t value {};
  };

  std::uint64_t elapsed_ms(std::chrono::steady_clock::time_point started) {
    return std::uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count());
  }

  // A fence signal from another thread after a delay, joined when it goes
  // (also when a check throws first).
  struct signal_later {
    std::thread thread;

    signal_later(cpu_fence &fence, UINT64 value):
        thread([&fence, value] {
          Sleep(60);
          fence.Signal(value);
        }) {}

    ~signal_later() {
      thread.join();
    }
  };

  template<class F>
  std::string failure_of(F &&call) {
    try {
      call();
    } catch (const std::exception &error) {
      return error.what();
    }
    return {};
  }
}  // namespace

// The stand-in add-on's exports, over the real test branch of the clock.
extern "C" __declspec(dllexport) std::uint64_t SunshineGame3DTestClockEngage(std::uint64_t lead_ms) {
  calls.push_back("engage " + std::to_string(lead_ms));
  return sunshine_game3d::clock::engage(lead_ms);
}

extern "C" __declspec(dllexport) std::uint64_t SunshineGame3DTestClockNow() {
  return sunshine_game3d::clock::game_now();
}

extern "C" __declspec(dllexport) std::uint64_t SunshineGame3DTestClockAdvance(std::uint64_t ms) {
  calls.push_back("advance " + std::to_string(ms));
  return refuse_advance ? 0 : sunshine_game3d::clock::advance(ms);
}

extern "C" __declspec(dllexport) BOOL SunshineGame3DTestQuiesce(unsigned bound_ms, unsigned *missed_families) {
  calls.push_back("quiesce");
  quiesce_bound = bound_ms;
  if (inside_quiesce) inside_quiesce();
  if (missed_families) *missed_families = quiesce_result ? 0 : quiesce_missed;
  return quiesce_result;
}

int main() {
  namespace fixture = sunshine_game3d_fixture;
  namespace pool = sunshine_game3d::pool;
  using fixture::game_clock;
  try {
    // --wall-clock: taken out of the arguments wherever it stands, the rest
    // kept in order with argv's null end; the process's clock never engages.
    {
      char a0[] = "fixture", a1[] = "--wall-clock", a2[] = "official.dll", a3[] = "--trace", a4[] = "--wall-clock";
      char *argv[] {a0, a1, a2, a3, a4, nullptr};
      int argc = 5;
      require(fixture::take_wall_clock_option(argc, argv), "--wall-clock was not found");
      require(argc == 3 && std::string(argv[0]) == "fixture" && std::string(argv[1]) == "official.dll" &&
          std::string(argv[2]) == "--trace" && argv[3] == nullptr,
        "--wall-clock was not removed from the arguments in order");
      require(fixture::fixture_clock().wall_clock(), "--wall-clock did not set the process clock's mode");
      char b0[] = "fixture", b1[] = "official.dll";
      char *plain[] {b0, b1, nullptr};
      int plain_count = 2;
      require(!fixture::take_wall_clock_option(plain_count, plain) && plain_count == 2 && plain[1] == b1,
        "Arguments without --wall-clock changed");
    }

    // Unbound: the real tick; a hitch is a real sleep; nothing engages.
    game_clock clock;
    clock.bound_ms = 300;
    const auto before = GetTickCount64();
    const auto unbound = clock.now();
    require(before <= unbound && unbound <= GetTickCount64(), "An unbound clock did not read the real tick");
    require(!clock.bound() && !clock.engaged(), "A new clock is bound or engaged");
    auto started = std::chrono::steady_clock::now();
    clock.hitch(20);
    require(elapsed_ms(started) >= 10 && calls.empty(), "An unengaged hitch did not sleep in real time, or called the add-on");
    require(!failure_of([&] { clock.frame_boundary(); }).empty() && calls.empty(),
      "An unengaged frame boundary was accepted");
    require(!failure_of([&] { clock.engage(); }).empty() && calls.empty(), "An unbound clock engaged");

    // A module without the exports (the production add-on) binds nothing.
    require(!clock.bind(GetModuleHandleW(L"kernel32.dll")) && !clock.bound(), "A module without the exports was bound");
    require(clock.bind(GetModuleHandleW(nullptr)) && clock.bound(), "The stand-in add-on's exports were not bound");

    // --wall-clock on a bound clock: engage refuses without calling the add-on.
    clock.set_wall_clock(true);
    require(!clock.engage() && !clock.engaged() && calls.empty(), "A --wall-clock clock engaged");
    clock.set_wall_clock(false);

    // Rings: one in flight that completes 60 ms from now, one complete, one
    // released (no fence), one behind a closed gate that never opens here.
    cpu_fence in_flight_fence, complete_fence, gated_fence;
    ring_t in_flight, complete, released, behind_gate;
    in_flight.done.p = &in_flight_fence;
    in_flight.value = 5;
    in_flight_fence.Signal(3);
    complete.done.p = &complete_fence;
    complete.value = 2;
    complete_fence.Signal(2);
    released.value = 9;
    behind_gate.done.p = &gated_fence;
    behind_gate.value = 7;
    bool gate_closed = true;
    auto watch_in_flight = clock.watch("in_flight", in_flight);
    auto watch_complete = clock.watch("complete", &complete.done.p, &complete.value);
    auto watch_released = clock.watch("released", released);
    auto watch_gated = clock.watch("behind_gate", behind_gate, [&] { return gate_closed; });

    // Engage: rings, then the barrier, then the engage with its lead.
    std::uint64_t seen_at_quiesce = 0;
    inside_quiesce = [&] { seen_at_quiesce = in_flight_fence.GetCompletedValue(); };
    {
      signal_later completes(in_flight_fence, 5);
      require(clock.engage(1000), "A bound clock did not engage");
    }
    require(calls == std::vector<std::string> {"quiesce", "engage 1000"} && seen_at_quiesce == 5,
      "Engage did not wait for its rings, then quiesce, then engage");
    require(quiesce_bound == clock.bound_ms, "The barrier did not pass its bound to the add-on");
    const auto engaged = clock.now();
    require(clock.engaged() && engaged == sunshine_game3d::clock::game_now() && engaged >= GetTickCount64() + 900,
      "The engaged clock does not read the add-on's virtual clock lead ms ahead");
    require(clock.engage(1000) && calls.size() == 2 && clock.now() == engaged, "A second engage changed the clock");
    require(clock.totals().ring_waits == 1 && clock.totals().ring_skips == 1,
      "Engage did not wait for the ring in flight and skip the gated one");

    // A frame boundary: the ring in flight is waited for before the barrier,
    // and the clock moves frame_ms forward only after it.
    calls.clear();
    in_flight.value = 6;
    {
      signal_later completes(in_flight_fence, 6);
      clock.frame_boundary();
    }
    require(calls == std::vector<std::string> {"quiesce", "advance 16"} && seen_at_quiesce == 6,
      "A frame boundary did not wait for its rings, then quiesce, then advance 16 ms");
    require(clock.now() == engaged + game_clock::frame_ms && clock.totals().frames == 1,
      "A frame boundary did not move game time one frame");
    require(clock.totals().ring_waits == 2 && clock.totals().ring_skips == 2, "The gated ring was waited for");

    // An engaged hitch: the same barrier, ms of game time, no real sleep.
    calls.clear();
    started = std::chrono::steady_clock::now();
    clock.hitch(120000);
    require(elapsed_ms(started) < 100 && calls == std::vector<std::string> {"quiesce", "advance 120000"},
      "An engaged hitch slept in real time or skipped its barrier");
    require(clock.now() == engaged + game_clock::frame_ms + 120000 && clock.totals().hitches == 1 &&
        clock.totals().hitch_ms == 120000,
      "An engaged hitch did not move game time by its length");
    auto now = clock.now();

    // The gate opens with its ring still in flight: the bound passes, the
    // barrier fails by naming the ring, and neither quiesce nor time moves.
    gate_closed = false;
    calls.clear();
    started = std::chrono::steady_clock::now();
    auto failure = failure_of([&] { clock.frame_boundary(); });
    require(failure.find("ring behind_gate did not reach value 7") != std::string::npos,
      "A ring past its bound did not fail the frame boundary by name: " + failure);
    require(elapsed_ms(started) >= 250 && calls.empty() && clock.now() == now,
      "A ring past its bound did not hold the barrier for its bound, or time moved");
    gated_fence.Signal(7);

    // That abandoned wait's late completion woke the wait event; the next
    // ring wait is not fooled by it.
    in_flight.value = 7;
    calls.clear();
    {
      signal_later completes(in_flight_fence, 7);
      clock.frame_boundary();
    }
    require(seen_at_quiesce == 7 && calls == std::vector<std::string> {"quiesce", "advance 16"} &&
        clock.now() == now + game_clock::frame_ms,
      "A stale wake let a frame boundary pass a ring still in flight");
    now = clock.now();

    // The add-on does not quiesce: the barrier fails naming the families
    // still out, and the clock does not move.
    quiesce_result = FALSE;
    quiesce_missed = pool::family_bit(pool::family::ngx_discover) | pool::family_bit(pool::family::test_held);
    calls.clear();
    failure = failure_of([&] { clock.frame_boundary(); });
    require(failure.find("did not quiesce") != std::string::npos && failure.find("ngx_discover, test_held") != std::string::npos,
      "A failed quiesce did not name the families still out: " + failure);
    require(calls == std::vector<std::string> {"quiesce"} && clock.now() == now, "Time moved past a failed quiesce");
    failure = failure_of([&] { clock.hitch(400); });
    require(failure.find("Game clock hitch") == 0 && clock.now() == now, "A hitch moved time past a failed quiesce");
    quiesce_result = TRUE;

    // A refused advance fails the barrier.
    refuse_advance = true;
    failure = failure_of([&] { clock.frame_boundary(); });
    require(failure.find("refused to advance") != std::string::npos && clock.now() == now, "A refused advance was not reported");
    refuse_advance = false;

    // A ring no longer watched is not waited for.
    watch_gated.reset();
    in_flight.value = 8;
    watch_in_flight = {};
    calls.clear();
    started = std::chrono::steady_clock::now();
    clock.frame_boundary();
    require(elapsed_ms(started) < 100 && calls == std::vector<std::string> {"quiesce", "advance 16"} &&
        clock.now() == now + game_clock::frame_ms,
      "An unwatched ring was still waited for");
    clock.report("fixture-clock-test");

    std::cout << "The fixture clock is the real tick until it engages (never under --wall-clock); engaged, every "
                 "frame boundary and hitch waits for its rings (gated ones skipped) and the add-on's barrier before "
                 "game time moves, and a ring past its bound or a failed barrier moves nothing\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
