// SPDX-License-Identifier: GPL-3.0-only
#include "diagnostic_log_gate.h"

#include <atomic>
#include <iostream>
#include <stdexcept>

static void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  try {
    sunshine_diagnostics::log_gate gate;
    int logged_state = -1;
    unsigned writes = 0;
    // Startup text can alternate available/missing depth on every effect call.
    // Exercise 200 calls/second followed by a stable ready state.
    for (std::uint64_t ms = 0; ms < 35000; ms += 5) {
      const int state = static_cast<int>((ms / 5) % 2);
      if (gate.due(ms, state != logged_state)) {
        logged_state = state;
        ++writes;
      }
    }
    require(writes <= 35, "Startup alternation flooded the log");
    bool recovery_logged = false;
    for (std::uint64_t ms = 35000; ms <= 36000; ms += 5) {
      if (gate.due(ms, logged_state != 2, false, true)) {
        logged_state = 2;
        recovery_logged = true;
      }
    }
    require(recovery_logged, "Stable recovery was never reported");

    sunshine_diagnostics::log_gate ready;
    require(ready.due(0, true), "Initial state missing");
    for (std::uint64_t ms = 1; ms < 10000; ++ms)
      require(!ready.due(ms, false, false, true), "Ready snapshots are too frequent");
    require(ready.due(10000, false, false, true), "Adaptive scale snapshot missing");
    require(ready.due(10001, false, true), "Source change/error was delayed");
    require(!ready.due(10002, false), "Unchanged state was repeated");
    require(ready.due(10003, true, true), "Binding failure with a source change was delayed");
    require(ready.due(1, true), "Clock rollback suppressed future logs");

    // A line shared across threads: the first call claims it, then at most
    // one per interval, whoever asks.
    std::atomic<std::uint64_t> next{};
    require(sunshine_diagnostics::claim_line(next, 0, 5000), "The first shared line was refused");
    require(!sunshine_diagnostics::claim_line(next, 4999, 5000), "A shared line came before its interval");
    require(sunshine_diagnostics::claim_line(next, 5000, 5000) && next.load() == 10000,
      "A shared line due at its interval was refused or did not move its slot");
    unsigned claimed = 0;
    for (std::uint64_t ms = 5000; ms != 20000; ++ms) claimed += sunshine_diagnostics::claim_line(next, ms, 5000);
    require(claimed == 2, "A shared line was claimed more than once per interval");
    std::atomic<std::uint64_t> late{7000};
    require(!sunshine_diagnostics::claim_line(late, 6999, 1000) && sunshine_diagnostics::claim_line(late, 7000, 1000),
      "A shared line was not due exactly at its slot");

    // A line whose timer one owner holds follows the same rule.
    std::uint64_t owned = 0;
    require(sunshine_diagnostics::claim_line(owned, 0, 5000) && owned == 5000, "The first owned line was refused");
    require(!sunshine_diagnostics::claim_line(owned, 4999, 5000) && owned == 5000,
      "An owned line came before its interval, or a refusal moved its slot");
    unsigned owned_claims = 0;
    for (std::uint64_t ms = 5000; ms != 20000; ++ms) owned_claims += sunshine_diagnostics::claim_line(owned, ms, 5000);
    require(owned_claims == 3 && owned == 20000, "An owned line was claimed more or less than once per interval");
    std::cout << "Startup log bound, recovery, adaptive snapshots, immediate events, shared and owned line rate passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
