// SPDX-License-Identifier: GPL-3.0-only
#include "diagnostic_log_gate.h"

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
    require(ready.due(10003, true, true), "Manual selection was delayed");
    require(ready.due(1, true), "Clock rollback suppressed future logs");

    sunshine_diagnostics::family_gap_observer family;
    for (std::uint64_t present = 1; present <= 16; ++present) {
      const bool frame_ready = present % 2 != 0;
      const bool recurring = family.observe(present, 7, true, frame_ready);
      require(recurring == (present == 16), "Alternating member gaps did not trigger at16 native presents");
      const auto count = family.observations(), misses = family.misses();
      for (unsigned repeat = 0; repeat < 4; ++repeat)
        family.observe(present, 7, true, !frame_ready);
      require(family.observations() == count && family.misses() == misses,
        "Repeated effect/accessor calls changed present-based gap evidence");
    }
    require(family.misses() == 8, "Alternating selected/zero source gaps were not retained by family");
    for (std::uint64_t present = 17; present <= 32; ++present) family.observe(present, 7, true, true);
    require(!family.recurring() && family.misses() == 0 && family.observations() == 16,
      "Recovered ready frames did not age gaps out of the rolling window");
    require(!family.observe(33, 8, true, false) && family.observations() == 1 && family.misses() == 1,
      "A new family inherited old gap evidence");
    family.observe(34, 8, false, false);
    require(family.observations() == 0 && family.misses() == 0, "Ineligible startup retained a trigger window");
    family.observe(34, 8, true, false);
    require(family.observations() == 0, "A repeated ineligible present was later counted twice");
    for (std::uint64_t present = 35; present < 51; ++present) family.observe(present, 8, true, false);
    require(family.recurring() && family.misses() == 16, "Persistent current-unavailable frames were not diagnosed");
    family.observe(1, 8, true, true);
    require(!family.recurring() && family.observations() == 1 && family.misses() == 0,
      "Present rollback retained an old device window");
    family.observe(2, 0, true, false);
    require(!family.recurring() && family.observations() == 0, "Missing family identity accumulated trigger evidence");
    std::cout << "Startup log bound, recovery, adaptive snapshots, immediate events and family gap diagnostics passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
