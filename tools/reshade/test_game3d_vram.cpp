// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_vram.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
  namespace vram = sunshine_game3d::vram;

  void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
  }

  vram::reading reading(std::uint32_t usage, std::uint32_t budget, std::uint32_t nonlocal = 0) {
    vram::reading out;
    out.local_usage_mib = usage;
    out.local_budget_mib = budget;
    out.nonlocal_usage_mib = nonlocal;
    out.nonlocal_budget_mib = 60000;
    out.valid = true;
    return out;
  }

  void window_extremes_and_time_over_budget() {
    vram::accumulator memory;
    memory.add(vram::reading{}, 100);  // A failed query records nothing.
    require(!memory.take().samples, "A failed query counted as a sample");
    memory.add(reading(14000, 15000), 1000);
    memory.add(reading(15200, 15000, 300), 1250);  // 200 MiB over: 250 ms over budget.
    memory.add(reading(14900, 14800), 1500);  // The budget dropped, 100 MiB over: 250 ms more.
    memory.add(reading(14000, 14800), 1750);
    // A freeze: the next sample is 3 s later, over budget; at most 1 s of it counts.
    memory.add(reading(15000, 14800), 4750);
    const auto w = memory.take();
    require(w.samples == 5 && w.max_usage_mib == 15200 && w.min_budget_mib == 14800,
      "The window lost its largest usage or smallest budget");
    require(w.min_headroom_mib == -200, "The window's smallest headroom is not its worst sample's budget minus usage");
    require(w.over_budget_ms == 250 + 250 + 1000, "Time over budget is not each over-budget sample's interval, capped at 1 s");
    require(w.budget_changes == 1, "A budget change was not counted once");
    require(w.last.local_usage_mib == 15000 && w.last.local_budget_mib == 14800, "The window's last sample is wrong");
    require(!memory.take().samples, "Taking a window did not start the next one empty");
    // The next window compares its first sample with the previous window's last.
    memory.add(reading(14000, 15000), 5000);
    const auto next = memory.take();
    require(next.budget_changes == 1 && next.over_budget_ms == 0 && next.min_headroom_mib == 1000,
      "A budget change across windows was lost, or an under-budget sample counted time over budget");
  }

  void lines() {
    char suffix[64];
    vram::format_suffix(suffix, sizeof(suffix), vram::reading{});
    require(!suffix[0], "An event line reported memory before the first sample");
    vram::format_suffix(suffix, sizeof(suffix), reading(9618, 14000, 208));
    require(std::string(suffix) == " vram_local_mib=9618/14000 nonlocal_mib=208", "The event suffix changed format");

    char line[384];
    require(!vram::format_window_line(line, sizeof(line), vram::window{}, 10000), "A window without samples logged a line");
    vram::accumulator memory;
    memory.add(reading(14000, 15000, 100), 0);
    memory.add(reading(15100, 15000, 400), 250);
    require(vram::format_window_line(line, sizeof(line), memory.take(), 10000), "A sampled window logged no line");
    require(std::string(line) ==
        "Sunshine Game 3D VRAM: local_mib={usage=15100 budget=15000} nonlocal_mib={usage=400 budget=60000} window={samples=2 "
        "max_usage_mib=15100 min_budget_mib=15000 min_headroom_mib=-100 over_budget_ms=250 budget_changes=0} window_ms=10000",
      "The window line changed format");
  }

  void sampler_period_and_latest() {
    vram::sampler memory;
    require(memory.due(0) && !memory.latest().valid, "A new sampler was not due, or had a sample");
    memory.record(reading(8000, 15000, 50), 1000);
    require(!memory.due(1249) && memory.due(1250), "The sampler is not due every 250 ms");
    const auto latest = memory.latest();
    require(latest.valid && latest.local_usage_mib == 8000 && latest.local_budget_mib == 15000 &&
        latest.nonlocal_usage_mib == 50 && latest.nonlocal_budget_mib == 60000,
      "The latest sample did not round-trip through its atomics");
    // A failed query waits for the next period and keeps the last good sample.
    memory.record(vram::reading{}, 1250);
    require(!memory.due(1499) && memory.latest().local_usage_mib == 8000, "A failed query retried at once or cleared the sample");
    require(memory.take_window().samples == 1, "The sampler's window lost its sample");
    require(vram::to_mib(3ull * vram::mib + 1) == 3, "Bytes do not round down to MiB");
  }
}  // namespace

int main() {
  try {
    window_extremes_and_time_over_budget();
    lines();
    sampler_period_and_latest();
    std::puts("PASS game3d_vram: window extremes, time over budget, line formats, sampler period");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL game3d_vram: " << error.what() << '\n';
    return 1;
  }
}
