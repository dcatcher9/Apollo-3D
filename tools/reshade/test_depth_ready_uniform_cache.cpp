// SPDX-License-Identifier: GPL-3.0-only
#include "depth_ready_uniform_cache.h"

#include <cstdio>
#include <stdexcept>
#include <utility>

namespace {
  void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
  }
  struct runtime {
    sunshine_depth::ready_uniform_cache<unsigned> cache;
    std::vector<unsigned> handles{1, 2};
    std::vector<std::pair<unsigned, bool>> writes;
    unsigned reflections{};

    void publish(bool ready) {
      cache.set(ready, [&](auto &out) { ++reflections; out = handles; },
        [&](unsigned handle, bool value) { writes.emplace_back(handle, value); });
    }
  };
}

int main() {
  try {
    runtime first;
    first.publish(false);
    require(first.reflections == 1 && first.writes.size() == 2,
      "initial mono readiness must be explicitly published to all effects");
    for (unsigned frame = 0; frame != 120; ++frame) first.publish(false);
    require(first.reflections == 1 && first.writes.size() == 2,
      "steady frames must neither reflect effects nor repeat readiness writes");
    first.publish(true); // API takes over from Generic.
    first.publish(true);
    first.publish(false); // Generic resumes without captured depth.
    require(first.reflections == 1 && first.writes.size() == 6 && !first.writes.back().second,
      "both depth owners must share readiness state without repeating reflection");

    first.handles = {3};
    first.cache.invalidate();
    first.publish(false);
    require(first.reflections == 2 && first.writes.size() == 7 && first.writes.back().first == 3,
      "reload must discard obsolete handles and publish even unchanged readiness");

    // ReShade also announces reload after destroying effects, before rebuilding
    // them. Empty reflection must be cached but invalidated by the final event.
    first.handles.clear();
    first.cache.invalidate();
    first.publish(true);
    first.publish(true);
    require(first.reflections == 3 && first.writes.size() == 7,
      "empty effects must not trigger reflection every frame");
    first.handles = {4, 5};
    first.cache.invalidate();
    first.publish(true);
    require(first.reflections == 4 && first.writes.size() == 9 && first.writes.back() == std::make_pair(5u, true),
      "completed reload must initialize the new effects even when readiness is unchanged");

    runtime second;
    second.handles = {7};
    second.publish(false);
    require(second.reflections == 1 && second.writes.size() == 1 && first.reflections == 4,
      "runtime uniform handles and readiness must remain isolated");
    std::puts("depth-ready uniform cache tests passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
