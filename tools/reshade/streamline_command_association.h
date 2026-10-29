// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cstdint>

namespace sunshine_streamline::commands {
  // Identity and ordering only. ReShade lifecycle/execute callbacks are pre-call
  // observations, not proof that native APIs succeeded or that the GPU completed.
  struct recording_marker {
    std::uint64_t command{}, object_generation{}, recording_generation{}, event{}, epoch{}, loss{};
    explicit operator bool() const { return command && object_generation && recording_generation && event && epoch; }
  };
}
