// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <Windows.h>

namespace sunshine_game3d::hook_host {
  // Per-Present source maintenance, sequenced under the Streamline-owned lock.
  void poll();
  void initialize(HMODULE addon);
  void shutdown();
}
