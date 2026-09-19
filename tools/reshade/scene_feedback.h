// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>

namespace sunshine_scene_feedback {
  // Frozen with one provider input, never sampled again at GPU completion.
  // Revision zero is unavailable. A reset advances the provider revision so
  // older in-flight readbacks cannot refine the new scene's depth scale.
  struct sample {
    std::uint64_t revision{};
    bool reset{};
  };
}
