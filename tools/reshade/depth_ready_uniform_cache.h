// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <vector>

namespace sunshine_depth {
  // Per-runtime ownership of the bufready_depth source. Generic and API depth
  // publish through the same cache, so either can hand readiness to the other.
  // ReShade invalidates uniform handles on every effects-reloaded event.
  template<class Uniform>
  class ready_uniform_cache {
  public:
    void invalidate() {
      uniforms_.clear();
      reflected_ = published_ = false;
    }

    template<class Reflect, class Write>
    void set(bool ready, Reflect &&reflect, Write &&write) {
      if (!reflected_) {
        reflect(uniforms_);
        reflected_ = true;
      }
      if (published_ && ready == ready_) return;
      for (const auto uniform : uniforms_) write(uniform, ready);
      ready_ = ready;
      published_ = true;
    }

  private:
    std::vector<Uniform> uniforms_;
    bool reflected_{}, published_{}, ready_{};
  };
}
