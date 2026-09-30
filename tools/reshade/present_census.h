// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>

// Counts one runtime's Presents against two independent counts, to show which presented frames
// Game 3D can capture. Game 3D renders at ReShade's Present event, so it sees exactly the
// Presents that pass through ReShade's swapchain wrapper.
//  - DXGI's own count (IDXGISwapChain::GetLastPresentCount) on the swapchain object below
//    ReShade. Present calls made underneath ReShade, such as a lower Streamline proxy
//    presenting frame-generated images, raise it without a ReShade Present.
//  - Streamline's game-frame count, from the frame tokens the game requests per rendered frame.
// A DXGI count above ReShade's means frames that Game 3D never sees; ReShade Presents above the
// game-frame count mean generated frames do pass through Game 3D.
namespace sunshine_present_census {
  struct sample {
    bool frame_generation = false;
    // DXGI's cumulative Present count on the swapchain below ReShade, when the getter succeeded.
    bool dxgi_known = false;
    std::uint32_t dxgi_count = 0;
    // Streamline frame tokens: successful requests, and the newest game-supplied frame index.
    // A game may request the same frame's token more than once, so a supplied index is preferred.
    std::uint64_t token_calls = 0, token_index = 0;
    bool token_index_supplied = false;
  };

  struct totals {
    std::uint64_t reshade = 0, dxgi = 0, game_frames = 0;
    // Presents with a DXGI or game-frame delta, so a partly unknown window is not misread as zero.
    std::uint64_t dxgi_samples = 0, frame_samples = 0;
  };

  class counter {
  public:
    // One ReShade Present. Deltas since the previous Present are attributed to this Present's FG
    // state; only a transition window mixes the two states.
    void observe(const sample &value) {
      auto &bucket = window[value.frame_generation ? 1 : 0];
      ++bucket.reshade;
      if (value.dxgi_known) {
        // A new swapchain restarts DXGI's count; unsigned wrap is an ordinary increase.
        const std::uint32_t delta = value.dxgi_count - last_dxgi;
        if (dxgi_valid && delta < 0x80000000u) {
          bucket.dxgi += delta;
          ++bucket.dxgi_samples;
        }
        last_dxgi = value.dxgi_count;
      }
      dxgi_valid = value.dxgi_known;
      const bool indexed = value.token_index_supplied;
      const std::uint64_t position = indexed ? value.token_index : value.token_calls;
      // A restarted or switched numbering has no delta; it only starts a new baseline.
      if (frames_valid && indexed == last_indexed && position >= last_position) {
        bucket.game_frames += position - last_position;
        ++bucket.frame_samples;
      }
      frames_valid = indexed || value.token_calls != 0;
      last_indexed = indexed;
      last_position = position;
    }

    // Returns the window since the previous take; [0] FG off or unknown, [1] FG requested.
    void take(totals (&out)[2]) {
      out[0] = window[0];
      out[1] = window[1];
      window[0] = {};
      window[1] = {};
    }

  private:
    totals window[2];
    std::uint32_t last_dxgi = 0;
    bool dxgi_valid = false;
    std::uint64_t last_position = 0;
    bool frames_valid = false, last_indexed = false;
  };
}  // namespace sunshine_present_census
