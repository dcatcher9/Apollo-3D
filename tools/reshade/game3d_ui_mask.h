// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "streamline_depth_capture.h"

// A live, bounded owner for the alpha of SL's real, pre-FG Backbuffer tag.
// It shares the native snapshot/retirement machinery with depth, but never
// nominates a depth source or interprets the alpha pixels on the CPU.
namespace sunshine_game3d::ui_mask {
  struct request {
    std::uint64_t runtime{}, device_identity{}, epoch{}, revision{};
    std::uint32_t viewport{}, width{}, height{};
    bool enabled{};
  };
  struct boundary {
    sunshine_scene_depth::frame source;
    std::uint64_t command{};
    std::uint32_t tag_scope{}; // 0 global, 1 explicit frame; never inferred from Present.
  };
  struct selection {
    sunshine_streamline::depth_capture::diagnostic_ticket ticket;
    sunshine_streamline::depth_capture::diagnostic_texture texture;
    boundary origin;
  };

  // Repeating an identical request preserves completed pixels. Disable or any
  // scope/size change revokes them, including in-flight attempts from the old scope.
  void set_request(const request &value);
  void invalidate(std::uint64_t runtime);
  void invalidate_all();
  void invalidate_scope(std::uint64_t epoch, std::uint32_t viewport);
  // Nonblocking GPU query; true exposes only completed immutable pixels within
  // maximum_source_age_ms. The consumer must still use copy_diagnostic_texture.
  bool acquire(std::uint64_t runtime, selection &out, std::uint64_t now_ms);

  // Streamline adapter boundary. No source reference is obtained when no live
  // request matches. Every begin attempt must finish after the original SDK call,
  // even if native recording had no free slot. A null/invalid tag revokes old alpha.
  bool interested(std::uint64_t epoch, std::uint64_t revision, std::uint32_t viewport);
  struct attempt {
    std::uint64_t runtime{}, generation{}, sequence{}, reservation{};
    sunshine_streamline::depth_capture::diagnostic_ticket ticket;
  };
  attempt begin(const boundary &where, const sunshine_streamline::depth_capture::input &input);
  void finish(const attempt &value, bool successful);
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing {
    bool last_attempt(std::uint64_t runtime, boundary &out);
  }
#endif
}
