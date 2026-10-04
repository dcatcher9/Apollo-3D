// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Rule H2 of the UI decision framework (M5, fix 2; docs/reshade-sbs.md, still
// screens without a UI source): some screens offer no UI source at all (a
// loading screen over a bare cleared target, a settings menu before any
// candidate is accepted), so no S1 source or H1 claim can flatten them and
// their 2D UI is warped by scene depth. In SDR Auto, a run of consecutive
// measured samples that read the presented frame's D at most
// still::d_percent / 100 (it does not show the depth's edges) while at least
// still::still_percent of all the D grid's cells keep their presented-luma
// mean (within still::tolerance of the previous measured sample's) enters
// the rule once it spans still::run_ms; the first sample that fails ends it
// at once. While it is active the renderer may push still::flatten (b2 word
// 5), and ui_selection::decide shows a frame that applies no source of its
// own, and was not reused by T1, flat as source 11. The default is a shadow:
// everything here runs and is logged, the flag is pushed only when the
// game's UIFlattenStillScreens is 1 (alpha_auto_policy::still_flatten).
//
// This header owns the predicate, the run and its episode bookkeeping; the
// hidden-scene guard (game3d_scene_guard.h) holds one run per renderer,
// feeds it every completed sample and clears it on an identity change, and
// the renderer ends it when the frame leaves scope (HDR, a manual mode) or a
// render in scope cannot be sampled (no offered candidate: a zero-offer T1
// grace frame, an inactive or unavailable detection), so that no frame is
// flattened by a run that nothing measures any more.
// Depth that is not the frame's own (per_frame_depth_not_current) does not
// invalidate a sample here, unlike H1: the warp of that frame uses the same
// reused depth, so D still answers whether the depth in use describes the
// image. No ReShade dependency.
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_selection.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace sunshine_game3d::still_screen {
  namespace still = ui_detection::still;
  // The cells of the D grid; a sample is measured only when every one of
  // them was compared with the previous measured sample.
  inline constexpr std::uint32_t cells = ui_detection::scene::cells_x * ui_detection::scene::cells_y;

  // Scope: SDR output, the renderer's swapchain colour space sRGB (1) with an
  // 8- or 10-bit UNORM or an 8-bit *_SRGB format. PQ (3), scRGB (2), HLG (4),
  // float and 16-bit formats are out of scope. The renderer passes its
  // source format, format_to_default_typed(backbuffer, 0), which is already
  // UNORM for an *_SRGB backbuffer, and its presented copy (t6) has that
  // format, so the stillness tolerance always compares code luma.
  constexpr bool sdr_output(std::uint32_t color_space, std::uint32_t dxgi) {
    using ui_selection::detail::family;
    const auto format = ui_selection::detail::classify(dxgi);
    return color_space == 1u && ((format.kind == family::unorm && (format.bits == 8u || format.bits == 10u)) ||
      (format.kind == family::srgb_typed && format.bits == 8u));
  }
  static_assert(sdr_output(1, 24) && sdr_output(1, 87) && sdr_output(1, 28) && sdr_output(1, 91) && !sdr_output(3, 24) &&
    !sdr_output(2, 10) && !sdr_output(4, 24) && !sdr_output(1, 10) && !sdr_output(1, 11) && !sdr_output(1, 0));

  // One completed detection sample as H2 reads it: its tick, whether the
  // renderer submitted it in scope (SDR Auto), the applied source and
  // whether T1 reused it (texels 0 and 9), whether the evidence passes ran,
  // and the presented frame's edge cells n and D (texel 5) and still and
  // compared cells (texel 12).
  struct input {
    std::uint64_t tick{};
    bool in_scope{};
    std::uint32_t source{};
    bool reused{}, ran{};
    std::uint32_t n{};
    float d{};
    std::uint32_t still{}, compared{};
  };

  // Why a run ended: a sample read the frame not hidden (D above
  // d_percent), moving (fewer than still_percent of the cells still),
  // unmeasured (no evidence, too few edge cells, cells without a previous
  // mean, a gap of more than max_gap_ms before a run entered, or a render in
  // scope that could not be sampled: no offered candidate), a source decided
  // (or T1 reused a decision), the frame left scope, or the renderer's
  // identity changed (game3d_scene_guard.h).
  enum class end_reason : std::uint8_t { not_hidden, moving, unmeasured, source, scope, identity };
  inline constexpr std::array<std::string_view, 6> end_reason_names{"not_hidden", "moving", "unmeasured", "source", "scope",
    "identity"};
  constexpr std::string_view name(end_reason reason) { return end_reason_names[std::size_t(reason)]; }

  // Why a sample fails the predicate, none when it passes. Blank frames are
  // not excluded: flattening black is harmless.
  constexpr std::optional<end_reason> failure(const input &s) {
    if (!s.in_scope) return end_reason::scope;
    if ((s.source && s.source != ui_detection::source_still) || s.reused) return end_reason::source;
    if (!s.ran || s.n < ui_detection::scene::min_edges || s.compared != cells) return end_reason::unmeasured;
    if (!(s.d * 100.f <= float(still::d_percent))) return end_reason::not_hidden;
    if (std::uint64_t(s.still) * 100u < std::uint64_t(still::still_percent) * cells) return end_reason::moving;
    return std::nullopt;
  }
  constexpr bool passes(const input &s) { return !failure(s); }
  static_assert(passes({0, true, 0, false, true, 128, .05f, (cells * 95 + 99) / 100, cells}) &&
    passes({0, true, 11, false, true, 400, -.068f, cells, cells}) &&
    failure({0, true, 0, false, true, 400, .0501f, cells, cells}) == end_reason::not_hidden &&
    failure({0, true, 0, false, true, 127, 0.f, cells, cells}) == end_reason::unmeasured &&
    failure({0, true, 0, false, true, 400, 0.f, cells - 1, cells - 1}) == end_reason::unmeasured &&
    failure({0, true, 0, false, true, 400, 0.f, (cells * 95 + 99) / 100 - 1, cells}) == end_reason::moving &&
    failure({0, true, 4, false, true, 400, 0.f, cells, cells}) == end_reason::source &&
    failure({0, true, 0, true, true, 400, 0.f, cells, cells}) == end_reason::source &&
    failure({0, false, 0, false, true, 400, 0.f, cells, cells}) == end_reason::scope);

  // A run of passing samples: its first and last ticks, the range of D, the
  // lowest still share and the samples it holds.
  struct episode {
    std::uint64_t first{}, last{};
    float d_min{}, d_max{}, share_min{};
    std::uint32_t samples{};
    std::uint64_t duration_ms() const { return last - first; }
  };

  // none: no run; pending: a run shorter than run_ms; active: the run spans
  // run_ms, so the screen would flatten (shadow) or flattens (enabled).
  enum class phase : std::uint8_t { none, pending, active };
  inline const char *name(phase value) {
    switch (value) {
      case phase::pending: return "pending";
      case phase::active: return "active";
      default: return "none";
    }
  }

  // What one observation, or one end, did: the run entered (became
  // active), an active episode ended (ended_episode, with its reason), or a
  // run ended before it entered (short_run: its length short_ms, the
  // gameplay-safety evidence).
  struct observation {
    bool entered{}, ended{};
    end_reason reason = end_reason::not_hidden;
    episode ended_episode;
    std::uint64_t short_ms{};
    bool short_run{};
  };

  struct run {
    // One completed sample, in tick order. A passing sample extends the run
    // and enters it once it spans run_ms; any other sample ends it at once.
    // A passing sample starts a new run instead when its tick is before the
    // last one, or more than still::max_gap_ms after it while the run has not
    // entered (the time between was not measured).
    observation observe(const input &s) {
      blocked_ = s.source && s.source != ui_detection::source_still;
      const auto failed = failure(s);
      observation result;
      const bool gap = running_ && (s.tick < current_.last || (!active_ && s.tick - current_.last > still::max_gap_ms));
      if (failed || gap) result = end(failed ? *failed : end_reason::unmeasured);
      if (failed) return result;
      const float share = float(s.still) / float(s.compared);
      if (!running_) {
        running_ = true;
        current_ = {s.tick, s.tick, s.d, s.d, share, 1u};
      } else {
        current_.last = s.tick;
        current_.d_min = std::min(current_.d_min, s.d);
        current_.d_max = std::max(current_.d_max, s.d);
        current_.share_min = std::min(current_.share_min, share);
        ++current_.samples;
      }
      if (!active_ && current_.duration_ms() >= still::run_ms) {
        active_ = true;
        result.entered = true;
      }
      return result;
    }
    // Ends the run without a sample: the frame left scope, the identity
    // changed, or a render in scope could not be sampled (unmeasured, which
    // keeps what the latest sample decided for wants_measure).
    observation leave(end_reason reason) {
      if (reason != end_reason::unmeasured) blocked_ = false;
      return end(reason);
    }
    bool active() const { return active_; }
    phase state() const { return active_ ? phase::active : running_ ? phase::pending : phase::none; }
    std::uint64_t run_ms() const { return running_ ? current_.duration_ms() : 0u; }
    // The current run (empty without one).
    const episode &current() const { return current_; }
    // Whether in-scope sample frames should run the evidence passes for H2:
    // always, except while the latest sample decided a source other than 11
    // (the predicate cannot pass until no source decides).
    bool wants_measure() const { return !blocked_; }

  private:
    observation end(end_reason reason) {
      observation result;
      result.reason = reason;
      if (running_) {
        result.ended_episode = current_;
        result.ended = active_;
        result.short_run = !active_;
        result.short_ms = active_ ? 0u : current_.duration_ms();
      }
      running_ = active_ = false;
      current_ = {};
      return result;
    }

    bool running_{}, active_{}, blocked_{};
    episode current_;
  };
}
