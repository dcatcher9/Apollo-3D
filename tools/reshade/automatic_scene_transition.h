// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <cstdint>

namespace sunshine_raw_scene {
  enum class stereo_basis { raw, projection };
  // Invalid depth cannot be rendered safely. Only the return to valid stereo
  // is gradual; this never blends an old depth/color frame into a new one.
  // The ramp restarts after depth was lost: Presents shown without ready
  // depth for more than 250 ms, counted from the first of them (a pause
  // after one counts, as its flat image stayed on screen), a clock rollback
  // or a basis change. A pause in Presents alone loses no depth: the last
  // ready Present's stereo image stays on screen through it, so when the
  // first Present after it has ready depth on the same basis, strength
  // continues (owner ruling 10-27; until then any Present gap over 250 ms
  // restarted the ramp, so a game hitch showed its first Present after the
  // hitch flat although that frame's depth was ready). The caller tells it
  // every Present it shows, also one without Game 3D stereo (exporter.cpp,
  // reentry_heard), so a stretch of those is a loss and not a pause.
  class reentry_transition {
  public:
    float update(bool ready, std::uint64_t now_ms, stereo_basis basis = stereo_basis::raw) noexcept {
      // Switching between estimated raw controls and validated projection
      // controls must not inherit full strength from the other numeric basis.
      // Physical depth-buffer rotation deliberately is not part of this key.
      if (basis != basis_) { *this = {}; basis_ = basis; }
      const bool backwards = have_time_ && now_ms < previous_ms_;
      // A loss starts at the first Present shown without ready depth, never
      // at the last ready one: a pause before that Present showed stereo
      // (until the 10-27 review the loss ran from the last ready Present, so
      // a pause whose first Present after it was mono, a generated Present
      // of a frame whose DLSS-G copy was still in flight, restarted the
      // ramp at the next Present 8 ms later).
      if (was_ready_ && !ready) loss_ms_ = now_ms;
      const bool long_gap = have_ready_ && !was_ready_ && (now_ms < loss_ms_ || now_ms - loss_ms_ > 250);
      if (backwards || long_gap || !have_ready_) elapsed_ms_ = 0;
      // Credit only the interval actually spent displaying a ready frame, at
      // most 250 ms of it (a pause in Presents earns no more). A missing
      // current frame still outputs mono, but a brief gap does not erase the
      // preceding ready exposure or credit its missing interval.
      else if (was_ready_)
        elapsed_ms_ = std::min<std::uint64_t>(500, elapsed_ms_ + std::min<std::uint64_t>(250, now_ms - previous_ms_));
      previous_ms_ = now_ms;
      have_time_ = true;
      was_ready_ = ready;
      have_ready_ = have_ready_ || ready;
      return ready ? float(elapsed_ms_) / 500.f : 0.f;
    }
    // A Present its caller showed without Game 3D stereo, on the basis the
    // ramp has (no basis change of its own).
    void shown_without_depth(std::uint64_t now_ms) noexcept { update(false, now_ms, basis_); }
#if defined(SUNSHINE_SBS_TEST) || defined(SUNSHINE_SBS_RUNTIME_TEST_ADDON)
    // Test view only (the test add-on's per-Present trace,
    // game3d_test_render.h): the ramp's phase without its times. none: it
    // never heard ready depth; lost: the last Present it heard was not ready;
    // ready: it was (ramping or at full strength, which time decides).
    enum class test_phase : std::uint8_t { none, lost, ready };
    test_phase test_state() const noexcept {
      return !have_ready_ ? test_phase::none : was_ready_ ? test_phase::ready : test_phase::lost;
    }
    stereo_basis test_basis() const noexcept { return basis_; }
#endif
  private:
    // loss_ms_: the first Present shown without ready depth since the last
    // ready one.
    std::uint64_t previous_ms_{}, loss_ms_{}, elapsed_ms_{};
    stereo_basis basis_{stereo_basis::raw};
    bool was_ready_{}, have_ready_{}, have_time_{};
  };
}
