// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>

namespace sunshine_game3d {
  // Earning or losing trust in an alpha channel as UI coverage takes this many
  // detection samples spanning this long (docs/reshade-sbs.md).
  inline constexpr std::uint32_t alpha_trust_samples = 3;
  inline constexpr std::uint64_t alpha_trust_span_ms = 2000;

  class alpha_auto_policy;
  struct alpha_auto_source {
    std::uint64_t now_ms{}, epoch{}, revision{}, sequence{}, tick_ms{};
    std::uint32_t viewport{};
    bool retained{};
    alpha_auto_policy *session{}; // Caller-owned game session; never owned by a renderer.
    bool dedicated_mask{}; // Declared source type, not evidence of usable UI semantics.
  };

  enum class alpha_auto_state { waiting_for_source, collecting, automatic_on, automatic_off, manual_on, manual_off, dedicated_ui };

  inline const char *name(alpha_auto_state value) {
    switch (value) {
      case alpha_auto_state::waiting_for_source: return "searching";
      case alpha_auto_state::automatic_on: return "detected";
      case alpha_auto_state::automatic_off: return "no_usable_mask";
      case alpha_auto_state::manual_on: return "manual_on";
      case alpha_auto_state::manual_off: return "manual_off";
      case alpha_auto_state::dedicated_ui: return "dedicated_ui";
      default: return "checking";
    }
  }

  // The detection inputs that decide a frame's mask (SunshineUIDetectionReduceCS).
  // The first trusted alpha channel offered decides alone, whatever else is
  // offered beside it; otherwise every candidate bit can matter. Flags qualify
  // only the UI color alpha channel (bit 2). A status sample still describes
  // later frames while this key is unchanged.
  inline std::uint64_t detection_decision_key(std::uint32_t candidates, std::uint32_t flags, std::uint32_t trusted) {
    const auto decisive = candidates & trusted & 15u;
    if (!decisive) return std::uint64_t(flags) << 32 | candidates;
    const auto first = decisive & (0u - decisive);
    return std::uint64_t(first == 2u ? flags : 0u) << 32 | std::uint64_t(first) << 8;
  }

  struct alpha_auto_decision {
    bool enabled{};
    alpha_auto_state state = alpha_auto_state::waiting_for_source;
    std::uint32_t covered{}, pixels{};
    std::uint64_t sample_sequence{}, sample_tick_ms{}, accepted_samples{};
    // Latest completed diagnostic: 1 UI R, 2 UI A, 3 backbuffer A, 4 current A,
    // 5 HUD-less difference, 6 full-frame UI (HUD-less differs almost everywhere).
    std::uint32_t source_kind{};
    // Inputs of that GPU decision, for diagnosis only: the candidate bits the
    // shader was offered (1, 2, 4, 8 alpha candidates, 16 HUD-less), each alpha
    // candidate's covered and invalid pixels, HUD-less changed, unchanged and
    // non-finite pixels, tiles (of 256) whose pixels are 99% unchanged, and
    // the alpha candidates the game session trusted.
    struct detection_evidence {
      std::uint32_t candidates{}, hudless_changed{}, hudless_unchanged{}, hudless_invalid{}, matching_tiles{}, hudless_lit{};
      std::array<std::uint32_t, 4> alpha_covered{}, alpha_invalid{};
      std::uint32_t trusted_alpha{};
    } evidence;
  };

  // UI protection mode and alpha-channel trust for one game process. Live Auto
  // validates each frame's inputs on the GPU; this holds only the mode and the
  // trust those samples earn. The owning contract is docs/reshade-sbs.md, UI
  // protection under Setup.
  class alpha_auto_policy {
  public:
    void set_manual(bool enabled) {
      std::lock_guard<std::mutex> lock(mutex_);
      manual_ = enabled;
    }

    void set_automatic() {
      std::lock_guard<std::mutex> lock(mutex_);
      manual_.reset();
    }

    // The mode alone: manual On or Off, or Auto before a rendered frame
    // reports its current-frame detection.
    alpha_auto_decision decision() {
      std::lock_guard<std::mutex> lock(mutex_);
      alpha_auto_decision result;
      if (manual_) {
        result.enabled = *manual_;
        result.state = *manual_ ? alpha_auto_state::manual_on : alpha_auto_state::manual_off;
      }
      return result;
    }

    // One completed GPU detection sample over `pixels` pixels. Alpha candidates
    // are 0 UI alpha R, 1 UI color A, 2 Backbuffer A and 3 current A. A channel
    // earns trust as UI coverage by covering some but under 90% of the frame,
    // steadily (within a factor of two) in consecutive samples: a full-frame
    // sample restarts the run, and fluctuating scene effects do not qualify. A
    // trusted channel then decides the mask whatever it covers. It loses trust
    // on contradiction: covering at least 90% while an exact HUD-less pair shows
    // at least 75% of the scene unchanged. A selective sample clears doubt.
    void observe_alpha_channels(const alpha_auto_decision::detection_evidence &evidence, std::uint32_t pixels,
        std::uint64_t tick_ms) {
      if (!pixels) return;
      std::uint32_t changed_to{};
      std::function<void(std::uint32_t)> listener;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto before = trusted_alpha_;
        observe_alpha_channels_locked(evidence, pixels, tick_ms);
        if (trusted_alpha_ == before || !trust_listener_) return;
        changed_to = trusted_alpha_;
        listener = trust_listener_;
      }
      // Outside the lock: the owner may persist the new trust.
      listener(changed_to);
    }

    // Trust that an earlier session of this game earned (bits as trusted_alpha()).
    // It is the same claim as earned trust and is revoked by the same contradiction.
    void restore_trusted_alpha(std::uint32_t remembered) {
      std::lock_guard<std::mutex> lock(mutex_);
      trusted_alpha_ |= remembered & 15u;
    }

    // Called with the new trusted bits whenever samples earn or revoke trust,
    // never for a restore.
    void on_trust_change(std::function<void(std::uint32_t)> listener) {
      std::lock_guard<std::mutex> lock(mutex_);
      trust_listener_ = std::move(listener);
    }

    // Bit i: the game session trusts alpha candidate i as UI coverage.
    std::uint32_t trusted_alpha() {
      std::lock_guard<std::mutex> lock(mutex_);
      return trusted_alpha_;
    }

  private:
    void observe_alpha_channels_locked(const alpha_auto_decision::detection_evidence &evidence, std::uint32_t pixels,
        std::uint64_t tick_ms) {
      const std::uint64_t total = pixels;
      const bool scene_visible = (evidence.candidates & 48u) == 48u && !evidence.hudless_invalid &&
        std::uint64_t(evidence.hudless_unchanged) * 100 >= total * 75;
      for (std::uint32_t channel = 0; channel < 4; ++channel) {
        const std::uint32_t bit = 1u << channel;
        if (!(evidence.candidates & bit) || evidence.alpha_invalid[channel]) continue;
        const std::uint64_t covered = evidence.alpha_covered[channel];
        if (covered && covered * 10 < total * 9) {
          doubt_[channel] = {};
          if (earned_[channel].add(tick_ms, covered)) trusted_alpha_ |= bit;
        } else if (covered * 10 >= total * 9) {
          earned_[channel] = {};
          if ((trusted_alpha_ & bit) && scene_visible && doubt_[channel].add(tick_ms)) {
            trusted_alpha_ &= ~bit;
            doubt_[channel] = {};
          }
        }
      }
    }

    // Consecutive samples of one kind of evidence about one channel.
    struct evidence_run {
      std::uint32_t samples{};
      std::uint64_t first_tick_ms{}, low{}, high{};
      // True once alpha_trust_samples samples span alpha_trust_span_ms. A value
      // outside a factor of two of the run's others starts a new run with it.
      bool add(std::uint64_t tick_ms, std::uint64_t value = 0) {
        const auto lo = samples ? std::min(low, value) : value, hi = samples ? std::max(high, value) : value;
        if (!samples || tick_ms < first_tick_ms || hi > lo * 2) *this = {0, tick_ms, value, value};
        else { low = lo; high = hi; }
        ++samples;
        return samples >= alpha_trust_samples && tick_ms - first_tick_ms >= alpha_trust_span_ms;
      }
    };

    std::mutex mutex_;
    std::optional<bool> manual_; // Empty in Auto.
    std::array<evidence_run, 4> earned_{}, doubt_{};
    std::uint32_t trusted_alpha_{};
    std::function<void(std::uint32_t)> trust_listener_;
  };
}
