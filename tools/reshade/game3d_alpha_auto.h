// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "game3d_ui_detection_contract.h"

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
  // Trust remembered from an earlier session protects from the first frame but
  // lapses unless this session earns it again within this long of the channel
  // first being offered.
  inline constexpr std::uint64_t alpha_trust_reconfirm_ms = 60000;

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
    // 5 HUD-less difference, 6 full-frame UI (HUD-less differs almost
    // everywhere), 8 full-frame UI over a hidden scene by the layer route, 9
    // by the HUD-less route. 7 is retired and never reused.
    std::uint32_t source_kind{};
    // Hidden-scene evidence of one image (docs/reshade-sbs.md, hidden-scene
    // evidence): edge cells n and D, whether the passes ran and the evidence
    // is valid, and the presented image's verdict and decided (untied)
    // comparisons.
    struct scene_evidence {
      std::uint32_t n{}, decided{};
      float d{};
      bool valid{}, ran{};
      ui_detection::scene_verdict verdict = ui_detection::scene_verdict::none;
    };
    // Inputs of that GPU decision, for diagnosis only: the candidate bits the
    // shader was offered (1, 2, 4, 8 alpha candidates, 16 HUD-less), each alpha
    // candidate's covered and invalid pixels, HUD-less changed, unchanged and
    // non-finite pixels, tiles (of 256) whose pixels are 99% unchanged, and
    // the alpha candidates the game session trusted.
    struct detection_evidence {
      std::uint32_t candidates{}, hudless_changed{}, hudless_unchanged{}, hudless_invalid{}, matching_tiles{}, hudless_lit{};
      std::array<std::uint32_t, 4> alpha_covered{}, alpha_invalid{};
      std::uint32_t trusted_alpha{};
      // The UI color candidate (bit 2) was the offscreen UI layer, not a tagged
      // UIColorAndAlpha.
      bool ui_layer{};
      // Pixels of alpha candidates 0 and 1 with alpha of at least 254/255.
      std::array<std::uint32_t, 2> alpha_opaque{};
      scene_evidence scene, hudless_scene;
      // How long consecutive samples have read the presented frame hidden
      // while no source decided, not blank, ending with this one; zero otherwise.
      std::uint64_t shadow_hidden_ms{};
    } evidence;
    // This render's state, not the sample's: the routes whose hidden-scene
    // verdict the renderer holds (1 layer, 2 HUD-less), and whether the
    // first-run shadow evaluates the evidence with the gates closed.
    std::uint32_t scene_hold{};
    bool scene_shadow{};
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

    // Trust belongs to a source, not to a detection slot: 0 UI alpha, 1 tagged
    // UI color+alpha, 2 Backbuffer, 3 current color and 4 the offscreen UI
    // layer. Slot 1 (UI color A) carries either source 1 or the layer
    // (evidence.ui_layer); Unreal's tagged UI color is the opaque final image
    // while its UI layer is a real mask, so one must never inherit the other's
    // trust (Stellar Blade flattened the whole frame for 15 s).
    static constexpr std::uint32_t source_count = 5, ui_layer_source = 4;
    static constexpr std::uint32_t source_of(std::uint32_t slot, bool ui_layer) {
      return slot == 1 && ui_layer ? ui_layer_source : slot;
    }
    // Slot-level trust remembered before sources were distinct. Its UI color
    // bit cannot tell a tagged UIColorAndAlpha (Resident Evil Requiem) from
    // the UI layer (The Witcher 3, Stellar Blade), so that source earns its
    // trust again.
    static constexpr std::uint32_t sources_from_slots(std::uint32_t slots) { return slots & 13u; }

    // One completed GPU detection sample over `pixels` pixels. Alpha candidates
    // are 0 UI alpha R, 1 UI color A, 2 Backbuffer A and 3 current A. A source
    // earns trust as UI coverage by covering some but under 90% of the frame,
    // steadily (within a factor of two) in consecutive samples: a full-frame
    // sample restarts the run, and fluctuating scene effects do not qualify. A
    // trusted channel then decides the mask whatever it covers. It loses trust
    // on contradiction: covering at least 90% while an exact HUD-less pair shows
    // at least 75% of the scene unchanged, or, for presented alpha, differing by
    // at least 10% of the frame from every trusted dedicated UI channel in the
    // same sample, menus included. A selective sample clears doubt.
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
    // It decides from the first frame and is revoked by the same contradiction,
    // but it is provisional: unless this session earns it again within
    // alpha_trust_reconfirm_ms of the channel first being offered, it lapses (and
    // is forgotten), so a wrong remembered claim cannot outlive every session.
    void restore_trusted_alpha(std::uint32_t remembered) {
      std::lock_guard<std::mutex> lock(mutex_);
      trusted_alpha_ |= remembered & 31u;
      provisional_ |= remembered & 31u;
    }

    // A game whose session started without remembered trust runs the
    // hidden-scene evidence as a first-run shadow: on sample frames whatever
    // the gates, logged only; it never changes a decision.
    void set_first_run(bool first_run) {
      std::lock_guard<std::mutex> lock(mutex_);
      first_run_ = first_run;
    }
    bool first_run() {
      std::lock_guard<std::mutex> lock(mutex_);
      return first_run_;
    }

    // Called with the new trusted bits whenever samples earn or revoke trust,
    // never for a restore.
    void on_trust_change(std::function<void(std::uint32_t)> listener) {
      std::lock_guard<std::mutex> lock(mutex_);
      trust_listener_ = std::move(listener);
    }

    // Bit i: the game session trusts alpha source i as UI coverage.
    std::uint32_t trusted_alpha() {
      std::lock_guard<std::mutex> lock(mutex_);
      return trusted_alpha_;
    }

    // Bit i: the source in detection slot i is trusted (Sunshine_UITrustedAlpha),
    // with the offscreen UI layer in slot 1 when ui_layer.
    std::uint32_t trusted_slots(bool ui_layer) {
      std::lock_guard<std::mutex> lock(mutex_);
      return (trusted_alpha_ & 13u) | ((trusted_alpha_ >> source_of(1, ui_layer)) & 1u) << 1;
    }

    // The offscreen UI layer should fill the UI color slot instead of a tagged
    // UIColorAndAlpha the session does not trust: when the layer is trusted, or
    // when the tagged buffer proved opaque over a visible scene.
    bool prefer_ui_layer() {
      std::lock_guard<std::mutex> lock(mutex_);
      return !(trusted_alpha_ & 2u) && ((trusted_alpha_ >> ui_layer_source & 1u) || ui_color_opaque_);
    }

  private:
    void observe_alpha_channels_locked(const alpha_auto_decision::detection_evidence &evidence, std::uint32_t pixels,
        std::uint64_t tick_ms) {
      const std::uint64_t total = pixels;
      const bool exact_pair = (evidence.candidates & 48u) == 48u && !evidence.hudless_invalid;
      const bool scene_visible = exact_pair && std::uint64_t(evidence.hudless_unchanged) * 100 >= total * 75;
      // A trusted dedicated UI source in slot 0 or 1 (UI alpha, UI color or the
      // UI layer) offered with at most 1% invalid pixels is the game's own UI
      // mask for this sample, menus included. disagreement(): the distance to
      // the nearest such mask.
      const auto disagreement = [&](std::uint64_t covered) {
        std::uint64_t nearest = UINT64_MAX;
        for (std::uint32_t slot = 0; slot < 2; ++slot) {
          if (!(evidence.candidates & (1u << slot)) || !(trusted_alpha_ >> source_of(slot, evidence.ui_layer) & 1u) ||
              std::uint64_t(evidence.alpha_invalid[slot]) * 100 > total) continue;
          const std::uint64_t mask = evidence.alpha_covered[slot];
          nearest = std::min(nearest, covered > mask ? covered - mask : mask - covered);
        }
        return nearest;
      };
      const bool dedicated_mask = disagreement(0) != UINT64_MAX;
      for (std::uint32_t slot = 0; slot < 4; ++slot) {
        if (!(evidence.candidates & (1u << slot)) || evidence.alpha_invalid[slot]) continue;
        const std::uint32_t channel = source_of(slot, evidence.ui_layer);
        const std::uint32_t bit = 1u << channel;
        const std::uint64_t covered = evidence.alpha_covered[slot];
        // A tagged UI color covering the whole frame while an exact HUD-less
        // pair shows most of the scene is the final image, not UI coverage.
        if (channel == 1 && covered * 100 >= total * 99 && exact_pair &&
            std::uint64_t(evidence.hudless_unchanged) * 2 >= total) {
          if (opaque_.add(tick_ms)) ui_color_opaque_ = true;
        } else if (channel == 1 && covered && covered * 10 < total * 9) {
          opaque_ = {};
          ui_color_opaque_ = false;
        }
        // Presented alpha (2 Backbuffer, 3 current) differing from every such
        // mask by at least 10% of the frame is not UI coverage: it earns nothing,
        // and the usual evidence interval revokes its trust.
        const auto revoke = [&] {
          trusted_alpha_ &= ~bit;
          provisional_ &= ~bit;
          doubt_[channel] = {};
        };
        if ((provisional_ & bit) && !reconfirm_from_[channel]) reconfirm_from_[channel] = tick_ms;
        if ((channel == 2 || channel == 3) && dedicated_mask && disagreement(covered) * 10 >= total) {
          earned_[channel] = {};
          if ((trusted_alpha_ & bit) && doubt_[channel].add(tick_ms)) revoke();
          continue;
        }
        if (covered && covered * 10 < total * 9) {
          doubt_[channel] = {};
          if (earned_[channel].add(tick_ms, covered)) {
            trusted_alpha_ |= bit;
            provisional_ &= ~bit;
          }
        } else if (covered * 10 >= total * 9) {
          earned_[channel] = {};
          if ((trusted_alpha_ & bit) && scene_visible && doubt_[channel].add(tick_ms)) revoke();
        }
        if ((provisional_ & bit) && tick_ms >= reconfirm_from_[channel] &&
            tick_ms - reconfirm_from_[channel] >= alpha_trust_reconfirm_ms) {
          earned_[channel] = {};
          revoke();
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
    std::array<evidence_run, source_count> earned_{}, doubt_{};
    std::uint32_t trusted_alpha_{};
    // Remembered bits not yet earned again this session, and when each such
    // source was first offered.
    std::uint32_t provisional_{};
    std::array<std::uint64_t, source_count> reconfirm_from_{};
    // The tagged UI color covered the whole frame over a visible scene.
    evidence_run opaque_{};
    bool ui_color_opaque_{};
    bool first_run_{};
    std::function<void(std::uint32_t)> trust_listener_;
  };
}
