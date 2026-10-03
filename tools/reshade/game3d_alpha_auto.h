// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "game3d_ui_counters.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_selection.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sunshine_game3d {
  // An inferred source earns acceptance as UI coverage by this many valid
  // selective detection samples spanning this long; doubt about an accepted
  // source takes the same (docs/reshade-sbs.md, UI decision framework, A1).
  inline constexpr std::uint32_t alpha_trust_samples = 3;
  inline constexpr std::uint64_t alpha_trust_span_ms = 2000;
  // Acceptance remembered from an earlier session protects from the first
  // frame but lapses unless this session earns it again within this long of
  // the source first being offered.
  inline constexpr std::uint64_t alpha_trust_reconfirm_ms = 60000;
  // At most this many remembered acceptance entries are restored.
  inline constexpr std::size_t max_stored_ui_sources = 32;

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

  // The detection inputs that decide a frame's mask (ui_selection::decide).
  // Candidate bits are ui_detection::candidate's; flags are the offscreen UI
  // layer's stored flags, which matter only while the layer (0x40) decides or
  // nothing accepted is offered. Only accepted candidates decide. With an
  // accepted alpha offered, the first in draw order keys the decision alone,
  // as it decides while V1-valid: a later accepted candidate decides only
  // while it is invalid, when the holds keep the previous inputs
  // (ui_temporal::detection_state::arbitrate). Which other candidates a
  // Present offers, accepted or not, and a HUD-less pairing beside it do not
  // change the key: frame generation pairs or tags them on some Presents only,
  // and a key that changed with them discarded nearly every sample while the
  // mask stayed applied (Resident Evil Requiem, Stellar Blade in SDR). With
  // only an accepted HUD-less pair, the key is the pair and its exactness,
  // which decides a full change set. Without an accepted candidate every
  // offered bit can matter.
  inline std::uint64_t detection_decision_key(std::uint32_t candidates, std::uint32_t flags, std::uint32_t accepted) {
    namespace candidate = ui_detection::candidate;
    const auto deciding = candidates & accepted & ui_selection::candidate_bits;
    if (!deciding) return std::uint64_t((candidates & candidate::layer) ? flags : 0u) << 32 | candidates;
    for (const auto k : ui_selection::draw_order) {
      const auto bit = ui_selection::bit(k);
      if (!ui_selection::alpha_kind(k) || !(deciding & bit)) continue;
      return std::uint64_t(k == ui_selection::kind::ui_layer ? flags : 0u) << 32 | std::uint64_t(bit) << 8;
    }
    return std::uint64_t(deciding | (candidates & candidate::exact)) << 8;
  }

  struct alpha_auto_decision {
    bool enabled{};
    alpha_auto_state state = alpha_auto_state::waiting_for_source;
    std::uint32_t covered{}, pixels{};
    std::uint64_t sample_sequence{}, sample_tick_ms{}, accepted_samples{};
    // Latest completed diagnostic: 1 UI R, 2 UI color tag A, 3 backbuffer A, 4
    // current A, 5 HUD-less difference, 6 full-frame UI (HUD-less differs
    // almost everywhere), 8 full-frame UI over a hidden scene by the layer
    // route, 9 by the HUD-less route, 10 the offscreen UI layer A. 7 is
    // retired and never reused.
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
    // Inputs of that GPU decision, for diagnosis and acceptance: the candidate
    // bits the shader was offered (ui_detection::candidate), the covered and
    // invalid pixels of the alpha candidates UIAlpha, UI color tag, Backbuffer
    // and current, and of the offscreen UI layer, HUD-less changed, unchanged
    // and non-finite pixels, tiles (of 256) whose pixels are 99% unchanged,
    // the accepted candidates pushed with the detection and the offered
    // candidates that passed V1/V2 (candidate-bit positions).
    struct detection_evidence {
      std::uint32_t candidates{}, hudless_changed{}, hudless_unchanged{}, hudless_invalid{}, matching_tiles{}, hudless_lit{};
      std::array<std::uint32_t, 4> alpha_covered{}, alpha_invalid{};
      std::uint32_t layer_covered{}, layer_invalid{}, layer_opaque{};
      std::uint32_t accepted{}, valid_bits{};
      // Pixels of UIAlpha and the UI color tag with alpha of at least 254/255.
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

  // One alpha candidate's covered and invalid pixels in a sample.
  struct alpha_counts {
    std::uint32_t covered{}, invalid{};
  };
  inline alpha_counts alpha_counts_of(const alpha_auto_decision::detection_evidence &evidence, ui_selection::kind k) {
    using ui_selection::kind;
    switch (k) {
      case kind::ui_alpha: return {evidence.alpha_covered[0], evidence.alpha_invalid[0]};
      case kind::ui_color: return {evidence.alpha_covered[1], evidence.alpha_invalid[1]};
      case kind::backbuffer: return {evidence.alpha_covered[2], evidence.alpha_invalid[2]};
      case kind::current: return {evidence.alpha_covered[3], evidence.alpha_invalid[3]};
      case kind::ui_layer: return {evidence.layer_covered, evidence.layer_invalid};
      default: return {};
    }
  }
  // The offered alpha candidates a sample over `pixels` read V1-invalid (more
  // than 1% invalid pixels), in candidate-bit positions.
  inline std::uint32_t invalid_alpha_bits(const alpha_auto_decision::detection_evidence &evidence, std::uint32_t pixels) {
    std::uint32_t bits = 0;
    for (const auto k : ui_selection::draw_order)
      if (ui_selection::alpha_kind(k) && (evidence.candidates & ui_selection::bit(k)) &&
          !ui_selection::alpha_valid(alpha_counts_of(evidence, k).invalid, pixels))
        bits |= ui_selection::bit(k);
    return bits;
  }

  // The acceptance signature of every candidate kind a detection offers
  // (A1): each kind's typed DXGI format and the swapchain color space. FG
  // mode is not part of it. Only offered kinds are read.
  struct candidate_signatures {
    std::array<std::uint32_t, ui_selection::kind_names.size()> format{}; // ui_selection::kind order.
    std::uint32_t color_space{};

    ui_selection::signature of(ui_selection::kind k) const { return {k, format[std::size_t(k)], color_space}; }
    candidate_signatures &set(ui_selection::kind k, std::uint32_t typed_format) {
      format[std::size_t(k)] = typed_format;
      return *this;
    }
    friend bool operator==(const candidate_signatures &a, const candidate_signatures &b) {
      return a.format == b.format && a.color_space == b.color_space;
    }
    friend bool operator!=(const candidate_signatures &a, const candidate_signatures &b) { return !(a == b); }
  };

  // UI protection mode and the acceptance ledger for one game process (M3,
  // docs/reshade-sbs.md, UI decision framework). Live Auto validates each
  // frame's inputs on the GPU; this holds only the mode and one accepted bit
  // per source signature that those samples earn. It knows nothing about
  // slots, holds, D or FG mode.
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

    // The offered candidates (ui_detection::candidate bits) this session
    // accepts, in candidate-bit positions: in Auto those whose signature the
    // ledger accepts, provisional ones included; manual On accepts every
    // offered candidate for this session only (S2) and manual Off none,
    // without touching the ledger.
    std::uint32_t accepted(std::uint32_t offered, const candidate_signatures &signatures) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (manual_) return *manual_ ? offered & ui_selection::candidate_bits : 0u;
      std::uint32_t result = 0;
      for (const auto k : ui_selection::draw_order)
        if ((offered & ui_selection::bit(k)) && accepts_locked(signatures.of(k))) result |= ui_selection::bit(k);
      return result;
    }

    // Whether the ledger accepts this signature (provisional included).
    bool accepts(const ui_selection::signature &signature) {
      std::lock_guard<std::mutex> lock(mutex_);
      return accepts_locked(signature);
    }

    // One completed GPU detection sample over `pixels` pixels, taken for
    // candidates of these signatures (those at submission). Manual modes
    // never earn or revoke. A1 earning: a declared source (UIAlpha, the UI
    // color tag, or a HUD-less pair from an exact sample only) is accepted by
    // its first valid selective sample; an inferred source (the offscreen UI
    // layer, Backbuffer or current alpha) by alpha_trust_samples valid
    // selective samples spanning alpha_trust_span_ms within a factor of two of
    // each other, where a full (at least 90%) sample restarts the run. Alpha
    // with more than 1% invalid pixels (V1) is no evidence. A sample in which
    // an offered declared alpha is V1-invalid is void for earning: no run
    // advances or restarts (Resident Evil Requiem's rejected-tag frames).
    // Revocation and the lapse (A2 and A3 as before S2a, per signature): an
    // accepted alpha covering at least 90% while an exact pair shows at least
    // 75% of the scene unchanged, or presented alpha (Backbuffer, current)
    // differing by at least 10% of the frame from every accepted, valid
    // UIAlpha, UI color tag or layer of the same sample, over the same
    // evidence interval; a selective sample clears doubt. Restored entries
    // lapse unless earned again within alpha_trust_reconfirm_ms of being
    // first offered valid.
    void observe(const alpha_auto_decision::detection_evidence &evidence, std::uint32_t pixels, std::uint64_t tick_ms,
        const candidate_signatures &signatures) {
      if (!pixels) return;
      std::string changed_to;
      std::function<void(const std::string &)> listener;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (manual_) return;
        const auto before = stored_locked();
        observe_locked(evidence, pixels, tick_ms, signatures);
        if (!change_listener_) return;
        changed_to = stored_locked();
        if (changed_to == before) return;
        listener = change_listener_;
      }
      // Outside the lock: the owner may persist the new set.
      listener(changed_to);
    }

    // Acceptance an earlier session of this game earned, as stored() wrote it
    // (a TrustedUISources value): keys separated by ',' or '\0', at most
    // max_stored_ui_sources, duplicates once. Restored entries decide from the
    // first frame and are revoked by the same contradiction, but they are
    // provisional (see observe). Anything that is not a signature key, such as
    // the integer bitmask of earlier versions, is discarded: each source is
    // accepted again by its own evidence. Never reported to the listener.
    struct restore_result {
      std::size_t restored{}, discarded{};
      std::string discarded_text; // The discarded entries, ',' separated.
    };
    restore_result restore(std::string_view stored) {
      std::lock_guard<std::mutex> lock(mutex_);
      restore_result result;
      std::size_t seen = 0;
      while (!stored.empty()) {
        const auto end = stored.find_first_of(std::string_view(",\0", 2));
        const auto text = stored.substr(0, end);
        stored.remove_prefix(end == std::string_view::npos ? stored.size() : end + 1);
        const auto parsed = ui_selection::signature::parse(text);
        if (!parsed) {
          if (text.find_first_not_of(" \t\r\n") == std::string_view::npos) continue;
          ++result.discarded;
          ++counters_[ui_counter::trust_discarded];
          if (!result.discarded_text.empty()) result.discarded_text += ',';
          result.discarded_text.append(text);
          continue;
        }
        if (seen == max_stored_ui_sources) break;
        ++seen;
        auto &entry = at(*parsed);
        if (entry.accepted) continue;
        entry.accepted = entry.provisional = true;
        entry.reconfirm_from = 0;
        ++result.restored;
        ++counters_[ui_counter::trust_restored];
      }
      return result;
    }

    // The accepted signatures (provisional included) as their keys, ','
    // separated in signature order: the TrustedUISources value.
    std::string stored() {
      std::lock_guard<std::mutex> lock(mutex_);
      return stored_locked();
    }

    // Called with stored() whenever samples earn or revoke acceptance, never
    // for a restore or in a manual mode.
    void on_change(std::function<void(const std::string &)> listener) {
      std::lock_guard<std::mutex> lock(mutex_);
      change_listener_ = std::move(listener);
    }

    // Exact UI protection counters (game3d_ui_counters.h). Renderers commit
    // each completed sample's counts, at most ten times a second each; the
    // session counts its own acceptance events: a signature earned (or a
    // restored one confirmed) by this session's samples, revoked by a full
    // claim over a visible scene or by presented alpha disagreeing with an
    // accepted dedicated channel, lapsed while provisional, restored, and
    // legacy entries discarded.
    void add_counters(const ui_counters &delta) {
      std::lock_guard<std::mutex> lock(mutex_);
      counters_ += delta;
    }
    ui_counters counters() {
      std::lock_guard<std::mutex> lock(mutex_);
      return counters_;
    }

    // A game whose session started without restored acceptance runs the
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

  private:
    // Consecutive samples of one kind of evidence about one signature.
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
    struct entry {
      ui_selection::signature signature;
      bool accepted{}, provisional{};
      // When a provisional entry's source was first offered valid; zero before.
      std::uint64_t reconfirm_from{};
      evidence_run earned, doubt;
    };

    bool accepts_locked(const ui_selection::signature &signature) const {
      for (const auto &e : entries_)
        if (e.signature == signature) return e.accepted;
      return false;
    }
    entry &at(const ui_selection::signature &signature) {
      for (auto &e : entries_)
        if (e.signature == signature) return e;
      entries_.emplace_back();
      entries_.back().signature = signature;
      return entries_.back();
    }
    std::string stored_locked() const {
      std::vector<ui_selection::signature> keys;
      for (const auto &e : entries_)
        if (e.accepted) keys.push_back(e.signature);
      std::sort(keys.begin(), keys.end());
      std::string text;
      for (const auto &key : keys) {
        if (!text.empty()) text += ',';
        text += key.key();
      }
      return text;
    }

    void accept(entry &e) {
      if (!e.accepted || e.provisional) ++counters_[ui_counter::trust_earned];
      e.accepted = true;
      e.provisional = false;
    }
    void revoke(entry &e, std::size_t event) {
      e.accepted = e.provisional = false;
      e.reconfirm_from = 0;
      e.doubt = {};
      ++counters_[event];
    }
    void lapse_if_due(entry &e, std::uint64_t tick_ms) {
      if (e.provisional && e.reconfirm_from && tick_ms >= e.reconfirm_from &&
          tick_ms - e.reconfirm_from >= alpha_trust_reconfirm_ms) {
        e.earned = {};
        revoke(e, ui_counter::trust_lapsed);
      }
    }

    void observe_locked(const alpha_auto_decision::detection_evidence &evidence, std::uint32_t pixels, std::uint64_t tick_ms,
        const candidate_signatures &signatures) {
      using ui_selection::kind;
      namespace candidate = ui_detection::candidate;
      const std::uint64_t total = pixels;
      const auto offered = evidence.candidates;
      const bool exact_pair = (offered & (candidate::hudless | candidate::exact)) == (candidate::hudless | candidate::exact) &&
        !evidence.hudless_invalid;
      const bool scene_visible = exact_pair && std::uint64_t(evidence.hudless_unchanged) * 100 >= total * 75;
      const auto invalid = invalid_alpha_bits(evidence, pixels);
      const bool void_sample = (offered & invalid & ui_selection::declared_alpha_bits) != 0;
      // An accepted dedicated UI source (UIAlpha, the UI color tag or the UI
      // layer) offered valid is the game's own UI mask for this sample, menus
      // included. disagreement(): the distance to the nearest such mask.
      std::array<std::uint64_t, 3> judges{};
      std::size_t judge_count = 0;
      for (const auto k : {kind::ui_alpha, kind::ui_color, kind::ui_layer})
        if ((offered & ui_selection::bit(k)) && !(invalid & ui_selection::bit(k)) && accepts_locked(signatures.of(k)))
          judges[judge_count++] = alpha_counts_of(evidence, k).covered;
      const auto disagreement = [&](std::uint64_t covered) {
        std::uint64_t nearest = UINT64_MAX;
        for (std::size_t i = 0; i != judge_count; ++i)
          nearest = std::min(nearest, covered > judges[i] ? covered - judges[i] : judges[i] - covered);
        return nearest;
      };
      for (const auto k : ui_selection::draw_order) {
        if (!ui_selection::alpha_kind(k) || !(offered & ui_selection::bit(k)) || (invalid & ui_selection::bit(k))) continue;
        auto &e = at(signatures.of(k));
        const std::uint64_t covered = alpha_counts_of(evidence, k).covered;
        if (e.provisional && !e.reconfirm_from) e.reconfirm_from = tick_ms;
        // Presented alpha differing from every such mask by at least 10% of the
        // frame is not UI coverage: it earns nothing, and the usual evidence
        // interval revokes it.
        if ((k == kind::backbuffer || k == kind::current) && judge_count && disagreement(covered) * 10 >= total) {
          if (!void_sample) e.earned = {};
          if (e.accepted && e.doubt.add(tick_ms)) revoke(e, ui_counter::trust_revoked_presented);
          continue;
        }
        if (ui_selection::selective(std::uint32_t(covered), pixels)) {
          e.doubt = {};
          if (!void_sample && (ui_selection::declared(k) || e.earned.add(tick_ms, covered))) accept(e);
        } else if (ui_selection::full(std::uint32_t(covered), pixels)) {
          if (!void_sample) e.earned = {};
          // A full claim while an exact pair shows most of the scene.
          if (e.accepted && scene_visible && e.doubt.add(tick_ms)) revoke(e, ui_counter::trust_revoked_full);
        }
        lapse_if_due(e, tick_ms);
      }
      // A HUD-less change set earns only from an exact pair (V2) whose partial
      // change set is valid; an inexact pair never earns.
      if (offered & candidate::hudless) {
        auto &e = at(signatures.of(kind::hudless));
        if (e.provisional && !e.reconfirm_from && !evidence.hudless_invalid) e.reconfirm_from = tick_ms;
        ui_selection::counts c;
        c.pixels = pixels;
        c.changed = evidence.hudless_changed;
        c.unchanged = evidence.hudless_unchanged;
        c.nonfinite = evidence.hudless_invalid;
        c.lit = evidence.hudless_lit;
        c.matching_tiles = evidence.matching_tiles;
        if (!void_sample && (offered & candidate::exact) && ui_selection::change_set_selective(c)) accept(e);
        lapse_if_due(e, tick_ms);
      }
    }

    std::mutex mutex_;
    std::optional<bool> manual_; // Empty in Auto.
    std::vector<entry> entries_;
    bool first_run_{};
    std::function<void(const std::string &)> change_listener_;
    ui_counters counters_;
  };
}
