// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "game3d_scene_guard.h"
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
  // selective detection samples spanning this long (A1), and a layer
  // signature its pre-UI proof by this many matching samples (H1 d); an
  // accepted source is revoked by this many contradicting samples within this
  // long (A2) (docs/reshade-sbs.md, UI decision framework).
  inline constexpr std::uint32_t alpha_trust_samples = 3;
  inline constexpr std::uint64_t alpha_trust_span_ms = 2000;
  // Acceptance remembered from an earlier session protects from the first
  // frame but lapses unless this session earns it again within this long of
  // testable time (A3): the time between consecutive samples that could earn
  // or refute it, each gap counted up to alpha_trust_reconfirm_gap_ms (two
  // and a half sample intervals; samples are at most ten a second), so time
  // while the source is not testable never counts beyond one capped gap.
  inline constexpr std::uint64_t alpha_trust_reconfirm_ms = 60000;
  inline constexpr std::uint64_t alpha_trust_reconfirm_gap_ms = 250;
  // At most this many remembered acceptance entries are restored.
  inline constexpr std::size_t max_stored_ui_sources = 32;

  class alpha_auto_policy;
  struct alpha_auto_source {
    std::uint64_t now_ms{}, epoch{}, revision{}, sequence{}, tick_ms{};
    std::uint32_t viewport{};
    bool retained{};
    alpha_auto_policy *session{}; // Caller-owned game session; never owned by a renderer.
  };

  enum class alpha_auto_state { waiting_for_source, collecting, automatic_on, automatic_off, manual_on, manual_off };

  inline const char *name(alpha_auto_state value) {
    switch (value) {
      case alpha_auto_state::waiting_for_source: return "searching";
      case alpha_auto_state::automatic_on: return "detected";
      case alpha_auto_state::automatic_off: return "no_usable_mask";
      case alpha_auto_state::manual_on: return "manual_on";
      case alpha_auto_state::manual_off: return "manual_off";
      default: return "checking";
    }
  }

  struct alpha_auto_decision {
    bool enabled{};
    alpha_auto_state state = alpha_auto_state::waiting_for_source;
    std::uint32_t covered{}, pixels{};
    std::uint64_t sample_sequence{}, sample_tick_ms{};
    // Latest completed diagnostic: 1 UI R, 2 UI color tag A, 3 backbuffer A, 4
    // current A, 5 HUD-less difference, 6 full-frame UI (HUD-less differs
    // almost everywhere), 8 full-frame UI over a hidden scene (H1), 10 the
    // offscreen UI layer A. 7, 9 and 11 are retired and never reused.
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
      // Pixels of UIAlpha and the UI color tag with alpha of at least 254/255,
      // and of Backbuffer and current alpha (inferred_opaque, texel 10).
      std::array<std::uint32_t, 2> alpha_opaque{}, inferred_opaque{};
      // The presented frame's evidence (texel 5) and the pre-UI scene
      // image's (texel 6), whose verdict the CPU reads from D, and the image
      // texel 6 measured (ui_detection::pre_ui_image).
      scene_evidence scene, pre_ui_scene;
      std::uint32_t pre_ui_image{};
      // H1, texel 10 (selection revision 3): the raw informative full claims
      // before refutation (candidate bits | ui_detection::claim_pre_ui), the
      // S1 winner's source, and whether H1 overrode it with source 8.
      std::uint32_t claims{}, s1_source{};
      bool h1_applied{};
      // A2, decision texels 8, 9 and 11 (selection revision 11), in
      // ui_selection::judged_kinds order (UIAlpha, UI color tag, Backbuffer,
      // current): pixels with alpha of at least 1/2, and those of them where
      // an offered exact pair's HUD-less image is lit and unchanged against
      // both the pair's colour and the presented frame, counted on sample
      // frames with an exact pair. The one-frame-late layer copy is
      // never judged (E2).
      std::array<std::uint32_t, 4> strong{}, contradicted{};
      // F1: the frame's own decision's refused candidate bit (zero when it
      // decided) and its reason (a ui_no_mask index, or
      // ui_detection::frame_reason_decided when it decided a source or the
      // sample has no texel 9), and whether the T1 grace applied the previous
      // real frame's decision instead.
      std::uint32_t refused{}, frame_reason = ui_detection::frame_reason_decided;
      bool reused{};
      // H1 (d), texel 11 (selection revision 4): the offscreen UI layer
      // against the presented frame at 8 times their pair threshold, pixels
      // whose colours match, and the layer's own lit pixels (8 times its own
      // step), which prove the layer the pre-UI scene image
      // (ui_selection::pre_ui_match). Both zero without a layer; matching
      // zero without a comparable pair (lit is counted there since 10-23).
      std::uint32_t pre_ui_match{}, pre_ui_image_lit{};
    } evidence;
    // This render's state, not the sample's: the hidden-scene guard's
    // verdicts pushed with this render (game3d_scene_guard.h: a held hidden
    // verdict, its samples reading the pre-UI image visible) and how many
    // source signatures a visible verdict refuted, whether the offered layer's
    // signature is proven the pre-UI scene image (H1 d: the ledger's key
    // pre_ui:<format>:<space>, alpha_auto_policy::pre_ui_proven).
    struct scene_guard_state {
      bool hidden{}, pre_ui{};
      std::uint32_t refuted{};
      bool proven{};
    } scene_guard;
  };

  // One alpha candidate's covered and invalid pixels in a sample.
  struct alpha_counts {
    std::uint32_t covered{}, invalid{};
  };
  // The layer's are its own; the other alpha kinds' are in judged_kinds order.
  inline alpha_counts alpha_counts_of(const alpha_auto_decision::detection_evidence &evidence, ui_selection::kind k) {
    if (k == ui_selection::kind::ui_layer) return {evidence.layer_covered, evidence.layer_invalid};
    if (!ui_selection::judged(k)) return {};
    const auto i = ui_selection::judged_index(k);
    return {evidence.alpha_covered[i], evidence.alpha_invalid[i]};
  }
  // One judged alpha kind's one-way counts (A2) in a sample: strong and
  // contradicted pixels; zero for a kind that is not judged.
  inline std::pair<std::uint32_t, std::uint32_t> one_way_of(const alpha_auto_decision::detection_evidence &evidence,
      ui_selection::kind k) {
    if (!ui_selection::judged(k)) return {};
    const auto i = ui_selection::judged_index(k);
    return {evidence.strong[i], evidence.contradicted[i]};
  }

  // The acceptance signature of every candidate kind a detection offers
  // (A1): each kind's typed DXGI format and the swapchain color space. FG
  // mode is not part of it. Only offered kinds are read.
  struct candidate_signatures {
    std::array<std::uint32_t, ui_selection::kinds.size()> format{}; // ui_selection::kind order.
    std::uint32_t color_space{};

    ui_selection::signature of(ui_selection::kind k) const { return {k, format[std::size_t(k)], color_space}; }
    // Every kind's signature, as the hidden-scene guard reads them.
    scene_guard::kind_signatures by_kind() const {
      scene_guard::kind_signatures result{};
      for (std::size_t i = 0; i != result.size(); ++i) result[i] = of(ui_selection::kind(i));
      return result;
    }
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
  // per source signature that those samples earn, and per offscreen layer
  // signature the proof that it holds the pre-UI scene image (H1 d, the
  // ledger-only kind ui_selection::kind::pre_ui). It knows nothing about
  // slots, holds or FG mode, and reads the presented frame's D verdict only
  // as the gate of that proof's reconfirm clock.
  class alpha_auto_policy {
  public:
    void set_manual(bool enabled);

    void set_automatic();

    // The mode alone: manual On or Off, or Auto before a rendered frame
    // reports its current-frame detection.
    alpha_auto_decision decision();

    // The offered candidates (ui_detection::candidate bits) this session
    // accepts, in candidate-bit positions: in Auto those whose signature the
    // ledger accepts, provisional ones included; manual On accepts every
    // offered candidate for this session only (S2) and manual Off none,
    // without touching the ledger.
    std::uint32_t accepted(std::uint32_t offered, const candidate_signatures &signatures);

    // Whether the ledger accepts this signature (provisional included).
    bool accepts(const ui_selection::signature &signature);

    // H1 (d): whether an offscreen layer of this signature is proven the
    // pre-UI scene image (its key ui_selection::pre_ui_key is accepted,
    // provisional included). Manual modes neither earn nor lapse the proof
    // but honour it.
    bool pre_ui_proven(const ui_selection::signature &layer);

    // One completed GPU detection sample over `pixels` pixels, taken for
    // candidates of these signatures (those at submission). Manual modes
    // never earn or revoke. A1 earning: a declared source (UIAlpha, the UI
    // color tag, or a HUD-less pair, exact or not, whose partial change set
    // passes V2) is accepted by its first valid selective sample; an inferred
    // source (the offscreen UI
    // layer, Backbuffer or current alpha) by alpha_trust_samples valid
    // selective samples spanning alpha_trust_span_ms within a factor of two of
    // each other, consecutive among its selective samples: a full (at least
    // 90%) sample, or a gap of more than alpha_trust_span_ms since the run's
    // last selective sample, restarts the run, while empty samples within
    // that bound stay neutral. Alpha with more than 1% invalid pixels (V1,
    // the GPU's valid bits) is no evidence, and a selective layer sample
    // whose colour without coverage outnumbers its opaque pixels
    // (ui_selection::layer_uncovered_color: V1's comparison without its 1%
    // allowance) earns nothing and restarts the layer's run, as a full sample
    // does; it is testable (A3). A sample in which an offered
    // declared alpha is V1-invalid is void for Backbuffer and current alpha,
    // the kinds whose declared-coverage judge it lacks: their runs neither
    // advance nor restart, and it is not testable for them (Resident Evil
    // Requiem's rejected-tag frames). The other declared tag, the layer and
    // the HUD-less pair meet the same judges as on any sample and earn as
    // usual, so a declared tag that is always invalid never stops them.
    // A2 revocation, by evidence of stronger provenance in the same sample,
    // whatever the drawing rank: every offered, V1-valid alpha but the
    // one-frame-late layer copy (UIAlpha, the UI color tag, Backbuffer,
    // current; ui_selection::judged_kinds) is judged by a V2-valid exact
    // change set (ui_selection::exact_judge; acceptance not required), which
    // contradicts it when at least a tenth of its pixels with alpha of at
    // least 1/2 lie where the HUD-less image is lit and unchanged against
    // both the pair's colour and the presented frame (the one-way test; real
    // UI, also UI composited after the tagged Backbuffer, and dims and tints
    // over dark or changed pixels never meet it, an opaque final image read
    // as UI does), and an inferred one
    // (Backbuffer, current) also by every offered, accepted, V1-valid
    // declared alpha, which contradict it when its coverage differs from each
    // of theirs by at least a tenth of the frame (not batch-aligned: the
    // newest offered tag, which may be another frame's). The one-way judge needs a
    // basis: a source without strong pixels in the sample is not judged by
    // it. The layer copy is not same-sample evidence (E2) and is never
    // judged.
    // An accepted source is revoked by alpha_trust_samples contradicting
    // judged samples within alpha_trust_span_ms ("3 contradictions within
    // 2 s"); agreeing and unjudged samples change nothing, and invalid or
    // ambiguous samples never revoke. A contradicted sample earns nothing
    // and restarts the earning run (unless void). For the rest of the session
    // a declared alpha that the one-way test revoked is accepted again only
    // by a selective sample that judged it without contradicting it, never by
    // an unjudged one. A3: restored entries lapse
    // unless earned again within alpha_trust_reconfirm_ms of testable time.
    // A testable sample could earn or refute the source: offered, valid (V1,
    // or V2 for a HUD-less pair) and not full (below 90% coverage, or a
    // partial or empty change set). The clock counts only the time between
    // consecutive testable samples of the entry, each gap capped at
    // alpha_trust_reconfirm_gap_ms, so an invalid run, a long full-screen
    // menu, samples of another signature or kind, a manual period or a time
    // without samples never count beyond one capped gap.
    // H1 (d), fix 1: an offered layer without coverage (layer_covered zero)
    // earns its signature's pre-UI proof (key ui_selection::pre_ui_key) like
    // an inferred source: alpha_trust_samples samples spanning
    // alpha_trust_span_ms whose layer equals the presented frame
    // (ui_selection::pre_ui_match), not necessarily consecutive. A mismatch
    // never withdraws it and never restarts the run (UI over the scene is a
    // mismatch); no A2 judge reads it and a void sample still counts. A
    // restored proof's reconfirm clock counts the same capped time between
    // testable samples (the layer offered without coverage while the
    // presented frame's evidence is valid and visible), so it lapses after
    // alpha_trust_reconfirm_ms of testable time without a match.
    void observe(const alpha_auto_decision::detection_evidence &evidence, std::uint32_t pixels, std::uint64_t tick_ms,
        const candidate_signatures &signatures);

    // Acceptance an earlier session of this game earned, as stored() wrote it
    // (a TrustedUISources value): keys separated by ',' or '\0', at most
    // max_stored_ui_sources, duplicates once. Restored entries decide from the
    // first frame and are revoked by the same contradiction, but they are
    // provisional (see observe). An entry that is not a signature key is
    // ignored. Never reported to the listener.
    struct restore_result {
      std::size_t restored{};
    };
    restore_result restore(std::string_view stored);

    // The accepted signatures (provisional included) as their keys, ','
    // separated in signature order: the TrustedUISources value.
    std::string stored();
    // Whether stored() names any signature, without building the value.
    bool has_stored();

    // Called with stored() whenever samples earn or revoke acceptance (never
    // in a manual mode) or forget() clears it; never for a restore.
    void on_change(std::function<void(const std::string &)> listener);

    // Forget (A3, the overlay's 'Forget learned UI sources'): clears every
    // entry of this game, accepted, provisional, earning and in doubt, counts
    // the accepted signatures cleared (trust.forgotten) and returns their
    // keys as stored() wrote them, empty when none was accepted. When that
    // changed stored(), the listener hears "" outside the lock, so the owner
    // persists an empty TrustedUISources. Each source is then accepted again
    // by its own evidence. The mode and the counters are unchanged.
    std::string forget();

    // Exact UI protection counters (game3d_ui_counters.h). Renderers commit
    // each completed sample's counts, at most ten times a second each; the
    // session counts its own acceptance events: a signature earned (or a
    // restored one confirmed) by this session's samples, revoked (A2) by an
    // exact change set's one-way test or by a declared alpha's coverage,
    // lapsed while provisional, restored, and accepted signatures forgotten.
    void add_counters(const ui_counters &delta);
    ui_counters counters();

  private:
    // Consecutive samples of one kind of evidence about one signature.
    struct evidence_run {
      std::uint32_t samples{};
      std::uint64_t first_tick_ms{}, last_tick_ms{}, low{}, high{};
      // True once alpha_trust_samples samples span alpha_trust_span_ms. A value
      // outside a factor of two of the run's others starts a new run with it,
      // and so does, when max_gap_ms is set, a sample more than max_gap_ms
      // after the run's last one.
      bool add(std::uint64_t tick_ms, std::uint64_t value = 0, std::uint64_t max_gap_ms = 0) {
        const auto lo = samples ? std::min(low, value) : value, hi = samples ? std::max(high, value) : value;
        if (!samples || tick_ms < last_tick_ms || hi > lo * 2 || (max_gap_ms && tick_ms - last_tick_ms > max_gap_ms))
          *this = {0, tick_ms, tick_ms, value, value};
        else { low = lo; high = hi; }
        ++samples;
        last_tick_ms = tick_ms;
        return samples >= alpha_trust_samples && tick_ms - first_tick_ms >= alpha_trust_span_ms;
      }
    };
    // The ticks of an accepted source's latest contradicting judged samples
    // (A2), oldest first.
    struct contradictions {
      std::array<std::uint64_t, alpha_trust_samples> ticks{};
      std::uint32_t count{};
      // True once alpha_trust_samples of them lie within alpha_trust_span_ms.
      // Older ones drop out; a tick before the newest starts again with it.
      bool add(std::uint64_t tick_ms) {
        if (count && tick_ms < ticks[count - 1]) count = 0;
        std::uint32_t kept = 0;
        for (std::uint32_t i = 0; i != count; ++i)
          if (tick_ms - ticks[i] <= alpha_trust_span_ms) ticks[kept++] = ticks[i];
        if (kept == ticks.size()) {
          std::move(ticks.begin() + 1, ticks.end(), ticks.begin());
          --kept;
        }
        ticks[kept] = tick_ms;
        count = kept + 1;
        return count >= alpha_trust_samples;
      }
    };
    struct entry {
      ui_selection::signature signature;
      bool accepted{}, provisional{};
      // A3, while provisional: the tick of the latest testable sample (zero
      // before the first) and the testable time counted so far.
      std::uint64_t last_testable{}, testable_ms{};
      evidence_run earned;
      contradictions doubt;
      // A2, for this session (Forget clears it): the exact pair's one-way test
      // revoked this entry. A declared alpha is then accepted again only by
      // a sample that judged it and did not contradict it, never by one
      // unjudged selective sample (an opaque final image read selective
      // during a fade over a dark, V2-invalid pair).
      bool revoked_by_exact{};
    };

    bool accepts_locked(const ui_selection::signature &signature) const;
    entry *find(const ui_selection::signature &signature);
    entry &at(const ui_selection::signature &signature);
    std::string stored_locked() const;

    void accept(entry &e);
    void revoke(entry &e, std::size_t event);
    // A3: a testable sample of a provisional entry counts the time since the
    // entry's previous testable sample, capped at alpha_trust_reconfirm_gap_ms.
    // Nothing else touches the clock, so time without testable samples never
    // counts beyond one capped gap.
    static void reconfirm_tested(entry &e, std::uint64_t tick_ms);
    // Revokes a provisional entry whose testable time reached
    // alpha_trust_reconfirm_ms, after the sample had its chance to earn.
    void lapse_if_due(entry &e);

    void observe_locked(const alpha_auto_decision::detection_evidence &evidence, std::uint32_t pixels, std::uint64_t tick_ms,
        const candidate_signatures &signatures);

    std::mutex mutex_;
    std::optional<bool> manual_; // Empty in Auto.
    std::vector<entry> entries_;
    std::function<void(const std::string &)> change_listener_;
    // Set by accept() and revoke() when an entry's acceptance flips; observe()
    // clears it before a sample and notifies the listener only when it is set.
    bool accepted_changed_{};
    ui_counters counters_;
  };
}
