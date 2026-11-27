// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "game3d_ui_temporal.h"
#include "ui_evidence.h"
#include <type_traits>

namespace sunshine_game3d {
  // One renderer's CPU UI decisions; GPU resources and submission stay in the renderer.
  // docs/reshade-sbs.md, UI decision framework, owns the policy and call order.
  class ui_decision {
  public:
    struct requested_mode {
      alpha_auto_state mode;
      bool auto_mode, manual_detection;
    };
    struct detection_policy { bool needs, requested, runs; };
    static requested_mode request(const alpha_auto_source *automatic, bool has_detection);
    static detection_policy detection_request(bool auto_mode, bool manual_detection, bool hudless,
      const alpha_auto_source *automatic, alpha_auto_state mode, const ui_temporal::hold_decision &arbitration,
      uint32_t bits);
    uint32_t accepted(const alpha_auto_source *automatic, uint32_t bits, const candidate_signatures &signatures);
    ui_temporal::hold_decision arbitrate(bool hold_previous, const alpha_auto_source &observation, uint32_t bits) const;
    void adopt(const ui_temporal::hold_decision &arbitration, uint32_t bits, uint32_t accepted);
    // A resumed renderer (renderer::resume) starts as a new one: no T1 state,
    // held verdicts, scene bits, actionable sample or consumed decision.
    void resume_scope();
    void begin_frame(bool detection_requested);
    void scene_frame(const alpha_auto_source &observation, uint32_t bits, const candidate_signatures &signatures,
      const alpha_auto_source *automatic);
    void held();
    void enter_scope(const alpha_auto_source &observation);
    uint32_t per_frame(bool hudless_reoffer, uint32_t bits, uint32_t unaligned_declared,
      const ui_temporal::hold_decision &arbitration, bool depth_current) const;
    void detected(const alpha_auto_source &observation);
    void unavailable();
    void inactive(bool detection_requested, std::size_t inactive_reason);
    bool sample_discarded(const alpha_auto_source &input) const;
    // A completed sample's readback: cleared first (an unread sample leaves it
    // empty), then decoded by observe_sample.
    void clear_sample() { temporal.latest = {}; }
    const alpha_auto_decision &latest_sample() const { return temporal.latest; }
    scene_guard::observation observe_sample(const uint32_t *counts, std::size_t count, uint64_t detection_submitted,
      alpha_auto_policy *session);
    void commit_counters(const alpha_auto_decision &sample, alpha_auto_policy *session,
      const scene_guard::observation &observed, const std::array<uint32_t, ui_counter_word::count> &words);
    // Whether the sample this frame submits runs the hidden-scene evidence
    // passes: only where they can run (measurable) and when the guard
    // measures this frame; the sample's evidence is actionable only then.
    bool measure_scene(bool measurable, uint64_t now, bool proven_image);
    // The sample this frame submits: the CPU counts its counter readback
    // commits (pend_counts, when it reads counters back), and its source,
    // status winner and acceptance signatures (pend_sample).
    void pend_counts() { pending_counts = cpu_counts; }
    void pend_sample(const alpha_auto_source &observation, const candidate_signatures &signatures);
    void update_alpha_auto(bool eligible, const alpha_auto_source *automatic, bool detection_active, bool &source_alpha_ui);
    // What this render consumed (renderer::consumed_alpha_auto), whether the
    // offered layer's signature is proven the pre-UI scene image, and the
    // generated Presents in a row that held the real frame's decision.
    const alpha_auto_decision &consumed() const { return consumed_auto; }
    bool layer_proven() const { return scene_layer_proven; }
    uint32_t held_presents() const { return temporal.holds; }

  private:
    alpha_auto_decision consumed_auto;
    // This render's scene guard bits and whether the offered layer's
    // signature is proven the pre-UI scene image by the session ledger.
    uint32_t scene_bits{};
    bool scene_layer_proven{};
    // Adopted inputs, T1 state and latest status sample; GPU grace stays in
    // the renderer's hold-store texture.
    ui_temporal::detection_state temporal;
    // Held D verdicts and refuted signatures; only identity changes clear it.
    scene_guard::state guard;
    // The pending sample ran the actionable evidence passes.
    bool detection_pending_actionable{};
    // Status winner and acceptance signatures at pending-sample submission.
    uint32_t detection_pending_status_key{};
    candidate_signatures detection_pending_signatures;
    alpha_auto_source detection_pending_source;
    ui_counters cpu_counts, pending_counts, committed_counts;
    std::array<uint32_t, ui_counter_word::count> committed_words{};
  };
  static_assert(std::is_trivially_destructible_v<ui_decision>);
}
