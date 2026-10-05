// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Exact per-session UI protection counters (docs/reshade-sbs.md, UI counters).
// The detection reduce (SunshineUIDetectionReduceCS, thread 0) accumulates the
// GPU words below on every detection frame into a small R32_UINT texture; the
// renderer copies it next to the decision texels on sample frames and commits
// the uint32 wrap-safe deltas, with its own CPU counts snapshotted when that
// sample was submitted, to the game session (alpha_auto_policy::add_counters),
// which also counts trust events. Totals are therefore exact per frame
// through the last committed sample (through_ms): frames rendered after it,
// at most one sample interval plus the pending copy, are counted by a later
// commit, or lost when their renderer is destroyed first. At every commit
//   auto_frames == detection_frames + held.generated + held.none + inactive.
// game3d_native.hlsl mirrors the GPU word indices as SUNSHINE_UI_COUNTER_*
// defines and test_game3d_ui_layer fails when they disagree. This header owns
// the log text after "Sunshine UI counters: ". No ReShade dependency.
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace sunshine_game3d {
  // Why a detection frame decided no mask (source 0). Exactly one reason per
  // such frame, the first that applies in this order (ui_selection::decide):
  // an informative full claim acted but H1 did not apply, because the CPU
  // holds no hidden verdict or the depth is not this frame's (gate_no_hold,
  // the name kept for log compatibility); an offered, accepted declared alpha blocked an accepted,
  // valid inferred alpha (presented_blocked: every such declared alpha was
  // V1-invalid, since a valid one decides); an offered, accepted alpha was
  // V1-invalid (trusted_invalid); the offered offscreen UI
  // layer was V1-invalid (layer_aside); an offered, unaccepted candidate was
  // valid and selective, so its acceptance is still being earned
  // (unaccepted); a HUD-less image was offered (difference_failed); an
  // unaccepted valid alpha was empty or nearly full (ambiguous); no alpha was
  // offered (no_candidate); otherwise other. The indices are the GPU word
  // order, not that priority.
  namespace ui_no_mask {
    inline constexpr std::size_t layer_aside = 0, trusted_invalid = 1, presented_blocked = 2, ambiguous = 3,
      difference_failed = 4, gate_no_hold = 5, no_candidate = 6, other = 7, unaccepted = 8, count = 9;
    inline constexpr std::array<std::string_view, count> names{"layer_aside", "trusted_invalid", "presented_blocked",
      "ambiguous", "difference_failed", "gate_no_hold", "no_candidate", "other", "unaccepted"};
  }

  // Words of the GPU counter texture, each a uint32 total over this renderer's
  // detection frames.
  namespace ui_counter_word {
    inline constexpr std::size_t detection_frames = 0;
    // One word per decided source 0-12 (7, 9, 11 and 12 are retired and stay
    // zero; 11, H2's still screen since fix 2, and 12, fix 3's pre-UI change
    // set, each shifted every later word by one and were removed; their
    // words stay reserved so that no later word moves again).
    inline constexpr std::size_t decided = 1, decided_count = 13;
    // An inferred source (the offscreen UI layer 10, Backbuffer 3 or current
    // 4) decided without being accepted. Zero by construction since S1, where
    // only accepted candidates decide; kept as an invariant.
    inline constexpr std::size_t untrusted_inferred = 14;
    // A HUD-less difference (5) from an inexact pair.
    inline constexpr std::size_t inexact_difference = 15;
    // The consumed depth was not this frame's (per_frame_depth_not_current).
    inline constexpr std::size_t depth_not_current = 16;
    // The frame's own decision was an accepted inferred alpha (Backbuffer 3,
    // current 4 or the offscreen UI layer 10) that the same frame's valid
    // exact pair contradicts in the one-way test (A2): at least a tenth of its
    // pixels with alpha of at least 1/2 lie where the HUD-less image is lit
    // and unchanged. The acceptance ledger revokes it from such samples.
    inline constexpr std::size_t contradicted = 17;
    // An inferred alpha (3, 4, 10) decided while an accepted declared alpha
    // (UIAlpha or the UI color tag) was offered. Zero by construction since
    // S1 (the declared-alpha block); kept as an invariant.
    inline constexpr std::size_t presented_over_dedicated = 18;
    // One word per ui_no_mask reason.
    inline constexpr std::size_t none = 19;
    // An alpha source (1-4, 10) was applied covering at least 99% of pixels:
    // a whole-frame mask from alpha, whatever the HUD-less pair showed.
    inline constexpr std::size_t full_alpha = none + ui_no_mask::count;
    // The T1 grace: a real frame without a decision of its own (an accepted
    // candidate missing or invalid) applied the previous real frame's
    // decision, once, from the GPU hold store.
    inline constexpr std::size_t reused = full_alpha + 1;
    // Word 30 is reserved and stays zero: fix 3's refined count (removed by
    // user decision); kept so that every later word keeps its index.
    inline constexpr std::size_t reserved_refined = reused + 1;
    // count: the words every counting shader writes (SUNSHINE_UI_COUNTER_WORDS).
    // Words 31-35 (S3's identity verdict counts, removed with S3's frame
    // identity) are reserved: no shader writes them.
    inline constexpr std::size_t count = reserved_refined + 1;
  }
  static_assert(ui_counter_word::decided + ui_counter_word::decided_count == ui_counter_word::untrusted_inferred &&
    ui_counter_word::full_alpha == 28 && ui_counter_word::reused == 29 && ui_counter_word::reserved_refined == 30 &&
    ui_counter_word::count == 31);
  // The game3d_native.hlsl define mirroring each GPU word index.
  inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 20> hlsl_counter_defines{{
    {"SUNSHINE_UI_COUNTER_WORDS", std::uint32_t(ui_counter_word::count)},
    {"SUNSHINE_UI_COUNTER_DETECTION_FRAMES", std::uint32_t(ui_counter_word::detection_frames)},
    {"SUNSHINE_UI_COUNTER_DECIDED", std::uint32_t(ui_counter_word::decided)},
    {"SUNSHINE_UI_COUNTER_UNTRUSTED_INFERRED", std::uint32_t(ui_counter_word::untrusted_inferred)},
    {"SUNSHINE_UI_COUNTER_INEXACT_DIFFERENCE", std::uint32_t(ui_counter_word::inexact_difference)},
    {"SUNSHINE_UI_COUNTER_DEPTH_NOT_CURRENT", std::uint32_t(ui_counter_word::depth_not_current)},
    {"SUNSHINE_UI_COUNTER_CONTRADICTED", std::uint32_t(ui_counter_word::contradicted)},
    {"SUNSHINE_UI_COUNTER_PRESENTED_OVER_DEDICATED", std::uint32_t(ui_counter_word::presented_over_dedicated)},
    {"SUNSHINE_UI_COUNTER_NONE", std::uint32_t(ui_counter_word::none)},
    {"SUNSHINE_UI_COUNTER_FULL_ALPHA", std::uint32_t(ui_counter_word::full_alpha)},
    {"SUNSHINE_UI_COUNTER_REUSED", std::uint32_t(ui_counter_word::reused)},
    {"SUNSHINE_UI_NONE_LAYER_ASIDE", std::uint32_t(ui_no_mask::layer_aside)},
    {"SUNSHINE_UI_NONE_TRUSTED_INVALID", std::uint32_t(ui_no_mask::trusted_invalid)},
    {"SUNSHINE_UI_NONE_PRESENTED_BLOCKED", std::uint32_t(ui_no_mask::presented_blocked)},
    {"SUNSHINE_UI_NONE_AMBIGUOUS", std::uint32_t(ui_no_mask::ambiguous)},
    {"SUNSHINE_UI_NONE_DIFFERENCE_FAILED", std::uint32_t(ui_no_mask::difference_failed)},
    {"SUNSHINE_UI_NONE_GATE_NO_HOLD", std::uint32_t(ui_no_mask::gate_no_hold)},
    {"SUNSHINE_UI_NONE_NO_CANDIDATE", std::uint32_t(ui_no_mask::no_candidate)},
    {"SUNSHINE_UI_NONE_OTHER", std::uint32_t(ui_no_mask::other)},
    {"SUNSHINE_UI_NONE_UNACCEPTED", std::uint32_t(ui_no_mask::unaccepted)},
  }};

  // Indices of ui_counters::value. Renderer CPU counts: renders that requested
  // GPU detection (Auto, or an explicit HUD-less difference input), the
  // Presents without detection of their own by T1 (ui_temporal::hold_kind):
  // generated Presents that applied the decision of the real frame they show
  // (held.generated) and those with no mask because no such decision exists
  // in their scope (held.none), and those without detection by reason: no
  // usable candidate bits, a frame larger than 3840, or detection resources
  // that could not be prepared. reused counts detection frames whose GPU
  // reduce applied the previous real frame's decision (the T1 grace); they
  // are detection frames, not holds.
  // full_d: per committed sample that decided H1 (source 8, a full-frame UI
  // over a hidden scene), the hidden-scene verdict that same sample
  // measured: invalid when the evidence was not valid or not measured.
  // full_alpha_d: the same for a sample that decided an accepted whole-frame
  // decision (an alpha covering at least 99%, or the exact full change set
  // 6), measured by a diagnostic evidence run that was removed with the
  // first-run shadow: reserved, they stay zero and are still logged. scene: the
  // hidden-scene guard's observations of committed samples
  // (game3d_scene_guard.h): its hidden hold entered, a hold released by a
  // visible sample (every visible H1 sample is one), and source signatures
  // newly refuted. These are sample counts, not frame counts. The still
  // group (rule H2's still screens, fix 2) was removed with that rule.
  // Trust events are the session acceptance ledger's (alpha_auto_policy):
  // signatures earned, revoked (A2) by an exact change set's one-way test or
  // by a declared alpha's coverage, lapsed, restored from an earlier session,
  // legacy TrustedUISources entries discarded, and accepted signatures the
  // overlay's Forget cleared.
  namespace ui_counter {
    inline constexpr std::size_t auto_frames = 0, detection_frames = 1;
    inline constexpr std::size_t held_generated = 2, held_none = 3, reused = 4;
    inline constexpr std::size_t inactive_no_candidates = 5, inactive_size = 6, inactive_unprepared = 7;
    inline constexpr std::size_t decided = 8; // Thirteen counters, sources 0-12 (7, 9, 11 and 12 retired).
    inline constexpr std::size_t none = decided + ui_counter_word::decided_count; // ui_no_mask order.
    inline constexpr std::size_t depth_not_current = none + ui_no_mask::count;
    inline constexpr std::size_t full_d_hidden = depth_not_current + 1, full_d_ambiguous = full_d_hidden + 1,
      full_d_visible = full_d_hidden + 2, full_d_invalid = full_d_hidden + 3;
    inline constexpr std::size_t untrusted_inferred = full_d_invalid + 1, inexact_difference = untrusted_inferred + 1,
      contradicted = untrusted_inferred + 2, presented_over_dedicated = untrusted_inferred + 3;
    inline constexpr std::size_t full_alpha = presented_over_dedicated + 1, full_alpha_d_hidden = full_alpha + 1,
      full_alpha_d_ambiguous = full_alpha + 2, full_alpha_d_visible = full_alpha + 3, full_alpha_d_invalid = full_alpha + 4;
    inline constexpr std::size_t trust_earned = full_alpha_d_invalid + 1, trust_revoked_exact = trust_earned + 1,
      trust_revoked_declared = trust_earned + 2, trust_lapsed = trust_earned + 3, trust_restored = trust_earned + 4,
      trust_discarded = trust_earned + 5, trust_forgotten = trust_earned + 6;
    inline constexpr std::size_t samples = trust_forgotten + 1;
    inline constexpr std::size_t scene_entered = samples + 1, scene_released = samples + 2, scene_refuted = samples + 3;
    inline constexpr std::size_t count = scene_refuted + 1;
  }

  struct ui_counters {
    std::array<std::uint64_t, ui_counter::count> value{};
    // The sample tick (ms) the totals are exact through; the latest of every
    // commit added.
    std::uint64_t through_ms{};

    std::uint64_t &operator[](std::size_t index) { return value[index]; }
    std::uint64_t operator[](std::size_t index) const { return value[index]; }
    std::uint64_t decided(std::uint32_t source) const {
      return source < ui_counter_word::decided_count ? value[ui_counter::decided + source] : 0;
    }
    // Presents without detection of their own (T1).
    std::uint64_t held() const { return value[ui_counter::held_generated] + value[ui_counter::held_none]; }
    std::uint64_t inactive() const {
      return value[ui_counter::inactive_no_candidates] + value[ui_counter::inactive_size] +
        value[ui_counter::inactive_unprepared];
    }
    // The accounting identity every commit preserves.
    bool reconciled() const { return value[ui_counter::auto_frames] == value[ui_counter::detection_frames] + held() + inactive(); }

    ui_counters &operator+=(const ui_counters &other) {
      for (std::size_t i = 0; i != value.size(); ++i) value[i] += other.value[i];
      if (other.through_ms > through_ms) through_ms = other.through_ms;
      return *this;
    }
    // Field-wise difference of two snapshots of the same running totals.
    friend ui_counters operator-(const ui_counters &later, const ui_counters &earlier) {
      ui_counters result;
      for (std::size_t i = 0; i != result.value.size(); ++i) result.value[i] = later.value[i] - earlier.value[i];
      result.through_ms = later.through_ms;
      return result;
    }
    // Adds the uint32 wrap-safe change of the GPU words since an earlier read.
    void add_gpu_delta(const std::array<std::uint32_t, ui_counter_word::count> &now,
        const std::array<std::uint32_t, ui_counter_word::count> &before) {
      const auto delta = [&](std::size_t word) { return std::uint64_t(std::uint32_t(now[word] - before[word])); };
      value[ui_counter::detection_frames] += delta(ui_counter_word::detection_frames);
      for (std::size_t s = 0; s != ui_counter_word::decided_count; ++s)
        value[ui_counter::decided + s] += delta(ui_counter_word::decided + s);
      for (std::size_t r = 0; r != ui_no_mask::count; ++r) value[ui_counter::none + r] += delta(ui_counter_word::none + r);
      value[ui_counter::depth_not_current] += delta(ui_counter_word::depth_not_current);
      value[ui_counter::untrusted_inferred] += delta(ui_counter_word::untrusted_inferred);
      value[ui_counter::inexact_difference] += delta(ui_counter_word::inexact_difference);
      value[ui_counter::contradicted] += delta(ui_counter_word::contradicted);
      value[ui_counter::presented_over_dedicated] += delta(ui_counter_word::presented_over_dedicated);
      value[ui_counter::full_alpha] += delta(ui_counter_word::full_alpha);
      value[ui_counter::reused] += delta(ui_counter_word::reused);
    }
  };

  // The log text after "Sunshine UI counters: " (docs/reshade-sbs.md, UI
  // counters): space-separated key=value fields, groups as key={key=value ...}.
  inline std::string format_ui_counters(const ui_counters &c) {
    namespace n = ui_counter;
    std::string text;
    const auto field = [&](std::string_view key, std::uint64_t value) {
      if (!text.empty() && text.back() != '{') text += ' ';
      text.append(key).append("=").append(std::to_string(value));
    };
    const auto group = [&](std::string_view key, auto &&fields) {
      if (!text.empty()) text += ' ';
      text.append(key).append("={");
      fields();
      text += '}';
    };
    field("auto_frames", c[n::auto_frames]);
    field("detection_frames", c[n::detection_frames]);
    group("held", [&] {
      field("generated", c[n::held_generated]);
      field("none", c[n::held_none]);
    });
    field("reused", c[n::reused]);
    group("inactive", [&] {
      field("no_candidates", c[n::inactive_no_candidates]);
      field("size", c[n::inactive_size]);
      field("unprepared", c[n::inactive_unprepared]);
    });
    group("decided", [&] {
      for (std::uint32_t source = 0; source != ui_counter_word::decided_count; ++source)
        if (source != 7 && source != 9 && source != 11 && source != 12) field(std::to_string(source), c.decided(source));
    });
    group("none", [&] {
      for (std::size_t reason = 0; reason != ui_no_mask::count; ++reason) field(ui_no_mask::names[reason], c[n::none + reason]);
    });
    group("full", [&] {
      field("6", c.decided(6));
      field("8", c.decided(8));
      field("depth_not_current", c[n::depth_not_current]);
    });
    group("full_d", [&] {
      field("hidden", c[n::full_d_hidden]);
      field("ambiguous", c[n::full_d_ambiguous]);
      field("visible", c[n::full_d_visible]);
      field("invalid", c[n::full_d_invalid]);
    });
    group("scene", [&] {
      field("entered", c[n::scene_entered]);
      field("released", c[n::scene_released]);
      field("refuted", c[n::scene_refuted]);
    });
    field("untrusted_inferred", c[n::untrusted_inferred]);
    field("inexact_difference", c[n::inexact_difference]);
    field("contradicted", c[n::contradicted]);
    field("presented_over_dedicated", c[n::presented_over_dedicated]);
    field("full_alpha", c[n::full_alpha]);
    group("full_alpha_d", [&] {
      field("hidden", c[n::full_alpha_d_hidden]);
      field("ambiguous", c[n::full_alpha_d_ambiguous]);
      field("visible", c[n::full_alpha_d_visible]);
      field("invalid", c[n::full_alpha_d_invalid]);
    });
    group("trust", [&] {
      field("earned", c[n::trust_earned]);
      field("revoked_exact", c[n::trust_revoked_exact]);
      field("revoked_declared", c[n::trust_revoked_declared]);
      field("lapsed", c[n::trust_lapsed]);
      field("restored", c[n::trust_restored]);
      field("discarded", c[n::trust_discarded]);
      field("forgotten", c[n::trust_forgotten]);
    });
    field("samples", c[n::samples]);
    field("through_ms", c.through_ms);
    return text;
  }
}
