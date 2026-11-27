// SPDX-License-Identifier: GPL-3.0-only
// The UI decision framework's CPU policy bodies (docs/reshade-sbs.md, UI
// decision framework): the acceptance ledger (alpha_auto_policy), the
// hidden-scene guard (scene_guard) and the T1 decoders and hold state
// (ui_temporal). Their headers own the contracts and call order; this file
// holds only the definitions. Pure C++: no ReShade, Windows or GPU
// dependency, so the replays and policy tests link it alone.
#include "game3d_alpha_auto.h"
#include "game3d_scene_guard.h"
#include "game3d_ui_temporal.h"

namespace sunshine_game3d {
  void alpha_auto_policy::set_manual(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    manual_ = enabled;
  }

  void alpha_auto_policy::set_automatic() {
    std::lock_guard<std::mutex> lock(mutex_);
    manual_.reset();
  }

  alpha_auto_decision alpha_auto_policy::decision() {
    std::lock_guard<std::mutex> lock(mutex_);
    alpha_auto_decision result;
    if (manual_) {
      result.enabled = *manual_;
      result.state = *manual_ ? alpha_auto_state::manual_on : alpha_auto_state::manual_off;
    }
    return result;
  }

  std::uint32_t alpha_auto_policy::accepted(std::uint32_t offered, const candidate_signatures &signatures) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (manual_) return *manual_ ? offered & ui_selection::candidate_bits : 0u;
    std::uint32_t result = 0;
    for (const auto k : ui_selection::draw_order)
      if ((offered & ui_selection::bit(k)) && accepts_locked(signatures.of(k))) result |= ui_selection::bit(k);
    return result;
  }

  bool alpha_auto_policy::accepts(const ui_selection::signature &signature) {
    std::lock_guard<std::mutex> lock(mutex_);
    return accepts_locked(signature);
  }

  bool alpha_auto_policy::pre_ui_proven(const ui_selection::signature &layer) {
    std::lock_guard<std::mutex> lock(mutex_);
    return accepts_locked(ui_selection::pre_ui_key(layer));
  }

  void alpha_auto_policy::observe(const alpha_auto_decision::detection_evidence &evidence, std::uint32_t pixels, std::uint64_t tick_ms,
      const candidate_signatures &signatures) {
    if (!pixels) return;
    std::string changed_to;
    std::function<void(const std::string &)> listener;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (manual_) return;
      // accept() and revoke() mark a change of the accepted set, so a sample
      // that changes nothing builds no key list.
      accepted_changed_ = false;
      observe_locked(evidence, pixels, tick_ms, signatures);
      if (!accepted_changed_ || !change_listener_) return;
      changed_to = stored_locked();
      listener = change_listener_;
    }
    // Outside the lock: the owner may persist the new set.
    listener(changed_to);
  }

  alpha_auto_policy::restore_result alpha_auto_policy::restore(std::string_view stored) {
    std::lock_guard<std::mutex> lock(mutex_);
    restore_result result;
    std::size_t seen = 0;
    while (!stored.empty()) {
      const auto end = stored.find_first_of(std::string_view(",\0", 2));
      const auto text = stored.substr(0, end);
      stored.remove_prefix(end == std::string_view::npos ? stored.size() : end + 1);
      const auto parsed = ui_selection::signature::parse(text);
      if (!parsed) continue;
      if (seen == max_stored_ui_sources) break;
      ++seen;
      auto &entry = at(*parsed);
      if (entry.accepted) continue;
      entry.accepted = entry.provisional = true;
      entry.last_testable = entry.testable_ms = 0;
      ++result.restored;
      ++counters_[ui_counter::trust_restored];
    }
    return result;
  }

  std::string alpha_auto_policy::stored() {
    std::lock_guard<std::mutex> lock(mutex_);
    return stored_locked();
  }

  bool alpha_auto_policy::has_stored() {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::any_of(entries_.begin(), entries_.end(), [](const entry &e) { return e.accepted; });
  }

  void alpha_auto_policy::on_change(std::function<void(const std::string &)> listener) {
    std::lock_guard<std::mutex> lock(mutex_);
    change_listener_ = std::move(listener);
  }

  std::string alpha_auto_policy::forget() {
    std::string cleared;
    std::function<void(const std::string &)> listener;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cleared = stored_locked();
      for (const auto &e : entries_)
        if (e.accepted) ++counters_[ui_counter::trust_forgotten];
      entries_.clear();
      if (cleared.empty() || !change_listener_) return cleared;
      listener = change_listener_;
    }
    listener(std::string());
    return cleared;
  }

  void alpha_auto_policy::add_counters(const ui_counters &delta) {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_ += delta;
  }

  ui_counters alpha_auto_policy::counters() {
    std::lock_guard<std::mutex> lock(mutex_);
    return counters_;
  }

  bool alpha_auto_policy::accepts_locked(const ui_selection::signature &signature) const {
    for (const auto &e : entries_)
      if (e.signature == signature) return e.accepted;
    return false;
  }

  alpha_auto_policy::entry *alpha_auto_policy::find(const ui_selection::signature &signature) {
    for (auto &e : entries_)
      if (e.signature == signature) return &e;
    return nullptr;
  }

  alpha_auto_policy::entry &alpha_auto_policy::at(const ui_selection::signature &signature) {
    if (auto *e = find(signature)) return *e;
    entries_.emplace_back();
    entries_.back().signature = signature;
    return entries_.back();
  }

  std::string alpha_auto_policy::stored_locked() const {
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

  void alpha_auto_policy::accept(entry &e) {
    if (!e.accepted || e.provisional) ++counters_[ui_counter::trust_earned];
    if (!e.accepted) accepted_changed_ = true;
    e.accepted = true;
    e.provisional = false;
  }

  void alpha_auto_policy::revoke(entry &e, std::size_t event) {
    if (e.accepted) accepted_changed_ = true;
    e.accepted = e.provisional = false;
    e.last_testable = e.testable_ms = 0;
    e.doubt = {};
    if (event == ui_counter::trust_revoked_exact) e.revoked_by_exact = true;
    ++counters_[event];
  }

  void alpha_auto_policy::reconfirm_tested(entry &e, std::uint64_t tick_ms) {
    if (!e.provisional) return;
    if (e.last_testable && tick_ms > e.last_testable)
      e.testable_ms += std::min(tick_ms - e.last_testable, alpha_trust_reconfirm_gap_ms);
    e.last_testable = tick_ms;
  }

  void alpha_auto_policy::lapse_if_due(entry &e) {
    if (e.provisional && e.testable_ms >= alpha_trust_reconfirm_ms) {
      e.earned = {};
      revoke(e, ui_counter::trust_lapsed);
    }
  }

  void alpha_auto_policy::observe_locked(const alpha_auto_decision::detection_evidence &evidence, std::uint32_t pixels, std::uint64_t tick_ms,
      const candidate_signatures &signatures) {
    using ui_selection::kind;
    namespace candidate = ui_detection::candidate;
    const auto offered = evidence.candidates;
    // V1 as the GPU judged it: the offered alpha candidates outside the
    // sample's valid bits (decision texel 7 .w).
    const auto invalid = offered & ui_selection::alpha_bits & ~evidence.valid_bits;
    // A void sample: an offered declared alpha is V1-invalid, so the
    // declared-coverage judge of Backbuffer and current alpha is missing.
    // It voids only those two kinds (voided below); every other source
    // meets the same judges as on any sample and earns as usual.
    const bool void_sample = (offered & invalid & ui_selection::declared_alpha_bits) != 0;
    // A2 judges of this sample: a V2-valid exact change set, and the
    // coverage of every offered, accepted, V1-valid declared alpha.
    const bool exact_judge = ui_selection::exact_judge(offered, evidence.valid_bits);
    std::array<std::uint32_t, 2> declared{};
    std::size_t declared_count = 0;
    for (const auto k : {kind::ui_alpha, kind::ui_color})
      if ((offered & ui_selection::bit(k)) && !(invalid & ui_selection::bit(k)) && accepts_locked(signatures.of(k)))
        declared[declared_count++] = alpha_counts_of(evidence, k).covered;
    for (const auto k : ui_selection::draw_order) {
      if (!ui_selection::alpha_kind(k) || !(offered & ui_selection::bit(k))) continue;
      // A3: an alpha offered but invalid can neither earn nor be refuted,
      // so it is not testable and its reconfirm clock does not count.
      if (invalid & ui_selection::bit(k)) continue;
      auto &e = at(signatures.of(k));
      const auto covered = alpha_counts_of(evidence, k).covered;
      // A2: every judged kind (all alpha but the one-frame-late layer copy,
      // E2) meets the one-way test, with strong pixels to test; inferred
      // alpha also the declared alphas' coverage.
      const bool judged = ui_selection::judged(k);
      // A void sample leaves Backbuffer and current alpha as they are: no
      // run advances or restarts, and since it cannot earn them it is not
      // testable for them either (A3).
      const bool voided = void_sample && judged && !ui_selection::declared(k);
      // A3: neither can a sample opaque almost everywhere (a full menu).
      if (!voided && !ui_selection::full(covered, pixels)) reconfirm_tested(e, tick_ms);
      const auto [strong, contradicted] = one_way_of(evidence, k);
      const bool exact_basis = judged && exact_judge && strong != 0;
      const bool declared_basis = judged && !ui_selection::declared(k) && declared_count != 0;
      if (exact_basis || declared_basis) {
        const bool by_exact = exact_basis && ui_selection::one_way_contradicted(strong, contradicted);
        bool by_declared = declared_basis;
        for (std::size_t j = 0; j != declared_count; ++j)
          by_declared = by_declared && ui_selection::coverage_disagrees(declared[j], covered, pixels);
        if (by_exact || by_declared) {
          // Not UI coverage: it earns nothing, and repeated contradiction
          // revokes it.
          if (!voided) e.earned = {};
          if (e.accepted && e.doubt.add(tick_ms))
            revoke(e, by_exact ? ui_counter::trust_revoked_exact : ui_counter::trust_revoked_declared);
          lapse_if_due(e);
          continue;
        }
      }
      if (k == kind::ui_layer && ui_selection::selective(covered, pixels) &&
          ui_selection::layer_uncovered_color(evidence.pre_ui_image_lit, covered, evidence.layer_opaque)) {
        // A layer whose colour without coverage outnumbers its opaque
        // pixels is a scene image in the cleared target, not UI: it earns
        // nothing and restarts the run, as a full sample does.
        e.earned = {};
        ++counters_[ui_counter::trust_uncovered_color];
      } else if (ui_selection::selective(covered, pixels)) {
        // An inferred run is consecutive and bounded: a selective sample
        // more than alpha_trust_span_ms after the run's last one restarts it.
        // A declared alpha the one-way test revoked earns again only from a
        // sample that judged it (exact_basis, not contradicted: above).
        if (!voided && (ui_selection::declared(k) ? !e.revoked_by_exact || exact_basis :
              e.earned.add(tick_ms, covered, alpha_trust_span_ms)))
          accept(e);
      } else if (ui_selection::full(covered, pixels)) {
        if (!voided) e.earned = {};
      }
      lapse_if_due(e);
    }
    // H1 (d): the offered layer's pre-UI proof, testable while the
    // presented frame's evidence is valid and visible.
    if ((offered & candidate::layer) && !evidence.layer_covered) {
      auto &e = at(ui_selection::pre_ui_key(signatures.of(kind::ui_layer)));
      if (evidence.scene.valid && evidence.scene.verdict == ui_detection::scene_verdict::visible) reconfirm_tested(e, tick_ms);
      if (ui_selection::pre_ui_match(evidence.pre_ui_match, evidence.pre_ui_image_lit, pixels) && e.earned.add(tick_ms))
        accept(e);
      lapse_if_due(e);
    }
    // A HUD-less change set is declared: it earns from its first V2-valid
    // partial change set, exact or not, since V2's tile test is the proof of
    // its pixels (a game may tag only HUDLessColor, whose pairs are then all
    // counted and inexact: Hogwarts Legacy 10-05). Only an exact pair judges
    // (A2). A3: its testable samples are those that could earn or refute
    // it, offered V2-valid and not full (a partial or empty change set), as
    // an alpha's are; a full menu or a mispaired (V2-invalid) sample is not.
    if (offered & candidate::hudless) {
      auto &e = at(signatures.of(kind::hudless));
      // The GPU's V2 verdict (valid bits): a valid set is partial (some
      // changed pixels, fewer than a quarter), empty or, only from an exact
      // pair, full (98%), so a valid selective one is the partial change
      // set (ui_selection::change_set_selective).
      const bool valid = (evidence.valid_bits & candidate::hudless) != 0u;
      if (valid && !ui_selection::full(evidence.hudless_changed, pixels)) reconfirm_tested(e, tick_ms);
      if (valid && ui_selection::selective(evidence.hudless_changed, pixels)) accept(e);
      lapse_if_due(e);
    }
  }
}

namespace sunshine_game3d::scene_guard {
  bool run_hold::hit(std::uint64_t tick) {
    bool entered = false;
    if (held(tick)) until = tick + ui_detection::scene::hold_ms;
    else if (last && tick >= last && tick - last <= ui_detection::scene::hold_ms) {
      until = tick + ui_detection::scene::hold_ms;
      entered = true;
    }
    last = tick;
    return entered;
  }

  void state::enter_scope(std::uint64_t new_epoch, std::uint32_t new_viewport) {
    if (scoped && new_epoch == epoch && new_viewport == viewport) return;
    *this = state{};
    scoped = true;
    epoch = new_epoch;
    viewport = new_viewport;
  }

  bool state::refuted(const ui_selection::signature &signature) const {
    for (std::size_t i = 0; i != refuted_count; ++i)
      if (refutations[i] == signature) return true;
    return false;
  }

  std::uint32_t state::refuted_bits(std::uint32_t offered, const kind_signatures &signatures) const {
    std::uint32_t bits = 0;
    for (const auto k : ui_selection::draw_order)
      if ((offered & ui_selection::bit(k)) && refuted(signatures[std::size_t(k)])) bits |= ui_selection::bit(k);
    return bits;
  }

  std::uint32_t state::per_frame(std::uint64_t now, std::uint32_t offered, const kind_signatures &signatures,
      bool layer_proven) const {
    const bool layer_offered = (offered & ui_detection::candidate::layer) != 0u;
    const bool proven = layer_offered && layer_proven;
    const bool layer_image = ui_selection::pre_ui_image_of(offered) == ui_detection::pre_ui_image::layer;
    const bool pre_ui_visible = pre_ui.held(now) && (!layer_image || proven);
    return (hidden.held(now) ? ui_detection::per_frame_scene_hidden : 0u) |
      (pre_ui_visible ? ui_detection::per_frame_pre_ui_visible : 0u) |
      (refuted_bits(offered, signatures) << ui_detection::per_frame_refuted_shift) |
      (proven ? ui_detection::per_frame_pre_ui_proven : 0u);
  }

  bool state::measure(std::uint64_t now, bool proven_image) const {
    return gate_open || hidden.held(now) || pre_ui.held(now) || proven_image;
  }

  observation state::observe(const sample &s, bool actionable, const kind_signatures &signatures) {
    observation result;
    // A refuted candidate offered, valid and below 99% opaque (or an exact
    // pair without a full change set) is an overlay again.
    for (const auto k : ui_selection::draw_order) {
      const auto b = ui_selection::bit(k);
      if (!(s.offered & s.valid_bits & b)) continue;
      const bool overlay = ui_selection::alpha_kind(k) ?
        !ui_selection::opaque_full(s.opaque[ui_selection::alpha_index(k)], s.pixels) :
        (s.offered & ui_detection::candidate::exact) && !(s.claims & b);
      if (overlay) restore(signatures[std::size_t(k)]);
    }
    const auto &presented = s.presented;
    if (actionable && presented.valid) {
      if (presented.verdict == scene_verdict::visible) {
        result.released = hidden.held(s.tick) || pre_ui.held(s.tick) || s.source == 8u;
        hidden.release();
        pre_ui.release();
        for (const auto k : ui_selection::draw_order)
          if ((s.claims & ui_selection::bit(k)) && refute(signatures[std::size_t(k)])) ++result.refuted;
      } else if (presented.verdict == scene_verdict::hidden) {
        result.entered = hidden.hit(s.tick);
        // The pre-UI image's own evidence, read only for the image the
        // claim names (a layer's claim exists only while it is proven);
        // invalid evidence changes nothing.
        if ((s.claims & ui_detection::claim_pre_ui) && s.pre_ui_image &&
            s.pre_ui_image == ui_selection::pre_ui_image_of(s.offered) && s.pre_ui.valid) {
          if (ui_detection::scene_visible(s.pre_ui.d)) pre_ui.hit(s.tick);
          else pre_ui.release();
        }
      } else {
        hidden.miss();
        pre_ui.miss();
      }
    }
    gate_open = (s.claims & ui_selection::candidate_bits & ~refuted_bits(s.offered, signatures)) != 0u ||
      (s.claims & ui_detection::claim_pre_ui) != 0u;
    return result;
  }

  bool state::refute(const ui_selection::signature &signature) {
    if (refuted(signature)) return false;
    if (refuted_count == max_refuted) {
      for (std::size_t i = 1; i != max_refuted; ++i) refutations[i - 1] = refutations[i];
      --refuted_count;
    }
    refutations[refuted_count++] = signature;
    return true;
  }

  void state::restore(const ui_selection::signature &signature) {
    for (std::size_t i = 0; i != refuted_count; ++i)
      if (refutations[i] == signature) {
        for (std::size_t j = i + 1; j != refuted_count; ++j) refutations[j - 1] = refutations[j];
        --refuted_count;
        return;
      }
  }
}

namespace sunshine_game3d::ui_temporal {
  bool scope_changed(const alpha_auto_source &a, const alpha_auto_source &b) {
    return a.epoch != b.epoch || a.revision != b.revision || a.viewport != b.viewport;
  }

  bool hold_scope_changed(const alpha_auto_source &a, const alpha_auto_source &b) {
    return a.epoch != b.epoch || a.viewport != b.viewport;
  }

  bool sample_discarded(const alpha_auto_source &input, const alpha_auto_source &pending) {
    return scope_changed(input, pending) || input.now_ms < pending.now_ms || input.now_ms - pending.now_ms > sample_fresh_ms;
  }

  alpha_auto_decision decode_detection_sample(const std::uint32_t *words, std::size_t count,
      std::uint64_t tick_ms, std::uint64_t sequence) {
    namespace word = ui_detection::decision_word;
    alpha_auto_decision sample;
    if (count < 4 * ui_detection::decision_texels) return sample;
    sample.source_kind = words[word::source];
    sample.enabled = words[word::source] != 0;
    sample.state = words[word::source] ? alpha_auto_state::automatic_on : alpha_auto_state::automatic_off;
    // The pixel counts are the ones the shader's CPU oracle decides from
    // (ui_selection::counts_from_words): one reading of the words serves both,
    // so the shipped decoder and the oracle cannot come to read them differently.
    const auto counts = ui_selection::counts_from_words(words, count);
    using ui_selection::alpha_index;
    using ui_selection::kind;
    // The evidence keeps the four presented-frame alphas in judged_kinds order.
    const auto alpha_kinds = [](const std::array<std::uint32_t, 5> &by_kind) {
      std::array<std::uint32_t, 4> result{};
      for (std::size_t i = 0; i != result.size(); ++i) result[i] = by_kind[alpha_index(ui_selection::judged_kinds[i])];
      return result;
    };
    sample.covered = words[word::covered]; sample.pixels = counts.pixels;
    sample.sample_sequence = sequence;
    sample.sample_tick_ms = tick_ms;
    auto &evidence = sample.evidence;
    evidence.matching_tiles = counts.matching_tiles;
    evidence.candidates = words[word::candidates]; evidence.hudless_changed = counts.changed;
    evidence.hudless_unchanged = counts.unchanged; evidence.hudless_invalid = counts.nonfinite;
    evidence.alpha_covered = alpha_kinds(counts.covered);
    evidence.alpha_invalid = alpha_kinds(counts.invalid);
    evidence.hudless_lit = counts.lit;
    evidence.accepted = words[word::accepted];
    const auto opaque = alpha_kinds(counts.opaque);
    evidence.alpha_opaque = {opaque[0], opaque[1]};
    evidence.layer_covered = counts.covered[alpha_index(kind::ui_layer)];
    evidence.layer_invalid = counts.invalid[alpha_index(kind::ui_layer)];
    evidence.layer_opaque = counts.opaque[alpha_index(kind::ui_layer)];
    evidence.valid_bits = words[word::valid_bits];
    evidence.strong = counts.strong;
    evidence.contradicted = counts.contradicted;
    evidence.refused = words[word::refused];
    evidence.frame_reason = words[word::frame_reason] & ui_detection::frame_reason_reason_mask;
    evidence.reused = (words[word::frame_reason] & ui_detection::frame_reason_reused) != 0;
    evidence.inferred_opaque = {opaque[2], opaque[3]};
    evidence.claims = words[word::claims];
    evidence.s1_source = words[word::h1] & ui_detection::h1_winner_mask;
    evidence.h1_applied = (words[word::h1] & ui_detection::h1_applied) != 0;
    evidence.pre_ui_match = counts.pre_ui_match;
    evidence.pre_ui_image_lit = counts.pre_ui_lit;
    const auto decode = [&](std::size_t n, std::size_t d, std::size_t state) {
      alpha_auto_decision::scene_evidence result;
      result.n = words[n];
      std::memcpy(&result.d, &words[d], sizeof(result.d));
      result.valid = ui_detection::scene_state_valid(words[state]);
      result.ran = ui_detection::scene_state_ran(words[state]);
      result.verdict = ui_detection::scene_state_verdict(words[state]);
      return result;
    };
    evidence.scene = decode(word::scene_n, word::scene_d, word::scene_state);
    evidence.scene.decided = words[word::scene_decided];
    // The pre-UI image's texel carries no verdict: the CPU reads it from D.
    evidence.pre_ui_scene = decode(word::pre_ui_scene_n, word::pre_ui_scene_d, word::pre_ui_scene_state);
    evidence.pre_ui_scene.verdict = evidence.pre_ui_scene.valid ?
      ui_detection::scene_verdict_of(evidence.pre_ui_scene.d) : ui_detection::scene_verdict::none;
    evidence.pre_ui_image = words[word::pre_ui_scene_image];
    return sample;
  }

  scene_guard::sample guard_sample(const alpha_auto_decision &decoded) {
    const auto &evidence = decoded.evidence;
    scene_guard::sample s;
    s.tick = decoded.sample_tick_ms;
    s.pixels = decoded.pixels;
    s.offered = evidence.candidates;
    s.valid_bits = evidence.valid_bits;
    s.source = decoded.source_kind;
    s.claims = evidence.claims;
    using ui_selection::alpha_index;
    using ui_selection::kind;
    s.opaque[alpha_index(kind::ui_alpha)] = evidence.alpha_opaque[0];
    s.opaque[alpha_index(kind::ui_color)] = evidence.alpha_opaque[1];
    s.opaque[alpha_index(kind::ui_layer)] = evidence.layer_opaque;
    s.opaque[alpha_index(kind::backbuffer)] = evidence.inferred_opaque[0];
    s.opaque[alpha_index(kind::current)] = evidence.inferred_opaque[1];
    const auto image = [](const alpha_auto_decision::scene_evidence &e) {
      scene_guard::image_evidence result;
      result.d = e.d;
      result.valid = e.valid;
      result.verdict = e.verdict;
      return result;
    };
    s.presented = image(evidence.scene);
    s.pre_ui = image(evidence.pre_ui_scene);
    s.pre_ui_image = evidence.pre_ui_image;
    return s;
  }

  ui_counters sample_counters(const alpha_auto_decision &sample, const ui_counters &cpu,
      const ui_counters &committed_cpu, const std::array<std::uint32_t, ui_counter_word::count> &words,
      const std::array<std::uint32_t, ui_counter_word::count> &committed_words, std::uint64_t through_ms,
      const scene_guard::observation &observed) {
    auto delta = cpu - committed_cpu;
    delta.add_gpu_delta(words, committed_words);
    if (sample.source_kind == 8) {
      using ui_detection::scene_verdict;
      const auto &scene = sample.evidence.scene;
      const auto verdict = scene.valid ? scene.verdict : scene_verdict::none;
      ++delta[ui_counter::full_d_hidden + (verdict == scene_verdict::hidden ? 0 : verdict == scene_verdict::ambiguous ? 1 :
        verdict == scene_verdict::visible ? 2 : 3)];
    }
    static_assert(ui_counter::full_d_ambiguous == ui_counter::full_d_hidden + 1 &&
      ui_counter::full_d_visible == ui_counter::full_d_hidden + 2 && ui_counter::full_d_invalid == ui_counter::full_d_hidden + 3);
    delta[ui_counter::scene_entered] = observed.entered;
    delta[ui_counter::scene_released] = observed.released;
    delta[ui_counter::scene_refuted] = observed.refuted;
    delta[ui_counter::samples] = 1;
    delta.through_ms = through_ms;
    return delta;
  }

  const char *name(hold_kind value) {
    switch (value) {
      case hold_kind::generated: return "generated";
      case hold_kind::unavailable: return "unavailable";
      default: return "none";
    }
  }

  hold_decision detection_state::arbitrate(const present_identity &identity, const alpha_auto_source &scope, std::uint32_t offered) const {
    hold_decision result;
    const bool same_scope = have_decision && !hold_scope_changed(scope, decision_scope);
    if (identity.generated) {
      result.hold = same_scope;
      result.kind = result.hold ? hold_kind::generated : hold_kind::unavailable;
      return result;
    }
    result.detect = true;
    if (!same_scope) result.per_frame |= ui_detection::per_frame_hold_reset;
    else if (bits & accepted & ui_selection::candidate_bits & ~offered)
      result.per_frame |= ui_detection::per_frame_accepted_missing;
    result.adopt = !(result.per_frame & ui_detection::per_frame_accepted_missing);
    return result;
  }

  void detection_state::adopt(std::uint32_t new_bits, std::uint32_t new_accepted) {
    bits = new_bits;
    accepted = new_accepted;
  }

  bool detection_state::status_fresh(const alpha_auto_source &observation) const {
    return !sample_stale(latest.sample_tick_ms, observation.now_ms) && latest_key == status_key() &&
      !scope_changed(observation, latest_source);
  }

  void detection_state::enter_scope(const alpha_auto_source &observation) {
    if (scope_changed(observation, latest_source)) latest = {};
  }

  void detection_state::detected(const alpha_auto_source &scope) {
    have_decision = true;
    decision_scope = scope;
    holds = 0;
  }

  void detection_state::unavailable() {
    have_decision = false;
    holds = 0;
  }
}
