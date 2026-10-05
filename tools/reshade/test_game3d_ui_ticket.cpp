// SPDX-License-Identifier: GPL-3.0-only
// The S3 snapshot ticket (game3d_ui_ticket.h): label arithmetic per boundary,
// the one pair rule, same-frame identity, real spans, refusal precedence, the
// FG interposer table, the identity shadow under DLSS-G ordering, T1 by
// ticket (game3d_ui_temporal.h) and the identity line's text.
#include "game3d_ui_temporal.h"
#include "game3d_ui_ticket.h"

#include <array>
#include <cstdio>
#include <iterator>
#include <stdexcept>
#include <string>

namespace ticket = sunshine_game3d::ui_ticket;
namespace temporal = sunshine_game3d::ui_temporal;
using sunshine_game3d::ui_selection::kind;

namespace {
  void require(bool value, const std::string &message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  namespace ip = ticket::interposer;
  namespace n = ticket::identity_counter;
  using ticket::boundary;
  using ticket::label_space;
  using ticket::refusal;

  void labels() {
    // before_clear = read; at_tag = read + 1; present = own label; 0 unstamped.
    require(ticket::present_label(boundary::before_clear, 41) == ticket::valid_label(label_space::present, 41), "A before-clear copy holds the frame of the Present it read");
    require(ticket::present_label(boundary::at_tag, 41) == ticket::valid_label(label_space::present, 42), "A tag snapshot belongs to the frame after the Present it read");
    require(ticket::present_label(boundary::present, 41, 43) == ticket::valid_label(label_space::present, 43), "A presented image carries its own label");
    for (const auto at : {boundary::before_clear, boundary::at_tag}) {
      const auto unstamped = ticket::present_label(at, 0);
      require(!unstamped.valid && unstamped.reason == refusal::unstamped && unstamped.space == label_space::present, "A read of 0 is unstamped: " + std::string(ticket::name(at)));
    }
    require(ticket::present_label(boundary::present, 41, 0).reason == refusal::unstamped, "A Present without a label is unstamped");
    require(!ticket::present_label(boundary::at_tag, 0xffffffffu).valid, "A wrapped at-tag label is unstamped");
    require(ticket::token_label_of_copy(7) == ticket::valid_label(label_space::token, 7) && !ticket::token_label_of_copy(0).valid, "Token label of a copy");
    require(ticket::token_label_of_tag(0x100000007ull).value == 7 && !ticket::token_label_of_tag(0x100000000ull).valid, "A token label keeps the low 32 bits of the generation");
    const auto refused = ticket::refuse(ticket::valid_label(label_space::present, 9), refusal::foreign_queue);
    require(!refused.valid && refused.reason == refusal::foreign_queue && refused.value == 9 && refused.space == label_space::present, "A refused label keeps its space and read");
  }

  void pair_rule() {
    const auto p9 = ticket::valid_label(label_space::present, 9), p10 = ticket::valid_label(label_space::present, 10);
    const auto t9 = ticket::valid_label(label_space::token, 9);
    require(ticket::pair_exact(p9, p9) && ticket::pair_exact(t9, t9), "Equal valid labels in one space pair");
    require(!ticket::pair_exact(p9, p10) && ticket::pair_refusal(p9, p10) == refusal::mismatch, "Different present labels");
    require(!ticket::pair_exact(p9, t9) && ticket::pair_refusal(p9, t9) == refusal::mismatch, "Labels of two spaces never pair");
    require(ticket::pair_refusal(p9, ticket::label {}) == refusal::no_reference, "No reference");
    require(ticket::pair_refusal(ticket::present_label(boundary::before_clear, 0), p9) == refusal::unstamped, "Unstamped evidence");
    require(ticket::pair_refusal(ticket::refuse(p9, refusal::not_real_span), p9) == refusal::not_real_span, "Evidence over a span that is not real");
    require(ticket::pair_refusal(p9, ticket::refuse(p9, refusal::foreign_queue)) == refusal::foreign_queue, "A refused reference");
    require(ticket::pair_refusal(p9, p9) == refusal::none, "An exact pair has no refusal");
    // The labels the provider's layer ticket expects: the Present its count
    // names (present label minus presents_since_copy), or the newest ready
    // Backbuffer token.
    require(ticket::pair_exact(ticket::present_label(boundary::before_clear, 9), ticket::valid_label(label_space::present, 10u - 1u)), "A copy of Present 9 pairs with the Present one before Present 10");
    require(ticket::pair_refusal(ticket::present_label(boundary::before_clear, 8), ticket::valid_label(label_space::present, 10u - 1u)) == refusal::mismatch, "A copy executed a Present earlier than the count is a mismatch");
    require(ticket::pair_exact(ticket::token_label_of_copy(77), ticket::valid_label(label_space::token, 77u)), "A layer copy after the tagged frame pairs by token");
  }

  ticket::ticket tag(kind source, std::uint64_t generation, std::uint64_t epoch = 7, std::uint32_t viewport = 1) {
    ticket::ticket t;
    t.kind = ticket::capture_kind::sl_tag;
    t.source = source;
    t.at = boundary::at_tag;
    t.token = ticket::token_label_of_tag(generation);
    t.token_generation = generation;
    t.epoch = epoch;
    t.viewport = viewport;
    t.typed_format = 24;
    t.color_space = 1;
    t.how = ticket::state_basis_proof(1);
    return t;
  }

  void frames_and_spans() {
    // Numbered and unnumbered tokens alike: one generation per token issue.
    require(ticket::same_frame(tag(kind::hudless, 300), tag(kind::backbuffer, 300)), "One token generation is one frame");
    require(!ticket::same_frame(tag(kind::hudless, 300), tag(kind::backbuffer, 301)), "Two token generations");
    require(!ticket::same_frame(tag(kind::hudless, 0), tag(kind::backbuffer, 0)), "No token is no frame");
    require(!ticket::same_frame(tag(kind::hudless, 300, 7), tag(kind::backbuffer, 300, 8)), "A frame across scopes");
    require(!ticket::same_frame(tag(kind::hudless, 300, 7, 1), tag(kind::backbuffer, 300, 7, 2)), "A frame across viewports");
    require(tag(kind::hudless, 3).key() == "hudless:24:srgb", "A ticket's signature key");
    std::array<ticket::ticket, ticket::slot::count> offered {};
    require(!ticket::newest_token(offered), "No offered token");
    offered[ticket::slot::backbuffer] = tag(kind::backbuffer, 41);
    offered[ticket::slot::hudless] = tag(kind::hudless, 42);
    offered[ticket::slot::ui_alpha] = tag(kind::ui_alpha, 0x7fff00000000ull);  // Low 32 bits zero: unstamped.
    require(ticket::newest_token(offered) == 42, "The newest valid token");
    require(ticket::slot_of_candidate(0) == ticket::slot::ui_alpha && ticket::slot_of_candidate(2) == ticket::slot::backbuffer && ticket::slot_of_candidate(3) == ticket::slot::hudless && ticket::slot_of_candidate(4) == ticket::slot::layer && ticket::slot_of_candidate(5) == ticket::slot::count, "Ticket slots of candidate slots");

    // real_span(fg_off_run, presents_ago, bits, fg_known, fg_enabled).
    require(ticket::real_span(3, 2, 0, true, false), "FG known off over the span");
    require(!ticket::real_span(2, 2, 0, true, false), "The run does not cover the labelled Present");
    require(!ticket::real_span(9, 1, ip::sl_interposer | ip::sl_dlss_g, true, true), "FG suspended (known and requested) is not real");
    require(ticket::real_span(9, 1, ip::sl_interposer | ip::sl_dlss_g, true, false), "DLSS-G loaded with FG known off");
    require(!ticket::real_span(9, 1, ip::sl_interposer | ip::fidelityfx_fg, true, false), "FidelityFX frame generation loaded");
    require(!ticket::real_span(9, 1, ip::xess_fg, true, false), "XeSS frame generation loaded");
    require(!ticket::real_span(9, 1, ip::sl_interposer, false, false), "FG unknown with sl.interposer");
  }

  void refusals() {
    ticket::refusal_inputs all;
    all.begin_refused = all.budget = all.render_pass = all.recording_not_covered = all.enhanced_barrier = true;
    all.stale_scope = all.foreign_queue = all.unstamped = all.not_real_span = all.no_reference = all.mismatch = true;
    const refusal order[] {refusal::begin_refused, refusal::budget, refusal::render_pass, refusal::recording_not_covered, refusal::enhanced_barrier, refusal::stale_scope, refusal::foreign_queue, refusal::unstamped, refusal::not_real_span, refusal::no_reference, refusal::mismatch};
    bool *const flags[] {&all.begin_refused, &all.budget, &all.render_pass, &all.recording_not_covered, &all.enhanced_barrier, &all.stale_scope, &all.foreign_queue, &all.unstamped, &all.not_real_span, &all.no_reference, &all.mismatch};
    for (std::size_t i = 0; i != std::size(order); ++i) {
      require(ticket::first_refusal(all) == order[i], "Refusal precedence at " + std::string(ticket::name(order[i])));
      *flags[i] = false;
    }
    require(ticket::first_refusal(all) == refusal::none, "No refusal");
    require(ticket::name(refusal::begin_refused) == "begin_refused" && ticket::name(ticket::begin_stage::no_reservation) == "no_reservation" && ticket::name(ticket::capture_kind::dump) == "dump" && ticket::name(boundary::before_clear) == "before_clear" && ticket::name(ticket::gpu_verdict::unproposed) == "unproposed" && ticket::name(refusal::count) == "unknown", "Names");
    require(ticket::state_basis_proof(1) == ticket::proof::sl_observed_legacy && ticket::state_basis_proof(2) == ticket::proof::sl_observed_enhanced && ticket::state_basis_proof(4) == ticket::proof::sl_contract && ticket::state_basis_proof(9) == ticket::proof::none, "State basis proofs");
  }

  void interposers() {
    require(ticket::interposer_bit(L"sl.interposer.dll") == ip::sl_interposer && ticket::interposer_bit(L"SL.Interposer.DLL") == ip::sl_interposer, "sl.interposer, case-insensitive");
    require(ticket::interposer_bit(L"sl.dlss_g.dll") == ip::sl_dlss_g && ticket::interposer_bit(L"NVNGX_DLSSG.dll") == ip::ngx_dlssg, "DLSS-G modules");
    for (const auto *name : {L"amd_fidelityfx_dx12.dll", L"amd_fidelityfx_framegeneration_dx12.dll", L"AMD_FidelityFX_Loader_DX12.dll", L"ffx_frameinterpolation_x64.dll", L"ffx_fsr3_x64.dll"}) {
      require(ticket::interposer_bit(name) == ip::fidelityfx_fg, "A FidelityFX frame generation module");
    }
    require(ticket::interposer_bit(L"libxess_fg.dll") == ip::xess_fg, "XeSS-FG");
    for (const auto *name : {L"", L"sl.interposer", L"libxess.dll", L"nvngx_dlss.dll", L"sl.interposer.dll.bak", L"dxgi.dll"}) {
      require(!ticket::interposer_bit(name), "An unknown module counted as an FG interposer");
    }
    require(ticket::interposer_names(0) == "none" && ticket::interposer_names(ip::sl_interposer | ip::xess_fg) == "sl_interposer,xess_fg", "Interposer names");
    require(ticket::format_interposers(3, true, false) == "bits=3 names=sl_interposer,sl_dlss_g fg_known=1 fg_enabled=0", "The interposer line");

    struct row {
      std::uint32_t bits;
      bool known, enabled, exact;
    };

    const row rows[] {
      {0, false, false, true},
      {0, true, true, false},
      {ip::sl_interposer, false, false, false},
      {ip::sl_interposer, true, false, true},
      {ip::sl_interposer | ip::fidelityfx_fg, true, false, false},
      {ip::ngx_dlssg, false, false, false},
    };
    for (const auto &r : rows) {
      require(ticket::present_time_exact(r.bits, r.known, r.enabled) == r.exact, "present_time_exact row " + std::to_string(r.bits));
    }
  }

  ticket::today_view today(bool detects, bool holds = false) {
    ticket::today_view v;
    v.offered = true;
    v.detects = detects;
    v.holds = holds;
    return v;
  }

  void shadow() {
    {
      // DLSS-G 2x: the generated Present carries a new token first; today
      // detects on the real one after it.
      ticket::identity_shadow s;
      ticket::identity_counters c;
      for (std::uint64_t token = 1; token <= 10; ++token) {
        const auto generated = s.step(today(false, true), token, false, false, c);
        const auto real = s.step(today(true), token, false, false, c);
        require(generated.fresh && !generated.hold_previous && !real.fresh && real.hold_previous && real.real_frame == token, "The ticket is fresh on a token's first Present only");
      }
      s.end_scope(c);
      require(c[n::frames_total] == 10 && c[n::frames_once] == 10 && !c[n::frames_missed] && !c[n::frames_repeated], "A matching cadence detects once per frame");
      require(c[n::renders] == 20 && c[n::tagged] == 20 && c[n::detect_today_only] == 10 && c[n::exact_inexact] == 20, "Matching cadence render counts");
    }
    {
      // 4x presented, 2x reported: today detects three Presents per token.
      ticket::identity_shadow s;
      ticket::identity_counters c;
      for (std::uint64_t token = 1; token <= 10; ++token) {
        require(s.step(today(false, true), token, false, false, c).fresh, "Lagging token start");
        for (int i = 0; i != 3; ++i) {
          require(s.step(today(true), token, false, false, c).hold_previous, "A misread Present holds in the ticket view");
        }
      }
      require(c[n::frames_total] == 9, "The current token finalizes only at its end");
      s.end_scope(c);
      require(c[n::frames_total] == 10 && c[n::frames_repeated] == 10 && c[n::frames_extra] == 20 && !c[n::frames_once], "Lagging repeats");
    }
    {
      // A scope change forgets the finalized identity; a late render of a
      // finalized one only counts extra; token 0 is no identity.
      ticket::identity_shadow s;
      ticket::identity_counters c;
      s.step(today(false), 5, false, false, c);
      s.end_scope(c);
      require(c[n::frames_missed] == 1 && s.step(today(true), 5, false, false, c).fresh, "A new scope's identity is fresh");
      s.step(today(false), 6, false, false, c);
      const auto late = s.step(today(true), 5, false, false, c);
      require(!late.fresh && late.hold_previous && c[n::frames_extra] == 1, "A late render of a finalized identity");
      const auto none = s.step(today(true), 0, false, false, c);
      require(!none.fresh && !none.hold_previous && c[n::tagged] == 4, "An untagged render");
    }
    {
      // Exact and batch agreement kinds.
      ticket::identity_shadow s;
      ticket::identity_counters c;
      auto t = today(true);
      t.exact = t.batch = true;
      s.step(t, 1, true, true, c);
      s.step(t, 2, false, false, c);
      t.exact = t.batch = false;
      s.step(t, 3, true, true, c);
      require(c[n::exact_agree] == 1 && c[n::exact_today_only] == 1 && c[n::exact_ticket_only] == 1 && c[n::batch_agree] == 1 && c[n::batch_today_only] == 1 && c[n::batch_ticket_only] == 1, "Exact and batch agreement");
      require(c[n::detect_agree] == 2 && c[n::detect_today_only] == 1 && !c[n::detect_ticket_only], "Detect agreement");
    }
  }

  void t1_by_ticket() {
    using temporal::present_identity;
    // A re-offered token holds, a newer one detects, a Present offering
    // nothing keeps today's count bound; evidence without a token is new.
    const auto again = temporal::ticket_identity({false}, 41, 41, true);
    require(again.token_label == 41 && !again.new_label, "A re-offered token is not new");
    require(temporal::ticket_identity({true}, 42, 41, true).new_label, "A newer token is new");
    require(!temporal::ticket_identity({false}, 40, 41, true).new_label, "An older token is not new");
    require(temporal::ticket_identity({false}, 0, 41, true).new_label && !temporal::ticket_identity({true}, 0, 41, false).new_label, "Evidence without a token is new; nothing offered is not");
    // applied_identity: today's unless authoritative.
    const auto applied_default = temporal::applied_identity(again, ticket::authoritative(false));
    require(!applied_default.generated && applied_default.token_label == 41, "The default applies today's identity");
    const auto held = temporal::applied_identity(again, true);
    require(held.generated && held.token_label == 41, "A re-offered token holds the last decision");
    const auto fresh = temporal::applied_identity(temporal::ticket_identity({true}, 42, 41, true), true);
    require(!fresh.generated && fresh.token_label == 42, "A new token detects, whatever counting said");
    const auto zero = temporal::applied_identity(temporal::ticket_identity({true}, 0, 41, false), true);
    require(zero.generated && !zero.token_label, "A zero-offer Present keeps the count bound");
    // detection_state: the default arbitrates today's identity; the
    // test-only override arbitrates the ticket's.
    sunshine_game3d::alpha_auto_source scope;
    scope.epoch = scope.viewport = 1;
    for (const bool override_on : {false, true}) {
      temporal::detection_state state;
      state.identity_override = override_on;
      const auto first = temporal::ticket_identity({false}, 41, state.decision_token, true);
      require(state.arbitrate(first, scope, 0x10).detect, "The first Present detects");
      state.detected(scope, first);
      require(state.decision_token == 41, "The decision's token is kept");
      // Today's counting misreads the re-offering Present as real.
      const auto misread = temporal::ticket_identity({false}, 41, state.decision_token, true);
      const auto t = state.arbitrate(misread, scope, 0x10);
      require(override_on ? t.hold && !t.detect : t.detect && !t.hold, "The override holds a re-offered token; the default detects as today");
    }
    static_assert(!ticket::identity_authoritative, "S3 ships in shadow");
  }

  void counters_and_format() {
    ticket::identity_counters c;
    auto t = tag(kind::hudless, 300);
    t.present = ticket::present_label(boundary::at_tag, 9);
    ticket::count_ticket(t, c);
    ticket::ticket layer;
    layer.kind = ticket::capture_kind::layer_copy;
    layer.at = boundary::before_clear;
    layer.present = ticket::refuse(ticket::present_label(boundary::before_clear, 9), refusal::foreign_queue);
    layer.refused = refusal::foreign_queue;
    ticket::count_ticket(layer, c);
    ticket::count_gpu_verdict(ticket::gpu_verdict::exact, true, c);
    ticket::count_gpu_verdict(ticket::gpu_verdict::exact, false, c);
    ticket::count_gpu_verdict(ticket::gpu_verdict::mismatch, false, c);
    ticket::count_gpu_verdict(ticket::gpu_verdict::unproposed, false, c);
    ticket::count_gpu_verdict(ticket::gpu_verdict::none, false, c);
    c[n::interposers] = ip::sl_interposer | ip::sl_dlss_g;
    c[n::renders] = 9;
    c[n::tagged] = 8;
    c[n::frames_total] = 4;
    c[n::frames_once] = 2;
    c[n::frames_missed] = 1;
    c[n::frames_repeated] = 1;
    c[n::frames_extra] = 2;
    c[n::exact_agree] = 3;
    c[n::batch_today_only] = 1;
    const std::string expected =
      "renders=9 tagged=8 frames={total=4 once=2 missed=1 repeated=1 extra=2} "
      "detect={agree=0 today_only=0 ticket_only=0} exact={agree=3 today_only=0 ticket_only=0 inexact=0} "
      "batch={agree=0 today_only=1 ticket_only=0} label={token=1 present=1 none=1} "
      "refused={unstamped=0 foreign_queue=1 not_real_span=0 stale_scope=0 no_reference=0 mismatch=0 enhanced_barrier=0 "
      "render_pass=0 recording_not_covered=0 budget=0 begin_refused=0} "
      "gpu={exact=2 mismatch=1 unstamped=0 unproposed=1 token_exact=1} interposers=3";
    const auto text = ticket::format_identity_counters(c);
    require(text == expected, "format_identity_counters: " + text);
    ticket::identity_counters later = c;
    later[n::renders] = 12;
    later[n::interposers] = ip::xess_fg;
    const auto delta = later - c;
    require(delta[n::renders] == 3 && delta[n::interposers] == ip::xess_fg && !delta[n::tagged], "Counter difference");
    auto sum = c;
    sum += delta;
    require(sum == later, "Counter sum");
  }
}  // namespace

int main() {
  try {
    labels();
    pair_rule();
    frames_and_spans();
    refusals();
    interposers();
    shadow();
    t1_by_ticket();
    counters_and_format();
    std::puts("PASS: S3 UI snapshot ticket: labels, pair rule, frames, spans, refusals, interposers, identity shadow, T1 by ticket, log text");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
