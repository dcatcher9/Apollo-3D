// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_qualification.h"
#include "game3d_alpha_auto.h"
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
  using namespace sunshine_game3d::ui_qualification;
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
  scope candidate() {
    scope value;
    value.runtime = 1; value.device = 2; value.source_id = 3;
    value.epoch = 4; value.revision = 5; value.semantic_contract = 6;
    value.provider = 1; value.viewport = 7; value.source = choice::sl_ui_color_alpha;
    value.output_width = value.mask_width = value.width = 1920;
    value.output_height = value.mask_height = value.height = 1080;
    value.output_format = value.mask_format = 28;
    return value;
  }
  void sources_are_available_without_human_approval() {
    session state;
    const auto first = state.observe(candidate());
    require(first.available && first.token, "Captured source required human approval");
    state.unavailable();
    require(!state.snapshot().available && state.snapshot().token == first.token, "Capture gap lost logical identity or kept availability");
    require(!state.observe(candidate(), false).available, "Pending pixels became available");
    require(state.observe(candidate()).available, "Fresh pixels required another user action");
    auto hudless = candidate(); hudless.source = choice::sl_hudless;
    require(state.observe(hudless).available, "Automatic discovery could not switch to HUDless input");
    auto current = candidate(); current.source = choice::current_color;
    require(state.observe(current).available, "Automatic fallback could not report current color");
    auto combined = candidate(); combined.source = choice::automatic;
    require(state.observe(combined).available, "Current GPU candidate set was not reportable");
    auto ambiguous = candidate(); ambiguous.source = choice::sl_ui_alpha; ambiguous.semantic_contract = 0;
    require(state.observe(ambiguous).available, "Availability incorrectly claimed to enforce automatic semantic admission");
  }
  void explicit_selection_filters_sources_without_approval() {
    session state;
    require(state.set_choice(choice::sl_hudless), "HUDless source selection failed");
    auto input = candidate(); input.source = choice::sl_hudless;
    const auto selected = state.observe(input);
    require(selected.available && selected.selected == choice::sl_hudless, "Selected captured source was unavailable");
    require(!state.observe(candidate()).available, "A different source bypassed explicit selection");
    require(state.observe(input).available, "Returning selected source required approval");
    require(!state.set_choice(choice::sl_hudless) && state.snapshot().available, "Repeated selection lost current input");
    require(state.set_choice(choice::automatic) && !state.snapshot().available &&
      state.snapshot().choice_revision > selected.choice_revision, "Selection edit retained stale pixels");
    require(!state.set_choice(static_cast<choice>(99)), "Invalid source accepted");
    state.observe(candidate()); state.clear();
    require(!state.snapshot().available && state.snapshot().selected == choice::automatic, "Suspension retained pixels or changed selection");
  }
  void logical_changes_update_diagnostics() {
    const auto original = candidate();
    const auto check = [&](scope changed) {
      session state;
      const auto old = state.observe(original);
      const auto next = state.observe(changed);
      require(next.token != old.token && next.candidate == changed, "Source change retained stale diagnostic identity");
    };
    for (auto member : {&scope::runtime, &scope::device, &scope::source_id, &scope::epoch, &scope::revision, &scope::semantic_contract}) {
      auto next = original; ++(next.*member); check(next);
    }
    for (auto member : {&scope::provider, &scope::viewport, &scope::fg_generated_frames, &scope::output_width,
        &scope::output_height, &scope::output_format, &scope::color_space, &scope::mask_width, &scope::mask_height,
        &scope::mask_format, &scope::channel, &scope::left, &scope::top, &scope::width, &scope::height}) {
      auto next = original; ++(next.*member); check(next);
    }
    for (auto member : {&scope::fg_known, &scope::fg_enabled, &scope::fg_automatic}) {
      auto next = original; next.*member = !(next.*member); check(next);
    }
    session state;
    auto token = state.observe(original).token;
    for (unsigned i=0;i!=100;++i) require(state.observe(original).token == token, "Repeated capture changed logical source token");
    session saturated(std::numeric_limits<std::uint64_t>::max());
    require(saturated.observe(original).token == std::numeric_limits<std::uint64_t>::max(), "Diagnostic token wrapped");
  }
  void invalid_layout_and_independent_modes() {
    auto check = [](scope input) { session state; require(!state.observe(input).available, "Malformed input became available"); };
    auto input = candidate(); input.runtime=0; check(input);
    input=candidate(); input.device=0; check(input);
    input=candidate(); input.mask_format=0; check(input);
    input=candidate(); input.output_format=0; check(input);
    input=candidate(); input.channel=2; check(input);
    input=candidate(); input.width=0; check(input);
    input=candidate(); input.left=std::numeric_limits<std::uint32_t>::max(); check(input);
    input=candidate(); input.top=input.mask_height; check(input);
    sunshine_game3d::alpha_auto_policy mode;
    mode.set_manual(false);
    session a,b; a.observe(candidate());
    require(!mode.decision(1000).enabled && !b.snapshot().available, "Source observation changed mode or another runtime");
    mode.set_manual(true); a.clear();
    require(mode.decision(1000).enabled, "Source invalidation changed manual On");
  }
}
int main() {
  try {
    sources_are_available_without_human_approval();
    explicit_selection_filters_sources_without_approval();
    logical_changes_update_diagnostics();
    invalid_layout_and_independent_modes();
    std::puts("Automatic UI source status: 4 policy groups passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr,"Automatic UI source status failed: %s\n",error.what()); return 1;
  }
}