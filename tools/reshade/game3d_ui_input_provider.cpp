// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_input_provider.h"
#include "game3d_ui_mask.h"
#include "game3d_ui_layer.h"
#include "game3d_ui_ticket.h"
#include "game3d_frame_clock.h"
#include "game3d_capture_diagnostic.h"
#include "game3d_diagnostics.h"
#include "streamline_camera_probe.h"
#include "streamline_buffer_contract.h"
#include <d3d12.h>
#include <reshade.hpp>
#include "async_log.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace sunshine_game3d::ui_input {
  static_assert(ui_detection_inputs::max_retained_presents == ui_mask::max_late_presents,
    "The renderer must retain the color of every admitted late HUD-less frame");
  namespace api = reshade::api;
  namespace capture = sunshine_streamline::depth_capture;
  namespace {
    using choice = ui_qualification::choice;
    // Paired with its batch's tagged Backbuffer, on time, late but paired with
    // retained color, held on a generated present, too old to pair, otherwise
    // unpaired, and no HUD-less capture.
    enum hudless_outcome : unsigned { hudless_batch, hudless_real, hudless_late, hudless_generated, hudless_stale,
      hudless_other, hudless_none, hudless_outcome_count };
    struct source_entry {
      ui_qualification::session selection;
      ui_qualification::scope base;
      std::uint64_t instance{}, last_observe_tick{}, frame_sequence{}, next_gate_log{};
      // The capture owner's identity for this runtime's native device. It is
      // resolved from a backbuffer once instead of on every presentation.
      std::uint64_t native_device{}, device_identity{};
      // HUD-less pairing outcomes since the last gate log.
      std::array<std::uint32_t, hudless_outcome_count> hudless_outcomes{};
      // T1 Present counting without a HUD-less pairing: the frame_sequence of
      // the last render that offered a Streamline UI tag, in this epoch and
      // viewport (ui_mask::generated_without_input).
      std::uint64_t input_present{}, input_epoch{};
      std::uint32_t input_viewport{};
      bool suspended{};
      // S3 shadow (game3d_ui_ticket.h): today's Present-counting pairing beside
      // the ticket's identity, the running totals and what was last logged,
      // and the run of renders with frame generation known off (this one
      // included), which bounds a real span.
      ui_ticket::identity_shadow identity;
      ui_ticket::identity_counters identity_totals, identity_logged;
      // The renderer's cumulative GPU verdict counts (renderer::identity_counts)
      // last folded into identity_totals' gpu group.
      ui_ticket::identity_counters identity_gpu_seen;
      std::uint64_t next_identity_log{}, identity_epoch{};
      std::uint32_t identity_viewport{}, fg_off_run{};
      std::uint32_t logged_interposers{UINT32_MAX};
      bool logged_fg_known{}, logged_fg_enabled{};
    };
    std::mutex source_mutex;
    std::unordered_map<api::effect_runtime *, source_entry> sources;
    std::uint64_t instance_serial{};
    source_entry &entry_for(api::effect_runtime *runtime) {
      const auto [found, inserted] = sources.try_emplace(runtime);
      if (inserted) found->second.instance = ++instance_serial;
      return found->second;
    }
    void expire_available(source_entry &entry, std::uint64_t now) {
      if (!entry.last_observe_tick || now < entry.last_observe_tick ||
          now - entry.last_observe_tick >= sunshine_scene_depth::maximum_source_age_ms)
        entry.selection.unavailable();
    }
    bool present_has_alpha(std::uint32_t format) {
      return ui_mask::supported_format(ui_mask::source_kind::backbuffer,
        static_cast<std::uint32_t>(api::format_to_default_typed(static_cast<api::format>(format), 0)));
    }
    std::uint32_t source_filter(choice source) {
      switch (source) {
        case choice::sl_ui_alpha: return ui_mask::source_mask(ui_mask::source_kind::alpha);
        case choice::sl_ui_color_alpha: return ui_mask::source_mask(ui_mask::source_kind::color_and_alpha);
        case choice::sl_backbuffer: return ui_mask::source_mask(ui_mask::source_kind::backbuffer);
        // The tagged Backbuffer is captured as HUD-less's exact color pair.
        case choice::sl_hudless: return ui_mask::source_mask(ui_mask::source_kind::hudless) |
          ui_mask::source_mask(ui_mask::source_kind::backbuffer);
        case choice::current_color: return 0;
        default: return ui_mask::all_sources;
      }
    }
    bool capture_needed(const source_alpha_ui_decision &status, api::device_api backend) {
      return status.requested && backend == api::device_api::d3d12;
    }
    choice choice_for(ui_mask::source_kind kind) {
      switch (kind) {
        case ui_mask::source_kind::alpha: return choice::sl_ui_alpha;
        case ui_mask::source_kind::color_and_alpha: return choice::sl_ui_color_alpha;
        case ui_mask::source_kind::hudless: return choice::sl_hudless;
        default: return choice::sl_backbuffer;
      }
    }
    std::string pointer_text(const void *value) {
      char text[32]{};
      std::snprintf(text, sizeof(text), "%p", value);
      return text;
    }
    ui_qualification::scope captured_scope(ui_qualification::scope base, const ui_mask::selection &selected) {
      const auto &origin = selected.origin;
      base.source = choice_for(origin.kind); base.provider = 1; base.source_id = origin.source.source_id;
      base.epoch = origin.source.epoch; base.revision = origin.source.observation_revision; base.viewport = origin.source.viewport;
      base.channel = origin.kind == ui_mask::source_kind::alpha ? 1 : 0;
      base.mask_width = selected.texture.width; base.mask_height = selected.texture.height; base.mask_format = selected.texture.format;
      base.left = selected.texture.area.left; base.top = selected.texture.area.top;
      base.width = selected.texture.area.width; base.height = selected.texture.area.height;
      base.semantic_contract = origin.kind != ui_mask::source_kind::alpha ||
        sunshine_streamline::buffers::ui_alpha_authenticated(sunshine_streamline::buffers::active()) ? 1 : 0;
      return base;
    }
    // S3: the frame-generation modules loaded in this process, probed at most
    // once a second (frame_clock::loaded_interposers).
    std::uint32_t interposer_bits(std::uint64_t now) {
      static std::atomic<std::uint64_t> probed{};
      static std::atomic<std::uint32_t> bits{};
      const auto last = probed.load(std::memory_order_relaxed);
      if (!last || now < last || now - last >= 1000) {
        bits.store(frame_clock::loaded_interposers(), std::memory_order_relaxed);
        probed.store(now ? now : 1, std::memory_order_relaxed);
      }
      return bits.load(std::memory_order_relaxed);
    }
    // Once per process and kind: the first candidate read where it lies.
    void log_direct_binding(unsigned kind) {
      static std::atomic<bool> logged[2]{};
      if (logged[kind].exchange(true, std::memory_order_relaxed)) return;
      sunshine_log::message(reshade::log::level::info, kind ? "Sunshine UI input: reading the live UI layer copy directly (no copy)" :
        "Sunshine UI input: reading leased Streamline UI snapshots directly (no copy)");
    }
    ui_ticket::queue_relation relation(std::uint64_t executed, std::uint64_t wanted) {
      return !executed || !wanted ? ui_ticket::queue_relation::unknown :
        executed == wanted ? ui_ticket::queue_relation::same : ui_ticket::queue_relation::foreign;
    }
    // The CPU half of a Streamline tag snapshot's ticket: its own token label
    // (contract 6); its present label (C_P read + 1) is known on the GPU only.
    ui_ticket::ticket tag_ticket(const ui_mask::selection &selected, ui_selection::kind source, std::uint32_t typed_format,
        const ui_qualification::scope &base, std::uint64_t presenting_queue) {
      ui_ticket::ticket t;
      t.kind = ui_ticket::capture_kind::sl_tag; t.source = source; t.at = ui_ticket::boundary::at_tag;
      t.token = ui_ticket::token_label_of_tag(selected.token_generation);
      t.present = {ui_ticket::label_space::present, 0, false, ui_ticket::refusal::none};
      t.how = ui_ticket::state_basis_proof(static_cast<std::uint8_t>(selected.texture.basis));
      t.present_queue = relation(selected.texture.producer_queue, presenting_queue);
      t.token_queue = ui_ticket::queue_relation::same;
      t.typed_format = typed_format; t.color_space = base.color_space;
      t.epoch = selected.origin.source.epoch; t.viewport = selected.origin.source.viewport;
      t.token_generation = selected.token_generation;
      ui_ticket::refusal_inputs in;
      in.stale_scope = t.epoch != base.epoch || t.viewport != base.viewport;
      in.foreign_queue = t.present_queue == ui_ticket::queue_relation::foreign;
      in.unstamped = !selected.texture.stamped;
      t.refused = ui_ticket::first_refusal(in);
      return t;
    }
    nlohmann::json label_json(const ui_ticket::label &value) {
      return {{"space", ui_ticket::name(value.space)}, {"value", value.value}, {"valid", value.valid},
        {"reason", ui_ticket::name(value.reason)}};
    }
    nlohmann::json ticket_json(const ui_ticket::ticket &t, bool stamped) {
      return {{"kind", ui_ticket::name(t.kind)}, {"signature", t.key()}, {"boundary", ui_ticket::name(t.at)},
        {"token", label_json(t.token)}, {"present", label_json(t.present)}, {"proof", ui_ticket::name(t.how)},
        {"present_queue", ui_ticket::name(t.present_queue)}, {"token_queue", ui_ticket::name(t.token_queue)},
        {"refusal", ui_ticket::name(t.refused)}, {"begin_stage", ui_ticket::name(t.stage)},
        {"encoding", {{"typed_format", t.typed_format}, {"color_space", t.color_space}}},
        {"epoch", t.epoch}, {"viewport", t.viewport}, {"token_generation", t.token_generation}, {"stamped", stamped}};
    }
    nlohmann::json captured_metadata(const ui_mask::selection &selected, std::uint64_t now, bool admitted, bool paired) {
      const auto &origin = selected.origin; const auto &source = origin.source; const auto &copy = selected.texture;
      const std::uint32_t candidate_bit = origin.kind == ui_mask::source_kind::alpha ? ui_detection::candidate::ui_alpha :
        origin.kind == ui_mask::source_kind::color_and_alpha ? ui_detection::candidate::ui_color :
        origin.kind == ui_mask::source_kind::hudless ? ui_detection::candidate::hudless : ui_detection::candidate::backbuffer;
      return {
        {"source", ui_qualification::name(choice_for(origin.kind))}, {"candidate_bit", candidate_bit},
        {"tag_type", static_cast<std::uint32_t>(origin.kind)}, {"tag_scope", origin.tag_scope},
        {"channel", origin.kind == ui_mask::source_kind::hudless ? "rgb_difference" : origin.kind == ui_mask::source_kind::alpha ? "red" : "alpha"}, {"format", copy.format},
        {"available_for_detection", admitted}, {"current_present_pair", paired},
        {"association", origin.kind == ui_mask::source_kind::hudless ? "real_frame_present_generation" : "latest_submitted_input_approximation"},
        {"epoch", source.epoch}, {"observation_revision", source.observation_revision},
        {"sequence", source.sequence}, {"tick_ms", source.tick}, {"age_ms", now >= source.tick ? now - source.tick : 0},
        {"viewport", source.viewport}, {"source_native", source.resource.native}, {"source_command", origin.command},
        {"source_frame_generation", source.source_frame_generation}, {"source_frame_token", source.source_frame_token},
        {"source_frame_numeric", source.source_frame_numeric}, {"source_frame_has_numeric", source.source_frame_has_numeric},
        {"source_frame_explicit", source.source_frame_explicit}, {"capture_id", copy.capture_id},
        {"source_present_generation", origin.source_present_generation},
        {"current_source_present_generation", selected.current_source_present_generation},
        {"producer_queue", copy.producer_queue}, {"producer_fence", copy.producer_fence},
        {"producer_completed", copy.producer_completed}, {"producer_recording_retired", copy.producer_recording_retired},
        {"token_generation", selected.token_generation}, {"state_basis", capture::name(copy.basis)}
      };
    }
  }
  presentation observe() {
    sunshine_streamline::frame_generation_snapshot fg;
    presentation result;
    if (sunshine_streamline::query_frame_generation(UINT32_MAX, fg))
      result.mode = {fg.known, fg.enabled, fg.automatic, fg.generated_frames, fg.viewport, fg.epoch, fg.sequence};
    result.observer_active = sunshine_streamline::source_enabled() || sunshine_streamline::enabled();
    return result;
  }
  source_alpha_ui_decision resolve(source_alpha_ui_policy &policy, const render_settings &settings, const presentation &observed) {
    auto result = policy.update(settings.ui_protection != source_alpha_mode::off, observed.mode, observed.observer_active);
    result.mode = settings.ui_protection;
    return result;
  }
  void invalidate(api::effect_runtime *runtime) {
    ui_mask::invalidate(reinterpret_cast<std::uint64_t>(runtime));
    std::lock_guard<std::mutex> lock(source_mutex);
    sources.erase(runtime);
  }
  void suspend(api::effect_runtime *runtime) {
    ui_mask::invalidate(reinterpret_cast<std::uint64_t>(runtime));
    std::lock_guard<std::mutex> lock(source_mutex);
    if (const auto found = sources.find(runtime); found != sources.end()) {
      auto &entry = found->second;
      if (!entry.suspended) entry.selection.clear();
      entry.selection.unavailable(); entry.last_observe_tick = 0; entry.suspended = true;
    }
  }
  ui_qualification::status source_status(api::effect_runtime *runtime) {
    std::lock_guard<std::mutex> lock(source_mutex);
    const auto found = sources.find(runtime);
    if (found != sources.end()) expire_available(found->second, GetTickCount64());
    return found == sources.end() ? ui_qualification::status{} : found->second.selection.snapshot();
  }
  void select_source(api::effect_runtime *runtime, choice source) {
    if (!runtime || !ui_qualification::valid(source)) return;
    std::lock_guard<std::mutex> lock(source_mutex);
    auto &entry = entry_for(runtime);
    if (entry.selection.set_choice(source)) entry.last_observe_tick = 0;
  }
  void configure(api::effect_runtime *runtime, const source_alpha_ui_decision &status, api::device_api backend) {
    if (!capture_needed(status, backend)) ui_mask::invalidate(reinterpret_cast<std::uint64_t>(runtime));
    if (!status.requested) {
      std::lock_guard<std::mutex> lock(source_mutex);
      if (auto found = sources.find(runtime); found != sources.end()) found->second.selection.unavailable();
    }
  }
  void request(api::effect_runtime *runtime, api::resource color, const source_alpha_ui_decision &status,
      const render_settings &, api::color_space color_space) {
    auto *device = runtime->get_device();
    const auto desc = device->get_resource_desc(color);
    sunshine_streamline::ui_observation_scope observed;
    const bool have_scope = sunshine_streamline::query_ui_scope(UINT32_MAX, observed);
    ui_qualification::scope base;
    base.runtime = reinterpret_cast<std::uint64_t>(runtime); base.device = device->get_native(); base.semantic_contract = 1;
    base.epoch = have_scope ? observed.epoch : 0;
    base.revision = have_scope ? observed.revision : sunshine_streamline::depth_observation_revision();
    base.viewport = have_scope ? observed.viewport : status.fg.viewport;
    base.fg_known = status.fg.known; base.fg_enabled = status.fg.enabled;
    base.fg_automatic = status.fg.automatic; base.fg_generated_frames = status.fg.generated_frames;
    base.output_width = base.mask_width = base.width = desc.texture.width;
    base.output_height = base.mask_height = base.height = desc.texture.height;
    base.output_format = base.mask_format = static_cast<std::uint32_t>(desc.texture.format);
    base.color_space = static_cast<std::uint32_t>(color_space);
    std::uint32_t filter;
    std::uint64_t device_identity{};
    {
      std::lock_guard<std::mutex> lock(source_mutex);
      auto &entry = entry_for(runtime); entry.suspended = false;
      if (entry.base != base) { entry.selection.clear(); entry.last_observe_tick = 0; entry.base = base; }
      filter = source_filter(entry.selection.selected_choice());
      if (entry.native_device == base.device) device_identity = entry.device_identity;
    }
    if (!capture_needed(status, device->get_api()) || !have_scope || !filter) {
      ui_mask::invalidate(reinterpret_cast<std::uint64_t>(runtime)); return;
    }
    if (!device_identity) {
      capture::record_diagnostic identity;
      if (!capture::retain_source(color.handle, &identity) || !identity.expected_device_identity) {
        ui_mask::invalidate(reinterpret_cast<std::uint64_t>(runtime)); return;
      }
      device_identity = identity.expected_device_identity;
      std::lock_guard<std::mutex> lock(source_mutex);
      auto &entry = entry_for(runtime);
      entry.native_device = base.device; entry.device_identity = device_identity;
    }
    ui_mask::set_request({reinterpret_cast<std::uint64_t>(runtime), device_identity,
      observed.epoch, observed.revision, observed.viewport, desc.texture.width, desc.texture.height, true, filter});
  }
  frame acquire(api::effect_runtime *runtime, renderer &renderer, source_alpha_ui_decision status,
      alpha_auto_policy &session, bool diagnostic) {
    frame result;
    // Auto and manual On both offer the filtered candidates through detection
    // (S2); only Off requests none.
    result.status = status; result.automatic_detection = status.mode != source_alpha_mode::off;
    result.detection.current_color = false;
    ui_qualification::scope base;
    ui_qualification::status before;
    std::uint64_t instance{}, frame_sequence{}, input_present{};
    {
      std::lock_guard<std::mutex> lock(source_mutex);
      const auto found = sources.find(runtime);
      if (found != sources.end()) {
        const auto &entry = found->second;
        base = entry.base; before = entry.selection.snapshot();
        instance = entry.instance; frame_sequence = ++found->second.frame_sequence;
        if (entry.input_epoch == base.epoch && entry.input_viewport == base.viewport) input_present = entry.input_present;
      }
    }
    const auto now = GetTickCount64();
    auto &input = result.observation;
    input.now_ms = input.tick_ms = now; input.session = &session;
    input.epoch = base.epoch; input.revision = base.revision; input.viewport = base.viewport; input.sequence = frame_sequence;
    const auto wanted_source = before.selected;
    auto candidate = base;
    bool available = false;
    nlohmann::json candidates = nlohmann::json::array();
    auto *queue = runtime->get_command_queue();
    auto *commands = queue->get_immediate_command_list();
    constexpr ui_mask::source_kind kinds[]{ui_mask::source_kind::alpha, ui_mask::source_kind::color_and_alpha,
      ui_mask::source_kind::backbuffer, ui_mask::source_kind::hudless};
    // S3 shadow (game3d_ui_ticket.h): this render's Present label, one ticket
    // per offered candidate, and the labels the CPU proposes for the GPU to
    // verify. Nothing here decides while ui_ticket::identity_authoritative is
    // false; today's Present-counting pairing is computed unchanged beside it.
    const bool authoritative = ui_ticket::authoritative(false);
    // The shadow runs only with the Diagnostics switch on (G2): off, nothing
    // is stamped or proposed and no identity totals or lines are kept. The
    // tickets' own token labels come from the capture owner either way.
    const bool shadow = diagnostics::enabled();
    const auto present_label = shadow ? frame_clock::label(runtime->get_device()) : 0u;
    const auto presenting_queue = queue->get_native();
    const auto interposers = shadow ? interposer_bits(now) : 0u;
    std::uint32_t fg_off_run{};
    {
      std::lock_guard<std::mutex> lock(source_mutex);
      if (const auto found = sources.find(runtime); found != sources.end()) {
        auto &entry = found->second;
        entry.fg_off_run = status.fg.known && !status.fg.enabled ? entry.fg_off_run + 1 : 0;
        fg_off_run = entry.fg_off_run;
      }
    }
    std::array<ui_ticket::ticket, ui_ticket::slot::count> tickets{};
    std::array<bool, ui_ticket::slot::count> offered{}, stamped{};
    ui_ticket::today_view today;
    std::uint32_t expected_layer_present{}, expected_layer_token{}, expected_hudless_present{};
    std::uint64_t hudless_token{}, backbuffer_producer{}, backbuffer_token{};
    bool hudless_present_space{};
    // The hooked interposer's own header range decides UIAlpha's raw number
    // (2.11.x: 68, 2.12+: 69). Outside the surveyed range it stays manual-only.
    const bool alpha_authenticated =
      sunshine_streamline::buffers::ui_alpha_authenticated(sunshine_streamline::buffers::active());
    auto hudless_result = hudless_none;
    // The tagged Backbuffer is the game's own final color. Captured in the same
    // tag batch as HUD-less, it is that image's exact pair on any Present.
    struct { api::resource_view view{}; std::uint64_t tagged{}, current{}; } backbuffer;
    const bool capturing = capture_needed(status, runtime->get_device()->get_api());
    // Direct binding (C4): with the shadow off and no dump armed, a candidate
    // is read where it lies (a leased Streamline snapshot, the live layer
    // copy) instead of being copied into the renderer's slot first. A dump or
    // the shadow keeps today's copies (and their stamp entries).
    const bool direct_binding = !shadow && !diagnostic && commands == queue->get_immediate_command_list();
    if (capturing && direct_binding)
      capture::observe_runtime_list(commands->get_native(), queue->get_native()); // Registered once; an SRW check after.
    // Manual On offers the same filtered candidates through detection, each
    // accepted for this session only (S2). Its explicit first filtered
    // candidate, in draw order, remains the reported manual input and the
    // renderer's fallback where detection cannot run.
    const bool manual = status.mode != source_alpha_mode::automatic;
    struct explicit_capture {
      bool set{};
      ui_mask::selection selected;
      api::resource_view view{};
      std::string metadata;
    };
    std::array<explicit_capture, 4> explicit_captures{}; // Capture slot order.
    auto &signatures = result.detection.signatures;
    signatures.color_space = base.color_space;
    const auto typed_format = [](std::uint32_t format) {
      return static_cast<std::uint32_t>(api::format_to_default_typed(static_cast<api::format>(format), 0));
    };
    constexpr ui_selection::kind slot_kinds[]{ui_selection::kind::ui_alpha, ui_selection::kind::ui_color,
      ui_selection::kind::backbuffer, ui_selection::kind::hudless};
    const bool hudless_wanted = capturing && (source_filter(wanted_source) & ui_mask::source_mask(ui_mask::source_kind::hudless));
    // S3: the newest HUD-less and Backbuffer pair of one game frame by token
    // (read-only, before acquire_kind retires an older ready snapshot).
    ui_mask::selection batch_hudless, batch_backbuffer;
    const bool token_batch = hudless_wanted && ui_mask::acquire_batch(reinterpret_cast<std::uint64_t>(runtime), now,
      presenting_queue, batch_hudless, batch_backbuffer);
    if (capturing) for (unsigned slot = 0; slot != 4; ++slot) {
      const auto kind = kinds[slot];
      if (!(source_filter(wanted_source) & ui_mask::source_mask(kind))) continue;
      ui_mask::selection selected;
      if (authoritative && token_batch && (kind == ui_mask::source_kind::hudless || kind == ui_mask::source_kind::backbuffer))
        selected = kind == ui_mask::source_kind::hudless ? batch_hudless : batch_backbuffer;
      else if (!ui_mask::acquire_kind(reinterpret_cast<std::uint64_t>(runtime), kind, selected, now, queue->get_native())) continue;
      const bool hudless = kind == ui_mask::source_kind::hudless;
      const auto ticket_slot = ui_ticket::slot_of_candidate(slot);
      tickets[ticket_slot] = tag_ticket(selected, slot_kinds[slot], typed_format(selected.texture.format), base, presenting_queue);
      offered[ticket_slot] = true; stamped[ticket_slot] = shadow && selected.texture.stamped;
      if (kind == ui_mask::source_kind::backbuffer) {
        backbuffer_producer = selected.texture.producer_queue; backbuffer_token = selected.token_generation;
      }
      // Pairing is by Present count only. Streamline can present on its own
      // queue; pixels from another queue were already admitted only after the
      // producer completed and its recording retired.
      const auto tagged = selected.origin.source_present_generation, current = selected.current_source_present_generation;
      const auto pairing = ui_mask::pair_hudless_present(tagged, current, status.fg_active(), status.fg.generated_frames);
      using ui_mask::hudless_present;
      const bool batch = hudless && backbuffer.view.handle &&
        ui_mask::same_tag_interval(tagged, current, backbuffer.tagged, backbuffer.current);
      const bool counted = pairing.kind == hudless_present::real_frame || pairing.kind == hudless_present::earlier_real_frame;
      // S3: the ticket's pair for this HUD-less image: the same-token
      // Backbuffer (token space, any frame generation), else, with frame
      // generation off over a real span on the presenting queue, the
      // presented colour today's count names, proposed for the GPU to verify.
      bool ticket_exact = false;
      if (hudless) {
        hudless_token = selected.token_generation;
        hudless_present_space = !token_batch && counted && !status.fg_active() &&
          ui_ticket::real_span(fg_off_run, pairing.presents_ago, interposers, status.fg.known, status.fg.enabled) &&
          tickets[ticket_slot].present_queue == ui_ticket::queue_relation::same && present_label > pairing.presents_ago;
        if (hudless_present_space) expected_hudless_present = present_label - pairing.presents_ago;
        ticket_exact = (token_batch && batch_hudless.token_generation == selected.token_generation) || hudless_present_space;
        today.offered = true; today.batch = batch; today.real_frame = tagged;
      }
      const bool paired = authoritative && hudless ? ticket_exact || (token_batch && batch) :
        batch || counted;
      if (hudless) hudless_result = batch ? hudless_batch : pairing.kind == hudless_present::real_frame ? hudless_real :
        pairing.kind == hudless_present::earlier_real_frame ? hudless_late :
        pairing.kind == hudless_present::generated_frame ? hudless_generated :
        tagged && tagged != UINT64_MAX && current > tagged ? hudless_stale : hudless_other;
      if (hudless && !batch && pairing.kind == hudless_present::generated_frame) {
        // Interpolated color cannot be differenced against this tag's scene.
        // Nothing is copied; the renderer shows the decision of the real frame
        // this Present shows (T1), identified by the tag's present generation.
        if (diagnostic) {
          candidates.push_back(captured_metadata(selected, now, false, false));
          candidates.back()["held_for_generated_present"] = true;
        }
        result.status.retained_alpha_ready = true;
        result.detection.hold_previous = true;
        result.detection.real_frame = tagged;
        available = true;
        if (manual) explicit_captures[slot] = {true, selected, {}, diagnostic ? candidates.back().dump() : std::string{}};
        if (diagnostic) candidates.back()["ticket"] = ticket_json(tickets[ticket_slot], stamped[ticket_slot]);
        continue;
      }
      // An unauthenticated UIAlpha tag cannot hide independently usable
      // lower-priority candidates in Auto; an authenticated one is validated on
      // the GPU. A manual choice may use it.
      const bool admissible = (manual || kind != ui_mask::source_kind::alpha || alpha_authenticated) && (!hudless || paired);
      api::resource_view view{};
      const auto typed = typed_format(selected.texture.format);
      // Direct binding: the leased snapshot itself, at the output's extent
      // (the renderer's slot copy has that extent too), else the copy.
      if (admissible && direct_binding && selected.texture.width == base.output_width &&
          selected.texture.height == base.output_height) {
        capture::local_view leased;
        if (capture::lease_local_view(commands->get_native(), queue->get_native(), selected.ticket, typed, leased))
          view = renderer.bind_ui_snapshot(slot, selected.ticket.id, {leased.resource}, leased.view_format);
        if (view.handle) log_direct_binding(0);
        result.leased |= view.handle != 0;
      }
      const auto copy = [&](api::resource destination, api::resource stamps, std::uint64_t stamp_offset) {
        return capture::copy_local_texture(commands->get_native(), queue->get_native(), selected.ticket,
          destination.handle, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
          nullptr, commands == queue->get_immediate_command_list(), {stamps.handle, stamp_offset});
      };
      // The shadow's copy carries the snapshot's S3 stamp entry into the
      // renderer's stamp buffer beside it; without it no stamp buffer exists.
      if (admissible && !view.handle) {
        if (shadow) view = renderer.prepare_ui_candidate(slot, selected.ticket.id, copy, static_cast<api::format>(selected.texture.format));
        else view = renderer.prepare_ui_candidate(slot, selected.ticket.id,
            [&](api::resource destination) { return copy(destination, {}, 0); }, static_cast<api::format>(selected.texture.format));
      }
      if (diagnostic) {
        candidates.push_back(captured_metadata(selected, now, view.handle != 0, pairing.kind == hudless_present::real_frame));
        if (hudless) {
          candidates.back()["paired_with"] = batch ? "tagged_backbuffer_same_batch" : "presented_color";
          candidates.back()["real_frame_presents_ago"] = pairing.presents_ago;
          candidates.back()["ticket_pair"] = token_batch && batch_hudless.token_generation == selected.token_generation ?
            "same_token_backbuffer" : hudless_present_space ? "proposed_present" : "absent";
        }
        candidates.back()["ticket"] = ticket_json(tickets[ticket_slot], stamped[ticket_slot]);
      }
      if (!view.handle) continue;
      if (kind == ui_mask::source_kind::backbuffer) backbuffer = {view, tagged, current};
      // A manual HUD-less choice captures the Backbuffer only as its pair.
      if (manual && wanted_source == choice::sl_hudless && !hudless) continue;
      if (hudless) {
        if (batch) result.detection.hudless_pair = backbuffer.view;
        else result.detection.hudless_presents_ago = pairing.presents_ago;
        // Present counting is exact only without frame generation; Hogwarts
        // showed FG-on Present pairs that belonged to another frame. Once S3
        // is authoritative the ticket's pair decides exactness instead.
        result.detection.hudless_exact = authoritative ? ticket_exact : batch || !status.fg_active();
        result.detection.hudless = view;
        // T1 identifies the real frame by the HUD-less tag's present
        // generation; once S3 is authoritative the renderer identifies it by
        // the newest offered token (ui_temporal::ticket_identity).
        result.detection.real_frame = tagged;
        today.detects = true; today.exact = result.detection.hudless_exact;
      } else result.detection.masks[slot] = view;
      signatures.set(slot_kinds[slot], typed_format(selected.texture.format));
      result.status.retained_alpha_ready = true;
      available = true;
      if (manual) explicit_captures[slot] = {true, selected, view, diagnostic ? candidates.back().dump() : std::string{}};
    }
    // The game's offscreen UI layer (game3d_ui_layer.h) is a candidate of its
    // own beside any tagged UIColorAndAlpha (E1), in renderer slot 4. Its copy
    // holds the previous frame's UI; detection reads it only while V1-valid,
    // premultiplied included.
    ui_layer::live_capture layer;
    const bool layer_wanted = status.requested &&
      (source_filter(wanted_source) & ui_mask::source_mask(ui_mask::source_kind::color_and_alpha));
    if (layer_wanted && ui_layer::latest(runtime->get_device(), now, layer)) {
      constexpr std::uint64_t layer_capture = std::uint64_t(1) << 62; // Never a Streamline ticket id.
      const auto copy = [&](api::resource destination, api::resource stamps, std::uint64_t stamp_offset) {
        commands->barrier(layer.copy, api::resource_usage::shader_resource, api::resource_usage::copy_source);
        commands->barrier(destination, api::resource_usage::shader_resource, api::resource_usage::copy_dest);
        commands->copy_resource(layer.copy, destination);
        commands->barrier(destination, api::resource_usage::copy_dest, api::resource_usage::shader_resource);
        commands->barrier(layer.copy, api::resource_usage::copy_source, api::resource_usage::shader_resource);
        if (stamps.handle && layer.stamp.handle) commands->copy_buffer_region(layer.stamp, 0, stamps, stamp_offset, 16);
        return true;
      };
      // Direct binding reads the live copy where it lies when its lists ran
      // on the presenting queue; else it is copied into the renderer's slot
      // (with the shadow, also its S3 stamp into the renderer's stamp buffer;
      // both rest in COMMON; implicit promotion on this list). Either read
      // holds the ring entry until this render's completion (complete()).
      api::resource_view view{};
      if (direct_binding && layer.direct)
        view = renderer.bind_ui_candidate(4, layer_capture | layer.capture_id, layer.view, layer.copy);
      if (view.handle && direct_binding && layer.direct) log_direct_binding(1);
      else if (shadow) view = renderer.prepare_ui_candidate(4, layer_capture | layer.capture_id, copy, static_cast<api::format>(layer.format));
      else view = renderer.prepare_ui_candidate(4, layer_capture | layer.capture_id,
          [&](api::resource destination) { return copy(destination, {}, 0); }, static_cast<api::format>(layer.format));
      result.layer_device = runtime->get_device();
      result.layer_capture = layer.capture_id;
      // S3 ticket: a before-clear copy whose labels only its stamp knows. The
      // CPU proposes its present label (the Present its count names, within a
      // real span on the presenting queue) and, with frame generation on,
      // unknown or suspended, its token label (the newest ready Backbuffer's,
      // when the layer's lists run on the Backbuffer's queue).
      auto &layer_ticket = tickets[ui_ticket::slot::layer];
      layer_ticket.kind = ui_ticket::capture_kind::layer_copy; layer_ticket.source = ui_selection::kind::ui_layer;
      layer_ticket.at = ui_ticket::boundary::before_clear; layer_ticket.how = ui_ticket::proof::clear_precall;
      layer_ticket.token = {ui_ticket::label_space::token, 0, false, ui_ticket::refusal::none};
      layer_ticket.present = {ui_ticket::label_space::present, 0, false, ui_ticket::refusal::none};
      layer_ticket.present_queue = layer.foreign_present ? ui_ticket::queue_relation::foreign :
        relation(layer.executed_queue, presenting_queue);
      layer_ticket.token_queue = layer.queue_mixed ? ui_ticket::queue_relation::foreign :
        relation(layer.executed_queue, backbuffer_producer);
      layer_ticket.typed_format = typed_format(layer.format); layer_ticket.color_space = base.color_space;
      layer_ticket.epoch = base.epoch; layer_ticket.viewport = base.viewport;
      offered[ui_ticket::slot::layer] = true; stamped[ui_ticket::slot::layer] = shadow && layer.stamped;
      const bool fg_known_off = status.fg.known && !status.fg.enabled;
      const bool present_space = fg_known_off && layer.presents_since_copy < present_label &&
        ui_ticket::real_span(fg_off_run, layer.presents_since_copy, interposers, status.fg.known, status.fg.enabled);
      const bool token_space = !fg_known_off && backbuffer_token &&
        layer_ticket.token_queue == ui_ticket::queue_relation::same;
      if (present_space && layer_ticket.present_queue != ui_ticket::queue_relation::foreign)
        expected_layer_present = present_label - layer.presents_since_copy;
      if (token_space && shadow) expected_layer_token = static_cast<std::uint32_t>(backbuffer_token);
      {
        ui_ticket::refusal_inputs in;
        in.foreign_queue = (present_space && layer_ticket.present_queue == ui_ticket::queue_relation::foreign) ||
          (!fg_known_off && layer_ticket.token_queue == ui_ticket::queue_relation::foreign);
        in.unstamped = !stamped[ui_ticket::slot::layer];
        in.not_real_span = fg_known_off && !present_space;
        in.no_reference = !expected_layer_present && !expected_layer_token;
        layer_ticket.refused = ui_ticket::first_refusal(in);
      }
      if (diagnostic) candidates.push_back({{"source", "ui_layer"}, {"channel", "alpha"}, {"format", layer.format},
        {"candidate_bit", ui_detection::candidate::layer}, {"available_for_detection", view.handle != 0},
        {"association", "previous_frame_offscreen_ui_layer"}, {"capture_id", layer.capture_id},
        {"presents_since_copy", layer.presents_since_copy}, {"age_ms", now >= layer.tick ? now - layer.tick : 0},
        {"ticket", ticket_json(layer_ticket, layer.stamped)},
        {"executed_queue", layer.executed_queue}, {"expected_present_label", expected_layer_present},
        {"expected_token_label", expected_layer_token}});
      if (view.handle) {
        result.detection.layer = view;
        result.detection.layer_flags = ui_layer::detection_flags(static_cast<api::format>(layer.format));
        signatures.set(ui_selection::kind::ui_layer, typed_format(layer.format));
        result.status.retained_alpha_ready = true; available = true;
      }
    }
    // T1 by Present counting without a HUD-less pairing: a render that offers
    // nothing, within the reported generated count of the last one that
    // offered a Streamline UI tag, is a generated Present and shows that
    // real frame's decision. It carries no real-frame id (no tag bound); the
    // count bounds it.
    const bool tag_offered = result.detection.masks[0].handle || result.detection.masks[1].handle ||
      result.detection.masks[2].handle;
    std::string identity_line, interposer_line;
    if (capturing && !available && !result.detection.hold_previous &&
        ui_mask::generated_without_input(frame_sequence, input_present, status.fg_active(), status.fg.generated_frames)) {
      if (diagnostic) candidates.push_back({{"source", "none"}, {"held_for_generated_present", true},
        {"association", "present_count_after_last_ui_tag"}, {"presents_since_ui_tag", frame_sequence - input_present}});
      result.status.retained_alpha_ready = true;
      result.detection.hold_previous = true;
      result.detection.real_frame = 0;
      available = true;
    }
    const bool current_allowed = status.requested && !status.fg_active() && present_has_alpha(base.output_format) &&
      (wanted_source == choice::automatic || wanted_source == choice::current_color);
    if (current_allowed) {
      // S3: the presented colour carries this render's own Present label.
      auto &current_ticket = tickets[ui_ticket::slot::current];
      current_ticket.kind = ui_ticket::capture_kind::present_color; current_ticket.source = ui_selection::kind::current;
      current_ticket.at = ui_ticket::boundary::present; current_ticket.how = ui_ticket::proof::present_boundary;
      current_ticket.present = ui_ticket::present_label(ui_ticket::boundary::present, 0, present_label);
      current_ticket.token = {};
      current_ticket.present_queue = ui_ticket::queue_relation::same;
      current_ticket.typed_format = typed_format(base.output_format); current_ticket.color_space = base.color_space;
      current_ticket.epoch = base.epoch; current_ticket.viewport = base.viewport;
      current_ticket.refused = current_ticket.present.valid ? ui_ticket::refusal::none : ui_ticket::refusal::unstamped;
      offered[ui_ticket::slot::current] = true;
      if (diagnostic) candidates.push_back({{"source", "current_color"}, {"candidate_bit", ui_detection::candidate::current},
        {"available_for_detection", true}, {"association", "current_color_allocation"}, {"format", base.output_format},
        {"ticket", ticket_json(current_ticket, false)}});
      result.detection.current_color = true; available = true;
      signatures.set(ui_selection::kind::current, typed_format(base.output_format));
    }
    // A manual choice observes its selected capture's own provenance.
    const auto observe_origin = [&](const ui_mask::selection &selected, bool dedicated_mask) {
      const auto &source = selected.origin.source;
      auto origin = input;
      origin.retained = true; origin.dedicated_mask = dedicated_mask;
      origin.epoch = source.epoch; origin.revision = source.observation_revision;
      origin.viewport = source.viewport; origin.sequence = source.sequence; origin.tick_ms = source.tick;
      result.explicit_origin = origin;
    };
    if (manual) {
      // The explicit first filtered candidate in draw order: UIAlpha, UI color
      // tag, Backbuffer, current color, then HUD-less (which needs detection).
      bool chosen = false;
      for (unsigned slot = 0; slot != 3 && !chosen; ++slot) {
        const auto &pick = explicit_captures[slot];
        if (!pick.set) continue;
        const auto kind = kinds[slot];
        candidate = captured_scope(base, pick.selected); result.view = pick.view;
        result.kind = kind == ui_mask::source_kind::backbuffer ? ui_input_kind::captured_color_alpha : ui_input_kind::dedicated_mask;
        result.channel = kind == ui_mask::source_kind::alpha ? ui_mask_channel::red : ui_mask_channel::alpha;
        result.status.input = kind == ui_mask::source_kind::alpha ? source_alpha_input::sl_ui_alpha :
          kind == ui_mask::source_kind::color_and_alpha ? source_alpha_input::sl_ui_color_alpha : source_alpha_input::sl_backbuffer_alpha;
        result.source_metadata = pick.metadata;
        observe_origin(pick.selected, result.kind == ui_input_kind::dedicated_mask);
        chosen = true;
      }
      if (!chosen && current_allowed) {
        candidate.source = choice::current_color; chosen = true;
        result.kind = ui_input_kind::current_color_alpha; result.status.input = source_alpha_input::present_alpha;
      }
      if (!chosen && explicit_captures[3].set) {
        const auto &pick = explicit_captures[3];
        candidate = captured_scope(base, pick.selected); result.view = pick.view;
        result.kind = ui_input_kind::hudless_difference;
        result.status.input = source_alpha_input::sl_hudless_difference;
        result.source_metadata = pick.metadata;
        observe_origin(pick.selected, false);
        chosen = true;
      }
      if (!chosen) result.status.input = source_alpha_input::none;
    } else {
      candidate.source = wanted_source;
      result.status.input = available ? source_alpha_input::automatic_mask : source_alpha_input::none;
    }
    // The derived mask is current-frame GPU output. Delayed CPU quality
    // feedback must never be promoted into exact current-winner provenance.
    input.retained = false; input.tick_ms = now; input.sequence = frame_sequence;
    // S3 shadow: today's view of this render beside the ticket's.
    today.holds = result.detection.hold_previous;
    today.detects = today.detects && today.offered;
    const auto frame_id = ui_ticket::newest_token(tickets);
    const bool ticket_batch = token_batch && hudless_token && batch_hudless.token_generation == hudless_token;
    const bool ticket_pair_exact = today.offered && (ticket_batch || hudless_present_space);
    result.detection.tickets = tickets;
    result.detection.expected_layer_present = expected_layer_present;
    result.detection.expected_layer_token = expected_layer_token;
    result.detection.expected_hudless_present = expected_hudless_present;
    nlohmann::json identity_json;
    if (diagnostic) identity_json = {
      {"meaning", "S3 shadow (docs/reshade-sbs.md, UI decision framework, S3 snapshot ticket): this render's pairing by "
        "Present counting (today) beside what the snapshot tickets would decide (ticket). Nothing decides from the ticket "
        "while identity_authoritative is false. expected holds the labels proposed for the GPU to verify against the "
        "stamps (0: not proposed)."},
      {"authoritative", authoritative}, {"present_label", present_label}, {"fg_off_run", fg_off_run},
      {"interposers", ui_ticket::interposer_names(interposers)},
      {"today", {{"offered", today.offered}, {"detects", today.detects && !today.holds}, {"holds", today.holds},
        {"exact", today.exact}, {"batch", today.batch}, {"real_frame", today.real_frame}}},
      {"ticket", {{"frame_id", frame_id}, {"paired", ticket_pair_exact}, {"batch", ticket_batch},
        {"token_batch", token_batch ? nlohmann::json(batch_hudless.token_generation) : nlohmann::json(nullptr)},
        {"pair_space", ticket_batch ? "token" : hudless_present_space ? "present" : "none"}}},
      {"expected", {{"hudless_present", expected_hudless_present}, {"layer_present", expected_layer_present},
        {"layer_token", expected_layer_token}}}};
    if (diagnostic && !manual) result.source_metadata = nlohmann::json{
      {"source", "automatic_candidate_set"}, {"association", "current_render_gpu_validation"},
      {"meaning", "Current render validates all admitted candidates and produces its mask on the GPU. Delayed quality statistics do not identify the exact current winner or authorize pixels."},
      {"candidates", candidates}, {"identity_shadow", identity_json}}.dump();
    {
      std::lock_guard<std::mutex> lock(source_mutex);
      const auto found = sources.find(runtime);
      if (found != sources.end()) {
        auto &entry = found->second;
        if (hudless_wanted) ++entry.hudless_outcomes[hudless_result];
        if (tag_offered) {
          entry.input_present = frame_sequence; entry.input_epoch = base.epoch; entry.input_viewport = base.viewport;
        }
        // S3 shadow counters: a scope change ends the identity run.
        if (entry.identity_epoch != base.epoch || entry.identity_viewport != base.viewport) {
          entry.identity.end_scope(entry.identity_totals);
          entry.identity_epoch = base.epoch; entry.identity_viewport = base.viewport;
        }
        if (shadow && (capturing || layer_wanted)) {
          entry.identity.step(today, frame_id, ticket_pair_exact, ticket_batch, entry.identity_totals);
          for (std::size_t i = 0; i != tickets.size(); ++i) if (offered[i]) ui_ticket::count_ticket(tickets[i], entry.identity_totals);
        }
        if (shadow) entry.identity_totals[ui_ticket::identity_counter::interposers] = interposers;
        // The GPU verdicts committed since the last render join the gpu group.
        // A recreated renderer restarts its cumulative counts from zero.
        if (shadow) {
          namespace n = ui_ticket::identity_counter;
          const auto gpu = renderer.identity_counts();
          for (std::size_t i = n::gpu_exact; i <= n::gpu_token_exact; ++i) {
            const auto seen = entry.identity_gpu_seen[i];
            entry.identity_totals[i] += gpu[i] >= seen ? gpu[i] - seen : gpu[i];
            entry.identity_gpu_seen[i] = gpu[i];
          }
        }
        if (shadow && now >= entry.next_identity_log && entry.identity_totals != entry.identity_logged &&
            entry.identity_totals[ui_ticket::identity_counter::renders]) {
          entry.next_identity_log = now + 5000;
          entry.identity_logged = entry.identity_totals;
          identity_line = "Sunshine UI identity: runtime=" + pointer_text(runtime) + ' ' +
            ui_ticket::format_identity_counters(entry.identity_totals);
        }
        if (shadow && (interposers != entry.logged_interposers || status.fg.known != entry.logged_fg_known ||
            status.fg.enabled != entry.logged_fg_enabled)) {
          entry.logged_interposers = interposers; entry.logged_fg_known = status.fg.known; entry.logged_fg_enabled = status.fg.enabled;
          interposer_line = "Sunshine FG interposers: runtime=" + pointer_text(runtime) + ' ' +
            ui_ticket::format_interposers(interposers, status.fg.known, status.fg.enabled);
        }
        const bool matches = entry.instance == instance && !entry.suspended && entry.base == base &&
          entry.selection.snapshot().choice_revision == before.choice_revision;
        if (matches && available) {
          entry.selection.observe(candidate, status.requested);
          entry.last_observe_tick = input.tick_ms; expire_available(entry, now);
        } else { entry.selection.unavailable(); entry.last_observe_tick = 0; }
        result.status.qualification = entry.selection.snapshot();
        if (!matches || !result.status.qualification.available) available = false;
      } else available = false;
    }
    if (!available) {
      result.kind = ui_input_kind::unavailable; result.view = {}; result.explicit_origin.reset();
      result.detection = {}; result.detection.current_color = false;
      result.status.retained_alpha_ready = false; result.status.input = source_alpha_input::none;
    }
    if (!interposer_line.empty()) sunshine_log::message(reshade::log::level::info, interposer_line.c_str());
    if (!identity_line.empty()) sunshine_log::message(reshade::log::level::info, identity_line.c_str());
    ui_mask::diagnostic_snapshot latest;
    const bool have_diagnostic = status.requested && ui_mask::query_diagnostic(reinterpret_cast<std::uint64_t>(runtime), latest);
    bool log_gate = false;
    std::array<std::uint32_t, hudless_outcome_count> outcomes{};
    if (have_diagnostic) {
      std::lock_guard<std::mutex> lock(source_mutex);
      if (auto found = sources.find(runtime); found != sources.end() && now >= found->second.next_gate_log) {
        found->second.next_gate_log = now + 5000;
        outcomes = std::exchange(found->second.hudless_outcomes, {});
        log_gate = true;
      }
    }
    if (log_gate) {
      const auto &gate = latest.hook_gate;
      // S3: where tag begin() calls stopped without an attempt (process
      // counts per stage), and this request's latest such stage.
      std::string refused = "begin_refused={";
      for (std::size_t stage = 1; stage != latest.begin_refusals.size(); ++stage) {
        if (stage != 1) refused += ' ';
        refused.append(ui_ticket::begin_stage_names[stage]).append("=").append(std::to_string(latest.begin_refusals[stage]));
      }
      refused.append("} begin_last=").append(ui_ticket::name(latest.begin_refusal));
      char message[1152]{};
      std::snprintf(message, sizeof(message),
        "Sunshine UI capture gate: runtime=%p request_generation=%llu wanted={epoch=%llu revision=%llu viewport=%u device=%llu size=%ux%u kinds=0x%x} hook={state=%s epoch=%llu revision=%llu viewport=%u sequence=%llu tick=%llu kinds=0x%x matches=%u} boundary=%llu attempted=%u recorded=%u hudless_presents={batch=%u real=%u late=%u generated=%u stale=%u other=%u none=%u} %s; gate metadata does not authorize pixels",
        static_cast<void *>(runtime), static_cast<unsigned long long>(latest.request_generation),
        static_cast<unsigned long long>(latest.wanted.epoch), static_cast<unsigned long long>(latest.wanted.revision),
        latest.wanted.viewport, static_cast<unsigned long long>(latest.wanted.device_identity),
        latest.wanted.width, latest.wanted.height, latest.wanted.allowed_kinds,
        ui_mask::name(gate.state), static_cast<unsigned long long>(gate.epoch), static_cast<unsigned long long>(gate.revision),
        gate.viewport, static_cast<unsigned long long>(gate.sequence), static_cast<unsigned long long>(gate.tick),
        gate.seen_kinds, gate.matching_requests, static_cast<unsigned long long>(latest.latest_boundary.source.sequence),
        unsigned(latest.record_attempted), unsigned(latest.record_completed),
        outcomes[hudless_batch], outcomes[hudless_real], outcomes[hudless_late], outcomes[hudless_generated],
        outcomes[hudless_stale], outcomes[hudless_other], outcomes[hudless_none], refused.c_str());
      sunshine_log::message(reshade::log::level::info, message);
    }
    if (have_diagnostic && latest.latest_boundary.source.sequence) {
      result.status.input_state = source_alpha_input_state::seen;
      if (latest.record_completed) {
        if (latest.record.result == capture::status::conflicting_state) result.status.input_state = source_alpha_input_state::state_conflict;
        else if (latest.record.result != capture::status::recorded) result.status.input_state = source_alpha_input_state::rejected;
      }
      if (latest.sdk_result_known && !latest.sdk_successful) result.status.input_state = source_alpha_input_state::rejected;
    }
    if (diagnostic && have_diagnostic) {
      const auto &wanted = latest.wanted; const auto &boundary = latest.latest_boundary; const auto &source = boundary.source;
      result.capture_metadata = nlohmann::json{
        {"meaning", "Latest live input attempt, independent of the current GPU-selected mask and other available candidates."},
        {"request_generation", latest.request_generation},
        {"hook_gate", {{"state", ui_mask::name(latest.hook_gate.state)},
          {"epoch", latest.hook_gate.epoch}, {"revision", latest.hook_gate.revision},
          {"sequence", latest.hook_gate.sequence}, {"tick_ms", latest.hook_gate.tick},
          {"viewport", latest.hook_gate.viewport}, {"seen_kinds", latest.hook_gate.seen_kinds},
          {"matching_requests", latest.hook_gate.matching_requests},
          {"meaning", "Latest global SDK tag entry; it need not match this request. seen_kinds bits: 1 Backbuffer, 2 UIColorAndAlpha, 4 UIAlpha, 8 HUDLessColor."}}},
        {"request", {{"epoch", wanted.epoch}, {"revision", wanted.revision}, {"viewport", wanted.viewport},
          {"device_identity", wanted.device_identity}, {"width", wanted.width}, {"height", wanted.height}}},
        {"input_observed", source.sequence != 0}, {"sequence", source.sequence}, {"tick_ms", source.tick},
        {"source_native", source.resource.native}, {"command", boundary.command}, {"tag_scope", boundary.tag_scope},
        {"tag_type", static_cast<std::uint32_t>(boundary.kind)},
        {"record_attempted", latest.record_attempted}, {"record_completed", latest.record_completed},
        {"sdk_success", latest.sdk_result_known ? nlohmann::json(latest.sdk_successful) : nlohmann::json(nullptr)},
        {"capture_diagnostic", latest.record_completed ? capture_diagnostic_json(latest.record) : nlohmann::json(nullptr)}
      }.dump();
    }
    return result;
  }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  // The D3D12 runtime fixture's view of a runtime's S3 shadow totals.
  extern "C" __declspec(dllexport) BOOL SunshineUIInputTestIdentity(api::effect_runtime *runtime,
      ui_ticket::identity_counters *out) {
    if (!runtime || !out) return FALSE;
    std::lock_guard<std::mutex> lock(source_mutex);
    const auto found = sources.find(runtime);
    if (found == sources.end()) return FALSE;
    *out = found->second.identity_totals;
    return TRUE;
  }
#endif
  ui_adaptive::source frame::match_scene(ui_adaptive::source source, bool scene_ready) const {
    source.now_ms = observation.now_ms; source.eligible = source.eligible && scene_ready;
    // A retained capture, Manual On's explicit one included (the renderer
    // applies it directly where detection cannot run), matches only its own
    // scope and is no fresher than its capture.
    const auto *retained = explicit_origin ? &*explicit_origin :
      !automatic_detection && observation.retained ? &observation : nullptr;
    if (retained) {
      source.mask_sequence = retained->sequence; source.tick_ms = std::min(source.tick_ms, retained->tick_ms);
      source.eligible = source.eligible && retained->epoch == source.epoch &&
        retained->revision == source.revision && retained->viewport == source.viewport;
    } else if (automatic_detection) {
      source.mask_sequence = observation.sequence;
    }
    return source;
  }
  ui_render_input frame::for_render(const ui_plane_parameters &plane, const ui_adaptive::source &adaptive) {
    const bool requested = status.requested && status.mode != source_alpha_mode::off;
    // Recheck presentation policy at the normalized handoff independently of
    // delayed CPU quality feedback. Captured candidates remain eligible.
    if (status.fg_active()) detection.current_color = false;
    const bool direct = requested && !(status.fg_active() && kind == ui_input_kind::current_color_alpha);
    // A HUD-less override also carries detection inputs, so a generated
    // present can hold the real frame's mask.
    const bool detected = automatic_detection || kind == ui_input_kind::hudless_difference;
    return {direct ? kind : ui_input_kind::unavailable, direct ? view : api::resource_view{}, plane,
      &observation, &adaptive, channel, requested && detected ? &detection : nullptr};
  }
  void frame::complete(const renderer &renderer, bool rendered) {
    status.rendered = rendered; status.applied = rendered && renderer.consumed_source_alpha_ui();
    status.automatic = true;
    status.coverage = rendered ? renderer.consumed_alpha_auto() : observation.session->decision();
    // The live layer copy this Present read (bound directly or copied into the
    // renderer's slot, recorded on the renderer's list either way) stays
    // unwritten until the renderer's next completion signal passes.
    if (layer_capture)
      ui_layer::bound(layer_device, layer_capture, renderer.completion_fence(), renderer.completion_value());
  }
}
