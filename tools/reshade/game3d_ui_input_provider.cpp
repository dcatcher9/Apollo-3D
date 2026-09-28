// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_input_provider.h"
#include "game3d_ui_mask.h"
#include "game3d_capture_diagnostic.h"
#include "streamline_camera_probe.h"
#include <d3d12.h>
#include <reshade.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace sunshine_game3d::ui_input {
  namespace api = reshade::api;
  namespace capture = sunshine_streamline::depth_capture;
  namespace {
    using choice = ui_qualification::choice;
    struct source_entry {
      ui_qualification::session selection;
      ui_qualification::scope base;
      std::uint64_t instance{}, last_observe_tick{}, frame_sequence{}, next_gate_log{};
      bool suspended{};
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
        case choice::sl_hudless: return ui_mask::source_mask(ui_mask::source_kind::hudless);
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
    ui_qualification::scope captured_scope(ui_qualification::scope base, const ui_mask::selection &selected) {
      const auto &origin = selected.origin;
      base.source = choice_for(origin.kind); base.provider = 1; base.source_id = origin.source.source_id;
      base.epoch = origin.source.epoch; base.revision = origin.source.observation_revision; base.viewport = origin.source.viewport;
      base.channel = origin.kind == ui_mask::source_kind::alpha ? 1 : 0;
      base.mask_width = selected.texture.width; base.mask_height = selected.texture.height; base.mask_format = selected.texture.format;
      base.left = selected.texture.area.left; base.top = selected.texture.area.top;
      base.width = selected.texture.area.width; base.height = selected.texture.area.height;
      base.semantic_contract = origin.kind == ui_mask::source_kind::alpha ? 0 : 1;
      return base;
    }
    nlohmann::json captured_metadata(const ui_mask::selection &selected, std::uint64_t now, bool admitted, bool paired) {
      const auto &origin = selected.origin; const auto &source = origin.source; const auto &copy = selected.texture;
      return {
        {"source", ui_qualification::name(choice_for(origin.kind))},
        {"tag_type", static_cast<std::uint32_t>(origin.kind)}, {"tag_scope", origin.tag_scope},
        {"channel", origin.kind == ui_mask::source_kind::hudless ? "rgb_difference" : origin.kind == ui_mask::source_kind::alpha ? "red" : "alpha"}, {"format", copy.format},
        {"available_for_detection", admitted}, {"current_present_pair", paired},
        {"association", origin.kind == ui_mask::source_kind::hudless ? "same_queue_current_present_generation" : "latest_submitted_input_approximation"},
        {"epoch", source.epoch}, {"observation_revision", source.observation_revision},
        {"sequence", source.sequence}, {"tick_ms", source.tick}, {"age_ms", now >= source.tick ? now - source.tick : 0},
        {"viewport", source.viewport}, {"source_native", source.resource.native}, {"source_command", origin.command},
        {"source_frame_generation", source.source_frame_generation}, {"source_frame_token", source.source_frame_token},
        {"source_frame_numeric", source.source_frame_numeric}, {"source_frame_has_numeric", source.source_frame_has_numeric},
        {"source_frame_explicit", source.source_frame_explicit}, {"capture_id", copy.capture_id},
        {"source_present_generation", origin.source_present_generation},
        {"current_source_present_generation", selected.current_source_present_generation},
        {"producer_queue", copy.producer_queue}, {"producer_fence", copy.producer_fence},
        {"producer_completed", copy.producer_completed}, {"producer_recording_retired", copy.producer_recording_retired}
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
    {
      std::lock_guard<std::mutex> lock(source_mutex);
      auto &entry = entry_for(runtime); entry.suspended = false;
      if (entry.base != base) { entry.selection.clear(); entry.last_observe_tick = 0; entry.base = base; }
      filter = source_filter(entry.selection.selected_choice());
    }
    if (!capture_needed(status, device->get_api()) || !have_scope || !filter) {
      ui_mask::invalidate(reinterpret_cast<std::uint64_t>(runtime)); return;
    }
    capture::record_diagnostic identity;
    const auto reference = capture::retain_source(color.handle, &identity);
    ui_mask::set_request({reinterpret_cast<std::uint64_t>(runtime), identity.expected_device_identity,
      observed.epoch, observed.revision, observed.viewport, desc.texture.width, desc.texture.height, true, 0, true, filter});
  }
  frame acquire(api::effect_runtime *runtime, renderer &renderer, source_alpha_ui_decision status,
      alpha_auto_policy &session, bool diagnostic) {
    frame result;
    result.status = status; result.automatic_detection = status.mode == source_alpha_mode::automatic;
    result.detection.current_color = false;
    ui_qualification::scope base;
    ui_qualification::status before;
    std::uint64_t instance{}, frame_sequence{};
    {
      std::lock_guard<std::mutex> lock(source_mutex);
      const auto found = sources.find(runtime);
      if (found != sources.end()) {
        base = found->second.base; before = found->second.selection.snapshot();
        instance = found->second.instance; frame_sequence = ++found->second.frame_sequence;
      }
    }
    const auto now = GetTickCount64();
    auto &input = result.observation;
    input.now_ms = input.tick_ms = now; input.session = &session;
    input.epoch = base.epoch; input.revision = base.revision; input.viewport = base.viewport; input.sequence = frame_sequence;
    const auto wanted_source = before.selected;
    auto candidate = base;
    bool available = false;
    std::uint64_t signature = base.epoch ^ (base.revision << 1) ^ (std::uint64_t(base.viewport) << 32);
    nlohmann::json candidates = nlohmann::json::array();
    auto *queue = runtime->get_command_queue();
    auto *commands = queue->get_immediate_command_list();
    constexpr ui_mask::source_kind kinds[]{ui_mask::source_kind::alpha, ui_mask::source_kind::color_and_alpha,
      ui_mask::source_kind::backbuffer, ui_mask::source_kind::hudless};
    if (capture_needed(status, runtime->get_device()->get_api())) for (unsigned slot = 0; slot != 4; ++slot) {
      const auto kind = kinds[slot];
      if (!(source_filter(wanted_source) & ui_mask::source_mask(kind))) continue;
      ui_mask::selection selected;
      if (!ui_mask::acquire_kind(reinterpret_cast<std::uint64_t>(runtime), kind, selected, now, queue->get_native())) continue;
      const bool hudless = kind == ui_mask::source_kind::hudless;
      const bool paired = selected.texture.producer_queue == queue->get_native() &&
        selected.origin.source_present_generation && selected.origin.source_present_generation != UINT64_MAX &&
        selected.current_source_present_generation == selected.origin.source_present_generation + 1;
      // Tag 69 is not authenticated against this game's caller SDK. Its physical
      // availability cannot hide independently usable lower-priority candidates.
      const bool admissible = (!result.automatic_detection || kind != ui_mask::source_kind::alpha) && (!hudless || paired);
      api::resource_view view{};
      if (admissible) view = renderer.prepare_ui_candidate(slot, selected.ticket.id, [&](api::resource destination) {
        return capture::copy_local_texture(commands->get_native(), queue->get_native(), selected.ticket,
          destination.handle, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      }, static_cast<api::format>(selected.texture.format));
      if (diagnostic) candidates.push_back(captured_metadata(selected, now, view.handle != 0, paired));
      if (!view.handle) continue;
      signature ^= (std::uint64_t(1) << (slot + 48)) ^ (selected.origin.source.source_id * (slot + 1));
      result.status.retained_alpha_ready = true;
      if (result.automatic_detection) {
        if (hudless) result.detection.hudless = view;
        else result.detection.masks[slot] = view;
        available = true;
      } else if (!available) {
        available = true; candidate = captured_scope(base, selected); result.view = view;
        result.kind = hudless ? ui_input_kind::hudless_difference : kind == ui_mask::source_kind::backbuffer ?
          ui_input_kind::captured_color_alpha : ui_input_kind::dedicated_mask;
        result.channel = kind == ui_mask::source_kind::alpha ? ui_mask_channel::red : ui_mask_channel::alpha;
        result.status.input = hudless ? source_alpha_input::sl_hudless_difference : kind == ui_mask::source_kind::alpha ?
          source_alpha_input::sl_ui_alpha : kind == ui_mask::source_kind::color_and_alpha ?
          source_alpha_input::sl_ui_color_alpha : source_alpha_input::sl_backbuffer_alpha;
        input.retained = true; input.dedicated_mask = result.kind == ui_input_kind::dedicated_mask;
        input.epoch = selected.origin.source.epoch; input.revision = selected.origin.source.observation_revision;
        input.viewport = selected.origin.source.viewport; input.sequence = selected.origin.source.sequence; input.tick_ms = selected.origin.source.tick;
        if (diagnostic) result.source_metadata = candidates.back().dump();
      }
    }
    const bool current_allowed = status.requested && !status.fg_active() && present_has_alpha(base.output_format) &&
      (wanted_source == choice::automatic || wanted_source == choice::current_color);
    if (current_allowed) {
      signature ^= std::uint64_t(1) << 52;
      if (diagnostic) candidates.push_back({{"source", "current_color"}, {"available_for_detection", true},
        {"association", "current_color_allocation"}, {"format", base.output_format}});
      if (result.automatic_detection) { result.detection.current_color = true; available = true; }
      else if (!available) {
        candidate.source = choice::current_color; available = true;
        result.kind = ui_input_kind::current_color_alpha; result.status.input = source_alpha_input::present_alpha;
      }
    }
    if (result.automatic_detection) {
      candidate.source = wanted_source;
      result.status.input = available ? source_alpha_input::automatic_mask : source_alpha_input::none;
      // The derived mask is current-frame GPU output. Delayed CPU quality
      // feedback must never be promoted into exact current-winner provenance.
      input.retained = false; input.tick_ms = now; input.sequence = frame_sequence;
      if (diagnostic) result.source_metadata = nlohmann::json{
        {"source", "automatic_candidate_set"}, {"association", "current_render_gpu_validation"},
        {"meaning", "Current render validates all admitted candidates and produces its mask on the GPU. Delayed quality statistics do not identify the exact current winner or authorize pixels."},
        {"candidates", candidates}}.dump();
    }
    result.candidate_signature = signature;
    {
      std::lock_guard<std::mutex> lock(source_mutex);
      const auto found = sources.find(runtime);
      if (found != sources.end()) {
        auto &entry = found->second;
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
      result.kind = ui_input_kind::unavailable; result.view = {};
      result.detection = {}; result.detection.current_color = false;
      result.status.retained_alpha_ready = false; result.status.input = source_alpha_input::none;
    }
    ui_mask::diagnostic_snapshot latest;
    const bool have_diagnostic = status.requested && ui_mask::query_diagnostic(reinterpret_cast<std::uint64_t>(runtime), latest);
    bool log_gate = false;
    if (have_diagnostic) {
      std::lock_guard<std::mutex> lock(source_mutex);
      if (auto found = sources.find(runtime); found != sources.end() && now >= found->second.next_gate_log) {
        found->second.next_gate_log = now + 5000;
        log_gate = true;
      }
    }
    if (log_gate) {
      const auto &gate = latest.hook_gate;
      char message[768]{};
      std::snprintf(message, sizeof(message),
        "Sunshine UI capture gate: runtime=%p request_generation=%llu wanted={epoch=%llu revision=%llu viewport=%u device=%llu size=%ux%u kinds=0x%x} hook={state=%s epoch=%llu revision=%llu viewport=%u sequence=%llu tick=%llu kinds=0x%x matches=%u} boundary=%llu attempted=%u recorded=%u; gate metadata does not authorize pixels",
        static_cast<void *>(runtime), static_cast<unsigned long long>(latest.request_generation),
        static_cast<unsigned long long>(latest.wanted.epoch), static_cast<unsigned long long>(latest.wanted.revision),
        latest.wanted.viewport, static_cast<unsigned long long>(latest.wanted.device_identity),
        latest.wanted.width, latest.wanted.height, latest.wanted.allowed_kinds,
        ui_mask::name(gate.state), static_cast<unsigned long long>(gate.epoch), static_cast<unsigned long long>(gate.revision),
        gate.viewport, static_cast<unsigned long long>(gate.sequence), static_cast<unsigned long long>(gate.tick),
        gate.seen_kinds, gate.matching_requests, static_cast<unsigned long long>(latest.latest_boundary.source.sequence),
        unsigned(latest.record_attempted), unsigned(latest.record_completed));
      reshade::log::message(reshade::log::level::info, message);
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
  ui_adaptive::source frame::match_scene(ui_adaptive::source source, bool scene_ready) const {
    source.now_ms = observation.now_ms; source.eligible = source.eligible && scene_ready;
    if (automatic_detection) {
      source.mask_sequence = observation.sequence;
      source.revision ^= candidate_signature;
    } else if (observation.retained) {
      source.mask_sequence = observation.sequence; source.tick_ms = std::min(source.tick_ms, observation.tick_ms);
      source.eligible = source.eligible && observation.epoch == source.epoch &&
        observation.revision == source.revision && observation.viewport == source.viewport;
    }
    return source;
  }
  ui_render_input frame::for_render(const ui_plane_parameters &plane, const ui_adaptive::source &adaptive) {
    const bool requested = status.requested && status.mode != source_alpha_mode::off;
    // Recheck presentation policy at the normalized handoff independently of
    // delayed CPU quality feedback. Captured candidates remain eligible.
    if (status.fg_active()) detection.current_color = false;
    const bool direct = requested && !(status.fg_active() && kind == ui_input_kind::current_color_alpha);
    return {direct ? kind : ui_input_kind::unavailable, direct ? view : api::resource_view{}, plane,
      &observation, &adaptive, channel, requested && automatic_detection ? &detection : nullptr};
  }
  void frame::complete(const renderer &renderer, bool rendered) {
    status.rendered = rendered; status.applied = rendered && renderer.consumed_source_alpha_ui();
    status.automatic = true;
    status.coverage = rendered ? renderer.consumed_alpha_auto() : observation.session->decision(observation.now_ms);
  }
}
