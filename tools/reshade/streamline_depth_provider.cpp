// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_depth_provider.h"
#include "streamline_depth_provider_log.h"
#include "streamline_camera_probe.h"
#include "depth_hold.h"
#include "game3d_frame_record.h"
#include "streamline_depth_capture.h"
#include "streamline_native_observer.h"
#include "depth_content_sampler.h"
#include "depth_ready_uniform_cache.h"
#include "diagnostic_log_gate.h"
#include "freshness.h"
#include "projection_depth_scale.h"
#include <reshade.hpp>
#include "async_log.h"
#include "game3d_slow_step.h"
#include "addon_lifetime.h"
#include <d3d12.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <cstdio>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  #include "game3d_test_depth_events.h"
#endif

namespace sunshine_streamline::provider {
  namespace api = reshade::api;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  namespace depth_events = sunshine_game3d::test::depth_events;
#endif
  // A provider handoff (one API source released, another about to establish
  // on the same queue) keeps API ownership in mono at most this long instead
  // of starting the generic selector (docs/reshade-sbs.md, depth handoff).
  using sunshine_freshness::handoff_ms;
  namespace handoff_contract {
    constexpr handoff_evidence base{true, true, 1200, 1000, 1920, 1080, 1920, 1080};
    constexpr handoff_evidence with(handoff_evidence e, int field) {
      switch (field) {
        case 0: e.previously_owned = false; break;
        case 1: e.evaluation_live = false; break;
        case 2: e.now_ms = 1251; break;
        case 3: e.associated_ms = 0; break;
        case 4: e.width = 1280; break;
        case 5: e.now_ms = 999; break;
        default: e.now_ms = 1250; break;
      }
      return e;
    }
    static_assert(handoff_hold(base, handoff_ms) && handoff_hold(with(base, 6), handoff_ms));
    static_assert(!handoff_hold(with(base, 0), handoff_ms) && !handoff_hold(with(base, 1), handoff_ms) &&
      !handoff_hold(with(base, 2), handoff_ms) && !handoff_hold(with(base, 3), handoff_ms) &&
      !handoff_hold(with(base, 4), handoff_ms) && !handoff_hold(with(base, 5), handoff_ms));
  }
    struct __declspec(uuid("e82eab46-4d62-4e80-9e02-0e60b7059f36")) state {
      depth_capture::packet frame;
      sunshine_scene_depth::frame source_metadata;
      sunshine_depth::frame_depth output;
      sunshine_game3d::depth_output::display_depth_cache cache;
      sunshine_depth_stats::presentation_window presentations;
      std::array<std::uint64_t, 5> presentation_source{};
      source_description last_valid;
      center_sample latest;
      depth_capture::input pending_metadata;
      sunshine_depth_statistics::tile_layout pending_grid;
      std::uint64_t pending_id{}, pending_texture{}, next_sample{};
      bool owns_pass{}, open{}, have_latest{};
      frame_generation_policy source_policy;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      // The last begin's FG decision (test_frame_generation_decision).
      unsigned test_fg_decision{};
#endif
      // The required FG scope was silent at the last Present. Its turns are
      // counted; the line comes at most every 5 s, naming the newest state.
      bool fg_scope_silent{};
      std::uint64_t fg_scope_changes{}, fg_scope_logged_changes{}, fg_scope_line_tick{};
      // The game's newest FG mode (0 before any options call) and the
      // Presents after its last change (track_fg_switch).
      std::uint32_t fg_mode{};
      fg_switch_line fg_switch;
      api::resource display_texture{};
      api::resource_view display_view{}, bound_view{};
      sunshine_depth::ready_uniform_cache<api::effect_uniform_variable> depth_ready;
      std::uint32_t display_width{}, display_height{}, display_format{}, display_view_format{};
      // Display storage per shape/format (NGX and Streamline usually differ),
      // so a provider flip reuses the other entry instead of reallocating
      // (until that entry is unused for 2 s: display_used, below).
      // The active entry is mirrored in display_texture/display_view above.
      struct display_entry {
        api::resource texture{};
        api::resource_view view{};
        std::uint32_t width{}, height{}, format{}, view_format{};
      };
      std::array<display_entry, 2> displays{};
      // The view last passed to update_texture_bindings by this provider.
      api::resource_view effects_view{};
      // Replaced display storage, released on a pool thread once the runtime
      // queue passed a fence signaled on a later Present (never a CPU wait on
      // this path). bytes: its texture's size.
      struct retired_display {
        api::resource texture{};
        api::resource_view view{};
        std::uint64_t present{}, fence_value{}, bytes{};
      };
      std::vector<retired_display> retired_displays;
      // Created with the state (initialize), not at a Present.
      api::fence retire_fence{};
      std::uint64_t retire_value{};
      // The pool jobs releasing this runtime's retired displays
      // (collect_displays); destroy() waits until none is outstanding.
      struct release_jobs {
        std::mutex mutex;
        std::condition_variable idle;
        unsigned outstanding{};
      };
      std::shared_ptr<release_jobs> releases{std::make_shared<release_jobs>()};
      // Handoff hysteresis: the last associated Present and its color size.
      std::uint64_t associated_tick{};
      std::uint32_t associated_width{}, associated_height{};
      bool handoff_hold{};
      sunshine_diagnostics::log_gate logging;
      depth_capture::status last_status{depth_capture::status::inactive};
      depth_capture::capture_failure last_capture_failure{depth_capture::capture_failure::none};
      // Observation-loss revision at the previous status line, to report the rate.
      std::uint64_t logged_loss_revision{};
      depth_capture::selection_reason last_selection{depth_capture::selection_reason::not_attempted};
      display_status last_display{display_status::not_attempted};
      depth_capture::consumer_status last_consumer{depth_capture::consumer_status::not_attempted};
      depth_capture::recording_loss last_loss{depth_capture::recording_loss::none};
      bool last_output_ready{};
      // Diagnostic transition state advances on every decision, independently
      // of the ordinary status log's coalescing gate.
      bool traced_ready{}, traced_loss{};
      std::uint64_t next_trace_episode{}, trace_loss_tick{}, suppressed_trace_episodes{};
      observation_loss_episode trace_observation_loss;
      render_size_state render_size;
      // Per displays entry: the tick of the last Present at which it was the
      // active display or the effects' binding (collect_displays retires an
      // entry unused for display_idle_ms).
      std::array<std::uint64_t, 2> display_used{};
    };
}
#if defined(__MINGW32__)
__CRT_UUID_DECL(sunshine_streamline::provider::state, 0xe82eab46, 0x4d62, 0x4e80, 0x9e, 0x02, 0x0e, 0x60, 0xb7, 0x05, 0x9f, 0x36)
#endif
namespace sunshine_streamline::provider {
  namespace {
    readiness_evidence before_decision(const state &data, std::uint64_t present) {
      readiness_evidence out;
      out.prior = data.cache.depth();
      out.prior_capture = data.cache.reference().capture_id;
      out.prior_output_ready = data.output.ready;
      out.capture.provider = data.source_metadata.provider;
      out.prior_provider = data.source_metadata.provider;
      out.prior_provider_known = data.last_valid.capture || (data.source_metadata.epoch && data.source_metadata.sequence);
      out.present = present;
      return out;
    }
    sunshine_depth::frame_depth::projection_t projection(const sunshine_scene_depth::frame &metadata) {
      return {metadata.projection.supplied, metadata.epoch, metadata.viewport,
        metadata.projection.depth_offset, metadata.projection.depth_scale,
        metadata.projection.raw_scale, metadata.projection.raw_bias, metadata.projection.encoding};
    }
    source_description describe_source(const depth_capture::packet &packet) {
      return {packet.metadata.resource.native, packet.capture_id, packet.metadata.sequence,
        packet.area.width, packet.area.height, packet.format,
        packet.metadata.resource.kind == sunshine_scene_depth::resource_kind::display_depth ? 48u : 0u,
        packet.resource_id};
    }
    void describe_copied_depth(sunshine_depth::frame_depth &out, const depth_capture::packet &packet,
        const sunshine_depth::frame_depth &captured, api::resource display_texture,
        api::resource_view display_view, std::uint64_t present) {
      out.provided = packet.metadata;
      out.projection = projection(packet.metadata);
      out.source_resource = {packet.metadata.resource.native};
      out.resource = display_texture;
      out.shader_resource = display_view;
      out.source_id = packet.metadata.source_id ? packet.metadata.source_id : packet.resource_id;
      out.layout_epoch = packet.metadata.epoch;
      out.frame_index = present;
      out.runtime_epoch = packet.metadata.epoch;
      out.width = packet.width; out.height = packet.height;
      out.x = packet.area.left; out.y = packet.area.top;
      out.active_width = packet.area.width; out.active_height = packet.area.height;
      out.command_queue = packet.queue;
      out.backup_id = captured.backup_id;
      out.ready = true;
      out.aligned_viewport_assumed = true;
      out.detected_orientation = out.orientation = !(packet.metadata.projection.supplied || packet.metadata.projection.direction_supplied) ?
        sunshine_depth::depth_orientation::automatic : packet.metadata.projection.reversed ?
          sunshine_depth::depth_orientation::reversed : sunshine_depth::depth_orientation::normal;
    }
    void bind_api(api::effect_runtime *runtime, state &data, api::resource_view view, bool ready);
    void destroy_display(api::effect_runtime *runtime, state &data);
    // Retired displays whose fence completed, released on a pool thread: a
    // texture's last Release frees its video memory in the driver and
    // ReShade's view release takes its device-wide lock, and either can wait
    // on video memory or the GPU (ACS 10-27b: 17 and 81 ms on the present
    // thread). Counted per runtime, so no job outlives its runtime or device:
    // destroy() waits for the count to reach zero first.
    struct display_release {
      api::device *device{};
      std::shared_ptr<state::release_jobs> owner;
      std::vector<state::retired_display> entries;
    };
    void release_displays(display_release &job, bool pooled);
    void CALLBACK release_task(PTP_CALLBACK_INSTANCE, void *context);
    void submit_release(std::unique_ptr<display_release> job);
    void wait_releases(state::release_jobs &jobs);
    void collect_displays(api::effect_runtime *runtime, state &data, std::uint64_t present);
    // The frame record's API rows (game3d_frame_record.h): this Present's
    // decision as begin() made it. Nothing reads them.
    void record_depth(api::effect_runtime *runtime, const sunshine_game3d::api_depth_decision &decision,
        std::uint64_t present) {
      if (auto *record = sunshine_game3d::frame_record_store::writable(runtime))
        sunshine_game3d::record_api_depth(*record, decision, present, GetTickCount64());
    }
    void track_fg_switch(state &data, const readiness_evidence &e, bool ready, const char *why);
    void trace_readiness(api::effect_runtime *runtime, state &data, const readiness_evidence &e,
        bool ready, const char *why, bool presenting = true);
    void log_render_size(render_size_state &size, std::uint64_t now);
    void track_render_size(state &data, bool copied, bool held, bool selected);
    void activate_display(state &data, const state::display_entry &entry);
    display_status prepare_display(api::effect_runtime *runtime, state &data, const depth_capture::packet &packet,
        std::uint64_t present);
  }
#ifdef SUNSHINE_SBS_TEST
  // Exercise the exact value-only publication used by the renderer and dump
  // without allocating resources or executing a GPU capture.
  void test_describe_copied_depth(sunshine_depth::frame_depth &out, const depth_capture::packet &packet,
      const sunshine_depth::frame_depth &captured, api::resource display_texture,
      api::resource_view display_view, std::uint64_t present) {
    describe_copied_depth(out, packet, captured, display_texture, display_view, present);
  }
#endif
  const char *name(display_status value);
  const char *name(sunshine_game3d::depth_output::display_action action);
  void format_readiness(char *text, std::size_t size, const readiness_evidence &e, const readiness_line &line);
  void format_observation_evidence(char *text, std::size_t size, bool ready, std::uint64_t runtime,
      std::uint64_t loss_tick, const observation_loss_episode &loss);
  void format_render_size(char *text, std::size_t size, const render_size_state &value);
  // The silent line names the FG scope's silence (freshness.h) in its text.
  static_assert(sunshine_freshness::fg_silent_ms == 1000);
  void format_frame_generation_scope(char *text, std::size_t size, bool silent, std::uint32_t viewport,
      std::uint64_t epoch, std::uint64_t changes);
  void format_fg_switch(char *text, std::size_t size, const fg_switch_line &line);
  void format_unassociated(char *text, std::size_t size, depth_capture::status status,
      const depth_capture::capture_diagnostic &capture, bool hold, std::uint64_t hold_ms);
  void format_selected(char *text, std::size_t size, const selected_line &line, const depth_capture::packet &frame,
      const depth_capture::consumer_diagnostic &consumer, const depth_capture::capture_diagnostic &capture);
  void format_held(char *text, std::size_t size, const char *path, depth_capture::status status, bool output_ready,
      display_status display, const depth_capture::capture_diagnostic &capture, const native_observer::counters &observer);
  void append_selection(char *text, std::size_t size, const depth_capture::capture_diagnostic &capture,
      const sunshine_game3d::depth_output::display_decision &update, std::uint64_t prior_depth_age,
      const sunshine_scene_depth::jitter_offset &jitter);
  void append_loss(char *text, std::size_t size, bool known, const loss_diagnostics::event &loss, std::uint64_t age_ms,
      std::uint64_t losses_since);
  namespace {
    sunshine_game3d::record_key recorded_source(const sunshine_scene_depth::frame &value) {
      return {sunshine_scene_depth::reporting_api(value.provider), value.viewport, value.source_id};
    }
  }
  sunshine_game3d::api_depth_decision describe_acquisition(const depth_capture::selection_policy &policy,
      const frame_generation_selection &scope, bool associated, bool handoff_hold,
      const depth_capture::capture_diagnostic &capture, const sunshine_scene_depth::frame *selected_metadata) {
    sunshine_game3d::api_depth_decision out;
    out.allowed = true;
    out.fg_required = policy.require_frame_generation; out.fg_silent = scope.silent;
    out.fg_excluded = scope.selection.exclude_unconfirmed_fg;
    out.fg_epoch = policy.epoch; out.fg_viewport = policy.viewport;
    out.selected = selected_metadata != nullptr; out.associated = associated; out.handoff_hold = handoff_hold;
    out.reported = capture.capture_id || capture.sequence;
    out.capture_reason = capture.selection != depth_capture::selection_reason::not_attempted ?
      depth_capture::name(capture.selection) : depth_capture::name(capture.result);
    if (selected_metadata) {
      out.source = recorded_source(*selected_metadata);
      out.epoch = selected_metadata->epoch; out.frame_generation_input = selected_metadata->frame_generation_input;
    } else if (out.reported) {
      out.source = {sunshine_scene_depth::reporting_api(capture.provider), capture.viewport, capture.source_id};
      out.epoch = capture.epoch; out.frame_generation_input = capture.frame_generation_input;
    }
    return out;
  }
  // shown: the output's metadata, the fresh copy's or the held one's.
  void describe_display(sunshine_game3d::api_depth_decision &decision, bool copied, bool output_ready,
      const char *cache_reason, const sunshine_scene_depth::frame &shown, std::uint64_t shown_capture) {
    if (!decision.selected && !decision.associated) {
      decision.display_reason = decision.handoff_hold ? "source_handoff" : "source_not_associated";
      return;
    }
    decision.copied = copied; decision.reused = !copied && output_ready;
    decision.display_reason = copied ? "fresh_copy" : output_ready ? "retained_copy" : cache_reason;
    decision.source_tick = copied || decision.reused ? shown.tick : 0;
    decision.source_capture = copied || decision.reused ? shown_capture : 0;
    if (decision.reused) {
      // The held copy names its own source.
      decision.source = recorded_source(shown);
      decision.epoch = shown.epoch; decision.frame_generation_input = shown.frame_generation_input;
    }
  }
  void initialize(api::effect_runtime *runtime);
  void destroy(api::effect_runtime *runtime);
  void reload(api::effect_runtime *runtime) {
    if (auto *data = runtime->get_private_data<state>()) {
      const auto evidence = before_decision(*data, data->output.frame_index);
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      depth_events::reset(runtime, depth_events::site_id::provider_runtime_reload);
#endif
      data->depth_ready.invalidate();
      data->open = data->owns_pass = data->have_latest = false;
      data->cache.invalidate("runtime_reload");
      data->presentations.reset();
      data->presentation_source = {};
      data->frame = {}; data->output = {};
      data->last_valid = {};
      data->source_metadata = {};
      data->source_policy = {};
      data->fg_scope_silent = false;
      data->bound_view = {};
      data->effects_view = {};
      data->associated_tick = 0;
      data->associated_width = data->associated_height = 0;
      data->handoff_hold = false;
      data->render_size = {};
      // The sampler independently retains any submitted slot until completion.
      data->pending_id = 0;
      trace_readiness(runtime, *data, evidence, false, "runtime_reload");
    }
  }
  void reload_effect_bindings(api::effect_runtime *runtime) {
    if (auto *data = runtime->get_private_data<state>()) {
      data->depth_ready.invalidate();
      if (data->owns_pass) {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        depth_events::binding(runtime, depth_events::site_id::provider_reload_bind, data->bound_view.handle != 0);
#endif
        runtime->update_texture_bindings("DEPTH", data->bound_view, data->bound_view);
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        depth_events::readiness(runtime, depth_events::site_id::provider_reload_ready, data->output.ready);
#endif
        set_depth_ready(runtime, data->output.ready);
      }
    }
  }
  void set_depth_ready(api::effect_runtime *runtime, bool ready) {
    if (auto *data = runtime->get_private_data<state>()) {
      data->depth_ready.set(ready, [runtime](auto &uniforms) {
        runtime->enumerate_uniform_variables(nullptr, [&uniforms](api::effect_runtime *owner, api::effect_uniform_variable variable) {
          char source[32]{};
          if (owner->get_annotation_string_from_uniform_variable(variable, "source", source) &&
              std::strcmp(source, "bufready_depth") == 0)
            uniforms.push_back(variable);
        });
      }, [runtime](api::effect_uniform_variable variable, bool value) {
        runtime->set_uniform_value_bool(variable, value);
      });
    }
  }
  bool complete(api::effect_runtime *runtime, const sunshine_depth::sample_result &value) {
    auto *data = runtime->get_private_data<state>();
    if (!data || !data->pending_id || value.capture_id != data->pending_id || value.token != 0)
      return false;
    const bool matches = value.valid && value.moments.supplied && value.source.handle == data->pending_texture &&
      value.width == data->pending_grid.x && value.height == data->pending_grid.y &&
      value.values.size() == std::size_t(data->pending_grid.x) * data->pending_grid.y &&
      value.values.size() <= sunshine_depth_statistics::maximum_tiles &&
      value.source_width == data->pending_metadata.resource.width && value.source_height == data->pending_metadata.resource.height &&
      value.viewport_x == data->pending_metadata.resource.area.left && value.viewport_y == data->pending_metadata.resource.area.top &&
      value.viewport_width == data->pending_metadata.resource.area.width && value.viewport_height == data->pending_metadata.resource.area.height;
    if (matches) {
      data->latest.metadata = data->pending_metadata;
      data->latest.projection = projection(data->pending_metadata);
      data->latest.id = data->pending_id;
      data->latest.tick = data->pending_metadata.tick;
      data->latest.width = value.width;
      data->latest.height = value.height;
      data->latest.range_valid = value.range_valid;
      data->latest.range_min = value.range_min;
      data->latest.range_max = value.range_max;
      data->latest.moments = value.moments;
      data->have_latest = true;
    }
    data->pending_id = 0;
    data->pending_metadata = {};
    data->pending_grid = {};
    return true;
  }
  bool begin(api::effect_runtime *runtime, api::command_list *commands, std::uint64_t present, bool allowed) {
    auto *data = runtime->get_private_data<state>();
    if (!data) return false;
    auto evidence = before_decision(*data, present);
    const bool previously_owned = data->owns_pass;
    const bool previously_held = data->handoff_hold;
    data->open = data->owns_pass = data->handoff_hold = false;
    data->frame = {}; data->output = {};
    if (!allowed || runtime->get_device()->get_api() != api::device_api::d3d12) {
      data->source_policy = {};
      data->fg_scope_silent = false;
      data->cache.invalidate(!allowed ? "provider_disallowed" : "unsupported_API");
      data->presentations.reset();
      data->presentation_source = {};
      data->bound_view = {};
      data->associated_tick = 0;
      trace_readiness(runtime, *data, evidence, false, data->cache.reason());
      sunshine_game3d::api_depth_decision recorded;
      recorded.refusal = data->cache.reason();
      record_depth(runtime, recorded, present);
      return false;
    }
    {
      const sunshine_game3d::slow_step step("depth display retirement");
      collect_displays(runtime, *data, present);
    }
    {
      // The effects list, ReShade's immediate list and their queue: once per
      // Present, one interface check each and one capture-lock round trip.
      const sunshine_game3d::slow_step step("depth present observation");
      auto *queue = runtime->get_command_queue();
      depth_capture::observe_present(commands->get_native(), queue->get_native(),
        queue->get_immediate_command_list()->get_native());
    }
    {
      // Capture admissions by ReShade's list lifecycle, which decides coverage.
      const sunshine_game3d::slow_step step("depth list coverage report");
      char text[256];
      if (depth_capture::list_coverage_report(GetTickCount64(), text, sizeof(text)))
        sunshine_log::message(reshade::log::level::info, text);
      // A safety bound of the capture owner that refused a capture, once each.
      char bound[384];
      if (depth_capture::capacity_report(bound, sizeof(bound)))
        sunshine_log::message(reshade::log::level::warning, bound);
    }
    frame_generation_snapshot fg_source;
    frame_generation_query_status fg_status;
    {
      const sunshine_game3d::slow_step step("frame generation query");
      query_frame_generation(UINT32_MAX, fg_source, &fg_status);
    }
    const auto policy = data->source_policy.update(fg_status, fg_source);
    // A silent FG scope selects as without FG (select_frame_generation_scope).
    // The policy's own confirmed FG scope is left unchanged, and it still
    // holds the queue in mono when no API source is available (associated).
    const auto scope = select_frame_generation_scope(policy);
    const bool fg_silent = scope.silent;
    // The Presents DLSS-G adds per real frame, from the confirmed policy, so
    // a silent scope keeps them (frame_presenting grants them only to a
    // frame that made a frame-generation call); 1 while FG is on with no
    // count, 0 without FG.
    auto selection = scope.selection;
    selection.generated_presents = policy.require_frame_generation ? std::max(1u, data->source_policy.generated_frames()) : 0u;
    // Logged when a required scope turns silent, and when it is live again
    // (FG Off ends a silent scope without a line of its own). A scope turns
    // at each frame FG supplies or misses, so the line comes at most every
    // 5 s, naming the newest state and the turns counted so far.
    if (fg_silent != std::exchange(data->fg_scope_silent, fg_silent) && (fg_silent || policy.require_frame_generation))
      ++data->fg_scope_changes;
    if (!policy.require_frame_generation) data->fg_scope_logged_changes = data->fg_scope_changes;
    if (data->fg_scope_changes != data->fg_scope_logged_changes) {
      const auto now = GetTickCount64();
      if (!data->fg_scope_line_tick || now - data->fg_scope_line_tick >= 5000) {
        char text[384]{};
        format_frame_generation_scope(text, sizeof(text), fg_silent, policy.viewport, policy.epoch, data->fg_scope_changes);
        sunshine_log::message(reshade::log::level::info, text);
        data->fg_scope_logged_changes = data->fg_scope_changes;
        data->fg_scope_line_tick = now;
      }
    }
    evidence.fg = fg_source;
    evidence.fg_query = fg_status;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    data->test_fg_decision = (fg_status == frame_generation_query_status::observed ? 1u : 0u) |
      (policy.require_frame_generation ? 2u : 0u) | (policy.exclude_unconfirmed_fg ? 4u : 0u) |
      (selection.generated_presents << 8);
#endif
    evidence.selection = selection;
    evidence.fg_silent = fg_silent;
    if (selection.exclude_unconfirmed_fg && data->source_metadata.frame_generation_input) {
      data->source_metadata = {};
      data->cache.invalidate("FG_scope_unconfirmed");
      data->last_valid = {};
      data->have_latest = false;
      data->pending_id = 0;
      data->pending_metadata = {};
      data->pending_grid = {};
      data->presentations.reset();
      data->presentation_source = {};
    }
    if (selection.require_frame_generation) {
      const auto &previous = data->source_metadata;
      if (previous.provider != sunshine_scene_depth::provider_kind::streamline || !previous.frame_generation_input ||
          previous.epoch != selection.epoch || previous.viewport != selection.viewport) {
        // FG selects a logical SL source before its first pixels are ready.
        // Never carry NGX's encoding or sampled center into it. The display
        // decides the old copy (decide_display): the viewport's ordinary copy
        // of the scope's epoch, its lineage, is held until the scope's first
        // readable FG copy, and any other is dropped (FG_scope_changed).
        data->source_metadata = {};
        data->source_metadata.provider = sunshine_scene_depth::provider_kind::streamline;
        data->source_metadata.epoch = selection.epoch;
        data->source_metadata.viewport = selection.viewport;
        data->source_metadata.frame_generation_input = true;
        data->last_valid = {};
        data->have_latest = false;
        data->pending_id = 0;
      }
    }
    depth_capture::capture_diagnostic capture_info;
    depth_capture::acquisition_decision acquisition;
    {
      const sunshine_game3d::slow_step step("depth capture acquisition");
      depth_capture::acquire(runtime->get_command_queue()->get_native(), present, data->frame, acquisition, &capture_info, selection);
    }
    // O1 (fail-safe): a copy read in flight on a fence that a signal first
    // submitted after that read has since poisoned (late_conflict) may hold
    // pixels its producer had not written yet; it is not held any further,
    // from the first Present after the conflict was seen. Decided before
    // anything reads the retained copy (the FG hold, the association), so a
    // fresh candidate of this Present stands in at once (until 10-27 a
    // silent scope still held the conflicted FG copy's place, and the
    // Present was mono), and its reason survives the display decision.
    bool late_conflict = false;
    if (const auto held = data->cache.reference().capture_id; held && depth_capture::read_conflicted(held)) {
      data->cache.invalidate("late_conflict");
      late_conflict = true;
    }
    // FG keeps its viewport while it can supply (frame_generation_holds): a
    // silent scope's ordinary candidate, fresh or pending, stands in only
    // once the scope's own newest copy can no longer be held, so a frame
    // without an FG copy holds as any missing frame of a source does. Every
    // other handover between the viewport's FG and ordinary sources holds the
    // displayed copy until the newcomer's first readable frame
    // (decide_display's lineage).
    // So does a live scope's ordinary stand-in for its pending input
    // (lineage_snapshot): it never replaces the scope's own held copy.
    const char *fg_hold = nullptr;
    if ((fg_silent || (selection.require_frame_generation && acquisition.source_selected)) &&
        !(acquisition.source_selected && data->frame.metadata.frame_generation_input) &&
        depth_capture::frame_generation_holds(data->cache.reference(), policy, GetTickCount64()))
      fg_hold = "FG_copy_held";
    const bool available = acquisition.source_selected && !fg_hold;
    evidence.capture = capture_info;
    evidence.available = available;
    evidence.snapshot_ready = data->frame.pixel_ready;
    if (available) {
      data->frame.metadata.resource.width = data->frame.width;
      data->frame.metadata.resource.height = data->frame.height;
      data->frame.metadata.resource.area = data->frame.area;
    }
    const auto status = capture_info.result;
    // Session ownership and current pixel readiness are different facts. Once
    // a supported API has identified a valid source on this queue, a missing frame stays
    // mono; it cannot silently select another encoding and recalibrate its gain.
    // Enabled FG holds the queue from its policy, not from this Present's
    // selection: a silent scope releases only the selection restriction, so
    // without an API source the game stays mono instead of handing real and
    // generated presentations to the generic selector. FG Off releases the
    // queue (retire_source) while the display may hold the scope's newest
    // copy: the queue stays the API's while that copy is younger than the
    // source age (frame_generation_retained), so it is held as any source's
    // missing frame is.
    const bool associated = policy.require_frame_generation || depth_capture::provider_active(runtime->get_command_queue()->get_native()) ||
      depth_capture::frame_generation_retained(data->cache.reference(), policy, GetTickCount64());
    std::uint32_t frame_width{}, frame_height{};
    runtime->get_screenshot_width_and_height(&frame_width, &frame_height);
    const auto handoff_tick = GetTickCount64();
    if (available || associated) {
      data->associated_tick = handoff_tick;
      data->associated_width = frame_width; data->associated_height = frame_height;
    }
    // Handoff hysteresis: a provider flip (Streamline FG scope released while
    // NGX, or the other way round, is evaluating and about to establish on
    // this queue) keeps API ownership in mono instead of enabling the generic
    // selector, its challenger and two effect rebinds for a few Presents.
    // Bounded by handoff_ms after the last associated Present, the same
    // runtime (reload resets it) and swapchain size, and live evidence of an
    // API evaluation. An explicit release with no other evaluation falls back
    // at once, as before.
    const bool hold = !available && !associated && [&] {
      handoff_evidence e;
      e.previously_owned = previously_owned || previously_held;
      e.now_ms = handoff_tick; e.associated_ms = data->associated_tick;
      e.width = frame_width; e.height = frame_height;
      e.associated_width = data->associated_width; e.associated_height = data->associated_height;
      // Queried last: the capture lock only when everything else holds.
      e.evaluation_live = handoff_hold({e.previously_owned, true, e.now_ms, e.associated_ms, e.width, e.height,
        e.associated_width, e.associated_height}, handoff_ms) && depth_capture::evaluation_live(handoff_tick, handoff_ms);
      return handoff_hold(e, handoff_ms);
    }();
    auto recorded = describe_acquisition(policy, scope, associated, hold, capture_info,
      available ? &data->frame.metadata : nullptr);
    if (!available && !associated) {
      data->cache.invalidate(hold ? "source_handoff" : "source_not_associated");
      data->presentations.reset();
      data->presentation_source = {};
      if (!hold) data->associated_tick = 0;
      if (data->logging.due(GetTickCount64(), status != data->last_status, previously_owned)) {
        // The initial failure needs the same handoff evidence as an established
        // provider. Otherwise a valid captured depth rejected on another queue
        // looks indistinguishable from an absent API or an unknown queue.
        char text[768]{};
        format_unassociated(text, sizeof(text), status, capture_info, hold, handoff_ms);
        sunshine_log::message(reshade::log::level::info, text);
        data->last_status = status;
      }
      if (hold) {
        data->owns_pass = data->open = data->handoff_hold = true;
        data->output.provided = data->source_metadata;
        data->output.projection = projection(data->source_metadata);
        // The stable descriptor stays bound; readiness is withdrawn.
        bind_api(runtime, *data, data->display_view, false);
        trace_readiness(runtime, *data, evidence, false, "source_handoff");
        describe_display(recorded, false, false, nullptr, {});
        record_depth(runtime, recorded, present);
        return true;
      }
      data->bound_view = {};
      trace_readiness(runtime, *data, evidence, false, "source_not_associated");
      describe_display(recorded, false, false, nullptr, {});
      record_depth(runtime, recorded, present);
      return false;
    }
    auto kind = available ? data->frame.metadata.provider : data->source_metadata.provider;
    std::uint64_t established_source{};
    // Name the provider of the reported capture: selection can report the
    // other provider's pending capture in front of the owner. Only an empty
    // diagnostic falls back to the established owner.
    if (!available && (capture_info.capture_id || capture_info.sequence)) kind = capture_info.provider;
    else if (!available && !selection.require_frame_generation)
      depth_capture::provider_identity(runtime->get_command_queue()->get_native(), kind, established_source);
    if (!available && !recorded.reported) {
      // Nothing reported this Present: the source H holds the queue for.
      recorded.source = {sunshine_scene_depth::reporting_api(kind), data->source_metadata.viewport,
        selection.require_frame_generation ? 0 : established_source};
      recorded.epoch = data->source_metadata.epoch;
    }
    const char *path_name = kind == sunshine_scene_depth::provider_kind::ngx ? "NGX" : "Streamline";
    data->owns_pass = data->open = true;
    if (selection.require_frame_generation && !available)
      data->source_metadata.source_id = acquisition.source_id;
    if (available) data->source_metadata = data->frame.metadata;
    // Keep the encoding identity through gaps. Only the explicit bounded hold
    // below can authorize the already-owned display pixels during a pending copy.
    data->output.provided = data->source_metadata;
    data->output.projection = projection(data->frame.metadata);
    if (!available) {
      data->output.projection = projection(data->source_metadata);
    }
    display_status display{display_status::not_attempted};
    depth_capture::consumer_diagnostic consumer;
    sunshine_depth::frame_depth captured;
    bool copied = false;
    const auto reuse_tick = GetTickCount64();
    const auto &cached = data->cache.depth();
    const auto prior_depth_age = cached.ready && reuse_tick >= cached.provided.tick ?
      reuse_tick - cached.provided.tick : UINT64_MAX;
    auto update = fg_hold ? sunshine_game3d::depth_output::display_decision{sunshine_game3d::depth_output::display_action::hold, fg_hold} :
      sunshine_game3d::depth_output::decide_display(acquisition, data->frame, data->cache.reference(), selection, reuse_tick);
    evidence.action = update.action;
    evidence.source_reason = update.reason;
    if (available) display = display_status::waiting_capture;
#include "depth_display_copy.inl"
    if (update.action == sunshine_game3d::depth_output::display_action::copy_fresh && !copied) {
      // A failed fresh copy recorded nothing into the display storage (an
      // observation loss refuses the consumer's recording, for one): its
      // frame is a failed newer frame, so the retained copy is held as for a
      // pending one (decide_failed_copy).
      update = sunshine_game3d::depth_output::decide_failed_copy(acquisition, data->frame, data->cache.reference(), selection, reuse_tick);
      evidence.action = update.action;
      evidence.source_reason = update.reason;
    }
    if (copied) {
      const auto &packet = data->frame;
      depth_capture::complete_frame(packet, present);
      auto &out = data->output;
      describe_copied_depth(out, packet, captured, data->display_texture, data->display_view, present);
      data->last_valid = describe_source(packet);
      data->source_metadata = packet.metadata;
      // Private display storage is already ordered on this runtime queue.
      // Retaining it never extends a game texture's lifetime or adds a GPU wait.
      std::uint32_t color_width{}, color_height{};
      runtime->get_screenshot_width_and_height(&color_width, &color_height);
      data->cache.commit(out, packet.capture_id, packet.format, packet.srv_format, color_width, color_height, packet.present);
    } else if (update.action == sunshine_game3d::depth_output::display_action::hold) {
      sunshine_game3d::depth_output::display_cache_context context;
      std::uint32_t color_width{}, color_height{};
      runtime->get_screenshot_width_and_height(&color_width, &color_height);
      context.color_width = color_width; context.color_height = color_height;
      if (data->cache.matches_color(color_width, color_height)) {
        auto *queue = runtime->get_command_queue();
        context.immediate_commands = commands == queue->get_immediate_command_list();
        if (context.immediate_commands) context.queue = queue->get_native();
      }
      context.texture = data->display_texture.handle; context.view = data->display_view.handle;
      context.now = GetTickCount64();
      sunshine_depth::frame_depth reused;
      if (data->cache.reuse(update, context, present, reused)) {
        data->output = std::move(reused);
        display = display_status::reused_depth;
      }
    } else if (!late_conflict) {
      // Once invalidated, only a successful new copy can establish another
      // cache entry. An O1 drop keeps its own reason.
      data->cache.invalidate(update.reason);
    }
    track_render_size(*data, copied, display == display_status::reused_depth, available);
    // The display cache's copy is the one shown: committed by this fresh
    // copy, or the retained one reused.
    describe_display(recorded, copied, data->output.ready, data->cache.reason(), data->output.provided,
      data->output.ready ? data->cache.reference().capture_id : 0);
    if (copied) {
      recorded.read = static_cast<sunshine_game3d::read_order>(consumer.read_order);
      recorded.read_queue = consumer.producer_queue;
      recorded.read_ticket = consumer.producer_ticket;
    }
    const char *readiness_reason = recorded.display_reason;
    record_depth(runtime, recorded, present);
    evidence.cache_reason = data->cache.reason();
    evidence.copied = copied;
    evidence.display = display;
    evidence.consumer = consumer;
    if (!evidence.tick) evidence.tick = reuse_tick;
    trace_readiness(runtime, *data, evidence, data->output.ready, readiness_reason);
    const auto &stats_source = data->output.provided;
    // FG stays active while a silent scope's viewport supplies ordinary depth.
    data->output.frame_generation_active = policy.require_frame_generation && policy.epoch &&
      stats_source.provider == sunshine_scene_depth::provider_kind::streamline &&
      (stats_source.frame_generation_input || fg_silent) &&
      policy.epoch == stats_source.epoch && policy.viewport == stats_source.viewport;
    const std::array<std::uint64_t, 5> identity{static_cast<std::uint64_t>(stats_source.provider),
      stats_source.epoch, stats_source.source_id, stats_source.viewport,
      data->output.frame_generation_active ? data->source_policy.generated_frames() : 0u};
    if (identity != data->presentation_source) {
      data->presentations.reset();
      data->presentation_source = identity;
    }
    data->presentations.observe(present, reuse_tick, !data->output.ready ? sunshine_depth_stats::presentation_kind::unavailable :
      data->output.reused_depth ? sunshine_depth_stats::presentation_kind::reused : sunshine_depth_stats::presentation_kind::fresh);
    // Fresh and deliberately reused depth share the existing stable descriptor.
    bind_api(runtime, *data, data->display_view, data->output.ready);
#include "provider_report_present.inl"
    return true;
  }
  bool finish(api::effect_runtime *runtime, api::command_list *commands) {
    auto *data = runtime->get_private_data<state>();
    if (!data || !data->owns_pass) return false;
    data->open = false;
    const auto now = GetTickCount64();
    if (!data->output.ready || data->output.reused_depth) return true;
    if (now >= data->next_sample && !sunshine_depth::busy()) {
      sunshine_depth::sample_request request;
      request.collect_range = true;
      // Decode on the GPU with this capture's coefficients. Never retrofit a
      // later camera or inferred orientation when asynchronous results arrive.
      const auto &basis = data->frame.metadata.projection;
      const auto coefficients = basis.supplied || basis.encoding == sunshine_scene_depth::depth_encoding::linear_distance ? sunshine_projection_depth::make(
        basis.encoding, basis.depth_offset, basis.depth_scale, basis.raw_scale, basis.raw_bias) :
        sunshine_projection_depth::coefficients{};
      if (coefficients.valid()) {
        request.collect_moments = true;
        request.moments_A = coefficients.shader_A;
        request.moments_inverseB = coefficients.inverseB;
        request.moments_encoding = coefficients.encoding;
      } else if (basis.encoding == sunshine_scene_depth::depth_encoding::device &&
          !basis.supplied && basis.direction_supplied) {
        request.collect_moments = true;
        request.moments_A = basis.reversed ? 0.f : 1.f;
        request.moments_inverseB = basis.reversed ? 1.f : -1.f;
      }
      request.capture_id = data->frame.capture_id;
      // Both transports publish to this stable display allocation. The sampler
      // retains its COM source and submits after these effects on the same
      // runtime queue; no pooled preservation backup escapes its owner.
      request.source = data->display_texture;
      request.view_format = data->display_view_format;
      request.before = api::resource_usage::shader_resource;
      request.source_width = data->frame.width;
      request.source_height = data->frame.height;
      request.x = data->frame.area.left; request.y = data->frame.area.top;
      request.width = data->frame.area.width; request.height = data->frame.area.height;
      if (sunshine_depth::submit(runtime, commands, request)) {
        data->pending_id = request.capture_id;
        data->pending_texture = request.source.handle;
        data->pending_metadata = data->frame.metadata;
        data->pending_metadata.resource.width = data->frame.width;
        data->pending_metadata.resource.height = data->frame.height;
        data->pending_metadata.resource.area = data->frame.area;
        data->pending_grid = sunshine_depth_statistics::tile_grid(request.width, request.height);
        data->next_sample = now + 125;
      }
    }
    return true;
  }
  bool current(api::effect_runtime *runtime, sunshine_depth::frame_depth &out) {
    const auto *data = runtime->get_private_data<state>();
    if (!data || !data->owns_pass || !data->open) return false;
    out = data->output;
    return true;
  }
  bool latest_center(api::effect_runtime *runtime, center_sample &out) {
    const auto *data = runtime->get_private_data<state>();
    if (!data || !data->have_latest) return false;
    out = data->latest;
    return true;
  }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  // Read-only failure evidence from the real sampler; never injects readiness
  // or calibration state, and is absent from the production add-on.
  extern "C" __declspec(dllexport) bool SunshineStreamlineTestCenter(api::effect_runtime *runtime, center_sample *out) {
    return out && latest_center(runtime, *out);
  }
  // The FG decision of the runtime's last depth frame (provider begin): bit 0
  // the FG read observed the mode, bit 1 the policy requires FG, bit 2 it
  // excludes unconfirmed FG input, bits 8 and up the generated Presents.
  unsigned test_frame_generation_decision(api::effect_runtime *runtime) {
    const auto *data = runtime->get_private_data<state>();
    return data ? data->test_fg_decision : 0u;
  }
  // The active display texture (its native resource), read only.
  extern "C" __declspec(dllexport) bool SunshineStreamlineTestDisplay(api::effect_runtime *runtime, std::uint64_t *texture) {
    const auto *data = runtime->get_private_data<state>();
    if (!data || !texture) return false;
    *texture = data->display_texture.handle;
    return true;
  }
#endif
  bool selected(api::effect_runtime *runtime) {
    const auto *data = runtime->get_private_data<state>();
    return data && data->owns_pass;
  }
  void invalidate_reused_depth(api::effect_runtime *runtime) {
    if (auto *data = runtime->get_private_data<state>()) {
      const auto evidence = before_decision(*data, data->output.frame_index);
      data->cache.invalidate("explicit_reuse_invalidation");
      data->presentations.reset();
      data->presentation_source = {};
      if (data->output.reused_depth) {
        data->output = {};
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        depth_events::readiness(runtime, depth_events::site_id::provider_reuse_ready, false);
#endif
        set_depth_ready(runtime, false);
      }
      trace_readiness(runtime, *data, evidence, data->output.ready, "explicit_reuse_invalidation", false);
    }
  }
  source_status describe(api::effect_runtime *runtime, std::uint64_t present) {
    source_status out;
    const auto *data = runtime->get_private_data<state>();
    if (!data) return out;
    out.selected = data->owns_pass;
    out.last_valid = data->last_valid;
    out.presentations = data->presentations.snapshot(GetTickCount64());
    out.provider = data->source_metadata.provider;
    std::uint64_t source{};
    if (!data->owns_pass)
      depth_capture::provider_identity(runtime->get_command_queue()->get_native(), out.provider, source);
    out.ready = out.selected && data->output.ready && data->output.frame_index == present;
    out.reused_depth = out.ready && data->output.reused_depth;
    out.frame_generation_input = out.ready && data->output.provided.frame_generation_input;
    if (out.ready) out.current = data->output.reused_depth ? data->last_valid : describe_source(data->frame);
    return out;
  }

#include "depth_display.inl"
#include "provider_report.inl"
}
