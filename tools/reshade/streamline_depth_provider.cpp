// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_depth_provider.h"
#include "streamline_depth_capture.h"
#include "streamline_native_observer.h"
#include "depth_content_sampler.h"
#include "depth_ready_uniform_cache.h"
#include "diagnostic_log_gate.h"
#include <reshade.hpp>
#include <algorithm>
#include <cstring>
#include <cstdio>

namespace sunshine_streamline::provider {
  namespace api = reshade::api;
  enum class display_status { not_attempted, ready, reused_depth, texture_creation_failed, view_creation_failed, waiting_capture };
    struct __declspec(uuid("e82eab46-4d62-4e80-9e02-0e60b7059f36")) state {
      depth_capture::packet frame;
      sunshine_scene_depth::frame last_metadata;
      sunshine_scene_depth::frame last_successful_metadata;
      sunshine_depth::frame_depth output;
      sunshine_depth::frame_depth last_real_depth;
      std::uint32_t last_color_width{}, last_color_height{};
      sunshine_depth_stats::presentation_window presentations;
      std::array<std::uint64_t, 5> presentation_source{};
      source_description last_valid;
      center_sample latest;
      depth_capture::input pending_metadata;
      std::uint64_t pending_id{}, pending_texture{}, next_sample{};
      bool owns_pass{}, open{}, have_latest{};
      bool shared_preservation{};
      frame_generation_policy source_policy;
      api::resource display_texture{};
      api::resource_view display_view{}, bound_view{};
      sunshine_depth::ready_uniform_cache<api::effect_uniform_variable> depth_ready;
      std::uint32_t display_width{}, display_height{}, display_format{};
      sunshine_diagnostics::log_gate logging;
      depth_capture::status last_status{depth_capture::status::inactive};
      depth_capture::capture_failure last_capture_failure{depth_capture::capture_failure::none};
      display_status last_display{display_status::not_attempted};
      depth_capture::consumer_status last_consumer{depth_capture::consumer_status::not_attempted};
      depth_capture::recording_loss last_loss{depth_capture::recording_loss::none};
      bool last_output_ready{};
    };
}
#if defined(__MINGW32__)
__CRT_UUID_DECL(sunshine_streamline::provider::state, 0xe82eab46, 0x4d62, 0x4e80, 0x9e, 0x02, 0x0e, 0x60, 0xb7, 0x05, 0x9f, 0x36)
#endif
namespace sunshine_streamline::provider {
  namespace {
    sunshine_depth::frame_depth::projection_t projection(const sunshine_scene_depth::frame &metadata) {
      return {metadata.projection.supplied, metadata.epoch, metadata.viewport,
        metadata.projection.depth_offset, metadata.projection.depth_scale,
        metadata.projection.raw_scale, metadata.projection.raw_bias};
    }
    source_description describe_source(const depth_capture::packet &packet) {
      return {packet.metadata.resource.native, packet.capture_id, packet.metadata.sequence,
        packet.area.width, packet.area.height, packet.format,
        packet.metadata.resource.kind == sunshine_scene_depth::resource_kind::display_depth ? 48u : 0u,
        packet.resource_id};
    }
    bool same_successful_fg_source(const state &data, const depth_capture::capture_diagnostic &current) {
      const auto &previous = data.last_successful_metadata;
      // Resource pointers may rotate, but the validated logical FG source must
      // not. A nomination alone must never overwrite this successful identity.
      return data.last_valid.capture && previous.frame_generation_input && previous.epoch && previous.source_id &&
        previous.provider == sunshine_scene_depth::provider_kind::streamline &&
        current.provider == previous.provider && current.epoch == previous.epoch &&
        current.source_id == previous.source_id && current.viewport == previous.viewport &&
        current.sequence >= previous.sequence && (!current.finished || current.success) && !current.invalid &&
        current.failure == depth_capture::capture_failure::none;
    }
    void bind(api::effect_runtime *runtime, state &data, api::resource_view view, bool ready) {
      // ReShade 6.8 waits for the whole queue on every update, even if the
      // handle is unchanged. Keep the descriptor stable while pixels rotate.
      if (view != data.bound_view) {
        runtime->update_texture_bindings("DEPTH", view, view);
        data.bound_view = view;
      }
      set_depth_ready(runtime, ready);
    }
    void destroy_display(api::effect_runtime *runtime, state &data) {
      data.last_real_depth = {};
      if (data.display_view.handle) runtime->get_device()->destroy_resource_view(data.display_view);
      if (data.display_texture.handle) runtime->get_device()->destroy_resource(data.display_texture);
      data.display_view = {}; data.display_texture = {};
    }
    const char *name(display_status value) {
      switch (value) {
        case display_status::not_attempted: return "not_attempted";
        case display_status::ready: return "ready";
        case display_status::reused_depth: return "reused_depth";
        case display_status::texture_creation_failed: return "texture_creation_failed";
        case display_status::view_creation_failed: return "view_creation_failed";
        case display_status::waiting_capture: return "waiting_capture";
      }
      return "unknown";
    }
    display_status prepare_display(api::effect_runtime *runtime, state &data, const depth_capture::packet &packet) {
      if (data.display_texture.handle && data.display_width == packet.width && data.display_height == packet.height &&
          data.display_format == packet.format) return display_status::ready;
      // Resolution/format changes are rare lifecycle events. Ordinary rotation
      // never calls wait_idle, allocates a texture or changes descriptors.
      if (data.display_texture.handle) {
        runtime->get_command_queue()->wait_idle();
        bind(runtime, data, {}, false);
        destroy_display(runtime, data);
      }
      api::resource_desc desc(packet.width, packet.height, 1, 1, static_cast<api::format>(packet.format), 1,
        api::memory_heap::default_, api::resource_usage::shader_resource | api::resource_usage::copy_dest);
      if (!runtime->get_device()->create_resource(desc, nullptr, api::resource_usage::shader_resource, &data.display_texture)) return display_status::texture_creation_failed;
      api::resource_view_desc view(static_cast<api::format>(packet.srv_format));
      view.type = api::resource_view_type::texture_2d;
      if (!runtime->get_device()->create_resource_view(data.display_texture, api::resource_usage::shader_resource, view, &data.display_view)) {
        destroy_display(runtime, data);
        return display_status::view_creation_failed;
      }
      data.display_width = packet.width; data.display_height = packet.height; data.display_format = packet.format;
      return display_status::ready;
    }
  }
  void initialize(api::effect_runtime *runtime) { runtime->create_private_data<state>(); }
  void destroy(api::effect_runtime *runtime) {
    if (auto *data = runtime->get_private_data<state>()) {
      if (data->display_texture.handle) runtime->get_command_queue()->wait_idle();
      destroy_display(runtime, *data);
    }
    runtime->destroy_private_data<state>();
  }
  void reload(api::effect_runtime *runtime) {
    if (auto *data = runtime->get_private_data<state>()) {
      data->depth_ready.invalidate();
      data->open = data->owns_pass = data->have_latest = data->shared_preservation = false;
      data->last_real_depth = {};
      data->presentations.reset();
      data->presentation_source = {};
      data->frame = {}; data->output = {};
      data->last_valid = {};
      data->last_metadata = {};
      data->last_successful_metadata = {};
      data->source_policy = {};
      data->bound_view = {};
      // The sampler independently retains any submitted slot until completion.
      data->pending_id = 0;
    }
  }
  void reload_effect_bindings(api::effect_runtime *runtime) {
    if (auto *data = runtime->get_private_data<state>()) {
      data->depth_ready.invalidate();
      if (data->owns_pass) {
        runtime->update_texture_bindings("DEPTH", data->bound_view, data->bound_view);
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
    const bool matches = value.valid && value.source.handle == data->pending_texture &&
      value.width == 32 && value.height == 18 && value.values.size() == data->latest.raw.size() &&
      value.source_width == data->pending_metadata.resource.width && value.source_height == data->pending_metadata.resource.height &&
      value.viewport_x == data->pending_metadata.resource.area.left && value.viewport_y == data->pending_metadata.resource.area.top &&
      value.viewport_width == data->pending_metadata.resource.area.width && value.viewport_height == data->pending_metadata.resource.area.height;
    if (matches) {
      data->latest.metadata = data->pending_metadata;
      data->latest.projection = projection(data->pending_metadata);
      data->latest.id = data->pending_id;
      data->latest.tick = data->pending_metadata.tick;
      std::copy(value.values.begin(), value.values.end(), data->latest.raw.begin());
      data->have_latest = true;
    }
    data->pending_id = 0;
    data->pending_metadata = {};
    return true;
  }
  bool begin(api::effect_runtime *runtime, api::command_list *commands, std::uint64_t present, bool allowed) {
    auto *data = runtime->get_private_data<state>();
    if (!data) return false;
    const bool previously_owned = data->owns_pass;
    data->open = data->owns_pass = false;
    data->frame = {}; data->output = {};
    if (!allowed || runtime->get_device()->get_api() != api::device_api::d3d12) {
      data->source_policy = {};
      data->last_real_depth = {};
      data->presentations.reset();
      data->presentation_source = {};
      data->bound_view = {};
      return false;
    }
    depth_capture::observe_command(commands->get_native());
    depth_capture::observe_queue(runtime->get_command_queue()->get_native());
    frame_generation_snapshot fg_source;
    frame_generation_query_status fg_status;
    query_frame_generation(UINT32_MAX, fg_source, &fg_status);
    const auto selection = data->source_policy.update(fg_status, fg_source);
    if (selection.require_frame_generation) {
      const auto &previous = data->last_metadata;
      if (previous.provider != sunshine_scene_depth::provider_kind::streamline || !previous.frame_generation_input ||
          previous.epoch != selection.epoch || previous.viewport != selection.viewport) {
        // FG selects a logical SL source before its first pixels are ready.
        // Never carry NGX's encoding, sampled center or old display into it.
        data->last_metadata = {};
        data->last_metadata.provider = sunshine_scene_depth::provider_kind::streamline;
        data->last_metadata.epoch = selection.epoch;
        data->last_metadata.viewport = selection.viewport;
        data->last_metadata.frame_generation_input = true;
        data->last_successful_metadata = {};
        data->last_real_depth = {};
        data->last_valid = {};
        data->have_latest = data->shared_preservation = false;
        data->pending_id = 0;
      }
    }
    depth_capture::capture_diagnostic capture_info;
    const bool available = depth_capture::acquire(runtime->get_command_queue()->get_native(), present, data->frame, &capture_info, selection);
    if (available) {
      data->frame.metadata.resource.width = data->frame.width;
      data->frame.metadata.resource.height = data->frame.height;
      data->frame.metadata.resource.area = data->frame.area;
    }
    const auto status = capture_info.result;
    // Session ownership and current pixel readiness are different facts. Once
    // a supported API has identified a valid source on this queue, a missing frame stays
    // mono; it cannot silently select another encoding and recalibrate its gain.
    const bool associated = selection.require_frame_generation || depth_capture::provider_active(runtime->get_command_queue()->get_native());
    if (!available && !associated) {
      data->last_real_depth = {};
      data->presentations.reset();
      data->presentation_source = {};
      if (data->logging.due(GetTickCount64(), status != data->last_status, previously_owned)) {
        // The initial failure needs the same handoff evidence as an established
        // provider. Otherwise a valid captured depth rejected on another queue
        // looks indistinguishable from an absent API or an unknown queue.
        char text[768]{};
        std::snprintf(text, sizeof(text), "Sunshine API depth: %s; capture=%llu sequence=%llu epoch=%llu producer_queue=0x%llx consumer_queue=0x%llx requested_queue=0x%llx producer_fence=%llu producer_completed=%llu completion_valid=%u producer_retired=%u finished=%u success=%u submitted=%u invalid=%u cause=%s; using generic depth fallback",
          depth_capture::name(status), static_cast<unsigned long long>(capture_info.capture_id),
          static_cast<unsigned long long>(capture_info.sequence), static_cast<unsigned long long>(capture_info.epoch),
          static_cast<unsigned long long>(capture_info.queue), static_cast<unsigned long long>(capture_info.consumer_queue),
          static_cast<unsigned long long>(capture_info.requested_queue), static_cast<unsigned long long>(capture_info.producer_fence),
          static_cast<unsigned long long>(capture_info.producer_completed), unsigned(capture_info.producer_completion_valid),
          unsigned(capture_info.producer_recording_retired),
          unsigned(capture_info.finished), unsigned(capture_info.success), unsigned(capture_info.submitted),
          unsigned(capture_info.invalid), depth_capture::name(capture_info.failure));
        reshade::log::message(reshade::log::level::info, text);
        data->last_status = status;
      }
      data->bound_view = {};
      return false;
    }
    auto kind = available ? data->frame.metadata.provider : data->last_metadata.provider;
    std::uint64_t established_source{};
    if (!available && !selection.require_frame_generation)
      depth_capture::provider_identity(runtime->get_command_queue()->get_native(), kind, established_source);
    const char *path_name = kind == sunshine_scene_depth::provider_kind::ngx ? "NGX" : "Streamline";
    data->owns_pass = data->open = true;
    if (selection.require_frame_generation && !available)
      data->last_metadata.source_id = capture_info.source_id;
    if (available) {
      data->shared_preservation = data->frame.shared_preservation;
      data->last_metadata = data->frame.metadata;
    }
    // Keep the encoding identity through gaps, but never expose old pixels as ready.
    data->output.provided = data->last_metadata;
    data->output.projection = projection(data->frame.metadata);
    if (!available) {
      data->output.projection = projection(data->last_metadata);
    }
    display_status display{display_status::not_attempted};
    depth_capture::consumer_diagnostic consumer;
    bool copied = false;
    const bool observation_current = !available || data->frame.metadata.provider != sunshine_scene_depth::provider_kind::streamline ||
      data->frame.metadata.observation_revision == depth_observation_revision();
    if (available && (!observation_current || (!data->frame.shared_preservation && !data->frame.pixel_ready)))
      display = display_status::waiting_capture;
    if (available && observation_current && (data->frame.shared_preservation || data->frame.pixel_ready) &&
        (display = prepare_display(runtime, *data, data->frame)) == display_status::ready) {
      const auto &packet = data->frame;
      sunshine_depth::frame_depth captured;
      copied = sunshine_depth::copy_selected_depth(runtime, commands, packet, data->display_texture, captured, &consumer);
      if (!copied) display = display_status::waiting_capture;
      else if (!(packet.metadata.projection.supplied || packet.metadata.projection.direction_supplied) &&
            captured.detected_orientation != sunshine_depth::depth_orientation::automatic) {
          // Reuse the existing repeated clear-value evidence only when the API
          // supplies no direction. This never creates projection coefficients.
          data->frame.metadata.projection.direction_supplied = true;
          data->frame.metadata.projection.reversed = captured.detected_orientation == sunshine_depth::depth_orientation::reversed;
      }
    }
    if (copied) {
      const auto &packet = data->frame;
      depth_capture::complete_frame(packet, present);
      auto &out = data->output;
      out.provided = packet.metadata;
      out.projection = projection(packet.metadata);
      out.source_resource = {packet.metadata.resource.native};
      out.resource = data->display_texture;
      out.shader_resource = data->display_view;
      out.source_id = packet.metadata.source_id ? packet.metadata.source_id : packet.resource_id;
      out.layout_epoch = packet.metadata.epoch;
      out.frame_index = present;
      out.runtime_epoch = packet.metadata.epoch;
      out.width = packet.width; out.height = packet.height;
      out.x = packet.area.left; out.y = packet.area.top;
      out.active_width = packet.area.width; out.active_height = packet.area.height;
      out.command_queue = packet.queue;
      out.ready = true;
      out.aligned_viewport_assumed = true;
      out.detected_orientation = out.orientation = !(packet.metadata.projection.supplied || packet.metadata.projection.direction_supplied) ?
        sunshine_depth::depth_orientation::automatic : packet.metadata.projection.reversed ?
          sunshine_depth::depth_orientation::reversed : sunshine_depth::depth_orientation::normal;
      data->last_valid = describe_source(packet);
      data->last_metadata = packet.metadata;
      data->last_successful_metadata = packet.metadata;
      // Private display storage is already ordered on this runtime queue.
      // Retaining it never extends a game texture's lifetime or adds a GPU wait.
      data->last_real_depth = out;
      runtime->get_screenshot_width_and_height(&data->last_color_width, &data->last_color_height);
    }
    const auto reuse_tick = GetTickCount64();
    const auto prior_depth_age = data->last_real_depth.ready && reuse_tick >= data->last_real_depth.provided.tick ?
      reuse_tick - data->last_real_depth.provided.tick : UINT64_MAX;
    if (!copied && data->last_real_depth.ready && same_successful_fg_source(*data, capture_info)) {
      const bool repeated = !available && capture_info.repeated_frame &&
        capture_info.capture_id == data->last_valid.capture && capture_info.sequence == data->last_valid.sequence;
      const bool pending = capture_info.pending_frame &&
        (!available || (!data->frame.shared_preservation && !data->frame.pixel_ready));
      const auto &previous = data->last_real_depth;
      const auto now = GetTickCount64();
      std::uint32_t color_width{}, color_height{};
      runtime->get_screenshot_width_and_height(&color_width, &color_height);
      const bool enabled = selection.require_frame_generation &&
        selection.epoch == previous.provided.epoch && selection.viewport == previous.provided.viewport;
      const bool same_shape = !available || (data->frame.width == previous.width && data->frame.height == previous.height &&
        data->frame.format == data->display_format && data->frame.area.left == previous.x && data->frame.area.top == previous.y &&
        data->frame.area.width == previous.active_width && data->frame.area.height == previous.active_height);
      if ((repeated || pending) && enabled && same_shape &&
          previous.provided.observation_revision == depth_observation_revision() &&
          color_width == data->last_color_width && color_height == data->last_color_height &&
          commands == runtime->get_command_queue()->get_immediate_command_list() &&
          previous.command_queue == runtime->get_command_queue()->get_native() &&
          previous.resource == data->display_texture && previous.shader_resource == data->display_view &&
          previous.provided.tick && now >= previous.provided.tick && now - previous.provided.tick < sunshine_scene_depth::maximum_source_age_ms) {
        data->output = previous;
        data->output.frame_index = present;
        data->output.reused_depth = true;
        display = display_status::reused_depth;
      }
    }
    // A failed/missing/expired frame breaks the reuse interval. A later pending
    // nomination cannot resurrect it; a fresh successful copy must rearm it.
    if (!data->output.ready) data->last_real_depth = {};
    const auto &stats_source = data->output.provided;
    data->output.frame_generation_active = selection.require_frame_generation && selection.epoch &&
      stats_source.provider == sunshine_scene_depth::provider_kind::streamline && stats_source.frame_generation_input &&
      selection.epoch == stats_source.epoch && selection.viewport == stats_source.viewport;
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
    bind(runtime, *data, data->display_view, data->output.ready);
    const bool handoff_changed = display != data->last_display || consumer.result != data->last_consumer ||
      consumer.invalidation != data->last_loss || data->output.ready != data->last_output_ready;
    if (data->logging.due(GetTickCount64(), status != data->last_status ||
        capture_info.failure != data->last_capture_failure || handoff_changed, !previously_owned, !available)) {
      char text[1536]{};
      if (available)
        std::snprintf(text, sizeof(text), "Sunshine %s depth: %s; source_selected=1 snapshot_ready=%u final_ready=%u display=%s consumer=%s loss=%s; allocation=%ux%u active=%u,%u,%u,%u source=%u viewport=%u; metadata_projection=%u output_projection=%u; state=0x%x proof=%u; capture=%llu sequence=%llu present=%llu command=0x%llx cookie=%llu device=%llu expected_device=%llu slot_invalid=%u tracked_commands=%u; producer_queue=0x%llx consumer_queue=0x%llx producer_fence=%llu producer_completed=%llu completion_valid=%u producer_retired=%u; transport=%s resource_id=%llu",
          path_name, depth_capture::name(status), unsigned(data->frame.pixel_ready), unsigned(data->output.ready), name(display), depth_capture::name(consumer.result),
          depth_capture::name(consumer.invalidation),
          data->frame.width, data->frame.height, data->frame.area.left, data->frame.area.top, data->frame.area.width, data->frame.area.height,
          unsigned(data->frame.metadata.resource.kind), data->frame.metadata.viewport,
          unsigned(data->frame.metadata.projection.supplied), unsigned(data->output.projection.supplied),
          data->frame.metadata.native_state, unsigned(data->frame.metadata.proof),
          static_cast<unsigned long long>(data->frame.capture_id), static_cast<unsigned long long>(data->frame.metadata.sequence),
          static_cast<unsigned long long>(present), static_cast<unsigned long long>(commands->get_native()),
          static_cast<unsigned long long>(consumer.cookie), static_cast<unsigned long long>(consumer.device_identity),
          static_cast<unsigned long long>(consumer.expected_device_identity), unsigned(consumer.slot_invalid), consumer.tracked_commands,
          static_cast<unsigned long long>(capture_info.queue), static_cast<unsigned long long>(capture_info.consumer_queue),
          static_cast<unsigned long long>(capture_info.producer_fence), static_cast<unsigned long long>(capture_info.producer_completed),
          unsigned(capture_info.producer_completion_valid), unsigned(capture_info.producer_recording_retired),
          data->frame.shared_preservation ? "ReShade preservation" : "API snapshot", static_cast<unsigned long long>(data->frame.resource_id));
      else {
        // A failed acquisition has no output packet. Its zeroed fields say
        // nothing about the resource/camera that the provider actually tagged.
        const auto observer = native_observer::counts();
        std::snprintf(text, sizeof(text), "Sunshine %s depth: %s; source_selected=0 provider_held=1 final_ready=%u display=%s cause=%s; capture=%llu sequence=%llu epoch=%llu producer_recording=%llu queue=0x%llx consumer_queue=0x%llx producer_fence=%llu producer_completed=%llu completion_valid=%u producer_retired=%u retire_fence=%llu finished=%u success=%u submitted=%u invalid=%u; observer_dropped=%llu unreadable=%llu rejected=%llu barrier_overflow=%llu submission_overflow=%llu discovery_contention=%llu",
          path_name, depth_capture::name(status), unsigned(data->output.ready), name(display), depth_capture::name(capture_info.failure),
          static_cast<unsigned long long>(capture_info.capture_id), static_cast<unsigned long long>(capture_info.sequence),
          static_cast<unsigned long long>(capture_info.epoch), static_cast<unsigned long long>(capture_info.command),
          static_cast<unsigned long long>(capture_info.queue), static_cast<unsigned long long>(capture_info.consumer_queue),
          static_cast<unsigned long long>(capture_info.producer_fence), static_cast<unsigned long long>(capture_info.producer_completed),
          unsigned(capture_info.producer_completion_valid), unsigned(capture_info.producer_recording_retired),
          static_cast<unsigned long long>(capture_info.retire_fence), unsigned(capture_info.finished), unsigned(capture_info.success),
          unsigned(capture_info.submitted), unsigned(capture_info.invalid), static_cast<unsigned long long>(observer.dropped),
          static_cast<unsigned long long>(observer.unreadable), static_cast<unsigned long long>(observer.rejected),
          static_cast<unsigned long long>(observer.barrier_overflow), static_cast<unsigned long long>(observer.submission_overflow),
          static_cast<unsigned long long>(observer.discovery_contention));
      }
      const auto used = std::strlen(text);
      const auto &jitter = data->output.provided.jitter;
      std::snprintf(text + used, sizeof(text) - used, "; newest_sequence=%llu observation_current=%u prior_depth_age_ms=%llu jitter_supplied=%u jitter_px=(%.6g,%.6g) jitter_render=%ux%u",
        static_cast<unsigned long long>(capture_info.newest_sequence), unsigned(observation_current),
        static_cast<unsigned long long>(prior_depth_age), unsigned(jitter.supplied),
        double(jitter.x), double(jitter.y), jitter.width, jitter.height);
      reshade::log::message(reshade::log::level::info, text);
      data->last_status = status;
      data->last_capture_failure = capture_info.failure;
      data->last_display = display;
      data->last_consumer = consumer.result;
      data->last_loss = consumer.invalidation;
      data->last_output_ready = data->output.ready;
    }
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
      request.capture_id = data->frame.capture_id;
      // Both transports publish to this stable display allocation. The sampler
      // retains its COM source and submits after these effects on the same
      // runtime queue; no pooled preservation backup escapes its owner.
      request.source = data->display_texture;
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
  extern "C" __declspec(dllexport) bool SunshineStreamlineTestSnapshot(api::effect_runtime *runtime,
      std::uint64_t *texture, bool *forced, bool *shared) {
    const auto *data = runtime->get_private_data<state>();
    if (!data || !texture || !forced || !shared) return false;
    *texture = data->frame.texture;
    *forced = data->frame.metadata.force_snapshot;
    *shared = data->frame.shared_preservation;
    return data->output.ready && !data->output.reused_depth;
  }
#endif
  bool selected(api::effect_runtime *runtime) {
    const auto *data = runtime->get_private_data<state>();
    return data && data->owns_pass;
  }
  void invalidate_reused_depth(api::effect_runtime *runtime) {
    if (auto *data = runtime->get_private_data<state>()) {
      data->last_real_depth = {};
      data->presentations.reset();
      data->presentation_source = {};
      if (data->output.reused_depth) {
        data->output = {};
        set_depth_ready(runtime, false);
      }
    }
  }
  bool uses_shared_preservation(api::effect_runtime *runtime) {
    const auto *data = runtime->get_private_data<state>();
    return data && data->owns_pass && data->shared_preservation;
  }
  source_status describe(api::effect_runtime *runtime, std::uint64_t present) {
    source_status out;
    const auto *data = runtime->get_private_data<state>();
    if (!data) return out;
    out.selected = data->owns_pass;
    out.shared_preservation = data->shared_preservation;
    out.last_valid = data->last_valid;
    out.presentations = data->presentations.snapshot(GetTickCount64());
    out.provider = data->last_metadata.provider;
    std::uint64_t source{};
    if (!data->owns_pass)
      depth_capture::provider_identity(runtime->get_command_queue()->get_native(), out.provider, source);
    out.ready = out.selected && data->output.ready && data->output.frame_index == present;
    out.reused_depth = out.ready && data->output.reused_depth;
    if (out.ready) out.current = data->output.reused_depth ? data->last_valid : describe_source(data->frame);
    return out;
  }
}
