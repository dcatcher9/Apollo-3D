// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_depth_provider.h"
#include "streamline_camera_probe.h"
#include "display_depth_cache.h"
#include "streamline_depth_capture.h"
#include "streamline_native_observer.h"
#include "depth_content_sampler.h"
#include "depth_ready_uniform_cache.h"
#include "diagnostic_log_gate.h"
#include "projection_depth_scale.h"
#include <reshade.hpp>
#include <algorithm>
#include <cstring>
#include <cstdio>

namespace sunshine_streamline::provider {
  namespace api = reshade::api;
  enum class display_status { not_attempted, ready, reused_depth, texture_creation_failed, view_creation_failed, waiting_capture };
  struct observation_loss_lookup {
    std::uint64_t requested_revision{};
    loss_diagnostics::event value;
    bool found{};
  };
  struct observation_loss_episode {
    std::uint64_t present{}, sampled_tick{};
    observation_loss_lookup current_check, retained_check, sampled;
  };
    struct __declspec(uuid("e82eab46-4d62-4e80-9e02-0e60b7059f36")) state {
      depth_capture::packet frame;
      sunshine_scene_depth::frame source_metadata;
      sunshine_depth::frame_depth output;
      display_depth_cache cache;
      sunshine_depth_stats::presentation_window presentations;
      std::array<std::uint64_t, 5> presentation_source{};
      source_description last_valid;
      center_sample latest;
      depth_capture::input pending_metadata;
      sunshine_depth_statistics::tile_layout pending_grid;
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
    };
}
#if defined(__MINGW32__)
__CRT_UUID_DECL(sunshine_streamline::provider::state, 0xe82eab46, 0x4d62, 0x4e80, 0x9e, 0x02, 0x0e, 0x60, 0xb7, 0x05, 0x9f, 0x36)
#endif
namespace sunshine_streamline::provider {
  namespace {
    struct readiness_evidence {
      sunshine_depth::frame_depth prior;
      depth_capture::capture_diagnostic capture;
      depth_capture::consumer_diagnostic consumer;
      frame_generation_snapshot fg;
      depth_capture::selection_policy selection;
      frame_generation_query_status fg_query{frame_generation_query_status::unavailable};
      display_status display{display_status::not_attempted};
      sunshine_scene_depth::provider_kind prior_provider{sunshine_scene_depth::provider_kind::streamline};
      std::uint64_t present{}, tick{}, current_check_revision{}, retained_check_revision{}, prior_capture{};
      bool prior_provider_known{}, prior_output_ready{}, available{}, copied{}, snapshot_ready{};
      depth_capture::display_action action{depth_capture::display_action::invalidate};
      const char *source_reason{"not_checked"}, *cache_reason{"not_checked"};
      int current_observation{-1};
    };
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
        metadata.projection.raw_scale, metadata.projection.raw_bias};
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
      // Retain the upstream preserved-copy observation through the owned
      // display copy. These IDs describe that capture, not the display texture
      // or a color/depth frame match. Native snapshots leave them unavailable.
      out.capture_marker = captured.capture_marker;
      out.depth_copy = captured.depth_copy;
      out.backup_id = captured.backup_id;
      out.ready = true;
      out.aligned_viewport_assumed = true;
      out.detected_orientation = out.orientation = !(packet.metadata.projection.supplied || packet.metadata.projection.direction_supplied) ?
        sunshine_depth::depth_orientation::automatic : packet.metadata.projection.reversed ?
          sunshine_depth::depth_orientation::reversed : sunshine_depth::depth_orientation::normal;
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
      data.cache.invalidate("display_recreated");
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
    const char *name(depth_capture::display_action action) {
      switch (action) {
        case depth_capture::display_action::copy_fresh: return "copy_fresh";
        case depth_capture::display_action::hold: return "hold";
        case depth_capture::display_action::invalidate: return "invalidate";
      }
      return "unknown";
    }
    void trace_readiness(api::effect_runtime *runtime, state &data, const readiness_evidence &e,
        bool ready, const char *why) {
      const bool changed = data.traced_ready != ready;
      data.traced_ready = ready;
      if (!changed) return;
      const auto now = e.tick ? e.tick : GetTickCount64();
      if (!ready) {
        data.trace_loss_tick = now;
        data.traced_loss = now >= data.next_trace_episode;
        if (!data.traced_loss) { ++data.suppressed_trace_episodes; return; }
        // At most four admitted loss episodes per second, each with its paired
        // recovery. No ready/unavailable decision depends on this diagnostic.
        data.next_trace_episode = now + 250;
        const auto lookup = [](std::uint64_t revision) {
          observation_loss_lookup out;
          out.requested_revision = revision;
          out.found = query_depth_observation_loss(revision, out.value);
          return out;
        };
        auto &loss = data.trace_observation_loss;
        loss.present = e.present;
        loss.sampled_tick = GetTickCount64();
        // Preserve both actual checks: callbacks can advance the revision between
        // them. The additional sample is only contemporaneous evidence when no
        // check ran (for example an absent nomination), never a causal verdict.
        loss.current_check = lookup(e.current_check_revision);
        loss.retained_check = lookup(e.retained_check_revision);
        loss.sampled = lookup(depth_observation_revision());
      } else if (!data.traced_loss) return; // No startup/recovery-only noise.
      const auto age = [now](std::uint64_t tick) -> unsigned long long {
        return tick && now >= tick ? now - tick : UINT64_MAX;
      };
      const auto u = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
      const auto &c = e.capture;
      const auto selection_age = [&c](std::uint64_t tick) -> unsigned long long {
        return c.selection_tick_ms && tick && c.selection_tick_ms >= tick ? c.selection_tick_ms - tick : UINT64_MAX;
      };
      const auto &p = e.prior.provided;
      const auto &current = data.frame.metadata;
      const auto provider_name = [](sunshine_scene_depth::provider_kind provider) {
        return provider == sunshine_scene_depth::provider_kind::ngx ? "NGX" : "Streamline";
      };
      // An empty acquisition diagnostic's default enum is not source evidence.
      // FG policy can still establish Streamline authority before a tag exists.
      const char *current_provider = c.capture_id || c.sequence ? provider_name(c.provider) :
        e.selection.require_frame_generation ? "Streamline" : "unknown";
      char text[3072]{};
      std::snprintf(text, sizeof(text),
        "Sunshine depth readiness: %s reason=%s runtime=0x%llx native_swapchain=0x%llx present=%llu unavailable_ms=%llu suppressed_episodes=%llu; "
        "provider=%s prior_provider=%s selected=%u snapshot_ready=%u copied=%u output_reused=%u display=%s selection=%s status=%s failure=%s "
        "capture=%llu sequence=%llu newest_sequence=%llu epoch=%llu source=%llu viewport=%u "
        "selection_tick_ms=%llu source_age_ms=%llu newest_view_age_ms=%llu active_views=%u "
        "finished=%u success=%u submitted=%u invalid=%u capture_pending=%u capture_repeated=%u "
        "fence=%llu completed=%llu completion_valid=%u producer_retired=%u; "
        "FG_query=%u FG_known=%u FG_enabled=%u FG_mode=%u FG_generated=%u FG_epoch=%llu FG_viewport=%u require_FG=%u policy_epoch=%llu; "
        "prior_output_ready=%u prior_retained=%u prior_age_ms=%llu prior_present=%llu prior_capture=%llu prior_sequence=%llu prior_epoch=%llu prior_source=%llu "
        "prior_viewport=%u prior_observation=%llu current_observation=%llu current_check_revision=%llu retained_check_revision=%llu prior_feedback=%llu current_feedback=%llu "
        "prior_reset=%u current_reset=%u retained_clear=%s; "
        "source_action=%s source_reason=%s cache_reason=%s current_observation_ok=%d "
        "consumer=%s consumer_loss=%s cookie=%llu slot_invalid=%u",
        ready ? "recovered" : "lost", why, u(reinterpret_cast<std::uint64_t>(runtime)), u(runtime->get_native()), u(e.present),
        ready && now >= data.trace_loss_tick ? u(now - data.trace_loss_tick) : 0ull, u(data.suppressed_trace_episodes),
        current_provider, e.prior_provider_known ? provider_name(e.prior_provider) : "unknown", unsigned(e.available),
        unsigned(e.snapshot_ready), unsigned(e.copied), unsigned(data.output.reused_depth), name(e.display),
        depth_capture::name(c.selection), depth_capture::name(c.result), depth_capture::name(c.failure),
        u(c.capture_id), u(c.sequence), u(c.newest_sequence), u(c.epoch), u(c.source_id), c.viewport,
        u(c.selection_tick_ms), selection_age(c.source_tick_ms), selection_age(c.newest_view_tick_ms), c.active_view_count,
        unsigned(c.finished), unsigned(c.success), unsigned(c.submitted), unsigned(c.invalid),
        unsigned(c.pending_frame), unsigned(c.repeated_frame), u(c.producer_fence),
        u(c.producer_completed), unsigned(c.producer_completion_valid), unsigned(c.producer_recording_retired),
        unsigned(e.fg_query), unsigned(e.fg.known), unsigned(e.fg.enabled), e.fg.mode, e.fg.generated_frames,
        u(e.fg.epoch), e.fg.viewport, unsigned(e.selection.require_frame_generation), u(e.selection.epoch),
        unsigned(e.prior_output_ready), unsigned(e.prior.ready), age(p.tick), u(e.prior.frame_index), u(e.prior_capture), u(p.sequence), u(p.epoch),
        u(p.source_id), p.viewport, u(p.observation_revision), u(current.observation_revision),
        u(e.current_check_revision), u(e.retained_check_revision),
        u(p.feedback.revision), u(current.feedback.revision), unsigned(p.feedback.reset), unsigned(current.feedback.reset),
        data.cache.reason(), name(e.action), e.source_reason, e.cache_reason, e.current_observation,
        depth_capture::name(e.consumer.result), depth_capture::name(e.consumer.invalidation),
        u(e.consumer.cookie), unsigned(e.consumer.slot_invalid));
      reshade::log::message(reshade::log::level::info, text);
      // Repeat the frozen first-loss evidence on recovery, without querying a
      // newer revision or retroactively filling a missing/busy journal entry.
      const auto format_loss = [](const observation_loss_lookup &lookup, char *out, std::size_t size) {
        const auto &event = lookup.value;
        const auto &context = event.details;
        std::snprintf(out, size,
          "revision=%llu found=%u cause=%s site=%s line=%u tick_ms=%llu thread=%u sequence=%llu viewport=%u feature=%u sdk_known=%u sdk_result=%d reset=%u",
          static_cast<unsigned long long>(lookup.requested_revision), unsigned(lookup.found),
          lookup.found ? loss_diagnostics::name(event.cause) : "unavailable",
          lookup.found && event.site ? event.site : "unavailable", event.line,
          static_cast<unsigned long long>(event.tick), event.thread_id,
          static_cast<unsigned long long>(context.sequence), context.viewport, context.feature,
          unsigned(context.has_sdk_result), context.sdk_result, context.reset);
      };
      char current_loss[512]{}, retained_loss[512]{}, sampled_loss[512]{};
      const auto &loss = data.trace_observation_loss;
      format_loss(loss.current_check, current_loss, sizeof(current_loss));
      format_loss(loss.retained_check, retained_loss, sizeof(retained_loss));
      format_loss(loss.sampled, sampled_loss, sizeof(sampled_loss));
      std::snprintf(text, sizeof(text),
        "Sunshine depth observation evidence: %s runtime=0x%llx loss_present=%llu loss_tick_ms=%llu sampled_tick_ms=%llu origin=episode_start; "
        "current_check={%s}; retained_check={%s}; sampled_only={%s}",
        ready ? "recovered" : "lost", u(reinterpret_cast<std::uint64_t>(runtime)), u(loss.present),
        u(data.trace_loss_tick), u(loss.sampled_tick), current_loss, retained_loss, sampled_loss);
      reshade::log::message(reshade::log::level::info, text);
      data.suppressed_trace_episodes = 0;
      if (ready) data.traced_loss = false;
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
#ifdef SUNSHINE_SBS_TEST
  // Exercise the exact value-only publication used by the renderer and dump
  // without allocating resources or executing a GPU capture.
  void test_describe_copied_depth(sunshine_depth::frame_depth &out, const depth_capture::packet &packet,
      const sunshine_depth::frame_depth &captured, api::resource display_texture,
      api::resource_view display_view, std::uint64_t present) {
    describe_copied_depth(out, packet, captured, display_texture, display_view, present);
  }
#endif
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
      const auto evidence = before_decision(*data, data->output.frame_index);
      data->depth_ready.invalidate();
      data->open = data->owns_pass = data->have_latest = data->shared_preservation = false;
      data->cache.invalidate("runtime_reload");
      data->presentations.reset();
      data->presentation_source = {};
      data->frame = {}; data->output = {};
      data->last_valid = {};
      data->source_metadata = {};
      data->source_policy = {};
      data->bound_view = {};
      // The sampler independently retains any submitted slot until completion.
      data->pending_id = 0;
      trace_readiness(runtime, *data, evidence, false, "runtime_reload");
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
    const bool matches = value.valid && value.moments.supplied && value.source.handle == data->pending_texture &&
      value.width == data->pending_grid.x && value.height == data->pending_grid.y &&
      value.values.size() == std::size_t(data->pending_grid.x) * data->pending_grid.y &&
      value.values.size() <= data->latest.raw.size() &&
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
      data->latest.raw = {};
      std::copy(value.values.begin(), value.values.end(), data->latest.raw.begin());
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
    data->open = data->owns_pass = false;
    data->frame = {}; data->output = {};
    if (!allowed || runtime->get_device()->get_api() != api::device_api::d3d12) {
      data->source_policy = {};
      data->cache.invalidate(!allowed ? "provider_disallowed" : "unsupported_API");
      data->presentations.reset();
      data->presentation_source = {};
      data->bound_view = {};
      trace_readiness(runtime, *data, evidence, false, data->cache.reason());
      return false;
    }
    depth_capture::observe_command(commands->get_native());
    depth_capture::observe_queue(runtime->get_command_queue()->get_native());
    frame_generation_snapshot fg_source;
    frame_generation_query_status fg_status;
    query_frame_generation(UINT32_MAX, fg_source, &fg_status);
    const auto selection = data->source_policy.update(fg_status, fg_source);
    evidence.fg = fg_source;
    evidence.fg_query = fg_status;
    evidence.selection = selection;
    if (selection.require_frame_generation) {
      const auto &previous = data->source_metadata;
      if (previous.provider != sunshine_scene_depth::provider_kind::streamline || !previous.frame_generation_input ||
          previous.epoch != selection.epoch || previous.viewport != selection.viewport) {
        // FG selects a logical SL source before its first pixels are ready.
        // Never carry NGX's encoding, sampled center or old display into it.
        data->source_metadata = {};
        data->source_metadata.provider = sunshine_scene_depth::provider_kind::streamline;
        data->source_metadata.epoch = selection.epoch;
        data->source_metadata.viewport = selection.viewport;
        data->source_metadata.frame_generation_input = true;
        data->cache.invalidate("FG_scope_changed");
        data->last_valid = {};
        data->have_latest = data->shared_preservation = false;
        data->pending_id = 0;
      }
    }
    depth_capture::capture_diagnostic capture_info;
    depth_capture::acquisition_decision acquisition;
    depth_capture::acquire(runtime->get_command_queue()->get_native(), present, data->frame, acquisition, &capture_info, selection);
    const bool available = acquisition.source_selected;
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
    const bool associated = selection.require_frame_generation || depth_capture::provider_active(runtime->get_command_queue()->get_native());
    if (!available && !associated) {
      data->cache.invalidate("source_not_associated");
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
      trace_readiness(runtime, *data, evidence, false, "source_not_associated");
      return false;
    }
    auto kind = available ? data->frame.metadata.provider : data->source_metadata.provider;
    std::uint64_t established_source{};
    if (!available && !selection.require_frame_generation)
      depth_capture::provider_identity(runtime->get_command_queue()->get_native(), kind, established_source);
    const char *path_name = kind == sunshine_scene_depth::provider_kind::ngx ? "NGX" : "Streamline";
    data->owns_pass = data->open = true;
    if (selection.require_frame_generation && !available)
      data->source_metadata.source_id = acquisition.source_id;
    if (available) {
      data->shared_preservation = data->frame.shared_preservation;
      data->source_metadata = data->frame.metadata;
    }
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
    const auto update = depth_capture::decide_display(acquisition, data->frame, data->cache.reference(),
      selection, present, reuse_tick, depth_observation_revision());
    evidence.action = update.action;
    evidence.source_reason = update.reason;
    evidence.current_check_revision = update.current_check_revision;
    evidence.retained_check_revision = update.retained_check_revision;
    evidence.current_observation = update.current_observation;
    if (available) display = display_status::waiting_capture;
    if (update.action == depth_capture::display_action::copy_fresh &&
        (display = prepare_display(runtime, *data, data->frame)) == display_status::ready) {
      const auto &packet = data->frame;
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
      describe_copied_depth(out, packet, captured, data->display_texture, data->display_view, present);
      data->last_valid = describe_source(packet);
      data->source_metadata = packet.metadata;
      // Private display storage is already ordered on this runtime queue.
      // Retaining it never extends a game texture's lifetime or adds a GPU wait.
      std::uint32_t color_width{}, color_height{};
      runtime->get_screenshot_width_and_height(&color_width, &color_height);
      data->cache.commit(out, packet.capture_id, packet.format, color_width, color_height);
    } else if (update.action == depth_capture::display_action::hold) {
      display_cache_context context;
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
    } else {
      // Failed fresh copies cannot fall back into hold. Once invalidated, only
      // a successful new copy can establish another cache entry.
      data->cache.invalidate(update.action == depth_capture::display_action::copy_fresh ?
        "display_copy_failed" : update.reason);
    }
    const char *readiness_reason = copied ? "fresh_copy" : data->output.ready ? "retained_copy" : data->cache.reason();
    evidence.cache_reason = data->cache.reason();
    evidence.copied = copied;
    evidence.display = display;
    evidence.consumer = consumer;
    if (!evidence.tick) evidence.tick = reuse_tick;
    trace_readiness(runtime, *data, evidence, data->output.ready, readiness_reason);
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
      capture_info.selection != data->last_selection ||
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
      std::snprintf(text + used, sizeof(text) - used, "; selection=%s consumed_completed_capture=%llu newest_sequence=%llu observation_current=%u prior_depth_age_ms=%llu jitter_supplied=%u jitter_px=(%.6g,%.6g) jitter_render=%ux%u",
        depth_capture::name(capture_info.selection), static_cast<unsigned long long>(capture_info.consumed_completed_capture),
        static_cast<unsigned long long>(capture_info.newest_sequence), unsigned(update.current_observation),
        static_cast<unsigned long long>(prior_depth_age), unsigned(jitter.supplied),
        double(jitter.x), double(jitter.y), jitter.width, jitter.height);
      reshade::log::message(reshade::log::level::info, text);
      data->last_status = status;
      data->last_selection = capture_info.selection;
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
      request.collect_range = true;
      // Decode on the GPU with this capture's coefficients. Never retrofit a
      // later camera or inferred orientation when asynchronous results arrive.
      const auto &basis = data->frame.metadata.projection;
      const auto coefficients = basis.supplied ? sunshine_projection_depth::make(
        basis.depth_offset, basis.depth_scale, basis.raw_scale, basis.raw_bias) :
        sunshine_projection_depth::coefficients{};
      if (coefficients.valid()) {
        request.collect_moments = true;
        request.moments_A = coefficients.shader_A;
        request.moments_inverseB = coefficients.inverseB;
      } else if (!basis.supplied && basis.direction_supplied) {
        request.collect_moments = true;
        request.moments_A = basis.reversed ? 0.f : 1.f;
        request.moments_inverseB = basis.reversed ? 1.f : -1.f;
      }
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
      const auto evidence = before_decision(*data, data->output.frame_index);
      data->cache.invalidate("explicit_reuse_invalidation");
      data->presentations.reset();
      data->presentation_source = {};
      if (data->output.reused_depth) {
        data->output = {};
        set_depth_ready(runtime, false);
      }
      trace_readiness(runtime, *data, evidence, data->output.ready, "explicit_reuse_invalidation");
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
    out.provider = data->source_metadata.provider;
    std::uint64_t source{};
    if (!data->owns_pass)
      depth_capture::provider_identity(runtime->get_command_queue()->get_native(), out.provider, source);
    out.ready = out.selected && data->output.ready && data->output.frame_index == present;
    out.reused_depth = out.ready && data->output.reused_depth;
    if (out.ready) out.current = data->output.reused_depth ? data->last_valid : describe_source(data->frame);
    return out;
  }
}
