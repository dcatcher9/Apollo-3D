// SPDX-License-Identifier: GPL-3.0-only
// Exercise the actual publisher state transitions without loading an add-on into a game.
#include "exporter.cpp"
#include "depth_cache_update.h"
#include "streamline_depth_provider.h"

#include <algorithm>
#include <atomic>
#include <future>
#include <stdexcept>
#include <thread>
#include "test_exporter_async_d3d12.h"

namespace sunshine_streamline::provider {
  void test_describe_copied_depth(sunshine_depth::frame_depth &, const depth_capture::packet &,
    const sunshine_depth::frame_depth &, reshade::api::resource, reshade::api::resource_view, std::uint64_t);
}

namespace {
  void require(bool result, const char *message) {
    if (!result) {
      throw std::runtime_error(message);
    }
  }

  namespace runtime_fixture {
    using namespace reshade::api;
    // A real interface implementation with no attached add-on state. Lifecycle
    // invalidation may query private data; every other unexpected call fails.
    struct empty_runtime : effect_runtime {
      void get_private_data(const uint8_t[16], uint64_t *data) const override { *data = 0; }
#define UNEXPECTED_RUNTIME_CALL(...) __VA_ARGS__ override { throw std::runtime_error("Unexpected ReShade runtime call: " #__VA_ARGS__); }
      UNEXPECTED_RUNTIME_CALL(uint64_t get_native() const)
      UNEXPECTED_RUNTIME_CALL(void set_private_data(const uint8_t[16], uint64_t))
      UNEXPECTED_RUNTIME_CALL(device *get_device())
      UNEXPECTED_RUNTIME_CALL(void *get_hwnd() const)
      UNEXPECTED_RUNTIME_CALL(resource get_back_buffer(uint32_t index))
      UNEXPECTED_RUNTIME_CALL(uint32_t get_back_buffer_count() const)
      UNEXPECTED_RUNTIME_CALL(uint32_t get_current_back_buffer_index() const)
      UNEXPECTED_RUNTIME_CALL(command_queue *get_command_queue())
      UNEXPECTED_RUNTIME_CALL(void render_effects(command_list *cmd_list, resource_view rtv, resource_view rtv_srgb))
      UNEXPECTED_RUNTIME_CALL(bool capture_screenshot(void *pixels))
      UNEXPECTED_RUNTIME_CALL(void get_screenshot_width_and_height(uint32_t *out_width, uint32_t *out_height) const)
      UNEXPECTED_RUNTIME_CALL(bool is_key_down(uint32_t keycode) const)
      UNEXPECTED_RUNTIME_CALL(bool is_key_pressed(uint32_t keycode) const)
      UNEXPECTED_RUNTIME_CALL(bool is_key_released(uint32_t keycode) const)
      UNEXPECTED_RUNTIME_CALL(bool is_mouse_button_down(uint32_t button) const)
      UNEXPECTED_RUNTIME_CALL(bool is_mouse_button_pressed(uint32_t button) const)
      UNEXPECTED_RUNTIME_CALL(bool is_mouse_button_released(uint32_t button) const)
      UNEXPECTED_RUNTIME_CALL(void get_mouse_cursor_position(uint32_t *out_x, uint32_t *out_y, int16_t *out_wheel_delta = nullptr) const)
      UNEXPECTED_RUNTIME_CALL(void enumerate_uniform_variables(const char *effect_name, void(*callback)(effect_runtime *runtime, effect_uniform_variable variable, void *user_data), void *user_data))
      UNEXPECTED_RUNTIME_CALL(effect_uniform_variable find_uniform_variable(const char *effect_name, const char *variable_name) const)
      UNEXPECTED_RUNTIME_CALL(void get_uniform_variable_type(effect_uniform_variable variable, format *out_base_type, uint32_t *out_rows = nullptr, uint32_t *out_columns = nullptr, uint32_t *out_array_length = nullptr) const)
      UNEXPECTED_RUNTIME_CALL(void get_uniform_variable_name(effect_uniform_variable variable, char *name, size_t *name_size) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_bool_from_uniform_variable(effect_uniform_variable variable, const char *name, bool *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_float_from_uniform_variable(effect_uniform_variable variable, const char *name, float *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_int_from_uniform_variable(effect_uniform_variable variable, const char *name, int32_t *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_uint_from_uniform_variable(effect_uniform_variable variable, const char *name, uint32_t *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_string_from_uniform_variable(effect_uniform_variable variable, const char *name, char *value, size_t *value_size) const)
      UNEXPECTED_RUNTIME_CALL(void get_uniform_value_bool(effect_uniform_variable variable, bool *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(void get_uniform_value_float(effect_uniform_variable variable, float *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(void get_uniform_value_int(effect_uniform_variable variable, int32_t *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(void get_uniform_value_uint(effect_uniform_variable variable, uint32_t *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(void set_uniform_value_bool(effect_uniform_variable variable, const bool *values, size_t count, size_t array_index = 0))
      UNEXPECTED_RUNTIME_CALL(void set_uniform_value_float(effect_uniform_variable variable, const float *values, size_t count, size_t array_index = 0))
      UNEXPECTED_RUNTIME_CALL(void set_uniform_value_int(effect_uniform_variable variable, const int32_t *values, size_t count, size_t array_index = 0))
      UNEXPECTED_RUNTIME_CALL(void set_uniform_value_uint(effect_uniform_variable variable, const uint32_t *values, size_t count, size_t array_index = 0))
      UNEXPECTED_RUNTIME_CALL(void enumerate_texture_variables(const char *effect_name, void(*callback)(effect_runtime *runtime, effect_texture_variable variable, void *user_data), void *user_data))
      UNEXPECTED_RUNTIME_CALL(effect_texture_variable find_texture_variable(const char *effect_name, const char *variable_name) const)
      UNEXPECTED_RUNTIME_CALL(void get_texture_variable_name(effect_texture_variable variable, char *name, size_t *name_size) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_bool_from_texture_variable(effect_texture_variable variable, const char *name, bool *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_float_from_texture_variable(effect_texture_variable variable, const char *name, float *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_int_from_texture_variable(effect_texture_variable variable, const char *name, int32_t *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_uint_from_texture_variable(effect_texture_variable variable, const char *name, uint32_t *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_string_from_texture_variable(effect_texture_variable variable, const char *name, char *value, size_t *value_size) const)
      UNEXPECTED_RUNTIME_CALL(void update_texture(effect_texture_variable variable, const uint32_t width, const uint32_t height, const void *pixels))
      UNEXPECTED_RUNTIME_CALL(void get_texture_binding(effect_texture_variable variable, resource_view *out_srv, resource_view *out_srv_srgb) const)
      UNEXPECTED_RUNTIME_CALL(void update_texture_bindings(const char *semantic, resource_view srv, resource_view srv_srgb))
      UNEXPECTED_RUNTIME_CALL(void enumerate_techniques(const char *effect_name, void(*callback)(effect_runtime *runtime, effect_technique technique, void *user_data), void *user_data))
      UNEXPECTED_RUNTIME_CALL(effect_technique find_technique(const char *effect_name, const char *technique_name))
      UNEXPECTED_RUNTIME_CALL(void get_technique_name(effect_technique technique, char *name, size_t *name_size) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_bool_from_technique(effect_technique technique, const char *name, bool *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_float_from_technique(effect_technique technique, const char *name, float *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_int_from_technique(effect_technique technique, const char *name, int32_t *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_uint_from_technique(effect_technique technique, const char *name, uint32_t *values, size_t count, size_t array_index = 0) const)
      UNEXPECTED_RUNTIME_CALL(bool get_annotation_string_from_technique(effect_technique technique, const char *name, char *value, size_t *value_size) const)
      UNEXPECTED_RUNTIME_CALL(bool get_technique_state(effect_technique technique) const)
      UNEXPECTED_RUNTIME_CALL(void set_technique_state(effect_technique technique, bool enabled))
      UNEXPECTED_RUNTIME_CALL(bool get_preprocessor_definition(const char *name, char *value, size_t *value_size) const)
      UNEXPECTED_RUNTIME_CALL(void set_preprocessor_definition(const char *name, const char *value))
      UNEXPECTED_RUNTIME_CALL(void render_technique(effect_technique technique, command_list *cmd_list, resource_view rtv, resource_view rtv_srgb = { 0 }))
      UNEXPECTED_RUNTIME_CALL(bool get_effects_state() const)
      UNEXPECTED_RUNTIME_CALL(void set_effects_state(bool enabled))
      UNEXPECTED_RUNTIME_CALL(void get_current_preset_path(char *path, size_t *path_size) const)
      UNEXPECTED_RUNTIME_CALL(void set_current_preset_path(const char *path))
      UNEXPECTED_RUNTIME_CALL(void reorder_techniques(size_t count, const effect_technique *techniques))
      UNEXPECTED_RUNTIME_CALL(void block_input_next_frame())
      UNEXPECTED_RUNTIME_CALL(uint32_t last_key_pressed() const)
      UNEXPECTED_RUNTIME_CALL(uint32_t last_key_released() const)
      UNEXPECTED_RUNTIME_CALL(void get_uniform_variable_effect_name(effect_uniform_variable variable, char *effect_name, size_t *effect_name_size) const)
      UNEXPECTED_RUNTIME_CALL(void get_texture_variable_effect_name(effect_texture_variable variable, char *effect_name, size_t *effect_name_size) const)
      UNEXPECTED_RUNTIME_CALL(void get_technique_effect_name(effect_technique technique, char *effect_name, size_t *effect_name_size) const)
      UNEXPECTED_RUNTIME_CALL(void save_current_preset() const)
      UNEXPECTED_RUNTIME_CALL(bool get_preprocessor_definition_for_effect(const char *effect_name, const char *name, char *value, size_t *value_size) const)
      UNEXPECTED_RUNTIME_CALL(void set_preprocessor_definition_for_effect(const char *effect_name, const char *name, const char *value))
      UNEXPECTED_RUNTIME_CALL(bool open_overlay(bool open, input_source source))
      UNEXPECTED_RUNTIME_CALL(void set_color_space(color_space color_space))
      UNEXPECTED_RUNTIME_CALL(void reset_uniform_value(effect_uniform_variable variable))
      UNEXPECTED_RUNTIME_CALL(void reload_effect_next_frame(const char *effect_name))
      UNEXPECTED_RUNTIME_CALL(void export_current_preset(const char *path) const)
      UNEXPECTED_RUNTIME_CALL(void save_screenshot(const char *postfix = nullptr))
#undef UNEXPECTED_RUNTIME_CALL
    };
  }

  struct publisher_tests {
    static void adaptive_ui_source_provenance() {
      sunshine_depth::frame_depth depth;
      depth.ready = true; depth.runtime_epoch = 3;
      depth.source_id = 100; depth.layout_epoch = 7; depth.frame_index = 50;
      scene_parameters_t scene;
      scene.ready = true; scene.generic_routing_epoch = 12;
      const auto a = resolve_ui_observation(depth, {}, scene, 17, 1000);
      require(a.eligible && a.epoch == 3 && a.source_id == 100 && a.revision == 7 &&
          a.sequence == 50 && a.tick_ms == 1000 && a.generic_basis_epoch == 17 && a.generic_routing_epoch == 12,
        "Generic UI observation lost capture provenance or logical placement scope");
      depth.source_id = 200; depth.layout_epoch = 11; depth.frame_index = 51;
      const auto b = resolve_ui_observation(depth, a, scene, 17, 1016);
      require(b.eligible && b.source_id != a.source_id && b.revision != a.revision && b.sequence == 51 &&
          b.tick_ms == 1016 && b.epoch == a.epoch && b.generic_basis_epoch == a.generic_basis_epoch &&
          b.generic_routing_epoch == a.generic_routing_epoch && sunshine_game3d::ui_adaptive::same_scope(a, b),
        "Physical buffer rotation changed the logical UI scope or erased member provenance");
      const auto repeated = resolve_ui_observation(depth, b, scene, 17, 1050);
      require(repeated.tick_ms == b.tick_ms && repeated.sequence == b.sequence,
        "Repeated access made the same generic depth capture fresh again");
      scene.generic_routing_epoch = 13;
      const auto rerouted = resolve_ui_observation(depth, b, scene, 17, 1066);
      require(rerouted.generic_routing_epoch == 13 && !sunshine_game3d::ui_adaptive::same_scope(b, rerouted),
        "A new preferred capture group retained the old UI routing identity");
      require(!resolve_ui_observation(depth, b, scene, 0, 1066).eligible,
        "Generic UI observation without a basis fell back to provider matching");
      scene.generic_routing_epoch = 0;
      require(!resolve_ui_observation(depth, b, scene, 17, 1066).eligible,
        "Generic UI observation without authoritative routing was admitted");
      scene.generic_routing_epoch = 12; scene.ready = false;
      require(!resolve_ui_observation(depth, b, scene, 17, 1066).eligible,
        "Uncalibrated generic depth became adaptive UI evidence");
      scene.ready = true;
      depth.provided.epoch = 9; depth.provided.observation_revision = 4;
      depth.provided.source_id = 1000; depth.provided.viewport = 2;
      depth.provided.sequence = 55; depth.provided.tick = 990;
      const auto provided = resolve_ui_observation(depth, b, scene, 17, 1066);
      require(provided.eligible && provided.epoch == 9 && provided.revision == 4 && provided.source_id == 1000 &&
          provided.viewport == 2 && provided.sequence == 55 && provided.tick_ms == 990 &&
          provided.generic_basis_epoch == 0 && provided.generic_routing_epoch == 0,
        "Provider logical identity was replaced by generic resource or roster metadata");
      depth.reused_depth = true;
      const auto held = resolve_ui_observation(depth, provided, scene, 19, 1100);
      require(held.eligible && held.epoch == provided.epoch && held.revision == provided.revision &&
          held.source_id == provided.source_id && held.sequence == provided.sequence && held.tick_ms == provided.tick_ms &&
          held.generic_basis_epoch == 0 && held.generic_routing_epoch == 0,
        "FG reuse fabricated new UI evidence or acquired a generic placement scope");
      depth.ready = false;
      require(!resolve_ui_observation(depth, held, scene, 19, 1116).eligible,
        "Unavailable depth remained eligible for adaptive UI observation");
    }

    static void retained_depth_decision() {
      using namespace sunshine_streamline;
      using action = depth_capture::display_action;
      using sunshine_scene_depth::provider_kind;
      depth_capture::retained_depth previous;
      previous.capture_id = 500; previous.real_present = 100;
      previous.width = 1920; previous.height = 1080; previous.format = 39;
      previous.area = {0, 0, 1920, 1080};
      auto &metadata = previous.metadata;
      metadata.provider = provider_kind::streamline;
      metadata.epoch = 8; metadata.viewport = 2; metadata.sequence = 30; metadata.tick = 1000;
      metadata.observation_revision = 4; metadata.feedback.revision = 5;
      // The next ordinary SR evaluation is still in flight in a pipelined game.
      depth_capture::acquisition_decision pending;
      pending.source_valid = pending.pending_frame = true;
      pending.provider = metadata.provider; pending.epoch = metadata.epoch;
      pending.source_id = metadata.source_id; pending.viewport = metadata.viewport;
      pending.sequence = 31; pending.capture_id = 501;
      const auto decide = [&](const depth_capture::retained_depth &before, const depth_capture::acquisition_decision &proof,
          const depth_capture::packet &candidate = {}, depth_capture::selection_policy policy = {}, std::uint64_t now = 1032) {
        return depth_capture::decide_display(proof, candidate, before, policy, now);
      };
      const auto held = [&](const depth_capture::retained_depth &before, const depth_capture::acquisition_decision &proof,
          const depth_capture::packet &candidate = {}, depth_capture::selection_policy policy = {}, std::uint64_t now = 1032) {
        const auto result = decide(before, proof, candidate, policy, now);
        return result.action == action::hold && std::strcmp(result.reason, "retained_copy") == 0;
      };
      require(held(previous, pending), "A pending SR frame discarded the newest finished depth");
      // Finished pixels outlive newer failures, gaps and metadata bookkeeping.
      auto failed = pending; failed.source_valid = failed.pending_frame = false;
      require(held(previous, failed) && held(previous, {}), "A failed or missing newer evaluation discarded finished depth");
      auto repeated = pending;
      repeated.sequence = metadata.sequence; repeated.capture_id = previous.capture_id;
      repeated.pending_frame = false; repeated.repeated_frame = true;
      require(held(previous, repeated), "A repeated frame discarded its own copy");
      auto reset = previous;
      reset.metadata.feedback.reset = true; ++reset.metadata.feedback.revision; ++reset.metadata.observation_revision;
      require(held(reset, pending), "A camera reset or observation bookkeeping discarded finished depth");
      auto ngx = previous; ngx.metadata.provider = provider_kind::ngx; ngx.metadata.source_id = 7;
      auto ngx_pending = pending; ngx_pending.provider = provider_kind::ngx; ngx_pending.source_id = 7;
      require(held(ngx, ngx_pending, {}, {}, 1100), "NGX was limited to a special one-present hold");
      auto fg = previous;
      fg.metadata.frame_generation_input = true; fg.metadata.source_id = (1ull << 63) | 2;
      auto fg_pending = pending; fg_pending.source_id = fg.metadata.source_id;
      const depth_capture::selection_policy fg_policy{true, metadata.epoch, metadata.viewport};
      require(held(fg, fg_pending, {}, fg_policy), "FG pending frame discarded its finished input depth");
      // A present with no FG nomination (for example an unusable tag lifetime) gets the FG
      // scope without a sequence from classify_acquisition. It contradicts nothing: hold.
      depth_capture::acquisition_decision fg_gap;
      fg_gap.epoch = fg_policy.epoch; fg_gap.viewport = fg_policy.viewport; fg_gap.source_id = fg.metadata.source_id;
      require(held(fg, fg_gap, {}, fg_policy), "A present without an FG nomination discarded finished FG input depth");
      auto other_scope = fg_gap; ++other_scope.source_id;
      require(!held(fg, other_scope, {}, fg_policy), "A different FG source without a sequence held old depth");
      require(!held(previous, pending, {}, fg_policy) &&
          std::strcmp(decide(previous, pending, {}, fg_policy).reason, "FG_scope_changed") == 0,
        "Ordinary depth stood in for required FG input");
      for (unsigned mismatch = 0; mismatch != 2; ++mismatch) {
        auto policy = fg_policy;
        if (mismatch == 0) ++policy.epoch; else ++policy.viewport;
        require(!held(fg, fg_pending, {}, policy), "A different FG scope held old depth");
      }

      require(decide({}, pending).action == action::invalidate &&
          std::strcmp(decide({}, pending).reason, "no_retained_depth") == 0,
        "An empty cache was revived without fresh pixels");
      for (unsigned field = 0; field != 6; ++field) {
        auto before = previous;
        auto current = pending;
        switch (field) {
          case 0: current.provider = provider_kind::ngx; break;
          case 1: ++current.epoch; break;
          case 2: ++current.source_id; break;
          case 3: ++current.viewport; break;
          case 4: current.sequence = metadata.sequence - 1; break;
          case 5: before.metadata.epoch = 0; break;
        }
        require(decide(before, current).action == action::invalidate, "A changed or unknown source inherited retained depth");
      }
      for (const auto now : {999ull, 1250ull}) {
        const auto expired = decide(previous, pending, {}, {}, now);
        require(expired.action == action::invalidate && std::strcmp(expired.reason, "retained_depth_expired") == 0,
          "Hold extended the original capture's age or accepted a backwards clock");
      }
      require(held(previous, pending, {}, {}, 1249), "Hold ended before the source age bound");

      depth_capture::packet candidate;
      static_cast<sunshine_scene_depth::frame &>(candidate.metadata) = metadata;
      candidate.metadata.sequence = pending.sequence;
      candidate.width = previous.width; candidate.height = previous.height; candidate.format = previous.format;
      candidate.area = previous.area; candidate.capture_id = pending.capture_id;
      auto selected = pending; selected.source_selected = true;
      require(held(previous, selected, candidate), "A selected unready snapshot could not retain finished private pixels");
      auto changed_encoding = candidate;
      changed_encoding.metadata.projection.encoding = sunshine_scene_depth::depth_encoding::linear_distance;
      require(std::strcmp(decide(previous, selected, changed_encoding).reason, "depth_encoding_changed") == 0,
        "Retained pixels crossed a changed depth encoding");
      for (unsigned field = 0; field != 7; ++field) {
        auto next = candidate;
        switch (field) {
          case 0: ++next.width; break;
          case 1: ++next.height; break;
          case 2: ++next.format; break;
          case 3: ++next.area.left; break;
          case 4: ++next.area.top; break;
          case 5: --next.area.width; break;
          case 6: --next.area.height; break;
        }
        const auto rejected = decide(previous, selected, next);
        require(rejected.action == action::invalidate && std::strcmp(rejected.reason, "depth_shape_changed") == 0,
          "A changed selected layout inherited old private pixels");
      }
      candidate.pixel_ready = true;
      require(decide({}, selected, candidate).action == action::copy_fresh &&
          decide(previous, selected, candidate).action == action::copy_fresh,
        "Fresh pixels did not take precedence over a held or absent copy");
      candidate.metadata.feedback.reset = true; ++candidate.metadata.observation_revision;
      require(decide(previous, selected, candidate).action == action::copy_fresh,
        "A valid reset frame or later observation could not establish a fresh private copy");
      candidate.pixel_ready = false; candidate.shared_preservation = true;
      require(decide({}, selected, candidate).action == action::copy_fresh,
        "Shared preservation lost its separate fresh-copy authorization");

      provider::frame_generation_policy source_policy;
      frame_generation_snapshot fg_snapshot;
      fg_snapshot.known = fg_snapshot.enabled = true; fg_snapshot.epoch = metadata.epoch;
      fg_snapshot.viewport = metadata.viewport; fg_snapshot.generated_frames = 1;
      source_policy.update(frame_generation_query_status::observed, fg_snapshot);
      const auto excluded = source_policy.update(frame_generation_query_status::ambiguous, {});
      require(!excluded.require_frame_generation && excluded.exclude_unconfirmed_fg && !excluded.epoch,
        "Ambiguous FG kept a mandatory missing scope or failed to revoke FG input authority");
      require(!held(fg, fg_pending, {}, excluded), "Ambiguous FG inherited a previous FG display copy");
      auto fg_candidate = candidate;
      fg_candidate.shared_preservation = false; fg_candidate.pixel_ready = true;
      fg_candidate.metadata.frame_generation_input = true; fg_candidate.metadata.source_id = fg.metadata.source_id;
      auto fg_selected = fg_pending; fg_selected.source_selected = true;
      require(decide(fg, fg_selected, fg_candidate, excluded).action == action::invalidate &&
          decide({}, fg_selected, fg_candidate, excluded).action == action::invalidate,
        "Ready old-scope FG pixels bypassed explicit FG exclusion");

      depth_capture::packet fallback;
      fallback.capture_id = 900; fallback.pixel_ready = true;
      fallback.width = 1280; fallback.height = 720; fallback.format = 41; fallback.area = {0, 0, 1280, 720};
      fallback.metadata.provider = provider_kind::ngx;
      fallback.metadata.epoch = 17; fallback.metadata.source_id = 99;
      fallback.metadata.sequence = 70; fallback.metadata.tick = 1032;
      fallback.metadata.projection.encoding = sunshine_scene_depth::depth_encoding::linear_distance;
      fallback.metadata.projection.direction_supplied = true;
      depth_capture::acquisition_decision fresh;
      fresh.source_selected = fresh.source_valid = true; fresh.capture_id = fallback.capture_id;
      fresh.provider = fallback.metadata.provider; fresh.epoch = fallback.metadata.epoch;
      fresh.source_id = fallback.metadata.source_id; fresh.sequence = fallback.metadata.sequence;
      fresh.viewport = fallback.metadata.viewport;
      require(decide(fg, fresh, fallback, excluded).action == action::copy_fresh &&
          decide({}, fresh, fallback, excluded).action == action::copy_fresh,
        "Revoked SL FG scope blocked an independently valid fresh NGX linear-depth copy");
      const auto busy_fallback = source_policy.update(frame_generation_query_status::busy, {});
      require(decide(fg, fresh, fallback, busy_fallback).action == action::copy_fresh,
        "Busy query disabled independently valid fallback after FG revocation");
      fallback.pixel_ready = false;
      require(decide(fg, fresh, fallback, excluded).action == action::invalidate &&
          decide(previous, fresh, fallback).action == action::invalidate,
        "Pending NGX fallback inherited another source's pixels or encoding");
    }

    static void copied_depth_provenance() {
      using namespace sunshine_streamline;
      depth_capture::packet packet;
      packet.metadata.provider = sunshine_scene_depth::provider_kind::ngx;
      packet.metadata.resource.native = 100;
      packet.metadata.source_id = 9;
      packet.metadata.sequence = 10472;
      packet.metadata.epoch = 4;
      packet.metadata.projection.direction_supplied = packet.metadata.projection.reversed = true;
      packet.width = 2228; packet.height = 1256;
      packet.area = {0, 0, 2228, 1253};
      packet.queue = 200;
      packet.resource_id = 7;
      packet.shared_preservation = true;
      sunshine_depth::frame_depth captured;
      captured.resource = {300};
      captured.source_id = 7;
      captured.backup_id = 231;
      captured.frame_index = 11190;
      captured.capture_marker = {400, 8, 9, 10, 11, 0};
      captured.depth_copy.state = content::status::valid_observation;
      captured.depth_copy.copy_id = 541;
      captured.depth_copy.source.resource = {100, 7};
      captured.depth_copy.source.version = 812;
      captured.depth_copy.backup = {300, 231};
      captured.depth_copy.recording = captured.capture_marker;
      sunshine_depth::frame_depth out;
      provider::test_describe_copied_depth(out, packet, captured, {500}, {501}, 11193);
      require(out.ready && out.resource.handle == 500 && out.shader_resource.handle == 501 &&
          out.source_resource.handle == 100 && out.provided.sequence == 10472 && out.frame_index == 11193 &&
          out.source_id == 9 && out.width == 2228 && out.height == 1256 && out.active_height == 1253 &&
          out.command_queue == 200 && out.orientation == sunshine_depth::depth_orientation::reversed,
        "Preservation diagnostics replaced the renderer's current display or SDK metadata");
      require(out.backup_id == 231 && out.capture_marker.command == 400 && out.capture_marker.event == 10 &&
          out.depth_copy.valid() && out.depth_copy.copy_id == 541 &&
          out.depth_copy.source.resource.lifetime == 7 && out.depth_copy.source.version == 812 &&
          out.depth_copy.backup.native == 300 && out.depth_copy.backup.lifetime == 231 &&
          content::same_recording(out.depth_copy.recording, captured.capture_marker),
        "SDK display publication discarded the preserved depth's copy/content/recording identity");
      const auto selected = sunshine_depth::camera_selection_snapshot(out);
      require(selected.capture_marker.command == 400 && selected.depth_copy.copy_id == 541 && selected.backup_id == 231,
        "Renderer diagnostic selection discarded the consumed preservation provenance");

      // Independent native snapshots do not supply ReShade preservation
      // metadata. An already populated output must not retain the prior copy.
      packet.shared_preservation = false;
      ++packet.metadata.sequence;
      provider::test_describe_copied_depth(out, packet, {}, {500}, {501}, 11194);
      require(out.ready && out.provided.sequence == 10473 && !out.backup_id && !out.capture_marker &&
          out.depth_copy.state == content::status::unavailable && !out.depth_copy.copy_id &&
          !out.depth_copy.source.version && !out.depth_copy.backup.lifetime,
        "An unobserved native snapshot inherited or invented preservation provenance");
    }

    static void jitter_publication() {
      struct runtime : runtime_fixture::empty_runtime {
        std::array<float, 2> offset{};
        void set_uniform_value_float(api::effect_uniform_variable uniform, const float *values,
            size_t count, size_t) override {
          if (uniform.handle == 101 && count == 2) std::copy_n(values, 2, offset.begin());
          else require(false, "Jitter publication touched an unrelated uniform");
        }
      } runtime;
      runtime_t proof;
      proof.depth_jitter = {101};
      frame_decision_t frame;
      frame.jitter = sunshine_depth_jitter::make({.5f, -.25f, 1920, 1080, true},
        2048, 1152, {32, 16, 1920, 1080});
      publisher_t::publish_frame(&runtime, proof, frame);
      require(runtime.offset == frame.jitter.offset,
        "Captured-depth jitter was not published");
      frame.reused_depth = true;
      publisher_t::publish_frame(&runtime, proof, frame);
      require(runtime.offset == frame.jitter.offset, "FG reuse did not retain the captured-depth offset");
      publisher_t::publish_frame(&runtime, proof, {});
      require(runtime.offset == std::array<float, 2>{}, "Missing/current generic depth retained previous API jitter");
    }

    static void pending_scene_identity() {
      frame_decision_t previous;
      previous.depth_ready = previous.scene.ready = previous.scene.owned = true;
      previous.scene_source = {4, 7, 100, 0};
      previous.scene.basis = 2;
      previous.scene.scale = 300.f;
      previous.scene.zero = {.01f, .25f};
      previous.scene.ui_plane = {sunshine_game3d::ui_plane_mode::depth_midpoint, .375f};
      previous.scene.raw_range = {.001f, .08f};
      previous.scene.blend = .75f;
      previous.scene.admitted_strength = 40.f;
      for (bool generated : {false, true}) {
        frame_decision_t frame;
        frame.depth_ready = frame.reused_depth = true;
        frame.diagnostics.frame_generation_active = generated;
        frame.scene_source = previous.scene_source;
        restore_reused_scene(frame, previous, true);
        require(frame.depth_ready && frame.scene.ready && frame.scene.scale == previous.scene.scale &&
            frame.scene.zero == previous.scene.zero && frame.scene.raw_range == previous.scene.raw_range &&
            frame.scene.ui_plane.mode == previous.scene.ui_plane.mode &&
            frame.scene.ui_plane.inverse_depth == previous.scene.ui_plane.inverse_depth &&
            frame.scene.blend == previous.scene.blend && frame.scene.admitted_strength == previous.scene.admitted_strength,
          "Known pending NGX/FG depth did not retain its exact previously admitted scene");
        for (unsigned field = 0; field != frame.scene_source.size(); ++field) {
          auto changed = frame;
          changed.scene = {};
          ++changed.scene_source[field];
          restore_reused_scene(changed, previous, true);
          require(!changed.depth_ready && !changed.scene.ready &&
              changed.scene.ui_plane.mode == sunshine_game3d::ui_plane_mode::display_fraction &&
              changed.scene.ui_plane.inverse_depth == 0.f &&
              changed.scene.ui.phase == sunshine_game3d::automatic_phase::waiting_for_depth,
            "Held depth inherited geometry from a different epoch/source/capture/viewport");
        }
        auto missing = previous;
        missing.depth_ready = false;
        frame.scene = {};
        restore_reused_scene(frame, missing, true);
        require(!frame.depth_ready && !frame.scene.ready && frame.gap == depth_gap::reuse_after_gap,
          "A missing previous pair resurrected cached geometry or lost its reason");
        auto other = frame;
        other.depth_ready = true; other.gap = depth_gap::none; other.scene = {};
        ++other.scene_source[1];
        restore_reused_scene(other, previous, true);
        require(!other.depth_ready && other.gap == depth_gap::reuse_other_source,
          "Reuse refused for another source was not named");
        // The real frame still calibrating its range shows its depth; its generated frames
        // keep that depth and the same not-yet-placed scene instead of turning flat.
        auto calibrating = previous;
        calibrating.scene.ready = false;
        calibrating.scene.ui = {sunshine_game3d::automatic_phase::waiting_for_depth, true};
        frame = {};
        frame.depth_ready = frame.reused_depth = true;
        frame.diagnostics.frame_generation_active = generated;
        frame.scene_source = previous.scene_source;
        restore_reused_scene(frame, calibrating, true);
        require(frame.depth_ready && !frame.scene.ready && frame.scene.scale == calibrating.scene.scale &&
            frame.scene.ui.phase == calibrating.scene.ui.phase,
          "A generated frame dropped depth that its still-calibrating real frame shows");
      }
    }

    static void camera_missing_holds_projection_scene() {
      frame_decision_t previous;
      previous.depth_ready = previous.scene.ready = previous.scene.owned = true;
      previous.scene_source = {4, 7, 100, 9};
      previous.scene.scale = 12.f; previous.scene.blend = .6f;
      previous.scene.projection = {-2.9e-5f, 5.f};
      previous.scene.projection_ms = 1000;
      sunshine_depth::frame_depth depth;
      depth.ready = true;
      frame_decision_t frame;
      frame.depth_ready = true;
      frame.scene_source = {4, 7, 101, 9}; // A fresh capture of the same source.
      require(hold_projection_scene(frame, previous, depth, 1100) && frame.scene.scale == 12.f &&
          frame.scene.blend == .6f && frame.scene.projection == previous.scene.projection && frame.scene.projection_ms == 1000,
        "A capture without its own camera did not keep the projection scene");
      const auto refused = [&](auto change, std::uint64_t now, const char *message) {
        auto f = frame; auto p = previous; auto d = depth;
        f.scene = {};
        change(f, p, d);
        require(!hold_projection_scene(f, p, d, now) && !f.scene.ready, message);
      };
      refused([](auto &, auto &, auto &d) { d.projection.supplied = true; }, 1100, "A capture with its own camera was held");
      refused([](auto &, auto &, auto &d) { d.projection.encoding = sunshine_scene_depth::depth_encoding::linear_distance; }, 1100,
        "Linear depth needs no camera but was held");
      refused([](auto &, auto &, auto &) {}, 1000 + sunshine_scene_depth::maximum_source_age_ms + 1,
        "A projection scene older than the source age was held");
      refused([](auto &, auto &p, auto &) { p.scene.projection_ms = 0; }, 1100, "A raw-placed scene was held");
      refused([](auto &, auto &p, auto &) { p.depth_ready = false; }, 1100, "A scene was held across a depth gap");
      refused([](auto &f, auto &, auto &) { ++f.scene_source[1]; }, 1100, "Another source's scene was held");
      refused([](auto &f, auto &, auto &) { ++f.scene_source[3]; }, 1100, "Another viewport's scene was held");
      refused([](auto &f, auto &, auto &) { ++f.scene_source[0]; }, 1100, "Another epoch's scene was held");
    }

    static void cached_scene_strength_publication() {
      struct runtime : runtime_fixture::empty_runtime {
        float scale{}, blend{};
        bool ready{};
        int basis{};
        std::array<float, 2> projection{}, zero{};
        void set_uniform_value_bool(api::effect_uniform_variable variable, const bool *values,
            size_t count, size_t) override {
          require(variable.handle == 101 && count == 4, "Strength publication touched an unrelated bool uniform");
          ready = values[0];
        }
        void set_uniform_value_int(api::effect_uniform_variable variable, const int *values,
            size_t count, size_t) override {
          require(variable.handle == 102 && count == 4, "Strength publication touched an unrelated int uniform");
          basis = values[0];
        }
        void set_uniform_value_float(api::effect_uniform_variable variable, const float *values,
            size_t count, size_t) override {
          if (variable.handle == 103 && count == 2) std::copy_n(values, 2, projection.begin());
          else if (variable.handle == 104 && count == 4) scale = values[0];
          else if (variable.handle == 105 && count == 2) std::copy_n(values, 2, zero.begin());
          else if (variable.handle == 106 && count == 4) blend = values[0];
          else require(false, "Strength publication touched an unrelated float uniform");
        }
      } runtime;
      runtime_t proof;
      proof.camera_ready = {101}; proof.camera_basis = {102}; proof.camera_projection = {103};
      proof.camera_scale = {104}; proof.camera_zero = {105}; proof.camera_blend = {106};
      proof.frame_strength = 50.f;
      frame_decision_t fresh;
      fresh.depth_ready = fresh.diagnostics.frame_generation_active = true;
      fresh.scene.owned = fresh.scene.ready = true;
      fresh.scene.basis = 1;
      fresh.scene.scale = 4.f;
      fresh.scene.projection = {0.f, 1.f};
      fresh.scene.zero = {.05f, .25f};
      fresh.scene.blend = 1.f;
      fresh.scene.admitted_strength = 50.f;
      publisher_t::publish_frame(&runtime, proof, fresh);
      require(runtime.ready && runtime.basis == 1 && runtime.scale == 4.f,
        "Fresh reference publication changed its admitted geometry");
      for (const float strength : {100.f, 100.f, 25.f, 0.f, 75.f, 0.f, 100.f, 50.f}) {
        auto reused = proof.frame;
        reused.reused_depth = true;
        proof.frame_strength = strength;
        publisher_t::publish_frame(&runtime, proof, reused);
        const float effective = runtime.scale * runtime.blend * strength;
        require(std::isfinite(runtime.scale) && std::abs(effective - 4.f * std::min(strength, 50.f)) < 1e-5f,
          "Cached FG scene exceeded its admitted strength or delayed an immediate decrease");
        require(runtime.zero == fresh.scene.zero && runtime.projection == fresh.scene.projection && runtime.scale == fresh.scene.scale,
          "FG strength cap changed convergence, normalized depth or its diagnostic geometry");
        require(proof.frame.scene.scale == fresh.scene.scale && proof.frame.scene.admitted_strength == 50.f,
          "FX strength publication modified its cached scene and compounded repeated attenuation");
      }
      // A fresh real-depth decision authorizes the new strength independently;
      // the previous cached cap cannot remain latched after that decision.
      auto resumed = proof.frame;
      resumed.reused_depth = false;
      resumed.scene.admitted_strength = proof.frame_strength = 100.f;
      resumed.scene.scale = 3.f;
      publisher_t::publish_frame(&runtime, proof, resumed);
      require(runtime.scale == 3.f && runtime.blend == 1.f && proof.frame.scene.admitted_strength == 100.f,
        "Fresh real-depth scene retained the previous FG strength cap");
      resumed.scene.admitted_strength = 0.f;
      resumed.reused_depth = true;
      publisher_t::publish_frame(&runtime, proof, resumed);
      require(runtime.scale == 3.f && runtime.blend == 0.f,
        "Scene admitted at zero strength authorized nonzero FG disparity");
    }

    static void automatic_raw_and_matrix_resolver(bool native_cropped = false) {
      runtime_fixture::empty_runtime runtime;
      publisher_t publisher;
      runtime_t proof;
      proof.addon_native = native_cropped;
      proof.frame_strength = 37.f;
      proof.width = 1920; proof.height = 1080;
      proof.raw_supported = true;
      proof.raw_basis_epoch = 77;
      require(proof.provided_raw_policy.reset(proof.raw_basis_epoch, 1000),
        "NGX resolver fixture could not initialize its raw reference");
      sunshine_depth::frame_depth depth;
      depth.ready = true;
      if (native_cropped) {
        // DLSS may pad an allocation while tagging a slightly shorter active
        // extent. Native constants must carry this rect without any FX handle.
        depth.width = depth.active_width = 2228;
        depth.height = 1256;
        depth.active_height = 1253;
      }
      depth.provided.provider = sunshine_scene_depth::provider_kind::ngx;
      depth.provided.epoch = 3;
      depth.provided.source_id = 9;
      depth.provided.sequence = 1;
      depth.provided.projection.direction_supplied = depth.provided.projection.reversed = true;
      const std::array<float, 4> active_rect{0.f, 0.f, 1.f, 1253.f / 1256.f};
      const auto check_native_crop = [&](const scene_parameters_t &scene, std::uint64_t now) {
        if (!native_cropped) return;
        require(!proof.camera_rect.handle && scene.rect == active_rect,
          "Native scene ignored the active depth rect because no FX camera-rect uniform exists");
        frame_decision_t frame;
        frame.depth_ready = depth.ready;
        frame.scene = scene;
        // The empty runtime rejects all uniform calls. Native publication must
        // retain the normalized crop and current status without FX reflection.
        publisher_t::publish_frame(&runtime, proof, frame);
        require(proof.frame.prepared && proof.frame.scene.rect == active_rect &&
            sunshine_game3d::query_automatic(&runtime).phase == scene.ui.phase,
          "Native publication dropped the resolved depth crop or calibration status");
        proof.addon_native = false;
        const auto unsupported_fx = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, now);
        require(!unsupported_fx.ready && unsupported_fx.ui.phase == sunshine_game3d::automatic_phase::unavailable,
          "Legacy FX without camera-rect support accepted padded scene depth");
        proof.camera_rect = {991};
        const auto supported_fx = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, now);
        require(supported_fx.ready == (scene.basis != 2) && supported_fx.rect == active_rect,
          "Legacy FX crop support changed affine readiness or admitted unsupported linear depth");
        proof.camera_rect = {};
        proof.addon_native = true;
      };
      {
        const auto pending = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 1000);
        require(!pending.ready && (!native_cropped || pending.rect == active_rect) &&
            pending.ui_plane.mode == sunshine_game3d::ui_plane_mode::display_fraction &&
            pending.ui.phase == sunshine_game3d::automatic_phase::calibrating &&
            pending.ui.scale.basis == sunshine_game3d::automatic_scale_basis::relative_depth &&
            !pending.ui.scale.depth_statistics_valid(),
          "NGX depth did not enter raw calibration with the current render configuration");
      }
      std::array<float, sunshine_raw_scene::grid_width * sunshine_raw_scene::grid_height> raw;
      const auto measure = [&](float value, std::uint64_t now) {
        for (size_t i = 0; i < raw.size(); ++i) raw[i] = value * (i % 2 ? 1.5f : .5f);
        ++depth.provided.sequence;
        depth.provided.tick = now;
        const auto current = sunshine_provided_raw::selected(depth.provided, proof.raw_basis_epoch, true);
        auto sample = sunshine_provided_raw::measured(depth.provided, depth.provided.sequence,
          proof.raw_basis_epoch, raw);
        sample.range_supplied = sample.range_valid = true;
        sample.range_min = value * .5f; sample.range_max = value * 1.5f;
        sample.moments.supplied = sample.moments.valid = true;
        sample.moments.count = native_cropped ? 2228u * 1253u : 1920u * 1080u;
        sample.moments.sum = double(sample.moments.count) * value;
        sample.moments.sum_squares = double(sample.moments.count) * value * value * 1.25;
        sample.moments.tiles_x = 32; sample.moments.tiles_y = 18;
        proof.provided_raw_policy.update(current, &sample, now);
      };
      for (unsigned i = 0; i != 4; ++i) measure(.25f, 1000 + i * 250);
      const auto scene = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 1750);
      const double initial_L = sunshine_scene_gain::render_limits(37.f, 1920, 1080).normalization;
      require(scene.ready && scene.basis == 1 && std::abs(scene.scale - initial_L / .375) < 1e-5 && std::isfinite(scene.zero[1]) &&
          scene.admitted_strength == 37.f,
        "NGX resolver did not retain ready relative stereo");
      require(scene.ui.scale.depth_statistics_valid() && scene.ui.scale.reference_inverse == .375 &&
          scene.ui_plane.mode == sunshine_game3d::ui_plane_mode::display_fraction && scene.ui_plane.inverse_depth == 0.f &&
          scene.ui.scale.has_ui_midpoint && scene.ui.scale.ui_midpoint_inverse == .25f &&
          scene.ui.scale.has_ui_midpoint_target() && scene.ui.scale.target_ui_midpoint_inverse == .25 &&
          scene.ui.scale.mean_inverse == .25 && scene.ui.scale.normalization == initial_L &&
          scene.ui.scale.minimum_inverse == .125 && scene.ui.scale.maximum_inverse == .375 &&
          scene.ui.scale.mean_square_inverse == .078125 && scene.ui.scale.depth_pixel_count > 0 &&
          scene.ui.scale.depth_tiles_x == 32 && scene.ui.scale.depth_tiles_y == 18,
        "NGX resolver lost accepted oriented depth statistics");
      check_native_crop(scene, 1750);
      measure(.125f, 2000);
      proof.frame_strength = 83.f;
      const auto tracking = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 2000);
      require(tracking.ready && tracking.scale > scene.scale && tracking.scale <= scene.scale * std::exp2(.25) * (1. + 2e-7) &&
          tracking.admitted_strength == 83.f,
        "NGX resolver stopped bounded live range learning or failed to admit current strength");
      require(tracking.ui.scale.depth_statistics_valid() && tracking.ui.scale.reference_inverse == .1875 &&
          tracking.ui.scale.mean_inverse == .125 && tracking.ui.scale.normalization == initial_L &&
          tracking.ui.scale.minimum_inverse == .0625 && tracking.ui.scale.maximum_inverse == .1875,
        "NGX resolver derived measured statistics from smoothed scale");

      proof.width = 1080; proof.height = 1920;
      const auto portrait = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 2000);
      const double portrait_L = sunshine_scene_gain::render_limits(83.f, 1080, 1920).normalization;
      require(portrait.ready && portrait.ui.scale.depth_statistics_valid() &&
          portrait.ui.scale.reference_inverse == tracking.ui.scale.reference_inverse &&
          portrait.ui.scale.mean_inverse == tracking.ui.scale.mean_inverse &&
          portrait.ui.scale.normalization == portrait_L && portrait_L != initial_L &&
          std::abs(portrait.ui.scale.target_value - portrait_L / .1875) < 1e-5 &&
          portrait.scale == tracking.scale && portrait.zero == tracking.zero,
        "Output aspect change altered captured depth or zero instead of updating only normalization and the gain target");
      proof.width = 1920; proof.height = 1080;
      proof.frame_strength = 0.f;
      const auto zero_strength_resize = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 2000);
      require(zero_strength_resize.ready && zero_strength_resize.admitted_strength == 0.f &&
          zero_strength_resize.ui.scale.depth_statistics_valid() && zero_strength_resize.ui.scale.has_target() &&
          zero_strength_resize.ui.scale.reference_inverse == portrait.ui.scale.reference_inverse &&
          zero_strength_resize.ui.scale.mean_inverse == portrait.ui.scale.mean_inverse &&
          zero_strength_resize.ui.scale.normalization == initial_L &&
          std::abs(zero_strength_resize.ui.scale.target_value - initial_L / .1875) < 1e-5 &&
          zero_strength_resize.ui.scale.target_value != portrait.ui.scale.target_value &&
          zero_strength_resize.scale == portrait.scale && zero_strength_resize.zero == portrait.zero,
        "Zero-strength resize kept a stale normalization target or moved the held gain and zero");
      proof.frame_strength = 83.f;

      depth.projection.supplied = true;
      depth.projection.epoch = 3;
      depth.projection.A = -.01;
      depth.projection.B = .2;
      proof.projection_domain = {3, 0};
      proof.projection_policy.reset(proof.projection_domain, 2000);
      const auto coefficients = sunshine_projection_depth::make(depth.projection.A, depth.projection.B);
      {
        const auto pending = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 2000);
        require(!pending.ready && (!native_cropped || pending.rect == active_rect) &&
            pending.ui.phase == sunshine_game3d::automatic_phase::calibrating &&
            pending.ui.scale.basis == sunshine_game3d::automatic_scale_basis::camera_matrix &&
            !pending.ui.scale.depth_statistics_valid(),
          "Camera depth did not enter projection calibration with the current render configuration");
      }
      for (unsigned i = 0; i != 4; ++i) {
        sunshine_projection_depth::sample sample;
        sample.id = i + 1;
        sample.capture_ms = 2001 + i * 250;
        sample.logical_domain = proof.projection_domain;
        sample.projection = coefficients;
        for (size_t j = 0; j < sample.raw.size(); ++j) sample.raw[j] = j % 2 ? .375f : .125f;
        sample.range_supplied = sample.range_valid = true;
        sample.range_min = .125f; sample.range_max = .375f;
        const float far_value = (.125f-coefficients.shader_A)*coefficients.inverseB;
        const float near_value = (.375f-coefficients.shader_A)*coefficients.inverseB;
        sample.moments.supplied = sample.moments.valid = true;
        sample.moments.A = coefficients.shader_A; sample.moments.inverseB = coefficients.inverseB;
        sample.moments.count = native_cropped ? 2228u * 1253u : 1920u * 1080u;
        sample.moments.sum = double(sample.moments.count) * (double(far_value)+near_value) * .5;
        sample.moments.sum_squares = double(sample.moments.count) * (double(far_value)*far_value+double(near_value)*near_value) * .5;
        sample.moments.tiles_x = 32; sample.moments.tiles_y = 18;
        proof.projection_policy.observe(sample, sample.capture_ms);
      }
      proof.frame_strength = 71.f;
      const auto matrix = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 2751);
      const float expected_far = (.125f - coefficients.shader_A) * coefficients.inverseB;
      const float expected_near = (.375f - coefficients.shader_A) * coefficients.inverseB;
      require(matrix.ready && matrix.basis == 0 && matrix.projection[0] == coefficients.shader_A &&
          matrix.projection[1] == coefficients.inverseB && std::isfinite(matrix.zero[1]) &&
          std::abs(matrix.scale - initial_L / expected_near) < 2e-5 && matrix.admitted_strength == 71.f,
        "Associated matrix did not take precedence over the relative-depth reference");
      require(matrix.ui.scale.depth_statistics_valid() && matrix.ui.scale.minimum_inverse == expected_far &&
          matrix.ui_plane.mode == sunshine_game3d::ui_plane_mode::display_fraction && matrix.ui_plane.inverse_depth == 0.f &&
          matrix.ui.scale.has_ui_midpoint && matrix.ui.scale.ui_midpoint_inverse == float((double(expected_far) + expected_near) * .5) &&
          matrix.ui.scale.has_ui_midpoint_target() &&
          matrix.ui.scale.target_ui_midpoint_inverse == (double(expected_far) + expected_near) * .5 &&
          matrix.ui.scale.maximum_inverse == expected_near &&
          matrix.ui.scale.reference_inverse == expected_near &&
          matrix.ui.scale.mean_inverse == (double(expected_far) + expected_near) * .5 &&
          matrix.ui.scale.normalization == initial_L &&
          matrix.ui.scale.depth_pixel_count > 0 && matrix.ui.scale.depth_tiles_x == 32 &&
          matrix.ui.scale.mean_square_inverse > 0.,
        "Camera resolver did not publish captured projection statistics in zero-plane units");
      check_native_crop(matrix, 2751);
      // Explicit linear distance routes independently of camera evidence. Old
      // FX has no reciprocal basis and cannot silently treat it as raw depth.
      depth.projection.supplied = false;
      depth.projection.encoding = sunshine_scene_depth::depth_encoding::linear_distance;
      depth.projection.A = depth.projection.B = NAN;
      auto linear = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 3000);
      require(!linear.ready && linear.basis == 2 &&
          linear.ui.scale.basis == sunshine_game3d::automatic_scale_basis::linear_distance &&
          linear.ui.phase == (native_cropped ? sunshine_game3d::automatic_phase::calibrating : sunshine_game3d::automatic_phase::suspended),
        "Linear input required a fake camera or entered the affine raw fallback");
      if (native_cropped) {
        const auto logical_domain = proof.projection_domain;
        for (unsigned i = 0; i != 4; ++i) {
          sunshine_projection_depth::sample sample;
          sample.id = i + 1; sample.capture_ms = 3001 + i * 250;
          sample.logical_domain = logical_domain;
          sample.projection = sunshine_projection_depth::make_linear();
          sample.range_supplied = sample.range_valid = true;
          sample.range_min = 2.f; sample.range_max = 8.f;
          auto &m = sample.moments;
          m.supplied = m.valid = true; m.encoding = depth.projection.encoding;
          m.A = 0.f; m.inverseB = 1.f; m.count = 2228u * 1253u;
          m.sum = double(m.count) * .3125;
          m.sum_squares = double(m.count) * .1328125;
          m.tiles_x = 32; m.tiles_y = 18;
          proof.projection_policy.observe(sample, sample.capture_ms);
        }
        linear = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 3751);
        require(linear.ready && linear.basis == 2 && linear.projection == std::array<float, 2>{0.f, 1.f} &&
            linear.ui.scale.minimum_inverse == .125 && linear.ui.scale.maximum_inverse == .5 &&
            !linear.ui.scale.has_projection_conversion,
          "Linear depth did not publish calibrated reciprocal geometry without camera matrices");
        check_native_crop(linear, 3751);
        ++depth.provided.source_id;
        const auto handoff = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 4000);
        require(!handoff.ready && !(proof.projection_domain == logical_domain),
          "New feature generation inherited another source's inverse-distance calibration");
      }
    }

    static void session_teardown_ownership() {
      using phase = sunshine_game3d::automatic_phase;
      auto &session = addon_session();
      const auto runtime = owner();
      unsigned cleanup_calls = 0, reentrant_calls = 0;
      for (unsigned run = 0; run != 2; ++run) {
        session.publisher = std::make_unique<publisher_t>();
        require(session.publisher->init(), "Reinitialized session could not create its own publisher mapping");
        require(session.begin(), "Clean session rejected initialization");
        publish_automatic_ui(runtime, {phase::ready, true});
        const auto cleanup = [&] {
          ++cleanup_calls;
          require(!session.publisher && session.lifecycle.load() == addon_session_t::phase::stopping,
            "Session teardown retained a competing public owner");
          require(!session.end([&] { ++reentrant_calls; }), "Reentrant session teardown claimed ownership twice");
          require(!session.begin(), "Session accepted reinitialization before old resources finished disposal");
        };
        require(session.end(cleanup) && !session.end(cleanup), "Repeated session teardown was not idempotent");
        require(!session.publisher && session.lifecycle.load() == addon_session_t::phase::inactive &&
            sunshine_game3d::query_automatic(runtime).phase == phase::unavailable,
          "Explicit session teardown failed to release publisher and UI state");
      }
      require(cleanup_calls == 2 && reentrant_calls == 0, "Session cleanup count did not match initialized sessions");
    }
    static inline unsigned guarded_calls{};
    static bool guarded_operation(unsigned &value) { ++guarded_calls; ++value; return true; }
    static void process_detach_lifetime() {
      unsigned value = 0;
      require(sunshine_addon_lifetime::guarded<guarded_operation>(value) && value == 1 && guarded_calls == 1,
        "Live callback guard changed forwarding, reference arguments or return value");
      using phase = sunshine_game3d::automatic_phase;
      const auto runtime = owner();
      publish_automatic_ui(runtime, {phase::ready, true});
      ::publisher = std::make_unique<publisher_t>();
      auto *before = ::publisher.get();
      DllMain(nullptr, DLL_PROCESS_DETACH, reinterpret_cast<void *>(1));
      require(sunshine_addon_lifetime::stopping(), "DLL detach did not fence callbacks before C++ cleanup");
      require(!sunshine_addon_lifetime::guarded<guarded_operation>(value) && value == 1 && guarded_calls == 1,
        "Detached callback entered destroyed owner or cancelled a host operation");
      // These poison addresses must never be inspected by a fenced callback.
      sunshine_addon_lifetime::guarded<on_destroy>(reinterpret_cast<api::effect_runtime *>(1));
      AddonUninit(nullptr, nullptr);
      require(::publisher.get() == before, "Late ReShade AddonUninit destroyed the publisher twice");
      ::publisher.reset();
      sunshine_addon_lifetime::detaching.store(false, std::memory_order_release); // Isolated fixture only.
      require(sunshine_game3d::query_automatic(runtime).phase == phase::ready,
        "Process-detach publisher destructor accessed UI owners that may already be destroyed");
      { publisher_t normal_unload; }
      require(sunshine_game3d::query_automatic(runtime).phase == phase::unavailable,
        "Ordinary publisher unload stopped cleaning its live UI owners");
    }
    static void automatic_ui_lifetime() {
      using phase = sunshine_game3d::automatic_phase;
      const auto runtime = owner();
      clear_automatic_ui(runtime);
      require(sunshine_game3d::query_automatic(runtime).phase == phase::unavailable,
        "New runtime inherited automatic UI state");
      require(!sunshine_game3d::recalibrate_automatic(runtime), "Unavailable runtime accepted recalibration");
      publish_automatic_ui(runtime, {phase::ready, true});
      require(sunshine_game3d::recalibrate_automatic(runtime), "Ready runtime rejected recalibration");
      publish_automatic_ui(runtime, {phase::ready, true});
      require(sunshine_game3d::query_automatic(runtime).phase == phase::calibrating &&
              !sunshine_game3d::recalibrate_automatic(runtime), "Pending action was overwritten or duplicated");
      require(take_recalibration(runtime) && !take_recalibration(runtime), "Recalibration not consumed exactly once");
      publish_automatic_ui(runtime, {phase::suspended, true});
      require(sunshine_game3d::recalibrate_automatic(runtime), "Changed basis cannot be explicitly recalibrated");
      publisher_t publisher;
      publisher.invalidate(runtime, false);
      require(!take_recalibration(runtime) && sunshine_game3d::query_automatic(runtime).phase == phase::unavailable,
        "Effect reload retained a pending action or stale status");
      publish_automatic_ui(runtime, {phase::waiting_for_depth, false});
      publisher.invalidate(runtime, true);
      require(sunshine_game3d::query_automatic(runtime).phase == phase::unavailable,
        "Destroyed runtime left UI state for a reused address");
      {
        publisher_t reloaded;
        publish_automatic_ui(runtime, {phase::ready, true});
        require(sunshine_game3d::recalibrate_automatic(runtime), "Reload fixture could not queue action");
      }
      require(!take_recalibration(runtime) && sunshine_game3d::query_automatic(runtime).phase == phase::unavailable,
        "Publisher teardown retained status/action in a pinned module");
    }

    static api::effect_runtime *owner() {
      static runtime_fixture::empty_runtime runtime;
      return &runtime;
    }

    static void seed_fx_proof(runtime_t &proof) {
      proof.texture = {101};
      proof.effect_name = "RetiredReference.fx";
      proof.game_technique = {102};
      proof.native_technique = {103};
      std::uint64_t handle = 104;
      for (auto *uniform : {&proof.depth_ready, &proof.calibrated, &proof.raw_anchor,
          &proof.raw_gain, &proof.depth_rect, &proof.direction, &proof.camera_ready,
          &proof.camera_basis, &proof.camera_projection, &proof.camera_scale,
          &proof.camera_zero, &proof.camera_blend, &proof.camera_rect,
          &proof.camera_raw_range, &proof.depth_jitter, &proof.effect_strength})
        uniform->handle = handle++;
      proof.native_uniforms_checked = proof.proof_checked = true;
    }

    static void require_fx_proof_cleared(const publisher_t &publisher, api::effect_runtime *runtime) {
      const auto found = publisher.runtimes_.find(runtime);
      require(found != publisher.runtimes_.end(), "Effect reload erased the live runtime owner");
      const auto &proof = found->second;
      require(!proof.texture.handle && !proof.game_technique.handle && !proof.native_technique.handle &&
          !proof.native_uniforms_checked && !proof.proof_checked,
        "Effect reload retained cached FX texture, technique or proof");
      for (const auto uniform : {proof.depth_ready, proof.calibrated, proof.raw_anchor,
          proof.raw_gain, proof.depth_rect, proof.direction, proof.camera_ready,
          proof.camera_basis, proof.camera_projection, proof.camera_scale,
          proof.camera_zero, proof.camera_blend, proof.camera_rect,
          proof.camera_raw_range, proof.depth_jitter, proof.effect_strength})
        require(!uniform.handle, "Effect reload retained a stale FX uniform handle");
    }

    static void activate(publisher_t &publisher, bool rendered = true) {
      auto metadata = publisher.identity_;
      metadata.window = reinterpret_cast<std::uint64_t>(GetForegroundWindow());
      metadata.generation = 1;
      publisher.runtimes_[owner()].rendered_since_present = rendered;
      publisher.publish(metadata, true, owner());
    }

    static std::unique_ptr<generation_t> pending_generation() {
      auto generation = std::make_unique<generation_t>();
      generation->backend = api::device_api::d3d12;
      generation->runtime = generation->owner_runtime = owner();
      generation->id = 1;
      generation->nonce = 42;
      generation->last_submitted = 1;
      generation->native_swapchain = 99;
      generation->pending_slot = 0;
      generation->pending_qpc = 1234;
      return generation;
    }

    static void present_invalidation() {
      publisher_t publisher;
      require(publisher.init(), "mapping creation failed");
      activate(publisher);
      publisher.present(owner());
      require(publisher.shared_->metadata.generation == 1, "executing technique was invalidated");
      // A paused game has no callback and retains this valid frame. A new Present without
      // an executing technique is an explicit disable/failure and invalidates immediately.
      publisher.present(owner());
      require(publisher.shared_->metadata.generation == 0, "disabled technique retained a stale image");
      const auto sequence = publisher.shared_->metadata_sequence;
      publisher.present(owner());
      require(publisher.shared_->metadata_sequence == sequence, "inactive metadata was rewritten each present");
      activate(publisher, false);
      publisher.shared_->metadata.generation = 0;
      publisher.present(owner());
      require(!publisher.shared_->metadata.window, "consumer-free discovery identity remained active after disable");
    }

    static void contended_reload() {
      publisher_t publisher;
      require(publisher.init(), "mapping creation failed");
      activate(publisher);
      seed_fx_proof(publisher.runtimes_.at(owner()));
      std::promise<void> entered;
      auto started = entered.get_future();
      std::unique_lock<std::mutex> lock(publisher.mutex_);
      auto invalidation = std::async(std::launch::async, [&] {
        entered.set_value();
        publisher.invalidate(owner(), false);
      });
      started.wait();
      require(invalidation.wait_for(std::chrono::milliseconds(25)) == std::future_status::timeout, "contended reload silently dropped its invalidation");
      lock.unlock();
      invalidation.get();
      require_fx_proof_cleared(publisher, owner());
      require(!publisher.published_runtime_, "reload retained published runtime proof");
      require(!publisher.shared_->metadata.generation, "reload retained published metadata");
    }

    static void overlay_reload() {
      publisher_t publisher;
      require(publisher.init(), "mapping creation failed");
      activate(publisher);
      seed_fx_proof(publisher.runtimes_.at(owner()));
      const auto before_open = publisher.shared_->metadata_sequence;
      publisher.set_overlay_open(owner(), true);
      require(publisher.overlay_open(owner()) && publisher.shared_->metadata.generation == 1, "overlay opening discarded live stereo proof");
      require(publisher.shared_->metadata_sequence == before_open, "overlay opening churned metadata without a source change");
      publisher.set_overlay_open(owner(), false);
      require(!publisher.overlay_open(owner()) && publisher.shared_->metadata.generation == 1, "overlay closing discarded live stereo proof");
      require(publisher.shared_->metadata_sequence == before_open, "overlay closing churned metadata without a source change");
      publisher.set_overlay_open(owner(), true);
      publisher.invalidate(owner(), false);
      require(publisher.overlay_open(owner()), "shader reload forgot that the overlay is open");
      require_fx_proof_cleared(publisher, owner());
      require(!publisher.shared_->metadata.generation, "shader reload retained stale stereo proof while overlay was open");
      publisher.set_overlay_open(owner(), false);
      require(!publisher.overlay_open(owner()) && !publisher.shared_->metadata.generation, "overlay close must permit a fresh technique rather than restore stale metadata");
      publisher.set_overlay_open(owner(), true);
      publisher.invalidate(owner(), true);
      require(!publisher.overlay_open(owner()), "runtime destruction retained overlay state across pointer reuse");
      require(publisher.runtimes_.empty(), "runtime destruction retained its native owner");
    }

    static void normalized_ui_input_preserves_source_authority() {
      using namespace sunshine_game3d;
      source_alpha_ui_policy policy;
      render_settings settings;
      alpha_auto_policy session;
      const auto generic = session.decision();
      require(generic.state == alpha_auto_state::waiting_for_source && !generic.enabled,
        "Normalized UI fixture did not begin with an Auto session that detected nothing");
      ui_input::presentation observed{{true, true, false, 1, 11, 7, 41}, true};
      ui_input::frame input;
      input.status = ui_input::resolve(policy, settings, observed);
      input.observation.session = &session;
      input.observation.now_ms = 1001;
      ui_plane_parameters plane;
      ui_adaptive::source scene;
      require(input.status.requested && input.status.fg_active() &&
        !input.for_render(plane, scene).available(),
        "Auto Off stopped dedicated input discovery or missing FG input became usable");
      // Even a stale view or an incorrect current-color hint cannot bypass the
      // presentation adapter's requirement for a usable submitted real input.
      input.kind = ui_input_kind::current_color_alpha;
      input.view = {123};
      require(!input.for_render(plane, scene).available(),
        "Missing FG input silently selected generated current color");
      input.automatic_detection = true;
      input.detection.current_color = true; // A stale candidate set is guarded too.
      require(!input.for_render(plane, scene).available() && !input.detection.current_color,
        "Automatic discovery admitted generated current color at the normalized boundary");
      input.detection.current_color = true;
      input.detection.masks[1] = {456};
      const auto captured_candidates = input.for_render(plane, scene);
      require(captured_candidates.available() && captured_candidates.kind == ui_input_kind::unavailable &&
        captured_candidates.detection && !captured_candidates.detection->current_color &&
        captured_candidates.detection->masks[1].handle == 456,
        "FG current-color filtering discarded an independent captured candidate");
      input.automatic_detection = false;
      input.detection = {};
      const auto held = ui_input::resolve(policy, settings, {{}, true});
      require(held.fg_active() && held.fg.epoch == 7 && !held.effective(),
        "Missing observer data cleared the provider's confirmed FG requirement");

      input.status.retained_alpha_ready = true;
      input.observation.retained = true;
      input.kind = ui_input_kind::captured_color_alpha;
      auto render_input = input.for_render(plane, scene);
      require(render_input.available() && render_input.kind == ui_input_kind::captured_color_alpha &&
        render_input.automatic == &input.observation && render_input.automatic->session == &session,
        "Generic real-input alpha lost its shared automatic policy or acquired dedicated semantics");
      input.kind = ui_input_kind::dedicated_mask;
      input.observation.dedicated_mask = true;
      input.channel = ui_mask_channel::red;
      render_input = input.for_render(plane, scene);
      require(render_input.available() && render_input.kind == ui_input_kind::dedicated_mask &&
        render_input.channel == ui_mask_channel::red && render_input.view.handle == 123 &&
        session.decision().state == alpha_auto_state::waiting_for_source,
        "Dedicated input was blocked by an undetected Auto session or changed its mode");
      input.view = {};
      require(!input.for_render(plane, scene).available(),
        "Failed captured-mask materialization fell back to current color");
      input.view = {123};
      settings.ui_protection = source_alpha_mode::off;
      input.status = ui_input::resolve(policy, settings, observed);
      input.status.retained_alpha_ready = true;
      require(!input.for_render(plane, scene).available(),
        "A retained dedicated mask bypassed Manual Off at the normalized boundary");

      settings.ui_protection = source_alpha_mode::automatic;
      observed.mode.enabled = false;
      ++observed.mode.sequence;
      input.status = ui_input::resolve(policy, settings, observed);
      input.kind = ui_input_kind::current_color_alpha;
      input.view = {};
      input.observation.retained = input.observation.dedicated_mask = false;
      input.channel = ui_mask_channel::alpha;
      require(input.for_render(plane, scene).available() && !input.status.fg_active(),
        "Confirmed FG Off did not restore eligible current-color alpha");
      renderer unconfigured;
      input.complete(unconfigured, false);
      require(!input.status.rendered && !input.status.applied &&
        input.status.coverage.state == alpha_auto_state::waiting_for_source,
        "Failed rendering published an applied input or changed the shared Auto decision");
    }

    static void normalized_ui_scene_match_only_gates_adaptive_evidence() {
      using namespace sunshine_game3d;
      ui_input::frame input;
      input.status.requested = input.status.retained_alpha_ready = true;
      input.status.fg = {true, true, false, 1, 11, 7, 41};
      input.kind = ui_input_kind::dedicated_mask;
      input.view = {123};
      input.observation.retained = input.observation.dedicated_mask = true;
      input.observation.now_ms = 1200; input.observation.tick_ms = 1100;
      input.observation.epoch = 7; input.observation.revision = 3;
      input.observation.viewport = 11; input.observation.sequence = 101;
      ui_adaptive::source scene;
      scene.eligible = true; scene.epoch = 7; scene.revision = 3; scene.viewport = 11;
      scene.tick_ms = 1150; scene.sequence = 99; scene.source_id = 56;
      const auto matched = input.match_scene(scene, true);
      require(matched.eligible && matched.now_ms == 1200 && matched.tick_ms == 1100 &&
        matched.mask_sequence == 101 && matched.sequence == scene.sequence && matched.source_id == scene.source_id,
        "Matching retained UI did not preserve independent depth identity and conservative freshness");
      ui_plane_parameters plane;
      for (unsigned field = 0; field != 3; ++field) {
        auto unrelated = scene;
        if (field == 0) ++unrelated.epoch;
        if (field == 1) ++unrelated.revision;
        if (field == 2) ++unrelated.viewport;
        const auto rejected = input.match_scene(unrelated, true);
        require(!rejected.eligible && input.for_render(plane, rejected).available() &&
          input.for_render(plane, rejected).kind == ui_input_kind::dedicated_mask,
          "Depth/camera scope mismatch disabled an independently valid UI mask");
      }
      require(!input.match_scene(scene, false).eligible,
        "Unready scene admitted adaptive conflict evidence");
      scene.eligible = false;
      require(!input.match_scene(scene, true).eligible,
        "UI scope matching revived rejected depth evidence");
      input.observation.retained = false;
      scene.eligible = true; scene.epoch = 99;
      const auto current = input.match_scene(scene, true);
      require(current.eligible && current.tick_ms == scene.tick_ms && !current.mask_sequence,
        "Current-color alpha invented a retained-mask association or rejected unrelated provider metadata");
    }

    static void source_alpha_scope_lifetime() {
      publisher_t publisher;
      auto &proof = publisher.runtimes_[owner()];
      const sunshine_game3d::frame_generation_mode fg {true, true, false, 1, 0, 7, 41};
      proof.frame.source_alpha = proof.source_alpha_policy.update(true, fg, true);
      proof.frame.source_alpha.input_state = sunshine_game3d::source_alpha_input_state::state_conflict;
      proof.frame.depth_ready = true;
      proof.frame.diagnostics.frame_generation_active = false; // Ready manual/Generic depth has no SL FG provenance.
      require(!proof.frame.source_alpha.effective(), "Manual depth bypassed the FG alpha guard");
      publish_source_alpha_ui(owner(), proof.frame.source_alpha);
      const auto visible = sunshine_game3d::query_source_alpha_ui(owner());
      require(visible.blocked_by_fg() && visible.fg.sequence == 41 &&
        visible.input_state == sunshine_game3d::source_alpha_input_state::state_conflict,
        "UI did not receive the renderer's frozen FG decision");
      auto presented = proof.frame.source_alpha;
      presented.mode = sunshine_game3d::source_alpha_mode::automatic;
      presented.input = sunshine_game3d::source_alpha_input::sl_ui_alpha;
      presented.rendered = presented.applied = presented.retained_alpha_ready = true;
      presented.automatic = presented.coverage.enabled = true;
      presented.coverage.state = sunshine_game3d::alpha_auto_state::dedicated_ui;
      publish_source_alpha_ui(owner(), presented);
      const auto active = sunshine_game3d::query_source_alpha_ui(owner());
      require(active.rendered && active.applied && active.input == sunshine_game3d::source_alpha_input::sl_ui_alpha &&
        active.coverage.state == sunshine_game3d::alpha_auto_state::dedicated_ui &&
        active.dedicated_ui_active(sunshine_game3d::source_alpha_mode::automatic, true),
        "UI did not receive the renderer's dedicated input and applied decision");
      publisher.invalidate(owner(), false);
      const auto invalidated = sunshine_game3d::query_source_alpha_ui(owner());
      require(!invalidated.rendered && !invalidated.applied &&
        invalidated.input == sunshine_game3d::source_alpha_input::none &&
        !invalidated.dedicated_ui_active(sunshine_game3d::source_alpha_mode::automatic, true),
        "An effects reload retained the prior applied UI source");
      const auto retained = publisher.runtimes_.at(owner()).source_alpha_policy.update(true, {}, true);
      require(retained.blocked_by_fg() && retained.fg.sequence == 41 &&
        retained.input_state == sunshine_game3d::source_alpha_input_state::not_observed &&
        !retained.rendered && !retained.applied && retained.input == sunshine_game3d::source_alpha_input::none,
        "An effects reload cleared a confirmed game FG mode");
      publish_source_alpha_ui(owner(), retained);
      require(!sunshine_game3d::query_source_alpha_ui(owner()).dedicated_ui_active(
        sunshine_game3d::source_alpha_mode::automatic, true),
        "FG mode continuity revived an unapplied dedicated UI source");
      publish_source_alpha_ui(owner(), presented);
      publisher.invalidate(owner(), true);
      const auto destroyed = sunshine_game3d::query_source_alpha_ui(owner());
      require(!destroyed.fg.known && !destroyed.rendered && !destroyed.applied &&
        destroyed.input == sunshine_game3d::source_alpha_input::none && publisher.runtimes_.empty(),
        "Runtime destruction retained FG mode or its UI status");
    }

    static void ui_source_choice_and_runtime_lifetime() {
      namespace ui = sunshine_game3d::ui_input;
      using choice = sunshine_game3d::ui_qualification::choice;
      runtime_fixture::empty_runtime runtime;
      ui::invalidate(&runtime);
      // Selecting before a GPU frame must work without touching the device or
      // manufacturing available pixels or requiring human approval.
      ui::select_source(&runtime, choice::sl_backbuffer);
      const auto selected = ui::source_status(&runtime);
      require(selected.selected == choice::sl_backbuffer && selected.token && !selected.available,
        "Pre-frame source selection disappeared or manufactured captured pixels");
      ui::suspend(&runtime);
      const auto suspended = ui::source_status(&runtime);
      require(suspended.selected == choice::sl_backbuffer && suspended.token > selected.token &&
        !suspended.available, "Disabling Game 3D lost the source choice or kept stale pixels");
      ui::suspend(&runtime);
      require(ui::source_status(&runtime).token == suspended.token,
        "Repeated disabled presentations continually restarted source identity");
      ui::select_source(&runtime, choice::sl_hudless);
      require(ui::source_status(&runtime).selected == choice::sl_hudless,
        "Source choice could not be changed while Game 3D was disabled");
      ui::invalidate(&runtime);
      require(ui::source_status(&runtime).selected == choice::automatic && !ui::source_status(&runtime).available,
        "Destroyed runtime retained session choice or pixels");
      ui::select_source(&runtime, choice::sl_backbuffer);
      require(ui::source_status(&runtime).selected == choice::sl_backbuffer && !ui::source_status(&runtime).available,
        "Recreated runtime inherited pixels from its previous lifetime");
      ui::invalidate(&runtime);
    }

    static void disabled_native_reload_lifetime() {
      struct tracked_runtime : runtime_fixture::empty_runtime {
        std::uint64_t get_native() const override { return 0x123456; }
      } runtime;
      struct tracked_swapchain : api::swapchain {
        std::uint64_t get_native() const override { return 0x123456; }
#define UNEXPECTED_SWAPCHAIN_CALL(...) __VA_ARGS__ override { throw std::runtime_error("Disabled Present accessed swapchain GPU state: " #__VA_ARGS__); }
        UNEXPECTED_SWAPCHAIN_CALL(void get_private_data(const std::uint8_t[16], std::uint64_t *) const)
        UNEXPECTED_SWAPCHAIN_CALL(void set_private_data(const std::uint8_t[16], std::uint64_t))
        UNEXPECTED_SWAPCHAIN_CALL(api::device *get_device())
        UNEXPECTED_SWAPCHAIN_CALL(void *get_hwnd() const)
        UNEXPECTED_SWAPCHAIN_CALL(api::resource get_back_buffer(std::uint32_t))
        UNEXPECTED_SWAPCHAIN_CALL(std::uint32_t get_back_buffer_count() const)
        UNEXPECTED_SWAPCHAIN_CALL(std::uint32_t get_current_back_buffer_index() const)
        UNEXPECTED_SWAPCHAIN_CALL(bool check_color_space_support(api::color_space) const)
        UNEXPECTED_SWAPCHAIN_CALL(api::color_space get_color_space() const)
#undef UNEXPECTED_SWAPCHAIN_CALL
      } swapchain;
      publisher_t publisher;
      require(publisher.init(), "mapping creation failed");
      publisher.track_runtime(&runtime);
      auto &proof = publisher.runtimes_.at(&runtime);
      proof.renderer = std::make_unique<sunshine_game3d::renderer>();
      const auto *renderer = proof.renderer.get();
      seed_fx_proof(proof);
      // Exercise the real disabled Present branch. Controls are uninitialized
      // in this isolated fixture, so it must not touch a device or GPU queue.
      publisher.render_present(&swapchain);
      require(!publisher.native_enabled(&runtime), "Disabled native path was reported enabled");
      publisher.invalidate(&runtime, false);
      require_fx_proof_cleared(publisher, &runtime);
      auto &reloaded = publisher.runtimes_.at(&runtime);
      require(reloaded.swapchain == runtime.get_native() && reloaded.renderer.get() == renderer,
        "Disabled reload lost the swapchain mapping or native renderer needed for re-enable");
      require(!reloaded.addon_native && !reloaded.native_output.handle && !reloaded.frame.prepared,
        "Disabled reload retained stale native frame state");
      // Mirror the next enabled Present after lookup, without configuring GPU
      // objects. A real no-FX runtime fixture separately exercises rendering.
      const auto found = std::find_if(publisher.runtimes_.begin(), publisher.runtimes_.end(),
        [&](const auto &entry) { return entry.second.swapchain == runtime.get_native(); });
      require(found != publisher.runtimes_.end() && found->first == &runtime,
        "Re-enable could no longer locate the runtime from its swapchain");
      found->second.addon_native = true;
      require(publisher.native_enabled(&runtime), "Re-enabled native owner was not retained");
      publisher.invalidate(&runtime, false);
      require(publisher.runtimes_.at(&runtime).renderer.get() == renderer,
        "Re-enabled reload replaced the native owner");
      publisher.invalidate(&runtime, true, runtime.get_native());
      require(publisher.runtimes_.empty() && !publisher.native_enabled(&runtime),
        "Runtime destruction retained a disabled/re-enabled native owner");
    }

    static void native_reload_clears_fx_handles() {
      using phase = sunshine_game3d::automatic_phase;
      publisher_t publisher;
      require(publisher.init(), "mapping creation failed");
      activate(publisher);
      auto &proof = publisher.runtimes_.at(owner());
      proof.addon_native = true;
      proof.swapchain = 99;
      proof.renderer = std::make_unique<sunshine_game3d::renderer>();
      const auto *renderer = proof.renderer.get();
      proof.native_output = {777};
      proof.native_depth_prepared = proof.frame.prepared = proof.frame.depth_ready = true;
      proof.frame.scene.ready = true;
      proof.frame.scene.scale = 4.f;
      proof.frame.scene_source = {1, 2, 3, 4};
      proof.raw_supported = true;
      proof.raw_basis_epoch = 77;
      seed_fx_proof(proof);
      publish_automatic_ui(owner(), {phase::ready, true});
      publisher.set_overlay_open(owner(), true);
      const auto before_reload = publisher.shared_->metadata_sequence;
      publisher.invalidate(owner(), false);
      require_fx_proof_cleared(publisher, owner());
      const auto &reloaded = publisher.runtimes_.at(owner());
      require(reloaded.addon_native && reloaded.swapchain == 99 && reloaded.renderer.get() == renderer &&
          reloaded.native_output.handle == 777 && reloaded.rendered_since_present,
        "Native effects reload invalidated its runtime, renderer or published output");
      require(reloaded.native_depth_prepared && reloaded.frame.prepared && reloaded.frame.depth_ready &&
          reloaded.frame.scene.ready && reloaded.frame.scene.scale == 4.f &&
          reloaded.frame.scene_source == std::array<std::uint64_t, 4>{1, 2, 3, 4} &&
          reloaded.raw_supported && reloaded.raw_basis_epoch == 77,
        "Native effects reload discarded a matching depth frame or scene calibration");
      require(publisher.shared_->metadata_sequence == before_reload && publisher.shared_->metadata.generation == 1 &&
          publisher.published_runtime_ == owner() && publisher.overlay_open(owner()) &&
          sunshine_game3d::query_automatic(owner()).phase == phase::ready,
        "Native effects reload churned active metadata, overlay or automatic status");
      publisher.invalidate(owner(), true, 99);
      require(publisher.runtimes_.empty() && !publisher.published_runtime_ &&
          !publisher.shared_->metadata.generation && !publisher.overlay_open(owner()) &&
          sunshine_game3d::query_automatic(owner()).phase == phase::unavailable,
        "Native runtime destruction retained output, owner or automatic status");
    }

    static void color_contract() {
      for (std::uint32_t input = 0; input <= 5; ++input) {
        for (const char *output : {"srgb", "scrgb", "pq", ""}) {
          export_color_t color;
          const bool expected = (input == 1 && std::strcmp(output, "srgb") == 0) ||
                                ((input == 2 || input == 3) && std::strcmp(output, "scrgb") == 0);
          require(decode_export_color(output, input, color) == expected, "export transfer declaration accepted a contradictory or unknown color space");
        }
      }
      publisher_t publisher;
      require(publisher.init(), "mapping creation failed");
      activate(publisher);
      runtime_t proof;
      require(decode_export_color("scrgb", 3, proof.color), "PQ export contract rejected");
      require(publisher.source_color_matches(owner(), proof, api::color_space::hdr10_pq), "matching PQ source rejected");
      require(!publisher.source_color_matches(owner(), proof, api::color_space::scrgb), "stale PQ decoder accepted a scRGB source");
      require(!publisher.shared_->metadata.generation, "changed game transfer retained stale HDR publication");
      require(decode_export_color("srgb", 1, proof.color), "SDR source contract rejected");
      require(publisher.source_color_matches(owner(), proof, api::color_space::srgb), "SDR game requires an SDR desktop");

      wire::metadata_t metadata;
      source_t source;
      source.format = DXGI_FORMAT_R16G16B16A16_FLOAT;
      source.color = proof.color;
      publisher_t::set_source(metadata, source);
      require(metadata.color_transfer == wire::transfer::srgb, "floating point format incorrectly implied HDR");
      require(decode_export_color("scrgb", 2, source.color), "native scRGB contract rejected");
      publisher_t::set_source(metadata, source);
      require(metadata.color_transfer == wire::transfer::scrgb && metadata.dxgi_format == 10, "explicit scRGB metadata lost its transfer");
    }

    static void bounded_retirement() {
      publisher_t publisher;
      require(publisher.init(), "mapping creation failed");
      publisher.generation_ = pending_generation();
      require(!publisher.generation_->finished(), "unsubmitted copy appeared complete");
      publisher.generation_->owner_destroyed = true;
      publisher.retire_destroyed_generation();
      require(!publisher.generation_ && publisher.retired_, "destroyed runtime blocked its successor");
      publisher.generation_ = pending_generation();
      publisher.generation_->owner_destroyed = true;
      publisher.retire_destroyed_generation();
      require(publisher.generation_ && publisher.retired_ && publisher.retirement_limit_logged_, "unconfirmed generations were not bounded");
      // No GPU commands exist in this test; mark those synthetic copies discarded.
      publisher.generation_->pending_slot = publisher.retired_->pending_slot = wire::slot_count;
      publisher.generation_->last_submitted = publisher.retired_->last_submitted = 0;
      publisher.retire_destroyed_generation();
      require(!publisher.generation_ && !publisher.retired_, "completed retired generations were not reclaimed");
    }

    static void d3d12_deferred_signal(bool retire_before_signal) {
      publisher_t publisher;
      require(publisher.init(), "mapping creation failed");
      activate(publisher);
      auto generation = pending_generation();
      require(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(generation->device12.put()))), "D3D12 device creation failed");
      D3D12_COMMAND_QUEUE_DESC queue_desc {};
      require(SUCCEEDED(generation->device12->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(generation->queue12.put()))), "D3D12 queue creation failed");
      require(SUCCEEDED(generation->device12->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(generation->fence12.put()))), "D3D12 shared fence creation failed");
      const auto native_queue = reinterpret_cast<std::uint64_t>(generation->queue12.get());
      publisher.generation_ = std::move(generation);
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&publisher.shared_->consumer_nonce), 42);
      publisher.generation_->pending_ui_parallax_uv = .0037f;
      store_state(publisher.shared_->slots[0], 1, wire::slot_state::writing);
      publisher.finish_present(native_queue + 1, 99);
      publisher.finish_present(native_queue, 100);
      require(publisher.generation_->pending_slot == 0, "unrelated queue/swapchain consumed a pending publication");
      require(publisher.shared_->slots[0].sequence == 0, "copy became visible before matching finish_present");
      if (retire_before_signal) {
        publisher.begin_present(99, api::color_space::srgb_nonlinear);
        require(!publisher.shared_->metadata.generation, "missing finish_present retained stale metadata");
        require(!publisher.generation_->finished(), "missing finish_present released pending resources");
      }
      publisher.finish_present(native_queue, 99);
      require(publisher.generation_->pending_slot == wire::slot_count, "matching Present did not fence the copy");
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (!publisher.generation_->finished() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      require(publisher.generation_->finished(), "native D3D12 publication fence did not complete");
      const auto &slot = publisher.shared_->slots[0];
      require(wire::control_state(slot.control) == (retire_before_signal ? wire::slot_state::writing : wire::slot_state::ready), "retired copy was republished or valid copy was hidden");
      if (!retire_before_signal) {
        require(slot.sequence == 1 && slot.qpc == 1234, "published copy lost its source timestamp");
        require(slot.cursor_plane_flags == wire::cursor_plane_present && slot.ui_parallax_uv == .0037f,
          "Published cursor plane did not follow its fenced source frame");
      }
    }

    static void d3d12_async_ring() {
      publisher_t publisher;
      require(publisher.init(), "Async ring mapping creation failed");
      activate(publisher);
      // Drain the native test queue before publisher destruction, even when an
      // assertion throws. The publisher remains the sole owner of copy sources.
      exporter_async_fixture::gpu_t gpu;
      auto next = std::make_unique<generation_t>();
      next->runtime = next->owner_runtime = owner();
      next->id = 1; next->nonce = 42; next->native_swapchain = 99;
      next->source.width = exporter_async_fixture::width / 2;
      next->source.height = exporter_async_fixture::height;
      next->source.format = DXGI_FORMAT_R8G8B8A8_UNORM;
      require(next->create_native(api::device_api::d3d12, reinterpret_cast<std::uint64_t>(gpu.device.get()),
        reinterpret_cast<std::uint64_t>(gpu.queue.get())), "Async ring production allocations failed");
      publisher.generation_ = std::move(next);
      auto &generation = *publisher.generation_;
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&publisher.shared_->consumer_nonce), 42);
      const auto queue = reinterpret_cast<std::uint64_t>(gpu.queue.get());
      std::array<IUnknown *, wire::slot_count> expected_sources{};
      std::array<std::uint64_t, wire::slot_count> timestamps{};

      auto record = [&](unsigned pattern, unsigned index) {
        auto source = gpu.prepare(pattern);
        const auto source_pointer = reinterpret_cast<IUnknown *>(source.get());
        const auto sequence = generation.last_submitted + 1;
        generation.pending_slot = index;
        generation.pending_qpc = 1000 + sequence;
        generation.pending_ui_parallax_uv = float(sequence) * .001f;
        exporter_async_fixture::native_commands commands;
        commands.native = gpu.copies[pattern].commands.get();
        require(generation.submit(&commands, {reinterpret_cast<std::uint64_t>(source.get())}, index, sequence),
          "Production generation failed to record an asynchronous copy");
        require(publisher.acquire_slot(generation.completed()) == wire::slot_count,
          "An unfenced pending recording admitted another export");
        source.reset(); // Only production slot ownership keeps this source alive.
        expected_sources[index] = source_pointer;
        for (unsigned slot = 0; slot != wire::slot_count; ++slot) {
          if (expected_sources[slot]) require(generation.submitted_sources[slot].get() == expected_sources[slot],
            "A later copy released a different slot's in-flight source");
        }
        gpu.submit_and_read(pattern, generation.textures12[index].get());
        publisher.finish_present(queue, 99);
        require(generation.pending_slot == wire::slot_count && !generation.signal_failed,
          "Native finish_present did not submit the copy's fence");
        require(wire::control_state(publisher.shared_->slots[index].control) == wire::slot_state::ready &&
          publisher.shared_->slots[index].sequence == sequence && publisher.shared_->slots[index].qpc == 1000 + sequence,
          "A queued export lost slot ownership, sequence or timestamp");
        require(publisher.shared_->slots[index].cursor_plane_flags == wire::cursor_plane_present &&
            publisher.shared_->slots[index].ui_parallax_uv == float(sequence) * .001f,
          "A queued export inherited another frame's cursor plane");
        publisher.next_slot_ = (index + 1) % wire::slot_count;
      };

      gpu.hold();
      // No CPU or GPU completion wait between these three independent copies.
      // The old one-copy admission gate fails on the second acquisition.
      for (unsigned index = 0; index != wire::slot_count; ++index) {
        require(generation.completed() == 0, "GPU gate released before asynchronous admission was tested");
        require(publisher.acquire_slot(generation.completed()) == index,
          "An incomplete earlier export prevented filling a different ring slot");
        record(index, index);
        timestamps[index] = publisher.shared_->slots[index].qpc;
      }
      require(generation.last_submitted == wire::slot_count && generation.completed() == 0 && !generation.finished(),
        "Async regression did not leave all three production copies in flight");
      const auto admission_start = GetTickCount64();
      require(publisher.acquire_slot(generation.completed()) == wire::slot_count,
        "A fourth in-flight copy exceeded the bounded three-slot ring");
      // A receiver may discard an unfinished publication without reading it.
      // Free is a host ownership state, not evidence of producer completion.
      store_state(publisher.shared_->slots[0], 1, wire::slot_state::free);
      store_state(publisher.shared_->slots[1], 1, wire::slot_state::reading);
      require(publisher.acquire_slot(generation.completed()) == wire::slot_count,
        "An unfinished free/reading/ready slot was overwritten");
      require(GetTickCount64() - admission_start < 1000, "Backpressure blocked CPU admission for a second");
      for (unsigned index = 0; index != wire::slot_count; ++index)
        require(publisher.shared_->slots[index].sequence == index + 1 && publisher.shared_->slots[index].qpc == timestamps[index],
          "Rejected admission modified an in-flight slot's identity");

      gpu.release();
      auto wait_completed = [&](std::uint64_t sequence) {
        const auto deadline = GetTickCount64() + 2000;
        while (generation.completed() < sequence && GetTickCount64() < deadline) Sleep(1);
        require(generation.completed() == sequence, "Queued export fences failed to complete within two seconds");
      };
      wait_completed(wire::slot_count);
      for (unsigned index = 0; index != wire::slot_count; ++index) gpu.check_pixels(index);
      require(generation.finished(), "Completed async copies left their generation in flight");
      require(wire::control_state(publisher.shared_->slots[1].control) == wire::slot_state::reading,
        "Producer completion stole the receiver's reading slot");

      // GPU completion permits reuse only after the host releases its claim.
      for (auto &slot : publisher.shared_->slots) store_state(slot, 1, wire::slot_state::reading);
      require(publisher.acquire_slot(generation.completed()) == wire::slot_count,
        "Completed producer work overwrote a receiver-owned slot");
      store_state(publisher.shared_->slots[0], 1, wire::slot_state::free);
      require(publisher.acquire_slot(generation.completed()) == 0, "Released completed slot did not resume exports");
      record(3, 0);
      wait_completed(4);
      gpu.check_pixels(3);
      for (unsigned index = 1; index != wire::slot_count; ++index)
        require(wire::control_state(publisher.shared_->slots[index].control) == wire::slot_state::reading &&
          publisher.shared_->slots[index].sequence == index + 1 && publisher.shared_->slots[index].qpc == timestamps[index],
          "Resumed export changed a different receiver-owned slot");
      store_state(publisher.shared_->slots[0], 1, wire::slot_state::reading);
      store_state(publisher.shared_->slots[2], 1, wire::slot_state::ready);
      require(publisher.acquire_slot(generation.completed()) == 2, "Completed unclaimed ready slot could not be reused");
      store_state(publisher.shared_->slots[2], 1, wire::slot_state::ready); // No copy was recorded by this final ownership check.
      std::puts("PASS native D3D12 async ring: 3 gated copies, fourth backpressure, independent source pixels, free/ready/reading ownership, delayed recovery");
    }

    static void native_sharing(bool d3d12, DXGI_FORMAT format) {
      com_ptr<ID3D11Device> receiver;
      com_ptr<ID3D11DeviceContext> receiver_context;
      require(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, receiver.put(), nullptr, receiver_context.put())), "D3D11 receiver device failed");
      com_ptr<ID3D11Device5> receiver5;
      require(SUCCEEDED(receiver->QueryInterface(IID_PPV_ARGS(receiver5.put()))), "D3D11 receiver fence interface unavailable");

      com_ptr<ID3D11Device> producer11;
      com_ptr<ID3D11DeviceContext> context11;
      com_ptr<ID3D12Device> producer12;
      com_ptr<ID3D12CommandQueue> queue12;
      generation_t generation;
      generation.source.width = generation.source.height = 32;
      generation.source.format = format;
      if (d3d12) {
        require(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(producer12.put()))), "D3D12 producer device failed");
        D3D12_COMMAND_QUEUE_DESC queue_desc {};
        require(SUCCEEDED(producer12->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(queue12.put()))), "D3D12 producer queue failed");
        require(generation.create_native(api::device_api::d3d12, reinterpret_cast<std::uint64_t>(producer12.get()), reinterpret_cast<std::uint64_t>(queue12.get())), "production D3D12 allocation failed");
      } else {
        require(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, producer11.put(), nullptr, context11.put())), "D3D11 producer device failed");
        require(generation.create_native(api::device_api::d3d11, reinterpret_cast<std::uint64_t>(producer11.get()), reinterpret_cast<std::uint64_t>(context11.get())), "production D3D11 allocation failed");
      }

      for (const auto &handle : generation.texture_handles) {
        com_ptr<ID3D11Texture2D> imported;
        require(SUCCEEDED(receiver5->OpenSharedResource1(handle.get(), IID_PPV_ARGS(imported.put()))), "receiver could not import a production texture");
        D3D11_TEXTURE2D_DESC desc {};
        imported->GetDesc(&desc);
        require(desc.Width == 64 && desc.Height == 32 && desc.Format == format, "production shared texture dimensions/format changed at import");
      }
      com_ptr<ID3D11Fence> imported_fence;
      require(SUCCEEDED(receiver5->OpenSharedFence(generation.fence_handle.get(), IID_PPV_ARGS(imported_fence.put()))), "receiver could not import the production ready fence");
      generation.last_submitted = 1;
      if (d3d12) {
        require(SUCCEEDED(generation.queue12->Signal(generation.fence12.get(), 1)), "D3D12 shared signal failed");
      } else {
        require(SUCCEEDED(generation.context11->Signal(generation.fence11.get(), 1)), "D3D11 shared signal failed");
        generation.context11->Flush();
      }
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (imported_fence->GetCompletedValue() < 1 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      require(imported_fence->GetCompletedValue() == 1, "shared producer signal was not visible to receiver");
    }
  };
}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::strcmp(argv[1], "--scene-strength") == 0) {
      publisher_tests::adaptive_ui_source_provenance();
      publisher_tests::source_alpha_scope_lifetime();
      publisher_tests::retained_depth_decision();
      publisher_tests::pending_scene_identity();
      publisher_tests::camera_missing_holds_projection_scene();
      publisher_tests::cached_scene_strength_publication();
      publisher_tests::automatic_raw_and_matrix_resolver();
      publisher_tests::automatic_raw_and_matrix_resolver(true);
      std::puts("PASS: current scene strength and cached FG publication (CPU only)");
      return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--ui-status") == 0) {
      publisher_tests::normalized_ui_input_preserves_source_authority();
      publisher_tests::normalized_ui_scene_match_only_gates_adaptive_evidence();
      publisher_tests::source_alpha_scope_lifetime();
      publisher_tests::ui_source_choice_and_runtime_lifetime();
      std::puts("PASS: normalized UI input authority, adaptive association, publication and lifecycle cleanup (CPU only)");
      return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--depth-provenance") == 0) {
      publisher_tests::copied_depth_provenance();
      std::puts("PASS: consumed SDK depth preservation provenance (CPU only)");
      return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--async-ring") == 0) {
      // Opt-in GPU regression; a stuck driver must not leave the game-like gate
      // waiting indefinitely. The process exits before this detached watchdog.
      std::thread([] {
        Sleep(15000);
        std::fputs("FAIL async exporter ring watchdog\n", stderr);
        TerminateProcess(GetCurrentProcess(), 3);
      }).detach();
      publisher_tests::d3d12_async_ring();
      return 0;
    }
    require(argc == 1, "usage: reshade_exporter_tests [--async-ring|--depth-provenance|--scene-strength|--ui-status]");
    publisher_tests::copied_depth_provenance();
    publisher_tests::adaptive_ui_source_provenance();
    publisher_tests::normalized_ui_input_preserves_source_authority();
    publisher_tests::normalized_ui_scene_match_only_gates_adaptive_evidence();
    publisher_tests::source_alpha_scope_lifetime();
    publisher_tests::ui_source_choice_and_runtime_lifetime();
    publisher_tests::retained_depth_decision();
    publisher_tests::automatic_ui_lifetime();
    publisher_tests::jitter_publication();
    publisher_tests::cached_scene_strength_publication();
    publisher_tests::pending_scene_identity();
    publisher_tests::camera_missing_holds_projection_scene();
    publisher_tests::automatic_raw_and_matrix_resolver();
    publisher_tests::automatic_raw_and_matrix_resolver(true);
    publisher_tests::session_teardown_ownership();
    publisher_tests::process_detach_lifetime();
    publisher_tests::present_invalidation();
    publisher_tests::contended_reload();
    publisher_tests::overlay_reload();
    publisher_tests::disabled_native_reload_lifetime();
    publisher_tests::native_reload_clears_fx_handles();
    publisher_tests::color_contract();
    publisher_tests::bounded_retirement();
    publisher_tests::d3d12_deferred_signal(false);
    publisher_tests::d3d12_deferred_signal(true);
    publisher_tests::native_sharing(false, DXGI_FORMAT_R10G10B10A2_UNORM);
    publisher_tests::native_sharing(true, DXGI_FORMAT_R10G10B10A2_UNORM);
    publisher_tests::native_sharing(false, DXGI_FORMAT_R16G16B16A16_FLOAT);
    publisher_tests::native_sharing(true, DXGI_FORMAT_R16G16B16A16_FLOAT);
    std::puts("PASS: exporter lifecycle, native D3D12 deferred publication, production D3D11/D3D12 sharing");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
