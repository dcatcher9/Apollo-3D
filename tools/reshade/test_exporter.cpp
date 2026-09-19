// SPDX-License-Identifier: GPL-3.0-only
// Exercise the actual publisher state transitions without loading an add-on into a game.
#include "exporter.cpp"

#include <algorithm>
#include <atomic>
#include <future>
#include <stdexcept>
#include <thread>
#include "test_exporter_async_d3d12.h"

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

    static void automatic_raw_and_matrix_resolver(bool native_cropped = false) {
      runtime_fixture::empty_runtime runtime;
      publisher_t publisher;
      runtime_t proof;
      proof.addon_native = native_cropped;
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
      if (native_cropped) depth.provided.sequence = 1;
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
        require(supported_fx.ready && supported_fx.rect == active_rect,
          "Legacy FX with camera-rect support lost a valid padded depth scene");
        proof.camera_rect = {};
        proof.addon_native = true;
      };
      if (native_cropped) {
        const auto pending = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 1000);
        require(!pending.ready && pending.rect == active_rect &&
            pending.ui.phase == sunshine_game3d::automatic_phase::calibrating &&
            pending.ui.scale.basis == sunshine_game3d::automatic_scale_basis::relative_depth,
          "Padded native NGX depth did not enter raw calibration");
      }
      std::array<float, sunshine_raw_scene::grid_width * sunshine_raw_scene::grid_height> raw;
      const auto measure = [&](float value, std::uint64_t now) {
        raw.fill(value);
        ++depth.provided.sequence;
        depth.provided.tick = now;
        const auto current = sunshine_provided_raw::selected(depth.provided, proof.raw_basis_epoch, true);
        const auto sample = sunshine_provided_raw::measured(depth.provided, depth.provided.sequence,
          proof.raw_basis_epoch, raw);
        proof.provided_raw_policy.update(current, &sample, now);
      };
      for (unsigned i = 0; i != 4; ++i) measure(.25f, 1000 + i * 250);
      const auto scene = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 1750);
      require(scene.ready && scene.basis == 1 && scene.scale == 4.f && scene.zero[1] == .25f,
        "NGX resolver did not retain ready relative stereo");
      check_native_crop(scene, 1750);
      measure(.125f, 2000);
      const auto tracking = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 2000);
      require(tracking.ready && tracking.scale > scene.scale && tracking.zero[1] < scene.zero[1],
        "NGX resolver stopped ordinary screen-plane tracking");

      depth.projection.supplied = true;
      depth.projection.epoch = 3;
      depth.projection.A = -.01;
      depth.projection.B = .2;
      proof.projection_domain = {3, 0};
      proof.projection_policy.reset(proof.projection_domain, 2000);
      const auto coefficients = sunshine_projection_depth::make(depth.projection.A, depth.projection.B);
      if (native_cropped) {
        const auto pending = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 2000);
        require(!pending.ready && pending.rect == active_rect &&
            pending.ui.phase == sunshine_game3d::automatic_phase::calibrating &&
            pending.ui.scale.basis == sunshine_game3d::automatic_scale_basis::camera_matrix,
          "Padded native camera depth did not enter projection calibration");
      }
      for (unsigned i = 0; i != 4; ++i) {
        sunshine_projection_depth::sample sample;
        sample.id = i + 1;
        sample.capture_ms = 2001 + i * 250;
        sample.logical_domain = proof.projection_domain;
        sample.projection = coefficients;
        sample.raw.fill(.25f);
        proof.projection_policy.observe(sample, sample.capture_ms);
      }
      const auto matrix = publisher.resolve_raw_scene(&runtime, proof, depth, true, true, 2751);
      require(matrix.ready && matrix.basis == 0 && matrix.projection[0] == coefficients.shader_A &&
          matrix.projection[1] == coefficients.inverseB && std::abs(matrix.zero[1] - 1.3f) < 1e-6f &&
          std::abs(matrix.scale * matrix.zero[1] - 1.f) < 1e-6f,
        "Associated matrix did not take precedence over the relative-depth reference");
      check_native_crop(matrix, 2751);
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
          &proof.camera_raw_range, &proof.depth_jitter})
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
          proof.camera_raw_range, proof.depth_jitter})
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
    require(argc == 1, "usage: reshade_exporter_tests [--async-ring]");
    publisher_tests::automatic_ui_lifetime();
    publisher_tests::jitter_publication();
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
