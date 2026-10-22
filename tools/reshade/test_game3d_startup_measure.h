// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The present-thread cost of a Game 3D start at 4K through the production
// configure path (configure with prepare_in_background, as the exporter calls
// it every Present): each configure call until the renderer is ready, the
// ready Present's setup (configure, the PQ export choice and the back-buffer
// view, as the exporter's setup section), and the CPU time of the first and
// second render with automatic UI detection and the adaptive UI probe, whose
// GPU objects a renderer used to create on its first render. Evidence only:
// the driver's shader cache decides most of these numbers (a fresh executable
// name gives a cold one).
#include "game3d_renderer.h"
#include "game3d_diagnostics.h"
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace sunshine_game3d_test {
  struct startup_timing {
    unsigned configure_calls = 0;
    double configure_max_ms = 0, configure_sum_ms = 0, ready_setup_ms = 0, ready_wall_ms = 0;
    double first_render_ms = 0, second_render_ms = 0;
  };

  inline startup_timing measure_startup(reshade::api::effect_runtime *runtime, unsigned color, const char *api_name,
      std::uint32_t width = 3840, std::uint32_t height = 2160) {
    namespace api = reshade::api;
    using namespace sunshine_game3d;
    using clock = std::chrono::steady_clock;
    const auto check = [](bool condition, const char *message) {
      if (!condition) throw std::runtime_error(message);
    };
    const auto ms = [](clock::time_point from, clock::time_point to) {
      return std::chrono::duration<double, std::milli>(to - from).count();
    };
    auto *device = runtime->get_device();
    auto *queue = runtime->get_command_queue();
    const auto format = color == 1 ? api::format::r8g8b8a8_unorm : color == 2 ? api::format::r16g16b16a16_float :
      api::format::r10g10b10a2_unorm;
    const auto space = color == 1 ? api::color_space::srgb : color == 2 ? api::color_space::scrgb : api::color_space::hdr10_pq;
    // A standalone game colour texture (resting in COMMON, as a presented
    // back buffer) and a full-resolution depth view.
    api::resource source{}, depth{};
    api::resource_view depth_view{};
    check(device->create_resource(api::resource_desc(width, height, 1, 1, format, 1, api::memory_heap::default_,
        api::resource_usage::render_target | api::resource_usage::shader_resource | api::resource_usage::copy_source |
        api::resource_usage::copy_dest), nullptr, api::resource_usage::general, &source),
      "The startup measurement cannot create its colour texture");
    check(device->create_resource(api::resource_desc(width, height, 1, 1, api::format::r32_float, 1, api::memory_heap::default_,
        api::resource_usage::shader_resource), nullptr, api::resource_usage::shader_resource, &depth) &&
        device->create_resource_view(depth, api::resource_usage::shader_resource, api::resource_view_desc(api::format::r32_float),
          &depth_view), "The startup measurement cannot create its depth view");
    // Live default: the Diagnostics switch is off (no GPU timestamps).
    const bool diagnostics_enabled = diagnostics::enabled();
    diagnostics::set_enabled(false);
    startup_timing result;
    {
      renderer target;
      const auto start = clock::now();
      bool ready = false;
      while (!ready) {
        check(clock::now() - start < std::chrono::seconds(60), "The 4K startup renderer was not ready within 60 s");
        const auto before = clock::now();
        ready = target.configure(runtime, source, space, {}, true);
        if (ready) {
          // The rest of the exporter's setup section on the ready Present.
          target.set_pq_output(color != 1);
          check(target.native_rtv(source).handle != 0, "The 4K startup renderer has no back-buffer view");
        }
        const auto after = clock::now();
        const double call_ms = ms(before, after);
        ++result.configure_calls;
        result.configure_max_ms = std::max(result.configure_max_ms, call_ms);
        result.configure_sum_ms += call_ms;
        if (ready) {
          result.ready_setup_ms = call_ms;
          result.ready_wall_ms = ms(start, after);
        } else {
          check(target.preparing(), "The 4K startup renderer was unavailable instead of preparing");
          Sleep(1);
        }
      }
      render_parameters p;
      p.strength = 50.f;
      p.depth_ready = p.camera_ready = 1;
      p.coordinate_basis = 1;
      p.depth_scale = 128.f;
      p.strength_blend = 1.f;
      p.projection = {0.f, 1.f};
      p.convergence = {.05f, 1.f / 128.f};
      alpha_auto_policy session;
      alpha_auto_source automatic;
      automatic.epoch = 41; automatic.revision = 1; automatic.viewport = 1; automatic.session = &session;
      ui_adaptive::source adaptive;
      adaptive.epoch = 41; adaptive.revision = 1; adaptive.source_id = 7; adaptive.viewport = 1; adaptive.eligible = true;
      render_frame_input frame;
      frame.color = source;
      frame.depth = depth_view;
      frame.scene = p;
      frame.ui.kind = ui_input_kind::current_color_alpha;
      frame.ui.plane = {ui_plane_mode::display_fraction, 0.f};
      frame.ui.automatic = &automatic;
      frame.ui.adaptive = &adaptive;
      for (unsigned index = 0; index != 2; ++index) {
        const std::uint64_t now = GetTickCount64();
        automatic.now_ms = automatic.tick_ms = adaptive.now_ms = adaptive.tick_ms = now;
        automatic.sequence = adaptive.sequence = index + 1;
        target.begin_present();
        auto *commands = queue->get_immediate_command_list();
        const auto before = clock::now();
        const bool rendered = target.render(commands, frame, true) && target.pack(commands);
        const double render_ms = ms(before, clock::now());
        check(rendered, "The 4K startup renderer rejected its frame");
        (index ? result.second_render_ms : result.first_render_ms) = render_ms;
        queue->flush_immediate_command_list();
        target.finish_present();
        queue->wait_idle();
      }
      check(target.consumed_detection().state != ui_detection_snapshot::run_state::inactive,
        "The 4K startup frame did not run automatic UI detection");
      queue->wait_idle();
      target.reset_after_runtime_drain();
    }
    diagnostics::set_enabled(diagnostics_enabled);
    device->destroy_resource_view(depth_view);
    device->destroy_resource(depth);
    device->destroy_resource(source);
    std::printf("MEASURE %s startup %ux%u color=%u configure_calls=%u configure_max_ms=%.3f configure_sum_ms=%.3f "
      "ready_setup_ms=%.3f ready_wall_ms=%.1f first_render_ms=%.3f second_render_ms=%.3f\n", api_name, width, height, color,
      result.configure_calls, result.configure_max_ms, result.configure_sum_ms, result.ready_setup_ms, result.ready_wall_ms,
      result.first_render_ms, result.second_render_ms);
    // Loose bounds (an idle machine measures well under a few ms): before the
    // background build a cold start took ~470 ms of configure and ~340 ms of
    // first render on D3D12 at 4K.
    check(result.configure_max_ms < 50.0 && result.first_render_ms < 50.0,
      "A 4K renderer start blocked the present thread (configure or first render over 50 ms)");
    return result;
  }
}
