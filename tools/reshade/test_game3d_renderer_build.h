// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The background renderer build (renderer::configure with
// prepare_in_background) on an actual ReShade device: the game stays 2D
// while it runs, the ready call only adopts it, and every lifecycle path ends
// it exactly once: a configuration change, a toggle back to the current or
// cached renderer, a runtime reset or park, a destroyed device, a failed
// step, and the present-thread executor of a single-threaded device.
// Requires SUNSHINE_GAME3D_RENDERER_TEST (the build hooks).
#include "game3d_renderer.h"
#include <windows.h>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace sunshine_game3d_test {
  inline void check_background_build(reshade::api::effect_runtime *runtime, unsigned color, const char *api_name) {
    namespace api = reshade::api;
    using sunshine_game3d::renderer;
    using clock = std::chrono::steady_clock;
    const auto check = [api_name](bool condition, const char *message) {
      if (!condition) throw std::runtime_error(std::string(api_name) + " background build: " + message);
    };
    auto *device = runtime->get_device();
    auto *queue = runtime->get_command_queue();
    const auto format = color == 1 ? api::format::r8g8b8a8_unorm : color == 2 ? api::format::r16g16b16a16_float :
      api::format::r10g10b10a2_unorm;
    const auto space = color == 1 ? api::color_space::srgb : color == 2 ? api::color_space::scrgb : api::color_space::hdr10_pq;
    const auto other_space = color == 1 ? api::color_space::scrgb : api::color_space::srgb;
    // Two game colour textures of different extents, resting in COMMON.
    struct target { std::uint32_t width, height; api::resource resource{}; };
    target a{1280, 720}, b{960, 540};
    for (auto *t : {&a, &b})
      check(device->create_resource(api::resource_desc(t->width, t->height, 1, 1, format, 1, api::memory_heap::default_,
          api::resource_usage::render_target | api::resource_usage::shader_resource | api::resource_usage::copy_source |
          api::resource_usage::copy_dest), nullptr, api::resource_usage::general, &t->resource), "cannot create a colour texture");
    const auto hooks = [](const char *fail_step, unsigned delay_ms, bool present_thread) {
      renderer::build_test_hooks value;
      value.fail_step = fail_step;
      value.step_delay_ms = delay_ms;
      value.present_thread = present_thread;
      renderer::set_build_test_hooks(value);
    };
    // Calls configure as the exporter does each Present until it is ready;
    // every call before is preparing (2D). Returns the calls and the longest.
    struct calls_t { unsigned count = 0; double longest_ms = 0; };
    const auto until_ready = [&](renderer &r, const target &t, api::color_space s) {
      calls_t calls;
      const auto start = clock::now();
      for (;;) {
        check(clock::now() - start < std::chrono::seconds(60), "a renderer was not ready within 60 s");
        const auto before = clock::now();
        const bool ready = r.configure(runtime, t.resource, s, {}, true);
        const double ms = std::chrono::duration<double, std::milli>(clock::now() - before).count();
        ++calls.count;
        calls.longest_ms = std::max(calls.longest_ms, ms);
        if (ready) return calls;
        check(r.preparing(), "a pending build was not reported as preparing");
        Sleep(1);
      }
    };
    const auto settled = [&] {
      const auto start = clock::now();
      while (renderer::builds_in_flight()) {
        check(clock::now() - start < std::chrono::seconds(30), "an abandoned build never ended");
        Sleep(1);
      }
    };
    const auto render_once = [&](renderer &r, const target &t) {
      sunshine_game3d::render_frame_input frame;
      frame.color = t.resource;
      frame.scene.depth_ready = frame.scene.camera_ready = 0;
      r.begin_present();
      auto *commands = queue->get_immediate_command_list();
      check(r.render(commands, frame, true) && r.pack(commands), "an adopted renderer rejected its frame");
      queue->flush_immediate_command_list();
      r.finish_present();
      queue->wait_idle();
      const auto output = device->get_resource_desc(r.output());
      check(output.texture.width == t.width * 2 && output.texture.height == t.height,
        "an adopted renderer packs another configuration's extent");
    };
    settled();
    hooks(nullptr, 0, false);
    {
      // The ready call adopts a build that also prepared what the first
      // render used to create; the build leaves the registry on adoption.
      renderer r;
      check(!r.configure(runtime, a.resource, space, {}, true) && r.preparing(), "the first call did not stay 2D");
      until_ready(r, a, space);
      check(r.first_render_prepared() && !renderer::builds_in_flight() && !r.building(),
        "the adopted renderer lacks its first-render objects, or its build stayed registered");
      render_once(r, a);
      r.reset_after_runtime_drain();
    }
    {
      // A configuration change mid-build cancels it and builds the new one;
      // a toggle back to the current renderer mid-build cancels that build.
      hooks(nullptr, 10, false);
      renderer r;
      const auto started = renderer::builds_started();
      check(!r.configure(runtime, a.resource, space, {}, true), "the first build did not stay 2D");
      Sleep(30);
      check(!r.configure(runtime, b.resource, space, {}, true) && r.preparing(), "an extent change did not stay 2D");
      until_ready(r, b, space);
      render_once(r, b);
      check(!r.configure(runtime, a.resource, space, {}, true) && r.building(), "a second extent did not build");
      Sleep(30);
      const auto back = clock::now();
      check(r.configure(runtime, b.resource, space, {}, true) && !r.building(), "a toggle back did not keep the current renderer");
      check(std::chrono::duration<double, std::milli>(clock::now() - back).count() < 50.0, "a toggle back waited for a build");
      settled();
      check(renderer::builds_started() == started + 3, "an extent change did not build exactly once per configuration");
      // An SDR and HDR toggle: the other transfer builds; the toggle back
      // reuses the cached renderer at once.
      until_ready(r, b, other_space);
      render_once(r, b);
      const auto cached = r.output();
      check(r.configure(runtime, b.resource, space, {}, true) && !r.building() && r.output().handle != cached.handle,
        "the toggle back did not reuse the cached renderer at once");
      check(renderer::builds_started() == started + 4, "the cached toggle back built again");
      r.reset_after_runtime_drain();
      std::printf("PASS %s background build: an extent change cancels the build in progress, a toggle back to the current "
        "or cached renderer cancels it or builds nothing\n", api_name);
    }
    {
      // A reset cancels the build; a park keeps it for the same configuration.
      renderer r;
      const auto started = renderer::builds_started();
      check(!r.configure(runtime, a.resource, space, {}, true) && r.building(), "a reset fixture did not start a build");
      Sleep(20);
      r.reset_after_runtime_drain();
      check(!r.building() && r.device() == nullptr, "a reset kept its build");
      until_ready(r, a, space);
      check(!r.configure(runtime, b.resource, space, {}, true) && r.building(), "a park fixture did not start a build");
      Sleep(20);
      r.park_after_runtime_drain();
      check(r.building() && !r.preparing() && r.device() == device, "a park dropped its build");
      until_ready(r, b, space);
      settled();
      check(renderer::builds_started() == started + 3, "a park did not keep its build for the same configuration");
      render_once(r, b);
      r.reset_after_runtime_drain();
      std::printf("PASS %s background build: a runtime reset cancels the build, a park keeps it for the reinitialised "
        "runtime\n", api_name);
    }
    {
      // A destroyed device waits for the step in progress and releases the
      // build; the owner then starts again.
      renderer r;
      const auto started = renderer::builds_started();
      check(!r.configure(runtime, a.resource, space, {}, true), "a teardown fixture did not start a build");
      Sleep(25);
      check(renderer::builds_in_flight() == 1, "the teardown fixture's build is not registered");
      renderer::device_destroyed(device);
      check(!renderer::builds_in_flight(), "device teardown returned while a build was still running");
      until_ready(r, a, space);
      check(renderer::builds_started() == started + 2, "a released build was not started again");
      render_once(r, a);
      r.reset_after_runtime_drain();
      std::printf("PASS %s background build: device teardown waits for the step in progress and releases the build\n", api_name);
    }
    {
      // A failed step leaves the game 2D, unavailable rather than preparing,
      // with no retry until another configuration or a reset.
      hooks("SunshineHostVerticalCS", 0, false);
      renderer r;
      const auto started = renderer::builds_started();
      const auto failed = [&](const target &t) {
        const auto start = clock::now();
        while (!r.configure(runtime, t.resource, space, {}, true) && r.preparing()) {
          check(clock::now() - start < std::chrono::seconds(30), "a failing build never ended");
          Sleep(1);
        }
        return !r.preparing() && !r.building();
      };
      check(failed(a), "a failed build was not reported unavailable");
      for (unsigned i = 0; i != 5; ++i)
        check(!r.configure(runtime, a.resource, space, {}, true) && !r.preparing(), "a failed build became available");
      check(renderer::builds_started() == started + 1, "a failed build was retried");
      check(failed(b) && renderer::builds_started() == started + 2, "another configuration did not build again");
      hooks(nullptr, 0, false);
      check(!r.configure(runtime, b.resource, space, {}, true) && renderer::builds_started() == started + 2,
        "a failed configuration was retried before a reset");
      r.reset_after_runtime_drain();
      until_ready(r, b, space);
      render_once(r, b);
      r.reset_after_runtime_drain();
      settled();
      std::printf("PASS %s background build: a failed step leaves the game 2D and unavailable, without retry until "
        "another configuration or a reset\n", api_name);
    }
    {
      // The present-thread executor (a single-threaded D3D11 device's): a
      // slice of steps per call, never registered, cancelled in place.
      hooks(nullptr, 2, true);
      renderer r;
      check(!r.configure(runtime, a.resource, space, {}, true) && r.building() && !renderer::builds_in_flight(),
        "a present-thread build was registered or finished at once");
      check(!r.configure(runtime, b.resource, space, {}, true) && r.building(), "a present-thread build was not replaced");
      const auto calls = until_ready(r, b, space);
      check(calls.count > 3 && calls.longest_ms < 40.0 && !renderer::builds_in_flight(),
        "the present-thread build did not run in bounded slices");
      render_once(r, b);
      {
        renderer abandoned;
        check(!abandoned.configure(runtime, a.resource, space, {}, true) && abandoned.building(),
          "a second present-thread build did not start");
      }
      r.reset_after_runtime_drain();
      std::printf("PASS %s background build: the present-thread executor builds over %u calls (longest %.1f ms) and ends "
        "in place when cancelled\n", api_name, calls.count, calls.longest_ms);
    }
    hooks(nullptr, 0, false);
    settled();
    queue->wait_idle();
    for (auto *t : {&a, &b}) device->destroy_resource(t->resource);
    std::printf("PASS %s background build: the game stays 2D while the renderer builds off the present thread; the ready "
      "call adopts it with the first render's objects prepared\n", api_name);
  }
}
