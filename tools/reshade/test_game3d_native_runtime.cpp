// SPDX-License-Identifier: GPL-3.0-only
// Actual ReShade D3D12 parity: capture the frozen FX oracle, unload every effect,
// then exercise the production add-on renderer on the same inputs with no FX.
#define SUNSHINE_DEPTH_SELECTION_RUNTIME
#include "test_depth3d_runtime_d3d12.cpp"
#include "game3d_renderer.h"
#include "test_game3d_debug_dump_runtime.h"
#include "test_game3d_budget.h"

namespace {
  struct native_case {
    std::string name;
    sunshine_game3d::render_parameters parameters;
    unsigned depth_width = 0, depth_height = 0;
    bool reverse = false;
  };

  std::vector<native_case> native_cases() {
    sunshine_game3d::render_parameters p;
    // Frozen-FX parity covers the unchanged warp below the new display cap.
    // The default cap and hostile stale gains are exercised independently below.
    p.disparity_limit_uv = .04f;
    p.depth_ready = p.camera_ready = 1;
    p.coordinate_basis = 1;
    p.depth_scale = 128.f;
    p.strength_blend = 1.f;
    p.projection = {0.f, 1.f};
    p.convergence = {.05f, 1.f / 128.f};
    std::vector<native_case> result;
    const auto add = [&](const char *name, const sunshine_game3d::render_parameters &parameters) {
      result.push_back({name, parameters, width, height, false});
    };
    add("raw-cliffs-strength50", p);
    p.strength = 100; add("strength100", p);
    p.strength = 0; add("zero-strength", p);
    p.strength = 50; p.depth_ready = 0; add("depth-unavailable", p);
    p.depth_ready = 1; p.camera_ready = 0; add("camera-unavailable", p);
    p.camera_ready = 1; p.strength_blend = 0; add("transition-mono", p);
    p.strength_blend = .35f; add("transition-partial", p);
    p.strength_blend = 1; p.depth_view = 2; add("normal-depth", p);
    p.depth_view = 1; add("stereo-depth", p);
    p.depth_view = 0; p.projection = {1, -1};
    add("reversed-raw", p); result.back().reverse = true;
    p.coordinate_basis = 0; p.projection = {0, 16.666666f};
    p.depth_scale = 7.68f; p.convergence = {.05f, .13020833f};
    add("projection-depth", p);
    p.coordinate_basis = 1; p.projection = {0, 1};
    p.depth_scale = 128; p.convergence = {.05f, 1.f / 128.f};
    const unsigned dw = width / 2 + 64, dh = height / 2 + 32;
    p.depth_rect = {16.f / dw, 8.f / dh, float(dw - 32) / dw, float(dh - 16) / dh};
    p.jitter = {.35f / dw, -.25f / dh};
    result.push_back({"cropped-lowres-jitter", p, dw, dh, false});
    p.depth_rect = {0, 0, 1, 1}; p.jitter = {0, 0};
    add("fullres-recovered", p);
    return result;
  }

  void prepare_depth(fixture_t &fixture, const native_case &test) {
    const auto current = fixture.depth->GetDesc();
    if (current.Width != test.depth_width || current.Height != test.depth_height)
      fixture.replace_depth(test.depth_width, test.depth_height);
    std::vector<float> depth(size_t(test.depth_width) * test.depth_height);
    for (unsigned y = 0; y < test.depth_height; ++y) {
      for (unsigned x = 0; x < test.depth_width; ++x) {
        const bool foreground = x > test.depth_width / 3 && x < test.depth_width / 2 &&
          y > test.depth_height / 5 && y < test.depth_height * 4 / 5;
        const bool thin = x > test.depth_width * 2 / 3 && x < test.depth_width * 2 / 3 + 3;
        const float raw = foreground || thin ? .085f : .0005f + .016f * float(x) / test.depth_width;
        depth[size_t(y) * test.depth_width + x] = test.reverse ? 1.f - raw : raw;
      }
    }
    fixture.upload_depth(depth);
  }

  void set_fx_parameters(fixture_t &fixture, const sunshine_game3d::render_parameters &p) {
    fixture.set_float("Depth_Adjustment", p.strength);
    fixture.set_int("Depth_Map_View", p.depth_view);
    fixture.set_int("Sunshine_CameraCoordinateBasis", p.coordinate_basis);
    fixture.set_float("Sunshine_CameraDepthScale", p.depth_scale);
    fixture.set_float("Sunshine_CameraStrengthBlend", p.strength_blend);
    observed.runtime->set_uniform_value_bool(fixture.uniform("Sunshine_CameraDepthReady"), p.camera_ready != 0);
    for (const auto &input : std::array<std::pair<const char *, const float *>, 4> {{
      {"Sunshine_CameraProjection", p.projection.data()}, {"Sunshine_CameraRawDepthRange", p.raw_depth_range.data()},
      {"Sunshine_CameraConvergence", p.convergence.data()}, {"Sunshine_DepthJitter", p.jitter.data()},
    }}) observed.runtime->set_uniform_value_float(fixture.uniform(input.first), input.second, 2);
    observed.runtime->set_uniform_value_float(fixture.uniform("Sunshine_CameraDepthRect"), p.depth_rect.data(), 4);
    observed.ready = p.depth_ready != 0;
  }

  std::vector<std::uint8_t> load_pixels(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "Missing frozen FX pixels");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  }

  void compare_pixels(const native_case &test, unsigned color,
      const std::vector<std::uint8_t> &reference, const std::vector<std::uint8_t> &native, std::ostream &report) {
    require(reference.size() == native.size(), "Native/FX export byte sizes differ");
    double maximum = 0;
    size_t different = 0;
    for (size_t index = 0; index < reference.size(); ++index) different += reference[index] != native[index];
    const size_t pixels = size_t(width) * height * 2;
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
      for (unsigned channel = 0; channel < 4; ++channel) {
        float a, b;
        if (color == 1) {
          std::uint32_t av, bv;
          std::memcpy(&av, reference.data() + pixel * 4, 4);
          std::memcpy(&bv, native.data() + pixel * 4, 4);
          const unsigned mask = channel == 3 ? 3 : 1023;
          a = float((av >> (channel * 10)) & mask) / mask;
          b = float((bv >> (channel * 10)) & mask) / mask;
        } else {
          std::uint16_t av, bv;
          std::memcpy(&av, reference.data() + pixel * 8 + channel * 2, 2);
          std::memcpy(&bv, native.data() + pixel * 8 + channel * 2, 2);
          a = half_float(av); b = half_float(bv);
        }
        require(std::isfinite(a) && std::isfinite(b), "Native/FX parity contains non-finite color");
        maximum = std::max(maximum, double(std::abs(a - b) / std::max({1.f, std::abs(a), std::abs(b)})));
      }
    }
    report << test.name << " different_bytes=" << different << " maximum_relative_error=" << maximum << '\n';
    std::printf("MEASURE native/FX %s differing_bytes=%zu max_relative_error=%.9g\n", test.name.c_str(), different, maximum);
    // Native HLSL may reorder ordinary (non-precise) arithmetic compared with
    // ReShade's generated HLSL. Permit only export quantization, across all pixels.
    require(maximum <= (color == 1 ? 1.01 / 1023 : .002), "Native GPU rendering differs from frozen FX beyond export quantization");
  }

  void write_native_source(fixture_t &fixture, ID3D12Resource *backbuffer) {
    fixture.begin_commands();
    transition(fixture.commands.p, backbuffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION from {}, to {};
    from.pResource = fixture.source_upload.p;
    from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    from.PlacedFootprint = fixture.source_footprint;
    to.pResource = backbuffer;
    to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    fixture.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    transition(fixture.commands.p, backbuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
    fixture.submit();
  }

  void check_native_parity(fixture_t &fixture, const fs::path &directory) {
    fixture.discover();
    const auto tests = native_cases();
    const auto results = directory / "native-parity";
    fs::create_directories(results);
    std::ofstream report(results / "measurements.txt");
    for (const auto &test : tests) {
      prepare_depth(fixture, test);
      set_fx_parameters(fixture, test.parameters);
      fixture.measured_frames();
      const auto pixels = fixture.read(fixture.exported.p);
      sunshine_parity::write_bytes(results / (test.name + ".fx.bin"), pixels.data(), pixels.size());
    }

    // No effect is allowed to provide resources, uniforms or begin/finish events
    // to the native path. Keep the isolated source as a renamed test artifact.
    observed.inject = observed.capture = false;
    observed.ready_uniforms.clear();
    observed.runtime->update_texture_bindings("DEPTH", {}, {});
    observed.bound_depth_view = {};
    fs::rename(directory / "effects" / effect_file, directory / "effects" / "SunshineGame3D.fx.disabled");
    const unsigned previous_reloads = observed.reloads;
    observed.runtime->reload_effect_next_frame(nullptr);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    unsigned techniques = 1;
    while ((observed.reloads == previous_reloads || techniques != 0) && std::chrono::steady_clock::now() < deadline) {
      fixture.step();
      techniques = 0;
      observed.runtime->enumerate_techniques(nullptr, [&](api::effect_runtime *, api::effect_technique) { ++techniques; });
    }
    require(observed.reloads > previous_reloads && techniques == 0, "Native fixture still has compiled FX techniques");
    for (const auto &entry : fs::recursive_directory_iterator(directory / "effects"))
      require(entry.path().extension() != ".fx", "Native fixture still has an installed FX shader");
    const auto effect_renders = observed.renders;
    std::puts("PASS native phase has zero installed/loaded FX techniques");

    sunshine_game3d::renderer renderer;
    sunshine_game3d_test::dump_fixture dump;
    auto *owner_queue = observed.runtime->get_command_queue();
    for (const auto &test : tests) {
      prepare_depth(fixture, test);
      auto *backbuffer = fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
      write_native_source(fixture, backbuffer);
      const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
      require(renderer.configure(observed.runtime, source, static_cast<api::color_space>(fixture.color)), "Native renderer configure failed");
      require(renderer.render(owner_queue->get_immediate_command_list(), source,
        test.parameters.depth_ready ? fixture.depth_view : api::resource_view{}, test.parameters), "Native renderer rejected ready parity frame");
      const bool dump_case = test.name == "raw-cliffs-strength50" || test.name == "depth-unavailable" || test.name == "cropped-lowres-jitter";
      if (dump_case) dump.begin(observed.runtime,renderer,test.parameters,
        test.parameters.depth_ready ? fixture.depth_view : api::resource_view{},test.name == "cropped-lowres-jitter",
        static_cast<api::color_space>(fixture.color));
      owner_queue->flush_immediate_command_list();
      renderer.finish_present();
      if (dump_case) dump.submitted(observed.runtime);
      owner_queue->wait_idle(); // Test-only readback; production remains asynchronous.
      if (dump_case) dump.verify(observed.runtime,[&](api::resource texture,bool common) {
        return fixture.read(reinterpret_cast<ID3D12Resource *>(texture.handle),common ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      }, directory / ("dump-" + test.name));
      const auto output = renderer.output();
      require(output.handle != 0, "Native renderer has no packed output");
      auto *texture = reinterpret_cast<ID3D12Resource *>(output.handle);
      const auto desc = texture->GetDesc();
      require(desc.Width == width * 2 && desc.Height == height &&
        desc.Format == (fixture.color == 1 ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R16G16B16A16_FLOAT),
        "Native packed dimensions/format changed");
      const auto pixels = fixture.read(texture);
      sunshine_parity::write_bytes(results / (test.name + ".native.bin"), pixels.data(), pixels.size());
      compare_pixels(test, fixture.color, load_pixels(results / (test.name + ".fx.bin")), pixels, report);
      require(fixture.read(backbuffer, D3D12_RESOURCE_STATE_PRESENT) == fixture.source_bytes,
        "Native renderer changed the game's mono backbuffer");
    }
    // Exercise the production cap independently of the deliberately uncapped
    // frozen-FX oracle. Read both the candidate and authoritative final field.
    prepare_depth(fixture, {"budget-depth", {}, width, height, false});
    const auto original_depth = fixture.read(fixture.depth.p);
    for (const auto &test : sunshine_game3d_test::budget_cases()) {
      const auto parameters = sunshine_game3d_test::budget_parameters(test);
      auto *backbuffer = fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
      write_native_source(fixture, backbuffer);
      const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
      require(renderer.configure(observed.runtime, source, static_cast<api::color_space>(fixture.color)), "Budget renderer configure failed");
      require(renderer.render(owner_queue->get_immediate_command_list(), source, fixture.depth_view, parameters),
        "Budget renderer rejected frame");
      const bool dump_case = std::strcmp(test.name, "strength100") == 0;
      if (dump_case) dump.begin(observed.runtime, renderer, parameters, fixture.depth_view, false,
        static_cast<api::color_space>(fixture.color));
      owner_queue->flush_immediate_command_list();
      renderer.finish_present();
      if (dump_case) dump.submitted(observed.runtime);
      owner_queue->wait_idle();
      if (dump_case) {
        const auto output = directory / "dump-default-budget";
        dump.verify(observed.runtime, [&](api::resource texture, bool common) {
          return fixture.read(reinterpret_cast<ID3D12Resource *>(texture.handle),
            common ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        }, output);
        sunshine_game3d_test::verify_budget_dump(output, parameters.disparity_limit_uv);
      }
      const auto diagnostics = renderer.diagnostics();
      require(diagnostics.candidate.handle && diagnostics.final_field.handle, "Budget field diagnostics missing");
      sunshine_game3d_test::verify_budget_fields(test, width, height,
        fixture.read(reinterpret_cast<ID3D12Resource *>(diagnostics.candidate.handle)),
        fixture.read(reinterpret_cast<ID3D12Resource *>(diagnostics.final_field.handle)), report);
      require(fixture.read(fixture.depth.p) == original_depth, "Budget renderer changed raw source depth");
      require(fixture.read(backbuffer, D3D12_RESOURCE_STATE_PRESENT) == fixture.source_bytes,
        "Budget renderer changed mono source color");
    }
    // An enabled source-alpha frame must retain its separate b1 selection in
    // the real cross-API dump, not merely in the offline package parser.
    {
      auto parameters = tests.front().parameters;
      auto *backbuffer = fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
      write_native_source(fixture, backbuffer);
      const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
      require(renderer.render(owner_queue->get_immediate_command_list(), source, fixture.depth_view, parameters, true),
        "UI-protected D3D12 render failed");
      require(renderer.consumed_source_alpha_ui(), "Renderer lost enabled UI source");
      dump.begin(observed.runtime, renderer, parameters, fixture.depth_view, false, static_cast<api::color_space>(fixture.color));
      owner_queue->flush_immediate_command_list();
      renderer.finish_present(); dump.submitted(observed.runtime); owner_queue->wait_idle();
      dump.verify(observed.runtime, [&](api::resource texture, bool common) {
        return fixture.read(reinterpret_cast<ID3D12Resource *>(texture.handle),
          common ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      }, directory / "dump-source-alpha-ui");
      const auto field = fixture.read(reinterpret_cast<ID3D12Resource *>(renderer.diagnostics().final_field.handle));
      require(std::all_of(field.begin(), field.end(), [](std::uint8_t v) { return v == 0; }),
        "Fully opaque UI source was not pinned through the D3D12 b1 binding");
      require(fixture.read(backbuffer, D3D12_RESOURCE_STATE_PRESENT) == fixture.source_bytes,
        "UI protection changed the original color allocation");
    }
    // Actual D3D12 u4/u5 -> t8/t9 reduction and UI pinning. The fixture source
    // is fully opaque, so every final-field texel must use one global plane.
    // Exact binary depths put the maximum in the partial bottom-right tile.
    struct nearest_case {
      const char *name;
      float maximum, floor, strength, blend;
      bool reverse;
    };
    const std::array<nearest_case, 3> nearest_tests {{
      {"normal", .75f, .25f, 100.f, 1.f, false},
      {"reversed", .625f, .25f, 100.f, 1.f, true},
      {"floor-partial-strength", .5f, .875f, 50.f, .5f, false},
    }};
    for (const auto &test : nearest_tests) {
      std::vector<float> raw(size_t(width) * height, .125f);
      raw[size_t(height / 2) * width + width / 2] = .375f;
      raw.back() = test.maximum;
      if (test.reverse) for (auto &value : raw) value = 1.f - value;
      fixture.upload_depth(raw);
      const auto frozen_depth = fixture.read(fixture.depth.p);
      auto parameters = tests.front().parameters;
      parameters.coordinate_basis = 0;
      parameters.strength = test.strength;
      parameters.strength_blend = test.blend;
      parameters.depth_scale = 432.f / height;
      parameters.convergence = {.05f, .125f};
      parameters.projection = test.reverse ? std::array<float, 2>{1.f, -1.f} : std::array<float, 2>{0.f, 1.f};
      const sunshine_game3d::ui_plane_parameters plane {
        sunshine_game3d::ui_plane_mode::depth_midpoint_nearest_ui, test.floor};
      auto *backbuffer = fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
      write_native_source(fixture, backbuffer);
      const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
      require(renderer.render(owner_queue->get_immediate_command_list(), source, fixture.depth_view,
        parameters, true, {}, plane), "Nearest-UI D3D12 render failed");
      require(sunshine_game3d::ui_parameter_words(true, renderer.consumed_ui_plane()) ==
          sunshine_game3d::ui_parameter_words(true, plane), "Nearest-UI D3D12 changed submitted floor/mode bits");
      const bool dump_case = !test.reverse && test.maximum == .75f;
      if (dump_case) dump.begin(observed.runtime, renderer, parameters, fixture.depth_view, false,
        static_cast<api::color_space>(fixture.color));
      owner_queue->flush_immediate_command_list();
      renderer.finish_present();
      if (dump_case) dump.submitted(observed.runtime);
      owner_queue->wait_idle(); // Test-only GPU evidence; no production readback.
      const auto resources = renderer.diagnostics();
      require(resources.ui_plane_tiles.handle && resources.ui_plane_resolved.handle && resources.final_field.handle,
        "Nearest-UI D3D12 omitted current-render reduction resources");
      auto *tiles = reinterpret_cast<ID3D12Resource *>(resources.ui_plane_tiles.handle);
      auto *resolved = reinterpret_cast<ID3D12Resource *>(resources.ui_plane_resolved.handle);
      const auto tiles_desc = tiles->GetDesc(), resolved_desc = resolved->GetDesc();
      require(tiles_desc.Width == (width + 15) / 16 && tiles_desc.Height == (height + 15) / 16 &&
          tiles_desc.Format == DXGI_FORMAT_R32_FLOAT && resolved_desc.Width == 1 && resolved_desc.Height == 1 &&
          resolved_desc.Format == DXGI_FORMAT_R32_FLOAT, "Nearest-UI D3D12 reduction dimensions/format changed");
      const auto tile_bytes = fixture.read(tiles), scalar_bytes = fixture.read(resolved);
      require(tile_bytes.size() == size_t(tiles_desc.Width) * tiles_desc.Height * sizeof(float) &&
          scalar_bytes.size() == sizeof(float), "Nearest-UI D3D12 readback lost float32 reduction values");
      float tile_maximum = 0, resolved_q = 0;
      for (size_t offset = 0; offset != tile_bytes.size(); offset += sizeof(float)) {
        float value;
        std::memcpy(&value, tile_bytes.data() + offset, sizeof(value));
        require(std::isfinite(value) && value >= 0 && value <= test.maximum,
          "Nearest-UI D3D12 tile contained invalid decoded depth");
        tile_maximum = std::max(tile_maximum, value);
      }
      std::memcpy(&resolved_q, scalar_bytes.data(), sizeof(resolved_q));
      const float expected_q = std::max(test.floor, test.maximum);
      require(tile_maximum == test.maximum && resolved_q == expected_q,
        "Nearest-UI D3D12 missed current corner depth, reversed decoding or midpoint floor");

      // Closed-form projection of a constant plane, not a simulated warp.
      const double strength = double(parameters.strength) * .01 * parameters.strength_blend;
      const double unit_uv = double(height) * 100. / (2160. * width);
      const double displacement = double(parameters.convergence[0]) * parameters.depth_scale *
        (double(expected_q) - parameters.convergence[1]) * strength * unit_uv;
      const double budget = double(parameters.disparity_limit_uv) * strength;
      const double expected_parallax = std::clamp(std::clamp(displacement, -2.5 * unit_uv, 1.5 * unit_uv), -budget, budget);
      const float rounded = static_cast<float>(expected_parallax);
      const double tolerance = 8. * (double(std::nextafter(rounded, std::numeric_limits<float>::infinity())) - rounded);
      const auto field = fixture.read(reinterpret_cast<ID3D12Resource *>(resources.final_field.handle));
      require(field.size() == size_t(width) * height * sizeof(float), "Nearest-UI final field is not full-resolution R32");
      float first = 0;
      for (size_t offset = 0; offset != field.size(); offset += sizeof(float)) {
        float actual;
        std::memcpy(&actual, field.data() + offset, sizeof(actual));
        require(std::isfinite(actual) && std::abs(double(actual) - expected_parallax) <= tolerance,
          "Nearest-UI D3D12 final field did not use the GPU-resolved global plane");
        if (!offset) first = actual;
        else require(actual == first, "Nearest-UI D3D12 gave different pixels different planes");
      }
      if (dump_case) {
        const auto destination = directory / "dump-source-alpha-nearest-ui";
        dump.verify(observed.runtime, [&](api::resource texture, bool common) {
          return fixture.read(reinterpret_cast<ID3D12Resource *>(texture.handle),
            common ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        }, destination);
        std::ifstream input(destination / "manifest.json");
        require(input.good(), "Nearest-UI D3D12 dump manifest missing");
        const auto manifest = nlohmann::json::parse(input);
        const auto &replay = manifest.at("producer_metadata").at("replay");
        require(replay.at("ui_parameter_abi") == "sunshine_game3d.ui_parameters.v3" &&
            replay.at("ui_constant_binding").at("mode") == 2 &&
            replay.at("ui_constant_binding").at("inverse_depth") == test.floor &&
            replay.at("ui_plane_resolution").at("reduction_ran") == true &&
            replay.at("ui_plane_resolution").at("resolved_value").is_null(),
          "Nearest-UI D3D12 dump confused submitted floor with asynchronous GPU result");
        bool tile_pass = false, reduce_pass = false, field_pass = false;
        for (const auto &pass : replay.at("passes")) {
          if (pass.at("entry") == "SunshineUINearestTilesCS") tile_pass = pass.at("enabled") == true &&
            pass.at("uavs").at("u4") == "ui_plane_tiles:R32_FLOAT";
          if (pass.at("entry") == "SunshineUINearestReduceCS") reduce_pass = pass.at("enabled") == true &&
            pass.at("srvs").at("t8") == "ui_plane_tiles" && pass.at("uavs").at("u5") == "ui_plane_resolved:R32_FLOAT";
          if (pass.at("entry") == "SunshineHostHorizontalCS") field_pass = pass.at("srvs").at("t9") == "ui_plane_resolved";
        }
        require(tile_pass && reduce_pass && field_pass, "Nearest-UI D3D12 dump lost actual u4/u5/t8/t9 bindings");
      }
      require(fixture.read(fixture.depth.p) == frozen_depth, "Nearest-UI reduction changed source depth");
      require(fixture.read(backbuffer, D3D12_RESOURCE_STATE_PRESENT) == fixture.source_bytes,
        "Nearest-UI D3D12 changed source color");
      report << "nearest-ui " << test.name << " q=" << resolved_q << " floor=" << test.floor <<
        " rigid_parallax_uv=" << first << " expected_uv=" << expected_parallax << '\n';
      std::printf("MEASURE nearest-ui D3D12 %s q=%.9g floor=%.9g rigid_parallax_uv=%.9g expected_uv=%.9g\n",
        test.name, resolved_q, test.floor, first, expected_parallax);
    }
    // Actual D3D12 front-limit placement is stateless. Vary the inputs that
    // moved mode2 while requiring one identical UI field at the current cap.
    struct front_case {
      const char *name;
      float maximum, gain, zero, strength, blend, budget;
      bool depth_ready = true, camera_ready = true;
    };
    const std::array<front_case, 12> front_tests {{
      {"base", .25f, 8.f, .125f, 100.f, 1.f, .01f},
      {"depth-changed", .875f, 8.f, .125f, 100.f, 1.f, .01f},
      {"gain-changed", .875f, 128.f, .125f, 100.f, 1.f, .01f},
      {"zero-changed", .875f, 128.f, .75f, 100.f, 1.f, .01f},
      {"strength50", .875f, 128.f, .75f, 50.f, 1.f, .01f},
      {"blend50", .875f, 128.f, .75f, 100.f, .5f, .01f},
      {"strength50-blend50", .875f, 128.f, .75f, 50.f, .5f, .01f},
      {"smaller-budget", .875f, 128.f, .75f, 100.f, 1.f, .003f},
      {"zero-strength", .875f, 128.f, .75f, 0.f, 1.f, .01f},
      {"transition-mono", .875f, 128.f, .75f, 100.f, 0.f, .01f},
      {"no-depth", .875f, 128.f, .75f, 100.f, 1.f, .01f, false, true},
      {"no-camera", .875f, 128.f, .75f, 100.f, 1.f, .01f, true, false},
    }};
    const sunshine_game3d::ui_plane_parameters front_plane{sunshine_game3d::ui_plane_mode::front_limit, 0.f};
    std::vector<std::uint8_t> first_front_field, first_scene_candidate;
    bool scene_changed = false;
    for (size_t case_index = 0; case_index != front_tests.size(); ++case_index) {
      const auto &test = front_tests[case_index];
      std::vector<float> raw(size_t(width) * height, .0625f);
      for (unsigned y = height / 4; y != height * 3 / 4; ++y)
        for (unsigned x = width / 3; x != width * 2 / 3; ++x) raw[size_t(y) * width + x] = test.maximum;
      raw.back() = test.maximum;
      fixture.upload_depth(raw);
      const auto frozen_depth = fixture.read(fixture.depth.p);
      auto parameters = tests.front().parameters;
      parameters.coordinate_basis = 0;
      parameters.projection = {0.f, 1.f};
      parameters.depth_scale = test.gain;
      parameters.convergence = {.05f, test.zero};
      parameters.strength = test.strength;
      parameters.strength_blend = test.blend;
      parameters.disparity_limit_uv = test.budget;
      parameters.depth_ready = test.depth_ready;
      parameters.camera_ready = test.camera_ready;
      auto *backbuffer = fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
      const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
      const auto render = [&](bool protect, const sunshine_game3d::ui_plane_parameters &plane) {
        write_native_source(fixture, backbuffer);
        require(renderer.render(owner_queue->get_immediate_command_list(), source, fixture.depth_view,
          parameters, protect, {}, plane), "Front-limit D3D12 render failed");
        const bool dump_case = protect && case_index == 0 && plane.mode == sunshine_game3d::ui_plane_mode::front_limit;
        if (dump_case) dump.begin(observed.runtime, renderer, parameters, fixture.depth_view, false,
          static_cast<api::color_space>(fixture.color));
        owner_queue->flush_immediate_command_list();
        renderer.finish_present();
        if (dump_case) dump.submitted(observed.runtime);
        owner_queue->wait_idle(); // Test-only GPU evidence; no production readback.
        const auto resources = renderer.diagnostics();
        require(resources.candidate.handle && resources.vertical_field.handle && resources.final_field.handle &&
            !resources.ui_plane_tiles.handle && !resources.ui_plane_resolved.handle,
          "Front-limit D3D12 used stale/active nearest-depth reduction resources");
        if (dump_case) {
          const auto destination = directory / "dump-source-alpha-front-ui";
          dump.verify(observed.runtime, [&](api::resource texture, bool common) {
            return fixture.read(reinterpret_cast<ID3D12Resource *>(texture.handle),
              common ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
          }, destination);
          std::ifstream input(destination / "manifest.json");
          require(input.good(), "Front-limit D3D12 dump manifest missing");
          const auto manifest = nlohmann::json::parse(input);
          const auto &replay = manifest.at("producer_metadata").at("replay");
          require(replay.at("ui_parameter_abi") == "sunshine_game3d.ui_parameters.v4" &&
              replay.at("ui_constant_binding").at("mode") == 3 &&
              replay.at("ui_constant_binding").at("mode_name") == "front_limit" &&
              replay.at("ui_constant_binding").at("inverse_depth") == 0.f &&
              replay.at("ui_plane_resolution").at("policy") == "fixed_current_front_limit" &&
              replay.at("ui_plane_resolution").at("inverse_depth_role") == "unused" &&
              replay.at("ui_plane_resolution").at("reduction_ran") == false,
            "Front-limit D3D12 dump did not preserve its stateless display-cap policy");
          for (const auto &pass : replay.at("passes"))
            if (pass.at("entry") == "SunshineUINearestTilesCS" || pass.at("entry") == "SunshineUINearestReduceCS")
              require(pass.at("enabled") == false, "Front-limit dump incorrectly enabled nearest-depth reduction");
        }
        return resources;
      };
      const auto read = [&](api::resource resource) {
        return fixture.read(reinterpret_cast<ID3D12Resource *>(resource.handle));
      };
      const auto unprotected = render(false, front_plane);
      const auto candidate = read(unprotected.candidate), vertical = read(unprotected.vertical_field);
      if (case_index == 0) {
        first_scene_candidate = candidate;
        const auto disabled_field = read(unprotected.final_field), disabled_sbs = read(unprotected.sbs);
        const auto screen = render(false, {});
        require(read(screen.final_field) == disabled_field && read(screen.sbs) == disabled_sbs,
          "Disabled front-limit metadata changed ordinary scene rendering");
      } else if (case_index < 4) scene_changed = scene_changed || candidate != first_scene_candidate;
      const auto protected_resources = render(true, front_plane);
      require(sunshine_game3d::ui_parameter_words(true, renderer.consumed_ui_plane()) ==
          sunshine_game3d::ui_parameter_words(true, front_plane), "Front-limit D3D12 changed submitted mode/unused-depth bits");
      require(read(protected_resources.candidate) == candidate && read(protected_resources.vertical_field) == vertical,
        "Front-limit UI placement changed scene candidate or vertical geometry");
      const double expected = test.depth_ready && test.camera_ready ?
        double(test.budget) * (double(test.strength) / 100.) * test.blend : 0.;
      const float rounded = static_cast<float>(expected);
      const double tolerance = expected == 0 ? 0. :
        4. * (double(std::nextafter(rounded, std::numeric_limits<float>::infinity())) - rounded);
      const auto field = read(protected_resources.final_field);
      require(field.size() == size_t(width) * height * sizeof(float), "Front-limit field lost full-resolution R32 storage");
      float first = 0;
      for (size_t offset = 0; offset != field.size(); offset += sizeof(float)) {
        float value;
        std::memcpy(&value, field.data() + offset, sizeof(value));
        require(std::isfinite(value) && std::abs(double(value) - expected) <= tolerance,
          "Front-limit D3D12 did not apply the current positive display cap/strength/mono state");
        if (!offset) first = value;
        else require(value == first, "Opaque UI did not form one rigid front-limit plane");
      }
      if (case_index == 0) first_front_field = field;
      else if (case_index < 4) require(field == first_front_field,
        "Fixed front UI moved when only current depth, scene gain or zero changed");
      require(fixture.read(fixture.depth.p) == frozen_depth, "Front-limit UI changed source depth");
      require(fixture.read(backbuffer, D3D12_RESOURCE_STATE_PRESENT) == fixture.source_bytes,
        "Front-limit UI changed the game's source color");
      report << "front-ui " << test.name << " rigid_parallax_uv=" << first << " expected_uv=" << expected << '\n';
      std::printf("MEASURE front-ui D3D12 %s rigid_parallax_uv=%.9g expected_uv=%.9g\n", test.name, first, expected);
    }
    require(scene_changed, "Front-limit invariance fixture did not actually change the scene geometry");
    std::puts("PASS fixed-front D3D12 UI remains invariant across depth/gain/zero; current strength/blend/budget/mono and unchanged scene/input fields verified");
    std::vector<float> restored_depth(size_t(width) * height);
    require(original_depth.size() == restored_depth.size() * sizeof(float), "Original depth readback size changed");
    std::memcpy(restored_depth.data(), original_depth.data(), original_depth.size());
    fixture.upload_depth(restored_depth);
    std::puts("PASS nearest-UI D3D12 current GPU scalar, corner coverage, both depth directions, floor/strength and v3 dump bindings");
    require(observed.renders == effect_renders, "An FX technique ran during native parity");
    owner_queue->wait_idle();
    renderer.reset_after_runtime_drain();
    require(report.good(), "Could not write native parity report");
    std::printf("PASS native D3D12 renderer: %zu FX-matched cases; no installed FX; HDR/SDR, depth loss, both diagnostics, camera/raw modes, jitter/crop and full-resolution recovery\n", tests.size());
  }
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 5 && argc != 7) {
    std::fputs("usage: test_game3d_native_runtime official-ReShade64.dll frozen-shader-directory fresh-output-directory srgb|scrgb|pq [width height]\n", stderr);
    return 2;
  }
  std::thread([] {
    Sleep(180000);
    std::fputs("FAIL native Game 3D parity watchdog\n", stderr);
    TerminateProcess(GetCurrentProcess(), 124);
  }).detach();
  try {
    require(_putenv_s("SUNSHINE_DEPTH3D_EFFECT", "SunshineGame3D") == 0, "Cannot select frozen oracle");
    const unsigned color = !std::strcmp(argv[4], "srgb") ? 1 : !std::strcmp(argv[4], "scrgb") ? 2 : !std::strcmp(argv[4], "pq") ? 3 : 0;
    require(color != 0, "Unknown native fixture source color");
    if (argc == 7) {
      width = unsigned(std::stoul(argv[5])); height = unsigned(std::stoul(argv[6]));
      require(width >= 640 && height >= 360 && width <= 3840 && height <= 3840 && width % 2 == 0 && height % 2 == 0,
        "Native parity source must have even supported dimensions");
    }
    const auto directory = fs::absolute(argv[3]);
    require(!fs::exists(directory), "Native parity requires a fresh output directory");
    fixture_t fixture;
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), directory, color, 0);
    check_native_parity(fixture, directory);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
