// SPDX-License-Identifier: GPL-3.0-only
// D3D11 companion to the native no-FX parity test. The old full-effect fixture
// supplies only the official runtime, wrapped device and readback helpers.
#define main sunshine_legacy_d3d11_fixture_main
#include "test_depth3d_runtime.cpp"
#undef main
#include "test_game3d_render_input.h"
#include "test_game3d_debug_dump_runtime.h"
#include "test_game3d_budget.h"
#include "test_game3d_ui_pin_band.h"

namespace {
  struct native11_case {
    const char *name;
    float strength;
    int view;
    bool ready;
  };
  constexpr native11_case native11_cases[] {
    {"strength50", 50, 0, true}, {"strength100", 100, 0, true},
    {"zero-strength", 0, 0, true}, {"depth-unavailable", 50, 0, false},
    {"normal-depth", 50, 2, true}, {"stereo-depth", 50, 1, true},
  };

  std::vector<std::uint8_t> source_pattern11(fixture_t &fixture) {
    D3D11_TEXTURE2D_DESC desc {};
    fixture.backbuffer->GetDesc(&desc);
    desc.BindFlags = desc.MiscFlags = 0;
    const unsigned bytes = fixture.color == 2 ? 8 : 4;
    std::vector<std::uint8_t> pattern(size_t(width) * height * bytes);
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
      const size_t pixel = size_t(y) * width + x;
      const bool stripe = ((x / 16) % 2) != (((x / 16) / 7 + (x / 16) / 13) % 2);
      if (fixture.color == 1) {
        const std::uint8_t values[4] {std::uint8_t(stripe ? 224 : 32), std::uint8_t(y < height / 3 ? 200 : 60), std::uint8_t(((x + 2 * y) / 31) % 2 ? 170 : 80), 255};
        std::memcpy(pattern.data() + pixel * 4, values, 4);
      } else if (fixture.color == 2) {
        const std::uint16_t values[4] {std::uint16_t(stripe ? 0x4000 : 0x3400), std::uint16_t(y < height / 3 ? 0x3c00 : 0x3800), std::uint16_t(((x + 2 * y) / 31) % 2 ? 0x3a00 : 0xb000), 0x3c00};
        std::memcpy(pattern.data() + pixel * 8, values, 8);
      } else {
        const std::uint32_t value = (stripe ? 700u : 256u) | ((y < height / 3 ? 600u : 300u) << 10) |
          ((((x + 2 * y) / 31) % 2 ? 500u : 200u) << 20) | (3u << 30);
        std::memcpy(pattern.data() + pixel * 4, &value, 4);
      }
    }
    const D3D11_SUBRESOURCE_DATA initial{pattern.data(), width * bytes, 0};
    checked(fixture.device->CreateTexture2D(&desc, &initial, fixture.source_pattern.put()), "Create D3D11 parity color");
    return pattern;
  }

  void check_native11(fixture_t &fixture, const fs::path &directory) {
    fixture.discover();
    require(observed.runtime->get_device()->get_api() == api::device_api::d3d11, "Native D3D11 fixture used the wrong API");
    const auto source = source_pattern11(fixture);
    std::vector<float> raw(size_t(width) * height);
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
      const bool foreground = x > width / 3 && x < width / 2 && y > height / 5 && y < height * 4 / 5;
      raw[size_t(y) * width + x] = foreground ? .085f : .0005f + .016f * float(x) / width;
    }
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = width; desc.Height = height; desc.ArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_R32_FLOAT; desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    const D3D11_SUBRESOURCE_DATA depth_data{raw.data(), width * sizeof(float), 0};
    checked(fixture.device->CreateTexture2D(&desc, &depth_data, fixture.depth.put()), "Create D3D11 parity depth");
    checked(fixture.device->CreateShaderResourceView(fixture.depth.p, nullptr, fixture.depth_view.put()), "Create D3D11 parity depth SRV");
    observed.depth_view = {reinterpret_cast<std::uint64_t>(fixture.depth_view.p)};
    observed.inject_depth = true;
    observed.depth_ready_uniforms = {fixture.uniform("Sunshine_DepthReady")};
    sunshine_game3d::render_parameters p;
    p.disparity_limit_uv = .04f; // Unclipped parity; default-cap regressions below.
    p.depth_ready = p.camera_ready = 1; p.coordinate_basis = 1;
    p.depth_scale = 128; p.strength_blend = 1; p.projection = {0, 1}; p.convergence = {.05f, 1.f / 128.f};
    fixture.set_int("Sunshine_CameraCoordinateBasis", p.coordinate_basis);
    fixture.set_float("Sunshine_CameraDepthScale", p.depth_scale);
    fixture.set_float("Sunshine_CameraStrengthBlend", p.strength_blend);
    observed.runtime->set_uniform_value_bool(fixture.uniform("Sunshine_CameraDepthReady"), true);
    observed.runtime->set_uniform_value_float(fixture.uniform("Sunshine_CameraProjection"), p.projection.data(), 2);
    observed.runtime->set_uniform_value_float(fixture.uniform("Sunshine_CameraRawDepthRange"), p.raw_depth_range.data(), 2);
    observed.runtime->set_uniform_value_float(fixture.uniform("Sunshine_CameraConvergence"), p.convergence.data(), 2);
    observed.runtime->set_uniform_value_float(fixture.uniform("Sunshine_DepthJitter"), p.jitter.data(), 2);
    observed.runtime->set_uniform_value_float(fixture.uniform("Sunshine_CameraDepthRect"), p.depth_rect.data(), 4);
    std::vector<std::vector<std::uint8_t>> reference;
    for (const auto &test : native11_cases) {
      fixture.set_float("Depth_Adjustment", test.strength); fixture.set_int("Depth_Map_View", test.view);
      observed.depth_ready = test.ready;
      fixture.measured_frames();
      reference.push_back(fixture.read(fixture.exported.p));
    }
    observed.inject_depth = observed.capture = false;
    observed.depth_ready_uniforms.clear();
    observed.runtime->update_texture_bindings("DEPTH", {}, {});
    fs::rename(directory / "effects" / effect_file, directory / "effects" / "SunshineGame3D.fx.disabled");
    const auto before_reload = observed.reloads;
    observed.runtime->reload_effect_next_frame(nullptr);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    unsigned techniques = 1;
    while ((observed.reloads == before_reload || techniques != 0) && std::chrono::steady_clock::now() < deadline) {
      fixture.step();
      techniques = 0;
      observed.runtime->enumerate_techniques(nullptr, [&](api::effect_runtime *, api::effect_technique) { ++techniques; });
    }
    require(observed.reloads > before_reload && techniques == 0, "D3D11 native phase retains compiled FX");
    for (const auto &entry : fs::recursive_directory_iterator(directory / "effects"))
      require(entry.path().extension() != ".fx", "D3D11 native phase retains installed FX");
    const auto effects = observed.renders;
    sunshine_game3d::renderer renderer;
    sunshine_game3d_test::dump_fixture dump;
    auto *queue = observed.runtime->get_command_queue();
    const api::resource backbuffer{reinterpret_cast<std::uint64_t>(fixture.backbuffer.p)};
    require(renderer.configure(observed.runtime, backbuffer, static_cast<api::color_space>(fixture.color)), "D3D11 native renderer configure failed");
    std::ofstream report(directory / "native-parity.txt");
    unsigned index = 0;
    for (const auto &test : native11_cases) {
      p.strength = test.strength; p.depth_view = test.view; p.depth_ready = test.ready;
      fixture.context->CopyResource(fixture.backbuffer.p, fixture.source_pattern.p);
      require(sunshine_game3d::test::render_frame(renderer, queue->get_immediate_command_list(), backbuffer, test.ready ? observed.depth_view : api::resource_view{}, p), "D3D11 native renderer rejected frame");
      const bool dump_case = std::strcmp(test.name,"strength50")==0 || std::strcmp(test.name,"depth-unavailable")==0;
      if (dump_case) dump.begin(observed.runtime,renderer,p,test.ready ? observed.depth_view : api::resource_view{},false,
        static_cast<api::color_space>(fixture.color));
      queue->flush_immediate_command_list(); renderer.finish_present();
      if (dump_case) dump.submitted(observed.runtime);
      queue->wait_idle();
      if (dump_case) dump.verify(observed.runtime,[&](api::resource texture,bool) {
        return fixture.read(reinterpret_cast<ID3D11Texture2D *>(texture.handle));
      }, directory / (std::string("dump-") + test.name));
      const auto output = renderer.output();
      require(output.handle != 0, "D3D11 native renderer output missing");
      com_ptr<ID3D11Texture2D> texture;
      checked(reinterpret_cast<ID3D11Resource *>(output.handle)->QueryInterface(IID_PPV_ARGS(texture.put())), "D3D11 native export is not a texture");
      const auto pixels = fixture.read(texture.p);
      require(pixels.size() == reference[index].size(), "D3D11 native/FX export sizes differ");
      double maximum = 0;
      for (size_t pixel = 0; pixel < size_t(width) * height * 2; ++pixel) for (unsigned channel = 0; channel < 4; ++channel) {
        float a, b;
        if (fixture.color == 1) {
          std::uint32_t av, bv;
          std::memcpy(&av, reference[index].data() + pixel * 4, 4); std::memcpy(&bv, pixels.data() + pixel * 4, 4);
          const unsigned mask = channel == 3 ? 3 : 1023;
          a = float((av >> (channel * 10)) & mask) / mask; b = float((bv >> (channel * 10)) & mask) / mask;
        } else {
          std::uint16_t av, bv;
          std::memcpy(&av, reference[index].data() + pixel * 8 + channel * 2, 2); std::memcpy(&bv, pixels.data() + pixel * 8 + channel * 2, 2);
          a = half_float(av); b = half_float(bv);
        }
        require(std::isfinite(a) && std::isfinite(b), "D3D11 native/FX has non-finite color");
        maximum = std::max(maximum, double(std::abs(a - b) / std::max({1.f, std::abs(a), std::abs(b)})));
      }
      report << test.name << " byte_exact=" << (pixels == reference[index]) << " max_relative_error=" << maximum << '\n';
      std::printf("MEASURE D3D11 native/FX %s byte_exact=%u max_relative_error=%.9g\n", test.name, unsigned(pixels == reference[index]), maximum);
      require(maximum <= (fixture.color == 1 ? 1.01 / 1023 : .002), "D3D11 native/FX parity exceeded export quantization");
      require(fixture.read(fixture.backbuffer.p) == source, "D3D11 native renderer changed mono game color");
      {
        // Owed conditioning (C2): with the pack owed, pack() records the
        // conditioning inside its isolated device-context state, and the
        // export is the same bytes; a mono frame records none.
        const auto before = renderer.conditioning_activity();
        renderer.begin_present();
        fixture.context->CopyResource(fixture.backbuffer.p, fixture.source_pattern.p);
        sunshine_game3d::render_frame_input frame;
        frame.color = backbuffer;
        frame.depth = test.ready ? observed.depth_view : api::resource_view{};
        frame.scene = p;
        auto *commands = queue->get_immediate_command_list();
        require(renderer.render(commands, frame, true) && renderer.pack(commands), "D3D11 native renderer rejected an owed pack");
        queue->flush_immediate_command_list(); renderer.finish_present();
        queue->wait_idle();
        require(fixture.read(texture.p) == pixels, (std::string("D3D11 owed pack differs from the immediate pack of ") + test.name).c_str());
        const auto after = renderer.conditioning_activity();
        const bool mono = !p.depth_ready || !p.camera_ready || p.depth_view == 2 || !(p.strength > 0.f) || !(p.strength_blend > 0.f);
        require(after.recorded - before.recorded == (mono ? 0u : 1u) && after.mono - before.mono == (mono ? 1u : 0u),
          (std::string("D3D11 owed pack recorded the wrong conditioning for ") + test.name).c_str());
      }
      ++index;
    }
    std::puts("PASS D3D11 owed conditioning: every parity frame packs the same bytes with its conditioning owed to pack(); mono frames record none");
    {
      // A released export ring (the exporter's release_ring) first drops the
      // renderer's cached view of each slot: a D3D11 view holds its texture,
      // so the slot's VRAM would otherwise stay allocated.
      com_ptr<ID3D11Texture2D> output;
      checked(reinterpret_cast<ID3D11Resource *>(renderer.output().handle)->QueryInterface(IID_PPV_ARGS(output.put())),
        "D3D11 native export is not a texture");
      D3D11_TEXTURE2D_DESC slot_desc {};
      output->GetDesc(&slot_desc);
      slot_desc.Usage = D3D11_USAGE_DEFAULT;
      slot_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
      slot_desc.CPUAccessFlags = slot_desc.MiscFlags = 0;
      com_ptr<ID3D11Texture2D> slot;
      checked(fixture.device->CreateTexture2D(&slot_desc, nullptr, slot.put()), "Create D3D11 export slot");
      bool destroyed = false;
      {
        com_ptr<ID3DDestructionNotifier> notifier;
        checked(slot.p->QueryInterface(IID_PPV_ARGS(notifier.put())), "D3D11 slot has no destruction notifier");
        UINT callback = 0;
        checked(notifier.p->RegisterDestructionCallback([](void *flag) { *static_cast<bool *>(flag) = true; }, &destroyed, &callback),
          "Register the D3D11 slot destruction callback");
      }
      const api::resource target{reinterpret_cast<std::uint64_t>(slot.p)};
      renderer.begin_present();
      fixture.context->CopyResource(fixture.backbuffer.p, fixture.source_pattern.p);
      sunshine_game3d::render_frame_input frame;
      frame.color = backbuffer;
      frame.depth = observed.depth_view;
      frame.scene = p;
      auto *commands = queue->get_immediate_command_list();
      require(renderer.render(commands, frame, true) && renderer.pack(commands, target, 7), "D3D11 renderer rejected a slot pack");
      queue->flush_immediate_command_list(); renderer.finish_present();
      queue->wait_idle();
      slot.reset();
      fixture.context->Flush(); queue->wait_idle();
      require(!destroyed, "The cached export view did not hold its D3D11 slot");
      renderer.release_export_view(target);
      fixture.context->Flush(); queue->wait_idle();
      require(destroyed, "A released D3D11 export slot stayed allocated through the renderer's cached view");
      std::puts("PASS D3D11 export slot release: dropping the renderer's cached view frees the slot texture");
    }
    const auto original_depth = fixture.read(fixture.depth.p);
    for (const auto &test : sunshine_game3d_test::budget_cases()) {
      const auto parameters = sunshine_game3d_test::budget_parameters(test);
      fixture.context->CopyResource(fixture.backbuffer.p, fixture.source_pattern.p);
      require(sunshine_game3d::test::render_frame(renderer, queue->get_immediate_command_list(), backbuffer, observed.depth_view, parameters),
        "D3D11 budget renderer rejected frame");
      const bool dump_case = std::strcmp(test.name, "strength100") == 0;
      if (dump_case) dump.begin(observed.runtime, renderer, parameters, observed.depth_view, false,
        static_cast<api::color_space>(fixture.color));
      queue->flush_immediate_command_list();
      renderer.finish_present();
      if (dump_case) dump.submitted(observed.runtime);
      queue->wait_idle();
      if (dump_case) {
        const auto output = directory / "dump-default-budget";
        dump.verify(observed.runtime, [&](api::resource texture, bool) {
          return fixture.read(reinterpret_cast<ID3D11Texture2D *>(texture.handle));
        }, output);
        sunshine_game3d_test::verify_budget_dump(output, parameters.disparity_limit_uv);
      }
      const auto diagnostics = renderer.diagnostics();
      require(diagnostics.candidate.handle && diagnostics.final_field.handle, "D3D11 budget diagnostics missing");
      sunshine_game3d_test::verify_budget_fields(test, width, height,
        fixture.read(reinterpret_cast<ID3D11Texture2D *>(diagnostics.candidate.handle)),
        fixture.read(reinterpret_cast<ID3D11Texture2D *>(diagnostics.final_field.handle)), report);
      require(fixture.read(fixture.depth.p) == original_depth, "D3D11 budget renderer changed raw source depth");
      require(fixture.read(fixture.backbuffer.p) == source, "D3D11 budget renderer changed mono source color");
    }
    {
      // The soft UI pin band on actual D3D11 GPU fields, through an R32 red mask.
      using namespace sunshine_game3d;
      auto band_parameters = p;
      band_parameters.strength = 100; band_parameters.depth_view = 0; band_parameters.depth_ready = 1;
      const ui_plane_parameters plane{ui_plane_mode::display_fraction, .25f};
      std::uint64_t capture = 20000;
      sunshine_game3d_test::ui_pin_band_fixture band;
      band.width = width; band.height = height;
      band.gain = float(shader_marker(renderer::shader_source(), "SUNSHINE_UI_SOFT_PIN_GAIN"));
      band.depth = [&](bool structured) {
        std::vector<float> depth(size_t(width) * height, 0.f);
        if (structured) depth = raw;
        fixture.context->UpdateSubresource(fixture.depth.p, 0, nullptr, depth.data(), width * sizeof(float), 0);
      };
      band.render = [&](const std::vector<float> *alpha) {
        api::resource_view view{};
        if (alpha) {
          view = renderer.prepare_ui_candidate(0, ++capture, [&](api::resource texture) {
            fixture.context->UpdateSubresource(reinterpret_cast<ID3D11Resource *>(texture.handle), 0, nullptr, alpha->data(),
              width * sizeof(float), 0);
            return true;
          }, api::format::r32_float);
          require(view.handle, "UI pin band D3D11 mask unavailable");
        }
        fixture.context->CopyResource(fixture.backbuffer.p, fixture.source_pattern.p);
        require(sunshine_game3d::test::render_frame(renderer, queue->get_immediate_command_list(), backbuffer, observed.depth_view,
          band_parameters, alpha != nullptr, view, plane, nullptr, nullptr, ui_mask_channel::red), "UI pin band D3D11 render failed");
        queue->flush_immediate_command_list(); renderer.finish_present(); queue->wait_idle();
        const auto bytes = fixture.read(reinterpret_cast<ID3D11Texture2D *>(renderer.diagnostics().final_field.handle));
        std::vector<float> field(bytes.size() / sizeof(float));
        std::memcpy(field.data(), bytes.data(), field.size() * sizeof(float));
        return field;
      };
      sunshine_game3d_test::check_ui_pin_band(band, report, "D3D11");
      fixture.context->UpdateSubresource(fixture.depth.p, 0, nullptr, raw.data(), width * sizeof(float), 0);
      require(fixture.read(fixture.depth.p) == original_depth, "D3D11 UI pin band fixture did not restore raw depth");
    }
    require(observed.renders == effects, "An FX technique ran during native D3D11 phase");
    queue->wait_idle(); renderer.reset_after_runtime_drain();
    require(report.good(), "Cannot write native D3D11 parity report");
    std::puts("PASS D3D11 native renderer: six frozen-FX-matched cases, no installed/loaded FX, exact native mono preservation");
  }
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 5) {
    std::fputs("usage: test_game3d_native_runtime_d3d11 official-ReShade64.dll frozen-shader-directory fresh-output-directory srgb|scrgb|pq\n", stderr);
    return 2;
  }
  std::thread([] { Sleep(120000); TerminateProcess(GetCurrentProcess(), 124); }).detach();
  try {
    require(_putenv_s("SUNSHINE_DEPTH3D_EFFECT", "SunshineGame3D") == 0, "Cannot select frozen FX oracle");
    const unsigned color = !std::strcmp(argv[4], "srgb") ? 1 : !std::strcmp(argv[4], "scrgb") ? 2 : !std::strcmp(argv[4], "pq") ? 3 : 0;
    require(color != 0, "Unknown source transfer");
    const auto directory = fs::absolute(argv[3]); require(!fs::exists(directory), "Use a fresh native parity directory");
    fixture_t fixture;
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), directory, color, 0);
    check_native11(fixture, directory);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what()); return 1;
  }
}
