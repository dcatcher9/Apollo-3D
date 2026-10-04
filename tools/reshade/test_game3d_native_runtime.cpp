// SPDX-License-Identifier: GPL-3.0-only
// Actual ReShade D3D12 parity: capture the frozen FX oracle, unload every effect,
// then exercise the production add-on renderer on the same inputs with no FX.
#define SUNSHINE_DEPTH_SELECTION_RUNTIME
#include "test_depth3d_runtime_d3d12.cpp"
#include "test_game3d_render_input.h"
#include "test_game3d_debug_dump_runtime.h"
#include "test_game3d_budget.h"
#include "test_game3d_ui_pin_band.h"
#include "game3d_ui_change_set.h"
#include "game3d_ui_darkening.h"
#include "game3d_ui_layer.h"
#include <d3d12sdklayers.h>
#include <map>
#include <set>
#include <atomic>

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

  void check_alpha_auto_d3d12(fixture_t &fixture, sunshine_game3d::renderer &renderer, std::ostream &report) {
    using namespace sunshine_game3d;
    auto *queue = observed.runtime->get_command_queue();
    auto *backbuffer = fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
    const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
    const auto original = fixture.source_bytes;
    const auto parameters = native_cases().front().parameters;
    const ui_plane_parameters plane{ui_plane_mode::shallow_front, 0.f};
    const auto upload = [&] {
      void *mapped = nullptr;
      const D3D12_RANGE no_read{0, 0};
      checked(fixture.source_upload->Map(0, &no_read, &mapped), "Map D3D12 alpha fixture source");
      const auto row = size_t(width) * bytes_per_pixel(fixture.source_format);
      for (unsigned y = 0; y < height; ++y)
        std::memcpy(static_cast<std::uint8_t *>(mapped) + fixture.source_footprint.Offset +
          size_t(y) * fixture.source_footprint.Footprint.RowPitch, fixture.source_bytes.data() + size_t(y) * row, row);
      fixture.source_upload->Unmap(0, nullptr);
    };
    // 0 = full, 1 = selective, 2 = empty. RGB is identical in every case.
    const auto pattern = [&](unsigned mask) {
      fixture.source_bytes = original;
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const bool covered = mask != 2 && (mask == 0 ||
          (x >= width / 3 && x < width / 2 && y >= height / 4 && y < height * 3 / 4));
        const auto pixel = size_t(y) * width + x;
        if (fixture.color == 1) fixture.source_bytes[pixel * 4 + 3] = covered ? 255 : 0;
        else if (fixture.color == 2) {
          const std::uint16_t alpha = covered ? 0x3c00 : 0;
          std::memcpy(fixture.source_bytes.data() + pixel * 8 + 6, &alpha, sizeof(alpha));
        } else {
          std::uint32_t packed = 0;
          std::memcpy(&packed, fixture.source_bytes.data() + pixel * 4, sizeof(packed));
          packed = (packed & 0x3fffffffu) | (covered ? 0xc0000000u : 0u);
          std::memcpy(fixture.source_bytes.data() + pixel * 4, &packed, sizeof(packed));
        }
      }
      upload();
    };
    const auto render = [&](bool eligible, const alpha_auto_source *automatic) {
      write_native_source(fixture, backbuffer);
      require(sunshine_game3d::test::render_frame(renderer, queue->get_immediate_command_list(), source, fixture.depth_view,
        parameters, eligible, {}, plane, automatic), "Automatic-alpha D3D12 render failed");
      const auto decision = renderer.consumed_alpha_auto();
      queue->flush_immediate_command_list();
      renderer.finish_present();
      queue->wait_idle(); // Test-only drain: next render must consume this exact completed observation.
      return decision;
    };
    struct result { std::vector<std::uint8_t> field, color; };
    const auto pixels = [&] {
      const auto resources = renderer.diagnostics();
      return result{fixture.read(reinterpret_cast<ID3D12Resource *>(resources.final_field.handle)),
        fixture.read(reinterpret_cast<ID3D12Resource *>(resources.sbs.handle))};
    };
    const auto equal = [&](const result &expected, const char *message) {
      const auto actual = pixels();
      require(actual.field == expected.field && actual.color == expected.color, message);
    };
    // Resolve independent references through the actual native shader. The
    // detection tests compare both the authoritative field and SBS pixels.
    std::array<result, 3> off, on;
    for (unsigned mask = 0; mask != 3; ++mask) {
      pattern(mask); render(false, nullptr); off[mask] = pixels();
      render(true, nullptr); on[mask] = pixels();
    }
    require(on[0].field != off[0].field && on[1].field != off[1].field,
      "D3D12 automatic fixture has no meaningful full/selective UI difference");
    alpha_auto_policy session;
    alpha_auto_source input;
    input.epoch = 17; input.revision = 23; input.viewport = 1;
    input.now_ms = input.tick_ms = 1000; input.sequence = 1; input.session = &session;
    // The acceptance key (A1) of this fixture's current color alpha.
    const auto current_key = ui_selection::signature{ui_selection::kind::current,
      std::uint32_t(api::format_to_default_typed(static_cast<api::format>(fixture.source_format), 0)), fixture.color}.key();
    // Exact counters (game3d_ui_counters.h): each Auto frame's tick and the
    // source it decides: 4 once current alpha is accepted, whatever its
    // coverage (S1, P1), and 0 before.
    std::vector<std::pair<std::uint64_t, std::uint32_t>> counted;
    bool counting = true, accepted = false;
    const auto verify = [&](unsigned mask, bool eligible, bool enabled, bool inspect_mask = true) {
      pattern(mask);
      input.now_ms += 100; input.tick_ms = input.now_ms; ++input.sequence;
      render(eligible, &input);
      if (counting) counted.emplace_back(input.now_ms, accepted ? 4u : 0u);
      equal(enabled ? on[mask] : off[mask], "D3D12 automatic mask differs from explicit field/SBS reference");
      if (inspect_mask) {
        const auto resource = renderer.diagnostics().ui_source;
        require(resource.handle, "D3D12 automatic detection did not expose its current GPU mask");
        auto *texture = reinterpret_cast<ID3D12Resource *>(resource.handle);
        require(texture->GetDesc().Format == DXGI_FORMAT_R32_FLOAT, "D3D12 automatic mask has the wrong format");
        const auto bytes = fixture.read(texture);
        require(bytes.size() == size_t(width) * height * sizeof(float), "D3D12 automatic mask size differs");
        for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
          float value{}; std::memcpy(&value, bytes.data() + (size_t(y) * width + x) * sizeof(float), sizeof(float));
          const bool wanted = enabled && (mask == 0 || (mask == 1 && x >= width / 3 && x < width / 2 &&
            y >= height / 4 && y < height * 3 / 4));
          require(value == (wanted ? 1.f : 0.f), "D3D12 current GPU mask retained an unsuitable frame");
        }
      }
    };
    // Before acceptance nothing decides, however selective (S1). Acceptance an
    // earlier session earned is restored by its key and decides from the
    // first frame. Then alternate inputs without review or a CPU policy
    // observation: the accepted source decides each frame's own coverage,
    // flat when full. An asynchronous status from the previous frame may never
    // authorize pixels.
    sunshine_game3d::gpu_timing timing;
    renderer.take_gpu_timing(timing); // Time the automatic frames alone.
    verify(1, true, false);
    verify(1, true, false);
    require(session.restore(current_key).restored == 1, "D3D12 current-alpha key was not restored");
    accepted = true;
    for (const unsigned mask : {1u, 0u, 2u, 1u}) verify(mask, true, mask != 2);
    verify(1, true, true);
    require(renderer.consumed_alpha_auto().enabled && renderer.consumed_alpha_auto().source_kind == 4 &&
        renderer.consumed_alpha_auto().pixels == size_t(width) * height,
      "D3D12 native status readback did not retain its measured current-alpha source");
    // Status is delayed evidence, but it must never be attributed to a new
    // source scope merely because that scope has the same candidate bitset.
    for (unsigned scope = 0; scope != 3; ++scope) {
      if (scope == 0) ++input.epoch;
      else if (scope == 1) ++input.revision;
      else ++input.viewport;
      verify(1, true, true);
      require(!renderer.consumed_alpha_auto().sample_sequence,
        "D3D12 status readback crossed epoch/revision/viewport provenance");
      verify(1, true, true);
      require(renderer.consumed_alpha_auto().enabled && renderer.consumed_alpha_auto().source_kind == 4,
        "D3D12 new source scope did not acquire its own status sample");
    }
    counting = false;
    {
      // The session's totals are exact through the last committed sample: every
      // frame up to it detected and decided as scripted, unaccepted current
      // alpha never as inferred coverage; nothing was held.
      const auto counts = session.counters();
      std::uint64_t frames = 0, current = 0;
      for (const auto &[tick, decided] : counted)
        if (tick <= counts.through_ms) { ++frames; current += decided == 4u; }
      require(counts.reconciled() && frames + 1 >= counted.size() && counts[ui_counter::samples] &&
          counts[ui_counter::auto_frames] == frames && counts[ui_counter::detection_frames] == frames &&
          counts.decided(4) == current && counts.decided(0) == frames - current &&
          !counts[ui_counter::untrusted_inferred] && current && frames > current &&
          counts[ui_counter::none + ui_no_mask::unaccepted] == frames - current && !counts.held() && !counts.inactive() &&
          !counts[ui_counter::contradicted] && !counts[ui_counter::reused] && !counts[ui_counter::presented_over_dedicated] &&
          counts[ui_counter::trust_restored] == 1 && session.stored() == current_key,
        "D3D12 exact UI counters differ from the scripted Auto frames");
      std::printf("PASS D3D12 exact UI counters: %llu Auto frames through %llu ms reconcile; decided 4=%llu 0=%llu, unaccepted no-mask frames exact and no inferred source decided unaccepted\n",
        static_cast<unsigned long long>(frames), static_cast<unsigned long long>(counts.through_ms),
        static_cast<unsigned long long>(current), static_cast<unsigned long long>(frames - current));
    }
    // Evidence only: the detection stage's GPU time (tiles, reduce, mask and
    // the sample's readback copies) on these frames.
    if (renderer.take_gpu_timing(timing))
      std::printf("MEASURE D3D12 automatic detection GPU: frames=%u detection_mean=%.4f ms detection_max=%.4f ms total_mean=%.4f ms\n",
        timing.frames, timing.mean_ms[gpu_timing::detection], timing.max_ms[gpu_timing::detection], timing.mean_ms[gpu_timing::total]);
    // T1: a real frame that offers nothing while the accepted current alpha of
    // the previous real frame is missing has no decision of its own. It runs
    // the reduce and mask passes only, reuses that frame's decision once and
    // submits no sample; the next such frame has no mask.
    const auto submitted = renderer.alpha_probe_activity().submitted;
    verify(1, false, true, false);
    require(renderer.consumed_detection().state == ui_detection_snapshot::run_state::ran &&
        !renderer.consumed_detection().candidates &&
        (renderer.consumed_detection().flags & ui_detection::per_frame_accepted_missing) &&
        renderer.alpha_probe_activity().submitted == submitted,
      "D3D12 zero-offer frame did not run the T1 grace alone, or submitted a sample");
    input.retained = true; // No captured view: Auto cannot manufacture availability.
    verify(1, true, false, false);
    input.retained = false;
    verify(1, true, true);
    require(renderer.alpha_probe_activity().submitted && renderer.alpha_probe_activity().mapped,
      "D3D12 automatic status never submitted and mapped its native asynchronous readback");
    renderer.reset_after_runtime_drain();
    require(renderer.configure(observed.runtime, source, static_cast<api::color_space>(fixture.color)),
      "D3D12 reconfigure after automatic detection failed");
    verify(1, true, true);
    for (unsigned mask = 0; mask != 3; ++mask) {
      session.set_manual(true);
      verify(mask, true, true, false);
      require(renderer.consumed_alpha_auto().state == alpha_auto_state::manual_on, "D3D12 Manual On was lost");
      verify(mask, false, false, false);
      session.set_manual(false);
      verify(mask, true, false, false);
      require(renderer.consumed_alpha_auto().state == alpha_auto_state::manual_off, "D3D12 Manual Off was lost");
      session.set_automatic();
      verify(mask, true, mask != 2);
    }
    input.session = nullptr;
    verify(1, true, false, false);
    pattern(0); render(true, nullptr);
    equal(on[0], "D3D12 automatic detection leaked into explicit replay");
    render(false, nullptr);
    fixture.source_bytes = original;
    upload();
    write_native_source(fixture, backbuffer);
    require(fixture.read(backbuffer, D3D12_RESOURCE_STATE_PRESENT) == original,
      "D3D12 alpha regression failed to restore its original source");
    report << "automatic-ui D3D12 full_scene_rejected=1 empty_rejected=1 selective_without_review=1 immediate_bad_frame_rejection=1 resolved_gpu_mask_checked=1 missing_capture_rejected=1 renderer_recreation=1 manual_override=1 explicit_field_and_sbs_parity=1 asynchronous_readback=1\n";
    std::puts("PASS D3D12 automatic UI: nothing decides before acceptance, then the restored source decides each frame's coverage (selective, flat or empty) with the exact GPU mask and field/SBS without review, native readback, manual modes and replay parity");
  }

  void check_automatic_hudless_d3d12(fixture_t &fixture, sunshine_game3d::renderer &renderer, std::ostream &report) {
    using namespace sunshine_game3d;
    auto *queue = observed.runtime->get_command_queue();
    auto *backbuffer = fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
    const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
    const auto original = fixture.source_bytes;
    const auto parameters = native_cases().front().parameters;
    const ui_plane_parameters plane{ui_plane_mode::shallow_front, 0.f};
    const auto count = size_t(width) * height;
    const auto stride = bytes_per_pixel(fixture.source_format);
    std::vector<float> mask(count);
    std::vector<std::uint8_t> hudless(count * stride), shifted(count * stride), final_color(count * stride);
    const auto pixel = [&](std::vector<std::uint8_t> &bytes, size_t index, bool hud, bool mismatch, bool alpha) {
      if (fixture.color == 1) {
        const std::uint8_t value[]{static_cast<std::uint8_t>(hud ? 255 : 64), static_cast<std::uint8_t>(hud ? 64 : 128),
          static_cast<std::uint8_t>(mismatch ? 0 : hud ? 128 : 192), static_cast<std::uint8_t>(alpha ? 255 : 0)};
        std::memcpy(bytes.data() + index * stride, value, sizeof(value));
      } else if (fixture.color == 2) {
        const std::uint16_t value[]{static_cast<std::uint16_t>(hud ? 0x3c00 : 0x3400), static_cast<std::uint16_t>(hud ? 0x3400 : 0x3800),
          static_cast<std::uint16_t>(mismatch ? 0 : hud ? 0x3800 : 0x3a00), static_cast<std::uint16_t>(alpha ? 0x3c00 : 0)};
        std::memcpy(bytes.data() + index * stride, value, sizeof(value));
      } else {
        const std::uint32_t value = (hud ? 1023u : 256u) | ((hud ? 256u : 512u) << 10) |
          ((mismatch ? 0u : hud ? 512u : 768u) << 20) | (alpha ? 0xc0000000u : 0u);
        std::memcpy(bytes.data() + index * stride, &value, sizeof(value));
      }
    };
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
      const auto index = size_t(y) * width + x;
      const bool hud = x >= width / 3 && x < width / 2 && y >= height / 4 && y < height * 3 / 4;
      mask[index] = hud ? 1.f : 0.f;
      pixel(hudless, index, false, false, true);
      pixel(shifted, index, false, true, true);
      pixel(final_color, index, hud, false, hud);
    }
    const auto upload_source = [&](const std::vector<std::uint8_t> &bytes) {
      fixture.source_bytes = bytes;
      void *mapped{}; const D3D12_RANGE no_read{0, 0};
      checked(fixture.source_upload->Map(0, &no_read, &mapped), "Map automatic HUDless source");
      for (unsigned y = 0; y < height; ++y)
        std::memcpy(static_cast<std::uint8_t *>(mapped) + fixture.source_footprint.Offset +
          size_t(y) * fixture.source_footprint.Footprint.RowPitch, bytes.data() + size_t(y) * width * stride, size_t(width) * stride);
      fixture.source_upload->Unmap(0, nullptr);
    };
    struct result { std::vector<std::uint8_t> field, color; };
    const auto render = [&](ui_render_input ui) {
      write_native_source(fixture, backbuffer);
      render_frame_input frame; frame.color = source; frame.depth = fixture.depth_view; frame.scene = parameters;
      frame.ui = ui; frame.ui.plane = plane;
      require(renderer.render(queue->get_immediate_command_list(), frame), "D3D12 automatic HUDless render failed");
      queue->flush_immediate_command_list(); renderer.finish_present(); queue->wait_idle();
      const auto resources = renderer.diagnostics();
      return result{fixture.read(reinterpret_cast<ID3D12Resource *>(resources.final_field.handle)),
        fixture.read(reinterpret_cast<ID3D12Resource *>(resources.sbs.handle))};
    };
    upload_source(final_color);
    const auto off = render({});
    ui_render_input explicit_ui; explicit_ui.kind = ui_input_kind::current_color_alpha;
    const auto on = render(explicit_ui);
    require(on.field != off.field && on.color != off.color, "D3D12 HUDless fixture has no visible UI protection change");
    for (size_t i = 0; i < count; ++i) pixel(final_color, i, mask[i] > 0.f, false, true);
    upload_source(final_color);

    std::array<com_ptr<ID3D12Resource>, 5> textures;
    std::array<api::resource_view, 5> views{};
    auto *device = observed.runtime->get_device();
    const auto create_candidate = [&](unsigned slot, DXGI_FORMAT format, const void *bytes) {
      fixture.texture(textures[slot], format, D3D12_RESOURCE_STATE_COPY_DEST);
      com_ptr<ID3D12Resource> upload; D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
      fixture.fill_upload(upload, textures[slot]->GetDesc(), bytes, footprint);
      fixture.begin_commands();
      D3D12_TEXTURE_COPY_LOCATION from{}, to{};
      from.pResource = upload.p; from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint = footprint;
      to.pResource = textures[slot].p; to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      fixture.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
      transition(fixture.commands.p, textures[slot].p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      fixture.submit();
      require(device->create_resource_view({reinterpret_cast<std::uint64_t>(textures[slot].p)},
          api::resource_usage::shader_resource, api::resource_view_desc(static_cast<api::format>(format)), &views[slot]),
        "Create automatic HUDless D3D12 view");
    };
    create_candidate(0, fixture.source_format, hudless.data());
    create_candidate(1, fixture.source_format, shifted.data());
    create_candidate(2, fixture.source_format, final_color.data());
    create_candidate(3, DXGI_FORMAT_R32_FLOAT, mask.data());
    const std::vector<float> full(count, 1.f);
    create_candidate(4, DXGI_FORMAT_R32_FLOAT, full.data());
    // The fixture's HUD-less pair is accepted as an earlier session earned it
    // (A1); only accepted candidates decide (S1).
    const auto key = [&](ui_selection::kind kind, DXGI_FORMAT format) {
      return ui_selection::signature{kind, std::uint32_t(api::format_to_default_typed(static_cast<api::format>(format), 0)),
        fixture.color}.key();
    };
    alpha_auto_policy policy;
    require(policy.restore(key(ui_selection::kind::hudless, fixture.source_format)).restored == 1,
      "D3D12 HUD-less key was not restored");
    alpha_auto_source observation;
    observation.session = &policy; observation.now_ms = observation.tick_ms = 40000;
    observation.sequence = 1; observation.epoch = 81; observation.revision = 1;
    ui_render_input ui; ui.kind = ui_input_kind::hudless_difference; ui.view = views[0]; ui.automatic = &observation;
    const auto run = [&](bool accepted, const char *message, bool inspect = true) {
      observation.now_ms += 100; observation.tick_ms = observation.now_ms; ++observation.sequence;
      const auto actual = render(ui); const auto &expected = accepted ? on : off;
      require(actual.field == expected.field && actual.color == expected.color, message);
      if (inspect) {
        const auto resource = renderer.diagnostics().ui_source;
        require(resource.handle, "D3D12 HUDless result has no resolved mask");
        auto *texture = reinterpret_cast<ID3D12Resource *>(resource.handle);
        require(texture->GetDesc().Format == DXGI_FORMAT_R32_FLOAT, "D3D12 HUDless mask is not R32");
        const auto bytes = fixture.read(texture);
        require(bytes.size() == count * sizeof(float), "D3D12 HUDless mask size differs");
        for (size_t i = 0; i < count; ++i) {
          float value{}; std::memcpy(&value, bytes.data() + i * sizeof(float), sizeof(float));
          require(value == (accepted ? mask[i] : 0.f), "D3D12 HUDless resolved mask retained an unsuitable candidate");
        }
      }
    };
    run(true, "D3D12 matching final/HUDless differs from explicit protection");
    // T1: the accepted pair invalid in a real frame leaves it without a
    // decision of its own; it reuses the previous real frame's once, and the
    // next such frame has no mask.
    ui.view = views[1]; run(true, "D3D12 a mismatched scene did not reuse the preceding decision once");
    run(false, "D3D12 mismatched scene retained preceding HUD protection");
    ui.view = views[2]; run(false, "D3D12 identical final/HUDless invented a mask");
    ui.view = views[0]; run(true, "D3D12 matching pair did not recover without review");
    ui_detection_inputs inputs;
    inputs.current_color = true;
    inputs.masks[1] = inputs.masks[2] = views[2]; inputs.hudless = views[0];
    ui.detection = &inputs;
    run(true, "D3D12 unaccepted opaque UI/backbuffer candidates prevented HUDless fallback");
    inputs.hudless = views[1]; run(true, "D3D12 an invalid accepted pair did not reuse the preceding decision once");
    run(false, "D3D12 all-unsuitable candidates did not produce Off");
    inputs.masks[0] = views[4]; run(false, "D3D12 an unaccepted opaque R32 UIAlpha decided");
    inputs.hudless = views[0]; run(true, "D3D12 an unaccepted opaque R32 UIAlpha blocked the HUD-less pair");
    require(policy.restore(key(ui_selection::kind::ui_alpha, DXGI_FORMAT_R32_FLOAT)).restored == 1,
      "D3D12 UIAlpha key was not restored");
    inputs.masks[0] = views[3]; inputs.hudless = views[1];
    run(true, "D3D12 accepted R32 UIAlpha did not decide its HUD over unsuitable RGB inputs");
    inputs.masks[0] = {}; inputs.hudless = views[0];
    policy.set_manual(false); run(false, "D3D12 Manual Off did not override valid HUDless", false);
    require(renderer.consumed_alpha_auto().state == alpha_auto_state::manual_off, "D3D12 HUDless lost Manual Off state");
    policy.set_automatic(); run(true, "D3D12 Auto resume required manual review");
    queue->wait_idle();
    for (const auto view : views) device->destroy_resource_view(view);
    upload_source(original); write_native_source(fixture, backbuffer);
    require(fixture.read(backbuffer, D3D12_RESOURCE_STATE_PRESENT) == original, "D3D12 HUDless fixture did not restore source");
    report << "automatic-hudless D3D12 final_alpha_opaque=1 paired_HUD_exact=1 scene_wide_mismatch_rejected=1 t1_grace_once=1 identical_pair_empty=1 same_frame_bad_pair_rejection=1 all_candidate_descriptors=1 unaccepted_never_blocks=1 accepted_ui_alpha_decides=1 current_RGB_preserved=1 manual_off_wins=1 no_review=1\n";
    std::puts("PASS D3D12 automatic HUDless: exact paired HUD mask from an accepted pair, mismatched/empty rejection after the T1 grace reuses the previous real frame's decision once, all native candidate descriptors, unaccepted candidates never block, an accepted R32 UIAlpha decides, and Manual Off");
  }

  void check_adaptive_ui_d3d12(fixture_t &fixture, sunshine_game3d::renderer &renderer,
      sunshine_game3d_test::dump_fixture &dump, std::ostream &report, const fs::path &directory) {
    using namespace sunshine_game3d;
    auto *queue = observed.runtime->get_command_queue();
    auto *backbuffer = fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
    const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
    const auto original = fixture.source_bytes, original_depth = fixture.read(fixture.depth.p);
    const auto upload = [&] {
      void *mapped = nullptr;
      const D3D12_RANGE no_read{0, 0};
      checked(fixture.source_upload->Map(0, &no_read, &mapped), "Map adaptive D3D12 source");
      const auto row = size_t(width) * bytes_per_pixel(fixture.source_format);
      for (unsigned y = 0; y < height; ++y)
        std::memcpy(static_cast<std::uint8_t *>(mapped) + fixture.source_footprint.Offset +
          size_t(y) * fixture.source_footprint.Footprint.RowPitch, fixture.source_bytes.data() + size_t(y) * row, row);
      fixture.source_upload->Unmap(0, nullptr);
    };
    const auto central = [&](unsigned x, unsigned y) {
      return x >= width / 8 && x < width * 7 / 8 && y >= height / 8 && y < height * 7 / 8;
    };
    std::uint64_t total_covered{};
    const auto set_mask = [&](const auto &includes) {
      fixture.source_bytes = original;
      std::uint64_t covered{};
      total_covered = 0;
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const bool ui = includes(x, y);
        covered += ui && central(x, y);
        total_covered += ui;
        const auto pixel = size_t(y) * width + x;
        if (fixture.color == 1) fixture.source_bytes[pixel * 4 + 3] = ui ? 255 : 0;
        else if (fixture.color == 2) {
          const std::uint16_t alpha = ui ? 0x3400 : 0; // Soft 0.25 alpha remains covered.
          std::memcpy(fixture.source_bytes.data() + pixel * 8 + 6, &alpha, sizeof(alpha));
        } else {
          std::uint32_t packed{};
          std::memcpy(&packed, fixture.source_bytes.data() + pixel * 4, sizeof(packed));
          packed = (packed & 0x3fffffffu) | (ui ? 0x40000000u : 0u);
          std::memcpy(fixture.source_bytes.data() + pixel * 4, &packed, sizeof(packed));
        }
      }
      upload();
      return covered;
    };
    const auto pattern = [&](bool full) {
      return set_mask([&](unsigned x, unsigned y) {
        return full || (x >= width / 3 && x < width / 2 && y >= height / 4 && y < height * 3 / 4);
      });
    };
    auto covered = pattern(false);
    require(covered && covered < std::uint64_t(width) * height, "Adaptive D3D12 fixture needs selective UI");
    auto parameters = native_cases().front().parameters;
    parameters.strength = 100; parameters.depth_scale = 100000; parameters.strength_blend = 1;
    parameters.coordinate_basis = 1; parameters.projection = {0, 1};
    parameters.convergence = {.05f, .1f}; parameters.disparity_limit_uv = .01f;
    fixture.upload_depth(std::vector<float>(size_t(width) * height, .9f));
    ui_adaptive::source input;
    input.epoch = 7100; input.revision = 3; input.source_id = 91; input.viewport = 2; input.eligible = true;
    const ui_plane_parameters plane{ui_plane_mode::display_fraction, 0.f};
    api::resource_view selected_alpha{};
    sunshine_game3d::renderer frozen;
    require(frozen.configure(observed.runtime, source, static_cast<api::color_space>(fixture.color)),
      "Configure frozen adaptive D3D12 reference");
    struct snapshot {
      ui_adaptive::decision decision;
      ui_plane_parameters plane;
      std::vector<std::uint8_t> candidate, vertical, field, color;
    };
    unsigned ui_x = (width / 3 + width / 2) / 2, ui_y = height / 2;
    const auto ui_pixel = [&](const snapshot &value) {
      float displacement{};
      const size_t pixel = size_t(ui_y) * width + ui_x;
      require(value.field.size() == size_t(width) * height * sizeof(float), "Adaptive D3D12 field extent changed");
      std::memcpy(&displacement, value.field.data() + pixel * sizeof(float), sizeof(displacement));
      return displacement;
    };
    unsigned renders{};
    const auto render = [&](sunshine_game3d::renderer &active, bool protect, const ui_plane_parameters &selected,
                            const ui_adaptive::source *observation, bool capture = false) {
      write_native_source(fixture, backbuffer);
      require(sunshine_game3d::test::render_frame(active, queue->get_immediate_command_list(), source, fixture.depth_view,
          parameters, protect, selected_alpha, selected, nullptr, observation), "Adaptive D3D12 render failed");
      if (capture) dump.begin(observed.runtime, active, parameters, fixture.depth_view, false,
        static_cast<api::color_space>(fixture.color));
      queue->flush_immediate_command_list(); active.finish_present();
      if (capture) dump.submitted(observed.runtime);
      queue->wait_idle(); // Test-only completion; production observations remain asynchronous.
      if (capture) dump.verify(observed.runtime, [&](api::resource texture, bool common) {
        return fixture.read(reinterpret_cast<ID3D12Resource *>(texture.handle),
          common ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      }, directory / "dump-adaptive-ui");
      const auto resources = active.diagnostics();
      const auto read = [&](api::resource value) { return fixture.read(reinterpret_cast<ID3D12Resource *>(value.handle)); };
      ++renders;
      const snapshot output{active.consumed_ui_adaptive(), active.consumed_ui_plane(),
        read(resources.candidate), read(resources.vertical_field), read(resources.final_field), read(resources.sbs)};
      if (protect) {
        const float cursor_plane = admitted_ui_parallax_uv(parameters, output.plane, true, true, width, height);
        require(double(width) * std::abs(double(cursor_plane) - ui_pixel(output)) <= 1e-6,
          "Exported cursor plane disagrees with the actual D3D12 UI field");
      }
      return output;
    };
    const auto sample = [&](std::uint64_t tick, bool capture = false) {
      input.front_cap_uv = display_parallax_cap_uv(parameters);
      input.now_ms = input.tick_ms = tick; ++input.sequence;
      const auto submitted = render(renderer, true, plane, &input, capture);
      require(submitted.decision.accepted_sequence != input.sequence,
        "Adaptive D3D12 consumed its just-submitted observation synchronously");
      const auto accepted = render(renderer, true, plane, &input);
      require(accepted.decision.accepted_sequence == input.sequence && accepted.decision.accepted_tick == tick &&
          accepted.decision.covered_pixels == covered,
        "Adaptive D3D12 lost completed source identity or central selected-mask coverage");
      require(accepted.plane.mode == ui_plane_mode::display_fraction &&
          accepted.plane.inverse_depth == accepted.decision.applied_fraction &&
          accepted.plane.inverse_depth <= ui_adaptive::max_live_fraction &&
          accepted.decision.applied_uv <= ui_adaptive::comfort_cap_uv &&
          std::abs(ui_pixel(accepted) - accepted.decision.applied_uv) <= 1e-8f,
        "Adaptive D3D12 consumed a different fraction than its diagnostic decision");
      require(ui_parameter_words(true, submitted.plane) == ui_parameter_words(true, accepted.plane) &&
          submitted.field == accepted.field && submitted.color == accepted.color &&
          submitted.candidate == accepted.candidate && submitted.vertical == accepted.vertical,
        "Adaptive D3D12 probe split changed pixels or scene geometry at the same consumed fraction");
      return accepted;
    };
    auto current = sample(1000);
    require(current.decision.target_uv == 0.f && current.decision.required_index == 4 &&
        current.decision.capped_conflict && current.decision.required_uv >= current.decision.limit_uv &&
        current.decision.applied_uv == 0.f &&
        current.decision.conflict_counts == std::array<std::uint64_t, 5>{covered, covered, covered, covered, covered},
      "Adaptive D3D12 lost saturated absolute-plane conflicts or skipped upward confirmation");
    current = sample(1100);
    require(current.decision.target_uv == ui_adaptive::comfort_cap_uv && current.decision.applied_uv == 0.f,
      "Adaptive D3D12 upward target did not wait for sustained fresh evidence");
    current = sample(1250, true);
    require(current.decision.applied_uv == ui_adaptive::comfort_cap_uv,
      "Adaptive D3D12 approach slew did not reach its bounded actual-UV target");
    require(std::abs(ui_pixel(current) - ui_adaptive::comfort_cap_uv) <= 1e-8f,
      "Adaptive D3D12 actual UI field did not reach its absolute comfort cap");
    const auto dumped_fraction = current.plane.inverse_depth;
    const auto explicit_fraction = render(frozen, true, current.plane, nullptr);
    require(current.field == explicit_fraction.field && current.color == explicit_fraction.color &&
        current.candidate == explicit_fraction.candidate && current.vertical == explicit_fraction.vertical,
      "Adaptive split passes differ from explicit frozen-fraction rendering");
    const auto off = render(frozen, false, current.plane, nullptr);
    require(current.candidate == off.candidate && current.vertical == off.vertical && current.field != off.field,
      "Adaptive UI changed scene candidate/vertical geometry or the fixture has no UI effect");
    const auto historical = render(frozen, true, {ui_plane_mode::display_fraction, .75f}, nullptr);
    require(std::abs(ui_pixel(historical) - .75f * parameters.disparity_limit_uv) <= 1e-8f,
      "Live comfort ceiling changed historical frozen 75-percent rendering");
    parameters.strength = 37.f; parameters.strength_blend = .63f;
    const auto partial_strength = render(frozen, true, {ui_plane_mode::display_fraction, .5f}, nullptr);
    require(ui_pixel(partial_strength) > 0.f && ui_pixel(partial_strength) < ui_pixel(current),
      "Partial strength/blend did not exercise a different positive cursor/UI plane");
    parameters.strength = 100.f; parameters.strength_blend = 1.f;
    {
      std::ifstream file(directory / "dump-adaptive-ui" / "manifest.json");
      require(file.good(), "Adaptive D3D12 dump missing");
      const auto manifest = nlohmann::json::parse(file);
      const auto &replay = manifest.at("producer_metadata").at("replay");
      require(replay.at("ui_parameter_abi") == "sunshine_game3d.ui_parameters.v7" &&
          replay.at("ui_constant_binding").at("mode") == 5 &&
          replay.at("ui_constant_binding").at("front_limit_fraction") == dumped_fraction,
        "Adaptive D3D12 dump did not freeze the exact applied fraction");
    }
    // Fresh, low-conflict GPU samples must fill the complete release dwell.
    fixture.upload_depth(std::vector<float>(size_t(width) * height, 0.f));
    for (std::uint64_t tick = 1500; tick <= 3000; tick += 250) {
      current = sample(tick);
      require(!current.decision.capped_conflict && current.decision.required_uv == 0.f &&
          current.decision.conflict_counts == std::array<std::uint64_t, 5>{} &&
          current.decision.target_uv == (tick == 3000 ? 0.f : ui_adaptive::comfort_cap_uv),
        "Adaptive D3D12 retreat skipped its fresh-observation dwell or used old conflict bins");
    }
    current = sample(3200);
    require(std::abs(current.decision.applied_uv - .0025f) <= 1e-8f,
      "Adaptive D3D12 no-conflict retreat did not use the bounded actual-UV slew");
    require(std::abs(ui_pixel(current) - current.decision.applied_uv) <= 1e-8f,
      "Adaptive D3D12 actual UI field failed to follow the retreating actual-UV plane");
    const auto held = current;
    input.eligible = false; input.now_ms = 3500;
    current = render(renderer, true, plane, &input);
    require(current.decision.applied_fraction == held.decision.applied_fraction &&
        current.decision.accepted_sequence == held.decision.accepted_sequence,
      "Missing adaptive source moved the plane or fabricated new evidence");
    // Present an opaque frame while selecting a retained selective input. Only
    // that selected alpha may enter the probe; its RGB never enters the eyes.
    pattern(false); write_native_source(fixture, backbuffer);
    selected_alpha = renderer.prepare_ui_source(100, [&](api::resource destination) {
      auto *cmd = queue->get_immediate_command_list();
      cmd->barrier(source, api::resource_usage::present, api::resource_usage::copy_source);
      cmd->barrier(destination, api::resource_usage::shader_resource, api::resource_usage::copy_dest);
      cmd->copy_resource(source, destination);
      cmd->barrier(destination, api::resource_usage::copy_dest, api::resource_usage::shader_resource);
      cmd->barrier(source, api::resource_usage::copy_source, api::resource_usage::present);
      return true;
    });
    require(selected_alpha.handle, "Adaptive retained-mask fixture failed to preserve alpha");
    queue->flush_immediate_command_list(); queue->wait_idle();
    pattern(true); input.eligible = true; input.mask_sequence = 100;
    current = sample(3600);
    const auto retained = current;
    for (const auto now : {3900u, 4200u}) {
      input.now_ms = now;
      current = render(renderer, true, plane, &input);
      require(current.decision.accepted_sequence == retained.decision.accepted_sequence &&
          current.decision.accepted_tick == retained.decision.accepted_tick &&
          current.decision.applied_fraction == retained.decision.applied_fraction,
        "Duplicate expired FG input advanced adaptive evidence or moved the plane");
    }
    input.now_ms = input.tick_ms = 4250; ++input.sequence;
    current = render(renderer, true, plane, &input);
    require(current.decision.accepted_sequence == retained.decision.accepted_sequence &&
        current.decision.applied_fraction == retained.decision.applied_fraction,
      "Fresh depth with the same retained mask incorrectly advanced adaptive evidence");
    const auto retained_explicit = render(frozen, true, current.plane, nullptr);
    require(current.field == retained_explicit.field && current.color == retained_explicit.color,
      "Adaptive retained-mask rendering differs from exact frozen-fraction replay");
    // Hold actual queue completion while availability drops. Releasing the
    // queue must not turn a pre-gap observation into new release evidence.
    // The guard also releases on an exception or a bounded watchdog timeout.
    struct gated_queue {
      com_ptr<ID3D12Fence> fence;
      std::atomic<bool> released{}, forced{};
      std::thread watchdog;
      void release() noexcept {
        released.store(true, std::memory_order_release);
        if (fence.p) fence->Signal(1);
      }
      ~gated_queue() { release(); if (watchdog.joinable()) watchdog.join(); }
    } gate;
    checked(fixture.game->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(gate.fence.put())),
      "Create adaptive observation gate");
    write_native_source(fixture, backbuffer);
    gate.watchdog = std::thread([&] {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (!gate.released.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) Sleep(1);
      if (!gate.released.exchange(true, std::memory_order_acq_rel)) {
        gate.forced.store(true, std::memory_order_release);
        gate.fence->Signal(1);
      }
    });
    checked(fixture.queue->Wait(gate.fence.p, 1), "Gate adaptive GPU completion");
    input.now_ms = input.tick_ms = 5000; ++input.sequence; input.mask_sequence = 101;
    const auto blocked_sequence = input.sequence;
    const auto blocked_render = [&] {
      require(sunshine_game3d::test::render_frame(renderer, queue->get_immediate_command_list(), source, fixture.depth_view,
          parameters, true, selected_alpha, plane, nullptr, &input), "Record gated adaptive D3D12 render");
      queue->flush_immediate_command_list(); renderer.finish_present(); ++renders;
    };
    blocked_render();
    require(renderer.consumed_ui_adaptive().accepted_sequence != blocked_sequence,
      "Gated adaptive observation completed before the queue was released");
    input.eligible = false; input.now_ms = 5010;
    blocked_render();
    input.eligible = true; input.now_ms = 5020;
    gate.release();
    queue->wait_idle();
    require(!gate.forced.load(std::memory_order_acquire), "Adaptive gate required watchdog release");
    current = render(renderer, true, plane, &input);
    require(current.decision.accepted_sequence != blocked_sequence &&
        current.decision.applied_fraction == retained.decision.applied_fraction,
      "Quick recovery accepted a pending adaptive observation from before source loss");
    input.mask_sequence = 102;
    current = sample(5100);
    require(current.decision.accepted_sequence == blocked_sequence + 1 &&
        current.decision.accepted_mask_sequence == 102,
      "A fresh depth/mask observation failed to recover after pending-sample invalidation");

    // Ignore edge UI only when choosing a plane. Its actual rendering must
    // still match an explicit frozen plane, even over uniformly near geometry.
    selected_alpha = {};
    input.mask_sequence = 0; ++input.epoch;
    covered = set_mask([&](unsigned x, unsigned y) { return !central(x, y); });
    require(!covered && total_covered, "D3D12 edge-only mask has central coverage or no edge UI");
    ui_x = ui_y = 0;
    fixture.upload_depth(std::vector<float>(size_t(width) * height, .9f));
    current = sample(6000);
    current = sample(6250);
    require(current.decision.target_uv == 0.f && current.decision.required_uv == 0.f &&
        !current.decision.capped_conflict && current.decision.conflict_pixels == 0 &&
        current.decision.applied_uv == 0.f,
      "Near UI outside the central 75-percent region promoted the D3D12 UI plane");
    const auto edge_frozen = render(frozen, true, current.plane, nullptr);
    const auto edge_off = render(frozen, false, current.plane, nullptr);
    require(current.field == edge_frozen.field && current.color == edge_frozen.color &&
        current.candidate == edge_off.candidate && current.vertical == edge_off.vertical &&
        current.field != edge_off.field && ui_pixel(current) == 0.f && ui_pixel(edge_off) > 0.f,
      "Ignoring border UI for placement removed its protection or changed D3D12 scene geometry");

    // A small central near-UI patch must promote independently of the much
    // larger far-background UI outside the evidence region. Both comparisons
    // use real conditioned GPU depth, before UI pinning.
    ui_x = (width / 3 + width / 2) / 2; ui_y = height / 2;
    const auto near_patch = [&](unsigned x, unsigned y) {
      return x >= ui_x - width / 64 && x < ui_x + width / 64 &&
        y >= ui_y - height / 32 && y < ui_y + height / 32;
    };
    std::vector<float> patch_depth(size_t(width) * height, 0.f);
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x)
      if (near_patch(x, y)) patch_depth[size_t(y) * width + x] = .9f;
    fixture.upload_depth(patch_depth);
    ++input.epoch; covered = set_mask(near_patch);
    const auto patch_coverage = covered;
    require(patch_coverage && patch_coverage == total_covered,
      "D3D12 near-UI fixture must lie wholly inside the central evidence region");
    current = sample(6500);
    require(current.decision.required_uv > 0.f && current.decision.target_uv == 0.f &&
        current.decision.required_index == 4 && current.decision.capped_conflict,
      "Significant central near-UI conflict lost its required UV or skipped confirmation");
    current = sample(6600);
    require(current.decision.target_uv == ui_adaptive::comfort_cap_uv,
      "Sustained central near-UI conflict failed to promote the D3D12 plane");
    const auto patch_required = current.decision.required_uv;
    const auto patch_conflicts = current.decision.conflict_counts;

    ++input.epoch;
    covered = set_mask([&](unsigned x, unsigned y) { return !central(x, y) || near_patch(x, y); });
    require(covered == patch_coverage && total_covered > covered * 10,
      "D3D12 border-dilution fixture lacks enough irrelevant UI coverage");
    const auto before_pin = render(frozen, false, plane, nullptr);
    std::uint64_t whole_mask_conflicts{}, central_conflicts{};
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
      if (central(x, y) && !near_patch(x, y)) continue;
      float value{};
      std::memcpy(&value, before_pin.field.data() + (size_t(y) * width + x) * sizeof(float), sizeof(value));
      require(std::isfinite(value), "D3D12 conflict witness has nonfinite conditioned depth");
      if (value + .05f * parameters.disparity_limit_uv > 0.f) {
        ++whole_mask_conflicts;
        central_conflicts += central(x, y);
      }
    }
    require(central_conflicts * 100 > covered * ui_adaptive::entry_conflict_percent &&
        whole_mask_conflicts * 100 <= total_covered * ui_adaptive::entry_conflict_percent,
      "D3D12 actual GPU witness does not distinguish central from whole-mask conflict ratios");
    current = sample(6750);
    require(current.decision.required_uv == patch_required && current.decision.target_uv == 0.f &&
        current.decision.capped_conflict && current.decision.conflict_counts == patch_conflicts,
      "Far border UI diluted central conflict evidence or changed its required D3D12 UV");
    current = sample(6850);
    require(current.decision.target_uv == ui_adaptive::comfort_cap_uv,
      "Far border UI prevented central conflict from confirming its D3D12 target");
    current = sample(7000);
    const auto border_frozen = render(frozen, true, current.plane, nullptr);
    require(current.decision.applied_uv == ui_adaptive::comfort_cap_uv && current.field == border_frozen.field &&
        current.color == border_frozen.color && current.candidate == before_pin.candidate &&
        current.vertical == before_pin.vertical,
      "Central-only placement changed full-frame UI rendering or D3D12 scene geometry");
    ui_x = ui_y = 0;
    require(std::abs(ui_pixel(current) - ui_adaptive::comfort_cap_uv) <= 1e-8f,
      "Border UI did not share the central UI's resolved plane");
    report << "adaptive-ui central-region D3D12 covered=" << covered << " conflicts=" << central_conflicts
      << " whole_mask_covered=" << total_covered << " whole_mask_conflicts=" << whole_mask_conflicts << '\n';

    // A uniform shallow scene chooses an intermediate absolute plane, rather
    // than only exercising the ceiling. Count every candidate against actual
    // GPU depth before UI pinning, independently of temporal placement.
    ++input.epoch;
    ui_x = (width / 3 + width / 2) / 2; ui_y = height / 2;
    covered = pattern(false);
    parameters.depth_scale = 1.f; parameters.convergence = {1.f, 0.f};
    const float shallow_raw = .0014f * float(width) * 2160.f / (float(height) * 100.f);
    require(shallow_raw > 0.f && shallow_raw < 1.f, "D3D12 absolute-plane witness is outside raw-depth range");
    fixture.upload_depth(std::vector<float>(size_t(width) * height, shallow_raw));
    const auto shallow_off = render(frozen, false, plane, nullptr);
    const float cap = display_parallax_cap_uv(parameters);
    std::array<std::uint64_t, 5> absolute_conflicts{};
    for (unsigned y = height / 4; y < height * 3 / 4; ++y)
      for (unsigned x = width / 3; x < width / 2; ++x) {
        float value{};
        std::memcpy(&value, shallow_off.field.data() + (size_t(y) * width + x) * sizeof(float), sizeof(value));
        require(std::isfinite(value), "D3D12 absolute-plane witness has nonfinite conditioned depth");
        const float required = value + .05f * cap;
        for (unsigned level = 0; level < absolute_conflicts.size(); ++level)
          absolute_conflicts[level] += required > ui_adaptive::level_uv(level, cap);
      }
    unsigned required_index{};
    while (required_index + 1 < absolute_conflicts.size() &&
        absolute_conflicts[required_index] * 100 > covered * ui_adaptive::entry_conflict_percent) ++required_index;
    require(required_index == 2, "D3D12 shallow witness did not distinguish the 0.002 UV candidate");
    const float selected_uv = ui_adaptive::level_uv(required_index, cap);
    current = sample(8000);
    require(!current.decision.capped_conflict && current.decision.conflict_counts == absolute_conflicts &&
        current.decision.required_index == required_index && current.decision.target_uv == 0.f &&
        current.decision.required_uv == selected_uv,
      "D3D12 absolute-plane GPU counts did not choose the first acceptable candidate");
    current = sample(8100);
    require(current.decision.target_uv == selected_uv,
      "D3D12 absolute-plane candidate did not become the confirmed target");
    current = sample(8300);
    require(std::abs(current.decision.applied_uv - selected_uv) <= 1e-8f &&
        current.plane.inverse_depth > .1f && current.plane.inverse_depth < .25f,
      "D3D12 adaptive placement rescaled an absolute plane into a relative level");
    const auto absolute_frozen = render(frozen, true, current.plane, nullptr);
    require(current.field == absolute_frozen.field && current.color == absolute_frozen.color &&
        current.candidate == shallow_off.candidate && current.vertical == shallow_off.vertical,
      "D3D12 absolute plane differs from frozen rendering or modified scene geometry");
    const auto stable_uv = current.decision.applied_uv;
    const auto stable_fraction = current.plane.inverse_depth;
    parameters.disparity_limit_uv = .0102f;
    current = sample(8500);
    current = sample(8750);
    require(current.decision.applied_uv == stable_uv && current.decision.target_uv == stable_uv &&
        current.plane.inverse_depth < stable_fraction &&
        std::abs(ui_pixel(current) - stable_uv) <= 1e-8f,
      "D3D12 nonbinding display-cap change moved actual UI UV instead of rescaling its fraction");
    report << "adaptive-ui absolute-plane D3D12 required_index=" << required_index << " applied_uv=" << stable_uv
      << " fraction_before=" << stable_fraction << " fraction_after=" << current.plane.inverse_depth << '\n';

    // A smaller current scene cap clips all candidates above half that cap.
    // The displayed UI and exported cursor must obey that bound immediately.
    parameters.disparity_limit_uv = .003f;
    const float relative_limit = .5f * display_parallax_cap_uv(parameters);
    const auto clipped_off = render(frozen, false, plane, nullptr);
    require(ui_pixel(clipped_off) + .05f * display_parallax_cap_uv(parameters) > relative_limit,
      "D3D12 relative-cap witness does not conflict beyond the allowed UI plane");
    current = sample(9000);
    require(current.decision.limit_uv == relative_limit && current.decision.applied_uv == relative_limit &&
        current.decision.target_uv == relative_limit && current.decision.capped_conflict &&
        current.decision.conflict_counts == std::array<std::uint64_t, 5>{covered, covered, covered, covered, covered},
      "D3D12 absolute candidates failed to clip to the current relative limit");
    const auto clipped_frozen = render(frozen, true, current.plane, nullptr);
    require(current.field == clipped_frozen.field && current.color == clipped_frozen.color &&
        current.candidate == clipped_off.candidate && current.vertical == clipped_off.vertical,
      "D3D12 relative-cap change broke frozen UI rendering or modified scene geometry");

    // Large far-depth UI must not dilute a foreground overlap that exceeds 2%
    // of the central rectangle. Keep the near UI away from depth boundaries and
    // derive all five conflict counters from the actual GPU field before UI pins.
    ++input.epoch;
    parameters.disparity_limit_uv = .01f;
    parameters.depth_scale = 10.f; parameters.convergence = {1.f, .1f};
    const float area_raw = .1f + .0024f * float(width) * 2160.f / (float(height) * 1000.f);
    require(area_raw > 0.f && area_raw < 1.f, "D3D12 conflict-area raw depth is outside range");
    const auto area_near = [&](unsigned x, unsigned y) {
      return x >= width * 5 / 32 && x < width * 9 / 32 &&
        y >= height * 5 / 32 && y < height * 9 / 32;
    };
    const auto area_far = [&](unsigned x, unsigned y) {
      return x >= width * 9 / 16 && x < width * 14 / 16 &&
        y >= height * 9 / 16 && y < height * 14 / 16;
    };
    std::vector<float> area_depth(size_t(width) * height, 0.f);
    for (unsigned y = height * 2 / 16; y < height * 5 / 16; ++y)
      for (unsigned x = width * 2 / 16; x < width * 5 / 16; ++x)
        area_depth[size_t(y) * width + x] = area_raw;
    fixture.upload_depth(area_depth);
    covered = set_mask([&](unsigned x, unsigned y) { return area_near(x, y) || area_far(x, y); });
    ui_x = (width * 5 / 32 + width * 9 / 32) / 2;
    ui_y = (height * 5 / 32 + height * 9 / 32) / 2;
    const auto area_off = render(frozen, false, plane, nullptr);
    const float area_cap = display_parallax_cap_uv(parameters);
    const std::uint64_t center_pixels = std::uint64_t(width * 7 / 8 - width / 8) *
      (height * 7 / 8 - height / 8);
    std::array<std::uint64_t, 5> area_totals{};
    std::uint64_t area_covered{};
    for (unsigned y = height / 8; y < height * 7 / 8; ++y)
      for (unsigned x = width / 8; x < width * 7 / 8; ++x) {
        if (!area_near(x, y) && !area_far(x, y)) continue;
        ++area_covered;
        float value{};
        std::memcpy(&value, area_off.field.data() + (size_t(y) * width + x) * sizeof(float), sizeof(value));
        require(std::isfinite(value), "D3D12 conflict-area witness has nonfinite conditioned depth");
        const float required = value + .05f * area_cap;
        for (unsigned level = 0; level < area_totals.size(); ++level) {
          area_totals[level] += required > ui_adaptive::level_uv(level, area_cap);
        }
      }
    const auto area_conflicts = [&](unsigned level) {
      return area_totals[level] * 100 > covered * ui_adaptive::entry_conflict_percent ||
        area_totals[level] * 100 > center_pixels * 2;
    };
    unsigned area_required{};
    while (area_required < ui_adaptive::max_target_index && area_conflicts(area_required)) ++area_required;
    require(area_covered == covered && covered == total_covered &&
        area_totals[0] * 100 < covered * ui_adaptive::entry_conflict_percent &&
        area_totals[0] * 100 > center_pixels * 2 &&
        area_required == 3 && !area_conflicts(area_required),
      "D3D12 actual GPU witness does not distinguish conflict area from a diluted UI ratio");
    current = sample(9500);
    require(current.decision.covered_pixels == area_covered && current.decision.conflict_counts == area_totals &&
        current.decision.center_pixels == center_pixels &&
        current.decision.required_index == area_required && current.decision.target_uv == 0.f &&
        !current.decision.capped_conflict,
      "D3D12 conflict-area decision lost exact GPU counts or skipped upward confirmation");
    current = sample(9600);
    const float area_target = ui_adaptive::level_uv(area_required, area_cap);
    require(current.decision.target_uv == area_target,
      "Far UI diluted the D3D12 foreground target despite sufficient conflict area");
    current = sample(9750);
    const auto area_frozen = render(frozen, true, current.plane, nullptr);
    require(std::abs(current.decision.applied_uv - area_target) <= 1e-8f &&
        current.field == area_frozen.field && current.color == area_frozen.color &&
        current.candidate == area_off.candidate && current.vertical == area_off.vertical,
      "D3D12 conflict-area placement changed scene geometry or differs from frozen UI rendering");
    ui_x = (width * 9 / 16 + width * 14 / 16) / 2;
    ui_y = (height * 9 / 16 + height * 14 / 16) / 2;
    require(std::abs(ui_pixel(current) - area_target) <= 1e-8f,
      "D3D12 far UI and near UI do not share the selected global plane");
    report << "adaptive-ui conflict-area D3D12 covered=" << covered << " zero_conflicts=" << area_totals[0]
      << " center_pixels=" << center_pixels << " required_index=" << area_required
      << " applied_uv=" << current.decision.applied_uv << '\n';

    // Generic depth presents A/B/C allocation identities within one admitted
    // scene basis and routing group. Readback completes on the next identity,
    // so a physical source/layout comparison would starve this policy forever.
    ++input.epoch;
    input.generic_basis_epoch = 71; input.generic_routing_epoch = 73;
    input.front_cap_uv = display_parallax_cap_uv(parameters);
    struct probe_identity {
      std::uint64_t sequence, tick, source_id, revision;
    };
    std::vector<probe_identity> submitted_probes;
    auto submissions = renderer.ui_probe_submissions();
    const auto before_rotation = submissions;
    std::uint64_t last_submission_tick = 9750, last_accepted_sequence{};
    unsigned off_turn_accepts{};
    const auto record_submission = [&] {
      const auto total = renderer.ui_probe_submissions();
      const bool submitted = total != submissions;
      if (submitted) {
        require(total == submissions + 1 && input.now_ms - last_submission_tick >= 100,
          "D3D12 generic identity changes bypassed the 100ms conflict-probe budget");
        last_submission_tick = input.now_ms;
        submitted_probes.push_back({input.sequence, input.tick_ms, input.source_id, input.revision});
      }
      submissions = total;
      return submitted;
    };
    for (unsigned frame = 0; frame < 40; ++frame) {
      input.now_ms = input.tick_ms = 10000 + frame * 16;
      ++input.sequence;
      input.source_id = 91 + frame % 3;
      input.revision = 300 + frame % 3;
      current = render(renderer, true, plane, &input);
      record_submission();
      if (current.decision.accepted_sequence && current.decision.accepted_sequence != last_accepted_sequence) {
        const auto accepted = std::find_if(submitted_probes.begin(), submitted_probes.end(), [&](const auto &probe) {
          return probe.sequence == current.decision.accepted_sequence;
        });
        require(accepted != submitted_probes.end() && accepted->tick == current.decision.accepted_tick &&
            accepted->tick < input.now_ms && accepted->source_id != input.source_id && accepted->revision != input.revision,
          "D3D12 generic readback did not retain its off-turn source identity");
        require(current.decision.covered_pixels == covered && current.decision.center_pixels == center_pixels &&
            current.decision.conflict_counts == area_totals && current.decision.required_uv == area_target,
          "D3D12 generic rotation changed the accepted GPU scene/conflict evidence");
        ++off_turn_accepts;
        last_accepted_sequence = current.decision.accepted_sequence;
      }
    }
    require(off_turn_accepts >= 4 && submissions - before_rotation >= 4 && submissions - before_rotation <= 7 &&
        current.decision.target_uv == area_target && std::abs(current.decision.applied_uv - area_target) <= 1e-8f,
      "D3D12 A/B/C generic rotation failed to settle with bounded fresh probes");
    const auto rotating_frozen = render(frozen, true, current.plane, nullptr);
    require(current.field == rotating_frozen.field && current.color == rotating_frozen.color,
      "D3D12 generic rotation differs from rendering the resolved frozen UI plane");

    // Unrelated routing groups must discard completed observations from the
    // previous group, while retaining the renderer's submission cadence.
    const auto before_churn = submissions;
    bool previous_submitted = false;
    unsigned discarded_pending{};
    for (unsigned frame = 0; frame < 14; ++frame) {
      input.now_ms = input.tick_ms = 10640 + frame * 16;
      ++input.sequence; ++input.generic_routing_epoch;
      input.source_id = 91 + frame % 3; input.revision = 300 + frame % 3;
      current = render(renderer, true, plane, &input);
      require(current.decision.accepted_sequence == 0 && current.decision.applied_uv == 0 &&
          current.decision.target_uv == 0,
        "D3D12 logical routing change consumed the preceding group's pending probe");
      if (previous_submitted) ++discarded_pending;
      previous_submitted = record_submission();
    }
    require(discarded_pending && submissions - before_churn <= 3,
      "D3D12 routing churn did not exercise pending rejection or scanned once per presentation");
    report << "adaptive-ui generic-rotation D3D12 identities=ABC layout_revisions=3 frame_step_ms=16 off_turn_accepts="
      << off_turn_accepts << " scans=" << before_churn - before_rotation
      << " churn_scans=" << submissions - before_churn << " pending_discarded=" << discarded_pending
      << " minimum_submission_interval_ms=100 frozen_plane_pixels=1\n";

    // The conflict probe counts only texels the mask pins exactly, at or above
    // the soft pin knee; fainter UI blends toward scene depth and never places
    // the plane. PQ's two-bit presented alpha has no value below the knee.
    if (fixture.color != 3) {
      ++input.epoch;
      fixture.source_bytes = original;
      covered = 0;
      // Four levels in diagonal stripes: opaque, the 8-bit code or half just
      // below the knee 1/gain, the first one at or above it, and faint.
      const float gain = float(sunshine_game3d::shader_marker(sunshine_game3d::renderer::shader_source(), "SUNSHINE_UI_SOFT_PIN_GAIN"));
      const auto knee_code = std::uint8_t(std::ceil(255.f / gain));
      const float knee = 1.f / gain;
      std::uint32_t knee_bits;
      std::memcpy(&knee_bits, &knee, sizeof(knee_bits));
      // The least half at or above 1/gain (a normal half for any gain up to 2^14).
      const auto knee_half = std::uint16_t((((knee_bits >> 23 & 255u) - 112u) << 10 | (knee_bits & 0x7fffffu) >> 13) +
        ((knee_bits & 0x1fffu) != 0));
      const std::uint8_t codes[]{255, std::uint8_t(knee_code - 1), knee_code, 8};
      const std::uint16_t halves[]{0x3400, std::uint16_t(knee_half - 1), knee_half, 0x2c00}; // 1/4 and 1/16 outside.
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const bool ui = x >= width / 3 && x < width / 2 && y >= height / 4 && y < height * 3 / 4;
        const unsigned level = (x + y) % 4;
        covered += ui && level % 2 == 0 && central(x, y);
        const auto pixel = size_t(y) * width + x;
        if (fixture.color == 1) fixture.source_bytes[pixel * 4 + 3] = ui ? codes[level] : 0;
        else {
          const std::uint16_t alpha = ui ? halves[level] : 0;
          std::memcpy(fixture.source_bytes.data() + pixel * 8 + 6, &alpha, sizeof(alpha));
        }
      }
      upload();
      ui_x = (width / 3 + width / 2) / 2; ui_y = height / 2;
      ui_x += (4 - (ui_x + ui_y) % 4) % 4; // An exactly pinned (opaque) texel.
      current = sample(11000);
      require(current.decision.covered_pixels == covered && covered > 0,
        "D3D12 conflict probe miscounted UI around the soft pin knee");
      report << "adaptive-ui soft-knee D3D12 covered=" << covered << " knee_code=" << unsigned(knee_code)
        << " knee_half=0x" << std::hex << knee_half << std::dec << " below_knee_uncounted=1\n";
    }

    std::vector<float> restored(size_t(width) * height);
    require(original_depth.size() == restored.size() * sizeof(float), "Adaptive fixture depth extent changed");
    std::memcpy(restored.data(), original_depth.data(), original_depth.size()); fixture.upload_depth(restored);
    fixture.source_bytes = original; upload(); write_native_source(fixture, backbuffer);
    queue->wait_idle(); frozen.reset_after_runtime_drain();
    report << "adaptive-ui D3D12 renders=" << renders << " asynchronous_identity=1 selected_mask=1 absolute_uv_cap=1 relative_cap50=1 historical75=1 absolute_candidate_counts=1 stable_actual_uv=1 capped_conflict=1 retreat_dwell=1 duplicate_and_loss_hold=1 gated_pending_loss=1 exact_frozen_pixels=1 unchanged_scene=1 v6_dump=1 exported_cursor_ui_parity=1 central75=1 edge_only_no_promotion=1 border_ui_not_denominator=1 conflict_area_exact_counts=1 central_far_ui_cannot_dilute=1 full_frame_ui_protection=1\n";
    std::printf("PASS adaptive D3D12 UI: %u renders; asynchronous GPU absolute-plane counts and stable actual UV, central75 conflict ratio or area and full-frame protection, cap/ramp/retreat, selected retained mask, duplicate/loss hold, exact frozen parity, v6 dump and exported cursor/UI plane parity\n", renders);
  }

  void check_typed_ui_masks_d3d12(fixture_t &fixture, sunshine_game3d::renderer &renderer,
      sunshine_game3d_test::dump_fixture &dump, std::ostream &report, const fs::path &directory) {
    using namespace sunshine_game3d;
    auto *queue = observed.runtime->get_command_queue();
    auto *backbuffer = fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
    const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
    const auto original = fixture.source_bytes, original_depth = fixture.read(fixture.depth.p);
    auto parameters = native_cases().front().parameters;
    parameters.strength = 100; parameters.depth_scale = 100000; parameters.disparity_limit_uv = .01f;
    const ui_plane_parameters plane{ui_plane_mode::display_fraction, .25f};
    fixture.upload_depth(std::vector<float>(size_t(width) * height, .9f));
    struct pixels { std::vector<std::uint8_t> field, color; };
    const auto render = [&](bool enabled, api::resource_view mask, ui_mask_channel channel,
                            const alpha_auto_source *automatic = nullptr, bool capture = false,
                            const char *dump_name = "dump-typed-ui-red") {
      write_native_source(fixture, backbuffer);
      require(sunshine_game3d::test::render_frame(renderer, queue->get_immediate_command_list(), source, fixture.depth_view, parameters,
        enabled, mask, plane, automatic, nullptr, channel), "Typed D3D12 UI render failed");
      if (capture) dump.begin(observed.runtime, renderer, parameters, fixture.depth_view, false,
        static_cast<api::color_space>(fixture.color));
      queue->flush_immediate_command_list(); renderer.finish_present();
      if (capture) dump.submitted(observed.runtime);
      queue->wait_idle();
      if (capture) dump.verify(observed.runtime, [&](api::resource texture, bool common) {
        return fixture.read(reinterpret_cast<ID3D12Resource *>(texture.handle),
          common ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      }, directory / dump_name);
      const auto resources = renderer.diagnostics();
      const auto mode = automatic && automatic->session ? automatic->session->decision().state : alpha_auto_state::manual_off;
      const bool detecting = automatic && automatic->session && mode != alpha_auto_state::manual_on && mode != alpha_auto_state::manual_off;
      require(renderer.consumed_ui_channel() == (detecting ? ui_mask_channel::red : channel),
        "D3D12 UI channel metadata differs from the consumed mask");
      return pixels{fixture.read(reinterpret_cast<ID3D12Resource *>(resources.final_field.handle)),
        fixture.read(reinterpret_cast<ID3D12Resource *>(resources.sbs.handle))};
    };
    const auto equal = [](const pixels &a, const pixels &b, const char *message) {
      require(a.field == b.field && a.color == b.color, message);
    };
    unsigned case_index = 0;
    for (const auto format : {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R32_FLOAT,
                             DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM}) {
      const bool byte_color_mask = format == DXGI_FORMAT_R8G8B8A8_UNORM || format == DXGI_FORMAT_B8G8R8A8_UNORM;
      const bool color_mask = format == DXGI_FORMAT_R16G16B16A16_FLOAT || byte_color_mask;
      // R8 and RGBA are selective; R16 is all white; R32 is all black.
      const auto covered_at = [&](unsigned x, unsigned y) {
        if (format == DXGI_FORMAT_R16_FLOAT) return true;
        if (format == DXGI_FORMAT_R32_FLOAT) return false;
        return x >= width / 3 && x < width / 2 && y >= height / 4 && y < height * 3 / 4;
      };
      const unsigned stride = format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8 :
        format == DXGI_FORMAT_R8_UNORM ? 1 : format == DXGI_FORMAT_R16_FLOAT ? 2 : 4;
      std::vector<std::uint8_t> mask_bytes(size_t(width) * height * stride);
      fixture.source_bytes = original;
      std::uint32_t covered{};
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const auto pixel = size_t(y) * width + x;
        const bool ui = covered_at(x, y); covered += ui;
        const std::uint16_t half = ui ? 0x3400 : 0; // Alpha 0.25 is above the soft pin knee: pinned exactly.
        if (stride == 1) mask_bytes[pixel] = ui ? 64 : 0;
        else if (stride == 2) std::memcpy(mask_bytes.data() + pixel * stride, &half, 2);
        else if (byte_color_mask) {
          const std::uint8_t channels[]{255, 127, 255, static_cast<std::uint8_t>(ui ? 64 : 0)};
          std::memcpy(mask_bytes.data() + pixel * stride, channels, 4); // RGB deliberately disagrees with alpha.
          if (format == DXGI_FORMAT_B8G8R8A8_UNORM) {
            // Exercise Hogwarts' typed snapshot format with independently
            // varying B/G/R and sparse soft/full alpha, never inferred RGB.
            // Every level pins exactly (at or above 32/255, the soft pin knee),
            // as the opaque presented-alpha reference does.
            const std::uint8_t alpha_levels[]{32, 64, 255};
            auto *bgra = mask_bytes.data() + pixel * stride;
            bgra[0] = static_cast<std::uint8_t>((x * 3 + y * 7 + 17) & 255);
            bgra[1] = static_cast<std::uint8_t>((x * 11 + y * 5 + 31) & 255);
            bgra[2] = static_cast<std::uint8_t>((x * 19 + y * 13 + 47) & 255);
            bgra[3] = ui ? alpha_levels[(x + y) % 3] : 0;
          }
        } else if (stride == 4) {
          const float value = ui ? .25f : 0.f;
          std::memcpy(mask_bytes.data() + pixel * stride, &value, 4);
        } else {
          const std::uint16_t channels[]{0x3c00, 0, 0, half}; // Red deliberately disagrees with alpha.
          std::memcpy(mask_bytes.data() + pixel * stride, channels, 8);
        }
        if (fixture.color == 1) fixture.source_bytes[pixel * 4 + 3] = ui ? 255 : 0;
        else if (fixture.color == 2) std::memcpy(fixture.source_bytes.data() + pixel * 8 + 6, &half, 2);
        else {
          std::uint32_t packed{}; std::memcpy(&packed, fixture.source_bytes.data() + pixel * 4, 4);
          packed = (packed & 0x3fffffffu) | (ui ? 0x40000000u : 0u);
          std::memcpy(fixture.source_bytes.data() + pixel * 4, &packed, 4);
        }
      }
      const auto upload_source = [&] {
        void *mapped{}; const D3D12_RANGE no_read{0, 0};
        checked(fixture.source_upload->Map(0, &no_read, &mapped), "Map typed UI source color");
        const auto row = size_t(width) * bytes_per_pixel(fixture.source_format);
        for (unsigned y = 0; y < height; ++y)
          std::memcpy(static_cast<std::uint8_t *>(mapped) + fixture.source_footprint.Offset +
            size_t(y) * fixture.source_footprint.Footprint.RowPitch, fixture.source_bytes.data() + size_t(y) * row, row);
        fixture.source_upload->Unmap(0, nullptr);
      };
      upload_source();
      const auto reference_off = render(false, {}, ui_mask_channel::alpha);
      const auto reference_on = render(true, {}, ui_mask_channel::alpha);
      require(!covered || reference_on.field != reference_off.field, "Typed UI reference does not expose mask application");
      com_ptr<ID3D12Resource> mask, upload;
      fixture.texture(mask, format, D3D12_RESOURCE_STATE_COPY_DEST);
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
      UINT64 bytes{}; const auto desc = mask->GetDesc();
      fixture.game->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
      fixture.buffer(upload, bytes, D3D12_HEAP_TYPE_UPLOAD);
      void *mapped{}; const D3D12_RANGE no_read{0, 0};
      checked(upload->Map(0, &no_read, &mapped), "Map typed UI mask");
      for (unsigned y = 0; y < height; ++y)
        std::memcpy(static_cast<std::uint8_t *>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch,
          mask_bytes.data() + size_t(y) * width * stride, size_t(width) * stride);
      upload->Unmap(0, nullptr);
      fixture.begin_commands();
      D3D12_TEXTURE_COPY_LOCATION from{}, to{};
      from.pResource = upload.p; from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint = footprint;
      to.pResource = mask.p; to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      fixture.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
      transition(fixture.commands.p, mask.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      fixture.submit();
      unsigned copies{};
      const auto copy = [&](api::resource destination) {
        ++copies;
        auto *target = reinterpret_cast<ID3D12Resource *>(destination.handle);
        require(target->GetDesc().Format == format, "Typed UI private allocation changed the captured format");
        auto *commands = reinterpret_cast<ID3D12GraphicsCommandList *>(queue->get_immediate_command_list()->get_native());
        transition(commands, mask.p, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(commands, target, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        commands->CopyResource(target, mask.p);
        transition(commands, target, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        transition(commands, mask.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        return true;
      };
      const auto selected = renderer.prepare_ui_source(10000 + case_index, copy, static_cast<api::format>(format));
      require(selected.handle, "Typed UI private texture unavailable");
      const auto channel = color_mask ? ui_mask_channel::alpha : ui_mask_channel::red;
      equal(render(true, selected, channel), reference_on, "Typed UI field/SBS differs from equivalent color-alpha coverage");
      if (format == DXGI_FORMAT_R8_UNORM || format == DXGI_FORMAT_B8G8R8A8_UNORM) {
        const bool red = format == DXGI_FORMAT_R8_UNORM;
        const char *dump_name = red ? "dump-typed-ui-red" : "dump-typed-ui-bgra";
        equal(render(true, selected, channel, nullptr, true, dump_name), reference_on,
          "Typed UI diagnostic capture changed its field/SBS pixels");
        std::ifstream file(directory / dump_name / "manifest.json");
        require(file.good(), "Typed UI diagnostic dump missing");
        const auto manifest = nlohmann::json::parse(file);
        const auto &replay = manifest.at("producer_metadata").at("replay");
        require(replay.at("ui_parameter_abi") == "sunshine_game3d.ui_parameters.v7" &&
            replay.at("ui_constant_binding").at("uint32")[3] == (red ? 1u : 0u) &&
            replay.at("ui_constant_binding").at("mask_channel") == (red ? "red" : "alpha") &&
            replay.at("ui_alpha_source") == "ui_source_color",
          "Typed UI dump lost its exact selected channel or texture binding");
        bool found_mask = false;
        for (const auto &artifact : manifest.at("artifacts")) if (artifact.at("kind") == "ui_source_color") {
          require(artifact.at("dxgi_format") == unsigned(format) &&
              artifact.at("row_bytes") == size_t(width) * stride && artifact.at("byte_count") == mask_bytes.size(),
            "Typed UI dump changed the source format or packed byte layout");
          const auto path = directory / dump_name / artifact.at("file").get<std::string>();
          require(fs::file_size(path) == mask_bytes.size(), "Typed UI binary dump has the wrong byte count");
          std::ifstream input(path, std::ios::binary);
          std::vector<std::uint8_t> saved(mask_bytes.size());
          input.read(reinterpret_cast<char *>(saved.data()), saved.size());
          require(input.good() && saved == mask_bytes,
            "Consumed UI binary dump changed selective alpha or independent color bytes");
          found_mask = true;
        }
        require(found_mask, "Typed UI dump omitted the immutable mask texture");
        require(fixture.read(mask.p) == mask_bytes &&
            fixture.read(reinterpret_cast<ID3D12Resource *>(renderer.ui_source(static_cast<api::format>(format)).handle)) == mask_bytes,
          "Diagnostic snapshot lifecycle changed the source or renderer-owned mask bytes");
      }
      require(renderer.prepare_ui_source(10000 + case_index, copy, static_cast<api::format>(format)).handle == selected.handle && copies == 1,
        "Repeated immutable UI capture performed another copy or changed its allocation");
      alpha_auto_policy session;
      alpha_auto_source observation;
      observation.now_ms = observation.tick_ms = 400000 + case_index * 1000;
      observation.sequence = 1; observation.epoch = 300 + case_index; observation.revision = 1;
      observation.retained = observation.dedicated_mask = true; observation.session = &session;
      // Before acceptance nothing decides (S1). Accepted by the key an earlier
      // session earned (A1), the typed source decides its own coverage, a full
      // one as a flat frame (P1).
      const auto key = [&](ui_selection::kind kind) {
        return ui_selection::signature{kind, std::uint32_t(api::format_to_default_typed(static_cast<api::format>(format), 0)),
          fixture.color}.key();
      };
      equal(render(true, selected, channel, &observation), reference_off, "Automatic typed UI decided before acceptance");
      const auto restored = session.restore(channel == ui_mask_channel::red ? key(ui_selection::kind::ui_alpha) :
        key(ui_selection::kind::ui_color) + ',' + key(ui_selection::kind::backbuffer));
      require(restored.restored == (channel == ui_mask_channel::red ? 1u : 2u), "Typed UI keys were not restored");
      equal(render(true, selected, channel, &observation), reference_on,
        "Accepted automatic typed UI did not decide its own coverage at once");
      require(renderer.consumed_source_alpha_ui(), "Automatic typed UI did not arm its current-frame mask path");
      auto *resolved = reinterpret_cast<ID3D12Resource *>(renderer.diagnostics().ui_source.handle);
      require(resolved && resolved->GetDesc().Format == DXGI_FORMAT_R32_FLOAT, "Automatic typed UI did not expose R32 mask");
      const auto selected_bytes = fixture.read(resolved);
      require(selected_bytes.size() == size_t(width) * height * sizeof(float), "Automatic typed mask size differs");
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        float value{}; std::memcpy(&value, selected_bytes.data() + (size_t(y) * width + x) * sizeof(float), sizeof(float));
        require(std::isfinite(value) && value >= 0.f && value <= 1.f &&
            (value > 0.f) == covered_at(x, y), "Automatic typed mask has incorrect selected-channel support");
      }
      if (color_mask) {
        observation.dedicated_mask = false;
        equal(render(true, selected, channel, &observation), reference_on,
          "Automatic captured color alpha incorrectly depended on its dedicated tag");
      }
      observation.dedicated_mask = true;
      session.set_manual(true);
      equal(render(true, selected, channel, &observation), reference_on,
        "Manual On failed to admit an available typed UI source");
      session.set_manual(false);
      equal(render(true, selected, channel, &observation), reference_off,
        "Automatic typed UI ignored explicit Manual Off");
      session.set_automatic();
      equal(render(true, selected, channel, &observation), reference_on,
        "Automatic typed UI resume required manual review");
      ++case_index;
    }
    std::vector<float> restored(size_t(width) * height);
    std::memcpy(restored.data(), original_depth.data(), original_depth.size()); fixture.upload_depth(restored);
    fixture.source_bytes = original;
    void *mapped{}; const D3D12_RANGE no_read{0, 0};
    checked(fixture.source_upload->Map(0, &no_read, &mapped), "Restore typed UI fixture source");
    const auto row = size_t(width) * bytes_per_pixel(fixture.source_format);
    for (unsigned y = 0; y < height; ++y)
      std::memcpy(static_cast<std::uint8_t *>(mapped) + fixture.source_footprint.Offset +
        size_t(y) * fixture.source_footprint.Footprint.RowPitch, original.data() + size_t(y) * row, row);
    fixture.source_upload->Unmap(0, nullptr); write_native_source(fixture, backbuffer);
    report << "typed-ui D3D12 R8_selective=1 R16_white=1 R32_black=1 RGBA_alpha_not_red=1 RGBA8_alpha=1 BGRA8_alpha=1 exact_frozen_pixels=1 copy_once=1 unaccepted_decides_nothing=1 accepted_any_coverage=1 resolved_mask_checked=1 manual_modes=1 typed_red_v7_dump=1 typed_bgra_v7_dump=1 selective_alpha_binary_exact=1 source_immutable_after_dump=1\n";
    std::puts("PASS D3D12 typed UI masks: R8/R16/R32/RGBA/BGRA, explicit channels, nothing automatic before acceptance and accepted coverage at any extent, exact field/SBS parity");
  }

  // The soft UI pin band on actual D3D12 GPU fields, through an R32 red mask.
  void check_ui_pin_band_d3d12(fixture_t &fixture, sunshine_game3d::renderer &renderer, std::ostream &report) {
    using namespace sunshine_game3d;
    auto *queue = observed.runtime->get_command_queue();
    auto *backbuffer = fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
    const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
    const auto original_depth = fixture.read(fixture.depth.p);
    auto parameters = native_cases().front().parameters;
    parameters.strength = 100;
    const ui_plane_parameters plane{ui_plane_mode::display_fraction, .25f};
    com_ptr<ID3D12Resource> mask;
    fixture.texture(mask, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    std::uint64_t capture = 20000;
    sunshine_game3d_test::ui_pin_band_fixture band;
    band.width = width; band.height = height;
    band.gain = float(shader_marker(renderer::shader_source(), "SUNSHINE_UI_SOFT_PIN_GAIN"));
    band.depth = [&](bool structured) {
      std::vector<float> depth(size_t(width) * height, 0.f);
      if (structured) for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const bool foreground = x > width / 3 && x < width / 2 && y > height / 5 && y < height * 4 / 5;
        depth[size_t(y) * width + x] = foreground ? .085f : .0005f + .016f * float(x) / width;
      }
      fixture.upload_depth(depth);
    };
    band.render = [&](const std::vector<float> *alpha) {
      api::resource_view view{};
      if (alpha) {
        com_ptr<ID3D12Resource> upload;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        fixture.fill_upload(upload, mask->GetDesc(), alpha->data(), footprint);
        fixture.begin_commands();
        transition(fixture.commands.p, mask.p, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION from{}, to{};
        from.pResource = upload.p; from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint = footprint;
        to.pResource = mask.p; to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        fixture.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        transition(fixture.commands.p, mask.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        fixture.submit();
        view = renderer.prepare_ui_source(++capture, [&](api::resource destination) {
          auto *target = reinterpret_cast<ID3D12Resource *>(destination.handle);
          auto *commands = reinterpret_cast<ID3D12GraphicsCommandList *>(queue->get_immediate_command_list()->get_native());
          transition(commands, mask.p, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
          transition(commands, target, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
          commands->CopyResource(target, mask.p);
          transition(commands, target, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
          transition(commands, mask.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
          return true;
        }, api::format::r32_float);
        require(view.handle, "UI pin band D3D12 mask unavailable");
      }
      write_native_source(fixture, backbuffer);
      require(sunshine_game3d::test::render_frame(renderer, queue->get_immediate_command_list(), source, fixture.depth_view,
        parameters, alpha != nullptr, view, plane, nullptr, nullptr, ui_mask_channel::red), "UI pin band D3D12 render failed");
      queue->flush_immediate_command_list(); renderer.finish_present(); queue->wait_idle();
      const auto bytes = fixture.read(reinterpret_cast<ID3D12Resource *>(renderer.diagnostics().final_field.handle));
      std::vector<float> field(bytes.size() / sizeof(float));
      std::memcpy(field.data(), bytes.data(), field.size() * sizeof(float));
      return field;
    };
    sunshine_game3d_test::check_ui_pin_band(band, report, "D3D12");
    std::vector<float> restored(size_t(width) * height);
    std::memcpy(restored.data(), original_depth.data(), original_depth.size()); fixture.upload_depth(restored);
    write_native_source(fixture, backbuffer);
  }

  // Whether main enabled the D3D12 debug layer (d3d12SDKLayers, the Graphics
  // Tools feature) before the device existed: the pre-UI change-set section
  // then fails on any validation error or corruption message it records.
  bool d3d12_debug_layer = false;
  void require(bool condition, const std::string &message) { require(condition, message.c_str()); }
  // A finite value in the normal half range, rounded to nearest.
  std::uint16_t half_of(float value) {
    if (value == 0.f) return 0;
    const bool negative = value < 0; value = std::abs(value);
    int exponent{}; const float fraction = std::frexp(value, &exponent);
    require(exponent + 14 > 0 && exponent + 14 < 31, "D3D12 UI fixture value outside the normal half range");
    return std::uint16_t((negative ? 0x8000 : 0) | ((exponent + 14) << 10) | unsigned(std::lround((fraction * 2 - 1) * 1024)));
  }

  // Fix 1 (the pre-UI proof), fix 3 (the pre-UI change set) and fix 4 (rule
  // P2, pin only UI) on D3D12 (docs/reshade-sbs.md, UI decision framework),
  // with Stellar Blade SDR's scene as the D3D11 source-alpha fixture builds it
  // (verify_hidden_scene, verify_pre_ui_change_set): a bare cleared offscreen
  // target that holds the scene drawn before the UI at alpha 0, offered one
  // Present late (layer_presents_ago 1) with frame generation known off.
  // Gameplay shows exactly the layer, so the session's ledger proves it the
  // pre-UI scene image by its pixels (texel 11); then, with the presented
  // alpha accepted, the shadow measures the change set against the retained
  // Present (t2, t3, the retention ring copies) through the bit-plane passes
  // (u4 <-> t8), UIPinOnlyUI=1 refines it to source 12 and the darkening
  // passes unpin a dim band, full pages stay flat, T1 reuses once, frame
  // generation is a late pairing and Dump 3D defers the render's own
  // retention. Every count is exact against the CPU construction or
  // game3d_ui_darkening.h's reference. Last, the opacity darkening path on an
  // HDR-like float layer (source 10, linear tolerance, HDR headroom).
  void check_pre_ui_change_set_d3d12(fixture_t &fixture, sunshine_game3d::renderer &renderer,
      sunshine_game3d_test::dump_fixture &dump, std::ostream &report, const fs::path &directory) {
    using namespace sunshine_game3d;
    namespace candidate = ui_detection::candidate;
    if (fixture.color != 1 && fixture.color != 2) {
      std::puts("MEASURE D3D12 pre-UI change set and pin only UI not run for a PQ source (sRGB and scRGB, as the D3D11 fixture)");
      return;
    }
    auto *queue = observed.runtime->get_command_queue();
    auto *device = observed.runtime->get_device();
    auto *backbuffer = fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
    const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
    const auto original = fixture.source_bytes;
    const auto parameters = native_cases().front().parameters;
    const auto pixels = size_t(width) * height;
    const auto all_pixels = std::uint32_t(pixels);
    const unsigned color = fixture.color;
    const auto format = fixture.source_format;
    // A flat depth, as the D3D11 fixture's: the synthetic pages show none of
    // its edges, so no hidden-scene (H1) or still-screen (H2) evidence is
    // valid and only the change set and the darkening decide.
    const auto original_depth = fixture.read(fixture.depth.p);
    const auto depth_desc = fixture.depth->GetDesc();
    fixture.upload_depth(std::vector<float>(size_t(depth_desc.Width) * depth_desc.Height, .02f));
    com_ptr<ID3D12InfoQueue> validation;
    if (d3d12_debug_layer) {
      checked(reinterpret_cast<ID3D12Device *>(device->get_native())->QueryInterface(IID_PPV_ARGS(validation.put())),
        "Query the D3D12 debug layer's info queue");
      validation->ClearStoredMessages();
    }
    using rgba = std::array<float, 4>;
    const auto encode_as = [&](DXGI_FORMAT f, const std::function<rgba(unsigned, unsigned)> &pixel) {
      const auto stride = bytes_per_pixel(f);
      std::vector<std::uint8_t> bytes(pixels * stride);
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const size_t i = size_t(y) * width + x;
        const auto v = pixel(x, y);
        if (f == DXGI_FORMAT_R16G16B16A16_FLOAT)
          for (unsigned c = 0; c != 4; ++c) { const auto half = half_of(v[c]); std::memcpy(bytes.data() + i * stride + c * 2, &half, 2); }
        else
          for (unsigned c = 0; c != 4; ++c) bytes[i * stride + c] = static_cast<std::uint8_t>(std::lround(std::clamp(v[c], 0.f, 1.f) * 255));
      }
      return bytes;
    };
    const auto encode = [&](const std::function<rgba(unsigned, unsigned)> &pixel) { return encode_as(format, pixel); };
    // The values the shader loads from an encoded image.
    const auto decode_as = [&](DXGI_FORMAT f, const std::vector<std::uint8_t> &bytes) {
      const auto stride = bytes_per_pixel(f);
      std::vector<rgba> result(pixels);
      for (size_t i = 0; i != pixels; ++i)
        for (unsigned c = 0; c != 4; ++c) {
          if (f == DXGI_FORMAT_R16G16B16A16_FLOAT) {
            std::uint16_t half; std::memcpy(&half, bytes.data() + i * stride + c * 2, 2); result[i][c] = half_float(half);
          } else result[i][c] = float(bytes[i * stride + c]) / 255.f;
        }
      return result;
    };
    const auto colours = [&](const std::vector<std::uint8_t> &bytes) {
      std::vector<ui_darkening::rgb> result(pixels);
      const auto loaded = decode_as(format, bytes);
      for (size_t i = 0; i != pixels; ++i) result[i] = {loaded[i][0], loaded[i][1], loaded[i][2]};
      return result;
    };
    // Game-owned layer textures rest in ALL_SHADER_RESOURCE between uploads.
    std::vector<api::resource_view> views;
    const auto upload_texture = [&](ID3D12Resource *texture, const std::vector<std::uint8_t> &bytes, D3D12_RESOURCE_STATES before) {
      com_ptr<ID3D12Resource> upload;
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
      fixture.fill_upload(upload, texture->GetDesc(), bytes.data(), footprint);
      fixture.begin_commands();
      if (before != D3D12_RESOURCE_STATE_COPY_DEST) transition(fixture.commands.p, texture, before, D3D12_RESOURCE_STATE_COPY_DEST);
      D3D12_TEXTURE_COPY_LOCATION from{}, to{};
      from.pResource = upload.p; from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint = footprint;
      to.pResource = texture; to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      fixture.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
      transition(fixture.commands.p, texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      fixture.submit();
    };
    const auto create_layer = [&](com_ptr<ID3D12Resource> &texture, DXGI_FORMAT f, const std::vector<std::uint8_t> &bytes) {
      fixture.texture(texture, f, D3D12_RESOURCE_STATE_COPY_DEST);
      upload_texture(texture.p, bytes, D3D12_RESOURCE_STATE_COPY_DEST);
      api::resource_view view{};
      require(device->create_resource_view({reinterpret_cast<std::uint64_t>(texture.p)}, api::resource_usage::shader_resource,
        api::resource_view_desc(static_cast<api::format>(f)), &view), "Create a D3D12 UI layer view");
      views.push_back(view);
      return view;
    };
    // The presented colour: the game's backbuffer for the next render.
    const auto present = [&](const std::vector<std::uint8_t> &bytes) {
      fixture.source_bytes = bytes;
      void *mapped{}; const D3D12_RANGE no_read{0, 0};
      checked(fixture.source_upload->Map(0, &no_read, &mapped), "Map the D3D12 pre-UI presented colour");
      const auto row = size_t(width) * bytes_per_pixel(format);
      for (unsigned y = 0; y < height; ++y)
        std::memcpy(static_cast<std::uint8_t *>(mapped) + fixture.source_footprint.Offset +
          size_t(y) * fixture.source_footprint.Footprint.RowPitch, bytes.data() + size_t(y) * row, row);
      fixture.source_upload->Unmap(0, nullptr);
      write_native_source(fixture, backbuffer);
    };
    const auto read_mask = [&] {
      auto *texture = reinterpret_cast<ID3D12Resource *>(renderer.diagnostics().ui_source.handle);
      require(texture && texture->GetDesc().Format == DXGI_FORMAT_R32_FLOAT, "D3D12 UI detection exposed no R32 mask");
      const auto bytes = fixture.read(texture);
      require(bytes.size() == pixels * sizeof(float), "D3D12 detected mask size differs");
      std::vector<float> mask(pixels);
      std::memcpy(mask.data(), bytes.data(), bytes.size());
      return mask;
    };
    struct outcome { alpha_auto_decision sample; ui_detection_snapshot run; std::vector<float> mask; };
    // One real Present: begin, render, the retention a dump capture owes,
    // signal, drain (test-only), then the detected mask.
    const auto render = [&](const std::vector<std::uint8_t> &presented, alpha_auto_source &observation,
        const ui_render_input &ui, const fs::path &dump_directory = {}) {
      present(presented);
      observation.now_ms += 100; observation.tick_ms = observation.now_ms; ++observation.sequence;
      renderer.begin_present();
      render_frame_input frame;
      frame.color = source; frame.depth = fixture.depth_view; frame.scene = parameters; frame.ui = ui;
      auto *commands = queue->get_immediate_command_list();
      require(renderer.render(commands, frame), "D3D12 pre-UI detection render failed");
      const bool dumped = !dump_directory.empty();
      if (dumped) dump.begin(observed.runtime, renderer, parameters, fixture.depth_view, false, static_cast<api::color_space>(color));
      renderer.finish_retention(commands);
      queue->flush_immediate_command_list();
      renderer.finish_present();
      if (dumped) dump.submitted(observed.runtime);
      queue->wait_idle(); // Test-only drain: the next render reads this one's sample.
      if (dumped) dump.verify(observed.runtime, [&](api::resource texture, bool common) {
        return fixture.read(reinterpret_cast<ID3D12Resource *>(texture.handle),
          common ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      }, dump_directory);
      require(fixture.read(backbuffer, D3D12_RESOURCE_STATE_PRESENT) == presented, "D3D12 UI detection changed the game's backbuffer");
      return outcome{renderer.consumed_alpha_auto(), renderer.consumed_detection(), read_mask()};
    };
    const auto all_equal = [](const std::vector<float> &mask, float value) {
      return std::all_of(mask.begin(), mask.end(), [value](float v) { return v == value; });
    };

    // The UI of the Equipment page (verify_pre_ui_change_set): a tab row, the
    // left icon panel, the bottom-right prompts, a one-pixel hint line and an
    // isolated speck. A changed pixel stays with at least 3 changed pixels in
    // its 3x3 window: the speck and the line's two ends go.
    std::vector<std::uint8_t> ui(pixels, 0);
    const auto fill_ui = [&](unsigned x0, unsigned y0, unsigned x1, unsigned y1) {
      for (unsigned y = y0; y < y1 && y < height; ++y) for (unsigned x = x0; x < x1 && x < width; ++x) ui[size_t(y) * width + x] = 1;
    };
    fill_ui(width / 4, height / 20, width * 3 / 4, height / 20 + std::max(2u, height / 40));
    fill_ui(width / 40, height / 4, width / 40 + std::max(2u, width / 20), height * 3 / 4);
    fill_ui(width * 3 / 4, height * 9 / 10, width * 19 / 20, height * 19 / 20);
    fill_ui(width / 2, height / 2, width / 2 + width / 10, height / 2 + 1);
    fill_ui(width * 3 / 5, height * 3 / 10, width * 3 / 5 + 1, height * 3 / 10 + 1);
    std::vector<std::uint8_t> filtered(pixels, 0);
    std::uint32_t ui_pixels = 0, kept = 0;
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
      const size_t i = size_t(y) * width + x;
      if (!ui[i]) continue;
      ++ui_pixels;
      unsigned count = 0;
      for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
        const int nx = int(x) + dx, ny = int(y) + dy;
        if (nx >= 0 && ny >= 0 && nx < int(width) && ny < int(height)) count += ui[size_t(ny) * width + nx];
      }
      if (count >= ui_detection::change_set::min_neighbourhood) { filtered[i] = 1; ++kept; }
    }
    require(kept < ui_pixels && size_t(kept) * 4 < pixels, "The D3D12 fixture's UI has nothing for the 3x3 rule to remove, or is not selective");
    // A moving scene (checkers that shift every frame, lit everywhere), the
    // page's presented colour over it (opaque), and the bottom dim band of
    // fix 4 (opacity rising to 0.3 over the bottom fifth, under the UI).
    const auto scene = [&](unsigned phase, bool lit = true) {
      return [=](unsigned x, unsigned y) -> rgba {
        return lit ? rgba{0.f, ((x + 3 * phase) / 4 + y / 4) % 2 ? .75f : .25f, .25f, 1.f} : rgba{0.f, 0.f, 0.f, 1.f};
      };
    };
    enum class page { equipment, settings, loading, dimmed };
    const unsigned dim_rows = height / 5, dim_top = height - dim_rows;
    const auto presented_of = [&](page kind, unsigned phase) {
      const auto base = scene(phase, kind != page::loading);
      return encode([&](unsigned x, unsigned y) -> rgba {
        if (kind == page::settings) return {.5f, .5f, .5f, 1.f};
        if (ui[size_t(y) * width + x]) return rgba{1.f, .5f, 1.f, 1.f};
        auto v = base(x, y);
        if (kind == page::dimmed && y >= dim_top) {
          const float keep = 1.f - .3f * float(y - dim_top + 1) / float(dim_rows + 1);
          for (unsigned c = 0; c != 3; ++c) v[c] *= keep;
        }
        return v;
      });
    };
    // The cleared target: the pre-UI scene at alpha 0.
    const auto layer_of = [&](page kind, unsigned phase) {
      const auto base = scene(phase, kind != page::loading);
      return encode([&](unsigned x, unsigned y) -> rgba { auto v = base(x, y); v[3] = 0.f; return v; });
    };
    com_ptr<ID3D12Resource> layer_texture;
    const auto layer_view = create_layer(layer_texture, format, layer_of(page::equipment, 0));
    const auto layer_format = static_cast<api::format>(format);
    const auto layer_flags = ui_layer::detection_flags(layer_format);
    const auto typed_format = std::uint32_t(api::format_to_default_typed(layer_format, 0));
    const ui_selection::signature layer_signature{ui_selection::kind::ui_layer, typed_format, color};
    const auto current_key = ui_selection::signature{ui_selection::kind::current, typed_format, color}.key();

    // (1) Fix 1: an unproven layer, nothing accepted. Gameplay shows exactly
    // the scene the layer holds, lit everywhere, so the third matching sample
    // 2 s after the first proves it: the 22nd frame reads that sample (a frame
    // reads the previous frame's sample, 100 ms apart), as on D3D11.
    alpha_auto_policy sb;
    alpha_auto_source observation;
    observation.session = &sb; observation.now_ms = observation.tick_ms = 500000;
    observation.epoch = 61; observation.revision = 1; observation.sequence = 1;
    ui_detection_inputs inputs;
    inputs.current_color = true; inputs.layer = layer_view; inputs.layer_flags = layer_flags;
    inputs.layer_presents_ago = 1; inputs.fg_known_off = true;
    ui_render_input ui_input;
    ui_input.automatic = &observation; ui_input.detection = &inputs;
    const auto gameplay = encode(scene(0));
    unsigned proving_frames = 0;
    outcome proving{};
    while (!sb.pre_ui_proven(layer_signature) && proving_frames != 30) {
      proving = render(gameplay, observation, ui_input);
      ++proving_frames;
      const auto &e = proving.sample.evidence;
      require(all_equal(proving.mask, 0.f) && !proving.sample.source_kind && (proving_frames == 1 ||
          (e.pre_ui_match == all_pixels && e.pre_ui_image_lit == all_pixels && e.presented_lit == all_pixels && !e.presented_lit_differs)),
        "D3D12 gameplay before the proof pinned, or its texel 11 pixel counts are not exact");
    }
    require(proving_frames == 22 && proving.sample.scene_guard.proven &&
        (renderer.consumed_detection().flags & ui_detection::per_frame_pre_ui_proven) &&
        sb.stored() == ui_selection::pre_ui_key(layer_signature).key(),
      "D3D12 gameplay did not prove the bare cleared layer by its pixels at the sample 2 s after the first: " +
        std::to_string(proving_frames));
    // The presented alpha (a uniform 1.0 in menus) is accepted.
    require(sb.restore(current_key).restored == 1, "The D3D12 current-alpha key was not restored");

    unsigned phase = 0;
    page last = page::equipment;
    // The presented colours of the last three renders, newest last.
    std::array<std::vector<std::uint8_t>, 3> presented_history;
    // One render of a page at the next phase. The layer offered is the
    // previous render's pre-UI image (exact when presents_ago is 1).
    const auto step = [&](page kind, std::uint32_t presents_ago = 1, bool fg = false, const fs::path &dump_directory = {}) {
      upload_texture(layer_texture.p, layer_of(last, phase), D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      ++phase;
      last = kind;
      inputs.layer_presents_ago = presents_ago; inputs.fg_known_off = !fg;
      auto presented = presented_of(kind, phase);
      std::rotate(presented_history.begin(), presented_history.begin() + 1, presented_history.end());
      presented_history.back() = presented;
      return render(presented, observation, ui_input, dump_directory);
    };
    const auto is_flat = [&](const std::vector<float> &mask) { return all_equal(mask, 1.f); };
    const auto is_filtered_ui = [&](const std::vector<float> &mask) {
      for (size_t i = 0; i != pixels; ++i) if (mask[i] != float(filtered[i])) return false;
      return true;
    };
    const auto pre_ui_rules = [](const ui_detection_snapshot &run) {
      return run.rules_bits & (ui_detection::rules::pin_only_ui | ui_detection::change_set::pair_mask);
    };
    // (2) The shadow (UIPinOnlyUI=0). The first page frame pairs the copy of
    // gameplay; the second's sample is the first that shows the page.
    step(page::equipment);
    step(page::equipment);
    const auto stored = sb.stored();
    const auto counters_before = sb.counters();
    outcome shadow{};
    for (unsigned i = 0; i != 4; ++i) {
      shadow = step(page::equipment);
      require(!(shadow.run.candidates & candidate::pre_ui) && shadow.run.layer_pairing == change_set::pair_class::retained &&
          shadow.run.layer_presents_ago == 1u && !(shadow.run.rules_bits & ui_detection::rules::pin_only_ui) &&
          ui_detection::change_set::pair_offset(shadow.run.rules_bits) == 1u && is_flat(shadow.mask),
        "The D3D12 shadow offered the pre-UI change set, lost the retained pairing or did not leave the menu flat");
    }
    const auto &cs = shadow.sample.evidence.change_set;
    require(shadow.sample.source_kind == 4u && shadow.sample.covered == all_pixels && !shadow.sample.evidence.refined &&
        shadow.sample.change_set.measured && shadow.sample.change_set.pairing == change_set::pair_class::retained &&
        shadow.sample.change_set.offset == 1u && shadow.sample.change_set.valid && shadow.sample.change_set.would_refine &&
        !shadow.sample.change_set.enabled && cs.changed == ui_pixels && cs.filtered == kept && cs.changed_1 == ui_pixels &&
        cs.changed_2 > ui_pixels && cs.matching_tiles >= 128u && !cs.nonfinite && !cs.judge_kind,
      "The D3D12 shadow sample did not measure exactly the UI pixels against the retained Present: changed=" +
        std::to_string(cs.changed) + " filtered=" + std::to_string(cs.filtered) + " ui=" + std::to_string(ui_pixels) +
        " kept=" + std::to_string(kept) + " changed_1=" + std::to_string(cs.changed_1) + " changed_2=" +
        std::to_string(cs.changed_2) + " tiles=" + std::to_string(cs.matching_tiles));
    const auto counted = sb.counters() - counters_before;
    require(sb.stored() == stored && sb.pre_ui_proven(layer_signature) && counted[ui_counter::change_set_samples] >= 3 &&
        counted[ui_counter::change_set_retained] == counted[ui_counter::change_set_samples] &&
        counted[ui_counter::change_set_valid] == counted[ui_counter::change_set_samples] &&
        counted[ui_counter::change_set_would_refine] == counted[ui_counter::change_set_samples] &&
        counted[ui_counter::change_set_pair_verified] == counted[ui_counter::change_set_samples] &&
        !counted[ui_counter::change_set_pair_contradicted] && !counted[ui_counter::refined] && !counted.decided(12) &&
        counted.decided(4) >= 1,
      "The D3D12 shadow changed the ledger, or its change-set counters are not exact");
    const auto shadow_samples = counted[ui_counter::change_set_samples];
    // (3) UIPinOnlyUI=1: offered with the retained pairing and refined; only
    // the changed pixels the 3x3 rule keeps pin.
    sb.set_pin_only_ui(true);
    outcome pinned{};
    const auto mask_differs = [&](const std::vector<float> &mask) {
      std::uint64_t differs = 0;
      for (size_t i = 0; i != pixels; ++i) differs += mask[i] != float(filtered[i]);
      return differs;
    };
    for (unsigned i = 0; i != 3; ++i) {
      pinned = step(page::equipment);
      require((pinned.run.candidates & candidate::pre_ui) && (pinned.run.accepted & candidate::pre_ui) &&
          !(pinned.run.candidates & candidate::exact) &&
          pre_ui_rules(pinned.run) == (ui_detection::rules::pin_only_ui | (1u << ui_detection::change_set::pair_shift)) &&
          is_filtered_ui(pinned.mask),
        "D3D12 UIPinOnlyUI=1 did not offer the pre-UI change set or pin exactly the changed pixels the 3x3 rule keeps: frame " +
          std::to_string(i) + " candidates=" + std::to_string(pinned.run.candidates) + " accepted=" +
          std::to_string(pinned.run.accepted) + " rules=" + std::to_string(pinned.run.rules_bits) + " differing_pixels=" +
          std::to_string(mask_differs(pinned.mask)));
    }
    require(pinned.sample.source_kind == ui_detection::source_pre_ui && pinned.sample.covered == ui_pixels &&
        pinned.sample.evidence.refined && pinned.sample.evidence.s1_source == ui_detection::source_pre_ui &&
        pinned.sample.change_set.enabled && pinned.sample.change_set.would_refine && sb.stored() == stored,
      "The D3D12 refined sample did not report source 12 with the refined bit");
    // (4) Fix 4, rule P2: the page with its bottom dim band. The band's
    // changed pixels do not pin unless structure keeps them, exactly as the
    // reference of the pair says; the UI's colour pixels keep their pins.
    std::uint64_t dimmed_unpinned = 0, dimmed_kept = 0;
    {
      step(page::dimmed);
      const auto pre_ui = colours(layer_of(page::dimmed, phase));
      const auto dimmed = step(page::dimmed);
      const auto paired = colours(presented_history[1]);
      float t;
      std::memcpy(&t, &dimmed.run.threshold_bits, sizeof(t));
      const auto k = float(ui_detection::change_set::inferred_scale);
      const auto mask = ui_darkening::change_set_mask(width, height, paired, pre_ui, t, k, color, true);
      const auto reference = ui_darkening::reference_change_set(width, height, paired, pre_ui, mask, t, k, color);
      std::uint64_t band_unpinned = 0, band_changed = 0, differs = 0;
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const auto i = size_t(y) * width + x;
        const bool unpinned = reference.classes[i] == ui_darkening::pixel_class::unpinned;
        differs += dimmed.mask[i] != (mask[i] && !unpinned ? 1.f : 0.f);
        require(!filtered[i] || (mask[i] && reference.classes[i] == ui_darkening::pixel_class::colour),
          "A UI colour pixel of the D3D12 dimmed page was not pinned as colour by the reference");
        if (y >= dim_top && !ui[i] && mask[i]) { ++band_changed; band_unpinned += unpinned; }
      }
      require(dimmed.sample.source_kind == ui_detection::source_pre_ui && (dimmed.run.candidates & candidate::pre_ui) &&
          pre_ui_rules(dimmed.run) == (ui_detection::rules::pin_only_ui | (1u << ui_detection::change_set::pair_shift)) &&
          (dimmed.run.rules_bits & ui_detection::rules::darkening_measured) && !differs,
        "The D3D12 dimmed page's mask differs from the CPU reference on " + std::to_string(differs) + " pixels");
      require(band_changed && band_unpinned * 2 >= band_changed && reference.n.unpinned == band_unpinned && reference.n.kept,
        "The D3D12 dimmed page's band was not mostly unpinned by the reference, nothing was kept, or something else unpinned: " +
          std::to_string(band_unpinned) + " of " + std::to_string(band_changed));
      const auto next = step(page::dimmed);
      require(next.sample.source_kind == ui_detection::source_pre_ui && next.sample.darkening.measured &&
          next.sample.darkening.applied && next.sample.darkening.unpinned == reference.n.unpinned &&
          next.sample.darkening.kept == reference.n.kept,
        "The D3D12 dimmed page's sample words 62-63 differ from the CPU reference: unpinned=" +
          std::to_string(next.sample.darkening.unpinned) + " kept=" + std::to_string(next.sample.darkening.kept) + " reference " +
          std::to_string(reference.n.unpinned) + "/" + std::to_string(reference.n.kept));
      dimmed_unpinned = reference.n.unpinned;
      dimmed_kept = reference.n.kept;
      step(page::equipment);
    }
    // (5) A full settings page (an invalid set) and a black loading screen
    // stay flat by the whole-frame alpha; checked from their second frame.
    for (const auto kind : {page::settings, page::loading}) {
      const std::string label = kind == page::settings ? "D3D12 settings page" : "D3D12 black loading screen";
      step(kind);
      const auto second = step(kind);
      const auto flat = step(kind);
      require((flat.run.candidates & candidate::pre_ui) && is_flat(second.mask) && is_flat(flat.mask) &&
          flat.sample.source_kind == 4u && !flat.sample.evidence.refined && flat.sample.change_set.measured &&
          !flat.sample.change_set.valid,
        label + ": the pre-UI change set refined an invalid set");
    }
    step(page::equipment);
    require(is_filtered_ui(step(page::equipment).mask), "The D3D12 Equipment page did not refine again after a full page");
    // (6) T1: a frame without its pair reuses the refined decision once
    // through the change-set gap; frame generation is a late pairing, never
    // offered, and so is the first Present after it.
    const auto missing = step(page::equipment, 0);
    require(!(missing.run.candidates & candidate::pre_ui) && missing.run.layer_pairing == change_set::pair_class::unavailable &&
        (missing.run.rules_bits & ui_detection::change_set::gap) &&
        !(missing.run.flags & ui_detection::per_frame_accepted_missing) && is_filtered_ui(missing.mask),
      "A D3D12 frame without its pair did not reuse the refined decision once through the change-set gap");
    const auto spent = step(page::equipment, 0);
    require(!(spent.run.candidates & candidate::pre_ui) && is_flat(spent.mask) && spent.sample.evidence.reused,
      "A second D3D12 frame without its pair reused again, or the sample lost the reuse");
    const auto generated = step(page::equipment, 1, true);
    require(!(generated.run.candidates & candidate::pre_ui) && generated.run.layer_pairing == change_set::pair_class::late &&
        !ui_detection::change_set::pair_offset(generated.run.rules_bits) && is_flat(generated.mask),
      "D3D12 frame generation offered the pre-UI change set or paired the layer with a retained Present");
    const auto late = step(page::equipment, 1, true);
    require(late.sample.change_set.pairing == change_set::pair_class::late && late.sample.change_set.offset == 0u &&
        !late.sample.change_set.valid && late.sample.evidence.change_set.changed > ui_pixels,
      "The D3D12 late pair's shadow was not measured against the current Present");
    const auto first_off = step(page::equipment);
    require(!(first_off.run.candidates & candidate::pre_ui) && first_off.run.layer_pairing == change_set::pair_class::late &&
        is_flat(first_off.mask), "The first D3D12 Present after frame generation paired the Present before it");
    require(is_filtered_ui(step(page::equipment).mask), "The D3D12 retained pairing did not refine again after frame generation");
    // (7) Dump 3D: an armed dump retains every Present and the dumped render
    // defers its own retention copy (finish_retention), so the package holds
    // both retained Presents and the consumed layer copy; the next render
    // still pairs.
    const auto dump_directory = directory / "pre-ui-change-set-dump-d3d12";
    renderer.set_dump_retention(true, true);
    const auto dumped = step(page::equipment, 1, false, dump_directory);
    renderer.set_dump_retention(false, false);
    require(is_filtered_ui(dumped.mask), "The D3D12 dumped render did not refine");
    {
      std::ifstream stored_manifest(dump_directory / "manifest.json");
      require(stored_manifest.good(), "The D3D12 pre-UI dump has no manifest");
      const auto manifest = nlohmann::json::parse(stored_manifest);
      const auto &metadata = manifest.at("producer_metadata");
      const auto &detection = metadata.at("replay").at("ui_detection");
      std::set<std::string> kinds;
      std::map<std::string, std::vector<std::uint8_t>> retained_bytes;
      for (const auto &artifact : manifest.at("artifacts")) {
        const auto kind = artifact.at("kind").get<std::string>();
        kinds.insert(kind);
        if (kind.rfind("retained_present_", 0) == 0) {
          std::ifstream file(dump_directory / artifact.at("file").get<std::string>(), std::ios::binary);
          retained_bytes[kind] = {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        }
      }
      bool all_captured = true;
      for (const auto &row : metadata.at("change_set_artifacts").at("artifacts")) all_captured = all_captured && row.at("captured").get<bool>();
      require(kinds.count("retained_present_1") && kinds.count("retained_present_2") && kinds.count("ui_layer_detected") &&
          all_captured && detection.at("layer_pairing") == "retained" && detection.at("layer_presents_ago") == 1u &&
          detection.at("rules_bits") == dumped.run.rules_bits &&
          metadata.at("replay").at("ui_pin").at("decision_texels") == ui_detection::change_set_decision_texels &&
          retained_bytes["retained_present_1"] == presented_history[1] && retained_bytes["retained_present_2"] == presented_history[0],
        "D3D12 Dump 3D lost the retained Presents (one and two back), the consumed layer copy or the pairing metadata");
    }
    const auto after_dump = step(page::equipment);
    require((after_dump.run.candidates & candidate::pre_ui) && after_dump.run.layer_pairing == change_set::pair_class::retained &&
        is_filtered_ui(after_dump.mask), "The D3D12 render after a dump lost its retained pair");
    sb.set_pin_only_ui(false);
    report << "pre-ui-change-set D3D12 proof_frames=" << proving_frames << " ui=" << ui_pixels << " kept=" << kept
           << " shadow_samples=" << shadow_samples << " shadow_retained_valid_would_refine=1 ledger_unchanged=1 refined_source12=1"
              " settings_flat=1 loading_flat=1 t1_reuse_once=1 fg_late_never_offered=1 dump_retained_presents=1"
              " dimmed_band_unpinned=" << dimmed_unpinned << " dimmed_kept=" << dimmed_kept << '\n';
    std::printf("PASS D3D12 pre-UI change set (fixes 1, 3, 4; Stellar Blade SDR): %u gameplay frames prove the bare cleared layer by exact texel 11 counts; the shadow measures exactly the UI pixels (%u, %u after the 3x3 rule) against the retained Present through the bit-plane passes, verified against the Presents 0 and 2 back, with exact change-set counters and an unchanged ledger; UIPinOnlyUI=1 refines to source 12; the dim band unpins %llu pixels and keeps %llu exactly as the CPU reference; full pages stay flat, T1 reuses once, frame generation never pairs, and Dump 3D defers the retention copy\n",
      proving_frames, ui_pixels, kept, static_cast<unsigned long long>(dimmed_unpinned), static_cast<unsigned long long>(dimmed_kept));

    // (8) The opacity darkening path on an HDR-like float layer (source 10):
    // a smooth black dim band (alpha rising to 0.6), sharp black 2-pixel
    // strokes, an HDR white icon at 4x SDR white (beyond the UNORM layer's
    // premultiplied bound, within the float layer's headroom) and a faint
    // grey panel whose premultiplied colour (0.003) is colour in linear light
    // but would read as a dim against the sRGB code tolerance (4/255). In the
    // shadow the mask is the raw alpha and the samples carry the reference's
    // counts; with UIPinOnlyUI=1 only the band unpins.
    std::uint64_t hdr_band = 0, hdr_strokes = 0, hdr_unpinned = 0, hdr_kept = 0;
    {
      constexpr DXGI_FORMAT hdr_format = DXGI_FORMAT_R16G16B16A16_FLOAT;
      std::vector<rgba> value(pixels, rgba{0.f, 0.f, 0.f, 0.f});
      // 1 the dim band, 2 a stroke, 3 the HDR icon, 4 the faint panel.
      std::vector<std::uint8_t> region(pixels, 0);
      const unsigned band = height * 3 / 8, top = height - band;
      for (unsigned y = top; y < height; ++y)
        for (unsigned x = 0; x < width; ++x) {
          value[size_t(y) * width + x] = {0.f, 0.f, 0.f, .6f * float(y - top + 1) / float(band + 1)};
          region[size_t(y) * width + x] = 1;
        }
      const auto fill = [&](unsigned x0, unsigned y0, unsigned x1, unsigned y1, rgba v, std::uint8_t r) {
        for (unsigned y = y0; y < y1 && y < top; ++y)
          for (unsigned x = x0; x < x1 && x < width; ++x) { value[size_t(y) * width + x] = v; region[size_t(y) * width + x] = r; }
      };
      const rgba black{0.f, 0.f, 0.f, 1.f}, hdr_white{4.f, 4.f, 4.f, 1.f}, faint{.003f, .003f, .003f, .5f};
      const unsigned stroke = 2, glyph = std::max(8u, height / 4), left = width / 8, row = height / 8;
      fill(left, row, left + glyph, row + stroke, black, 2);
      fill(left + glyph / 2 - 1, row + stroke + 2, left + glyph / 2 + 1, row + glyph, black, 2);
      fill(left + glyph + 4, row, left + glyph + 4 + stroke, row + glyph - stroke - 2, black, 2);
      fill(left + glyph + 8, row + glyph - stroke, left + 2 * glyph, row + glyph, black, 2);
      const unsigned icon_w = std::max(4u, width / 10), icon_h = std::max(4u, height / 8);
      fill(width * 5 / 8, row, width * 5 / 8 + icon_w, row + icon_h, hdr_white, 3);
      fill(width * 5 / 8, row + icon_h + 8, width * 5 / 8 + icon_w, row + 2 * icon_h + 8, faint, 4);
      const auto bytes = encode_as(hdr_format, [&](unsigned x, unsigned y) { return value[size_t(y) * width + x]; });
      const auto loaded = decode_as(hdr_format, bytes);
      std::vector<ui_darkening::rgb> colour(pixels);
      std::vector<float> alpha(pixels);
      for (size_t i = 0; i != pixels; ++i) { colour[i] = {loaded[i][0], loaded[i][1], loaded[i][2]}; alpha[i] = loaded[i][3]; }
      // A float layer's colour is linear (stored_hdr_headroom).
      const auto reference = ui_darkening::reference_opacity(width, height, colour, alpha, true);
      const auto code_tolerance = ui_darkening::reference_opacity(width, height, colour, alpha, false);
      std::uint64_t faint_dim_in_code = 0;
      for (size_t i = 0; i != pixels; ++i) {
        const auto c = reference.classes[i];
        const bool band_ui = region[i] == 1 && alpha[i] > 0.f;
        hdr_band += band_ui;
        hdr_strokes += region[i] == 2;
        faint_dim_in_code += region[i] == 4 && code_tolerance.classes[i] == ui_darkening::pixel_class::unpinned;
        require(band_ui ? c == ui_darkening::pixel_class::unpinned : region[i] == 2 ? c == ui_darkening::pixel_class::kept :
            region[i] >= 3 ? c == ui_darkening::pixel_class::colour : c == ui_darkening::pixel_class::not_ui,
          "The D3D12 HDR darkening fixture does not mean what it says (pixel " + std::to_string(i % width) + "," +
            std::to_string(i / width) + " region " + std::to_string(region[i]) + " class " + std::to_string(int(c)) + ")");
      }
      require(reference.n.unpinned == hdr_band && reference.n.kept == hdr_strokes && faint_dim_in_code,
        "The D3D12 HDR darkening fixture's reference counts differ, or its faint panel does not separate linear from code tolerance");
      hdr_unpinned = reference.n.unpinned; hdr_kept = reference.n.kept;
      com_ptr<ID3D12Resource> hdr_texture;
      const auto hdr_view = create_layer(hdr_texture, hdr_format, bytes);
      const auto hdr_api_format = static_cast<api::format>(hdr_format);
      alpha_auto_policy policy;
      require(policy.restore(ui_selection::signature{ui_selection::kind::ui_layer,
          std::uint32_t(api::format_to_default_typed(hdr_api_format, 0)), color}.key()).restored == 1,
        "The D3D12 HDR layer key was not restored");
      alpha_auto_source layer_observation;
      layer_observation.session = &policy; layer_observation.now_ms = layer_observation.tick_ms = 2000000;
      layer_observation.epoch = 71; layer_observation.revision = 1; layer_observation.sequence = 1;
      ui_detection_inputs layer_inputs;
      layer_inputs.layer = hdr_view;
      layer_inputs.layer_flags = ui_layer::detection_flags(hdr_api_format);
      require(layer_inputs.layer_flags & ui_detection::stored_hdr_headroom, "The float layer has no HDR headroom flag");
      ui_render_input layer_ui;
      layer_ui.automatic = &layer_observation; layer_ui.detection = &layer_inputs;
      const auto frame = [&](bool enabled) {
        policy.set_pin_only_ui(enabled);
        return render(original, layer_observation, layer_ui);
      };
      const auto masked = [&](const std::vector<float> &mask, bool enabled) {
        for (size_t i = 0; i != pixels; ++i) {
          const float expected = enabled && reference.classes[i] == ui_darkening::pixel_class::unpinned ? 0.f : alpha[i];
          if (!(std::fabs(mask[i] - expected) <= 1e-6f)) return false;
        }
        return true;
      };
      const auto rules = [](const ui_detection_snapshot &run) {
        return run.rules_bits & (ui_detection::rules::pin_only_ui | ui_detection::rules::darkening_measured);
      };
      const auto layer_counters_before = policy.counters();
      outcome layer_shadow{};
      for (unsigned i = 0; i != 4; ++i) {
        layer_shadow = frame(false);
        require(layer_shadow.run.state == ui_detection_snapshot::run_state::ran &&
            rules(layer_shadow.run) == ui_detection::rules::darkening_measured && masked(layer_shadow.mask, false),
          "The D3D12 shadow changed the HDR layer's raw alpha mask or did not run the darkening passes");
        if (i)
          require(layer_shadow.sample.source_kind == ui_detection::source_layer && layer_shadow.sample.covered == reference.n.covered &&
              layer_shadow.sample.darkening.measured && !layer_shadow.sample.darkening.applied &&
              layer_shadow.sample.darkening.unpinned == reference.n.unpinned && layer_shadow.sample.darkening.kept == reference.n.kept,
            "A D3D12 HDR layer shadow sample's words 62-63 differ from the CPU reference: unpinned=" +
              std::to_string(layer_shadow.sample.darkening.unpinned) + " kept=" + std::to_string(layer_shadow.sample.darkening.kept) +
              " reference " + std::to_string(reference.n.unpinned) + "/" + std::to_string(reference.n.kept));
      }
      const auto layer_counted = policy.counters() - layer_counters_before;
      require(layer_counted[ui_counter::darkening_samples] >= 2 &&
          layer_counted[ui_counter::darkening_unpinned_samples] == layer_counted[ui_counter::darkening_samples] &&
          layer_counted[ui_counter::darkening_unpinned_px] == layer_counted[ui_counter::darkening_samples] * reference.n.unpinned &&
          layer_counted[ui_counter::darkening_kept_px] == layer_counted[ui_counter::darkening_samples] * reference.n.kept &&
          layer_counted.decided(ui_detection::source_layer) >= 1,
        "The D3D12 darkening counters do not sum the committed HDR layer samples");
      outcome layer_pinned{};
      for (unsigned i = 0; i != 3; ++i) {
        layer_pinned = frame(true);
        require(rules(layer_pinned.run) == (ui_detection::rules::pin_only_ui | ui_detection::rules::darkening_measured) &&
            masked(layer_pinned.mask, true), "D3D12 UIPinOnlyUI=1 did not unpin exactly the HDR layer reference's darkening pixels");
      }
      require(layer_pinned.sample.source_kind == ui_detection::source_layer && layer_pinned.sample.darkening.measured &&
          layer_pinned.sample.darkening.applied && layer_pinned.sample.darkening.unpinned == reference.n.unpinned &&
          layer_pinned.sample.darkening.kept == reference.n.kept,
        "The D3D12 applied HDR layer sample's words 62-63 differ from the CPU reference");
      require(masked(frame(false).mask, false), "Clearing the switch did not restore the D3D12 HDR layer's raw alpha mask");
      policy.set_pin_only_ui(false);
    }
    report << "pin-only-ui-hdr-layer D3D12 band=" << hdr_band << " strokes=" << hdr_strokes << " unpinned=" << hdr_unpinned
           << " kept=" << hdr_kept << " shadow_raw_alpha=1 words_equal_reference=1 switch_on_band_unpinned=1"
              " strokes_hdr_icon_faint_panel_pinned=1 linear_tolerance=1\n";
    std::printf("PASS D3D12 pin only UI on an HDR-like float layer (fix 4, opacity path): the dim band (%llu px) is measured in the shadow and unpinned with UIPinOnlyUI=1; sharp black strokes (%llu px), a 4x HDR white icon and a faint linear-colour panel keep their pins; words 62-63, counters and masks equal the CPU reference\n",
      static_cast<unsigned long long>(hdr_band), static_cast<unsigned long long>(hdr_strokes));

    queue->wait_idle();
    for (const auto view : views) device->destroy_resource_view(view);
    present(original);
    require(fixture.read(backbuffer, D3D12_RESOURCE_STATE_PRESENT) == original, "The D3D12 pre-UI section did not restore the source");
    std::vector<float> restored_depth(original_depth.size() / sizeof(float));
    std::memcpy(restored_depth.data(), original_depth.data(), original_depth.size());
    fixture.upload_depth(restored_depth);
    if (validation.p) {
      unsigned errors = 0;
      for (UINT64 i = 0, count = validation->GetNumStoredMessages(); i != count; ++i) {
        SIZE_T length = 0;
        if (FAILED(validation->GetMessage(i, nullptr, &length)) || !length) continue;
        std::vector<std::uint8_t> storage(length);
        auto *message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
        if (FAILED(validation->GetMessage(i, message, &length))) continue;
        if (message->Severity != D3D12_MESSAGE_SEVERITY_CORRUPTION && message->Severity != D3D12_MESSAGE_SEVERITY_ERROR) continue;
        if (++errors <= 10) std::fprintf(stderr, "D3D12 validation: %.*s\n", int(message->DescriptionByteLength), message->pDescription);
      }
      require(!errors, "The D3D12 debug layer reported " + std::to_string(errors) + " validation errors in the pre-UI section");
      std::puts("PASS D3D12 debug layer: no validation error or corruption message in the pre-UI change-set and pin-only-UI section");
    }
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
      require(sunshine_game3d::test::render_frame(renderer, owner_queue->get_immediate_command_list(), source,
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
    {
      // Completed parity frames report GPU stage times without a CPU wait.
      sunshine_game3d::gpu_timing timing;
      using stage = sunshine_game3d::gpu_timing;
      const bool took = renderer.take_gpu_timing(timing);
      // The packed-eye pass renders both eyes into the side-by-side target, so
      // its time is reported as eyes and no separate pack stage remains.
      require(took && timing.frames > 0 && timing.mean_ms[stage::eyes] > 0 &&
          timing.mean_ms[stage::pack] == 0 && timing.mean_ms[stage::total] >= timing.mean_ms[stage::eyes] &&
          timing.max_ms[stage::total] >= timing.mean_ms[stage::total],
        "Native renderer did not report completed GPU stage timing");
      std::printf("PASS native GPU timing: frames=%u total=%.3f ms eyes=%.3f ms pack=%.3f ms\n", timing.frames,
        timing.mean_ms[stage::total], timing.mean_ms[stage::eyes], timing.mean_ms[stage::pack]);
      require(!renderer.take_gpu_timing(timing) && timing.frames == 0, "GPU timing window did not reset");
    }
    {
      // A Present that never reaches finish_present must not wedge native
      // rendering; its work is retired by the next signal, not assumed complete.
      const auto &test = tests.front();
      prepare_depth(fixture, test);
      auto *backbuffer = fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
      write_native_source(fixture, backbuffer);
      const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
      const auto color = static_cast<api::color_space>(fixture.color);
      const auto depth = test.parameters.depth_ready ? fixture.depth_view : api::resource_view{};
      auto *commands = owner_queue->get_immediate_command_list();
      require(renderer.configure(observed.runtime, source, color) &&
        sunshine_game3d::test::render_frame(renderer, commands, source, depth, test.parameters),
        "Missed-finish fixture rejected its first frame");
      require(!sunshine_game3d::test::render_frame(renderer, commands, source, depth, test.parameters),
        "Renderer recorded twice in one presentation");
      renderer.begin_present(); // The preceding finish_present never arrived.
      require(sunshine_game3d::test::render_frame(renderer, commands, source, depth, test.parameters),
        "Renderer stayed wedged after a missed finish_present");
      const std::string alternate(sunshine_game3d::renderer::shader_source());
      require(!renderer.configure(observed.runtime, source, color, alternate),
        "Renderer replaced resources before the missed presentation was signaled");
      owner_queue->flush_immediate_command_list();
      renderer.finish_present();
      owner_queue->wait_idle();
      require(renderer.configure(observed.runtime, source, color, alternate) &&
        renderer.configure(observed.runtime, source, color),
        "Renderer stayed busy after the covering signal completed");
      require(fixture.read(backbuffer, D3D12_RESOURCE_STATE_PRESENT) == fixture.source_bytes,
        "Missed-finish recovery changed the game's mono backbuffer");
      std::puts("PASS missed finish_present: next presentation renders, resource replacement waits for the covering signal");
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
      require(sunshine_game3d::test::render_frame(renderer, owner_queue->get_immediate_command_list(), source, fixture.depth_view, parameters),
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
    // Compare two documented encodings of the same inverse-distance field
    // through the actual production D3D12 compute pipeline. Powers of two
    // make the independent storage/unit transform exact in FP32.
    {
      auto parameters=tests.front().parameters;
      parameters.coordinate_basis=0; parameters.projection={0.f,1.f};
      parameters.depth_scale=4.f; parameters.convergence={.05f,.25f};
      parameters.strength=100.f; parameters.strength_blend=1.f;
      parameters.depth_rect={.125f,.125f,.75f,.75f};
      std::vector<float> inverse(size_t(width)*height);
      for (unsigned y=0; y<height; ++y) for (unsigned x=0; x<width; ++x)
        inverse[size_t(y)*width+x]=std::ldexp(.125f,int((x/17+y/13)%4));
      const auto render_candidate=[&](const std::vector<float> &raw, bool capture = false) {
        fixture.upload_depth(raw);
        auto *backbuffer=fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
        write_native_source(fixture,backbuffer);
        const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
        require(sunshine_game3d::test::render_frame(renderer, owner_queue->get_immediate_command_list(),source,fixture.depth_view,parameters),
          "Linear-equivalence D3D12 render failed");
        if (capture) dump.begin(observed.runtime, renderer, parameters, fixture.depth_view, false,
          static_cast<api::color_space>(fixture.color));
        owner_queue->flush_immediate_command_list(); renderer.finish_present();
        if (capture) dump.submitted(observed.runtime);
        owner_queue->wait_idle();
        if (capture) dump.verify(observed.runtime, [&](api::resource texture, bool common) {
          return fixture.read(reinterpret_cast<ID3D12Resource *>(texture.handle),
            common ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        }, directory / "dump-linear-distance");
        return fixture.read(reinterpret_cast<ID3D12Resource *>(renderer.diagnostics().candidate.handle));
      };
      const auto reference=render_candidate(inverse);
      for (unsigned scenario=0; scenario!=4; ++scenario) {
        const float units=scenario==1 ? 1024.f : 1.f;
        const float A=scenario==2 ? 2.f : 0.f, scale=scenario==2 ? -2.f : 1.f;
        parameters.coordinate_basis=2; parameters.projection={A,scale};
        parameters.depth_scale=4.f*units; parameters.convergence={.05f,.25f/units};
        std::vector<float> raw(inverse.size());
        for (size_t i=0; i<raw.size(); ++i) raw[i]=A+(units/inverse[i])/scale;
        if (scenario==3) {
          // Whole source invalid: no reciprocal, overflow or stale candidate
          // may move pixels, including after three successful linear frames.
          for (size_t i=0; i<raw.size(); ++i)
            raw[i]=(i%4)==0 ? 0.f : (i%4)==1 ? -1.f : (i%4)==2 ? NAN : INFINITY;
        }
        const auto actual=render_candidate(raw, scenario==0);
        require(actual.size()==reference.size(),"Linear field shape changed");
        for (size_t offset=0; offset<actual.size(); offset+=sizeof(float)) {
          float got{}, wanted{};
          std::memcpy(&got,actual.data()+offset,sizeof(got));
          if (scenario!=3) std::memcpy(&wanted,reference.data()+offset,sizeof(wanted));
          require(std::isfinite(got) && std::abs(got-wanted)<=2e-8f,
            "Linear distance changed equivalent geometry or invalid pixels acquired disparity");
        }
      }
      report << "linear-depth D3D12 reciprocal=1 device_equivalence=1 units=1 negative_storage_scale=1 crop=1 all_invalid_neutral=1\n";
      std::puts("PASS linear-distance native D3D12: device-equivalent geometry, units/storage, crop and invalid pixels");
      std::vector<float> restored(size_t(width)*height);
      std::memcpy(restored.data(),original_depth.data(),original_depth.size());
      fixture.upload_depth(restored);
    }
    // An enabled source-alpha frame must retain its separate b1 selection in
    // the real cross-API dump, not merely in the offline package parser.
    {
      auto parameters = tests.front().parameters;
      auto *backbuffer = fixture.backbuffers[fixture.swapchain->GetCurrentBackBufferIndex()].p;
      write_native_source(fixture, backbuffer);
      const api::resource source{reinterpret_cast<std::uint64_t>(backbuffer)};
      require(sunshine_game3d::test::render_frame(renderer, owner_queue->get_immediate_command_list(), source, fixture.depth_view, parameters, true),
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
      bool linear = false;
    };
    const std::array<nearest_case, 4> nearest_tests {{
      {"normal", .75f, .25f, 100.f, 1.f, false},
      {"reversed", .625f, .25f, 100.f, 1.f, true},
      {"floor-partial-strength", .5f, .875f, 50.f, .5f, false},
      {"linear-distance", .5f, .25f, 100.f, 1.f, false, true},
    }};
    for (const auto &test : nearest_tests) {
      std::vector<float> raw(size_t(width) * height, .125f);
      raw[size_t(height / 2) * width + width / 2] = .375f;
      raw.back() = test.maximum;
      if (test.reverse) for (auto &value : raw) value = 1.f - value;
      if (test.linear) {
        for (auto &value : raw) value = 1.f / value;
        raw.front() = NAN; // Invalid sky must not poison the covered maximum.
      }
      fixture.upload_depth(raw);
      const auto frozen_depth = fixture.read(fixture.depth.p);
      auto parameters = tests.front().parameters;
      parameters.coordinate_basis = test.linear ? 2 : 0;
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
      require(sunshine_game3d::test::render_frame(renderer, owner_queue->get_immediate_command_list(), source, fixture.depth_view,
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
        require(replay.at("ui_parameter_abi") == "sunshine_game3d.ui_parameters.v7" &&
            replay.at("ui_constant_binding").at("mode") == 2 &&
            replay.at("ui_constant_binding").at("inverse_depth") == test.floor &&
            replay.at("ui_plane_resolution").at("reduction_ran") == true &&
            replay.at("ui_plane_resolution").at("resolved_value").is_null(),
          "Nearest-UI D3D12 dump confused submitted floor with asynchronous GPU result");
        bool tile_pass = false, reduce_pass = false, field_pass = false, pin_pass = false;
        for (const auto &pass : replay.at("passes")) {
          if (pass.at("entry") == "SunshineUINearestTilesCS") tile_pass = pass.at("enabled") == true &&
            pass.at("uavs").at("u4") == "ui_plane_tiles:R32_FLOAT";
          if (pass.at("entry") == "SunshineUINearestReduceCS") reduce_pass = pass.at("enabled") == true &&
            pass.at("srvs").at("t8") == "ui_plane_tiles" && pass.at("uavs").at("u5") == "ui_plane_resolved:R32_FLOAT";
          // The scene limiter reads no UI input; the separate pin pass reads the resolved plane.
          if (pass.at("entry") == "SunshineHostHorizontalCS") field_pass = !pass.at("srvs").contains("t9");
          if (pass.at("entry") == "SunshineApplyUICS") pin_pass = pass.at("enabled") == true &&
            pass.at("srvs").at("t9") == "ui_plane_resolved";
        }
        require(tile_pass && reduce_pass && field_pass && pin_pass, "Nearest-UI D3D12 dump lost actual u4/u5/t8/t9 bindings");
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
      bool calibration_valid = true;
    };
    const std::array<front_case, 15> front_tests {{
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
      {"float-order", .875f, 128.f, .75f, 3.f, .7f, .02f},
      {"zero-gain", .875f, 0.f, .75f, 100.f, 1.f, .01f, true, true, false},
      {"invalid-zero", .875f, 128.f, std::numeric_limits<float>::quiet_NaN(), 100.f, 1.f, .01f, true, true, false},
    }};
    const sunshine_game3d::ui_plane_parameters front_plane{sunshine_game3d::ui_plane_mode::front_limit, 0.f};
    std::vector<std::uint8_t> first_front_field, first_shallow_field, first_scene_candidate;
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
        require(sunshine_game3d::test::render_frame(renderer, owner_queue->get_immediate_command_list(), source, fixture.depth_view,
          parameters, protect, {}, plane), "Front-limit D3D12 render failed");
        const bool shallow = plane.mode == sunshine_game3d::ui_plane_mode::shallow_front;
        const bool dump_case = protect && case_index == 0 &&
          (plane.mode == sunshine_game3d::ui_plane_mode::front_limit || shallow);
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
          const auto destination = directory / (shallow ? "dump-source-alpha-shallow-ui" : "dump-source-alpha-front-ui");
          dump.verify(observed.runtime, [&](api::resource texture, bool common) {
            return fixture.read(reinterpret_cast<ID3D12Resource *>(texture.handle),
              common ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
          }, destination);
          std::ifstream input(destination / "manifest.json");
          require(input.good(), "Front-limit D3D12 dump manifest missing");
          const auto manifest = nlohmann::json::parse(input);
          const auto &replay = manifest.at("producer_metadata").at("replay");
          require(replay.at("ui_parameter_abi") == "sunshine_game3d.ui_parameters.v7" &&
              replay.at("ui_constant_binding").at("mode") == (shallow ? 4 : 3) &&
              replay.at("ui_constant_binding").at("mode_name") == (shallow ? "shallow_front" : "front_limit") &&
              replay.at("ui_constant_binding").at("inverse_depth") == 0.f &&
              replay.at("ui_plane_resolution").at("policy") == (shallow ? "fixed_shallow_front" : "fixed_current_front_limit") &&
              replay.at("ui_plane_resolution").at("inverse_depth_role") == "unused" &&
              replay.at("ui_plane_resolution").at("reduction_ran") == false,
            "Front-limit D3D12 dump did not preserve its stateless display-cap policy");
          if (shallow) require(replay.at("ui_plane_resolution").at("front_limit_fraction") == .25f,
            "Shallow UI dump lost its quarter-cap placement policy");
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
      const auto unprotected_field = read(unprotected.final_field), unprotected_sbs = read(unprotected.sbs);
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
      const double expected = test.depth_ready && test.camera_ready && test.calibration_valid ?
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
      // The actual mode-3 GPU field is the authoritative cap oracle. Scaling
      // that readback by an exact binary quarter checks mode 4 without a CPU
      // copy of the scene or warp algorithms, including the float-order case.
      const float unused_depth = case_index == 1 ? std::numeric_limits<float>::quiet_NaN() :
        case_index == 2 ? std::numeric_limits<float>::infinity() : case_index == 3 ? -1.f : 0.f;
      const sunshine_game3d::ui_plane_parameters shallow_plane{sunshine_game3d::ui_plane_mode::shallow_front, unused_depth};
      const auto shallow_disabled = render(false, shallow_plane);
      require(read(shallow_disabled.final_field) == unprotected_field && read(shallow_disabled.sbs) == unprotected_sbs,
        "Disabled shallow UI metadata changed ordinary scene rendering");
      const auto shallow_resources = render(true, shallow_plane);
      require(sunshine_game3d::ui_parameter_words(true, renderer.consumed_ui_plane()) ==
          sunshine_game3d::ui_parameter_words(true, shallow_plane), "Shallow UI changed submitted mode/unused-depth bits");
      require(read(shallow_resources.candidate) == candidate && read(shallow_resources.vertical_field) == vertical,
        "Shallow UI changed scene candidate or vertical geometry");
      const auto shallow_field = read(shallow_resources.final_field);
      require(shallow_field.size() == field.size(), "Shallow UI field lost full-resolution R32 storage");
      const float quarter_cap = .25f * first;
      for (size_t offset = 0; offset != shallow_field.size(); offset += sizeof(float)) {
        float value;
        std::memcpy(&value, shallow_field.data() + offset, sizeof(value));
        require(std::isfinite(value) && value == quarter_cap,
          "Shallow UI is not exactly one quarter of the authoritative current GPU cap");
      }
      if (case_index == 0) first_shallow_field = shallow_field;
      else if (case_index < 4) require(shallow_field == first_shallow_field,
        "Shallow UI moved with depth/gain/zero or interpreted its unused inverse-depth word");
      require(fixture.read(fixture.depth.p) == frozen_depth, "Front-limit UI changed source depth");
      require(fixture.read(backbuffer, D3D12_RESOURCE_STATE_PRESENT) == fixture.source_bytes,
        "Front-limit UI changed the game's source color");
      report << "front-ui " << test.name << " rigid_parallax_uv=" << first << " expected_uv=" << expected << '\n';
      std::printf("MEASURE front-ui D3D12 %s rigid_parallax_uv=%.9g expected_uv=%.9g\n", test.name, first, expected);
      report << "shallow-ui " << test.name << " rigid_parallax_uv=" << quarter_cap << " cap_uv=" << first << '\n';
      std::printf("MEASURE shallow-ui D3D12 %s rigid_parallax_uv=%.9g cap_uv=%.9g\n", test.name, quarter_cap, first);
    }
    require(scene_changed, "Front-limit invariance fixture did not actually change the scene geometry");
    std::puts("PASS fixed-front D3D12 UI remains invariant across depth/gain/zero; current strength/blend/budget/mono and unchanged scene/input fields verified");
    std::puts("PASS shallow-front D3D12 UI is exactly one quarter of the current GPU front cap; unchanged scene/off path, depth/gain/zero stability, unused malformed depth and admission verified");
    std::vector<float> restored_depth(size_t(width) * height);
    require(original_depth.size() == restored_depth.size() * sizeof(float), "Original depth readback size changed");
    std::memcpy(restored_depth.data(), original_depth.data(), original_depth.size());
    fixture.upload_depth(restored_depth);
    check_alpha_auto_d3d12(fixture, renderer, report);
    check_automatic_hudless_d3d12(fixture, renderer, report);
    check_adaptive_ui_d3d12(fixture, renderer, dump, report, directory);
    check_typed_ui_masks_d3d12(fixture, renderer, dump, report, directory);
    check_ui_pin_band_d3d12(fixture, renderer, report);
    check_pre_ui_change_set_d3d12(fixture, renderer, dump, report, directory);
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
    // The D3D12 debug layer when this machine has it (d3d12SDKLayers.dll, the
    // Graphics Tools optional feature), enabled before any device exists;
    // SUNSHINE_D3D12_DEBUG_LAYER=0 opts out. The pre-UI change-set section
    // fails on the validation errors it records.
    if (const char *layer = std::getenv("SUNSHINE_D3D12_DEBUG_LAYER"); !layer || std::strcmp(layer, "0")) {
      com_ptr<ID3D12Debug> debug;
      const HRESULT result = D3D12GetDebugInterface(IID_PPV_ARGS(debug.put()));
      if (SUCCEEDED(result)) { debug->EnableDebugLayer(); d3d12_debug_layer = true; }
      std::printf("MEASURE D3D12 debug layer %s (0x%08lx)\n", d3d12_debug_layer ? "enabled" :
        "unavailable: d3d12SDKLayers is not installed", static_cast<unsigned long>(result));
    }
    fixture_t fixture;
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), directory, color, 0);
    check_native_parity(fixture, directory);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
