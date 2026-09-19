// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// Shared actual-runtime observations only. Calibration expectations and scenario
// entrypoints belong to each fixture, so current adaptive tests never compile
// or inherit the historical fixed-reference test program.
#define SUNSHINE_RAW_SCENE_RUNTIME
#include "test_depth_selection_runtime.cpp"
#include <map>

namespace {
  struct raw_runtime_fixture : selection_fixture_t {
    bool normal = false;
    fs::path runtime_directory;
    std::vector<std::uint8_t> selected_native_depth() {
      ID3D12Resource *selected = nullptr;
      observed.runtime->enumerate_texture_variables(effect_file, [&](api::effect_runtime *runtime, api::effect_texture_variable variable) {
        char name[256] {};
        runtime->get_texture_variable_name(variable, name);
        if (!named(name, "DepthBufferTex")) return;
        api::resource_view view {};
        runtime->get_texture_binding(variable, &view, nullptr);
        require(view.handle, "Reference oracle lost the real selected DEPTH binding");
        selected = reinterpret_cast<ID3D12Resource *>(runtime->get_device()->get_resource_from_view(view).handle);
      });
      require(selected, "Reference oracle cannot read selected native depth");
      const auto desc = selected->GetDesc();
      require(desc.Width == scene->width && desc.Height == scene->height &&
        (desc.Format == DXGI_FORMAT_R32_FLOAT || desc.Format == DXGI_FORMAT_R32_TYPELESS),
        "Reference oracle selected unexpected geometry/format or the decoy");
      return read(selected);
    }
    float scalar(const char *name) {
      float value = 0;
      observed.runtime->get_uniform_value_float(uniform(name), &value, 1);
      return value;
    }
    bool ready() {
      if (!observed.runtime || !observed.renders) return false;
      const auto handle = uniform("Sunshine_CameraDepthReady", false);
      bool value = false;
      if (handle.handle) observed.runtime->get_uniform_value_bool(handle, &value, 1);
      return value;
    }
    std::array<float, 2> zero() {
      std::array<float, 2> value {};
      observed.runtime->get_uniform_value_float(uniform("Sunshine_CameraConvergence"), value.data(), 2);
      return value;
    }
    void frames_for(unsigned milliseconds) {
      const auto until = GetTickCount64() + milliseconds;
      do { step(); } while (GetTickCount64() < until);
    }
    std::vector<std::uint8_t> check_current_mono() {
      const auto pixels = read(exported.p);
      std::map<std::uint32_t, sunshine_depth3d_color::rgb> pq_cache;
      double maximum = 0;
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const size_t i = size_t(y) * width + x;
        sunshine_depth3d_color::rgb expected {};
        if (color == 2) {
          for (unsigned c = 0; c < 3; ++c) {
            std::uint16_t half = 0;
            std::memcpy(&half, source_bytes.data() + i * 8 + c * 2, 2);
            expected[c] = half_float(half);
          }
        } else {
          std::uint32_t packed = 0;
          std::memcpy(&packed, source_bytes.data() + i * 4, 4);
          auto found = pq_cache.find(packed);
          if (found == pq_cache.end()) {
            const sunshine_depth3d_color::rgb code {double(packed & 1023) / 1023., double((packed >> 10) & 1023) / 1023., double((packed >> 20) & 1023) / 1023.};
            found = pq_cache.emplace(packed, sunshine_depth3d_color::decoded(code, color)).first;
          }
          expected = found->second;
        }
        for (unsigned c = 0; c < 3; ++c) {
          const float left = channel(pixels, x, y, c), right = channel(pixels, width + x, y, c);
          require(std::isfinite(left) && std::isfinite(right), "Automatic mono fallback contains non-finite color");
          const double scale = std::max(1., std::abs(expected[c]));
          maximum = std::max({maximum, std::abs(left - expected[c]) / scale, std::abs(right - expected[c]) / scale});
          require(std::abs(left - right) <= .002 * scale, "Automatic fallback does not repeat the same source image in both eyes");
        }
      }
      std::printf("MEASURE current-source mono fallback relative color error=%.9g\n", maximum);
      require(maximum < .003, "Automatic fallback differs from actual current source color beyond HDR export precision");
      return pixels;
    }
  };
}
