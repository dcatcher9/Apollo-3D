/**
 * @file tests/unit/test_host_sbs_subtitle_contour_distance_gpu.cpp
 * @brief Native five-center Euclidean scans and shared subtitle smoothing guards.
 */
#include <gtest/gtest.h>

#ifdef _WIN32

  #include "src/depth_coordinate_v2.h"
  #include "src/generated/depth_coordinate_v2_contract.h"
  #include "src/host_sbs_shader_cache.h"

  #include <algorithm>
  #include <array>
  #include <cmath>
  #include <cstdint>
  #include <cstring>
  #include <d3d11.h>
  #include <limits>
  #include <set>
  #include <string>
  #include <utility>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace v2 = models::depth_coordinate_v2;
  using rect_t = std::array<std::uint32_t, 4u>;
  using point_t = std::pair<int, int>;
  constexpr int center_offset = static_cast<int>(v2::subtitle_contour_gaussian_center_offset_source_pixels);
  constexpr std::uint32_t limiter_lines = 8u;

  struct distance_constants_t {
    rect_t source;  // W, H, crop top, crop height.
    rect_t field;  // W, H, reserved, reserved.
    rect_t content;
  };

  static_assert(sizeof(distance_constants_t) == 48u);

  struct distance_case_t {
    std::uint32_t source_w, source_h, field_w, field_h;
    rect_t content;
    std::vector<point_t> seeds;
  };

  distance_constants_t constants(const distance_case_t &test) {
    const auto crop_h = std::min(test.source_h, static_cast<std::uint32_t>(v2::subtitle_ocr_ceil_div(static_cast<std::uint64_t>(test.source_w) * v2::subtitle_ocr_crop_aspect_height, v2::subtitle_ocr_crop_aspect_width)));
    return {{test.source_w, test.source_h, test.source_h - crop_h, crop_h}, {test.field_w, test.field_h, 0u, 0u}, test.content};
  }

  bool inside(const rect_t &content, std::uint32_t x, std::uint32_t y) {
    return x >= content[0] && y >= content[1] && x < content[2] && y < content[3];
  }

  std::uint32_t bits(float value) {
    std::uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    return word;
  }

  float from_bits(std::uint32_t word) {
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
  }

  struct texture_t {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
  };

  bool create_texture(ID3D11Device *device, std::uint32_t width, std::uint32_t height, DXGI_FORMAT format, UINT binds, const void *initial, texture_t &out) {
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1u;
    desc.Format = format;
    desc.Usage = initial ? D3D11_USAGE_IMMUTABLE : D3D11_USAGE_DEFAULT;
    desc.BindFlags = binds;
    D3D11_SUBRESOURCE_DATA data {};
    data.pSysMem = initial;
    data.SysMemPitch = width * sizeof(std::uint32_t);
    if (FAILED(device->CreateTexture2D(&desc, initial ? &data : nullptr, &out.texture))) {
      return false;
    }
    if ((binds & D3D11_BIND_SHADER_RESOURCE) != 0u && FAILED(device->CreateShaderResourceView(out.texture.Get(), nullptr, &out.srv))) {
      return false;
    }
    return (binds & D3D11_BIND_UNORDERED_ACCESS) == 0u ||
           SUCCEEDED(device->CreateUnorderedAccessView(out.texture.Get(), nullptr, &out.uav));
  }

  template<class T>
  bool read_texture(ID3D11Device *device, ID3D11DeviceContext *context, ID3D11Texture2D *source, std::vector<T> &out) {
    D3D11_TEXTURE2D_DESC desc {};
    source->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0u;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging))) {
      return false;
    }
    context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
      return false;
    }
    out.resize(static_cast<std::size_t>(desc.Width) * desc.Height);
    for (std::uint32_t y = 0u; y < desc.Height; ++y) {
      std::memcpy(out.data() + static_cast<std::size_t>(y) * desc.Width, static_cast<const std::uint8_t *>(mapped.pData) + static_cast<std::size_t>(y) * mapped.RowPitch, static_cast<std::size_t>(desc.Width) * sizeof(T));
    }
    context->Unmap(staging.Get(), 0u);
    return true;
  }

  bool create_buffer(ID3D11Device *device, const void *data, std::uint32_t bytes, UINT bind, ComPtr<ID3D11Buffer> &out, UINT misc = 0u, std::uint32_t stride = 0u) {
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = bytes;
    desc.Usage = (misc & D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS) != 0u ? D3D11_USAGE_DEFAULT : D3D11_USAGE_IMMUTABLE;
    desc.BindFlags = bind;
    desc.MiscFlags = misc;
    desc.StructureByteStride = stride;
    D3D11_SUBRESOURCE_DATA initial {};
    initial.pSysMem = data;
    const auto result = device->CreateBuffer(&desc, &initial, &out);
    if (FAILED(result)) {
      ADD_FAILURE() << "CreateBuffer HRESULT=" << static_cast<unsigned long>(result) << " bytes=" << bytes << " bind=" << bind << " misc=" << misc << " stride=" << stride;
      return false;
    }
    return true;
  }

  class HostSbsSubtitleContourDistanceGpuTest: public ::testing::Test {
  protected:
    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL levels[] {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_WARP,
        nullptr,
        0u,
        levels,
        1u,
        D3D11_SDK_VERSION,
        &device_,
        &actual,
        &context_
      )));
      ASSERT_EQ(actual, D3D_FEATURE_LEVEL_11_0);
      ASSERT_TRUE(compile("host_sbs_subtitle_contour_distance_cs.hlsl", "horizontal_main", horizontal_shader_));
      ASSERT_TRUE(compile("host_sbs_subtitle_contour_distance_cs.hlsl", "vertical_main", vertical_shader_));
    }

    bool compile(const char *file, const char *entry, ComPtr<ID3D11ComputeShader> &out) {
      namespace cache = models::host_sbs_shader_cache;
      const std::array specs {cache::shader_spec {file, entry, "cs_5_0"}};
      const auto sources = cache::snapshot_sources(SUNSHINE_SHADERS_DIR, specs);
      const auto code = cache::get(sources, specs[0]);
      return code && !code->empty() && SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &out));
    }

    void unbind() {
      std::array<ID3D11ShaderResourceView *, 4u> null_srvs {};
      std::array<ID3D11UnorderedAccessView *, 2u> null_uavs {};
      context_->CSSetShaderResources(0u, null_srvs.size(), null_srvs.data());
      context_->CSSetUnorderedAccessViews(0u, null_uavs.size(), null_uavs.data(), nullptr);
    }

    bool run(const distance_case_t &test, const distance_constants_t *override_constants = nullptr, bool zero_dispatch = false, bool wrong_input_width = false) {
      const auto actual = constants(test);
      const auto data = override_constants ? *override_constants : actual;
      const auto input_w = test.source_w + (wrong_input_width ? 1u : 0u);
      std::vector<std::uint32_t> mask(static_cast<std::size_t>(input_w) * actual.source[3], 0u);
      for (const auto &[x, y] : test.seeds) {
        if (x < 0 || static_cast<std::uint32_t>(x) >= input_w || y < 0 || static_cast<std::uint32_t>(y) >= actual.source[3]) {
          return false;
        }
        mask[static_cast<std::size_t>(y) * input_w + x] = 1u;
      }
      texture_t seeds, scratch, q;
      if (!create_texture(device_.Get(), input_w, actual.source[3], DXGI_FORMAT_R32_UINT, D3D11_BIND_SHADER_RESOURCE, mask.data(), seeds) || !create_texture(device_.Get(), test.field_w, actual.source[3] + 4u, DXGI_FORMAT_R32_FLOAT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, nullptr, scratch) || !create_texture(device_.Get(), test.field_w, test.field_h, DXGI_FORMAT_R32_FLOAT, D3D11_BIND_UNORDERED_ACCESS, nullptr, q)) {
        return false;
      }
      // No legal input UAV exists. The distinct immutable support, scratch, and final resources
      // exercise actual SRV/UAV transitions rather than relying on an in-place scan.
      EXPECT_NE(seeds.texture.Get(), scratch.texture.Get());
      EXPECT_NE(seeds.texture.Get(), q.texture.Get());
      EXPECT_NE(scratch.texture.Get(), q.texture.Get());
      D3D11_TEXTURE2D_DESC seed_desc {};
      seeds.texture->GetDesc(&seed_desc);
      EXPECT_EQ(seed_desc.Usage, D3D11_USAGE_IMMUTABLE);
      EXPECT_EQ(seed_desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS, 0u);
      const std::vector<std::uint32_t> scratch_poison(static_cast<std::size_t>(test.field_w) * (actual.source[3] + 4u), 0x7fc12345u);
      const std::vector<std::uint32_t> q_poison(static_cast<std::size_t>(test.field_w) * test.field_h, 0x7fc12345u);
      context_->UpdateSubresource(scratch.texture.Get(), 0u, nullptr, scratch_poison.data(), test.field_w * sizeof(float), 0u);
      context_->UpdateSubresource(q.texture.Get(), 0u, nullptr, q_poison.data(), test.field_w * sizeof(float), 0u);
      ComPtr<ID3D11Buffer> cb;
      if (!create_buffer(device_.Get(), &data, sizeof(data), D3D11_BIND_CONSTANT_BUFFER, cb)) {
        return false;
      }
      context_->CSSetConstantBuffers(0u, 1u, cb.GetAddressOf());
      const auto grid_x = (test.field_w + 15u) / 16u;
      // Production calibrated grids have enough groups for all source rows and field columns.
      // Tiny analytic domains deliberately add rows of the same linearized grid so every output
      // is tested; they do not claim membership in the calibrated production shape family.
      const auto groups = std::max(actual.source[3] + 4u, test.field_w);
      const auto grid_y = std::max((test.field_h + 15u) / 16u, (groups + grid_x - 1u) / grid_x);
      ComPtr<ID3D11Buffer> zero_args;
      const std::array<std::uint32_t, 3u> zeros {};
      if (zero_dispatch && !create_buffer(device_.Get(), zeros.data(), sizeof(zeros), 0u, zero_args, D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS)) {
        return false;
      }
      const auto dispatch = [&]() {
        if (zero_dispatch) {
          context_->DispatchIndirect(zero_args.Get(), 0u);
        } else {
          context_->Dispatch(grid_x, grid_y, 1u);
        }
      };
      context_->CSSetShader(horizontal_shader_.Get(), nullptr, 0u);
      context_->CSSetShaderResources(0u, 1u, seeds.srv.GetAddressOf());
      context_->CSSetUnorderedAccessViews(0u, 1u, scratch.uav.GetAddressOf(), nullptr);
      dispatch();
      unbind();
      context_->CSSetShader(vertical_shader_.Get(), nullptr, 0u);
      context_->CSSetShaderResources(1u, 1u, scratch.srv.GetAddressOf());
      context_->CSSetUnorderedAccessViews(0u, 1u, q.uav.GetAddressOf(), nullptr);
      dispatch();
      unbind();
      std::vector<std::uint32_t> after;
      if (!read_texture(device_.Get(), context_.Get(), scratch.texture.Get(), horizontal_) || !read_texture(device_.Get(), context_.Get(), q.texture.Get(), final_) || !read_texture(device_.Get(), context_.Get(), seeds.texture.Get(), after)) {
        return false;
      }
      EXPECT_EQ(after, mask);
      return after == mask;
    }

    void expect_oracle(const distance_case_t &test) {
      const auto data = constants(test);
      ASSERT_EQ(horizontal_.size(), static_cast<std::size_t>(test.field_w) * (data.source[3] + 4u));
      ASSERT_EQ(final_.size(), static_cast<std::size_t>(test.field_w) * test.field_h);
      std::set<point_t> union_set;
      for (auto seed : test.seeds) {
        for (const auto &[dx, dy] : std::array<point_t, 5u> {{{0, 0}, {center_offset, 0}, {-center_offset, 0}, {0, center_offset}, {0, -center_offset}}}) {
          union_set.emplace(seed.first + dx, seed.second + dy);
        }
      }
      const std::vector<point_t> centers(union_set.begin(), union_set.end());
      constexpr double epsilon = std::numeric_limits<float>::epsilon();
      const double x_error = 4.0 * epsilon * test.source_w;
      const double y_error = 6.0 * epsilon * test.source_h;
      const double position_error = std::hypot(x_error, y_error);

      struct error_stats_t {
        std::size_t failures = 0u;
        double maximum_absolute = 0.0, maximum_relative = 0.0, maximum_bound_ratio = 0.0;
        std::string first_failure;
      } horizontal_error, q_error;

      const auto check = [](error_stats_t &stats, float value, double expected, double error_bound, std::uint32_t x, std::uint32_t y) {
        bool passed;
        if (std::isinf(expected)) {
          passed = std::isinf(value) && value > 0.0f;
        } else {
          // Real-coordinate rounding and final FP32 cost arithmetic, not bit-identical hulls.
          const double allowed = 2.0 * std::sqrt(expected) * error_bound + error_bound * error_bound +
                                 4.0 * std::numeric_limits<float>::epsilon() * std::max(expected, 1.0);
          const double error = std::abs(static_cast<double>(value) - expected);
          stats.maximum_absolute = std::max(stats.maximum_absolute, error);
          if (expected > 0.0) {
            stats.maximum_relative = std::max(stats.maximum_relative, error / expected);
          }
          stats.maximum_bound_ratio = std::max(stats.maximum_bound_ratio, error / allowed);
          passed = std::isfinite(value) && value >= 0.0f && error <= allowed;
        }
        if (!passed) {
          ++stats.failures;
          if (stats.first_failure.empty()) {
            stats.first_failure = std::to_string(x) + "," +
                                  std::to_string(y) + " actual=" + std::to_string(value) + " expected=" + std::to_string(expected);
          }
        }
      };
      // Exhaustive combinatorial nearest-center oracle, independent of prefix chunk carries or
      // parabola hull construction. It computes distance only, never a renderer/classifier.
      for (std::uint32_t row = 0u; row < data.source[3] + 4u; ++row) {
        for (std::uint32_t x = 0u; x < test.field_w; ++x) {
          double expected = std::numeric_limits<double>::infinity();
          if (x >= test.content[0] && x < test.content[2]) {
            const double sx = (static_cast<double>(x - test.content[0]) + 0.5) * test.source_w /
                                (test.content[2] - test.content[0]) -
                              0.5;
            for (const auto &[cx, cy] : centers) {
              if (cy + center_offset == static_cast<int>(row)) {
                expected = std::min(expected, (sx - cx) * (sx - cx));
              }
            }
          }
          check(horizontal_error, horizontal_[static_cast<std::size_t>(row) * test.field_w + x], expected, x_error, x, row);
        }
      }
      for (std::uint32_t y = 0u; y < test.field_h; ++y) {
        for (std::uint32_t x = 0u; x < test.field_w; ++x) {
          double expected = std::numeric_limits<double>::infinity();
          if (inside(test.content, x, y)) {
            const double sx = (static_cast<double>(x - test.content[0]) + 0.5) * test.source_w /
                                (test.content[2] - test.content[0]) -
                              0.5;
            const double sy = (static_cast<double>(y - test.content[1]) + 0.5) * test.source_h /
                                (test.content[3] - test.content[1]) -
                              0.5 - data.source[2];
            for (const auto &[cx, cy] : centers) {
              expected = std::min(expected, (sx - cx) * (sx - cx) + (sy - cy) * (sy - cy));
            }
          }
          check(q_error, final_[static_cast<std::size_t>(y) * test.field_w + x], expected, position_error, x, y);
        }
      }
      EXPECT_EQ(horizontal_error.failures, 0u) << horizontal_error.first_failure;
      EXPECT_EQ(q_error.failures, 0u) << q_error.first_failure;
      maximum_q_absolute_error_ = std::max(maximum_q_absolute_error_, q_error.maximum_absolute);
      maximum_q_relative_error_ = std::max(maximum_q_relative_error_, q_error.maximum_relative);
      maximum_q_bound_ratio_ = std::max(maximum_q_bound_ratio_, q_error.maximum_bound_ratio);
      RecordProperty("gaussian_q_max_absolute_error", maximum_q_absolute_error_);
      RecordProperty("gaussian_q_max_relative_error_nonzero", maximum_q_relative_error_);
      RecordProperty("gaussian_q_max_error_to_bound_ratio", maximum_q_bound_ratio_);
    }

    double maximum_q_absolute_error_ = 0.0, maximum_q_relative_error_ = 0.0, maximum_q_bound_ratio_ = 0.0;
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11ComputeShader> horizontal_shader_, vertical_shader_;
    std::vector<float> horizontal_, final_;

    struct smoothing_result_t {
      std::vector<std::uint32_t> upper, vertical, final;
    };

    bool smooth(std::uint32_t width, std::uint32_t height, const rect_t &content, const std::vector<float> &proposal, const std::vector<float> &base, const std::vector<std::uint32_t> &control_words, bool subtitle, smoothing_result_t &result, bool zero_dispatch = false, std::uint32_t control_stride = sizeof(std::uint32_t)) {
      const auto size = static_cast<std::size_t>(width) * height;
      if (proposal.size() != size || base.size() != size) {
        return false;
      }
      ComPtr<ID3D11ComputeShader> vertical_shader, horizontal_shader;
      if (!compile("depth_coordinate_v2_vertical_limit_cs.hlsl", subtitle ? "subtitle_main" : "main", vertical_shader) || !compile("depth_coordinate_v2_limit_cs.hlsl", subtitle ? "subtitle_main" : "main", horizontal_shader)) {
        return false;
      }
      texture_t candidate, immutable_base, upper, vertical, final;
      const UINT output_binds = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      if (!create_texture(device_.Get(), width, height, DXGI_FORMAT_R32_FLOAT, D3D11_BIND_SHADER_RESOURCE, proposal.data(), candidate) || !create_texture(device_.Get(), width, height, DXGI_FORMAT_R32_FLOAT, D3D11_BIND_SHADER_RESOURCE, base.data(), immutable_base) || !create_texture(device_.Get(), width, height, DXGI_FORMAT_R32_FLOAT, output_binds, nullptr, upper) || !create_texture(device_.Get(), width, height, DXGI_FORMAT_R32_FLOAT, output_binds, nullptr, vertical) || !create_texture(device_.Get(), width, height, DXGI_FORMAT_R32_FLOAT, output_binds, nullptr, final)) {
        return false;
      }
      const std::vector<std::uint32_t> poison(size, 0x7fc12345u);
      for (auto *texture : {upper.texture.Get(), vertical.texture.Get(), final.texture.Get()}) {
        context_->UpdateSubresource(texture, 0u, nullptr, poison.data(), width * sizeof(float), 0u);
      }
      std::array<std::uint32_t, 16u> depth_words {};
      depth_words[0] = width;
      depth_words[1] = height;
      std::copy(content.begin(), content.end(), depth_words.begin() + 9u);
      const v2::constants_t coordinate_words {
        v2::model_calibrations.front().raw_coordinate_scale,
        v2::collapse_abs_epsilon,
        0.0f,
        0.0f,
        v2::requested_gain_for_config(v2::reference_pop_strength),
        v2::max_horizontal_slope,
        v2::direct_container_limit,
        v2::convergence_curve_default,
        v2::adaptive_policy_id,
      };
      ComPtr<ID3D11Buffer> depth_cb, coordinate_cb, control, zero_args;
      if (!create_buffer(device_.Get(), depth_words.data(), sizeof(depth_words), D3D11_BIND_CONSTANT_BUFFER, depth_cb) || !create_buffer(device_.Get(), &coordinate_words, sizeof(coordinate_words), D3D11_BIND_CONSTANT_BUFFER, coordinate_cb)) {
        return false;
      }
      ComPtr<ID3D11ShaderResourceView> control_srv;
      if (!control_words.empty()) {
        if (!create_buffer(device_.Get(), control_words.data(), control_words.size() * sizeof(std::uint32_t), D3D11_BIND_SHADER_RESOURCE, control, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, control_stride)) {
          return false;
        }
        D3D11_SHADER_RESOURCE_VIEW_DESC view {};
        view.Format = DXGI_FORMAT_UNKNOWN;
        view.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        view.Buffer.NumElements = control_words.size() * sizeof(std::uint32_t) / control_stride;
        if (FAILED(device_->CreateShaderResourceView(control.Get(), &view, &control_srv))) {
          return false;
        }
      }
      const std::array<std::uint32_t, 3u> zeros {};
      if (zero_dispatch && !create_buffer(device_.Get(), zeros.data(), sizeof(zeros), 0u, zero_args, D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS)) {
        return false;
      }
      ID3D11Buffer *cbs[2] {depth_cb.Get(), coordinate_cb.Get()};
      context_->CSSetConstantBuffers(0u, 2u, cbs);
      ID3D11ShaderResourceView *srvs[4] {candidate.srv.Get(), nullptr, control_srv.Get(), immutable_base.srv.Get()};
      context_->CSSetShaderResources(0u, 4u, srvs);
      ID3D11UnorderedAccessView *vertical_uavs[2] {upper.uav.Get(), vertical.uav.Get()};
      context_->CSSetUnorderedAccessViews(0u, 2u, vertical_uavs, nullptr);
      context_->CSSetShader(vertical_shader.Get(), nullptr, 0u);
      if (zero_dispatch) {
        context_->DispatchIndirect(zero_args.Get(), 0u);
      } else {
        context_->Dispatch((width + limiter_lines - 1u) / limiter_lines, 1u, 1u);
      }
      unbind();
      srvs[0] = vertical.srv.Get();
      context_->CSSetShaderResources(0u, 4u, srvs);
      context_->CSSetUnorderedAccessViews(0u, 1u, final.uav.GetAddressOf(), nullptr);
      context_->CSSetShader(horizontal_shader.Get(), nullptr, 0u);
      if (zero_dispatch) {
        context_->DispatchIndirect(zero_args.Get(), 0u);
      } else {
        context_->Dispatch((height + limiter_lines - 1u) / limiter_lines, 1u, 1u);
      }
      unbind();
      std::vector<std::uint32_t> candidate_after, base_after, candidate_before(size), base_before(size);
      std::memcpy(candidate_before.data(), proposal.data(), size * sizeof(float));
      std::memcpy(base_before.data(), base.data(), size * sizeof(float));
      if (!read_texture(device_.Get(), context_.Get(), upper.texture.Get(), result.upper) || !read_texture(device_.Get(), context_.Get(), vertical.texture.Get(), result.vertical) || !read_texture(device_.Get(), context_.Get(), final.texture.Get(), result.final) || !read_texture(device_.Get(), context_.Get(), candidate.texture.Get(), candidate_after) || !read_texture(device_.Get(), context_.Get(), immutable_base.texture.Get(), base_after)) {
        return false;
      }
      EXPECT_EQ(candidate_after, candidate_before);
      EXPECT_EQ(base_after, base_before);
      return candidate_after == candidate_before && base_after == base_before;
    }
  };

  TEST_F(HostSbsSubtitleContourDistanceGpuTest, EmptyAllAndTinyNativeDomainsWriteEveryCell) {
    for (const auto &[width, height] : std::array {
           std::pair {1u, 1u},
           std::pair {1u, 7u},
           std::pair {7u, 1u},
           std::pair {3u, 5u},
           std::pair {17u, 13u},
           std::pair {43u, 29u},
         }) {
      SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
      distance_case_t test {width, height, width + 2u, height + 2u, {1u, 1u, width + 1u, height + 1u}, {}};
      ASSERT_TRUE(run(test));
      expect_oracle(test);
      const auto crop_h = constants(test).source[3];
      for (std::uint32_t y = 0u; y < crop_h; ++y) {
        for (std::uint32_t x = 0u; x < width; ++x) {
          test.seeds.emplace_back(x, y);
        }
      }
      ASSERT_TRUE(run(test));
      expect_oracle(test);
    }
  }

  TEST_F(HostSbsSubtitleContourDistanceGpuTest, FractionalAnisotropicNodesUseFiveCircularSourceCenters) {
    const distance_case_t test {31u, 29u, 23u, 17u, {2u, 1u, 21u, 16u}, {{13, 4}}};
    ASSERT_TRUE(run(test));
    expect_oracle(test);
    // Fractional field samples generally do not land on a center. A projected field-pin or L1
    // transform cannot silently replace real-source Euclidean distances.
    bool fractional = false;
    for (auto q : final_) {
      if (std::isfinite(q) && std::abs(q - std::round(q)) > 0.01f) {
        fractional = true;
      }
    }
    EXPECT_TRUE(fractional);
  }

  TEST_F(HostSbsSubtitleContourDistanceGpuTest, VirtualEdgeCentersStayOutsideCropAndPaddingIsInfinity) {
    const distance_case_t test {257u, 130u, 129u, 65u, {1u, 2u, 128u, 64u}, {{0, 0}, {256, 0}, {0, 42}, {256, 42}, {128, 21}}};
    ASSERT_TRUE(run(test));
    expect_oracle(test);
    // Both virtual rows and both edge columns participate in the scratch: clipping shifted
    // centers to the source/crop would turn one of these finite costs into infinity.
    EXPECT_TRUE(std::isfinite(horizontal_[1u]));
    EXPECT_TRUE(std::isfinite(horizontal_[(43u + 3u) * 129u + 127u]));
    EXPECT_TRUE(std::isinf(final_[0u]));
  }

  TEST_F(HostSbsSubtitleContourDistanceGpuTest, ChunkCarriesAndSuccessorsHandleDisjointSeedsAndHoles) {
    const distance_case_t test {521u, 217u, 127u, 69u, {0u, 0u, 127u, 69u}, {{0, 0}, {7, 4}, {8, 4}, {63, 10}, {64, 10}, {65, 10}, {259, 40}, {260, 40}, {520, 86}}};
    ASSERT_TRUE(run(test));
    expect_oracle(test);
    EXPECT_TRUE(std::isfinite(horizontal_[126u]));
    EXPECT_GT(final_[30u * 127u + 80u], 0.0f);
  }

  TEST_F(HostSbsSubtitleContourDistanceGpuTest, ParabolaHullCrossingsMatchExhaustiveNearestCenter) {
    distance_case_t test {1023u, 768u, 77u, 141u, {2u, 3u, 75u, 139u}, {}};
    for (int y = 0; y < 171; ++y) {
      test.seeds.emplace_back((y * 127 + (y % 7) * 31) % 1023, y);
    }
    ASSERT_TRUE(run(test));
    expect_oracle(test);
  }

  TEST_F(HostSbsSubtitleContourDistanceGpuTest, MaximumCropLandscapeAndPortraitKeepBoundedNumericalError) {
    for (const auto &test : std::array {
           distance_case_t {5120u, 5120u, 3080u, 868u, {0u, 0u, 3080u, 868u}, {{0, 0}, {5119, 0}, {0, 853}, {5119, 853}, {2559, 427}}},
           distance_case_t {1536u, 5120u, 868u, 3080u, {1u, 1u, 867u, 3079u}, {{0, 0}, {1535, 255}, {767, 128}}},
         }) {
      SCOPED_TRACE(std::to_string(test.field_w) + "x" + std::to_string(test.field_h));
      ASSERT_TRUE(run(test));
      expect_oracle(test);
    }
    distance_case_t hull {5120u, 5120u, 37u, 868u, {1u, 0u, 36u, 868u}, {}};
    for (int y = 0; y < 854; y += 7) {
      hull.seeds.emplace_back((y * 97 + (y % 11) * 83) % 5120, y);
    }
    ASSERT_TRUE(run(hull));
    expect_oracle(hull);
  }

  TEST_F(HostSbsSubtitleContourDistanceGpuTest, EveryGeneratedFieldShapeUsesSufficientGrid16Groups) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> shapes;
    for (const auto &shape : v2::model_calibrated_shapes) {
      shapes.emplace_back(shape.width, shape.height);
    }
    for (const auto &shape : v2::subtitle_ocr_live_field_shapes) {
      shapes.emplace_back(shape[0], shape[1]);
    }
    ASSERT_EQ(shapes.size(), v2::model_calibrated_shapes.size() + v2::subtitle_ocr_live_field_shapes.size());
    for (const auto &[width, height] : shapes) {
      SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
      const auto groups = ((width + 15u) / 16u) * ((height + 15u) / 16u);
      ASSERT_GE(groups, 858u);
      ASSERT_GE(groups, width);
      const distance_case_t test {31u, 29u, width, height, {0u, 0u, width, height}, {{13, 4}}};
      ASSERT_TRUE(run(test));
      expect_oracle(test);
    }
  }

  TEST_F(HostSbsSubtitleContourDistanceGpuTest, MalformedDomainAndDimensionsPublishOnlyPositiveInfinity) {
    const distance_case_t test {97u, 83u, 83u, 61u, {0u, 0u, 83u, 61u}, {{40, 8}}};
    auto data = constants(test);
    const auto check_empty = [&]() {
      EXPECT_TRUE(std::all_of(horizontal_.begin(), horizontal_.end(), [](float q) {
        return std::isinf(q) && q > 0.0f;
      }));
      EXPECT_TRUE(std::all_of(final_.begin(), final_.end(), [](float q) {
        return std::isinf(q) && q > 0.0f;
      }));
    };
    for (int fault = 0; fault < 6; ++fault) {
      auto bad = data;
      switch (fault) {
        case 0:
          bad.field[2] = 1u;
          break;
        case 1:
          ++bad.source[2];
          break;
        case 2:
          ++bad.source[3];
          break;
        case 3:
          bad.content[2] = test.field_w + 1u;
          break;
        case 4:
          bad.content[2] = bad.content[0];
          break;
        case 5:
          bad.source[0] = 5121u;
          break;
      }
      SCOPED_TRACE(fault);
      ASSERT_TRUE(run(test, &bad));
      check_empty();
    }
    ASSERT_TRUE(run(test, nullptr, false, true));
    check_empty();
  }

  TEST_F(HostSbsSubtitleContourDistanceGpuTest, ZeroIndirectReuseDoesNotWriteDistanceBackingsOrSource) {
    const distance_case_t test {257u, 130u, 129u, 65u, {0u, 0u, 129u, 65u}, {{128, 21}}};
    ASSERT_TRUE(run(test, nullptr, true));
    for (auto value : horizontal_) {
      EXPECT_EQ(bits(value), 0x7fc12345u);
    }
    for (auto value : final_) {
      EXPECT_EQ(bits(value), 0x7fc12345u);
    }
  }

  TEST_F(HostSbsSubtitleContourDistanceGpuTest, DisabledAndMalformedSmoothingCopyBaseBitsAndNearestContentPadding) {
    constexpr std::uint32_t width = 35u, height = 27u;
    const rect_t content {3u, 4u, 31u, 24u};
    std::vector<float> base(width * height, 0.03f), proposal(width * height, -0.04f);
    for (std::uint32_t y = content[1]; y < content[3]; ++y) {
      for (std::uint32_t x = content[0]; x < content[2]; ++x) {
        const auto index = y * width + x;
        base[index] = (x + y) % 3u == 0u ? from_bits(0x80000000u) :
                      (x + y) % 3u == 1u ? from_bits(0x3b123457u) :
                                           from_bits(0xbb123459u);
      }
    }
    // Flag1 with an extra word is malformed; no loose truthy/first-word check is allowed.
    for (const auto &control : std::array {std::vector<std::uint32_t> {0u}, std::vector<std::uint32_t> {2u}, std::vector<std::uint32_t> {1u, 0u}, std::vector<std::uint32_t> {}}) {
      smoothing_result_t result;
      ASSERT_TRUE(smooth(width, height, content, proposal, base, control, true, result));
      for (std::uint32_t y = 0u; y < height; ++y) {
        for (std::uint32_t x = 0u; x < width; ++x) {
          const auto sx = std::clamp(x, content[0], content[2] - 1u);
          const auto sy = std::clamp(y, content[1], content[3] - 1u);
          const auto expected = bits(base[sy * width + sx]);
          const auto index = y * width + x;
          ASSERT_EQ(result.upper[index], expected);
          ASSERT_EQ(result.vertical[index], expected);
          ASSERT_EQ(result.final[index], expected);
        }
      }
    }
    smoothing_result_t wrong_stride;
    ASSERT_TRUE(smooth(width, height, content, proposal, base, {1u, 0u}, true, wrong_stride, false, 8u));
    for (std::uint32_t y = 0u; y < height; ++y) {
      for (std::uint32_t x = 0u; x < width; ++x) {
        const auto sx = std::clamp(x, content[0], content[2] - 1u);
        const auto sy = std::clamp(y, content[1], content[3] - 1u);
        const auto expected = bits(base[sy * width + sx]);
        const auto index = y * width + x;
        ASSERT_EQ(wrong_stride.upper[index], expected);
        ASSERT_EQ(wrong_stride.vertical[index], expected);
        ASSERT_EQ(wrong_stride.final[index], expected);
      }
    }
  }

  TEST_F(HostSbsSubtitleContourDistanceGpuTest, EnabledSubtitleEntriesUseIdenticalOriginalCoreAndBothSlopeBounds) {
    for (const auto &[width, height] : std::array {std::pair {7u, 5u}, std::pair {509u, 293u}}) {
      SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
      const rect_t content {0u, 0u, width, height};
      std::vector<float> proposal(static_cast<std::size_t>(width) * height), base(proposal.size(), -0.031f);
      bool input_exceeds_slope = false;
      for (std::uint32_t y = 0u; y < height; ++y) {
        for (std::uint32_t x = 0u; x < width; ++x) {
          // A deliberately unsafe scalar proposal. The native shared limiter, rather than a
          // CPU recurrence or a hard-pin correction, must restore the two slope bounds.
          proposal[static_cast<std::size_t>(y) * width + x] =
            width == 7u ? ((x & 1u) != 0u ? -v2::direct_container_limit : v2::direct_container_limit) :
                          v2::direct_container_limit * std::sin(0.7f * x) * std::cos(0.8f * y);
          const auto index = static_cast<std::size_t>(y) * width + x;
          if (x != 0u) {
            input_exceeds_slope |= std::abs(proposal[index] - proposal[index - 1u]) > v2::max_horizontal_slope / width;
          }
          if (y != 0u) {
            input_exceeds_slope |= std::abs(proposal[index] - proposal[index - width]) > v2::max_vertical_shear / width;
          }
        }
      }
      ASSERT_TRUE(input_exceeds_slope);
      smoothing_result_t original, subtitle;
      ASSERT_TRUE(smooth(width, height, content, proposal, base, {1u}, false, original));
      ASSERT_TRUE(smooth(width, height, content, proposal, base, {1u}, true, subtitle));
      EXPECT_EQ(subtitle.upper, original.upper);
      EXPECT_EQ(subtitle.vertical, original.vertical);
      EXPECT_EQ(subtitle.final, original.final);
      bool changed = false;
      for (std::uint32_t y = 0u; y < height; ++y) {
        for (std::uint32_t x = 0u; x < width; ++x) {
          const auto index = static_cast<std::size_t>(y) * width + x;
          const auto value = from_bits(subtitle.final[index]);
          ASSERT_TRUE(std::isfinite(value));
          EXPECT_LE(std::abs(value), v2::direct_container_limit);
          changed |= subtitle.final[index] != bits(proposal[index]);
          if (x != 0u) {
            EXPECT_LE(std::abs(value - from_bits(subtitle.final[index - 1u])), v2::max_horizontal_slope / width + 1.0e-7f);
          }
          if (y != 0u) {
            EXPECT_LE(std::abs(value - from_bits(subtitle.final[index - width])), v2::max_vertical_shear / width + 1.0e-7f);
          }
        }
      }
      EXPECT_TRUE(changed);
    }
  }

  TEST_F(HostSbsSubtitleContourDistanceGpuTest, ZeroIndirectReuseHoldsAllSubtitleSmoothingOutputs) {
    constexpr std::uint32_t width = 35u, height = 27u;
    std::vector<float> proposal(width * height, 0.025f), base(width * height, -0.025f);
    smoothing_result_t result;
    ASSERT_TRUE(smooth(width, height, {0u, 0u, width, height}, proposal, base, {1u}, true, result, true));
    for (const auto *output : {&result.upper, &result.vertical, &result.final}) {
      for (auto word : *output) {
        EXPECT_EQ(word, 0x7fc12345u);
      }
    }
  }

  TEST_F(HostSbsSubtitleContourDistanceGpuTest, LiteralGaussianProposalAloneDoesNotGuaranteeContractivity) {
    // Mathematical exposed-arc witness, not a generated field or a renderer. The production
    // proposal must be followed by the actual shared limiter tested above.
    constexpr double gap = 32.0, sigma = 12.0;
    const double slope = gap * std::exp(-0.5) / sigma;
    EXPECT_GT(slope, 1.0);
  }
}  // namespace

#endif
