/** Executes the actual adaptive Host state, candidate, quantile and horizontal shaders on D3D11 WARP. */
#include "../tests_common.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <src/depth_coordinate_v2.h>
#include <src/host_sbs_v2_gpu_executor.h>
#include <src/generated/sbs_adaptive_state_contract.h>

#ifdef _WIN32
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

namespace {
  namespace v2 = models::depth_coordinate_v2;
  using Microsoft::WRL::ComPtr;

  struct gain_buffer_t {
    ComPtr<ID3D11Buffer> buffer;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
  };

  class HostSbsAdaptiveCameraGpuTest: public ::testing::Test {
  protected:
    static constexpr UINT width = 64u;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11ComputeShader> state_shader, map_shader, horizontal_shader;
    ComPtr<ID3D11Buffer> constants, depth_constants, owner_constants;
    gain_buffer_t state, stats, cut, raw, minmax, owner;
    ComPtr<ID3D11Texture2D> candidate, final, exclusion;
    ComPtr<ID3D11UnorderedAccessView> candidate_uav, final_uav;
    ComPtr<ID3D11ShaderResourceView> candidate_srv, exclusion_srv;
    v2::constants_t config {};
    sbs_adaptive_state::words_t cut_words = sbs_adaptive_state::initial_words;
    v2::state_words_t words = v2::state_initial_words;

    bool buffer(const void *data, UINT bytes, UINT stride, UINT flags, gain_buffer_t &out) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = bytes;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = flags;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = stride;
      D3D11_SUBRESOURCE_DATA initial {data, 0u, 0u};
      return SUCCEEDED(device->CreateBuffer(&desc, data ? &initial : nullptr, &out.buffer)) &&
        (!(flags & D3D11_BIND_SHADER_RESOURCE) ||
          SUCCEEDED(device->CreateShaderResourceView(out.buffer.Get(), nullptr, &out.srv))) &&
        (!(flags & D3D11_BIND_UNORDERED_ACCESS) ||
          SUCCEEDED(device->CreateUnorderedAccessView(out.buffer.Get(), nullptr, &out.uav)));
    }

    bool cbuffer(const void *data, UINT bytes, ComPtr<ID3D11Buffer> &out) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = bytes;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA initial {data, 0u, 0u};
      return SUCCEEDED(device->CreateBuffer(&desc, &initial, &out));
    }

    bool compile(const char *name, ComPtr<ID3D11ComputeShader> &out) {
      ComPtr<ID3DBlob> code, errors;
      const auto path = std::filesystem::path(SUNSHINE_SHADERS_DIR) / name;
      const auto status = D3DCompileFromFile(path.c_str(), nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0u, &code, &errors);
      if (FAILED(status)) {
        ADD_FAILURE() << name << ": " << (errors ?
          static_cast<const char *>(errors->GetBufferPointer()) : "no compiler diagnostic");
        return false;
      }
      return SUCCEEDED(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(),
        nullptr, &out));
    }

    void unbind() {
      ID3D11ShaderResourceView *srvs[8] {};
      ID3D11UnorderedAccessView *uavs[6] {};
      context->CSSetShaderResources(0u, 8u, srvs);
      context->CSSetUnorderedAccessViews(0u, 6u, uavs, nullptr);
    }

    void SetUp() override {
      D3D_FEATURE_LEVEL level {};
      const D3D_FEATURE_LEVEL requested[] {D3D_FEATURE_LEVEL_11_0};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0u,
        requested, 1u, D3D11_SDK_VERSION, &device, &level, &context)));
      ASSERT_TRUE(compile("depth_coordinate_v2_state_resolve_cs.hlsl", state_shader));
      ASSERT_TRUE(compile("depth_coordinate_v2_map_cs.hlsl", map_shader));
      ASSERT_TRUE(compile("depth_coordinate_v2_limit_cs.hlsl", horizontal_shader));
      config.raw_coordinate_scale = v2::model_calibrations.front().raw_coordinate_scale;
      config.collapse_abs_epsilon = v2::collapse_abs_epsilon;
      config.far_tau = v2::far_tau;
      config.near_log_tau = v2::near_log_tau;
      config.requested_gain = v2::requested_gain_for_config(1.75f);
      config.max_horizontal_slope = v2::max_horizontal_slope;
      config.direct_container_limit = v2::direct_container_limit;
      config.convergence_curve_default = v2::convergence_curve_default;
      config.joint_plane_mode = 3u;
      ASSERT_TRUE(cbuffer(&config, sizeof(config), constants));
      std::array<std::uint32_t, 16u> depth {};
      depth[0] = width;
      depth[1] = 1u;
      depth[11] = width;
      depth[12] = 1u;
      ASSERT_TRUE(cbuffer(depth.data(), sizeof(depth), depth_constants));
      const std::array<std::uint32_t, 20u> zero_owner {};
      ASSERT_TRUE(cbuffer(zero_owner.data(), sizeof(zero_owner), owner_constants));
      const std::array<float, v2::frame_stats_float_count> zero_stats {};
      const std::array<float, 4u> zero_minmax {};
      const std::array<float, width> zero_raw {};
      ASSERT_TRUE(buffer(words.data(), sizeof(words), 16u,
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, state));
      ASSERT_TRUE(buffer(zero_stats.data(), sizeof(zero_stats), 16u,
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, stats));
      ASSERT_TRUE(buffer(cut_words.data(), sizeof(cut_words), 16u, D3D11_BIND_SHADER_RESOURCE, cut));
      ASSERT_TRUE(buffer(zero_minmax.data(), sizeof(zero_minmax), 16u, D3D11_BIND_SHADER_RESOURCE, minmax));
      ASSERT_TRUE(buffer(zero_raw.data(), sizeof(zero_raw), 4u, D3D11_BIND_SHADER_RESOURCE, raw));
      ASSERT_TRUE(buffer(zero_owner.data(), sizeof(zero_owner), 4u, D3D11_BIND_UNORDERED_ACCESS, owner));
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = 1u;
      desc.MipLevels = desc.ArraySize = 1u;
      desc.SampleDesc.Count = 1u;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.Format = DXGI_FORMAT_R32_FLOAT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      ASSERT_TRUE(SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &candidate)));
      ASSERT_TRUE(SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &final)));
      ASSERT_TRUE(SUCCEEDED(device->CreateUnorderedAccessView(candidate.Get(), nullptr, &candidate_uav)));
      ASSERT_TRUE(SUCCEEDED(device->CreateUnorderedAccessView(final.Get(), nullptr, &final_uav)));
      ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(candidate.Get(), nullptr, &candidate_srv)));
      desc.Format = DXGI_FORMAT_R32_UINT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      const std::array<std::uint32_t, width> zeros {};
      D3D11_SUBRESOURCE_DATA data {zeros.data(), sizeof(zeros), 0u};
      ASSERT_TRUE(SUCCEEDED(device->CreateTexture2D(&desc, &data, &exclusion)));
      ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(exclusion.Get(), nullptr, &exclusion_srv)));
    }

    void read_state() {
      D3D11_BUFFER_DESC desc {};
      state.buffer->GetDesc(&desc);
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = desc.MiscFlags = desc.StructureByteStride = 0u;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Buffer> staging;
      ASSERT_TRUE(SUCCEEDED(device->CreateBuffer(&desc, nullptr, &staging)));
      context->CopyResource(staging.Get(), state.buffer.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      ASSERT_TRUE(SUCCEEDED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped)));
      std::memcpy(words.data(), mapped.pData, sizeof(words));
      context->Unmap(staging.Get(), 0u);
    }

    float scalar(std::size_t index) const { return std::bit_cast<float>(words[index]); }

    std::array<float, v2::frame_stats_float_count> measure_quantiles(
        const std::array<float, width> &values) {
      namespace executor = models::host_sbs_v2_gpu;
      std::array<float, v2::frame_stats_float_count> result {};
      ComPtr<ID3D11ComputeShader> moments_shader, frame_shader, histogram_shader, quantile_shader;
      if (!compile("depth_coordinate_v2_moments_cs.hlsl", moments_shader) ||
          !compile("depth_coordinate_v2_frame_resolve_cs.hlsl", frame_shader) ||
          !compile("depth_coordinate_v2_histogram_cs.hlsl", histogram_shader) ||
          !compile("depth_coordinate_v2_quantiles_cs.hlsl", quantile_shader)) return result;
      const std::array<std::uint32_t, 12u> partial_words {};
      const std::array<std::uint32_t, 256u> histogram_words {};
      gain_buffer_t partials, histogram;
      if (!buffer(partial_words.data(), sizeof(partial_words), 16u,
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, partials) ||
          !buffer(histogram_words.data(), sizeof(histogram_words), 16u,
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, histogram)) return result;
      D3D11_BUFFER_DESC normalization_desc {};
      normalization_desc.ByteWidth = 16u;
      normalization_desc.Usage = D3D11_USAGE_DEFAULT;
      normalization_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      normalization_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
      ComPtr<ID3D11Buffer> normalization;
      ComPtr<ID3D11UnorderedAccessView> normalization_uav;
      if (FAILED(device->CreateBuffer(&normalization_desc, nullptr, &normalization))) return result;
      D3D11_UNORDERED_ACCESS_VIEW_DESC view {};
      view.Format = DXGI_FORMAT_R32_TYPELESS;
      view.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      view.Buffer.NumElements = 4u;
      view.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      if (FAILED(device->CreateUnorderedAccessView(normalization.Get(), &view, &normalization_uav)))
        return result;
      std::array<std::uint32_t, 16u> depth {};
      depth[0] = width;
      depth[1] = 1u;
      depth[5] = 256u;
      depth[11] = width;
      depth[12] = 1u;
      context->UpdateSubresource(depth_constants.Get(), 0u, nullptr, depth.data(), 0u, 0u);
      context->UpdateSubresource(constants.Get(), 0u, nullptr, &config, 0u, 0u);
      context->UpdateSubresource(raw.buffer.Get(), 0u, nullptr, values.data(), 0u, 0u);
      const executor::moments_frame_command_t command {
        .constants = {depth_constants.Get(), constants.Get()},
        .moments_shader = moments_shader.Get(),
        .frame_resolve_shader = frame_shader.Get(),
        .raw_depth = raw.srv.Get(),
        .tensor_exclusion = exclusion_srv.Get(),
        .partials_output = partials.uav.Get(),
        .partials = partials.srv.Get(),
        .frame_stats_output = stats.uav.Get(),
        .minmax_raw_output = normalization_uav.Get(),
        .moments_dispatch = executor::dispatch_command_t::direct(1u, 1u, 1u),
        .frame_resolve_dispatch = executor::dispatch_command_t::direct(1u, 1u, 1u),
        .robust_quantiles = true,
        .histogram_shader = histogram_shader.Get(),
        .quantile_shader = quantile_shader.Get(),
        .frame_stats = stats.srv.Get(),
        .histogram_output = histogram.uav.Get(),
        .histogram = histogram.srv.Get(),
      };
      if (!executor::record_moments_frame(context.Get(), command)) {
        ADD_FAILURE() << "shared executor rejected native quantile test";
        return result;
      }
      D3D11_BUFFER_DESC staging_desc {};
      stats.buffer->GetDesc(&staging_desc);
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = staging_desc.MiscFlags = staging_desc.StructureByteStride = 0u;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Buffer> staging;
      if (FAILED(device->CreateBuffer(&staging_desc, nullptr, &staging))) return result;
      context->CopyResource(staging.Get(), stats.buffer.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return result;
      std::memcpy(result.data(), mapped.pData, sizeof(result));
      context->Unmap(staging.Get(), 0u);
      return result;
    }

    std::string state_diagnostic() const {
      std::ostringstream out;
      out << " state_words=";
      for (auto word : words) out << word << ',';
      out << " expected_seal=" << v2::camera_center_integrity_for_state_words(words)
          << " tail_valid=" << v2::adaptive_camera_tail_is_valid(words);
      return out.str();
    }

    void observe(std::uint64_t timestamp_us, float low, float high, bool pulse = false,
                 bool collapsed = false, bool quantile_valid = true) {
      config.joint_observation_timestamp_low = static_cast<std::uint32_t>(timestamp_us);
      config.joint_observation_timestamp_high = static_cast<std::uint32_t>(timestamp_us >> 32u);
      context->UpdateSubresource(constants.Get(), 0u, nullptr, &config, 0u, 0u);
      // Exact equal-area two-layer moments provide a known distribution without a CPU renderer.
      std::array<float, v2::frame_stats_float_count> frame {low * .5f + high * .5f,
        collapsed ? 0.0f : (high - low) * .5f, low, high, float(width), float(width), 1.0f, 0.0f};
      if (config.joint_plane_mode == 3u) {
        frame[v2::frame_stat_percentile_low] = low;
        frame[v2::frame_stat_percentile_high] = high;
        frame[v2::frame_stat_percentile_valid] = quantile_valid ? 1.0f : 0.0f;
        frame[v2::frame_stat_percentile_bin_width] = (high - low) / 256.0f;
      }
      context->UpdateSubresource(stats.buffer.Get(), 0u, nullptr, frame.data(), 0u, 0u);
      cut_words[sbs_adaptive_state::index(sbs_adaptive_state::word_e::hard_cut_pulse)] =
        std::bit_cast<std::uint32_t>(pulse ? 1.0f : 0.0f);
      if (pulse) ++cut_words[sbs_adaptive_state::index(sbs_adaptive_state::word_e::hard_cut_count)];
      context->UpdateSubresource(cut.buffer.Get(), 0u, nullptr, cut_words.data(), 0u, 0u);
      ID3D11Buffer *cb = constants.Get();
      context->CSSetConstantBuffers(1u, 1u, &cb);
      ID3D11ShaderResourceView *srvs[] {stats.srv.Get(), cut.srv.Get()};
      context->CSSetShaderResources(0u, 2u, srvs);
      ID3D11UnorderedAccessView *uav = state.uav.Get();
      context->CSSetUnorderedAccessViews(0u, 1u, &uav, nullptr);
      context->CSSetShader(state_shader.Get(), nullptr, 0u);
      context->Dispatch(1u, 1u, 1u);
      unbind();
      read_state();
    }

    void seed(float low = 0.0f, float high = 4.0f, std::uint64_t first_us = 100000u) {
      observe(first_us, low, high);
      ASSERT_EQ(scalar(v2::frame_valid), 1.0f);
      ASSERT_TRUE(v2::parallax_state_words_are_authenticated(words, config.raw_coordinate_scale, 3u));
    }

    std::vector<float> map_and_limit(float value, bool horizontal = false) {
      std::array<float, width> values;
      values.fill(value);
      context->UpdateSubresource(raw.buffer.Get(), 0u, nullptr, values.data(), 0u, 0u);
      ID3D11Buffer *cbs[] {depth_constants.Get(), constants.Get(), owner_constants.Get()};
      context->CSSetConstantBuffers(0u, 3u, cbs);
      ID3D11ShaderResourceView *srvs[] {raw.srv.Get(), state.srv.Get(), exclusion_srv.Get(),
        minmax.srv.Get(), nullptr, nullptr, cut.srv.Get(), nullptr};
      context->CSSetShaderResources(0u, 8u, srvs);
      ID3D11UnorderedAccessView *uavs[] {candidate_uav.Get(), nullptr, nullptr, nullptr, nullptr, owner.uav.Get()};
      context->CSSetUnorderedAccessViews(0u, 6u, uavs, nullptr);
      context->CSSetShader(map_shader.Get(), nullptr, 0u);
      context->Dispatch((width + 15u) / 16u, 1u, 1u);
      unbind();
      ID3D11Texture2D *source = candidate.Get();
      if (horizontal) {
        ID3D11ShaderResourceView *input = candidate_srv.Get();
        ID3D11UnorderedAccessView *output = final_uav.Get();
        context->CSSetShaderResources(0u, 1u, &input);
        context->CSSetUnorderedAccessViews(0u, 1u, &output, nullptr);
        context->CSSetShader(horizontal_shader.Get(), nullptr, 0u);
        context->Dispatch(1u, 1u, 1u);
        unbind();
        source = final.Get();
      }
      D3D11_TEXTURE2D_DESC desc {};
      source->GetDesc(&desc);
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = 0u;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      ComPtr<ID3D11Texture2D> staging;
      if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging))) return {};
      context->CopyResource(staging.Get(), source);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) return {};
      std::vector<float> result(width);
      std::memcpy(result.data(), mapped.pData, width * sizeof(float));
      context->Unmap(staging.Get(), 0u);
      return result;
    }
  };
}

TEST_F(HostSbsAdaptiveCameraGpuTest, CapsCurrentNearPixelsWhileLogGainAndZeroAdapt) {
  seed(0.0f, 4.0f, 850000u);
  const float old_gain = scalar(v2::inverse_scale), old_zero = scalar(v2::center);
  observe(950000u, 0.0f, 400.0f);
  EXPECT_NEAR(scalar(v2::inverse_scale), old_gain * std::exp(-std::log(2.0f) * .1f), 2.0e-7f);
  EXPECT_GT(scalar(v2::center), old_zero);
  EXPECT_LE(scalar(v2::center) - old_zero, .1f / scalar(v2::inverse_scale) + 1.0e-6f);
  EXPECT_TRUE(v2::parallax_state_words_are_authenticated(words, config.raw_coordinate_scale, 3u));
  for (const bool horizontal : {false, true}) {
    const auto field = map_and_limit(400.0f, horizontal);
    ASSERT_EQ(field.size(), width);
    for (float value : field) {
      EXPECT_TRUE(std::isfinite(value));
      EXPECT_LE(std::abs(value), v2::direct_container_limit);
      EXPECT_FLOAT_EQ(value, v2::direct_container_limit);
    }
  }
}

TEST_F(HostSbsAdaptiveCameraGpuTest, GapCutAndBadClocksEarnNoCatchupAcrossLowWordRollover) {
  const std::uint64_t first = 0xffffffffull - 350000ull;
  seed(0.0f, 4.0f, first);
  const auto last = first;
  float held_zero = scalar(v2::center), held_gain = scalar(v2::inverse_scale);
  EXPECT_EQ(words[v2::gain_last_observation_high], 0u);
  observe(last + 300000u, 0.0f, 40.0f);
  EXPECT_FLOAT_EQ(scalar(v2::center), held_zero);
  EXPECT_FLOAT_EQ(scalar(v2::inverse_scale), held_gain);
  observe(last + 400000u, 0.0f, 40.0f, true);
  EXPECT_GT(scalar(v2::center), held_zero);
  EXPECT_LT(scalar(v2::inverse_scale), held_gain);
  held_zero = scalar(v2::center);
  held_gain = scalar(v2::inverse_scale);
  EXPECT_NE(words[v2::gain_last_observation_high], 0u);
  for (const auto clock : {std::uint64_t(0u), last + 400000u, last + 399999u}) {
    observe(clock, 0.0f, 40.0f);
    EXPECT_FLOAT_EQ(scalar(v2::frame_valid), 0.0f);
    EXPECT_FLOAT_EQ(scalar(v2::center), held_zero);
    EXPECT_FLOAT_EQ(scalar(v2::inverse_scale), held_gain);
    EXPECT_TRUE(v2::parallax_state_words_are_authenticated(words, config.raw_coordinate_scale, 3u));
  }
  observe(last + 500000u, 0.0f, 40.0f);
  EXPECT_FLOAT_EQ(scalar(v2::center), held_zero);
  observe(last + 600000u, 0.0f, 40.0f);
  EXPECT_GT(scalar(v2::center), held_zero);
}

TEST_F(HostSbsAdaptiveCameraGpuTest, CollapseFailsFlatAndSignedDepthRetainsCamera) {
  seed(-2.0f, 4.0f); // Finite negative values remain part of the mean zero.
  EXPECT_FLOAT_EQ(scalar(v2::center), 1.0f);
  const float held = scalar(v2::inverse_scale);
  observe(950000u, 0.0f, 4.0f, false, true);
  EXPECT_FLOAT_EQ(scalar(v2::frame_valid), 0.0f);
  EXPECT_FLOAT_EQ(scalar(v2::inverse_scale), held);
  const auto field = map_and_limit(2.0f);
  ASSERT_EQ(field.size(), width);
  EXPECT_TRUE(std::all_of(field.begin(), field.end(), [](float value) { return value == 0.0f; }));
  observe(1050000u, -4.0f, -2.0f);
  EXPECT_FLOAT_EQ(scalar(v2::frame_valid), 1.0f);
  EXPECT_FLOAT_EQ(scalar(v2::inverse_scale), held);
  EXPECT_TRUE(v2::parallax_state_words_are_authenticated(words, config.raw_coordinate_scale, 3u));
}

TEST_F(HostSbsAdaptiveCameraGpuTest, TailTamperingCannotAuthorizeOrPreservePseudoCamera) {
  seed();
  const auto valid = words;
  for (std::size_t index = v2::gain_last_observation_low; index < v2::state_float_count; ++index) {
    auto corrupt = valid;
    corrupt[index] ^= 1u;
    EXPECT_FALSE(v2::parallax_state_words_are_authenticated(corrupt, config.raw_coordinate_scale, 3u)) << index;
  }
  words[v2::gain_reserved0] = 1u;
  words[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_state_words(words);
  EXPECT_FALSE(v2::parallax_state_words_are_authenticated(words, config.raw_coordinate_scale, 3u));
  context->UpdateSubresource(state.buffer.Get(), 0u, nullptr, words.data(), 0u, 0u);
  observe(950000u, 0.0f, 8.0f);
  EXPECT_FLOAT_EQ(scalar(v2::inverse_scale), 0.25f);
  EXPECT_FLOAT_EQ(scalar(v2::center), 4.0f);
  EXPECT_EQ(words[v2::gain_seed_count], 1u);
  EXPECT_TRUE(v2::parallax_state_words_are_authenticated(words, config.raw_coordinate_scale, 3u));
}

TEST_F(HostSbsAdaptiveCameraGpuTest, MissingQuantileAuthorityCannotFallBackToExtrema) {
  seed();
  const float held_zero = scalar(v2::center), held_inverse = scalar(v2::inverse_scale);
  observe(200000u, -100.0f, 100.0f, false, false, false);
  EXPECT_FLOAT_EQ(scalar(v2::frame_valid), 0.0f);
  EXPECT_FLOAT_EQ(scalar(v2::center), held_zero);
  EXPECT_FLOAT_EQ(scalar(v2::inverse_scale), held_inverse);
  const auto field = map_and_limit(100.0f);
  ASSERT_EQ(field.size(), width);
  EXPECT_TRUE(std::all_of(field.begin(), field.end(), [](float value) { return value == 0.0f; }));
  EXPECT_TRUE(v2::parallax_state_words_are_authenticated(words, config.raw_coordinate_scale, 3u));
  observe(300000u, -100.0f, 100.0f);
  EXPECT_FLOAT_EQ(scalar(v2::frame_valid), 1.0f);
  EXPECT_FLOAT_EQ(scalar(v2::center), held_zero);
  EXPECT_FLOAT_EQ(scalar(v2::inverse_scale), held_inverse);
}

TEST_F(HostSbsAdaptiveCameraGpuTest, ProductionReferenceRetainsLatchAndZeroControllerTail) {
  config.joint_plane_mode = 0u;
  observe(100000u, 0.0f, 4.0f);
  EXPECT_FLOAT_EQ(scalar(v2::center), 2.0f);
  EXPECT_FLOAT_EQ(scalar(v2::inverse_scale), 1.0f / config.raw_coordinate_scale);
  observe(200000u, 0.0f, 8.0f);
  EXPECT_FLOAT_EQ(scalar(v2::center), 2.0f);
  EXPECT_TRUE(std::all_of(words.begin() + v2::gain_last_observation_low, words.end(),
    [](std::uint32_t word) { return word == 0u; }));
  EXPECT_TRUE(v2::parallax_state_words_are_authenticated(words, config.raw_coordinate_scale, 0u));
}

TEST_F(HostSbsAdaptiveCameraGpuTest, RemovedCameraModesCannotAuthorizeGeometry) {
  for (const std::uint32_t mode : {1u, 2u, 4u}) {
    config.joint_plane_mode = mode;
    observe(100000u, 0.0f, 4.0f);
    EXPECT_FLOAT_EQ(scalar(v2::frame_valid), 0.0f);
    EXPECT_EQ(words[v2::renderer_authorization_bits], 0u);
    const auto field = map_and_limit(3.0f);
    ASSERT_EQ(field.size(), width);
    EXPECT_TRUE(std::all_of(field.begin(), field.end(), [](float value) { return value == 0.0f; }));
  }
}

TEST_F(HostSbsAdaptiveCameraGpuTest, HostMeanAdapterStartsImmediatelyWithModelAmplitudePrior) {
  config.joint_plane_mode = 3u;
  observe(100000u, 10.0f, 14.0f);
  ASSERT_FLOAT_EQ(scalar(v2::frame_valid), 1.0f);
  EXPECT_EQ(words[v2::gain_seed_count], 1u);
  EXPECT_FLOAT_EQ(scalar(v2::center), 12.0f);
  EXPECT_FLOAT_EQ(scalar(v2::inverse_scale), 1.0f / config.raw_coordinate_scale);
  EXPECT_FLOAT_EQ(scalar(v2::gain_display_limit), v2::direct_container_limit);
  ASSERT_TRUE(v2::parallax_state_words_are_authenticated(words, config.raw_coordinate_scale, 3u));
  auto zero = map_and_limit(12.0f);
  ASSERT_EQ(zero.size(), width);
  EXPECT_FLOAT_EQ(zero.front(), 0.0f);
  auto foreground = map_and_limit(13.0f);
  ASSERT_EQ(foreground.size(), width);
  EXPECT_NEAR(foreground.front(), config.requested_gain / config.raw_coordinate_scale, 1.0e-7f);
}

TEST_F(HostSbsAdaptiveCameraGpuTest, HostMeanAdapterTracksAcrossCutsWithoutSpendingGapTime) {
  config.joint_plane_mode = 3u;
  observe(100000u, 0.0f, 4.0f);
  const float original = scalar(v2::center);
  observe(200000u, 10.0f, 14.0f, true);
  EXPECT_GT(scalar(v2::center), original);
  EXPECT_LE(scalar(v2::center) - original, .1f / scalar(v2::inverse_scale) + 1.0e-6f);
  const float held = scalar(v2::center);
  observe(600000u, 100.0f, 104.0f);
  EXPECT_FLOAT_EQ(scalar(v2::center), held);
  observe(700000u, 100.0f, 104.0f);
  EXPECT_GT(scalar(v2::center), held);
  EXPECT_TRUE(v2::parallax_state_words_are_authenticated(words, config.raw_coordinate_scale, 3u));
  observe(700000u, 200.0f, 204.0f);
  EXPECT_FLOAT_EQ(scalar(v2::frame_valid), 0.0f);
  EXPECT_TRUE(v2::parallax_state_words_are_authenticated(words, config.raw_coordinate_scale, 3u));
}

TEST_F(HostSbsAdaptiveCameraGpuTest, HostMeanAdapterIsLinearUntilHardRepresentationLimit) {
  config.joint_plane_mode = 3u;
  observe(100000u, 0.0f, 4.0f);
  const float inverse = scalar(v2::inverse_scale);
  const float original_zero = scalar(v2::center);
  const auto foreground = map_and_limit(6.5f); // coordinate +2, beyond the old linear curve segment.
  const auto nearer = map_and_limit(8.75f); // coordinate +3, still below the representation cap.
  const auto background = map_and_limit(-2.5f); // coordinate -2 must also remain linear.
  ASSERT_EQ(foreground.size(), width);
  ASSERT_EQ(nearer.size(), width);
  ASSERT_EQ(background.size(), width);
  EXPECT_NEAR(foreground.front(), config.requested_gain * 2.0f, 1.0e-8f);
  EXPECT_NEAR(nearer.front(), config.requested_gain * 3.0f, 1.0e-8f);
  EXPECT_NEAR(background.front(), -config.requested_gain * 2.0f, 1.0e-8f);
  EXPECT_GT(nearer.front(), foreground.front());
  const float separation = nearer.front() - foreground.front();

  // A genuine next-clock observation moves zero while its spread, D and artistic gain stay fixed.
  // This must translate uncapped disparities equally, preserving foreground separation.
  observe(200000u, 10.0f, 14.0f);
  ASSERT_GT(scalar(v2::center), original_zero);
  EXPECT_FLOAT_EQ(scalar(v2::inverse_scale), inverse);
  const auto shifted_foreground = map_and_limit(6.5f);
  const auto shifted_nearer = map_and_limit(8.75f);
  ASSERT_EQ(shifted_foreground.size(), width);
  ASSERT_EQ(shifted_nearer.size(), width);
  EXPECT_NEAR(shifted_nearer.front() - shifted_foreground.front(), separation, 1.0e-8f);
  const float expected_translation =
    -config.requested_gain * (scalar(v2::center) - original_zero) * inverse;
  EXPECT_NEAR(shifted_foreground.front() - foreground.front(), expected_translation, 1.0e-8f);
  EXPECT_NEAR(shifted_nearer.front() - nearer.front(), expected_translation, 1.0e-8f);

  for (const bool horizontal : {false, true}) {
    const auto capped_near = map_and_limit(1000.0f, horizontal);
    const auto capped_far = map_and_limit(-1000.0f, horizontal);
    ASSERT_EQ(capped_near.size(), width);
    ASSERT_EQ(capped_far.size(), width);
    for (UINT index = 0u; index < width; ++index) {
      EXPECT_FLOAT_EQ(capped_near[index], v2::direct_container_limit);
      EXPECT_FLOAT_EQ(capped_far[index], -v2::direct_container_limit);
    }
  }
  EXPECT_FLOAT_EQ(scalar(v2::gain_display_limit), v2::direct_container_limit);

  // Shift raw and zero together on a fresh genuine acquisition; their coordinate is unchanged.
  words = v2::state_initial_words;
  context->UpdateSubresource(state.buffer.Get(), 0u, nullptr, words.data(), 0u, 0u);
  observe(300000u, 10.0f, 14.0f);
  const auto translated_coordinate = map_and_limit(16.5f);
  ASSERT_EQ(translated_coordinate.size(), width);
  EXPECT_NEAR(translated_coordinate.front(), foreground.front(), 1.0e-8f);
  observe(400000u, -14.0f, -10.0f);
  EXPECT_FLOAT_EQ(scalar(v2::frame_valid), 1.0f);
  EXPECT_TRUE(v2::parallax_state_words_are_authenticated(words, config.raw_coordinate_scale, 3u));
}

TEST_F(HostSbsAdaptiveCameraGpuTest, NativeHostQuantilesExcludeSparseTailsAndAdmitSignedDepth) {
  config.joint_plane_mode = 3u;
  std::array<float, width> values {};
  for (UINT index = 0u; index < width; ++index) values[index] = index < width / 2u ? 10.0f : 14.0f;
  values[0] = -10.0f;
  values[width - 1u] = 30.0f;
  auto measured = measure_quantiles(values);
  ASSERT_FLOAT_EQ(measured[v2::frame_stat_valid], 1.0f);
  ASSERT_FLOAT_EQ(measured[v2::frame_stat_percentile_valid], 1.0f);
  const float bin = measured[v2::frame_stat_percentile_bin_width];
  EXPECT_GE(measured[v2::frame_stat_percentile_low], 10.0f - bin);
  EXPECT_LE(measured[v2::frame_stat_percentile_low], 10.0f);
  EXPECT_GE(measured[v2::frame_stat_percentile_high], 14.0f);
  EXPECT_LE(measured[v2::frame_stat_percentile_high], 14.0f + bin);
  EXPECT_GT(measured[v2::frame_stat_percentile_low], measured[v2::frame_stat_minimum]);
  EXPECT_LT(measured[v2::frame_stat_percentile_high], measured[v2::frame_stat_maximum]);
  for (UINT index = 0u; index < width; ++index) values[index] = index < width / 2u ? -14.0f : -10.0f;
  measured = measure_quantiles(values);
  ASSERT_FLOAT_EQ(measured[v2::frame_stat_percentile_valid], 1.0f);
  EXPECT_FLOAT_EQ(measured[v2::frame_stat_mean], -12.0f);
  EXPECT_LE(measured[v2::frame_stat_percentile_high], -10.0f);
}

TEST_F(HostSbsAdaptiveCameraGpuTest, NativeHostQuantilesFailClosedOnNonfiniteContent) {
  config.joint_plane_mode = 3u;
  std::array<float, width> values {};
  values.fill(12.0f);
  values[0] = std::numeric_limits<float>::quiet_NaN();
  const auto measured = measure_quantiles(values);
  EXPECT_FLOAT_EQ(measured[v2::frame_stat_valid], 0.0f);
  EXPECT_FLOAT_EQ(measured[v2::frame_stat_percentile_valid], 0.0f);
  EXPECT_FLOAT_EQ(measured[v2::frame_stat_percentile_low], 0.0f);
  EXPECT_FLOAT_EQ(measured[v2::frame_stat_percentile_high], 0.0f);
  EXPECT_FLOAT_EQ(measured[v2::frame_stat_percentile_bin_width], 0.0f);
}
#endif
