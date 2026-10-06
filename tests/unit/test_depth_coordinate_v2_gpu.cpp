/**
 * @file tests/unit/test_depth_coordinate_v2_gpu.cpp
 * @brief Executable D3D11-WARP replay checks for depth-coordinate V2.
 *
 * The replay is intentionally the test surface here: it authenticates the generated contract,
 * compiles and dispatches the production V2 shaders, keeps the real GPU state buffer alive
 * across frames, and emits the same trace consumed by the NumPy comparison gate.
 */
#include "../tests_common.h"

#ifdef _WIN32

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <nlohmann/json.hpp>

#include <src/crypto.h>
#include <src/depth_coordinate_v2.h>
#include <src/host_sbs_v2_gpu_executor.h>
#include <src/prod_zipdepth_convex2x.h>
#include <src/sbs_bench_depth_coordinate_v2.h>
#include <src/video_depth_estimator.h>

namespace {
  using Microsoft::WRL::ComPtr;

  struct warp_device_t {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;

    bool initialize() {
      constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      return SUCCEEDED(D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_WARP,
        nullptr,
        0,
        requested,
        static_cast<UINT>(std::size(requested)),
        D3D11_SDK_VERSION,
        &device,
        &actual,
        &context
      )) && actual >= D3D_FEATURE_LEVEL_11_0;
    }
  };

  class temporary_tree_t {
  public:
    temporary_tree_t() {
      path = std::filesystem::temp_directory_path() /
        ("apollo-depth-coordinate-v2-replay-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
      std::filesystem::create_directories(path);
    }

    ~temporary_tree_t() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }

    std::filesystem::path path;
  };

  std::string read_bytes(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
      return {};
    }
    return {
      std::istreambuf_iterator<char>(stream),
      std::istreambuf_iterator<char>(),
    };
  }

  bool write_bytes(const std::filesystem::path &path, const std::string_view bytes) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return stream.good();
  }

  std::string sha256_hex(const std::string_view bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    const auto digest = crypto::hash(bytes);
    std::string result(digest.size() * 2u, '\0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
      result[index * 2u] = digits[digest[index] >> 4u];
      result[index * 2u + 1u] = digits[digest[index] & 0x0fu];
    }
    return result;
  }

  std::string float_bytes(const std::vector<float> &values) {
    return {
      reinterpret_cast<const char *>(values.data()),
      values.size() * sizeof(values.front()),
    };
  }

  const std::string &test_source_sha256() {
    static const std::string value = sha256_hex("depth-coordinate-v2-test-source");
    return value;
  }

  nlohmann::ordered_json make_replay_manifest(
    const models::depth_coordinate_v2::model_calibration_t &calibration,
    const std::string_view contract_sha256,
    const std::uint32_t width,
    const std::uint32_t height,
    const nlohmann::ordered_json &frames,
    const float pop_strength = 2.0f,
    const std::uint32_t source_width = 0u,
    const std::uint32_t source_height = 0u
  ) {
    namespace v2 = models::depth_coordinate_v2;
    auto authenticated_frames = frames;
    std::size_t observation_index = 0u;
    for (auto &frame : authenticated_frames) {
      if (!frame.contains("source_sha256")) {
        frame["source_sha256"] = test_source_sha256();
      }
      if (!frame.contains("observation_timestamp_us")) {
        frame["observation_timestamp_us"] = std::uint64_t(100000u + observation_index * 100000u);
      }
      ++observation_index;
    }
    return {
      {"schema", 11u},
      {"mode", "depth-coordinate-v2-production-gpu-sequence-v13"},
      {"calibration_contract", {
        {"file", "contracts/depth-coordinate-v2-v1.json"},
        {"schema", v2::contract_schema},
        {"sha256", contract_sha256},
      }},
      {"model_identity", {
        {"calibration_id", calibration.calibration_id},
        {"model", calibration.depth_model},
        {"depth_model_url", calibration.depth_model_url},
        {"onnx_sha256", calibration.onnx_sha256},
        {"preprocess_profile", calibration.preprocess.profile},
        {"preprocess_source_closure_sha256",
         calibration.preprocess.source_closure_sha256},
      }},
      {"raw_shape", {
        {"width", width},
        {"height", height},
        {"dtype", "float32-le"},
        {"layout", "row-major"},
      }},
      {"source_shape", {
        {"width", source_width ? source_width : width},
        {"height", source_height ? source_height : height},
      }},
      {"mapping_config", {
        {"raw_coordinate_scale", calibration.raw_coordinate_scale},
        {"collapse_abs_epsilon", v2::collapse_abs_epsilon},
        {"joint_plane_mode", v2::adaptive_policy_id},
        {"pop_strength", pop_strength},
        {"gain_per_pop", v2::gain_per_pop},
        {"max_horizontal_slope", v2::max_horizontal_slope},
        {"max_vertical_shear", v2::max_vertical_shear},
        {"vertical_majorant_share", v2::vertical_majorant_share},
        {"direct_container_limit", v2::direct_container_limit},
      }},
      {"cut_source", "unit-authenticated-hard-cut-generation"},
      {"frames", authenticated_frames},
    };
  }

  std::vector<float> least_rowwise_lipschitz_majorant(
    const std::vector<float> &candidate,
    const std::uint32_t width,
    const std::uint32_t height
  ) {
    std::vector<float> result = candidate;
    if (width == 0u || result.size() != static_cast<std::size_t>(width) * height) {
      return {};
    }
    const float max_step = models::depth_coordinate_v2::max_horizontal_slope /
                           static_cast<float>(width);
    for (std::uint32_t y = 0; y < height; ++y) {
      const std::size_t row = static_cast<std::size_t>(y) * width;
      for (std::uint32_t x = 1u; x < width; ++x) {
        const auto index = row + x;
        result[index] = std::max(candidate[index], result[index - 1u] - max_step);
      }
      for (std::uint32_t x = width - 1u; x > 0u; --x) {
        const auto index = row + x - 1u;
        result[index] = std::max(result[index], result[index + 1u] - max_step);
      }
    }
    return result;
  }

  std::vector<float> least_columnwise_lipschitz_majorant(
    const std::vector<float> &candidate,
    const std::uint32_t width,
    const std::uint32_t height
  ) {
    std::vector<float> result = candidate;
    if (width == 0u || height == 0u ||
        result.size() != static_cast<std::size_t>(width) * height) {
      return {};
    }
    const float max_step = models::depth_coordinate_v2::max_vertical_shear /
                           static_cast<float>(width);
    for (std::uint32_t x = 0; x < width; ++x) {
      for (std::uint32_t y = 1u; y < height; ++y) {
        const auto index = static_cast<std::size_t>(y) * width + x;
        result[index] = std::max(
          candidate[index],
          result[index - width] - max_step
        );
      }
      for (std::uint32_t y = height - 1u; y > 0u; --y) {
        const auto index = static_cast<std::size_t>(y - 1u) * width + x;
        result[index] = std::max(
          result[index],
          result[index + width] - max_step
        );
      }
    }
    return result;
  }

  std::vector<float> greatest_columnwise_lipschitz_minorant(
    const std::vector<float> &candidate,
    const std::uint32_t width,
    const std::uint32_t height
  ) {
    std::vector<float> result = candidate;
    if (width == 0u || height == 0u ||
        result.size() != static_cast<std::size_t>(width) * height) {
      return {};
    }
    const float max_step = models::depth_coordinate_v2::max_vertical_shear /
                           static_cast<float>(width);
    for (std::uint32_t x = 0; x < width; ++x) {
      for (std::uint32_t y = 1u; y < height; ++y) {
        const auto index = static_cast<std::size_t>(y) * width + x;
        result[index] = std::min(
          candidate[index],
          result[index - width] + max_step
        );
      }
      for (std::uint32_t y = height - 1u; y > 0u; --y) {
        const auto index = static_cast<std::size_t>(y - 1u) * width + x;
        result[index] = std::min(
          result[index],
          result[index + width] + max_step
        );
      }
    }
    return result;
  }

  float vertical_envelope_share(const float majorant, const float minorant) {
    const float majorant_share = models::depth_coordinate_v2::vertical_majorant_share;
    volatile float majorant_term = majorant_share * majorant;
    volatile float minorant_term = (1.0f - majorant_share) * minorant;
    return majorant_term + minorant_term;
  }

  std::vector<float> orientation_selective_vertical_conditioner(
    const std::vector<float> &candidate,
    const std::uint32_t width,
    const std::uint32_t height
  ) {
    const auto majorant = least_columnwise_lipschitz_majorant(candidate, width, height);
    const auto minorant = greatest_columnwise_lipschitz_minorant(candidate, width, height);
    if (majorant.size() != candidate.size() || minorant.size() != candidate.size()) {
      return {};
    }
    std::vector<float> result(candidate.size());
    for (std::size_t index = 0; index < result.size(); ++index) {
      result[index] = vertical_envelope_share(majorant[index], minorant[index]);
    }
    return result;
  }

  struct q30_vertical_oracle_t {
    std::vector<float> majorant;
    std::vector<float> conditioned;
  };

  std::int32_t limiter_upper_q30(float value) {
    namespace v2 = models::depth_coordinate_v2;
    value = std::isfinite(value) ?
      std::clamp(value, -v2::direct_container_limit, v2::direct_container_limit) :
      0.0f;
    return static_cast<std::int32_t>(
      std::ceil(value * static_cast<float>(v2::limiter_q_scale))
    );
  }

  std::int32_t limiter_lower_q30(float value) {
    namespace v2 = models::depth_coordinate_v2;
    value = std::isfinite(value) ?
      std::clamp(value, -v2::direct_container_limit, v2::direct_container_limit) :
      0.0f;
    return static_cast<std::int32_t>(
      std::floor(value * static_cast<float>(v2::limiter_q_scale))
    );
  }

  float limiter_from_q30(const std::int32_t value) {
    return static_cast<float>(value) /
           static_cast<float>(models::depth_coordinate_v2::limiter_q_scale);
  }

  std::int32_t limiter_step_q30(
    const std::uint32_t numerator,
    const std::uint32_t content_width
  ) {
    namespace v2 = models::depth_coordinate_v2;
    if (content_width == 0u) {
      return 0;
    }
    const std::uint32_t max_decay =
      2u * static_cast<std::uint32_t>(v2::limiter_container_q_limit);
    return static_cast<std::int32_t>(std::max(
      1u,
      std::min(numerator / content_width, max_decay)
    ));
  }

  q30_vertical_oracle_t exact_q30_vertical_oracle(
    const std::vector<float> &candidate,
    const std::uint32_t width,
    const std::uint32_t height,
    const std::uint32_t content_width
  ) {
    namespace v2 = models::depth_coordinate_v2;
    if (width == 0u || height <= v2::limiter_serial_max_lines || content_width == 0u ||
        candidate.size() != static_cast<std::size_t>(width) * height) {
      return {};
    }

    std::vector<std::int32_t> upper(candidate.size());
    std::vector<std::int32_t> lower(candidate.size());
    for (std::size_t index = 0; index < candidate.size(); ++index) {
      upper[index] = limiter_upper_q30(candidate[index]);
      lower[index] = limiter_lower_q30(candidate[index]);
    }
    const std::int32_t step = limiter_step_q30(
      v2::limiter_vertical_step_q_numerator,
      content_width
    );
    for (std::uint32_t x = 0u; x < width; ++x) {
      for (std::uint32_t y = 1u; y < height; ++y) {
        const std::size_t index = static_cast<std::size_t>(y) * width + x;
        upper[index] = std::max(upper[index], upper[index - width] - step);
        lower[index] = std::min(lower[index], lower[index - width] + step);
      }
      for (std::uint32_t y = height - 1u; y > 0u; --y) {
        const std::size_t index = static_cast<std::size_t>(y - 1u) * width + x;
        upper[index] = std::max(upper[index], upper[index + width] - step);
        lower[index] = std::min(lower[index], lower[index + width] + step);
      }
    }

    q30_vertical_oracle_t result;
    result.majorant.resize(candidate.size());
    result.conditioned.resize(candidate.size());
    for (std::size_t index = 0; index < candidate.size(); ++index) {
      const float majorant = limiter_from_q30(upper[index]);
      const float minorant = limiter_from_q30(lower[index]);
      result.majorant[index] = majorant;
      result.conditioned[index] = std::clamp(
        vertical_envelope_share(majorant, minorant),
        minorant,
        majorant
      );
    }
    return result;
  }

  std::vector<float> exact_q30_horizontal_oracle(
    const std::vector<float> &candidate,
    const std::uint32_t width,
    const std::uint32_t height,
    const std::uint32_t content_width
  ) {
    namespace v2 = models::depth_coordinate_v2;
    if (width <= v2::limiter_serial_max_lines || height == 0u || content_width == 0u ||
        candidate.size() != static_cast<std::size_t>(width) * height) {
      return {};
    }

    std::vector<std::int32_t> result_q30(candidate.size());
    std::transform(
      candidate.begin(),
      candidate.end(),
      result_q30.begin(),
      limiter_upper_q30
    );
    const std::int32_t step = limiter_step_q30(
      v2::limiter_horizontal_step_q_numerator,
      content_width
    );
    for (std::uint32_t y = 0u; y < height; ++y) {
      const std::size_t row = static_cast<std::size_t>(y) * width;
      for (std::uint32_t x = 1u; x < width; ++x) {
        const std::size_t index = row + x;
        result_q30[index] = std::max(
          result_q30[index],
          result_q30[index - 1u] - step
        );
      }
      for (std::uint32_t x = width - 1u; x > 0u; --x) {
        const std::size_t index = row + x - 1u;
        result_q30[index] = std::max(
          result_q30[index],
          result_q30[index + 1u] - step
        );
      }
    }

    std::vector<float> result(candidate.size());
    std::transform(
      result_q30.begin(),
      result_q30.end(),
      result.begin(),
      limiter_from_q30
    );
    return result;
  }

  void expect_float_fields_bitwise_equal(
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    const std::uint32_t width,
    const std::uint32_t content_width,
    const std::string_view field
  ) {
    ASSERT_EQ(actual.size(), expected.size()) << field;
    for (std::size_t index = 0; index < expected.size(); ++index) {
      const auto expected_bits = std::bit_cast<std::uint32_t>(expected[index]);
      const auto actual_bits = std::bit_cast<std::uint32_t>(actual[index]);
      if (actual_bits != expected_bits) {
        FAIL() << field << " differs at x=" << index % width
               << ", y=" << index / width
               << ", content_width=" << content_width
               << ": expected=" << expected[index]
               << " (0x" << std::hex << expected_bits << "), actual="
               << actual[index] << " (0x" << actual_bits << ')';
      }
    }
  }

  bool dispatch_depth_coordinate_v2_limit(
    warp_device_t &warp,
    ID3D11ComputeShader *shader,
    const std::uint32_t width,
    const std::uint32_t height,
    const std::uint32_t dispatch_lines,
    const std::vector<float> &candidate,
    std::vector<float> &result,
    std::vector<float> *secondary_result = nullptr,
    const std::uint32_t analysis_content_width = 0u
  ) {
    const std::uint32_t content_width = analysis_content_width == 0u ?
      width : analysis_content_width;
    if (!shader || width == 0u || height == 0u || content_width > width ||
        candidate.size() != static_cast<std::size_t>(width) * height) {
      return false;
    }

    D3D11_TEXTURE2D_DESC candidate_desc {};
    candidate_desc.Width = width;
    candidate_desc.Height = height;
    candidate_desc.MipLevels = 1u;
    candidate_desc.ArraySize = 1u;
    candidate_desc.Format = DXGI_FORMAT_R32_FLOAT;
    candidate_desc.SampleDesc.Count = 1u;
    candidate_desc.Usage = D3D11_USAGE_IMMUTABLE;
    candidate_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA candidate_data {};
    candidate_data.pSysMem = candidate.data();
    candidate_data.SysMemPitch = width * sizeof(float);

    ComPtr<ID3D11Texture2D> candidate_texture;
    ComPtr<ID3D11ShaderResourceView> candidate_srv;
    if (FAILED(warp.device->CreateTexture2D(
          &candidate_desc,
          &candidate_data,
          &candidate_texture
        )) ||
        FAILED(warp.device->CreateShaderResourceView(
          candidate_texture.Get(),
          nullptr,
          &candidate_srv
        ))) {
      return false;
    }

    D3D11_TEXTURE2D_DESC final_desc = candidate_desc;
    final_desc.Usage = D3D11_USAGE_DEFAULT;
    final_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    ComPtr<ID3D11Texture2D> final_texture;
    ComPtr<ID3D11UnorderedAccessView> final_uav;
    if (FAILED(warp.device->CreateTexture2D(
          &final_desc,
          nullptr,
          &final_texture
        )) ||
        FAILED(warp.device->CreateUnorderedAccessView(
          final_texture.Get(),
          nullptr,
          &final_uav
        ))) {
      return false;
    }
    ComPtr<ID3D11Texture2D> secondary_texture;
    ComPtr<ID3D11UnorderedAccessView> secondary_uav;
    if (secondary_result &&
        (FAILED(warp.device->CreateTexture2D(
           &final_desc,
           nullptr,
           &secondary_texture
         )) ||
         FAILED(warp.device->CreateUnorderedAccessView(
           secondary_texture.Get(),
           nullptr,
           &secondary_uav
         )))) {
      return false;
    }

    std::array<std::uint32_t, 16> constants {};
    constants[0] = width;
    constants[1] = height;
    constants[11] = content_width;
    constants[12] = height;
    D3D11_BUFFER_DESC constant_desc {};
    constant_desc.ByteWidth = sizeof(constants);
    constant_desc.Usage = D3D11_USAGE_IMMUTABLE;
    constant_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA constant_data {};
    constant_data.pSysMem = constants.data();
    ComPtr<ID3D11Buffer> constant_buffer;
    if (FAILED(warp.device->CreateBuffer(
          &constant_desc,
          &constant_data,
          &constant_buffer
        ))) {
      return false;
    }

    namespace v2 = models::depth_coordinate_v2;
    const v2::constants_t v2_constants {
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
    D3D11_BUFFER_DESC v2_constant_desc {};
    v2_constant_desc.ByteWidth = sizeof(v2_constants);
    v2_constant_desc.Usage = D3D11_USAGE_IMMUTABLE;
    v2_constant_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA v2_constant_data {};
    v2_constant_data.pSysMem = &v2_constants;
    ComPtr<ID3D11Buffer> v2_constant_buffer;
    if (FAILED(warp.device->CreateBuffer(
          &v2_constant_desc,
          &v2_constant_data,
          &v2_constant_buffer
        ))) {
      return false;
    }

    ID3D11ShaderResourceView *srvs[] = {candidate_srv.Get()};
    ID3D11UnorderedAccessView *uavs[] = {final_uav.Get(), secondary_uav.Get()};
    const UINT uav_count = secondary_result ? 2u : 1u;
    ID3D11Buffer *constant_buffers[] = {
      constant_buffer.Get(),
      v2_constant_buffer.Get()
    };
    const float unwritten[4] = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
    };
    warp.context->ClearUnorderedAccessViewFloat(final_uav.Get(), unwritten);
    if (secondary_uav) {
      warp.context->ClearUnorderedAccessViewFloat(secondary_uav.Get(), unwritten);
    }
    warp.context->CSSetShader(shader, nullptr, 0u);
    warp.context->CSSetShaderResources(0u, 1u, srvs);
    warp.context->CSSetUnorderedAccessViews(0u, uav_count, uavs, nullptr);
    warp.context->CSSetConstantBuffers(0u, 2u, constant_buffers);
    warp.context->Dispatch(models::host_sbs_v2_gpu::limiter_groups(dispatch_lines), 1u, 1u);

    ID3D11ShaderResourceView *null_srvs[] = {nullptr};
    ID3D11UnorderedAccessView *null_uavs[] = {nullptr, nullptr};
    warp.context->CSSetShaderResources(0u, 1u, null_srvs);
    warp.context->CSSetUnorderedAccessViews(0u, uav_count, null_uavs, nullptr);
    warp.context->CSSetShader(nullptr, nullptr, 0u);

    D3D11_TEXTURE2D_DESC staging_desc = final_desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0u;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(warp.device->CreateTexture2D(&staging_desc, nullptr, &staging))) {
      return false;
    }
    warp.context->CopyResource(staging.Get(), final_texture.Get());
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(warp.context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
      return false;
    }
    result.resize(candidate.size());
    for (std::uint32_t y = 0; y < height; ++y) {
      const auto *source = reinterpret_cast<const float *>(
        static_cast<const std::byte *>(mapped.pData) +
        static_cast<std::size_t>(y) * mapped.RowPitch
      );
      std::copy_n(
        source,
        width,
        result.begin() + static_cast<std::size_t>(y) * width
      );
    }
    warp.context->Unmap(staging.Get(), 0u);
    if (secondary_result) {
      ComPtr<ID3D11Texture2D> secondary_staging;
      if (FAILED(warp.device->CreateTexture2D(
            &staging_desc,
            nullptr,
            &secondary_staging
          ))) {
        return false;
      }
      warp.context->CopyResource(secondary_staging.Get(), secondary_texture.Get());
      D3D11_MAPPED_SUBRESOURCE secondary_mapped {};
      if (FAILED(warp.context->Map(
            secondary_staging.Get(),
            0u,
            D3D11_MAP_READ,
            0u,
            &secondary_mapped
          ))) {
        return false;
      }
      secondary_result->resize(candidate.size());
      for (std::uint32_t y = 0; y < height; ++y) {
        const auto *source = reinterpret_cast<const float *>(
          static_cast<const std::byte *>(secondary_mapped.pData) +
          static_cast<std::size_t>(y) * secondary_mapped.RowPitch
        );
        std::copy_n(
          source,
          width,
          secondary_result->begin() + static_cast<std::size_t>(y) * width
        );
      }
      warp.context->Unmap(secondary_staging.Get(), 0u);
    }
    return true;
  }

}  // namespace

TEST(HostSbsV2GpuExecutorTest, DispatchCommandsRejectInvalidShapesAndOffsets) {
  using dispatch_command_t = models::host_sbs_v2_gpu::dispatch_command_t;
  auto *const arguments = reinterpret_cast<ID3D11Buffer *>(std::uintptr_t {1u});

  EXPECT_FALSE(dispatch_command_t {}.valid());
  EXPECT_TRUE(dispatch_command_t::direct(1u, 1u, 1u).valid());
  EXPECT_FALSE(dispatch_command_t::direct(0u, 1u, 1u).valid());
  EXPECT_FALSE(dispatch_command_t::direct(1u, 0u, 1u).valid());
  EXPECT_FALSE(dispatch_command_t::direct(1u, 1u, 0u).valid());
  EXPECT_FALSE(dispatch_command_t::indirect(nullptr, 0u).valid());
  EXPECT_TRUE(dispatch_command_t::indirect(arguments, 0u).valid());
  EXPECT_TRUE(dispatch_command_t::indirect(arguments, 4u).valid());
  EXPECT_FALSE(dispatch_command_t::indirect(arguments, 2u).valid());
  auto mixed_direct = dispatch_command_t::direct(1u, 1u, 1u);
  mixed_direct.indirect_byte_offset = 4u;
  EXPECT_FALSE(mixed_direct.valid());
  auto mixed_indirect = dispatch_command_t::indirect(arguments, 0u);
  mixed_indirect.group_count_x = 1u;
  EXPECT_FALSE(mixed_indirect.valid());
}

TEST(HostSbsV2GpuExecutorTest, EveryStageRejectsMissingOperandsWithoutRecording) {
  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());

  using namespace models::host_sbs_v2_gpu;
  EXPECT_FALSE(record_moments_frame(warp.context.Get(), moments_frame_command_t {}));
  EXPECT_FALSE(record_state(warp.context.Get(), state_command_t {}));
  EXPECT_FALSE(record_map_history(warp.context.Get(), map_history_command_t {}));
  EXPECT_FALSE(record_vertical(warp.context.Get(), vertical_command_t {}));
  EXPECT_FALSE(record_horizontal(warp.context.Get(), horizontal_command_t {}));
}

TEST(HostSbsV2GpuExecutorTest, BindingContractAndExecutorCallSitesStayShared) {
  const auto source_root = std::filesystem::path(SUNSHINE_SOURCE_DIR) / "src";
  const auto executor = read_bytes(source_root / "host_sbs_v2_gpu_executor.cpp");
  const auto replay = read_bytes(source_root / "sbs_bench_depth_coordinate_v2.cpp");
  const auto live = read_bytes(source_root / "video_depth_estimator.cpp");
  ASSERT_FALSE(executor.empty());
  ASSERT_FALSE(replay.empty());
  ASSERT_FALSE(live.empty());

  const auto count_occurrences = [](
                                   const std::string_view haystack,
                                   const std::string_view needle
                                 ) {
    std::size_t count = 0u;
    for (std::size_t position = 0u;
         (position = haystack.find(needle, position)) != std::string_view::npos;
         position += needle.size()) {
      ++count;
    }
    return count;
  };
  const auto compact = [](const std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const char character : value) {
      if (character != ' ' && character != '\t' && character != '\r' &&
          character != '\n') {
        result.push_back(character);
      }
    }
    return result;
  };
  const auto compact_executor = compact(executor);
  const auto compact_replay = compact(replay);
  const auto compact_live = compact(live);
  const auto stage = [&compact_executor](
                       const std::string_view begin,
                       const std::string_view end
                     ) {
    const auto first = compact_executor.find(begin);
    const auto last = first == std::string::npos ?
                        std::string::npos :
                        compact_executor.find(end, first + begin.size());
    EXPECT_NE(first, std::string::npos) << begin;
    EXPECT_NE(last, std::string::npos) << end;
    if (first == std::string::npos || last == std::string::npos) {
      return std::string_view {};
    }
    return std::string_view(compact_executor).substr(first, last - first);
  };
  const auto moments = stage("boolrecord_moments_frame(", "boolrecord_state(");
  const auto state = stage("boolrecord_state(", "boolrecord_map_history(");
  const auto map = stage("boolrecord_map_history(", "boolrecord_vertical(");
  const auto vertical = stage("boolrecord_vertical(", "boolrecord_horizontal(");
  const auto horizontal = stage("boolrecord_horizontal(", "}//namespacemodels");

  // Behavioral GPU tests below cover the generated fields. This narrow structural guard proves
  // that both production call sites still cross the shared recorder API. Normalize formatting and
  // assert binding cardinality without coupling those assertions to local input/output array names.
  EXPECT_NE(moments.find("CSSetShaderResources(0u,2u,"),
            std::string::npos);
  EXPECT_NE(moments.find("CSSetUnorderedAccessViews(0u,1u"), std::string::npos);
  EXPECT_NE(moments.find("CSSetShaderResources(0u,1u,&command.partials)"),
            std::string::npos);
  EXPECT_NE(moments.find("CSSetUnorderedAccessViews(0u,2u"), std::string::npos);
  EXPECT_NE(state.find("CSSetShaderResources(0u,2u,"), std::string::npos);
  EXPECT_NE(state.find("CSSetUnorderedAccessViews(0u,1u"), std::string::npos);
  EXPECT_NE(compact_executor.find("CSSetConstantBuffers(0u,3u,"),
            std::string::npos);
  EXPECT_NE(map.find("bind_stage_constants(context,command.constants"),
            std::string::npos);
  EXPECT_NE(map.find("CSSetShaderResources(0u,8u,"), std::string::npos);
  EXPECT_NE(map.find("CSSetUnorderedAccessViews(0u,6u,"),
            std::string::npos);
  EXPECT_EQ(count_occurrences(map, "CSSetShaderResources(0u,8u,"), 2u);
  EXPECT_EQ(count_occurrences(map, "CSSetUnorderedAccessViews(0u,6u,"), 2u);
  EXPECT_NE(vertical.find("CSSetShaderResources(0u,1u"), std::string::npos);
  EXPECT_NE(vertical.find("CSSetUnorderedAccessViews(0u,2u"), std::string::npos);
  EXPECT_NE(horizontal.find("CSSetShaderResources(0u,1u"), std::string::npos);
  EXPECT_NE(horizontal.find("CSSetUnorderedAccessViews(0u,1u"),
            std::string::npos);

  for (const std::string_view recorder : {
         "record_moments_frame",
         "record_state",
         "record_map_history",
         "record_vertical",
         "record_horizontal",
       }) {
    EXPECT_EQ(count_occurrences(compact_replay, recorder), 1u) << recorder;
    EXPECT_EQ(count_occurrences(compact_live, recorder), 1u) << recorder;
  }
  EXPECT_EQ(count_occurrences(compact_replay, ".tensor_exclusion="), 2u);
  EXPECT_EQ(count_occurrences(compact_live, ".tensor_exclusion="), 2u);
  EXPECT_EQ(compact_replay.find("CSSetShader(moments_shader.Get()"),
            std::string::npos);
  EXPECT_EQ(compact_replay.find("CSSetShader(map_shader.Get()"), std::string::npos);
  EXPECT_EQ(compact_live.find("CSSetShader(depth_coordinate_v2_moments_cs.Get()"),
            std::string::npos);
  EXPECT_EQ(compact_live.find("CSSetShader(depth_coordinate_v2_map_cs.Get()"),
            std::string::npos);
  EXPECT_NE(compact_live.find("dispatch_command_t::direct("), std::string::npos);
  EXPECT_NE(compact_live.find("dispatch_command_t::indirect("), std::string::npos);
  EXPECT_NE(compact_live.find("near_identical_transaction.dispatch.Get()"),
            std::string::npos);
  for (const std::string_view offset : {
         "near_identical_gpu_infer_reduce_byte_offset",
         "near_identical_gpu_infer_one_byte_offset",
         "near_identical_gpu_infer_grid16_byte_offset",
         "near_identical_gpu_infer_columns_byte_offset",
         "near_identical_gpu_infer_rows_byte_offset",
       }) {
    EXPECT_NE(compact_live.find(offset), std::string::npos) << offset;
  }
  EXPECT_NE(compact_live.find("near_identical_history_owner.uav.Get()"),
            std::string::npos);
  EXPECT_NE(compact_live.find(".near_identical_constants=near_identical_cbuffer.Get()"),
            std::string::npos);
  EXPECT_NE(compact_live.find("CSSetConstantBuffers(1,2,"),
            std::string::npos);
  EXPECT_EQ(compact_live.find("depth_coordinate_v2_ownership"), std::string::npos);
  EXPECT_EQ(compact_live.find("pending_source_srv"), std::string::npos);
  EXPECT_NE(compact_live.find("mark_d3d_parallax_map_start(perf_slot)"),
            std::string::npos);
  EXPECT_NE(compact_live.find("mark_d3d_parallax_subtitle_start(perf_slot)"),
            std::string::npos);
}

TEST(DepthCoordinateV2GpuTest, LimiterIsExactLeastRowwiseLipschitzMajorant) {
  namespace v2 = models::depth_coordinate_v2;

  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());

  const std::filesystem::path shader_path =
    std::filesystem::path(SUNSHINE_SHADERS_DIR) /
    "depth_coordinate_v2_limit_cs.hlsl";
  ComPtr<ID3DBlob> shader_blob;
  ComPtr<ID3DBlob> shader_errors;
  const HRESULT compile_status = D3DCompileFromFile(
    shader_path.c_str(),
    nullptr,
    D3D_COMPILE_STANDARD_FILE_INCLUDE,
    "main",
    "cs_5_0",
    D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
    0u,
    &shader_blob,
    &shader_errors
  );
  ASSERT_TRUE(SUCCEEDED(compile_status))
    << (shader_errors ?
          static_cast<const char *>(shader_errors->GetBufferPointer()) :
          "no compiler diagnostics");
  ComPtr<ID3D11ComputeShader> shader;
  ASSERT_TRUE(SUCCEEDED(warp.device->CreateComputeShader(
    shader_blob->GetBufferPointer(),
    shader_blob->GetBufferSize(),
    nullptr,
    &shader
  )));

  const auto verify_case = [&](const std::uint32_t width,
                               const std::uint32_t height,
                               const std::vector<float> &candidate) {
    ASSERT_EQ(candidate.size(), static_cast<std::size_t>(width) * height);
    const auto oracle = least_rowwise_lipschitz_majorant(
      candidate,
      width,
      height
    );
    ASSERT_EQ(oracle.size(), candidate.size());

    std::vector<float> gpu;
    ASSERT_TRUE(dispatch_depth_coordinate_v2_limit(
      warp,
      shader.Get(),
      width,
      height,
      height,
      candidate,
      gpu
    ));
    ASSERT_EQ(gpu.size(), oracle.size());

    const float max_step = v2::max_horizontal_slope /
                           static_cast<float>(width);
    for (std::uint32_t y = 0; y < height; ++y) {
      const std::size_t row = static_cast<std::size_t>(y) * width;
      for (std::uint32_t x = 0; x < width; ++x) {
        const std::size_t index = row + x;
        if (width <= 32u) {
          EXPECT_FLOAT_EQ(gpu[index], oracle[index])
            << "x=" << x << ", y=" << y << ", width=" << width;
        } else {
          EXPECT_NEAR(gpu[index], oracle[index], 2.0e-7f)
            << "x=" << x << ", y=" << y << ", width=" << width;
        }
        EXPECT_TRUE(std::isfinite(gpu[index]))
          << "unwritten/nonfinite x=" << x << ", y=" << y << ", width=" << width;
        EXPECT_GE(gpu[index], candidate[index])
          << "x=" << x << ", y=" << y << ", width=" << width;
        if (x > 0u) {
          EXPECT_LE(std::abs(gpu[index] - gpu[index - 1u]), max_step + 2.0e-7f)
            << "slope x=" << x << ", y=" << y << ", width=" << width;
        }

        // The pointwise supremum is the definition of the least Lipschitz majorant.
        // Checking it independently prevents a shared one-direction scan bug in the GPU and
        // two-scan oracle from passing merely because both satisfy the weaker inequalities.
        float least = -std::numeric_limits<float>::infinity();
        for (std::uint32_t source_x = 0; source_x < width; ++source_x) {
          least = std::max(
            least,
            candidate[row + source_x] -
              max_step * static_cast<float>(
                x > source_x ? x - source_x : source_x - x
              )
          );
        }
        EXPECT_NEAR(gpu[index], least, 2.0e-7f)
          << "x=" << x << ", y=" << y << ", width=" << width;
      }
    }
  };

  constexpr std::uint32_t width = 32u;
  constexpr std::uint32_t height = 5u;
  std::vector<float> adversarial(static_cast<std::size_t>(width) * height);
  for (std::uint32_t x = 0; x < width; ++x) {
    // A high left plateau followed by a low right plateau requires the right-to-left scan.
    adversarial[x] = x < 19u ? 0.038f : -0.036f;
    // The mirrored cliff requires the left-to-right scan.
    adversarial[width + x] = x < 11u ? -0.039f : 0.034f;
    // A foreground plateau bracketed by background exercises both scans on the same row.
    adversarial[2u * width + x] = x >= 8u && x < 24u ? 0.036f : -0.032f;
    // Keep every value negative while retaining two cliffs and a nontrivial plateau.
    adversarial[3u * width + x] = x >= 7u && x < 22u ? -0.004f : -0.039f;
    // An already-valid flat plateau must remain byte-for-byte unchanged.
    adversarial[4u * width + x] = -0.017f;
  }
  verify_case(width, height, adversarial);

  // Exercise every balanced-chunk boundary around powers of two and at all authenticated axis
  // lengths. Impulses immediately before/after boundaries force carries to cross chunks.
  for (const std::uint32_t parallel_width :
       std::array<std::uint32_t, 9> {
         33u, 63u, 64u, 65u, 434u, 770u, 1022u, 1036u, 2072u
       }) {
    std::vector<float> boundary_impulses(parallel_width, -0.039f);
    for (std::uint32_t chunk = 1u; chunk < 32u; ++chunk) {
      const std::uint32_t boundary = chunk * parallel_width / 32u;
      boundary_impulses[boundary - 1u] = chunk % 2u == 0u ? 0.037f : -0.038f;
      boundary_impulses[boundary] = chunk % 2u == 0u ? -0.038f : 0.039f;
    }
    verify_case(parallel_width, 1u, boundary_impulses);
  }

  // The shader's target_w==1 path has no scan iterations and must preserve each row exactly.
  verify_case(1u, 3u, {-0.031f, 0.0f, 0.039f});
}

TEST(DepthCoordinateV2GpuTest, VerticalPassPublishesExactMajorantAndConditionedShare) {
  namespace v2 = models::depth_coordinate_v2;

  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());

  const std::filesystem::path shader_path =
    std::filesystem::path(SUNSHINE_SHADERS_DIR) /
    "depth_coordinate_v2_vertical_limit_cs.hlsl";
  ComPtr<ID3DBlob> shader_blob;
  ComPtr<ID3DBlob> shader_errors;
  const HRESULT compile_status = D3DCompileFromFile(
    shader_path.c_str(),
    nullptr,
    D3D_COMPILE_STANDARD_FILE_INCLUDE,
    "main",
    "cs_5_0",
    D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
    0u,
    &shader_blob,
    &shader_errors
  );
  ASSERT_TRUE(SUCCEEDED(compile_status))
    << (shader_errors ?
          static_cast<const char *>(shader_errors->GetBufferPointer()) :
          "no compiler diagnostics");
  ComPtr<ID3D11ComputeShader> shader;
  ASSERT_TRUE(SUCCEEDED(warp.device->CreateComputeShader(
    shader_blob->GetBufferPointer(),
    shader_blob->GetBufferSize(),
    nullptr,
    &shader
  )));

  const auto verify_case = [&](const std::uint32_t width,
                               const std::uint32_t height,
                               const std::vector<float> &candidate,
                               const bool check_pointwise = true) {
    ASSERT_EQ(candidate.size(), static_cast<std::size_t>(width) * height);
    const auto oracle = least_columnwise_lipschitz_majorant(
      candidate,
      width,
      height
    );
    const auto minorant_oracle = greatest_columnwise_lipschitz_minorant(
      candidate,
      width,
      height
    );
    const auto conditioned_oracle = orientation_selective_vertical_conditioner(
      candidate,
      width,
      height
    );
    ASSERT_EQ(oracle.size(), candidate.size());
    ASSERT_EQ(minorant_oracle.size(), candidate.size());
    ASSERT_EQ(conditioned_oracle.size(), candidate.size());

    std::vector<float> gpu;
    std::vector<float> conditioned_gpu;
    ASSERT_TRUE(dispatch_depth_coordinate_v2_limit(
      warp,
      shader.Get(),
      width,
      height,
      width,
      candidate,
      gpu,
      &conditioned_gpu
    ));
    ASSERT_EQ(gpu.size(), oracle.size());
    ASSERT_EQ(conditioned_gpu.size(), conditioned_oracle.size());

    const float max_step = v2::max_vertical_shear /
                           static_cast<float>(width);
    for (std::uint32_t y = 0; y < height; ++y) {
      for (std::uint32_t x = 0; x < width; ++x) {
        const std::size_t index = static_cast<std::size_t>(y) * width + x;
        if (height <= 32u) {
          EXPECT_FLOAT_EQ(gpu[index], oracle[index])
            << "x=" << x << ", y=" << y << ", height=" << height;
          EXPECT_FLOAT_EQ(conditioned_gpu[index], conditioned_oracle[index])
            << "conditioned x=" << x << ", y=" << y << ", height=" << height;
        } else {
          EXPECT_NEAR(gpu[index], oracle[index], 2.0e-7f)
            << "x=" << x << ", y=" << y << ", height=" << height;
          EXPECT_NEAR(conditioned_gpu[index], conditioned_oracle[index], 2.0e-7f)
            << "conditioned x=" << x << ", y=" << y << ", height=" << height;
        }
        EXPECT_TRUE(std::isfinite(gpu[index]));
        EXPECT_TRUE(std::isfinite(conditioned_gpu[index]));
        EXPECT_GE(gpu[index], candidate[index])
          << "x=" << x << ", y=" << y << ", height=" << height;
        EXPECT_GE(conditioned_gpu[index] + 2.0e-7f, minorant_oracle[index]);
        EXPECT_LE(conditioned_gpu[index], oracle[index] + 2.0e-7f);
        if (y > 0u) {
          const std::size_t prior = index - width;
          EXPECT_LE(std::abs(gpu[index] - gpu[prior]), max_step + 2.0e-7f);
          EXPECT_LE(
            std::abs(conditioned_gpu[index] - conditioned_gpu[prior]),
            max_step + 2.0e-7f
          );
        }

        // Check the defining pointwise supremum independently of the two directional scans.
        if (check_pointwise) {
          float least = -std::numeric_limits<float>::infinity();
          for (std::uint32_t source_y = 0; source_y < height; ++source_y) {
            least = std::max(
              least,
              candidate[static_cast<std::size_t>(source_y) * width + x] -
                max_step * static_cast<float>(
                  y > source_y ? y - source_y : source_y - y
                )
            );
          }
          EXPECT_NEAR(gpu[index], least, 2.0e-7f)
            << "x=" << x << ", y=" << y << ", height=" << height;
        }
      }
    }
  };

  constexpr std::uint32_t width = 5u;
  constexpr std::uint32_t height = 32u;
  std::vector<float> adversarial(static_cast<std::size_t>(width) * height);
  for (std::uint32_t y = 0; y < height; ++y) {
    // Each column independently requires one or both directional scans.
    adversarial[static_cast<std::size_t>(y) * width] =
      y < 19u ? 0.038f : -0.036f;
    adversarial[static_cast<std::size_t>(y) * width + 1u] =
      y < 11u ? -0.039f : 0.034f;
    adversarial[static_cast<std::size_t>(y) * width + 2u] =
      y >= 8u && y < 24u ? 0.036f : -0.032f;
    adversarial[static_cast<std::size_t>(y) * width + 3u] =
      y >= 7u && y < 22u ? -0.004f : -0.039f;
    adversarial[static_cast<std::size_t>(y) * width + 4u] = -0.017f;
  }
  verify_case(width, height, adversarial);

  constexpr std::uint32_t production_width = 434u;
  constexpr std::uint32_t production_height = 1036u;
  std::vector<float> production_boundaries(
    static_cast<std::size_t>(production_width) * production_height,
    -0.039f
  );
  for (std::uint32_t chunk = 1u; chunk < 32u; ++chunk) {
    const std::uint32_t boundary = chunk * production_height / 32u;
    const float before = chunk % 2u == 0u ? 0.037f : -0.038f;
    const float after = chunk % 2u == 0u ? -0.038f : 0.039f;
    std::fill_n(
      production_boundaries.begin() +
        static_cast<std::size_t>(boundary - 1u) * production_width,
      production_width,
      before
    );
    std::fill_n(
      production_boundaries.begin() +
        static_cast<std::size_t>(boundary) * production_width,
      production_width,
      after
    );
  }
  verify_case(production_width, production_height, production_boundaries, false);

  // The single-high fused portrait profiles have columns up to 2072 cells.  A narrow field
  // exercises the full shared-memory/carry length without turning this exact WARP contract test
  // into a multi-million-pixel fixture.
  constexpr std::uint32_t high_portrait_width = 7u;
  constexpr std::uint32_t high_portrait_height = 2072u;
  std::vector<float> high_portrait_boundaries(
    static_cast<std::size_t>(high_portrait_width) * high_portrait_height,
    -0.039f
  );
  for (std::uint32_t chunk = 1u; chunk < 32u; ++chunk) {
    const std::uint32_t boundary = chunk * high_portrait_height / 32u;
    for (std::uint32_t x = 0u; x < high_portrait_width; ++x) {
      high_portrait_boundaries[
        static_cast<std::size_t>(boundary - 1u) * high_portrait_width + x
      ] = chunk % 2u == 0u ? 0.037f : -0.038f;
      high_portrait_boundaries[
        static_cast<std::size_t>(boundary) * high_portrait_width + x
      ] = chunk % 2u == 0u ? -0.038f : 0.039f;
    }
  }
  verify_case(
    high_portrait_width,
    high_portrait_height,
    high_portrait_boundaries,
    false
  );

  // The target_h==1 path has no scan iterations and must preserve every column exactly.
  verify_case(3u, 1u, {-0.031f, 0.0f, 0.039f});
}

TEST(DepthCoordinateV2GpuTest, VerticalShareThenHorizontalMajorantMatchesC75AndBounds) {
  namespace v2 = models::depth_coordinate_v2;

  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());

  const auto compile = [&](const char *filename,
                           ComPtr<ID3D11ComputeShader> &shader) {
    const std::filesystem::path path =
      std::filesystem::path(SUNSHINE_SHADERS_DIR) / filename;
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> errors;
    const HRESULT status = D3DCompileFromFile(
      path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      "main",
      "cs_5_0",
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
      0u,
      &blob,
      &errors
    );
    ASSERT_TRUE(SUCCEEDED(status))
      << filename << ": "
      << (errors ? static_cast<const char *>(errors->GetBufferPointer()) :
                   "no compiler diagnostics");
    ASSERT_TRUE(SUCCEEDED(warp.device->CreateComputeShader(
      blob->GetBufferPointer(),
      blob->GetBufferSize(),
      nullptr,
      &shader
    )));
  };

  ComPtr<ID3D11ComputeShader> vertical_shader;
  ComPtr<ID3D11ComputeShader> horizontal_shader;
  compile("depth_coordinate_v2_vertical_limit_cs.hlsl", vertical_shader);
  compile("depth_coordinate_v2_limit_cs.hlsl", horizontal_shader);

  constexpr std::uint32_t width = 11u;
  constexpr std::uint32_t height = 13u;
  std::vector<float> candidate(static_cast<std::size_t>(width) * height);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      candidate[static_cast<std::size_t>(y) * width + x] =
        -0.039f + 0.001f * static_cast<float>((x * 5u + y * 7u) % 9u);
    }
  }
  candidate[2u * width + 8u] = 0.038f;
  candidate[9u * width + 3u] = 0.031f;
  candidate[6u * width + 5u] = 0.020f;
  for (std::uint32_t y = 4u; y <= 7u; ++y) {
    candidate[static_cast<std::size_t>(y) * width + 9u] = 0.026f;
  }

  std::vector<float> vertical_gpu;
  std::vector<float> vertical_conditioned_gpu;
  ASSERT_TRUE(dispatch_depth_coordinate_v2_limit(
    warp,
    vertical_shader.Get(),
    width,
    height,
    width,
    candidate,
    vertical_gpu,
    &vertical_conditioned_gpu
  ));
  std::vector<float> final_gpu;
  ASSERT_TRUE(dispatch_depth_coordinate_v2_limit(
    warp,
    horizontal_shader.Get(),
    width,
    height,
    height,
    vertical_conditioned_gpu,
    final_gpu
  ));

  const auto vertical_majorant_oracle = least_columnwise_lipschitz_majorant(
    candidate, width, height
  );
  const auto vertical_conditioned_oracle = orientation_selective_vertical_conditioner(
    candidate, width, height
  );
  const auto final_oracle = least_rowwise_lipschitz_majorant(
    vertical_conditioned_oracle, width, height
  );
  const auto shipped_upper_oracle = least_rowwise_lipschitz_majorant(
    vertical_majorant_oracle, width, height
  );
  ASSERT_EQ(vertical_gpu.size(), candidate.size());
  ASSERT_EQ(vertical_conditioned_gpu.size(), candidate.size());
  ASSERT_EQ(final_gpu.size(), candidate.size());
  ASSERT_EQ(final_oracle.size(), candidate.size());

  const float horizontal_step = v2::max_horizontal_slope /
                                static_cast<float>(width);
  const float vertical_step = v2::max_vertical_shear /
                              static_cast<float>(width);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const auto index = static_cast<std::size_t>(y) * width + x;
      EXPECT_FLOAT_EQ(vertical_gpu[index], vertical_majorant_oracle[index]);
      EXPECT_FLOAT_EQ(
        vertical_conditioned_gpu[index],
        vertical_conditioned_oracle[index]
      );
      EXPECT_FLOAT_EQ(final_gpu[index], final_oracle[index]);
      EXPECT_GE(vertical_gpu[index], candidate[index]);
      EXPECT_GE(final_gpu[index], vertical_conditioned_gpu[index]);
      EXPECT_LE(final_gpu[index], shipped_upper_oracle[index] + 2.0e-7f);

      if (x > 0u) {
        EXPECT_LE(
          std::abs(final_gpu[index] - final_gpu[index - 1u]),
          horizontal_step + 2.0e-7f
        ) << "horizontal bound at x=" << x << ", y=" << y;
      }
      if (y > 0u) {
        EXPECT_LE(
          std::abs(final_gpu[index] - final_gpu[index - width]),
          vertical_step + 2.0e-7f
        ) << "vertical bound at x=" << x << ", y=" << y;
      }
    }
  }
}

TEST(DepthCoordinateV2GpuTest, ParallelQ30LimitersMatchSerialOracleAtAuthenticatedShape) {
  namespace v2 = models::depth_coordinate_v2;

  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());

  const auto compile = [&](const char *filename,
                           ComPtr<ID3D11ComputeShader> &shader) {
    const std::filesystem::path path =
      std::filesystem::path(SUNSHINE_SHADERS_DIR) / filename;
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> errors;
    const HRESULT status = D3DCompileFromFile(
      path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      "main",
      "cs_5_0",
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
      0u,
      &blob,
      &errors
    );
    ASSERT_TRUE(SUCCEEDED(status))
      << filename << ": "
      << (errors ? static_cast<const char *>(errors->GetBufferPointer()) :
                   "no compiler diagnostics");
    ASSERT_TRUE(SUCCEEDED(warp.device->CreateComputeShader(
      blob->GetBufferPointer(),
      blob->GetBufferSize(),
      nullptr,
      &shader
    )));
  };

  ComPtr<ID3D11ComputeShader> vertical_shader;
  ComPtr<ID3D11ComputeShader> horizontal_shader;
  compile("depth_coordinate_v2_vertical_limit_cs.hlsl", vertical_shader);
  compile("depth_coordinate_v2_limit_cs.hlsl", horizontal_shader);

  constexpr std::uint32_t width = 770u;
  constexpr std::uint32_t height = 434u;
  ASSERT_GT(width, v2::limiter_serial_max_lines);
  ASSERT_GT(height, v2::limiter_serial_max_lines);
  // A partial last group exercises the idle threads that still reach every group barrier.
  ASSERT_NE(width % v2::limiter_group_lines, 0u);
  ASSERT_NE(height % v2::limiter_group_lines, 0u);
  ASSERT_TRUE(std::any_of(
    v2::model_calibrated_shapes.begin(),
    v2::model_calibrated_shapes.end(),
    [](const auto &shape) {
      return shape.width == width && shape.height == height;
    }
  ));

  std::vector<float> candidate(static_cast<std::size_t>(width) * height);
  for (std::uint32_t y = 0u; y < height; ++y) {
    for (std::uint32_t x = 0u; x < width; ++x) {
      std::uint32_t hash = x * 1664525u + y * 1013904223u;
      hash ^= hash >> 16u;
      const float unit = static_cast<float>(hash % 79001u) / 79000.0f;
      candidate[static_cast<std::size_t>(y) * width + x] =
        -0.039f + 0.078f * unit;
    }
  }

  // Put alternating cliffs on both sides of every balanced-chunk boundary so a serial oracle
  // catches a wrong carry distance/direction in either parallel pass.
  constexpr std::uint32_t boundary_row = 17u;
  constexpr std::uint32_t boundary_column = 29u;
  for (std::uint32_t chunk = 1u; chunk < v2::limiter_line_chunks; ++chunk) {
    const std::uint32_t boundary_x = chunk * width / v2::limiter_line_chunks;
    candidate[static_cast<std::size_t>(boundary_row) * width + boundary_x - 1u] =
      chunk % 2u == 0u ? 0.039f : -0.039f;
    candidate[static_cast<std::size_t>(boundary_row) * width + boundary_x] =
      chunk % 2u == 0u ? -0.039f : 0.039f;

    const std::uint32_t boundary_y = chunk * height / v2::limiter_line_chunks;
    candidate[static_cast<std::size_t>(boundary_y - 1u) * width + boundary_column] =
      chunk % 2u == 0u ? -0.039f : 0.039f;
    candidate[static_cast<std::size_t>(boundary_y) * width + boundary_column] =
      chunk % 2u == 0u ? 0.039f : -0.039f;
  }

  for (const std::uint32_t content_width :
       std::array<std::uint32_t, 2> {1u, 300u}) {
    SCOPED_TRACE("content_width=" + std::to_string(content_width));
    ASSERT_LT(content_width, width);

    const auto vertical_oracle = exact_q30_vertical_oracle(
      candidate,
      width,
      height,
      content_width
    );
    ASSERT_EQ(vertical_oracle.majorant.size(), candidate.size());
    ASSERT_EQ(vertical_oracle.conditioned.size(), candidate.size());
    std::vector<float> vertical_gpu;
    std::vector<float> conditioned_gpu;
    ASSERT_TRUE(dispatch_depth_coordinate_v2_limit(
      warp,
      vertical_shader.Get(),
      width,
      height,
      width,
      candidate,
      vertical_gpu,
      &conditioned_gpu,
      content_width
    ));
    expect_float_fields_bitwise_equal(
      vertical_oracle.majorant,
      vertical_gpu,
      width,
      content_width,
      "vertical_majorant"
    );
    expect_float_fields_bitwise_equal(
      vertical_oracle.conditioned,
      conditioned_gpu,
      width,
      content_width,
      "vertical_conditioned"
    );

    const auto final_oracle = exact_q30_horizontal_oracle(
      vertical_oracle.conditioned,
      width,
      height,
      content_width
    );
    ASSERT_EQ(final_oracle.size(), candidate.size());
    std::vector<float> final_gpu;
    ASSERT_TRUE(dispatch_depth_coordinate_v2_limit(
      warp,
      horizontal_shader.Get(),
      width,
      height,
      height,
      conditioned_gpu,
      final_gpu,
      nullptr,
      content_width
    ));
    expect_float_fields_bitwise_equal(
      final_oracle,
      final_gpu,
      width,
      content_width,
      "final"
    );
  }
}

TEST(DepthCoordinateV2ShapeTest, StandardSourceAspectsFitEveryAuthenticatedTensorShape) {
  namespace v2 = models::depth_coordinate_v2;
  const auto &calibration = v2::model_calibrations.front();
  struct shape_case_t {
    std::uint32_t source_width;
    std::uint32_t source_height;
    int tensor_width;
    int tensor_height;
  };
  constexpr std::array cases {
    shape_case_t {3840u, 2160u, 770, 434},
    shape_case_t {5120u, 2160u, 1022, 434},
    shape_case_t {3840u, 1600u, 1036, 434},
    shape_case_t {2160u, 3840u, 434, 770},
    shape_case_t {2160u, 5120u, 434, 1022},
    shape_case_t {1600u, 3840u, 434, 1036},
    shape_case_t {2048u, 1536u, 574, 434},
    shape_case_t {2388u, 1668u, 616, 434},
    shape_case_t {2360u, 1640u, 630, 434},
    shape_case_t {2160u, 1440u, 658, 434},
    shape_case_t {1920u, 1200u, 700, 434},
    shape_case_t {2160u, 1080u, 868, 434},
    shape_case_t {2340u, 1080u, 938, 434},
    shape_case_t {2400u, 1080u, 966, 434},
    shape_case_t {2424u, 1080u, 980, 434},
    shape_case_t {1536u, 2048u, 434, 574},
    shape_case_t {1668u, 2388u, 434, 616},
    shape_case_t {1640u, 2360u, 434, 630},
    shape_case_t {1440u, 2160u, 434, 658},
    shape_case_t {1200u, 1920u, 434, 700},
    shape_case_t {1080u, 2160u, 434, 868},
    shape_case_t {1080u, 2340u, 434, 938},
    shape_case_t {1080u, 2400u, 434, 966},
    shape_case_t {1080u, 2424u, 434, 980},
  };
  static_assert(cases.size() == 24u);

  for (const auto &test_case : cases) {
    SCOPED_TRACE(
      std::to_string(test_case.source_width) + "x" +
      std::to_string(test_case.source_height)
    );
    const auto fitted = models::fit_depth_tensor_shape(
      test_case.source_width,
      test_case.source_height,
      432,
      4.0f
    );
    EXPECT_EQ(fitted.width, test_case.tensor_width);
    EXPECT_EQ(fitted.height, test_case.tensor_height);
    EXPECT_TRUE(v2::model_calibration_supports_shape(
      calibration,
      static_cast<std::uint32_t>(fitted.width),
      static_cast<std::uint32_t>(fitted.height)
    ));
  }

  EXPECT_EQ(
    models::fit_depth_tensor_shape(0u, 2160u, 432, 4.0f),
    models::depth_tensor_shape_t {}
  );
  EXPECT_EQ(
    models::fit_depth_tensor_shape(3840u, 0u, 432, 4.0f),
    models::depth_tensor_shape_t {}
  );
}

TEST(DepthCoordinateV2GpuTest, EveryAuthenticatedTensorShapeExecutesProductionProducer) {
  namespace fs = std::filesystem;
  namespace prod = models::prod_zipdepth_convex2x;
  namespace v2 = models::depth_coordinate_v2;

  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());
  temporary_tree_t tree;

  const fs::path contract_source = fs::path(SUNSHINE_SOURCE_DIR) /
    "tools/sbsbench/contracts/depth-coordinate-v2-v1.json";
  const std::string contract_bytes = read_bytes(contract_source);
  ASSERT_FALSE(contract_bytes.empty());
  const std::string contract_sha256 = sha256_hex(contract_bytes);
  const auto &calibration = v2::model_calibrations.front();

  std::vector<prod::high_shape_t> authenticated_shapes;
  for (const auto &shape : v2::model_calibrated_shapes) {
    if (shape.calibration_id != calibration.calibration_id) {
      continue;
    }
    authenticated_shapes.push_back({shape.width, shape.height});
  }
  ASSERT_EQ(authenticated_shapes.size(), 24u);
  // Exercise the same producer closure at every public grid emitted by the fused runtime, in
  // addition to retaining every internal DAV2 calibration-grid replay above.
  for (const auto shape : prod::fixed_profile_shapes) {
    ASSERT_TRUE(prod::live_geometry_shape_relation(
      shape.width / prod::scale,
      shape.height / prod::scale,
      shape.width,
      shape.height
    ));
    authenticated_shapes.push_back(shape);
  }
  ASSERT_EQ(authenticated_shapes.size(), 48u);

  for (const auto shape : authenticated_shapes) {
    const std::uint32_t width = shape.width;
    const std::uint32_t height = shape.height;
    const std::uint32_t split = width / 2u;
    const std::uint32_t source_width = width * 2u;
    const std::uint32_t source_height = height * 2u;
    SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));

    const fs::path shape_root = tree.path /
      (std::to_string(width) + "x" + std::to_string(height));
    ASSERT_TRUE(write_bytes(
      shape_root / "contracts/depth-coordinate-v2-v1.json",
      contract_bytes
    ));

    const std::size_t element_count = static_cast<std::size_t>(width) * height;
    std::vector<float> raw(element_count, 0.0f);
    for (std::uint32_t y = 0; y < height; ++y) {
      std::fill(
        raw.begin() + static_cast<std::size_t>(y) * width + split,
        raw.begin() + static_cast<std::size_t>(y + 1u) * width,
        20.0f
      );
    }
    const std::string raw_bytes = float_bytes(raw);
    ASSERT_TRUE(write_bytes(shape_root / "raw_0001.f32", raw_bytes));
    const nlohmann::ordered_json frames = nlohmann::ordered_json::array({{
      {"frame_id", "0001"},
      {"raw_file", "raw_0001.f32"},
      {"raw_sha256", sha256_hex(raw_bytes)},
      {"hard_cut_count", 0u},
      {"hard_cut_pulse", false},
    }});
    const auto manifest = make_replay_manifest(
      calibration,
      contract_sha256,
      width,
      height,
      frames,
      1.0f,
      source_width,
      source_height
    );
    const fs::path manifest_path = shape_root / "manifest.json";
    ASSERT_TRUE(write_bytes(manifest_path, manifest.dump(2) + "\n"));

    std::string error;
    auto replay = sbs_bench::depth_coordinate_v2_gpu_replay::create(
      warp.device.Get(),
      warp.context.Get(),
      manifest_path,
      error
    );
    ASSERT_NE(replay, nullptr) << error;
    EXPECT_EQ(replay->width(), width);
    EXPECT_EQ(replay->height(), height);
    sbs_bench::depth_coordinate_v2_gpu_frame output;
    ASSERT_TRUE(replay->dispatch(
      0u, "0001", test_source_sha256(), output, error)) << error;
    EXPECT_EQ(output.canonical_values.size(), element_count);
    EXPECT_EQ(output.candidate_parallax_values.size(), element_count);
    EXPECT_EQ(output.vertical_majorant_values.size(), element_count);
    EXPECT_EQ(output.vertical_conditioned_values.size(), element_count);
    EXPECT_EQ(output.encoded_parallax_values.size(), element_count);
    EXPECT_LT(output.order_minimum, output.order_maximum);
    EXPECT_LE(
      output.maximum_absolute_source_u,
      v2::direct_container_limit + 2.0e-7f
    );
    for (std::size_t index = 0u; index < element_count; ++index) {
      EXPECT_GE(
        output.vertical_majorant_values[index] + 1.0e-8f,
        output.candidate_parallax_values[index]
      );
    }

    const fs::path trace_path = shape_root / "trace.json";
    ASSERT_TRUE(replay->write_state_trace(trace_path, error)) << error;
    const auto trace = nlohmann::ordered_json::parse(read_bytes(trace_path));
    ASSERT_EQ(trace.at("frames").size(), 1u);
    EXPECT_EQ(trace["producer"]["contract_canonical_sha256"],
              v2::contract_canonical_sha256);
    EXPECT_EQ(trace["frames"][0]["input_valid"], true);
    EXPECT_EQ(trace["frames"][0]["frame_valid"], true);
    EXPECT_EQ(trace["frames"][0]["camera_valid"], true);
    EXPECT_EQ(trace["frames"][0]["calibration_revision"], 1u);
  }
}

TEST(DepthCoordinateV2GpuTest, ArithmeticMeanCenterAdaptsAcrossCutsAndHoldsUnusableInputOnGpu) {
  namespace fs = std::filesystem;
  namespace v2 = models::depth_coordinate_v2;

  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());
  temporary_tree_t tree;

  const fs::path contract_source = fs::path(SUNSHINE_SOURCE_DIR) /
    "tools/sbsbench/contracts/depth-coordinate-v2-v1.json";
  const std::string contract_bytes = read_bytes(contract_source);
  ASSERT_FALSE(contract_bytes.empty());
  ASSERT_TRUE(write_bytes(
    tree.path / "contracts/depth-coordinate-v2-v1.json", contract_bytes));

  const auto &calibration = v2::model_calibrations.front();
  const auto shape_it = std::find_if(
    v2::model_calibrated_shapes.begin(),
    v2::model_calibrated_shapes.end(),
    [&](const auto &shape) {
      return shape.calibration_id == calibration.calibration_id;
    });
  ASSERT_NE(shape_it, v2::model_calibrated_shapes.end());
  const std::uint32_t width = shape_it->width;
  const std::uint32_t height = shape_it->height;
  const std::size_t element_count = static_cast<std::size_t>(width) * height;

  // A deliberately multimodal field proves that acquisition uses the arithmetic mean rather
  // than restaging around a histogram valley.
  std::vector<float> acquired(element_count);
  const std::size_t far_count = element_count * 4u / 10u;
  const std::size_t middle_count = element_count * 4u / 10u;
  const auto fill_linear = [&](const std::size_t begin, const std::size_t count,
                               const float low, const float high) {
    for (std::size_t offset = 0u; offset < count; ++offset) {
      const float unit = count > 1u ?
        static_cast<float>(offset) / static_cast<float>(count - 1u) : 0.0f;
      acquired[begin + offset] = low + (high - low) * unit;
    }
  };
  fill_linear(0u, far_count, 0.5f, 1.5f);
  fill_linear(far_count, middle_count, 2.5f, 3.5f);
  fill_linear(
    far_count + middle_count,
    element_count - far_count - middle_count,
    5.0f,
    5.5f
  );

  std::vector<std::vector<float>> fields(12u, acquired);
  for (float &value : fields[1]) {
    value += 0.25f;  // no cut: continuous source-time adaptation must move the camera
  }
  for (std::size_t index = 2u; index < fields.size(); ++index) {
    for (float &value : fields[index]) {
      value += index == 11u ? 1.25f : 1.0f;
    }
  }
  // A later unusable update and changed evidence after it must not replace the camera acquired
  // on the confirmed cut. Twelve updates deliberately extend beyond the former age-eight
  // replacement horizon.
  std::fill(fields[10].begin(), fields[10].end(), 2.0f);

  nlohmann::ordered_json frames = nlohmann::ordered_json::array();
  for (std::size_t index = 0u; index < fields.size(); ++index) {
    const std::string frame_id = "000" + std::to_string(index + 1u);
    const std::string raw_file = "raw_" + frame_id + ".f32";
    const std::string bytes = float_bytes(fields[index]);
    ASSERT_TRUE(write_bytes(tree.path / raw_file, bytes));
    frames.push_back({
      {"frame_id", frame_id},
      {"raw_file", raw_file},
      {"raw_sha256", sha256_hex(bytes)},
      {"hard_cut_count", index >= 2u ? 1u : 0u},
      {"hard_cut_pulse", index == 2u},
    });
  }

  const auto manifest = make_replay_manifest(
    calibration,
    sha256_hex(contract_bytes),
    width,
    height,
    frames,
    1.0f
  );
  const fs::path manifest_path = tree.path / "manifest.json";
  ASSERT_TRUE(write_bytes(manifest_path, manifest.dump(2) + "\n"));

  std::string error;
  auto replay = sbs_bench::depth_coordinate_v2_gpu_replay::create(
    warp.device.Get(), warp.context.Get(), manifest_path, error);
  ASSERT_NE(replay, nullptr) << error;
  for (std::size_t index = 0u; index < fields.size(); ++index) {
    sbs_bench::depth_coordinate_v2_gpu_frame output;
    const std::string frame_id = "000" + std::to_string(index + 1u);
    ASSERT_TRUE(replay->dispatch(
      index, frame_id, test_source_sha256(), output, error)) << error;
  }

  const fs::path trace_path = tree.path / "trace.json";
  ASSERT_TRUE(replay->write_state_trace(trace_path, error)) << error;
  const auto trace = nlohmann::ordered_json::parse(read_bytes(trace_path));
  ASSERT_EQ(trace.at("frames").size(), fields.size());
  const auto &rows = trace.at("frames");
  double center_sum = 0.0;
  for (const float value : acquired) {
    center_sum += value;
  }
  const float first_center = static_cast<float>(
    center_sum / static_cast<double>(acquired.size()));
  EXPECT_NEAR(rows[0]["center"].get<float>(), first_center, 2.0e-5f);
  EXPECT_FLOAT_EQ(
    rows[0]["convergence_curve"].get<float>(), v2::convergence_curve_default);
  EXPECT_GT(rows[1]["center"].get<float>(), first_center);
  EXPECT_LT(rows[1]["center"].get<float>(), first_center + 0.25f);
  EXPECT_EQ(rows[1]["calibration_revision"], 2u);
  EXPECT_EQ(rows[2]["confirmed_cut"], true);
  EXPECT_GT(rows[2]["center"].get<float>(), rows[1]["center"].get<float>());
  EXPECT_LT(rows[2]["center"].get<float>(), first_center + 1.0f);
  EXPECT_EQ(rows[2]["calibration_revision"], 3u);
  for (std::size_t index = 3u; index < 10u; ++index) {
    EXPECT_GT(rows[index]["center"].get<float>(), rows[index-1u]["center"].get<float>());
    EXPECT_LT(rows[index]["center"].get<float>(), first_center + 1.0f);
    EXPECT_EQ(rows[index]["calibration_revision"], index + 1u);
  }
  EXPECT_EQ(rows[10]["frame_valid"], false);
  EXPECT_FLOAT_EQ(rows[10]["center"].get<float>(), rows[9]["center"].get<float>());
  EXPECT_EQ(rows[11]["frame_valid"], true);
  EXPECT_FLOAT_EQ(rows[11]["center"].get<float>(), rows[9]["center"].get<float>());
  EXPECT_EQ(rows[11]["cut_attribution"], "none");
}

TEST(DepthCoordinateV2GpuTest, CoordinatePassReplayAuthenticatesAdaptiveStateAndRecoversExactly) {
  namespace fs = std::filesystem;
  namespace v2 = models::depth_coordinate_v2;

  static_assert(v2::constant_float_count == 16u);
  static_assert(v2::state_float_count == 28u);
  static_assert(v2::convergence_curve_default == 0.0f);

  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());
  temporary_tree_t tree;

  const fs::path contract_source = fs::path(SUNSHINE_SOURCE_DIR) /
    "tools/sbsbench/contracts/depth-coordinate-v2-v1.json";
  const std::string contract_bytes = read_bytes(contract_source);
  ASSERT_FALSE(contract_bytes.empty());
  const fs::path contract_copy = tree.path / "contracts/depth-coordinate-v2-v1.json";
  ASSERT_TRUE(write_bytes(contract_copy, contract_bytes));

  const auto &calibration = v2::model_calibrations.front();
  const auto shape_it = std::find_if(
    v2::model_calibrated_shapes.begin(),
    v2::model_calibrated_shapes.end(),
    [&](const auto &shape) {
      return shape.calibration_id == calibration.calibration_id;
    });
  ASSERT_NE(shape_it, v2::model_calibrated_shapes.end());
  const std::uint32_t width = shape_it->width;
  const std::uint32_t height = shape_it->height;
  const std::size_t element_count = static_cast<std::size_t>(width) * height;

  std::vector<float> base(element_count);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      base[static_cast<std::size_t>(y) * width + x] =
        -0.35f + 0.70f * static_cast<float>(x) / static_cast<float>(width - 1u) +
        ((y & 1u) ? 0.03f : -0.03f);
    }
  }

  std::vector<std::vector<float>> raw_fields;
  raw_fields.push_back(base);  // acquire
  raw_fields.push_back(base);  // same shot, transient extreme
  raw_fields.back().back() = 1000.0f;
  raw_fields.push_back(base);  // same shot, an outlier must not rescale this field
  const std::size_t foreground_count = element_count / 4u;
  raw_fields.emplace_back(element_count, 0.0f);  // confirmed cut, broad foreground
  std::fill_n(raw_fields.back().begin(), foreground_count, 20.0f);
  raw_fields.push_back(base);  // one non-finite texel invalidates the authenticated field
  raw_fields.back()[element_count / 2u] = std::numeric_limits<float>::quiet_NaN();
  raw_fields.emplace_back(element_count, 2.0f);  // finite but collapsed
  raw_fields.push_back(base);  // retained camera resumes without a gauge jump
  for (float &value : raw_fields.back()) {
    value += 0.5f;
  }
  raw_fields.push_back(base);  // the same one-NaN field on a cut cannot seed the new camera
  raw_fields.back()[element_count / 3u] = std::numeric_limits<float>::quiet_NaN();
  const std::size_t ramp_foreground_count = element_count * 18u / 100u;
  raw_fields.emplace_back(element_count, 0.0f);  // invalid+cut retained; next valid re-arms
  std::fill_n(raw_fields.back().begin(), ramp_foreground_count, 20.0f);

  nlohmann::ordered_json frames = nlohmann::ordered_json::array();
  for (std::size_t index = 0; index < raw_fields.size(); ++index) {
    const std::string frame_id = "0000" + std::to_string(index + 1u);
    const std::string raw_file = "raw_" + frame_id + ".f32";
    const std::string bytes = float_bytes(raw_fields[index]);
    ASSERT_TRUE(write_bytes(tree.path / raw_file, bytes));
    frames.push_back({
      {"frame_id", frame_id},
      {"raw_file", raw_file},
      {"raw_sha256", sha256_hex(bytes)},
      {"hard_cut_count", index >= 7u ? 2u : (index >= 3u ? 1u : 0u)},
      {"hard_cut_pulse", false},
    });
  }

  constexpr float pop_strength = 2.0f;
  const float requested_gain = pop_strength * v2::gain_per_pop;
  auto manifest = make_replay_manifest(
    calibration,
    sha256_hex(contract_bytes),
    width,
    height,
    frames,
    pop_strength
  );
  const fs::path manifest_path = tree.path / "manifest.json";
  const std::string manifest_bytes = manifest.dump(2) + "\n";
  ASSERT_TRUE(write_bytes(manifest_path, manifest_bytes));

  std::string error;
  auto replay = sbs_bench::depth_coordinate_v2_gpu_replay::create(
    warp.device.Get(), warp.context.Get(), manifest_path, error);
  ASSERT_NE(replay, nullptr) << error;
  ASSERT_EQ(replay->frame_count(), raw_fields.size());
  EXPECT_EQ(replay->width(), width);
  EXPECT_EQ(replay->height(), height);
  EXPECT_EQ(replay->manifest_sha256(), sha256_hex(manifest_bytes));
  // Same contract tag and otherwise-valid fixed calibration, but the center was changed without
  // resealing its integrity word. The state resolver must classify this center-only corruption
  // as invalid and replace it rather than
  // retaining a plausible same-tag pseudo-camera indefinitely.
  std::vector<std::uint32_t> corrupt_same_tag(
    v2::state_initial_words.begin(), v2::state_initial_words.end());
  corrupt_same_tag[v2::center] = std::bit_cast<std::uint32_t>(123.0f);
  corrupt_same_tag[v2::inverse_scale] = std::bit_cast<std::uint32_t>(
    1.0f / calibration.raw_coordinate_scale);
  corrupt_same_tag[v2::calibration_revision] = 7u;
  corrupt_same_tag[v2::frame_valid] = std::bit_cast<std::uint32_t>(1.0f);
  ASSERT_NE(
    corrupt_same_tag[v2::camera_center_integrity_bits],
    v2::camera_center_integrity_for_words(
      corrupt_same_tag[v2::center],
      corrupt_same_tag[v2::inverse_scale],
      corrupt_same_tag[v2::convergence_curve],
      corrupt_same_tag[v2::calibration_revision]
    )
  );
  ASSERT_TRUE(replay->overwrite_state_for_testing(corrupt_same_tag, error)) << error;

  std::vector<sbs_bench::depth_coordinate_v2_gpu_frame> outputs(raw_fields.size());
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (index == 2u) {
      // Exercise the independent unknown-contract recovery path on a field equal to the original
      // acquisition. Successful dispatch must restore the authenticated tag/camera atomically.
      auto unknown_contract = corrupt_same_tag;
      unknown_contract[v2::contract_tag_bits] = v2::contract_tag ^ 0x00000001u;
      ASSERT_TRUE(replay->overwrite_state_for_testing(unknown_contract, error)) << error;
    }
    const std::string frame_id = "0000" + std::to_string(index + 1u);
    ASSERT_TRUE(replay->dispatch(
      index,
      frame_id,
      test_source_sha256(),
      outputs[index],
      error
    )) << error;
    ASSERT_EQ(outputs[index].canonical_values.size(), element_count);
    ASSERT_EQ(outputs[index].candidate_parallax_values.size(), element_count);
    ASSERT_EQ(outputs[index].vertical_majorant_values.size(), element_count);
    ASSERT_EQ(outputs[index].vertical_conditioned_values.size(), element_count);
    ASSERT_EQ(outputs[index].encoded_parallax_values.size(), element_count);
    EXPECT_FALSE(outputs[index].vertical_majorant_sha256.empty());
    EXPECT_FALSE(outputs[index].vertical_conditioned_sha256.empty());
    EXPECT_GE(outputs[index].encoded_minimum, 0.0f);
    EXPECT_LE(outputs[index].encoded_maximum, 1.0f);
    EXPECT_LE(outputs[index].maximum_absolute_source_u,
              v2::direct_container_limit + 2.0e-7f);
  }
  EXPECT_LT(outputs[0].order_minimum, outputs[0].order_maximum);
  EXPECT_FLOAT_EQ(outputs[4].order_minimum, 0.0f);
  EXPECT_FLOAT_EQ(outputs[4].order_maximum, 0.0f);
  EXPECT_FLOAT_EQ(outputs[5].encoded_minimum, 0.5f);
  EXPECT_FLOAT_EQ(outputs[5].encoded_maximum, 0.5f);

  const fs::path trace_path = tree.path / "depth_coordinate_v2_state_trace.json";
  ASSERT_TRUE(replay->write_state_trace(trace_path, error)) << error;
  const auto trace = nlohmann::ordered_json::parse(read_bytes(trace_path));
  ASSERT_EQ(trace.at("schema"), sbs_bench::depth_coordinate_v2_state_trace_schema);
  ASSERT_EQ(trace.at("frames").size(), raw_fields.size());
  ASSERT_EQ(trace.at("frame_fields").size(), 45u);
  EXPECT_EQ(trace["producer"]["authority"],
            "authenticated-raw-depth-plus-eight-v2-compute-shaders-persistent-gpu-state-v11");
  EXPECT_EQ(trace["producer"]["tensor_shape"]["width"], replay->width());
  EXPECT_EQ(trace["producer"]["tensor_shape"]["height"], replay->height());
  ASSERT_EQ(trace["producer"]["shader_sequence"].size(), 8u);
  EXPECT_EQ(trace["producer"]["shader_sequence"][4],
            "depth_coordinate_v2_state_resolve_cs.hlsl");
  EXPECT_EQ(trace["producer"]["shader_sequence"][6],
            "depth_coordinate_v2_vertical_limit_cs.hlsl");
  EXPECT_EQ(trace["producer"]["contract_canonical_sha256"],
            v2::contract_canonical_sha256);

  const auto &rows = trace.at("frames");
  // Frame three deliberately injects a foreign contract. Its counters are not trusted: acquire
  // revision one again, then advance only on usable observations; invalid frames hold it.
  const std::array<std::uint32_t, 9u> revisions {1u, 2u, 1u, 2u, 2u, 2u, 3u, 3u, 4u};
  for (std::size_t index = 0u; index < revisions.size(); ++index) {
    EXPECT_EQ(rows[index]["calibration_revision"], revisions[index]);
    EXPECT_FLOAT_EQ(rows[index]["convergence_curve"].get<float>(), 0.0f);
    if (rows[index]["camera_valid"].get<bool>()) {
      EXPECT_GE(rows[index]["latched_scale"].get<float>(), calibration.raw_coordinate_scale - 2.0e-6f);
    }
  }
  EXPECT_EQ(rows[3]["confirmed_cut"], true);
  EXPECT_EQ(rows[3]["confirmed_cut_count"], 1u);
  const float first_center = rows[0]["center"].get<float>();
  EXPECT_GT(rows[1]["center"].get<float>(), first_center);
  EXPECT_LT(rows[3]["center"].get<float>(), rows[3]["observed_mean"].get<float>());

  // The ABI field remains present but the source-U container is pointwise and stateless.
  for (const auto &row : rows) {
    EXPECT_FLOAT_EQ(row["requested_gain"].get<float>(), requested_gain);
    EXPECT_FLOAT_EQ(row["container_scale"].get<float>(), 1.0f);
    if (row["frame_valid"].get<bool>()) {
      EXPECT_FLOAT_EQ(row["effective_gain"].get<float>(), requested_gain);
      EXPECT_LE(row["pre_limiter_max_abs_source_u"].get<float>(),
                v2::direct_container_limit + 2.0e-7f);
    }
  }
  EXPECT_EQ(rows[4]["input_valid"], false);
  EXPECT_EQ(rows[4]["collapsed"], false);
  EXPECT_EQ(rows[4]["frame_valid"], false);
  EXPECT_EQ(rows[4]["camera_valid"], true);
  EXPECT_FLOAT_EQ(outputs[4].encoded_minimum, 0.5f);
  EXPECT_FLOAT_EQ(outputs[4].encoded_maximum, 0.5f);
  EXPECT_NEAR(rows[4]["center"].get<float>(), rows[3]["center"].get<float>(), 2.0e-6f);
  EXPECT_FLOAT_EQ(rows[4]["effective_gain"].get<float>(), 0.0f);
  EXPECT_EQ(rows[5]["input_valid"], true);
  EXPECT_EQ(rows[5]["collapsed"], true);
  EXPECT_EQ(rows[5]["frame_valid"], false);
  EXPECT_EQ(rows[5]["camera_valid"], true);
  EXPECT_FLOAT_EQ(rows[5]["effective_gain"].get<float>(), 0.0f);
  EXPECT_EQ(rows[6]["frame_valid"], true);
  EXPECT_EQ(rows[6]["camera_valid"], true);
  EXPECT_NEAR(rows[6]["center"].get<float>(), rows[3]["center"].get<float>(), 2.0e-5f);
  EXPECT_EQ(rows[7]["frame_valid"], false);
  EXPECT_EQ(rows[7]["camera_valid"], true);
  EXPECT_FLOAT_EQ(rows[7]["center"].get<float>(), rows[6]["center"].get<float>());
  EXPECT_FLOAT_EQ(outputs[7].encoded_minimum, 0.5f);
  EXPECT_FLOAT_EQ(outputs[7].encoded_maximum, 0.5f);
  EXPECT_EQ(rows[8]["frame_valid"], true);
  EXPECT_EQ(rows[8]["camera_valid"], true);
  EXPECT_FLOAT_EQ(rows[8]["center"].get<float>(), rows[6]["center"].get<float>());

  const auto expect_rejected = [&](const nlohmann::ordered_json &candidate,
                                   const std::string_view name) {
    const fs::path path = tree.path / (std::string(name) + ".json");
    EXPECT_TRUE(write_bytes(path, candidate.dump(2) + "\n"));
    std::string candidate_error;
    auto invalid = sbs_bench::depth_coordinate_v2_gpu_replay::create(
      warp.device.Get(), warp.context.Get(), path, candidate_error);
    EXPECT_EQ(invalid, nullptr) << name;
    EXPECT_FALSE(candidate_error.empty()) << name;
  };

  auto tampered_contract = nlohmann::ordered_json::parse(contract_bytes);
  tampered_contract["calibrated_defaults"]["adaptive_time_constant_seconds"] = 0.151f;
  const std::string tampered_bytes = tampered_contract.dump(2) + "\n";
  ASSERT_TRUE(write_bytes(contract_copy, tampered_bytes));
  auto invalid = manifest;
  invalid["calibration_contract"]["sha256"] = sha256_hex(tampered_bytes);
  expect_rejected(invalid, "self-attested-noncanonical-contract");
  ASSERT_TRUE(write_bytes(contract_copy, contract_bytes));

  invalid = manifest;
  invalid["calibration_contract"]["schema"] =
    static_cast<std::uint64_t>(v2::contract_schema) + 0x100000000ull;
  expect_rejected(invalid, "overflow-calibration-schema");
  invalid = manifest;
  auto mismatched_preprocess = std::string {
    calibration.preprocess.source_closure_sha256
  };
  ASSERT_FALSE(mismatched_preprocess.empty());
  mismatched_preprocess.front() = mismatched_preprocess.front() == '0' ? '1' : '0';
  invalid["model_identity"]["preprocess_source_closure_sha256"] =
    mismatched_preprocess;
  expect_rejected(invalid, "mismatched-preprocess-source-closure");
  invalid = manifest;
  invalid["frames"][0]["hard_cut_count"] = 0xfffffffeu;
  expect_rejected(invalid, "reserved-hard-cut-generation");
  invalid = manifest;
  invalid["frames"][0]["hard_cut_count"] = 0x100000000ull;
  expect_rejected(invalid, "overflow-hard-cut-generation");
  invalid = manifest;
  invalid["mapping_config"]["joint_plane_mode"] = 0u;
  expect_rejected(invalid, "retired-policy-zero");
  invalid = manifest;
  invalid["source_color_mode"] = 0u;
  expect_rejected(invalid, "retired-source-color-mode");

  auto identity_replay = sbs_bench::depth_coordinate_v2_gpu_replay::create(
    warp.device.Get(), warp.context.Get(), manifest_path, error);
  ASSERT_NE(identity_replay, nullptr) << error;
  sbs_bench::depth_coordinate_v2_gpu_frame identity_output;
  EXPECT_FALSE(identity_replay->dispatch(
    0u,
    "00001",
    std::string(64u, '0'),
    identity_output,
    error
  ));
  EXPECT_FALSE(error.empty());
  EXPECT_TRUE(identity_replay->dispatch(
    0u,
    "00001",
    test_source_sha256(),
    identity_output,
    error
  )) << error;
}

// Frozen pre-fusion shaders are test-only D3D11 oracles. They deliberately retain both original
// dispatches and arithmetic so a future production edit cannot silently rewrite its own oracle.
TEST(DepthCoordinateV2GpuTest, SharedRawScanPreservesIndependentHistogramAuthorities) {
  namespace fs = std::filesystem;
  namespace v2 = models::depth_coordinate_v2;
  namespace gpu = models::host_sbs_v2_gpu;
  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());
  const fs::path shader_root = fs::path(SUNSHINE_SOURCE_DIR) /
    "src_assets/windows/assets/shaders/directx";
  const std::string reference_histogram = R"HIST_REF(
// Mode-3 robust raw-depth histogram. Every reduction group overwrites its 256 bins;
// no clear, global atomic accumulator, CPU readback or reuse dispatch is required.
// Unlike the private cut-normalization histogram, all finite signed eligible values count.
StructuredBuffer<float> InputBuffer : register(t0);
Texture2D<uint> TensorExclusion : register(t1);
StructuredBuffer<float4> FrameStats : register(t2);
RWStructuredBuffer<uint4> HistogramPartials : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2_contract.generated.hlsl"

groupshared uint histogram[256];

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID,
          uint3 gid : SV_GroupID) {
    histogram[tid.x] = 0u;
    GroupMemoryBarrierWithGroupSync();
    float4 frame0 = FrameStats[V2_FRAME_STATS_VECTOR_MEAN];
    float4 frame1 = FrameStats[V2_FRAME_STATS_VECTOR_VALID_COUNT];
    float minimum = V2_FRAME_STATS_MINIMUM(frame0);
    float range = V2_FRAME_STATS_MAXIMUM(frame0) - minimum;
    bool valid = v2_joint_plane_mode == 3u && V2_FRAME_STATS_VALID(frame1) == 1.0f &&
        !isnan(range) && !isinf(range) && range >= 0.0f;
    float inverse_range = range > 0.0f ? 256.0f / range : 0.0f;
    valid = valid && !isnan(inverse_range) && !isinf(inverse_range);
    uint position_y = dtid.x / target_w;
    uint position_x = dtid.x - position_y * target_w;
    uint stride_y = reduce_threads / target_w;
    uint stride_x = reduce_threads - stride_y * target_w;
    [loop] for (uint index = dtid.x; index < target_w * target_h; index += reduce_threads) {
        if (valid && TensorExclusion[uint2(position_x, position_y)] == 0u) {
            float value = InputBuffer[index];
            if (!isnan(value) && !isinf(value)) {
                uint bin = range > 0.0f ?
                    min((uint)max((value - minimum) * inverse_range, 0.0f), 255u) : 0u;
                InterlockedAdd(histogram[bin], 1u);
            }
        }
        position_x += stride_x;
        position_y += stride_y;
        if (position_x >= target_w) {
            position_x -= target_w;
            position_y++;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // One lane owns one whole uint4 record; concurrent component writes cannot race.
    if (tid.x < 64u) {
        uint bin = tid.x * 4u;
        HistogramPartials[gid.x * 64u + tid.x] = uint4(
            histogram[bin], histogram[bin + 1u], histogram[bin + 2u], histogram[bin + 3u]);
    }
}
)HIST_REF";
  const std::string reference_quantiles = R"QUANT_REF(
// Resolve GPU histogram into the mode-3 quantile tail. P05 uses the lower crossing-bin
// edge and P95 the upper edge, clipped to the true range. These are FP32 256-bin bounds,
// not exact interpolated percentiles. All original Welford statistics remain untouched.
StructuredBuffer<uint4> HistogramPartials : register(t0);
RWStructuredBuffer<float4> FrameStats : register(u0);

#include "include/depth_constants.hlsl"
#include "include/depth_coordinate_v2_contract.generated.hlsl"

groupshared uint histogram[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
    uint count = 0u;
    [loop] for (uint group = 0u; group < max(reduce_threads / 256u, 1u); ++group)
        count += HistogramPartials[group * 64u + tid.x / 4u][tid.x % 4u];
    histogram[tid.x] = count;
    GroupMemoryBarrierWithGroupSync();
    if (tid.x != 0u) return;
    float4 frame0 = FrameStats[V2_FRAME_STATS_VECTOR_MEAN];
    float4 frame1 = FrameStats[V2_FRAME_STATS_VECTOR_VALID_COUNT];
    float minimum = V2_FRAME_STATS_MINIMUM(frame0);
    float maximum = V2_FRAME_STATS_MAXIMUM(frame0);
    float range = maximum - minimum;
    float bin_width = range / v2_host_percentile_bin_count;
    uint total = 0u;
    bool low_found = false, high_found = false;
    float low = 0.0f, high = 0.0f;
    [loop] for (uint bin = 0u; bin < 256u; ++bin) {
        total += histogram[bin];
        if (!low_found && (float)total >=
            v2_host_percentile_low * V2_FRAME_STATS_VALID_COUNT(frame1)) {
            low = clamp(minimum + float(bin) * bin_width, minimum, maximum);
            low_found = true;
        }
        if (!high_found && (float)total >=
            v2_host_percentile_high * V2_FRAME_STATS_VALID_COUNT(frame1)) {
            high = clamp(minimum + float(bin + 1u) * bin_width, minimum, maximum);
            high_found = true;
        }
    }
    bool valid = v2_joint_plane_mode == 3u && V2_FRAME_STATS_VALID(frame1) == 1.0f &&
        !isnan(bin_width) && !isinf(bin_width) && bin_width >= 0.0f &&
        total > 0u && (float)total == V2_FRAME_STATS_VALID_COUNT(frame1) &&
        low_found && high_found && low <= high;
    float4 tail = 0.0f;
    if (valid) {
        V2_FRAME_STATS_PERCENTILE_LOW(tail) = low;
        V2_FRAME_STATS_PERCENTILE_HIGH(tail) = high;
        V2_FRAME_STATS_PERCENTILE_VALID(tail) = 1.0f;
        V2_FRAME_STATS_PERCENTILE_BIN_WIDTH(tail) = bin_width;
    }
    FrameStats[V2_FRAME_STATS_VECTOR_PERCENTILE_LOW] = tail;
}
)QUANT_REF";
  const auto compile = [&](const std::string_view filename,
                           const std::string_view reference = {}) {
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> diagnostics;
    const auto path = shader_root / filename;
    const auto status = reference.empty() ?
      D3DCompileFromFile(path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0u, &bytecode, &diagnostics) :
      D3DCompile(reference.data(), reference.size(), path.string().c_str(), nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0u, &bytecode, &diagnostics);
    EXPECT_TRUE(SUCCEEDED(status)) << filename << ": " <<
      (diagnostics ? static_cast<const char *>(diagnostics->GetBufferPointer()) : "");
    ComPtr<ID3D11ComputeShader> shader;
    if (bytecode) {
      EXPECT_TRUE(SUCCEEDED(warp.device->CreateComputeShader(
        bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &shader)));
    }
    return shader;
  };
  const auto moments = compile("depth_coordinate_v2_moments_cs.hlsl");
  const auto frame_resolve = compile("depth_coordinate_v2_frame_resolve_cs.hlsl");
  const auto histogram = compile("depth_coordinate_v2_histogram_cs.hlsl");
  const auto quantiles = compile("depth_coordinate_v2_quantiles_cs.hlsl");
  const auto old_histogram = compile("depth_coordinate_v2_histogram_cs.hlsl", reference_histogram);
  const auto old_quantiles = compile("depth_coordinate_v2_quantiles_cs.hlsl", reference_quantiles);
  // Frozen pre-fusion normalization arithmetic remains an independent test-only oracle.
  const auto normalization_histogram = compile("depth_coordinate_v2_histogram_cs.hlsl",
    read_bytes(fs::path(SUNSHINE_SOURCE_DIR) /
      "tests/fixtures/host_sbs_cut_normalization_histogram_reference_cs.hlsl"));
  ASSERT_TRUE(moments && frame_resolve && histogram && quantiles && old_histogram &&
    old_quantiles && normalization_histogram);

  struct buffer_t {
    ComPtr<ID3D11Buffer> buffer;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
  };
  const auto structured = [&](const UINT stride, const UINT count) {
    buffer_t result;
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = stride * count;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = stride;
    if (SUCCEEDED(warp.device->CreateBuffer(&desc, nullptr, &result.buffer))) {
      EXPECT_TRUE(SUCCEEDED(warp.device->CreateShaderResourceView(
        result.buffer.Get(), nullptr, &result.srv)));
      EXPECT_TRUE(SUCCEEDED(warp.device->CreateUnorderedAccessView(
        result.buffer.Get(), nullptr, &result.uav)));
    }
    return result;
  };
  const auto read_words = [&](ID3D11Buffer *source) {
    std::vector<std::uint32_t> result;
    D3D11_BUFFER_DESC desc {};
    source->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0u;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0u;
    desc.StructureByteStride = 0u;
    ComPtr<ID3D11Buffer> staging;
    if (FAILED(warp.device->CreateBuffer(&desc, nullptr, &staging))) {
      return result;
    }
    warp.context->CopyResource(staging.Get(), source);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(warp.context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
      return result;
    }
    const auto *words = static_cast<const std::uint32_t *>(mapped.pData);
    result.assign(words, words + desc.ByteWidth / sizeof(std::uint32_t));
    warp.context->Unmap(staging.Get(), 0u);
    return result;
  };
  const auto constants_buffer = [&](const void *data, const UINT bytes) {
    ComPtr<ID3D11Buffer> result;
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = bytes;
    desc.Usage = D3D11_USAGE_IMMUTABLE;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA initial {data, 0u, 0u};
    EXPECT_TRUE(SUCCEEDED(warp.device->CreateBuffer(&desc, &initial, &result)));
    return result;
  };
  const auto unbind = [&] {
    ID3D11ShaderResourceView *inputs[3] = {};
    ID3D11UnorderedAccessView *outputs[2] = {};
    warp.context->CSSetShaderResources(0u, 3u, inputs);
    warp.context->CSSetUnorderedAccessViews(0u, 2u, outputs, nullptr);
  };

  // The first grid has more lanes than texels; the second crosses rows on a grid-stride visit.
  for (const auto dimensions : {std::array<UINT, 3> {37u, 19u, 5u},
                               std::array<UINT, 3> {67u, 31u, 3u}}) {
    const UINT width = dimensions[0], height = dimensions[1], groups = dimensions[2];
    const UINT count = width * height;
    std::array<std::uint32_t, 16> common_words {};
    common_words[0] = width;
    common_words[1] = height;
    common_words[5] = groups * 256u;
    common_words[11] = width;
    common_words[12] = height;
    auto common_constants = constants_buffer(common_words.data(), sizeof(common_words));
    v2::constants_t v2_words {};
    v2_words.joint_plane_mode = 3u;
    auto v2_constants = constants_buffer(&v2_words, sizeof(v2_words));
    ASSERT_TRUE(common_constants && v2_constants);
    auto raw = structured(sizeof(float), count);
    auto partials = structured(sizeof(std::uint32_t) * 4u, groups * 3u);
    auto shared_bins = structured(sizeof(std::uint32_t) * 4u, groups * 128u);
    auto reference_bins = structured(sizeof(std::uint32_t) * 4u, groups * 64u);
    auto shared_frame = structured(sizeof(float) * 4u,
      static_cast<UINT>(v2::frame_stats_vector_count));
    auto reference_frame = structured(sizeof(float) * 4u,
      static_cast<UINT>(v2::frame_stats_vector_count));
    auto shared_normalization = structured(sizeof(std::uint32_t), 256u);
    auto reference_normalization = structured(sizeof(std::uint32_t), 256u);
    ASSERT_TRUE(raw.srv && partials.uav && shared_bins.uav && reference_bins.uav &&
      shared_frame.uav && reference_frame.uav && shared_normalization.uav &&
      reference_normalization.uav);
    buffer_t minmax;
    D3D11_BUFFER_DESC minmax_desc {};
    minmax_desc.ByteWidth = sizeof(std::uint32_t) * 4u;
    minmax_desc.Usage = D3D11_USAGE_DEFAULT;
    minmax_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    minmax_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    ASSERT_TRUE(SUCCEEDED(warp.device->CreateBuffer(&minmax_desc, nullptr, &minmax.buffer)));
    D3D11_UNORDERED_ACCESS_VIEW_DESC minmax_view {};
    minmax_view.Format = DXGI_FORMAT_R32_TYPELESS;
    minmax_view.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    minmax_view.Buffer.NumElements = 4u;
    minmax_view.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    ASSERT_TRUE(SUCCEEDED(warp.device->CreateUnorderedAccessView(
      minmax.buffer.Get(), &minmax_view, &minmax.uav)));
    D3D11_TEXTURE2D_DESC exclusion_desc {};
    exclusion_desc.Width = width;
    exclusion_desc.Height = height;
    exclusion_desc.MipLevels = 1u;
    exclusion_desc.ArraySize = 1u;
    exclusion_desc.Format = DXGI_FORMAT_R32_UINT;
    exclusion_desc.SampleDesc.Count = 1u;
    exclusion_desc.Usage = D3D11_USAGE_DEFAULT;
    exclusion_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    ComPtr<ID3D11Texture2D> exclusion;
    ComPtr<ID3D11ShaderResourceView> exclusion_srv;
    ASSERT_TRUE(SUCCEEDED(warp.device->CreateTexture2D(&exclusion_desc, nullptr, &exclusion)));
    ASSERT_TRUE(SUCCEEDED(warp.device->CreateShaderResourceView(
      exclusion.Get(), nullptr, &exclusion_srv)));
    gpu::moments_frame_command_t command {
      .constants = {.depth = common_constants.Get(), .coordinate_v2 = v2_constants.Get()},
      .moments_shader = moments.Get(),
      .frame_resolve_shader = frame_resolve.Get(),
      .raw_depth = raw.srv.Get(),
      .tensor_exclusion = exclusion_srv.Get(),
      .partials_output = partials.uav.Get(),
      .partials = partials.srv.Get(),
      .frame_stats_output = shared_frame.uav.Get(),
      .minmax_raw_output = minmax.uav.Get(),
      .moments_dispatch = gpu::dispatch_command_t::direct(groups, 1u, 1u),
      .frame_resolve_dispatch = gpu::dispatch_command_t::direct(1u, 1u, 1u),
            .histogram_shader = histogram.Get(),
      .quantile_shader = quantiles.Get(),
      .frame_stats = shared_frame.srv.Get(),
      .histogram_output = shared_bins.uav.Get(),
      .histogram = shared_bins.srv.Get(),
      .normalization_histogram_output = shared_normalization.uav.Get(),
    };
    const auto missing_normalization = [&] {
      auto missing = command;
      missing.normalization_histogram_output = nullptr;
      return gpu::record_moments_frame(warp.context.Get(), missing);
    };
    EXPECT_FALSE(missing_normalization());
    for (const std::string_view sample : {"positive_crossings", "finite_negative",
      "mixed_signed_zero", "negative_zero_only", "collapsed", "tiny_span", "subnormal",
      "large_atoms_outliers", "nan", "positive_infinity", "negative_infinity",
      "all_nan", "all_excluded", "roi_exclusion", "overflowing_moments"}) {
      SCOPED_TRACE(std::string(sample) + " " + std::to_string(width) + "x" + std::to_string(height));
      std::vector<float> values(count);
      std::vector<std::uint32_t> exclusions(count, 0u);
      for (UINT index = 0u; index < count; ++index) {
        values[index] = 1.0f + float(index % 257u) / 32.0f;
        if (sample == "finite_negative") values[index] -= 6.0f;
        if (sample == "mixed_signed_zero") values[index] = index % 2u ? -0.0f : 0.0f;
        if (sample == "negative_zero_only") values[index] = -0.0f;
        if (sample == "collapsed") values[index] = 3.0f;
        if (sample == "tiny_span") values[index] = float(index % 17u) * 1.0e-15f;
        if (sample == "subnormal") values[index] = float(index % 17u) * 1.0e-39f;
        if (sample == "large_atoms_outliers") {
          values[index] = index % 100u == 0u ? 1000.0f : (index % 3u ? 1.0f : 7.0f);
        }
        if (sample == "all_nan") values[index] = std::numeric_limits<float>::quiet_NaN();
        if (sample == "all_excluded") exclusions[index] = 1u;
        if (sample == "roi_exclusion") {
          const UINT x = index % width, y = index / width;
          if (x < 3u || y < 2u || x >= width - 4u || y >= height - 3u) {
            exclusions[index] = 1u;
            values[index] = std::numeric_limits<float>::quiet_NaN();
          }
        }
        if (sample == "overflowing_moments") {
          values[index] = index % 2u ? 1.0e30f : 2.0e30f;
        }
      }
      if (sample == "nan") values[count / 2u] = std::numeric_limits<float>::quiet_NaN();
      if (sample == "positive_infinity") values[count / 2u] = std::numeric_limits<float>::infinity();
      if (sample == "negative_infinity") values[count / 2u] = -std::numeric_limits<float>::infinity();
      warp.context->UpdateSubresource(raw.buffer.Get(), 0u, nullptr, values.data(), 0u, 0u);
      warp.context->UpdateSubresource(exclusion.Get(), 0u, nullptr, exclusions.data(),
        width * sizeof(std::uint32_t), 0u);
      constexpr std::array<UINT, 4> poison {0xcdcdcdcdu, 0xcdcdcdcdu, 0xcdcdcdcdu, 0xcdcdcdcdu};
      constexpr std::array<UINT, 4> zero {};
      warp.context->ClearUnorderedAccessViewUint(shared_bins.uav.Get(), poison.data());
      warp.context->ClearUnorderedAccessViewUint(shared_normalization.uav.Get(), poison.data());
      warp.context->ClearUnorderedAccessViewUint(reference_bins.uav.Get(), poison.data());
      warp.context->ClearUnorderedAccessViewUint(reference_normalization.uav.Get(), zero.data());
      ASSERT_TRUE(gpu::record_moments_frame(warp.context.Get(), command));
      warp.context->CopyResource(reference_frame.buffer.Get(), shared_frame.buffer.Get());

      ID3D11ShaderResourceView *histogram_inputs[] = {
        raw.srv.Get(), exclusion_srv.Get(), reference_frame.srv.Get(),
      };
      warp.context->CSSetShader(old_histogram.Get(), nullptr, 0u);
      warp.context->CSSetShaderResources(0u, 3u, histogram_inputs);
      warp.context->CSSetUnorderedAccessViews(0u, 1u, reference_bins.uav.GetAddressOf(), nullptr);
      warp.context->Dispatch(groups, 1u, 1u);
      unbind();
      warp.context->CSSetShader(old_quantiles.Get(), nullptr, 0u);
      warp.context->CSSetShaderResources(0u, 1u, reference_bins.srv.GetAddressOf());
      warp.context->CSSetUnorderedAccessViews(0u, 1u, reference_frame.uav.GetAddressOf(), nullptr);
      warp.context->Dispatch(1u, 1u, 1u);
      unbind();
      ID3D11UnorderedAccessView *normalization_outputs[] = {
        reference_normalization.uav.Get(), minmax.uav.Get(),
      };
      warp.context->CSSetShader(normalization_histogram.Get(), nullptr, 0u);
      warp.context->CSSetShaderResources(0u, 2u, histogram_inputs);
      warp.context->CSSetUnorderedAccessViews(0u, 2u, normalization_outputs, nullptr);
      warp.context->Dispatch(groups, 1u, 1u);
      unbind();

      const auto actual_frame = read_words(shared_frame.buffer.Get());
      const auto expected_frame = read_words(reference_frame.buffer.Get());
      ASSERT_EQ(actual_frame.size(), v2::frame_stats_vector_count * 4u);
      EXPECT_EQ(actual_frame, expected_frame) << "FP32 moments and quantile tail changed";
      const auto actual_histogram = read_words(shared_bins.buffer.Get());
      const auto expected_histogram = read_words(reference_bins.buffer.Get());
      ASSERT_EQ(actual_histogram.size(), groups * 512u);
      ASSERT_EQ(expected_histogram.size(), groups * 256u);
      for (UINT group = 0u; group < groups; ++group) {
        for (UINT bin = 0u; bin < 256u; ++bin) {
          ASSERT_EQ(actual_histogram[group * 512u + bin],
            expected_histogram[group * 256u + bin]) << group << ":" << bin;
        }
      }
      const auto actual_normalization = read_words(shared_normalization.buffer.Get());
      const auto expected_normalization = read_words(reference_normalization.buffer.Get());
      ASSERT_EQ(actual_normalization.size(), 256u);
      EXPECT_EQ(actual_normalization, expected_normalization) << "P02/P98 population changed";
    }

    // Zero-group indirect work is the real reuse gate. No moments, quantile or histogram word moves.
    const auto held_frame = read_words(shared_frame.buffer.Get());
    const auto held_histogram = read_words(shared_bins.buffer.Get());
    const auto held_normalization = read_words(shared_normalization.buffer.Get());
    constexpr std::array<UINT, 6> no_infer {0u, 1u, 1u, 0u, 1u, 1u};
    D3D11_BUFFER_DESC indirect_desc {};
    indirect_desc.ByteWidth = sizeof(no_infer);
    // Match the production indirect-argument allocation; immutable, unbound buffers
    // are rejected by the D3D11 runtime before the zero-group dispatch can execute.
    indirect_desc.Usage = D3D11_USAGE_DEFAULT;
    indirect_desc.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
    D3D11_SUBRESOURCE_DATA indirect_data {no_infer.data(), 0u, 0u};
    ComPtr<ID3D11Buffer> indirect;
    ASSERT_TRUE(SUCCEEDED(warp.device->CreateBuffer(&indirect_desc, &indirect_data, &indirect)));
    command.moments_dispatch = gpu::dispatch_command_t::indirect(indirect.Get(), 0u);
    command.frame_resolve_dispatch = gpu::dispatch_command_t::indirect(indirect.Get(), 12u);
    ASSERT_TRUE(gpu::record_moments_frame(warp.context.Get(), command));
    EXPECT_EQ(read_words(shared_frame.buffer.Get()), held_frame);
    EXPECT_EQ(read_words(shared_bins.buffer.Get()), held_histogram);
    EXPECT_EQ(read_words(shared_normalization.buffer.Get()), held_normalization);
  }
}

TEST(DepthCoordinateV2GpuTest, ContinuousHostPlaneTracksAcrossCutsAndHoldsUnusableInput) {
  namespace fs = std::filesystem;
  namespace v2 = models::depth_coordinate_v2;
  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());
  temporary_tree_t tree;
  const std::string contract = read_bytes(fs::path(SUNSHINE_SOURCE_DIR) /
    "tools/sbsbench/contracts/depth-coordinate-v2-v1.json");
  ASSERT_FALSE(contract.empty());
  ASSERT_TRUE(write_bytes(tree.path / "contracts/depth-coordinate-v2-v1.json", contract));
  const auto &calibration = v2::model_calibrations.front();
  const auto shape = std::find_if(v2::model_calibrated_shapes.begin(),
    v2::model_calibrated_shapes.end(), [&](const auto &value) {
      return value.calibration_id == calibration.calibration_id;
    });
  ASSERT_NE(shape, v2::model_calibrated_shapes.end());
  const std::size_t count = static_cast<std::size_t>(shape->width) * shape->height;
  std::vector<std::vector<float>> fields(9u, std::vector<float>(count));
  constexpr std::array first_values {2.0f, 3.0f, 4.0f, 8.0f};
  constexpr std::array next_values {2.0f, 3.0f, 8.0f, 8.0f};
  for (std::size_t index = 0u; index < count; ++index) {
    const auto band = std::min(index * 4u / count, std::size_t {3u});
    fields[0][index] = first_values[band];
    fields[1][index] = next_values[band];
    fields[2][index] = 0.5f * fields[1][index] + 1.0f;
    fields[3][index] = 2.0f * fields[1][index] + 5.0f;
  }
  fields[4].assign(count, 3.0f);
  fields[5] = fields[1];
  fields[6] = fields[1];
  fields[6][count / 2u] = std::numeric_limits<float>::quiet_NaN();
  fields[7].assign(count, 3.0f);
  fields[8] = fields[3];

  nlohmann::ordered_json frames = nlohmann::ordered_json::array();
  for (std::size_t index = 0u; index < fields.size(); ++index) {
    const std::string id = "0000" + std::to_string(index + 1u);
    const std::string name = "raw_" + id + ".f32";
    const std::string bytes = float_bytes(fields[index]);
    ASSERT_TRUE(write_bytes(tree.path / name, bytes));
    frames.push_back({{"frame_id", id}, {"raw_file", name},
      {"raw_sha256", sha256_hex(bytes)}, {"hard_cut_count", index >= 7u ? 1u : 0u},
      {"hard_cut_pulse", false}, {"observation_timestamp_us", std::uint64_t(100000u + index * 100000u)}});
  }
  constexpr float pop = 1.75f;
  auto manifest = make_replay_manifest(calibration, sha256_hex(contract),
    shape->width, shape->height, frames, pop);
  manifest["mapping_config"]["joint_plane_mode"] = 3u;
  const auto path = tree.path / "manifest.json";
  ASSERT_TRUE(write_bytes(path, manifest.dump(2) + "\n"));
  std::string error;
  auto replay = sbs_bench::depth_coordinate_v2_gpu_replay::create(
    warp.device.Get(), warp.context.Get(), path, error);
  ASSERT_NE(replay, nullptr) << error;
  std::vector<sbs_bench::depth_coordinate_v2_gpu_frame> outputs(fields.size());
  for (std::size_t index = 0u; index < fields.size(); ++index) {
    ASSERT_TRUE(replay->dispatch(index, "0000" + std::to_string(index + 1u),
      test_source_sha256(), outputs[index], error)) << error;
  }
  const auto trace_path = tree.path / "trace.json";
  ASSERT_TRUE(replay->write_state_trace(trace_path, error)) << error;
  const auto trace = nlohmann::ordered_json::parse(read_bytes(trace_path));
  const auto &rows = trace.at("frames");
  ASSERT_EQ(rows.size(), fields.size());
  const std::array<std::uint32_t, 9> revisions {1u, 2u, 3u, 4u, 4u, 5u, 5u, 5u, 6u};
  for (std::size_t index = 0u; index < rows.size(); ++index) {
    EXPECT_EQ(rows[index]["joint_plane_mode"], 3u);
    EXPECT_EQ(rows[index]["calibration_revision"], revisions[index]);
    EXPECT_FLOAT_EQ(rows[index]["convergence_curve"].get<float>(), 0.0f);
    EXPECT_LE(rows[index]["inverse_scale"].get<float>(), 1.0f / calibration.raw_coordinate_scale);
    EXPECT_LE(outputs[index].maximum_absolute_source_u, v2::direct_container_limit + 2.0e-7f);
  }
  // Acquisition uses all finite content, and subsequent source-time observations are bounded.
  EXPECT_NEAR(rows[0]["center"].get<float>(), 4.25f, 2.0e-5f);
  EXPECT_GT(rows[1]["center"].get<float>(), rows[0]["center"].get<float>());
  EXPECT_LE(rows[1]["center"].get<float>() - rows[0]["center"].get<float>(),
    0.1f / rows[1]["inverse_scale"].get<float>() + 2.0e-5f);
  EXPECT_FALSE(rows[1]["confirmed_cut"].get<bool>());
  ASSERT_EQ(outputs[0].candidate_parallax_values.size(), count);
  const auto gain = v2::requested_gain_for_config(pop);
  // Observe actual native candidates across far/near raw samples; no CPU warp replica.
  for (const std::size_t index : {std::size_t(0u), count - 1u}) {
    const auto coordinate = outputs[0].canonical_values[index];
    EXPECT_NEAR(outputs[0].candidate_parallax_values[index], gain * coordinate, 2.0e-7f);
  }
  for (const auto index : {4u, 6u, 7u}) {
    EXPECT_FALSE(rows[index]["frame_valid"].get<bool>());
    EXPECT_TRUE(rows[index]["camera_valid"].get<bool>());
    EXPECT_FLOAT_EQ(rows[index]["center"].get<float>(), rows[index - 1u]["center"].get<float>());
    EXPECT_FLOAT_EQ(rows[index]["inverse_scale"].get<float>(), rows[index - 1u]["inverse_scale"].get<float>());
    EXPECT_FLOAT_EQ(outputs[index].encoded_minimum, 0.5f);
    EXPECT_FLOAT_EQ(outputs[index].encoded_maximum, 0.5f);
  }
  EXPECT_TRUE(rows[7]["confirmed_cut"].get<bool>());
  EXPECT_TRUE(rows[8]["frame_valid"].get<bool>());
  // First observation after unusable evidence rearms the clock without spending skipped time.
  EXPECT_FLOAT_EQ(rows[8]["center"].get<float>(), rows[7]["center"].get<float>());
}

TEST(DepthCoordinateV2GpuTest, JointPlaneModeTamperingRecoversAndUnknownModesAreRejected) {
  namespace fs = std::filesystem;
  namespace v2 = models::depth_coordinate_v2;
  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());
  temporary_tree_t tree;
  const std::string contract = read_bytes(fs::path(SUNSHINE_SOURCE_DIR) /
    "tools/sbsbench/contracts/depth-coordinate-v2-v1.json");
  ASSERT_FALSE(contract.empty());
  ASSERT_TRUE(write_bytes(tree.path / "contracts/depth-coordinate-v2-v1.json", contract));
  const auto &calibration = v2::model_calibrations.front();
  const auto shape = std::find_if(v2::model_calibrated_shapes.begin(),
    v2::model_calibrated_shapes.end(), [&](const auto &value) {
      return value.calibration_id == calibration.calibration_id;
    });
  ASSERT_NE(shape, v2::model_calibrated_shapes.end());
  const auto count = static_cast<std::size_t>(shape->width) * shape->height;
  std::vector<float> raw(count, 0.0f);
  std::fill_n(raw.begin(), count / 4u, 4.0f);
  const auto bytes = float_bytes(raw);
  constexpr std::array ids {"00001", "00002"};
  nlohmann::ordered_json frames = nlohmann::ordered_json::array();
  for (const auto id : ids) {
    const std::string raw_file = "raw_" + std::string(id) + ".f32";
    ASSERT_TRUE(write_bytes(tree.path / raw_file, bytes));
    frames.push_back({{"frame_id", id}, {"raw_file", raw_file},
      {"raw_sha256", sha256_hex(bytes)}, {"hard_cut_count", 0u}, {"hard_cut_pulse", false}});
  }
  auto manifest = make_replay_manifest(calibration, sha256_hex(contract),
    shape->width, shape->height, frames);
  manifest["mapping_config"]["joint_plane_mode"] = 3u;
  for (std::size_t index = 0u; index < manifest["frames"].size(); ++index) {
    manifest["frames"][index]["observation_timestamp_us"] = std::uint64_t(100000u + index * 100000u);
  }
  const auto path = tree.path / "manifest.json";
  ASSERT_TRUE(write_bytes(path, manifest.dump(2) + "\n"));
  std::string error;
  auto replay = sbs_bench::depth_coordinate_v2_gpu_replay::create(
    warp.device.Get(), warp.context.Get(), path, error);
  ASSERT_NE(replay, nullptr) << error;
  auto state = v2::state_initial_words;
  state[v2::center] = std::bit_cast<std::uint32_t>(123.0f);
  state[v2::inverse_scale] = std::bit_cast<std::uint32_t>(1.0f / calibration.raw_coordinate_scale);
  state[v2::calibration_revision] = 7u;
  state[v2::frame_valid] = std::bit_cast<std::uint32_t>(1.0f);
  state[v2::renderer_authorization_bits] = v2::contract_tag;
  state[v2::joint_plane_mode_bits] = 3u;
  state[v2::gain_last_observation_low] = 100000u;
  state[v2::gain_clock_armed] = 1u;
  state[v2::gain_seed_count] = 1u;
  state[v2::gain_target_zero] = state[v2::center];
  state[v2::gain_target_inverse_scale] = state[v2::inverse_scale];
  state[v2::gain_target_nearest] = std::bit_cast<std::uint32_t>(calibration.raw_coordinate_scale);
  state[v2::gain_display_limit] = std::bit_cast<std::uint32_t>(v2::direct_container_limit);
  state[v2::gain_seed_first_low] = state[v2::gain_seed_last_low] = 100000u;
  state[v2::gain_seed_mean_nearest] = state[v2::gain_target_nearest];
  state[v2::gain_seed_mean_zero] = state[v2::center];
  state[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_state_words(state);
  ASSERT_TRUE(v2::parallax_state_words_are_authenticated(state, calibration.raw_coordinate_scale, 3u));
  for (std::size_t index = 0u; index < 2u; ++index) {
    auto corrupt = state;
    corrupt[v2::joint_plane_mode_bits] = index == 0u ? 0u : 2u;
    ASSERT_FALSE(v2::parallax_state_words_are_authenticated(corrupt, calibration.raw_coordinate_scale));
    ASSERT_TRUE(replay->overwrite_state_for_testing(
      std::vector<std::uint32_t>(corrupt.begin(), corrupt.end()), error)) << error;
    sbs_bench::depth_coordinate_v2_gpu_frame output;
    ASSERT_TRUE(replay->dispatch(index, ids[index],
      test_source_sha256(), output, error)) << error;
    EXPECT_NEAR(output.order_minimum, -1.0f / calibration.raw_coordinate_scale, 2.0e-6f);
  }
  const auto trace_path = tree.path / "trace.json";
  ASSERT_TRUE(replay->write_state_trace(trace_path, error)) << error;
  const auto rows = nlohmann::ordered_json::parse(read_bytes(trace_path)).at("frames");
  for (const auto &row : rows) {
    EXPECT_EQ(row["joint_plane_mode"], 3u);
    EXPECT_EQ(row["calibration_revision"], 1u);
    EXPECT_NEAR(row["center"].get<float>(), 1.0f, 2.0e-5f);
  }
  for (const nlohmann::ordered_json &mode : {
         nlohmann::ordered_json(1u), nlohmann::ordered_json(2u),
         nlohmann::ordered_json(4u), nlohmann::ordered_json(-1),
         nlohmann::ordered_json(1.5), nlohmann::ordered_json(true),
       }) {
    auto invalid = manifest;
    invalid["mapping_config"]["joint_plane_mode"] = mode;
    ASSERT_TRUE(write_bytes(path, invalid.dump(2) + "\n"));
    auto rejected = sbs_bench::depth_coordinate_v2_gpu_replay::create(
      warp.device.Get(), warp.context.Get(), path, error);
    EXPECT_EQ(rejected, nullptr) << mode;
  }
}

#endif  // _WIN32
