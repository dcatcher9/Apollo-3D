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

  bool create_bgra_source_srv(
    ID3D11Device *device,
    const std::uint32_t width,
    const std::uint32_t height,
    const std::vector<std::uint32_t> &pixels,
    ComPtr<ID3D11Texture2D> &texture,
    ComPtr<ID3D11ShaderResourceView> &srv
  ) {
    if (pixels.size() != static_cast<std::size_t>(width) * height) {
      return false;
    }
    D3D11_TEXTURE2D_DESC description {};
    description.Width = width;
    description.Height = height;
    description.MipLevels = 1u;
    description.ArraySize = 1u;
    description.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    description.SampleDesc.Count = 1u;
    description.Usage = D3D11_USAGE_IMMUTABLE;
    description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial_data {
      pixels.data(), static_cast<UINT>(width * sizeof(std::uint32_t)), 0u
    };
    return SUCCEEDED(device->CreateTexture2D(
             &description, &initial_data, texture.ReleaseAndGetAddressOf())) &&
           SUCCEEDED(device->CreateShaderResourceView(
             texture.Get(), nullptr, srv.ReleaseAndGetAddressOf()));
  }

  bool create_solid_source_srv(
    ID3D11Device *device,
    const std::uint32_t width,
    const std::uint32_t height,
    ComPtr<ID3D11Texture2D> &texture,
    ComPtr<ID3D11ShaderResourceView> &srv
  ) {
    return create_bgra_source_srv(
      device,
      width,
      height,
      std::vector<std::uint32_t>(static_cast<std::size_t>(width) * height, 0xff202020u),
      texture,
      srv
    );
  }

  bool create_r32_source_srv(
    ID3D11Device *device,
    const std::uint32_t width,
    const std::uint32_t height,
    ComPtr<ID3D11Texture2D> &texture,
    ComPtr<ID3D11ShaderResourceView> &srv
  ) {
    std::vector<float> pixels(static_cast<std::size_t>(width) * height, 0.5f);
    D3D11_TEXTURE2D_DESC description {};
    description.Width = width;
    description.Height = height;
    description.MipLevels = 1u;
    description.ArraySize = 1u;
    description.Format = DXGI_FORMAT_R32_FLOAT;
    description.SampleDesc.Count = 1u;
    description.Usage = D3D11_USAGE_IMMUTABLE;
    description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial_data {
      pixels.data(), static_cast<UINT>(width * sizeof(float)), 0u
    };
    return SUCCEEDED(device->CreateTexture2D(
             &description, &initial_data, texture.ReleaseAndGetAddressOf())) &&
           SUCCEEDED(device->CreateShaderResourceView(
             texture.Get(), nullptr, srv.ReleaseAndGetAddressOf()));
  }

  bool create_rgba16f_source_srv(
    ID3D11Device *device,
    const std::uint32_t width,
    const std::uint32_t height,
    const std::vector<std::array<std::uint16_t, 4>> &pixels,
    ComPtr<ID3D11Texture2D> &texture,
    ComPtr<ID3D11ShaderResourceView> &srv
  ) {
    if (pixels.size() != static_cast<std::size_t>(width) * height) {
      return false;
    }
    D3D11_TEXTURE2D_DESC description {};
    description.Width = width;
    description.Height = height;
    description.MipLevels = 1u;
    description.ArraySize = 1u;
    description.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    description.SampleDesc.Count = 1u;
    description.Usage = D3D11_USAGE_IMMUTABLE;
    description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initial_data {
      pixels.data(),
      static_cast<UINT>(width * sizeof(pixels.front())),
      0u
    };
    return SUCCEEDED(device->CreateTexture2D(
             &description, &initial_data, texture.ReleaseAndGetAddressOf())) &&
           SUCCEEDED(device->CreateShaderResourceView(
             texture.Get(), nullptr, srv.ReleaseAndGetAddressOf()));
  }

  nlohmann::ordered_json make_replay_manifest(
    const models::depth_coordinate_v2::model_calibration_t &calibration,
    const std::string_view contract_sha256,
    const std::uint32_t width,
    const std::uint32_t height,
    const nlohmann::ordered_json &frames,
    const float pop_strength = 2.0f,
    const std::uint32_t source_width = 0u,
    const std::uint32_t source_height = 0u,
    const std::uint32_t source_color_mode = 0u,
    const float source_linear_scale = 1.0f
  ) {
    namespace v2 = models::depth_coordinate_v2;
    auto authenticated_frames = frames;
    for (auto &frame : authenticated_frames) {
      if (!frame.contains("source_sha256")) {
        frame["source_sha256"] = test_source_sha256();
      }
    }
    return {
      {"schema", 8u},
      {"mode", "depth-coordinate-v2-production-gpu-sequence-v10"},
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
        {"far_tau", v2::far_tau},
        {"near_log_tau", v2::near_log_tau},
        {"pop_strength", pop_strength},
        {"gain_per_pop", v2::gain_per_pop},
        {"max_horizontal_slope", v2::max_horizontal_slope},
        {"max_vertical_shear", v2::max_vertical_shear},
        {"vertical_majorant_share", v2::vertical_majorant_share},
        {"direct_container_limit", v2::direct_container_limit},
      }},
      {"source_color_mode", source_color_mode},
      {"source_linear_scale", source_linear_scale},
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
    if (width == 0u || height <= v2::limiter_group_threads || content_width == 0u ||
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
    if (width <= v2::limiter_group_threads || height == 0u || content_width == 0u ||
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
    const std::uint32_t dispatch_groups,
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
      v2::far_tau,
      v2::near_log_tau,
      v2::requested_gain_for_config(v2::reference_pop_strength),
      v2::max_horizontal_slope,
      v2::direct_container_limit,
      v2::convergence_curve_default,
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
    warp.context->Dispatch(dispatch_groups, 1u, 1u);

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

  bool dispatch_depth_coordinate_v2_ownership(
    warp_device_t &warp,
    ID3D11ComputeShader *shader,
    const std::uint32_t target_width,
    const std::uint32_t target_height,
    ID3D11ShaderResourceView *source_srv,
    const models::depth_source_rect_t source_region,
    const std::vector<float> &candidate,
    std::vector<float> &result
  ) {
    const auto element_count =
      static_cast<std::size_t>(target_width) * target_height;
    if (!shader || !source_srv || target_width == 0u || target_height == 0u ||
        !source_region.valid() || candidate.size() != element_count) {
      return false;
    }

    D3D11_TEXTURE2D_DESC input_desc {};
    input_desc.Width = target_width;
    input_desc.Height = target_height;
    input_desc.MipLevels = 1u;
    input_desc.ArraySize = 1u;
    input_desc.Format = DXGI_FORMAT_R32_FLOAT;
    input_desc.SampleDesc.Count = 1u;
    input_desc.Usage = D3D11_USAGE_IMMUTABLE;
    input_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA candidate_data {
      candidate.data(),
      static_cast<UINT>(static_cast<std::size_t>(target_width) * sizeof(float)),
      0u
    };
    ComPtr<ID3D11Texture2D> candidate_texture;
    ComPtr<ID3D11ShaderResourceView> candidate_srv;
    if (FAILED(warp.device->CreateTexture2D(
          &input_desc, &candidate_data, &candidate_texture
        )) ||
        FAILED(warp.device->CreateShaderResourceView(
          candidate_texture.Get(), nullptr, &candidate_srv
        ))) {
      return false;
    }

    const std::vector<std::uint32_t> exclusion(element_count, 0u);
    D3D11_TEXTURE2D_DESC exclusion_desc = input_desc;
    exclusion_desc.Format = DXGI_FORMAT_R32_UINT;
    D3D11_SUBRESOURCE_DATA exclusion_data {
      exclusion.data(),
      static_cast<UINT>(
        static_cast<std::size_t>(target_width) * sizeof(std::uint32_t)
      ),
      0u
    };
    ComPtr<ID3D11Texture2D> exclusion_texture;
    ComPtr<ID3D11ShaderResourceView> exclusion_srv;
    if (FAILED(warp.device->CreateTexture2D(
          &exclusion_desc, &exclusion_data, &exclusion_texture
        )) ||
        FAILED(warp.device->CreateShaderResourceView(
          exclusion_texture.Get(), nullptr, &exclusion_srv
        ))) {
      return false;
    }

    D3D11_TEXTURE2D_DESC output_desc = input_desc;
    output_desc.Usage = D3D11_USAGE_DEFAULT;
    output_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    ComPtr<ID3D11Texture2D> output_texture;
    ComPtr<ID3D11UnorderedAccessView> output_uav;
    if (FAILED(warp.device->CreateTexture2D(
          &output_desc, nullptr, &output_texture
        )) ||
        FAILED(warp.device->CreateUnorderedAccessView(
          output_texture.Get(), nullptr, &output_uav
        ))) {
      return false;
    }

    const auto create_constant_buffer = [&]<typename T>(
                                          const T &words,
                                          ComPtr<ID3D11Buffer> &buffer) {
      D3D11_BUFFER_DESC description {};
      description.ByteWidth = sizeof(T);
      description.Usage = D3D11_USAGE_IMMUTABLE;
      description.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA initial_data {&words, 0u, 0u};
      return SUCCEEDED(warp.device->CreateBuffer(
        &description, &initial_data, &buffer
      ));
    };
    std::array<std::uint32_t, 16> common_constants {};
    common_constants[0] = target_width;
    common_constants[1] = target_height;
    common_constants[11] = target_width;
    common_constants[12] = target_height;
    namespace v2 = models::depth_coordinate_v2;
    const v2::constants_t v2_constants {
      v2::model_calibrations.front().raw_coordinate_scale,
      v2::collapse_abs_epsilon,
      v2::far_tau,
      v2::near_log_tau,
      v2::requested_gain_for_config(v2::reference_pop_strength),
      v2::max_horizontal_slope,
      v2::direct_container_limit,
      v2::convergence_curve_default,
    };
    const std::array<std::uint32_t, 4> source_constants {
      source_region.left,
      source_region.top,
      source_region.right,
      source_region.bottom,
    };
    ComPtr<ID3D11Buffer> common_buffer;
    ComPtr<ID3D11Buffer> v2_buffer;
    ComPtr<ID3D11Buffer> source_buffer;
    if (!create_constant_buffer(common_constants, common_buffer) ||
        !create_constant_buffer(v2_constants, v2_buffer) ||
        !create_constant_buffer(source_constants, source_buffer)) {
      return false;
    }

    ID3D11ShaderResourceView *srvs[] = {
      candidate_srv.Get(), source_srv, exclusion_srv.Get()
    };
    ID3D11Buffer *constant_buffers[] = {
      common_buffer.Get(), v2_buffer.Get(), source_buffer.Get()
    };
    const float unwritten[4] = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
    };
    warp.context->ClearUnorderedAccessViewFloat(output_uav.Get(), unwritten);
    warp.context->CSSetShader(shader, nullptr, 0u);
    warp.context->CSSetShaderResources(0u, 3u, srvs);
    warp.context->CSSetUnorderedAccessViews(
      0u, 1u, output_uav.GetAddressOf(), nullptr
    );
    warp.context->CSSetConstantBuffers(0u, 3u, constant_buffers);
    warp.context->Dispatch(
      (target_width + 7u) / 8u,
      (target_height + 7u) / 8u,
      1u
    );

    ID3D11ShaderResourceView *null_srvs[3] = {nullptr, nullptr, nullptr};
    ID3D11UnorderedAccessView *null_uav = nullptr;
    warp.context->CSSetShaderResources(0u, 3u, null_srvs);
    warp.context->CSSetUnorderedAccessViews(0u, 1u, &null_uav, nullptr);
    warp.context->CSSetShader(nullptr, nullptr, 0u);

    D3D11_TEXTURE2D_DESC staging_desc = output_desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0u;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging_texture;
    if (FAILED(warp.device->CreateTexture2D(
          &staging_desc, nullptr, &staging_texture
        ))) {
      return false;
    }
    warp.context->CopyResource(staging_texture.Get(), output_texture.Get());
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(warp.context->Map(
          staging_texture.Get(), 0u, D3D11_MAP_READ, 0u, &mapped
        ))) {
      return false;
    }
    result.resize(element_count);
    for (std::uint32_t y = 0u; y < target_height; ++y) {
      const auto *row = reinterpret_cast<const float *>(
        static_cast<const std::byte *>(mapped.pData) +
        static_cast<std::size_t>(y) * mapped.RowPitch
      );
      std::copy_n(
        row,
        target_width,
        result.begin() + static_cast<std::size_t>(y) * target_width
      );
    }
    warp.context->Unmap(staging_texture.Get(), 0u);
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
  EXPECT_FALSE(record_ownership(warp.context.Get(), ownership_command_t {}));
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
  const auto map = stage("boolrecord_map_history(", "boolrecord_ownership(");
  const auto ownership = stage("boolrecord_ownership(", "boolrecord_vertical(");
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
  EXPECT_NE(ownership.find("CSSetShaderResources(0u,3u,"),
            std::string::npos);
  EXPECT_NE(ownership.find("CSSetUnorderedAccessViews(0u,1u"),
            std::string::npos);
  EXPECT_NE(vertical.find("CSSetShaderResources(0u,1u"), std::string::npos);
  EXPECT_NE(vertical.find("CSSetUnorderedAccessViews(0u,2u"), std::string::npos);
  EXPECT_NE(horizontal.find("CSSetShaderResources(0u,1u"), std::string::npos);
  EXPECT_NE(horizontal.find("CSSetUnorderedAccessViews(0u,1u"),
            std::string::npos);

  for (const std::string_view recorder : {
         "record_moments_frame",
         "record_state",
         "record_map_history",
         "record_ownership",
         "record_vertical",
         "record_horizontal",
       }) {
    EXPECT_EQ(count_occurrences(compact_replay, recorder), 1u) << recorder;
    EXPECT_EQ(count_occurrences(compact_live, recorder), 1u) << recorder;
  }
  EXPECT_EQ(count_occurrences(compact_replay, ".tensor_exclusion="), 3u);
  EXPECT_EQ(count_occurrences(compact_live, ".tensor_exclusion="), 3u);
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
         "near_identical_gpu_infer_grid8_byte_offset",
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
  EXPECT_NE(compact_live.find("depth_coordinate_v2_ownership_tex.Get(),"),
            std::string::npos);
  EXPECT_NE(compact_live.find("mark_d3d_parallax_map_start(perf_slot)"),
            std::string::npos);
  EXPECT_NE(compact_live.find("mark_d3d_parallax_subtitle_start(perf_slot)"),
            std::string::npos);
}

TEST(DepthCoordinateV2GpuTest, DirectRoiOwnershipMatchesPriorCropTextureBitwise) {
  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());

  const std::filesystem::path shader_path =
    std::filesystem::path(SUNSHINE_SHADERS_DIR) /
    "depth_coordinate_v2_ownership_cs.hlsl";
  ComPtr<ID3DBlob> shader_blob;
  ComPtr<ID3DBlob> errors;
  const HRESULT compile_status = D3DCompileFromFile(
    shader_path.c_str(),
    nullptr,
    D3D_COMPILE_STANDARD_FILE_INCLUDE,
    "main",
    "cs_5_0",
    D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
    0u,
    &shader_blob,
    &errors
  );
  ASSERT_TRUE(SUCCEEDED(compile_status))
    << (errors ? static_cast<const char *>(errors->GetBufferPointer()) :
                 "no compiler diagnostics");
  ComPtr<ID3D11ComputeShader> shader;
  ASSERT_TRUE(SUCCEEDED(warp.device->CreateComputeShader(
    shader_blob->GetBufferPointer(),
    shader_blob->GetBufferSize(),
    nullptr,
    &shader
  )));

  constexpr std::uint32_t target_width = 64u;
  constexpr std::uint32_t target_height = 36u;
  constexpr std::uint32_t source_width = 320u;
  constexpr std::uint32_t source_height = 180u;
  constexpr std::uint32_t source_left = 17u;
  constexpr std::uint32_t source_top = 13u;
  constexpr std::uint32_t texture_width = 360u;
  constexpr std::uint32_t texture_height = 220u;
  constexpr std::uint32_t split = target_width / 2u;
  constexpr std::uint32_t source_scale = source_width / target_width;
  constexpr std::uint32_t source_edge = split * source_scale - 2u;
  static_assert(source_width % target_width == 0u);

  std::vector<float> candidate(
    static_cast<std::size_t>(target_width) * target_height,
    -0.039f
  );
  for (std::uint32_t y = 0u; y < target_height; ++y) {
    std::fill(
      candidate.begin() + static_cast<std::size_t>(y) * target_width + split,
      candidate.begin() + static_cast<std::size_t>(y + 1u) * target_width,
      0.039f
    );
  }

  std::vector<std::uint32_t> crop_pixels(
    static_cast<std::size_t>(source_width) * source_height,
    0xff101010u
  );
  for (std::uint32_t y = 0u; y < source_height; ++y) {
    std::fill(
      crop_pixels.begin() + static_cast<std::size_t>(y) * source_width + source_edge,
      crop_pixels.begin() + static_cast<std::size_t>(y + 1u) * source_width,
      0xfff0f0f0u
    );
  }
  std::vector<std::uint32_t> full_pixels(
    static_cast<std::size_t>(texture_width) * texture_height,
    0xffff00ffu
  );
  for (std::uint32_t y = 0u; y < source_height; ++y) {
    std::copy_n(
      crop_pixels.begin() + static_cast<std::size_t>(y) * source_width,
      source_width,
      full_pixels.begin() +
        static_cast<std::size_t>(source_top + y) * texture_width + source_left
    );
  }

  ComPtr<ID3D11Texture2D> crop_texture;
  ComPtr<ID3D11ShaderResourceView> crop_srv;
  ASSERT_TRUE(create_bgra_source_srv(
    warp.device.Get(),
    source_width,
    source_height,
    crop_pixels,
    crop_texture,
    crop_srv
  ));
  ComPtr<ID3D11Texture2D> full_texture;
  ComPtr<ID3D11ShaderResourceView> full_srv;
  ASSERT_TRUE(create_bgra_source_srv(
    warp.device.Get(),
    texture_width,
    texture_height,
    full_pixels,
    full_texture,
    full_srv
  ));

  std::vector<float> crop_result;
  std::vector<float> direct_result;
  ASSERT_TRUE(dispatch_depth_coordinate_v2_ownership(
    warp,
    shader.Get(),
    target_width,
    target_height,
    crop_srv.Get(),
    {0u, 0u, source_width, source_height},
    candidate,
    crop_result
  ));
  ASSERT_TRUE(dispatch_depth_coordinate_v2_ownership(
    warp,
    shader.Get(),
    target_width,
    target_height,
    full_srv.Get(),
    {
      source_left,
      source_top,
      source_left + source_width,
      source_top + source_height,
    },
    candidate,
    direct_result
  ));

  ASSERT_EQ(direct_result.size(), crop_result.size());
  std::size_t refined = 0u;
  for (std::size_t index = 0u; index < crop_result.size(); ++index) {
    EXPECT_EQ(
      std::bit_cast<std::uint32_t>(direct_result[index]),
      std::bit_cast<std::uint32_t>(crop_result[index])
    ) << "ownership index=" << index;
    refined += crop_result[index] > candidate[index] + 1.0e-7f ? 1u : 0u;
  }
  EXPECT_GT(refined, target_height / 2u)
    << "The parity fixture must exercise translated full-resolution color loads";
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
       std::array<std::uint32_t, 8> {33u, 63u, 64u, 65u, 434u, 770u, 1022u, 1036u}) {
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
  ASSERT_GT(width, v2::limiter_group_threads);
  ASSERT_GT(height, v2::limiter_group_threads);
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
  for (std::uint32_t chunk = 1u; chunk < v2::limiter_group_threads; ++chunk) {
    const std::uint32_t boundary_x = chunk * width / v2::limiter_group_threads;
    candidate[static_cast<std::size_t>(boundary_row) * width + boundary_x - 1u] =
      chunk % 2u == 0u ? 0.039f : -0.039f;
    candidate[static_cast<std::size_t>(boundary_row) * width + boundary_x] =
      chunk % 2u == 0u ? -0.039f : 0.039f;

    const std::uint32_t boundary_y = chunk * height / v2::limiter_group_threads;
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
  };

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

  std::size_t tested_shapes = 0u;
  for (const auto &shape : v2::model_calibrated_shapes) {
    if (shape.calibration_id != calibration.calibration_id) {
      continue;
    }
    ++tested_shapes;
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
    ComPtr<ID3D11Texture2D> source_texture;
    ComPtr<ID3D11ShaderResourceView> source_srv;
    const std::uint32_t source_edge = split * 2u - 1u;
    std::vector<std::uint32_t> pixels(
      static_cast<std::size_t>(source_width) * source_height, 0xff101010u);
    for (std::uint32_t y = 0u; y < source_height; ++y) {
      std::fill(
        pixels.begin() + static_cast<std::size_t>(y) * source_width + source_edge,
        pixels.begin() + static_cast<std::size_t>(y + 1u) * source_width,
        0xfff0f0f0u
      );
    }
    ASSERT_TRUE(create_bgra_source_srv(
      warp.device.Get(), source_width, source_height, pixels, source_texture, source_srv));

    sbs_bench::depth_coordinate_v2_gpu_frame output;
    ASSERT_TRUE(replay->dispatch(
      0u, "0001", source_srv.Get(), 0u, 1.0f, test_source_sha256(), output, error)) << error;
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
    const std::uint32_t boundary_x = split - 1u;
    for (std::uint32_t y = 0u; y < height; ++y) {
      for (std::uint32_t x = 0u; x < width; ++x) {
        const std::size_t index = static_cast<std::size_t>(y) * width + x;
        if (x == boundary_x) {
          EXPECT_GT(output.ownership_refined_parallax_values[index],
                    output.candidate_parallax_values[index] + 1.0e-5f);
        } else {
          EXPECT_FLOAT_EQ(output.ownership_refined_parallax_values[index],
                          output.candidate_parallax_values[index]);
        }
      }
    }
    const std::size_t boundary_index =
      static_cast<std::size_t>(height / 2u) * width + boundary_x;
    const float jump = output.candidate_parallax_values[boundary_index + 1u] -
                       output.candidate_parallax_values[boundary_index];
    ASSERT_GT(jump, 0.0f);
    const float correction_fraction =
      (output.ownership_refined_parallax_values[boundary_index] -
       output.candidate_parallax_values[boundary_index]) / jump;
    EXPECT_NEAR(correction_fraction, 0.5f, 2.0e-4f);

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
  EXPECT_EQ(tested_shapes, 6u);
}

TEST(DepthCoordinateV2GpuTest, FullResolutionContourRaisesOwnedBoundaryAndRejectsAmbiguity) {
  namespace fs = std::filesystem;
  namespace v2 = models::depth_coordinate_v2;
  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());
  temporary_tree_t tree;

  const auto &calibration = v2::model_calibrations.front();
  const auto shape_iterator = std::find_if(
    v2::model_calibrated_shapes.begin(),
    v2::model_calibrated_shapes.end(),
    [&](const auto &candidate) {
      return candidate.calibration_id == calibration.calibration_id;
    }
  );
  ASSERT_NE(shape_iterator, v2::model_calibrated_shapes.end());
  const std::uint32_t width = shape_iterator->width;
  const std::uint32_t height = shape_iterator->height;
  const std::size_t element_count = static_cast<std::size_t>(width) * height;
  const std::uint32_t split = width / 2u;
  const std::uint32_t source_width = 3840u;
  const std::uint32_t source_height = 2160u;

  const fs::path contract_source =
    fs::path(SUNSHINE_SOURCE_DIR) / "tools/sbsbench/contracts/depth-coordinate-v2-v1.json";
  const std::string contract_bytes = read_bytes(contract_source);
  ASSERT_FALSE(contract_bytes.empty());
  ASSERT_TRUE(write_bytes(
    tree.path / "contracts/depth-coordinate-v2-v1.json", contract_bytes));

  std::vector<float> raw(element_count, 0.0f);
  for (std::uint32_t y = 0; y < height; ++y) {
    std::fill(
      raw.begin() + static_cast<std::size_t>(y) * width + split,
      raw.begin() + static_cast<std::size_t>(y + 1u) * width,
      20.0f
    );
  }
  const std::string raw_bytes = float_bytes(raw);
  ASSERT_TRUE(write_bytes(tree.path / "raw_00001.f32", raw_bytes));
  ASSERT_TRUE(write_bytes(tree.path / "raw_00002.f32", raw_bytes));
  ASSERT_TRUE(write_bytes(tree.path / "raw_00003.f32", raw_bytes));
  ASSERT_TRUE(write_bytes(tree.path / "raw_00004.f32", raw_bytes));
  ASSERT_TRUE(write_bytes(tree.path / "raw_00005.f32", raw_bytes));
  const std::string source_a_sha256 = sha256_hex("ownership-positive-control-a");
  const std::string source_b_sha256 = sha256_hex("ownership-positive-control-b");
  const std::string ambiguous_source_sha256 = sha256_hex("ownership-ambiguous-combined");
  const std::string reversed_source_sha256 = sha256_hex("ownership-subprofile-reversal");
  const std::string refinement_source_sha256 =
    sha256_hex("ownership-refinement-reversal");
  const nlohmann::ordered_json frames = nlohmann::ordered_json::array({
    {
      {"frame_id", "00001"},
      {"raw_file", "raw_00001.f32"},
      {"raw_sha256", sha256_hex(raw_bytes)},
      {"source_sha256", source_a_sha256},
      {"hard_cut_count", 0u},
      {"hard_cut_pulse", false},
    },
    {
      {"frame_id", "00002"},
      {"raw_file", "raw_00002.f32"},
      {"raw_sha256", sha256_hex(raw_bytes)},
      {"source_sha256", source_b_sha256},
      {"hard_cut_count", 0u},
      {"hard_cut_pulse", false},
    },
    {
      {"frame_id", "00003"},
      {"raw_file", "raw_00003.f32"},
      {"raw_sha256", sha256_hex(raw_bytes)},
      {"source_sha256", ambiguous_source_sha256},
      {"hard_cut_count", 0u},
      {"hard_cut_pulse", false},
    },
    {
      {"frame_id", "00004"},
      {"raw_file", "raw_00004.f32"},
      {"raw_sha256", sha256_hex(raw_bytes)},
      {"source_sha256", reversed_source_sha256},
      {"hard_cut_count", 0u},
      {"hard_cut_pulse", false},
    },
    {
      {"frame_id", "00005"},
      {"raw_file", "raw_00005.f32"},
      {"raw_sha256", sha256_hex(raw_bytes)},
      {"source_sha256", refinement_source_sha256},
      {"hard_cut_count", 0u},
      {"hard_cut_pulse", false},
    },
  });
  const auto manifest = make_replay_manifest(
    calibration, sha256_hex(contract_bytes), width, height, frames, 1.0f,
    source_width, source_height);
  const fs::path manifest_path = tree.path / "manifest.json";
  ASSERT_TRUE(write_bytes(manifest_path, manifest.dump(2) + "\n"));

  // At 4K, both isolated contours below are inside the supported far-cell ownership range.
  // Keeping them as explicit positive controls proves the combined case abstains because of
  // competing contour evidence, not because either contour lies outside the accepted window.
  const float source_scale = static_cast<float>(source_width) / width;
  const std::uint32_t source_edge_a = static_cast<std::uint32_t>(std::lround(
    static_cast<float>(split) * source_scale - 0.8f * source_scale
  ));
  const std::uint32_t source_edge_b = static_cast<std::uint32_t>(std::lround(
    static_cast<float>(split) * source_scale - 0.4f * source_scale
  ));
  std::vector<std::uint32_t> pixels(
    static_cast<std::size_t>(source_width) * source_height, 0xff000000u);
  for (std::uint32_t y = 0; y < source_height; ++y) {
    std::fill(
      pixels.begin() + static_cast<std::size_t>(y) * source_width + source_edge_a,
      pixels.begin() + static_cast<std::size_t>(y + 1u) * source_width,
      0xffd0d0d0u
    );
  }
  ComPtr<ID3D11Texture2D> source_texture;
  ComPtr<ID3D11ShaderResourceView> source_srv;
  ASSERT_TRUE(create_bgra_source_srv(
    warp.device.Get(), source_width, source_height, pixels, source_texture, source_srv));

  std::string error;
  auto replay = sbs_bench::depth_coordinate_v2_gpu_replay::create(
    warp.device.Get(), warp.context.Get(), manifest_path, error);
  ASSERT_NE(replay, nullptr) << error;

  ComPtr<ID3D11Texture2D> wrong_size_texture;
  ComPtr<ID3D11ShaderResourceView> wrong_size_srv;
  ASSERT_TRUE(create_solid_source_srv(
    warp.device.Get(), source_width - 1u, source_height,
    wrong_size_texture, wrong_size_srv));
  sbs_bench::depth_coordinate_v2_gpu_frame rejected_output;
  EXPECT_FALSE(replay->dispatch(
    0u, "00001", wrong_size_srv.Get(), 0u, 1.0f,
    source_a_sha256, rejected_output, error));
  EXPECT_NE(error.find("authenticated shape/format"), std::string::npos) << error;
  error.clear();

  ComPtr<ID3D11Texture2D> wrong_format_texture;
  ComPtr<ID3D11ShaderResourceView> wrong_format_srv;
  ASSERT_TRUE(create_r32_source_srv(
    warp.device.Get(), source_width, source_height,
    wrong_format_texture, wrong_format_srv));
  EXPECT_FALSE(replay->dispatch(
    0u, "00001", wrong_format_srv.Get(), 0u, 1.0f,
    source_a_sha256, rejected_output, error));
  EXPECT_NE(error.find("authenticated shape/format"), std::string::npos) << error;
  error.clear();

  sbs_bench::depth_coordinate_v2_gpu_frame output;
  ASSERT_TRUE(replay->dispatch(
    0u, "00001", source_srv.Get(), 0u, 1.0f, source_a_sha256, output, error)) << error;
  ASSERT_EQ(output.candidate_parallax_values.size(), element_count);
  ASSERT_EQ(output.ownership_refined_parallax_values.size(), element_count);

  std::size_t raised = 0u;
  for (std::size_t index = 0; index < element_count; ++index) {
    EXPECT_GE(
      output.ownership_refined_parallax_values[index] + 1.0e-8f,
      output.candidate_parallax_values[index]
    );
    raised += output.ownership_refined_parallax_values[index] >
      output.candidate_parallax_values[index] + 1.0e-7f ? 1u : 0u;
  }
  EXPECT_GT(raised, height / 2u);
  const std::size_t center = static_cast<std::size_t>(height / 2u) * width + split - 1u;
  EXPECT_GT(output.ownership_refined_parallax_values[center],
            output.candidate_parallax_values[center] + 1.0e-5f);
  EXPECT_LT(output.ownership_refined_parallax_values[center],
            output.candidate_parallax_values[center + 1u] - 1.0e-5f);
  EXPECT_FLOAT_EQ(output.ownership_refined_parallax_values[center - 1u],
                  output.candidate_parallax_values[center - 1u]);
  const float source_a_jump =
    output.candidate_parallax_values[center + 1u] -
    output.candidate_parallax_values[center];
  ASSERT_GT(source_a_jump, 0.0f);
  const float source_a_fraction =
    (output.ownership_refined_parallax_values[center] -
     output.candidate_parallax_values[center]) / source_a_jump;
  const float expected_source_a_fraction =
    (static_cast<float>(split) * source_scale -
     static_cast<float>(source_edge_a)) / source_scale;
  EXPECT_NEAR(source_a_fraction, expected_source_a_fraction, 2.0e-4f);

  // A second isolated contour at the other supported position must also be accepted.
  std::vector<std::uint32_t> source_b_pixels(
    static_cast<std::size_t>(source_width) * source_height, 0xffa0a0a0u);
  for (std::uint32_t y = 0; y < source_height; ++y) {
    std::fill(
      source_b_pixels.begin() + static_cast<std::size_t>(y) * source_width + source_edge_b,
      source_b_pixels.begin() + static_cast<std::size_t>(y + 1u) * source_width,
      0xffffffffu
    );
  }
  ComPtr<ID3D11Texture2D> source_b_texture;
  ComPtr<ID3D11ShaderResourceView> source_b_srv;
  ASSERT_TRUE(create_bgra_source_srv(
    warp.device.Get(), source_width, source_height, source_b_pixels,
    source_b_texture, source_b_srv));
  sbs_bench::depth_coordinate_v2_gpu_frame source_b_output;
  ASSERT_TRUE(replay->dispatch(
    1u, "00002", source_b_srv.Get(), 0u, 1.0f,
    source_b_sha256, source_b_output, error)) << error;
  EXPECT_GT(
    source_b_output.ownership_refined_parallax_values[center],
    source_b_output.candidate_parallax_values[center] + 1.0e-5f
  );
  const float source_b_jump =
    source_b_output.candidate_parallax_values[center + 1u] -
    source_b_output.candidate_parallax_values[center];
  ASSERT_GT(source_b_jump, 0.0f);
  const float source_b_fraction =
    (source_b_output.ownership_refined_parallax_values[center] -
     source_b_output.candidate_parallax_values[center]) / source_b_jump;
  const float expected_source_b_fraction =
    (static_cast<float>(split) * source_scale -
     static_cast<float>(source_edge_b)) / source_scale;
  EXPECT_NEAR(source_b_fraction, expected_source_b_fraction, 2.0e-4f);

  // Combining both individually valid transitions must expose disagreement between the smoothed
  // crossing interval and its raw bracket, and abstain instead of selecting either contour.
  std::vector<std::uint32_t> ambiguous_pixels(
    static_cast<std::size_t>(source_width) * source_height, 0xff000000u);
  for (std::uint32_t y = 0; y < source_height; ++y) {
    const auto row = ambiguous_pixels.begin() + static_cast<std::size_t>(y) * source_width;
    std::fill(row + source_edge_a, row + source_edge_a + 1u, 0xffd0d0d0u);
    std::fill(row + source_edge_a + 1u, row + source_edge_b, 0xffa0a0a0u);
    std::fill(row + source_edge_b, row + source_width, 0xffffffffu);
  }
  ComPtr<ID3D11Texture2D> ambiguous_source_texture;
  ComPtr<ID3D11ShaderResourceView> ambiguous_source_srv;
  ASSERT_TRUE(create_bgra_source_srv(
    warp.device.Get(), source_width, source_height, ambiguous_pixels,
    ambiguous_source_texture, ambiguous_source_srv));
  sbs_bench::depth_coordinate_v2_gpu_frame ambiguous_output;
  ASSERT_TRUE(replay->dispatch(
    2u,
    "00003",
    ambiguous_source_srv.Get(),
    0u,
    1.0f,
    ambiguous_source_sha256,
    ambiguous_output,
    error
  )) << error;
  ASSERT_EQ(ambiguous_output.candidate_parallax_values.size(), element_count);
  ASSERT_EQ(ambiguous_output.ownership_refined_parallax_values.size(), element_count);
  for (std::size_t index = 0; index < element_count; ++index) {
    EXPECT_FLOAT_EQ(
      ambiguous_output.ownership_refined_parallax_values[index],
      ambiguous_output.candidate_parallax_values[index]
    ) << "index=" << index;
  }

  // Two individually valid contours can also sit only two 4K source pixels apart. The coarse
  // uniqueness filter merges them, but their raw projection reverses direction between rises;
  // ownership must conservatively abstain on that sub-profile evidence.
  std::vector<std::uint32_t> reversed_pixels(
    static_cast<std::size_t>(source_width) * source_height, 0xff000000u);
  for (std::uint32_t y = 0; y < source_height; ++y) {
    const auto row = reversed_pixels.begin() + static_cast<std::size_t>(y) * source_width;
    row[1917u] = 0xffffffffu;
    std::fill(row + 1919u, row + source_width, 0xffffffffu);
  }
  ComPtr<ID3D11Texture2D> reversed_source_texture;
  ComPtr<ID3D11ShaderResourceView> reversed_source_srv;
  ASSERT_TRUE(create_bgra_source_srv(
    warp.device.Get(), source_width, source_height, reversed_pixels,
    reversed_source_texture, reversed_source_srv));
  sbs_bench::depth_coordinate_v2_gpu_frame reversed_output;
  ASSERT_TRUE(replay->dispatch(
    3u, "00004", reversed_source_srv.Get(), 0u, 1.0f,
    reversed_source_sha256, reversed_output, error)) << error;
  ASSERT_EQ(reversed_output.candidate_parallax_values.size(), element_count);
  ASSERT_EQ(reversed_output.ownership_refined_parallax_values.size(), element_count);
  for (std::size_t index = 0; index < element_count; ++index) {
    EXPECT_FLOAT_EQ(
      reversed_output.ownership_refined_parallax_values[index],
      reversed_output.candidate_parallax_values[index]
    ) << "index=" << index;
  }

  // A second two-pixel pattern aliases to a monotone coarse profile; the already-required
  // bisection sample exposes its hidden reversal and must also force an exact no-op.
  std::vector<std::uint32_t> refinement_pixels(
    static_cast<std::size_t>(source_width) * source_height, 0xff000000u);
  for (std::uint32_t y = 0; y < source_height; ++y) {
    const auto row = refinement_pixels.begin() + static_cast<std::size_t>(y) * source_width;
    row[1916u] = 0xffffffffu;
    std::fill(row + 1918u, row + source_width, 0xffffffffu);
  }
  ComPtr<ID3D11Texture2D> refinement_source_texture;
  ComPtr<ID3D11ShaderResourceView> refinement_source_srv;
  ASSERT_TRUE(create_bgra_source_srv(
    warp.device.Get(), source_width, source_height, refinement_pixels,
    refinement_source_texture, refinement_source_srv));
  sbs_bench::depth_coordinate_v2_gpu_frame refinement_output;
  ASSERT_TRUE(replay->dispatch(
    4u, "00005", refinement_source_srv.Get(), 0u, 1.0f,
    refinement_source_sha256, refinement_output, error)) << error;
  ASSERT_EQ(refinement_output.candidate_parallax_values.size(), element_count);
  ASSERT_EQ(refinement_output.ownership_refined_parallax_values.size(), element_count);
  for (std::size_t index = 0; index < element_count; ++index) {
    EXPECT_FLOAT_EQ(
      refinement_output.ownership_refined_parallax_values[index],
      refinement_output.candidate_parallax_values[index]
    ) << "index=" << index;
  }
}

TEST(DepthCoordinateV2GpuTest, OwnershipIsResolutionAndPixelPhaseInvariant) {
  namespace fs = std::filesystem;
  namespace v2 = models::depth_coordinate_v2;
  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());
  temporary_tree_t tree;

  const auto &calibration = v2::model_calibrations.front();
  const auto shape_iterator = std::find_if(
    v2::model_calibrated_shapes.begin(),
    v2::model_calibrated_shapes.end(),
    [&](const auto &candidate) {
      return candidate.calibration_id == calibration.calibration_id;
    }
  );
  ASSERT_NE(shape_iterator, v2::model_calibrated_shapes.end());
  const std::uint32_t width = shape_iterator->width;
  const std::uint32_t height = shape_iterator->height;
  const std::size_t element_count = static_cast<std::size_t>(width) * height;
  const fs::path contract_source =
    fs::path(SUNSHINE_SOURCE_DIR) / "tools/sbsbench/contracts/depth-coordinate-v2-v1.json";
  const std::string contract_bytes = read_bytes(contract_source);
  ASSERT_FALSE(contract_bytes.empty());
  // Express the synthetic cliff in canonical units so a model raw-scale recalibration cannot
  // silently move this ownership test below the production strong-cliff threshold.
  constexpr float canonical_depth_step = 40.0f;
  const float raw_depth_step =
    canonical_depth_step * calibration.raw_coordinate_scale;

  constexpr std::array source_sizes {
    std::pair {1280u, 720u},
    std::pair {1920u, 1080u},
    std::pair {2560u, 1440u},
    std::pair {3840u, 2160u},
  };
  // 38 is the former 720p smoothing-threshold hole; 41 is the former contaminated-plateau
  // under-correction; 645 and 722 exercise an exact unfiltered-0.5 endpoint phase.
  constexpr std::array splits {38u, 39u, 41u, 256u, 257u, 645u, 722u};
  for (std::size_t source_index = 0; source_index < source_sizes.size(); ++source_index) {
    const auto &[source_width, source_height] = source_sizes[source_index];
    SCOPED_TRACE(
      "source=" + std::to_string(source_width) + "x" + std::to_string(source_height)
    );
    const fs::path case_root =
      tree.path / (std::to_string(source_width) + "x" + std::to_string(source_height));
    ASSERT_TRUE(write_bytes(
      case_root / "contracts/depth-coordinate-v2-v1.json", contract_bytes));

    nlohmann::ordered_json frames = nlohmann::ordered_json::array();
    std::array<std::string, splits.size()> source_hashes;
    for (std::size_t index = 0; index < splits.size(); ++index) {
      std::vector<float> raw(element_count, 0.0f);
      for (std::uint32_t y = 0; y < height; ++y) {
        std::fill(
          raw.begin() + static_cast<std::size_t>(y) * width + splits[index],
          raw.begin() + static_cast<std::size_t>(y + 1u) * width,
          raw_depth_step
        );
      }
      const std::string frame_id = "0000" + std::to_string(index + 1u);
      const std::string raw_name = "raw_" + frame_id + ".f32";
      const std::string raw_bytes = float_bytes(raw);
      ASSERT_TRUE(write_bytes(case_root / raw_name, raw_bytes));
      source_hashes[index] = sha256_hex(
        std::to_string(source_width) + "x" + std::to_string(source_height) +
        "-phase-" + std::to_string(index)
      );
      frames.push_back({
        {"frame_id", frame_id},
        {"raw_file", raw_name},
        {"raw_sha256", sha256_hex(raw_bytes)},
        {"source_sha256", source_hashes[index]},
        {"hard_cut_count", index == 0u ? 0u : 1u},
        {"hard_cut_pulse", index != 0u},
      });
    }
    const auto manifest = make_replay_manifest(
      calibration,
      sha256_hex(contract_bytes),
      width,
      height,
      frames,
      1.0f,
      source_width,
      source_height
    );
    const fs::path manifest_path = case_root / "manifest.json";
    ASSERT_TRUE(write_bytes(manifest_path, manifest.dump(2) + "\n"));
    std::string error;
    auto replay = sbs_bench::depth_coordinate_v2_gpu_replay::create(
      warp.device.Get(), warp.context.Get(), manifest_path, error);
    ASSERT_NE(replay, nullptr) << error;

    for (std::size_t index = 0; index < splits.size(); ++index) {
      const float source_scale = static_cast<float>(source_width) / width;
      const std::uint32_t source_edge = static_cast<std::uint32_t>(std::lround(
        static_cast<float>(splits[index]) * source_scale - 0.4f * source_scale
      ));
      std::vector<std::uint32_t> pixels(
        static_cast<std::size_t>(source_width) * source_height, 0xff101010u);
      for (std::uint32_t y = 0; y < source_height; ++y) {
        std::fill(
          pixels.begin() + static_cast<std::size_t>(y) * source_width + source_edge,
          pixels.begin() + static_cast<std::size_t>(y + 1u) * source_width,
          0xfff0f0f0u
        );
      }
      ComPtr<ID3D11Texture2D> source_texture;
      ComPtr<ID3D11ShaderResourceView> source_srv;
      ASSERT_TRUE(create_bgra_source_srv(
        warp.device.Get(), source_width, source_height, pixels, source_texture, source_srv));
      sbs_bench::depth_coordinate_v2_gpu_frame output;
      const std::string frame_id = "0000" + std::to_string(index + 1u);
      ASSERT_TRUE(replay->dispatch(
        index, frame_id, source_srv.Get(), 0u, 1.0f,
        source_hashes[index], output, error)) << error;

      std::size_t raised = 0u;
      for (std::size_t element = 0; element < element_count; ++element) {
        EXPECT_GE(
          output.ownership_refined_parallax_values[element] + 1.0e-8f,
          output.candidate_parallax_values[element]
        );
        raised += output.ownership_refined_parallax_values[element] >
          output.candidate_parallax_values[element] + 1.0e-7f ? 1u : 0u;
      }
      EXPECT_GT(raised, height / 2u);
      const std::size_t boundary =
        static_cast<std::size_t>(height / 2u) * width + splits[index] - 1u;
      const float correction =
        output.ownership_refined_parallax_values[boundary] -
        output.candidate_parallax_values[boundary];
      const float full_jump =
        output.candidate_parallax_values[boundary + 1u] -
        output.candidate_parallax_values[boundary];
      ASSERT_GT(full_jump, 0.0f);
      const float cliff_floor =
        8.0f * v2::max_horizontal_slope / static_cast<float>(width);
      ASSERT_GE(full_jump + 1.0e-8f, cliff_floor);
      const float correction_fraction = correction / full_jump;
      const float expected_fraction =
        (static_cast<float>(splits[index]) * source_scale -
         static_cast<float>(source_edge)) / source_scale;
      EXPECT_GE(correction_fraction, 0.10f);
      EXPECT_LE(correction_fraction, 0.85f);
      EXPECT_NEAR(
        correction_fraction,
        expected_fraction,
        2.0e-4f
      );
    }
  }
}

TEST(DepthCoordinateV2GpuTest, OwnershipColorModesPreserveEquivalentContourDecision) {
  namespace fs = std::filesystem;
  namespace v2 = models::depth_coordinate_v2;
  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());
  temporary_tree_t tree;

  const auto &calibration = v2::model_calibrations.front();
  const auto shape_iterator = std::find_if(
    v2::model_calibrated_shapes.begin(),
    v2::model_calibrated_shapes.end(),
    [&](const auto &candidate) {
      return candidate.calibration_id == calibration.calibration_id;
    }
  );
  ASSERT_NE(shape_iterator, v2::model_calibrated_shapes.end());
  const std::uint32_t width = shape_iterator->width;
  const std::uint32_t height = shape_iterator->height;
  const std::size_t element_count = static_cast<std::size_t>(width) * height;
  constexpr std::uint32_t source_width = 1920u;
  constexpr std::uint32_t source_height = 1080u;
  constexpr std::uint32_t split = 256u;

  std::vector<float> raw(element_count, 0.0f);
  for (std::uint32_t y = 0; y < height; ++y) {
    std::fill(
      raw.begin() + static_cast<std::size_t>(y) * width + split,
      raw.begin() + static_cast<std::size_t>(y + 1u) * width,
      20.0f
    );
  }
  const std::string raw_bytes = float_bytes(raw);
  const std::string source_hash = sha256_hex("equivalent-color-mode-contour");
  const fs::path contract_source =
    fs::path(SUNSHINE_SOURCE_DIR) / "tools/sbsbench/contracts/depth-coordinate-v2-v1.json";
  const std::string contract_bytes = read_bytes(contract_source);
  ASSERT_FALSE(contract_bytes.empty());
  const float source_scale = static_cast<float>(source_width) / width;
  const std::uint32_t source_edge = static_cast<std::uint32_t>(std::lround(
    static_cast<float>(split) * source_scale - 0.4f * source_scale
  ));

  std::array<float, 3> correction_fractions {};
  for (std::uint32_t color_mode = 0u; color_mode < 3u; ++color_mode) {
    SCOPED_TRACE("color_mode=" + std::to_string(color_mode));
    // source_linear_scale authenticates scaling already applied before upload; the shader must
    // not multiply it again. Exercise a >1 scRGB texel and non-unity provenance in HDR mode.
    const float source_linear_scale = color_mode == 2u ? 2.0f : 1.0f;
    const fs::path case_root = tree.path / ("color-mode-" + std::to_string(color_mode));
    ASSERT_TRUE(write_bytes(
      case_root / "contracts/depth-coordinate-v2-v1.json", contract_bytes));
    ASSERT_TRUE(write_bytes(case_root / "raw_00001.f32", raw_bytes));
    const nlohmann::ordered_json frames = nlohmann::ordered_json::array({{
      {"frame_id", "00001"},
      {"raw_file", "raw_00001.f32"},
      {"raw_sha256", sha256_hex(raw_bytes)},
      {"source_sha256", source_hash},
      {"hard_cut_count", 0u},
      {"hard_cut_pulse", false},
    }});
    const auto manifest = make_replay_manifest(
      calibration,
      sha256_hex(contract_bytes),
      width,
      height,
      frames,
      1.0f,
      source_width,
      source_height,
      color_mode,
      source_linear_scale
    );
    const fs::path manifest_path = case_root / "manifest.json";
    ASSERT_TRUE(write_bytes(manifest_path, manifest.dump(2) + "\n"));
    std::string error;
    auto replay = sbs_bench::depth_coordinate_v2_gpu_replay::create(
      warp.device.Get(), warp.context.Get(), manifest_path, error);
    ASSERT_NE(replay, nullptr) << error;

    ComPtr<ID3D11Texture2D> source_texture;
    ComPtr<ID3D11ShaderResourceView> source_srv;
    if (color_mode == 0u) {
      std::vector<std::uint32_t> pixels(
        static_cast<std::size_t>(source_width) * source_height, 0xff000000u);
      for (std::uint32_t y = 0; y < source_height; ++y) {
        std::fill(
          pixels.begin() + static_cast<std::size_t>(y) * source_width + source_edge,
          pixels.begin() + static_cast<std::size_t>(y + 1u) * source_width,
          0xffffffffu
        );
      }
      ASSERT_TRUE(create_bgra_source_srv(
        warp.device.Get(), source_width, source_height, pixels,
        source_texture, source_srv));
    } else {
      constexpr std::uint16_t half_one = 0x3c00u;
      constexpr std::uint16_t half_point_two = 0x3266u;
      constexpr std::uint16_t half_point_three = 0x34cdu;
      constexpr std::uint16_t half_two = 0x4000u;
      constexpr std::uint16_t half_four = 0x4400u;
      // These ranges discriminate the transfer branches rather than merely proving that every
      // monotone hard edge normalizes to the same crossing. Linear 0.2->0.3 clears the calibrated
      // linear contrast floor but would fail if treated as sRGB; scRGB 2->4 survives Reinhard but
      // collapses if the linear-SDR clamp is selected.
      const std::uint16_t far_value =
        color_mode == 2u ? half_two : half_point_two;
      const std::uint16_t near_value =
        color_mode == 2u ? half_four : half_point_three;
      const std::array<std::uint16_t, 4> far_pixel {
        far_value, far_value, far_value, half_one,
      };
      const std::array<std::uint16_t, 4> near_pixel {
        near_value, near_value, near_value, half_one,
      };
      std::vector<std::array<std::uint16_t, 4>> pixels(
        static_cast<std::size_t>(source_width) * source_height, far_pixel);
      for (std::uint32_t y = 0; y < source_height; ++y) {
        std::fill(
          pixels.begin() + static_cast<std::size_t>(y) * source_width + source_edge,
          pixels.begin() + static_cast<std::size_t>(y + 1u) * source_width,
          near_pixel
        );
      }
      ASSERT_TRUE(create_rgba16f_source_srv(
        warp.device.Get(), source_width, source_height, pixels,
        source_texture, source_srv));
    }

    sbs_bench::depth_coordinate_v2_gpu_frame output;
    ASSERT_TRUE(replay->dispatch(
      0u, "00001", source_srv.Get(), color_mode, source_linear_scale,
      source_hash, output, error)) << error;
    const std::size_t boundary =
      static_cast<std::size_t>(height / 2u) * width + split - 1u;
    const float correction =
      output.ownership_refined_parallax_values[boundary] -
      output.candidate_parallax_values[boundary];
    const float jump =
      output.candidate_parallax_values[boundary + 1u] -
      output.candidate_parallax_values[boundary];
    ASSERT_GT(jump, 0.0f);
    correction_fractions[color_mode] = correction / jump;
    const float expected_fraction =
      (static_cast<float>(split) * source_scale -
       static_cast<float>(source_edge)) / source_scale;
    EXPECT_NEAR(correction_fractions[color_mode], expected_fraction, 2.0e-4f);
  }

  EXPECT_NEAR(correction_fractions[1], correction_fractions[0], 2.0e-4f);
  EXPECT_NEAR(correction_fractions[2], correction_fractions[0], 2.0e-4f);
}

TEST(DepthCoordinateV2GpuTest, OwnershipHandlesVerticalAndReverseNearDirections) {
  namespace fs = std::filesystem;
  namespace v2 = models::depth_coordinate_v2;
  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());
  temporary_tree_t tree;

  const auto &calibration = v2::model_calibrations.front();
  const auto shape_iterator = std::find_if(
    v2::model_calibrated_shapes.begin(),
    v2::model_calibrated_shapes.end(),
    [&](const auto &candidate) {
      return candidate.calibration_id == calibration.calibration_id;
    }
  );
  ASSERT_NE(shape_iterator, v2::model_calibrated_shapes.end());
  const std::uint32_t width = shape_iterator->width;
  const std::uint32_t height = shape_iterator->height;
  const std::size_t element_count = static_cast<std::size_t>(width) * height;
  constexpr std::uint32_t source_width = 1920u;
  constexpr std::uint32_t source_height = 1080u;
  struct direction_case_t {
    bool vertical;
    bool reverse;
    std::uint32_t split;
  };
  constexpr std::array cases {
    direction_case_t {false, true, 256u},
    direction_case_t {true, false, 144u},
    direction_case_t {true, true, 145u},
  };

  const fs::path contract_source =
    fs::path(SUNSHINE_SOURCE_DIR) / "tools/sbsbench/contracts/depth-coordinate-v2-v1.json";
  const std::string contract_bytes = read_bytes(contract_source);
  ASSERT_FALSE(contract_bytes.empty());
  ASSERT_TRUE(write_bytes(
    tree.path / "contracts/depth-coordinate-v2-v1.json", contract_bytes));
  nlohmann::ordered_json frames = nlohmann::ordered_json::array();
  std::array<std::string, cases.size()> source_hashes;
  for (std::size_t index = 0; index < cases.size(); ++index) {
    std::vector<float> raw(element_count, 0.0f);
    const auto &test_case = cases[index];
    for (std::uint32_t y = 0; y < height; ++y) {
      for (std::uint32_t x = 0; x < width; ++x) {
        const bool near_side = test_case.vertical ?
                                 (test_case.reverse ? y < test_case.split : y >= test_case.split) :
                                 (test_case.reverse ? x < test_case.split : x >= test_case.split);
        if (near_side) {
          raw[static_cast<std::size_t>(y) * width + x] = 20.0f;
        }
      }
    }
    const std::string frame_id = "0000" + std::to_string(index + 1u);
    const std::string raw_name = "raw_" + frame_id + ".f32";
    const std::string raw_bytes = float_bytes(raw);
    ASSERT_TRUE(write_bytes(tree.path / raw_name, raw_bytes));
    source_hashes[index] = sha256_hex("direction-" + std::to_string(index));
    frames.push_back({
      {"frame_id", frame_id},
      {"raw_file", raw_name},
      {"raw_sha256", sha256_hex(raw_bytes)},
      {"source_sha256", source_hashes[index]},
      {"hard_cut_count", static_cast<std::uint32_t>(index)},
      {"hard_cut_pulse", index != 0u},
    });
  }
  const auto manifest = make_replay_manifest(
    calibration, sha256_hex(contract_bytes), width, height, frames, 1.0f,
    source_width, source_height);
  const fs::path manifest_path = tree.path / "manifest.json";
  ASSERT_TRUE(write_bytes(manifest_path, manifest.dump(2) + "\n"));
  std::string error;
  auto replay = sbs_bench::depth_coordinate_v2_gpu_replay::create(
    warp.device.Get(), warp.context.Get(), manifest_path, error);
  ASSERT_NE(replay, nullptr) << error;

  for (std::size_t index = 0; index < cases.size(); ++index) {
    const auto &test_case = cases[index];
    const float scale = test_case.vertical ?
                          static_cast<float>(source_height) / height :
                          static_cast<float>(source_width) / width;
    const float boundary = static_cast<float>(test_case.split) * scale;
    const std::uint32_t source_edge = static_cast<std::uint32_t>(std::lround(
      boundary + (test_case.reverse ? 0.4f : -0.4f) * scale
    ));
    std::vector<std::uint32_t> pixels(
      static_cast<std::size_t>(source_width) * source_height, 0xff101010u);
    for (std::uint32_t y = 0; y < source_height; ++y) {
      for (std::uint32_t x = 0; x < source_width; ++x) {
        const bool near_side = test_case.vertical ?
                                 (test_case.reverse ? y < source_edge : y >= source_edge) :
                                 (test_case.reverse ? x < source_edge : x >= source_edge);
        if (near_side) {
          pixels[static_cast<std::size_t>(y) * source_width + x] = 0xfff0f0f0u;
        }
      }
    }
    ComPtr<ID3D11Texture2D> source_texture;
    ComPtr<ID3D11ShaderResourceView> source_srv;
    ASSERT_TRUE(create_bgra_source_srv(
      warp.device.Get(), source_width, source_height, pixels, source_texture, source_srv));
    sbs_bench::depth_coordinate_v2_gpu_frame output;
    const std::string frame_id = "0000" + std::to_string(index + 1u);
    ASSERT_TRUE(replay->dispatch(
      index, frame_id, source_srv.Get(), 0u, 1.0f,
      source_hashes[index], output, error)) << error;
    const std::uint32_t boundary_x = test_case.vertical ?
                                       width / 2u :
                                       (test_case.reverse ? test_case.split : test_case.split - 1u);
    const std::uint32_t boundary_y = test_case.vertical ?
                                       (test_case.reverse ? test_case.split : test_case.split - 1u) :
                                       height / 2u;
    const std::size_t boundary_index =
      static_cast<std::size_t>(boundary_y) * width + boundary_x;
    EXPECT_GT(
      output.ownership_refined_parallax_values[boundary_index],
      output.candidate_parallax_values[boundary_index] + 1.0e-5f
    );
    std::size_t raised = 0u;
    for (std::uint32_t y = 0u; y < height; ++y) {
      for (std::uint32_t x = 0u; x < width; ++x) {
        const std::size_t element = static_cast<std::size_t>(y) * width + x;
        const float candidate = output.candidate_parallax_values[element];
        const float refined = output.ownership_refined_parallax_values[element];
        EXPECT_GE(refined + 1.0e-8f, candidate);
        const bool expected_boundary = test_case.vertical ?
                                         y == boundary_y : x == boundary_x;
        if (expected_boundary) {
          EXPECT_GT(refined, candidate + 1.0e-5f);
          ++raised;
        } else {
          EXPECT_FLOAT_EQ(refined, candidate);
        }
      }
    }
    EXPECT_EQ(raised, test_case.vertical ? width : height);

    const float jump = test_case.vertical ?
                         output.candidate_parallax_values[
                           static_cast<std::size_t>(test_case.split) * width + boundary_x] -
                           output.candidate_parallax_values[
                             static_cast<std::size_t>(test_case.split - 1u) * width + boundary_x] :
                         output.candidate_parallax_values[
                           static_cast<std::size_t>(boundary_y) * width + test_case.split] -
                           output.candidate_parallax_values[
                             static_cast<std::size_t>(boundary_y) * width + test_case.split - 1u];
    ASSERT_NE(jump, 0.0f);
    const float correction_fraction =
      (output.ownership_refined_parallax_values[boundary_index] -
       output.candidate_parallax_values[boundary_index]) / std::abs(jump);
    const float expected_fraction = test_case.reverse ?
      (static_cast<float>(source_edge) - boundary) / scale :
      (boundary - static_cast<float>(source_edge)) / scale;
    EXPECT_NEAR(correction_fraction, expected_fraction, 2.0e-4f);
  }
}

TEST(DepthCoordinateV2GpuTest, OwnershipExecutesOnPortraitTensorAndSource) {
  namespace fs = std::filesystem;
  namespace v2 = models::depth_coordinate_v2;
  warp_device_t warp;
  ASSERT_TRUE(warp.initialize());
  temporary_tree_t tree;

  const auto &calibration = v2::model_calibrations.front();
  const auto shape_iterator = std::find_if(
    v2::model_calibrated_shapes.begin(),
    v2::model_calibrated_shapes.end(),
    [&](const auto &candidate) {
      return candidate.calibration_id == calibration.calibration_id &&
             candidate.width == 434u && candidate.height == 770u;
    }
  );
  ASSERT_NE(shape_iterator, v2::model_calibrated_shapes.end());
  const std::uint32_t width = shape_iterator->width;
  const std::uint32_t height = shape_iterator->height;
  constexpr std::uint32_t source_width = 720u;
  constexpr std::uint32_t source_height = 1280u;
  const std::uint32_t split = height / 2u;
  const std::size_t element_count = static_cast<std::size_t>(width) * height;

  std::vector<float> raw(element_count, 0.0f);
  std::fill(raw.begin() + static_cast<std::size_t>(split) * width, raw.end(), 20.0f);
  const std::string raw_bytes = float_bytes(raw);
  const std::string source_hash = sha256_hex("portrait-ownership-contour");
  const fs::path contract_source =
    fs::path(SUNSHINE_SOURCE_DIR) / "tools/sbsbench/contracts/depth-coordinate-v2-v1.json";
  const std::string contract_bytes = read_bytes(contract_source);
  ASSERT_FALSE(contract_bytes.empty());
  ASSERT_TRUE(write_bytes(
    tree.path / "contracts/depth-coordinate-v2-v1.json", contract_bytes));
  ASSERT_TRUE(write_bytes(tree.path / "raw_00001.f32", raw_bytes));
  const nlohmann::ordered_json frames = nlohmann::ordered_json::array({{
    {"frame_id", "00001"},
    {"raw_file", "raw_00001.f32"},
    {"raw_sha256", sha256_hex(raw_bytes)},
    {"source_sha256", source_hash},
    {"hard_cut_count", 0u},
    {"hard_cut_pulse", false},
  }});
  const auto manifest = make_replay_manifest(
    calibration, sha256_hex(contract_bytes), width, height, frames, 1.0f,
    source_width, source_height);
  const fs::path manifest_path = tree.path / "manifest.json";
  ASSERT_TRUE(write_bytes(manifest_path, manifest.dump(2) + "\n"));

  const float source_scale = static_cast<float>(source_height) / height;
  const float source_boundary = static_cast<float>(split) * source_scale;
  const std::uint32_t source_edge = static_cast<std::uint32_t>(std::lround(
    source_boundary - 0.4f * source_scale));
  std::vector<std::uint32_t> pixels(
    static_cast<std::size_t>(source_width) * source_height, 0xff101010u);
  std::fill(
    pixels.begin() + static_cast<std::size_t>(source_edge) * source_width,
    pixels.end(),
    0xfff0f0f0u
  );
  ComPtr<ID3D11Texture2D> source_texture;
  ComPtr<ID3D11ShaderResourceView> source_srv;
  ASSERT_TRUE(create_bgra_source_srv(
    warp.device.Get(), source_width, source_height, pixels, source_texture, source_srv));

  std::string error;
  auto replay = sbs_bench::depth_coordinate_v2_gpu_replay::create(
    warp.device.Get(), warp.context.Get(), manifest_path, error);
  ASSERT_NE(replay, nullptr) << error;
  sbs_bench::depth_coordinate_v2_gpu_frame output;
  ASSERT_TRUE(replay->dispatch(
    0u, "00001", source_srv.Get(), 0u, 1.0f, source_hash, output, error)) << error;

  const std::uint32_t boundary_y = split - 1u;
  for (std::uint32_t y = 0u; y < height; ++y) {
    for (std::uint32_t x = 0u; x < width; ++x) {
      const std::size_t index = static_cast<std::size_t>(y) * width + x;
      if (y == boundary_y) {
        EXPECT_GT(output.ownership_refined_parallax_values[index],
                  output.candidate_parallax_values[index] + 1.0e-5f);
      } else {
        EXPECT_FLOAT_EQ(output.ownership_refined_parallax_values[index],
                        output.candidate_parallax_values[index]);
      }
    }
  }
  const std::size_t boundary_index =
    static_cast<std::size_t>(boundary_y) * width + width / 2u;
  const float jump = output.candidate_parallax_values[boundary_index + width] -
                     output.candidate_parallax_values[boundary_index];
  ASSERT_GT(jump, 0.0f);
  const float correction_fraction =
    (output.ownership_refined_parallax_values[boundary_index] -
     output.candidate_parallax_values[boundary_index]) / jump;
  const float expected_fraction =
    (source_boundary - static_cast<float>(source_edge)) / source_scale;
  EXPECT_NEAR(correction_fraction, expected_fraction, 2.0e-4f);
}

TEST(DepthCoordinateV2GpuTest, ArithmeticMeanCenterLatchesAtCutAndHoldsAcrossLaterFramesOnGpu) {
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
    value += 0.25f;  // no cut: the first camera must remain latched
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
  ComPtr<ID3D11Texture2D> source_texture;
  ComPtr<ID3D11ShaderResourceView> source_srv;
  ASSERT_TRUE(create_solid_source_srv(
    warp.device.Get(), width, height, source_texture, source_srv));
  for (std::size_t index = 0u; index < fields.size(); ++index) {
    sbs_bench::depth_coordinate_v2_gpu_frame output;
    const std::string frame_id = "000" + std::to_string(index + 1u);
    ASSERT_TRUE(replay->dispatch(
      index, frame_id, source_srv.Get(), 0u, 1.0f, test_source_sha256(), output, error)) << error;
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
  EXPECT_NEAR(rows[1]["center"].get<float>(), first_center, 2.0e-5f);
  EXPECT_FLOAT_EQ(
    rows[1]["convergence_curve"].get<float>(), v2::convergence_curve_default);
  EXPECT_EQ(rows[1]["calibration_revision"], 1u);
  EXPECT_NEAR(rows[2]["center"].get<float>(), first_center + 1.0f, 2.0e-5f);
  EXPECT_FLOAT_EQ(
    rows[2]["convergence_curve"].get<float>(), v2::convergence_curve_default);
  EXPECT_EQ(rows[2]["confirmed_cut"], true);
  EXPECT_EQ(rows[2]["calibration_revision"], 2u);
  for (std::size_t index = 3u; index < fields.size(); ++index) {
    EXPECT_NEAR(rows[index]["center"].get<float>(), first_center + 1.0f, 2.0e-5f);
    EXPECT_EQ(rows[index]["calibration_revision"], 2u);
  }
  EXPECT_EQ(rows[10]["frame_valid"], false);
  EXPECT_EQ(rows[11]["frame_valid"], true);
  EXPECT_EQ(rows[11]["cut_attribution"], "none");
}

TEST(DepthCoordinateV2GpuTest, SevenCoordinatePassReplayLatchesRecoversAndRelatchesExactly) {
  namespace fs = std::filesystem;
  namespace v2 = models::depth_coordinate_v2;

  static_assert(v2::constant_float_count == 8u);
  static_assert(v2::state_float_count == 12u);
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
  raw_fields.emplace_back(element_count, 0.0f);  // invalid+cut cleared; ramp reacquires
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
  ComPtr<ID3D11Texture2D> source_texture;
  ComPtr<ID3D11ShaderResourceView> source_srv;
  ASSERT_TRUE(create_solid_source_srv(
    warp.device.Get(), width, height, source_texture, source_srv));

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
      source_srv.Get(),
      0u,
      1.0f,
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
  ASSERT_EQ(trace.at("frame_fields").size(), 41u);
  EXPECT_EQ(trace["producer"]["authority"],
            "authenticated-raw-source-color-plus-seven-v2-compute-shaders-persistent-gpu-state-v8");
  EXPECT_EQ(trace["producer"]["tensor_shape"]["width"], replay->width());
  EXPECT_EQ(trace["producer"]["tensor_shape"]["height"], replay->height());
  ASSERT_EQ(trace["producer"]["shader_sequence"].size(), 7u);
  EXPECT_EQ(trace["producer"]["shader_sequence"][2],
            "depth_coordinate_v2_state_resolve_cs.hlsl");
  EXPECT_EQ(trace["producer"]["shader_sequence"][4],
            "depth_coordinate_v2_ownership_cs.hlsl");
  EXPECT_EQ(trace["producer"]["contract_canonical_sha256"],
            v2::contract_canonical_sha256);

  const auto &rows = trace.at("frames");
  EXPECT_EQ(rows[0]["calibration_revision"], 1u);
  EXPECT_EQ(rows[1]["calibration_revision"], 1u);
  EXPECT_EQ(rows[2]["calibration_revision"], 1u);
  EXPECT_EQ(rows[3]["calibration_revision"], 2u);
  EXPECT_EQ(rows[4]["calibration_revision"], 2u);
  EXPECT_EQ(rows[5]["calibration_revision"], 2u);
  EXPECT_EQ(rows[6]["calibration_revision"], 2u);
  EXPECT_EQ(rows[7]["calibration_revision"], 2u);
  EXPECT_EQ(rows[8]["calibration_revision"], 3u);
  EXPECT_EQ(rows[3]["confirmed_cut"], true);
  EXPECT_EQ(rows[3]["confirmed_cut_count"], 1u);

  const float first_center = rows[0]["center"].get<float>();
  const float first_scale = rows[0]["latched_scale"].get<float>();
  EXPECT_NEAR(rows[1]["center"].get<float>(), first_center, 2.0e-5f);
  EXPECT_NEAR(rows[2]["center"].get<float>(), first_center, 2.0e-5f);
  EXPECT_NEAR(rows[1]["latched_scale"].get<float>(), first_scale, 2.0e-5f);
  EXPECT_NEAR(rows[2]["latched_scale"].get<float>(), first_scale, 2.0e-5f);
  EXPECT_FLOAT_EQ(rows[0]["convergence_curve"].get<float>(), 0.0f);
  EXPECT_FLOAT_EQ(rows[3]["convergence_curve"].get<float>(), 0.0f);
  for (const auto &row : rows) {
    if (row["camera_valid"].get<bool>()) {
      EXPECT_NEAR(row["latched_scale"].get<float>(),
                  calibration.raw_coordinate_scale, 2.0e-6f);
    }
  }

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
  EXPECT_EQ(rows[7]["camera_valid"], false);
  EXPECT_FLOAT_EQ(rows[7]["center"].get<float>(), 0.0f);
  EXPECT_FLOAT_EQ(outputs[7].encoded_minimum, 0.5f);
  EXPECT_FLOAT_EQ(outputs[7].encoded_maximum, 0.5f);
  EXPECT_EQ(rows[8]["frame_valid"], true);
  EXPECT_EQ(rows[8]["camera_valid"], true);
  EXPECT_NEAR(rows[8]["center"].get<float>(),
              rows[8]["observed_mean"].get<float>(), 2.0e-5f);

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
  tampered_contract["calibrated_defaults"]["far_tau"] = 0.151f;
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
  invalid["mapping_config"]["near_log_tau"] = 1.9f;
  expect_rejected(invalid, "mismatched-fixed-near-log-tau");
  invalid = manifest;
  invalid["source_color_mode"] = 3u;
  expect_rejected(invalid, "unknown-source-color-mode");
  invalid = manifest;
  invalid["source_linear_scale"] = 4.0f;
  expect_rejected(invalid, "nonlinear-scale-on-srgb-source");

  auto identity_replay = sbs_bench::depth_coordinate_v2_gpu_replay::create(
    warp.device.Get(), warp.context.Get(), manifest_path, error);
  ASSERT_NE(identity_replay, nullptr) << error;
  sbs_bench::depth_coordinate_v2_gpu_frame identity_output;
  EXPECT_FALSE(identity_replay->dispatch(
    0u,
    "00001",
    source_srv.Get(),
    2u,
    1.0f,
    test_source_sha256(),
    identity_output,
    error
  ));
  EXPECT_FALSE(error.empty());
  EXPECT_FALSE(identity_replay->dispatch(
    0u,
    "00001",
    source_srv.Get(),
    0u,
    4.0f,
    test_source_sha256(),
    identity_output,
    error
  ));
  EXPECT_FALSE(error.empty());
  EXPECT_TRUE(identity_replay->dispatch(
    0u,
    "00001",
    source_srv.Get(),
    0u,
    1.0f,
    test_source_sha256(),
    identity_output,
    error
  )) << error;
}

#endif  // _WIN32
