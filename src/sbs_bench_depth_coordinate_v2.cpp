/**
 * @file src/sbs_bench_depth_coordinate_v2.cpp
 * @brief Native, persistent-GPU whole-sequence depth-coordinate-v2 replay for sbs-bench.
 */
#include "sbs_bench_depth_coordinate_v2.h"

#ifdef _WIN32

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <charconv>
  #include <cmath>
  #include <cstdint>
  #include <cstring>
  #include <fstream>
  #include <iomanip>
  #include <limits>
  #include <locale>
  #include <optional>
  #include <sstream>
  #include <utility>

  #include <d3dcompiler.h>
  #include <nlohmann/json.hpp>

  #include "crypto.h"
  #include "depth_coordinate_v2.h"
  #include "generated/sbs_adaptive_state_contract.h"
  #include "host_sbs_shader_cache.h"
  #include "host_sbs_v2_gpu_executor.h"

  #ifndef SUNSHINE_SHADERS_DIR
    #define SUNSHINE_SHADERS_DIR SUNSHINE_ASSETS_DIR "/shaders/directx"
  #endif

using Microsoft::WRL::ComPtr;

namespace sbs_bench {
  namespace fs = std::filesystem;
  namespace v2 = models::depth_coordinate_v2;
  namespace v2_gpu = models::host_sbs_v2_gpu;
  namespace shader_cache = models::host_sbs_shader_cache;

  namespace {
    constexpr std::uint32_t manifest_schema = 10u;
    constexpr std::string_view manifest_mode =
      "depth-coordinate-v2-production-gpu-sequence-v12";
    constexpr float direct_container_limit = v2::direct_container_limit;

    const std::array<const char *, 38> trace_field_names {{
      "frame_id",
      "input_source_frame_id",
      "rendered_source_frame_id",
      "calibration_revision",
      "confirmed_cut_count",
      "confirmed_cut",
      "cut_attribution",
      "input_valid",
      "frame_valid",
      "camera_valid",
      "collapsed",
      "center",
      "inverse_scale",
      "latched_scale",
      "convergence_curve",
      "requested_gain",
      "container_scale",
      "effective_gain",
      "observed_mean",
      "observed_std",
      "observed_raw_minimum",
      "observed_raw_maximum",
      "candidate_center_drift_u",
      "predicted_zero_translation_source_u",
      "pre_limiter_max_abs_source_u",
      "vertical_majorant_raised_fraction",
      "vertical_majorant_max_raise_source_u",
      "final_max_abs_source_u",
      "conditioner_raised_fraction",
      "conditioner_max_raise_source_u",
      "conditioner_lowered_fraction",
      "conditioner_max_lower_source_u",
      "final_horizontal_slope_max",
      "final_vertical_shear_max",
      "order_sha256",
      "vertical_majorant_sha256",
      "vertical_conditioned_sha256",
      "parallax_sha256",
    }};

    std::string sha256_hex(std::string_view bytes) {
      static constexpr char hex[] = "0123456789abcdef";
      const auto digest = crypto::hash(bytes);
      std::string encoded;
      encoded.reserve(digest.size() * 2u);
      for (const std::uint8_t byte : digest) {
        encoded.push_back(hex[byte >> 4u]);
        encoded.push_back(hex[byte & 0x0fu]);
      }
      return encoded;
    }

    bool read_file_bytes(const fs::path &path, std::string &bytes) {
      std::ifstream stream(path, std::ios::binary);
      if (!stream) {
        return false;
      }
      stream.seekg(0, std::ios::end);
      const auto end = stream.tellg();
      if (end < 0 || static_cast<std::uint64_t>(end) >
                       std::numeric_limits<std::size_t>::max()) {
        return false;
      }
      bytes.resize(static_cast<std::size_t>(end));
      stream.seekg(0, std::ios::beg);
      if (!bytes.empty()) {
        stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      }
      return stream.good() || (stream.eof() && stream.gcount() ==
                                              static_cast<std::streamsize>(bytes.size()));
    }

    bool sha256_is_lower_hex(const std::string &value) {
      return value.size() == 64u && std::all_of(
        value.begin(), value.end(), [](const unsigned char c) {
          return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
    }

    bool finite_number(const nlohmann::json &value, float &result) {
      if (!value.is_number()) {
        return false;
      }
      const double candidate = value.get<double>();
      if (!std::isfinite(candidate) ||
          candidate < -std::numeric_limits<float>::max() ||
          candidate > std::numeric_limits<float>::max()) {
        return false;
      }
      result = static_cast<float>(candidate);
      return std::isfinite(result);
    }

    bool uint32_number(const nlohmann::json &value, std::uint32_t &result) {
      if (!value.is_number_unsigned()) {
        return false;
      }
      const auto candidate = value.get<std::uint64_t>();
      if (candidate > std::numeric_limits<std::uint32_t>::max()) {
        return false;
      }
      result = static_cast<std::uint32_t>(candidate);
      return true;
    }

    bool relative_filename(const std::string &value) {
      const fs::path path = value;
      return !value.empty() && !path.is_absolute() && path == path.filename() &&
             value != "." && value != "..";
    }

    bool safe_relative_path(const std::string &value) {
      const fs::path path = value;
      return !value.empty() && !path.is_absolute() &&
             std::none_of(path.begin(), path.end(), [](const fs::path &part) {
               return part == "..";
             });
    }

    ComPtr<ID3DBlob> compile_encode_shader(std::string &error) {
      // Interchange only. The controller and conditioning remain exclusively in the seven
      // production V2 coordinate shaders; this pass changes signed [-.04,+.04] storage into the
      // existing direct-renderer's [0,1] storage without a GPU->CPU->GPU round trip.
      static constexpr char source[] = R"(
Texture2D<float> SignedParallax : register(t0);
RWTexture2D<float> EncodedParallax : register(u0);
cbuffer EncodeConstants : register(b2) { float source_u_limit; float3 reserved; };
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    SignedParallax.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) return;
    EncodedParallax[id.xy] = 0.5f + SignedParallax[id.xy] / (2.0f * source_u_limit);
}
)";
      ComPtr<ID3DBlob> blob;
      ComPtr<ID3DBlob> diagnostics;
      const HRESULT result = D3DCompile(
        source,
        sizeof(source) - 1u,
        "sbs-bench-v2-interchange-encode",
        nullptr,
        nullptr,
        "main",
        "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &blob,
        &diagnostics
      );
      if (FAILED(result) || !blob) {
        error = "cannot compile v2 interchange encoder: " +
          (diagnostics ?
             std::string(
               static_cast<const char *>(diagnostics->GetBufferPointer()),
               diagnostics->GetBufferSize()
             ) :
             "D3DCompile failed");
        return nullptr;
      }
      return blob;
    }

    template<class T>
    bool create_immutable_constant_buffer(
      ID3D11Device *device,
      const T &value,
      ComPtr<ID3D11Buffer> &buffer
    ) {
      static_assert(sizeof(T) % 16u == 0u);
      D3D11_BUFFER_DESC desc {};
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.ByteWidth = static_cast<UINT>(sizeof(T));
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA initial {&value, 0, 0};
      return SUCCEEDED(device->CreateBuffer(&desc, &initial, &buffer));
    }

    bool create_structured_buffer(
      ID3D11Device *device,
      const UINT stride,
      const UINT count,
      const void *initial_data,
      const bool writable,
      ComPtr<ID3D11Buffer> &buffer,
      ComPtr<ID3D11ShaderResourceView> &srv,
      ComPtr<ID3D11UnorderedAccessView> &uav
    ) {
      if (!stride || !count || count > std::numeric_limits<UINT>::max() / stride) {
        return false;
      }
      D3D11_BUFFER_DESC desc {};
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.ByteWidth = stride * count;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE |
                       (writable ? D3D11_BIND_UNORDERED_ACCESS : 0u);
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = stride;
      D3D11_SUBRESOURCE_DATA initial {initial_data, 0, 0};
      if (FAILED(device->CreateBuffer(
            &desc, initial_data ? &initial : nullptr, &buffer
          )) ||
          FAILED(device->CreateShaderResourceView(buffer.Get(), nullptr, &srv))) {
        return false;
      }
      return !writable || SUCCEEDED(
        device->CreateUnorderedAccessView(buffer.Get(), nullptr, &uav));
    }

    bool create_float_texture(
      ID3D11Device *device,
      const UINT width,
      const UINT height,
      ComPtr<ID3D11Texture2D> &texture,
      ComPtr<ID3D11ShaderResourceView> &srv,
      ComPtr<ID3D11UnorderedAccessView> &uav
    ) {
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = 1u;
      desc.ArraySize = 1u;
      desc.Format = DXGI_FORMAT_R32_FLOAT;
      desc.SampleDesc.Count = 1u;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      return SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &texture)) &&
             SUCCEEDED(device->CreateShaderResourceView(texture.Get(), nullptr, &srv)) &&
             SUCCEEDED(device->CreateUnorderedAccessView(texture.Get(), nullptr, &uav));
    }

    bool create_uint_texture(
      ID3D11Device *device,
      const UINT width,
      const UINT height,
      ComPtr<ID3D11Texture2D> &texture,
      ComPtr<ID3D11ShaderResourceView> &srv,
      ComPtr<ID3D11UnorderedAccessView> &uav
    ) {
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = 1u;
      desc.ArraySize = 1u;
      desc.Format = DXGI_FORMAT_R32_UINT;
      desc.SampleDesc.Count = 1u;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      return SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &texture)) &&
             SUCCEEDED(device->CreateShaderResourceView(texture.Get(), nullptr, &srv)) &&
             SUCCEEDED(device->CreateUnorderedAccessView(texture.Get(), nullptr, &uav));
    }

    std::vector<std::uint32_t> read_buffer_words(
      ID3D11Device *device,
      ID3D11DeviceContext *context,
      ID3D11Buffer *source,
      const std::size_t word_count
    ) {
      if (!source || word_count == 0u ||
          word_count > std::numeric_limits<UINT>::max() / sizeof(std::uint32_t)) {
        return {};
      }
      D3D11_BUFFER_DESC desc {};
      source->GetDesc(&desc);
      if (desc.ByteWidth < word_count * sizeof(std::uint32_t)) {
        return {};
      }
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = 0u;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      desc.MiscFlags = 0u;
      ComPtr<ID3D11Buffer> staging;
      if (FAILED(device->CreateBuffer(&desc, nullptr, &staging))) {
        return {};
      }
      context->CopyResource(staging.Get(), source);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return {};
      }
      std::vector<std::uint32_t> values(word_count);
      std::memcpy(values.data(), mapped.pData, values.size() * sizeof(values.front()));
      context->Unmap(staging.Get(), 0);
      return values;
    }

    std::vector<float> read_float_texture(
      ID3D11Device *device,
      ID3D11DeviceContext *context,
      ID3D11Texture2D *source,
      const UINT width,
      const UINT height
    ) {
      if (!source || !width || !height) {
        return {};
      }
      D3D11_TEXTURE2D_DESC desc {};
      source->GetDesc(&desc);
      if (desc.Width != width || desc.Height != height ||
          desc.Format != DXGI_FORMAT_R32_FLOAT) {
        return {};
      }
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = 0u;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      desc.MiscFlags = 0u;
      ComPtr<ID3D11Texture2D> staging;
      if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging))) {
        return {};
      }
      context->CopyResource(staging.Get(), source);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return {};
      }
      std::vector<float> values(static_cast<std::size_t>(width) * height);
      for (UINT y = 0; y < height; ++y) {
        const auto *row = reinterpret_cast<const float *>(
          static_cast<const std::byte *>(mapped.pData) +
          static_cast<std::size_t>(y) * mapped.RowPitch
        );
        std::copy_n(
          row,
          width,
          values.begin() + static_cast<std::size_t>(y) * width
        );
      }
      context->Unmap(staging.Get(), 0);
      return values;
    }

    float v2_curve(const float coordinate, const float far_tau, const float near_log_tau) {
      if (coordinate < 0.0f) {
        return far_tau * std::expm1(coordinate / far_tau);
      }
      if (coordinate <= 1.0f) {
        return coordinate;
      }
      return 1.0f + near_log_tau * std::log1p((coordinate - 1.0f) / near_log_tau);
    }

    struct input_frame {
      std::string frame_id;
      std::uint64_t frame_number = 0u;
      fs::path raw_path;
      std::string raw_sha256;
      std::string source_sha256;
      std::uint32_t cut_generation = 0u;
      bool cut_pulse = false;
    };
  }  // namespace

  struct depth_coordinate_v2_gpu_replay::impl {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    fs::path manifest_path;
    std::string manifest_sha256;
    std::string model_name;
    std::string cut_source;
    UINT width = 0u;
    UINT height = 0u;
    UINT reduce_groups = 0u;
    nlohmann::ordered_json calibration_contract;
    nlohmann::ordered_json mapping_config;
    v2::constants_t constants {};
    std::vector<input_frame> frames;
    std::vector<nlohmann::ordered_json> trace_rows;

    ComPtr<ID3D11ComputeShader> moments_shader;
    ComPtr<ID3D11ComputeShader> frame_shader;
    ComPtr<ID3D11ComputeShader> state_shader;
    ComPtr<ID3D11ComputeShader> map_shader;
    ComPtr<ID3D11ComputeShader> coordinate_diagnostic_shader;
    ComPtr<ID3D11ComputeShader> vertical_limit_shader;
    ComPtr<ID3D11ComputeShader> limit_shader;
    ComPtr<ID3D11ComputeShader> encode_shader;
    ComPtr<ID3D11Buffer> common_constants;
    ComPtr<ID3D11Buffer> v2_constants;
    ComPtr<ID3D11Buffer> near_constants;
    ComPtr<ID3D11Buffer> encode_constants;
    ComPtr<ID3D11Buffer> raw_buffer;
    ComPtr<ID3D11ShaderResourceView> raw_srv;
    ComPtr<ID3D11Buffer> partial_buffer;
    ComPtr<ID3D11ShaderResourceView> partial_srv;
    ComPtr<ID3D11UnorderedAccessView> partial_uav;
    ComPtr<ID3D11Buffer> frame_buffer;
    ComPtr<ID3D11ShaderResourceView> frame_srv;
    ComPtr<ID3D11UnorderedAccessView> frame_uav;
    ComPtr<ID3D11Buffer> normalization_buffer;
    ComPtr<ID3D11UnorderedAccessView> normalization_uav;
    ComPtr<ID3D11Buffer> state_buffer;
    ComPtr<ID3D11ShaderResourceView> state_srv;
    ComPtr<ID3D11UnorderedAccessView> state_uav;
    ComPtr<ID3D11Buffer> cut_state_buffer;
    ComPtr<ID3D11ShaderResourceView> cut_state_srv;
    ComPtr<ID3D11Buffer> dummy_minmax_buffer;
    ComPtr<ID3D11ShaderResourceView> dummy_minmax_srv;
    ComPtr<ID3D11Buffer> dummy_current_model_input_buffer;
    ComPtr<ID3D11ShaderResourceView> dummy_current_model_input_srv;
    ComPtr<ID3D11Buffer> dummy_previous_model_input_buffer;
    ComPtr<ID3D11ShaderResourceView> dummy_previous_model_input_srv;
    ComPtr<ID3D11UnorderedAccessView> dummy_previous_model_input_uav;
    ComPtr<ID3D11Buffer> dummy_current_appearance_buffer;
    ComPtr<ID3D11ShaderResourceView> dummy_current_appearance_srv;
    ComPtr<ID3D11Buffer> dummy_previous_appearance_buffer;
    ComPtr<ID3D11ShaderResourceView> dummy_previous_appearance_srv;
    ComPtr<ID3D11UnorderedAccessView> dummy_previous_appearance_uav;
    ComPtr<ID3D11Buffer> dummy_history_owner_buffer;
    ComPtr<ID3D11ShaderResourceView> dummy_history_owner_srv;
    ComPtr<ID3D11UnorderedAccessView> dummy_history_owner_uav;
    ComPtr<ID3D11Texture2D> dummy_exclusion_texture;
    ComPtr<ID3D11ShaderResourceView> dummy_exclusion_srv;
    ComPtr<ID3D11UnorderedAccessView> dummy_exclusion_uav;
    ComPtr<ID3D11Texture2D> dummy_previous_exclusion_texture;
    ComPtr<ID3D11ShaderResourceView> dummy_previous_exclusion_srv;
    ComPtr<ID3D11UnorderedAccessView> dummy_previous_exclusion_uav;
    ComPtr<ID3D11Texture2D> dummy_current_depth_texture;
    ComPtr<ID3D11ShaderResourceView> dummy_current_depth_srv;
    ComPtr<ID3D11UnorderedAccessView> dummy_current_depth_uav;
    ComPtr<ID3D11Texture2D> dummy_previous_depth_texture;
    ComPtr<ID3D11ShaderResourceView> dummy_previous_depth_srv;
    ComPtr<ID3D11UnorderedAccessView> dummy_previous_depth_uav;
    ComPtr<ID3D11Texture2D> coordinate_texture;
    ComPtr<ID3D11ShaderResourceView> coordinate_srv;
    ComPtr<ID3D11UnorderedAccessView> coordinate_uav;
    ComPtr<ID3D11Texture2D> candidate_texture;
    ComPtr<ID3D11ShaderResourceView> candidate_srv;
    ComPtr<ID3D11UnorderedAccessView> candidate_uav;
    ComPtr<ID3D11Texture2D> vertical_majorant_texture;
    ComPtr<ID3D11ShaderResourceView> vertical_majorant_srv;
    ComPtr<ID3D11UnorderedAccessView> vertical_majorant_uav;
    ComPtr<ID3D11Texture2D> vertical_conditioned_texture;
    ComPtr<ID3D11ShaderResourceView> vertical_conditioned_srv;
    ComPtr<ID3D11UnorderedAccessView> vertical_conditioned_uav;
    ComPtr<ID3D11Texture2D> final_texture;
    ComPtr<ID3D11ShaderResourceView> final_srv;
    ComPtr<ID3D11UnorderedAccessView> final_uav;
    ComPtr<ID3D11Texture2D> encoded_texture;
    ComPtr<ID3D11ShaderResourceView> encoded_srv;
    ComPtr<ID3D11UnorderedAccessView> encoded_uav;

    bool load_manifest(std::string &error) {
      std::string bytes;
      if (!read_file_bytes(manifest_path, bytes)) {
        error = "cannot read v2 GPU replay manifest";
        return false;
      }
      manifest_sha256 = sha256_hex(bytes);
      nlohmann::ordered_json root;
      try {
        root = nlohmann::ordered_json::parse(bytes);
      } catch (const nlohmann::json::exception &exc) {
        error = std::string("invalid v2 GPU replay manifest JSON: ") + exc.what();
        return false;
      }
      const std::array<const char *, 9> root_keys {{
        "schema", "mode", "calibration_contract", "model_identity", "raw_shape",
        "source_shape", "mapping_config", "cut_source", "frames",
      }};
      if (!root.is_object() || root.size() != root_keys.size() ||
          std::any_of(root_keys.begin(), root_keys.end(), [&](const char *key) {
            return !root.contains(key);
          }) ||
          root["schema"] != manifest_schema || root["mode"] != manifest_mode) {
        error = "v2 GPU replay manifest has missing or unknown root semantics";
        return false;
      }
      std::uint32_t calibration_schema = 0u;
      const auto &calibration_evidence = root["calibration_contract"];
      if (!calibration_evidence.is_object() || calibration_evidence.size() != 3u) {
        error = "v2 GPU replay manifest has invalid calibration-contract object";
        return false;
      }
      if (!calibration_evidence.contains("file") ||
          !calibration_evidence["file"].is_string() ||
          calibration_evidence["file"].get<std::string>() !=
            "contracts/depth-coordinate-v2-v1.json") {
        error = "v2 GPU replay manifest has invalid calibration-contract file";
        return false;
      }
      if (!calibration_evidence.contains("schema") ||
          !uint32_number(calibration_evidence["schema"], calibration_schema) ||
          calibration_schema != v2::contract_schema) {
        error = "v2 GPU replay manifest has invalid calibration-contract schema";
        return false;
      }
      if (!calibration_evidence.contains("sha256") ||
          !calibration_evidence["sha256"].is_string() ||
          !sha256_is_lower_hex(calibration_evidence["sha256"].get<std::string>())) {
        error = "v2 GPU replay manifest has invalid calibration-contract sha256";
        return false;
      }
      calibration_contract = calibration_evidence;
      const auto calibration_file = calibration_evidence["file"].get<std::string>();
      if (!safe_relative_path(calibration_file)) {
        error = "v2 GPU replay calibration-contract path is unsafe";
        return false;
      }
      std::string calibration_bytes;
      const fs::path calibration_path = manifest_path.parent_path() / calibration_file;
      if (!read_file_bytes(calibration_path, calibration_bytes) ||
          sha256_hex(calibration_bytes) !=
            root["calibration_contract"]["sha256"].get<std::string>()) {
        error = "v2 GPU replay calibration-contract file does not match its exact digest";
        return false;
      }
      try {
        // The manifest digest authenticates transport, but cannot authenticate itself. Bind the
        // copied JSON's normalized semantics to the digest compiled into this executable.
        const auto semantic_contract = nlohmann::json::parse(calibration_bytes);
        const auto canonical = semantic_contract.dump(-1, ' ', true);
        if (sha256_hex(canonical) != v2::contract_canonical_sha256) {
          error = "v2 GPU replay calibration contract differs from the compiled canonical digest";
          return false;
        }
      } catch (const nlohmann::json::exception &exc) {
        error = std::string("invalid copied v2 calibration contract JSON: ") + exc.what();
        return false;
      }

      const auto &identity = root["model_identity"];
      const std::array<const char *, 6> identity_keys {{
        "calibration_id", "model", "depth_model_url", "onnx_sha256",
        "preprocess_profile", "preprocess_source_closure_sha256",
      }};
      if (!identity.is_object() || identity.size() != identity_keys.size() ||
          std::any_of(identity_keys.begin(), identity_keys.end(), [&](const char *key) {
            return !identity.contains(key) || !identity[key].is_string();
          })) {
        error = "v2 GPU replay manifest has invalid model identity";
        return false;
      }
      const auto &shape = root["raw_shape"];
      std::uint32_t manifest_width = 0u;
      std::uint32_t manifest_height = 0u;
      if (!shape.is_object() || shape.size() != 4u ||
          !shape.contains("dtype") || !shape["dtype"].is_string() ||
          shape["dtype"].get<std::string>() != "float32-le" ||
          !shape.contains("layout") || !shape["layout"].is_string() ||
          shape["layout"].get<std::string>() != "row-major" ||
          !shape.contains("width") || !shape.contains("height") ||
          !uint32_number(shape["width"], manifest_width) ||
          !uint32_number(shape["height"], manifest_height)) {
        error = "v2 GPU replay manifest has invalid raw-shape contract";
        return false;
      }
      width = static_cast<UINT>(manifest_width);
      height = static_cast<UINT>(manifest_height);
      const auto &source_shape = root["source_shape"];
      std::uint32_t manifest_source_width = 0u;
      std::uint32_t manifest_source_height = 0u;
      if (!source_shape.is_object() || source_shape.size() != 2u ||
          !source_shape.contains("width") || !source_shape.contains("height") ||
          !uint32_number(source_shape["width"], manifest_source_width) ||
          !uint32_number(source_shape["height"], manifest_source_height) ||
          manifest_source_width == 0u || manifest_source_height == 0u ||
          manifest_source_width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
          manifest_source_height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION) {
        error = "v2 GPU replay manifest has invalid source-shape contract";
        return false;
      }
      model_name = identity["model"].get<std::string>();
      if (!width || !height ||
          static_cast<std::uint64_t>(width) * height >
            std::numeric_limits<UINT>::max() ||
          !v2::capture_identity_is_calibrated(
            model_name,
            identity["depth_model_url"].get<std::string>(),
            identity["onnx_sha256"].get<std::string>(),
            identity["preprocess_profile"].get<std::string>(),
            identity["preprocess_source_closure_sha256"].get<std::string>(),
            width,
            height
          )) {
        error = "v2 GPU replay model identity or raw shape is outside the calibrated allowlist";
        return false;
      }
      const auto calibration_id = identity["calibration_id"].get<std::string>();
      const auto calibration = std::find_if(
        v2::model_calibrations.begin(),
        v2::model_calibrations.end(),
        [&](const auto &candidate) {
          return candidate.calibration_id == calibration_id &&
                 candidate.depth_model == model_name &&
                 candidate.depth_model_url ==
                   identity["depth_model_url"].get<std::string>() &&
                 candidate.onnx_sha256 == identity["onnx_sha256"].get<std::string>() &&
                 candidate.preprocess.profile ==
                   identity["preprocess_profile"].get<std::string>() &&
                 candidate.preprocess.source_closure_sha256 ==
                   identity["preprocess_source_closure_sha256"].get<std::string>() &&
                 v2::model_calibration_supports_shape(candidate, width, height);
        }
      );
      if (calibration == v2::model_calibrations.end()) {
        error = "v2 GPU replay calibration ID is not generated into the native contract";
        return false;
      }

      mapping_config = root["mapping_config"];
      const std::array<const char *, 10> mapping_keys {{
        "raw_coordinate_scale", "collapse_abs_epsilon", "far_tau", "near_log_tau",
        "pop_strength", "gain_per_pop", "max_horizontal_slope",
        "max_vertical_shear", "vertical_majorant_share", "direct_container_limit",
      }};
      if (!mapping_config.is_object() || mapping_config.size() != mapping_keys.size() ||
          std::any_of(mapping_keys.begin(), mapping_keys.end(), [&](const char *key) {
            return !mapping_config.contains(key);
          })) {
        error = "v2 GPU replay manifest has invalid mapping configuration";
        return false;
      }
      float pop_strength = 0.0f;
      float gain_per_pop = 0.0f;
      float max_vertical_shear = 0.0f;
      float vertical_majorant_share = 0.0f;
      if (!finite_number(mapping_config["raw_coordinate_scale"],
                         constants.raw_coordinate_scale) ||
          !finite_number(mapping_config["collapse_abs_epsilon"],
                         constants.collapse_abs_epsilon) ||
          !finite_number(mapping_config["far_tau"], constants.far_tau) ||
          !finite_number(mapping_config["near_log_tau"], constants.near_log_tau) ||
          !finite_number(mapping_config["pop_strength"], pop_strength) ||
          !finite_number(mapping_config["gain_per_pop"], gain_per_pop) ||
          !finite_number(mapping_config["max_horizontal_slope"],
                         constants.max_horizontal_slope) ||
          !finite_number(mapping_config["max_vertical_shear"], max_vertical_shear) ||
          !finite_number(
            mapping_config["vertical_majorant_share"],
            vertical_majorant_share
          ) ||
          !finite_number(mapping_config["direct_container_limit"],
                         constants.direct_container_limit)) {
        error = "v2 GPU replay mapping values must be finite float32-compatible numbers";
        return false;
      }
      constants.requested_gain = pop_strength * gain_per_pop;
      constants.convergence_curve_default = v2::convergence_curve_default;
      // This harness proves the exact production V2 algorithm, not a family of similarly shaped
      // experiments. The requested pop authority may vary within the admitted range;
      // every algorithm/safety constant must be the generated contract value bit-for-bit.
      if (constants.raw_coordinate_scale != calibration->raw_coordinate_scale ||
          constants.collapse_abs_epsilon != v2::collapse_abs_epsilon ||
          constants.far_tau != v2::far_tau ||
          constants.near_log_tau != v2::near_log_tau ||
          gain_per_pop != v2::gain_per_pop ||
          constants.max_horizontal_slope != v2::max_horizontal_slope ||
          max_vertical_shear != v2::max_vertical_shear ||
          vertical_majorant_share != v2::vertical_majorant_share ||
          constants.direct_container_limit != v2::direct_container_limit ||
          constants.convergence_curve_default != v2::convergence_curve_default ||
          pop_strength < 0.25f || pop_strength > 2.0f ||
          !v2::parallax_runtime_constants_are_valid(
            constants.raw_coordinate_scale,
            pop_strength,
            constants.requested_gain
          )) {
        error = "v2 GPU replay mapping differs from the generated production V2 contract";
        return false;
      }

      if (!root["cut_source"].is_string() || root["cut_source"].get<std::string>().empty()) {
        error = "v2 GPU replay manifest must identify cut-generation authority";
        return false;
      }
      cut_source = root["cut_source"].get<std::string>();
      const auto &frame_values = root["frames"];
      if (!frame_values.is_array() || frame_values.empty()) {
        error = "v2 GPU replay manifest requires a non-empty ordered frame list";
        return false;
      }
      const fs::path root_directory = manifest_path.parent_path();
      std::optional<std::uint32_t> previous_generation;
      std::optional<std::uint64_t> previous_frame_number;
      for (const auto &value : frame_values) {
        const std::array<const char *, 6> frame_keys {{
          "frame_id", "raw_file", "raw_sha256", "source_sha256", "hard_cut_count",
          "hard_cut_pulse",
        }};
        if (!value.is_object() || value.size() != frame_keys.size() ||
            std::any_of(frame_keys.begin(), frame_keys.end(), [&](const char *key) {
              return !value.contains(key);
            }) ||
            !value["frame_id"].is_string() || !value["raw_file"].is_string() ||
            !value["raw_sha256"].is_string() || !value["source_sha256"].is_string() ||
            !value["hard_cut_count"].is_number_unsigned() ||
            !value["hard_cut_pulse"].is_boolean()) {
          error = "v2 GPU replay manifest has an invalid frame record";
          return false;
        }
        input_frame frame;
        frame.frame_id = value["frame_id"].get<std::string>();
        const auto parsed = std::from_chars(
          frame.frame_id.data(),
          frame.frame_id.data() + frame.frame_id.size(),
          frame.frame_number
        );
        const auto raw_file = value["raw_file"].get<std::string>();
        frame.raw_sha256 = value["raw_sha256"].get<std::string>();
        frame.source_sha256 = value["source_sha256"].get<std::string>();
        if (!uint32_number(value["hard_cut_count"], frame.cut_generation)) {
          error = "v2 GPU replay hard-cut generation is outside uint32";
          return false;
        }
        frame.cut_pulse = value["hard_cut_pulse"].get<bool>();
        if (frame.frame_id.empty() || frame.frame_id.size() > 10u ||
            parsed.ec != std::errc {} ||
            parsed.ptr != frame.frame_id.data() + frame.frame_id.size() ||
            (previous_frame_number && frame.frame_number <= *previous_frame_number) ||
            !relative_filename(raw_file) || raw_file != "raw_" + frame.frame_id + ".f32" ||
            !sha256_is_lower_hex(frame.raw_sha256) ||
            !sha256_is_lower_hex(frame.source_sha256) ||
            frame.cut_generation > sbs_adaptive_state::counter_max) {
          error = "v2 GPU replay frame identity/hash/generation is invalid";
          return false;
        }
        if (previous_generation) {
          const auto delta = frame.cut_generation - *previous_generation;
          if (frame.cut_generation < *previous_generation || delta > 1u) {
            error = "v2 GPU replay hard-cut generation is not monotonic";
            return false;
          }
        }
        frame.raw_path = root_directory / raw_file;
        std::error_code ec;
        if (!fs::is_regular_file(frame.raw_path, ec) || ec ||
            fs::file_size(frame.raw_path, ec) !=
              static_cast<std::uint64_t>(width) * height * sizeof(float) || ec) {
          error = "v2 GPU replay raw field is missing or has the wrong byte size: " +
                  frame.raw_path.string();
          return false;
        }
        frames.push_back(std::move(frame));
        previous_generation = frames.back().cut_generation;
        previous_frame_number = frames.back().frame_number;
      }
      return true;
    }

    bool initialize_gpu(std::string &error) {
      reduce_groups = std::min<UINT>(
        64u,
        std::max<UINT>(1u, (width * height + 255u) / 256u)
      );
      const fs::path shader_root = SUNSHINE_SHADERS_DIR;
      static_assert(v2::shader_source_closure_schema == shader_cache::source_closure_schema);
      static_assert(v2::shader_source_compile_flags == shader_cache::shader_compile_flags);
      static_assert(v2::shader_source_macro_count == 0u);
      const auto shader_sources = shader_cache::snapshot_sources(
        shader_root,
        shader_cache::parallax_v2_producer_specs
      );
      if (!shader_sources ||
          shader_cache::source_closure_sha256(shader_sources) !=
            v2::shader_source_closure_sha256) {
        error = "production V2 shader source closure does not match the generated V2 contract";
        return false;
      }
      struct shader_target {
        const shader_cache::shader_spec *spec;
        ComPtr<ID3D11ComputeShader> *shader;
      };
      const std::array<shader_target, 6> targets {{
        {&shader_cache::depth_coordinate_v2_moments, &moments_shader},
        {&shader_cache::depth_coordinate_v2_frame_resolve, &frame_shader},
        {&shader_cache::depth_coordinate_v2_state_resolve, &state_shader},
        {&shader_cache::depth_coordinate_v2_map, &map_shader},
        {&shader_cache::depth_coordinate_v2_vertical_limit, &vertical_limit_shader},
        {&shader_cache::depth_coordinate_v2_limit, &limit_shader},
      }};
      for (const auto &target : targets) {
        const auto bytecode = shader_cache::get(shader_sources, *target.spec);
        if (!bytecode || bytecode->empty() || FAILED(device->CreateComputeShader(
              bytecode->data(), bytecode->size(), nullptr,
              target.shader->ReleaseAndGetAddressOf()
            )) || !*target.shader) {
          error = std::string("cannot create authenticated production V2 compute shader ") +
            std::string(target.spec->filename);
          return false;
        }
      }
      const auto diagnostic_sources = shader_cache::snapshot_sources(
        shader_root,
        shader_cache::parallax_v2_coordinate_diagnostic_specs
      );
      const auto diagnostic_bytecode = diagnostic_sources ?
        shader_cache::get(
          diagnostic_sources,
          shader_cache::depth_coordinate_v2_coordinate_diagnostic
        ) : nullptr;
      if (!diagnostic_bytecode || diagnostic_bytecode->empty() ||
          FAILED(device->CreateComputeShader(
            diagnostic_bytecode->data(),
            diagnostic_bytecode->size(),
            nullptr,
            coordinate_diagnostic_shader.ReleaseAndGetAddressOf()
          )) || !coordinate_diagnostic_shader) {
        error = "cannot create V2 canonical-coordinate diagnostic compute shader";
        return false;
      }
      auto encode_blob = compile_encode_shader(error);
      if (!encode_blob || FAILED(device->CreateComputeShader(
            encode_blob->GetBufferPointer(), encode_blob->GetBufferSize(), nullptr,
            &encode_shader
          )) || !encode_shader) {
        if (error.empty()) {
          error = "cannot create v2 interchange encode shader";
        }
        return false;
      }

      std::array<std::uint32_t, 16> common_words {};
      common_words[0] = width;
      common_words[1] = height;
      common_words[5] = reduce_groups * 256u;
      common_words[9] = 0u;
      common_words[10] = 0u;
      common_words[11] = width;
      common_words[12] = height;
      const std::array<float, 4> encode_words {{
        v2::direct_container_limit, 0.0f, 0.0f, 0.0f,
      }};
      const std::array<std::uint32_t, 20> near_words {};
      // The shared geometry shaders retain the live depth-cbuffer layout, but none reads the
      // color-mode slot. Freeze it to zero instead of manufacturing three equivalent buffers.
      common_words[2] = 0u;
      if (!create_immutable_constant_buffer(device.Get(), common_words, common_constants) ||
          !create_immutable_constant_buffer(device.Get(), constants, v2_constants) ||
          !create_immutable_constant_buffer(device.Get(), near_words, near_constants) ||
          !create_immutable_constant_buffer(device.Get(), encode_words, encode_constants)) {
        error = "cannot create v2 GPU replay constant buffers";
        return false;
      }
      ComPtr<ID3D11ShaderResourceView> unused_srv;
      ComPtr<ID3D11UnorderedAccessView> unused_uav;
      if (!create_structured_buffer(
            device.Get(), sizeof(float), width * height, nullptr, false,
            raw_buffer, raw_srv, unused_uav
          ) ||
          !create_structured_buffer(
            device.Get(), sizeof(float) * 4u, reduce_groups * 3u, nullptr, true,
            partial_buffer, partial_srv, partial_uav
          ) ||
          !create_structured_buffer(
            device.Get(), sizeof(float) * 4u,
            static_cast<UINT>(v2::frame_stats_vector_count), nullptr, true,
            frame_buffer, frame_srv, frame_uav
          ) ||
          !create_structured_buffer(
            device.Get(), sizeof(float) * 4u,
            static_cast<UINT>(v2::state_vector_count), v2::state_initial_words.data(), true,
            state_buffer, state_srv, state_uav
          )) {
        error = "cannot create v2 GPU replay structured resources";
        return false;
      }
      {
        const std::array<std::uint32_t, 4> normalization_identity {{
          0xFFFFFFFFu, 0u, 0u, 0u,
        }};
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = sizeof(normalization_identity);
        desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        D3D11_SUBRESOURCE_DATA initial {normalization_identity.data(), 0, 0};
        D3D11_UNORDERED_ACCESS_VIEW_DESC view {};
        view.Format = DXGI_FORMAT_R32_TYPELESS;
        view.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        view.Buffer.NumElements = static_cast<UINT>(normalization_identity.size());
        view.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        if (FAILED(device->CreateBuffer(&desc, &initial, &normalization_buffer)) ||
            FAILED(device->CreateUnorderedAccessView(
              normalization_buffer.Get(), &view, &normalization_uav
            ))) {
          error = "cannot create v2 GPU replay normalization output";
          return false;
        }
      }
      auto cut_state_words = sbs_adaptive_state::initial_words;
      const std::array<float, 4> invalid_minmax {{0.0f, 0.0f, 0.0f, 0.0f}};
      if (!create_structured_buffer(
            device.Get(), sizeof(float) * 4u,
            static_cast<UINT>(sbs_adaptive_state::vector_count), cut_state_words.data(), false,
            cut_state_buffer, cut_state_srv, unused_uav
          ) ||
          !create_structured_buffer(
            device.Get(), sizeof(float) * 4u, 1u, invalid_minmax.data(), false,
            dummy_minmax_buffer, dummy_minmax_srv, unused_uav
          ) ||
          !create_structured_buffer(
            device.Get(), sizeof(float), 3u * width * height, nullptr, false,
            dummy_current_model_input_buffer, dummy_current_model_input_srv, unused_uav
          ) ||
          !create_structured_buffer(
            device.Get(), sizeof(float), 3u * width * height, nullptr, true,
            dummy_previous_model_input_buffer, dummy_previous_model_input_srv,
            dummy_previous_model_input_uav
          ) ||
          !create_structured_buffer(
            device.Get(), sizeof(float), width * height, nullptr, false,
            dummy_current_appearance_buffer, dummy_current_appearance_srv, unused_uav
          ) ||
          !create_structured_buffer(
            device.Get(), sizeof(float), width * height, nullptr, true,
            dummy_previous_appearance_buffer, dummy_previous_appearance_srv,
            dummy_previous_appearance_uav
          ) ||
          !create_structured_buffer(
            device.Get(), sizeof(std::uint32_t), 10u, nullptr, true,
            dummy_history_owner_buffer, dummy_history_owner_srv, dummy_history_owner_uav
          ) ||
          !create_uint_texture(
            device.Get(), width, height,
            dummy_exclusion_texture, dummy_exclusion_srv, dummy_exclusion_uav
          ) ||
          !create_uint_texture(
            device.Get(), width, height,
            dummy_previous_exclusion_texture, dummy_previous_exclusion_srv,
            dummy_previous_exclusion_uav
          ) ||
          !create_float_texture(
            device.Get(), width, height,
            dummy_current_depth_texture, dummy_current_depth_srv, dummy_current_depth_uav
          ) ||
          !create_float_texture(
            device.Get(), width, height,
            dummy_previous_depth_texture, dummy_previous_depth_srv, dummy_previous_depth_uav
          ) ||
          !create_float_texture(
            device.Get(), width, height,
            coordinate_texture, coordinate_srv, coordinate_uav
          ) ||
          !create_float_texture(
            device.Get(), width, height,
            candidate_texture, candidate_srv, candidate_uav
          ) ||
          !create_float_texture(
            device.Get(), width, height,
            vertical_majorant_texture, vertical_majorant_srv, vertical_majorant_uav
          ) ||
          !create_float_texture(
            device.Get(), width, height,
            vertical_conditioned_texture,
            vertical_conditioned_srv,
            vertical_conditioned_uav
          ) ||
          !create_float_texture(
            device.Get(), width, height,
            final_texture, final_srv, final_uav
          ) ||
          !create_float_texture(
            device.Get(), width, height,
            encoded_texture, encoded_srv, encoded_uav
          )) {
        error = "cannot create V2 GPU replay textures or cut-state buffer";
        return false;
      }
      const UINT zero_uint[4] = {};
      const float zero_float[4] = {};
      context->ClearUnorderedAccessViewUint(dummy_exclusion_uav.Get(), zero_uint);
      context->ClearUnorderedAccessViewUint(dummy_previous_exclusion_uav.Get(), zero_uint);
      context->ClearUnorderedAccessViewFloat(dummy_current_depth_uav.Get(), zero_float);
      context->ClearUnorderedAccessViewFloat(dummy_previous_depth_uav.Get(), zero_float);
      return true;
    }

    void unbind(const UINT srv_count, const UINT uav_count) {
      ID3D11ShaderResourceView *null_srvs[8] = {};
      ID3D11UnorderedAccessView *null_uavs[6] = {};
      context->CSSetShaderResources(0, srv_count, null_srvs);
      context->CSSetUnorderedAccessViews(0, uav_count, null_uavs, nullptr);
    }

    bool load_raw(const input_frame &frame, std::vector<float> &values, std::string &error) {
      std::string bytes;
      if (!read_file_bytes(frame.raw_path, bytes) ||
          bytes.size() != static_cast<std::size_t>(width) * height * sizeof(float)) {
        error = "cannot read exact raw field " + frame.raw_path.string();
        return false;
      }
      const auto digest = sha256_hex(bytes);
      if (digest != frame.raw_sha256) {
        error = "raw field hash changed after manifest authentication: " +
                frame.raw_path.string();
        return false;
      }
      values.resize(static_cast<std::size_t>(width) * height);
      std::memcpy(values.data(), bytes.data(), bytes.size());
      return true;
    }

    bool dispatch_frame(
      const std::size_t sequence_index,
      const std::string_view expected_frame_id,
      const std::string_view expected_source_sha256,
      depth_coordinate_v2_gpu_frame &result,
      std::string &error
    ) {
      if (sequence_index != trace_rows.size() || sequence_index >= frames.size()) {
        error = "v2 GPU replay frames must be dispatched exactly once and in manifest order";
        return false;
      }
      const auto &frame = frames[sequence_index];
      if (expected_frame_id != frame.frame_id) {
        error = "source frame ID disagrees with v2 GPU replay manifest at sequence index " +
                std::to_string(sequence_index);
        return false;
      }
      if (expected_source_sha256 != frame.source_sha256) {
        error = "source provenance disagrees with v2 GPU replay manifest at sequence index " +
                std::to_string(sequence_index);
        return false;
      }
      std::vector<float> raw_values;
      if (!load_raw(frame, raw_values, error)) {
        return false;
      }
      context->UpdateSubresource(raw_buffer.Get(), 0, nullptr, raw_values.data(), 0, 0);
      const bool confirmed_cut_input = sequence_index > 0u && (
        frame.cut_pulse ||
        frame.cut_generation != frames[sequence_index - 1u].cut_generation
      );
      auto cut_state_words = sbs_adaptive_state::initial_words;
      cut_state_words[sbs_adaptive_state::index(
        sbs_adaptive_state::word_e::hard_cut_count)] = frame.cut_generation;
      cut_state_words[sbs_adaptive_state::index(
        sbs_adaptive_state::word_e::hard_cut_pulse)] =
          std::bit_cast<std::uint32_t>(frame.cut_pulse ? 1.0f : 0.0f);
      context->UpdateSubresource(
        cut_state_buffer.Get(), 0, nullptr, cut_state_words.data(), 0, 0);

      const v2_gpu::base_constants_t base_constants {
        .depth = common_constants.Get(),
        .coordinate_v2 = v2_constants.Get(),
      };
      const v2_gpu::moments_frame_command_t moments_frame_command {
        .constants = base_constants,
        .moments_shader = moments_shader.Get(),
        .frame_resolve_shader = frame_shader.Get(),
        .raw_depth = raw_srv.Get(),
        .tensor_exclusion = dummy_exclusion_srv.Get(),
        .partials_output = partial_uav.Get(),
        .partials = partial_srv.Get(),
        .frame_stats_output = frame_uav.Get(),
        .minmax_raw_output = normalization_uav.Get(),
        .moments_dispatch = v2_gpu::dispatch_command_t::direct(
          reduce_groups, 1u, 1u
        ),
        .frame_resolve_dispatch = v2_gpu::dispatch_command_t::direct(1u, 1u, 1u),
      };
      if (!v2_gpu::record_moments_frame(context.Get(), moments_frame_command)) {
        error = "shared V2 GPU executor rejected replay moments/frame operands";
        return false;
      }

      const v2_gpu::state_command_t state_command {
        .constants = base_constants,
        .shader = state_shader.Get(),
        .frame_stats = frame_srv.Get(),
        .cut_bridge_state = cut_state_srv.Get(),
        .state_output = state_uav.Get(),
        .dispatch = v2_gpu::dispatch_command_t::direct(1u, 1u, 1u),
      };
      if (!v2_gpu::record_state(context.Get(), state_command)) {
        error = "shared V2 GPU executor rejected replay state operands";
        return false;
      }

      ID3D11ShaderResourceView *coordinate_srvs[] = {
        raw_srv.Get(), state_srv.Get(), dummy_exclusion_srv.Get(),
      };

      // The canonical coordinate is a replay diagnostic, matching the live Dump-3D-only pass.
      context->CSSetShader(coordinate_diagnostic_shader.Get(), nullptr, 0);
      context->CSSetShaderResources(0, 3, coordinate_srvs);
      context->CSSetUnorderedAccessViews(0, 1, coordinate_uav.GetAddressOf(), nullptr);
      context->Dispatch((width + 15u) / 16u, (height + 15u) / 16u, 1u);
      unbind(3u, 1u);

      const v2_gpu::map_history_command_t map_history_command {
        .constants = base_constants,
        .near_identical_constants = near_constants.Get(),
        .shader = map_shader.Get(),
        .raw_depth = raw_srv.Get(),
        .state = state_srv.Get(),
        .tensor_exclusion = dummy_exclusion_srv.Get(),
        .minmax_ema = dummy_minmax_srv.Get(),
        .current_model_input = dummy_current_model_input_srv.Get(),
        .current_appearance_ordinal = dummy_current_appearance_srv.Get(),
        .cut_bridge_state = cut_state_srv.Get(),
        .current_depth = dummy_current_depth_srv.Get(),
        .candidate_output = candidate_uav.Get(),
        .previous_model_input_output = dummy_previous_model_input_uav.Get(),
        .previous_appearance_ordinal_output = dummy_previous_appearance_uav.Get(),
        .previous_reliable_depth_output = dummy_previous_depth_uav.Get(),
        .previous_tensor_exclusion_output = dummy_previous_exclusion_uav.Get(),
        .history_owner_output = dummy_history_owner_uav.Get(),
        .dispatch = v2_gpu::dispatch_command_t::direct(
          (width + 15u) / 16u,
          (height + 15u) / 16u,
          1u
        ),
      };
      if (!v2_gpu::record_map_history(context.Get(), map_history_command)) {
        error = "shared V2 GPU executor rejected replay map/history operands";
        return false;
      }

      const v2_gpu::vertical_command_t vertical_command {
        .constants = base_constants,
        .shader = vertical_limit_shader.Get(),
        .candidate = candidate_srv.Get(),
        .vertical_majorant_output = vertical_majorant_uav.Get(),
        .vertical_conditioned_output = vertical_conditioned_uav.Get(),
        .dispatch = v2_gpu::dispatch_command_t::direct(width, 1u, 1u),
      };
      if (!v2_gpu::record_vertical(context.Get(), vertical_command)) {
        error = "shared V2 GPU executor rejected replay vertical operands";
        return false;
      }

      const v2_gpu::horizontal_command_t horizontal_command {
        .constants = base_constants,
        .shader = limit_shader.Get(),
        .vertical_conditioned = vertical_conditioned_srv.Get(),
        .final_output = final_uav.Get(),
        .dispatch = v2_gpu::dispatch_command_t::direct(height, 1u, 1u),
      };
      if (!v2_gpu::record_horizontal(context.Get(), horizontal_command)) {
        error = "shared V2 GPU executor rejected replay horizontal operands";
        return false;
      }

      context->CSSetShader(encode_shader.Get(), nullptr, 0);
      ID3D11Buffer *encode_constant = encode_constants.Get();
      context->CSSetConstantBuffers(2, 1, &encode_constant);
      ID3D11ShaderResourceView *encode_srvs[] = {final_srv.Get()};
      ID3D11UnorderedAccessView *encode_uavs[] = {encoded_uav.Get()};
      context->CSSetShaderResources(0, 1, encode_srvs);
      context->CSSetUnorderedAccessViews(0, 1, encode_uavs, nullptr);
      context->Dispatch((width + 15u) / 16u, (height + 15u) / 16u, 1u);
      unbind(1u, 1u);
      ID3D11Buffer *null_constants[3] = {nullptr, nullptr, nullptr};
      context->CSSetConstantBuffers(0, 3, null_constants);

      const auto frame_words = read_buffer_words(
        device.Get(), context.Get(), frame_buffer.Get(), v2::frame_stats_float_count);
      const auto state_words = read_buffer_words(
        device.Get(), context.Get(), state_buffer.Get(), v2::state_float_count);
      const auto canonical = read_float_texture(
        device.Get(), context.Get(), coordinate_texture.Get(), width, height);
      const auto candidate = read_float_texture(
        device.Get(), context.Get(), candidate_texture.Get(), width, height);
      const auto vertical_majorant = read_float_texture(
        device.Get(), context.Get(), vertical_majorant_texture.Get(), width, height);
      const auto vertical_conditioned = read_float_texture(
        device.Get(), context.Get(), vertical_conditioned_texture.Get(), width, height);
      const auto final_values = read_float_texture(
        device.Get(), context.Get(), final_texture.Get(), width, height);
      const auto encoded = read_float_texture(
        device.Get(), context.Get(), encoded_texture.Get(), width, height);
      const std::size_t element_count = static_cast<std::size_t>(width) * height;
      if (frame_words.size() != v2::frame_stats_float_count ||
          state_words.size() != v2::state_float_count ||
          canonical.size() != element_count || candidate.size() != element_count ||
          vertical_majorant.size() != element_count ||
          vertical_conditioned.size() != element_count ||
          final_values.size() != element_count || encoded.size() != element_count ||
          state_words[v2::contract_tag_bits] != v2::contract_tag) {
        error = "v2 GPU replay readback or contract tag is incomplete";
        return false;
      }
      const bool all_outputs_finite =
        std::all_of(canonical.begin(), canonical.end(), [](const float value) {
          return std::isfinite(value);
        }) &&
        std::all_of(candidate.begin(), candidate.end(), [](const float value) {
          return std::isfinite(value);
        }) &&
        std::all_of(vertical_majorant.begin(), vertical_majorant.end(), [](const float value) {
          return std::isfinite(value);
        }) &&
        std::all_of(
          vertical_conditioned.begin(),
          vertical_conditioned.end(),
          [](const float value) { return std::isfinite(value); }
        ) &&
        std::all_of(final_values.begin(), final_values.end(), [](const float value) {
          return std::isfinite(value);
        }) &&
        std::all_of(encoded.begin(), encoded.end(), [](const float value) {
          return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
        });
      if (!all_outputs_finite) {
        error = "v2 GPU replay produced non-finite or out-of-container geometry";
        return false;
      }

      const auto float_state = [&](const std::size_t index) {
        return std::bit_cast<float>(state_words[index]);
      };
      const auto float_stat = [&](const std::size_t index) {
        return std::bit_cast<float>(frame_words[index]);
      };
      const bool input_valid = float_stat(v2::frame_stat_valid) > 0.5f;
      const bool frame_valid = float_state(v2::frame_valid) > 0.5f;
      const bool collapsed = input_valid &&
        float_stat(v2::frame_stat_population_std) <= constants.collapse_abs_epsilon;
      const float center = float_state(v2::center);
      const float inverse_scale = float_state(v2::inverse_scale);
      const float latched_scale = inverse_scale > 0.0f ? 1.0f / inverse_scale : 0.0f;
      const float convergence_curve = float_state(v2::convergence_curve);
      const std::uint32_t camera_center_integrity =
        state_words[v2::camera_center_integrity_bits];
      const float container_scale = float_state(v2::container_scale);
      const bool renderer_authorization_valid =
        state_words[v2::renderer_authorization_bits] ==
          (frame_valid ? v2::contract_tag : 0u);
      const bool mapping_state_reserved_valid =
        state_words[v2::mapping_state_reserved_1] == 0u &&
        state_words[v2::mapping_state_reserved_2] == 0u;
      const std::uint32_t calibration_revision =
        state_words[v2::calibration_revision];
      const bool camera_integrity_valid = v2::camera_center_integrity_is_valid(
        state_words[v2::center],
        state_words[v2::inverse_scale],
        state_words[v2::convergence_curve],
        calibration_revision,
        camera_center_integrity
      );
      const bool camera_valid = camera_integrity_valid && renderer_authorization_valid &&
        mapping_state_reserved_valid &&
        v2::convergence_curve_is_valid(convergence_curve) &&
        inverse_scale > 0.0f &&
        v2::acquired_calibration_revision_is_valid(calibration_revision);
      const float effective_gain = frame_valid ? constants.requested_gain : 0.0f;
      const float observed_mean = input_valid ? float_stat(v2::frame_stat_mean) : 0.0f;
      const float observed_std = input_valid ?
                                   float_stat(v2::frame_stat_population_std) : 0.0f;
      const float observed_raw_minimum = input_valid ?
                                         float_stat(v2::frame_stat_minimum) : 0.0f;
      const float observed_raw_maximum = input_valid ?
                                         float_stat(v2::frame_stat_maximum) : 0.0f;
      const bool confirmed_cut = confirmed_cut_input;
      const bool state_values_finite =
        std::isfinite(center) && std::isfinite(inverse_scale) &&
        std::isfinite(latched_scale) && std::isfinite(convergence_curve) &&
        std::isfinite(container_scale);
      v2::state_words_t authenticated_state_words {};
      std::copy(state_words.begin(), state_words.end(), authenticated_state_words.begin());
      const bool state_authenticated = v2::parallax_state_words_are_authenticated(
        authenticated_state_words,
        constants.raw_coordinate_scale
      );
      if (!state_values_finite || !state_authenticated ||
          frame_valid != (input_valid && !collapsed) ||
          (frame_valid && !camera_valid) ||
          container_scale != 1.0f ||
          (confirmed_cut && !frame_valid && camera_valid)) {
        error = "v2 GPU replay state violates frame/camera validity semantics";
        return false;
      }
      if (!renderer_authorization_valid || !mapping_state_reserved_valid) {
        error = "v2 GPU replay produced invalid persistent mapping state";
        return false;
      }
      const bool prior_camera_valid = !trace_rows.empty() &&
        trace_rows.back().at("camera_valid").get<bool>();
      if (!trace_rows.empty() && !confirmed_cut && prior_camera_valid) {
        const auto &previous = trace_rows.back();
        if (!camera_valid ||
            std::abs(center - previous.at("center").get<float>()) > 2.0e-6f ||
            std::abs(inverse_scale - previous.at("inverse_scale").get<float>()) > 2.0e-6f ||
            calibration_revision != previous.at("calibration_revision").get<std::uint32_t>()) {
          error = "v2 GPU replay moved the retained camera without a confirmed cut";
          return false;
        }
      }
      float candidate_center_drift = 0.0f;
      float predicted_zero_translation = 0.0f;
      if (frame_valid && camera_valid && observed_std > 0.0f) {
        candidate_center_drift = (observed_mean - center) * inverse_scale;
        predicted_zero_translation = v2::pointwise_container(
          effective_gain * (
            v2_curve(candidate_center_drift, constants.far_tau, constants.near_log_tau) -
            convergence_curve
          ),
          constants.direct_container_limit
        );
      }

      float pre_limiter_max = 0.0f;
      float final_max = 0.0f;
      float vertical_majorant_max_raise = 0.0f;
      std::size_t vertical_majorant_raised = 0u;
      bool vertical_majorant_illegally_lowered = false;
      float vertical_majorant_shear_max = 0.0f;
      float vertical_conditioned_shear_max = 0.0f;
      bool vertical_conditioned_above_majorant = false;
      float conditioner_max_raise = 0.0f;
      float conditioner_max_lower = 0.0f;
      std::size_t conditioner_raised = 0u;
      std::size_t conditioner_lowered = 0u;
      bool row_majorant_illegally_lowered = false;
      float horizontal_slope_max = 0.0f;
      float final_vertical_shear_max = 0.0f;
      bool pointwise_container_mismatch = false;
      const float limiter_tolerance = std::max(
        1.0e-12f,
        constants.max_horizontal_slope / static_cast<float>(width) * 1.0e-9f
      );
      const float vertical_tolerance = std::max(
        1.0e-12f,
        v2::max_vertical_shear / static_cast<float>(width) * 1.0e-9f
      );
      for (std::size_t index = 0; index < element_count; ++index) {
        const float expected_candidate = frame_valid ? v2::pointwise_container(
          constants.requested_gain * (
            v2_curve(canonical[index], constants.far_tau, constants.near_log_tau) -
            convergence_curve
          ),
          constants.direct_container_limit
        ) : 0.0f;
        pointwise_container_mismatch |=
          std::abs(candidate[index] - expected_candidate) > 2.0e-6f;
        pre_limiter_max = std::max(pre_limiter_max, std::abs(candidate[index]));
        final_max = std::max(final_max, std::abs(final_values[index]));
        const float vertical_raise = vertical_majorant[index] - candidate[index];
        vertical_majorant_max_raise = std::max(
          vertical_majorant_max_raise,
          vertical_raise
        );
        vertical_majorant_raised += vertical_raise > vertical_tolerance ? 1u : 0u;
        vertical_majorant_illegally_lowered |= vertical_raise < -vertical_tolerance;
        const float correction = final_values[index] - candidate[index];
        conditioner_max_raise = std::max(conditioner_max_raise, correction);
        conditioner_max_lower = std::max(conditioner_max_lower, -correction);
        conditioner_raised += correction > limiter_tolerance ? 1u : 0u;
        conditioner_lowered += correction < -limiter_tolerance ? 1u : 0u;
        vertical_conditioned_above_majorant |=
          vertical_conditioned[index] - vertical_majorant[index] > vertical_tolerance;
        row_majorant_illegally_lowered |=
          final_values[index] - vertical_conditioned[index] < -limiter_tolerance;
        if (index % width != 0u) {
          horizontal_slope_max = std::max(
            horizontal_slope_max,
            std::abs(final_values[index] - final_values[index - 1u]) * width
          );
        }
        if (index >= width) {
          vertical_majorant_shear_max = std::max(
            vertical_majorant_shear_max,
            std::abs(vertical_majorant[index] - vertical_majorant[index - width]) * width
          );
          vertical_conditioned_shear_max = std::max(
            vertical_conditioned_shear_max,
            std::abs(
              vertical_conditioned[index] - vertical_conditioned[index - width]
            ) * width
          );
          final_vertical_shear_max = std::max(
            final_vertical_shear_max,
            std::abs(final_values[index] - final_values[index - width]) * width
          );
        }
        const float expected_encoded = 0.5f + final_values[index] /
          (2.0f * v2::direct_container_limit);
        if (std::abs(encoded[index] - expected_encoded) > 2.0e-7f) {
          error =
            "v2 GPU replay encoded interchange differs from experimental final parallax";
          return false;
        }
      }
      if (pointwise_container_mismatch || vertical_majorant_illegally_lowered ||
          vertical_conditioned_above_majorant ||
          row_majorant_illegally_lowered ||
          pre_limiter_max > direct_container_limit + 2.0e-7f ||
          final_max > direct_container_limit + 2.0e-7f ||
          horizontal_slope_max > constants.max_horizontal_slope + 2.0e-5f ||
          vertical_majorant_shear_max > v2::max_vertical_shear + 2.0e-5f ||
          vertical_conditioned_shear_max > v2::max_vertical_shear + 2.0e-5f ||
          final_vertical_shear_max > v2::max_vertical_shear + 2.0e-5f ||
          (!frame_valid && (pre_limiter_max > 2.0e-7f || final_max > 2.0e-7f ||
                       std::any_of(
                         vertical_majorant.begin(),
                         vertical_majorant.end(),
                         [](const float value) { return value != 0.0f; }
                       ) ||
                       std::any_of(
                         vertical_conditioned.begin(),
                         vertical_conditioned.end(),
                         [](const float value) { return value != 0.0f; }
                       ) ||
                       std::any_of(canonical.begin(), canonical.end(), [](const float value) {
                         return value != 0.0f;
                       })))) {
        error = "v2 GPU replay violated pointwise-container, vertical-share, "
                "row-majorant, flat, or spatial-bound contract";
        return false;
      }

      const std::string order_hash = sha256_hex(std::string_view {
        reinterpret_cast<const char *>(canonical.data()),
        canonical.size() * sizeof(float)
      });
      const std::string candidate_hash = sha256_hex(std::string_view {
        reinterpret_cast<const char *>(candidate.data()),
        candidate.size() * sizeof(float)
      });
      const std::string vertical_majorant_hash = sha256_hex(std::string_view {
        reinterpret_cast<const char *>(vertical_majorant.data()),
        vertical_majorant.size() * sizeof(float)
      });
      const std::string vertical_conditioned_hash = sha256_hex(std::string_view {
        reinterpret_cast<const char *>(vertical_conditioned.data()),
        vertical_conditioned.size() * sizeof(float)
      });
      const std::string parallax_hash = sha256_hex(std::string_view {
        reinterpret_cast<const char *>(encoded.data()),
        encoded.size() * sizeof(float)
      });
      const auto [encoded_min, encoded_max] = std::minmax_element(
        encoded.begin(), encoded.end());
      const auto [order_min, order_max] = std::minmax_element(
        canonical.begin(), canonical.end());

      nlohmann::ordered_json row = {
        {"frame_id", frame.frame_id},
        {"input_source_frame_id", frame.frame_id},
        {"rendered_source_frame_id", frame.frame_id},
        {"calibration_revision", state_words[v2::calibration_revision]},
        {"confirmed_cut_count", state_words[v2::confirmed_cut_count]},
        {"confirmed_cut", confirmed_cut},
        {"cut_attribution", sequence_index == 0u ?
          "initialization" : (confirmed_cut ? cut_source : "none")},
        {"input_valid", input_valid},
        {"frame_valid", frame_valid},
        {"camera_valid", camera_valid},
        {"collapsed", collapsed},
        {"center", center},
        {"inverse_scale", inverse_scale},
        {"latched_scale", latched_scale},
        {"convergence_curve", convergence_curve},
        {"requested_gain", constants.requested_gain},
        {"container_scale", container_scale},
        {"effective_gain", effective_gain},
        {"observed_mean", observed_mean},
        {"observed_std", observed_std},
        {"observed_raw_minimum", observed_raw_minimum},
        {"observed_raw_maximum", observed_raw_maximum},
        {"candidate_center_drift_u", candidate_center_drift},
        {"predicted_zero_translation_source_u", predicted_zero_translation},
        {"pre_limiter_max_abs_source_u", pre_limiter_max},
        {"vertical_majorant_raised_fraction",
          static_cast<double>(vertical_majorant_raised) /
            static_cast<double>(element_count)},
        {"vertical_majorant_max_raise_source_u", vertical_majorant_max_raise},
        {"final_max_abs_source_u", final_max},
        {"conditioner_raised_fraction",
          static_cast<double>(conditioner_raised) / static_cast<double>(element_count)},
        {"conditioner_max_raise_source_u", conditioner_max_raise},
        {"conditioner_lowered_fraction",
          static_cast<double>(conditioner_lowered) / static_cast<double>(element_count)},
        {"conditioner_max_lower_source_u", conditioner_max_lower},
        {"final_horizontal_slope_max", horizontal_slope_max},
        {"final_vertical_shear_max", final_vertical_shear_max},
        {"order_sha256", order_hash},
        {"vertical_majorant_sha256", vertical_majorant_hash},
        {"vertical_conditioned_sha256", vertical_conditioned_hash},
        {"parallax_sha256", parallax_hash},
      };
      trace_rows.push_back(std::move(row));

      result = {};
      result.frame_id = frame.frame_id;
      result.raw_depth = raw_srv;
      result.canonical_order = coordinate_srv;
      result.candidate_parallax = candidate_srv;
      result.vertical_majorant = vertical_majorant_srv;
      result.vertical_conditioned = vertical_conditioned_srv;
      result.encoded_final_parallax = encoded_srv;
      result.cut_state = cut_state_srv;
      result.raw_values = std::move(raw_values);
      result.canonical_values = canonical;
      result.candidate_parallax_values = candidate;
      result.vertical_majorant_values = vertical_majorant;
      result.vertical_conditioned_values = vertical_conditioned;
      result.encoded_parallax_values = encoded;
      result.encoded_minimum = *encoded_min;
      result.encoded_maximum = *encoded_max;
      result.maximum_absolute_source_u = final_max;
      result.order_minimum = *order_min;
      result.order_maximum = *order_max;
      result.raw_sha256 = frame.raw_sha256;
      result.order_sha256 = order_hash;
      result.candidate_sha256 = candidate_hash;
      result.vertical_majorant_sha256 = vertical_majorant_hash;
      result.vertical_conditioned_sha256 = vertical_conditioned_hash;
      result.parallax_sha256 = parallax_hash;
      return true;
    }
  };

  depth_coordinate_v2_gpu_replay::depth_coordinate_v2_gpu_replay() = default;
  depth_coordinate_v2_gpu_replay::~depth_coordinate_v2_gpu_replay() = default;

  std::unique_ptr<depth_coordinate_v2_gpu_replay>
  depth_coordinate_v2_gpu_replay::create(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    const fs::path &manifest_path,
    std::string &error
  ) {
    error.clear();
    if (!device || !context || manifest_path.empty()) {
      error = "v2 GPU replay requires a device, context, and manifest";
      return {};
    }
    auto replay = std::unique_ptr<depth_coordinate_v2_gpu_replay>(
      new depth_coordinate_v2_gpu_replay());
    replay->impl_ = std::make_unique<impl>();
    replay->impl_->device = device;
    replay->impl_->context = context;
    replay->impl_->manifest_path = fs::absolute(manifest_path).lexically_normal();
    if (!replay->impl_->load_manifest(error) || !replay->impl_->initialize_gpu(error)) {
      return {};
    }
    return replay;
  }

  std::size_t depth_coordinate_v2_gpu_replay::frame_count() const {
    return impl_->frames.size();
  }

  unsigned depth_coordinate_v2_gpu_replay::width() const {
    return impl_->width;
  }

  unsigned depth_coordinate_v2_gpu_replay::height() const {
    return impl_->height;
  }

  const std::string &depth_coordinate_v2_gpu_replay::model_name() const {
    return impl_->model_name;
  }

  const std::string &depth_coordinate_v2_gpu_replay::manifest_sha256() const {
    return impl_->manifest_sha256;
  }

  bool depth_coordinate_v2_gpu_replay::dispatch(
    const std::size_t sequence_index,
    const std::string_view expected_frame_id,
    const std::string_view expected_source_sha256,
    depth_coordinate_v2_gpu_frame &result,
    std::string &error
  ) {
    return impl_->dispatch_frame(
      sequence_index,
      expected_frame_id,
      expected_source_sha256,
      result,
      error
    );
  }

  bool depth_coordinate_v2_gpu_replay::write_state_trace(
    const fs::path &path,
    std::string &error
  ) const {
    error.clear();
    if (impl_->trace_rows.size() != impl_->frames.size()) {
      error = "v2 GPU replay state trace is incomplete";
      return false;
    }
    nlohmann::ordered_json fields = nlohmann::ordered_json::array();
    for (const char *name : trace_field_names) {
      fields.push_back(name);
    }
    nlohmann::ordered_json document = {
      {"schema", depth_coordinate_v2_state_trace_schema},
      {"policy", std::string(depth_coordinate_v2_state_trace_policy)},
      {"calibration_contract", impl_->calibration_contract},
      {"cut_source", impl_->cut_source},
      {"diagnostic_role", std::string(depth_coordinate_v2_diagnostic_role)},
      {"diagnostic_method", "gpu-frame-moments-and-rendered-fields-v5"},
      {"mapping_config", impl_->mapping_config},
      {"frame_fields", std::move(fields)},
      {"frames", impl_->trace_rows},
      {"producer", {
        {"authority", std::string(depth_coordinate_v2_gpu_authority)},
        {"manifest_sha256", impl_->manifest_sha256},
        {"contract_canonical_sha256", v2::contract_canonical_sha256},
        {"tensor_shape", {
          {"width", impl_->width},
          {"height", impl_->height},
        }},
        // Production state/geometry passes. The replay's coordinate_main dispatch is
        // diagnostic report materialization, exactly like live Dump 3D, and is intentionally
        // excluded from the authenticated producer source closure.
        {"shader_sequence", {
          "depth_coordinate_v2_moments_cs.hlsl",
          "depth_coordinate_v2_frame_resolve_cs.hlsl",
          "depth_coordinate_v2_state_resolve_cs.hlsl",
          "depth_coordinate_v2_map_cs.hlsl",
          "depth_coordinate_v2_vertical_limit_cs.hlsl",
          "depth_coordinate_v2_limit_cs.hlsl",
        }},
        {"state_persistence", "single-buffer-whole-sequence"},
        {"numpy_role", "comparison-only-not-render-authority"},
      }},
    };
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
      error = "cannot create v2 GPU state trace";
      return false;
    }
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<float>::max_digits10)
           << document.dump(2) << '\n';
    stream.flush();
    if (!stream.good()) {
      error = "cannot finish v2 GPU state trace";
      return false;
    }
    return true;
  }

#ifdef SUNSHINE_TESTS
  bool depth_coordinate_v2_gpu_replay::overwrite_state_for_testing(
    const std::vector<std::uint32_t> &words,
    std::string &error
  ) {
    if (!impl_ || !impl_->context || !impl_->state_buffer ||
        words.size() != v2::state_float_count) {
      error = "test state injection does not match the exact V2 state contract";
      return false;
    }
    impl_->context->UpdateSubresource(
      impl_->state_buffer.Get(), 0, nullptr, words.data(), 0, 0);
    return true;
  }
#endif

}  // namespace sbs_bench

#endif  // _WIN32
