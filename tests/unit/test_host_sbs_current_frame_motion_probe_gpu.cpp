/**
 * @file tests/unit/test_host_sbs_current_frame_motion_probe_gpu.cpp
 * @brief Executes the shadow current-frame motion probe on a D3D11 WARP device.
 */
#include "../tests_common.h"

#ifdef _WIN32

  #include <array>
  #include <bit>
  #include <cmath>
  #include <cstddef>
  #include <cstdint>
  #include <cstring>
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <limits>
  #include <numeric>
  #include <span>
  #include <src/generated/sbs_adaptive_state_contract.h>
  #include <src/video_depth_estimator.h>
  #include <string>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;

  constexpr UINT field_width = 18u;
  constexpr UINT field_height = 18u;
  constexpr UINT field_texels = field_width * field_height;
  constexpr UINT bottom_top = 14u;
  constexpr UINT bottom_bottom = field_height;
  constexpr std::uint64_t baseline_frame_id = 0x12345678abcdef01ull;
  constexpr std::uint64_t current_frame_id = 0x12345679abcdef02ull;

  using probe_words_t =
    std::array<std::uint32_t, models::adaptive_motion_probe_word_count>;

  struct probe_inputs_t {
    std::vector<float> current_model = std::vector<float>(3u * field_texels, 0.0f);
    std::vector<float> previous_model = std::vector<float>(3u * field_texels, 0.0f);
    std::vector<float> current_appearance = std::vector<float>(field_texels, 0.0f);
    std::vector<float> previous_appearance = std::vector<float>(field_texels, 0.0f);
    std::vector<std::uint32_t> current_exclusion =
      std::vector<std::uint32_t>(field_texels, 0u);
    std::vector<std::uint32_t> previous_exclusion =
      std::vector<std::uint32_t>(field_texels, 0u);
  };

  struct probe_execution_t {
    probe_words_t words {};
    models::adaptive_motion_probe_sample sample {};
    bool decoded = false;
  };

  std::array<float, sbs_adaptive_state::word_count> settled_cut_state() {
    auto state = sbs_adaptive_state::initial_values;
    const auto index = sbs_adaptive_state::index;
    state[index(sbs_adaptive_state::word_e::initialized)] = 1.0f;
    state[index(sbs_adaptive_state::word_e::scene_age)] = 12.0f;
    state[index(sbs_adaptive_state::word_e::cut_flags)] = static_cast<float>(
      sbs_adaptive_state::cut_flag_geometry_armed |
      sbs_adaptive_state::cut_flag_appearance_armed
    );
    state[index(sbs_adaptive_state::word_e::model_input_history_state)] = 1.0f;
    state[index(sbs_adaptive_state::word_e::hard_cut_count)] =
      std::bit_cast<float>(7u);
    state[index(sbs_adaptive_state::word_e::range_collapsed)] = 0.0f;
    state[index(sbs_adaptive_state::word_e::depth_ready)] = 1.0f;
    state[index(sbs_adaptive_state::word_e::hard_cut_pulse)] = 0.0f;
    state[index(sbs_adaptive_state::word_e::analysis_flags)] = 0.0f;
    return state;
  }

  class motion_probe_gpu_fixture_t {
  public:
    enum class initialize_result_e {
      ready,
      d3d_unavailable,
      failed,
    };

    initialize_result_e initialize(std::string &error) {
      constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      const auto device_status = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_WARP,
        nullptr,
        0u,
        requested,
        static_cast<UINT>(std::size(requested)),
        D3D11_SDK_VERSION,
        &device_,
        &actual,
        &context_
      );
      if (FAILED(device_status) || actual < D3D_FEATURE_LEVEL_11_0) {
        error = "D3D11 WARP feature level 11 is unavailable";
        return initialize_result_e::d3d_unavailable;
      }

      const std::filesystem::path shader_path =
        SUNSHINE_SHADERS_DIR "/host_sbs_current_frame_motion_probe_cs.hlsl";
      ComPtr<ID3DBlob> shader_blob;
      ComPtr<ID3DBlob> shader_errors;
      const auto compile_status = D3DCompileFromFile(
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
      if (FAILED(compile_status) || !shader_blob) {
        error = shader_errors ?
                  static_cast<const char *>(shader_errors->GetBufferPointer()) :
                  "motion-probe shader compilation failed without diagnostics";
        return initialize_result_e::failed;
      }
      if (FAILED(device_->CreateComputeShader(shader_blob->GetBufferPointer(), shader_blob->GetBufferSize(), nullptr, &shader_))) {
        error = "could not create the motion-probe compute shader";
        return initialize_result_e::failed;
      }

      D3D11_BUFFER_DESC constants_desc {};
      constants_desc.ByteWidth = 16u * sizeof(std::uint32_t);
      constants_desc.Usage = D3D11_USAGE_DEFAULT;
      constants_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      if (FAILED(device_->CreateBuffer(&constants_desc, nullptr, &depth_constants_))) {
        error = "could not create the depth constant buffer";
        return initialize_result_e::failed;
      }
      constants_desc.ByteWidth = 8u * sizeof(std::uint32_t);
      if (FAILED(device_->CreateBuffer(&constants_desc, nullptr, &probe_constants_))) {
        error = "could not create the motion-probe constant buffer";
        return initialize_result_e::failed;
      }

      D3D11_BUFFER_DESC output_desc {};
      output_desc.ByteWidth = static_cast<UINT>(sizeof(probe_words_t));
      output_desc.Usage = D3D11_USAGE_DEFAULT;
      output_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      output_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      output_desc.StructureByteStride = sizeof(std::uint32_t);
      if (FAILED(device_->CreateBuffer(&output_desc, nullptr, &output_buffer_)) || FAILED(device_->CreateUnorderedAccessView(output_buffer_.Get(), nullptr, &output_uav_))) {
        error = "could not create the motion-probe output buffer";
        return initialize_result_e::failed;
      }
      output_desc.Usage = D3D11_USAGE_STAGING;
      output_desc.BindFlags = 0u;
      output_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      output_desc.MiscFlags = 0u;
      output_desc.StructureByteStride = 0u;
      if (FAILED(device_->CreateBuffer(&output_desc, nullptr, &staging_buffer_))) {
        error = "could not create the motion-probe staging buffer";
        return initialize_result_e::failed;
      }
      return initialize_result_e::ready;
    }

    bool run(
      const probe_inputs_t &inputs,
      const std::array<float, sbs_adaptive_state::word_count> &cut_state,
      probe_execution_t &execution,
      std::string &error
    ) {
      if (
        inputs.current_model.size() != 3u * field_texels ||
        inputs.previous_model.size() != 3u * field_texels ||
        inputs.current_appearance.size() != field_texels ||
        inputs.previous_appearance.size() != field_texels ||
        inputs.current_exclusion.size() != field_texels ||
        inputs.previous_exclusion.size() != field_texels
      ) {
        error = "motion-probe input sizes are invalid";
        return false;
      }

      ComPtr<ID3D11Buffer> current_model_buffer;
      ComPtr<ID3D11ShaderResourceView> current_model_srv;
      ComPtr<ID3D11Buffer> previous_model_buffer;
      ComPtr<ID3D11ShaderResourceView> previous_model_srv;
      ComPtr<ID3D11Buffer> current_appearance_buffer;
      ComPtr<ID3D11ShaderResourceView> current_appearance_srv;
      ComPtr<ID3D11Buffer> previous_appearance_buffer;
      ComPtr<ID3D11ShaderResourceView> previous_appearance_srv;
      ComPtr<ID3D11Buffer> cut_state_buffer;
      ComPtr<ID3D11ShaderResourceView> cut_state_srv;
      if (
        !create_structured_srv(
          inputs.current_model,
          sizeof(float),
          current_model_buffer,
          current_model_srv
        ) ||
        !create_structured_srv(
          inputs.previous_model,
          sizeof(float),
          previous_model_buffer,
          previous_model_srv
        ) ||
        !create_structured_srv(
          inputs.current_appearance,
          sizeof(float),
          current_appearance_buffer,
          current_appearance_srv
        ) ||
        !create_structured_srv(
          inputs.previous_appearance,
          sizeof(float),
          previous_appearance_buffer,
          previous_appearance_srv
        ) ||
        !create_structured_srv(
          std::span<const float> {cut_state},
          4u * sizeof(float),
          cut_state_buffer,
          cut_state_srv
        )
      ) {
        error = "could not create a structured motion-probe input";
        return false;
      }

      ComPtr<ID3D11Texture2D> current_exclusion_texture;
      ComPtr<ID3D11ShaderResourceView> current_exclusion_srv;
      ComPtr<ID3D11Texture2D> previous_exclusion_texture;
      ComPtr<ID3D11ShaderResourceView> previous_exclusion_srv;
      if (
        !create_exclusion_srv(
          inputs.current_exclusion,
          current_exclusion_texture,
          current_exclusion_srv
        ) ||
        !create_exclusion_srv(
          inputs.previous_exclusion,
          previous_exclusion_texture,
          previous_exclusion_srv
        )
      ) {
        error = "could not create an exclusion texture";
        return false;
      }

      std::array<std::uint32_t, 16> depth_constants {};
      depth_constants[0] = field_width;
      depth_constants[1] = field_height;
      depth_constants[9] = 0u;
      depth_constants[10] = 0u;
      depth_constants[11] = field_width;
      depth_constants[12] = field_height;
      const std::array<std::uint32_t, 8> probe_constants {
        static_cast<std::uint32_t>(current_frame_id),
        static_cast<std::uint32_t>(current_frame_id >> 32u),
        static_cast<std::uint32_t>(baseline_frame_id),
        static_cast<std::uint32_t>(baseline_frame_id >> 32u),
        bottom_top,
        bottom_bottom,
        0u,
        0u,
      };
      context_->UpdateSubresource(
        depth_constants_.Get(),
        0u,
        nullptr,
        depth_constants.data(),
        0u,
        0u
      );
      context_->UpdateSubresource(
        probe_constants_.Get(),
        0u,
        nullptr,
        probe_constants.data(),
        0u,
        0u
      );
      const UINT zero[4] = {};
      context_->ClearUnorderedAccessViewUint(output_uav_.Get(), zero);

      ID3D11ShaderResourceView *srvs[] = {
        current_model_srv.Get(),
        previous_model_srv.Get(),
        current_appearance_srv.Get(),
        previous_appearance_srv.Get(),
        current_exclusion_srv.Get(),
        previous_exclusion_srv.Get(),
        cut_state_srv.Get(),
      };
      ID3D11Buffer *constant_buffers[] = {
        depth_constants_.Get(),
        probe_constants_.Get(),
      };
      ID3D11UnorderedAccessView *uavs[] = {output_uav_.Get()};
      context_->CSSetShader(shader_.Get(), nullptr, 0u);
      context_->CSSetShaderResources(0u, static_cast<UINT>(std::size(srvs)), srvs);
      context_->CSSetConstantBuffers(
        0u,
        static_cast<UINT>(std::size(constant_buffers)),
        constant_buffers
      );
      context_->CSSetUnorderedAccessViews(
        0u,
        static_cast<UINT>(std::size(uavs)),
        uavs,
        nullptr
      );
      context_->Dispatch(
        (field_width + 15u) / 16u,
        (field_height + 15u) / 16u,
        1u
      );

      ID3D11ShaderResourceView *null_srvs[std::size(srvs)] {};
      ID3D11Buffer *null_constant_buffers[std::size(constant_buffers)] {};
      ID3D11UnorderedAccessView *null_uavs[std::size(uavs)] {};
      context_->CSSetUnorderedAccessViews(
        0u,
        static_cast<UINT>(std::size(null_uavs)),
        null_uavs,
        nullptr
      );
      context_->CSSetShaderResources(
        0u,
        static_cast<UINT>(std::size(null_srvs)),
        null_srvs
      );
      context_->CSSetConstantBuffers(
        0u,
        static_cast<UINT>(std::size(null_constant_buffers)),
        null_constant_buffers
      );
      context_->CSSetShader(nullptr, nullptr, 0u);

      context_->CopyResource(staging_buffer_.Get(), output_buffer_.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context_->Map(staging_buffer_.Get(), 0u, D3D11_MAP_READ, 0u, &mapped)) || !mapped.pData) {
        error = "could not read the motion-probe output";
        return false;
      }
      std::memcpy(execution.words.data(), mapped.pData, sizeof(execution.words));
      context_->Unmap(staging_buffer_.Get(), 0u);
      execution.decoded = models::decode_adaptive_motion_probe_words(
        execution.words,
        current_frame_id,
        baseline_frame_id,
        static_cast<int>(field_width),
        static_cast<int>(field_height),
        execution.sample
      );
      return true;
    }

  private:
    template<typename Container>
    bool create_structured_srv(
      const Container &values,
      const UINT stride,
      ComPtr<ID3D11Buffer> &buffer,
      ComPtr<ID3D11ShaderResourceView> &srv
    ) {
      const std::span data {values};
      if (data.empty() || stride == 0u || data.size_bytes() % stride != 0u) {
        return false;
      }
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = static_cast<UINT>(data.size_bytes());
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = stride;
      D3D11_SUBRESOURCE_DATA initial {data.data(), 0u, 0u};
      if (FAILED(device_->CreateBuffer(&desc, &initial, &buffer))) {
        return false;
      }
      D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc {};
      srv_desc.Format = DXGI_FORMAT_UNKNOWN;
      srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
      srv_desc.Buffer.FirstElement = 0u;
      srv_desc.Buffer.NumElements = desc.ByteWidth / stride;
      return SUCCEEDED(device_->CreateShaderResourceView(
        buffer.Get(),
        &srv_desc,
        &srv
      ));
    }

    bool create_exclusion_srv(
      const std::vector<std::uint32_t> &values,
      ComPtr<ID3D11Texture2D> &texture,
      ComPtr<ID3D11ShaderResourceView> &srv
    ) {
      if (values.size() != field_texels) {
        return false;
      }
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = field_width;
      desc.Height = field_height;
      desc.MipLevels = 1u;
      desc.ArraySize = 1u;
      desc.Format = DXGI_FORMAT_R32_UINT;
      desc.SampleDesc.Count = 1u;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA initial {
        values.data(),
        field_width * sizeof(std::uint32_t),
        0u
      };
      return SUCCEEDED(device_->CreateTexture2D(&desc, &initial, &texture)) &&
             SUCCEEDED(device_->CreateShaderResourceView(
               texture.Get(),
               nullptr,
               &srv
             ));
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11ComputeShader> shader_;
    ComPtr<ID3D11Buffer> depth_constants_;
    ComPtr<ID3D11Buffer> probe_constants_;
    ComPtr<ID3D11Buffer> output_buffer_;
    ComPtr<ID3D11UnorderedAccessView> output_uav_;
    ComPtr<ID3D11Buffer> staging_buffer_;
  };
}  // namespace

TEST(HostSbsCurrentFrameMotionProbeGpuTest, PreservesExactEvidenceAndStateIdentity) {
  motion_probe_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = fixture.initialize(error);
  if (initialized == motion_probe_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(
    initialized,
    motion_probe_gpu_fixture_t::initialize_result_e::ready
  ) << error;

  const auto state = settled_cut_state();
  probe_execution_t quiet;
  ASSERT_TRUE(fixture.run(probe_inputs_t {}, state, quiet, error)) << error;
  ASSERT_TRUE(quiet.decoded);
  EXPECT_EQ(quiet.sample.current_frame_id, current_frame_id);
  EXPECT_EQ(quiet.sample.baseline_frame_id, baseline_frame_id);
  EXPECT_EQ(
    quiet.sample.prior_state_flags,
    models::adaptive_motion_probe_settled_flags
  );
  EXPECT_EQ(quiet.sample.hard_cut_count, 7u);
  EXPECT_EQ(quiet.sample.scene_age, 12u);
  EXPECT_EQ(quiet.sample.cut_flags, 3u);
  EXPECT_EQ(quiet.sample.model_input_history_state, 1u);
  EXPECT_EQ(quiet.sample.admitted_texels, field_texels);
  EXPECT_EQ(quiet.sample.exclusion_mismatch_texels, 0u);
  EXPECT_EQ(quiet.sample.exact_changed_texels, 0u);
  EXPECT_EQ(quiet.sample.maximum_exact_changed_in_16x16_tile, 0u);
  EXPECT_EQ(
    quiet.sample.bottom_band_admitted_texels,
    field_width * (bottom_bottom - bottom_top)
  );
  EXPECT_EQ(quiet.sample.bottom_band_exact_changed_texels, 0u);
  EXPECT_EQ(
    models::adaptive_motion_probe_exact_verdict(quiet.sample),
    models::adaptive_motion_probe_exact_verdict_e::quiet_evidence
  );
  models::adaptive_motion_probe_sample wrong_identity_sample;
  EXPECT_FALSE(models::decode_adaptive_motion_probe_words(quiet.words, current_frame_id + 1u, baseline_frame_id, static_cast<int>(field_width), static_cast<int>(field_height), wrong_identity_sample));
  EXPECT_FALSE(models::decode_adaptive_motion_probe_words(quiet.words, current_frame_id, baseline_frame_id - 1u, static_cast<int>(field_width), static_cast<int>(field_height), wrong_identity_sample));

  probe_inputs_t one_bit_inputs;
  constexpr std::size_t one_bit_index = 2u * field_width + 3u;
  one_bit_inputs.previous_model[one_bit_index] = 1.0f;
  one_bit_inputs.current_model[one_bit_index] =
    std::bit_cast<float>(std::bit_cast<std::uint32_t>(1.0f) + 1u);
  probe_execution_t one_bit;
  ASSERT_TRUE(fixture.run(one_bit_inputs, state, one_bit, error)) << error;
  ASSERT_TRUE(one_bit.decoded);
  EXPECT_EQ(one_bit.sample.exact_changed_texels, 1u);
  EXPECT_EQ(one_bit.sample.maximum_exact_changed_in_16x16_tile, 1u);
  EXPECT_EQ(
    models::adaptive_motion_probe_exact_verdict(one_bit.sample),
    models::adaptive_motion_probe_exact_verdict_e::motion_veto
  );

  probe_inputs_t swap_inputs;
  constexpr std::size_t swap_a = 4u * field_width + 4u;
  constexpr std::size_t swap_b = 4u * field_width + 7u;
  swap_inputs.previous_model[swap_a] = 1.0f;
  swap_inputs.previous_model[swap_b] = -1.0f;
  swap_inputs.current_model[swap_a] = -1.0f;
  swap_inputs.current_model[swap_b] = 1.0f;
  ASSERT_FLOAT_EQ(
    std::accumulate(
      swap_inputs.previous_model.begin(),
      swap_inputs.previous_model.end(),
      0.0f
    ),
    std::accumulate(
      swap_inputs.current_model.begin(),
      swap_inputs.current_model.end(),
      0.0f
    )
  );
  probe_execution_t swap;
  ASSERT_TRUE(fixture.run(swap_inputs, state, swap, error)) << error;
  ASSERT_TRUE(swap.decoded);
  EXPECT_EQ(swap.sample.exact_changed_texels, 2u);
  EXPECT_EQ(swap.sample.maximum_exact_changed_in_16x16_tile, 2u);
  EXPECT_EQ(
    models::adaptive_motion_probe_exact_verdict(swap.sample),
    models::adaptive_motion_probe_exact_verdict_e::motion_veto
  );

  probe_inputs_t bottom_inputs;
  constexpr std::size_t bottom_index = 17u * field_width + 17u;
  bottom_inputs.current_model[bottom_index] = 1.0f;
  probe_execution_t bottom;
  ASSERT_TRUE(fixture.run(bottom_inputs, state, bottom, error)) << error;
  ASSERT_TRUE(bottom.decoded);
  EXPECT_EQ(bottom.sample.exact_changed_texels, 1u);
  EXPECT_EQ(bottom.sample.rgb_delta_1_over_1024_texels, 1u);
  EXPECT_EQ(bottom.sample.rgb_delta_1_over_256_texels, 1u);
  EXPECT_EQ(bottom.sample.rgb_delta_1_over_64_texels, 1u);
  EXPECT_EQ(bottom.sample.bottom_band_exact_changed_texels, 1u);
  EXPECT_EQ(bottom.sample.bottom_band_rgb_delta_1_over_1024_texels, 1u);
  EXPECT_NEAR(bottom.sample.maximum_rgb_delta, 0.229f, 1e-6f);
  EXPECT_NEAR(bottom.sample.bottom_band_maximum_rgb_delta, 0.229f, 1e-6f);

  probe_inputs_t exclusion_inputs;
  exclusion_inputs.current_exclusion[16u] = 1u;
  probe_execution_t exclusion;
  ASSERT_TRUE(fixture.run(exclusion_inputs, state, exclusion, error)) << error;
  ASSERT_TRUE(exclusion.decoded);
  EXPECT_EQ(exclusion.sample.admitted_texels, field_texels - 1u);
  EXPECT_EQ(exclusion.sample.exclusion_mismatch_texels, 1u);
  EXPECT_EQ(exclusion.sample.exact_changed_texels, 0u);
  EXPECT_EQ(exclusion.sample.maximum_exact_changed_in_16x16_tile, 0u);
  EXPECT_EQ(
    models::adaptive_motion_probe_exact_verdict(exclusion.sample),
    models::adaptive_motion_probe_exact_verdict_e::motion_veto
  );

  auto malformed_tag_state = state;
  malformed_tag_state[sbs_adaptive_state::index(
    sbs_adaptive_state::word_e::cut_contract_tag_bits
  )] = 0.0f;
  probe_execution_t malformed_tag;
  ASSERT_TRUE(fixture.run(
    probe_inputs_t {},
    malformed_tag_state,
    malformed_tag,
    error
  )) << error;
  ASSERT_TRUE(malformed_tag.decoded);
  EXPECT_EQ(
    malformed_tag.sample.prior_state_flags &
      models::adaptive_motion_probe_flag_cut_contract,
    0u
  );
  EXPECT_EQ(
    models::adaptive_motion_probe_exact_verdict(malformed_tag.sample),
    models::adaptive_motion_probe_exact_verdict_e::invalid
  );

  auto unsettled_state = state;
  unsettled_state[sbs_adaptive_state::index(
    sbs_adaptive_state::word_e::cut_flags
  )] = static_cast<float>(sbs_adaptive_state::cut_flag_geometry_armed | sbs_adaptive_state::cut_flag_appearance_armed | sbs_adaptive_state::cut_flag_geometry_confirmation_pending);
  probe_execution_t unsettled;
  ASSERT_TRUE(fixture.run(
    probe_inputs_t {},
    unsettled_state,
    unsettled,
    error
  )) << error;
  ASSERT_TRUE(unsettled.decoded);
  EXPECT_EQ(
    unsettled.sample.prior_state_flags &
      models::adaptive_motion_probe_flag_cut_flags_settled,
    0u
  );
  EXPECT_EQ(
    models::adaptive_motion_probe_exact_verdict(unsettled.sample),
    models::adaptive_motion_probe_exact_verdict_e::invalid
  );

  auto malformed_numeric_state = state;
  malformed_numeric_state[sbs_adaptive_state::index(
    sbs_adaptive_state::word_e::scene_age
  )] = std::numeric_limits<float>::quiet_NaN();
  probe_execution_t malformed_numeric;
  ASSERT_TRUE(fixture.run(
    probe_inputs_t {},
    malformed_numeric_state,
    malformed_numeric,
    error
  )) << error;
  EXPECT_FALSE(malformed_numeric.decoded);
}

#else

TEST(HostSbsCurrentFrameMotionProbeGpuTest, WindowsOnly) {
  GTEST_SKIP() << "D3D11 WARP is Windows-only";
}

#endif
