/**
 * @file tests/unit/test_sbs_frame_roi_transform_gpu.cpp
 * @brief WARP execution tests for the frame-owned Host SBS ROI transform contract.
 */

#ifdef _WIN32

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <cmath>
  #include <cstdint>
  #include <cstring>
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <gtest/gtest.h>
  #include <src/generated/sbs_adaptive_state_contract.h>
  #include <src/generated/sbs_scene_controller_contract.h>
  #include <src/sbs_frame_roi_transform.h>
  #include <src/sbs_roi_shape_request.h>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;

  using rule_words_t = std::array<
    std::uint32_t,
    sbs_scene_controller::rule_state_word_count>;

  struct alignas(16) frame_roi_transform_t {
    std::array<std::uint32_t, 4> header {};
    std::array<std::uint32_t, 4> identity {};
    std::array<std::uint32_t, 4> model {};
    std::array<std::uint32_t, 4> focus_bits {};
    std::array<std::uint32_t, 4> crop_bits {};
    std::array<std::uint32_t, 4> accepted_bounds {};
    std::array<std::uint32_t, 4> feather_bits {};
    std::array<std::uint32_t, 4> diagnostics {};
  };

  static_assert(
    sizeof(frame_roi_transform_t) ==
    models::frame_roi_transform_vector_count * 4u *
      sizeof(std::uint32_t)
  );
  static_assert(alignof(frame_roi_transform_t) == 16u);

  constexpr std::uint32_t flag_valid = 1u << 0u;
  constexpr std::uint32_t flag_full_frame = 1u << 1u;
  constexpr std::uint32_t flag_active_roi = 1u << 2u;
  constexpr std::uint32_t flag_reset_debt = 1u << 3u;

  constexpr std::uint32_t fallback_none =
    models::frame_roi_fallback_reason_value(
      models::frame_roi_fallback_reason::none
    );
  constexpr std::uint32_t fallback_inactive =
    models::frame_roi_fallback_reason_value(
      models::frame_roi_fallback_reason::inactive
    );
  constexpr std::uint32_t fallback_full_frame_shape_mismatch =
    models::frame_roi_fallback_reason_value(
      models::frame_roi_fallback_reason::full_frame_shape_mismatch
    );

  constexpr std::uint32_t word_index(
    const sbs_scene_controller::rule_state_word_e word
  ) {
    return static_cast<std::uint32_t>(
      sbs_scene_controller::index(word)
    );
  }

  std::uint32_t float_bits(const float value) {
    return std::bit_cast<std::uint32_t>(value);
  }

  float bits_float(const std::uint32_t value) {
    return std::bit_cast<float>(value);
  }

  std::array<std::uint32_t, 4> rect_bits(
    const std::array<float, 4> &rect
  ) {
    return {
      float_bits(rect[0]),
      float_bits(rect[1]),
      float_bits(rect[2]),
      float_bits(rect[3]),
    };
  }

  std::array<float, 4> rect_floats(
    const std::array<std::uint32_t, 4> &bits
  ) {
    return {
      bits_float(bits[0]),
      bits_float(bits[1]),
      bits_float(bits[2]),
      bits_float(bits[3]),
    };
  }

  rule_words_t make_rule_state(
    const std::array<float, 4> &roi,
    const std::uint32_t update_count = 42u,
    const std::uint32_t backend_generation = 7u,
    const std::uint32_t roi_generation = 3u
  ) {
    rule_words_t words {};
    words[word_index(
      sbs_scene_controller::rule_state_word_e::schema_version
    )] = float_bits(
      static_cast<float>(sbs_scene_controller::schema_version)
    );
    words[word_index(
      sbs_scene_controller::rule_state_word_e::output_valid
    )] = float_bits(1.0f);
    words[word_index(
      sbs_scene_controller::rule_state_word_e::backend_generation
    )] = backend_generation;
    words[word_index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )] = roi_generation;
    words[word_index(
      sbs_scene_controller::rule_state_word_e::update_count
    )] = update_count;
    words[word_index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] =
      sbs_scene_controller::state_flags_initialized |
      sbs_scene_controller::state_flags_roi_locked;
    const auto bits = rect_bits(roi);
    words[word_index(
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    )] = bits[0];
    words[word_index(
      sbs_scene_controller::rule_state_word_e::committed_roi_y0
    )] = bits[1];
    words[word_index(
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    )] = bits[2];
    words[word_index(
      sbs_scene_controller::rule_state_word_e::committed_roi_y1
    )] = bits[3];
    return words;
  }

  models::frame_roi_builder_constants make_constants(
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const std::uint32_t model_width,
    const std::uint32_t model_height
  ) {
    models::frame_roi_builder_constants constants;
    constants.source_width = source_width;
    constants.source_height = source_height;
    constants.model_width = model_width;
    constants.model_height = model_height;
    constants.source_frame_id_low = 0x55667788u;
    constants.source_frame_id_high = 0x11223344u;
    constants.expected_backend_generation = 7u;
    constants.transform_version_low = 0xDDEEFF00u;
    constants.transform_version_high = 0x99AABBCCu;
    constants.gpu_bank_identity = 1u;
    constants.lifecycle_reserved = 1u;
    constants.quiet_halo_cells = 2.0f;
    constants.feather_cells = 1.0f;
    constants.min_focus_cells = 2.0f;
    constants.analysis_canvas_size = 128u;
    return constants;
  }

  void configure_active(
    models::frame_roi_builder_constants &constants,
    const std::array<float, 4> &roi,
    const std::uint32_t sampled_update_count = 42u,
    const std::uint32_t backend_generation = 7u,
    const std::uint32_t roi_generation = 3u
  ) {
    constants.active_rules = 1u;
    constants.expected_request_authority =
      (models::sbs_roi_shape_flag_bits(
         models::sbs_roi_shape_request_flag::valid
       ) |
       models::sbs_roi_shape_flag_bits(
         models::sbs_roi_shape_request_flag::active_roi
       ));
    constants.expected_backend_generation = backend_generation;
    constants.expected_roi_generation = roi_generation;
    constants.expected_rule_update_count = sampled_update_count;
    constants.expected_committed_roi_bits = rect_bits(roi);
    constants.shape_request_id =
      models::sbs_roi_shape_request_id_from_fields(
        models::sbs_roi_shape_request_schema_version,
        backend_generation,
        roi_generation,
        sampled_update_count,
        constants.source_width,
        constants.source_height,
        constants.model_width,
        constants.model_height,
        constants.expected_committed_roi_bits
      );
  }

  bool output_valid(const frame_roi_transform_t &value) {
    return (value.header[1] & flag_valid) != 0u;
  }

  bool output_active(const frame_roi_transform_t &value) {
    return output_valid(value) &&
           (value.header[1] & flag_active_roi) != 0u;
  }

  void expect_canonical_geometry(
    const frame_roi_transform_t &value,
    const std::uint32_t model_width,
    const std::uint32_t model_height
  ) {
    const auto canonical =
      rect_bits({0.0f, 0.0f, 1.0f, 1.0f});
    EXPECT_EQ(value.focus_bits, canonical);
    EXPECT_EQ(value.crop_bits, canonical);
    EXPECT_EQ(
      value.accepted_bounds,
      (std::array<std::uint32_t, 4> {
        0u,
        0u,
        model_width,
        model_height,
      })
    );
    EXPECT_EQ(value.model[2], model_width * model_height);
    EXPECT_EQ(
      value.feather_bits,
      (std::array<std::uint32_t, 4> {})
    );
  }

  void expect_active_geometry(
    const frame_roi_transform_t &value,
    const models::frame_roi_builder_constants &constants,
    const std::array<float, 4> &expected_focus,
    const std::array<std::uint32_t, 4> &expected_bounds
  ) {
    ASSERT_TRUE(output_active(value));
    EXPECT_EQ(value.focus_bits, rect_bits(expected_focus));
    EXPECT_EQ(value.accepted_bounds, expected_bounds);
    EXPECT_EQ(
      value.model[2],
      (expected_bounds[2] - expected_bounds[0]) *
        (expected_bounds[3] - expected_bounds[1])
    );
    EXPECT_EQ(value.model[3], constants.shape_request_id);

    const auto crop = rect_floats(value.crop_bits);
    const auto focus = rect_floats(value.focus_bits);
    EXPECT_LE(crop[0], focus[0]);
    EXPECT_LE(crop[1], focus[1]);
    EXPECT_GE(crop[2], focus[2]);
    EXPECT_GE(crop[3], focus[3]);
    const float crop_pixel_width =
      (crop[2] - crop[0]) *
      static_cast<float>(constants.source_width);
    const float crop_pixel_height =
      (crop[3] - crop[1]) *
      static_cast<float>(constants.source_height);
    const float lhs =
      crop_pixel_width * static_cast<float>(constants.model_height);
    const float rhs =
      crop_pixel_height * static_cast<float>(constants.model_width);
    EXPECT_NEAR(lhs, rhs, std::max(std::abs(lhs), std::abs(rhs)) * 1e-5f);

    const float expected_feather = 1.0f / 128.0f;
    for (const auto bits : value.feather_bits) {
      EXPECT_FLOAT_EQ(bits_float(bits), expected_feather);
    }
  }

  class SbsFrameRoiTransformGpu: public testing::Test {
  protected:
    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL levels[] {D3D_FEATURE_LEVEL_11_0};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_WARP,
        nullptr,
        D3D11_CREATE_DEVICE_SINGLETHREADED,
        levels,
        static_cast<UINT>(std::size(levels)),
        D3D11_SDK_VERSION,
        &device_,
        &feature_level_,
        &context_
      )));

      const auto shader_path =
        std::filesystem::path(SUNSHINE_SHADERS_DIR) /
        "sbs_frame_roi_transform_cs.hlsl";
      ComPtr<ID3DBlob> bytecode;
      ComPtr<ID3DBlob> diagnostics;
      const auto compile_status = D3DCompileFromFile(
        shader_path.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main",
        "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS |
          D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &bytecode,
        &diagnostics
      );
      ASSERT_TRUE(SUCCEEDED(compile_status))
        << (diagnostics ?
              static_cast<const char *>(diagnostics->GetBufferPointer()) :
              "no compiler diagnostics");
      ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(
        bytecode->GetBufferPointer(),
        bytecode->GetBufferSize(),
        nullptr,
        &shader_
      )));
    }

    template<typename T>
    ComPtr<ID3D11ShaderResourceView> make_structured_input(
      const T &value,
      const std::uint32_t vector_count
    ) {
      D3D11_BUFFER_DESC descriptor {};
      descriptor.Usage = D3D11_USAGE_IMMUTABLE;
      descriptor.ByteWidth = sizeof(value);
      descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      descriptor.MiscFlags =
        D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      descriptor.StructureByteStride =
        4u * sizeof(std::uint32_t);
      D3D11_SUBRESOURCE_DATA initial {&value, 0, 0};
      ComPtr<ID3D11Buffer> buffer;
      if (FAILED(device_->CreateBuffer(
            &descriptor,
            &initial,
            &buffer
          ))) {
        ADD_FAILURE() << "Could not create structured input";
        return {};
      }

      D3D11_SHADER_RESOURCE_VIEW_DESC view_descriptor {};
      view_descriptor.Format = DXGI_FORMAT_UNKNOWN;
      view_descriptor.ViewDimension =
        D3D11_SRV_DIMENSION_BUFFER;
      view_descriptor.Buffer.FirstElement = 0;
      view_descriptor.Buffer.NumElements = vector_count;
      ComPtr<ID3D11ShaderResourceView> view;
      if (FAILED(device_->CreateShaderResourceView(
            buffer.Get(),
            &view_descriptor,
            &view
          ))) {
        ADD_FAILURE() << "Could not create structured-input SRV";
        return {};
      }
      return view;
    }

    frame_roi_transform_t dispatch(
      const rule_words_t &rule,
      const models::frame_roi_builder_constants &constants,
      const frame_roi_transform_t &previous = {}
    ) {
      const auto rule_view = make_structured_input(
        rule,
        static_cast<std::uint32_t>(
          sbs_scene_controller::rule_state_vector_count
        )
      );
      const auto previous_view = make_structured_input(
        previous,
        models::frame_roi_transform_vector_count
      );
      if (!rule_view || !previous_view) {
        return {};
      }

      D3D11_BUFFER_DESC output_descriptor {};
      output_descriptor.Usage = D3D11_USAGE_DEFAULT;
      output_descriptor.ByteWidth = sizeof(frame_roi_transform_t);
      output_descriptor.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      output_descriptor.MiscFlags =
        D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      output_descriptor.StructureByteStride =
        4u * sizeof(std::uint32_t);
      ComPtr<ID3D11Buffer> output_buffer;
      if (FAILED(device_->CreateBuffer(
            &output_descriptor,
            nullptr,
            &output_buffer
          ))) {
        ADD_FAILURE() << "Could not create transform output buffer";
        return {};
      }
      D3D11_UNORDERED_ACCESS_VIEW_DESC output_view_descriptor {};
      output_view_descriptor.Format = DXGI_FORMAT_UNKNOWN;
      output_view_descriptor.ViewDimension =
        D3D11_UAV_DIMENSION_BUFFER;
      output_view_descriptor.Buffer.FirstElement = 0;
      output_view_descriptor.Buffer.NumElements =
        models::frame_roi_transform_vector_count;
      ComPtr<ID3D11UnorderedAccessView> output_view;
      if (FAILED(device_->CreateUnorderedAccessView(
            output_buffer.Get(),
            &output_view_descriptor,
            &output_view
          ))) {
        ADD_FAILURE() << "Could not create transform output UAV";
        return {};
      }

      D3D11_BUFFER_DESC constants_descriptor {};
      constants_descriptor.Usage = D3D11_USAGE_IMMUTABLE;
      constants_descriptor.ByteWidth = sizeof(constants);
      constants_descriptor.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA constants_initial {&constants, 0, 0};
      ComPtr<ID3D11Buffer> constants_buffer;
      if (FAILED(device_->CreateBuffer(
            &constants_descriptor,
            &constants_initial,
            &constants_buffer
          ))) {
        ADD_FAILURE() << "Could not create transform constants";
        return {};
      }

      ID3D11ShaderResourceView *inputs[] {
        rule_view.Get(),
        previous_view.Get(),
      };
      ID3D11UnorderedAccessView *outputs[] {output_view.Get()};
      ID3D11Buffer *constant_buffers[] {constants_buffer.Get()};
      context_->CSSetShader(shader_.Get(), nullptr, 0);
      context_->CSSetShaderResources(0, 2, inputs);
      context_->CSSetUnorderedAccessViews(0, 1, outputs, nullptr);
      context_->CSSetConstantBuffers(0, 1, constant_buffers);
      context_->Dispatch(1, 1, 1);

      ID3D11ShaderResourceView *null_inputs[] {nullptr, nullptr};
      ID3D11UnorderedAccessView *null_outputs[] {nullptr};
      ID3D11Buffer *null_constants[] {nullptr};
      context_->CSSetShaderResources(0, 2, null_inputs);
      context_->CSSetUnorderedAccessViews(0, 1, null_outputs, nullptr);
      context_->CSSetConstantBuffers(0, 1, null_constants);
      context_->CSSetShader(nullptr, nullptr, 0);

      auto staging_descriptor = output_descriptor;
      staging_descriptor.Usage = D3D11_USAGE_STAGING;
      staging_descriptor.BindFlags = 0;
      staging_descriptor.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      staging_descriptor.MiscFlags = 0;
      ComPtr<ID3D11Buffer> staging;
      if (FAILED(device_->CreateBuffer(
            &staging_descriptor,
            nullptr,
            &staging
          ))) {
        ADD_FAILURE() << "Could not create transform staging buffer";
        return {};
      }
      context_->CopyResource(staging.Get(), output_buffer.Get());

      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context_->Map(
            staging.Get(),
            0,
            D3D11_MAP_READ,
            0,
            &mapped
          ))) {
        ADD_FAILURE() << "Could not map transform staging buffer";
        return {};
      }
      frame_roi_transform_t result;
      std::memcpy(&result, mapped.pData, sizeof(result));
      context_->Unmap(staging.Get(), 0);
      return result;
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    D3D_FEATURE_LEVEL feature_level_ {};
    ComPtr<ID3D11ComputeShader> shader_;
  };

  TEST_F(
    SbsFrameRoiTransformGpu,
    InactiveCanonicalFullFramePreservesExactLifecycleIdentity
  ) {
    const auto constants = make_constants(3840u, 2160u, 770u, 434u);
    const rule_words_t rule {};
    const auto actual = dispatch(rule, constants);

    EXPECT_EQ(
      actual.header,
      (std::array<std::uint32_t, 4> {
        models::frame_roi_transform_contract_version,
        flag_valid | flag_full_frame | flag_reset_debt,
        constants.source_frame_id_low,
        constants.source_frame_id_high,
      })
    );
    EXPECT_EQ(
      actual.identity,
      (std::array<std::uint32_t, 4> {
        0u,
        0u,
        constants.source_width,
        constants.source_height,
      })
    );
    EXPECT_EQ(
      actual.model,
      (std::array<std::uint32_t, 4> {
        constants.model_width,
        constants.model_height,
        constants.model_width * constants.model_height,
        0u,
      })
    );
    EXPECT_EQ(
      actual.diagnostics,
      (std::array<std::uint32_t, 4> {
        constants.transform_version_low,
        constants.transform_version_high,
        constants.gpu_bank_identity,
        fallback_inactive,
      })
    );
    expect_canonical_geometry(
      actual,
      constants.model_width,
      constants.model_height
    );
  }

  TEST_F(
    SbsFrameRoiTransformGpu,
    ActiveLandscapeAndPortraitProduceExactUnpaddedTransforms
  ) {
    const std::array<float, 4> landscape_roi {
      0.1f,
      0.2f,
      0.9f,
      0.8f,
    };
    auto landscape =
      make_constants(3840u, 2160u, 882u, 378u);
    configure_active(landscape, landscape_roi);
    const auto landscape_actual = dispatch(
      make_rule_state(landscape_roi),
      landscape
    );
    EXPECT_EQ(
      landscape_actual.header[1],
      flag_valid | flag_active_roi | flag_reset_debt
    );
    EXPECT_EQ(
      landscape_actual.identity,
      (std::array<std::uint32_t, 4> {
        landscape.expected_roi_generation,
        landscape.expected_backend_generation,
        landscape.source_width,
        landscape.source_height,
      })
    );
    EXPECT_EQ(landscape_actual.diagnostics[3], fallback_none);
    expect_active_geometry(
      landscape_actual,
      landscape,
      landscape_roi,
      {17u, 10u, 865u, 368u}
    );

    const std::array<float, 4> portrait_roi {
      0.2f,
      0.1f,
      0.8f,
      0.9f,
    };
    auto portrait =
      make_constants(2160u, 3840u, 378u, 882u);
    configure_active(portrait, portrait_roi);
    const auto portrait_actual = dispatch(
      make_rule_state(portrait_roi),
      portrait
    );
    EXPECT_EQ(
      portrait_actual.header[1],
      flag_valid | flag_active_roi | flag_reset_debt
    );
    EXPECT_EQ(portrait_actual.diagnostics[3], fallback_none);
    expect_active_geometry(
      portrait_actual,
      portrait,
      portrait_roi,
      {10u, 17u, 368u, 865u}
    );
  }

  TEST_F(
    SbsFrameRoiTransformGpu,
    DelayedSampleAcceptsOnlyNewerExactSameGenerationRule
  ) {
    const std::array<float, 4> roi {0.1f, 0.2f, 0.9f, 0.8f};
    auto constants =
      make_constants(3840u, 2160u, 882u, 378u);
    configure_active(constants, roi, 42u);

    const auto newer = dispatch(
      make_rule_state(roi, 45u),
      constants
    );
    ASSERT_TRUE(output_active(newer));
    EXPECT_EQ(newer.model[3], constants.shape_request_id);

    auto changed_roi = roi;
    changed_roi[2] = 0.85f;
    const auto roi_mismatch = dispatch(
      make_rule_state(changed_roi, 45u),
      constants
    );
    EXPECT_FALSE(output_valid(roi_mismatch));
    EXPECT_EQ(
      roi_mismatch.diagnostics[3],
      fallback_full_frame_shape_mismatch
    );

    const auto generation_mismatch = dispatch(
      make_rule_state(roi, 45u, 7u, 4u),
      constants
    );
    EXPECT_FALSE(output_valid(generation_mismatch));

    const auto backend_mismatch = dispatch(
      make_rule_state(roi, 45u, 8u, 3u),
      constants
    );
    EXPECT_FALSE(output_valid(backend_mismatch));

    auto older_constants = constants;
    older_constants.expected_rule_update_count = 46u;
    older_constants.shape_request_id =
      models::sbs_roi_shape_request_id_from_fields(
        models::sbs_roi_shape_request_schema_version,
        older_constants.expected_backend_generation,
        older_constants.expected_roi_generation,
        older_constants.expected_rule_update_count,
        older_constants.source_width,
        older_constants.source_height,
        older_constants.model_width,
        older_constants.model_height,
        older_constants.expected_committed_roi_bits
      );
    const auto stale_update = dispatch(
      make_rule_state(roi, 45u),
      older_constants
    );
    EXPECT_FALSE(output_valid(stale_update));
  }

  TEST_F(
    SbsFrameRoiTransformGpu,
    FullFrameFallbackRequestCannotAuthorizeLockedRoiWithSameHashFields
  ) {
    const std::array<float, 4> roi {0.1f, 0.2f, 0.9f, 0.8f};
    auto constants =
      make_constants(3840u, 2160u, 770u, 434u);
    configure_active(constants, roi);

    models::sbs_roi_shape_request fallback_request;
    fallback_request.header = {
      models::sbs_roi_shape_request_schema_version,
      models::sbs_roi_shape_flag_bits(
        models::sbs_roi_shape_request_flag::valid
      ) |
        models::sbs_roi_shape_flag_bits(
          models::sbs_roi_shape_request_flag::full_frame
        ) |
        models::sbs_roi_shape_flag_bits(
          models::sbs_roi_shape_request_flag::fallback
        ),
      static_cast<std::uint32_t>(
        models::sbs_roi_shape_request_reason::
          controller_schema_mismatch
      ),
      0u,
    };
    fallback_request.identity = {
      constants.expected_backend_generation,
      constants.expected_roi_generation,
      constants.expected_rule_update_count,
      models::sbs_roi_shape_controller_schema_float_bits,
    };
    fallback_request.shape = {
      constants.source_width,
      constants.source_height,
      constants.model_width,
      constants.model_height,
    };
    fallback_request.committed_roi_bits =
      constants.expected_committed_roi_bits;
    fallback_request.header[3] =
      models::sbs_roi_shape_request_id(fallback_request);

    const models::sbs_roi_shape_request_limits limits {
      constants.model_width,
      constants.model_height,
      constants.model_width,
      constants.model_height,
      4.0f,
    };
    ASSERT_TRUE(models::sbs_roi_shape_request_valid(
      fallback_request,
      limits
    ));
    ASSERT_EQ(
      fallback_request.header[3],
      constants.shape_request_id
    ) << "flags and fallback reason are intentionally outside the v1 hash";

    // Copy every request-derived field that the v1 builder ABI currently carries. The locked
    // controller record must still fail closed: a valid fallback record is not ROI authority,
    // even when its diagnostic hash aliases the active request's geometry/identity tuple.
    constants.expected_backend_generation =
      fallback_request.identity[0];
    constants.expected_roi_generation =
      fallback_request.identity[1];
    constants.expected_rule_update_count =
      fallback_request.identity[2];
    constants.shape_request_id =
      fallback_request.header[3];
    constants.expected_request_authority =
      (fallback_request.header[1] & 0xFFFFu) |
      ((fallback_request.header[2] & 0xFFFFu) << 16u);
    constants.expected_committed_roi_bits =
      fallback_request.committed_roi_bits;

    const auto actual = dispatch(
      make_rule_state(roi),
      constants
    );
    EXPECT_FALSE(output_active(actual));
  }

  TEST_F(
    SbsFrameRoiTransformGpu,
    MutatedControllerSchemaBitsFailClosed
  ) {
    const std::array<float, 4> roi {0.1f, 0.2f, 0.9f, 0.8f};
    auto constants =
      make_constants(3840u, 2160u, 882u, 378u);
    configure_active(constants, roi);
    auto rule = make_rule_state(roi);
    rule[word_index(
      sbs_scene_controller::rule_state_word_e::schema_version
    )] = float_bits(1.1f);

    const auto actual = dispatch(rule, constants);
    EXPECT_FALSE(output_valid(actual));
    EXPECT_FALSE(output_active(actual));
  }

  TEST_F(
    SbsFrameRoiTransformGpu,
    InvalidLifecycleRequestAndGeometryFailClosed
  ) {
    const std::array<float, 4> roi {0.1f, 0.2f, 0.9f, 0.8f};
    const auto rule = make_rule_state(roi);

    auto lifecycle =
      make_constants(3840u, 2160u, 770u, 434u);
    lifecycle.lifecycle_reserved = 0u;
    const auto lifecycle_actual = dispatch(rule, lifecycle);
    EXPECT_FALSE(output_valid(lifecycle_actual));
    EXPECT_EQ(lifecycle_actual.diagnostics[3], fallback_inactive);
    expect_canonical_geometry(
      lifecycle_actual,
      lifecycle.model_width,
      lifecycle.model_height
    );

    auto mismatched_request =
      make_constants(3840u, 2160u, 882u, 378u);
    configure_active(mismatched_request, roi);
    mismatched_request.shape_request_id ^= 1u;
    const auto mismatched_actual = dispatch(
      rule,
      mismatched_request
    );
    EXPECT_FALSE(output_valid(mismatched_actual));
    EXPECT_EQ(
      mismatched_actual.diagnostics[3],
      fallback_full_frame_shape_mismatch
    );
    expect_canonical_geometry(
      mismatched_actual,
      mismatched_request.model_width,
      mismatched_request.model_height
    );

    auto inactive_roi_shape =
      make_constants(3840u, 2160u, 882u, 378u);
    const auto inactive_roi_actual = dispatch(
      rule,
      inactive_roi_shape
    );
    EXPECT_FALSE(output_valid(inactive_roi_actual));
    EXPECT_EQ(
      inactive_roi_actual.diagnostics[3],
      fallback_full_frame_shape_mismatch
    );

    const std::array<float, 4> malformed_roi {
      std::bit_cast<float>(0x7FC00001u),
      0.2f,
      0.9f,
      0.8f,
    };
    auto malformed =
      make_constants(3840u, 2160u, 882u, 378u);
    configure_active(malformed, malformed_roi);
    const auto malformed_actual = dispatch(
      make_rule_state(malformed_roi),
      malformed
    );
    EXPECT_FALSE(output_valid(malformed_actual));
    EXPECT_EQ(
      malformed_actual.diagnostics[3],
      fallback_full_frame_shape_mismatch
    );

    auto impossible =
      make_constants(3840u, 2160u, 1036u, 14u);
    configure_active(impossible, roi);
    const auto impossible_actual = dispatch(rule, impossible);
    EXPECT_FALSE(output_valid(impossible_actual));
    EXPECT_EQ(
      impossible_actual.diagnostics[3],
      fallback_full_frame_shape_mismatch
    );
    expect_canonical_geometry(
      impossible_actual,
      impossible.model_width,
      impossible.model_height
    );
  }

  TEST_F(
    SbsFrameRoiTransformGpu,
    ResetDebtClearsOnlyAfterPriorDepthHasSameGeometry
  ) {
    const std::array<float, 4> roi {0.1f, 0.2f, 0.9f, 0.8f};
    auto first_constants =
      make_constants(3840u, 2160u, 882u, 378u);
    configure_active(first_constants, roi);
    const auto rule = make_rule_state(roi);
    const auto first = dispatch(rule, first_constants);
    ASSERT_TRUE(output_active(first));
    ASSERT_NE(first.header[1] & flag_reset_debt, 0u);

    auto same_constants = first_constants;
    same_constants.source_frame_id_low += 1u;
    same_constants.transform_version_low += 1u;
    same_constants.gpu_bank_identity = 0u;
    const auto same = dispatch(rule, same_constants, first);
    ASSERT_TRUE(output_active(same));
    EXPECT_EQ(same.header[1] & flag_reset_debt, 0u);
    EXPECT_EQ(same.header[2], same_constants.source_frame_id_low);
    EXPECT_EQ(same.diagnostics[0], same_constants.transform_version_low);
    EXPECT_EQ(same.diagnostics[2], same_constants.gpu_bank_identity);

    const std::array<float, 4> changed_roi {
      0.15f,
      0.2f,
      0.85f,
      0.8f,
    };
    auto changed_constants = same_constants;
    changed_constants.source_frame_id_low += 1u;
    changed_constants.transform_version_low += 1u;
    configure_active(changed_constants, changed_roi, 43u, 7u, 4u);
    const auto changed = dispatch(
      make_rule_state(changed_roi, 43u, 7u, 4u),
      changed_constants,
      same
    );
    ASSERT_TRUE(output_active(changed));
    EXPECT_NE(changed.header[1] & flag_reset_debt, 0u);
  }

  class DepthValidHistoryGpu: public testing::Test {
  protected:
    template<typename T>
    struct structured_resource_t {
      ComPtr<ID3D11Buffer> buffer;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };

    template<typename T>
    struct texture_resource_t {
      ComPtr<ID3D11Texture2D> texture;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };

    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL levels[] {D3D_FEATURE_LEVEL_11_0};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_WARP,
        nullptr,
        D3D11_CREATE_DEVICE_SINGLETHREADED,
        levels,
        static_cast<UINT>(std::size(levels)),
        D3D11_SDK_VERSION,
        &device_,
        &feature_level_,
        &context_
      )));

      const auto shader_path =
        std::filesystem::path(SUNSHINE_SHADERS_DIR) /
        "depth_valid_history_cs.hlsl";
      ComPtr<ID3DBlob> bytecode;
      ComPtr<ID3DBlob> diagnostics;
      const auto compile_status = D3DCompileFromFile(
        shader_path.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main",
        "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS |
          D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &bytecode,
        &diagnostics
      );
      ASSERT_TRUE(SUCCEEDED(compile_status))
        << (diagnostics ?
              static_cast<const char *>(diagnostics->GetBufferPointer()) :
              "no compiler diagnostics");
      ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(
        bytecode->GetBufferPointer(),
        bytecode->GetBufferSize(),
        nullptr,
        &shader_
      )));
    }

    template<typename T>
    structured_resource_t<T> make_structured(
      const std::vector<T> &values,
      const UINT bind_flags
    ) {
      structured_resource_t<T> result;
      if (values.empty()) {
        ADD_FAILURE() << "Structured test resource cannot be empty";
        return result;
      }

      D3D11_BUFFER_DESC descriptor {};
      descriptor.Usage = D3D11_USAGE_DEFAULT;
      descriptor.ByteWidth =
        static_cast<UINT>(values.size() * sizeof(T));
      descriptor.BindFlags = bind_flags;
      descriptor.MiscFlags =
        D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      descriptor.StructureByteStride = sizeof(T);
      D3D11_SUBRESOURCE_DATA initial {
        values.data(),
        0u,
        0u,
      };
      if (FAILED(device_->CreateBuffer(
            &descriptor,
            &initial,
            &result.buffer
          ))) {
        ADD_FAILURE() << "Could not create structured test buffer";
        return {};
      }

      if ((bind_flags & D3D11_BIND_SHADER_RESOURCE) != 0u) {
        D3D11_SHADER_RESOURCE_VIEW_DESC view_descriptor {};
        view_descriptor.Format = DXGI_FORMAT_UNKNOWN;
        view_descriptor.ViewDimension =
          D3D11_SRV_DIMENSION_BUFFER;
        view_descriptor.Buffer.FirstElement = 0u;
        view_descriptor.Buffer.NumElements =
          static_cast<UINT>(values.size());
        if (FAILED(device_->CreateShaderResourceView(
              result.buffer.Get(),
              &view_descriptor,
              &result.srv
            ))) {
          ADD_FAILURE() << "Could not create structured test SRV";
          return {};
        }
      }

      if ((bind_flags & D3D11_BIND_UNORDERED_ACCESS) != 0u) {
        D3D11_UNORDERED_ACCESS_VIEW_DESC view_descriptor {};
        view_descriptor.Format = DXGI_FORMAT_UNKNOWN;
        view_descriptor.ViewDimension =
          D3D11_UAV_DIMENSION_BUFFER;
        view_descriptor.Buffer.FirstElement = 0u;
        view_descriptor.Buffer.NumElements =
          static_cast<UINT>(values.size());
        if (FAILED(device_->CreateUnorderedAccessView(
              result.buffer.Get(),
              &view_descriptor,
              &result.uav
            ))) {
          ADD_FAILURE() << "Could not create structured test UAV";
          return {};
        }
      }
      return result;
    }

    template<typename T>
    texture_resource_t<T> make_texture(
      const UINT width,
      const UINT height,
      const DXGI_FORMAT format,
      const std::vector<T> &values,
      const UINT bind_flags
    ) {
      texture_resource_t<T> result;
      if (values.size() !=
          static_cast<std::size_t>(width) * height) {
        ADD_FAILURE() << "Texture test data has the wrong size";
        return result;
      }

      D3D11_TEXTURE2D_DESC descriptor {};
      descriptor.Width = width;
      descriptor.Height = height;
      descriptor.MipLevels = 1u;
      descriptor.ArraySize = 1u;
      descriptor.Format = format;
      descriptor.SampleDesc.Count = 1u;
      descriptor.Usage = D3D11_USAGE_DEFAULT;
      descriptor.BindFlags = bind_flags;
      D3D11_SUBRESOURCE_DATA initial {
        values.data(),
        width * static_cast<UINT>(sizeof(T)),
        0u,
      };
      if (FAILED(device_->CreateTexture2D(
            &descriptor,
            &initial,
            &result.texture
          ))) {
        ADD_FAILURE() << "Could not create texture test resource";
        return {};
      }
      if ((bind_flags & D3D11_BIND_SHADER_RESOURCE) != 0u &&
          FAILED(device_->CreateShaderResourceView(
            result.texture.Get(),
            nullptr,
            &result.srv
          ))) {
        ADD_FAILURE() << "Could not create texture test SRV";
        return {};
      }
      if ((bind_flags & D3D11_BIND_UNORDERED_ACCESS) != 0u &&
          FAILED(device_->CreateUnorderedAccessView(
            result.texture.Get(),
            nullptr,
            &result.uav
          ))) {
        ADD_FAILURE() << "Could not create texture test UAV";
        return {};
      }
      return result;
    }

    template<typename T>
    std::vector<T> read_structured(
      ID3D11Buffer *source,
      const std::size_t count
    ) {
      D3D11_BUFFER_DESC descriptor {};
      source->GetDesc(&descriptor);
      descriptor.Usage = D3D11_USAGE_STAGING;
      descriptor.BindFlags = 0u;
      descriptor.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      descriptor.MiscFlags = 0u;
      ComPtr<ID3D11Buffer> staging;
      if (FAILED(device_->CreateBuffer(
            &descriptor,
            nullptr,
            &staging
          ))) {
        ADD_FAILURE() << "Could not create structured staging buffer";
        return {};
      }
      context_->CopyResource(staging.Get(), source);

      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context_->Map(
            staging.Get(),
            0u,
            D3D11_MAP_READ,
            0u,
            &mapped
          ))) {
        ADD_FAILURE() << "Could not map structured staging buffer";
        return {};
      }
      std::vector<T> result(count);
      std::memcpy(
        result.data(),
        mapped.pData,
        result.size() * sizeof(T)
      );
      context_->Unmap(staging.Get(), 0u);
      return result;
    }

    template<typename T>
    std::vector<T> read_texture(
      ID3D11Texture2D *source,
      const UINT width,
      const UINT height
    ) {
      D3D11_TEXTURE2D_DESC descriptor {};
      source->GetDesc(&descriptor);
      descriptor.Usage = D3D11_USAGE_STAGING;
      descriptor.BindFlags = 0u;
      descriptor.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      descriptor.MiscFlags = 0u;
      ComPtr<ID3D11Texture2D> staging;
      if (FAILED(device_->CreateTexture2D(
            &descriptor,
            nullptr,
            &staging
          ))) {
        ADD_FAILURE() << "Could not create texture staging resource";
        return {};
      }
      context_->CopyResource(staging.Get(), source);

      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context_->Map(
            staging.Get(),
            0u,
            D3D11_MAP_READ,
            0u,
            &mapped
          ))) {
        ADD_FAILURE() << "Could not map texture staging resource";
        return {};
      }
      std::vector<T> result(
        static_cast<std::size_t>(width) * height
      );
      for (UINT y = 0u; y < height; ++y) {
        std::memcpy(
          result.data() + static_cast<std::size_t>(y) * width,
          static_cast<const std::byte *>(mapped.pData) +
            static_cast<std::size_t>(y) * mapped.RowPitch,
          static_cast<std::size_t>(width) * sizeof(T)
        );
      }
      context_->Unmap(staging.Get(), 0u);
      return result;
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    D3D_FEATURE_LEVEL feature_level_ {};
    ComPtr<ID3D11ComputeShader> shader_;
  };

  TEST_F(
    DepthValidHistoryGpu,
    ActivePromotionCopiesExactTransformAcrossMultipleThreadgroups
  ) {
    using float4_t = std::array<float, 4>;
    using uint4_t = std::array<std::uint32_t, 4>;
    constexpr UINT width = 42u;
    constexpr UINT height = 28u;
    constexpr UINT plane = width * height;
    constexpr std::array<std::uint32_t, 4> accepted {
      10u,
      7u,
      31u,
      21u,
    };

    std::vector<float4_t> minmax(1u);
    minmax[0][3] = 1.0f;
    std::vector<float> model(3u * plane);
    std::vector<float> ordinal(plane);
    std::vector<float> raw_depth(plane);
    std::vector<float> depth(plane);
    for (UINT index = 0u; index < plane; ++index) {
      model[index] = 0.10f + 0.0001f * index;
      model[index + plane] = 0.20f + 0.0001f * index;
      model[index + 2u * plane] = 0.30f + 0.0001f * index;
      ordinal[index] = static_cast<float>(index % 251u) / 250.0f;
      raw_depth[index] = 0.40f + 0.0001f * index;
      depth[index] = 0.50f + 0.0001f * index;
    }
    std::vector<float4_t> subject(
      sbs_adaptive_state::vector_count
    );
    subject[2][3] = 1.0f;

    const auto focus = rect_bits({0.25f, 0.25f, 0.75f, 0.75f});
    const auto canonical =
      rect_bits({0.0f, 0.0f, 1.0f, 1.0f});
    const auto feather =
      rect_bits({0.01f, 0.01f, 0.01f, 0.01f});
    const std::vector<uint4_t> transform {
      {
        models::frame_roi_transform_contract_version,
        flag_valid | flag_active_roi,
        0x55667788u,
        0x11223344u,
      },
      {3u, 7u, 420u, 280u},
      {
        width,
        height,
        (accepted[2] - accepted[0]) *
          (accepted[3] - accepted[1]),
        0x12345678u,
      },
      focus,
      canonical,
      accepted,
      feather,
      {9u, 0u, 1u, fallback_none},
    };
    ASSERT_EQ(
      transform.size(),
      models::frame_roi_transform_vector_count
    );

    const auto minmax_resource = make_structured(
      minmax,
      D3D11_BIND_SHADER_RESOURCE
    );
    const auto model_resource = make_structured(
      model,
      D3D11_BIND_SHADER_RESOURCE
    );
    const auto ordinal_resource = make_structured(
      ordinal,
      D3D11_BIND_SHADER_RESOURCE
    );
    const auto subject_resource = make_structured(
      subject,
      D3D11_BIND_SHADER_RESOURCE
    );
    const auto current_depth_resource = make_texture(
      width,
      height,
      DXGI_FORMAT_R32_FLOAT,
      depth,
      D3D11_BIND_SHADER_RESOURCE
    );
    const auto raw_depth_resource = make_structured(
      raw_depth,
      D3D11_BIND_SHADER_RESOURCE
    );
    const auto transform_resource = make_structured(
      transform,
      D3D11_BIND_SHADER_RESOURCE
    );

    const auto previous_model_resource = make_structured(
      std::vector<float>(3u * plane, -9.0f),
      D3D11_BIND_UNORDERED_ACCESS
    );
    const auto previous_ordinal_resource = make_structured(
      std::vector<float>(plane, -9.0f),
      D3D11_BIND_UNORDERED_ACCESS
    );
    const auto previous_depth_resource = make_texture(
      width,
      height,
      DXGI_FORMAT_R32_FLOAT,
      std::vector<float>(plane, -9.0f),
      D3D11_BIND_UNORDERED_ACCESS
    );
    std::vector<uint4_t> transform_sentinel(
      models::frame_roi_transform_vector_count,
      {0xDEADBEEFu, 0xBAADF00Du, 0xCCCCCCCCu, 0x55555555u}
    );
    const auto previous_transform_resource = make_structured(
      transform_sentinel,
      D3D11_BIND_UNORDERED_ACCESS
    );
    const auto previous_validity_resource = make_texture(
      width,
      height,
      DXGI_FORMAT_R32_UINT,
      std::vector<std::uint32_t>(plane, 7u),
      D3D11_BIND_UNORDERED_ACCESS
    );

    ASSERT_TRUE(minmax_resource.srv);
    ASSERT_TRUE(model_resource.srv);
    ASSERT_TRUE(ordinal_resource.srv);
    ASSERT_TRUE(subject_resource.srv);
    ASSERT_TRUE(current_depth_resource.srv);
    ASSERT_TRUE(raw_depth_resource.srv);
    ASSERT_TRUE(transform_resource.srv);
    ASSERT_TRUE(previous_model_resource.uav);
    ASSERT_TRUE(previous_ordinal_resource.uav);
    ASSERT_TRUE(previous_depth_resource.uav);
    ASSERT_TRUE(previous_transform_resource.uav);
    ASSERT_TRUE(previous_validity_resource.uav);

    std::array<std::uint32_t, 16> constants {};
    constants[0] = width;
    constants[1] = height;
    D3D11_BUFFER_DESC constants_descriptor {};
    constants_descriptor.Usage = D3D11_USAGE_IMMUTABLE;
    constants_descriptor.ByteWidth = sizeof(constants);
    constants_descriptor.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA constants_initial {
      constants.data(),
      0u,
      0u,
    };
    ComPtr<ID3D11Buffer> constants_buffer;
    ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(
      &constants_descriptor,
      &constants_initial,
      &constants_buffer
    )));

    ID3D11ShaderResourceView *inputs[] {
      minmax_resource.srv.Get(),
      model_resource.srv.Get(),
      ordinal_resource.srv.Get(),
      subject_resource.srv.Get(),
      current_depth_resource.srv.Get(),
      raw_depth_resource.srv.Get(),
      transform_resource.srv.Get(),
    };
    ID3D11UnorderedAccessView *outputs[] {
      previous_model_resource.uav.Get(),
      previous_ordinal_resource.uav.Get(),
      previous_depth_resource.uav.Get(),
      previous_transform_resource.uav.Get(),
      previous_validity_resource.uav.Get(),
    };
    ID3D11Buffer *constant_buffers[] {constants_buffer.Get()};
    context_->CSSetShader(shader_.Get(), nullptr, 0u);
    context_->CSSetShaderResources(0u, 7u, inputs);
    context_->CSSetUnorderedAccessViews(
      0u,
      5u,
      outputs,
      nullptr
    );
    context_->CSSetConstantBuffers(0u, 1u, constant_buffers);
    context_->Dispatch(
      (width + 15u) / 16u,
      (height + 15u) / 16u,
      1u
    );

    ID3D11ShaderResourceView *null_inputs[7] {};
    ID3D11UnorderedAccessView *null_outputs[5] {};
    ID3D11Buffer *null_constants[] {nullptr};
    context_->CSSetShaderResources(0u, 7u, null_inputs);
    context_->CSSetUnorderedAccessViews(
      0u,
      5u,
      null_outputs,
      nullptr
    );
    context_->CSSetConstantBuffers(0u, 1u, null_constants);
    context_->CSSetShader(nullptr, nullptr, 0u);

    const auto promoted_transform = read_structured<uint4_t>(
      previous_transform_resource.buffer.Get(),
      models::frame_roi_transform_vector_count
    );
    ASSERT_EQ(promoted_transform.size(), transform.size());
    EXPECT_EQ(promoted_transform, transform);

    const auto validity = read_texture<std::uint32_t>(
      previous_validity_resource.texture.Get(),
      width,
      height
    );
    ASSERT_EQ(validity.size(), plane);
    for (UINT y = 0u; y < height; ++y) {
      for (UINT x = 0u; x < width; ++x) {
        const bool inside =
          x >= accepted[0] && x < accepted[2] &&
          y >= accepted[1] && y < accepted[3];
        EXPECT_EQ(validity[y * width + x], inside ? 1u : 0u)
          << "pixel (" << x << ", " << y << ')';
      }
    }
  }
}  // namespace

#endif
