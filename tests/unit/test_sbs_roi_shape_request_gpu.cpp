/**
 * @file tests/unit/test_sbs_roi_shape_request_gpu.cpp
 * @brief WARP execution tests for the exact GPU ROI shape-request contract.
 */

#ifdef _WIN32

  #include <array>
  #include <bit>
  #include <cstdint>
  #include <cstring>
  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <filesystem>
  #include <gtest/gtest.h>
  #include <limits>
  #include <src/generated/sbs_scene_controller_contract.h>
  #include <src/sbs_roi_shape_request.h>
  #include <string_view>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  using rule_words_t = std::array<
    std::uint32_t,
    sbs_scene_controller::rule_state_word_count>;

  struct alignas(16) shape_constants_t {
    std::uint32_t source_width = 3840u;
    std::uint32_t source_height = 2160u;
    std::uint32_t canonical_model_width = 770u;
    std::uint32_t canonical_model_height = 434u;

    std::uint32_t target_pixel_budget = 770u * 434u;
    std::uint32_t profile_max_width = 1036u;
    std::uint32_t profile_max_height = 1036u;
    std::uint32_t expected_backend_generation = 7u;

    float quiet_halo_cells = 2.0f;
    std::uint32_t analysis_canvas_size = 128u;
    float max_model_aspect = 4.0f;
    std::uint32_t active_rules = 1u;

    std::array<std::uint32_t, 4> reserved {};
  };

  static_assert(sizeof(shape_constants_t) == 64u);
  static_assert(alignof(shape_constants_t) == 16u);

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

  rule_words_t make_rule_state(
    const std::array<float, 4> &roi,
    const std::uint32_t update_count = 42u,
    const std::uint32_t flags =
      sbs_scene_controller::state_flags_initialized |
      sbs_scene_controller::state_flags_roi_locked
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
    )] = 7u;
    words[word_index(
      sbs_scene_controller::rule_state_word_e::roi_generation
    )] = 3u;
    words[word_index(
      sbs_scene_controller::rule_state_word_e::update_count
    )] = update_count;
    words[word_index(
      sbs_scene_controller::rule_state_word_e::state_flags
    )] = flags;
    words[word_index(
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    )] = float_bits(roi[0]);
    words[word_index(
      sbs_scene_controller::rule_state_word_e::committed_roi_y0
    )] = float_bits(roi[1]);
    words[word_index(
      sbs_scene_controller::rule_state_word_e::committed_roi_x1
    )] = float_bits(roi[2]);
    words[word_index(
      sbs_scene_controller::rule_state_word_e::committed_roi_y1
    )] = float_bits(roi[3]);
    return words;
  }

  models::sbs_roi_shape_request expected_request(
    const rule_words_t &rule,
    const shape_constants_t &constants,
    const std::uint32_t flags,
    const models::sbs_roi_shape_request_reason reason,
    const std::uint32_t model_width,
    const std::uint32_t model_height
  ) {
    models::sbs_roi_shape_request expected;
    expected.header = {
      models::sbs_roi_shape_request_schema_version,
      flags,
      static_cast<std::uint32_t>(reason),
      0u,
    };
    expected.identity = {
      rule[word_index(
        sbs_scene_controller::rule_state_word_e::backend_generation
      )],
      rule[word_index(
        sbs_scene_controller::rule_state_word_e::roi_generation
      )],
      rule[word_index(
        sbs_scene_controller::rule_state_word_e::update_count
      )],
      rule[word_index(
        sbs_scene_controller::rule_state_word_e::schema_version
      )],
    };
    expected.shape = {
      constants.source_width,
      constants.source_height,
      model_width,
      model_height,
    };
    expected.committed_roi_bits = {
      rule[word_index(
        sbs_scene_controller::rule_state_word_e::committed_roi_x0
      )],
      rule[word_index(
        sbs_scene_controller::rule_state_word_e::committed_roi_y0
      )],
      rule[word_index(
        sbs_scene_controller::rule_state_word_e::committed_roi_x1
      )],
      rule[word_index(
        sbs_scene_controller::rule_state_word_e::committed_roi_y1
      )],
    };
    expected.header[3] = models::sbs_roi_shape_request_id(expected);
    return expected;
  }

  void expect_request_equal(
    const models::sbs_roi_shape_request &actual,
    const models::sbs_roi_shape_request &expected
  ) {
    EXPECT_EQ(actual.header, expected.header);
    EXPECT_EQ(actual.identity, expected.identity);
    EXPECT_EQ(actual.shape, expected.shape);
    EXPECT_EQ(actual.committed_roi_bits, expected.committed_roi_bits);
  }

  models::sbs_roi_shape_request_limits validation_limits(
    const shape_constants_t &constants
  ) {
    return {
      constants.canonical_model_width,
      constants.canonical_model_height,
      constants.profile_max_width,
      constants.profile_max_height,
      constants.max_model_aspect,
    };
  }

  TEST(
    SbsRoiShapeRequestContract,
    FullFrameAspectBudgetReducesAreaWithoutChangingGeometry
  ) {
    constexpr float source_aspect = 16.0f / 9.0f;
    constexpr float requested_short_side = 432.0f;
    constexpr float budget_aspect = 1.5f;
    const float actual_short_side =
      models::sbs_roi_budgeted_full_frame_short_side(
        source_aspect,
        requested_short_side,
        budget_aspect
      );

    EXPECT_LT(actual_short_side, requested_short_side);
    EXPECT_NEAR(
      actual_short_side * actual_short_side * source_aspect,
      requested_short_side * requested_short_side * budget_aspect,
      1.0f
    );
    EXPECT_FLOAT_EQ(
      models::sbs_roi_budgeted_full_frame_short_side(
        source_aspect,
        requested_short_side,
        4.0f
      ),
      requested_short_side
    );
    EXPECT_FLOAT_EQ(
      models::sbs_roi_budgeted_full_frame_short_side(
        1.0f / source_aspect,
        requested_short_side,
        budget_aspect
      ),
      actual_short_side
    );

    const auto landscape = models::sbs_roi_full_frame_model_shape(
      source_aspect,
      requested_short_side,
      budget_aspect,
      1036u,
      1036u
    );
    EXPECT_EQ(landscape, (std::array<std::uint32_t, 2> {700u, 392u}));
    EXPECT_NEAR(
      static_cast<float>(landscape[0]) /
        static_cast<float>(landscape[1]),
      source_aspect,
      source_aspect * 0.02f
    );
    const auto ultrawide = models::sbs_roi_full_frame_model_shape(
      5120.0f / 720.0f,
      requested_short_side,
      4.0f,
      1036u,
      720u
    );
    EXPECT_EQ(ultrawide, (std::array<std::uint32_t, 2> {994u, 140u}));
    EXPECT_EQ(
      models::sbs_roi_full_frame_model_shape(
        source_aspect,
        requested_short_side,
        budget_aspect,
        13u,
        1036u
      ),
      (std::array<std::uint32_t, 2> {})
    );
    EXPECT_EQ(
      models::sbs_roi_full_frame_model_shape(
        1.5f,
        14.0f,
        4.0f,
        28u,
        14u
      ),
      (std::array<std::uint32_t, 2> {})
    );
    EXPECT_EQ(
      models::sbs_roi_full_frame_model_shape(
        std::numeric_limits<float>::max(),
        requested_short_side,
        4.0f,
        1036u,
        1036u
      ),
      (std::array<std::uint32_t, 2> {})
    );
  }

  class SbsRoiShapeRequestGpu: public testing::Test {
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
        "sbs_roi_shape_request_cs.hlsl";
      ComPtr<ID3DBlob> bytecode;
      ComPtr<ID3DBlob> diagnostics;
      const auto compile_status = D3DCompileFromFile(
        shader_path.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main",
        "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS |
          D3DCOMPILE_WARNINGS_ARE_ERRORS |
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

    models::sbs_roi_shape_request dispatch(
      const rule_words_t &rule,
      const shape_constants_t &constants
    ) {
      D3D11_BUFFER_DESC rule_descriptor {};
      rule_descriptor.Usage = D3D11_USAGE_IMMUTABLE;
      rule_descriptor.ByteWidth = sizeof(rule);
      rule_descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      rule_descriptor.MiscFlags =
        D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      rule_descriptor.StructureByteStride =
        4u * sizeof(std::uint32_t);
      D3D11_SUBRESOURCE_DATA rule_initial {rule.data(), 0, 0};
      ComPtr<ID3D11Buffer> rule_buffer;
      if (FAILED(device_->CreateBuffer(
            &rule_descriptor,
            &rule_initial,
            &rule_buffer
          ))) {
        ADD_FAILURE() << "Could not create rule-state buffer";
        return {};
      }
      ComPtr<ID3D11ShaderResourceView> rule_view;
      if (FAILED(device_->CreateShaderResourceView(
            rule_buffer.Get(),
            nullptr,
            &rule_view
          ))) {
        ADD_FAILURE() << "Could not create rule-state SRV";
        return {};
      }

      D3D11_BUFFER_DESC output_descriptor {};
      output_descriptor.Usage = D3D11_USAGE_DEFAULT;
      output_descriptor.ByteWidth =
        sizeof(models::sbs_roi_shape_request);
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
        ADD_FAILURE() << "Could not create request output buffer";
        return {};
      }
      ComPtr<ID3D11UnorderedAccessView> output_view;
      if (FAILED(device_->CreateUnorderedAccessView(
            output_buffer.Get(),
            nullptr,
            &output_view
          ))) {
        ADD_FAILURE() << "Could not create request output UAV";
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
        ADD_FAILURE() << "Could not create request constants";
        return {};
      }

      ID3D11ShaderResourceView *inputs[] {rule_view.Get()};
      ID3D11UnorderedAccessView *outputs[] {output_view.Get()};
      ID3D11Buffer *constant_buffers[] {constants_buffer.Get()};
      context_->CSSetShader(shader_.Get(), nullptr, 0);
      context_->CSSetShaderResources(0, 1, inputs);
      context_->CSSetUnorderedAccessViews(0, 1, outputs, nullptr);
      context_->CSSetConstantBuffers(0, 1, constant_buffers);
      context_->Dispatch(1, 1, 1);

      ID3D11ShaderResourceView *null_inputs[] {nullptr};
      ID3D11UnorderedAccessView *null_outputs[] {nullptr};
      ID3D11Buffer *null_constants[] {nullptr};
      context_->CSSetShaderResources(0, 1, null_inputs);
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
        ADD_FAILURE() << "Could not create request staging buffer";
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
        ADD_FAILURE() << "Could not map request staging buffer";
        return {};
      }

      models::sbs_roi_shape_request result;
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
    SbsRoiShapeRequestGpu,
    ActiveLandscapeIsExactStableAndBitSensitive
  ) {
    const shape_constants_t constants;
    const auto rule = make_rule_state({0.1f, 0.2f, 0.9f, 0.8f});
    const auto first = dispatch(rule, constants);
    const auto second = dispatch(rule, constants);

    const auto expected = expected_request(
      rule,
      constants,
      models::sbs_roi_shape_flag_bits(
        models::sbs_roi_shape_request_flag::valid
      ) |
        models::sbs_roi_shape_flag_bits(
          models::sbs_roi_shape_request_flag::active_roi
        ),
      models::sbs_roi_shape_request_reason::none,
      882u,
      378u
    );
    expect_request_equal(first, expected);
    EXPECT_EQ(std::memcmp(&first, &second, sizeof(first)), 0);
    EXPECT_TRUE(models::sbs_roi_shape_request_valid(
      first,
      validation_limits(constants)
    ));

    auto changed_roi = rule;
    changed_roi[word_index(
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    )] ^= 1u;
    const auto roi_changed = dispatch(changed_roi, constants);
    EXPECT_NE(first.header[3], roi_changed.header[3]);
    EXPECT_EQ(first.shape, roi_changed.shape);

    const auto update_changed = dispatch(
      make_rule_state({0.1f, 0.2f, 0.9f, 0.8f}, 43u),
      constants
    );
    EXPECT_NE(first.header[3], update_changed.header[3]);
  }

  TEST_F(SbsRoiShapeRequestGpu, ActivePortraitUsesPortraitShape) {
    shape_constants_t constants;
    constants.source_width = 2160u;
    constants.source_height = 3840u;
    constants.canonical_model_width = 434u;
    constants.canonical_model_height = 770u;
    const auto rule = make_rule_state({0.2f, 0.1f, 0.8f, 0.9f});

    const auto actual = dispatch(rule, constants);
    const auto expected = expected_request(
      rule,
      constants,
      models::sbs_roi_shape_flag_bits(
        models::sbs_roi_shape_request_flag::valid
      ) |
        models::sbs_roi_shape_flag_bits(
          models::sbs_roi_shape_request_flag::active_roi
        ),
      models::sbs_roi_shape_request_reason::none,
      378u,
      882u
    );
    expect_request_equal(actual, expected);
    EXPECT_TRUE(models::sbs_roi_shape_request_valid(
      actual,
      validation_limits(constants)
    ));
  }

  TEST_F(SbsRoiShapeRequestGpu, NativeAndProfileCapsClampShape) {
    shape_constants_t constants;
    constants.source_width = 1280u;
    constants.source_height = 720u;
    constants.canonical_model_width = 504u;
    constants.canonical_model_height = 280u;
    constants.profile_max_width = 504u;
    constants.profile_max_height = 280u;
    const auto rule = make_rule_state({0.3f, 0.3f, 0.7f, 0.7f});

    const auto actual = dispatch(rule, constants);
    const auto expected = expected_request(
      rule,
      constants,
      models::sbs_roi_shape_flag_bits(
        models::sbs_roi_shape_request_flag::valid
      ) |
        models::sbs_roi_shape_flag_bits(
          models::sbs_roi_shape_request_flag::active_roi
        ) |
        models::sbs_roi_shape_flag_bits(
          models::sbs_roi_shape_request_flag::profile_clamped
        ),
      models::sbs_roi_shape_request_reason::none,
      504u,
      280u
    );
    expect_request_equal(actual, expected);
    EXPECT_LE(actual.shape[2], constants.profile_max_width);
    EXPECT_LE(actual.shape[3], constants.profile_max_height);
    EXPECT_TRUE(models::sbs_roi_shape_request_valid(
      actual,
      validation_limits(constants)
    ));
  }

  TEST_F(
    SbsRoiShapeRequestGpu,
    AspectPreservingCanonicalFallbackMayExceedActiveRoiAspectCap
  ) {
    shape_constants_t constants;
    constants.source_width = 5120u;
    constants.source_height = 720u;
    constants.canonical_model_width = 994u;
    constants.canonical_model_height = 140u;
    constants.target_pixel_budget =
      constants.canonical_model_width *
      constants.canonical_model_height;
    constants.max_model_aspect = 4.0f;
    constants.active_rules = 0u;
    const auto rule = make_rule_state({0.4f, 0.1f, 0.6f, 0.9f});

    const auto fallback = dispatch(rule, constants);
    const auto expected = expected_request(
      rule,
      constants,
      models::sbs_roi_shape_flag_bits(
        models::sbs_roi_shape_request_flag::valid
      ) |
        models::sbs_roi_shape_flag_bits(
          models::sbs_roi_shape_request_flag::full_frame
        ) |
        models::sbs_roi_shape_flag_bits(
          models::sbs_roi_shape_request_flag::fallback
        ),
      models::sbs_roi_shape_request_reason::inactive,
      constants.canonical_model_width,
      constants.canonical_model_height
    );
    expect_request_equal(fallback, expected);
    EXPECT_GT(
      static_cast<float>(fallback.shape[2]) /
        static_cast<float>(fallback.shape[3]),
      constants.max_model_aspect
    );
    EXPECT_TRUE(models::sbs_roi_shape_request_valid(
      fallback,
      validation_limits(constants)
    ));

    // The same canonical source still produces an active request inside the configured aspect
    // envelope. Only exact full-frame fallback is exempt from the active model-shape cap.
    constants.active_rules = 1u;
    const auto active = dispatch(rule, constants);
    EXPECT_TRUE(models::sbs_roi_shape_has_flag(
      active,
      models::sbs_roi_shape_request_flag::active_roi
    ));
    EXPECT_LE(
      static_cast<float>(active.shape[2]) /
        static_cast<float>(active.shape[3]),
      constants.max_model_aspect
    );
    EXPECT_TRUE(models::sbs_roi_shape_request_valid(
      active,
      validation_limits(constants)
    ));
  }

  TEST_F(
    SbsRoiShapeRequestGpu,
    NoLockAndMalformedRoiUseCanonicalFallback
  ) {
    const shape_constants_t constants;
    const auto no_lock = make_rule_state(
      {0.1f, 0.2f, 0.9f, 0.8f},
      42u,
      sbs_scene_controller::state_flags_initialized
    );
    const auto no_lock_actual = dispatch(no_lock, constants);
    const auto no_lock_expected = expected_request(
      no_lock,
      constants,
      models::sbs_roi_shape_flag_bits(
        models::sbs_roi_shape_request_flag::valid
      ) |
        models::sbs_roi_shape_flag_bits(
          models::sbs_roi_shape_request_flag::full_frame
        ) |
        models::sbs_roi_shape_flag_bits(
          models::sbs_roi_shape_request_flag::fallback
        ),
      models::sbs_roi_shape_request_reason::roi_not_locked,
      constants.canonical_model_width,
      constants.canonical_model_height
    );
    expect_request_equal(no_lock_actual, no_lock_expected);
    EXPECT_TRUE(models::sbs_roi_shape_request_valid(
      no_lock_actual,
      validation_limits(constants)
    ));

    auto malformed = make_rule_state({0.1f, 0.2f, 0.9f, 0.8f});
    malformed[word_index(
      sbs_scene_controller::rule_state_word_e::committed_roi_x0
    )] = 0x7FC00001u;
    const auto malformed_actual = dispatch(malformed, constants);
    const auto malformed_expected = expected_request(
      malformed,
      constants,
      models::sbs_roi_shape_flag_bits(
        models::sbs_roi_shape_request_flag::valid
      ) |
        models::sbs_roi_shape_flag_bits(
          models::sbs_roi_shape_request_flag::full_frame
        ) |
        models::sbs_roi_shape_flag_bits(
          models::sbs_roi_shape_request_flag::fallback
        ),
      models::sbs_roi_shape_request_reason::malformed_committed_roi,
      constants.canonical_model_width,
      constants.canonical_model_height
    );
    expect_request_equal(malformed_actual, malformed_expected);
    EXPECT_TRUE(models::sbs_roi_shape_request_valid(
      malformed_actual,
      validation_limits(constants)
    ));
    EXPECT_EQ(
      malformed_actual.committed_roi_bits[0],
      0x7FC00001u
    );
  }

  TEST_F(
    SbsRoiShapeRequestGpu,
    DelayedSampleAcceptsOnlyNewerExactSameGenerationState
  ) {
    const shape_constants_t constants;
    const std::array<float, 4> roi {0.1f, 0.2f, 0.9f, 0.8f};
    const auto sampled_rule = make_rule_state(roi, 42u);
    const auto sampled = dispatch(sampled_rule, constants);
    const auto newer = dispatch(make_rule_state(roi, 45u), constants);

    EXPECT_TRUE(models::sbs_roi_shape_current_rule_matches(
      sampled,
      newer.identity[0],
      newer.identity[1],
      newer.identity[2],
      newer.shape[0],
      newer.shape[1],
      newer.committed_roi_bits
    ));
    EXPECT_TRUE(models::sbs_roi_shape_same_bound_rule_state(
      sampled,
      newer
    ));
    EXPECT_NE(sampled.header[3], newer.header[3]);

    EXPECT_FALSE(models::sbs_roi_shape_current_rule_matches(
      sampled,
      newer.identity[0],
      newer.identity[1],
      sampled.identity[2] - 1u,
      newer.shape[0],
      newer.shape[1],
      newer.committed_roi_bits
    ));
    auto different_roi_bits = newer.committed_roi_bits;
    different_roi_bits[2] ^= 1u;
    EXPECT_FALSE(models::sbs_roi_shape_current_rule_matches(
      sampled,
      newer.identity[0],
      newer.identity[1],
      newer.identity[2],
      newer.shape[0],
      newer.shape[1],
      different_roi_bits
    ));
    EXPECT_FALSE(models::sbs_roi_shape_current_rule_matches(
      sampled,
      newer.identity[0],
      newer.identity[1] + 1u,
      newer.identity[2],
      newer.shape[0],
      newer.shape[1],
      newer.committed_roi_bits
    ));
  }
}  // namespace

#endif
