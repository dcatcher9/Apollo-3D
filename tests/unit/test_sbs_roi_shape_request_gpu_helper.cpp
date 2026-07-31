/**
 * @file tests/unit/test_sbs_roi_shape_request_gpu_helper.cpp
 * @brief WARP tests for the production nonblocking ROI shape-request readback helper.
 */

#ifdef _WIN32

  #include <array>
  #include <bit>
  #include <cstdint>
  #include <d3d11.h>
  #include <filesystem>
  #include <gtest/gtest.h>
  #include <memory>
  #include <src/generated/sbs_scene_controller_contract.h>
  #include <src/sbs_roi_shape_request_gpu.h>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;
  using rule_words_t = std::array<
    std::uint32_t,
    sbs_scene_controller::rule_state_word_count>;

  constexpr std::size_t word_index(
    const sbs_scene_controller::rule_state_word_e word
  ) {
    return sbs_scene_controller::index(word);
  }

  std::uint32_t float_bits(const float value) {
    return std::bit_cast<std::uint32_t>(value);
  }

  rule_words_t make_rule_state(
    const std::uint32_t backend_generation = 7u,
    const std::uint32_t roi_generation = 3u,
    const std::uint32_t update_count = 42u,
    const std::array<float, 4> &roi =
      {0.1f, 0.2f, 0.9f, 0.8f}
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

  std::array<std::uint32_t, 4> roi_bits(
    const rule_words_t &words
  ) {
    return {
      words[word_index(
        sbs_scene_controller::rule_state_word_e::committed_roi_x0
      )],
      words[word_index(
        sbs_scene_controller::rule_state_word_e::committed_roi_y0
      )],
      words[word_index(
        sbs_scene_controller::rule_state_word_e::committed_roi_x1
      )],
      words[word_index(
        sbs_scene_controller::rule_state_word_e::committed_roi_y1
      )],
    };
  }

  class SbsRoiShapeRequestGpuHelperTest: public testing::Test {
  protected:
    void SetUp() override {
      constexpr D3D_FEATURE_LEVEL levels[] {
        D3D_FEATURE_LEVEL_11_0,
      };
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

      const auto assets_dir =
        std::filesystem::path(SUNSHINE_SHADERS_DIR)
          .parent_path()
          .parent_path();
      helper_ =
        std::make_unique<models::sbs_roi_shape_request_gpu>(
          device_,
          context_,
          assets_dir
        );
      ASSERT_TRUE(helper_->valid());

      D3D11_BUFFER_DESC gpu_sync_descriptor {};
      gpu_sync_descriptor.Usage = D3D11_USAGE_DEFAULT;
      gpu_sync_descriptor.ByteWidth = 16u;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(
        &gpu_sync_descriptor,
        nullptr,
        &gpu_sync_source_
      )));
      auto staging_sync_descriptor = gpu_sync_descriptor;
      staging_sync_descriptor.Usage = D3D11_USAGE_STAGING;
      staging_sync_descriptor.CPUAccessFlags =
        D3D11_CPU_ACCESS_READ;
      ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(
        &staging_sync_descriptor,
        nullptr,
        &gpu_sync_staging_
      )));
    }

    ComPtr<ID3D11ShaderResourceView> make_rule_view(
      ID3D11Device *device,
      const rule_words_t &words
    ) {
      D3D11_BUFFER_DESC descriptor {};
      descriptor.Usage = D3D11_USAGE_IMMUTABLE;
      descriptor.ByteWidth = sizeof(words);
      descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      descriptor.MiscFlags =
        D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      descriptor.StructureByteStride =
        4u * sizeof(std::uint32_t);
      D3D11_SUBRESOURCE_DATA initial {
        words.data(),
        0,
        0,
      };
      ComPtr<ID3D11Buffer> buffer;
      if (FAILED(device->CreateBuffer(
            &descriptor,
            &initial,
            &buffer
          ))) {
        ADD_FAILURE() << "Could not create rule-state buffer";
        return {};
      }

      ComPtr<ID3D11ShaderResourceView> view;
      if (FAILED(device->CreateShaderResourceView(
            buffer.Get(),
            nullptr,
            &view
          ))) {
        ADD_FAILURE() << "Could not create rule-state SRV";
        return {};
      }
      return view;
    }

    ComPtr<ID3D11ShaderResourceView> make_malformed_texture_view() {
      D3D11_TEXTURE2D_DESC descriptor {};
      descriptor.Width = 1u;
      descriptor.Height = 1u;
      descriptor.MipLevels = 1u;
      descriptor.ArraySize = 1u;
      descriptor.Format = DXGI_FORMAT_R32_UINT;
      descriptor.SampleDesc.Count = 1u;
      descriptor.Usage = D3D11_USAGE_DEFAULT;
      descriptor.BindFlags = D3D11_BIND_SHADER_RESOURCE;

      const std::uint32_t value = 0xFFFFFFFFu;
      D3D11_SUBRESOURCE_DATA initial {
        &value,
        sizeof(value),
        sizeof(value),
      };
      ComPtr<ID3D11Texture2D> texture;
      if (FAILED(device_->CreateTexture2D(
            &descriptor,
            &initial,
            &texture
          ))) {
        ADD_FAILURE() << "Could not create malformed texture";
        return {};
      }

      ComPtr<ID3D11ShaderResourceView> view;
      if (FAILED(device_->CreateShaderResourceView(
            texture.Get(),
            nullptr,
            &view
          ))) {
        ADD_FAILURE() << "Could not create malformed texture SRV";
        return {};
      }
      return view;
    }

    models::sbs_roi_shape_request_gpu_submission submission(
      ID3D11ShaderResourceView *rule_view,
      const std::uint32_t source_width = 3840u,
      const std::uint32_t source_height = 2160u,
      const std::uint32_t backend_generation = 7u,
      const std::uint64_t source_frame_id = 101u
    ) const {
      models::sbs_roi_shape_request_gpu_submission value;
      value.rule_state = rule_view;
      value.source_frame_id = source_frame_id;
      value.source_width = source_width;
      value.source_height = source_height;
      value.canonical_model_width = 770u;
      value.canonical_model_height = 434u;
      value.target_pixel_budget = 770u * 434u;
      value.profile_max_width = 1036u;
      value.profile_max_height = 1036u;
      value.expected_backend_generation = backend_generation;
      value.quiet_halo_cells = 2.0f;
      value.analysis_canvas_size =
        static_cast<std::uint32_t>(
          sbs_scene_controller::analysis_canvas_size
        );
      value.max_model_aspect = 4.0f;
      value.active_rules = 1u;
      return value;
    }

    void test_only_flush_with_event() {
      // Production deliberately never flushes or waits. This benign event is only a bounded WARP
      // test driver so command completion does not depend on wall-clock sleeps or a busy loop.
      D3D11_QUERY_DESC descriptor {
        D3D11_QUERY_EVENT,
        0,
      };
      ComPtr<ID3D11Query> completion;
      ASSERT_TRUE(SUCCEEDED(device_->CreateQuery(
        &descriptor,
        &completion
      )));
      context_->End(completion.Get());
      context_->CopyResource(
        gpu_sync_staging_.Get(),
        gpu_sync_source_.Get()
      );
      context_->Flush();
      // A blocking map of an unrelated 16-byte copy is a deterministic test-only queue drain.
      // It is intentionally absent from the production helper, whose maps remain DO_NOT_WAIT.
      D3D11_MAPPED_SUBRESOURCE mapped {};
      ASSERT_TRUE(SUCCEEDED(context_->Map(
        gpu_sync_staging_.Get(),
        0,
        D3D11_MAP_READ,
        0,
        &mapped
      )));
      context_->Unmap(gpu_sync_staging_.Get(), 0);
    }

    models::sbs_roi_shape_request_gpu_result await_fresh_after(
      models::sbs_roi_shape_request_gpu &helper,
      const std::uint64_t sequence
    ) {
      models::sbs_roi_shape_request_gpu_result result;
      constexpr std::size_t max_poll_rounds = 32u;
      for (std::size_t round = 0u;
           round < max_poll_rounds;
           ++round) {
        result = helper.poll();
        if (
          result.fresh_sample &&
          result.completed_sequence > sequence
        ) {
          return result;
        }
        test_only_flush_with_event();
      }
      ADD_FAILURE()
        << "WARP did not complete a shape request in "
        << max_poll_rounds << " bounded polls; last failed="
        << result.failed << ", sequence="
        << result.completed_sequence;
      return result;
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    D3D_FEATURE_LEVEL feature_level_ {};
    ComPtr<ID3D11Buffer> gpu_sync_source_;
    ComPtr<ID3D11Buffer> gpu_sync_staging_;
    std::unique_ptr<models::sbs_roi_shape_request_gpu> helper_;
  };

  TEST_F(
    SbsRoiShapeRequestGpuHelperTest,
    ActiveOutputBecomesFreshThenRemainsCachedWithoutRelatching
  ) {
    const auto first_rule = make_rule_state();
    const auto first_view =
      make_rule_view(device_.Get(), first_rule);
    ASSERT_TRUE(first_view);

    const auto submitted =
      helper_->submit(submission(first_view.Get()));
    EXPECT_TRUE(submitted.copy_scheduled);
    EXPECT_FALSE(submitted.failed);
    EXPECT_FALSE(submitted.fresh_sample);
    EXPECT_EQ(submitted.completed_sequence, 0u);
    EXPECT_FALSE(submitted.request.has_value());

    const auto first = await_fresh_after(*helper_, 0u);
    ASSERT_TRUE(first.request.has_value());
    EXPECT_FALSE(first.failed);
    EXPECT_GT(first.completed_sequence, 0u);
    EXPECT_EQ(first.completed_source_frame_id, 101u);
    EXPECT_TRUE(models::sbs_roi_shape_has_flag(
      *first.request,
      models::sbs_roi_shape_request_flag::active_roi
    ));
    EXPECT_EQ(first.request->identity[0], 7u);
    EXPECT_EQ(first.request->identity[1], 3u);
    EXPECT_EQ(first.request->identity[2], 42u);

    const auto cached = helper_->poll();
    ASSERT_TRUE(cached.request.has_value());
    EXPECT_FALSE(cached.fresh_sample);
    EXPECT_FALSE(cached.copy_scheduled);
    EXPECT_FALSE(cached.failed);
    EXPECT_EQ(
      cached.completed_sequence,
      first.completed_sequence
    );
    EXPECT_EQ(
      cached.completed_source_frame_id,
      first.completed_source_frame_id
    );
    EXPECT_EQ(cached.request->header, first.request->header);

    const auto second_rule =
      make_rule_state(7u, 3u, 43u);
    const auto second_view =
      make_rule_view(device_.Get(), second_rule);
    ASSERT_TRUE(second_view);
    const auto second_submit =
      helper_->submit(submission(
        second_view.Get(),
        3840u,
        2160u,
        7u,
        102u
      ));
    EXPECT_TRUE(second_submit.copy_scheduled);
    EXPECT_FALSE(second_submit.failed);

    const auto second = await_fresh_after(
      *helper_,
      first.completed_sequence
    );
    ASSERT_TRUE(second.request.has_value());
    EXPECT_GT(
      second.completed_sequence,
      first.completed_sequence
    );
    EXPECT_EQ(second.completed_source_frame_id, 102u);
    // A proven paused player keeps exactly this state shape: update_count advances while the
    // ROI lock, generation, and committed rectangle remain unchanged. The downstream request
    // must therefore continue using ROI-shaped inference rather than silently falling back.
    EXPECT_TRUE(models::sbs_roi_shape_has_flag(
      *second.request,
      models::sbs_roi_shape_request_flag::active_roi
    ));
    EXPECT_FALSE(models::sbs_roi_shape_has_flag(
      *second.request,
      models::sbs_roi_shape_request_flag::full_frame
    ));
    EXPECT_EQ(second.request->identity[1], 3u);
    EXPECT_EQ(second.request->identity[2], 43u);
    EXPECT_EQ(second.request->committed_roi_bits, roi_bits(second_rule));
    EXPECT_EQ(second.request->shape[2], first.request->shape[2]);
    EXPECT_EQ(second.request->shape[3], first.request->shape[3]);
  }

  TEST_F(
    SbsRoiShapeRequestGpuHelperTest,
    ThreePendingSlotsApplyNormalBackpressure
  ) {
    const auto rule = make_rule_state();
    const auto view = make_rule_view(device_.Get(), rule);
    ASSERT_TRUE(view);
    const auto request = submission(view.Get());

    const auto first = helper_->submit(request);
    const auto second = helper_->submit(request);
    const auto third = helper_->submit(request);
    const auto saturated = helper_->submit(request);

    EXPECT_TRUE(first.copy_scheduled);
    EXPECT_TRUE(second.copy_scheduled);
    EXPECT_TRUE(third.copy_scheduled);
    EXPECT_FALSE(first.failed);
    EXPECT_FALSE(second.failed);
    EXPECT_FALSE(third.failed);
    EXPECT_FALSE(saturated.copy_scheduled);
    EXPECT_FALSE(saturated.failed);

    const auto completed = await_fresh_after(*helper_, 0u);
    ASSERT_TRUE(completed.request.has_value());
    EXPECT_GT(completed.completed_sequence, 0u);
  }

  TEST_F(
    SbsRoiShapeRequestGpuHelperTest,
    OlderSourceAndBackendAreDistinguishableAndSuperseded
  ) {
    const auto stale_rule = make_rule_state(7u, 3u, 42u);
    const auto stale_view =
      make_rule_view(device_.Get(), stale_rule);
    ASSERT_TRUE(stale_view);
    ASSERT_TRUE(
      helper_->submit(
        submission(stale_view.Get(), 3840u, 2160u, 7u)
      ).copy_scheduled
    );
    const auto stale = await_fresh_after(*helper_, 0u);
    ASSERT_TRUE(stale.request.has_value());

    const auto current_rule = make_rule_state(8u, 4u, 43u);
    EXPECT_FALSE(models::sbs_roi_shape_current_rule_matches(
      *stale.request,
      8u,
      4u,
      43u,
      1920u,
      1080u,
      roi_bits(current_rule)
    ));

    const auto current_view =
      make_rule_view(device_.Get(), current_rule);
    ASSERT_TRUE(current_view);
    ASSERT_TRUE(
      helper_->submit(
        submission(current_view.Get(), 1920u, 1080u, 8u)
      ).copy_scheduled
    );
    const auto current = await_fresh_after(
      *helper_,
      stale.completed_sequence
    );
    ASSERT_TRUE(current.request.has_value());
    EXPECT_GT(
      current.completed_sequence,
      stale.completed_sequence
    );
    EXPECT_EQ(current.request->identity[0], 8u);
    EXPECT_EQ(current.request->shape[0], 1920u);
    EXPECT_EQ(current.request->shape[1], 1080u);
    EXPECT_TRUE(models::sbs_roi_shape_current_rule_matches(
      *current.request,
      8u,
      4u,
      43u,
      1920u,
      1080u,
      roi_bits(current_rule)
    ));
  }

  TEST_F(
    SbsRoiShapeRequestGpuHelperTest,
    NullRuleViewPublishesCanonicalInactiveFallback
  ) {
    auto request = submission(nullptr);
    request.active_rules = 0u;
    const auto submitted = helper_->submit(request);
    EXPECT_TRUE(submitted.copy_scheduled);
    EXPECT_FALSE(submitted.failed);

    const auto completed = await_fresh_after(*helper_, 0u);
    ASSERT_TRUE(completed.request.has_value());
    EXPECT_TRUE(models::sbs_roi_shape_has_flag(
      *completed.request,
      models::sbs_roi_shape_request_flag::full_frame
    ));
    EXPECT_TRUE(models::sbs_roi_shape_has_flag(
      *completed.request,
      models::sbs_roi_shape_request_flag::fallback
    ));
    EXPECT_FALSE(models::sbs_roi_shape_has_flag(
      *completed.request,
      models::sbs_roi_shape_request_flag::active_roi
    ));
    EXPECT_EQ(
      completed.request->header[2],
      static_cast<std::uint32_t>(
        models::sbs_roi_shape_request_reason::inactive
      )
    );
    EXPECT_EQ(completed.request->shape[2], 770u);
    EXPECT_EQ(completed.request->shape[3], 434u);
  }

  TEST_F(
    SbsRoiShapeRequestGpuHelperTest,
    AspectPreservingCanonicalFallbackMayExceedActiveAspectCap
  ) {
    auto wrong_aspect = submission(
      nullptr,
      5120u,
      720u,
      7u,
      110u
    );
    wrong_aspect.canonical_model_width = 966u;
    wrong_aspect.canonical_model_height = 140u;
    wrong_aspect.target_pixel_budget = 966u * 140u;
    wrong_aspect.max_model_aspect = 4.0f;
    wrong_aspect.active_rules = 0u;
    const auto rejected = helper_->submit(wrong_aspect);
    EXPECT_TRUE(rejected.failed);
    EXPECT_FALSE(rejected.copy_scheduled);

    auto request = submission(
      nullptr,
      5120u,
      720u,
      7u,
      111u
    );
    request.canonical_model_width = 994u;
    request.canonical_model_height = 140u;
    request.target_pixel_budget = 994u * 140u;
    request.max_model_aspect = 4.0f;
    request.active_rules = 0u;
    ASSERT_TRUE(models::sbs_roi_full_frame_shape_matches(
      request.source_width,
      request.source_height,
      request.canonical_model_width,
      request.canonical_model_height
    ));

    const auto submitted = helper_->submit(request);
    EXPECT_TRUE(submitted.copy_scheduled);
    EXPECT_FALSE(submitted.failed);

    const auto completed = await_fresh_after(*helper_, 0u);
    ASSERT_TRUE(completed.request.has_value());
    EXPECT_TRUE(models::sbs_roi_shape_has_flag(
      *completed.request,
      models::sbs_roi_shape_request_flag::full_frame
    ));
    EXPECT_TRUE(models::sbs_roi_shape_has_flag(
      *completed.request,
      models::sbs_roi_shape_request_flag::fallback
    ));
    EXPECT_FALSE(models::sbs_roi_shape_has_flag(
      *completed.request,
      models::sbs_roi_shape_request_flag::active_roi
    ));
    EXPECT_EQ(completed.request->shape[0], 5120u);
    EXPECT_EQ(completed.request->shape[1], 720u);
    EXPECT_EQ(completed.request->shape[2], 994u);
    EXPECT_EQ(completed.request->shape[3], 140u);
  }

  TEST_F(
    SbsRoiShapeRequestGpuHelperTest,
    FreshCompletionSurvivesFailureSchedulingTheNextSubmission
  ) {
    const auto valid_rule = make_rule_state();
    const auto valid_view =
      make_rule_view(device_.Get(), valid_rule);
    ASSERT_TRUE(valid_view);
    const auto first_submit =
      helper_->submit(submission(
        valid_view.Get(),
        3840u,
        2160u,
        7u,
        201u
      ));
    ASSERT_TRUE(first_submit.copy_scheduled);
    ASSERT_FALSE(first_submit.failed);
    test_only_flush_with_event();

    const auto malformed = make_malformed_texture_view();
    ASSERT_TRUE(malformed);
    const auto combined = helper_->submit(submission(
      malformed.Get(),
      3840u,
      2160u,
      7u,
      202u
    ));

    // submit() polls first. The valid prior completion remains independently consumable even
    // though validation of the new foreign SRV reports a scheduling-side failure.
    EXPECT_TRUE(combined.failed);
    EXPECT_TRUE(combined.fresh_sample);
    ASSERT_TRUE(combined.request.has_value());
    EXPECT_EQ(combined.completed_source_frame_id, 201u);
    EXPECT_TRUE(models::sbs_roi_shape_has_flag(
      *combined.request,
      models::sbs_roi_shape_request_flag::active_roi
    ));
  }

  TEST_F(
    SbsRoiShapeRequestGpuHelperTest,
    MalformedViewCannotPromoteItsContentsAndHelperRecovers
  ) {
    const auto malformed = make_malformed_texture_view();
    ASSERT_TRUE(malformed);
    const auto rejected =
      helper_->submit(submission(malformed.Get()));
    EXPECT_TRUE(rejected.failed);
    EXPECT_TRUE(rejected.copy_scheduled);
    EXPECT_FALSE(rejected.fresh_sample);

    // The rejected SRV is never read. The helper's explicit zero buffer may safely publish only
    // a canonical fallback, never data derived from the malformed resource.
    const auto safe_fallback = await_fresh_after(*helper_, 0u);
    ASSERT_TRUE(safe_fallback.request.has_value());
    EXPECT_FALSE(models::sbs_roi_shape_has_flag(
      *safe_fallback.request,
      models::sbs_roi_shape_request_flag::active_roi
    ));
    EXPECT_EQ(safe_fallback.request->identity[0], 0u);
    EXPECT_EQ(safe_fallback.request->shape[2], 770u);
    EXPECT_EQ(safe_fallback.request->shape[3], 434u);

    const auto valid_rule = make_rule_state();
    const auto valid_view =
      make_rule_view(device_.Get(), valid_rule);
    ASSERT_TRUE(valid_view);
    const auto recovery =
      helper_->submit(submission(valid_view.Get()));
    EXPECT_TRUE(recovery.copy_scheduled);
    EXPECT_FALSE(recovery.failed);
    const auto recovered = await_fresh_after(
      *helper_,
      safe_fallback.completed_sequence
    );
    ASSERT_TRUE(recovered.request.has_value());
    EXPECT_GT(
      recovered.completed_sequence,
      safe_fallback.completed_sequence
    );
    EXPECT_TRUE(models::sbs_roi_shape_has_flag(
      *recovered.request,
      models::sbs_roi_shape_request_flag::active_roi
    ));
    EXPECT_EQ(recovered.request->identity[0], 7u);
  }

  TEST_F(
    SbsRoiShapeRequestGpuHelperTest,
    ForeignDeviceViewIsRejectedWithoutPoisoningLaterRequests
  ) {
    ComPtr<ID3D11Device> foreign_device;
    ComPtr<ID3D11DeviceContext> foreign_context;
    D3D_FEATURE_LEVEL foreign_feature_level {};
    constexpr D3D_FEATURE_LEVEL levels[] {
      D3D_FEATURE_LEVEL_11_0,
    };
    ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(
      nullptr,
      D3D_DRIVER_TYPE_WARP,
      nullptr,
      D3D11_CREATE_DEVICE_SINGLETHREADED,
      levels,
      static_cast<UINT>(std::size(levels)),
      D3D11_SDK_VERSION,
      &foreign_device,
      &foreign_feature_level,
      &foreign_context
    )));
    const auto foreign_words =
      make_rule_state(77u, 88u, 99u);
    const auto foreign_view =
      make_rule_view(foreign_device.Get(), foreign_words);
    ASSERT_TRUE(foreign_view);

    const auto rejected =
      helper_->submit(submission(foreign_view.Get()));
    EXPECT_TRUE(rejected.failed);
    EXPECT_TRUE(rejected.copy_scheduled);
    const auto fallback = await_fresh_after(*helper_, 0u);
    ASSERT_TRUE(fallback.request.has_value());
    EXPECT_NE(fallback.request->identity[0], 77u);
    EXPECT_NE(fallback.request->identity[1], 88u);
    EXPECT_FALSE(models::sbs_roi_shape_has_flag(
      *fallback.request,
      models::sbs_roi_shape_request_flag::active_roi
    ));

    const auto valid_words = make_rule_state();
    const auto valid_view =
      make_rule_view(device_.Get(), valid_words);
    ASSERT_TRUE(valid_view);
    const auto recovery =
      helper_->submit(submission(valid_view.Get()));
    EXPECT_TRUE(recovery.copy_scheduled);
    EXPECT_FALSE(recovery.failed);
    const auto recovered = await_fresh_after(
      *helper_,
      fallback.completed_sequence
    );
    ASSERT_TRUE(recovered.request.has_value());
    EXPECT_TRUE(models::sbs_roi_shape_has_flag(
      *recovered.request,
      models::sbs_roi_shape_request_flag::active_roi
    ));
    EXPECT_EQ(recovered.request->identity[0], 7u);
  }

  TEST_F(
    SbsRoiShapeRequestGpuHelperTest,
    ZeroBasedSourceFrameIdentityPublishesCanonicalFallback
  ) {
    const auto submitted = helper_->submit(
      submission(nullptr, 3840u, 2160u, 7u, 0u)
    );
    EXPECT_FALSE(submitted.failed);
    EXPECT_TRUE(submitted.copy_scheduled);
    EXPECT_FALSE(submitted.fresh_sample);
    EXPECT_FALSE(submitted.request.has_value());

    const auto completed = await_fresh_after(*helper_, 0u);
    ASSERT_TRUE(completed.request.has_value());
    EXPECT_FALSE(completed.failed);
    EXPECT_EQ(completed.completed_source_frame_id, 0u);
    EXPECT_TRUE(models::sbs_roi_shape_has_flag(
      *completed.request,
      models::sbs_roi_shape_request_flag::full_frame
    ));
    EXPECT_TRUE(models::sbs_roi_shape_has_flag(
      *completed.request,
      models::sbs_roi_shape_request_flag::fallback
    ));
  }
}  // namespace

#endif
