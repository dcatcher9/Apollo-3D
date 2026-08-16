#include <gtest/gtest.h>

#include <src/video_depth_estimator.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

TEST(HostSbsNearIdenticalPolicyTest, RawDecisionLayoutMatchesCudaBridge) {
  EXPECT_EQ(models::near_identical_gpu_decision_byte_count, 176u);
  EXPECT_EQ(models::near_identical_gpu_decision_record_byte_offset, 0u);
  EXPECT_EQ(models::near_identical_gpu_request_record_byte_offset, 32u);
  EXPECT_EQ(models::near_identical_gpu_decision_record_byte_offset % 16u, 0u);
  EXPECT_EQ(models::near_identical_gpu_request_record_byte_offset % 16u, 0u);
  EXPECT_EQ(models::near_identical_gpu_infer_reduce_byte_offset, 64u);
  EXPECT_EQ(models::near_identical_gpu_infer_one_byte_offset, 80u);
  EXPECT_EQ(models::near_identical_gpu_infer_grid16_byte_offset, 96u);
  EXPECT_EQ(models::near_identical_gpu_infer_grid8_byte_offset, 112u);
  EXPECT_EQ(models::near_identical_gpu_infer_columns_byte_offset, 128u);
  EXPECT_EQ(models::near_identical_gpu_infer_rows_byte_offset, 144u);
  EXPECT_EQ(models::near_identical_gpu_reuse_grid16_byte_offset, 160u);
  EXPECT_EQ(models::near_identical_proposal_magic, 0x504F5250u);
  EXPECT_EQ(models::near_identical_request_magic, 0x54535152u);
  EXPECT_EQ(models::near_identical_receipt_magic, 0x47524243u);
}

TEST(HostSbsNearIdenticalPolicyTest, SourceWiresGpuConditionalBranchWithoutReadback) {
  const auto read = [](const std::string &path) {
    std::ifstream stream(path, std::ios::binary);
    return std::string {
      std::istreambuf_iterator<char> {stream},
      std::istreambuf_iterator<char> {}
    };
  };
  const auto estimator = read(
    std::string {SUNSHINE_SOURCE_DIR} + "/src/video_depth_estimator.cpp"
  );
  const auto shader = read(
    std::string {SUNSHINE_SHADERS_DIR} +
    "/host_sbs_near_identical_detector_cs.hlsl"
  );
  ASSERT_FALSE(estimator.empty());
  ASSERT_FALSE(shader.empty());
  EXPECT_EQ(
    estimator.find("gpu_conditional_bridge_runtime_enabled"),
    std::string::npos
  );
  EXPECT_NE(estimator.find("ensure_depth_conditional_graph("), std::string::npos);
  EXPECT_NE(
    estimator.find("cuda.cuGraphLaunch(\n                depth_conditional_graph.get()"),
    std::string::npos
  );
  EXPECT_EQ(estimator.find("gpu_decision_resource_requested"), std::string::npos);
  EXPECT_NE(
    estimator.find(
      "std::array<CUgraphicsResource, 3> depth_resources {\n"
      "        cuda_in_res,\n"
      "        cuda_out_res,\n"
      "        cuda_near_identical_decision_res,"
    ),
    std::string::npos
  );
  EXPECT_NE(estimator.find("depth_resource_count"), std::string::npos);
  EXPECT_NE(estimator.find("near_identical_gpu_dispatch_buf"), std::string::npos);
  EXPECT_NE(
    estimator.find(
      "context->CopyResource(\n        near_identical_gpu_dispatch_buf.Get(),\n        "
      "near_identical_gpu_decision_buf.Get()"
    ),
    std::string::npos
  );
  EXPECT_NE(
    estimator.find("desc.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS"),
    std::string::npos
  );
  EXPECT_NE(
    estimator.find("accepted_optional_work == depth_optional_work_mode_e::ordinary"),
    std::string::npos
  );
  EXPECT_NE(estimator.find("!snapshot_raw_model_depth"), std::string::npos);
  EXPECT_NE(estimator.find("dispatch_infer_postprocess("), std::string::npos);
  EXPECT_NE(estimator.find("dispatch_near_identical_reuse_depth();"), std::string::npos);
  EXPECT_NE(estimator.find("publish_force_infer_transaction("), std::string::npos);
  EXPECT_NE(estimator.find("prepare_depth_inference_graph("), std::string::npos);
  EXPECT_EQ(estimator.find("bool enqueue_inference("), std::string::npos);
  const auto submit_begin = estimator.find("if (bindings_ok && !terminal_failure)");
  const auto ocr_submit = estimator.find("if (enqueued && ocr_bindings_ok)", submit_begin);
  ASSERT_NE(submit_begin, std::string::npos);
  ASSERT_NE(ocr_submit, std::string::npos);
  const auto depth_submit = estimator.substr(submit_begin, ocr_submit - submit_begin);
  EXPECT_NE(depth_submit.find("prepare_depth_inference_graph(cuda)"), std::string::npos);
  EXPECT_NE(depth_submit.find("ensure_depth_conditional_graph("), std::string::npos);
  EXPECT_NE(depth_submit.find("depth_conditional_graph.get()"), std::string::npos);
  EXPECT_EQ(depth_submit.find("enqueueV3("), std::string::npos);
  EXPECT_EQ(depth_submit.find("depth_inference_graph.executable"), std::string::npos);
  EXPECT_EQ(
    estimator.find("depth_conditional_graph.empty()) {\n                  enqueued ="),
    std::string::npos
  );
  EXPECT_NE(
    depth_submit.find(
      "\"conditional graph launch failed\",\n                  launched,\n"
      "                  true,\n                  true"
    ),
    std::string::npos
  );
  const auto prepare_begin = estimator.find("bool prepare_depth_inference_graph(");
  const auto prepare_end = estimator.find("bool enqueue_ocr_inference(", prepare_begin);
  ASSERT_NE(prepare_begin, std::string::npos);
  ASSERT_NE(prepare_end, std::string::npos);
  const auto prepare_body = estimator.substr(prepare_begin, prepare_end - prepare_begin);
  EXPECT_NE(prepare_body.find("private bootstrap only"), std::string::npos);
  EXPECT_NE(prepare_body.find("exec_context->enqueueV3(cu_stream)"), std::string::npos);
  EXPECT_EQ(prepare_body.find("cuGraphInstantiateWithFlags"), std::string::npos);

  // A CPU-known force-infer still launches the wrapper, but its postprocess is unconditional: an
  // absent/malformed CBRG must not let the CPU advance known force-infer lineage without new GPU state.
  // Only a GPU-undecided completion consumes receipt-authenticated indirect dispatch arguments.
  const auto dispatch_begin = estimator.find("void dispatch_infer_postprocess(");
  const auto dispatch_end = estimator.find(
    "bool dispatch_near_identical_postprocess_args()",
    dispatch_begin
  );
  ASSERT_NE(dispatch_begin, std::string::npos);
  ASSERT_NE(dispatch_end, std::string::npos);
  const auto dispatch_body = estimator.substr(dispatch_begin, dispatch_end - dispatch_begin);
  EXPECT_NE(
    dispatch_body.find("if (!gpu_undecided_postprocess_pending())"),
    std::string::npos
  );
  EXPECT_NE(dispatch_body.find("context->Dispatch(direct_x, direct_y, direct_z)"), std::string::npos);
  EXPECT_NE(dispatch_body.find("context->DispatchIndirect("), std::string::npos);
  EXPECT_NE(estimator.find("postprocess_temporal_reset"), std::string::npos);
  EXPECT_NE(estimator.find("completed_postprocess_temporal_reset"), std::string::npos);
  EXPECT_EQ(estimator.find("completed_input_domain_reset = true"), std::string::npos);
  EXPECT_EQ(shader.find("exact_changed"), std::string::npos);
  EXPECT_NE(
    shader.find("abs(current_nchw - previous_nchw)"),
    std::string::npos
  );
  EXPECT_NE(shader.find("* 40u <="), std::string::npos);
  EXPECT_NE(shader.find("* 10u <="), std::string::npos);
  EXPECT_NE(shader.find("tile.admitted >= 64u"), std::string::npos);
  EXPECT_NE(
    shader.find("tile.strong_changed * 4u > tile.admitted * 3u"),
    std::string::npos
  );
  EXPECT_NE(shader.find("NearIdenticalExpectedReduceGroups()"), std::string::npos);
  EXPECT_NE(
    shader.find("near_identical_reduce_groups == NearIdenticalExpectedReduceGroups()"),
    std::string::npos
  );
  EXPECT_EQ(shader.find("NearIdenticalResolveHottestStrong"), std::string::npos);
}

#ifdef _WIN32

  #include <d3d11.h>
  #include <d3dcompiler.h>
  #include <src/generated/sbs_adaptive_state_contract.h>
  #include <algorithm>
  #include <cmath>
  #include <cstring>
  #include <filesystem>
  #include <limits>
  #include <span>
  #include <string>
  #include <vector>
  #include <wrl/client.h>

namespace {
  using Microsoft::WRL::ComPtr;

  constexpr UINT field_width = 112u;
  constexpr UINT field_height = 96u;
  constexpr UINT field_texels = field_width * field_height;
  constexpr UINT tile_group_width = (field_width + 15u) / 16u;
  constexpr UINT tile_group_height = (field_height + 15u) / 16u;
  constexpr UINT tile_group_count = tile_group_width * tile_group_height;
  constexpr UINT reduction_group_count = 42u;
  constexpr models::depth_tensor_content_rect_t content {0u, 0u, 97u, 81u};
  constexpr std::uint64_t baseline_frame_id = 0x12345678abcdef01ull;
  constexpr std::uint64_t current_frame_id = 0x12345679abcdef02ull;
  constexpr std::uint64_t domain_tag = 0xfedcba9876543210ull;
  constexpr std::uint64_t request_token = 0x13579bdf2468ace0ull;

  using decision_words_t =
    std::array<std::uint32_t, models::near_identical_gpu_decision_word_count>;
  using tile_evidence_t = std::array<std::uint32_t, 4>;
  using tile_records_t = std::array<tile_evidence_t, tile_group_count>;

  constexpr std::size_t word(const models::near_identical_gpu_decision_word_e value) {
    return models::near_identical_gpu_decision_word_index(value);
  }

  std::array<float, sbs_adaptive_state::word_count> history_state(
    const std::uint32_t model_input_history_state
  ) {
    auto state = sbs_adaptive_state::initial_values;
    state[sbs_adaptive_state::index(
      sbs_adaptive_state::word_e::model_input_history_state
    )] = static_cast<float>(model_input_history_state);
    return state;
  }

  class near_identical_gpu_fixture_t {
  public:
    enum class initialize_result_e {ready, d3d_unavailable, failed};

    initialize_result_e initialize(std::string &error) {
      constexpr D3D_FEATURE_LEVEL requested[] = {D3D_FEATURE_LEVEL_11_0};
      D3D_FEATURE_LEVEL actual {};
      const auto status = D3D11CreateDevice(
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
      if (FAILED(status) || actual < D3D_FEATURE_LEVEL_11_0) {
        error = "D3D11 WARP feature level 11 is unavailable";
        return initialize_result_e::d3d_unavailable;
      }
      const std::filesystem::path shader_path =
        SUNSHINE_SHADERS_DIR "/host_sbs_near_identical_detector_cs.hlsl";
      if (!compile(shader_path, "compare_main", compare_shader_, error) ||
          !compile(shader_path, "resolve_main", resolve_shader_, error) ||
          !compile(shader_path, "history_owner_main", history_owner_shader_, error) ||
          !compile(shader_path, "postprocess_args_main", args_shader_, error)) {
        return initialize_result_e::failed;
      }

      D3D11_BUFFER_DESC constants_desc {};
      constants_desc.ByteWidth = 16u * sizeof(std::uint32_t);
      constants_desc.Usage = D3D11_USAGE_DEFAULT;
      constants_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      if (FAILED(device_->CreateBuffer(&constants_desc, nullptr, &depth_constants_)) ||
          FAILED(device_->CreateBuffer(&constants_desc, nullptr, &detector_constants_))) {
        error = "could not create detector constant buffers";
        return initialize_result_e::failed;
      }
      if (!create_structured_output(
            tile_group_count,
            4u * sizeof(std::uint32_t),
            tile_buffer_,
            tile_srv_,
            tile_uav_
          ) ||
          !create_structured_output(
            static_cast<UINT>(models::near_identical_history_owner_word_count),
            sizeof(std::uint32_t),
            history_owner_buffer_,
            history_owner_srv_,
            history_owner_uav_
          ) ||
          !create_structured_output(
            10u,
            sizeof(std::uint32_t),
            scene_evidence_buffer_,
            scene_evidence_srv_,
            scene_evidence_uav_
          ) ||
          !create_raw_decision_buffer()) {
        error = "could not create near-identical GPU buffers";
        return initialize_result_e::failed;
      }
      D3D11_BUFFER_DESC staging_desc {};
      decision_buffer_->GetDesc(&staging_desc);
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0u;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      staging_desc.MiscFlags = 0u;
      if (FAILED(device_->CreateBuffer(&staging_desc, nullptr, &decision_staging_))) {
        error = "could not create decision staging buffer";
        return initialize_result_e::failed;
      }
      scene_evidence_buffer_->GetDesc(&staging_desc);
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0u;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      staging_desc.MiscFlags = 0u;
      staging_desc.StructureByteStride = 0u;
      if (FAILED(device_->CreateBuffer(&staging_desc, nullptr, &scene_evidence_staging_))) {
        error = "could not create scene-evidence staging buffer";
        return initialize_result_e::failed;
      }
      return initialize_result_e::ready;
    }

    bool run_detector(
      const std::vector<float> &current,
      const std::vector<float> &previous,
      const std::array<float, sbs_adaptive_state::word_count> &cut_state,
      decision_words_t &decision,
      std::string &error,
      const std::uint32_t reduce_groups = reduction_group_count
    ) {
      if (current.size() != 3u * field_texels || previous.size() != current.size()) {
        error = "invalid model input size";
        return false;
      }
      ComPtr<ID3D11Buffer> current_buffer;
      ComPtr<ID3D11ShaderResourceView> current_srv;
      ComPtr<ID3D11Buffer> previous_buffer;
      ComPtr<ID3D11ShaderResourceView> previous_srv;
      ComPtr<ID3D11Buffer> state_buffer;
      ComPtr<ID3D11ShaderResourceView> state_srv;
      if (!create_structured_srv(
            std::span<const float> {current}, sizeof(float), current_buffer, current_srv
          ) ||
          !create_structured_srv(
            std::span<const float> {previous}, sizeof(float), previous_buffer, previous_srv
          ) ||
          !create_structured_srv(
            std::span<const float> {cut_state},
            4u * sizeof(float),
            state_buffer,
            state_srv
          )) {
        error = "could not create detector inputs";
        return false;
      }
      update_depth_constants();
      initialize_transaction();
      update_detector_constants(0u, 0u, 0u, baseline_frame_id, 0u, 0u);
      ID3D11Buffer *constants[2] = {depth_constants_.Get(), detector_constants_.Get()};
      ID3D11ShaderResourceView *owner_inputs[5] = {
        nullptr, nullptr, nullptr, nullptr, state_srv.Get(),
      };
      ID3D11UnorderedAccessView *owner_outputs[3] = {
        nullptr, nullptr, history_owner_uav_.Get(),
      };
      context_->CSSetShader(history_owner_shader_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(0u, 2u, constants);
      context_->CSSetShaderResources(0u, 5u, owner_inputs);
      context_->CSSetUnorderedAccessViews(0u, 3u, owner_outputs, nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind(5u, 3u, 2u);

      update_detector_constants(
        1u,
        tile_group_width,
        tile_group_height,
        current_frame_id,
        baseline_frame_id,
        request_token,
        0u,
        reduce_groups
      );
      ID3D11ShaderResourceView *compare_inputs[2] = {
        current_srv.Get(), previous_srv.Get(),
      };
      context_->CSSetShader(compare_shader_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(0u, 2u, constants);
      context_->CSSetShaderResources(0u, 2u, compare_inputs);
      context_->CSSetUnorderedAccessViews(0u, 1u, tile_uav_.GetAddressOf(), nullptr);
      context_->Dispatch(tile_group_width, tile_group_height, 1u);
      unbind(2u, 1u, 0u);

      ID3D11ShaderResourceView *resolve_inputs[4] = {
        nullptr, nullptr, tile_srv_.Get(), history_owner_srv_.Get(),
      };
      ID3D11UnorderedAccessView *resolve_outputs[4] = {
        nullptr, nullptr, nullptr, decision_uav_.Get(),
      };
      context_->CSSetShader(resolve_shader_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(0u, 2u, constants);
      context_->CSSetShaderResources(0u, 4u, resolve_inputs);
      context_->CSSetUnorderedAccessViews(0u, 4u, resolve_outputs, nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind(4u, 4u, 2u);
      return read_decision(decision, error);
    }

    bool resolve_seeded_tiles(
      const tile_records_t &tiles,
      decision_words_t &decision,
      std::string &error,
      const std::uint32_t reduce_groups = reduction_group_count
    ) {
      context_->UpdateSubresource(
        tile_buffer_.Get(), 0u, nullptr, tiles.data(), 0u, 0u
      );
      initialize_transaction();
      update_detector_constants(
        1u,
        tile_group_width,
        tile_group_height,
        current_frame_id,
        baseline_frame_id,
        request_token,
        0u,
        reduce_groups
      );
      ID3D11Buffer *constants[2] = {depth_constants_.Get(), detector_constants_.Get()};
      ID3D11ShaderResourceView *resolve_inputs[4] = {
        nullptr, nullptr, tile_srv_.Get(), history_owner_srv_.Get(),
      };
      ID3D11UnorderedAccessView *resolve_outputs[4] = {
        nullptr, nullptr, nullptr, decision_uav_.Get(),
      };
      context_->CSSetShader(resolve_shader_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(0u, 2u, constants);
      context_->CSSetShaderResources(0u, 4u, resolve_inputs);
      context_->CSSetUnorderedAccessViews(0u, 4u, resolve_outputs, nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind(4u, 4u, 2u);
      return read_decision(decision, error);
    }

    void seed_scene_evidence(const std::array<std::uint32_t, 10> &words) {
      context_->UpdateSubresource(
        scene_evidence_buffer_.Get(), 0u, nullptr, words.data(), 0u, 0u
      );
    }

    bool read_scene_evidence(
      std::array<std::uint32_t, 10> &words,
      std::string &error
    ) {
      context_->CopyResource(scene_evidence_staging_.Get(), scene_evidence_buffer_.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context_->Map(
            scene_evidence_staging_.Get(), 0u, D3D11_MAP_READ, 0u, &mapped
          )) || !mapped.pData) {
        error = "could not read scene-evidence buffer";
        return false;
      }
      std::memcpy(words.data(), mapped.pData, sizeof(words));
      context_->Unmap(scene_evidence_staging_.Get(), 0u);
      return true;
    }

    bool resolve_postprocess_args(
      const models::near_identical_gpu_branch_e branch,
      const bool valid_receipt,
      decision_words_t &decision,
      std::string &error,
      const std::uint64_t constants_token = request_token,
      const std::uint32_t reduce_groups = reduction_group_count
    ) {
      if (!read_decision(decision, error)) return false;
      const auto resolved = static_cast<std::uint32_t>(branch);
      decision[word(models::near_identical_gpu_decision_word_e::decision)] = resolved;
      decision[word(models::near_identical_gpu_decision_word_e::decision_cookie)] =
        resolved ^ models::near_identical_decision_cookie;
      const auto low = static_cast<std::uint32_t>(request_token);
      const auto high = static_cast<std::uint32_t>(request_token >> 32u);
      decision[word(models::near_identical_gpu_decision_word_e::decision_token_low)] = low;
      decision[word(models::near_identical_gpu_decision_word_e::decision_token_high)] = high;
      decision[word(models::near_identical_gpu_decision_word_e::decision_token_low_cookie)] =
        low ^ models::near_identical_token_low_cookie;
      decision[word(models::near_identical_gpu_decision_word_e::decision_token_high_cookie)] =
        high ^ models::near_identical_token_high_cookie;
      decision[word(models::near_identical_gpu_decision_word_e::decision_magic)] =
        valid_receipt ? models::near_identical_receipt_magic :
                        models::near_identical_proposal_magic;
      context_->UpdateSubresource(
        decision_buffer_.Get(), 0u, nullptr, decision.data(), 0u, 0u
      );
      update_detector_constants(
        1u,
        tile_group_width,
        tile_group_height,
        current_frame_id,
        baseline_frame_id,
        constants_token,
        1u,
        reduce_groups
      );
      ID3D11Buffer *constants[2] = {depth_constants_.Get(), detector_constants_.Get()};
      ID3D11UnorderedAccessView *outputs[6] = {
        nullptr, nullptr, nullptr, decision_uav_.Get(), nullptr, scene_evidence_uav_.Get(),
      };
      context_->CSSetShader(args_shader_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(0u, 2u, constants);
      context_->CSSetUnorderedAccessViews(0u, 6u, outputs, nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind(0u, 6u, 2u);
      return read_decision(decision, error);
    }

  private:
    bool compile(
      const std::filesystem::path &path,
      const char *entrypoint,
      ComPtr<ID3D11ComputeShader> &shader,
      std::string &error
    ) {
      ComPtr<ID3DBlob> bytecode;
      ComPtr<ID3DBlob> diagnostics;
      const auto status = D3DCompileFromFile(
        path.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entrypoint,
        "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0u,
        &bytecode,
        &diagnostics
      );
      if (FAILED(status) || !bytecode) {
        error = diagnostics ?
          static_cast<const char *>(diagnostics->GetBufferPointer()) :
          "shader compilation failed without diagnostics";
        return false;
      }
      return SUCCEEDED(device_->CreateComputeShader(
        bytecode->GetBufferPointer(), bytecode->GetBufferSize(), nullptr, &shader
      ));
    }

    bool create_structured_output(
      const UINT elements,
      const UINT stride,
      ComPtr<ID3D11Buffer> &buffer,
      ComPtr<ID3D11ShaderResourceView> &srv,
      ComPtr<ID3D11UnorderedAccessView> &uav
    ) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = elements * stride;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = stride;
      return SUCCEEDED(device_->CreateBuffer(&desc, nullptr, &buffer)) &&
             SUCCEEDED(device_->CreateShaderResourceView(buffer.Get(), nullptr, &srv)) &&
             SUCCEEDED(device_->CreateUnorderedAccessView(buffer.Get(), nullptr, &uav));
    }

    bool create_raw_decision_buffer() {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = static_cast<UINT>(models::near_identical_gpu_decision_byte_count);
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS |
                       D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      if (FAILED(device_->CreateBuffer(&desc, nullptr, &decision_buffer_))) return false;
      D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc {};
      srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
      srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
      srv_desc.BufferEx.NumElements =
        static_cast<UINT>(models::near_identical_gpu_decision_word_count);
      srv_desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
      D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc {};
      uav_desc.Format = DXGI_FORMAT_R32_TYPELESS;
      uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      uav_desc.Buffer.NumElements =
        static_cast<UINT>(models::near_identical_gpu_decision_word_count);
      uav_desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
      return SUCCEEDED(device_->CreateShaderResourceView(
               decision_buffer_.Get(), &srv_desc, &decision_srv_
             )) &&
             SUCCEEDED(device_->CreateUnorderedAccessView(
               decision_buffer_.Get(), &uav_desc, &decision_uav_
             ));
    }

    bool create_structured_srv(
      const std::span<const float> values,
      const UINT stride,
      ComPtr<ID3D11Buffer> &buffer,
      ComPtr<ID3D11ShaderResourceView> &srv
    ) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = static_cast<UINT>(values.size_bytes());
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = stride;
      D3D11_SUBRESOURCE_DATA initial {values.data(), 0u, 0u};
      return SUCCEEDED(device_->CreateBuffer(&desc, &initial, &buffer)) &&
             SUCCEEDED(device_->CreateShaderResourceView(buffer.Get(), nullptr, &srv));
    }

    void update_depth_constants() {
      std::array<std::uint32_t, 16> constants {};
      constants[0] = field_width;
      constants[1] = field_height;
      constants[9] = content.left;
      constants[10] = content.top;
      constants[11] = content.right;
      constants[12] = content.bottom;
      context_->UpdateSubresource(
        depth_constants_.Get(), 0u, nullptr, constants.data(), 0u, 0u
      );
    }

    void update_detector_constants(
      const std::uint32_t flags,
      const std::uint32_t groups_x,
      const std::uint32_t groups_y,
      const std::uint64_t current,
      const std::uint64_t baseline,
      const std::uint64_t token,
      const std::uint32_t stream_frame_delta = 0u,
      const std::uint32_t reduce_groups = reduction_group_count
    ) {
      const std::array<std::uint32_t, 16> constants {
        flags,
        groups_x,
        groups_y,
        groups_x * groups_y,
        static_cast<std::uint32_t>(current),
        static_cast<std::uint32_t>(current >> 32u),
        static_cast<std::uint32_t>(baseline),
        static_cast<std::uint32_t>(baseline >> 32u),
        static_cast<std::uint32_t>(domain_tag),
        static_cast<std::uint32_t>(domain_tag >> 32u),
        static_cast<std::uint32_t>(token),
        static_cast<std::uint32_t>(token >> 32u),
        reduce_groups,
        stream_frame_delta,
        0u,
        0u,
      };
      context_->UpdateSubresource(
        detector_constants_.Get(), 0u, nullptr, constants.data(), 0u, 0u
      );
    }

    void initialize_transaction() {
      decision_words_t initial {};
      initial[word(models::near_identical_gpu_decision_word_e::decision)] =
        static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer);
      const auto low = static_cast<std::uint32_t>(request_token);
      const auto high = static_cast<std::uint32_t>(request_token >> 32u);
      initial[word(models::near_identical_gpu_decision_word_e::request_token_low)] = low;
      initial[word(models::near_identical_gpu_decision_word_e::request_token_high)] = high;
      initial[word(models::near_identical_gpu_decision_word_e::request_token_low_cookie)] =
        low ^ models::near_identical_token_low_cookie;
      initial[word(models::near_identical_gpu_decision_word_e::request_token_high_cookie)] =
        high ^ models::near_identical_token_high_cookie;
      initial[word(models::near_identical_gpu_decision_word_e::request_magic)] =
        models::near_identical_request_magic;
      context_->UpdateSubresource(
        decision_buffer_.Get(), 0u, nullptr, initial.data(), 0u, 0u
      );
    }

    bool read_decision(decision_words_t &words, std::string &error) {
      context_->CopyResource(decision_staging_.Get(), decision_buffer_.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context_->Map(
            decision_staging_.Get(), 0u, D3D11_MAP_READ, 0u, &mapped
          )) || !mapped.pData) {
        error = "could not read decision buffer";
        return false;
      }
      std::memcpy(words.data(), mapped.pData, sizeof(words));
      context_->Unmap(decision_staging_.Get(), 0u);
      return true;
    }

    void unbind(const UINT srv_count, const UINT uav_count, const UINT cbuffer_count) {
      std::array<ID3D11ShaderResourceView *, 6> null_srvs {};
      std::array<ID3D11UnorderedAccessView *, 6> null_uavs {};
      std::array<ID3D11Buffer *, 2> null_constants {};
      if (srv_count != 0u) {
        context_->CSSetShaderResources(0u, srv_count, null_srvs.data());
      }
      if (uav_count != 0u) {
        context_->CSSetUnorderedAccessViews(0u, uav_count, null_uavs.data(), nullptr);
      }
      if (cbuffer_count != 0u) {
        context_->CSSetConstantBuffers(0u, cbuffer_count, null_constants.data());
      }
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11ComputeShader> compare_shader_;
    ComPtr<ID3D11ComputeShader> resolve_shader_;
    ComPtr<ID3D11ComputeShader> history_owner_shader_;
    ComPtr<ID3D11ComputeShader> args_shader_;
    ComPtr<ID3D11Buffer> depth_constants_;
    ComPtr<ID3D11Buffer> detector_constants_;
    ComPtr<ID3D11Buffer> tile_buffer_;
    ComPtr<ID3D11ShaderResourceView> tile_srv_;
    ComPtr<ID3D11UnorderedAccessView> tile_uav_;
    ComPtr<ID3D11Buffer> history_owner_buffer_;
    ComPtr<ID3D11ShaderResourceView> history_owner_srv_;
    ComPtr<ID3D11UnorderedAccessView> history_owner_uav_;
    ComPtr<ID3D11Buffer> scene_evidence_buffer_;
    ComPtr<ID3D11ShaderResourceView> scene_evidence_srv_;
    ComPtr<ID3D11UnorderedAccessView> scene_evidence_uav_;
    ComPtr<ID3D11Buffer> scene_evidence_staging_;
    ComPtr<ID3D11Buffer> decision_buffer_;
    ComPtr<ID3D11ShaderResourceView> decision_srv_;
    ComPtr<ID3D11UnorderedAccessView> decision_uav_;
    ComPtr<ID3D11Buffer> decision_staging_;
  };

  std::vector<float> uniform_model_input(const float pixel = 0.5f) {
    constexpr std::array mean {0.485f, 0.456f, 0.406f};
    constexpr std::array stddev {0.229f, 0.224f, 0.225f};
    std::vector<float> input(3u * field_texels);
    for (UINT channel = 0u; channel < 3u; ++channel) {
      const float normalized = (pixel - mean[channel]) / stddev[channel];
      std::fill_n(
        input.begin() + static_cast<std::size_t>(channel) * field_texels,
        field_texels,
        normalized
      );
    }
    return input;
  }

  void add_red_pixel_delta(
    std::vector<float> &input,
    const UINT x,
    const UINT y,
    const float model_rgb_delta
  ) {
    input[y * field_width + x] += model_rgb_delta / 0.229f;
  }

  void add_first_content_pixel_deltas(
    std::vector<float> &input,
    const UINT count,
    const float model_rgb_delta
  ) {
    UINT changed = 0u;
    for (UINT y = content.top; y < content.bottom && changed < count; ++y) {
      for (UINT x = content.left; x < content.right && changed < count; ++x) {
        add_red_pixel_delta(input, x, y, model_rgb_delta);
        ++changed;
      }
    }
    ASSERT_EQ(changed, count);
  }

  tile_records_t quiet_tile_records() {
    tile_records_t records {};
    for (UINT group_y = 0u; group_y < tile_group_height; ++group_y) {
      for (UINT group_x = 0u; group_x < tile_group_width; ++group_x) {
        const UINT left = std::max(content.left, group_x * 16u);
        const UINT top = std::max(content.top, group_y * 16u);
        const UINT right = std::min(content.right, (group_x + 1u) * 16u);
        const UINT bottom = std::min(content.bottom, (group_y + 1u) * 16u);
        records[group_y * tile_group_width + group_x][0] =
          right > left && bottom > top ? (right - left) * (bottom - top) : 0u;
      }
    }
    return records;
  }

  near_identical_gpu_fixture_t::initialize_result_e initialize_fixture(
    near_identical_gpu_fixture_t &fixture,
    std::string &error
  ) {
    return fixture.initialize(error);
  }
}

TEST(HostSbsNearIdenticalDetectorGpuTest, PublishesAuthenticatedQuietProposalOnly) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;
  const auto input = uniform_model_input();
  decision_words_t decision {};
  ASSERT_TRUE(fixture.run_detector(input, input, history_state(1u), decision, error)) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::reuse)
  );
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision_magic)],
    models::near_identical_proposal_magic
  );
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::request_magic)],
    models::near_identical_request_magic
  );
  for (std::size_t index = word(models::near_identical_gpu_decision_word_e::infer_reduce_x);
       index < models::near_identical_gpu_decision_word_count;
       ++index) {
    EXPECT_EQ(decision[index], 0u) << "indirect word " << index;
  }
}

TEST(HostSbsNearIdenticalDetectorGpuTest, InvalidHistoryAndNonfiniteInputInfer) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;
  const auto previous = uniform_model_input();
  decision_words_t decision {};
  ASSERT_TRUE(fixture.run_detector(
    previous, previous, history_state(3u), decision, error
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );
  auto nonfinite = previous;
  nonfinite[field_width + 1u] = std::numeric_limits<float>::quiet_NaN();
  ASSERT_TRUE(fixture.run_detector(
    nonfinite, previous, history_state(1u), decision, error
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );
}

TEST(HostSbsNearIdenticalDetectorGpuTest, ReductionGroupCountMustMatchTensorShape) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;
  const auto input = uniform_model_input();
  decision_words_t decision {};
  ASSERT_TRUE(fixture.run_detector(
    input,
    input,
    history_state(1u),
    decision,
    error,
    reduction_group_count + 1u
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );

  ASSERT_TRUE(fixture.run_detector(input, input, history_state(1u), decision, error)) << error;
  ASSERT_TRUE(fixture.resolve_postprocess_args(
    models::near_identical_gpu_branch_e::infer,
    true,
    decision,
    error,
    request_token,
    reduction_group_count + 1u
  )) << error;
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_reduce_x)], 0u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_one_x)], 0u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::reuse_grid16_x)], 7u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::reuse_grid16_y)], 6u);
}

TEST(HostSbsNearIdenticalDetectorGpuTest, MalformedUnsupportedTileInfers) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;
  const auto input = uniform_model_input();
  decision_words_t decision {};
  ASSERT_TRUE(fixture.run_detector(input, input, history_state(1u), decision, error)) << error;

  auto tiles = quiet_tile_records();
  const UINT unsupported_edge_tile = tile_group_width - 1u;
  ASSERT_EQ(tiles[unsupported_edge_tile][0], 16u);
  tiles[unsupported_edge_tile][2] = 17u;
  ASSERT_TRUE(fixture.resolve_seeded_tiles(tiles, decision, error)) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );
}

TEST(HostSbsNearIdenticalDetectorGpuTest, SupportedTileThresholdIsInclusive) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;

  const auto input = uniform_model_input();
  decision_words_t decision {};
  ASSERT_TRUE(fixture.run_detector(input, input, history_state(1u), decision, error)) << error;

  auto tiles = quiet_tile_records();
  constexpr UINT supported_tile = 0u;
  tiles[supported_tile] = {256u, 192u, 192u, 0u};
  ASSERT_TRUE(fixture.resolve_seeded_tiles(tiles, decision, error)) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::reuse)
  );

  tiles[supported_tile] = {256u, 193u, 193u, 0u};
  ASSERT_TRUE(fixture.resolve_seeded_tiles(tiles, decision, error)) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );
}

TEST(HostSbsNearIdenticalDetectorGpuTest, GlobalThresholdsAreInclusiveAndVetoAboveBoundary) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;
  const auto previous = uniform_model_input();
  decision_words_t decision {};

  auto current = previous;
  add_first_content_pixel_deltas(current, 785u, 0.02f);
  ASSERT_TRUE(fixture.run_detector(
    current, previous, history_state(1u), decision, error
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::reuse)
  );

  current = previous;
  add_first_content_pixel_deltas(current, 786u, 0.02f);
  ASSERT_TRUE(fixture.run_detector(
    current, previous, history_state(1u), decision, error
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );

  current = previous;
  add_first_content_pixel_deltas(current, 196u, 0.21f);
  ASSERT_TRUE(fixture.run_detector(
    current, previous, history_state(1u), decision, error
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::reuse)
  );

  current = previous;
  add_first_content_pixel_deltas(current, 197u, 0.21f);
  ASSERT_TRUE(fixture.run_detector(
    current, previous, history_state(1u), decision, error
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );
}

TEST(HostSbsNearIdenticalDetectorGpuTest, UnsupportedEdgeTileCannotMaskSupportedHotTile) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;
  const auto previous = uniform_model_input();
  auto current = previous;
  for (UINT index = 0u; index < 193u; ++index) {
    add_red_pixel_delta(current, index % 16u, index / 16u, 0.21f);
  }
  // The final content tile contains exactly one admitted texel. Its 1/1 ratio outranks 193/256,
  // but it has no local-veto authority and must not hide the supported tile.
  add_red_pixel_delta(current, 96u, 80u, 0.21f);
  decision_words_t decision {};
  ASSERT_TRUE(fixture.run_detector(
    current, previous, history_state(1u), decision, error
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );
}

TEST(HostSbsNearIdenticalDetectorGpuTest, ReceiptWritesExactIndirectShapes) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;
  const auto input = uniform_model_input();
  decision_words_t decision {};
  ASSERT_TRUE(fixture.run_detector(input, input, history_state(1u), decision, error)) << error;
  ASSERT_TRUE(fixture.resolve_postprocess_args(
    models::near_identical_gpu_branch_e::infer, true, decision, error
  )) << error;
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_reduce_x)], 42u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_one_x)], 1u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_grid16_x)], 7u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_grid16_y)], 6u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_grid8_x)], 14u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_grid8_y)], 12u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_columns_x)], 112u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_rows_x)], 96u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::reuse_grid16_x)], 0u);
  std::array<std::uint32_t, 10> scene_evidence {};
  ASSERT_TRUE(fixture.read_scene_evidence(scene_evidence, error)) << error;
  for (std::size_t index = 0u; index < 9u; ++index) {
    EXPECT_EQ(scene_evidence[index], 0u) << "scene-evidence word " << index;
  }
  EXPECT_EQ(scene_evidence[9], 1u);

  scene_evidence.fill(0xA5A5A5A5u);
  fixture.seed_scene_evidence(scene_evidence);
  ASSERT_TRUE(fixture.resolve_postprocess_args(
    models::near_identical_gpu_branch_e::reuse, true, decision, error
  )) << error;
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_one_x)], 0u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::reuse_grid16_x)], 7u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::reuse_grid16_y)], 6u);
  std::array<std::uint32_t, 10> preserved_scene_evidence {};
  ASSERT_TRUE(fixture.read_scene_evidence(preserved_scene_evidence, error)) << error;
  EXPECT_EQ(preserved_scene_evidence, scene_evidence);

  ASSERT_TRUE(fixture.run_detector(input, input, history_state(1u), decision, error)) << error;
  ASSERT_TRUE(fixture.resolve_postprocess_args(
    models::near_identical_gpu_branch_e::infer, false, decision, error
  )) << error;
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_one_x)], 0u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::reuse_grid16_x)], 7u);

  ASSERT_TRUE(fixture.run_detector(input, input, history_state(1u), decision, error)) << error;
  ASSERT_TRUE(fixture.resolve_postprocess_args(
    models::near_identical_gpu_branch_e::infer,
    true,
    decision,
    error,
    request_token + 1u
  )) << error;
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_one_x)], 0u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::reuse_grid16_x)], 7u);
  ASSERT_TRUE(fixture.read_scene_evidence(preserved_scene_evidence, error)) << error;
  EXPECT_EQ(preserved_scene_evidence, scene_evidence);
}

#else

TEST(HostSbsNearIdenticalDetectorGpuTest, WindowsOnly) {
  GTEST_SKIP() << "D3D11 WARP is Windows-only";
}

#endif
