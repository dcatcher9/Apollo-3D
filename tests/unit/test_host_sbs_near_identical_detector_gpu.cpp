#include <gtest/gtest.h>

#include <src/cuda_conditional_graph.h>
#include <src/host_sbs_shader_cache.h>
#include <src/video_depth_estimator.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

TEST(HostSbsNearIdenticalPolicyTest, RawDecisionLayoutMatchesCudaBridge) {
  EXPECT_EQ(models::near_identical_gpu_decision_byte_count, 256u);
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
  EXPECT_EQ(models::near_identical_gpu_optional_preprocess_byte_offset, 176u);
  EXPECT_EQ(models::near_identical_gpu_subtitle_condition_grid16_byte_offset, 176u);
  EXPECT_EQ(models::near_identical_gpu_optional_cells_byte_offset, 192u);
  EXPECT_EQ(models::near_identical_gpu_optional_one_byte_offset, 208u);
  EXPECT_EQ(models::near_identical_gpu_infer_without_optional_one_byte_offset, 224u);
  EXPECT_EQ(models::near_identical_gpu_subtitle_record_one_byte_offset, 224u);
  EXPECT_EQ(models::near_identical_gpu_observation_one_byte_offset, 240u);
  EXPECT_EQ(models::near_identical_proposal_magic, 0x504F5250u);
  EXPECT_EQ(models::near_identical_request_magic, 0x54535152u);
  EXPECT_EQ(models::near_identical_receipt_magic, 0x47524243u);
  EXPECT_EQ(models::near_identical_history_owner_contract_tag, 0x3142484Eu);
  EXPECT_EQ(models::near_identical_history_owner_contract_schema, 2u);
  EXPECT_EQ(models::near_identical_history_owner_word_count, 10u);
  EXPECT_EQ(models::near_identical_work_optional_ocr, 1u);
  EXPECT_EQ(models::near_identical_work_subtitle_observation, 2u);
  EXPECT_EQ(models::near_identical_work_optional_ocr_due, 8u);
  EXPECT_EQ(models::near_identical_work_subtitle_observation_due, 16u);
}

TEST(HostSbsNearIdenticalPolicyTest, SoleFusedPreprocessIsMandatoryAndNoComparatorRemains) {
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
  const auto cache = read(
    std::string {SUNSHINE_SOURCE_DIR} + "/src/host_sbs_shader_cache.cpp"
  );
  const auto cache_header = read(
    std::string {SUNSHINE_SOURCE_DIR} + "/src/host_sbs_shader_cache.h"
  );
  const auto detector = read(
    std::string {SUNSHINE_SHADERS_DIR} +
    "/host_sbs_near_identical_detector_cs.hlsl"
  );
  const auto preprocess = read(
    std::string {SUNSHINE_SHADERS_DIR} + "/rgb_to_nchw_cs.hlsl"
  );
  const auto fused = read(
    std::string {SUNSHINE_SHADERS_DIR} +
    "/rgb_to_nchw_near_identical_cs.hlsl"
  );
  ASSERT_FALSE(estimator.empty());
  ASSERT_FALSE(cache.empty());
  ASSERT_FALSE(cache_header.empty());
  ASSERT_FALSE(detector.empty());
  ASSERT_FALSE(preprocess.empty());
  ASSERT_FALSE(fused.empty());

  const auto producer_loop = estimator.find(
    "for (const auto &binding : host_sbs_shader_cache::parallax_v2_producer_bindings)"
  );
  const auto producer_end = estimator.find(
    "parallax_v2_producer_shaders_ready = true", producer_loop
  );
  ASSERT_NE(producer_loop, std::string::npos);
  ASSERT_NE(producer_end, std::string::npos);
  const auto producer_body = estimator.substr(
    producer_loop, producer_end - producer_loop
  );
  const auto mandatory_assignment = producer_body.find(
    "producer_shaders_ok = output && create_shader"
  );
  ASSERT_NE(mandatory_assignment, std::string::npos);
  EXPECT_EQ(producer_body.find("near_identical_fused_preprocess_cs.Reset()"), std::string::npos);
  EXPECT_EQ(producer_body.find("Optional Host SBS fused preprocess"), std::string::npos);
  EXPECT_EQ(cache.find("prewarm_producer_set"), std::string::npos);
  EXPECT_NE(
    cache.find("prewarm_shader_set(shader_root, parallax_v2_producer_specs"),
    std::string::npos
  );

  const auto route_begin = estimator.find("const bool gpu_undecided_host_candidate =");
  const auto route_end = estimator.find(
    "auto accepted_submission_class =", route_begin
  );
  ASSERT_NE(route_begin, std::string::npos);
  ASSERT_NE(route_end, std::string::npos);
  const auto route = estimator.substr(route_begin, route_end - route_begin);
  EXPECT_EQ(route.find("near_identical_fused_preprocess_cs"), std::string::npos)
    << "mandatory producer availability must not silently change a root to force-infer";
  EXPECT_NE(
    estimator.find(
      "context->CSSetShader(near_identical_fused_preprocess_cs.Get(), nullptr, 0u)"
    ),
    std::string::npos
  );
  EXPECT_NE(
    estimator.find(
      "gpu_undecided_host_candidate ?\n          fused_preprocess_compare_cbuffer.Get() :\n"
      "          fused_preprocess_force_cbuffer.Get()"
    ),
    std::string::npos
  );
  EXPECT_NE(estimator.find("tensor_previous_input_srv.Get(),"), std::string::npos);
  EXPECT_NE(estimator.find("near_identical_tile_uav.Get(),"), std::string::npos);
  EXPECT_NE(
    estimator.find("dispatch_near_identical_detector()", route_end),
    std::string::npos
  );
  EXPECT_EQ(estimator.find("near_identical_compare_cs"), std::string::npos);
  EXPECT_EQ(estimator.find("rgb_to_nchw_content_cs"), std::string::npos);
  EXPECT_EQ(estimator.find("rgb_to_nchw_pad_cs"), std::string::npos);
  EXPECT_EQ(estimator.find("Microsoft::WRL::ComPtr<ID3D11ComputeShader> rgb_to_nchw_cs"), std::string::npos);
  EXPECT_EQ(cache_header.find("inline constexpr shader_spec rgb_to_nchw "), std::string::npos);
  EXPECT_EQ(cache_header.find("rgb_to_nchw_content"), std::string::npos);
  EXPECT_EQ(cache_header.find("rgb_to_nchw_pad"), std::string::npos);
  EXPECT_EQ(cache_header.find("host_sbs_near_identical_compare"), std::string::npos);
  EXPECT_EQ(detector.find("compare_main"), std::string::npos);
  EXPECT_EQ(detector.find("CurrentModelInput"), std::string::npos);
  EXPECT_EQ(preprocess.find("void content_main"), std::string::npos);
  EXPECT_EQ(preprocess.find("void pad_main"), std::string::npos);
  EXPECT_NE(fused.find("NearIdenticalEvidenceEnabled"), std::string::npos);
  EXPECT_NE(fused.find("disabled_evidence.admitted = 0u"), std::string::npos);
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
  const auto compare_contract = read(
    std::string {SUNSHINE_SHADERS_DIR} +
    "/include/host_sbs_near_identical_compare.hlsl"
  );
  ASSERT_FALSE(estimator.empty());
  ASSERT_FALSE(shader.empty());
  ASSERT_FALSE(compare_contract.empty());
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
  EXPECT_EQ(
    estimator.find("fit_near_identical_subtitle_tile_band("), std::string::npos
  );
  EXPECT_NE(estimator.find("prepare_depth_inference_graph("), std::string::npos);
  EXPECT_NE(estimator.find("prepare_ocr_inference_graph("), std::string::npos);
  EXPECT_EQ(estimator.find("bool enqueue_inference("), std::string::npos);
  EXPECT_EQ(estimator.find("bool enqueue_ocr_inference("), std::string::npos);
  EXPECT_EQ(estimator.find("ocr_execution_stream"), std::string::npos);
  EXPECT_EQ(estimator.find("ocr_cu_stream"), std::string::npos);
  EXPECT_EQ(estimator.find("ocr_inference_done_event"), std::string::npos);
  const auto submit_begin = estimator.find(
    "if (bindings_ok && !terminal_failure && !dropped_for_signature_change)"
  );
  const auto submit_end = estimator.find("CUresult ocr_unmap_res", submit_begin);
  ASSERT_NE(submit_begin, std::string::npos);
  ASSERT_NE(submit_end, std::string::npos);
  const auto unified_submit = estimator.substr(submit_begin, submit_end - submit_begin);
  EXPECT_NE(
    unified_submit.find("prepare_depth_inference_graph(\n              cuda, allow_private_bootstrap"),
    std::string::npos
  );
  EXPECT_NE(
    unified_submit.find("prepare_ocr_inference_graph(\n                cuda, allow_private_bootstrap"),
    std::string::npos
  );
  EXPECT_NE(unified_submit.find("ensure_depth_conditional_graph("), std::string::npos);
  EXPECT_NE(unified_submit.find("optional_child"), std::string::npos);
  EXPECT_NE(estimator.find("if (ocr_frame_eligible)"), std::string::npos);
  EXPECT_NE(
    unified_submit.find("conditional_optional_child_policy_e::retain_if_present"),
    std::string::npos
  );
  EXPECT_NE(
    unified_submit.find("conditional_optional_child_policy_e::build_ready"),
    std::string::npos
  );
  EXPECT_NE(
    unified_submit.find(
      "optional_child_ready = buildable_optional_child != nullptr"
    ),
    std::string::npos
  );
  EXPECT_NE(
    unified_submit.find("subtitle_work == cuda_conditional_graph::work_flag_e::optional_ocr"),
    std::string::npos
  );
  EXPECT_NE(estimator.find("pending_subtitle_work = subtitle_work"), std::string::npos);
  EXPECT_NE(unified_submit.find("depth_conditional_graph.get()"), std::string::npos);
  EXPECT_EQ(unified_submit.find("enqueueV3("), std::string::npos);
  EXPECT_EQ(unified_submit.find("depth_inference_graph.executable"), std::string::npos);
  const auto signature_refresh_assignment =
    estimator.find("gpu_undecided_requires_force_infer_refresh = true;");
  ASSERT_NE(signature_refresh_assignment, std::string::npos);
  EXPECT_EQ(
    estimator.find(
      "gpu_undecided_requires_force_infer_refresh = true;",
      signature_refresh_assignment + 1u
    ),
    std::string::npos
  ) << "An accepted opaque root must not require a blind force successor";
  EXPECT_NE(
    estimator.find("!detail::cuda_graph_signature_matches("),
    std::string::npos
  );
  EXPECT_NE(estimator.find("gpu_undecided_baseline_authorized("), std::string::npos);
  EXPECT_NE(estimator.find("request.opaque_followup"), std::string::npos);
  EXPECT_NE(
    estimator.find("last_gpu_opaque_transaction_frame_id == 0u &&"),
    std::string::npos
  ) << "A stale CPU-known baseline must not masquerade as a new initial root after opaque work";
  EXPECT_NE(
    estimator.find("last_gpu_opaque_transaction_frame_id ="),
    std::string::npos
  );
  EXPECT_NE(estimator.find("GPU-undecided initial roots"), std::string::npos);
  EXPECT_NE(estimator.find("GPU-undecided follow-up roots"), std::string::npos);
  EXPECT_NE(
    estimator.find("using near_identical_constants_t = std::array<std::uint32_t, 20u>"),
    std::string::npos
  );
  EXPECT_NE(
    estimator.find("constants_desc.ByteWidth = static_cast<UINT>(sizeof(near_identical_constants_t))"),
    std::string::npos
  );
  EXPECT_NE(shader.find("#define NEAR_IDENTICAL_MAX_INFER_OWNER_AGE 4u"),
            std::string::npos);
  EXPECT_NE(
    shader.find("#define NEAR_IDENTICAL_MAX_INFER_OWNER_OBSERVATION_AGE_US 100000u"),
    std::string::npos
  );
  EXPECT_NE(shader.find("#define NEAR_IDENTICAL_HISTORY_OWNER_SCHEMA 2u"),
            std::string::npos);
  EXPECT_NE(shader.find("owner_at_or_before_host_baseline"), std::string::npos);
  EXPECT_NE(shader.find("low_delta <= NEAR_IDENTICAL_MAX_INFER_OWNER_AGE"),
            std::string::npos);
  EXPECT_NE(
    shader.find(
      "observation_delta_low < NEAR_IDENTICAL_MAX_INFER_OWNER_OBSERVATION_AGE_US"
    ),
    std::string::npos
  );
  EXPECT_NE(shader.find("near_identical_observation_timestamp_low"), std::string::npos);
  EXPECT_NE(shader.find("near_identical_observation_timestamp_high"), std::string::npos);
  EXPECT_NE(shader.find("uint near_identical_timestamp_padding1;"), std::string::npos);
  EXPECT_EQ(
    estimator.find("depth_conditional_graph.empty()) {\n                  enqueued ="),
    std::string::npos
  );
  EXPECT_NE(
    unified_submit.find(
      "\"conditional graph launch failed\",\n                  launched,\n"
      "                  true,\n                  true"
    ),
    std::string::npos
  );
  const auto wrapper_reset_begin = estimator.find(
    "[[nodiscard]] bool reset_depth_conditional_graph()"
  );
  const auto wrapper_reset_end = estimator.find(
    "[[nodiscard]] bool destroy_inference_graph(", wrapper_reset_begin
  );
  ASSERT_NE(wrapper_reset_begin, std::string::npos);
  ASSERT_NE(wrapper_reset_end, std::string::npos);
  const auto wrapper_reset = estimator.substr(
    wrapper_reset_begin, wrapper_reset_end - wrapper_reset_begin
  );
  EXPECT_NE(
    wrapper_reset.find("if (depth_conditional_optional_child_graph)"),
    std::string::npos
  );
  const auto prepare_begin = estimator.find("bool prepare_depth_inference_graph(");
  const auto prepare_end = estimator.find("// Caching", prepare_begin);
  ASSERT_NE(prepare_begin, std::string::npos);
  ASSERT_NE(prepare_end, std::string::npos);
  const auto prepare_body = estimator.substr(prepare_begin, prepare_end - prepare_begin);
  EXPECT_NE(prepare_body.find("private bootstrap only"), std::string::npos);
  EXPECT_NE(prepare_body.find("exec_context->enqueueV3(cu_stream)"), std::string::npos);
  EXPECT_NE(prepare_body.find("ocr_exec_context->enqueueV3(cu_stream)"), std::string::npos);
  EXPECT_EQ(prepare_body.find("cuGraphInstantiateWithFlags"), std::string::npos);

  const auto wrapper_launch = estimator.find(
    "const CUresult launched = cuda.cuGraphLaunch(", submit_begin
  );
  const auto optional_unmap = estimator.find(
    "ocr_unmap_res = cuda.cuGraphicsUnmapResources(", submit_end
  );
  const auto depth_unmap = estimator.find(
    "auto unmap_res = cuda.cuGraphicsUnmapResources(", submit_end
  );
  const auto joined_event = estimator.find(
    "record_inference_done_event(\n          cuda,", depth_unmap
  );
  ASSERT_NE(wrapper_launch, std::string::npos);
  ASSERT_NE(optional_unmap, std::string::npos);
  ASSERT_NE(depth_unmap, std::string::npos);
  ASSERT_NE(joined_event, std::string::npos);
  EXPECT_LT(wrapper_launch, optional_unmap);
  EXPECT_LT(wrapper_launch, depth_unmap);
  EXPECT_LT(optional_unmap, joined_event);
  EXPECT_LT(depth_unmap, joined_event);

  // A CPU-known force-infer still launches the wrapper. Depth postprocess is direct, but every
  // optional observation is receipt-gated: force failure emits infer-without-optional while an
  // opaque invalid receipt restores only the retained normalized depth and freezes all infer,
  // scene, history, OCR, and SLR authority.
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
  const auto optional_args_begin = estimator.find(
    "bool dispatch_near_identical_postprocess_args()"
  );
  const auto optional_args_end = estimator.find(
    "void dispatch_near_identical_reuse_depth()", optional_args_begin
  );
  ASSERT_NE(optional_args_begin, std::string::npos);
  ASSERT_NE(optional_args_end, std::string::npos);
  const auto optional_args_body = estimator.substr(
    optional_args_begin, optional_args_end - optional_args_begin
  );
  EXPECT_NE(optional_args_body.find("context->Dispatch(1u, 1u, 1u)"), std::string::npos);
  EXPECT_NE(
    optional_args_body.find("near_identical_optional_postprocess_args_cs.Get()"),
    std::string::npos
  );
  EXPECT_NE(optional_args_body.find("if (gpu_undecided)"), std::string::npos);
  EXPECT_NE(
    optional_args_body.find("near_identical_postprocess_args_cs.Get()"),
    std::string::npos
  );
  EXPECT_EQ(estimator.find("postprocess_temporal_reset"), std::string::npos);
  EXPECT_EQ(estimator.find("completed_postprocess_temporal_reset"), std::string::npos);
  EXPECT_EQ(estimator.find("pending_force_temporal_reseed"), std::string::npos);
  EXPECT_EQ(estimator.find("completed_input_domain_reset = true"), std::string::npos);
  EXPECT_EQ(shader.find("exact_changed"), std::string::npos);
  EXPECT_NE(
    compare_contract.find("abs(current_nchw - previous_nchw)"),
    std::string::npos
  );
  EXPECT_NE(shader.find("* 40u <="), std::string::npos);
  EXPECT_NE(shader.find("* 10u <="), std::string::npos);
  EXPECT_EQ(shader.find("NEAR_IDENTICAL_SUBTITLE_MEDIUM_PERCENT"), std::string::npos);
  EXPECT_EQ(shader.find("NEAR_IDENTICAL_SUBTITLE_STRONG_PERCENT"), std::string::npos);
  EXPECT_EQ(shader.find("NearIdenticalResolveSubtitle"), std::string::npos);
  EXPECT_EQ(
    shader.find("NearIdenticalSubtitleLocatorState : register(t6)"),
    std::string::npos
  );
  EXPECT_NE(
    shader.find("NearIdenticalMinMaxEma : register(t6)"),
    std::string::npos
  );
  EXPECT_NE(
    shader.find("#include \"include/depth_valid_history_contract.hlsl\""),
    std::string::npos
  );
  EXPECT_NE(
    shader.find("DepthValidHistoryAdvances(NearIdenticalMinMaxEma[0].w, history_state)"),
    std::string::npos
  );
  EXPECT_NE(estimator.find("minmax_ema_srv.Get(),"), std::string::npos);
  EXPECT_NE(estimator.find("CSSetShaderResources(0u, 7u, inputs)"), std::string::npos);
  EXPECT_NE(estimator.find("CSSetShaderResources(0u, 4u, resolve_inputs)"), std::string::npos);
  EXPECT_EQ(shader.find("NearIdenticalSubtitleCoherentTransition"), std::string::npos);
  EXPECT_EQ(shader.find("NearIdenticalSubtitleStateSteady"), std::string::npos);
  EXPECT_NE(shader.find("near_identical_expected_work"), std::string::npos);
  EXPECT_NE(shader.find("force_exact_publication"), std::string::npos);
  EXPECT_NE(shader.find("NEAR_IDENTICAL_WORK_OPTIONAL_OCR_DUE"), std::string::npos);
  EXPECT_NE(
    shader.find("NEAR_IDENTICAL_WORK_SUBTITLE_OBSERVATION_DUE"),
    std::string::npos
  );
  EXPECT_EQ(shader.find("flags == 4u"), std::string::npos);
  EXPECT_EQ(shader.find("void ocr_restamp_main"), std::string::npos);
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
  #include <bit>
  #include <cmath>
  #include <cstring>
  #include <filesystem>
  #include <limits>
  #include <span>
  #include <string>
  #include <type_traits>
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
  constexpr std::uint64_t baseline_frame_id = 0x12345678ffffffffull;
  constexpr std::uint64_t current_frame_id = 0x1234567900000000ull;
  constexpr std::uint64_t domain_tag = 0xfedcba9876543210ull;
  constexpr std::uint64_t request_token = 0x13579bdf2468ace0ull;
  constexpr std::uint64_t owner_observation_timestamp_us = 1'000'000u;
  constexpr std::uint64_t current_observation_timestamp_us =
    owner_observation_timestamp_us + 99'999u;
  constexpr std::array<float, 4> valid_minmax_ema {0.1f, 0.9f, 1.0f, 1.0f};
  constexpr std::array<float, 4> invalid_minmax_ema {0.1f, 0.9f, 1.0f, 0.0f};

  using decision_words_t =
    std::array<std::uint32_t, models::near_identical_gpu_decision_word_count>;
  using tile_evidence_t = std::array<std::uint32_t, 4>;
  using tile_records_t = std::array<tile_evidence_t, tile_group_count>;

  constexpr std::size_t word(const models::near_identical_gpu_decision_word_e value) {
    return models::near_identical_gpu_decision_word_index(value);
  }

  tile_records_t classify_model_inputs(
    const std::span<const float> current,
    const std::span<const float> previous,
    const models::depth_tensor_content_rect_t tensor_content,
    const std::span<const std::uint32_t> exclusion = {}
  ) {
    tile_records_t records {};
    if (current.size() != 3u * field_texels || previous.size() != current.size() ||
        (!exclusion.empty() && exclusion.size() != field_texels)) {
      return records;
    }
    constexpr std::array standard_deviation {0.229f, 0.224f, 0.225f};
    for (UINT y = 0u; y < field_height; ++y) {
      for (UINT x = 0u; x < field_width; ++x) {
        const auto index = static_cast<std::size_t>(y) * field_width + x;
        const bool admitted =
          x >= tensor_content.left && x < tensor_content.right &&
          y >= tensor_content.top && y < tensor_content.bottom &&
          (exclusion.empty() || exclusion[index] == 0u);
        if (!admitted) {
          continue;
        }
        auto &tile = records[(y / 16u) * tile_group_width + x / 16u];
        ++tile[0];
        bool finite = true;
        float maximum_delta = 0.0f;
        for (UINT channel = 0u; channel < 3u; ++channel) {
          const auto channel_index = index + channel * field_texels;
          finite = finite && std::isfinite(current[channel_index]) &&
                   std::isfinite(previous[channel_index]);
          maximum_delta = std::max(
            maximum_delta,
            std::abs(current[channel_index] - previous[channel_index]) *
              standard_deviation[channel]
          );
        }
        if (!finite) {
          ++tile[3];
        } else {
          tile[1] += maximum_delta >= 1.0f / 64.0f ? 1u : 0u;
          tile[2] += maximum_delta >= 0.20f ? 1u : 0u;
        }
      }
    }
    return records;
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
      const std::filesystem::path preprocess_path =
        SUNSHINE_SHADERS_DIR "/rgb_to_nchw_cs.hlsl";
      const std::filesystem::path fused_preprocess_path =
        SUNSHINE_SHADERS_DIR "/rgb_to_nchw_near_identical_cs.hlsl";
      if (!compile(preprocess_path, "main", preprocess_shader_, error) ||
          !compile_specialized_oracle(error) ||
          !compile(fused_preprocess_path, "fused_main", fused_preprocess_shader_, error) ||
          !compile(shader_path, "resolve_main", resolve_shader_, error) ||
          !compile(shader_path, "history_owner_main", history_owner_shader_, error) ||
          !compile(shader_path, "scene_seed_main", scene_seed_shader_, error) ||
          !compile(shader_path, "postprocess_args_main", args_shader_, error) ||
          !compile(
            shader_path,
            "optional_postprocess_args_main",
            optional_args_shader_,
            error
          )) {
        return initialize_result_e::failed;
      }

      D3D11_BUFFER_DESC constants_desc {};
      constants_desc.ByteWidth = 16u * sizeof(std::uint32_t);
      constants_desc.Usage = D3D11_USAGE_DEFAULT;
      constants_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      if (FAILED(device_->CreateBuffer(&constants_desc, nullptr, &depth_constants_))) {
        error = "could not create depth constant buffer";
        return initialize_result_e::failed;
      }
      constants_desc.ByteWidth = 20u * sizeof(std::uint32_t);
      if (FAILED(device_->CreateBuffer(&constants_desc, nullptr, &detector_constants_))) {
        error = "could not create detector constant buffers";
        return initialize_result_e::failed;
      }
      constants_desc.ByteWidth = 4u * sizeof(std::uint32_t);
      if (FAILED(device_->CreateBuffer(&constants_desc, nullptr, &source_region_constants_))) {
        error = "could not create source-region constant buffer";
        return initialize_result_e::failed;
      }
      const auto create_fused_mode = [&] (
                                       const std::uint32_t enabled,
                                       ComPtr<ID3D11Buffer> &buffer
                                     ) {
        const std::array<std::uint32_t, 4u> constants {enabled, 0u, 0u, 0u};
        D3D11_BUFFER_DESC desc {};
        desc.ByteWidth = static_cast<UINT>(sizeof(constants));
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA data {constants.data(), 0u, 0u};
        return SUCCEEDED(device_->CreateBuffer(&desc, &data, &buffer));
      };
      if (!create_fused_mode(0u, fused_force_constants_) ||
          !create_fused_mode(1u, fused_compare_constants_)) {
        error = "could not create fused preprocess mode buffers";
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
          )) {
        error = "could not create near-identical GPU buffers";
        return initialize_result_e::failed;
      }
      if (!create_raw_decision_buffer()) {
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

    struct preprocess_result_t {
      std::vector<float> tensor;
      std::vector<float> appearance;
      std::vector<std::uint32_t> exclusion;
      tile_records_t tiles {};
    };

    bool run_preprocess_and_tile_evidence(
      const bool fused,
      const std::span<const std::array<float, 4>> source,
      const UINT source_width,
      const UINT source_height,
      const std::array<std::uint32_t, 4> source_region,
      const models::depth_tensor_content_rect_t tensor_content,
      const std::span<const float> previous,
      preprocess_result_t &result,
      std::string &error,
      const bool evidence_enabled = true
    ) {
      if (source.size() != static_cast<std::size_t>(source_width) * source_height ||
          previous.size() != 3u * field_texels) {
        error = "invalid preprocess parity input size";
        return false;
      }

      D3D11_TEXTURE2D_DESC input_desc {};
      input_desc.Width = source_width;
      input_desc.Height = source_height;
      input_desc.MipLevels = 1u;
      input_desc.ArraySize = 1u;
      input_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
      input_desc.SampleDesc.Count = 1u;
      input_desc.Usage = D3D11_USAGE_IMMUTABLE;
      input_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA input_data {
        source.data(), source_width * static_cast<UINT>(sizeof(source.front())), 0u
      };
      ComPtr<ID3D11Texture2D> input_texture;
      ComPtr<ID3D11ShaderResourceView> input_srv;
      if (FAILED(device_->CreateTexture2D(&input_desc, &input_data, &input_texture)) ||
          FAILED(device_->CreateShaderResourceView(
            input_texture.Get(), nullptr, &input_srv
          ))) {
        error = "could not create preprocess parity source texture";
        return false;
      }

      ComPtr<ID3D11Buffer> tensor_buffer;
      ComPtr<ID3D11ShaderResourceView> tensor_srv;
      ComPtr<ID3D11UnorderedAccessView> tensor_uav;
      ComPtr<ID3D11Buffer> appearance_buffer;
      ComPtr<ID3D11ShaderResourceView> appearance_srv;
      ComPtr<ID3D11UnorderedAccessView> appearance_uav;
      ComPtr<ID3D11Buffer> previous_buffer;
      ComPtr<ID3D11ShaderResourceView> previous_srv;
      if (!create_structured_output(
            3u * field_texels,
            sizeof(float),
            tensor_buffer,
            tensor_srv,
            tensor_uav
          ) ||
          !create_structured_output(
            field_texels,
            sizeof(float),
            appearance_buffer,
            appearance_srv,
            appearance_uav
          ) ||
          !create_structured_srv(
            previous, sizeof(float), previous_buffer, previous_srv
          )) {
        error = "could not create preprocess parity structured buffers";
        return false;
      }

      D3D11_TEXTURE2D_DESC exclusion_desc {};
      exclusion_desc.Width = field_width;
      exclusion_desc.Height = field_height;
      exclusion_desc.MipLevels = 1u;
      exclusion_desc.ArraySize = 1u;
      exclusion_desc.Format = DXGI_FORMAT_R32_UINT;
      exclusion_desc.SampleDesc.Count = 1u;
      exclusion_desc.Usage = D3D11_USAGE_DEFAULT;
      exclusion_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      ComPtr<ID3D11Texture2D> exclusion_texture;
      ComPtr<ID3D11UnorderedAccessView> exclusion_uav;
      if (FAILED(device_->CreateTexture2D(
            &exclusion_desc, nullptr, &exclusion_texture
          )) ||
          FAILED(device_->CreateUnorderedAccessView(
            exclusion_texture.Get(), nullptr, &exclusion_uav
          ))) {
        error = "could not create preprocess parity exclusion texture";
        return false;
      }

      update_depth_constants(tensor_content);
      context_->UpdateSubresource(
        source_region_constants_.Get(), 0u, nullptr, source_region.data(), 0u, 0u
      );
      ID3D11Buffer *preprocess_constants[3] = {
        depth_constants_.Get(),
        fused ?
          (evidence_enabled ? fused_compare_constants_.Get() :
                              fused_force_constants_.Get()) :
          nullptr,
        source_region_constants_.Get(),
      };
      ID3D11ShaderResourceView *preprocess_inputs[2] = {
        input_srv.Get(), fused ? previous_srv.Get() : nullptr,
      };
      ID3D11UnorderedAccessView *preprocess_outputs[4] = {
        tensor_uav.Get(), appearance_uav.Get(), exclusion_uav.Get(),
        fused ? tile_uav_.Get() : nullptr,
      };
      if (fused && !evidence_enabled) {
        tile_records_t stale_tiles {};
        for (auto &tile : stale_tiles) {
          tile = {0xA5A5A5A5u, 0x5A5A5A5Au, 0xDEADBEEFu, 0xFFFFFFFFu};
        }
        context_->UpdateSubresource(
          tile_buffer_.Get(), 0u, nullptr, stale_tiles.data(), 0u, 0u
        );
      }
      context_->CSSetConstantBuffers(0u, 3u, preprocess_constants);
      context_->CSSetShaderResources(0u, 2u, preprocess_inputs);
      context_->CSSetUnorderedAccessViews(0u, 4u, preprocess_outputs, nullptr);
      constexpr models::depth_tensor_shape_t shape {field_width, field_height};
      const auto content_area = static_cast<std::uint64_t>(tensor_content.width()) *
                                tensor_content.height();
      const bool specialize_padding =
        !fused && tensor_content.valid(shape) && !tensor_content.full(shape) &&
        (static_cast<std::uint64_t>(field_texels) - content_area) * 8u >= field_texels;
      if (specialize_padding) {
        context_->CSSetShader(oracle_content_shader_.Get(), nullptr, 0u);
        context_->Dispatch(
          (tensor_content.width() + 15u) / 16u,
          (tensor_content.height() + 15u) / 16u,
          1u
        );
        context_->CSSetShader(oracle_pad_shader_.Get(), nullptr, 0u);
        context_->Dispatch(tile_group_width, tile_group_height, 1u);
      } else {
        context_->CSSetShader(
          fused ? fused_preprocess_shader_.Get() : preprocess_shader_.Get(), nullptr, 0u
        );
        context_->Dispatch(tile_group_width, tile_group_height, 1u);
      }
      unbind(2u, 4u, 3u);

      if (!read_buffer(tensor_buffer.Get(), result.tensor, 3u * field_texels, error) ||
          !read_buffer(
            appearance_buffer.Get(), result.appearance, field_texels, error
          ) ||
          !read_texture_u32(exclusion_texture.Get(), result.exclusion, error)) {
        return false;
      }
      if (fused) {
        if (!read_buffer(tile_buffer_.Get(), result.tiles, error)) {
          return false;
        }
      } else {
        result.tiles = classify_model_inputs(
          result.tensor, previous, tensor_content, result.exclusion
        );
      }
      return true;
    }

    bool run_detector(
      const std::vector<float> &current,
      const std::vector<float> &previous,
      const std::array<float, sbs_adaptive_state::word_count> &cut_state,
      decision_words_t &decision,
      std::string &error,
      const std::uint32_t reduce_groups = reduction_group_count,
      const std::uint32_t request_work = 0u,
      const std::uint64_t owner_frame_id = baseline_frame_id,
      const std::uint64_t owner_timestamp_us = owner_observation_timestamp_us,
      const std::uint64_t current_timestamp_us = current_observation_timestamp_us,
      const std::array<float, 4> &minmax_ema = valid_minmax_ema,
      const bool publish_owner = true
    ) {
      if (current.size() != 3u * field_texels || previous.size() != current.size()) {
        error = "invalid model input size";
        return false;
      }
      ComPtr<ID3D11Buffer> state_buffer;
      ComPtr<ID3D11ShaderResourceView> state_srv;
      ComPtr<ID3D11Buffer> minmax_buffer;
      ComPtr<ID3D11ShaderResourceView> minmax_srv;
      if (!create_structured_srv(
            std::span<const float> {cut_state},
            4u * sizeof(float),
            state_buffer,
            state_srv
          ) ||
          !create_structured_srv(
            std::span<const float> {minmax_ema},
            4u * sizeof(float),
            minmax_buffer,
            minmax_srv
          )) {
        error = "could not create detector inputs";
        return false;
      }
      update_depth_constants();
      initialize_transaction(request_work);
      update_detector_constants(
        0u, 0u, 0u, owner_frame_id, 0u, 0u, owner_timestamp_us
      );
      ID3D11Buffer *constants[2] = {depth_constants_.Get(), detector_constants_.Get()};
      ID3D11ShaderResourceView *owner_inputs[7] = {
        nullptr, nullptr, nullptr, nullptr, state_srv.Get(), nullptr, minmax_srv.Get(),
      };
      ID3D11UnorderedAccessView *owner_outputs[3] = {
        nullptr, nullptr, history_owner_uav_.Get(),
      };
      if (publish_owner) {
        context_->CSSetShader(history_owner_shader_.Get(), nullptr, 0u);
        context_->CSSetConstantBuffers(0u, 2u, constants);
        context_->CSSetShaderResources(0u, 7u, owner_inputs);
        context_->CSSetUnorderedAccessViews(0u, 3u, owner_outputs, nullptr);
        context_->Dispatch(1u, 1u, 1u);
        unbind(7u, 3u, 2u);
      }

      update_detector_constants(
        1u,
        tile_group_width,
        tile_group_height,
        current_frame_id,
        baseline_frame_id,
        request_token,
        current_timestamp_us,
        0u,
        reduce_groups,
        request_work
      );
      const auto tiles = classify_model_inputs(current, previous, content);
      context_->UpdateSubresource(
        tile_buffer_.Get(), 0u, nullptr, tiles.data(), 0u, 0u
      );

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
      const std::uint32_t reduce_groups = reduction_group_count,
      const std::uint32_t groups_x = tile_group_width,
      const std::uint32_t groups_y = tile_group_height,
      const std::uint32_t group_count = tile_group_count
    ) {
      context_->UpdateSubresource(
        tile_buffer_.Get(), 0u, nullptr, tiles.data(), 0u, 0u
      );
      initialize_transaction();
      update_detector_constants(
        1u,
        groups_x,
        groups_y,
        current_frame_id,
        baseline_frame_id,
        request_token,
        current_observation_timestamp_us,
        0u,
        reduce_groups,
        0u,
        group_count
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

    bool publish_history_owner(
      const std::uint64_t frame_id,
      const std::uint32_t model_input_history_state,
      std::string &error,
      const std::uint64_t observation_timestamp_us = owner_observation_timestamp_us,
      const std::array<float, 4> &minmax_ema = valid_minmax_ema
    ) {
      ComPtr<ID3D11Buffer> state_buffer;
      ComPtr<ID3D11ShaderResourceView> state_srv;
      ComPtr<ID3D11Buffer> minmax_buffer;
      ComPtr<ID3D11ShaderResourceView> minmax_srv;
      const auto state = history_state(model_input_history_state);
      if (!create_structured_srv(
            std::span<const float> {state},
            4u * sizeof(float),
            state_buffer,
            state_srv
          ) ||
          !create_structured_srv(
            std::span<const float> {minmax_ema},
            4u * sizeof(float),
            minmax_buffer,
            minmax_srv
          )) {
        error = "could not create history-owner state";
        return false;
      }
      update_depth_constants();
      update_detector_constants(
        0u, 0u, 0u, frame_id, 0u, 0u, observation_timestamp_us
      );
      ID3D11Buffer *constants[2] = {depth_constants_.Get(), detector_constants_.Get()};
      ID3D11ShaderResourceView *inputs[7] = {
        nullptr, nullptr, nullptr, nullptr, state_srv.Get(), nullptr, minmax_srv.Get(),
      };
      ID3D11UnorderedAccessView *outputs[3] = {
        nullptr, nullptr, history_owner_uav_.Get(),
      };
      context_->CSSetShader(history_owner_shader_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(0u, 2u, constants);
      context_->CSSetShaderResources(0u, 7u, inputs);
      context_->CSSetUnorderedAccessViews(0u, 3u, outputs, nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind(7u, 3u, 2u);
      return true;
    }

    bool seed_scene_from_history_owner(
      const std::uint64_t frame_id,
      std::array<std::uint32_t, 10> &words,
      std::string &error
    ) {
      update_depth_constants();
      update_detector_constants(
        0u, 0u, 0u, frame_id, 0u, 0u, current_observation_timestamp_us
      );
      ID3D11Buffer *constants[2] = {depth_constants_.Get(), detector_constants_.Get()};
      ID3D11ShaderResourceView *inputs[4] = {
        nullptr, nullptr, nullptr, history_owner_srv_.Get(),
      };
      ID3D11UnorderedAccessView *outputs[6] = {
        nullptr, nullptr, nullptr, nullptr, nullptr, scene_evidence_uav_.Get(),
      };
      context_->CSSetShader(scene_seed_shader_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(0u, 2u, constants);
      context_->CSSetShaderResources(0u, 4u, inputs);
      context_->CSSetUnorderedAccessViews(0u, 6u, outputs, nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind(4u, 6u, 2u);
      return read_scene_evidence(words, error);
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
      update_depth_constants();
      update_detector_constants(
        1u,
        tile_group_width,
        tile_group_height,
        current_frame_id,
        baseline_frame_id,
        constants_token,
        current_observation_timestamp_us,
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

    bool initialize_and_read_transaction(
      const std::uint32_t request_work,
      decision_words_t &decision,
      std::string &error
    ) {
      initialize_transaction(request_work);
      return read_decision(decision, error);
    }

    bool resolve_optional_postprocess_args(
      const models::near_identical_gpu_branch_e branch,
      const bool valid_receipt,
      const bool optional_receipt,
      const bool force_infer,
      decision_words_t &decision,
      std::string &error,
      const std::uint64_t record_token = request_token,
      const std::uint64_t constants_token = request_token,
      const std::uint32_t request_work = models::near_identical_work_optional_ocr,
      const bool bind_optional_cookie = true,
      const std::uint32_t expected_work = std::numeric_limits<std::uint32_t>::max()
    ) {
      if (!read_decision(decision, error)) return false;
      const auto resolved = static_cast<std::uint32_t>(branch);
      const auto optional_marker = optional_receipt ?
                                     models::near_identical_optional_receipt_magic :
                                     0u;
      decision[word(models::near_identical_gpu_decision_word_e::decision)] = resolved;
      decision[word(models::near_identical_gpu_decision_word_e::decision_cookie)] =
        resolved ^ models::near_identical_decision_cookie ^
        (bind_optional_cookie ? optional_marker : 0u);
      const auto low = static_cast<std::uint32_t>(record_token);
      const auto high = static_cast<std::uint32_t>(record_token >> 32u);
      decision[word(models::near_identical_gpu_decision_word_e::decision_token_low)] = low;
      decision[word(models::near_identical_gpu_decision_word_e::decision_token_high)] = high;
      decision[word(models::near_identical_gpu_decision_word_e::decision_token_low_cookie)] =
        low ^ models::near_identical_token_low_cookie;
      decision[word(models::near_identical_gpu_decision_word_e::decision_token_high_cookie)] =
        high ^ models::near_identical_token_high_cookie;
      decision[word(models::near_identical_gpu_decision_word_e::decision_magic)] =
        valid_receipt ? models::near_identical_receipt_magic :
                        models::near_identical_proposal_magic;
      decision[word(models::near_identical_gpu_decision_word_e::decision_reserved)] =
        optional_marker;
      decision[word(models::near_identical_gpu_decision_word_e::request_token_low)] = low;
      decision[word(models::near_identical_gpu_decision_word_e::request_token_high)] = high;
      decision[word(models::near_identical_gpu_decision_word_e::request_token_low_cookie)] =
        low ^ models::near_identical_token_low_cookie;
      decision[word(models::near_identical_gpu_decision_word_e::request_token_high_cookie)] =
        high ^ models::near_identical_token_high_cookie;
      decision[word(models::near_identical_gpu_decision_word_e::request_magic)] =
        models::near_identical_request_magic;
      const auto work_flags = request_work;
      decision[word(models::near_identical_gpu_decision_word_e::request_work_flags)] =
        work_flags;
      decision[word(models::near_identical_gpu_decision_word_e::request_work_flags_cookie)] =
        work_flags == 0u ? 0u : work_flags ^ models::near_identical_work_flags_cookie;
      decision[word(models::near_identical_gpu_decision_word_e::request_reserved)] = 0u;
      context_->UpdateSubresource(
        decision_buffer_.Get(), 0u, nullptr, decision.data(), 0u, 0u
      );
      update_depth_constants();
      update_detector_constants(
        force_infer ? 2u : 1u,
        tile_group_width,
        tile_group_height,
        current_frame_id,
        baseline_frame_id,
        constants_token,
        current_observation_timestamp_us,
        1u,
        reduction_group_count,
        expected_work == std::numeric_limits<std::uint32_t>::max() ?
          request_work : expected_work
      );
      ID3D11Buffer *constants[2] = {depth_constants_.Get(), detector_constants_.Get()};
      ID3D11UnorderedAccessView *outputs[4] = {
        nullptr, nullptr, nullptr, decision_uav_.Get(),
      };
      context_->CSSetShader(optional_args_shader_.Get(), nullptr, 0u);
      context_->CSSetConstantBuffers(0u, 2u, constants);
      context_->CSSetUnorderedAccessViews(0u, 4u, outputs, nullptr);
      context_->Dispatch(1u, 1u, 1u);
      unbind(0u, 4u, 2u);
      return read_decision(decision, error);
    }

  private:
    bool compile_specialized_oracle(std::string &error) {
      namespace cache = models::host_sbs_shader_cache;
      constexpr std::array oracle_specs {
        cache::shader_spec {
          "tests/fixtures/rgb_to_nchw_specialized_oracle_cs.hlsl",
          "oracle_content_main",
          "cs_5_0",
        },
        cache::shader_spec {
          "tests/fixtures/rgb_to_nchw_specialized_oracle_cs.hlsl",
          "oracle_pad_main",
          "cs_5_0",
        },
      };
      const auto sources = cache::snapshot_sources(
        std::filesystem::path {SUNSHINE_SOURCE_DIR}, oracle_specs
      );
      if (!sources) {
        error = "could not snapshot the retired preprocess test oracle";
        return false;
      }
      const auto content_bytecode = cache::get(sources, oracle_specs[0]);
      const auto pad_bytecode = cache::get(sources, oracle_specs[1]);
      if (!content_bytecode || !pad_bytecode) {
        error = "could not compile the retired preprocess test oracle";
        return false;
      }
      if (FAILED(device_->CreateComputeShader(
            content_bytecode->data(),
            content_bytecode->size(),
            nullptr,
            &oracle_content_shader_
          )) ||
          FAILED(device_->CreateComputeShader(
            pad_bytecode->data(),
            pad_bytecode->size(),
            nullptr,
            &oracle_pad_shader_
          ))) {
        error = "could not create the retired preprocess test-oracle shaders";
        return false;
      }
      return true;
    }

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

    template<typename T>
    bool create_structured_srv(
      const std::span<const T> values,
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

    void update_depth_constants(
      const models::depth_tensor_content_rect_t tensor_content = content
    ) {
      std::array<std::uint32_t, 16> constants {};
      constants[0] = field_width;
      constants[1] = field_height;
      constants[9] = tensor_content.left;
      constants[10] = tensor_content.top;
      constants[11] = tensor_content.right;
      constants[12] = tensor_content.bottom;
      context_->UpdateSubresource(
        depth_constants_.Get(), 0u, nullptr, constants.data(), 0u, 0u
      );
    }

    template<typename T>
    bool read_buffer(
      ID3D11Buffer *source,
      T &output,
      std::string &error
    ) {
      D3D11_BUFFER_DESC desc {};
      source->GetDesc(&desc);
      D3D11_BUFFER_DESC staging_desc = desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0u;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      staging_desc.MiscFlags = 0u;
      staging_desc.StructureByteStride = 0u;
      ComPtr<ID3D11Buffer> staging;
      if (FAILED(device_->CreateBuffer(&staging_desc, nullptr, &staging))) {
        error = "could not create preprocess parity staging buffer";
        return false;
      }
      context_->CopyResource(staging.Get(), source);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context_->Map(
            staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped
          )) || !mapped.pData) {
        error = "could not read preprocess parity buffer";
        return false;
      }
      static_assert(std::is_trivially_copyable_v<T>);
      if (sizeof(output) > desc.ByteWidth) {
        context_->Unmap(staging.Get(), 0u);
        error = "preprocess parity output is larger than its buffer";
        return false;
      }
      std::memcpy(std::addressof(output), mapped.pData, sizeof(output));
      context_->Unmap(staging.Get(), 0u);
      return true;
    }

    bool read_texture_u32(
      ID3D11Texture2D *source,
      std::vector<std::uint32_t> &output,
      std::string &error
    ) {
      D3D11_TEXTURE2D_DESC desc {};
      source->GetDesc(&desc);
      D3D11_TEXTURE2D_DESC staging_desc = desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0u;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      staging_desc.MiscFlags = 0u;
      ComPtr<ID3D11Texture2D> staging;
      if (FAILED(device_->CreateTexture2D(&staging_desc, nullptr, &staging))) {
        error = "could not create preprocess parity texture staging";
        return false;
      }
      context_->CopyResource(staging.Get(), source);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context_->Map(
            staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped
          )) || !mapped.pData) {
        error = "could not read preprocess parity texture";
        return false;
      }
      output.resize(static_cast<std::size_t>(desc.Width) * desc.Height);
      for (UINT y = 0u; y < desc.Height; ++y) {
        std::memcpy(
          output.data() + static_cast<std::size_t>(y) * desc.Width,
          static_cast<const std::byte *>(mapped.pData) +
            static_cast<std::size_t>(y) * mapped.RowPitch,
          static_cast<std::size_t>(desc.Width) * sizeof(std::uint32_t)
        );
      }
      context_->Unmap(staging.Get(), 0u);
      return true;
    }

    bool read_buffer(
      ID3D11Buffer *source,
      std::vector<float> &output,
      const std::size_t count,
      std::string &error
    ) {
      D3D11_BUFFER_DESC desc {};
      source->GetDesc(&desc);
      if (desc.ByteWidth < count * sizeof(float)) {
        error = "preprocess parity buffer is undersized";
        return false;
      }
      output.resize(count);
      D3D11_BUFFER_DESC staging_desc = desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0u;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      staging_desc.MiscFlags = 0u;
      staging_desc.StructureByteStride = 0u;
      ComPtr<ID3D11Buffer> staging;
      if (FAILED(device_->CreateBuffer(&staging_desc, nullptr, &staging))) {
        error = "could not create preprocess parity staging buffer";
        return false;
      }
      context_->CopyResource(staging.Get(), source);
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context_->Map(
            staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped
          )) || !mapped.pData) {
        error = "could not read preprocess parity buffer";
        return false;
      }
      std::memcpy(output.data(), mapped.pData, count * sizeof(float));
      context_->Unmap(staging.Get(), 0u);
      return true;
    }

    void update_detector_constants(
      const std::uint32_t flags,
      const std::uint32_t groups_x,
      const std::uint32_t groups_y,
      const std::uint64_t current,
      const std::uint64_t baseline,
      const std::uint64_t token,
      const std::uint64_t observation_timestamp_us,
      const std::uint32_t stream_frame_delta = 0u,
      const std::uint32_t reduce_groups = reduction_group_count,
      const std::uint32_t expected_work = 0u,
      const std::uint32_t group_count = std::numeric_limits<std::uint32_t>::max()
    ) {
      const std::array<std::uint32_t, 20> constants {
        flags,
        groups_x,
        groups_y,
        group_count == std::numeric_limits<std::uint32_t>::max() ?
          groups_x * groups_y : group_count,
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
        expected_work,
        expected_work == 0u ?
          0u : expected_work ^ models::near_identical_work_flags_cookie,
        static_cast<std::uint32_t>(observation_timestamp_us),
        static_cast<std::uint32_t>(observation_timestamp_us >> 32u),
        0u,
        0u,
      };
      context_->UpdateSubresource(
        detector_constants_.Get(), 0u, nullptr, constants.data(), 0u, 0u
      );
    }

    void initialize_transaction(const std::uint32_t work_flags = 0u) {
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
      initial[word(models::near_identical_gpu_decision_word_e::request_work_flags)] =
        work_flags;
      initial[word(models::near_identical_gpu_decision_word_e::request_work_flags_cookie)] =
        work_flags == 0u ? 0u : work_flags ^ models::near_identical_work_flags_cookie;
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
      std::array<ID3D11ShaderResourceView *, 7> null_srvs {};
      std::array<ID3D11UnorderedAccessView *, 6> null_uavs {};
      std::array<ID3D11Buffer *, 3> null_constants {};
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
    ComPtr<ID3D11ComputeShader> preprocess_shader_;
    ComPtr<ID3D11ComputeShader> oracle_content_shader_;
    ComPtr<ID3D11ComputeShader> oracle_pad_shader_;
    ComPtr<ID3D11ComputeShader> fused_preprocess_shader_;
    ComPtr<ID3D11ComputeShader> resolve_shader_;
    ComPtr<ID3D11ComputeShader> history_owner_shader_;
    ComPtr<ID3D11ComputeShader> scene_seed_shader_;
    ComPtr<ID3D11ComputeShader> args_shader_;
    ComPtr<ID3D11ComputeShader> optional_args_shader_;
    ComPtr<ID3D11Buffer> depth_constants_;
    ComPtr<ID3D11Buffer> detector_constants_;
    ComPtr<ID3D11Buffer> source_region_constants_;
    ComPtr<ID3D11Buffer> fused_force_constants_;
    ComPtr<ID3D11Buffer> fused_compare_constants_;
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
       index += 4u) {
    EXPECT_EQ(decision[index], 0u) << "indirect X word " << index;
  }
}

TEST(HostSbsNearIdenticalDetectorGpuTest, NonfiniteInputInfers) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;
  const auto previous = uniform_model_input();
  decision_words_t decision {};
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

TEST(HostSbsNearIdenticalDetectorGpuTest, OptionalPreprocessRequiresAuthenticatedProposalAndWork) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;
  decision_words_t decision {};
  ASSERT_TRUE(fixture.initialize_and_read_transaction(true, decision, error)) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_x)],
    0u
  );
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_y)],
    0u
  );
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_z)],
    0u
  );

  const auto previous = uniform_model_input();
  ASSERT_TRUE(fixture.run_detector(
    previous,
    previous,
    history_state(1u),
    decision,
    error,
    reduction_group_count,
    true
  )) << error;
  ASSERT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::reuse)
  );
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_x)],
    0u
  ) << "ordinary DAV2 reuse must hold the coherent OCR/SLR/condition tuple";
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_y)],
    1u
  );
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_z)],
    1u
  );

  ASSERT_TRUE(fixture.run_detector(
    previous,
    previous,
    history_state(1u),
    decision,
    error,
    reduction_group_count,
    models::near_identical_work_optional_ocr_due
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::reuse)
  );
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_x)],
    60u
  ) << "cadence-due OCR preprocesses independently of the depth branch";
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_y)],
    10u
  );

  ASSERT_TRUE(fixture.run_detector(
    previous,
    previous,
    history_state(1u),
    decision,
    error,
    reduction_group_count,
    models::near_identical_work_subtitle_observation_due
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::reuse)
  );
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_x)],
    0u
  ) << "cadence-due abstention publishes later without an OCR child";

  ASSERT_TRUE(fixture.run_detector(
    previous,
    previous,
    history_state(1u),
    decision,
    error,
    reduction_group_count,
    1u << 2u
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision_magic)],
    0u
  ) << "retired flag 4 is not authenticated";
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_x)],
    0u
  );

  auto changed = previous;
  add_first_content_pixel_deltas(changed, 786u, 0.02f);
  ASSERT_TRUE(fixture.run_detector(
    changed,
    previous,
    history_state(1u),
    decision,
    error,
    reduction_group_count,
    true
  )) << error;
  ASSERT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_x)],
    60u
  );
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_y)],
    10u
  );
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_z)],
    1u
  );

  ASSERT_TRUE(fixture.run_detector(
    changed,
    previous,
    history_state(1u),
    decision,
    error,
    reduction_group_count + 1u,
    true
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_x)],
    0u
  ) << "A malformed detector request must not prepare stale OCR input";
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision_magic)],
    0u
  ) << "Detector authority and optional-preprocess authority must publish atomically";

  const cuda_conditional_graph::decision_record_t proposal {
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    decision[word(models::near_identical_gpu_decision_word_e::decision_cookie)],
    decision[word(models::near_identical_gpu_decision_word_e::decision_token_low)],
    decision[word(models::near_identical_gpu_decision_word_e::decision_token_high)],
    decision[word(models::near_identical_gpu_decision_word_e::decision_token_low_cookie)],
    decision[word(models::near_identical_gpu_decision_word_e::decision_token_high_cookie)],
    decision[word(models::near_identical_gpu_decision_word_e::decision_magic)],
    decision[word(models::near_identical_gpu_decision_word_e::decision_reserved)],
  };
  const cuda_conditional_graph::request_record_t request {
    decision[word(models::near_identical_gpu_decision_word_e::request_token_low)],
    decision[word(models::near_identical_gpu_decision_word_e::request_token_high)],
    decision[word(models::near_identical_gpu_decision_word_e::request_token_low_cookie)],
    decision[word(models::near_identical_gpu_decision_word_e::request_token_high_cookie)],
    decision[word(models::near_identical_gpu_decision_word_e::request_magic)],
    decision[word(models::near_identical_gpu_decision_word_e::request_work_flags)],
    decision[word(models::near_identical_gpu_decision_word_e::request_work_flags_cookie)],
    decision[word(models::near_identical_gpu_decision_word_e::request_reserved)],
  };
  const auto receipt = cuda_conditional_graph::resolve_proposal(
    proposal, request, true
  );
  EXPECT_EQ(
    receipt.decision,
    static_cast<std::uint32_t>(cuda_conditional_graph::branch_e::infer)
  );
  EXPECT_FALSE(cuda_conditional_graph::authenticated_optional_infer_receipt(
    receipt, request
  )) << "The CUDA setter must fail depth open without running OCR";
  EXPECT_EQ(receipt.reserved, 0u);
}

TEST(HostSbsNearIdenticalDetectorGpuTest, InferOwnerRequiresN4AndSub100msObservationAge) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;

  const auto input = uniform_model_input();
  decision_words_t decision {};

  // The retained infer owner remains authoritative at exactly four source frames when its
  // observation is still strictly younger than 100 ms.
  ASSERT_TRUE(fixture.run_detector(
    input,
    input,
    history_state(1u),
    decision,
    error,
    reduction_group_count,
    models::near_identical_work_optional_ocr,
    current_frame_id - 4u,
    owner_observation_timestamp_us,
    owner_observation_timestamp_us + 99'999u
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::reuse)
  );
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision_magic)],
    models::near_identical_proposal_magic
  );
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_x)],
    0u
  );

  // A fifth frame is outside the retained-owner proof even when its timestamp is young.
  ASSERT_TRUE(fixture.run_detector(
    input,
    input,
    history_state(1u),
    decision,
    error,
    reduction_group_count,
    0u,
    current_frame_id - 5u,
    owner_observation_timestamp_us,
    owner_observation_timestamp_us + 99'999u
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );

  // The time bound is strict: exactly 100 ms is stale even at a one-frame owner age.
  ASSERT_TRUE(fixture.run_detector(
    input,
    input,
    history_state(1u),
    decision,
    error,
    reduction_group_count,
    0u,
    baseline_frame_id,
    owner_observation_timestamp_us,
    owner_observation_timestamp_us + 100'000u
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );

  // Missing or regressed observation clocks cannot authenticate an otherwise matching owner.
  ASSERT_TRUE(fixture.run_detector(
    input,
    input,
    history_state(1u),
    decision,
    error,
    reduction_group_count,
    0u,
    baseline_frame_id,
    0u,
    current_observation_timestamp_us
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );
  ASSERT_TRUE(fixture.run_detector(
    input,
    input,
    history_state(1u),
    decision,
    error,
    reduction_group_count,
    0u,
    baseline_frame_id,
    owner_observation_timestamp_us,
    owner_observation_timestamp_us - 1u
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );
}

TEST(HostSbsNearIdenticalDetectorGpuTest, HistoryOwnerUsesExactValidHistoryPredicate) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;

  const auto quiet = quiet_tile_records();
  decision_words_t decision {};
  const auto expect_owner = [&fixture, &quiet, &decision, &error](
    const std::uint32_t history_state_value,
    const std::array<float, 4> &minmax_ema,
    const models::near_identical_gpu_branch_e expected_branch
  ) {
    SCOPED_TRACE(::testing::Message()
                 << "history_state=" << history_state_value
                 << " minmax_validity=" << minmax_ema[3]);
    ASSERT_TRUE(fixture.publish_history_owner(
      baseline_frame_id,
      history_state_value,
      error,
      owner_observation_timestamp_us,
      minmax_ema
    )) << error;
    ASSERT_TRUE(fixture.resolve_seeded_tiles(quiet, decision, error)) << error;
    EXPECT_EQ(
      decision[word(models::near_identical_gpu_decision_word_e::decision)],
      static_cast<std::uint32_t>(expected_branch)
    );
  };

  expect_owner(1u, valid_minmax_ema, models::near_identical_gpu_branch_e::reuse);
  expect_owner(3u, valid_minmax_ema, models::near_identical_gpu_branch_e::reuse);
  expect_owner(2u, valid_minmax_ema, models::near_identical_gpu_branch_e::infer);
  expect_owner(4u, valid_minmax_ema, models::near_identical_gpu_branch_e::infer);
  expect_owner(1u, invalid_minmax_ema, models::near_identical_gpu_branch_e::infer);
}

TEST(HostSbsNearIdenticalDetectorGpuTest, InvalidSuccessorClearsPreviouslyValidOwner) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;

  // A is a valid copied owner. B enters the held geometry-confirmation state and must invalidate
  // that owner instead of leaving A authenticated under B's newer tuple metadata.
  ASSERT_TRUE(fixture.publish_history_owner(
    baseline_frame_id - 1u,
    1u,
    error,
    owner_observation_timestamp_us,
    valid_minmax_ema
  )) << error;
  ASSERT_TRUE(fixture.publish_history_owner(
    baseline_frame_id,
    4u,
    error,
    owner_observation_timestamp_us + 1u,
    valid_minmax_ema
  )) << error;

  // C has pixels identical to A, but B invalidated the owner proof. Skip C owner publication to
  // model the detector consuming exactly the A/B history state left by those two publications.
  const auto input_a = uniform_model_input();
  decision_words_t decision {};
  ASSERT_TRUE(fixture.run_detector(
    input_a,
    input_a,
    history_state(1u),
    decision,
    error,
    reduction_group_count,
    0u,
    baseline_frame_id,
    owner_observation_timestamp_us,
    current_observation_timestamp_us,
    valid_minmax_ema,
    false
  )) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );
}

TEST(HostSbsNearIdenticalDetectorGpuTest, SubtitlePublicationAuthenticatesOrdinaryAndDueWork) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;
  decision_words_t decision {};
  ASSERT_TRUE(fixture.initialize_and_read_transaction(
    models::near_identical_work_optional_ocr, decision, error
  )) << error;

  const auto expect_gates = [&decision](
    const std::uint32_t optional_cells_x,
    const std::uint32_t optional_one_x,
    const std::uint32_t record_x,
    const std::uint32_t observation_x,
    const std::uint32_t condition_grid_x
  ) {
    EXPECT_EQ(
      decision[word(models::near_identical_gpu_decision_word_e::optional_cells_x)],
      optional_cells_x
    );
    EXPECT_EQ(
      decision[word(models::near_identical_gpu_decision_word_e::optional_one_x)],
      optional_one_x
    );
    EXPECT_EQ(
      decision[word(
        models::near_identical_gpu_decision_word_e::infer_without_optional_one_x
      )],
      record_x
    );
    EXPECT_EQ(
      decision[word(models::near_identical_gpu_decision_word_e::observation_one_x)],
      observation_x
    );
    EXPECT_EQ(
      decision[word(models::near_identical_gpu_decision_word_e::optional_preprocess_x)],
      condition_grid_x
    );
  };

  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::infer,
    true,
    true,
    false,
    decision,
    error
  )) << error;
  expect_gates(4u, 1u, 0u, 1u, 7u);
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::optional_cells_y)],
    40u
  );

  // This is a correctly cookied forged OOCR marker, not merely a missing optional receipt.
  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::reuse,
    true,
    true,
    false,
    decision,
    error
  )) << error;
  expect_gates(0u, 0u, 0u, 0u, 0u);

  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::reuse,
    true,
    false,
    false,
    decision,
    error
  )) << error;
  expect_gates(0u, 0u, 0u, 0u, 0u);

  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::reuse,
    true,
    true,
    false,
    decision,
    error,
    request_token,
    request_token,
    models::near_identical_work_optional_ocr_due
  )) << error;
  expect_gates(4u, 1u, 0u, 1u, 7u);

  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::reuse,
    true,
    false,
    false,
    decision,
    error,
    request_token,
    request_token,
    models::near_identical_work_optional_ocr_due
  )) << error;
  expect_gates(0u, 0u, 1u, 1u, 7u);

  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::reuse,
    true,
    false,
    false,
    decision,
    error,
    request_token,
    request_token,
    models::near_identical_work_subtitle_observation_due
  )) << error;
  expect_gates(0u, 0u, 1u, 1u, 7u);

  // Due observation is branch-independent but can never authenticate an OOCR marker.
  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::reuse,
    true,
    true,
    false,
    decision,
    error,
    request_token,
    request_token,
    models::near_identical_work_subtitle_observation_due
  )) << error;
  expect_gates(0u, 0u, 0u, 0u, 0u);

  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::infer,
    false,
    true,
    false,
    decision,
    error
  )) << error;
  expect_gates(0u, 0u, 0u, 0u, 0u);

  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::infer,
    false,
    true,
    true,
    decision,
    error
  )) << error;
  expect_gates(0u, 0u, 1u, 1u, 7u);

  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::infer,
    false,
    false,
    true,
    decision,
    error,
    request_token,
    request_token,
    models::near_identical_work_subtitle_observation_due
  )) << error;
  expect_gates(0u, 0u, 1u, 1u, 7u);

  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::infer,
    true,
    true,
    false,
    decision,
    error,
    request_token,
    request_token,
    true,
    false
  )) << error;
  expect_gates(0u, 0u, 0u, 0u, 0u);

  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::infer,
    true,
    true,
    false,
    decision,
    error,
    0u,
    0u
  )) << error;
  expect_gates(0u, 0u, 0u, 0u, 0u);

  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::infer,
    true,
    true,
    false,
    decision,
    error,
    request_token,
    request_token,
    0u
  )) << error;
  expect_gates(0u, 0u, 0u, 0u, 0u);

  // The retired flag-4 mode is correctly cookied but outside the exact work allowlist.
  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::reuse,
    true,
    false,
    false,
    decision,
    error,
    request_token,
    request_token,
    1u << 2u
  )) << error;
  expect_gates(0u, 0u, 0u, 0u, 0u);

  ASSERT_TRUE(fixture.resolve_optional_postprocess_args(
    models::near_identical_gpu_branch_e::reuse,
    true,
    false,
    false,
    decision,
    error,
    request_token,
    request_token,
    models::near_identical_work_subtitle_observation
  )) << error;
  expect_gates(0u, 0u, 0u, 0u, 0u);
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

TEST(HostSbsNearIdenticalDetectorGpuTest, MalformedTileShapeFailsClosed) {
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
    input, input, history_state(1u), decision, error
  )) << error;
  const auto quiet = quiet_tile_records();
  const auto expect_infer = [&](const UINT groups_x, const UINT groups_y,
                                const UINT group_count) {
    ASSERT_TRUE(fixture.resolve_seeded_tiles(
      quiet,
      decision,
      error,
      reduction_group_count,
      groups_x,
      groups_y,
      group_count
    )) << error;
    EXPECT_EQ(
      decision[word(models::near_identical_gpu_decision_word_e::decision)],
      static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
    );
  };
  expect_infer(21u, 2u, tile_group_count);
  expect_infer(tile_group_width, tile_group_height, tile_group_count - 1u);
  expect_infer(tile_group_width, tile_group_height, tile_group_count + 1u);
  expect_infer(6u, 7u, tile_group_count);

  ASSERT_TRUE(fixture.resolve_seeded_tiles(quiet, decision, error)) << error;
  EXPECT_EQ(
    decision[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::reuse)
  ) << "Malformed constants must not remove WARP or poison the next valid resolve";
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

TEST(HostSbsNearIdenticalDetectorGpuTest, SceneSeedUsesLastAcceptedInferOwner) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;

  constexpr std::uint64_t frame_a = 100u;
  constexpr std::uint64_t frame_b = 107u;
  constexpr std::uint64_t frame_c = 112u;
  std::array<std::uint32_t, 10> evidence {};

  // A noncanonical CutBridge history state clears owner word 0, but a receipt-authorized infer
  // postprocess still publishes words 1..7. Scene age consumes those words independently.
  ASSERT_TRUE(fixture.publish_history_owner(frame_a, 3u, error)) << error;
  ASSERT_TRUE(fixture.seed_scene_from_history_owner(frame_b, evidence, error)) << error;
  for (std::size_t index = 0u; index < 9u; ++index) {
    EXPECT_EQ(evidence[index], 0u) << "scene-evidence word " << index;
  }
  EXPECT_EQ(evidence[9], frame_b - frame_a);

  // If B actually inferred, its owner publication makes C's exact age C-B.
  ASSERT_TRUE(fixture.publish_history_owner(frame_b, 3u, error)) << error;
  ASSERT_TRUE(fixture.seed_scene_from_history_owner(frame_c, evidence, error)) << error;
  EXPECT_EQ(evidence[9], frame_c - frame_b);

  // If B reused, the owner stays A and C's exact age remains C-A.
  ASSERT_TRUE(fixture.publish_history_owner(frame_a, 3u, error)) << error;
  ASSERT_TRUE(fixture.seed_scene_from_history_owner(frame_c, evidence, error)) << error;
  EXPECT_EQ(evidence[9], frame_c - frame_a);

  // The uint64 owner subtraction handles a low-word rollover and saturates long gaps without
  // wrapping scene age back to a small value.
  constexpr std::uint64_t rollover_owner = 0x00000000fffffff0ull;
  constexpr std::uint64_t rollover_current = 0x0000000100000010ull;
  ASSERT_TRUE(fixture.publish_history_owner(rollover_owner, 3u, error)) << error;
  ASSERT_TRUE(fixture.seed_scene_from_history_owner(
    rollover_current, evidence, error
  )) << error;
  EXPECT_EQ(evidence[9], 32u);

  ASSERT_TRUE(fixture.publish_history_owner(1u, 3u, error)) << error;
  ASSERT_TRUE(fixture.seed_scene_from_history_owner(70000u, evidence, error)) << error;
  EXPECT_EQ(evidence[9], 65535u);
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
  std::array<std::uint32_t, 10> scene_evidence {};
  scene_evidence.fill(0xA5A5A5A5u);
  fixture.seed_scene_evidence(scene_evidence);
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
  std::array<std::uint32_t, 10> preserved_scene_evidence {};
  ASSERT_TRUE(fixture.read_scene_evidence(preserved_scene_evidence, error)) << error;
  EXPECT_EQ(preserved_scene_evidence, scene_evidence);

  ASSERT_TRUE(fixture.resolve_postprocess_args(
    models::near_identical_gpu_branch_e::reuse, true, decision, error
  )) << error;
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_one_x)], 0u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::reuse_grid16_x)], 7u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::reuse_grid16_y)], 6u);
  ASSERT_TRUE(fixture.read_scene_evidence(preserved_scene_evidence, error)) << error;
  EXPECT_EQ(preserved_scene_evidence, scene_evidence);

  ASSERT_TRUE(fixture.run_detector(input, input, history_state(1u), decision, error)) << error;
  ASSERT_TRUE(fixture.resolve_postprocess_args(
    models::near_identical_gpu_branch_e::infer, false, decision, error
  )) << error;
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::infer_one_x)], 0u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::reuse_grid16_x)], 7u);
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::reuse_grid16_y)], 6u);

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
  EXPECT_EQ(decision[word(models::near_identical_gpu_decision_word_e::reuse_grid16_y)], 6u);
  ASSERT_TRUE(fixture.read_scene_evidence(preserved_scene_evidence, error)) << error;
  EXPECT_EQ(preserved_scene_evidence, scene_evidence);
}

TEST(HostSbsNearIdenticalDetectorGpuTest, FusedModeSwitchesCompareForceCompareWithoutStaleEvidence) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;

  std::vector<std::array<float, 4>> source(field_texels);
  for (UINT y = 0u; y < field_height; ++y) {
    for (UINT x = 0u; x < field_width; ++x) {
      source[static_cast<std::size_t>(y) * field_width + x] = {
        static_cast<float>((13u * x + 7u * y) % 251u) / 250.0f,
        static_cast<float>((5u * x + 17u * y) % 241u) / 240.0f,
        static_cast<float>((19u * x + 3u * y) % 239u) / 238.0f,
        1.0f,
      };
    }
  }
  const auto previous = uniform_model_input(0.5f);
  std::vector<float> poisoned_previous(
    3u * field_texels, std::numeric_limits<float>::quiet_NaN()
  );
  constexpr std::array<std::uint32_t, 4u> source_region {
    0u, 0u, field_width, field_height,
  };
  constexpr models::depth_tensor_content_rect_t full_content {
    0u, 0u, field_width, field_height,
  };

  near_identical_gpu_fixture_t::preprocess_result_t compare_before;
  near_identical_gpu_fixture_t::preprocess_result_t canonical;
  near_identical_gpu_fixture_t::preprocess_result_t force;
  near_identical_gpu_fixture_t::preprocess_result_t compare_after;
  ASSERT_TRUE(fixture.run_preprocess_and_tile_evidence(
    true, source, field_width, field_height, source_region, full_content,
    previous, compare_before, error
  )) << error;
  ASSERT_TRUE(fixture.run_preprocess_and_tile_evidence(
    false, source, field_width, field_height, source_region, full_content,
    poisoned_previous, canonical, error
  )) << error;
  ASSERT_TRUE(fixture.run_preprocess_and_tile_evidence(
    true, source, field_width, field_height, source_region, full_content,
    poisoned_previous, force, error, false
  )) << error;
  ASSERT_TRUE(fixture.run_preprocess_and_tile_evidence(
    true, source, field_width, field_height, source_region, full_content,
    previous, compare_after, error
  )) << error;

  const auto expect_same_outputs = [&] (
                                     const auto &left,
                                     const auto &right
                                   ) {
    ASSERT_EQ(left.tensor.size(), right.tensor.size());
    EXPECT_EQ(
      std::memcmp(
        left.tensor.data(), right.tensor.data(),
        left.tensor.size() * sizeof(float)
      ),
      0
    );
    ASSERT_EQ(left.appearance.size(), right.appearance.size());
    EXPECT_EQ(
      std::memcmp(
        left.appearance.data(), right.appearance.data(),
        left.appearance.size() * sizeof(float)
      ),
      0
    );
    EXPECT_EQ(left.exclusion, right.exclusion);
  };
  expect_same_outputs(canonical, force);
  expect_same_outputs(compare_before, force);
  expect_same_outputs(compare_before, compare_after);
  for (const auto &tile : force.tiles) {
    EXPECT_EQ(tile, (tile_evidence_t {0u, 0u, 0u, 0u}));
  }
  EXPECT_EQ(compare_before.tiles, compare_after.tiles);
  EXPECT_TRUE(std::ranges::any_of(
    compare_before.tiles,
    [](const tile_evidence_t &tile) { return tile[0] != 0u; }
  ));
}

TEST(HostSbsNearIdenticalDetectorGpuTest, FusedPreprocessMatchesCanonicalAndSpecializedOracle) {
  near_identical_gpu_fixture_t fixture;
  std::string error;
  const auto initialized = initialize_fixture(fixture, error);
  if (initialized == near_identical_gpu_fixture_t::initialize_result_e::d3d_unavailable) {
    GTEST_SKIP() << error;
  }
  ASSERT_EQ(initialized, near_identical_gpu_fixture_t::initialize_result_e::ready) << error;

  constexpr UINT source_width = 144u;
  constexpr UINT source_height = 112u;
  std::vector<std::array<float, 4>> source(source_width * source_height);
  for (UINT y = 0u; y < source_height; ++y) {
    for (UINT x = 0u; x < source_width; ++x) {
      const auto index = y * source_width + x;
      source[index] = {
        static_cast<float>((x * 17u + y * 3u) % 251u) / 250.0f,
        static_cast<float>((x * 5u + y * 19u) % 241u) / 240.0f,
        static_cast<float>((x * 11u + y * 7u) % 239u) / 238.0f,
        1.0f,
      };
    }
  }
  auto previous = uniform_model_input(0.5f);
  add_red_pixel_delta(previous, 3u, 5u, 0.20f);
  add_red_pixel_delta(previous, 48u, 31u, 1.0f / 64.0f);
  previous[2u * field_texels + 73u * field_width + 91u] =
    std::numeric_limits<float>::quiet_NaN();

  struct parity_case_t {
    const char *label;
    std::array<std::uint32_t, 4> source_region;
    models::depth_tensor_content_rect_t tensor_content;
  };
  constexpr std::array cases {
    parity_case_t {
      "full texture",
      {0u, 0u, source_width, source_height},
      {0u, 0u, field_width, field_height},
    },
    parity_case_t {
      "translated source ROI",
      {16u, 8u, 16u + field_width, 8u + field_height},
      {0u, 0u, field_width, field_height},
    },
    parity_case_t {
      "translated ROI with small one-pass tensor padding",
      {16u, 8u, 16u + field_width, 8u + field_height},
      {2u, 1u, field_width - 2u, field_height - 1u},
    },
    parity_case_t {
      "translated ROI with specialized tensor padding",
      {16u, 8u, 16u + field_width, 8u + field_height},
      {7u, 5u, 104u, 86u},
    },
    parity_case_t {
      "invalid reset domain",
      {16u, 8u, 16u + field_width, 8u + field_height},
      {0u, 0u, 0u, 0u},
    },
  };

  ASSERT_TRUE(fixture.publish_history_owner(
    baseline_frame_id, 1u, error, owner_observation_timestamp_us
  )) << error;
  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.label);
    near_identical_gpu_fixture_t::preprocess_result_t reference;
    near_identical_gpu_fixture_t::preprocess_result_t fused;
    ASSERT_TRUE(fixture.run_preprocess_and_tile_evidence(
      false,
      source,
      source_width,
      source_height,
      test_case.source_region,
      test_case.tensor_content,
      previous,
      reference,
      error
    )) << error;
    ASSERT_TRUE(fixture.run_preprocess_and_tile_evidence(
      true,
      source,
      source_width,
      source_height,
      test_case.source_region,
      test_case.tensor_content,
      previous,
      fused,
      error
    )) << error;
    ASSERT_EQ(reference.tensor.size(), fused.tensor.size());
    EXPECT_EQ(
      std::memcmp(
        reference.tensor.data(), fused.tensor.data(),
        reference.tensor.size() * sizeof(float)
      ),
      0
    ) << "calibrated NCHW words changed";
    ASSERT_EQ(reference.appearance.size(), fused.appearance.size());
    EXPECT_EQ(
      std::memcmp(
        reference.appearance.data(), fused.appearance.data(),
        reference.appearance.size() * sizeof(float)
      ),
      0
    ) << "appearance ordinal words changed";
    EXPECT_EQ(reference.exclusion, fused.exclusion)
      << "tensor exclusion words changed";
    EXPECT_EQ(reference.tiles, fused.tiles) << "tile evidence changed";

    decision_words_t reference_proposal {};
    decision_words_t fused_proposal {};
    ASSERT_TRUE(fixture.resolve_seeded_tiles(
      reference.tiles, reference_proposal, error
    )) << error;
    ASSERT_TRUE(fixture.resolve_seeded_tiles(
      fused.tiles, fused_proposal, error
    )) << error;
    EXPECT_EQ(reference_proposal, fused_proposal) << "proposal changed";

    const auto branch = static_cast<models::near_identical_gpu_branch_e>(
      reference_proposal[word(models::near_identical_gpu_decision_word_e::decision)]
    );
    decision_words_t reference_receipt {};
    decision_words_t fused_receipt {};
    ASSERT_TRUE(fixture.resolve_seeded_tiles(
      reference.tiles, reference_receipt, error
    )) << error;
    ASSERT_TRUE(fixture.resolve_postprocess_args(
      branch, true, reference_receipt, error
    )) << error;
    ASSERT_TRUE(fixture.resolve_seeded_tiles(
      fused.tiles, fused_receipt, error
    )) << error;
    ASSERT_TRUE(fixture.resolve_postprocess_args(
      branch, true, fused_receipt, error
    )) << error;
    EXPECT_EQ(reference_receipt, fused_receipt) << "receipt/indirect args changed";

    decision_words_t reference_malformed {};
    decision_words_t fused_malformed {};
    ASSERT_TRUE(fixture.resolve_seeded_tiles(
      reference.tiles,
      reference_malformed,
      error,
      reduction_group_count,
      tile_group_width,
      tile_group_height,
      tile_group_count - 1u
    )) << error;
    ASSERT_TRUE(fixture.resolve_seeded_tiles(
      fused.tiles,
      fused_malformed,
      error,
      reduction_group_count,
      tile_group_width,
      tile_group_height,
      tile_group_count - 1u
    )) << error;
    EXPECT_EQ(reference_malformed, fused_malformed)
      << "malformed-shape fail-open transaction changed";
    EXPECT_EQ(
      reference_malformed[word(models::near_identical_gpu_decision_word_e::decision)],
      static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
    );
  }

  // A malformed retained-source rectangle is outside normal host admission. Canonical
  // preprocessing marks every tensor texel excluded; the fused detector must consume that
  // committed exclusion and fail open to inference rather than comparing its zero fill.
  near_identical_gpu_fixture_t::preprocess_result_t malformed_source;
  ASSERT_TRUE(fixture.run_preprocess_and_tile_evidence(
    true,
    source,
    source_width,
    source_height,
    {source_width - 1u, 0u, source_width + 1u, source_height},
    {0u, 0u, field_width, field_height},
    previous,
    malformed_source,
    error
  )) << error;
  EXPECT_TRUE(std::ranges::all_of(
    malformed_source.exclusion, [](const std::uint32_t value) { return value == 1u; }
  ));
  for (const auto &tile : malformed_source.tiles) {
    EXPECT_EQ(tile, (tile_evidence_t {0u, 0u, 0u, 0u}));
  }
  decision_words_t malformed_source_proposal {};
  ASSERT_TRUE(fixture.resolve_seeded_tiles(
    malformed_source.tiles, malformed_source_proposal, error
  )) << error;
  EXPECT_EQ(
    malformed_source_proposal[word(models::near_identical_gpu_decision_word_e::decision)],
    static_cast<std::uint32_t>(models::near_identical_gpu_branch_e::infer)
  );
}

#else

TEST(HostSbsNearIdenticalDetectorGpuTest, WindowsOnly) {
  GTEST_SKIP() << "D3D11 WARP is Windows-only";
}

#endif
