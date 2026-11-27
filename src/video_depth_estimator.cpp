#include "video_depth_estimator.h"
#include "live_model_preparation.h"

#ifdef SUNSHINE_TESTS
  #include "live_model_preparation_test.h"
#endif

#include "cuda_conditional_graph.h"
#include "cuda_driver_api.h"
#include "depth_coordinate_v2.h"
#include "generated/sbs_adaptive_state_contract.h"
#include "host_sbs_gpu_outcomes.h"
#include "host_sbs_eye_field.h"
#include "host_sbs_gpu_trace.h"
#include "host_sbs_shader_cache.h"
#include "host_sbs_depth_gpu_timing.h"
#include "host_sbs_direct_glyph_route.h"
#include "host_sbs_publication_readback.h"
#include "host_sbs_subtitle_publication.h"
#include "host_sbs_v2_gpu_executor.h"
#include "logging.h"
#include "model_manager.h"
#include "platform/common.h"
#include "platform/windows/misc.h"
#include "platform/windows/utils.h"
#include "prod_zipdepth_convex2x.h"
#include "sbs_perf.h"
#include "tensorrt_engine_pool.h"
#include "utility.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <NvInfer.h>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <windows.h>

using namespace std::literals;

static std::atomic<std::uint64_t> g_publication_generation {1u};

// CUDA's CU_EVENT_DISABLE_TIMING flag. The local driver shim intentionally exposes only the
// small ABI surface this translation unit needs, so keep the readiness-only flag local too.
static constexpr unsigned int cuda_event_disable_timing = 0x2u;

namespace models {
  using namespace tensorrt_engine_pool;

  namespace {

    static_assert(near_identical_decision_cookie == cuda_conditional_graph::decision_cookie);
    static_assert(near_identical_token_low_cookie == cuda_conditional_graph::token_low_cookie);
    static_assert(near_identical_token_high_cookie == cuda_conditional_graph::token_high_cookie);
    static_assert(near_identical_proposal_magic == cuda_conditional_graph::proposal_magic);
    static_assert(near_identical_request_magic == cuda_conditional_graph::request_magic);
    static_assert(near_identical_receipt_magic == cuda_conditional_graph::receipt_magic);
    static_assert(near_identical_optional_receipt_magic ==
                  cuda_conditional_graph::optional_ocr_receipt_magic);
    static_assert(near_identical_work_flags_cookie ==
                  cuda_conditional_graph::work_flags_cookie);
    static_assert(near_identical_work_optional_ocr ==
                  cuda_conditional_graph::work_flags_value(
                    cuda_conditional_graph::work_flag_e::optional_ocr
                  ));
    static_assert(near_identical_work_subtitle_observation ==
                  cuda_conditional_graph::work_flags_value(
                    cuda_conditional_graph::work_flag_e::subtitle_observation
                  ));
    static_assert(near_identical_gpu_decision_record_byte_offset == 0u);
    static_assert(near_identical_gpu_request_record_byte_offset ==
                  sizeof(cuda_conditional_graph::decision_record_t));

    [[nodiscard]] std::array<std::uint32_t, near_identical_gpu_decision_word_count>
    initialize_near_identical_transaction(
      const std::uint64_t token,
      const cuda_conditional_graph::work_flag_e subtitle_work
    ) noexcept {
      // Clear every old proposal, receipt and indirect argument before either admission path
      // publishes this frame's request. Only force admission supplies a proposal on the CPU.
      std::array<std::uint32_t, near_identical_gpu_decision_word_count> transaction {};
      const auto request = cuda_conditional_graph::make_request(token, subtitle_work);
      std::memcpy(
        reinterpret_cast<std::byte *>(transaction.data()) +
          near_identical_gpu_request_record_byte_offset,
        &request,
        sizeof(request)
      );
      return transaction;
    }

    [[nodiscard]] constexpr cuda_conditional_graph::work_flag_e
    subtitle_transaction_work(
      const depth_optional_work_mode_e requested,
      const bool ocr_ready
    ) noexcept {
      switch (requested) {
        case depth_optional_work_mode_e::ordinary:
          return ocr_ready ?
                   cuda_conditional_graph::work_flag_e::optional_ocr :
                   cuda_conditional_graph::work_flag_e::subtitle_observation;
        case depth_optional_work_mode_e::suppress_subtitle:
        case depth_optional_work_mode_e::disable_subtitle:
          return cuda_conditional_graph::work_flag_e::none;
      }
      return cuda_conditional_graph::work_flag_e::none;
    }

    static_assert(fit_subtitle_analysis_geometry(
                    1920u, 1080u, {770, 434}).roi_top == 325u);
    static_assert(fit_subtitle_analysis_geometry(
                    1920u, 1080u, {770, 434}).roi_bottom == 430u);
    static_assert(fit_subtitle_analysis_geometry(
                    2560u, 1080u, {1022, 434}).roi_top == 289u);
    static_assert(fit_subtitle_analysis_geometry(
                    3440u, 1440u, {1036, 434}).roi_top == 287u);
    static_assert(fit_subtitle_analysis_geometry(
                    1080u, 1920u, {434, 770}).roi_top == 709u);
    static_assert(fit_subtitle_analysis_geometry(
                    1080u, 1920u, {434, 770}).roi_bottom == 768u);
    static_assert(host_sbs_v2_depth_shape_is_authenticated({770, 434}));
    static_assert(host_sbs_v2_depth_shape_is_authenticated({1022, 434}));
    static_assert(host_sbs_v2_depth_shape_is_authenticated({1036, 434}));
    static_assert(host_sbs_v2_depth_shape_is_authenticated({434, 770}));
    static_assert(host_sbs_v2_depth_shape_is_authenticated({434, 1022}));
    static_assert(host_sbs_v2_depth_shape_is_authenticated({434, 1036}));
    static_assert(!host_sbs_v2_depth_shape_is_authenticated({1008, 434}));

    bool r32_float_texture_srv_matches(
      ID3D11ShaderResourceView *view,
      const std::uint32_t width,
      const std::uint32_t height,
      ID3D11Resource **identity = nullptr
    ) {
      if (!view || width == 0u || height == 0u) {
        return false;
      }
      Microsoft::WRL::ComPtr<ID3D11Resource> resource;
      view->GetResource(&resource);
      Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
      if (!resource || FAILED(resource.As(&texture)) || !texture) {
        return false;
      }
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      D3D11_SHADER_RESOURCE_VIEW_DESC view_desc {};
      view->GetDesc(&view_desc);
      const bool valid =
        desc.Width == width && desc.Height == height &&
        desc.MipLevels == 1u && desc.ArraySize == 1u &&
        desc.Format == DXGI_FORMAT_R32_FLOAT &&
        desc.SampleDesc.Count == 1u && desc.SampleDesc.Quality == 0u &&
        view_desc.Format == DXGI_FORMAT_R32_FLOAT &&
        view_desc.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2D &&
        view_desc.Texture2D.MostDetailedMip == 0u &&
        view_desc.Texture2D.MipLevels == 1u;
      if (valid && identity) {
        *identity = resource.Detach();
      }
      return valid;
    }

  }  // namespace

  bool parallax_v2_result_is_authenticated(const depth_observation_views_t &result) {
    const bool expects_base_diagnostics = !result.base_diagnostics_omitted;
    if (!result.completed_frame_valid || result.completed_frame_id == 0u ||
        !result.parallax_v2_producer_active ||
        static_cast<bool>(result.shadow_candidate_parallax) != expects_base_diagnostics ||
        static_cast<bool>(result.shadow_vertical_majorant) != expects_base_diagnostics ||
        static_cast<bool>(result.shadow_vertical_conditioned) != expects_base_diagnostics ||
        !result.shadow_base_final_parallax ||
        (result.subtitle_conditioning_diagnostics_current &&
         !result.shadow_conditioned_final_parallax) ||
        !result.shadow_state || !result.shadow_frame_stats ||
        !result.raw_model_provenance || !result.composite_depth_runtime_provenance ||
        !result.parallax_v2_shader_provenance ||
        result.raw_width <= 0 || result.raw_height <= 0 ||
        result.field_width <= 0 || result.field_height <= 0) {
      return false;
    }

    const auto &input_region = result.input_region;
    const auto calibrated_dav2_shape = fit_host_sbs_v2_depth_tensor_shape(
      input_region.source_width,
      input_region.source_height
    );
    const auto fused_shape = host_sbs_convex2x_field_shape(calibrated_dav2_shape);
    const depth_tensor_shape_t result_shape {result.raw_width, result.raw_height};
    const depth_tensor_shape_t field_shape {result.field_width, result.field_height};
    if (!input_region.valid() ||
        !host_sbs_v2_source_resolution_is_supported(
          input_region.source_width, input_region.source_height) ||
        !fused_shape || *fused_shape != result_shape ||
        !input_region.tensor_content.valid(result_shape)) {
      return false;
    }
    if (input_region.is_video_region()) {
      const auto expected = plan_host_sbs_v2_video_region(
        {input_region.left, input_region.top, input_region.right, input_region.bottom},
        input_region.source_width,
        input_region.source_height,
        calibrated_dav2_shape
      );
      const auto expected_content = expected ?
                                      host_sbs_convex2x_content_rect(
                                        expected->tensor_content,
                                        calibrated_dav2_shape
                                      ) : std::nullopt;
      if (!expected || expected->source_rect != depth_source_rect_t {
            input_region.left, input_region.top, input_region.right, input_region.bottom
          } || !expected_content || *expected_content != input_region.tensor_content) {
        return false;
      }
    } else if (!input_region.tensor_content.full(result_shape)) {
      return false;
    }
    if (field_shape != result_shape ||
        result.field_content != input_region.tensor_content) {
      return false;
    }

    std::array<Microsoft::WRL::ComPtr<ID3D11Resource>, 6u> field_resources;
    const std::array<ID3D11ShaderResourceView *, 5u> field_views {
      result.shadow_candidate_parallax.Get(),
      result.shadow_vertical_majorant.Get(),
      result.shadow_vertical_conditioned.Get(),
      result.shadow_base_final_parallax.Get(),
      result.shadow_conditioned_final_parallax.Get(),
    };
    for (std::size_t index = 0u; index < field_views.size(); ++index) {
      if (!field_views[index]) {
        // The completeness check above permits absent diagnostics.
        continue;
      }
      ID3D11Resource *resource = nullptr;
      if (!r32_float_texture_srv_matches(
            field_views[index],
            static_cast<std::uint32_t>(result.field_width),
            static_cast<std::uint32_t>(result.field_height),
            &resource
          )) {
        return false;
      }
      field_resources[index].Attach(resource);
      for (std::size_t previous = 0u; previous < index; ++previous) {
        if (field_resources[previous].Get() == field_resources[index].Get()) {
          return false;
        }
      }
    }
    if (result.shadow_coordinate) {
      ID3D11Resource *resource = nullptr;
      if (!r32_float_texture_srv_matches(
            result.shadow_coordinate.Get(),
            static_cast<std::uint32_t>(result.field_width),
            static_cast<std::uint32_t>(result.field_height),
            &resource
          )) {
        return false;
      }
      field_resources.back().Attach(resource);
      for (std::size_t previous = 0u; previous + 1u < field_resources.size(); ++previous) {
        if (field_resources[previous].Get() == field_resources.back().Get()) {
          return false;
        }
      }
    }
    if (!result.ocr_box_record || !result.subtitle_locator_state) {
      return false;
    }

    if (!parallax_v2_shader_provenance_matches_current_contract(
          *result.parallax_v2_shader_provenance
        )) {
      return false;
    }

    const auto &model = *result.raw_model_provenance;
    const auto *calibration = depth_coordinate_v2::find_capture_calibration(
      model.depth_model,
      model.depth_model_url,
      model.onnx_sha256,
      model.preprocess_profile,
      model.preprocess_source_closure_sha256,
      static_cast<std::uint32_t>(calibrated_dav2_shape.width),
      static_cast<std::uint32_t>(calibrated_dav2_shape.height)
    );
    if (!calibration ||
        !depth_coordinate_v2::joint_plane_mode_word_is_valid(result.parallax_v2_joint_plane_mode) ||
        std::abs(result.parallax_v2_raw_coordinate_scale -
                 calibration->raw_coordinate_scale) > 1.0e-6f ||
        !depth_coordinate_v2::parallax_runtime_constants_are_valid(
          result.parallax_v2_raw_coordinate_scale,
          result.parallax_v2_requested_pop_strength,
          result.parallax_v2_requested_gain
        )) {
      return false;
    }
    const auto &composite = *result.composite_depth_runtime_provenance;
    // These names are fixed by the compiled contract, not by the completed frame. Keep their
    // storage while retaining every resource, provenance, and owner check on each result.
    static const std::string manifest_name =
      std::string {prod_zipdepth_convex2x::logical_model} +
      ".active-engine.json";
    if (composite.model != prod_zipdepth_convex2x::logical_model ||
        composite.onnx_sha256 != prod_zipdepth_convex2x::fused_onnx_sha256 ||
        composite.embedded_dav2_onnx_sha256 !=
          prod_zipdepth_convex2x::dav2_onnx_sha256 ||
        composite.embedded_dav2_onnx_sha256 != model.onnx_sha256 ||
        composite.zipdepth_checkpoint_sha256 !=
          prod_zipdepth_convex2x::zipdepth_checkpoint_sha256 ||
        composite.guidance_preprocess_source_closure_sha256 !=
          model.preprocess_source_closure_sha256 ||
        composite.engine_recipe != prod_zipdepth_convex2x::engine_recipe ||
        !is_current_depth_engine_filename(composite.engine_artifact) ||
        composite.active_engine_manifest != manifest_name ||
        result.guidance_width != result_shape.width ||
        result.guidance_height != result_shape.height ||
        result.refined_width != result_shape.width ||
        result.refined_height != result_shape.height ||
        static_cast<bool>(result.guidance_model_input_snapshot) !=
          static_cast<bool>(result.refined_model_depth_snapshot) ||
        (result.guidance_model_input_snapshot &&
         result.guidance_model_input_snapshot.Get() != result.model_input_snapshot.Get()) ||
        (result.refined_model_depth_snapshot &&
         result.refined_model_depth_snapshot.Get() != result.raw_model_depth_snapshot.Get())) {
      return false;
    }
    return true;
  }

  namespace {
    namespace publication_receipt = host_sbs_gpu_completion_receipt;
    using host_sbs_publication_readback::inference_completion_notification_t;
    using host_sbs_publication_readback::publication_readback_t;
  }  // namespace

  struct video_depth_estimator::impl {
    static constexpr std::uint32_t ocr_grid_width = ocr_engine_width / 8;
    static constexpr std::uint32_t ocr_grid_height = ocr_engine_height;
    static constexpr std::uint32_t ocr_cells_per_group = 32;
    static constexpr std::uint32_t ocr_cell_rows_per_group = 4;
    static constexpr std::uint32_t ocr_cell_group_count =
      (ocr_grid_width + ocr_cells_per_group - 1u) / ocr_cells_per_group;
    static constexpr std::uint32_t ocr_cell_group_row_count =
      (ocr_grid_height + ocr_cell_rows_per_group - 1u) / ocr_cell_rows_per_group;
    static constexpr std::uint32_t ocr_cell_words = 1;
    static constexpr std::uint32_t ocr_cell_stats_word_count =
      ocr_grid_width * ocr_grid_height * ocr_cell_words;
    static constexpr std::uint32_t ocr_box_record_word_count =
      depth_coordinate_v2::subtitle_ocr_record_word_count;
    static constexpr std::uint32_t ocr_box_record_schema =
      depth_coordinate_v2::subtitle_ocr_record_schema;
    static constexpr std::uint32_t ocr_box_record_tag =
      depth_coordinate_v2::subtitle_ocr_record_tag;
    static constexpr std::uint32_t subtitle_locator_state_word_count =
      depth_coordinate_v2::subtitle_plane_state_word_count;
    static constexpr std::uint32_t subtitle_condition_param_word_count =
      depth_coordinate_v2::subtitle_condition_param_word_count;
    using near_identical_constants_t = std::array<std::uint32_t, 20u>;
    using ocr_resolve_constants_t = std::array<std::uint32_t, 24u>;
    static_assert(sizeof(near_identical_constants_t) == 80u);
    static_assert(sizeof(near_identical_constants_t) % 16u == 0u);
    static_assert(sizeof(ocr_resolve_constants_t) == 96u);
    static_assert(sizeof(ocr_resolve_constants_t) % 16u == 0u);
    static_assert(depth_coordinate_v2::subtitle_ocr_local_candidate_word_count % 4u == 0u);
    static_assert(
      host_sbs_gpu_trace::transaction_word_count == near_identical_gpu_decision_word_count
    );
    static_assert(
      host_sbs_gpu_trace::subtitle_locator_word_count == subtitle_locator_state_word_count
    );
    static_assert(
      host_sbs_gpu_trace::subtitle_condition_word_count == subtitle_condition_param_word_count
    );
    static_assert(ocr_grid_width * 8u ==
                  depth_coordinate_v2::subtitle_ocr_output_width);
    static_assert(ocr_grid_height ==
                  depth_coordinate_v2::subtitle_ocr_output_height);

    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
    const std::filesystem::path shader_root;

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *exec_context = nullptr;
    models::prod_zipdepth_convex2x::engine_io_e depth_engine_io_kind =
      models::prod_zipdepth_convex2x::engine_io_e::invalid;
    CUcontext cuda_ctx = nullptr;
    CUstream cu_stream = nullptr;
    CUdevice cuda_device = -1;
    std::string engine_key;
    // PP-OCR owns a distinct mutable TensorRT context. Its captured graph is a sibling IF node in
    // the one joined CUDA root, so DAV2 and OCR may overlap without a second host-managed stream.
    nvinfer1::ICudaEngine *ocr_engine = nullptr;
    nvinfer1::IExecutionContext *ocr_exec_context = nullptr;
    std::string ocr_engine_key;
    bool ocr_context_pooled = false;
    bool ocr_context_warmed = false;
    detail::warmed_execution_context_health_t ocr_context_health;
    bool ocr_available = false;

    float ema_alpha;
    float ema_edge_change;
    float ema_edge_gradient;
    float ema_edge_strength;
    int depth_short_side;  // depth map short-side resolution (clamped to native short side)
    float max_aspect;  // aspect cap for short-side mode
    float minmax_alpha;  // temporal EMA blend for the normalized min/max
    const bool diagnostics_enabled;
    const bool normalized_depth_diagnostics_enabled;
    const subtitle_legacy_matte_e subtitle_legacy_matte;
    struct tensorrt_cuda_graph_t {
      CUgraph graph = nullptr;
      detail::cuda_graph_replay_policy_t policy;
    };

    tensorrt_cuda_graph_t depth_inference_graph;
    cuda_conditional_graph::executable_t depth_conditional_graph;
    CUdeviceptr depth_conditional_decision_ptr = 0u;
    CUdeviceptr depth_conditional_request_ptr = 0u;
    CUgraph depth_conditional_child_graph = nullptr;
    CUgraph depth_conditional_optional_child_graph = nullptr;
    bool gpu_conditional_bridge_available = true;
    bool gpu_conditional_bridge_error_logged = false;
    bool gpu_conditional_bridge_context_failed = false;
    // OCR owns a distinct TensorRT context and graph. The graph captures only enqueueV3; D3D/CUDA
    // interop map/unmap and mutable tensor bindings always remain outside the capture boundary.
    // Depth retains only the captured source CUgraph. Every authoritative depth launch uses the
    // conditional wrapper; there is deliberately no standalone raw depth CUgraphExec.
    tensorrt_cuda_graph_t ocr_inference_graph;

    enum class lifecycle_state_e : std::uint8_t {
      unavailable,
      operational,
      terminal,
    };
    lifecycle_state_e lifecycle_state = lifecycle_state_e::unavailable;

    [[nodiscard]] bool is_operational() const noexcept {
      return lifecycle_state == lifecycle_state_e::operational;
    }

    [[nodiscard]] bool is_terminal() const noexcept {
      return lifecycle_state == lifecycle_state_e::terminal;
    }
    float parallax_v2_raw_coordinate_scale = 0.0f;
    const float parallax_v2_requested_pop_strength;
    const float parallax_v2_requested_gain;
    static constexpr std::uint32_t joint_plane_mode = depth_coordinate_v2::adaptive_policy_id;
    std::shared_ptr<const raw_model_provenance_t> raw_model_provenance;
    std::shared_ptr<const composite_depth_runtime_provenance_t>
      composite_depth_runtime_provenance;
    std::shared_ptr<const parallax_v2_shader_provenance_t>
      parallax_v2_shader_provenance;
    bool parallax_v2_producer_active = false;

    // Debug consumers read this CPU cache; the publication receipt owns the only routine
    // staging/query ring. Its original copy time and transaction identity remain unchanged.
    depth_telemetry_poll_result cached_depth_telemetry;
    std::uint64_t latest_diagnostic_sequence = 0u;

    // Throughput telemetry for the permanent stream-cadence matched-frame pipeline.
    std::chrono::steady_clock::time_point throughput_stats_start {};
    unsigned throughput_stats_calls = 0;
    unsigned throughput_stats_busy_drops = 0;
    unsigned throughput_stats_enqueues = 0;
    unsigned throughput_stats_force_infer_enqueues = 0;
    unsigned throughput_stats_gpu_undecided_enqueues = 0;
    unsigned throughput_stats_completions = 0;
    unsigned throughput_stats_subtitle_suppressed = 0;
    unsigned throughput_stats_ocr_armed = 0;

    // GPU-stream timing of the async TensorRT enqueues (diagnostics only), resolved lazily and
    // pushed to sbs_perf. The ring's CUDA calls run on the estimator thread with cuda_ctx
    // current, like the rest of estimate(); each failure it returns goes through
    // observe_timing_event_failure. No-ops entirely when diagnostics are off.
    host_sbs_depth_gpu_timing::cuda_event_ring_t perf_depth_conditional;  // GPU arbitration plus either infer or reuse
    cuda_conditional_graph::branch_timing_pending_t pending_branch_timing;
    std::shared_ptr<host_sbs_telemetry::collector> telemetry_performance;

    // D3D11 timing for the work around TensorRT. CUDA events above deliberately measure only
    // the inference enqueue; these timestamp queries expose the resize/normalization input pass
    // and the depth normalization/EMA passes without ever synchronizing the CPU. A ring
    // is required because query results commonly become available several source frames later.
    using d3d_perf_slot = host_sbs_depth_gpu_timing::d3d_slot_t;
    host_sbs_depth_gpu_timing::d3d_ring_t d3d_perf;

    void initialize_d3d_perf() {
      d3d_perf.initialize(device.Get(), diagnostics_enabled);
    }

    d3d_perf_slot *begin_d3d_perf(bool has_post, bool has_pre) {
      return d3d_perf.begin(context.Get(), telemetry_performance, has_post, has_pre);
    }

    void mark_d3d_post_end(d3d_perf_slot *slot) {
      if (slot) {
        context->End(slot->post_end.Get());
      }
    }

    void mark_d3d_pre_start(d3d_perf_slot *slot) {
      if (slot) {
        context->End(slot->pre_start.Get());
      }
    }

    void end_d3d_perf(d3d_perf_slot *slot) {
      if (!slot) {
        return;
      }
      context->End(slot->pre_end.Get());
      context->End(slot->disjoint.Get());
      slot->pending = true;
    }

    static bool decode_telemetry_words(
      const sbs_adaptive_state::words_t &words,
      int depth_width,
      int depth_height,
      std::uint64_t sampled_frame_id,
      std::chrono::steady_clock::time_point sampled_at,
      depth_telemetry_sample &sample
    ) {
      using sbs_adaptive_state::word_e;
      const auto scalar = [&](const word_e word) {
        return std::bit_cast<float>(words[sbs_adaptive_state::index(word)]);
      };
      if (!publication_receipt::valid_depth_health_words(words)) {
        return false;
      }
      sample.depth_width = depth_width;
      sample.depth_height = depth_height;
      // Protocol-13 telemetry retains these pre-V2 columns for client wire compatibility. The
      // V2 state has no adaptive-pop, subject or zero-anchor authority, so publish
      // explicit unavailable/default values instead of decoding reserved GPU words.
      sample.adaptive_pop_ratio = 1.0f;
      sample.edge_fraction = -1.0f;
      sample.valid_depth_fraction = scalar(word_e::valid_depth_fraction);
      sample.effective_range_width = scalar(word_e::effective_raw_range_width);
      sample.current_edge_fraction = -1.0f;
      sample.current_zero_anchor_candidate_shift_px = -1.0f;
      sample.zero_anchor_shift_px = 0.0f;
      sample.subject_depth = 0.0f;
      sample.empty_raw_count =
        words[sbs_adaptive_state::index(word_e::empty_raw_count)];
      sample.collapsed_raw_count =
        words[sbs_adaptive_state::index(word_e::collapsed_raw_count)];
      sample.sampled_frame_id = sampled_frame_id;
      sample.sampled_at = sampled_at;
      sample.anchor_valid = false;
      sample.range_collapsed = scalar(word_e::range_collapsed) > 0.5f;
      sample.depth_ready = scalar(word_e::depth_ready) > 0.5f;
      // The first valid depth frame initializes the normalization profile.
      sample.profile_initialized = sample.depth_ready;
      return true;
    }

    // Optional timing calls get the joined CUDA failure policy of the stream they measure.
    void observe_timing_event_failure(const host_sbs_depth_gpu_timing::cuda_event_failure_t &failure) {
      (void) observe_joined_cuda_failure(failure.operation, failure.result, false);
    }

    void perf_try_resolve(int slot, cuda_driver_api &cuda) {
      const auto resolved = perf_depth_conditional.try_resolve(slot, cuda);
      if (resolved.failure) {
        observe_timing_event_failure(*resolved.failure);
        return;
      }
      if (!resolved.elapsed_ms) {
        return;
      }
      const float ms = *resolved.elapsed_ms;
      sbs_perf::add_sample_ms("depth_conditional_transaction", ms);
      if (telemetry_performance) {
        telemetry_performance->record(host_sbs_telemetry::stage::conditional_transaction, ms, std::chrono::steady_clock::now());
      }
    }

    void resolve_completed_branch_timing() {
      if (!pending_root) {
        return;
      }
      // The callers proved this exact root complete. Branch events can still name an earlier launch.
      const auto generation = pending_branch_timing.consume(
        pending_root->transaction_token, true
      );
      if (!generation || *generation != sbs_perf::generation()) {
        return;
      }
      cuda_conditional_graph::branch_completion_timing_t timing;
      const auto result = depth_conditional_graph.read_branch_completion_timing(timing, true);
      if (result != CUDA_SUCCESS) {
        (void) observe_joined_cuda_failure(
          "diagnostic sibling branch completion elapsed query", result, false
        );
        return;
      }
      sbs_perf::add_sample_ms_if_current(
        "depth_conditional_depth_branch_gpu", timing.depth_ms, *generation
      );
      if (timing.has_optional) {
        sbs_perf::add_sample_ms_if_current(
          "depth_conditional_ocr_branch_gpu", timing.optional_ms, *generation
        );
      }
    }

    void perf_drain() {
      auto &cuda = cuda_driver_api::get();
      for (int i = 0; i < host_sbs_depth_gpu_timing::cuda_event_ring_t::slot_count; i++) {
        perf_try_resolve(i, cuda);
        if (is_terminal()) {
          break;
        }
      }
    }

    // Record a start event before an enqueue; returns the ring slot (or -1 to skip timing).
    int perf_begin(CUstream stream) {
      if (!diagnostics_enabled) {
        return -1;
      }
      auto &cuda = cuda_driver_api::get();
      if (!host_sbs_depth_gpu_timing::cuda_event_ring_t::recording_available(cuda)) {
        return -1;
      }
      const int slot = perf_depth_conditional.next_slot();
      perf_try_resolve(slot, cuda);  // reclaim the slot if its prior sample is ready
      if (is_terminal() || perf_depth_conditional.in_flight(slot)) {
        return -1;  // terminal, or still in flight: do not start another measurement
      }
      if (const auto failure = perf_depth_conditional.record_start(slot, stream, cuda)) {
        observe_timing_event_failure(*failure);
        return -1;
      }
      return slot;
    }

    // Record the stop event after the enqueue and mark the slot pending.
    void perf_end(int slot, CUstream stream) {
      if (slot < 0) {
        return;
      }
      if (const auto failure = perf_depth_conditional.record_stop(slot, stream, cuda_driver_api::get())) {
        observe_timing_event_failure(*failure);
      }
    }

    [[nodiscard]] bool reset_depth_conditional_graph() noexcept {
      pending_branch_timing.clear();
      const bool reset_ok = depth_conditional_graph.reset();
      const bool wrapper_released = depth_conditional_graph.empty();
      if (!reset_ok) {
        // A failed restore after successful destruction no longer owns dependencies, but it still
        // proves the caller's CUDA-context state is unsafe for reuse. A failure before destruction
        // additionally leaves wrapper_released false, which makes every caller retain backing.
        execution_context_poisoned = true;
        gpu_conditional_bridge_context_failed = true;
        if (depth_conditional_optional_child_graph) {
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::unsafe_teardown
          );
        }
        mark_terminal_failure(true);
      }
      if (!wrapper_released) {
        // The wrapper still owns nodes embedded from the TensorRT source graph and captures the
        // mapped decision pointer. Losing either dependency here would be a use-after-free in the
        // CUDA driver. Quarantine the context and retain every dependency until terminal teardown.
        execution_context_poisoned = true;
        if (depth_conditional_optional_child_graph) {
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::unsafe_teardown
          );
        }
        mark_terminal_failure(true);
        return false;
      }
      depth_conditional_decision_ptr = 0u;
      depth_conditional_request_ptr = 0u;
      depth_conditional_child_graph = nullptr;
      depth_conditional_optional_child_graph = nullptr;
      // Even when every wrapper handle was destroyed, a failed context restore is a CUDA teardown
      // failure. Callers must stop before unregistering or destroying the wrapper's dependencies.
      return reset_ok;
    }

    [[nodiscard]] bool destroy_inference_graph(
      cuda_driver_api &cuda,
      tensorrt_cuda_graph_t &state
    ) {
      // The wrapper owns node-cloned children embedded from both captured TensorRT source graphs.
      // It must die before either source is replaced after a signature transition. If wrapper
      // teardown cannot release those clones, retain the source graph/context/interops rather than
      // creating a dangling embedded child.
      const bool wrapper_owns_state =
        (&state == &depth_inference_graph &&
         depth_conditional_child_graph == state.graph) ||
        (&state == &ocr_inference_graph &&
         depth_conditional_optional_child_graph == state.graph);
      if (wrapper_owns_state) {
        if (!reset_depth_conditional_graph()) {
          return false;
        }
      }
      if (state.graph) {
        if (!cuda.cuGraphDestroy ||
            cuda.cuGraphDestroy(state.graph) != CUDA_SUCCESS) {
          mark_shared_execution_failure(ocr_exec_context != nullptr);
          return false;
        }
        state.graph = nullptr;
      }
      state.policy.signature_warmed = false;
      return true;
    }

    // D3D interop is allowed to return different device pointers after any map. TensorRT graph
    // nodes retain the context's addresses (and depth's dynamic shape), so a graph is reusable
    // only for one exact signature. OCR calls this before rebinding; the common enqueue path also
    // calls it defensively so the depth and OCR lifecycle cannot drift apart.
    [[nodiscard]] bool select_inference_graph_signature(
      cuda_driver_api &cuda,
      tensorrt_cuda_graph_t &state,
      const detail::cuda_graph_signature_t &signature
    ) {
      if (!detail::select_cuda_graph_signature(state.policy, signature)) {
        return true;
      }
      return destroy_inference_graph(cuda, state);
    }

    bool prepare_depth_inference_graph(
      cuda_driver_api &cuda,
      const bool allow_private_bootstrap
    ) {
      const bool graph_api = cuda.cuStreamBeginCapture &&
                             cuda.cuStreamEndCapture && cuda.cuGraphDestroy;
      if (!graph_api || depth_inference_graph.policy.capture_failed) {
        fail_gpu_conditional_bridge_once(
          !graph_api ?
            "required DAV2 graph-capture APIs are unavailable" :
            "DAV2 graph capture previously failed"
        );
        return false;
      }

      // The quiescent pre-binding path selected this exact signature before mutating TensorRT
      // addresses. This helper owns only the private warm/capture bootstrap for that selection.
      if (depth_inference_graph.graph) {
        return true;
      }
      if (!allow_private_bootstrap) {
        fail_gpu_conditional_bridge_once(
          "DAV2 graph signature changed during a GPU-undecided transaction"
        );
        return false;
      }

      // TensorRT requires one ordinary enqueue after a dynamic-shape/address signature change
      // before that exact signature may be captured. This is a private bootstrap only: the raw
      // result is immediately overwritten by the wrapper's force-infer launch on the same stream
      // and is never exposed as a pending or rendered completion.
      if (!depth_inference_graph.policy.signature_warmed) {
        if (!exec_context->enqueueV3(cu_stream)) {
          depth_inference_graph.policy.capture_failed = true;
          fail_gpu_conditional_bridge_once(
            "private DAV2 signature bootstrap enqueue failed",
            CUDA_SUCCESS,
            true
          );
          return false;
        }
        joined_stream_work_ever_submitted = true;
        depth_inference_graph.policy.signature_warmed = true;
      }

      CUgraph captured = nullptr;
      const CUresult begin = cuda.cuStreamBeginCapture(
        cu_stream,
        CU_STREAM_CAPTURE_MODE_RELAXED
      );
      const bool captured_enqueue =
        begin == CUDA_SUCCESS && exec_context->enqueueV3(cu_stream);
      const CUresult end = begin == CUDA_SUCCESS ?
                             cuda.cuStreamEndCapture(cu_stream, &captured) :
                             begin;
      if (captured_enqueue && end == CUDA_SUCCESS && captured) {
        depth_inference_graph.graph = captured;
        BOOST_LOG(info)
          << "DAV2 private signature bootstrap captured " << target_w << 'x' << target_h
          << "; authoritative launches now require the GPU conditional wrapper.";
        return true;
      }

      depth_inference_graph.graph = captured;
      (void) destroy_inference_graph(cuda, depth_inference_graph);
      depth_inference_graph.policy.capture_failed = true;
      fail_gpu_conditional_bridge_once(
        "private DAV2 graph capture failed (begin=" + std::to_string(begin) +
          ", enqueue=" + std::to_string(captured_enqueue) +
          ", end=" + std::to_string(end) + ')',
        end,
        true
      );
      return false;
    }

    // Capture only the OCR TensorRT source graph. It is never instantiated or launched raw:
    // build() embeds it under the authenticated optional-infer handle beside DAV2, and one root
    // event therefore joins both isolated execution contexts without a branch readback.
    bool prepare_ocr_inference_graph(
      cuda_driver_api &cuda,
      const bool allow_private_bootstrap
    ) {
      if (!ocr_available) {
        return true;
      }
      const bool graph_api = cuda.cuStreamBeginCapture &&
                             cuda.cuStreamEndCapture && cuda.cuGraphDestroy;
      if (!graph_api || ocr_inference_graph.policy.capture_failed) {
        ocr_available = false;
        if (!ocr_error_logged) {
          BOOST_LOG(warning)
            << "PP-OCRv6 tiny conditional graph capture is unavailable; depth remains active "
               "and subtitle conditioning stays flat.";
          ocr_error_logged = true;
        }
        return true;
      }
      if (ocr_inference_graph.graph) {
        return true;
      }
      if (!allow_private_bootstrap) {
        fail_gpu_conditional_bridge_once(
          "OCR graph signature changed during a GPU-undecided transaction",
          CUDA_SUCCESS,
          true
        );
        return false;
      }
      if (!ocr_inference_graph.policy.signature_warmed) {
        if (!ocr_exec_context->enqueueV3(cu_stream)) {
          ocr_inference_graph.policy.capture_failed = true;
          mark_ocr_enqueue_failure("private signature bootstrap enqueueV3", CUDA_SUCCESS);
          return false;
        }
        joined_stream_work_ever_submitted = true;
        joined_stream_ocr_ever_submitted_or_armed = true;
        ocr_inference_graph.policy.signature_warmed = true;
      }

      CUgraph captured = nullptr;
      const CUresult begin = cuda.cuStreamBeginCapture(
        cu_stream,
        CU_STREAM_CAPTURE_MODE_RELAXED
      );
      if (begin != CUDA_SUCCESS) {
        ocr_inference_graph.policy.capture_failed = true;
        // BeginCapture may surface the immediately preceding private bootstrap's asynchronous
        // failure. That enqueue can have partially submitted either shared-stream context, so an
        // ordinary depth-only continuation would hide a poisoned joined transaction.
        mark_ocr_enqueue_failure("conditional source graph capture begin", begin);
        return false;
      }
      const bool captured_enqueue = ocr_exec_context->enqueueV3(cu_stream);
      const CUresult end = cuda.cuStreamEndCapture(cu_stream, &captured);
      if (captured_enqueue && end == CUDA_SUCCESS && captured) {
        ocr_inference_graph.graph = captured;
        BOOST_LOG(info)
          << "PP-OCRv6 tiny private source graph captured for fixed "
          << ocr_engine_width << 'x' << ocr_engine_height
          << "; it now runs only under authenticated optional inference.";
        return true;
      }

      ocr_inference_graph.graph = captured;
      const bool graph_released = destroy_inference_graph(
        cuda, ocr_inference_graph
      );
      ocr_inference_graph.policy.capture_failed = true;
      mark_ocr_enqueue_failure(
        "conditional source graph capture",
        end == CUDA_SUCCESS ? static_cast<CUresult>(-1) : end
      );
      execution_context_poisoned = execution_context_poisoned || !graph_released;
      return false;
    }

    // Caching
    int target_w = 0;
    int target_h = 0;
    // The required fused runtime uses one public high-resolution tensor throughout. Every
    // analysis/history/publication resource is target_w/target_h.
    int field_w = 0;
    int field_h = 0;
    UINT reduce_groups = 0;  // threadgroups for the shared moments/range reduction
    int cb_color_mode = -1;  // input_color_space baked into constant buffers
    depth_tensor_content_rect_t cb_tensor_content {};
    depth_source_rect_t cb_source_region {};
    // TensorRT retains dynamic shape and tensor-address bindings on an execution context. The
    // interop pointer is allowed to change after any map, so cache each piece independently and
    // rebind only the values that actually changed.
    int trt_bound_width = 0;
    int trt_bound_height = 0;
    int trt_bound_profile = -1;
    CUdeviceptr trt_bound_input = 0;
    CUdeviceptr trt_bound_output = 0;
    bool trt_output_sizes_validated = false;
    CUdeviceptr ocr_bound_input = 0;
    CUdeviceptr ocr_bound_output = 0;

    [[nodiscard]] bool production_depth_runtime_active() const noexcept {
      return depth_engine_io_kind == prod_zipdepth_convex2x::engine_io_e::
                                       production_dav2_zipdepth_convex2x;
    }

    [[nodiscard]] std::optional<depth_tensor_shape_t>
    calibrated_dav2_shape() const noexcept {
      if (target_w <= 0 || target_h <= 0) {
        return std::nullopt;
      }
      if (target_w % static_cast<int>(prod_zipdepth_convex2x::scale) != 0 ||
          target_h % static_cast<int>(prod_zipdepth_convex2x::scale) != 0) {
        return std::nullopt;
      }
      return depth_tensor_shape_t {
        target_w / static_cast<int>(prod_zipdepth_convex2x::scale),
        target_h / static_cast<int>(prod_zipdepth_convex2x::scale),
      };
    }

    Microsoft::WRL::ComPtr<ID3D11ComputeShader> near_identical_fused_preprocess_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> near_identical_tiled_preprocess_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> buffer_to_tex_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> buffer_to_tex_pad_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_moments_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_frame_resolve_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_histogram_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_publish_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_map_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_map_compact_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_coordinate_diagnostic_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_vertical_limit_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_limit_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_vertical_limit_compact_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_limit_compact_cs;
    // Eye-specific horizontal limiter; indexed by the compact-limiter flag.
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> depth_coordinate_v2_eye_limit_cs;
    std::array<Microsoft::WRL::ComPtr<ID3D11Buffer>, 2u> depth_coordinate_v2_eye_limit_cbuffers;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> ocr_preprocess_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> ocr_box_cells_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> ocr_box_resolve_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_locator_resolve_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_condition_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_contour_palette_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_contour_temporal_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_glyph_matte_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_contour_project_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_contour_history_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_contour_history_identity_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_contour_horizontal_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_contour_vertical_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_contour_snapshot_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_gaussian_vertical_limit_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> subtitle_gaussian_horizontal_limit_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> near_identical_resolve_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> near_identical_finalize_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> near_identical_reuse_depth_cs;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> gpu_trace_cs;
    Microsoft::WRL::ComPtr<ID3D11Buffer> cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> source_region_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> fused_preprocess_force_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> fused_preprocess_compare_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> depth_coordinate_v2_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> ocr_preprocess_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> ocr_resolve_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> subtitle_locator_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> subtitle_contour_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> near_identical_cbuffer;
    struct d3d_buffer_views_t {
      Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
      Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
      Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> uav;

      [[nodiscard]] bool ready() const noexcept {
        return buffer && srv && uav;
      }

      void reset() noexcept {
        buffer.Reset();
        srv.Reset();
        uav.Reset();
      }
    } near_identical_tile, near_identical_history_owner, analysis_domain;

    struct d3d_texture_views_t {
      Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
      Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
      Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> uav;

      [[nodiscard]] bool ready() const noexcept {
        return texture && srv && uav;
      }

      void reset() noexcept {
        texture.Reset();
        srv.Reset();
        uav.Reset();
      }
    } subtitle_contour_source, subtitle_contour_previous_features, subtitle_contour_next_features,
      subtitle_contour_support, subtitle_glyph_matte, subtitle_contour_source_snapshot;
    d3d_buffer_views_t subtitle_contour_identity, subtitle_contour_probability_snapshot;
    d3d_buffer_views_t subtitle_contour_observation;
    struct subtitle_conditioning_diagnostics_t {
      d3d_texture_views_t field_mask, horizontal_distance, distance;
      d3d_texture_views_t weight, blend, vertical_conditioned, final;
      d3d_buffer_views_t control;
      Microsoft::WRL::ComPtr<ID3D11Buffer> distance_constants;
      UINT horizontal_distance_height = 0u;
    };
    std::optional<subtitle_conditioning_diagnostics_t> subtitle_conditioning_diagnostics;
    struct completed_contour_views_t {
      Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> support, identity;
      Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> history, source, probability;
      Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> glyph_matte, condition_params;
      Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> subtitle_selected_matte, subtitle_selected_matte_receipt;
    } completed_contour;
    subtitle_publication::pool_t subtitle_draw_pool;
    std::weak_ptr<const subtitle_publication::views_t> completed_subtitle_draw;
    bool subtitle_draw_copy_failure_logged = false;

    bool subtitle_selected_matte_commands = false;
    host_sbs_direct_glyph_route::route_t subtitle_direct_glyph;
    // A completion whose Direct Glyph score was still pending resumes from here.
    struct subtitle_direct_glyph_tail_t {
      bool input_domain_reset;
      bool snapshot_requested;
    };
    std::optional<subtitle_direct_glyph_tail_t> subtitle_direct_glyph_tail;
    bool subtitle_direct_glyph_requested = false;
    // Set by the first estimate_depth(); configuration accepted only before the first input.
    bool configuration_locked = false;

    struct near_identical_transaction_resources_t : d3d_buffer_views_t {
      Microsoft::WRL::ComPtr<ID3D11Buffer> dispatch;

      [[nodiscard]] bool ready() const noexcept {
        return d3d_buffer_views_t::ready() && dispatch;
      }

      void reset() noexcept {
        d3d_buffer_views_t::reset();
        dispatch.Reset();
      }
    } near_identical_transaction;
    Microsoft::WRL::ComPtr<ID3D11Buffer> gpu_trace_cbuffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> gpu_trace_ring_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> gpu_trace_ring_srv;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> gpu_trace_ring_uav;
    std::shared_ptr<const host_sbs_gpu_trace_provenance_t> gpu_trace_provenance;
    bool gpu_trace_error_logged = false;
    Microsoft::WRL::ComPtr<ID3D11Buffer> gpu_outcome_buffer;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> gpu_outcome_uav;
    host_sbs_gpu_outcomes::delta_tracker_t gpu_outcome_counts;
    std::chrono::steady_clock::time_point gpu_outcome_next_log {};
    bool gpu_outcome_report_pending = false;
    bool gpu_outcome_error_logged = false;
    publication_readback_t publication_readback {device.Get(), context.Get()};
    const std::uint64_t publication_generation =
      g_publication_generation.fetch_add(1u, std::memory_order_relaxed);
    std::uint64_t publication_sequence = 0u;
    host_sbs_gpu_completion_receipt::expected_t current_publication {};
    struct pending_gpu_trace_append_t {
      host_sbs_gpu_trace::host_subtitle_outcome_e host_subtitle_outcome =
        host_sbs_gpu_trace::host_subtitle_outcome_e::suppressed;
      bool ocr_record_submitted = false;
      bool subtitle_work_suppressed = false;
      bool condition_executed = false;
      bool subtitle_branch_gated = false;
      bool input_domain_reset = false;
      bool valid = false;
    } pending_gpu_trace_append;

    Microsoft::WRL::ComPtr<ID3D11Buffer> tensor_in_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> tensor_in_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> tensor_in_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> tensor_previous_input_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> tensor_previous_input_srv;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> tensor_previous_input_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> tensor_out_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> tensor_out_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> raw_snapshot_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> raw_snapshot_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> model_input_snapshot_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> model_input_snapshot_srv;
    bool raw_snapshot_error_logged = false;
    bool model_input_snapshot_error_logged = false;
    unsigned raw_snapshot_retry_frames = 0;
    unsigned model_input_snapshot_retry_frames = 0;

    // GPU-resident min/max for per-frame disparity normalization (no CPU readback).
    Microsoft::WRL::ComPtr<ID3D11Buffer> minmax_raw_buf;  // min/max bits, valid and unmasked-eligible counts
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> minmax_raw_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> minmax_ema_buf;  // float4 {min,max,initialized,frame_state}
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> minmax_ema_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> minmax_ema_srv;
    // Established 32-word tagged state ABI. V2 writes only its depth-health fields; every
    // retired word keeps its zero initial value.
    Microsoft::WRL::ComPtr<ID3D11Buffer> depth_health_state_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_health_state_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_health_state_srv;

    // Optional evaluator fields. The two depth textures rotate for the diagnostic temporal EMA.
    d3d_texture_views_t normalized_depth;
    d3d_texture_views_t previous_normalized_depth;
    d3d_texture_views_t ema_motion_mask;
    // Synthetic letterbox support is visible to DAV2 but excluded from every analysis statistic,
    // history comparison and positioned renderer sample.

    // GPU-only raw-coordinate V2 producer. The canonical-coordinate texture is exceptional: its
    // shader, texture, and views are created only for an explicit Dump 3D snapshot. The
    // authenticated final field is the sole live position authority.
    Microsoft::WRL::ComPtr<ID3D11Buffer> depth_coordinate_v2_partials_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_partials_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_partials_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> depth_coordinate_v2_frame_stats_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_frame_stats_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_frame_stats_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> depth_coordinate_v2_histogram_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_histogram_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_histogram_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> depth_coordinate_v2_state_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_state_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_state_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_coordinate_v2_coordinate_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_coordinate_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_coordinate_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_coordinate_v2_candidate_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_candidate_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_candidate_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_coordinate_v2_vertical_majorant_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_vertical_majorant_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_vertical_majorant_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_coordinate_v2_vertical_conditioned_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView>
      depth_coordinate_v2_vertical_conditioned_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>
      depth_coordinate_v2_vertical_conditioned_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_coordinate_v2_final_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_final_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_final_srv;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_coordinate_v2_eye_final_tex;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> depth_coordinate_v2_eye_final_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_coordinate_v2_eye_final_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> subtitle_locator_state_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> subtitle_locator_state_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> subtitle_locator_state_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> subtitle_condition_params_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> subtitle_condition_params_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> subtitle_condition_params_srv;
    bool depth_coordinate_v2_coordinate_diagnostic_error_logged = false;

    CUgraphicsResource cuda_in_res = nullptr;
    CUgraphicsResource cuda_out_res = nullptr;
    CUgraphicsResource cuda_near_identical_decision_res = nullptr;
    CUgraphicsResource cuda_ocr_in_res = nullptr;
    CUgraphicsResource cuda_ocr_out_res = nullptr;
    Microsoft::WRL::ComPtr<ID3D11Buffer> ocr_input_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> ocr_input_uav;
    Microsoft::WRL::ComPtr<ID3D11Buffer> ocr_output_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ocr_output_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> ocr_cell_stats_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> ocr_cell_stats_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ocr_cell_stats_srv;
    Microsoft::WRL::ComPtr<ID3D11Buffer> ocr_box_record_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> ocr_box_record_uav;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ocr_box_record_srv;
    // Bounded local geometry workspace belongs to this estimator's isolated execution context.
    // The resolver initializes its entries for each observation; no prior mask lives here.
    Microsoft::WRL::ComPtr<ID3D11Buffer> ocr_local_candidate_scratch_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> ocr_local_candidate_scratch_uav;
    // The joined completion event is recorded after the single root stream has issued both child
    // branches and every interop-unmap tail. A ready event therefore proves the same resource
    // release boundary as the conservative full-stream query without exposing either branch.
    CUevent depth_inference_done_event = nullptr;
    inference_completion_notification_t root_completion_notification;
    bool completion_notification_disabled = false;
#ifdef SUNSHINE_TESTS
    std::unique_ptr<joined_completion_test_gate::controller_t> completion_test_gate;
    bool benchmark_joined_completion_gate_configuration_seen = false;
#endif
    bool near_identical_detector_available = false;
    bool near_identical_detector_error_logged = false;
    bool pending_depth_inference_event_recorded = false;
    bool inference_event_poll_available = false;
    bool inference_event_ever_recorded = false;
    // CUDA may report an earlier asynchronous bootstrap/root error through a later map, pointer,
    // or timing-event call even after nominal completion. Retain stream-lifetime participation so
    // those late failures quarantine every TensorRT context that could have executed.
    bool joined_stream_work_ever_submitted = false;
    bool joined_stream_ocr_ever_submitted_or_armed = false;
    enum class submission_class_e : std::uint8_t {
      force_infer,
      gpu_undecided,
    };
    struct subtitle_context_t {
      std::uint64_t generation = 0u;
      bool disabled = false;
      bool operator==(const subtitle_context_t &) const = default;
    };
    struct accepted_inference_t {
      std::uint64_t frame_id;
      std::uint64_t observation_timestamp_us;
      std::uint64_t transaction_token;
      std::uint64_t baseline_frame_id;
      depth_input_region_t input_region;
      input_color_space color_space;
      subtitle_context_t subtitle_context;
      subtitle_line_policy_e subtitle_line_policy;
      depth_optional_work_mode_e optional_work;
      submission_class_e submission_class;
      // The submitted disposition is used to authenticate the device receipt at postprocess.
      cuda_conditional_graph::work_flag_e subtitle_work;
      // This flag records OCR participation in the root, before reading the device receipt.
      bool ocr_armed;
      bool depth_only_history;
      bool dump_forced;
      Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> submitted_source;
    };
    std::optional<accepted_inference_t> pending_root;
    // Admission compares the completed subtitle context. Private diagnostics also need the
    // exact input region and certificate after the pending root has been consumed.
    struct {
      std::uint64_t frame_id = 0u;
      depth_input_region_t region {};
      subtitle_context_t subtitle_context;
    } completed_input;
    // Field pin, native distance, Gaussian proposal and both compatibility limiters feed only
    // Dump 3D and harness artifacts; the renderer binds Base. Live streaming runs them only
    // while completing an observation that a Dump 3D request snapshots (or one a dump forced)
    // and otherwise exposes the same immutable Base view for Final.
    bool subtitle_conditioning_diagnostics_always = true;
    bool completion_dump_snapshot_requested = false;
    bool completed_subtitle_conditioning_diagnostics = false;
    [[nodiscard]] bool subtitle_conditioning_diagnostics_requested() const noexcept {
      return subtitle_conditioning_diagnostics_always || (pending_root && pending_root->dump_forced) ||
             completion_dump_snapshot_requested;
    }
    [[nodiscard]] bool subtitle_legacy_matte_required() const noexcept {
      return subtitle_legacy_matte == subtitle_legacy_matte_e::always;
    }
    std::uint64_t force_infer_transaction_sequence = 0u;
    bool gpu_undecided_requires_force_infer_refresh = false;
    bool ocr_signature_refresh_required = false;
    bool ocr_shape_bound = false;
    bool ocr_output_size_validated = false;
    bool ocr_error_logged = false;
    depth_input_domain_tracker_t processed_input_domain;
    // Only a consumed final-publication receipt supplies this owner. A reused publication B may
    // retain owner A; completed color identity is never substituted for the actual infer owner.
    std::optional<publication_receipt::receipt_t> cached_current_publication;
    bool stream_error_logged = false;
    bool fused_input_region_error_logged = false;
    bool shape_mismatch_warning_logged = false;
    bool execution_context_poisoned = false;
    bool readiness_preflighted = false;  // can_accept_frame() already counted/queried this source opportunity
    bool depth_context_pooled = false;  // context reused from the pool (modules already loaded -> skip warmup)
    bool context_warmed = false;  // only warmed contexts may return to context_pool

    static depth_input_region_t resolved_input_region(
      depth_input_region_t requested,
      const D3D11_TEXTURE2D_DESC &input_desc
    ) noexcept {
      if (!requested.is_video_region()) {
        const auto tensor_shape = fit_host_sbs_v2_depth_tensor_shape(
          input_desc.Width,
          input_desc.Height
        );
        requested.source_width = input_desc.Width;
        requested.source_height = input_desc.Height;
        requested.left = 0u;
        requested.top = 0u;
        requested.right = input_desc.Width;
        requested.bottom = input_desc.Height;
        requested.tensor_content = {
          0u,
          0u,
          static_cast<std::uint32_t>(std::max(tensor_shape.width, 0)),
          static_cast<std::uint32_t>(std::max(tensor_shape.height, 0)),
        };
        requested.analysis_generation = 0u;
        return requested;
      }
      // Live ROI analysis samples the exact rectangle directly from the retained full matched
      // frame. A crop-sized SRV is deliberately rejected: accepting both layouts would make the
      // same placement metadata select two different physical coordinate systems.
      if (!requested.valid() || requested.source_width != input_desc.Width ||
          requested.source_height != input_desc.Height) {
        return {};
      }
      return requested;
    }

    depth_input_region_t resolved_runtime_input_region(
      depth_input_region_t requested,
      const D3D11_TEXTURE2D_DESC &input_desc
    ) {
      auto resolved = resolved_input_region(requested, input_desc);
      if (!resolved.valid() || !production_depth_runtime_active()) {
        return resolved;
      }
      const auto dav2_shape = fit_host_sbs_v2_depth_tensor_shape(
        resolved.source_width,
        resolved.source_height
      );
      const auto high_content = host_sbs_convex2x_content_rect(
        resolved.tensor_content,
        dav2_shape
      );
      const auto high_shape = host_sbs_convex2x_field_shape(dav2_shape);
      if (!high_content || !high_shape || !high_content->valid(*high_shape)) {
        if (!fused_input_region_error_logged) {
          std::ostringstream reason;
          reason << "Fused single-high input-region realization failed: source="
                 << resolved.source_width << 'x' << resolved.source_height
                 << ", coarse_fit=" << dav2_shape.width << 'x' << dav2_shape.height
                 << ", coarse_content=[" << resolved.tensor_content.left << ','
                 << resolved.tensor_content.top << ',' << resolved.tensor_content.right << ','
                 << resolved.tensor_content.bottom << ']';
          if (high_shape) {
            reason << ", expected_high=" << high_shape->width << 'x' << high_shape->height;
          } else {
            reason << ", expected_high=unavailable";
          }
          BOOST_LOG(error) << reason.str()
                           << "; this frame cannot submit depth work.";
          fused_input_region_error_logged = true;
        }
        return {};
      }
      resolved.tensor_content = *high_content;
      return resolved;
    }

#ifdef SUNSHINE_TESTS
    bool install_joined_completion_test_gate(
      const joined_completion_test_gate::configuration_t &configuration
    ) {
      if (!is_operational() || benchmark_joined_completion_gate_configuration_seen ||
          !configuration.valid()) {
        return false;
      }
      benchmark_joined_completion_gate_configuration_seen = true;
      if (!configuration.enabled) {
        return true;
      }
      // Event-query fallback cannot prove a literal held root.
      if (!inference_event_poll_available) {
        return false;
      }
      auto gate = std::make_unique<joined_completion_test_gate::controller_t>(configuration);
      if (!gate->available(cuda_driver_api::get())) {
        return false;
      }
      completion_test_gate = std::move(gate);
      return true;
    }
#endif

    void clear_pending_inference_event_state() noexcept {
      pending_depth_inference_event_recorded = false;
    }

    void invalidate_publication() noexcept {
      current_publication = {};
      cached_current_publication.reset();
    }

    // Host V2 fails flat on any producer error. Context quarantine is owned separately by failure
    // provenance: pre-enqueue validation may leave a context reusable, while a CUDA error after
    // asynchronous bootstrap/root work was submitted may be deferred and quarantines every
    // context that participated.
    void mark_terminal_failure(const bool poison_execution_context = false) {
      pending_branch_timing.clear();
      if (pending_root) {
        pending_root->submitted_source.Reset();
      }
      invalidate_publication();
      clear_pending_inference_event_state();
      execution_context_poisoned =
        execution_context_poisoned || poison_execution_context;
      lifecycle_state = lifecycle_state_e::terminal;
    }

    void mark_ocr_context_failure(
      const detail::warmed_execution_context_failure_e failure
    ) noexcept {
      ocr_context_health.observe(failure);
    }

    /** Apply the canonical fail-flat response to a CUDA call around the joined root stream.
     *
     * Required map/pointer operations are terminal even before the first launch. Optional timing
     * calls remain optional only until asynchronous bootstrap or root work has been submitted;
     * afterward their error result may be a deferred report from DAV2 or OCR and therefore owns
     * the same quarantine response as a root query failure.
     */
    bool observe_joined_cuda_failure(
      const std::string_view operation,
      const CUresult result,
      const bool required_operation
    ) {
      const auto policy = detail::joined_cuda_failure_policy(
        required_operation,
        joined_stream_work_ever_submitted,
        joined_stream_ocr_ever_submitted_or_armed
      );
      if (!policy.terminal) {
        return false;
      }
      if (!stream_error_logged) {
        BOOST_LOG(error) << "DAV2/OCR joined CUDA " << operation
                         << " failed: result=" << result;
        stream_error_logged = true;
      }
      if (policy.quarantine_ocr) {
        mark_ocr_context_failure(
          detail::warmed_execution_context_failure_e::asynchronous_execution_or_query
        );
      }
      mark_terminal_failure(policy.quarantine_depth);
      return true;
    }

    void fail_gpu_conditional_bridge_once(
      const std::string_view reason,
      const CUresult result = CUDA_SUCCESS,
      const bool poison_execution_context = false,
      const bool retain_live_wrapper = false
    ) {
      gpu_conditional_bridge_available = false;
      if (detail::bridge_failure_quarantines_ocr(
            poison_execution_context,
            joined_stream_ocr_ever_submitted_or_armed
          )) {
        // A later depth-only rebuild/launch API can surface an asynchronous fault from an earlier
        // optional child on the same stream. Current suppression does not erase that ownership.
        mark_ocr_context_failure(
          detail::warmed_execution_context_failure_e::asynchronous_execution_or_query
        );
      }
      // A failed launch can leave work from this executable in flight. Never destroy its embedded
      // child or mirror operands here; terminal teardown first quiesces the CUDA stream, then
      // releases the wrapper and every captured dependency in order.
      const bool wrapper_released = retain_live_wrapper ?
                                      depth_conditional_graph.empty() :
                                      reset_depth_conditional_graph();
      // Active arbitration is a required execution contract, not an optional accelerator. Even a
      // clean capability/topology rejection is terminal. Launch or unsafe teardown failures also
      // quarantine the TensorRT context so none of its captured dependencies can be reused.
      mark_terminal_failure(
        poison_execution_context || (!wrapper_released && !retain_live_wrapper)
      );
      if (!gpu_conditional_bridge_error_logged) {
        BOOST_LOG(error)
          << "Host SBS GPU near-identical arbitration failed (" << reason
          << (result == CUDA_SUCCESS ? "" : ", CUDA result " + std::to_string(result))
          << (retain_live_wrapper && !wrapper_released ?
                "); the live wrapper is retained until quiescent terminal teardown." :
              wrapper_released ?
                "); this estimator is terminal and will render flat without raw-graph fallback." :
                "); teardown was unsafe; dependencies are retained and the estimator is quarantined.");
        gpu_conditional_bridge_error_logged = true;
      }
    }

    [[nodiscard]] bool adopt_depth_conditional_candidate(
      cuda_conditional_graph::executable_t &&candidate,
      const CUdeviceptr decision_record,
      const CUdeviceptr request_record,
      const CUgraph child_graph,
      const CUgraph optional_child_graph
    ) {
      if (!depth_conditional_graph.adopt_from_empty(std::move(candidate))) {
        // ensure_depth_conditional_graph() proves the destination empty before construction. If
        // that invariant is ever broken, two wrappers may retain overlapping graph operands and
        // mapped records. Forget the uninstalled candidate without a CUDA call, retain every
        // dependency at terminal teardown, and never publish its pointer metadata.
        candidate.abandon_unsafe();
        execution_context_poisoned = true;
        gpu_conditional_bridge_context_failed = true;
        if (optional_child_graph || depth_conditional_optional_child_graph) {
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::unsafe_teardown
          );
        }
        fail_gpu_conditional_bridge_once(
          "conditional-wrapper adoption found a nonempty destination",
          CUDA_SUCCESS,
          true,
          true
        );
        return false;
      }
      depth_conditional_decision_ptr = decision_record;
      depth_conditional_request_ptr = request_record;
      depth_conditional_child_graph = child_graph;
      depth_conditional_optional_child_graph = optional_child_graph;
      return true;
    }

    bool ensure_depth_conditional_graph(
      cuda_driver_api &cuda,
      const CUdeviceptr decision_record,
      const CUdeviceptr request_record,
      const CUgraph optional_infer_child,
      const conditional_optional_child_policy_e optional_child_policy
    ) {
      const char *precondition_failure = nullptr;
      if (!gpu_conditional_bridge_available) {
        precondition_failure = "the bridge was previously disabled";
      } else if (!cuda.has_conditional_graph_support()) {
        precondition_failure = "required CUDA conditional/topology APIs are unavailable";
      } else if (!cuda_ctx) {
        precondition_failure = "the CUDA context is unavailable";
      } else if (!depth_inference_graph.graph) {
        precondition_failure = "the captured DAV2 graph is unavailable";
      } else if (depth_inference_graph.policy.capture_failed) {
        precondition_failure = "DAV2 graph capture previously failed";
      } else if (
                 optional_child_policy ==
                   conditional_optional_child_policy_e::build_ready &&
                 !optional_infer_child) {
        precondition_failure = "the mapped OCR child requested for construction is unavailable";
      } else if (optional_infer_child &&
                 (optional_infer_child != ocr_inference_graph.graph ||
                  ocr_inference_graph.policy.capture_failed)) {
        precondition_failure = "the captured OCR graph is not a valid optional child";
      } else if (
                 optional_infer_child &&
                 optional_child_policy !=
                   conditional_optional_child_policy_e::build_ready) {
        precondition_failure = "an unmapped OCR child cannot be used for construction";
      } else if (!decision_record || !request_record ||
                 (decision_record & 15u) != 0u || (request_record & 15u) != 0u) {
        precondition_failure = "the device transaction records are missing or misaligned";
      }
      if (precondition_failure) {
        if (!is_terminal()) {
          fail_gpu_conditional_bridge_once(precondition_failure);
        }
        return false;
      }
      const bool wrapper_base_matches =
        depth_conditional_graph.ready() &&
        depth_conditional_decision_ptr == decision_record &&
        depth_conditional_request_ptr == request_record &&
        depth_conditional_child_graph == depth_inference_graph.graph;
      const auto optional_topology = select_conditional_optional_topology(
        wrapper_base_matches,
        depth_conditional_optional_child_graph != nullptr,
        depth_conditional_optional_child_graph == optional_infer_child,
        optional_child_policy
      );
      if (optional_topology.reuse_existing) {
        return true;
      }

      if (!reset_depth_conditional_graph() || is_terminal()) {
        return false;
      }
      const CUgraph optional_child_for_build =
        optional_topology.build_with_optional ? optional_infer_child : nullptr;
      auto candidate = cuda_conditional_graph::executable_t::build(
        cuda,
        {
          .context = cuda_ctx,
          .infer_child = depth_inference_graph.graph,
          .optional_infer_child = optional_child_for_build,
          .reuse_child = nullptr,
          .decision_record = decision_record,
          .request_record = request_record,
          .record_branch_timing = diagnostics_enabled,
        }
      );
      if (!candidate.ready()) {
        const auto result = candidate.cuda_result();
        const auto build_failure = candidate.failure();
        const auto audit = candidate.audit_result();
        if (!candidate.empty()) {
          // A failed build normally self-cleans. If the CUDA driver could not destroy a partial
          // wrapper, take ownership of it so the child graph and mapped records cannot outlive
          // their backing. This estimator is terminal; teardown will retry under its context and
          // deliberately retain dependencies if the driver still refuses destruction.
          if (!adopt_depth_conditional_candidate(
                std::move(candidate),
                decision_record,
                request_record,
                depth_inference_graph.graph,
                optional_child_for_build
              )) {
            return false;
          }
          gpu_conditional_bridge_available = false;
          execution_context_poisoned = true;
          if (optional_child_for_build) {
            mark_ocr_context_failure(
              detail::warmed_execution_context_failure_e::
                asynchronous_execution_or_query
            );
          }
          mark_terminal_failure(true);
          if (!gpu_conditional_bridge_error_logged) {
            BOOST_LOG(warning)
              << "Host SBS GPU near-identical conditional graph construction left live CUDA "
                 "objects (CUDA result " << result
              << "); retaining all dependencies and quarantining this estimator.";
            gpu_conditional_bridge_error_logged = true;
          }
          return false;
        }
        BOOST_LOG(error)
          << "Conditional child topology audit: graphs=" << audit.visited_graphs
          << ", nodes=" << audit.visited_nodes
          << ", kernel=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_KERNEL]
          << ", memcpy=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_MEMCPY]
          << ", memset=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_MEMSET]
          << ", child=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_GRAPH]
          << ", empty=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_EMPTY]
          << ", wait-event=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_WAIT_EVENT]
          << ", record-event=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_EVENT_RECORD]
          << ", conditional=" << audit.node_type_counts[CU_GRAPH_NODE_TYPE_CONDITIONAL]
          << ".";
        if (optional_child_for_build) {
          // Any CUDA error while constructing the joined root may be a deferred report from one
          // of the private TensorRT bootstraps that preceded construction on this stream.
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::
              asynchronous_execution_or_query
          );
        }
        fail_gpu_conditional_bridge_once(
          "conditional graph construction failed at stage " +
            std::to_string(static_cast<unsigned>(build_failure)) +
            ", child audit " +
            std::to_string(static_cast<unsigned>(audit.failure)) +
            ", rejected node " +
            std::to_string(static_cast<int>(audit.rejected_type)),
          result,
          true
        );
        return false;
      }
      if (!adopt_depth_conditional_candidate(
            std::move(candidate),
            decision_record,
            request_record,
            depth_inference_graph.graph,
            optional_child_for_build
          )) {
        return false;
      }
      BOOST_LOG(info)
        << "Host SBS GPU near-identical arbitration is active with device-owned branch "
           "selection, optional OCR sibling=" << (optional_child_for_build ? "on" : "off")
        << ", and no decision readback.";
      return true;
    }

    // CUDA may report an earlier asynchronous launch error through a later root query,
    // synchronization, map, or unmap. Conservatively quarantine every TensorRT context embedded
    // in that root instead of assuming the failure belongs only to DAV2.
    void mark_shared_execution_failure(const bool ocr_may_have_participated) noexcept {
      mark_terminal_failure(true);
      if (ocr_may_have_participated || joined_stream_ocr_ever_submitted_or_armed) {
        mark_ocr_context_failure(
          detail::warmed_execution_context_failure_e::asynchronous_execution_or_query
        );
      }
    }

    void initialize_inference_done_events(cuda_driver_api &cuda) {
      clear_pending_inference_event_state();
      inference_event_poll_available = false;
      if (
        !cuda.cuEventCreate || !cuda.cuEventRecord || !cuda.cuEventQuery ||
        !cuda.cuEventDestroy
      ) {
        BOOST_LOG(warning)
          << "CUDA inference-event API is incomplete; same-frame completion uses one full-stream "
             "query without bounded waiting.";
        return;
      }

      const CUresult depth_create = cuda.cuEventCreate(
        &depth_inference_done_event,
        cuda_event_disable_timing
      );
      if (depth_create != CUDA_SUCCESS || !depth_inference_done_event) {
        BOOST_LOG(warning)
          << "Depth inference-event creation failed (" << depth_create
          << "); same-frame completion uses one full-stream query without bounded waiting.";
        (void) destroy_inference_done_events(cuda);
        return;
      }

      inference_event_poll_available = true;
    }

    bool destroy_inference_done_events(cuda_driver_api &cuda) noexcept {
      clear_pending_inference_event_state();
      inference_event_poll_available = false;
      if (!depth_inference_done_event) {
        return true;
      }
      const CUresult result = cuda.cuEventDestroy ?
                                cuda.cuEventDestroy(depth_inference_done_event) :
                                static_cast<CUresult>(-1);
      if (result != CUDA_SUCCESS) {
        BOOST_LOG(warning) << "CUDA root inference-event destruction failed: " << result;
        return false;
      }
      depth_inference_done_event = nullptr;
      return true;
    }

    bool record_inference_done_event(
      cuda_driver_api &cuda,
      CUevent event,
      CUstream stream,
      bool &recorded,
      const char *member,
      const bool ocr_was_submitted
    ) {
      recorded = false;
      const CUresult result = event && stream && cuda.cuEventRecord ?
                                cuda.cuEventRecord(event, stream) :
                                static_cast<CUresult>(-1);
      if (result != CUDA_SUCCESS) {
        BOOST_LOG(error) << member << " inference-event record failed: " << result;
        mark_shared_execution_failure(ocr_was_submitted);
        return false;
      }
      recorded = true;
      inference_event_ever_recorded = true;
      return true;
    }

    // Failing to select the shared CUDA context prevents proving the joined root's completion.
    void mark_cuda_context_failure() noexcept {
      mark_shared_execution_failure(pending_root && pending_root->ocr_armed);
    }

    void mark_ocr_enqueue_failure(
      const char *operation,
      const CUresult result
    ) noexcept {
      ocr_available = false;
      // enqueueV3(false) is not proof that no kernels were partially dispatched, and CUDA may
      // surface the failure through another context on the same driver context. Discard the depth
      // member too and quarantine both TensorRT contexts.
      mark_shared_execution_failure(true);
      if (!ocr_error_logged) {
        BOOST_LOG(warning) << "PP-OCRv6 tiny " << operation << " failed (" << result
                           << "); the exact-frame DAV2/OCR unit fails closed.";
        ocr_error_logged = true;
      }
    }

    static detail::async_stream_readiness_e normalized_cuda_readiness(
      const CUresult result
    ) noexcept {
      if (result == CUDA_SUCCESS) {
        return detail::async_stream_readiness_e::ready;
      }
      if (result == CUDA_ERROR_NOT_READY) {
        return detail::async_stream_readiness_e::busy;
      }
      return detail::async_stream_readiness_e::failed;
    }

    enum class pending_execution_readiness_e : std::uint8_t {
      ready,
      busy,
      failed,
    };

    pending_execution_readiness_e query_pending_execution(
      cuda_driver_api &cuda,
      const char *phase
    ) {
      const bool ocr_context_participated = pending_root && pending_root->ocr_armed;
      if (!cu_stream || !cuda.cuStreamQuery) {
        if (!stream_error_logged) {
          BOOST_LOG(error) << "DAV2/OCR joined CUDA root query is unavailable during " << phase
                           << " (stream=" << (cu_stream ? "present" : "missing")
                           << ", cuStreamQuery="
                           << (cuda.cuStreamQuery ? "present" : "missing") << ").";
          stream_error_logged = true;
        }
        mark_shared_execution_failure(ocr_context_participated);
        return pending_execution_readiness_e::failed;
      }

      const CUresult root_query = cuda.cuStreamQuery(cu_stream);
      switch (normalized_cuda_readiness(root_query)) {
        case detail::async_stream_readiness_e::ready:
          return pending_execution_readiness_e::ready;
        case detail::async_stream_readiness_e::busy:
          return pending_execution_readiness_e::busy;
        case detail::async_stream_readiness_e::failed:
          if (!stream_error_logged) {
            BOOST_LOG(error) << "DAV2/OCR joined CUDA root query failed during " << phase
                             << ": result=" << root_query;
            stream_error_logged = true;
          }
          mark_shared_execution_failure(ocr_context_participated);
          return pending_execution_readiness_e::failed;
      }
      return pending_execution_readiness_e::failed;
    }

    pending_execution_readiness_e query_pending_inference_events(
      cuda_driver_api &cuda,
      const char *phase
    ) {
      const bool event_state_valid = cuda.cuEventQuery && depth_inference_done_event &&
                                     pending_depth_inference_event_recorded;
      if (!event_state_valid) {
        if (!stream_error_logged) {
          BOOST_LOG(error)
            << "DAV2/OCR root inference-event state was incomplete during " << phase << '.';
          stream_error_logged = true;
        }
        mark_shared_execution_failure(pending_root && pending_root->ocr_armed);
        return pending_execution_readiness_e::failed;
      }

      const CUresult root_query = cuda.cuEventQuery(depth_inference_done_event);
#ifdef SUNSHINE_TESTS
      if (completion_test_gate) {
        completion_test_gate->note_actual_event_query(pending_root->frame_id, root_query);
      }
#endif
      switch (normalized_cuda_readiness(root_query)) {
        case detail::async_stream_readiness_e::ready:
          return pending_execution_readiness_e::ready;
        case detail::async_stream_readiness_e::busy:
          return pending_execution_readiness_e::busy;
        case detail::async_stream_readiness_e::failed:
          if (!stream_error_logged) {
            BOOST_LOG(error) << "DAV2/OCR root event query failed during " << phase
                             << ": result=" << root_query;
            stream_error_logged = true;
          }
          mark_shared_execution_failure(pending_root && pending_root->ocr_armed);
          return pending_execution_readiness_e::failed;
      }
      return pending_execution_readiness_e::failed;
    }

    bool synchronize_pending_execution(cuda_driver_api &cuda) {
      const bool ocr_participated = pending_root && pending_root->ocr_armed;
      if (!cu_stream || !cuda.cuStreamSynchronize) {
        if (!stream_error_logged) {
          BOOST_LOG(error) << "DAV2/OCR root synchronization is unavailable"
                           << " (stream=" << (cu_stream ? "present" : "missing")
                           << ", cuStreamSynchronize="
                           << (cuda.cuStreamSynchronize ? "present" : "missing") << ").";
          stream_error_logged = true;
        }
        mark_shared_execution_failure(ocr_participated);
        return false;
      }
      const CUresult root_sync = cuda.cuStreamSynchronize(cu_stream);
      if (root_sync != CUDA_SUCCESS) {
        BOOST_LOG(error) << "DAV2/OCR root synchronization failed: " << root_sync;
        mark_shared_execution_failure(ocr_participated);
        return false;
      }
      return true;
    }

    bool create_shader(
      const host_sbs_shader_cache::source_snapshot_t &sources,
      const host_sbs_shader_cache::shader_spec &spec,
      Microsoft::WRL::ComPtr<ID3D11ComputeShader> &out_cs
    ) {
      auto bytecode = host_sbs_shader_cache::get(sources, spec);
      if (!bytecode) {
        return false;
      }
      return SUCCEEDED(device->CreateComputeShader(bytecode->data(), bytecode->size(), nullptr, &out_cs));
    }

    void log_near_identical_detector_failure_once(const std::string_view reason) {
      if (!near_identical_detector_error_logged) {
        BOOST_LOG(error) << "Host SBS GPU near-identical detector is unavailable ("
                         << reason << "); the mandatory DAV2 wrapper cannot run.";
        near_identical_detector_error_logged = true;
      }
    }

    void initialize_near_identical_detector() {
      const auto sources = host_sbs_shader_cache::snapshot_sources(
        shader_root,
        host_sbs_shader_cache::near_identical_detector_specs
      );
      if (!sources) {
        log_near_identical_detector_failure_once("shader source snapshot failed");
        return;
      }
      if (
        host_sbs_shader_cache::source_closure_sha256(sources) !=
          host_sbs_shader_cache::near_identical_detector_source_closure_sha256
      ) {
        log_near_identical_detector_failure_once(
          "shader source closure authentication failed"
        );
        return;
      }
      const bool shaders_ready =
        create_shader(
          sources,
          host_sbs_shader_cache::host_sbs_near_identical_resolve,
          near_identical_resolve_cs
        ) &&
        create_shader(
          sources,
          host_sbs_shader_cache::host_sbs_near_identical_finalize,
          near_identical_finalize_cs
        ) &&
        (!normalized_depth_diagnostics_enabled ||
         create_shader(
           sources,
           host_sbs_shader_cache::host_sbs_near_identical_reuse_depth,
           near_identical_reuse_depth_cs
         ));
      D3D11_BUFFER_DESC constants_desc {};
      constants_desc.Usage = D3D11_USAGE_DEFAULT;
      constants_desc.ByteWidth = static_cast<UINT>(sizeof(near_identical_constants_t));
      constants_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      const bool constants_ready = shaders_ready && SUCCEEDED(device->CreateBuffer(
        &constants_desc,
        nullptr,
        near_identical_cbuffer.ReleaseAndGetAddressOf()
      ));
      if (!constants_ready) {
        near_identical_resolve_cs.Reset();
        near_identical_finalize_cs.Reset();
        near_identical_reuse_depth_cs.Reset();
        near_identical_cbuffer.Reset();
        log_near_identical_detector_failure_once(
          "shader or constant-buffer setup failed"
        );
        return;
      }
      near_identical_detector_available = true;
      BOOST_LOG(info)
        << "Host SBS GPU near-identical detector initialized from its independently "
           "authenticated mandatory closure.";
    }

    void log_gpu_trace_failure_once(const std::string_view reason) {
      if (!gpu_trace_error_logged) {
        BOOST_LOG(warning) << "Host SBS diagnostic GPU completion trace is unavailable ("
                           << reason << "); live rendering is unaffected.";
        gpu_trace_error_logged = true;
      }
    }

    void reset_gpu_trace_resources() noexcept {
      gpu_trace_cs.Reset();
      gpu_trace_cbuffer.Reset();
      gpu_trace_ring_buf.Reset();
      gpu_trace_ring_srv.Reset();
      gpu_trace_ring_uav.Reset();
      gpu_trace_provenance.reset();
      pending_gpu_trace_append = {};
      reset_gpu_outcome_resources();
    }

    void reset_gpu_outcome_resources() noexcept {
      if (telemetry_performance) {
        telemetry_performance->invalidate_outcomes();
      }
      gpu_outcome_buffer.Reset();
      gpu_outcome_uav.Reset();
      gpu_outcome_counts.reset();
      gpu_outcome_next_log = {};
      gpu_outcome_report_pending = false;
    }

    void disable_gpu_outcome_logging(const std::string_view reason) {
      reset_gpu_outcome_resources();
      if (!gpu_outcome_error_logged) {
        BOOST_LOG(warning) << "Host SBS diagnostic GPU outcome counters are unavailable ("
                           << reason << "); completion trace and live rendering are unaffected.";
        gpu_outcome_error_logged = true;
      }
    }

    void initialize_gpu_outcome_resources() {
      reset_gpu_outcome_resources();
      D3D11_BUFFER_DESC counter_desc {};
      counter_desc.Usage = D3D11_USAGE_DEFAULT;
      counter_desc.ByteWidth = host_sbs_gpu_outcomes::byte_count;
      counter_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      counter_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      counter_desc.StructureByteStride = sizeof(std::uint32_t);
      D3D11_SUBRESOURCE_DATA initial {host_sbs_gpu_outcomes::initial_words.data(), 0u, 0u};
      if (FAILED(device->CreateBuffer(
            &counter_desc, &initial, gpu_outcome_buffer.ReleaseAndGetAddressOf()
          )) || FAILED(device->CreateUnorderedAccessView(
            gpu_outcome_buffer.Get(), nullptr, gpu_outcome_uav.ReleaseAndGetAddressOf()
          ))) {
        disable_gpu_outcome_logging("optional counter-resource setup failed");
      }
    }

    void service_gpu_outcome_logging() {
      // Receipt consumption already collected cumulative counts. Idle logging performs no GPU
      // access and never creates conversion demand or affects the adaptive scheduler.
      if (!diagnostics_enabled || !gpu_outcome_report_pending) {
        return;
      }
      const auto now = std::chrono::steady_clock::now();
      if (now < gpu_outcome_next_log) {
        return;
      }
      const auto delta = gpu_outcome_counts.take_delta();
      const auto totals = gpu_outcome_counts.totals();
      BOOST_LOG(info) << "Host SBS GPU outcomes: infer=" << delta.infer
                      << " reuse=" << delta.reuse << " invalid=" << delta.invalid
                      << " reuse_percent=" << std::round(delta.reuse_percent() * 10.0) / 10.0
                      << " cumulative_infer=" << totals.infer
                      << " cumulative_reuse=" << totals.reuse
                      << " cumulative_invalid=" << totals.invalid;
      gpu_outcome_next_log = now + 5s;
      gpu_outcome_report_pending = false;
    }

    void consume_completion_diagnostics(const publication_receipt_poll_result &batch) {
      for (std::size_t index = 0u; index < batch.diagnostic_count; ++index) {
        const auto &diagnostic = batch.diagnostics[index];
        if (diagnostic.expected.publication_sequence <= latest_diagnostic_sequence) {
          continue;
        }
        latest_diagnostic_sequence = diagnostic.expected.publication_sequence;
        depth_telemetry_sample sample;
        if (diagnostic.depth_health_state && decode_telemetry_words(
              *diagnostic.depth_health_state,
              diagnostic.expected.width, diagnostic.expected.height,
              diagnostic.expected.frame_id, diagnostic.sampled_at, sample
            )) {
          cached_depth_telemetry = {.sample = sample};
        } else {
          cached_depth_telemetry = {.failed = true};
        }
        if (diagnostic.outcome_counts) {
          if (gpu_outcome_counts.observe(*diagnostic.outcome_counts)) {
            BOOST_LOG(warning) << "Host SBS diagnostic GPU outcome counters restarted; "
                                  "reporting a new counter epoch.";
          }
          if (telemetry_performance) {
            telemetry_performance->record_outcomes(*diagnostic.outcome_counts, diagnostic.sampled_at);
          }
          if (gpu_outcome_next_log == std::chrono::steady_clock::time_point {}) {
            gpu_outcome_next_log = std::chrono::steady_clock::now() + 5s;
          }
          gpu_outcome_report_pending = true;
        } else if (diagnostics_enabled && gpu_outcome_buffer && telemetry_performance) {
          telemetry_performance->invalidate_outcomes();
        }
      }
      if (batch.transport_failed) {
        cached_depth_telemetry = {.failed = true};
        if (telemetry_performance) {
          telemetry_performance->invalidate_outcomes();
        }
      }
      service_gpu_outcome_logging();
    }

    /**
     * Eye-specific horizontal limiter (constants in host_sbs_eye_field.h). The shader lies outside
     * the authenticated producer closure, and the Base field never depends on it: any failure here
     * leaves Base as the only field and the renderer on its ordinary route.
     */
    void initialize_eye_limiter() {
      static constexpr host_sbs_shader_cache::shader_spec eye_spec {
        "depth_coordinate_v2_eye_limit_cs.hlsl",
        "main",
        "cs_5_0"
      };
      const auto sources = host_sbs_shader_cache::snapshot_sources(shader_root, std::span {&eye_spec, 1u});
      bool ready = sources && create_shader(sources, eye_spec, depth_coordinate_v2_eye_limit_cs);
      const std::uint32_t covered_cells = host_sbs_eye_field::covered_extension_cells;
      const std::uint32_t uncovered_cells = host_sbs_eye_field::uncovered_extension_cells;
      for (std::uint32_t compact = 0u; ready && compact < 2u; ++compact) {
        struct constants_t {
          float occlusion_slope;
          float hole_slope;
          std::uint32_t extension_cells;
          std::uint32_t compact;
        } constants {
          host_sbs_eye_field::covered_side_slope,
          depth_coordinate_v2::max_horizontal_slope,
          host_sbs_eye_field::extension_word(covered_cells, uncovered_cells),
          compact,
        };
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.ByteWidth = static_cast<UINT>(sizeof(constants));
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA data {&constants, 0u, 0u};
        ready = SUCCEEDED(device->CreateBuffer(
          &desc,
          &data,
          depth_coordinate_v2_eye_limit_cbuffers[compact].ReleaseAndGetAddressOf()
        ));
      }
      if (!ready) {
        depth_coordinate_v2_eye_limit_cs.Reset();
        BOOST_LOG(warning) << "Host SBS per-eye limiter shader unavailable; using the Base field.";
        return;
      }
      BOOST_LOG(info) << "Host SBS eye-specific limiter active: covered-side slope "
                      << host_sbs_eye_field::covered_side_slope << ", uncovered-side slope "
                      << depth_coordinate_v2::max_horizontal_slope << ", foreground extension "
                      << covered_cells << " covered-side and " << uncovered_cells << " uncovered-side cell(s).";
    }

    void initialize_gpu_trace() {
      const auto sources = host_sbs_shader_cache::snapshot_sources(
        shader_root,
        host_sbs_shader_cache::gpu_trace_specs
      );
      const std::string closure_sha256 = sources ?
        host_sbs_shader_cache::source_closure_sha256(sources) : std::string {};
      if (!sources || closure_sha256 !=
            host_sbs_shader_cache::gpu_trace_source_closure_sha256 || !create_shader(
            sources,
            host_sbs_shader_cache::host_sbs_gpu_trace,
            gpu_trace_cs
          )) {
        reset_gpu_trace_resources();
        log_gpu_trace_failure_once(
          "shader source authentication or compilation failed"
        );
        return;
      }

      std::vector<std::uint32_t> initial_ring(host_sbs_gpu_trace::ring_word_count, 0u);
      initial_ring[host_sbs_gpu_trace::word_index(
        host_sbs_gpu_trace::header_word_e::schema
      )] = host_sbs_gpu_trace::ring_schema;
      initial_ring[host_sbs_gpu_trace::word_index(
        host_sbs_gpu_trace::header_word_e::tag
      )] = host_sbs_gpu_trace::ring_tag;
      initial_ring[host_sbs_gpu_trace::word_index(
        host_sbs_gpu_trace::header_word_e::capacity
      )] = host_sbs_gpu_trace::capacity;
      initial_ring[host_sbs_gpu_trace::word_index(
        host_sbs_gpu_trace::header_word_e::record_words
      )] = host_sbs_gpu_trace::record_word_count;
      initial_ring[host_sbs_gpu_trace::word_index(
        host_sbs_gpu_trace::header_word_e::next_sequence_low
      )] = 1u;
      D3D11_BUFFER_DESC ring_desc {};
      ring_desc.Usage = D3D11_USAGE_DEFAULT;
      ring_desc.ByteWidth = host_sbs_gpu_trace::ring_byte_count;
      ring_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      ring_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      ring_desc.StructureByteStride = sizeof(std::uint32_t);
      D3D11_SUBRESOURCE_DATA ring_data {initial_ring.data(), 0u, 0u};

      D3D11_BUFFER_DESC constants_desc {};
      constants_desc.Usage = D3D11_USAGE_DEFAULT;
      constants_desc.ByteWidth =
        host_sbs_gpu_trace::constant_word_count * sizeof(std::uint32_t);
      constants_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

      const bool resources_ready =
        SUCCEEDED(device->CreateBuffer(
          &ring_desc,
          &ring_data,
          gpu_trace_ring_buf.ReleaseAndGetAddressOf()
        )) &&
        SUCCEEDED(device->CreateShaderResourceView(
          gpu_trace_ring_buf.Get(), nullptr, gpu_trace_ring_srv.ReleaseAndGetAddressOf()
        )) &&
        SUCCEEDED(device->CreateUnorderedAccessView(
          gpu_trace_ring_buf.Get(), nullptr, gpu_trace_ring_uav.ReleaseAndGetAddressOf()
        )) &&
        SUCCEEDED(device->CreateBuffer(
          &constants_desc, nullptr, gpu_trace_cbuffer.ReleaseAndGetAddressOf()
        ));
      if (!resources_ready) {
        reset_gpu_trace_resources();
        log_gpu_trace_failure_once("optional shader-resource setup failed");
        return;
      }

      gpu_trace_provenance =
        std::make_shared<const host_sbs_gpu_trace_provenance_t>(
          host_sbs_gpu_trace_provenance_t {
            .source_closure_schema = host_sbs_shader_cache::source_closure_schema,
            .source_compile_flags = host_sbs_shader_cache::shader_compile_flags,
            .source_macro_count = 0u,
            .source_closure_sha256 = closure_sha256,
          }
        );
      initialize_gpu_outcome_resources();
      BOOST_LOG(info) << "Host SBS diagnostic GPU completion trace initialized ("
                      << host_sbs_gpu_trace::capacity << " records, "
                      << host_sbs_gpu_trace::record_word_count * sizeof(std::uint32_t)
                      << " bytes each).";
    }

    bool unregister_near_identical_decision_interop(cuda_driver_api &cuda) {
      if (!reset_depth_conditional_graph()) {
        fail_gpu_conditional_bridge_once(
          "conditional-wrapper teardown failed before decision-buffer unregister",
          CUDA_SUCCESS,
          true,
          true
        );
        return false;
      }
      if (unregister_cuda_graphics_resource(
            cuda, cuda_near_identical_decision_res
          )) {
        return true;
      }
      execution_context_poisoned = true;
      fail_gpu_conditional_bridge_once(
        "decision-buffer interop unregister failed",
        CUDA_SUCCESS,
        true,
        true
      );
      return false;
    }

    bool ensure_near_identical_decision_interop(cuda_driver_api &cuda) {
      if (cuda_near_identical_decision_res) {
        return true;
      }
      if (
        !gpu_conditional_bridge_available ||
        !cuda.has_conditional_graph_support() ||
        !near_identical_transaction.buffer ||
        !cuda.cuGraphicsD3D11RegisterResource
      ) {
        return false;
      }
      const CUresult registered = cuda.cuGraphicsD3D11RegisterResource(
        &cuda_near_identical_decision_res,
        near_identical_transaction.buffer.Get(),
        0u
      );
      if (
        registered == CUDA_SUCCESS && cuda_near_identical_decision_res
      ) {
        return true;
      }

      bool cleanup_ok = true;
      if (cuda_near_identical_decision_res) {
        cleanup_ok = unregister_cuda_graphics_resource(
          cuda, cuda_near_identical_decision_res
        );
      }
      if (!cleanup_ok) {
        execution_context_poisoned = true;
        mark_terminal_failure(true);
      }
      fail_gpu_conditional_bridge_once(
        cleanup_ok ?
          "decision-buffer CUDA registration failed" :
          "decision-buffer registration cleanup failed",
        registered
      );
      return false;
    }

    bool ensure_near_identical_detector_resources() {
      if (!near_identical_detector_available || target_w <= 0 || target_h <= 0) {
        return false;
      }
      if (near_identical_tile.ready() && near_identical_transaction.ready() &&
          near_identical_history_owner.ready()) {
        return true;
      }

      invalidate_publication();
      auto &cuda = cuda_driver_api::get();
      if (
        cuda_near_identical_decision_res &&
        !unregister_near_identical_decision_interop(cuda)
      ) {
        mark_terminal_failure(true);
        return false;
      }

      near_identical_tile.reset();
      near_identical_transaction.reset();
      near_identical_history_owner.reset();

      const auto tile_layout = detail::checked_near_identical_tile_layout(
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h)
      );
      if (!tile_layout) {
        log_near_identical_detector_failure_once(
          "GPU evidence-tile layout exceeds the checked D3D11 buffer boundary"
        );
        return false;
      }
      const auto create_structured_uint_buffer = [&] (
                                                   const std::uint32_t word_count,
                                                   const std::uint32_t stride_words,
                                                   Microsoft::WRL::ComPtr<ID3D11Buffer> &buffer,
                                                   Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> &srv,
                                                   Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> &uav
                                                 ) {
        if (word_count == 0u || stride_words == 0u ||
            word_count % stride_words != 0u) {
          return false;
        }
        const auto byte_width =
          static_cast<std::uint64_t>(word_count) * sizeof(std::uint32_t);
        const auto stride_bytes =
          static_cast<std::uint64_t>(stride_words) * sizeof(std::uint32_t);
        if (byte_width > std::numeric_limits<UINT>::max() ||
            stride_bytes > std::numeric_limits<UINT>::max()) {
          return false;
        }
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = static_cast<UINT>(byte_width);
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = static_cast<UINT>(stride_bytes);
        return SUCCEEDED(device->CreateBuffer(
                 &desc, nullptr, buffer.ReleaseAndGetAddressOf()
               )) &&
               SUCCEEDED(device->CreateShaderResourceView(
                 buffer.Get(), nullptr, srv.ReleaseAndGetAddressOf()
               )) &&
               SUCCEEDED(device->CreateUnorderedAccessView(
                 buffer.Get(), nullptr, uav.ReleaseAndGetAddressOf()
               ));
      };
      const auto create_decision_buffers = [&]() {
        std::array<std::uint32_t, near_identical_gpu_decision_word_count> initial {};
        initial[near_identical_gpu_decision_word_index(
          near_identical_gpu_decision_word_e::decision
        )] = static_cast<std::uint32_t>(near_identical_gpu_branch_e::infer);
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = static_cast<UINT>(sizeof(initial));
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        // CUDA rejects a D3D11 buffer carrying DRAWINDIRECT_ARGS. Keep the authenticated
        // proposal/receipt in this registerable raw buffer, let postprocess write every argument,
        // then copy the 256-byte transaction into a D3-only twin before indirect consumers.
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        D3D11_SUBRESOURCE_DATA initial_data {initial.data(), 0u, 0u};
        if (FAILED(device->CreateBuffer(
              &desc,
              &initial_data,
              near_identical_transaction.buffer.ReleaseAndGetAddressOf()
            ))) {
          return false;
        }
        D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc {};
        uav_desc.Format = DXGI_FORMAT_R32_TYPELESS;
        uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uav_desc.Buffer.FirstElement = 0u;
        uav_desc.Buffer.NumElements =
          static_cast<UINT>(near_identical_gpu_decision_word_count);
        uav_desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        if (FAILED(device->CreateUnorderedAccessView(
              near_identical_transaction.buffer.Get(),
              &uav_desc,
              near_identical_transaction.uav.ReleaseAndGetAddressOf()
            ))) {
          return false;
        }
        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc {};
        srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
        srv_desc.BufferEx.FirstElement = 0u;
        srv_desc.BufferEx.NumElements =
          static_cast<UINT>(near_identical_gpu_decision_word_count);
        srv_desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
        if (FAILED(device->CreateShaderResourceView(
              near_identical_transaction.buffer.Get(),
              &srv_desc,
              near_identical_transaction.srv.ReleaseAndGetAddressOf()
            ))) {
          return false;
        }
        desc.BindFlags = 0u;
        desc.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
        return SUCCEEDED(device->CreateBuffer(
                 &desc,
                 &initial_data,
                 near_identical_transaction.dispatch.ReleaseAndGetAddressOf()
               ));
      };
      const bool resources_ready =
        create_structured_uint_buffer(
          tile_layout->word_count,
          4u,
          near_identical_tile.buffer,
          near_identical_tile.srv,
          near_identical_tile.uav
        ) &&
        create_structured_uint_buffer(
          static_cast<std::uint32_t>(near_identical_history_owner_word_count),
          1u,
          near_identical_history_owner.buffer,
          near_identical_history_owner.srv,
          near_identical_history_owner.uav
        ) &&
        create_decision_buffers();
      if (!resources_ready) {
        near_identical_tile.reset();
        near_identical_transaction.reset();
        near_identical_history_owner.reset();
        near_identical_detector_available = false;
        std::ostringstream reason;
        reason << "GPU evidence/transaction-buffer setup failed for "
               << target_w << 'x' << target_h << " (tiles "
               << tile_layout->group_width << 'x' << tile_layout->group_height
               << '=' << tile_layout->group_count << ", "
               << tile_layout->byte_count << " bytes)";
        log_near_identical_detector_failure_once(reason.str());
        return false;
      }
      const UINT zero[4] = {};
      context->ClearUnorderedAccessViewUint(
        near_identical_history_owner.uav.Get(), zero
      );
      return true;
    }

    void update_near_identical_constants(
      const std::uint32_t request_flags,
      const std::uint32_t tile_group_width,
      const std::uint32_t tile_group_height,
      const std::uint64_t current_frame_id,
      const std::uint64_t baseline_frame_id,
      const std::uint64_t domain_tag,
      const std::uint64_t request_token,
      const std::uint32_t stream_frame_delta = 0u,
      const cuda_conditional_graph::work_flag_e expected_work =
        cuda_conditional_graph::work_flag_e::none,
      const std::uint64_t observation_timestamp_us = 0u
    ) {
      const near_identical_constants_t constants {
        request_flags,
        tile_group_width,
        tile_group_height,
        tile_group_width * tile_group_height,
        static_cast<std::uint32_t>(current_frame_id),
        static_cast<std::uint32_t>(current_frame_id >> 32u),
        static_cast<std::uint32_t>(baseline_frame_id),
        static_cast<std::uint32_t>(baseline_frame_id >> 32u),
        static_cast<std::uint32_t>(domain_tag),
        static_cast<std::uint32_t>(domain_tag >> 32u),
        static_cast<std::uint32_t>(request_token),
        static_cast<std::uint32_t>(request_token >> 32u),
        reduce_groups,
        stream_frame_delta,
        cuda_conditional_graph::work_flags_value(expected_work),
        cuda_conditional_graph::work_flags_value(expected_work) == 0u ?
          0u : cuda_conditional_graph::work_flags_value(expected_work) ^
                 cuda_conditional_graph::work_flags_cookie,
        static_cast<std::uint32_t>(observation_timestamp_us),
        static_cast<std::uint32_t>(observation_timestamp_us >> 32u),
        0u,
        0u,
      };
      context->UpdateSubresource(
        near_identical_cbuffer.Get(), 0u, nullptr, constants.data(), 0u, 0u
      );
    }

    [[nodiscard]] std::uint64_t next_force_infer_transaction_token() noexcept {
      // Keep host-forced tokens in a visibly distinct nonzero namespace. The GPU owns selection;
      // a later final-publication snapshot may authenticate the completed receipt. Uniqueness
      // across this estimator lifetime rejects stale records from an earlier mapped transaction.
      constexpr std::uint64_t force_namespace = 0xf000000000000000ull;
      constexpr std::uint64_t sequence_mask = 0x0fffffffffffffffull;
      force_infer_transaction_sequence =
        (force_infer_transaction_sequence + 1u) & sequence_mask;
      if (force_infer_transaction_sequence == 0u) {
        force_infer_transaction_sequence = 1u;
      }
      return force_namespace | force_infer_transaction_sequence;
    }

    bool publish_force_infer_transaction(
      const std::uint64_t token,
      const cuda_conditional_graph::work_flag_e subtitle_work
    ) {
      if (token == 0u || !near_identical_transaction.buffer) {
        return false;
      }
      auto transaction = initialize_near_identical_transaction(token, subtitle_work);
      const auto proposal = cuda_conditional_graph::make_proposal(
        cuda_conditional_graph::branch_e::infer,
        token
      );
      std::memcpy(
        reinterpret_cast<std::byte *>(transaction.data()) +
          near_identical_gpu_decision_record_byte_offset,
        &proposal,
        sizeof(proposal)
      );
      if (subtitle_work == cuda_conditional_graph::work_flag_e::optional_ocr) {
        transaction[near_identical_gpu_decision_word_index(
          near_identical_gpu_decision_word_e::optional_preprocess_x
        )] = (ocr_engine_width + 15u) / 16u;
        transaction[near_identical_gpu_decision_word_index(
          near_identical_gpu_decision_word_e::optional_preprocess_y
        )] = (ocr_engine_height + 15u) / 16u;
        transaction[near_identical_gpu_decision_word_index(
          near_identical_gpu_decision_word_e::optional_preprocess_z
        )] = 1u;
      }
      context->UpdateSubresource(
        near_identical_transaction.buffer.Get(),
        0u,
        nullptr,
        transaction.data(),
        0u,
        0u
      );
      return true;
    }

    // The fused map root authenticates the history owner it publishes against these exact b2
    // words. Upload them once per infer postprocess, with no host update before that dispatch.
    bool upload_near_identical_history_owner_constants(const bool skip_input_history) {
      if (
        !near_identical_cbuffer || !near_identical_history_owner.srv || !cbuffer ||
        pending_root->frame_id == 0u ||
        !pending_root->input_region.tensor_content.valid({target_w, target_h})
      ) {
        return false;
      }
      const auto domain_tag = near_identical_input_domain_tag(
        pending_root->input_region,
        pending_root->color_space,
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h)
      );
      update_near_identical_constants(
        skip_input_history ? near_identical_map_skip_input_history : 0u,
        0u, 0u, pending_root->frame_id, 0u, domain_tag, 0u, 0u,
        cuda_conditional_graph::work_flag_e::none,
        pending_root->observation_timestamp_us
      );
      return true;
    }

    [[nodiscard]] bool gpu_undecided_baseline_authorized(
      const gpu_adaptive_reuse_request &request
    ) const noexcept {
      return cached_current_publication &&
             publication_is_current(cached_current_publication->expected) &&
             gpu_adaptive_baseline_matches(*cached_current_publication, request.depth_only_history) &&
             request.baseline_frame_id == cached_current_publication->depth_owner_frame_id;
    }

    bool prepare_near_identical_detector(
      const gpu_adaptive_reuse_request &request,
      const std::uint64_t current_frame_id,
      const depth_input_region_t &input_region,
      const input_color_space color_space,
      const cuda_conditional_graph::work_flag_e subtitle_work
    ) {
      if (
        !request.authorize_gpu_undecided_reuse ||
        request.gpu_reuse_decision_token == 0u ||
        request.observation_timestamp_us == 0u ||
        !near_identical_detector_available ||
        !ensure_near_identical_detector_resources() ||
        current_frame_id == 0u || request.baseline_frame_id == 0u ||
        current_frame_id <= request.baseline_frame_id ||
        !gpu_undecided_baseline_authorized(request) ||
        !processed_input_domain.matches_analysis_domain(input_region, color_space) ||
        !input_region.tensor_content.valid({target_w, target_h}) ||
        !near_identical_fused_preprocess_cs || !near_identical_resolve_cs ||
        !near_identical_cbuffer || !near_identical_tile.uav ||
        !near_identical_tile.srv || !near_identical_history_owner.srv ||
        !near_identical_transaction.buffer || !near_identical_transaction.uav ||
        !near_identical_transaction.dispatch ||
        !tensor_in_srv || !tensor_previous_input_srv || !cbuffer
      ) {
        return false;
      }

      const auto tile_layout = detail::checked_near_identical_tile_layout(
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h)
      );
      if (!tile_layout) {
        return false;
      }
      const auto domain_tag = near_identical_input_domain_tag(
        input_region,
        color_space,
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h)
      );
      update_near_identical_constants(
        1u,
        tile_layout->group_width,
        tile_layout->group_height,
        current_frame_id,
        request.baseline_frame_id,
        domain_tag,
        request.gpu_reuse_decision_token,
        0u,
        subtitle_work,
        request.observation_timestamp_us
      );
      const auto initialized_transaction = initialize_near_identical_transaction(
        request.gpu_reuse_decision_token,
        subtitle_work
      );
      // One full-record write invalidates every prior proposal/receipt and every indirect arg
      // before this frame's fused evidence pass. Optional preprocess starts at zero and is enabled
      // only by a matching authenticated proposal; a missing/malformed proposal cannot waste OCR work or
      // authorize stale OCR output even though CUDA safely fails the mandatory depth branch open.
      context->UpdateSubresource(
        near_identical_transaction.buffer.Get(),
        0u,
        nullptr,
        initialized_transaction.data(),
        0u,
        0u
      );
      return true;
    }

    bool dispatch_near_identical_detector() {
      if (
        !near_identical_detector_available ||
        !near_identical_resolve_cs ||
        !near_identical_cbuffer || !near_identical_tile.srv ||
        !near_identical_history_owner.srv ||
        !near_identical_transaction.uav || !cbuffer
      ) {
        return false;
      }

      ID3D11Buffer *constant_buffers[2] = {
        cbuffer.Get(),
        near_identical_cbuffer.Get(),
      };
      context->CSSetConstantBuffers(0u, 2u, constant_buffers);

      ID3D11ShaderResourceView *resolve_inputs[4] = {
        nullptr,
        nullptr,
        near_identical_tile.srv.Get(),
        near_identical_history_owner.srv.Get(),
      };
      ID3D11UnorderedAccessView *resolve_outputs[4] = {
        nullptr,
        nullptr,
        nullptr,
        near_identical_transaction.uav.Get(),
      };
      context->CSSetShader(near_identical_resolve_cs.Get(), nullptr, 0u);
      context->CSSetShaderResources(0u, 4u, resolve_inputs);
      context->CSSetUnorderedAccessViews(0u, 4u, resolve_outputs, nullptr);
      context->Dispatch(1u, 1u, 1u);
      ID3D11ShaderResourceView *null_resolve_inputs[4] = {};
      ID3D11UnorderedAccessView *null_resolve_outputs[4] = {};
      ID3D11Buffer *null_constants[2] = {};
      context->CSSetUnorderedAccessViews(0u, 4u, null_resolve_outputs, nullptr);
      context->CSSetShaderResources(0u, 4u, null_resolve_inputs);
      context->CSSetConstantBuffers(0u, 2u, null_constants);

      return true;
    }

    [[nodiscard]] bool gpu_undecided_postprocess_pending() const noexcept {
      return pending_root && pending_root->submission_class == submission_class_e::gpu_undecided;
    }


    void dispatch_infer_postprocess(
      const UINT direct_x,
      const UINT direct_y,
      const UINT direct_z,
      const std::uint32_t indirect_byte_offset
    ) {
      if (!gpu_undecided_postprocess_pending()) {
        context->Dispatch(direct_x, direct_y, direct_z);
      } else if (near_identical_transaction.dispatch) {
        context->DispatchIndirect(
          near_identical_transaction.dispatch.Get(),
          indirect_byte_offset
        );
      }
    }

    [[nodiscard]] host_sbs_v2_gpu::dispatch_command_t
      parallax_v2_dispatch_command(
        const UINT direct_x,
        const UINT direct_y,
        const UINT direct_z,
        const std::uint32_t indirect_byte_offset
      ) const noexcept {
      if (!gpu_undecided_postprocess_pending()) {
        return host_sbs_v2_gpu::dispatch_command_t::direct(
          direct_x,
          direct_y,
          direct_z
        );
      }
      return host_sbs_v2_gpu::dispatch_command_t::indirect(
        near_identical_transaction.dispatch.Get(),
        indirect_byte_offset
      );
    }

    [[nodiscard]] std::uint32_t pending_near_identical_postprocess_mode() const noexcept {
      return gpu_undecided_postprocess_pending() ? 1u : 2u;
    }
    void update_pending_near_identical_postprocess_constants() {
      const auto mode = pending_near_identical_postprocess_mode();
      const bool gpu_undecided = mode == 1u;
      const auto domain_tag = near_identical_input_domain_tag(
        pending_root->input_region,
        pending_root->color_space,
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h)
      );
      const std::uint32_t groups_x =
        (static_cast<std::uint32_t>(target_w) + 15u) / 16u;
      const std::uint32_t groups_y =
        (static_cast<std::uint32_t>(target_h) + 15u) / 16u;
      std::uint64_t stream_frame_delta = 1u;
      if (pending_root->baseline_frame_id != 0u &&
          pending_root->frame_id > pending_root->baseline_frame_id) {
        stream_frame_delta = pending_root->frame_id - pending_root->baseline_frame_id;
      }
      update_near_identical_constants(
        mode,
        gpu_undecided ? groups_x : 0u,
        gpu_undecided ? groups_y : 0u,
        pending_root->frame_id,
        gpu_undecided ? pending_root->baseline_frame_id : 0u,
        domain_tag,
        pending_root->transaction_token,
        static_cast<std::uint32_t>(
          std::min<std::uint64_t>(stream_frame_delta, 65535u)
        ),
        pending_root->subtitle_work,
        pending_root->observation_timestamp_us
      );
    }

    bool dispatch_near_identical_finalizer(const bool input_domain_reset) {
      const bool subtitle_publication_expected =
        cuda_conditional_graph::work_flags_value(pending_root->subtitle_work) != 0u;
      ID3D11Buffer *geometry_cbuffer = cbuffer.Get();
      if (
        !near_identical_finalize_cs ||
        !near_identical_transaction.uav ||
        !near_identical_transaction.buffer || !near_identical_transaction.dispatch ||
        !near_identical_history_owner.srv ||
        !near_identical_cbuffer || !cbuffer || !geometry_cbuffer ||
        pending_root->transaction_token == 0u ||
        (subtitle_publication_expected &&
         (!ocr_box_record_uav || !ocr_resolve_cbuffer ||
          !update_pending_ocr_constants(input_domain_reset)))
      ) {
        return false;
      }
      update_pending_near_identical_postprocess_constants();
      ID3D11Buffer *constant_buffers[4] = {
        cbuffer.Get(),
        near_identical_cbuffer.Get(),
        subtitle_publication_expected ? ocr_resolve_cbuffer.Get() : nullptr,
        geometry_cbuffer,
      };
      ID3D11UnorderedAccessView *outputs[4] = {
        nullptr,
        subtitle_publication_expected ? ocr_box_record_uav.Get() : nullptr,
        nullptr,
        near_identical_transaction.uav.Get(),
      };
      context->CSSetConstantBuffers(0u, 4u, constant_buffers);
      ID3D11ShaderResourceView *receipt_inputs[4] = {
        nullptr,
        nullptr,
        nullptr,
        near_identical_history_owner.srv.Get(),
      };
      context->CSSetShaderResources(0u, 4u, receipt_inputs);
      context->CSSetUnorderedAccessViews(0u, 4u, outputs, nullptr);
      context->CSSetShader(near_identical_finalize_cs.Get(), nullptr, 0u);
      context->Dispatch(1u, 1u, 1u);
      ID3D11UnorderedAccessView *null_outputs[4] = {};
      ID3D11ShaderResourceView *null_inputs[4] = {};
      ID3D11Buffer *null_constants[4] = {};
      context->CSSetUnorderedAccessViews(0u, 4u, null_outputs, nullptr);
      context->CSSetShaderResources(0u, 4u, null_inputs);
      context->CSSetConstantBuffers(0u, 4u, null_constants);
      // CUDA cannot register a DRAWINDIRECT_ARGS buffer. Once the receipt validator has written
      // every args record and any deterministic abstention into their registerable resources,
      // copy the complete 256-byte transaction into the D3-only indirect twin consumed below.
      context->CopyResource(
        near_identical_transaction.dispatch.Get(),
        near_identical_transaction.buffer.Get()
      );
      return true;
    }

    void dispatch_near_identical_reuse_depth() {
      if (
        !gpu_undecided_postprocess_pending() || !near_identical_reuse_depth_cs ||
        !near_identical_transaction.dispatch || !previous_normalized_depth.srv || !normalized_depth.uav ||
        !cbuffer
      ) {
        return;
      }
      ID3D11ShaderResourceView *inputs[6] = {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        previous_normalized_depth.srv.Get(),
      };
      ID3D11UnorderedAccessView *outputs[5] = {
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        normalized_depth.uav.Get(),
      };
      context->CSSetShader(near_identical_reuse_depth_cs.Get(), nullptr, 0u);
      context->CSSetConstantBuffers(0u, 1u, cbuffer.GetAddressOf());
      context->CSSetShaderResources(0u, 6u, inputs);
      context->CSSetUnorderedAccessViews(0u, 5u, outputs, nullptr);
      context->DispatchIndirect(
        near_identical_transaction.dispatch.Get(),
        near_identical_gpu_reuse_grid16_byte_offset
      );
      ID3D11ShaderResourceView *null_inputs[6] = {};
      ID3D11UnorderedAccessView *null_outputs[5] = {};
      ID3D11Buffer *null_constant = nullptr;
      context->CSSetUnorderedAccessViews(0u, 5u, null_outputs, nullptr);
      context->CSSetShaderResources(0u, 6u, null_inputs);
      context->CSSetConstantBuffers(0u, 1u, &null_constant);
    }


    bool initialize_ocr_context(
      const std::filesystem::path &assets_dir,
      cuda_driver_api &cuda
    ) {
      engine_artifact artifact;
      if (!ensure_ocr_tensorrt_engine_for_device(
            assets_dir,
            cuda,
            cuda_device,
            artifact
          )) {
        return false;
      }
      auto engine_path = artifact.engine_path;
      ocr_engine_key = std::to_string(cuda_device) + ":" + artifact.name;
      bool engine_repair_allowed = true;
      {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        ocr_engine = acquire_engine_locked(
          ocr_engine_key,
          engine_path,
          ocr_exec_context,
          &ocr_context_pooled
        );
        auto &slot = g_engines[ocr_engine_key];
        if (!validate_ocr_engine_io_locked(ocr_engine, slot)) {
          return_execution_context_locked(ocr_engine_key, ocr_exec_context);
          ocr_context_pooled = false;
          ocr_context_warmed = false;
          engine_repair_allowed =
            detach_incompatible_engine_locked(ocr_engine_key);
          ocr_engine = nullptr;
        }
        ocr_context_warmed = ocr_context_pooled;
      }

      if (!ocr_engine) {
        // Startup normally repairs this cache first, but standalone evaluators can construct the
        // estimator directly. Keep the same one-shot, no-live-owner repair at this second seam.
        if (!engine_repair_allowed) {
          BOOST_LOG(error)
            << "PP-OCRv6 tiny cannot replace an incompatible TensorRT plan while another "
               "session owns a context.";
          return false;
        }
        {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          erase_empty_engine_slot_locked(ocr_engine_key);
        }
        std::error_code remove_error;
        std::filesystem::remove(engine_path, remove_error);
        BOOST_LOG(warning)
          << "Depth estimator found an unreadable or incompatible PP-OCRv6 tiny plan; "
             "rebuilding "
          << engine_path.filename() << '.';
        if (!ensure_ocr_tensorrt_engine_for_device(
              assets_dir, cuda, cuda_device,
              artifact
            )) {
          return false;
        }
        engine_path = artifact.engine_path;
        ocr_engine_key = std::to_string(cuda_device) + ":" + artifact.name;
        {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          ocr_engine = acquire_engine_locked(
            ocr_engine_key,
            engine_path,
            ocr_exec_context,
            &ocr_context_pooled
          );
          auto &slot = g_engines[ocr_engine_key];
          if (!validate_ocr_engine_io_locked(ocr_engine, slot)) {
            return_execution_context_locked(ocr_engine_key, ocr_exec_context);
            ocr_context_pooled = false;
            ocr_context_warmed = false;
            detach_incompatible_engine_locked(ocr_engine_key);
            ocr_engine = nullptr;
          } else {
            ocr_context_warmed = ocr_context_pooled;
          }
        }
        if (!ocr_engine) {
          BOOST_LOG(error)
            << "Rebuilt PP-OCRv6 tiny plan is unreadable or violates the FP32 I/O contract.";
          return false;
        }
      }

      bool create_context = false;
      if (!ocr_exec_context) {
        std::unique_lock<std::mutex> lock(g_trt_mutex);
        auto &slot = g_engines[ocr_engine_key];
        const std::size_t context_limit = slot_context_limit(slot);
        if (slot.context_pool.empty() &&
            allocated_context_count(slot) >= context_limit) {
          g_trt_context_available.wait_for(
            lock,
            std::chrono::seconds(2),
            [&slot, context_limit]() {
              return !slot.context_pool.empty() ||
                     allocated_context_count(slot) < context_limit;
            }
          );
        }
        if (!slot.context_pool.empty()) {
          ocr_exec_context = slot.context_pool.back();
          slot.context_pool.pop_back();
          ocr_context_pooled = true;
          ocr_context_warmed = true;
        } else if (allocated_context_count(slot) < context_limit) {
          create_context = slot.context_accounting.reserve(context_limit);
        }
      }
      if (create_context) {
        ocr_exec_context = ocr_engine->createExecutionContext();
        if (!ocr_exec_context) {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          release_context_reservation_locked(ocr_engine_key);
          return false;
        }
        if (!warmup_ocr_execution_context(cuda, cuda_ctx, ocr_exec_context)) {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          quarantine_execution_context_locked(
            ocr_engine_key, ocr_exec_context,
            false
          );
          return false;
        }
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        mark_execution_context_warmed_locked(ocr_engine_key);
        ocr_context_warmed = true;
      }
      if (!ocr_exec_context || !ocr_context_warmed) {
        BOOST_LOG(warning)
          << "No isolated warmed PP-OCRv6 tiny context is available; conditioning stays flat.";
        return false;
      }
      // A pooled context was warmed before it entered the global cache. Revalidate its fixed
      // context-dependent output bound in this owning instance before any D3D interop binding.
      const auto fixed_dims = make_input_dims(ocr_engine_height, ocr_engine_width);
      constexpr std::int64_t fixed_output_bytes =
        static_cast<std::int64_t>(ocr_engine_width) * ocr_engine_height * sizeof(float);
      if (!ocr_exec_context->setInputShape("x", fixed_dims)) {
        BOOST_LOG(error)
          << "PP-OCRv6 tiny could not restore its authenticated fixed input shape.";
        mark_ocr_context_failure(
          detail::warmed_execution_context_failure_e::pre_enqueue_interop_or_binding
        );
        return false;
      }
      ocr_shape_bound = true;
      const std::int64_t required_output_bytes =
        ocr_exec_context->getMaxOutputSize("fetch_name_0");
      if (required_output_bytes <= 0 || required_output_bytes > fixed_output_bytes) {
        BOOST_LOG(error)
          << "PP-OCRv6 tiny resolved output requires " << required_output_bytes
          << " bytes; the authenticated fixed buffer provides " << fixed_output_bytes << '.';
        mark_ocr_context_failure(
          detail::warmed_execution_context_failure_e::pre_enqueue_interop_or_binding
        );
        return false;
      }
      ocr_output_size_validated = true;
      if (!publish_active_engine_manifest(assets_dir, ocr_model_name, artifact)) {
        BOOST_LOG(warning) << "Could not publish the active PP-OCRv6 tiny engine manifest.";
      }
      BOOST_LOG(info) << "PP-OCRv6 tiny live context acquired (fixed 960x160).";
      return true;
    }

    impl(
      Microsoft::WRL::ComPtr<ID3D11Device> d,
      Microsoft::WRL::ComPtr<ID3D11DeviceContext> c,
      const std::filesystem::path &assets_dir,
      const config::video_t::sbs_t &cfg,
      const config::depth_model_info &model,
      normalized_depth_diagnostics_e normalized_depth_diagnostics,
      subtitle_legacy_matte_e legacy_matte
    ):
        device(d),
        context(c),
        shader_root(assets_dir / "shaders" / "directx"),
        ema_alpha((float) config::host_sbs_v2_live_calibration::depth_ema),
        ema_edge_change((float) config::host_sbs_v2_live_calibration::edge_change),
        ema_edge_gradient((float) config::host_sbs_v2_live_calibration::edge_gradient),
        ema_edge_strength((float) config::host_sbs_v2_live_calibration::edge_strength),
        depth_short_side(config::host_sbs_v2_live_calibration::depth_short_side),
        max_aspect((float) config::host_sbs_v2_live_calibration::depth_max_aspect),
        minmax_alpha((float) config::host_sbs_v2_live_calibration::minmax_ema),
        diagnostics_enabled(config::sunshine.diagnostics_enabled),
        normalized_depth_diagnostics_enabled(normalized_depth_diagnostics == normalized_depth_diagnostics_e::enabled),
        subtitle_legacy_matte(legacy_matte),
        parallax_v2_requested_pop_strength(
          depth_coordinate_v2::requested_pop_strength(
            static_cast<float>(cfg.pop_strength)
          )
        ),
        parallax_v2_requested_gain(depth_coordinate_v2::requested_gain_for_config(static_cast<float>(cfg.pop_strength))) {
      const auto init_started = std::chrono::steady_clock::now();
      // Enable the process-wide rolling collector for diagnostic runs. Do not reset it here:
      // Galaxy XR and local-AR estimators may coexist, and one session must not invalidate the
      // other session's pending D3D query generation. The offline harness resets explicitly.
      sbs_perf::set_enabled(diagnostics_enabled);
      initialize_d3d_perf();

      auto &cuda = cuda_driver_api::get();
      if (cuda.is_valid() && ensure_cuda_initialized(cuda)) {
        if (cuda_device_for_d3d(cuda, device.Get(), cuda_device)) {
          cuda_ctx = primary_context(cuda, cuda_device);
          if (cuda_ctx) {
            const auto context_result = cuda.cuCtxSetCurrent(cuda_ctx);
            if (context_result != CUDA_SUCCESS) {
              BOOST_LOG(error)
                << "Depth estimator failed: could not select the D3D adapter's CUDA context ("
                << context_result << ").";
              cuda_ctx = nullptr;
            } else {
              const auto stream_result =
                cuda.cuStreamCreate(&cu_stream, CU_STREAM_NON_BLOCKING);
              if (stream_result != CUDA_SUCCESS || !cu_stream) {
                BOOST_LOG(error)
                  << "Depth estimator failed: CUDA stream creation failed ("
                  << stream_result << ").";
                // CUDA documents a null stream on failure, but defensively destroy any handle it
                // returned before rejecting construction so no wrong/partial stream can survive.
                if (cu_stream && cuda.cuStreamDestroy) {
                  cuda.cuStreamDestroy(cu_stream);
                }
                cu_stream = nullptr;
              }
            }
          }
        }
      }
      if (!cuda_ctx || !cu_stream) {
        BOOST_LOG(error) << "Depth estimator failed: CUDA context/stream initialization failed.";
        return;
      }

      resolved_depth_runtime_t depth_runtime;
      if (!resolve_depth_runtime(assets_dir, model, depth_runtime)) {
        BOOST_LOG(error) << "Depth estimator failed: composite runtime resolution failed.";
        return;
      }
      const auto &runtime_model = depth_runtime.model;

      engine_artifact artifact;
      if (!ensure_tensorrt_engine_for_device(
            assets_dir, runtime_model, cuda, cuda_device, artifact
          )) {
        BOOST_LOG(error) << "Depth estimator failed: TensorRT engine preparation failed.";
        return;
      }
      auto model_path = artifact.engine_path;
      engine_key = std::to_string(cuda_device) + ":" + artifact.name;
      bool engine_repair_allowed = true;

      {  // Scope this lock to the g_engines/g_runtime access only: it MUST be released before
         // warmup_inference() at the end of the ctor (which re-locks g_trt_mutex) -- a
         // non-recursive std::mutex would otherwise self-deadlock and hang construction.
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        // Load (once) the engine for this configured model into its own slot and take a pooled
        // execution context if one is free. Different startup configurations remain isolated.
        engine = acquire_engine_locked(
          engine_key,
          model_path,
          exec_context,
          &depth_context_pooled
        );
        context_warmed = depth_context_pooled;
        if (depth_context_pooled) {
          BOOST_LOG(info) << "Reusing pooled TensorRT execution context.";
        }
        auto &slot = g_engines[engine_key];

        if (!validate_engine_io_locked(engine, slot)) {
          BOOST_LOG(error) << "Depth engine I/O contract is incompatible with Sunshine 3D; streaming flat SBS.";
          return_execution_context_locked(engine_key, exec_context);
          depth_context_pooled = false;
          context_warmed = false;
          engine_repair_allowed = detach_incompatible_engine_locked(engine_key);
          engine = nullptr;
        } else {
          depth_engine_io_kind = slot.io_kind;
          if (depth_engine_io_kind != prod_zipdepth_convex2x::engine_io_e::
                                        production_dav2_zipdepth_convex2x) {
            return_execution_context_locked(engine_key, exec_context);
            depth_context_pooled = false;
            context_warmed = false;
            engine_repair_allowed = detach_incompatible_engine_locked(engine_key);
            engine = nullptr;
          }
        }

      }  // release g_trt_mutex before the shader/buffer setup and warmup below

      if (!engine) {
        // The startup preparation normally repairs this already. The constructor also serves the
        // standalone evaluator, so retain the same one-shot self-heal when it is the first owner.
        if (!engine_repair_allowed) {
          BOOST_LOG(error) << "Depth estimator cannot replace an incompatible TensorRT plan "
                              "while another session still owns its execution context.";
          return;
        }
        {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          erase_empty_engine_slot_locked(engine_key);
        }
        std::error_code ec;
        std::filesystem::remove(model_path, ec);
        BOOST_LOG(warning) << "Depth estimator found an unreadable or incompatible TensorRT plan; rebuilding "
                           << model_path.filename() << '.';
        if (!ensure_tensorrt_engine_for_device(
              assets_dir, runtime_model, cuda, cuda_device, artifact
            )) {
          return;
        }
        model_path = artifact.engine_path;
        engine_key = std::to_string(cuda_device) + ":" + artifact.name;
        {
          std::lock_guard<std::mutex> lock(g_trt_mutex);
          engine = acquire_engine_locked(
            engine_key,
            model_path,
            exec_context,
            &depth_context_pooled
          );
          context_warmed = depth_context_pooled;
          auto &slot = g_engines[engine_key];
          if (!validate_engine_io_locked(engine, slot)) {
            return_execution_context_locked(engine_key, exec_context);
            depth_context_pooled = false;
            context_warmed = false;
            detach_incompatible_engine_locked(engine_key);
            engine = nullptr;
          } else {
            depth_engine_io_kind = slot.io_kind;
            if (depth_engine_io_kind != prod_zipdepth_convex2x::engine_io_e::
                                          production_dav2_zipdepth_convex2x) {
              return_execution_context_locked(engine_key, exec_context);
              depth_context_pooled = false;
              context_warmed = false;
              detach_incompatible_engine_locked(engine_key);
              engine = nullptr;
            }
          }
        }
        if (!engine) {
          BOOST_LOG(error) << "Depth estimator failed: rebuilt TensorRT plan is unreadable or incompatible.";
          return;
        }
      }

      const auto *model_coordinate_calibration =
        depth_coordinate_v2::find_model_calibration(
          model.name,
          model.url,
          prod_zipdepth_convex2x::dav2_onnx_sha256
        );
      if (engine && !exec_context) {
        // Pool empty. On a back-to-back session rebuild the previous estimator is often
        // still tearing down on the async-teardown thread and will return its context to
        // the pool momentarily -- wait briefly for that before paying seconds (and ~1.3 GB
        // of device scratch) for a fresh context. Only a usable context that someone still
        // holds can come back: the first owner in a process (the offline worker, a harness
        // run) has none to wait for.
        {
          std::unique_lock<std::mutex> lock(g_trt_mutex);
          auto &slot = g_engines[engine_key];
          if (slot.context_accounting.usable() != 0u) {
            g_trt_context_available.wait_for(lock, std::chrono::seconds(1), [&slot]() {
              return !slot.context_pool.empty();
            });
          }
          if (!slot.context_pool.empty()) {
            exec_context = slot.context_pool.back();
            slot.context_pool.pop_back();
            depth_context_pooled = true;
            context_warmed = true;
            BOOST_LOG(info) << "Reusing pooled TensorRT execution context (freed by a racing teardown).";
          }
        }
        bool create_context = false;
        if (!exec_context) {
          std::unique_lock<std::mutex> lock(g_trt_mutex);
          auto &slot = g_engines[engine_key];
          const std::size_t context_limit = slot_context_limit(slot);
          if (slot.context_pool.empty() &&
              allocated_context_count(slot) >= context_limit) {
            BOOST_LOG(warning) << "TensorRT context cap reached for this depth model; waiting for "
                                  "an asynchronous encoder teardown to return one.";
            const bool available = g_trt_context_available.wait_for(
              lock,
              std::chrono::seconds(5),
              [&slot, context_limit]() {
                return !slot.context_pool.empty() ||
                       allocated_context_count(slot) < context_limit;
              }
            );
            if (!available) {
              if (context_limit == detail::fused_tensorrt_context_limit &&
                  slot.context_accounting.quarantined() != 0u) {
                BOOST_LOG(error)
                  << "The sole fused TensorRT execution context is quarantined; leaving this "
                     "session flat until the process restarts.";
              } else {
                BOOST_LOG(error)
                  << "TensorRT context cap remained saturated; leaving this encode session flat "
                     "instead of allocating unbounded GPU memory.";
              }
              engine = nullptr;
            }
          }
          if (engine && !slot.context_pool.empty()) {
            exec_context = slot.context_pool.back();
            slot.context_pool.pop_back();
            depth_context_pooled = true;
            context_warmed = true;
            BOOST_LOG(info) << "Reusing pooled TensorRT execution context after bounded wait.";
          } else if (engine) {
            // Reserve atomically so concurrent constructors cannot exceed the cap.
            create_context = slot.context_accounting.reserve(context_limit);
          }
        }
        if (create_context) {
          // Deliberately OUTSIDE g_trt_mutex: creation allocates device scratch and can
          // take many seconds; holding the lock would block a concurrent estimator
          // destructor from returning its context to the pool (observed 46 s teardown)
          // and any concurrent enqueueV3. ICudaEngine is thread-safe for this call.
          BOOST_LOG(info) << "Creating TensorRT execution context (allocates device scratch; may take several seconds)...";
          exec_context = engine->createExecutionContext();
          if (!exec_context) {
            std::lock_guard<std::mutex> lock(g_trt_mutex);
            release_context_reservation_locked(engine_key);
          }
        }
      }

      // Preprocessing plus the private normalized depth and its min/max health reduction.
      const auto preprocess_sources = host_sbs_shader_cache::snapshot_sources(
        shader_root,
        host_sbs_shader_cache::preprocess_specs
      );
      if (!preprocess_sources) {
        BOOST_LOG(error)
          << "Depth estimator failed: the calibrated RGB-preprocess source closure is unavailable.";
        return;
      }
      const std::string preprocess_source_closure_sha256 =
        host_sbs_shader_cache::source_closure_sha256(preprocess_sources);
      const auto *coordinate_calibration =
        model_coordinate_calibration &&
        model_coordinate_calibration->preprocess.source_closure_schema ==
          host_sbs_shader_cache::source_closure_schema &&
        model_coordinate_calibration->preprocess.source_file ==
          host_sbs_shader_cache::host_sbs_near_identical_fused_preprocess.filename &&
        model_coordinate_calibration->preprocess.source_entrypoint ==
          host_sbs_shader_cache::host_sbs_near_identical_fused_preprocess.entrypoint &&
        model_coordinate_calibration->preprocess.source_target ==
          host_sbs_shader_cache::host_sbs_near_identical_fused_preprocess.target &&
        model_coordinate_calibration->preprocess.source_compile_flags ==
          host_sbs_shader_cache::shader_compile_flags &&
        model_coordinate_calibration->preprocess.source_macro_count == 0u &&
        model_coordinate_calibration->preprocess.source_closure_sha256 ==
          preprocess_source_closure_sha256 ?
          model_coordinate_calibration : nullptr;
      if (!coordinate_calibration) {
        BOOST_LOG(error)
          << "Host SBS V2 cannot authenticate uncalibrated depth model '"
          << model.name << "' (raw DAV2 ONNX SHA-256 "
          << prod_zipdepth_convex2x::dav2_onnx_sha256
          << ", preprocess source SHA-256 " << preprocess_source_closure_sha256
          << "); Host SBS will fail flat.";
        return;
      }
      parallax_v2_raw_coordinate_scale = coordinate_calibration->raw_coordinate_scale;
      raw_model_provenance = std::make_shared<const raw_model_provenance_t>(
        raw_model_provenance_t {
          .depth_model = model.name,
          .depth_model_url = model.url,
          .onnx_sha256 = std::string {prod_zipdepth_convex2x::dav2_onnx_sha256},
          .preprocess_profile = std::string {
            coordinate_calibration->preprocess.profile
          },
          .preprocess_source_closure_sha256 = preprocess_source_closure_sha256,
        }
      );
      composite_depth_runtime_provenance =
        std::make_shared<const composite_depth_runtime_provenance_t>(
          composite_depth_runtime_provenance_t {
            .model = runtime_model.name,
            .onnx_sha256 = artifact.source_sha256,
            .embedded_dav2_onnx_sha256 =
              std::string {prod_zipdepth_convex2x::dav2_onnx_sha256},
            .zipdepth_checkpoint_sha256 =
              std::string {prod_zipdepth_convex2x::zipdepth_checkpoint_sha256},
            .guidance_preprocess_source_closure_sha256 =
              preprocess_source_closure_sha256,
            .engine_recipe = std::string {prod_zipdepth_convex2x::engine_recipe},
            .engine_artifact = artifact.name,
            .active_engine_manifest = runtime_model.name + ".active-engine.json",
          }
        );

      const auto producer_sources = host_sbs_shader_cache::snapshot_sources(
        shader_root,
        host_sbs_shader_cache::parallax_v2_producer_specs
      );
      static_assert(
        depth_coordinate_v2::shader_source_closure_schema ==
        host_sbs_shader_cache::source_closure_schema
      );
      static_assert(
        depth_coordinate_v2::shader_source_compile_flags ==
        host_sbs_shader_cache::shader_compile_flags
      );
      static_assert(depth_coordinate_v2::shader_source_macro_count == 0u);
      static_assert(
        depth_coordinate_v2::shader_source_specs.size() ==
        host_sbs_shader_cache::parallax_v2_producer_specs.size()
      );
      const std::string shader_source_closure_sha256 =
        producer_sources ?
          host_sbs_shader_cache::source_closure_sha256(producer_sources) :
          std::string {};
      const bool shader_identity_matches =
        shader_source_closure_sha256 ==
          depth_coordinate_v2::shader_source_closure_sha256;

      // Create the required compute objects from the one authenticated producer snapshot.
      // Startup prewarm populates these exact cache entries, so the constructor neither
      // recompiles the fused preprocess from its identity-only calibration closure nor compiles the shared
      // analysis roots once as "core" and then again as V2.
      const std::array producer_shader_outputs {
        std::addressof(near_identical_fused_preprocess_cs),
        std::addressof(near_identical_tiled_preprocess_cs),
        normalized_depth_diagnostics_enabled ? std::addressof(buffer_to_tex_cs) : nullptr,
        normalized_depth_diagnostics_enabled ? std::addressof(buffer_to_tex_pad_cs) : nullptr,
        std::addressof(depth_coordinate_v2_moments_cs),
        std::addressof(depth_coordinate_v2_frame_resolve_cs),
        std::addressof(depth_coordinate_v2_histogram_cs),
        std::addressof(depth_coordinate_v2_publish_cs),
        std::addressof(depth_coordinate_v2_map_cs),
        normalized_depth_diagnostics_enabled ? nullptr : std::addressof(depth_coordinate_v2_map_compact_cs),
        std::addressof(depth_coordinate_v2_vertical_limit_cs),
        std::addressof(depth_coordinate_v2_limit_cs),
        normalized_depth_diagnostics_enabled ? nullptr : std::addressof(depth_coordinate_v2_vertical_limit_compact_cs),
        normalized_depth_diagnostics_enabled ? nullptr : std::addressof(depth_coordinate_v2_limit_compact_cs),
        std::addressof(ocr_preprocess_cs),
        std::addressof(ocr_box_cells_cs),
        std::addressof(ocr_box_resolve_cs),
        std::addressof(subtitle_locator_resolve_cs),
        std::addressof(subtitle_condition_cs),
        std::addressof(subtitle_contour_palette_cs),
        std::addressof(subtitle_contour_temporal_cs),
        std::addressof(subtitle_glyph_matte_cs),
        std::addressof(subtitle_contour_project_cs),
        std::addressof(subtitle_contour_history_cs),
        std::addressof(subtitle_contour_history_identity_cs),
        std::addressof(subtitle_contour_horizontal_cs),
        std::addressof(subtitle_contour_vertical_cs),
        std::addressof(subtitle_contour_snapshot_cs),
        std::addressof(subtitle_gaussian_vertical_limit_cs),
        std::addressof(subtitle_gaussian_horizontal_limit_cs),
      };
      static_assert(
        producer_shader_outputs.size() ==
        host_sbs_shader_cache::parallax_v2_producer_specs.size()
      );
      bool producer_shaders_ok = producer_sources && shader_identity_matches;
      for (std::size_t index = 0; index < producer_shader_outputs.size(); ++index) {
        if (!producer_shaders_ok) {
          break;
        }
        auto *output = producer_shader_outputs[index];
        if (output == std::addressof(near_identical_tiled_preprocess_cs) ||
            output == std::addressof(depth_coordinate_v2_map_compact_cs) ||
            output == std::addressof(depth_coordinate_v2_vertical_limit_compact_cs) ||
            output == std::addressof(depth_coordinate_v2_limit_compact_cs)) {
          // A device that cannot create a specialization uses its original shader route.
          if (!create_shader(producer_sources, host_sbs_shader_cache::parallax_v2_producer_specs[index], *output)) {
            output->Reset();
          }
          continue;
        }
        producer_shaders_ok = !output || create_shader(
          producer_sources,
          host_sbs_shader_cache::parallax_v2_producer_specs[index],
          *output
        );
      }
      if (!producer_shaders_ok) {
        BOOST_LOG(error)
          << "Host SBS V2 shader initialization or source-identity verification failed; "
             "Host SBS will fail flat.";
        return;
      }
      initialize_eye_limiter();

      // One fused preprocess dispatch runs for every root. Two immutable mode buffers make the
      // comparison decision explicit at b1 without rebuilding per-frame state: force/bootstrap
      // exits after canonical NCHW production, while GPU-undecided continues into tile evidence.
      const auto create_fused_mode_cbuffer = [&] (
                                                const std::uint32_t enabled,
                                                Microsoft::WRL::ComPtr<ID3D11Buffer> &output
                                              ) {
        const std::array<std::uint32_t, 4u> constants {enabled, 0u, 0u, 0u};
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.ByteWidth = static_cast<UINT>(sizeof(constants));
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA data {constants.data(), 0u, 0u};
        return SUCCEEDED(device->CreateBuffer(
          &desc, &data, output.ReleaseAndGetAddressOf()
        ));
      };
      if (!create_fused_mode_cbuffer(0u, fused_preprocess_force_cbuffer) ||
          !create_fused_mode_cbuffer(1u, fused_preprocess_compare_cbuffer)) {
        BOOST_LOG(error)
          << "Host SBS fused preprocess mode-buffer initialization failed; "
             "Host SBS will fail flat.";
        return;
      }

      parallax_v2_shader_provenance =
        std::make_shared<const parallax_v2_shader_provenance_t>(
          parallax_v2_shader_provenance_t {
            .source_closure_schema =
              depth_coordinate_v2::shader_source_closure_schema,
            .source_compile_flags =
              depth_coordinate_v2::shader_source_compile_flags,
            .source_macro_count =
              depth_coordinate_v2::shader_source_macro_count,
            .source_closure_sha256 = shader_source_closure_sha256,
          }
        );
      BOOST_LOG(info)
        << "Host SBS parallax-v2 GPU producer loaded for model '" << model.name
        << "' (fixed raw coordinate scale " << parallax_v2_raw_coordinate_scale
        << ", requested pop " << parallax_v2_requested_pop_strength
        << ", requested gain " << parallax_v2_requested_gain
        << "); completed fields remain separately authenticated before rendering.";
      BOOST_LOG(info)
        << "Host SBS RGB preprocess + near-identical tile fusion is active for every frame.";

      if (!cuda.has_conditional_graph_support()) {
        fail_gpu_conditional_bridge_once(
          "the installed CUDA driver lacks conditional-graph support"
        );
        return;
      }

      initialize_near_identical_detector();
      if (!near_identical_detector_available) {
        fail_gpu_conditional_bridge_once(
          "mandatory GPU detector/postprocess shader initialization failed"
        );
        return;
      }
      if (diagnostics_enabled) {
        initialize_gpu_trace();
      }

      // OCR is optional. Its authenticated source, engine and mutable execution context are
      // isolated from depth; an unavailable detector leaves the compact OCR8 record invalid and
      // the SLR13 conditioner copies BaseField exactly.
      ocr_available = initialize_ocr_context(assets_dir, cuda);
      if (!ocr_available) {
        BOOST_LOG(warning)
          << "PP-OCRv6 tiny is unavailable for this estimator; subtitle conditioning is flat.";
      }
      initialize_inference_done_events(cuda);
      // Raw normalization record, pre-seeded to
      // {min = 0xFFFFFFFF, max = 0, valid = 0, eligible = 0}.
      // The fused V2 frame resolve overwrites all four words before the histogram consumes them;
      // Frame publication resets the raw reduction identity after each inference.
      {
        uint32_t init_raw[4] = {0xFFFFFFFFu, 0u, 0u, 0u};
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.ByteWidth = sizeof(init_raw);
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        D3D11_SUBRESOURCE_DATA sd = {init_raw, 0, 0};
        device->CreateBuffer(&bd, &sd, &minmax_raw_buf);

        D3D11_UNORDERED_ACCESS_VIEW_DESC uav = {};
        uav.Format = DXGI_FORMAT_R32_TYPELESS;
        uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uav.Buffer.FirstElement = 0;
        uav.Buffer.NumElements = 4;
        uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        device->CreateUnorderedAccessView(minmax_raw_buf.Get(), &uav, &minmax_raw_uav);
      }

      // EMA'd P2/P98 bounds. initialized = 0 so the first frame seeds directly.
      {
        float init_ema[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.ByteWidth = sizeof(init_ema);
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = sizeof(float) * 4;
        D3D11_SUBRESOURCE_DATA sd = {init_ema, 0, 0};
        device->CreateBuffer(&bd, &sd, &minmax_ema_buf);
        device->CreateUnorderedAccessView(minmax_ema_buf.Get(), nullptr, &minmax_ema_uav);
        device->CreateShaderResourceView(minmax_ema_buf.Get(), nullptr, &minmax_ema_srv);
      }

      // The tagged depth-health state serves offline traces, Dump 3D, and live telemetry. Its
      // CPU-initialized tag is never rewritten on the GPU.
      {
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        const auto &init_state = sbs_adaptive_state::initial_words;
        bd.ByteWidth = sizeof(init_state);
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        bd.StructureByteStride = sizeof(float) * 4;
        D3D11_SUBRESOURCE_DATA sd2 = {init_state.data(), 0, 0};
        device->CreateBuffer(&bd, &sd2, &depth_health_state_buf);
        if (depth_health_state_buf) {
          device->CreateUnorderedAccessView(depth_health_state_buf.Get(), nullptr, &depth_health_state_uav);
          device->CreateShaderResourceView(depth_health_state_buf.Get(), nullptr, &depth_health_state_srv);
        }
      }

      const bool mandatory_resources_ready =
        engine && exec_context && cu_stream &&
        near_identical_fused_preprocess_cs &&
        fused_preprocess_force_cbuffer && fused_preprocess_compare_cbuffer &&
        (!normalized_depth_diagnostics_enabled || (buffer_to_tex_cs && buffer_to_tex_pad_cs)) &&
        depth_coordinate_v2_publish_cs &&
        minmax_raw_uav && minmax_ema_uav && minmax_ema_srv &&
        depth_health_state_uav && depth_health_state_srv;
      if (!mandatory_resources_ready) {
        BOOST_LOG(error) << "Depth estimator failed: required engine or Host SBS GPU resource initialization failed.";
        return;
      }

      // Constant buffers are created in ensure_cbuffers() once the model resolution is
      // known: every field is fixed for the session, so they are built once (immutable)
      // instead of being re-mapped on the encode thread every frame.

      // Warm up here so TensorRT's CUDA lazy kernel load / JIT (~20 s on the big models)
      // happens during construction -- which ensure_depth_estimator() runs on a background
      // thread -- rather than stalling the first real convert() on the encode thread and
      // freezing the stream when Host SBS first becomes active.
      if (!warmup_inference()) {
        BOOST_LOG(error) << "Depth estimator failed: TensorRT execution-context warmup failed.";
        return;
      }
      lifecycle_state = lifecycle_state_e::operational;
      if (!publish_active_engine_manifest(
            assets_dir, runtime_model.name, artifact, depth_engine_io_kind
          )) {
        BOOST_LOG(error) << "Could not publish the active TensorRT engine manifest for model '"
                         << runtime_model.name << "'.";
      }
      BOOST_LOG(info) << "Depth estimator pipeline initialized in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - init_started
                         )
                           .count()
                      << " ms";
    }

    // Run one throwaway inference at the engine's optimization shape so TensorRT loads its
    // CUDA modules now. The bulk of the "first inference" cost is module loading, which is
    // shape-independent, so a warmup at the OPT shape spares the first real frame the stall
    // even if its resolution differs. Uses its own scratch device buffers because the per-
    // frame D3D-interop buffers aren't allocated until convert() knows the frame resolution.
    // Pure CUDA + TensorRT (no D3D immediate context), so it's safe on the construction thread.
    bool warmup_inference() {
      if (!exec_context || !cu_stream) {
        return false;
      }
      if (depth_context_pooled) {
        return true;  // only successfully warmed contexts are admitted to the pool
      }
      auto &cuda = cuda_driver_api::get();
      if (!cuda.is_valid()) {
        return false;
      }
      if (!warmup_execution_context(cuda, cuda_ctx, exec_context)) {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        quarantine_execution_context_locked(engine_key, exec_context, false);
        return false;
      }
      {
        std::lock_guard<std::mutex> lock(g_trt_mutex);
        mark_execution_context_warmed_locked(engine_key);
      }
      context_warmed = true;
      return true;
    }

    ~impl() {
#ifdef SUNSHINE_TESTS
      // Release private callback before any existing bounded CUDA teardown query.
      // Queued userdata owns its state even if the old quarantine/retain path is taken.
      if (completion_test_gate) completion_test_gate->cancel_and_release();
#endif
      subtitle_direct_glyph.release();
      auto &cuda = cuda_driver_api::get();
      bool cleanup_execution_ok = cuda.is_valid() && cuda_ctx;
      const auto retain_all_cuda_backing = [&]() noexcept {
        // When CUDA teardown cannot positively release every handle, preserve every possible D3
        // operand rather than let ComPtr destruction create a driver-side use-after-free.
        (void) near_identical_transaction.uav.Detach();
        (void) near_identical_transaction.buffer.Detach();
        (void) tensor_in_srv.Detach();
        (void) tensor_in_uav.Detach();
        (void) tensor_in_buf.Detach();
        (void) tensor_out_srv.Detach();
        (void) tensor_out_buf.Detach();
        (void) ocr_input_uav.Detach();
        (void) ocr_input_buf.Detach();
        (void) ocr_output_srv.Detach();
        (void) ocr_output_buf.Detach();
      };
      if (cuda.is_valid() && cuda_ctx) {
        const auto set_current = cuda.cuCtxSetCurrent(cuda_ctx);
        if (set_current != CUDA_SUCCESS) {
          cleanup_execution_ok = false;
          BOOST_LOG(warning)
            << "cuCtxSetCurrent failed during depth-estimator teardown: " << set_current;
          retain_all_cuda_backing();
          depth_conditional_graph.abandon_unsafe();
        } else {
          bool stream_quiesced = !cu_stream;
          if (cu_stream) {
            if (cuda.cuStreamQuery) {
              // Teardown runs on the lifecycle worker, not the presenter. Allow a weak GPU up to a
              // quarter second to finish one already-submitted root before choosing the deliberate
              // retain/quarantine path; 1 ms pacing bounds driver calls and avoids a busy spin.
              constexpr auto teardown_quiescence_budget =
                std::chrono::milliseconds {250};
              const auto deadline =
                std::chrono::steady_clock::now() + teardown_quiescence_budget;
              CUresult last_query = CUDA_ERROR_NOT_READY;
              unsigned int query_count = 0u;
              while (true) {
                last_query = cuda.cuStreamQuery(cu_stream);
                ++query_count;
                const auto readiness = normalized_cuda_readiness(last_query);
                const auto action = detail::teardown_quiescence_action(
                  readiness,
                  std::chrono::steady_clock::now() < deadline
                );
                if (action == detail::teardown_quiescence_action_e::release_operands) {
                  stream_quiesced = true;
                  break;
                }
                if (action == detail::teardown_quiescence_action_e::retain_operands) {
                  cleanup_execution_ok = false;
                  BOOST_LOG(warning)
                    << "Depth-estimator teardown could not prove stream quiescence after "
                    << query_count << " nonblocking query attempt(s): result=" << last_query
                    << (readiness == detail::async_stream_readiness_e::busy ?
                          " (250 ms deadline expired)." : " (CUDA query failed).");
                  break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds {1});
              }
            } else {
              cleanup_execution_ok = false;
              BOOST_LOG(warning)
                << "Depth-estimator teardown cannot query stream quiescence; retaining all CUDA "
                   "operands.";
            }
          }
          if (!stream_quiesced) {
            // A failed query may be the launch fault being reported; a timeout still leaves work
            // potentially live. Neither proves that graph operands are idle. Retain the wrapper,
            // source graphs, stream, registrations, and their D3 backing.
            retain_all_cuda_backing();
            depth_conditional_graph.abandon_unsafe();
          } else {
            // Every CUDA teardown call may be the first API to report a deferred root fault. The
            // chain is monotonic: after the first failure, retain every not-yet-proven handle and
            // D3 operand and issue no later CUDA destroy/unregister call.
            bool cuda_teardown_may_continue = true;
            std::string_view failed_teardown_operation;
            const auto observe_teardown = [&](const bool succeeded, const std::string_view name) {
              const bool next = detail::cuda_teardown_chain_may_continue(
                cuda_teardown_may_continue,
                succeeded
              );
              if (cuda_teardown_may_continue && !next) {
                failed_teardown_operation = name;
              }
              cuda_teardown_may_continue = next;
            };

            // A previously failed context restore is already a failed first teardown operation.
            // Do not use that context for even an otherwise independent event destroy.
            observe_teardown(
              !gpu_conditional_bridge_context_failed,
              "conditional wrapper context state"
            );
            if (cuda_teardown_may_continue) {
              observe_teardown(
                perf_depth_conditional.destroy_events(cuda),
                "diagnostic event destruction"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                root_completion_notification.destroy_stream(cuda),
                "inference notification stream destruction"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                destroy_inference_done_events(cuda),
                inference_event_ever_recorded ?
                  "recorded inference event destruction" :
                  "unrecorded inference event destruction"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                unregister_near_identical_decision_interop(cuda),
                "conditional wrapper/decision interop release"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                destroy_inference_graph(cuda, depth_inference_graph),
                "DAV2 source graph destruction"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                destroy_inference_graph(cuda, ocr_inference_graph),
                "OCR source graph destruction"
              );
            }
            if (cuda_teardown_may_continue && cu_stream) {
              const CUresult stream_destroyed = cuda.cuStreamDestroy ?
                                                  cuda.cuStreamDestroy(cu_stream) :
                                                  static_cast<CUresult>(-1);
              observe_teardown(
                stream_destroyed == CUDA_SUCCESS,
                "joined stream destruction"
              );
              if (cuda_teardown_may_continue) {
                cu_stream = nullptr;
              }
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                unregister_cuda_graphics_resource(cuda, cuda_in_res),
                "DAV2 input interop unregister"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                unregister_cuda_graphics_resource(cuda, cuda_out_res),
                "DAV2 output interop unregister"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                unregister_cuda_graphics_resource(cuda, cuda_ocr_in_res),
                "OCR input interop unregister"
              );
            }
            if (cuda_teardown_may_continue) {
              observe_teardown(
                unregister_cuda_graphics_resource(cuda, cuda_ocr_out_res),
                "OCR output interop unregister"
              );
            }

            if (!cuda_teardown_may_continue) {
              cleanup_execution_ok = false;
              retain_all_cuda_backing();
              depth_conditional_graph.abandon_unsafe();
              BOOST_LOG(warning)
                << "Depth-estimator CUDA teardown stopped after "
                << failed_teardown_operation
                << " failed; retaining all remaining CUDA/D3 operands.";
            }
          }
        }
      } else {
        retain_all_cuda_backing();
        depth_conditional_graph.abandon_unsafe();
      }
      if (!cleanup_execution_ok) {
        // Any teardown API can be the first call to surface a deferred launch fault.
        // CUDA can surface one stream's earlier launch fault through another stream's cleanup.
        // Never give either participating context to a later estimator in that case.
        execution_context_poisoned = true;
        if (joined_stream_ocr_ever_submitted_or_armed) {
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::unsafe_teardown
          );
        }
      }

      // Return only a successfully warmed context to the reusable pool. Construction can fail
      // after createExecutionContext() but before warmup (shader/resource allocation is one such
      // path); admitting that context would make the next instance skip the failed lazy load.
      // Contexts cannot be destroyed safely across the DLL boundary, so quarantine them instead.
      std::lock_guard<std::mutex> lock(g_trt_mutex);
      if (exec_context) {
        if (execution_context_poisoned) {
          BOOST_LOG(warning)
            << "Quarantining a TensorRT execution context after CUDA/TensorRT execution or "
               "teardown failure.";
        }
        recycle_or_quarantine_execution_context_locked(
          engine_key,
          exec_context,
          context_warmed,
          execution_context_poisoned
        );
      }
      if (ocr_exec_context) {
        if (ocr_context_health.poisoned()) {
          BOOST_LOG(warning)
            << "Quarantining the PP-OCRv6 tiny execution context after asynchronous GPU "
               "execution or unsafe teardown.";
        }
        recycle_or_quarantine_execution_context_locked(
          ocr_engine_key,
          ocr_exec_context,
          ocr_context_warmed,
          ocr_context_health.poisoned()
        );
      }
      // TRT runtime/engines are cached globally, do not destroy them here.
    }

    void mark_d3d_parallax_start(d3d_perf_slot *slot) {
      if (slot) {
        slot->has_parallax = true;
        context->End(slot->parallax_start.Get());
      }
    }

    void mark_d3d_parallax_frame_stats_start(d3d_perf_slot *slot) {
      if (slot) {
        slot->has_parallax_frame_stats = true;
        context->End(slot->parallax_frame_stats_start.Get());
      }
    }

    void mark_d3d_parallax_frame_stats_end(d3d_perf_slot *slot) {
      if (slot && slot->has_parallax_frame_stats) {
        context->End(slot->parallax_frame_stats_end.Get());
      }
    }

    void mark_d3d_parallax_end(d3d_perf_slot *slot) {
      if (slot && slot->has_parallax) {
        context->End(slot->parallax_end.Get());
      }
    }

    void mark_d3d_parallax_subtitle_start(d3d_perf_slot *slot) {
      if (slot && slot->has_parallax) {
        slot->has_parallax_subtitle_start = true;
        context->End(slot->parallax_subtitle_start.Get());
      }
    }

    void release_parallax_v2_resources() {
      invalidate_publication();
      parallax_v2_producer_active = false;
      subtitle_contour_source.reset();
      subtitle_contour_previous_features.reset();
      subtitle_contour_next_features.reset();
      subtitle_contour_support.reset();
      subtitle_glyph_matte.reset();
      subtitle_contour_identity.reset();
      subtitle_contour_source_snapshot.reset();
      subtitle_contour_probability_snapshot.reset();
      subtitle_contour_observation.reset();
      subtitle_conditioning_diagnostics.reset();
      completed_contour = {};
      completed_subtitle_conditioning_diagnostics = false;
      subtitle_direct_glyph.clear_reported_failure();
      subtitle_selected_matte_commands = false;
      subtitle_contour_cbuffer.Reset();
      depth_coordinate_v2_cbuffer.Reset();
      depth_coordinate_v2_partials_buf.Reset();
      depth_coordinate_v2_partials_uav.Reset();
      depth_coordinate_v2_partials_srv.Reset();
      depth_coordinate_v2_frame_stats_buf.Reset();
      depth_coordinate_v2_frame_stats_uav.Reset();
      depth_coordinate_v2_frame_stats_srv.Reset();
      depth_coordinate_v2_histogram_buf.Reset();
      depth_coordinate_v2_histogram_uav.Reset();
      depth_coordinate_v2_histogram_srv.Reset();
      depth_coordinate_v2_state_buf.Reset();
      depth_coordinate_v2_state_uav.Reset();
      depth_coordinate_v2_state_srv.Reset();
      depth_coordinate_v2_coordinate_tex.Reset();
      depth_coordinate_v2_coordinate_uav.Reset();
      depth_coordinate_v2_coordinate_srv.Reset();
      depth_coordinate_v2_candidate_tex.Reset();
      depth_coordinate_v2_candidate_uav.Reset();
      depth_coordinate_v2_candidate_srv.Reset();
      depth_coordinate_v2_vertical_majorant_tex.Reset();
      depth_coordinate_v2_vertical_majorant_uav.Reset();
      depth_coordinate_v2_vertical_majorant_srv.Reset();
      depth_coordinate_v2_vertical_conditioned_tex.Reset();
      depth_coordinate_v2_vertical_conditioned_uav.Reset();
      depth_coordinate_v2_vertical_conditioned_srv.Reset();
      depth_coordinate_v2_final_tex.Reset();
      depth_coordinate_v2_final_uav.Reset();
      depth_coordinate_v2_final_srv.Reset();
      depth_coordinate_v2_eye_final_tex.Reset();
      depth_coordinate_v2_eye_final_uav.Reset();
      depth_coordinate_v2_eye_final_srv.Reset();
      subtitle_locator_state_buf.Reset();
      subtitle_locator_state_uav.Reset();
      subtitle_locator_state_srv.Reset();
      subtitle_condition_params_buf.Reset();
      subtitle_condition_params_uav.Reset();
      subtitle_condition_params_srv.Reset();
      subtitle_locator_cbuffer.Reset();
    }
    void fail_parallax_v2_producer(std::string_view reason) {
      // The lazy resource initializer is the sole caller, and its sole caller immediately
      // latches the terminal lifecycle state on false. Keep teardown here without a parallel
      // failure flag.
      BOOST_LOG(error) << "Host SBS V2 producer failed: " << reason
                       << "; this stream will remain live flat identity.";
      release_parallax_v2_resources();
    }

    // The pending observation's inputs to the Direct Glyph step; only dispatch reads them.
    [[nodiscard]] host_sbs_direct_glyph_route::frame_t subtitle_direct_glyph_frame(const bool input_domain_reset) const {
      return {
        .region = pending_root->input_region,
        .frame = pending_root->frame_id,
        .observation_us = pending_root->observation_timestamp_us,
        .token = pending_root->transaction_token,
        .subtitle_context_generation = pending_root->subtitle_context.generation,
        .color_space = static_cast<UINT>(pending_root->color_space),
        .video_caption = pending_root->subtitle_line_policy == subtitle_line_policy_e::video_caption,
        .postprocess_mode = pending_near_identical_postprocess_mode(),
        .expected_work = cuda_conditional_graph::work_flags_value(pending_root->subtitle_work),
        .input_domain_reset = input_domain_reset,
        .display_snapshot_band = subtitle_contour_source_snapshot.srv.Get(),
        .parent_raw_source = pending_root->submitted_source.Get(),
        .contour_constants = subtitle_contour_cbuffer.Get(),
        .contour_identity = subtitle_contour_identity.srv.Get(),
        .root_dispatch = near_identical_transaction.dispatch.Get(),
        .root = near_identical_transaction.srv.Get(),
        .ocr_box_record = ocr_box_record_srv.Get(),
        .condition_params = subtitle_condition_params_srv.Get(),
        .locator_state = subtitle_locator_state_srv.Get(),
      };
    }

    void dispatch_subtitle_direct_glyph(const bool input_domain_reset) {
      if (const auto published = subtitle_direct_glyph.dispatch(
            device.Get(), context.Get(), subtitle_direct_glyph_frame(input_domain_reset)
          )) {
        subtitle_selected_matte_commands = *published;
      }
    }

    bool retire_subtitle_direct_glyph(const bool evaluation_wait) {
      const auto retirement = subtitle_direct_glyph.retire(context.Get(), evaluation_wait);
      if (retirement.selected_matte_commands) {
        subtitle_selected_matte_commands = *retirement.selected_matte_commands;
      }
      return retirement.retired;
    }

    bool prepare_subtitle_direct_glyph(const subtitle_direct_glyph_configuration_t &cfg, std::string &error) {
      return subtitle_direct_glyph.prepare(
        device.Get(), context.Get(), cfg, shader_root.parent_path().parent_path(), cuda_device, cuda_ctx, error
      );
    }

    bool create_shader_texture(
      const UINT width,
      const UINT height,
      const DXGI_FORMAT format,
      d3d_texture_views_t &views
    ) {
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = 1u;
      desc.ArraySize = 1u;
      desc.Format = format;
      desc.SampleDesc.Count = 1u;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      return SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &views.texture)) &&
             SUCCEEDED(device->CreateShaderResourceView(views.texture.Get(), nullptr, &views.srv)) &&
             SUCCEEDED(device->CreateUnorderedAccessView(views.texture.Get(), nullptr, &views.uav));
    }

    bool ensure_subtitle_contour_texture(
      const UINT width,
      const UINT height,
      const DXGI_FORMAT format,
      d3d_texture_views_t &views
    ) {
      if (views.ready()) {
        D3D11_TEXTURE2D_DESC previous {};
        views.texture->GetDesc(&previous);
        if (previous.Width == width && previous.Height == height && previous.Format == format) {
          return true;
        }
      }
      d3d_texture_views_t next;
      if (!create_shader_texture(width, height, format, next)) {
        return false;
      }
      const UINT zero[4] {};
      context->ClearUnorderedAccessViewUint(next.uav.Get(), zero);
      views = std::move(next);
      return true;
    }

    bool ensure_subtitle_contour_textures(
      const std::uint32_t source_width,
      const std::uint32_t source_height
    ) {
      const auto crop_height = subtitle_ocr_source_crop_height(source_width, source_height);
      const bool crop_valid = source_width != 0u &&
                              source_width <= depth_coordinate_v2::subtitle_contour_source_max_extent &&
                              source_height <= depth_coordinate_v2::subtitle_contour_source_max_extent &&
                              crop_height != 0u &&
                              crop_height <= source_height;
      // Ineligible inputs need only a placeholder. The classifier publishes no support unless
      // its exact source/crop dimensions and current OCR authority validate together.
      const UINT native_width = crop_valid ? source_width : 1u;
      const UINT native_height = crop_valid ? crop_height : 1u;
      // References come directly from the source band and the feature codes are exact in FP16.
      constexpr auto native_float_format = DXGI_FORMAT_R16G16B16A16_FLOAT;
      return field_w > 0 && field_h > 0 &&
             ensure_subtitle_contour_texture(native_width, native_height, native_float_format,
                                             subtitle_contour_source) &&
             ensure_subtitle_contour_texture(native_width, native_height, native_float_format,
                                             subtitle_contour_source_snapshot) &&
             (!subtitle_legacy_matte_required() ||
              ensure_subtitle_contour_texture(native_width, native_height, DXGI_FORMAT_R32_UINT,
                                              subtitle_glyph_matte));
    }

    bool ensure_subtitle_dump_matte() {
      // Admission has already sized the source band, including ineligible placeholders.
      D3D11_TEXTURE2D_DESC band {};
      subtitle_contour_source.texture->GetDesc(&band);
      return ensure_subtitle_contour_texture(band.Width, band.Height, DXGI_FORMAT_R32_UINT,
                                             subtitle_glyph_matte);
    }

    // temporal_main publishes native support and the RGB feature history. Only the dense legacy
    // matte, the conditioned-field diagnostics (including every Dump 3D) and evaluation exports
    // read them. Allocate them with the current source band on the first observation that needs
    // them and release them once one does not, so a later request starts its history at count one
    // instead of reading stale features.
    [[nodiscard]] bool subtitle_contour_history_required(const bool legacy_matte) const noexcept {
      return legacy_matte;
    }

    bool ensure_subtitle_contour_history() {
      D3D11_TEXTURE2D_DESC band {};
      subtitle_contour_source.texture->GetDesc(&band);
      return ensure_subtitle_contour_texture(band.Width, band.Height, band.Format,
                                             subtitle_contour_previous_features) &&
             ensure_subtitle_contour_texture(band.Width, band.Height, band.Format,
                                             subtitle_contour_next_features) &&
             ensure_subtitle_contour_texture(band.Width, band.Height, DXGI_FORMAT_R32_UINT,
                                             subtitle_contour_support);
    }

    void release_subtitle_contour_history() {
      subtitle_contour_previous_features.reset();
      subtitle_contour_next_features.reset();
      subtitle_contour_support.reset();
    }

    // A new estimator and every hard-excluded admission publish initialized one-pixel native-band
    // placeholders, native support and feature history included. Replacing an eligible band
    // releases its backing at the admitted context boundary; the next observation that reads the
    // history resizes and clears it, so its count starts at one.
    bool ensure_subtitle_contour_placeholders() {
      return ensure_subtitle_contour_textures(1u, 1u) && ensure_subtitle_contour_history();
    }

    bool reset_subtitle_state() {
      invalidate_publication();
      subtitle_selected_matte_commands = false;
      // The preceding joined root and its consumers must be retired first. These are the same
      // zero-initialized subtitle resources used at startup; depth and camera stay intact.
      const UINT zero[4] = {};
      for (auto *uav : {
             ocr_box_record_uav.Get(), subtitle_locator_state_uav.Get(),
             subtitle_condition_params_uav.Get(), subtitle_contour_identity.uav.Get(),
             subtitle_contour_probability_snapshot.uav.Get(),
             subtitle_contour_source.uav.Get(), subtitle_contour_source_snapshot.uav.Get(),
             subtitle_contour_previous_features.uav.Get(), subtitle_contour_next_features.uav.Get(),
             subtitle_contour_support.uav.Get(),
             subtitle_legacy_matte_required() ? subtitle_glyph_matte.uav.Get() : nullptr,
           }) {
        if (uav) {
          context->ClearUnorderedAccessViewUint(uav, zero);
        }
      }
      if (const auto &diagnostics = subtitle_conditioning_diagnostics) {
        clear_subtitle_conditioning_diagnostics(*diagnostics);
      }
      completed_subtitle_conditioning_diagnostics = false;
      // Disabled completions may expose these initialized views only as unused diagnostics.
      // Reentry cannot retain snapshots belonging to a prior source extent or subtitle context.
      completed_contour = {
        subtitle_contour_support.srv, subtitle_contour_identity.srv,
        subtitle_contour_previous_features.srv, subtitle_contour_source_snapshot.srv,
        subtitle_contour_probability_snapshot.srv,
        subtitle_legacy_matte_required() ? subtitle_glyph_matte.srv : nullptr,
        subtitle_condition_params_srv
      };
      return true;
    }

    bool create_shader_uint_buffer(
      const std::uint32_t word_count,
      Microsoft::WRL::ComPtr<ID3D11Buffer> &buffer,
      Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> &srv,
      Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> &uav,
      const std::uint32_t words_per_element = 1u
    ) {
      D3D11_BUFFER_DESC desc {};
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.ByteWidth = word_count * sizeof(std::uint32_t);
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      desc.StructureByteStride = words_per_element * sizeof(std::uint32_t);
      return SUCCEEDED(device->CreateBuffer(&desc, nullptr, &buffer)) &&
             SUCCEEDED(device->CreateShaderResourceView(buffer.Get(), nullptr, &srv)) &&
             SUCCEEDED(device->CreateUnorderedAccessView(buffer.Get(), nullptr, &uav));
    }

    bool ensure_subtitle_constant_buffer(
      const UINT byte_width,
      Microsoft::WRL::ComPtr<ID3D11Buffer> &buffer
    ) {
      if (buffer) {
        return true;
      }
      D3D11_BUFFER_DESC desc {};
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.ByteWidth = byte_width;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      return SUCCEEDED(device->CreateBuffer(&desc, nullptr, &buffer));
    }

    bool ensure_subtitle_resources() {
      invalidate_publication();
      bool resources_ok = true;
      if (!ocr_box_record_buf || !ocr_box_record_srv || !ocr_box_record_uav) {
        ocr_box_record_buf.Reset();
        ocr_box_record_srv.Reset();
        ocr_box_record_uav.Reset();
        resources_ok = create_shader_uint_buffer(
          ocr_box_record_word_count,
          ocr_box_record_buf,
          ocr_box_record_srv,
          ocr_box_record_uav
        );
      }
      if (!subtitle_locator_state_buf || !subtitle_locator_state_srv ||
          !subtitle_locator_state_uav) {
        subtitle_locator_state_buf.Reset();
        subtitle_locator_state_srv.Reset();
        subtitle_locator_state_uav.Reset();
        resources_ok = resources_ok && create_shader_uint_buffer(
          subtitle_locator_state_word_count,
          subtitle_locator_state_buf,
          subtitle_locator_state_srv,
          subtitle_locator_state_uav
        );
      }
      if (!subtitle_condition_params_buf || !subtitle_condition_params_srv ||
          !subtitle_condition_params_uav) {
        subtitle_condition_params_buf.Reset();
        subtitle_condition_params_srv.Reset();
        subtitle_condition_params_uav.Reset();
        resources_ok = resources_ok && create_shader_uint_buffer(
          subtitle_condition_param_word_count,
          subtitle_condition_params_buf,
          subtitle_condition_params_srv,
          subtitle_condition_params_uav
        );
      }
      // One observation header, plus three rectangles and the palette for each cover.
      constexpr auto contour_observation_records = 1u + depth_coordinate_v2::subtitle_plane_cover_capacity *
                                                   (4u + depth_coordinate_v2::subtitle_contour_palette_max_modes);
      // An exact-current abstention needs the OCR identity constants even when the optional
      // detector never acquired an engine or its interop buffers.
      resources_ok = resources_ok &&
        ensure_subtitle_constant_buffer(sizeof(ocr_resolve_constants_t), ocr_resolve_cbuffer) &&
        ensure_subtitle_constant_buffer(80u, subtitle_locator_cbuffer) &&
        ensure_subtitle_constant_buffer(96u, subtitle_contour_cbuffer) &&
        create_shader_uint_buffer(depth_coordinate_v2::subtitle_contour_history_word_count,
                                    subtitle_contour_identity.buffer,
                                    subtitle_contour_identity.srv, subtitle_contour_identity.uav) &&
        create_shader_uint_buffer(depth_coordinate_v2::subtitle_ocr_output_width *
                                      depth_coordinate_v2::subtitle_ocr_output_height,
                                    subtitle_contour_probability_snapshot.buffer,
                                    subtitle_contour_probability_snapshot.srv,
                                    subtitle_contour_probability_snapshot.uav) &&
        (subtitle_contour_observation.ready() ||
         create_shader_uint_buffer(4u * contour_observation_records,
                                     subtitle_contour_observation.buffer,
                                     subtitle_contour_observation.srv,
                                     subtitle_contour_observation.uav, 4u)) &&
        ensure_subtitle_contour_placeholders();
      if (!resources_ok) {
        return false;
      }

      if (!reset_subtitle_state()) return false;

      if (!ocr_available) {
        return true;
      }
      if (!ocr_local_candidate_scratch_buf || !ocr_local_candidate_scratch_uav) {
        ocr_local_candidate_scratch_buf.Reset();
        ocr_local_candidate_scratch_uav.Reset();
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = depth_coordinate_v2::subtitle_ocr_local_candidate_capacity *
                         depth_coordinate_v2::subtitle_ocr_local_candidate_word_count *
                         sizeof(std::uint32_t);
        desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = 4u * sizeof(std::uint32_t);
        if (FAILED(device->CreateBuffer(&desc, nullptr, &ocr_local_candidate_scratch_buf)) ||
            FAILED(device->CreateUnorderedAccessView(
              ocr_local_candidate_scratch_buf.Get(), nullptr,
              &ocr_local_candidate_scratch_uav))) {
          return false;
        }
      }
      if (cuda_ocr_in_res && cuda_ocr_out_res && ocr_input_uav && ocr_output_srv &&
          ocr_cell_stats_srv && ocr_cell_stats_uav && ocr_preprocess_cbuffer &&
          ocr_resolve_cbuffer) {
        return true;
      }

      auto create_float_buffer = [&](
                                   const std::uint32_t float_count,
                                    const UINT bind_flags,
                                    Microsoft::WRL::ComPtr<ID3D11Buffer> &buffer,
                                    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> *srv,
                                    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> *uav
                                 ) {
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = float_count * sizeof(float);
        desc.BindFlags = bind_flags;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);
        return SUCCEEDED(device->CreateBuffer(&desc, nullptr, &buffer)) &&
               (!srv || SUCCEEDED(device->CreateShaderResourceView(
                          buffer.Get(), nullptr,
                          srv->ReleaseAndGetAddressOf()
                        ))) &&
               (!uav || SUCCEEDED(device->CreateUnorderedAccessView(
                          buffer.Get(), nullptr,
                          uav->ReleaseAndGetAddressOf()
                        )));
      };

      const std::uint32_t ocr_pixels = ocr_engine_width * ocr_engine_height;
      const bool ocr_input_resources_ok = create_float_buffer(
        3u * ocr_pixels,
        D3D11_BIND_UNORDERED_ACCESS,
        ocr_input_buf,
        nullptr,
        &ocr_input_uav
      );
      bool ocr_resources_ok = ocr_input_resources_ok &&
                              create_float_buffer(
                                ocr_pixels,
                                D3D11_BIND_SHADER_RESOURCE,
                                ocr_output_buf,
                                &ocr_output_srv,
                                nullptr
                              ) &&
                              create_shader_uint_buffer(
                                ocr_cell_stats_word_count,
                                ocr_cell_stats_buf,
                                ocr_cell_stats_srv,
                                ocr_cell_stats_uav
                              ) &&
                              ensure_subtitle_constant_buffer(32u, ocr_preprocess_cbuffer);
      auto &cuda = cuda_driver_api::get();
      bool ocr_interop_teardown_failed = false;
      const auto unregister_ocr_interop = [&]() {
        // Do not short-circuit: every successfully registered member must be released before any
        // backing D3D resource can be replaced.
        const bool input_unregistered =
          unregister_cuda_graphics_resource(cuda, cuda_ocr_in_res);
        const bool output_unregistered =
          unregister_cuda_graphics_resource(cuda, cuda_ocr_out_res);
        return input_unregistered && output_unregistered;
      };
      if (ocr_resources_ok && cuda_ctx && cuda.cuCtxSetCurrent(cuda_ctx) == CUDA_SUCCESS) {
        CUresult input_status = static_cast<CUresult>(-1);
        CUresult output_status = static_cast<CUresult>(-1);
        const auto register_ocr_interop = [&]() {
          input_status = cuda.cuGraphicsD3D11RegisterResource(
            &cuda_ocr_in_res, ocr_input_buf.Get(), 0
          );
          output_status = cuda.cuGraphicsD3D11RegisterResource(
            &cuda_ocr_out_res, ocr_output_buf.Get(), 0
          );
          return input_status == CUDA_SUCCESS && output_status == CUDA_SUCCESS &&
                 cuda_ocr_in_res && cuda_ocr_out_res;
        };
        ocr_resources_ok = register_ocr_interop();
        if (!ocr_resources_ok) {
          BOOST_LOG(warning)
            << "PP-OCRv6 tiny D3D11/CUDA registration failed (" << input_status << ", "
            << output_status << "); subtitle conditioning remains flat.";
        }
      } else {
        ocr_resources_ok = false;
      }
      if (!ocr_resources_ok) {
        const bool interop_released = unregister_ocr_interop();
        if (!interop_released && !ocr_interop_teardown_failed) {
          ocr_interop_teardown_failed = true;
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::unsafe_teardown
          );
        }
        if (interop_released) {
          ocr_input_buf.Reset();
          ocr_input_uav.Reset();
          ocr_output_buf.Reset();
          ocr_output_srv.Reset();
        } else {
          BOOST_LOG(error)
            << "PP-OCRv6 tiny interop cleanup did not release every CUDA resource; retaining "
               "the backing D3D resources and quarantining the OCR context.";
        }
        ocr_cell_stats_buf.Reset();
        ocr_cell_stats_uav.Reset();
        ocr_cell_stats_srv.Reset();
        ocr_available = false;
      }
      return true;
    }

    bool ensure_parallax_v2_resources() {
      using namespace depth_coordinate_v2;
      if (parallax_v2_producer_active) {
        return true;
      }
      invalidate_publication();
      if (target_w <= 0 || target_h <= 0 || field_w <= 0 || field_h <= 0 ||
          reduce_groups == 0) {
        std::ostringstream reason;
        reason << "invalid model/field/reduction dimensions model="
               << target_w << 'x' << target_h << ", field="
               << field_w << 'x' << field_h << ", reduce_groups=" << reduce_groups;
        fail_parallax_v2_producer(reason.str());
        return false;
      }
      if (field_w != target_w || field_h != target_h || !production_depth_runtime_active()) {
        fail_parallax_v2_producer("single-grid model/field identity is invalid"sv);
        return false;
      }
      if (!prod_zipdepth_convex2x::live_geometry_field_shape_is_supported(
            static_cast<std::uint32_t>(field_w),
            static_cast<std::uint32_t>(field_h)
          )) {
        std::ostringstream reason;
        reason << "single-high field " << field_w << 'x' << field_h
               << " is outside the exact fused profile allowlist";
        fail_parallax_v2_producer(reason.str());
        return false;
      }
      const auto calibration_shape = calibrated_dav2_shape();
      if (!raw_model_provenance || !calibration_shape ||
          !depth_coordinate_v2::capture_identity_is_calibrated(raw_model_provenance->depth_model, raw_model_provenance->depth_model_url,
            raw_model_provenance->onnx_sha256,
            raw_model_provenance->preprocess_profile,
            raw_model_provenance->preprocess_source_closure_sha256,
            static_cast<std::uint32_t>(calibration_shape->width),
            static_cast<std::uint32_t>(calibration_shape->height))) {
        std::ostringstream reason;
        reason << "model identity, preprocess profile/source closure, or input shape "
               << target_w << 'x' << target_h
               << " is outside the calibrated allowlist";
        fail_parallax_v2_producer(reason.str());
        return false;
      }

      auto create_float4_buffer = [&](
                                    std::size_t vector_count,
                                    const void *initial_data,
                                    Microsoft::WRL::ComPtr<ID3D11Buffer> &buffer,
                                    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> &srv,
                                    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> &uav
                                  ) {
        const std::size_t byte_count = vector_count * sizeof(float) * 4u;
        if (byte_count == 0 || byte_count > std::numeric_limits<UINT>::max()) {
          return false;
        }
        D3D11_BUFFER_DESC desc {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = static_cast<UINT>(byte_count);
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float) * 4u;
        D3D11_SUBRESOURCE_DATA init {initial_data, 0, 0};
        return SUCCEEDED(device->CreateBuffer(
                 &desc,
                 initial_data ? &init : nullptr,
                 &buffer
               )) &&
               SUCCEEDED(device->CreateShaderResourceView(buffer.Get(), nullptr, &srv)) &&
               SUCCEEDED(device->CreateUnorderedAccessView(buffer.Get(), nullptr, &uav));
      };

      bool resources_ok = create_float4_buffer(
                            static_cast<std::size_t>(reduce_groups) * 3u,
                            nullptr,
                            depth_coordinate_v2_partials_buf,
                            depth_coordinate_v2_partials_srv,
                            depth_coordinate_v2_partials_uav
                          ) &&
                          create_float4_buffer(
                            frame_stats_vector_count,
                            nullptr,
                            depth_coordinate_v2_frame_stats_buf,
                            depth_coordinate_v2_frame_stats_srv,
                            depth_coordinate_v2_frame_stats_uav
                          );
      resources_ok = resources_ok && create_float4_buffer(
        static_cast<std::size_t>(reduce_groups) * 128u,
        nullptr,
        depth_coordinate_v2_histogram_buf,
        depth_coordinate_v2_histogram_srv,
        depth_coordinate_v2_histogram_uav
      );

      resources_ok = resources_ok &&
                     create_float4_buffer(
                       state_vector_count,
                       state_initial_words.data(),
                       depth_coordinate_v2_state_buf,
                       depth_coordinate_v2_state_srv,
                       depth_coordinate_v2_state_uav
                     );

      D3D11_TEXTURE2D_DESC texture_desc {};
      texture_desc.Width = static_cast<UINT>(field_w);
      texture_desc.Height = static_cast<UINT>(field_h);
      texture_desc.MipLevels = 1;
      texture_desc.ArraySize = 1;
      texture_desc.Format = DXGI_FORMAT_R32_FLOAT;
      texture_desc.SampleDesc.Count = 1;
      texture_desc.Usage = D3D11_USAGE_DEFAULT;
      texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      auto create_float_texture = [&](
                                    Microsoft::WRL::ComPtr<ID3D11Texture2D> &texture,
                                  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> *srv,
                                  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> &uav
                                  ) {
        return SUCCEEDED(device->CreateTexture2D(&texture_desc, nullptr, &texture)) &&
               (!srv || SUCCEEDED(device->CreateShaderResourceView(
                           texture.Get(),
                           nullptr,
                           srv->ReleaseAndGetAddressOf()
                         ))) &&
               SUCCEEDED(device->CreateUnorderedAccessView(texture.Get(), nullptr, &uav));
      };
      resources_ok = resources_ok &&
                     create_float_texture(
                        depth_coordinate_v2_candidate_tex,
                        &depth_coordinate_v2_candidate_srv,
                        depth_coordinate_v2_candidate_uav
                      ) &&
                     create_float_texture(
                       depth_coordinate_v2_vertical_majorant_tex,
                       &depth_coordinate_v2_vertical_majorant_srv,
                       depth_coordinate_v2_vertical_majorant_uav
                     ) &&
                     create_float_texture(
                       depth_coordinate_v2_vertical_conditioned_tex,
                       &depth_coordinate_v2_vertical_conditioned_srv,
                       depth_coordinate_v2_vertical_conditioned_uav
                     ) &&
                     create_float_texture(
                       depth_coordinate_v2_final_tex,
                       &depth_coordinate_v2_final_srv,
                       depth_coordinate_v2_final_uav
                     );
      if (resources_ok && depth_coordinate_v2_eye_limit_cs) {
        // Twice the field height: left-eye rows, then right-eye rows. A failure here only loses
        // the research field; the Base view then has no companion and renders as before.
        auto eye_desc = texture_desc;
        eye_desc.Height *= 2u;
        const bool eye_ok =
          SUCCEEDED(device->CreateTexture2D(&eye_desc, nullptr, &depth_coordinate_v2_eye_final_tex)) &&
          SUCCEEDED(device->CreateShaderResourceView(
            depth_coordinate_v2_eye_final_tex.Get(),
            nullptr,
            &depth_coordinate_v2_eye_final_srv
          )) &&
          SUCCEEDED(device->CreateUnorderedAccessView(
            depth_coordinate_v2_eye_final_tex.Get(),
            nullptr,
            &depth_coordinate_v2_eye_final_uav
          )) &&
          host_sbs_eye_field::attach(depth_coordinate_v2_final_srv.Get(), depth_coordinate_v2_eye_final_srv.Get());
        if (!eye_ok) {
          depth_coordinate_v2_eye_final_tex.Reset();
          depth_coordinate_v2_eye_final_uav.Reset();
          depth_coordinate_v2_eye_final_srv.Reset();
          BOOST_LOG(warning) << "Host SBS per-eye limiter resources unavailable; using the Base field.";
        }
      }

      const constants_t constants {
        .raw_coordinate_scale = parallax_v2_raw_coordinate_scale,
        .collapse_abs_epsilon = collapse_abs_epsilon,
        .coordinate_reserved0 = 0.0f,
        .coordinate_reserved1 = 0.0f,
        .requested_gain = parallax_v2_requested_gain,
        .max_horizontal_slope = max_horizontal_slope,
        .direct_container_limit = direct_container_limit,
        .convergence_curve_default = convergence_curve_default,
        .joint_plane_mode = joint_plane_mode,
        .joint_plane_reserved0 = 0u,
        .joint_plane_reserved1 = 0u,
        .joint_plane_reserved2 = 0u,
      };
      D3D11_BUFFER_DESC constants_desc {};
      constants_desc.Usage = D3D11_USAGE_DEFAULT;
      constants_desc.ByteWidth = sizeof(constants);
      constants_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA constants_data {&constants, 0, 0};
      resources_ok = resources_ok && SUCCEEDED(device->CreateBuffer(
        &constants_desc,
        &constants_data,
        &depth_coordinate_v2_cbuffer
      ));
      resources_ok = resources_ok && ensure_subtitle_resources();

      if (!resources_ok) {
        fail_parallax_v2_producer("GPU resource allocation failed"sv);
        return false;
      }

      const float clear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      context->ClearUnorderedAccessViewFloat(depth_coordinate_v2_candidate_uav.Get(), clear);
      context->ClearUnorderedAccessViewFloat(
        depth_coordinate_v2_vertical_majorant_uav.Get(),
        clear
      );
      context->ClearUnorderedAccessViewFloat(
        depth_coordinate_v2_vertical_conditioned_uav.Get(),
        clear
      );
      context->ClearUnorderedAccessViewFloat(depth_coordinate_v2_final_uav.Get(), clear);
      if (depth_coordinate_v2_eye_final_uav) {
        context->ClearUnorderedAccessViewFloat(depth_coordinate_v2_eye_final_uav.Get(), clear);
      }
      parallax_v2_producer_active = true;
      BOOST_LOG(info)
        << "Host SBS adaptive plane: continuous mean zero, robust Host depth scale, "
           "independent artistic gain, linear mapping, adaptive subtitle plane; "
        << "maximum per-eye source-U displacement=" << depth_coordinate_v2::direct_container_limit
        << ", source clock=monotonic microseconds.";
      BOOST_LOG(info)
        << "Host SBS V2 GPU producer active on one "
        << target_w << 'x' << target_h << " fused convex2x analysis/live grid"
        << "; the candidate feeds the 75% vertical upper and 25% vertical lower envelope "
           "followed by one horizontal majorant; the renderer will authenticate its first "
           "completed field before the one-time "
           "V2-or-flat latch.";
      return true;
    }

    bool upload_parallax_v2_observation_constants() {
      if (!depth_coordinate_v2_cbuffer) {
        return false;
      }
      using namespace depth_coordinate_v2;
      const constants_t constants {
        .raw_coordinate_scale = parallax_v2_raw_coordinate_scale,
        .collapse_abs_epsilon = collapse_abs_epsilon,
        .coordinate_reserved0 = 0.0f,
        .coordinate_reserved1 = 0.0f,
        .requested_gain = parallax_v2_requested_gain,
        .max_horizontal_slope = max_horizontal_slope,
        .direct_container_limit = direct_container_limit,
        .convergence_curve_default = convergence_curve_default,
        .joint_plane_mode = joint_plane_mode,
        .joint_plane_reserved0 = 0u,
        .joint_plane_reserved1 = 0u,
        .joint_plane_reserved2 = 0u,
        .joint_observation_timestamp_low = static_cast<std::uint32_t>(pending_root->observation_timestamp_us),
        .joint_observation_timestamp_high = static_cast<std::uint32_t>(pending_root->observation_timestamp_us >> 32u),
        .joint_observation_reserved0 = 0u,
        .joint_observation_reserved1 = 0u,
      };
      context->UpdateSubresource(depth_coordinate_v2_cbuffer.Get(), 0, nullptr, &constants, 0, 0);
      return true;
    }

    bool dispatch_parallax_v2_frame_publication(d3d_perf_slot *perf_slot) {
      if (!parallax_v2_producer_active) {
        return false;
      }
      if (!upload_parallax_v2_observation_constants()) {
        return false;
      }

      mark_d3d_parallax_frame_stats_start(perf_slot);
      const host_sbs_v2_gpu::moments_frame_command_t command {
        .constants = {
          .depth = cbuffer.Get(),
          .coordinate_v2 = depth_coordinate_v2_cbuffer.Get(),
        },
        .moments_shader = depth_coordinate_v2_moments_cs.Get(),
        .frame_resolve_shader = depth_coordinate_v2_frame_resolve_cs.Get(),
        .raw_depth = tensor_out_srv.Get(),
        .analysis_domain = analysis_domain.srv.Get(),
        .partials_output = depth_coordinate_v2_partials_uav.Get(),
        .partials = depth_coordinate_v2_partials_srv.Get(),
        .frame_stats_output = depth_coordinate_v2_frame_stats_uav.Get(),
        .minmax_raw_output = minmax_raw_uav.Get(),
        .moments_dispatch = parallax_v2_dispatch_command(reduce_groups, 1u, 1u, near_identical_gpu_infer_reduce_byte_offset),
        .frame_resolve_dispatch = parallax_v2_dispatch_command(1u, 1u, 1u, near_identical_gpu_infer_one_byte_offset),
        .histogram_shader = depth_coordinate_v2_histogram_cs.Get(),
        .quantile_shader = depth_coordinate_v2_publish_cs.Get(),
        .frame_stats = depth_coordinate_v2_frame_stats_srv.Get(),
        .histogram_output = depth_coordinate_v2_histogram_uav.Get(),
        .histogram = depth_coordinate_v2_histogram_srv.Get(),
        .publication_outputs = host_sbs_v2_gpu::frame_publication_outputs_t {
          .minmax_ema = minmax_ema_uav.Get(),
          .depth_health = depth_health_state_uav.Get(),
          .camera_state = depth_coordinate_v2_state_uav.Get(),
        },
      };
      const bool recorded = host_sbs_v2_gpu::record_moments_frame(
        context.Get(),
        command
      );
      mark_d3d_parallax_frame_stats_end(perf_slot);
      return recorded;
    }

    bool dispatch_parallax_v2_producer(d3d_perf_slot *perf_slot, const bool skip_input_history) {
      if (!parallax_v2_producer_active) {
        return false;
      }

      const auto content = live_field_content(pending_root->input_region);
      const bool use_compact_limiters =
        depth_coordinate_v2_vertical_limit_compact_cs && depth_coordinate_v2_limit_compact_cs &&
        !gpu_undecided_postprocess_pending() && !normalized_depth_diagnostics_enabled &&
        !subtitle_conditioning_diagnostics_requested() && pending_root->input_region.is_video_region() &&
        content && content->valid(live_field_shape()) &&
        content->right - content->left > depth_coordinate_v2::limiter_serial_max_lines &&
        (content->left != 0u || content->right < static_cast<std::uint32_t>(field_w));

      const bool use_compact_map = use_compact_limiters && depth_coordinate_v2_map_compact_cs;
      const auto map_width = use_compact_map && skip_input_history ?
                               content->width() :
                               static_cast<UINT>(target_w);

      const bool eye_pass = depth_coordinate_v2_eye_limit_cs && depth_coordinate_v2_eye_final_uav &&
                            depth_coordinate_v2_eye_limit_cbuffers[0u] && depth_coordinate_v2_eye_limit_cbuffers[1u];

      // Mapping publishes the history owner and may copy NCHW input. An explicit Dump 3D
      // separately materializes the full-size canonical-coordinate field.
      const host_sbs_v2_gpu::base_geometry_command_t command {
        .constants = {
          .depth = cbuffer.Get(),
          .coordinate_v2 = depth_coordinate_v2_cbuffer.Get(),
        },
        .near_identical_constants = near_identical_cbuffer.Get(),
        .map_shader = use_compact_map ? depth_coordinate_v2_map_compact_cs.Get() : depth_coordinate_v2_map_cs.Get(),
        .vertical_shader = use_compact_limiters ? depth_coordinate_v2_vertical_limit_compact_cs.Get() :
                                                  depth_coordinate_v2_vertical_limit_cs.Get(),
        .horizontal_shader = use_compact_limiters ? depth_coordinate_v2_limit_compact_cs.Get() :
                                                    depth_coordinate_v2_limit_cs.Get(),
        .raw_depth = tensor_out_srv.Get(),
        .state = depth_coordinate_v2_state_srv.Get(),
        .analysis_domain = analysis_domain.srv.Get(),
        .minmax_ema = minmax_ema_srv.Get(),
        .current_model_input = tensor_in_srv.Get(),
        .depth_health_state = depth_health_state_srv.Get(),
        .candidate = depth_coordinate_v2_candidate_srv.Get(),
        .vertical_conditioned = depth_coordinate_v2_vertical_conditioned_srv.Get(),
        .candidate_output = depth_coordinate_v2_candidate_uav.Get(),
        .previous_model_input_output = tensor_previous_input_uav.Get(),
        .history_owner_output = near_identical_history_owner.uav.Get(),
        .vertical_majorant_output = depth_coordinate_v2_vertical_majorant_uav.Get(),
        .vertical_conditioned_output = depth_coordinate_v2_vertical_conditioned_uav.Get(),
        .final_output = depth_coordinate_v2_final_uav.Get(),
        .map_dispatch = parallax_v2_dispatch_command(
          (map_width + 15u) / 16u,
          (target_h + 15) / 16,
          1u,
          near_identical_gpu_infer_grid16_byte_offset
        ),
        .vertical_dispatch = parallax_v2_dispatch_command(
          host_sbs_v2_gpu::limiter_groups(static_cast<UINT>(field_w)),
          1u,
          1u,
          near_identical_gpu_infer_columns_byte_offset
        ),
        .horizontal_dispatch = parallax_v2_dispatch_command(
          host_sbs_v2_gpu::limiter_groups(static_cast<UINT>(field_h)),
          1u,
          1u,
          near_identical_gpu_infer_rows_byte_offset
        ),
        .map_start = perf_slot && perf_slot->has_parallax ? perf_slot->parallax_map_start.Get() : nullptr,
        .limits_start = perf_slot ? perf_slot->parallax_limits_start.Get() : nullptr,
      };
      if (!host_sbs_v2_gpu::record_base_geometry(context.Get(), command)) {
        return false;
      }
      if (eye_pass) {
        // Same vertical share and the same (possibly GPU-gated) dispatch as the horizontal
        // majorant just recorded, so both fields always describe the same frame.
        const host_sbs_v2_gpu::eye_limit_command_t eye_command {
          .constants = command.constants,
          .near_identical_constants = command.near_identical_constants,
          .eye_constants = depth_coordinate_v2_eye_limit_cbuffers[use_compact_limiters ? 1u : 0u].Get(),
          .shader = depth_coordinate_v2_eye_limit_cs.Get(),
          .vertical_conditioned = depth_coordinate_v2_vertical_conditioned_srv.Get(),
          .eye_output = depth_coordinate_v2_eye_final_uav.Get(),
          .dispatch = command.horizontal_dispatch,
        };
        if (!host_sbs_v2_gpu::record_eye_limit(context.Get(), eye_command)) {
          return false;
        }
      }
      if (command.map_start) {
        perf_slot->has_parallax_map_start = true;
      }
      if (command.limits_start) {
        perf_slot->has_parallax_limits_start = true;
      }

      mark_d3d_parallax_subtitle_start(perf_slot);

      ID3D11Buffer *null_constants[2] = {nullptr, nullptr};
      context->CSSetConstantBuffers(1, 2, null_constants);
      return true;
    }

    bool update_pending_ocr_constants(const bool input_domain_reset) {
      const auto tensor_content = live_field_content(pending_root->input_region);
      if (!ocr_resolve_cbuffer || pending_root->frame_id == 0u || target_w <= 0 || target_h <= 0 ||
          !tensor_content || !tensor_content->valid(live_field_shape()) ||
          !depth_coordinate_v2::subtitle_ocr_field_is_calibrated(
            static_cast<std::uint32_t>(field_w), static_cast<std::uint32_t>(field_h))) {
        return false;
      }
      const std::uint32_t source_width = pending_root->input_region.width();
      const std::uint32_t source_height = pending_root->input_region.height();
      const auto roi = fit_subtitle_analysis_geometry(
        source_width,
        source_height,
        live_field_shape(),
        *tensor_content
      );
      // A legal unprojectable OCR crop still owes this frame an exact abstention header.
      // It cannot authorize an optional detector child or manufacture a search region.
      if (!roi.valid() &&
          pending_root->subtitle_work != cuda_conditional_graph::work_flag_e::subtitle_observation) {
        return false;
      }
      const std::uint32_t crop_height = subtitle_ocr_source_crop_height(
        source_width,
        source_height
      );
      if (crop_height == 0u || crop_height > source_height) {
        return false;
      }
      auto line_reference = subtitle_ocr_line_reference(
        source_width, source_height,
        pending_root->input_region.source_width, pending_root->input_region.source_height,
        pending_root->input_region.left, pending_root->input_region.top,
        pending_root->input_region.is_video_region()
      );
      if (!line_reference) {
        if (pending_root->subtitle_work != cuda_conditional_graph::work_flag_e::subtitle_observation) {
          return false;
        }
        // The finalizer's abstention consumes only the exact source/field identity prefix.
        // No classifier or optional OCR child may consume an unsupported reference.
        line_reference.emplace();
      }
      const ocr_resolve_constants_t constants {
        static_cast<std::uint32_t>(pending_root->frame_id),
        static_cast<std::uint32_t>(pending_root->frame_id >> 32u),
        static_cast<std::uint32_t>(pending_root->input_region.analysis_generation),
        static_cast<std::uint32_t>(pending_root->input_region.analysis_generation >> 32u),
        source_width,
        source_height,
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h),
        source_height - crop_height,
        crop_height,
        roi.roi_top,
        roi.roi_bottom,
        tensor_content->left,
        tensor_content->top,
        tensor_content->right,
        tensor_content->bottom,
        std::bit_cast<std::uint32_t>(line_reference->center_twice_source),
        line_reference->width,
        std::bit_cast<std::uint32_t>(line_reference->search_top_source),
        static_cast<std::uint32_t>(pending_root->subtitle_line_policy),
        static_cast<std::uint32_t>(pending_root->observation_timestamp_us),
        static_cast<std::uint32_t>(pending_root->observation_timestamp_us >> 32u),
        input_domain_reset ? 1u : 0u,
        0u,
      };
      context->UpdateSubresource(
        ocr_resolve_cbuffer.Get(), 0u, nullptr, constants.data(), 0u, 0u
      );
      return true;
    }

    bool dispatch_ocr_postprocess() {
      const bool optional_child_armed = std::exchange(pending_root->ocr_armed, false);
      if (!near_identical_transaction.dispatch || !ocr_box_record_uav ||
          !ocr_resolve_cbuffer) {
        return false;
      }
      const bool optional_postprocess_ready =
        ocr_box_cells_cs && ocr_box_resolve_cs && ocr_output_srv &&
        ocr_cell_stats_srv && ocr_cell_stats_uav && ocr_local_candidate_scratch_uav;
      if (optional_child_armed && !optional_postprocess_ready) {
        // Once the root has joined an OCR child, losing its exact postprocess resources must fail
        // this unit closed. Publishing a miss would hide a completed-but-unconsumed OCR result.
        return false;
      }
      ID3D11ShaderResourceView *null_srv = nullptr;
      ID3D11UnorderedAccessView *null_uav = nullptr;
      if (optional_postprocess_ready) {
        context->CSSetShader(ocr_box_cells_cs.Get(), nullptr, 0u);
        context->CSSetShaderResources(0u, 1u, ocr_output_srv.GetAddressOf());
        context->CSSetUnorderedAccessViews(
          0u, 1u, ocr_cell_stats_uav.GetAddressOf(), nullptr
        );
        context->DispatchIndirect(
          near_identical_transaction.dispatch.Get(),
          near_identical_gpu_optional_cells_byte_offset
        );
        context->CSSetShaderResources(0u, 1u, &null_srv);
        context->CSSetUnorderedAccessViews(0u, 1u, &null_uav, nullptr);

        context->CSSetShader(ocr_box_resolve_cs.Get(), nullptr, 0u);
        context->CSSetConstantBuffers(0u, 1u, ocr_resolve_cbuffer.GetAddressOf());
        context->CSSetShaderResources(1u, 1u, ocr_cell_stats_srv.GetAddressOf());
        ID3D11UnorderedAccessView *resolve_uavs[2] = {
          ocr_box_record_uav.Get(), ocr_local_candidate_scratch_uav.Get(),
        };
        context->CSSetUnorderedAccessViews(1u, 2u, resolve_uavs, nullptr);
        context->DispatchIndirect(
          near_identical_transaction.dispatch.Get(),
          near_identical_gpu_optional_one_byte_offset
        );
        context->CSSetShaderResources(1u, 1u, &null_srv);
        ID3D11UnorderedAccessView *null_resolve_uavs[2] = {};
        context->CSSetUnorderedAccessViews(1u, 2u, null_resolve_uavs, nullptr);
      }

      // The detector finalizer already published the exact-current abstention for every
      // authenticated observation without OOCR. Only a matching optional receipt reaches the
      // resolver above, so there is no second fallback dispatch here.
      ID3D11Buffer *null_buffer = nullptr;
      context->CSSetConstantBuffers(0u, 1u, &null_buffer);
      return true;
    }

    bool dispatch_subtitle_contours(
      const bool ocr_record_submitted,
      const bool input_domain_reset,
      const bool conditioning_diagnostics,
      const bool legacy_matte,
      d3d_perf_slot *perf_slot = nullptr
    ) {
      if (!subtitle_contour_palette_cs || !subtitle_contour_observation.ready() ||
          !subtitle_contour_temporal_cs || !subtitle_contour_history_cs ||
          !subtitle_contour_history_identity_cs ||
          !subtitle_contour_snapshot_cs || !subtitle_contour_cbuffer || !subtitle_contour_identity.ready() ||
          !subtitle_contour_source.ready() ||
          (legacy_matte && (!subtitle_glyph_matte_cs || !subtitle_glyph_matte.ready())) ||
          !subtitle_contour_source_snapshot.ready() || !subtitle_contour_probability_snapshot.ready()) {
        return false;
      }
      const bool contour_history = subtitle_contour_history_required(legacy_matte);
      if (!contour_history) {
        release_subtitle_contour_history();
      } else if (!ensure_subtitle_contour_history()) {
        return false;
      }
      const auto content = live_field_content(pending_root->input_region)
                             .value_or(depth_tensor_content_rect_t {});
      const auto roi = fit_subtitle_analysis_geometry(
        pending_root->input_region.width(), pending_root->input_region.height(), live_field_shape(), content
      );
      const auto crop_height = subtitle_ocr_source_crop_height(
        pending_root->input_region.width(), pending_root->input_region.height()
      );
      std::array<std::uint32_t, 24u> constants {
        pending_root->input_region.width(), pending_root->input_region.height(),
        pending_root->input_region.height() - std::min(crop_height, pending_root->input_region.height()),
        crop_height,
        static_cast<std::uint32_t>(field_w), static_cast<std::uint32_t>(field_h),
        roi.roi_top, roi.roi_bottom,
        content.left, content.top, content.right, content.bottom,
        static_cast<std::uint32_t>(pending_root->frame_id),
        static_cast<std::uint32_t>(pending_root->frame_id >> 32u),
        static_cast<std::uint32_t>(pending_root->input_region.analysis_generation),
        static_cast<std::uint32_t>(pending_root->input_region.analysis_generation >> 32u),
        static_cast<std::uint32_t>(pending_root->observation_timestamp_us),
        static_cast<std::uint32_t>(pending_root->observation_timestamp_us >> 32u),
        input_domain_reset ? 1u : 0u, 0u,
        ocr_record_submitted ? 1u : 0u,
        depth_coordinate_v2::subtitle_contour_recipe, 0u, 0u
      };
      context->UpdateSubresource(subtitle_contour_cbuffer.Get(), 0u, nullptr,
                                 constants.data(), 0u, 0u);
      ID3D11Buffer *contour_constant_buffers[2] {
        subtitle_contour_cbuffer.Get(), depth_coordinate_v2_cbuffer.Get()
      };
      context->CSSetConstantBuffers(0u, 2u, contour_constant_buffers);
      ID3D11ShaderResourceView *srvs[13] {
        subtitle_contour_source.srv.Get(), ocr_output_srv.Get(),
        subtitle_locator_state_srv.Get(), subtitle_condition_params_srv.Get(),
        nullptr, nullptr, contour_history ? subtitle_contour_previous_features.srv.Get() : nullptr,
        subtitle_contour_identity.srv.Get(), depth_health_state_srv.Get(), ocr_box_record_srv.Get(),
        depth_coordinate_v2_state_srv.Get(), nullptr, nullptr
      };
      context->CSSetShaderResources(0u, 13u, srvs);
      // Prepare current evidence for the same fresh observation that runs the contour grid.
      // An OCR abstention must publish its own verdict even when palette learning has no work.
      ID3D11UnorderedAccessView *observation_output = subtitle_contour_observation.uav.Get();
      ID3D11UnorderedAccessView *null_uav = nullptr;
      context->CSSetUnorderedAccessViews(7u, 1u, &observation_output, nullptr);
      context->CSSetShader(subtitle_contour_palette_cs.Get(), nullptr, 0u);
      context->DispatchIndirect(near_identical_transaction.dispatch.Get(),
                               near_identical_gpu_observation_one_byte_offset);
      context->CSSetUnorderedAccessViews(7u, 1u, &null_uav, nullptr);
      ID3D11ShaderResourceView *observation_input = subtitle_contour_observation.srv.Get();
      context->CSSetShaderResources(12u, 1u, &observation_input);
      ID3D11UnorderedAccessView *output = nullptr;
      if (contour_history) {
        // Reuse submits zero groups for every pass. Do not clear or copy the published support
        // unconditionally: its backing holds the previous complete tuple until fresh completion.
        output = subtitle_contour_support.uav.Get();
        context->CSSetUnorderedAccessViews(0u, 1u, &output, nullptr);
        ID3D11UnorderedAccessView *next_features = subtitle_contour_next_features.uav.Get();
        context->CSSetUnorderedAccessViews(2u, 1u, &next_features, nullptr);
        context->CSSetShader(subtitle_contour_temporal_cs.Get(), nullptr, 0u);
        context->DispatchIndirect(near_identical_transaction.dispatch.Get(),
                                 near_identical_gpu_subtitle_condition_grid16_byte_offset);
        context->CSSetUnorderedAccessViews(0u, 1u, &null_uav, nullptr);
        context->CSSetUnorderedAccessViews(2u, 1u, &null_uav, nullptr);

        srvs[5] = subtitle_contour_support.srv.Get();
        context->CSSetShaderResources(5u, 1u, &srvs[5u]);
      }
      if (legacy_matte) {
        // Reuse submits zero groups and holds the previous dense output. Fresh observations
        // write the full native band, including literal zero for empty observations.
        output = subtitle_glyph_matte.uav.Get();
        context->CSSetUnorderedAccessViews(6u, 1u, &output, nullptr);
        context->CSSetShader(subtitle_glyph_matte_cs.Get(), nullptr, 0u);
        context->DispatchIndirect(near_identical_transaction.dispatch.Get(),
                                 near_identical_gpu_subtitle_condition_grid16_byte_offset);
        context->CSSetUnorderedAccessViews(6u, 1u, &null_uav, nullptr);
      }

      if (conditioning_diagnostics) {
        output = subtitle_conditioning_diagnostics->field_mask.uav.Get();
        context->CSSetUnorderedAccessViews(1u, 1u, &output, nullptr);
        context->CSSetShader(subtitle_contour_project_cs.Get(), nullptr, 0u);
        context->DispatchIndirect(near_identical_transaction.dispatch.Get(),
                                 near_identical_gpu_subtitle_condition_grid16_byte_offset);
        context->CSSetUnorderedAccessViews(1u, 1u, &null_uav, nullptr);
      }

      // CPU-completed force work has an exact-current observation even when OCR abstains. Its
      // staged features can become history by exchanging owners after every old-feature read.
      // Receipt-gated work can issue zero groups for reuse or rejected authority, so it must keep
      // the existing GPU-conditional copy and the same history owner regardless of host guesses.
      // Without feature history only the SCT identity is published.
      const bool copy_feature_history = contour_history && gpu_undecided_postprocess_pending();
      ID3D11ShaderResourceView *null_history[2] {};
      context->CSSetShaderResources(6u, 2u, null_history);
      srvs[11] = contour_history ? subtitle_contour_next_features.srv.Get() : nullptr;
      context->CSSetShaderResources(11u, 1u, &srvs[11u]);
      ID3D11UnorderedAccessView *history_outputs[2] {
        contour_history ? subtitle_contour_previous_features.uav.Get() : nullptr,
        subtitle_contour_identity.uav.Get()
      };
      context->CSSetUnorderedAccessViews(2u, 2u, history_outputs, nullptr);
      context->CSSetShader(copy_feature_history ? subtitle_contour_history_cs.Get() :
                                                  subtitle_contour_history_identity_cs.Get(), nullptr, 0u);
      context->DispatchIndirect(near_identical_transaction.dispatch.Get(),
                               near_identical_gpu_subtitle_condition_grid16_byte_offset);
      ID3D11UnorderedAccessView *null_outputs[2] {};
      context->CSSetUnorderedAccessViews(2u, 2u, null_outputs, nullptr);
      ID3D11ShaderResourceView *null_next = nullptr;
      context->CSSetShaderResources(11u, 1u, &null_next);
      if (contour_history && !copy_feature_history) {
        std::swap(subtitle_contour_previous_features, subtitle_contour_next_features);
      }
      // Preserve the completed OCR source/probability before the next asynchronous admission
      // overwrites or CUDA-maps their working resources. Reuse's zero indirect groups hold the
      // snapshots byte-for-byte; no CPU verdict/readback or unconditional copy is introduced.
      ID3D11UnorderedAccessView *snapshot_outputs[2] {
        subtitle_contour_source_snapshot.uav.Get(), subtitle_contour_probability_snapshot.uav.Get()
      };
      context->CSSetUnorderedAccessViews(4u, 2u, snapshot_outputs, nullptr);
      context->CSSetShader(subtitle_contour_snapshot_cs.Get(), nullptr, 0u);
      context->DispatchIndirect(near_identical_transaction.dispatch.Get(),
                               near_identical_gpu_subtitle_condition_grid16_byte_offset);
      context->CSSetUnorderedAccessViews(4u, 2u, null_outputs, nullptr);
      ID3D11ShaderResourceView *null_srvs[13] {};
      context->CSSetShaderResources(0u, 13u, null_srvs);

      // Direct Glyph publishes after the contour passes, from the completion poll.
      subtitle_selected_matte_commands = false;
      subtitle_direct_glyph_requested = subtitle_direct_glyph.enabled();

      // The native distance transform is consumed only by the diagnostic Gaussian proposal.
      if (!conditioning_diagnostics) {
        return true;
      }
      const auto &diagnostics = *subtitle_conditioning_diagnostics;
      const std::array<std::uint32_t, 12u> distance_constants {
        pending_root->input_region.width(), pending_root->input_region.height(),
        pending_root->input_region.height() - std::min(crop_height, pending_root->input_region.height()),
        crop_height,
        static_cast<std::uint32_t>(field_w), static_cast<std::uint32_t>(field_h), 0u, 0u,
        content.left, content.top, content.right, content.bottom
      };
      context->UpdateSubresource(diagnostics.distance_constants.Get(), 0u, nullptr,
                                 distance_constants.data(), 0u, 0u);
      context->CSSetConstantBuffers(0u, 1u, diagnostics.distance_constants.GetAddressOf());
      // q is measured from native source-pixel centers, including the four offset centers.
      // Projected field pins remain a separate diagnostic and do not seed this transform.
      ID3D11ShaderResourceView *seed = subtitle_contour_support.srv.Get();
      context->CSSetShaderResources(0u, 1u, &seed);
      output = diagnostics.horizontal_distance.uav.Get();
      context->CSSetUnorderedAccessViews(0u, 1u, &output, nullptr);
      context->CSSetShader(subtitle_contour_horizontal_cs.Get(), nullptr, 0u);
      context->DispatchIndirect(near_identical_transaction.dispatch.Get(),
                               near_identical_gpu_subtitle_condition_grid16_byte_offset);
      context->CSSetUnorderedAccessViews(0u, 1u, &null_uav, nullptr);
      context->CSSetShaderResources(0u, 1u, null_srvs);
      ID3D11ShaderResourceView *horizontal = diagnostics.horizontal_distance.srv.Get();
      context->CSSetShaderResources(1u, 1u, &horizontal);
      output = diagnostics.distance.uav.Get();
      context->CSSetUnorderedAccessViews(0u, 1u, &output, nullptr);
      context->CSSetShader(subtitle_contour_vertical_cs.Get(), nullptr, 0u);
      context->DispatchIndirect(near_identical_transaction.dispatch.Get(),
                               near_identical_gpu_subtitle_condition_grid16_byte_offset);
      context->CSSetUnorderedAccessViews(0u, 1u, &null_uav, nullptr);
      context->CSSetShaderResources(1u, 1u, null_srvs);
      return true;
    }

    void clear_subtitle_conditioning_diagnostics(const subtitle_conditioning_diagnostics_t &diagnostics) {
      const UINT zero[4] {};
      for (auto *uav : {
             diagnostics.field_mask.uav.Get(), diagnostics.horizontal_distance.uav.Get(), diagnostics.distance.uav.Get(),
             diagnostics.weight.uav.Get(), diagnostics.blend.uav.Get(), diagnostics.vertical_conditioned.uav.Get(),
             diagnostics.control.uav.Get(),
           }) {
        context->ClearUnorderedAccessViewUint(uav, zero);
      }
      const float zero_float[4] {};
      context->ClearUnorderedAccessViewFloat(diagnostics.final.uav.Get(), zero_float);
    }

    bool ensure_subtitle_conditioning_diagnostics() {
      if (!subtitle_contour_project_cs || !subtitle_contour_horizontal_cs || !subtitle_contour_vertical_cs ||
          !subtitle_condition_cs || !subtitle_gaussian_vertical_limit_cs || !subtitle_gaussian_horizontal_limit_cs ||
          !subtitle_contour_source.ready()) {
        return false;
      }
      // The field shape belongs to the estimator resource epoch. The native crop may resize
      // within that epoch, so only the horizontal-distance scratch depends on this texture.
      D3D11_TEXTURE2D_DESC native {};
      subtitle_contour_source.texture->GetDesc(&native);
      const auto horizontal_height = native.Height + 4u;
      const auto width = static_cast<UINT>(field_w);
      const auto height = static_cast<UINT>(field_h);
      if (auto &diagnostics = subtitle_conditioning_diagnostics) {
        if (diagnostics->horizontal_distance_height != horizontal_height) {
          d3d_texture_views_t horizontal_distance;
          if (!create_shader_texture(width, horizontal_height, DXGI_FORMAT_R32_FLOAT, horizontal_distance)) {
            return false;
          }
          const UINT zero[4] {};
          context->ClearUnorderedAccessViewUint(horizontal_distance.uav.Get(), zero);
          diagnostics->horizontal_distance = std::move(horizontal_distance);
          diagnostics->horizontal_distance_height = horizontal_height;
        }
        return true;
      }

      subtitle_conditioning_diagnostics_t next;
      if (!create_shader_texture(width, height, DXGI_FORMAT_R32_UINT, next.field_mask) ||
          !create_shader_texture(width, horizontal_height, DXGI_FORMAT_R32_FLOAT, next.horizontal_distance) ||
          !create_shader_texture(width, height, DXGI_FORMAT_R32_FLOAT, next.distance) ||
          !create_shader_texture(width, height, DXGI_FORMAT_R32_FLOAT, next.weight) ||
          !create_shader_texture(width, height, DXGI_FORMAT_R32_FLOAT, next.blend) ||
          !create_shader_texture(width, height, DXGI_FORMAT_R32_FLOAT, next.vertical_conditioned) ||
          !create_shader_texture(width, height, DXGI_FORMAT_R32_FLOAT, next.final) ||
          !create_shader_uint_buffer(1u, next.control.buffer, next.control.srv, next.control.uav) ||
          !ensure_subtitle_constant_buffer(48u, next.distance_constants)) {
        return false;
      }
      next.horizontal_distance_height = horizontal_height;
      // Commit only the complete set. A failed optional allocation cannot expose partial views
      // or make a later request mistake them for initialized scratch.
      clear_subtitle_conditioning_diagnostics(next);
      subtitle_conditioning_diagnostics.emplace(std::move(next));
      return true;
    }

    bool dispatch_subtitle_conditioner(
      const bool ocr_record_submitted,
      const bool input_domain_reset,
      d3d_perf_slot *perf_slot = nullptr
    ) {
      // An authenticated abstention has no detector probability buffer. Its inactive OCR8
      // record still advances current identity and clears support while rendering from Base.
      if (!subtitle_locator_resolve_cs ||
          !subtitle_locator_cbuffer || !subtitle_locator_state_srv ||
          !subtitle_locator_state_uav || !depth_coordinate_v2_state_srv ||
          !depth_coordinate_v2_final_srv ||
          !depth_coordinate_v2_final_uav ||
          !subtitle_condition_params_srv || !subtitle_condition_params_uav ||
          !ocr_box_record_srv ||
          (pending_root->subtitle_work == cuda_conditional_graph::work_flag_e::optional_ocr &&
           !ocr_output_srv) || !subtitle_contour_identity.ready() ||
          !subtitle_contour_probability_snapshot.ready() ||
          !depth_health_state_srv || !near_identical_transaction.dispatch) {
        return false;
      }

      const bool legacy_matte_required = subtitle_legacy_matte_required();
      bool conditioning_diagnostics = subtitle_conditioning_diagnostics_requested();
      if (conditioning_diagnostics &&
          !(ensure_subtitle_conditioning_diagnostics() &&
            (legacy_matte_required || ensure_subtitle_dump_matte()))) {
        BOOST_LOG(warning) << "Host SBS conditioned parallax diagnostic resources are unavailable.";
        if (subtitle_conditioning_diagnostics_always) {
          return false;
        }
        // A live dump can wait for a qualified observation while rendering continues.
        conditioning_diagnostics = false;
      }
      const bool legacy_matte = legacy_matte_required || conditioning_diagnostics;
      const std::uint32_t source_width = pending_root->input_region.width();
      const std::uint32_t source_height = pending_root->input_region.height();
      const std::uint32_t field_width = field_w > 0 ?
                                          static_cast<std::uint32_t>(field_w) :
                                          0u;
      const std::uint32_t field_height = field_h > 0 ?
                                           static_cast<std::uint32_t>(field_h) :
                                           0u;
      const auto tensor_content = live_field_content(pending_root->input_region);
      const auto roi = fit_subtitle_analysis_geometry(
        source_width,
        source_height,
        live_field_shape(),
        tensor_content.value_or(depth_tensor_content_rect_t {})
      );
      // Direct OCR8 coverage follows the current authenticated analysis field. An unsupported tensor or an
      // unprojectable bottom crop has no subtitle authority. If requested, diagnostics
      // still produce a complete field with exact Base bits.
      const auto field_content = tensor_content.value_or(depth_tensor_content_rect_t {});
      const std::array<std::uint32_t, 20> constants {
        field_width,
        field_height,
        roi.roi_top,
        roi.roi_bottom,
        source_width,
        source_height,
        ocr_record_submitted ? 1u : 0u,
        input_domain_reset ? 1u : 0u,
        static_cast<std::uint32_t>(pending_root->frame_id),
        static_cast<std::uint32_t>(pending_root->frame_id >> 32u),
        static_cast<std::uint32_t>(pending_root->input_region.analysis_generation),
        static_cast<std::uint32_t>(pending_root->input_region.analysis_generation >> 32u),
        field_content.left,
        field_content.top,
        field_content.right,
        field_content.bottom,
        static_cast<std::uint32_t>(pending_root->observation_timestamp_us),
        static_cast<std::uint32_t>(pending_root->observation_timestamp_us >> 32u),
        0u,
        0u,
      };
      context->UpdateSubresource(
        subtitle_locator_cbuffer.Get(), 0, nullptr, constants.data(), 0,
        0
      );

      ID3D11Buffer *constant_buffers[3] = {
        cbuffer.Get(),
        depth_coordinate_v2_cbuffer.Get(),
        subtitle_locator_cbuffer.Get(),
      };
      ID3D11ShaderResourceView *null_srvs[8] = {};
      context->CSSetConstantBuffers(0, 3, constant_buffers);
      // Keep OCR ownership and UI timing in the same authenticated observation dispatch.
      // An unusable geometry publication suspends UI timing and publishes Base exactly.
      // Joint depth reuse and explicit subtitle suppression hold the complete subtitle tuple.
      context->CSSetShader(subtitle_locator_resolve_cs.Get(), nullptr, 0);
      // The resolve reads t0..t2, t6 and t7; t3..t5 are the condition pass's inputs.
      ID3D11ShaderResourceView *resolve_srvs[8] = {
        depth_coordinate_v2_state_srv.Get(),
        depth_health_state_srv.Get(),
        depth_coordinate_v2_final_srv.Get(),
        nullptr,
        nullptr,
        nullptr,
        subtitle_contour_identity.srv.Get(),
        ocr_box_record_srv.Get(),
      };
      // The prior completed identity remains immutable through this resolve; the contour
      // history/snapshot passes overwrite it only after these SRVs are unbound.
      context->CSSetShaderResources(0, 8, resolve_srvs);
      ID3D11UnorderedAccessView *resolve_uavs[3] = {
        subtitle_locator_state_uav.Get(),
        nullptr,
        subtitle_condition_params_uav.Get(),
      };
      context->CSSetUnorderedAccessViews(2, 3, resolve_uavs, nullptr);
      context->DispatchIndirect(
        near_identical_transaction.dispatch.Get(),
        near_identical_gpu_observation_one_byte_offset
      );
      context->CSSetShaderResources(0, 8, null_srvs);
      ID3D11UnorderedAccessView *null_resolve_uavs[3] = {};
      context->CSSetUnorderedAccessViews(2, 3, null_resolve_uavs, nullptr);

      if (!dispatch_subtitle_contours(
            ocr_record_submitted, input_domain_reset, conditioning_diagnostics, legacy_matte, perf_slot
          )) {
        return false;
      }
      const auto retain_completed_contour = [&]() {
        // Retain completed backing views before estimate() can allocate the next source extent.
        // Working OCR inputs may change immediately; the joined completed snapshots cannot.
        completed_contour = {
          subtitle_contour_support.srv, subtitle_contour_identity.srv,
          subtitle_contour_previous_features.srv, subtitle_contour_source_snapshot.srv,
          subtitle_contour_probability_snapshot.srv,
          legacy_matte ? subtitle_glyph_matte.srv : nullptr, subtitle_condition_params_srv
        };
      };
      if (!conditioning_diagnostics) {
        // No renderer consumes the conditioned field. The effective Final remains Base.
        ID3D11Buffer *null_constants[3] = {};
        context->CSSetConstantBuffers(0, 3, null_constants);
        completed_subtitle_conditioning_diagnostics = false;
        retain_completed_contour();
        return true;
      }
      const auto &diagnostics = *subtitle_conditioning_diagnostics;
      context->CSSetConstantBuffers(0, 3, constant_buffers);

      // Base remains immutable. Every diagnostic subtitle observation writes the complete
      // Gaussian proposal, weight and smoothing-control tuple. Reuse holds this tuple with the
      // complete depth, subtitle and final field.
      ID3D11ShaderResourceView *condition_srvs[8] = {
        depth_coordinate_v2_state_srv.Get(),
        depth_health_state_srv.Get(),
        depth_coordinate_v2_final_srv.Get(),
        subtitle_locator_state_srv.Get(),
        subtitle_condition_params_srv.Get(),
        diagnostics.distance.srv.Get(),
        subtitle_contour_identity.srv.Get(),
        ocr_box_record_srv.Get(),
      };
      context->CSSetShaderResources(0, 8, condition_srvs);
      ID3D11UnorderedAccessView *condition_outputs[3] = {
        diagnostics.blend.uav.Get(), diagnostics.control.uav.Get(), diagnostics.weight.uav.Get(),
      };
      context->CSSetUnorderedAccessViews(3, 3, condition_outputs, nullptr);
      context->CSSetShader(subtitle_condition_cs.Get(), nullptr, 0);
      context->DispatchIndirect(
        near_identical_transaction.dispatch.Get(),
        near_identical_gpu_subtitle_condition_grid16_byte_offset
      );
      context->CSSetShaderResources(0, 8, null_srvs);
      ID3D11UnorderedAccessView *null_condition_outputs[3] = {};
      context->CSSetUnorderedAccessViews(3, 3, null_condition_outputs, nullptr);

      // Subtitle transitions use the shared horizontal Q30 core. The compatibility vertical
      // entry copies the proposal; unchanged rows and disabled control preserve immutable Base
      // bits. Private resources keep the original scene-stage diagnostics intact.
      ID3D11ShaderResourceView *vertical_srvs[4] = {
        diagnostics.blend.srv.Get(), nullptr, diagnostics.control.srv.Get(),
        depth_coordinate_v2_final_srv.Get(),
      };
      ID3D11UnorderedAccessView *vertical_outputs[2] = {
        nullptr,
        diagnostics.vertical_conditioned.uav.Get(),
      };
      context->CSSetShaderResources(0u, 4u, vertical_srvs);
      context->CSSetUnorderedAccessViews(0u, 2u, vertical_outputs, nullptr);
      context->CSSetShader(subtitle_gaussian_vertical_limit_cs.Get(), nullptr, 0u);
      dispatch_infer_postprocess(
        host_sbs_v2_gpu::limiter_groups(static_cast<UINT>(field_w)), 1u, 1u,
        near_identical_gpu_infer_columns_byte_offset
      );
      context->CSSetUnorderedAccessViews(0u, 2u, null_condition_outputs, nullptr);
      context->CSSetShaderResources(0u, 4u, null_srvs);

      ID3D11ShaderResourceView *horizontal_srvs[4] = {
        diagnostics.vertical_conditioned.srv.Get(), nullptr,
        diagnostics.control.srv.Get(), depth_coordinate_v2_final_srv.Get(),
      };
      ID3D11UnorderedAccessView *final_output = diagnostics.final.uav.Get();
      context->CSSetShaderResources(0u, 4u, horizontal_srvs);
      context->CSSetUnorderedAccessViews(0u, 1u, &final_output, nullptr);
      context->CSSetShader(subtitle_gaussian_horizontal_limit_cs.Get(), nullptr, 0u);
      dispatch_infer_postprocess(
        host_sbs_v2_gpu::limiter_groups(static_cast<UINT>(field_h)), 1u, 1u,
        near_identical_gpu_infer_rows_byte_offset
      );
      context->CSSetUnorderedAccessViews(0u, 1u, null_condition_outputs, nullptr);
      context->CSSetShaderResources(0u, 4u, null_srvs);
      ID3D11Buffer *null_constants[3] = {};
      context->CSSetConstantBuffers(0, 3, null_constants);
      // A receipt-gated root may issue zero groups and hold the prior tuple, so it can only
      // continue an already current diagnostic tuple; a CPU-completed root establishes one.
      completed_subtitle_conditioning_diagnostics =
        !gpu_undecided_postprocess_pending() || completed_subtitle_conditioning_diagnostics;
      retain_completed_contour();
      return true;
    }

    [[nodiscard]] std::uint32_t pending_gpu_completion_flags(
      const pending_gpu_trace_append_t &append
    ) const noexcept {
      std::uint32_t flags = 0u;
      if (append.input_domain_reset) {
        flags |= host_sbs_gpu_trace::record_flag_e::input_domain_reset;
      }
      if (pending_root->dump_forced) {
        flags |= host_sbs_gpu_trace::record_flag_e::dump_forced;
      }
      if (append.ocr_record_submitted) {
        flags |= host_sbs_gpu_trace::record_flag_e::ocr_record_submitted;
      }
      if (append.subtitle_work_suppressed) {
        flags |= host_sbs_gpu_trace::record_flag_e::subtitle_suppressed;
      }
      if (append.condition_executed) {
        flags |= host_sbs_gpu_trace::record_flag_e::condition_executed;
      }
      if (append.subtitle_branch_gated) {
        flags |= host_sbs_gpu_trace::record_flag_e::subtitle_branch_gated;
      }
      return flags;
    }

    [[nodiscard]] publication_receipt::expected_t pending_publication_expected() const {
      const auto content = live_field_content(pending_root->input_region).value_or(depth_tensor_content_rect_t {});
      const auto subtitle_roi = fit_subtitle_analysis_geometry(pending_root->input_region.width(),
        pending_root->input_region.height(), live_field_shape(), content);
      return {
        .estimator_generation = publication_generation,
        .publication_sequence = publication_sequence,
        .analysis_generation = pending_root->input_region.analysis_generation,
        .frame_id = pending_root->frame_id,
        .transaction_token = pending_root->transaction_token,
        .domain_tag = near_identical_input_domain_tag(
          pending_root->input_region, pending_root->color_space,
          static_cast<std::uint32_t>(target_w), static_cast<std::uint32_t>(target_h)
        ),
        .observation_timestamp_us = pending_root->observation_timestamp_us,
        .baseline_frame_id = pending_root->baseline_frame_id,
        .width = static_cast<std::uint32_t>(target_w),
        .height = static_cast<std::uint32_t>(target_h),
        .subtitle_domain = {pending_root->input_region.width(), pending_root->input_region.height(),
          static_cast<std::uint32_t>(field_w), static_cast<std::uint32_t>(field_h),
          subtitle_roi.roi_top, subtitle_roi.roi_bottom,
          content.left, content.top, content.right, content.bottom},
        .raw_coordinate_scale = parallax_v2_raw_coordinate_scale,
        .joint_plane_mode = joint_plane_mode,
        .expected_work = cuda_conditional_graph::work_flags_value(pending_root->subtitle_work),
        .submission_class = gpu_undecided_postprocess_pending() ?
                              publication_receipt::submission_class_e::gpu_undecided :
                              publication_receipt::submission_class_e::force_infer,
        .flags = pending_gpu_completion_flags(pending_gpu_trace_append),
        .host_subtitle_outcome = pending_gpu_trace_append.host_subtitle_outcome,
        .depth_only_history = pending_root->depth_only_history,
      };
    }

    void snapshot_pending_publication_receipt() {
      current_publication = pending_publication_expected();
      publication_readback.submit(
        {{
          near_identical_transaction.buffer.Get(),
          near_identical_history_owner.buffer.Get(),
          subtitle_locator_state_buf.Get(),
          subtitle_condition_params_buf.Get(),
          depth_coordinate_v2_state_buf.Get(),
          minmax_ema_buf.Get(),
          depth_health_state_buf.Get(),
          gpu_outcome_buffer.Get(),
        }},
        current_publication
      );
    }

    [[nodiscard]] bool publication_is_current(
      const publication_receipt::expected_t &publication
    ) const noexcept {
      return is_operational() && publication_receipt::valid_expected(current_publication) &&
             current_publication == publication;
    }

    publication_receipt_poll_result poll_publication_receipts() {
      auto result = publication_readback.poll(publication_generation);
      consume_completion_diagnostics(result);
      if (result.transport_failed) {
        cached_current_publication.reset();
      }
      // A receipt for B may arrive after C was already accepted. It still describes B's final
      // views, but cannot supply admission authority across the pending C transaction.
      if (!pending_root && !result.transport_failed) {
        for (std::size_t index = 0u; index < result.count; ++index) {
          const auto &receipt = result.receipts[index];
          if (publication_is_current(receipt.expected)) {
            cached_current_publication = receipt;
          }
        }
      }
      // Idle diagnostics may drain a slot before the rendering caller polls. Keep one current
      // proof, replacing the old scalar owner cache; no extra queue or duplicate GPU reads.
      if (cached_current_publication && publication_is_current(cached_current_publication->expected)) {
        result.current_publication = cached_current_publication;
      }
      return result;
    }

    void dispatch_pending_gpu_completion_trace() {
      if (!diagnostics_enabled) {
        return;
      }
      const auto append = pending_gpu_trace_append;
      if (!append.valid || !gpu_trace_cs || !near_identical_transaction.srv ||
          !gpu_trace_cbuffer || !gpu_trace_ring_uav || !subtitle_locator_state_srv ||
          !subtitle_condition_params_srv || !gpu_trace_provenance) {
        return;
      }
      // Append after complete postprocessing and before the shared completion snapshot, so its
      // cumulative counters include this transaction even when it is the final frame.
      // The immediate context orders this read before the next transaction upload. Its existing
      // raw SRV is safe to borrow here after finalization has unbound the transaction UAV.

      const auto domain_tag = near_identical_input_domain_tag(
        pending_root->input_region,
        pending_root->color_space,
        static_cast<std::uint32_t>(target_w),
        static_cast<std::uint32_t>(target_h)
      );
      const auto flags = pending_gpu_completion_flags(append);
      const auto submission_class =
        pending_root->submission_class == submission_class_e::gpu_undecided ?
          host_sbs_gpu_trace::submission_class_e::gpu_undecided :
          host_sbs_gpu_trace::submission_class_e::force_infer;
      const std::array<std::uint32_t, host_sbs_gpu_trace::constant_word_count> constants {
        static_cast<std::uint32_t>(pending_root->frame_id),
        static_cast<std::uint32_t>(pending_root->frame_id >> 32u),
        static_cast<std::uint32_t>(pending_root->input_region.analysis_generation),
        static_cast<std::uint32_t>(pending_root->input_region.analysis_generation >> 32u),
        static_cast<std::uint32_t>(domain_tag),
        static_cast<std::uint32_t>(domain_tag >> 32u),
        static_cast<std::uint32_t>(pending_root->transaction_token),
        static_cast<std::uint32_t>(pending_root->transaction_token >> 32u),
        cuda_conditional_graph::work_flags_value(pending_root->subtitle_work),
        static_cast<std::uint32_t>(submission_class),
        flags,
        static_cast<std::uint32_t>(append.host_subtitle_outcome),
        pending_root->input_region.width(),
        pending_root->input_region.height(),
        static_cast<std::uint32_t>(field_w),
        static_cast<std::uint32_t>(field_h),
        static_cast<std::uint32_t>(pending_root->observation_timestamp_us),
        static_cast<std::uint32_t>(pending_root->observation_timestamp_us >> 32u),
        0u,
        0u,
      };
      context->UpdateSubresource(
        gpu_trace_cbuffer.Get(), 0u, nullptr, constants.data(), 0u, 0u
      );
      ID3D11ShaderResourceView *inputs[3] = {
        near_identical_transaction.srv.Get(),
        subtitle_locator_state_srv.Get(),
        subtitle_condition_params_srv.Get(),
      };
      context->CSSetShader(gpu_trace_cs.Get(), nullptr, 0u);
      context->CSSetConstantBuffers(0u, 1u, gpu_trace_cbuffer.GetAddressOf());
      context->CSSetShaderResources(0u, 3u, inputs);
      ID3D11UnorderedAccessView *outputs[2] = {
        gpu_trace_ring_uav.Get(), gpu_outcome_uav.Get()
      };
      context->CSSetUnorderedAccessViews(
        0u, 2u, outputs, nullptr
      );
      context->Dispatch(1u, 1u, 1u);

      ID3D11ShaderResourceView *null_inputs[3] = {};
      ID3D11UnorderedAccessView *null_outputs[2] = {};
      ID3D11Buffer *null_constant = nullptr;
      context->CSSetShaderResources(0u, 3u, null_inputs);
      context->CSSetUnorderedAccessViews(0u, 2u, null_outputs, nullptr);
      context->CSSetConstantBuffers(0u, 1u, &null_constant);
      context->CSSetShader(nullptr, nullptr, 0u);

    }

    bool ensure_parallax_v2_coordinate_diagnostic_resource() {
      if (depth_coordinate_v2_coordinate_tex && depth_coordinate_v2_coordinate_uav &&
          depth_coordinate_v2_coordinate_srv) {
        return true;
      }
      if (!parallax_v2_producer_active || target_w <= 0 || target_h <= 0) {
        return false;
      }
      if (!depth_coordinate_v2_coordinate_diagnostic_cs) {
        const auto diagnostic_sources = host_sbs_shader_cache::snapshot_sources(
          shader_root,
          host_sbs_shader_cache::parallax_v2_coordinate_diagnostic_specs
        );
        if (!diagnostic_sources || !create_shader(diagnostic_sources, host_sbs_shader_cache::depth_coordinate_v2_coordinate_diagnostic, depth_coordinate_v2_coordinate_diagnostic_cs)) {
          if (!depth_coordinate_v2_coordinate_diagnostic_error_logged) {
            BOOST_LOG(warning)
              << "Host SBS V2 canonical-coordinate Dump 3D shader is unavailable; "
                 "live rendering remains active.";
            depth_coordinate_v2_coordinate_diagnostic_error_logged = true;
          }
          return false;
        }
      }

      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = static_cast<UINT>(field_w);
      desc.Height = static_cast<UINT>(field_h);
      desc.MipLevels = 1;
      desc.ArraySize = 1;
      desc.Format = DXGI_FORMAT_R32_FLOAT;
      desc.SampleDesc.Count = 1;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      const bool created =
        SUCCEEDED(device->CreateTexture2D(
          &desc,
          nullptr,
          depth_coordinate_v2_coordinate_tex.ReleaseAndGetAddressOf()
        )) &&
        SUCCEEDED(device->CreateShaderResourceView(
          depth_coordinate_v2_coordinate_tex.Get(),
          nullptr,
          depth_coordinate_v2_coordinate_srv.ReleaseAndGetAddressOf()
        )) &&
        SUCCEEDED(device->CreateUnorderedAccessView(
          depth_coordinate_v2_coordinate_tex.Get(),
          nullptr,
          depth_coordinate_v2_coordinate_uav.ReleaseAndGetAddressOf()
        ));
      if (!created) {
        depth_coordinate_v2_coordinate_tex.Reset();
        depth_coordinate_v2_coordinate_uav.Reset();
        depth_coordinate_v2_coordinate_srv.Reset();
        if (!depth_coordinate_v2_coordinate_diagnostic_error_logged) {
          BOOST_LOG(warning)
            << "Host SBS V2 could not allocate the Dump 3D canonical-coordinate texture; "
               "live rendering is unaffected.";
          depth_coordinate_v2_coordinate_diagnostic_error_logged = true;
        }
      }
      return created;
    }

    bool dispatch_parallax_v2_coordinate_diagnostic() {
      if (!ensure_parallax_v2_coordinate_diagnostic_resource()) {
        return false;
      }

      ID3D11Buffer *constant_buffers[2] = {
        cbuffer.Get(),
        depth_coordinate_v2_cbuffer.Get(),
      };
      ID3D11ShaderResourceView *inputs[3] = {
        tensor_out_srv.Get(),
        depth_coordinate_v2_state_srv.Get(),
        analysis_domain.srv.Get(),
      };
      context->CSSetShader(
        depth_coordinate_v2_coordinate_diagnostic_cs.Get(),
        nullptr,
        0
      );
      context->CSSetConstantBuffers(0, 2, constant_buffers);
      context->CSSetShaderResources(0, 3, inputs);
      context->CSSetUnorderedAccessViews(
        0,
        1,
        depth_coordinate_v2_coordinate_uav.GetAddressOf(),
        nullptr
      );
      context->Dispatch(
        (static_cast<UINT>(field_w) + 15u) / 16u,
        (static_cast<UINT>(field_h) + 15u) / 16u,
        1u
      );

      ID3D11ShaderResourceView *null_srvs[3] = {nullptr, nullptr, nullptr};
      ID3D11UnorderedAccessView *null_uav = nullptr;
      ID3D11Buffer *null_constant = nullptr;
      context->CSSetShaderResources(0, 3, null_srvs);
      context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
      context->CSSetConstantBuffers(1, 1, &null_constant);
      return true;
    }

    struct completion_snapshot_validity_t {
      bool raw_model_depth = false;
      bool model_input = false;
      bool composite = false;
      bool coordinate = false;
      bool base_diagnostics = false;
    };

    struct completed_depth_metadata_t {
      bool valid = false;
      std::uint64_t frame_id = 0u;
      depth_input_region_t input_region {};
      input_color_space color_space = input_color_space::srgb;
      bool input_domain_reset = false;
      bool subtitle_work_suppressed = false;
      bool gpu_undecided = false;
      completion_snapshot_validity_t snapshots;
      std::shared_ptr<const subtitle_publication::views_t> subtitle_draw_publication;
    };

    // Both completion paths call this only after ordered postprocess, receipt and debug copies.
    // Retain the completed source/domain and publication views; consume only pending authority.
    completed_depth_metadata_t consume_pending_completion(
      const bool input_domain_reset,
      const completion_snapshot_validity_t snapshots
    ) {
      completed_depth_metadata_t completed {
        .valid = true,
        .frame_id = pending_root->frame_id,
        .input_region = pending_root->input_region,
        .color_space = pending_root->color_space,
        .input_domain_reset = input_domain_reset,
        .subtitle_work_suppressed = depth_optional_work_suppresses_subtitle(pending_root->optional_work),
        .gpu_undecided = gpu_undecided_postprocess_pending(),
        .snapshots = snapshots,
      };
      if (!completed.gpu_undecided) {
        gpu_undecided_requires_force_infer_refresh = false;
      }
      if (subtitle_direct_glyph.enabled() && !completed.subtitle_work_suppressed &&
          completed_contour.subtitle_selected_matte && completed_contour.subtitle_selected_matte_receipt) {
        completed.subtitle_draw_publication = subtitle_draw_pool.publish(device.Get(), context.Get(), {
          depth_coordinate_v2_final_srv.Get(), depth_coordinate_v2_state_srv.Get(),
          completed_contour.subtitle_selected_matte.Get(), completed_contour.source.Get(),
          completed_contour.condition_params.Get(), completed_contour.identity.Get(),
          completed_contour.subtitle_selected_matte_receipt.Get()
        });
        if (!completed.subtitle_draw_publication && !subtitle_draw_copy_failure_logged) {
          BOOST_LOG(warning) << "Could not retain the completed Direct Glyph draw inputs; using ordinary scene depth.";
          subtitle_draw_copy_failure_logged = true;
        } else if (completed.subtitle_draw_publication) {
          subtitle_draw_copy_failure_logged = false;
        }
      }
      completed_subtitle_draw = completed.subtitle_draw_publication;
      completed_input = {
        .frame_id = pending_root->frame_id,
        .region = pending_root->input_region,
        .subtitle_context = pending_root->subtitle_context,
      };
      pending_root.reset();
      clear_pending_inference_event_state();
      if (diagnostics_enabled) {
        throughput_stats_completions++;
      }
      return completed;
    }

    depth_step_result_t make_result(
      const completed_depth_metadata_t &completed,
      const depth_submission_t &submission = {}
    ) {
      depth_step_result_t step;
      step.submission = submission;
      step.status.cuda_graph_active = depth_conditional_graph.ready() &&
                                      !depth_inference_graph.policy.capture_failed;
      if (!completed.valid) {
        return step;
      }

      auto &r = step.observation;
      r.depth = normalized_depth.srv;
      r.depth_health_state = depth_health_state_srv;
      r.depth_frame_state = minmax_ema_srv;
      r.ema_motion_mask = ema_motion_mask.srv;
      r.raw_model_depth = tensor_out_srv;
      if (completed.snapshots.raw_model_depth) {
        r.raw_model_depth_snapshot = raw_snapshot_srv;
      }
      if (completed.snapshots.model_input) {
        r.model_input_snapshot = model_input_snapshot_srv;
      }
      r.raw_model_provenance = raw_model_provenance;
      r.composite_depth_runtime_provenance = composite_depth_runtime_provenance;
      if (composite_depth_runtime_provenance) {
        r.guidance_width = target_w;
        r.guidance_height = target_h;
        r.refined_width = target_w;
        r.refined_height = target_h;
        if (completed.snapshots.composite) {
          r.guidance_model_input_snapshot = r.model_input_snapshot;
          r.refined_model_depth_snapshot = r.raw_model_depth_snapshot;
        }
      }
      if (parallax_v2_producer_active) {
        if (completed.snapshots.coordinate) {
          r.shadow_coordinate = depth_coordinate_v2_coordinate_srv;
        }
        r.base_diagnostics_omitted = !completed.snapshots.base_diagnostics;
        if (!r.base_diagnostics_omitted) {
          r.shadow_candidate_parallax = depth_coordinate_v2_candidate_srv;
          r.shadow_vertical_majorant = depth_coordinate_v2_vertical_majorant_srv;
          r.shadow_vertical_conditioned = depth_coordinate_v2_vertical_conditioned_srv;
        }
        r.shadow_base_final_parallax = depth_coordinate_v2_final_srv;
        r.subtitle_conditioning_diagnostics_current = completed_subtitle_conditioning_diagnostics;
        if (r.subtitle_conditioning_diagnostics_current) {
          r.shadow_conditioned_final_parallax = subtitle_conditioning_diagnostics->final.srv;
        }
        r.shadow_state = depth_coordinate_v2_state_srv;
        r.shadow_frame_stats = depth_coordinate_v2_frame_stats_srv;
        const auto calibration_shape = calibrated_dav2_shape();
        if (calibration_shape &&
            host_sbs_v2_depth_shape_is_authenticated(*calibration_shape)) {
          r.ocr_box_record = ocr_box_record_srv;
          r.subtitle_locator_state = subtitle_locator_state_srv;
          r.subtitle_contour_support = completed_contour.support;
          if (!completed.subtitle_work_suppressed || subtitle_legacy_matte_required()) {
            r.subtitle_glyph_matte = completed_contour.glyph_matte;
          }
          r.subtitle_selected_matte = completed_contour.subtitle_selected_matte;
          r.subtitle_selected_matte_receipt = completed_contour.subtitle_selected_matte_receipt;
          r.subtitle_condition_params = completed_contour.condition_params;
          r.subtitle_contour_identity = completed_contour.identity;
          r.subtitle_contour_history = completed_contour.history;
          r.subtitle_contour_source = completed_contour.source;
          r.subtitle_ocr_probability = completed_contour.probability;
          if (subtitle_direct_glyph.enabled()) {
            r.subtitle_draw_publication = completed.subtitle_draw_publication;
            if (r.subtitle_draw_publication) {
              const auto &srv = r.subtitle_draw_publication->srv;
              r.shadow_base_final_parallax = srv[subtitle_publication::base];
              r.shadow_state = srv[subtitle_publication::state];
              r.subtitle_selected_matte = srv[subtitle_publication::matte];
              r.subtitle_contour_source = srv[subtitle_publication::display_band];
              r.subtitle_condition_params = srv[subtitle_publication::condition];
              r.subtitle_contour_identity = srv[subtitle_publication::identity];
              r.subtitle_selected_matte_receipt = srv[subtitle_publication::receipt];
            } else {
              r.subtitle_selected_matte.Reset();
              r.subtitle_selected_matte_receipt.Reset();
            }
          }
          // Always-on evaluation can inspect the held or cleared auxiliary fields when
          // suppression produced no new conditioning observation. Final still comes from Base.
          if (subtitle_conditioning_diagnostics &&
              (r.subtitle_conditioning_diagnostics_current ||
               (completed.subtitle_work_suppressed && subtitle_conditioning_diagnostics_always))) {
            const auto &diagnostics = *subtitle_conditioning_diagnostics;
            r.subtitle_contour_field_pin = diagnostics.field_mask.srv;
            r.subtitle_contour_distance = diagnostics.distance.srv;
            r.subtitle_gaussian_weight = diagnostics.weight.srv;
            r.subtitle_gaussian_blend = diagnostics.blend.srv;
            r.subtitle_gaussian_vertical_conditioned = diagnostics.vertical_conditioned.srv;
            r.subtitle_gaussian_control = diagnostics.control.srv;
          }
        }
        r.parallax_v2_shader_provenance = parallax_v2_shader_provenance;
        if (gpu_trace_ring_srv && gpu_trace_provenance) {
          r.gpu_trace_ring = gpu_trace_ring_srv;
          r.gpu_trace_provenance = gpu_trace_provenance;
        }
        r.parallax_v2_producer_active = true;
        r.parallax_v2_raw_coordinate_scale = parallax_v2_raw_coordinate_scale;
        r.parallax_v2_requested_pop_strength = parallax_v2_requested_pop_strength;
        r.parallax_v2_requested_gain = parallax_v2_requested_gain;
        r.parallax_v2_joint_plane_mode = joint_plane_mode;
      }
      r.raw_width = target_w;
      r.raw_height = target_h;
      r.field_width = field_w;
      r.field_height = field_h;
      r.completed_frame_valid = true;
      r.completed_frame_id = completed.frame_id;
      r.gpu_undecided_completion = completed.gpu_undecided;
      if (current_publication.frame_id == completed.frame_id) {
        r.publication = current_publication;
      }
      r.input_region = completed.input_region;
      r.color_space = completed.color_space;
      step.completion.input_domain_reset = completed.input_domain_reset;
      r.subtitle_work_suppressed = completed.subtitle_work_suppressed;
      r.field_content = live_field_content(completed.input_region).value_or(depth_tensor_content_rect_t {});
      return step;
    }

    depth_step_result_t make_result() {
      return make_result(completed_depth_metadata_t {});
    }

    bool snapshot_buffer(
      ID3D11Buffer *source,
      Microsoft::WRL::ComPtr<ID3D11Buffer> &snapshot,
      Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> &snapshot_srv,
      bool &error_logged,
      unsigned &retry_frames,
      std::string_view label,
      const std::optional<UINT> logical_element_count = std::nullopt
    ) {
      if (!source) {
        return false;
      }
      if (retry_frames > 0) {
        --retry_frames;
        return false;
      }

      D3D11_BUFFER_DESC source_desc {};
      source->GetDesc(&source_desc);
      bool recreate = !snapshot || !snapshot_srv;
      if (!recreate) {
        D3D11_BUFFER_DESC snapshot_desc {};
        snapshot->GetDesc(&snapshot_desc);
        recreate = snapshot_desc.ByteWidth != source_desc.ByteWidth ||
                   snapshot_desc.StructureByteStride != source_desc.StructureByteStride ||
                   snapshot_desc.MiscFlags != source_desc.MiscFlags;
        if (!recreate && logical_element_count) {
          D3D11_SHADER_RESOURCE_VIEW_DESC snapshot_srv_desc {};
          snapshot_srv->GetDesc(&snapshot_srv_desc);
          recreate = snapshot_srv_desc.ViewDimension != D3D11_SRV_DIMENSION_BUFFER ||
                     snapshot_srv_desc.Buffer.FirstElement != 0u ||
                     snapshot_srv_desc.Buffer.NumElements != *logical_element_count;
        }
      }
      if (recreate) {
        snapshot.Reset();
        snapshot_srv.Reset();
        D3D11_SHADER_RESOURCE_VIEW_DESC logical_srv_desc {};
        const D3D11_SHADER_RESOURCE_VIEW_DESC *srv_desc = nullptr;
        if (logical_element_count) {
          logical_srv_desc.Format = DXGI_FORMAT_UNKNOWN;
          logical_srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
          logical_srv_desc.Buffer.FirstElement = 0u;
          logical_srv_desc.Buffer.NumElements = *logical_element_count;
          srv_desc = &logical_srv_desc;
        }
        if (FAILED(device->CreateBuffer(&source_desc, nullptr, &snapshot)) ||
            FAILED(device->CreateShaderResourceView(
              snapshot.Get(), srv_desc, &snapshot_srv
            ))) {
          snapshot.Reset();
          snapshot_srv.Reset();
          if (!error_logged) {
            BOOST_LOG(warning) << "Depth estimator could not allocate the Dump 3D " << label
                               << " snapshot.";
            error_logged = true;
          }
          // Keep a retained request recoverable after transient device pressure without retrying
          // two allocations on every completed inference indefinitely.
          retry_frames = 120;
          return false;
        }
      }
      context->CopyResource(snapshot.Get(), source);
      retry_frames = 0;
      return true;
    }

    // The caller has proved the root ready and prepared its constants. Retire that observation
    // before a later submission changes either the constants or the raw tensors.
    completed_depth_metadata_t complete_ready_pending(
      d3d_perf_slot *d3d_timer,
      const bool request_snapshot,
      const bool evaluation_wait = false
    ) {
      const bool snapshot_requested = subtitle_direct_glyph_tail ? subtitle_direct_glyph_tail->snapshot_requested : request_snapshot;
      bool input_domain_reset;
      if (subtitle_direct_glyph_tail) {
        input_domain_reset = subtitle_direct_glyph_tail->input_domain_reset;
      } else {
        input_domain_reset = prepare_pending_input_domain();
        completion_dump_snapshot_requested = snapshot_requested;
        subtitle_direct_glyph_requested = false;
        if (!normalize_depth_output(d3d_timer, input_domain_reset)) {
          mark_d3d_post_end(d3d_timer);
          return {};
        }
        if (subtitle_direct_glyph_requested) dispatch_subtitle_direct_glyph(input_domain_reset);
      }
      if (subtitle_direct_glyph_requested) {
        if (!retire_subtitle_direct_glyph(evaluation_wait)) {
          subtitle_direct_glyph_tail.emplace(subtitle_direct_glyph_tail_t {input_domain_reset, snapshot_requested});
          mark_d3d_post_end(d3d_timer);
          return {};
        }
        subtitle_direct_glyph_tail.reset();
        if (subtitle_selected_matte_commands) {
          const auto selected = subtitle_direct_glyph.views();
          completed_contour.subtitle_selected_matte = selected.matte;
          completed_contour.subtitle_selected_matte_receipt = selected.receipt;
        }
      }

      dispatch_pending_gpu_completion_trace();
      snapshot_pending_publication_receipt();
      pending_gpu_trace_append = {};
      mark_d3d_post_end(d3d_timer);

      // Dump 3D copies the exact completed input and output, outside the live postprocess interval.
      // For ROI this uses the pending crop-local domain and its exact tensor-content rectangle.
      completion_snapshot_validity_t snapshots;
      // An indirect dispatch can preserve padding from an earlier compact completion. Expose
      // the diagnostic fields only when their writer is known to have filled the full grid.
      snapshots.base_diagnostics = normalized_depth_diagnostics_enabled ||
                                   (snapshot_requested && !gpu_undecided_postprocess_pending());
      if (snapshot_requested && !gpu_undecided_postprocess_pending()) {
        snapshots.coordinate = dispatch_parallax_v2_coordinate_diagnostic();
        snapshots.raw_model_depth = snapshot_buffer(
          tensor_out_buf.Get(),
          raw_snapshot_buf,
          raw_snapshot_srv,
          raw_snapshot_error_logged,
          raw_snapshot_retry_frames,
          "raw-depth",
          static_cast<UINT>(target_w * target_h)
        );
        snapshots.model_input = snapshot_buffer(
          tensor_in_buf.Get(),
          model_input_snapshot_buf,
          model_input_snapshot_srv,
          model_input_snapshot_error_logged,
          model_input_snapshot_retry_frames,
          "model-input"
        );
        snapshots.composite = composite_depth_runtime_provenance &&
                              snapshots.raw_model_depth && snapshots.model_input;
      }
      return consume_pending_completion(input_domain_reset, snapshots);
    }

    // estimate() has already submitted one inference. Wait for that exact inference, consume it
    // once, and deliberately do NOT enqueue a duplicate. This is the synchronous quality oracle.
    depth_step_result_t finish_pending(
      input_color_space color_space,
      const bool snapshot_debug_inputs = false,
      const bool root_completion_proven = false
    ) {
      // A caller may first prove nonblocking readiness with can_accept(). Consuming instead of
      // enqueueing must retire that admission token so a later estimate cannot skip its own query.
      readiness_preflighted = false;
      auto &cuda = cuda_driver_api::get();
      if (!pending_root) {
        return make_result();
      }
      if (cuda_ctx && cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
        BOOST_LOG(error) << "cuCtxSetCurrent failed while finishing pending depth.";
        mark_cuda_context_failure();
        return make_result();
      }
      // A nonblocking or bounded try-finish caller may prove the joined root complete through its
      // post-unmap event or, when events are unavailable, a successful root-stream query. D3D11
      // postprocess is then ordered behind the same release boundary without another synchronize.
      // Offline evaluation alone retains the conservative synchronization path.
      if (!root_completion_proven && !synchronize_pending_execution(cuda)) {
        return make_result();
      }
      if (diagnostics_enabled) {
        resolve_completed_branch_timing();
        perf_drain();
        if (is_terminal()) {
          return make_result();
        }
      }
      (void) color_space;  // the pending frame owns its transfer mode
      ensure_cbuffers(pending_root->color_space, pending_root->input_region);
      ensure_source_region_cbuffer(pending_root->input_region);
      if (!cbuffer || !source_region_cbuffer) {
        // A persistent constant-buffer allocation failure must latch terminal like every other
        // resource failure; returning empty without latching leaves the caller retrying at the
        // minimum-FPS cadence forever.
        BOOST_LOG(error)
          << "Depth/source-region constant-buffer creation failed while finishing a pending frame";
        mark_terminal_failure();
        return make_result();
      }
      auto *d3d_timer = diagnostics_enabled ? begin_d3d_perf(true, false) : nullptr;
      const auto completed = complete_ready_pending(
        d3d_timer, snapshot_debug_inputs && !gpu_undecided_postprocess_pending(), !root_completion_proven
      );
      mark_d3d_pre_start(d3d_timer);
      end_d3d_perf(d3d_timer);
      return make_result(completed);
    }

    pending_depth_poll_result try_finish_pending(
      input_color_space color_space,
      bool snapshot_debug_inputs,
      const bool allow_wait,
      const std::chrono::steady_clock::time_point deadline,
      const std::uint32_t max_queries,
      const bool allow_completion_notification,
      const pending_depth_source_access_t source_access
    ) {
      readiness_preflighted = false;
      auto &cuda = cuda_driver_api::get();
      const auto started = std::chrono::steady_clock::now();
      const bool use_inference_events = inference_event_poll_available;
      const bool repeated_wait_allowed = allow_wait && use_inference_events;
      pending_depth_poll_result polled;
#ifdef SUNSHINE_TESTS
      polled.notification_callback_pending = root_completion_notification.callback_pending();
#endif
      if (!pending_root) {
        polled.result = make_result();
        polled.ready = true;
        return polled;
      }
      if (cuda_ctx && cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
        mark_cuda_context_failure();
        polled.result = make_result();
        polled.ready = true;
        return polled;
      }

      const auto query_limit = std::max<std::uint32_t>(1u, max_queries);
      constexpr auto repeated_query_interval = std::chrono::microseconds {50};
      constexpr auto deadline_poll_tail = std::chrono::milliseconds {1};
      while (true) {
        // The first joined query is always immediate. Before every repeated query, preserve the
        // absolute deadline even if the thread was descheduled beyond its intended poll interval.
        if (
          polled.query_count > 0u &&
          std::chrono::steady_clock::now() >= deadline
        ) {
          polled.timed_out = repeated_wait_allowed;
          polled.wait_duration = std::chrono::steady_clock::now() - started;
          return polled;
        }
        ++polled.query_count;
        const auto readiness = use_inference_events ?
                                 query_pending_inference_events(
                                   cuda,
                                   allow_wait ? "bounded same-frame completion" :
                                                "nonblocking completion"
                                 ) :
                                 query_pending_execution(
                                   cuda,
                                   "same-frame completion fallback"
                                 );
        switch (readiness) {
          case pending_execution_readiness_e::busy: {
            const auto now = std::chrono::steady_clock::now();
            if (
              !repeated_wait_allowed || polled.query_count >= query_limit ||
              now >= deadline
            ) {
              polled.timed_out = repeated_wait_allowed;
              polled.wait_duration = now - started;
              return polled;
            }
            polled.wait_attempted = true;
            // Keep the final portion of the budget on the 50 us query cadence. The notification
            // is a wake hint; only another literal joined query can prove release of the root.
            const auto next_query_at = std::min(
              deadline,
              now + repeated_query_interval
            );
            const auto notification_cutoff = deadline - deadline_poll_tail;
            if (
              allow_completion_notification && !completion_notification_disabled &&
              polled.query_count == 1u && next_query_at < notification_cutoff &&
              platf::host_sbs_completion_wait_available()
            ) {
              const auto armed = root_completion_notification.arm(cuda, depth_inference_done_event);
#ifdef SUNSHINE_TESTS
              polled.notification_callback_pending |= root_completion_notification.callback_pending();
              polled.notification_armed = armed == CUDA_SUCCESS;
#endif
              if (armed && *armed != CUDA_SUCCESS) {
                (void) observe_joined_cuda_failure("completion notification", *armed, true);
                polled.result = make_result();
                polled.ready = true;
                polled.wait_duration = std::chrono::steady_clock::now() - started;
                return polled;
              }
              if (armed && std::chrono::steady_clock::now() < notification_cutoff) {
#ifdef SUNSHINE_TESTS
                polled.notification_waited = true;
#endif
                const bool waited = root_completion_notification.wait_until(notification_cutoff);
                completion_notification_disabled = !waited || std::chrono::steady_clock::now() >= deadline;
              }
            }
            while (std::chrono::steady_clock::now() < next_query_at) {
              std::this_thread::yield();
            }
            continue;
          }
          case pending_execution_readiness_e::failed:
            polled.result = make_result();
            polled.ready = true;
            polled.wait_duration = std::chrono::steady_clock::now() - started;
            return polled;
          case pending_execution_readiness_e::ready:
            polled.wait_duration = std::chrono::steady_clock::now() - started;
            break;
        }
        break;
      }
      if (source_access.acquire && !source_access.acquire(source_access.owner)) {
        return polled;
      }
      polled.result = finish_pending(
        color_space,
        snapshot_debug_inputs,
        true
      );
      polled.ready = !subtitle_direct_glyph_tail || is_terminal();
      return polled;
    }

    pending_depth_poll_result try_finish_pending_nonblocking(
      input_color_space color_space,
      const bool snapshot_debug_inputs,
      const pending_depth_source_access_t source_access
    ) {
      return try_finish_pending(
        color_space,
        snapshot_debug_inputs,
        false,
        std::chrono::steady_clock::time_point {},
        1u,
        false,
        source_access
      );
    }

    pending_depth_poll_result try_finish_pending_until(
      input_color_space color_space,
      const std::chrono::steady_clock::time_point deadline,
      const std::uint32_t max_queries,
      bool snapshot_debug_inputs,
      bool allow_completion_notification,
      const pending_depth_source_access_t source_access
    ) {
      return try_finish_pending(
        color_space,
        snapshot_debug_inputs,
        true,
        deadline,
        max_queries,
        allow_completion_notification,
        source_access
      );
    }

    // (Re)build the depth constant buffer. The fixed tensor dimensions stay session-constant,
    // while transfer mode and an ROI's centered tensor-content rectangle follow the exact pending
    // frame. Domain changes are rare, so an immutable replacement keeps all dispatches simple.
    void ensure_cbuffers(
      input_color_space color_space,
      const depth_input_region_t &input_region
    ) {
      const int color_mode = (int) color_space;
      const auto tensor_content = input_region.tensor_content.valid() ?
                                    input_region.tensor_content :
                                    depth_tensor_content_rect_t {
                                      0u, 0u,
                                      static_cast<std::uint32_t>(std::max(target_w, 0)),
                                      static_cast<std::uint32_t>(std::max(target_h, 0)),
                                    };
      if (cb_color_mode == color_mode && cb_tensor_content == tensor_content && cbuffer) {
        return;
      }
      cb_color_mode = color_mode;
      cb_tensor_content = tensor_content;

      D3D11_BUFFER_DESC cb_desc = {};
      cb_desc.Usage = D3D11_USAGE_IMMUTABLE;
      cb_desc.ByteWidth = 64;  // shared depth-pass cbuffer (16 floats/uints; see below)
      cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

      // Shared depth-pass constants, 16 scalars = 4 float4 registers. THIS fill is the
      // single source of truth for the canonical layout in
      // shaders/directx/include/depth_constants.hlsl -- every cbf[N] below must stay
      // slot-for-slot with the include (which every depth shader #includes). To add a
      // field: append it here AND to the include.
      uint32_t cb[16] = {};
      float *cbf = (float *) cb;
      cb[0] = (uint32_t) target_w;
      cb[1] = (uint32_t) target_h;
      cb[2] = (uint32_t) color_mode;
      cbf[3] = ema_alpha;
      cbf[4] = minmax_alpha;
      cb[5] = reduce_groups * 256u;  // total threads for the reduction grid-stride
      cbf[6] = ema_edge_change;
      cbf[7] = ema_edge_gradient;
      cbf[8] = ema_edge_strength;
      cb[9] = tensor_content.left;
      cb[10] = tensor_content.top;
      cb[11] = tensor_content.right;
      cb[12] = tensor_content.bottom;
      D3D11_SUBRESOURCE_DATA sd = {cb, 0, 0};
      cbuffer.Reset();
      device->CreateBuffer(&cb_desc, &sd, &cbuffer);
    }

    [[nodiscard]] depth_tensor_shape_t live_field_shape() const noexcept {
      return {field_w, field_h};
    }

    [[nodiscard]] std::optional<depth_tensor_content_rect_t> live_field_content(
      const depth_input_region_t &input_region
    ) const noexcept {
      if (!input_region.tensor_content.valid({target_w, target_h})) {
        return std::nullopt;
      }
      return input_region.tensor_content;
    }

    // Bind the physical rectangle inside the retained full matched frame separately from the
    // analysis-domain constants. All shader resize/coordinate math remains crop-local; only its
    // final integer Texture2D.Load address receives this offset. Translation changes this buffer
    // without changing temporal analysis-domain identity.
    void ensure_source_region_cbuffer(const depth_input_region_t &input_region) {
      const depth_source_rect_t source_region {
        input_region.left,
        input_region.top,
        input_region.right,
        input_region.bottom,
      };
      if (source_region_cbuffer && cb_source_region == source_region) {
        return;
      }
      if (!source_region_cbuffer) {
        D3D11_BUFFER_DESC cb_desc {};
        cb_desc.Usage = D3D11_USAGE_DEFAULT;
        cb_desc.ByteWidth = 16u;
        cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (FAILED(device->CreateBuffer(
              &cb_desc, nullptr, &source_region_cbuffer
            ))) {
          return;
        }
      }
      const std::array<std::uint32_t, 4> constants {
        source_region.left,
        source_region.top,
        source_region.right,
        source_region.bottom,
      };
      context->UpdateSubresource(
        source_region_cbuffer.Get(), 0u, nullptr, constants.data(), 0u, 0u
      );
      cb_source_region = source_region;
    }

    void reset_temporal_state_for_input_domain() {
      invalidate_publication();
      // Domain changes are rare (video selection/extent or full-frame fallback). Reset every
      // history that could otherwise compare browser pixels with video-local pixels. All writes
      // are ordered on the owning immediate context; there is no CPU readback or synchronization.
      const std::uint32_t raw_identity[4] = {0xFFFFFFFFu, 0u, 0u, 0u};
      const float zero4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      const UINT zero_uint4[4] = {0u, 0u, 0u, 0u};
      if (minmax_raw_buf) {
        context->UpdateSubresource(minmax_raw_buf.Get(), 0, nullptr, raw_identity, 0, 0);
      }
      if (minmax_ema_buf) {
        context->UpdateSubresource(minmax_ema_buf.Get(), 0, nullptr, zero4, 0, 0);
      }
      if (depth_health_state_buf) {
        context->UpdateSubresource(
          depth_health_state_buf.Get(),
          0,
          nullptr,
          sbs_adaptive_state::initial_words.data(),
          0,
          0
        );
      }
      if (tensor_previous_input_uav) {
        context->ClearUnorderedAccessViewUint(tensor_previous_input_uav.Get(), zero_uint4);
      }
      if (near_identical_history_owner.uav) {
        context->ClearUnorderedAccessViewUint(
          near_identical_history_owner.uav.Get(), zero_uint4
        );
      }
      clear_normalized_depth_diagnostics();
      if (subtitle_locator_state_uav) {
        context->ClearUnorderedAccessViewUint(
          subtitle_locator_state_uav.Get(),
          zero_uint4
        );
      }
      if (depth_coordinate_v2_state_buf) {
        context->UpdateSubresource(
          depth_coordinate_v2_state_buf.Get(),
          0,
          nullptr,
          depth_coordinate_v2::state_initial_words.data(),
          0,
          0
        );
      }
      for (auto *uav : {
             depth_coordinate_v2_partials_uav.Get(),
             depth_coordinate_v2_frame_stats_uav.Get(),
           }) {
        if (uav) {
          context->ClearUnorderedAccessViewUint(uav, zero_uint4);
        }
      }
      for (auto *uav : {
             depth_coordinate_v2_coordinate_uav.Get(),
             depth_coordinate_v2_candidate_uav.Get(),
             depth_coordinate_v2_vertical_majorant_uav.Get(),
             depth_coordinate_v2_vertical_conditioned_uav.Get(),
             depth_coordinate_v2_final_uav.Get(),
             depth_coordinate_v2_eye_final_uav.Get(),
           }) {
        if (uav) {
          context->ClearUnorderedAccessViewFloat(uav, zero4);
        }
      }
      if (const auto &diagnostics = subtitle_conditioning_diagnostics) {
        context->ClearUnorderedAccessViewFloat(diagnostics->final.uav.Get(), zero4);
      }
      completed_subtitle_conditioning_diagnostics = false;
    }

    bool prepare_pending_input_domain() {
      const bool changed = processed_input_domain.update(
        pending_root->input_region,
        pending_root->color_space
      );
      if (changed) {
        reset_temporal_state_for_input_domain();
      }
      return changed;
    }

    void clear_normalized_depth_diagnostics() {
      if (!normalized_depth.ready() || !previous_normalized_depth.ready() || !ema_motion_mask.ready()) {
        return;
      }
      const float zero_depth[4] {};
      const UINT zero_mask[4] {};
      context->ClearUnorderedAccessViewFloat(normalized_depth.uav.Get(), zero_depth);
      context->ClearUnorderedAccessViewFloat(previous_normalized_depth.uav.Get(), zero_depth);
      context->ClearUnorderedAccessViewUint(ema_motion_mask.uav.Get(), zero_mask);
    }

    void dispatch_normalized_depth_diagnostics() {
      // Rotate the two identically provisioned depth textures. The old current side becomes the
      // immutable previous-frame SRV, while the older side becomes this completion's UAV target.
      // This preserves the exact EMA history without a full-field CopyResource every frame.
      std::swap(normalized_depth, previous_normalized_depth);

      // Buffer to Texture: normalize disparity, classify the deterministic moving-edge
      // stencil, apply temporal EMA, and publish the diagnostic mask in one content dispatch.
      // Consuming the mask decision in-register removes the R32 mask round trip while retaining
      // the exact independently observable mask. Contain-fit padding is filled afterward by a
      // lightweight UAV copy entry point, avoiding repeated boundary-stencil work. Full-content
      // fields remain one dispatch. MinMaxEma.frame_state makes the first valid frame snap and
      // makes an all-invalid frame hold entirely on the GPU.
      context->CSSetShader(buffer_to_tex_cs.Get(), nullptr, 0);
      context->CSSetConstantBuffers(0, 1, cbuffer.GetAddressOf());
      ID3D11ShaderResourceView *bt_srvs[4] = {
        tensor_out_srv.Get(),
        minmax_ema_srv.Get(),
        previous_normalized_depth.srv.Get(),
        analysis_domain.srv.Get(),
      };
      ID3D11UnorderedAccessView *bt_uavs[2] = {
        normalized_depth.uav.Get(),
        ema_motion_mask.uav.Get(),
      };
      context->CSSetShaderResources(0, 4, bt_srvs);
      context->CSSetUnorderedAccessViews(0, 2, bt_uavs, nullptr);

      dispatch_infer_postprocess(
        (target_w + 15) / 16,
        (target_h + 15) / 16,
        1u,
        near_identical_gpu_infer_grid16_byte_offset
      );

      const auto &tensor_content = pending_root->input_region.tensor_content;
      const bool has_synthetic_padding =
        tensor_content.left != 0u || tensor_content.top != 0u ||
        tensor_content.right != target_w || tensor_content.bottom != target_h;
      if (has_synthetic_padding) {
        // The preceding dispatch has completed its UAV accesses before this dispatch begins.
        // pad_main reads only an already-written admitted boundary texel and writes one excluded
        // destination, so no cross-thread or cross-group dependency exists inside this pass.
        context->CSSetShader(buffer_to_tex_pad_cs.Get(), nullptr, 0);
        dispatch_infer_postprocess(
          (target_w + 15) / 16,
          (target_h + 15) / 16,
          1u,
          near_identical_gpu_infer_grid16_byte_offset
        );
      }

      ID3D11UnorderedAccessView *null_uav2[2] = {nullptr, nullptr};
      ID3D11ShaderResourceView *null_srvs[4] = {
        nullptr, nullptr, nullptr, nullptr
      };
      context->CSSetUnorderedAccessViews(0, 2, null_uav2, nullptr);
      context->CSSetShaderResources(0, 4, null_srvs);
      dispatch_near_identical_reuse_depth();
    }

    // Publish the live V2 depth and subtitle state from finished raw disparity. The shared
    // scale passes also supply the evaluator's optional normalized texture and motion mask.
    bool normalize_depth_output(
      d3d_perf_slot *perf_slot,
      const bool input_domain_reset
    ) {
      // Borrowed SRVs stop identifying the prior publication before any write, including a
      // failed attempt or a repeated caller frame ID. Readback slots keep their immutable key.
      invalidate_publication();
      ++publication_sequence;
      pending_gpu_trace_append = {};
      const bool subtitle_publication_branch_opaque =
        gpu_undecided_postprocess_pending();
      if (!dispatch_near_identical_finalizer(input_domain_reset)) {
        // A conditional completion without its single receipt/argument/abstention finalizer must
        // not consume possibly stale tensor_out or mutate history. This is a mandatory active-path
        // resource failure: retain prior GPU state but terminally fail flat.
        fail_gpu_conditional_bridge_once(
          "GPU-undecided postprocess finalization failed"
        );
        return false;
      }
      // DAV2 and optional OCR are sibling nodes under one conditional CUDA root. Its completion
      // event follows both children and every interop-unmap tail, so both outputs are safe before
      // any next-frame mapping reuses the fixed buffers.
      const bool subtitle_work_suppressed =
        depth_optional_work_suppresses_subtitle(pending_root->optional_work);
      // Native move/resize and ineligible video contexts keep depth/camera live and publish
      // fresh Base. Move/resize freezes same-context subtitle memory; context admission separately
      // clears subtitle state when entering/leaving hard disable or changing the scope epoch.
      bool ocr_record_submitted = false;
      auto trace_host_subtitle_outcome =
        host_sbs_gpu_trace::host_subtitle_outcome_e::suppressed;
      if (subtitle_work_suppressed) {
        pending_root->ocr_armed = false;
      } else {
        ocr_record_submitted = dispatch_ocr_postprocess();
        if (ocr_record_submitted) {
          trace_host_subtitle_outcome =
            host_sbs_gpu_trace::host_subtitle_outcome_e::ordinary_record;
        }
      }
      if (!subtitle_work_suppressed && !ocr_record_submitted) {
        // Never publish a nominally exact completion whose OCR8 identity may still describe an
        // older frame. The expected-work fallback makes OCR-unavailable ordinary frames a valid
        // current abstention; reaching this branch is therefore a mandatory producer failure.
        mark_terminal_failure();
        if (!ocr_error_logged) {
          BOOST_LOG(error)
            << "Host SBS could not publish the authenticated current-frame OCR record; "
               "the V2 producer is terminal.";
          ocr_error_logged = true;
        }
        return false;
      }
      // Resolve quantiles, normalization and camera state from the shared scan before mapping.
      // The authenticated zero-group argument preserves all three on depth reuse.
      if (!dispatch_parallax_v2_frame_publication(perf_slot)) {
        mark_terminal_failure();
        return false;
      }

      if (normalized_depth_diagnostics_enabled) {
        dispatch_normalized_depth_diagnostics();
      }
      // Only an explicitly maintained depth-only baseline needs the suppressed input tensor.
      // The depth receipt still needs its owner even when the map skips the copy.
      const bool skip_input_history =
        subtitle_work_suppressed && !pending_root->depth_only_history && !subtitle_publication_branch_opaque;
      if (!upload_near_identical_history_owner_constants(skip_input_history)) {
        fail_gpu_conditional_bridge_once(
          "GPU inference-observation history-owner constants failed"
        );
        return false;
      }

      if (parallax_v2_producer_active) {
        // Join this mapping interval with the earlier frame-publication interval. Normalized
        // diagnostic textures are outside the sum. Both intervals sit inside depth_postprocess_gpu.
        mark_d3d_parallax_start(perf_slot);
      }
      if (!dispatch_parallax_v2_producer(perf_slot, skip_input_history)) {
        mark_terminal_failure();
        return false;
      }
      if (subtitle_work_suppressed) {
        // Preserve subtitle history and expose the new Base without a duplicate texture write.
        completed_subtitle_conditioning_diagnostics = false;
        if (subtitle_conditioning_diagnostics_always && !ensure_subtitle_conditioning_diagnostics()) {
          BOOST_LOG(warning) << "Host SBS conditioned parallax diagnostic resources are unavailable.";
          mark_terminal_failure();
          return false;
        }
      } else if (!dispatch_subtitle_conditioner(ocr_record_submitted, input_domain_reset, perf_slot)) {
        mark_terminal_failure();
        return false;
      }
      if (parallax_v2_producer_active) {
        mark_d3d_parallax_end(perf_slot);
      }

      pending_gpu_trace_append = {
        .host_subtitle_outcome = trace_host_subtitle_outcome,
        .ocr_record_submitted =
          !subtitle_publication_branch_opaque && ocr_record_submitted,
        .subtitle_work_suppressed = subtitle_work_suppressed,
        // For opaque roots the trace shader derives execution from the device receipt.
        .condition_executed = !subtitle_publication_branch_opaque && !subtitle_work_suppressed,
        .subtitle_branch_gated =
          subtitle_publication_branch_opaque && !subtitle_work_suppressed,
        .input_domain_reset = input_domain_reset,
        .valid = true,
      };

      return true;
    }

    // Diagnostics-only accounting for achieved inference throughput and busy drops. Callers
    // bypass this function entirely when diagnostics are disabled, avoiding even a clock read.
    void update_throughput_stats() {
      const auto now = std::chrono::steady_clock::now();

      // A five-second window is responsive enough for headset tuning without flooding the log.
      if (throughput_stats_start.time_since_epoch().count() == 0) {
        throughput_stats_start = now;
      } else {
        float stats_seconds = std::chrono::duration<float>(now - throughput_stats_start).count();
        if (stats_seconds >= 5.0f) {
          float calls = (float) std::max(1u, throughput_stats_calls);
          // Admission attempts include retained-source retries. This is neither capture FPS
          // nor model invocation rate; the GPU outcome counters distinguish infer from reuse.
          BOOST_LOG(info) << "Depth throughput: admission attempts ~"
                          << (int) (throughput_stats_calls / stats_seconds + 0.5f)
                          << "/s, completed ~" << (int) (throughput_stats_completions / stats_seconds + 0.5f)
                          << "fps, enqueued ~" << (int) (throughput_stats_enqueues / stats_seconds + 0.5f)
                          << "fps, busy drops " << (int) (100.0f * throughput_stats_busy_drops / calls + 0.5f)
                          << "% (" << throughput_stats_busy_drops << '/' << throughput_stats_calls
                           << "), force-infer roots " << throughput_stats_force_infer_enqueues
                           << ", GPU-undecided roots " << throughput_stats_gpu_undecided_enqueues
                          << ", optional OCR armed " << throughput_stats_ocr_armed
                          << ", native subtitle suppressions " << throughput_stats_subtitle_suppressed;
          throughput_stats_start = now;
          throughput_stats_calls = 0;
          throughput_stats_busy_drops = 0;
          throughput_stats_enqueues = 0;
          throughput_stats_force_infer_enqueues = 0;
          throughput_stats_gpu_undecided_enqueues = 0;
          throughput_stats_completions = 0;
          throughput_stats_subtitle_suppressed = 0;
          throughput_stats_ocr_armed = 0;
        }
      }
      throughput_stats_calls++;
    }

    // Query-only producer preflight. It leaves the accepted root and the finished
    // output buffer untouched; estimate() consumes that result after the caller has copied the
    // exact color frame that will own the next inference.
    bool can_accept() {
      if (!is_operational()) {
        return false;
      }
      auto &cuda = cuda_driver_api::get();
      if (!cuda.is_valid() || !cu_stream || !cuda.cuStreamQuery) {
        if (!stream_error_logged) {
          BOOST_LOG(error)
            << "Depth readiness preflight lacks its CUDA query capability"
            << " (driver=" << (cuda.is_valid() ? "valid" : "invalid")
            << ", stream=" << (cu_stream ? "present" : "missing")
            << ", cuStreamQuery=" << (cuda.cuStreamQuery ? "present" : "missing") << ").";
          stream_error_logged = true;
        }
        mark_cuda_context_failure();
        return false;
      }
      if (cuda_ctx && cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
        if (!stream_error_logged) {
          BOOST_LOG(error) << "cuCtxSetCurrent failed during depth readiness preflight.";
          stream_error_logged = true;
        }
        mark_cuda_context_failure();
        return false;
      }

      if (diagnostics_enabled) {
        update_throughput_stats();
      }
      switch (query_pending_execution(cuda, "depth readiness preflight")) {
        case pending_execution_readiness_e::busy:
          if (diagnostics_enabled) {
            throughput_stats_busy_drops++;
          }
          readiness_preflighted = false;
          return false;
        case pending_execution_readiness_e::failed:
          readiness_preflighted = false;
          return false;
        case pending_execution_readiness_e::ready:
          break;
      }
      readiness_preflighted = true;
      return true;
    }

    depth_step_result_t estimate(
      ID3D11ShaderResourceView *input_srv,
      input_color_space color_space,
      std::uint64_t frame_id,
      bool snapshot_raw_model_depth,
      depth_input_region_t input_region,
      depth_optional_work_mode_e optional_work,
      const gpu_adaptive_reuse_request &adaptive_reuse_request,
      const std::uint64_t subtitle_context_generation,
      const subtitle_line_policy_e subtitle_line_policy
    ) {
      if (!is_operational() || !input_srv) {
        return {};
      }
      completed_depth_metadata_t completed;
      // Dump may override temporary move/size suppression, but never enables an ineligible
      // subtitle context. Suppressed dump sections remain explicitly unused in the package.
      const auto accepted_optional_work =
        snapshot_raw_model_depth && optional_work != depth_optional_work_mode_e::disable_subtitle ?
          depth_optional_work_mode_e::ordinary : optional_work;

      auto &cuda = cuda_driver_api::get();
      if (!cuda.is_valid()) {
        BOOST_LOG(error) << "CUDA Driver API is not available.";
        mark_cuda_context_failure();
        return {};
      }

      if (cuda_ctx && cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
        if (!stream_error_logged) {
          BOOST_LOG(error) << "cuCtxSetCurrent failed during depth estimation.";
          stream_error_logged = true;
        }
        mark_cuda_context_failure();
        return {};
      }

      // Production preflights before its expensive full-resolution color copy. The evaluator and
      // any direct callers do not, so retain the self-contained query/counting path here.
      const bool preflighted = std::exchange(readiness_preflighted, false);
      if (!preflighted && diagnostics_enabled) {
        update_throughput_stats();
      }

      // Resolve completed inference-timing events only for diagnostic runs.
      if (diagnostics_enabled) {
        perf_drain();
        if (is_terminal()) {
          return {};
        }
      }

      // Prevent GPU starvation: if the previous AI frame is still crunching, drop this frame.
      // This prevents an infinite queue of heavy TensorRT workloads from starving the DWM and Edge Browser.
      if (!preflighted) {
        switch (query_pending_execution(cuda, "depth estimate admission")) {
          case pending_execution_readiness_e::busy:
            // Reuse the last normalized depth and health state while the joined exact-frame root is
            // busy. This keeps the bounded one-observation contract without exposing its branch.
            if (diagnostics_enabled) {
              throughput_stats_busy_drops++;
            }
            return make_result();
          case pending_execution_readiness_e::failed:
            return make_result();
          case pending_execution_readiness_e::ready:
            break;
        }
      }

      // The fresh admission query or exact retained preflight proved the preceding full stream
      // complete. Retire its timing before any signature change, wrapper reset or new launch.
      if (diagnostics_enabled) {
        resolve_completed_branch_timing();
        if (is_terminal()) {
          return {};
        }
      }

      D3D11_TEXTURE2D_DESC input_desc = {0};
      Microsoft::WRL::ComPtr<ID3D11Resource> input_res;
      input_srv->GetResource(&input_res);
      Microsoft::WRL::ComPtr<ID3D11Texture2D> input_tex;
      if (SUCCEEDED(input_res.As(&input_tex))) {
        input_tex->GetDesc(&input_desc);
      }
      if (input_desc.Width == 0u || input_desc.Height == 0u) {
        return make_result();
      }
      input_region = resolved_runtime_input_region(input_region, input_desc);
      if (!input_region.valid()) {
        return make_result();
      }
      const auto requested_dav2_shape = models::fit_depth_tensor_shape(
        input_region.source_width,
        input_region.source_height,
        depth_short_side,
        max_aspect
      );
      const auto requested_fused_shape = host_sbs_convex2x_field_shape(
        requested_dav2_shape
      );
      if (!requested_fused_shape) {
        BOOST_LOG(error)
          << "Fused DAV2 + ZipDepth cannot derive its authenticated high grid from "
          << requested_dav2_shape.width << 'x' << requested_dav2_shape.height << '.';
        mark_terminal_failure();
        return {};
      }
      const depth_tensor_shape_t requested_shape = *requested_fused_shape;
      const bool current_shape_matches =
        target_w == 0 || target_h == 0 ||
        (target_w == requested_shape.width && target_h == requested_shape.height);

      if (target_w == 0 || target_h == 0) {
        // The capture surface can report a 0x0 descriptor mid HDR/mode transition or
        // before the first real frame. Deriving the model resolution from that yields a
        // garbage size (NaN aspect -> integer-overflow -> clamps to the profile max) that would
        // be cached for the whole session. Wait for a valid frame instead.
        if (input_desc.Width == 0 || input_desc.Height == 0) {
          return {};
        }
        const auto fused_profile = prod_zipdepth_convex2x::fixed_profile_index(
          static_cast<std::uint32_t>(requested_shape.width),
          static_cast<std::uint32_t>(requested_shape.height)
        );
        if (!fused_profile) {
          BOOST_LOG(error) << "Fused DAV2 + ZipDepth has no authenticated point profile for "
                           << requested_shape.width << 'x' << requested_shape.height << '.';
          mark_terminal_failure();
          return {};
        }

        // Keep the patch-aligned tensor as close as possible to source aspect while respecting
        // the TensorRT profile, configured aspect cap, and native size.
        target_w = requested_shape.width;
        target_h = requested_shape.height;

        // Threads for the shared moments/range reduction; grid-stride handles any element count.
        int elems = target_w * target_h;
        reduce_groups = (UINT) std::min(64, std::max(1, (elems + 255) / 256));

        // A failed CUDA registration resets target_w/target_h and retries shape setup on a later
        // frame. Never let optional textures/state from that abandoned shape survive into the
        // retry (whose source aspect may already have changed).
        release_parallax_v2_resources();
        field_w = target_w;
        field_h = target_h;
        if (!unregister_near_identical_decision_interop(cuda)) {
          mark_terminal_failure(true);
          return {};
        }
        near_identical_tile.reset();
        near_identical_transaction.reset();
        near_identical_history_owner.reset();

        BOOST_LOG(info) << "Depth Estimator dynamic resolution set to " << target_w << "x" << target_h;

        const bool previous_input_unregistered =
          unregister_cuda_graphics_resource(cuda, cuda_in_res);
        const bool previous_output_unregistered =
          unregister_cuda_graphics_resource(cuda, cuda_out_res);
        if (!previous_input_unregistered || !previous_output_unregistered) {
          // A retained registration still owns the old D3 backing. Shape setup must not release
          // or overwrite it; quarantine this context and let terminal teardown retry safely.
          BOOST_LOG(error)
            << "Depth shape change to " << target_w << 'x' << target_h
            << " could not unregister prior CUDA interop resources"
            << " (input=" << (previous_input_unregistered ? "released" : "failed")
            << ", output=" << (previous_output_unregistered ? "released" : "failed")
            << "); the execution context is quarantined and the stream fails flat.";
          execution_context_poisoned = true;
          mark_terminal_failure(true);
          return {};
        }

        D3D11_BUFFER_DESC buf_desc = {};
        buf_desc.Usage = D3D11_USAGE_DEFAULT;
        buf_desc.ByteWidth = target_w * target_h * 3 * sizeof(float);
        buf_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        buf_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        buf_desc.StructureByteStride = sizeof(float);
        bool resources_ok = SUCCEEDED(device->CreateBuffer(&buf_desc, nullptr, &tensor_in_buf)) &&
                            SUCCEEDED(device->CreateUnorderedAccessView(
                              tensor_in_buf.Get(),
                              nullptr,
                              &tensor_in_uav
                            )) &&
                            SUCCEEDED(device->CreateShaderResourceView(
                              tensor_in_buf.Get(),
                              nullptr,
                              &tensor_in_srv
                            ));

        auto previous_input_desc = buf_desc;
        previous_input_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateBuffer(&previous_input_desc, nullptr, &tensor_previous_input_buf)) &&
                       SUCCEEDED(device->CreateShaderResourceView(
                         tensor_previous_input_buf.Get(),
                         nullptr,
                         &tensor_previous_input_srv
                       )) &&
                       SUCCEEDED(device->CreateUnorderedAccessView(
                         tensor_previous_input_buf.Get(),
                         nullptr,
                         &tensor_previous_input_uav
                       ));

        const std::size_t output_logical_bytes =
          static_cast<std::size_t>(target_w) * target_h * sizeof(float);
        const std::optional<std::size_t> output_allocation_bytes =
          detail::checked_tensorrt_output_allocation_bytes(output_logical_bytes);
        resources_ok = resources_ok && output_allocation_bytes &&
          *output_allocation_bytes <= std::numeric_limits<UINT>::max();
        buf_desc.ByteWidth = output_allocation_bytes ?
                               static_cast<UINT>(*output_allocation_bytes) :
                               0u;
        buf_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SHADER_RESOURCE_VIEW_DESC output_srv_desc {};
        output_srv_desc.Format = DXGI_FORMAT_UNKNOWN;
        output_srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        output_srv_desc.Buffer.FirstElement = 0u;
        output_srv_desc.Buffer.NumElements =
          static_cast<UINT>(target_w * target_h);
        resources_ok = resources_ok &&
                       SUCCEEDED(device->CreateBuffer(&buf_desc, nullptr, &tensor_out_buf)) &&
                       SUCCEEDED(device->CreateShaderResourceView(
                         tensor_out_buf.Get(),
                         &output_srv_desc,
                         &tensor_out_srv
                       ));

        if (normalized_depth_diagnostics_enabled) {
          resources_ok = resources_ok &&
                         create_shader_texture(target_w, target_h, DXGI_FORMAT_R32_FLOAT, normalized_depth) &&
                         create_shader_texture(target_w, target_h, DXGI_FORMAT_R32_FLOAT, previous_normalized_depth) &&
                         create_shader_texture(target_w, target_h, DXGI_FORMAT_R32_UINT, ema_motion_mask);
        }

        constexpr auto domain_words =
          sizeof(host_sbs_v2_gpu::analysis_domain_record_t) / sizeof(std::uint32_t);
        resources_ok = resources_ok && create_shader_uint_buffer(
          domain_words, analysis_domain.buffer, analysis_domain.srv, analysis_domain.uav, domain_words
        );

        if (!resources_ok) {
          BOOST_LOG(error)
            << "Depth estimator D3D11 resource creation failed; Host SBS V2 will fail flat.";
          target_w = target_h = 0;
          field_w = field_h = 0;
          mark_terminal_failure();
          return {};
        }

        if (!ensure_parallax_v2_resources()) {
          mark_terminal_failure();
          return {};
        }

        // Initialize the comparison history before the first eligible fused preprocess.
        const UINT clear_uint[4] {};
        context->ClearUnorderedAccessViewUint(tensor_previous_input_uav.Get(), clear_uint);
        clear_normalized_depth_diagnostics();
        context->ClearUnorderedAccessViewUint(analysis_domain.uav.Get(), clear_uint);

        auto res1 = cuda.cuGraphicsD3D11RegisterResource(&cuda_in_res, tensor_in_buf.Get(), 0);
        auto res2 = cuda.cuGraphicsD3D11RegisterResource(&cuda_out_res, tensor_out_buf.Get(), 0);
        if (res1 != CUDA_SUCCESS || res2 != CUDA_SUCCESS) {
          BOOST_LOG(error) << "Atomic depth TensorRT interop registration failed: "
                           << res1 << ", " << res2;
          const bool input_cleanup_ok =
            unregister_cuda_graphics_resource(cuda, cuda_in_res);
          const bool output_cleanup_ok =
            unregister_cuda_graphics_resource(cuda, cuda_out_res);
          if (!input_cleanup_ok || !output_cleanup_ok) {
            execution_context_poisoned = true;
          }
          target_w = target_h = 0;
          field_w = field_h = 0;
          mark_terminal_failure(
            !input_cleanup_ok || !output_cleanup_ok
          );
          return {};
        }
        // Direct Glyph's score buffers register on this drawing thread as depth starts, like the
        // tensors above, so the first caption does not pay for it.
        if (subtitle_direct_glyph.enabled()) {
          subtitle_direct_glyph.register_interop_early();
        }
      }

      // Keep the pending transfer and crop constants through its publication and Dump copies.
      // With no preceding observation, prepare the first submission's constants directly.
      ensure_cbuffers(
        pending_root ? pending_root->color_space : color_space,
        pending_root ? pending_root->input_region : input_region
      );
      ensure_source_region_cbuffer(
        pending_root ? pending_root->input_region : input_region
      );
      if (!cbuffer || !source_region_cbuffer) {
        mark_terminal_failure();
        return {};
      }

      auto *d3d_timer = diagnostics_enabled ?
                          begin_d3d_perf(pending_root.has_value(), true) :
                          nullptr;

      // tensor_out_buf holds the finished raw disparity from the previous asynchronous submit
      // (fully unmapped from CUDA), so consuming it here never blocks the encode thread. The
      // caller uses completed_frame_id to select the color slot that produced this exact result.
      if (pending_root) {
        completed = complete_ready_pending(d3d_timer, snapshot_raw_model_depth);
        if (!completed.valid) {
          mark_d3d_pre_start(d3d_timer);
          end_d3d_perf(d3d_timer);
          return {};
        }
      } else {
        mark_d3d_post_end(d3d_timer);
      }

      const subtitle_context_t subtitle_context {
        subtitle_context_generation,
        accepted_optional_work == depth_optional_work_mode_e::disable_subtitle,
      };
      const bool subtitle_context_changed = subtitle_context != completed_input.subtitle_context;
      if (completed.valid && subtitle_context_changed) {
        // This call must hand the preceding publication to its caller before clearing any of
        // the singleton subtitle/final resources it owns. Admit the changed context on a later
        // call; neither its epoch nor its optional work has been accepted yet.
        mark_d3d_pre_start(d3d_timer);
        end_d3d_perf(d3d_timer);
        return make_result(completed);
      }

      // The first ROI trial deliberately keeps one session tensor shape. A validated crop whose
      // fitted shape differs is not allowed to poison the authenticated producer; consume any old
      // completion above, skip this enqueue, and let the caller use its full-frame fallback.
      if (!current_shape_matches) {
        if (!shape_mismatch_warning_logged) {
          BOOST_LOG(warning)
            << "Depth input " << input_region.source_width << 'x' << input_region.source_height
            << " maps to " << requested_shape.width << 'x' << requested_shape.height
            << " but this estimator session owns " << target_w << 'x' << target_h
            << "; skipping this ROI depth submission and using full-frame fallback."
            << " Further shape-mismatch warnings are suppressed for this estimator.";
          shape_mismatch_warning_logged = true;
        }
        mark_d3d_pre_start(d3d_timer);
        end_d3d_perf(d3d_timer);
        return make_result(completed);
      }

      if (completed.valid) {
        ensure_cbuffers(color_space, input_region);
        ensure_source_region_cbuffer(input_region);
        if (!cbuffer || !source_region_cbuffer) {
          mark_terminal_failure();
          mark_d3d_pre_start(d3d_timer);
          end_d3d_perf(d3d_timer);
          return {};
        }
      }

      if (!ocr_available) {
        ocr_signature_refresh_required = false;
      }

      // Every DAV2 transaction, including a CPU-known force-infer, maps the same authenticated
      // request/receipt buffer and launches the conditional wrapper. These are mandatory pipeline
      // resources: losing one is terminal rather than a reason to bypass the wrapper.
      if (!ensure_near_identical_detector_resources() ||
          !ensure_near_identical_decision_interop(cuda) ||
          !near_identical_fused_preprocess_cs ||
          !fused_preprocess_force_cbuffer ||
          !fused_preprocess_compare_cbuffer ||
          !tensor_previous_input_srv || !near_identical_tile.uav) {
        if (!is_terminal()) {
          fail_gpu_conditional_bridge_once(
            "mandatory GPU transaction resources or interop are unavailable"
          );
        }
        return {};
      }

      // Decide whether this transaction has a complete optional OCR path using host-owned
      // resource/geometry state only. GPU-undecided must publish its RQST before the fused
      // preprocess can generate image-derived tile evidence.
      const auto current_field_content = live_field_content(input_region);
      const auto current_ocr_roi = fit_subtitle_analysis_geometry(
        input_region.width(),
        input_region.height(),
        live_field_shape(),
        current_field_content.value_or(depth_tensor_content_rect_t {})
      );
      // Hard exclusion never consumes native subtitle evidence. Keep only initialized
      // placeholders, including when retiring a formerly eligible context. Temporary
      // suppression instead keeps that context's backing untouched until ordinary work
      // resumes; it must not resize or discard frozen subtitle memory.
      const bool contour_resources_ready =
        accepted_optional_work == depth_optional_work_mode_e::suppress_subtitle ||
        (subtitle_context.disabled ?
           ensure_subtitle_contour_placeholders() :
           ensure_subtitle_contour_textures(input_region.width(), input_region.height()));
      if (!contour_resources_ready) {
        mark_terminal_failure();
        mark_d3d_pre_start(d3d_timer);
        end_d3d_perf(d3d_timer);
        return {};
      }
      if (subtitle_context_changed) {
        // Busy/invalid/shape-rejected calls and calls returning a preceding completion have
        // returned before this point. Queue only subtitle clears before this source's
        // preprocessing; remember the epoch only after successful admission.
        if (!reset_subtitle_state()) {
          mark_d3d_pre_start(d3d_timer);
          end_d3d_perf(d3d_timer);
          return {};
        }
      }
      const std::uint32_t ocr_crop_height = subtitle_ocr_source_crop_height(
        input_region.width(),
        input_region.height()
      );
      const auto current_ocr_line_reference = subtitle_ocr_line_reference(
        input_region.width(), input_region.height(),
        input_region.source_width, input_region.source_height,
        input_region.left, input_region.top, input_region.is_video_region()
      );
      const bool ocr_interop_available =
        ocr_available && ocr_exec_context && ocr_input_uav && ocr_output_srv &&
        cuda_ocr_in_res && cuda_ocr_out_res;
      bool ocr_frame_eligible =
        accepted_optional_work == depth_optional_work_mode_e::ordinary &&
        ocr_interop_available && ocr_preprocess_cs && ocr_preprocess_cbuffer &&
        ocr_box_cells_cs && ocr_box_resolve_cs &&
        near_identical_finalize_cs && ocr_cell_stats_srv &&
        ocr_cell_stats_uav && ocr_box_record_uav && ocr_local_candidate_scratch_uav &&
        ocr_resolve_cbuffer &&
        current_field_content && current_ocr_roi.valid() && current_ocr_line_reference &&
        ocr_crop_height != 0u &&
        ocr_crop_height <= input_region.height();
      const auto subtitle_work = subtitle_transaction_work(
        accepted_optional_work,
        ocr_frame_eligible
      );
      const bool depth_only_history = adaptive_reuse_request.depth_only_history &&
        accepted_optional_work == depth_optional_work_mode_e::disable_subtitle && !snapshot_raw_model_depth;
      // The CPU may only classify a frame as GPU-undecided from host-owned facts. The wrapper is
      // prepared after the exact D3D interop pointers are mapped; no pre-existing source graph is
      // an admission prerequisite and no admitted class may bypass the wrapper.
      const bool gpu_undecided_host_candidate =
        gpu_conditional_bridge_available &&
        cuda.has_conditional_graph_support() &&
        !depth_inference_graph.policy.capture_failed &&
        adaptive_reuse_request.authorize_gpu_undecided_reuse &&
        adaptive_reuse_request.observation_timestamp_us != 0u &&
        adaptive_reuse_request.gpu_reuse_decision_token != 0u &&
        adaptive_reuse_request.baseline_frame_id != 0u &&
        gpu_undecided_baseline_authorized(adaptive_reuse_request) &&
        frame_id > adaptive_reuse_request.baseline_frame_id &&
        !gpu_undecided_requires_force_infer_refresh &&
        (!ocr_signature_refresh_required || depth_only_history) &&
        !subtitle_context_changed &&
        // Every new OCR-eligible source observation runs the joined depth/subtitle path.
        // Depth and subtitle work complete as one tuple, so similarity reuse would also hold
        // back OCR and the Direct Glyph mask for a caption that appears on a similar frame.
        !ocr_frame_eligible &&
        depth_optional_work_allows_gpu_undecided(accepted_optional_work, adaptive_reuse_request.depth_only_history) &&
        !snapshot_raw_model_depth && current_shape_matches &&
        processed_input_domain.matches_analysis_domain(input_region, color_space) &&
        input_region.tensor_content.valid({target_w, target_h});
      auto accepted_submission_class =
        gpu_undecided_host_candidate ?
          submission_class_e::gpu_undecided :
          submission_class_e::force_infer;
      const std::uint64_t accepted_transaction_token =
        gpu_undecided_host_candidate ?
          adaptive_reuse_request.gpu_reuse_decision_token :
          next_force_infer_transaction_token();

      if (gpu_undecided_host_candidate && !prepare_near_identical_detector(
            adaptive_reuse_request,
            frame_id,
            input_region,
            color_space,
            subtitle_work
          )) {
        fail_gpu_conditional_bridge_once(
          "GPU near-identical request preparation failed"
        );
        mark_d3d_pre_start(d3d_timer);
        end_d3d_perf(d3d_timer);
        return {};
      }

      mark_d3d_pre_start(d3d_timer);
      ID3D11ShaderResourceView *analysis_input_srv = input_srv;

      // The direct program is faster for small footprints after the area loop was unified.
      // Use the shared tile for stronger downsampling when both axes fit its 48-cell extent.
      const auto source_tile_axis_fits = [](std::uint32_t source, std::uint32_t content) {
        return content != 0u && source > 2ull * content && 16ull * source <= 46ull * content;
      };
      const bool use_source_tile = near_identical_tiled_preprocess_cs &&
        source_tile_axis_fits(input_region.width(), input_region.tensor_content.width()) &&
        source_tile_axis_fits(input_region.height(), input_region.tensor_content.height());
      auto *preprocess_shader = use_source_tile ?
        near_identical_tiled_preprocess_cs.Get() : near_identical_fused_preprocess_cs.Get();
      context->CSSetShader(preprocess_shader, nullptr, 0u);
      // Force/bootstrap/debug exits after canonical preprocessing. GPU-undecided also writes
      // comparison evidence in this dispatch.
      ID3D11Buffer *preprocess_cbuffers[3] = {
        cbuffer.Get(),
        gpu_undecided_host_candidate ?
          fused_preprocess_compare_cbuffer.Get() :
          fused_preprocess_force_cbuffer.Get(),
        source_region_cbuffer.Get(),
      };
      context->CSSetConstantBuffers(0u, 3u, preprocess_cbuffers);
      ID3D11ShaderResourceView *preprocess_srvs[2] = {
        analysis_input_srv,
        tensor_previous_input_srv.Get(),
      };
      context->CSSetShaderResources(0, 2, preprocess_srvs);
      ID3D11UnorderedAccessView *preprocess_uavs[3] = {
        tensor_in_uav.Get(),
        analysis_domain.uav.Get(),
        near_identical_tile.uav.Get(),
      };
      context->CSSetUnorderedAccessViews(0, 3, preprocess_uavs, nullptr);
      context->Dispatch((target_w + 15) / 16, (target_h + 15) / 16, 1);

      ID3D11UnorderedAccessView *null_uavs[3] = {};
      ID3D11ShaderResourceView *null_srvs[2] = {};
      ID3D11Buffer *null_preprocess_cbuffers[3] = {};
      context->CSSetUnorderedAccessViews(0, 3, null_uavs, nullptr);
      context->CSSetShaderResources(0, 2, null_srvs);
      context->CSSetConstantBuffers(0u, 3u, null_preprocess_cbuffers);
      // GPU-undecided publishes its dense proposal. CPU-known force-infer publishes a complete
      // authenticated infer PROP/RQST instead. Both records carry the exact subtitle disposition;
      // only prepared OCR carries the optional-child request, and every completion consumes CBRG.
      const bool transaction_published =
        accepted_submission_class == submission_class_e::gpu_undecided ?
          dispatch_near_identical_detector() :
          publish_force_infer_transaction(
            accepted_transaction_token,
            subtitle_work
          );
      if (!transaction_published) {
        fail_gpu_conditional_bridge_once(
          accepted_submission_class == submission_class_e::gpu_undecided ?
            "GPU near-identical proposal dispatch failed" :
            "force-infer transaction publication failed"
        );
        mark_d3d_pre_start(d3d_timer);
        end_d3d_perf(d3d_timer);
        return {};
      }

      if (ocr_frame_eligible) {
        // The pre-CUDA crop consumes the proposal's indirect arguments. The finalizer publishes
        // the postprocess arguments separately after CUDA releases the raw transaction.
        context->CopyResource(
          near_identical_transaction.dispatch.Get(), near_identical_transaction.buffer.Get()
        );
        const std::array<std::uint32_t, 8> ocr_constants {
          input_region.width(),
          input_region.height(),
          input_region.height() - ocr_crop_height,
          ocr_crop_height,
          static_cast<std::uint32_t>(color_space),
          input_region.left,
          input_region.top,
          0u,
        };
        context->UpdateSubresource(
          ocr_preprocess_cbuffer.Get(), 0, nullptr, ocr_constants.data(), 0,
          0
        );
        context->CSSetShader(ocr_preprocess_cs.Get(), nullptr, 0);
        context->CSSetConstantBuffers(0, 1, ocr_preprocess_cbuffer.GetAddressOf());
        context->CSSetShaderResources(0, 1, &analysis_input_srv);
        ID3D11UnorderedAccessView *ocr_preprocess_outputs[2] {
          ocr_input_uav.Get(), subtitle_contour_source.uav.Get()
        };
        context->CSSetUnorderedAccessViews(0, 2, ocr_preprocess_outputs, nullptr);
        context->DispatchIndirect(
          near_identical_transaction.dispatch.Get(),
          near_identical_gpu_optional_preprocess_byte_offset
        );
        context->CSSetUnorderedAccessViews(0, 2, null_uavs, nullptr);
        context->CSSetShaderResources(0, 1, null_srvs);
        ID3D11Buffer *null_cbuffer = nullptr;
        context->CSSetConstantBuffers(0, 1, &null_cbuffer);
      }

      end_d3d_perf(d3d_timer);
      // No explicit Flush: cuGraphicsMapResources() below already guarantees the
      // preceding D3D11 compute work completes before the CUDA stream reads the buffer.
      // Force-flushing every frame only prevents the driver from interleaving other GPU
      // consumers (DWM / Edge / the Widgets panel), which starves them and can trigger a TDR.

      // 2. CUDA Execution (for CURRENT frame). Both conditional children share one root stream
      // and completion event, so transfer their D3D ownership in one batch. Private unmap
      // experiments retain their original split calls, including warmup and separate receipts.
      std::array<CUgraphicsResource, 5> interop_resources {
        cuda_in_res,
        cuda_out_res,
        cuda_near_identical_decision_res,
        cuda_ocr_in_res,
        cuda_ocr_out_res,
      };
      constexpr unsigned int depth_resource_count = 3u;
      constexpr unsigned int ocr_resource_count = 2u;
      const unsigned int interop_resource_count =
        depth_resource_count + (ocr_frame_eligible ? ocr_resource_count : 0u);
      const auto map_res = cuda.cuGraphicsMapResources(
        interop_resource_count, interop_resources.data(), cu_stream
      );
      if (map_res != CUDA_SUCCESS) {
        if (ocr_frame_eligible) {
          ocr_available = false;
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::pre_enqueue_interop_or_binding
          );
        }
        // A failed multi-resource call does not prove that none of its resources mapped.
        // Keep the terminal teardown/quarantine policy; never retry individual resources.
        (void) observe_joined_cuda_failure(
          ocr_frame_eligible ?
            "joined depth/OCR interop map" : "depth interop map",
          map_res, true
        );
        return make_result(completed);
      }
      const bool ocr_mapped = ocr_frame_eligible;
      if (is_terminal()) {
        ocr_frame_eligible = false;
      }

      CUdeviceptr d_in = 0;
      CUdeviceptr d_out = 0;
      std::size_t input_mapped_bytes = 0u;
      std::size_t output_mapped_bytes = 0u;
      auto in_ptr_res = cuda.cuGraphicsResourceGetMappedPointer(
        &d_in,
        &input_mapped_bytes,
        cuda_in_res
      );
      auto out_ptr_res = cuda.cuGraphicsResourceGetMappedPointer(
        &d_out,
        &output_mapped_bytes,
        cuda_out_res
      );
      CUdeviceptr d_near_identical_decision = 0u;
      std::size_t near_identical_decision_mapped_bytes = 0u;
      const CUresult near_identical_decision_ptr_res =
        cuda.cuGraphicsResourceGetMappedPointer(
          &d_near_identical_decision,
          &near_identical_decision_mapped_bytes,
          cuda_near_identical_decision_res
        );

      CUdeviceptr d_ocr_in = 0;
      CUdeviceptr d_ocr_out = 0;
      std::size_t ocr_input_mapped_bytes = 0u;
      std::size_t ocr_output_mapped_bytes = 0u;
      CUresult ocr_in_ptr_res = CUDA_ERROR_NOT_READY;
      CUresult ocr_out_ptr_res = CUDA_ERROR_NOT_READY;
      if (ocr_mapped) {
        ocr_in_ptr_res = cuda.cuGraphicsResourceGetMappedPointer(
          &d_ocr_in, &ocr_input_mapped_bytes,
          cuda_ocr_in_res
        );
        ocr_out_ptr_res = cuda.cuGraphicsResourceGetMappedPointer(
          &d_ocr_out, &ocr_output_mapped_bytes,
          cuda_ocr_out_res
        );
      }

      const bool ocr_pointer_contract_valid =
        !ocr_mapped ||
        (ocr_in_ptr_res == CUDA_SUCCESS && ocr_out_ptr_res == CUDA_SUCCESS &&
         d_ocr_in != 0u && d_ocr_out != 0u);
      if (ocr_mapped && !ocr_pointer_contract_valid) {
        ocr_available = false;
        mark_ocr_context_failure(
          detail::warmed_execution_context_failure_e::pre_enqueue_interop_or_binding
        );
        const CUresult pointer_failure =
          ocr_in_ptr_res != CUDA_SUCCESS ? ocr_in_ptr_res :
          ocr_out_ptr_res != CUDA_SUCCESS ? ocr_out_ptr_res :
          static_cast<CUresult>(-1);
        (void) observe_joined_cuda_failure(
          "optional OCR mapped-pointer contract", pointer_failure, true
        );
      }

      // Admission proved the preceding observation's full stream reusable. From here on these
      // flags describe only the candidate whose interop unmaps will be issued below.
      clear_pending_inference_event_state();
      bool enqueued = false;
      // This is request authority, not merely wrapper topology: a retained superset wrapper may
      // contain the OCR sibling while suppression leaves that child dormant.
      bool ocr_armed = false;
      if (in_ptr_res != CUDA_SUCCESS || out_ptr_res != CUDA_SUCCESS || !d_in || !d_out) {
        const CUresult pointer_failure =
          in_ptr_res != CUDA_SUCCESS ? in_ptr_res :
          out_ptr_res != CUDA_SUCCESS ? out_ptr_res :
          static_cast<CUresult>(-1);
        (void) observe_joined_cuda_failure(
          "depth mapped-pointer contract", pointer_failure, true
        );
      } else if (!is_terminal()) {
        constexpr std::size_t transaction_last_byte =
          near_identical_gpu_request_record_byte_offset +
          sizeof(cuda_conditional_graph::request_record_t) - 1u;
        const bool mapped_decision_valid =
          near_identical_decision_ptr_res == CUDA_SUCCESS &&
          d_near_identical_decision != 0u &&
          near_identical_decision_mapped_bytes >=
            near_identical_gpu_decision_byte_count &&
          d_near_identical_decision <=
            std::numeric_limits<CUdeviceptr>::max() -
              transaction_last_byte &&
          ((d_near_identical_decision +
            near_identical_gpu_decision_record_byte_offset) & 15u) == 0u &&
          ((d_near_identical_decision +
            near_identical_gpu_request_record_byte_offset) & 15u) == 0u;
        if (!mapped_decision_valid) {
          if (near_identical_decision_ptr_res != CUDA_SUCCESS) {
            (void) observe_joined_cuda_failure(
              "conditional transaction mapped-pointer query",
              near_identical_decision_ptr_res,
              true
            );
          }
          fail_gpu_conditional_bridge_once(
            "mapped decision-buffer contract is invalid",
            near_identical_decision_ptr_res,
            false,
            near_identical_decision_ptr_res != CUDA_SUCCESS
          );
        }

        // Graph signatures include D3D interop addresses, which CUDA permits to change on any
        // map. A force-infer transaction may privately recapture that legal new signature. A
        // GPU-undecided transaction cannot bootstrap raw TensorRT work, so retain the old wrapper,
        // drop this candidate after unmap, and require the next transaction to be force-infer.
        const auto selected_profile = prod_zipdepth_convex2x::fixed_profile_index(
          static_cast<std::uint32_t>(target_w),
          static_cast<std::uint32_t>(target_h)
        );
        const std::size_t expected_input_bytes =
          static_cast<std::size_t>(3u) * target_w * target_h * sizeof(float);
        const std::size_t expected_output_bytes =
          static_cast<std::size_t>(target_w) * target_h * sizeof(float);
        const auto expected_output_allocation_bytes =
          detail::checked_tensorrt_output_allocation_bytes(expected_output_bytes);
        const bool mapped_tensor_sizes_valid =
          selected_profile && expected_output_allocation_bytes &&
          input_mapped_bytes >= expected_input_bytes &&
          output_mapped_bytes >= *expected_output_allocation_bytes;
        if (!mapped_tensor_sizes_valid) {
          BOOST_LOG(error)
            << "Fused DAV2 + ZipDepth interop buffers violate the exact input/output allocation "
               "boundary: high "
            << input_mapped_bytes << '/' << expected_input_bytes << " in, "
            << output_mapped_bytes << '/'
            << expected_output_allocation_bytes.value_or(0u)
            << " out (" << expected_output_bytes << " logical).";
          mark_terminal_failure();
        }
        const detail::cuda_graph_signature_t depth_signature {
          d_in,
          d_out,
          target_w,
          target_h,
          selected_profile.value_or(0u),
        };
        bool dropped_for_signature_change =
          accepted_submission_class == submission_class_e::gpu_undecided &&
          !detail::cuda_graph_signature_matches(
            depth_inference_graph.policy, depth_signature
          );
        if (dropped_for_signature_change) {
          gpu_undecided_requires_force_infer_refresh = true;
          BOOST_LOG(info)
            << "DAV2 interop address changed during a GPU-undecided candidate; "
               "dropping it and rebuilding on the next force-infer transaction.";
        }
        if (!is_terminal() && !dropped_for_signature_change &&
            !select_inference_graph_signature(
              cuda,
              depth_inference_graph,
              depth_signature
            )) {
          mark_terminal_failure(true);
        }
        bool bindings_ok = !is_terminal() && !dropped_for_signature_change;
        bool profile_selection_attempted = false;
        if (bindings_ok &&
            trt_bound_profile != static_cast<int>(*selected_profile)) {
          profile_selection_attempted = true;
          bindings_ok = exec_context->setOptimizationProfileAsync(
            static_cast<int>(*selected_profile), cu_stream
          );
          if (bindings_ok) {
            trt_bound_profile = static_cast<int>(*selected_profile);
            trt_bound_width = 0;
            trt_bound_height = 0;
            trt_bound_input = 0;
            trt_bound_output = 0;
            trt_output_sizes_validated = false;
          }
        }
        if (trt_bound_width != target_w || trt_bound_height != target_h) {
          nvinfer1::Dims in_dims = make_input_dims(target_h, target_w);
          bindings_ok = bindings_ok && exec_context->setInputShape(
            prod_zipdepth_convex2x::input.data(), in_dims
          );
          if (bindings_ok) {
            trt_bound_width = target_w;
            trt_bound_height = target_h;
            trt_output_sizes_validated = false;
          } else {
            BOOST_LOG(error) << "TensorRT setInputShape failed for " << target_w << "x" << target_h
                             << " (outside the engine's optimization profile?)";
          }
        }
        if (bindings_ok && !trt_output_sizes_validated) {
          const std::int64_t required = exec_context->getMaxOutputSize(
            prod_zipdepth_convex2x::refined_output.data()
          );
          bindings_ok = expected_output_allocation_bytes && required > 0 &&
            static_cast<std::uint64_t>(required) <=
              *expected_output_allocation_bytes;
          trt_output_sizes_validated = bindings_ok;
          if (!bindings_ok) {
            BOOST_LOG(error)
              << "Fused DAV2 + ZipDepth resolved output bounds exceed their registered buffers: "
              << required << '/' << expected_output_allocation_bytes.value_or(0u)
              << " high.";
          }
        }
        if (bindings_ok && trt_bound_input != d_in) {
          bindings_ok = exec_context->setTensorAddress(
            prod_zipdepth_convex2x::input.data(),
            reinterpret_cast<void *>(d_in)
          );
          if (bindings_ok) {
            trt_bound_input = d_in;
          }
        }
        if (bindings_ok && trt_bound_output != d_out) {
          bindings_ok = exec_context->setTensorAddress(
            prod_zipdepth_convex2x::refined_output.data(),
            reinterpret_cast<void *>(d_out)
          );
          if (bindings_ok) {
            trt_bound_output = d_out;
          }
        }
        if (!bindings_ok && !dropped_for_signature_change) {
          mark_terminal_failure(profile_selection_attempted);
          if (!stream_error_logged) {
            BOOST_LOG(error)
              << "TensorRT shape/tensor binding failed; Host SBS V2 will fail flat.";
            stream_error_logged = true;
          }
        }
        bool ocr_bindings_ok = !is_terminal() && ocr_mapped &&
                               ocr_pointer_contract_valid && ocr_exec_context;
        constexpr std::size_t expected_ocr_input_bytes =
          static_cast<std::size_t>(3u) * ocr_engine_width * ocr_engine_height *
          sizeof(float);
        constexpr std::size_t expected_ocr_output_bytes =
          static_cast<std::size_t>(ocr_engine_width) * ocr_engine_height *
          sizeof(float);
        if (ocr_bindings_ok &&
            (ocr_input_mapped_bytes < expected_ocr_input_bytes ||
             ocr_output_mapped_bytes < expected_ocr_output_bytes)) {
          BOOST_LOG(error)
            << "PP-OCRv6 tiny interop buffer is smaller than its authenticated FP32 boundary: "
            << "input " << ocr_input_mapped_bytes << '/' << expected_ocr_input_bytes
            << ", output " << ocr_output_mapped_bytes << '/' << expected_ocr_output_bytes
            << " bytes.";
          ocr_bindings_ok = false;
        }
        const detail::cuda_graph_signature_t ocr_signature {
          d_ocr_in, d_ocr_out, ocr_engine_width, ocr_engine_height
        };
        if (
          ocr_bindings_ok &&
          accepted_submission_class == submission_class_e::gpu_undecided &&
          !detail::cuda_graph_signature_matches(
            ocr_inference_graph.policy, ocr_signature
          )
        ) {
          dropped_for_signature_change = true;
          ocr_signature_refresh_required = true;
          ocr_bindings_ok = false;
          BOOST_LOG(info)
            << "PP-OCR interop address changed during a GPU-undecided candidate; "
               "dropping it and rebuilding on the next force-infer transaction.";
        }
        if (ocr_bindings_ok && !dropped_for_signature_change) {
          // TensorRT graph nodes retain the execution context's bound addresses. Tear down a
          // graph for the previous D3D mapping before mutating those bindings.
          ocr_bindings_ok = select_inference_graph_signature(
            cuda,
            ocr_inference_graph,
            ocr_signature
          );
        }
        if (ocr_bindings_ok && !ocr_shape_bound) {
          ocr_bindings_ok = ocr_exec_context->setInputShape(
            "x", make_input_dims(ocr_engine_height, ocr_engine_width)
          );
          ocr_shape_bound = ocr_bindings_ok;
        }
        if (ocr_bindings_ok && !ocr_output_size_validated) {
          const std::int64_t required_output_bytes =
            ocr_exec_context->getMaxOutputSize("fetch_name_0");
          ocr_bindings_ok = required_output_bytes > 0 &&
                            static_cast<std::uint64_t>(required_output_bytes) <=
                              expected_ocr_output_bytes;
          ocr_output_size_validated = ocr_bindings_ok;
          if (!ocr_bindings_ok) {
            BOOST_LOG(error)
              << "PP-OCRv6 tiny resolved output requires " << required_output_bytes
              << " bytes; the authenticated interop buffer provides "
              << expected_ocr_output_bytes << '.';
          }
        }
        if (ocr_bindings_ok && ocr_bound_input != d_ocr_in) {
          ocr_bindings_ok = ocr_exec_context->setTensorAddress(
            "x", reinterpret_cast<void *>(d_ocr_in)
          );
          if (ocr_bindings_ok) {
            ocr_bound_input = d_ocr_in;
          }
        }
        if (ocr_bindings_ok && ocr_bound_output != d_ocr_out) {
          ocr_bindings_ok = ocr_exec_context->setTensorAddress(
            "fetch_name_0", reinterpret_cast<void *>(d_ocr_out)
          );
          if (ocr_bindings_ok) {
            ocr_bound_output = d_ocr_out;
          }
        }
        if (ocr_mapped && !ocr_bindings_ok && !dropped_for_signature_change &&
            !is_terminal()) {
          ocr_available = false;
          mark_ocr_context_failure(
            detail::warmed_execution_context_failure_e::pre_enqueue_interop_or_binding
          );
          mark_terminal_failure();
          if (!ocr_error_logged) {
            BOOST_LOG(error)
              << "PP-OCRv6 tiny fixed-shape interop/tensor binding failed; the exact-frame "
                 "DAV2/OCR unit fails flat.";
            ocr_error_logged = true;
          }
        }

        if (bindings_ok && !is_terminal() && !dropped_for_signature_change) {
          // TensorRT host-side binding/capture remains serialized across estimator instances. The
          // resulting depth and optional OCR children are siblings under one authenticated root;
          // no raw OCR enqueue or CPU-visible branch result exists.
          {
            std::lock_guard<std::mutex> lock(g_trt_mutex);
            const bool allow_private_bootstrap =
              accepted_submission_class == submission_class_e::force_infer;
            bool wrapper_ready = prepare_depth_inference_graph(
              cuda, allow_private_bootstrap
            );
            // Once captured, retain the OCR child in a superset wrapper even on USER32-suppressed
            // transactions. The authenticated request flag keeps its handle at
            // zero when OCR is not armed, so rebuilding the entire DAV2 root on every optional
            // mode transition is unnecessary. This is the same lifetime the wrapper already has
            // between ordinary frames: the child keeps its captured addresses while interop is
            // unmapped and can execute only after a later map validates that signature.
            CUgraph buildable_optional_child = nullptr;
            auto optional_child_policy =
              ocr_available && ocr_inference_graph.graph &&
                  !ocr_inference_graph.policy.capture_failed ?
                conditional_optional_child_policy_e::retain_if_present :
                conditional_optional_child_policy_e::disabled;
            bool optional_child_ready = false;
            if (wrapper_ready && ocr_bindings_ok) {
              wrapper_ready = prepare_ocr_inference_graph(
                cuda, allow_private_bootstrap
              );
              if (wrapper_ready && ocr_available) {
                buildable_optional_child = ocr_inference_graph.graph;
                optional_child_ready = buildable_optional_child != nullptr;
                optional_child_policy = optional_child_ready ?
                  conditional_optional_child_policy_e::build_ready :
                  conditional_optional_child_policy_e::disabled;
              }
            }
            if (wrapper_ready) {
              wrapper_ready = ensure_depth_conditional_graph(
                cuda,
                d_near_identical_decision +
                  near_identical_gpu_decision_record_byte_offset,
                d_near_identical_decision +
                  near_identical_gpu_request_record_byte_offset,
                buildable_optional_child,
                optional_child_policy
              );
            }
            const int depth_perf_slot = wrapper_ready ?
                                          perf_begin(cu_stream) :
                                          -1;
            if (wrapper_ready && !is_terminal()) {
              const bool launch_ocr_may_participate =
                optional_child_ready &&
                subtitle_work == cuda_conditional_graph::work_flag_e::optional_ocr;
              // cuGraphLaunch failure does not prove that neither child was partially submitted.
              // Establish attempt lineage before the call so a suppressed transaction also retains
              // OCR ownership from any earlier root in this shared stream lifetime.
              joined_stream_work_ever_submitted = true;
              joined_stream_ocr_ever_submitted_or_armed =
                joined_stream_ocr_ever_submitted_or_armed || launch_ocr_may_participate;
              const auto branch_timing_generation = diagnostics_enabled ?
                                                      sbs_perf::generation() : 0u;
              const CUresult launched = cuda.cuGraphLaunch(
                depth_conditional_graph.get(), cu_stream
              );
              enqueued = launched == CUDA_SUCCESS;
              if (enqueued && diagnostics_enabled &&
                  depth_conditional_graph.has_branch_completion_timing()) {
                if (!pending_branch_timing.arm(accepted_transaction_token, branch_timing_generation)) {
                  // Never overwrite an unresolved launch's event identity. Geometry authority is
                  // unchanged; this sample is unavailable rather than attributed to another frame.
                  pending_branch_timing.clear();
                }
              }
              if (!enqueued) {
                fail_gpu_conditional_bridge_once(
                  "conditional graph launch failed",
                  launched,
                  true,
                  true
                );
              }
              ocr_armed = enqueued && launch_ocr_may_participate;
            }
            perf_end(depth_perf_slot, cu_stream);
          }
          if (!enqueued && !is_terminal()) {
            mark_terminal_failure(true);
            if (!stream_error_logged) {
              BOOST_LOG(error) << "DAV2 conditional-wrapper submission failed; the execution "
                                   "context is quarantined and this estimator cannot continue.";
              stream_error_logged = true;
            }
          }
        }
      }

      const auto unmap_res = cuda.cuGraphicsUnmapResources(
        interop_resource_count, interop_resources.data(), cu_stream
      );
      if (unmap_res != CUDA_SUCCESS) {
        (void) observe_joined_cuda_failure(
          ocr_mapped ? "joined depth/OCR interop unmap" : "depth interop unmap",
          unmap_res, true
        );
        enqueued = false;
        ocr_armed = false;
      }

#ifdef SUNSHINE_TESTS
      // Native pacing test gate: actual CUDA ordering after the unmap tail, before the actual
      // root completion event. Does not change source, observation or Base/SCT clocks.
      if (enqueued && !is_terminal() && completion_test_gate &&
          completion_test_gate->targets(frame_id)) {
        const auto gate_result = inference_event_poll_available ?
          completion_test_gate->enqueue_after_interop_unmaps(cuda, cu_stream, frame_id) :
          CUDA_ERROR_NOT_SUPPORTED;
        if (gate_result != CUDA_SUCCESS) {
          completion_test_gate->cancel_and_release();
          (void) observe_joined_cuda_failure("private benchmark completion gate", gate_result, true);
          enqueued = false;
          ocr_armed = false;
        }
      }
#endif

      // Record completion after every interop unmap tail. Event polling and the stream-query
      // fallback now prove the same joined depth/OCR transaction and all of its resource release.
      if (
        enqueued && !is_terminal() && inference_event_poll_available &&
        !record_inference_done_event(
          cuda,
          depth_inference_done_event,
          cu_stream,
          pending_depth_inference_event_recorded,
          "DAV2/OCR root",
          ocr_armed
        )
      ) {
        enqueued = false;
        ocr_armed = false;
      }

      if (is_terminal()) {
        enqueued = false;
        ocr_armed = false;
      }

      if (ocr_signature_refresh_satisfied(
            ocr_signature_refresh_required,
            ocr_available,
            enqueued &&
              accepted_submission_class == submission_class_e::force_infer,
            accepted_optional_work,
            ocr_armed
          )) {
        ocr_signature_refresh_required = false;
      }

      if (enqueued) {
        pending_root = accepted_inference_t {
          .frame_id = frame_id,
          .observation_timestamp_us = adaptive_reuse_request.observation_timestamp_us,
          .transaction_token = accepted_transaction_token,
          .baseline_frame_id = accepted_submission_class == submission_class_e::gpu_undecided ?
                                 adaptive_reuse_request.baseline_frame_id : 0u,
          .input_region = input_region,
          .color_space = color_space,
          .subtitle_context = subtitle_context,
          .subtitle_line_policy = subtitle_line_policy,
          .optional_work = accepted_optional_work,
          .submission_class = accepted_submission_class,
          .subtitle_work = subtitle_work,
          .ocr_armed = ocr_armed,
          .depth_only_history = depth_only_history,
          .dump_forced = snapshot_raw_model_depth,
          .submitted_source = {},
        };
        if (subtitle_direct_glyph.enabled()) {
          // Only the descriptor is needed after admission; immutable RGB is the OCR snapshot.
          pending_root->submitted_source = input_srv;
        }
        cached_current_publication.reset();
        if (diagnostics_enabled) {
          throughput_stats_enqueues++;
          if (accepted_submission_class == submission_class_e::force_infer) {
            throughput_stats_force_infer_enqueues++;
          } else if (
            accepted_submission_class == submission_class_e::gpu_undecided
          ) {
            throughput_stats_gpu_undecided_enqueues++;
          }
          if (depth_optional_work_suppresses_subtitle(accepted_optional_work)) {
            throughput_stats_subtitle_suppressed++;
          }
          if (ocr_armed) {
            // GPU-undecided roots are branch-opaque: this counts requested/available optional
            // work, never a CPU claim that the OCR child actually ran.
            throughput_stats_ocr_armed++;
          }
        }
      }

      return make_result(
        completed,
        {
          .inference_enqueued = enqueued && accepted_submission_class == submission_class_e::force_infer,
          .gpu_undecided_transaction_enqueued = enqueued &&
                                                accepted_submission_class == submission_class_e::gpu_undecided,
          .subtitle_ocr_inference_enqueued = enqueued && ocr_armed &&
                                             accepted_optional_work == depth_optional_work_mode_e::ordinary,
        }
      );
    }
  };

  video_depth_estimator::video_depth_estimator(
    Microsoft::WRL::ComPtr<ID3D11Device> device,
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context,
    const std::filesystem::path &assets_dir,
    const config::video_t::sbs_t &cfg,
    const config::depth_model_info &model,
    normalized_depth_diagnostics_e normalized_depth_diagnostics,
    subtitle_legacy_matte_e legacy_matte
  ):
      pimpl(std::make_unique<impl>(
        device,
        context,
        assets_dir,
        cfg,
        model,
        normalized_depth_diagnostics,
        legacy_matte
      )) {}

  video_depth_estimator::~video_depth_estimator() = default;

  void video_depth_estimator::set_telemetry_performance(std::shared_ptr<host_sbs_telemetry::collector> performance) {
    if (pimpl && pimpl->diagnostics_enabled) {
      pimpl->telemetry_performance = std::move(performance);
    }
  }

  bool video_depth_estimator::is_valid() const {
    return pimpl && pimpl->is_operational();
  }

  bool video_depth_estimator::configure_subtitle_direct_glyph(const subtitle_direct_glyph_configuration_t &cfg, std::string &error) {
    error.clear();
    if (!pimpl || !pimpl->is_operational() || pimpl->configuration_locked ||
        pimpl->subtitle_direct_glyph.enabled()) {
      error = "Direct glyph must configure once before first input";
      return false;
    }
    return pimpl->prepare_subtitle_direct_glyph(cfg, error);
  }

  void video_depth_estimator::set_subtitle_conditioning_diagnostics_always(const bool always) noexcept {
    if (pimpl) {
      pimpl->subtitle_conditioning_diagnostics_always = always;
    }
  }

#ifdef SUNSHINE_TESTS
  bool video_depth_estimator::subtitle_conditioning_diagnostics_allocated_for_tests() const noexcept {
    return pimpl && pimpl->subtitle_conditioning_diagnostics.has_value();
  }
#endif

#ifdef SUNSHINE_TESTS
  bool video_depth_estimator::configure_joined_completion_gate_for_native_pacing_test(
    const joined_completion_test_gate::configuration_t &configuration
  ) {
    return pimpl && !pimpl->pending_root && !pimpl->pending_depth_inference_event_recorded &&
           pimpl->install_joined_completion_test_gate(configuration);
  }

  std::optional<joined_completion_test_gate::snapshot_t>
  video_depth_estimator::joined_completion_gate_status_for_benchmark() const noexcept {
    if (!pimpl || !pimpl->completion_test_gate) return std::nullopt;
    return pimpl->completion_test_gate->snapshot();
  }
#endif

  bool video_depth_estimator::can_accept_frame() {
    return pimpl && pimpl->can_accept();
  }

  bool video_depth_estimator::has_terminal_failure() const {
    return !pimpl || !pimpl->is_operational();
  }

  depth_step_result_t video_depth_estimator::estimate_depth(
    ID3D11ShaderResourceView *input_srv,
    input_color_space color_space,
    std::uint64_t frame_id,
    bool snapshot_debug_inputs,
    depth_input_region_t input_region,
    depth_optional_work_mode_e optional_work,
    gpu_adaptive_reuse_request adaptive_reuse,
    std::uint64_t subtitle_context_generation,
    subtitle_line_policy_e subtitle_line_policy
  ) {
    pimpl->configuration_locked = true;
    return pimpl->estimate(
      input_srv,
      color_space,
      frame_id,
      snapshot_debug_inputs,
      input_region,
      optional_work,
      adaptive_reuse,
      subtitle_context_generation,
      subtitle_line_policy
    );
  }

  depth_step_result_t video_depth_estimator::finish_pending_depth_for_evaluation(
    input_color_space color_space,
    bool snapshot_debug_inputs
  ) {
    return pimpl->finish_pending(color_space, snapshot_debug_inputs);
  }

  pending_depth_poll_result video_depth_estimator::try_finish_pending_depth_nonblocking(
    input_color_space color_space,
    bool snapshot_debug_inputs,
    const pending_depth_source_access_t source_access
  ) {
    return pimpl ? pimpl->try_finish_pending_nonblocking(
                     color_space,
                      snapshot_debug_inputs,
                      source_access
                   ) :
                   pending_depth_poll_result {.ready = true};
  }

  pending_depth_poll_result video_depth_estimator::try_finish_pending_depth_until(
    input_color_space color_space,
    const std::chrono::steady_clock::time_point deadline,
    const std::uint32_t max_queries,
    bool snapshot_debug_inputs,
    bool allow_completion_notification,
    const pending_depth_source_access_t source_access
  ) {
    return pimpl ? pimpl->try_finish_pending_until(
                     color_space,
                     deadline,
                     max_queries,
                     snapshot_debug_inputs,
                      allow_completion_notification,
                      source_access
                   ) :
                   pending_depth_poll_result {.ready = true};
  }

  void video_depth_estimator::poll_gpu_completion() {
    // Use the same poll (including its measured CPU cost) during idle owner-thread checks.
    // Functional proof survives in the single exact-current cache for the rendering caller.
    (void) poll_publication_receipts();
  }

  publication_receipt_poll_result video_depth_estimator::poll_publication_receipts() {
    if (!pimpl) {
      return {};
    }
    const auto started = pimpl->diagnostics_enabled ?
                           std::chrono::steady_clock::now() :
                           std::chrono::steady_clock::time_point {};
    auto result = pimpl->poll_publication_receipts();
    if (pimpl->diagnostics_enabled) {
      sbs_perf::add_sample_ms(
        "publication_receipt_poll_cpu",
        std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started
        ).count()
      );
    }
    return result;
  }

  bool video_depth_estimator::publication_is_current(
    const host_sbs_gpu_completion_receipt::expected_t &publication
  ) const noexcept {
    return pimpl && pimpl->publication_is_current(publication);
  }

  std::optional<subtitle_direct_glyph_diagnostic_views_t>
  video_depth_estimator::subtitle_direct_glyph_diagnostic_views_for_evaluation(std::uint64_t frame, std::uint64_t observation) const noexcept {
    if (!pimpl || !pimpl->is_operational() || !pimpl->subtitle_direct_glyph.enabled() || !pimpl->subtitle_selected_matte_commands ||
        frame == 0u || observation == 0u) return std::nullopt;
    const auto &publication = pimpl->current_publication;
    if (pimpl->pending_root || publication.frame_id != frame || publication.observation_timestamp_us != observation ||
        !pimpl->publication_is_current(publication)) return std::nullopt;
    const auto draw = pimpl->completed_subtitle_draw.lock();
    if (!draw) return std::nullopt;
    subtitle_direct_glyph_diagnostic_views_t result;
    result.model_frame = frame;
    result.model_observation_us = observation;
    result.selected_matte = draw->srv[subtitle_publication::matte];
    result.selected_receipt = draw->srv[subtitle_publication::receipt];
    result.ocr_record = pimpl->ocr_box_record_srv;
    pimpl->subtitle_direct_glyph.fill_diagnostic_views(result);
    return result;
  }

  std::filesystem::path video_depth_estimator::direct_glyph_engine_path_for_evaluation() const {
    if (!pimpl || !pimpl->subtitle_direct_glyph.enabled()) return {};
    const auto &key = pimpl->subtitle_direct_glyph.engine_key();
    const auto delimiter = key.find(':');
    if (delimiter == std::string::npos) return {};
    return pimpl->shader_root.parent_path().parent_path() / key.substr(delimiter + 1);
  }

  depth_telemetry_poll_result video_depth_estimator::latest_depth_telemetry() const {
    return pimpl ? pimpl->cached_depth_telemetry : depth_telemetry_poll_result {.failed = true};
  }

}  // namespace models
