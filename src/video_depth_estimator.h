#pragma once

#include "config.h"
#include "host_sbs_resolution.h"

#include <chrono>
#include <cstdint>
#include <d3d11.h>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <wrl/client.h>

namespace models {

  /** Immutable identity of the exact model bytes behind raw depth.
   *
   * preprocess_profile is populated only when the complete model name/URL/SHA identity resolves
   * to a calibrated coordinate-v2 entry and the runtime RGB-to-NCHW source closure has the exact
   * calibrated digest. An empty profile is explicitly uncalibrated rather than an inferred match
   * by registry position.
   */
  struct raw_model_provenance_t {
    std::string depth_model;
    std::string depth_model_url;
    std::string onnx_sha256;
    std::string preprocess_profile;
    std::string preprocess_source_closure_sha256;
  };

  /** Immutable identity of the complete producer closure behind a Host SBS parallax result. */
  struct parallax_v2_shader_provenance_t {
    std::uint32_t source_closure_schema = 0;
    std::uint32_t source_compile_flags = 0;
    std::uint32_t source_macro_count = 0;
    std::string source_closure_sha256;
  };

  enum class engine_build_status {
    unknown,
    building,
    ready,
    failed,
  };

  enum class input_color_space : uint32_t {
    srgb = 0,  ///< gamma-encoded SDR UNORM capture
    linear_sdr = 1,  ///< linear FP16 capture targeting an SDR stream
    scrgb_hdr = 2,  ///< linear scRGB FP16 HDR capture; tone-map for the SDR-trained model
  };

  constexpr bool input_color_space_is_linear(const input_color_space color_space) {
    return color_space != input_color_space::srgb;
  }

  /**
   * Exact full-source placement of the image submitted to DAV2.
   *
   * video_region=false means the estimator input and final parallax cover the whole source and
   * analysis_generation is zero.
   * video_region=true means the estimator input is a same-format crop of source_rect while final
   * color still covers source_width x source_height. analysis_generation changes only when the
   * semantic video identity or crop extent changes; a pure position move may retain scene state.
   */
  struct depth_input_region_t {
    std::uint32_t source_width = 0u;
    std::uint32_t source_height = 0u;
    std::uint32_t left = 0u;
    std::uint32_t top = 0u;
    std::uint32_t right = 0u;
    std::uint32_t bottom = 0u;
    std::uint64_t analysis_generation = 0u;
    bool video_region = false;

    constexpr std::uint32_t width() const noexcept {
      return right > left ? right - left : 0u;
    }

    constexpr std::uint32_t height() const noexcept {
      return bottom > top ? bottom - top : 0u;
    }

    constexpr bool valid() const noexcept {
      if (source_width == 0u || source_height == 0u || left >= right || top >= bottom ||
          right > source_width || bottom > source_height) {
        return false;
      }
      if (video_region) {
        return analysis_generation != 0u;
      }
      return left == 0u && top == 0u && right == source_width &&
             bottom == source_height && analysis_generation == 0u;
    }

    constexpr bool same_analysis_domain(const depth_input_region_t &other) const noexcept {
      return video_region == other.video_region &&
             source_width == other.source_width && source_height == other.source_height &&
             width() == other.width() && height() == other.height() &&
             analysis_generation == other.analysis_generation;
    }

    constexpr bool operator==(const depth_input_region_t &) const = default;
  };

  /** Stable semantic identity used to assign an ROI analysis generation.
   *
   * Position and observer snapshot generation are deliberately absent. Moving the same video or
   * receiving an otherwise-identical helper heartbeat does not create a new analysis domain.
   */
  struct video_depth_domain_key_t {
    std::uint32_t source_width = 0u;
    std::uint32_t source_height = 0u;
    std::uint32_t semantic_width = 0u;
    std::uint32_t semantic_height = 0u;
    std::uint32_t crop_width = 0u;
    std::uint32_t crop_height = 0u;
    std::uint64_t hwnd = 0u;
    std::uint32_t process_id = 0u;
    std::int32_t document_id = 0;
    std::int32_t video_id = 0;

    constexpr bool operator==(const video_depth_domain_key_t &) const = default;
  };

  /** Assign monotonically changing nonzero generations to stable ROI semantic identities. */
  class video_depth_analysis_generation_tracker_t {
  public:
    std::uint64_t select(const video_depth_domain_key_t &key) noexcept {
      if (!key_ || *key_ != key) {
        key_ = key;
        if (++generation_ == 0u) {
          ++generation_;
        }
      }
      return generation_;
    }

    /** Mark the active route full-source; a later ROI must rearm even for the prior identity. */
    void select_full_source() noexcept {
      key_.reset();
    }

  private:
    std::optional<video_depth_domain_key_t> key_;
    std::uint64_t generation_ = 0u;
  };

  /** Pure transition decision used immediately before temporal/camera state is reset. */
  class depth_input_domain_tracker_t {
  public:
    bool update(
      const depth_input_region_t &region,
      const input_color_space color_space
    ) noexcept {
      const bool changed = !initialized_ || color_space_ != color_space ||
                           !region_.same_analysis_domain(region);
      if (changed) {
        region_ = region;
        color_space_ = color_space;
        initialized_ = true;
      }
      return changed;
    }

  private:
    depth_input_region_t region_ {};
    input_color_space color_space_ = input_color_space::srgb;
    bool initialized_ = false;
  };

  /** Build and warm the active model and prewarm fixed-shape Host SBS shader bytecode. */
  bool prepare_tensorrt_model(
    const std::filesystem::path &assets_dir,
    const config::depth_model_info &model,
    const std::string &adapter_name
  );
  /** Build and warm the optional authenticated PP-OCRv6 tiny detector. Failure is fail-flat. */
  bool prepare_ocr_tensorrt_model(
    const std::filesystem::path &assets_dir,
    const std::string &adapter_name
  );
  engine_build_status tensorrt_model_prepare_status(const config::depth_model_info &model);

  /**
   * @brief Result of one estimate call: private cut-analysis depth, cut state, and the
   *        authenticated Host SBS parallax field/state.
   */
  struct estimate_result {
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> cut_state;  ///< Cut-analysis state shared with telemetry and coordinate production.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_frame_state;  ///< {min,max,initialized,frame_state}; frame_state 0 means an all-invalid completion held the prior depth.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ema_motion_mask;  ///< Edge-selective EMA snap mask.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> raw_model_depth;  ///< Raw model output buffer, before normalization/EMA/curvature; primarily for the offline evaluator.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> raw_model_depth_snapshot;  ///< Optional stable copy of the completed frame's raw output for a live Dump 3D request.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> model_input_snapshot;  ///< Optional stable NCHW/ImageNet-normalized input for the same live Dump 3D frame.
    std::shared_ptr<const raw_model_provenance_t> raw_model_provenance;  ///< Capture-time model-byte identity; copied by pointer on ordinary frames.
    // Production V2 outputs. candidate_parallax is immutable pre-conditioner evidence;
    // ownership_refined_parallax is the full-resolution source-contour ownership result consumed
    // by the vertical pass, vertical_majorant is the upper-envelope diagnostic,
    // vertical_conditioned is the fixed upper/lower vertical share consumed by the pure row
    // majorant. base_final_parallax is the ordinary post-limiter field; final_parallax is the
    // OCR-conditioned full-source live position authority in ordinary mode. In ROI
    // mode it is crop-local producer q and becomes renderer authority only with input_region's
    // authenticated scale/collar embedding. coordinate is an optional Dump-3D-only snapshot,
    // never a live resource or authentication prerequisite. The legacy `shadow_*` prefix remains
    // for dump compatibility.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_coordinate;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_candidate_parallax;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_ownership_refined_parallax;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_vertical_majorant;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_vertical_conditioned;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_base_final_parallax;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_final_parallax;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_state;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> shadow_frame_stats;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ocr_box_record;  ///< Exact-frame OCR8 208-word record; flags==1 is authoritative, including empty observations.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> subtitle_locator_state;  ///< Current compact 80-word SLR8 state after consuming ocr_box_record.
    std::shared_ptr<const parallax_v2_shader_provenance_t>
      parallax_v2_shader_provenance;  ///< Exact producer shader closure when V2 is active.
    int raw_width = 0;
    int raw_height = 0;
    // A TensorRT result completed and its GPU normalization passes were submitted. The associated
    // depth_frame_state decides on-GPU whether this completion contains valid depth or must hold
    // the previous matched color/depth output.
    bool completed_frame_valid = false;
    std::uint64_t completed_frame_id = 0;  ///< Caller-provided identity of that completed result.
    bool inference_enqueued = false;  ///< This call submitted inference for the supplied input frame.
    bool cuda_graph_active = false;  ///< DAV2 TensorRT enqueue is currently replaying a captured graph.
    bool ocr_cuda_graph_active = false;  ///< PP-OCRv6 enqueue is currently replaying its own captured graph.
    bool parallax_v2_producer_active = false;  ///< All production V2 producer shaders/resources are active.
    float parallax_v2_raw_coordinate_scale = 0.0f;  ///< Fixed authenticated model/shape coordinate scale.
    float parallax_v2_requested_pop_strength = 0.0f;  ///< Fixed V2 request from cfg.pop_strength only; no legacy adaptive ratio or ceiling is consumed.
    float parallax_v2_requested_gain = 0.0f;  ///< One-eye source-U gain before safety attenuation.
    depth_input_region_t input_region {};  ///< Exact source domain that owns this completion.
    input_color_space color_space = input_color_space::srgb;  ///< Exact transfer domain used for this completion.
    bool input_domain_reset = false;  ///< Temporal/camera state was reset before this completion.
  };

  /** Fail-closed CPU authentication for a completed live V2 result.
   *
   * This verifies the complete model/preprocess/shape and producer source identities, the exact
   * full-source/analysis-region shape relation, the presence of every production V2 resource, and
   * the fixed-pop gain relation. Dump-only canonical coordinate evidence is deliberately excluded.
   * It does not map GPU state; the live shader authenticates the per-frame contract tag before
   * sampling geometry.
   */
  bool parallax_v2_result_is_authenticated(const estimate_result &result);

  enum class host_sbs_renderer_e {
    awaiting_v2,
    parallax_v2,
    failed_flat,
  };

  /** One-way production latch. Host SBS authenticates the first completed V2 field before it may
   * render geometry; rejection is terminal for this stream and can only render live identity.
   */
  constexpr host_sbs_renderer_e latch_host_sbs_renderer(
    const host_sbs_renderer_e current,
    const bool result_authenticated
  ) {
    if (current != host_sbs_renderer_e::awaiting_v2) {
      return current;
    }
    return result_authenticated ?
             host_sbs_renderer_e::parallax_v2 :
             host_sbs_renderer_e::failed_flat;
  }

  /** A failed V2 producer is terminal for geometry, but not for the video stream. The host
   * stops producing unused depth and draws each current color frame through flat identity.
   */
  constexpr bool host_sbs_renderer_uses_depth_pipeline(
    const host_sbs_renderer_e renderer
  ) {
    return renderer != host_sbs_renderer_e::failed_flat;
  }

  inline constexpr auto host_sbs_v2_max_matched_repeat_age =
    std::chrono::milliseconds {250};

  /** A retained capture may be reconverted after an arbitrarily long static interval. Its exact
   * matched completion is still current; wall-clock age only expires geometry for an older source.
   */
  constexpr bool host_sbs_matched_completion_is_current(
    const bool source_unchanged,
    const std::chrono::steady_clock::duration completion_age
  ) {
    return source_unchanged ||
           completion_age <= host_sbs_v2_max_matched_repeat_age;
  }

  constexpr bool host_sbs_should_repeat_matched_output(
    const host_sbs_renderer_e renderer,
    const bool has_matched_frame,
    const bool matched_output_valid,
    const std::chrono::steady_clock::duration repeat_source_age =
      std::chrono::steady_clock::duration::zero(),
    const bool source_unchanged = false
  ) {
    return renderer == host_sbs_renderer_e::parallax_v2 &&
           !has_matched_frame && matched_output_valid &&
           host_sbs_matched_completion_is_current(
             source_unchanged,
             repeat_source_age
           );
  }

  /** A terminal producer failure moves every nonfailed Host SBS stream to live flat identity. */
  constexpr host_sbs_renderer_e fail_host_sbs_renderer_flat(
    const host_sbs_renderer_e current
  ) {
    return current == host_sbs_renderer_e::failed_flat ? current :
                                                        host_sbs_renderer_e::failed_flat;
  }

  /**
   * One opportunistic, nonblocking readback of the append-only diagnostic portion of
   * cut-analysis state. Live samples may be skipped or coalesced while the GPU is busy; absence
   * of a sample is not evidence that the detector did not update.
   */
  struct depth_telemetry_sample {
    int depth_width = 0;
    int depth_height = 0;
    float adaptive_pop_ratio = 1.0f;  ///< Retained wire field; V2 always publishes 1.
    float edge_fraction = -1.0f;  ///< Retained wire field; unavailable in V2.
    float change_fraction = 0.0f;
    float zero_anchor_shift_px = 0.0f;  ///< Retained wire field; unavailable in V2.
    float subject_depth = 0.0f;  ///< Retained wire field; unavailable in V2.
    float valid_depth_fraction = 0.0f;
    float effective_range_width = 0.0f;
    // Current-frame cut evidence used to distinguish detector state from incoming observations.
    float current_edge_fraction = -1.0f;
    float current_zero_anchor_candidate_shift_px = -1.0f;
    float structural_change_fraction = -1.0f;
    float raw_rgb_change_fraction = -1.0f;
    std::uint32_t scene_age = 0;
    std::uint32_t cut_flags = 0;
    std::uint32_t hard_cut_count = 0;
    std::uint32_t external_cut_count = 0;
    std::uint32_t empty_raw_count = 0;
    std::uint32_t collapsed_raw_count = 0;
    std::uint64_t sampled_frame_id = 0;
    bool profile_initialized = false;
    bool anchor_valid = false;
    bool range_collapsed = false;
    bool depth_ready = false;
    bool hard_cut_pulse = false;
  };

  struct depth_telemetry_poll_result {
    std::optional<depth_telemetry_sample> sample;
    bool copy_scheduled = false;
    bool failed = false;
  };

  class video_depth_estimator {
  public:
    /**
     * @brief Construct a new video depth estimator
     *
     * @param device D3D11 Device used for the capture pipeline
     * @param context D3D11 Device Context
     * @param assets_dir Path to the assets directory (for model loading)
     * @param cfg Shared SBS controls; see config::video_t::sbs_t (the estimator consumes
     *            pop_strength and cuda_graph; the depth-side analysis parameters are the fixed
     *            config::host_sbs_v2_live_calibration values).
     * @param model The selected depth model: name/url (which engine to load/build) plus the
     *            DA-V2-compatible model contract (pixel_values -> predicted_depth).
     */
    video_depth_estimator(
      Microsoft::WRL::ComPtr<ID3D11Device> device,
      Microsoft::WRL::ComPtr<ID3D11DeviceContext> context,
      const std::filesystem::path &assets_dir,
      const config::video_t::sbs_t &cfg,
      const config::depth_model_info &model
    );

    ~video_depth_estimator();

    /** True only when every mandatory engine, shader, and session resource initialized. */
    bool is_valid() const;

    /**
     * @brief Nonblocking producer-side readiness check for matched-frame capture.
     *
     * Returns true only when the estimator can accept a new input immediately. A false return
     * accounts for the source opportunity (and, when CUDA is still working, one busy drop) in the
     * throughput telemetry. It never consumes a completed depth result; the next estimate_depth()
     * call performs that consumption. The live pipeline uses this before copying a full-resolution
     * color frame into its private matched slot.
     */
    bool can_accept_frame();

    /** True after CUDA/TensorRT reports a non-recoverable execution/interoperability error. */
    bool has_terminal_failure() const;

    // Non-copyable
    video_depth_estimator(const video_depth_estimator &) = delete;
    video_depth_estimator &operator=(const video_depth_estimator &) = delete;

    /**
     * @brief Estimate private analysis depth and the authenticated Host SBS parallax field.
     *
     * @param input_srv D3D11 ShaderResourceView containing the RGB image (usually B8G8R8A8_UNORM or R8G8B8A8_UNORM).
     * @param snapshot_debug_inputs Copy the completed frame's exact model input and raw output
     *        before the next asynchronous inference can overwrite them. Intended only for an
     *        explicit Dump 3D.
     * @return estimate_result; all views are owned by the estimator and overwritten by later calls.
     */
    estimate_result estimate_depth(
      ID3D11ShaderResourceView *input_srv,
      input_color_space color_space = input_color_space::srgb,
      std::uint64_t frame_id = 0,
      bool snapshot_debug_inputs = false,
      depth_input_region_t input_region = {}
    );

    /**
     * @brief Finish and consume exactly one inference previously submitted by estimate_depth().
     *
     * It synchronizes the estimator stream and applies normalization, EMA, cut analysis, and
     * coordinate production exactly once without enqueueing another inference. The offline evaluator uses this as its
     * exact current-frame quality path; production uses the separate idle-recovery entry point.
     */
    estimate_result finish_pending_depth_for_evaluation(input_color_space color_space = input_color_space::srgb);

    /**
     * @brief Consume the just-submitted live inference when capture resumed after an idle gap.
     *
     * The normal Host SBS path stays asynchronous. This one-inference drain is used only when the
     * current source frame would otherwise be left flat because capture went idle before a later
     * convert() could poll its completion. The encode thread remains the D3D context owner.
     */
    estimate_result finish_pending_depth_for_idle_recovery(
      input_color_space color_space,
      bool snapshot_debug_inputs = false
    );

    /**
     * Poll completed telemetry copies and optionally enqueue one new copy after the caller has
     * submitted the critical warp/output work. Never flushes, waits, or maps an unsignaled slot.
     * A busy three-slot ring deliberately drops the sampling opportunity rather than delaying
     * capture, so callers must not compare its sample count one-for-one with offline traces.
     */
    depth_telemetry_poll_result poll_depth_telemetry(
      bool schedule_copy,
      std::uint64_t sampled_frame_id
    );

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
  };

}  // namespace models
