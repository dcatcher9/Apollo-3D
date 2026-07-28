#pragma once

#include "config.h"

#include <cstdint>
#include <d3d11.h>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <wrl/client.h>

namespace models {

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

  /** Build, deserialize, create, and warm one reusable execution context for the active model. */
  bool prepare_tensorrt_model(
    const std::filesystem::path &assets_dir,
    const config::depth_model_info &model,
    const std::string &adapter_name
  );
  engine_build_status tensorrt_model_prepare_status(const config::depth_model_info &model);

  /**
   * @brief Result of one estimate call: the depth map for the reprojection (t1), plus the
   *        permanent Bestv2 subject state (t2).
   */
  struct estimate_result {
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> subject;  ///< permanent Bestv2 subject state (t2 of the reprojection)
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_frame_state;  ///< {min,max,initialized,frame_state}; frame_state 0 means an all-invalid completion held the prior depth.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ema_motion_mask;  ///< Edge-selective EMA snap mask.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> raw_model_depth;  ///< Raw model output buffer, before normalization/EMA/curvature; primarily for the offline evaluator.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> raw_model_depth_snapshot;  ///< Optional stable copy of the completed frame's raw output for a live Dump 3D request.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> model_input_snapshot;  ///< Optional stable NCHW/ImageNet-normalized input for the same live Dump 3D frame.
    int raw_width = 0;
    int raw_height = 0;
    // A TensorRT result completed and its GPU normalization passes were submitted. The associated
    // depth_frame_state decides on-GPU whether this completion contains valid depth or must hold
    // the previous matched color/depth output.
    bool completed_frame_valid = false;
    std::uint64_t completed_frame_id = 0;  ///< Caller-provided identity of that completed result.
    bool inference_enqueued = false;  ///< This call submitted inference for the supplied input frame.
    bool cuda_graph_active = false;  ///< TensorRT enqueue is currently replaying a captured graph.
  };

  /**
   * One opportunistic, nonblocking readback of the append-only diagnostic portion of
   * SubjectState. Live samples may be skipped or coalesced while the GPU is busy; absence of a
   * sample is not evidence that the controller did not update.
   * SubjectState[0..2] remain the production warp contract; diagnostics begin at element 3.
   */
  struct depth_telemetry_sample {
    int depth_width = 0;
    int depth_height = 0;
    float adaptive_pop_ratio = 1.0f;
    float edge_fraction = -1.0f;
    float change_fraction = 0.0f;
    float zero_anchor_shift_px = 0.0f;
    float subject_depth = 0.0f;
    float valid_depth_fraction = 0.0f;
    float effective_range_width = 0.0f;
    // Append-only current-frame evidence. These remain independent of the shot-latched controls
    // above so offline/live diagnostics can validate a boundary with two-sided evidence.
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
     * @param cfg Tuning knobs; see config::video_t::sbs_t (the estimator uses the depth-side
     *            fields: ema, depth_short_side, depth_max_aspect, and minmax_ema).
     * @param model The selected depth model: name/url (which engine to load/build) plus the
     *            DA-V2-compatible model contract (pixel_values -> predicted_depth).
     */
    video_depth_estimator(Microsoft::WRL::ComPtr<ID3D11Device> device, Microsoft::WRL::ComPtr<ID3D11DeviceContext> context, const std::filesystem::path &assets_dir, const config::video_t::sbs_t &cfg, const config::depth_model_info &model);

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

    // Non-copyable
    video_depth_estimator(const video_depth_estimator &) = delete;
    video_depth_estimator &operator=(const video_depth_estimator &) = delete;

    /**
     * @brief Estimate depth (and the subject-tracking state) for the given RGB frame.
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
      bool snapshot_debug_inputs = false
    );

    /**
     * @brief Finish and consume exactly one inference previously submitted by estimate_depth().
     *
     * It synchronizes the estimator stream, applies normalization/EMA/subject tracking exactly
     * once, and does not enqueue another inference. The offline evaluator uses this as its
     * exact current-frame quality path; production remains bounded matched-frame async.
     */
    estimate_result finish_pending_depth_for_evaluation(input_color_space color_space = input_color_space::srgb);

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
