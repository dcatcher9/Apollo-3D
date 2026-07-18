#pragma once

#include "config.h"

#include <cstdint>
#include <d3d11.h>
#include <filesystem>
#include <memory>
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
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> ema_motion_mask;  ///< Edge-selective EMA snap mask.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> raw_model_depth;  ///< Raw model output buffer, before normalization/EMA/curvature; primarily for the offline evaluator.
    int raw_width = 0;
    int raw_height = 0;
    bool completed_frame_valid = false;  ///< A new processed depth result completed during this call.
    std::uint64_t completed_frame_id = 0;  ///< Caller-provided identity of that completed result.
    bool inference_enqueued = false;  ///< This call submitted inference for the supplied input frame.
    bool cuda_graph_active = false;  ///< TensorRT enqueue is currently replaying a captured graph.
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
     * @return estimate_result; all views are owned by the estimator and overwritten by later calls.
     */
    estimate_result estimate_depth(ID3D11ShaderResourceView *input_srv, input_color_space color_space = input_color_space::srgb, std::uint64_t frame_id = 0);

    /**
     * @brief Finish and consume exactly one inference previously submitted by estimate_depth().
     *
     * It synchronizes the estimator stream, applies normalization/EMA/subject tracking exactly
         * once, and does not enqueue another inference. The offline evaluator uses this as its
         * exact current-frame quality path; production remains bounded matched-frame async.
         */
    estimate_result finish_pending_depth_for_evaluation(input_color_space color_space = input_color_space::srgb);

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
  };

}  // namespace models
