#pragma once

#include "config.h"

#include <array>
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

  void precompile_tensorrt_engine(const std::filesystem::path &assets_dir, const config::depth_model_info &model);
  /** Build, deserialize, create, and warm one reusable execution context for the active model. */
  bool prepare_tensorrt_model(
    const std::filesystem::path &assets_dir,
    const config::depth_model_info &model,
    const std::string &adapter_name
  );
  engine_build_status tensorrt_model_prepare_status(const config::depth_model_info &model);
  engine_build_status tensorrt_engine_build_status(
    const std::filesystem::path &assets_dir,
    const config::depth_model_info &model
  );

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
    std::uint64_t enqueued_frame_id = 0;  ///< Identity attached to the newly submitted inference.
    bool cuda_graph_active = false;  ///< TensorRT enqueue is currently replaying a captured graph.
  };

  /**
   * @brief Offline readback of the shipping SubjectState scene latch after one completed depth
   *        frame. Never populated or read on the live stream path.
   *
   * `scene_age` is the authoritative SubjectState[0].y value written by
   * depth_subject_resolve_cs.hlsl. `hard_cut` is reconstructed from its only post-initialization
   * reset transition (prior age >= 7, current age == 0) when evidence is read at every completed
   * depth frame. `runtime_scene_id` starts at zero and increments on those resets.
   */
  struct runtime_scene_evidence {
    bool valid = false;
    std::uint64_t completed_frame_id = 0;
    std::uint64_t runtime_scene_id = 0;
    float scene_age = 0.0f;
    bool subject_initialized = false;
    bool hard_cut = false;
    bool scene_start = false;
    /** Exact three-float4 SubjectState snapshot; offline harness only. */
    std::array<float, 12> subject_state {};
  };

  struct artistic_policy_provenance {
    bool consumed = false;
    std::string authorization = "none";
    std::string model_onnx_sha256;
    std::string policy_metadata_sha256;
    std::string deployment_geometry_allowlist_sha256;
  };

  enum class artistic_policy_authorization {
    deployment,
    candidate_evaluation,
    headset_review,
  };

  class video_depth_estimator {
  public:
    /**
     * @brief Construct a new video depth estimator
     *
     * @param device D3D11 Device used for the capture pipeline
     * @param context D3D11 Device Context
     * @param assets_dir Path to the assets directory (for model loading)
     * @param cfg Resolved depth, temporal, subject, camera, and artistic-policy tuning.
     * @param model The selected depth model: name/url (which engine to load/build) plus the
     *            DA-V2-compatible base contract (pixel_values -> predicted_depth) and, when
     *            present and authorized, artistic_global = [safe scale ceiling, confidence].
     * @param consume_artistic_policy Whether a trusted optional artistic head may be consumed.
     * @param artistic_scale_override Offline scale-grid override; zero uses the learned head.
     * @param authorization Required trust level: final deployment, offline candidate evaluation,
     *                      or explicitly enabled live headset review of a fully gated stage.
     */
    video_depth_estimator(Microsoft::WRL::ComPtr<ID3D11Device> device, Microsoft::WRL::ComPtr<ID3D11DeviceContext> context, const std::filesystem::path &assets_dir, const config::video_t::sbs_t &cfg, const config::depth_model_info &model, bool consume_artistic_policy = true, float artistic_scale_override = 0.0f, artistic_policy_authorization authorization = artistic_policy_authorization::deployment);

    ~video_depth_estimator();

    /** True only when every mandatory engine, shader, and session resource initialized. */
    bool is_valid() const;

    /** Exact accepted policy identity; empty and false when the learned head is disabled. */
    artistic_policy_provenance artistic_policy_status() const;

    /** Set the exact destination-eye raster before first inference; later changes fail closed. */
    void set_artistic_output_geometry(
      std::uint32_t eye_width,
      std::uint32_t eye_height,
      float content_scale_x,
      float content_scale_y
    );

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

    /**
     * @brief Synchronizing SubjectState readback for the offline harness only.
     *
     * Call exactly once after each valid finish_pending_depth_for_evaluation() result. Skipping a
     * completed depth frame makes a later age reset ambiguous, so the returned evidence then must
     * not be treated as exact live cut parity. Re-reading the same frame returns the cached row.
     */
    runtime_scene_evidence read_runtime_scene_evidence_for_evaluation(
      std::uint64_t completed_frame_id
    );

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
  };

}  // namespace models
