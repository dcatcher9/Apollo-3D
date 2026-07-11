#pragma once

#include <d3d11.h>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <wrl/client.h>

#include "config.h"

namespace models {

    enum class input_color_space : uint32_t {
        srgb = 0,       ///< gamma-encoded SDR UNORM capture
        linear_sdr = 1, ///< linear FP16 capture targeting an SDR stream
        scrgb_hdr = 2,  ///< linear scRGB FP16 HDR capture; tone-map for the SDR-trained model
    };

    void precompile_tensorrt_engine(const std::filesystem::path& assets_dir, const config::depth_model_info& model);

    /**
     * @brief Result of one estimate call: the depth map for the reprojection (t1), plus the
     *        subject-tracking state (t2) when sbs_3d_subject_track is on.
     */
    struct estimate_result {
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> subject;  ///< subject-tracking state (t2 of the reprojection); null unless sbs_3d_subject_track
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> plane_lock;  ///< Bestv2's smoothed subject silhouette mask (t4); null outside the exact Bestv2 path.
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> raw_model_depth;  ///< Raw model output buffer, before normalization/EMA/curvature; primarily for the offline evaluator.
        int raw_width = 0;
        int raw_height = 0;
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
         *            fields: ema, depth_short_side, depth_max_aspect, minmax_ema, depth_fps,
         *            guided_upsample, guided_sigma).
         * @param model The selected depth model: name/url (which engine to load/build) plus the
         *            per-model contract (input_rank, output_transform, output/input tensor names).
         */
        video_depth_estimator(Microsoft::WRL::ComPtr<ID3D11Device> device,
                              Microsoft::WRL::ComPtr<ID3D11DeviceContext> context,
                              const std::filesystem::path& assets_dir,
                              const config::video_t::sbs_t& cfg,
                              const config::depth_model_info& model);

        ~video_depth_estimator();

        // Non-copyable
        video_depth_estimator(const video_depth_estimator&) = delete;
        video_depth_estimator& operator=(const video_depth_estimator&) = delete;

        /**
         * @brief Estimate depth (and the subject-tracking state) for the given RGB frame.
         *
         * @param input_srv D3D11 ShaderResourceView containing the RGB image (usually B8G8R8A8_UNORM or R8G8B8A8_UNORM).
         * @return estimate_result; all views are owned by the estimator and overwritten by later calls.
         */
        estimate_result estimate_depth(ID3D11ShaderResourceView* input_srv,
                                       input_color_space color_space = input_color_space::srgb);

        /**
         * @brief Finish and consume exactly one inference previously submitted by estimate_depth().
         *
         * This is an offline-evaluation operation. It synchronizes the estimator stream, applies
         * normalization/EMA/subject tracking exactly once, and does not enqueue another inference.
         * The live capture path must continue to use estimate_depth() alone so it stays asynchronous.
         */
        estimate_result finish_pending_depth_for_benchmark(ID3D11ShaderResourceView* input_srv,
          input_color_space color_space = input_color_space::srgb);

    private:
        struct impl;
        std::unique_ptr<impl> pimpl;
    };

}
