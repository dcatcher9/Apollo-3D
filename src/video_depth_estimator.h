#pragma once

#include <d3d11.h>
#include <filesystem>
#include <memory>
#include <string>
#include <wrl/client.h>

namespace models {

    void precompile_tensorrt_engine(const std::filesystem::path& assets_dir, const std::string& model_name, const std::string& model_url);

    /**
     * @brief Depth-estimation tuning knobs. Values are sensible fallbacks; display_vram
     *        populates every field from config::video.sbs.
     */
    struct depth_estimator_config {
        float ema_alpha = 0.2f;      ///< Temporal smoothing of the depth map (0-1); higher = snappier.
        int depth_short_side = 392;  ///< Depth map short-side resolution (iw3-style); clamped to the frame's native short side (never upscales).
        float max_aspect = 4.0f;     ///< Aspect-ratio cap (long side <= short * this).
        bool normalize = true;       ///< Per-frame min/max normalization of raw disparity. False = legacy 1-exp curve.
        float depth_gamma = 1.0f;    ///< Shaping exponent on normalized depth (normalize mode only); 1.0 = linear.
        float minmax_alpha = 0.1f;   ///< Temporal EMA blend for the normalized min/max (0-1).
        float edge_dilation = 0.0f;  ///< Foreground-biased edge smoothing strength (0 = off); reduces jaggy silhouette fringe.
        float depth_fps = 30.0f;     ///< Target depth-update rate; inference interval auto-derived from measured video fps. 0 = every frame.
        int depth_interval = 0;      ///< Manual override for the inference interval. 0 = auto from depth_fps.
        std::string model_name = "depth_anything_v2_fp16";  ///< Local file stem; engine cached as <model_name>.engine. Different models coexist.
        std::string model_url = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_fp16.onnx";  ///< Where to download the ONNX if absent.
    };

    class video_depth_estimator {
    public:
        /**
         * @brief Construct a new video depth estimator
         *
         * @param device D3D11 Device used for the capture pipeline
         * @param context D3D11 Device Context
         * @param assets_dir Path to the assets directory (for model loading)
         * @param input_width Width of the input video frame
         * @param input_height Height of the input video frame
         * @param cfg Tuning knobs (see depth_estimator_config)
         */
        video_depth_estimator(Microsoft::WRL::ComPtr<ID3D11Device> device,
                              Microsoft::WRL::ComPtr<ID3D11DeviceContext> context,
                              const std::filesystem::path& assets_dir,
                              int input_width, int input_height,
                              const depth_estimator_config& cfg = {});

        ~video_depth_estimator();

        // Non-copyable
        video_depth_estimator(const video_depth_estimator&) = delete;
        video_depth_estimator& operator=(const video_depth_estimator&) = delete;

        /**
         * @brief Estimate depth for the given RGB frame.
         * 
         * @param input_srv D3D11 ShaderResourceView containing the RGB image (usually B8G8R8A8_UNORM or R8G8B8A8_UNORM).
         * @return Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> A shader resource view for the resulting depth map.
         *         This texture is owned by the estimator and will be overwritten on the next call.
         */
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> estimate_depth(Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> input_srv, bool is_hdr = false);

    private:
        struct impl;
        std::unique_ptr<impl> pimpl;
    };

}
