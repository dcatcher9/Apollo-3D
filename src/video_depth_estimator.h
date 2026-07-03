#pragma once

#include <d3d11.h>
#include <filesystem>
#include <memory>
#include <string>
#include <wrl/client.h>

#include "config.h"

namespace models {

    void precompile_tensorrt_engine(const std::filesystem::path& assets_dir, const std::string& model_name, const std::string& model_url);

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
         *            guided_upsample, guided_sigma, depth_model, depth_model_url).
         */
        video_depth_estimator(Microsoft::WRL::ComPtr<ID3D11Device> device,
                              Microsoft::WRL::ComPtr<ID3D11DeviceContext> context,
                              const std::filesystem::path& assets_dir,
                              const config::video_t::sbs_t& cfg);

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
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> estimate_depth(ID3D11ShaderResourceView* input_srv, bool is_hdr = false);

    private:
        struct impl;
        std::unique_ptr<impl> pimpl;
    };

}
