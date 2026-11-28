/**
 * @file src/host_sbs_direct_glyph_route.h
 * @brief Per-observation dispatch and retirement of the Direct Glyph subtitle model.
 */
#pragma once

#include "cuda_driver_api.h"
#include "host_sbs_direct_glyph_gpu.h"
#include "host_sbs_direct_glyph_publication_gpu.h"
#include "host_sbs_display_snapshot_rgb8_analysis_gpu.h"
#include "host_sbs_shader_cache.h"
#include "video_depth_estimator.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <d3d11.h>
#include <filesystem>
#include <memory>
#include <NvInferRuntime.h>
#include <optional>
#include <string>
#include <wrl/client.h>

namespace platf {
  struct high_precision_timer;
}  // namespace platf

namespace models::host_sbs_direct_glyph_route {
  // What one Direct Glyph step reads of the estimator's in-flight observation. Every view belongs
  // to the same ordered observation as the joined root the estimator proved complete.
  struct frame_t {
    depth_input_region_t region {};
    std::uint64_t frame = 0u, observation_us = 0u, token = 0u;
    std::uint64_t subtitle_context_generation = 0u;
    UINT color_space = 0u;  // The pending transfer mode; never applied a second time.
    bool video_caption = false;  // The observation's subtitle line policy is video_caption.
    // The actual pending postprocess function result and work, never guessed from the root.
    UINT postprocess_mode = 0u, expected_work = 0u;
    bool input_domain_reset = false;
    bool carry_unchanged_lines = false;  // Unchanged lines are carried over without a score.
    ID3D11ShaderResourceView *display_snapshot_band = nullptr;
    // Descriptor-only parent reference; source pixels come from the joined snapshot band.
    ID3D11ShaderResourceView *parent_raw_source = nullptr;
    ID3D11Buffer *contour_constants = nullptr;  // The actual C24 constants.
    ID3D11ShaderResourceView *contour_identity = nullptr;  // SCT
    ID3D11Buffer *root_dispatch = nullptr;  // The final root's indirect arguments.
    ID3D11ShaderResourceView *root = nullptr;
    ID3D11ShaderResourceView *ocr_box_record = nullptr;
    ID3D11ShaderResourceView *condition_params = nullptr;
    ID3D11ShaderResourceView *locator_state = nullptr;  // The actual current UIP40.
  };

  // What retirement asks the estimator to write.
  struct retirement_t {
    // False while the score is pending before its deadline; the observation stays reserved.
    bool retired = true;
    // The call waited for a GPU step of the score.
    bool waited = false;
    // The publisher's selected-matte command state, when this call published.
    std::optional<bool> selected_matte_commands;
  };

  // Direct Glyph for one estimator: the source audit, plan, score and publication resources, the
  // leased glyph model context, the one in-flight score and the last reported failure. Serialized
  // by the estimator's immediate context. The estimator calls each step from the place and thread
  // it always did, so CUDA registration stays on the thread that draws with the device.
  class route_t {
  public:
    route_t();
    ~route_t();
    route_t(const route_t &) = delete;
    route_t &operator=(const route_t &) = delete;

    // Authenticates the source audit closure, loads the plan and publication programs, leases a
    // glyph model context from the engine cache under assets_dir and prepares the score
    // transport. A failure leaves the route disabled.
    bool prepare(
      ID3D11Device *device,
      ID3D11DeviceContext *context,
      const subtitle_direct_glyph_configuration_t &cfg,
      const std::filesystem::path &assets_dir,
      CUdevice cuda_device,
      CUcontext cuda_ctx,
      std::string &error
    );

    // Discards a finished score for up to 250 ms, then recycles the context lease, or quarantines
    // it when the score transport could not release.
    void release() noexcept;

    [[nodiscard]] bool enabled() const noexcept {
      return enabled_;
    }

    // Registers the score buffers with CUDA ahead of the first submission.
    void register_interop_early() noexcept;

    // Audits the source band, plans the observation's glyph work and submits its score. Returns
    // the selected-matte command state when the step published without a score, and nothing when
    // it submitted one or stopped before publishing.
    [[nodiscard]] std::optional<bool> dispatch(ID3D11Device *device, ID3D11DeviceContext *context, const frame_t &frame);

    // Publishes the submitted score's alpha, or no protection once it fails or misses its
    // deadline. With `wait_until` it works the score through its GPU steps until that time: the
    // file-fed evaluation entry gives the score its whole deadline, a live completion the rest of
    // its own bounded wait. Without it the call takes at most one step and never waits.
    [[nodiscard]] retirement_t retire(ID3D11DeviceContext *context, std::optional<std::chrono::steady_clock::time_point> wait_until);

    // Works the submitted score through its GPU steps until `wait_until` and publishes nothing:
    // the next retire() publishes whatever this reached. For the time between two conversions,
    // when the score would otherwise wait for the next one to take its next step. Returns true
    // once the score has no step left to wait for.
    bool advance(std::chrono::steady_clock::time_point wait_until);

    // True when no submitted score has a step left to wait for.
    [[nodiscard]] bool settled() const noexcept {
      return !job_ || settled_.has_value();
    }

    [[nodiscard]] host_sbs_direct_glyph_publication::views_t views() const {
      return publication_.views();
    }

    // The lines of the score last planned: scored, and carried over unchanged.
    [[nodiscard]] host_sbs_direct_glyph_gpu::line_counts_t planned_lines() const noexcept {
      return score_.planned_lines();
    }

    // Forgets the last reported failure, so the next one logs again.
    void clear_reported_failure() noexcept {
      error_.clear();
    }

    // The last observation's alpha8, Work and source audit, with the native lines of a retired
    // score when the plan is captured.
    void fill_diagnostic_views(subtitle_direct_glyph_diagnostic_views_t &views) const noexcept;

    [[nodiscard]] const std::string &engine_key() const noexcept {
      return engine_key_;
    }

  private:
    struct job_t {
      host_sbs_direct_glyph_publication::request_t publication;
      host_sbs_direct_glyph_gpu::request_t score;
      std::chrono::steady_clock::time_point deadline;
    };

    void report_failure(const std::string &error);

    host_sbs_direct_glyph_publication::resources_t publication_;
    host_sbs_direct_glyph_gpu::resources_t score_;
    display_snapshot_rgb8_analysis_gpu_v1::resources_t source_;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> source_audit_;
    std::array<Microsoft::WRL::ComPtr<ID3D11ComputeShader>, host_sbs_shader_cache::direct_glyph_source_audit_specs.size()>
      source_shaders_;
    nvinfer1::IExecutionContext *execution_ = nullptr;
    std::string engine_key_;
    // Takes the job's score through its steps until `wait_until`; returns whether it waited.
    bool work_score(std::optional<std::chrono::steady_clock::time_point> wait_until);

    struct settled_t {
      host_sbs_direct_glyph_gpu::poll_status_e status = host_sbs_direct_glyph_gpu::poll_status_e::pending;
      host_sbs_direct_glyph_gpu::result_t result;
      std::string error;
    };
    std::optional<job_t> job_;
    // What the poll that took the score past its last step returned, until retire() publishes it.
    std::optional<settled_t> settled_;
    bool enabled_ = false;
    bool late_score_ = false;
    bool score_failed_ = false;
    bool capture_plan_ = false;
    bool score_retired_ = false;
    std::string error_;
  };
}  // namespace models::host_sbs_direct_glyph_route
