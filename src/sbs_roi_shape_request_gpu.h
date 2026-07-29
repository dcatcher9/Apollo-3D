/**
 * @file src/sbs_roi_shape_request_gpu.h
 * @brief Nonblocking D3D11 dispatcher/readback ring for Host SBS ROI shape requests.
 */
#pragma once

#include "sbs_roi_shape_request.h"

#include <cstdint>
#include <d3d11.h>
#include <filesystem>
#include <memory>
#include <optional>
#include <wrl/client.h>

namespace models {
  struct sbs_roi_shape_request_gpu_submission {
    /**
     * GPU-resident scene-controller rule state. A null view intentionally binds an all-zero
     * fallback buffer, allowing the shader to publish a canonical full-frame request.
     */
    ID3D11ShaderResourceView *rule_state = nullptr;
    /** CPU-only provenance retained with the staging slot; never enters the GPU request ABI. */
    std::uint64_t source_frame_id = 0u;

    std::uint32_t source_width = 0u;
    std::uint32_t source_height = 0u;
    std::uint32_t canonical_model_width = 0u;
    std::uint32_t canonical_model_height = 0u;

    std::uint32_t target_pixel_budget = 0u;
    std::uint32_t profile_max_width = 0u;
    std::uint32_t profile_max_height = 0u;
    std::uint32_t expected_backend_generation = 0u;

    float quiet_halo_cells = 0.0f;
    std::uint32_t analysis_canvas_size = 0u;
    float max_model_aspect = 0.0f;
    std::uint32_t active_rules = 0u;
  };

  struct sbs_roi_shape_request_gpu_result {
    /** Newest valid completed sample retained by this helper, if any. */
    std::optional<sbs_roi_shape_request> request;
    /** Monotonic helper-local sequence associated with request; zero means no completion. */
    std::uint64_t completed_sequence = 0u;
    /** Source-frame provenance supplied with the dispatch that produced request. */
    std::uint64_t completed_source_frame_id = 0u;
    /** True only when this call promoted a newly completed request. */
    bool fresh_sample = false;
    /** True only when this call enqueued dispatch, staging copy, and completion event work. */
    bool copy_scheduled = false;
    /** Infrastructure, resource-contract, query, map, or output-validation failure. */
    bool failed = false;
  };

  /**
   * Three-slot, nonblocking GPU request dispatcher.
   *
   * All methods must run on the owner of the supplied immediate context. poll() uses
   * GetData(DONOTFLUSH) and maps only signaled staging buffers with DO_NOT_WAIT; it never flushes,
   * waits, or maps an in-flight resource.
   */
  class sbs_roi_shape_request_gpu {
  public:
    sbs_roi_shape_request_gpu(
      Microsoft::WRL::ComPtr<ID3D11Device> device,
      Microsoft::WRL::ComPtr<ID3D11DeviceContext> context,
      const std::filesystem::path &assets_dir
    );
    ~sbs_roi_shape_request_gpu();

    sbs_roi_shape_request_gpu(
      const sbs_roi_shape_request_gpu &
    ) = delete;
    sbs_roi_shape_request_gpu &operator=(
      const sbs_roi_shape_request_gpu &
    ) = delete;

    [[nodiscard]] bool valid() const;

    /**
     * Resolve any ready older slots, then schedule one request into a free ring slot.
     *
     * Ring saturation is normal backpressure: copy_scheduled remains false without setting
     * failed. A non-null malformed/foreign rule-state view is replaced by the zero fallback and
     * reported as failed, while a deliberately null view is a supported fallback input.
     */
    [[nodiscard]] sbs_roi_shape_request_gpu_result submit(
      const sbs_roi_shape_request_gpu_submission &submission
    );

    /** Resolve ready slots without scheduling new GPU work. */
    [[nodiscard]] sbs_roi_shape_request_gpu_result poll();

  private:
    struct impl;
    std::unique_ptr<impl> pimpl_;
  };
}  // namespace models
