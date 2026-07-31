/**
 * @file src/sbs_scene_controller_gpu.h
 * @brief GPU-only deterministic scene-controller backend for Host SBS.
 */
#pragma once

#include "config.h"
#include "video_depth_estimator.h"

#include <cstdint>
#include <d3d11.h>
#include <filesystem>
#include <memory>
#include <wrl/client.h>

namespace models {
  /**
   * GPU-resident snapshot produced for one completed, matched depth frame.
   *
   * The views are diagnostic while the backend is in shadow mode. No live render decision may
   * depend on CPU readback of these resources.
   */
  struct scene_controller_gpu_snapshot {
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> scene_rgb;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> analysis_grid;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> dense_output;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> global_output;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> layout_history;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> depth_history;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> hidden_output;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> meta;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> rule_state;
    // Private reducer scratch. Offline diagnostics may map this snapshot; live rendering never
    // consumes it on the CPU.
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> rule_summary;
#ifdef SUNSHINE_TESTS
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>
      rule_evidence_for_testing;
#endif
    std::uint64_t source_frame_id = 0;
    std::uint32_t backend_generation = 0;
    bool snapshot_available = false;
    bool shadow = false;
  };

  /**
   * The production rule backend never maps or reads its state on the encode thread. It consumes
   * the exact scene buffer retained for a submitted DA-V2 frame, then writes evidence, histories,
   * and resolver state through D3D11 compute passes when that same inference completes.
   */
  class sbs_scene_controller_gpu {
  public:
    sbs_scene_controller_gpu(
      Microsoft::WRL::ComPtr<ID3D11Device> device,
      Microsoft::WRL::ComPtr<ID3D11DeviceContext> context,
      const std::filesystem::path &assets_dir,
      config::sbs_scene_controller_e backend,
      const config::video_t::sbs_t &sbs_config
#ifdef SUNSHINE_TESTS
      ,
      bool use_serial_reducer_reference_for_testing = false,
      bool enable_isolated_timing_for_testing = false
#endif
    );
    ~sbs_scene_controller_gpu();

    sbs_scene_controller_gpu(const sbs_scene_controller_gpu &) = delete;
    sbs_scene_controller_gpu &operator=(const sbs_scene_controller_gpu &) = delete;

    [[nodiscard]] bool enabled() const;
    [[nodiscard]] bool valid() const;

    /**
     * Build the full-frame scene tensors for a frame that is about to be submitted to DA-V2.
     * This is GPU work only. Call mark_enqueued() only if the matching TensorRT enqueue succeeds.
     */
    bool prepare_scene(
      ID3D11ShaderResourceView *input,
      input_color_space color_space,
      std::uint64_t source_frame_id
    );

    /** Commit or discard the prepared scene ownership after the DA-V2 enqueue result is known. */
    void mark_enqueued(std::uint64_t source_frame_id);
    void discard_prepared(std::uint64_t source_frame_id);

    /**
     * Consume the completed matched depth result. Must run before the legacy appearance/depth
     * histories advance, while the retained scene and depth inputs still belong to
     * source_frame_id. frame_roi_transform must be the exact GPU transform retained with that
     * completed inference; the explicit all-zero eight-vector resource selects legacy full-frame
     * sampling.
     */
    bool resolve_completed(
      std::uint64_t source_frame_id,
      ID3D11ShaderResourceView *raw_depth,
      ID3D11ShaderResourceView *normalized_depth,
      ID3D11ShaderResourceView *depth_frame_state,
      ID3D11ShaderResourceView *adaptive_state,
      int depth_width,
      int depth_height,
      ID3D11ShaderResourceView *frame_roi_transform
    );

    [[nodiscard]] scene_controller_gpu_snapshot snapshot() const;

    /**
     * Override the wall-clock delta consumed by the next prepared scene.
     *
     * The offline whole-clip evaluator uses exact stream-presentation deltas so
     * dwell and release decisions do not depend on workstation load. This is a
     * one-shot diagnostic input: production never calls it and therefore keeps
     * using elapsed time between accepted live stream submissions. Embedded
     * video cadence is never an input to the controller.
     */
    bool set_next_elapsed_seconds_for_evaluation(float elapsed_seconds);

#ifdef SUNSHINE_TESTS
    void set_next_reset_flags_for_testing(std::uint32_t reset_flags);
    void set_next_elapsed_seconds_for_testing(float elapsed_seconds);
#endif

  private:
    struct impl;
    std::unique_ptr<impl> pimpl_;
  };
}  // namespace models
