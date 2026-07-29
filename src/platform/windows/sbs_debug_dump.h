/**
 * @file src/platform/windows/sbs_debug_dump.h
 * @brief Debug-only: atomically publish a matched Host-SBS package spanning model input, raw and
 *        processed depth, adaptive state, exact warp mapping/coverage, and packed output.
 */
#pragma once

// platform includes
#include <d3d11.h>

// standard includes
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

// local includes
#include "src/config.h"

namespace models {
  enum class input_color_space : std::uint32_t;
}

namespace platf::sbs_debug {

  /**
   * @brief One exact, completed Host-SBS frame and all optional diagnostic render passes that
   *        belong to it.
   *
   * model_input and raw_depth are immutable snapshots of the estimator buffers. The caller must
   * pass the normalized depth and the actual (possibly prefiltered) depth used by reprojection.
   * warp_map/warp_mask may be null only when the dump-only diagnostic shaders could not be made.
   */
  struct frame {
    ID3D11ShaderResourceView *source = nullptr;
    ID3D11ShaderResourceView *model_input = nullptr;
    ID3D11ShaderResourceView *raw_depth = nullptr;
    ID3D11ShaderResourceView *depth = nullptr;
    ID3D11ShaderResourceView *warp_depth = nullptr;
    ID3D11ShaderResourceView *adaptive_state = nullptr;
    ID3D11ShaderResourceView *depth_frame_state = nullptr;
    // Exact eight-uint4 transform sampled by preprocessing, normalization, scene evidence, and
    // reprojection for matched_frame_id. The dump serializes both raw words and decoded geometry.
    ID3D11ShaderResourceView *depth_roi_transform = nullptr;
    // Optional, exact-frame GPU Scene Controller ABI v1 tensors. These are populated only when
    // the controller reports a completed output for matched_frame_id. They must never become a
    // prerequisite for the core Host-SBS dump because the controller defaults to off.
    ID3D11ShaderResourceView *scene_controller_scene_rgb = nullptr;
    ID3D11ShaderResourceView *scene_controller_analysis_grid = nullptr;
    ID3D11ShaderResourceView *scene_controller_dense_output = nullptr;
    ID3D11ShaderResourceView *scene_controller_global_output = nullptr;
    ID3D11ShaderResourceView *scene_controller_layout_history = nullptr;
    ID3D11ShaderResourceView *scene_controller_depth_history = nullptr;
    ID3D11ShaderResourceView *scene_controller_hidden_output = nullptr;
    ID3D11ShaderResourceView *scene_controller_meta = nullptr;
    ID3D11ShaderResourceView *scene_controller_rule_state = nullptr;
    ID3D11ShaderResourceView *warp_map = nullptr;
    ID3D11ShaderResourceView *warp_mask = nullptr;
    ID3D11ShaderResourceView *sbs = nullptr;
    int model_width = 0;
    int model_height = 0;
    int raw_width = 0;
    int raw_height = 0;
    std::uint64_t matched_frame_id = 0;
    std::uint64_t scene_controller_frame_id = 0;
    std::uint32_t scene_controller_backend_generation = 0;
    bool scene_controller_snapshot_available = false;
    bool scene_controller_shadow = false;
    bool warp_depth_prefilter_applied = false;
    bool cuda_graph_active = false;
    models::input_color_space color_space {};
    std::string depth_model;
  };

  /**
   * @brief Owns the dump destination and performs one-frame package dumps when a trigger fires.
   *
   * Triggers: the encoder/session request latch installed with set_button_request(), or, in
   * diagnostic mode with APOLLO_SBS_DUMP set, a "dump.trigger" file in that directory.
   * Button-triggered output falls back to an "sbs_dump" folder next to the sunshine log.
   */
  class dumper {
  public:
    dumper();  ///< Resolves the output directory (APOLLO_SBS_DUMP, else <log dir>/sbs_dump).

    /**
     * @brief Attach the request latch owned by this encoder/session.
     *
     * A null latch disables remote-button requests without falling back to process-global state.
     * The shared ownership prevents a late render callback from observing a destroyed session.
     */
    void set_button_request(std::shared_ptr<std::atomic<bool>> request);

    /**
     * @brief True when the button or diagnostic file trigger needs the next completed frame.
     *
     * The encoder calls this before depth inference so the estimator can preserve the completed
     * raw tensor before immediately reusing its CUDA/D3D buffer for the next frame.
     */
    bool snapshot_requested();

    /**
     * @brief Validate and cache the requested frame's 16-byte normalization state.
     *
     * Call this only after the estimator has produced the complete stable snapshot set and before
     * launching the full-resolution dump-only mapping/coverage passes. Invalid or temporarily
     * unreadable state retains the request with a bounded retry delay.
     */
    bool preflight_requested_frame(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *depth_frame_state,
      std::uint64_t matched_frame_id
    ) noexcept;

    /** Cancel a request that can never complete, such as after permanent estimator failure. */
    void cancel_pending_request() noexcept;

    /**
     * @brief Atomically publish a fresh timestamped package for the supplied completed frame.
     *
     * An incomplete matched set or invalid completion defers the trigger. Writer failures retain
     * it with bounded retry cadence; no partial folder is exposed as a completed dump. Cheap no-op
     * otherwise. Call once per Host-SBS convert().
     */
    bool maybe_dump(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      const frame &completed,
      const config::video_t::sbs_t &cfg
    );

  private:
    std::filesystem::path dir_;
    std::shared_ptr<std::atomic<bool>> button_request_;
    bool file_trigger_enabled_ = false;
    bool file_trigger_pending_ = false;
    unsigned poll_counter_ = 0;  ///< Rate-limits the dump.trigger file stat to ~1/s.
    unsigned retry_backoff_frames_ = 0;
    bool snapshot_armed_for_dump_ = false;
    bool prepared_normalization_valid_ = false;
    std::uint64_t prepared_frame_id_ = 0;
    std::array<float, 4> prepared_normalization_ {};
  };

}  // namespace platf::sbs_debug
