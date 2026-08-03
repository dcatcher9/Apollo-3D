/**
 * @file src/platform/windows/sbs_debug_dump.h
 * @brief Debug-only: atomically publish one authenticated production-V2 Host-SBS frame.
 */
#pragma once

// platform includes
#include <d3d11.h>

// standard includes
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

// local includes
#include "src/config.h"

namespace models {
  enum class input_color_space : std::uint32_t;
  struct parallax_v2_shader_provenance_t;
  struct raw_model_provenance_t;
}

namespace platf::sbs_debug {

  /**
   * @brief One exact, completed Host-SBS frame and all optional diagnostic render passes that
   *        belong to it.
   *
   * model_input and raw_depth are immutable estimator snapshots. warp_depth is the exact signed,
   * anisotropically slope-limited final source-U parallax consumed by production V2. The
   * adaptive_state/depth_frame_state pair is optional comparison-only evidence from the retained
   * scene-cut bridge; it never authorizes a dump or controls live geometry. The immutable V2
   * candidate first produces shadow_vertical_majorant (the exact shear-2 column majorant), then
   * the row majorant produces shadow_final_parallax. shadow_coordinate is allocated and written
   * only for this explicit dump; it is never a live resource. V2 supplies an exact fixed-point
   * inverse warp_map when its matching dump-only
   * shader is available. V2 has no internal owner/fill mask; warp_mask attributes only inverse
   * samples outside the finite source interval that the live renderer clamps to the nearest
   * boundary column.
   */
  struct frame {
    ID3D11ShaderResourceView *source = nullptr;
    ID3D11ShaderResourceView *model_input = nullptr;
    ID3D11ShaderResourceView *raw_depth = nullptr;
    ID3D11ShaderResourceView *warp_depth = nullptr;
    ID3D11ShaderResourceView *adaptive_state = nullptr;
    ID3D11ShaderResourceView *depth_frame_state = nullptr;
    ID3D11ShaderResourceView *warp_map = nullptr;
    ID3D11ShaderResourceView *warp_mask = nullptr;
    ID3D11ShaderResourceView *sbs = nullptr;
    ID3D11ShaderResourceView *shadow_coordinate = nullptr;
    ID3D11ShaderResourceView *shadow_candidate_parallax = nullptr;
    ID3D11ShaderResourceView *shadow_vertical_majorant = nullptr;
    ID3D11ShaderResourceView *shadow_final_parallax = nullptr;
    ID3D11ShaderResourceView *shadow_state = nullptr;
    ID3D11ShaderResourceView *shadow_frame_stats = nullptr;
    std::shared_ptr<const models::raw_model_provenance_t> raw_model_provenance;
    std::shared_ptr<const models::parallax_v2_shader_provenance_t>
      parallax_v2_shader_provenance;
    int model_width = 0;
    int model_height = 0;
    int raw_width = 0;
    int raw_height = 0;
    std::uint64_t matched_frame_id = 0;
    bool cuda_graph_active = false;
    bool parallax_v2_producer_active = false;
    bool parallax_v2_render_selected = false;
    std::string parallax_v2_live_renderer_source_closure_sha256;
    float parallax_v2_raw_coordinate_scale = 0.0f;
    float parallax_v2_requested_pop_strength = 0.0f;
    float parallax_v2_requested_gain = 0.0f;
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

    /** Require a current valid authenticated V2 camera before publishing a live-render dump.
     * Invalid V2 completions hold the prior packed SBS via pixel-shader discard, so pairing their
     * current source/raw field with that prior output would create a false package.
     */
    bool preflight_requested_v2_frame(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *shadow_state,
      std::uint64_t matched_frame_id
    ) noexcept;

    /** Cancel a request that can never complete, such as after permanent estimator failure. */
    void cancel_pending_request() noexcept;

    /** Consume one impossible request without disabling future file-trigger polling. */
    void reject_pending_request() noexcept;

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
    std::uint64_t prepared_frame_id_ = 0;
  };

}  // namespace platf::sbs_debug
