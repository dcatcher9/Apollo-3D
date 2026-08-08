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
#include <optional>
#include <string>

// local includes
#include "src/config.h"
#include "src/video_depth_estimator.h"
#include "sbs_debug_dump_border.h"

namespace models {
  enum class input_color_space : std::uint32_t;
  struct parallax_v2_shader_provenance_t;
  struct raw_model_provenance_t;
}

namespace platf::sbs_debug {

  namespace detail {
    class publication_state;
  }

  /**
   * @brief One exact, completed Host-SBS frame and all optional diagnostic render passes that
   *        belong to it.
   *
   * model_input and raw_depth are immutable estimator snapshots. warp_depth is the exact signed,
   * anisotropically slope-limited final parallax consumed by production V2: full-source-U in the
   * ordinary mode and ROI-local-U in video-region mode. ROI renderer authority is the pair of
   * that crop-local field and depth_input_region's scale/outside-collar embedding. The
   * adaptive_state/depth_frame_state pair is optional comparison-only evidence from the retained
   * scene-cut bridge; it never authorizes a dump or controls live geometry. The immutable V2
   * candidate first produces shadow_ownership_refined_parallax from the full-resolution source
   * contour, then shadow_vertical_majorant (the exact upper-envelope diagnostic) and
   * shadow_vertical_conditioned (the fixed 75/25 vertical share), then the row majorant produces
   * shadow_final_parallax. In ROI mode that final field remains crop-local and is not by itself a
   * full-source position field. shadow_coordinate is allocated and written only for this explicit
   * dump; it is never a live resource. V2 supplies an exact fixed-point inverse warp_map when its
   * matching dump-only
   * shader is available. V2 has no internal owner/fill mask; warp_mask attributes only inverse
   * samples outside the finite source interval that the live renderer clamps to the nearest
   * boundary column.
   */
  struct frame {
    ID3D11ShaderResourceView *source = nullptr;
    /** Exact full-frame or inward-cropped color texture submitted to DAV2 and ownership. */
    ID3D11ShaderResourceView *depth_input_source = nullptr;
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
    ID3D11ShaderResourceView *shadow_ownership_refined_parallax = nullptr;
    ID3D11ShaderResourceView *shadow_vertical_majorant = nullptr;
    ID3D11ShaderResourceView *shadow_vertical_conditioned = nullptr;
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
    /** Exact authenticated analysis domain bound to every model/depth/parallax artifact. */
    models::depth_input_region_t depth_input_region {};
    /** ROI planner result, present exactly when depth_input_region.video_region is true. */
    std::optional<models::depth_video_region_plan_t> depth_video_plan;
    /** True when this completion was the first frame after its analysis domain was rearmed. */
    bool input_domain_reset = false;
    /** Optional, diagnostic-only browser-video border stamped onto matched_frame_id. */
    std::optional<window_video_border_snapshot> window_video_border;
    std::string window_video_observer_status = "not-observed";
    std::string window_video_mapping_status = "not-mapped";
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
    ~dumper();

    dumper(const dumper &) = delete;
    dumper &operator=(const dumper &) = delete;

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
     * @brief Snapshot a completed frame and queue its package for background publication.
     *
     * An incomplete matched set or invalid completion defers the trigger. Writer failures retain
     * it with bounded retry cadence; no partial folder is exposed as a completed dump. GPU
     * readback remains on the owning render thread so the resources form one stable frame, while
     * PNG generation, hashing, JSON serialization, and filesystem I/O run on the process-lifetime
     * publication worker. Cheap no-op otherwise. Call once per Host-SBS convert().
     */
    bool maybe_dump(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      const frame &completed,
      const config::video_t::sbs_t &cfg
    );

  private:
    std::filesystem::path dir_;
    std::shared_ptr<detail::publication_state> async_;
    std::shared_ptr<std::atomic<bool>> button_request_;
    bool file_trigger_enabled_ = false;
    bool file_trigger_pending_ = false;
    unsigned poll_counter_ = 0;  ///< Rate-limits the dump.trigger file stat to ~1/s.
    unsigned retry_backoff_frames_ = 0;
    bool snapshot_armed_for_dump_ = false;
    std::uint64_t prepared_frame_id_ = 0;
  };

}  // namespace platf::sbs_debug
