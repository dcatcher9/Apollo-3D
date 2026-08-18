/**
 * @file src/platform/windows/sbs_debug_dump.h
 * @brief Debug-only: atomically publish one authenticated production-V2 Host-SBS frame.
 */
#pragma once

// platform includes
#include <d3d11.h>

// standard includes
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// local includes
#include "src/config.h"
#include "src/video_depth_estimator.h"
#include "sbs_debug_dump_border.h"

namespace models {
  enum class input_color_space : std::uint32_t;
  struct parallax_v2_shader_provenance_t;
  struct host_sbs_gpu_trace_provenance_t;
  struct raw_model_provenance_t;
}

namespace platf::sbs_debug {

  struct frame;

  namespace detail {
    class publication_state;
    struct diagnostic_roi_crop;
    struct pending_gpu_capture;

    /** Maximum staging payload copied by one render-thread polling turn. */
    inline constexpr std::size_t cpu_collection_byte_budget = 64u * 1024u * 1024u;

    /**
     * Return one aligned copy chunk. An otherwise empty turn may overshoot by one alignment unit
     * so a texture row wider than the nominal budget cannot stall forever.
     */
    std::size_t bounded_collection_chunk_bytes(
      std::size_t remaining_bytes,
      std::size_t alignment,
      std::size_t available_budget,
      bool poll_is_empty
    ) noexcept;

    /** Pure exact-frame OCR8/SLR13 validation used before diagnostic publication. */
    bool subtitle_records_match_frame(
      const std::vector<std::uint8_t> &ocr,
      const std::vector<std::uint8_t> &locator,
      const frame &completed,
      std::uint32_t confirmed_cut_count
    );

    /** Accept exact-current evidence (including cadence-due reuse), or an authenticated held tuple. */
    bool subtitle_records_match_completion(
      const std::vector<std::uint8_t> &ocr,
      const std::vector<std::uint8_t> &locator,
      const std::vector<std::uint8_t> &ring,
      const frame &completed,
      std::uint32_t confirmed_cut_count
    );

    bool subtitle_ocr_record_is_canonical_for_frame(
      const std::vector<std::uint8_t> &ocr,
      const frame &completed
    );

    /** Validate one optional diagnostic completion ring and require the dump's matched root. */
    bool gpu_trace_ring_is_canonical(
      const std::vector<std::uint8_t> &ring,
      const frame &completed
    );

    /** Serialize the exact GPU-trace wire contract used by Dump 3D diagnostics. */
    std::string gpu_trace_contract_json(
      const models::host_sbs_gpu_trace_provenance_t &provenance
    );

    /** Decode one already-authenticated GPU trace with the current diagnostic semantics. */
    std::string gpu_trace_decoded_json(
      const std::vector<std::uint8_t> &ring,
      const frame &completed
    );
  }

  /**
   * @brief One exact, completed Host-SBS frame and all optional diagnostic render passes that
   *        belong to it.
   *
   * model_input and raw_depth are immutable estimator snapshots. warp_depth is a non-owning alias
   * used only to prove that production V2 consumed shadow_final_parallax itself; it is never
   * staged or serialized separately. That single authenticated final artifact is full-source-U in
   * ordinary mode and ROI-local-U in window-region mode. ROI renderer authority is the pair of
   * that ROI-local field and depth_input_region's scale/outside-collar embedding. The
   * adaptive_state/depth_frame_state pair is optional comparison-only evidence from the retained
   * scene-cut bridge; it never authorizes a dump or controls live geometry. The immutable V2
   * candidate first produces shadow_ownership_refined_parallax from the full-resolution source
   * contour, then shadow_vertical_majorant (the exact upper-envelope diagnostic) and
   * shadow_vertical_conditioned (the fixed 75/25 vertical share), then the row majorant produces
   * shadow_base_final_parallax. When OCR8 and SLR13 are active, the compact publication record and
   * locator state condition that Base into shadow_final_parallax; an empty current-authority block
   * copies Base bit for bit. That complete atomic field is sampled directly by the renderer. In
   * ROI mode it remains crop-local and is not by itself a full-source position field.
   * shadow_coordinate is allocated and written only for this explicit
   * dump; it is never a live resource. V2 supplies an exact fixed-point inverse warp_map when its
   * matching dump-only
   * shader is available. V2 has no internal owner/fill mask; warp_mask attributes only inverse
   * samples outside the finite source interval that the live renderer clamps to the nearest
   * boundary column.
   */
  struct frame {
    ID3D11ShaderResourceView *source = nullptr;
    /** Exact full-frame or whole-ROI color texture submitted to DAV2 and ownership. */
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
    ID3D11ShaderResourceView *shadow_base_final_parallax = nullptr;
    ID3D11ShaderResourceView *shadow_final_parallax = nullptr;
    ID3D11ShaderResourceView *ocr_box_record = nullptr;
    ID3D11ShaderResourceView *subtitle_locator_state = nullptr;
    ID3D11ShaderResourceView *gpu_trace_ring = nullptr;
    ID3D11ShaderResourceView *shadow_state = nullptr;
    ID3D11ShaderResourceView *shadow_frame_stats = nullptr;
    std::shared_ptr<const models::raw_model_provenance_t> raw_model_provenance;
    std::shared_ptr<const models::parallax_v2_shader_provenance_t>
      parallax_v2_shader_provenance;
    std::shared_ptr<const models::host_sbs_gpu_trace_provenance_t>
      gpu_trace_provenance;
    int model_width = 0;
    int model_height = 0;
    int raw_width = 0;
    int raw_height = 0;
    std::uint64_t matched_frame_id = 0;
    /** Exact authenticated analysis domain bound to every model/depth/parallax artifact. */
    models::depth_input_region_t depth_input_region {};
    /** ROI planner result, present exactly when depth_input_region.is_video_region() is true. */
    std::optional<models::depth_video_region_plan_t> depth_video_plan;
    /** True when this completion was the first frame after its analysis domain was rearmed. */
    bool input_domain_reset = false;
    /** True when the completed depth branch remains device-owned and needs the trace for proof. */
    bool gpu_undecided_completion = false;
    /** True when native interaction published Base as target without advancing OCR8/SLR13. */
    bool subtitle_work_suppressed = false;
    /** Optional matched-window provenance stamped onto matched_frame_id. */
    std::optional<window_region_snapshot> window_region;
    std::string window_region_observer_status = "not-observed";
    std::string window_region_mapping_status = "not-mapped";
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

    /**
     * @brief Keep retained-source conversion alive for a newly latched button request or until a
     *        submitted event-query batch is collected.
     *
     * Publication remains single-flight: a later click waits without reconverting while the CPU
     * worker owns the preceding package, then wakes the retained source when that worker is idle.
     */
    bool needs_conversion_poll() const noexcept;

    /**
     * @brief Poll a previously submitted staging batch without flushing the D3D11 queue.
     *
     * The owning render thread calls this once per Host-SBS conversion. A ready batch is mapped
     * into bounded CPU-owned chunks over later conversions before the publication worker is
     * queued. Every successful map is unmapped before this call returns.
     */
    void poll_pending_readback(ID3D11DeviceContext *ctx) noexcept;

    /** Arm dump-only geometry for the requested matched frame.
     *
     * Camera validity is checked from the same staged shadow-state bytes as the rest of the
     * package. This preparation step deliberately performs no GPU readback or synchronization.
     */
    bool prepare_requested_v2_frame(std::uint64_t matched_frame_id) noexcept;

    /**
     * @brief Copy one authenticated completed ROI into dumper-owned diagnostic storage.
     *
     * The caller retains completion/authentication ownership and invokes this only after matching
     * the completed estimator result to its retained source slot. The returned SRV remains valid
     * until release_diagnostic_roi_crop(), rejection, cancellation, or the next snapshot request.
     */
    ID3D11ShaderResourceView *prepare_diagnostic_roi_crop(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11Texture2D *source,
      const models::depth_input_region_t &region
    ) noexcept;

    /** Release the dump-only ROI source after its staging copy has been queued. */
    void release_diagnostic_roi_crop() noexcept;

    /** Cancel a request that can never complete, such as after permanent estimator failure. */
    void cancel_pending_request() noexcept;

    /** Consume one impossible request without disabling future file-trigger polling. */
    void reject_pending_request() noexcept;

    /**
     * @brief Submit an immutable same-frame staging batch for background publication.
     *
     * An incomplete matched set or invalid completion defers the trigger. Writer failures retain
     * it with bounded retry cadence; no partial folder is exposed as a completed dump. GPU copies
     * and nonblocking query polling remain on the owning render thread so the resources form one
     * stable frame. PNG generation, hashing, JSON serialization, and filesystem I/O run on the
     * process-lifetime publication worker. Cheap no-op otherwise. Call once per Host-SBS convert().
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
    std::unique_ptr<detail::pending_gpu_capture> pending_gpu_capture_;
    std::unique_ptr<detail::diagnostic_roi_crop> diagnostic_roi_crop_;
    std::shared_ptr<std::atomic<bool>> button_request_;
    bool file_trigger_enabled_ = false;
    mutable bool file_trigger_pending_ = false;
    mutable std::chrono::steady_clock::time_point next_file_trigger_poll_ {};
    std::chrono::steady_clock::time_point retry_not_before_ {};
    bool snapshot_armed_for_dump_ = false;
    std::uint64_t prepared_frame_id_ = 0;
  };

}  // namespace platf::sbs_debug
