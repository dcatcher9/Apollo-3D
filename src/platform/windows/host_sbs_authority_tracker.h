/**
 * @file src/platform/windows/host_sbs_authority_tracker.h
 * @brief What live Host 3D converts and where: the foreground window, the per-application
 * subtitle context, the browser video lease and each matched frame's analysis region.
 *
 * The tracker owns that authority state and decides from it. It holds no reference into the
 * depth pipeline: an observation reports its consequences in effects_t, and the converter
 * applies them to its own completions, caches and reprocess requests in the listed order.
 */
#pragma once

// standard includes
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

// platform includes
#include <d3d11.h>
#include <dxgi.h>

// local includes
#include "foreground_window_region.h"
#include "host_sbs_converter_test_observer.h"
#include "sbs_debug_dump_border.h"
#include "subtitle_context_policy.h"
#include "video_dom_client.h"
#include "src/host_subtitle_context.h"
#include "src/video_depth_estimator.h"

namespace platf::video_player {
  class registry_t;
}

namespace platf::dxgi {
  class display_base_t;
  class text_game_pin_controller_t;

  /** The authority fields a matched frame carries. The tracker decides them; the converter stores them. */
  struct host_sbs_region_attribution_t {
    std::optional<sbs_debug::window_region_snapshot> window_region;
    std::optional<models::depth_video_region_plan_t> depth_video_plan;
    models::depth_input_region_t depth_input_region {};
    std::string_view window_region_observer_status = "not-observed";
    std::string_view window_region_mapping_status = "not-mapped";
    std::uint64_t live_window_authority_generation = 0u;
    std::uint64_t live_browser_authority_generation = 0u;
    std::int32_t live_browser_screen_left = 0;
    std::int32_t live_browser_screen_top = 0;
    std::int32_t live_browser_screen_right = 0;
    std::int32_t live_browser_screen_bottom = 0;
    std::chrono::steady_clock::time_point live_browser_geometry_valid_since {};
    video_dom::geometry_authority_class_e live_browser_authority_class =
      video_dom::geometry_authority_class_e::none;
    models::host_subtitle_context::presentation_authority_t observed_authority;
    std::uint64_t subtitle_context_generation = 0u;
    bool subtitle_context_eligible = false;
    // Semantic retention during a freshly verified modal loop permits no positioned glyph
    // authority. Keep it separate from the normal per-frame eligibility used by rendering.
    bool subtitle_context_suppression_eligible = false;
    // Submitted with the same context generation, so it cannot change within one context.
    models::subtitle_line_policy_e subtitle_line_policy = models::subtitle_line_policy_e::video_caption;
    std::uint64_t test_instance = 0u;
  };

  class host_sbs_authority_tracker_t {
  public:
    struct browser_live_authority_key_t {
      video_dom::status_e status {video_dom::status_e::starting};
      std::uint64_t window = 0u;
      std::uint32_t process_id = 0u;
      std::int32_t document_id = 0;
      std::int32_t video_id = 0;
      video_dom::rect_t screen_rect {};
      std::chrono::steady_clock::time_point geometry_valid_since {};

      bool operator==(const browser_live_authority_key_t &) const = default;
    };

    /** Depth-pipeline consequences of one observation, applied by the converter in this order. */
    struct effects_t {
      /** Ask the retained-source owner for another conversion so full-source depth starts. */
      bool depth_reprocess = false;
      /** The foreground root or region changed: positioned packed output leaves with it. */
      bool window_changed = false;
      /** The browser authority changed: Chromium-video or positioned packed output leaves with it. */
      bool browser_changed = false;
      /** A new subtitle context: revoke completions, lineage and packed output, and reprocess. */
      bool subtitle_context_changed = false;
    };

    /** What the converter lends one observation; nothing here is retained. */
    struct inputs_t {
      display_base_t *display = nullptr;
      const video_player::registry_t *registry = nullptr;
      /** Present only while a Game 3D text game is converted by Host AI. */
      const text_game_pin_controller_t *text_game = nullptr;
      /** The estimator's pending root is an ROI completion. */
      bool pending_roi_completion = false;
    };

    struct observation_t {
      foreground_window::snapshot_t foreground;
      std::optional<foreground_window::capture_target_t> capture_target;
      foreground_window::mapping_result_t foreground_projection {
        .status = foreground_window::mapping_status_e::invalid_capture_target,
      };
      video_dom::snapshot_ptr browser_snapshot;
      std::optional<video_dom::mapping_result_t> browser_projection;
      std::optional<browser_live_authority_key_t> browser_authority;
      std::uint64_t browser_authority_epoch = 0u;
      effects_t effects;
    };

    /** The labelled converter fixture's synthetic scheduling domain; null in production. */
    void set_test_context(const converter_test_observer::context_t *context) noexcept {
      test_context = context;
    }

    /** Start or stop the supervised browser helper whose snapshots this tracker reads. */
    void enable_browser_helper(bool enabled);

    /** Forget every observation, for a new stream. The browser helper lease and the
     * foreground process image cache are kept. */
    void reset();

    /** Sample the foreground window and the browser helper, then update the subtitle context. */
    observation_t observe(
      const D3D11_TEXTURE2D_DESC &source_desc,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
      const inputs_t &inputs,
      const foreground_window::snapshot_t *revalidate_expected = nullptr,
      bool sampling_for_attribution = false
    );

    // ---- Decisions for one matched frame ----

    std::optional<sbs_debug::window_region_snapshot> capture_browser_window_region(
      const D3D11_TEXTURE2D_DESC &source_desc,
      display_base_t *display,
      const observation_t &authority,
      std::uint64_t frame_id,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
      std::chrono::steady_clock::time_point captured_at,
      video_dom::status_e &video_status,
      video_dom::mapping_status_e &mapping_status
    ) const;

    std::optional<sbs_debug::window_region_snapshot> capture_foreground_window_region(
      const D3D11_TEXTURE2D_DESC &source_desc,
      display_base_t *display,
      std::uint64_t frame_id,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
      std::chrono::steady_clock::time_point captured_at,
      const observation_t &authority,
      std::string_view &observer_status,
      std::string_view &mapping_status
    ) const;

    /** The analysis region (ROI crop) for the slot's window region, or the full source. */
    void select_depth_input_region(
      host_sbs_region_attribution_t &slot,
      const D3D11_TEXTURE2D_DESC &source_desc,
      bool finalize_full_source = true
    );

    /** Attach the live window/browser generations and the browser rectangle to a captured region. */
    void attribute_window_generations(host_sbs_region_attribution_t &slot, const observation_t &authority) const;

    /** Attach the subtitle context generation, eligibility and line policy. */
    void attribute_subtitles(host_sbs_region_attribution_t &slot, const D3D11_TEXTURE2D_DESC &source_desc) const;

    /** A same-size programmatic move whose captured content predates the live region. */
    [[nodiscard]] bool requires_foreground_causal_wait(
      const foreground_window::snapshot_t &foreground,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp
    ) const noexcept;

    [[nodiscard]] bool window_region_matches_live_authority(const sbs_debug::window_region_snapshot &region) const noexcept;

    [[nodiscard]] bool window_region_authorized_for_render(const host_sbs_region_attribution_t &slot) const noexcept;

    /** Log a window-region transition once per change of the attributed region or its status. */
    void publish_window_region_transition(const host_sbs_region_attribution_t &slot, std::uint64_t frame_id);

    // ---- Live state ----

    [[nodiscard]] models::host_subtitle_context::presentation_authority_t presentation_authority() const noexcept;
    [[nodiscard]] models::host_subtitle_context::presentation_authority_t presentation_authority(
      std::uint64_t browser_authority_epoch
    ) const noexcept;

    [[nodiscard]] const models::host_subtitle_context::tracker_t &subtitle_context() const noexcept {
      return subtitle_context_;
    }

    /** A route switch retires work submitted under the previous route's subtitle context. */
    void revoke_subtitle_context() {
      subtitle_context_.revoke();
    }

    [[nodiscard]] bool registry_ready_observed() const noexcept {
      return video_player_registry_ready_observed;
    }

#ifdef SUNSHINE_TESTS
    void set_registry_ready_observed_for_test(bool observed) noexcept {
      video_player_registry_ready_observed = observed;
    }
#endif

  private:
    struct subtitle_admission_diagnostic_key_t {
      std::uint64_t context_generation = 0u;
      std::string_view reason;
      foreground_window::status_e foreground_status {};
      foreground_window::mapping_status_e foreground_mapping_status {};
      video_dom::status_e browser_status {};
      video_dom::mapping_status_e browser_mapping_status {};
      std::uint64_t foreground_window = 0u;
      std::uint32_t foreground_process = 0u;
      std::uint64_t browser_window = 0u;
      std::uint32_t browser_process = 0u;
      std::int32_t browser_document = 0;
      std::int32_t browser_video = 0;
      bool foreground_root = false;
      bool foreground_region = false;
      bool foreground_target_usable = false;
      bool foreground_usable = false;
      bool content_timestamp_available = false;
      bool browser_snapshot = false;
      bool browser_heartbeat_fresh = false;
      bool browser_after_move = false;
      bool browser_authority = false;
      bool browser_identity_matches = false;
      bool browser_usable = false;
      bool mapping_attempted = false;
      bool mapping_succeeded = false;
      bool fullscreen_checked = false;
      bool fullscreen_match = false;
      bool registered_player_candidate = false;
      bool registry_ready = false;
      bool retained_for_interactive_move_size = false;

      bool operator==(const subtitle_admission_diagnostic_key_t &) const = default;
    };

    void observe_live_browser_authority(const video_dom::snapshot_ptr &browser, effects_t &effects);

    subtitle_context_policy::evaluation_t observe_subtitle_context(
      const foreground_window::snapshot_t &foreground,
      const video_dom::snapshot_ptr &browser,
      const DXGI_OUTPUT_DESC &output,
      const D3D11_TEXTURE2D_DESC &source,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
      const foreground_window::mapping_result_t &foreground_mapping,
      const inputs_t &inputs,
      effects_t &effects
    );

    const std::optional<std::wstring> &subtitle_process_image_path(
      const foreground_window::snapshot_t &foreground,
      std::chrono::steady_clock::time_point now
    );

    std::uint64_t depth_analysis_generation(const models::depth_analysis_domain_key_t &key) noexcept {
      return depth_analysis_generation_tracker.select(key);
    }

    const converter_test_observer::context_t *test_context = nullptr;
    std::optional<video_dom::lease_t> video_dom_lease;
    bool video_player_registry_ready_observed = false;
    /// See subtitle_process_image_path().
    struct {
      std::uintptr_t window = 0;
      std::uint32_t process_id = 0;
      std::chrono::steady_clock::time_point queried_at {};
      std::optional<std::wstring> path;
    } subtitle_process_image;
    models::host_subtitle_context::tracker_t subtitle_context_;
    std::optional<subtitle_admission_diagnostic_key_t> last_subtitle_admission_diagnostic;
    std::chrono::steady_clock::time_point last_subtitle_admission_diagnostic_at {};
    models::depth_analysis_generation_tracker_t depth_analysis_generation_tracker;
    foreground_window::continuity_tracker_t foreground_window_tracker;
    std::optional<foreground_window::snapshot_t> live_window_authority;
    std::optional<foreground_window::snapshot_t> live_foreground_region;
    std::optional<browser_live_authority_key_t> live_browser_authority;
    std::uint64_t live_browser_authority_epoch = 0u;
    bool interactive_move_size_observed = false;
    std::chrono::steady_clock::time_point browser_roi_not_before {};
    std::optional<sbs_debug::window_region_snapshot> last_window_region;
    std::optional<std::string_view> last_window_region_observer_status;
    std::optional<std::string_view> last_window_region_mapping_status;
  };

}  // namespace platf::dxgi
