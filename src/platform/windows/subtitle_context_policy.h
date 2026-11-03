/**
 * @file src/platform/windows/subtitle_context_policy.h
 * @brief Pure subtitle admission from current foreground and browser geometry.
 */
#pragma once

#include "foreground_window_region.h"
#include "video_dom_client.h"
#include "src/host_subtitle_context.h"

namespace platf::subtitle_context_policy {
  // These facts also supply transition diagnostics; evaluation performs no provider I/O.
  struct evaluation_t {
    models::host_subtitle_context::key_t browser_key;
    bool foreground_usable = false;
    bool foreground_target_usable = false;
    bool browser_identity_matches = false;
    bool browser_usable = false;
    bool mapping_attempted = false;
    bool mapping_succeeded = false;
    video_dom::mapping_status_e mapping_status = video_dom::mapping_status_e::invalid_video_rect;
    bool fullscreen_check_evaluated = false;
    bool fullscreen_match = false;
    bool registered_player_candidate = false;
  };

  [[nodiscard]] inline evaluation_t evaluate(
    const foreground_window::snapshot_t *foreground,
    const bool foreground_region_present,
    const foreground_window::mapping_result_t &foreground_mapping,
    const video_dom::snapshot_t *browser,
    const std::uint64_t browser_authority_epoch,
    const video_dom::rect_t capture_desktop_rect,
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
    const std::chrono::steady_clock::time_point now
  ) noexcept {
    evaluation_t result;
    result.foreground_usable = foreground && foreground_window::usable_for_content(
      *foreground, content_timestamp, now, std::chrono::milliseconds {250});
    // Reuse the capture owner's single mapping result. Partial client containment is the only
    // relaxed prerequisite; orientation, extent, monitor identity and failed GetDesc stay closed.
    result.foreground_target_usable =
      foreground_mapping.status == foreground_window::mapping_status_e::ok ||
      foreground_mapping.status == foreground_window::mapping_status_e::partially_outside_capture;
    if (!result.foreground_usable || !result.foreground_target_usable) {
      return result;
    }

    result.browser_identity_matches = browser && browser_authority_epoch != 0u &&
      browser->window == foreground->window && browser->process_id == foreground->process_id;
    // A registered native player retains the whole-client ROI requirement. A browser instead
    // supplies its independently authenticated video rectangle, which may fit a partially visible
    // client. Failed browser evidence must not fall through to native-player classification.
    result.registered_player_candidate = foreground_region_present &&
      foreground_mapping.status == foreground_window::mapping_status_e::ok && !result.browser_identity_matches;
    if (!result.browser_identity_matches) {
      return result;
    }
    result.browser_usable = video_dom::detail::usable_for_content(
      *browser, content_timestamp, now, std::chrono::milliseconds {2500});
    if (!result.browser_usable) {
      return result;
    }

    const auto mapped = video_dom::map_screen_rect_to_capture(
      browser->screen_rect, capture_desktop_rect, source_width, source_height,
      video_dom::rotation_e::identity);
    result.mapping_attempted = true;
    result.mapping_status = mapped.status;
    result.mapping_succeeded = static_cast<bool>(mapped);
    if (result.mapping_succeeded) {
      result.fullscreen_check_evaluated = true;
      result.fullscreen_match = browser->status != video_dom::status_e::ok_fullscreen ||
        (mapped.capture_pixels.left == 0 && mapped.capture_pixels.top == 0 &&
         mapped.capture_pixels.right == source_width && mapped.capture_pixels.bottom == source_height);
    }
    if (result.mapping_succeeded && result.fullscreen_match) {
      result.browser_key = {models::host_subtitle_context::kind_e::browser_video,
        foreground->window, foreground->process_id, foreground->generation, browser_authority_epoch};
    }
    return result;
  }
}  // namespace platf::subtitle_context_policy
