/**
 * @file src/platform/windows/host_sbs_authority_tracker.cpp
 * @brief What live Host 3D converts and where, decided from the foreground window, the
 * subtitle context and the browser video lease.
 */
#include "host_sbs_authority_tracker.h"

// standard includes
#include <algorithm>
#include <limits>
#include <span>

// local includes
#include "display.h"
#include "host3d_subtitle_apps.h"
#include "text_game_pin_controller.h"
#include "video_player_context.h"
#include "src/logging.h"
#include "src/utility.h"

using namespace std::literals;

namespace platf::dxgi {

  namespace {
    [[nodiscard]] std::uint64_t authority_generation(
      const std::optional<foreground_window::snapshot_t> &authority
    ) noexcept {
      return authority ? authority->generation : 0u;
    }

    models::depth_input_region_t full_depth_input_region(
      const D3D11_TEXTURE2D_DESC &source_desc
    ) noexcept {
      const auto tensor_shape = models::fit_host_sbs_v2_depth_tensor_shape(
        source_desc.Width,
        source_desc.Height
      );
      return models::depth_input_region_t {
        .source_width = source_desc.Width,
        .source_height = source_desc.Height,
        .left = 0u,
        .top = 0u,
        .right = source_desc.Width,
        .bottom = source_desc.Height,
        .tensor_content = {
          0u,
          0u,
          static_cast<std::uint32_t>(std::max(tensor_shape.width, 0)),
          static_cast<std::uint32_t>(std::max(tensor_shape.height, 0)),
        },
        .analysis_generation = 0u,
        .authority = models::depth_analysis_authority_e::full_source,
      };
    }

    bool same_window_analysis_identity_shape(
      const foreground_window::snapshot_t &left,
      const foreground_window::snapshot_t &right
    ) noexcept {
      const auto width = [](const foreground_window::rect_t rect) {
        return static_cast<std::int64_t>(rect.right) - rect.left;
      };
      const auto height = [](const foreground_window::rect_t rect) {
        return static_cast<std::int64_t>(rect.bottom) - rect.top;
      };
      return left.window == right.window && left.process_id == right.process_id &&
             left.monitor == right.monitor && left.source_window() == right.source_window() &&
             width(left.source_screen_rect()) == width(right.source_screen_rect()) &&
             height(left.source_screen_rect()) == height(right.source_screen_rect());
    }

    template<typename CaptureRect>
    sbs_debug::window_region_snapshot make_window_region_snapshot(
      sbs_debug::window_region_snapshot identity,
      const D3D11_TEXTURE2D_DESC &source_desc,
      const std::uint64_t frame_id,
      const CaptureRect &capture_pixels,
      const std::chrono::steady_clock::time_point observed_at,
      const std::chrono::steady_clock::time_point geometry_valid_since,
      const std::chrono::steady_clock::time_point content_timestamp,
      const std::chrono::steady_clock::time_point captured_at,
      const std::chrono::milliseconds maximum_age
    ) noexcept {
      identity.matched_frame_id = frame_id;
      identity.source_width = source_desc.Width;
      identity.source_height = source_desc.Height;
      identity.left = capture_pixels.left;
      identity.top = capture_pixels.top;
      identity.right = capture_pixels.right;
      identity.bottom = capture_pixels.bottom;
      identity.latest_observation_age_ms_at_capture = static_cast<std::uint32_t>(
        std::clamp<std::int64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
            captured_at - observed_at
          )
            .count(),
          0,
          std::numeric_limits<std::uint32_t>::max()
        )
      );
      identity.maximum_observation_age_ms = static_cast<std::uint32_t>(
        maximum_age.count()
      );
      identity.geometry_continuity_ms_at_capture = static_cast<std::uint64_t>(
        std::max<std::int64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
            captured_at - geometry_valid_since
          )
            .count(),
          0
        )
      );
      identity.source_content_age_ms_at_capture = static_cast<std::uint64_t>(
        std::max<std::int64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
            captured_at - content_timestamp
          )
            .count(),
          0
        )
      );
      return identity;
    }

    bool same_window_region(
      const sbs_debug::window_region_snapshot &left,
      const sbs_debug::window_region_snapshot &right
    ) noexcept {
      return left.authority_kind == right.authority_kind &&
             left.source_width == right.source_width &&
             left.source_height == right.source_height &&
             left.left == right.left && left.top == right.top &&
             left.right == right.right && left.bottom == right.bottom &&
             left.hwnd == right.hwnd && left.process_id == right.process_id &&
             left.document_id == right.document_id && left.video_id == right.video_id;
    }

  }  // namespace

  void host_sbs_authority_tracker_t::enable_browser_helper(const bool enabled) {
    if (enabled) {
      // Cross-process accessibility traversal is isolated in the supervised helper. This lease
      // only exposes an atomic snapshot to the matched-frame path and cannot block streaming.
      if (!video_dom_lease) {
        video_dom_lease.emplace();
      }
    } else {
      video_dom_lease.reset();
    }
  }

  void host_sbs_authority_tracker_t::reset() {
    depth_analysis_generation_tracker = {};
    foreground_window_tracker.reset();
    live_window_authority.reset();
    live_foreground_region.reset();
    live_browser_authority.reset();
    live_browser_authority_epoch = 0u;
    subtitle_context_ = {};
    last_subtitle_admission_diagnostic.reset();
    last_subtitle_admission_diagnostic_at = {};
    video_player_registry_ready_observed = false;
    interactive_move_size_observed = false;
    browser_roi_not_before = {};
    last_window_region.reset();
    last_window_region_observer_status.reset();
    last_window_region_mapping_status.reset();
  }

  models::host_subtitle_context::presentation_authority_t host_sbs_authority_tracker_t::presentation_authority() const noexcept {
    return {
      authority_generation(live_window_authority),
      authority_generation(live_foreground_region),
      live_browser_authority_epoch,
      interactive_move_size_observed,
    };
  }

  models::host_subtitle_context::presentation_authority_t host_sbs_authority_tracker_t::presentation_authority(
    const std::uint64_t browser_authority_epoch
  ) const noexcept {
    return {
      authority_generation(live_window_authority),
      authority_generation(live_foreground_region),
      browser_authority_epoch,
      interactive_move_size_observed,
    };
  }

  bool host_sbs_authority_tracker_t::requires_foreground_causal_wait(
    const foreground_window::snapshot_t &foreground,
    const std::optional<std::chrono::steady_clock::time_point> &content_timestamp
  ) const noexcept {
    return live_foreground_region &&
           foreground_window::requires_full_source_causal_fallback(
             foreground,
             *live_foreground_region,
             content_timestamp
           );
  }

  void host_sbs_authority_tracker_t::attribute_window_generations(
    host_sbs_region_attribution_t &attribution,
    const observation_t &matched_authority
  ) const {
    attribution.live_window_authority_generation =
      attribution.window_region && window_region_matches_live_authority(*attribution.window_region) ?
        (
          attribution.window_region->authority_kind ==
              sbs_debug::window_region_authority_kind_e::chromium_video ?
            live_window_authority->generation :
            live_foreground_region->generation
        ) :
        0u;
    attribution.live_browser_authority_generation =
      attribution.window_region &&
          attribution.window_region->authority_kind ==
            sbs_debug::window_region_authority_kind_e::chromium_video ?
        matched_authority.browser_authority_epoch :
        0u;
    if (attribution.live_browser_authority_generation != 0u && video_dom_lease) {
      if (matched_authority.browser_authority) {
        attribution.live_browser_screen_left =
          matched_authority.browser_authority->screen_rect.left;
        attribution.live_browser_screen_top =
          matched_authority.browser_authority->screen_rect.top;
        attribution.live_browser_screen_right =
          matched_authority.browser_authority->screen_rect.right;
        attribution.live_browser_screen_bottom =
          matched_authority.browser_authority->screen_rect.bottom;
        attribution.live_browser_geometry_valid_since =
          matched_authority.browser_authority->geometry_valid_since;
        attribution.live_browser_authority_class =
          video_dom::geometry_authority_class(
            matched_authority.browser_authority->status
          );
      }
    }
  }

  void host_sbs_authority_tracker_t::attribute_subtitles(
    host_sbs_region_attribution_t &attribution,
    const D3D11_TEXTURE2D_DESC &source_desc
  ) const {
    attribution.subtitle_context_generation = subtitle_context_.generation();
    attribution.subtitle_context_suppression_eligible =
      attribution.observed_authority.interactive_move_size && subtitle_context_.eligible();
    attribution.test_instance = test_context ? test_context->instance : 0u;
    attribution.subtitle_line_policy =
      models::host_subtitle_context::requires_centered_lines(subtitle_context_.kind()) ?
        models::subtitle_line_policy_e::video_caption :
        models::subtitle_line_policy_e::application_text;
    attribution.subtitle_context_eligible = test_context ?
      subtitle_context_.eligible() && source_desc.Width == test_context->width && source_desc.Height == test_context->height :
      subtitle_context_.eligible() && attribution.window_region &&
      window_region_matches_live_authority(*attribution.window_region) &&
      ((subtitle_context_.kind() == models::host_subtitle_context::kind_e::browser_video) ==
       (attribution.window_region->authority_kind == sbs_debug::window_region_authority_kind_e::chromium_video));
  }

  host_sbs_authority_tracker_t::observation_t host_sbs_authority_tracker_t::observe(
    const D3D11_TEXTURE2D_DESC &source_desc,
    const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
    const inputs_t &inputs,
    const foreground_window::snapshot_t *revalidate_expected,
    bool sampling_for_attribution
  ) {
    if (test_context) {
      if (source_desc.Width != test_context->width || source_desc.Height != test_context->height || !test_context->instance) {
        subtitle_context_.update({});
      } else {
        subtitle_context_.update({models::host_subtitle_context::kind_e::registered_video_player,
          test_context->instance, GetCurrentProcessId()});
      }
      // Deliberately no actual USER32/browser/player claim. Only this labelled fixture
      // instance uses the synthetic scheduling domain; source/receipt authority stays real.
      live_window_authority.reset(); live_foreground_region.reset();
      live_browser_authority.reset(); live_browser_authority_epoch = 0u;
      interactive_move_size_observed = false;
      return {};
    }
    auto *const display = inputs.display;
    effects_t effects;
    foreground_window::snapshot_t observed;
    if (content_timestamp) {
      const auto observation = revalidate_expected ?
                                 foreground_window::sample_after_copy(
                                   *revalidate_expected
                                 ) :
                                 foreground_window::sample();
      observed = foreground_window_tracker.update(observation);
    } else {
      const bool interactive_move_size =
        foreground_window::interactive_move_size_active();
      observed = foreground_window_tracker.update({
        .status = interactive_move_size ?
                    foreground_window::status_e::interactive_move_size :
                    foreground_window::status_e::no_foreground,
        .observed_at = std::chrono::steady_clock::now(),
      });
    }

    const bool interactive_move_size =
      observed.status == foreground_window::status_e::interactive_move_size;
    if (interactive_move_size) {
      if (!interactive_move_size_observed) {
        BOOST_LOG(info)
          << "Host SBS selected full-source 3D for an interactive foreground-window "sv
             "move/resize."sv;
        if (inputs.pending_roi_completion) {
          // Consuming this exact-source ROI completion does not enqueue a replacement. Ask
          // the retained-source owner for one more conversion so full-source depth starts
          // even if USER32 stops producing desktop damage during the modal move loop.
          effects.depth_reprocess = true;
        }
      }
      interactive_move_size_observed = true;
      browser_roi_not_before = observed.observed_at;
    } else if (interactive_move_size_observed) {
      BOOST_LOG(info)
        << "Host SBS foreground-window move/resize ended; waiting for fresh ROI authority."sv;
      interactive_move_size_observed = false;
      browser_roi_not_before = observed.observed_at;
    }

    std::optional<foreground_window::snapshot_t> next_root;
    if (foreground_window::carries_geometry(observed.status)) {
      next_root = observed;
    }
    std::optional<foreground_window::snapshot_t> next_region;
    foreground_window::mapping_result_t foreground_mapping {
      .status = foreground_window::mapping_status_e::invalid_capture_target,
    };
    std::optional<foreground_window::capture_target_t> capture_target;
    foreground_window::mapping_result_t foreground_projection {
      .status = foreground_window::mapping_status_e::invalid_capture_target,
    };
    const bool can_map_live_target = display &&
      display->display_rotation == DXGI_MODE_ROTATION_IDENTITY &&
      static_cast<std::uint32_t>(display->width) == source_desc.Width &&
      static_cast<std::uint32_t>(display->height) == source_desc.Height;
    DXGI_OUTPUT_DESC output_desc {};
    if (next_root && display && display->output && (can_map_live_target || sampling_for_attribution) &&
        SUCCEEDED(display->output->GetDesc(&output_desc))) {
      capture_target = foreground_window::capture_target_t {
        .screen_rect = {
          output_desc.DesktopCoordinates.left,
          output_desc.DesktopCoordinates.top,
          output_desc.DesktopCoordinates.right,
          output_desc.DesktopCoordinates.bottom,
        },
        .width = source_desc.Width,
        .height = source_desc.Height,
        .monitor = reinterpret_cast<std::uintptr_t>(output_desc.Monitor),
        .identity_orientation = display->display_rotation == DXGI_MODE_ROTATION_IDENTITY,
      };
      foreground_projection = foreground_window::map_to_capture(observed, *capture_target);
    }
    if (capture_target && can_map_live_target) {
      // Keep liveness's stricter target gate and its diagnostic status. Attribution below
      // reuses the raw projection, including unsupported orientation/extent failures.
      // Liveness follows the current root/client/content geometry, not the content-causality
      // gate. A move must revoke an old positioned completion immediately, while the later
      // private-copy path still requires geometry_valid_since <= content_timestamp before
      // authorizing a new foreground ROI. This separation preserves the position-independent
      // DAV2 domain.
      foreground_mapping = foreground_projection;
      if (foreground_mapping) {
        next_region = observed;
      }
    }

    const auto changed = [](const auto &before, const auto &after) {
      return before.has_value() != after.has_value() ||
             (before && after && before->generation != after->generation);
    };
    const bool programmatic_causal_gap =
      live_foreground_region && next_region &&
      live_foreground_region->generation != next_region->generation &&
      foreground_window::requires_full_source_causal_fallback(
        *live_foreground_region,
        *next_region,
        content_timestamp
      );
    if (programmatic_causal_gap) {
      // This observation runs even while TensorRT owns the prior ROI slot. Keep requesting
      // retained-source conversions after that old completion is retired, so one exact
      // full-source submission occurs even if Desktop Duplication never presents again.
      effects.depth_reprocess = true;
    }
    const bool root_changed = changed(live_window_authority, next_root);
    const bool region_changed = changed(live_foreground_region, next_region);
    const bool analysis_source_description_changed = [&]() {
      if (!next_root) {
        return false;
      }
      if (!live_window_authority) {
        return true;
      }
      const auto relative_source = [](const foreground_window::snapshot_t &snapshot) {
        const auto source = snapshot.source_screen_rect();
        return foreground_window::rect_t {
          source.left - snapshot.client_screen_rect.left,
          source.top - snapshot.client_screen_rect.top,
          source.right - snapshot.client_screen_rect.left,
          source.bottom - snapshot.client_screen_rect.top,
        };
      };
      return live_window_authority->window != next_root->window ||
             live_window_authority->process_id != next_root->process_id ||
             live_window_authority->source_window() != next_root->source_window() ||
             relative_source(*live_window_authority) != relative_source(*next_root);
    }();
    const bool analysis_authority_changed =
      live_window_authority &&
      (!next_root ||
       !same_window_analysis_identity_shape(*live_window_authority, *next_root));
    if (root_changed || region_changed) {
      effects.window_changed = true;
      if (analysis_source_description_changed) {
        const auto source_rect = next_root->source_screen_rect();
        const bool structural_child = next_root->source_window() != next_root->window;
        BOOST_LOG(info)
          << "Host SBS foreground analysis source changed: source="sv
          << (structural_child ? "structural-child"sv : "whole-client"sv)
          << " root_client="sv << next_root->client_screen_rect.left << ':'
          << next_root->client_screen_rect.top << '-'
          << next_root->client_screen_rect.right << ':'
          << next_root->client_screen_rect.bottom << " analysis="sv
          << source_rect.left << ':' << source_rect.top << '-'
          << source_rect.right << ':' << source_rect.bottom << " root_hwnd=0x"sv
          << util::hex(next_root->window).to_string_view() << " content_hwnd=0x"sv
          << util::hex(next_root->source_window()).to_string_view()
          << " generation="sv << next_root->generation << '.';
      }
    }
    if (
      analysis_authority_changed ||
      (live_foreground_region && !next_region) ||
      !next_root
    ) {
      depth_analysis_generation_tracker.select_full_source();
    }
    live_window_authority = next_root;
    live_foreground_region = next_region;
    // Read the helper exactly once for this Win32 observation. The immutable shared snapshot,
    // the live epoch derived from it, and any later ROI mapping remain one attribution unit.
    const auto browser_snapshot = video_dom_lease ? video_dom_lease->latest() : nullptr;
    observe_live_browser_authority(browser_snapshot, effects);
    const auto subtitle_admission = observe_subtitle_context(
      observed, browser_snapshot, output_desc, source_desc, content_timestamp, foreground_mapping, inputs, effects);
    return {
      .foreground = observed,
      .capture_target = capture_target,
      .foreground_projection = foreground_projection,
      .browser_snapshot = browser_snapshot,
      .browser_projection = subtitle_admission.browser_mapping,
      .browser_authority = live_browser_authority,
      .browser_authority_epoch = live_browser_authority_epoch,
      .effects = effects,
    };
  }

  void host_sbs_authority_tracker_t::observe_live_browser_authority(
    const video_dom::snapshot_ptr &browser,
    effects_t &effects
  ) {
    std::optional<browser_live_authority_key_t> next;
    if (browser) {
      constexpr auto maximum_age = std::chrono::milliseconds {2500};
      if (
        video_dom::carries_video_geometry(browser->status) &&
        browser->received_at >= browser_roi_not_before &&
        video_dom::usable(*browser, std::chrono::steady_clock::now(), maximum_age)
      ) {
        next = browser_live_authority_key_t {
          .status = browser->status,
          .window = browser->window,
          .process_id = browser->process_id,
          .document_id = browser->document_id,
          .video_id = browser->video_id,
          .screen_rect = browser->screen_rect,
          .geometry_valid_since = browser->geometry_valid_since,
        };
      }
    }
    if (live_browser_authority == next) {
      return;
    }
    const auto same_analysis_identity_shape = [](const auto &left, const auto &right) {
      const auto width = [](const video_dom::rect_t rect) {
        return static_cast<std::int64_t>(rect.right) - rect.left;
      };
      const auto height = [](const video_dom::rect_t rect) {
        return static_cast<std::int64_t>(rect.bottom) - rect.top;
      };
      return left.window == right.window && left.process_id == right.process_id &&
             left.document_id == right.document_id && left.video_id == right.video_id &&
             video_dom::geometry_authority_class(left.status) ==
               video_dom::geometry_authority_class(right.status) &&
             width(left.screen_rect) == width(right.screen_rect) &&
             height(left.screen_rect) == height(right.screen_rect);
    };
    const bool analysis_rearm =
      live_browser_authority &&
      (!next || !same_analysis_identity_shape(*live_browser_authority, *next));
    live_browser_authority = next;
    if (++live_browser_authority_epoch == 0u) {
      ++live_browser_authority_epoch;
    }
    // Chromium-video or positioned packed output leaves with the old browser authority.
    effects.browser_changed = true;
    if (analysis_rearm || !next) {
      depth_analysis_generation_tracker.select_full_source();
    }
  }

  // The image path of the process that owns the foreground window. A window belongs to one
  // process for its whole life, so the path is asked for again only for another window or
  // process, and once a second in case both identifiers were reused together. Only the path
  // is kept: who may have subtitle protection is still decided on every observation.
  const std::optional<std::wstring> &host_sbs_authority_tracker_t::subtitle_process_image_path(
    const foreground_window::snapshot_t &foreground,
    const std::chrono::steady_clock::time_point now
  ) {
    auto &cached = subtitle_process_image;
    if (cached.window != foreground.window || cached.process_id != foreground.process_id ||
        now - cached.queried_at >= std::chrono::seconds(1)) {
      cached = {foreground.window, foreground.process_id, now,
                foreground_window::process_image_path(foreground.process_id)};
    }
    return cached.path;
  }

  // Subtitle permission is independent of the generic foreground depth ROI. Fullscreen keeps
  // its semantic video identity even when depth analysis canonicalizes to the whole source.
  subtitle_context_policy::evaluation_t host_sbs_authority_tracker_t::observe_subtitle_context(
    const foreground_window::snapshot_t &foreground,
    const video_dom::snapshot_ptr &browser,
    const DXGI_OUTPUT_DESC &output,
    const D3D11_TEXTURE2D_DESC &source,
    const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
    const foreground_window::mapping_result_t &foreground_mapping,
    const inputs_t &inputs,
    effects_t &effects
  ) {
    using namespace models::host_subtitle_context;
    key_t next;
    video_player_registry_ready_observed = inputs.registry && inputs.registry->ready();
    const auto now = std::chrono::steady_clock::now();
    const bool foreground_region_present = static_cast<bool>(live_foreground_region);
    const auto admission = subtitle_context_policy::evaluate(
      foreground.status == foreground_window::status_e::interactive_move_size ? &foreground :
        live_window_authority ? &*live_window_authority : nullptr,
      foreground_region_present, foreground_mapping, browser.get(),
      live_browser_authority ? live_browser_authority_epoch : 0u,
      {output.DesktopCoordinates.left, output.DesktopCoordinates.top,
       output.DesktopCoordinates.right, output.DesktopCoordinates.bottom},
      source.Width, source.Height, content_timestamp, now, subtitle_context_.key());
    const bool foreground_usable = admission.foreground_usable;
    const bool browser_identity_matches = admission.browser_identity_matches;
    const bool browser_usable = admission.browser_usable;
    const bool mapping_attempted = admission.browser_mapping.has_value();
    const auto mapping_status = mapping_attempted ?
      admission.browser_mapping->status : video_dom::mapping_status_e::invalid_video_rect;
    const bool mapping_succeeded = mapping_attempted && static_cast<bool>(*admission.browser_mapping);
    const bool fullscreen_check_evaluated = admission.fullscreen_check_evaluated;
    const bool fullscreen_match = admission.fullscreen_match;
    const bool browser_video = admission.browser_key.kind == kind_e::browser_video;
    if (browser_video) {
      next = admission.browser_key;
    } else if (admission.registered_player_candidate) {
      // A Game 3D text game is an application its owner approved for exactly this purpose. The
      // approval names that process: the game's answer is refreshed a few times a second, so
      // another window that has just come in front must not inherit it meanwhile.
      // Unless its helper has lately shown that it sees the text, the picture-based search
      // covers it.
      const bool approved_text_game = inputs.text_game &&
                                      inputs.text_game->picture_search_covers(foreground.process_id, std::chrono::steady_clock::now());
      const auto registered_executables = inputs.registry ?
        inputs.registry->executables_if_ready() : std::span<const std::wstring> {};
      const auto executable = !registered_executables.empty() || !approved_text_game ?
        subtitle_process_image_path(foreground, now) : std::nullopt;
      // Registration keeps precedence. Both policies inspect the same process image while
      // the owner's current settings still take effect on every authority observation.
      const bool registered = executable && video_player::matches_registered_executable(registered_executables, *executable);
      const bool owner_enabled = !registered &&
        (approved_text_game || (executable && host3d_subtitle_apps::observe_executable(*executable)));
      next = {native_kind(registered, owner_enabled), foreground.window, foreground.process_id};
    }
    const bool context_changed = subtitle_context_.update(next);
    if (context_changed) {
      // Revoke even full-source completions and already-packed output. The generation survives
      // busy drops, so returning to the same video cannot revive pre-transition protection.
      effects.subtitle_context_changed = true;
    }
    const char *reason = "no-video-authority";
    if (subtitle_context_.kind() == kind_e::browser_video) {
      reason = "browser-video-authorized";
    } else if (subtitle_context_.kind() == kind_e::registered_video_player) {
      reason = "registered-player-authorized";
    } else if (subtitle_context_.kind() == kind_e::foreground_app) {
      reason = "foreground-app-authorized";
    } else if (!live_window_authority) {
      reason = "foreground-root-unavailable";
    } else if (!foreground_usable) {
      reason = "foreground-unusable-for-content";
    } else if (!admission.foreground_target_usable) {
      reason = "foreground-target-unavailable";
    } else if (!foreground_region_present && !browser_identity_matches) {
      reason = "foreground-region-unavailable";
    } else if (!browser) {
      reason = "browser-snapshot-unavailable";
    } else if (!live_browser_authority) {
      reason = "browser-authority-unavailable";
    } else if (!browser_identity_matches) {
      reason = "browser-identity-mismatch";
    } else if (!browser_usable) {
      reason = "browser-unusable-for-content";
    } else if (!mapping_succeeded) {
      reason = "browser-mapping-failed";
    } else if (!fullscreen_match) {
      reason = "browser-fullscreen-mismatch";
    } else if (browser_video) {
      reason = "browser-key-rejected";
    }
    const bool heartbeat_age_available = browser &&
                                         browser->received_at.time_since_epoch().count() != 0;
    const bool browser_heartbeat_fresh = heartbeat_age_available && now >= browser->received_at &&
                                         now - browser->received_at <= std::chrono::milliseconds {2500};
    const bool browser_after_move = browser && browser->received_at >= browser_roi_not_before;
    const subtitle_admission_diagnostic_key_t diagnostic_key {
      .context_generation = subtitle_context_.generation(),
      .reason = reason,
      .foreground_status = foreground.status,
      .foreground_mapping_status = foreground_mapping.status,
      .browser_status = browser ? browser->status : video_dom::status_e::stopped,
      .browser_mapping_status = mapping_status,
      .foreground_window = foreground.window,
      .foreground_process = foreground.process_id,
      .browser_window = browser ? browser->window : 0u,
      .browser_process = browser ? browser->process_id : 0u,
      .browser_document = browser ? browser->document_id : 0,
      .browser_video = browser ? browser->video_id : 0,
      .foreground_root = static_cast<bool>(live_window_authority),
      .foreground_region = foreground_region_present,
      .foreground_target_usable = admission.foreground_target_usable,
      .foreground_usable = foreground_usable,
      .content_timestamp_available = content_timestamp.has_value(),
      .browser_snapshot = static_cast<bool>(browser),
      .browser_heartbeat_fresh = browser_heartbeat_fresh,
      .browser_after_move = browser_after_move,
      .browser_authority = static_cast<bool>(live_browser_authority),
      .browser_identity_matches = browser_identity_matches,
      .browser_usable = browser_usable,
      .mapping_attempted = mapping_attempted,
      .mapping_succeeded = mapping_succeeded,
      .fullscreen_checked = fullscreen_check_evaluated,
      .fullscreen_match = fullscreen_match,
      .registered_player_candidate = admission.registered_player_candidate,
      .registry_ready = video_player_registry_ready_observed,
      .retained_for_interactive_move_size = admission.retained_for_interactive_move_size,
    };
    // Admission can change between disabled reasons without changing semantic context. Keep
    // that evidence visible without resetting history or logging every heartbeat/geometry tick.
    // Rapid nonsemantic fluctuations coalesce to the latest state at most once per second;
    // actual context transitions retain their immediate log and existing revocation semantics.
    if (!context_changed && last_subtitle_admission_diagnostic &&
        (*last_subtitle_admission_diagnostic == diagnostic_key ||
         now - last_subtitle_admission_diagnostic_at < std::chrono::seconds {1})) {
      return admission;
    }
    last_subtitle_admission_diagnostic = diagnostic_key;
    last_subtitle_admission_diagnostic_at = now;
    const double heartbeat_age_ms = heartbeat_age_available ?
                                      std::chrono::duration<double, std::milli>(now - browser->received_at).count() :
                                      0.0;
    const bool geometry_content_delta_available = browser && content_timestamp &&
                                                  browser->geometry_valid_since.time_since_epoch().count() != 0;
    const double content_minus_geometry_ms = geometry_content_delta_available ?
                                               std::chrono::duration<double, std::milli>(*content_timestamp - browser->geometry_valid_since).count() :
                                               0.0;
    BOOST_LOG(info) << "Host SBS subtitle context="sv
                    << kind_name(subtitle_context_.kind())
                    << " generation="sv << subtitle_context_.generation() << '.'
                    << " reason="sv << reason
                    << " context_changed="sv << context_changed
                    << " foreground_status="sv << foreground_window::status_name(foreground.status)
                    << " foreground_root="sv << static_cast<bool>(live_window_authority)
                    << " foreground_region="sv << foreground_region_present
                    << " foreground_target_usable="sv << admission.foreground_target_usable
                    << " foreground_mapping_status="sv << foreground_window::mapping_status_name(foreground_mapping.status)
                    << " foreground_usable="sv << foreground_usable
                    << " foreground_hwnd=0x"sv << util::hex(foreground.window).to_string_view()
                    << " foreground_pid="sv << foreground.process_id
                    << " foreground_generation="sv << foreground.generation
                    << " foreground_client="sv << foreground.client_screen_rect.left << ':'
                    << foreground.client_screen_rect.top << '-'
                    << foreground.client_screen_rect.right << ':' << foreground.client_screen_rect.bottom
                    << " capture="sv << output.DesktopCoordinates.left << ':'
                    << output.DesktopCoordinates.top << '-' << output.DesktopCoordinates.right << ':'
                    << output.DesktopCoordinates.bottom
                    << " source="sv << source.Width << 'x' << source.Height
                    << " content_timestamp_available="sv << content_timestamp.has_value()
                    << " content_age_ms="sv << (content_timestamp ? std::chrono::duration<double, std::milli>(now - *content_timestamp).count() : 0.0)
                    << " browser_snapshot="sv << static_cast<bool>(browser)
                    << " browser_status="sv << (browser ? video_dom::status_name(browser->status) : "none")
                    << " live_browser_authority="sv << static_cast<bool>(live_browser_authority)
                    << " browser_heartbeat_fresh="sv << browser_heartbeat_fresh
                    << " browser_after_move="sv << browser_after_move
                    << " browser_epoch="sv << live_browser_authority_epoch
                    << " browser_identity_checked="sv << (foreground_usable && admission.foreground_target_usable)
                    << " browser_identity_match="sv << browser_identity_matches
                    << " browser_usable_checked="sv << (foreground_usable && browser_identity_matches)
                    << " browser_usable="sv << browser_usable
                    << " mapping_attempted="sv << mapping_attempted
                    << " mapping_status="sv << (mapping_attempted ? video_dom::mapping_status_name(mapping_status) : "not-evaluated")
                    << " fullscreen_checked="sv << fullscreen_check_evaluated
                    << " fullscreen_match="sv << fullscreen_match
                    << " registered_player_candidate="sv << admission.registered_player_candidate
                    << " registry_ready="sv << video_player_registry_ready_observed
                    << " retained_for_interactive_move_size="sv << admission.retained_for_interactive_move_size
                    << " browser_sequence="sv << (browser ? browser->helper_sequence : 0u)
                    << " browser_hwnd=0x"sv << util::hex(browser ? browser->window : 0u).to_string_view()
                    << " browser_pid="sv << (browser ? browser->process_id : 0u)
                    << " browser_document="sv << (browser ? browser->document_id : 0)
                    << " browser_video="sv << (browser ? browser->video_id : 0)
                    << " browser_rect="sv << (browser ? browser->screen_rect.left : 0) << ':'
                    << (browser ? browser->screen_rect.top : 0) << '-'
                    << (browser ? browser->screen_rect.right : 0) << ':'
                    << (browser ? browser->screen_rect.bottom : 0)
                    << " heartbeat_age_available="sv << heartbeat_age_available
                    << " heartbeat_age_ms="sv << heartbeat_age_ms
                    << " geometry_content_delta_available="sv << geometry_content_delta_available
                    << " content_minus_geometry_ms="sv << content_minus_geometry_ms;
    return admission;
  }

  bool host_sbs_authority_tracker_t::window_region_matches_live_authority(
    const sbs_debug::window_region_snapshot &region
  ) const noexcept {
    const auto &authority =
      region.authority_kind == sbs_debug::window_region_authority_kind_e::chromium_video ?
        live_window_authority :
        live_foreground_region;
    return authority && authority->window == region.hwnd &&
           authority->process_id == region.process_id;
  }

  bool host_sbs_authority_tracker_t::window_region_authorized_for_render(
    const host_sbs_region_attribution_t &slot
  ) const noexcept {
    if (test_context) {
      return slot.test_instance == test_context->instance && slot.test_instance != 0u &&
        slot.depth_input_region.source_width == test_context->width &&
        slot.depth_input_region.source_height == test_context->height &&
        subtitle_context_.matches(slot.subtitle_context_generation);
    }
    if (!slot.window_region || !window_region_matches_live_authority(*slot.window_region) || slot.live_window_authority_generation == 0u) {
      return false;
    }
    if (
      slot.window_region->authority_kind ==
      sbs_debug::window_region_authority_kind_e::chromium_video
    ) {
      const auto browser = video_dom_lease ? video_dom_lease->latest() : nullptr;
      return live_window_authority->generation == slot.live_window_authority_generation &&
             slot.live_browser_authority_generation == live_browser_authority_epoch &&
             browser && video_dom::carries_video_geometry(browser->status) &&
             browser->window == slot.window_region->hwnd &&
             browser->process_id == slot.window_region->process_id &&
             browser->document_id == slot.window_region->document_id &&
             browser->video_id == slot.window_region->video_id &&
             browser->screen_rect.left == slot.live_browser_screen_left &&
             browser->screen_rect.top == slot.live_browser_screen_top &&
             browser->screen_rect.right == slot.live_browser_screen_right &&
             browser->screen_rect.bottom == slot.live_browser_screen_bottom &&
             browser->geometry_valid_since ==
               slot.live_browser_geometry_valid_since &&
             video_dom::geometry_authority_class(browser->status) ==
               slot.live_browser_authority_class;
    }
    return live_foreground_region->generation == slot.live_window_authority_generation;
  }

  void host_sbs_authority_tracker_t::select_depth_input_region(
    host_sbs_region_attribution_t &slot,
    const D3D11_TEXTURE2D_DESC &source_desc,
    const bool finalize_full_source
  ) {
    slot.depth_video_plan.reset();
    slot.depth_input_region = full_depth_input_region(source_desc);

    if (slot.window_region) {
      const auto &region = *slot.window_region;
      const models::depth_source_rect_t semantic_rect {
        static_cast<std::uint32_t>(region.left),
        static_cast<std::uint32_t>(region.top),
        static_cast<std::uint32_t>(region.right),
        static_cast<std::uint32_t>(region.bottom),
      };
      const auto authority =
        region.authority_kind == sbs_debug::window_region_authority_kind_e::chromium_video ?
          models::depth_analysis_authority_e::chromium_video :
          models::depth_analysis_authority_e::foreground_client;
      const auto make_domain_key =
        [&](const models::depth_source_rect_t analysis_rect) {
          return models::depth_analysis_domain_key_t {
            .source_width = source_desc.Width,
            .source_height = source_desc.Height,
            .semantic_width = semantic_rect.width(),
            .semantic_height = semantic_rect.height(),
            .crop_width = analysis_rect.width(),
            .crop_height = analysis_rect.height(),
            .hwnd = region.hwnd,
            .process_id = region.process_id,
            .document_id = region.document_id,
            .video_id = region.video_id,
            .authority = authority,
          };
        };
      const auto active_shape = models::fit_host_sbs_v2_depth_tensor_shape(
        source_desc.Width,
        source_desc.Height
      );
      auto plan = models::plan_host_sbs_v2_video_region(
        semantic_rect,
        source_desc.Width,
        source_desc.Height,
        active_shape
      );
      if (plan) {
        const auto key = make_domain_key(plan->source_rect);
        slot.depth_input_region = models::depth_input_region_t {
          .source_width = source_desc.Width,
          .source_height = source_desc.Height,
          .left = plan->source_rect.left,
          .top = plan->source_rect.top,
          .right = plan->source_rect.right,
          .bottom = plan->source_rect.bottom,
          .tensor_content = plan->tensor_content,
          .analysis_generation = depth_analysis_generation(key),
          .authority = authority,
        };
        slot.depth_video_plan = std::move(plan);
      }
    }

    if (!slot.depth_input_region.is_video_region()) {
      if (finalize_full_source) {
        depth_analysis_generation_tracker.select_full_source();
      }
      if (
        slot.window_region &&
        !(
          slot.window_region->left == 0 && slot.window_region->top == 0 &&
          slot.window_region->right == slot.window_region->source_width &&
          slot.window_region->bottom == slot.window_region->source_height
        )
      ) {
        slot.window_region.reset();
      }
    }
  }

  std::optional<sbs_debug::window_region_snapshot>
    host_sbs_authority_tracker_t::capture_browser_window_region(
      const D3D11_TEXTURE2D_DESC &source_desc,
      display_base_t *display,
      const observation_t &authority,
      const std::uint64_t frame_id,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
      const std::chrono::steady_clock::time_point captured_at,
      video_dom::status_e &video_status,
      video_dom::mapping_status_e &mapping_status
    ) const {
    const auto &observed = authority.browser_snapshot;
    video_status = video_dom::status_e::stopped;
    mapping_status = video_dom::mapping_status_e::invalid_video_rect;
    if (!observed || !display) {
      return std::nullopt;
    }
    video_status = observed->status;
    if (observed->received_at < browser_roi_not_before) {
      if (video_dom::carries_video_geometry(video_status)) {
        video_status = video_dom::status_e::stale;
      }
      return std::nullopt;
    }
    constexpr auto maximum_age = std::chrono::milliseconds {2500};
    // The newest heartbeat proves liveness; the uninterrupted exact-geometry run proves
    // causality. An identical heartbeat may accompany unchanged pixels, but a newly changed
    // identity or rectangle must never be attached retroactively to an older matched frame.
    if (
      !video_dom::detail::usable_for_content(
        *observed,
        content_timestamp,
        captured_at,
        maximum_age
      )
    ) {
      if (video_dom::carries_video_geometry(video_status)) {
        video_status = video_dom::status_e::stale;
      }
      return std::nullopt;
    }
    const auto observed_window = reinterpret_cast<HWND>(
      static_cast<std::uintptr_t>(observed->window)
    );
    DWORD foreground_process_id = 0;
    const auto foreground_window = GetAncestor(GetForegroundWindow(), GA_ROOT);
    if (
      !observed_window || !IsWindow(observed_window) ||
      foreground_window != observed_window ||
      GetWindowThreadProcessId(observed_window, &foreground_process_id) == 0 ||
      foreground_process_id != observed->process_id
    ) {
      video_status = video_dom::status_e::changed;
      mapping_status = video_dom::mapping_status_e::foreground_mismatch;
      return std::nullopt;
    }
    if (
      display->display_rotation != DXGI_MODE_ROTATION_IDENTITY ||
      display->width <= 0 || display->height <= 0 ||
      static_cast<std::uint32_t>(display->width) != source_desc.Width ||
      static_cast<std::uint32_t>(display->height) != source_desc.Height
    ) {
      mapping_status =
        display->display_rotation == DXGI_MODE_ROTATION_IDENTITY ?
          video_dom::mapping_status_e::extent_mismatch :
          video_dom::mapping_status_e::unsupported_rotation;
      return std::nullopt;
    }

    if (!authority.capture_target) {
      mapping_status = video_dom::mapping_status_e::invalid_capture_rect;
      return std::nullopt;
    }
    const auto &target = *authority.capture_target;
    const video_dom::rect_t desktop_rect {
      target.screen_rect.left,
      target.screen_rect.top,
      target.screen_rect.right,
      target.screen_rect.bottom,
    };
    const auto mapped = authority.browser_projection ?
                          *authority.browser_projection :
                          video_dom::map_screen_rect_to_capture(
                            observed->screen_rect, desktop_rect,
                            source_desc.Width, source_desc.Height,
                            video_dom::rotation_e::identity);
    mapping_status = mapped.status;
    if (!mapped) {
      return std::nullopt;
    }

    return make_window_region_snapshot(
      {
        .authority_kind = sbs_debug::window_region_authority_kind_e::chromium_video,
        .hwnd = observed->window,
        .process_id = observed->process_id,
        .document_id = observed->document_id,
        .video_id = observed->video_id,
        .generation = observed->generation,
      },
      source_desc,
      frame_id,
      mapped.capture_pixels,
      observed->received_at,
      observed->geometry_valid_since,
      *content_timestamp,
      captured_at,
      maximum_age
    );
  }

  std::optional<sbs_debug::window_region_snapshot>
    host_sbs_authority_tracker_t::capture_foreground_window_region(
      const D3D11_TEXTURE2D_DESC &source_desc,
      display_base_t *display,
      const std::uint64_t frame_id,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
      const std::chrono::steady_clock::time_point captured_at,
      const observation_t &authority,
      std::string_view &observer_status,
      std::string_view &mapping_status
    ) const {
    const auto &snapshot = authority.foreground;
    constexpr auto maximum_age = std::chrono::milliseconds {250};
    observer_status = foreground_window::status_name(snapshot.status);
    mapping_status = "not-mapped";
    if (
      !display ||
      !foreground_window::usable_for_content(
        snapshot,
        content_timestamp,
        captured_at,
        maximum_age
      )
    ) {
      return std::nullopt;
    }

    const auto &mapped = authority.foreground_projection;
    mapping_status = foreground_window::mapping_status_name(mapped.status);
    if (!mapped) {
      return std::nullopt;
    }

    return make_window_region_snapshot(
      {
        .authority_kind = sbs_debug::window_region_authority_kind_e::foreground_client,
        .hwnd = snapshot.window,
        .process_id = snapshot.process_id,
        .generation = snapshot.generation,
      },
      source_desc,
      frame_id,
      mapped.capture_pixels,
      snapshot.observed_at,
      snapshot.geometry_valid_since,
      *content_timestamp,
      captured_at,
      maximum_age
    );
  }

  void host_sbs_authority_tracker_t::publish_window_region_transition(
    const host_sbs_region_attribution_t &slot,
    const std::uint64_t frame_id
  ) {
    const bool observer_state_changed =
      !last_window_region_observer_status ||
      *last_window_region_observer_status != slot.window_region_observer_status ||
      !last_window_region_mapping_status ||
      *last_window_region_mapping_status != slot.window_region_mapping_status;
    if (slot.window_region) {
      if (
        !last_window_region ||
        !same_window_region(*last_window_region, *slot.window_region) ||
        observer_state_changed
      ) {
        const auto &region = *slot.window_region;
        BOOST_LOG(info)
          << "Host SBS window region attributed to matched frame "sv
          << region.matched_frame_id << ": authority="sv
          << sbs_debug::window_region_authority_kind_name(region.authority_kind)
          << " capture="sv << region.left << ':' << region.top << '-'
          << region.right << ':' << region.bottom << " source="sv
          << region.source_width << 'x' << region.source_height << " hwnd=0x"sv
          << util::hex(region.hwnd).to_string_view() << " pid="sv
          << region.process_id << " document="sv << region.document_id
          << " video="sv << region.video_id << " observation_age_ms="sv
          << region.latest_observation_age_ms_at_capture
          << " geometry_before_content_ms="sv
          << region.geometry_continuity_ms_at_capture -
               region.source_content_age_ms_at_capture
          << '.';
        if (slot.depth_video_plan) {
          const auto &plan = *slot.depth_video_plan;
          BOOST_LOG(info)
            << "Host SBS window-region DAV2 analysis active: crop="sv
            << plan.source_rect.left << ':' << plan.source_rect.top << '-'
            << plan.source_rect.right << ':' << plan.source_rect.bottom
            << " tensor="sv << plan.tensor_shape.width << 'x'
            << plan.tensor_shape.height << " letterbox_padding_pct="sv
            << plan.padded_area_fraction * 100.0f
            << "; the scene center is region-local."sv;
        } else if (
          region.left == 0 && region.top == 0 &&
          region.right == region.source_width &&
          region.bottom == region.source_height
        ) {
          BOOST_LOG(info)
            << "Host SBS window region covers the exact capture extent; using canonical "sv
               "ordinary full-frame V2."sv;
        } else if (
          region.authority_kind ==
            sbs_debug::window_region_authority_kind_e::chromium_video &&
          slot.window_region_observer_status == "ok-fullscreen"
        ) {
          BOOST_LOG(info)
            << "Host SBS fullscreen-only browser-client authority does not map to the exact "sv
               "capture extent; the windowed ROI remains disabled."sv;
        } else {
          BOOST_LOG(info)
            << "Host SBS window region is not eligible for the active authenticated tensor; "sv
               "using ordinary full-frame V2."sv;
        }
      }
      last_window_region = slot.window_region;
      last_window_region_observer_status = slot.window_region_observer_status;
      last_window_region_mapping_status = slot.window_region_mapping_status;
      return;
    }

    if (last_window_region) {
      BOOST_LOG(info)
        << "Host SBS window region released at matched frame "sv
        << frame_id << " (observer_status="sv
        << slot.window_region_observer_status
        << ", mapping_status="sv << slot.window_region_mapping_status
        << "); full-frame V2 remains active."sv;
      last_window_region.reset();
    } else if (observer_state_changed) {
      BOOST_LOG(debug)
        << "Host SBS window region unavailable at matched frame "sv
        << frame_id << " (observer_status="sv
        << slot.window_region_observer_status
        << ", mapping_status="sv << slot.window_region_mapping_status
        << "); full-frame V2 remains active."sv;
    }
    last_window_region_observer_status = slot.window_region_observer_status;
    last_window_region_mapping_status = slot.window_region_mapping_status;
  }

}  // namespace platf::dxgi
