/**
 * @file src/platform/windows/game_export_pipeline.h
 * @brief What one converter owns of Game 3D: the add-on export it receives, the Windows cursor
 *        composited into it, the source state it reports to the client and its Dump 3D package.
 */
#pragma once

// standard includes
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <optional>

// platform includes
#include <d3d11.h>

// local includes
#include "display.h"
#include "game3d_debug_dump.h"
#include "reshade_bridge.h"
#include "src/game_source_tracker.h"
#include "src/logging.h"
#include "src/thread_safe.h"
#include "src/video.h"

namespace platf::dxgi {
  /**
   * The Game 3D export pipeline of one converter.
   *
   * In a Game mode the converter receives what the Game 3D add-on exports from the game in front,
   * composites the Windows cursor into both of its eyes, resamples eyes authored at another size to
   * its own packed raster, and tells the client whether a game source is ready. This holds that
   * state; the converter sets it up in init_output() and draws with it in convert().
   */
  struct game_export_pipeline_t {
    std::unique_ptr<platf::reshade_bridge::receiver_t> receiver;
    std::shared_ptr<void> setup_watch;
    std::chrono::steady_clock::time_point next_wake_log {};  ///< See log_wakes().
    std::unique_ptr<sbs_cursor::compositor_t> cursor;
    float cursor_white_multiplier = 203.0f / 80.0f;
    bool cursor_logged = false;
    safe::mail_raw_t::event_t<::video::game_source_state_t> status_event;
    std::shared_ptr<::video::effective_video_mode_publisher_t> effective_mode;
    ::video::game_source_tracker_t source_tracker;
    bool source_supported = false;
    int source_width = 0;
    int source_height = 0;
    int output_width = 0;
    int output_height = 0;
    // Resizes same-aspect external full-SBS eyes to the packed output before YUV conversion.
    ps_t resample_ps;
    ps_t resample_pq_ps;  ///< Decodes PQ taps to FP16 scRGB while resizing.
    platf::game3d_debug::dumper dumper;  ///< Exact add-on inputs and provider observations, independent of AI dumps.

    /**
     * The add-on source in front of this display. A packed mode converts it: the frame is set
     * when one is ready for the packed raster. Game mono only observes it.
     *
     * @param packed_viewport The converter's packed raster.
     */
    std::optional<::video::game_source_identity_t> observe(
      int sbs_mode,
      const RECT &source_rect,
      const D3D11_VIEWPORT &packed_viewport,
      std::optional<platf::reshade_bridge::frame_t> &external
    ) {
      std::optional<::video::game_source_identity_t> game_source;
      const int export_width = source_width * 2;
      const int export_height = source_height;
      if (::video::is_packed_mode(sbs_mode)) {
        // The export is scaled or fitted into this converter's own packed raster, so the
        // receiver places it for exactly that raster (Game SBS: the export size; Local AR: its
        // presentation target) and refuses what the converter's resample could not place.
        // A returned frame's texture always has the packed size poll() placed for this raster.
        const auto packed_width = static_cast<std::uint32_t>(std::lround(packed_viewport.Width));
        const auto packed_height = static_cast<std::uint32_t>(std::lround(packed_viewport.Height));
        external = receiver->poll(source_rect, static_cast<int>(packed_width), static_cast<int>(packed_height));
        if (external) {
          game_source = ::video::game_source_identity_t {external->producer_process_id, external->producer_creation_time, external->resource_generation};
        }
      } else if (const auto status = receiver->status(source_rect, export_width, export_height)) {
        // Game mono never shows the export: observe it without becoming its consumer, so the
        // game neither allocates the shared ring nor packs stereo that would go unread.
        game_source = ::video::game_source_identity_t {status->producer_process_id, status->producer_creation_time, status->resource_generation};
      }
      return game_source;
    }

    /** Tell the client whether a game source is ready, once the applied mode is this one. */
    void publish_status(int sbs_mode, const std::optional<::video::game_source_identity_t> &source) {
      if (!::video::is_game_mode(sbs_mode) || !status_event || !effective_mode || source_width < 2 || source_width > 32767 || source_height < 2 || source_height > 65535) {
        return;
      }
      const auto applied = effective_mode->current();
      if (applied.sbs_mode != sbs_mode || applied.source_width != source_width || applied.source_height != source_height || applied.encoded_width != output_width || applied.encoded_height != output_height) {
        return;
      }
      ::video::game_source_state_t state {};
      state.state = !source_supported ? ::video::GAME_SOURCE_UNSUPPORTED :
                                        (source ? ::video::GAME_SOURCE_READY : ::video::GAME_SOURCE_WAITING);
      state.provider = source ? ::video::GAME_PROVIDER_RESHADE : ::video::GAME_PROVIDER_NONE;
      state.presentation_generation = applied.generation;
      state.source_width = static_cast<std::uint16_t>(source_width);
      state.source_height = static_cast<std::uint16_t>(source_height);
      state.packed_width = static_cast<std::uint16_t>(source_width * 2);
      state.packed_height = state.source_height;
      if (const auto update = source_tracker.observe(state, source.value_or(::video::game_source_identity_t {}), std::chrono::steady_clock::now())) {
        status_event->raise(*update);
      }
    }

    // Whether the export's completion wake reaches the encode loop: every 20 s while diagnostics
    // are enabled, the frames claimed and the claims that found their frame complete first.
    void log_wakes() {
      constexpr auto interval = std::chrono::seconds(20);
      const auto now = std::chrono::steady_clock::now();
      if (now < next_wake_log) {
        return;
      }
      const auto counts = receiver->take_wake_counts();
      if (next_wake_log != std::chrono::steady_clock::time_point {} && counts.claims) {
        const auto ms = [](std::chrono::nanoseconds value) {
          return std::chrono::duration<double, std::milli>(value).count();
        };
        BOOST_LOG(info) << "Game 3D export: " << counts.claims << " new frames claimed and " << counts.wakes
                        << " fence wakes in " << interval.count() << " s; " << counts.claims_before_wake
                        << " claims found their frame complete before its wake (late or missing wake); "
                        << std::format(
                             "fence wake to claim avg {:.2f} ms, max {:.2f} ms over {} claims.",
                             counts.claims_after_wake ? ms(counts.wake_to_claim_total) / static_cast<double>(counts.claims_after_wake) : 0.0,
                             ms(counts.wake_to_claim_max),
                             counts.claims_after_wake
                           );
      }
      next_wake_log = now + interval;
    }
  };
}  // namespace platf::dxgi
