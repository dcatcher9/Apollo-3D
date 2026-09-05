/**
 * @file src/platform/windows/display_base.cpp
 * @brief Definitions for the Windows display base code.
 */
// standard includes
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>

// platform includes
#include <initguid.h>

// lib includes
#include <boost/algorithm/string/join.hpp>
#include <MinHook.h>

// We have to include boost/process/v1.hpp before display.h due to WinSock.h,
// but that prevents the definition of NTSTATUS so we must define it ourself.
typedef long NTSTATUS;

// Definition from the WDK's d3dkmthk.h
typedef enum _D3DKMT_GPU_PREFERENCE_QUERY_STATE : DWORD {
  D3DKMT_GPU_PREFERENCE_STATE_UNINITIALIZED,  ///< The GPU preference isn't initialized.
  D3DKMT_GPU_PREFERENCE_STATE_HIGH_PERFORMANCE,  ///< The highest performing GPU is preferred.
  D3DKMT_GPU_PREFERENCE_STATE_MINIMUM_POWER,  ///< The minimum-powered GPU is preferred.
  D3DKMT_GPU_PREFERENCE_STATE_UNSPECIFIED,  ///< A GPU preference isn't specified.
  D3DKMT_GPU_PREFERENCE_STATE_NOT_FOUND,  ///< A GPU preference isn't found.
  D3DKMT_GPU_PREFERENCE_STATE_USER_SPECIFIED_GPU  ///< A specific GPU is preferred.
} D3DKMT_GPU_PREFERENCE_QUERY_STATE;

#include "display.h"
#include "misc.h"
#include "src/config.h"
#include "src/display_device.h"
#include "src/logging.h"
#include "src/platform/common.h"
#include "src/platform/windows/virtual_display.h"
#include "src/video.h"

namespace platf {
  using namespace std::literals;
}

namespace platf::dxgi {

  namespace detail {
    namespace {
      [[nodiscard]] bool valid_damage_rect(
        const RECT &rect,
        const LONG width,
        const LONG height
      ) noexcept {
        return width > 0 && height > 0 && rect.left >= 0 && rect.top >= 0 &&
               rect.right > rect.left && rect.bottom > rect.top &&
               rect.right <= width && rect.bottom <= height;
      }

      [[nodiscard]] std::uint64_t damage_rect_intersection_area(
        const RECT &a,
        const RECT &b
      ) noexcept {
        const auto width = static_cast<std::int64_t>(
          std::min(a.right, b.right)
        ) - std::max(a.left, b.left);
        const auto height = static_cast<std::int64_t>(
          std::min(a.bottom, b.bottom)
        ) - std::max(a.top, b.top);
        if (width <= 0 || height <= 0) {
          return 0u;
        }
        return static_cast<std::uint64_t>(width) *
               static_cast<std::uint64_t>(height);
      }
    }  // namespace

    ddup_damage_update_t make_ddup_damage_update(
      const std::span<const RECT> dirty_rects,
      const std::span<const DXGI_OUTDUPL_MOVE_RECT> move_rects,
      const LONG width,
      const LONG height,
      const bool metadata_valid
    ) {
      ddup_damage_update_t update;
      if (
        !metadata_valid || width <= 0 || height <= 0 ||
        dirty_rects.size() > ddup_damage_frame_rect_budget ||
        move_rects.size() >
          (ddup_damage_frame_rect_budget - dirty_rects.size()) / 2u
      ) {
        return update;
      }

      update.rects.reserve(dirty_rects.size() + 2u * move_rects.size());
      for (const auto &rect : dirty_rects) {
        if (!valid_damage_rect(rect, width, height)) {
          update.rects.clear();
          return update;
        }
        update.rects.push_back(rect);
      }

      for (const auto &move : move_rects) {
        const auto &destination = move.DestinationRect;
        if (!valid_damage_rect(destination, width, height)) {
          update.rects.clear();
          return update;
        }

        const auto move_width = static_cast<std::int64_t>(destination.right) -
                                destination.left;
        const auto move_height = static_cast<std::int64_t>(destination.bottom) -
                                 destination.top;
        const auto source_right = static_cast<std::int64_t>(move.SourcePoint.x) +
                                  move_width;
        const auto source_bottom = static_cast<std::int64_t>(move.SourcePoint.y) +
                                   move_height;
        if (
          move.SourcePoint.x < 0 || move.SourcePoint.y < 0 ||
          source_right > width || source_bottom > height
        ) {
          update.rects.clear();
          return update;
        }

        const RECT source {
          move.SourcePoint.x,
          move.SourcePoint.y,
          static_cast<LONG>(source_right),
          static_cast<LONG>(source_bottom),
        };
        if (!valid_damage_rect(source, width, height)) {
          update.rects.clear();
          return update;
        }

        update.rects.push_back(source);
        update.rects.push_back(destination);
      }

      update.known = true;
      return update;
    }

    ddup_damage_snapshot_t ddup_damage_history_t::commit(ddup_damage_update_t update) {
      std::lock_guard lock {mutex_};
      if (next_token_ == 0u) {
        return {};
      }

      if (update.rects.size() > ddup_damage_frame_rect_budget) {
        update.known = false;
        update.rects.clear();
      }

      entry_t entry;
      entry.token = next_token_++;
      entry.known = update.known;
      entry.rects = std::move(update.rects);
      retained_rects_ += entry.rects.size();
      entries_.push_back(std::move(entry));

      while (
        entries_.size() > ddup_damage_history_frame_budget ||
        retained_rects_ > ddup_damage_history_rect_budget
      ) {
        retained_rects_ -= entries_.front().rects.size();
        entries_.pop_front();
      }

      return {
        std::static_pointer_cast<const ddup_damage_history_t>(shared_from_this()),
        entries_.back().token,
      };
    }

    ddup_damage_intersection_e ddup_damage_history_t::query(
      const std::uint64_t from_exclusive,
      const std::uint64_t through_inclusive,
      const RECT &region
    ) const {
      const auto coverage = query_coverage(
        from_exclusive,
        through_inclusive,
        region
      );
      if (!coverage.known) {
        return ddup_damage_intersection_e::unknown;
      }
      return coverage.potentially_changed_area == 0u ?
               ddup_damage_intersection_e::unchanged :
               ddup_damage_intersection_e::changed;
    }

    ddup_damage_coverage_t ddup_damage_history_t::query_coverage(
      const std::uint64_t from_exclusive,
      const std::uint64_t through_inclusive,
      const RECT &region,
      const bool collect_max_single_intersection
    ) const {
      ddup_damage_coverage_t result;
      if (
        from_exclusive == 0u || through_inclusive == 0u ||
        from_exclusive > through_inclusive || region.left < 0 || region.top < 0 ||
        region.right <= region.left || region.bottom <= region.top
      ) {
        return result;
      }
      result.region_area =
        static_cast<std::uint64_t>(
          static_cast<std::int64_t>(region.right) - region.left
        ) *
        static_cast<std::uint64_t>(
          static_cast<std::int64_t>(region.bottom) - region.top
        );
      if (from_exclusive == through_inclusive) {
        result.known = true;
        return result;
      }

      std::lock_guard lock {mutex_};
      if (
        entries_.empty() || from_exclusive == UINT64_MAX ||
        from_exclusive + 1u < entries_.front().token ||
        through_inclusive > entries_.back().token
      ) {
        return result;
      }

      const auto first_token = from_exclusive + 1u;
      const auto first_index = static_cast<std::size_t>(
        first_token - entries_.front().token
      );
      const auto count = static_cast<std::size_t>(
        through_inclusive - first_token + 1u
      );
      if (first_index > entries_.size() || count > entries_.size() - first_index) {
        return result;
      }

      for (std::size_t offset = 0u; offset < count; ++offset) {
        const auto &entry = entries_[first_index + offset];
        if (entry.token != first_token + offset) {
          return result;
        }
        if (!entry.known) {
          return result;
        }
        for (const auto &rect : entry.rects) {
          const auto overlap = damage_rect_intersection_area(rect, region);
          if (collect_max_single_intersection) {
            result.max_single_intersection_area =
              std::max(result.max_single_intersection_area, overlap);
          }
          if (result.potentially_changed_area == result.region_area) {
            if (!collect_max_single_intersection) {
              break;
            }
            continue;
          }
          if (overlap >= result.region_area - result.potentially_changed_area) {
            result.potentially_changed_area = result.region_area;
            if (!collect_max_single_intersection) {
              break;
            }
            continue;
          }
          result.potentially_changed_area += overlap;
        }
      }

      result.known = true;
      return result;
    }

    ddup_damage_intersection_e query_ddup_damage_between(
      const std::optional<ddup_damage_snapshot_t> &from,
      const std::optional<ddup_damage_snapshot_t> &through,
      const RECT &region
    ) {
      if (
        !from || !through || !from->history || !through->history ||
        from->history.get() != through->history.get()
      ) {
        return ddup_damage_intersection_e::unknown;
      }
      return through->history->query(from->token, through->token, region);
    }

    ddup_damage_coverage_t query_ddup_damage_coverage_between(
      const std::optional<ddup_damage_snapshot_t> &from,
      const std::optional<ddup_damage_snapshot_t> &through,
      const RECT &region,
      const bool collect_max_single_intersection
    ) {
      if (
        !from || !through || !from->history || !through->history ||
        from->history.get() != through->history.get()
      ) {
        return {};
      }
      return through->history->query_coverage(
        from->token,
        through->token,
        region,
        collect_max_single_intersection
      );
    }

    host_sbs_ddup_reuse_proof_e classify_host_sbs_ddup_reuse(
      const std::optional<std::chrono::steady_clock::time_point> &inferred_content,
      const std::optional<ddup_damage_snapshot_t> &inferred_damage,
      const bool video_region,
      const RECT &input_region,
      const std::optional<std::chrono::steady_clock::time_point> &current_content,
      const std::optional<ddup_damage_snapshot_t> &current_damage
    ) {
      if (!inferred_content || !current_content) {
        return host_sbs_ddup_reuse_proof_e::none;
      }
      if (!video_region) {
        return *inferred_content == *current_content ?
                 host_sbs_ddup_reuse_proof_e::content_clock :
                 host_sbs_ddup_reuse_proof_e::none;
      }
      if (
        query_ddup_damage_between(inferred_damage, current_damage, input_region) !=
        ddup_damage_intersection_e::unchanged
      ) {
        return host_sbs_ddup_reuse_proof_e::none;
      }
      return *inferred_content == *current_content ?
               host_sbs_ddup_reuse_proof_e::content_clock :
               host_sbs_ddup_reuse_proof_e::roi_damage;
    }
  }  // namespace detail

  /**
   * DDAPI-specific initialization goes here.
   */
  int duplication_t::init(display_base_t *display, const ::video::config_t &config) {
    HRESULT status;

    // Capture format will be determined from the first call to AcquireNextFrame()
    display->capture_format = DXGI_FORMAT_UNKNOWN;

    // FIXME: Duplicate output on RX580 in combination with DOOM (2016) --> BSOD
    {
      // IDXGIOutput5 is optional, but can provide improved performance and wide color support
      dxgi::output5_t output5 {};
      status = display->output->QueryInterface(IID_IDXGIOutput5, (void **) &output5);
      if (SUCCEEDED(status)) {
        // Ask the display implementation which formats it supports
        auto supported_formats = display->get_supported_capture_formats();
        if (supported_formats.empty()) {
          BOOST_LOG(warning) << "No compatible capture formats for this encoder"sv;
          return -1;
        }

        // We try this twice, in case we still get an error on reinitialization
        for (int x = 0; x < 2; ++x) {
          // Ensure we can duplicate the current display
          syncThreadDesktop();

          status = output5->DuplicateOutput1((IUnknown *) display->device.get(), 0, supported_formats.size(), supported_formats.data(), &dup);
          if (SUCCEEDED(status)) {
            break;
          }
          std::this_thread::sleep_for(200ms);
        }

        // We don't retry with DuplicateOutput() because we can hit this codepath when we're racing
        // with mode changes and we don't want to accidentally fall back to suboptimal capture if
        // we get unlucky and succeed below.
        if (FAILED(status)) {
          BOOST_LOG(warning) << "DuplicateOutput1 Failed [0x"sv << util::hex(status).to_string_view() << ']';
          return -1;
        }
      } else {
        BOOST_LOG(warning) << "IDXGIOutput5 is not supported by your OS. Capture performance may be reduced."sv;

        dxgi::output1_t output1 {};
        status = display->output->QueryInterface(IID_IDXGIOutput1, (void **) &output1);
        if (FAILED(status)) {
          BOOST_LOG(error) << "Failed to query IDXGIOutput1 from the output"sv;
          return -1;
        }

        for (int x = 0; x < 2; ++x) {
          // Ensure we can duplicate the current display
          syncThreadDesktop();

          status = output1->DuplicateOutput((IUnknown *) display->device.get(), &dup);
          if (SUCCEEDED(status)) {
            break;
          }
          std::this_thread::sleep_for(200ms);
        }

        if (FAILED(status)) {
          BOOST_LOG(error) << "DuplicateOutput Failed [0x"sv << util::hex(status).to_string_view() << ']';
          return -1;
        }
      }
    }

    DXGI_OUTDUPL_DESC dup_desc {};
    dup->GetDesc(&dup_desc);

    BOOST_LOG(info) << "Desktop resolution ["sv << dup_desc.ModeDesc.Width << 'x' << dup_desc.ModeDesc.Height << ']';
    BOOST_LOG(info) << "Desktop format ["sv << display->dxgi_format_to_string(dup_desc.ModeDesc.Format) << ']';

    display->display_refresh_rate = dup_desc.ModeDesc.RefreshRate;
    double display_refresh_rate_decimal = (double) display->display_refresh_rate.Numerator / display->display_refresh_rate.Denominator;
    BOOST_LOG(info) << "Display refresh rate [" << display_refresh_rate_decimal << "Hz]";
    if (display->client_frame_rate_strict.Numerator > 0) {
      const auto numerator = display->client_frame_rate_strict.Numerator;
      const auto denominator = display->client_frame_rate_strict.Denominator;
      BOOST_LOG(info) << "Requested frame rate [" << numerator << '/' << denominator << " exactly "
                      << (static_cast<double>(numerator) / denominator) << "fps]";
    } else {
      BOOST_LOG(info) << "Requested frame rate [" << display->client_frame_rate << "fps]";
    }
    display->display_refresh_rate_rounded = lround(display_refresh_rate_decimal);

    return 0;
  }

  capture_e duplication_t::next_frame(DXGI_OUTDUPL_FRAME_INFO &frame_info, std::chrono::milliseconds timeout, resource_t::pointer *res_p) {
    auto capture_status = release_frame();
    if (capture_status != capture_e::ok) {
      return capture_status;
    }

    auto status = dup->AcquireNextFrame(timeout.count(), &frame_info, res_p);

    switch (status) {
      case S_OK:
        // ProtectedContentMaskedOut seems to semi-randomly be TRUE or FALSE even when protected content
        // is on screen the whole time, so we can't just print when it changes. Instead we'll keep track
        // of the last time we printed the warning and print another if we haven't printed one recently.
        if (frame_info.ProtectedContentMaskedOut && std::chrono::steady_clock::now() > last_protected_content_warning_time + 10s) {
          BOOST_LOG(warning) << "Windows is currently blocking DRM-protected content from capture. You may see black regions where this content would be."sv;
          last_protected_content_warning_time = std::chrono::steady_clock::now();
        }

        has_frame = true;
        return capture_e::ok;
      case DXGI_ERROR_WAIT_TIMEOUT:
        return capture_e::timeout;
      case WAIT_ABANDONED:
      case DXGI_ERROR_ACCESS_LOST:
      case DXGI_ERROR_ACCESS_DENIED:
        return capture_e::reinit;
      default:
        BOOST_LOG(error) << "Couldn't acquire next frame [0x"sv << util::hex(status).to_string_view();
        return capture_e::error;
    }
  }

  detail::ddup_damage_update_t duplication_t::damage_update(
    const DXGI_OUTDUPL_FRAME_INFO &frame_info,
    const DXGI_MODE_ROTATION rotation,
    const LONG width,
    const LONG height
  ) {
    if (
      !dup || rotation != DXGI_MODE_ROTATION_IDENTITY ||
      !detail::ddup_frame_damage_metadata_available(frame_info)
    ) {
      return {};
    }

    // TotalMetadataBufferSize is the API-provided capacity for both lists. Querying either method
    // with a null buffer is invalid on real drivers; follow the Desktop Duplication sample's
    // single-buffer order (moves first, then dirty rectangles in the remaining bytes).
    std::vector<std::byte> metadata(frame_info.TotalMetadataBufferSize);
    UINT dirty_bytes = 0u;
    UINT move_bytes = 0u;
    auto *move_rects = reinterpret_cast<DXGI_OUTDUPL_MOVE_RECT *>(metadata.data());
    auto status = dup->GetFrameMoveRects(
      frame_info.TotalMetadataBufferSize,
      move_rects,
      &move_bytes
    );
    if (
      status != S_OK || move_bytes > frame_info.TotalMetadataBufferSize ||
      move_bytes % sizeof(DXGI_OUTDUPL_MOVE_RECT) != 0u
    ) {
      return {};
    }
    const auto dirty_capacity = frame_info.TotalMetadataBufferSize - move_bytes;
    auto *dirty_rects = reinterpret_cast<RECT *>(metadata.data() + move_bytes);
    if (dirty_capacity != 0u) {
      status = dup->GetFrameDirtyRects(dirty_capacity, dirty_rects, &dirty_bytes);
      if (
        status != S_OK || dirty_bytes > dirty_capacity ||
        dirty_bytes % sizeof(RECT) != 0u
      ) {
        return {};
      }
    }

    return detail::make_ddup_damage_update(
      std::span<const RECT> {dirty_rects, dirty_bytes / sizeof(RECT)},
      std::span<const DXGI_OUTDUPL_MOVE_RECT> {
        move_rects,
        move_bytes / sizeof(DXGI_OUTDUPL_MOVE_RECT),
      },
      width,
      height
    );
  }

  capture_e duplication_t::reset(dup_t::pointer dup_p) {
    auto capture_status = release_frame();

    dup.reset(dup_p);
    return capture_status;
  }

  capture_e duplication_t::release_frame() {
    if (!has_frame) {
      return capture_e::ok;
    }

    auto status = dup->ReleaseFrame();
    has_frame = false;
    switch (status) {
      case S_OK:
        return capture_e::ok;

      case DXGI_ERROR_INVALID_CALL:
        BOOST_LOG(warning) << "Duplication frame already released";
        return capture_e::ok;

      case DXGI_ERROR_ACCESS_LOST:
        return capture_e::reinit;

      default:
        BOOST_LOG(error) << "Error while releasing duplication frame [0x"sv << util::hex(status).to_string_view();
        return capture_e::error;
    }
  }

  duplication_t::~duplication_t() {
    release_frame();
  }

  void display_base_t::set_client_frame_rate(int framerate, int framerate_x100) {
    if (framerate <= 0) {
      return;
    }

    DXGI_RATIONAL strict {};
    if (framerate_x100 > 0) {
      const auto rational = ::video::framerate_x100_to_rational(framerate_x100);
      strict = {
        static_cast<UINT>(rational.num),
        static_cast<UINT>(rational.den),
      };
    }

    {
      std::lock_guard lock(client_frame_rate_mutex);
      if (client_frame_rate == framerate && client_frame_rate_strict.Numerator == strict.Numerator && client_frame_rate_strict.Denominator == strict.Denominator) {
        return;
      }
      client_frame_rate = framerate;
      client_frame_rate_strict = strict;
    }

    // Publish after the values are visible so the capture loop never re-derives a torn cadence.
    client_frame_rate_generation.fetch_add(1, std::memory_order_release);
  }

  capture_e display_base_t::capture(const push_captured_image_cb_t &push_captured_image_cb, const pull_free_image_cb_t &pull_free_image_cb, bool *cursor) {
    return capture_with_pending_work(push_captured_image_cb, pull_free_image_cb, cursor, {});
  }

  capture_e display_base_t::capture_with_pending_work(
    const push_captured_image_cb_t &push_captured_image_cb,
    const pull_free_image_cb_t &pull_free_image_cb,
    bool *cursor,
    const std::function<bool()> &has_pending_work
  ) {
    auto adjust_client_frame_rate = [&]() -> DXGI_RATIONAL {
      int requested_frame_rate;
      DXGI_RATIONAL requested_frame_rate_strict;
      {
        std::lock_guard lock(client_frame_rate_mutex);
        requested_frame_rate = client_frame_rate;
        requested_frame_rate_strict = client_frame_rate_strict;
      }

      if (requested_frame_rate_strict.Numerator > 0) {
        return requested_frame_rate_strict;
      }

      // Adjust capture frame interval when display refresh rate is not integral but very close to requested fps.
      if (display_refresh_rate.Denominator > 1) {
        DXGI_RATIONAL candidate = display_refresh_rate;
        if (requested_frame_rate % display_refresh_rate_rounded == 0) {
          candidate.Numerator *= requested_frame_rate / display_refresh_rate_rounded;
        } else if (display_refresh_rate_rounded % requested_frame_rate == 0) {
          candidate.Denominator *= display_refresh_rate_rounded / requested_frame_rate;
        }
        double candidate_rate = (double) candidate.Numerator / candidate.Denominator;
        // Can only decrease requested fps, otherwise client may start accumulating frames and suffer increased latency.
        if (requested_frame_rate > candidate_rate && candidate_rate / requested_frame_rate > 0.99) {
          BOOST_LOG(info) << "Adjusted capture rate to " << candidate_rate << "fps to better match display";
          return candidate;
        }
      }

      return {(uint32_t) requested_frame_rate, 1};
    };

    auto observed_frame_rate_generation = client_frame_rate_generation.load(std::memory_order_acquire);
    DXGI_RATIONAL client_frame_rate_adjusted = adjust_client_frame_rate();
    std::optional<std::chrono::steady_clock::time_point> frame_pacing_group_start;
    uint32_t frame_pacing_group_frames = 0;

    // Keep the display awake during capture. If the display goes to sleep during
    // capture, best case is that capture stops until it powers back on. However,
    // worst case it will trigger us to reinit DD, waking the display back up in
    // a neverending cycle of waking and sleeping the display of an idle machine.
    SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED);
    auto clear_display_required = util::fail_guard([]() {
      SetThreadExecutionState(ES_CONTINUOUS);
    });

    sleep_overshoot_logger.reset();

    while (true) {
      // A live 0x3007 video-mode change republishes the client cadence while this capture session
      // keeps running. Re-derive the pacing interval and start a fresh pacing group so the new
      // rate takes effect without throwing away the display and its retained-image optimization.
      const auto current_frame_rate_generation = client_frame_rate_generation.load(std::memory_order_acquire);
      if (current_frame_rate_generation != observed_frame_rate_generation) {
        observed_frame_rate_generation = current_frame_rate_generation;
        client_frame_rate_adjusted = adjust_client_frame_rate();
        frame_pacing_group_start = std::nullopt;
        frame_pacing_group_frames = 0;
        BOOST_LOG(info) << "Capture pacing updated to "
                        << ((double) client_frame_rate_adjusted.Numerator / client_frame_rate_adjusted.Denominator)
                        << "fps for a live video-mode change";
      }

      // This will return false if the HDR state changes or for any number of other
      // display or GPU changes. We should reinit to examine the updated state of
      // the display subsystem. It is recommended to call this once per frame.
      if (!factory->IsCurrent()) {
        return platf::capture_e::reinit;
      }

      platf::capture_e status = capture_e::ok;
      std::shared_ptr<img_t> img_out;
      const detail::capture_wait_policy_t wait_policy {
        .pending_local_work = has_pending_work && has_pending_work(),
      };

      // Try to continue frame pacing group, snapshot() is called with zero timeout after waiting for client frame interval
      if (frame_pacing_group_start) {
        const uint32_t seconds = (uint64_t) frame_pacing_group_frames * client_frame_rate_adjusted.Denominator / client_frame_rate_adjusted.Numerator;
        const uint32_t remainder = (uint64_t) frame_pacing_group_frames * client_frame_rate_adjusted.Denominator % client_frame_rate_adjusted.Numerator;
        const auto sleep_target = *frame_pacing_group_start +
                                  std::chrono::nanoseconds(1s) * seconds +
                                  std::chrono::nanoseconds(1s) * remainder / client_frame_rate_adjusted.Numerator;
        const auto sleep_period = wait_policy.pacing_sleep(
          sleep_target - std::chrono::steady_clock::now()
        );

        if (sleep_period <= 0ns) {
          // We missed next frame time, invalidating current frame pacing group
          frame_pacing_group_start = std::nullopt;
          frame_pacing_group_frames = 0;
          status = capture_e::timeout;
        } else {
          bool elastic = false;
          if (sleep_period >= 2ms) {
            elastic = true;
            timer->sleep_for(sleep_period - 2ms);
            if (!wait_policy.pending_local_work) {
              sleep_overshoot_logger.first_point(sleep_target);
              sleep_overshoot_logger.second_point_now_and_log();
            }
          }

          status = snapshot(pull_free_image_cb, img_out, elastic ? 2ms : std::chrono::duration_cast<std::chrono::milliseconds>(sleep_period), *cursor);

          if (status == capture_e::ok && img_out) {
            frame_pacing_group_frames += 1;
          } else {
            frame_pacing_group_start = std::nullopt;
            frame_pacing_group_frames = 0;
          }
        }
      }

      // Start new frame pacing group if necessary, snapshot() is called with non-zero timeout
      if ((status == capture_e::timeout && wait_policy.retry_after_pacing_timeout()) ||
          (status == capture_e::ok && !frame_pacing_group_start)) {
        status = snapshot(pull_free_image_cb, img_out, wait_policy.source_timeout(), *cursor);

        if (status == capture_e::ok && img_out) {
          frame_pacing_group_start = img_out->frame_timestamp;

          if (!frame_pacing_group_start) {
            BOOST_LOG(warning) << "snapshot() provided image without timestamp";
            frame_pacing_group_start = std::chrono::steady_clock::now();
          }

          frame_pacing_group_frames = 1;
        } else if (status == platf::capture_e::timeout) {
          // The D3D11 device is protected by an unfair lock that is held the entire time that
          // IDXGIOutputDuplication::AcquireNextFrame() is running. This is normally harmless,
          // however sometimes the encoding thread needs to interact with our ID3D11Device to
          // create dummy images or initialize the shared state that is used to pass textures
          // between the capture and encoding ID3D11Devices.
          //
          // When we're in a state where we're not actively receiving frames regularly, we will
          // spend almost 100% of our time in AcquireNextFrame() holding that critical lock.
          // Worse still, since it's unfair, we can monopolize it while the encoding thread
          // is starved. The encoding thread may acquire it for a few moments across a few
          // ID3D11Device calls before losing it again to us for another long time waiting in
          // AcquireNextFrame(). The starvation caused by this lock contention causes encoder
          // reinitialization to take several seconds instead of a fraction of a second.
          //
          // To avoid starving the encoding thread, sleep without the lock held for a little
          // while each time we reach our max frame timeout. This will only happen when nothing
          // is updating the display, so no visible stutter should be introduced by the sleep.
          std::this_thread::sleep_for(wait_policy.idle_backoff());
        }
      } else if (status == capture_e::timeout && wait_policy.pending_local_work) {
        // A short pacing probe already timed out. Service retained local work now instead of
        // entering the ordinary idle wait, while still yielding the unfair capture-device lock.
        std::this_thread::sleep_for(wait_policy.idle_backoff());
      }

      switch (status) {
        case platf::capture_e::reinit:
        case platf::capture_e::error:
        case platf::capture_e::interrupted:
          return status;
        case platf::capture_e::timeout:
          if (!push_captured_image_cb(std::move(img_out), false)) {
            return capture_e::ok;
          }
          break;
        case platf::capture_e::ok:
          if (!push_captured_image_cb(std::move(img_out), true)) {
            return capture_e::ok;
          }
          break;
        default:
          BOOST_LOG(error) << "Unrecognized capture status ["sv << (int) status << ']';
          return status;
      }

      status = release_snapshot();
      if (status != platf::capture_e::ok) {
        return status;
      }
    }

    return capture_e::ok;
  }

  /**
   * @brief Tests to determine if the Desktop Duplication API can capture the given output.
   * @details When testing for enumeration only, we avoid resyncing the thread desktop.
   * @param adapter The DXGI adapter to use for capture.
   * @param output The DXGI output to capture.
   * @param enumeration_only Specifies whether this test is occurring for display enumeration.
   */
  bool test_dxgi_duplication(adapter_t &adapter, output_t &output, bool enumeration_only) {
    D3D_FEATURE_LEVEL featureLevels[] {
      D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_11_0,
      D3D_FEATURE_LEVEL_10_1,
      D3D_FEATURE_LEVEL_10_0,
      D3D_FEATURE_LEVEL_9_3,
      D3D_FEATURE_LEVEL_9_2,
      D3D_FEATURE_LEVEL_9_1
    };

    device_t device;
    auto status = D3D11CreateDevice(
      adapter.get(),
      D3D_DRIVER_TYPE_UNKNOWN,
      nullptr,
      D3D11_CREATE_DEVICE_FLAGS,
      featureLevels,
      sizeof(featureLevels) / sizeof(D3D_FEATURE_LEVEL),
      D3D11_SDK_VERSION,
      &device,
      nullptr,
      nullptr
    );
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to create D3D11 device for DD test [0x"sv << util::hex(status).to_string_view() << ']';
      return false;
    }

    output1_t output1;
    status = output->QueryInterface(IID_IDXGIOutput1, (void **) &output1);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to query IDXGIOutput1 from the output"sv;
      return false;
    }

    // Check if we can use the Desktop Duplication API on this output
    for (int x = 0; x < 2; ++x) {
      dup_t dup;

      // Only resynchronize the thread desktop when not enumerating displays.
      // During enumeration, the caller will do this only once to ensure
      // a consistent view of available outputs.
      if (!enumeration_only) {
        syncThreadDesktop();
      }

      status = output1->DuplicateOutput((IUnknown *) device.get(), &dup);
      if (SUCCEEDED(status)) {
        return true;
      }

      // If we're not resyncing the thread desktop and we don't have permission to
      // capture the current desktop, just bail immediately. Retrying won't help.
      if (enumeration_only && status == E_ACCESSDENIED) {
        break;
      } else {
        std::this_thread::sleep_for(200ms);
      }
    }

    BOOST_LOG(error) << "DuplicateOutput() test failed [0x"sv << util::hex(status).to_string_view() << ']';
    return false;
  }

  /**
   * @brief Hook for NtGdiDdDDIGetCachedHybridQueryValue() from win32u.dll.
   * @param gpuPreference A pointer to the location where the preference will be written.
   * @return Always STATUS_SUCCESS if valid arguments are provided.
   */
  NTSTATUS __stdcall NtGdiDdDDIGetCachedHybridQueryValueHook(D3DKMT_GPU_PREFERENCE_QUERY_STATE *gpuPreference) {
    // By faking a cached GPU preference state of D3DKMT_GPU_PREFERENCE_STATE_UNSPECIFIED, this will
    // prevent DXGI from performing the normal GPU preference resolution that looks at the registry,
    // power settings, and the hybrid adapter DDI interface to pick a GPU. Instead, we will not be
    // bound to any specific GPU. This will prevent DXGI from performing output reparenting (moving
    // outputs from their true location to the render GPU), which breaks DDA.
    if (gpuPreference) {
      *gpuPreference = D3DKMT_GPU_PREFERENCE_STATE_UNSPECIFIED;
      return 0;  // STATUS_SUCCESS
    } else {
      return STATUS_INVALID_PARAMETER;
    }
  }

  int display_base_t::init(const ::video::config_t &config, const std::string &display_name, capture_backend_e backend) {
    static std::once_flag windows_cpp_once_flag;

    std::call_once(windows_cpp_once_flag, []() {
      DECLARE_HANDLE(DPI_AWARENESS_CONTEXT);

      typedef BOOL (*User32_SetProcessDpiAwarenessContext)(DPI_AWARENESS_CONTEXT value);

      {
        auto user32 = LoadLibraryA("user32.dll");
        auto f = (User32_SetProcessDpiAwarenessContext) GetProcAddress(user32, "SetProcessDpiAwarenessContext");
        if (f) {
          f(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        }

        FreeLibrary(user32);
      }

      {
        // We aren't calling MH_Uninitialize(), but that's okay because this hook lasts for the life of the process
        MH_Initialize();
        MH_CreateHookApi(L"win32u.dll", "NtGdiDdDDIGetCachedHybridQueryValue", (void *) NtGdiDdDDIGetCachedHybridQueryValueHook, nullptr);
        MH_EnableHook(MH_ALL_HOOKS);
      }
    });

    // Get rectangle of full desktop for absolute mouse coordinates
    env_width = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    env_height = GetSystemMetrics(SM_CYVIRTUALSCREEN);

    HRESULT status;

    status = CreateDXGIFactory1(IID_IDXGIFactory1, (void **) &factory);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to create DXGIFactory1 [0x"sv << util::hex(status).to_string_view() << ']';
      return -1;
    }

    auto adapter_name = from_utf8(config::video.adapter_name);
    auto output_name = from_utf8(display_name);

    adapter_t::pointer adapter_p;
    for (int tries = 0; tries < 2; ++tries) {
      for (int x = 0; factory->EnumAdapters1(x, &adapter_p) != DXGI_ERROR_NOT_FOUND; ++x) {
        dxgi::adapter_t adapter_tmp {adapter_p};

        DXGI_ADAPTER_DESC1 adapter_desc {};
        status = adapter_tmp->GetDesc1(&adapter_desc);
        if (FAILED(status)) {
          BOOST_LOG(warning) << "Failed to query DXGI adapter description [0x"sv
                             << util::hex(status).to_string_view() << ']';
          continue;
        }

        if (!adapter_name.empty() && adapter_desc.Description != adapter_name) {
          continue;
        }

        dxgi::output_t::pointer output_p;
        for (int y = 0; adapter_tmp->EnumOutputs(y, &output_p) != DXGI_ERROR_NOT_FOUND; ++y) {
          dxgi::output_t output_tmp {output_p};

          DXGI_OUTPUT_DESC desc {};
          status = output_tmp->GetDesc(&desc);
          if (FAILED(status)) {
            BOOST_LOG(warning) << "Failed to query DXGI output description [0x"sv
                               << util::hex(status).to_string_view() << ']';
            continue;
          }

          if (!output_name.empty() && desc.DeviceName != output_name) {
            continue;
          }

          const bool backend_available = backend == capture_backend_e::wgc ||
                                         test_dxgi_duplication(adapter_tmp, output_tmp, false);
          if (desc.AttachedToDesktop && backend_available) {
            output = std::move(output_tmp);

            offset_x = desc.DesktopCoordinates.left;
            offset_y = desc.DesktopCoordinates.top;
            width = desc.DesktopCoordinates.right - offset_x;
            height = desc.DesktopCoordinates.bottom - offset_y;

            display_rotation = desc.Rotation;
            if (display_rotation == DXGI_MODE_ROTATION_ROTATE90 || display_rotation == DXGI_MODE_ROTATION_ROTATE270) {
              width_before_rotation = height;
              height_before_rotation = width;
            } else {
              width_before_rotation = width;
              height_before_rotation = height;
            }

            // left and bottom may be negative, yet absolute mouse coordinates start at 0x0
            // Ensure offset starts at 0x0
            offset_x -= GetSystemMetrics(SM_XVIRTUALSCREEN);
            offset_y -= GetSystemMetrics(SM_YVIRTUALSCREEN);

            break;
          }
        }

        if (output) {
          adapter = std::move(adapter_tmp);
          break;
        }
      }

      if (output) {
        break;
      }

      // If we made it here without finding an output, try to power on the display and retry.
      if (tries == 0) {
        SetThreadExecutionState(ES_DISPLAY_REQUIRED);
        Sleep(500);
      }
    }

    if (!output) {
      BOOST_LOG(error) << "Failed to locate an output device"sv;
      return -1;
    }

    D3D_FEATURE_LEVEL featureLevels[] {
      D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_11_0,
      D3D_FEATURE_LEVEL_10_1,
      D3D_FEATURE_LEVEL_10_0,
      D3D_FEATURE_LEVEL_9_3,
      D3D_FEATURE_LEVEL_9_2,
      D3D_FEATURE_LEVEL_9_1
    };

    status = adapter->QueryInterface(IID_IDXGIAdapter, (void **) &adapter_p);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to query IDXGIAdapter interface"sv;
      return -1;
    }

    status = D3D11CreateDevice(
      adapter_p,
      D3D_DRIVER_TYPE_UNKNOWN,
      nullptr,
      D3D11_CREATE_DEVICE_FLAGS,
      featureLevels,
      sizeof(featureLevels) / sizeof(D3D_FEATURE_LEVEL),
      D3D11_SDK_VERSION,
      &device,
      &feature_level,
      &device_ctx
    );

    adapter_p->Release();

    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to create D3D11 device [0x"sv << util::hex(status).to_string_view() << ']';

      return -1;
    }

    DXGI_ADAPTER_DESC adapter_desc {};
    status = adapter->GetDesc(&adapter_desc);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to query selected DXGI adapter description [0x"sv
                       << util::hex(status).to_string_view() << ']';
      return -1;
    }

    auto description = to_utf8(adapter_desc.Description);
    BOOST_LOG(info)
      << std::endl
      << "Device Description : " << description << std::endl
      << "Device Vendor ID   : 0x"sv << util::hex(adapter_desc.VendorId).to_string_view() << std::endl
      << "Device Device ID   : 0x"sv << util::hex(adapter_desc.DeviceId).to_string_view() << std::endl
      << "Device Video Mem   : "sv << adapter_desc.DedicatedVideoMemory / 1048576 << " MiB"sv << std::endl
      << "Device Sys Mem     : "sv << adapter_desc.DedicatedSystemMemory / 1048576 << " MiB"sv << std::endl
      << "Share Sys Mem      : "sv << adapter_desc.SharedSystemMemory / 1048576 << " MiB"sv << std::endl
      << "Feature Level      : 0x"sv << util::hex(feature_level).to_string_view() << std::endl
      << "Capture size       : "sv << width << 'x' << height << std::endl
      << "Offset             : "sv << offset_x << 'x' << offset_y << std::endl
      << "Virtual Desktop    : "sv << env_width << 'x' << env_height;

    // Bump up thread priority
    {
      const DWORD flags = TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY;
      TOKEN_PRIVILEGES tp;
      HANDLE token;
      LUID val;

      if (OpenProcessToken(GetCurrentProcess(), flags, &token) && !!LookupPrivilegeValue(nullptr, SE_INC_BASE_PRIORITY_NAME, &val)) {
        tp.PrivilegeCount = 1;
        tp.Privileges[0].Luid = val;
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

        if (!AdjustTokenPrivileges(token, false, &tp, sizeof(tp), nullptr, nullptr)) {
          BOOST_LOG(warning) << "Could not set privilege to increase GPU priority";
        }
      }

      CloseHandle(token);

      HMODULE gdi32 = GetModuleHandleA("GDI32");
      if (gdi32) {
        auto check_hags = [&](const LUID &adapter) -> bool {
          auto d3dkmt_open_adapter = (PD3DKMTOpenAdapterFromLuid) GetProcAddress(gdi32, "D3DKMTOpenAdapterFromLuid");
          auto d3dkmt_query_adapter_info = (PD3DKMTQueryAdapterInfo) GetProcAddress(gdi32, "D3DKMTQueryAdapterInfo");
          auto d3dkmt_close_adapter = (PD3DKMTCloseAdapter) GetProcAddress(gdi32, "D3DKMTCloseAdapter");
          if (!d3dkmt_open_adapter || !d3dkmt_query_adapter_info || !d3dkmt_close_adapter) {
            BOOST_LOG(error) << "Couldn't load d3dkmt functions from gdi32.dll to determine GPU HAGS status";
            return false;
          }

          D3DKMT_OPENADAPTERFROMLUID d3dkmt_adapter = {adapter};
          if (FAILED(d3dkmt_open_adapter(&d3dkmt_adapter))) {
            BOOST_LOG(error) << "D3DKMTOpenAdapterFromLuid() failed while trying to determine GPU HAGS status";
            return false;
          }

          bool result;

          D3DKMT_WDDM_2_7_CAPS d3dkmt_adapter_caps = {};
          D3DKMT_QUERYADAPTERINFO d3dkmt_adapter_info = {};
          d3dkmt_adapter_info.hAdapter = d3dkmt_adapter.hAdapter;
          d3dkmt_adapter_info.Type = KMTQAITYPE_WDDM_2_7_CAPS;
          d3dkmt_adapter_info.pPrivateDriverData = &d3dkmt_adapter_caps;
          d3dkmt_adapter_info.PrivateDriverDataSize = sizeof(d3dkmt_adapter_caps);

          if (SUCCEEDED(d3dkmt_query_adapter_info(&d3dkmt_adapter_info))) {
            result = d3dkmt_adapter_caps.HwSchEnabled;
          } else {
            BOOST_LOG(warning) << "D3DKMTQueryAdapterInfo() failed while trying to determine GPU HAGS status";
            result = false;
          }

          D3DKMT_CLOSEADAPTER d3dkmt_close_adapter_wrap = {d3dkmt_adapter.hAdapter};
          if (FAILED(d3dkmt_close_adapter(&d3dkmt_close_adapter_wrap))) {
            BOOST_LOG(error) << "D3DKMTCloseAdapter() failed while trying to determine GPU HAGS status";
          }

          return result;
        };

        auto d3dkmt_set_process_priority = (PD3DKMTSetProcessSchedulingPriorityClass) GetProcAddress(gdi32, "D3DKMTSetProcessSchedulingPriorityClass");
        if (d3dkmt_set_process_priority) {
          auto priority = D3DKMT_SCHEDULINGPRIORITYCLASS_REALTIME;
          bool hags_enabled = check_hags(adapter_desc.AdapterLuid);
          if (adapter_desc.VendorId == 0x10DE) {
            // As of 2023.07, NVIDIA driver has unfixed bug(s) where "realtime" can cause unrecoverable encoding freeze or outright driver crash
            // This issue happens more frequently with HAGS, in DX12 games or when VRAM is filled close to max capacity
            // Track OBS to see if they find better workaround or NVIDIA fixes it on their end, they seem to be in communication
            if (hags_enabled && !config::video.nv_realtime_hags) {
              priority = D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH;
            }
          }
          BOOST_LOG(info) << "Active GPU has HAGS " << (hags_enabled ? "enabled" : "disabled");
          BOOST_LOG(info) << "Using " << (priority == D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH ? "high" : "realtime") << " GPU priority";
          if (FAILED(d3dkmt_set_process_priority(GetCurrentProcess(), priority))) {
            BOOST_LOG(warning) << "Failed to adjust GPU priority. Please run application as administrator for optimal performance.";
          }
        } else {
          BOOST_LOG(error) << "Couldn't load D3DKMTSetProcessSchedulingPriorityClass function from gdi32.dll to adjust GPU priority";
        }
      }

      dxgi::dxgi_t dxgi;
      status = device->QueryInterface(IID_IDXGIDevice, (void **) &dxgi);
      if (FAILED(status)) {
        BOOST_LOG(warning) << "Failed to query DXGI interface from device [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }

      status = dxgi->SetGPUThreadPriority(0x4000001E);
      if (FAILED(status)) {
        BOOST_LOG(info) << "Failed to request absoloute capture GPU thread priority. Trying relative priority.";
        status = dxgi->SetGPUThreadPriority(7);
        if (FAILED(status)) {
          BOOST_LOG(warning) << "Failed to request relative capture GPU thread priority. Please run application as administrator for optimal performance.";
        } else {
          BOOST_LOG(info) << "Relative capture GPU thread priority request success.";
        }
      }
    }

    // Try to reduce latency
    {
      dxgi::dxgi1_t dxgi {};
      status = device->QueryInterface(IID_IDXGIDevice, (void **) &dxgi);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to query DXGI interface from device [0x"sv << util::hex(status).to_string_view() << ']';
        return -1;
      }

      status = dxgi->SetMaximumFrameLatency(1);
      if (FAILED(status)) {
        BOOST_LOG(warning) << "Failed to set maximum frame latency [0x"sv << util::hex(status).to_string_view() << ']';
      }
    }

    client_frame_rate = config.framerate;
    client_frame_rate_strict = {};
    if (config.framerateX100 > 0) {
      const auto framerate = ::video::framerate_x100_to_rational(config.framerateX100);
      client_frame_rate_strict = {
        static_cast<UINT>(framerate.num),
        static_cast<UINT>(framerate.den),
      };
    }
    dxgi::output6_t output6 {};
    status = output->QueryInterface(IID_IDXGIOutput6, (void **) &output6);
    if (SUCCEEDED(status)) {
      DXGI_OUTPUT_DESC1 desc1 {};
      status = output6->GetDesc1(&desc1);
      if (SUCCEEDED(status)) {
        BOOST_LOG(info)
          << std::endl
          << "Colorspace         : "sv << colorspace_to_string(desc1.ColorSpace) << std::endl
          << "Bits Per Color     : "sv << desc1.BitsPerColor << std::endl
          << "Red Primary        : ["sv << desc1.RedPrimary[0] << ',' << desc1.RedPrimary[1] << ']' << std::endl
          << "Green Primary      : ["sv << desc1.GreenPrimary[0] << ',' << desc1.GreenPrimary[1] << ']' << std::endl
          << "Blue Primary       : ["sv << desc1.BluePrimary[0] << ',' << desc1.BluePrimary[1] << ']' << std::endl
          << "White Point        : ["sv << desc1.WhitePoint[0] << ',' << desc1.WhitePoint[1] << ']' << std::endl
          << "Min Luminance      : "sv << desc1.MinLuminance << " nits"sv << std::endl
          << "Max Luminance      : "sv << desc1.MaxLuminance << " nits"sv << std::endl
          << "Max Full Luminance : "sv << desc1.MaxFullFrameLuminance << " nits"sv;
      } else {
        BOOST_LOG(warning) << "Failed to query IDXGIOutput6 description [0x"sv << util::hex(status).to_string_view() << ']';
      }
    }

    if (!timer || !*timer) {
      BOOST_LOG(error) << "Uninitialized high precision timer";
      return -1;
    }

    return 0;
  }

  bool display_base_t::is_hdr() {
    dxgi::output6_t output6 {};

    auto status = output->QueryInterface(IID_IDXGIOutput6, (void **) &output6);
    if (FAILED(status)) {
      BOOST_LOG(warning) << "Failed to query IDXGIOutput6 from the output"sv;
      return false;
    }

    DXGI_OUTPUT_DESC1 desc1 {};
    status = output6->GetDesc1(&desc1);
    if (FAILED(status)) {
      BOOST_LOG(warning) << "Failed to query IDXGIOutput6 description [0x"sv << util::hex(status).to_string_view() << ']';
      return false;
    }

    return desc1.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
  }

  bool display_base_t::get_hdr_metadata(SS_HDR_METADATA &metadata) {
    dxgi::output6_t output6 {};

    std::memset(&metadata, 0, sizeof(metadata));

    auto status = output->QueryInterface(IID_IDXGIOutput6, (void **) &output6);
    if (FAILED(status)) {
      BOOST_LOG(warning) << "Failed to query IDXGIOutput6 from the output"sv;
      return false;
    }

    DXGI_OUTPUT_DESC1 desc1 {};
    status = output6->GetDesc1(&desc1);
    if (FAILED(status)) {
      BOOST_LOG(warning) << "Failed to query IDXGIOutput6 HDR metadata [0x"sv << util::hex(status).to_string_view() << ']';
      return false;
    }

    // The primaries reported here seem to correspond to scRGB (Rec. 709)
    // which we then convert to Rec 2020 in our scRGB FP16 -> PQ shader
    // prior to encoding. It's not clear to me if we're supposed to report
    // the primaries of the original colorspace or the one we've converted
    // it to, but let's just report Rec 2020 primaries and D65 white level
    // to avoid confusing clients by reporting Rec 709 primaries with a
    // Rec 2020 colorspace. It seems like most clients ignore the primaries
    // in the metadata anyway (luminance range is most important).
    desc1.RedPrimary[0] = 0.708f;
    desc1.RedPrimary[1] = 0.292f;
    desc1.GreenPrimary[0] = 0.170f;
    desc1.GreenPrimary[1] = 0.797f;
    desc1.BluePrimary[0] = 0.131f;
    desc1.BluePrimary[1] = 0.046f;
    desc1.WhitePoint[0] = 0.3127f;
    desc1.WhitePoint[1] = 0.3290f;

    metadata.displayPrimaries[0].x = desc1.RedPrimary[0] * 50000;
    metadata.displayPrimaries[0].y = desc1.RedPrimary[1] * 50000;
    metadata.displayPrimaries[1].x = desc1.GreenPrimary[0] * 50000;
    metadata.displayPrimaries[1].y = desc1.GreenPrimary[1] * 50000;
    metadata.displayPrimaries[2].x = desc1.BluePrimary[0] * 50000;
    metadata.displayPrimaries[2].y = desc1.BluePrimary[1] * 50000;

    metadata.whitePoint.x = desc1.WhitePoint[0] * 50000;
    metadata.whitePoint.y = desc1.WhitePoint[1] * 50000;

    metadata.maxDisplayLuminance = desc1.MaxLuminance;
    metadata.minDisplayLuminance = desc1.MinLuminance * 10000;

    // These are content-specific metadata parameters that this interface doesn't give us
    metadata.maxContentLightLevel = 0;
    metadata.maxFrameAverageLightLevel = 0;

    metadata.maxFullFrameLuminance = desc1.MaxFullFrameLuminance;

    return true;
  }

  std::optional<float> display_base_t::get_sdr_white_nits() {
    DXGI_OUTPUT_DESC output_desc {};
    if (!output || FAILED(output->GetDesc(&output_desc))) {
      return std::nullopt;
    }

    std::vector<DISPLAYCONFIG_PATH_INFO> paths;
    std::vector<DISPLAYCONFIG_MODE_INFO> modes;
    if (!VDISPLAY::queryActiveDisplayConfig(paths, modes)) {
      return std::nullopt;
    }

    for (const auto &path : paths) {
      DISPLAYCONFIG_SOURCE_DEVICE_NAME source_name {};
      source_name.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
      source_name.header.size = sizeof(source_name);
      source_name.header.adapterId = path.sourceInfo.adapterId;
      source_name.header.id = path.sourceInfo.id;
      if (DisplayConfigGetDeviceInfo(&source_name.header) != ERROR_SUCCESS || std::wstring_view(source_name.viewGdiDeviceName) != std::wstring_view(output_desc.DeviceName)) {
        continue;
      }

      DISPLAYCONFIG_SDR_WHITE_LEVEL white_level {};
      white_level.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
      white_level.header.size = sizeof(white_level);
      white_level.header.adapterId = path.targetInfo.adapterId;
      white_level.header.id = path.targetInfo.id;
      if (DisplayConfigGetDeviceInfo(&white_level.header) != ERROR_SUCCESS) {
        return std::nullopt;
      }

      // Windows encodes SDR reference white relative to 80 nits, with 1000 == 80 nits.
      return (float) white_level.SDRWhiteLevel * 80.0f / 1000.0f;
    }

    return std::nullopt;
  }

  const char *format_str[] = {
    "DXGI_FORMAT_UNKNOWN",
    "DXGI_FORMAT_R32G32B32A32_TYPELESS",
    "DXGI_FORMAT_R32G32B32A32_FLOAT",
    "DXGI_FORMAT_R32G32B32A32_UINT",
    "DXGI_FORMAT_R32G32B32A32_SINT",
    "DXGI_FORMAT_R32G32B32_TYPELESS",
    "DXGI_FORMAT_R32G32B32_FLOAT",
    "DXGI_FORMAT_R32G32B32_UINT",
    "DXGI_FORMAT_R32G32B32_SINT",
    "DXGI_FORMAT_R16G16B16A16_TYPELESS",
    "DXGI_FORMAT_R16G16B16A16_FLOAT",
    "DXGI_FORMAT_R16G16B16A16_UNORM",
    "DXGI_FORMAT_R16G16B16A16_UINT",
    "DXGI_FORMAT_R16G16B16A16_SNORM",
    "DXGI_FORMAT_R16G16B16A16_SINT",
    "DXGI_FORMAT_R32G32_TYPELESS",
    "DXGI_FORMAT_R32G32_FLOAT",
    "DXGI_FORMAT_R32G32_UINT",
    "DXGI_FORMAT_R32G32_SINT",
    "DXGI_FORMAT_R32G8X24_TYPELESS",
    "DXGI_FORMAT_D32_FLOAT_S8X24_UINT",
    "DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS",
    "DXGI_FORMAT_X32_TYPELESS_G8X24_UINT",
    "DXGI_FORMAT_R10G10B10A2_TYPELESS",
    "DXGI_FORMAT_R10G10B10A2_UNORM",
    "DXGI_FORMAT_R10G10B10A2_UINT",
    "DXGI_FORMAT_R11G11B10_FLOAT",
    "DXGI_FORMAT_R8G8B8A8_TYPELESS",
    "DXGI_FORMAT_R8G8B8A8_UNORM",
    "DXGI_FORMAT_R8G8B8A8_UNORM_SRGB",
    "DXGI_FORMAT_R8G8B8A8_UINT",
    "DXGI_FORMAT_R8G8B8A8_SNORM",
    "DXGI_FORMAT_R8G8B8A8_SINT",
    "DXGI_FORMAT_R16G16_TYPELESS",
    "DXGI_FORMAT_R16G16_FLOAT",
    "DXGI_FORMAT_R16G16_UNORM",
    "DXGI_FORMAT_R16G16_UINT",
    "DXGI_FORMAT_R16G16_SNORM",
    "DXGI_FORMAT_R16G16_SINT",
    "DXGI_FORMAT_R32_TYPELESS",
    "DXGI_FORMAT_D32_FLOAT",
    "DXGI_FORMAT_R32_FLOAT",
    "DXGI_FORMAT_R32_UINT",
    "DXGI_FORMAT_R32_SINT",
    "DXGI_FORMAT_R24G8_TYPELESS",
    "DXGI_FORMAT_D24_UNORM_S8_UINT",
    "DXGI_FORMAT_R24_UNORM_X8_TYPELESS",
    "DXGI_FORMAT_X24_TYPELESS_G8_UINT",
    "DXGI_FORMAT_R8G8_TYPELESS",
    "DXGI_FORMAT_R8G8_UNORM",
    "DXGI_FORMAT_R8G8_UINT",
    "DXGI_FORMAT_R8G8_SNORM",
    "DXGI_FORMAT_R8G8_SINT",
    "DXGI_FORMAT_R16_TYPELESS",
    "DXGI_FORMAT_R16_FLOAT",
    "DXGI_FORMAT_D16_UNORM",
    "DXGI_FORMAT_R16_UNORM",
    "DXGI_FORMAT_R16_UINT",
    "DXGI_FORMAT_R16_SNORM",
    "DXGI_FORMAT_R16_SINT",
    "DXGI_FORMAT_R8_TYPELESS",
    "DXGI_FORMAT_R8_UNORM",
    "DXGI_FORMAT_R8_UINT",
    "DXGI_FORMAT_R8_SNORM",
    "DXGI_FORMAT_R8_SINT",
    "DXGI_FORMAT_A8_UNORM",
    "DXGI_FORMAT_R1_UNORM",
    "DXGI_FORMAT_R9G9B9E5_SHAREDEXP",
    "DXGI_FORMAT_R8G8_B8G8_UNORM",
    "DXGI_FORMAT_G8R8_G8B8_UNORM",
    "DXGI_FORMAT_BC1_TYPELESS",
    "DXGI_FORMAT_BC1_UNORM",
    "DXGI_FORMAT_BC1_UNORM_SRGB",
    "DXGI_FORMAT_BC2_TYPELESS",
    "DXGI_FORMAT_BC2_UNORM",
    "DXGI_FORMAT_BC2_UNORM_SRGB",
    "DXGI_FORMAT_BC3_TYPELESS",
    "DXGI_FORMAT_BC3_UNORM",
    "DXGI_FORMAT_BC3_UNORM_SRGB",
    "DXGI_FORMAT_BC4_TYPELESS",
    "DXGI_FORMAT_BC4_UNORM",
    "DXGI_FORMAT_BC4_SNORM",
    "DXGI_FORMAT_BC5_TYPELESS",
    "DXGI_FORMAT_BC5_UNORM",
    "DXGI_FORMAT_BC5_SNORM",
    "DXGI_FORMAT_B5G6R5_UNORM",
    "DXGI_FORMAT_B5G5R5A1_UNORM",
    "DXGI_FORMAT_B8G8R8A8_UNORM",
    "DXGI_FORMAT_B8G8R8X8_UNORM",
    "DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM",
    "DXGI_FORMAT_B8G8R8A8_TYPELESS",
    "DXGI_FORMAT_B8G8R8A8_UNORM_SRGB",
    "DXGI_FORMAT_B8G8R8X8_TYPELESS",
    "DXGI_FORMAT_B8G8R8X8_UNORM_SRGB",
    "DXGI_FORMAT_BC6H_TYPELESS",
    "DXGI_FORMAT_BC6H_UF16",
    "DXGI_FORMAT_BC6H_SF16",
    "DXGI_FORMAT_BC7_TYPELESS",
    "DXGI_FORMAT_BC7_UNORM",
    "DXGI_FORMAT_BC7_UNORM_SRGB",
    "DXGI_FORMAT_AYUV",
    "DXGI_FORMAT_Y410",
    "DXGI_FORMAT_Y416",
    "DXGI_FORMAT_NV12",
    "DXGI_FORMAT_P010",
    "DXGI_FORMAT_P016",
    "DXGI_FORMAT_420_OPAQUE",
    "DXGI_FORMAT_YUY2",
    "DXGI_FORMAT_Y210",
    "DXGI_FORMAT_Y216",
    "DXGI_FORMAT_NV11",
    "DXGI_FORMAT_AI44",
    "DXGI_FORMAT_IA44",
    "DXGI_FORMAT_P8",
    "DXGI_FORMAT_A8P8",
    "DXGI_FORMAT_B4G4R4A4_UNORM",

    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,

    "DXGI_FORMAT_P208",
    "DXGI_FORMAT_V208",
    "DXGI_FORMAT_V408"
  };

  const char *display_base_t::dxgi_format_to_string(DXGI_FORMAT format) {
    return format_str[format];
  }

  const char *display_base_t::colorspace_to_string(DXGI_COLOR_SPACE_TYPE type) {
    const char *type_str[] = {
      "DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709",
      "DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709",
      "DXGI_COLOR_SPACE_RGB_STUDIO_G22_NONE_P709",
      "DXGI_COLOR_SPACE_RGB_STUDIO_G22_NONE_P2020",
      "DXGI_COLOR_SPACE_RESERVED",
      "DXGI_COLOR_SPACE_YCBCR_FULL_G22_NONE_P709_X601",
      "DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P601",
      "DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P601",
      "DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709",
      "DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709",
      "DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P2020",
      "DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P2020",
      "DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020",
      "DXGI_COLOR_SPACE_YCBCR_STUDIO_G2084_LEFT_P2020",
      "DXGI_COLOR_SPACE_RGB_STUDIO_G2084_NONE_P2020",
      "DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_TOPLEFT_P2020",
      "DXGI_COLOR_SPACE_YCBCR_STUDIO_G2084_TOPLEFT_P2020",
      "DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P2020",
      "DXGI_COLOR_SPACE_YCBCR_STUDIO_GHLG_TOPLEFT_P2020",
      "DXGI_COLOR_SPACE_YCBCR_FULL_GHLG_TOPLEFT_P2020",
      "DXGI_COLOR_SPACE_RGB_STUDIO_G24_NONE_P709",
      "DXGI_COLOR_SPACE_RGB_STUDIO_G24_NONE_P2020",
      "DXGI_COLOR_SPACE_YCBCR_STUDIO_G24_LEFT_P709",
      "DXGI_COLOR_SPACE_YCBCR_STUDIO_G24_LEFT_P2020",
      "DXGI_COLOR_SPACE_YCBCR_STUDIO_G24_TOPLEFT_P2020",
    };

    if (type < ARRAYSIZE(type_str)) {
      return type_str[type];
    } else {
      return "UNKNOWN";
    }
  }

}  // namespace platf::dxgi

namespace platf {
  /**
   * Pick a display adapter and native D3D11 capture method.
   */
  std::shared_ptr<display_t> display(
    const std::string &display_name,
    const video::config_t &config,
    capture_backend_e preferred_backend
  ) {
    auto open_backend = [&](capture_backend_e backend) -> std::shared_ptr<display_t> {
      if (backend == capture_backend_e::ddup) {
        auto disp = std::make_shared<dxgi::display_ddup_vram_t>();
        return !disp->init(config, display_name) ? std::move(disp) : nullptr;
      }

      auto disp = std::make_shared<dxgi::display_wgc_vram_t>();
      return !disp->init(config, display_name) ? std::move(disp) : nullptr;
    };

    if (auto disp = open_backend(preferred_backend)) {
      return disp;
    }
    const auto fallback_backend = preferred_backend == capture_backend_e::ddup ?
                                    capture_backend_e::wgc :
                                    capture_backend_e::ddup;
    if (auto disp = open_backend(fallback_backend)) {
      return disp;
    }

    // DDUP and WGC failed.
    return nullptr;
  }

  std::vector<std::string> display_names() {
    std::vector<std::string> display_names;

    HRESULT status;

    BOOST_LOG(debug) << "Detecting monitors..."sv;

    // We sync the thread desktop once before we start the enumeration process
    // to ensure test_dxgi_duplication() returns consistent results for all GPUs
    // even if the current desktop changes during our enumeration process.
    // It is critical that we either fully succeed in enumeration or fully fail,
    // otherwise it can lead to the capture code switching monitors unexpectedly.
    syncThreadDesktop();

    dxgi::factory1_t factory;
    status = CreateDXGIFactory1(IID_IDXGIFactory1, (void **) &factory);
    if (FAILED(status)) {
      BOOST_LOG(error) << "Failed to create DXGIFactory1 [0x"sv << util::hex(status).to_string_view() << ']';
      return {};
    }

    for (int x = 0;; ++x) {
      dxgi::adapter_t::pointer adapter_p {};
      status = factory->EnumAdapters1(x, &adapter_p);
      if (status == DXGI_ERROR_NOT_FOUND) {
        break;
      }
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to enumerate DXGI adapter [0x"sv << util::hex(status).to_string_view() << ']';
        return {};
      }

      dxgi::adapter_t adapter {adapter_p};
      DXGI_ADAPTER_DESC1 adapter_desc {};
      status = adapter->GetDesc1(&adapter_desc);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to query DXGI adapter [0x"sv << util::hex(status).to_string_view() << ']';
        return {};
      }

      BOOST_LOG(debug)
        << std::endl
        << "====== ADAPTER ====="sv << std::endl
        << "Device Name      : "sv << to_utf8(adapter_desc.Description) << std::endl
        << "Device Vendor ID : 0x"sv << util::hex(adapter_desc.VendorId).to_string_view() << std::endl
        << "Device Device ID : 0x"sv << util::hex(adapter_desc.DeviceId).to_string_view() << std::endl
        << "Device Video Mem : "sv << adapter_desc.DedicatedVideoMemory / 1048576 << " MiB"sv << std::endl
        << "Device Sys Mem   : "sv << adapter_desc.DedicatedSystemMemory / 1048576 << " MiB"sv << std::endl
        << "Share Sys Mem    : "sv << adapter_desc.SharedSystemMemory / 1048576 << " MiB"sv << std::endl
        << std::endl
        << "    ====== OUTPUT ======"sv << std::endl;

      for (int y = 0;; ++y) {
        dxgi::output_t::pointer output_p {};
        status = adapter->EnumOutputs(y, &output_p);
        if (status == DXGI_ERROR_NOT_FOUND) {
          break;
        }
        if (FAILED(status)) {
          BOOST_LOG(error) << "Failed to enumerate DXGI output [0x"sv << util::hex(status).to_string_view() << ']';
          return {};
        }

        dxgi::output_t output {output_p};

        DXGI_OUTPUT_DESC desc {};
        status = output->GetDesc(&desc);
        if (FAILED(status)) {
          BOOST_LOG(error) << "Failed to query DXGI output [0x"sv << util::hex(status).to_string_view() << ']';
          return {};
        }

        auto device_name = to_utf8(desc.DeviceName);

        auto width = desc.DesktopCoordinates.right - desc.DesktopCoordinates.left;
        auto height = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;

        BOOST_LOG(debug)
          << "    Output Name       : "sv << device_name << std::endl
          << "    AttachedToDesktop : "sv << (desc.AttachedToDesktop ? "yes"sv : "no"sv) << std::endl
          << "    Resolution        : "sv << width << 'x' << height << std::endl
          << std::endl;

        // Keep outputs supported by either automatic capture backend. WGC must remain
        // discoverable even when Desktop Duplication cannot open the monitor.
        if (desc.AttachedToDesktop && (dxgi::test_dxgi_duplication(adapter, output, true) || dxgi::test_wgc_capture(output))) {
          display_names.emplace_back(std::move(device_name));
        }
      }
    }

    return display_names;
  }

  /**
   * @brief Returns if GPUs/drivers have changed since the last call to this function.
   * @return `true` if a change has occurred or if it is unknown whether a change occurred.
   */
  bool needs_encoder_reenumeration() {
    // Serialize access to the static DXGI factory
    static std::mutex reenumeration_state_lock;
    auto lg = std::lock_guard(reenumeration_state_lock);

    // Keep a reference to the DXGI factory, which will keep track of changes internally.
    static dxgi::factory1_t factory;
    if (!factory || !factory->IsCurrent()) {
      factory.reset();

      auto status = CreateDXGIFactory1(IID_IDXGIFactory1, (void **) &factory);
      if (FAILED(status)) {
        BOOST_LOG(error) << "Failed to create DXGIFactory1 [0x"sv << util::hex(status).to_string_view() << ']';
        factory.release();
      }

      // Always request reenumeration on the first streaming session just to ensure we
      // can deal with any initialization races that may occur when the system is booting.
      BOOST_LOG(info) << "Encoder reenumeration is required"sv;
      return true;
    } else {
      // The DXGI factory from last time is still current, so no encoder changes have occurred.
      return false;
    }
  }
}  // namespace platf
