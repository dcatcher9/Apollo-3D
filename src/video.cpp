/**
 * @file src/video.cpp
 * @brief Definitions for video.
 */
// standard includes
#include <atomic>
#include <list>
#include <mutex>
#include <optional>
#include <thread>

// local includes
#include "config.h"
#include "display_device.h"
#include "globals.h"
#include "input.h"
#include "logging.h"
#include "nvenc/nvenc_base.h"
#include "platform/common.h"
#include "process.h"
#include "sync.h"
#include "video.h"

#ifdef _WIN32
  #include "platform/windows/virtual_display.h"
#endif

using namespace std::literals;

namespace video {

  // Debug frame-dump request flag (declared in video.h); set by the 0x3004 control message
  // handler in stream.cpp, consumed by display_vram's SBS convert().
  std::atomic<bool> sbs_debug_dump_pending {false};

  sbs_output_dimensions_t host_sbs_output_dimensions(
    int base_width,
    int base_height,
    int video_format,
    int configured_max_width,
    int runtime_max_width
  ) {
    // These are the capabilities reported by NV_ENC_CAPS_WIDTH_MAX on the production RTX 5080.
    // The runtime value remains authoritative and may reduce either conservative default.
    const int codec_max_width = video_format == 1 || video_format == 2 ? 8192 : 4096;
    int effective_max_width = std::min(configured_max_width, codec_max_width);
    if (runtime_max_width > 0) {
      effective_max_width = std::min(effective_max_width, runtime_max_width);
    }
    const int capped_width = std::max(2, effective_max_width) & ~1;
    const std::int64_t packed_width = static_cast<std::int64_t>(base_width) * 2;
    if (packed_width <= capped_width) {
      return {static_cast<int>(packed_width), base_height};
    }

    const int scaled_height = std::max(
      2,
      static_cast<int>(std::lround(
        static_cast<double>(base_height) * capped_width / packed_width
      )) &
        ~1
    );
    return {capped_width, scaled_height};
  }

  platf::capture_backend_e capture_backend_failover_t::preferred_backend() const noexcept {
    return preferred_backend_;
  }

  void capture_backend_failover_t::reset() noexcept {
    preferred_backend_ = platf::capture_backend_e::ddup;
    early_ddup_failures_ = 0;
  }

  void capture_backend_failover_t::note_backend_opened(platf::capture_backend_e backend) noexcept {
    if (backend == platf::capture_backend_e::wgc) {
      preferred_backend_ = platf::capture_backend_e::wgc;
    }
  }

  void capture_backend_failover_t::note_capture_result(
    platf::capture_backend_e backend,
    platf::capture_e result,
    std::uint64_t captured_frames,
    std::chrono::steady_clock::duration lifetime
  ) noexcept {
    if (backend != platf::capture_backend_e::ddup || (result != platf::capture_e::reinit && result != platf::capture_e::error)) {
      return;
    }

    constexpr auto stable_lifetime = 2s;
    constexpr std::uint64_t stable_frame_count = 120;
    const bool stable = captured_frames >= stable_frame_count ||
                        (captured_frames > 0 && lifetime >= stable_lifetime);
    if (stable) {
      early_ddup_failures_ = 0;
      return;
    }

    if (++early_ddup_failures_ >= 2) {
      preferred_backend_ = platf::capture_backend_e::wgc;
    }
  }

  // Resolve the profile-configured model name against the registry, else synthesize a custom
  // entry from the sbs_3d_depth_model/_url escape hatch.
  config::depth_model_info depth_model_for_profile(const config::video_t::sbs_t &profile) {
    for (const auto &m : config::depth_model_registry()) {
      if (m.name == profile.depth_model) {
        return m;
      }
    }
    config::depth_model_info custom;
    custom.name = profile.depth_model;
    custom.url = profile.depth_model_url;
    return custom;
  }

  config::depth_model_info active_depth_model() {
    return depth_model_for_profile(config::video.sbs);
  }

  /**
   * @brief Check if we can allow probing for the encoders.
   * @return True if there should be no issues with the probing, false if we should prevent it.
   */
  bool allow_encoder_probing() {
    const auto devices {display_device::enumerate_devices()};

    // // If there are no devices, then either the API is not working correctly or OS does not support the lib.
    // // Either way we should not block the probing in this case as we can't tell what's wrong.
    // if (devices.empty()) {
    //   return true;
    // }

    if (devices.empty()) {
#ifdef _WIN32
      // We'll create a temporary virtual display for probing anyways.
      if (proc::vDisplayDriverStatus == VDISPLAY::DRIVER_STATUS::OK) {
        return false;
      }
#endif
      return true;
    }

    // Since Windows 11 24H2, it is possible that there will be no active devices present
    // for some reason (probably a bug). Trying to probe encoders in such a state locks/breaks the DXGI
    // and also the display device for Windows. So we must have at least 1 active device.
    const bool at_least_one_device_is_active = std::any_of(std::begin(devices), std::end(devices), [](const auto &device) {
      // If device has additional info, it is active.
      return static_cast<bool>(device.m_info);
    });

    if (at_least_one_device_is_active) {
      return true;
    }

    BOOST_LOG(error) << "No display devices are active at the moment! Cannot probe the encoders.";
    return false;
  }

  class nvenc_encode_session_t: public encode_session_t {
  public:
    nvenc_encode_session_t(std::unique_ptr<platf::nvenc_encode_device_t> encode_device):
        device(std::move(encode_device)) {
    }

    int convert(platf::img_t &img) override {
      if (!device) {
        return -1;
      }
      return device->convert(img);
    }

    void request_idr_frame() override {
      force_idr = true;
    }

    void request_normal_frame() override {
      force_idr = false;
    }

    void invalidate_ref_frames(int64_t first_frame, int64_t last_frame) override {
      if (!device || !device->nvenc) {
        return;
      }

      if (!device->nvenc->invalidate_ref_frames(first_frame, last_frame)) {
        force_idr = true;
      }
    }

    nvenc::nvenc_encoded_frame encode_frame(uint64_t frame_index) {
      if (!device || !device->nvenc) {
        return {};
      }

      auto result = device->nvenc->encode_frame(frame_index, force_idr);
      force_idr = false;
      return result;
    }

  private:
    std::unique_ptr<platf::nvenc_encode_device_t> device;
    bool force_idr = false;
  };

  struct capture_ctx_t {
    img_event_t images;
    config_t config;
  };

  struct capture_thread_async_ctx_t {
    std::shared_ptr<safe::queue_t<capture_ctx_t>> capture_ctx_queue;
    std::thread capture_thread;

    safe::signal_t reinit_event;
    sync_util::sync_t<std::weak_ptr<platf::display_t>> display_wp;
  };

  int start_capture_async(capture_thread_async_ctx_t &ctx);
  void end_capture_async(capture_thread_async_ctx_t &ctx);

#ifndef _WIN32
  #error Apollo requires Windows with native NVIDIA NVENC support
#endif

  namespace {
    constexpr std::string_view h264_encoder_name = "h264_nvenc"sv;
    constexpr std::string_view hevc_encoder_name = "hevc_nvenc"sv;
    constexpr std::string_view av1_encoder_name = "av1_nvenc"sv;

    constexpr std::uint32_t nvenc_h264 = 1U << 0;
    constexpr std::uint32_t nvenc_hevc = 1U << 1;
    constexpr std::uint32_t nvenc_hevc_hdr = 1U << 2;
    constexpr std::uint32_t nvenc_av1 = 1U << 3;
    constexpr std::uint32_t nvenc_av1_hdr = 1U << 4;

    std::atomic_uint32_t nvenc_capabilities {0};
    std::mutex nvenc_probe_mutex;

    std::string_view nvenc_codec_name(int video_format) {
      switch (video_format) {
        case 1:
          return hevc_encoder_name;
        case 2:
          return av1_encoder_name;
        default:
          if (video_format != 0) {
            BOOST_LOG(error) << "Unknown video format " << video_format << ", falling back to H.264";
          }
          return h264_encoder_name;
      }
    }
  }  // namespace

  nvenc_capabilities_t nvenc_capabilities_snapshot() noexcept {
    const auto capabilities = nvenc_capabilities.load(std::memory_order_acquire);
    return {
      (capabilities & nvenc_hevc) != 0,
      (capabilities & nvenc_hevc_hdr) != 0,
      (capabilities & nvenc_av1) != 0,
      (capabilities & nvenc_av1_hdr) != 0,
    };
  }

  void reset_display(
    std::shared_ptr<platf::display_t> &disp,
    const std::string &display_name,
    const config_t &config,
    platf::capture_backend_e preferred_backend = platf::capture_backend_e::ddup
  ) {
    // We try this twice, in case we still get an error on reinitialization
    for (int x = 0; x < 2; ++x) {
      disp.reset();
      disp = platf::display(display_name, config, preferred_backend);
      if (disp) {
        break;
      }

      // The capture code depends on us to sleep between failures
      std::this_thread::sleep_for(200ms);
    }
  }

  /**
   * @brief Update the list of display names before or during a stream.
   * @details This will attempt to keep `current_display_index` pointing at the same display.
   * @param display_names The list of display names to repopulate.
   * @param current_display_index The current display index or -1 if not yet known.
   */
  void refresh_displays(std::vector<std::string> &display_names, int &current_display_index, std::string &preferred_display_name) {
    // It is possible that the output name may be empty even if it wasn't before (device disconnected) or vice-versa
    const auto output_name {display_device::map_output_name(config::video.output_name)};
    std::string current_display_name = preferred_display_name;

    // If we have a current display index, let's start with that
    if (current_display_name.empty() && current_display_index >= 0 && current_display_index < display_names.size()) {
      current_display_name = display_names.at(current_display_index);
    }

    // Refresh the display names
    auto old_display_names = std::move(display_names);
    display_names = platf::display_names();

    // If we now have no displays, let's put the old display array back and fail
    if (display_names.empty() && !old_display_names.empty()) {
      BOOST_LOG(error) << "No displays were found after reenumeration!"sv;
      display_names = std::move(old_display_names);
      return;
    } else if (display_names.empty()) {
      display_names.emplace_back(output_name);
    }

    // We now have a new display name list, so reset the index back to 0
    current_display_index = 0;

    if (current_display_name.empty()) {
      current_display_name = display_device::map_output_name(config::video.output_name);
    }

    // If we had a name previously, let's try to find it in the new list
    if (!current_display_name.empty()) {
      for (int x = 0; x < display_names.size(); ++x) {
        if (display_names[x] == current_display_name) {
          current_display_index = x;
          return;
        }
      }

      // The old display was removed, so we'll start back at the first display again
      BOOST_LOG(warning) << "Previous active display ["sv << current_display_name << "] is no longer present"sv;
    } else {
      for (int x = 0; x < display_names.size(); ++x) {
        if (display_names[x] == output_name) {
          current_display_index = x;
          return;
        }
      }
    }
  }

  void refresh_displays(std::vector<std::string> &display_names, int &current_display_index) {
    static std::string empty_str = "";
    refresh_displays(display_names, current_display_index, empty_str);
  }

  void captureThread(
    std::shared_ptr<safe::queue_t<capture_ctx_t>> capture_ctx_queue,
    sync_util::sync_t<std::weak_ptr<platf::display_t>> &display_wp,
    safe::signal_t &reinit_event
  ) {
    std::optional<capture_ctx_t> capture_ctx;

    auto fg = util::fail_guard([&]() {
      capture_ctx_queue->stop();

      if (capture_ctx) {
        capture_ctx->images->stop();
      }
      for (auto &queued_ctx : capture_ctx_queue->unsafe()) {
        queued_ctx.images->stop();
      }
    });

    // Wait for the initial capture context or a request to stop the queue
    auto initial_capture_ctx = capture_ctx_queue->pop();
    if (!initial_capture_ctx) {
      return;
    }
    capture_ctx.emplace(std::move(*initial_capture_ctx));

    std::vector<std::string> display_names;
    int display_p = -1;
    std::shared_ptr<platf::display_t> disp;
    capture_backend_failover_t capture_failover;
    std::string active_display_name;
    const auto initial_display_name = proc::proc.get_display_name();
    if (!initial_display_name.empty()) {
      active_display_name = initial_display_name;
      disp = platf::display(
        initial_display_name,
        capture_ctx->config,
        capture_failover.preferred_backend()
      );
      if (disp) {
        capture_failover.note_backend_opened(disp->capture_backend());
      }
    }
    if (!disp) {
      // Get all the monitor names now, rather than at boot, to
      // get the most up-to-date list available monitors
      refresh_displays(display_names, display_p);
      if (active_display_name != display_names[display_p]) {
        capture_failover.reset();
        active_display_name = display_names[display_p];
      }
      disp = platf::display(
        active_display_name,
        capture_ctx->config,
        capture_failover.preferred_backend()
      );
      if (disp) {
        capture_failover.note_backend_opened(disp->capture_backend());
        proc::proc.set_display_name(active_display_name);
      } else {
        return;
      }
    }

    display_wp = disp;

    constexpr auto capture_buffer_size = 12;
    std::list<std::shared_ptr<platf::img_t>> imgs(capture_buffer_size);

    std::vector<std::optional<std::chrono::steady_clock::time_point>> imgs_used_timestamps;
    const std::chrono::seconds trim_timeot = 3s;
    auto trim_imgs = [&]() {
      // count allocated and used within current pool
      size_t allocated_count = 0;
      size_t used_count = 0;
      for (const auto &img : imgs) {
        if (img) {
          allocated_count += 1;
          if (img.use_count() > 1) {
            used_count += 1;
          }
        }
      }

      // remember the timestamp of currently used count
      const auto now = std::chrono::steady_clock::now();
      if (imgs_used_timestamps.size() <= used_count) {
        imgs_used_timestamps.resize(used_count + 1);
      }
      imgs_used_timestamps[used_count] = now;

      // decide whether to trim allocated unused above the currently used count
      // based on last used timestamp and universal timeout
      size_t trim_target = used_count;
      for (size_t i = used_count; i < imgs_used_timestamps.size(); i++) {
        if (imgs_used_timestamps[i] && now - *imgs_used_timestamps[i] < trim_timeot) {
          trim_target = i;
        }
      }

      // trim allocated unused above the newly decided trim target
      if (allocated_count > trim_target) {
        size_t to_trim = allocated_count - trim_target;
        // prioritize trimming least recently used
        for (auto it = imgs.rbegin(); it != imgs.rend(); it++) {
          auto &img = *it;
          if (img && img.use_count() == 1) {
            img.reset();
            to_trim -= 1;
            if (to_trim == 0) {
              break;
            }
          }
        }
        // forget timestamps that no longer relevant
        imgs_used_timestamps.resize(trim_target + 1);
      }
    };

    auto pull_free_image_callback = [&](std::shared_ptr<platf::img_t> &img_out) -> bool {
      img_out.reset();
      while (capture_ctx_queue->running()) {
        // pick first allocated but unused
        for (auto it = imgs.begin(); it != imgs.end(); it++) {
          if (*it && it->use_count() == 1) {
            img_out = *it;
            if (it != imgs.begin()) {
              // move image to the front of the list to prioritize its reusal
              imgs.erase(it);
              imgs.push_front(img_out);
            }
            break;
          }
        }
        // otherwise pick first unallocated
        if (!img_out) {
          for (auto it = imgs.begin(); it != imgs.end(); it++) {
            if (!*it) {
              // allocate image
              *it = disp->alloc_img();
              img_out = *it;
              if (it != imgs.begin()) {
                // move image to the front of the list to prioritize its reusal
                imgs.erase(it);
                imgs.push_front(img_out);
              }
              break;
            }
          }
        }
        if (img_out) {
          // trim allocated but unused portion of the pool based on timeouts
          trim_imgs();
          img_out->frame_timestamp.reset();
          return true;
        } else {
          // sleep and retry if image pool is full
          std::this_thread::sleep_for(1ms);
        }
      }
      return false;
    };

    // Capture takes place on this thread
    platf::adjust_thread_priority(platf::thread_priority_e::critical);

    bool capture_cursor = true;
    while (capture_ctx_queue->running()) {
      std::uint64_t captured_frames = 0;
      auto push_captured_image_callback = [&](std::shared_ptr<platf::img_t> &&img, bool frame_captured) -> bool {
        if (!capture_ctx_queue->running() || !capture_ctx->images->running()) {
          return false;
        }

        if (frame_captured) {
          ++captured_frames;
          capture_ctx->images->raise(std::move(img));
        }

        return true;
      };

      const auto backend = disp->capture_backend();
      const auto capture_started = std::chrono::steady_clock::now();
      auto status = disp->capture(push_captured_image_callback, pull_free_image_callback, &capture_cursor);
      const auto previous_preference = capture_failover.preferred_backend();
      capture_failover.note_capture_result(
        backend,
        status,
        captured_frames,
        std::chrono::steady_clock::now() - capture_started
      );
      if (previous_preference != capture_failover.preferred_backend()) {
        BOOST_LOG(warning) << "Desktop Duplication failed repeatedly before stable capture; "sv
                              "using Windows.Graphics.Capture for the rest of this session."sv;
      }

      switch (status) {
        case platf::capture_e::error:
          BOOST_LOG(warning) << "Display capture failed; rebuilding the automatic capture backend."sv;
          [[fallthrough]];
        case platf::capture_e::reinit:
          {
            reinit_event.raise(true);

            // Some classes of images contain references to the display --> display won't delete unless img is deleted
            for (auto &img : imgs) {
              img.reset();
            }

            // display_wp is modified in this thread only
            // Wait for the other shared_ptr's of display to be destroyed.
            // New displays will only be created in this thread.
            while (display_wp->use_count() != 1) {
              // Free images that weren't consumed by the encoder. These can reference the display and prevent
              // the ref count from reaching 1. We do this here rather than on the encoder thread to avoid race
              // conditions where the encoding loop might free a good frame after reinitializing if we capture
              // a new frame here before the encoder has finished reinitializing.
              while (capture_ctx->images->try_pop()) {
              }

              if (!capture_ctx->images->running()) {
                return;
              }

              std::this_thread::sleep_for(20ms);
            }

            while (capture_ctx_queue->running()) {
              // Release the display before reenumerating displays, since some capture backends
              // only support a single display session per device/application.
              disp.reset();

              // Refresh display names since a display removal might have caused the reinitialization
              auto preferred_display_name = proc::proc.get_display_name();
              refresh_displays(display_names, display_p, preferred_display_name);

              if (active_display_name != display_names[display_p]) {
                capture_failover.reset();
                active_display_name = display_names[display_p];
              }

              // reset_display() will sleep between retries
              reset_display(
                disp,
                active_display_name,
                capture_ctx->config,
                capture_failover.preferred_backend()
              );
              if (disp) {
                capture_failover.note_backend_opened(disp->capture_backend());
                proc::proc.set_display_name(active_display_name);
                break;
              }
            }
            if (!disp) {
              return;
            }

            display_wp = disp;

            reinit_event.reset();
            continue;
          }
        case platf::capture_e::ok:
        case platf::capture_e::timeout:
        case platf::capture_e::interrupted:
          return;
        default:
          BOOST_LOG(error) << "Unrecognized capture status ["sv << (int) status << ']';
          return;
      }
    }
  }

  int encode(int64_t frame_nr, nvenc_encode_session_t &session, safe::mail_raw_t::queue_t<packet_t> &packets, const std::shared_ptr<void> &channel_data, std::optional<std::chrono::steady_clock::time_point> frame_timestamp) {
    auto encoded_frame = session.encode_frame(frame_nr);
    if (encoded_frame.data.empty()) {
      BOOST_LOG(error) << "NvENC returned empty packet";
      return -1;
    }

    if (frame_nr != encoded_frame.frame_index) {
      BOOST_LOG(error) << "NvENC frame index mismatch " << frame_nr << " " << encoded_frame.frame_index;
    }

    auto packet = std::make_unique<packet_raw_generic>(std::move(encoded_frame.data), encoded_frame.frame_index, encoded_frame.idr);
    packet->channel_data = channel_data;
    packet->after_ref_frame_invalidation = encoded_frame.after_ref_frame_invalidation;
    packet->frame_timestamp = frame_timestamp;
    packets->raise(std::move(packet));

    return 0;
  }

  std::unique_ptr<nvenc_encode_session_t> make_encode_session(const config_t &client_config, std::unique_ptr<platf::nvenc_encode_device_t> encode_device) {
    if (!encode_device->init_encoder(client_config, encode_device->colorspace)) {
      return nullptr;
    }

    return std::make_unique<nvenc_encode_session_t>(std::move(encode_device));
  }

  void refresh_mouse_keys_if_due(std::chrono::steady_clock::time_point &next_refresh) {
    const auto now = std::chrono::steady_clock::now();
    if (now < next_refresh) {
      return;
    }

    next_refresh = now + 1s;
    platf::refresh_mouse_keys();
  }

  bool encode_run(
    int &frame_nr,  // Store progress of the frame number
    safe::mail_t mail,
    img_event_t images,
    config_t config,
    std::shared_ptr<platf::display_t> disp,
    std::unique_ptr<platf::nvenc_encode_device_t> encode_device,
    safe::signal_t &reinit_event,
    std::shared_ptr<void> channel_data
  ) {
    auto session = make_encode_session(config, std::move(encode_device));
    if (!session) {
      return false;
    }

    // As a workaround for NVENC hangs and to generally speed up encoder reinit,
    // we will complete the encoder teardown in a separate thread if supported.
    // This will move expensive processing off the encoder thread to allow us
    // to restart encoding as soon as possible. For cases where the NVENC driver
    // hang occurs, this thread may probably never exit, but it will allow
    // streaming to continue without requiring a full restart of Sunshine.
    auto fail_guard = util::fail_guard([&session] {
      std::thread encoder_teardown_thread {[session = std::move(session)]() mutable {
        BOOST_LOG(info) << "Starting async encoder teardown";
        session.reset();
        BOOST_LOG(info) << "Async encoder teardown complete";
      }};
      encoder_teardown_thread.detach();
    });

    // set max frame time based on client-requested target framerate.
    double minimum_fps_target = (config::video.minimum_fps_target > 0.0) ? config::video.minimum_fps_target * 1000 : std::max(config.encodingFramerate / 5, 10000);
    auto max_frametime = std::chrono::nanoseconds(1000ms) * 1000 / minimum_fps_target;
    auto encode_frame_threshold = std::chrono::nanoseconds(1000ms) * 1000 / config.encodingFramerate;
    auto frame_variation_threshold = encode_frame_threshold / 4;
    auto min_frame_diff = encode_frame_threshold - frame_variation_threshold;
    BOOST_LOG(info) << "Minimum FPS target set to ~"sv << (minimum_fps_target / 1000) << "fps ("sv << max_frametime << ")"sv;
    BOOST_LOG(info) << "Encoding Frame threshold: "sv << encode_frame_threshold;

    auto shutdown_event = mail->event<bool>(mail::shutdown);
    auto packets = mail::man->queue<packet_t>(mail::video_packets);
    auto idr_events = mail->event<bool>(mail::idr);
    auto invalidate_ref_frames_events = mail->event<std::pair<int64_t, int64_t>>(mail::invalidate_ref_frames);
    // A pending host SBS mode change means we must rebuild the encode session at the new
    // resolution. We only peek here; capture_async pops it and applies the new mode.
    auto sbs_mode_event = mail->event<int>(mail::sbs_mode);
    auto depth_pipeline_ready_event = config.sbs_depth_pipeline_ready_event;

    {
      // Load a dummy image into the encoder input to ensure we have something to encode
      // even if we timeout waiting on the first frame. This is a relatively large
      // allocation which can be freed immediately after convert(), so we do this
      // in a separate scope.
      auto dummy_img = disp->alloc_img();
      if (!dummy_img || disp->dummy_img(dummy_img.get()) || session->convert(*dummy_img)) {
        return true;
      }
    }

    std::chrono::steady_clock::time_point encode_frame_timestamp;
    auto next_mouse_keys_refresh = std::chrono::steady_clock::now() + 1s;
    bool missing_frame_timestamp_warning_logged = false;

    // Most recent real captured frame. On a host SBS toggle the display and its capture session
    // survive, so no new frame is delivered until
    // the desktop actually changes; the fresh session would encode its dummy (black) prime at
    // min-FPS until then. Re-queue this frame on rebuild so the next session starts with the
    // current desktop instead.
    std::shared_ptr<platf::img_t> last_img;

    auto lifecycle_change_requested = [&]() {
      const bool shutting_down = shutdown_event->peek();
      const bool capture_stopped = !images->running();
      const bool display_reinit_pending = reinit_event.peek();
      const bool sbs_mode_change_pending = sbs_mode_event->peek();

      // If capture has to reinitialize before it has produced a frame, encode the dummy once so
      // Artemis knows the host is alive. An SBS-only change always rebuilds immediately because
      // it changes the encode dimensions.
      if (!shutting_down && !capture_stopped && !(display_reinit_pending && frame_nr > 1) && !sbs_mode_change_pending) {
        return false;
      }

      // Same-display SBS rebuild: hand the current desktop to the next session. Never retain an
      // image across a real display reinitialization because it may own resources from the old
      // display. Do not overwrite a newer frame that capture has already queued.
      if (last_img && !shutting_down && !capture_stopped && !display_reinit_pending && sbs_mode_change_pending) {
        images->try_raise(std::move(last_img));
      }

      return true;
    };

    while (true) {
      if (lifecycle_change_requested()) {
        break;
      }

      bool requested_idr_frame = false;

      while (invalidate_ref_frames_events->peek()) {
        if (auto frames = invalidate_ref_frames_events->pop(0ms)) {
          session->invalidate_ref_frames(frames->first, frames->second);
        }
      }

      if (idr_events->peek()) {
        requested_idr_frame = true;
        idr_events->pop();
      }

      if (requested_idr_frame) {
        session->request_idr_frame();
      }

      std::optional<std::chrono::steady_clock::time_point> frame_timestamp;
      bool converted_frame = false;

      // Encode at a minimum FPS to avoid image quality issues with static content
      if (!requested_idr_frame || images->peek()) {
        const auto image_wait = depth_pipeline_ready_event && depth_pipeline_ready_event->peek() ?
                                  0ns :
                                  max_frametime;
        if (auto img = images->pop(image_wait)) {
          last_img = img;
          frame_timestamp = img->frame_timestamp;
          if (!frame_timestamp) {
            if (!missing_frame_timestamp_warning_logged) {
              BOOST_LOG(warning) << "Encoder received image without frame timestamp; substituting steady_clock::now()"sv;
              missing_frame_timestamp_warning_logged = true;
            }
            frame_timestamp = std::chrono::steady_clock::now();
          }

          // Re-check after the potentially blocking image wait and before conversion. On Windows,
          // conversion can include the full D3D11/TensorRT SBS pipeline, so avoid starting it after
          // capture teardown or an encode-dimension change has been requested.
          if (lifecycle_change_requested()) {
            break;
          }

          auto current_timestamp = *frame_timestamp;
          auto time_diff = current_timestamp - encode_frame_timestamp;

          // If new frame comes in way too fast, just drop
          if (time_diff < -frame_variation_threshold) {
            continue;
          }

          if (session->convert(*img)) {
            BOOST_LOG(error) << "Could not convert image"sv;
            break;
          }
          converted_frame = true;

          if (time_diff < frame_variation_threshold) {
            *frame_timestamp = encode_frame_timestamp;
          } else {
            encode_frame_timestamp = current_timestamp;
          }

          encode_frame_timestamp += encode_frame_threshold;
        } else if (!images->running()) {
          break;
        }
      }

      // Host SBS initializes only its per-stream D3D/CUDA resources in the background; the model
      // engine and execution context are already process-resident. If initialization completes
      // while the desktop is static, reconvert the retained source once so the ready pipeline is
      // installed and the client receives depth immediately instead of waiting for desktop motion.
      if (!converted_frame && depth_pipeline_ready_event && depth_pipeline_ready_event->peek() && last_img) {
        if (lifecycle_change_requested()) {
          break;
        }
        frame_timestamp = std::chrono::steady_clock::now();
        if (session->convert(*last_img)) {
          BOOST_LOG(error) << "Could not activate the initialized Host SBS GPU pipeline"sv;
          break;
        }
        converted_frame = true;
      }

      if (converted_frame && depth_pipeline_ready_event && depth_pipeline_ready_event->peek()) {
        depth_pipeline_ready_event->pop(0ms);
      }

      // Keep this check as close as possible to encode(). A reinit may be requested while the
      // current frame is being converted, and encoding afterward can leave packets or GPU work in
      // flight while the session and display resources are torn down.
      if (lifecycle_change_requested()) {
        break;
      }

      if (encode(frame_nr++, *session, packets, channel_data, frame_timestamp)) {
        BOOST_LOG(error) << "Could not encode video packet"sv;
        break;
      }

      session->request_normal_frame();
      refresh_mouse_keys_if_due(next_mouse_keys_refresh);
    }
    return true;
  }

  input::touch_port_t make_port(platf::display_t *display, const config_t &config) {
    float wd = display->width;
    float hd = display->height;

    float wt = config.width;
    float ht = config.height;

    auto scalar = std::fminf(wt / wd, ht / hd);

    auto w2 = scalar * wd;
    auto h2 = scalar * hd;

    auto offsetX = (config.width - w2) * 0.5f;
    auto offsetY = (config.height - h2) * 0.5f;

    return input::touch_port_t {
      {
        display->offset_x,
        display->offset_y,
        config.width,
        config.height,
      },
      display->env_width,
      display->env_height,
      offsetX,
      offsetY,
      1.0f / scalar,
    };
  }

  std::unique_ptr<platf::nvenc_encode_device_t> make_encode_device(platf::display_t &disp, const config_t &config) {
    auto colorspace = colorspace_from_client_config(config, disp.is_hdr());

    const auto pix_fmt = colorspace.bit_depth == 10 ? platf::pix_fmt_e::p010 : platf::pix_fmt_e::nv12;

    {
      const auto encoder_name = nvenc_codec_name(config.videoFormat);

      BOOST_LOG(info) << "Creating encoder " << logging::bracket(std::string {encoder_name});

      auto color_coding = colorspace.colorspace == colorspace_e::bt2020    ? "HDR (Rec. 2020 + SMPTE 2084 PQ)" :
                          colorspace.colorspace == colorspace_e::rec601    ? "SDR (Rec. 601)" :
                          colorspace.colorspace == colorspace_e::rec709    ? "SDR (Rec. 709)" :
                          colorspace.colorspace == colorspace_e::bt2020sdr ? "SDR (Rec. 2020)" :
                                                                             "unknown";

      BOOST_LOG(info) << "Color coding: " << color_coding;
      BOOST_LOG(info) << "Color depth: " << colorspace.bit_depth << "-bit";
      BOOST_LOG(info) << "Color range: " << (colorspace.full_range ? "JPEG" : "MPEG");
    }

    auto result = disp.make_nvenc_encode_device(pix_fmt);

    if (result) {
      result->colorspace = colorspace;
    }

    return result;
  }

  void capture_async(
    safe::mail_t mail,
    config_t &config,
    std::shared_ptr<void> channel_data
  ) {
    auto shutdown_event = mail->event<bool>(mail::shutdown);

    auto images = std::make_shared<img_event_t::element_type>();
    auto lg = util::fail_guard([&]() {
      images->stop();
      shutdown_event->raise(true);
    });

    capture_thread_async_ctx_t capture_thread_ctx;
    if (start_capture_async(capture_thread_ctx)) {
      return;
    }
    auto capture_guard = util::fail_guard([&]() {
      end_capture_async(capture_thread_ctx);
    });

    capture_thread_ctx.capture_ctx_queue->raise(capture_ctx_t {images, config});

    if (!capture_thread_ctx.capture_ctx_queue->running()) {
      return;
    }

    int frame_nr = 1;

    auto touch_port_event = mail->event<input::touch_port_t>(mail::touch_port);
    auto hdr_event = mail->event<hdr_info_t>(mail::hdr);

    // Host SBS toggle (0x3003 control message). The client-negotiated width is the "base"
    // width; when SBS is on we double it so the encoder emits a 2W x H side-by-side frame.
    auto sbs_mode_event = mail->event<int>(mail::sbs_mode);
    auto sbs_depth_status_event = mail->event<int>(mail::sbs_depth_status);
    const int base_width = config.width;
    int current_sbs_mode = config.sbs_mode;

    // Encoding takes place on this thread
    platf::adjust_thread_priority(platf::thread_priority_e::high);

    while (!shutdown_event->peek() && images->running()) {
      // Wait for the main capture event when the display is being reinitialized
      if (capture_thread_ctx.reinit_event.peek()) {
        std::this_thread::sleep_for(20ms);
        continue;
      }
      // Wait for the display to be ready
      std::shared_ptr<platf::display_t> display;
      {
        auto lg = capture_thread_ctx.display_wp.lock();
        if (capture_thread_ctx.display_wp->expired()) {
          continue;
        }

        display = capture_thread_ctx.display_wp->lock();
      }

      // Apply the latest requested host SBS mode (drain to the most recent value).
      while (sbs_mode_event->peek()) {
        if (auto m = sbs_mode_event->pop(0ms)) {
          current_sbs_mode = *m;
        }
      }
      // Build the effective config for this encode session. SBS doubles the output width to
      // 2*base. If that exceeds the encoder's max width, cap the packed width and scale the
      // height proportionally to preserve the per-eye aspect. The SBS pipeline then renders
      // directly at this capped target (see display_vram init_output), so the down-resolution
      // is folded into the depth warp's color sampling rather than a post-warp resample.
      config_t session_config = config;
      session_config.sbs_mode = current_sbs_mode;
      session_config.sbs_depth_status_event = sbs_depth_status_event;
      session_config.sbs_depth_pipeline_ready_event =
        std::make_shared<safe::event_t<bool>>();
      session_config.sbs_config = config::video.sbs;
      int runtime_max_width = nvenc::max_encode_width_for_codec(session_config.videoFormat).value_or(0);
      if (current_sbs_mode != SBS_OFF) {
        const auto dimensions = host_sbs_output_dimensions(
          base_width,
          config.height,
          session_config.videoFormat,
          session_config.sbs_config.max_encode_width,
          runtime_max_width
        );
        session_config.width = dimensions.width;
        session_config.height = dimensions.height;
        const std::int64_t packed_width = static_cast<std::int64_t>(base_width) * 2;
        if (session_config.width != packed_width) {
          BOOST_LOG(info) << "Host SBS: requested packed width "sv << packed_width
                          << " exceeds the effective encoder width limit; capping to "sv
                          << session_config.width << 'x'
                          << session_config.height << " (per-eye "sv
                          << (session_config.width / 2) << 'x' << session_config.height << ')';
        }
      } else {
        session_config.width = base_width;
      }
      BOOST_LOG(info) << "Encode session: host SBS mode "sv
                      << current_sbs_mode << ", profile '"sv
                      << session_config.sbs_config.profile << "'"sv
                      << ", output "sv << session_config.width << 'x' << session_config.height;

      auto recover_failed_sbs_session = [&]() {
        if (session_config.sbs_mode == SBS_OFF) {
          return false;
        }
        const int refreshed_max_width = nvenc::max_encode_width_for_codec(
                                          session_config.videoFormat
        )
                                          .value_or(0);
        if (refreshed_max_width > 0 && refreshed_max_width < session_config.width && refreshed_max_width != runtime_max_width) {
          BOOST_LOG(info) << "Host SBS learned a lower runtime NVENC width limit ("sv
                          << refreshed_max_width
                          << "); retrying with aspect-preserving SBS scaling."sv;
          return true;
        }
        BOOST_LOG(error) << "Failed to create encoder at codec-safe SBS resolution "sv
                         << session_config.width << 'x' << session_config.height
                         << "; refusing to silently replace the requested SBS stream with flat 2D."sv;
        sbs_depth_status_event->raise(0);
        return false;
      };

      auto encode_device = make_encode_device(*display, session_config);
      if (!encode_device) {
        if (recover_failed_sbs_session()) {
          continue;
        }
        return;
      }

      // absolute mouse coordinates require that the dimensions of the screen are known
      // Host SBS changes only the encoded frame geometry. Absolute input remains addressed in
      // the client's logical W x H viewport and must map to the captured desktop, not the packed
      // 2W x H encoder surface (or its capped derivative).
      touch_port_event->raise(make_port(display.get(), config));

      // Update client with our current HDR display state
      hdr_info_t hdr_info = std::make_unique<hdr_info_raw_t>(false);
      if (colorspace_is_hdr(encode_device->colorspace)) {
        if (display->get_hdr_metadata(hdr_info->metadata)) {
          hdr_info->enabled = true;
        } else {
          BOOST_LOG(error) << "Couldn't get display hdr metadata when colorspace selection indicates it should have one";
        }
      }
      hdr_event->raise(std::move(hdr_info));

      const bool encode_session_created = encode_run(
        frame_nr,
        mail,
        images,
        session_config,
        display,
        std::move(encode_device),
        capture_thread_ctx.reinit_event,
        channel_data
      );
      if (!encode_session_created) {
        if (recover_failed_sbs_session()) {
          continue;
        }
        return;
      }
    }
  }

  void capture(
    safe::mail_t mail,
    config_t config,
    std::shared_ptr<void> channel_data
  ) {
    // Initialize this stream's depth status independently of every other session.
    mail->event<int>(mail::sbs_depth_status)->raise(0);
    mail->event<bool>(mail::idr)->raise(true);
    capture_async(std::move(mail), config, channel_data);
  }

  int validate_config(std::shared_ptr<platf::display_t> disp, const config_t &config) {
    auto encode_device = make_encode_device(*disp, config);
    if (!encode_device) {
      return -1;
    }

    auto session = make_encode_session(config, std::move(encode_device));
    if (!session) {
      return -1;
    }

    {
      // Image buffers are large, so we use a separate scope to free it immediately after convert()
      auto img = disp->alloc_img();
      if (!img || disp->dummy_img(img.get()) || session->convert(*img)) {
        return -1;
      }
    }

    session->request_idr_frame();

    auto packets = mail::man->queue<packet_t>(mail::video_packets);
    while (!packets->peek()) {
      if (encode(1, *session, packets, nullptr, {})) {
        return -1;
      }
    }

    auto packet = packets->pop();
    if (!packet->is_idr()) {
      BOOST_LOG(error) << "First packet type is not an IDR frame"sv;

      return -1;
    }

    return 0;
  }

  namespace {
    std::optional<std::uint32_t> validate_nvenc_unlocked() {
      const auto output_name {display_device::map_output_name(config::video.output_name)};
      std::shared_ptr<platf::display_t> disp;

      BOOST_LOG(info) << "Trying native NVENC encoder"sv;
      auto fg = util::fail_guard([&]() {
        BOOST_LOG(info) << "Native NVENC encoder validation failed"sv;
      });

      // First, test encoder viability
      config_t config_max_ref_frames {1920, 1080, 60, 1000, 1, 1, 1, 0, 0};
      config_t config_autoselect {1920, 1080, 60, 1000, 1, 0, 1, 0, 0};

      // If the encoder isn't supported at all (not even H.264), bail early
      reset_display(disp, output_name, config_autoselect);
      if (!disp) {
        return std::nullopt;
      }
      if (!disp->is_codec_supported(h264_encoder_name, config_autoselect)) {
        fg.disable();
        BOOST_LOG(info) << "Native NVENC is not supported on this GPU"sv;
        return std::nullopt;
      }

      auto max_ref_frames_h264 = validate_config(disp, config_max_ref_frames);
      auto autoselect_h264 = max_ref_frames_h264 >= 0 ? max_ref_frames_h264 : validate_config(disp, config_autoselect);
      if (autoselect_h264 < 0) {
        return std::nullopt;
      }

      config_max_ref_frames.videoFormat = 1;
      config_autoselect.videoFormat = 1;

      bool hevc_available = false;
      if (disp->is_codec_supported(hevc_encoder_name, config_autoselect)) {
        auto max_ref_frames_hevc = validate_config(disp, config_max_ref_frames);

        auto autoselect_hevc = max_ref_frames_hevc >= 0 ? max_ref_frames_hevc : validate_config(disp, config_autoselect);
        hevc_available = autoselect_hevc >= 0;
      } else {
        BOOST_LOG(info) << "Encoder ["sv << hevc_encoder_name << "] is not supported on this GPU"sv;
      }

      config_max_ref_frames.videoFormat = 2;
      config_autoselect.videoFormat = 2;

      bool av1_available = false;
      if (disp->is_codec_supported(av1_encoder_name, config_autoselect)) {
        auto max_ref_frames_av1 = validate_config(disp, config_max_ref_frames);

        auto autoselect_av1 = max_ref_frames_av1 >= 0 ? max_ref_frames_av1 : validate_config(disp, config_autoselect);
        av1_available = autoselect_av1 >= 0;
      } else {
        BOOST_LOG(info) << "Encoder ["sv << av1_encoder_name << "] is not supported on this GPU"sv;
      }

      // Test HDR support for the 4:2:0 formats used by Artemis.
      bool hevc_hdr_available = false;
      bool av1_hdr_available = false;
      {
        const config_t generic_hdr_config = {1920, 1080, 60, 1000, 1, 0, 3, 1, 1};

        // Reset the display since we're switching from SDR to HDR
        reset_display(disp, output_name, generic_hdr_config);
        if (!disp) {
          BOOST_LOG(warning) << "Could not initialize 10-bit capture for the NVENC HDR probe; keeping validated SDR codecs."sv;
        } else {
          auto test_hdr = [&](bool codec_available, int video_format) {
            auto config = generic_hdr_config;
            config.videoFormat = video_format;

            if (!codec_available) {
              return false;
            }

            const auto encoder_codec_name = nvenc_codec_name(video_format);
            return disp->is_codec_supported(encoder_codec_name, config) && validate_config(disp, config) >= 0;
          };

          hevc_hdr_available = test_hdr(hevc_available, 1);
          av1_hdr_available = test_hdr(av1_available, 2);
        }
      }

      std::uint32_t capabilities = nvenc_h264;
      if (hevc_available) {
        capabilities |= nvenc_hevc;
      }
      if (hevc_hdr_available) {
        capabilities |= nvenc_hevc_hdr;
      }
      if (av1_available) {
        capabilities |= nvenc_av1;
      }
      if (av1_hdr_available) {
        capabilities |= nvenc_av1_hdr;
      }

      fg.disable();
      return capabilities;
    }
  }  // namespace

  int probe_encoders() {
    std::lock_guard lock(nvenc_probe_mutex);
    if (!allow_encoder_probing()) {
      nvenc_capabilities.store(0, std::memory_order_release);
      return -1;
    }

    const auto current_capabilities = nvenc_capabilities.load(std::memory_order_acquire);
    if ((current_capabilities & nvenc_h264) != 0 && !platf::needs_encoder_reenumeration()) {
      return 0;
    }

    BOOST_LOG(info) << "// Testing native NVENC support. Probe errors below are expected. //"sv;
    const auto capabilities = validate_nvenc_unlocked();
    if (!capabilities) {
      nvenc_capabilities.store(0, std::memory_order_release);
      const auto output_name {display_device::map_output_name(config::video.output_name)};
      BOOST_LOG(fatal) << "Unable to initialize the NVIDIA display capture or NVENC encoder."sv;
      if (!config::video.adapter_name.empty() || !output_name.empty()) {
        BOOST_LOG(fatal) << "Please ensure the selected NVIDIA GPU and monitor are connected and powered on."sv;
      }
      return -1;
    }

    nvenc_capabilities.store(*capabilities, std::memory_order_release);

    auto log_codec = [&](std::string_view label, std::string_view codec_name, std::uint32_t sdr_flag, std::uint32_t hdr_flag) {
      if ((*capabilities & sdr_flag) == 0) {
        return;
      }
      BOOST_LOG(debug) << "------  "sv << label << " ------"sv;
      BOOST_LOG(debug) << "4:2:0 SDR: supported"sv;
      if (hdr_flag != 0) {
        BOOST_LOG(debug) << "4:2:0 HDR"sv << ((*capabilities & hdr_flag) != 0 ? ": supported"sv : ": unsupported"sv);
      }
      BOOST_LOG(info) << "Found "sv << label << " encoder: "sv << codec_name << " [nvenc]"sv;
    };
    log_codec("H.264"sv, h264_encoder_name, nvenc_h264, 0);
    log_codec("HEVC"sv, hevc_encoder_name, nvenc_hevc, nvenc_hevc_hdr);
    log_codec("AV1"sv, av1_encoder_name, nvenc_av1, nvenc_av1_hdr);

    return 0;
  }

  int start_capture_async(capture_thread_async_ctx_t &capture_thread_ctx) {
    capture_thread_ctx.reinit_event.reset();

    capture_thread_ctx.capture_ctx_queue = std::make_shared<safe::queue_t<capture_ctx_t>>(1);

    capture_thread_ctx.capture_thread = std::thread {
      captureThread,
      capture_thread_ctx.capture_ctx_queue,
      std::ref(capture_thread_ctx.display_wp),
      std::ref(capture_thread_ctx.reinit_event)
    };

    return 0;
  }

  void end_capture_async(capture_thread_async_ctx_t &capture_thread_ctx) {
    capture_thread_ctx.capture_ctx_queue->stop();

    capture_thread_ctx.capture_thread.join();
  }

}  // namespace video
