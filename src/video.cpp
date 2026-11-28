/**
 * @file src/video.cpp
 * @brief Definitions for video.
 */
// standard includes
#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

// local includes
#include "atomic_presentation.h"
#include "capture_display_retirement.h"
#include "capture_image_pool.h"
#include "config.h"
#include "depth_coordinate_v2.h"
#include "display_device.h"
#include "globals.h"
#include "host_sbs_provider.h"
#include "host_sbs_resolution.h"
#include "input.h"
#include "logging.h"
#include "nvenc/nvenc_base.h"
#include "platform/common.h"
#include "process.h"
#include "system_tray.h"
#include "sync.h"
#include "tracked_async_worker.h"
#include "video.h"
#include "video_capture_generation.h"
#include "video_capture_test.h"
#include "video_encode_pipeline.h"
#include "video_encode_session.h"
#include "video_session_recovery.h"

#ifdef SUNSHINE_TESTS
  #include <stdexcept>
#endif

#include "platform/windows/exclusive_display_reconcile_retry.h"
#include "platform/windows/virtual_display.h"

using namespace std::literals;

namespace video {

  namespace {
    std::atomic_uint32_t sbs_telemetry_generation_counter {0};

    void log_diagnostic_elapsed(
      logging::min_max_avg_periodic_logger<double> &logger,
      const detail::diagnostic_timestamp_t &started,
      detail::diagnostic_clock_t::time_point ended
    ) {
      if (const auto elapsed = detail::diagnostic_elapsed_ms(started, ended)) {
        logger.collect_and_log(*elapsed);
      }
    }

    // Construct these only inside the diagnostics gate: the existing logger owns strings and
    // its tracker reads the clock during construction as well as collection.
    struct capture_stage_diagnostics_t {
      detail::diagnostic_content_tracker_t content;
      logging::min_max_avg_periodic_logger<double> new_content_age {
        info,
        "Video capture: new content age at handoff",
        "ms",
      };
      logging::min_max_avg_periodic_logger<double> repeated_content_age {
        info,
        "Video capture: retained content age at handoff",
        "ms",
      };

      void record(platf::img_t &img) {
        const auto ready = detail::diagnostic_clock_t::now();
        // An image whose pixels capture skipped is not the encoded content: it has neither a
        // content age nor a handoff to conversion (detail::diagnostic_captured_content_timestamp).
        img.diagnostic_capture_ready_timestamp = img.pixels_skipped ? detail::diagnostic_timestamp_t {} : ready;
        const auto timestamp = detail::diagnostic_captured_content_timestamp(img);
        const auto kind = content.observe(timestamp);
        if (kind != detail::diagnostic_content_e::unknown) {
          log_diagnostic_elapsed(
            kind == detail::diagnostic_content_e::new_content ? new_content_age : repeated_content_age,
            timestamp,
            ready
          );
        }
      }
    };

    struct encode_stage_diagnostics_t {
      std::shared_ptr<host_sbs_telemetry::collector> performance;
      detail::diagnostic_content_tracker_t input_content;
      detail::diagnostic_content_tracker_t input_capture;
      detail::diagnostic_content_tracker_t output_content;
      logging::min_max_avg_periodic_logger<double> new_input_age {
        info,
        "Video input: new content age at conversion",
        "ms",
      };
      logging::min_max_avg_periodic_logger<double> repeated_input_age {
        info,
        "Video input: retained content age at conversion",
        "ms",
      };
      logging::min_max_avg_periodic_logger<double> capture_to_conversion {
        info,
        "Video input: capture handoff to first conversion",
        "ms",
      };
      logging::min_max_avg_periodic_logger<double> capture_to_reconversion {
        info,
        "Video input: capture handoff to reconversion",
        "ms",
      };

      void begin_conversion(const platf::img_t &img) {
        const auto entered = detail::diagnostic_clock_t::now();
        const auto timestamp = detail::diagnostic_captured_content_timestamp(img);
        const auto content_kind = input_content.observe(timestamp);
        if (content_kind != detail::diagnostic_content_e::unknown) {
          log_diagnostic_elapsed(
            content_kind == detail::diagnostic_content_e::new_content ? new_input_age : repeated_input_age,
            timestamp,
            entered
          );
        }
        const auto capture_kind = input_capture.observe(img.diagnostic_capture_ready_timestamp);
        if (capture_kind != detail::diagnostic_content_e::unknown) {
          log_diagnostic_elapsed(
            capture_kind == detail::diagnostic_content_e::new_content ? capture_to_conversion : capture_to_reconversion,
            img.diagnostic_capture_ready_timestamp,
            entered
          );
        }
      }
    };

    detail::tracked_async_worker_pool_t &async_teardown_workers() {
      static detail::tracked_async_worker_pool_t workers;
      return workers;
    }

    sbs_telemetry_snapshot_t make_unavailable_sbs_telemetry_snapshot(
      std::uint32_t generation,
      const config::video_t::sbs_t &settings
    ) {
      sbs_telemetry_snapshot_t snapshot;
      snapshot.status = sbs_telemetry_sample_status_e::unavailable;
      snapshot.generation = generation;
      apply_sbs_telemetry_config(snapshot, settings);
      return snapshot;
    }

    sbs_output_dimensions_t fit_aligned_encode_dimensions(
      const std::int64_t width,
      const int height,
      const int max_width,
      const int max_height,
      const int width_alignment = 2
    ) noexcept {
      if (width <= 0 || height <= 0) {
        return {0, 0};
      }
      const int width_mask = ~(width_alignment - 1);
      const int capped_width = std::max(width_alignment, max_width) & width_mask;
      const int capped_height = std::max(2, max_height) & ~1;
      if (width <= capped_width && height <= capped_height) {
        return {
          std::max(width_alignment, static_cast<int>(width) & width_mask),
          std::max(2, height & ~1),
        };
      }
      const double scale = std::min(
        static_cast<double>(capped_width) / width,
        static_cast<double>(capped_height) / height
      );
      return {
        std::max(width_alignment, static_cast<int>(std::lround(width * scale)) & width_mask),
        std::max(2, static_cast<int>(std::lround(height * scale)) & ~1),
      };
    }

    struct encode_limits_t {
      int width;
      int height;
    };

    encode_limits_t effective_encode_limits(
      const int video_format,
      const std::optional<int> configured_max_width,
      const int runtime_max_width,
      const int runtime_max_height
    ) {
      // Conservative per-axis defaults observed on the production RTX 5080. Runtime capability
      // queries remain authoritative when they report a lower value.
      const int codec_limit = video_format == 1 || video_format == 2 ? 8192 : 4096;
      int width = configured_max_width ?
                    std::min(*configured_max_width, codec_limit) :
                    codec_limit;
      int height = codec_limit;
      if (runtime_max_width > 0) {
        width = std::min(width, runtime_max_width);
      }
      if (runtime_max_height > 0) {
        height = std::min(height, runtime_max_height);
      }
      return {width, height};
    }
  }  // namespace

  void launch_async_teardown_worker(std::packaged_task<void()> task) {
    async_teardown_workers().launch(std::move(task));
  }

  void drain_async_teardown_workers() noexcept {
    async_teardown_workers().seal_and_drain();
  }

  void apply_sbs_telemetry_config(
    sbs_telemetry_snapshot_t &snapshot,
    const config::video_t::sbs_t &settings
  ) noexcept {
    snapshot.valid_fields |= sbs_telemetry_valid_field::config;
    snapshot.runtime_flags &= ~sbs_telemetry_runtime_flag::adaptive_enabled;
    // Telemetry v1 requires one of its three legacy plane values whenever VALID_CONFIG is set,
    // but it cannot represent V2's continuous mean zero and linear relative-depth geometry. Keep the neutral
    // compatibility value stable; V2 has no configurable zero-plane field for this protocol.
    snapshot.zero_plane_mode = 2;
    snapshot.pop_floor = static_cast<float>(settings.pop_strength);
    snapshot.pop_ceiling = snapshot.pop_floor;
    snapshot.effective_pop = snapshot.pop_floor;
  }

  std::uint32_t next_sbs_telemetry_generation() noexcept {
    // Zero is reserved for "no renderer has existed yet". Skip it on the theoretical wrap.
    for (;;) {
      const auto generation =
        sbs_telemetry_generation_counter.fetch_add(1, std::memory_order_relaxed) + 1;
      if (generation != 0) {
        return generation;
      }
    }
  }

  bool hdr_stream_negotiation_is_coherent(bool launch_hdr, int dynamic_range) noexcept {
    return launch_hdr == (dynamic_range > 0);
  }

  sbs_output_dimensions_t host_sbs_output_dimensions(
    int base_width,
    int base_height,
    int video_format,
    int configured_max_width,
    int runtime_max_width,
    int runtime_max_height
  ) {
    const auto limits = effective_encode_limits(
      video_format,
      configured_max_width,
      runtime_max_width,
      runtime_max_height
    );
    const std::int64_t packed_width = static_cast<std::int64_t>(base_width) * 2;
    // Each eye must own complete 4:2:0 chroma cells. An even packed width alone can leave
    // an odd eye width and one UV cell shared across the eye seam. Align the width cap before
    // fitting either axis, and reject a cap that cannot contain two even-width eyes.
    if (packed_width < 4 || base_height < 2 || limits.width < 4 || limits.height < 2) {
      return {0, 0};
    }
    return fit_aligned_encode_dimensions(
      packed_width,
      base_height,
      limits.width,
      limits.height,
      4
    );
  }

  sbs_output_dimensions_t clamp_encode_dimensions(
    int width,
    int height,
    int video_format,
    int runtime_max_width,
    int runtime_max_height
  ) {
    const auto limits = effective_encode_limits(
      video_format,
      std::nullopt,
      runtime_max_width,
      runtime_max_height
    );
    return fit_aligned_encode_dimensions(
      width,
      height,
      limits.width,
      limits.height
    );
  }

  platf::capture_backend_e capture_backend_failover_t::preferred_backend() const noexcept {
    return preferred_backend_;
  }

  void capture_backend_failover_t::reset() noexcept {
    preferred_backend_ = platf::capture_backend_e::ddup;
    unexplained_failure_epoch_.reset();
  }

  void capture_backend_failover_t::note_backend_opened(platf::capture_backend_e backend) noexcept {
    if (backend == platf::capture_backend_e::wgc) {
      preferred_backend_ = platf::capture_backend_e::wgc;
    }
  }

  bool capture_backend_failover_t::note_capture_result(const capture_attempt_t &attempt) noexcept {
    if (attempt.backend != platf::capture_backend_e::ddup || (attempt.result != platf::capture_e::reinit && attempt.result != platf::capture_e::error)) {
      return false;
    }

    constexpr auto stable_lifetime = 2s;
    constexpr std::uint64_t stable_frame_count = 120;
    const bool stable = attempt.captured_frames >= stable_frame_count ||
                        (attempt.captured_frames > 0 && attempt.lifetime >= stable_lifetime);
    // A mode change while this attempt was open explains its restart and any earlier pending one.
    const bool explained = attempt.display_epoch_at_end != attempt.display_epoch_at_open;
    if (stable || explained) {
      unexplained_failure_epoch_.reset();
      return false;
    }

    // A change between the earlier failure and this attempt's open explains the earlier failure,
    // whose notification arrived late. Only an unbroken pair of unexplained failures latches.
    if (!unexplained_failure_epoch_ || *unexplained_failure_epoch_ != attempt.display_epoch_at_open) {
      unexplained_failure_epoch_ = attempt.display_epoch_at_end;
      return false;
    }

    unexplained_failure_epoch_.reset();
    const bool latched = preferred_backend_ != platf::capture_backend_e::wgc;
    preferred_backend_ = platf::capture_backend_e::wgc;
    return latched;
  }

  config::depth_model_info host_sbs_v2_depth_model() {
    static_assert(!models::depth_coordinate_v2::model_calibrations.empty());
    const auto &production = models::depth_coordinate_v2::model_calibrations.front();
    return {
      std::string {production.depth_model},
      std::string {production.depth_model_url},
    };
  }

  /**
   * @brief Check if we can allow probing for the encoders.
   * @return True if there should be no issues with the probing, false if we should prevent it.
   */
  bool allow_encoder_probing() {
    const auto devices {display_device::enumerate_devices()};

    if (devices.empty()) {
      // We'll create a temporary virtual display for probing anyways.
      if (proc::vDisplayDriverStatus == VDISPLAY::DRIVER_STATUS::OK) {
        return false;
      }
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

  nvenc_encode_session_t::nvenc_encode_session_t(platf::native_encode_owner_t encode_device):
      device(std::move(encode_device)),
      encoder(device->encoder()) {
  }

  int nvenc_encode_session_t::convert(platf::img_t &img) {
    return device->convert(img);
  }

  int nvenc_encode_session_t::convert_with_encode_target(
    platf::img_t &img,
    const std::chrono::steady_clock::time_point next_encode_target
  ) {
    return device->convert_with_encode_target(img, next_encode_target);
  }

  bool nvenc_encode_session_t::needs_conversion_poll() const {
    return device->needs_conversion_poll();
  }

  bool nvenc_encode_session_t::advance_pending_conversion(const std::chrono::steady_clock::time_point deadline) {
    return device->advance_pending_conversion(deadline);
  }

  bool nvenc_encode_session_t::conversion_repeats_pending_picture() const {
    return device->conversion_repeats_pending_picture();
  }

  void nvenc_encode_session_t::set_external_frame_wake(std::function<void()> wake) {
    device->set_external_frame_wake(std::move(wake));
  }

  bool nvenc_encode_session_t::conversion_kept_input() const {
    return device->conversion_kept_input();
  }

  bool nvenc_encode_session_t::external_frame_held() const {
    return device->external_frame_held();
  }

  bool nvenc_encode_session_t::capture_paced() const {
    return device->capture_paced();
  }

  bool nvenc_encode_session_t::set_stream_gamma(stream_gamma_mode_e mode) {
    return device->set_stream_gamma(mode);
  }

  stream_gamma_mode_e nvenc_encode_session_t::stream_gamma_mode() const {
    return device->stream_gamma_mode();
  }

  float nvenc_encode_session_t::stream_gamma_white_nits() const {
    return device->stream_gamma_white_nits();
  }

  bool nvenc_encode_session_t::stream_is_hdr() const {
    return colorspace_is_hdr(device->colorspace);
  }

  std::optional<std::chrono::steady_clock::time_point> nvenc_encode_session_t::rendered_content_timestamp() const {
    return device->rendered_content_timestamp();
  }

  void nvenc_encode_session_t::request_idr_frame() {
    force_idr = true;
  }

  void nvenc_encode_session_t::request_normal_frame() {
    force_idr = false;
  }

  bool nvenc_encode_session_t::invalidate_ref_frames(int64_t first_frame, int64_t last_frame) {
    using result_e = nvenc::nvenc_base::ref_frame_invalidation_e;
    const auto result = encoder.invalidate_ref_frames(first_frame, last_frame);
    if (result == result_e::needs_idr) {
      force_idr = true;
    }
    return result != result_e::unchanged;
  }

  /** With pictures in flight: whether invalidate_ref_frames() must first wait for them, because it
   *  would invalidate (an already handled, malformed or too-large range is decided at once). */
  bool nvenc_encode_session_t::invalidation_needs_drain(int64_t first_frame, int64_t last_frame) const {
    return encoder.frames_in_flight() != 0 &&
           encoder.would_invalidate_ref_frames(first_frame, last_frame);
  }

  bool nvenc_encode_session_t::reconfigure_bitrate(int bitrate_kbps) {
    return encoder.reconfigure_bitrate(bitrate_kbps);
  }

  nvenc::nvenc_encoded_frame nvenc_encode_session_t::encode_frame(
    uint64_t frame_index,
    std::vector<std::uint8_t> &&frame_buffer
  ) {
    auto result = encoder.encode_frame(
      frame_index,
      force_idr,
      std::move(frame_buffer)
    );
    force_idr = false;
    retire_input_reads();
    return result;
  }

  nvenc::nvenc_base::frame_timing_t nvenc_encode_session_t::last_frame_timing() const {
    return encoder.last_frame_timing();
  }

  /** This process's video memory on the encoder's adapter (nvenc_base::video_memory_diagnostics()). */
  std::string nvenc_encode_session_t::video_memory_diagnostics() const {
    return encoder.video_memory_diagnostics();
  }

  /** NVENC pictures that may be in flight (nvenc_base::pipeline_depth()). */
  unsigned nvenc_encode_session_t::pipeline_depth() const {
    return encoder.pipeline_depth();
  }

  bool nvenc_encode_session_t::input_waits_for_retirement() const noexcept {
    return encoder.input_waits_for_retirement();
  }

  void nvenc_encode_session_t::retire_input_reads() {
    // Nonblocking fallback for export slots whose conversion reads completed.
    // The receiver's read-fence wait normally hands them back without this check.
    device->encoder_consumed_input();
  }

  /** Encode thread, with pictures in flight: hand off the current converted input. */
  bool nvenc_encode_session_t::submit_frame(uint64_t frame_index) {
    const bool submitted = encoder.submit_frame(frame_index, force_idr);
    force_idr = false;
    retire_input_reads();
    return submitted;
  }

  /** Retrieving thread: the oldest submitted picture, waited for (nvenc_base::retrieve_frame()). */
  nvenc::nvenc_encoded_frame nvenc_encode_session_t::retrieve_frame() {
    return encoder.retrieve_frame(acquire_frame_buffer());
  }

  std::chrono::nanoseconds nvenc_encode_session_t::last_submit_timing() const {
    return encoder.last_submit_timing();
  }

  hdr_info_t nvenc_encode_session_t::hdr_info() const {
    auto info = std::make_unique<hdr_info_raw_t>(false);
    if (colorspace_is_hdr(device->colorspace)) {
      if (device->hdr_metadata) {
        info->metadata = *device->hdr_metadata;
        info->enabled = true;
      } else {
        BOOST_LOG(error) << "Couldn't get display hdr metadata when colorspace selection indicates it should have one";
      }
    }
    return info;
  }

  std::future<void> nvenc_encode_session_t::retire() && {
    return std::move(device).retire();
  }

  std::vector<std::uint8_t> nvenc_encode_session_t::acquire_frame_buffer() {
    return frame_buffer_pool->acquire();
  }

  void nvenc_encode_session_t::recycle_frame_buffer(std::vector<std::uint8_t> &&frame_buffer) {
    frame_buffer_pool->recycle(std::move(frame_buffer));
  }

  const std::shared_ptr<encoded_frame_buffer_pool_t> &nvenc_encode_session_t::frame_pool() const {
    return frame_buffer_pool;
  }

  namespace detail {
    /**
     * What the capture worker asks of the machine: which display the session wants, opening
     * it, and the topology around it. Production uses real_capture_platform(); a test runs
     * the same worker against inert operations (capture_reinit_test_t).
     */
    struct capture_platform_t {
      using open_display_t = std::function<std::shared_ptr<platf::display_t>(
        const std::string &, const config_t &, platf::capture_backend_e, std::weak_ptr<platf::capture_display_retirement_t>
      )>;

      std::function<std::string()> preferred_display_name;
      std::function<void(const std::string &)> set_display_name;
      /// Re-enumerate the displays, keeping the index on the preferred (or else the current) one.
      std::function<void(std::vector<std::string> &, int &, const std::string &)> refresh_displays;
      open_display_t open_display;
      /// Open after a reinitialization, where a first failure is retried.
      open_display_t reopen_display;
      std::function<void()> adjust_thread_priority;
      std::function<std::uint64_t()> display_change_epoch;
      /// Complete an owned topology repair before outputs are probed. The argument tells whether
      /// the worker is still wanted; false ends the worker.
      std::function<bool(const std::function<bool()> &)> reconcile_topology;
      /// Optional: called on each pass of the wait for the previous display's release.
      std::function<void()> waited_for_display_release;
    };
  }  // namespace detail

  static detail::capture_platform_t real_capture_platform();

  class capture_session_t {
  public:
    capture_session_t(
      img_event_t images,
      const config_t &config
#ifdef SUNSHINE_TESTS
      ,
      detail::capture_worker_test_callback_t test_callback = nullptr,
      void *test_state = nullptr,
      detail::capture_worker_test_callback_t test_before_admission = nullptr,
      std::optional<detail::capture_platform_t> test_platform = std::nullopt
#endif
    ):
        images_(std::move(images)),
        config_(config),
        platform_(
#ifdef SUNSHINE_TESTS
          test_platform ? std::move(*test_platform) :
#endif
                          real_capture_platform()
        )
#ifdef SUNSHINE_TESTS
        ,
        test_callback_(test_callback),
        test_state_(test_state),
        test_before_admission_(test_before_admission)
#endif
        ,
        thread_(&capture_session_t::run, this) {
    }

    ~capture_session_t() {
      stop();
    }

    capture_session_t(const capture_session_t &) = delete;
    capture_session_t &operator=(const capture_session_t &) = delete;
    capture_session_t(capture_session_t &&) = delete;
    capture_session_t &operator=(capture_session_t &&) = delete;

    void stop() {
      // A tracked native teardown may still retain the old display. End the image wait before
      // cancellation and join so the capture worker can leave its display-reference drain.
      images_->stop();
      run_control_.stop();
      generations_.stop();
      if (thread_.joinable()) {
        thread_.join();
      }
    }

    bool running() const {
      return run_control_.running();
    }

    detail::capture_generation_handoff_t::display_lease_t try_acquire_display() {
      return generations_.acquire();
    }

    bool reinit_pending() const {
      return generations_.reinit_pending();
    }

    void set_live_video_mode(std::optional<video_mode_change_t> mode) {
      live_video_mode_ = std::move(mode);
    }

#ifdef SUNSHINE_TESTS
    std::optional<video_mode_change_t> live_video_mode() {
      auto lock = live_video_mode_.lock();
      return *live_video_mode_;
    }
#endif

  private:
    void run();

    img_event_t images_;
    config_t config_;
    safe::signal_t run_control_;
    detail::capture_generation_handoff_t generations_;

    // Latest live video mode (0x3007) applied by the encode thread. run() folds it into
    // its own capture config so a *later* display reinitialization rebuilds at the current mode
    // instead of reverting to the mode negotiated at launch. The in-flight capture session is
    // retuned through display_t::set_client_frame_rate() rather than through this copy.
    sync_util::sync_t<std::optional<video_mode_change_t>> live_video_mode_;

    const detail::capture_platform_t platform_;

#ifdef SUNSHINE_TESTS
    detail::capture_worker_test_callback_t test_callback_;
    void *test_state_;
    detail::capture_worker_test_callback_t test_before_admission_;
#endif

    std::thread thread_;  ///< Last: run() starts only after all referenced state exists.
  };

#ifdef SUNSHINE_TESTS
  namespace detail {
    capture_worker_test_context_t::capture_worker_test_context_t(
      const config_t &config,
      const img_event_t &images,
      void *cancel_state,
      bool (*running)(const void *) noexcept,
      bool (*wait_for_cancel)(void *, std::chrono::milliseconds) noexcept
    ):
        config(config),
        images(images),
        cancel_state_(cancel_state),
        running_(running),
        wait_for_cancel_(wait_for_cancel) {
    }

    bool capture_worker_test_context_t::running() const noexcept {
      return running_(cancel_state_);
    }

    bool capture_worker_test_context_t::wait_for_cancel(std::chrono::milliseconds timeout) const noexcept {
      return wait_for_cancel_(cancel_state_, timeout);
    }

    struct capture_worker_test_access {
      static void invoke(
        const config_t &config,
        const img_event_t &images,
        safe::signal_t &run_control,
        capture_worker_test_callback_t callback,
        void *state
      ) noexcept {
        const capture_worker_test_context_t borrowed {
          config,
          images,
          &run_control,
          [](const void *cancel_state) noexcept {
            return static_cast<const safe::signal_t *>(cancel_state)->running();
          },
          [](void *cancel_state, std::chrono::milliseconds timeout) noexcept {
            auto &run_control = *static_cast<safe::signal_t *>(cancel_state);
            (void) run_control.pop(timeout);
            return !run_control.running();
          },
        };
        callback(state, borrowed);
      }
    };
  }  // namespace detail
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
    platf::capture_backend_e preferred_backend = platf::capture_backend_e::ddup,
    std::weak_ptr<platf::capture_display_retirement_t> retirement = {}
  ) {
    // We try this twice, in case we still get an error on reinitialization
    for (int x = 0; x < 2; ++x) {
      disp.reset();
      disp = platf::display(display_name, config, preferred_backend, retirement);
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

  static detail::capture_platform_t real_capture_platform() {
    detail::capture_platform_t platform;
    platform.preferred_display_name = []() {
      return proc::proc.get_display_name();
    };
    platform.set_display_name = [](const std::string &name) {
      proc::proc.set_display_name(name);
    };
    platform.refresh_displays = [](std::vector<std::string> &display_names, int &current_display_index, const std::string &preferred) {
      auto preferred_display_name = preferred;
      refresh_displays(display_names, current_display_index, preferred_display_name);
    };
    platform.open_display = [](const std::string &name, const config_t &config, platf::capture_backend_e backend, std::weak_ptr<platf::capture_display_retirement_t> retirement) {
      return platf::display(name, config, backend, std::move(retirement));
    };
    platform.reopen_display = [](const std::string &name, const config_t &config, platf::capture_backend_e backend, std::weak_ptr<platf::capture_display_retirement_t> retirement) {
      // reset_display() sleeps between its retries.
      std::shared_ptr<platf::display_t> display;
      reset_display(display, name, config, backend, std::move(retirement));
      return display;
    };
    platform.adjust_thread_priority = []() {
      platf::adjust_thread_priority(platf::thread_priority_e::critical);
    };
    platform.display_change_epoch = []() {
      return platf::display_change_epoch();
    };
    platform.reconcile_topology = [](const std::function<bool()> &still_wanted) {
      const auto topology_generation = platf::primary_display::exclusive_session_generation();
      const auto topology = platf::primary_display::detail::reconcile_before_capture(
        [topology_generation]() {
          return platf::primary_display::reconcile_exclusive_display_topology(topology_generation).result;
        },
        [&]() {
          return still_wanted() &&
                 platf::primary_display::exclusive_session_generation() == topology_generation;
        },
        []() {
          return std::chrono::steady_clock::now();
        },
        [](auto delay) {
          std::this_thread::sleep_for(delay);
        }
      );
      if (topology == platf::primary_display::detail::capture_reconcile_result_e::cancelled) {
        return false;
      }
      if (topology == platf::primary_display::detail::capture_reconcile_result_e::exhausted) {
        BOOST_LOG(warning) << "Exclusive display topology is still settling; continuing normal capture recovery."sv;
      }
      return true;
    };
    return platform;
  }

  void capture_session_t::run() {
    auto &images = images_;
    auto &config = config_;
    auto &run_control = run_control_;
    auto &live_video_mode = live_video_mode_;

    auto fg = util::fail_guard([&]() {
      run_control.stop();
      images->stop();
      generations_.stop();
    });

#ifdef SUNSHINE_TESTS
    if (test_before_admission_) {
      detail::capture_worker_test_access::invoke(config, images, run_control, test_before_admission_, test_state_);
    }
#endif

    // The old initial-context pop rejected a stop that won before worker admission.
    if (!run_control.running()) {
      return;
    }

#ifdef SUNSHINE_TESTS
    if (test_callback_) {
      detail::capture_worker_test_access::invoke(config, images, run_control, test_callback_, test_state_);
      return;
    }
#endif

    std::vector<std::string> display_names;
    int display_p = -1;
    std::shared_ptr<platf::display_t> disp;
    auto generation = generations_.begin();
    if (!generation) {
      return;
    }
    // The capture thread detaches on every exit. A final owner arriving later deletes normally;
    // a display already returned to this thread is destroyed here, outside admission/slot locks.
    auto detach_generation = util::fail_guard([&]() {
      (void) generations_.revoke(generation);
      auto returned = generation->retirement()->detach();
    });
    capture_backend_failover_t capture_failover;
    // Display-change evidence starts before a duplication opens: a change after open invalidates it.
    auto display_epoch_at_open = platform_.display_change_epoch();
    std::string active_display_name;
    const auto initial_display_name = platform_.preferred_display_name();
    if (!initial_display_name.empty()) {
      active_display_name = initial_display_name;
      disp = platform_.open_display(initial_display_name, config, capture_failover.preferred_backend(), generation->retirement());
      if (disp) {
        capture_failover.note_backend_opened(disp->capture_backend());
      }
    }
    if (!disp && detail::capture_display_release_wait_cancelled(run_control.running(), images->running())) {
      return;
    }
    if (!disp) {
      // Get all the monitor names now, rather than at boot, to
      // get the most up-to-date list available monitors
      platform_.refresh_displays(display_names, display_p, {});
      if (active_display_name != display_names[display_p]) {
        capture_failover.reset();
        active_display_name = display_names[display_p];
      }
      disp = platform_.open_display(active_display_name, config, capture_failover.preferred_backend(), generation->retirement());
      if (disp) {
        capture_failover.note_backend_opened(disp->capture_backend());
        platform_.set_display_name(active_display_name);
      } else {
        return;
      }
    }

    if (!generations_.publish(generation, disp)) {
      return;
    }

    detail::capture_image_pool_t<12> image_pool {detail::capture_image_retention_e::reclaim_idle};
    auto pull_free_image_callback = [&](std::shared_ptr<platf::img_t> &img_out) -> bool {
      img_out.reset();
      while (run_control.running()) {
        img_out = image_pool.try_acquire([&] {
          return disp->alloc_img();
        });
        if (img_out) {
          return true;
        }
        std::this_thread::sleep_for(1ms);
      }
      return false;
    };

    // Capture takes place on this thread
    platform_.adjust_thread_priority();

    bool capture_cursor = true;
    auto capture_diagnostics = detail::make_diagnostic_state<capture_stage_diagnostics_t>(
      config::sunshine.diagnostics_enabled
    );
    while (run_control.running()) {
      std::uint64_t captured_frames = 0;
      auto push_captured_image_callback = [&](std::shared_ptr<platf::img_t> &&img, bool frame_captured) -> bool {
        if (!run_control.running() || !images->running()) {
          return false;
        }

        if (frame_captured) {
          if (capture_diagnostics && img) {
            capture_diagnostics->record(*img);
          }
          ++captured_frames;
          images->raise(std::move(img));
        }

        return true;
      };

      const auto backend = disp->capture_backend();
      const auto capture_started = std::chrono::steady_clock::now();
      auto status = disp->capture(push_captured_image_callback, pull_free_image_callback, &capture_cursor);
      if (capture_failover.note_capture_result({
            backend,
            status,
            captured_frames,
            std::chrono::steady_clock::now() - capture_started,
            display_epoch_at_open,
            platform_.display_change_epoch(),
          })) {
        BOOST_LOG(warning) << "Desktop Duplication failed repeatedly before stable capture without a "sv
                              "display-mode change; using Windows.Graphics.Capture for the rest of this session."sv;
        system_tray::update_tray_notice(
          "Screen capture changed for this session",
          "Desktop Duplication kept failing, so Windows.Graphics.Capture is used until the session ends."
        );
      }

      switch (status) {
        case platf::capture_e::error:
          BOOST_LOG(warning) << "Display capture failed; rebuilding the automatic capture backend."sv;
          [[fallthrough]];
        case platf::capture_e::reinit:
          {
            (void) generations_.revoke(generation);
            // End the idle wait so the encode loop can release the revoked display now.
            if (config.encode_wake) {
              config.encode_wake->wake();
            }

            // Some classes of images contain references to the display --> display won't delete unless img is deleted
            image_pool.clear();

            // Revoke admission before dropping capture's strong reference. The original control
            // block includes ordinary borrowers, partial/retiring encoders and shared_from_this.
            disp.reset();
            std::unique_ptr<platf::display_t> returned_display;
            while (!(returned_display = generation->retirement()->wait_for(0ms))) {
              // Preserve the old wait's drain, including a retained source returned by encode_run.
              while (images->try_pop()) {
              }

              if (platform_.waited_for_display_release) {
                platform_.waited_for_display_release();
              }

              if (detail::capture_display_release_wait_cancelled(run_control.running(), images->running())) {
                return;
              }
              returned_display = generation->retirement()->wait_for(20ms);
              if (returned_display) {
                break;
              }
            }
            // Normal reinitialization deletes on the capture thread, before topology or reopen.
            returned_display.reset();
            // A plain Windows image does not retain display_t. Always discard it after final
            // display release, even when no encoder ever acquired this generation.
            while (images->try_pop()) {
            }

            // Adopt the newest live video mode before rebuilding the display. Without this a
            // reinitialization triggered after a 0x3007 change would recreate the capture session
            // with the launch-time cadence and geometry.
            {
              auto lg = live_video_mode.lock();
              if (*live_video_mode) {
                const auto &mode = **live_video_mode;
                config.width = mode.width;
                config.height = mode.height;
                config.framerate = mode.framerate;
                config.framerateX100 = mode.framerateX100;
                config.encodingFramerate = mode.encodingFramerate;
                config.bitrate = mode.bitrate;
              }
            }

            // The old DDUP/NVENC references are gone. Complete the owned topology
            // repair before probing outputs, rather than racing a game-exit repair
            // on the session-monitor worker and then opening a transient display.
            if (!platform_.reconcile_topology([&]() {
                  return run_control.running() && images->running();
                })) {
              return;
            }
            auto next_generation = generations_.begin();
            if (!next_generation) {
              return;
            }
            generation = std::move(next_generation);
            display_epoch_at_open = platform_.display_change_epoch();
            // Reopen the display this session was already capturing before looking for another
            // one, as the first open of the session does. Enumerating the outputs probes every
            // adapter and output while the viewer waits, and is only needed when that display is
            // gone or cannot be opened.
            {
              const auto session_display_name = platform_.preferred_display_name();
              if (!session_display_name.empty() && session_display_name == active_display_name) {
                disp = platform_.open_display(active_display_name, config, capture_failover.preferred_backend(), generation->retirement());
                if (disp) {
                  capture_failover.note_backend_opened(disp->capture_backend());
                }
              }
            }
            while (!disp && run_control.running()) {
              // Release the display before reenumerating displays, since some capture backends
              // only support a single display session per device/application.
              disp.reset();

              // Refresh display names since a display removal might have caused the reinitialization
              const auto preferred_display_name = platform_.preferred_display_name();
              platform_.refresh_displays(display_names, display_p, preferred_display_name);

              if (active_display_name != display_names[display_p]) {
                capture_failover.reset();
                active_display_name = display_names[display_p];
              }

              // The production reopen sleeps between its retries.
              disp = platform_.reopen_display(active_display_name, config, capture_failover.preferred_backend(), generation->retirement());
              if (disp) {
                capture_failover.note_backend_opened(disp->capture_backend());
                platform_.set_display_name(active_display_name);
                break;
              }
            }
            if (!disp) {
              return;
            }

            if (!generations_.publish(generation, disp)) {
              return;
            }
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

  struct encoded_packet_publish_result_t {
    bool failed;
    std::size_t dropped_packets;
    bool request_idr;
  };

  /** What a picture's packet carries, captured on the encode thread when the picture is submitted. */
  struct encoded_frame_context_t {
    int64_t frame_nr = 0;
    std::optional<std::chrono::steady_clock::time_point> frame_timestamp;
    std::optional<std::chrono::steady_clock::time_point> content_timestamp;
    detail::optional_frame_time_point_t processing_started;
    bool encoder_input_retained = false;
    detail::diagnostic_content_e diagnostic_content = detail::diagnostic_content_e::unknown;
    detail::diagnostic_timestamp_t encode_started;  ///< Diagnostics only.
  };

  encoded_frame_context_t make_encoded_frame_context(
    int64_t frame_nr,
    std::optional<std::chrono::steady_clock::time_point> frame_timestamp,
    std::optional<std::chrono::steady_clock::time_point> content_timestamp,
    encode_stage_diagnostics_t *diagnostics,
    bool converted_frame,
    detail::optional_frame_time_point_t processing_started,
    bool first_encoder_output
  ) {
    encoded_frame_context_t frame;
    frame.frame_nr = frame_nr;
    frame.frame_timestamp = frame_timestamp;
    frame.content_timestamp = content_timestamp;
    frame.processing_started = processing_started;
    frame.encoder_input_retained = detail::encoder_input_was_retained(first_encoder_output, converted_frame);
    if (diagnostics) {
      // In submission order on the encode thread, which is also the packets' order.
      frame.diagnostic_content = diagnostics->output_content.observe(content_timestamp, !converted_frame);
      frame.encode_started = detail::diagnostic_clock_t::now();
    }
    return frame;
  }

  /** Publish one retrieved picture. Runs on whichever thread retrieved it; with pictures in flight
   *  that is the retrieving thread. */
  encoded_packet_publish_result_t publish_encoded_frame(
    nvenc::nvenc_encoded_frame &&encoded_frame,
    const encoded_frame_context_t &frame,
    nvenc_encode_session_t &session,
    packet_queue_t &packets,
    encode_stage_diagnostics_t *diagnostics
  ) {
    if (diagnostics && diagnostics->performance) {
      const auto measured_at = detail::diagnostic_clock_t::now();
      if (const auto elapsed = detail::diagnostic_elapsed_ms(frame.encode_started, measured_at)) {
        diagnostics->performance->record(host_sbs_telemetry::stage::encode, *elapsed, measured_at);
      }
    }
    if (encoded_frame.data.empty()) {
      BOOST_LOG(error) << "NvENC returned empty packet";
      session.recycle_frame_buffer(std::move(encoded_frame.data));
      return {true, 0, false};
    }

    if (frame.frame_nr != static_cast<int64_t>(encoded_frame.frame_index)) {
      BOOST_LOG(error) << "NvENC frame index mismatch " << frame.frame_nr << " " << encoded_frame.frame_index;
    }

    auto packet = std::make_unique<encoded_packet_t>(
      std::move(encoded_frame.data),
      encoded_frame.frame_index,
      encoded_frame.idr,
      session.frame_pool()
    );
    packet->after_ref_frame_invalidation = encoded_frame.after_ref_frame_invalidation;
    packet->frame_timestamp = frame.frame_timestamp;
    packet->content_timestamp = frame.content_timestamp;
    packet->processing_started = frame.processing_started;
    packet->encoder_input_retained = frame.encoder_input_retained;
    if (diagnostics) {
      packet->diagnostic_content = frame.diagnostic_content;
      packet->sbs_telemetry_performance = diagnostics->performance;
    }
    const bool packet_is_idr = packet->is_idr();
    // A queued IDR can replace stale packets and immediately repair the reference chain. A delta
    // frame cannot: if the queue is full, discard both the stale queue and this delta frame, then
    // ask the encoder for a fresh IDR on its next iteration.
    const auto publish_result = packets->raise_with_overflow_policy(
      packet_is_idr,
      std::move(packet)
    );
    const auto dropped_packets = publish_result.dropped +
                                 (!publish_result.queued && publish_result.dropped > 0 ? 1 : 0);

    return {
      false,
      dropped_packets,
      publish_result.dropped > 0 && !packet_is_idr,
    };
  }

  /** One picture at a time: submit, wait for and publish it on the encode thread. */
  encoded_packet_publish_result_t encode(
    nvenc_encode_session_t &session,
    packet_queue_t &packets,
    const encoded_frame_context_t &frame,
    encode_stage_diagnostics_t *diagnostics
  ) {
    auto encoded_frame = session.encode_frame(frame.frame_nr, session.acquire_frame_buffer());
    return publish_encoded_frame(std::move(encoded_frame), frame, session, packets, diagnostics);
  }

  /** Encoded-queue backpressure, on whichever thread publishes packets: a dropped delta frame asks
   *  for a recovery IDR; drops are logged at most every 5 s. */
  class encoded_queue_drop_report_t {
  public:
    void record(const encoded_packet_publish_result_t &result, safe::mail_raw_t::event_t<bool> &idr_events) {
      if (result.dropped_packets == 0) {
        return;
      }
      if (result.request_idr) {
        idr_events->try_raise(true);
      }
      log(result.dropped_packets, result.request_idr ? ", recovery IDR requested."sv : ", newest IDR retained."sv);
    }

    /** A delta frame discarded while the recovery IDR an earlier drop requested is still due. */
    void record_awaiting_idr() {
      log(1, ", awaiting the recovery IDR."sv);
    }

  private:
    void log(std::size_t dropped, std::string_view outcome) {
      drops_since_log += dropped;
      const auto now = std::chrono::steady_clock::now();
      if (now >= next_log) {
        BOOST_LOG(warning) << "Encoded video queue backpressure dropped "sv
                           << drops_since_log
                           << " frame(s); queue capacity="sv
                           << ENCODED_PACKET_QUEUE_LIMIT
                           << outcome;
        drops_since_log = 0;
        next_log = now + 5s;
      }
    }

    std::size_t drops_since_log = 0;
    std::chrono::steady_clock::time_point next_log = std::chrono::steady_clock::now();
  };

  /**
   * One encode loop's stream gamma requests (0x300C) and acknowledgements (0x300D). Only the
   * encode thread mutates the converter, and a setting is acknowledged only after a real
   * conversion and successful encode prove it. A request still pending when the loop ends is
   * answered as failed.
   */
  class stream_gamma_tracker_t {
  public:
    stream_gamma_tracker_t(
      const safe::mail_t &mail,
      const config_t &config,
      nvenc_encode_session_t &session,
      const safe::mail_raw_t::event_t<bool> &shutdown_event
    ):
        config_ {config},
        shutdown_event_ {shutdown_event},
        requests_ {mail->queue<stream_gamma_request_t>(mail::stream_gamma, stream_gamma_queue_limit)},
        acks_ {mail->queue<stream_gamma_ack_t>(mail::stream_gamma_ack, 2 * stream_gamma_queue_limit)} {
      // SDR bypass must not replace the user's selection. Retry that selection when a rebuilt
      // capture/encoder becomes HDR again, while keeping its applied proof separate.
      const auto initial_gamma = config.stream_gamma_state ? config.stream_gamma_state->requested_mode() : config.stream_gamma;
      const bool initial_gamma_accepted = session.set_stream_gamma(initial_gamma);
      // The black startup dummy does not prove that real source pixels used the new transform.
      pending_ = stream_gamma_request_t {initial_gamma, 0};
      pending_status_ = initial_gamma_accepted ? stream_gamma_status_e::applied : rejection_status(session, initial_gamma);
    }

    ~stream_gamma_tracker_t() {
      if (pending_ && pending_->request_id != 0 && config_.stream_gamma_supported) {
        queue_ack({
          stream_gamma_status_e::failed,
          pending_->mode,
          pending_->request_id,
          config_.stream_gamma_state->current(),
        });
      }
    }

    stream_gamma_tracker_t(const stream_gamma_tracker_t &) = delete;
    stream_gamma_tracker_t &operator=(const stream_gamma_tracker_t &) = delete;

    /** Whether a setting still waits for the encoded picture that proves it. */
    [[nodiscard]] bool pending() const noexcept {
      return pending_.has_value();
    }

    /** Apply the next client request, once the previous one has been answered. */
    void poll(nvenc_encode_session_t &session) {
      if (pending_) {
        return;
      }
      if (auto request = requests_->pop(0ms)) {
        // The control decoder admits only valid modes. Retain even an unsupported/failed
        // choice so a later capable encoder can apply it; an explicit default replaces it.
        if (config_.stream_gamma_state) {
          config_.stream_gamma_state->set_requested_mode(request->mode);
        }
        const bool accepted = session.set_stream_gamma(request->mode);
        if (!accepted) {
          queue_ack({
            rejection_status(session, request->mode),
            request->mode,
            request->request_id,
            config_.stream_gamma_state->current(),
          });
        } else {
          pending_ = *request;
          pending_status_ = stream_gamma_status_e::applied;
        }
      }
    }

    /**
     * Publish the transform a submitted real picture used, answering any pending request. The
     * caller checks that the picture was converted and the session owns a gamma publisher.
     */
    void prove(nvenc_encode_session_t &session) {
      const auto actual_mode = session.stream_gamma_mode();
      const auto actual_white = session.stream_gamma_white_nits();
      const auto proven = config_.stream_gamma_state->current();
      if (pending_ || proven.generation == 0 || actual_mode != proven.mode || actual_white != proven.white_nits) {
        const auto applied = config_.stream_gamma_state->publish(actual_mode, actual_white);
        const auto requested_mode = config_.stream_gamma_state->requested_mode();
        const auto status = pending_ ? pending_status_ :
                            actual_mode == requested_mode ? stream_gamma_status_e::applied :
                                                            rejection_status(session, requested_mode);
        BOOST_LOG(info) << "Stream gamma encoder proof: requested="sv
                        << static_cast<int>(requested_mode)
                        << " applied="sv << static_cast<int>(actual_mode)
                        << " white="sv << actual_white << " nits generation="sv
                        << applied.generation << " request="sv
                        << (pending_ ? pending_->request_id : 0);
        if (config_.stream_gamma_supported) {
          queue_ack({
            status,
            requested_mode,
            pending_ ? pending_->request_id : 0,
            applied,
          });
        }
        pending_.reset();
      }
    }

  private:
    static stream_gamma_status_e rejection_status(nvenc_encode_session_t &session, stream_gamma_mode_e requested) {
      return !session.stream_is_hdr() && requested != stream_gamma_mode_e::windows_default ?
               stream_gamma_status_e::rejected_unsupported :
               stream_gamma_status_e::failed;
    }

    void queue_ack(const stream_gamma_ack_t &ack) {
      const auto result = acks_->raise_with_overflow_policy(false, ack);
      if (result.dropped != 0) {
        BOOST_LOG(error) << "Stream gamma acknowledgement queue exceeded its session bound"sv;
        shutdown_event_->raise(true);
      }
    }

    const config_t &config_;
    const safe::mail_raw_t::event_t<bool> &shutdown_event_;
    safe::mail_raw_t::queue_t<stream_gamma_request_t> requests_;
    safe::mail_raw_t::queue_t<stream_gamma_ack_t> acks_;
    std::optional<stream_gamma_request_t> pending_;
    stream_gamma_status_e pending_status_ {};
  };

  /**
   * capture_async's live video-mode (0x3007) state: the client's per-eye source width, the Host
   * SBS mode the next encode session runs, the last mode that produced a working encoder, whether
   * the mode in hand came from a live request, and the requests still owed an answer.
   *
   * These change only together with the session config, the capture worker's live mode, the
   * session's requested SBS mode and the depth-status event, so each transition below takes the
   * collaborators it updates and changes them in one fixed order. Only the encode thread uses it.
   */
  class live_mode_state_t {
  public:
    explicit live_mode_state_t(const config_t &config):
        base_width_ {config.width},
        sbs_mode_ {config.sbs_mode} {
    }

    /** Per-eye source width; SBS doubling is applied on top of it. */
    [[nodiscard]] int base_width() const noexcept {
      return base_width_;
    }

    /** The Host SBS mode the next encode session is built with. */
    [[nodiscard]] int sbs_mode() const noexcept {
      return sbs_mode_;
    }

    /** Install one drained client request before the reinitialization gate sees the config. */
    void apply_request(
      const video_mode_change_t &mode,
      config_t &config,
      capture_session_t &capture_session,
      const safe::mail_raw_t::event_t<int> &depth_status_event
    ) {
      install(mode, config);
      if (sbs_mode_ != SBS_AI) {
        if (config.sbs_debug_dump_pending) {
          config.sbs_debug_dump_pending->store(false, std::memory_order_release);
        }
        depth_status_event->raise(0);
      }
      capture_session.set_live_video_mode(std::optional<video_mode_change_t> {mode});
      config_from_live_video_mode_ = true;
      if (active_ack_request_) {
        superseded_ack_requests_.push_back(*active_ack_request_);
      }
      active_ack_request_ = video_mode_request_ref_t {
        mode.request_id,
        mode.transaction_id,
      };
      BOOST_LOG(info) << "Applying live video mode "sv << config.width << 'x' << config.height
                      << '@' << (config.framerateX100 > 0 ? config.framerateX100 / 100.0 : (double) config.framerate)
                      << "Hz, "sv << config.bitrate << "kbps"sv;
    }

    /** Run without Host SBS when its V2 model refuses this source geometry (@p rejection). */
    void refuse_unsupported_host_sbs(
      std::string_view rejection,
      config_t &config,
      const safe::mail_raw_t::event_t<int> &depth_status_event
    ) {
      if (sbs_mode_ != SBS_AI || rejection.empty()) {
        return;
      }
      // Launch state is checked here too, and a live request is rechecked at the final encoder
      // construction boundary. Never construct an unauthenticated Host SBS geometry.
      BOOST_LOG(warning)
        << "Rejecting Host SBS V2 encode-session construction at "sv
        << base_width_ << 'x' << config.height << ": "sv
        << rejection << '.';
      set_sbs_mode(SBS_OFF, config);
      depth_status_event->raise(0);
    }

    /**
     * A live mode arrives from the client mid-stream, so an unencodable request must not be
     * fatal. Fall back to the last geometry that actually produced an encoder; the stream then
     * continues at the previous mode instead of dropping.
     * @return `false` when there is no live request to revert or no proven mode to revert to.
     */
    bool revert_failed_request(
      config_t &config,
      capture_session_t &capture_session,
      const safe::mail_raw_t::event_t<int> &depth_status_event
    ) {
      if (!config_from_live_video_mode_ || !proven_video_mode_) {
        return false;
      }

      BOOST_LOG(warning) << "The client's live video mode "sv << config.width << 'x' << config.height
                         << " could not be encoded; reverting to "sv
                         << proven_video_mode_->width << 'x' << proven_video_mode_->height << '.';
      install(*proven_video_mode_, config);
      if (sbs_mode_ != SBS_AI) {
        depth_status_event->raise(0);
      }
      capture_session.set_live_video_mode(proven_video_mode_);
      config_from_live_video_mode_ = false;
      // The request never made it into an encode session, so it must not be reported as applied.
      // It is answered once the fallback session comes up and its real mode is known.
      if (active_ack_request_) {
        superseded_ack_requests_.push_back(*active_ack_request_);
        active_ack_request_.reset();
      }
      return true;
    }

    /** A fully initialized encode session makes the config in hand the safe fallback. */
    void prove_encoder(const config_t &config) {
      proven_video_mode_ = video_mode_change_t {
        config.width,
        config.height,
        config.framerate,
        config.framerateX100,
        config.encodingFramerate,
        config.bitrate,
        active_ack_request_ ? active_ack_request_->request_id : 0,
        active_ack_request_ ? active_ack_request_->transaction_id : 0,
      };
      proven_video_mode_->requested_sbs_mode = sbs_mode_;
      config_from_live_video_mode_ = false;
    }

    /**
     * Answer the live requests this session resolves. Superseded ones are reported as not
     * applied so the client can drop them and resynchronize on the mode that is really running.
     */
    void complete_requests(
      const std::shared_ptr<stream::detail::atomic_presentation_t> &atomic_presentation,
      const effective_video_mode_t &effective_mode,
      const desktop_video_mode_t &desktop_mode
    ) {
      if (!superseded_ack_requests_.empty()) {
        if (atomic_presentation) {
          (void) atomic_presentation->complete(video_mode_applied_t {
            std::move(superseded_ack_requests_),
            false,
            effective_mode,
            desktop_mode,
          });
        }
        superseded_ack_requests_.clear();
      }
      if (active_ack_request_) {
        if (atomic_presentation) {
          (void) atomic_presentation->complete(video_mode_applied_t {
            {*active_ack_request_},
            true,
            effective_mode,
            desktop_mode,
          });
        }
        active_ack_request_.reset();
      }
    }

    /**
     * Keep every rebuild/fallback owner on the bitrate NVENC actually accepted. Without this, a
     * later SBS toggle or display reinitialization would reconstruct the session from the
     * pre-reconfigure bitrate even though the client was told otherwise.
     */
    void accept_reconfigured_bitrate(
      const video_mode_change_t &mode,
      config_t &config,
      capture_session_t &capture_session
    ) {
      config.bitrate = mode.bitrate;
      capture_session.set_live_video_mode(std::optional<video_mode_change_t> {mode});
      proven_video_mode_ = mode;
      config_from_live_video_mode_ = false;
    }

  private:
    /** Copy one mode's geometry, cadence, bitrate and Host SBS mode into the config. */
    void install(const video_mode_change_t &mode, config_t &config) {
      base_width_ = mode.width;
      config.width = mode.width;
      config.height = mode.height;
      config.framerate = mode.framerate;
      config.framerateX100 = mode.framerateX100;
      config.encodingFramerate = mode.encodingFramerate;
      config.bitrate = mode.bitrate;
      // Atomic presentation applies its immutable desired mode from this same mode entry.
      set_sbs_mode(mode.requested_sbs_mode, config);
    }

    void set_sbs_mode(int mode, const config_t &config) {
      sbs_mode_ = mode;
      if (config.requested_sbs_mode) {
        config.requested_sbs_mode->store(sbs_mode_, std::memory_order_release);
      }
    }

    int base_width_;
    int sbs_mode_;
    // Geometry/cadence that has already produced a working encoder for this session, plus whether
    // the config currently in hand came from a live client request. A client-requested mode that
    // NVENC ultimately refuses must fall back to the proven one rather than end the session.
    std::optional<video_mode_change_t> proven_video_mode_;
    bool config_from_live_video_mode_ = false;
    // Every live request owes the client exactly one answer, and only this loop knows the geometry
    // that actually gets installed. The host-generated transaction id (unlike the opaque client
    // request id) lets the serialized stream worker match this completion unambiguously.
    std::optional<video_mode_request_ref_t> active_ack_request_;
    std::vector<video_mode_request_ref_t> superseded_ack_requests_;
  };

  void refresh_mouse_keys_if_due(std::chrono::steady_clock::time_point &next_refresh) {
    const auto now = std::chrono::steady_clock::now();
    if (now < next_refresh) {
      return;
    }

    next_refresh = now + 1s;
    platf::refresh_mouse_keys();
  }

  void encode_run(
    int &frame_nr,  // Store progress of the frame number
    bool &awaiting_desktop_after_export,  // In and out; see detail::startup_input_hold_t
    safe::mail_t mail,
    img_event_t images,
    config_t config,
    detail::capture_generation_handoff_t::display_lease_t generation,
    std::unique_ptr<nvenc_encode_session_t> session,
    detail::encoder_recovery_t &recovery,
    packet_queue_t packets,
    const std::function<void(const video_mode_change_t &)> &on_bitrate_reconfigured
#ifdef SUNSHINE_TESTS
    , const detail::native_encode_loop_test_hooks_t *test_hooks = nullptr
#endif
  ) {
    const auto &disp = generation.display();
    // All session exits retire their backend through its owner. Failed encoding additionally
    // keeps the destruction receipt so capture can wait for the existing recovery gate.
    bool failed = true;
    auto fail_guard = util::fail_guard([&session, &recovery, &failed] {
      if (failed) {
        // A timeout keeps NVENC's global creation gate closed until its owner is destroyed.
        // Keep this completion so capture_async does not treat that temporary refusal as a
        // fatal encoder/configuration error. Ordinary mode changes still retire asynchronously.
        recovery.start(std::move(*session).retire());
      }
    });

    // set max frame time based on client-requested target framerate; never shorter than its interval.
    const double minimum_fps_target = detail::minimum_fps_target(config::video.minimum_fps_target, config.encodingFramerate);
    auto max_frametime = std::chrono::nanoseconds(1000ms) * 1000 / minimum_fps_target;
    auto encode_frame_threshold = std::chrono::nanoseconds(1000ms) * 1000 / config.encodingFramerate;
    auto frame_variation_threshold = encode_frame_threshold / 4;
    const bool independent_provider = converts_independent_provider(config);
    BOOST_LOG(info) << "Minimum FPS target set to ~"sv << (minimum_fps_target / 1000) << "fps ("sv << max_frametime << ")"sv;
    BOOST_LOG(info) << "Encoding Frame threshold: "sv << encode_frame_threshold;

    auto shutdown_event = mail->event<bool>(mail::shutdown);
    auto idr_events = mail->event<bool>(mail::idr);
    auto invalidate_ref_frames_events = mail->event<std::pair<int64_t, int64_t>>(mail::invalidate_ref_frames);
    // Atomic client-requested mode/geometry/cadence/bitrate changes arrive on one queue.
    auto video_mode_event = mail->queue<video_mode_change_t>(mail::video_mode);
    auto depth_pipeline_ready_event = config.sbs_depth_pipeline_ready_event;
    stream_gamma_tracker_t stream_gamma {mail, config, *session, shutdown_event};

    {
      // Load a dummy image into the encoder input to ensure we have something to encode
      // even if we timeout waiting on the first frame. This is a relatively large
      // allocation which can be freed immediately after convert(), so we do this
      // in a separate scope.
      auto dummy_img = disp->alloc_img();
      if (!dummy_img || disp->dummy_img(dummy_img.get()) || session->convert(*dummy_img)) {
        return;
      }
    }

    std::chrono::steady_clock::time_point encode_frame_timestamp;
    // A wake meant for a previous encode loop is stale here. Left pending, it would end this loop's
    // first wait at once, before capture delivers a frame, and encode the black startup input.
    images->discard_wake();
    // An independent provider (Game 3D) wakes this loop when its own frame finishes, so a frame is
    // converted as soon as it is ready instead of at the next stream deadline or desktop capture.
    // Like an early capture, a provider frame may convert one variation threshold before its
    // presentation target; the presentation timestamps keep the stream cadence, which caps the
    // encode rate. Without new content such a provider encodes only the minimum-FPS keepalive,
    // measured from the last encode so that metadata-only captures cannot postpone it.
    session->set_external_frame_wake([images]() {
      images->wake();
    });
    // A static capture or provider can leave any session waiting until its keepalive.
    if (config.encode_wake) {
      config.encode_wake->install([images]() {
        images->wake();
      });
    }
    const auto keepalive_interval = std::chrono::duration_cast<std::chrono::nanoseconds>(max_frametime);
    auto last_encode_at = std::chrono::steady_clock::now();
    auto next_mouse_keys_refresh = std::chrono::steady_clock::now() + 1s;
    bool missing_frame_timestamp_warning_logged = false;
    bool first_encoder_output = true;
    // The presentation time of the last conversion: one frame interval later the next capture
    // of a source at the stream's rate is due.
    std::optional<std::chrono::steady_clock::time_point> last_delivery_timestamp;
    detail::optional_frame_time_point_t processing_started;
    // Used by the encode thread with one picture at a time, else only by the retrieving thread.
    encoded_queue_drop_report_t drop_report;

    // Most recent real captured frame. On a same-display rebuild, no new frame is delivered
    // until the desktop changes. Keep this source so the replacement can resume real content.
    detail::latest_encode_source_t<std::shared_ptr<platf::img_t>> source;
    const auto &last_img = source.latest();
    // Declared before retain_source, which reads it on exit.
    detail::startup_input_hold_t startup_input_hold {std::chrono::steady_clock::now(), awaiting_desktop_after_export};
    auto retain_source = util::fail_guard([&] {
      // Encoder failures also rebuild against the same capture display. Preserve its last real
      // image so a static desktop does not keep encoding the replacement's black dummy frame.
      // A newer queued capture wins; the capture thread drains old images on display reinit.
      // Metadata for pixels a live export owned has none to show: capture copies the whole
      // desktop again for the replacement once the guard below releases the output.
      const bool holds_pixels = !last_img || !last_img->pixels_skipped;
      source.return_for_rebuild(*images, shutdown_event->peek(), generation.revoked(), holds_pixels);
      // The replacement then starts from its black startup input, as this loop may still do:
      // it holds that input back until capture delivers the desktop.
      awaiting_desktop_after_export = !holds_pixels || startup_input_hold.active(std::chrono::steady_clock::now());
    });
    // Declared after retain_source so that it runs first. No provider or control wake outlives
    // this loop, and capture stops skipping pixels for an encoder that no longer converts.
    auto release_external_frames = util::fail_guard([&] {
      session->set_external_frame_wake({});
      if (config.encode_wake) {
        config.encode_wake->install({});
      }
    });
    auto encode_diagnostics = detail::make_diagnostic_state<encode_stage_diagnostics_t>(
      config::sunshine.diagnostics_enabled
    );
    if (encode_diagnostics) {
      encode_diagnostics->performance = config.sbs_telemetry_performance;
    }
    // Where this loop's time goes (holds, waits, conversion, encode), one line every 20 s.
    auto loop_stats = detail::make_diagnostic_state<detail::encode_loop_stats_t>(config::sunshine.diagnostics_enabled);
    auto loop_stats_since = detail::diagnostic_timestamp(loop_stats.has_value(), std::chrono::steady_clock::now);
    detail::diagnostic_content_tracker_t loop_content;
    // Every bounded image wait (a pending capture's target, a retained-source poll, the keepalive)
    // ends at the first of a capture, a wake and its bound, and that bound on this high-resolution
    // timer rather than at the next scheduler tick (detail::pop_encode_image). Only this thread
    // waits on it, and the image event notifies it only during such a wait.
    const auto image_deadline_waiter = platf::create_deadline_waiter();
    if (loop_stats) {
      loop_stats->set_deadline_timer(image_deadline_waiter != nullptr);
    }
    // An independent provider's pending frame converts at its poll target. It sleeps to the target
    // on this timer, which captures and wakes do not end (detail::provider_hold); without the timer
    // it falls back to the image wait.
    std::unique_ptr<platf::high_precision_timer> hold_timer;
    if (independent_provider) {
      hold_timer = platf::create_high_precision_timer();
      if (hold_timer && !static_cast<bool>(*hold_timer)) {
        hold_timer.reset();
      }
    }
    // With pictures in flight (an independent provider or desktop Host SBS;
    // nvenc_base::pipeline_depth()) the next frame converts and its picture starts encoding while
    // the previous one still encodes. Conversion writes an unmapped input surface and submitted
    // pictures keep their input until retirement. A
    // thread of their own retrieves the pictures in submission order and publishes each packet as
    // soon as its picture completes, whatever this loop is doing. Declared after everything it
    // uses; its destruction on any exit delivers the pictures still in flight, as returning from
    // encode_frame() did, unless retrieval failed.
    std::optional<detail::encode_pipeline_t<encoded_frame_context_t>> pipeline;
    if (session->pipeline_depth() > 1) {
      pipeline.emplace(
        session->pipeline_depth(),
        [&, recovery_gate = detail::recovery_gate_t {}](encoded_frame_context_t &frame) mutable {
#ifdef SUNSHINE_TESTS
          if (test_hooks && test_hooks->before_retrieve) {
            test_hooks->before_retrieve(static_cast<int>(frame.frame_nr), frame.content_timestamp);
          }
#endif
          auto encoded_frame = session->retrieve_frame();
          if (!encoded_frame.data.empty() && !recovery_gate.admits(encoded_frame.idr)) {
            // Submitted before the recovery IDR was requested: it references the dropped frame.
            session->recycle_frame_buffer(std::move(encoded_frame.data));
            drop_report.record_awaiting_idr();
            return true;
          }
          const auto result = publish_encoded_frame(std::move(encoded_frame), frame, *session, packets, encode_diagnostics ? &*encode_diagnostics : nullptr);
          if (result.failed) {
            BOOST_LOG(error) << "Could not encode video packet"sv;
            return false;
          }
          recovery_gate.published(result.request_idr);
          drop_report.record(result, idr_events);
          if (result.request_idr) {
            // Service the recovery IDR at the same wait boundary as session control mail.
            images->wake();
          }
          return true;
        },
        loop_stats.has_value(),
        [images]() {
          // The loop notices the failure at its next iteration.
          images->wake();
        },
        []() {
          platf::adjust_thread_priority(platf::thread_priority_e::high);
        }
      );
      if (loop_stats) {
        loop_stats->set_pipeline_depth(pipeline->depth());
      }
    }
    const auto convert_frame = [&](platf::img_t &img, std::optional<std::chrono::steady_clock::time_point> target = std::nullopt) {
      if (encode_diagnostics) {
        encode_diagnostics->begin_conversion(img);
      }
      // Start after input-stat collection so periodic logger work is not charged to conversion.
      processing_started = std::chrono::steady_clock::now();
      const auto started = processing_started;
      const int result = target ? session->convert_with_encode_target(img, *target) : session->convert(img);
      if (loop_stats) {
        loop_stats->converted(std::chrono::steady_clock::now() - *started);
      }
      if (encode_diagnostics && encode_diagnostics->performance) {
        const auto measured_at = detail::diagnostic_clock_t::now();
        if (const auto elapsed = detail::diagnostic_elapsed_ms(started, measured_at)) {
          encode_diagnostics->performance->record(host_sbs_telemetry::stage::conversion, *elapsed, measured_at);
        }
      }
      return result;
    };

    auto try_reconfigure_pending_bitrate = [&]() {
      auto requested = video_mode_event->pop(0ms);
      if (!requested) {
        return false;
      }

      // The serial 0x3007 worker admits at most one request to this queue. Until the driver and
      // every publication below succeed, put that exact request back so capture_async's existing
      // proven-mode rebuild/fallback remains authoritative.
      auto restore_for_rebuild = util::fail_guard([&] {
        video_mode_event->raise(std::move(*requested));
      });

      const auto published_mode = config.effective_mode ?
                                    config.effective_mode->current() :
                                    effective_video_mode_t {};
      video_mode_change_t active {
        published_mode.source_width > 0 ?
          published_mode.source_width :
          (is_packed_mode(config.sbs_mode) ? config.width / 2 : config.width),
        published_mode.source_height > 0 ? published_mode.source_height : config.height,
        config.framerate,
        config.framerateX100,
        config.encodingFramerate,
        config.bitrate,
        0,
        0,
      };
      active.requested_sbs_mode = config.sbs_mode;
      if (!detail::is_nvenc_bitrate_only_reconfigure_candidate(
            active,
            *requested
          )) {
        return false;
      }

      // Rate control changes between pictures: none may still be in flight.
      if ((pipeline && !pipeline->drain()) || !session->reconfigure_bitrate(requested->bitrate)) {
        return false;
      }

      config.bitrate = requested->bitrate;
      effective_video_mode_t effective_mode = config.effective_mode ?
                                                published_mode :
                                                effective_video_mode_t {
                                                  active.width,
                                                  active.height,
                                                  active.framerateX100 > 0 ?
                                                    active.framerateX100 :
                                                    active.framerate * 100,
                                                  config.bitrate,
                                                  requested->width,
                                                  requested->height,
                                                  config.width,
                                                  config.height,
                                                  config.sbs_mode,
                                                  0,
                                                };
      effective_mode.bitrate = requested->bitrate;
      effective_mode.generation = next_effective_video_mode_generation(
        effective_mode.generation
      );
      if (config.effective_mode) {
        // Publish pacing only after NVENC accepts the new rate-control configuration. The sender
        // can therefore never pace at a bitrate that the live encoder rejected.
        config.effective_mode->publish(effective_mode);
      }
      on_bitrate_reconfigured(*requested);

      if (config.atomic_presentation) {
        (void) config.atomic_presentation->complete(video_mode_applied_t {
          {{requested->request_id, requested->transaction_id}},
          true,
          effective_mode,
          {
            requested->width,
            requested->height,
            requested->encodingFramerate,
          },
        });
      }
      BOOST_LOG(info) << "Applied live bitrate-only video mode at "sv
                      << requested->bitrate
                      << "kbps without rebuilding NVENC or forcing an IDR."sv;
      restore_for_rebuild.disable();
      return true;
    };

    auto lifecycle_change_requested = [&]() {
      // A picture that failed on the retrieving thread fails this encoder (`failed` stays set), as
      // a failed encode_frame() does. Its wake ends the loop's image wait.
      const auto pipeline_failed = [&] {
        return pipeline && pipeline->failed();
      };
      if (pipeline_failed()) {
        return true;
      }
      const bool shutting_down = shutdown_event->peek();
      const bool capture_stopped = !images->running();
      const bool display_reinit_pending = generation.revoked();

      // A completed encode_frame(), or a drained pipeline, has no NVENC work in flight, and all
      // submissions run on this thread. Use that serialized seam for a same-geometry/same-cadence
      // bitrate update. Any mismatch, unsupported GPU, or driver failure restores the request and
      // takes the unchanged lifecycle rebuild path below.
      if (!shutting_down && !capture_stopped && !display_reinit_pending && video_mode_event->peek() && try_reconfigure_pending_bitrate()) {
        return false;
      }
      if (pipeline_failed()) {
        return true;  // While draining for the reconfiguration.
      }
      // Any queued atomic presentation change that is not bitrate-only rebuilds the encode
      // session in place.
      const bool encode_config_change_pending = video_mode_event->peek();

      // If capture has to reinitialize before it has produced a frame, encode the dummy once so
      // Artemis knows the host is alive. An encode-config-only change always rebuilds immediately
      // because it changes the encode dimensions.
      if (!shutting_down && !capture_stopped && !(display_reinit_pending && frame_nr > 1) && !encode_config_change_pending) {
        return false;
      }

      // Deliver the pictures still in flight before the exit counts as clean: one that fails meanwhile
      // (its budget, a removed device) fails this encoder, so fail_guard tracks the teardown that new
      // encoders then wait for. Nothing is in flight afterwards, so nothing can fail later.
      if (pipeline && !pipeline->drain()) {
        return true;  // `failed` stays set.
      }
      failed = false;
      return true;
    };

    const auto poll_encode_requests = [&](bool &requested_idr_frame, bool &invalidated_ref_frames) {
      if (lifecycle_change_requested()) {
        return false;
      }

      // Only this encode thread mutates the converter. A setting is acknowledged after a real
      // conversion and successful encode, including retained-source conversion on a static desktop.
      stream_gamma.poll(*session);

      while (auto frames = invalidate_ref_frames_events->pop(0ms)) {
        // Only an actual invalidation waits for pictures in flight that may reference the lost
        // frames. The next submission confirms the recovery.
        if (pipeline && session->invalidation_needs_drain(frames->first, frames->second) && !pipeline->drain()) {
          return false;
        }
        invalidated_ref_frames |= session->invalidate_ref_frames(frames->first, frames->second);
      }
      if (pipeline && pipeline->failed()) {
        return false;
      }

      if (idr_events->pop(0ms)) {
        requested_idr_frame = true;
        session->request_idr_frame();
      }
      return true;
    };

    while (true) {
      if (loop_stats) {
        if (const auto now = std::chrono::steady_clock::now(); now - *loop_stats_since >= 20s) {
          if (pipeline) {
            const auto pictures = pipeline->take_stats();
            loop_stats->pictures(pictures.pushed_behind, pictures.retired, pictures.latency_total, pictures.latency_max);
          }
          BOOST_LOG(info) << loop_stats->report(now - *loop_stats_since) << session->video_memory_diagnostics();
          loop_stats->reset();
          loop_stats_since = now;
        }
        loop_stats->iteration();
      }

      bool requested_idr_frame = false;
      bool invalidated_ref_frames = false;
      if (!poll_encode_requests(requested_idr_frame, invalidated_ref_frames)) {
        break;
      }
      bool recovery_frame_requested = requested_idr_frame || invalidated_ref_frames;

      std::optional<std::chrono::steady_clock::time_point> frame_timestamp;
      bool converted_frame = false;
      bool conversion_ran = false;
      // A conversion that kept the encoder input (metadata for pixels an export no longer owns)
      // still retires its source and schedule, but leaves nothing new to encode.
      const auto finish_conversion = [&]() {
        conversion_ran = true;
        converted_frame = !session->conversion_kept_input();
        source.converted();
      };
      processing_started.reset();
      bool consume_sampled_depth_pipeline_ready = false;
      // One delivery owns the retained source and its schedule through conversion. Sample the
      // readiness event before converting, and commit cadence only after the converter returns.
      // `completion_wait_target` bounds the converter's wait for pending work, as the encode
      // target does, for a delivery that leaves the encode schedule as it is.
      const auto convert_delivery = [&](
        std::chrono::steady_clock::time_point presentation_timestamp,
        std::optional<std::chrono::steady_clock::time_point> next_encode_target = std::nullopt,
        std::optional<std::chrono::steady_clock::time_point> completion_wait_target = std::nullopt
      ) {
        consume_sampled_depth_pipeline_ready =
          depth_pipeline_ready_event && depth_pipeline_ready_event->peek();
        if (convert_frame(*last_img, next_encode_target ? next_encode_target : completion_wait_target)) {
          return false;
        }
        finish_conversion();
        frame_timestamp = presentation_timestamp;
        last_delivery_timestamp = presentation_timestamp;
        if (next_encode_target) {
          encode_frame_timestamp = *next_encode_target;
        }
        return true;
      };
      // A Game 3D session whose converter draws the captured picture itself for now, as it does
      // with a text game in front, is paced like desktop Host SBS for as long as that lasts:
      // served on every capture and on every completion of its depth, not at poll targets.
      const bool provider_paced = independent_provider && !session->capture_paced();
      const auto provider_poll_target = provider_paced ?
                                          encode_frame_timestamp - frame_variation_threshold :
                                          encode_frame_timestamp;

      // Idle keepalives preserve static image quality. Pending retained-source conversion is
      // serviced at the requested cadence instead of waiting for that slower heartbeat.
      // While the black startup input is held, even a recovery request waits for capture first.
      const bool hold_startup_input = startup_input_hold.active(std::chrono::steady_clock::now());
      bool pending_picture_ready = false;
      if (last_img && last_delivery_timestamp && !recovery_frame_requested && !hold_startup_input && !images->peek()) {
        // Until the next capture can arrive this thread would only wait. Work the last conversion
        // left pending goes on meanwhile; that conversion's own wait has already ended. A capture
        // at the stream's rate arrives one frame interval after the last one, and the step ends
        // a little before that, so that such a capture is not kept waiting.
        const auto next_capture_expected = *last_delivery_timestamp + encode_frame_threshold - detail::pending_conversion_advance_margin;
        if (std::chrono::steady_clock::now() < next_capture_expected) {
          pending_picture_ready = session->advance_pending_conversion(next_capture_expected);
        }
      }
      if (!recovery_frame_requested || images->peek() || hold_startup_input) {
        const bool conversion_poll_pending = last_img && (stream_gamma.pending() || session->needs_conversion_poll());
        const bool depth_pipeline_ready = depth_pipeline_ready_event && depth_pipeline_ready_event->peek();
        const auto wait_started = std::chrono::steady_clock::now();
        const auto keepalive_wait = detail::encode_keepalive_wait(wait_started, last_encode_at, keepalive_interval);
        auto pending_source_wait = source.remaining_wait(wait_started, provider_poll_target, provider_paced && conversion_poll_pending);
        if (provider_paced && last_img) {
          // The export's fence wake, captures and control requests end this wait; a pending frame
          // that is not yet due is held exactly to its poll target instead.
          const auto hold = hold_timer && !hold_startup_input && !recovery_frame_requested && !depth_pipeline_ready ?
                              detail::provider_hold(pending_source_wait, keepalive_wait, encode_frame_threshold) :
                              std::nullopt;
          if (hold) {
            hold_timer->sleep_for(*hold);
            if (loop_stats) {
              loop_stats->held(*hold, std::chrono::steady_clock::now() - wait_started);
            }
            // Take the latest capture without a second wait. Control mail is read before
            // converting or encoding it.
            pending_source_wait = std::chrono::nanoseconds::zero();
          }
        }
        // Nothing is encoded before capture delivers a frame while the startup input is held, so
        // only the hold's deadline bounds this wait: a due poll or keepalive must not spin it.
        const auto idle_wait = hold_startup_input ?
                                 startup_input_hold.wait(wait_started, keepalive_interval) :
                                 keepalive_wait;
        bool captured = false;
        auto image_wait = detail::encode_image_wait(idle_wait, encode_frame_threshold, last_img && !hold_startup_input, depth_pipeline_ready, conversion_poll_pending, pending_source_wait);
        if (pending_picture_ready) {
          // The picture is finished: a capture already here takes it along, and without one the
          // retained source is converted at once below. Neither waits out a frame interval.
          image_wait = {};
        }
#ifdef SUNSHINE_TESTS
        if (test_hooks && test_hooks->before_image_wait) {
          test_hooks->before_image_wait(static_cast<bool>(last_img), std::chrono::duration_cast<std::chrono::nanoseconds>(image_wait));
        }
#endif
        const auto image_wait_started = detail::diagnostic_timestamp(loop_stats.has_value(), std::chrono::steady_clock::now);
        auto img = detail::pop_encode_image(*images, image_wait, image_deadline_waiter.get());
        if (loop_stats && !safe::no_wait(image_wait)) {
          loop_stats->waited(std::chrono::duration_cast<std::chrono::nanoseconds>(image_wait), std::chrono::steady_clock::now() - *image_wait_started);
        }
        if (img) {
          // Metadata for pixels an external provider owns, with an unchanged cursor, only
          // replaces the retained source; the provider's own wake or the keepalive converts it.
          const bool new_content = img->conversion_needed;
          source.observe(std::move(img), new_content);
          captured = source.pending();
        } else if (!images->running()) {
          break;
        }
        // The same wake source carries client controls, sender recovery and display reinit.
        // Read requests raised during the wait before using either the new or retained input.
        if (!poll_encode_requests(requested_idr_frame, invalidated_ref_frames)) {
          break;
        }
        recovery_frame_requested = requested_idr_frame || invalidated_ref_frames;
        if (captured) {
          frame_timestamp = last_img->frame_timestamp;
          if (!frame_timestamp) {
            if (!missing_frame_timestamp_warning_logged) {
              BOOST_LOG(warning) << "Encoder received image without frame timestamp; substituting steady_clock::now()"sv;
              missing_frame_timestamp_warning_logged = true;
            }
            frame_timestamp = std::chrono::steady_clock::now();
            // Persist the substitute on the retained image. Host SBS uses this value as the
            // exact source identity when a later encode-thread timeout consumes its pending
            // inference; a local-only timestamp would make that safe completion unmatchable.
            last_img->frame_timestamp = frame_timestamp;
          }

          auto current_timestamp = *frame_timestamp;
          auto time_diff = current_timestamp - encode_frame_timestamp;

          // Defer an early presentation, but retain the source until conversion. If capture
          // stops here, the pending-source deadline below must still display these final pixels.
          // A capture that may replace a held startup input is never early: nothing was encoded.
          if (time_diff < -frame_variation_threshold) {
            const auto now = std::chrono::steady_clock::now();
            if (!source.due(now, provider_poll_target, recovery_frame_requested || hold_startup_input)) {
              continue;
            }
            // A stale capture timestamp must not starve a source whose presentation is due.
            // This is only the encode schedule; retain the original source/content identity.
            current_timestamp = now;
          }

          const auto schedule = detail::select_encode_frame_schedule(
            current_timestamp,
            encode_frame_timestamp,
            encode_frame_threshold,
            frame_variation_threshold
          );
          if (!convert_delivery(schedule.presentation_timestamp, schedule.next_encode_target)) {
            BOOST_LOG(error) << "Could not convert image"sv;
            break;
          }
        }
      }

      // A final early capture may never be followed by another image event. Convert it when
      // its scheduled presentation is due, before falling back to the slower idle keepalive.
      // IDR recovery uses the newest retained source immediately. The single source slot is
      // replaced by newer captures, so waiting cannot accumulate a queue of deferred images.
      if (!conversion_ran && source.pending() && last_img) {
        const auto now = std::chrono::steady_clock::now();
        // An independent provider's wait ends at its poll target; convert there rather than
        // waking repeatedly until the later presentation target.
        if (source.due(now, provider_poll_target, recovery_frame_requested)) {
          if (lifecycle_change_requested()) {
            break;
          }
          const auto schedule = detail::select_encode_frame_schedule(
            now,
            encode_frame_timestamp,
            encode_frame_threshold,
            frame_variation_threshold
          );
          if (!convert_delivery(schedule.presentation_timestamp, schedule.next_encode_target)) {
            BOOST_LOG(error) << "Could not convert deferred capture image"sv;
            break;
          }
        }
      }

      if (!conversion_ran && stream_gamma.pending() && last_img) {
        if (lifecycle_change_requested()) {
          break;
        }
        const auto now = std::chrono::steady_clock::now();
        frame_timestamp = now;
        if (convert_frame(*last_img)) {
          BOOST_LOG(error) << "Could not convert the retained source with stream gamma"sv;
          break;
        }
        finish_conversion();
        if (provider_paced && !converted_frame) {
          // Kept input: the gamma request stays pending for the next real source. Poll for it at
          // stream cadence; with the past target left as it was, the next wait would be zero.
          const auto schedule = detail::select_encode_frame_schedule(
            now, encode_frame_timestamp, encode_frame_threshold, frame_variation_threshold
          );
          encode_frame_timestamp = schedule.next_encode_target;
        }
      }

      // Host SBS initializes only its per-stream D3D/CUDA resources in the background; the model
      // engine and execution context are already process-resident. If initialization completes
      // while the desktop is static, reconvert the retained source once so the ready pipeline is
      // installed and the client receives depth immediately instead of waiting for desktop motion.
      if (!conversion_ran && depth_pipeline_ready_event && depth_pipeline_ready_event->peek() && last_img) {
        if (lifecycle_change_requested()) {
          break;
        }
        if (!convert_delivery(std::chrono::steady_clock::now())) {
          BOOST_LOG(error) << "Could not activate the initialized Host SBS GPU pipeline"sv;
          break;
        }
      }

      // Host SBS inference may still be pending when capture goes idle. Reconvert the retained
      // source once on this encode-thread timeout so the device can poll its completion without
      // blocking and keep D3D rendering on its normal owner. A busy poll keeps the exact slot
      // pending and suppresses duplicate inference for those same pixels. An independent provider
      // also converts for its keepalive, which re-checks the export while no frame arrives.
      const bool provider_keepalive_due = provider_paced && std::chrono::steady_clock::now() >= last_encode_at + keepalive_interval;
      if (!conversion_ran && last_img && (provider_keepalive_due || session->needs_conversion_poll())) {
        const auto now = std::chrono::steady_clock::now();
        // A faster minimum-FPS heartbeat may encode retained output before this deadline,
        // but must not advance an independent provider's schedule ahead of actual time.
        // Host AI completion is served at once, on a schedule left as it is. Its work has had a
        // whole frame interval by now: the converter may wait for what is left of it for as
        // long as a conversion of a new capture would, since no newer capture can be admitted
        // before this one's work retires.
        if (!provider_paced || source.due(now, provider_poll_target, recovery_frame_requested, true)) {
          if (lifecycle_change_requested()) {
            break;
          }
          const auto schedule = provider_paced ?
                                  std::optional {detail::select_encode_frame_schedule(
                                    now, encode_frame_timestamp, encode_frame_threshold, frame_variation_threshold
                                  )} :
                                  std::nullopt;
          // The picture delivered here is that of the retained capture, however late its work
          // finished. Its presentation time, which becomes the packet timestamp, is the time of
          // that capture: a client that spaces frames by it shows the picture where it belongs.
          if (!convert_delivery(
                schedule ? schedule->presentation_timestamp : last_img->frame_timestamp.value_or(now),
                schedule ? std::optional {schedule->next_encode_target} : std::nullopt,
                schedule ? std::nullopt : std::optional {now + encode_frame_threshold}
              )) {
            BOOST_LOG(error) << "Could not consume pending Host SBS depth for retained source"sv;
            break;
          }
        }
      }

      if (conversion_ran && consume_sampled_depth_pipeline_ready && depth_pipeline_ready_event) {
        // Consume only readiness observed before conversion. A build that finishes during the
        // conversion remains armed so a static retained source adopts it on the next loop. A
        // conversion that kept the input consumes it too: the next real source installs it.
        depth_pipeline_ready_event->pop(0ms);
      }

      if (converted_frame) {
        startup_input_hold.replaced();
      } else if (startup_input_hold.active(std::chrono::steady_clock::now())) {
        // The encoder input is still the black startup dummy while capture re-reads the desktop:
        // neither a keepalive nor a recovery request encodes it. An IDR stays latched.
        continue;
      }

      const bool wake_without_picture = !converted_frame && (!conversion_ran || independent_provider);
      if (wake_without_picture && !recovery_frame_requested && std::chrono::steady_clock::now() < last_encode_at + keepalive_interval) {
        // A wake can end the wait without authorizing a picture. Preserve the idle deadline.
        continue;
      }

      // Keep this check as close as possible to encode(). A reinit may be requested while the
      // current frame is being converted, and encoding afterward can leave packets or GPU work in
      // flight while the session and display resources are torn down.
      if (lifecycle_change_requested()) {
        break;
      }

      if (
        conversion_ran && session->conversion_repeats_pending_picture() &&
        !first_encoder_output && !requested_idr_frame && !invalidated_ref_frames &&
        !(stream_gamma.pending() && config.stream_gamma_state) &&
        std::chrono::steady_clock::now() < last_encode_at + keepalive_interval
      ) {
        // The work of this capture ran past the conversion's wait, and the converter drew the
        // previous picture again. That picture went out already. The new one is sent as soon as
        // it is ready, so a second copy of the old one would only delay it at the client.
#ifdef SUNSHINE_TESTS
        if (test_hooks && test_hooks->repeat_coalesced) {
          test_hooks->repeat_coalesced();
        }
#endif
        continue;
      }

      const auto encode_started = detail::diagnostic_timestamp(loop_stats.has_value(), std::chrono::steady_clock::now);
      if (pipeline && session->input_waits_for_retirement()) {
        // These exact pixels already have a submitted picture. Keep servicing capture and depth
        // while that picture retires. Recovery, gamma proof and due keepalives still deliver.
        if (
          !first_encoder_output && !requested_idr_frame && !invalidated_ref_frames &&
          !(stream_gamma.pending() && config.stream_gamma_state) &&
          std::chrono::steady_clock::now() < last_encode_at + keepalive_interval
        ) {
          session->retire_input_reads();
#ifdef SUNSHINE_TESTS
          if (test_hooks && test_hooks->repeat_coalesced) {
            test_hooks->repeat_coalesced();
          }
#endif
          continue;
        }
        if (!pipeline->drain() || lifecycle_change_requested()) {
          break;
        }
      }
      const auto content_timestamp = session->rendered_content_timestamp();
      // A keepalive or poll that re-renders an unchanged export still carries the same content.
      const bool new_loop_content = loop_stats && loop_content.observe(content_timestamp, !converted_frame) == detail::diagnostic_content_e::new_content;
      auto frame = make_encoded_frame_context(
        frame_nr++,
        frame_timestamp,
        content_timestamp,
        encode_diagnostics ? &*encode_diagnostics : nullptr,
        converted_frame,
        processing_started,
        first_encoder_output
      );
      bool encoded = false;
      if (pipeline) {
        // Waits only while every picture is in flight, and then for the oldest one. The claim above
        // stays at its poll target: waiting for a slot before it would rebase the stream cadence
        // after every slow picture (RemoteEncodeProviderPacingTest.
        // ClaimingAtThePollTargetKeepsTheCadenceWhenEveryPictureIsInFlight). The picture is
        // published by the retrieving thread when it completes.
        encoded = pipeline->acquire();
        const auto completion_finished = detail::diagnostic_timestamp(loop_stats.has_value(), std::chrono::steady_clock::now);
        encoded = encoded && session->submit_frame(frame.frame_nr);
        if (encoded) {
#ifdef SUNSHINE_TESTS
          if (test_hooks && test_hooks->after_submit) {
            test_hooks->after_submit(static_cast<int>(frame.frame_nr), frame.content_timestamp);
          }
#endif
          pipeline->push(std::move(frame));
        }
        if (loop_stats) {
          loop_stats->encoded(new_loop_content, std::chrono::steady_clock::now() - *encode_started, session->last_submit_timing(), *completion_finished - *encode_started);
        }
      } else {
        const auto publish_result = encode(*session, packets, frame, encode_diagnostics ? &*encode_diagnostics : nullptr);
        encoded = !publish_result.failed;
        if (loop_stats) {
          const auto timing = session->last_frame_timing();
          const auto call = std::chrono::steady_clock::now() - *encode_started;
          loop_stats->encoded(new_loop_content, call, timing.submit, timing.completion_wait);
          if (encoded) {
            loop_stats->pictures(0, 1, call, call);
          }
        }
        if (encoded) {
          drop_report.record(publish_result, idr_events);
        }
      }
      if (!encoded) {
        BOOST_LOG(error) << "Could not encode video packet"sv;
        break;
      }
      // A submitted picture counts as encoded for the keepalive and the stream-gamma proof: it is
      // delivered unless the encoder fails, which rebuilds the session and reapplies the request.
      last_encode_at = std::chrono::steady_clock::now();
      if (converted_frame && session->rendered_content_timestamp() && config.stream_gamma_state) {
        stream_gamma.prove(*session);
      }
      first_encoder_output = false;

      session->request_normal_frame();
      refresh_mouse_keys_if_due(next_mouse_keys_refresh);
    }
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

  std::unique_ptr<nvenc_encode_session_t> make_encode_session(
    platf::display_t &disp,
    const config_t &config,
    bool enforce_launch_hdr_contract = false
  ) {
    if (enforce_launch_hdr_contract) {
      const auto process_status = proc::proc.get_status();
      if (!hdr_stream_negotiation_is_coherent(process_status.enable_hdr, config.dynamicRange)) {
        BOOST_LOG(error)
          << "Rejecting incoherent color session: HTTP hdrMode="sv
          << process_status.enable_hdr << " but RTSP dynamicRange="sv
          << config.dynamicRange << '.';
        return nullptr;
      }
    }

    auto colorspace = colorspace_from_client_config(config, disp.is_hdr());

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

    auto device = disp.make_nvenc_encode_device(config, colorspace);
    if (!device) {
      return nullptr;
    }
    return std::make_unique<nvenc_encode_session_t>(std::move(device));
  }

#ifdef SUNSHINE_TESTS
  namespace detail {
    native_encode_loop_test_result_t run_native_encode_loop_for_test(
      std::shared_ptr<platf::display_t> display,
      safe::mail_t mail,
      img_event_t images,
      config_t config,
      packet_queue_t packets,
      const native_encode_loop_test_hooks_t &hooks
    ) {
      if (!display || !mail || !images || !packets) {
        throw std::invalid_argument("The native encode-loop fixture requires its display, mail, images and packet queue");
      }
      const auto drain = util::fail_guard([]() {
        async_teardown_workers().drain_for_test();
      });
      capture_generation_handoff_t generations;
      const auto generation = generations.begin();
      if (!generation || !platf::arm_capture_display_retirement(display, generation->retirement()) ||
          !generations.publish(generation, display)) {
        throw std::logic_error("The native encode-loop fixture requires an original unarmed capture display");
      }
      auto lease = generations.acquire();
      if (!lease) {
        throw std::logic_error("The native encode-loop fixture could not acquire its published display generation");
      }
      auto revoke = util::fail_guard([&]() {
        generations.stop();
      });
      native_encode_loop_test_result_t result;
      auto session = make_encode_session(*display, config, false);
      if (!session) {
        return result;
      }
      result.session_created = true;
      result.pipeline_depth = session->pipeline_depth();
      encoder_recovery_t recovery;
      encode_run(
        result.next_frame_nr,
        result.awaiting_desktop_after_export,
        std::move(mail),
        std::move(images),
        std::move(config),
        std::move(lease),
        std::move(session),
        recovery,
        std::move(packets),
        [](const video_mode_change_t &) {},
        &hooks
      );
      return result;
    }

    void drain_async_teardown_workers_for_test() noexcept {
      async_teardown_workers().drain_for_test();
    }

    std::unique_ptr<nvenc_encode_session_t> make_native_encode_session_for_test(
      platf::display_t &display,
      const config_t &config,
      bool enforce_launch_hdr_contract
    ) {
      return make_encode_session(display, config, enforce_launch_hdr_contract);
    }

    hdr_info_t native_encode_session_hdr_info_for_test(const nvenc_encode_session_t &session) {
      return session.hdr_info();
    }

    std::future<void> retire_native_encode_session_for_test(std::unique_ptr<nvenc_encode_session_t> session) {
      return session ? std::move(*session).retire() : std::future<void> {};
    }
  }  // namespace detail
#endif

  void capture_async(
    safe::mail_t mail,
    config_t &config,
    packet_queue_t packets
  ) {
    auto shutdown_event = mail->event<bool>(mail::shutdown);

    auto images = std::make_shared<img_event_t::element_type>();
    auto lg = util::fail_guard([&]() {
      images->stop();
      shutdown_event->raise(true);
    });

    capture_session_t capture_session {images, config};
    if (!capture_session.running()) {
      return;
    }

    int frame_nr = 1;
    // Set when an encode loop ends with an input that had no pixels; see startup_input_hold_t.
    bool awaiting_desktop_after_export = false;
    detail::encoder_recovery_t recovery;
    bool recovery_wait_logged = false;

    auto touch_port_event = mail->event<input::touch_port_t>(mail::touch_port);
    auto hdr_event = mail->event<hdr_info_t>(mail::hdr);

    auto sbs_depth_status_event = mail->event<int>(mail::sbs_depth_status);
    auto sbs_telemetry_event =
      mail->event<sbs_telemetry_snapshot_t>(mail::sbs_telemetry);
    auto game_source_status_event = mail->event<game_source_state_t>(mail::game_source_status);
    // Live client-requested geometry/rate/bitrate change (0x3007 control message). It rewrites
    // `config`, which every derived value below is rebuilt from on each iteration. This is a queue
    // rather than a single-slot event: every request owes the client an acknowledgement, so a
    // burst arriving inside one frame must not silently overwrite the earlier requests' ids.
    auto video_mode_event = mail->queue<video_mode_change_t>(mail::video_mode);
    live_mode_state_t live_mode {config};

    // Encoding takes place on this thread
    platf::adjust_thread_priority(platf::thread_priority_e::high);

    while (!shutdown_event->peek() && images->running()) {
      // Drain live video-mode changes before the reinitialization gate. Capture may already be
      // rebuilding its display, and it has to see the new mode before it does so.
      while (video_mode_event->peek()) {
        if (auto mode = video_mode_event->pop(0ms)) {
          live_mode.apply_request(*mode, config, capture_session, sbs_depth_status_event);
        }
      }

      // A failed encode retires on a tracked worker. Wait before acquiring the display or
      // attempting another encoder, and keep servicing cancellation/live-mode requests above.
      // Waiting only on this owner cannot stall behind unrelated process teardown workers.
      const auto recovery_state = recovery.poll();
      if (recovery_state == detail::encoder_recovery_t::state_t::pending) {
        if (!recovery_wait_logged) {
          BOOST_LOG(info) << "Waiting for failed encoder teardown before rebuilding the stream"sv;
          recovery_wait_logged = true;
        }
        std::this_thread::sleep_for(20ms);
        continue;
      }
      if (recovery_state != detail::encoder_recovery_t::state_t::ready) {
        BOOST_LOG(error) << (recovery_state == detail::encoder_recovery_t::state_t::timed_out ? "Failed encoder teardown exceeded the 10-second recovery window; ending the stream"sv : "Failed encoder teardown raised an exception; ending the stream"sv);
        return;
      }
      if (recovery_wait_logged) {
        BOOST_LOG(info) << "Failed encoder teardown complete; rebuilding the stream"sv;
        recovery_wait_logged = false;
      }

      // Acquisition and revocation share one admission lock. A lease stays bound to this
      // generation through the entire encode loop, including its retained-source return guard.
      auto generation = capture_session.try_acquire_display();
      if (!generation) {
        // Keep servicing live-mode requests while capture opens or retires a display.
        std::this_thread::sleep_for(capture_session.reinit_pending() ? 20ms : 1ms);
        continue;
      }
      const auto &display = generation.display();

      // Retune the running capture session every iteration rather than only when the mode
      // changes. The call is a no-op when the cadence already matches, and republishing it
      // unconditionally means a display rebuilt by the capture worker is corrected here even if it
      // raced the mode change and was created with the previous rate.
      display->set_client_frame_rate(config.framerate, config.framerateX100);

      const auto host_sbs_rejection =
        models::host_sbs_v2_source_resolution_rejection_reason(
          static_cast<std::uint32_t>(live_mode.base_width()),
          static_cast<std::uint32_t>(config.height)
        );
      live_mode.refuse_unsupported_host_sbs(host_sbs_rejection, config, sbs_depth_status_event);
      // This iteration's encode session is built from one live mode. Only a revert changes it,
      // and every revert starts the next iteration.
      const int base_width = live_mode.base_width();
      const int current_sbs_mode = live_mode.sbs_mode();
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
      // Remote provider selection is explicit and immutable within this presentation request.
      const bool game_provider = is_game_mode(current_sbs_mode);
      session_config.game_source_status_event = is_game_mode(current_sbs_mode) ? game_source_status_event : nullptr;
      session_config.game_source_width = base_width;
      session_config.game_source_height = config.height;
      session_config.sbs_telemetry_event = sbs_telemetry_event;
      session_config.sbs_telemetry_generation = next_sbs_telemetry_generation();
      session_config.sbs_telemetry_performance = config::sunshine.diagnostics_enabled && current_sbs_mode == SBS_AI ?
                                                   std::make_shared<host_sbs_telemetry::collector>() :
                                                   nullptr;
      // Invalidate the previous renderer's state before any encoder/GPU initialization work. This
      // is CPU-only and does not activate telemetry sampling; the render path remains completely
      // dormant until the client's subscription latch is enabled.
      sbs_telemetry_event->raise(make_unavailable_sbs_telemetry_snapshot(
        session_config.sbs_telemetry_generation,
        session_config.sbs_config
      ));
      int runtime_max_width = nvenc::max_encode_width_for_codec(session_config.videoFormat).value_or(0);
      int runtime_max_height = nvenc::max_encode_height_for_codec(session_config.videoFormat).value_or(0);
      const auto game_packed_dimensions = host_sbs_output_dimensions(
        base_width,
        config.height,
        session_config.videoFormat,
        session_config.sbs_config.max_encode_width,
        runtime_max_width,
        runtime_max_height
      );
      session_config.game_source_transport_supported = external_sbs_dimensions_match(
        base_width,
        config.height,
        game_packed_dimensions.width,
        game_packed_dimensions.height
      );
      if (is_packed_mode(current_sbs_mode)) {
        const auto dimensions = host_sbs_output_dimensions(
          base_width,
          config.height,
          session_config.videoFormat,
          session_config.sbs_config.max_encode_width,
          runtime_max_width,
          runtime_max_height
        );
        session_config.width = dimensions.width;
        session_config.height = dimensions.height;
        const std::int64_t packed_width = static_cast<std::int64_t>(base_width) * 2;
        if (!game_provider && (session_config.width != packed_width || session_config.height != config.height)) {
          BOOST_LOG(info) << "Host SBS: requested packed dimensions "sv << packed_width
                          << 'x' << config.height
                          << " exceed the effective encoder limits; capping to "sv
                          << session_config.width << 'x'
                          << session_config.height << " (per-eye "sv
                          << (session_config.width / 2) << 'x' << session_config.height << ')';
        }
      } else {
        // A live 0x3007 change can request any width the wire format can carry, and the control
        // thread deliberately does not reject an oversized one: a failed non-SBS encoder creation
        // returns from capture_async and kills the session, whereas capping always succeeds.
        const auto dimensions = clamp_encode_dimensions(
          base_width,
          config.height,
          session_config.videoFormat,
          runtime_max_width,
          runtime_max_height
        );
        session_config.width = dimensions.width;
        session_config.height = dimensions.height;
        if (dimensions.width != base_width || dimensions.height != config.height) {
          BOOST_LOG(info) << "Requested encode dimensions "sv << base_width << 'x'
                          << config.height
                          << " exceed the effective encoder limits; capping to "sv
                          << session_config.width << 'x' << session_config.height;
        }
      }
      BOOST_LOG(info) << "Encode session: Host SBS mode "sv
                      << current_sbs_mode << ", strength "sv
                      << session_config.sbs_config.pop_strength
                      << ", provider '"sv << (game_provider ? "ReShade" : host_sbs_v2_depth_model().name) << "', output "sv
                      << session_config.width << 'x' << session_config.height;

      auto recover_failed_sbs_session = [&]() {
        if (session_config.sbs_mode != SBS_AI) {
          return false;
        }
        const int refreshed_max_width = nvenc::max_encode_width_for_codec(
                                          session_config.videoFormat
        )
                                          .value_or(0);
        const int refreshed_max_height = nvenc::max_encode_height_for_codec(
                                           session_config.videoFormat
        )
                                           .value_or(0);
        const bool learned_stricter_limit =
          (refreshed_max_width > 0 && refreshed_max_width < session_config.width) ||
          (refreshed_max_height > 0 && refreshed_max_height < session_config.height);
        if (learned_stricter_limit && (refreshed_max_width != runtime_max_width || refreshed_max_height != runtime_max_height)) {
          BOOST_LOG(info) << "Host SBS learned lower runtime NVENC limits ("sv
                          << refreshed_max_width << 'x' << refreshed_max_height
                          << "); retrying with aspect-preserving SBS scaling."sv;
          return true;
        }
        BOOST_LOG(error) << "Failed to create encoder at codec-safe SBS resolution "sv
                         << session_config.width << 'x' << session_config.height
                         << "; refusing to silently replace the requested SBS stream with flat 2D."sv;
        sbs_depth_status_event->raise(0);
        return false;
      };

      if (current_sbs_mode == SBS_GAME_SBS && !external_sbs_dimensions_match(base_width, config.height, session_config.width, session_config.height)) {
        BOOST_LOG(error) << "ReShade Host 3D cannot encode the exact final-SBS raster "sv
                         << static_cast<std::int64_t>(base_width) * 2 << 'x' << config.height
                         << "; refusing to resize authored stereo."sv;
        if (live_mode.revert_failed_request(config, capture_session, sbs_depth_status_event)) {
          continue;
        }
        return;
      }

      auto encode_session = make_encode_session(*display, session_config, true);
      if (!encode_session) {
        if (recover_failed_sbs_session()) {
          continue;
        }
        if (live_mode.revert_failed_request(config, capture_session, sbs_depth_status_event)) {
          continue;
        }
        return;
      }

      // Only the fully initialized session may publish HDR, become the safe mode fallback,
      // or complete a live mode acknowledgement.
      auto hdr_info = encode_session->hdr_info();
      live_mode.prove_encoder(config);

      // Publish what this session really runs. SBS packs two eyes into one encoded frame, so the
      // base per-eye width is half the encoded width; both are reported after the codec-width cap,
      // which can differ from the request. The bitrate is the encoder budget, not the wire budget.
      effective_video_mode_t effective_mode {
        is_packed_mode(session_config.sbs_mode) ? session_config.width / 2 : session_config.width,
        session_config.height,
        // A launch that never negotiated an exact refresh rate leaves framerateX100 at zero. The
        // client is owed a real rate, so fall back to the whole-frames-per-second figure.
        session_config.framerateX100 > 0 ? session_config.framerateX100 : session_config.framerate * 100,
        session_config.bitrate,
        config.width,
        config.height,
        session_config.width,
        session_config.height,
        session_config.sbs_mode,
        0,
      };
      if (config.effective_mode) {
        effective_mode.generation = next_effective_video_mode_generation(
          config.effective_mode->current().generation
        );
      } else {
        effective_mode.generation = 1;
      }
      if (config.effective_mode) {
        // This is also the packet sender's pacing publication. It must remain after successful
        // encoder creation: a speculative request that fails never changes effective cadence.
        config.effective_mode->publish(effective_mode);
      }
      const desktop_video_mode_t desktop_mode {
        config.width,
        config.height,
        config.encodingFramerate,
      };

      live_mode.complete_requests(config.atomic_presentation, effective_mode, desktop_mode);

      // absolute mouse coordinates require that the dimensions of the screen are known
      // Host SBS changes only the encoded frame geometry. Absolute input remains addressed in
      // the client's logical W x H viewport and must map to the captured desktop, not the packed
      // 2W x H encoder surface (or its capped derivative).
      touch_port_event->raise(make_port(display.get(), config));

      // Update client with our current HDR display state
      hdr_event->raise(std::move(hdr_info));

      encode_run(
        frame_nr,
        awaiting_desktop_after_export,
        mail,
        images,
        session_config,
        generation,
        std::move(encode_session),
        recovery,
        packets,
        [&](const video_mode_change_t &mode) {
          live_mode.accept_reconfigured_bitrate(mode, config, capture_session);
        }
      );
    }
  }

  void capture(
    safe::mail_t mail,
    config_t config,
    packet_queue_t packets
  ) {
    // Initialize this stream's depth status independently of every other session.
    mail->event<int>(mail::sbs_depth_status)->raise(0);
    mail->event<bool>(mail::idr)->raise(true);
    capture_async(std::move(mail), config, std::move(packets));
  }

  int validate_config(std::shared_ptr<platf::display_t> disp, const config_t &config) {
    auto session = make_encode_session(*disp, config);
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

    auto packets = std::make_shared<packet_queue_t::element_type>(ENCODED_PACKET_QUEUE_LIMIT);
    while (!packets->peek()) {
      if (encode(*session, packets, make_encoded_frame_context(1, {}, {}, nullptr, true, {}, true), nullptr).failed) {
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

#ifdef SUNSHINE_TESTS
  namespace detail {
    struct capture_worker_test_t::impl_t {
      capture_session_t capture;

      impl_t(
        config_t config,
        img_event_t images,
        capture_worker_test_callback_t callback,
        void *state,
        capture_worker_test_callback_t before_admission
      ):
          capture(std::move(images), config, callback, state, before_admission) {
      }
    };

    capture_worker_test_t::capture_worker_test_t(
      config_t config,
      img_event_t images,
      capture_worker_test_callback_t callback,
      void *state,
      capture_worker_test_callback_t before_admission
    ) {
      if (!images || !callback) {
        throw std::invalid_argument("The inert capture worker requires images and a callback");
      }
      impl_ = std::make_unique<impl_t>(std::move(config), std::move(images), callback, state, before_admission);
    }

    capture_worker_test_t::~capture_worker_test_t() = default;

    void capture_worker_test_t::stop() {
      impl_->capture.stop();
    }

    bool capture_worker_test_t::running() const {
      return impl_->capture.running();
    }

    void capture_worker_test_t::set_live_video_mode(std::optional<video_mode_change_t> mode) {
      impl_->capture.set_live_video_mode(std::move(mode));
    }

    std::optional<video_mode_change_t> capture_worker_test_t::live_video_mode() const {
      return impl_->capture.live_video_mode();
    }

    struct capture_reinit_test_t::impl_t {
      capture_session_t capture;

      // A test opens and reopens through the same operation, and its repair takes no argument.
      static capture_platform_t platform_of(capture_reinit_test_hooks_t hooks) {
        capture_platform_t platform;
        platform.preferred_display_name = std::move(hooks.preferred_display_name);
        platform.set_display_name = std::move(hooks.set_display_name);
        platform.refresh_displays = std::move(hooks.refresh_displays);
        platform.reopen_display = hooks.open_display;
        platform.open_display = std::move(hooks.open_display);
        platform.adjust_thread_priority = std::move(hooks.adjust_thread_priority);
        platform.display_change_epoch = std::move(hooks.display_change_epoch);
        platform.reconcile_topology = [reconcile = std::move(hooks.reconcile_topology)](const std::function<bool()> &) {
          return reconcile();
        };
        platform.waited_for_display_release = std::move(hooks.waited_for_display_release);
        return platform;
      }

      impl_t(config_t config, img_event_t images, capture_reinit_test_hooks_t hooks):
          capture(std::move(images), config, nullptr, nullptr, nullptr, platform_of(std::move(hooks))) {
      }
    };

    capture_reinit_test_t::capture_reinit_test_t(config_t config, img_event_t images, capture_reinit_test_hooks_t hooks) {
      if (!images || !hooks.preferred_display_name || !hooks.set_display_name || !hooks.refresh_displays || !hooks.open_display || !hooks.adjust_thread_priority || !hooks.display_change_epoch || !hooks.reconcile_topology) {
        throw std::invalid_argument("The full inert capture worker requires every physical operation");
      }
      impl_ = std::make_unique<impl_t>(std::move(config), std::move(images), std::move(hooks));
    }

    capture_reinit_test_t::~capture_reinit_test_t() = default;

    void capture_reinit_test_t::stop() {
      impl_->capture.stop();
    }

    bool capture_reinit_test_t::running() const {
      return impl_->capture.running();
    }

    bool capture_reinit_test_t::reinit_pending() const {
      return impl_->capture.reinit_pending();
    }

    std::shared_ptr<platf::display_t> capture_reinit_test_t::current_display() {
      return impl_->capture.try_acquire_display().display();
    }

    void capture_reinit_test_t::set_live_video_mode(std::optional<video_mode_change_t> mode) {
      impl_->capture.set_live_video_mode(std::move(mode));
    }
  }  // namespace detail
#endif

}  // namespace video
