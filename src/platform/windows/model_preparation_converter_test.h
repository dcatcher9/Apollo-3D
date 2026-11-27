#pragma once

#ifdef SUNSHINE_TESTS
  #include "src/video.h"
  #include "src/video_depth_estimator.h"
  #include "host_sbs_converter_test_observer.h"

  #include <atomic>
  #include <chrono>
  #include <cstdint>
  #include <dxgi1_4.h>
  #include <filesystem>
  #include <future>
  #include <memory>
  #include <span>
  #include <string>
  #include <string_view>
  #include <vector>

namespace platf::dxgi {
  class display_vram_t;
}

namespace platf::dxgi::testing {
  struct model_preparation_adapter_t {
    std::uint32_t luid_low = 0;
    std::int32_t luid_high = 0;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
  };

  struct native_pacing_delivery_t {
    std::chrono::steady_clock::time_point started, finished;
    std::optional<std::chrono::steady_clock::time_point> source_timestamp;
    int result = -1;
  };

  struct native_pacing_poll_t {
    std::uint64_t conversion = 0;
    bool prior = false, eligible = false;
    std::chrono::steady_clock::time_point deadline;
    bool ready = false, timed_out = false;
    std::uint32_t query_count = 0;
    std::chrono::steady_clock::duration wait_duration;
    bool notification_armed = false;
    bool notification_waited = false;
    bool notification_callback_pending = false;
  };

  struct native_pacing_fixture_t {
    const std::atomic<std::int64_t> *held_source_stamp_ticks = nullptr;
    unsigned hold_ms = 120;
    converter_test_observer::observer_t converter_observer;
    void (*record_delivery)(void *, const native_pacing_delivery_t &) noexcept = nullptr;
    void (*record_poll)(void *, const native_pacing_poll_t &) noexcept = nullptr;
  };

  // No desktop duplication or output window. The native encoder sees the same shared
  // capture image and keyed mutex it receives from a normal display.
  std::shared_ptr<display_vram_t> make_native_pacing_display(
    const model_preparation_adapter_t &adapter, unsigned width, unsigned height,
    const native_pacing_fixture_t &probe = {}
  );
  std::shared_ptr<platf::img_t> upload_native_pacing_image(
    display_vram_t &display, std::span<const std::uint32_t> bgra
  );

  /** Actual local-presentation converter, opened as the AR and shutter presenters open theirs,
   * on a native pacing display whose probe observes it as it observes the native encoder.
   * Construct, convert and destroy it on its one conversion-owner thread.
   */
  class native_pacing_local_converter_t {
  public:
    native_pacing_local_converter_t(std::shared_ptr<display_vram_t> display, video::sbs_mode_e mode);
    ~native_pacing_local_converter_t();
    native_pacing_local_converter_t(const native_pacing_local_converter_t &) = delete;
    native_pacing_local_converter_t &operator=(const native_pacing_local_converter_t &) = delete;

    /** Convert into an owned RGB target within the presenter's render-budget deadline. */
    bool convert(platf::img_t &image, std::chrono::steady_clock::time_point presentation_target);

    /** Take the depth pipeline's terminal flat fallback, as a failed estimator does. */
    void fail_depth_pipeline_flat();
    bool estimator_present() const;
    /** Failed estimators kept until this converter's device is destroyed. */
    std::size_t parked_failed_estimators() const;
    /** The converter's own D3D11 device (not owned). */
    ID3D11Device *device() const;

  private:
    class impl_t;
    std::unique_ptr<impl_t> impl_;
  };

  struct model_preparation_memory_t {
    model_preparation_adapter_t adapter;
    std::string description;
    long local_status = E_FAIL;
    long nonlocal_status = E_FAIL;
    DXGI_QUERY_VIDEO_MEMORY_INFO local {};
    DXGI_QUERY_VIDEO_MEMORY_INFO nonlocal {};
  };

  // Queries DXGI on the explicitly bound adapter. Does not load CUDA or create a D3D device.
  model_preparation_memory_t query_model_preparation_memory(const model_preparation_adapter_t &adapter);

  struct model_preparation_converter_request_t {
    model_preparation_adapter_t adapter;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> bgra;
    std::filesystem::path output_root;
    // Opt-in original consumer path for registry readiness; default preserves the B41 fixture.
    bool production_observer_path = false;
    // TEST-only ineligible synthetic context with real committed DDup damage history.
    bool hard_disabled_dirty_source = false;
    std::shared_future<std::vector<std::wstring>> registry_executables;
  };

  struct model_preparation_artifact_t {
    std::filesystem::path path;
    std::string sha256;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t row_bytes = 0;
  };

  struct model_preparation_converter_snapshot_t {
    std::optional<models::host_sbs_gpu_completion_receipt::receipt_t> receipt;
    bool last_admission_gpu_undecided = false;
    std::uint32_t last_admission_optional_work = 0u;
    std::uint64_t conversions = 0;
    std::uint64_t source_uploads = 0;
    std::int64_t source_stamp_ticks = 0;
    std::int64_t content_stamp_ticks = 0;
    std::int64_t rendered_stamp_ticks = 0;
    int mode = video::SBS_OFF;
    models::engine_build_status model_status = models::engine_build_status::unknown;
    models::host_sbs_renderer_e renderer = models::host_sbs_renderer_e::awaiting_v2;
    bool pipeline_ready = false;
    bool build_pending = false;
    bool estimator_present = false;
    bool estimator_valid = false;
    bool estimator_terminal = false;
    bool subtitle_warp_ready = false;
    bool text_game_warp_ready = false;
    bool v2_live_warp_seen = false;
    bool latest_lineage_authenticated = false;
    bool latest_lineage_known_depth = false;
    bool latest_lineage_receipt_present = false;
    bool latest_lineage_receipt_depth_cache_authorized = false;
    bool latest_lineage_receipt_matches_publication = false;
    std::uint64_t latest_lineage_publication_frame_id = 0;
    std::uint64_t sbs_frame_sequence = 0;
    std::uint64_t flat_outputs = 0;
    std::uint64_t warped_outputs = 0;
    std::uint64_t repeat_outputs = 0;
    std::uint64_t setup_watch_factory_calls = 0;
    std::uint64_t setup_watch_tokens_alive = 0;
    RECT setup_watch_rect {};
    bool setup_watch_present = false;
    bool receiver_present = false;
    bool production_observer_path = false;
    bool registry_present = false;
    bool registry_ready = false;
    bool registry_ready_observed = false;
    std::uint64_t registry_factory_calls = 0;
    bool registry_consumption_eligible = false;
    bool pipeline_enabled = false;
    bool depth_completion_pending = false;
    bool authority_reprocess_pending = false;
    bool subtitle_reprocess_pending = false;
    bool dump_pending = false;
    bool gamma_pending = false;
    bool gamma_white_query_due = false;
    bool last_draw_present = false;
    bool last_draw_flat = false;
    bool last_draw_repeat = false;
    bool last_draw_publication_current = false;
    bool last_draw_ema_motion_mask_present = false;
    std::uint64_t last_draw_model_frame = 0;
    std::uint64_t last_draw_observation_us = 0;
    std::uint32_t last_draw_expected_work = 0;
    std::uint32_t last_draw_host_subtitle_outcome = 0;
    std::int64_t last_draw_source_stamp_ticks = 0;
    std::int64_t last_draw_content_stamp_ticks = 0;
    std::int64_t last_draw_current_source_stamp_ticks = 0;
    std::int64_t last_draw_current_content_stamp_ticks = 0;
  };

  /** Actual offscreen shared-image converter on one synchronous D3D owner.
   * No preparation or direct ensure_depth_estimator call occurs here. The fixture owns one
   * immutable upload/stamp and explicitly drives real conversion and its real poll/readiness
   * event. Synthetic probe context is scheduling evidence only, never native video eligibility.
   * Game mono uses the real mode with unsupported provider transport. Only the external setup
   * watch factory is replaced by an observed owned inert token during synchronous initialization.
   * Caller joins the preparation owner and, after destroying this facade, invokes the existing
   * video::drain_async_teardown_workers once at the end of its isolated process fixture.
   */
  class model_preparation_converter_t {
  public:
    explicit model_preparation_converter_t(model_preparation_converter_request_t request);
    ~model_preparation_converter_t();
    model_preparation_converter_t(const model_preparation_converter_t &) = delete;
    model_preparation_converter_t &operator=(const model_preparation_converter_t &) = delete;
    model_preparation_converter_t(model_preparation_converter_t &&) = delete;
    model_preparation_converter_t &operator=(model_preparation_converter_t &&) = delete;

    void set_mode(video::sbs_mode_e mode);
    model_preparation_converter_snapshot_t observe() const;
    model_preparation_converter_snapshot_t convert_retained();
    bool needs_conversion_poll() const;
    bool pipeline_ready() const;
    model_preparation_artifact_t snapshot_output(std::string_view label);
    void upload_dirty_source(std::span<const std::uint8_t> bgra, const RECT &damage);
    bool drawn_source_matches_upload() const;
    model_preparation_artifact_t source_artifact() const;
    model_preparation_memory_t memory() const;

  private:
    class impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace platf::dxgi::testing
#endif
