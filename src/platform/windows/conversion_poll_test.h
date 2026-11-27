#pragma once

#ifdef SUNSHINE_TESTS
  #include "src/video.h"
  #include "src/video_depth_estimator.h"

  #include <cstdint>
  #include <functional>
  #include <future>
  #include <memory>
  #include <optional>
  #include <string>
  #include <vector>

namespace platf::dxgi::testing {
  enum class conversion_poll_build_e {
    absent,
    pending,
    ready,
  };

  // Only existing scalar owners are configured. No estimator/device/context is created.
  struct conversion_poll_inputs_t {
    video::sbs_mode_e mode = video::SBS_AI;
    models::host_sbs_renderer_e renderer = models::host_sbs_renderer_e::awaiting_v2;
    conversion_poll_build_e build = conversion_poll_build_e::absent;
    bool registry_ready_observed = false;
    bool depth_completion_pending = false;
    bool depth_authority_reprocess_pending = false;
    bool subtitle_context_reprocess_pending = false;
    bool dump_button_requested = false;
    bool stream_gamma_conversion_pending = false;
    // Disengaged selects Windows default. Engaged selects gamma2.2 with due/future white query.
    std::optional<bool> white_query_due;
  };

  struct conversion_poll_snapshot_t {
    bool needs_conversion = false;
    bool registry_present = false;
    bool registry_ready = false;
    bool registry_ready_observed = false;
    bool build_valid = false;
    bool build_ready = false;
    bool estimator_present = false;
    bool device_present = false;
    bool context_present = false;
    std::uint32_t model_status_reads = 0;
  };

  /** Calls the actual d3d_base_encode_device::needs_conversion_poll on an uninitialized owner.
   * No init/convert/observer/discovery or graphics entrypoint is called. The supplied registry
   * future belongs to the real registry_t. The status callback is scoped to this synchronous CPU
   * sample; ordinary/GPU converters retain the real process getter. Diagnostic file triggers
   * must be disabled before construction. Empty owned build results are consumed before teardown.
   */
  class conversion_poll_probe_t {
  public:
    explicit conversion_poll_probe_t(std::optional<std::shared_future<std::vector<std::wstring>>> registry_future);
    ~conversion_poll_probe_t();
    conversion_poll_probe_t(const conversion_poll_probe_t &) = delete;
    conversion_poll_probe_t &operator=(const conversion_poll_probe_t &) = delete;

    conversion_poll_snapshot_t observe(
      const conversion_poll_inputs_t &inputs,
      const std::function<models::engine_build_status()> &read_model_status
    );

  private:
    class impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace platf::dxgi::testing
#endif
