/**
 * @file src/platform/windows/ar_glasses_session.h
 * @brief One local presentation session: an owned SudoVDA source bound to one physical output, its
 * display set-up, presenter thread, pause, resume and teardown, and the target contracts the
 * controller uses to decide whether a new display generation may keep the session.
 */
#pragma once

#include "ar_glasses.h"
#include "ar_glasses_devices.h"
#include "ar_glasses_topology_journal.h"
#include "ar_glasses_virtual_source.h"
#include "display.h"
#include "monitor_desktop_handoff.h"
#include "src/gpu_workload_arbiter.h"
#include "src/thread_safe.h"
#include "src/video.h"
#include "virtual_display.h"
#include "virtual_display_session.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <windows.h>

namespace ar_glasses {
  namespace internal {
    bool same_physical_output(
      const std::optional<target_state_t> &left,
      const std::optional<target_state_t> &right
    );
    bool same_presentation_contract(
      const std::optional<target_state_t> &left,
      const std::optional<target_state_t> &right
    );
    bool can_re_isolate_local_session(
      const std::optional<target_state_t> &left,
      const std::optional<target_state_t> &right,
      bool transition_presenter_paused
    );
    bool monitor_source_only_change(const target_state_t &before, const target_state_t &after);

    enum class monitor_source_switch_progress_e {
      waiting,
      presented,
      failed,
    };

    using monitor_source_switch_t = platf::dxgi::local_monitor_source_switch_t;

    const char *mode_name(presentation_mode_e mode);

    class local_session_t {
    public:
      local_session_t(gpu_workload::lease_t live_gpu_lease, std::function<bool()> session_requested);
      void initialize(const target_state_t &target, std::stop_token controller_stop_token);
      void pause_presenter(const std::optional<target_state_t> &target);
      bool same_retained_device(const target_state_t &target) const;
      bool suspend_for_resume();
      std::optional<target_state_t> resume_retained(const target_state_t &target, std::stop_token stop_token);
      void refresh_monitor_windows();
      void refresh_pointer_isolation();
      void release_pointer_isolation();
      bool virtual_display_authoritatively_absent();
      bool has_monitor_source_switch() const;
      monitor_source_switch_progress_e advance_monitor_source_switch(
        const target_state_t &expected,
        std::uint64_t display_epoch,
        std::stop_token controller_stop_token
      );
      std::optional<target_state_t> reconfigure_target(
        const target_state_t &target,
        std::stop_token controller_stop_token
      );
      bool prepare_same_output_teardown(const target_state_t &target);
      bool rebase_teardown_layout(const target_state_t &target);
      ~local_session_t();
      bool valid() const;
      bool can_reconfigure(
        const std::optional<target_state_t> &before,
        const std::optional<target_state_t> &after
      ) const;
      bool running() const;
      bool failed() const;
      bool stable() const;
      bool has_presented() const;
      std::optional<target_state_t> re_isolate_target(const target_state_t &expected_target);

    private:
#ifdef SUNSHINE_TESTS
      friend detail::local_pause_order_result_t detail::local_pause_order_for_test();
#endif
      bool activate_monitor_desktop();
      bool configure_monitor_source_hdr(
        int requested_millihz,
        bool expected_hdr,
        bool reset_before_enable,
        const std::function<bool()> &cancelled
      );
      void keep_monitor_refresh(const target_state_t &selected, const std::function<bool()> &cancelled);
      bool establish_native_source_mode(int requested_millihz, const std::function<bool()> &cancelled);
      bool configure_source_refresh(int requested_millihz, const std::function<bool()> &cancelled);
      bool final_presentation_contract_matches(const target_state_t &expected, const target_state_t &actual) const;
      bool update_display_topology(const std::function<bool()> &update);
      std::optional<target_state_t> isolate_target();
      void retire_exclusive_display();
      void cancel_monitor_source_switch();
      void stop_presenter();
      std::optional<resolved_virtual_display_t> refresh_virtual_display_reference(bool require_geometry = true);
      VDISPLAY::creation_result_t virtual_display_binding() const;
      bool start_presenter(const target_state_t &presentation_target);

      VDISPLAY::session_t virtual_display_;
      // Hold the live reservation across presenter restarts. Reconnect retention releases it.
      gpu_workload::lease_t live_gpu_lease_;
      std::shared_ptr<pre_removal_topology_cleanup_t> resume_topology_cleanup_;
      bool exclusive_;
      bool monitor_surface_ = false;
      // Held while this is a shutter monitor session on a Windows desktop refresh rate.
      std::optional<VDISPLAY::desktop_refresh_rate_keeper_t> desktop_refresh_rate_keeper_;
      std::shared_ptr<platf::monitor_desktop_handoff_t> desktop_handoff_;
      int source_width_ = source_width;
      int source_height_ = source_height;
      std::function<bool()> session_requested_;
      RECT original_target_rect_ {};
      LUID target_adapter_id_ {};
      std::wstring target_device_path_;
      bool virtual_hdr_active_ = false;
      bool ready_ = false;
      mutable std::mutex virtual_display_mutex_;
      std::optional<std::chrono::steady_clock::time_point> virtual_display_absence_started_;
      unsigned virtual_display_absence_observations_ = 0;
      std::shared_ptr<platf::dxgi::local_presenter_config_t::target_t> live_target_;
      // A 2D/3D mode switch keeps this source. Keep any working capture-backend fallback too.
      std::shared_ptr<::video::capture_backend_failover_t> capture_failover_ {
        std::make_shared<::video::capture_backend_failover_t>()
      };
      std::shared_ptr<std::atomic<std::uint64_t>> presented_frames_;
      std::shared_ptr<platf::dxgi::local_presenter_cursor_clip_t> cursor_clip_;
      std::shared_ptr<monitor_source_switch_t> monitor_source_switch_;
      std::shared_ptr<safe::event_t<bool>> presenter_quiesced_;
      std::jthread presenter_;
      std::atomic<bool> running_ {false};
      std::atomic<bool> failed_ {false};
    };
  }  // namespace internal
}  // namespace ar_glasses
