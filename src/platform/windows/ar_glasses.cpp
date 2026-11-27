// Boost.Asio in stream.h requires Winsock 2 before the Windows display declarations.
// clang-format off
#include <winsock2.h>
// clang-format on

#include "ar_glasses.h"

#include "ar_glasses_arbiter.h"
#include "ar_glasses_config_store.h"
#include "ar_glasses_devices.h"
#include "ar_glasses_requests.h"
#include "ar_glasses_security.h"
#include "ar_glasses_session.h"
#include "ar_glasses_topology_journal.h"
#include "misc.h"
#include "open3d_emitter.h"
#include "rayneo_wear_monitor.h"
#include "shutter_presentation_clock.h"
#include "src/config.h"
#include "src/gpu_workload_arbiter.h"
#include "src/logging.h"
#include "src/process.h"
#include "src/session_resume_lifecycle.h"
#include "src/stream.h"
#include "src/utility.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ar_glasses {
  using namespace std::literals;
  using namespace internal;

  namespace internal {
    std::atomic<bool> physical_monitor_active {false};
    std::atomic<bool> physical_monitor_requested {false};
    // Local AR glasses present only after an explicit Start 3D, like every other local surface.
    std::atomic<bool> ar_session_requested {false};

    // The 3D source of each local surface: Host 3D converts the desktop, Game 3D takes the game's
    // own stereo. It starts from the saved setting and is switched live from the web UI, so it
    // is not read from the configuration after start-up. -1 means not switched yet.
    std::atomic<int> ar_source_game {-1};
    std::atomic<int> monitor_source_game {-1};

    bool ar_uses_game_source() {
      const int switched = ar_source_game.load();
      return switched < 0 ? config::video.sbs.reshade : switched != 0;
    }

    bool monitor_uses_game_source() {
      const int switched = monitor_source_game.load();
      return switched < 0 ? config::video.local_monitor.mode == "game" : switched != 0;
    }

    bool ar_session_allowed() {
      return config::video.local_surface == config::video_t::local_surface_e::ar_glasses && ar_session_requested.load();
    }
    std::mutex physical_monitor_error_mutex;
    std::string physical_monitor_error;
  }  // namespace internal

  namespace {
    constexpr auto topology_poll_interval = 250ms;
    constexpr auto topology_debounce = 750ms;
    constexpr auto incompatible_transition_grace = 5s;
    constexpr auto failed_session_retry = 2s;
    constexpr auto maximum_failed_session_retry = 30s;

    class controller_t final: public platf::deinit_t {
    public:
      controller_t() {
        // The thread must start only after every state member below has completed construction.
        // Starting a jthread from the member initializer list lets run() observe later members
        // before their lifetime begins.
        worker_ = std::jthread([this](std::stop_token stop_token) {
          run(stop_token);
        });
        if (config::video.local_surface != config::video_t::local_surface_e::off) {
          monitor_hotkeys_ = std::jthread([](std::stop_token stop_token) {
            const bool desktop_synced = platf::syncThreadDesktop();
            if (!desktop_synced) {
              BOOST_LOG(warning) << "Could not register monitor shortcuts on the interactive desktop; use the web UI or tray."sv;
            }
            constexpr int toggle_id = 0x5333;
            constexpr int stop_id = 0x5330;
            const UINT modifiers = MOD_CONTROL | MOD_ALT | MOD_SHIFT | MOD_NOREPEAT;
            const bool toggle_registered = desktop_synced && RegisterHotKey(nullptr, toggle_id, modifiers, '3');
            const bool stop_registered = desktop_synced && RegisterHotKey(nullptr, stop_id, modifiers, '0');
            auto cleanup = util::fail_guard([&]() {
              if (toggle_registered) UnregisterHotKey(nullptr, toggle_id);
              if (stop_registered) UnregisterHotKey(nullptr, stop_id);
            });
            if (desktop_synced && (!toggle_registered || !stop_registered)) {
              BOOST_LOG(warning) << "A local-monitor shortcut is already reserved. Use the web UI or tray to stop 3D."sv;
            }
            // The presenter always works on the application desktop. This thread's own desktop is
            // whichever one had input when the host started, which is the sign-in desktop for a
            // host started before sign-in, so it must not stand in for the presenter's.
            constexpr std::wstring_view presentation_desktop = L"Default";
            auto next_desktop_check = std::chrono::steady_clock::now();
            while (!stop_token.stop_requested()) {
              if (monitor_requested() && std::chrono::steady_clock::now() >= next_desktop_check) {
                next_desktop_check = std::chrono::steady_clock::now() + 100ms;
                const auto rect = monitor_watchdog_rect(platf::from_utf8(config::video.local_monitor.display_id));
                if (const auto reason = query_monitor_desktop_block_reason(presentation_desktop, rect); !reason.empty()) {
                  // A private primary source must not hide a sign-in/UAC screen on the secure
                  // desktop, or a lock-screen overlay which Windows can place on Default.
                  request_monitor_session(false);
                  {
                    std::lock_guard lock(physical_monitor_error_mutex);
                    physical_monitor_error = reason;
                  }
                }
              }
              MSG message {};
              while (PeekMessageW(&message, nullptr, WM_HOTKEY, WM_HOTKEY, PM_REMOVE)) {
                if (message.wParam == stop_id) {
                  request_local_session(false);
                } else if (message.wParam == toggle_id) {
                  request_local_session(!local_session_requested());
                }
              }
              std::this_thread::sleep_for(25ms);
            }
          });
        }
      }

      ~controller_t() override {
        monitor_hotkeys_.request_stop();
        if (monitor_hotkeys_.joinable()) {
          monitor_hotkeys_.join();
        }
        worker_.request_stop();
        if (worker_.joinable()) {
          worker_.join();
        }
      }

    private:
      void reset_failure_backoff() {
        retry_after_ = {};
        presenter_retry_after_ = {};
        failure_retry_delay_ = failed_session_retry;
      }

      void arm_presenter_retry() {
        presenter_retry_after_ = std::chrono::steady_clock::now() + failure_retry_delay_;
        failure_retry_delay_ = std::min(failure_retry_delay_ * 2, maximum_failed_session_retry);
      }

      void schedule_failure_retry() {
        BOOST_LOG(warning) << "Local AR session retry deferred for "sv
                           << std::chrono::duration_cast<std::chrono::seconds>(failure_retry_delay_).count()
                           << " seconds."sv;
        retry_after_ = std::chrono::steady_clock::now() + failure_retry_delay_;
        failure_retry_delay_ = std::min(failure_retry_delay_ * 2, maximum_failed_session_retry);
      }

      void stop_session() {
        resume_state_.clear();
        suspended_topology_complete_ = false;
        transition_presenter_paused_ = false;
        incompatible_transition_started_.reset();
        std::unique_ptr<local_session_t> retiring_session;
        {
          std::lock_guard lock(ownership_mutex);
          if (local_session_construction_stop) {
            local_session_construction_stop->request_stop();
          }
          retiring_session = std::move(session_);
        }

        if (!retiring_session) {
          release_local_virtual_display_claim_impl();
          return;
        }

        // Presenter shutdown and SudoVDA topology removal can take seconds. Keep ownership marked
        // local until teardown is complete, but never hold ownership_mutex across those waits so a
        // remote launch can publish its reservation and wait on the condition variable.
        retiring_session.reset();
        release_local_virtual_display_claim_impl();
      }

      void suspend_for_remote_if_needed() {
        bool should_stop = false;
        {
          std::lock_guard lock(ownership_mutex);
          if (!remote_blocks_local_locked(std::chrono::steady_clock::now())) {
            return;
          }
          if (local_session_construction_stop) {
            local_session_construction_stop->request_stop();
          }
          should_stop = session_ != nullptr;
          if (!deferred_for_remote_) {
            retry_after_ = std::chrono::steady_clock::now() + failed_session_retry;
          }
          deferred_for_remote_ = true;
        }

        if (should_stop) {
          BOOST_LOG(info) << "Remote virtual-display session requested ownership; stopping local AR presentation."sv;
          session_request_.consume();
          stop_session();
        }
      }

      void progress_inactive_retirement() {
        const auto now = std::chrono::steady_clock::now();
        if (session_ || now < retirement_retry_after_) {
          return;
        }
        {
          std::lock_guard lock(ownership_mutex);
          // Remote handoff already progresses this same barrier. Do not acquire a session or a
          // GPU lease just to complete cleanup while wear state keeps local presentation off.
          if (local_session_present || !retired_local_virtual_display || remote_blocks_local_locked(now)) {
            return;
          }
        }
        // Leave enough time for three absent observations and the final topology-settle check.
        // Failed work remains owned by the barrier and is retried on the next bounded interval.
        const bool completed = wait_for_local_virtual_display_retirement_impl(500ms);
        retirement_retry_after_ = std::chrono::steady_clock::now() + failed_session_retry;
        if (completed) {
          BOOST_LOG(info) << "Completed pending local AR display cleanup while presentation is inactive."sv;
        }
      }

      void monitor_target(const std::optional<target_state_t> &target) {
        const auto device_path = target ? target->device_path : std::wstring {};
        const bool previous_monitor_surface = monitor_surface_selected_;
        monitor_surface_selected_ = target && is_monitor_mode(target->mode);
        if (equal_device_paths(device_path, wear_device_path_) && previous_monitor_surface == monitor_surface_selected_) {
          return;
        }
        // The physical device owns wear authority. Keep this subscription alive when its virtual
        // desktop exits off-head or yields to a remote client, and across 2D/SBS mode changes.
        wear_monitor_.reset();
        wear_device_path_ = device_path;
        activation_ = {};
        logged_activation_.reset();
        session_request_.reset();
        if (target && !monitor_surface_selected_) {
          wear_monitor_ = rayneo::create_wear_monitor(target->device_id);
        }
      }

      bool wear_allows_session() {
        if (monitor_surface_selected_) {
          const bool requested = physical_monitor_requested.load();
          session_request_.observe(requested);
          return requested;
        }
        if (wear_device_path_.empty()) {
          return false;
        }
        if (!ar_session_allowed()) {
          session_request_.observe(false);
          return false;
        }
        const auto sensor = wear_monitor_ ? std::make_optional(wear_monitor_->snapshot()) : std::nullopt;
        // Read remembered authority after the live snapshot: a confirmed event received while
        // Windows blocked display setup must survive a subsequent unknown/unavailable report.
        const auto remembered = wear_monitor_ ? wear_monitor_->last_confirmed_state() : rayneo::wear_state_e::unknown;
        const auto activation = activation_.observe(
          sensor,
          std::chrono::steady_clock::now(),
          remembered
        );
        if (activation != logged_activation_) {
          logged_activation_ = activation;
          switch (activation) {
            case detail::local_session_activation_e::waiting_for_sensor:
              BOOST_LOG(info) << "Local AR is waiting for the glasses' initial wear state before creating a virtual desktop."sv;
              break;
            case detail::local_session_activation_e::connection:
              BOOST_LOG(info) << "No confirmed wear state is available; local AR virtual-desktop activation follows the glasses' display connection."sv;
              break;
            case detail::local_session_activation_e::worn:
              BOOST_LOG(info) << "AR glasses report worn; local AR virtual-desktop activation is on."sv;
              break;
            case detail::local_session_activation_e::off_head:
              BOOST_LOG(info) << "AR glasses report off-head; local AR virtual-desktop activation is off."sv;
              break;
          }
        }
        const bool requested = detail::local_session_requested(activation);
        session_request_.observe(requested);
        return requested;
      }

      bool suspend_inactive_session(const std::optional<target_state_t> &observed, bool topology_known) {
        const bool requested = wear_allows_session();
        if (!session_ || resume_state_.retained() || (requested && (!topology_known || observed))) {
          return false;
        }
        if (requested && topology_known && !observed && wear_monitor_ && wear_monitor_->snapshot().availability == rayneo::wear_availability_e::available && (!incompatible_transition_started_ || std::chrono::steady_clock::now() - *incompatible_transition_started_ < incompatible_transition_grace)) {
          // A still-valid RayNeo USB subscription distinguishes a DP-only hardware mode gap.
          // USB removal/power callbacks invalidate snapshot availability immediately; other
          // displays without this evidence use authoritative Windows display absence directly.
          return false;
        }
        transition_presenter_paused_ = false;
        BOOST_LOG(info) << "Local AR presentation is inactive; restoring the desktop and retaining its session for reconnect."sv;
        resume_state_.disconnect(std::chrono::steady_clock::now(), config::stream.session_resume_grace);
        suspended_topology_complete_ = session_->suspend_for_resume();
        reset_failure_backoff();
        retry_after_ = std::chrono::steady_clock::now() + (suspended_topology_complete_ ? 0s : failed_session_retry);
        return true;
      }

      bool replacement_requests_session(const target_state_t &target) {
        if (target.mode == presentation_mode_e::unsupported || target.is_cloned || (!target_exclusive(target) && target.is_primary)) {
          return false;
        }
        if (is_monitor_mode(target.mode)) {
          return physical_monitor_requested.load();
        }
        if (!ar_session_allowed()) {
          return false;
        }
        if (!replacement_target_ || !equal_device_paths(replacement_target_->device_path, target.device_path)) {
          replacement_target_ = target;
          replacement_wear_monitor_ = rayneo::create_wear_monitor(target.device_id);
          replacement_activation_ = {};
        }
        const auto sensor = replacement_wear_monitor_ ? std::make_optional(replacement_wear_monitor_->snapshot()) : std::nullopt;
        const auto remembered = replacement_wear_monitor_ ? replacement_wear_monitor_->last_confirmed_state() : rayneo::wear_state_e::unknown;
        return detail::local_session_requested(replacement_activation_.observe(sensor, std::chrono::steady_clock::now(), remembered));
      }

      bool progress_retained_session(
        const std::optional<target_state_t> &observed,
        bool topology_known,
        std::chrono::steady_clock::time_point stable_since,
        std::stop_token stop_token
      ) {
        if (!session_ || !resume_state_.retained()) {
          return false;
        }
        const auto now = std::chrono::steady_clock::now();
        if (resume_state_.expired(now)) {
          BOOST_LOG(info) << "Local AR session resume grace expired; retiring its retained virtual desktop."sv;
          stop_session();
          applied_.reset();
          return true;
        }
        if (!suspended_topology_complete_ && now >= retry_after_) {
          suspended_topology_complete_ = session_->suspend_for_resume();
          retry_after_ = std::chrono::steady_clock::now() + (suspended_topology_complete_ ? 0s : failed_session_retry);
        }
        if (topology_known && observed && !session_->same_retained_device(*observed)) {
          if (!replacement_requests_session(*observed)) {
            return true;
          }
          BOOST_LOG(info) << "A different local AR display requested presentation; retiring the previous retained session."sv;
          stop_session();
          applied_.reset();
          monitor_target(observed);
          wear_monitor_ = std::move(replacement_wear_monitor_);
          activation_ = replacement_activation_;
          replacement_target_.reset();
          return true;
        }
        if (topology_known && !observed && now - stable_since >= incompatible_transition_grace) {
          // Display absence restores the desktop immediately. Only sensor-identity reset keeps
          // the hardware-switch grace, so a transient DP gap does not forget a confirmed state.
          monitor_target(std::nullopt);
        }
        if (!suspended_topology_complete_ || !topology_known || !observed || !wear_allows_session() || now - stable_since < topology_debounce || now < retry_after_) {
          return true;
        }
        std::stop_source resume_stop;
        std::stop_callback controller_stop_callback(stop_token, [&resume_stop]() {
          resume_stop.request_stop();
        });
        {
          std::lock_guard lock(ownership_mutex);
          if (remote_blocks_local_locked(std::chrono::steady_clock::now())) {
            return true;
          }
          local_session_construction_stop = resume_stop;
        }
        const auto resumed = session_->resume_retained(*observed, resume_stop.get_token());
        bool remote_requested = false;
        {
          std::lock_guard lock(ownership_mutex);
          local_session_construction_stop.reset();
          remote_requested = remote_blocks_local_locked(std::chrono::steady_clock::now());
        }
        if (remote_requested) {
          session_request_.consume();
          stop_session();
          return true;
        }
        if (!resumed) {
          suspended_topology_complete_ = session_->suspend_for_resume();
          retry_after_ = std::chrono::steady_clock::now() + failed_session_retry;
          return true;
        }
        resume_state_.activate();
        replacement_target_.reset();
        replacement_wear_monitor_.reset();
        suspended_topology_complete_ = false;
        session_request_.consume();
        applied_ = resumed;
        session_stability_confirmed_ = false;
        transition_presenter_paused_ = false;
        incompatible_transition_started_.reset();
        reset_failure_backoff();
        arm_presenter_retry();
        BOOST_LOG(info) << "Resumed the retained local AR virtual desktop and reapplied its display settings."sv;
        return true;
      }

      void start_session(const target_state_t &target, std::stop_token stop_token) {
        if (stop_token.stop_requested() || !wear_allows_session()) {
          return;
        }
        std::stop_source construction_stop;
        std::stop_callback controller_stop_callback(stop_token, [&construction_stop]() {
          construction_stop.request_stop();
        });
        auto live_gpu_lease =
          gpu_workload::try_acquire(gpu_workload::kind_e::live_stream);
        if (!live_gpu_lease) {
          BOOST_LOG(info)
            << "Local AR presentation is waiting for the active offline SBS GPU job."sv;
          schedule_failure_retry();
          return;
        }
        if (construction_stop.stop_requested() || !wear_allows_session()) {
          return;
        }
        // HTTP may hold the platform lock while waiting for this controller's teardown. The
        // shared admission helper is nonblocking and must run before any ownership claim.
        // Reserve GPU availability before replacing another session's reconnect state.
        const auto admission = stream::session::prepare_local_ar_session(session_request_.fresh());
        if (admission != stream::session::local_session_admission_e::ready) {
          if (admission == stream::session::local_session_admission_e::remote_busy) {
            session_request_.consume();
          }
          retry_after_ = std::chrono::steady_clock::now() + failed_session_retry;
          return;
        }
        const auto handoff = proc::proc.prepare_local_ar_handoff(construction_stop);
        if (handoff != proc::local_ar_handoff_e::ready) {
          if (!deferred_for_remote_) {
            BOOST_LOG(info) << (handoff == proc::local_ar_handoff_e::remote_busy ? "Local AR presentation is waiting for the active/connecting remote virtual-display session."sv : "Local AR presentation is waiting for inactive remote-display cleanup."sv);
          }
          deferred_for_remote_ = true;
          retry_after_ = std::chrono::steady_clock::now() + failed_session_retry;
          return;
        }
        // prepare_local_ar_handoff() atomically claims local ownership. Keep release exception-safe,
        // and declare the guard before the candidate so a partially constructed session retires its
        // display before remote creation can observe ownership as free.
        auto release_claim = util::fail_guard([]() {
          release_local_virtual_display_claim_impl();
        });
        if (construction_stop.stop_requested() || !wear_allows_session()) {
          return;
        }
        if (!recover_saved_topology(target.device_path) || construction_stop.stop_requested()) {
          BOOST_LOG(warning) << "Deferring local AR startup until this display's previous desktop topology can be restored."sv;
          schedule_failure_retry();
          return;
        }

        auto candidate = std::make_unique<local_session_t>(std::move(*live_gpu_lease), [this]() {
          return wear_allows_session();
        });
        candidate->initialize(target, construction_stop.get_token());
        const bool requested = wear_allows_session();
        bool rejected_for_remote = false;
        {
          std::lock_guard lock(ownership_mutex);
          rejected_for_remote = remote_blocks_local_locked(std::chrono::steady_clock::now());
          if (candidate->valid() && requested && !construction_stop.stop_requested() && !rejected_for_remote) {
            session_ = std::move(candidate);
            resume_state_.activate();
            replacement_target_.reset();
            replacement_wear_monitor_.reset();
            session_request_.consume();
            local_session_construction_stop.reset();
            local_session_present = true;
            ownership_changed.notify_all();
            if (deferred_for_remote_) {
              BOOST_LOG(info) << "Remote virtual-display ownership ended; starting deferred local AR presentation."sv;
            }
            deferred_for_remote_ = false;
            session_stability_confirmed_ = false;
            transition_presenter_paused_ = false;
            incompatible_transition_started_.reset();
            // A spawned thread is not yet a healthy presenter. Arm bounded backoff now and reset it
            // only after the controller observes a sustained run of presented frames.
            arm_presenter_retry();
            release_claim.disable();
            return;
          }
        }

        // Destruction removes any partially created virtual display. Keep local ownership true
        // until that removal and topology restoration finish, so the remote side cannot overlap.
        candidate.reset();
        if (rejected_for_remote) {
          deferred_for_remote_ = true;
        }
        if (!rejected_for_remote && !construction_stop.stop_requested() && wear_allows_session()) {
          if (is_monitor_mode(target.mode)) {
            std::lock_guard lock(physical_monitor_error_mutex);
            physical_monitor_error = "Could not prepare the monitor's 3D desktop. Check the display and emitter settings, then start 3D again.";
            physical_monitor_requested.store(false);
          }
          schedule_failure_retry();
        }
      }

      void apply(const std::optional<target_state_t> &target, std::stop_token stop_token) {
        stop_session();
        reset_failure_backoff();
        applied_ = target;
        monitor_target(target);
        if (!target) {
          deferred_for_remote_ = false;
          BOOST_LOG(info) << "Approved AR display disconnected; local presentation is off."sv;
          return;
        }

        const int width = target->rect.right - target->rect.left;
        const int height = target->rect.bottom - target->rect.top;
        if (target->mode == presentation_mode_e::unsupported) {
          BOOST_LOG(warning) << "AR display mode "sv << width << 'x' << height
                             << " is unsupported; expected 1920x1080 or 3840x1080."sv;
          return;
        }
        if ((!target_exclusive(*target) && target->is_primary) || target->is_cloned) {
          BOOST_LOG(warning) << "AR display cannot start local presentation while it is "sv
                             << (target->is_primary ? "the Windows primary display"sv : "part of a cloned display source"sv)
                             << "; waiting for a unique extended-desktop topology."sv;
          return;
        }

        BOOST_LOG(info) << "Approved AR display ["sv << target->friendly_name << "] detected at "sv
                        << width << 'x' << height << '@'
                        << (target->refresh_millihz / 1000.0) << "; "sv
                        << mode_name(target->mode) << " local presentation is available."sv;
        start_session(*target, stop_token);
      }

      void run(std::stop_token stop_token) {
        std::optional<target_state_t> pending;
        auto pending_since = std::chrono::steady_clock::now();
        const auto accept_reconfigured_target = [&](const target_state_t &target, std::chrono::steady_clock::time_point now) {
          pending = target;
          applied_ = pending;
          pending_since = now;
          session_stability_confirmed_ = false;
          transition_presenter_paused_ = false;
          incompatible_transition_started_.reset();
          arm_presenter_retry();
        };

        while (!stop_token.stop_requested()) {
          suspend_for_remote_if_needed();
          progress_inactive_retirement();
          physical_monitor_active.store(session_ && applied_ && is_monitor_mode(applied_->mode) &&
                                        session_->running() && session_->has_presented());

          const std::wstring_view preferred_device_path = wear_device_path_;
          bool topology_query_succeeded = false;
          std::vector<target_state_t> approved_targets;
          const auto display_epoch = platf::display_change_epoch();
          auto observed = find_target({}, preferred_device_path, &topology_query_succeeded, &approved_targets);
          if (topology_query_succeeded) {
            if (!wear_device_path_.empty()) {
              const bool connected = std::ranges::any_of(approved_targets, [&](const auto &target) {
                return equal_device_paths(target.device_path, wear_device_path_);
              });
              const bool transport_available = wear_monitor_ &&
                                               wear_monitor_->snapshot().availability == rayneo::wear_availability_e::available;
              session_request_.observe_connection(connected, transport_available);
            }
            if (session_ && resume_state_.retained()) {
              const auto arrived = std::ranges::find_if(approved_targets, [&](const auto &target) {
                return !session_->same_retained_device(target) &&
                       std::ranges::none_of(connected_device_paths_, [&](const auto &connected) {
                         return equal_device_paths(connected, target.device_path);
                       });
              });
              if (arrived != approved_targets.end()) {
                (void) replacement_requests_session(*arrived);
              }
              if (replacement_target_) {
                const auto candidate = std::ranges::find_if(approved_targets, [&](const auto &target) {
                  return equal_device_paths(target.device_path, replacement_target_->device_path);
                });
                if (candidate == approved_targets.end()) {
                  replacement_target_.reset();
                  replacement_wear_monitor_.reset();
                } else if (replacement_requests_session(*candidate)) {
                  observed = *candidate;
                }
              }
            }
            connected_device_paths_.clear();
            for (const auto &target : approved_targets) {
              connected_device_paths_.push_back(target.device_path);
            }
          }
          if (!topology_query_succeeded) {
            // A failed query is not a disconnect. Retain the previous observation and live session;
            // Windows commonly reports an undersized snapshot while applying display topology.
            // Unknown time cannot count toward either stable debounce or destructive disconnect
            // grace: require a fresh continuous sequence of authoritative observations afterward.
            const auto indeterminate_at = std::chrono::steady_clock::now();
            // Reset unconditionally: a paused transition may already have observed applied_ again,
            // and unknown time must not count toward its return-to-stable resume debounce either.
            pending_since = indeterminate_at;
            if (incompatible_transition_started_) {
              incompatible_transition_started_ = indeterminate_at;
            }
            if (session_ && session_->has_monitor_source_switch()) {
              session_->pause_presenter(std::nullopt);
              transition_presenter_paused_ = true;
            }
            if (suspend_inactive_session(applied_, false)) {
              pending.reset();
              pending_since = std::chrono::steady_clock::now();
            }
            if (progress_retained_session(applied_, false, pending_since, stop_token)) {
              std::this_thread::sleep_for(topology_poll_interval);
              continue;
            }
            if (transition_presenter_paused_ && session_) {
              session_->refresh_pointer_isolation();
            } else if (session_) {
              session_->refresh_monitor_windows();
            }
            std::this_thread::sleep_for(topology_poll_interval);
            continue;
          }
          const auto now = std::chrono::steady_clock::now();
          if (wear_device_path_.empty() && observed) {
            monitor_target(observed);
          }
          const bool source_only = session_ && !resume_state_.retained() && !transition_presenter_paused_ &&
                                   session_->running() && session_->has_presented() && applied_ && observed &&
                                   monitor_source_only_change(*applied_, *observed) &&
                                   display_epoch == platf::display_change_epoch();
          const bool source_switch_revoked = !source_only && session_ && session_->has_monitor_source_switch();
          if (observed != pending || source_switch_revoked) {
            pending = observed;
            pending_since = now;
            // Stop using old-size textures immediately, but keep SudoVDA attached during the
            // debounce interval. A physical 2D/SBS switch is allowed to pass through transient
            // modes; removing the source here would make Windows migrate all of its windows.
            // A reversal can also return to the applied mode with a replacement still in flight.
            if (source_switch_revoked || (!source_only && !resume_state_.retained() && observed != applied_ && !same_presentation_contract(observed, applied_))) {
              if (session_) {
                session_->pause_presenter(observed);
                transition_presenter_paused_ = true;
              }
            }
            // Every distinct incompatible generation gets its own grace period. Churn through
            // several temporary modes must not accumulate enough time to make the newest 750 ms
            // observation look like a stable disconnect.
            if (session_ && !resume_state_.retained() && observed != applied_ && !session_->can_reconfigure(applied_, observed)) {
              incompatible_transition_started_ = now;
            } else {
              incompatible_transition_started_.reset();
            }
          }
          if (suspend_inactive_session(observed, true)) {
            // Restoring the desktop changes physical geometry. Resume only after a fresh stable
            // observation; pending still describes the previously active presentation layout.
            pending.reset();
            pending_since = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(topology_poll_interval);
            continue;
          }
          if (progress_retained_session(observed, true, pending_since, stop_token)) {
            std::this_thread::sleep_for(topology_poll_interval);
            continue;
          }
          if (transition_presenter_paused_ && session_) {
            session_->refresh_pointer_isolation();
          } else if (session_) {
            session_->refresh_monitor_windows();
          }
          if (session_ && !wear_allows_session()) {
            // Extended-mode recovery may still need a durable sink-mode rebase. Never restart
            // presentation or mutate to another mode while that off-head teardown is pending.
            std::this_thread::sleep_for(topology_poll_interval);
            continue;
          }

          if (source_only && now >= retry_after_) {
            switch (session_->advance_monitor_source_switch(*observed, display_epoch, stop_token)) {
              case monitor_source_switch_progress_e::waiting:
                break;
              case monitor_source_switch_progress_e::presented:
                applied_ = pending = observed;
                pending_since = now;
                incompatible_transition_started_.reset();
                BOOST_LOG(info) << "Local monitor presentation switched to "sv << mode_name(observed->mode)
                                << " at a complete stereo pair."sv;
                break;
              case monitor_source_switch_progress_e::failed:
                // A stalled conversion or lost proof invalidates this handoff. Re-observe before
                // running the ordinary repair path on the retained desktop.
                session_->pause_presenter(std::nullopt);
                transition_presenter_paused_ = true;
                pending_since = std::chrono::steady_clock::now();
                break;
            }
            std::this_thread::sleep_for(topology_poll_interval);
            continue;
          }

          if (!pending && !wear_device_path_.empty()) {
            // Keep the sensor and its last authority across brief DP mode-switch gaps, including
            // when off-head teardown has already removed the virtual desktop.
            if (now - pending_since >= incompatible_transition_grace) {
              apply(std::nullopt, stop_token);
            }
            std::this_thread::sleep_for(topology_poll_interval);
            continue;
          }

          if (pending != applied_ && now - pending_since >= topology_debounce) {
            if (can_re_isolate_local_session(pending, applied_, transition_presenter_paused_)) {
              if (session_) {
                if (now < retry_after_) {
                  std::this_thread::sleep_for(topology_poll_interval);
                  continue;
                }
                const auto isolated = session_->re_isolate_target(*pending);
                if (!isolated) {
                  session_->pause_presenter(pending);
                  transition_presenter_paused_ = true;
                  if (session_->virtual_display_authoritatively_absent()) {
                    BOOST_LOG(warning) << "The local AR virtual desktop is authoritatively absent; rebuilding the session."sv;
                    apply(pending, stop_token);
                  } else {
                    BOOST_LOG(warning) << "AR display position drifted and could not yet be re-isolated; preserving its virtual desktop for retry."sv;
                    schedule_failure_retry();
                    pending_since = now;
                  }
                } else {
                  pending = isolated;
                  pending_since = now;
                  applied_ = pending;
                  BOOST_LOG(info) << "AR display position changed; restored physical-output isolation without recreating its virtual desktop."sv;
                }
              } else {
                applied_ = pending;
              }
            } else if (session_ && session_->can_reconfigure(applied_, pending)) {
              if (now < retry_after_) {
                std::this_thread::sleep_for(topology_poll_interval);
                continue;
              }
              const auto reconfigured = session_->reconfigure_target(
                *pending,
                stop_token
              );
              if (!reconfigured) {
                if (session_->virtual_display_authoritatively_absent()) {
                  BOOST_LOG(warning) << "The local AR virtual desktop disappeared during its mode switch; rebuilding the session."sv;
                  apply(pending, stop_token);
                } else {
                  BOOST_LOG(warning) << "AR presentation mode change is waiting for a stable in-place reconfiguration; the virtual desktop remains attached."sv;
                  schedule_failure_retry();
                  pending_since = now;
                }
              } else {
                accept_reconfigured_target(*reconfigured, now);
              }
            } else if (!session_ || !incompatible_transition_started_ || now - *incompatible_transition_started_ >= incompatible_transition_grace) {
              if (session_ && pending && applied_ && same_physical_output(applied_, pending)) {
                if (now < retry_after_) {
                  std::this_thread::sleep_for(topology_poll_interval);
                  continue;
                }
                if (!session_->prepare_same_output_teardown(*pending)) {
                  schedule_failure_retry();
                  pending_since = now;
                  incompatible_transition_started_ = now;
                  std::this_thread::sleep_for(topology_poll_interval);
                  continue;
                }
              }
              apply(pending, stop_token);
            }
          } else if (pending == applied_ && transition_presenter_paused_ && session_ && pending && now - pending_since >= topology_debounce && now >= retry_after_) {
            // A transient display generation can settle back to the original contract after the
            // old presenter was deliberately paused. Restart resources on the still-attached
            // desktop instead of leaving a valid session silently idle.
            const auto reconfigured = session_->reconfigure_target(
              *pending,
              stop_token
            );
            if (!reconfigured) {
              if (session_->virtual_display_authoritatively_absent()) {
                apply(pending, stop_token);
              } else {
                schedule_failure_retry();
                pending_since = now;
              }
            } else {
              accept_reconfigured_target(*reconfigured, now);
            }
          } else if (pending == applied_ && session_ && !session_stability_confirmed_ && session_->stable()) {
            // Construction only proves that a presenter thread was spawned. Reset exponential
            // retry backoff after sustained scanout so permanent DXGI/swapchain failures cannot
            // recreate the whole topology every two seconds forever.
            session_stability_confirmed_ = true;
            reset_failure_backoff();
          } else if (pending == applied_ && session_ && !session_->running() && session_->failed() && now >= retry_after_ && now >= presenter_retry_after_) {
            transition_presenter_paused_ = true;
            session_->pause_presenter(pending);
            if (session_->virtual_display_authoritatively_absent()) {
              BOOST_LOG(warning) << "Local AR presentation lost its virtual desktop; scheduling a complete restart."sv;
              stop_session();
              schedule_failure_retry();
            } else if (pending) {
              BOOST_LOG(warning) << "Local AR presentation failed; restarting presenter resources while retaining its virtual desktop."sv;
              const auto reconfigured = session_->reconfigure_target(*pending, stop_token);
              if (reconfigured) {
                pending = reconfigured;
                applied_ = reconfigured;
                pending_since = now;
                session_stability_confirmed_ = false;
                transition_presenter_paused_ = false;
                arm_presenter_retry();
              } else {
                schedule_failure_retry();
              }
            }
          } else if (pending == applied_ && applied_ && !session_ && now - pending_since >= topology_debounce && applied_->mode != presentation_mode_e::unsupported && (target_exclusive(*applied_) || !applied_->is_primary) && !applied_->is_cloned && (now >= retry_after_ || session_request_.fresh())) {
            start_session(*applied_, stop_token);
          }

          std::this_thread::sleep_for(topology_poll_interval);
        }

        stop_session();
      }

      std::optional<target_state_t> applied_;
      bool monitor_surface_selected_ = false;
      std::unique_ptr<local_session_t> session_;
      session_lifecycle::resume_state_t resume_state_;
      bool suspended_topology_complete_ = false;
      detail::local_session_request_t session_request_;
      std::vector<std::wstring> connected_device_paths_;
      std::optional<target_state_t> replacement_target_;
      std::unique_ptr<rayneo::wear_monitor_t> replacement_wear_monitor_;
      detail::local_session_activation_t replacement_activation_;
      std::chrono::steady_clock::time_point retry_after_ {};
      std::chrono::steady_clock::time_point presenter_retry_after_ {};
      std::chrono::steady_clock::time_point retirement_retry_after_ {};
      std::chrono::seconds failure_retry_delay_ {failed_session_retry};
      bool deferred_for_remote_ = false;
      bool session_stability_confirmed_ = false;
      bool transition_presenter_paused_ = false;
      std::optional<std::chrono::steady_clock::time_point> incompatible_transition_started_;
      std::wstring wear_device_path_;
      std::unique_ptr<rayneo::wear_monitor_t> wear_monitor_;
      detail::local_session_activation_t activation_;
      std::optional<detail::local_session_activation_e> logged_activation_;
      std::jthread worker_;
      std::jthread monitor_hotkeys_;
    };
  }  // namespace

  bool monitor_active() {
    return physical_monitor_active.load();
  }

  bool monitor_requested() {
    return physical_monitor_requested.load();
  }

  std::string monitor_error() {
    std::lock_guard lock(physical_monitor_error_mutex);
    return physical_monitor_error;
  }

  bool request_monitor_session(bool start) {
    if (start) {
      const auto fail = [](std::string message) {
        std::lock_guard lock(physical_monitor_error_mutex);
        physical_monitor_error = std::move(message);
        return false;
      };
      constexpr std::wstring_view presentation_desktop = L"Default";
      if (const auto reason = query_monitor_desktop_block_reason(presentation_desktop); !reason.empty()) {
        request_monitor_session(false);
        return fail(std::string(reason));
      }
      if (physical_monitor_requested.load()) {
        return true;
      }
      const auto &settings = config::video.local_monitor;
      if (!settings.enabled) {
        return fail("Local monitor 3D is turned off, or its saved settings are invalid. Check Monitor settings, save, and restart Sunshine 3D before starting 3D.");
      }
      if (settings.display_id.empty() || settings.emitter_port.empty()) {
        return fail("Configure a physical monitor and emitter, save, and restart Sunshine 3D before starting 3D.");
      }
      const auto selected_port = platf::open3d::detail::normalize_port(settings.emitter_port);
      const auto ports = platf::open3d::serial_ports();
      if (!selected_port || std::none_of(ports.begin(), ports.end(), [&](const auto &port) {
            return platf::open3d::detail::normalize_port(port) == selected_port;
          })) {
        return fail("The selected emitter COM port is not connected. Connect the emitter, then start 3D again.");
      }
      const auto snapshot = discover_targets();
      if (!snapshot) {
        return fail("Windows display information is changing. Wait for the desktop to settle, then start 3D again.");
      }
      const auto &targets = snapshot->targets;
      const auto match = std::find_if(targets.begin(), targets.end(), [&](const auto &candidate) {
        return equal_device_paths(candidate.device_path, platf::from_utf8(settings.display_id));
      });
      const auto target = match == targets.end() ? std::nullopt :
                                                   std::make_optional(materialize_target(*snapshot, static_cast<std::size_t>(std::distance(targets.begin(), match))));
      if (!target || target->is_cloned || classify_monitor_mode(target->rect.right - target->rect.left, target->rect.bottom - target->rect.top, target->refresh_millihz, monitor_uses_game_source()) == presentation_mode_e::unsupported) {
        return fail("The selected monitor is not connected as an extended desktop in a supported mode. A monitor with a separate high-refresh mode is a different display to Windows in each mode; switch it to the mode it was selected in, then start 3D again.");
      }
      if (const auto reason = query_monitor_desktop_block_reason(presentation_desktop, target->rect); !reason.empty()) {
        return fail(std::string(reason));
      }
      const auto period = platf::open3d::eye_period_us(target->refresh_millihz, settings.eye_repeats);
      // The presenter's own split of the delay between host and emitter decides what fits: a
      // window may reach into the next eye interval when the host can hold its command back.
      const auto emitter_limits = platf::open3d::timing_limits(settings.ir_protocol, settings.signal_spacing_us);
      const auto command_plan = period && emitter_limits ?
                                  platf::dxgi::detail::plan_shutter_command(
                                    settings.frame_delay_us,
                                    settings.frame_duration_us,
                                    settings.signal_spacing_us + emitter_limits->min_period_tail_base_us,
                                    emitter_limits->min_delay_us,
                                    *period
                                  ) :
                                  std::nullopt;
      const platf::open3d::settings_t emitter_settings {
        .port = settings.emitter_port,
        .protocol = settings.ir_protocol,
        .frame_delay_us = command_plan ? command_plan->emitter_delay_us : settings.frame_delay_us,
        .frame_duration_us = settings.frame_duration_us,
        .signal_spacing_us = settings.signal_spacing_us,
        .flip_eyes = settings.flip_eyes,
        .target_frametime_us = period.value_or(-1),
        .averaging = settings.emitter_averaging,
      };
      if (!command_plan || !platf::open3d::detail::host_feedback_budget_us(emitter_settings)) {
        return fail("The shutter timing does not fit this monitor's refresh rate. Shorten the timing window or increase refreshes per eye in Monitor settings before starting 3D.");
      }
      {
        std::lock_guard lock(physical_monitor_error_mutex);
        physical_monitor_error.clear();
        physical_monitor_requested.store(true);
      }
    } else {
      const bool was_requested = physical_monitor_requested.exchange(false);
      physical_monitor_active.store(false);
      {
        // Returning to 2D acknowledges whatever stopped or refused the previous session. A
        // refused start that calls this first sets its own reason afterwards.
        std::lock_guard lock(physical_monitor_error_mutex);
        physical_monitor_error.clear();
      }
      if (was_requested) {
        std::lock_guard lock(ownership_mutex);
        if (local_session_construction_stop) {
          local_session_construction_stop->request_stop();
        }
      }
    }
    return true;
  }

  bool local_session_requested() {
    return config::video.local_surface == config::video_t::local_surface_e::shutter_monitor ? monitor_requested() : ar_session_allowed();
  }

  bool local_session_active() {
    if (config::video.local_surface == config::video_t::local_surface_e::shutter_monitor) {
      return monitor_active();
    }
    if (!ar_session_allowed()) {
      return false;
    }
    std::lock_guard lock(ownership_mutex);
    return local_session_present;
  }

  bool local_source_is_game() {
    switch (config::video.local_surface) {
      case config::video_t::local_surface_e::shutter_monitor:
        return monitor_uses_game_source();
      case config::video_t::local_surface_e::ar_glasses:
        return ar_uses_game_source();
      default:
        return false;
    }
  }

  bool monitor_source_is_game() {
    return monitor_uses_game_source();
  }

  bool ar_source_is_game() {
    return ar_uses_game_source();
  }

  bool set_local_source(bool game) {
    const auto surface = config::video.local_surface;
    if (surface == config::video_t::local_surface_e::off) {
      return false;
    }
    const bool monitor = surface == config::video_t::local_surface_e::shutter_monitor;
    // The controller handles the request on its next look at the displays. A verified monitor
    // keeps scanout and switches the source at a complete stereo pair.
    (monitor ? monitor_source_game : ar_source_game).store(game ? 1 : 0);
    // Remember it for the next start. The surface keeps its own saved setting.
    std::lock_guard lock(device_mutex);
    auto contents = read_config_file_strict();
    if (!contents) {
      BOOST_LOG(warning) << "Could not read the configuration file; the 3D source is switched but not saved."sv;
      return true;
    }
    const auto updated = monitor ? replace_config_value(std::move(*contents), "local_monitor_mode", game ? "game" : "host") :
                                   replace_config_value(std::move(*contents), "sbs_reshade", game ? "enabled" : "disabled");
    if (!write_config_atomically(updated)) {
      BOOST_LOG(warning) << "Could not save the 3D source to the configuration file."sv;
    }
    return true;
  }

  bool local_surface_available() {
    switch (config::video.local_surface) {
      case config::video_t::local_surface_e::shutter_monitor:
        return true;
      case config::video_t::local_surface_e::ar_glasses:
        {
          std::lock_guard lock(device_mutex);
          return std::any_of(known_devices.begin(), known_devices.end(), [](const auto &device) {
            return device.connected && device.decision == device_decision_e::approved;
          });
        }
      default:
        return false;
    }
  }

  bool request_local_session(bool start) {
    const auto fail = [](std::string message) {
      std::lock_guard lock(physical_monitor_error_mutex);
      physical_monitor_error = std::move(message);
      return false;
    };
    switch (config::video.local_surface) {
      case config::video_t::local_surface_e::shutter_monitor:
        return request_monitor_session(start);
      case config::video_t::local_surface_e::ar_glasses:
        {
          if (!start) {
            ar_session_requested.store(false);
            std::lock_guard lock(physical_monitor_error_mutex);
            physical_monitor_error.clear();
            return true;
          }
          if (!local_surface_available()) {
            return fail("No approved AR glasses are connected. Connect them, or approve them in Local display settings, then start 3D again.");
          }
          {
            std::lock_guard lock(physical_monitor_error_mutex);
            physical_monitor_error.clear();
          }
          ar_session_requested.store(true);
          return true;
        }
      default:
        return start ? fail("Local 3D is turned off, or the chosen display's saved settings are invalid. Choose a local display in settings, save, and restart Sunshine 3D before starting 3D.") : true;
    }
  }

  std::unique_ptr<platf::deinit_t> init() {
    load_devices(true);
    const auto pending_recoveries = recover_connected_saved_topologies();
    if (pending_recoveries != 0) {
      BOOST_LOG(info) << pending_recoveries
                      << " saved AR topology recovery record(s) remain pending until their display reconnects."sv;
    }
    return std::make_unique<controller_t>();
  }
}  // namespace ar_glasses
