/**
 * @file src/platform/windows/ar_glasses_session.cpp
 * @brief One local presentation session (local_session_t) and the target contracts its controller
 * uses.
 */
#include "ar_glasses_session.h"

#include "ar_glasses_arbiter.h"
#include "ar_glasses_ccd.h"
#include "ar_glasses_requests.h"
#include "ar_glasses_security.h"
#include "display_config.h"
#include "misc.h"
#include "primary_display.h"
#include "src/config.h"
#include "src/logging.h"
#include "src/process.h"
#include "src/utility.h"
#include "src/uuid.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <thread>
#include <utility>

#ifdef SUNSHINE_TESTS
  #include <condition_variable>
  #include <future>
#endif

namespace ar_glasses {
  using namespace std::literals;
  using namespace internal;

  namespace {
    constexpr char virtual_display_uuid[] = "F43872A4-9948-45DF-BD84-5B33BD71E4E8";
    constexpr char virtual_display_name[] = "Apollo AR Desktop";

    constexpr auto monitor_source_switch_timeout = 5s;

    constexpr auto virtual_display_absence_grace = 5s;
    constexpr unsigned virtual_display_absence_confirmations = 3;

    bool can_reconfigure_local_session(
      const std::optional<target_state_t> &left,
      const std::optional<target_state_t> &right,
      bool exclusive
    ) {
      return same_physical_output(left, right) &&
             target_source_width(*left) == target_source_width(*right) &&
             target_source_height(*left) == target_source_height(*right) &&
             is_monitor_mode(left->mode) == is_monitor_mode(right->mode) &&
             left->mode != presentation_mode_e::unsupported &&
             right->mode != presentation_mode_e::unsupported &&
             (exclusive || !right->is_primary) && !right->is_cloned;
    }

    bool presentation_target_contract_matches(const target_state_t &expected, const target_state_t &actual) {
      // Placement may change GDI names and coordinates, but it must not silently adopt a new
      // capture/presentation generation or turn the physical sink into the interactive primary.
      return equal_device_paths(actual.device_path, expected.device_path) &&
             same_luid(actual.adapter_id, expected.adapter_id) &&
             actual.mode == expected.mode &&
             actual.refresh_millihz == expected.refresh_millihz &&
             actual.rect.right - actual.rect.left == expected.rect.right - expected.rect.left &&
             actual.rect.bottom - actual.rect.top == expected.rect.bottom - expected.rect.top &&
             actual.hdr.active == expected.hdr.active &&
             !actual.is_primary && !actual.is_cloned;
    }

    bool same_monitor_scanout_generation(const target_state_t &left, const target_state_t &right) {
      return is_monitor_mode(left.mode) && is_monitor_mode(right.mode) && !left.is_primary && !left.is_cloned &&
             left.hdr.known && left.same_display_observation(right) &&
             same_luid(left.source_adapter_id, right.source_adapter_id) && left.source_id == right.source_id &&
             left.target_id == right.target_id && left.hdr.user_enabled == right.hdr.user_enabled &&
             left.hdr.bits_per_color == right.hdr.bits_per_color;
    }

    template<class Current, class Verify>
    bool verify_monitor_source_request(Current current, Verify verify) {
      return current() && verify() && current();
    }

    bool restore_monitor_desktop(
      const std::shared_ptr<platf::monitor_desktop_handoff_t> &handoff,
      const std::wstring &device_path,
      RECT original_rect
    ) {
      if (handoff) {
        const auto target = find_target(device_path);
        return handoff->restore(target ? target->rect : original_rect);
      }
      return true;
    }
  }  // namespace

  namespace internal {
    bool same_physical_output(
      const std::optional<target_state_t> &left,
      const std::optional<target_state_t> &right
    ) {
      return left && right &&
             equal_device_paths(left->device_path, right->device_path) &&
             same_luid(left->adapter_id, right->adapter_id);
    }

    bool same_presentation_contract(
      const std::optional<target_state_t> &left,
      const std::optional<target_state_t> &right
    ) {
      if (!left || !right) {
        return !left && !right;
      }
      const auto left_width = left->rect.right - left->rect.left;
      const auto left_height = left->rect.bottom - left->rect.top;
      const auto right_width = right->rect.right - right->rect.left;
      const auto right_height = right->rect.bottom - right->rect.top;
      return equal_device_paths(left->device_path, right->device_path) &&
             same_luid(left->adapter_id, right->adapter_id) &&
             left_width == right_width && left_height == right_height &&
             left->refresh_millihz == right->refresh_millihz && left->mode == right->mode &&
             // The presenter is started for one source (Host 3D or a game's own stereo). A
             // placement-only update keeps that presenter, so it cannot serve the other source.
             left->game_source == right->game_source &&
             left->is_primary == right->is_primary && left->is_cloned == right->is_cloned &&
             left->desktop_topology == right->desktop_topology &&
             left->hdr.known == right->hdr.known &&
             left->hdr.supported == right->hdr.supported &&
             left->hdr.active == right->hdr.active &&
             left->hdr.limited_by_policy == right->hdr.limited_by_policy;
    }

    bool can_re_isolate_local_session(
      const std::optional<target_state_t> &left,
      const std::optional<target_state_t> &right,
      bool transition_presenter_paused
    ) {
      // A failed placement/color verification pauses the presenter. The next retained update
      // must run full mode/color repair; repeating placement alone cannot repair source HDR.
      return left && right && !transition_presenter_paused &&
             same_presentation_contract(left, right);
    }

    bool monitor_source_only_change(const target_state_t &before, const target_state_t &after) {
      return before.mode != after.mode && same_monitor_scanout_generation(before, after);
    }

    const char *mode_name(presentation_mode_e mode) {
      switch (mode) {
        case presentation_mode_e::monitor_host:
          return "monitor Host 3D";
        case presentation_mode_e::monitor_game:
          return "monitor Game 3D";
        case presentation_mode_e::normal:
          return "normal";
        case presentation_mode_e::sbs_ai:
          return "SBS AI";
        default:
          return "unsupported";
      }
    }

    local_session_t::local_session_t(gpu_workload::lease_t live_gpu_lease, std::function<bool()> session_requested):
        live_gpu_lease_(std::move(live_gpu_lease)),
        exclusive_(config::video.local_ar_virtual_display_only),
        session_requested_(std::move(session_requested)) {
    }

    void local_session_t::initialize(const target_state_t &target, std::stop_token controller_stop_token) {
      const auto cancelled = [this, controller_stop_token]() {
        return controller_stop_token.stop_requested() || !session_requested_();
      };
      original_target_rect_ = target.rect;
      exclusive_ = target_exclusive(target);
      monitor_surface_ = is_monitor_mode(target.mode);
      source_width_ = target_source_width(target);
      source_height_ = target_source_height(target);
      target_device_path_ = target.device_path;
      target_adapter_id_ = target.adapter_id;
      // A shutter monitor that Windows runs at a desktop refresh rate keeps that rate through
      // the session's display changes, its teardown included. Every other session changes
      // displays exactly as before.
      if (monitor_surface_ && VDISPLAY::targetUsesDesktopRefreshRate(target.adapter_id, target.target_id)) {
        desktop_refresh_rate_keeper_.emplace();
        BOOST_LOG(info) << "Local monitor runs at the Windows desktop refresh rate "sv << (target.refresh_millihz / 1000.0)
                        << " Hz; the session keeps that rate through its display changes."sv;
      }
      if (proc::vDisplayDriverStatus != VDISPLAY::DRIVER_STATUS::OK) {
        BOOST_LOG(error) << "AR glasses detected, but the SudoVDA driver is unavailable."sv;
        return;
      }
      if (cancelled()) {
        return;
      }
      auto active_target = target;
      if (monitor_surface_) {
        // API callers can live on a service/tray desktop. The input desktop and console
        // session are authoritative here, immediately before any new display preparation.
        if (const auto reason = query_monitor_desktop_block_reason(L"Default", target.rect); !reason.empty()) {
          request_monitor_session(false);
          std::lock_guard lock(physical_monitor_error_mutex);
          physical_monitor_error = reason;
          return;
        }
        desktop_handoff_ = std::make_shared<platf::monitor_desktop_handoff_t>();
        if (!desktop_handoff_->capture(target.rect)) {
          BOOST_LOG(error) << "Could not capture the selected monitor's interactive desktop before 3D activation."sv;
          return;
        }
      }

      if (target.hdr.known) {
        BOOST_LOG(info) << "AR display HDR: supported="sv << target.hdr.supported
                        << " enabled="sv << target.hdr.user_enabled
                        << " active="sv << target.hdr.active
                        << " bits_per_color="sv << target.hdr.bits_per_color << '.';
      } else {
        BOOST_LOG(warning) << "Could not query the AR display's HDR capability; keeping SDR."sv;
      }
      if (target.hdr.known && !target.hdr.supported) {
        BOOST_LOG(info) << "The AR display currently advertises SDR only. If the glasses have an "sv
                        << "internal HDR10 mode, enable it in the glasses menu; Sunshine 3D will detect "sv
                        << "the updated capability."sv;
      }

      std::string uuid_string = virtual_display_uuid;
      auto uuid = uuid_util::uuid_t::parse(uuid_string);
      VDISPLAY::display_spec_t display_spec {
        .client_uid = virtual_display_uuid,
        .client_name = virtual_display_name,
        .width = static_cast<std::uint32_t>(source_width_),
        .height = static_cast<std::uint32_t>(source_height_),
        .refresh_millihz = static_cast<std::uint32_t>(active_target.refresh_millihz),
        .render_adapter_luid = active_target.adapter_id,
        .exclusive = exclusive_ && !monitor_surface_,
        .monitor_replacement = monitor_surface_,
        .local_sink = target_device_path_,
        .primary_binding = exclusive_,
      };
      static_assert(sizeof(display_spec.guid) == sizeof(uuid));
      std::memcpy(&display_spec.guid, &uuid, sizeof(display_spec.guid));

      // Extended AR retains its row journal. Exclusive AR uses the shared owner's normal
      // durable preparation, including preserving this local presentation sink.
      decltype(build_pre_add_isolation_plan(target_device_path_)) pre_add_plan;
      const std::function<bool()> prepare_extended = exclusive_ ? std::function<bool()> {} : [&]() {
        pre_add_plan = build_pre_add_isolation_plan(target_device_path_);
        if (!pre_add_plan || !same_rect(pre_add_plan->physical_rect, original_target_rect_)) {
          BOOST_LOG(info) << "AR topology changed before virtual-display creation, or no safe "sv
                             "non-primary/non-cloned isolation row exists; preserving the current layout."sv;
          return false;
        }
        if (!begin_topology_recovery_move(
              target_device_path_,
              original_target_rect_,
              pre_add_plan->layout.physical_rect
            )) {
          BOOST_LOG(error) << "Refusing to attach the local AR virtual desktop without a durable pre-add topology transaction."sv;
          return false;
        }
        const auto confirmed_pre_add_plan = build_pre_add_isolation_plan(target_device_path_);
        if (!confirmed_pre_add_plan || !same_rect(confirmed_pre_add_plan->physical_rect, original_target_rect_) || !same_rect(confirmed_pre_add_plan->layout.virtual_rect, pre_add_plan->layout.virtual_rect) || !same_rect(confirmed_pre_add_plan->layout.physical_rect, pre_add_plan->layout.physical_rect)) {
          BOOST_LOG(info) << "Desktop topology changed while the pre-add AR recovery marker was being persisted; "sv
                             "cancelling before virtual-display creation."sv;
          if (!cancel_topology_recovery_move(target_device_path_)) {
            BOOST_LOG(error) << "Could not cancel the stale pre-add AR topology transaction."sv;
          }
          return false;
        }
        return true;
      };
      const bool acquired = virtual_display_.acquire(std::move(display_spec), prepare_extended, cancelled);
      const auto created_display = virtual_display_binding();
      if (created_display.identity) {
        BOOST_LOG(debug) << "Local AR virtual desktop identity: adapter="sv
                         << created_display.identity->AdapterLuid.HighPart << ':'
                         << created_display.identity->AdapterLuid.LowPart
                         << " target="sv << created_display.identity->TargetId << '.';
      }
      if (!acquired || created_display.display_name.empty()) {
        if (cancelled()) {
          return;
        }
        BOOST_LOG(error) << (virtual_display_.owns_display() ? "The local AR virtual desktop was added, but its published identity could not be bound."sv : "Failed to prepare or create the local AR virtual desktop."sv);
        return;
      }

      if (exclusive_) {
        if (monitor_surface_) {
          // acquire() has already bound the exact Add result to durable recovery. Windows may
          // publish neither its native raster nor a stable active CCD route immediately; let
          // the bounded identity-only repair wait before requiring the final geometry.
          if (!establish_native_source_mode(active_target.refresh_millihz, cancelled)) {
            BOOST_LOG(error) << "Could not establish the selected monitor's native local source mode before display replacement."sv;
            return;
          }
          // Before the replacement topology is built from what Windows reports now.
          keep_monitor_refresh(active_target, cancelled);
        } else if (!refresh_virtual_display_reference() || !virtual_display_.bind()) {
          BOOST_LOG(error) << "Could not resolve and bind the local AR source after virtual-display creation."sv;
          return;
        }
        if (!virtual_display_.promote()) {
          BOOST_LOG(error) << (monitor_surface_ ? "Could not apply the selected monitor replacement topology while preserving the other displays."sv : "Could not establish the local AR virtual desktop as the exclusive primary display."sv);
          return;
        }
      } else {
        bool post_add_query_succeeded = false;
        const auto post_add_target = find_target(target_device_path_, {}, &post_add_query_succeeded);
        if (!post_add_query_succeeded || !post_add_target) {
          BOOST_LOG(info) << "Physical AR topology became indeterminate while attaching SudoVDA; "sv
                             "stopping without claiming a new rectangle."sv;
          return;
        }
        if (!same_rect(post_add_target->rect, original_target_rect_) && !same_rect(post_add_target->rect, pre_add_plan->layout.physical_rect)) {
          BOOST_LOG(info) << "SudoVDA attachment coincided with an unexpected physical-display move; "sv
                             "preserving that position as user-owned and rebuilding later."sv;
          return;
        }
      }

      auto presentation_target = active_target;
      const bool configured = update_display_topology([&]() -> bool {
        // Promotion can renumber GDI sources. Resolve the retained driver identity inside the
        // serialized transaction before the first mode setter, not only after color setup.
        if (!refresh_virtual_display_reference()) {
          BOOST_LOG(warning) << "Could not resolve the local AR source after initial display promotion."sv;
          return false;
        }
        if (!configure_source_refresh(active_target.refresh_millihz, cancelled)) {
          return false;
        }
        if (const auto isolated_target = isolate_target()) {
          presentation_target = *isolated_target;
          active_target = *isolated_target;
        } else {
          BOOST_LOG(warning) << "Could not establish the local AR source/sink topology; "sv
                                "stopping before any unowned presentation can begin."sv;
          return false;
        }

        // Let DXGI observe the newly attached output before capture initializes.
        for (int sleep_step = 0; sleep_step < 6 && !cancelled(); ++sleep_step) {
          std::this_thread::sleep_for(50ms);
        }
        if (cancelled()) {
          return false;
        }
        if (!refresh_virtual_display_reference()) {
          BOOST_LOG(error) << "Could not resolve the newly created local AR virtual display by its driver identity."sv;
          return false;
        }

        // Color changes follow the initial durable placement. Exclusive sessions serialize the
        // complete update with display reconciliation; extended sessions keep their row journal.
        if (active_target.hdr.supported && !active_target.hdr.limited_by_policy && !active_target.hdr.active) {
          if (!enable_physical_hdr(active_target, cancelled)) {
            return false;
          }
          presentation_target = active_target;
        }
        const auto physical_hdr = active_target.hdr;
        if (physical_hdr.supported && physical_hdr.limited_by_policy) {
          BOOST_LOG(warning) << "HDR on the AR display is disabled by Windows policy."sv;
        }
        const bool requested_target_hdr = physical_hdr.active;
        if (physical_hdr.supported && !requested_target_hdr) {
          BOOST_LOG(warning) << "The AR display supports HDR, but HDR is not active in its current mode."sv;
        } else if (requested_target_hdr) {
          BOOST_LOG(info) << "AR display HDR is active at "sv << physical_hdr.bits_per_color
                          << " bits per color."sv;
        }

        if (monitor_surface_) {
          if (!configure_monitor_source_hdr(active_target.refresh_millihz, requested_target_hdr, true, cancelled)) {
            return false;
          }
          virtual_hdr_active_ = requested_target_hdr;
          // Colour changes on the source set its mode again; the final promotion below records
          // whatever rate the monitor is at then.
          keep_monitor_refresh(target, cancelled);
          // Advanced Color may temporarily deactivate the still-connected physical sink.
          // Return only after the exact source is ready, allowing this transaction's final
          // owner promotion to restore scanout. Validate the physical contract after promotion,
          // before activating desktop handoff or opening any presenter/emitter resources.
          return !cancelled();
        }

        // Every look at the source refreshes its binding, so the source keeps its learned identity
        // through any renumbering the color change causes.
        const auto source_color = configure_source_color(requested_target_hdr, true, active_target.refresh_millihz, cancelled);
        if (!source_color) {
          if (cancelled()) {
            return false;
          }
          BOOST_LOG(error) << "Local AR virtual display color-mode configuration failed; rebuilding the session."sv;
          return false;
        }
        const bool virtual_hdr_active = *source_color;
        BOOST_LOG(info) << "Local AR source color mode: "sv
                        << (virtual_hdr_active ? "HDR linear scRGB"sv : "SDR Rec.709"sv) << '.';
        if (virtual_hdr_active) {
          const auto source = virtual_display_binding();
          const auto source_white = query_sdr_white_nits(
            source.identity->AdapterLuid,
            source.identity->TargetId
          );
          const auto target_white = query_sdr_white_nits(active_target.adapter_id, active_target.target_id);
          BOOST_LOG(info) << "Local AR HDR SDR-reference white: source="sv
                          << (source_white ? std::to_string(*source_white) : "unknown"s)
                          << " nits target="sv
                          << (target_white ? std::to_string(*target_white) : "unknown"s) << " nits."sv;
        }

        // Advanced Color can renumber GDI sources. Refresh the stable physical PnP target before
        // publishing the live rectangle/name consumed by the DXGI presenter.
        if (const auto refreshed_target = find_target(target_device_path_)) {
          const int refreshed_width = refreshed_target->rect.right - refreshed_target->rect.left;
          const int refreshed_height = refreshed_target->rect.bottom - refreshed_target->rect.top;
          const int expected_width = active_target.rect.right - active_target.rect.left;
          const int expected_height = active_target.rect.bottom - active_target.rect.top;
          if (refreshed_width != expected_width || refreshed_height != expected_height || refreshed_target->refresh_millihz != active_target.refresh_millihz || refreshed_target->mode != active_target.mode) {
            BOOST_LOG(info) << "AR output mode changed during virtual HDR setup; waiting for the topology controller."sv;
            return false;
          }
          if (!physical_adapter_contract_valid(active_target.adapter_id, refreshed_target->adapter_id)) {
            BOOST_LOG(info) << "AR output migrated to another graphics adapter during setup; "sv
                               "waiting for the topology controller to rebuild the complete session."sv;
            return false;
          }
          if (refreshed_target->hdr.active != physical_hdr.active) {
            BOOST_LOG(info) << "AR output HDR state changed during virtual-display setup; "sv
                               "waiting for one stable topology generation before presentation."sv;
            return false;
          }
          if (virtual_hdr_active && !refreshed_target->hdr.active) {
            BOOST_LOG(error) << "Local AR virtual source is HDR while the final physical output is SDR; rebuilding instead of presenting with mismatched color state."sv;
            return false;
          }
          active_target = *refreshed_target;
          presentation_target = *refreshed_target;
        } else {
          BOOST_LOG(error) << "Could not refresh the physical AR output after virtual HDR setup."sv;
          return false;
        }

        // Advanced Color frequently lets Windows normalize the physical output back into the
        // interactive row. Reapply isolation after every HDR transition instead of accepting the
        // new position as the presentation target.
        if (const auto isolated_target = isolate_target()) {
          presentation_target = *isolated_target;
          active_target = *isolated_target;
        } else {
          BOOST_LOG(warning) << "Could not resolve the AR presentation topology after color-mode setup; "sv
                                "stopping instead of presenting through an unowned topology."sv;
          return false;
        }

        // SetDisplayConfig can itself change Advanced Color state or renumber the physical path.
        // Reconcile the final PnP target once more after the last topology mutation and start only
        // from a coherent adapter/mode/color generation.
        if (const auto final_target = find_target(target_device_path_)) {
          const int final_width = final_target->rect.right - final_target->rect.left;
          const int final_height = final_target->rect.bottom - final_target->rect.top;
          const int expected_width = active_target.rect.right - active_target.rect.left;
          const int expected_height = active_target.rect.bottom - active_target.rect.top;
          if (!physical_adapter_contract_valid(active_target.adapter_id, final_target->adapter_id) || final_width != expected_width || final_height != expected_height || final_target->refresh_millihz != active_target.refresh_millihz || final_target->mode != active_target.mode || final_target->hdr.active != active_target.hdr.active || (virtual_hdr_active && !final_target->hdr.active)) {
            BOOST_LOG(info) << "AR output contract changed during final placement; waiting for the topology controller to rebuild from the stable state."sv;
            return false;
          }
          active_target = *final_target;
          presentation_target = *final_target;
        } else {
          BOOST_LOG(error) << "Could not resolve the final physical AR output after placement."sv;
          return false;
        }

        virtual_hdr_active_ = virtual_hdr_active;
        return true;
      });
      if (!configured) {
        if (!cancelled()) {
          BOOST_LOG(warning) << "Local presentation display set-up did not complete; no picture was started."sv;
        }
        return;
      }
      // The exclusive transaction can renumber or move the sink during its final verification.
      // Publish its current exact identity and coordinates only after that transaction completes.
      if (!refresh_virtual_display_reference()) {
        BOOST_LOG(warning) << "The local source display could not be found after display set-up; no picture was started."sv;
        return;
      }
      const auto final_target = find_target(target_device_path_);
      if (!final_target || !final_presentation_contract_matches(presentation_target, *final_target)) {
        BOOST_LOG(warning) << "The physical output changed during display set-up; no picture was started. Planned "sv
                           << (presentation_target.rect.right - presentation_target.rect.left) << 'x'
                           << (presentation_target.rect.bottom - presentation_target.rect.top) << '@'
                           << (presentation_target.refresh_millihz / 1000.0) << " HDR="sv << presentation_target.hdr.active
                           << ", found "sv
                           << (final_target ? final_target->rect.right - final_target->rect.left : 0) << 'x'
                           << (final_target ? final_target->rect.bottom - final_target->rect.top : 0) << '@'
                           << (final_target ? final_target->refresh_millihz / 1000.0 : 0.0) << " HDR="sv
                           << (final_target ? final_target->hdr.active : false) << '.';
        return;
      }
      presentation_target = *final_target;
      BOOST_LOG(info) << "Local AR desktop layout: "sv
                      << (monitor_surface_ ? "selected monitor replaced by the interactive source; other displays remain active."sv :
                          exclusive_ ? "virtual desktop primary, glasses active, other displays disabled; cursor confined."sv : "extended desktop; physical displays remain active."sv);
      if (cancelled()) {
        return;
      }
      if (!activate_monitor_desktop()) {
        return;
      }
      start_presenter(presentation_target);
      ready_ = true;
    }

    void local_session_t::pause_presenter(const std::optional<target_state_t> &target) {
      stop_presenter();
      if (target && live_target_) {
        std::lock_guard lock(live_target_->mutex);
        live_target_->rect = target->rect;
        live_target_->display_name = target->gdi_name;
      }
      refresh_pointer_isolation();
    }

    bool local_session_t::same_retained_device(const target_state_t &target) const {
      return equal_device_paths(target.device_path, target_device_path_) && same_luid(target.adapter_id, target_adapter_id_) &&
             is_monitor_mode(target.mode) == monitor_surface_ &&
             target_source_width(target) == source_width_ && target_source_height(target) == source_height_;
    }

    bool local_session_t::suspend_for_resume() {
      presenter_.request_stop();
      cancel_monitor_source_switch();
      if (presenter_.joinable() && presenter_quiesced_) {
        // The renderer acknowledges that capture and input are stopped before its potentially
        // slow graphics/scheduling destructors. Keep those resources owned until the join below.
        presenter_quiesced_->view();
      }
      auto finish_presenter_cleanup = util::fail_guard([this]() {
        const auto cleanup_started = std::chrono::steady_clock::now();
        stop_presenter();
        live_gpu_lease_.reset();
        const auto cleanup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - cleanup_started
        );
        if (cleanup_ms > 250ms) {
          BOOST_LOG(info) << "Local AR presentation-resource cleanup completed "sv
                          << cleanup_ms.count() << " ms after the desktop pause attempt."sv;
        }
      });
      auto release_pointer = util::fail_guard([this]() {
        // Let the shared pause bookmark the cursor before restoring an older application clip,
        // which can itself move it. Still release on failure, before joining the presenter.
        release_pointer_isolation();
        cursor_clip_.reset();
      });

      if (!exclusive_ && !resume_topology_cleanup_) {
        bool topology_known = false;
        const auto current = find_target(target_device_path_, {}, &topology_known);
        if (!topology_known || (current && !rebase_teardown_layout(*current))) {
          return false;
        }
        resume_topology_cleanup_ = std::make_shared<pre_removal_topology_cleanup_t>(
          classify_pre_removal_topology(target_device_path_),
          target_device_path_
        );
      }
      if (desktop_handoff_) {
        desktop_handoff_->prepare_restore();
        release_pointer_isolation();
        cursor_clip_.reset();
      }
      if (!virtual_display_.paused()) {
        if (resume_topology_cleanup_) {
          const auto source = virtual_display_binding();
          const auto probe = source.identity ? probe_virtual_display(*source.identity, source.device_path, source.display_name, source_width_, source_height_) : virtual_display_probe_t {};
          if (probe.presence == virtual_display_presence_e::indeterminate || (probe.presence == virtual_display_presence_e::present && !resume_topology_cleanup_->prepare_removal())) {
            return false;
          }
        }
        if (!virtual_display_.pause(target_device_path_)) {
          return false;
        }
      }
      if (!restore_monitor_desktop(desktop_handoff_, target_device_path_, original_target_rect_)) {
        return false;
      }
      // In extended mode the source occupied the original sink position. Only restore that
      // rectangle after the shared transaction has detached the source and cleared its journal.
      return !resume_topology_cleanup_ || resume_topology_cleanup_->finish();
    }

    std::optional<target_state_t> local_session_t::resume_retained(const target_state_t &target, std::stop_token stop_token) {
      if (!same_retained_device(target) || target.mode == presentation_mode_e::unsupported || target.is_cloned || (!exclusive_ && target.is_primary) || stop_token.stop_requested() || !session_requested_()) {
        return std::nullopt;
      }
      auto lease = gpu_workload::try_acquire(gpu_workload::kind_e::live_stream);
      if (!lease) {
        return std::nullopt;
      }
      live_gpu_lease_ = std::move(*lease);
      if (desktop_handoff_ && !desktop_handoff_->capture(target.rect)) {
        live_gpu_lease_.reset();
        return std::nullopt;
      }
      // Reactivation may partially apply before verification fails. Every failure must run
      // pause again; the previous successful detachment no longer certifies the live topology.
      if (!virtual_display_.begin_resume()) {
        live_gpu_lease_.reset();
        return std::nullopt;
      }
      resume_topology_cleanup_.reset();
      // Reactivation may renumber both outputs. Reconfigure only from the actual physical
      // generation and let the shared source refresh/color path validate before presentation.
      const auto actual = find_target(target_device_path_);
      const auto resumed = actual && refresh_virtual_display_reference() ?
                             reconfigure_target(*actual, stop_token) :
                             std::nullopt;
      if (!resumed) {
        // An unsuccessful resume keeps its original deadline and must return the desktop
        // immediately, even when Windows accepted only part of the activation transaction.
        (void) suspend_for_resume();
        return std::nullopt;
      }
      virtual_display_.commit_resume();
      return resumed;
    }

    // A window that opens on the parked physical output while Monitor 3D is presenting would sit
    // behind the presenter. Send it to the 3D desktop; one that will not go is left where it is
    // by the handoff itself, and the session continues.
    void local_session_t::refresh_monitor_windows() {
      if (!monitor_surface_ || !desktop_handoff_ || !live_target_ || !running_.load()) {
        return;
      }
      if (const auto source = refresh_virtual_display_reference()) {
        RECT sink_rect {};
        {
          std::lock_guard lock(live_target_->mutex);
          sink_rect = live_target_->rect;
        }
        (void) desktop_handoff_->refresh(sink_rect, source->rect);
      }
    }

    void local_session_t::refresh_pointer_isolation() {
      if (exclusive_) {
        if (!monitor_surface_) {
          platf::primary_display::refresh_exclusive_cursor_clip();
        }
        return;
      }
      const auto source_name = virtual_display_binding().display_name;
      if (!source_name.empty()) {
        platf::dxgi::refresh_local_presenter_pointer_isolation(
          platf::to_utf8(source_name),
          live_target_,
          cursor_clip_
        );
      }
    }

    void local_session_t::release_pointer_isolation() {
      if (!cursor_clip_) {
        return;
      }
      for (int attempt = 0; attempt < 8; ++attempt) {
        if (cursor_clip_->restore()) {
          return;
        }
        std::this_thread::sleep_for(5ms);
      }
    }

    bool local_session_t::virtual_display_authoritatively_absent() {
      std::lock_guard lock(virtual_display_mutex_);
      const auto source = virtual_display_.binding();
      if (!source.identity) {
        return false;
      }
      const auto probe = probe_virtual_display(
        *source.identity,
        source.device_path,
        source.display_name,
        source_width_, source_height_
      );
      if (probe.presence != virtual_display_presence_e::absent) {
        virtual_display_absence_started_.reset();
        virtual_display_absence_observations_ = 0;
        if (probe.resolved) {
          auto refreshed = source;
          refreshed.identity = probe.resolved->identity;
          refreshed.display_name = probe.resolved->gdi_name;
          refreshed.device_path = probe.resolved->device_path;
          virtual_display_.update_binding(std::move(refreshed));
        }
        return false;
      }

      const auto now = std::chrono::steady_clock::now();
      if (!virtual_display_absence_started_) {
        virtual_display_absence_started_ = now;
        virtual_display_absence_observations_ = 1;
        return false;
      }
      ++virtual_display_absence_observations_;
      return virtual_display_absence_observations_ >= virtual_display_absence_confirmations &&
             now - *virtual_display_absence_started_ >= virtual_display_absence_grace;
    }

    bool local_session_t::has_monitor_source_switch() const {
      return monitor_source_switch_ && monitor_source_switch_->request.load();
    }

    monitor_source_switch_progress_e local_session_t::advance_monitor_source_switch(
      const target_state_t &expected,
      std::uint64_t display_epoch,
      std::stop_token controller_stop_token
    ) {
      using enum monitor_source_switch_progress_e;
      if (!monitor_source_switch_ || !desktop_handoff_ || !live_gpu_lease_) {
        return failed;
      }
      const bool game = expected.mode == presentation_mode_e::monitor_game;
      const int mode = game ? ::video::SBS_GAME_SBS : ::video::SBS_AI;
      const auto current = [&]() {
        return !controller_stop_token.stop_requested() && session_requested_() && running() && has_presented() &&
               game == monitor_uses_game_source() &&
               platf::display_change_epoch() == display_epoch && !remote_virtual_display_blocks_local();
      };
      const auto verify = [&]() {
        const auto sink = find_target(target_device_path_);
        if (!sink || sink->mode != expected.mode || !same_monitor_scanout_generation(expected, *sink)) {
          return false;
        }
        const auto binding = virtual_display_binding();
        const auto source = binding.identity ? resolve_virtual_display(*binding.identity, binding.device_path, binding.display_name, nullptr, source_width_, source_height_) : std::nullopt;
        if (!source || !equal_device_paths(source->device_path, binding.device_path) || !equal_device_paths(source->gdi_name, binding.display_name) || !same_virtual_display_identity(source->identity, *binding.identity) || !source_refresh_matches(source->refresh_millihz, sink->refresh_millihz)) {
          return false;
        }
        const auto hdr = VDISPLAY::queryDisplayHDR(source->identity.AdapterLuid, source->identity.TargetId);
        if (!hdr || *hdr != virtual_hdr_active_ || sink->hdr.active != virtual_hdr_active_ || !platf::primary_display::verify_local_monitor_replacement(source->device_path, sink->device_path)) {
          return false;
        }
        if (const auto reason = query_monitor_desktop_block_reason(L"Default", sink->rect); !reason.empty()) {
          request_monitor_session(false);
          std::lock_guard lock(physical_monitor_error_mutex);
          physical_monitor_error = reason;
          return false;
        }
        return true;
      };
      const auto request = monitor_source_switch_->request.load();
      if (!request) {
        if (!verify_monitor_source_request(current, verify)) {
          return failed;
        }
        // Preparation and scanout happen on their existing owners. The topology controller
        // keeps watching the desktop and remote admission while the old pair is displayed.
        const auto expires = std::chrono::steady_clock::now() + monitor_source_switch_timeout;
        monitor_source_switch_->request.store(std::make_shared<monitor_source_switch_t::request_t>(mode, display_epoch, expires));
        return waiting;
      }
      if (request->sbs_mode != mode || request->display_epoch != display_epoch) {
        return failed;
      }
      switch (request->state.load()) {
        case monitor_source_switch_t::state_e::requested:
        case monitor_source_switch_t::state_e::adopted:
          return current() && std::chrono::steady_clock::now() < request->expires ? waiting : failed;
        case monitor_source_switch_t::state_e::presented:
          if (monitor_source_switch_->presented_mode.load() != mode || !verify_monitor_source_request(current, verify)) {
            return failed;
          }
          monitor_source_switch_->request.store({});
          return presented;
        case monitor_source_switch_t::state_e::failed:
        case monitor_source_switch_t::state_e::cancelled:
          return failed;
      }
      return failed;
    }

    std::optional<target_state_t> local_session_t::reconfigure_target(
      const target_state_t &target,
      std::stop_token controller_stop_token
    ) {
      const bool requested_game = target.mode == presentation_mode_e::monitor_game;
      const auto cancelled = [this, controller_stop_token, requested_game]() {
        return controller_stop_token.stop_requested() || !session_requested_() ||
               (monitor_surface_ && requested_game != monitor_uses_game_source());
      };
      stop_presenter();
      if (cancelled() || !equal_device_paths(target.device_path, target_device_path_) || !same_luid(target.adapter_id, target_adapter_id_) || target.mode == presentation_mode_e::unsupported) {
        return std::nullopt;
      }

      std::optional<target_state_t> presentation_target;
      const bool configured = update_display_topology([&]() -> bool {
        if (!exclusive_) {
          const auto rebased_original = prepare_topology_recovery_for_mode_change(
            target_device_path_,
            original_target_rect_,
            target.rect
          );
          if (!rebased_original) {
            BOOST_LOG(warning) << "Could not durably rebase AR topology recovery for the new physical mode; preserving the virtual desktop and retrying later."sv;
            return false;
          }
          original_target_rect_ = *rebased_original;
        }

        if (monitor_surface_) {
          const bool desired_hdr = target.hdr.known && target.hdr.active;
          if (!configure_monitor_source_hdr(target.refresh_millihz, desired_hdr, false, cancelled)) {
            return false;
          }
          virtual_hdr_active_ = desired_hdr;
          presentation_target = target;
          // The owner applies the coherent source/sink topology after this callback. Its
          // physical result is checked by the shared final contract before handoff resumes.
          return !cancelled();
        }

        if (!refresh_virtual_display_reference()) {
          BOOST_LOG(warning) << "The local AR virtual desktop could not be resolved after the physical mode change; preserving it for a later topology retry."sv;
          return false;
        }

        if (!configure_source_refresh(target.refresh_millihz, cancelled)) {
          return false;
        }

        for (int sleep_step = 0; sleep_step < 6 && !cancelled(); ++sleep_step) {
          std::this_thread::sleep_for(50ms);
        }
        if (cancelled() || !refresh_virtual_display_reference()) {
          return false;
        }

        // Unknown physical color state must not preserve an old HDR source: if the new mode is
        // actually SDR, that mismatch makes the presenter reject every target generation. SDR is
        // the safe common contract and remains color-managed when the sink later proves to be HDR.
        const bool desired_virtual_hdr = target.hdr.known && target.hdr.active;
        // The same rule and SDR fallback as initial start-up: an HDR physical sink may consume a
        // color-managed SDR swapchain, so the desktop is never removed merely because its source
        // cannot activate Advanced Color in this mode. An HDR source on an SDR sink cannot be
        // presented coherently, so a source that cannot return to SDR is retried.
        const auto configured_virtual_hdr = configure_source_color(desired_virtual_hdr, false, target.refresh_millihz, cancelled);
        if (!configured_virtual_hdr) {
          if (!cancelled()) {
            BOOST_LOG(warning) << "The persistent local AR source could not settle its color; retaining the desktop for retry."sv;
          }
          return false;
        }
        virtual_hdr_active_ = *configured_virtual_hdr;
        // Address the owned source by its stable adapter/target identity. A GDI name such as
        // DISPLAYn can be renumbered or reused by another display during a mode switch.
        const auto query_source_hdr = [&]() -> std::optional<bool> {
          const auto binding = virtual_display_binding();
          return binding.identity ? VDISPLAY::queryDisplayHDR(binding.identity->AdapterLuid, binding.identity->TargetId) :
                                    std::nullopt;
        };

        presentation_target = isolate_target();
        if (!presentation_target || !same_luid(presentation_target->adapter_id, target_adapter_id_) || presentation_target->mode != target.mode || (target.hdr.known && presentation_target->hdr.active != target.hdr.active)) {
          BOOST_LOG(warning) << "The new AR physical mode could not be safely re-isolated; preserving the virtual desktop for retry."sv;
          return false;
        }
        const auto final_virtual_hdr = query_source_hdr();
        if (!final_virtual_hdr || *final_virtual_hdr != virtual_hdr_active_) {
          BOOST_LOG(warning) << "The persistent local AR source color state changed during final isolation; retaining the desktop for a coherent retry."sv;
          return false;
        }
        return true;
      });
      if (!configured) {
        return std::nullopt;
      }
      if (!refresh_virtual_display_reference()) {
        return std::nullopt;
      }
      const auto final_target = find_target(target_device_path_);
      if (!presentation_target || !final_target || !final_presentation_contract_matches(*presentation_target, *final_target)) {
        return std::nullopt;
      }
      presentation_target = *final_target;
      if (cancelled()) {
        return std::nullopt;
      }
      if (!activate_monitor_desktop() || cancelled()) {
        return std::nullopt;
      }
      start_presenter(*presentation_target);
      BOOST_LOG(info) << "Local AR presentation switched in place to "sv
                      << mode_name(presentation_target->mode)
                      << " without recreating its virtual desktop."sv;
      return presentation_target;
    }

    bool local_session_t::prepare_same_output_teardown(const target_state_t &target) {
      stop_presenter();
      return rebase_teardown_layout(target);
    }

    bool local_session_t::rebase_teardown_layout(const target_state_t &target) {
      if (!equal_device_paths(target.device_path, target_device_path_) || !same_luid(target.adapter_id, target_adapter_id_) || !valid_rect(target.rect)) {
        return false;
      }
      if (exclusive_) {
        return true;
      }
      const auto rebased_original = prepare_topology_recovery_for_mode_change(
        target_device_path_,
        original_target_rect_,
        target.rect
      );
      if (!rebased_original) {
        BOOST_LOG(warning) << "Could not durably rebase local-AR recovery evidence before an incompatible same-output transition; retaining the virtual desktop for retry."sv;
        return false;
      }
      original_target_rect_ = *rebased_original;
      return true;
    }

    local_session_t::~local_session_t() {
      stop_presenter();
      if (desktop_handoff_) {
        desktop_handoff_->prepare_restore();
      }
      // The shared clip deliberately survives presenter-only restarts, but whole-session
      // teardown restores it before Windows removes the source rectangle it was based on.
      release_pointer_isolation();
      cursor_clip_.reset();

      if (exclusive_) {
        retire_exclusive_display();
        return;
      }

      // Removing SudoVDA can normalize the physical sink before recovery inspects it. Classify
      // ownership while the source/sink row still exists and remember the exact rectangle that
      // must survive teardown. An unrecognized rectangle is user-owned and is preserved rather
      // than being reclassified after Windows removes the virtual source.
      const auto pre_removal_topology = target_device_path_.empty() ?
                                          std::nullopt :
                                          classify_pre_removal_topology(target_device_path_);
      const auto topology_cleanup = resume_topology_cleanup_ ? resume_topology_cleanup_ : target_device_path_.empty() ? std::shared_ptr<pre_removal_topology_cleanup_t> {} :
                                                                                                                        std::make_shared<pre_removal_topology_cleanup_t>(pre_removal_topology, target_device_path_);
      const bool had_virtual_display = virtual_display_.owns_display();
      bool virtual_display_retired = !had_virtual_display;
      const auto finish_topology_cleanup = [topology_cleanup](VDISPLAY::session_t &) {
        return topology_cleanup->finish();
      };
      if (had_virtual_display) {
        const bool retirement_tracked = begin_local_virtual_display_retirement(
          std::move(virtual_display_),
          {
            .prepare = [topology_cleanup](VDISPLAY::session_t &session) {
              const bool paused = session.paused();
              if (session.has_retained() && !session.restore()) {
                return false;
              }
              return paused || topology_cleanup->prepare_removal();
            },
            .finish = finish_topology_cleanup,
          }
        );
        if (!retirement_tracked) {
          BOOST_LOG(error) << "Local AR teardown could not register its virtual-display retirement; retaining the existing cleanup barrier."sv;
        } else {
          virtual_display_retired = wait_for_local_virtual_display_retirement_impl(
            3s
          );
          if (!virtual_display_retired) {
            BOOST_LOG(warning) << "The local AR virtual desktop is still retiring; subsequent local or remote display creation will wait for its stable identity to disappear."sv;
          }
        }
      } else {
        // No driver identity was acquired, but the row journal may already need recovery.
        // Complete that obligation through the same owner without a driver-absence wait.
        virtual_display_.begin_retirement({
          .prepare = [](VDISPLAY::session_t &) {
            return true;
          },
          .finish = finish_topology_cleanup,
        });
      }
      if (!target_device_path_.empty()) {
        if (!virtual_display_retired) {
          // The source can still occupy the middle of the source/sink row. Moving the physical
          // sink while that source remains present can overlap displays or make Windows perform
          // another normalization. Leave the recovery evidence intact and finish only after the
          // retirement barrier has authoritatively observed the exact SudoVDA identity absent.
          BOOST_LOG(warning) << "Deferring AR topology restoration until the local virtual display is authoritatively retired."sv;
          return;
        }
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        // A retired virtual display completed this callback as part of the authoritative
        // retirement barrier. Do not apply it a second time: the user may move the monitor as
        // soon as SudoVDA disappears. A failed Add has no retirement callback, so clean it here.
        bool recovery_complete = had_virtual_display;
        while (!recovery_complete) {
          recovery_complete = virtual_display_.retire_until(deadline);
          if (recovery_complete || std::chrono::steady_clock::now() >= deadline) {
            break;
          }
          std::this_thread::sleep_for(100ms);
        }
        if (!recovery_complete) {
          if (pre_removal_topology && !pre_removal_topology->apollo_owned) {
            BOOST_LOG(warning) << "Could not reapply the pre-removal user-owned AR rectangle after "sv
                                  "SudoVDA retirement; no Sunshine 3D ownership is claimed for that position."sv;
          } else {
            BOOST_LOG(warning) << "Could not complete local-AR topology recovery during teardown; "sv
                                  "the durable record is retained for the next stable observation."sv;
          }
        }
      }
    }

    bool local_session_t::valid() const {
      return ready_;
    }

    bool local_session_t::can_reconfigure(
      const std::optional<target_state_t> &before,
      const std::optional<target_state_t> &after
    ) const {
      return can_reconfigure_local_session(before, after, exclusive_);
    }

    bool local_session_t::running() const {
      return running_.load();
    }

    bool local_session_t::failed() const {
      return failed_.load();
    }

    bool local_session_t::stable() const {
      return presented_frames_ && presented_frames_->load(std::memory_order_relaxed) >= 60;
    }

    bool local_session_t::has_presented() const {
      return presented_frames_ && presented_frames_->load(std::memory_order_relaxed) > 0;
    }

    std::optional<target_state_t> local_session_t::re_isolate_target(const target_state_t &expected_target) {
      auto source = refresh_virtual_display_reference();
      if (!source) {
        return std::nullopt;
      }
      if (exclusive_) {
        // The presenter may be reinitializing concurrently. Keep an owned identity/name pair,
        // and serialize the owner's promotion read with its binding refresh.
        std::lock_guard lock(virtual_display_mutex_);
        if (!virtual_display_.promote()) {
          return std::nullopt;
        }
      }
      const auto target = exclusive_ ? find_target(target_device_path_) :
                                       isolate_physical_output(source->gdi_name, target_device_path_, original_target_rect_);
      if (!target || !presentation_target_contract_matches(expected_target, *target)) {
        return std::nullopt;
      }
      source = refresh_virtual_display_reference();
      if (!source) {
        return std::nullopt;
      }
      const auto source_hdr = VDISPLAY::queryDisplayHDRByName(source->gdi_name.c_str());
      if (!source_hdr || *source_hdr != virtual_hdr_active_ || (virtual_hdr_active_ && !target->hdr.active)) {
        return std::nullopt;
      }
      if (!live_target_) {
        return target;
      }
      std::lock_guard lock(live_target_->mutex);
      live_target_->rect = target->rect;
      live_target_->display_name = target->gdi_name;
      return target;
    }

    bool local_session_t::activate_monitor_desktop() {
      if (!desktop_handoff_) {
        return true;
      }
      const auto source = refresh_virtual_display_reference();
      if (!source || source->rect.right - source->rect.left != source_width_ ||
          source->rect.bottom - source->rect.top != source_height_) {
        return false;
      }
      if (!desktop_handoff_->activate(source->rect)) {
        BOOST_LOG(error) << "The monitor's application windows could not enter the private 3D desktop; restoring 2D."sv;
        return false;
      }
      const auto sink = find_target(target_device_path_);
      if (!sink) {
        return false;
      }
      // Record the parked identity before Stop can race the first periodic handoff pass.
      // New asynchronous arrivals are reconciled by the controller without stalling capture.
      desktop_handoff_->refresh(sink->rect, source->rect);
      return true;
    }

    bool local_session_t::enable_physical_hdr(target_state_t &target, const std::function<bool()> &cancelled) {
      enum class ended_e {
        disconnected,
        mode_changed,
      };
      std::optional<ended_e> ended;
      std::optional<target_state_t> seen;
      const color_settle_request_t request {.hdr = true, .budget = 10s, .repair_mode = false};
      const auto result = settle_color_with_io(
        request,
        [&]() -> color_observation_t {
          // HDR activation can renumber the target or change its mode. Follow the stable PnP
          // target rather than the now-stale source/target IDs from the original path. A new
          // mode is a new display generation for the topology controller, not this request's.
          bool topology_known = false;
          const auto current = find_target(target_device_path_, {}, &topology_known);
          if (!topology_known) {
            return {};
          }
          if (!current || current->mode != target.mode || current->rect.right - current->rect.left != target.rect.right - target.rect.left || current->rect.bottom - current->rect.top != target.rect.bottom - target.rect.top) {
            ended = current ? ended_e::mode_changed : ended_e::disconnected;
            return {.presence = color_presence_e::gone};
          }
          seen = current;
          return {
            .presence = color_presence_e::present,
            .mode_ready = true,
            .hdr = current->hdr.known ? std::make_optional(current->hdr.active) : std::nullopt,
          };
        },
        []() {
        },
        [&](bool enabled) {
          const auto &addressed = seen ? *seen : target;
          return platf::display_config::set_hdr_state(addressed.adapter_id, addressed.target_id, enabled);
        },
        [](std::chrono::milliseconds delay) {
          std::this_thread::sleep_for(delay);
        },
        []() {
          return std::chrono::steady_clock::now();
        },
        cancelled
      );
      log_color_settle("AR display"sv, request, result);
      if (seen) {
        target = *seen;
      }
      switch (result.outcome) {
        case color_settle_e::settled:
          return true;
        case color_settle_e::disconnected:
          BOOST_LOG(info) << (ended == ended_e::mode_changed ? "AR output mode changed while enabling HDR; waiting for the topology controller to rebuild the session."sv : "AR display disconnected while enabling HDR."sv);
          return false;
        case color_settle_e::cancelled:
          return false;
        case color_settle_e::rejected:
          BOOST_LOG(warning) << "The AR display reports HDR support, but Windows rejected HDR activation."sv;
          return true;
        case color_settle_e::timed_out:
          if (!target.hdr.active) {
            // Some glasses advertise HDR before their on-device HDR10 mode is active. Treat
            // that as a stable SDR presentation state rather than tearing down and recreating
            // the virtual desktop every retry interval.
            BOOST_LOG(warning) << "AR display did not enter HDR after Windows accepted the request; continuing in SDR."sv;
          }
          return true;
      }
      return false;
    }

    color_settle_result_t local_session_t::settle_source_color(bool hdr, int requested_millihz, const std::function<bool()> &cancelled) {
      std::optional<resolved_virtual_display_t> seen;
      // A monitor's source may have Windows' remembered raster or rate restored by HDR, so an
      // exact owned identity is set back to its native mode. Glasses keep their verified cadence.
      const color_settle_request_t request {.hdr = hdr, .budget = source_color_budget(hdr), .repair_mode = monitor_surface_};
      const auto result = settle_color_with_io(
        request,
        [&]() -> color_observation_t {
          if (monitor_surface_) {
            // The learned PnP path remains authoritative while Advanced Color restores a
            // remembered mode or renumbers the active route. The physical monitor may be
            // deactivated meanwhile; the owner's final promotion restores it and the final
            // contract is checked afterwards.
            seen = refresh_virtual_display_reference(false);
            if (!seen) {
              return {};
            }
            return {
              .presence = color_presence_e::present,
              .mode_ready = source_mode_ready(*seen, source_width_, source_height_, requested_millihz),
              .hdr = VDISPLAY::queryDisplayHDR(seen->identity.AdapterLuid, seen->identity.TargetId),
            };
          }
          // Glasses: the source's color matters only while the glasses are connected. The source
          // resolves only at its 1920x1080 raster.
          bool topology_known = false;
          const auto sink = find_target(target_device_path_, {}, &topology_known);
          if (!topology_known) {
            return {};
          }
          if (!sink) {
            return {.presence = color_presence_e::gone};
          }
          seen = refresh_virtual_display_reference();
          if (!seen) {
            return {};
          }
          return {
            .presence = color_presence_e::present,
            .mode_ready = true,
            .hdr = VDISPLAY::queryDisplayHDR(seen->identity.AdapterLuid, seen->identity.TargetId),
          };
        },
        [&]() {
          // Only reached for a monitor's source, after a look resolved it.
          BOOST_LOG(info) << "Local monitor color transition: repairing owned source mode "sv
                          << (seen->rect.right - seen->rect.left) << 'x'
                          << (seen->rect.bottom - seen->rect.top) << '@'
                          << (seen->refresh_millihz / 1000.0) << " -> "sv
                          << source_width_ << 'x' << source_height_ << '@'
                          << (requested_millihz / 1000.0) << '.';
          VDISPLAY::changeDisplaySettings(seen->gdi_name.c_str(), source_width_, source_height_, requested_millihz, false);
        },
        [&](bool enabled) {
          // Address the source by the identity the last look resolved: Windows may renumber it.
          if (monitor_surface_) {
            BOOST_LOG(info) << "Local monitor color transition: requesting owned source HDR="sv << enabled << '.';
          }
          return seen && VDISPLAY::setDisplayHDR(seen->identity.AdapterLuid, seen->identity.TargetId, enabled);
        },
        [](std::chrono::milliseconds delay) {
          std::this_thread::sleep_for(delay);
        },
        []() {
          return std::chrono::steady_clock::now();
        },
        cancelled
      );
      log_color_settle(monitor_surface_ ? "Local monitor source"sv : "Local AR source"sv, request, result);
      if (result.outcome == color_settle_e::disconnected) {
        BOOST_LOG(info) << "AR display disconnected during virtual-display color setup."sv;
      }
      return result;
    }

    std::optional<bool> local_session_t::configure_source_color(
      bool expected_hdr,
      bool after_creation,
      int requested_millihz,
      const std::function<bool()> &cancelled
    ) {
      // Glasses may present a color-managed SDR source on an HDR output. A shutter monitor's
      // presentation requires its source and output colors to match, so it has no fallback.
      const auto color = establish_source_color_with_io(expected_hdr, after_creation, !monitor_surface_, [&](bool hdr) {
        return settle_source_color(hdr, requested_millihz, cancelled);
      });
      if (color && *color != expected_hdr) {
        BOOST_LOG(warning) << "Local AR source HDR is unavailable; using color-managed SDR presentation."sv;
      }
      return color;
    }

    bool local_session_t::configure_monitor_source_hdr(
      int requested_millihz,
      bool expected_hdr,
      bool reset_before_enable,
      const std::function<bool()> &cancelled
    ) {
      // Preserve the initial IddCx SDR-then-HDR workaround. Subsequent mode/color changes
      // converge directly so an already coherent HDR source need not toggle through SDR.
      if (!configure_source_color(expected_hdr, reset_before_enable, requested_millihz, cancelled)) {
        if (!cancelled()) {
          BOOST_LOG(error) << "Local monitor source could not establish its requested native cadence and HDR="sv
                           << expected_hdr << "; stopping before desktop handoff."sv;
        }
        return false;
      }
      const auto source = refresh_virtual_display_reference();
      const auto hdr = source ? VDISPLAY::queryDisplayHDR(source->identity.AdapterLuid, source->identity.TargetId) : std::nullopt;
      if (cancelled() || !source || !source_refresh_matches(source->refresh_millihz, requested_millihz) || !hdr || *hdr != expected_hdr) {
        return false;
      }
      virtual_display_.update_mode(source_width_, source_height_, static_cast<std::uint32_t>(source->refresh_millihz));
      BOOST_LOG(info) << "Local monitor source color mode: "sv
                      << (expected_hdr ? "HDR linear scRGB"sv : "SDR Rec.709"sv)
                      << " at verified "sv << (source->refresh_millihz / 1000.0) << " Hz."sv;
      return true;
    }

    // Windows gives a variable-refresh monitor its lower refresh rates as a desktop rate on
    // top of the monitor's fastest signal, and attaching the virtual source can leave the
    // monitor at the signal's own rate. The session was planned for the rate the monitor was
    // selected at, and a shutter profile may depend on that rate's long vertical blanking,
    // so ask Windows for it again.
    void local_session_t::keep_monitor_refresh(const target_state_t &selected, const std::function<bool()> &cancelled) {
      // Only a session that started on a desktop refresh rate asks Windows for it again.
      if (!desktop_refresh_rate_keeper_) {
        return;
      }
      const auto current = find_target(target_device_path_);
      if (!current) {
        BOOST_LOG(info) << "Local monitor refresh check: the monitor is not active while its source is attached; its recorded "sv
                        << (selected.refresh_millihz / 1000.0) << " Hz mode is restored with the replacement topology."sv;
        return;
      }
      if (current->refresh_millihz == selected.refresh_millihz) {
        return;
      }
      BOOST_LOG(info) << "Local monitor left its selected "sv << (selected.refresh_millihz / 1000.0) << " Hz for "sv
                      << (current->refresh_millihz / 1000.0) << " Hz while its source was attached; asking Windows for the selected rate again."sv;
      const auto status = VDISPLAY::setTargetDesktopRefreshRate(current->adapter_id, current->target_id, selected.refresh_millihz);
      if (status != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "Windows refused the monitor's selected refresh rate (status "sv << status << ")."sv;
        return;
      }
      for (int attempt = 0; attempt < 20 && !cancelled(); ++attempt) {
        std::this_thread::sleep_for(100ms);
        const auto settled = find_target(target_device_path_);
        if (settled && settled->refresh_millihz == selected.refresh_millihz) {
          BOOST_LOG(info) << "Local monitor is back at "sv << (settled->refresh_millihz / 1000.0) << " Hz."sv;
          return;
        }
      }
      BOOST_LOG(warning) << "The monitor did not return to its selected refresh rate."sv;
    }

    bool local_session_t::establish_native_source_mode(int requested_millihz, const std::function<bool()> &cancelled) {
      const bool established = establish_source_mode_with_io(
        source_width_,
        source_height_,
        [&]() {
          return refresh_virtual_display_reference(false);
        },
        [&](const resolved_virtual_display_t &source) {
          BOOST_LOG(info) << "Local monitor source bootstrap: observed="sv
                          << (source.rect.right - source.rect.left) << 'x'
                          << (source.rect.bottom - source.rect.top) << '@'
                          << (source.refresh_millihz / 1000.0) << " requested="sv
                          << source_width_ << 'x' << source_height_ << '@'
                          << (requested_millihz / 1000.0) << " on owned source "sv
                          << platf::to_utf8(source.device_path) << '.';
          VDISPLAY::changeDisplaySettings(source.gdi_name.c_str(), source_width_, source_height_, requested_millihz, false);
        },
        [](std::chrono::milliseconds delay) {
          std::this_thread::sleep_for(delay);
        },
        []() {
          return std::chrono::steady_clock::now();
        },
        cancelled
      );
      // Mode establishment cannot relax capture, HDR or presentation geometry verification.
      if (!established || cancelled()) {
        return false;
      }
      const auto verified = refresh_virtual_display_reference();
      return verified.has_value() && !cancelled();
    }

    bool local_session_t::configure_source_refresh(int requested_millihz, const std::function<bool()> &cancelled) {
      // A successful mode setter is not proof of the active cadence. Conversely, a rejected
      // setter may have applied its baseline mode. Re-resolve the stable source for every
      // attempt and accept only CCD readback, without recreating the user's retained desktop.
      const auto verified_millihz = configure_source_refresh_with_io(
        requested_millihz,
        [&]() {
          return refresh_virtual_display_reference();
        },
        [&](const resolved_virtual_display_t &source) {
          VDISPLAY::changeDisplaySettings(
            source.gdi_name.c_str(),
            source_width_,
            source_height_,
            requested_millihz,
            !exclusive_
          );
        },
        [](std::chrono::milliseconds delay) {
          std::this_thread::sleep_for(delay);
        },
        cancelled
      );
      if (cancelled()) {
        return false;
      }
      if (!verified_millihz) {
        BOOST_LOG(warning) << "The local AR source mode could not be verified; retaining the desktop for a later topology retry."sv;
        return false;
      }
      virtual_display_.update_mode(source_width_, source_height_, static_cast<std::uint32_t>(*verified_millihz));
      if (source_refresh_matches(*verified_millihz, requested_millihz)) {
        return true;
      }
      // Some drivers cannot apply a new fractional rate in place. Keep the usable desktop
      // after this bounded repair attempt; a permanent mismatch must not cause a blank retry
      // loop or migrate its windows. A later real mode change can try again.
      BOOST_LOG(warning) << "Local AR source refresh remains "sv << (*verified_millihz / 1000.0)
                         << " Hz after requesting "sv << (requested_millihz / 1000.0)
                         << " Hz; preserving the desktop with its verified source cadence."sv;
      return true;
    }

    bool local_session_t::final_presentation_contract_matches(const target_state_t &expected, const target_state_t &actual) const {
      // Positions and GDI names may change during final promotion; mode, adapter and color may
      // not. In particular, never start a presenter from the pre-transaction HDR observation.
      if (!presentation_target_contract_matches(expected, actual)) {
        return false;
      }
      const auto binding = virtual_display_binding();
      const auto source = binding.identity ? resolve_virtual_display(*binding.identity, binding.device_path, binding.display_name, nullptr, source_width_, source_height_) : std::nullopt;
      if (!source || source->refresh_millihz <= 0 || (monitor_surface_ && !source_refresh_matches(source->refresh_millihz, actual.refresh_millihz))) {
        return false;
      }
      BOOST_LOG(info) << "Local AR verified refresh: source="sv << (source->refresh_millihz / 1000.0)
                      << " Hz, glasses="sv << (actual.refresh_millihz / 1000.0) << " Hz."sv;
      const auto source_hdr = VDISPLAY::queryDisplayHDRByName(source->gdi_name.c_str());
      return source_hdr && *source_hdr == virtual_hdr_active_ && (!virtual_hdr_active_ || actual.hdr.active);
    }

    bool local_session_t::update_display_topology(const std::function<bool()> &update) {
      return exclusive_ ?
               platf::primary_display::run_local_exclusive_update(virtual_display_binding().device_path, update) :
               update();
    }

    std::optional<target_state_t> local_session_t::isolate_target() {
      // Exclusive mode has one topology/clip owner. The transaction wrapper verifies its final
      // topology after mode and color setters finish; it must not enter the row-isolation journal.
      return exclusive_ ? find_target(target_device_path_) :
                          isolate_physical_output(virtual_display_binding().display_name, target_device_path_, original_target_rect_);
    }

    void local_session_t::retire_exclusive_display() {
      if (!virtual_display_.prepared() && !virtual_display_.owns_display()) {
        return;
      }
      if (!virtual_display_.owns_display()) {
        if (desktop_handoff_) {
          desktop_handoff_->prepare_restore();
        }
        if (!virtual_display_.restore()) {
          BOOST_LOG(warning) << "Local AR startup recovery is pending; retaining the physical desktop journal."sv;
        }
        restore_monitor_desktop(desktop_handoff_, target_device_path_, original_target_rect_);
        return;
      }
      // Restore ordinary monitors before the source is removed. Both callbacks survive this
      // session object, including an unpublished Add or teardown deferred to a later handoff.
      if (!begin_local_virtual_display_retirement(std::move(virtual_display_), {
                                                                                 .prepare = [handoff = desktop_handoff_, path = target_device_path_, original = original_target_rect_](VDISPLAY::session_t &session) {
                                                                                   if (handoff) handoff->prepare_restore();
                                                                                   if (!session.restore()) return false;
                                                                                   return restore_monitor_desktop(handoff, path, original);
                                                                                 },
                                                                                 .finish = [](VDISPLAY::session_t &session) {
                                                                                   return session.restore();
                                                                                 },
                                                                               }) ||
          !wait_for_local_virtual_display_retirement_impl(3s)) {
        BOOST_LOG(warning) << "Local AR exclusive display restoration/removal is pending; retaining the cleanup barrier."sv;
      }
    }

    void local_session_t::cancel_monitor_source_switch() {
      if (!monitor_source_switch_) {
        return;
      }
      const auto request = monitor_source_switch_->request.exchange({});
      if (request) {
        using enum monitor_source_switch_t::state_e;
        auto state = request->state.load();
        while (state == requested || state == adopted) {
          if (request->state.compare_exchange_weak(state, cancelled)) {
            break;
          }
        }
      }
    }

    void local_session_t::stop_presenter() {
      presenter_.request_stop();
      cancel_monitor_source_switch();
      if (presenter_.joinable()) {
        presenter_.join();
      }
      monitor_source_switch_.reset();
      running_.store(false);
      if (monitor_surface_) {
        physical_monitor_active.store(false);
      }
    }

    std::optional<resolved_virtual_display_t> local_session_t::refresh_virtual_display_reference(bool require_geometry) {
      std::lock_guard lock(virtual_display_mutex_);
      auto binding = virtual_display_.binding();
      if (!binding.identity) {
        return std::nullopt;
      }
      const auto resolved = resolve_virtual_display(
        *binding.identity,
        binding.device_path,
        binding.display_name,
        nullptr,
        source_width_,
        source_height_,
        require_geometry
      );
      if (!resolved) {
        return std::nullopt;
      }
      if (!same_virtual_display_identity(*binding.identity, resolved->identity) || !equal_device_paths(binding.display_name, resolved->gdi_name)) {
        BOOST_LOG(info) << "Local AR virtual desktop was renumbered ["sv
                        << platf::to_utf8(binding.display_name) << " -> "sv
                        << platf::to_utf8(resolved->gdi_name) << ", target "sv
                        << binding.identity->TargetId << " -> "sv
                        << resolved->identity.TargetId << "]."sv;
      }
      binding.identity = resolved->identity;
      binding.display_name = resolved->gdi_name;
      binding.device_path = resolved->device_path;
      if (!virtual_display_.update_binding(std::move(binding))) {
        return std::nullopt;
      }
      virtual_display_absence_started_.reset();
      virtual_display_absence_observations_ = 0;
      return resolved;
    }

    VDISPLAY::creation_result_t local_session_t::virtual_display_binding() const {
      std::lock_guard lock(virtual_display_mutex_);
      return virtual_display_.binding();
    }

    bool local_session_t::start_presenter(const target_state_t &presentation_target) {
      stop_presenter();
      if (!live_target_) {
        live_target_ = std::make_shared<platf::dxgi::local_presenter_config_t::target_t>();
      }
      const auto source_binding = virtual_display_binding();
      auto source_display_name = platf::to_utf8(source_binding.display_name);
      const auto source_device_path = source_binding.device_path;
      if (source_display_name.empty() || source_device_path.empty() || presentation_target.device_path.empty()) {
        BOOST_LOG(error) << "Local AR presenter refused to open without stable source/sink identities."sv;
        failed_.store(true);
        running_.store(false);
        return false;
      }
      {
        std::lock_guard lock(live_target_->mutex);
        live_target_->rect = presentation_target.rect;
        live_target_->display_name = presentation_target.gdi_name;
        // Seed the presenter with the stable paths established by the controller before it opens
        // either volatile \\.\DISPLAYn name. Later generations may verify these paths but must
        // never learn a replacement identity from a recycled GDI name.
        if (live_target_->source_device_path.empty()) {
          live_target_->source_device_path = source_device_path;
        }
        if (live_target_->target_device_path.empty()) {
          live_target_->target_device_path = presentation_target.device_path;
        }
      }
      if (!exclusive_ && !cursor_clip_) {
        cursor_clip_ = std::make_shared<platf::dxgi::local_presenter_cursor_clip_t>();
      }

      platf::dxgi::local_presenter_config_t presenter_config;
      presenter_config.source_display_name = std::move(source_display_name);
      presenter_config.target_adapter_id = presentation_target.adapter_id;
      presenter_config.target_refresh_millihz = presentation_target.refresh_millihz;
      presenter_config.frame_sequential = monitor_surface_;
      presenter_config.target_desktop_refresh_rate =
        desktop_refresh_rate_keeper_.has_value() && VDISPLAY::targetUsesDesktopRefreshRate(presentation_target.adapter_id, presentation_target.target_id);
      presenter_config.monitor_config = config::video.local_monitor;
      if (monitor_surface_) {
        presenter_config.shutter_recovery_allowed = [target = live_target_]() {
          RECT rect {};
          {
            std::lock_guard lock(target->mutex);
            rect = target->rect;
          }
          return monitor_requested() && query_monitor_desktop_block_reason(
            desktop_object_name(GetThreadDesktop(GetCurrentThreadId())), rect
          ).empty();
        };
        presenter_config.on_failure = [](std::string reason) {
          std::lock_guard lock(physical_monitor_error_mutex);
          if (physical_monitor_requested.load()) {
            physical_monitor_error = "Monitor 3D stopped: " + std::move(reason) + ". Check the emitter connection and fixed display refresh rate, then start 3D again.";
          }
        };
      }
      presenter_config.hdr = virtual_hdr_active_;
      presenter_config.sbs_mode = presentation_target.mode == presentation_mode_e::sbs_ai ?
                                    ::video::SBS_AI :
                                    ::video::SBS_OFF;
      if (monitor_surface_) {
        presenter_config.sbs_mode = presentation_target.mode == presentation_mode_e::monitor_game ? ::video::SBS_GAME_SBS : ::video::SBS_AI;
        monitor_source_switch_ = std::make_shared<monitor_source_switch_t>();
        monitor_source_switch_->presented_mode.store(presenter_config.sbs_mode);
        monitor_source_switch_->mode_is_requested = [](int mode) {
          return monitor_requested() &&
                 mode == (monitor_uses_game_source() ? ::video::SBS_GAME_SBS : ::video::SBS_AI);
        };
        presenter_config.source_switch = monitor_source_switch_;
      }
      presenter_config.sbs_config = config::video.sbs;
      // The saved choice only picks the source. A game's own stereo takes the same Game 3D
      // route on glasses as on the shutter monitor and in a remote stream.
      if (!monitor_surface_ && presentation_target.game_source && presenter_config.sbs_mode == ::video::SBS_AI) {
        presenter_config.sbs_mode = ::video::SBS_GAME_SBS;
      }
      presenter_config.capture_failover = capture_failover_;
      presenter_config.live_target = live_target_;
      presented_frames_ = std::make_shared<std::atomic<std::uint64_t>>(0);
      presenter_config.presented_frames = presented_frames_;
      presenter_config.cursor_clip = cursor_clip_;
      presenter_quiesced_ = std::make_shared<safe::event_t<bool>>();
      presenter_config.on_quiesced = [quiesced = presenter_quiesced_]() {
        quiesced->raise(true);
      };

      failed_.store(false);
      running_.store(true);
      presenter_ = std::jthread([this, presenter_config, quiesced = presenter_quiesced_](std::stop_token stop_token) mutable {
        // Match the remote conversion owner and keep its process scheduling lease across
        // capture reinits, instead of restoring and reapplying it for every DXGI attempt.
        auto presentation_scheduling = platf::acquire_presentation_scheduling();
        // Also acknowledge an early initialization failure or an exhausted reinit loop.
        // This guard runs before scheduling-lease cleanup and never captures local_session_t.
        auto notify_quiesced = util::fail_guard(presenter_config.on_quiesced);
        platf::adjust_thread_priority(platf::thread_priority_e::high);
        auto reinit_window_started = std::chrono::steady_clock::now();
        int consecutive_reinits = 0;
        bool reinitializing = false;
        while (true) {
          // Clear the previous attempt's proof before checking stop. A controller that already
          // observed that proof has requested stop, so no successor attempt can touch topology.
          quiesced->reset();
          if (stop_token.stop_requested()) {
            break;
          }
          if (reinitializing) {
            std::this_thread::sleep_for(100ms);
            if (stop_token.stop_requested()) {
              break;
            }
            if (const auto refreshed = refresh_virtual_display_reference()) {
              presenter_config.source_display_name = platf::to_utf8(refreshed->gdi_name);
            } else {
              BOOST_LOG(error) << "Local AR virtual desktop could not be resolved; pausing for a controller retry."sv;
              break;
            }
            if (presenter_config.source_switch) {
              presenter_config.sbs_mode = presenter_config.source_switch->presented_mode.load();
            }
          }
          // Stability is per continuous presenter attempt. Do not let several short failed
          // attempts accumulate enough frames to be mistaken for one stable session.
          presenter_config.presented_frames->store(0, std::memory_order_relaxed);
          const auto presented_before = presenter_config.presented_frames->load(std::memory_order_relaxed);
          const auto result = platf::dxgi::run_local_presenter(presenter_config, stop_token);
          if (result != platf::dxgi::local_presenter_result_e::reinit) {
            break;
          }

          const auto now = std::chrono::steady_clock::now();
          const auto presented_after = presenter_config.presented_frames->load(std::memory_order_relaxed);
          if (presented_after - presented_before >= 60) {
            // Reset only after the attempt demonstrably presented a stable run of frames. Slow
            // initialization failures no longer evade the retry breaker merely by taking time.
            consecutive_reinits = 0;
          }
          if (consecutive_reinits == 0) {
            // The window measures the current run of failed attempts, from its first failure.
            // Measured from thread start, one benign reinit (a stale output, a size change)
            // after ten quiet seconds would already count as "did not recover".
            reinit_window_started = now;
          }
          ++consecutive_reinits;
          if (consecutive_reinits >= 12 || now - reinit_window_started >= 10s) {
            BOOST_LOG(error) << "Local AR presenter did not recover after "sv
                             << consecutive_reinits << " reinitializations; pausing for a controller retry."sv;
            break;
          }

          reinitializing = true;
        }
        // Any exit not requested by the topology controller should be retried, including a
        // user-closed or driver-closed presenter window that otherwise exits cleanly.
        failed_.store(!stop_token.stop_requested());
        running_.store(false);
        if (monitor_surface_) {
          physical_monitor_active.store(false);
          // A lost emitter or unverifiable scanout restores an ordinary desktop instead of
          // repeatedly running a full-screen output without a valid shutter phase.
          if (!stop_token.stop_requested()) {
            {
              std::lock_guard lock(physical_monitor_error_mutex);
              if (physical_monitor_error.empty()) {
                physical_monitor_error = "Monitor 3D could not establish a stable display session. Check the display and emitter settings, then start 3D again.";
              }
            }
            physical_monitor_requested.store(false);
          }
        }
      });
      return true;
    }
  }  // namespace internal

#ifdef SUNSHINE_TESTS
  namespace {
    target_state_t make_test_target(const detail::local_session_contract_t &contract) {
      target_state_t target;
      target.device_path = contract.device_path;
      target.adapter_id = contract.adapter_id;
      target.mode = contract.mode;
      target.hdr.known = contract.hdr_known;
      target.hdr.supported = contract.hdr_supported;
      target.hdr.active = contract.hdr_active;
      target.hdr.limited_by_policy = contract.hdr_limited_by_policy;
      target.is_primary = contract.is_primary;
      target.is_cloned = contract.is_cloned;
      target.rect = contract.rect;
      target.refresh_millihz = contract.refresh_millihz;
      target.source_adapter_id = contract.source_adapter_id;
      target.source_id = contract.source_id;
      target.target_id = contract.target_id;
      target.gdi_name = contract.gdi_name;
      target.desktop_topology = contract.desktop_topology;
      target.hdr.user_enabled = contract.hdr_user_enabled;
      target.hdr.bits_per_color = contract.hdr_bits_per_color;
      target.game_source = contract.game_source;
      return target;
    }
  }  // namespace

  bool detail::local_session_can_reconfigure_for_test(
    const local_session_contract_t &before,
    const local_session_contract_t &after,
    bool exclusive
  ) {
    return can_reconfigure_local_session(make_test_target(before), make_test_target(after), exclusive);
  }

  bool detail::local_monitor_source_only_change_for_test(
    const local_session_contract_t &before,
    const local_session_contract_t &after
  ) {
    return monitor_source_only_change(make_test_target(before), make_test_target(after));
  }

  bool detail::verify_monitor_source_request_for_test(
    const std::function<bool()> &current,
    const std::function<bool()> &verify
  ) {
    return verify_monitor_source_request(current, verify);
  }

  bool detail::local_presenter_target_contract_matches_for_test(
    const local_session_contract_t &expected,
    const local_session_contract_t &actual
  ) {
    return presentation_target_contract_matches(make_test_target(expected), make_test_target(actual));
  }

  bool detail::local_session_can_re_isolate_for_test(
    const local_session_contract_t &before,
    const local_session_contract_t &after,
    bool transition_presenter_paused
  ) {
    return can_re_isolate_local_session(make_test_target(before), make_test_target(after), transition_presenter_paused);
  }

  detail::local_pause_order_result_t detail::local_pause_order_for_test() {
    local_pause_order_result_t result;
    const bool previous_exclusive = config::video.local_ar_virtual_display_only;
    config::video.local_ar_virtual_display_only = true;
    auto restore_config = util::fail_guard([&]() {
      config::video.local_ar_virtual_display_only = previous_exclusive;
    });
    std::atomic<bool> capture_stopped {false};
    std::atomic<bool> cleanup_done {false};
    std::atomic<bool> cleanup_saw_restored {false};
    std::promise<void> desktop_restored;
    const auto restored = desktop_restored.get_future();
    const auto retained_marker = std::make_shared<int>(0);
    const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT identity {{42, 0}, 7};
    const std::wstring device_path = L"test-local-pause-source";
    local_session_t session(gpu_workload::lease_t {}, []() { return true; });
    auto discard_test_owner = util::fail_guard([&]() {
      // Keep failure cleanup deterministic even if suspend stops joining. This fake device must
      // never reach the production local-session destructor's retirement adapter.
      session.stop_presenter();
      auto discarded_owner = std::move(session.virtual_display_);
    });
    session.target_device_path_ = L"test-local-pause-sink";
    session.virtual_display_ = VDISPLAY::session_t({
      .pause = [&](std::wstring_view path, auto &retained, std::wstring_view sink) {
        ++result.pause_calls;
        result.pause_saw_capture_stopped = capture_stopped.load();
        result.pause_preceded_cleanup = !cleanup_done.load();
        if (path != device_path || sink != session.target_device_path_) {
          return false;
        }
        retained = platf::primary_display::retained_display_ptr(
          retained_marker,
          reinterpret_cast<platf::primary_display::retained_display_t *>(retained_marker.get())
        );
        desktop_restored.set_value();
        return true;
      },
      .restore_cursor = [](const auto &) {},
    });
    session.virtual_display_.adopt(
      {.exclusive = true, .local_sink = session.target_device_path_},
      {.display_name = L"test-local-pause-display", .device_path = device_path, .identity = identity}
    );
    const auto generation = session.virtual_display_.generation();
    session.presenter_quiesced_ = std::make_shared<safe::event_t<bool>>();
    session.presenter_ = std::jthread([&](std::stop_token stop) {
      std::mutex mutex;
      std::condition_variable_any changed;
      std::unique_lock lock(mutex);
      changed.wait(lock, stop, []() { return false; });
      capture_stopped.store(true);
      session.presenter_quiesced_->raise(true);
      // A regression that joins before restoring fails after one second instead of deadlocking
      // the test executable. This models cleanup that cannot finish until DWM restores outputs.
      cleanup_saw_restored.store(restored.wait_for(1s) == std::future_status::ready);
      cleanup_done.store(true);
    });

    result.pause_completed = session.suspend_for_resume();
    result.cleanup_saw_desktop_restored = cleanup_saw_restored.load();
    result.cleanup_joined = cleanup_done.load() && !session.presenter_.joinable();
    const auto &binding = session.virtual_display_.binding();
    result.identity_retained = session.virtual_display_.owns_display() && session.virtual_display_.has_retained() &&
                               session.virtual_display_.generation() == generation && binding.identity &&
                               same_virtual_display_identity(*binding.identity, identity) && binding.device_path == device_path;
    return result;
  }
#endif
}  // namespace ar_glasses
