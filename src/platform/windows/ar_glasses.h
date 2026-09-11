#pragma once

#include "rayneo_wear_monitor.h"
#include "src/platform/common.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
  #include <windows.h>
#endif

namespace ar_glasses {
  using remote_virtual_display_lease_t = std::uint64_t;

  enum class presentation_mode_e {
    unsupported,
    normal,
    sbs_ai,
  };

  enum class device_decision_e {
    pending,
    approved,
    rejected,
  };

  struct device_info_t {
    std::string id;
    std::string name;
    device_decision_e decision = device_decision_e::pending;
    bool connected = false;
    bool auto_detected = false;
  };

  constexpr presentation_mode_e classify_mode(int width, int height) {
    if (height != 1080) {
      return presentation_mode_e::unsupported;
    }
    if (width == 1920) {
      return presentation_mode_e::normal;
    }
    if (width == 3840) {
      return presentation_mode_e::sbs_ai;
    }
    return presentation_mode_e::unsupported;
  }

  static_assert(classify_mode(1920, 1080) == presentation_mode_e::normal);
  static_assert(classify_mode(3840, 1080) == presentation_mode_e::sbs_ai);
  static_assert(classify_mode(3840, 2160) == presentation_mode_e::unsupported);

  namespace detail {
    enum class local_session_activation_e {
      waiting_for_sensor,
      connection,
      worn,
      off_head,
    };

    /** Wear authority belongs to the connected glasses, not the virtual desktop it controls. */
    class local_session_activation_t {
    public:
      using clock_t = std::chrono::steady_clock;
      static constexpr auto initial_probe_grace = std::chrono::seconds {3};

      local_session_activation_e observe(
        std::optional<rayneo::wear_snapshot_t> sensor,
        clock_t::time_point now,
        rayneo::wear_state_e remembered_state = rayneo::wear_state_e::unknown
      ) {
        if (!sensor) {
          return local_session_activation_e::connection;
        }
        if (!probe_started_) {
          probe_started_ = now;
        }
        if (remembered_state != rayneo::wear_state_e::unknown) {
          // The controller can spend seconds inside Windows topology changes. The HID worker
          // preserves the latest confirmed state even if its live sample changes again meanwhile.
          last_confirmed_ = remembered_state;
        } else if (sensor->availability == rayneo::wear_availability_e::available && sensor->state != rayneo::wear_state_e::unknown) {
          last_confirmed_ = sensor->state;
        }
        // Once the sensor has worked, gaps in its transport or ambiguous samples must not undo
        // an off-head exit or interrupt a worn session. Only unplug/target replacement resets it.
        if (last_confirmed_) {
          return *last_confirmed_ == rayneo::wear_state_e::worn ?
                   local_session_activation_e::worn :
                   local_session_activation_e::off_head;
        }
        return now - *probe_started_ < initial_probe_grace ?
                 local_session_activation_e::waiting_for_sensor :
                 local_session_activation_e::connection;
      }

    private:
      std::optional<clock_t::time_point> probe_started_;
      std::optional<rayneo::wear_state_e> last_confirmed_;
    };

    constexpr bool local_session_requested(local_session_activation_e activation) {
      return activation == local_session_activation_e::connection ||
             activation == local_session_activation_e::worn;
    }

    /** A fresh wearable/connection edge may replace retention; polling an old request may not. */
    class local_session_request_t {
    public:
      /** Call only for authoritative display observations. A live USB subscription identifies
       * a DP-only mode gap; an actual disconnect produces one fresh request on reappearance.
       */
      void observe_connection(bool display_present, bool transport_available) {
        if (!display_present && !transport_available) {
          reconnect_pending_ = true;
        } else if (display_present && reconnect_pending_) {
          reconnect_pending_ = false;
          requested_ = false;
          fresh_ = false;
        }
      }

      void observe(bool requested) {
        if (requested && !requested_) {
          fresh_ = true;
        }
        requested_ = requested;
        if (!requested) {
          fresh_ = false;
        }
      }

      void reset() {
        requested_ = false;
        fresh_ = false;
        reconnect_pending_ = false;
      }

      void consume() {
        fresh_ = false;
      }

      [[nodiscard]] bool fresh() const {
        return fresh_;
      }

    private:
      bool requested_ = false;
      bool fresh_ = false;
      bool reconnect_pending_ = false;
    };

    /** Remote ownership lasts for the configured client-connect window plus scheduling grace. */
    constexpr std::chrono::milliseconds remote_pending_duration(
      std::chrono::milliseconds connect_timeout
    ) {
      return std::max(connect_timeout, std::chrono::milliseconds::zero()) +
             std::chrono::milliseconds {2000};
    }

#ifdef SUNSHINE_TESTS
    /** Primary evidence is authoritative only when it names an active CCD source. */
    bool primary_source_is_authoritative_for_test(
      std::wstring_view primary_gdi_name,
      const std::vector<std::wstring> &active_gdi_names
    );

    /** Decode active source refresh without discarding fractional-Hz CCD evidence. */
    int local_source_refresh_millihz_for_test(std::uint32_t numerator, std::uint32_t denominator);

    /** Run the production bounded refresh repair against fake display I/O. The result retains
     * measured cadence, including terminal fallback; unavailable/cancelled work returns nullopt.
     */
    std::optional<int> configure_local_source_refresh_for_test(
      int requested_millihz,
      const std::function<std::optional<int>()> &query,
      const std::function<void()> &apply,
      const std::function<void(std::chrono::milliseconds)> &wait,
      const std::function<bool()> &cancelled
    );

    struct linear_layout_t {
      RECT virtual_rect {};
      RECT physical_rect {};
    };

    struct anchor_candidate_t {
      RECT rect {};
      LUID source_adapter_id {};
      UINT32 source_id = 0;
    };

    struct topology_path_identity_t {
      LUID source_adapter_id {};
      UINT32 source_id = 0;
      LUID target_adapter_id {};
      UINT32 target_id = 0;
    };

    /** Validate that the physical AR output stayed on its original adapter.
     *
     * The SudoVDA identity is intentionally supplied only to prove that it is not used as the
     * physical-output identity. SudoVDA may expose a different adapter LUID from its render GPU.
     */
    bool physical_adapter_contract_valid_for_test(
      const LUID &physical_before,
      const LUID &physical_after,
      const LUID &virtual_adapter
    );

    /** Compute the deterministic source/sink row anchored after the rightmost interactive output. */
    linear_layout_t compute_linear_layout_for_test(
      const RECT &anchor,
      LONG virtual_width,
      LONG virtual_height,
      LONG physical_width,
      LONG physical_height
    );

    /** Select a rightmost anchor with a total ordering independent of CCD path order. */
    std::optional<anchor_candidate_t> select_anchor_for_test(
      const std::vector<anchor_candidate_t> &candidates
    );

    /** Return whether the selected target shares its desktop source with another target. */
    bool source_is_cloned_for_test(
      const std::vector<topology_path_identity_t> &paths,
      std::size_t selected_index
    );

    /** Validate exact planned source/sink rectangles and their shared full-height edge. */
    bool isolated_layout_matches_for_test(
      const linear_layout_t &expected,
      const RECT &virtual_rect,
      const RECT &physical_rect
    );

    struct topology_recovery_parse_result_t {
      bool valid = false;
      std::size_t record_count = 0;
      std::string normalized_json;
    };

    /** Parse current recovery JSON through the production codec without touching disk. */
    topology_recovery_parse_result_t parse_topology_recovery_json_for_test(std::string_view contents);

    /** Classify whether a single persisted recovery record owns the supplied current rectangle. */
    bool topology_recovery_should_restore_for_test(
      std::string_view contents,
      const RECT &current_rect
    );

    struct local_session_contract_t {
      std::wstring device_path;
      LUID adapter_id {};
      presentation_mode_e mode = presentation_mode_e::unsupported;
      bool hdr_known = false;
      bool hdr_supported = false;
      bool hdr_active = false;
      bool hdr_limited_by_policy = false;
      bool is_primary = false;
      bool is_cloned = false;
      RECT rect {};
      int refresh_millihz = 60000;
    };

    /** Return whether a mode transition may retain the existing local SudoVDA desktop.
     * Exclusive sessions may repair a transiently primary sink; extended sessions may not.
     */
    bool local_session_can_reconfigure_for_test(
      const local_session_contract_t &before,
      const local_session_contract_t &after,
      bool exclusive
    );

    /** Reject a new presentation generation after a placement-only topology update. */
    bool local_presenter_target_contract_matches_for_test(
      const local_session_contract_t &expected,
      const local_session_contract_t &actual
    );

    /** Choose placement-only recovery only before a failed validation pauses the presenter. */
    bool local_session_can_re_isolate_for_test(
      const local_session_contract_t &before,
      const local_session_contract_t &after,
      bool transition_presenter_paused
    );

    struct virtual_display_identity_contract_t {
      LUID adapter_id {};
      UINT32 target_id = 0;
      std::wstring device_path;
      std::wstring gdi_name;
      std::wstring friendly_name;
    };

    /** Match a retiring Apollo source without trusting a possibly reused numeric target ID. */
    bool retirement_identity_matches_for_test(
      const virtual_display_identity_contract_t &retiring,
      const virtual_display_identity_contract_t &observed
    );

    /** Rebase one recovery record to a new physical mode without touching persistent state. */
    std::optional<std::string> rebase_topology_recovery_json_for_test(
      std::string_view contents,
      const RECT &previous_original_rect,
      const RECT &current_rect
    );

    /** Apply the remote exclusive-display preservation policy without querying Windows. */
    bool preserve_during_remote_virtual_display_for_test(
      std::string_view model_id,
      std::string_view friendly_name,
      const std::vector<device_info_t> &decisions
    );
#endif
  }  // namespace detail

  /** Return whether a monitor model/name is specific enough to identify AR glasses automatically. */
  bool is_recognized_ar_display(std::string_view model_id, std::string_view friendly_name);

  /** Snapshot the persisted monitor decisions and their current connection state. */
  std::vector<device_info_t> devices();

  /** Approve or reject a discovered monitor model. Pending is not accepted from the UI. */
  bool set_device_decision(std::string_view id, device_decision_e decision);

  /** Return whether an active CCD target should remain enabled during remote exclusive streaming. */
  bool preserve_during_remote_virtual_display(const LUID &adapter_id, UINT32 target_id);

  /** Load persisted preservation decisions without starting the controller or writing config. */
  void load_preservation_policy();

  /** Atomically write a general configuration snapshot while injecting the latest device list. */
  bool write_config_with_devices(std::string_view contents);

  /** Monitor approved AR displays and own the local virtual-desktop presentation lifecycle. */
  std::unique_ptr<platf::deinit_t> init();

  /** Atomically claim local construction while proc_t holds the process/session lock. */
  bool try_claim_local_virtual_display(const std::stop_source &construction_stop);

  /** Release a local construction claim after a handoff fails or the local session retires. */
  void release_local_virtual_display_claim();

  /** Wait until any asynchronously retired local SudoVDA output leaves Windows topology. */
  bool wait_for_local_virtual_display_retirement(std::chrono::milliseconds timeout);

  /** Reserve virtual-display ownership for a remote launch and synchronously stop local AR. */
  bool remote_virtual_display_starting(
    remote_virtual_display_lease_t lease,
    std::chrono::milliseconds connect_timeout,
    bool setup_in_progress = false
  );

  /** Renew the client-connect window after a potentially slow remote launch has completed. */
  void remote_virtual_display_awaiting_client(
    remote_virtual_display_lease_t lease,
    std::chrono::milliseconds connect_timeout
  );

  /** Mark the reserved remote virtual display as actively streamed. */
  bool remote_virtual_display_active(remote_virtual_display_lease_t lease);

  /** Release remote ownership after pause, termination, or launch failure. */
  void remote_virtual_display_ended(remote_virtual_display_lease_t lease);

  /** Return whether an active or connecting remote virtual display currently owns presentation. */
  bool remote_virtual_display_blocks_local();
}  // namespace ar_glasses
