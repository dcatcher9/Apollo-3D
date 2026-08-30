#pragma once

#include <memory>
#include <string_view>

#ifdef SUNSHINE_TESTS
  #include <chrono>
  #include <cstdint>
  #include <optional>
  #include <span>
#endif

namespace ar_glasses::rayneo {
  enum class wear_state_e {
    unknown,
    off_head,
    worn,
  };

  /** Monitor the supported RayNeo vendor HID collection for an authoritative wear state. */
  class wear_monitor_t {
  public:
    wear_monitor_t();
    ~wear_monitor_t();

    wear_monitor_t(const wear_monitor_t &) = delete;
    wear_monitor_t &operator=(const wear_monitor_t &) = delete;
    wear_monitor_t(wear_monitor_t &&) = delete;
    wear_monitor_t &operator=(wear_monitor_t &&) = delete;

    [[nodiscard]] wear_state_e state() const noexcept;

  private:
    class impl_t;
    std::unique_ptr<impl_t> impl_;
  };

  /** Return a retrying monitor only for the exact display model implementing this protocol. */
  std::unique_ptr<wear_monitor_t> create_wear_monitor(std::string_view display_model_id);

#ifdef SUNSHINE_TESTS
  namespace detail {
    enum class power_event_e {
      none,
      suspend,
      resume,
    };

    struct parsed_sensor_report_t {
      std::uint32_t tick = 0;
      float proximity = 0.0f;
      wear_state_e state = wear_state_e::unknown;
    };

    struct debounce_observation_t {
      std::chrono::milliseconds at {};
      std::uint32_t tick = 0;
      wear_state_e state = wear_state_e::unknown;
    };

    /** Parse one complete Windows HID report through the production decoder. */
    std::optional<parsed_sensor_report_t> parse_sensor_report_for_test(
      std::span<const std::uint8_t> report
    );

    /** Authenticate the exact supported device-info acknowledgement and board ID. */
    bool board_info_matches_for_test(std::span<const std::uint8_t> report);

    /** Match only this device's HID interface path, independent of path casing/suffixes. */
    bool hid_interface_matches_for_test(std::wstring_view path);

    /** Classify one Windows power callback through the production lifecycle mapping. */
    power_event_e power_event_for_test(std::uint32_t notification);

    /** Return whether this power callback should reopen the HID subscription. */
    bool power_event_requests_recovery_for_test(
      std::uint32_t notification,
      bool was_suspended
    );

    /** Return the production reconnect delay for deterministic recovery-policy coverage. */
    std::chrono::milliseconds recovery_retry_delay_for_test(
      bool device_unavailable,
      bool notifications_available,
      unsigned consecutive_failures
    );

    /** Replay timestamped classified samples through the production debounce/stale policy. */
    wear_state_e debounce_observations_for_test(
      std::span<const debounce_observation_t> observations,
      std::chrono::milliseconds evaluated_at
    );
  }  // namespace detail
#endif
}  // namespace ar_glasses::rayneo
