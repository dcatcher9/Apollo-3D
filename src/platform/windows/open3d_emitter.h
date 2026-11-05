#pragma once

#include <array>
#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace platf::open3d {
  // Interoperability limits for the emitter's 16 MHz ATmega32U4 firmware: Timer1 uses
  // half-microsecond ticks; Timer4 has a 1023-tick top at one microsecond per tick.
  namespace limits {
    inline constexpr int minimum_timing_us = 1;
    inline constexpr int timer1_wrap_us = 32768;
    inline constexpr int maximum_timer1_interval_us = timer1_wrap_us - 1;
    // Timer4 refuses endpoints within 5 us of its current counter. Use the emitter project's
    // 30-us working default as a conservative floor above its documented 17-us scheduling cost.
    inline constexpr int minimum_signal_spacing_us = 30;
    inline constexpr int maximum_signal_spacing_us = 1023;
    inline constexpr int scheduling_margin_us = 100;
    inline constexpr std::array supported_protocols {2, 4, 5, 6};
    // After PC serial/averaging-off readback, only old comparator work remains. An OPEN and
    // CLOSE each need at most one Timer1 wrap. The firmware's four-entry IR queue advances on
    // every Timer4 overflow (at most 1024 us/token); the 35 ms margin exceeds that 4.096 ms tail.
    inline constexpr auto ir_drain_margin = std::chrono::milliseconds {35};
    inline constexpr auto startup_settle = std::chrono::microseconds {2 * timer1_wrap_us} + ir_drain_margin;

    constexpr bool finite_close_protocol(int protocol) {
      for (const auto supported : supported_protocols) {
        if (protocol == supported) {
          return true;
        }
      }
      return false;
    }
  }  // namespace limits

  struct timing_limits_t {
    int protocol;
    int min_delay_us;
    int max_delay_us;
    int min_duration_base_us;  // Add configured signal spacing for the minimum duration.
    int min_duration_us;
    int max_duration_us;
    int min_signal_spacing_us;
    int max_signal_spacing_us;
    int max_window_us;  // Inclusive maximum of delay + duration.
    int min_period_tail_base_us;  // Eye period must include duration + spacing + this tail.
  };

  /** Protocol facts are token lengths, not copied firmware implementation. Timer1's mode-1
   * compensation advances its comparator by half the OPEN token length; the CLOSE interval
   * additionally subtracts the OPEN token length. The scheduling margin and current-eye queue
   * tail are conservative host policies. Timer4's 1023-us interval includes the whole IR token.
   */
  constexpr std::optional<timing_limits_t> timing_limits(int protocol, int signal_spacing_us) {
    int longest_open_us = 0;
    int longest_close_us = 0;
    int longest_token_us = 0;
    bool end_aligned_open = false;
    switch (protocol) {
      case 2:
        longest_open_us = 100;
        longest_close_us = longest_token_us = 141;
        end_aligned_open = true;
        break;
      case 4:
        longest_open_us = longest_token_us = 540;
        longest_close_us = 460;
        break;
      case 5:
      case 6:
        longest_open_us = longest_close_us = longest_token_us = 220;
        break;
      default:
        return std::nullopt;
    }
    const int maximum_spacing = limits::maximum_signal_spacing_us - longest_token_us;
    if (signal_spacing_us < limits::minimum_signal_spacing_us || signal_spacing_us > maximum_spacing) {
      return std::nullopt;
    }
    const int duration_base = (end_aligned_open ? 2 * longest_open_us : longest_open_us) + limits::scheduling_margin_us;
    return timing_limits_t {
      protocol,
      limits::scheduling_margin_us + (end_aligned_open ? (longest_open_us + 1) / 2 : 0),
      limits::maximum_timer1_interval_us,
      duration_base,
      duration_base + signal_spacing_us,
      limits::maximum_timer1_interval_us,
      limits::minimum_signal_spacing_us,
      maximum_spacing,
      limits::maximum_timer1_interval_us,
      longest_close_us + limits::scheduling_margin_us,
    };
  }

  constexpr std::optional<int> eye_period_us(int refresh_millihz, int eye_repeats) {
    if (refresh_millihz <= 0 || eye_repeats < 1 || eye_repeats > 4) {
      return std::nullopt;
    }
    const auto period = (1000000000LL * eye_repeats + refresh_millihz / 2) / refresh_millihz;
    if (period <= 0 || period > std::numeric_limits<int>::max()) {
      return std::nullopt;
    }
    return static_cast<int>(period);
  }

  struct settings_t {
    std::string port;
    int protocol = 6;
    int frame_delay_us = 500;
    int frame_duration_us = 2000;
    int signal_spacing_us = 30;
    bool flip_eyes = false;
    int target_frametime_us = 0;  // Host eye interval only; the disabled firmware averaging filter receives zero.
  };

  namespace detail {
    struct parameters_t {
      int firmware_version = 0;
      // Positional firmware fields following the version. Retain additional fields from newer
      // firmware instead of overwriting them with values from a hard-coded profile.
      std::vector<int> values;
    };

    std::optional<std::string> normalize_port(std::string_view port);
    bool valid_settings(const settings_t &settings);
    // Monitor presentation additionally needs a known eye interval and positive remaining
    // host-feedback time after delay, duration, spacing and the protocol close margin.
    std::optional<int> host_feedback_budget_us(const settings_t &settings);
    std::optional<parameters_t> parse_parameters(std::string_view line);
    std::optional<std::string> quiesce_command(const parameters_t &current);
    bool quiesce_matches(const parameters_t &actual, const parameters_t &previous);
    std::optional<std::string> configuration_command(const parameters_t &current, const settings_t &settings);
    bool configuration_matches(const parameters_t &actual, const parameters_t &previous, const settings_t &settings);

    // The query is a FIFO firmware-processing barrier behind the last complete eye command.
    // It must only read parameters; it may not purge, reopen, or change the current profile.
    struct quiesce_io_t {
      using clock = std::chrono::steady_clock;
      std::function<clock::time_point()> now;
      std::function<bool()> cancelled;
      std::function<bool(clock::time_point)> finish_signal;
      std::function<std::optional<parameters_t>(clock::time_point)> read_parameters;
      std::function<bool(clock::time_point)> wait_until;
    };

    bool quiesce_with_io(const parameters_t &expected, const quiesce_io_t &io);
  }  // namespace detail

  /** An explicitly selected Open3DOLED COM port. Configuration is volatile, never saved to EEPROM.
   * connect() performs bounded read/apply/readback transactions and a cancellable firmware drain
   * period before allowing eye signals. Only protocols with separate OPEN/CLOSE tokens are used.
   * signal() never waits: at most one overlapped write can be outstanding. A busy or failed write
   * invalidates the connection so presentation cannot silently keep running out of sync.
   * quiesce() stops new signals, finishes the last write, verifies an unchanged firmware profile
   * behind it, and waits for its finite IR work to expire. It keeps the same COM handle and is
   * cancellable; a failed/cancelled barrier closes the connection and cannot authorize a restart.
   * close() cancels outstanding serial I/O and releases the port. The presentation owner calls
   * connect/quiesce/signal; a control thread may close/cancel that owner during teardown.
   */
  class emitter_t {
  public:
    emitter_t();
    ~emitter_t();
    emitter_t(const emitter_t &) = delete;
    emitter_t &operator=(const emitter_t &) = delete;

    bool connect(const settings_t &settings, std::stop_token stop = {});
    bool quiesce(std::stop_token stop = {});
    bool signal(bool left_eye);
    bool healthy() const;
    void close();

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };

  std::vector<std::string> serial_ports();
}  // namespace platf::open3d
