#pragma once

#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace platf::open3d {
  struct settings_t {
    std::string port;
    int protocol = 6;
    int frame_delay_us = 500;
    int frame_duration_us = 2000;
    int signal_spacing_us = 30;
    bool flip_eyes = false;
    int target_frametime_us = 0;
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
    std::optional<parameters_t> parse_parameters(std::string_view line);
    std::optional<std::string> configuration_command(const parameters_t &current, const settings_t &settings);
    bool configuration_matches(const parameters_t &actual, const parameters_t &previous, const settings_t &settings);
  }  // namespace detail

  /** An explicitly selected Open3DOLED COM port. Configuration is volatile, never saved to EEPROM.
   * connect() performs bounded read/apply/readback transactions before allowing eye signals.
   * signal() never waits: at most one overlapped write can be outstanding. A busy or failed write
   * invalidates the connection so presentation cannot silently keep running out of sync.
   * close() cancels outstanding serial I/O and releases the port. connect/close are control-thread
   * operations; the presentation thread owns signal().
   */
  class emitter_t {
  public:
    emitter_t();
    ~emitter_t();
    emitter_t(const emitter_t &) = delete;
    emitter_t &operator=(const emitter_t &) = delete;

    bool connect(const settings_t &settings, std::stop_token stop = {});
    bool signal(bool left_eye);
    bool healthy() const;
    void close();

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl_;
  };

  std::vector<std::string> serial_ports();
}  // namespace platf::open3d
