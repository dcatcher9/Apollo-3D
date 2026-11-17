#include "open3d_emitter.h"

#include "src/logging.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include <windows.h>

// Protocol interoperability reference: Open3DOLED MPCVideoRenderer, PageFlip.cpp (GPLv3+),
// commit 26596cb1a12eb3b3f7e7152d672a4bfa4f4ebf93. This implementation owns its bounded Win32
// transport and volatile configuration transaction; no Python player or emitter firmware is copied.
namespace platf::open3d {
  using namespace std::chrono_literals;

  namespace {
    constexpr std::size_t minimum_parameter_count = 17;
    constexpr std::size_t maximum_parameter_count = 64;
    constexpr auto transaction_timeout = 1500ms;

    std::optional<int> parse_integer(std::string_view value) {
      int result = 0;
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
      if (value.empty() || parsed.ec != std::errc {} || parsed.ptr != value.data() + value.size() || result < 0) {
        return std::nullopt;
      }
      return result;
    }

    bool supported_parameters(const detail::parameters_t &parameters) {
      return parameters.firmware_version >= 20 &&
             parameters.values.size() >= minimum_parameter_count &&
             parameters.values.size() <= maximum_parameter_count &&
             std::all_of(parameters.values.begin(), parameters.values.end(), [](int value) {
               return value >= 0;
             });
    }

    std::vector<int> quiescent_values(const detail::parameters_t &current) {
      auto values = current.values;
      values[8] = 0;  // No duplicate-frame diagnostic stream on the frame-signalling transport.
      values[9] = 0;  // No optical statistics stream.
      values[14] = 0;  // No autonomous averaging after the last explicit trigger.
      values[16] = 1;  // Stop optical triggers without changing the previous protocol/timings.
      return values;
    }

    std::string command_for_values(const std::vector<int> &values) {
      auto command = std::string {"0"};
      for (int value : values) {
        command += ',';
        command += std::to_string(value);
      }
      return command;
    }

    std::vector<int> configured_values(const detail::parameters_t &current, const settings_t &settings) {
      auto values = quiescent_values(current);
      values[0] = settings.protocol;
      values[1] = settings.frame_delay_us;
      values[2] = settings.frame_duration_us;
      values[3] = settings.signal_spacing_us;
      values[12] = settings.flip_eyes ? 1 : 0;
      // Without averaging every eye boundary is explicitly acknowledged by the presenter, and a
      // stalled or closed presenter cannot leave a free-running shutter sequence. With averaging
      // the firmware continues by itself for a bounded number of signals after the last command;
      // quiesce() and close() stop that explicitly.
      values[14] = settings.averaging ? 1 : 0;
      // The firmware's averaging filter is uint16 and is given the eye interval only when it is
      // used. Otherwise it stays zero and the host eye interval is validated independently.
      values[15] = settings.averaging ? settings.target_frametime_us : 0;
      return values;
    }
  }  // namespace

  namespace detail {
    std::optional<std::string> normalize_port(std::string_view port) {
      if (port.starts_with(R"(\\.\)")) {
        port.remove_prefix(4);
      }
      if (port.size() < 4 || port.size() > 8 || (port[0] != 'C' && port[0] != 'c') || (port[1] != 'O' && port[1] != 'o') || (port[2] != 'M' && port[2] != 'm')) {
        return std::nullopt;
      }
      const auto number = parse_integer(port.substr(3));
      if (!number || *number == 0 || *number > 65535) {
        return std::nullopt;
      }
      return "COM" + std::to_string(*number);
    }

    bool valid_settings(const settings_t &settings) {
      const auto timing = timing_limits(settings.protocol, settings.signal_spacing_us);
      if (!normalize_port(settings.port) || !timing || settings.frame_delay_us < timing->min_delay_us || settings.frame_delay_us > timing->max_delay_us || settings.frame_duration_us < timing->min_duration_us || settings.frame_duration_us > timing->max_duration_us || settings.target_frametime_us < 0) {
        return false;
      }
      // The averaging filter holds the eye interval in sixteen bits.
      if (settings.averaging && (settings.target_frametime_us <= 0 || settings.target_frametime_us > 65535)) {
        return false;
      }
      const int window_end_us = settings.frame_delay_us + settings.frame_duration_us;
      // Timer1 cannot represent a full-wrap delay. The presentation policy additionally keeps
      // this eye's complete finite window inside its selected eye interval, rather than relying
      // on the firmware's ability to overlap work across opposite-eye boundaries.
      const int queue_end_us = settings.frame_duration_us + settings.signal_spacing_us + timing->min_period_tail_base_us;
      return window_end_us <= timing->max_window_us &&
             (settings.target_frametime_us == 0 ||
              (window_end_us <= settings.target_frametime_us && queue_end_us <= settings.target_frametime_us));
    }

    std::optional<int> host_feedback_budget_us(const settings_t &settings) {
      if (!valid_settings(settings) || settings.target_frametime_us <= 0) {
        return std::nullopt;
      }
      const auto timing = timing_limits(settings.protocol, settings.signal_spacing_us);
      const auto budget = static_cast<std::int64_t>(settings.target_frametime_us) - settings.frame_delay_us -
                          settings.frame_duration_us - settings.signal_spacing_us - timing->min_period_tail_base_us;
      return budget > 0 ? std::make_optional(static_cast<int>(budget)) : std::nullopt;
    }

    std::optional<parameters_t> parse_parameters(std::string_view line) {
      constexpr std::string_view prefix = "parameters ";
      if (!line.starts_with(prefix) || line.size() > 4096) {
        return std::nullopt;
      }
      line.remove_prefix(prefix.size());
      if (!line.empty() && line.back() == '\r') {
        line.remove_suffix(1);
      }
      parameters_t parameters;
      bool first = true;
      while (true) {
        const auto comma = line.find(',');
        const auto value = parse_integer(line.substr(0, comma));
        if (!value) {
          return std::nullopt;
        }
        if (first) {
          parameters.firmware_version = *value;
          first = false;
        } else {
          parameters.values.push_back(*value);
          if (parameters.values.size() > maximum_parameter_count) {
            return std::nullopt;
          }
        }
        if (comma == std::string_view::npos) {
          break;
        }
        line.remove_prefix(comma + 1);
      }
      if (!supported_parameters(parameters)) {
        return std::nullopt;
      }
      return parameters;
    }

    std::optional<std::string> quiesce_command(const parameters_t &current) {
      if (!supported_parameters(current)) {
        return std::nullopt;
      }
      return command_for_values(quiescent_values(current));
    }

    bool quiesce_matches(const parameters_t &actual, const parameters_t &previous) {
      return supported_parameters(actual) && supported_parameters(previous) &&
             actual.firmware_version == previous.firmware_version && actual.values == quiescent_values(previous);
    }

    std::optional<std::string> configuration_command(const parameters_t &current, const settings_t &settings) {
      if (!supported_parameters(current) || !valid_settings(settings)) {
        return std::nullopt;
      }
      return command_for_values(configured_values(current, settings));
    }

    bool configuration_matches(const parameters_t &actual, const parameters_t &previous, const settings_t &settings) {
      return supported_parameters(actual) && supported_parameters(previous) && valid_settings(settings) &&
             actual.firmware_version == previous.firmware_version &&
             actual.values == configured_values(previous, settings);
    }

    bool quiesce_with_io(const parameters_t &expected, const quiesce_io_t &io) {
      if (!supported_parameters(expected) || !limits::finite_close_protocol(expected.values[0]) ||
          expected.values[14] != 0 || expected.values[16] != 1 ||
          !io.now || !io.cancelled || !io.finish_signal || !io.read_parameters || !io.wait_until || io.cancelled()) {
        return false;
      }
      const auto deadline = io.now() + transaction_timeout;
      if (!io.finish_signal(deadline) || io.cancelled() || io.now() >= deadline) {
        return false;
      }
      const auto actual = io.read_parameters(deadline);
      if (!actual || io.cancelled() || io.now() >= deadline ||
          actual->firmware_version != expected.firmware_version || actual->values != expected.values) {
        return false;
      }
      // Firmware command9 invokes its trigger synchronously before the following command0 can
      // report parameters/OK. With unchanged PC serial/averaging-off settings, no new trigger
      // source remains; only comparator/IR work behind that processing barrier must expire.
      const auto drained_at = io.now() + limits::startup_settle;
      return io.wait_until(drained_at) && !io.cancelled() && io.now() >= drained_at;
    }
  }  // namespace detail

  struct emitter_t::impl_t {
    std::mutex mutex;
    std::atomic<HANDLE> port {INVALID_HANDLE_VALUE};
    std::atomic<bool> ready {false};
    std::atomic<bool> closing {false};
    OVERLAPPED signal_io {};
    std::array<char, 4> signal_bytes {};
    bool signal_pending = false;
    std::optional<detail::parameters_t> configured_parameters;
    settings_t configured_settings;
    std::int64_t qpc_frequency = 0;

    bool transfer(bool write, void *bytes, DWORD length, DWORD &transferred, std::chrono::steady_clock::time_point deadline) {
      OVERLAPPED io {};
      io.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
      if (!io.hEvent) {
        return false;
      }
      const HANDLE handle = port.load();
      const BOOL started = write ? WriteFile(handle, bytes, length, &transferred, &io) : ReadFile(handle, bytes, length, &transferred, &io);
      if (started) {
        CloseHandle(io.hEvent);
        return true;
      }
      if (GetLastError() != ERROR_IO_PENDING) {
        CloseHandle(io.hEvent);
        return false;
      }
      bool completed = false;
      while (!closing.load()) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        if (remaining <= 0ms) {
          break;
        }
        const auto result = WaitForSingleObject(io.hEvent, static_cast<DWORD>(std::min(remaining, 50ms).count()));
        if (result == WAIT_OBJECT_0) {
          completed = GetOverlappedResult(handle, &io, &transferred, FALSE);
          break;
        }
        if (result != WAIT_TIMEOUT) {
          break;
        }
      }
      if (!completed) {
        CancelIoEx(handle, &io);
        // Win32 owns the OVERLAPPED and buffer until cancellation completion.
        GetOverlappedResult(handle, &io, &transferred, TRUE);
      }
      CloseHandle(io.hEvent);
      return completed;
    }

    bool transaction(std::string command, std::vector<std::string> &response, std::chrono::steady_clock::time_point deadline) {
      response.clear();
      command += '\n';
      DWORD written = 0;
      if (!transfer(true, command.data(), static_cast<DWORD>(command.size()), written, deadline) || written != command.size()) {
        return false;
      }
      std::string line;
      std::size_t received = 0;
      while (!closing.load() && std::chrono::steady_clock::now() < deadline) {
        char next = 0;
        DWORD count = 0;
        if (!transfer(false, &next, 1, count, deadline)) {
          return false;
        }
        if (count == 0) {
          continue;
        }
        if (++received > 16384 || line.size() >= 4096) {
          return false;
        }
        if (next != '\n') {
          if (next != '\r') {
            line += next;
          }
          continue;
        }
        if (line == "OK") {
          return true;
        }
        if (!line.empty() && line.front() != '+') {
          response.push_back(std::move(line));
          if (response.size() > 64) {
            return false;
          }
        }
        line.clear();
      }
      return false;
    }

    bool transaction(std::string command, std::vector<std::string> &response) {
      return transaction(std::move(command), response, std::chrono::steady_clock::now() + transaction_timeout);
    }

    std::optional<detail::parameters_t> read_parameters(std::chrono::steady_clock::time_point deadline) {
      std::vector<std::string> response;
      if (!transaction("0", response, deadline)) {
        return std::nullopt;
      }
      for (const auto &line : response) {
        if (line.starts_with("parameters")) {
          return detail::parse_parameters(line);
        }
      }
      return std::nullopt;
    }

    std::optional<detail::parameters_t> read_parameters() {
      return read_parameters(std::chrono::steady_clock::now() + transaction_timeout);
    }

    bool wait_until(std::chrono::steady_clock::time_point until) const {
      while (!closing.load()) {
        const auto remaining = until - std::chrono::steady_clock::now();
        if (remaining <= std::chrono::steady_clock::duration::zero()) {
          return true;
        }
        std::this_thread::sleep_for(std::min(remaining, std::chrono::duration_cast<std::chrono::steady_clock::duration>(5ms)));
      }
      return false;
    }

    bool settle() const {
      // Resetting firmware averaging does not cancel already scheduled OPEN/CLOSE comparators.
      // Let both wrap horizons plus the bounded IR-send drain margin elapse before a first new
      // trigger, especially when changing protocol while the previous profile was still active.
      return wait_until(std::chrono::steady_clock::now() + limits::startup_settle);
    }

    bool finish_signal(std::chrono::steady_clock::time_point deadline) {
      const auto handle = port.load();
      while (!closing.load() && std::chrono::steady_clock::now() < deadline) {
        if (signal_pending) {
          DWORD count = 0;
          if (GetOverlappedResult(handle, &signal_io, &count, FALSE)) {
            if (count != signal_bytes.size()) {
              return false;
            }
            signal_pending = false;
          } else if (GetLastError() != ERROR_IO_INCOMPLETE) {
            return false;
          }
        }
        DWORD errors = 0;
        COMSTAT state {};
        if (!ClearCommError(handle, &errors, &state) || errors) {
          return false;
        }
        if (!signal_pending && state.cbOutQue == 0) {
          return true;
        }
        const auto remaining = deadline - std::chrono::steady_clock::now();
        if (remaining <= std::chrono::steady_clock::duration::zero()) {
          break;
        }
        if (signal_pending) {
          const auto wait = std::min(50ms, std::chrono::ceil<std::chrono::milliseconds>(remaining));
          const auto result = WaitForSingleObject(signal_io.hEvent, static_cast<DWORD>(wait.count()));
          if (result != WAIT_OBJECT_0 && result != WAIT_TIMEOUT) {
            return false;
          }
        } else {
          // A completed host write can still be queued in the serial driver. Wait only within
          // the shared barrier deadline; FlushFileBuffers would introduce an unbounded wait.
          std::this_thread::sleep_for(std::min(remaining, std::chrono::duration_cast<std::chrono::steady_clock::duration>(5ms)));
        }
      }
      return false;
    }

    // With averaging the firmware keeps signalling after the last eye command. Stop it by
    // writing the same profile with averaging off, let scheduled signals drain, then write the
    // averaging profile again so the next eye commands start a new rhythm from nothing.
    bool restart_averaging() {
      const auto deadline = std::chrono::steady_clock::now() + transaction_timeout;
      const auto averaging = *configured_parameters;
      const auto quiet_command = detail::quiesce_command(averaging);
      std::vector<std::string> response;
      if (!finish_signal(deadline) || !quiet_command || !transaction(*quiet_command, response)) {
        return false;
      }
      const auto quiet = read_parameters();
      if (!quiet || !detail::quiesce_matches(*quiet, averaging) || !settle()) {
        return false;
      }
      const auto command = detail::configuration_command(*quiet, configured_settings);
      if (!command || !transaction(*command, response)) {
        return false;
      }
      const auto actual = read_parameters();
      if (!actual || !detail::configuration_matches(*actual, *quiet, configured_settings) || closing.load()) {
        return false;
      }
      configured_parameters = *actual;
      return true;
    }

    // Best effort before the port closes; if it fails the firmware still stops by itself after
    // its bounded number of signals without a command.
    void stop_averaging() {
      if (!ready.load() || !configured_parameters || !configured_settings.averaging) {
        return;
      }
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds {150};
      std::vector<std::string> response;
      if (const auto quiet_command = detail::quiesce_command(*configured_parameters); quiet_command && finish_signal(deadline)) {
        (void) transaction(*quiet_command, response, deadline);
      }
    }

    void close_locked() {
      ready.store(false);
      configured_parameters.reset();
      qpc_frequency = 0;
      const auto handle = port.load();
      if (handle != INVALID_HANDLE_VALUE) {
        CancelIoEx(handle, nullptr);
        if (signal_pending) {
          DWORD count = 0;
          GetOverlappedResult(handle, &signal_io, &count, TRUE);
          signal_pending = false;
        }
        CloseHandle(handle);
        port.store(INVALID_HANDLE_VALUE);
      }
      if (signal_io.hEvent) {
        CloseHandle(signal_io.hEvent);
        signal_io = {};
      }
    }
  };

  emitter_t::emitter_t():
      impl_(std::make_unique<impl_t>()) {}

  emitter_t::~emitter_t() {
    close();
  }

  bool emitter_t::connect(const settings_t &settings, std::stop_token stop) {
    close();
    if (!detail::valid_settings(settings)) {
      if (!stop.stop_requested()) {
        BOOST_LOG(error) << "Open3DOLED emitter settings or COM port are invalid.";
      }
      return false;
    }
    std::lock_guard lock(impl_->mutex);
    impl_->closing.store(false);
    std::stop_callback cancel(stop, [this]() {
      impl_->closing.store(true);
    });
    if (impl_->closing.load()) {
      return false;
    }
    const auto normalized = *detail::normalize_port(settings.port);
    const auto path = std::string {R"(\\.\)"} + normalized;
    const auto handle = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
      if (!stop.stop_requested() && !impl_->closing.load()) {
        BOOST_LOG(error) << "Could not open Open3DOLED emitter on " << normalized << ": " << GetLastError();
      }
      return false;
    }
    impl_->port.store(handle);
    DCB dcb {};
    dcb.DCBlength = sizeof(dcb);
    COMMTIMEOUTS timeouts {};
    timeouts.ReadIntervalTimeout = MAXDWORD;
    timeouts.ReadTotalTimeoutMultiplier = MAXDWORD;
    timeouts.ReadTotalTimeoutConstant = 20;
    timeouts.WriteTotalTimeoutConstant = 20;
    if (!GetCommState(handle, &dcb)) {
      impl_->close_locked();
      return false;
    }
    dcb.BaudRate = 115200;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fParity = FALSE;
    dcb.fOutxCtsFlow = FALSE;
    dcb.fOutxDsrFlow = FALSE;
    dcb.fDtrControl = DTR_CONTROL_ENABLE;
    dcb.fDsrSensitivity = FALSE;
    dcb.fOutX = FALSE;
    dcb.fInX = FALSE;
    dcb.fRtsControl = RTS_CONTROL_DISABLE;
    dcb.fAbortOnError = FALSE;
    if (!SetCommState(handle, &dcb) || !SetCommTimeouts(handle, &timeouts) || !SetupComm(handle, 4096, 4096) || !PurgeComm(handle, PURGE_RXCLEAR | PURGE_TXCLEAR)) {
      impl_->close_locked();
      return false;
    }
    impl_->signal_io.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    const auto previous = impl_->read_parameters();
    const auto quiet_command = previous ? detail::quiesce_command(*previous) : std::nullopt;
    std::vector<std::string> response;
    if (!impl_->signal_io.hEvent || !quiet_command || !impl_->transaction(*quiet_command, response)) {
      if (!stop.stop_requested() && !impl_->closing.load()) {
        BOOST_LOG(error) << "Open3DOLED emitter could not enter a quiescent PC serial state on " << normalized;
      }
      impl_->close_locked();
      return false;
    }
    const auto quiet = impl_->read_parameters();
    if (!quiet || !detail::quiesce_matches(*quiet, *previous) || !impl_->settle()) {
      impl_->close_locked();
      return false;
    }
    const auto command = detail::configuration_command(*quiet, settings);
    if (!command || !impl_->transaction(*command, response)) {
      if (!stop.stop_requested() && !impl_->closing.load()) {
        BOOST_LOG(error) << "Open3DOLED emitter did not acknowledge a supported PC serial configuration on " << normalized;
      }
      impl_->close_locked();
      return false;
    }
    const auto actual = impl_->read_parameters();
    if (!actual || !detail::configuration_matches(*actual, *quiet, settings) || impl_->closing.load()) {
      if (!stop.stop_requested() && !impl_->closing.load()) {
        BOOST_LOG(error) << "Open3DOLED emitter configuration readback failed on " << normalized;
      }
      impl_->close_locked();
      return false;
    }
    if (!impl_->settle()) {
      impl_->close_locked();
      return false;
    }
    impl_->configured_parameters = *actual;
    impl_->configured_settings = settings;
    LARGE_INTEGER frequency {};
    impl_->qpc_frequency = QueryPerformanceFrequency(&frequency) && frequency.QuadPart > 0 ? frequency.QuadPart : 0;
    impl_->ready.store(true);
    BOOST_LOG(info) << "Open3DOLED emitter ready on " << normalized << ", firmware " << actual->firmware_version << (settings.averaging ? " (volatile PC serial mode, firmware averaging)." : " (volatile PC serial mode).");
    return true;
  }

  bool emitter_t::quiesce(std::stop_token stop) {
    std::lock_guard lock(impl_->mutex);
    const bool was_ready = impl_->ready.exchange(false);
    std::stop_callback cancel(stop, [this]() {
      impl_->closing.store(true);
    });
    const detail::quiesce_io_t io {
      .now = []() { return std::chrono::steady_clock::now(); },
      .cancelled = [this]() { return impl_->closing.load(); },
      .finish_signal = [this](auto deadline) { return impl_->finish_signal(deadline); },
      .read_parameters = [this](auto deadline) { return impl_->read_parameters(deadline); },
      .wait_until = [this](auto until) { return impl_->wait_until(until); },
    };
    const bool quiet = was_ready && impl_->configured_parameters &&
                       (impl_->configured_settings.averaging ? impl_->restart_averaging() : detail::quiesce_with_io(*impl_->configured_parameters, io));
    if (!quiet || impl_->closing.load()) {
      impl_->closing.store(true);
      impl_->close_locked();
      return false;
    }
    impl_->ready.store(true);
    return true;
  }

  bool emitter_t::signal(bool left_eye, signal_timing_t *timing) {
    if (timing) {
      *timing = {};
    }
    std::unique_lock lock(impl_->mutex, std::try_to_lock);
    if (!lock || !impl_->ready.load() || impl_->closing.load()) {
      return false;
    }
    const auto handle = impl_->port.load();
    DWORD count = 0;
    if (impl_->signal_pending) {
      if (!GetOverlappedResult(handle, &impl_->signal_io, &count, FALSE) || count != impl_->signal_bytes.size()) {
        impl_->ready.store(false);
        CancelIoEx(handle, &impl_->signal_io);
        return false;
      }
      impl_->signal_pending = false;
    }
    DWORD errors = 0;
    COMSTAT state {};
    if (!ClearCommError(handle, &errors, &state) || errors || state.cbOutQue != 0) {
      impl_->ready.store(false);
      return false;
    }
    impl_->signal_bytes = {'9', ',', left_eye ? '0' : '1', '\n'};
    ResetEvent(impl_->signal_io.hEvent);
    LARGE_INTEGER write_begin {}, write_end {};
    BOOL begin_clock_valid = FALSE;
    if (timing) {
      begin_clock_valid = QueryPerformanceCounter(&write_begin);
    }
    const BOOL started = WriteFile(handle, impl_->signal_bytes.data(), static_cast<DWORD>(impl_->signal_bytes.size()), &count, &impl_->signal_io);
    // Preserve WriteFile's result before the optional clock query can change last-error.
    const DWORD write_error = started ? ERROR_SUCCESS : GetLastError();
    if (timing) {
      const auto end_clock_valid = QueryPerformanceCounter(&write_end);
      *timing = {
        begin_clock_valid ? write_begin.QuadPart : 0,
        end_clock_valid ? write_end.QuadPart : 0,
        impl_->qpc_frequency,
        !started && write_error == ERROR_IO_PENDING,
      };
    }
    if (started) {
      if (count == impl_->signal_bytes.size()) {
        return true;
      }
    } else if (write_error == ERROR_IO_PENDING) {
      impl_->signal_pending = true;
      return true;
    }
    impl_->ready.store(false);
    return false;
  }

  bool emitter_t::healthy() const {
    return impl_->ready.load() && !impl_->closing.load();
  }

  void emitter_t::close() {
    {
      std::unique_lock lock(impl_->mutex, std::try_to_lock);
      if (lock && !impl_->closing.load()) {
        impl_->stop_averaging();
      }
    }
    impl_->closing.store(true);
    // The setup wait observes this flag at least every 50 ms and cancels its own operation.
    // Close/cancel the handle only under the mutex, so concurrent teardown cannot race a recycled
    // Windows HANDLE. Without averaging the last configured finite shutter pulse expires.
    std::lock_guard lock(impl_->mutex);
    impl_->close_locked();
  }

  std::vector<std::string> serial_ports() {
    std::vector<char> names(65536);
    DWORD length = QueryDosDeviceA(nullptr, names.data(), static_cast<DWORD>(names.size()));
    if (length == 0) {
      return {};
    }
    std::vector<std::string> ports;
    for (const char *name = names.data(); *name; name += std::char_traits<char>::length(name) + 1) {
      if (const auto normalized = detail::normalize_port(name)) {
        ports.push_back(*normalized);
      }
    }
    std::sort(ports.begin(), ports.end(), [](const auto &a, const auto &b) {
      return a.size() == b.size() ? a < b : a.size() < b.size();
    });
    ports.erase(std::unique(ports.begin(), ports.end()), ports.end());
    return ports;
  }
}  // namespace platf::open3d
