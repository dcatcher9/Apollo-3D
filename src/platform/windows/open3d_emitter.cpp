#include "open3d_emitter.h"

#include "src/logging.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <mutex>
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

    std::vector<int> configured_values(const detail::parameters_t &current, const settings_t &settings) {
      auto values = current.values;
      values[0] = settings.protocol;
      values[1] = settings.frame_delay_us;
      values[2] = settings.frame_duration_us;
      values[3] = settings.signal_spacing_us;
      values[8] = 0;  // No duplicate-frame diagnostic stream on the frame-signalling transport.
      values[9] = 0;  // No optical statistics stream.
      values[12] = settings.flip_eyes ? 1 : 0;
      // Every eye boundary is explicitly acknowledged by the presenter. Disable autonomous
      // averaging so a stalled/closed presenter cannot leave a free-running shutter sequence.
      values[14] = 0;
      values[15] = settings.target_frametime_us;
      values[16] = 1;  // PC serial drive mode.
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
      return normalize_port(settings.port).has_value() &&
             settings.protocol >= 0 && settings.protocol <= 6 &&
             settings.frame_delay_us >= 0 && settings.frame_delay_us <= 65535 &&
             settings.frame_duration_us > 0 && settings.frame_duration_us <= 65535 &&
             settings.signal_spacing_us >= 0 && settings.signal_spacing_us <= 65535 &&
             settings.target_frametime_us >= 0 && settings.target_frametime_us <= 65535;
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

    std::optional<std::string> configuration_command(const parameters_t &current, const settings_t &settings) {
      if (!supported_parameters(current) || !valid_settings(settings)) {
        return std::nullopt;
      }
      auto command = std::string {"0"};
      for (int value : configured_values(current, settings)) {
        command += ',';
        command += std::to_string(value);
      }
      return command;
    }

    bool configuration_matches(const parameters_t &actual, const parameters_t &previous, const settings_t &settings) {
      return supported_parameters(actual) && supported_parameters(previous) && valid_settings(settings) &&
             actual.firmware_version == previous.firmware_version &&
             actual.values == configured_values(previous, settings);
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

    bool transaction(std::string command, std::vector<std::string> &response) {
      response.clear();
      command += '\n';
      const auto deadline = std::chrono::steady_clock::now() + transaction_timeout;
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

    std::optional<detail::parameters_t> read_parameters() {
      std::vector<std::string> response;
      if (!transaction("0", response)) {
        return std::nullopt;
      }
      for (const auto &line : response) {
        if (line.starts_with("parameters")) {
          return detail::parse_parameters(line);
        }
      }
      return std::nullopt;
    }

    void close_locked() {
      ready.store(false);
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
      BOOST_LOG(error) << "Open3DOLED emitter settings or COM port are invalid.";
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
      BOOST_LOG(error) << "Could not open Open3DOLED emitter on " << normalized << ": " << GetLastError();
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
    const auto command = previous ? detail::configuration_command(*previous, settings) : std::nullopt;
    std::vector<std::string> response;
    if (!impl_->signal_io.hEvent || !command || !impl_->transaction(*command, response)) {
      BOOST_LOG(error) << "Open3DOLED emitter did not acknowledge a supported PC serial configuration on " << normalized;
      impl_->close_locked();
      return false;
    }
    const auto actual = impl_->read_parameters();
    if (!actual || !detail::configuration_matches(*actual, *previous, settings) || impl_->closing.load()) {
      BOOST_LOG(error) << "Open3DOLED emitter configuration readback failed on " << normalized;
      impl_->close_locked();
      return false;
    }
    impl_->ready.store(true);
    BOOST_LOG(info) << "Open3DOLED emitter ready on " << normalized << ", firmware " << actual->firmware_version << " (volatile PC serial mode).";
    return true;
  }

  bool emitter_t::signal(bool left_eye) {
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
    const auto started = WriteFile(handle, impl_->signal_bytes.data(), static_cast<DWORD>(impl_->signal_bytes.size()), &count, &impl_->signal_io);
    if (started) {
      if (count == impl_->signal_bytes.size()) {
        return true;
      }
    } else if (GetLastError() == ERROR_IO_PENDING) {
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
    impl_->closing.store(true);
    // The setup wait observes this flag at least every 50 ms and cancels its own operation.
    // Close/cancel the handle only under the mutex, so concurrent teardown cannot race a recycled
    // Windows HANDLE. No averaging is enabled: the last configured finite shutter pulse expires.
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
