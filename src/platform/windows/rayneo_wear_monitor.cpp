#include "rayneo_wear_monitor.h"

// The Windows SDK HID headers require the Win32 base types. Keep this block
// ahead of Boost.Log, which may otherwise leave the SDK headers in a partial
// include state under MinGW.
// clang-format off
#include <windows.h>
#include <cfgmgr32.h>
#include <hidsdi.h>
#include <pbt.h>
#include <powrprof.h>
#include <setupapi.h>
// clang-format on

#include "src/logging.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ar_glasses::rayneo {
  using namespace std::literals;

  namespace {
    constexpr std::string_view supported_display_model = "DISPLAY:TCL03D4";
    constexpr USHORT rayneo_vendor_id = 0x1BBB;
    constexpr USHORT rayneo_product_id = 0xAF50;
    constexpr USAGE rayneo_usage_page = 0xFF00;
    constexpr USAGE rayneo_usage = 0x0001;

    constexpr std::size_t report_size = 65;
    constexpr std::size_t report_id_offset = 0;
    constexpr std::size_t vendor_payload_offset = 1;
    constexpr std::size_t sensor_tick_payload_offset = 40;
    constexpr std::size_t sensor_proximity_payload_offset = 44;
    constexpr std::size_t board_command_payload_offset = 8;
    constexpr std::size_t board_id_payload_offset = 21;

    constexpr std::uint8_t vendor_report_id = 0x00;
    constexpr std::uint8_t command_prefix = 0x66;
    constexpr std::uint8_t board_info_command = 0x00;
    constexpr std::uint8_t sensor_start_command = 0x01;
    constexpr std::uint8_t sensor_stop_command = 0x02;
    constexpr std::uint8_t response_prefix = 0x99;
    constexpr std::uint8_t sensor_response = 0x65;
    constexpr std::uint8_t board_info_response = 0xC8;
    constexpr std::uint8_t supported_board_id = 0x3A;

    constexpr float off_head_maximum = 10000.0f;
    constexpr float worn_minimum = 15000.0f;
    constexpr float proximity_envelope_maximum = 65535.0f;
    constexpr auto stable_debounce = 150ms;
    constexpr auto stale_timeout = 10000ms;
    constexpr auto command_timeout = 500ms;
    constexpr auto board_response_timeout = 1000ms;
    constexpr auto stop_command_timeout = 250ms;
    constexpr auto notification_watchdog_interval = 30s;
    constexpr auto notification_failure_retry_interval = 1000ms;
    constexpr auto device_arrival_settle = 100ms;
    constexpr auto system_resume_settle = 250ms;
    constexpr auto failure_log_interval = 10s;

    enum class recovery_failure_e {
      unavailable,
      transient,
    };

    constexpr std::array transient_retry_schedule {
      1s,
      2s,
      5s,
      10s,
      30s,
    };

    std::chrono::milliseconds recovery_retry_delay(
      recovery_failure_e failure,
      bool notifications_available,
      unsigned consecutive_failures
    ) {
      if (!notifications_available) {
        return notification_failure_retry_interval;
      }
      if (failure == recovery_failure_e::unavailable) {
        return notification_watchdog_interval;
      }
      return transient_retry_schedule[std::min<std::size_t>(
        consecutive_failures,
        transient_retry_schedule.size() - 1
      )];
    }

    void advance_recovery_failure(unsigned &consecutive_failures) {
      if (consecutive_failures < transient_retry_schedule.size()) {
        ++consecutive_failures;
      }
    }

    constexpr DWORD power_device_notify_callback = 0x00000002;
    using power_notify_callback_t = ULONG(CALLBACK *)(PVOID, ULONG, PVOID);

    struct power_notify_subscribe_parameters_t {
      power_notify_callback_t callback = nullptr;
      PVOID context = nullptr;
    };

    enum lifecycle_notification_e : std::uint32_t {
      lifecycle_none = 0,
      lifecycle_interface_arrival = 1U << 0,
      lifecycle_interface_removal = 1U << 1,
      lifecycle_suspend = 1U << 2,
      lifecycle_resume = 1U << 3,
    };

    std::uint32_t power_notification_reason(ULONG notification) {
      switch (notification) {
        case PBT_APMSUSPEND:
          return lifecycle_suspend;
        case PBT_APMRESUMEAUTOMATIC:
        case PBT_APMRESUMECRITICAL:
        case PBT_APMRESUMESUSPEND:
          return lifecycle_resume;
        default:
          return lifecycle_none;
      }
    }

    bool power_notification_requests_recovery(
      ULONG notification,
      bool was_suspended
    ) noexcept {
      return notification == PBT_APMSUSPEND ||
             notification == PBT_APMRESUMEAUTOMATIC ||
             notification == PBT_APMRESUMECRITICAL ||
             (notification == PBT_APMRESUMESUSPEND && was_suspended);
    }

    static_assert(vendor_payload_offset + sensor_proximity_payload_offset + sizeof(float) <= report_size);
    static_assert(vendor_payload_offset + sensor_tick_payload_offset + sizeof(std::uint32_t) <= report_size);
    static_assert(vendor_payload_offset + board_id_payload_offset < report_size);

    class win_handle_t {
    public:
      win_handle_t() = default;

      explicit win_handle_t(HANDLE handle):
          handle_(handle) {
      }

      ~win_handle_t() {
        reset();
      }

      win_handle_t(const win_handle_t &) = delete;
      win_handle_t &operator=(const win_handle_t &) = delete;

      win_handle_t(win_handle_t &&other) noexcept:
          handle_(std::exchange(other.handle_, nullptr)) {
      }

      win_handle_t &operator=(win_handle_t &&other) noexcept {
        if (this != &other) {
          reset(std::exchange(other.handle_, nullptr));
        }
        return *this;
      }

      [[nodiscard]] bool valid() const noexcept {
        return handle_ && handle_ != INVALID_HANDLE_VALUE;
      }

      [[nodiscard]] HANDLE get() const noexcept {
        return handle_;
      }

      void reset(HANDLE handle = nullptr) noexcept {
        if (valid()) {
          CloseHandle(handle_);
        }
        handle_ = handle;
      }

    private:
      HANDLE handle_ = nullptr;
    };

    class device_info_set_t {
    public:
      explicit device_info_set_t(HDEVINFO set):
          set_(set) {
      }

      ~device_info_set_t() {
        if (valid()) {
          SetupDiDestroyDeviceInfoList(set_);
        }
      }

      device_info_set_t(const device_info_set_t &) = delete;
      device_info_set_t &operator=(const device_info_set_t &) = delete;

      [[nodiscard]] bool valid() const noexcept {
        return set_ != INVALID_HANDLE_VALUE;
      }

      [[nodiscard]] HDEVINFO get() const noexcept {
        return set_;
      }

    private:
      HDEVINFO set_ = INVALID_HANDLE_VALUE;
    };

    class preparsed_data_t {
    public:
      preparsed_data_t() = default;

      ~preparsed_data_t() {
        if (data_) {
          HidD_FreePreparsedData(data_);
        }
      }

      preparsed_data_t(const preparsed_data_t &) = delete;
      preparsed_data_t &operator=(const preparsed_data_t &) = delete;

      [[nodiscard]] PHIDP_PREPARSED_DATA get() const noexcept {
        return data_;
      }

      [[nodiscard]] PHIDP_PREPARSED_DATA *put() noexcept {
        return &data_;
      }

    private:
      PHIDP_PREPARSED_DATA data_ = nullptr;
    };

    std::string windows_error(std::string_view operation, DWORD error) {
      std::ostringstream message;
      message << operation << " failed with Windows error " << error;
      return message.str();
    }

    DWORD wait_timeout(std::optional<std::chrono::milliseconds> timeout) {
      if (!timeout) {
        return INFINITE;
      }
      constexpr auto maximum = static_cast<long long>(INFINITE) - 1;
      return static_cast<DWORD>(std::clamp(timeout->count(), 0LL, maximum));
    }

    enum class io_result_e {
      complete,
      pending,
      timeout,
      stopped,
      interrupted,
      error,
    };

    class overlapped_io_t {
    public:
      overlapped_io_t():
          event_(CreateEventW(nullptr, TRUE, FALSE, nullptr)),
          event_error_(event_.valid() ? ERROR_SUCCESS : GetLastError()) {
      }

      ~overlapped_io_t() = default;

      overlapped_io_t(const overlapped_io_t &) = delete;
      overlapped_io_t &operator=(const overlapped_io_t &) = delete;

      [[nodiscard]] bool valid() const noexcept {
        return event_.valid();
      }

      [[nodiscard]] bool pending() const noexcept {
        return pending_;
      }

      [[nodiscard]] DWORD transferred() const noexcept {
        return transferred_;
      }

      [[nodiscard]] DWORD error() const noexcept {
        return error_;
      }

      io_result_e start_read(HANDLE device, std::span<std::uint8_t> buffer) {
        if (!prepare()) {
          return io_result_e::error;
        }
        const BOOL accepted = ReadFile(
          device,
          buffer.data(),
          static_cast<DWORD>(buffer.size()),
          nullptr,
          &overlapped_
        );
        return finish_start(device, accepted);
      }

      io_result_e start_write(HANDLE device, std::span<const std::uint8_t> buffer) {
        if (!prepare()) {
          return io_result_e::error;
        }
        const BOOL accepted = WriteFile(
          device,
          buffer.data(),
          static_cast<DWORD>(buffer.size()),
          nullptr,
          &overlapped_
        );
        return finish_start(device, accepted);
      }

      io_result_e wait(
        HANDLE device,
        HANDLE stop_event,
        HANDLE control_event,
        std::optional<std::chrono::milliseconds> timeout,
        bool honor_stop,
        bool honor_control
      ) {
        if (!pending_) {
          error_ = ERROR_INVALID_STATE;
          return io_result_e::error;
        }

        if (honor_stop && WaitForSingleObject(stop_event, 0) == WAIT_OBJECT_0) {
          return io_result_e::stopped;
        }
        if (honor_control && WaitForSingleObject(control_event, 0) == WAIT_OBJECT_0) {
          return io_result_e::interrupted;
        }

        std::array<HANDLE, 3> handles {event_.get(), stop_event, control_event};
        const DWORD handle_count = 1 + static_cast<DWORD>(honor_stop) +
                                   static_cast<DWORD>(honor_control);
        if (!honor_stop && honor_control) {
          handles[1] = control_event;
        }
        const DWORD result = WaitForMultipleObjects(
          handle_count,
          handles.data(),
          FALSE,
          wait_timeout(timeout)
        );
        if (result == WAIT_OBJECT_0) {
          // WaitForMultipleObjects selects the lowest signaled index. Recheck lifecycle controls
          // before accepting an I/O completion so a simultaneous suspend/removal wins.
          if (honor_stop && WaitForSingleObject(stop_event, 0) == WAIT_OBJECT_0) {
            return io_result_e::stopped;
          }
          if (honor_control && WaitForSingleObject(control_event, 0) == WAIT_OBJECT_0) {
            return io_result_e::interrupted;
          }
          return finish(device, false);
        }
        if (honor_stop && result == WAIT_OBJECT_0 + 1) {
          return io_result_e::stopped;
        }
        const DWORD control_index = WAIT_OBJECT_0 + 1 + static_cast<DWORD>(honor_stop);
        if (honor_control && result == control_index) {
          return io_result_e::interrupted;
        }
        if (result == WAIT_TIMEOUT) {
          return io_result_e::timeout;
        }
        error_ = result == WAIT_FAILED ? GetLastError() : ERROR_INVALID_STATE;
        return io_result_e::error;
      }

      void cancel_and_drain(HANDLE device) noexcept {
        if (!pending_) {
          return;
        }
        if (!CancelIoEx(device, &overlapped_) && GetLastError() != ERROR_NOT_FOUND) {
          // The completion still has to be drained before OVERLAPPED storage is released.
        }
        DWORD ignored = 0;
        GetOverlappedResult(device, &overlapped_, &ignored, TRUE);
        pending_ = false;
      }

    private:
      bool prepare() {
        if (pending_) {
          error_ = ERROR_BUSY;
          return false;
        }
        if (!valid()) {
          error_ = event_error_;
          return false;
        }
        if (!ResetEvent(event_.get())) {
          error_ = GetLastError();
          return false;
        }
        overlapped_ = {};
        overlapped_.hEvent = event_.get();
        transferred_ = 0;
        error_ = ERROR_SUCCESS;
        return true;
      }

      io_result_e finish_start(HANDLE device, BOOL accepted) {
        if (!accepted) {
          error_ = GetLastError();
          if (error_ != ERROR_IO_PENDING) {
            return io_result_e::error;
          }
          pending_ = true;
          return io_result_e::pending;
        }
        pending_ = true;
        return finish(device, false);
      }

      io_result_e finish(HANDLE device, bool wait_for_completion) {
        DWORD transferred = 0;
        if (!GetOverlappedResult(device, &overlapped_, &transferred, wait_for_completion)) {
          error_ = GetLastError();
          pending_ = false;
          return io_result_e::error;
        }
        transferred_ = transferred;
        pending_ = false;
        error_ = ERROR_SUCCESS;
        return io_result_e::complete;
      }

      win_handle_t event_;
      DWORD event_error_ = ERROR_SUCCESS;
      OVERLAPPED overlapped_ {};
      DWORD transferred_ = 0;
      DWORD error_ = ERROR_SUCCESS;
      bool pending_ = false;
    };

    std::uint32_t read_u32_le(std::span<const std::uint8_t> bytes, std::size_t offset) {
      return static_cast<std::uint32_t>(bytes[offset]) |
             (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
             (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
             (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
    }

    wear_state_e classify_proximity(float proximity) {
      if (!std::isfinite(proximity) || proximity < 0.0f || proximity > proximity_envelope_maximum) {
        return wear_state_e::unknown;
      }
      if (proximity >= worn_minimum) {
        return wear_state_e::worn;
      }
      if (proximity <= off_head_maximum) {
        return wear_state_e::off_head;
      }
      return wear_state_e::unknown;
    }

    struct parsed_sensor_report_t {
      std::uint32_t tick = 0;
      float proximity = 0.0f;
      wear_state_e state = wear_state_e::unknown;
    };

    std::optional<parsed_sensor_report_t> parse_sensor_report(
      std::span<const std::uint8_t> report
    ) {
      if (report.size() != report_size || report[report_id_offset] != vendor_report_id || report[vendor_payload_offset] != response_prefix || report[vendor_payload_offset + 1] != sensor_response) {
        return std::nullopt;
      }

      const auto tick = read_u32_le(report, vendor_payload_offset + sensor_tick_payload_offset);
      const auto proximity_bits = read_u32_le(
        report,
        vendor_payload_offset + sensor_proximity_payload_offset
      );
      const auto proximity = std::bit_cast<float>(proximity_bits);
      return parsed_sensor_report_t {
        tick,
        proximity,
        classify_proximity(proximity),
      };
    }

    bool board_info_matches(std::span<const std::uint8_t> report) {
      return report.size() == report_size && report[report_id_offset] == vendor_report_id &&
             report[vendor_payload_offset] == response_prefix &&
             report[vendor_payload_offset + 1] == board_info_response &&
             report[vendor_payload_offset + board_command_payload_offset] == board_info_command &&
             report[vendor_payload_offset + board_id_payload_offset] == supported_board_id;
    }

    bool board_info_acknowledges_query(std::span<const std::uint8_t> report) {
      return report.size() == report_size && report[report_id_offset] == vendor_report_id &&
             report[vendor_payload_offset] == response_prefix &&
             report[vendor_payload_offset + 1] == board_info_response &&
             report[vendor_payload_offset + board_command_payload_offset] == board_info_command;
    }

    class wear_debounce_t {
    public:
      using clock_t = std::chrono::steady_clock;

      std::optional<wear_state_e> observe(
        std::uint32_t tick,
        wear_state_e observed,
        clock_t::time_point now
      ) {
        static_cast<void>(tick);
        last_valid_report_ = now;

        // A completed, correctly framed HID read is transport-fresh evidence. The embedded sensor
        // tick is payload metadata, not the Windows transport sequence: requiring it to advance
        // incorrectly expired a stable worn state on the 500 ms stale deadline. Keep the tick for
        // diagnostics and debounce classified proximity using the host receipt clock instead.

        if (!candidate_initialized_ || candidate_ != observed) {
          candidate_ = observed;
          candidate_since_ = now;
          candidate_initialized_ = true;
          return std::nullopt;
        }
        if (published_ != candidate_ && now - candidate_since_ >= stable_debounce) {
          published_ = candidate_;
          return published_;
        }
        return std::nullopt;
      }

      std::optional<wear_state_e> mark_stale(clock_t::time_point now) {
        if (!last_valid_report_ || now - *last_valid_report_ < stale_timeout) {
          return std::nullopt;
        }
        last_valid_report_.reset();
        candidate_initialized_ = false;
        if (published_ == wear_state_e::unknown) {
          return std::nullopt;
        }
        published_ = wear_state_e::unknown;
        return published_;
      }

      [[nodiscard]] wear_state_e state() const noexcept {
        return published_;
      }

    private:
      std::optional<clock_t::time_point> last_valid_report_;
      clock_t::time_point candidate_since_ {};
      wear_state_e candidate_ = wear_state_e::unknown;
      wear_state_e published_ = wear_state_e::unknown;
      bool candidate_initialized_ = false;
    };

    constexpr wchar_t ascii_lower(wchar_t character) noexcept {
      return character >= L'A' && character <= L'Z' ?
               static_cast<wchar_t>(character + (L'a' - L'A')) :
               character;
    }

    bool is_rayneo_interface_path(std::wstring_view path) noexcept {
      constexpr std::wstring_view identity = L"vid_1bbb&pid_af50";
      if (path.size() < identity.size()) {
        return false;
      }

      for (std::size_t offset = 0; offset + identity.size() <= path.size(); ++offset) {
        bool matches = true;
        for (std::size_t index = 0; index < identity.size(); ++index) {
          if (ascii_lower(path[offset + index]) != identity[index]) {
            matches = false;
            break;
          }
        }
        if (!matches) {
          continue;
        }

        const bool begins_at_boundary = offset == 0 || path[offset - 1] == L'#' || path[offset - 1] == L'&';
        const auto end = offset + identity.size();
        const bool ends_at_boundary = end == path.size() || path[end] == L'#' || path[end] == L'&';
        if (begins_at_boundary && ends_at_boundary) {
          return true;
        }
      }
      return false;
    }

    struct opened_device_t {
      win_handle_t handle;
    };

    struct discovery_result_t {
      std::optional<opened_device_t> device;
      recovery_failure_e failure_kind = recovery_failure_e::transient;
      std::string failure;
    };

    discovery_result_t discover_unique_device() {
      GUID hid_guid {};
      HidD_GetHidGuid(&hid_guid);
      device_info_set_t device_set(SetupDiGetClassDevsW(
        &hid_guid,
        nullptr,
        nullptr,
        DIGCF_PRESENT | DIGCF_DEVICEINTERFACE
      ));
      if (!device_set.valid()) {
        return {{}, recovery_failure_e::transient, windows_error("SetupDiGetClassDevsW", GetLastError())};
      }

      std::vector<std::wstring> matching_paths;
      std::optional<std::string> enumeration_failure;
      for (DWORD index = 0;; ++index) {
        SP_DEVICE_INTERFACE_DATA interface_data {};
        interface_data.cbSize = sizeof(interface_data);
        if (!SetupDiEnumDeviceInterfaces(
              device_set.get(),
              nullptr,
              &hid_guid,
              index,
              &interface_data
            )) {
          const auto error = GetLastError();
          if (error == ERROR_NO_MORE_ITEMS) {
            break;
          }
          return {{}, recovery_failure_e::transient, windows_error("SetupDiEnumDeviceInterfaces", error)};
        }

        DWORD required_size = 0;
        SetLastError(ERROR_SUCCESS);
        const auto size_result = SetupDiGetDeviceInterfaceDetailW(
          device_set.get(),
          &interface_data,
          nullptr,
          0,
          &required_size,
          nullptr
        );
        const auto size_error = GetLastError();
        if (size_result || size_error != ERROR_INSUFFICIENT_BUFFER || required_size < sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W)) {
          enumeration_failure = windows_error("SetupDiGetDeviceInterfaceDetailW size query", size_error);
          continue;
        }

        std::vector<std::byte> detail_storage(required_size);
        auto *detail = reinterpret_cast<SP_DEVICE_INTERFACE_DETAIL_DATA_W *>(detail_storage.data());
        detail->cbSize = sizeof(*detail);
        if (!SetupDiGetDeviceInterfaceDetailW(
              device_set.get(),
              &interface_data,
              detail,
              required_size,
              nullptr,
              nullptr
            )) {
          enumeration_failure = windows_error("SetupDiGetDeviceInterfaceDetailW", GetLastError());
          continue;
        }
        if (is_rayneo_interface_path(detail->DevicePath)) {
          matching_paths.emplace_back(detail->DevicePath);
        }
      }

      std::sort(matching_paths.begin(), matching_paths.end());
      matching_paths.erase(
        std::unique(matching_paths.begin(), matching_paths.end()),
        matching_paths.end()
      );
      if (enumeration_failure) {
        return {{}, recovery_failure_e::transient, std::move(*enumeration_failure)};
      }
      if (matching_paths.empty()) {
        return {{}, recovery_failure_e::unavailable, "no matching HID interface is present"};
      }
      if (matching_paths.size() != 1) {
        return {{}, recovery_failure_e::unavailable, "multiple matching HID interfaces are present; refusing an ambiguous binding"};
      }

      win_handle_t device(CreateFileW(
        matching_paths.front().c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
        nullptr
      ));
      if (!device.valid()) {
        return {{}, recovery_failure_e::transient, windows_error("exclusive RayNeo HID open", GetLastError())};
      }

      HIDD_ATTRIBUTES attributes {};
      attributes.Size = sizeof(attributes);
      if (!HidD_GetAttributes(device.get(), &attributes)) {
        return {{}, recovery_failure_e::transient, windows_error("HidD_GetAttributes", GetLastError())};
      }
      if (attributes.VendorID != rayneo_vendor_id || attributes.ProductID != rayneo_product_id) {
        return {{}, recovery_failure_e::unavailable, "the opened HID interface changed identity"};
      }

      preparsed_data_t preparsed;
      if (!HidD_GetPreparsedData(device.get(), preparsed.put())) {
        return {{}, recovery_failure_e::transient, windows_error("HidD_GetPreparsedData", GetLastError())};
      }
      HIDP_CAPS caps {};
      if (HidP_GetCaps(preparsed.get(), &caps) != HIDP_STATUS_SUCCESS) {
        return {{}, recovery_failure_e::transient, "HidP_GetCaps could not read the RayNeo HID descriptor"};
      }
      if (caps.UsagePage != rayneo_usage_page || caps.Usage != rayneo_usage || caps.InputReportByteLength != report_size || caps.OutputReportByteLength != report_size) {
        return {{}, recovery_failure_e::unavailable, "the RayNeo HID descriptor does not match the supported protocol"};
      }

      return {opened_device_t {std::move(device)}, recovery_failure_e::transient, {}};
    }

    std::array<std::uint8_t, report_size> command_report(std::uint8_t command) {
      std::array<std::uint8_t, report_size> report {};
      report[report_id_offset] = vendor_report_id;
      report[vendor_payload_offset] = command_prefix;
      report[vendor_payload_offset + 1] = command;
      return report;
    }

    io_result_e write_command(
      HANDLE device,
      HANDLE stop_event,
      HANDLE control_event,
      std::uint8_t command,
      std::chrono::milliseconds timeout,
      bool honor_stop,
      bool honor_control,
      DWORD &error
    ) {
      if (honor_stop && WaitForSingleObject(stop_event, 0) == WAIT_OBJECT_0) {
        return io_result_e::stopped;
      }
      if (honor_control && WaitForSingleObject(control_event, 0) == WAIT_OBJECT_0) {
        return io_result_e::interrupted;
      }
      const auto report = command_report(command);
      overlapped_io_t operation;
      auto result = operation.start_write(device, report);
      if (result == io_result_e::pending) {
        result = operation.wait(
          device,
          stop_event,
          control_event,
          timeout,
          honor_stop,
          honor_control
        );
      }
      if (result == io_result_e::timeout || result == io_result_e::stopped || result == io_result_e::interrupted || result == io_result_e::error) {
        error = operation.error();
        operation.cancel_and_drain(device);
        return result;
      }
      if (honor_stop && WaitForSingleObject(stop_event, 0) == WAIT_OBJECT_0) {
        return io_result_e::stopped;
      }
      if (honor_control && WaitForSingleObject(control_event, 0) == WAIT_OBJECT_0) {
        return io_result_e::interrupted;
      }
      if (result != io_result_e::complete || operation.transferred() != report.size()) {
        error = ERROR_WRITE_FAULT;
        return io_result_e::error;
      }
      error = ERROR_SUCCESS;
      return io_result_e::complete;
    }

    enum class protocol_result_e {
      complete,
      stopped,
      interrupted,
      stale,
      unsupported,
      failed,
    };

    struct protocol_outcome_t {
      protocol_result_e result = protocol_result_e::failed;
      std::string failure;
      bool had_valid_report = false;
    };

    protocol_outcome_t authenticate_board(
      HANDLE device,
      HANDLE stop_event,
      HANDLE control_event
    ) {
      HidD_FlushQueue(device);

      DWORD error = ERROR_SUCCESS;
      const auto write_result = write_command(
        device,
        stop_event,
        control_event,
        board_info_command,
        command_timeout,
        true,
        true,
        error
      );
      if (write_result == io_result_e::stopped) {
        return {protocol_result_e::stopped, {}};
      }
      if (write_result == io_result_e::interrupted) {
        return {protocol_result_e::interrupted, {}};
      }
      if (write_result != io_result_e::complete) {
        return {protocol_result_e::failed, windows_error("RayNeo board-info query", error)};
      }

      overlapped_io_t read_operation;
      if (!read_operation.valid()) {
        return {protocol_result_e::failed, windows_error("RayNeo read event creation", read_operation.error())};
      }
      std::array<std::uint8_t, report_size> report {};
      const auto deadline = std::chrono::steady_clock::now() + board_response_timeout;
      while (std::chrono::steady_clock::now() < deadline) {
        if (WaitForSingleObject(stop_event, 0) == WAIT_OBJECT_0) {
          return {protocol_result_e::stopped, {}};
        }
        if (WaitForSingleObject(control_event, 0) == WAIT_OBJECT_0) {
          return {protocol_result_e::interrupted, {}};
        }
        auto read_result = read_operation.start_read(device, report);
        if (read_result == io_result_e::pending) {
          const auto now = std::chrono::steady_clock::now();
          const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - now
          );
          read_result = read_operation.wait(
            device,
            stop_event,
            control_event,
            std::max(remaining, 0ms),
            true,
            true
          );
        }
        if (read_result == io_result_e::complete && WaitForSingleObject(stop_event, 0) == WAIT_OBJECT_0) {
          return {protocol_result_e::stopped, {}};
        }
        if (read_result == io_result_e::complete && WaitForSingleObject(control_event, 0) == WAIT_OBJECT_0) {
          return {protocol_result_e::interrupted, {}};
        }
        if (read_result == io_result_e::stopped) {
          read_operation.cancel_and_drain(device);
          return {protocol_result_e::stopped, {}};
        }
        if (read_result == io_result_e::interrupted) {
          read_operation.cancel_and_drain(device);
          return {protocol_result_e::interrupted, {}};
        }
        if (read_result == io_result_e::timeout) {
          read_operation.cancel_and_drain(device);
          return {protocol_result_e::failed, "the RayNeo board-info response timed out"};
        }
        if (read_result != io_result_e::complete) {
          const auto read_error = read_operation.error();
          read_operation.cancel_and_drain(device);
          return {protocol_result_e::failed, windows_error("RayNeo board-info read", read_error)};
        }
        if (read_operation.transferred() != report.size()) {
          continue;
        }
        if (board_info_matches(report)) {
          return {protocol_result_e::complete, {}};
        }
        if (board_info_acknowledges_query(report)) {
          return {protocol_result_e::unsupported, "the RayNeo board-info response named an unsupported board"};
        }
      }
      return {protocol_result_e::failed, "the RayNeo board-info response timed out"};
    }

    std::optional<std::wstring_view> notification_interface_path(
      PCM_NOTIFY_EVENT_DATA event_data,
      DWORD event_data_size
    ) noexcept {
      constexpr auto path_offset = static_cast<std::size_t>(
        FIELD_OFFSET(CM_NOTIFY_EVENT_DATA, u.DeviceInterface.SymbolicLink)
      );
      if (!event_data || event_data_size < path_offset + sizeof(wchar_t) || event_data->FilterType != CM_NOTIFY_FILTER_TYPE_DEVICEINTERFACE) {
        return std::nullopt;
      }

      const auto capacity = (event_data_size - path_offset) / sizeof(wchar_t);
      const auto *path = event_data->u.DeviceInterface.SymbolicLink;
      std::size_t length = 0;
      while (length < capacity && path[length] != L'\0') {
        ++length;
      }
      if (length == 0 || length == capacity) {
        return std::nullopt;
      }
      return std::wstring_view(path, length);
    }

    const char *wear_state_name(wear_state_e state) {
      switch (state) {
        case wear_state_e::off_head:
          return "off-head";
        case wear_state_e::worn:
          return "worn";
        default:
          return "unknown";
      }
    }
  }  // namespace

  class wear_monitor_t::impl_t {
  public:
    impl_t():
        stop_event_(CreateEventW(nullptr, TRUE, FALSE, nullptr)),
        control_event_(CreateEventW(nullptr, FALSE, FALSE, nullptr)),
        start_event_(CreateEventW(nullptr, TRUE, FALSE, nullptr)) {
      if (!stop_event_.valid() || !control_event_.valid() || !start_event_.valid()) {
        BOOST_LOG(error) << "Could not create the RayNeo wear-monitor control events: "sv
                         << GetLastError();
        return;
      }
      worker_ = std::jthread([this](std::stop_token stop_token) {
        std::stop_callback stop_callback(stop_token, [this]() {
          SetEvent(stop_event_.get());
        });
        const std::array startup_handles {stop_event_.get(), start_event_.get()};
        if (WaitForMultipleObjects(static_cast<DWORD>(startup_handles.size()), startup_handles.data(), FALSE, INFINITE) != WAIT_OBJECT_0 + 1) {
          return;
        }
        run(stop_token);
      });
      try {
        // The worker is held behind start_event_ so registration always precedes its initial
        // SetupAPI snapshot, while a thread-construction failure cannot leave callbacks behind.
        register_notifications();
      } catch (...) {
        shutdown_notifications();
        throw;
      }
      SetEvent(start_event_.get());
    }

    ~impl_t() {
      worker_.request_stop();
      if (stop_event_.valid()) {
        SetEvent(stop_event_.get());
      }
      if (control_event_.valid()) {
        SetEvent(control_event_.get());
      }
      if (start_event_.valid()) {
        SetEvent(start_event_.get());
      }
      shutdown_notifications();
      if (worker_.joinable()) {
        worker_.join();
      }
    }

    [[nodiscard]] wear_state_e state() const noexcept {
      return state_.load(std::memory_order_acquire);
    }

  private:
    enum class wait_result_e {
      timeout,
      notified,
      stopped,
      failed,
    };

    class callback_scope_t {
    public:
      explicit callback_scope_t(impl_t *owner) noexcept:
          owner_(owner) {
        AcquireSRWLockExclusive(&owner_->callback_lock_);
        if (!owner_->callbacks_closing_) {
          ++owner_->active_callbacks_;
          entered_ = true;
        }
        ReleaseSRWLockExclusive(&owner_->callback_lock_);
      }

      ~callback_scope_t() {
        if (!entered_) {
          return;
        }
        AcquireSRWLockExclusive(&owner_->callback_lock_);
        --owner_->active_callbacks_;
        if (owner_->callbacks_closing_ && owner_->active_callbacks_ == 0) {
          WakeAllConditionVariable(&owner_->callback_condition_);
        }
        ReleaseSRWLockExclusive(&owner_->callback_lock_);
      }

      callback_scope_t(const callback_scope_t &) = delete;
      callback_scope_t &operator=(const callback_scope_t &) = delete;

      explicit operator bool() const noexcept {
        return entered_;
      }

    private:
      impl_t *owner_;
      bool entered_ = false;
    };

    static DWORD CALLBACK device_notification_callback(
      HCMNOTIFICATION,
      PVOID context,
      CM_NOTIFY_ACTION action,
      PCM_NOTIFY_EVENT_DATA event_data,
      DWORD event_data_size
    ) noexcept {
      if (!context || (action != CM_NOTIFY_ACTION_DEVICEINTERFACEARRIVAL && action != CM_NOTIFY_ACTION_DEVICEINTERFACEREMOVAL)) {
        return ERROR_SUCCESS;
      }
      const auto path = notification_interface_path(event_data, event_data_size);
      if (!path || !is_rayneo_interface_path(*path)) {
        return ERROR_SUCCESS;
      }

      auto *self = static_cast<impl_t *>(context);
      callback_scope_t callback_scope(self);
      if (!callback_scope) {
        return ERROR_SUCCESS;
      }

      std::uint32_t reason = lifecycle_none;
      if (action == CM_NOTIFY_ACTION_DEVICEINTERFACEARRIVAL) {
        reason = lifecycle_interface_arrival;
        // An exact HID arrival is authoritative evidence that the device stack is awake. It also
        // prevents a missed power-resume callback from leaving recovery latched off forever.
        self->power_suspended_.store(false, std::memory_order_release);
      } else if (action == CM_NOTIFY_ACTION_DEVICEINTERFACEREMOVAL) {
        reason = lifecycle_interface_removal;
      }
      if (reason != lifecycle_none) {
        self->signal_notification(reason);
      }
      return ERROR_SUCCESS;
    }

    static ULONG CALLBACK power_notification_callback(
      PVOID context,
      ULONG notification,
      PVOID
    ) noexcept {
      if (!context) {
        return ERROR_SUCCESS;
      }
      const auto reason = power_notification_reason(notification);
      if (reason == lifecycle_none) {
        return ERROR_SUCCESS;
      }

      auto *self = static_cast<impl_t *>(context);
      callback_scope_t callback_scope(self);
      if (!callback_scope) {
        return ERROR_SUCCESS;
      }
      if (reason == lifecycle_suspend) {
        self->power_suspended_.store(true, std::memory_order_release);
        self->signal_notification(reason);
      } else {
        const auto was_suspended = self->power_suspended_.exchange(
          false,
          std::memory_order_acq_rel
        );
        // Windows sends AUTOMATIC for the resume itself and can deliver RESUMESUSPEND later when
        // the user becomes active. Ignore that normal follow-up, but use it as a wake fallback if
        // AUTOMATIC was missed and the monitor is still gated by the earlier suspend.
        if (power_notification_requests_recovery(notification, was_suspended)) {
          self->signal_notification(reason);
        }
      }
      return ERROR_SUCCESS;
    }

    void signal_notification(std::uint32_t reason) noexcept {
      pending_notifications_.fetch_or(reason, std::memory_order_release);
      SetEvent(control_event_.get());
    }

    void register_notifications() {
      GUID hid_guid {};
      HidD_GetHidGuid(&hid_guid);
      CM_NOTIFY_FILTER filter {};
      filter.cbSize = sizeof(filter);
      filter.FilterType = CM_NOTIFY_FILTER_TYPE_DEVICEINTERFACE;
      filter.u.DeviceInterface.ClassGuid = hid_guid;
      const auto device_result = CM_Register_Notification(
        &filter,
        this,
        device_notification_callback,
        &device_notification_
      );
      device_notifications_registered_ = device_result == CR_SUCCESS;
      if (!device_notifications_registered_) {
        device_notification_ = nullptr;
        BOOST_LOG(warning) << "Could not register RayNeo HID arrival notifications (ConfigMgr result "sv
                           << device_result << "); using a one-second discovery fallback."sv;
      }

      power_subscription_.callback = power_notification_callback;
      power_subscription_.context = this;
      const auto power_result = PowerRegisterSuspendResumeNotification(
        power_device_notify_callback,
        reinterpret_cast<HANDLE>(&power_subscription_),
        &power_notification_
      );
      if (power_result != ERROR_SUCCESS) {
        power_notification_ = nullptr;
        BOOST_LOG(warning) << "Could not register RayNeo suspend/resume notifications: "sv
                           << power_result << ". Stale-stream recovery remains active."sv;
      }
    }

    void unregister_notifications() noexcept {
      if (power_notification_) {
        PowerUnregisterSuspendResumeNotification(power_notification_);
        power_notification_ = nullptr;
      }
      if (device_notification_) {
        CM_Unregister_Notification(device_notification_);
        device_notification_ = nullptr;
      }
    }

    void shutdown_notifications() noexcept {
      AcquireSRWLockExclusive(&callback_lock_);
      callbacks_closing_ = true;
      ReleaseSRWLockExclusive(&callback_lock_);

      unregister_notifications();

      AcquireSRWLockExclusive(&callback_lock_);
      while (active_callbacks_ != 0) {
        SleepConditionVariableSRW(
          &callback_condition_,
          &callback_lock_,
          INFINITE,
          0
        );
      }
      ReleaseSRWLockExclusive(&callback_lock_);
    }

    void publish_state(wear_state_e state) {
      const auto previous = state_.exchange(state, std::memory_order_acq_rel);
      if (previous != state) {
        BOOST_LOG(info) << "RayNeo wear sensor state: "sv << wear_state_name(state) << '.';
      }
    }

    void log_retry(std::string_view failure) {
      const auto now = std::chrono::steady_clock::now();
      if (now < next_failure_log_) {
        return;
      }
      BOOST_LOG(warning) << "RayNeo wear monitor is unavailable: "sv << failure
                         << "; retrying discovery."sv;
      next_failure_log_ = now + failure_log_interval;
    }

    wait_result_e wait_for_control(
      std::stop_token stop_token,
      std::optional<std::chrono::milliseconds> timeout
    ) const {
      if (stop_token.stop_requested()) {
        return wait_result_e::stopped;
      }
      if (pending_notifications_.load(std::memory_order_acquire) != lifecycle_none) {
        return wait_result_e::notified;
      }
      const std::array handles {stop_event_.get(), control_event_.get()};
      const auto result = WaitForMultipleObjects(
        static_cast<DWORD>(handles.size()),
        handles.data(),
        FALSE,
        wait_timeout(timeout)
      );
      if (result == WAIT_OBJECT_0) {
        return wait_result_e::stopped;
      }
      if (result == WAIT_OBJECT_0 + 1) {
        return wait_result_e::notified;
      }
      if (result == WAIT_TIMEOUT) {
        return wait_result_e::timeout;
      }
      return wait_result_e::failed;
    }

    [[nodiscard]] bool lifecycle_invalidated() const noexcept {
      return power_suspended_.load(std::memory_order_acquire) ||
             pending_notifications_.load(std::memory_order_acquire) != lifecycle_none;
    }

    protocol_outcome_t monitor_sensor_stream(
      HANDLE device,
      std::stop_token stop_token
    ) {
      if (lifecycle_invalidated()) {
        return {protocol_result_e::interrupted, {}};
      }
      DWORD error = ERROR_SUCCESS;
      const auto start_result = write_command(
        device,
        stop_event_.get(),
        control_event_.get(),
        sensor_start_command,
        command_timeout,
        true,
        true,
        error
      );
      if (start_result == io_result_e::stopped) {
        return {protocol_result_e::stopped, {}};
      }
      if (start_result == io_result_e::interrupted) {
        return {protocol_result_e::interrupted, {}};
      }
      if (start_result != io_result_e::complete) {
        return {protocol_result_e::failed, windows_error("RayNeo sensor-start command", error)};
      }
      if (lifecycle_invalidated()) {
        return {protocol_result_e::interrupted, {}};
      }

      publish_state(wear_state_e::unknown);
      wear_debounce_t debounce;
      overlapped_io_t read_operation;
      std::array<std::uint8_t, report_size> report {};
      protocol_outcome_t outcome {protocol_result_e::failed, "the RayNeo sensor stream stopped"};
      auto valid_report_deadline = std::chrono::steady_clock::now() + stale_timeout;

      if (!read_operation.valid()) {
        outcome.failure = windows_error("RayNeo sensor read event creation", read_operation.error());
      } else {
        while (!stop_token.stop_requested()) {
          if (lifecycle_invalidated()) {
            outcome = {protocol_result_e::interrupted, {}, outcome.had_valid_report};
            break;
          }
          const auto before_read = std::chrono::steady_clock::now();
          if (before_read >= valid_report_deadline) {
            if (const auto stale = debounce.mark_stale(before_read)) {
              publish_state(*stale);
            }
            BOOST_LOG(debug) << "RayNeo wear sensor received no valid report for "sv
                             << stale_timeout.count()
                             << " ms; reopening its push subscription."sv;
            outcome = {
              protocol_result_e::stale,
              "the RayNeo sensor stream became stale",
              outcome.had_valid_report,
            };
            break;
          }

          if (WaitForSingleObject(control_event_.get(), 0) == WAIT_OBJECT_0) {
            outcome = {protocol_result_e::interrupted, {}, outcome.had_valid_report};
            break;
          }

          auto read_result = read_operation.start_read(device, report);
          if (read_result == io_result_e::pending) {
            const auto now = std::chrono::steady_clock::now();
            const auto read_timeout = std::max(
              std::chrono::ceil<std::chrono::milliseconds>(valid_report_deadline - now),
              0ms
            );
            read_result = read_operation.wait(
              device,
              stop_event_.get(),
              control_event_.get(),
              read_timeout,
              true,
              true
            );
          }
          if (read_result == io_result_e::complete && WaitForSingleObject(control_event_.get(), 0) == WAIT_OBJECT_0) {
            outcome = {protocol_result_e::interrupted, {}, outcome.had_valid_report};
            break;
          }
          if (read_result == io_result_e::stopped) {
            outcome = {protocol_result_e::stopped, {}};
            break;
          }
          if (read_result == io_result_e::interrupted) {
            read_operation.cancel_and_drain(device);
            outcome = {protocol_result_e::interrupted, {}, outcome.had_valid_report};
            break;
          }
          if (read_result == io_result_e::timeout) {
            read_operation.cancel_and_drain(device);
            if (const auto stale = debounce.mark_stale(std::chrono::steady_clock::now())) {
              publish_state(*stale);
            }
            BOOST_LOG(debug) << "RayNeo wear sensor received no valid report for "sv
                             << stale_timeout.count()
                             << " ms; reopening its push subscription."sv;
            outcome = {
              protocol_result_e::stale,
              "the RayNeo sensor stream became stale",
              outcome.had_valid_report,
            };
            break;
          }
          if (read_result != io_result_e::complete) {
            outcome.failure = windows_error("RayNeo sensor read", read_operation.error());
            break;
          }
          if (read_operation.transferred() != report.size()) {
            continue;
          }
          const auto sample = parse_sensor_report(report);
          if (!sample) {
            continue;
          }
          const auto received_at = std::chrono::steady_clock::now();
          outcome.had_valid_report = true;
          valid_report_deadline = received_at + stale_timeout;
          if (const auto changed = debounce.observe(sample->tick, sample->state, received_at)) {
            if (*changed == wear_state_e::unknown) {
              BOOST_LOG(debug) << "RayNeo wear proximity remained ambiguous at "sv
                               << sample->proximity << "."sv;
            }
            publish_state(*changed);
          }
        }
      }

      read_operation.cancel_and_drain(device);
      const bool lifecycle_invalidated =
        outcome.result == protocol_result_e::interrupted ||
        this->lifecycle_invalidated();
      if (!lifecycle_invalidated) {
        DWORD ignored_error = ERROR_SUCCESS;
        write_command(
          device,
          stop_event_.get(),
          control_event_.get(),
          sensor_stop_command,
          stop_command_timeout,
          false,
          false,
          ignored_error
        );
      }
      publish_state(wear_state_e::unknown);
      return outcome;
    }

    void run(std::stop_token stop_token) {
      unsigned consecutive_failures = 0;
      while (!stop_token.stop_requested()) {
        const auto notifications = pending_notifications_.exchange(
          lifecycle_none,
          std::memory_order_acq_rel
        );
        // Consume the auto-reset signal represented by the exchanged bits. A concurrent callback
        // keeps its own pending bit, which wait_for_control() checks before blocking.
        WaitForSingleObject(control_event_.get(), 0);
        if (power_suspended_.load(std::memory_order_acquire)) {
          publish_state(wear_state_e::unknown);
          const auto wait = wait_for_control(stop_token, std::nullopt);
          if (wait == wait_result_e::stopped || wait == wait_result_e::failed) {
            break;
          }
          consecutive_failures = 0;
          continue;
        }

        std::chrono::milliseconds settle_delay {0};
        if ((notifications & lifecycle_resume) != 0) {
          settle_delay = system_resume_settle;
        } else if ((notifications & lifecycle_interface_arrival) != 0) {
          settle_delay = device_arrival_settle;
        }
        if (settle_delay > 0ms) {
          const auto wait = wait_for_control(stop_token, settle_delay);
          if (wait == wait_result_e::stopped || wait == wait_result_e::failed) {
            break;
          }
          if (wait == wait_result_e::notified || lifecycle_invalidated()) {
            continue;
          }
        }

        auto discovery = discover_unique_device();
        if (!discovery.device) {
          publish_state(wear_state_e::unknown);
          log_retry(discovery.failure);
          const auto delay = recovery_retry_delay(
            discovery.failure_kind,
            device_notifications_registered_,
            consecutive_failures
          );
          if (discovery.failure_kind == recovery_failure_e::transient) {
            advance_recovery_failure(consecutive_failures);
          } else {
            consecutive_failures = 0;
          }
          const auto wait = wait_for_control(stop_token, delay);
          if (wait == wait_result_e::stopped || wait == wait_result_e::failed) {
            break;
          }
          if (wait == wait_result_e::notified) {
            consecutive_failures = 0;
          }
          continue;
        }
        if (lifecycle_invalidated()) {
          discovery.device.reset();
          consecutive_failures = 0;
          continue;
        }

        const auto authentication = authenticate_board(
          discovery.device->handle.get(),
          stop_event_.get(),
          control_event_.get()
        );
        if (authentication.result == protocol_result_e::stopped) {
          break;
        }
        if (authentication.result == protocol_result_e::interrupted) {
          discovery.device.reset();
          consecutive_failures = 0;
          continue;
        }
        if (authentication.result == protocol_result_e::unsupported) {
          publish_state(wear_state_e::unknown);
          BOOST_LOG(warning) << "RayNeo wear monitor rejected this HID incarnation: "sv
                             << authentication.failure
                             << "; deferring authentication until a device/power transition or the safety watchdog."sv;
          discovery.device.reset();
          consecutive_failures = 0;
          const auto wait = wait_for_control(stop_token, notification_watchdog_interval);
          if (wait == wait_result_e::stopped || wait == wait_result_e::failed) {
            break;
          }
          continue;
        }
        if (authentication.result != protocol_result_e::complete) {
          publish_state(wear_state_e::unknown);
          log_retry(authentication.failure);
          discovery.device.reset();
          const auto delay = recovery_retry_delay(
            recovery_failure_e::transient,
            device_notifications_registered_,
            consecutive_failures
          );
          advance_recovery_failure(consecutive_failures);
          const auto wait = wait_for_control(stop_token, delay);
          if (wait == wait_result_e::stopped || wait == wait_result_e::failed) {
            break;
          }
          if (wait == wait_result_e::notified) {
            consecutive_failures = 0;
          }
          continue;
        }
        if (lifecycle_invalidated()) {
          discovery.device.reset();
          consecutive_failures = 0;
          continue;
        }

        if (!ever_authenticated_) {
          BOOST_LOG(info) << "RayNeo wear monitor authenticated the supported HID board."sv;
          ever_authenticated_ = true;
        }
        const auto stream = monitor_sensor_stream(
          discovery.device->handle.get(),
          stop_token
        );
        if (stream.result == protocol_result_e::stopped) {
          break;
        }
        if (stream.result == protocol_result_e::interrupted) {
          discovery.device.reset();
          consecutive_failures = 0;
          continue;
        }
        if (stream.had_valid_report) {
          consecutive_failures = 0;
        }
        log_retry(stream.failure);
        discovery.device.reset();
        const auto delay = recovery_retry_delay(
          recovery_failure_e::transient,
          device_notifications_registered_,
          consecutive_failures
        );
        advance_recovery_failure(consecutive_failures);
        const auto wait = wait_for_control(stop_token, delay);
        if (wait == wait_result_e::stopped || wait == wait_result_e::failed) {
          break;
        }
        if (wait == wait_result_e::notified) {
          consecutive_failures = 0;
        }
      }
      publish_state(wear_state_e::unknown);
    }

    std::atomic<wear_state_e> state_ {wear_state_e::unknown};
    std::atomic<std::uint32_t> pending_notifications_ {lifecycle_none};
    std::atomic_bool power_suspended_ {false};
    std::chrono::steady_clock::time_point next_failure_log_ {};
    bool ever_authenticated_ = false;
    bool device_notifications_registered_ = false;
    HCMNOTIFICATION device_notification_ = nullptr;
    HPOWERNOTIFY power_notification_ = nullptr;
    power_notify_subscribe_parameters_t power_subscription_ {};
    SRWLOCK callback_lock_ = SRWLOCK_INIT;
    CONDITION_VARIABLE callback_condition_ = CONDITION_VARIABLE_INIT;
    bool callbacks_closing_ = false;
    unsigned active_callbacks_ = 0;
    win_handle_t stop_event_;
    win_handle_t control_event_;
    win_handle_t start_event_;
    std::jthread worker_;
  };

  wear_monitor_t::wear_monitor_t():
      impl_(std::make_unique<impl_t>()) {
  }

  wear_monitor_t::~wear_monitor_t() = default;

  wear_state_e wear_monitor_t::state() const noexcept {
    return impl_ ? impl_->state() : wear_state_e::unknown;
  }

  std::unique_ptr<wear_monitor_t> create_wear_monitor(std::string_view display_model_id) {
    if (display_model_id != supported_display_model) {
      return {};
    }
    return std::make_unique<wear_monitor_t>();
  }

#ifdef SUNSHINE_TESTS
  std::optional<detail::parsed_sensor_report_t> detail::parse_sensor_report_for_test(
    std::span<const std::uint8_t> report
  ) {
    const auto parsed = parse_sensor_report(report);
    if (!parsed) {
      return std::nullopt;
    }
    return detail::parsed_sensor_report_t {
      parsed->tick,
      parsed->proximity,
      parsed->state,
    };
  }

  bool detail::board_info_matches_for_test(std::span<const std::uint8_t> report) {
    return board_info_matches(report);
  }

  bool detail::hid_interface_matches_for_test(std::wstring_view path) {
    return is_rayneo_interface_path(path);
  }

  detail::power_event_e detail::power_event_for_test(std::uint32_t notification) {
    switch (power_notification_reason(notification)) {
      case lifecycle_suspend:
        return power_event_e::suspend;
      case lifecycle_resume:
        return power_event_e::resume;
      default:
        return power_event_e::none;
    }
  }

  bool detail::power_event_requests_recovery_for_test(
    std::uint32_t notification,
    bool was_suspended
  ) {
    return power_notification_requests_recovery(notification, was_suspended);
  }

  std::chrono::milliseconds detail::recovery_retry_delay_for_test(
    bool device_unavailable,
    bool notifications_available,
    unsigned consecutive_failures
  ) {
    return recovery_retry_delay(
      device_unavailable ? recovery_failure_e::unavailable : recovery_failure_e::transient,
      notifications_available,
      consecutive_failures
    );
  }

  wear_state_e detail::debounce_observations_for_test(
    std::span<const debounce_observation_t> observations,
    std::chrono::milliseconds evaluated_at
  ) {
    wear_debounce_t debounce;
    const wear_debounce_t::clock_t::time_point origin {};
    for (const auto &observation : observations) {
      if (observation.at > evaluated_at) {
        break;
      }
      debounce.mark_stale(origin + observation.at);
      debounce.observe(
        observation.tick,
        observation.state,
        origin + observation.at
      );
    }
    debounce.mark_stale(origin + evaluated_at);
    return debounce.state();
  }
#endif
}  // namespace ar_glasses::rayneo
