/**
 * @file src/offline_sbs_worker_io.cpp
 * @brief Worker I/O: UTF-8 paths, atomic files, and bounded FFmpeg/FFprobe child processes.
 */

#include "offline_sbs_worker_io.h"

#include "offline_sbs_digest.h"
#include "offline_sbs_wire_contract.h"
#include "offline_sbs_worker.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <limits>
#include <openssl/rand.h>
#include <thread>
#include <utility>

namespace offline_sbs {
  namespace fs = std::filesystem;
  using namespace std::chrono_literals;
  using namespace worker_detail;

  namespace {
    std::wstring quote_windows_argument(const std::wstring_view argument) {
      std::wstring result;
      result.push_back(L'"');
      std::size_t backslashes = 0;
      for (const wchar_t character : argument) {
        if (character == L'\\') {
          ++backslashes;
          continue;
        }
        if (character == L'"') {
          result.append(backslashes * 2 + 1, L'\\');
          result.push_back(L'"');
          backslashes = 0;
          continue;
        }
        result.append(backslashes, L'\\');
        backslashes = 0;
        result.push_back(character);
      }
      result.append(backslashes * 2, L'\\');
      result.push_back(L'"');
      return result;
    }

    constexpr std::size_t max_child_log_bytes =
      8ull * 1024ull * 1024ull;
    constexpr std::size_t child_log_marker_reserve = 128;
    static_assert(max_child_log_bytes > child_log_marker_reserve + 1024);

    class bounded_log_accumulator_t {
    public:
      explicit bounded_log_accumulator_t(
        const std::size_t max_output_bytes = max_child_log_bytes
      ):
          max_output_bytes_(max_output_bytes),
          payload_capacity_(
            max_output_bytes > child_log_marker_reserve ?
              max_output_bytes - child_log_marker_reserve :
              0
          ),
          prefix_capacity_(
            std::min<std::size_t>(64ull * 1024ull, payload_capacity_ / 8)
          ),
          tail_capacity_(payload_capacity_ - prefix_capacity_),
          tail_(tail_capacity_) {
        if (tail_capacity_ == 0) {
          throw worker_error("bounded child log capacity is invalid");
        }
        prefix_.reserve(prefix_capacity_);
      }

      void append(const char *bytes, std::size_t size) {
        if (!bytes && size != 0) {
          throw worker_error("bounded child log received a null payload");
        }
        if (size == 0) {
          return;
        }
        total_bytes_ += size;
        const auto prefix_bytes =
          std::min<std::size_t>(size, prefix_capacity_ - prefix_.size());
        prefix_.insert(
          prefix_.end(),
          bytes,
          bytes + prefix_bytes
        );
        bytes += prefix_bytes;
        size -= prefix_bytes;

        while (size != 0) {
          const auto chunk = std::min<std::size_t>(
            size,
            tail_capacity_ - tail_write_
          );
          std::memcpy(tail_.data() + tail_write_, bytes, chunk);
          tail_write_ = (tail_write_ + chunk) % tail_capacity_;
          tail_size_ = std::min<std::size_t>(
            tail_capacity_,
            tail_size_ + chunk
          );
          bytes += chunk;
          size -= chunk;
        }
      }

      [[nodiscard]] std::vector<std::uint8_t> render() const {
        const auto retained_payload = prefix_.size() + tail_size_;
        const auto dropped =
          total_bytes_ > retained_payload ?
            total_bytes_ - retained_payload :
            0;
        const auto marker =
          dropped == 0 ?
            std::string {} :
            "\n[Sunshine 3D: " + std::to_string(dropped) +
              " child-log bytes omitted; showing prefix and tail]\n";
        if (marker.size() > child_log_marker_reserve) {
          throw worker_error("bounded child log marker reserve is too small");
        }

        std::vector<std::uint8_t> output;
        output.reserve(prefix_.size() + marker.size() + tail_size_);
        output.insert(output.end(), prefix_.begin(), prefix_.end());
        output.insert(output.end(), marker.begin(), marker.end());
        if (tail_size_ != 0) {
          const auto oldest =
            tail_size_ == tail_capacity_ ? tail_write_ : 0;
          const auto first = std::min<std::size_t>(
            tail_size_,
            tail_capacity_ - oldest
          );
          output.insert(
            output.end(),
            tail_.begin() + static_cast<std::ptrdiff_t>(oldest),
            tail_.begin() + static_cast<std::ptrdiff_t>(oldest + first)
          );
          output.insert(
            output.end(),
            tail_.begin(),
            tail_.begin() +
              static_cast<std::ptrdiff_t>(tail_size_ - first)
          );
        }
        if (output.size() > max_output_bytes_) {
          throw worker_error("bounded child log exceeded its output contract");
        }
        return output;
      }

    private:
      std::size_t max_output_bytes_ = 0;
      std::size_t payload_capacity_ = 0;
      std::size_t prefix_capacity_ = 0;
      std::size_t tail_capacity_ = 0;
      std::vector<std::uint8_t> prefix_;
      std::vector<std::uint8_t> tail_;
      std::size_t tail_write_ = 0;
      std::size_t tail_size_ = 0;
      std::uint64_t total_bytes_ = 0;
    };

    [[nodiscard]] bool native_stdout_pipe_error_is_eof(
      const DWORD error
    ) noexcept {
      // A child can close its inherited stdout handle just before its process handle
      // becomes signaled.  Pipe closure is EOF regardless of that short-lived process
      // state; the caller still waits for and validates the child's exit status.
      return error == ERROR_BROKEN_PIPE || error == ERROR_NO_DATA;
    }

    std::wstring command_line(const std::vector<std::string> &arguments) {
      std::wstring result;
      for (const auto &argument : arguments) {
        if (!result.empty()) {
          result.push_back(L' ');
        }
        result += quote_windows_argument(path_from_utf8(argument).wstring());
      }
      return result;
    }

    std::wstring private_pipe_name() {
      std::array<unsigned char, 32> random {};
      if (RAND_bytes(random.data(), static_cast<int>(random.size())) != 1)
        throw worker_error("cannot create private media pipe identity");
      const auto token = sha256_hex(std::string_view(reinterpret_cast<const char *>(random.data()), random.size()));
      return L"\\\\.\\pipe\\sunshine-offline-" + std::wstring(token.begin(), token.end());
    }
  }  // namespace

  namespace worker_detail {
    // Native separators: these strings go to FFmpeg command lines and Windows logs. The job
    // manager's path_to_utf8() is the generic-separator form for job records; do not merge them.
    std::string path_utf8(const fs::path &path) {
      const auto wide = path.wstring();
      if (wide.empty()) {
        return {};
      }
      const int size = WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        wide.data(),
        static_cast<int>(wide.size()),
        nullptr,
        0,
        nullptr,
        nullptr
      );
      if (size <= 0) {
        throw worker_error("cannot encode a filesystem path as UTF-8");
      }
      std::string result(static_cast<std::size_t>(size), '\0');
      if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), static_cast<int>(wide.size()), result.data(), size, nullptr, nullptr) != size) {
        throw worker_error("cannot encode a filesystem path as UTF-8");
      }
      return result;
    }

    fs::path path_from_utf8(const std::string &value) {
      if (value.empty()) {
        return {};
      }
      const int size = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        value.data(),
        static_cast<int>(value.size()),
        nullptr,
        0
      );
      if (size <= 0) {
        throw worker_error("worker contract contains invalid UTF-8");
      }
      std::wstring wide(static_cast<std::size_t>(size), L'\0');
      if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), wide.data(), size) != size) {
        throw worker_error("worker contract contains invalid UTF-8");
      }
      return fs::path(wide);
    }

    std::string lower(std::string value) {
      std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](const unsigned char character) {
          return static_cast<char>(std::tolower(character));
        }
      );
      return value;
    }

    void write_bytes_atomic(
      const fs::path &path,
      const std::vector<std::uint8_t> &bytes,
      const publication_e publication
    ) {
      std::error_code ec;
      fs::create_directories(path.parent_path(), ec);
      if (ec) {
        throw worker_error("cannot create contract directory: " + ec.message());
      }
      std::string error;
      if (!safe_filesystem::publish_bytes_atomically(
            path,
            std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()),
            publication,
            error
          )) {
        throw worker_error(error);
      }
    }

    void write_json_atomic(
      const fs::path &path,
      const nlohmann::json &value,
      const publication_e publication
    ) {
      const auto text = value.dump(2) + "\n";
      write_bytes_atomic(
        path,
        std::vector<std::uint8_t>(text.begin(), text.end()),
        publication
      );
    }

    void write_json_atomic_bounded(
      const fs::path &path,
      const nlohmann::json &value,
      const std::uintmax_t max_bytes,
      const std::string_view description
    ) {
      const auto text = value.dump(2) + "\n";
      if (text.size() > max_bytes) {
        throw worker_error(
          std::string {description} + " exceeds its " +
          std::to_string(max_bytes) + "-byte serialized contract"
        );
      }
      write_bytes_atomic(
        path,
        std::vector<std::uint8_t>(text.begin(), text.end())
      );
    }

    [[maybe_unused]] void remove_file_checked(const fs::path &path) {
      std::error_code ec;
      if (!fs::remove(path, ec) || ec) {
        throw worker_error("cannot release temporary file: " + path_utf8(path));
      }
    }

    void write_text_atomic(const fs::path &path, const std::string &text) {
      write_bytes_atomic(
        path,
        std::vector<std::uint8_t>(text.begin(), text.end())
      );
    }

    class bounded_child_log_t {
    public:
      explicit bounded_child_log_t(fs::path path):
          path_(std::move(path)) {
        SECURITY_ATTRIBUTES inheritable {
          .nLength = sizeof(SECURITY_ATTRIBUTES),
          .lpSecurityDescriptor = nullptr,
          .bInheritHandle = TRUE,
        };
        if (
          !CreatePipe(
            &read_,
            &write_,
            &inheritable,
            1024 * 1024
          ) ||
          !SetHandleInformation(read_, HANDLE_FLAG_INHERIT, 0)
        ) {
          const auto error = GetLastError();
          close_handle(read_);
          close_handle(write_);
          throw worker_error(
            "cannot create bounded child-log pipe (Windows error " +
            std::to_string(error) + ")"
          );
        }
        try {
          drain_ = std::thread([this]() {
            drain_pipe();
          });
        } catch (...) {
          close_handle(read_);
          close_handle(write_);
          throw;
        }
      }

      bounded_child_log_t(const bounded_child_log_t &) = delete;
      bounded_child_log_t &operator=(const bounded_child_log_t &) = delete;

      ~bounded_child_log_t() {
        finish();
      }

      [[nodiscard]] HANDLE write_handle() const {
        return write_;
      }

      void close_parent_writer() noexcept {
        close_handle(write_);
      }

      void finish() noexcept {
        if (finished_) {
          return;
        }
        close_parent_writer();
        if (drain_.joinable()) {
          drain_.join();
        }
        close_handle(read_);
        try {
          if (read_error_ != ERROR_SUCCESS) {
            const auto note =
              "\n[Sunshine 3D: child-log pipe read failed with Windows error " +
              std::to_string(read_error_) + "]\n";
            accumulator_.append(note.data(), note.size());
          }
          write_bytes_atomic(path_, accumulator_.render());
        } catch (...) {
          // Process teardown is noexcept. A missing log must never strand a child/job; all
          // actual media/result contracts are published independently and remain authoritative.
        }
        finished_ = true;
      }

    private:
      static void close_handle(HANDLE &handle) noexcept {
        if (handle && handle != INVALID_HANDLE_VALUE) {
          CloseHandle(handle);
        }
        handle = nullptr;
      }

      void drain_pipe() noexcept {
        std::array<char, 64ull * 1024ull> buffer {};
        while (true) {
          DWORD received = 0;
          if (
            ReadFile(
              read_,
              buffer.data(),
              static_cast<DWORD>(buffer.size()),
              &received,
              nullptr
            )
          ) {
            if (received == 0) {
              return;
            }
            try {
              accumulator_.append(buffer.data(), received);
            } catch (...) {
              read_error_ = ERROR_NOT_ENOUGH_MEMORY;
              // Keep draining after an allocation failure so a verbose child cannot block
              // forever on a full pipe while its supervisor is trying to terminate it.
            }
            continue;
          }
          const auto error = GetLastError();
          if (error != ERROR_BROKEN_PIPE && error != ERROR_NO_DATA) {
            read_error_ = error;
          }
          return;
        }
      }

      fs::path path_;
      HANDLE read_ = nullptr;
      HANDLE write_ = nullptr;
      std::thread drain_;
      bounded_log_accumulator_t accumulator_;
      DWORD read_error_ = ERROR_SUCCESS;
      bool finished_ = false;
    };

    void validate_windows_command_line_capacity(
      const std::vector<std::string> &arguments,
      const std::string_view description
    ) {
      // CreateProcessW's 32,767-character limit includes the terminating NUL.
      constexpr std::size_t max_command_characters = 32766;
      if (command_line(arguments).size() > max_command_characters) {
        throw worker_error(
          std::string(description) +
          " exceeds the Windows command-line limit"
        );
      }
    }

    child_process_t::child_process_t() = default;

    child_process_t::child_process_t(child_process_t &&other) noexcept {
      *this = std::move(other);
    }

    child_process_t &child_process_t::operator=(child_process_t &&other) noexcept {
      if (this != &other) {
        terminate();
        process_ = std::exchange(other.process_, nullptr);
        thread_ = std::exchange(other.thread_, nullptr);
        stdout_read_ = std::exchange(other.stdout_read_, nullptr);
        stdout_read_event_ = std::exchange(other.stdout_read_event_, nullptr);
        stdout_child_write_ = std::exchange(other.stdout_child_write_, nullptr);
        stdout_file_ = std::exchange(other.stdout_file_, nullptr);
        log_capture_ = std::move(other.log_capture_);
        stdin_null_ = std::exchange(other.stdin_null_, nullptr);
        stdin_write_ = std::exchange(other.stdin_write_, nullptr);
        stdin_write_event_ = std::exchange(other.stdin_write_event_, nullptr);
        job_ = std::exchange(other.job_, nullptr);
      }
      return *this;
    }

    child_process_t::~child_process_t() {
      terminate();
    }

    child_process_t child_process_t::launch(
      const std::vector<std::string> &arguments,
      const fs::path &working_directory,
      const fs::path &log_path,
      const std::optional<fs::path> &stdout_path,
      const bool pipe_stdout,
      const bool pipe_stdin
    ) {
      if (arguments.empty()) {
        throw worker_error("cannot launch an empty command");
      }
      validate_windows_command_line_capacity(
        arguments,
        "native child command"
      );
      std::error_code ec;
      fs::create_directories(log_path.parent_path(), ec);
      if (ec) {
        throw worker_error("cannot create process-log directory");
      }

      child_process_t child;
      SECURITY_ATTRIBUTES inheritable {
        .nLength = sizeof(SECURITY_ATTRIBUTES),
        .lpSecurityDescriptor = nullptr,
        .bInheritHandle = TRUE,
      };
      child.log_capture_ =
        std::make_unique<bounded_child_log_t>(log_path);
      const HANDLE log_write = child.log_capture_->write_handle();

      HANDLE stdout_write = log_write;
      if (pipe_stdout) {
        const auto name = private_pipe_name();
        child.stdout_read_ = CreateNamedPipeW(name.c_str(),
          PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
          PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
          1, 1024 * 1024, 1024 * 1024, 0, nullptr);
        if (child.stdout_read_ == INVALID_HANDLE_VALUE) {
          child.stdout_read_ = nullptr;
          throw worker_error("cannot create bounded decoder pipe");
        }
        child.stdout_read_event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!child.stdout_read_event_) throw worker_error("cannot create decoder read event");
        child.stdout_child_write_ = CreateFileW(name.c_str(), GENERIC_WRITE, 0, &inheritable,
                                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (child.stdout_child_write_ == INVALID_HANDLE_VALUE) {
          child.stdout_child_write_ = nullptr;
          throw worker_error("cannot open bounded decoder pipe");
        }
        stdout_write = child.stdout_child_write_;
      } else if (stdout_path) {
        fs::create_directories(stdout_path->parent_path(), ec);
        child.stdout_file_ = CreateFileW(
          stdout_path->c_str(),
          GENERIC_WRITE,
          FILE_SHARE_READ,
          &inheritable,
          CREATE_ALWAYS,
          FILE_ATTRIBUTE_NORMAL,
          nullptr
        );
        if (child.stdout_file_ == INVALID_HANDLE_VALUE) {
          child.stdout_file_ = nullptr;
          throw worker_error("cannot create child stdout file");
        }
        stdout_write = child.stdout_file_;
      }

      if (pipe_stdin) {
        const auto name = private_pipe_name();
        child.stdin_write_ = CreateNamedPipeW(name.c_str(),
          PIPE_ACCESS_OUTBOUND | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
          PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
          1, 64 * 1024, 64 * 1024, 0, nullptr);
        if (child.stdin_write_ == INVALID_HANDLE_VALUE) {
          child.stdin_write_ = nullptr;
          throw worker_error("cannot create bounded encoder stdin pipe");
        }
        child.stdin_write_event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!child.stdin_write_event_) throw worker_error("cannot create encoder write event");
        child.stdin_null_ = CreateFileW(name.c_str(), GENERIC_READ, 0, &inheritable,
                                       OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
      } else {
        child.stdin_null_ = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                       &inheritable, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
      }
      if (child.stdin_null_ == INVALID_HANDLE_VALUE) {
        child.stdin_null_ = nullptr;
        throw worker_error("cannot open child stdin");
      }
      std::vector<HANDLE> inherited_handles {
        child.stdin_null_,
        stdout_write,
        log_write,
      };
      std::sort(inherited_handles.begin(), inherited_handles.end());
      inherited_handles.erase(
        std::unique(inherited_handles.begin(), inherited_handles.end()),
        inherited_handles.end()
      );
      SIZE_T attribute_bytes = 0;
      InitializeProcThreadAttributeList(
        nullptr,
        1,
        0,
        &attribute_bytes
      );
      std::vector<std::uint8_t> attribute_storage(attribute_bytes);
      auto *attribute_list =
        reinterpret_cast<PPROC_THREAD_ATTRIBUTE_LIST>(
          attribute_storage.data()
        );
      bool attribute_initialized = false;
      if (attribute_bytes != 0) {
        attribute_initialized = InitializeProcThreadAttributeList(
          attribute_list,
          1,
          0,
          &attribute_bytes
        );
      }
      if (!attribute_initialized || !UpdateProcThreadAttribute(attribute_list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited_handles.data(), inherited_handles.size() * sizeof(HANDLE), nullptr, nullptr)) {
        if (attribute_initialized) {
          DeleteProcThreadAttributeList(attribute_list);
        }
        throw worker_error(
          "cannot restrict native child handle inheritance"
        );
      }
      STARTUPINFOEXW startup {};
      startup.StartupInfo.cb = sizeof(startup);
      startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
      startup.StartupInfo.hStdInput = child.stdin_null_;
      startup.StartupInfo.hStdOutput = stdout_write;
      startup.StartupInfo.hStdError = log_write;
      startup.lpAttributeList = attribute_list;
      PROCESS_INFORMATION process {};
      auto line = command_line(arguments);
      std::vector<wchar_t> mutable_line(line.begin(), line.end());
      mutable_line.push_back(L'\0');
      const auto application = path_from_utf8(arguments.front()).wstring();
      const auto cwd = working_directory.wstring();
      const BOOL launched = CreateProcessW(
        application.c_str(),
        mutable_line.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW | CREATE_SUSPENDED |
          EXTENDED_STARTUPINFO_PRESENT,
        nullptr,
        cwd.empty() ? nullptr : cwd.c_str(),
        &startup.StartupInfo,
        &process
      );
      const auto launch_error =
        launched ? ERROR_SUCCESS : GetLastError();
      DeleteProcThreadAttributeList(attribute_list);
      if (child.stdout_child_write_) {
        CloseHandle(child.stdout_child_write_);
        child.stdout_child_write_ = nullptr;
      }
      // CreateProcess has either inherited its own write handle or failed. Closing the
      // parent's copy lets the drain thread observe EOF as soon as the process tree exits.
      child.log_capture_->close_parent_writer();
      if (!launched) {
        throw worker_error(
          "cannot launch native child process (Windows error " +
          std::to_string(launch_error) + ")"
        );
      }
      child.job_ = CreateJobObjectW(nullptr, nullptr);
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_limits {};
      job_limits.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
      if (!child.job_ || !SetInformationJobObject(child.job_, JobObjectExtendedLimitInformation, &job_limits, sizeof(job_limits)) || !AssignProcessToJobObject(child.job_, process.hProcess) || ResumeThread(process.hThread) == static_cast<DWORD>(-1)) {
        const auto error = GetLastError();
        TerminateProcess(process.hProcess, 9);
        CloseHandle(process.hProcess);
        CloseHandle(process.hThread);
        if (child.job_) {
          CloseHandle(child.job_);
          child.job_ = nullptr;
        }
        throw worker_error(
          "cannot establish kill-on-close child ownership (Windows error " +
          std::to_string(error) + ")"
        );
      }
      if (pipe_stdin) { CloseHandle(child.stdin_null_); child.stdin_null_ = nullptr; }
      child.process_ = process.hProcess;
      child.thread_ = process.hThread;
      return child;
    }

    bool child_process_t::running() const {
      return process_ &&
             WaitForSingleObject(process_, 0) == WAIT_TIMEOUT;
    }

    int child_process_t::exit_code() const {
      if (!process_) {
        return -1;
      }
      DWORD code = 0;
      if (!GetExitCodeProcess(process_, &code)) {
        return -1;
      }
      return static_cast<int>(code);
    }

    void child_process_t::write_exact(const void *source, std::size_t size,
                     const std::chrono::steady_clock::time_point deadline) {
      if (!stdin_write_) throw worker_error("child has no writable stdin pipe");
      const auto *bytes = static_cast<const std::uint8_t *>(source);
      while (size) {
        if (writes_cancelled_.load() || std::chrono::steady_clock::now() >= deadline)
          throw worker_error("bounded encoder frame write cancelled or timed out");
        OVERLAPPED operation {};
        operation.hEvent = stdin_write_event_;
        ResetEvent(operation.hEvent);
        DWORD written = 0;
        bool succeeded = WriteFile(stdin_write_, bytes, static_cast<DWORD>(std::min<std::size_t>(size, 64 * 1024)), &written, &operation);
        if (!succeeded && GetLastError() == ERROR_IO_PENDING) {
          while (WaitForSingleObject(operation.hEvent, 25) == WAIT_TIMEOUT && !writes_cancelled_.load() && running() &&
                 std::chrono::steady_clock::now() < deadline) {}
          if (!GetOverlappedResult(stdin_write_, &operation, &written, FALSE)) {
            CancelIoEx(stdin_write_, &operation);
            GetOverlappedResult(stdin_write_, &operation, &written, TRUE);
          } else succeeded = true;
        }
        if (!succeeded || !written) throw worker_error("bounded encoder stdin write failed or timed out");
        bytes += written;
        size -= written;
      }
    }

    void child_process_t::cancel_writes() noexcept { writes_cancelled_.store(true); }

    void child_process_t::close_stdin() noexcept {
      if (stdin_write_) { CloseHandle(stdin_write_); stdin_write_ = nullptr; }
      if (stdin_write_event_) { CloseHandle(stdin_write_event_); stdin_write_event_ = nullptr; }
    }

    void child_process_t::read_exact(void *destination, const std::size_t size) {
      if (!stdout_read_) {
        throw worker_error("child process has no stdout pipe");
      }
      auto *bytes = static_cast<std::uint8_t *>(destination);
      std::size_t offset = 0;
      const auto deadline =
        std::chrono::steady_clock::now() + frame_io_timeout;
      while (offset < size) {
        const auto received = read_some(bytes + offset, size - offset, deadline);
        if (received == 0) {
          throw worker_error(
            "streaming decoder ended inside a frame (" +
            std::to_string(offset) + " of " + std::to_string(size) +
            " bytes)"
          );
        }
        offset += received;
      }
    }

    std::string child_process_t::read_line(const std::size_t limit) {
      std::string line;
      while (line.size() < limit) {
        char character = 0;
        read_exact(&character, 1);
        line.push_back(character);
        if (character == '\n') {
          return line;
        }
      }
      throw worker_error("decoder emitted an overlong frame header");
    }

    std::size_t child_process_t::read_some(
      void *destination,
      const std::size_t capacity,
      const std::chrono::steady_clock::time_point absolute_deadline
    ) {
      if (!stdout_read_ || capacity == 0) {
        throw worker_error("child process has no readable stdout pipe");
      }
      const auto deadline = std::min(
        std::chrono::steady_clock::now() + frame_io_timeout,
        absolute_deadline
      );
      if (std::chrono::steady_clock::now() >= deadline)
        throw worker_error("native child stdout pipe timed out");
      OVERLAPPED operation {};
      operation.hEvent = stdout_read_event_;
      ResetEvent(operation.hEvent);
      DWORD received = 0;
      const auto request = static_cast<DWORD>(std::min<std::size_t>(capacity, 1024 * 1024));
      BOOL succeeded = ReadFile(stdout_read_, destination, request, &received, &operation);
      if (!succeeded && GetLastError() == ERROR_IO_PENDING) {
        DWORD status;
        do {
          status = WaitForSingleObject(operation.hEvent, 25);
        } while (status == WAIT_TIMEOUT && std::chrono::steady_clock::now() < deadline);
        if (status != WAIT_OBJECT_0) {
          // Drain the canceled operation before its buffer and OVERLAPPED leave scope.
          CancelIoEx(stdout_read_, &operation);
          GetOverlappedResult(stdout_read_, &operation, &received, TRUE);
          throw worker_error("native child stdout pipe timed out or wait failed");
        }
        succeeded = GetOverlappedResult(stdout_read_, &operation, &received, FALSE);
      }
      if (!succeeded) {
        if (native_stdout_pipe_error_is_eof(GetLastError())) return 0;
        throw worker_error("native child stdout pipe read failed");
      }
      return received;
    }

    bool child_process_t::read_one_if_available() {
      if (!stdout_read_) {
        return false;
      }
      DWORD available = 0;
      if (!PeekNamedPipe(
            stdout_read_,
            nullptr,
            0,
            nullptr,
            &available,
            nullptr
          )) {
        return false;
      }
      if (available == 0) {
        return false;
      }
      char byte = 0;
      read_exact(&byte, 1);
      return true;
    }

    int child_process_t::wait(const std::chrono::milliseconds timeout) {
      if (!process_) {
        throw worker_error("cannot wait for an invalid child");
      }
      const DWORD wait_ms = timeout.count() >
                                std::numeric_limits<DWORD>::max() ?
                              INFINITE :
                              static_cast<DWORD>(timeout.count());
      if (WaitForSingleObject(process_, wait_ms) != WAIT_OBJECT_0) {
        throw worker_error("native child process timed out");
      }
      return exit_code();
    }

    void child_process_t::terminate() noexcept {
      close_stdin();
      if (job_) {
        TerminateJobObject(job_, 9);
      }
      if (process_) {
        if (WaitForSingleObject(process_, 0) == WAIT_TIMEOUT) {
          TerminateProcess(process_, 9);
          WaitForSingleObject(process_, 5000);
        }
        CloseHandle(process_);
        process_ = nullptr;
      }
      if (thread_) {
        CloseHandle(thread_);
        thread_ = nullptr;
      }
      if (stdout_read_) {
        CloseHandle(stdout_read_);
        stdout_read_ = nullptr;
      }
      if (stdout_read_event_) {
        CloseHandle(stdout_read_event_);
        stdout_read_event_ = nullptr;
      }
      if (stdout_child_write_) {
        CloseHandle(stdout_child_write_);
        stdout_child_write_ = nullptr;
      }
      if (stdout_file_) {
        CloseHandle(stdout_file_);
        stdout_file_ = nullptr;
      }
      if (stdin_null_) {
        CloseHandle(stdin_null_);
        stdin_null_ = nullptr;
      }
      if (job_) {
        CloseHandle(job_);
        job_ = nullptr;
      }
      if (log_capture_) {
        log_capture_->finish();
        log_capture_.reset();
      }
    }

    void run_logged(
      const std::vector<std::string> &command,
      const fs::path &working_directory,
      const fs::path &log_path,
      const std::optional<fs::path> &stdout_path
    ) {
      auto child = child_process_t::launch(
        command,
        working_directory,
        log_path,
        stdout_path
      );
      const int code = child.wait(
        std::chrono::duration_cast<std::chrono::milliseconds>(child_timeout)
      );
      if (code != 0) {
        throw worker_error(
          "child process exited " + std::to_string(code) +
          "; see " + path_utf8(log_path)
        );
      }
      child.terminate();
    }

    nlohmann::json run_logged_json_bounded(
      const std::vector<std::string> &command,
      const fs::path &working_directory,
      const fs::path &log_path,
      const std::uintmax_t max_bytes,
      const std::string_view description
    ) {
      if (max_bytes == 0 || max_bytes > std::string {}.max_size()) {
        throw worker_error(
          std::string(description) + " has an invalid byte bound"
        );
      }
      const auto deadline =
        std::chrono::steady_clock::now() + child_timeout;
      auto child = child_process_t::launch(
        command,
        working_directory,
        log_path,
        std::nullopt,
        true
      );
      std::string bytes;
      bytes.reserve(
        static_cast<std::size_t>(
          std::min<std::uintmax_t>(
            max_bytes,
            1024ull * 1024ull
          )
        )
      );
      std::array<char, 64ull * 1024ull> buffer {};
      while (true) {
        const auto received = child.read_some(
          buffer.data(),
          buffer.size(),
          deadline
        );
        if (received == 0) {
          break;
        }
        if (received > max_bytes - bytes.size()) {
          throw worker_error(
            std::string(description) + " exceeds its " +
            std::to_string(max_bytes) + "-byte contract"
          );
        }
        bytes.append(buffer.data(), received);
      }
      const auto now = std::chrono::steady_clock::now();
      const auto remaining = now < deadline ?
                               std::chrono::duration_cast<
                                 std::chrono::milliseconds
                               >(deadline - now) :
                               std::chrono::milliseconds {0};
      const int code = child.wait(remaining);
      if (code != 0) {
        throw worker_error(
          "child process exited " + std::to_string(code) +
          "; see " + path_utf8(log_path)
        );
      }
      child.terminate();
      if (bytes.empty()) {
        throw worker_error(std::string(description) + " is empty");
      }
      try {
        return wire::parse_json_without_duplicate_keys(bytes);
      } catch (const std::exception &exception) {
        throw worker_error(
          "cannot parse " + std::string(description) + ": " +
          exception.what()
        );
      }
    }
  }  // namespace worker_detail

#ifdef SUNSHINE_TESTS
  bool native_stdout_pipe_error_is_eof_for_test(
    const std::uint32_t error
  ) noexcept {
    return native_stdout_pipe_error_is_eof(static_cast<DWORD>(error));
  }

  std::size_t child_process_log_byte_limit_for_test() {
    return max_child_log_bytes;
  }

  std::string bound_child_process_log_for_test(
    const std::string_view bytes
  ) {
    bounded_log_accumulator_t accumulator;
    constexpr std::size_t chunk_bytes = 3137;
    for (std::size_t offset = 0; offset < bytes.size();) {
      const auto size = std::min<std::size_t>(
        chunk_bytes,
        bytes.size() - offset
      );
      accumulator.append(bytes.data() + offset, size);
      offset += size;
    }
    const auto result = accumulator.render();
    return std::string(result.begin(), result.end());
  }

  void validate_windows_command_line_capacity_for_test(
    const std::vector<std::string> &arguments
  ) {
    validate_windows_command_line_capacity(arguments, "test command");
  }
#endif
}  // namespace offline_sbs
