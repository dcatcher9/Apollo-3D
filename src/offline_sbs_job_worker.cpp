/**
 * @file src/offline_sbs_job_worker.cpp
 * @brief The job manager's side of the isolated worker process and its media tools.
 */

#include "offline_sbs_job_worker.h"

#include "offline_sbs_contract.h"
#include "offline_sbs_job_files.h"
#include "offline_sbs_wire_contract.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "platform/common.h"

#include <boost/filesystem/path.hpp>
#include <boost/process/v1/child.hpp>
#include <boost/process/v1/environment.hpp>
#include <boost/process/v1/group.hpp>

#include "logging.h"

  #include <fcntl.h>
  #include <io.h>
  #include <windows.h>

using namespace std::literals;

namespace offline_sbs {
  namespace fs = std::filesystem;
  namespace bp = boost::process::v1;
  using namespace job_detail;

  namespace {
    constexpr std::uintmax_t max_progress_contract_bytes = 256ull * 1024ull;
    constexpr auto media_probe_timeout = std::chrono::seconds(10);
    constexpr auto worker_termination_timeout = std::chrono::seconds(5);
    // The outer manager captures the native worker's merged stdout/stderr. Keep the
    // retained diagnostic strictly below the much larger artifact quota even when a
    // broken codec or logger emits continuously for the 12-hour worker allowance.
    constexpr std::size_t max_native_worker_log_bytes = 1ull * 1024ull * 1024ull;
    constexpr std::string_view native_worker_log_limit_marker =
      "\n[Sunshine 3D stopped this offline worker because its diagnostic "
      "output exceeded the bounded log limit.]\n";
    constexpr std::size_t max_native_worker_log_payload_bytes =
      max_native_worker_log_bytes - native_worker_log_limit_marker.size();

    static_assert(max_contract_bytes == worker_result_max_bytes);
    static_assert(
      native_worker_log_limit_marker.size() < max_native_worker_log_bytes
    );

    std::string quote_windows_process_argument(const std::string_view argument) {
      // The command is passed to CreateProcess, not cmd.exe. Implement the documented
      // CommandLineToArgvW-compatible backslash/quote convention and always quote.
      std::string result;
      result.push_back('"');
      std::size_t backslashes = 0;
      for (const char character : argument) {
        if (character == '\\') {
          ++backslashes;
          continue;
        }
        if (character == '"') {
          result.append(backslashes * 2 + 1, '\\');
          result.push_back('"');
          backslashes = 0;
          continue;
        }
        result.append(backslashes, '\\');
        backslashes = 0;
        result.push_back(character);
      }
      result.append(backslashes * 2, '\\');
      result.push_back('"');
      return result;
    }

    std::string build_worker_command(
      const fs::path &sunshine_executable,
      const fs::path &sunshine_config,
      const fs::path &worker_spec,
      const std::string_view worker_spec_sha256
    ) {
      return quote_windows_process_argument(path_to_utf8(sunshine_executable)) +
             " " +
             quote_windows_process_argument(path_to_utf8(sunshine_config)) +
             " --offline-sbs-worker " +
             quote_windows_process_argument(path_to_utf8(worker_spec)) +
             " " +
             quote_windows_process_argument(worker_spec_sha256);
    }

    FILE *open_binary_log(const fs::path &path) {
      return _wfopen(path.c_str(), L"wb");
    }

    /**
     * Retains the beginning of a worker diagnostic and appends a deterministic
     * explanation when the source exceeds its allowance. The marker's space is
     * reserved up front, so the sink can never write more than
     * max_native_worker_log_bytes even when one read returns a very large chunk.
     *
     * Exceeding the limit is fatal to the worker (the supervisor enforces that
     * policy). This is deliberately a bounded head capture rather than a rotating
     * file: startup diagnostics remain intact, and the child cannot hide a runaway
     * logger by continually replacing the evidence that caused termination.
     */
    class bounded_native_worker_log_writer_t {
    public:
      using sink_t = std::function<bool(std::string_view)>;

      explicit bounded_native_worker_log_writer_t(sink_t sink):
          sink_(std::move(sink)) {
      }

      void append(const std::string_view bytes) {
        if (bytes.empty() || write_failed_.load(std::memory_order_relaxed)) {
          return;
        }

        const auto available =
          max_native_worker_log_payload_bytes - payload_bytes_;
        const auto retained = std::min(available, bytes.size());
        if (
          retained != 0 &&
          !sink_(bytes.substr(0, retained))
        ) {
          write_failed_.store(true, std::memory_order_release);
          return;
        }
        payload_bytes_ += retained;

        if (
          retained != bytes.size() &&
          !limit_exceeded_.exchange(true, std::memory_order_acq_rel)
        ) {
          if (!sink_(native_worker_log_limit_marker)) {
            write_failed_.store(true, std::memory_order_release);
          }
        }
      }

      [[nodiscard]] bool limit_exceeded() const noexcept {
        return limit_exceeded_.load(std::memory_order_acquire);
      }

      [[nodiscard]] bool write_failed() const noexcept {
        return write_failed_.load(std::memory_order_acquire);
      }

    private:
      sink_t sink_;
      std::size_t payload_bytes_ = 0;
      std::atomic<bool> limit_exceeded_ {false};
      std::atomic<bool> write_failed_ {false};
    };

    /**
     * Pipe-backed capture for the native worker's merged stdout/stderr.
     *
     * The child receives the duplicated pipe writer as its STARTF_USESTDHANDLES
     * standard output/error. The elevated-tray token path uses plain STARTUPINFO;
     * no other handle is named by that launch contract. A manager-owned reader drains
     * it into the strictly bounded file above. The reader uses non-blocking availability
     * checks, so abort() always joins promptly even if a failed process-tree cleanup
     * leaves a writer handle alive; this keeps cancellation and Sunshine shutdown
     * deadlock-free.
     */
    class bounded_native_worker_log_pipe_t {
    public:
      explicit bounded_native_worker_log_pipe_t(FILE *output):
          output_(output, &std::fclose) {
      }

      ~bounded_native_worker_log_pipe_t() {
        abort();
      }

      bounded_native_worker_log_pipe_t(
        const bounded_native_worker_log_pipe_t &
      ) = delete;
      bounded_native_worker_log_pipe_t &operator=(
        const bounded_native_worker_log_pipe_t &
      ) = delete;

      bool initialize(std::string &error) {
        HANDLE read_handle = INVALID_HANDLE_VALUE;
        HANDLE write_handle = INVALID_HANDLE_VALUE;
        if (
          !CreatePipe(
            &read_handle,
            &write_handle,
            nullptr,
            64u * 1024u
          )
        ) {
          error =
            "cannot create bounded native worker diagnostic pipe (Windows error " +
            std::to_string(GetLastError()) + ")";
          return false;
        }
        read_handle_ = read_handle;
        const auto descriptor = _open_osfhandle(
          reinterpret_cast<std::intptr_t>(write_handle),
          _O_BINARY | _O_WRONLY
        );
        if (descriptor == -1) {
          CloseHandle(write_handle);
          close_read_handle();
          error = "cannot bind bounded native worker diagnostic pipe";
          return false;
        }
        child_stream_ = _fdopen(descriptor, "wb");
        if (!child_stream_) {
          _close(descriptor);
          close_read_handle();
          error = "cannot open bounded native worker diagnostic stream";
          return false;
        }

        writer_ = std::make_unique<bounded_native_worker_log_writer_t>(
          [this](const std::string_view bytes) {
            if (!output_) {
              return false;
            }
            const auto written =
              std::fwrite(bytes.data(), 1, bytes.size(), output_.get());
            return written == bytes.size() && std::fflush(output_.get()) == 0;
          }
        );
        try {
          reader_ = std::jthread([this](const std::stop_token stop) {
            drain(stop);
          });
        } catch (const std::exception &exception) {
          error =
            std::string {"cannot start native worker diagnostic reader: "} +
            exception.what();
          abort();
          return false;
        }
        return true;
      }

      [[nodiscard]] FILE *child_stream() const noexcept {
        return child_stream_;
      }

      void close_parent_writer() noexcept {
        if (child_stream_) {
          std::fclose(child_stream_);
          child_stream_ = nullptr;
        }
      }

      // Call only after the complete process group is known to be stopped. All pipe
      // writers are then closed and the reader drains the final buffered bytes to EOF.
      void finish() noexcept {
        close_parent_writer();
        if (reader_.joinable()) {
          reader_.join();
        }
        close_read_handle();
      }

      // Safe even when process-tree cleanup failed: the reader never blocks in a
      // consuming read, so its stop request is observed within one short poll.
      void abort() noexcept {
        close_parent_writer();
        if (reader_.joinable()) {
          reader_.request_stop();
          reader_.join();
        }
        close_read_handle();
      }

      [[nodiscard]] bool limit_exceeded() const noexcept {
        return writer_ && writer_->limit_exceeded();
      }

      [[nodiscard]] bool failed() const noexcept {
        return
          pipe_error_.load(std::memory_order_acquire) != 0 ||
          (writer_ && writer_->write_failed());
      }

      [[nodiscard]] std::string failure() const {
        if (writer_ && writer_->write_failed()) {
          return "cannot write bounded native worker diagnostic log";
        }
        const auto error = pipe_error_.load(std::memory_order_acquire);
        if (error == 0) {
          return {};
        }
        return
          "native worker diagnostic pipe failed (Windows error " +
          std::to_string(error) + ")";
      }

    private:
      static constexpr auto reader_idle_poll = std::chrono::milliseconds(2);

      void drain(const std::stop_token stop) noexcept {
        std::array<char, 64u * 1024u> buffer {};
        while (!stop.stop_requested()) {
          DWORD available = 0;
          if (
            !PeekNamedPipe(
              read_handle_,
              nullptr,
              0,
              nullptr,
              &available,
              nullptr
            )
          ) {
            const auto error = GetLastError();
            if (
              error != ERROR_BROKEN_PIPE &&
              error != ERROR_PIPE_NOT_CONNECTED &&
              error != ERROR_NO_DATA
            ) {
              pipe_error_.store(error, std::memory_order_release);
            }
            break;
          }
          if (available == 0) {
            std::this_thread::sleep_for(reader_idle_poll);
            continue;
          }
          DWORD bytes_read = 0;
          const auto requested = static_cast<DWORD>(
            std::min<std::size_t>(buffer.size(), available)
          );
          if (
            !ReadFile(
              read_handle_,
              buffer.data(),
              requested,
              &bytes_read,
              nullptr
            )
          ) {
            const auto error = GetLastError();
            if (
              error != ERROR_BROKEN_PIPE &&
              error != ERROR_PIPE_NOT_CONNECTED &&
              error != ERROR_NO_DATA
            ) {
              pipe_error_.store(error, std::memory_order_release);
            }
            break;
          }
          if (bytes_read == 0) {
            break;
          }
          writer_->append(
            std::string_view {buffer.data(), static_cast<std::size_t>(bytes_read)}
          );
        }
      }

      void close_read_handle() noexcept {
        if (read_handle_ != INVALID_HANDLE_VALUE) {
          CloseHandle(read_handle_);
          read_handle_ = INVALID_HANDLE_VALUE;
        }
      }

      std::unique_ptr<FILE, decltype(&std::fclose)> output_ {
        nullptr,
        &std::fclose,
      };
      FILE *child_stream_ = nullptr;
      std::unique_ptr<bounded_native_worker_log_writer_t> writer_;
      std::jthread reader_;
      std::atomic<int> pipe_error_ {0};
      HANDLE read_handle_ = INVALID_HANDLE_VALUE;
    };

    bool terminate_and_reap_process_tree(
      bp::child &child,
      bp::group &group,
      const std::chrono::milliseconds timeout,
      std::string &error
    ) {
      std::error_code terminate_error;
      const bool group_was_running =
        group.valid() &&
        platf::process_group_running(
          reinterpret_cast<std::uintptr_t>(group.native_handle())
        );
      const bool child_was_running = child.valid() && child.running();
      if (group_was_running) {
        group.terminate(terminate_error);
      } else if (child_was_running) {
        child.terminate(terminate_error);
      }

      const auto deadline = std::chrono::steady_clock::now() + timeout;
      while (std::chrono::steady_clock::now() < deadline) {
        const bool child_running = child.valid() && child.running();
        const bool group_running =
          group.valid() &&
          platf::process_group_running(
            reinterpret_cast<std::uintptr_t>(group.native_handle())
          );
        if (!child_running && !group_running) {
          break;
        }
        std::this_thread::sleep_for(20ms);
      }

      const bool child_running = child.valid() && child.running();
      const bool group_running =
        group.valid() &&
        platf::process_group_running(
          reinterpret_cast<std::uintptr_t>(group.native_handle())
        );
      if (child_running || group_running) {
        if (child_running) {
          std::error_code ignored;
          child.terminate(ignored);
          child.detach();
        }
        // Do not detach a still-running group. Its destructor closes the native job handle,
        // providing the final kill-on-close fallback without blocking Sunshine shutdown.
        error =
          "process tree did not terminate within " +
          std::to_string(timeout.count()) + " ms";
        if (terminate_error) {
          error += ": " + terminate_error.message();
        }
        return false;
      }

      if (child.valid()) {
        std::error_code wait_error;
        child.wait(wait_error);
        if (wait_error) {
          error = "cannot reap child process: " + wait_error.message();
          if (group.valid()) {
            group.detach();
          }
          return false;
        }
      }
      if (group.valid()) {
        group.detach();
      }
      if (terminate_error) {
        error = "process-tree termination reported: " + terminate_error.message();
        return false;
      }
      return true;
    }

    bool harden_worker_process_group(bp::group &group, std::string &error) {
      if (!group.valid()) {
        error = "native worker process group is invalid";
        return false;
      }
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits {};
      const auto handle = reinterpret_cast<HANDLE>(group.native_handle());
      if (!QueryInformationJobObject(
            handle,
            JobObjectExtendedLimitInformation,
            &limits,
            sizeof(limits),
            nullptr
          )) {
        error =
          "cannot query native worker job object (Windows error " +
          std::to_string(GetLastError()) + ")";
        return false;
      }
      limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
      limits.BasicLimitInformation.LimitFlags &=
        ~(JOB_OBJECT_LIMIT_BREAKAWAY_OK | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK);
      if (!SetInformationJobObject(
            handle,
            JobObjectExtendedLimitInformation,
            &limits,
            sizeof(limits)
          )) {
        error =
          "cannot harden native worker job object (Windows error " +
          std::to_string(GetLastError()) + ")";
        return false;
      }
      return true;
    }

    std::optional<std::string> run_media_tool_probe(
      const service_config_t &config,
      const fs::path &tool,
      const std::string_view tool_name,
      const std::string_view probe_name,
      const std::string_view arguments,
      const bool allow_empty_output,
      std::string &error
    ) {
      const auto log_path =
        config.state_root /
        path_from_utf8(
          std::string {tool_name} + "-" + std::string {probe_name} + "-probe.log"
        );
      std::unique_ptr<FILE, decltype(&std::fclose)> log {
        open_binary_log(log_path),
        &std::fclose,
      };
      if (!log) {
        error = "cannot create FFmpeg probe log";
        return std::nullopt;
      }
      auto environment = boost::this_process::environment();
      boost::filesystem::path working_directory {
        // Native wide path, so an installation folder outside the ANSI code page still works.
        config.sunshine_executable.parent_path().wstring()
      };
      bp::group group;
      std::string group_error;
      if (!harden_worker_process_group(group, group_error)) {
        error =
          std::string {tool_name} + " probe isolation failed before launch: " +
          group_error;
        return std::nullopt;
      }
      std::error_code launch_error;
      const auto command =
        quote_windows_process_argument(path_to_utf8(tool)) +
        " " + std::string {arguments};
      auto child = platf::run_command_unelevated(
        false,
        command,
        working_directory,
        environment,
        log.get(),
        launch_error,
        &group,
        config.expected_user_id
      );
      if (launch_error || !child.valid()) {
        error =
          "cannot launch " + std::string {tool_name} +
          " probe: " + launch_error.message();
        return std::nullopt;
      }
      const auto deadline = std::chrono::steady_clock::now() + media_probe_timeout;
      while (child.running() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(20ms);
      }
      if (child.running()) {
        std::string cleanup_error;
        terminate_and_reap_process_tree(
          child,
          group,
          worker_termination_timeout,
          cleanup_error
        );
        error =
          std::string {tool_name} + " probe timed out" +
          (cleanup_error.empty() ? "" : ": " + cleanup_error);
        return std::nullopt;
      }
      const auto exit_code = child.exit_code();
      std::string cleanup_error;
      if (!terminate_and_reap_process_tree(
            child,
            group,
            worker_termination_timeout,
            cleanup_error
          )) {
        error = std::string {tool_name} + " probe cleanup failed: " + cleanup_error;
        return std::nullopt;
      }
      if (exit_code != 0) {
        error =
          std::string {tool_name} + " " + std::string {probe_name} +
          " probe failed";
        return std::nullopt;
      }
      log.reset();
      std::error_code size_error;
      const auto output_bytes = fs::file_size(log_path, size_error);
      if (size_error || output_bytes > 8ull * 1024ull * 1024ull) {
        error =
          std::string {tool_name} + " " + std::string {probe_name} +
          " probe output is invalid";
        return std::nullopt;
      }
      std::ifstream input(log_path, std::ios::binary);
      std::string output {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>(),
      };
      if (output.empty() && !allow_empty_output) {
        error =
          std::string {tool_name} + " " + std::string {probe_name} +
          " probe produced no output";
        return std::nullopt;
      }
      return output;
    }

    progress_t parse_worker_progress(
      const nlohmann::json &value,
      const std::string &job_id
    ) {
      if (
        value.value("schema", 0) != 1 ||
        value.value("job_id", "") != job_id
      ) {
        throw std::runtime_error("worker progress identity mismatch");
      }
      progress_t progress;
      progress.phase = value.value("phase", "");
      if (progress.phase.empty() || progress.phase.size() > 64) {
        throw std::runtime_error("worker progress phase is invalid");
      }
      progress.processed_frames = value.value("processed_frames", 0ull);
      if (value.contains("total_frames") && !value["total_frames"].is_null()) {
        progress.total_frames = value["total_frames"].get<std::uint64_t>();
        if (
          *progress.total_frames == 0 ||
          progress.processed_frames > *progress.total_frames
        ) {
          throw std::runtime_error("worker progress frame counts are invalid");
        }
      }
      if (
        value.contains("source_time_seconds") &&
        !value["source_time_seconds"].is_null()
      ) {
        const auto number = value["source_time_seconds"].get<double>();
        if (!std::isfinite(number) || number < 0) {
          throw std::runtime_error("worker source time is invalid");
        }
        progress.source_time_seconds = number;
      }
      if (
        value.contains("source_duration_seconds") &&
        !value["source_duration_seconds"].is_null()
      ) {
        const auto number = value["source_duration_seconds"].get<double>();
        if (!std::isfinite(number) || number <= 0) {
          throw std::runtime_error("worker source duration is invalid");
        }
        progress.source_duration_seconds = number;
      }
      return progress;
    }

    bool validate_worker_result_contract(
      const nlohmann::json &result,
      const worker_context_t &context,
      std::string &error
    ) {
      try {
        const auto typed_result = wire::parse_worker_result_contract(result);
        if (
          typed_result.job_id != context.job_id ||
          typed_result.operation != to_string(context.operation) ||
          typed_result.codec != context.codec ||
          context.worker_spec_sha256.empty() ||
          typed_result.worker_spec_sha256 != context.worker_spec_sha256
        ) {
          error = "worker result identity/configuration attestation mismatch";
          return false;
        }
        const auto &source = typed_result.source;
        if (
          source.width == 0 || source.height == 0 || source.frame_count == 0 ||
          !std::isfinite(source.duration_seconds) ||
          source.duration_seconds <= 0
        ) {
          error = "worker source summary is invalid";
          return false;
        }
        const auto expected_source_contract =
          (context.result_directory / "source-contract.json").lexically_normal();
        if (
          path_from_utf8(source.contract_path)
              .lexically_normal() != expected_source_contract ||
          !regular_nonempty_file(expected_source_contract)
        ) {
          error = "worker result artifact paths are invalid";
          return false;
        }
        if (typed_result.cache.hard_cap_bytes != context.transient_raster_hard_cap_bytes) {
          error = "worker transient-raster attestation is invalid";
          return false;
        }

        if (context.operation == operation_e::convert) {
          if (
            !context.staging_output ||
            !context.final_output ||
            !typed_result.output_path ||
            !typed_result.staging_identity ||
            path_from_utf8(*typed_result.output_path)
                .lexically_normal() != context.staging_output->lexically_normal() ||
            !file_matches_publish_identity(
              *context.staging_output,
              *typed_result.staging_identity
            ) ||
            !regular_nonempty_file(*context.staging_output) ||
            !regular_nonempty_file(
              context.result_directory / "output-contract.json"
            )
          ) {
            error = "worker direct conversion output attestation is invalid";
            return false;
          }
        } else if (typed_result.output_path || typed_result.staging_identity) {
          error = "evaluation worker unexpectedly reported rendered output";
          return false;
        }
      } catch (const std::exception &exception) {
        error = std::string {"invalid worker result: "} + exception.what();
        return false;
      }
      return true;
    }
  }  // namespace

  namespace job_detail {
    const char *to_string(operation_e value) {
      switch (value) {
        case operation_e::evaluate:
          return "evaluate";
        case operation_e::convert:
          return "convert";
      }
      return "unknown";
    }

    std::optional<fs::path> discover_media_tool(
      const service_config_t &config,
      const std::optional<fs::path> &trusted_override,
      const std::string_view tool_name,
      std::string &error
    ) {
      std::vector<fs::path> candidates;
      if (trusted_override) {
        if (!trusted_override->is_absolute()) {
          error = "trusted media-tool override must be an absolute path";
          return std::nullopt;
        }
        candidates.push_back(*trusted_override);
      } else {
        const auto installation = config.sunshine_executable.parent_path();
        candidates.push_back(
          installation / path_from_utf8(std::string {tool_name} + ".exe")
        );
        candidates.push_back(
          installation / "tools" /
          path_from_utf8(std::string {tool_name} + ".exe")
        );
      }

      for (const auto &candidate : candidates) {
        std::string ignored;
        auto resolved = canonical_regular_file(candidate, ignored);
        if (!resolved) {
          continue;
        }
        auto name = resolved->filename().string();
        std::transform(name.begin(), name.end(), name.begin(), [](const unsigned char c) {
          return static_cast<char>(std::tolower(c));
        });
        if (name != std::string {tool_name} + ".exe") {
          error =
            "trusted media-tool executable must be named " +
            std::string {tool_name};
          return std::nullopt;
        }
        return resolved;
      }
      error =
        std::string {tool_name} +
        " was not found beside Sunshine (or in its tools directory); "
        "offline video jobs are unavailable until the trusted media tools are installed";
      return std::nullopt;
    }

    std::optional<std::string> probe_media_tool_version(
      const service_config_t &config,
      const fs::path &tool,
      const std::string_view tool_name,
      std::string &error
    ) {
      if (!config.probe_media_tools) {
        return "not-probed";
      }
      auto output = run_media_tool_probe(
        config,
        tool,
        tool_name,
        "version",
        "-hide_banner -version",
        false,
        error
      );
      if (!output) {
        return std::nullopt;
      }
      std::istringstream lines {*output};
      std::string line;
      while (std::getline(lines, line)) {
        if (!line.empty()) {
          if (line.size() > 512) {
            line.resize(512);
          }
          return line;
        }
      }
      error = std::string {tool_name} + " probe produced no version line";
      return std::nullopt;
    }

    std::optional<std::string> media_tool_version_token(
      const std::string_view line,
      const std::string_view tool_name
    ) {
      const std::string prefix = std::string {tool_name} + " version ";
      if (!line.starts_with(prefix)) {
        return std::nullopt;
      }
      const auto token_start = prefix.size();
      const auto token_end = line.find_first_of(" \t\r\n", token_start);
      if (token_end == token_start) {
        return std::nullopt;
      }
      return std::string {
        line.substr(
          token_start,
          token_end == std::string_view::npos ?
            line.size() - token_start :
            token_end - token_start
        )
      };
    }

    std::optional<std::vector<std::string>> probe_ffmpeg_codecs(
      const service_config_t &config,
      const fs::path &ffmpeg,
      std::string &error
    ) {
#ifdef SUNSHINE_TESTS
      if (config.test_probed_codecs) {
        return *config.test_probed_codecs;
      }
#endif
      if (!config.probe_media_tools) {
        return std::vector<std::string> {"hevc_nvenc", "av1_nvenc"};
      }
      auto encoders = run_media_tool_probe(
        config,
        ffmpeg,
        "ffmpeg",
        "encoders",
        "-hide_banner -encoders",
        false,
        error
      );
      if (!encoders) {
        return std::nullopt;
      }
      auto filters = run_media_tool_probe(
        config,
        ffmpeg,
        "ffmpeg",
        "filters",
        "-hide_banner -filters",
        false,
        error
      );
      if (!filters) {
        return std::nullopt;
      }
      if (filters->find("zscale") == std::string::npos) {
        error =
          "approved FFmpeg lacks the zscale filter required for static HDR conversion";
        return std::nullopt;
      }

      std::vector<std::string> codecs;
      for (const auto codec : {"hevc_nvenc"sv, "av1_nvenc"sv}) {
        if (encoders->find(codec) == std::string::npos) {
          continue;
        }
        codecs.emplace_back(codec);
      }
      return codecs;
    }

    bool probe_ffmpeg_codec_runtime(
      const service_config_t &config,
      const fs::path &ffmpeg,
      const std::string_view codec,
      std::string &error
    ) {
      const auto arguments = codec == "hevc_nvenc" ?
        "-hide_banner -loglevel error -nostdin -f lavfi "
        "-i color=c=black:s=256x256:d=0.04 -frames:v 1 -an "
        "-vf format=p010le -c:v hevc_nvenc -profile:v main10 -f null -" :
        "-hide_banner -loglevel error -nostdin -f lavfi "
        "-i color=c=black:s=256x256:d=0.04 -frames:v 1 -an "
        "-vf format=p010le -c:v av1_nvenc -f null -";
      return run_media_tool_probe(
        config,
        ffmpeg,
        "ffmpeg",
        std::string {codec} + "-10bit",
        arguments,
        true,
        error
      ).has_value();
    }

    nlohmann::json worker_spec_json(const worker_context_t &context) {
      return wire::to_json(wire::worker_spec_contract_t {
        .job_id = context.job_id,
        .operation = to_string(context.operation),
        .input_path = path_to_utf8(context.input_path),
        .job_directory = path_to_utf8(context.job_directory),
        .result_directory = path_to_utf8(context.result_directory),
        .progress_path = path_to_utf8(context.worker_progress),
        .result_path = path_to_utf8(context.worker_result),
        .staging_output = context.staging_output ?
                            std::optional<std::string> {
                              path_to_utf8(*context.staging_output)
                            } :
                            std::nullopt,
        .sunshine_executable = path_to_utf8(context.sunshine_executable),
        .sunshine_config = path_to_utf8(context.sunshine_config),
        .ffmpeg_path = path_to_utf8(context.ffmpeg_executable),
        .ffmpeg_version = context.ffmpeg_version,
        .ffprobe_path = path_to_utf8(context.ffprobe_executable),
        .ffprobe_version = context.ffprobe_version,
        .codec = context.codec,
        .transient_raster_hard_cap_bytes = context.transient_raster_hard_cap_bytes,
      });
    }

    nlohmann::json compact_worker_result(
      const nlohmann::json &result,
      const worker_context_t &context
    ) {
      // The child-owned worker result retains the complete attestation. Durable service
      // state stays deliberately small so listing jobs cannot deep-copy a full report
      // under the service mutex and job.json remains bounded and restart-safe.
      nlohmann::json compact = nlohmann::json::object();
      constexpr std::array scalar_fields {
        "schema"sv,
        "job_id"sv,
        "status"sv,
        "operation"sv,
        "codec"sv,
        "worker_spec_sha256"sv,
        "python_dependency"sv,
        "output"sv,
      };
      for (const auto field : scalar_fields) {
        const auto iterator = result.find(field);
        if (
          iterator != result.end() &&
          (
            iterator->is_null() ||
            iterator->is_boolean() ||
            iterator->is_number() ||
            iterator->is_string()
          )
        ) {
          compact[std::string {field}] = *iterator;
        }
      }
      const auto copy_summary_object = [&](const std::string_view name,
                                           const auto &allowed_fields) {
        const auto source = result.find(name);
        if (source == result.end() || !source->is_object()) {
          return;
        }
        nlohmann::json summary = nlohmann::json::object();
        for (const auto field : allowed_fields) {
          const auto value = source->find(field);
          if (
            value != source->end() &&
            (
              value->is_null() ||
              value->is_boolean() ||
              value->is_number() ||
              value->is_string()
            )
          ) {
            summary[std::string {field}] = *value;
          }
        }
        compact[std::string {name}] = std::move(summary);
      };
      copy_summary_object(
        "source",
        std::array {
          "width"sv,
          "height"sv,
          "frame_count"sv,
          "duration_seconds"sv,
          "variable_frame_rate"sv,
          "color_mode"sv,
          "contract"sv,
        }
      );
      copy_summary_object("cache", std::array {"hard_cap_bytes"sv});
      compact["full_result_contract"] = path_to_utf8(context.worker_result);
      compact["details_storage"] = "worker-artifacts";
      return compact;
    }

    worker_outcome_t run_native_worker(
      const service_config_t &config,
      const worker_context_t &context,
      const std::stop_token stop,
      const progress_callback_t &publish_progress
    ) {
      const auto run_user_filesystem_action = [&config](
        const std::function<void()> &action,
        std::string &error
      ) {
        return run_bound_user_filesystem_action(
          config.expected_user_id,
          action,
          error
        );
      };
      if (context.worker_spec_sha256.size() != 64) {
        return {.error = "native worker specification digest is missing"};
      }
      const auto command = build_worker_command(
        config.sunshine_executable,
        config.sunshine_config,
        context.worker_spec,
        context.worker_spec_sha256
      );
      FILE *raw_log = nullptr;
      std::string filesystem_error;
      if (!run_user_filesystem_action([&]() {
            raw_log = open_binary_log(context.worker_log);
          }, filesystem_error)) {
        if (raw_log) {
          std::fclose(raw_log);
        }
        return {.error = filesystem_error};
      }
      if (!raw_log) {
        return {.error = "cannot create native worker log"};
      }
      bounded_native_worker_log_pipe_t log {raw_log};
      std::string log_error;
      if (!log.initialize(log_error)) {
        return {.error = log_error};
      }
      auto environment = boost::this_process::environment();
      boost::filesystem::path working_directory {
        // Native wide path, so an installation folder outside the ANSI code page still works.
        config.sunshine_executable.parent_path().wstring()
      };
      bp::group group;
      std::string group_error;
      if (!harden_worker_process_group(group, group_error)) {
        return {
          .error =
            "native worker isolation failed before launch: " + group_error,
        };
      }
      std::error_code process_error;
      auto child = platf::run_command_unelevated(
        false,
        command,
        working_directory,
        environment,
        log.child_stream(),
        process_error,
        &group,
        config.expected_user_id
      );
      // run_command_unelevated() duplicates this stream into inheritable standard-output/error
      // handles. Its elevated-tray token path supplies them through STARTF_USESTDHANDLES with
      // plain STARTUPINFO. Closing the manager's writer immediately guarantees EOF once the
      // isolated process group and its duplicated writer are gone.
      log.close_parent_writer();
      if (process_error || !child.valid()) {
        log.finish();
        return {
          .error = "cannot launch native offline worker: " +
                   process_error.message(),
        };
      }

      const auto terminate_worker = [&](std::string &cleanup_error) {
        const bool stopped = terminate_and_reap_process_tree(
          child,
          group,
          worker_termination_timeout,
          cleanup_error
        );
        if (stopped) {
          log.finish();
        } else {
          log.abort();
        }
        return stopped;
      };

      fs::file_time_type last_progress_write {};
      const auto consume_progress_update = [&](
        const bool force_read = false
      ) -> std::optional<std::string> {
        std::error_code time_error;
        fs::file_time_type write_time {};
        std::optional<nlohmann::json> value;
        std::string contract_error;
        filesystem_error.clear();
        const bool progress_read = run_user_filesystem_action([&]() {
          write_time = fs::last_write_time(
            context.worker_progress,
            time_error
          );
          if (
            !time_error &&
            (force_read || write_time != last_progress_write)
          ) {
            // Worker progress is an atomic but intentionally non-durable live handoff. A
            // filesystem filter can briefly deny the newly replaced name, so retry only this
            // exceptional read path. There is no delay before a normal progress read.
            constexpr int transient_progress_read_attempts = 50;
            for (int attempt = 0;
                 attempt < transient_progress_read_attempts;
                 ++attempt) {
              contract_error.clear();
              bool acquisition_failure = false;
              value = read_json_contract(
                context.worker_progress,
                contract_error,
                max_progress_contract_bytes,
                &acquisition_failure
              );
              if (value || !acquisition_failure) {
                break;
              }
              if (attempt + 1 < transient_progress_read_attempts) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
              }
            }
          }
        }, filesystem_error);
        if (!progress_read) {
          return "cannot inspect native worker progress: " + filesystem_error;
        }
        if (
          time_error ||
          (!force_read && write_time == last_progress_write)
        ) {
          return std::nullopt;
        }
        if (!value) {
          if (contract_error == "contract does not exist") {
            return std::nullopt;
          }
          return "invalid native worker progress: " + contract_error;
        }
        try {
          publish_progress(parse_worker_progress(*value, context.job_id));
          last_progress_write = write_time;
        } catch (const std::exception &exception) {
          return std::string {"invalid native worker progress: "} +
                 exception.what();
        }
        return std::nullopt;
      };

      while (child.running()) {
        if (stop.stop_requested()) {
          std::string cleanup_error;
          if (!terminate_worker(cleanup_error)) {
            BOOST_LOG(error) << "Offline SBS worker cancellation cleanup failed: "sv
                             << cleanup_error;
          }
          return {.canceled = true};
        }
        if (log.limit_exceeded() || log.failed()) {
          std::string cleanup_error;
          terminate_worker(cleanup_error);
          return {
            .error =
              log.limit_exceeded() ?
                "native offline worker diagnostic output exceeded its " +
                  std::to_string(max_native_worker_log_bytes) +
                  "-byte retained limit" +
                  (cleanup_error.empty() ? "" : "; cleanup: " + cleanup_error) :
                log.failure() +
                  (cleanup_error.empty() ? "" : "; cleanup: " + cleanup_error),
          };
        }

        if (const auto progress_error = consume_progress_update()) {
          std::string cleanup_error;
          terminate_worker(cleanup_error);
          return {
            .error = *progress_error +
                     (cleanup_error.empty() ? "" : "; cleanup: " + cleanup_error),
          };
        }
        std::this_thread::sleep_for(config.process_poll_interval);
      }

      // The child can atomically publish its final progress and exit between two
      // polling ticks. Drain that last contract before terminalizing the job so
      // its final progress is not lost on fast success or failure exits.
      if (const auto progress_error = consume_progress_update(true)) {
        std::string cleanup_error;
        terminate_worker(cleanup_error);
        return {
          .error = *progress_error +
                   (cleanup_error.empty() ? "" : "; cleanup: " + cleanup_error),
        };
      }

      const auto exit_code = child.exit_code();
      std::string cleanup_error;
      if (!terminate_worker(cleanup_error)) {
        return {.error = "native worker cleanup failed: " + cleanup_error};
      }
      if (log.limit_exceeded()) {
        return {
          .error =
            "native offline worker diagnostic output exceeded its " +
            std::to_string(max_native_worker_log_bytes) +
            "-byte retained limit",
        };
      }
      if (log.failed()) {
        return {.error = log.failure()};
      }
      if (exit_code != 0) {
        return {
          .error = "native offline worker exited with code " +
                   std::to_string(exit_code),
        };
      }

      std::string contract_error;
      std::optional<nlohmann::json> result;
      bool contract_valid = false;
      if (!run_user_filesystem_action([&]() {
            result = read_json_contract(
              context.worker_result,
              contract_error
            );
            if (result) {
              contract_valid = validate_worker_result_contract(
                *result,
                context,
                contract_error
              );
            }
          }, filesystem_error)) {
        return {
          .error =
            "cannot inspect native worker result: " + filesystem_error,
        };
      }
      if (!result) {
        return {
          .error = "native worker did not publish a valid result: " +
                   contract_error,
        };
      }
      if (!contract_valid) {
        return {.error = "native worker result validation failed: " + contract_error};
      }
      return {
        .completed = true,
        .result = std::move(*result),
      };
    }
  }  // namespace job_detail

#ifdef SUNSHINE_TESTS
  progress_t read_worker_progress_for_test(const worker_context_t &context) {
    std::string error;
    const auto value = read_json_contract(
      context.worker_progress,
      error,
      max_progress_contract_bytes
    );
    if (!value) {
      throw std::runtime_error(error);
    }
    return parse_worker_progress(*value, context.job_id);
  }

  bounded_worker_log_test_result_t bound_worker_log_for_test(
    const std::vector<std::string> &chunks
  ) {
    bounded_worker_log_test_result_t result;
    bounded_native_worker_log_writer_t writer {
      [&result](const std::string_view bytes) {
        result.retained.append(bytes);
        return true;
      },
    };
    for (const auto &chunk : chunks) {
      writer.append(chunk);
    }
    result.limit_exceeded = writer.limit_exceeded();
    result.write_failed = writer.write_failed();
    return result;
  }

  bounded_worker_log_test_result_t capture_worker_log_pipe_for_test(
    const fs::path &output,
    const std::vector<std::string> &chunks,
    std::string &error
  ) {
    bounded_worker_log_test_result_t result;
    auto *raw_output = open_binary_log(output);
    if (!raw_output) {
      error = "cannot create test worker log";
      result.write_failed = true;
      return result;
    }
    {
      bounded_native_worker_log_pipe_t pipe {raw_output};
      if (!pipe.initialize(error)) {
        result.write_failed = true;
        return result;
      }
      for (const auto &chunk : chunks) {
        if (
          std::fwrite(
            chunk.data(),
            1,
            chunk.size(),
            pipe.child_stream()
          ) != chunk.size() ||
          std::fflush(pipe.child_stream()) != 0
        ) {
          error = "cannot write test worker diagnostic pipe";
          break;
        }
      }
      pipe.close_parent_writer();
      pipe.finish();
      result.limit_exceeded = pipe.limit_exceeded();
      result.write_failed = pipe.failed() || !error.empty();
      if (pipe.failed() && error.empty()) {
        error = pipe.failure();
      }
    }

    std::ifstream input(output, std::ios::binary);
    if (!input) {
      error = "cannot read retained test worker log";
      result.write_failed = true;
      return result;
    }
    result.retained.assign(
      std::istreambuf_iterator<char> {input},
      std::istreambuf_iterator<char> {}
    );
    return result;
  }

  bool abort_worker_log_pipe_with_open_writer_for_test(
    const fs::path &output,
    std::string &error
  ) {
    auto *raw_output = open_binary_log(output);
    if (!raw_output) {
      error = "cannot create test worker log";
      return false;
    }
    bounded_native_worker_log_pipe_t pipe {raw_output};
    if (!pipe.initialize(error)) {
      return false;
    }

    const auto source_handle = reinterpret_cast<HANDLE>(
      _get_osfhandle(_fileno(pipe.child_stream()))
    );
    HANDLE duplicate_handle = INVALID_HANDLE_VALUE;
    if (
      source_handle == INVALID_HANDLE_VALUE ||
      !DuplicateHandle(
        GetCurrentProcess(),
        source_handle,
        GetCurrentProcess(),
        &duplicate_handle,
        0,
        FALSE,
        DUPLICATE_SAME_ACCESS
      )
    ) {
      error =
        "cannot duplicate test worker diagnostic writer (Windows error " +
        std::to_string(GetLastError()) + ")";
      pipe.abort();
      return false;
    }
    pipe.close_parent_writer();
    pipe.abort();
    CloseHandle(duplicate_handle);

    if (pipe.failed()) {
      error = pipe.failure();
      return false;
    }
    return true;
  }

  std::size_t native_worker_log_max_bytes_for_test() {
    return max_native_worker_log_bytes;
  }

  std::string build_worker_command_for_test(
    const fs::path &sunshine_executable,
    const fs::path &sunshine_config,
    const fs::path &worker_spec,
    const std::string_view worker_spec_sha256
  ) {
    return build_worker_command(
      sunshine_executable,
      sunshine_config,
      worker_spec,
      worker_spec_sha256
    );
  }
#endif
}  // namespace offline_sbs
