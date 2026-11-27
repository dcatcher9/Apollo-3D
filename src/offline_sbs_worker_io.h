/**
 * @file src/offline_sbs_worker_io.h
 * @brief Worker-private I/O: UTF-8 paths, atomic files, and bounded child processes.
 *
 * Every FFmpeg/FFprobe child the worker starts runs through child_process_t: restricted handle
 * inheritance, a kill-on-close job object, private overlapped pipes, and a bounded log capture.
 */
#pragma once

#include "offline_sbs_filesystem.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <windows.h>

namespace offline_sbs::worker_detail {
  namespace fs = std::filesystem;

  class worker_error: public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
  };

  inline constexpr auto child_poll = std::chrono::milliseconds(20);
  inline constexpr auto child_timeout = std::chrono::hours(12);
  inline constexpr auto frame_io_timeout = std::chrono::minutes(5);

  // Native separators: these strings go to FFmpeg command lines and Windows logs.
  std::string path_utf8(const fs::path &path);
  fs::path path_from_utf8(const std::string &value);
  std::string lower(std::string value);

  using safe_filesystem::publication_e;

  void write_bytes_atomic(
    const fs::path &path,
    const std::vector<std::uint8_t> &bytes,
    publication_e publication = publication_e::durable
  );
  void write_json_atomic(
    const fs::path &path,
    const nlohmann::json &value,
    publication_e publication = publication_e::durable
  );
  void write_json_atomic_bounded(
    const fs::path &path,
    const nlohmann::json &value,
    std::uintmax_t max_bytes,
    std::string_view description
  );
  void write_text_atomic(const fs::path &path, const std::string &text);
  void remove_file_checked(const fs::path &path);

  void validate_windows_command_line_capacity(
    const std::vector<std::string> &arguments,
    std::string_view description
  );

  class bounded_child_log_t;

  class child_process_t {
  public:
    child_process_t();
    child_process_t(const child_process_t &) = delete;
    child_process_t &operator=(const child_process_t &) = delete;
    child_process_t(child_process_t &&other) noexcept;
    child_process_t &operator=(child_process_t &&other) noexcept;
    ~child_process_t();
    static child_process_t launch(
      const std::vector<std::string> &arguments,
      const fs::path &working_directory,
      const fs::path &log_path,
      const std::optional<fs::path> &stdout_path = std::nullopt,
      const bool pipe_stdout = false,
      const bool pipe_stdin = false
    );
    [[nodiscard]] bool running() const;
    [[nodiscard]] int exit_code() const;
    void write_exact(const void *source, std::size_t size, const std::chrono::steady_clock::time_point deadline);
    void cancel_writes() noexcept;
    void close_stdin() noexcept;
    void read_exact(void *destination, const std::size_t size);
    std::string read_line(const std::size_t limit);
    std::size_t read_some(
      void *destination,
      const std::size_t capacity,
      const std::chrono::steady_clock::time_point absolute_deadline
    );
    bool read_one_if_available();
    int wait(const std::chrono::milliseconds timeout);
    void terminate() noexcept;

  private:
    HANDLE process_ = nullptr;
    HANDLE thread_ = nullptr;
    HANDLE stdout_read_ = nullptr;
    HANDLE stdout_read_event_ = nullptr;
    HANDLE stdout_child_write_ = nullptr;
    HANDLE stdout_file_ = nullptr;
    std::unique_ptr<bounded_child_log_t> log_capture_;
    HANDLE stdin_null_ = nullptr;
    HANDLE stdin_write_ = nullptr;
    HANDLE stdin_write_event_ = nullptr;
    std::atomic_bool writes_cancelled_ {false};
    HANDLE job_ = nullptr;
  };

  void run_logged(
    const std::vector<std::string> &command,
    const fs::path &working_directory,
    const fs::path &log_path,
    const std::optional<fs::path> &stdout_path = std::nullopt
  );

  nlohmann::json run_logged_json_bounded(
    const std::vector<std::string> &command,
    const fs::path &working_directory,
    const fs::path &log_path,
    std::uintmax_t max_bytes,
    std::string_view description
  );
}  // namespace offline_sbs::worker_detail
