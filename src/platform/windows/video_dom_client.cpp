/**
 * @file src/platform/windows/video_dom_client.cpp
 * @brief Asynchronous, fail-closed client for the Chromium video-DOM helper.
 */

#include "video_dom_client.h"

#include "misc.h"
#include "src/platform/common.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifndef BOOST_PROCESS_VERSION
  #define BOOST_PROCESS_VERSION 1
#endif

#include <boost/filesystem/path.hpp>
#include <boost/process/v1/child.hpp>
#include <boost/process/v1/environment.hpp>
#include <boost/process/v1/group.hpp>
#include <fcntl.h>
#include <io.h>
#include <windows.h>

namespace platf::video_dom {
  using namespace std::chrono_literals;
  namespace bp = boost::process::v1;
  namespace fs = std::filesystem;

  namespace {
    constexpr std::string_view protocol_prefix = "SUNSHINE_VIDEO_DOM_V1";
    constexpr std::size_t protocol_field_count = 11;
    constexpr std::size_t maximum_protocol_line_bytes = 512;
    constexpr auto stale_after = 2500ms;
    // A normal full traversal is bounded to two seconds and may begin just before the one-second
    // heartbeat is due. A small scheduling margin avoids killing that valid scan; longer silence
    // is a potentially hung cross-process COM call, which is why the helper is out-of-process.
    constexpr auto watchdog_after = 3500ms;
    constexpr auto restart_delay = 1500ms;
    constexpr auto missing_helper_retry = 5s;
    constexpr auto termination_grace = 1s;

    [[nodiscard]] bool rect_valid(const rect_t &rect) noexcept {
      return rect.right > rect.left && rect.bottom > rect.top;
    }

    template<class Integer>
    [[nodiscard]] bool parse_integer(std::string_view text, Integer &value) noexcept {
      if (text.empty()) {
        return false;
      }
      Integer parsed {};
      const auto [end, error] = std::from_chars(
        text.data(),
        text.data() + text.size(),
        parsed,
        10
      );
      if (error != std::errc {} || end != text.data() + text.size()) {
        return false;
      }
      value = parsed;
      return true;
    }

    [[nodiscard]] std::optional<status_e> parse_status(
      const std::string_view token
    ) noexcept {
      if (token == "ok") {
        return status_e::ok;
      }
      if (token == "ok-fullscreen") {
        return status_e::ok_fullscreen;
      }
      if (token == "no-foreground") {
        return status_e::no_foreground;
      }
      if (token == "unsupported") {
        return status_e::unsupported;
      }
      if (token == "unavailable") {
        return status_e::unavailable;
      }
      if (token == "accessibility") {
        return status_e::accessibility;
      }
      if (token == "warming") {
        return status_e::warming;
      }
      if (token == "incomplete") {
        return status_e::incomplete;
      }
      if (token == "changed") {
        return status_e::changed;
      }
      if (token == "no-video") {
        return status_e::no_video;
      }
      if (token == "ambiguous") {
        return status_e::ambiguous;
      }
      return std::nullopt;
    }

    bool stop_aware_sleep(
      const std::stop_token stop,
      const std::chrono::milliseconds duration
    ) {
      std::mutex mutex;
      std::condition_variable_any changed;
      std::unique_lock lock {mutex};
      return changed.wait_for(lock, stop, duration, []() {
        return false;
      });
    }

    [[nodiscard]] std::optional<fs::path> executable_path() {
      std::vector<wchar_t> buffer(512);
      for (;;) {
        SetLastError(ERROR_SUCCESS);
        const DWORD length = GetModuleFileNameW(
          nullptr,
          buffer.data(),
          static_cast<DWORD>(buffer.size())
        );
        if (length == 0) {
          return std::nullopt;
        }
        if (length < buffer.size() - 1) {
          return fs::path {std::wstring_view(buffer.data(), length)};
        }
        if (buffer.size() >= 32768) {
          return std::nullopt;
        }
        buffer.resize(std::min<std::size_t>(buffer.size() * 2, 32768));
      }
    }

    [[nodiscard]] std::optional<fs::path> helper_path() {
      const auto executable = executable_path();
      if (!executable) {
        return std::nullopt;
      }
      const auto directory = executable->parent_path();
      const std::array candidates {
        directory / L"tools" / L"video-dom-info.exe",
        directory / L"video-dom-info.exe",
      };
      std::error_code error;
      for (const auto &candidate : candidates) {
        error.clear();
        if (fs::is_regular_file(candidate, error) && !error) {
          const auto canonical = fs::canonical(candidate, error);
          if (!error) {
            return canonical;
          }
        }
      }
      return std::nullopt;
    }

    [[nodiscard]] std::string quote_process_argument(const fs::path &path) {
      auto utf8 = platf::to_utf8(path.native());
      if (utf8.find('"') != std::string::npos) {
        return {};
      }
      return '"' + utf8 + '"';
    }

    class protocol_pipe_t {
    public:
      protocol_pipe_t() = default;

      ~protocol_pipe_t() {
        close_writer();
        close_reader();
      }

      protocol_pipe_t(const protocol_pipe_t &) = delete;
      protocol_pipe_t &operator=(const protocol_pipe_t &) = delete;

      [[nodiscard]] bool initialize() {
        HANDLE writer = INVALID_HANDLE_VALUE;
        if (!CreatePipe(&reader_, &writer, nullptr, 64u * 1024u)) {
          reader_ = INVALID_HANDLE_VALUE;
          return false;
        }
        const int descriptor = _open_osfhandle(
          reinterpret_cast<std::intptr_t>(writer),
          _O_BINARY | _O_WRONLY
        );
        if (descriptor == -1) {
          CloseHandle(writer);
          close_reader();
          return false;
        }
        writer_ = _fdopen(descriptor, "wb");
        if (!writer_) {
          _close(descriptor);
          close_reader();
          return false;
        }
        return true;
      }

      [[nodiscard]] FILE *writer() const noexcept {
        return writer_;
      }

      void close_writer() noexcept {
        if (writer_) {
          std::fclose(writer_);
          writer_ = nullptr;
        }
      }

      enum class read_status_e {
        ok,
        empty,
        closed,
        failed,
      };

      [[nodiscard]] read_status_e read_available(std::string &output) {
        DWORD available = 0;
        if (!PeekNamedPipe(reader_, nullptr, 0, nullptr, &available, nullptr)) {
          return GetLastError() == ERROR_BROKEN_PIPE ?
                   read_status_e::closed :
                   read_status_e::failed;
        }
        if (available == 0) {
          return read_status_e::empty;
        }

        std::array<char, 1024> bytes {};
        DWORD read = 0;
        if (!ReadFile(
              reader_,
              bytes.data(),
              std::min<DWORD>(available, static_cast<DWORD>(bytes.size())),
              &read,
              nullptr
            )) {
          return GetLastError() == ERROR_BROKEN_PIPE ?
                   read_status_e::closed :
                   read_status_e::failed;
        }
        if (read == 0) {
          return read_status_e::closed;
        }
        output.append(bytes.data(), read);
        return read_status_e::ok;
      }

    private:
      void close_reader() noexcept {
        if (reader_ != INVALID_HANDLE_VALUE) {
          CloseHandle(reader_);
          reader_ = INVALID_HANDLE_VALUE;
        }
      }

      HANDLE reader_ {INVALID_HANDLE_VALUE};
      FILE *writer_ {};
    };

    [[nodiscard]] bool wait_for_exit(
      const bp::child &child,
      const std::chrono::milliseconds timeout
    ) noexcept {
      if (!child.valid()) {
        return true;
      }
      return WaitForSingleObject(
               child.native_handle(),
               static_cast<DWORD>(timeout.count())
             ) == WAIT_OBJECT_0;
    }

    void terminate_and_reap(bp::child &child, bp::group &group) noexcept {
      if (!child.valid()) {
        return;
      }

      std::error_code group_error;
      if (!wait_for_exit(child, 0ms)) {
        group.terminate(group_error);
        if (!wait_for_exit(child, termination_grace)) {
          // Boost.Process closes the process handle in child::terminate(), which prevents a
          // bounded post-termination wait. Use Win32 directly and retain the handle instead.
          if (TerminateProcess(child.native_handle(), EXIT_FAILURE)) {
            (void) wait_for_exit(child, termination_grace);
          }
        }
      }

      // Windows has no zombie process to reap: closing the process handle is sufficient after
      // the bounded wait. Detach so Boost's destructor cannot issue another termination attempt.
      child.detach();
      if (!group_error) {
        group.detach();
      }
    }

    class controller_t {
    public:
      ~controller_t() {
        worker_.request_stop();
        if (worker_.joinable()) {
          worker_.join();
        }
      }

      [[nodiscard]] bool acquire() noexcept {
        try {
          std::lock_guard lock {lifecycle_mutex_};
          if (lifecycle_.acquire() != detail::lifecycle_action_e::start_worker) {
            return true;
          }
          if (start_worker_locked()) {
            return true;
          }

          lifecycle_.worker_start_failed();
          (void) lifecycle_.release();
          publish_noexcept(status_e::launch_failed);
          return false;
        } catch (...) {
          publish_noexcept(status_e::launch_failed);
          return false;
        }
      }

      void release() noexcept {
        std::jthread stopping_worker;
        try {
          std::lock_guard lock {lifecycle_mutex_};
          const auto action = lifecycle_.release();
          if (action == detail::lifecycle_action_e::publish_stopped) {
            publish_noexcept(status_e::stopped);
            return;
          }
          if (action != detail::lifecycle_action_e::stop_worker) {
            return;
          }
          worker_.request_stop();
          stopping_worker = std::move(worker_);
        } catch (...) {
          publish_noexcept(status_e::launch_failed);
          return;
        }

        if (stopping_worker.joinable()) {
          stopping_worker.join();
        }

        try {
          std::lock_guard lock {lifecycle_mutex_};
          switch (lifecycle_.worker_stopped()) {
            case detail::lifecycle_action_e::start_worker:
              if (!start_worker_locked()) {
                lifecycle_.worker_start_failed();
                publish_noexcept(status_e::launch_failed);
              }
              break;
            case detail::lifecycle_action_e::publish_stopped:
              publish_noexcept(status_e::stopped);
              break;
            case detail::lifecycle_action_e::none:
            case detail::lifecycle_action_e::stop_worker:
              break;
          }
        } catch (...) {
          publish_noexcept(status_e::launch_failed);
        }
      }

      [[nodiscard]] snapshot_ptr latest() const noexcept {
        return snapshot_.load(std::memory_order_acquire);
      }

    private:
      [[nodiscard]] bool start_worker_locked() noexcept {
        try {
          publish(status_e::starting);
          worker_ = std::jthread([this](const std::stop_token stop) {
            run(stop);
          });
          return true;
        } catch (...) {
          return false;
        }
      }

      void publish(const status_e status) {
        std::lock_guard lock {publish_mutex_};
        auto snapshot = std::make_shared<snapshot_t>();
        snapshot->status = status;
        snapshot->generation = generation_.fetch_add(
                                 1,
                                 std::memory_order_relaxed
                               ) +
                               1;
        snapshot->received_at = std::chrono::steady_clock::now();
        snapshot_.store(std::move(snapshot), std::memory_order_release);
      }

      void publish_noexcept(const status_e status) noexcept {
        try {
          publish(status);
        } catch (...) {
        }
      }

      void publish(const detail::protocol_record_t &record) {
        std::lock_guard lock {publish_mutex_};
        const auto received_at = std::chrono::steady_clock::now();
        const auto previous = snapshot_.load(std::memory_order_acquire);
        auto snapshot = std::make_shared<snapshot_t>();
        snapshot->status = record.status;
        snapshot->generation = generation_.fetch_add(
                                 1,
                                 std::memory_order_relaxed
                               ) +
                               1;
        snapshot->helper_sequence = record.sequence;
        snapshot->window = record.window;
        snapshot->process_id = record.process_id;
        snapshot->document_id = record.document_id;
        snapshot->video_id = record.video_id;
        snapshot->screen_rect = record.screen_rect;
        snapshot->received_at = received_at;
        snapshot->geometry_valid_since = detail::continued_geometry_valid_since(
          previous.get(),
          record,
          received_at
        );
        snapshot_.store(std::move(snapshot), std::memory_order_release);
      }

      void run(const std::stop_token stop) noexcept {
        try {
          run_loop(stop);
        } catch (...) {
          publish_noexcept(status_e::launch_failed);
        }
      }

      void run_loop(const std::stop_token stop) {
        while (!stop.stop_requested()) {
          const auto helper = helper_path();
          if (!helper) {
            publish(status_e::helper_missing);
            stop_aware_sleep(stop, missing_helper_retry);
            continue;
          }
          const auto quoted_helper = quote_process_argument(*helper);
          const auto expected_user = platf::active_user_id();
          if (quoted_helper.empty() || !expected_user) {
            publish(status_e::launch_failed);
            stop_aware_sleep(stop, restart_delay);
            continue;
          }

          protocol_pipe_t pipe;
          if (!pipe.initialize()) {
            publish(status_e::launch_failed);
            stop_aware_sleep(stop, restart_delay);
            continue;
          }

          try {
            bp::group group;
            auto environment = boost::this_process::environment();
            boost::filesystem::path working_directory {
              helper->parent_path().string()
            };
            std::error_code launch_error;
            auto child = platf::run_command_unelevated(
              false,
              quoted_helper + " --machine --interval-ms 100",
              working_directory,
              environment,
              pipe.writer(),
              launch_error,
              &group,
              expected_user
            );
            pipe.close_writer();
            if (launch_error || !child.valid()) {
              publish(status_e::launch_failed);
              stop_aware_sleep(stop, restart_delay);
              continue;
            }

            std::string pending;
            pending.reserve(1024);
            std::uint64_t previous_sequence = 0;
            auto last_heartbeat = std::chrono::steady_clock::now();
            bool stale_published = false;
            bool terminal_published = false;
            while (!stop.stop_requested()) {
              const auto read_status = pipe.read_available(pending);
              if (
                read_status == protocol_pipe_t::read_status_e::closed ||
                read_status == protocol_pipe_t::read_status_e::failed
              ) {
                if (!pending.empty()) {
                  publish(status_e::protocol_error);
                  terminal_published = true;
                }
                break;
              }

              bool protocol_failed = false;
              for (;;) {
                const auto newline = pending.find('\n');
                if (newline == std::string::npos) {
                  break;
                }
                std::string_view line {pending.data(), newline};
                if (!line.empty() && line.back() == '\r') {
                  line.remove_suffix(1);
                }
                const auto record = detail::parse_protocol_record(line);
                if (
                  !record ||
                  !detail::next_sequence(previous_sequence, record->sequence)
                ) {
                  protocol_failed = true;
                  break;
                }
                previous_sequence = record->sequence;
                last_heartbeat = std::chrono::steady_clock::now();
                stale_published = false;
                publish(*record);
                pending.erase(0, newline + 1);
              }
              if (
                protocol_failed ||
                pending.size() > maximum_protocol_line_bytes
              ) {
                publish(status_e::protocol_error);
                terminal_published = true;
                break;
              }

              const auto now = std::chrono::steady_clock::now();
              if (!stale_published && now - last_heartbeat > stale_after) {
                publish(status_e::stale);
                stale_published = true;
              }
              if (now - last_heartbeat > watchdog_after) {
                terminal_published = true;
                break;
              }

              std::error_code child_error;
              if (!child.running(child_error)) {
                break;
              }
              std::this_thread::sleep_for(25ms);
            }

            terminate_and_reap(child, group);
            if (!stop.stop_requested() && !terminal_published) {
              publish(status_e::helper_exited);
            }
          } catch (...) {
            publish(status_e::launch_failed);
          }

          if (!stop.stop_requested()) {
            stop_aware_sleep(stop, restart_delay);
          }
        }
      }

      mutable std::mutex lifecycle_mutex_;
      mutable std::mutex publish_mutex_;
      detail::lifecycle_t lifecycle_;
      std::jthread worker_;
      std::atomic<std::uint64_t> generation_ {};
      std::atomic<snapshot_ptr> snapshot_ {};
    };

    controller_t &controller() {
      static controller_t instance;
      return instance;
    }
  }  // namespace

  const char *status_name(const status_e status) noexcept {
    switch (status) {
      case status_e::starting:
        return "starting";
      case status_e::ok:
        return "ok";
      case status_e::ok_fullscreen:
        return "ok-fullscreen";
      case status_e::no_foreground:
        return "no-foreground";
      case status_e::unsupported:
        return "unsupported";
      case status_e::unavailable:
        return "unavailable";
      case status_e::accessibility:
        return "accessibility";
      case status_e::warming:
        return "warming";
      case status_e::incomplete:
        return "incomplete";
      case status_e::changed:
        return "changed";
      case status_e::no_video:
        return "no-video";
      case status_e::ambiguous:
        return "ambiguous";
      case status_e::helper_missing:
        return "helper-missing";
      case status_e::launch_failed:
        return "launch-failed";
      case status_e::helper_exited:
        return "helper-exited";
      case status_e::protocol_error:
        return "protocol-error";
      case status_e::stale:
        return "stale";
      case status_e::stopped:
        return "stopped";
    }
    return "unknown";
  }

  bool usable(
    const snapshot_t &snapshot,
    const std::chrono::steady_clock::time_point now,
    const std::chrono::milliseconds maximum_age
  ) noexcept {
    return carries_video_geometry(snapshot.status) &&
           rect_valid(snapshot.screen_rect) &&
           snapshot.window != 0 && snapshot.process_id != 0 &&
           snapshot.document_id != 0 && snapshot.video_id != 0 &&
           maximum_age.count() >= 0 && now >= snapshot.received_at &&
           now - snapshot.received_at <= maximum_age;
  }

  const char *mapping_status_name(const mapping_status_e status) noexcept {
    switch (status) {
      case mapping_status_e::ok:
        return "ok";
      case mapping_status_e::invalid_video_rect:
        return "invalid-video-rect";
      case mapping_status_e::invalid_capture_rect:
        return "invalid-capture-rect";
      case mapping_status_e::unsupported_rotation:
        return "unsupported-rotation";
      case mapping_status_e::extent_mismatch:
        return "extent-mismatch";
      case mapping_status_e::foreground_mismatch:
        return "foreground-mismatch";
      case mapping_status_e::outside_capture:
        return "outside-capture";
    }
    return "unknown";
  }

  mapping_result_t map_screen_rect_to_capture(
    const rect_t &video_screen_rect,
    const rect_t &capture_desktop_rect,
    const std::uint32_t capture_width,
    const std::uint32_t capture_height,
    const rotation_e rotation
  ) noexcept {
    if (!rect_valid(video_screen_rect)) {
      return {.status = mapping_status_e::invalid_video_rect};
    }
    if (!rect_valid(capture_desktop_rect) || capture_width == 0 || capture_height == 0) {
      return {.status = mapping_status_e::invalid_capture_rect};
    }
    if (rotation != rotation_e::identity) {
      return {.status = mapping_status_e::unsupported_rotation};
    }

    const auto desktop_width =
      static_cast<std::int64_t>(capture_desktop_rect.right) -
      capture_desktop_rect.left;
    const auto desktop_height =
      static_cast<std::int64_t>(capture_desktop_rect.bottom) -
      capture_desktop_rect.top;
    if (
      desktop_width != capture_width || desktop_height != capture_height ||
      capture_width > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
      capture_height > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())
    ) {
      return {.status = mapping_status_e::extent_mismatch};
    }

    constexpr std::int64_t endpoint_tolerance = 1;
    if (
      static_cast<std::int64_t>(video_screen_rect.left) <
        static_cast<std::int64_t>(capture_desktop_rect.left) - endpoint_tolerance ||
      static_cast<std::int64_t>(video_screen_rect.top) <
        static_cast<std::int64_t>(capture_desktop_rect.top) - endpoint_tolerance ||
      static_cast<std::int64_t>(video_screen_rect.right) >
        static_cast<std::int64_t>(capture_desktop_rect.right) + endpoint_tolerance ||
      static_cast<std::int64_t>(video_screen_rect.bottom) >
        static_cast<std::int64_t>(capture_desktop_rect.bottom) + endpoint_tolerance
    ) {
      return {.status = mapping_status_e::outside_capture};
    }

    const rect_t clipped {
      std::max(video_screen_rect.left, capture_desktop_rect.left),
      std::max(video_screen_rect.top, capture_desktop_rect.top),
      std::min(video_screen_rect.right, capture_desktop_rect.right),
      std::min(video_screen_rect.bottom, capture_desktop_rect.bottom),
    };
    if (!rect_valid(clipped)) {
      return {.status = mapping_status_e::outside_capture};
    }

    const rect_t pixels {
      clipped.left - capture_desktop_rect.left,
      clipped.top - capture_desktop_rect.top,
      clipped.right - capture_desktop_rect.left,
      clipped.bottom - capture_desktop_rect.top,
    };
    return {
      .status = mapping_status_e::ok,
      .capture_pixels = pixels,
      .normalized = {
        static_cast<float>(pixels.left) / capture_width,
        static_cast<float>(pixels.top) / capture_height,
        static_cast<float>(pixels.right) / capture_width,
        static_cast<float>(pixels.bottom) / capture_height,
      },
    };
  }

  namespace detail {
    lifecycle_action_e lifecycle_t::acquire() noexcept {
      ++leases_;
      if (worker_state_ == worker_state_e::stopped) {
        worker_state_ = worker_state_e::running;
        return lifecycle_action_e::start_worker;
      }
      return lifecycle_action_e::none;
    }

    lifecycle_action_e lifecycle_t::release() noexcept {
      if (leases_ == 0 || --leases_ != 0) {
        return lifecycle_action_e::none;
      }
      if (worker_state_ == worker_state_e::running) {
        worker_state_ = worker_state_e::stopping;
        return lifecycle_action_e::stop_worker;
      }
      return worker_state_ == worker_state_e::stopped ?
               lifecycle_action_e::publish_stopped :
               lifecycle_action_e::none;
    }

    lifecycle_action_e lifecycle_t::worker_stopped() noexcept {
      if (worker_state_ != worker_state_e::stopping) {
        return lifecycle_action_e::none;
      }
      if (leases_ != 0) {
        worker_state_ = worker_state_e::running;
        return lifecycle_action_e::start_worker;
      }
      worker_state_ = worker_state_e::stopped;
      return lifecycle_action_e::publish_stopped;
    }

    void lifecycle_t::worker_start_failed() noexcept {
      if (worker_state_ == worker_state_e::running) {
        worker_state_ = worker_state_e::stopped;
      }
    }

    bool usable_for_content(
      const snapshot_t &snapshot,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
      const std::chrono::steady_clock::time_point capture_now,
      const std::chrono::milliseconds maximum_age
    ) noexcept {
      return content_timestamp &&
             usable(snapshot, capture_now, maximum_age) &&
             snapshot.geometry_valid_since.time_since_epoch().count() != 0 &&
             *content_timestamp >= snapshot.geometry_valid_since &&
             *content_timestamp <= capture_now;
    }

    std::chrono::steady_clock::time_point continued_geometry_valid_since(
      const snapshot_t *previous,
      const protocol_record_t &record,
      const std::chrono::steady_clock::time_point received_at
    ) noexcept {
      if (!carries_video_geometry(record.status)) {
        return {};
      }
      if (
        previous && previous->status == record.status &&
        carries_video_geometry(previous->status) &&
        previous->geometry_valid_since.time_since_epoch().count() != 0 &&
        previous->window == record.window && previous->process_id == record.process_id &&
        previous->document_id == record.document_id && previous->video_id == record.video_id &&
        previous->screen_rect == record.screen_rect
      ) {
        return previous->geometry_valid_since;
      }
      return received_at;
    }

    std::optional<protocol_record_t> parse_protocol_record(
      const std::string_view line
    ) noexcept {
      if (
        line.empty() || line.size() > maximum_protocol_line_bytes ||
        line.find('\0') != std::string_view::npos ||
        line.find('\r') != std::string_view::npos ||
        line.find('\n') != std::string_view::npos
      ) {
        return std::nullopt;
      }

      std::array<std::string_view, protocol_field_count> fields {};
      std::size_t begin = 0;
      for (std::size_t index = 0; index < fields.size(); ++index) {
        const auto separator = line.find('\t', begin);
        if (index + 1 == fields.size()) {
          if (separator != std::string_view::npos) {
            return std::nullopt;
          }
          fields[index] = line.substr(begin);
        } else {
          if (separator == std::string_view::npos) {
            return std::nullopt;
          }
          fields[index] = line.substr(begin, separator - begin);
          begin = separator + 1;
        }
      }
      if (fields[0] != protocol_prefix) {
        return std::nullopt;
      }

      protocol_record_t record;
      const auto parsed_status = parse_status(fields[2]);
      if (
        !parsed_status ||
        !parse_integer(fields[1], record.sequence) || record.sequence == 0 ||
        !parse_integer(fields[3], record.window) ||
        !parse_integer(fields[4], record.process_id) ||
        !parse_integer(fields[5], record.document_id) ||
        !parse_integer(fields[6], record.video_id) ||
        !parse_integer(fields[7], record.screen_rect.left) ||
        !parse_integer(fields[8], record.screen_rect.top) ||
        !parse_integer(fields[9], record.screen_rect.right) ||
        !parse_integer(fields[10], record.screen_rect.bottom)
      ) {
        return std::nullopt;
      }
      record.status = *parsed_status;

      if (carries_video_geometry(record.status)) {
        if (
          record.window == 0 || record.process_id == 0 ||
          record.document_id == 0 || record.video_id == 0 ||
          !rect_valid(record.screen_rect)
        ) {
          return std::nullopt;
        }
      } else if (
        record.window != 0 || record.process_id != 0 ||
        record.document_id != 0 || record.video_id != 0 ||
        record.screen_rect != rect_t {}
      ) {
        return std::nullopt;
      }
      return record;
    }

    bool next_sequence(
      const std::uint64_t previous,
      const std::uint64_t candidate
    ) noexcept {
      if (previous == 0) {
        return candidate == 1;
      }
      return previous != std::numeric_limits<std::uint64_t>::max() &&
             candidate == previous + 1;
    }
  }  // namespace detail

  lease_t::lease_t():
      active_(controller().acquire()) {
  }

  lease_t::~lease_t() {
    if (active_) {
      controller().release();
    }
  }

  lease_t::lease_t(lease_t &&other) noexcept:
      active_(std::exchange(other.active_, false)) {
  }

  lease_t &lease_t::operator=(lease_t &&other) noexcept {
    if (this != &other) {
      if (active_) {
        controller().release();
      }
      active_ = std::exchange(other.active_, false);
    }
    return *this;
  }

  snapshot_ptr lease_t::latest() const noexcept {
    return active_ ? controller().latest() : snapshot_ptr {};
  }
}  // namespace platf::video_dom
