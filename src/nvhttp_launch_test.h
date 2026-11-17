/**
 * @file src/nvhttp_launch_test.h
 * @brief Device-free edges for tests of the complete HTTP launch/resume transaction.
 */
#pragma once

#ifdef SUNSHINE_TESTS
  #include "nvhttp.h"
  #include "process.h"

  #include <chrono>
  #include <cstdint>
  #include <functional>
  #include <memory>

namespace nvhttp::testing {
  using response_t = std::shared_ptr<SimpleWeb::ServerBase<SunshineHTTPS>::Response>;
  using request_t = std::shared_ptr<SimpleWeb::ServerBase<SunshineHTTPS>::Request>;

  // Admission, authentication, lifecycle locking, GPU ownership and RTSP publication remain real.
  struct launch_native_hooks_t {
    std::function<int(proc::ctx_t &&, std::shared_ptr<rtsp_stream::launch_session_t>, bool)> execute;
    std::function<void()> terminate;
    std::function<int()> probe_encoders;
    std::function<int(std::shared_ptr<rtsp_stream::launch_session_t>)> reconfigure;
    std::function<bool(std::uint64_t, std::chrono::milliseconds, bool)> remote_starting;
    std::function<void(std::uint64_t, std::chrono::milliseconds)> remote_awaiting;
    std::function<void(std::uint64_t)> remote_ended;
  };

  void invoke_launch_handler_for_test(bool resume, response_t response, request_t request, launch_native_hooks_t &hooks);
}  // namespace nvhttp::testing

namespace proc {
  // A separate accessor avoids redefining the existing process_test_access from test_process.cpp.
  struct launch_transaction_test_access {
    static void seed(proc_t &process, ctx_t app, std::shared_ptr<rtsp_stream::launch_session_t> launch, std::uint64_t host_session_id, bool virtual_display, std::function<void()> primary_restored = {});
    static void clear(proc_t &process);
    static void publish_resume(proc_t &process, const rtsp_stream::launch_session_t &launch);
    static std::shared_ptr<rtsp_stream::launch_session_t> retained_launch(const proc_t &process);
    static std::uint32_t active_transport(const proc_t &process);
  };
}  // namespace proc
#endif
