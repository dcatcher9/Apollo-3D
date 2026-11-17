/**
 * @file src/process_display_context_test.h
 * @brief Device-free edges for the real retained process/display contract.
 */
#pragma once

#if defined(SUNSHINE_TESTS) && defined(_WIN32)
  // Winsock2 must precede the Windows headers pulled in by the process/RTSP declarations.
  #include <boost/asio/ip/tcp.hpp>

  #include "process.h"
  #include "rtsp.h"

namespace proc::testing {
  struct display_snapshot_t {
    std::string device_name;
    std::string unique_id;
    int width;
    int height;
    int fps;
    bool enable_hdr;
    bool virtual_display;
    bool virtual_display_only;
    std::uint32_t scale_factor;
    int sbs_mode;
    GUID display_guid;
  };

  struct native_edges_t {
    std::function<bool()> encoder_probing_allowed;
    std::function<bool()> recover_primary;
    std::function<bool(std::uint64_t, std::chrono::milliseconds, bool)> remote_starting;
    std::function<void(std::uint64_t, std::chrono::milliseconds)> remote_awaiting;
    std::function<void(std::uint64_t)> remote_ended;
    std::function<std::string(const std::string &)> map_display_name;
    std::function<std::vector<std::string>()> capture_display_names;
    std::function<void(bool)> start_hdr_worker;
    VDISPLAY::session_io_t display_io;
    display_topology_test_hook_t topology;
    std::function<std::optional<DEVMODEW>(std::wstring_view)> query_mode;
    std::function<LONG(std::wstring_view, int, int, int, bool)> change_mode;
    std::function<LONG(std::wstring_view, const VDISPLAY::session_mode_record_t &)> record_mode;
    std::function<bool(std::wstring_view)> baseline_active;
    std::function<bool()> settings_unavailable;
  };
}  // namespace proc::testing

namespace proc {
  struct process_display_context_test_access {
    static void configure(proc_t &process, const testing::native_edges_t &edges);
    static int execute(proc_t &process, ctx_t app, const std::shared_ptr<rtsp_stream::launch_session_t> &launch);
    static int reconfigure(proc_t &process, const std::shared_ptr<rtsp_stream::launch_session_t> &launch);
    static void set_display(rtsp_stream::launch_session_t &launch, const testing::display_snapshot_t &display);
    static testing::display_snapshot_t display(const rtsp_stream::launch_session_t &launch);
    static std::optional<testing::display_snapshot_t> retained(const proc_t &process);
    static bool shares_display_context(const proc_t &process, const rtsp_stream::launch_session_t &launch);
    static std::string environment(proc_t &process, const std::string &key);
    static std::uint32_t active_transport(const proc_t &process);
  };
}  // namespace proc

namespace rtsp_stream {
  void invoke_stale_announce_for_test(boost::asio::ip::tcp::socket &socket, launch_session_t &launch, std::string_view payload);
}

#endif
