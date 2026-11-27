/**
 * @file src/process_display_context_test.h
 * @brief Device-free edges for the real retained process/display contract.
 */
#pragma once

#ifdef SUNSHINE_TESTS
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
