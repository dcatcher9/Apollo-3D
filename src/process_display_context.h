/**
 * @file src/process_display_context.h
 * @brief Shared client/display metadata retained independently of an RTSP transport.
 */
#pragma once

#include <cstdint>
#include <string>

#ifdef _WIN32
  #include <guiddef.h>
#endif

namespace proc {
  /**
   * The process and its original launch share this separately allocated record. Process
   * publication, normalization and successful resume/live-mode commits mutate the same object,
   * preserving observations through old transport aliases without retaining their resources.
   */
  struct process_display_context_t {
    std::string device_name;
    std::string unique_id;
    int width;
    int height;
    int fps;
    bool enable_hdr;
    bool virtual_display;
    bool virtual_display_only = false;
    std::uint32_t scale_factor;
    int sbs_mode = 0;
#ifdef _WIN32
    GUID display_guid {};
#endif
  };
}  // namespace proc
