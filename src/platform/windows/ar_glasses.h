#pragma once

#include "src/platform/common.h"

#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ar_glasses {
  enum class presentation_mode_e {
    unsupported,
    normal,
    sbs_ai,
  };

  enum class device_decision_e {
    pending,
    approved,
    rejected,
  };

  struct device_info_t {
    std::string id;
    std::string name;
    device_decision_e decision = device_decision_e::pending;
    bool connected = false;
    bool auto_detected = false;
  };

  constexpr presentation_mode_e classify_mode(int width, int height) {
    if (height != 1080) {
      return presentation_mode_e::unsupported;
    }
    if (width == 1920) {
      return presentation_mode_e::normal;
    }
    if (width == 3840) {
      return presentation_mode_e::sbs_ai;
    }
    return presentation_mode_e::unsupported;
  }

  static_assert(classify_mode(1920, 1080) == presentation_mode_e::normal);
  static_assert(classify_mode(3840, 1080) == presentation_mode_e::sbs_ai);
  static_assert(classify_mode(3840, 2160) == presentation_mode_e::unsupported);

  /** Return whether a monitor model/name is specific enough to identify AR glasses automatically. */
  bool is_recognized_ar_display(std::string_view model_id, std::string_view friendly_name);

  /** Snapshot the persisted monitor decisions and their current connection state. */
  std::vector<device_info_t> devices();

  /** Approve or reject a discovered monitor model. Pending is not accepted from the UI. */
  bool set_device_decision(std::string_view id, device_decision_e decision);

  /** Atomically write a general configuration snapshot while injecting the latest device list. */
  bool write_config_with_devices(std::string_view contents);

  /** Monitor approved AR displays and own the local virtual-desktop presentation lifecycle. */
  std::unique_ptr<platf::deinit_t> init();

  /** Reserve virtual-display ownership for a remote launch and synchronously stop local AR. */
  bool remote_virtual_display_starting(std::chrono::milliseconds connect_timeout);

  /** Mark the reserved remote virtual display as actively streamed. */
  void remote_virtual_display_active();

  /** Release remote ownership after pause, termination, or launch failure. */
  void remote_virtual_display_ended();

  /** Return whether an active or connecting remote virtual display currently owns presentation. */
  bool remote_virtual_display_blocks_local();
}  // namespace ar_glasses
