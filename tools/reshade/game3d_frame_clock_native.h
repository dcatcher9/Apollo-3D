// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The ReShade-free half of the S3 frame clocks (game3d_frame_clock.h): the
// native clock buffers a capture owner that records natively (the Streamline
// capture owner, streamline_depth_capture.cpp) copies from or writes to,
// looked up by the native device. frame_clock publishes a device's clocks
// when it creates them and retracts them before it releases them.
#include <cstdint>
#include <mutex>
#include <vector>

namespace sunshine_game3d::frame_clock {
  // A clock buffer (an ID3D12Resource *, or an ID3D11Buffer *) and the byte
  // offset of its 4-byte clock word; null before the device's clocks exist.
  struct native_clock {
    void *resource {};
    std::uint64_t offset {};
  };

  namespace detail {
    struct native_entry {
      void *device {}, *present {}, *token {};
    };

    struct native_registry {
      std::mutex mutex;
      std::vector<native_entry> entries;
    };

    // Deliberately leaked: recordings can arrive during process exit.
    inline native_registry &registry() {
      static native_registry *value = new native_registry;
      return *value;
    }

    inline native_clock find(void *native_device, bool token) {
      auto &r = registry();
      std::lock_guard<std::mutex> lock(r.mutex);
      for (const auto &e : r.entries) {
        if (e.device == native_device) {
          return {token ? e.token : e.present, 0};
        }
      }
      return {};
    }
  }  // namespace detail

  // The present clock C_P and the token clock C_T of the device with this
  // native handle.
  inline native_clock native_present_clock(void *native_device) {
    return detail::find(native_device, false);
  }

  inline native_clock native_token_clock(void *native_device) {
    return detail::find(native_device, true);
  }

  // frame_clock's own bookkeeping.
  inline void publish_native(void *native_device, void *present, void *token) {
    auto &r = detail::registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (auto &e : r.entries) {
      if (e.device == native_device) {
        e.present = present;
        e.token = token;
        return;
      }
    }
    r.entries.push_back({native_device, present, token});
  }

  inline void retract_native(void *native_device) {
    auto &r = detail::registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (auto it = r.entries.begin(); it != r.entries.end(); ++it) {
      if (it->device == native_device) {
        r.entries.erase(it);
        return;
      }
    }
  }
}  // namespace sunshine_game3d::frame_clock
