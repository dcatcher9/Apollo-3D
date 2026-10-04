// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_frame_clock.h"

#include "addon_lifetime.h"
#include "game3d_ui_ticket.h"

#include <reshade.hpp>

#include <cstring>
#include <mutex>
#include <vector>

namespace sunshine_game3d::frame_clock {
  namespace {
    struct clocks {
      api::device *device {};
      api::command_queue *queue {};
      api::resource present {}, token {}, ring {};
      std::uint8_t *ring_data {};  // D3D12: the upload ring, mapped for the clocks' lifetime.
      std::uint32_t label {};
      bool failed {};  // Creation failed once: never retried for this device.
    };

    struct state_t {
      std::mutex mutex;
      std::vector<clocks> devices;
    };

    // Deliberately leaked: Presents and recordings can arrive during process exit.
    state_t &state() {
      static state_t *value = new state_t;
      return *value;
    }

    // Requires the state lock.
    clocks *find(state_t &s, api::device *device) {
      for (auto &c : s.devices) {
        if (c.device == device) {
          return &c;
        }
      }
      return nullptr;
    }

    void release(clocks &c) {
      retract_native(reinterpret_cast<void *>(c.device->get_native()));
      if (c.ring_data) {
        c.device->unmap_buffer_region(c.ring);
      }
      for (const auto resource : {c.present, c.token, c.ring}) {
        if (resource.handle) {
          c.device->destroy_resource(resource);
        }
      }
      c.present = c.token = c.ring = {};
      c.ring_data = nullptr;
    }

    // Requires the state lock. Creates the clocks in COMMON (D3D12 buffers
    // promote implicitly between copy destination and source) and, on D3D12,
    // the mapped upload ring.
    bool create(clocks &c) {
      if (c.present.handle) {
        return true;
      }
      if (c.failed) {
        return false;
      }
      auto *device = c.device;
      const api::resource_desc clock_desc(clock_bytes, api::memory_heap::default_, api::resource_usage::copy_source | api::resource_usage::copy_dest);
      const bool d3d12 = device->get_api() == api::device_api::d3d12;
      bool ok = device->create_resource(clock_desc, nullptr, api::resource_usage::general, &c.present) &&
                device->create_resource(clock_desc, nullptr, api::resource_usage::general, &c.token);
      if (ok && d3d12) {
        const api::resource_desc ring_desc(ring_entries * ring_entry_bytes, api::memory_heap::upload, api::resource_usage::copy_source);
        void *data = nullptr;
        ok = device->create_resource(ring_desc, nullptr, api::resource_usage::cpu_access, &c.ring) &&
             device->map_buffer_region(c.ring, 0, UINT64_MAX, api::map_access::write_only, &data) && data;
        c.ring_data = ok ? static_cast<std::uint8_t *>(data) : nullptr;
      }
      if (!ok) {
        release(c);
        c.failed = true;
        return false;
      }
      publish_native(reinterpret_cast<void *>(device->get_native()), reinterpret_cast<void *>(c.present.handle), reinterpret_cast<void *>(c.token.handle));
      return true;
    }

    void on_destroy_device(api::device *device) {
      auto &s = state();
      std::lock_guard<std::mutex> lock(s.mutex);
      for (auto it = s.devices.begin(); it != s.devices.end(); ++it) {
        if (it->device == device) {
          release(*it);
          s.devices.erase(it);
          return;
        }
      }
    }
  }  // namespace

  void advance(api::command_queue *queue, api::swapchain *swapchain) {
    if (!queue || !swapchain) {
      return;
    }
    auto *device = swapchain->get_device();
    if (!device) {
      return;
    }
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto *c = find(s, device);
    if (!c) {
      s.devices.push_back({device});
      c = &s.devices.back();
    }
    c->queue = queue;
    // 0 is never written: the label skips it on wraparound.
    if (!++c->label) {
      ++c->label;
    }
    if (!create(*c)) {
      return;
    }
    const std::uint32_t label = c->label;
    if (c->ring_data) {
      const auto offset = (label % ring_entries) * ring_entry_bytes;
      std::memcpy(c->ring_data + offset, &label, sizeof(label));
      if (auto *commands = queue->get_immediate_command_list()) {
        commands->copy_buffer_region(c->ring, offset, c->present, 0, sizeof(label));
      }
    } else {
      device->update_buffer_region(&label, c->present, 0, sizeof(label));
    }
  }

  std::uint32_t label(api::device *device) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto *c = find(s, device);
    return c ? c->label : 0u;
  }

  api::command_queue *presenting_queue(api::device *device) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto *c = find(s, device);
    return c ? c->queue : nullptr;
  }

  bool stamp(api::command_list *commands, api::resource destination, std::uint64_t offset) {
    if (!commands || !destination.handle) {
      return false;
    }
    api::resource present {}, token {};
    {
      auto &s = state();
      std::lock_guard<std::mutex> lock(s.mutex);
      const auto *c = find(s, commands->get_device());
      if (!c || !c->present.handle) {
        return false;
      }
      present = c->present;
      token = c->token;
    }
    commands->copy_buffer_region(present, 0, destination, offset + ui_ticket::stamp_present_offset, 4);
    commands->copy_buffer_region(token, 0, destination, offset + ui_ticket::stamp_token_offset, 4);
    return true;
  }

  std::uint32_t loaded_interposers() {
    std::uint32_t bits = 0;
    for (const auto &entry : ui_ticket::interposer_modules) {
      if (GetModuleHandleW(entry.module.data()) != nullptr) {
        bits |= entry.bit;
      }
    }
    return bits;
  }

  void register_events() {
    reshade::register_event<reshade::addon_event::destroy_device>(sunshine_addon_lifetime::guarded<on_destroy_device>);
  }

  void unregister_events() {
    reshade::unregister_event<reshade::addon_event::destroy_device>(sunshine_addon_lifetime::guarded<on_destroy_device>);
  }
}  // namespace sunshine_game3d::frame_clock
