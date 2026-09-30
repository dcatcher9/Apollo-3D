// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_layer.h"
#include "addon_lifetime.h"

#include <d3d11.h>
#include <d3d12.h>
#include <dxgi.h>

#include <algorithm>
#include <atomic>
#include <mutex>

namespace sunshine_game3d::ui_layer {
  namespace {
    // A copy recorded into a game command list may execute a frame or more
    // later; a withdrawn copy is destroyed only after this long.
    constexpr std::uint64_t retire_delay_ms = 2000;

    struct retired {
      api::device *device{};
      api::resource copy{};
      std::uint64_t tick{};
    };

    struct state_t {
      std::mutex mutex;
      std::atomic<bool> armed{false};
      api::device *device{};
      std::uint32_t width{}, height{};
      std::vector<std::uint64_t> back_buffers;
      std::vector<candidate> candidates;
      std::vector<retired> graveyard;
    };
    // Deliberately leaked: clear events can arrive during process exit.
    state_t &state() {
      static state_t *value = new state_t;
      return *value;
    }

    // Requires the state lock.
    void reap(state_t &s) {
      const auto now = GetTickCount64();
      auto keep = s.graveyard.begin();
      for (auto &r : s.graveyard) {
        if (now - r.tick >= retire_delay_ms) r.device->destroy_resource(r.copy);
        else *keep++ = r;
      }
      s.graveyard.erase(keep, s.graveyard.end());
    }

    bool on_clear(api::command_list *commands, api::resource_view view, const float color[4], std::uint32_t, const api::rect *) {
      auto &s = state();
      if (!s.armed.load(std::memory_order_acquire)) return false;
      try {
        auto *device = commands->get_device();
        const auto resource = device->get_resource_from_view(view);
        if (!resource.handle) return false;
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.armed.load(std::memory_order_relaxed) || device != s.device ||
            std::find(s.back_buffers.begin(), s.back_buffers.end(), resource.handle) != s.back_buffers.end()) return false;
        const auto desc = device->get_resource_desc(resource);
        if (!qualifies(desc, color, s.width, s.height)) return false;
        const auto known = std::find_if(s.candidates.begin(), s.candidates.end(),
          [&](const candidate &c) { return c.source == resource.handle; });
        if (known != s.candidates.end()) {
          ++known->clears;
          return false;
        }
        if (s.candidates.size() >= max_candidates) return false;
        candidate c;
        c.source = resource.handle;
        c.width = desc.texture.width;
        c.height = desc.texture.height;
        const auto typed = api::format_to_default_typed(desc.texture.format, 0);
        c.format = static_cast<std::uint32_t>(typed);
        c.clears = 1;
        // Clearing requires the render-target state on every API, so the
        // previous frame's layer can be copied here before the clear erases it.
        const api::resource_desc copy_desc(c.width, c.height, 1, 1, typed, 1, api::memory_heap::default_,
          api::resource_usage::copy_dest | api::resource_usage::shader_resource);
        if (device->create_resource(copy_desc, nullptr, api::resource_usage::copy_dest, &c.copy)) {
          commands->barrier(resource, api::resource_usage::render_target, api::resource_usage::copy_source);
          commands->copy_resource(resource, c.copy);
          commands->barrier(resource, api::resource_usage::copy_source, api::resource_usage::render_target);
          commands->barrier(c.copy, api::resource_usage::copy_dest, api::resource_usage::shader_resource);
          c.status = "captured_before_clear";
        } else {
          c.copy = {};
          c.status = "copy_allocation_failed";
        }
        s.candidates.push_back(c);
      } catch (...) {
        // A diagnostic failure never propagates into the game's command recording.
      }
      return false; // Never skip the game's clear.
    }
  }

  bool alpha_format(api::format format) {
    switch (api::format_to_typeless(format)) {
      case api::format::r8g8b8a8_typeless:
      case api::format::b8g8r8a8_typeless:
      case api::format::r10g10b10a2_typeless:
      case api::format::r16g16b16a16_typeless:
      case api::format::r32g32b32a32_typeless:
        return true;
      default:
        return false;
    }
  }

  bool qualifies(const api::resource_desc &desc, const float color[4], std::uint32_t width, std::uint32_t height) {
    return width && height && desc.type == api::resource_type::texture_2d && desc.texture.width == width &&
      desc.texture.height == height && desc.texture.samples == 1 && desc.texture.levels == 1 &&
      desc.texture.depth_or_layers == 1 && alpha_format(desc.texture.format) &&
      color[0] == 0.f && color[1] == 0.f && color[2] == 0.f && color[3] == 0.f;
  }

  void register_events() {
    reshade::register_event<reshade::addon_event::clear_render_target_view>(sunshine_addon_lifetime::guarded<on_clear>);
  }

  void unregister_events() {
    reshade::unregister_event<reshade::addon_event::clear_render_target_view>(sunshine_addon_lifetime::guarded<on_clear>);
    cancel();
  }

  void observe_output(api::swapchain *swapchain) {
    auto &s = state();
    auto *device = swapchain->get_device();
    // Native DXGI, as the exporter reads back buffers; identities only.
    auto *native = reinterpret_cast<IDXGISwapChain *>(swapchain->get_native());
    DXGI_SWAP_CHAIN_DESC desc{};
    if (!native || FAILED(native->GetDesc(&desc))) return;
    const bool d3d12 = device->get_api() == api::device_api::d3d12;
    std::vector<std::uint64_t> buffers;
    for (UINT i = 0; i < desc.BufferCount && i < 16; ++i) {
      IUnknown *buffer = nullptr;
      if (FAILED(native->GetBuffer(i, d3d12 ? __uuidof(ID3D12Resource) : __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&buffer))))
        break;
      buffers.push_back(reinterpret_cast<std::uint64_t>(buffer));
      buffer->Release();
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    // A copy owned by another (possibly destroyed) device is dropped, never
    // released through a stale device pointer.
    s.graveyard.erase(std::remove_if(s.graveyard.begin(), s.graveyard.end(),
      [device](const retired &r) { return r.device != device; }), s.graveyard.end());
    reap(s);
    if (s.armed.load(std::memory_order_relaxed) && s.device && s.device != device) return;
    s.device = device;
    s.width = desc.BufferDesc.Width;
    s.height = desc.BufferDesc.Height;
    s.back_buffers = std::move(buffers);
  }

  void arm() {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (auto &c : s.candidates)
      if (c.copy.handle) s.graveyard.push_back({s.device, c.copy, GetTickCount64()});
    s.candidates.clear();
    s.armed.store(true, std::memory_order_release);
  }

  std::vector<candidate> take(api::device *&device) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.armed.store(false, std::memory_order_release);
    device = s.device;
    std::vector<candidate> result;
    result.swap(s.candidates);
    return result;
  }

  void retire(api::device *device, api::resource copy) {
    if (!device || !copy.handle) return;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.graveyard.push_back({device, copy, GetTickCount64()});
  }

  void cancel() {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.armed.store(false, std::memory_order_release);
    for (auto &c : s.candidates)
      if (c.copy.handle) s.graveyard.push_back({s.device, c.copy, GetTickCount64()});
    s.candidates.clear();
  }
}
