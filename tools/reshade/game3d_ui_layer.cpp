// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_layer.h"
#include "addon_lifetime.h"
#include "game3d_ui_detection_contract.h"

#include <d3d11.h>
#include <d3d12.h>
#include <dxgi.h>

#include <algorithm>
#include <mutex>

namespace sunshine_game3d::ui_layer {
  bool layer_tracker::clear(std::uint64_t resource, std::uint64_t now_ms) {
    if (!resource) return false;
    const auto recent = [&](const entry &value) {
      return value.resource && now_ms >= value.last_ms && now_ms - value.last_ms <= max_clear_gap_ms;
    };
    // Otherwise replace a free or expired entry, then the shortest streak,
    // cleared longest ago: transient targets never displace a confirmed layer.
    const auto worse = [&](const entry &a, const entry &b) {
      return !recent(a) || (recent(b) && (a.streak < b.streak || (a.streak == b.streak && a.order < b.order)));
    };
    auto *slot = &entries_[0];
    bool found = false;
    for (auto &value : entries_) {
      if (value.resource == resource) { slot = &value; found = true; break; }
      if (worse(value, *slot)) slot = &value;
    }
    if (!found) *slot = {resource};
    // The streak counts frames: repeated clears within one frame count once.
    if (!recent(*slot)) slot->streak = 1;
    else if (slot->frame != frame_) ++slot->streak;
    slot->frame = frame_;
    slot->last_ms = now_ms;
    slot->order = ++order_;
    if (resource != active_ || captured_since_present_) return false;
    captured_since_present_ = true;
    return true;
  }

  void layer_tracker::present(std::uint64_t now_ms) {
    ++frame_;
    captured_since_present_ = false;
    const entry *best = nullptr;
    for (const auto &value : entries_)
      if (confirmed(value, now_ms) && (!best || value.order > best->order)) best = &value;
    active_ = best ? best->resource : 0;
  }

  namespace {
    bool transparent_black(const float color[4]) {
      return color[0] == 0.f && color[1] == 0.f && color[2] == 0.f && color[3] == 0.f;
    }

    // A copy recorded into a game command list may execute a frame or more
    // later; a withdrawn copy is destroyed only after this long.
    constexpr std::uint64_t retire_delay_ms = 2000;

    struct retired {
      api::device *device{};
      api::resource copy{};
      std::uint64_t tick{};
    };

    struct live_state {
      layer_tracker tracker;
      api::resource copy{};
      std::uint32_t width{}, height{};
      api::format format{};
      std::uint64_t capture_id{}, tick{};
    };

    // The live copy is recorded only while UI detection asked for the layer
    // this recently (latest() is called on every Present that wants it).
    constexpr std::uint64_t demand_window_ms = 1000;

    struct state_t {
      std::mutex mutex;
      bool armed{};
      std::uint64_t demand_ms{};
      api::device *device{};
      std::uint32_t width{}, height{};
      std::vector<std::uint64_t> back_buffers;
      std::vector<candidate> candidates;
      std::vector<retired> graveyard;
      live_state live;
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
        if (now - r.tick >= retire_delay_ms) { if (r.device) r.device->destroy_resource(r.copy); }
        else *keep++ = r;
      }
      s.graveyard.erase(keep, s.graveyard.end());
    }

    // Requires the state lock. Retires census copies that no dump took.
    void drop_candidates(state_t &s) {
      for (auto &c : s.candidates)
        if (c.copy.handle) s.graveyard.push_back({s.device, c.copy, GetTickCount64()});
      s.candidates.clear();
    }

    // Records the target's previous-frame content into copy (in copy_dest
    // state), leaving copy shader-readable and the target as the clear needs it.
    void record_copy(api::command_list *commands, api::resource target, api::resource copy) {
      commands->barrier(target, api::resource_usage::render_target, api::resource_usage::copy_source);
      commands->copy_resource(target, copy);
      commands->barrier(target, api::resource_usage::copy_source, api::resource_usage::render_target);
      commands->barrier(copy, api::resource_usage::copy_dest, api::resource_usage::shader_resource);
    }

    // Requires the state lock. Copies the active layer's previous-frame content
    // into the add-on's persistent texture, before the game's clear.
    void capture_live(state_t &s, api::command_list *commands, api::device *device, api::resource resource,
        const api::resource_desc &desc) {
      auto &live = s.live;
      const auto typed = api::format_to_default_typed(desc.texture.format, 0);
      if (live.copy.handle && (live.width != desc.texture.width || live.height != desc.texture.height || live.format != typed)) {
        s.graveyard.push_back({device, live.copy, GetTickCount64()});
        live.copy = {};
      }
      if (!live.copy.handle) {
        const api::resource_desc copy_desc(desc.texture.width, desc.texture.height, 1, 1, typed, 1, api::memory_heap::default_,
          api::resource_usage::copy_dest | api::resource_usage::copy_source | api::resource_usage::shader_resource);
        if (!device->create_resource(copy_desc, nullptr, api::resource_usage::shader_resource, &live.copy)) {
          live.copy = {};
          return;
        }
        live.width = desc.texture.width; live.height = desc.texture.height; live.format = typed;
      }
      commands->barrier(live.copy, api::resource_usage::shader_resource, api::resource_usage::copy_dest);
      record_copy(commands, resource, live.copy);
      ++live.capture_id;
      live.tick = GetTickCount64();
      // The Present count at the copy: it holds the previous Present's frame.
      live.tracker.copied();
    }

    // Requires the state lock. The dump census: each armed qualifying target once.
    void census(state_t &s, api::command_list *commands, api::device *device, api::resource resource,
        const api::resource_desc &desc) {
      const auto known = std::find_if(s.candidates.begin(), s.candidates.end(),
        [&](const candidate &c) { return c.source == resource.handle; });
      if (known != s.candidates.end()) {
        ++known->clears;
        return;
      }
      if (s.candidates.size() >= max_candidates) return;
      candidate c;
      c.source = resource.handle;
      c.width = desc.texture.width;
      c.height = desc.texture.height;
      const auto typed = api::format_to_default_typed(desc.texture.format, 0);
      c.format = static_cast<std::uint32_t>(typed);
      c.clears = 1;
      const api::resource_desc copy_desc(c.width, c.height, 1, 1, typed, 1, api::memory_heap::default_,
        api::resource_usage::copy_dest | api::resource_usage::shader_resource);
      if (device->create_resource(copy_desc, nullptr, api::resource_usage::copy_dest, &c.copy)) {
        record_copy(commands, resource, c.copy);
        c.status = "captured_before_clear";
      } else {
        c.copy = {};
        c.status = "copy_allocation_failed";
      }
      s.candidates.push_back(c);
    }

    bool on_clear(api::command_list *commands, api::resource_view view, const float color[4], std::uint32_t, const api::rect *) {
      auto &s = state();
      // Cheap rejection before any lock: only exact transparent black qualifies.
      if (!transparent_black(color)) return false;
      try {
        auto *device = commands->get_device();
        const auto resource = device->get_resource_from_view(view);
        if (!resource.handle) return false;
        std::lock_guard<std::mutex> lock(s.mutex);
        if (device != s.device ||
            std::find(s.back_buffers.begin(), s.back_buffers.end(), resource.handle) != s.back_buffers.end()) return false;
        const auto desc = device->get_resource_desc(resource);
        if (!qualifies(desc, color, s.width, s.height)) return false;
        // Clearing requires the render-target state on every API, so the
        // previous frame's layer can be copied here before the clear erases it.
        const auto now = GetTickCount64();
        if (s.live.tracker.clear(resource.handle, now) && s.demand_ms && now - s.demand_ms <= demand_window_ms)
          capture_live(s, commands, device, resource, desc);
        if (s.armed) census(s, commands, device, resource, desc);
      } catch (...) {
        // A failure never propagates into the game's command recording.
      }
      return false; // Never skip the game's clear.
    }
  }

  bool alpha_format(api::format format) {
    // A 2-bit alpha (R10G10B10A2) cannot hold blended UI coverage. The Witcher
    // 3 clears such a scene target with FG on; taken as its UI layer it
    // claimed the whole frame. While the layer is one candidate (the tracker's
    // active choice) this refusal stays: a 2-bit target chosen as the layer
    // would displace the real UI layer from its slot.
    switch (api::format_to_typeless(format)) {
      case api::format::r8g8b8a8_typeless:
      case api::format::b8g8r8a8_typeless:
      case api::format::r16g16b16a16_typeless:
      case api::format::r32g32b32a32_typeless:
        return true;
      default:
        return false;
    }
  }

  std::uint32_t detection_flags(api::format format) {
    return ui_detection::layer_detection_flags(ui_detection::float_layer_format(static_cast<std::uint32_t>(format)));
  }

  bool qualifies(const api::resource_desc &desc, const float color[4], std::uint32_t width, std::uint32_t height) {
    return width && height && desc.type == api::resource_type::texture_2d && desc.texture.width == width &&
      desc.texture.height == height && desc.texture.samples == 1 && desc.texture.levels == 1 &&
      desc.texture.depth_or_layers == 1 && alpha_format(desc.texture.format) && transparent_black(color);
  }

  bool latest(api::device *device, std::uint64_t now_ms, live_capture &out) {
    out = {};
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.demand_ms = now_ms;
    const auto &live = s.live;
    if (!device || device != s.device || !live.copy.handle || !live.capture_id || !live.tracker.active() ||
        now_ms < live.tick || now_ms - live.tick > max_clear_gap_ms) return false;
    out = {live.copy, live.capture_id, live.tick, static_cast<std::uint32_t>(live.format), live.tracker.presents_since_copy()};
    return true;
  }

  namespace {
    // Copies are destroyed through their own device while it still exists.
    void on_destroy_device(api::device *device) {
      auto &s = state();
      std::lock_guard<std::mutex> lock(s.mutex);
      auto keep = s.graveyard.begin();
      for (auto &r : s.graveyard) {
        if (r.device == device) device->destroy_resource(r.copy);
        else *keep++ = r;
      }
      s.graveyard.erase(keep, s.graveyard.end());
      for (auto &c : s.candidates) if (s.device == device && c.copy.handle) device->destroy_resource(c.copy);
      if (s.device == device) {
        if (s.live.copy.handle) device->destroy_resource(s.live.copy);
        s.live = {};
        s.candidates.clear();
        s.back_buffers.clear();
        s.device = nullptr;
      }
    }
  }

  void register_events() {
    reshade::register_event<reshade::addon_event::clear_render_target_view>(sunshine_addon_lifetime::guarded<on_clear>);
    reshade::register_event<reshade::addon_event::destroy_device>(sunshine_addon_lifetime::guarded<on_destroy_device>);
  }

  void unregister_events() {
    reshade::unregister_event<reshade::addon_event::clear_render_target_view>(sunshine_addon_lifetime::guarded<on_clear>);
    reshade::unregister_event<reshade::addon_event::destroy_device>(sunshine_addon_lifetime::guarded<on_destroy_device>);
    cancel();
  }

  void observe_output(api::swapchain *swapchain) {
    auto &s = state();
    auto *device = swapchain->get_device();
    // Native DXGI, as the exporter reads back buffers; identities only.
    auto *native = reinterpret_cast<IDXGISwapChain *>(swapchain->get_native());
    DXGI_SWAP_CHAIN_DESC desc{};
    if (!native || FAILED(native->GetDesc(&desc))) return;
    // Enumerated on every Present: a resize may recreate them at the same size.
    // (ReShade's own back-buffer getters crashed under MinGW.)
    const bool d3d12 = device->get_api() == api::device_api::d3d12;
    std::array<std::uint64_t, 16> buffers{};
    UINT count = 0;
    for (; count < desc.BufferCount && count < buffers.size(); ++count) {
      IUnknown *buffer = nullptr;
      if (FAILED(native->GetBuffer(count, d3d12 ? __uuidof(ID3D12Resource) : __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&buffer))))
        break;
      buffers[count] = reinterpret_cast<std::uint64_t>(buffer);
      buffer->Release();
    }
    const auto now = GetTickCount64();
    std::lock_guard<std::mutex> lock(s.mutex);
    // Every retired copy's device is alive: on_destroy_device releases its own.
    reap(s);
    if (s.armed && s.device && s.device != device) return;
    if (s.live.copy.handle && (s.device != device || s.width != desc.BufferDesc.Width || s.height != desc.BufferDesc.Height)) {
      s.graveyard.push_back({s.device, s.live.copy, now});
      s.live = {};
    } else if (s.device != device) s.live = {};
    s.device = device;
    s.width = desc.BufferDesc.Width;
    s.height = desc.BufferDesc.Height;
    s.back_buffers.assign(buffers.begin(), buffers.begin() + count);
    s.live.tracker.present(now);
  }

  void arm() {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    drop_candidates(s);
    s.armed = true;
  }

  std::vector<candidate> take(api::device *&device) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.armed = false;
    device = s.device;
    // The live tracker's choice is what UI detection reads; the others are candidates only.
    for (auto &c : s.candidates) c.active = c.source && c.source == s.live.tracker.active();
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
    s.armed = false;
    drop_candidates(s);
  }
}
