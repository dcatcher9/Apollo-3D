// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_layer.h"
#include "addon_lifetime.h"
#include "game3d_frame_clock.h"
#include "game3d_ui_detection_contract.h"

#include <d3d11.h>
#include <d3d12.h>
#include <dxgi.h>

#include <algorithm>
#include <atomic>
#include <mutex>

namespace sunshine_game3d::ui_layer {
  void queue_watch::carried(std::uint64_t list) {
    if (!list || std::find(lists_.begin(), lists_.end(), list) != lists_.end()) return;
    // Oldest first, free entries last: a full watch drops its oldest list.
    if (lists_.back()) {
      std::rotate(lists_.begin(), lists_.begin() + 1, lists_.end());
      lists_.back() = list;
    } else *std::find(lists_.begin(), lists_.end(), std::uint64_t{0}) = list;
  }

  void queue_watch::reset(std::uint64_t list) {
    if (!list) return;
    const auto end = std::remove(lists_.begin(), lists_.end(), list);
    std::fill(end, lists_.end(), std::uint64_t{0});
  }

  void queue_watch::move(std::uint64_t list, std::uint64_t context) {
    const bool carries = context && std::find(lists_.begin(), lists_.end(), context) != lists_.end();
    reset(list);
    reset(context);
    if (carries) carried(list);
  }

  void queue_watch::transfer(std::uint64_t primary, std::uint64_t secondary) {
    if (secondary && std::find(lists_.begin(), lists_.end(), secondary) != lists_.end()) carried(primary);
  }

  bool queue_watch::executed(std::uint64_t list, std::uint64_t queue, std::uint64_t presenting) {
    if (!list || std::find(lists_.begin(), lists_.end(), list) == lists_.end()) return false;
    executed_on(queue, presenting);
    return true;
  }

  void queue_watch::executed_on(std::uint64_t queue, std::uint64_t presenting) {
    if (!queue) return;
    if (last_queue_ && last_queue_ != queue) mixed_ = true;
    last_queue_ = queue;
    if (presenting && queue != presenting) foreign_present_ = true;
  }

  bool queue_watch::watching() const {
    return std::any_of(lists_.begin(), lists_.end(), [](std::uint64_t value) { return value != 0; });
  }

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
      // S3 (shadow): the copy's stamp, whether the newest copy's list
      // recorded it, and the scope's queue watch.
      api::resource stamp{};
      bool stamped{};
      queue_watch watch;
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
      // S3: the presenting queue's native handle at the last Present
      // (frame_clock::presenting_queue), and a copy of the queue watch's
      // lists that the execute and reset events check without the lock, so
      // lists that carried no stamped copy never take it.
      std::uint64_t presenting_queue{};
      std::array<std::atomic<std::uint64_t>, queue_watch::capacity> watched{};
    };
    // Deliberately leaked: clear events can arrive during process exit.
    state_t &state() {
      static state_t *value = new state_t;
      return *value;
    }

    // Requires the state lock: publishes the watch's lists to the lock-free copy.
    void publish_watch(state_t &s) {
      for (std::size_t i = 0; i != queue_watch::capacity; ++i)
        s.watched[i].store(s.live.watch.lists()[i], std::memory_order_relaxed);
    }

    bool watched(const state_t &s, const void *list) {
      const auto key = reinterpret_cast<std::uint64_t>(list);
      for (const auto &value : s.watched) if (key && value.load(std::memory_order_relaxed) == key) return true;
      return false;
    }

    // Requires the state lock. Retires the live copy and its stamp (a scope
    // change: the next copy starts a new stamp at 0 and a new queue watch).
    void retire_live(state_t &s, std::uint64_t now) {
      if (s.live.copy.handle) s.graveyard.push_back({s.device, s.live.copy, now});
      if (s.live.stamp.handle) s.graveyard.push_back({s.device, s.live.stamp, now});
      s.live = {};
      publish_watch(s);
    }

    // A 16-byte stamp: a default buffer at 0 in COMMON, or a CPU-readable one
    // for the dump census (readback heaps rest in COPY_DEST on D3D12).
    api::resource create_stamp(api::device *device, bool readback) {
      static std::uint32_t zeros[4]{};
      const api::resource_desc desc(16, readback ? api::memory_heap::readback : api::memory_heap::default_,
        readback ? api::resource_usage::copy_dest : api::resource_usage::copy_source | api::resource_usage::copy_dest);
      // D3D12 zeroes committed buffers; D3D11 gets its zeros explicitly.
      const bool d3d11 = device->get_api() == api::device_api::d3d11 && !readback;
      const api::subresource_data initial{zeros, 16, 16};
      api::resource stamp{};
      if (!device->create_resource(desc, d3d11 ? &initial : nullptr,
            readback ? api::resource_usage::copy_dest : api::resource_usage::general, &stamp)) return {};
      return stamp;
    }

    // Records the device's C_P and C_T into stamp right after a copy in the
    // same list. The clocks and a default stamp move from COMMON and back by
    // explicit barriers (the add-on's own buffers only), so other add-on
    // accesses in the same list stay legal; a readback stamp stays COPY_DEST.
    // False, recording nothing, before the device's clocks exist.
    bool record_stamp(api::command_list *commands, api::resource stamp, bool readback) {
      if (!stamp.handle) return false;
      const auto native = reinterpret_cast<void *>(commands->get_device()->get_native());
      const auto present = frame_clock::native_present_clock(native), token = frame_clock::native_token_clock(native);
      if (!present.resource || !token.resource) return false;
      const api::resource resources[3]{{reinterpret_cast<std::uint64_t>(present.resource)},
        {reinterpret_cast<std::uint64_t>(token.resource)}, stamp};
      const api::resource_usage common[3]{api::resource_usage::general, api::resource_usage::general, api::resource_usage::general};
      const api::resource_usage copy[3]{api::resource_usage::copy_source, api::resource_usage::copy_source, api::resource_usage::copy_dest};
      const std::uint32_t count = readback ? 2 : 3;
      commands->barrier(count, resources, common, copy);
      commands->copy_buffer_region(resources[0], present.offset, stamp, 0, 4);
      commands->copy_buffer_region(resources[1], token.offset, stamp, 4, 4);
      commands->barrier(count, resources, copy, common);
      return true;
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
      for (auto &c : s.candidates) {
        if (c.copy.handle) s.graveyard.push_back({s.device, c.copy, GetTickCount64()});
        if (c.stamp.handle) s.graveyard.push_back({s.device, c.stamp, GetTickCount64()});
      }
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
        // S3: a new copy is a new scope: its stamp restarts at 0.
        if (live.stamp.handle) s.graveyard.push_back({device, live.stamp, GetTickCount64()});
        live.stamp = {};
        live.watch.clear_scope();
        publish_watch(s);
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
      if (!live.stamp.handle) live.stamp = create_stamp(device, false);
      commands->barrier(live.copy, api::resource_usage::shader_resource, api::resource_usage::copy_dest);
      record_copy(commands, resource, live.copy);
      // S3 (shadow): the stamp in the same list, then the list joins the watch
      // (a D3D11 immediate context is the presenting queue itself).
      live.stamped = record_stamp(commands, live.stamp, false);
      const auto list = commands->get_native();
      if (list && list == s.presenting_queue) live.watch.executed_on(list, s.presenting_queue);
      else live.watch.carried(reinterpret_cast<std::uint64_t>(commands));
      publish_watch(s);
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
        // S3 (shadow): the census copy's own stamp, which the dump reads.
        c.stamp = create_stamp(device, true);
        c.stamped = record_stamp(commands, c.stamp, true);
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
    out.stamp = live.stamp; out.stamped = live.stamped;
    out.foreign_present = live.watch.foreign_present(); out.queue_mixed = live.watch.mixed();
    out.executed_queue = live.watch.last_queue();
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
      for (auto &c : s.candidates) if (s.device == device) {
        if (c.copy.handle) device->destroy_resource(c.copy);
        if (c.stamp.handle) device->destroy_resource(c.stamp);
      }
      if (s.device == device) {
        if (s.live.copy.handle) device->destroy_resource(s.live.copy);
        if (s.live.stamp.handle) device->destroy_resource(s.live.stamp);
        s.live = {};
        publish_watch(s);
        s.candidates.clear();
        s.back_buffers.clear();
        s.device = nullptr;
        s.presenting_queue = 0;
      }
    }

    // True for a D3D11 command list: the primary of FinishCommandList's event
    // (a deferred context or a D3D12 bundle's parent is not one).
    bool finished_command_list(api::command_list *primary) {
      if (primary->get_device()->get_api() != api::device_api::d3d11) return false;
      auto *native = reinterpret_cast<IUnknown *>(primary->get_native());
      ID3D11CommandList *list = nullptr;
      if (!native || FAILED(native->QueryInterface(IID_PPV_ARGS(&list)))) return false;
      list->Release();
      return true;
    }

    // S3 queue watch events. Lists that carried no stamped copy return before
    // any lock.
    void on_execute(api::command_queue *queue, api::command_list *commands) {
      auto &s = state();
      if (!queue || !watched(s, commands)) return;
      std::lock_guard<std::mutex> lock(s.mutex);
      s.live.watch.executed(reinterpret_cast<std::uint64_t>(commands), queue->get_native(), s.presenting_queue);
    }

    void on_execute_secondary(api::command_list *primary, api::command_list *secondary) {
      auto &s = state();
      if (!primary || !secondary || (!watched(s, secondary) && !watched(s, primary))) return;
      std::lock_guard<std::mutex> lock(s.mutex);
      const auto list = reinterpret_cast<std::uint64_t>(secondary);
      // D3D11: ExecuteCommandList on the immediate context, the presenting
      // queue itself, executes the list.
      if (primary->get_native() && primary->get_native() == s.presenting_queue)
        s.live.watch.executed(list, s.presenting_queue, s.presenting_queue);
      // D3D11 FinishCommandList (the primary is the new ID3D11CommandList):
      // the deferred context's commands move into it.
      else if (finished_command_list(primary))
        s.live.watch.move(reinterpret_cast<std::uint64_t>(primary), list);
      // A D3D12 bundle, or a command list executed on a deferred context: the
      // primary keeps its own commands and adds the secondary's.
      else s.live.watch.transfer(reinterpret_cast<std::uint64_t>(primary), list);
      publish_watch(s);
    }

    void on_reset(api::command_list *commands) {
      auto &s = state();
      if (!watched(s, commands)) return;
      std::lock_guard<std::mutex> lock(s.mutex);
      s.live.watch.reset(reinterpret_cast<std::uint64_t>(commands));
      publish_watch(s);
    }
  }

  void register_events() {
    reshade::register_event<reshade::addon_event::clear_render_target_view>(sunshine_addon_lifetime::guarded<on_clear>);
    reshade::register_event<reshade::addon_event::destroy_device>(sunshine_addon_lifetime::guarded<on_destroy_device>);
    reshade::register_event<reshade::addon_event::execute_command_list>(sunshine_addon_lifetime::guarded<on_execute>);
    reshade::register_event<reshade::addon_event::execute_secondary_command_list>(
      sunshine_addon_lifetime::guarded<on_execute_secondary>);
    reshade::register_event<reshade::addon_event::reset_command_list>(sunshine_addon_lifetime::guarded<on_reset>);
  }

  void unregister_events() {
    reshade::unregister_event<reshade::addon_event::clear_render_target_view>(sunshine_addon_lifetime::guarded<on_clear>);
    reshade::unregister_event<reshade::addon_event::destroy_device>(sunshine_addon_lifetime::guarded<on_destroy_device>);
    reshade::unregister_event<reshade::addon_event::execute_command_list>(sunshine_addon_lifetime::guarded<on_execute>);
    reshade::unregister_event<reshade::addon_event::execute_secondary_command_list>(
      sunshine_addon_lifetime::guarded<on_execute_secondary>);
    reshade::unregister_event<reshade::addon_event::reset_command_list>(sunshine_addon_lifetime::guarded<on_reset>);
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
    // S3: the queue this Present's frame_clock::advance ran on.
    const auto *presenting = frame_clock::presenting_queue(device);
    const auto presenting_queue = presenting ? presenting->get_native() : 0;
    std::lock_guard<std::mutex> lock(s.mutex);
    // Every retired copy's device is alive: on_destroy_device releases its own.
    reap(s);
    if (s.armed && s.device && s.device != device) return;
    if (s.live.copy.handle && (s.device != device || s.width != desc.BufferDesc.Width || s.height != desc.BufferDesc.Height)) {
      retire_live(s, now);
    } else if (s.device != device) retire_live(s, now);
    s.presenting_queue = presenting_queue;
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

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
// The live copy's S3 facts for the D3D12 runtime fixture; the stamp buffer is
// an ID3D12Resource * the fixture reads back itself.
extern "C" __declspec(dllexport) BOOL SunshineUILayerTestLive(sunshine_game3d::ui_layer::test_live_state *out) {
  using namespace sunshine_game3d::ui_layer;
  if (!out) return FALSE;
  *out = {};
  api::device *device = nullptr;
  {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto &live = s.live;
    device = s.device;
    out->stamp = live.stamp.handle; out->capture_id = live.capture_id;
    out->executed_queue = live.watch.last_queue(); out->presenting_queue = s.presenting_queue;
    out->presents_since_copy = live.tracker.presents_since_copy();
    out->stamped = live.stamped; out->foreign_present = live.watch.foreign_present();
    out->queue_mixed = live.watch.mixed(); out->watching = live.watch.watching();
  }
  out->present_label = device ? sunshine_game3d::frame_clock::label(device) : 0;
  return device != nullptr;
}
#endif
