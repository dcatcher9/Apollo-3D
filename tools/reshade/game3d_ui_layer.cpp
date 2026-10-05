// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_layer.h"
#include "addon_lifetime.h"
#include "game3d_diagnostics.h"
#include "game3d_frame_clock.h"
#include "game3d_ui_detection_contract.h"
#include "async_log.h"

#include <d3d11_4.h>
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
      api::resource_view view{};
    };

    // A reader of a ring entry: a renderer's completion fence (a native
    // reference, so a destroyed renderer cannot leave it dangling) and the
    // value its reading submission signals.
    struct reader {
      ID3D12Fence *fence12{};
      ID3D11Fence *fence11{};
      std::uint64_t value{};
      bool completed() const {
        const auto done = fence12 ? fence12->GetCompletedValue() : fence11 ? fence11->GetCompletedValue() : UINT64_MAX;
        return done >= value; // UINT64_MAX: removed device, nothing executes.
      }
      void release() {
        if (fence12) fence12->Release();
        if (fence11) fence11->Release();
        *this = {};
      }
    };

    // One live copy (game3d_ui_layer.h, ring_capacity).
    struct ring_entry {
      api::resource copy{};
      api::resource_view view{};
      // S3 (shadow, Diagnostics only): the copy's stamp and whether its list
      // recorded it.
      api::resource stamp{};
      bool stamped{};
      std::uint64_t capture_id{}, tick{};
      // At most two renderers (the current one and a cached one of the other
      // colour transfer) read an entry at once.
      std::array<reader, 2> readers{};
      // The game lists carrying this entry's copy (0: none), and whether the
      // copy is recorded but none of them executed yet. A pending entry is
      // never offered (it does not hold its copy yet) nor written again.
      std::array<std::uint64_t, 2> carriers{};
      bool pending{};
      bool free() const { return !pending && readers[0].completed() && readers[1].completed(); }
      bool carries(std::uint64_t list) const { return list && (carriers[0] == list || carriers[1] == list); }
    };

    struct live_state {
      layer_tracker tracker;
      std::array<ring_entry, ring_capacity> ring{};
      unsigned entries{};
      // The entry whose copy executed last: the content a single live texture
      // would hold now (-1: none). A recorded copy is promoted only when a
      // list carrying it executes.
      int newest = -1;
      std::uint32_t width{}, height{};
      api::format format{};
      // The newest recorded copy's id, tick and stamp recording (latest()
      // reports them as the single live copy did).
      std::uint64_t capture_id{}, tick{};
      bool stamped{};
      // The scope's queue watch: which queue executed the lists carrying the
      // live copies (direct binding; S3 reads it too).
      queue_watch watch;
      bool saturated_logged{};
      const ring_entry *latest() const { return newest >= 0 ? &ring[unsigned(newest)] : nullptr; }
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
      // The back buffers' cache key (observe_output): the native swapchain,
      // its size, buffer count and first buffer; ReShade's swapchain
      // creation and resize events clear it.
      std::uint64_t buffers_swapchain{}, buffers_first{};
      std::uint32_t buffers_count{}, buffers_width{}, buffers_height{};
      std::atomic<bool> buffers_valid{false};
      std::vector<candidate> candidates;
      std::vector<retired> graveyard;
      live_state live;
      // S3: the presenting queue's native handle at the last Present
      // (frame_clock::presenting_queue), and a copy of the queue watch's
      // lists that the execute and reset events check without the lock, so
      // lists that carried no stamped copy never take it.
      std::uint64_t presenting_queue{};
      // The queue watch's lists, then each ring entry's carriers.
      std::array<std::atomic<std::uint64_t>, queue_watch::capacity + ring_capacity * 2> watched{};
    };
    // Deliberately leaked: clear events can arrive during process exit.
    state_t &state() {
      static state_t *value = new state_t;
      return *value;
    }

    // Requires the state lock: publishes the watch's lists and the ring
    // entries' carriers to the lock-free copy.
    void publish_watch(state_t &s) {
      for (std::size_t i = 0; i != queue_watch::capacity; ++i)
        s.watched[i].store(s.live.watch.lists()[i], std::memory_order_relaxed);
      for (std::size_t i = 0; i != ring_capacity; ++i)
        for (std::size_t j = 0; j != 2; ++j)
          s.watched[queue_watch::capacity + i * 2 + j].store(s.live.ring[i].carriers[j], std::memory_order_relaxed);
    }

    // Requires the state lock. A list executed: the entries it carries hold
    // their copies now, and the newest of them becomes the offered copy (the
    // last executed copy, as a single live texture would hold).
    void carriers_executed(state_t &s, std::uint64_t list) {
      auto &live = s.live;
      int promoted = -1;
      for (unsigned i = 0; i != live.entries; ++i) {
        auto &entry = live.ring[i];
        if (!entry.carries(list)) continue;
        entry.pending = false;
        if (promoted < 0 || entry.capture_id > live.ring[unsigned(promoted)].capture_id) promoted = int(i);
      }
      if (promoted >= 0) live.newest = promoted;
    }

    // Requires the state lock. A list was reset: it carries nothing; an
    // entry no executed list wrote drops back to free (never offered).
    void carriers_reset(state_t &s, std::uint64_t list) {
      for (unsigned i = 0; i != s.live.entries; ++i) {
        auto &entry = s.live.ring[i];
        if (!entry.carries(list)) continue;
        for (auto &c : entry.carriers) if (c == list) c = 0;
        if (!entry.carriers[0] && !entry.carriers[1]) entry.pending = false;
      }
    }

    // Requires the state lock. D3D11 FinishCommandList moves a deferred
    // context's commands into a new list (replace), or a primary list also
    // runs a secondary's (add).
    void carriers_follow(state_t &s, std::uint64_t from, std::uint64_t to, bool replace) {
      if (!from || !to) return;
      for (unsigned i = 0; i != s.live.entries; ++i) {
        auto &entry = s.live.ring[i];
        if (!entry.carries(from) || entry.carries(to)) continue;
        if (replace) {
          for (auto &c : entry.carriers) if (c == from) c = to;
          continue;
        }
        // Two carriers at most: a full entry keeps from and replaces the other.
        auto &slot = !entry.carriers[0] ? entry.carriers[0] : !entry.carriers[1] ? entry.carriers[1] :
          entry.carriers[0] == from ? entry.carriers[1] : entry.carriers[0];
        slot = to;
      }
    }

    bool watched(const state_t &s, const void *list) {
      const auto key = reinterpret_cast<std::uint64_t>(list);
      for (const auto &value : s.watched) if (key && value.load(std::memory_order_relaxed) == key) return true;
      return false;
    }

    // Requires the state lock. Retires one ring entry's objects.
    void retire_entry(state_t &s, api::device *device, ring_entry &entry, std::uint64_t now) {
      if (entry.copy.handle) s.graveyard.push_back({device, entry.copy, now, entry.view});
      if (entry.stamp.handle) s.graveyard.push_back({device, entry.stamp, now});
      for (auto &r : entry.readers) r.release();
      entry = {};
    }

    // Requires the state lock. Retires the live copies and their stamps (a
    // scope change: the next copy starts a new stamp at 0 and a new queue watch).
    void retire_live(state_t &s, std::uint64_t now) {
      for (unsigned i = 0; i != s.live.entries; ++i) retire_entry(s, s.device, s.live.ring[i], now);
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
        if (now - r.tick >= retire_delay_ms) {
          if (r.device && r.view.handle) r.device->destroy_resource_view(r.view);
          if (r.device) r.device->destroy_resource(r.copy);
        } else *keep++ = r;
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
    // into a ring entry, before the game's clear.
    void capture_live(state_t &s, api::command_list *commands, api::device *device, api::resource resource,
        const api::resource_desc &desc) {
      auto &live = s.live;
      const auto typed = api::format_to_default_typed(desc.texture.format, 0);
      if (live.entries && (live.width != desc.texture.width || live.height != desc.texture.height || live.format != typed)) {
        // S3: a new copy shape is a new scope: its stamps restart at 0.
        const auto now = GetTickCount64();
        for (unsigned i = 0; i != live.entries; ++i) retire_entry(s, device, live.ring[i], now);
        live.entries = 0;
        live.newest = -1;
        live.watch.clear_scope();
        publish_watch(s);
      }
      live.width = desc.texture.width; live.height = desc.texture.height; live.format = typed;
      std::array<bool, ring_capacity> free{};
      std::array<std::uint64_t, ring_capacity> age{};
      const auto now = GetTickCount64();
      for (unsigned i = 0; i != live.entries; ++i) {
        auto &entry = live.ring[i];
        // A copy no carrying list executed or reset this long is abandoned.
        if (entry.pending && now - entry.tick >= retire_delay_ms) {
          entry.pending = false;
          entry.carriers = {};
        }
        free[i] = entry.free();
        age[i] = entry.capture_id;
      }
      const auto choice = choose_ring_entry(live.entries, free, age, live.newest);
      if (choice.index < 0) {
        // Every entry is still read by an unfinished renderer submission:
        // this frame's copy is skipped and the newest copy stays offered.
        if (!live.saturated_logged) {
          live.saturated_logged = true;
          sunshine_log::message(reshade::log::level::warning,
            "Sunshine UI layer: every live copy is still being read; skipping a layer copy (logged once)");
        }
        return;
      }
      auto &entry = live.ring[unsigned(choice.index)];
      if (choice.allocate) {
        const api::resource_desc copy_desc(desc.texture.width, desc.texture.height, 1, 1, typed, 1, api::memory_heap::default_,
          api::resource_usage::copy_dest | api::resource_usage::copy_source | api::resource_usage::shader_resource);
        if (!device->create_resource(copy_desc, nullptr, api::resource_usage::shader_resource, &entry.copy)) {
          entry = {};
          return;
        }
        if (!device->create_resource_view(entry.copy, api::resource_usage::shader_resource, api::resource_view_desc(typed),
              &entry.view))
          entry.view = {};
        ++live.entries;
      }
      // S3 (shadow): the stamp exists and is written only with Diagnostics on.
      const bool shadow = diagnostics::enabled();
      if (shadow && !entry.stamp.handle) entry.stamp = create_stamp(device, false);
      commands->barrier(entry.copy, api::resource_usage::shader_resource, api::resource_usage::copy_dest);
      record_copy(commands, resource, entry.copy);
      entry.stamped = shadow && record_stamp(commands, entry.stamp, false);
      entry.capture_id = ++live.capture_id;
      entry.tick = live.tick = now;
      live.stamped = entry.stamped;
      // The list joins the queue watch (a D3D11 immediate context is the
      // presenting queue itself, so the copy is executed; else the entry is
      // pending until a list carrying it executes).
      const auto list = commands->get_native();
      entry.carriers = {};
      if (list && list == s.presenting_queue) {
        live.watch.executed_on(list, s.presenting_queue);
        entry.pending = false;
        live.newest = choice.index;
      } else {
        live.watch.carried(reinterpret_cast<std::uint64_t>(commands));
        entry.carriers[0] = reinterpret_cast<std::uint64_t>(commands);
        entry.pending = true;
      }
      publish_watch(s);
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
        // S3 (shadow, Diagnostics only): the census copy's own stamp, which
        // the dump reads.
        if (diagnostics::enabled()) {
          c.stamp = create_stamp(device, true);
          c.stamped = record_stamp(commands, c.stamp, true);
        }
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
    const auto *entry = live.latest();
    // The executed copy is offered while the newest recorded one is recent,
    // as the single live texture was (its tick and stamping are the newest
    // recorded copy's; capture_id names the offered entry for bound()).
    if (!device || device != s.device || !entry || !entry->copy.handle || !entry->capture_id || !live.tracker.active() ||
        now_ms < live.tick || now_ms - live.tick > max_clear_gap_ms) return false;
    out.copy = entry->copy; out.view = entry->view; out.capture_id = entry->capture_id; out.tick = live.tick;
    out.format = static_cast<std::uint32_t>(live.format); out.presents_since_copy = live.tracker.presents_since_copy();
    out.stamp = entry->stamp; out.stamped = live.stamped;
    out.foreign_present = live.watch.foreign_present(); out.queue_mixed = live.watch.mixed();
    out.executed_queue = live.watch.last_queue();
    out.direct = entry->view.handle && s.presenting_queue && !out.foreign_present && !out.queue_mixed &&
      out.executed_queue == s.presenting_queue;
    return true;
  }

  void bound(api::device *device, std::uint64_t capture_id, api::fence fence, std::uint64_t value) {
    if (!device || !capture_id || !fence.handle) return;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (device != s.device) return;
    const bool d3d12 = device->get_api() == api::device_api::d3d12;
    for (unsigned i = 0; i != s.live.entries; ++i) {
      auto &entry = s.live.ring[i];
      if (entry.capture_id != capture_id) continue;
      const auto same = [&](const reader &r) {
        return d3d12 ? r.fence12 == reinterpret_cast<ID3D12Fence *>(fence.handle) :
                       r.fence11 == reinterpret_cast<ID3D11Fence *>(fence.handle);
      };
      reader *slot = nullptr;
      for (auto &r : entry.readers) if (same(r)) slot = &r;
      if (slot) {
        slot->value = std::max(slot->value, value);
        return;
      }
      // A new renderer: replace a reader that completed, else the first.
      slot = entry.readers[0].completed() ? &entry.readers[0] : entry.readers[1].completed() ? &entry.readers[1] :
        &entry.readers[0];
      slot->release();
      if (d3d12) (slot->fence12 = reinterpret_cast<ID3D12Fence *>(fence.handle))->AddRef();
      else (slot->fence11 = reinterpret_cast<ID3D11Fence *>(fence.handle))->AddRef();
      slot->value = value;
      return;
    }
  }

  namespace {
    // Copies are destroyed through their own device while it still exists.
    void on_destroy_device(api::device *device) {
      auto &s = state();
      std::lock_guard<std::mutex> lock(s.mutex);
      auto keep = s.graveyard.begin();
      for (auto &r : s.graveyard) {
        if (r.device == device) {
          if (r.view.handle) device->destroy_resource_view(r.view);
          device->destroy_resource(r.copy);
        } else *keep++ = r;
      }
      s.graveyard.erase(keep, s.graveyard.end());
      for (auto &c : s.candidates) if (s.device == device) {
        if (c.copy.handle) device->destroy_resource(c.copy);
        if (c.stamp.handle) device->destroy_resource(c.stamp);
      }
      if (s.device == device) {
        for (unsigned i = 0; i != s.live.entries; ++i) {
          auto &entry = s.live.ring[i];
          if (entry.view.handle) device->destroy_resource_view(entry.view);
          if (entry.copy.handle) device->destroy_resource(entry.copy);
          if (entry.stamp.handle) device->destroy_resource(entry.stamp);
          for (auto &r : entry.readers) r.release();
        }
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
      const auto list = reinterpret_cast<std::uint64_t>(commands);
      s.live.watch.executed(list, queue->get_native(), s.presenting_queue);
      carriers_executed(s, list);
    }

    void on_execute_secondary(api::command_list *primary, api::command_list *secondary) {
      auto &s = state();
      if (!primary || !secondary || (!watched(s, secondary) && !watched(s, primary))) return;
      std::lock_guard<std::mutex> lock(s.mutex);
      const auto list = reinterpret_cast<std::uint64_t>(secondary);
      // D3D11: ExecuteCommandList on the immediate context, the presenting
      // queue itself, executes the list.
      if (primary->get_native() && primary->get_native() == s.presenting_queue) {
        s.live.watch.executed(list, s.presenting_queue, s.presenting_queue);
        carriers_executed(s, list);
      }
      // D3D11 FinishCommandList (the primary is the new ID3D11CommandList):
      // the deferred context's commands move into it.
      else if (finished_command_list(primary)) {
        s.live.watch.move(reinterpret_cast<std::uint64_t>(primary), list);
        carriers_follow(s, list, reinterpret_cast<std::uint64_t>(primary), true);
      }
      // A D3D12 bundle, or a command list executed on a deferred context: the
      // primary keeps its own commands and adds the secondary's.
      else {
        s.live.watch.transfer(reinterpret_cast<std::uint64_t>(primary), list);
        carriers_follow(s, list, reinterpret_cast<std::uint64_t>(primary), false);
      }
      publish_watch(s);
    }

    void on_reset(api::command_list *commands) {
      auto &s = state();
      if (!watched(s, commands)) return;
      std::lock_guard<std::mutex> lock(s.mutex);
      s.live.watch.reset(reinterpret_cast<std::uint64_t>(commands));
      carriers_reset(s, reinterpret_cast<std::uint64_t>(commands));
      publish_watch(s);
    }

    // A created, resized or destroyed swapchain invalidates the cached back
    // buffers (observe_output re-enumerates them).
    void on_swapchain(api::swapchain *, bool) {
      state().buffers_valid.store(false, std::memory_order_relaxed);
    }
  }

  void register_events() {
    reshade::register_event<reshade::addon_event::clear_render_target_view>(sunshine_addon_lifetime::guarded<on_clear>);
    reshade::register_event<reshade::addon_event::destroy_device>(sunshine_addon_lifetime::guarded<on_destroy_device>);
    reshade::register_event<reshade::addon_event::execute_command_list>(sunshine_addon_lifetime::guarded<on_execute>);
    reshade::register_event<reshade::addon_event::execute_secondary_command_list>(
      sunshine_addon_lifetime::guarded<on_execute_secondary>);
    reshade::register_event<reshade::addon_event::reset_command_list>(sunshine_addon_lifetime::guarded<on_reset>);
    reshade::register_event<reshade::addon_event::init_swapchain>(sunshine_addon_lifetime::guarded<on_swapchain>);
    reshade::register_event<reshade::addon_event::destroy_swapchain>(sunshine_addon_lifetime::guarded<on_swapchain>);
  }

  void unregister_events() {
    reshade::unregister_event<reshade::addon_event::clear_render_target_view>(sunshine_addon_lifetime::guarded<on_clear>);
    reshade::unregister_event<reshade::addon_event::destroy_device>(sunshine_addon_lifetime::guarded<on_destroy_device>);
    reshade::unregister_event<reshade::addon_event::execute_command_list>(sunshine_addon_lifetime::guarded<on_execute>);
    reshade::unregister_event<reshade::addon_event::execute_secondary_command_list>(
      sunshine_addon_lifetime::guarded<on_execute_secondary>);
    reshade::unregister_event<reshade::addon_event::reset_command_list>(sunshine_addon_lifetime::guarded<on_reset>);
    reshade::unregister_event<reshade::addon_event::init_swapchain>(sunshine_addon_lifetime::guarded<on_swapchain>);
    reshade::unregister_event<reshade::addon_event::destroy_swapchain>(sunshine_addon_lifetime::guarded<on_swapchain>);
    cancel();
  }

  void observe_output(api::swapchain *swapchain, api::command_queue *queue) {
    auto &s = state();
    auto *device = swapchain->get_device();
    // Native DXGI, as the exporter reads back buffers; identities only.
    // (ReShade's own back-buffer getters crashed under MinGW.) A resize may
    // recreate the buffers at the same size, so the first buffer's identity
    // and ReShade's swapchain events guard the cache.
    auto *native = reinterpret_cast<IDXGISwapChain *>(swapchain->get_native());
    DXGI_SWAP_CHAIN_DESC desc{};
    if (!native || FAILED(native->GetDesc(&desc))) return;
    const bool d3d12 = device->get_api() == api::device_api::d3d12;
    const auto buffer = [&](UINT index) {
      IUnknown *value = nullptr;
      if (FAILED(native->GetBuffer(index, d3d12 ? __uuidof(ID3D12Resource) : __uuidof(ID3D11Texture2D),
            reinterpret_cast<void **>(&value))))
        return std::uint64_t{0};
      value->Release();
      return reinterpret_cast<std::uint64_t>(value);
    };
    const auto first = buffer(0);
    if (!first) return;
    const auto key = reinterpret_cast<std::uint64_t>(native);
    bool cached;
    {
      std::lock_guard<std::mutex> lock(s.mutex);
      cached = s.buffers_valid.load(std::memory_order_relaxed) && s.buffers_swapchain == key && s.buffers_first == first &&
        s.buffers_count == desc.BufferCount && s.buffers_width == desc.BufferDesc.Width &&
        s.buffers_height == desc.BufferDesc.Height && s.device == device;
    }
    std::array<std::uint64_t, 16> buffers{};
    UINT count = 0;
    if (!cached) {
      buffers[0] = first;
      for (count = 1; count < desc.BufferCount && count < buffers.size(); ++count)
        if (!(buffers[count] = buffer(count))) break;
    }
    const auto now = GetTickCount64();
    std::lock_guard<std::mutex> lock(s.mutex);
    // Every retired copy's device is alive: on_destroy_device releases its own.
    reap(s);
    if (s.armed && s.device && s.device != device) return;
    if (s.live.entries && (s.device != device || s.width != desc.BufferDesc.Width || s.height != desc.BufferDesc.Height)) {
      retire_live(s, now);
    } else if (s.device != device) retire_live(s, now);
    // The queue this Present runs on, which the queue watch compares the
    // live copies' executing queue with.
    s.presenting_queue = queue ? queue->get_native() : 0;
    s.device = device;
    s.width = desc.BufferDesc.Width;
    s.height = desc.BufferDesc.Height;
    if (!cached) {
      s.back_buffers.assign(buffers.begin(), buffers.begin() + count);
      s.buffers_swapchain = key; s.buffers_first = first; s.buffers_count = desc.BufferCount;
      s.buffers_width = desc.BufferDesc.Width; s.buffers_height = desc.BufferDesc.Height;
      s.buffers_valid.store(true, std::memory_order_relaxed);
    }
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
    const auto *entry = live.latest();
    // The offered (last executed) copy's stamp; the newest recorded copy's id.
    out->stamp = entry ? entry->stamp.handle : 0; out->capture_id = live.capture_id;
    out->executed_queue = live.watch.last_queue(); out->presenting_queue = s.presenting_queue;
    out->presents_since_copy = live.tracker.presents_since_copy();
    out->stamped = entry && live.stamped; out->foreign_present = live.watch.foreign_present();
    out->queue_mixed = live.watch.mixed(); out->watching = live.watch.watching();
  }
  out->present_label = device ? sunshine_game3d::frame_clock::label(device) : 0;
  return device != nullptr;
}
#endif
