// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_layer.h"
#include "addon_lifetime.h"
#include "game3d_ui_detection_contract.h"
#include "async_log.h"

#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
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

    // A clear without rectangles, or with one covering the whole target.
    bool whole_clear(const api::resource_desc &desc, std::uint32_t rect_count, const api::rect *rects) {
      if (!rect_count) return true;
      if (!rects) return false;
      return std::any_of(rects, rects + rect_count, [&](const api::rect &r) {
        return r.left <= 0 && r.top <= 0 && r.right >= std::int64_t(desc.texture.width) &&
          r.bottom >= std::int64_t(desc.texture.height);
      });
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

    void destroy(const retired &r) {
      if (!r.device) return;
      if (r.view.handle) r.device->destroy_resource_view(r.view);
      if (r.copy.handle) r.device->destroy_resource(r.copy);
    }
    // Destroys add-on objects outside the state lock, only through a device
    // that is alive meanwhile: the presenting, recording or destroying one.
    // A retired copy of another device is destroyed under the lock (reap), so
    // its on_destroy_device, which hands over the copies still listed, cannot
    // return before it.
    void destroy(const std::vector<retired> &values) {
      for (const auto &r : values) destroy(r);
    }

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
      std::uint64_t capture_id{}, tick{};
      // At most two renderers (the current one and a cached one of the other
      // colour transfer) read an entry at once.
      std::array<reader, 2> readers{};
      // The game lists carrying this entry's copy (0: none), and whether the
      // copy is recorded but none of them executed yet. A pending entry is
      // never offered (it does not hold its copy yet) nor written again.
      std::array<std::uint64_t, 2> carriers{};
      // The queue that executed the copy (merge_queue; 0 not yet).
      std::uint64_t queue{};
      bool pending{};
      // The entry's copy is being allocated outside the lock (capture_live).
      bool allocating{};
      bool free() const { return !pending && !allocating && readers[0].completed() && readers[1].completed(); }
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
      // The newest recorded copy's id and tick (latest() reports them as the
      // single live copy did).
      std::uint64_t capture_id{}, tick{};
      bool saturated_logged{};
      const ring_entry *latest() const { return newest >= 0 ? &ring[unsigned(newest)] : nullptr; }
    };

    // The live copy is recorded only while UI detection asked for the layer
    // this recently (latest() is called on every Present that wants it).
    constexpr std::uint64_t demand_window_ms = 1000;
    // Live copies are kept over a lapse in demand this short (a stall, a
    // Game 3D or UI-source toggle), so they are reused rather than allocated
    // again on the game's recording thread; after it they are retired.
    constexpr std::uint64_t live_release_ms = 10000;

    struct state_t {
      std::mutex mutex;
      // Read without the lock by every clear and Present (wanted()).
      std::atomic<bool> armed{false};
      std::atomic<std::uint64_t> demand_ms{0};
      // Live or retired copies exist: while the layer is not wanted,
      // observe_output retires the live ones after live_release_ms without
      // demand and destroys retired ones once due.
      std::atomic<bool> holding{false};
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
      // Advance whenever the live ring is retired, or the census armed, taken
      // or dropped: an allocation made outside the lock for an earlier scope
      // is destroyed instead of published.
      std::uint64_t live_scope{}, census_scope{};
      bool unordered_logged{};
      // The presenting queue's native handle at the last Present, and each
      // ring entry's carriers, which the execute and reset events check
      // without the lock, so lists that carried no live copy never take it.
      std::uint64_t presenting_queue{};
      std::array<std::atomic<std::uint64_t>, ring_capacity * 2> watched{};
    };
    // Deliberately leaked: clear events can arrive during process exit.
    state_t &state() {
      static state_t *value = new state_t;
      return *value;
    }

    // UI detection asked for the layer within demand_window_ms (a stamp
    // after now, from another thread meanwhile, counts as now).
    bool demanded(const state_t &s, std::uint64_t now) {
      const auto demand = s.demand_ms.load(std::memory_order_relaxed);
      return demand && (now <= demand || now - demand <= demand_window_ms);
    }
    // Some reader wants the layer: UI detection or an armed dump census.
    bool wanted(const state_t &s, std::uint64_t now) {
      return s.armed.load(std::memory_order_relaxed) || demanded(s, now);
    }

    // Requires the state lock: publishes the ring entries' carriers to the
    // lock-free copy.
    void publish_watch(state_t &s) {
      for (std::size_t i = 0; i != ring_capacity; ++i)
        for (std::size_t j = 0; j != 2; ++j)
          s.watched[i * 2 + j].store(s.live.ring[i].carriers[j], std::memory_order_relaxed);
    }

    // Requires the state lock. A list executed on queue: the entries it
    // carries hold their copies now, executed on that queue, and the newest of
    // them becomes the offered copy (the last executed copy, as a single live
    // texture would hold).
    void carriers_executed(state_t &s, std::uint64_t list, std::uint64_t queue) {
      auto &live = s.live;
      int promoted = -1;
      for (unsigned i = 0; i != live.entries; ++i) {
        auto &entry = live.ring[i];
        if (!entry.carries(list)) continue;
        entry.pending = false;
        entry.queue = merge_queue(entry.queue, queue);
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
      for (auto &r : entry.readers) r.release();
      entry = {};
    }

    // Requires the state lock. Retires the live copies (a scope change: the
    // next copy starts a new ring).
    void retire_live(state_t &s, std::uint64_t now) {
      for (unsigned i = 0; i != s.live.entries; ++i) retire_entry(s, s.device, s.live.ring[i], now);
      s.live = {};
      ++s.live_scope;
      publish_watch(s);
    }

    // Requires the state lock. Retired copies that are due: the presenting
    // device's move into due, for destruction after the lock (that device
    // lives through its Present); another device's are destroyed here, under
    // the lock its on_destroy_device takes. A copy retired after now (another
    // thread stamped it while the caller waited) is not due yet.
    void reap(state_t &s, std::uint64_t now, std::vector<retired> &due, api::device *presenting) {
      auto keep = s.graveyard.begin();
      for (auto &r : s.graveyard) {
        if (now < r.tick || now - r.tick < retire_delay_ms) *keep++ = r;
        else if (r.device == presenting) due.push_back(r);
        else destroy(r);
      }
      s.graveyard.erase(keep, s.graveyard.end());
    }

    // Requires the state lock. Retires census copies that no dump took.
    void drop_candidates(state_t &s) {
      for (auto &c : s.candidates)
        if (c.copy.handle) {
          s.graveyard.push_back({s.device, c.copy, GetTickCount64()});
          s.holding.store(true, std::memory_order_relaxed);
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

    // A ring entry reserved under the lock whose copy the caller allocates
    // outside it (index -1: none).
    struct live_allocation {
      int index = -1;
      std::uint64_t scope{};
      api::resource_desc desc;
      api::format format{};
    };

    // Requires the state lock. Records the target's previous-frame content
    // into an allocated ring entry.
    void record_live(state_t &s, api::command_list *commands, unsigned index, api::resource resource) {
      auto &live = s.live;
      auto &entry = live.ring[index];
      commands->barrier(entry.copy, api::resource_usage::shader_resource, api::resource_usage::copy_dest);
      record_copy(commands, resource, entry.copy);
      entry.capture_id = ++live.capture_id;
      entry.tick = live.tick = GetTickCount64();
      entry.queue = 0;
      entry.carriers = {};
      // A D3D11 immediate context is the presenting queue itself, so the copy
      // executed there; else the entry is pending until a list carrying it
      // executes, on the queue that runs it.
      const auto list = commands->get_native();
      if (list && list == s.presenting_queue) {
        entry.queue = list;
        entry.pending = false;
        live.newest = int(index);
      } else {
        entry.carriers[0] = reinterpret_cast<std::uint64_t>(commands);
        entry.pending = true;
      }
      publish_watch(s);
      // The Present count at the copy: it holds the previous Present's frame.
      live.tracker.copied();
      s.holding.store(true, std::memory_order_relaxed);
    }

    // Requires the state lock. Copies the active layer's previous-frame content
    // into a ring entry, before the game's clear. An entry without a copy is
    // reserved instead and returned for allocation outside the lock.
    live_allocation capture_live(state_t &s, api::command_list *commands, api::device *device, api::resource resource,
        const api::resource_desc &desc) {
      auto &live = s.live;
      const auto typed = api::format_to_default_typed(desc.texture.format, 0);
      const auto now = GetTickCount64();
      if (live.entries && (live.width != desc.texture.width || live.height != desc.texture.height || live.format != typed)) {
        // A new copy shape is a new scope.
        for (unsigned i = 0; i != live.entries; ++i) retire_entry(s, device, live.ring[i], now);
        live.entries = 0;
        live.newest = -1;
        ++s.live_scope;
        publish_watch(s);
      }
      live.width = desc.texture.width; live.height = desc.texture.height; live.format = typed;
      std::array<bool, ring_capacity> free{};
      std::array<std::uint64_t, ring_capacity> age{};
      for (unsigned i = 0; i != live.entries; ++i) {
        auto &entry = live.ring[i];
        // A copy no carrying list executed or reset this long is abandoned.
        if (entry.pending && now - entry.tick >= retire_delay_ms) {
          entry.pending = false;
          entry.carriers = {};
        }
        free[i] = entry.free();
        // An entry whose allocation failed is taken last: it is allocated
        // again only when no entry with a copy is free (the ring's growth).
        age[i] = entry.copy.handle ? entry.capture_id : UINT64_MAX;
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
        return {};
      }
      if (choice.allocate) ++live.entries;
      auto &entry = live.ring[unsigned(choice.index)];
      if (!entry.copy.handle) {
        // A new entry, or one whose allocation failed: reserved here, its copy
        // and view are created outside the lock (publish_live records it).
        entry = {};
        entry.allocating = true;
        return {choice.index, s.live_scope,
          api::resource_desc(desc.texture.width, desc.texture.height, 1, 1, typed, 1, api::memory_heap::default_,
            api::resource_usage::copy_dest | api::resource_usage::copy_source | api::resource_usage::shader_resource),
          typed};
      }
      record_live(s, commands, unsigned(choice.index), resource);
      return {};
    }

    // Requires the state lock. Publishes a reserved entry's new copy and view
    // and records the copy; false when the ring's scope moved on meanwhile
    // (the caller destroys the objects). A failed allocation leaves the entry
    // empty, to be allocated again when chosen.
    bool publish_live(state_t &s, api::command_list *commands, api::device *device, const live_allocation &plan,
        api::resource copy, api::resource_view view, api::resource resource) {
      auto &live = s.live;
      if (s.live_scope != plan.scope || device != s.device || unsigned(plan.index) >= live.entries) return false;
      auto &entry = live.ring[unsigned(plan.index)];
      if (!entry.allocating) return false;
      entry.allocating = false;
      if (!copy.handle) return true;
      entry.copy = copy;
      entry.view = view;
      record_live(s, commands, unsigned(plan.index), resource);
      return true;
    }

    // A census target whose copy the caller allocates outside the lock.
    struct census_allocation {
      bool wanted{};
      std::uint64_t scope{}, source{};
      api::resource_desc desc;
    };
    constexpr const char *census_allocating = "allocating";

    // Requires the state lock. The dump census: each armed qualifying target once.
    census_allocation census(state_t &s, api::resource resource, const api::resource_desc &desc) {
      const auto known = std::find_if(s.candidates.begin(), s.candidates.end(),
        [&](const candidate &c) { return c.source == resource.handle; });
      if (known != s.candidates.end()) {
        ++known->clears;
        return {};
      }
      if (s.candidates.size() >= max_candidates) return {};
      candidate c;
      c.source = resource.handle;
      c.width = desc.texture.width;
      c.height = desc.texture.height;
      const auto typed = api::format_to_default_typed(desc.texture.format, 0);
      c.format = static_cast<std::uint32_t>(typed);
      c.clears = 1;
      c.status = census_allocating;
      s.candidates.push_back(c);
      return {true, s.census_scope, resource.handle, api::resource_desc(c.width, c.height, 1, 1, typed, 1,
        api::memory_heap::default_, api::resource_usage::copy_dest | api::resource_usage::shader_resource)};
    }

    // Requires the state lock. Publishes a census copy allocated outside the
    // lock and records it; false when the census moved on (the caller
    // destroys the copy).
    bool publish_census(state_t &s, api::command_list *commands, const census_allocation &plan, api::resource copy,
        api::resource resource) {
      if (s.census_scope != plan.scope) return false;
      const auto c = std::find_if(s.candidates.begin(), s.candidates.end(), [&](const candidate &value) {
        return value.source == plan.source && value.status == census_allocating;
      });
      if (c == s.candidates.end()) return false;
      if (!copy.handle) {
        c->status = "copy_allocation_failed";
        return true;
      }
      c->copy = copy;
      record_copy(commands, resource, copy);
      c->status = "captured_before_clear";
      return true;
    }

    bool on_clear(api::command_list *commands, api::resource_view view, const float color[4], std::uint32_t rect_count,
        const api::rect *rects) {
      auto &s = state();
      // Cheap rejection before any lookup or lock: only exact transparent
      // black qualifies, and only while the layer is wanted.
      if (!transparent_black(color) || !wanted(s, GetTickCount64())) return false;
      try {
        auto *device = commands->get_device();
        const auto resource = device->get_resource_from_view(view);
        if (!resource.handle) return false;
        const auto desc = device->get_resource_desc(resource);
        live_allocation live;
        census_allocation dump;
        {
          std::lock_guard<std::mutex> lock(s.mutex);
          if (device != s.device ||
              std::find(s.back_buffers.begin(), s.back_buffers.end(), resource.handle) != s.back_buffers.end()) return false;
          if (!qualifies(desc, color, s.width, s.height, rect_count, rects)) return false;
          // Clearing requires the render-target state on every API, so the
          // previous frame's layer can be copied here before the clear erases it.
          const auto now = GetTickCount64();
          if (s.live.tracker.clear(resource.handle, now) && demanded(s, now))
            live = capture_live(s, commands, device, resource, desc);
          if (s.armed.load(std::memory_order_relaxed)) dump = census(s, resource, desc);
        }
        if (live.index < 0 && !dump.wanted) return false;
        // New copies are allocated outside the lock, so a slow allocation
        // never stalls the Present thread or another recording thread.
        api::resource live_copy{}, dump_copy{};
        api::resource_view live_view{};
        if (live.index >= 0) {
          if (!device->create_resource(live.desc, nullptr, api::resource_usage::shader_resource, &live_copy)) live_copy = {};
          else if (!device->create_resource_view(live_copy, api::resource_usage::shader_resource,
                     api::resource_view_desc(live.format), &live_view))
            live_view = {};
        }
        if (dump.wanted && !device->create_resource(dump.desc, nullptr, api::resource_usage::copy_dest, &dump_copy))
          dump_copy = {};
        bool live_kept = live.index < 0, dump_kept = !dump.wanted;
        {
          std::lock_guard<std::mutex> lock(s.mutex);
          if (live.index >= 0) live_kept = publish_live(s, commands, device, live, live_copy, live_view, resource);
          if (dump.wanted) dump_kept = publish_census(s, commands, dump, dump_copy, resource);
        }
        // Objects of a scope that moved on meanwhile were never used.
        std::vector<retired> lost;
        if (!live_kept) lost.push_back({device, live_copy, 0, live_view});
        if (!dump_kept) lost.push_back({device, dump_copy});
        destroy(lost);
      } catch (...) {
        // A failure never propagates into the game's command recording.
      }
      return false; // Never skip the game's clear.
    }

    // Requires no lock. The layer is not wanted: live copies are retired once
    // it was not asked for in live_release_ms, and retired copies destroyed
    // once due (presenting: the presenting device).
    void idle(state_t &s, std::uint64_t now, api::device *presenting) {
      std::vector<retired> due;
      {
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto demand = s.demand_ms.load(std::memory_order_relaxed);
        if (s.live.entries && !s.armed.load(std::memory_order_relaxed) &&
            (!demand || (now > demand && now - demand > live_release_ms)))
          retire_live(s, now);
        reap(s, now, due, presenting);
        s.holding.store(s.live.entries || !s.graveyard.empty() || !s.candidates.empty(), std::memory_order_relaxed);
      }
      destroy(due);
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

  bool qualifies(const api::resource_desc &desc, const float color[4], std::uint32_t width, std::uint32_t height,
      std::uint32_t rect_count, const api::rect *rects) {
    return width && height && desc.type == api::resource_type::texture_2d && desc.texture.width == width &&
      desc.texture.height == height && desc.texture.samples == 1 && desc.texture.levels == 1 &&
      desc.texture.depth_or_layers == 1 && alpha_format(desc.texture.format) && transparent_black(color) &&
      whole_clear(desc, rect_count, rects);
  }

  bool latest(api::device *device, std::uint64_t now_ms, live_capture &out) {
    out = {};
    auto &s = state();
    s.demand_ms.store(now_ms, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto &live = s.live;
    const auto *entry = live.latest();
    // The executed copy is offered while the newest recorded one is recent,
    // as the single live texture was (its tick is the newest recorded copy's;
    // capture_id names the offered entry for bound()).
    if (!device || device != s.device || !entry || !entry->copy.handle || !entry->capture_id || !live.tracker.active() ||
        now_ms < live.tick || now_ms - live.tick > max_clear_gap_ms) return false;
    out.copy = entry->copy; out.view = entry->view; out.capture_id = entry->capture_id; out.tick = live.tick;
    out.format = static_cast<std::uint32_t>(live.format); out.presents_since_copy = live.tracker.presents_since_copy();
    out.executed_queue = entry->queue;
    // Only a copy that ran on the presenting queue alone is bound directly:
    // queue order puts it before this Present's reads.
    out.in_order = presented_in_order(entry->queue, s.presenting_queue);
    out.direct = entry->view.handle && out.in_order;
    if (!out.in_order && !s.unordered_logged) {
      s.unordered_logged = true;
      char message[256]{};
      std::snprintf(message, sizeof(message),
        "Sunshine UI layer: the live layer copy executed on queue 0x%llx, not on the presenting queue 0x%llx; it is "
        "copied into the renderer, not bound directly (logged once)",
        static_cast<unsigned long long>(entry->queue), static_cast<unsigned long long>(s.presenting_queue));
      sunshine_log::message(reshade::log::level::info, message);
    }
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
      std::vector<retired> doomed;
      {
        std::lock_guard<std::mutex> lock(s.mutex);
        auto keep = s.graveyard.begin();
        for (auto &r : s.graveyard) {
          if (r.device == device) doomed.push_back(r);
          else *keep++ = r;
        }
        s.graveyard.erase(keep, s.graveyard.end());
        if (s.device == device) {
          for (auto &c : s.candidates) if (c.copy.handle) doomed.push_back({device, c.copy});
          for (unsigned i = 0; i != s.live.entries; ++i) {
            auto &entry = s.live.ring[i];
            doomed.push_back({device, entry.copy, 0, entry.view});
            for (auto &r : entry.readers) r.release();
          }
          s.live = {};
          ++s.live_scope;
          ++s.census_scope;
          publish_watch(s);
          s.candidates.clear();
          s.back_buffers.clear();
          s.device = nullptr;
          s.presenting_queue = 0;
        }
      }
      destroy(doomed);
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

    // Carrier events. Lists that carried no live copy return before any lock.
    void on_execute(api::command_queue *queue, api::command_list *commands) {
      auto &s = state();
      if (!queue || !watched(s, commands)) return;
      std::lock_guard<std::mutex> lock(s.mutex);
      carriers_executed(s, reinterpret_cast<std::uint64_t>(commands), queue->get_native());
    }

    void on_execute_secondary(api::command_list *primary, api::command_list *secondary) {
      auto &s = state();
      if (!primary || !secondary || (!watched(s, secondary) && !watched(s, primary))) return;
      std::lock_guard<std::mutex> lock(s.mutex);
      const auto list = reinterpret_cast<std::uint64_t>(secondary);
      // D3D11: ExecuteCommandList on the immediate context, the presenting
      // queue itself, executes the list.
      if (primary->get_native() && primary->get_native() == s.presenting_queue)
        carriers_executed(s, list, s.presenting_queue);
      // D3D11 FinishCommandList (the primary is the new ID3D11CommandList):
      // the deferred context's commands move into it.
      else if (finished_command_list(primary))
        carriers_follow(s, list, reinterpret_cast<std::uint64_t>(primary), true);
      // A D3D12 bundle, or a command list executed on a deferred context: the
      // primary keeps its own commands and adds the secondary's.
      else
        carriers_follow(s, list, reinterpret_cast<std::uint64_t>(primary), false);
      publish_watch(s);
    }

    void on_reset(api::command_list *commands) {
      auto &s = state();
      if (!watched(s, commands)) return;
      std::lock_guard<std::mutex> lock(s.mutex);
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
    const auto now = GetTickCount64();
    auto *device = swapchain->get_device();
    if (!wanted(s, now)) {
      if (s.holding.load(std::memory_order_relaxed)) idle(s, now, device);
      return;
    }
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
    std::vector<retired> due;
    {
      std::lock_guard<std::mutex> lock(s.mutex);
      // Only this device's due copies are destroyed after the lock; another
      // device's are destroyed under it (reap).
      reap(s, now, due, device);
      if (!s.armed.load(std::memory_order_relaxed) || !s.device || s.device == device) {
        if (s.live.entries && (s.device != device || s.width != desc.BufferDesc.Width || s.height != desc.BufferDesc.Height)) {
          retire_live(s, now);
        } else if (s.device != device) retire_live(s, now);
        // The queue this Present runs on, which a live copy's executing queue
        // must equal to be bound directly.
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
    }
    destroy(due);
  }

  void arm() {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    drop_candidates(s);
    ++s.census_scope;
    s.armed.store(true, std::memory_order_relaxed);
  }

  std::vector<candidate> take(api::device *&device) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.armed.store(false, std::memory_order_relaxed);
    ++s.census_scope;
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
    s.holding.store(true, std::memory_order_relaxed);
  }

  void cancel() {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.armed.store(false, std::memory_order_relaxed);
    ++s.census_scope;
    drop_candidates(s);
  }
}

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
// The live copy's queue facts for the D3D12 runtime fixture.
extern "C" __declspec(dllexport) BOOL SunshineUILayerTestLive(sunshine_game3d::ui_layer::test_live_state *out) {
  using namespace sunshine_game3d::ui_layer;
  if (!out) return FALSE;
  *out = {};
  auto &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const auto &live = s.live;
  // The newest recorded copy's id, and the last executed copy's queue.
  out->capture_id = live.capture_id;
  out->presenting_queue = s.presenting_queue;
  out->presents_since_copy = live.tracker.presents_since_copy();
  if (const auto *entry = live.latest()) {
    out->executed_queue = entry->queue;
    out->in_order = entry->copy.handle && presented_in_order(entry->queue, s.presenting_queue);
  }
  return s.device != nullptr;
}
#endif
