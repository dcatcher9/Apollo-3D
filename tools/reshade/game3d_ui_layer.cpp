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
#include <chrono>
#include <cstdio>
#include <cstring>
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
      std::uint64_t delay{}; // Milliseconds before it is destroyed (0: retire_delay_ms).
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
      // The live tracker's Present count when the copy was recorded
      // (layer_tracker::frame).
      std::uint64_t frame{};
      // At most two renderers (the current one and a cached one of the other
      // colour transfer) read an entry at once.
      std::array<reader, 2> readers{};
      // The copy's queue order (copy_order): the game lists carrying it, the
      // queue that executed it, the submission hook it is held for and the
      // queue fence value signalled after it. A pending entry is never offered
      // (its copy's write may not be submitted yet, or may still run) nor
      // written again; one whose owed fence signal never came (unobserved) is
      // offered unordered, logged as unobserved.
      copy_order order;
      // The entry's copy is being allocated outside the lock (capture_live).
      bool allocating{};
      bool free() const { return !order.pending && !allocating && readers[0].completed() && readers[1].completed(); }
    };

    // A Dump 3D census target: its row, its copy's queue order (copy_order,
    // judged by census_verdict when taken), the live tracker's Present count
    // when the copy was recorded, and the copy's API.
    struct census_entry {
      candidate c;
      copy_order order;
      std::uint64_t frame{};
      bool d3d12{};
    };

    struct live_state {
      layer_tracker tracker;
      std::array<ring_entry, ring_capacity> ring{};
      unsigned entries{};
      // The offered entry: the newest executed copy that is not held (-1:
      // none). A recorded copy is promoted only when a list carrying it
      // executes; on D3D12 while the submission hook is heard, only once that
      // hook ran after the native call, and one owing a fence signal only
      // once the CPU saw its fence reach it (settle).
      int newest = -1;
      // The entry latest() last offered, pinned until bound() registers the
      // reading Present (-1: none): between them it has no reader yet, and a
      // copy promoted meanwhile may no longer be the newest. Every Present it
      // was offered to calls bound(), one that renders nothing included (its
      // slot copy, recorded at acquisition, may still read the entry), and
      // its reader's value is the completion signal that Present claims,
      // which its finish_present sends whether or not it rendered
      // (renderer::claim_completion_value).
      int reading = -1;
      std::uint32_t width{}, height{};
      api::format format{};
      // The newest recorded copy's id (from state_t::next_capture_id) and
      // tick (latest()'s recency gate, recent_copy, with the offered copy's).
      std::uint64_t capture_id{}, tick{};
      bool saturated_logged{};
      // The time without a recorded copy that skipped copies fell in.
      skip_gap gap;
      const ring_entry *latest() const { return newest >= 0 ? &ring[unsigned(newest)] : nullptr; }
    };

    // The skip gap's clock: steady microseconds, finer than GetTickCount64's
    // ~16 ms ticks (an uncapped game clears the layer every 2 ms).
    std::uint64_t steady_us() {
      return std::uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    // The live copy is recorded only while UI detection asked for the layer
    // this recently (latest() is called on every Present that wants it).
    constexpr std::uint64_t demand_window_ms = 1000;
    // Live copies are kept over a lapse in demand this short (a stall, a
    // Game 3D or UI-source toggle), so they are reused rather than allocated
    // again on the game's recording thread; after it they are retired.
    constexpr std::uint64_t live_release_ms = 10000;

    // A fence of the add-on on a game queue that ran live copies away from the
    // presenting queue (game3d_ui_layer.h, queue order); one per device and
    // queue. Only that queue signals it, in increasing order, and only the
    // CPU reads it: no queue ever waits for it.
    struct queue_fence {
      api::device *device{};
      std::uint64_t queue{};  // Native queue, identity only (never referenced).
      ID3D12Fence *fence{};   // Owned reference; null with failed set: creation failed.
      std::uint64_t generation{};
      std::uint64_t signalled{}; // The last value signalled on the queue.
      bool failed{};
    };
    constexpr unsigned fence_capacity = 4;

    // One-time log lines.
    // log_census is the first of eight bits, one per census_status that omits
    // a census copy and per unordered reason (census_log_bit).
    enum log_bit : std::uint32_t {
      log_fenced = 1u << 0, log_unobserved = 1u << 1, log_mixed = 1u << 2, log_no_fence = 1u << 3,
      log_create_failed = 1u << 4, log_signal_failed = 1u << 5, log_fence_capacity = 1u << 6, log_census = 1u << 7,
    };

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
      // The armed census (max_candidates targets at most).
      std::vector<census_entry> census;
      std::vector<retired> graveyard;
      live_state live;
      // The last live copy id handed out. Strictly increasing for the life of
      // the process, across ring scopes: a renderer caches its slot copy by
      // this id, so an id restarting after a retire could name a copy seconds
      // old as already copied.
      std::uint64_t next_capture_id{};
      // Advance whenever the live ring is retired, or the census armed, taken
      // or dropped: an allocation made outside the lock for an earlier scope
      // is destroyed instead of published.
      std::uint64_t live_scope{}, census_scope{};
      std::uint32_t logged{}; // log_bit
      // The presenting queue's native handle at the last Present, and each
      // ring entry's and census copy's carriers (publish_watch), which the
      // execute and reset events check without the lock, so lists that
      // carried no layer copy never take it.
      std::uint64_t presenting_queue{};
      std::array<std::atomic<std::uint64_t>, (ring_capacity + max_candidates) * 2> watched{};
      // Cross-queue order: the queue fences, and each ring entry's and census
      // copy's owed native list, which the submission listener checks without
      // the lock.
      std::array<queue_fence, fence_capacity> fences{};
      std::uint64_t fence_generation{};
      std::array<std::atomic<std::uint64_t>, ring_capacity + max_candidates> owed{};
      // GetTickCount64 when the submission listener was last called (0:
      // never): within listener_window_ms, a D3D12 execution is held for it.
      std::atomic<std::uint64_t> listened_ms{0};
      // The periodic timing line's live-copy counters (take_stats), updated
      // under the lock and taken without it (the timing line holds the
      // exporter's lock).
      struct {
        std::atomic<std::uint64_t> copies{0}, skipped{0}, offers{0}, presents_since_sum{0}, skip_gap_max_us{0};
        std::atomic<std::uint32_t> presents_since_max{0};
      } counts;
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

    // Requires the state lock. Calls visit(order, index) for each live ring
    // entry's copy order (index: its ring index), then for each census copy's
    // (index -1): both follow the same transitions, and only a ring entry is
    // ever offered.
    template<class F> void each_order(state_t &s, F &&visit) {
      for (unsigned i = 0; i != s.live.entries; ++i) visit(s.live.ring[i].order, int(i));
      for (auto &e : s.census) visit(e.order, -1);
    }

    // Requires the state lock: publishes the ring entries' and census copies'
    // carriers and owed native lists to their lock-free copies.
    void publish_watch(state_t &s) {
      for (std::size_t i = 0; i != ring_capacity + max_candidates; ++i) {
        const auto census = i - ring_capacity;
        const copy_order *order = i < ring_capacity ? &s.live.ring[i].order :
          census < s.census.size() ? &s.census[census].order : nullptr;
        for (std::size_t j = 0; j != 2; ++j)
          s.watched[i * 2 + j].store(order ? order->carriers[j] : 0, std::memory_order_relaxed);
        s.owed[i].store(order ? order->owed : 0, std::memory_order_relaxed);
      }
    }

    // Requires the state lock. True the first time bit is logged.
    bool first(state_t &s, std::uint32_t bit) {
      if (s.logged & bit) return false;
      s.logged |= bit;
      return true;
    }

    // Requires the state lock. Offers entry index unless a newer copy is
    // offered already (a held copy can be signalled after a newer one ran).
    void promote(live_state &live, unsigned index) {
      if (live.newest < 0 || live.ring[index].capture_id > live.ring[unsigned(live.newest)].capture_id)
        live.newest = int(index);
    }

    // Requires the state lock. A list executed on queue: the entries it
    // carries hold their copies now, executed on that queue. This execution
    // writes each copy again, so a fence value signalled after an earlier one
    // no longer covers it. ReShade reports the execution before the native
    // call, so on D3D12, while the submission hook is heard (listening), a
    // copy run on one queue stays pending, neither offered nor rewritten,
    // until that hook runs after the native call (held_for_submission,
    // native_list, which observed_submission matches): a Present on another
    // thread meanwhile keeps reading the previous copy instead of one whose
    // write is not submitted yet. One on the presenting queue is offered then
    // (queue order); one on another queue owes a fence signal after this
    // submission and is offered once the CPU saw its fence reach it (settle).
    // Without a listener heard it is offered now (one owing a signal
    // unobserved). The newest of the others becomes the offered copy (the
    // last executed copy, as a single live texture would hold). A census copy
    // the list carries moves the same way (order_executed), never offered.
    void carriers_executed(state_t &s, std::uint64_t list, std::uint64_t native_list, std::uint64_t queue, bool d3d12,
        bool listening) {
      auto &live = s.live;
      int promoted = -1;
      each_order(s, [&](copy_order &order, int index) {
        if (!order.carries(list)) return;
        if (!order_executed(order, native_list, queue, d3d12, listening, s.presenting_queue) || index < 0) return;
        if (promoted < 0 || live.ring[unsigned(index)].capture_id > live.ring[unsigned(promoted)].capture_id)
          promoted = index;
      });
      if (promoted >= 0) live.newest = promoted;
    }

    // Requires the state lock. A list was reset: it carries nothing; an
    // entry no executed list wrote drops back to free (never offered). The
    // submission hook the list's execution owed can no longer come (it is
    // matched within that submission): an entry held for it executed without
    // it and is offered (unobserved: unordered unless it ran on the
    // presenting queue, whose order order_of still reports). An entry whose
    // fence signal is recorded stays held for its fence (a list may be reset
    // as soon as it was submitted, while its work still runs). A census copy
    // moves the same way (order_reset), never offered.
    void carriers_reset(state_t &s, std::uint64_t list, std::uint64_t native_list) {
      each_order(s, [&](copy_order &order, int index) {
        if (order_reset(order, list, native_list) && index >= 0) promote(s.live, unsigned(index));
      });
    }

    void release(queue_fence &f) {
      if (f.fence) f.fence->Release();
      f = {};
    }

    // Requires the state lock. The queue fence a copy names, while that
    // record still holds the fence it was signalled on (null: none).
    queue_fence *fence_of(state_t &s, const copy_order &order) {
      if (order.fence < 0 || unsigned(order.fence) >= fence_capacity) return nullptr;
      auto &f = s.fences[unsigned(order.fence)];
      return f.fence && f.generation == order.fence_generation ? &f : nullptr;
    }
    // Requires the state lock. How the presenting queue's reads are ordered
    // after an offered live copy, and the queue fence that orders it (null:
    // none). A fenced copy is offered only once its fence reached its value
    // (settle), and only its own queue signals that fence, in increasing
    // order, so it stays reached.
    read_order order_of(state_t &s, const ring_entry &entry, const queue_fence *&fence) {
      const bool same = presented_in_order(entry.order.queue, s.presenting_queue);
      fence = same ? nullptr : fence_of(s, entry.order);
      return order_for(same, fence ? entry.order.signalled : 0);
    }

    // Requires the state lock. Each copy held for its fence (awaiting) whose
    // fence the CPU now sees at its value (GetCompletedValue, never a wait;
    // fence_pending) is offered unless a newer copy is. One whose fence
    // record is gone can never be seen complete: it is dropped, never offered.
    // A census copy settles the same way (order_settled), never offered.
    void settle(state_t &s) {
      each_order(s, [&](copy_order &order, int index) {
        if (!order.awaiting()) return;
        const auto *f = fence_of(s, order);
        if (order_settled(order, f != nullptr, f ? f->fence->GetCompletedValue() : 0) && index >= 0)
          promote(s.live, unsigned(index));
      });
    }

    // Requires the state lock. A record no live or census copy names and
    // whose last signal ran.
    bool reusable(state_t &s, unsigned index) {
      const auto &f = s.fences[index];
      if (f.failed) return f.device != s.device;
      if (!f.fence) return false;
      bool named = false;
      each_order(s, [&](copy_order &order, int) {
        named |= order.fence == int(index) && order.fence_generation == f.generation;
      });
      return !named && f.fence->GetCompletedValue() >= f.signalled;
    }

    // Requires the state lock. The fence record of the current device's queue
    // (native), created on first use; -1 when none can be had (logged once).
    int fence_slot(state_t &s, std::uint64_t queue) {
      for (unsigned i = 0; i != fence_capacity; ++i) {
        const auto &f = s.fences[i];
        if ((f.fence || f.failed) && f.device == s.device && f.queue == queue) return f.fence ? int(i) : -1;
      }
      int slot = -1;
      for (unsigned i = 0; i != fence_capacity && slot < 0; ++i)
        if (!s.fences[i].fence && !s.fences[i].failed) slot = int(i);
      for (unsigned i = 0; i != fence_capacity && slot < 0; ++i)
        if (reusable(s, i)) slot = int(i);
      if (slot < 0) {
        if (first(s, log_fence_capacity))
          sunshine_log::message(reshade::log::level::warning,
            "Sunshine UI layer: every cross-queue fence is in use; a layer copy on another queue stays unordered (logged once)");
        return -1;
      }
      auto &f = s.fences[unsigned(slot)];
      release(f);
      f.device = s.device;
      f.queue = queue;
      f.generation = ++s.fence_generation;
      // The queue's own device: the copy's list ran there.
      ID3D12Device *device = nullptr;
      if (SUCCEEDED(reinterpret_cast<ID3D12CommandQueue *>(queue)->GetDevice(IID_PPV_ARGS(&device))) && device) {
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f.fence)))) f.fence = nullptr;
        device->Release();
      }
      if (f.fence) return slot;
      f.failed = true;
      if (first(s, log_create_failed)) {
        char message[200]{};
        std::snprintf(message, sizeof(message),
          "Sunshine UI layer: could not create a fence for queue 0x%llx; its layer copies stay unordered (logged once)",
          static_cast<unsigned long long>(queue));
        sunshine_log::message(reshade::log::level::warning, message);
      }
      return -1;
    }

    // Requires the state lock. A native submission on queue ran the lists
    // in commands, and the native call returned: each live copy held for one
    // of them that ran on the presenting queue is offered now, in queue order
    // (no fence). Each one that ran on another queue gets the queue fence's
    // next value, signalled now, after that submission, and is held until the
    // CPU sees the fence reach it (settle); one no value could be signalled
    // for is offered now, unordered. Census copies the submission ran are
    // proven the same way (order_submitted, order_signalled), sharing the one
    // signalled value, and never offered.
    void signal_owed(state_t &s, std::uint64_t queue, unsigned count,
        const sunshine_streamline::native_observer::command_identity *commands) {
      auto &live = s.live;
      // The copies owing this queue's fence signal and their ring index (-1:
      // a census copy).
      std::array<copy_order *, ring_capacity + max_candidates> due{};
      std::array<int, ring_capacity + max_candidates> due_index{};
      unsigned owing = 0;
      bool submitted = false;
      each_order(s, [&](copy_order &order, int index) {
        bool ran = false;
        for (unsigned c = 0; order.owed && c != count && !ran; ++c) ran = commands[c].native_command == order.owed;
        if (!ran) return;
        submitted = true;
        if (!presented_in_order(order.queue, s.presenting_queue)) {
          due[owing] = &order;
          due_index[owing++] = index;
          return;
        }
        if (order_submitted(order) && index >= 0) promote(live, unsigned(index));
      });
      if (!submitted) return;
      if (!owing) {
        publish_watch(s);
        return;
      }
      const int slot = fence_slot(s, queue);
      std::uint64_t value = 0, generation = 0;
      if (slot >= 0) {
        auto &f = s.fences[unsigned(slot)];
        generation = f.generation;
        if (SUCCEEDED(reinterpret_cast<ID3D12CommandQueue *>(queue)->Signal(f.fence, f.signalled + 1))) {
          value = ++f.signalled;
        } else if (first(s, log_signal_failed)) {
          char message[200]{};
          std::snprintf(message, sizeof(message),
            "Sunshine UI layer: could not signal the fence of queue 0x%llx after a layer copy; it stays unordered (logged once)",
            static_cast<unsigned long long>(queue));
          sunshine_log::message(reshade::log::level::warning, message);
        }
      }
      for (unsigned k = 0; k != owing; ++k)
        if (order_signalled(*due[k], slot, generation, value) && due_index[k] >= 0)
          promote(live, unsigned(due_index[k]));
      publish_watch(s);
    }

    // Requires the state lock. D3D11 FinishCommandList moves a deferred
    // context's commands into a new list (replace), or a primary list also
    // runs a secondary's (add).
    void carriers_follow(state_t &s, std::uint64_t from, std::uint64_t to, bool replace) {
      if (!from || !to) return;
      each_order(s, [&](copy_order &order, int) {
        if (!order.carries(from) || order.carries(to)) return;
        if (replace) {
          for (auto &c : order.carriers) if (c == from) c = to;
          return;
        }
        // Two carriers at most: a full entry keeps from and replaces the other.
        auto &slot = !order.carriers[0] ? order.carriers[0] : !order.carriers[1] ? order.carriers[1] :
          order.carriers[0] == from ? order.carriers[1] : order.carriers[0];
        slot = to;
      });
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
        if (now < r.tick || now - r.tick < (r.delay ? r.delay : retire_delay_ms)) *keep++ = r;
        else if (r.device == presenting) due.push_back(r);
        else destroy(r);
      }
      s.graveyard.erase(keep, s.graveyard.end());
    }

    // Requires the state lock. Retires a census entry's copy that no dump
    // reads. One a game list still carries, or whose write may still be
    // pending, is kept for live_release_ms rather than retire_delay_ms: its
    // list may run late (at most max_candidates such copies per census).
    void retire_census(state_t &s, const census_entry &e, std::uint64_t now) {
      if (!e.c.copy.handle) return;
      s.graveyard.push_back({s.device, e.c.copy, now, {}, e.order.pending || e.order.carried() ? live_release_ms : 0});
      s.holding.store(true, std::memory_order_relaxed);
    }

    // The log bit of an omitted census copy: one per status, and one per
    // reason of an unordered copy (eight from log_census).
    std::uint32_t census_log_bit(const census_result &verdict) {
      const auto index = static_cast<unsigned>(verdict.status) - 1u;
      if (verdict.status != census_status::unordered || !verdict.reason) return log_census << index;
      return log_census << (index + (std::strcmp(verdict.reason, "mixed_queue") == 0 ? 0u :
        std::strcmp(verdict.reason, "no_fence") == 0 ? 1u : 2u));
    }

    // Requires the state lock. Retires census copies that no dump took.
    void drop_candidates(state_t &s) {
      const auto now = GetTickCount64();
      for (const auto &e : s.census) retire_census(s, e, now);
      s.census.clear();
      publish_watch(s);
    }

    // Records the target's previous-frame content into copy, an add-on texture
    // kept in shader_resource state between uses (live and census copies are
    // created in it), leaving copy shader-readable and the target as the clear
    // needs it. A list executed again without a reset records the same
    // transitions again, from the state the previous execution left.
    void record_into(api::command_list *commands, api::resource target, api::resource copy) {
      commands->barrier(copy, api::resource_usage::shader_resource, api::resource_usage::copy_dest);
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

    // A skip gap (skip_gap) for the timing line's maximum.
    void note_skip_gap(state_t &s, std::uint64_t gap_us) {
      for (auto seen = s.counts.skip_gap_max_us.load(std::memory_order_relaxed); seen < gap_us &&
           !s.counts.skip_gap_max_us.compare_exchange_weak(seen, gap_us, std::memory_order_relaxed);) {}
    }

    // Requires the state lock. Records the target's previous-frame content
    // into an allocated ring entry.
    void record_live(state_t &s, api::command_list *commands, unsigned index, api::resource resource) {
      auto &live = s.live;
      auto &entry = live.ring[index];
      note_skip_gap(s, live.gap.record(steady_us()));
      record_into(commands, resource, entry.copy);
      entry.capture_id = live.capture_id = ++s.next_capture_id;
      entry.tick = live.tick = GetTickCount64();
      // The Present count at the copy: it holds the previous Present's frame.
      entry.frame = live.tracker.frame();
      // A D3D11 immediate context is the presenting queue itself, so the copy
      // executed there; else the entry is pending until a list carrying it
      // executes, on the queue that runs it.
      const auto list = reinterpret_cast<std::uint64_t>(commands);
      if (order_recorded(entry.order, list, commands->get_native(), s.presenting_queue)) live.newest = int(index);
      publish_watch(s);
      s.counts.copies.fetch_add(1, std::memory_order_relaxed);
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
        live.newest = live.reading = -1;
        live.gap = {};
        ++s.live_scope;
        publish_watch(s);
      }
      live.width = desc.texture.width; live.height = desc.texture.height; live.format = typed;
      // A held copy whose fence the CPU now sees reached is offered (and the
      // copy it replaces may be written again).
      settle(s);
      std::array<bool, ring_capacity> free{};
      std::array<std::uint64_t, ring_capacity> age{};
      bool abandoned = false;
      for (unsigned i = 0; i != live.entries; ++i) {
        auto &entry = live.ring[i];
        // A copy no carrying list executed or reset this long, or whose
        // submission hook never came, is abandoned (never offered). One whose
        // fence signal is recorded stays held until its fence reached it: its
        // queue may still write it.
        if (entry.order.pending && !entry.order.signalled && now - entry.tick >= retire_delay_ms) {
          entry.order.pending = false;
          entry.order.carriers = {};
          entry.order.owed = 0;
          abandoned = true;
        }
        free[i] = entry.free();
        // An entry whose allocation failed is taken last: it is allocated
        // again only when no entry with a copy is free (the ring's growth).
        age[i] = entry.copy.handle ? entry.capture_id : UINT64_MAX;
      }
      if (abandoned) publish_watch(s);
      const auto choice = choose_ring_entry(live.entries, free, age, live.newest, live.reading);
      if (choice.index < 0) {
        // Every entry is offered, held or still read by an unfinished
        // renderer submission: this frame's copy is skipped and the offered
        // copy stays offered, so Presents keep reading an older layer until a
        // copy is recorded again. The timing line counts the skips and the
        // longest time the layer went unrefreshed across them (skip_gap); the
        // readiness report judges that time against the stream's frame
        // interval, since a skip costs time, not a count (one skipped copy
        // leaves a gap of two clear intervals: about 3 ms at 590 clears a
        // second, 43 ms at 47).
        s.counts.skipped.fetch_add(1, std::memory_order_relaxed);
        note_skip_gap(s, live.gap.skip(steady_us()));
        if (!live.saturated_logged) {
          live.saturated_logged = true;
          sunshine_log::message(reshade::log::level::info,
            "Sunshine UI layer: every live copy is offered, held or still being read; skipping a layer copy, the offered "
            "copy stays offered (logged once; the timing line counts skipped copies and their longest refresh gap)");
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
      const auto known = std::find_if(s.census.begin(), s.census.end(),
        [&](const census_entry &e) { return e.c.source == resource.handle; });
      if (known != s.census.end()) {
        ++known->c.clears;
        return {};
      }
      if (s.census.size() >= max_candidates) return {};
      census_entry e;
      auto &c = e.c;
      c.source = resource.handle;
      c.width = desc.texture.width;
      c.height = desc.texture.height;
      const auto typed = api::format_to_default_typed(desc.texture.format, 0);
      c.format = static_cast<std::uint32_t>(typed);
      c.clears = 1;
      c.status = census_allocating;
      s.census.push_back(e);
      return {true, s.census_scope, resource.handle, api::resource_desc(c.width, c.height, 1, 1, typed, 1,
        api::memory_heap::default_,
        api::resource_usage::copy_dest | api::resource_usage::copy_source | api::resource_usage::shader_resource)};
    }

    // Requires the state lock. Publishes a census copy allocated outside the
    // lock (in shader_resource state, as live copies are) and records it into
    // the game's list, whose execution its queue order then follows (the live
    // copies' carriers, submission hook and queue fence); false when the
    // census moved on (the caller destroys the copy).
    bool publish_census(state_t &s, api::command_list *commands, const census_allocation &plan, api::resource copy,
        api::resource resource) {
      if (s.census_scope != plan.scope) return false;
      const auto e = std::find_if(s.census.begin(), s.census.end(), [&](const census_entry &value) {
        return value.c.source == plan.source && value.c.status == census_allocating;
      });
      if (e == s.census.end()) return false;
      if (!copy.handle) {
        e->c.status = "copy_allocation_failed";
        return true;
      }
      e->c.copy = copy;
      record_into(commands, resource, copy);
      // The Present count at the copy: it holds the previous Present's frame.
      e->frame = s.live.tracker.frame();
      e->d3d12 = commands->get_device()->get_api() == api::device_api::d3d12;
      order_recorded(e->order, reinterpret_cast<std::uint64_t>(commands), commands->get_native(), s.presenting_queue);
      e->c.status = "recorded";
      publish_watch(s);
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
        if (dump.wanted && !device->create_resource(dump.desc, nullptr, api::resource_usage::shader_resource, &dump_copy))
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
        s.holding.store(s.live.entries || !s.graveyard.empty() || !s.census.empty(), std::memory_order_relaxed);
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
    auto &live = s.live;
    // A copy held for its fence is offered once the CPU sees it reached;
    // until then the copy offered before it stays offered.
    settle(s);
    const auto *entry = live.latest();
    // The executed copy is offered while the newest recorded one is recent,
    // as the single live texture was, and only when it belongs to the same
    // unbroken run of copies (recent_copy): never a copy from before a gap
    // while the first one after it is still held. The reported tick,
    // capture_id (which names the entry for bound()) and presents_since_copy
    // all describe the offered entry. An offered entry whose list executes
    // again is held until the submission hook (and, on another queue, its
    // fence) again.
    if (!device || device != s.device || !entry || entry->order.pending || !entry->copy.handle || !entry->capture_id ||
        !live.tracker.active() || !recent_copy(now_ms, live.tick, entry->tick)) return false;
    // Pinned until bound() registers this Present as its reader.
    live.reading = live.newest;
    out.copy = entry->copy; out.view = entry->view; out.capture_id = entry->capture_id; out.tick = entry->tick;
    out.format = static_cast<std::uint32_t>(live.format); out.presents_since_copy = live.tracker.presents_since(entry->frame);
    out.executed_queue = entry->order.queue;
    s.counts.offers.fetch_add(1, std::memory_order_relaxed);
    s.counts.presents_since_sum.fetch_add(out.presents_since_copy, std::memory_order_relaxed);
    for (auto seen = s.counts.presents_since_max.load(std::memory_order_relaxed); seen < out.presents_since_copy &&
         !s.counts.presents_since_max.compare_exchange_weak(seen, out.presents_since_copy, std::memory_order_relaxed);) {}
    // Queue order puts a copy run on the presenting queue alone before this
    // Present's reads; one run on another queue was offered only once the CPU
    // saw the fence signalled after its submission reach its value (settle),
    // so it completed before anything this Present submits: no wait.
    const queue_fence *fence = nullptr;
    out.order = order_of(s, *entry, fence);
    out.fence_value = fence ? entry->order.signalled : 0;
    out.in_order = out.order != read_order::unordered;
    out.direct = entry->view.handle && out.in_order;
    if (out.order == read_order::queue) return true;
    // Logged once: the first fenced copy, and each reason a copy is unordered.
    const char *reason = nullptr;
    std::uint32_t bit = log_fenced;
    if (out.order == read_order::unordered) {
      if (entry->order.queue == mixed_queue) { bit = log_mixed; reason = "it ran on more than one queue"; }
      else if (entry->order.unobserved) { bit = log_unobserved; reason = "its submission was not observed by the add-on's submission hook"; }
      else { bit = log_no_fence; reason = "no fence value was signalled after its submission"; }
    }
    if (first(s, bit)) {
      char message[400]{};
      if (reason)
        std::snprintf(message, sizeof(message),
          "Sunshine UI layer: the live layer copy executed on queue 0x%llx, not on the presenting queue 0x%llx, and no "
          "fence orders it (%s); it is copied into the renderer unordered (logged once)",
          static_cast<unsigned long long>(entry->order.queue), static_cast<unsigned long long>(s.presenting_queue), reason);
      else
        std::snprintf(message, sizeof(message),
          "Sunshine UI layer: the live layer copy executed on queue 0x%llx, not on the presenting queue 0x%llx; such a "
          "copy is offered once the CPU sees the fence value signalled after it reached, never waited for (logged once)",
          static_cast<unsigned long long>(entry->order.queue), static_cast<unsigned long long>(s.presenting_queue));
      sunshine_log::message(reshade::log::level::info, message);
    }
    return true;
  }

  stats take_stats() {
    auto &c = state().counts;
    stats value;
    value.copies = c.copies.exchange(0, std::memory_order_relaxed);
    value.skipped = c.skipped.exchange(0, std::memory_order_relaxed);
    value.offers = c.offers.exchange(0, std::memory_order_relaxed);
    value.presents_since_sum = c.presents_since_sum.exchange(0, std::memory_order_relaxed);
    value.presents_since_max = c.presents_since_max.exchange(0, std::memory_order_relaxed);
    value.skip_gap_max_us = c.skip_gap_max_us.exchange(0, std::memory_order_relaxed);
    return value;
  }

  const char *name(read_order value) {
    switch (value) {
      case read_order::queue: return "queue";
      case read_order::fence_passed: return "fence_passed";
      default: return "unordered";
    }
  }

  const char *name(census_status value) {
    switch (value) {
      case census_status::captured: return "captured_before_clear";
      case census_status::not_executed: return "not_executed";
      case census_status::reset_unexecuted: return "reset_unexecuted";
      case census_status::awaiting_submission: return "awaiting_submission";
      case census_status::awaiting_fence: return "awaiting_fence";
      case census_status::awaiting_list_reset: return "awaiting_list_reset";
      default: return "unordered";
    }
  }

  void observed_submission(std::uint64_t queue, unsigned count,
      const sunshine_streamline::native_observer::command_identity *commands) noexcept {
    if (!queue || !count || !commands) return;
    try {
      auto &s = state();
      // The listener is heard: D3D12 executions are held for it.
      const auto now = GetTickCount64();
      if (s.listened_ms.load(std::memory_order_relaxed) != now) s.listened_ms.store(now, std::memory_order_relaxed);
      // Lock-free filter: only a submission of a list a held live copy
      // waits for takes the lock.
      const auto owing = [&](std::uint64_t list) {
        if (!list) return false;
        for (const auto &value : s.owed) if (value.load(std::memory_order_relaxed) == list) return true;
        return false;
      };
      bool any = false;
      for (const auto &value : s.owed) any |= value.load(std::memory_order_relaxed) != 0;
      if (!any) return;
      bool found = false;
      for (unsigned i = 0; i != count && !found; ++i) found = owing(commands[i].native_command);
      if (!found) return;
      std::lock_guard<std::mutex> lock(s.mutex);
      signal_owed(s, queue, count, commands);
    } catch (...) {
      // A failure never propagates into the game's submission.
    }
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
        if (s.live.reading == int(i)) s.live.reading = -1;
        return;
      }
      // A new renderer: replace a reader that completed, else the first.
      slot = entry.readers[0].completed() ? &entry.readers[0] : entry.readers[1].completed() ? &entry.readers[1] :
        &entry.readers[0];
      slot->release();
      if (d3d12) (slot->fence12 = reinterpret_cast<ID3D12Fence *>(fence.handle))->AddRef();
      else (slot->fence11 = reinterpret_cast<ID3D11Fence *>(fence.handle))->AddRef();
      slot->value = value;
      if (s.live.reading == int(i)) s.live.reading = -1;
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
          for (auto &e : s.census) if (e.c.copy.handle) doomed.push_back({device, e.c.copy});
          for (unsigned i = 0; i != s.live.entries; ++i) {
            auto &entry = s.live.ring[i];
            doomed.push_back({device, entry.copy, 0, entry.view});
            for (auto &r : entry.readers) r.release();
          }
          s.live = {};
          s.census.clear();
          ++s.live_scope;
          ++s.census_scope;
          publish_watch(s);
          s.back_buffers.clear();
          s.device = nullptr;
          s.presenting_queue = 0;
        }
        // Its queue fences (no queue waits for them).
        for (auto &f : s.fences)
          if (f.device == device) release(f);
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
    // ReShade reports an execution before the native submission; on D3D12 a
    // copy is held until the submission hook runs after it
    // (observed_submission) while that hook is heard, and one run away from
    // the presenting queue owes that hook its fence signal.
    void on_execute(api::command_queue *queue, api::command_list *commands) {
      auto &s = state();
      if (!queue || !watched(s, commands)) return;
      const bool d3d12 = queue->get_device()->get_api() == api::device_api::d3d12;
      const auto now = GetTickCount64(), heard = s.listened_ms.load(std::memory_order_relaxed);
      const bool listening = heard && (now <= heard || now - heard <= listener_window_ms);
      std::lock_guard<std::mutex> lock(s.mutex);
      carriers_executed(s, reinterpret_cast<std::uint64_t>(commands), commands->get_native(), queue->get_native(), d3d12,
        listening);
      publish_watch(s);
    }

    void on_execute_secondary(api::command_list *primary, api::command_list *secondary) {
      auto &s = state();
      if (!primary || !secondary || (!watched(s, secondary) && !watched(s, primary))) return;
      std::lock_guard<std::mutex> lock(s.mutex);
      const auto list = reinterpret_cast<std::uint64_t>(secondary);
      // D3D11: ExecuteCommandList on the immediate context, the presenting
      // queue itself, executes the list.
      if (primary->get_native() && primary->get_native() == s.presenting_queue)
        carriers_executed(s, list, 0, s.presenting_queue, false, false);
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
      carriers_reset(s, reinterpret_cast<std::uint64_t>(commands), commands->get_native());
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
    // A clear adds entries on the game's recording thread without allocating.
    s.census.reserve(max_candidates);
    ++s.census_scope;
    s.armed.store(true, std::memory_order_relaxed);
  }

  bool census_pending() {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    // A copy held for its fence that the CPU now sees reached is decided.
    settle(s);
    return std::any_of(s.census.begin(), s.census.end(), [&](const census_entry &e) {
      return e.c.copy.handle ? census_waiting(census_verdict(e.order, e.d3d12, s.presenting_queue).status) :
                               e.c.status == census_allocating;
    });
  }

  std::vector<candidate> take(api::device *&device, std::uint64_t reading_queue) {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.armed.store(false, std::memory_order_relaxed);
    ++s.census_scope;
    device = s.device;
    settle(s);
    const auto now = GetTickCount64();
    std::vector<candidate> result;
    result.reserve(s.census.size());
    for (const auto &e : s.census) {
      auto c = e.c;
      // The live tracker's choice is what UI detection reads; the others are candidates only.
      c.active = c.source && c.source == s.live.tracker.active();
      c.waiting = c.status == census_allocating;
      if (c.copy.handle) {
        const auto verdict = census_verdict(e.order, e.d3d12, reading_queue);
        c.order = verdict.order;
        c.unordered_reason = verdict.reason;
        c.executed_queue = e.order.queue;
        c.fence_value = e.order.signalled;
        c.presents_since_copy = s.live.tracker.presents_since(e.frame);
        c.waiting = census_waiting(verdict.status);
        c.status = name(verdict.status);
        if (verdict.status != census_status::captured) {
          // Never read: the dump omits its artifact.
          retire_census(s, e, now);
          c.copy = {};
          if (first(s, census_log_bit(verdict))) {
            char queue[48]{}, message[320]{};
            if (e.order.queue)
              std::snprintf(queue, sizeof(queue), ", executed on queue 0x%llx",
                static_cast<unsigned long long>(e.order.queue));
            std::snprintf(message, sizeof(message),
              "Sunshine UI layer: a Dump 3D census copy was not proven complete before the dump read it (%s%s%s%s); its "
              "artifact is omitted (logged once per status and reason)",
              c.status, verdict.reason ? ", " : "", verdict.reason ? verdict.reason : "", queue);
            sunshine_log::message(reshade::log::level::info, message);
          }
        }
      }
      result.push_back(c);
    }
    s.census.clear();
    publish_watch(s);
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
  // The newest recorded copy's id, and the offered copy's id, Present count
  // and queue, as the last latest() or layer clear left them (this never
  // settles a copy).
  out->capture_id = live.capture_id;
  out->presenting_queue = s.presenting_queue;
  for (unsigned i = 0; i != live.entries; ++i) {
    out->owing += live.ring[i].order.owed != 0;
    out->awaiting += live.ring[i].order.awaiting();
  }
  // The armed census's copies, judged for reads on the presenting queue.
  for (const auto &e : s.census) {
    if (!e.c.copy.handle) continue;
    ++out->census_copies;
    out->census_pending += census_waiting(census_verdict(e.order, e.d3d12, s.presenting_queue).status);
    out->census_awaiting += e.order.awaiting();
  }
  if (const auto *entry = live.latest()) {
    out->offered_id = entry->order.pending ? 0 : entry->capture_id;
    out->presents_since_copy = live.tracker.presents_since(entry->frame);
    out->executed_queue = entry->order.queue;
    // As latest() orders it.
    const queue_fence *fence = nullptr;
    const auto order = order_of(s, *entry, fence);
    out->fence_value = fence ? entry->order.signalled : 0;
    out->order = static_cast<std::uint32_t>(order);
    out->in_order = entry->copy.handle && order != read_order::unordered;
  }
  return s.device != nullptr;
}
#endif
