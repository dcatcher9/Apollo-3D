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
#include <utility>

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
      // copy is recorded but none of them executed yet, or executed but the
      // fence signal it owes is not recorded yet (owed). A pending entry is
      // never offered (it does not hold its copy yet, or that copy's write is
      // not yet ordered) nor written again.
      std::array<std::uint64_t, 2> carriers{};
      // The queue that executed the copy (merge_queue; 0 not yet).
      std::uint64_t queue{};
      // Cross-queue order (D3D12, run on one other queue than the presenting
      // one): the native list whose submission owes the copy its fence signal
      // (0: none owed), then the queue fence's slot and generation and the
      // value signalled after that submission (0: none).
      std::uint64_t owed{};
      int fence = -1;
      std::uint64_t fence_generation{}, signalled{};
      bool pending{};
      // The owed signal never came (its list was reset first, or no listener
      // was heard): offered unordered, logged as unobserved.
      bool unobserved{};
      // The entry's copy is being allocated outside the lock (capture_live).
      bool allocating{};
      bool free() const { return !pending && !allocating && readers[0].completed() && readers[1].completed(); }
      bool carries(std::uint64_t list) const { return list && (carriers[0] == list || carriers[1] == list); }
    };

    struct live_state {
      layer_tracker tracker;
      std::array<ring_entry, ring_capacity> ring{};
      unsigned entries{};
      // The offered entry: the newest executed copy that owes no fence signal
      // (-1: none). A recorded copy is promoted only when a list carrying it
      // executes, and one owing a signal only once that signal is recorded.
      int newest = -1;
      // The entry latest() last offered, pinned until bound() registers the
      // reading Present (-1: none): between them it has no reader yet, and a
      // copy promoted meanwhile may no longer be the newest. A Present that
      // renders nothing never calls bound(); the next latest() moves the pin.
      int reading = -1;
      std::uint32_t width{}, height{};
      api::format format{};
      // The newest recorded copy's id (from state_t::next_capture_id) and
      // tick (latest() reports them as the single live copy did).
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

    // A fence of the add-on on a game queue that ran live copies away from the
    // presenting queue (game3d_ui_layer.h, queue order); one per device and
    // queue.
    struct queue_fence {
      api::device *device{};
      std::uint64_t queue{};  // Native queue, identity only (never referenced).
      ID3D12Fence *fence{};   // Owned reference; null with failed set: creation failed.
      std::uint64_t generation{};
      // The last value signalled on the queue (the queue's own signals are
      // queued in increasing order, so its last one leaves exactly this) and
      // the highest value the presenting queue was told to wait for.
      std::uint64_t signalled{}, waited{};
      // revoked: a stalled wait was released from the CPU; the queue's copies
      // are read unordered for this device's life and it gets no new signal.
      bool revoked{}, failed{};
    };
    constexpr unsigned fence_capacity = 4;

    // The presenting queue's marker (game3d_ui_layer.h, queue order): it
    // signals the next value just before each cross-queue wait and the one
    // after just after it, in increasing order, so its completed value tells
    // the watchdog whether that queue reached or passed each wait. One per
    // device and presenting queue, created on first use.
    struct marker_fence {
      api::device *device{};
      std::uint64_t queue{};  // Native presenting queue, identity only.
      ID3D12Fence *fence{};   // Owned reference; null with failed set: creation failed.
      std::uint64_t generation{}, value{}; // value: the last one signalled.
      bool failed{};
    };
    constexpr unsigned marker_capacity = 2;

    // A GPU wait the presenting queue queued (order_read) and has not been
    // seen to pass: the queue fence and the value it waits for, the marker
    // and the value signalled just before it (reached; reached + 1 follows
    // it), and the fence's completed value when the watchdog last saw it
    // progress, or first saw the presenting queue at the wait (progress_ms;
    // 0 while not at it).
    struct wait_record {
      int fence = -1, marker = -1;
      std::uint64_t fence_generation{}, marker_generation{}, value{}, reached{};
      std::uint64_t progress{}, progress_ms{};
    };
    constexpr unsigned wait_capacity = 8;

    // One-time log lines.
    enum log_bit : std::uint32_t {
      log_fenced = 1u << 0, log_unobserved = 1u << 1, log_mixed = 1u << 2, log_revoked = 1u << 3,
      log_no_fence = 1u << 4, log_create_failed = 1u << 5, log_signal_failed = 1u << 6, log_wait_failed = 1u << 7,
      log_watchdog_failed = 1u << 8, log_rescued = 1u << 9, log_fence_capacity = 1u << 10, log_marker_failed = 1u << 11,
      log_wait_capacity = 1u << 12, log_released = 1u << 13,
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
      std::vector<candidate> candidates;
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
      // ring entry's carriers, which the execute and reset events check
      // without the lock, so lists that carried no live copy never take it.
      std::uint64_t presenting_queue{};
      std::array<std::atomic<std::uint64_t>, ring_capacity * 2> watched{};
      // Cross-queue order: the queue fences, and each ring entry's owed native
      // list, which the submission listener checks without the lock.
      std::array<queue_fence, fence_capacity> fences{};
      std::uint64_t fence_generation{};
      std::array<std::atomic<std::uint64_t>, ring_capacity> owed{};
      // The presenting queues' markers and the waits not yet seen passed.
      std::array<marker_fence, marker_capacity> markers{};
      std::uint64_t marker_generation{};
      std::array<wait_record, wait_capacity> outstanding_waits{};
      // The watchdog's one-shot timer, armed while a wait is not passed;
      // closed between unregister_events and register_events (no new wait).
      PTP_TIMER watchdog{};
      bool watchdog_armed{}, watchdog_closed{};
      std::atomic<std::uint64_t> waits{0}, rescues{0}, releases{0}, window_waits{0};
      // GetTickCount64 when the submission listener was last called (0:
      // never): within listener_window_ms, an execution owing a signal is
      // held for it.
      std::atomic<std::uint64_t> listened_ms{0};
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

    // Requires the state lock: publishes the ring entries' carriers and owed
    // native lists to their lock-free copies.
    void publish_watch(state_t &s) {
      for (std::size_t i = 0; i != ring_capacity; ++i) {
        for (std::size_t j = 0; j != 2; ++j)
          s.watched[i * 2 + j].store(s.live.ring[i].carriers[j], std::memory_order_relaxed);
        s.owed[i].store(s.live.ring[i].owed, std::memory_order_relaxed);
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
    // no longer covers it; on D3D12 a copy run on one queue other than the
    // presenting one owes a signal after this submission (native_list, which
    // observed_submission matches). ReShade reports the execution before the
    // native call, so such a copy stays pending, neither offered nor
    // rewritten, until that signal is recorded (held_for_signal): a Present
    // meanwhile keeps reading the previous copy. Without a listener heard
    // (listening) it is offered now, unobserved. The newest of the others
    // becomes the offered copy (the last executed copy, as a single live
    // texture would hold).
    void carriers_executed(state_t &s, std::uint64_t list, std::uint64_t native_list, std::uint64_t queue, bool d3d12,
        bool listening) {
      auto &live = s.live;
      int promoted = -1;
      for (unsigned i = 0; i != live.entries; ++i) {
        auto &entry = live.ring[i];
        if (!entry.carries(list)) continue;
        entry.queue = merge_queue(entry.queue, queue);
        entry.fence = -1;
        entry.fence_generation = entry.signalled = 0;
        const bool owes = native_list && owes_signal(d3d12, entry.queue, s.presenting_queue);
        entry.unobserved = owes && !listening;
        entry.owed = held_for_signal(owes, listening) ? native_list : 0;
        entry.pending = entry.owed != 0;
        if (entry.pending) continue;
        if (promoted < 0 || entry.capture_id > live.ring[unsigned(promoted)].capture_id) promoted = int(i);
      }
      if (promoted >= 0) live.newest = promoted;
    }

    // Requires the state lock. A list was reset: it carries nothing; an
    // entry no executed list wrote drops back to free (never offered). A
    // signal the list's submission owed can no longer come (it is matched
    // within that submission): an entry held for it executed without one and
    // is offered unordered (unobserved).
    void carriers_reset(state_t &s, std::uint64_t list, std::uint64_t native_list) {
      for (unsigned i = 0; i != s.live.entries; ++i) {
        auto &entry = s.live.ring[i];
        bool unsigned_execution = false;
        if (native_list && entry.owed == native_list) {
          entry.owed = 0;
          unsigned_execution = entry.pending;
        }
        if (entry.carries(list)) {
          for (auto &c : entry.carriers) if (c == list) c = 0;
          if (!entry.carriers[0] && !entry.carriers[1] && entry.pending) {
            if (entry.owed) {
              entry.owed = 0;
              unsigned_execution = true;
            } else entry.pending = false;
          }
        }
        if (!unsigned_execution) continue;
        entry.pending = false;
        entry.unobserved = true;
        promote(s.live, i);
      }
    }

    void release(queue_fence &f) {
      if (f.fence) f.fence->Release();
      f = {};
    }

    // Requires the state lock. The queue fence in slot, while that record
    // still holds generation's fence (null: none).
    queue_fence *fence_at(state_t &s, int slot, std::uint64_t generation) {
      if (slot < 0 || unsigned(slot) >= fence_capacity) return nullptr;
      auto &f = s.fences[unsigned(slot)];
      return f.fence && f.generation == generation ? &f : nullptr;
    }
    // Requires the state lock. The queue fence a live copy names, while that
    // record still holds the fence it was signalled on (null: none).
    queue_fence *fence_of(state_t &s, const ring_entry &entry) {
      return fence_at(s, entry.fence, entry.fence_generation);
    }
    // Requires the state lock. The marker a wait names, while that record
    // still holds it (null: none).
    marker_fence *marker_of(state_t &s, const wait_record &w) {
      if (w.marker < 0 || unsigned(w.marker) >= marker_capacity) return nullptr;
      auto &m = s.markers[unsigned(w.marker)];
      return m.fence && m.generation == w.marker_generation ? &m : nullptr;
    }

    // Requires the state lock. True while the presenting queue has not passed
    // the wait in w; a passed wait, or one whose fence or marker was released
    // (its device destroyed, its waits released first), is cleared.
    bool outstanding(state_t &s, wait_record &w) {
      if (w.fence < 0) return false;
      const auto *m = marker_of(s, w);
      if (m && fence_at(s, w.fence, w.fence_generation) && m->fence->GetCompletedValue() <= w.reached) return true;
      w = {};
      return false;
    }

    // Requires the state lock. How the presenting queue's reads are ordered
    // after a live copy now, and the queue fence that orders it (null: none).
    read_order order_of(state_t &s, const ring_entry &entry, const queue_fence *&fence) {
      const bool same = presented_in_order(entry.queue, s.presenting_queue);
      fence = same ? nullptr : fence_of(s, entry);
      const auto signalled = fence ? entry.signalled : 0;
      return order_for(same, signalled, signalled ? fence->fence->GetCompletedValue() : 0, fence && fence->revoked);
    }

    // Requires the state lock. A record no live copy names and no GPU work
    // still uses: every signal and wait passed. A revoked record of the
    // current device stays, so its queue stays revoked.
    bool reusable(state_t &s, unsigned index) {
      const auto &f = s.fences[index];
      if (f.failed) return f.device != s.device;
      if (!f.fence || (f.revoked && f.device == s.device)) return false;
      for (unsigned i = 0; i != s.live.entries; ++i)
        if (s.live.ring[i].fence == int(index) && s.live.ring[i].fence_generation == f.generation) return false;
      for (auto &w : s.outstanding_waits)
        if (w.fence == int(index) && w.fence_generation == f.generation && outstanding(s, w)) return false;
      const auto completed = f.fence->GetCompletedValue();
      return completed >= f.signalled && completed >= f.waited;
    }

    void release(marker_fence &m) {
      if (m.fence) m.fence->Release();
      m = {};
    }

    // Requires the state lock. The marker of the current device's presenting
    // queue (native), created on first use from that queue's device; -1 when
    // none can be had (logged once). A marker is reused only once every signal
    // it was given ran (so every wait it brackets passed).
    int marker_slot(state_t &s, std::uint64_t queue) {
      for (unsigned i = 0; i != marker_capacity; ++i) {
        const auto &m = s.markers[i];
        if ((m.fence || m.failed) && m.device == s.device && m.queue == queue) return m.fence ? int(i) : -1;
      }
      int slot = -1;
      for (unsigned i = 0; i != marker_capacity && slot < 0; ++i)
        if (!s.markers[i].fence && !s.markers[i].failed) slot = int(i);
      for (unsigned i = 0; i != marker_capacity && slot < 0; ++i) {
        const auto &m = s.markers[i];
        if (m.failed ? m.device != s.device : m.fence->GetCompletedValue() >= m.value) slot = int(i);
      }
      if (slot >= 0) {
        auto &m = s.markers[unsigned(slot)];
        release(m);
        m.device = s.device;
        m.queue = queue;
        m.generation = ++s.marker_generation;
        ID3D12Device *device = nullptr;
        if (SUCCEEDED(reinterpret_cast<ID3D12CommandQueue *>(queue)->GetDevice(IID_PPV_ARGS(&device))) && device) {
          if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m.fence)))) m.fence = nullptr;
          device->Release();
        }
        if (m.fence) return slot;
        m.failed = true;
      }
      if (first(s, log_marker_failed)) {
        char message[240]{};
        std::snprintf(message, sizeof(message),
          "Sunshine UI layer: no marker fence for the presenting queue 0x%llx; layer copies on another queue are copied "
          "into the renderer unordered (logged once)",
          static_cast<unsigned long long>(queue));
        sunshine_log::message(reshade::log::level::warning, message);
      }
      return -1;
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
    // in commands: each live copy owing a signal to one of them gets the
    // queue fence's next value, signalled now, after that submission, and is
    // offered from now on (unordered when no value could be signalled).
    void signal_owed(state_t &s, std::uint64_t queue, unsigned count,
        const sunshine_streamline::native_observer::command_identity *commands) {
      auto &live = s.live;
      std::array<unsigned, ring_capacity> due{};
      unsigned owing = 0;
      for (unsigned i = 0; i != live.entries; ++i) {
        const auto owed = live.ring[i].owed;
        for (unsigned c = 0; owed && c != count; ++c)
          if (commands[c].native_command == owed) { due[owing++] = i; break; }
      }
      if (!owing) return;
      const int slot = fence_slot(s, queue);
      std::uint64_t value = 0, generation = 0;
      if (slot >= 0) {
        auto &f = s.fences[unsigned(slot)];
        generation = f.generation;
        if (f.revoked) {
          // Its copies stay unordered; latest() names the reason.
        } else if (SUCCEEDED(reinterpret_cast<ID3D12CommandQueue *>(queue)->Signal(f.fence, f.signalled + 1))) {
          value = ++f.signalled;
        } else if (first(s, log_signal_failed)) {
          char message[200]{};
          std::snprintf(message, sizeof(message),
            "Sunshine UI layer: could not signal the fence of queue 0x%llx after a layer copy; it stays unordered (logged once)",
            static_cast<unsigned long long>(queue));
          sunshine_log::message(reshade::log::level::warning, message);
        }
      }
      for (unsigned k = 0; k != owing; ++k) {
        auto &entry = live.ring[due[k]];
        entry.owed = 0;
        entry.fence = slot;
        entry.fence_generation = generation;
        entry.signalled = value;
        if (!entry.pending) continue;
        entry.pending = false;
        promote(live, due[k]);
      }
      publish_watch(s);
    }

    void CALLBACK watchdog_tick(PTP_CALLBACK_INSTANCE, PVOID, PTP_TIMER);

    // Requires the state lock. Arms the watchdog's next look (one-shot), unless
    // armed; false when its timer cannot be created.
    bool arm_watchdog(state_t &s) {
      if (s.watchdog_closed) return false;
      if (!s.watchdog) s.watchdog = CreateThreadpoolTimer(watchdog_tick, nullptr, nullptr);
      if (!s.watchdog) return false;
      if (!s.watchdog_armed) {
        ULARGE_INTEGER relative;
        relative.QuadPart = static_cast<ULONGLONG>(-static_cast<LONGLONG>(watchdog_period_ms * 10000));
        FILETIME due{relative.LowPart, relative.HighPart};
        SetThreadpoolTimer(s.watchdog, &due, 0, 0);
        s.watchdog_armed = true;
      }
      return true;
    }

    // Requires the state lock. Releases every wait the presenting queue queued
    // on f by signalling it from the CPU to the last value its queue was told
    // to signal (at least every value waited for). The queue's own signals
    // still queued may move it back below a later wait (D3D12 fences follow
    // the last signal); the watchdog releases such a wait again.
    bool release_waits(queue_fence &f) {
      return f.fence && SUCCEEDED(f.fence->Signal(f.signalled));
    }

    // Requires the state lock. A stalled wait: the presenting queue has been
    // blocked at it for blocked_ms while the copy's fence made no progress;
    // the game's queue waits for presenting-queue work behind the add-on's
    // wait. The wait is released and the queue's copies are read unordered for
    // this device's life.
    void rescue(state_t &s, queue_fence &f, std::uint64_t completed, std::uint64_t value, std::uint64_t blocked_ms) {
      f.revoked = true;
      const bool released = release_waits(f);
      s.rescues.fetch_add(1, std::memory_order_relaxed);
      if (!first(s, log_rescued)) return;
      char message[400]{};
      std::snprintf(message, sizeof(message),
        "Sunshine UI layer: the presenting queue was blocked at its wait for a layer copy on queue 0x%llx for %llu ms "
        "with no fence progress (fence %llu of %llu; the game's queue waits for later presenting-queue work); %s it "
        "from the CPU and reads that queue's copies unordered from now on (logged once)",
        static_cast<unsigned long long>(f.queue), static_cast<unsigned long long>(blocked_ms),
        static_cast<unsigned long long>(completed), static_cast<unsigned long long>(value),
        released ? "released" : "could not release");
      sunshine_log::message(reshade::log::level::warning, message);
    }

    // Requires the state lock. A revoked fence that the game queue's own
    // earlier signal moved back below a wait the presenting queue is blocked
    // at: released again.
    void release_again(state_t &s, queue_fence &f, std::uint64_t completed, std::uint64_t value) {
      const bool released = release_waits(f);
      s.releases.fetch_add(1, std::memory_order_relaxed);
      if (!first(s, log_released)) return;
      char message[400]{};
      std::snprintf(message, sizeof(message),
        "Sunshine UI layer: the revoked fence of queue 0x%llx went back to %llu below a wait for %llu by that queue's own "
        "earlier signal; %s it from the CPU again (logged once)",
        static_cast<unsigned long long>(f.queue), static_cast<unsigned long long>(completed),
        static_cast<unsigned long long>(value), released ? "released" : "could not release");
      sunshine_log::message(reshade::log::level::warning, message);
    }

    // The watchdog's look, every watchdog_period_ms while a wait is not
    // passed (watch_action): the stall clock of a wait runs only while the
    // presenting queue is at it (its marker) and the fence makes no progress.
    // Teardown releases waits itself.
    void CALLBACK watchdog_tick(PTP_CALLBACK_INSTANCE, PVOID, PTP_TIMER) {
      if (sunshine_addon_lifetime::stopping()) return;
      try {
        auto &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        s.watchdog_armed = false;
        const auto now = GetTickCount64();
        bool any = false;
        for (auto &w : s.outstanding_waits) {
          if (!outstanding(s, w)) continue;
          auto &f = *fence_at(s, w.fence, w.fence_generation);
          const auto marker = marker_of(s, w)->fence->GetCompletedValue();
          const auto completed = f.fence->GetCompletedValue(); // UINT64_MAX once removed: passed.
          if (marker == w.reached && completed < w.value) {
            if (!w.progress_ms || completed != w.progress) {
              w.progress = completed;
              w.progress_ms = now;
            }
          } else w.progress_ms = 0;
          switch (watch(marker, w.reached, completed, w.value, f.revoked, w.progress_ms, now)) {
            case watch_action::done: w = {}; continue;
            case watch_action::rescue: rescue(s, f, completed, w.value, now - w.progress_ms); break;
            case watch_action::release: release_again(s, f, completed, w.value); break;
            case watch_action::waiting: break;
          }
          any = true;
        }
        if (any) arm_watchdog(s);
      } catch (...) {
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
      entry.capture_id = live.capture_id = ++s.next_capture_id;
      entry.tick = live.tick = GetTickCount64();
      entry.queue = 0;
      entry.owed = 0;
      entry.unobserved = false;
      entry.fence = -1;
      entry.fence_generation = entry.signalled = 0;
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
        live.newest = live.reading = -1;
        ++s.live_scope;
        publish_watch(s);
      }
      live.width = desc.texture.width; live.height = desc.texture.height; live.format = typed;
      std::array<bool, ring_capacity> free{};
      std::array<std::uint64_t, ring_capacity> age{};
      bool abandoned = false;
      for (unsigned i = 0; i != live.entries; ++i) {
        auto &entry = live.ring[i];
        // A copy no carrying list executed or reset this long, or whose owed
        // signal never came, is abandoned (never offered).
        if (entry.pending && now - entry.tick >= retire_delay_ms) {
          entry.pending = false;
          entry.carriers = {};
          entry.owed = 0;
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
    auto &live = s.live;
    const auto *entry = live.latest();
    // The executed copy is offered while the newest recorded one is recent,
    // as the single live texture was (its tick is the newest recorded copy's;
    // capture_id names the offered entry for bound()). An offered entry whose
    // list executes again and owes a new signal is held until that signal.
    if (!device || device != s.device || !entry || entry->pending || !entry->copy.handle || !entry->capture_id ||
        !live.tracker.active() || now_ms < live.tick || now_ms - live.tick > max_clear_gap_ms) return false;
    // Pinned until bound() registers this Present as its reader.
    live.reading = live.newest;
    out.copy = entry->copy; out.view = entry->view; out.capture_id = entry->capture_id; out.tick = live.tick;
    out.format = static_cast<std::uint32_t>(live.format); out.presents_since_copy = live.tracker.presents_since_copy();
    out.executed_queue = entry->queue;
    // Queue order puts a copy run on the presenting queue alone before this
    // Present's reads; a fence signalled after its submission orders one run
    // on another queue (a wait unless the fence already passed it).
    const queue_fence *fence = nullptr;
    out.order = order_of(s, *entry, fence);
    out.fence_value = fence ? entry->signalled : 0;
    out.in_order = out.order != read_order::unordered;
    out.direct = entry->view.handle && out.in_order;
    if (out.order == read_order::fence_wait) {
      out.wait_fence = {reinterpret_cast<std::uint64_t>(fence->fence)};
      out.wait_value = out.fence_value;
    }
    if (out.order == read_order::queue) return true;
    // Logged once: the first fenced copy, and each reason a copy is unordered.
    const char *reason = nullptr;
    std::uint32_t bit = log_fenced;
    if (out.order == read_order::unordered) {
      if (entry->queue == mixed_queue) { bit = log_mixed; reason = "it ran on more than one queue"; }
      else if (entry->unobserved) { bit = log_unobserved; reason = "its submission was not observed by the add-on's submission hook"; }
      else if (fence && fence->revoked) { bit = log_revoked; reason = "that queue's ordering was revoked after a stalled wait"; }
      else { bit = log_no_fence; reason = "no fence value was signalled after its submission"; }
    }
    if (first(s, bit)) {
      char message[400]{};
      if (reason)
        std::snprintf(message, sizeof(message),
          "Sunshine UI layer: the live layer copy executed on queue 0x%llx, not on the presenting queue 0x%llx, and no "
          "fence orders it (%s); it is copied into the renderer unordered (logged once)",
          static_cast<unsigned long long>(entry->queue), static_cast<unsigned long long>(s.presenting_queue), reason);
      else
        std::snprintf(message, sizeof(message),
          "Sunshine UI layer: the live layer copy executed on queue 0x%llx, not on the presenting queue 0x%llx; the "
          "presenting queue's reads wait for the fence value signalled after it (logged once)",
          static_cast<unsigned long long>(entry->queue), static_cast<unsigned long long>(s.presenting_queue));
      sunshine_log::message(reshade::log::level::info, message);
    }
    return true;
  }

  const char *name(read_order value) {
    switch (value) {
      case read_order::queue: return "queue";
      case read_order::fence_passed: return "fence_passed";
      case read_order::fence_wait: return "fence_wait";
      default: return "unordered";
    }
  }

  bool order_read(api::command_queue *queue, const live_capture &capture) {
    if (capture.order != read_order::fence_wait) return capture.in_order;
    if (!queue || !capture.wait_fence.handle || !capture.wait_value ||
        queue->get_device()->get_api() != api::device_api::d3d12) return false;
    const auto presenting = queue->get_native();
    if (!presenting) return false;
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    queue_fence *f = nullptr;
    int fence_index = -1;
    for (unsigned i = 0; i != fence_capacity; ++i)
      if (s.fences[i].fence && reinterpret_cast<std::uint64_t>(s.fences[i].fence) == capture.wait_fence.handle) {
        f = &s.fences[i];
        fence_index = int(i);
      }
    // Revoked since latest(): unordered, as that queue's later copies.
    if (!f || f->revoked || capture.wait_value > f->signalled) {
      if (f && f->revoked && first(s, log_revoked))
        sunshine_log::message(reshade::log::level::info,
          "Sunshine UI layer: a layer copy's queue was revoked before its wait; it is copied into the renderer unordered (logged once)");
      return false;
    }
    const int marker_index = marker_slot(s, presenting);
    if (marker_index < 0) return false;
    // A wait this presenting queue already queued on this fence for at least
    // this value, and has not passed yet, orders every later read on the
    // queue: a copy re-offered on the Presents of one real frame (frame
    // generation) reuses it, with no signal, wait, record or watchdog.
    for (auto &w : s.outstanding_waits)
      if (w.fence == fence_index && w.fence_generation == f->generation && w.marker == marker_index &&
          w.marker_generation == s.markers[unsigned(marker_index)].generation && w.value >= capture.wait_value &&
          outstanding(s, w))
        return true;
    // A record for the watchdog: the waits not yet passed are bounded.
    wait_record *record = nullptr;
    for (auto &w : s.outstanding_waits)
      if (!outstanding(s, w)) { record = &w; break; }
    if (!record) {
      if (first(s, log_wait_capacity))
        sunshine_log::message(reshade::log::level::warning,
          "Sunshine UI layer: too many cross-queue waits are not passed yet; a layer copy on another queue is copied into "
          "the renderer unordered (logged once)");
      return false;
    }
    // The watchdog first: a wait it cannot bound is never queued.
    if (!arm_watchdog(s)) {
      if (first(s, log_watchdog_failed))
        sunshine_log::message(reshade::log::level::warning,
          "Sunshine UI layer: could not start the cross-queue watchdog; layer copies on another queue are copied into "
          "the renderer unordered (logged once)");
      return false;
    }
    // The marker brackets the wait on the presenting queue: reached just
    // before it, reached + 1 just after it, so the watchdog's stall clock runs
    // only while that queue is blocked there. Native signals (ReShade's
    // command_queue::signal would flush its immediate list first) and
    // ReShade's wait (the native Wait) queue in call order on the same queue.
    auto &m = s.markers[unsigned(marker_index)];
    auto *native = reinterpret_cast<ID3D12CommandQueue *>(presenting);
    if (FAILED(native->Signal(m.fence, m.value + 1))) {
      if (first(s, log_signal_failed))
        sunshine_log::message(reshade::log::level::warning,
          "Sunshine UI layer: could not signal the presenting queue's marker before a wait; the layer copy is copied into "
          "the renderer unordered (logged once)");
      return false;
    }
    const auto reached = ++m.value;
    const bool waited = queue->wait(capture.wait_fence, capture.wait_value);
    // After the wait, or in its place: the marker always passes reached.
    if (SUCCEEDED(native->Signal(m.fence, m.value + 1))) ++m.value;
    else if (waited && first(s, log_signal_failed))
      sunshine_log::message(reshade::log::level::warning,
        "Sunshine UI layer: could not signal the presenting queue's marker after a wait; the watchdog treats that queue "
        "as at the wait (logged once)");
    if (!waited) {
      if (first(s, log_wait_failed))
        sunshine_log::message(reshade::log::level::warning,
          "Sunshine UI layer: the presenting queue could not wait for a layer copy's fence; it is copied into the "
          "renderer unordered (logged once)");
      return false;
    }
    *record = {};
    record->fence = fence_index;
    record->fence_generation = f->generation;
    record->marker = marker_index;
    record->marker_generation = m.generation;
    record->value = capture.wait_value;
    record->reached = reached;
    f->waited = std::max(f->waited, capture.wait_value);
    s.waits.fetch_add(1, std::memory_order_relaxed);
    s.window_waits.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  std::uint64_t take_wait_count() { return state().window_waits.exchange(0, std::memory_order_relaxed); }

  void observed_submission(std::uint64_t queue, unsigned count,
      const sunshine_streamline::native_observer::command_identity *commands) noexcept {
    if (!queue || !count || !commands) return;
    try {
      auto &s = state();
      // The listener is heard: executions owing a signal are held for it.
      const auto now = GetTickCount64();
      if (s.listened_ms.load(std::memory_order_relaxed) != now) s.listened_ms.store(now, std::memory_order_relaxed);
      // Lock-free filter: only a submission of a list a live copy owes its
      // signal to takes the lock.
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
        // Its queue fences: an outstanding wait is released first, so none
        // outlives the fence; then its markers. The waits they name are
        // cleared with them (outstanding()).
        for (auto &f : s.fences) {
          if (f.device != device) continue;
          if (f.fence && f.waited && f.fence->GetCompletedValue() < f.waited) release_waits(f);
          release(f);
        }
        for (auto &m : s.markers)
          if (m.device == device) release(m);
        for (auto &w : s.outstanding_waits) outstanding(s, w);
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
    // copy run away from the presenting queue owes its fence signal to the
    // submission hook, which runs after it (observed_submission), and is held
    // until then while that hook is heard.
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
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.watchdog_closed = false;
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
    // No wait of the add-on outlives it: outstanding waits are released and
    // their queues revoked (no new wait or signal), then the watchdog stops
    // (its look takes the lock, so it is awaited outside it).
    auto &s = state();
    PTP_TIMER timer = nullptr;
    {
      std::lock_guard<std::mutex> lock(s.mutex);
      for (auto &f : s.fences)
        if (f.fence && f.waited && f.fence->GetCompletedValue() < f.waited) {
          release_waits(f);
          f.revoked = true;
        }
      timer = std::exchange(s.watchdog, nullptr);
      s.watchdog_armed = false;
      s.watchdog_closed = true;
    }
    if (timer) {
      SetThreadpoolTimer(timer, nullptr, 0, 0);
      WaitForThreadpoolTimerCallbacks(timer, TRUE);
      CloseThreadpoolTimer(timer);
    }
    // A game queue's own signals still queued can move a released fence back
    // below a wait the presenting queue has not passed; it is released again
    // until every wait passed, for at most unload_drain_ms.
    for (const auto deadline = GetTickCount64() + unload_drain_ms;;) {
      bool any = false;
      {
        std::lock_guard<std::mutex> lock(s.mutex);
        for (auto &w : s.outstanding_waits) {
          if (!outstanding(s, w)) continue;
          any = true;
          auto &f = *fence_at(s, w.fence, w.fence_generation);
          f.revoked = true;
          if (f.fence->GetCompletedValue() < w.value) release_waits(f);
        }
      }
      if (!any || GetTickCount64() >= deadline) break;
      Sleep(1);
    }
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
  // The newest recorded copy's id, and the offered copy's id and queue.
  out->capture_id = live.capture_id;
  out->presenting_queue = s.presenting_queue;
  out->presents_since_copy = live.tracker.presents_since_copy();
  for (unsigned i = 0; i != live.entries; ++i) out->owing += live.ring[i].owed != 0;
  if (const auto *entry = live.latest()) {
    out->offered_id = entry->pending ? 0 : entry->capture_id;
    out->executed_queue = entry->queue;
    // As latest() orders it now.
    const queue_fence *fence = nullptr;
    const auto order = order_of(s, *entry, fence);
    out->fence_value = fence ? entry->signalled : 0;
    out->order = static_cast<std::uint32_t>(order);
    out->revoked = fence && fence->revoked;
    out->in_order = entry->copy.handle && order != read_order::unordered;
  }
  out->waits = s.waits.load(std::memory_order_relaxed);
  out->rescues = s.rescues.load(std::memory_order_relaxed);
  out->releases = s.releases.load(std::memory_order_relaxed);
  return s.device != nullptr;
}
#endif
