// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <windows.h>
#include <reshade.hpp>
#include "streamline_native_observer.h"
#include <array>
#include <cstdint>
#include <vector>

// Offscreen UI layers. HDR pipelines (Frostbite; Unreal's HDR UI composite; The
// Witcher 3 Remastered) draw UI into a color target at the output resolution
// that is cleared to transparent black each frame and composited in the final
// pass. Its alpha is a per-pixel UI mask that no vendor API exposes, including
// when Streamline's UI buffers are absent (The Witcher 3 tags them only while
// frame generation is on).
//
// Live tracking: while the layer is wanted (UI detection asked for it within
// the last second, or a Dump 3D is armed), every qualifying clear is observed;
// otherwise a clear returns before any lookup or lock, and Presents skip the
// output bookkeeping. A target cleared whole (no rectangles, or one covering
// the target) in at least confirm_clears frames with gaps under
// max_clear_gap_ms is a confirmed layer; of
// several, the one cleared last in the frame (UI draws last) is active. The
// active layer is copied just before the first clear after each Present: D3D12
// requires RENDER_TARGET for a clear, so the state is known, and the copy holds
// the previous frame's UI: the frame the previous Present showed, which the
// Present count since the copy names (presents_since_copy). UI detection
// receives the newest offered copy as the
// offscreen UI layer candidate, in a slot of its own beside any tagged
// UIColorAndAlpha (UI framework E1, candidate bit 0x40 at t7). It decides only
// once accepted by its own evidence (A1) and only in a frame where it is valid
// (V1): at most 1% of pixels out of range or above the premultiplied bound (no
// RGB above twice its alpha). The same cleared target can hold a scene image
// instead (Stellar Blade in SDR: color nearly everywhere, alpha nowhere); such
// a copy is V1-invalid, so it neither decides nor blocks another candidate
// (see docs/reshade-sbs.md, UI decision framework).
//
// Dump census: while a Dump 3D is armed, qualifying clears are also recorded and
// copied as diagnostic artifacts.
//
// Queue order: each live copy remembers the queue that executed the list
// carrying it. ReShade reports a D3D12 execution before the native
// ExecuteCommandLists, so while the add-on's submission hook (native_observer,
// observed_submission) is heard, an executed copy stays pending (never
// offered) until that hook runs after the native call returned
// (held_for_submission): a Present on another thread (a frame-generation
// pacer, a separate submission thread) meanwhile keeps reading the copy
// offered before it. A copy that ran on the presenting queue alone is then
// offered, ordered by queue order before the Present's reads (its write was
// submitted there first); without the hook heard it is offered at the
// execution and relies on the game submitting and presenting from one
// thread. The presenting queue never waits for a copy on the GPU: on D3D12
// one that ran on one other queue is ordered by a fence of the add-on that
// only the CPU reads. The hook signals that queue's fence after the native
// call, and the copy is offered only once the CPU sees the fence reach that
// value (GetCompletedValue, no wait: fence_pending); until then the copy
// offered before it stays offered (the layer is a one-frame-late input by
// design, E2). Stellar Blade's and The Witcher 3's copies ran on another queue
// than the presenting one in their dumps (both load Streamline), so refusing
// such copies would remove their layer protection. A GPU wait of the
// presenting queue for that fence (10-05) froze Stellar Blade with frame
// generation for a second: the game's queue may be paced by, or wait for,
// presenting-queue work queued after such a wait. A copy no fence orders (its
// submission unobserved, its fence not created or signalled, or run on two
// queues) is still offered at once and copied into the renderer's slot
// unordered, as before the fence, logged once per reason. D3D11 runs every
// copy on its immediate context, the presenting queue, so it never needs a
// fence.
// Copies are allocated outside the tracker's lock and destroyed outside it
// through a device alive meanwhile (presenting, recording or being
// destroyed); a retired copy of another device is destroyed under it.
namespace sunshine_game3d::ui_layer {
  namespace api = reshade::api;
  inline constexpr unsigned max_candidates = 3; // game3d_debug::ui_layer_count
  inline constexpr std::uint32_t confirm_clears = 3;
  inline constexpr std::uint64_t max_clear_gap_ms = 250;

  // Formats that can carry blended UI coverage in alpha: 8 bits or more.
  bool alpha_format(api::format format);
  // Sunshine_UIDetectionFlags of the layer slot for a copy of this format
  // (ui_detection::layer_detection_flags): the late-layer identity, the
  // premultiplied bound of V1 (no color above twice its alpha: UI blended over
  // transparent black, allowing tints brighter than white) and, for a float
  // layer, HDR headroom. A copy with color but no alpha fails that bound
  // nearly everywhere and is V1-invalid for that frame.
  std::uint32_t detection_flags(api::format format);
  // A single-sample 2D color target at the output size cleared whole to
  // exactly (0, 0, 0, 0): without rectangles, or with one that covers the
  // whole target (a partial clear, such as a viewport strip, is not a layer
  // clear). Callers separately exclude swapchain back buffers.
  bool qualifies(const api::resource_desc &desc, const float color[4], std::uint32_t width, std::uint32_t height,
    std::uint32_t rect_count, const api::rect *rects);

  // latest()'s recency gate, on GetTickCount64 ticks: the layer is still
  // being cleared (its newest recorded copy is at most max_clear_gap_ms old)
  // and the offered copy belongs to that unbroken run of copies (recorded at
  // most max_clear_gap_ms before the newest). After a gap in clears or in
  // demand the first new copy refreshes the newest tick while it is still
  // held (its submission hook, its fence on another queue), so without the
  // second bound the copy from before the gap, up to the ring's release time
  // old, was offered as this frame's layer.
  constexpr bool recent_copy(std::uint64_t now_ms, std::uint64_t newest_tick, std::uint64_t offered_tick) {
    return now_ms >= newest_tick && now_ms - newest_tick <= max_clear_gap_ms && newest_tick >= offered_tick &&
      newest_tick - offered_tick <= max_clear_gap_ms;
  }

  // Pure selection state of the live tracker; no GPU or runtime calls.
  class layer_tracker {
  public:
    // A qualifying clear. True when it is the active layer's first clear since
    // the last Present, which is when its previous-frame content is copied.
    bool clear(std::uint64_t resource, std::uint64_t now_ms);
    // A Present: choose the active layer among confirmed targets.
    void present(std::uint64_t now_ms);
    std::uint64_t active() const { return active_; }
    // The Present count: a copy records frame() when it is recorded.
    std::uint64_t frame() const { return frame_; }
    // Presents observed since a copy recorded when frame() read frame: 1 when
    // the copy was recorded after the previous Present and this Present was
    // observed, so the copy holds the frame that previous Present showed; 2
    // when an interval passed without a newer one offered (an older copy
    // still offered while a newer one is held reads its own count); 0 within
    // the copy's own interval, or for a frame after frame() (before a reset).
    std::uint32_t presents_since(std::uint64_t frame) const {
      return frame <= frame_ && frame_ - frame <= 0xffffffffu ? static_cast<std::uint32_t>(frame_ - frame) : 0u;
    }
    void reset() { *this = {}; }
  private:
    struct entry {
      std::uint64_t resource{}, last_ms{}, order{}, frame{};
      std::uint32_t streak{}; // Frames with a clear, each within max_clear_gap_ms of the last.
    };
    bool confirmed(const entry &value, std::uint64_t now_ms) const {
      return value.resource && value.streak >= confirm_clears && now_ms >= value.last_ms &&
        now_ms - value.last_ms <= max_clear_gap_ms;
    }
    std::array<entry, 8> entries_{};
    std::uint64_t order_{}, frame_{}, active_{};
    bool captured_since_present_{};
  };

  // The queue that executed a live copy (pure). A copy recorded on the
  // presenting queue itself (D3D11's immediate context) executed there; one
  // recorded into a game list executed on the queue that ran the list. A list
  // run on two queues (executed again elsewhere) has no single queue:
  // mixed_queue, which is never the presenting one.
  inline constexpr std::uint64_t mixed_queue = UINT64_MAX;
  constexpr std::uint64_t merge_queue(std::uint64_t executed, std::uint64_t queue) {
    return !queue ? executed : !executed || executed == queue ? queue : mixed_queue;
  }
  // Queue order puts a live copy before the Present's reads only when it
  // executed on the presenting queue alone (presenting: 0 unknown); only such
  // a copy is bound directly.
  constexpr bool presented_in_order(std::uint64_t executed, std::uint64_t presenting) {
    return presenting && executed == presenting;
  }
  static_assert(merge_queue(0, 5) == 5 && merge_queue(5, 5) == 5 && merge_queue(5, 6) == mixed_queue &&
    merge_queue(mixed_queue, 5) == mixed_queue && merge_queue(5, 0) == 5 && presented_in_order(5, 5) &&
    !presented_in_order(6, 5) && !presented_in_order(mixed_queue, 5) && !presented_in_order(0, 0) &&
    !presented_in_order(5, 0));

  // How the presenting queue's reads of an offered live copy are ordered
  // after it (pure). queue: it ran on the presenting queue alone.
  // fence_passed: it ran on another queue and the CPU saw that queue's fence
  // reach the value signalled after it before offering it (fence_pending).
  // unordered: no value was signalled after it.
  enum class read_order : std::uint32_t { queue, fence_passed, unordered };
  constexpr read_order order_for(bool same_queue, std::uint64_t signalled) {
    return same_queue ? read_order::queue : signalled ? read_order::fence_passed : read_order::unordered;
  }
  const char *name(read_order value);
  // A copy run on another queue owes a fence signal after its submission
  // (pure): on D3D12 only (D3D11's immediate context is the presenting
  // queue), and only when it ran on one queue that is not the presenting one.
  constexpr bool owes_signal(bool d3d12, std::uint64_t executed, std::uint64_t presenting) {
    return d3d12 && executed && executed != mixed_queue && !presented_in_order(executed, presenting);
  }
  // A D3D12 copy executed on one queue is held pending (never offered, never
  // rewritten) until the submission hook runs after its native
  // ExecuteCommandLists returned, since ReShade reports the execution before
  // that call: one on the presenting queue is then offered in queue order,
  // one owing a fence signal (owes_signal) gets it recorded. A reset of its
  // list first offers it (unobserved: unordered unless it ran on the
  // presenting queue). A D3D11 copy, one run on two queues, and one no
  // listener can see are offered at once: the submission hook was not heard
  // within listener_window_ms (the capture owner's observer inactive), so its
  // submission is not observed (pure).
  inline constexpr std::uint64_t listener_window_ms = 1000;
  constexpr bool held_for_submission(bool d3d12, std::uint64_t executed, bool listening) {
    return d3d12 && listening && executed && executed != mixed_queue;
  }
  // A copy whose fence signal is recorded stays held (pending) while the CPU
  // sees its queue's fence below that value (completed: GetCompletedValue,
  // never a wait; a removed device reads UINT64_MAX, which reaches it). Once
  // it reached it, the copy is offered unless a newer copy is (pure).
  constexpr bool fence_pending(std::uint64_t signalled, std::uint64_t completed) {
    return signalled && completed < signalled;
  }
  static_assert(order_for(true, 0) == read_order::queue && order_for(true, 3) == read_order::queue &&
    order_for(false, 0) == read_order::unordered && order_for(false, 3) == read_order::fence_passed &&
    owes_signal(true, 6, 5) && !owes_signal(false, 6, 5) && !owes_signal(true, 5, 5) && !owes_signal(true, mixed_queue, 5) &&
    !owes_signal(true, 0, 5) && owes_signal(true, 6, 0) && held_for_submission(true, 5, true) &&
    held_for_submission(true, 6, true) && !held_for_submission(true, 5, false) && !held_for_submission(false, 5, true) &&
    !held_for_submission(true, mixed_queue, true) && !held_for_submission(true, 0, true) && fence_pending(3, 2) &&
    !fence_pending(3, 3) && !fence_pending(3, 4) && !fence_pending(3, UINT64_MAX) && !fence_pending(0, 0));

  // Live copies: a ring of add-on owned copies (ring_capacity at most). A
  // before-clear copy goes to an entry other than the offered one and the one
  // a Present is reading (offered by latest(), its reader not yet registered
  // by bound()) whose last reader, a renderer submission (bound()), completed
  // and whose own copy is not pending; with every entry busy the ring grows,
  // and once full that copy is skipped (counted, take_stats, and logged once:
  // saturated). A recorded copy is pending, never offered, until a list
  // carrying it executes (on D3D12, until the submission hook ran after the
  // native call, and when it owes a fence signal until the CPU saw the fence
  // reach it: held_for_submission, fence_pending); the offered entry is the
  // newest such copy. A clear therefore needs the offered entry, one entry
  // per copy held for its fence (one per frame the copy's queue runs behind),
  // one per earlier offered copy still read by an unfinished Present (one per
  // frame the presenting queue runs behind) and the target: 2L + 2 with both
  // queues L frames behind, 6 at L = 2. The ring only grows when every entry
  // is busy, so a game that needs three entries allocates three. Pure: no GPU
  // or runtime calls.
  inline constexpr unsigned ring_capacity = 6;
  struct ring_choice {
    int index = -1;        // The entry to write; -1 when saturated.
    bool allocate = false; // index is a new entry.
  };
  // allocated: entries [0, count) exist; free[i]: entry i's readers
  // completed and its copy neither pending nor being allocated; age[i]: its
  // capture id (smaller is older; UINT64_MAX for an entry whose allocation
  // failed, the least preferred: allocated again only when no entry with a
  // copy is free); newest: the offered entry, the last executed copy (-1:
  // none); reading: the entry a Present was offered and has not yet bound
  // (-1: none). The oldest free entry other than those two wins.
  inline ring_choice choose_ring_entry(unsigned count, const std::array<bool, ring_capacity> &free,
      const std::array<std::uint64_t, ring_capacity> &age, int newest, int reading = -1) {
    ring_choice choice;
    for (unsigned i = 0; i != count && i != ring_capacity; ++i)
      if (int(i) != newest && int(i) != reading && free[i] && (choice.index < 0 || age[i] < age[unsigned(choice.index)]))
        choice.index = int(i);
    if (choice.index < 0 && count < ring_capacity) choice = {int(count), true};
    return choice;
  }

  struct live_capture {
    api::resource copy{};       // Add-on owned, shader_resource state between uses.
    // The copy's shader view, for direct binding; direct when it exists and
    // queue or fence order puts the copy before this Present's reads
    // (in_order: order is not unordered).
    api::resource_view view{};
    bool in_order{}, direct{};
    read_order order{read_order::unordered};
    // The fence value signalled after the copy on another queue (0: none).
    std::uint64_t fence_value{};
    std::uint64_t capture_id{}; // The offered entry's copy id; never zero when valid.
    std::uint64_t tick{};       // GetTickCount64 when the offered copy was recorded.
    std::uint32_t format{};     // Typed format of the copy.
    // Presents observed since the offered copy (layer_tracker::
    // presents_since): the copy holds the frame of the Present that many back.
    std::uint32_t presents_since_copy{};
    // The native queue that executed the copy (mixed_queue: more than one).
    std::uint64_t executed_queue{};
  };
  // The active layer's offered copy on this device (the newest executed one
  // that is not held: held_for_submission, fence_pending; it never waits),
  // while its newest recorded copy was recorded less than max_clear_gap_ms
  // ago and the offered one at most that long before it (recent_copy).
  // capture_id, tick and presents_since_copy all describe the offered
  // copy. The first copy offered from another queue than the presenting one
  // with a fence, and the first without one for each reason, are logged. Each
  // call also asks for the next copies: the layer is tracked and copied only
  // while UI detection keeps asking.
  bool latest(api::device *device, std::uint64_t now_ms, live_capture &out);
  // The live copies since the last call, for the periodic timing line:
  // recorded copies, copies skipped by a saturated ring, and Presents
  // latest() offered a copy to with that copy's presents_since_copy (sum
  // and maximum). The counters restart with each call.
  struct stats {
    std::uint64_t copies{}, skipped{}, offers{}, presents_since_sum{};
    std::uint32_t presents_since_max{};
  };
  stats take_stats();
  // The native observer's submission listener (native_observer::
  // set_submission_listener): after a native submission that ran a list
  // carrying a held live copy, offers a copy run on the presenting queue
  // (queue order), and for one run on another queue signals that queue's
  // fence (created on first use, one per device and queue) and records the
  // value with the copy, which latest() offers once the CPU sees the fence
  // reach it. Submissions of lists carrying no held copy return before any
  // lock; each call only stamps that the listener is heard.
  void observed_submission(std::uint64_t queue, unsigned count,
    const sunshine_streamline::native_observer::command_identity *commands) noexcept;

  struct candidate {
    api::resource copy{};    // Add-on owned, shader_resource state; empty unless captured.
    std::uint64_t source{};  // Game resource handle, identity only.
    std::uint32_t width{}, height{}, format{}; // DXGI format of the copy.
    std::uint32_t clears{};  // Qualifying clears seen while armed.
    bool active{};           // The live tracker's layer when the census was taken.
    const char *status = "observed";
  };

  // Test add-on only (SunshineUILayerTestLive): the live copy's queue facts,
  // as the last latest() or layer clear left them (the query itself never
  // offers a held copy). capture_id is the newest recorded copy's
  // and offered_id the offered copy's; executed_queue the queue that ran the
  // offered copy (mixed_queue: several); in_order whether queue or fence
  // order puts that copy before the Present's reads, order how (read_order)
  // and fence_value the value signalled after it (0: none);
  // presents_since_copy the offered copy's count, as latest() reports it;
  // owing the copies held for the submission hook, and awaiting those whose
  // fence signal is recorded but whose fence the CPU has not seen reach it.
  struct test_live_state {
    std::uint64_t capture_id{}, executed_queue{}, presenting_queue{};
    std::uint32_t presents_since_copy{}, in_order{};
    std::uint64_t fence_value{}, offered_id{};
    std::uint32_t order{}, owing{}, awaiting{};
  };

  // A renderer submission read the live copy capture_id (a direct binding or
  // the copy into the renderer's slot): that ring entry is not rewritten
  // until fence reaches value (renderer::completion_fence/completion_value).
  void bound(api::device *device, std::uint64_t capture_id, api::fence fence, std::uint64_t value);

  void register_events();
  void unregister_events();
  // Every Present of the foreground swapchain only: output size, back buffers
  // (cached until the swapchain, its size or buffer count, or its first
  // buffer changes, or ReShade reports it created or resized), the presenting
  // queue and the live tracker's frame boundary, all only while the layer is
  // wanted. Otherwise it only retires the live copies once the layer was not
  // asked for in 10 s (a shorter lapse reuses them) and destroys retired
  // copies once due.
  void observe_output(api::swapchain *swapchain, api::command_queue *queue);
  void arm();
  // Stops the census and hands over its candidates. Each copy stays valid
  // while its device lives; the caller either keeps a reference and destroys
  // its handle, or passes it to retire().
  std::vector<candidate> take(api::device *&device);
  // Destroys a copy once any game command list that wrote it has executed.
  void retire(api::device *device, api::resource copy);
  // Drops an armed census whose dump was withdrawn.
  void cancel();
}
