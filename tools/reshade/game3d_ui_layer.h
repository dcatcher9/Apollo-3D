// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <windows.h>
#include <reshade.hpp>
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
// tracker's Present count names (presents_since_copy). UI detection receives
// the newest copy as the
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
// carrying it. A copy that ran on the presenting queue alone is bound
// directly, since queue order puts it before the Present's reads; one that
// ran elsewhere is still offered, copied into the renderer's slot, and no
// fence of the add-on orders it. Stellar Blade's and The Witcher 3's copies
// ran on another queue than the presenting one in their dumps (both load
// Streamline), so refusing such copies would remove their layer protection.
// Copies are allocated and destroyed outside the tracker's lock.
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

  // Pure selection state of the live tracker; no GPU or runtime calls.
  class layer_tracker {
  public:
    // A qualifying clear. True when it is the active layer's first clear since
    // the last Present, which is when its previous-frame content is copied.
    bool clear(std::uint64_t resource, std::uint64_t now_ms);
    // A Present: choose the active layer among confirmed targets.
    void present(std::uint64_t now_ms);
    std::uint64_t active() const { return active_; }
    // The active layer's copy was recorded now, in the current Present
    // interval (after the last present()).
    void copied() {
      copy_frame_ = frame_;
      copied_ = true;
    }
    // Presents observed since the last copied(): 1 when the copy was
    // recorded after the previous Present and this Present was observed,
    // so the copy holds the frame that previous Present showed; 2 when an
    // interval passed without one; 0 within the copy's own interval or
    // without a copy.
    std::uint32_t presents_since_copy() const {
      return copied_ && frame_ - copy_frame_ <= 0xffffffffu ? static_cast<std::uint32_t>(frame_ - copy_frame_) : 0u;
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
    std::uint64_t order_{}, frame_{}, active_{}, copy_frame_{};
    bool captured_since_present_{}, copied_{};
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

  // Live copies: a ring of add-on owned copies (ring_capacity at most; three
  // normally suffice). A before-clear copy goes to an entry other than the
  // offered one whose last reader, a renderer submission (bound()), completed
  // and whose own copy is not pending; with every entry busy the ring grows,
  // and once full that copy is skipped and logged (saturated). A recorded copy
  // is pending, never offered, until a list carrying it executes; the offered
  // entry is the last executed copy. Pure: no GPU or runtime calls.
  inline constexpr unsigned ring_capacity = 4;
  struct ring_choice {
    int index = -1;        // The entry to write; -1 when saturated.
    bool allocate = false; // index is a new entry.
  };
  // allocated: entries [0, count) exist; free[i]: entry i's readers
  // completed and its copy neither pending nor being allocated; age[i]: its
  // capture id (smaller is older; 0 for an entry whose allocation failed,
  // which is allocated again when chosen); newest: the offered entry, the
  // last executed copy (-1: none). The oldest free entry wins.
  inline ring_choice choose_ring_entry(unsigned count, const std::array<bool, ring_capacity> &free,
      const std::array<std::uint64_t, ring_capacity> &age, int newest) {
    ring_choice choice;
    for (unsigned i = 0; i != count && i != ring_capacity; ++i)
      if (int(i) != newest && free[i] && (choice.index < 0 || age[i] < age[unsigned(choice.index)])) choice.index = int(i);
    if (choice.index < 0 && count < ring_capacity) choice = {int(count), true};
    return choice;
  }

  struct live_capture {
    api::resource copy{};       // Add-on owned, shader_resource state between uses.
    // The copy's shader view, for direct binding; direct when it exists and
    // the copy ran on the presenting queue alone (in_order: presented_in_order),
    // so that queue order puts it before this Present's reads.
    api::resource_view view{};
    bool in_order{}, direct{};
    std::uint64_t capture_id{}; // The offered entry's copy id; never zero when valid.
    std::uint64_t tick{};       // GetTickCount64 when the newest copy was recorded.
    std::uint32_t format{};     // Typed format of the copy.
    // Presents observed since the copy (layer_tracker::presents_since_copy):
    // the copy holds the frame of the Present that many back.
    std::uint32_t presents_since_copy{};
    // The native queue that executed the copy (mixed_queue: more than one).
    std::uint64_t executed_queue{};
  };
  // The active layer's last executed copy on this device, while its newest
  // copy was recorded less than max_clear_gap_ms ago. The first copy offered
  // from another queue than the presenting one is logged. Each call also asks
  // for the next copies: the layer is tracked and copied only while UI
  // detection keeps asking.
  bool latest(api::device *device, std::uint64_t now_ms, live_capture &out);

  struct candidate {
    api::resource copy{};    // Add-on owned, shader_resource state; empty unless captured.
    std::uint64_t source{};  // Game resource handle, identity only.
    std::uint32_t width{}, height{}, format{}; // DXGI format of the copy.
    std::uint32_t clears{};  // Qualifying clears seen while armed.
    bool active{};           // The live tracker's layer when the census was taken.
    const char *status = "observed";
  };

  // Test add-on only (SunshineUILayerTestLive): the live copy's queue facts.
  // capture_id is the newest recorded copy's; executed_queue the queue that
  // ran the last executed copy (mixed_queue: several); in_order whether that
  // copy ran on the presenting queue alone (presented_in_order).
  struct test_live_state {
    std::uint64_t capture_id{}, executed_queue{}, presenting_queue{};
    std::uint32_t presents_since_copy{}, in_order{};
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
  // wanted. Otherwise it only retires copies left from an earlier demand and
  // destroys retired copies once due.
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
