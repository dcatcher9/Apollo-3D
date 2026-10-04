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
// Live tracking: every qualifying clear is observed. A target cleared in at
// least confirm_clears frames with gaps under max_clear_gap_ms is a confirmed layer; of
// several, the one cleared last in the frame (UI draws last) is active. The
// active layer is copied just before the first clear after each Present: D3D12
// requires RENDER_TARGET for a clear, so the state is known, and the copy holds
// the previous frame's UI: the frame the previous Present showed, which the
// tracker's Present count names (presents_since_copy, the pairing of the
// pre-UI change set, game3d_ui_change_set.h). UI detection receives the newest copy as the
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
  // A single-sample 2D color target at the output size cleared to exactly
  // (0, 0, 0, 0). Callers separately exclude swapchain back buffers.
  bool qualifies(const api::resource_desc &desc, const float color[4], std::uint32_t width, std::uint32_t height);

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
    // Presents observed since the last copied() (fix 3): 1 when the copy was
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

  struct live_capture {
    api::resource copy{};       // Add-on owned, shader_resource state between uses.
    std::uint64_t capture_id{}; // Increases with every copy; never zero when valid.
    std::uint64_t tick{};       // GetTickCount64 when the copy was recorded.
    std::uint32_t format{};     // Typed format of the copy.
    // Presents observed since the copy (layer_tracker::presents_since_copy):
    // the copy holds the frame of the Present that many back
    // (game3d_ui_change_set.h, pairing).
    std::uint32_t presents_since_copy{};
  };
  // The active layer's newest copy on this device, recorded less than
  // max_clear_gap_ms ago. Each call also asks for the next copies: the layer is
  // copied only while UI detection keeps asking.
  bool latest(api::device *device, std::uint64_t now_ms, live_capture &out);

  struct candidate {
    api::resource copy{};    // Add-on owned, shader_resource state; empty unless captured.
    std::uint64_t source{};  // Game resource handle, identity only.
    std::uint32_t width{}, height{}, format{}; // DXGI format of the copy.
    std::uint32_t clears{};  // Qualifying clears seen while armed.
    bool active{};           // The live tracker's layer when the census was taken.
    const char *status = "observed";
  };

  void register_events();
  void unregister_events();
  // Every Present of the foreground swapchain only: output size, back buffers
  // and the live tracker's frame boundary.
  void observe_output(api::swapchain *swapchain);
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
