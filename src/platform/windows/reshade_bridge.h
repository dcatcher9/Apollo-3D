// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "foreground_window_region.h"

#include <chrono>
#include <cstdint>
#include <d3d11.h>
#include <functional>
#include <memory>
#include <optional>

namespace platf::reshade_bridge {
  // The declared encoding of the exported pixels.
  enum class transfer_e : std::uint8_t {
    srgb,  ///< Display-referred sRGB code values.
    scrgb,  ///< Linear Rec.709, 1.0 = 80 cd/m2 (FP16).
    pq,  ///< Rec.2020 SMPTE ST 2084 code values, 1.0 = 10000 cd/m2 (R10G10B10A2).
  };

  struct frame_t {
    ID3D11Texture2D *texture = nullptr;
    ID3D11ShaderResourceView *view = nullptr;
    // True only for scRGB. PQ frames are HDR but not linear; read `transfer` for them.
    bool linear = false;
    transfer_e transfer = transfer_e::srgb;
    std::chrono::steady_clock::time_point timestamp {};
    std::uint64_t sequence = 0;
    // Validated publisher identity. Stable across frames within one resource generation;
    // unlike sequence, it distinguishes a restarted/replaced source for readiness revisions.
    std::uint32_t producer_process_id = 0;
    std::uint64_t producer_creation_time = 0;
    std::uint64_t resource_generation = 0;
    // Frozen with the received texture and sequence, including when a frame is reused.
    // Positive values put the cursor in front of the screen (left eye moves right).
    float ui_parallax_uv = 0.0f;
  };

  // A producer observed without attaching as its consumer. Its resource generation is zero while
  // no consumer is attached.
  struct source_status_t {
    std::uint32_t producer_process_id = 0;
    std::uint64_t producer_creation_time = 0;
    std::uint64_t resource_generation = 0;
  };

  class receiver_t {
  public:
    // Tests can supply foreground observations without moving real windows or displays.
    using observer_t = std::function<foreground_window::observation_t()>;
    receiver_t(ID3D11Device *device, ID3D11DeviceContext *context, observer_t observe = foreground_window::sample);
    ~receiver_t();
    receiver_t(const receiver_t &) = delete;
    receiver_t &operator=(const receiver_t &) = delete;

    // Nonblocking. A returned frame is the producer's shared slot, held in `reading` while it
    // is the newest frame, and may be reused while its exact foreground producer remains valid.
    // Callers only read it on this receiver's device context. Once a newer frame replaces it,
    // the slot returns to the producer only after an event query issued after its last use has
    // completed. Pointers live until the next poll/destruction.
    // Foreground ownership must cover source_rect. The returned texture keeps the authored
    // raster; its per-eye aspect must match output_width/output_height (within 0.5%) and the
    // consumer scales it. Fullscreen display scaling can make these extents independent.
    std::optional<frame_t> poll(RECT source_rect, int output_width, int output_height);

    // Status only, for a stream that does not show the export. Never attaches as the consumer, so
    // the producer creates no ring and packs no stereo for it. Returns the foreground producer
    // while it publishes a source whose eyes fit the output under poll()'s rules. A receiver is
    // used either for status() or for poll(); status() withdraws an earlier poll() connection.
    std::optional<source_status_t> status(RECT source_rect, int output_width, int output_height);

    // Tells the producer that this consumer encodes HDR10 PQ, so it may pack any HDR source
    // (scRGB included) as PQ (protocol consumer_stream_pq). The bits bind to the consumer nonce,
    // so a change while attached requests the connection again under a new nonce.
    void set_stream_pq(bool stream_pq);

    // Calls `wake` from a thread-pool thread each time the attached export's ready fence advances,
    // so the owner converts a finished frame when it completes instead of at its next poll. The
    // callback must only signal the owner. Never blocks; destruction waits for a running callback.
    void set_frame_wake(std::function<void()> wake);

    // True while poll() holds a frame and the wake is armed on its generation's fence. The owner
    // may then rely on the wake and frame_pending() instead of polling at stream cadence.
    [[nodiscard]] bool frame_wake_active() const;

    // True while poll() holds a frame of a live export.
    [[nodiscard]] bool frame_held() const;

    // Nonblocking, on the owner's thread: whether poll() could now return a newer frame or a changed
    // connection (metadata replaced, consumer replaced, producer exited or device lost). False while
    // nothing is attached, and while attached without an open generation until its metadata
    // changes: polling then cannot change anything until captures or keepalives call poll().
    [[nodiscard]] bool frame_pending() const;

    // Returns replaced slots whose reads have completed to the producer, without waiting. poll()
    // does the same; calling this once the encoder consumed a conversion frees them a frame sooner.
    void retire();

    // Every read of the frame poll() last returned is now recorded on this receiver's context (call
    // it once a conversion is recorded). The claim that replaces that frame then returns its slot
    // to the producer at once when those reads have completed, rather than one conversion later.
    void reads_recorded();

    // Diagnostics since the last call: newer frames claimed, fence wakes, claims whose frame had
    // completed before any wake reported it (found by a capture, keepalive or another frame's poll),
    // and for the others the delay from the first wake that reported their frame to the claim. That
    // delay is timed only after the first call, so a receiver without diagnostics never times it.
    struct wake_counts_t {
      std::uint64_t claims = 0;
      std::uint64_t wakes = 0;
      std::uint64_t claims_before_wake = 0;
      std::uint64_t claims_after_wake = 0;
      std::chrono::nanoseconds wake_to_claim_total {};
      std::chrono::nanoseconds wake_to_claim_max {};
    };

    wake_counts_t take_wake_counts();

  private:
    class impl_t;
    std::unique_ptr<impl_t> impl_;
  };
}  // namespace platf::reshade_bridge
