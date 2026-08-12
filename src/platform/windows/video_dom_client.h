/**
 * @file src/platform/windows/video_dom_client.h
 * @brief Asynchronous, fail-closed client for the Chromium video-DOM helper.
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace platf::video_dom {
  /** A Windows desktop/capture rectangle. Edges use the Win32 half-open convention. */
  struct rect_t {
    std::int32_t left {};
    std::int32_t top {};
    std::int32_t right {};
    std::int32_t bottom {};

    [[nodiscard]] bool operator==(const rect_t &) const = default;
  };

  enum class status_e : std::uint8_t {
    starting,
    ok,
    ok_fullscreen,
    no_foreground,
    unsupported,
    unavailable,
    accessibility,
    warming,
    incomplete,
    changed,
    no_video,
    ambiguous,
    helper_missing,
    launch_failed,
    helper_exited,
    protocol_error,
    stale,
    stopped,
  };

  /** Both positive protocol statuses carry an authenticated semantic video identity/rectangle. */
  [[nodiscard]] inline constexpr bool carries_video_geometry(
    const status_e status
  ) noexcept {
    return status == status_e::ok || status == status_e::ok_fullscreen;
  }

  /**
   * Preserve the helper's fullscreen-only provenance after screen-to-capture mapping.
   * Ordinary `ok` comes from the strict windowed selector and may authorize either a windowed ROI
   * or an exact full-capture rectangle. `ok_fullscreen` may authorize only the latter.
   */
  [[nodiscard]] inline constexpr bool allows_mapped_video_rect(
    const status_e status,
    const bool exactly_full_capture
  ) noexcept {
    return status == status_e::ok ||
           (status == status_e::ok_fullscreen && exactly_full_capture);
  }

  enum class geometry_authority_class_e : std::uint8_t {
    none,
    strict_windowed,
    fullscreen_only,
  };

  /** Stable route provenance for detector lineage; heartbeat/snapshot generations are excluded. */
  [[nodiscard]] inline constexpr geometry_authority_class_e geometry_authority_class(
    const status_e status
  ) noexcept {
    if (status == status_e::ok) {
      return geometry_authority_class_e::strict_windowed;
    }
    if (status == status_e::ok_fullscreen) {
      return geometry_authority_class_e::fullscreen_only;
    }
    return geometry_authority_class_e::none;
  }

  /**
   * One immutable helper observation. Only `ok` and `ok_fullscreen` carry non-zero
   * identity/geometry.
   * Both timestamps are host-local and therefore do not trust a child-process clock.
   * received_at tracks the newest helper heartbeat. geometry_valid_since tracks when this exact
   * OK identity and rectangle began one uninterrupted run; identical heartbeats preserve it.
   */
  struct snapshot_t {
    status_e status {status_e::starting};
    std::uint64_t generation {};
    std::uint64_t helper_sequence {};
    std::uint64_t window {};
    std::uint32_t process_id {};
    std::int32_t document_id {};
    std::int32_t video_id {};
    rect_t screen_rect {};
    std::chrono::steady_clock::time_point received_at {};
    std::chrono::steady_clock::time_point geometry_valid_since {};
  };

  using snapshot_ptr = std::shared_ptr<const snapshot_t>;

  [[nodiscard]] const char *status_name(status_e status) noexcept;

  /**
   * Whether an observation is an available, fresh video rectangle.
   * The helper publishes at least one heartbeat per second; 2500 ms tolerates one delayed tick
   * without allowing old browser geometry to survive a foreground/window transition.
   */
  [[nodiscard]] bool usable(
    const snapshot_t &snapshot,
    std::chrono::steady_clock::time_point now,
    std::chrono::milliseconds maximum_age = std::chrono::milliseconds {2500}
  ) noexcept;

  enum class rotation_e : std::uint8_t {
    identity,
    rotate_90,
    rotate_180,
    rotate_270,
  };

  enum class mapping_status_e : std::uint8_t {
    ok,
    invalid_video_rect,
    invalid_capture_rect,
    unsupported_rotation,
    extent_mismatch,
    foreground_mismatch,
    outside_capture,
  };

  [[nodiscard]] const char *mapping_status_name(mapping_status_e status) noexcept;

  struct normalized_rect_t {
    float left {};
    float top {};
    float right {};
    float bottom {};

    [[nodiscard]] bool operator==(const normalized_rect_t &) const = default;
  };

  struct mapping_result_t {
    mapping_status_e status {mapping_status_e::invalid_video_rect};
    rect_t capture_pixels {};
    normalized_rect_t normalized {};

    [[nodiscard]] explicit operator bool() const noexcept {
      return status == mapping_status_e::ok;
    }
  };

  /**
   * Convert a screen-space video rectangle to capture pixels and normalized UVs.
   *
   * This deliberately supports only identity-oriented, exact-sized desktop captures. The video
   * may exceed an output edge by one physical pixel to absorb Chromium/DPI endpoint rounding; it
   * is then clipped. Any larger overflow, rotation, or capture/output extent mismatch fails flat.
   */
  [[nodiscard]] mapping_result_t map_screen_rect_to_capture(
    const rect_t &video_screen_rect,
    const rect_t &capture_desktop_rect,
    std::uint32_t capture_width,
    std::uint32_t capture_height,
    rotation_e rotation = rotation_e::identity
  ) noexcept;

  namespace detail {
    enum class lifecycle_action_e : std::uint8_t {
      none,
      start_worker,
      stop_worker,
      publish_stopped,
    };

    /**
     * Lease/worker state transitions. The controller serializes calls with its lifecycle mutex,
     * but performs the potentially blocking worker join between release() and worker_stopped().
     */
    class lifecycle_t {
    public:
      [[nodiscard]] lifecycle_action_e acquire() noexcept;
      [[nodiscard]] lifecycle_action_e release() noexcept;
      [[nodiscard]] lifecycle_action_e worker_stopped() noexcept;
      void worker_start_failed() noexcept;

    private:
      enum class worker_state_e : std::uint8_t {
        stopped,
        running,
        stopping,
      };

      std::size_t leases_ {};
      worker_state_e worker_state_ {worker_state_e::stopped};
    };

    /** Fail closed when a captured image has no desktop/content presentation timestamp. */
    [[nodiscard]] bool usable_for_content(
      const snapshot_t &snapshot,
      const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
      std::chrono::steady_clock::time_point capture_now,
      std::chrono::milliseconds maximum_age = std::chrono::milliseconds {2500}
    ) noexcept;

    struct protocol_record_t {
      status_e status {status_e::protocol_error};
      std::uint64_t sequence {};
      std::uint64_t window {};
      std::uint32_t process_id {};
      std::int32_t document_id {};
      std::int32_t video_id {};
      rect_t screen_rect {};
    };

    /**
     * Preserve the beginning of an uninterrupted geometry run only across an identical positive
     * status and heartbeat. Every status, identity, or rectangle transition starts a new run.
     */
    [[nodiscard]] std::chrono::steady_clock::time_point continued_geometry_valid_since(
      const snapshot_t *previous,
      const protocol_record_t &record,
      std::chrono::steady_clock::time_point received_at
    ) noexcept;

    /** Strict parser for one newline-free SUNSHINE_VIDEO_DOM_V1 TSV record. */
    [[nodiscard]] std::optional<protocol_record_t> parse_protocol_record(
      std::string_view line
    ) noexcept;

    [[nodiscard]] bool next_sequence(
      std::uint64_t previous,
      std::uint64_t candidate
    ) noexcept;
  }  // namespace detail

  /**
   * A process-wide helper lease. The first lease starts the supervised helper and the last lease
   * stops it. Construction never blocks on accessibility work; consumers read atomic snapshots.
   */
  class lease_t {
  public:
    lease_t();
    ~lease_t();

    lease_t(const lease_t &) = delete;
    lease_t &operator=(const lease_t &) = delete;

    lease_t(lease_t &&other) noexcept;
    lease_t &operator=(lease_t &&other) noexcept;

    [[nodiscard]] snapshot_ptr latest() const noexcept;

  private:
    bool active_ {};
  };
}  // namespace platf::video_dom
