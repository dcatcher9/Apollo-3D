/**
 * @file src/platform/windows/foreground_window_region.h
 * @brief Synchronous, fail-closed foreground-window geometry for Host SBS.
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <span>

namespace platf::foreground_window {
  /** A half-open rectangle in physical virtual-screen or capture-pixel coordinates. */
  struct rect_t {
    std::int32_t left {};
    std::int32_t top {};
    std::int32_t right {};
    std::int32_t bottom {};

    [[nodiscard]] constexpr bool valid() const noexcept {
      return right > left && bottom > top;
    }

    [[nodiscard]] constexpr bool operator==(const rect_t &) const = default;
  };

  enum class status_e : std::uint8_t {
    ok,
    no_foreground,
    shell_surface,
    self_process,
    invalid_window,
    not_visible,
    minimized,
    cloaked,
    excluded_style,
    dpi_unavailable,
    geometry_unavailable,
    foreground_changed,
    interactive_move_size,
  };

  [[nodiscard]] const char *status_name(status_e status) noexcept;

  [[nodiscard]] inline constexpr bool carries_geometry(const status_e status) noexcept {
    return status == status_e::ok;
  }

  /** One synchronous Win32/DWM observation. Native handles are transported as integers. */
  struct observation_t {
    status_e status {status_e::no_foreground};
    // `window` remains the foreground top-level root. A structurally proven direct child may
    // narrow only the analysis source; it never replaces foreground identity authority.
    std::uintptr_t window {};
    std::uint32_t process_id {};
    std::uintptr_t monitor {};
    // Optional physical monitor bounds. Production fills this from GetMonitorInfoW; keeping it
    // alongside the HMONITOR lets cloned outputs prove coordinate equivalence without accepting
    // an unrelated monitor merely because its window rectangle happens to fit.
    rect_t monitor_screen_rect {};
    // Exact root client and selected analysis content in physical screen coordinates. Content is
    // the client itself unless the conservative direct-child resolver proves a stable viewport.
    rect_t client_screen_rect {};
    std::uintptr_t content_window {};
    rect_t content_screen_rect {};
    rect_t frame_screen_rect {};
    std::chrono::steady_clock::time_point observed_at {};

    [[nodiscard]] constexpr bool has_content_source() const noexcept {
      return content_window != 0 && content_screen_rect.valid();
    }

    [[nodiscard]] constexpr std::uintptr_t source_window() const noexcept {
      return has_content_source() ? content_window : window;
    }

    [[nodiscard]] constexpr rect_t source_screen_rect() const noexcept {
      return has_content_source() ? content_screen_rect : client_screen_rect;
    }
  };

  /**
   * One observation with continuity provenance.
   *
   * Positive snapshots have a monotonically changing, nonzero generation. `geometry_valid_since`
   * is preserved only while the exact positive status, identity, monitor, and geometry remain
   * unchanged. Invalid observations break continuity and carry zero authority fields.
   */
  struct snapshot_t {
    status_e status {status_e::no_foreground};
    std::uint64_t generation {};
    std::uintptr_t window {};
    std::uint32_t process_id {};
    std::uintptr_t monitor {};
    rect_t monitor_screen_rect {};
    rect_t client_screen_rect {};
    std::uintptr_t content_window {};
    rect_t content_screen_rect {};
    rect_t frame_screen_rect {};
    std::chrono::steady_clock::time_point observed_at {};
    std::chrono::steady_clock::time_point geometry_valid_since {};

    [[nodiscard]] constexpr bool has_content_source() const noexcept {
      return content_window != 0 && content_screen_rect.valid();
    }

    [[nodiscard]] constexpr std::uintptr_t source_window() const noexcept {
      return has_content_source() ? content_window : window;
    }

    [[nodiscard]] constexpr rect_t source_screen_rect() const noexcept {
      return has_content_source() ? content_screen_rect : client_screen_rect;
    }
  };

  /** Sample the current foreground root with no worker, blocking IPC, or retained native handle. */
  [[nodiscard]] observation_t sample() noexcept;

  /**
   * Sample only USER32's native move/size advisory.
   *
   * WGC has no desktop-content timestamp and therefore cannot authorize foreground geometry, but
   * it can still suppress optional subtitle work while the foreground root is in its modal
   * move/size loop. This path deliberately avoids DWM bounds, styles, and client mapping.
   */
  [[nodiscard]] bool interactive_move_size_active() noexcept;

  /** Assign continuity and observer generations to successive synchronous observations. */
  class continuity_tracker_t {
  public:
    [[nodiscard]] snapshot_t update(const observation_t &observation) noexcept;
    void reset() noexcept;

  private:
    struct key_t {
      status_e status {status_e::no_foreground};
      std::uintptr_t window {};
      std::uint32_t process_id {};
      std::uintptr_t monitor {};
      rect_t monitor_screen_rect {};
      rect_t client_screen_rect {};
      std::uintptr_t content_window {};
      rect_t content_screen_rect {};
      rect_t frame_screen_rect {};

      [[nodiscard]] constexpr bool operator==(const key_t &) const = default;
    };

    std::optional<key_t> key_;
    std::uint64_t generation_ {};
    std::chrono::steady_clock::time_point geometry_valid_since_ {};
  };

  /**
   * Require fresh observation liveness and geometry that predates the captured desktop content.
   * A missing content timestamp (including WGC) fails closed.
   */
  [[nodiscard]] bool usable_for_content(
    const snapshot_t &snapshot,
    const std::optional<std::chrono::steady_clock::time_point> &content_timestamp,
    std::chrono::steady_clock::time_point capture_now,
    std::chrono::milliseconds maximum_age = std::chrono::milliseconds {250}
  ) noexcept;

  /**
   * Detect the narrow same-analysis-shape causality gap used by the capture owner.
   *
   * A programmatic same-size move can be observed before Desktop Duplication publishes pixels at
   * the new location. That copied frame must use full-source analysis rather than stall waiting
   * forever for another desktop present. Identity or size changes are deliberately excluded; they
   * already take the ordinary unavailable/full-source route.
   */
  [[nodiscard]] bool requires_full_source_causal_fallback(
    const snapshot_t &previous_snapshot,
    const snapshot_t &current_snapshot,
    const std::optional<std::chrono::steady_clock::time_point> &content_timestamp
  ) noexcept;

  /** Raw selected-output identity supplied from DXGI_OUTPUT_DESC by the capture owner. */
  struct capture_target_t {
    rect_t screen_rect {};
    std::uint32_t width {};
    std::uint32_t height {};
    std::uintptr_t monitor {};
    bool identity_orientation {true};
  };

  enum class route_e : std::uint8_t {
    none,
    roi,
    full_capture,
  };

  enum class mapping_status_e : std::uint8_t {
    ok,
    invalid_observation,
    invalid_capture_target,
    unsupported_orientation,
    monitor_mismatch,
    outside_capture,
    partially_outside_capture,
  };

  [[nodiscard]] const char *mapping_status_name(mapping_status_e status) noexcept;

  struct mapping_result_t {
    mapping_status_e status {mapping_status_e::invalid_observation};
    route_e route {route_e::none};
    rect_t capture_pixels {};

    [[nodiscard]] explicit operator bool() const noexcept {
      return status == mapping_status_e::ok && route != route_e::none;
    }
  };

  /**
   * Map an available client rectangle only when it is wholly contained by the exact selected
   * output in raw virtual-screen coordinates. Spanning/partial windows are never clipped.
   */
  [[nodiscard]] mapping_result_t map_to_capture(
    const snapshot_t &snapshot,
    const capture_target_t &target
  ) noexcept;

  namespace detail {
    /** One direct-child query normalized for the pure structural viewport policy. */
    struct child_window_observation_t {
      std::uintptr_t window {};
      std::uintptr_t parent {};
      std::uint32_t process_id {};
      rect_t screen_rect {};
      bool query_succeeded {};
      bool visible {};
    };

    /** Root fallback or a structurally proven outer content container. */
    struct content_selection_t {
      std::uintptr_t window {};
      rect_t screen_rect {};
      bool child_selected {};

      [[nodiscard]] explicit constexpr operator bool() const noexcept {
        return window != 0 && screen_rect.valid();
      }
    };

    /**
     * Select a dominant, center-covering direct child only when separate edge-aligned sibling
     * chrome corroborates the split. Ambiguity, incomplete enumeration, or weak evidence returns
     * the complete root client. No executable, window-class, control ID, or pixel inset is known.
     */
    [[nodiscard]] content_selection_t select_content_source(
      std::uintptr_t root_window,
      std::uint32_t root_process_id,
      rect_t root_client_screen_rect,
      std::span<const child_window_observation_t> children,
      bool enumeration_complete = true
    ) noexcept;

    /**
     * Injectable result of native queries. Production `sample()` fills this record, while unit
     * tests exercise policy without creating or manipulating desktop windows.
     */
    struct raw_observation_t {
      std::uintptr_t window {};
      std::uintptr_t window_after {};
      std::uintptr_t shell_window {};
      std::uintptr_t desktop_window {};
      std::uint32_t process_id {};
      std::uint32_t process_id_after {};
      std::uint32_t own_process_id {};
      std::uintptr_t monitor {};
      rect_t monitor_screen_rect {};
      rect_t client_screen_rect {};
      std::uintptr_t content_window {};
      rect_t content_screen_rect {};
      rect_t frame_screen_rect {};
      std::uint64_t extended_style {};
      std::uint32_t layered_flags {};
      std::uint8_t layered_alpha {};
      bool layered_attributes_succeeded {};
      bool dpi_aware {};
      bool is_window {};
      bool is_window_after {};
      bool visible {};
      bool minimized {};
      bool cloak_query_succeeded {};
      bool cloaked {};
      bool process_query_succeeded {};
      bool process_query_after_succeeded {};
      bool client_rect_succeeded {};
      bool frame_rect_succeeded {};
      bool style_query_succeeded {};
      bool class_query_succeeded {};
      bool shell_class {};
      bool gui_thread_query_succeeded {};
      bool gui_in_move_size {};
      std::uintptr_t move_size_root {};
    };

    /** Apply the foreground-window admission policy to injected native-query results. */
    [[nodiscard]] observation_t classify(
      const raw_observation_t &raw,
      std::chrono::steady_clock::time_point observed_at
    ) noexcept;
  }  // namespace detail
}  // namespace platf::foreground_window
