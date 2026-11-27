/**
 * @file src/platform/windows/primary_display.h
 * @brief Temporary, recoverable primary-display changes for remote and local AR sessions.
 */
#pragma once

#include "display_config.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>

namespace platf::primary_display {

  /** Save the physical baseline before AddVirtualDisplay can restore remembered topology.
   * Exclusive mode also saves complete active CCD modes and Advanced Color state.
   */
  bool prepare(bool exclusive = false);

  /** Save the local-AR baseline and preserve only this approved presentation sink.
   * Its on-device mode/color remains user-controlled; ordinary outputs retain full recovery.
   */
  bool prepare_local_exclusive(std::wstring_view sink_device_path);

  /** Replace one physical desktop rectangle with the private source, retaining all other
   * displays and the selected monitor's primary/secondary role. The physical sink is parked
   * outside the interactive desktop for scanout; no global cursor clip is acquired.
   */
  bool prepare_local_monitor(std::wstring_view sink_device_path);

  /** Verify the active source and sink against their committed monitor replacement. */
  bool verify_local_monitor_replacement(std::wstring_view source_device_path, std::wstring_view sink_device_path);

  /** Serialize local source/sink mode changes with display-notification reconciliation.
   * The callback must not call this namespace. Promotion/verification runs after the callback.
   */
  bool run_local_exclusive_update(std::wstring_view source_device_path, const std::function<bool()> &update);

  /** Serialize a remote session's mode set with display-notification reconciliation, so the
   * reconcile worker cannot apply a topology between this change's query and its apply (or the
   * reverse). Unlike the local update above it neither promotes nor verifies: the caller settles
   * and rolls back itself. The callback must not call this namespace.
   */
  long run_serialized_mode_change(const std::function<long()> &change);

  /** Bind a prepared baseline to the exact published AddVirtualDisplay identity.
   * Call before mode changes. Idempotent for an already-bound identity or no pending journal.
   */
  bool bind_pending(std::wstring_view device_path);

  /** Make the exact active display primary, disabling ordinary outputs when exclusive is true.
   * Approved AR-glasses outputs remain active so Windows continues driving their scanout.
   * All mutations require durable recovery state and use temporary CCD configuration.
   */
  bool promote(std::wstring_view device_path, bool exclusive = false);

  /** Restore recorded positions in primary-only mode, preserving current display modes.
   * Exclusive mode reactivates original physical outputs with saved modes/color and keeps
   * an existing virtual output active at its current mode for retirement or reconnect.
   * Missing original monitors do not block completion once available outputs are verified.
   * Explicit user layout changes become the recovery baseline for active physical outputs.
   * Local AR preserves the selected sink's current mode/color.
   * A supplied identity prevents retiring an old display from restoring a newer session.
   * False leaves the journal intact and callers must defer removal of the virtual display.
   */
  bool restore(std::wstring_view expected_device_path = {});

  struct retained_display_t;
  using retained_display_ptr = std::shared_ptr<retained_display_t>;

  /** Restore a retained desktop and detach its virtual target without removing its device.
   * Capture the target's exact mode/color once; keep the pointer across retries and failures.
   * A sole virtual output stays active when no physical desktop can be restored.
   * Local exclusive sessions retain their recorded sink; extended local callers supply it.
   */
  bool pause(std::wstring_view device_path, retained_display_ptr &retained, std::wstring_view local_sink = {});

  /** Reactivate the retained target beside the current physical desktop, before GDI/HDR use.
   * Durable recovery precedes every mutation. The caller still promotes before capture and
   * resets retained only after success; failures remain recoverable through pause/restore.
   */
  bool reactivate(const retained_display_ptr &retained, bool exclusive);

  /** Best-effort cursor restoration after the exact retained display's final promotion.
   * The bookmark is captured once before pause and remains relative to that stable display.
   */
  void restore_retained_cursor(const retained_display_ptr &retained);

  /** True while the active desktop is still the user's own topology around this bound display:
   * the original primary is at the desktop origin and every active output is a recorded original
   * one. A recorded output that Windows switched off by itself does not change that. Without a
   * recovery record no host topology is applied; the display must then not be primary
   * unless it is the only active output. Unreadable or unexpected state returns false. Read-only.
   */
  bool baseline_topology_active(std::wstring_view device_path);

  /** Run the session's own change to its virtual display's color state (HDR on or off) as one
   * display transaction and record the state it produced as the exclusive session's state.
   * Promotion and display-change reconciliation restore the recorded state after a topology
   * change, so a change that is not recorded is undone by the next of them, and the session
   * and the reconciliation then keep toggling the display between the two states. The callback
   * applies and verifies the change and must not call this namespace. Returns its result.
   */
  bool change_virtual_display_color(std::wstring_view device_path, const std::function<bool()> &change);

  /** Recover an interrupted transaction on startup, before creating another virtual display. */
  bool recover();

  /** Reapply cursor isolation after Windows resets ClipCursor for a display or focus change.
   * No-op unless this process currently owns an exclusive-session cursor clip.
   */
  bool refresh_exclusive_cursor_clip();

  enum class exclusive_reconcile_result_e {
    settled,
    retry_active,
    pending_recovery,
  };

  struct exclusive_reconcile_result_t {
    exclusive_reconcile_result_e result = exclusive_reconcile_result_e::settled;
    std::uint64_t generation = 0;
  };

  /** Monotonic identity for the pending or active exclusive-display session. */
  std::uint64_t exclusive_session_generation();

  /** Reassert the owned exclusive topology after Windows reports a display change.
   * The generation overload makes delayed work harmless after restore or replacement.
   */
  exclusive_reconcile_result_t reconcile_exclusive_display_topology();
  exclusive_reconcile_result_t reconcile_exclusive_display_topology(std::uint64_t expected_generation);
}  // namespace platf::primary_display
