/**
 * @file src/platform/windows/primary_display.h
 * @brief Temporary, recoverable primary-display changes for remote and local AR sessions.
 */
#pragma once

#include "display_config.h"
#include "exclusive_cursor_clip.h"
#include "primary_display_journal.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <windows.h>

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

  namespace detail {
    enum class exclusive_reconcile_action_e {
      defer,
      recover,
      reconcile,
    };

    enum class exclusive_restore_action_e {
      keep,
      reject,
      disarm_before,
      disarm_after_success,
    };

    /** Process-local ownership state around the durable display journal.
     * A pending identity prevents recovery from treating a failed promotion attempt as an
     * abandoned session, while active is reserved for a fully verified promotion.
     * Access is serialized by the caller's display transaction mutex.
     */
    class exclusive_session_state_t {
    public:
      void prepared();
      bool can_use_identity(std::wstring_view device_path) const;
      void bound(std::wstring_view device_path);
      bool begin_promotion(std::wstring_view device_path);
      bool promotion_succeeded(std::wstring_view device_path);
      exclusive_restore_action_e restore_action(std::wstring_view expected_device_path) const;
      void restore_finished(exclusive_restore_action_e action, bool restored);
      void disarm();
      exclusive_reconcile_action_e reconcile_action() const;

      bool expected() const;
      const std::optional<std::wstring> &pending_identity() const;
      const std::optional<std::wstring> &active_identity() const;
      std::uint64_t generation() const;

    private:
      void changed();

      std::optional<std::wstring> pending_identity_;
      std::optional<std::wstring> active_identity_;
      bool expected_ = false;
      std::uint64_t generation_ = 0;
    };

    /** Process-local suppression of repeated mode sets which reached the same failed state.
     * The durable journal owns recovery intent; this cache only defers another identical
     * mutation until fresh active, available-target or desired observations change.
     */
    class restore_observation_cache_t {
    public:
      bool unchanged_failure(const snapshot_t &current, const snapshot_t &desired, const snapshot_t &available) const;
      void remember_failure(const snapshot_t &observed, const snapshot_t &desired, const snapshot_t &available);
      void clear();

    private:
      struct failure_t {
        snapshot_t observed;
        snapshot_t desired;
        snapshot_t available;
      };
      std::optional<failure_t> failure_;
    };

    // The transaction adapter is deliberately injectable: tests never change real displays.
    struct io_t {
      std::function<std::optional<snapshot_t>()> query;
      std::function<bool(snapshot_t)> apply;
      std::function<load_result_t()> load;
      std::function<bool(const journal_t &)> save;
      std::function<bool()> clear;
      std::function<std::optional<snapshot_t>()> query_all;
      std::function<bool(const DISPLAYCONFIG_PATH_INFO &, const display_config::advanced_color_state_t &)> set_color;
      std::function<bool(const DISPLAYCONFIG_PATH_INFO &, std::wstring_view)> preserve_exclusive;
      std::function<bool(std::wstring_view, std::optional<cursor_bounds_t>)> cursor_clip;
      std::function<std::optional<POINT>()> query_cursor;
      std::function<bool(POINT)> set_cursor;
      // Optional remote physical recovery path. Forces the supplied mode down to the driver;
      // callers without this adapter retain the ordinary apply behavior.
      std::function<bool(snapshot_t)> apply_physical_restore;
      // Waits between the looks at a color change that Windows has not reported yet. Without it
      // the looks follow each other at once, which only an in-memory adapter wants.
      std::function<void(std::chrono::milliseconds)> sleep;
    };

    class manager_t {
    public:
      explicit manager_t(io_t io, restore_observation_cache_t *restore_observations = nullptr);
      bool prepare(bool exclusive = false, std::wstring_view local_sink = {}, bool monitor_replacement = false);
      bool is_local_exclusive(std::wstring_view source_device_path);
      bool verify_local_monitor_replacement(std::wstring_view source_device_path, std::wstring_view sink_device_path);
      bool bind_pending(std::wstring_view device_path);
      bool promote(std::wstring_view device_path, bool exclusive = false);
      bool restore(std::wstring_view expected_device_path = {});
      bool pause(std::wstring_view device_path, retained_display_ptr &retained, std::wstring_view local_sink = {});
      bool reactivate(const retained_display_ptr &retained, bool exclusive);
      void restore_retained_cursor(const retained_display_ptr &retained);
      bool baseline_topology_active(std::wstring_view device_path);
      bool adopt_virtual_color(std::wstring_view device_path);
      bool reconcile_active_exclusive(std::wstring_view device_path);
      bool recover_inactive_exclusive();

    private:
      io_t io_;
      // Production managers share one owner under transaction_mutex. Independent transaction
      // adapters may omit it when testing their individual journal/API failure transitions.
      restore_observation_cache_t *restore_observations_;
      bool apply_verified(const snapshot_t &before, const layout_t &desired);
      bool promote_exclusive(std::wstring_view device_path, bool reconcile_active_source = false);
      bool promote_monitor(std::wstring_view device_path);
      bool restore_monitor(journal_t journal, bool keep_virtual_active, retained_display_ptr retained);
      bool apply_snapshot_verified(const snapshot_t &before, const snapshot_t &desired);
      bool refresh_exclusive_cursor_clip(std::wstring_view device_path);
      bool clip_cursor_to_display(const snapshot_t &snapshot, std::wstring_view device_path);
      bool restore_exclusive(journal_t journal, bool keep_virtual_active = true, retained_display_ptr retained = {});
      bool recover_prepared_outputs(journal_t journal);
    };
  }  // namespace detail
}  // namespace platf::primary_display
