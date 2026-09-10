/**
 * @file src/platform/windows/primary_display.h
 * @brief Temporary, recoverable primary-display changes for remote and local AR sessions.
 */
#pragma once

#include "display_config.h"
#include "exclusive_cursor_clip.h"

#include <cstdint>
#include <filesystem>
#include <functional>
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

  /** Serialize local source/sink mode changes with display-notification reconciliation.
   * The callback must not call this namespace. Promotion/verification runs after the callback.
   */
  bool run_local_exclusive_update(std::wstring_view source_device_path, const std::function<bool()> &update);

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
   * If an original monitor is missing, available physical outputs are restored first.
   * Local AR preserves the selected sink's current mode/color; its absence alone does not
   * block completion once all ordinary original outputs have been restored and verified.
   * A supplied identity prevents retiring an old display from restoring a newer session.
   * False leaves the journal intact and callers must defer removal of the virtual display.
   */
  bool restore(std::wstring_view expected_device_path = {});

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

    /** A process-lifetime file handle prevents a second host from recovering a live journal. */
    class ownership_t {
    public:
      ownership_t() = default;
      ~ownership_t();
      ownership_t(const ownership_t &) = delete;
      ownership_t &operator=(const ownership_t &) = delete;
      bool acquire(const std::filesystem::path &path);

    private:
      HANDLE handle_ = INVALID_HANDLE_VALUE;
      std::filesystem::path path_;
    };

    // The transaction adapter is deliberately injectable: tests never change real displays.
    struct position_t {
      std::wstring device_path;
      LONG x = 0;
      LONG y = 0;
    };

    using layout_t = std::vector<position_t>;

    struct snapshot_t {
      std::vector<DISPLAYCONFIG_PATH_INFO> paths;
      std::vector<DISPLAYCONFIG_MODE_INFO> modes;
      std::vector<std::wstring> device_paths;  // In the same order as paths.
      std::vector<std::optional<display_config::advanced_color_state_t>> colors;
    };

    struct journal_t {
      std::wstring original_primary;
      std::wstring promoted_primary;
      layout_t original;
      layout_t promoted;
      bool prepared = false;
      bool exclusive = false;
      bool exclusive_started = false;
      std::optional<snapshot_t> original_topology;
      std::optional<snapshot_t> before_exclusive;
      std::optional<snapshot_t> pending_restore;
      std::vector<std::wstring> exclusive_preserved;
      std::optional<snapshot_t> exclusive_topology;
      std::wstring local_sink;
    };

    struct load_result_t {
      bool success = false;
      std::optional<journal_t> journal;
    };

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
    };

    std::optional<layout_t> inspect(const snapshot_t &snapshot);
    std::string serialize(const journal_t &journal);
    std::optional<journal_t> deserialize(std::string_view contents);

    class manager_t {
    public:
      explicit manager_t(io_t io);
      bool prepare(bool exclusive = false, std::wstring_view local_sink = {});
      bool is_local_exclusive(std::wstring_view source_device_path);
      bool bind_pending(std::wstring_view device_path);
      bool promote(std::wstring_view device_path, bool exclusive = false);
      bool restore(std::wstring_view expected_device_path = {});
      bool reconcile_active_exclusive(std::wstring_view device_path);
      bool recover_inactive_exclusive();

    private:
      io_t io_;
      bool apply_verified(const snapshot_t &before, const layout_t &desired);
      bool promote_exclusive(std::wstring_view device_path);
      bool refresh_exclusive_cursor_clip(std::wstring_view device_path);
      bool clip_cursor_to_display(const snapshot_t &snapshot, std::wstring_view device_path);
      bool restore_exclusive(journal_t journal);
      bool recover_prepared_outputs(journal_t journal);
    };
  }  // namespace detail
}  // namespace platf::primary_display
