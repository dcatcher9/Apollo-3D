/**
 * @file src/platform/windows/primary_display_manager.h
 * @brief The primary-display transaction policy: the journal transitions that prepare, bind,
 * promote, restore, pause, reactivate and reconcile through an injectable display adapter, and
 * the process-local session state around them.
 */
#pragma once

#include "exclusive_cursor_clip.h"
#include "primary_display.h"
#include "primary_display_journal.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <windows.h>

namespace platf::primary_display {
  /** What pause captures of a retained virtual display, for reactivate and its rollback. */
  struct retained_display_t {
    detail::display_spec_t display;
    std::wstring local_sink;
    bool monitor_replacement = false;
    std::optional<POINT> cursor_offset;
    std::string_view last_resume_failure {};
    // Keep the last verified remote physical desktop after the durable transaction completes.
    // A monitor may briefly disappear while waking, leaving Windows on the retained VD again.
    std::optional<detail::journal_t> pause_recovery;
    bool partial_wake_recovery = false;
  };

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
