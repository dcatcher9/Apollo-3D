/**
 * @file src/platform/windows/primary_display.h
 * @brief Temporary, recoverable primary-display changes for remote sessions.
 */
#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <windows.h>

namespace platf::primary_display {

  /** Save the physical baseline before AddVirtualDisplay can restore remembered topology. */
  bool prepare();

  /** Bind a prepared baseline to the exact published AddVirtualDisplay identity.
   * Call before mode changes. Idempotent for an already-bound identity or no pending journal.
   */
  bool bind_pending(std::wstring_view device_path);

  /** Promote an active, extended display. Persist recovery before changing CCD. */
  bool promote(std::wstring_view device_path);

  /** Restore only our recorded positions, preserving current display modes.
   * A supplied identity prevents retiring an old display from restoring a newer session.
   * False leaves the journal intact and callers must defer removal of the virtual display.
   */
  bool restore(std::wstring_view expected_device_path = {});

  /** Recover an interrupted transaction on startup, before creating another virtual display. */
  bool recover();

  namespace detail {
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
    };

    struct journal_t {
      std::wstring original_primary;
      std::wstring promoted_primary;
      layout_t original;
      layout_t promoted;
      bool prepared = false;
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
    };

    std::optional<layout_t> inspect(const snapshot_t &snapshot);
    std::string serialize(const journal_t &journal);
    std::optional<journal_t> deserialize(std::string_view contents);

    class manager_t {
    public:
      explicit manager_t(io_t io);
      bool prepare();
      bool bind_pending(std::wstring_view device_path);
      bool promote(std::wstring_view device_path);
      bool restore(std::wstring_view expected_device_path = {});

    private:
      io_t io_;
      bool apply_verified(const snapshot_t &before, const layout_t &desired);
    };
  }  // namespace detail
}  // namespace platf::primary_display
