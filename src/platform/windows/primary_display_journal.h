/**
 * @file src/platform/windows/primary_display_journal.h
 * @brief The primary-display crash journal: its record, its JSON codec, its file beside the
 * configuration and the lock that keeps a second host from recovering it.
 */
#pragma once

#include "primary_display_topology.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <windows.h>

namespace platf::primary_display::detail {

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
    bool monitor_replacement = false;
  };

  struct load_result_t {
    bool success = false;
    std::optional<journal_t> journal;
  };

  /** Whether a journal is consistent enough to save or to recover from. The codec refuses any
   * other; the rules belong to the transactions in primary_display.cpp.
   */
  bool valid_journal(const journal_t &journal);

  /** The journal's JSON record. An invalid journal encodes as empty and decodes as nothing. */
  std::string serialize(const journal_t &journal);
  std::optional<journal_t> deserialize(std::string_view contents);

  std::filesystem::path journal_path();
  load_result_t load_journal();
  bool save_journal(const journal_t &journal);
  bool clear_journal();

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

}  // namespace platf::primary_display::detail
