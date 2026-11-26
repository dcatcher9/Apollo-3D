/**
 * @file src/platform/windows/primary_display_journal.h
 * @brief The primary-display crash journal: its JSON codec and its file beside the configuration.
 */
#pragma once

#include "primary_display.h"

#include <cstddef>
#include <filesystem>

namespace platf::primary_display::detail {

  /** The most displays one recorded layout or topology may hold. */
  inline constexpr size_t max_displays = 128;

  /** Whether a journal is consistent enough to save or to recover from. The codec refuses any
   * other; the rules belong to the transactions in primary_display.cpp.
   */
  bool valid_journal(const journal_t &journal);

  std::filesystem::path journal_path();
  load_result_t load_journal();
  bool save_journal(const journal_t &journal);
  bool clear_journal();

}  // namespace platf::primary_display::detail
