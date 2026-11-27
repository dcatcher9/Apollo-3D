/**
 * @file src/platform/windows/primary_display_win32.h
 * @brief The production adapter of the primary-display transactions: CCD queries and applies
 * through display_config, Advanced Color, the cursor and its clip, AR-glasses preservation and
 * the journal file.
 */
#pragma once

#include "primary_display_manager.h"

namespace platf::primary_display::detail {

  /** The adapter that observes and changes this machine's displays. */
  io_t system_io();

  /** Reapply the exclusive-session cursor clip this process owns after Windows reset it.
   * True when it owns none. The caller serializes this with every transaction.
   */
  bool refresh_owned_cursor_clip();

}  // namespace platf::primary_display::detail
