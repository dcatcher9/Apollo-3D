// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "../../src/game3d_debug_protocol.h"
#include <memory>
#include <string>

namespace sunshine_game3d {
  struct ui_capture_state;
  // Request-scoped optional pixels. Metadata observation, native GPU ownership,
  // and IPC publication remain separate; none of these resources selects depth.
  class ui_capture_batch {
  public:
    explicit ui_capture_batch(std::uint64_t diagnostic_session);
    ~ui_capture_batch();
    ui_capture_batch(const ui_capture_batch &) = delete;
    ui_capture_batch &operator=(const ui_capture_batch &) = delete;
    // Stop collecting new resources when the main color/depth snapshot is made.
    void freeze();
    // Nonblocking. Returns false while optional GPU work is pending, for at most
    // 250 ms after freeze. Failures/timeouts become explicit metadata, not a
    // failed primary dump. Shared texture leases survive until this batch dies.
    bool append(game3d_debug::response_t &, std::uint64_t &bytes, std::string &json, bool allow_pixels);
  private:
    bool append_transaction(game3d_debug::response_t &, std::uint64_t &, std::string &, bool);
    std::shared_ptr<ui_capture_state> state_;
  };
}
