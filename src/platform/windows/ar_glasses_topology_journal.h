/**
 * @file src/platform/windows/ar_glasses_topology_journal.h
 * @brief The extended-mode topology journal: durable evidence for every physical-output move made
 * for local AR, the journaled source/sink isolation, and recovery of those moves.
 */
#pragma once

#include "ar_glasses_devices.h"

#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <windows.h>

namespace ar_glasses {
  namespace internal {
    bool begin_topology_recovery_move(
      std::wstring_view device_path,
      const RECT &original_rect,
      const RECT &requested_rect
    );
    bool cancel_topology_recovery_move(std::wstring_view device_path);
    std::optional<RECT> prepare_topology_recovery_for_mode_change(
      std::wstring_view device_path,
      const RECT &previous_original_rect,
      const RECT &current_rect
    );

    std::optional<target_state_t> isolate_physical_output(
      const std::wstring &virtual_gdi_name,
      const std::wstring &target_device_path,
      const RECT &original_rect
    );

    struct pre_removal_topology_decision_t {
      RECT desired_rect {};
      bool apollo_owned = false;
      bool recovery_record_present = false;
    };

    std::optional<pre_removal_topology_decision_t> classify_pre_removal_topology(
      std::wstring_view device_path
    );

    class pre_removal_topology_cleanup_t {
    public:
      pre_removal_topology_cleanup_t(
        std::optional<pre_removal_topology_decision_t> decision,
        std::wstring device_path
      );
      bool prepare_removal();
      bool finish();

    private:
      std::mutex mutex_;
      std::optional<pre_removal_topology_decision_t> decision_;
      bool decision_resolved_ = false;
      bool position_restored_ = false;
      bool recovery_evidence_cleared_ = false;
      std::wstring device_path_;
    };

    bool recover_saved_topology(std::wstring_view device_path);
    std::size_t recover_connected_saved_topologies();
    bool recover_connected_saved_topologies_for_remote_handoff();
  }  // namespace internal
}  // namespace ar_glasses
