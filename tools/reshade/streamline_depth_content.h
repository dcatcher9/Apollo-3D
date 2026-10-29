// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "streamline_command_association.h"
#include <array>
#include <cstdint>

namespace sunshine_streamline::content {
  struct resource_key {
    std::uint64_t native{}, lifetime{};
    explicit operator bool() const { return native && lifetime; }
  };
  inline bool same(resource_key a, resource_key b) { return a.native == b.native && a.lifetime == b.lifetime; }
  enum class status {
    unavailable, valid_observation, unknown_resource, lifetime_mismatch, unsupported_resource,
    different_device, unknown_recording, superseded_marker, invalidated_recording,
    invalidated_resource, lost_observation, different_recording, different_source_content,
    backup_binding_mismatch, backup_overwritten, observed_content_match
  };
  struct use_snapshot {
    status state{status::unavailable};
    resource_key resource;
    commands::recording_marker recording;
    std::uint64_t version{}, ledger_epoch{}, resource_invalidation{}, registration{}, state_generation{}, resource_mutation{}, ambiguity_revision{};
    bool valid() const { return state == status::valid_observation; }
  };
  struct copy_snapshot {
    status state{status::unavailable};
    use_snapshot source;
    resource_key backup;
    commands::recording_marker recording;
    std::uint64_t copy_id{}, backup_mutation{}, backup_invalidation{}, ledger_epoch{}, backup_registration{};
    bool valid() const { return state == status::valid_observation; }
  };
  inline bool same_recording(const commands::recording_marker &a, const commands::recording_marker &b) {
    return a && b && a.command == b.command && a.object_generation == b.object_generation &&
      a.recording_generation == b.recording_generation && a.epoch == b.epoch && a.loss == b.loss;
  }
}
