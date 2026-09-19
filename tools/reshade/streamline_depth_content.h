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
  struct association {
    status state{status::unavailable};
    bool coverage_complete{}, final_color_registered{};
    bool matched() const { return state == status::observed_content_match; }
  };
  inline const char *name(status value) {
    switch (value) {
      case status::unavailable: return "missing-content-observation";
      case status::valid_observation: return "valid-content-observation";
      case status::unknown_resource: return "unknown-resource-lifetime";
      case status::lifetime_mismatch: return "resource-lifetime-mismatch";
      case status::unsupported_resource: return "unsupported-resource-layout";
      case status::different_device: return "resource-device-mismatch";
      case status::unknown_recording: return "missing-content-recording";
      case status::superseded_marker: return "content-changed-after-entry-marker";
      case status::invalidated_recording: return "unsupported-recording-mutation";
      case status::invalidated_resource: return "unsupported-resource-mutation";
      case status::lost_observation: return "lost-content-observation";
      case status::different_recording: return "cross-recording-content-unverified";
      case status::different_source_content: return "source-content-version-mismatch";
      case status::backup_binding_mismatch: return "selected-backup-binding-mismatch";
      case status::backup_overwritten: return "backup-copy-overwritten";
      case status::observed_content_match: return "observed-content-match-coverage-incomplete";
    }
    return "unknown";
  }

  inline bool same_recording(const commands::recording_marker &a, const commands::recording_marker &b) {
    return a && b && a.command == b.command && a.object_generation == b.object_generation &&
      a.recording_generation == b.recording_generation && a.epoch == b.epoch && a.loss == b.loss;
  }

  // Serialized, bounded CPU model. Versions describe observed mutations within
  // one recording; version zero is symbolic incoming content, not a known depth
  // value. No native API/GPU success or complete mutation coverage is inferred.
  class ledger {
    struct resource {
      struct writer { commands::recording_marker recording; std::uint64_t mutation{}; };
      resource_key key;
      std::uint64_t device{}, registration{}, invalidation{}, mutation{}, last_copy{};
      // Two latest distinct writers suffice to detect a foreign recording after
      // any snapshot, even when this recording writes again afterward.
      std::array<writer, 2> writers{};
      bool supported{};
    };
    struct local_state {
      commands::recording_marker recording;
      std::uint64_t registration{}, generation{}, version{}, last_write{};
      bool observed{}, poisoned{};
    };
    std::array<resource, 256> resources_{};
    std::array<local_state, 1024> states_{};
    std::array<commands::recording_marker, 256> invalid_recordings_{};
    std::uint64_t epoch_{1}, serial_{}, ambiguity_revision_{};
    bool saturated_{};

    resource *find(std::uint64_t native) {
      for (auto &value : resources_) if (native && value.key.native == native) return &value;
      return nullptr;
    }
    const resource *find(std::uint64_t native) const { return const_cast<ledger *>(this)->find(native); }
    local_state *find_state(const commands::recording_marker &marker, std::uint64_t registration) {
      for (auto &value : states_) if (value.registration == registration && same_recording(value.recording, marker)) return &value;
      return nullptr;
    }
    const local_state *find_state(const commands::recording_marker &marker, std::uint64_t registration) const {
      return const_cast<ledger *>(this)->find_state(marker, registration);
    }
    bool banned(const commands::recording_marker &marker) const {
      if (saturated_) return true;
      for (const auto &value : invalid_recordings_) if (same_recording(value, marker)) return true;
      return false;
    }
    local_state &state(const commands::recording_marker &marker, const resource &item) {
      if (auto *value = find_state(marker, item.registration)) return *value;
      auto *chosen = &states_[0];
      for (auto &value : states_) {
        if (!value.generation) { chosen = &value; break; }
        if (value.generation < chosen->generation) chosen = &value;
      }
      *chosen = {};
      chosen->recording = marker;
      chosen->registration = item.registration;
      chosen->generation = ++serial_;
      chosen->poisoned = banned(marker);
      return *chosen;
    }
    status check_resource(resource_key key, const resource *value) const {
      if (!key || !value) return status::unknown_resource;
      if (!same(key, value->key)) return status::lifetime_mismatch;
      if (!value->supported) return status::unsupported_resource;
      return status::valid_observation;
    }
    void mutate(const commands::recording_marker &marker, resource &item) {
      auto &value = state(marker, item);
      // Draws between observations coalesce. Every use/copy turns the next write
      // into a new version; the latest event still detects a delayed entry read.
      if (!value.last_write || value.observed) value.version = ++serial_;
      value.observed = false;
      value.last_write = marker.event;
      item.mutation = ++serial_;
      item.last_copy = 0;
      auto *writer = &item.writers[0];
      for (auto &candidate : item.writers) {
        if (same_recording(candidate.recording, marker)) { writer = &candidate; break; }
        if (candidate.mutation < writer->mutation) writer = &candidate;
      }
      *writer = {marker, item.mutation};
    }
    status validate_snapshot(const use_snapshot &value) const {
      if (!value.valid()) return value.state;
      if (value.ledger_epoch != epoch_) return status::lost_observation;
      if (value.ambiguity_revision != ambiguity_revision_) return status::invalidated_resource;
      const auto *item = find(value.resource.native);
      const auto resource_status = check_resource(value.resource, item);
      if (resource_status != status::valid_observation) return resource_status;
      if (value.registration != item->registration) return status::lifetime_mismatch;
      if (value.resource_invalidation != item->invalidation) return status::invalidated_resource;
      for (const auto &writer : item->writers)
        if (writer.mutation > value.resource_mutation && !same_recording(writer.recording, value.recording))
          return status::different_source_content;
      if (banned(value.recording)) return status::invalidated_recording;
      const auto *local = find_state(value.recording, value.registration);
      if (!local || local->generation != value.state_generation) return status::lost_observation;
      if (local->poisoned) return status::invalidated_recording;
      return status::valid_observation;
    }

  public:
    void restart_observations() {
      // Lifetimes are still observed while passive content capture is disabled.
      // Their identities survive, but no content/copy proof may span that gap.
      states_ = {}; invalid_recordings_ = {};
      ++epoch_; saturated_ = false;
      for (auto &item : resources_) { item.writers = {}; item.last_copy = 0; }
    }
    void clear() {
      resources_ = {}; states_ = {}; invalid_recordings_ = {};
      ++epoch_; saturated_ = false;
    }
    void resource_initialized(resource_key key, std::uint64_t device, bool supported) {
      if (!key || !device) return;
      auto *chosen = find(key.native);
      if (chosen && same(chosen->key, key) && chosen->device == device && chosen->supported == supported) return;
      if (!chosen) {
        chosen = &resources_[0];
        for (auto &value : resources_) {
          if (!value.key) { chosen = &value; break; }
          if (value.registration < chosen->registration) chosen = &value;
        }
      }
      *chosen = {};
      chosen->key = key; chosen->device = device; chosen->supported = supported;
      chosen->registration = ++serial_;
    }
    void resource_destroyed(resource_key key) {
      if (auto *value = find(key.native); value && same(value->key, key)) *value = {};
    }
    void command_reset(std::uint64_t native) {
      // Old poisoned local states stay poisoned. Only the remembered ban can be
      // retired: wrappers never create observations for closed/reset recordings.
      for (auto &value : invalid_recordings_) if (value.command == native) value = {};
    }
    void command_destroyed(std::uint64_t native) {
      // Destroyed objects cannot create future uses of this recording. Retire
      // their bans without clearing poisoned historical states or resurrecting
      // snapshots. Otherwise create/destroy churn would exhaust the ban table.
      command_reset(native);
    }
    void invalidate(const commands::recording_marker &marker, resource_key key = {}) {
      if (!marker) { clear(); return; }
      if (key.native) {
        if (auto *item = find(key.native)) {
          ++item->invalidation;
          item->last_copy = 0;
          state(marker, *item).poisoned = true;
        }
        return;
      }
      // With no affected resource identity, another recording may have aliased
      // either the source or backup. Revoke all earlier snapshots, not merely
      // local states in the recording that observed the unknown operation.
      ++ambiguity_revision_;
      for (auto &value : states_) if (same_recording(value.recording, marker)) value.poisoned = true;
      if (banned(marker)) return;
      for (auto &value : invalid_recordings_) if (!value) { value = marker; return; }
      // Never evict a ban and let a still-open recording become trusted again.
      // Capacity exhaustion disables content evidence until a full clear.
      saturated_ = true;
    }
    void write(const commands::recording_marker &marker, resource_key key) {
      auto *item = find(key.native);
      if (!item) return; // Mutation-only zero lifetime lookup ignores untracked resources.
      if (!marker) { ++item->invalidation; item->last_copy = 0; return; }
      if (key.lifetime && !same(key, item->key)) { invalidate(marker, item->key); return; }
      mutate(marker, *item);
    }
    use_snapshot use(const commands::recording_marker &marker, std::uint64_t native, std::uint64_t device) {
      use_snapshot out;
      out.recording = marker; out.ledger_epoch = epoch_;
      if (!marker) { out.state = status::unknown_recording; return out; }
      auto *item = find(native);
      if (!item) { out.state = status::unknown_resource; return out; }
      out.resource = item->key;
      out.state = check_resource(out.resource, item);
      if (!out.valid()) return out;
      if (!device || item->device != device) { out.state = status::different_device; return out; }
      auto &local = state(marker, *item);
      if (banned(marker) || local.poisoned) { out.state = status::invalidated_recording; return out; }
      if (local.last_write > marker.event) { out.state = status::superseded_marker; return out; }
      local.observed = true;
      out.version = local.version; out.resource_invalidation = item->invalidation;
      out.registration = item->registration; out.state_generation = local.generation;
      out.resource_mutation = item->mutation;
      out.ambiguity_revision = ambiguity_revision_;
      return out;
    }
    copy_snapshot copy(const commands::recording_marker &marker, resource_key source, resource_key backup, std::uint64_t device) {
      copy_snapshot out;
      out.recording = marker; out.backup = backup; out.ledger_epoch = epoch_;
      auto *src = find(source.native), *dst = find(backup.native);
      // The caller still issues the real copy if metadata is rejected. Every
      // attempted destination mutation must revoke an older copied binding.
      write(marker, backup);
      out.state = check_resource(source, src);
      if (!out.valid()) return out;
      out.state = check_resource(backup, dst);
      if (!out.valid()) return out;
      if (same(source, backup)) { out.state = status::backup_binding_mismatch; return out; }
      if (!device || src->device != device || dst->device != device) { out.state = status::different_device; return out; }
      out.source = use(marker, source.native, device); out.state = out.source.state;
      if (!out.valid()) return out;
      if (state(marker, *dst).poisoned) { out.state = status::invalidated_recording; return out; }
      out.copy_id = dst->last_copy = ++serial_;
      out.backup_mutation = dst->mutation; out.backup_invalidation = dst->invalidation;
      out.backup_registration = dst->registration;
      return out;
    }
    // Forward an already preserved image without pretending the original game
    // source was copied again. The second destination mutation is recorded even
    // when lineage is unavailable. Our content model proves same-recording order
    // only; another recording cannot inherit this association from CPU timing.
    copy_snapshot forward(const commands::recording_marker &marker, const copy_snapshot &preserved,
        resource_key destination, std::uint64_t device) {
      const auto prior = associate(preserved.source, preserved, preserved.source.resource, preserved.backup);
      auto out = copy(marker, preserved.backup, destination, device);
      if (!out.valid()) return out;
      if (!prior.matched()) { out.state = prior.state; return out; }
      if (!same_recording(marker, preserved.recording)) { out.state = status::different_recording; return out; }
      if (marker.event <= preserved.recording.event) { out.state = status::superseded_marker; return out; }
      out.source = preserved.source;
      return out;
    }
    status unchanged(const use_snapshot &value) const {
      const auto checked = validate_snapshot(value);
      if (checked != status::valid_observation) return checked;
      const auto *local = find_state(value.recording, value.registration);
      return local->version == value.version ? status::valid_observation : status::different_source_content;
    }
    association associate(const use_snapshot &evaluation, const copy_snapshot &copy,
        resource_key selected_source, resource_key selected_backup) const {
      association out;
      out.state = validate_snapshot(evaluation);
      if (out.state != status::valid_observation) return out;
      if (!copy.valid()) { out.state = copy.state; return out; }
      if (copy.ledger_epoch != epoch_) { out.state = status::lost_observation; return out; }
      out.state = validate_snapshot(copy.source);
      if (out.state != status::valid_observation) return out;
      if (!same_recording(evaluation.recording, copy.recording)) { out.state = status::different_recording; return out; }
      if (!same(selected_source, evaluation.resource) || !same(evaluation.resource, copy.source.resource) ||
          evaluation.registration != copy.source.registration || evaluation.state_generation != copy.source.state_generation ||
          evaluation.version != copy.source.version || evaluation.resource_mutation != copy.source.resource_mutation) {
        out.state = status::different_source_content; return out;
      }
      if (!selected_backup || !same(selected_backup, copy.backup)) { out.state = status::backup_binding_mismatch; return out; }
      const auto *backup = find(copy.backup.native);
      out.state = check_resource(copy.backup, backup);
      if (out.state != status::valid_observation) return out;
      if (backup->registration != copy.backup_registration) { out.state = status::lifetime_mismatch; return out; }
      if (backup->invalidation != copy.backup_invalidation) { out.state = status::invalidated_resource; return out; }
      if (backup->mutation != copy.backup_mutation || backup->last_copy != copy.copy_id) {
        out.state = status::backup_overwritten; return out;
      }
      out.state = status::observed_content_match;
      return out;
    }
  };
}
