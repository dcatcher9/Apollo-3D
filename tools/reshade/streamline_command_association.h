// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cstdint>

namespace sunshine_streamline::commands {
  // Identity and ordering only. ReShade lifecycle/execute callbacks are pre-call
  // observations, not proof that native APIs succeeded or that the GPU completed.
  struct recording_marker {
    std::uint64_t command{}, object_generation{}, recording_generation{}, event{}, epoch{}, loss{};
    explicit operator bool() const { return command && object_generation && recording_generation && event && epoch; }
  };
  enum class status {
    unavailable, busy, lost_observation, unknown_command, recording_evicted, command_recreated, recording_changed,
    open_recording, nested_execution, unknown_queue, queue_recreated, different_device,
    different_queue, repeated_submission, missing_close, missing_submission,
    ambiguous_marker, same_queue_callbacks_order_unverified, queue_ordered_observations
  };
  enum class order { unknown, copy_before_evaluation, evaluation_before_copy };
  struct association {
    status state{status::unavailable};
    order ordering{};
    // Submission fields are pre-call callback serials, not native/GPU order.
    std::uint64_t queue{}, queue_generation{}, copy_submission{}, evaluation_submission{};
    bool same_recording{}, content_registered{}, final_color_registered{};
    bool associated() const { return state == status::queue_ordered_observations; }
  };
  inline const char *name(status value) {
    switch (value) {
      case status::unavailable: return "missing-recording-marker";
      case status::busy: return "command-tracker-busy";
      case status::lost_observation: return "lost-command-observation";
      case status::unknown_command: return "unknown-command";
      case status::recording_evicted: return "recording-history-evicted";
      case status::command_recreated: return "command-lifetime-mismatch";
      case status::recording_changed: return "recording-changed-during-evaluation";
      case status::open_recording: return "recording-not-closed";
      case status::nested_execution: return "unsupported-secondary-execution";
      case status::unknown_queue: return "unknown-queue";
      case status::queue_recreated: return "queue-lifetime-mismatch";
      case status::different_device: return "different-device";
      case status::different_queue: return "different-queue";
      case status::repeated_submission: return "repeated-recording-submission";
      case status::missing_close: return "missing-close-observation";
      case status::missing_submission: return "missing-submit-observation";
      case status::ambiguous_marker: return "ambiguous-recording-markers";
      case status::same_queue_callbacks_order_unverified: return "same-queue-callbacks-order-unverified";
      case status::queue_ordered_observations: return "queue-ordered-api-observations-content-unverified";
    }
    return "unknown";
  }

  // Pure bounded model. The owner serializes all access; no allocations, native
  // pointers are dereferenced, or API callbacks are made by this class.
  class tracker {
    struct object { std::uint64_t native{}, device{}, generation{}, recording{}, cookie{}; };
    struct queue { std::uint64_t native{}, device{}, generation{}, cookie{}; };
    struct recording {
      std::uint64_t generation{}, command{}, object_generation{}, device{};
      std::uint64_t queue{}, queue_generation{}, submission{};
      unsigned submissions{};
      bool closed{}, nested{};
      status failure{status::queue_ordered_observations};
    };
    std::array<object, 128> objects_{};
    std::array<queue, 16> queues_{};
    std::array<recording, 256> recordings_{};
    std::uint64_t sequence_{}, epoch_{1};
    object *find_object(std::uint64_t native) {
      for (auto &value : objects_) if (value.native == native && native) return &value;
      return nullptr;
    }
    const object *find_object(std::uint64_t native) const { return const_cast<tracker *>(this)->find_object(native); }
    queue *find_queue(std::uint64_t native) {
      for (auto &value : queues_) if (value.native == native && native) return &value;
      return nullptr;
    }
    const queue *find_queue(std::uint64_t native) const { return const_cast<tracker *>(this)->find_queue(native); }
    recording *find_recording(std::uint64_t generation) {
      for (auto &value : recordings_) if (value.generation == generation && generation) return &value;
      return nullptr;
    }
    const recording *find_recording(std::uint64_t generation) const { return const_cast<tracker *>(this)->find_recording(generation); }
    void begin(object &value) {
      auto *chosen = &recordings_[0];
      for (auto &candidate : recordings_) {
        if (!candidate.generation) { chosen = &candidate; break; }
        if (candidate.generation < chosen->generation) chosen = &candidate;
      }
      *chosen = {};
      chosen->generation = value.recording = ++sequence_;
      chosen->command = value.native;
      chosen->object_generation = value.generation;
      chosen->device = value.device;
    }
    status check(const recording_marker &marker, const recording *&out, std::uint64_t loss) const {
      if (!marker) return status::unavailable;
      if (marker.epoch != epoch_ || marker.loss != loss) return status::lost_observation;
      const auto *owner = find_object(marker.command);
      if (!owner) return status::unknown_command;
      if (owner->generation != marker.object_generation) return status::command_recreated;
      out = find_recording(marker.recording_generation);
      if (!out) return status::recording_evicted;
      if (out->command != marker.command || out->object_generation != marker.object_generation) return status::command_recreated;
      if (out->nested) return status::nested_execution;
      if (out->submissions > 1) return status::repeated_submission;
      if (out->failure != status::queue_ordered_observations) return out->failure;
      if (!out->closed) return status::open_recording;
      if (!out->submissions) return status::missing_submission;
      const auto *submitted_queue = find_queue(out->queue);
      if (!submitted_queue) return status::unknown_queue;
      if (submitted_queue->generation != out->queue_generation) return status::queue_recreated;
      return status::queue_ordered_observations;
    }
  public:
    void clear() {
      objects_ = {}; queues_ = {}; recordings_ = {};
      ++epoch_;
    }
    // A nonzero cookie must come from per-ReShade-object private data, never be
    // computed from the native address. Repeated ensures are then idempotent.
    void command_initialized(std::uint64_t native, std::uint64_t device, std::uint64_t cookie = 0) {
      if (!native || !device) return;
      auto *chosen = find_object(native);
      if (chosen && cookie && chosen->cookie == cookie && chosen->device == device) return;
      if (!chosen) {
        chosen = &objects_[0];
        for (auto &candidate : objects_) {
          if (!candidate.native) { chosen = &candidate; break; }
          if (candidate.generation < chosen->generation) chosen = &candidate;
        }
      }
      *chosen = {native, device, ++sequence_, 0, cookie};
      begin(*chosen);
    }
    void command_destroyed(std::uint64_t native) { if (auto *value = find_object(native)) *value = {}; }
    void command_reset(std::uint64_t native, std::uint64_t device = 0, std::uint64_t cookie = 0) {
      auto *value = find_object(native);
      // A caller-owned private-data lifetime cookie re-establishes identity after
      // bounded eviction/loss. Native pointer equality alone cannot do that.
      if (device && cookie && (!value || value->device != device || value->cookie != cookie)) {
        command_initialized(native, device, cookie);
        return;
      }
      if (value && (!device || device == value->device) && (!cookie || cookie == value->cookie)) begin(*value);
    }
    void command_closed(std::uint64_t native) {
      if (auto *value = find_object(native)) if (auto *record = find_recording(value->recording)) record->closed = true;
    }
    void queue_initialized(std::uint64_t native, std::uint64_t device, std::uint64_t cookie = 0) {
      if (!native || !device) return;
      auto *chosen = find_queue(native);
      if (chosen && cookie && chosen->cookie == cookie && chosen->device == device) return;
      if (!chosen) {
        chosen = &queues_[0];
        for (auto &candidate : queues_) {
          if (!candidate.native) { chosen = &candidate; break; }
          if (candidate.generation < chosen->generation) chosen = &candidate;
        }
      }
      *chosen = {native, device, ++sequence_, cookie};
    }
    void queue_destroyed(std::uint64_t native) { if (auto *value = find_queue(native)) *value = {}; }
    recording_marker mark(std::uint64_t native, std::uint64_t loss) {
      const auto *value = find_object(native);
      const auto *record = value ? find_recording(value->recording) : nullptr;
      if (!record || record->closed || record->submissions || record->nested) return {};
      return {native, value->generation, record->generation, ++sequence_, epoch_, loss};
    }
    bool same_open_recording(const recording_marker &marker, std::uint64_t loss) const {
      if (!marker || marker.epoch != epoch_ || marker.loss != loss) return false;
      const auto *value = find_object(marker.command);
      const auto *record = value ? find_recording(value->recording) : nullptr;
      return record && value->generation == marker.object_generation && record->generation == marker.recording_generation &&
        !record->closed && !record->submissions && !record->nested;
    }
    std::uint64_t recording_device(const recording_marker &marker) const {
      const auto *record = find_recording(marker.recording_generation);
      return marker && marker.epoch == epoch_ && record && record->command == marker.command &&
        record->object_generation == marker.object_generation ? record->device : 0;
    }
    void command_executed(std::uint64_t native_queue, std::uint64_t native_command) {
      const auto *value = find_object(native_command);
      auto *record = value ? find_recording(value->recording) : nullptr;
      if (!record) return;
      ++record->submissions;
      if (record->submissions != 1) return;
      record->submission = ++sequence_;
      if (!record->closed) { record->failure = status::missing_close; return; }
      const auto *target = find_queue(native_queue);
      if (!target) { record->failure = status::unknown_queue; return; }
      if (target->device != record->device) { record->failure = status::different_device; return; }
      record->queue = target->native;
      record->queue_generation = target->generation;
    }
    void command_secondary_executed(std::uint64_t primary, std::uint64_t secondary) {
      for (auto native : {primary, secondary})
        if (const auto *value = find_object(native)) if (auto *record = find_recording(value->recording)) record->nested = true;
    }
    association associate(const recording_marker &evaluation, const recording_marker &copy,
        std::uint64_t expected_queue, std::uint64_t loss) const {
      association out;
      const recording *e{}, *c{};
      out.state = check(evaluation, e, loss);
      if (!out.associated()) return out;
      out.state = check(copy, c, loss);
      if (!out.associated()) return out;
      const auto *target = find_queue(expected_queue);
      if (!target) { out.state = status::unknown_queue; return out; }
      if (e->device != c->device || e->device != target->device) { out.state = status::different_device; return out; }
      if (e->queue != c->queue || e->queue != expected_queue || e->queue_generation != c->queue_generation ||
          e->queue_generation != target->generation) { out.state = status::different_queue; return out; }
      out.queue = expected_queue;
      out.queue_generation = target->generation;
      out.copy_submission = c->submission;
      out.evaluation_submission = e->submission;
      out.same_recording = e->generation == c->generation;
      // ReShade invokes execute callbacks before the native call, and releases
      // its queue lock before entering ExecuteCommandLists. Concurrent callers
      // can therefore reverse callback order at the actual queue submission.
      // These serials identify observations only; no cross-recording order proof.
      if (!out.same_recording) {
        out.state = status::same_queue_callbacks_order_unverified;
        return out;
      }
      if (out.same_recording && copy.event == evaluation.event) { out.state = status::ambiguous_marker; return out; }
      out.ordering = copy.event < evaluation.event ?
        order::copy_before_evaluation : order::evaluation_before_copy;
      // Content-write identity, final-color registration and GPU completion are
      // deliberately absent; callers must not promote this result into them.
      return out;
    }
  };
}
